"""
Re-entrainement du modele de debruitage L2 (U-Net) sous TensorFlow 2.10 / Keras 2.10.
====================================================================================
Le U-Net du livrable 2 avait ete sauvegarde avec Keras 3 (format .keras illisible par
TF 2.10). Ce script reproduit *exactement* son architecture (fonction `build_unet` de
notebooks/livrable2 copy.ipynb), ré-entraine un auto-encodeur debruiteur sur un
sous-ensemble de COCO bruite (bruit gaussien, noise_factor=0.15), puis sauvegarde le
modele dans src/models/unet_denoiser_tf210.keras (HDF5 sous TF 2.10, donc relisable
nativement par l'environnement de demonstration).

Usage :
    python src/retrain_l2.py                      # 2000 images, 10 epochs, bruit 0.15
    python src/retrain_l2.py --n-images 16 --epochs 1   # rodage rapide (smoke test)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, BatchNormalization,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR   = PROJECT_ROOT / "data" / "raw" / "train2014"
MODELS_DIR   = PROJECT_ROOT / "src" / "models"
OUT_PATH     = MODELS_DIR / "unet_denoiser_tf210.keras"

IMG_H = IMG_W = 256        # taille d'entree/sortie (identique au livrable 2)
CHANNELS = 3
SEED = 42
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------------------
# PERTE COMBINEE (identique au livrable 2 : 0.85*MSE + 0.15*(1-SSIM))
# ---------------------------------------------------------------------------
def combined_loss_weighted(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.85 * mse + 0.15 * (1.0 - ssim)


# ---------------------------------------------------------------------------
# ARCHITECTURE U-NET (reproduction exacte de build_unet, livrable 2 - cellule 34)
# ---------------------------------------------------------------------------
def build_unet(input_shape=(IMG_H, IMG_W, CHANNELS)):
    inputs = Input(shape=input_shape)

    # Encodeur
    c1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation="relu", padding="same")(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation="relu", padding="same")(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation="relu", padding="same")(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3, 3), activation="relu", padding="same")(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Pont
    b = Conv2D(512, (3, 3), activation="relu", padding="same")(p4)
    b = BatchNormalization()(b)
    b = Conv2D(512, (3, 3), activation="relu", padding="same")(b)

    # Decodeur (avec skip-connections)
    u4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(b)
    u4 = Concatenate()([u4, c4])
    u4 = Conv2D(256, (3, 3), activation="relu", padding="same")(u4)
    u4 = BatchNormalization()(u4)

    u3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(u4)
    u3 = Concatenate()([u3, c3])
    u3 = Conv2D(128, (3, 3), activation="relu", padding="same")(u3)
    u3 = BatchNormalization()(u3)

    u2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(u3)
    u2 = Concatenate()([u2, c2])
    u2 = Conv2D(64, (3, 3), activation="relu", padding="same")(u2)
    u2 = BatchNormalization()(u2)

    u1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(u2)
    u1 = Concatenate()([u1, c1])
    u1 = Conv2D(32, (3, 3), activation="relu", padding="same")(u1)
    u1 = BatchNormalization()(u1)

    outputs = Conv2D(CHANNELS, (1, 1), activation="sigmoid", padding="same")(u1)
    return Model(inputs, outputs, name="unet_denoiser")


# ---------------------------------------------------------------------------
# DONNEES
# ---------------------------------------------------------------------------
def load_clean_images(n_images):
    """Charge n images depuis data/raw/train2014/, normalisees dans [0, 1]."""
    files = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in EXTS)[:n_images]
    if not files:
        raise FileNotFoundError(f"Aucune image trouvee dans {IMAGES_DIR}")
    X = np.zeros((len(files), IMG_H, IMG_W, CHANNELS), dtype="float32")
    for i, p in enumerate(files):
        X[i] = img_to_array(load_img(p, target_size=(IMG_H, IMG_W))) / 255.0
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(files)} images chargees")
    print(f"Images chargees : {X.shape}")
    return X


def add_gaussian_noise(X, noise_factor, seed=SEED, chunk=256):
    """Ajoute un bruit gaussien (par blocs, pour limiter la memoire)."""
    rng = np.random.RandomState(seed)
    noisy = np.empty_like(X)
    for i in range(0, len(X), chunk):
        sl = slice(i, i + chunk)
        noise = rng.standard_normal(X[sl].shape).astype("float32")
        noisy[sl] = np.clip(X[sl] + noise_factor * noise, 0.0, 1.0)
    return noisy


class DenoiseSequence(Sequence):
    """Fournit les batchs (image bruitee -> image propre) depuis la RAM.

    Indispensable sur petit GPU (GTX 1050, 2 Go) : seules des images par batch sont
    copiees sur le GPU, jamais le dataset entier (sinon OOM 'Dst tensor not initialized').
    """

    def __init__(self, noisy, clean, batch_size, shuffle=True, seed=SEED):
        self.noisy, self.clean = noisy, clean
        self.bs, self.shuffle = batch_size, shuffle
        self.rng = np.random.RandomState(seed)
        self.order = np.arange(len(clean))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.clean) / self.bs))

    def __getitem__(self, i):
        sl = self.order[i * self.bs:(i + 1) * self.bs]
        return self.noisy[sl], self.clean[sl]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Re-entrainement du U-Net de debruitage (L2)")
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--noise-factor", type=float, default=0.15)
    ap.add_argument("--val-split", type=float, default=0.2)
    args = ap.parse_args()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    print("=" * 70)
    print("RE-ENTRAINEMENT L2 (U-Net debruiteur) — TF", tf.__version__)
    print(f"GPU : {tf.config.list_physical_devices('GPU') or 'aucun (CPU)'}")
    print(f"Images={args.n_images}  epochs={args.epochs}  batch={args.batch_size}  "
          f"bruit={args.noise_factor}")
    print("=" * 70)

    # 1) Donnees
    X = load_clean_images(args.n_images)
    print(f"Ajout du bruit gaussien (facteur {args.noise_factor})...")
    X_noisy = add_gaussian_noise(X, args.noise_factor)

    # melange puis split train/val (au niveau image)
    perm = np.random.RandomState(SEED).permutation(len(X))
    X, X_noisy = X[perm], X_noisy[perm]
    n_val = max(1, int(len(X) * args.val_split))
    X_val, Xn_val = X[:n_val], X_noisy[:n_val]
    X_tr, Xn_tr = X[n_val:], X_noisy[n_val:]
    print(f"Split train/val : {len(X_tr)} / {len(X_val)} images")

    train_seq = DenoiseSequence(Xn_tr, X_tr, args.batch_size, shuffle=True)
    val_seq = DenoiseSequence(Xn_val, X_val, args.batch_size, shuffle=False)

    # 2) Modele
    model = build_unet()
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=combined_loss_weighted, metrics=["mae"])
    print(f"U-Net construit : {model.count_params():,} parametres")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    ]

    # 3) Entrainement (entree = bruitee, cible = propre) — batchs depuis la RAM (cf. DenoiseSequence)
    t0 = time.time()
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    print(f"Entrainement termine en {(time.time()-t0)/60:.1f} min")

    # 4) Sauvegarde (HDF5 sous TF 2.10 -> relisible par l'env de demo)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_PATH)
    print(f"Modele sauvegarde -> {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    # 5) Verification : reload (compile=False, comme l'app) + PSNR avant/apres
    m2 = load_model(OUT_PATH, compile=False)
    sample_clean = X[:8]
    sample_noisy = X_noisy[:8]
    pred = np.clip(m2.predict(sample_noisy, verbose=0), 0, 1)

    def psnr(a, b):
        mse = np.mean((a - b) ** 2)
        return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)

    psnr_noisy = np.mean([psnr(c, n) for c, n in zip(sample_clean, sample_noisy)])
    psnr_denoise = np.mean([psnr(c, p) for c, p in zip(sample_clean, pred)])
    print(f"Reload compile=False : OK  | in/out {m2.input_shape} -> {m2.output_shape}")
    print(f"PSNR bruite   : {psnr_noisy:.2f} dB")
    print(f"PSNR debruite : {psnr_denoise:.2f} dB  (gain {psnr_denoise - psnr_noisy:+.2f} dB)")


if __name__ == "__main__":
    main()
