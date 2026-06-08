"""
Re-entrainement du modele de classification L1 (CNN from scratch) sous TensorFlow 2.10.
=====================================================================================
Le classifieur binaire Photo / Non-Photo du livrable 1 (cnn_scratch.keras) etait absent.
Ce script reproduit *exactement* l'architecture `build_cnn` du livrable 1
(notebooks/leyenda_groupe_2_livrable_1.ipynb, cellule du CNN_Scratch regularise) :

    4 blocs Conv2D (32 -> 64 -> 128 -> 256) + BatchNorm + MaxPooling   (regularisation L2)
    GlobalAveragePooling2D
    Dense(128) + BatchNorm + Dropout(0.5)
    Dense(64)  + BatchNorm + Dropout(0.3)
    Dense(1, sigmoid)                                                  -> P(Photo)

Difference volontaire avec le notebook : la couche `Rescaling(1./255)` n'est PAS integree
au modele. La normalisation [0, 1] est faite en amont (dans le pipeline, `step_classify`
divise deja par 255). Integrer Rescaling produirait une double division -> entrees a ~0.
L'augmentation de donnees est appliquee uniquement au flux d'entrainement (jamais sauvegardee
dans le modele), pour garder une inference propre et compatible avec pipeline.py.

Donnees (data/raw/) : Photo -> label 1 ; Painting/Schematics/Sketch/Text -> label 0.

Usage :
    python src/retrain_l1.py                       # 15 epochs, batch 16
    python src/retrain_l1.py --epochs 1            # smoke test rapide
"""

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
MODELS_DIR   = PROJECT_ROOT / "src" / "models"
OUT_PATH     = MODELS_DIR / "cnn_scratch.keras"

IMAGE_H = IMAGE_W = 224       # entree du classifieur (identique au livrable 1)
CHANNELS = 3
SEED = 42
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")

# Photo = classe cible (label 1). Tout le reste = Non-Photo (label 0).
# On accepte indifferemment la nomenclature du sujet (photos / non_photos) et celle
# reellement presente dans le depot (Photo / Painting / Schematics / Sketch / Text).
PHOTO_DIRNAMES    = ("photos", "Photo", "photo")
NONPHOTO_DIRNAMES = ("non_photos", "Painting", "Schematics", "Sketch", "Text")


# ---------------------------------------------------------------------------
# ARCHITECTURE CNN (reproduction de build_cnn, livrable 1 - CNN_Scratch regularise)
# ---------------------------------------------------------------------------
def build_cnn(input_shape=(IMAGE_H, IMAGE_W, CHANNELS)):
    """CNN binaire regularise : L2, BatchNorm, Dropout (entrees attendues dans [0, 1])."""
    l2 = tf.keras.regularizers.l2(1e-4)
    model = Sequential(name="CNN_Scratch")
    model.add(layers.Input(shape=input_shape))

    # 4 blocs convolutifs avec regularisation L2
    for filters in [32, 64, 128, 256]:
        model.add(Conv2D(filters, 3, padding="same", activation="relu",
                         kernel_regularizer=l2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation="relu", kernel_regularizer=l2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(64, activation="relu", kernel_regularizer=l2))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# DONNEES
# ---------------------------------------------------------------------------
def _collect_images(dirnames):
    """Collecte recursivement les images des dossiers existants parmi `dirnames`."""
    paths = []
    for name in dirnames:
        d = RAW_DIR / name
        if d.is_dir():
            paths += [p for p in d.rglob("*") if p.suffix.lower() in EXTS]
    return sorted(set(paths))


def load_dataset():
    """Charge Photo (label 1) et Non-Photo (label 0), normalises dans [0, 1]."""
    photo_paths    = _collect_images(PHOTO_DIRNAMES)
    nonphoto_paths = _collect_images(NONPHOTO_DIRNAMES)
    if not photo_paths:
        raise FileNotFoundError(f"Aucune image Photo trouvee dans {RAW_DIR} ({PHOTO_DIRNAMES})")
    if not nonphoto_paths:
        raise FileNotFoundError(f"Aucune image Non-Photo trouvee dans {RAW_DIR} ({NONPHOTO_DIRNAMES})")

    paths  = photo_paths + nonphoto_paths
    labels = [1] * len(photo_paths) + [0] * len(nonphoto_paths)
    print(f"Images : {len(photo_paths)} Photo (1) + {len(nonphoto_paths)} Non-Photo (0) "
          f"= {len(paths)} au total")

    X = np.zeros((len(paths), IMAGE_H, IMAGE_W, CHANNELS), dtype="float32")
    for i, p in enumerate(paths):
        X[i] = img_to_array(load_img(p, target_size=(IMAGE_H, IMAGE_W), color_mode="rgb")) / 255.0
    y = np.array(labels, dtype="float32")
    return X, y


def stratified_split(X, y, val_split=0.2, seed=SEED):
    """Split train/val stratifie par classe (chaque classe coupee a 80/20)."""
    rng = np.random.RandomState(seed)
    tr_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_split)))
        val_idx += list(idx[:n_val])
        tr_idx  += list(idx[n_val:])
    rng.shuffle(tr_idx)
    rng.shuffle(val_idx)
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def make_augmenter():
    """Augmentation appliquee uniquement a l'entrainement (cf. livrable 1)."""
    return Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Re-entrainement du classifieur L1 (CNN from scratch)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
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
    print("RE-ENTRAINEMENT L1 (CNN_Scratch Photo/Non-Photo) — TF", tf.__version__)
    print(f"GPU : {tf.config.list_physical_devices('GPU') or 'aucun (CPU)'}")
    print(f"epochs={args.epochs}  batch={args.batch_size}  val_split={args.val_split}  seed={SEED}")
    print("=" * 70)

    # 1) Donnees + split 80/20 stratifie
    X, y = load_dataset()
    X_tr, y_tr, X_val, y_val = stratified_split(X, y, args.val_split)
    print(f"Split train/val : {len(X_tr)} / {len(X_val)} images "
          f"(train: {int(y_tr.sum())} Photo / {int((1-y_tr).sum())} Non-Photo)")

    # Ponderation des classes : le jeu est desequilibre (Photo minoritaire). Sans cela,
    # le modele predirait systematiquement Non-Photo.
    n_pos, n_neg = max(1, int(y_tr.sum())), max(1, int((1 - y_tr).sum()))
    total = n_pos + n_neg
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}
    print(f"class_weight : {class_weight}")

    # tf.data : augmentation sur le train uniquement, puis batch
    augment = make_augmenter()
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .shuffle(len(X_tr), seed=SEED)
        .batch(args.batch_size)
        .map(lambda xb, yb: (augment(xb, training=True), yb),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 2) Modele
    model = build_cnn()
    model.summary()
    print(f"CNN construit : {model.count_params():,} parametres")

    # 3) Entrainement
    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        verbose=2,
    )
    print(f"Entrainement termine en {(time.time()-t0)/60:.1f} min")

    # 4) Sauvegarde
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_PATH)
    print(f"Modele sauvegarde -> {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    # 5) Verification : reload (compile=False, comme l'app) + predictions sur l'ensemble
    m2 = load_model(OUT_PATH, compile=False)
    preds = m2.predict(X, verbose=0).ravel()
    acc = float(((preds >= 0.5).astype("float32") == y).mean())
    print(f"Reload compile=False : OK  | in/out {m2.input_shape} -> {m2.output_shape}")
    print(f"Accuracy sur l'ensemble des donnees : {acc:.3f}")
    print(f"  proba moyenne Photo     : {preds[y == 1].mean():.3f}")
    print(f"  proba moyenne Non-Photo : {preds[y == 0].mean():.3f}")


if __name__ == "__main__":
    main()
