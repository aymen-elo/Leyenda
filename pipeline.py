"""
Pipeline TouNum - logique d'inference (sans dependance Streamlit).
=================================================================
Trois etapes, chacune robuste a l'absence de son modele :
    L1  Classification  Photo / Non-Photo   -> src/models/cnn_scratch.keras   (optionnel)
    L2  Debruitage      U-Net               -> notebooks/unet_model_tf210.h5
    L3  Captioning      InceptionV3 + LSTM   -> src/models/captioning_model.keras + tokenizer

La generation de legende est adaptee de notebooks/livrable3_captioning.ipynb
(generate_caption, encodeur InceptionV3, tokenizer pickle).

Ce module est volontairement independant de Streamlit pour pouvoir etre teste
et reutilise en dehors de l'interface.
"""

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# CHEMINS & CONSTANTES
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR   = PROJECT_ROOT / "src" / "models"

L1_MODEL_PATH = MODELS_DIR / "cnn_scratch.keras"               # Classification (peut etre absent)

# L2 : ordre de priorite des modeles de debruitage —
#   1. unet_denoiser_tf210.keras : U-Net re-entraine nativement sous TF 2.10 (src/retrain_l2.py)
#   2. unet_model_tf210.h5        : reconstruction HDF5 du modele Keras 3 d'origine
#   3. unet_model.keras           : modele Keras 3 d'origine (illisible par TF 2.10 -> neutralise)
_L2_RETRAINED = MODELS_DIR / "unet_denoiser_tf210.keras"
_L2_H5        = PROJECT_ROOT / "notebooks" / "unet_model_tf210.h5"
_L2_KERAS     = PROJECT_ROOT / "notebooks" / "unet_model.keras"
L2_MODEL_PATH = next((p for p in (_L2_RETRAINED, _L2_H5, _L2_KERAS) if p.exists()), _L2_KERAS)

L3_MODEL_PATH = MODELS_DIR / "captioning_model.keras"          # Captioning
TOKENIZER_PKL = MODELS_DIR / "captioning_tokenizer.pkl"

IMG_SIZE = 299     # entree InceptionV3 (encodeur L3)
L1_SIZE  = 224     # entree du classifieur L1
L2_SIZE  = 256     # entree/sortie de l'U-Net L2
PAD_TOK, START_TOK, END_TOK, UNK_TOK = "<pad>", "<start>", "<end>", "<unk>"

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")

# Dossier d'images embarque pour une demonstration immediate (bouton "Demo rapide")
DEMO_DIR = PROJECT_ROOT / "Dataset"


# ---------------------------------------------------------------------------
# CHARGEMENT DES MODELES (defensif : aucune etape ne fait planter l'appli)
# ---------------------------------------------------------------------------
def build_pipeline():
    """Charge encodeur + captioning + tokenizer + L1/L2 et renvoie un dict d'etat."""
    import pickle

    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.models import Model, load_model

    status = {}

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # --- Encodeur InceptionV3 (features 2048-d, gele) ---
    encoder = None
    try:
        base = InceptionV3(include_top=False, weights="imagenet", pooling="avg")
        base.trainable = False
        encoder = Model(base.input, base.output, name="inceptionv3_encoder")
        encoder.trainable = False
    except Exception as e:
        status["encoder_err"] = str(e)

    # --- L3 : captioning + tokenizer ---
    caption_model, tok = None, None
    if L3_MODEL_PATH.exists() and TOKENIZER_PKL.exists() and encoder is not None:
        try:
            caption_model = load_model(L3_MODEL_PATH, compile=False)
            with open(TOKENIZER_PKL, "rb") as f:
                tok = pickle.load(f)
            status["L3"] = ("ok", L3_MODEL_PATH.name)
        except Exception as e:
            status["L3"] = ("ko", f"erreur de chargement : {e}")
    else:
        status["L3"] = ("ko", "modele / tokenizer / encodeur absent")

    # --- L1 : classification (optionnel) ---
    l1_model = None
    if L1_MODEL_PATH.exists():
        try:
            l1_model = load_model(L1_MODEL_PATH, compile=False)
            status["L1"] = ("ok", L1_MODEL_PATH.name)
        except Exception as e:
            status["L1"] = ("warn", f"present mais non chargeable ({e}) -> Photo par defaut")
    else:
        status["L1"] = ("warn", "absent -> 'Photo (par defaut)'")

    # --- L2 : debruitage U-Net ---
    l2_model = None
    if L2_MODEL_PATH.exists():
        try:
            l2_model = load_model(L2_MODEL_PATH, compile=False)
            status["L2"] = ("ok", L2_MODEL_PATH.name)
        except Exception as e:
            status["L2"] = ("warn", f"present mais non chargeable ({e}) -> L2 desactive")
    else:
        status["L2"] = ("warn", "absent -> L2 desactive")

    return {
        "tf_version": tf.__version__,
        "gpu": [g.name for g in tf.config.list_physical_devices("GPU")],
        "encoder": encoder,
        "caption_model": caption_model,
        "tok": tok,
        "l1_model": l1_model,
        "l2_model": l2_model,
        "status": status,
    }


# ---------------------------------------------------------------------------
# ETAPES (adaptees du livrable 3)
# ---------------------------------------------------------------------------
def generate_caption(model, feature, tok):
    """Decodage glouton : mot le plus probable a chaque pas jusqu'a <end>.
    Adapte de generate_caption() du notebook livrable3_captioning.ipynb."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    word2idx, idx2word, max_len = tok["word2idx"], tok["idx2word"], tok["max_len"]
    seq = [word2idx[START_TOK]]
    for _ in range(max_len):
        padded = pad_sequences([seq], maxlen=max_len, padding="post")
        yhat = model.predict([feature[None, :], padded], verbose=0)
        nxt = int(np.argmax(yhat[0]))
        if nxt == word2idx[END_TOK]:
            break
        seq.append(nxt)
    words = [
        idx2word[i]
        for i in seq[1:]
        if i not in (word2idx[START_TOK], word2idx[END_TOK], word2idx[PAD_TOK])
    ]
    return " ".join(words) if words else "(aucune legende generee)"


def step_classify(P, pil_img):
    """L1 -> (label, proba, ms). 'Photo (par defaut)' si modele absent."""
    from tensorflow.keras.preprocessing.image import img_to_array

    t0 = time.perf_counter()
    if P["l1_model"] is None:
        return "Photo (par defaut)", None, (time.perf_counter() - t0) * 1000
    arr = img_to_array(pil_img.resize((L1_SIZE, L1_SIZE))) / 255.0
    p = float(P["l1_model"].predict(arr[None, ...], verbose=0).ravel()[0])
    return ("Photo" if p >= 0.5 else "Non-Photo"), p, (time.perf_counter() - t0) * 1000


def step_denoise(P, pil_img):
    """L2 -> (image debruitee PIL ou None, ms). None si modele absent."""
    from tensorflow.keras.preprocessing.image import img_to_array

    t0 = time.perf_counter()
    if P["l2_model"] is None:
        return None, (time.perf_counter() - t0) * 1000
    arr = img_to_array(pil_img.resize((L2_SIZE, L2_SIZE))) / 255.0
    out = np.clip(P["l2_model"].predict(arr[None, ...], verbose=0)[0], 0, 1)
    denoised = Image.fromarray((out * 255).astype("uint8"))
    # On rend l'image debruitee a la resolution d'origine (pas de perte de definition).
    if denoised.size != pil_img.size:
        denoised = denoised.resize(pil_img.size, Image.LANCZOS)
    return denoised, (time.perf_counter() - t0) * 1000


def step_caption(P, pil_img):
    """L3 -> (legende, ms). Extraction features InceptionV3 + decodage glouton."""
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    t0 = time.perf_counter()
    if P["caption_model"] is None or P["encoder"] is None or P["tok"] is None:
        return "(L3 indisponible)", (time.perf_counter() - t0) * 1000
    arr = preprocess_input(img_to_array(pil_img.resize((IMG_SIZE, IMG_SIZE))))
    feat = P["encoder"].predict(arr[None, ...], verbose=0)[0]
    return generate_caption(P["caption_model"], feat, P["tok"]), (time.perf_counter() - t0) * 1000


def process_image(P, name, pil_img):
    """Orchestration L1 -> L2 -> L3, robuste etape par etape."""
    res = {"name": name, "image": pil_img, "errors": []}

    try:
        res["label"], res["proba"], res["t1"] = step_classify(P, pil_img)
    except Exception as e:
        res["label"], res["proba"], res["t1"] = "Photo (par defaut)", None, 0.0
        res["errors"].append(f"L1: {e}")

    try:
        res["denoised"], res["t2"] = step_denoise(P, pil_img)
    except Exception as e:
        res["denoised"], res["t2"] = None, 0.0
        res["errors"].append(f"L2: {e}")

    # Intensite de correction L2 : ecart moyen original/debruite (0 = aucune correction)
    if res["denoised"] is not None:
        o = np.asarray(pil_img.resize((L2_SIZE, L2_SIZE)), dtype="float32") / 255.0
        d = np.asarray(res["denoised"].resize((L2_SIZE, L2_SIZE)), dtype="float32") / 255.0
        res["noise_delta"] = float(np.abs(o - d).mean())
    else:
        res["noise_delta"] = None

    src = res["denoised"] if res["denoised"] is not None else pil_img
    try:
        res["caption"], res["t3"] = step_caption(P, src)
    except Exception as e:
        res["caption"], res["t3"] = "(echec L3 sur cette image)", 0.0
        res["errors"].append(f"L3: {e}")

    return res


def model_params(P):
    """Nombre de parametres de chaque modele charge (None si absent)."""
    out = {}
    for key, name in [("encoder", "encoder"), ("caption_model", "L3"),
                      ("l1_model", "L1"), ("l2_model", "L2")]:
        m = P.get(key)
        try:
            out[name] = int(m.count_params()) if m is not None else None
        except Exception:
            out[name] = None
    return out


def results_to_csv(results):
    """Serialise les resultats du pipeline en CSV (pour export/livraison)."""
    import csv

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["image", "classe", "proba", "legende",
                "L1_ms", "L2_ms", "L3_ms", "total_ms", "correction_L2_pct"])
    for r in results:
        w.writerow([
            r["name"], r["label"],
            "" if r.get("proba") is None else round(r["proba"], 4),
            r["caption"], round(r["t1"]), round(r["t2"]), round(r["t3"]),
            round(r["t1"] + r["t2"] + r["t3"]),
            "" if r.get("noise_delta") is None else round(r["noise_delta"] * 100, 2),
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# COLLECTE DES IMAGES (upload + dossier)
# ---------------------------------------------------------------------------
def open_image(data):
    """Ouvre une image (chemin, bytes ou file-like) en RGB."""
    if isinstance(data, (bytes, bytearray)):
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(data).convert("RGB")


def gather_images(uploaded_files, folder_str, limit):
    """Retourne une liste (nom, PIL.Image) issue des uploads et/ou d'un dossier."""
    items = []
    for uf in uploaded_files or []:
        try:
            items.append((uf.name, open_image(uf.getvalue())))
        except Exception:
            pass
    if folder_str:
        folder = Path(folder_str.strip().strip('"'))
        if folder.is_dir():
            for p in sorted(q for q in folder.iterdir() if q.suffix.lower() in EXTS):
                try:
                    items.append((p.name, open_image(p)))
                except Exception:
                    pass
    return items[:limit]
