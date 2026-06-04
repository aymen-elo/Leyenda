"""
Projet Leyenda - Pipeline TouNum
================================
Interface de demonstration Streamlit pour la soutenance.

Pipeline complet sur des images inconnues :
    L1  Classification  (Photo / Non-Photo)   -> src/models/cnn_scratch.keras    (optionnel)
    L2  Debruitage      (U-Net)               -> notebooks/unet_model.keras
    L3  Captioning      (InceptionV3 + LSTM)   -> src/models/captioning_model.keras + tokenizer

La logique de generation de legende est adaptee de
    notebooks/livrable3_captioning.ipynb  (generate_caption, encodeur InceptionV3, tokenizer).

Lancement :  streamlit run app.py    (ou double-clic sur run_demo.bat)
"""

import io
import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIGURATION & CHEMINS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR   = PROJECT_ROOT / "src" / "models"

L1_MODEL_PATH = MODELS_DIR / "cnn_scratch.keras"               # Classification  (peut etre absent)
L2_MODEL_PATH = PROJECT_ROOT / "notebooks" / "unet_model.keras"  # Debruitage
L3_MODEL_PATH = MODELS_DIR / "captioning_model.keras"          # Captioning
TOKENIZER_PKL = MODELS_DIR / "captioning_tokenizer.pkl"

# Hyperparametres encodeur (coherents avec le livrable 3)
IMG_SIZE   = 299      # taille d'entree InceptionV3
L1_SIZE    = 224      # taille d'entree du classifieur L1
L2_SIZE    = 256      # taille d'entree de l'U-Net L2
PAD_TOK, START_TOK, END_TOK, UNK_TOK = "<pad>", "<start>", "<end>", "<unk>"

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")

# Palette TouNum : bleu nuit + accents ambre/orange
C_BG       = "#0e1525"
C_CARD     = "#161f33"
C_AMBER    = "#f5a623"
C_ORANGE   = "#ff7a18"
C_GREEN    = "#2ecc71"
C_RED      = "#e74c3c"
C_TEXT     = "#e8edf6"
C_MUTED    = "#8a97ad"

st.set_page_config(
    page_title="Projet Leyenda - TouNum",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME SOMBRE (CSS)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(1200px 600px at 20% -10%, #16223d 0%, {C_BG} 55%);
        color: {C_TEXT};
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #101a30 0%, #0b1222 100%);
        border-right: 1px solid #22304d;
    }}
    h1, h2, h3, h4 {{ color: {C_TEXT}; letter-spacing: .3px; }}
    .ly-title {{
        font-size: 2.2rem; font-weight: 800; margin-bottom: .1rem;
        background: linear-gradient(90deg, {C_AMBER}, {C_ORANGE});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .ly-sub {{ color: {C_MUTED}; font-size: 1.02rem; margin-top: 0; }}
    .ly-card {{
        background: {C_CARD}; border: 1px solid #233152; border-radius: 16px;
        padding: 18px 20px; margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(0,0,0,.35);
    }}
    .ly-caption {{
        font-size: 1.45rem; font-weight: 700; line-height: 1.4;
        color: {C_TEXT}; padding: 8px 4px;
    }}
    .ly-caption::before {{ content: "💬 "; }}
    .badge {{
        display: inline-block; padding: 5px 14px; border-radius: 999px;
        font-weight: 700; font-size: .92rem; color: #0b1222;
    }}
    .badge-photo   {{ background: {C_GREEN}; }}
    .badge-nonphoto{{ background: {C_RED}; color: #fff; }}
    .badge-default {{ background: {C_AMBER}; }}
    .chip {{
        display:inline-block; padding:3px 10px; border-radius:8px; margin-right:6px;
        font-size:.82rem; background:#1d2944; color:{C_MUTED}; border:1px solid #2a3a5c;
    }}
    .stat-num {{ font-size: 2.4rem; font-weight: 800; color: {C_AMBER}; }}
    .stat-lbl {{ color: {C_MUTED}; font-size: .95rem; }}
    .status-ok   {{ color:{C_GREEN};  font-weight:700; }}
    .status-warn {{ color:{C_AMBER};  font-weight:700; }}
    .status-ko   {{ color:{C_RED};    font-weight:700; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background: #14203a; border-radius: 10px 10px 0 0; padding: 8px 18px;
    }}
    .stTabs [aria-selected="true"] {{ background: #1d2d4e; color: {C_AMBER}; }}
    div.stButton > button {{
        background: linear-gradient(90deg, {C_ORANGE}, {C_AMBER});
        color:#0b1222; font-weight:800; border:0; border-radius:12px;
        padding:.6rem 1rem; width:100%;
    }}
    div.stButton > button:hover {{ filter: brightness(1.08); }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# CHARGEMENT DES MODELES (une seule fois)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des modeles (TensorFlow + InceptionV3)...")
def load_pipeline():
    """Charge l'encodeur, le modele de captioning, le tokenizer et les modeles L1/L2.

    Retourne un dict avec les objets et le statut de chaque etape. Defensif :
    l'absence ou l'echec de chargement d'un modele ne fait jamais planter l'appli.
    """
    import pickle

    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.models import Model, load_model

    status = {}

    # --- GPU : croissance memoire (GTX 1050) ---
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

    # --- L3 : modele de captioning + tokenizer ---
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
        status["L3"] = ("ko", "modele/tokenizer/encodeur absent")

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
# ETAPES DU PIPELINE (adaptees du livrable 3)
# ---------------------------------------------------------------------------
def generate_caption(model, feature, tok):
    """Decodage glouton : predit le mot le plus probable jusqu'a <end>.
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
    """L1 -> (label, proba, ms). Suppose 'Photo (par defaut)' si modele absent."""
    from tensorflow.keras.preprocessing.image import img_to_array

    t0 = time.perf_counter()
    if P["l1_model"] is None:
        return "Photo (par defaut)", None, (time.perf_counter() - t0) * 1000
    arr = img_to_array(pil_img.resize((L1_SIZE, L1_SIZE))) / 255.0
    p = float(P["l1_model"].predict(arr[None, ...], verbose=0).ravel()[0])
    label = "Photo" if p >= 0.5 else "Non-Photo"
    return label, p, (time.perf_counter() - t0) * 1000


def step_denoise(P, pil_img):
    """L2 -> (image debruitee, ms). Renvoie l'originale si modele absent."""
    from tensorflow.keras.preprocessing.image import img_to_array

    t0 = time.perf_counter()
    if P["l2_model"] is None:
        return None, (time.perf_counter() - t0) * 1000
    arr = img_to_array(pil_img.resize((L2_SIZE, L2_SIZE))) / 255.0
    out = P["l2_model"].predict(arr[None, ...], verbose=0)[0]
    out = np.clip(out, 0, 1)
    return Image.fromarray((out * 255).astype("uint8")), (time.perf_counter() - t0) * 1000


def step_caption(P, pil_img):
    """L3 -> (legende, ms). Extraction features InceptionV3 puis decodage."""
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    t0 = time.perf_counter()
    if P["caption_model"] is None or P["encoder"] is None or P["tok"] is None:
        return "(L3 indisponible)", (time.perf_counter() - t0) * 1000
    arr = preprocess_input(img_to_array(pil_img.resize((IMG_SIZE, IMG_SIZE))))
    feat = P["encoder"].predict(arr[None, ...], verbose=0)[0]
    cap = generate_caption(P["caption_model"], feat, P["tok"])
    return cap, (time.perf_counter() - t0) * 1000


def process_image(P, name, pil_img):
    """Orchestration L1 -> L2 -> L3 sur une image, robuste etape par etape."""
    res = {"name": name, "image": pil_img, "errors": []}

    # L1
    try:
        res["label"], res["proba"], res["t1"] = step_classify(P, pil_img)
    except Exception as e:
        res["label"], res["proba"], res["t1"] = "Photo (par defaut)", None, 0.0
        res["errors"].append(f"L1: {e}")

    # L2 (l'image debruitee alimente L3 si disponible)
    try:
        res["denoised"], res["t2"] = step_denoise(P, pil_img)
    except Exception as e:
        res["denoised"], res["t2"] = None, 0.0
        res["errors"].append(f"L2: {e}")

    src_for_caption = res["denoised"] if res["denoised"] is not None else pil_img

    # L3
    try:
        res["caption"], res["t3"] = step_caption(P, src_for_caption)
    except Exception as e:
        res["caption"], res["t3"] = "(echec L3 sur cette image)", 0.0
        res["errors"].append(f"L3: {e}")

    return res


# ---------------------------------------------------------------------------
# RENDU D'UNE CARTE RESULTAT
# ---------------------------------------------------------------------------
def badge_html(label):
    if label.startswith("Photo (par"):
        return f'<span class="badge badge-default">⚠️ {label}</span>'
    if label == "Photo":
        return f'<span class="badge badge-photo">✅ Photo</span>'
    return f'<span class="badge badge-nonphoto">⛔ {label}</span>'


def render_card(res):
    st.markdown('<div class="ly-card">', unsafe_allow_html=True)
    st.markdown(f"#### 🖼️ {res['name']}")

    col_o, col_d = st.columns(2)
    with col_o:
        st.markdown("**Originale**")
        st.image(res["image"], use_container_width=True)
    with col_d:
        if res["denoised"] is not None:
            st.markdown("**🧹 Debruitee (L2)**")
            st.image(res["denoised"], use_container_width=True)
        else:
            st.markdown("**🧹 Debruitage (L2)**")
            st.info("L2 desactive - image originale utilisee pour la suite.")

    proba_txt = f" &nbsp; <span class='chip'>p = {res['proba']:.2f}</span>" if res.get("proba") is not None else ""
    st.markdown(
        f"<div style='margin:10px 0;'>🔍 <b>Classification L1 :</b> {badge_html(res['label'])}{proba_txt}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='ly-caption'>{res['caption']}</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='margin-top:8px;'>"
        f"<span class='chip'>L1 : {res['t1']:.0f} ms</span>"
        f"<span class='chip'>L2 : {res['t2']:.0f} ms</span>"
        f"<span class='chip'>L3 : {res['t3']:.0f} ms</span>"
        f"<span class='chip'>Total : {res['t1']+res['t2']+res['t3']:.0f} ms</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if res["errors"]:
        st.warning("Avertissements : " + " | ".join(res["errors"]))
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# COLLECTE DES IMAGES (upload + dossier)
# ---------------------------------------------------------------------------
def gather_images(uploaded_files, folder_str, limit):
    """Retourne une liste de (nom, PIL.Image) a partir des uploads et/ou d'un dossier."""
    items = []
    if uploaded_files:
        for uf in uploaded_files:
            try:
                items.append((uf.name, Image.open(io.BytesIO(uf.getvalue())).convert("RGB")))
            except Exception:
                pass
    if folder_str:
        folder = Path(folder_str.strip().strip('"'))
        if folder.is_dir():
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
            for p in files:
                try:
                    items.append((p.name, Image.open(p).convert("RGB")))
                except Exception:
                    pass
    return items[:limit]


# ===========================================================================
# INTERFACE
# ===========================================================================
P = load_pipeline()
status = P["status"]

# --- En-tete ---
st.markdown('<div class="ly-title">Projet Leyenda — Pipeline TouNum</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="ly-sub">Numerisation intelligente : classification (L1) · débruitage (L2) · '
    'génération de légende (L3) — démonstration live sur des images inconnues.</p>',
    unsafe_allow_html=True,
)

# --- Barre laterale ---
with st.sidebar:
    st.markdown("<div style='font-size:3rem; text-align:center;'>🏛️</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; color:#8a97ad;'>CESI · Groupe 2 · TouNum</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Statut des modèles")

    def show_status(stage, label):
        kind, msg = status.get(stage, ("ko", "inconnu"))
        icon = {"ok": "✅", "warn": "⚠️", "ko": "❌"}[kind]
        cls = {"ok": "status-ok", "warn": "status-warn", "ko": "status-ko"}[kind]
        st.markdown(
            f"<div><span class='{cls}'>{icon} {label}</span><br>"
            f"<span style='color:#8a97ad; font-size:.82rem;'>{msg}</span></div>",
            unsafe_allow_html=True,
        )

    show_status("L1", "L1 · Classification")
    show_status("L2", "L2 · Débruitage")
    show_status("L3", "L3 · Captioning")

    st.markdown("---")
    st.markdown(f"<span class='chip'>TF {P['tf_version']}</span>"
                f"<span class='chip'>{'GPU ✅' if P['gpu'] else 'CPU'}</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    n_images = st.slider("Nombre d'images à traiter", 1, 50, 8)
    run = st.button("🚀 Lancer le pipeline")

# --- Onglets ---
tab_pipe, tab_stats, tab_arch = st.tabs(["🔍 Pipeline", "📊 Statistiques", "🏗️ Architecture"])

# ============================ ONGLET PIPELINE ==============================
with tab_pipe:
    st.markdown("#### 🖼️ Sélection des images")
    uploaded = st.file_uploader(
        "Glissez-déposez vos images ici (sélection multiple)",
        type=[e.strip(".") for e in EXTS],
        accept_multiple_files=True,
    )
    folder_str = st.text_input(
        "…ou indiquez le chemin d'un dossier d'images (sur cette machine)",
        placeholder=r"C:\Users\matis\Desktop\dataset_prof",
    )

    if run:
        items = gather_images(uploaded, folder_str, n_images)
        if not items:
            st.error("Aucune image trouvée. Uploadez des fichiers ou indiquez un dossier valide.")
        else:
            st.success(f"{len(items)} image(s) à traiter.")
            results = []
            progress = st.progress(0.0, text="Traitement en cours…")
            for i, (name, img) in enumerate(items):
                with st.spinner(f"Pipeline sur « {name} » ({i+1}/{len(items)})…"):
                    results.append(process_image(P, name, img))
                progress.progress((i + 1) / len(items), text=f"{i+1}/{len(items)} traitées")
            progress.empty()
            st.session_state["results"] = results

    results = st.session_state.get("results", [])
    if results:
        st.markdown(f"### Résultats ({len(results)})")
        for res in results:
            render_card(res)
    elif not run:
        st.info("Chargez des images puis cliquez sur **🚀 Lancer le pipeline** dans la barre latérale.")

# ============================ ONGLET STATISTIQUES ==========================
with tab_stats:
    results = st.session_state.get("results", [])
    st.markdown("#### 📊 Statistiques de la dernière exécution")
    if not results:
        st.info("Aucune exécution pour le moment.")
    else:
        n_photo = sum(1 for r in results if r["label"] == "Photo" or r["label"].startswith("Photo (par"))
        n_nonphoto = sum(1 for r in results if r["label"] == "Non-Photo")
        avg_total = np.mean([r["t1"] + r["t2"] + r["t3"] for r in results])
        avg_l3 = np.mean([r["t3"] for r in results])

        c1, c2, c3, c4 = st.columns(4)
        for col, num, lbl in [
            (c1, len(results), "Images traitées"),
            (c2, n_photo, "Photos"),
            (c3, n_nonphoto, "Non-Photos"),
            (c4, f"{avg_total:.0f} ms", "Temps moyen / image"),
        ]:
            col.markdown(
                f"<div class='ly-card' style='text-align:center;'>"
                f"<div class='stat-num'>{num}</div><div class='stat-lbl'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("##### Détail des temps par étape (ms)")
        import pandas as pd

        df_stats = pd.DataFrame(
            [{"Image": r["name"], "Classe": r["label"],
              "L1 (ms)": round(r["t1"]), "L2 (ms)": round(r["t2"]),
              "L3 (ms)": round(r["t3"]),
              "Total (ms)": round(r["t1"] + r["t2"] + r["t3"])}
             for r in results]
        )
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        st.markdown(f"<span class='chip'>Temps moyen L3 : {avg_l3:.0f} ms</span>", unsafe_allow_html=True)

# ============================ ONGLET ARCHITECTURE ==========================
with tab_arch:
    st.markdown("#### 🏗️ Architecture du pipeline TouNum")
    st.markdown(
        """
```
   Image inconnue
        │
        ▼
┌──────────────────┐   L1 · Classification (CNN from scratch)
│  PHOTO / NON-PHOTO │   src/models/cnn_scratch.keras
└──────────────────┘   → Photo (par défaut) si absent
        │
        ▼
┌──────────────────┐   L2 · Débruitage (U-Net, MSE+SSIM)
│   IMAGE NETTOYÉE  │   notebooks/unet_model.keras
└──────────────────┘   → image originale si désactivé
        │
        ▼
┌──────────────────┐   L3 · Captioning (InceptionV3 → LSTM)
│  LÉGENDE GÉNÉRÉE  │   src/models/captioning_model.keras
└──────────────────┘   + tokenizer (vocab 5000)
        │
        ▼
   « a man riding a wave on a surfboard »
```
        """
    )

    st.markdown("##### Détails techniques")
    cA, cB, cC = st.columns(3)
    with cA:
        st.markdown(
            f"<div class='ly-card'><b>🔍 L1 · Classification</b><br>"
            f"<span class='stat-lbl'>CNN entraîné from scratch.<br>"
            f"Entrée 224×224, sortie binaire (Photo / Non-Photo).<br>"
            f"Statut : {status['L1'][0].upper()}</span></div>",
            unsafe_allow_html=True,
        )
    with cB:
        st.markdown(
            f"<div class='ly-card'><b>🧹 L2 · Débruitage</b><br>"
            f"<span class='stat-lbl'>U-Net convolutif, perte MSE+SSIM.<br>"
            f"Entrée/sortie 256×256 RGB.<br>"
            f"Statut : {status['L2'][0].upper()}</span></div>",
            unsafe_allow_html=True,
        )
    with cC:
        st.markdown(
            f"<div class='ly-card'><b>💬 L3 · Captioning</b><br>"
            f"<span class='stat-lbl'>Encodeur InceptionV3 (2048-d, gelé) +<br>"
            f"décodeur LSTM par fusion. Vocab 5000, décodage glouton.<br>"
            f"Statut : {status['L3'][0].upper()}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("##### Environnement")
    st.markdown(
        f"<span class='chip'>TensorFlow {P['tf_version']}</span>"
        f"<span class='chip'>{'GPU : ' + P['gpu'][0] if P['gpu'] else 'CPU uniquement'}</span>"
        f"<span class='chip'>Encodeur : InceptionV3 / ImageNet</span>",
        unsafe_allow_html=True,
    )
