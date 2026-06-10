# -*- coding: utf-8 -*-
"""
Projet Leyenda - Pipeline TouNum
Interface de demonstration : classification -> debruitage -> captioning.

NOTE pour Matisse :
Ce fichier ne change RIEN a la logique IA. Toute la partie "metier"
(chargement des modeles, generate_caption, InceptionV3, tokenizer) est
isolee dans la section LOGIQUE METIER. Si tes noms de fichiers/fonctions
different, ce sont les SEULS endroits a ajuster -> cherche les balises
[A VERIFIER].
"""

import os
import time
import io

import numpy as np
import streamlit as st
from PIL import Image

# Imports TF gardes paresseux pour ne pas ralentir le 1er affichage.
import tensorflow as tf
import pickle

# =============================================================================
# CONFIG PAGE  (doit etre le tout premier appel Streamlit)
# =============================================================================
st.set_page_config(
    page_title="Leyenda - Pipeline TouNum",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CHEMINS  [A VERIFIER]  -> doivent matcher l'arbo de ton projet Github
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Logo CESI : fichier a la racine du projet (double extension reelle sur le disque).
LOGO_PATH = os.path.join(BASE_DIR, "cesi_logo.jpg.jpg")

L1_PATH = os.path.join(BASE_DIR, "src", "models", "cnn_scratch.keras")
# Le U-Net du livrable 2 existe en deux formats : unet_model.keras (Keras 3, illisible
# par TF 2.10) et unet_model_tf210.h5 (re-enregistre nativement sous TF 2.10, 256x256).
# On essaie donc le HDF5 en premier, puis le .keras en repli.
L2_PATH_H5 = os.path.join(BASE_DIR, "notebooks", "unet_model_tf210.h5")
L2_PATH_KERAS = os.path.join(BASE_DIR, "notebooks", "unet_model.keras")
L3_PATH = os.path.join(BASE_DIR, "src", "models", "captioning_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "src", "models", "captioning_tokenizer.pkl")
DEMO_DIR = r"C:\Users\matis\OneDrive - Association Cesi Viacesi mail\[3] CESI\A5\Projet Data Science\Projet Github\data\raw\train2014"  # bouton "demo rapide" : images MS COCO

MAX_CAPTION_LEN = 17  # = tokenizer['max_len'] du notebook livrable3 (lu dynamiquement plus bas)


# =============================================================================
# THEME / CSS  --  glass panel centre, bleu nuit + cyan
# Tout le design vit ici. C'est volontairement regroupe pour rester lisible.
# =============================================================================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {
          --bg-deep: #0a1628;
          --bg-mid: #0f1f3a;
          --panel: rgba(15, 28, 51, 0.7);
          --panel-light: rgba(30, 50, 85, 0.5);
          --border: rgba(120, 180, 255, 0.15);
          --border-bright: rgba(120, 200, 255, 0.35);
          --cyan: #4ecdc4;
          --cyan-bright: #5fd9d0;
          --blue: #4a9eff;
          --text: #e8f0fa;
          --muted: #8aa3c4;
          --ok: #2ee6a6;
          --bad: #ff6b8a;
        }

        .stApp {
          background:
            radial-gradient(circle at 20% 10%, rgba(74,158,255,0.15), transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(78,205,196,0.10), transparent 50%),
            radial-gradient(circle at 50% 90%, rgba(120,90,200,0.10), transparent 50%),
            linear-gradient(180deg, #050d1c 0%, #0a1628 100%);
          font-family: 'Inter', system-ui, sans-serif;
          color: var(--text);
        }

        /* Typographie moderne : Inter sur tout le corps et les titres */
        html, body, [class*="css"], .stApp, .stApp *,
        h1, h2, h3, h4, h5, h6, p, div, span, label,
        button, input, textarea, select {
          font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 { font-weight: 700; letter-spacing: -0.01em; }

        /* Sidebar entierement masquee : tout vit dans le glass panel central */
        section[data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }

        /* Master "glass panel" wrapping all content */
        .main .block-container {
          max-width: 1400px;
          padding: 2rem 3rem;
          background: linear-gradient(160deg, rgba(20,40,75,0.85), rgba(15,28,51,0.92));
          border: 1px solid var(--border);
          border-radius: 24px;
          box-shadow:
            0 30px 90px rgba(0,0,0,0.5),
            inset 0 1px 0 rgba(255,255,255,0.05);
          backdrop-filter: blur(20px);
          margin-top: 2rem;
          margin-bottom: 2rem;
        }

        /* Big gradient title */
        .hero-title {
          text-align: center;
          font-size: 2.6rem;
          font-weight: 800;
          letter-spacing: -0.02em;
          margin: 1rem 0 2rem;
          background: linear-gradient(90deg, #ffffff 0%, #6bd9d0 60%, #4a9eff 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .hero-sub {
          text-align: center;
          color: var(--muted);
          font-size: 0.95rem;
          margin-top: -1rem;
          margin-bottom: 1.5rem;
        }

        /* Section labels */
        .section-label {
          font-size: 0.8rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: var(--muted);
          margin: 1.5rem 0 0.8rem;
        }

        /* Status cards — 4 horizontal */
        .status-card {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 1rem 1.2rem;
          transition: all 0.25s;
        }
        .status-card:hover {
          border-color: #00d4aa;
          transform: translateY(-4px);
          box-shadow: 0 10px 34px rgba(0,212,170,0.28),
                      0 0 0 1px rgba(0,212,170,0.4);
        }
        .status-icon {
          width: 36px; height: 36px;
          background: rgba(78,205,196,0.15);
          border-radius: 10px;
          display: flex; align-items: center; justify-content: center;
          font-size: 1.2rem;
          margin-bottom: 0.6rem;
        }
        .status-title {
          font-weight: 700;
          font-size: 0.95rem;
          margin-bottom: 0.25rem;
        }
        .status-meta {
          color: var(--muted);
          font-size: 0.78rem;
          margin-bottom: 0.5rem;
        }
        .badge-ok {
          display: inline-block;
          background: rgba(46,230,166,0.15);
          color: var(--ok);
          border: 1px solid rgba(46,230,166,0.3);
          padding: 2px 10px;
          border-radius: 999px;
          font-size: 0.72rem;
          font-weight: 700;
        }

        /* KPI cards */
        .kpi {
          background: rgba(20,40,75,0.5);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 1.1rem 1.4rem;
          text-align: center;
          transition: all 0.25s;
        }
        .kpi:hover {
          border-color: var(--border-bright);
          box-shadow: 0 0 30px rgba(78,205,196,0.1);
        }
        .kpi-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
        .kpi-label {
          color: var(--muted);
          font-size: 0.72rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          margin-bottom: 0.3rem;
        }
        .kpi-val {
          font-size: 2rem;
          font-weight: 800;
          background: linear-gradient(135deg, #ffffff, #6bd9d0);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        /* Pipeline flow badges */
        .flow-row {
          display: flex; align-items: center; justify-content: center;
          gap: 0.6rem; margin: 1rem 0;
        }
        .flow-step {
          background: var(--panel-light);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 0.6rem 1rem;
          font-size: 0.85rem;
          font-weight: 600;
          display: inline-flex; align-items: center; gap: 0.4rem;
        }
        .flow-arrow {
          color: var(--cyan);
          font-size: 1.2rem;
        }
        /* Etape de pipeline terminee : s'allume en vert avec checkmark */
        .flow-step.done {
          background: rgba(46,230,166,0.15);
          border-color: var(--ok);
          color: var(--ok);
          box-shadow: 0 0 18px rgba(46,230,166,0.25);
          transition: all 0.3s ease;
        }
        .flow-step.active {
          border-color: var(--cyan);
          color: var(--cyan-bright);
          box-shadow: 0 0 18px rgba(78,205,196,0.25);
        }

        /* Animation fade-in (opacity 0 -> 1 sur 0.5s) pour les resultats */
        @keyframes leyenda-fade-in {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        /* Result card */
        .result-card {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 1rem;
          transition: all 0.25s;
          animation: leyenda-fade-in 0.5s ease both;
        }
        .result-caption {
          animation: leyenda-fade-in 0.5s ease both;
        }
        .result-card:hover {
          border-color: var(--border-bright);
          transform: translateY(-3px);
          box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        }
        .result-fname {
          font-weight: 700;
          font-size: 0.9rem;
          margin-bottom: 0.5rem;
          display: flex; align-items: center; justify-content: space-between;
        }
        .result-caption {
          font-style: italic;
          color: var(--text);
          font-size: 0.85rem;
          margin-top: 0.5rem;
          padding: 0.6rem;
          background: rgba(0,0,0,0.2);
          border-left: 3px solid var(--cyan);
          border-radius: 0 8px 8px 0;
        }
        .result-timing {
          color: var(--muted);
          font-size: 0.72rem;
          margin-top: 0.4rem;
          font-family: 'JetBrains Mono', monospace;
        }

        /* Buttons */
        .stButton > button {
          border-radius: 10px;
          font-weight: 600;
          border: 1px solid var(--border);
          background: var(--panel-light);
          color: var(--text);
          transition: all 0.2s;
        }
        .stButton > button:hover {
          border-color: var(--cyan);
          transform: translateY(-1px);
          box-shadow: 0 5px 20px rgba(78,205,196,0.2);
        }
        /* Bouton premium : gradient anime en continu de gauche a droite */
        @keyframes leyenda-gradient-flow {
          0%   { background-position: 0% 50%; }
          50%  { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .stButton > button[kind="primary"] {
          background: linear-gradient(90deg, #00d4aa, #0099ff, #00d4aa);
          background-size: 200% 100%;
          animation: leyenda-gradient-flow 3s ease infinite;
          border: none;
          color: #0a1628;
          font-weight: 700;
          box-shadow: 0 6px 24px rgba(0,153,255,0.25);
        }
        .stButton > button[kind="primary"]:hover {
          transform: translateY(-2px);
          box-shadow: 0 10px 34px rgba(0,212,170,0.4);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
          background: var(--panel-light);
          border-radius: 12px;
          padding: 4px;
          gap: 4px;
          justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
          border-radius: 8px;
          padding: 0.5rem 1.5rem;
          font-weight: 600;
          color: var(--muted);
        }
        .stTabs [aria-selected="true"] {
          background: rgba(78,205,196,0.15) !important;
          color: var(--cyan-bright) !important;
        }

        /* File uploader */
        [data-testid="stFileUploaderDropzone"] {
          background: rgba(20,40,75,0.4);
          border: 2px dashed var(--border-bright);
          border-radius: 14px;
        }

        /* Architecture tab nodes (conserves, reskinnes sur la nouvelle palette) */
        .arch-node {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 1rem 1.3rem;
          margin: 0.6rem 0;
        }
        .arch-node h4 { margin: 0 0 0.3rem; color: var(--cyan-bright); }
        .arch-node p { margin: 0; color: var(--muted); font-size: 0.88rem; }
        .arch-arrow { text-align: center; color: var(--cyan); font-size: 1.3rem; margin: 0.1rem 0; }

        /* Hide streamlit branding */
        #MainMenu, footer, header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# LOGIQUE METIER  --  ne pas modifier le comportement, seulement adapter les
# noms si besoin.  [A VERIFIER] sur chaque fonction.
# =============================================================================
# Chaque modele a son propre cache_resource : ainsi on peut afficher un
# chargement PROGRESSIF (L1 OK -> L2 OK -> L3 OK -> Encodeur) tout en gardant
# la mise en cache (aucun rechargement aux reruns Streamlit).
@st.cache_resource(show_spinner=False)
def load_l1():
    """L1 - classification (peut etre absent).

    compile=False : inference uniquement, evite de fournir les fonctions de
    perte/metriques custom enregistrees dans le modele."""
    try:
        return tf.keras.models.load_model(L1_PATH, compile=False), ("ok", "cnn_scratch.keras")
    except Exception:
        return None, ("warn", "absent - Photo par defaut")


@st.cache_resource(show_spinner=False)
def load_l2():
    """L2 - debruitage (perte custom MSE+SSIM -> compile=False obligatoire).

    Repli : on tente unet_model_tf210.h5 (lisible par TF 2.10) puis unet_model.keras."""
    for path, name in ((L2_PATH_H5, "unet_model_tf210.h5"),
                       (L2_PATH_KERAS, "unet_model.keras")):
        if not os.path.exists(path):
            continue
        try:
            return tf.keras.models.load_model(path, compile=False), ("ok", name)
        except Exception:
            continue
    return None, ("bad", "echec de chargement")


@st.cache_resource(show_spinner=False)
def load_l3():
    """L3 - captioning + tokenizer (charge sur CPU : inference matmul -> cuBLAS absent)."""
    try:
        with tf.device("/CPU:0"):
            model = tf.keras.models.load_model(L3_PATH, compile=False)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer, ("ok", "captioning_model.keras")
    except Exception:
        return None, None, ("bad", "echec de chargement")


@st.cache_resource(show_spinner=False)
def load_encoder():
    """Encodeur InceptionV3 (features pour le captioning) - construit sur CPU (Conv2D)."""
    try:
        with tf.device("/CPU:0"):
            base = tf.keras.applications.InceptionV3(weights="imagenet")
            encoder = tf.keras.Model(base.input, base.layers[-2].output)
        return encoder, ("ok", "InceptionV3 (ImageNet)")
    except Exception:
        return None, ("bad", "echec")


def get_models():
    """Assemble les 4 modeles en affichant un chargement progressif.

    Compatible avec les anciennes versions de Streamlit : st.status() (1.23+) n'est
    PAS utilise. On s'appuie sur un unique st.empty() dont on met a jour le texte via
    .info() a chaque etape (L1 OK -> L2 OK -> L3 OK -> Encodeur), puis qu'on efface
    (.empty()) une fois tout charge. Aux reruns, les loaders sont en cache -> instantane."""
    models, status = {}, {}
    box = st.empty()        # placeholder unique pour le message de statut
    lines = []              # historique des etapes deja terminees

    def show(current):
        """Reaffiche l'historique + l'etape en cours dans le placeholder unique."""
        body = "  \n".join(lines + ([current] if current else []))
        box.info("⏳ **Chargement des modèles…**  \n" + body)

    def done(key, label):
        state, name = status[key]
        mark = "✅" if state == "ok" else ("⚠️" if state == "warn" else "❌")
        lines.append(f"{mark}  {label} — {name}")

    show("⏳ Initialisation de TensorFlow…")

    show("⏳ L1 · Classification…")
    models["l1"], status["l1"] = load_l1()
    done("l1", "L1 · Classification")

    show("⏳ L2 · Débruitage…")
    models["l2"], status["l2"] = load_l2()
    done("l2", "L2 · Débruitage")

    show("⏳ L3 · Captioning…")
    models["l3"], models["tokenizer"], status["l3"] = load_l3()
    done("l3", "L3 · Captioning")

    show("⏳ Encodeur InceptionV3…")
    models["encoder"], status["encoder"] = load_encoder()
    done("encoder", "Encodeur InceptionV3")

    box.empty()             # efface le placeholder une fois tous les modeles charges
    return models, status


def classify_image(models, img_array):
    """L1 : renvoie ('Photo'|'Non-photo', confiance). Tolere l'absence du modele."""
    if models["l1"] is None:
        return "Photo", None  # comportement par defaut demande
    x = tf.image.resize(img_array, (224, 224))[None] / 255.0  # entree L1 = 224x224 (cnn_scratch)
    # CNN convolutif force sur CPU : Conv2D exige cuDNN sur GPU, indisponible ici
    # ("DNN library is not found"). Le CPU evite cette dependance.
    with tf.device("/CPU:0"):
        p = float(models["l1"].predict(x, verbose=0)[0][0])
    THRESHOLD = 0.5
    label = "Photo" if p >= THRESHOLD else "Non-photo"
    conf = p if p >= THRESHOLD else 1 - p
    return label, conf


def denoise_image(models, img_array):
    """L2 : renvoie l'image debruitee en uint8.

    Pretraitement identique au livrable 2 : redimensionnement 256x256, division par
    255 en entree, clip dans [0,1] puis multiplication par 255 en sortie. La prediction
    du U-Net est forcee sur CPU pour eviter un conflit memoire GPU (RESOURCE_EXHAUSTED)
    avec InceptionV3 et le modele de captioning deja charges. Si L2 est absent, on
    renvoie l'image originale inchangee.
    """
    if models["l2"] is None:
        return np.clip(img_array, 0, 255).astype("uint8")  # image originale inchangee
    arr = tf.image.resize(img_array, (256, 256)) / 255.0   # entree U-Net : 256x256, [0,1]
    with tf.device("/CPU:0"):                              # evite l'OOM GPU
        out = models["l2"].predict(arr[None, ...], verbose=0)[0]
    out = np.clip(out, 0, 1)                                # bornage dans [0,1]
    return (out * 255).astype("uint8")


def generate_caption(models, img_array):
    """L3 : decodage glouton, identique a generate_caption du notebook livrable3.

    Le tokenizer est un dict {word2idx, idx2word, max_len, vocab_size} avec tokens
    speciaux <start>/<end>/<pad>. On predit le mot le plus probable a chaque pas
    jusqu'a <end>, padding 'post', longueur fixe max_len.
    """
    if models["l3"] is None or models["encoder"] is None or models["tokenizer"] is None:
        return "(captioning indisponible)"
    tok = models["tokenizer"]
    word2idx, idx2word, max_len = tok["word2idx"], tok["idx2word"], tok["max_len"]
    START, END, PAD = "<start>", "<end>", "<pad>"

    # features InceptionV3 (299x299, preprocess officiel -> [-1, 1]) -> vecteur 2048-d
    # Tout est force sur CPU : InceptionV3 (Conv2D -> cuDNN) ET le decodeur L3 (matmul
    # Dense -> cuBLAS) echouent sur ce GPU faute de bibliotheques BLAS/cuDNN
    # ("BLAS operation ... without BLAS support"). Le CPU evite ces dependances.
    x = tf.image.resize(img_array, (299, 299))
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    with tf.device("/CPU:0"):
        feat = models["encoder"].predict(x[None], verbose=0)  # (1, 2048)

    seq = [word2idx[START]]
    for _ in range(max_len):
        pad = tf.keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_len, padding="post"
        )
        with tf.device("/CPU:0"):                            # decodeur L3 sur CPU
            yhat = models["l3"].predict([feat, pad], verbose=0)
        nxt = int(np.argmax(yhat[0]))
        if nxt == word2idx[END]:
            break
        seq.append(nxt)

    words = [idx2word[i] for i in seq[1:]
             if i not in (word2idx[START], word2idx[END], word2idx[PAD])]
    cap = " ".join(words)
    return cap.strip().capitalize() or "(legende vide)"


def run_pipeline(models, pil_img, step_cb=None):
    """Enchaine L1 -> L2 -> L3 sur une image. Robuste : chaque etape en try/except.

    step_cb(stage) est appele apres chaque etape ('l1', 'l2', 'l3') pour permettre
    a l'UI d'allumer progressivement la barre de progression du pipeline."""
    img = np.array(pil_img.convert("RGB")).astype("float32")
    res = {"label": "?", "conf": None, "denoised": None,
           "caption": "", "t_l1": 0, "t_l2": 0, "t_l3": 0, "error": None}
    try:
        t = time.time(); res["label"], res["conf"] = classify_image(models, img); res["t_l1"] = (time.time()-t)*1000
        if step_cb: step_cb("l1")
        t = time.time(); res["denoised"] = denoise_image(models, img); res["t_l2"] = (time.time()-t)*1000
        if step_cb: step_cb("l2")
        t = time.time(); res["caption"] = generate_caption(models, img); res["t_l3"] = (time.time()-t)*1000
        if step_cb: step_cb("l3")
    except Exception as e:
        res["error"] = str(e)
    return res


def build_csv(results, names):
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["fichier", "classe", "confiance", "legende", "t_l1_ms", "t_l2_ms", "t_l3_ms"])
    for n, r in zip(names, results):
        w.writerow([n, r["label"], f"{(r['conf'] or 0):.3f}", r["caption"],
                    f"{r['t_l1']:.0f}", f"{r['t_l2']:.0f}", f"{r['t_l3']:.0f}"])
    return buf.getvalue().encode("utf-8")


# =============================================================================
# RENDU UI  --  helpers de presentation (n'affectent pas la logique metier)
# =============================================================================
def render_pipeline_steps(box, done, active=None):
    """Affiche la barre L1 -> L2 -> L3 ; chaque etape de `done` s'allume en vert
    avec une checkmark, l'etape `active` est mise en surbrillance cyan."""
    steps = [("l1", "🔍 L1 Classification"),
             ("l2", "🧹 L2 Débruitage"),
             ("l3", "💬 L3 Captioning")]
    chips = []
    for key, label in steps:
        if key in done:
            chips.append(f'<div class="flow-step done">{label} ✓</div>')
        elif key == active:
            chips.append(f'<div class="flow-step active">{label} …</div>')
        else:
            chips.append(f'<div class="flow-step">{label}</div>')
    box.markdown(
        f'<div class="flow-row">{chips[0]}<span class="flow-arrow">→</span>'
        f'{chips[1]}<span class="flow-arrow">→</span>{chips[2]}</div>',
        unsafe_allow_html=True,
    )


def render_kpi_row(results):
    """5 cartes KPI horizontales (utilisees dans l'onglet Pipeline ET Statistiques)."""
    k1, k2, k3, k4, k5 = st.columns(5)
    photos = sum(1 for r in results if r["label"] == "Photo")
    non = sum(1 for r in results if r["label"] == "Non-photo")
    avg = np.mean([r["t_l1"] + r["t_l2"] + r["t_l3"] for r in results]) if results else 0
    tot = sum(r["t_l1"] + r["t_l2"] + r["t_l3"] for r in results) / 1000
    for col, icon, label, val in [
        (k1, '🖼️', 'Images', len(results)),
        (k2, '📷', 'Photos', photos),
        (k3, '🚫', 'Non-photos', non),
        (k4, '⚡', 'Moy. / image', f'{avg:.0f} ms'),
        (k5, '⏱️', 'Temps total', f'{tot:.1f} s')]:
        col.markdown(f"""
            <div class="kpi">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-val">{val}</div>
            </div>""", unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    inject_css()

    # === HEADER: titre (rendu AVANT le chargement des modeles) ===
    st.markdown('<div class="hero-title">Projet Leyenda · Groupe 2</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Pipeline TouNum — classification · '
                'débruitage · captioning</div>', unsafe_allow_html=True)

    # Chargement progressif des modeles, visible des le demarrage (spinner +
    # statut etape par etape). En cache -> instantane aux reruns suivants.
    models, status = get_models()

    # init session state
    if "results" not in st.session_state:
        st.session_state.results, st.session_state.names, st.session_state.images = [], [], []

    # === STATUT DES MODELES — 4 cards horizontal ===
    st.markdown('<div class="section-label">Statut des modèles</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, key, title, icon, fname in [
        (c1, 'l1', 'L1 · Classification', '🔍', 'cnn_scratch.keras'),
        (c2, 'l2', 'L2 · Débruitage', '🧹', 'unet_model_tf210.h5'),
        (c3, 'l3', 'L3 · Captioning', '💬', 'captioning_model.keras'),
        (c4, 'encoder', 'Encodeur', '⚙️', 'InceptionV3 (ImageNet)')]:
        state = status.get(key, ('bad', ''))[0]
        col.markdown(f"""
            <div class="status-card">
                <div class="status-icon">{icon}</div>
                <div class="status-title">{title}</div>
                <div class="status-meta">{fname}</div>
                <span class="badge-ok">{'✓ OK' if state == 'ok' else '⚠'}</span>
            </div>""", unsafe_allow_html=True)

    # === ENV + PARAMS + FLOW (3 columns) ===
    e1, e2, e3 = st.columns([1, 1, 2])
    with e1:
        gpu = "✓" if tf.config.list_physical_devices("GPU") else "—"
        st.markdown(f"""
            <div class="status-card">
                <div class="section-label" style="margin:0 0 0.5rem">Environnement</div>
                <div>TensorFlow {tf.__version__} · GPU {gpu}</div>
            </div>""", unsafe_allow_html=True)
    with e2:
        st.markdown('<div class="section-label" style="margin:0 0 0.5rem">Paramètres</div>',
                    unsafe_allow_html=True)
        n_max = st.slider("Nombre d'images", 1, 50, 4, label_visibility="collapsed")
    with e3:
        st.markdown("""
            <div class="flow-row">
                <div class="flow-step">🖼️ Image</div>
                <span class="flow-arrow">→</span>
                <div class="flow-step">🔍 L1</div>
                <span class="flow-arrow">→</span>
                <div class="flow-step">🧹 L2</div>
                <span class="flow-arrow">→</span>
                <div class="flow-step">💬 L3</div>
            </div>""", unsafe_allow_html=True)

    # === TABS ===
    tab_pipe, tab_stats, tab_arch = st.tabs(["🔬 Pipeline", "📊 Statistiques", "🏗️ Architecture"])

    # ----------------------------- PIPELINE -----------------------------
    with tab_pipe:
        # Description + upload
        st.markdown("""
            <div style="color: var(--muted); text-align: center;
                        font-style: italic; margin: 1rem 0;">
            "Numérisation intelligente : chaque image traverse une chaîne IA
            en trois temps — tri photo/non-photo, restauration par débruitage,
            puis génération automatique d'une légende descriptive."
            </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Sélection des images",
            type=["jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp"],
            accept_multiple_files=True,
        )
        folder = st.text_input("Ou dossier d'images:",
                               placeholder=r"C:\Users\matis\Desktop\dataset_prof")

        # buttons row
        b1, b2, b3 = st.columns([2, 1, 1])
        launch = b1.button("🚀 Lancer le pipeline", type="primary",
                           use_container_width=True)
        demo = b2.button("📂 Démo rapide", use_container_width=True)
        reset = b3.button("🗑️ Réinitialiser", use_container_width=True)

        if reset:
            st.session_state.results = []
            st.session_state.names = []
            st.session_state.images = []

        # --- collecte des sources (logique existante, inchangee) ---
        sources = []  # (nom, PIL.Image)
        if uploaded:
            for f in uploaded[:n_max]:
                try:
                    sources.append((f.name, Image.open(f)))
                except Exception:
                    pass
        if (demo or folder) and not uploaded:
            target = folder.strip() or DEMO_DIR
            if os.path.isdir(target):
                files = [x for x in sorted(os.listdir(target))
                         if x.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))]
                for fn in files[:n_max]:
                    try:
                        sources.append((fn, Image.open(os.path.join(target, fn))))
                    except Exception:
                        pass
            else:
                st.warning(f"Dossier introuvable : {target}")

        if sources:
            st.success(f"{len(sources)} image(s) prête(s) à être traitée(s).")

        # --- lancement (logique existante + barre d'etapes progressive) ---
        if (launch or demo) and sources:
            prog = st.progress(0, text="Initialisation du pipeline…")
            step_box = st.empty()  # barre L1 -> L2 -> L3 mise a jour en direct
            results, names, images = [], [], []
            for i, (name, img) in enumerate(sources):
                prog.progress((i) / len(sources), text=f"Traitement de {name} ({i+1}/{len(sources)})")
                done = set()
                render_pipeline_steps(step_box, done, active="l1")

                # callback : a la fin de chaque etape on l'allume en vert et on
                # passe la suivante en surbrillance (defauts -> capture par valeur).
                def step_cb(stage, _box=step_box, _done=done):
                    _done.add(stage)
                    nxt = {"l1": "l2", "l2": "l3", "l3": None}[stage]
                    render_pipeline_steps(_box, _done, active=nxt)

                results.append(run_pipeline(models, img, step_cb=step_cb))
                names.append(name)
                images.append(img)
            prog.progress(1.0, text="Terminé ✅")
            time.sleep(0.3); prog.empty(); step_box.empty()
            st.session_state.results, st.session_state.names, st.session_state.images = results, names, images

        # === KPI ROW + RESULTS GRID ===
        if st.session_state.results:
            st.markdown('<div class="section-label">Synthèse</div>',
                        unsafe_allow_html=True)
            render_kpi_row(st.session_state.results)

            # === RESULTS GRID — 4 columns ===
            st.markdown(f'<div class="section-label">Résultats ({len(st.session_state.results)})</div>',
                        unsafe_allow_html=True)
            results = list(zip(st.session_state.names,
                               st.session_state.images,
                               st.session_state.results))
            # render 4 per row
            for i in range(0, len(results), 4):
                cols = st.columns(4)
                for col, (name, img, res) in zip(cols, results[i:i+4]):
                    with col:
                        badge = ('<span class="badge-ok">✓ Photo</span>'
                                 if res["label"] == "Photo"
                                 else '<span style="background:rgba(255,107,138,0.15);color:#ff6b8a;'
                                      'border:1px solid rgba(255,107,138,0.3);padding:2px 10px;'
                                      'border-radius:999px;font-size:0.72rem;font-weight:700;">Non-photo</span>')
                        st.markdown(f"""
                            <div class="result-card">
                                <div class="result-fname">
                                    {name[:25]}{'...' if len(name) > 25 else ''}
                                    {badge}
                                </div>""", unsafe_allow_html=True)
                        st.image(img, use_column_width=True)
                        st.markdown(f"""
                                <div class="result-caption">« {res['caption']} »</div>
                                <div class="result-timing">
                                    L1 {res['t_l1']:.0f}ms · L2 {res['t_l2']:.0f}ms · L3 {res['t_l3']:.0f}ms
                                </div>
                            </div>""", unsafe_allow_html=True)

            # CSV download
            st.download_button(
                "⬇️ Télécharger les résultats (CSV)",
                data=build_csv(st.session_state.results, st.session_state.names),
                file_name="resultats_leyenda.csv",
                mime="text/csv",
            )
        else:
            st.info("Importez des images puis cliquez sur **Lancer le pipeline**.")

    # ----------------------------- STATISTIQUES -----------------------------
    with tab_stats:
        st.markdown('<div class="section-label">Statistiques de la session</div>',
                    unsafe_allow_html=True)
        res = st.session_state.results
        if not res:
            st.info("Aucune donnée pour l'instant — lancez d'abord le pipeline.")
        else:
            render_kpi_row(res)
            st.markdown("##### Répartition des temps par étape (moyenne)")
            avg_l1 = np.mean([r["t_l1"] for r in res])
            avg_l2 = np.mean([r["t_l2"] for r in res])
            avg_l3 = np.mean([r["t_l3"] for r in res])
            st.bar_chart({"L1 Classification": [avg_l1],
                          "L2 Débruitage": [avg_l2],
                          "L3 Captioning": [avg_l3]})
            st.caption("Temps moyen de traitement par étape, en millisecondes.")

    # ----------------------------- ARCHITECTURE -----------------------------
    with tab_arch:
        st.markdown('<div class="section-label">Architecture du pipeline</div>',
                    unsafe_allow_html=True)
        st.markdown(
            """
            <div class="arch-node"><h4>① Image d'entrée</h4>
                <p>Chargement RGB, normalisation unique [0,1]. Compatible JPG/PNG/TIFF.</p></div>
            <div class="arch-arrow">▼</div>
            <div class="arch-node"><h4>② L1 · Classification (CNN)</h4>
                <p>CNN entraîné from scratch. Tri binaire photo / non-photo (schémas, textes, peintures).</p></div>
            <div class="arch-arrow">▼</div>
            <div class="arch-node"><h4>③ L2 · Débruitage (auto-encodeur U-Net)</h4>
                <p>Auto-encodeur convolutif type U-Net. Restaure les images bruitées avant analyse.</p></div>
            <div class="arch-arrow">▼</div>
            <div class="arch-node"><h4>④ L3 · Captioning (CNN + RNN)</h4>
                <p>Encodeur InceptionV3 (transfer learning, ImageNet) → features. Décodeur RNN entraîné sur
                MS COCO génère la légende mot à mot (décodage glouton).</p></div>
            <div class="arch-arrow">▼</div>
            <div class="arch-node"><h4>⑤ Légende descriptive</h4>
                <p>Sortie textuelle finale, exportable en CSV avec les temps de traitement.</p></div>
            """,
            unsafe_allow_html=True,
        )

    # === FOOTER ===
    st.markdown(
        """
        <div style="text-align:center; color:var(--muted); font-size:0.75rem;
                    opacity:0.65; margin-top:2.5rem; padding-top:1rem;
                    border-top:1px solid var(--border);">
            Projet Leyenda · Groupe 2 · CESI A5 · TensorFlow 2.10 · MS COCO 2014
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
