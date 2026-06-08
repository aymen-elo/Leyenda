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
    initial_sidebar_state="expanded",
)

# =============================================================================
# CHEMINS  [A VERIFIER]  -> doivent matcher l'arbo de ton projet Github
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

L1_PATH = os.path.join(BASE_DIR, "src", "models", "cnn_scratch.keras")
# notebooks/unet_model.keras est au format Keras 3 (illisible par TF 2.10) ->
# on pointe sur le U-Net re-enregistre nativement sous TF 2.10 (HDF5, 256x256).
L2_PATH = os.path.join(BASE_DIR, "src", "models", "unet_denoiser_tf210.keras")
L3_PATH = os.path.join(BASE_DIR, "src", "models", "captioning_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "src", "models", "captioning_tokenizer.pkl")
DEMO_DIR = os.path.join(BASE_DIR, "Dataset")  # bouton "demo rapide"

MAX_CAPTION_LEN = 17  # = tokenizer['max_len'] du notebook livrable3 (lu dynamiquement plus bas)


# =============================================================================
# THEME / CSS  --  bleu nuit + cyan glace
# Tout le design vit ici. C'est volontairement regroupe pour rester lisible.
# =============================================================================
def inject_css():
    st.markdown(
        """
        <style>
        /* ---- palette ---- */
        :root {
            --bg-0:#0a0e1a; --bg-1:#0f1626; --bg-2:#161f33;
            --line:#22304d; --cyan:#38e1ff; --cyan-soft:#7df0ff;
            --txt:#e6edf7; --muted:#8aa0c2; --ok:#2ee6a6; --bad:#ff6b8a;
            --amber:#ffb454;
        }

        /* fond global avec un leger halo cyan */
        .stApp {
            background:
                radial-gradient(900px 500px at 12% -5%, rgba(56,225,255,.10), transparent 60%),
                radial-gradient(800px 500px at 100% 0%, rgba(125,90,255,.10), transparent 55%),
                var(--bg-0);
            color: var(--txt);
        }
        .block-container { padding-top: 1.4rem; max-width: 1320px; }

        /* ---------------- HERO ---------------- */
        .hero {
            border:1px solid var(--line);
            border-radius:20px;
            padding:26px 30px;
            background:
                linear-gradient(135deg, rgba(56,225,255,.08), rgba(125,90,255,.05)),
                var(--bg-1);
            box-shadow: 0 18px 50px rgba(0,0,0,.45);
            position:relative; overflow:hidden;
        }
        .hero:before{
            content:""; position:absolute; inset:0;
            background: linear-gradient(90deg, transparent, rgba(56,225,255,.06), transparent);
        }
        .hero h1{
            margin:0; font-size:2.05rem; font-weight:800; letter-spacing:.3px;
            background:linear-gradient(90deg,#fff,var(--cyan-soft));
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        }
        .hero p{ margin:.5rem 0 0; color:var(--muted); font-size:1.02rem; max-width:760px; }

        /* ---------------- PIPELINE BADGES ---------------- */
        .flow{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:18px; }
        .step{
            display:flex; align-items:center; gap:8px;
            border:1px solid var(--line); border-radius:12px;
            padding:9px 14px; background:var(--bg-2);
            font-weight:600; font-size:.92rem;
            transition:.2s; white-space:nowrap;
        }
        .step:hover{ border-color:var(--cyan); box-shadow:0 0 0 1px rgba(56,225,255,.3); }
        .step .dot{ width:9px; height:9px; border-radius:50%; background:var(--cyan); box-shadow:0 0 10px var(--cyan); }
        .arrow{ color:var(--muted); font-size:1.1rem; }

        /* ---------------- CARTES KPI ---------------- */
        .kpi-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:6px 0 4px; }
        .kpi{
            border:1px solid var(--line); border-radius:16px; padding:16px 18px;
            background:linear-gradient(160deg,var(--bg-2),var(--bg-1));
        }
        .kpi .lab{ color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.6px; }
        .kpi .val{ font-size:1.7rem; font-weight:800; margin-top:4px; }
        .kpi .val.cyan{ color:var(--cyan); } .kpi .val.ok{ color:var(--ok); } .kpi .val.bad{ color:var(--bad); }

        /* ---------------- CARTE RESULTAT ---------------- */
        .result-card{
            border:1px solid var(--line); border-radius:18px;
            background:var(--bg-1); padding:18px 20px; margin-bottom:16px;
            box-shadow:0 10px 30px rgba(0,0,0,.35);
        }
        .result-head{ display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; }
        .fname{ font-weight:700; font-size:1.05rem; }
        .caption-box{
            border-left:3px solid var(--cyan); background:var(--bg-2);
            border-radius:0 10px 10px 0; padding:12px 16px; margin-top:6px;
            font-size:1.05rem; font-style:italic; color:var(--txt);
        }
        .ts{ color:var(--muted); font-size:.82rem; margin-top:8px; }

        /* badges statut */
        .badge{ display:inline-flex; align-items:center; gap:6px; padding:5px 12px;
                border-radius:999px; font-size:.82rem; font-weight:700; }
        .badge.ok{ background:rgba(46,230,166,.14); color:var(--ok); border:1px solid rgba(46,230,166,.4); }
        .badge.bad{ background:rgba(255,107,138,.14); color:var(--bad); border:1px solid rgba(255,107,138,.4); }
        .badge.warn{ background:rgba(255,180,84,.14); color:var(--amber); border:1px solid rgba(255,180,84,.4); }

        /* ---------------- SIDEBAR ---------------- */
        section[data-testid="stSidebar"]{ background:var(--bg-1); border-right:1px solid var(--line); }
        .side-card{ border:1px solid var(--line); border-radius:14px; background:var(--bg-2); padding:14px 16px; margin-bottom:14px; }
        .side-title{ font-size:.75rem; text-transform:uppercase; letter-spacing:.8px; color:var(--muted); margin-bottom:10px; }
        .model-row{ display:flex; align-items:center; justify-content:space-between; padding:7px 0; border-bottom:1px solid rgba(34,48,77,.6); }
        .model-row:last-child{ border-bottom:none; }
        .model-name{ font-weight:600; font-size:.92rem; }
        .model-file{ color:var(--muted); font-size:.72rem; }
        .pill{ font-size:.7rem; font-weight:700; padding:3px 9px; border-radius:999px; }
        .pill.ok{ background:rgba(46,230,166,.15); color:var(--ok); }
        .pill.bad{ background:rgba(255,107,138,.15); color:var(--bad); }
        .pill.warn{ background:rgba(255,180,84,.15); color:var(--amber); }

        /* boutons */
        .stButton>button{
            border-radius:12px; font-weight:700; border:1px solid var(--line);
            padding:.55rem 1rem; transition:.18s;
        }
        .stButton>button[kind="primary"]{
            background:linear-gradient(90deg,var(--cyan),#3a8bff); color:#06121f; border:none;
            box-shadow:0 6px 20px rgba(56,225,255,.35);
        }
        .stButton>button[kind="primary"]:hover{ filter:brightness(1.08); transform:translateY(-1px); }

        /* tabs */
        .stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--line); }
        .stTabs [data-baseweb="tab"]{
            border-radius:10px 10px 0 0; padding:10px 18px; color:var(--muted); font-weight:600;
        }
        .stTabs [aria-selected="true"]{ color:var(--cyan); background:var(--bg-2); }

        /* uploader */
        [data-testid="stFileUploaderDropzone"]{
            border:1.5px dashed var(--line); border-radius:16px; background:var(--bg-2);
        }
        [data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--cyan); }

        /* section title */
        .sec{ font-size:1.25rem; font-weight:800; margin:22px 0 12px; display:flex; align-items:center; gap:10px; }
        .sec .bar{ width:4px; height:22px; border-radius:4px; background:linear-gradient(var(--cyan),#3a8bff); }

        /* arch flow vertical */
        .arch-node{ border:1px solid var(--line); border-radius:14px; background:var(--bg-2);
                    padding:14px 18px; margin:10px 0; }
        .arch-node h4{ margin:0 0 4px; color:var(--cyan); }
        .arch-node p{ margin:0; color:var(--muted); font-size:.9rem; }
        .arch-arrow{ text-align:center; color:var(--cyan); font-size:1.3rem; margin:-2px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# LOGIQUE METIER  --  ne pas modifier le comportement, seulement adapter les
# noms si besoin.  [A VERIFIER] sur chaque fonction.
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Charge les 3 modeles une seule fois. Renvoie un dict + statut."""
    status = {}
    models = {}

    # L1 - classification (peut etre absent)
    # compile=False : inference uniquement, et evite d'avoir a fournir les
    # fonctions de perte/metriques custom enregistrees dans les modeles.
    try:
        models["l1"] = tf.keras.models.load_model(L1_PATH, compile=False)
        status["l1"] = ("ok", "cnn_scratch.keras")
    except Exception:
        models["l1"] = None
        status["l1"] = ("warn", "absent - Photo par defaut")

    # L2 - debruitage (perte custom 'combined_loss_weighted' -> compile=False obligatoire)
    try:
        models["l2"] = tf.keras.models.load_model(L2_PATH, compile=False)
        status["l2"] = ("ok", "unet_denoiser_tf210.keras")
    except Exception:
        models["l2"] = None
        status["l2"] = ("bad", "echec de chargement")

    # L3 - captioning + tokenizer
    try:
        models["l3"] = tf.keras.models.load_model(L3_PATH, compile=False)
        with open(TOKENIZER_PATH, "rb") as f:
            models["tokenizer"] = pickle.load(f)
        status["l3"] = ("ok", "captioning_model.keras")
    except Exception:
        models["l3"] = None
        models["tokenizer"] = None
        status["l3"] = ("bad", "echec de chargement")

    # Encodeur InceptionV3 (features pour le captioning)
    try:
        base = tf.keras.applications.InceptionV3(weights="imagenet")
        models["encoder"] = tf.keras.Model(base.input, base.layers[-2].output)
        status["encoder"] = ("ok", "InceptionV3 (ImageNet)")
    except Exception:
        models["encoder"] = None
        status["encoder"] = ("bad", "echec")

    return models, status


def classify_image(models, img_array):
    """L1 : renvoie ('Photo'|'Non-photo', confiance). Tolere l'absence du modele."""
    if models["l1"] is None:
        return "Photo", None  # comportement par defaut demande
    x = tf.image.resize(img_array, (224, 224))[None] / 255.0  # entree L1 = 224x224 (cnn_scratch)
    p = float(models["l1"].predict(x, verbose=0)[0][0])
    label = "Photo" if p >= 0.5 else "Non-photo"
    conf = p if p >= 0.5 else 1 - p
    return label, conf


def denoise_image(models, img_array):
    """L2 : renvoie l'image debruitee en uint8, ou None si L2 absent."""
    if models["l2"] is None:
        return None
    h, w = 256, 256  # entree/sortie U-Net = 256x256 (unet_denoiser_tf210)
    x = tf.image.resize(img_array, (h, w)) / 255.0
    out = models["l2"].predict(x[None], verbose=0)[0]
    out = np.clip(out * 255.0, 0, 255).astype("uint8")
    return out


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
    x = tf.image.resize(img_array, (299, 299))
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    feat = models["encoder"].predict(x[None], verbose=0)  # (1, 2048)

    seq = [word2idx[START]]
    for _ in range(max_len):
        pad = tf.keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_len, padding="post"
        )
        yhat = models["l3"].predict([feat, pad], verbose=0)
        nxt = int(np.argmax(yhat[0]))
        if nxt == word2idx[END]:
            break
        seq.append(nxt)

    words = [idx2word[i] for i in seq[1:]
             if i not in (word2idx[START], word2idx[END], word2idx[PAD])]
    cap = " ".join(words)
    return cap.strip().capitalize() or "(legende vide)"


def run_pipeline(models, pil_img):
    """Enchaine L1 -> L2 -> L3 sur une image. Robuste : chaque etape en try/except."""
    img = np.array(pil_img.convert("RGB")).astype("float32")
    res = {"label": "?", "conf": None, "denoised": None,
           "caption": "", "t_l1": 0, "t_l2": 0, "t_l3": 0, "error": None}
    try:
        t = time.time(); res["label"], res["conf"] = classify_image(models, img); res["t_l1"] = (time.time()-t)*1000
        t = time.time(); res["denoised"] = denoise_image(models, img); res["t_l2"] = (time.time()-t)*1000
        t = time.time(); res["caption"] = generate_caption(models, img); res["t_l3"] = (time.time()-t)*1000
    except Exception as e:
        res["error"] = str(e)
    return res


# =============================================================================
# RENDU UI
# =============================================================================
def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>🛰️ Projet Leyenda — Pipeline TouNum</h1>
            <p>Numérisation intelligente : chaque image traverse une chaîne IA en
            trois temps — tri photo/non-photo, restauration par débruitage, puis
            génération automatique d'une légende descriptive.</p>
            <div class="flow">
                <div class="step"><span class="dot"></span>🖼️ Image</div>
                <span class="arrow">→</span>
                <div class="step"><span class="dot"></span>🔍 L1 · Classification</div>
                <span class="arrow">→</span>
                <div class="step"><span class="dot"></span>🧹 L2 · Débruitage</div>
                <span class="arrow">→</span>
                <div class="step"><span class="dot"></span>💬 L3 · Légende</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(status, models):
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding:6px 0 14px;">
                <div style="font-size:2.4rem;">🏛️</div>
                <div style="font-weight:800; letter-spacing:.5px;">CESI · Groupe 2</div>
                <div style="color:#8aa0c2; font-size:.82rem;">Projet TouNum · Leyenda</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # statut modeles
        rows = ""
        meta = {"l1": "L1 · Classification", "l2": "L2 · Débruitage",
                "l3": "L3 · Captioning", "encoder": "Encodeur"}
        for key, title in meta.items():
            state, label = status.get(key, ("bad", "?"))
            pill_txt = {"ok": "OK", "warn": "DÉFAUT", "bad": "ABSENT"}[state]
            rows += f"""
            <div class="model-row">
                <div>
                    <div class="model-name">{title}</div>
                    <div class="model-file">{label}</div>
                </div>
                <span class="pill {state}">{pill_txt}</span>
            </div>"""
        st.markdown(
            f'<div class="side-card"><div class="side-title">⚙️ Statut des modèles</div>{rows}</div>',
            unsafe_allow_html=True,
        )

        # backend
        gpu = "✅" if tf.config.list_physical_devices("GPU") else "—"
        st.markdown(
            f'<div class="side-card"><div class="side-title">🧩 Environnement</div>'
            f'<div class="model-row"><span class="model-name">TensorFlow</span>'
            f'<span class="pill ok">{tf.__version__}</span></div>'
            f'<div class="model-row"><span class="model-name">GPU</span>'
            f'<span class="pill {"ok" if gpu=="✅" else "warn"}">{gpu}</span></div></div>',
            unsafe_allow_html=True,
        )

        # parametres
        st.markdown('<div class="side-title">🎚️ Paramètres</div>', unsafe_allow_html=True)
        n = st.slider("Nombre d'images à traiter", 1, 50, 4)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        launch = st.button("🚀 Lancer le pipeline", use_container_width=True, type="primary")
        demo = st.button("📂 Démo rapide (dossier Dataset/)", use_container_width=True)
        reset = st.button("🗑️ Réinitialiser", use_container_width=True)

        return n, launch, demo, reset


def render_kpis(results):
    photos = sum(1 for r in results if r["label"] == "Photo")
    non = sum(1 for r in results if r["label"] == "Non-photo")
    avg = np.mean([r["t_l1"] + r["t_l2"] + r["t_l3"] for r in results]) if results else 0
    total = sum(r["t_l1"] + r["t_l2"] + r["t_l3"] for r in results) / 1000
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi"><div class="lab">Images</div><div class="val cyan">{len(results)}</div></div>
            <div class="kpi"><div class="lab">Photos</div><div class="val ok">{photos}</div></div>
            <div class="kpi"><div class="lab">Non-photos</div><div class="val bad">{non}</div></div>
            <div class="kpi"><div class="lab">Moy. / image</div><div class="val">{avg:.0f} ms</div></div>
            <div class="kpi"><div class="lab">Temps total</div><div class="val">{total:.1f} s</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(name, pil_img, res):
    badge = ('<span class="badge bad">🚫 Non-photo</span>'
             if res["label"] == "Non-photo"
             else '<span class="badge ok">✅ Photo</span>')
    conf = f" · {res['conf']*100:.0f}%" if res["conf"] is not None else ""

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-head"><div class="fname">🖼️ {name}</div>{badge}</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.image(pil_img, caption="Originale", use_container_width=True)
    with c2:
        if res["denoised"] is not None:
            st.image(res["denoised"], caption="Débruitée (L2)", use_container_width=True)
        else:
            st.info("L2 désactivé pour cette image.")

    st.markdown(f'<div class="caption-box">💬 « {res["caption"]} »{conf}</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="ts">⏱️ L1 {res["t_l1"]:.0f} ms · L2 {res["t_l2"]:.0f} ms · '
        f'L3 {res["t_l3"]:.0f} ms</div>',
        unsafe_allow_html=True,
    )
    if res["error"]:
        st.error(f"Erreur : {res['error']}")
    st.markdown("</div>", unsafe_allow_html=True)


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
# MAIN
# =============================================================================
def main():
    inject_css()

    if "results" not in st.session_state:
        st.session_state.results = []
        st.session_state.names = []
        st.session_state.images = []

    models, status = load_models()
    n_max, launch, demo, reset = render_sidebar(status, models)

    if reset:
        st.session_state.results, st.session_state.names, st.session_state.images = [], [], []

    render_hero()

    tab_pipe, tab_stats, tab_arch = st.tabs(["🔬 Pipeline", "📊 Statistiques", "🏗️ Architecture"])

    # ----------------------------- PIPELINE -----------------------------
    with tab_pipe:
        st.markdown('<div class="sec"><span class="bar"></span>📥 Sélection des images</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Glissez-déposez vos images (sélection multiple)",
            type=["jpg", "jpeg", "png", "bmp", "gif", "tif", "tiff", "webp"],
            accept_multiple_files=True,
        )
        folder = st.text_input(
            "…ou indiquez le chemin d'un dossier d'images (sur cette machine)",
            placeholder=r"C:\Users\matis\Desktop\dataset_prof",
        )

        # collecte des sources
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

        # lancement
        if (launch or demo) and sources:
            prog = st.progress(0, text="Initialisation du pipeline…")
            results, names, images = [], [], []
            for i, (name, img) in enumerate(sources):
                prog.progress((i) / len(sources), text=f"Traitement de {name} ({i+1}/{len(sources)})")
                results.append(run_pipeline(models, img))
                names.append(name)
                images.append(img)
            prog.progress(1.0, text="Terminé ✅")
            time.sleep(0.3); prog.empty()
            st.session_state.results, st.session_state.names, st.session_state.images = results, names, images

        # affichage resultats
        if st.session_state.results:
            st.markdown('<div class="sec"><span class="bar"></span>📈 Synthèse</div>',
                        unsafe_allow_html=True)
            render_kpis(st.session_state.results)

            st.download_button(
                "⬇️ Télécharger les résultats (CSV)",
                data=build_csv(st.session_state.results, st.session_state.names),
                file_name="resultats_leyenda.csv",
                mime="text/csv",
                use_container_width=False,
            )

            st.markdown(
                f'<div class="sec"><span class="bar"></span>🗂️ Résultats ({len(st.session_state.results)})</div>',
                unsafe_allow_html=True,
            )
            for name, img, res in zip(st.session_state.names,
                                      st.session_state.images,
                                      st.session_state.results):
                render_result_card(name, img, res)
        else:
            st.info("Importez des images puis cliquez sur **Lancer le pipeline** dans le panneau de gauche.")

    # ----------------------------- STATISTIQUES -----------------------------
    with tab_stats:
        st.markdown('<div class="sec"><span class="bar"></span>📊 Statistiques de la session</div>',
                    unsafe_allow_html=True)
        res = st.session_state.results
        if not res:
            st.info("Aucune donnée pour l'instant — lancez d'abord le pipeline.")
        else:
            render_kpis(res)
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
        st.markdown('<div class="sec"><span class="bar"></span>🏗️ Architecture du pipeline</div>',
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


if __name__ == "__main__":
    main()
