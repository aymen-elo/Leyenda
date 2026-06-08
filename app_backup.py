"""
Projet Leyenda - Pipeline TouNum  (interface de demonstration Streamlit)
=======================================================================
Demonstration live du pipeline complet sur des images inconnues :
    L1  Classification  (Photo / Non-Photo)
    L2  Debruitage      (U-Net a skip-connections)
    L3  Captioning      (InceptionV3 + LSTM)

Toute la logique d'inference vit dans pipeline.py (testable hors Streamlit).
Lancement :  streamlit run app.py    (ou double-clic sur run_demo.bat)

Compatibilite : ecrit pour Streamlit 1.22 (impose par TensorFlow 2.10 / protobuf < 3.20).
"""

import time

import numpy as np
import streamlit as st

import pipeline as pl

# ---------------------------------------------------------------------------
# PALETTE TouNum (bleu nuit + accents ambre/orange)
# ---------------------------------------------------------------------------
C_BG, C_CARD = "#0e1525", "#161f33"
C_AMBER, C_ORANGE = "#f5a623", "#ff7a18"
C_GREEN, C_RED = "#2ecc71", "#e74c3c"
C_TEXT, C_MUTED = "#e8edf6", "#8a97ad"

st.set_page_config(page_title="Projet Leyenda - TouNum", page_icon="🖼️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
    .stApp {{ background: radial-gradient(1200px 600px at 20% -10%, #16223d 0%, {C_BG} 55%); color:{C_TEXT}; }}
    section[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#101a30 0%,#0b1222 100%); border-right:1px solid #22304d; }}
    h1,h2,h3,h4 {{ color:{C_TEXT}; letter-spacing:.3px; }}
    .ly-title {{ font-size:2.3rem; font-weight:800; margin-bottom:.1rem;
        background:linear-gradient(90deg,{C_AMBER},{C_ORANGE}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
    .ly-sub {{ color:{C_MUTED}; font-size:1.02rem; margin-top:0; }}
    .ly-flow {{ margin:.4rem 0 1rem 0; }}
    .ly-flow span.node {{ display:inline-block; padding:6px 14px; border-radius:10px; background:#16223d; border:1px solid #2a3a5c; font-weight:700; font-size:.9rem; }}
    .ly-flow span.arrow {{ color:{C_AMBER}; font-weight:800; margin:0 8px; }}
    .ly-card {{ background:{C_CARD}; border:1px solid #233152; border-radius:16px; padding:18px 20px; margin-bottom:18px; box-shadow:0 8px 30px rgba(0,0,0,.35); }}
    .ly-caption {{ font-size:1.5rem; font-weight:700; line-height:1.4; color:{C_TEXT}; padding:10px 4px; }}
    .ly-caption::before {{ content:"💬 "; }}
    .badge {{ display:inline-block; padding:5px 14px; border-radius:999px; font-weight:700; font-size:.92rem; color:#0b1222; }}
    .badge-photo {{ background:{C_GREEN}; }} .badge-nonphoto {{ background:{C_RED}; color:#fff; }} .badge-default {{ background:{C_AMBER}; }}
    .chip {{ display:inline-block; padding:3px 10px; border-radius:8px; margin-right:6px; font-size:.82rem; background:#1d2944; color:{C_MUTED}; border:1px solid #2a3a5c; }}
    .stat-num {{ font-size:2.4rem; font-weight:800; color:{C_AMBER}; }} .stat-lbl {{ color:{C_MUTED}; font-size:.95rem; }}
    .status-ok {{ color:{C_GREEN}; font-weight:700; }} .status-warn {{ color:{C_AMBER}; font-weight:700; }} .status-ko {{ color:{C_RED}; font-weight:700; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
    .stTabs [data-baseweb="tab"] {{ background:#14203a; border-radius:10px 10px 0 0; padding:8px 18px; }}
    .stTabs [aria-selected="true"] {{ background:#1d2d4e; color:{C_AMBER}; }}
    div.stButton > button {{ border-radius:12px; font-weight:700; border:1px solid #2a3a5c; background:#16223d; color:{C_TEXT}; width:100%; }}
    div.stButton > button:hover {{ border-color:{C_AMBER}; color:{C_AMBER}; }}
    div.stButton > button[kind="primary"] {{ background:linear-gradient(90deg,{C_ORANGE},{C_AMBER}); color:#0b1222; font-weight:800; border:0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# CHARGEMENT DES MODELES (une seule fois, mis en cache)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des modeles (TensorFlow + InceptionV3)...")
def load_pipeline():
    return pl.build_pipeline()


def fmt_params(n):
    if n is None:
        return "—"
    if n >= 1e6:
        return f"{n/1e6:.1f} M"
    if n >= 1e3:
        return f"{n/1e3:.0f} k"
    return str(n)


def thumb(img, max_side=480):
    """Vignette pour l'affichage (limite la charge navigateur sur grosses images)."""
    t = img.copy()
    t.thumbnail((max_side, max_side))
    return t


# ---------------------------------------------------------------------------
# RENDU D'UNE CARTE RESULTAT
# ---------------------------------------------------------------------------
def badge_html(label):
    if label.startswith("Photo (par"):
        return f'<span class="badge badge-default">⚠️ {label}</span>'
    if label == "Photo":
        return '<span class="badge badge-photo">✅ Photo</span>'
    return f'<span class="badge badge-nonphoto">⛔ {label}</span>'


def render_card(res):
    st.markdown('<div class="ly-card">', unsafe_allow_html=True)
    st.markdown(f"#### 🖼️ {res['name']}")

    col_o, col_d = st.columns(2)
    with col_o:
        st.markdown("**Originale**")
        st.image(thumb(res["image"]), use_column_width=True)
    with col_d:
        if res["denoised"] is not None:
            corr = res.get("noise_delta")
            corr_txt = f" · correction {corr*100:.1f}%" if corr is not None else ""
            st.markdown(f"**🧹 Débruitée (L2){corr_txt}**")
            st.image(thumb(res["denoised"]), use_column_width=True)
        else:
            st.markdown("**🧹 Débruitage (L2)**")
            st.info("L2 désactivé — image originale utilisée pour la suite.")

    proba_txt = (f" &nbsp; <span class='chip'>p = {res['proba']:.2f}</span>"
                 if res.get("proba") is not None else "")
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
        f"<span class='chip'>Total : {res['t1']+res['t2']+res['t3']:.0f} ms</span></div>",
        unsafe_allow_html=True,
    )
    if res["errors"]:
        st.warning("Avertissements : " + " | ".join(res["errors"]))
    st.markdown("</div>", unsafe_allow_html=True)


def summary_metrics(results, elapsed):
    n_photo = sum(1 for r in results if r["label"] == "Photo" or r["label"].startswith("Photo (par"))
    n_nonphoto = sum(1 for r in results if r["label"] == "Non-Photo")
    avg = np.mean([r["t1"] + r["t2"] + r["t3"] for r in results])
    c = st.columns(5)
    c[0].metric("🖼️ Images", len(results))
    c[1].metric("✅ Photos", n_photo)
    c[2].metric("⛔ Non-photos", n_nonphoto)
    c[3].metric("⚡ Moy. / image", f"{avg:.0f} ms")
    c[4].metric("⏱️ Temps total", f"{elapsed:.1f} s")


# ---------------------------------------------------------------------------
# TRAITEMENT
# ---------------------------------------------------------------------------
def run_pipeline(P, items):
    t0 = time.time()
    results = []
    progress = st.progress(0.0, text="Traitement en cours…")
    for i, (name, img) in enumerate(items):
        with st.spinner(f"Pipeline sur « {name} » ({i+1}/{len(items)})…"):
            results.append(pl.process_image(P, name, img))
        progress.progress((i + 1) / len(items), text=f"{i+1}/{len(items)} traitées")
    progress.empty()
    st.session_state["results"] = results
    st.session_state["elapsed"] = time.time() - t0


# ===========================================================================
# INTERFACE
# ===========================================================================
P = load_pipeline()
status = P["status"]

st.markdown('<div class="ly-title">Projet Leyenda — Pipeline TouNum</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="ly-sub">Numérisation intelligente : classification, débruitage et génération '
    'de légende — démonstration live sur des images inconnues.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='ly-flow'>"
    "<span class='node'>🖼️ Image</span><span class='arrow'>→</span>"
    "<span class='node'>🔍 L1 · Classification</span><span class='arrow'>→</span>"
    "<span class='node'>🧹 L2 · Débruitage</span><span class='arrow'>→</span>"
    "<span class='node'>💬 L3 · Légende</span></div>",
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
                f"<span class='chip'>{'GPU ✅' if P['gpu'] else 'CPU'}</span>", unsafe_allow_html=True)
    st.markdown("---")
    n_images = st.slider("Nombre d'images à traiter", 1, 50, 8)
    run = st.button("🚀 Lancer le pipeline", type="primary")
    demo = st.button("🎞️ Démo rapide (dossier Dataset/)")
    reset = st.button("🗑️ Réinitialiser")

# --- Onglets ---
tab_pipe, tab_stats, tab_arch = st.tabs(["🔍 Pipeline", "📊 Statistiques", "🏗️ Architecture"])

# ============================ ONGLET PIPELINE ==============================
with tab_pipe:
    if reset:
        st.session_state.pop("results", None)
        st.session_state.pop("elapsed", None)

    st.markdown("#### 🖼️ Sélection des images")
    uploaded = st.file_uploader(
        "Glissez-déposez vos images ici (sélection multiple)",
        type=[e.strip(".") for e in pl.EXTS], accept_multiple_files=True,
    )
    folder_str = st.text_input(
        "…ou indiquez le chemin d'un dossier d'images (sur cette machine)",
        placeholder=r"C:\Users\matis\Desktop\dataset_prof",
    )

    items = None
    if demo:
        items = pl.gather_images(None, str(pl.DEMO_DIR), n_images)
        if not items:
            st.error(f"Dossier de démo introuvable ou vide : {pl.DEMO_DIR}")
    elif run:
        items = pl.gather_images(uploaded, folder_str, n_images)
        if not items:
            st.error("Aucune image trouvée. Uploadez des fichiers ou indiquez un dossier valide.")

    if items:
        st.success(f"{len(items)} image(s) à traiter.")
        run_pipeline(P, items)

    results = st.session_state.get("results", [])
    if results:
        summary_metrics(results, st.session_state.get("elapsed", 0.0))
        st.download_button(
            "⬇️ Télécharger les résultats (CSV)",
            data=pl.results_to_csv(results),
            file_name="resultats_tounum.csv",
            mime="text/csv",
        )
        st.markdown(f"### Résultats ({len(results)})")
        for res in results:
            render_card(res)
    elif not (run or demo):
        st.info("Chargez des images (ou cliquez **🎞️ Démo rapide**) puis lancez le pipeline depuis la barre latérale.")

# ============================ ONGLET STATISTIQUES ==========================
with tab_stats:
    results = st.session_state.get("results", [])
    st.markdown("#### 📊 Statistiques de la dernière exécution")
    if not results:
        st.info("Aucune exécution pour le moment.")
    else:
        import pandas as pd

        summary_metrics(results, st.session_state.get("elapsed", 0.0))

        st.markdown("##### ⏱️ Temps total par image (ms)")
        df_time = pd.DataFrame(
            {"Total (ms)": [round(r["t1"] + r["t2"] + r["t3"]) for r in results]},
            index=[r["name"] for r in results],
        )
        st.bar_chart(df_time)

        st.markdown("##### 🧩 Temps moyen par étape (ms)")
        df_stage = pd.DataFrame(
            {"Temps moyen (ms)": [
                round(np.mean([r["t1"] for r in results])),
                round(np.mean([r["t2"] for r in results])),
                round(np.mean([r["t3"] for r in results])),
            ]},
            index=["L1 · Classification", "L2 · Débruitage", "L3 · Captioning"],
        )
        st.bar_chart(df_stage)

        st.markdown("##### 📋 Détail")
        df_stats = pd.DataFrame(
            [{"Image": r["name"], "Classe": r["label"],
              "Correction L2 (%)": None if r.get("noise_delta") is None else round(r["noise_delta"] * 100, 1),
              "L1 (ms)": round(r["t1"]), "L2 (ms)": round(r["t2"]),
              "L3 (ms)": round(r["t3"]), "Total (ms)": round(r["t1"] + r["t2"] + r["t3"]),
              "Légende": r["caption"]}
             for r in results]
        )
        st.dataframe(df_stats, use_container_width=True)

# ============================ ONGLET ARCHITECTURE ==========================
with tab_arch:
    st.markdown("#### 🏗️ Architecture du pipeline TouNum")
    st.markdown(
        """
```
   Image inconnue
        │
        ▼
┌────────────────────┐   L1 · Classification (CNN from scratch)
│  PHOTO / NON-PHOTO  │   src/models/cnn_scratch.keras
└────────────────────┘   → Photo (par défaut) si absent
        │
        ▼
┌────────────────────┐   L2 · Débruitage (U-Net à skip-connections)
│   IMAGE NETTOYÉE    │   src/models/unet_denoiser_tf210.keras
└────────────────────┘   → image originale si désactivé
        │
        ▼
┌────────────────────┐   L3 · Captioning (InceptionV3 → LSTM)
│  LÉGENDE GÉNÉRÉE    │   src/models/captioning_model.keras + tokenizer
└────────────────────┘   (vocab 5000, décodage glouton)
        │
        ▼
   « a man riding a wave on a surfboard »
```
        """
    )

    params = pl.model_params(P)
    st.markdown("##### Détails techniques")
    cA, cB, cC = st.columns(3)
    cA.markdown(
        f"<div class='ly-card'><b>🔍 L1 · Classification</b><br><span class='stat-lbl'>"
        f"CNN entraîné from scratch. Entrée 224×224, sortie binaire (Photo / Non-Photo).<br>"
        f"Paramètres : {fmt_params(params['L1'])}<br>Statut : {status['L1'][0].upper()}</span></div>",
        unsafe_allow_html=True)
    cB.markdown(
        f"<div class='ly-card'><b>🧹 L2 · Débruitage</b><br><span class='stat-lbl'>"
        f"U-Net convolutif à skip-connections (5 niveaux). Entrée/sortie 256×256 RGB.<br>"
        f"Paramètres : {fmt_params(params['L2'])}<br>Statut : {status['L2'][0].upper()}</span></div>",
        unsafe_allow_html=True)
    cC.markdown(
        f"<div class='ly-card'><b>💬 L3 · Captioning</b><br><span class='stat-lbl'>"
        f"Encodeur InceptionV3 (2048-d, gelé, {fmt_params(params['encoder'])} param.) + "
        f"décodeur LSTM par fusion ({fmt_params(params['L3'])} param.). Vocab 5000.<br>"
        f"Statut : {status['L3'][0].upper()}</span></div>",
        unsafe_allow_html=True)

    st.markdown("##### Note technique — compatibilité L2")
    st.markdown(
        "<div class='ly-card'><span class='stat-lbl'>"
        "Le U-Net a été entraîné sous <b>Keras 3</b> (format <code>.keras</code> illisible par "
        "TensorFlow 2.10). Il a été <b>reconstruit à l'identique</b> en Keras 2.10 puis ré-enregistré "
        "en HDF5 natif (<code>unet_model_tf210.h5</code>) — sorties numériquement identiques à l'original "
        "(écart &lt; 1e-6).</span></div>",
        unsafe_allow_html=True)

    st.markdown("##### Environnement")
    st.markdown(
        f"<span class='chip'>TensorFlow {P['tf_version']}</span>"
        f"<span class='chip'>{'GPU : ' + P['gpu'][0] if P['gpu'] else 'CPU uniquement'}</span>"
        f"<span class='chip'>Encodeur : InceptionV3 / ImageNet</span>"
        f"<span class='chip'>Streamlit 1.22</span>", unsafe_allow_html=True)
