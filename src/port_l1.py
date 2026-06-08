"""
Portage du classifieur L1 (CNN_Scratch) de Keras 3 vers TensorFlow 2.10.
======================================================================
Le modele de classification a ete entraine et sauvegarde sous **Keras 3.12**
(format `.keras` = archive ZIP). TensorFlow 2.10 / Keras 2.10 — impose par le
reste du pipeline (protobuf < 3.20) — ne sait pas lire ce format et echoue avec
« Unable to synchronously open file (file signature not found) » : l'app retombe
alors sur « Photo (par defaut) ».

Ce script reconstruit l'architecture *a l'identique* (`build_cnn` de retrain_l1.py)
sous Keras 2.10, y injecte les poids entraines extraits de l'archive Keras 3
(`model.weights.h5`), puis re-enregistre le modele dans un format lisible par
TF 2.10. AUCUN re-entrainement : les poids sont ceux du modele fourni.

L'ordre des variables est identique entre Keras 3 et Keras 2.10 :
    Conv2D            -> [kernel, bias]
    BatchNormalization-> [gamma, beta, moving_mean, moving_variance]
    Dense             -> [kernel, bias]

Usage :
    python src/port_l1.py [--src <chemin.keras Keras3>]
"""

import argparse
import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
from tensorflow.keras.models import load_model

from retrain_l1 import OUT_PATH, build_cnn  # architecture de reference


def load_keras3_layer_weights(keras3_path):
    """Extrait {nom_de_couche: [arrays]} depuis l'archive .keras (Keras 3)."""
    with zipfile.ZipFile(keras3_path) as z:
        raw = z.read("model.weights.h5")
    weights = {}
    with h5py.File(io.BytesIO(raw), "r") as f:
        layers = f["layers"]
        for name in layers.keys():
            vars_grp = layers[name].get("vars")
            if vars_grp is None or len(vars_grp.keys()) == 0:
                continue
            ordered = sorted(vars_grp.keys(), key=int)  # '0','1','2',... -> ordre numerique
            weights[name] = [np.array(vars_grp[i]) for i in ordered]
    return weights


def main():
    ap = argparse.ArgumentParser(description="Portage L1 Keras 3 -> TF 2.10")
    ap.add_argument("--src", default=str(Path.home() / "Downloads" / "cnn_scratch.keras"),
                    help="Chemin du .keras Keras 3 a porter")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"Modele source introuvable : {src}")

    print("=" * 70)
    print("PORTAGE L1 (CNN_Scratch)  Keras 3 -> TensorFlow 2.10")
    print(f"Source : {src}")
    print(f"Cible  : {OUT_PATH}")
    print("=" * 70)

    src_weights = load_keras3_layer_weights(src)
    print(f"Couches a poids dans l'archive : {len(src_weights)}")

    model = build_cnn()  # meme architecture, fraichement initialisee sous Keras 2.10

    ported, skipped = 0, []
    for layer in model.layers:
        if not layer.weights:
            continue
        if layer.name not in src_weights:
            skipped.append(layer.name)
            continue
        cur = layer.get_weights()
        new = src_weights[layer.name]
        if len(cur) != len(new) or any(c.shape != n.shape for c, n in zip(cur, new)):
            raise ValueError(
                f"Incompatibilite de poids pour '{layer.name}': "
                f"{[c.shape for c in cur]} vs {[n.shape for n in new]}")
        layer.set_weights(new)
        ported += 1

    if skipped:
        raise RuntimeError(f"Couches a poids non appariees : {skipped}")
    print(f"Poids injectes dans {ported} couches.")

    model.save(OUT_PATH)
    print(f"Modele sauvegarde -> {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    # Verification : rechargement comme dans l'app (compile=False) + inference
    m2 = load_model(OUT_PATH, compile=False)
    print(f"Reload compile=False : OK  | {m2.input_shape} -> {m2.output_shape}")
    probe = np.random.rand(2, 224, 224, 3).astype("float32")
    p = m2.predict(probe, verbose=0).ravel()
    print(f"Inference de controle : proba sur 2 images aleatoires = {np.round(p, 4)}")
    print("OK — L1 est desormais chargeable par TF 2.10.")


if __name__ == "__main__":
    main()
