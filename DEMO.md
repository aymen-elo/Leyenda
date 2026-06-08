# 🖼️ Démo soutenance — Pipeline TouNum (Projet Leyenda)

Interface Streamlit pour faire tourner **en direct** le pipeline complet sur le
dataset d'images fourni par la prof :

| Étape | Rôle | Modèle |
|------|------|--------|
| **L1** | Classification *Photo / Non-Photo* | `src/models/cnn_scratch.keras` *(optionnel)* |
| **L2** | Débruitage (U-Net) | `src/models/unet_denoiser_tf210.keras` |
| **L3** | Génération de légende (InceptionV3 + LSTM) | `src/models/captioning_model.keras` + `captioning_tokenizer.pkl` |

---

## 🚀 Lancer la démo (le jour J)

### Option simple — double-clic
Double-cliquez sur **`run_demo.bat`** à la racine du projet.
Une fenêtre console s'ouvre, puis le navigateur s'ouvre tout seul sur l'appli
(`http://localhost:8501`). **Laissez la console ouverte** pendant toute la démo.

### Option manuelle (si besoin)
Dans un terminal, à la racine du projet :
```bat
C:\Users\matis\envs\tf-gpu\Scripts\streamlit.exe run app.py
```

> ⏳ Le **premier chargement prend ~30 s à 1 min** (TensorFlow + InceptionV3).
> C'est normal. Les images suivantes sont ensuite traitées en quelques centaines
> de millisecondes (GPU GTX 1050).

Pour arrêter : fermez l'onglet du navigateur puis `Ctrl + C` dans la console
(ou fermez simplement la fenêtre console).

---

## 🎬 Déroulé conseillé en soutenance

1. **Lancer** `run_demo.bat` **avant de commencer à parler** (le temps que les
   modèles chargent en arrière-plan).
2. Dans la **barre latérale**, vérifier le **statut des 3 modèles** :
   - ✅ vert = modèle chargé
   - ⚠️ ambre = absent ou non chargeable → étape neutralisée proprement
   - ❌ rouge = erreur
3. Régler le **« Nombre d'images à traiter »** (ex. 5–8 pour une démo fluide).
4. Onglet **🔍 Pipeline** :
   - soit **glisser-déposer** les images de la prof dans la zone d'upload,
   - soit **coller le chemin du dossier** d'images dans le champ texte.
5. Cliquer sur **🚀 Lancer le pipeline**. Une barre de progression et des
   spinners s'affichent pendant le traitement.
6. Pour chaque image, une **carte** montre : originale | débruitée (L2),
   le **badge de classification** (L1), la **légende générée** (L3) et le
   **temps par étape** (ms).
7. Onglet **📊 Statistiques** : nombre de photos / non-photos, temps moyen,
   tableau détaillé des temps.
8. Onglet **🏗️ Architecture** : schéma du pipeline + infos techniques des modèles
   (utile pour répondre aux questions du jury).

---

## 🧩 Gestion des modèles manquants

L'interface est **robuste** : elle ne plante jamais si un modèle manque.

- **L1 absent** (`cnn_scratch.keras` non fourni) → affiche le badge
  **« Photo (par défaut) »**. Dès que le fichier sera placé dans
  `src/models/cnn_scratch.keras`, il sera **utilisé automatiquement** au prochain
  lancement (sortie binaire attendue, entrée 224×224).
- **L2** → utilise `src/models/unet_denoiser_tf210.keras` (puis, à défaut,
  `notebooks/unet_model_tf210.h5`). Si aucun n'est présent, le débruitage est
  neutralisé et l'image originale est transmise à L3.
- Si **une image fait planter une étape**, le traitement **continue** avec les
  images suivantes ; l'erreur est affichée en avertissement sur la carte.

> **Note technique L2.** Le U-Net du livrable 2 avait été sauvegardé avec **Keras 3**
> (format `.keras` zip), illisible par l'environnement de démo **TensorFlow 2.10
> (Keras 2.10)**. Il a donc été **ré-entraîné nativement sous TF 2.10** via
> `src/retrain_l2.py` (même architecture U-Net à skip-connections, bruit gaussien
> 0.15) → **`src/models/unet_denoiser_tf210.keras`**. C'est ce modèle que charge l'appli.
> *(Une reconstruction HDF5 du modèle d'origine, `notebooks/unet_model_tf210.h5`, sert
> de repli automatique si le modèle ré-entraîné est absent.)*

---

## 🛠️ Installation (déjà faite — pour mémoire / autre PC)

Environnement Python utilisé : `C:\Users\matis\envs\tf-gpu` (TensorFlow 2.10 GPU).

```bat
REM Streamlit, en gardant protobuf compatible avec TensorFlow 2.10
C:\Users\matis\envs\tf-gpu\Scripts\pip.exe install "streamlit==1.22.0" "protobuf==3.19.6"
```

> ⚠️ **Ne pas faire `pip install streamlit` sans pin.** Streamlit récent force
> `protobuf >= 3.20`, ce qui **casse TensorFlow 2.10** (qui exige `protobuf < 3.20`).
> Les versions ci-dessus sont le compromis testé et fonctionnel.

### Ré-entraîner le modèle de débruitage L2 (si besoin)

Le script `src/retrain_l2.py` reproduit l'architecture exacte du U-Net du livrable 2
et le ré-entraîne **dans l'environnement TF 2.10** (donc directement compatible) :

```bat
C:\Users\matis\envs\tf-gpu\Scripts\python.exe src\retrain_l2.py
REM 2000 images de data\raw\train2014\, bruit gaussien 0.15, 10 epochs
REM -> src\models\unet_denoiser_tf210.keras  (~30-40 min sur GTX 1050)
```

> ⚠️ **Fermez l'appli Streamlit avant d'entraîner** : la GTX 1050 n'a que 2 Go de
> VRAM, partagés. Options utiles : `--n-images`, `--epochs`, `--batch-size`
> (gardez `--batch-size 4` au maximum sur cette carte).

---

## ✅ Vérification rapide (hors interface)

Pour tester le pipeline en ligne de commande sur le dossier `Dataset/` :

```bat
C:\Users\matis\envs\tf-gpu\Scripts\python.exe -c "import pipeline as pl; P=pl.build_pipeline(); print(P['status']); [print(r['name'], '->', r['caption']) for r in (pl.process_image(P,n,i) for n,i in pl.gather_images(None,'Dataset',3))]"
```

Sortie attendue (exemple) :
```
{'L3': ('ok', ...), 'L1': ('warn', ...), 'L2': ('ok', 'unet_denoiser_tf210.keras')}
noisy_001.jpg -> a man riding a wave on a surfboard in the ocean
noisy_002.jpg -> a large plane is parked on the runway
noisy_003.jpg -> a man riding a motorcycle down a street
```

---

## 📁 Fichiers de la démo

| Fichier | Rôle |
|---------|------|
| `app.py` | Interface Streamlit (UI, thème sombre, onglets) |
| `pipeline.py` | Logique d'inférence L1/L2/L3 (testable hors Streamlit) |
| `src/retrain_l2.py` | Ré-entraînement du U-Net de débruitage sous TF 2.10 |
| `run_demo.bat` | Lancement en un double-clic |
| `DEMO.md` | Ce guide |
