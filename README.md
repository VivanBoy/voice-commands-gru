# 🎙️ Voice Commands Recognition (Keyword Spotting) — MFCC + GRU  
**Cours : Apprentissage profond appliqué avancé — Projet final**

Ce projet réalise un système de **reconnaissance de commandes vocales** (*keyword spotting*) : détecter des mots courts (ex. `yes`, `no`, `up`, `down`…) à partir d’un enregistrement audio.  
Pipeline : **Audio (WAV) → Extraction MFCC → Modèle GRU → Prédiction (Softmax)**, avec des démonstrations interactives.

---

## ✅ Objectifs du projet
- Construire un pipeline complet de classification audio (prétraitement + entraînement + évaluation).
- Reconnaître des **commandes vocales courtes** (≈ 1 seconde) parmi des classes ciblées.
- Fournir des **démos** pour montrer l’usage réel du modèle :
  - **Gradio** : micro / upload audio → top prédictions
  - **Turtle** : contrôle d’un curseur/flèche par la voix

---

## 🌍 Importance / Applications réelles
La reconnaissance de mots-clés est au cœur de nombreux usages :
- assistants vocaux (déclenchement de commandes, “wake word”)
- domotique (on/off), IoT, appareils embarqués
- accessibilité (contrôle mains libres)
- interfaces vocales offline (faible latence, confidentialité)

Ce projet correspond à un cas réel très courant :  
> comprendre une commande simple, vite, sans transcription complète de phrases.

---

## 📦 Données (Collecte)
### Source : Google Speech Commands Dataset
Dataset public de Google conçu pour le **keyword spotting**, contenant des enregistrements audio de **mots isolés** prononcés par de nombreux locuteurs.

**Caractéristiques générales :**
- clips courts (≈ 1 seconde)
- multiples locuteurs / conditions d’enregistrement
- présence de bruit de fond (`_background_noise_`) utile pour générer la classe “silence” et augmenter la robustesse

### Classes utilisées dans ce projet
Le projet est construit autour de **12 classes** :
- **10 commandes** : `yes, no, up, down, left, right, on, off, stop, go`
- **unknown** : regroupe tous les autres mots (hors liste)
- **silence** : générée à partir de bruit de fond / segments silencieux

---

## 🧪 Prétraitement (Features)
- Resample / standardisation audio (typ. `16 kHz`)
- Durée fixée à ~**1 seconde** (padding / trim)
- Extraction **MFCC** :
  - `N_MFCC = 40`
  - séquence temporelle d’environ `97 frames`
- Normalisation par statistiques du train :
  - `X_norm = (X - mu) / sigma`

> Les scripts de démo appliquent le même prétraitement que l’entraînement.

---

## 🧠 Modèle (Deep Learning)
Architecture de type :
- **GRU** (réseau récurrent) sur séquences MFCC
- couches denses + **Softmax** (classification multi-classes)

Entraînement :
- `Adam` + `SparseCategoricalCrossentropy`
- callbacks de régularisation / stabilité :
  - `EarlyStopping`
  - `ReduceLROnPlateau`
  - `ModelCheckpoint`

📌 Résultat obtenu (exemple) : **~91% accuracy sur test**.

---

## 📁 Structure du projet
```bash
voice-commands-gru/
│── .gitignore
│── app_gradio.py                  # Démo web (micro/upload) avec Gradio
│── demo_turtle_voice.py           # Démo Turtle contrôlée par la voix
│── reco_vocal_v.ipynb             # Notebook principal (pipeline complet)
│── models/
│   └── gru_speech_commands.keras  # Modèle entraîné (~1MB)
│── data/
│   └── processed/
│       └── stats_mfcc40_T97.npz   # (mu, sigma, labels) léger, nécessaire aux démos
```

---

## ⚙️ Installation (Windows / VS Code)
Recommandé : **Python 3.12** (compatibilité TensorFlow Windows).

Créer un environnement virtuel :
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
```

Installer les dépendances principales :
```bash
pip install --upgrade pip
pip install tensorflow librosa soundfile scikit-learn matplotlib tqdm gradio sounddevice scipy
```

Optionnel (mais pro) :
```bash
pip freeze > requirements.txt
```

---

## ▶️ Utilisation

### 1) Démo Gradio (Micro / Upload)
Lancer :
```bash
python app_gradio.py
```

Puis ouvrir l’URL affichée (ex. `http://127.0.0.1:7860`).

**Notes :**
- la démo prend automatiquement la **meilleure seconde** (segment le plus “parlé”) pour éviter le silence au début.
- si un mot est hors liste, la prédiction peut tomber sur `unknown`.

---

### 2) Démo Turtle (contrôle par la voix)
Lancer :
```bash
python demo_turtle_voice.py
```

Commandes (selon la version finale du script) :
- `up/down/left/right` : tourner vers la direction
- `go` : avancer d’un pas
- `on` : commencer à tracer (pen down)
- `off` : arrêter de tracer (pen up)
- `yes` : **clear** + recentrer (effacer l’écran)
- `stop` : quitter

✅ Exemple d’usage :
1. dire `on` (activer tracé)
2. dire `right` (tourner)
3. dire `go` plusieurs fois (avancer + dessiner)
4. dire `yes` pour effacer

---

## 🧾 Fichiers volumineux (Important GitHub)
Certaines features peuvent être gigantesques si on sauvegarde tout le dataset prétraité.  
Exemple : `data/processed/speech_mfcc40_T97.npz` peut faire **> 600 MB**.

➡️ Pour GitHub, on versionne uniquement un fichier léger :
- `data/processed/stats_mfcc40_T97.npz` contenant `mu`, `sigma`, `labels`

Le fichier “gros” (si présent en local) doit rester ignoré via `.gitignore`.

---

## 🚀 Améliorations possibles
- Data augmentation audio (bruit, time shift, pitch léger)
- Modèle CRNN (CNN + GRU) sur log-mel spectrogrammes
- Export TensorFlow Lite (déploiement mobile/edge)
- Vrai temps réel (fenêtre glissante + vote majoritaire)

---

## 👤 Auteur
Projet réalisé dans le cadre du cours **Apprentissage profond appliqué avancé**  
Reconnaissance de commandes vocales — **MFCC + GRU** — Speech Commands Dataset
