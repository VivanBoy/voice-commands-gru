# app_gradio.py
# Démo Keyword Spotting (Speech Commands) avec Gradio
# - Prend une entrée micro OU upload
# - Convertit en mono, resample 16kHz
# - Sélectionne automatiquement la fenêtre de 1 seconde la plus "parlée" (énergie RMS max)
# - Extrait MFCC (97x40), normalise (mu/sigma), prédit Top-5

import numpy as np
import librosa
import gradio as gr
from pathlib import Path
from tensorflow import keras

# Paths
ROOT = Path(".").resolve()
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"

MODEL_PATH = MODELS_DIR / "gru_speech_commands.keras"
NPZ_PATH = DATA_PROCESSED / "speech_mfcc40_T97.npz"  

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not NPZ_PATH.exists():
    # fallback: prendre le premier .npz si le nom exact n'existe pas
    candidates = sorted(DATA_PROCESSED.glob("speech_mfcc*_T*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No NPZ found in: {DATA_PROCESSED}")
    NPZ_PATH = candidates[0]

print("Using model:", MODEL_PATH)
print("Using npz:", NPZ_PATH)

# Load model + preprocessing stats
model = keras.models.load_model(MODEL_PATH)

data = np.load(NPZ_PATH, allow_pickle=True)
mu = data["mu"]
sigma = data["sigma"]
LABELS = list(data["labels"])
id2label = {i: lab for i, lab in enumerate(LABELS)}

# Audio / MFCC params (doivent matcher le training)
SR = 16000
SAMPLES = SR * 1          # 1 seconde
N_MFCC = 40
WIN_LENGTH = int(0.025 * SR)   # 25ms
HOP_LENGTH = int(0.010 * SR)   # 10ms
N_FFT = 512
T_FRAMES = 97  # Speech Commands 1s ~97 frames avec hop 10ms

# Helpers
def pick_loudest_1s(y: np.ndarray, sr: int, win_samples: int, hop_samples: int):
    """
    Retourne (segment_1s, start_index) avec le plus d'énergie RMS.
    """
    if len(y) <= win_samples:
        y = np.pad(y, (0, max(0, win_samples - len(y))))
        return y[:win_samples], 0

    starts = range(0, len(y) - win_samples + 1, hop_samples)
    best_start = 0
    best_rms = -1.0

    for s in starts:
        seg = y[s:s + win_samples]
        rms = float(np.sqrt(np.mean(seg**2)) + 1e-12)
        if rms > best_rms:
            best_rms = rms
            best_start = s

    return y[best_start:best_start + win_samples], best_start


def audio_to_mfcc_1s(y: np.ndarray):
    """
    y: audio 1 seconde à 16kHz mono
    retourne MFCC (T_FRAMES, N_MFCC) float32
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        center=False
    ).T

    # longueur fixe
    if mfcc.shape[0] < T_FRAMES:
        mfcc = np.pad(mfcc, ((0, T_FRAMES - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:T_FRAMES]

    return mfcc.astype(np.float32)


def predict_topk_from_audio(sr_in: int, y: np.ndarray, topk: int = 5):
    """
    sr_in: sample rate d'entrée (gradio)
    y: numpy array audio (mono ou multi)
    """
    # mono
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # resample -> 16k
    if sr_in != SR:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=SR)

    # nettoyage léger
    y = y - np.mean(y)
    y = y / (np.max(np.abs(y)) + 1e-9)

    # choisir la meilleure seconde (au lieu de la première)
    y1, _ = pick_loudest_1s(y, SR, SAMPLES, hop_samples=int(0.1 * SR))

    # MFCC + normalisation
    mfcc = audio_to_mfcc_1s(y1)
    mfcc = (mfcc - mu) / sigma

    x = mfcc[None, ...]  # (1, T, N_MFCC)
    proba = model.predict(x, verbose=0)[0]

    idxs = np.argsort(proba)[::-1][:topk]
    return {id2label[int(i)]: float(proba[i]) for i in idxs}


# Gradio function
def gradio_predict(audio):
    """
    audio: None ou (sr, np.array)
    """
    if audio is None:
        # fallback
        return {"silence": 1.0} if "silence" in LABELS else {"unknown": 1.0}

    sr_in, y = audio
    scores = predict_topk_from_audio(sr_in, y, topk=5)
    return scores


# UI
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        type="numpy",
        label="🎙️ Micro ou Upload (parle ~1-2 sec: yes/no/up/down/left/right/on/off/stop/go)"
    ),
    outputs=gr.Label(num_top_classes=5, label="Top prédictions"),
    title="🎙️ Keyword Spotting — MFCC + GRU (Speech Commands)",
    description="Démo: on choisit automatiquement la fenêtre de 1 seconde la plus parlée. "
                "Si tu dis un mot hors liste → 'unknown'."
)

if __name__ == "__main__":
    demo.launch()