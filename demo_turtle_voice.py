

import time
import threading
import queue
from collections import deque, Counter
from pathlib import Path

import numpy as np
import librosa
import sounddevice as sd
from tensorflow import keras
import turtle

# 1) Paths 

ROOT = Path(".").resolve()
MODEL_PATH = ROOT / "models" / "gru_speech_commands.keras"
PROCESSED_DIR = ROOT / "data" / "processed"
NPZ_PATH = PROCESSED_DIR / "speech_mfcc40_T97.npz"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not NPZ_PATH.exists():
    # fallback 
    candidates = sorted(PROCESSED_DIR.glob("speech_mfcc*_T*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No NPZ found in: {PROCESSED_DIR}")
    NPZ_PATH = candidates[0]

print("Model:", MODEL_PATH)
print("NPZ  :", NPZ_PATH)


# 2) Charger modèle + stats (depuis le notebook)

model = keras.models.load_model(MODEL_PATH)
data = np.load(NPZ_PATH, allow_pickle=True)

mu = data["mu"]
sigma = data["sigma"]
LABELS = list(data["labels"])
id2label = {i: lab for i, lab in enumerate(LABELS)}



# 3) Paramètres audio/MFCC (doivent matcher le training)

SR = 16000
WINDOW_SEC = 1.0
SAMPLES_1S = int(SR * WINDOW_SEC)

N_MFCC = 40
WIN_LENGTH = int(0.025 * SR)  # 25ms
HOP_LENGTH = int(0.010 * SR)  # 10ms
N_FFT = 512
T_FRAMES = 97

# 4) Paramètres de stabilité (à ajuster)
CONF_TH = 0.65        # si trop strict -> rien ne passe ; si trop faible -> bouge trop
RMS_TH = 0.010        # anti-bruit: ignore si trop silencieux (ajuste selon ton micro)
PRED_EVERY = 0.20     # prédire toutes les 0.20s (~5 fois/sec)
VOTE_N = 5            # vote sur les 5 dernières prédictions valides
VOTE_MIN = 3          # au moins 3/5 identiques pour agir
COOLDOWN_SEC = 0.60   # délai minimum entre 2 actions
STEP = 70             # distance déplacement

# 5) Ring buffer audio (stream)
audio_buffer = deque(maxlen=SAMPLES_1S * 3)  # on garde jusqu'à 3s (pour être safe)
buffer_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    # indata shape: (frames, channels)
    y = indata[:, 0].astype(np.float32)
    with buffer_lock:
        audio_buffer.extend(y)

# 6) Prétraitement + prédiction sur "dernière seconde"
def mfcc_from_last_1s(y1s: np.ndarray) -> np.ndarray:
    # y1s: (SAMPLES_1S,)
    mfcc = librosa.feature.mfcc(
        y=y1s, sr=SR, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        center=False
    ).T  # (T, 40)

    if mfcc.shape[0] < T_FRAMES:
        mfcc = np.pad(mfcc, ((0, T_FRAMES - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:T_FRAMES]

    mfcc = (mfcc - mu) / sigma
    return mfcc.astype(np.float32)


def predict_on_buffer():
    # récupère les dernières 1s
    with buffer_lock:
        if len(audio_buffer) < SAMPLES_1S:
            return "silence", 1.0, 0.0
        y = np.array(list(audio_buffer)[-SAMPLES_1S:], dtype=np.float32)

    # normaliser amplitude + enlever DC
    y = y - np.mean(y)
    y = y / (np.max(np.abs(y)) + 1e-9)

    # RMS gating
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    if rms < RMS_TH:
        return "silence", 1.0, rms

    x = mfcc_from_last_1s(y)[None, ...]   # (1, 97, 40)
    proba = model.predict(x, verbose=0)[0]
    i = int(np.argmax(proba))
    return id2label[i], float(proba[i]), rms
# 7) Thread inférence → envoie commandes vers UI
cmd_queue = queue.Queue()
stop_flag = threading.Event()

def inference_loop():
    last_time = 0.0
    last_action = 0.0
    votes = deque(maxlen=VOTE_N)

    while not stop_flag.is_set():
        now = time.time()
        if now - last_time < PRED_EVERY:
            time.sleep(0.01)
            continue
        last_time = now

        label, conf, rms = predict_on_buffer()

        # ignore unknown/silence
        if label in ("unknown", "silence"):
            votes.clear()
            cmd_queue.put(("status", label, conf, rms, "ignored"))
            continue

        # confidence
        if conf < CONF_TH:
            votes.clear()
            cmd_queue.put(("status", label, conf, rms, "low_conf"))
            continue

        # accumuler votes
        votes.append(label)
        cmd_queue.put(("status", label, conf, rms, f"vote {len(votes)}/{VOTE_N}"))

        if len(votes) < VOTE_N:
            continue

        # majorité
        c = Counter(votes)
        best, count = c.most_common(1)[0]

        if count < VOTE_MIN:
            continue

        # cooldown
        if now - last_action < COOLDOWN_SEC:
            continue

        last_action = now
        votes.clear()
        cmd_queue.put(("cmd", best, conf, rms, f"majority {count}/{VOTE_N}"))

        if best == "stop":
            stop_flag.set()
            break


# 8) Turtle UI (main thread)
screen = turtle.Screen()
screen.title("🎙️ Real-time Voice → Turtle (MFCC + GRU)")
screen.bgcolor("white")
screen.setup(width=1000, height=700)
screen.tracer(0)

pen = turtle.Turtle()
pen.shape("arrow")
pen.color("blue")
pen.pensize(6)
pen.speed(0)
pen.penup()
pen.goto(0, 0)
pen.setheading(0)

ui = turtle.Turtle()
ui.hideturtle()
ui.penup()
ui.goto(-480, 300)
ui.write(
    "Parle: up/down/left/right | on=draw | off=no draw | yes=clear | stop=quit | go=step",
    font=("Arial", 12, "normal"),
)

pred_t = turtle.Turtle()
pred_t.hideturtle()
pred_t.penup()
pred_t.goto(-480, 270)

def show_status(label, conf, rms, note=""):
    pred_t.clear()
    pred_t.write(
        f"Pred: {label} | conf={conf:.2f} | rms={rms:.3f}  ({note})",
        font=("Arial", 14, "bold"),
    )

def apply_command(cmd: str):
    cmd = cmd.lower()
    if cmd == "up":
        pen.setheading(90)
    elif cmd == "down":
        pen.setheading(270)
    elif cmd == "left":
        pen.setheading(180)
    elif cmd == "right":
        pen.setheading(0)
    elif cmd == "on":
        pen.pendown()
    elif cmd == "off":
        pen.penup()
    elif cmd == "go":
        pen.pendown()      # commence à tracer
        pen.forward(STEP)  # avance d'un pas
    elif cmd == "yes":   # commande CLEAR
        pen.clear()
        pen.penup()
        pen.goto(0, 0)
        pen.setheading(0)

def ui_tick():
    # traiter toutes les infos disponibles
    try:
        while True:
            kind, label, conf, rms, note = cmd_queue.get_nowait()
            show_status(label, conf, rms, note)

            if kind == "cmd":
                if label == "stop":
                    stop_flag.set()
                    try:
                        turtle.bye()
                    except Exception:
                        pass
                    return
                apply_command(label)

    except queue.Empty:
        pass

    screen.update()
    if not stop_flag.is_set():
        screen.ontimer(ui_tick, 30)  # ~33 fps


# 9) Run
def main():
    # Stream micro non-bloquant
    stream = sd.InputStream(
        samplerate=SR,
        channels=1,
        callback=audio_callback,
        blocksize=int(0.05 * SR),  # 50ms
    )

    with stream:
        # thread inférence
        t = threading.Thread(target=inference_loop, daemon=True)
        t.start()

        show_status("ready", 1.0, 0.0, "listening...")
        screen.ontimer(ui_tick, 30)
        turtle.mainloop()

    stop_flag.set()

if __name__ == "__main__":
    main()