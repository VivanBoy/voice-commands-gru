import numpy as np

big = r"data/processed/speech_mfcc40_T97.npz"
d = np.load(big, allow_pickle=True)

np.savez_compressed(
    r"data/processed/stats_mfcc40_T97.npz",
    mu=d["mu"], sigma=d["sigma"], labels=d["labels"]
)

print("✅ created: data/processed/stats_mfcc40_T97.npz")