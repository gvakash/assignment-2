"""This script extracts log-mel spectrograms from speech data."""
import time
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch, torchaudio
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR, N_FFT, WIN, HOP, N_MELS = 16000, 512, 400, 160, 64   # 25 ms window, 10 ms hop
TOP_DB, PAD_S = 35, 0.05                                  # trim below peak-35 dB, keep 50 ms
DATA, OUT = Path("data"), Path("features")
OUT.mkdir(exist_ok=True); Path("Results").mkdir(exist_ok=True)

df = pd.read_csv("metadata.csv")
mel_fn = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                                              n_mels=N_MELS, f_min=20, f_max=8000)

def load(path):
    wav, sr = sf.read(DATA / path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = torch.from_numpy(wav)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav / (wav.abs().max() + 1e-8)

def trim(wav):
    if len(wav) < WIN:
        return wav
    rms = wav.unfold(0, WIN, HOP).pow(2).mean(1).sqrt()
    idx = torch.nonzero(rms > rms.max() * 10 ** (-TOP_DB / 20)).flatten()
    pad = int(PAD_S * SR)
    start = max(int(idx[0]) * HOP - pad, 0)
    end = min(int(idx[-1]) * HOP + WIN + pad, len(wav))
    return wav[start:end]

mels, raw_d, trim_d = [], [], []
t0 = time.time()
for p in tqdm(df.path, desc="extracting"):
    w = load(p)
    wt = trim(w)
    raw_d.append(len(w) / SR); trim_d.append(len(wt) / SR)
    mels.append(torch.log(mel_fn(wt) + 1e-6))            

torch.save({"mels": mels, "paths": df.path.tolist(), "sr": SR, "hop": HOP, "n_mels": N_MELS},
           OUT / "speech_logmel.pt")

frames = np.array([m.shape[1] for m in mels])
print(f"\nclips: {len(mels)}  time: {time.time() - t0:.0f}s")
print(f"duration raw  : mean {np.mean(raw_d):.2f}s")
print(f"duration trimmed: mean {np.mean(trim_d):.2f}s  min {np.min(trim_d):.2f}  max {np.max(trim_d):.2f}")
print(f"frames: min {frames.min()}  mean {frames.mean():.0f}  max {frames.max()}   (10 ms each)")
print("feature shape example:", tuple(mels[0].shape))

# one example spectrogram per emotion 
word = "back" if "back" in set(df.word) else df.word.iloc[0]
fig, axes = plt.subplots(2, 4, figsize=(14, 5.5)); axes = axes.ravel()
for ax, e in zip(axes, sorted(df.emotion.unique())):
    i = df.index[(df.emotion == e) & (df.word == word) & (df.speaker == "OAF")][0]
    ax.imshow(mels[i].numpy(), origin="lower", aspect="auto")
    ax.set_title(f"{e} ('{word}', OAF)"); ax.set_xlabel("frame (10 ms)"); ax.set_ylabel("mel band")
axes[-1].axis("off")
plt.tight_layout(); plt.savefig("Results/example_logmels.png", dpi=120)
print("saved Results/example_logmels.png")
