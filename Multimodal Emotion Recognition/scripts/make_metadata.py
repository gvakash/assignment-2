"""Scan data/ for TESS wavs and build metadata.csv (one row per clip)."""
import sys
from pathlib import Path
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
data = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
EMO = {"ps": "pleasant_surprise", "pleasant": "pleasant_surprise",
       "angry": "angry", "disgust": "disgust", "fear": "fear",
       "happy": "happy", "neutral": "neutral", "sad": "sad"}

rows, seen, skipped = [], set(), []
for p in sorted(data.rglob("*.wav")):
    if p.name in seen:               # Kaggle zips sometimes contain the folder twice
        continue
    parts = p.stem.split("_")
    if len(parts) != 3 or parts[2].lower() not in EMO:
        skipped.append(p.name); continue
    seen.add(p.name)
    speaker, word, emo = p.parent.name.split("_")[0].upper(), parts[1].lower(), EMO[parts[2].lower()]
    info = sf.info(str(p))
    rows.append(dict(path=str(p.relative_to(data)), speaker=speaker, word=word,
                     emotion=emo, text=f"say the word {word}",
                     duration_s=round(info.duration, 3), sr=info.samplerate))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "metadata.csv", index=False)
print("clips:", len(df), "| unique words:", df.word.nunique())
print("\nper emotion:\n", df.emotion.value_counts().to_string())
print("\nper speaker:\n", df.speaker.value_counts().to_string())
print("\nsample rates:", df.sr.unique().tolist())
print("duration (s): min %.2f  mean %.2f  max %.2f" % (df.duration_s.min(), df.duration_s.mean(), df.duration_s.max()))
if skipped:
    print("\nSKIPPED (unexpected filename):", skipped[:10], "... total", len(skipped))
