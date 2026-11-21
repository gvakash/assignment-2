"""This script splits the dataset into train, val and test splits based on words. It ensures that no word is shared between splits."""
from pathlib import Path
import numpy as np, pandas as pd

META = Path(__file__).resolve().parents[1] / "metadata.csv"

df = pd.read_csv(META)
words = np.array(sorted(df.word.unique()))
rng = np.random.RandomState(42)
rng.shuffle(words)
n = len(words)
n_val, n_test = int(round(0.15 * n)), int(round(0.15 * n))
test_w, val_w, train_w = words[:n_test], words[n_test:n_test + n_val], words[n_test + n_val:]
split_of = {**{w: "test" for w in test_w}, **{w: "val" for w in val_w}, **{w: "train" for w in train_w}}
df["split"] = df.word.map(split_of)
df.to_csv(META, index=False)

# sanity checks
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    assert not set(df[df.split == a].word) & set(df[df.split == b].word), (a, b)
print("words per split:", {s: g.word.nunique() for s, g in df.groupby("split")})
print("clips per split:", df.split.value_counts().to_dict())
print("\nemotion x split:\n", pd.crosstab(df.emotion, df.split).to_string())
print("\nspeaker x split:\n", pd.crosstab(df.speaker, df.split).to_string())
print("\nno word is shared between splits: OK")
