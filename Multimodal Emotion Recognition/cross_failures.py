"""Cross-speaker failure cases: the 5 most common error patterns of the speech model, each with its most
confidently wrong example, drawn next to the TRAINING actress's recordings of the same word.
Usage: python cross_failures.py"""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_mels, load_metadata

R = Path(__file__).resolve().parent / "Results"
d = pd.read_csv(R / "cross_speaker_misclassified.csv"); d = d[d.model == "speech"]
top = (d.groupby(["train_speaker", "true", "pred"]).size().reset_index(name="n")
       .sort_values("n", ascending=False).head(5))
df, mels = load_metadata(), None
mels = load_mels(df)

fig, axes = plt.subplots(len(top), 3, figsize=(13, 3.1 * len(top)), squeeze=False)
out = []
for r, t in enumerate(top.itertuples()):
    g = d[(d.train_speaker == t.train_speaker) & (d.true == t.true) & (d.pred == t.pred)]
    ex = g.sort_values("confidence", ascending=False).iloc[0]
    i = df.index[df.path == ex.path][0]
    find = lambda emo: df.index[(df.speaker == t.train_speaker) & (df.word == ex.word) & (df.emotion == emo)][0]
    panels = [(i, f"TEST {ex.speaker} '{ex.word}': true {t.true}\npredicted {t.pred} (confidence {ex.confidence})"),
              (find(t.true), f"TRAIN speaker {t.train_speaker}, same word, {t.true}"),
              (find(t.pred), f"TRAIN speaker {t.train_speaker}, same word, {t.pred}")]
    for c, (k, title) in enumerate(panels):
        axes[r, c].imshow(mels[k].numpy(), origin="lower", aspect="auto"); axes[r, c].set_title(title, fontsize=8)
    out.append(dict(train_speaker=t.train_speaker, true=t.true, predicted=t.pred, n_errors_all_seeds=t.n,
                    example_clip=ex.path, example_confidence=ex.confidence, example_seed=ex.seed))
plt.tight_layout(); plt.savefig(R / "cross_speaker_failure_cases.png", dpi=115)
res = pd.DataFrame(out); res.to_csv(R / "cross_speaker_failure_cases.csv", index=False)
print(res.to_string(index=False))
print("\nsaved Results/cross_speaker_failure_cases.png and .csv")
