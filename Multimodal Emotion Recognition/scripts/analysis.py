"""This script produces the analysis tables and figures for the paper. It requires that the models have already been trained and evaluated (see train.py)."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "helpers"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from common import LABELS, load_mels, load_metadata

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0, help="seed used for the cluster plots")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--fusion", default="concat", choices=["concat", "gated"])
args = ap.parse_args()
R = ROOT / "Results"
df = load_metadata()
fus = f"fusion_{args.fusion}"

# ---- 1. accuracy table + per-emotion recall --------------------------------------------------
names = (("speech", "Speech-only"), ("text", "Text-only"), ("fusion_concat", "Fusion (concat)"), ("fusion_gated", "Fusion (gated)"))
rows, recall = [], {}
for name, label in names:
    f = R / f"{name}_results.json"
    if not f.exists():
        continue
    j = json.load(open(f))
    rows.append({"model": label, "test_accuracy_%": f"{100*j['acc_mean']:.2f} +- {100*j['acc_std']:.2f}",
                 "macro_F1_%": f"{100*j['f1_mean']:.2f} +- {100*j['f1_std']:.2f}",
                 **{f"acc_seed{s}_%": round(100 * m["acc"], 2) for s, m in zip(j["seeds"], j["per_seed"])}})
    recall[label] = {k: round(100 * v, 1) for k, v in j["recall"].items()}
tab = pd.DataFrame(rows); tab.to_csv(R / "accuracy_table.csv", index=False)
rec = pd.DataFrame(recall); rec.to_csv(R / "per_emotion_recall.csv")
print("\n=== accuracy table (test, mean +- std over seeds) ===\n" + tab.to_string(index=False))
print("\n=== per-emotion recall % (pooled over seeds) ===\n" + rec.to_string())

# ---- 2. does fusion change anything vs speech-only? ------------------------------------------
print(f"\n=== speech-only vs {fus}: clip-by-clip ===")
for s in args.seeds:
    a, b = (np.load(R / "embeddings" / f"{n}_seed{s}_test.npz") for n in ("speech", fus))
    assert (a["id"] == b["id"]).all()
    ca, cb = a["logits"].argmax(1) == a["y"], b["logits"].argmax(1) == b["y"]
    print(f"seed {s}: fusion fixed {int((~ca & cb).sum())} speech errors, broke {int((ca & ~cb).sum())} correct clips, "
          f"both wrong {int((~ca & ~cb).sum())}")
    if args.fusion == "gated":
        print(f"         mean speech-gate {b['gate'].mean():.3f}  (1.0 = ignores text entirely)")

# ---- 3. failure cases ------------------------------------------------------------------------
frames = []
for name in ("speech", "fusion_concat", "fusion_gated"):
    f = R / f"{name}_misclassified.csv"
    if f.exists():
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if len(d):
            d["model"] = name; frames.append(d)
if frames:
    allf = pd.concat(frames)
    g = (allf.groupby(["path", "true"])
         .agg(predicted_as=("pred", lambda x: ", ".join(sorted(set(x)))), n_failures=("pred", "size"),
              max_confidence=("confidence", "max"), models=("model", lambda x: ", ".join(sorted(set(x)))))
         .reset_index().sort_values("n_failures", ascending=False))
    g.to_csv(R / "failure_cases.csv", index=False)
    print(f"\n=== failure cases: {len(g)} distinct clips ===\n" + g.to_string(index=False))
    mels = load_mels(df)
    top = g.head(6)
    fig, axes = plt.subplots(2, len(top), figsize=(3.8 * len(top), 6.2), squeeze=False)
    for c, r in enumerate(top.itertuples()):
        i = df.index[df.path == r.path][0]; row = df.loc[i]
        pred = r.predicted_as.split(", ")[0]
        j = df.index[(df.speaker == row.speaker) & (df.word == row.word) & (df.emotion == pred)][0]
        axes[0, c].imshow(mels[i].numpy(), origin="lower", aspect="auto")
        axes[0, c].set_title(f"{row.speaker} '{row.word}'\nTRUE {row.emotion}", fontsize=9)
        axes[1, c].imshow(mels[j].numpy(), origin="lower", aspect="auto")
        axes[1, c].set_title(f"same speaker+word, {pred}", fontsize=9)
    plt.tight_layout(); plt.savefig(R / "failure_spectrograms.png", dpi=120); plt.close()
else:
    print("\nno misclassified clips found")

# ---- 4. cluster separability (t-SNE) at the three blocks --------------------------------------
sp = np.load(R / "embeddings" / f"speech_seed{args.seed}_test.npz")
fu = np.load(R / "embeddings" / f"{fus}_seed{args.seed}_test.npz")
y, spk = sp["y"], df.speaker.values[sp["id"]]
blocks = [("Temporal block\n(speech CNN)", sp["temporal"]), ("Contextual block\n(speech BiGRU + attention)", sp["context"]),
          (f"Fusion block\n({args.fusion})", fu["fused"])]
fig, axes = plt.subplots(2, 3, figsize=(16, 9.5)); cmap = plt.get_cmap("tab10"); scores = []
for j, (title, Z) in enumerate(blocks):
    P = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
    se, ss = silhouette_score(Z, y), silhouette_score(Z, spk)
    scores.append(dict(block=title.replace("\n", " "), silhouette_by_emotion=round(se, 3), silhouette_by_speaker=round(ss, 3)))
    for k, lab in enumerate(LABELS):
        m = y == k; axes[0, j].scatter(P[m, 0], P[m, 1], s=10, color=cmap(k), label=lab)
    axes[0, j].set_title(f"{title}\nsilhouette (emotion) = {se:.2f}")
    for k, sname in enumerate(["OAF", "YAF"]):
        m = spk == sname; axes[1, j].scatter(P[m, 0], P[m, 1], s=10, color=cmap(k + 7), label=sname)
    axes[1, j].set_title(f"same points coloured by speaker\nsilhouette (speaker) = {ss:.2f}")
    for a in axes[:, j]:
        a.set_xticks([]); a.set_yticks([])
axes[0, 0].legend(markerscale=2, fontsize=8); axes[1, 0].legend(markerscale=2, fontsize=8)
plt.tight_layout(); plt.savefig(R / "cluster_tsne.png", dpi=130); plt.close()
sc = pd.DataFrame(scores); sc.to_csv(R / "cluster_scores.csv", index=False)
print("\n=== cluster separability (silhouette on the raw embeddings; higher = better separated) ===\n" + sc.to_string(index=False))
print("\nsaved: accuracy_table.csv, per_emotion_recall.csv, failure_cases.csv, failure_spectrograms.png, cluster_tsne.png, cluster_scores.csv")
