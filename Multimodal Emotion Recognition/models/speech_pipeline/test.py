"""This script evaluates the speech-only model on the test set, for each of the specified random seeds. It saves the embeddings and predictions to disk, and prints a summary of the results."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from common import LABELS, SpeechClassifier, SpeechDataset, collate_speech, load_mels, load_metadata, metrics, predict_speech

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()

df = load_metadata(); mels = load_mels(df)
te = df.index[df.split == "test"].tolist()
res_dir = ROOT / "Results"; (res_dir / "embeddings").mkdir(parents=True, exist_ok=True)

per_seed, cms, rows = [], [], []
for seed in args.seeds:
    ck = torch.load(ROOT / "models" / "speech_pipeline" / "checkpoints" / f"seed{seed}.pt")
    model = SpeechClassifier(); model.load_state_dict(ck["state"]); model.eval()
    dl = DataLoader(SpeechDataset(mels, df, te, ck["mean"], ck["std"]), batch_size=64, collate_fn=collate_speech)
    o = predict_speech(model, dl)
    m = metrics(o["logits"], o["y"]); per_seed.append(m)
    pred = o["logits"].argmax(1)
    cm = confusion_matrix(o["y"], pred, labels=range(7)); cms.append(cm)
    print(f"seed {seed}: test acc {m['acc']:.3f}  macro-F1 {m['macro_f1']:.3f}  (best epoch {ck['epoch']}, val acc {ck['val_acc']:.3f})")
    np.savez(res_dir / "embeddings" / f"speech_seed{seed}_test.npz", temporal=o["temporal"], context=o["context"],
             y=o["y"], id=o["id"], logits=o["logits"])
    prob = torch.softmax(torch.tensor(o["logits"]), 1).numpy()
    for k in np.where(pred != o["y"])[0]:
        r = df.iloc[o["id"][k]]
        rows.append(dict(seed=seed, path=r.path, speaker=r.speaker, word=r.word, true=LABELS[o["y"][k]],
                         pred=LABELS[pred[k]], confidence=round(float(prob[k, pred[k]]), 3)))

accs = [m["acc"] for m in per_seed]; f1s = [m["macro_f1"] for m in per_seed]
print(f"\nSPEECH-ONLY  test accuracy {np.mean(accs):.3f} +- {np.std(accs):.3f}   macro-F1 {np.mean(f1s):.3f} +- {np.std(f1s):.3f}   (seeds {args.seeds})")
cm = np.sum(cms, 0)
recall = cm.diagonal() / cm.sum(1)
print("per-emotion recall (all seeds pooled):")
for l, r in zip(LABELS, recall):
    print(f"  {l:18s} {r:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(7)); ax.set_xticklabels(LABELS, rotation=45, ha="right"); ax.set_yticks(range(7)); ax.set_yticklabels(LABELS)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Speech-only: confusion matrix (test, all seeds)")
plt.tight_layout(); plt.savefig(res_dir / "speech_confusion.png", dpi=130)

pd.DataFrame(rows).to_csv(res_dir / "speech_misclassified.csv", index=False)
json.dump(dict(seeds=args.seeds, per_seed=per_seed, acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
               f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)), recall=dict(zip(LABELS, map(float, recall)))),
          open(res_dir / "speech_results.json", "w"), indent=2)
print("saved Results/speech_confusion.png, speech_misclassified.csv, speech_results.json")
