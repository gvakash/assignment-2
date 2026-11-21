"""This script trains and evaluates the speech and fusion models in a cross-speaker setting, i.e. training on one actress and testing on the other. It produces a CSV file with the results, a CSV file with the misclassified examples, a CSV file with per-emotion recall, and a PNG file with confusion matrices."""
import argparse, copy, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "helpers"))
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from common import (LABELS, SpeechClassifier, SpeechDataset, collate_speech, load_mels, load_metadata,
                    mel_stats, metrics, predict_speech, seed_everything)
from fusion_common import FusionClassifier, FusionDataset, collate_fusion, predict_fusion

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3)
args = ap.parse_args()

R = ROOT / "Results"; R.mkdir(exist_ok=True)
df, mels = load_metadata(), None
mels = load_mels(df)


def make_split(train_spk):
    d = df[df.speaker == train_spk]
    words = np.array(sorted(d.word.unique())); np.random.RandomState(7).shuffle(words)
    val_w = set(words[:30])
    tr = d.index[~d.word.isin(val_w)].tolist(); va = d.index[d.word.isin(val_w)].tolist()
    te = df.index[df.speaker != train_spk].tolist()
    return tr, va, te


def run(kind, train_spk, seed):
    seed_everything(seed)
    tr, va, te = make_split(train_spk)
    mean, std = mel_stats(mels, tr)
    if kind == "speech":
        DS, collate, predict, model = SpeechDataset, collate_speech, predict_speech, SpeechClassifier()
        fwd, ygetter = (lambda b: model(b[0], b[1])[0]), (lambda b: b[2])
    else:
        DS, collate, predict, model = FusionDataset, collate_fusion, predict_fusion, FusionClassifier(mode="concat")
        fwd, ygetter = (lambda b: model(b[0], b[1], b[2], b[3])[0]), (lambda b: b[4])
    mk = lambda idx, aug, shuf: DataLoader(DS(mels, df, idx, mean, std, aug), batch_size=args.bs, shuffle=shuf, collate_fn=collate)
    train_dl, val_dl, test_dl = mk(tr, True, True), mk(va, False, False), mk(te, False, False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state, bad = (-1, 9e9), None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for b in train_dl:
            loss = lossf(fwd(b), ygetter(b)); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        v = predict(model, val_dl); vm = metrics(v["logits"], v["y"])
        vl = float(nn.functional.cross_entropy(torch.tensor(v["logits"]), torch.tensor(v["y"])))
        if (vm["acc"], -vl) > (best[0], -best[1]):
            best, best_state, bad = (vm["acc"], vl), copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    o = predict(model, test_dl)
    pred = o["logits"].argmax(1); prob = torch.softmax(torch.tensor(o["logits"]), 1).numpy()
    rows = []
    for k in np.where(pred != o["y"])[0]:
        r = df.iloc[o["id"][k]]
        rows.append(dict(model=kind, train_speaker=train_spk, seed=seed, path=r.path, speaker=r.speaker, word=r.word,
                         true=LABELS[o["y"][k]], pred=LABELS[pred[k]], confidence=round(float(prob[k, pred[k]]), 3)))
    return metrics(o["logits"], o["y"]), confusion_matrix(o["y"], pred, labels=range(7)), rows, ep, best[0]


results, cms, miss = [], {}, []
for train_spk in ("OAF", "YAF"):
    for kind in ("speech", "fusion"):
        for seed in args.seeds:
            t0 = time.time()
            m, cm, rows, ep, vacc = run(kind, train_spk, seed)
            miss += rows; cms[(train_spk, kind)] = cms.get((train_spk, kind), 0) + cm
            results.append(dict(model=kind, train=train_spk, test="YAF" if train_spk == "OAF" else "OAF", seed=seed,
                                test_acc=m["acc"], macro_f1=m["macro_f1"], epochs=ep, val_acc_same_speaker=vacc))
            print(f"{kind:6s} train {train_spk} -> test {'YAF' if train_spk=='OAF' else 'OAF'}  seed {seed}: "
                  f"acc {m['acc']:.3f}  F1 {m['macro_f1']:.3f}  (stopped ep {ep}, {time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(results); res.to_csv(R / "cross_speaker_runs.csv", index=False)
summ = (res.groupby(["model", "train", "test"])
        .agg(acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"), f1_mean=("macro_f1", "mean")).reset_index())
summ["acc_%"] = summ.apply(lambda r: f"{100*r.acc_mean:.2f} +- {100*(0 if np.isnan(r.acc_std) else r.acc_std):.2f}", axis=1)
summ["macro_F1_%"] = (100 * summ.f1_mean).round(2)
summ[["model", "train", "test", "acc_%", "macro_F1_%"]].to_csv(R / "cross_speaker_table.csv", index=False)
print("\n=== cross-speaker results (test on the OTHER actress; mean +- std over seeds) ===")
print(summ[["model", "train", "test", "acc_%", "macro_F1_%"]].to_string(index=False))
pd.DataFrame(miss).to_csv(R / "cross_speaker_misclassified.csv", index=False)

rec = pd.DataFrame({f"{k[1]} {k[0]}->{'YAF' if k[0]=='OAF' else 'OAF'}": np.round(100 * cm.diagonal() / cm.sum(1), 1)
                    for k, cm in cms.items()}, index=LABELS)
rec.to_csv(R / "cross_speaker_per_emotion.csv")
print("\n=== per-emotion recall % ===\n" + rec.to_string())

fig, axes = plt.subplots(1, 4, figsize=(21, 5.2))
for ax, (k, cm) in zip(axes, cms.items()):
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_xticklabels(LABELS, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(7)); ax.set_yticklabels(LABELS, fontsize=8)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{k[1]}: train {k[0]} -> test {'YAF' if k[0]=='OAF' else 'OAF'}", fontsize=10)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
plt.tight_layout(); plt.savefig(R / "cross_speaker_confusion.png", dpi=120)
print("\nsaved Results/cross_speaker_{runs,table,per_emotion,misclassified}.csv and cross_speaker_confusion.png")
