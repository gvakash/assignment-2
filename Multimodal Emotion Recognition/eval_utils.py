"""Shared reporting for test scripts: summary numbers, confusion matrix figure, misclassified csv, json."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import LABELS


def summarise(name, title, seeds, per_seed, cms, rows, res_dir):
    accs = [m["acc"] for m in per_seed]; f1s = [m["macro_f1"] for m in per_seed]
    print(f"\n{title.upper()}  test accuracy {np.mean(accs):.3f} +- {np.std(accs):.3f}   "
          f"macro-F1 {np.mean(f1s):.3f} +- {np.std(f1s):.3f}   (seeds {seeds})")
    cm = np.sum(cms, 0); recall = cm.diagonal() / cm.sum(1)
    print("per-emotion recall (all seeds pooled):")
    for l, r in zip(LABELS, recall):
        print(f"  {l:18s} {r:.3f}")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(7)); ax.set_yticklabels(LABELS)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title}: confusion matrix (test, all seeds)")
    plt.tight_layout(); plt.savefig(res_dir / f"{name}_confusion.png", dpi=130); plt.close()
    pd.DataFrame(rows).to_csv(res_dir / f"{name}_misclassified.csv", index=False)
    json.dump(dict(seeds=seeds, per_seed=per_seed, acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                   f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
                   recall=dict(zip(LABELS, map(float, recall)))),
              open(res_dir / f"{name}_results.json", "w"), indent=2)
    print(f"saved Results/{name}_confusion.png, {name}_misclassified.csv, {name}_results.json")
