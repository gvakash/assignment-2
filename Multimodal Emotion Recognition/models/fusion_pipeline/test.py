"""This script evaluates the fusion model on the test set, for each of the specified random seeds. It saves the embeddings and predictions to disk, and prints a summary of the results."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from common import LABELS, load_mels, load_metadata, metrics
from eval_utils import summarise
from fusion_common import FusionClassifier, FusionDataset, collate_fusion, predict_fusion

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["concat", "gated"], default="concat")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()

df = load_metadata(); mels = load_mels(df)
te = df.index[df.split == "test"].tolist()
res_dir = ROOT / "Results"; (res_dir / "embeddings").mkdir(parents=True, exist_ok=True)
per_seed, cms, rows, gates = [], [], [], []
for seed in args.seeds:
    ck = torch.load(ROOT / "models" / "fusion_pipeline" / "checkpoints" / f"{args.mode}_seed{seed}.pt")
    model = FusionClassifier(mode=args.mode); model.load_state_dict(ck["state"]); model.eval()
    dl = DataLoader(FusionDataset(mels, df, te, ck["mean"], ck["std"]), batch_size=64, collate_fn=collate_fusion)
    o = predict_fusion(model, dl)
    m = metrics(o["logits"], o["y"]); per_seed.append(m)
    pred = o["logits"].argmax(1)
    cms.append(confusion_matrix(o["y"], pred, labels=range(7)))
    gates.append(float(o["gate"].mean()))
    print(f"seed {seed}: test acc {m['acc']:.3f}  macro-F1 {m['macro_f1']:.3f}  (best epoch {ck['epoch']}, val acc {ck['val_acc']:.3f})"
          + (f"  mean speech-gate {gates[-1]:.2f}" if args.mode == "gated" else ""))
    np.savez(res_dir / "embeddings" / f"fusion_{args.mode}_seed{seed}_test.npz", speech_temporal=o["speech_temporal"],
             speech_context=o["speech_context"], fused=o["fused"], gate=o["gate"], y=o["y"], id=o["id"], logits=o["logits"])
    prob = torch.softmax(torch.tensor(o["logits"]), 1).numpy()
    for k in np.where(pred != o["y"])[0]:
        r = df.iloc[o["id"][k]]
        rows.append(dict(seed=seed, path=r.path, speaker=r.speaker, word=r.word, true=LABELS[o["y"][k]],
                         pred=LABELS[pred[k]], confidence=round(float(prob[k, pred[k]]), 3)))
summarise(f"fusion_{args.mode}", f"Fusion ({args.mode})", args.seeds, per_seed, cms, rows, res_dir)
