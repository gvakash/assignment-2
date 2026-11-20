"""This script evaluates the text-only model on the test set, for each of the specified random seeds. It saves the embeddings and predictions to disk, and prints a summary of the results.
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from common import LABELS, load_metadata, metrics
from eval_utils import summarise
from text_common import TextClassifier, TextDataset, collate_text, predict_text

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()

df = load_metadata()
te = df.index[df.split == "test"].tolist()
res_dir = ROOT / "Results"; (res_dir / "embeddings").mkdir(parents=True, exist_ok=True)
per_seed, cms, rows = [], [], []
for seed in args.seeds:
    ck = torch.load(ROOT / "models" / "text_pipeline" / "checkpoints" / f"seed{seed}.pt")
    model = TextClassifier(); model.load_state_dict(ck["state"]); model.eval()
    o = predict_text(model, DataLoader(TextDataset(df, te), batch_size=64, collate_fn=collate_text))
    m = metrics(o["logits"], o["y"]); per_seed.append(m)
    pred = o["logits"].argmax(1)
    cms.append(confusion_matrix(o["y"], pred, labels=range(7)))
    print(f"seed {seed}: test acc {m['acc']:.3f}  macro-F1 {m['macro_f1']:.3f}  (best epoch {ck['epoch']}, val acc {ck['val_acc']:.3f})")
    np.savez(res_dir / "embeddings" / f"text_seed{seed}_test.npz", temporal=o["temporal"], context=o["context"],
             y=o["y"], id=o["id"], logits=o["logits"])
    prob = torch.softmax(torch.tensor(o["logits"]), 1).numpy()
    for k in np.where(pred != o["y"])[0]:
        r = df.iloc[o["id"][k]]
        rows.append(dict(seed=seed, path=r.path, speaker=r.speaker, word=r.word, true=LABELS[o["y"][k]],
                         pred=LABELS[pred[k]], confidence=round(float(prob[k, pred[k]]), 3)))
summarise("text", "Text-only", args.seeds, per_seed, cms, rows, res_dir)
