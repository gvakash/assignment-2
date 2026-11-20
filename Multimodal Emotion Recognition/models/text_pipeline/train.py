"""This script trains the text-only model on the training set, for each of the specified random seeds. It saves the best model checkpoint to disk, and prints a summary of the results.
"""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from common import load_metadata, metrics, seed_everything
from text_common import TextClassifier, TextDataset, collate_text, predict_text

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--patience", type=int, default=8)
args = ap.parse_args()

seed_everything(args.seed)
df = load_metadata()
tr, va = (df.index[df.split == s].tolist() for s in ("train", "val"))
train_dl = DataLoader(TextDataset(df, tr), batch_size=args.bs, shuffle=True, collate_fn=collate_text)
val_dl = DataLoader(TextDataset(df, va), batch_size=args.bs, collate_fn=collate_text)

model = TextClassifier()
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

ck_dir = ROOT / "models" / "text_pipeline" / "checkpoints"; ck_dir.mkdir(exist_ok=True)
(ROOT / "Results").mkdir(exist_ok=True)
log, best, bad = [], (-1, 9e9), 0
for ep in range(1, args.epochs + 1):
    t0 = time.time(); model.train(); tl = tc = n = 0
    for T, lengths, y, _ in train_dl:
        logits, _, _ = model(T, lengths)
        loss = lossf(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * len(y); tc += (logits.argmax(1) == y).sum().item(); n += len(y)
    sched.step()
    v = predict_text(model, val_dl)
    vm = metrics(v["logits"], v["y"])
    vloss = float(nn.functional.cross_entropy(torch.tensor(v["logits"]), torch.tensor(v["y"])))
    log.append(dict(epoch=ep, train_loss=tl / n, train_acc=tc / n, val_loss=vloss, val_acc=vm["acc"], val_f1=vm["macro_f1"]))
    print(f"ep {ep:02d}  train {tl/n:.3f}/{tc/n:.3f}  val {vloss:.3f}/{vm['acc']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if (vm["acc"], -vloss) > (best[0], -best[1]):
        best, bad = (vm["acc"], vloss), 0
        torch.save(dict(state=model.state_dict(), seed=args.seed, epoch=ep, val_acc=vm["acc"]), ck_dir / f"seed{args.seed}.pt")
    else:
        bad += 1
        if bad >= args.patience:
            print("early stop"); break
pd.DataFrame(log).to_csv(ROOT / "Results" / f"text_train_seed{args.seed}.csv", index=False)
print(f"best val acc {best[0]:.3f}  -> checkpoint seed{args.seed}.pt")
