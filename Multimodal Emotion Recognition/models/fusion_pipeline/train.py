""" This script trains the fusion model on the training set, for a specified random seed. It saves the best checkpoint to disk, and logs the training and validation metrics to a CSV file."""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from common import load_mels, load_metadata, mel_stats, metrics, seed_everything
from fusion_common import FusionClassifier, FusionDataset, collate_fusion, predict_fusion

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mode", choices=["concat", "gated"], default="concat")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--patience", type=int, default=10)
args = ap.parse_args()

seed_everything(args.seed)
df = load_metadata(); mels = load_mels(df)
tr, va = (df.index[df.split == s].tolist() for s in ("train", "val"))
mean, std = mel_stats(mels, tr)
mk = lambda idx, aug, shuf: DataLoader(FusionDataset(mels, df, idx, mean, std, aug), batch_size=args.bs,
                                        shuffle=shuf, collate_fn=collate_fusion)
train_dl, val_dl = mk(tr, True, True), mk(va, False, False)

model = FusionClassifier(mode=args.mode)
print("mode:", args.mode, "| parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

ck_dir = ROOT / "models" / "fusion_pipeline" / "checkpoints"; ck_dir.mkdir(exist_ok=True)
(ROOT / "Results").mkdir(exist_ok=True)
tag = f"{args.mode}_seed{args.seed}"
log, best, bad = [], (-1, 9e9), 0
for ep in range(1, args.epochs + 1):
    t0 = time.time(); model.train(); tl = tc = n = 0
    for X, lengths, T, tlen, y, _ in train_dl:
        logits = model(X, lengths, T, tlen)[0]
        loss = lossf(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * len(y); tc += (logits.argmax(1) == y).sum().item(); n += len(y)
    sched.step()
    v = predict_fusion(model, val_dl)
    vm = metrics(v["logits"], v["y"])
    vloss = float(nn.functional.cross_entropy(torch.tensor(v["logits"]), torch.tensor(v["y"])))
    log.append(dict(epoch=ep, train_loss=tl / n, train_acc=tc / n, val_loss=vloss, val_acc=vm["acc"], val_f1=vm["macro_f1"]))
    print(f"ep {ep:02d}  train {tl/n:.3f}/{tc/n:.3f}  val {vloss:.3f}/{vm['acc']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if (vm["acc"], -vloss) > (best[0], -best[1]):
        best, bad = (vm["acc"], vloss), 0
        torch.save(dict(state=model.state_dict(), mean=mean, std=std, seed=args.seed, mode=args.mode,
                        epoch=ep, val_acc=vm["acc"]), ck_dir / f"{tag}.pt")
    else:
        bad += 1
        if bad >= args.patience:
            print("early stop"); break
pd.DataFrame(log).to_csv(ROOT / "Results" / f"fusion_{tag.split('_seed')[0]}_train_seed{args.seed}.csv", index=False)
print(f"best val acc {best[0]:.3f}  -> checkpoint {tag}.pt")
