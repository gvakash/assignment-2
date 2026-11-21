"""Shared utilities for the three pipelines: data loading, speech encoder, metrics."""
import random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]      # project folder (holds metadata.csv, features/, Results/)
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
L2I = {l: i for i, l in enumerate(LABELS)}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_metadata():
    df = pd.read_csv(ROOT / "metadata.csv")
    df["y"] = df.emotion.map(L2I)
    return df


def load_mels(df):
    d = torch.load(ROOT / "features" / "speech_logmel.pt")
    assert d["paths"] == df.path.tolist(), "feature cache does not match metadata.csv"
    return d["mels"]


def mel_stats(mels, idx):
    """Per-mel-band mean/std computed on TRAIN clips only."""
    cat = torch.cat([mels[i] for i in idx], dim=1)
    return cat.mean(1, keepdim=True), cat.std(1, keepdim=True) + 1e-5


def spec_augment(x, f_max=8, t_max=20):
    x = x.clone()
    n_mels, T = x.shape
    f = random.randint(0, f_max); f0 = random.randint(0, n_mels - f); x[f0:f0 + f] = 0
    t = random.randint(0, min(t_max, T // 4)); t0 = random.randint(0, T - t); x[:, t0:t0 + t] = 0
    return x


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, mels, df, idx, mean, std, augment=False):
        self.mels, self.y, self.idx = mels, df.y.values, list(idx)
        self.mean, self.std, self.augment = mean, std, augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        x = (self.mels[i] - self.mean) / self.std
        if self.augment:
            x = spec_augment(x)
        return x, int(self.y[i]), i


def collate_speech(batch):
    xs, ys, ids = zip(*batch)
    lengths = torch.tensor([x.shape[1] for x in xs])
    X = torch.zeros(len(xs), xs[0].shape[0], int(lengths.max()))   # 0 = the mean after normalisation
    for k, x in enumerate(xs):
        X[k, :, :x.shape[1]] = x
    return X, lengths, torch.tensor(ys), torch.tensor(ids)


class SpeechEncoder(nn.Module):
    """Temporal block (CNN over the spectrogram) -> contextual block (BiGRU + attention pooling)."""

    def __init__(self, n_mels=64, dim=128, drop=0.3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.cnn = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))   # time and freq both /8
        self.proj = nn.Linear(64 * (n_mels // 8), dim)
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.att = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)
        self.dim = dim

    def forward(self, x, lengths):
        h = self.cnn(x.unsqueeze(1))                         # (B, C, F', T')
        B, C, Fp, Tp = h.shape
        h = F.relu(self.proj(h.permute(0, 3, 1, 2).reshape(B, Tp, C * Fp)))   # (B, T', dim)
        tl = torch.clamp(lengths // 8, min=1, max=Tp)         # valid frames per clip
        mask = torch.arange(Tp, device=h.device)[None, :] < tl[:, None]
        temporal_vec = (h * mask.unsqueeze(-1)).sum(1) / tl.unsqueeze(-1)       # clip vector after temporal block
        packed = pack_padded_sequence(self.drop(h), tl.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=Tp)
        w = torch.softmax(self.att(out).squeeze(-1).masked_fill(~mask, -1e9), dim=1).unsqueeze(-1)
        context_vec = (out * w).sum(1)                        # clip vector after contextual block
        return temporal_vec, context_vec


class SpeechClassifier(nn.Module):
    def __init__(self, n_classes=7, **kw):
        super().__init__()
        self.encoder = SpeechEncoder(**kw)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.encoder.dim, n_classes))

    def forward(self, x, lengths):
        t, c = self.encoder(x, lengths)
        return self.head(c), t, c


@torch.no_grad()
def predict_speech(model, loader):
    model.eval()
    keys = ("logits", "y", "id", "temporal", "context")
    out = {k: [] for k in keys}
    for X, lengths, y, ids in loader:
        logits, t, c = model(X, lengths)
        for k, v in zip(keys, (logits, y, ids, t, c)):
            out[k].append(v)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def metrics(logits, y):
    pred = logits.argmax(1)
    return dict(acc=float((pred == y).mean()), macro_f1=float(f1_score(y, pred, average="macro")))
