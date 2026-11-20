"""Text pipeline pieces: character tokenisation, text encoder, dataset, prediction helper."""
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ALPHABET = "abcdefghijklmnopqrstuvwxyz '-"
C2I = {c: i + 2 for i, c in enumerate(ALPHABET)}      # 0 = padding, 1 = unknown character


def tokenize(text):
    """Preprocessing: lowercase, strip, map each character to an id (handles words never seen in training)."""
    return [C2I.get(c, 1) for c in str(text).lower().strip()]


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df, idx):
        self.tok = [tokenize(t) for t in df.text.values]
        self.y, self.idx = df.y.values, list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        return self.tok[i], int(self.y[i]), i


def collate_text(batch):
    toks, ys, ids = zip(*batch)
    lengths = torch.tensor([len(t) for t in toks])
    T = torch.zeros(len(toks), int(lengths.max()), dtype=torch.long)
    for k, t in enumerate(toks):
        T[k, :len(t)] = torch.tensor(t)
    return T, lengths, torch.tensor(ys), torch.tensor(ids)


class TextEncoder(nn.Module):
    """Temporal block (1-D CNN over characters = letter n-grams) -> contextual block (BiGRU + attention)."""

    def __init__(self, emb=32, dim=128, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(len(ALPHABET) + 2, emb, padding_idx=0)
        self.conv = nn.Sequential(nn.Conv1d(emb, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(64, dim, 3, padding=1), nn.ReLU())
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.att = nn.Linear(dim, 1)
        self.drop = nn.Dropout(drop)
        self.dim = dim

    def forward(self, tok, lengths):
        h = self.conv(self.emb(tok).transpose(1, 2)).transpose(1, 2)        # (B, L, dim)
        L = h.shape[1]
        mask = torch.arange(L, device=h.device)[None, :] < lengths[:, None]
        temporal_vec = (h * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        packed = pack_padded_sequence(self.drop(h), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        w = torch.softmax(self.att(out).squeeze(-1).masked_fill(~mask, -1e9), dim=1).unsqueeze(-1)
        return temporal_vec, (out * w).sum(1)


class TextClassifier(nn.Module):
    def __init__(self, n_classes=7, **kw):
        super().__init__()
        self.encoder = TextEncoder(**kw)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.encoder.dim, n_classes))

    def forward(self, tok, lengths):
        t, c = self.encoder(tok, lengths)
        return self.head(c), t, c


@torch.no_grad()
def predict_text(model, loader):
    model.eval()
    keys = ("logits", "y", "id", "temporal", "context")
    out = {k: [] for k in keys}
    for T, lengths, y, ids in loader:
        logits, t, c = model(T, lengths)
        for k, v in zip(keys, (logits, y, ids, t, c)):
            out[k].append(v)
    return {k: torch.cat(v).numpy() for k, v in out.items()}
