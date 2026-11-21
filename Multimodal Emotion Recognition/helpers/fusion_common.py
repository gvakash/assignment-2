"""Fusion pipeline pieces: paired speech+text dataset, fusion block, full model, prediction helper."""
import torch, torch.nn as nn
from common import SpeechDataset, SpeechEncoder, collate_speech
from text_common import TextEncoder, collate_text, tokenize


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, mels, df, idx, mean, std, augment=False):
        self.sp = SpeechDataset(mels, df, idx, mean, std, augment)
        self.tok = [tokenize(t) for t in df.text.values]

    def __len__(self):
        return len(self.sp)

    def __getitem__(self, k):
        x, y, i = self.sp[k]
        return x, self.tok[i], y, i


def collate_fusion(batch):
    xs, toks, ys, ids = zip(*batch)
    X, lengths, y, i = collate_speech(list(zip(xs, ys, ids)))
    T, tlen, _, _ = collate_text(list(zip(toks, ys, ids)))
    return X, lengths, T, tlen, y, i


class FusionHead(nn.Module):
    """concat: [speech ; text] -> Linear+ReLU.   gated: g*speech + (1-g)*text with a learned gate g."""

    def __init__(self, dim=128, n_classes=7, mode="concat"):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.mix = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU())
        elif mode == "gated":
            self.gate = nn.Linear(2 * dim, dim)
        else:
            raise ValueError(mode)
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, n_classes))

    def forward(self, s, t):
        cat = torch.cat([s, t], dim=1)
        if self.mode == "concat":
            fused, gate = self.mix(cat), torch.zeros(s.shape[0])
        else:
            g = torch.sigmoid(self.gate(cat))
            fused, gate = g * s + (1 - g) * t, g.mean(1)       # gate = how much the speech side is used
        return self.out(fused), fused, gate


class FusionClassifier(nn.Module):
    def __init__(self, n_classes=7, mode="concat", dim=128):
        super().__init__()
        self.speech = SpeechEncoder(dim=dim)
        self.text = TextEncoder(dim=dim)
        self.fusion = FusionHead(dim, n_classes, mode)

    def forward(self, X, lengths, T, tlen):
        s_t, s_c = self.speech(X, lengths)
        _, t_c = self.text(T, tlen)
        logits, fused, gate = self.fusion(s_c, t_c)
        return logits, s_t, s_c, fused, gate


@torch.no_grad()
def predict_fusion(model, loader):
    model.eval()
    keys = ("logits", "y", "id", "speech_temporal", "speech_context", "fused", "gate")
    out = {k: [] for k in keys}
    for X, lengths, T, tlen, y, ids in loader:
        logits, s_t, s_c, fused, gate = model(X, lengths, T, tlen)
        for k, v in zip(keys, (logits, y, ids, s_t, s_c, fused, gate)):
            out[k].append(v)
    return {k: torch.cat(v).numpy() for k, v in out.items()}
