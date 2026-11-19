"""Build an MFA corpus (wav + .lab pairs) from raw/ (wav + .txt/.TXT transcripts).

Usage:  python scripts/prepare_corpus.py raw corpus
"""
import re, shutil, sys
from pathlib import Path

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

for wav in sorted(raw.glob("*.wav")):
    txt = next((p for p in raw.iterdir()
                if p.stem == wav.stem and p.suffix.lower() == ".txt"), None)
    if txt is None:
        print("NO TRANSCRIPT for", wav.name); continue
    text = txt.read_text(encoding="utf-8", errors="replace")
    clean = re.sub(r"-\s+", " ", text)          # "seventy- six" -> "seventy six"
    clean = re.sub(r"\s+", " ", clean).strip().upper()
    shutil.copy(wav, out / wav.name)
    (out / (wav.stem + ".lab")).write_text(clean + "\n", encoding="utf-8")
    print(f"{wav.stem}: {len(clean.split())} words")
    print("   ", clean[:90] + ("..." if len(clean) > 90 else ""))
