import re, sys
from pathlib import Path

def read_tiers(path):
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    tiers = {}
    for block in re.split(r"\n\s*item \[\d+\]:", txt)[1:]:
        name = re.search(r'name = "(.*?)"', block).group(1)
        ivs = re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', block, re.S)
        tiers[name] = [(float(a), float(b), t) for a, b, t in ivs]
    return tiers

for f in sorted(Path(sys.argv[1]).glob("*.TextGrid")):
    t = read_tiers(f)
    words = next(v for k, v in t.items() if "word" in k)
    phones = next(v for k, v in t.items() if "phone" in k)
    for a, b, p in phones:
        if p == "spn":
            mid = (a + b) / 2
            w = next((x[2] for x in words if x[0] <= mid <= x[1]), "?")
            print(f"{f.stem:32s} {a:7.2f}-{b:7.2f}s  ({b-a:.2f}s)  word: {w}")
