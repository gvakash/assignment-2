import re, sys, csv
from pathlib import Path

def read_tiers(path):
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    tiers = {}
    for block in re.split(r"\n\s*item \[\d+\]:", txt)[1:]:
        name = re.search(r'name = "(.*?)"', block).group(1)
        ivs = re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', block, re.S)
        tiers[name] = [(float(a), float(b), t) for a, b, t in ivs]
    return tiers

targets = {"dukakis", "wbur's", "melnicove", "politicize", "maffy"}
base, cust = Path(sys.argv[1]), Path(sys.argv[2])

def show(folder, label):
    print(f"--- {label} ---")
    for f in sorted(folder.glob("F2BJRLP*.TextGrid")):
        t = read_tiers(f)
        words = next(v for k, v in t.items() if "word" in k)
        phones = next(v for k, v in t.items() if "phone" in k)
        for a, b, w in words:
            if w.lower() in targets or "j.c" in w.lower() or "sjc" in w.lower():
                ph = [f"{p}({y-x:.2f})" for x, y, p in phones if x >= a - 1e-6 and y <= b + 1e-6 and p]
                print(f"{f.stem} {w:12s} {a:6.2f}-{b:6.2f}s  {' '.join(ph)}")

show(base, "BASELINE (stock dictionary)")
show(cust, "CUSTOM (stock + 5 hand-checked entries)")

print("\n--- alignment_analysis.csv: log-likelihood columns ---")
for folder, label in ((base, "baseline"), (cust, "custom")):
    rows = list(csv.DictReader(open(folder / "alignment_analysis.csv")))
    cols = [c for c in rows[0] if "likelihood" in c.lower()]
    print(label, "columns:", cols)
    for r in rows:
        print("  ", r[list(r)[0]][:34].ljust(34), *[f"{c}={float(r[c]):.2f}" for c in cols if r[c]])
