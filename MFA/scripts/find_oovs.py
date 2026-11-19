import re, sys, glob
from pathlib import Path
from collections import Counter, defaultdict

corpus = Path(sys.argv[1])
dict_path = Path(sys.argv[2]).expanduser()

vocab = set()
for line in dict_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if line.strip():
        vocab.add(re.split(r"\s+", line.strip())[0].lower())
print(f"dictionary: {dict_path.name}, {len(vocab)} entries")

counts, where = Counter(), defaultdict(set)
for lab in sorted(corpus.glob("*.lab")):
    for tok in lab.read_text(encoding="utf-8").split():
        w = re.sub(r"^[^\w']+|[^\w']+$", "", tok.lower())
        if w and w not in vocab:
            counts[w] += 1
            where[w].add(lab.stem)

print(f"\nOOV types: {len(counts)}   OOV tokens: {sum(counts.values())}\n")
for w, c in sorted(counts.items()):
    print(f"{w:20s} x{c}   {', '.join(sorted(where[w]))}")
