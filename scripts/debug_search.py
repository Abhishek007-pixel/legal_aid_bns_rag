# scripts/debug_search.py
import json, sys, re
from pathlib import Path

q = (sys.argv[1] if len(sys.argv) > 1 else "theft").lower()
chunks = Path("data/processed/chunks.jsonl")
hits = []
with open(chunks, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        r = json.loads(line)
        if q in r["text"].lower():
            snippet = re.sub(r"\s+", " ", r["text"])[:240]
            hits.append((r["source"], r.get("title", ""), snippet))

print(f"query={q!r} hits={len(hits)}")
for src, title, snip in hits[:20]:
    print(f"- {src} | {title}\n  {snip}\n")
