# scripts/peek.py
import sys, json
sys.path.append(".")

from app.retriever import retrieve

q = "theft"
if len(sys.argv) > 1:
    q = " ".join(sys.argv[1:])

docs = retrieve(q)
print(f"TOP {len(docs)} for: {q!r}")
for d in docs:
    print(f"\n[{d['rank']}] score={d['score']:.3f} source={d['source']}")
    print(d['text'][:400].replace("\n"," "))
