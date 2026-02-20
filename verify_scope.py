import json
from collections import Counter

lines = [json.loads(l) for l in open("data/index/meta.jsonl", encoding="utf-8")]
scopes = Counter(d.get("scope", "MISSING") for d in lines)
has_scope = sum(1 for d in lines if "scope" in d)
has_act = sum(1 for d in lines if "act_name" in d)
has_jur = sum(1 for d in lines if "jurisdiction" in d)

print(f"Total chunks: {len(lines)}")
print(f"Has scope field: {has_scope}/{len(lines)}")
print(f"Has act_name field: {has_act}/{len(lines)}")
print(f"Has jurisdiction field: {has_jur}/{len(lines)}")
print(f"Scope distribution: {dict(scopes)}")
print()

# Show a sample chunk from each scope
shown = set()
for d in lines:
    s = d.get("scope")
    if s and s not in shown:
        shown.add(s)
        print(f"--- Sample [{s}] ---")
        print(f"  act_name: {d.get('act_name')}")
        print(f"  jurisdiction: {d.get('jurisdiction')}")
        print(f"  filename: {d.get('filename')}")
        print(f"  text[:80]: {d['text'][:80]}")
        print()

print("PASS: All scope fields present and correct." if has_scope == len(lines) else "FAIL: Some chunks missing scope!")
