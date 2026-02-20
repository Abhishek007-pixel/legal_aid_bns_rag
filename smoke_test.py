"""Quick smoke test — hits the live server on localhost:8000"""
import requests, json

BASE = "http://localhost:8000"

def test(label, endpoint, payload):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    try:
        r = requests.post(f"{BASE}{endpoint}", json=payload, timeout=60)
        data = r.json()
        print(f"Status: {r.status_code}")
        ans = data.get("answer", "")
        print(f"Answer (first 400 chars):\n{ans[:400]}")
        cites = data.get("citations", [])
        if cites:
            print(f"\nCitations ({len(cites)}):")
            for c in cites:
                print(f"  {c.get('ref')} -> {c.get('where')}")
        if data.get("fallback"):
            print("  [FALLBACK MODE - LLM unavailable]")
        return r.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

results = {}

# 1. Health check
print("\n" + "="*60)
print("TEST: /health")
print("="*60)
r = requests.get(f"{BASE}/health", timeout=10)
print(f"Status: {r.status_code} | Body: {r.json()}")
results["health"] = r.status_code == 200

# 2. Law corpus query (no user_id → should come from global_law / supreme_court)
results["law_query_bns"] = test(
    "/ask — BNS theft query (law corpus only)",
    "/ask",
    {"question": "What is theft under BNS? What section defines it?"}
)

# 3. Labour law query
results["law_query_labour"] = test(
    "/ask — Labour law query",
    "/ask",
    {"question": "What are the rules for working hours under labour law?"}
)

# 4. Consumer protection query
results["law_query_consumer"] = test(
    "/ask — Consumer protection query",
    "/ask",
    {"question": "What are consumer rights under the Consumer Protection Act 2019?"}
)

# 5. General chat (no RAG)
results["chat"] = test(
    "/chat — General greeting",
    "/chat",
    {"question": "What is LegalAid and how can it help me?"}
)

# Summary
print("\n\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
for name, ok in results.items():
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
all_ok = all(results.values())
print(f"\nOverall: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
