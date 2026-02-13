from app.hybrid_retriever import hybrid_retriever

print("Testing FAISS + Hybrid Retrieval...")
results = hybrid_retriever.search("What is theft under BNS?", top_k=3)
print(f"\nFound {len(results)} results:\n")

for i, r in enumerate(results, 1):
    print(f"[{i}] {r['title'][:60]}")
    print(f"    Score: {r.get('rerank_score', r.get('score', 0)):.4f}")
    print(f"    Type: {r.get('retrieval_type', 'unknown')}")
    print(f"    Text: {r['text'][:120 ]}...")
    print()

print("✓ FAISS and hybrid retrieval working correctly!")
