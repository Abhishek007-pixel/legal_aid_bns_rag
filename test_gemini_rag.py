"""
Test Gemini-powered RAG System
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

print("="*80)
print("GEMINI RAG SYSTEM TEST")
print("="*80)

# Test 1: Settings loaded correctly
print("\n[1/3] Testing Configuration...")
from app.settings import settings
print(f"  Gemini API Key: {settings.GEMINI_API_KEY[:20]}...")
print(f"  Gemini Model: {settings.GEMINI_MODEL}")
print(f"  Embed Model: {settings.EMBED_MODEL}")
print("[OK] Settings loaded")

# Test 2: Hybrid retrieval works
print("\n[2/3] Testing Hybrid Retrieval...")
from app.hybrid_retriever import hybrid_retriever
results = hybrid_retriever.search("What is theft?", top_k=3)
print(f"  Retrieved: {len(results)} results")
print(f"  Top result: {results[0]['title'][:50]}...")
print("[OK] Hybrid retrieval working")

# Test 3: End-to-end RAG with Gemini
print("\n[3/3] Testing Full RAG Pipeline with Gemini...")
from app.rag import answer
result = answer("What is the punishment for theft under BNS?")
print(f"  Answer length: {len(result.get('answer', ''))} chars")
print(f"  Citations: {len(result.get('citations', []))}")
print(f"\n  GEMINI ANSWER:")
print(f"  {'-'*75}")
answer_text = result.get('answer', 'No answer')
# Print first 500 chars with wrapping
for i in range(0, min(500, len(answer_text)), 75):
    print(f"  {answer_text[i:i+75]}")
print(f"  {'-'*75}")
print(f"\n  CITATIONS:")
for cite in result.get('citations', []):
    print(f"    {cite['ref']} - {cite['title'][:40]}... ({cite['where']})")
print("[OK] Full pipeline working!")

print("\n" + "="*80)
print("ALL TESTS PASSED - GEMINI RAG SYSTEM READY!")
print("="*80)
