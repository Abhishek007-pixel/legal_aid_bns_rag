"""
Complete RAG Test - Show Full Gemini Answer
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

print("="*80)
print(" LEGAL AID RAG SYSTEM - FULL TEST WITH GEMINI")
print("="*80)

from app.rag import answer

# Test question
question = "What is the punishment for theft under BNS?"

print(f"\nQUESTION: {question}")
print("="*80)

print("\n[Processing...]")
print("  1. Retrieving relevant sections from BNS...")
print("  2. Ranking with hybrid search (FAISS + BM25)...")
print("  3. Reranking with cross-encoder...")
print("  4. Generating answer with Gemini Flash 1.5...")

result = answer(question)

print("\n" + "="*80)
print(" GEMINI ANSWER")
print("="*80)
print(result.get('answer', 'No answer generated'))

print("\n" + "="*80)
print(f" CITATIONS ({len(result.get('citations', []))} sources)")
print("="*80)
for cite in result.get('citations', []):
    print(f"{cite['ref']} {cite['title'][:60]}...")
    print(f"    Source: {cite['where']}")
    print()

print("="*80)
print(" TEST COMPLETE")
print("="*80)
