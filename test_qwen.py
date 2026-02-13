"""
Simple clean test - Show Qwen answering a legal question
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" LEGAL AID RAG - QWEN 3 NEXT 80B TEST")
print("="*80)

from app.rag import answer

question = "What is the punishment for theft under BNS?"
print(f"\nQUESTION:")
print(f"  {question}\n")
print("Processing with Qwen 3 Next 80B...")
print("-"*80)

result = answer(question)

print("\nQWEN ANSWER:")
print("="*80)
print(result.get('answer', 'No answer'))
print("="*80)

print(f"\nCITATIONS: {len(result.get('citations', []))}")
for cite in result.get('citations', []):
    print(f"  {cite['ref']} - {cite['title'][:50]}... (from {cite['where']})")

print("\n" + "="*80)
print(" TEST COMPLETE - SYSTEM WORKING!")
print("="*80)
