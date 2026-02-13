"""
Test OpenRouter SDK - Embeddings and Chat Completion
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from openai import OpenAI
from app.settings import settings
import traceback

print("="*80)
print("OpenRouter SDK Test")
print("="*80)

# Check settings
print(f"\nSettings:")
print(f"  Base URL: {settings.OPENROUTER_BASE_URL}")
print(f"  API Key: {settings.OPENROUTER_API_KEY[:20]}..." if settings.OPENROUTER_API_KEY else "  API Key: NOT SET")
print(f"  Embed Model: {settings.EMBED_MODEL}")
print(f"  LLM Model: {settings.OR_MODEL}")

client = OpenAI(
    base_url=settings.OPENROUTER_BASE_URL,
    api_key=settings.OPENROUTER_API_KEY,
)

# Test 1: Embedding API
print("\n" + "="*80)
print("TEST 1: Embedding API")
print("="*80)
try:
    print(f"Calling: {settings.EMBED_MODEL}")
    resp = client.embeddings.create(
        model=settings.EMBED_MODEL,
        input=["test query for embeddings"]
    )
    embedding = resp.data[0].embedding
    print(f"[OK] SUCCESS")
    print(f"   Dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")
    print("\nFull error:")
    traceback.print_exc()

# Test 2: Chat Completion API
print("\n" + "="*80)
print("TEST 2: Chat Completion API")
print("="*80)
try:
    print(f"Calling: {settings.OR_MODEL}")
    resp = client.chat.completions.create(
        model=settings.OR_MODEL,
        messages=[
            {"role": "user", "content": "Say 'Hello! I am working.' in exactly those words."}
        ],
        max_tokens=20
    )
    content = resp.choices[0].message.content
    print(f"[OK] SUCCESS")
    print(f"   Response: {content}")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")
    print("\nFull error:")
    traceback.print_exc()

# Test 3: Full RAG Pipeline Test
print("\n" + "="*80)
print("TEST 3: Full RAG Answer Generation")
print("="*80)
try:
    from app.rag import answer
    print("Testing: answer('What is theft?')")
    result = answer("What is theft under BNS?")
    print(f"[OK] SUCCESS")
    print(f"   Answer length: {len(result.get('answer', ''))} chars")
    print(f"   Citations: {len(result.get('citations', []))}")
    print(f"\n   Answer preview:")
    print(f"   {result.get('answer', 'No answer')[:200]}...")
except Exception as e:
    print(f"[FAIL] FAILED: {e}")
    print("\nFull error:")
    traceback.print_exc()

print("\n" + "="*80)
print("Test Complete")
print("="*80)
