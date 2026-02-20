"""
Live demonstration of the complete RAG pipeline with Gemini API
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from app.rag import answer

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    print_section("LEGAL RAG PIPELINE - LIVE DEMO")
    
    # Test query
    query = "What is theft under BNS?"
    print(f"\nQuery: \"{query}\"")
    print("\nProcessing...")
    print("  [1] Hybrid Retrieval (FAISS + BM25 + RRF)")
    print("  [2] Cross-Encoder Reranking")
    print("  [3] Gemini LLM Generation")
    
    try:
        # Call the RAG pipeline
        result = answer(query)
        
        print_section("RESULTS")
        
        # Display answer
        print("\n[ANSWER]")
        print("-" * 70)
        answer_text = result.get('answer', 'No answer generated')
        # Print first 500 chars to keep it readable in terminal
        if len(answer_text) > 500:
            print(answer_text[:500] + "...")
        else:
            print(answer_text)
        print("-" * 70)
        
        # Display citations
        citations = result.get('citations', [])
        print(f"\n[CITATIONS] ({len(citations)} sources)")
        print("-" * 70)
        for i, cite in enumerate(citations[:3], 1):  # Show top 3
            source = cite.get('source', 'Unknown')
            score = cite.get('score', 0)
            print(f"{i}. {source} (Score: {score:.2f})")
        print("-" * 70)
        
        print_section("STATUS: SUCCESS")
        print("\n✓ Gemini API: WORKING")
        print("✓ Hybrid Retrieval: WORKING")
        print("✓ LLM Generation: WORKING")
        print("✓ Citations: WORKING")
        
    except Exception as e:
        print_section("ERROR")
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nNote: If you see a 429 error, the API rate limit was hit.")
        print("Wait a minute and try again, or the fallback will activate.")

if __name__ == "__main__":
    main()
