import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from app.hybrid_retriever import hybrid_retriever

def test_add_document():
    print("Testing dynamic document indexing...")
    
    # Create dummy text
    text = """
    This is a test document about Space Law.
    Section 1. Outer Space Treaty.
    The exploration and use of outer space shall be carried out for the benefit and in the interests of all countries.
    """
    filename = "test_space_law.pdf"
    
    # Add document
    try:
        chunks_added = hybrid_retriever.add_document(text, filename)
        print(f"Added {chunks_added} chunks.")
        
        if chunks_added == 0:
            print("FAILURE: No chunks added.")
            return

        # Verify retrieval
        print("Verifying retrieval...")
        # Note: Depending on implementation, these might not be immediately available if indexes aren't refreshed
        # But my implementation updates in-memory structures immediately.
        
        results = hybrid_retriever.search("outer space treaty", filename=filename)
        
        found = False
        print(f"Search returned {len(results)} results.")
        for r in results:
            # Check filename match (might be missing in some result types, so check carefully)
            r_fname = r.get('filename') or r.get('source')
            if r_fname == filename and "Outer Space Treaty" in r['text']:
                found = True
                print(f"Found relevant chunk: {r['text'][:50]}...")
                break
                
        if found:
            print("SUCCESS: Document indexed and retrieved correctly.")
        else:
            print("FAILURE: Could not retrieve indexed document.")
            print("Top results:")
            for r in results:
                print(f" - {r.get('source')}: {r['text'][:30]}...")
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_add_document()
