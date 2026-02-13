import json
import faiss
import numpy as np
from pathlib import Path

# Paths
INDEX_DIR = Path("data/index")
META_FILE = INDEX_DIR / "meta.jsonl"
FAISS_FILE = INDEX_DIR / "faiss.index"

def main():
    if not FAISS_FILE.exists() or not META_FILE.exists():
        print("❌ Error: Index files not found. Run ingest.py first.")
        return

    # 1. Load the FAISS Index (The Vectors)
    print(f"Loading index from {FAISS_FILE}...")
    index = faiss.read_index(str(FAISS_FILE))
    print(f"✅ Index contains {index.ntotal} vectors.")

    # 2. Load the Metadata (The Text)
    print(f"Loading metadata from {META_FILE}...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]

    # 3. Inspect the first 3 chunks
    print("\n--- INSPECTING FIRST 3 CHUNKS ---\n")
    for i in range(3):
        if i >= index.ntotal: break
        
        # Get the vector (embedding) for ID 'i'
        # Note: reconstruct() works for IndexFlatIP (simple indexes)
        vector = index.reconstruct(i)
        
        doc = meta[i]
        
        print(f"🔹 CHUNK #{i}")
        print(f"📄 File:   {doc.get('filename')}")
        print(f"📜 Text:   {doc['text'][:100]}...")  # Show first 100 chars
        print(f"🔢 Vector: [Length: {len(vector)}] {vector[:5]} ... (first 5 numbers)")
        print("-" * 50)

if __name__ == "__main__":
    main()