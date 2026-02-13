import json
import faiss
import numpy as np
from pathlib import Path

# Paths
INDEX_DIR = Path("data/index")
META_FILE = INDEX_DIR / "meta.jsonl"
FAISS_FILE = INDEX_DIR / "faiss.index"

def main():
    if not FAISS_FILE.exists():
        print("❌ Error: faiss.index not found.")
        return

    print(f"🔍 Peeking into {FAISS_FILE} using Memory Mapping (Safe Mode)...")

    # 1. LOAD WITH MEMORY MAPPING (IO_FLAG_MMAP)
    # This prevents loading the whole file into RAM. It reads directly from disk.
    try:
        index = faiss.read_index(str(FAISS_FILE), faiss.IO_FLAG_MMAP)
    except AttributeError:
        # Fallback for older FAISS versions that might not support the flag easily
        print("⚠️ MMAP flag not found, attempting standard read...")
        index = faiss.read_index(str(FAISS_FILE))

    print(f"✅ Index connected. Total Vectors: {index.ntotal}")

    # 2. Load Metadata (Text)
    print("📖 Reading text...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]

    # 3. View just the first 3 chunks
    print("\n--- 🔬 VECTOR INSPECTOR ---\n")
    
    # We only look at i = 0, 1, 2
    for i in range(3):
        if i >= index.ntotal: break
        
        # reconstruct(i) grabs ONLY the ith vector from the disk
        vector = index.reconstruct(i)
        
        doc = meta[i]
        
        print(f"🔹 CHUNK #{i+1}")
        print(f"📄 File: {doc.get('filename')}")
        print(f"📝 Text: {doc['text'][:60]}...") 
        # Print the first 5 numbers of the 1536-dimensional vector
        print(f"🔢 Vector Data: {vector[:5]} ... (Total dims: {len(vector)})")
        print("-" * 50)

if __name__ == "__main__":
    main()