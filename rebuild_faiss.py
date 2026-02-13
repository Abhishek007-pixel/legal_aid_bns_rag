"""
Quick script to rebuild FAISS index with better Python 3.13 compatibility
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import json
import numpy as np
from openai import OpenAI
from app.settings import settings

print("Loading metadata...")
meta_path = Path("data/index/meta.jsonl")
meta = []
with open(meta_path, "r", encoding="utf-8") as f:
    meta = [json.loads(line) for line in f]

print(f"Found {len(meta)} chunks")

# Use a simpler approach - save embeddings as numpy array instead of FAISS first
embeddings_file = Path("data/index/embeddings.npy")

if not embeddings_file.exists():
    print("Generating embeddings...")
    client = OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY,
    )
    
    texts = [m["text"] for m in meta]
    embeddings = []
    
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        resp = client.embeddings.create(
            model=settings.EMBED_MODEL,
            input=batch
        )
        embeddings.extend([d.embedding for d in resp.data])
    
    X = np.array(embeddings, dtype="float32")
    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = X / norms
    
    print(f"Saving embeddings to {embeddings_file}")
    np.save(embeddings_file, X)
else:
    print("Loading existing embeddings...")
    X = np.load(embeddings_file)

print(f"Embeddings shape: {X.shape}")

# Now try to build FAISS index
print("Building FAISS index...")
try:
    import faiss
    
    # Try IndexFlatIP (simplest, should work)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    
    faiss_path = Path("data/index/faiss.index")
    print(f"Writing index to {faiss_path}")
    faiss.write_index(index, str(faiss_path))
    print("✓ FAISS index created successfully")
    
    # Test loading
    print("Testing load...")
    test_idx = faiss.read_index(str(faiss_path))
    print(f"✓ Index loaded successfully: {test_idx.ntotal} vectors")
    
except MemoryError as e:
    print(f"✗ MemoryError building FAISS: {e}")
    print("Python 3.13 has issues with FAISS. Workaround:")
    print("1. Use numpy-based search instead of FAISS")
    print("2. Or downgrade to Python 3.11/3.12")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
