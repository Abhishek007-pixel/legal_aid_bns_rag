import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from openai import OpenAI

from app.settings import settings

class HybridRetriever:
    def __init__(self):
        print("[init] Loading Hybrid Retriever components...")
        self.index_dir = Path("data/index")
        self.meta_path = self.index_dir / "meta.jsonl"
        self.faiss_path = self.index_dir / "faiss.index"
        self.sec_map_path = self.index_dir / "section_map.json"
        
        # 1. Load Metadata
        self.meta = []
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = [json.loads(line) for line in f]
        else:
            print("[warning] No metadata found. Please run ingest.")
        
        # 2. Build/Load BM25
        print("[init] Building BM25 index...")
        tokenized_corpus = [doc["text"].split() for doc in self.meta]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        
        # 3. Load FAISS Index
        self.faiss_index = None
        if self.faiss_path.exists():
            print("[init] Loading FAISS index...")
            self.faiss_index = faiss.read_index(str(self.faiss_path))
            
        # 4. Load Section Map
        self.section_map = {}
        if self.sec_map_path.exists():
            self.section_map = json.loads(self.sec_map_path.read_text(encoding="utf-8"))

        # 5. Load Cross-Encoder (The "Judge")
        # 'ms-marco-TinyBERT-L-2-v2' is fast, effective, and lightweight
        if settings.ENABLE_RERANKING:
            print("[init] Loading Cross-Encoder model...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
        else:
            print("[init] Reranking DISABLED (Memory Optimization)")
            self.reranker = None

        # 6. OpenAI Client for Query Embedding (Must match ingest model)
        self.client = OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )

    def _get_query_embedding(self, text: str) -> np.ndarray:
        """Embed query using the same API as ingestion."""
        try:
            resp = self.client.embeddings.create(
                model=settings.EMBED_MODEL,
                input=[text]
            )
            emb = resp.data[0].embedding
            # Normalize (L2) - crucial for Cosine Similarity in FAISS
            vec = np.array(emb, dtype="float32")
            norm = np.linalg.norm(vec)
            # Safety check: avoid division by zero (shouldn't happen with real embeddings)
            if norm > 0:
                vec = vec / norm
            else:
                print("[warning] Query embedding has zero norm, using unnormalized vector")
            return vec.reshape(1, -1)
        except Exception as e:
            print(f"[error] Embedding failed: {e}")
            return None

    def _retrieve_faiss(self, query_vec, k=30, filename=None) -> List[Dict]:
        """Fetch vectors, then filter by filename."""
        if not self.faiss_index or query_vec is None: return []
        
        # Strategy: Over-fetch k*4 to ensure we have enough results after filtering
        search_k = k * 4 if filename else k
        if search_k > len(self.meta): search_k = len(self.meta)
        
        scores, indices = self.faiss_index.search(query_vec, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.meta): continue
            rec = self.meta[idx]
            
            # Filter
            if filename and rec.get("filename") != filename:
                continue
                
            results.append({**rec, "score": float(score), "retrieval_type": "faiss"})
            
            if len(results) >= k: break
        return results

    def _retrieve_bm25(self, query, k=30, filename=None) -> List[Dict]:
        """Fetch keyword matches, then filter."""
        if not self.bm25: return []
        
        scores = self.bm25.get_scores(query.split())
        # Get top indices
        top_n = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_n:
            rec = self.meta[idx]
            # Filter
            if filename and rec.get("filename") != filename:
                continue
            
            # Optimization: Stop if score is too low or we have enough
            if len(results) >= k: break
            
            results.append({**rec, "score": float(scores[idx]), "retrieval_type": "bm25"})
        return results

    def _retrieve_section(self, query) -> List[Dict]:
        """Deterministic regex lookup for Section numbers."""
        match = re.search(r"sec(?:tion)?\.?\s+(\d+[A-Za-z]?)", query, re.I)
        if match:
            sec_num = match.group(1)
            # Normalize to key if needed, assuming direct match for now
            if sec_num in self.section_map:
                idx = self.section_map[sec_num]["idx"]
                if idx < len(self.meta):
                    rec = self.meta[idx]
                    return [{**rec, "score": 999.0, "retrieval_type": "section_map"}]
        return []

    def reciprocal_rank_fusion(self, results_dict: Dict[str, List[Dict]], k=60):
        """Combine lists using RRF."""
        fused_scores = {}
        doc_map = {}
        
        for source, docs in results_dict.items():
            for rank, doc in enumerate(docs):
                # Use ID if available, else fallback to text hash
                doc_id = doc.get("id") or str(hash(doc["text"]))
                
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                    doc_map[doc_id] = doc
                
                # RRF Formula
                fused_scores[doc_id] += 1.0 / (k + rank + 1)
        
        # Sort by RRF score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    def search(self, query: str, filename: str = None, top_k: int = 5):
        # 1. Parallel Retrieval
        q_vec = self._get_query_embedding(query)
        
        # Retrieve candidates (Top 30 from each source)
        faiss_hits = self._retrieve_faiss(q_vec, k=30, filename=filename)
        bm25_hits = self._retrieve_bm25(query, k=30, filename=filename)
        sec_hits = self._retrieve_section(query)

        # 2. Fuse (RRF)
        # We merge them to get a broad candidate pool
        candidates = self.reciprocal_rank_fusion({
            "faiss": faiss_hits,
            "bm25": bm25_hits,
            "section": sec_hits
        }, k=settings.RRF_K)
        
        # Take top N for heavy re-ranking
        top_candidates = candidates[:settings.RERANK_CANDIDATES]

        if not top_candidates:
            return []

        # 3. Cross-Encoder Re-ranking
        if self.reranker:
            # Pair query with every doc text
            pairs = [[query, doc["text"]] for doc in top_candidates]
            scores = self.reranker.predict(pairs)
            
            # Attach new scores and sort
            for doc, score in zip(top_candidates, scores):
                doc["rerank_score"] = float(score)
                
            final_results = sorted(top_candidates, key=lambda x: x["rerank_score"], reverse=True)
            return final_results[:top_k] # Return reranked top_k
        else:
            # Fallback: Just return RRF top_k if Reranker disabled
            return top_candidates[:top_k]

# Global instance
hybrid_retriever = HybridRetriever()