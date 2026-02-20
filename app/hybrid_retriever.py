import json
import re
import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.settings import settings


# Keywords that suggest the user is asking about their own uploaded document
USER_DOC_SIGNALS = [
    "my contract", "my document", "this pdf", "uploaded", "my case",
    "my agreement", "this document", "my file", "my lease", "my deed",
    "this file", "my complaint", "my fir", "my petition",
]

LAW_SCOPES = ["global_law", "supreme_court", "labour_law", "state_law"]


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
            print(f"[init] Loaded {len(self.meta)} chunks from metadata.")
            # Show scope distribution on startup
            from collections import Counter
            scope_dist = Counter(d.get("scope", "unknown") for d in self.meta)
            print(f"[init] Scope distribution: {dict(scope_dist)}")
        else:
            print("[warning] No metadata found. Please run ingest.")

        # 2. Build BM25
        if self.meta:
            print("[init] Building BM25 index...")
            tokenized_corpus = [doc["text"].split() for doc in self.meta]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

        # 3. Load FAISS Index (only if USE_FAISS=true)
        self.faiss_index = None
        if settings.USE_FAISS and self.faiss_path.exists():
            try:
                import faiss
                print("[init] Loading FAISS index...")
                self.faiss_index = faiss.read_index(str(self.faiss_path))
            except ImportError:
                print("[warning] faiss-cpu not installed. Vector search disabled.")
        elif settings.USE_FAISS:
            print("[warning] USE_FAISS=true but no FAISS index found. Run ingest first.")

        # 4. Load Section Map
        self.section_map = {}
        if self.sec_map_path.exists():
            self.section_map = json.loads(self.sec_map_path.read_text(encoding="utf-8"))

        # 5. Load Sentence-Transformers embedding model (only if USE_EMBEDDINGS=true)
        self.embed_model = None
        if settings.USE_EMBEDDINGS:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[init] Loading embedding model: {settings.EMBED_MODEL} ...")
                self.embed_model = SentenceTransformer(settings.EMBED_MODEL)
                print("[init] Embedding model loaded.")
            except Exception as e:
                print(f"[warning] Failed to load embedding model: {e}")

        # 6. Load Cross-Encoder reranker
        if settings.ENABLE_RERANKING:
            print("[init] Loading Cross-Encoder model...")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
        else:
            print("[init] Reranking DISABLED.")
            self.reranker = None

        print("[init] Hybrid Retriever ready.")

    # ------------------------------------------------------------------ #
    #  Intent Detection
    # ------------------------------------------------------------------ #
    def _query_is_about_user_doc(self, query: str) -> bool:
        """Return True if the query seems to be about the user's uploaded document."""
        q = query.lower()
        return any(sig in q for sig in USER_DOC_SIGNALS)

    # ------------------------------------------------------------------ #
    #  Retrieval Logging
    # ------------------------------------------------------------------ #
    def _log_retrieval(self, query: str, results: List[Dict]) -> None:
        log_entry = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "query": query[:120],
            "scopes_returned": list({d.get("scope", "?") for d in results}),
            "num_docs": len(results),
            "top_score": (
                results[0].get("rerank_score", results[0].get("score", 0))
                if results else 0
            ),
        }
        print(f"[retrieval_log] {json.dumps(log_entry)}")

    # ------------------------------------------------------------------ #
    #  Embedding
    # ------------------------------------------------------------------ #
    def _get_query_embedding(self, text: str) -> Optional[np.ndarray]:
        """Embed query using local sentence-transformers model."""
        if self.embed_model is None:
            return None
        try:
            vec = self.embed_model.encode([text], normalize_embeddings=True)
            return vec.astype("float32")
        except Exception as e:
            print(f"[error] Embedding failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  FAISS Retrieval (with scope filter)
    # ------------------------------------------------------------------ #
    def _retrieve_faiss(
        self, query_vec, k=30,
        filename: str = None,
        scope_filter: List[str] = None,
        user_id: str = None
    ) -> List[Dict]:
        """Fetch vectors, then filter by scope/filename/user_id."""
        if not self.faiss_index or query_vec is None:
            return []

        search_k = k * 4 if (filename or scope_filter) else k
        if search_k > len(self.meta):
            search_k = len(self.meta)

        scores, indices = self.faiss_index.search(query_vec, search_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.meta):
                continue
            rec = self.meta[idx]
            # Filename filter (legacy support)
            if filename and rec.get("filename") != filename:
                continue
            # Scope filter
            if scope_filter and rec.get("scope") not in scope_filter:
                continue
            # User isolation for user_upload scope
            if rec.get("scope") == "user_upload" and user_id and rec.get("user_id") != user_id:
                continue
            results.append({**rec, "score": float(score), "retrieval_type": "faiss"})
            if len(results) >= k:
                break
        return results

    # ------------------------------------------------------------------ #
    #  BM25 Retrieval (with scope filter)
    # ------------------------------------------------------------------ #
    def _retrieve_bm25(
        self, query, k=30,
        filename: str = None,
        scope_filter: List[str] = None,
        user_id: str = None
    ) -> List[Dict]:
        """Fetch keyword matches, then filter by scope/user_id."""
        if not self.bm25:
            return []

        scores = self.bm25.get_scores(query.split())
        top_n = np.argsort(scores)[::-1]

        results = []
        for idx in top_n:
            if len(results) >= k:
                break
            rec = self.meta[idx]
            if filename and rec.get("filename") != filename:
                continue
            if scope_filter and rec.get("scope") not in scope_filter:
                continue
            if rec.get("scope") == "user_upload" and user_id and rec.get("user_id") != user_id:
                continue
            results.append({**rec, "score": float(scores[idx]), "retrieval_type": "bm25"})
        return results

    # ------------------------------------------------------------------ #
    #  Section-map Retrieval
    # ------------------------------------------------------------------ #
    def _retrieve_section(self, query, scope_filter: List[str] = None) -> List[Dict]:
        """Deterministic regex lookup for Section numbers."""
        match = re.search(r"sec(?:tion)?\.?\s+(\d+[A-Za-z]?)", query, re.I)
        if match:
            sec_num = match.group(1)
            if sec_num in self.section_map:
                entry = self.section_map[sec_num]
                idx = entry["idx"]
                # Apply scope filter
                if scope_filter and entry.get("scope") not in scope_filter:
                    return []
                if idx < len(self.meta):
                    rec = self.meta[idx]
                    return [{**rec, "score": 999.0, "retrieval_type": "section_map"}]
        return []

    # ------------------------------------------------------------------ #
    #  Reciprocal Rank Fusion
    # ------------------------------------------------------------------ #
    def reciprocal_rank_fusion(self, results_dict: Dict[str, List[Dict]], k=60):
        fused_scores = {}
        doc_map = {}
        for source, docs in results_dict.items():
            for rank, doc in enumerate(docs):
                doc_id = doc.get("id") or str(hash(doc["text"]))
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                    doc_map[doc_id] = doc
                fused_scores[doc_id] += 1.0 / (k + rank + 1)
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    # ------------------------------------------------------------------ #
    #  Core Search (scoped)
    # ------------------------------------------------------------------ #
    def search(
        self, query: str,
        filename: str = None,
        scope_filter: List[str] = None,
        user_id: str = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Core search with optional scope filtering and user isolation.
        scope_filter: list of allowed scopes e.g. ["global_law", "supreme_court"]
        user_id: isolates user_upload scope per user
        """
        q_vec = None
        if settings.USE_FAISS and settings.USE_EMBEDDINGS:
            q_vec = self._get_query_embedding(query)

        faiss_hits = self._retrieve_faiss(q_vec, k=30, filename=filename, scope_filter=scope_filter, user_id=user_id)
        bm25_hits = self._retrieve_bm25(query, k=30, filename=filename, scope_filter=scope_filter, user_id=user_id)
        sec_hits = self._retrieve_section(query, scope_filter=scope_filter)

        candidates = self.reciprocal_rank_fusion(
            {"faiss": faiss_hits, "bm25": bm25_hits, "section": sec_hits},
            k=settings.RRF_K,
        )

        top_candidates = candidates[:settings.RERANK_CANDIDATES]

        if not top_candidates:
            return []

        if self.reranker:
            pairs = [[query, doc["text"]] for doc in top_candidates]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(top_candidates, scores):
                doc["rerank_score"] = float(score)
            final_results = sorted(top_candidates, key=lambda x: x["rerank_score"], reverse=True)
            return final_results[:top_k]

        return top_candidates[:top_k]

    # ------------------------------------------------------------------ #
    #  Hybrid Search — Weighted Score Merge (PRIMARY ENTRY POINT)
    # ------------------------------------------------------------------ #
    def hybrid_search(self, query: str, user_id: str = None, top_k: int = 5) -> List[Dict]:
        """
        Scoped hybrid search:
        1. Fetch from user's uploads (if user_id provided)
        2. Fetch from law corpus
        3. Merge by score (weighted, not fixed split)
        4. Apply confidence threshold
        5. Boost user docs if query signals user-document intent
        6. Log retrieval for debugging
        """
        # Detect intent
        user_intent = self._query_is_about_user_doc(query) if user_id else False

        # Retrieve from both pools
        user_docs = []
        if user_id:
            user_docs = self.search(
                query, scope_filter=["user_upload"], user_id=user_id, top_k=10
            )
        law_docs = self.search(
            query, scope_filter=LAW_SCOPES, top_k=10
        )

        # Boost user doc scores if intent suggests user is asking about their document
        if user_intent and user_docs:
            for d in user_docs:
                score_key = "rerank_score" if "rerank_score" in d else "score"
                d[score_key] = d[score_key] * 1.5
            print(f"[hybrid_search] Intent detected: boosting user_upload scores by 1.5x")

        # Merge and deduplicate
        combined = user_docs + law_docs
        seen, unique = set(), []
        for doc in combined:
            key = doc.get("id") or hash(doc["text"][:80])
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        # Apply confidence threshold
        threshold = settings.MIN_SIM_SCORE
        confident = [
            d for d in unique
            if d.get("rerank_score", d.get("score", 0)) >= threshold
        ]

        # Sort by best available score
        ranked = sorted(
            confident,
            key=lambda x: x.get("rerank_score", x.get("score", 0)),
            reverse=True,
        )

        result = ranked[:top_k]
        self._log_retrieval(query, result)
        return result

    # ------------------------------------------------------------------ #
    #  Dynamic Document Adding (for /upload endpoint)
    # ------------------------------------------------------------------ #
    def add_document(
        self, text: str, filename: str,
        scope: str = "user_upload",
        user_id: str = None
    ) -> int:
        print(f"[index] Adding document: {filename}  scope={scope}  user_id={user_id}")

        words = text.split()
        chunks = []
        chunk_size = 300
        overlap = 50
        if len(words) <= chunk_size:
            chunks.append(text)
        else:
            for i in range(0, len(words), chunk_size - overlap):
                chunks.append(" ".join(words[i:i + chunk_size]))

        if not chunks:
            return 0

        new_meta = []
        new_embeddings = []

        print(f"[index] Processing {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            rec = {
                "text": chunk,
                "filename": filename,
                "source": filename,
                "title": filename,
                "chunk_id": i,
                # --- Scope & jurisdiction metadata ---
                "scope": scope,
                "act_name": filename,
                "jurisdiction": "india",
                "user_id": user_id,
            }
            new_meta.append(rec)

            if settings.USE_FAISS and settings.USE_EMBEDDINGS:
                emb = self._get_query_embedding(chunk)
                if emb is not None:
                    new_embeddings.append(emb)
                else:
                    print(f"[warning] Embedding failed for chunk {i}, skipping vector index.")

        if not new_meta:
            print("[error] No chunks processed")
            return 0

        # Update FAISS (only in hybrid mode)
        if new_embeddings and settings.USE_FAISS:
            try:
                import faiss
                vectors = np.vstack(new_embeddings)
                if self.faiss_index is None:
                    self.faiss_index = faiss.IndexFlatIP(vectors.shape[1])
                self.faiss_index.add(vectors)
            except Exception as e:
                print(f"[error] Failed to update FAISS index: {e}")

        # Update metadata and BM25
        self.meta.extend(new_meta)
        try:
            tokenized_corpus = [doc["text"].split() for doc in self.meta]
            self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            print(f"[error] Failed to update BM25 index: {e}")

        # Persist metadata
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            with open(self.meta_path, "a", encoding="utf-8") as f:
                for rec in new_meta:
                    f.write(json.dumps(rec) + "\n")

            if new_embeddings and settings.USE_FAISS and self.faiss_index is not None:
                import faiss
                faiss.write_index(self.faiss_index, str(self.faiss_path))
            print("[index] Persisted updates to disk.")
        except Exception as e:
            print(f"[warning] Failed to persist index: {e}")

        print(f"[index] Successfully added {len(new_meta)} chunks from {filename} (scope={scope})")
        return len(new_meta)


# Global singleton
hybrid_retriever = HybridRetriever()