#!/usr/bin/env python3
"""
Comprehensive RAG Pipeline Verification Script (Memory Optimized)
Tests: FAISS, BM25, RRF, Cross-Encoder, OpenRouter SDK
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import json
import gc
from typing import List, Dict

# ANSI color codes for better terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*len(title)}{Colors.ENDC}")

def print_result(doc: Dict, rank: int):
    """Print a single retrieval result"""
    print(f"{Colors.BOLD}[{rank}] Score: {doc.get('score', 0):.4f}{Colors.ENDC}")
    print(f"    Type: {Colors.YELLOW}{doc.get('retrieval_type', 'unknown')}{Colors.ENDC}")
    print(f"    File: {Colors.BLUE}{doc.get('filename', 'unknown')}{Colors.ENDC}")
    print(f"    Title: {doc.get('title', 'N/A')}")
    print(f"    Text: {doc['text'][:150]}...")
    if 'rerank_score' in doc:
        print(f"    {Colors.GREEN}Rerank Score: {doc['rerank_score']:.4f}{Colors.ENDC}")
    print()

def test_individual_retrievers(query: str):
    """Test FAISS, BM25, and Section retrieval individually"""
    print_section("STEP 1: Individual Retriever Testing")
    
    print("Attempting to load Hybrid Retriever...")
    try:
        from app.hybrid_retriever import hybrid_retriever
        print(f"{Colors.GREEN}✓ Hybrid Retriever loaded successfully{Colors.ENDC}")
    except MemoryError as e:
        print(f"{Colors.RED}✗ MemoryError loading retriever: {e}{Colors.ENDC}")
        print(f"{Colors.YELLOW}  Tip: The FAISS index may be too large. Try reducing the dataset.{Colors.ENDC}")
        return [], [], []
    except Exception as e:
        print(f"{Colors.RED}✗ Error loading retriever: {e}{Colors.ENDC}")
        return [], [], []
    
    # Test FAISS
    print_subsection("🔍 FAISS Dense Retrieval (Top 5)")
    try:
        q_vec = hybrid_retriever._get_query_embedding(query)
        if q_vec is not None:
            faiss_results = hybrid_retriever._retrieve_faiss(q_vec, k=5)
            if faiss_results:
                print(f"{Colors.GREEN}✓ FAISS returned {len(faiss_results)} results{Colors.ENDC}")
                for i, doc in enumerate(faiss_results[:5], 1):
                    print_result(doc, i)
            else:
                print(f"{Colors.RED}✗ FAISS returned no results{Colors.ENDC}")
        else:
            print(f"{Colors.RED}✗ Failed to generate query embedding{Colors.ENDC}")
            faiss_results = []
    except Exception as e:
        print(f"{Colors.RED}✗ FAISS test failed: {e}{Colors.ENDC}")
        faiss_results = []
    
    # Test BM25
    print_subsection("🔍 BM25 Sparse Retrieval (Top 5)")
    try:
        bm25_results = hybrid_retriever._retrieve_bm25(query, k=5)
        if bm25_results:
            print(f"{Colors.GREEN}✓ BM25 returned {len(bm25_results)} results{Colors.ENDC}")
            for i, doc in enumerate(bm25_results[:5], 1):
                print_result(doc, i)
        else:
            print(f"{Colors.RED}✗ BM25 returned no results{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}✗ BM25 test failed: {e}{Colors.ENDC}")
        bm25_results = []
    
    # Test Section Map
    print_subsection("🔍 Section Map Lookup")
    try:
        section_results = hybrid_retriever._retrieve_section(query)
        if section_results:
            print(f"{Colors.GREEN}✓ Section map found direct match{Colors.ENDC}")
            for i, doc in enumerate(section_results, 1):
                print_result(doc, i)
        else:
            print(f"{Colors.YELLOW}ℹ No direct section match found (this is normal for general queries){Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}✗ Section lookup failed: {e}{Colors.ENDC}")
        section_results = []
    
    return faiss_results, bm25_results, section_results

def test_rrf(query: str, faiss_results: List[Dict], bm25_results: List[Dict], section_results: List[Dict]):
    """Test Reciprocal Rank Fusion"""
    print_section("STEP 2: Reciprocal Rank Fusion (RRF)")
    
    from app.settings import settings
    from app.hybrid_retriever import hybrid_retriever
    
    print(f"Input sources:")
    print(f"  - FAISS: {len(faiss_results)} results")
    print(f"  - BM25: {len(bm25_results)} results")
    print(f"  - Section: {len(section_results)} results")
    print(f"  - RRF K parameter: {settings.RRF_K}")
    
    fused = hybrid_retriever.reciprocal_rank_fusion({
        "faiss": faiss_results,
        "bm25": bm25_results,
        "section": section_results
    }, k=settings.RRF_K)
    
    print(f"\n{Colors.GREEN}✓ RRF produced {len(fused)} unique results{Colors.ENDC}")
    
    print_subsection(f"Top {min(10, len(fused))} RRF Results")
    for i, doc in enumerate(fused[:10], 1):
        print_result(doc, i)
    
    return fused

def test_reranking(query: str, candidates: List[Dict]):
    """Test Cross-Encoder Reranking"""
    print_section("STEP 3: Cross-Encoder Reranking")
    
    from app.settings import settings
    from app.hybrid_retriever import hybrid_retriever
    
    if not hybrid_retriever.reranker:
        print(f"{Colors.YELLOW}⚠ Reranking is DISABLED (ENABLE_RERANKING=False){Colors.ENDC}")
        print(f"Returning top {settings.TOP_K} from RRF...")
        return candidates[:settings.TOP_K]
    
    print(f"Reranking top {settings.RERANK_CANDIDATES} candidates...")
    print(f"Model: cross-encoder/ms-marco-TinyBERT-L-2-v2")
    
    # Take top candidates for reranking
    top_candidates = candidates[:settings.RERANK_CANDIDATES]
    
    # Pair query with docs
    pairs = [[query, doc["text"]] for doc in top_candidates]
    scores = hybrid_retriever.reranker.predict(pairs)
    
    # Attach scores
    for doc, score in zip(top_candidates, scores):
        doc["rerank_score"] = float(score)
    
    # Sort by rerank score
    reranked = sorted(top_candidates, key=lambda x: x["rerank_score"], reverse=True)
    
    print(f"{Colors.GREEN}✓ Reranking complete{Colors.ENDC}")
    
    print_subsection(f"Top {settings.TOP_K} After Reranking")
    for i, doc in enumerate(reranked[:settings.TOP_K], 1):
        print_result(doc, i)
    
    return reranked[:settings.TOP_K]

def test_full_pipeline(query: str):
    """Test the complete hybrid search pipeline"""
    print_section("STEP 4: Full Pipeline Test")
    
    from app.settings import settings
    from app.hybrid_retriever import hybrid_retriever
    
    print(f"Query: {Colors.CYAN}\"{query}\"{Colors.ENDC}")
    print(f"Top-K: {settings.TOP_K}")
    
    results = hybrid_retriever.search(query, top_k=settings.TOP_K)
    
    print(f"\n{Colors.GREEN}✓ Pipeline returned {len(results)} results{Colors.ENDC}")
    
    print_subsection("Final Top-K Results")
    for i, doc in enumerate(results, 1):
        print_result(doc, i)
    
    return results

def test_openrouter_sdk():
    """Test OpenRouter SDK connection and embedding"""
    print_section("STEP 5: OpenRouter SDK Verification")
    
    from openai import OpenAI
    from app.settings import settings
    
    client = OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY,
    )
    
    # Test 1: Embedding API
    print_subsection("Test 1: Embedding API")
    try:
        resp = client.embeddings.create(
            model=settings.EMBED_MODEL,
            input=["test query"]
        )
        embedding = resp.data[0].embedding
        print(f"{Colors.GREEN}✓ Embedding API works{Colors.ENDC}")
        print(f"  Model: {settings.EMBED_MODEL}")
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"{Colors.RED}✗ Embedding API failed: {e}{Colors.ENDC}")
        return False
    
    # Test 2: Chat Completion API
    print_subsection("Test 2: Chat Completion API")
    try:
        resp = client.chat.completions.create(
            model=settings.OR_MODEL,
            messages=[
                {"role": "user", "content": "Say 'hello' if you can hear me."}
            ],
            max_tokens=20
        )
        content = resp.choices[0].message.content
        print(f"{Colors.GREEN}✓ Chat Completion API works{Colors.ENDC}")
        print(f"  Model: {settings.OR_MODEL}")
        print(f"  Response: {content}")
    except Exception as e:
        print(f"{Colors.RED}✗ Chat Completion API failed: {e}{Colors.ENDC}")
        return False
    
    return True

def main():
    """Main test execution"""
    print_section("RAG Pipeline Verification Test")
    
    from app.settings import settings
    
    print(f"Settings:")
    print(f"  USE_EMBEDDINGS: {settings.USE_EMBEDDINGS}")
    print(f"  USE_FAISS: {settings.USE_FAISS}")
    print(f"  USE_BM25: {settings.USE_BM25}")
    print(f"  ENABLE_RERANKING: {settings.ENABLE_RERANKING}")
    print(f"  TOP_K: {settings.TOP_K}")
    print(f"  RERANK_CANDIDATES: {settings.RERANK_CANDIDATES}")
    
    # Test query - change this to test different queries
    test_query = "What is theft under BNS?"
    
    # Run individual retriever tests
    faiss_results, bm25_results, section_results = test_individual_retrievers(test_query)
    
    # Test RRF
    fused_results = test_rrf(test_query, faiss_results, bm25_results, section_results)
    
    # Test reranking
    final_results = test_reranking(test_query, fused_results)
    
    # Test full pipeline
    pipeline_results = test_full_pipeline(test_query)
    
    # Test OpenRouter SDK
    sdk_ok = test_openrouter_sdk()
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    print(f"{'Component':<30} {'Status':<10}")
    print(f"{'-'*40}")
    print(f"{'FAISS Retrieval':<30} {Colors.GREEN}✓ PASS{Colors.ENDC if faiss_results else Colors.RED + '✗ FAIL' + Colors.ENDC}")
    print(f"{'BM25 Retrieval':<30} {Colors.GREEN}✓ PASS{Colors.ENDC if bm25_results else Colors.RED + '✗ FAIL' + Colors.ENDC}")
    print(f"{'Reciprocal Rank Fusion':<30} {Colors.GREEN}✓ PASS{Colors.ENDC if fused_results else Colors.RED + '✗ FAIL' + Colors.ENDC}")
    print(f"{'Cross-Encoder Reranking':<30} {Colors.GREEN}✓ PASS{Colors.ENDC if (final_results or not settings.ENABLE_RERANKING) else Colors.RED + '✗ FAIL' + Colors.ENDC}")
    print(f"{'Full Pipeline':<30} {Colors.GREEN}✓ PASS{Colors.ENDC if pipeline_results else Colors.RED + '✗ FAIL' + Colors.ENDC}")
    print(f"{'OpenRouter SDK':<30} {Colors.GREEN}✓ PASS{Colors.ENDC if sdk_ok else Colors.RED + '✗ FAIL' + Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}All tests completed!{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
