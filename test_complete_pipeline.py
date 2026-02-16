#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete RAG Pipeline Test with Gemini API
Tests the full flow: Query -> FAISS -> BM25 -> RRF -> Cross-Encoder -> Gemini LLM -> Final Answer
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import json
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

def test_gemini_connection():
    """Test Gemini API connection"""
    print_section("STEP 1: Testing Gemini API Connection")
    
    try:
        import google.generativeai as genai
        from app.settings import settings
        
        print(f"API Key: {settings.GEMINI_API_KEY[:20]}...")
        print(f"Model: {settings.GEMINI_MODEL}")
        
        # Configure Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Simple test
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content("Say 'Hello! Gemini is working!' in one sentence.")
        
        print(f"\n{Colors.GREEN}[OK] Gemini API Connection Successful!{Colors.ENDC}")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}[FAIL] Gemini API Connection Failed: {e}{Colors.ENDC}")
        return False

def test_hybrid_retrieval(query: str):
    """Test the hybrid retrieval system"""
    print_section("STEP 2: Testing Hybrid Retrieval (FAISS + BM25 + RRF)")
    
    try:
        from app.hybrid_retriever import hybrid_retriever
        from app.settings import settings
        
        print(f"Query: {Colors.CYAN}\"{query}\"{Colors.ENDC}")
        print(f"Top-K: {settings.TOP_K}")
        
        # Get retrieval results
        results = hybrid_retriever.search(query, top_k=settings.TOP_K)
        
        if results:
            print(f"\n{Colors.GREEN}[OK] Retrieval Successful! Found {len(results)} results{Colors.ENDC}\n")
            
            for i, doc in enumerate(results[:5], 1):
                print(f"{Colors.BOLD}[{i}] Score: {doc.get('score', 0):.4f}{Colors.ENDC}")
                print(f"    File: {Colors.BLUE}{doc.get('filename', 'unknown')}{Colors.ENDC}")
                print(f"    Title: {doc.get('title', 'N/A')}")
                print(f"    Text: {doc['text'][:200]}...")
                if 'rerank_score' in doc:
                    print(f"    {Colors.GREEN}Rerank Score: {doc['rerank_score']:.4f}{Colors.ENDC}")
                print()
            
            return results
        else:
            print(f"{Colors.RED}[FAIL] No results found{Colors.ENDC}")
            return []
            
    except Exception as e:
        print(f"{Colors.RED}[FAIL] Retrieval Failed: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return []

def test_full_rag_pipeline(query: str):
    """Test the complete RAG pipeline including LLM generation"""
    print_section("STEP 3: Testing Complete RAG Pipeline with Gemini")
    
    try:
        from app.rag import answer
        
        print(f"Query: {Colors.CYAN}\"{query}\"{Colors.ENDC}")
        print(f"\nGenerating answer with Gemini...")
        
        # Get answer from RAG system
        result = answer(query)
        
        if result and result.get('answer'):
            print(f"\n{Colors.GREEN}[OK] RAG Pipeline Successful!{Colors.ENDC}\n")
            
            # Print answer
            print_subsection("Answer from Gemini")
            print(result['answer'])
            
            # Print citations
            if result.get('citations'):
                print_subsection(f"Citations ({len(result['citations'])} sources)")
                for cite in result['citations']:
                    print(f"{Colors.YELLOW}{cite['ref']}{Colors.ENDC} {cite['title']} (from {cite['where']})")
            
            return True
        else:
            print(f"{Colors.RED}[FAIL] Failed to generate answer{Colors.ENDC}")
            return False
            
    except Exception as e:
        print(f"{Colors.RED}[FAIL] RAG Pipeline Failed: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_queries():
    """Test multiple queries to verify consistency"""
    print_section("STEP 4: Testing Multiple Queries")
    
    test_queries = [
        "What is theft under BNS?",
        "What are the punishments for murder?",
        "Explain criminal conspiracy"
    ]
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n{Colors.BOLD}Query {i}/{len(test_queries)}: {Colors.CYAN}{query}{Colors.ENDC}")
        print("-" * 80)
        
        try:
            from app.rag import answer
            result = answer(query)
            
            if result and result.get('answer'):
                print(f"{Colors.GREEN}[OK] Success{Colors.ENDC}")
                print(f"Answer preview: {result['answer'][:150]}...")
                print(f"Citations: {len(result.get('citations', []))}")
                results.append(True)
            else:
                print(f"{Colors.RED}[FAIL] Failed{Colors.ENDC}")
                results.append(False)
                
        except Exception as e:
            print(f"{Colors.RED}[FAIL] Error: {e}{Colors.ENDC}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n{Colors.BOLD}Success Rate: {success_rate:.0f}% ({sum(results)}/{len(results)}){Colors.ENDC}")
    
    return all(results)

def main():
    """Main test execution"""
    print_section("RAG Pipeline Complete Test with Gemini")
    
    print(f"{Colors.BOLD}This script tests:{Colors.ENDC}")
    print("1. [OK] Gemini API connection")
    print("2. [OK] Hybrid retrieval (FAISS + BM25 + RRF)")
    print("3. [OK] Complete RAG pipeline (Retrieval -> Reranking -> Gemini LLM)")
    print("4. [OK] Multiple query consistency")
    
    # Test 1: Gemini connection
    gemini_ok = test_gemini_connection()
    if not gemini_ok:
        print(f"\n{Colors.RED}Cannot proceed without Gemini API connection{Colors.ENDC}")
        return
    
    # Test 2: Hybrid retrieval
    test_query = "What is theft under BNS?"
    retrieval_results = test_hybrid_retrieval(test_query)
    
    # Test 3: Full RAG pipeline
    rag_ok = test_full_rag_pipeline(test_query)
    
    # Test 4: Multiple queries
    multi_ok = test_multiple_queries()
    
    # Summary
    print_section("FINAL SUMMARY")
    print(f"{'Component':<40} {'Status':<10}")
    print(f"{'-'*50}")
    status_ok = f"{Colors.GREEN}[OK] PASS{Colors.ENDC}"
    status_fail = f"{Colors.RED}[FAIL] FAIL{Colors.ENDC}"
    
    print(f"{'Gemini API Connection':<40} {status_ok if gemini_ok else status_fail}")
    print(f"{'Hybrid Retrieval (FAISS+BM25+RRF)':<40} {status_ok if retrieval_results else status_fail}")
    print(f"{'Complete RAG Pipeline with Gemini':<40} {status_ok if rag_ok else status_fail}")
    print(f"{'Multiple Query Consistency':<40} {status_ok if multi_ok else status_fail}")
    
    all_passed = gemini_ok and retrieval_results and rag_ok and multi_ok
    
    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}*** ALL TESTS PASSED! Pipeline is working correctly with Gemini! ***{Colors.ENDC}\n")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}WARNING: Some tests failed. Please review the errors above.{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
