#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive RAG Evaluation Script

Tests the complete pipeline with metrics collection and generates:
- Console output with real-time progress
- JSON metrics report
- HTML visual report with examples

Usage:
    python evaluate_rag.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import json
import time
from typing import List, Dict
from datetime import datetime

from app.metrics import RAGMetrics, Timer
from app.rag import answer
from app.hybrid_retriever import hybrid_retriever
from app.settings import settings

# ANSI colors
class C:
    H = '\033[95m'; B = '\033[94m'; G = '\033[92m'; Y = '\033[93m'
    R = '\033[91m'; E = '\033[0m'; BOLD = '\033[1m'

def load_test_queries(filepath: str = "test_queries.json") -> List[Dict]:
    """Load test queries from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{C.Y}Warning: {filepath} not found. Using sample queries{C.E}")
        return [
            {"query": "What is theft under BNS?", "expected_sections": ["303"], "expected_keywords": ["dishonestly", "movable property"]},
            {"query": "Explain criminal conspiracy", "expected_sections": ["61"], "expected_keywords": ["two or more persons", "agree"]}
        ]

def print_header(title: str):
    print(f"\n{C.H}{C.BOLD}{'='*80}{C.E}")
    print(f"{C.H}{C.BOLD}{title.center(80)}{C.E}")
    print(f"{C.H}{C.BOLD}{'='*80}{C.E}\n")

def evaluate_single_query(metrics: RAGMetrics, test_case: Dict, query_num: int, total: int):
    """Evaluate a single query and collect metrics"""
    query = test_case['query']
    print(f"\n{C.BOLD}[Query {query_num}/{total}]{C.E} {query}")
    print("-" * 80)
    
    result = {
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'test_case': test_case
    }
    
    try:
        # Track overall query time
        query_start = time.time()
        
        # Step 1: Retrieval with timing
        with Timer(metrics, 'hybrid_retrieval'):
            retrieved_docs = hybrid_retriever.search(query, top_k=settings.TOP_K)
        
        if retrieved_docs:
            print(f"{C.G}[OK]{C.E} Retrieved {len(retrieved_docs)} documents")
            result['num_retrieved'] = len(retrieved_docs)
            
            # Evaluate retrieval quality
            retrieval_metrics = metrics.evaluate_retrieval(
                query=query,
                retrieved_docs=retrieved_docs,
                relevant_doc_ids=test_case.get('expected_sections'),
                k=5
            )
            result['retrieval_metrics'] = retrieval_metrics
            
            if 'precision@5' in retrieval_metrics:
                print(f"  Precision@5: {retrieval_metrics['precision@5']:.2f}")
                print(f"  MRR: {retrieval_metrics.get('mrr', 0):.2f}")
        else:
            print(f"{C.R}[FAIL]{C.E} No documents retrieved")
            result['error'] = 'No retrieval results'
            return result
        
        # Step 2: Answer generation with timing
        with Timer(metrics, 'llm_generation'):
            answer_result = answer(query)
        
        if answer_result and answer_result.get('answer'):
            answer_text = answer_result['answer']
            citations = answer_result.get('citations', [])
            
            # Truncate for display
            display_answer = answer_text[:200] + "..." if len(answer_text) > 200 else answer_text
            # Remove problematic Unicode for console
            display_answer = display_answer.encode('ascii', 'ignore').decode('ascii')
            
            print(f"{C.G}[OK]{C.E} Generated answer ({len(answer_text)} chars, {len(citations)} citations)")
            print(f"  Preview: {display_answer}")
            
            result['answer'] = answer_text
            result['citations'] = citations
            
            # Evaluate answer quality
            answer_metrics = metrics.evaluate_answer(
                query=query,
                answer=answer_text,
                citations=citations,
                expected_keywords=test_case.get('expected_keywords'),
                expected_sections=test_case.get('expected_sections')
            )
            result['answer_metrics'] = answer_metrics
            
            if 'keyword_coverage' in answer_metrics:
                print(f"  Keyword Coverage: {answer_metrics['keyword_coverage']:.0%}")
            if 'section_coverage' in answer_metrics:
                print(f"  Section Coverage: {answer_metrics['section_coverage']:.0%}")
        else:
            print(f"{C.R}[FAIL]{C.E} Failed to generate answer")
            result['error'] = 'Answer generation failed'
            return result
        
        # Track total query time
        query_duration = (time.time() - query_start) * 1000
        metrics.track_latency('total_query', query_duration)
        result['total_time_ms'] = query_duration
        
        print(f"{C.B}Completed in {query_duration:.0f}ms{C.E}")
        result['status'] = 'success'
        
    except Exception as e:
        print(f"{C.R}[ERROR]{C.E} {str(e)}")
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result

def generate_html_report(metrics: RAGMetrics, results: List[Dict], report_path: str):
    """Generate an HTML report with visualizations"""
    
    report = metrics.generate_report()
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RAG Performance Report - {report['session_id']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; border-left: 4px solid #1a73e8; padding-left: 15px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #1a73e8; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #1a73e8; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .score {{ font-size: 48px; font-weight: bold; }}
        .score.good {{ color: #0f9d58; }}
        .score.warning {{ color: #f4b400; }}
        .score.bad {{ color: #db4437; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #1a73e8; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .recommendation {{ background: #fff3cd; border-left: 4px solid #f4b400; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        .recommendation.good {{ background: #d4edda; border-color: #0f9d58; }}
        .query-example {{ background: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 6px; border: 1px solid #ddd; }}
        .answer {{ background: white; padding: 15px; margin-top: 10px; border-left: 3px solid #1a73e8; }}
        .timestamp {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG Performance Evaluation Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p class="timestamp">Session ID: {report['session_id']}</p>
        
        <h2>Overall Performance Score</h2>
        <div class="metric-card" style="text-align: center;">
            <div class="score {'good' if report['summary']['overall_score'] >= 70 else 'warning' if report['summary']['overall_score'] >= 50 else 'bad'}">
                {report['summary']['overall_score']:.1f}/100
            </div>
            <div class="metric-label">Overall RAG Performance</div>
        </div>
        
        <h2>Summary Metrics</h2>
        <div class="metric-grid">
"""
    
    # Add summary metrics
    if 'retrieval' in report['summary'] and report['summary']['retrieval'].get('avg_precision@5') is not None:
        html += f"""
            <div class="metric-card">
                <div class="metric-value">{report['summary']['retrieval']['avg_precision@5']:.2f}</div>
                <div class="metric-label">Avg Precision@5</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['summary']['retrieval'].get('avg_mrr', 0):.2f}</div>
                <div class="metric-label">Avg MRR</div>
            </div>
"""
    
    if 'answers' in report['summary']:
        html += f"""
            <div class="metric-card">
                <div class="metric-value">{report['summary']['answers']['total_queries']}</div>
                <div class="metric-label">Total Queries</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['summary']['answers']['completion_rate']:.0%}</div>
                <div class="metric-label">Success Rate</div>
            </div>
"""
        if report['summary']['answers'].get('avg_keyword_coverage') is not None:
            html += f"""
            <div class="metric-card">
                <div class="metric-value">{report['summary']['answers']['avg_keyword_coverage']:.0%}</div>
                <div class="metric-label">Keyword Coverage</div>
            </div>
"""
    
    html += """
        </div>
        
        <h2>Latency Breakdown</h2>
        <table>
            <tr><th>Component</th><th>Avg (ms)</th><th>Min (ms)</th><th>Max (ms)</th></tr>
"""
    
    for component, data in report['summary'].get('latency', {}).items():
        html += f"""
            <tr>
                <td>{component.replace('_', ' ').title()}</td>
                <td>{data['avg_ms']:.1f}</td>
                <td>{data['min_ms']:.1f}</td>
                <td>{data['max_ms']:.1f}</td>
            </tr>
"""
    
    html += """
        </table>
        
        <h2>Recommendations</h2>
"""
    
    for rec in report['recommendations']:
        rec_class = 'good' if 'GOOD' in rec else 'recommendation'
        html += f'<div class="{rec_class}">{rec}</div>\n'
    
    html += """
        <h2>Example Queries</h2>
"""
    
    # Add example queries
    for i, result in enumerate(results[:3], 1):
        if result['status'] == 'success':
            answer_preview = result['answer'][:500].replace('<', '&lt;').replace('>', '&gt;')
            html += f"""
        <div class="query-example">
            <strong>Query {i}:</strong> {result['query']}
            <div class="answer">
                <strong>Answer:</strong><br>
                {answer_preview}...
                <br><br>
                <strong>Citations:</strong> {len(result.get('citations', []))} sources
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return report_path

def main():
    """Main evaluation function"""
    print_header("RAG Comprehensive Evaluation")
    
    # Initialize metrics
    metrics = RAGMetrics()
    
    # Load test queries
    test_queries = load_test_queries()
    print(f"Loaded {len(test_queries)} test queries\n")
    
    # Evaluate each query
    results = []
    for i, test_case in enumerate(test_queries, 1):
        result = evaluate_single_query(metrics, test_case, i, len(test_queries))
        results.append(result)
    
    # Generate reports
    print_header("Generating Reports")
    
    # JSON report
    json_report_path = "rag_metrics_report.json"
    metrics.save_report(json_report_path)
    print(f"{C.G}[OK]{C.E} Saved JSON report: {json_report_path}")
    
    # HTML report
    html_report_path = "rag_performance_report.html"
    generate_html_report(metrics, results, html_report_path)
    print(f"{C.G}[OK]{C.E} Saved HTML report: {html_report_path}")
    
    # Print summary
    report = metrics.generate_report()
    print_header("Performance Summary")
    
    print(f"{C.BOLD}Overall Score:{C.E} {C.G if report['summary']['overall_score'] >= 70 else C.Y}{report['summary']['overall_score']:.1f}/100{C.E}")
    
    if 'answers' in report['summary']:
        print(f"{C.BOLD}Success Rate:{C.E} {report['summary']['answers']['completion_rate']:.0%}")
        print(f"{C.BOLD}Avg Citations:{C.E} {report['summary']['answers']['avg_citations']:.1f}")
    
    if 'latency' in report['summary']:
        total_latency = sum([data['avg_ms'] for data in report['summary']['latency'].values()])
        print(f"{C.BOLD}Avg Total Latency:{C.E} {total_latency:.0f}ms")
    
    print(f"\n{C.BOLD}Recommendations:{C.E}")
    for rec in report['recommendations']:
        color = C.G if 'GOOD' in rec else C.Y
        print(f"  {color}•{C.E} {rec}")
    
    print(f"\n{C.BOLD}Reports saved. Open {html_report_path} in browser for detailed view.{C.E}\n")

if __name__ == "__main__":
    main()
