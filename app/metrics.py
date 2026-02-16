"""
RAG Performance Metrics and Evaluation System

Tracks and evaluates:
- Retrieval quality (Precision@K, Recall@K, MRR)
- Answer relevance and quality
- Citation accuracy
- Component latency
- Provides self-improvement recommendations
"""

import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import statistics

class RAGMetrics:
    """Comprehensive RAG evaluation and metrics tracking"""
    
    def __init__(self):
        self.metrics_history = []
        self.current_session = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'queries': [],
            'retrieval_metrics': [],
            'answer_metrics': [],
            'latency_metrics': {}
        }
    
    # ===== RETRIEVAL METRICS =====
    
    def evaluate_retrieval(self, 
                          query: str,
                          retrieved_docs: List[Dict],
                          relevant_doc_ids: Optional[List[str]] = None,
                          k: int = 5) -> Dict[str, float]:
        """
        Evaluate retrieval quality
        
        Args:
            query: The search query
            retrieved_docs: List of retrieved documents with scores
            relevant_doc_ids: List of truly relevant document IDs (for ground truth)
            k: Top-K to evaluate
        
        Returns:
            Dictionary with precision@k, recall@k, MRR
        """
        metrics = {
            'query': query,
            'num_retrieved': len(retrieved_docs),
            'timestamp': datetime.now().isoformat()
        }
        
        # If we have ground truth, calculate precision/recall
        if relevant_doc_ids:
            top_k_docs = retrieved_docs[:k]
            retrieved_ids = [doc.get('id', doc.get('filename', '')) for doc in top_k_docs]
            
            # Precision@K = relevant docs in top-K / K
            relevant_in_topk = len(set(retrieved_ids) & set(relevant_doc_ids))
            metrics['precision@' + str(k)] = relevant_in_topk / k if k > 0 else 0
            
            # Recall@K = relevant docs in top-K / total relevant
            metrics['recall@' + str(k)] = relevant_in_topk / len(relevant_doc_ids) if relevant_doc_ids else 0
            
            # MRR (Mean Reciprocal Rank) - rank of first relevant doc
            first_relevant_rank = None
            for i, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_doc_ids:
                    first_relevant_rank = i
                    break
            
            metrics['mrr'] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        
        # Score distribution analysis
        if retrieved_docs:
            scores = [doc.get('score', 0) for doc in retrieved_docs[:k]]
            metrics['avg_score'] = statistics.mean(scores) if scores else 0
            metrics['max_score'] = max(scores) if scores else 0
            metrics['min_score'] = min(scores) if scores else 0
            metrics['score_variance'] = statistics.variance(scores) if len(scores) > 1 else 0
        
        self.current_session['retrieval_metrics'].append(metrics)
        return metrics
    
    # ===== ANSWER QUALITY METRICS =====
    
    def evaluate_answer(self,
                       query: str,
                       answer: str,
                       citations: List[Dict],
                       expected_keywords: Optional[List[str]] = None,
                       expected_sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate answer quality
        
        Args:
            query: The original query
            answer: Generated answer
            citations: List of citation dictionaries
            expected_keywords: Keywords that should appear in answer
            expected_sections: Expected legal sections to be cited
        
        Returns:
            Dictionary with answer quality metrics
        """
        metrics = {
            'query': query,
            'answer_length': len(answer),
            'num_citations': len(citations),
            'timestamp': datetime.now().isoformat()
        }
        
        # Answer completeness
        metrics['has_answer'] = len(answer) > 50  # Non-trivial answer
        metrics['has_citations'] = len(citations) > 0
        
        # Keyword coverage (if provided)
        if expected_keywords:
            answer_lower = answer.lower()
            found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
            metrics['keyword_coverage'] = len(found_keywords) / len(expected_keywords)
            metrics['found_keywords'] = found_keywords
            metrics['missing_keywords'] = list(set(expected_keywords) - set(found_keywords))
        
        # Section citation accuracy (if provided)
        if expected_sections:
            cited_sections = []
            for cite in citations:
                # Extract section numbers from citations
                ref_text = cite.get('ref', '') + cite.get('title', '')
                for section in expected_sections:
                    if section in ref_text:
                        cited_sections.append(section)
            
            metrics['section_coverage'] = len(set(cited_sections)) / len(expected_sections) if expected_sections else 0
            metrics['cited_sections'] = list(set(cited_sections))
            metrics['missing_sections'] = list(set(expected_sections) - set(cited_sections))
        
        # Answer structure checks
        metrics['is_structured'] = any(marker in answer for marker in ['Section', 'Definition', '1)', '2)'])
        metrics['cites_sources'] = any(marker in answer for marker in ['[', 'Section', 'according to'])
        
        self.current_session['answer_metrics'].append(metrics)
        return metrics
    
    # ===== LATENCY TRACKING =====
    
    def track_latency(self, component: str, duration: float):
        """Track component execution time"""
        if component not in self.current_session['latency_metrics']:
            self.current_session['latency_metrics'][component] = []
        
        self.current_session['latency_metrics'][component].append(duration)
    
    def get_avg_latency(self, component: str) -> float:
        """Get average latency for a component"""
        latencies = self.current_session['latency_metrics'].get(component, [])
        return statistics.mean(latencies) if latencies else 0.0
    
    # ===== REPORTING =====
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'session_id': self.current_session['session_id'],
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_metrics': self.current_session,
            'recommendations': self.get_recommendations()
        }
        
        # Aggregate retrieval metrics
        if self.current_session['retrieval_metrics']:
            retrieval_data = self.current_session['retrieval_metrics']
            report['summary']['retrieval'] = {
                'avg_precision@5': statistics.mean([m.get('precision@5', 0) for m in retrieval_data if 'precision@5' in m]) if any('precision@5' in m for m in retrieval_data) else None,
                'avg_recall@5': statistics.mean([m.get('recall@5', 0) for m in retrieval_data if 'recall@5' in m]) if any('recall@5' in m for m in retrieval_data) else None,
                'avg_mrr': statistics.mean([m.get('mrr', 0) for m in retrieval_data if 'mrr' in m]) if any('mrr' in m for m in retrieval_data) else None,
                'avg_num_retrieved': statistics.mean([m['num_retrieved'] for m in retrieval_data])
            }
        
        # Aggregate answer metrics
        if self.current_session['answer_metrics']:
            answer_data = self.current_session['answer_metrics']
            report['summary']['answers'] = {
                'total_queries': len(answer_data),
                'avg_answer_length': statistics.mean([m['answer_length'] for m in answer_data]),
                'avg_citations': statistics.mean([m['num_citations'] for m in answer_data]),
                'completion_rate': sum([m['has_answer'] for m in answer_data]) / len(answer_data),
                'citation_rate': sum([m['has_citations'] for m in answer_data]) / len(answer_data),
                'avg_keyword_coverage': statistics.mean([m.get('keyword_coverage', 0) for m in answer_data if 'keyword_coverage' in m]) if any('keyword_coverage' in m for m in answer_data) else None,
            }
        
        # Aggregate latency metrics
        report['summary']['latency'] = {}
        for component, latencies in self.current_session['latency_metrics'].items():
            report['summary']['latency'][component] = {
                'avg_ms': statistics.mean(latencies),
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0
            }
        
        # Overall performance score (0-100)
        report['summary']['overall_score'] = self._calculate_overall_score(report['summary'])
        
        return report
    
    def _calculate_overall_score(self, summary: Dict) -> float:
        """Calculate overall RAG performance score (0-100)"""
        score = 0
        factors = 0
        
        # Retrieval quality (40 points max)
        if 'retrieval' in summary and summary['retrieval'].get('avg_precision@5'):
            score += summary['retrieval']['avg_precision@5'] * 40
            factors += 1
        
        # Answer quality (40 points max)
        if 'answers' in summary:
            answer_score = 0
            if summary['answers'].get('completion_rate'):
                answer_score += summary['answers']['completion_rate'] * 15
            if summary['answers'].get('citation_rate'):
                answer_score += summary['answers']['citation_rate'] * 15
            if summary['answers'].get('avg_keyword_coverage'):
                answer_score += summary['answers']['avg_keyword_coverage'] * 10
            score += answer_score
            factors += 1
        
        # Latency (20 points max - faster is better)
        if 'latency' in summary:
            total_latency = sum([data['avg_ms'] for data in summary['latency'].values()])
            # Score decreases as latency increases (target < 5000ms)
            latency_score = max(0, min(20, 20 * (1 - total_latency / 10000)))
            score += latency_score
            factors += 1
        
        return score / factors if factors > 0 else 0
    
    def get_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        # Analyze retrieval metrics
        if self.current_session['retrieval_metrics']:
            metrics = self.current_session['retrieval_metrics']
            avg_precision = statistics.mean([m.get('precision@5', 0) for m in metrics if 'precision@5' in m]) if any('precision@5' in m for m in metrics) else None
            
            if avg_precision is not None and avg_precision < 0.6:
                recommendations.append("LOW PRECISION: Consider improving embeddings or adjusting BM25 weights")
            
            avg_score_variance = statistics.mean([m.get('score_variance', 0) for m in metrics])
            if avg_score_variance < 0.5:
                recommendations.append("LOW SCORE VARIANCE: Results may be too similar - consider diversification")
        
        # Analyze answer metrics
        if self.current_session['answer_metrics']:
            metrics = self.current_session['answer_metrics']
            avg_coverage = statistics.mean([m.get('keyword_coverage', 1) for m in metrics if 'keyword_coverage' in m]) if any('keyword_coverage' in m for m in metrics) else 1
            
            if avg_coverage < 0.7:
                recommendations.append("LOW KEYWORD COVERAGE: LLM may not be using retrieved context effectively")
            
            citation_rate = sum([m['has_citations'] for m in metrics]) / len(metrics)
            if citation_rate < 0.9:
                recommendations.append("MISSING CITATIONS: Ensure prompt requires citations for all answers")
        
        # Analyze latency
        if self.current_session['latency_metrics']:
            total_avg = sum([statistics.mean(latencies) for latencies in self.current_session['latency_metrics'].values()])
            if total_avg > 5000:
                recommendations.append("HIGH LATENCY: Consider caching, async processing, or model optimization")
        
        if not recommendations:
            recommendations.append("PERFORMANCE GOOD: All metrics within acceptable ranges")
        
        return recommendations
    
    def save_report(self, filepath: str):
        """Save report to JSON file"""
        report = self.generate_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return filepath

# Context manager for timing
class Timer:
    """Context manager for measuring execution time"""
    def __init__(self, metrics: RAGMetrics, component_name: str):
        self.metrics = metrics
        self.component_name = component_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000  # Convert to ms
        self.metrics.track_latency(self.component_name, duration)
