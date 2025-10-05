"""
Evaluation metrics cho RAG system.
Updated để xử lý format output từ retriever với metadata source
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Union
from collections import defaultdict
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluator cho RAG system với nhiều metrics"""
    
    def __init__(self):
        self.results = defaultdict(list)
        
    def extract_source_from_metadata(self, metadata: Union[Dict, str]) -> str:
        """
        Extract source information từ metadata
        
        Args:
            metadata: Dictionary hoặc string chứa source info
        """
        if isinstance(metadata, dict):
            return metadata.get('source', '')
        elif isinstance(metadata, str):
            # Extract từ string format {'source': '...'}
            match = re.search(r"'source':\s*'([^']+)'", metadata)
            if match:
                return match.group(1)
        return ''
        
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Tính Precision@K
        
        Args:
            retrieved_docs: Danh sách doc_ids được retrieve
            relevant_docs: Danh sách doc_ids thực sự relevant
            k: Số lượng top documents để xem xét
        """
        if not retrieved_docs or k == 0:
            return 0.0
            
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant_docs))
        
        return relevant_retrieved / k
    
    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Tính Recall@K
        """
        if not relevant_docs:
            return 1.0  # Không có relevant docs -> recall = 1
            
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant_docs))
        
        return relevant_retrieved / len(relevant_docs)
    
    def f1_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Tính F1@K
        """
        p = self.precision_at_k(retrieved_docs, relevant_docs, k)
        r = self.recall_at_k(retrieved_docs, relevant_docs, k)
        
        if p + r == 0:
            return 0.0
            
        return 2 * (p * r) / (p + r)
    
    def average_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Tính Average Precision cho một query
        """
        if not relevant_docs:
            return 1.0  # Không có relevant docs
            
        ap = 0.0
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_count += 1
                ap += relevant_count / (i + 1)
        
        if relevant_count == 0:
            return 0.0
            
        return ap / len(relevant_docs)
    
    def mean_average_precision(self, all_results: List[Tuple[List[str], List[str]]]) -> float:
        """
        Tính MAP cho tất cả queries
        
        Args:
            all_results: List của tuples (retrieved_docs, relevant_docs)
        """
        if not all_results:
            return 0.0
            
        ap_scores = []
        for retrieved, relevant in all_results:
            ap = self.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    def keyword_coverage(self, text: str, keywords: List[str]) -> float:
        """
        Tính tỷ lệ keywords xuất hiện trong text
        """
        if not keywords:
            return 1.0  # Không có keywords nào để check
            
        text_lower = text.lower()
        found_keywords = 0
        
        for keyword in keywords:
            # Normalize keyword
            keyword_lower = keyword.lower().replace(",", "").replace(".", "").replace("đ", "d")
            if keyword_lower in text_lower.replace("đ", "d"):
                found_keywords += 1
        
        return found_keywords / len(keywords)
    
    def source_match_score(self, retrieved_sources: List[str], expected_sources: List[str]) -> float:
        """
        Tính điểm match cho sources
        """
        if not expected_sources:
            return 1.0
            
        matched = 0
        for expected_src in expected_sources:
            for retrieved_src in retrieved_sources:
                # Partial match cho source names
                if expected_src.lower() in retrieved_src.lower():
                    matched += 1
                    break
        
        return matched / len(expected_sources)
    
    def evaluate_retrieval(
        self, 
        query: str,
        retrieved_results: List[Dict[str, Any]], 
        expected_doc_ids: List[str],
        expected_keywords: List[str],
        expected_sources: List[str] = None,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Đánh giá kết quả retrieval cho một query
        
        Args:
            query: Query string
            retrieved_results: Kết quả từ search function
            expected_doc_ids: Doc IDs expected
            expected_keywords: Keywords expected trong kết quả
            expected_sources: Expected source files
            k_values: Các giá trị k để tính metrics
        """
        # Extract doc_ids, texts, sources từ retrieved results
        retrieved_doc_ids = []
        retrieved_texts = []
        retrieved_sources = []
        
        for result in retrieved_results:
            # Get doc_id
            doc_id = result.get('doc_id', '')
            if doc_id:
                retrieved_doc_ids.append(str(doc_id))
            
            # Get text content
            text = result.get('text', '')
            retrieved_texts.append(text)
            
            # Get source metadata
            metadata = result.get('metadata', {})
            source = self.extract_source_from_metadata(metadata)
            if source:
                retrieved_sources.append(source)
        
        # Tính metrics
        metrics = {
            'query': query,
            'num_retrieved': len(retrieved_doc_ids),
            'num_expected': len(expected_doc_ids) if expected_doc_ids else 0
        }
        
        # Check if negative query (no expected results)
        if not expected_doc_ids and not expected_keywords:
            # For negative queries, good if no results retrieved
            metrics['is_negative_query'] = True
            metrics['negative_query_success'] = (len(retrieved_doc_ids) == 0)
            return metrics
        
        # Precision, Recall, F1 tại các k values
        for k in k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(retrieved_doc_ids, expected_doc_ids, k)
            metrics[f'recall@{k}'] = self.recall_at_k(retrieved_doc_ids, expected_doc_ids, k)
            metrics[f'f1@{k}'] = self.f1_at_k(retrieved_doc_ids, expected_doc_ids, k)
        
        # Average Precision
        metrics['average_precision'] = self.average_precision(retrieved_doc_ids, expected_doc_ids)
        
        # Keyword coverage cho top 5 results
        if retrieved_texts:
            combined_text = ' '.join(retrieved_texts[:5])
            metrics['keyword_coverage'] = self.keyword_coverage(combined_text, expected_keywords)
        else:
            metrics['keyword_coverage'] = 0.0
        
        # Source match score
        if expected_sources:
            metrics['source_match'] = self.source_match_score(retrieved_sources, expected_sources)
        
        # Hit rate (có ít nhất 1 relevant doc trong top k)
        for k in k_values:
            retrieved_at_k = retrieved_doc_ids[:k]
            metrics[f'hit@{k}'] = 1.0 if any(doc in expected_doc_ids for doc in retrieved_at_k) else 0.0
        
        # Debug info
        metrics['debug'] = {
            'retrieved_docs': retrieved_doc_ids[:5],  # Top 5 for debug
            'expected_docs': expected_doc_ids[:5],
            'retrieved_sources': retrieved_sources[:5]
        }
        
        return metrics
    
    def aggregate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Tổng hợp metrics từ nhiều queries
        """
        if not all_metrics:
            return {}
        
        # Separate negative queries
        negative_metrics = [m for m in all_metrics if m.get('is_negative_query', False)]
        positive_metrics = [m for m in all_metrics if not m.get('is_negative_query', False)]
        
        aggregated = {}
        
        # Process negative queries
        if negative_metrics:
            success_rate = sum(1 for m in negative_metrics if m.get('negative_query_success', False))
            aggregated['negative_query_accuracy'] = success_rate / len(negative_metrics)
            aggregated['num_negative_queries'] = len(negative_metrics)
        
        # Process positive queries
        if positive_metrics:
            # Lấy tất cả metric keys (trừ các non-numeric fields)
            exclude_keys = {'query', 'num_retrieved', 'num_expected', 'debug', 'is_negative_query'}
            metric_keys = set()
            for m in positive_metrics:
                metric_keys.update(k for k in m.keys() if k not in exclude_keys)
            
            # Tính mean cho mỗi metric
            for key in metric_keys:
                values = [m[key] for m in positive_metrics if key in m]
                if values:
                    aggregated[f'mean_{key}'] = np.mean(values)
                    aggregated[f'std_{key}'] = np.std(values)
            
            # Tính MAP
            all_ap = [m['average_precision'] for m in positive_metrics if 'average_precision' in m]
            if all_ap:
                aggregated['MAP'] = np.mean(all_ap)
            
            # Overall statistics
            aggregated['num_positive_queries'] = len(positive_metrics)
            aggregated['avg_retrieved'] = np.mean([m['num_retrieved'] for m in positive_metrics])
        
        aggregated['total_queries'] = len(all_metrics)
        
        return aggregated
    
    def print_metrics_summary(self, metrics: Dict[str, float], title: str = "Metrics Summary"):
        """
        In summary của metrics một cách đẹp
        """
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        
        # Group metrics by type
        precision_metrics = {k: v for k, v in metrics.items() if 'precision' in k.lower()}
        recall_metrics = {k: v for k, v in metrics.items() if 'recall' in k.lower()}
        f1_metrics = {k: v for k, v in metrics.items() if 'f1' in k.lower()}
        hit_metrics = {k: v for k, v in metrics.items() if 'hit' in k.lower()}
        other_metrics = {k: v for k, v in metrics.items() 
                        if k not in precision_metrics and k not in recall_metrics 
                        and k not in f1_metrics and k not in hit_metrics}
        
        # Print each group
        if precision_metrics:
            print("\nPRECISION METRICS:")
            for key, value in sorted(precision_metrics.items()):
                print(f"  {key:30}: {value:.4f}")
        
        if recall_metrics:
            print("\nRECALL METRICS:")
            for key, value in sorted(recall_metrics.items()):
                print(f"  {key:30}: {value:.4f}")
        
        if f1_metrics:
            print("\nF1 METRICS:")
            for key, value in sorted(f1_metrics.items()):
                print(f"  {key:30}: {value:.4f}")
        
        if hit_metrics:
            print("\nHIT RATE METRICS:")
            for key, value in sorted(hit_metrics.items()):
                print(f"  {key:30}: {value:.4f}")
        
        if other_metrics:
            print("\nOTHER METRICS:")
            for key, value in sorted(other_metrics.items()):
                if isinstance(value, float):
                    print(f"  {key:30}: {value:.4f}")
                else:
                    print(f"  {key:30}: {value}")
        
        print(f"{'='*60}\n")