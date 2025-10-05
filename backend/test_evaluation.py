"""
Test script để evaluate RAG system với 3 search methods
Updated để handle doc_id format và metadata
"""

import json
import sys
from typing import Dict, Any, List
from pathlib import Path
import logging

# Import modules
from ground_truth_queries import (
    GROUND_TRUTH_QUERIES, 
    get_queries_by_category, 
    get_all_categories,
    convert_index_to_doc_id
)
from evaluation import RAGEvaluator
from search import hybrid_search, search_dense_only, search_bm25_only

# Optional: import LangChain RAG
USE_LANGCHAIN = False
if USE_LANGCHAIN:
    try:
        from langchain_rag import LangChainRAG, SearchMethod, get_langchain_rag
        USE_LANGCHAIN = True
    except ImportError:
        print("LangChain RAG not available")
        USE_LANGCHAIN = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystemTester:
    """Main tester class cho RAG system"""
    
    def __init__(self):
        self.evaluator = RAGEvaluator()
        self.search_methods = {
            'hybrid': hybrid_search,
            'dense': search_dense_only,
            'bm25': search_bm25_only
        }
        
        # Load final_output.json for mapping
        self.load_document_mapping()
    
    
        
        
    def load_document_mapping(self):
        """Load document mapping từ final_output.json"""
        try:
            with open('final_output.json', 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from final_output.json")
        except Exception as e:
            logger.warning(f"Could not load final_output.json: {e}")
            self.documents = []
    
    def test_search_method(
        self, 
        method_name: str,
        queries: List[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Test một search method với ground truth queries
        
        Args:
            method_name: Tên method (hybrid/dense/bm25)
            queries: Danh sách queries để test (default: all)
            top_k: Số documents retrieve
        """
        if queries is None:
            queries = GROUND_TRUTH_QUERIES
            
        search_func = self.search_methods[method_name]
        all_metrics = []
        
        print(f"\nTesting {method_name.upper()} search with {len(queries)} queries...")
        
        for i, query_data in enumerate(queries):
            query = query_data['query']
            expected_doc_ids = query_data.get('expected_doc_ids', [])
            expected_keywords = query_data.get('expected_keywords', [])
            expected_sources = query_data.get('expected_sources', [])
            
            # Retrieve documents
            try:
                results = search_func(query, top_k)
                
                # Process results to ensure correct format
                processed_results = []
                for res in results:
                    # Ensure doc_id is in correct format
                    doc_id = res.get('doc_id', '')
                    if doc_id and not doc_id.startswith('doc_'):
                        # Convert index to doc_id format if needed
                        try:
                            idx = int(doc_id)
                            doc_id = convert_index_to_doc_id(idx)
                        except:
                            pass
                    
                    processed_result = {
                        'doc_id': doc_id,
                        'text': res.get('text', ''),
                        'score': res.get('score', 0),
                        'metadata': res.get('metadata', {})
                    }
                    
                    # Add source info if available
                    if 'search_type' in res:
                        processed_result['search_type'] = res['search_type']
                    
                    processed_results.append(processed_result)
                
                results = processed_results
                
            except Exception as e:
                logger.error(f"Error in query {i+1} '{query[:50]}...': {e}")
                continue
            
            # Evaluate
            metrics = self.evaluator.evaluate_retrieval(
                query=query,
                retrieved_results=results,
                expected_doc_ids=expected_doc_ids,
                expected_keywords=expected_keywords,
                expected_sources=expected_sources,
                k_values=[1, 3, 5, 10]
            )
            
            all_metrics.append(metrics)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(queries)} queries...")
        
        # Aggregate metrics
        aggregated = self.evaluator.aggregate_metrics(all_metrics)
        aggregated['method'] = method_name
        aggregated['num_queries'] = len(all_metrics)
        
        return aggregated
    
    def test_langchain_rag(
        self,
        queries: List[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Test LangChain RAG if available"""
        if not USE_LANGCHAIN:
            print("LangChain RAG not available")
            return {}
        
        if queries is None:
            queries = GROUND_TRUTH_QUERIES
        
        rag = get_langchain_rag()
        # Update top_k in config
        rag.config.default_top_k = top_k
        
        all_metrics = []
        print(f"\nTesting LANGCHAIN RAG with {len(queries)} queries...")
        
        for i, query_data in enumerate(queries):
            query = query_data['query']
            expected_doc_ids = query_data.get('expected_doc_ids', [])
            expected_keywords = query_data.get('expected_keywords', [])
            expected_sources = query_data.get('expected_sources', [])
            
            try:
                # Process with LangChain RAG
                result = rag.process(
                    query=query,
                    search_method=SearchMethod.HYBRID,
                    include_history=False
                )
                
                # Extract retrieved documents from sources
                retrieved_results = []
                if 'sources' in result:
                    for src in result['sources']:
                        retrieved_results.append({
                            'doc_id': src.get('doc_id', ''),
                            'text': src.get('content', ''),
                            'score': src.get('score', 0),
                            'metadata': {'source': src.get('doc_id', '')}
                        })
                
                # Evaluate
                metrics = self.evaluator.evaluate_retrieval(
                    query=query,
                    retrieved_results=retrieved_results,
                    expected_doc_ids=expected_doc_ids,
                    expected_keywords=expected_keywords,
                    expected_sources=expected_sources
                )
                
                # Add answer quality check
                answer = result.get('answer', '')
                if answer:
                    metrics['answer_keyword_coverage'] = self.evaluator.keyword_coverage(
                        answer, expected_keywords
                    )
                
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error in LangChain query {i+1}: {e}")
                continue
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(queries)} queries...")
        
        # Aggregate
        aggregated = self.evaluator.aggregate_metrics(all_metrics)
        aggregated['method'] = 'langchain_rag'
        aggregated['num_queries'] = len(all_metrics)
        
        return aggregated
    
    def compare_all_methods(self, queries: List[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        So sánh tất cả search methods
        """
        if queries is None:
            queries = GROUND_TRUTH_QUERIES
            
        results = {}
        
        # Test basic search methods
        for method_name in self.search_methods.keys():
            results[method_name] = self.test_search_method(method_name, queries)
        
        # Test LangChain RAG if available
        if USE_LANGCHAIN:
            results['langchain'] = self.test_langchain_rag(queries)
        
        return results
    
    def test_by_category(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Test theo từng category của queries
        """
        categories = get_all_categories()
        results = {}
        
        for category in categories:
            print(f"\n{'='*60}")
            print(f"Testing category: {category}")
            print(f"{'='*60}")
            
            category_queries = get_queries_by_category(category)
            print(f"  Found {len(category_queries)} queries in category")
            
            results[category] = self.compare_all_methods(category_queries)
        
        return results
    
    def generate_report(self, results: Dict[str, Any], save_to_file: bool = True) -> str:
        """
        Generate báo cáo chi tiết
        """
        report = []
        report.append("=" * 80)
        report.append("RAG SYSTEM EVALUATION REPORT".center(80))
        report.append("=" * 80)
        
        # Check type of results
        is_category_results = False
        if results and isinstance(list(results.values())[0], dict):
            first_value = list(results.values())[0]
            if isinstance(first_value, dict) and any(isinstance(v, dict) for v in first_value.values()):
                is_category_results = True
        
        # Overall results
        if not is_category_results:
            report.append("\n## OVERALL COMPARISON\n")
            
            # Create comparison table
            methods = list(results.keys())
            metrics_to_show = ['MAP', 'mean_precision@1', 'mean_precision@3','mean_precision@5','mean_recall@1', 
                              'mean_recall@3','mean_recall@5', 'mean_recall@10', 'mean_f1@10','mean_f1@5', 'mean_hit@1', 
                              'mean_keyword_coverage',
                              ]
            
            # Header
            header = f"{'Metric':<30}"
            for method in methods:
                header += f"{method:>15}"
            report.append(header)
            report.append("-" * (30 + 15 * len(methods)))
            
            # Metrics rows
            for metric in metrics_to_show:
                row = f"{metric:<30}"
                best_value = -1
                best_method = None
                
                # Find best value
                for method in methods:
                    if metric in results[method]:
                        value = results[method][metric]
                        if value > best_value:
                            best_value = value
                            best_method = method
                
                # Format row with highlighting
                for method in methods:
                    if metric in results[method]:
                        value = results[method][metric]
                        if method == best_method and best_value > 0:
                            row += f"{'*%.4f' % value:>15}"
                        else:
                            row += f"{value:>15.4f}"
                    else:
                        row += f"{'N/A':>15}"
                
                report.append(row)
            
            report.append("\n* = Best performing method")
        
        # Category results
        if is_category_results:
            report.append("\n\n## RESULTS BY CATEGORY\n")
            
            for category, cat_results in results.items():
                report.append(f"\n### {category.upper()}")
                report.append("-" * 40)
                
                # Summary for this category
                for method, metrics in cat_results.items():
                    if isinstance(metrics, dict):
                        map_score = metrics.get('MAP', 0)
                        p5 = metrics.get('mean_precision@5', 0)
                        r5 = metrics.get('mean_recall@5', 0)
                        f1_5 = metrics.get('mean_f1@5', 0)
                        
                        report.append(
                            f"  {method:10}: MAP={map_score:.3f}, "
                            f"P@5={p5:.3f}, R@5={r5:.3f}, F1@5={f1_5:.3f}"
                        )
        
        # Recommendations
        report.append("\n\n## RECOMMENDATIONS\n")
        
        # Find best method overall
        all_methods = {}
        if not is_category_results:
            all_methods = results
        else:
            # Aggregate across categories
            for cat_results in results.values():
                for method, metrics in cat_results.items():
                    if method not in all_methods:
                        all_methods[method] = []
                    if 'MAP' in metrics:
                        all_methods[method].append(metrics['MAP'])
        
        if all_methods:
            # Calculate average MAP
            avg_map = {}
            for method, maps in all_methods.items():
                if isinstance(maps, list):
                    avg_map[method] = np.mean(maps) if maps else 0
                elif isinstance(maps, dict):
                    avg_map[method] = maps.get('MAP', 0)
                else:
                    avg_map[method] = 0
            
            if avg_map:
                best_method = max(avg_map, key=avg_map.get)
                report.append(f"1. Best overall method: {best_method.upper()} (MAP={avg_map[best_method]:.4f})")
                
                # Analyze strengths/weaknesses
                if 'hybrid' in avg_map and 'dense' in avg_map and 'bm25' in avg_map:
                    if avg_map['hybrid'] > max(avg_map['dense'], avg_map['bm25']):
                        report.append("2. Hybrid search effectively combines semantic and lexical matching")
                    elif avg_map['dense'] > avg_map['bm25']:
                        report.append("3. Dense retrieval performs better - corpus has good semantic structure")
                    else:
                        report.append("4. BM25 performs better - consider improving embeddings")
        
        report.append("\n" + "=" * 80)
        
        # Join report
        report_text = "\n".join(report)
        
        # Save to file
        if save_to_file:
            with open("evaluation_report.txt", "w", encoding="utf-8") as f:
                f.write(report_text)
            print("\nReport saved to evaluation_report.txt")
        
        return report_text

def main():
    """Main test function"""
    
    print("="*60)
    print("RAG SYSTEM EVALUATION".center(60))
    print("="*60)
    
    tester = RAGSystemTester()
    
    # Menu
    print("\nSelect test mode:")
    print("1. Test all methods with all queries")
    print("2. Test by query category")
    print("3. Test specific method")
    print("4. Quick test (5 queries only)")
    print("5. Full evaluation with report")
    if USE_LANGCHAIN:
        print("6. Test LangChain RAG only")
    
    choice = input(f"\nEnter choice (1-{6 if USE_LANGCHAIN else 5}): ").strip()
    
    if choice == "1":
        # Test all methods
        results = tester.compare_all_methods()
        
        # Print results
        for method, metrics in results.items():
            tester.evaluator.print_metrics_summary(metrics, f"{method.upper()} Search Results")
        
        # Generate report
        report = tester.generate_report(results)
        print("\n" + report)
        
    elif choice == "2":
        # Test by category
        results = tester.test_by_category()
        
        # Print summary
        for category, cat_results in results.items():
            print(f"\n### Category: {category}")
            for method, metrics in cat_results.items():
                map_val = metrics.get('MAP', 0)
                print(f"  {method}: MAP={map_val:.4f}")
        
        # Generate report
        report = tester.generate_report(results)
        print("\n" + report)
        
    elif choice == "3":
        # Test specific method
        print("\nSelect method:")
        print("1. Hybrid")
        print("2. Dense")
        print("3. BM25")
        if USE_LANGCHAIN:
            print("4. LangChain RAG")
        
        method_choice = input(f"Enter choice (1-{4 if USE_LANGCHAIN else 3}): ").strip()
        method_map = {"1": "hybrid", "2": "dense", "3": "bm25"}
        if USE_LANGCHAIN:
            method_map["4"] = "langchain"
        
        if method_choice in method_map:
            method = method_map[method_choice]
            
            if method == "langchain":
                results = tester.test_langchain_rag()
            else:
                results = tester.test_search_method(method)
                
            tester.evaluator.print_metrics_summary(results, f"{method.upper()} Results")
        
    elif choice == "4":
        # Quick test
        print("\nRunning quick test with 5 queries...")
        test_queries = GROUND_TRUTH_QUERIES[:5]
        
        results = tester.compare_all_methods(test_queries)
        
        for method, metrics in results.items():
            map_val = metrics.get('MAP', 0)
            p5_val = metrics.get('mean_precision@5', 0)
            print(f"\n{method.upper()}: MAP={map_val:.4f}, P@5={p5_val:.4f}")
    
    elif choice == "5":
        # Full evaluation
        print("\nRunning full evaluation...")
        
        # Test all methods
        overall_results = tester.compare_all_methods()
        
        # Test by category
        category_results = tester.test_by_category()
        
        # Combine results
        combined_results = {
            'overall': overall_results,
            'by_category': category_results
        }
        
        # Save detailed results to JSON
        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        
        print("\nDetailed results saved to evaluation_results.json")
        
        # Generate and save report
        report = tester.generate_report(overall_results)
        print("\n" + report)
        
    elif choice == "6" and USE_LANGCHAIN:
        # Test LangChain RAG only
        print("\nTesting LangChain RAG...")
        results = tester.test_langchain_rag()
        tester.evaluator.print_metrics_summary(results, "LANGCHAIN RAG Results")
        
    else:
        print("Invalid choice")
        
    print("\nEvaluation complete!")

if __name__ == "__main__":
    import numpy as np  # Import numpy for report generation
    main()