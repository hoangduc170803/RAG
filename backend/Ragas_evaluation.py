"""
RAGAS Evaluation với Offline Embeddings
Sử dụng HuggingFace embeddings local và TEI service thay vì API calls
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import requests

# RAGAS imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_recall,
    context_precision,
    answer_correctness
)

# LangChain imports for local models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

# Local imports
from ground_truth_queries import GROUND_TRUTH_QUERIES
from evaluation import RAGEvaluator
from search import hybrid_search, search_dense_only, search_bm25_only

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TEIEmbeddings(Embeddings):
    """Custom embeddings class sử dụng TEI service (Text Embeddings Inference)"""
    
    def __init__(self, tei_url: str = "http://localhost:8081"):
        self.tei_url = tei_url.rstrip('/')
        
        # Test connection
        try:
            response = requests.get(f"{self.tei_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ TEI service connected at {self.tei_url}")
            else:
                logger.warning(f"⚠️ TEI service responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to TEI service: {e}")
            raise ConnectionError(f"TEI service not available at {self.tei_url}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using TEI service"""
        try:
            response = requests.post(
                f"{self.tei_url}/embed",
                json={"inputs": texts},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            embeddings = response.json()
            if isinstance(embeddings, list) and len(embeddings) == len(texts):
                return embeddings
            else:
                raise ValueError("Invalid response format from TEI service")
                
        except Exception as e:
            logger.error(f"Error getting embeddings from TEI: {e}")
            # Fallback to zero embeddings
            return [[0.0] * 384 for _ in texts]  # Assuming 384-dim embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query using TEI service"""
        try:
            embeddings = self.embed_documents([text])
            return embeddings[0]
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 384

@dataclass
class RAGASTestCase:
    """Data structure cho RAGAS test case"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

class OfflineRAGASEvaluator:
    """RAGAS Evaluator sử dụng offline models và local services"""
    
    def __init__(self, 
                 gemini_api_key: str,
                 tei_url: str = "http://localhost:8081",
                 hf_model_name: str = "bge",
                 use_tei: bool = True):
        """
        Initialize offline RAGAS evaluator
        
        Args:
            gemini_api_key: Google API key cho LLM
            tei_url: URL của TEI service
            hf_model_name: HuggingFace model name cho embeddings
            use_tei: Sử dụng TEI service (True) hoặc HuggingFace local (False)
        """
        self.gemini_api_key = gemini_api_key
        self.tei_url = tei_url
        self.hf_model_name = hf_model_name
        self.use_tei = use_tei
        
        # Initialize LLM (Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-3-12b-it",
            google_api_key=self.gemini_api_key,
            temperature=0.1
        )
        
        # Initialize Embeddings
        self.embeddings = self._init_embeddings()
        
        # RAGAS metrics (reduced để tránh quá nhiều API calls)
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
        ]
        
        logger.info(f"Initialized offline RAGAS evaluator:")
        logger.info(f"  - LLM: Gemini 1.5 Flash")
        logger.info(f"  - Embeddings: {'TEI Service' if use_tei else 'HuggingFace Local'}")
        logger.info(f"  - Metrics: {len(self.metrics)} metrics")
    
    def _init_embeddings(self) -> Embeddings:
        """Initialize embeddings model"""
        if self.use_tei:
            try:
                return TEIEmbeddings(self.tei_url)
            except Exception as e:
                logger.warning(f"TEI initialization failed: {e}")
                logger.info("Falling back to HuggingFace local embeddings...")
                self.use_tei = False
        
        # Fallback to HuggingFace local
        try:
            return HuggingFaceEmbeddings(
                model_name=self.hf_model_name,
                model_kwargs={'device': 'cpu'},  # Sử dụng CPU
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"HuggingFace embeddings initialization failed: {e}")
            raise RuntimeError("Cannot initialize any embeddings model")
    
    def create_ragas_dataset(self, test_cases: List[RAGASTestCase]) -> Dataset:
        """Tạo RAGAS dataset từ test cases"""
        data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        for case in test_cases:
            data['question'].append(case.question)
            data['answer'].append(case.answer)
            data['contexts'].append(case.contexts[:3])  # Limit contexts
            
            # Use provided ground truth hoặc tạo simple fallback
            if case.ground_truth:
                data['ground_truth'].append(case.ground_truth)
            else:
                # Simple ground truth generation
                keywords = self._extract_keywords(case.question)
                gt = f"Thông tin về {keywords[0] if keywords else 'câu hỏi này'} có trong tài liệu."
                data['ground_truth'].append(gt)
        
        return Dataset.from_dict(data)
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords từ question để tạo ground truth"""
        # Simple keyword extraction
        stop_words = {'là', 'có', 'gì', 'nào', 'bao', 'nhiêu', 'như', 'thế', 'của', 'và', 'với'}
        words = question.lower().split()
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        return keywords[:3]  # Top 3 keywords
    
    def evaluate_test_cases(self, test_cases: List[RAGASTestCase]) -> Dict[str, Any]:
        """
        Evaluate test cases với offline models
        """
        try:
            # Create dataset
            dataset = self.create_ragas_dataset(test_cases)
            logger.info(f"Created dataset with {len(test_cases)} test cases")
            
            # Run RAGAS evaluation
            logger.info("Running RAGAS evaluation with offline models...")
            
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Convert to pandas DataFrame
            results_df = result.to_pandas()
            
            # Compute aggregates
            aggregates = self._compute_aggregates(results_df)
            
            return {
                'status': 'success',
                'method': 'offline_ragas',
                'results_df': results_df,
                'aggregates': aggregates,
                'num_cases': len(test_cases),
                'embeddings_type': 'TEI' if self.use_tei else 'HuggingFace'
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            
            # Fallback to basic evaluation
            return self._fallback_evaluation(test_cases, str(e))
    
    def _compute_aggregates(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Compute aggregate scores"""
        aggregates = {}
        
        metric_columns = [col for col in results_df.columns 
                         if col not in ['question', 'answer', 'contexts', 'ground_truth']]
        
        for col in metric_columns:
            if col in results_df.columns:
                values = pd.to_numeric(results_df[col], errors='coerce').dropna()
                if len(values) > 0:
                    aggregates[f'mean_{col}'] = float(values.mean())
                    aggregates[f'std_{col}'] = float(values.std())
                    aggregates[f'min_{col}'] = float(values.min())
                    aggregates[f'max_{col}'] = float(values.max())
        
        return aggregates
    
    def _fallback_evaluation(self, test_cases: List[RAGASTestCase], error: str) -> Dict[str, Any]:
        """Fallback evaluation sử dụng basic metrics"""
        logger.info("Using fallback basic evaluation...")
        
        scores = {
            'basic_relevancy': [],
            'basic_context_match': [],
            'basic_completeness': []
        }
        
        for case in test_cases:
            # Basic keyword matching
            question_words = set(case.question.lower().split())
            answer_words = set(case.answer.lower().split())
            context_words = set(' '.join(case.contexts).lower().split())
            
            # Calculate scores
            relevancy = len(question_words & answer_words) / max(len(question_words), 1)
            context_match = len(answer_words & context_words) / max(len(answer_words), 1)
            completeness = min(len(case.answer.split()) / 15, 1.0)
            
            scores['basic_relevancy'].append(relevancy)
            scores['basic_context_match'].append(context_match)
            scores['basic_completeness'].append(completeness)
        
        # Compute aggregates
        aggregates = {
            'mean_basic_relevancy': np.mean(scores['basic_relevancy']),
            'mean_basic_context_match': np.mean(scores['basic_context_match']),
            'mean_basic_completeness': np.mean(scores['basic_completeness']),
        }
        
        aggregates['overall_basic_score'] = np.mean([
            aggregates['mean_basic_relevancy'],
            aggregates['mean_basic_context_match'],
            aggregates['mean_basic_completeness']
        ])
        
        return {
            'status': 'fallback',
            'method': 'basic_offline',
            'aggregates': aggregates,
            'num_cases': len(test_cases),
            'error': error
        }

class OfflineRAGASIntegration:
    """Integration class cho offline RAGAS evaluation"""
    
    def __init__(self, 
                 gemini_api_key: str,
                 tei_url: str = "http://localhost:8081",
                 use_docker_tei: bool = False):
        """
        Initialize integration
        
        Args:
            gemini_api_key: Google API key
            tei_url: TEI service URL
            use_docker_tei: Sử dụng TEI trong Docker (True) hoặc local (False)
        """
        self.gemini_api_key = gemini_api_key
        
        # Adjust TEI URL for Docker environment
        if use_docker_tei:
            self.tei_url = "http://tei:80"  # Docker service name
        else:
            self.tei_url = tei_url
        
        # Initialize evaluator
        self.evaluator = OfflineRAGASEvaluator(
            gemini_api_key=gemini_api_key,
            tei_url=self.tei_url,
            use_tei=True
        )
    
    def evaluate_from_ground_truth_queries(self,
                                         queries: List[Dict[str, Any]],
                                         search_func,
                                         max_queries: int = 10,
                                         top_k: int = 10) -> Dict[str, Any]:
        """
        Evaluate từ ground truth queries
        """
        if len(queries) > max_queries:
            logger.info(f"Limiting to {max_queries} queries for efficiency")
            queries = queries[:max_queries]
        
        test_cases = []
        
        logger.info(f"Preparing {len(queries)} test cases...")
        
        for i, query_data in enumerate(queries):
            question = query_data['query']
            expected_keywords = query_data.get('expected_keywords', [])
            
            try:
                # Get search results
                search_results = search_func(question, top_k)
                
                # Extract contexts
                contexts = []
                for result in search_results[:10]:  # Top 3 results
                    text = result.get('text', '').strip()
                    if text and len(text) > 20:
                        # Truncate để giảm token usage
                        contexts.append(text[:300])
                
                if not contexts:
                    contexts = ["Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."]
                
                # Generate answer từ contexts
                answer = self._generate_answer_from_contexts(question, contexts, expected_keywords)
                
                # Create ground truth
                ground_truth = self._create_ground_truth(question, expected_keywords)
                
                test_case = RAGASTestCase(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth
                )
                
                test_cases.append(test_case)
                
            except Exception as e:
                logger.error(f"Error processing query {i+1}: {e}")
                continue
            

        
        # Run evaluation
        return self.evaluator.evaluate_test_cases(test_cases)
    
    def _generate_answer_from_contexts(self, 
                                     question: str, 
                                     contexts: List[str], 
                                     keywords: List[str]) -> str:
        """Generate answer từ retrieved contexts"""
        if not contexts or contexts == ["Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."]:
            return "Không tìm thấy thông tin để trả lời câu hỏi này."
        
        # Combine contexts
        combined_context = " ".join(contexts)
        
        # Check for keyword presence
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in combined_context.lower():
                found_keywords.append(keyword)
        
        # Generate answer based on found keywords
        if found_keywords:
            if any(x in question.lower() for x in ['phí', 'giá', 'cost', 'fee']):
                return f"Dựa trên thông tin tìm được, {found_keywords[0]} có các thông tin về phí như sau: {combined_context[:200]}..."
            elif any(x in question.lower() for x in ['mã', 'code', 'loại']):
                return f"Theo tài liệu, {found_keywords[0]} có các mã/loại sau: {combined_context[:200]}..."
            else:
                return f"Thông tin về {found_keywords[0]}: {combined_context[:200]}..."
        else:
            return f"Dựa trên tài liệu, thông tin liên quan: {combined_context[:150]}..."
    
    def _create_ground_truth(self, question: str, keywords: List[str]) -> str:
        """Create ground truth answer"""
        if not keywords:
            return "Thông tin liên quan đến câu hỏi có trong tài liệu."
        
        main_keyword = keywords[0]
        
        if any(x in question.lower() for x in ['phí', 'giá', 'cost']):
            return f"Thông tin về phí của {main_keyword} được quy định trong tài liệu."
        elif any(x in question.lower() for x in ['mã', 'code']):
            return f"Các mã liên quan đến {main_keyword} được liệt kê trong hệ thống."
        else:
            return f"Thông tin về {main_keyword} có sẵn trong cơ sở dữ liệu."

def setup_offline_ragas_evaluation(gemini_api_key: str, 
                                 tei_url: str = "http://localhost:8081",
                                 use_docker: bool = False) -> OfflineRAGASIntegration:
    """
    Setup offline RAGAS evaluation
    
    Args:
        gemini_api_key: Google API key
        tei_url: TEI service URL  
        use_docker: True nếu chạy trong Docker environment
    """
    return OfflineRAGASIntegration(
        gemini_api_key=gemini_api_key,
        tei_url=tei_url,
        use_docker_tei=use_docker
    )

def test_offline_setup():
    """Test offline RAGAS setup"""
    print("🧪 Testing Offline RAGAS Setup")
    print("=" * 50)
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No Google/Gemini API key found")
        return False
    
    try:
        # Test TEI connection
        tei_url = os.getenv("TEI_URL", "http://localhost:8081")
        print(f"Testing TEI connection at {tei_url}...")
        
        response = requests.get(f"{tei_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ TEI service is running")
        else:
            print(f"⚠️ TEI responded with status {response.status_code}")
        
        # Initialize integration
        integration = setup_offline_ragas_evaluation(api_key, tei_url)
        print("✅ Offline RAGAS integration initialized")
        
        # Test with sample data
        test_queries = GROUND_TRUTH_QUERIES  # Just 2 queries for testing
        
        result = integration.evaluate_from_ground_truth_queries(
            queries=test_queries,
            search_func=hybrid_search,
            top_k=10
        )
        
        print(f"✅ Test evaluation completed")
        print(f"Status: {result.get('status')}")
        print(f"Method: {result.get('method')}")
        
        if result.get('aggregates'):
            for key, value in list(result['aggregates'].items())[:3]:
                print(f"  {key}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_offline_setup()