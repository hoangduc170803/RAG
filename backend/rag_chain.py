import requests
from typing import List, Dict, Optional, Generator, Union
from search import hybrid_search, search_dense_only, search_bm25_only
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchMethod(Enum):
    HYBRID = "hybrid"
    DENSE = "dense"  
    BM25 = "bm25"

@dataclass
class RAGConfig:
    """Configuration for RAG Chain"""
    vllm_url: str = "http://vllm:8000/v1"
    model_name: str = "gemma-3-12b-it"
    temperature: float = 0.1
    max_tokens: int = 512
    default_top_k: int = 10
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        import os
        return cls(
            vllm_url=os.getenv("VLLM_URL", cls.vllm_url),
            model_name=os.getenv("VLLM_MODEL_NAME", cls.model_name)
        )

@dataclass 
class RAGResponse:
    """Structured RAG response"""
    query: str
    answer: str
    success: bool
    metadata: Dict
    sources: Optional[List[Dict]] = None

class RAGChain:
    """Improved RAG Chain with better structure"""
    
    def __init__(self, config: Optional[RAGConfig] = None, **kwargs):
        if config:
            self.config = config
        else:
            # Support backward compatibility
            self.config = RAGConfig(**kwargs)
    
    def retrieve_context(
        self, 
        query: str, 
        method: Union[str, SearchMethod] = SearchMethod.HYBRID, 
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Retrieve relevant documents"""
        if isinstance(method, str):
            method = SearchMethod(method)
            
        top_k = top_k or self.config.default_top_k
        
        logger.info(f"Retrieving {top_k} documents using {method.value} search")
        
        # Strategy pattern for search methods
        search_strategies = {
            SearchMethod.HYBRID: hybrid_search,
            SearchMethod.DENSE: search_dense_only,
            SearchMethod.BM25: search_bm25_only
        }
        
        search_func = search_strategies.get(method)
        if not search_func:
            raise ValueError(f"Unknown search method: {method}")
            
        results = search_func(query, top_k)
        logger.info(f"Retrieved {len(results)} documents")
        return results
    

    def _format_context(self, documents: List[Dict]) -> str:
        """
        Formats documents into a clean, structured, and easy-to-parse string for the LLM.
        Each document is presented as a distinct record with clear labels.
        """
        if not documents:
            return "Không có thông tin liên quan được tìm thấy."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Trích xuất source từ bên trong trường 'text' nếu có
            text_content = doc.get('text', '')
            source_info = "Không rõ nguồn" # Giá trị mặc định

            # Giả định rằng source nằm trong một dòng riêng biệt dạng {'source': '...'}
            lines = text_content.split('\n')
            remaining_lines = []
            for line in lines:
                if line.strip().startswith("{'source':"):
                    try:
                        # Dùng eval để parse chuỗi dict một cách an toàn (cẩn thận nếu input không đáng tin)
                        source_dict = eval(line)
                        source_info = source_dict.get('source', source_info)
                    except:
                        remaining_lines.append(line) # Nếu parse lỗi, giữ lại dòng đó
                else:
                    remaining_lines.append(line)
            
            clean_text = "\n".join(remaining_lines).strip()

            # Tạo một khối văn bản có cấu trúc cho mỗi tài liệu
            context_parts.append(
                f"--- Tài liệu {i} ---\nNGUỒN: {source_info}\nNỘI DUNG: {clean_text}"
            )

        return "\n\n".join(context_parts)
    


    def _create_prompt(self, query: str, context: str) -> str:
        """A prompt template designed to extract both answer and source."""
        return f"""Bạn là một trợ lý AI chuyên trích xuất thông tin. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin tham khảo và trích dẫn nguồn một cách chính xác.

    **QUY TẮC BẮT BUỘC:**
    1.  Đọc kỹ CÂU HỎI và tìm câu trả lời trong THÔNG TIN THAM KHẢO.
    2.  Nếu có nhiều nguồn hãy liệt kê tất cả nguồn đó ra.
    3.  Lấy tên nguồn từ dòng `NGUỒN:` của tài liệu đó.
    4.  Trình bày kết quả theo đúng ĐỊNH DẠNG ĐẦU RA.
    5.  Nếu không tìm thấy câu trả lời trong NỘI DUNG của bất kỳ tài liệu nào, hãy trả lời theo định dạng sau:
        [TRẢ LỜI]: Tôi không tìm thấy thông tin về vấn đề này trong tài liệu.
        [NGUỒN]: Không có

    **ĐỊNH DẠNG ĐẦU RA:**
    [TRẢ LỜI]: <Nội dung câu trả lời được trích xuất từ THÔNG TIN THAM KHẢO>
    [NGUỒN]: <Tên file nguồn được trích xuất từ dòng NGUỒN>



    **BẮT ĐẦU THỰC HIỆN:**

    **THÔNG TIN THAM KHẢO:**
    {context}

    **CÂU HỎI:** {query}

    **TRẢ LỜI:**"""
    
    def _call_llm(self, prompt: str, stream: bool = False) -> Dict:
        """Call VLLM service"""
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(
                f"{self.config.vllm_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return {"success": True, "stream": response}
            
            result = response.json()
            return {
                "success": True,
                "response": result['choices'][0]['message']['content'],
                "usage": result.get('usage', {})
            }
            
        except requests.RequestException as e:
            logger.error(f"Error calling VLLM service: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "Xin lỗi, tôi không thể tạo câu trả lời lúc này."
            }
    
    def process(
        self, 
        query: str,
        search_method: Union[str, SearchMethod] = SearchMethod.HYBRID,
        top_k: Optional[int] = None,
        include_sources: bool = True
    ) -> RAGResponse:
        """Main RAG pipeline"""
        logger.info(f"Processing RAG query: {query}")
        
        try:
            # Retrieve documents
            documents = self.retrieve_context(query, search_method, top_k)
            
            # Generate response
            context = self._format_context(documents)
            logger.info(f"CONTEXT PASSED TO PROMPT:\n---\n{context}\n---")
            prompt = self._create_prompt(query, context)
            logger.info(f"PROMPT:\n---\n{prompt}\n---")
            generation_result = self._call_llm(prompt)
            
            # Build response
            response = RAGResponse(
                query=query,
                answer=generation_result.get("response", ""),
                success=generation_result.get("success", False),
                metadata={
                    "search_method": search_method.value if isinstance(search_method, SearchMethod) else search_method,
                    "num_documents": len(documents),
                    "model": self.config.model_name,
                    "usage": generation_result.get("usage", {})
                }
            )
            
            if include_sources:
                response.sources = [
                    {
                        "doc_id": doc.get("doc_id"),
                        "text": doc.get("text"),
                        "score": doc.get("score"),
                        "search_type": doc.get("search_type")
                    }
                    for doc in documents
                ]
            
            logger.info(f"RAG processing completed. Success: {response.success}")
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG process: {e}")
            return RAGResponse(
                query=query,
                answer=f"Lỗi xử lý: {str(e)}",
                success=False,
                metadata={"error": str(e)}
            )
    
    def process_streaming(
        self,
        query: str,
        search_method: Union[str, SearchMethod] = SearchMethod.HYBRID,
        top_k: Optional[int] = None
    ) -> Generator[Dict, None, None]:
        """Stream RAG response"""
        try:
            # Retrieve context
            documents = self.retrieve_context(query, search_method, top_k)
            context = self._format_context(documents)
            prompt = self._create_prompt(query, context)
            
            # Yield metadata first
            yield {
                "type": "metadata",
                "data": {
                    "query": query,
                    "num_documents": len(documents),
                    "search_method": search_method.value if isinstance(search_method, SearchMethod) else search_method
                }
            }
            
            # Yield sources
            yield {"type": "sources", "data": documents}
            
            # Stream response
            stream_result = self._call_llm(prompt, stream=True)
            
            if stream_result["success"]:
                for line in stream_result["stream"].iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: "):
                            data = line[6:]
                            if data != "[DONE]":
                                try:
                                    chunk = json.loads(data)
                                    if 'choices' in chunk and chunk['choices']:
                                        delta = chunk['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            yield {
                                                "type": "content",
                                                "data": delta['content']
                                            }
                                except json.JSONDecodeError:
                                    continue
            
            yield {"type": "done", "data": {}}
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {"type": "error", "data": str(e)}

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self.config)


# Factory function
def create_rag_chain(**kwargs) -> RAGChain:
    """Factory function to create RAG chain"""
    config = RAGConfig.from_env()
    
    # Override with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return RAGChain(config)


# Utility functions
def simple_rag_query(query: str, **kwargs) -> str:
    """Simple wrapper for quick RAG queries"""
    rag = create_rag_chain(**kwargs)
    result = rag.process(query)
    return result.answer


def compare_search_methods(query: str, top_k: int = 3) -> Dict[str, RAGResponse]:
    """Compare all search methods"""
    rag = create_rag_chain()
    results = {}
    
    for method in SearchMethod:
        results[method.value] = rag.process(
            query, 
            search_method=method, 
            top_k=top_k,
            include_sources=True
        )
    
    return results


