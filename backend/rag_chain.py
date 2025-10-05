import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from langchain.schema import Document, BaseRetriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import VLLMOpenAI


# Import các hàm search có sẵn
from search import (
    hybrid_search,
    search_dense_only, 
    search_bm25_only,
    get_dense_embedding
)


class PromptLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        # với LLM (text)
        for i, p in enumerate(prompts):
            logger.info(f"PROMPT[{i}]:\n---\n{p}\n---")

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # với ChatModel (list of messages)
        for i, msg_list in enumerate(messages):
            pretty = "\n".join([f"{m.type.upper()}: {getattr(m, 'content', m)}" for m in msg_list])
            logger.info(f"CHAT_PROMPT[{i}]:\n---\n{pretty}\n---")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchMethod(Enum):
    HYBRID = "hybrid"
    DENSE = "dense"
    BM25 = "bm25"

@dataclass
class RAGConfig:
    """LangChain RAG Configuration"""
    vllm_url: str = "http://vllm:8000/v1"
    model_name: str = "gemma-3-1b-it"  
    temperature: float = 0.7
    max_tokens: int = 512
    default_top_k: int = 10
    

class CustomSearchRetriever(BaseRetriever):
    """Wrapper để dùng search.py functions với LangChain"""
    # Khai báo FIELDS (được phép set qua __init__)
    search_method: SearchMethod = SearchMethod.HYBRID
    top_k: int = 10
    
    def __init__(self, **data):
        super().__init__(**data)
        # map hàm search runtime
        self._search_functions = {
            SearchMethod.HYBRID: hybrid_search,
            SearchMethod.DENSE: search_dense_only,
            SearchMethod.BM25: search_bm25_only,
        }
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Convert search results to LangChain Documents"""
        
        # Gọi hàm search tương ứng từ search.py
        search_func = self._search_functions[self.search_method]
        results = search_func(query, self.top_k)
        
        # Convert to LangChain Document format
        documents = []
        for result in results:
            doc = Document(
                page_content=result['text'],
                metadata={
                    'doc_id': result['doc_id'],
                    'score': result['score'],
                    'search_type': result['search_type']
                }
            )
            documents.append(doc)
        
        return documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version"""
        return self._get_relevant_documents(query)

class LangChainRAG:
    """LangChain RAG using existing search.py"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LangChain components"""
        
        # Initialize LLM
        self.llm = VLLMOpenAI(
            openai_api_base=self.config.vllm_url,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key="dummy-key"
        )
        
        # Initialize retrievers using search.py functions
        self.retrievers = {
            SearchMethod.HYBRID: CustomSearchRetriever(
                search_method=SearchMethod.HYBRID,
                top_k=self.config.default_top_k
            ),
            SearchMethod.DENSE: CustomSearchRetriever(
                search_method=SearchMethod.DENSE,
                top_k=self.config.default_top_k
            ),
            SearchMethod.BM25: CustomSearchRetriever(
                search_method=SearchMethod.BM25,
                top_k=self.config.default_top_k
            )
        }
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create prompt templates
        self._create_prompts()
    
    def _create_prompts(self):
        """Create prompt templates"""
        
        
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Bạn là trợ lý AI thông minh. Trả lời câu hỏi dựa trên thông tin được cung cấp.



THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""
        )
        
        # Conversational prompt with history
        self.conv_prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template="""Bạn là trợ lý AI thông minh trong cuộc hội thoại.

LỊCH SỬ HỘI THOẠI:
{chat_history}

THÔNG TIN THAM KHẢO MỚI:
{context}

CÂU HỎI HIỆN TẠI: {question}



**TRẢ LỜI:"""
        )
    
    def create_qa_chain(self, search_method: SearchMethod = SearchMethod.HYBRID):
        """Create basic QA chain"""
        retriever = self.retrievers[search_method]
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.qa_prompt
            }
        )
        
        return qa_chain
    
    def create_conversational_chain(
        self, 
        search_method: SearchMethod = SearchMethod.HYBRID,
        session_id: Optional[str] = None
    ):
        """Create conversational chain with memory"""
        retriever = self.retrievers[search_method]
        
        # Clear memory cho session mới
        self.memory.clear()
        
        # Load chat history if session_id provided
        if session_id:
            from db_utils import get_chat_history
            history = get_chat_history(session_id, limit=10)
            
            # Add to memory
            for msg in history:
                if msg["role"] == "user":
                    self.memory.chat_memory.add_user_message(msg["content"])
                else:
                    self.memory.chat_memory.add_ai_message(msg["content"])
        
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": self.conv_prompt
            }
        )
        
        return conv_chain
    
    
    
    def process(
        self,
        query: str,
        session_id: Optional[str] = None,
        search_method: SearchMethod = SearchMethod.HYBRID,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """Main processing function"""
        
        try:
            logger.info(f"Processing query: {query} with method: {search_method.value}")

            # Tạo callback để log prompt
            cb = PromptLogger()

            # Tạo chain
            if session_id and include_history:
                chain = self.create_conversational_chain(search_method, session_id)
                payload = {"question": query}
            else:
                chain = self.create_qa_chain(search_method)
                payload = {"query": query}

            # Gọi chain với callback 
            try:
                result = chain.invoke(payload, config={"callbacks": [cb]})
            except Exception:
                result = chain(payload, callbacks=[cb])

            # Lấy answer
            answer = result.get("result", result.get("answer", ""))
            
            # Format response
            response = {
                "query": query,
                "answer": answer,
                "success": True,
                "metadata": {
                    "search_method": search_method.value,
                    "num_documents": len(result.get("source_documents", [])),
                    "model": self.config.model_name,
                    "session_id": session_id
                }
            }
            
            # Add sources if available
            if "source_documents" in result:
                response["sources"] = [
                    {
                        "content": doc.page_content[:200],  # Preview
                        "doc_id": doc.metadata.get("doc_id"),
                        "score": doc.metadata.get("score"),
                        "search_type": doc.metadata.get("search_type")
                    }
                    for doc in result["source_documents"]
                ]
            
            logger.info(f"Successfully processed query. Answer length: {len(answer)}")
            return response
            
        except Exception as e:
            logger.error(f"Error in LangChain RAG: {e}", exc_info=True)
            return {
                "query": query,
                "answer": f"Lỗi xử lý: {str(e)}",
                "success": False,
                "metadata": {"error": str(e)}
            }

# Singleton instance
_rag_instance = None

def get_langchain_rag() -> LangChainRAG:
    """Get singleton RAG instance"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LangChainRAG()
    return _rag_instance