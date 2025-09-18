# FILE: backend/main.py

from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, List
import os
import uuid
import logging
from contextlib import asynccontextmanager

# Import các thành phần của hệ thống
from pydantic_models import QueryInput, QueryResponse, SessionInfo, RenameSessionRequest
from rag_chain import RAGChain
from db_utils import (
    insert_application_logs,
    get_chat_history,
    get_all_sessions,
    delete_session,
    rename_session, 
)
from pymilvus import Collection, connections




from fastapi.middleware.cors import CORSMiddleware



# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global resources
app_state: Dict = {}

class AppSettings:
    """Centralized settings management"""
    MILVUS_URI = os.getenv("MILVUS_URI", "milvus")
    COLLECTION = os.getenv("COLLECTION", "my_rag_collection")
    VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000/v1")
    MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "gemma-3-12b-it")

settings = AppSettings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý lifecycle của app"""
    logger.info("Backend server đang khởi động...")
    
    # Initialize Milvus connection
    try:
        connections.connect(alias="default", uri=settings.MILVUS_URI)
        collection = Collection(settings.COLLECTION)
        collection.load()
        app_state["milvus_collection"] = collection
        logger.info("Đã kết nối Milvus thành công")
    except Exception as e:
        logger.error(f"Lỗi kết nối Milvus: {e}")
        app_state["milvus_collection"] = None

    # Initialize RAG Chain
    app_state["rag_chain"] = RAGChain(
        vllm_url=settings.VLLM_URL,
        model_name=settings.MODEL_NAME
    )
    logger.info("RAG Chain đã sẵn sàng")
    yield
    # Cleanup
    logger.info("Backend server đang tắt...")
    if "milvus_collection" in app_state:
        connections.disconnect(alias="default")
        logger.info("Đã ngắt kết nối Milvus")

app = FastAPI(title="Milvus RAG Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://app:8501"],  # hoặc "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat-history/{session_id}")
async def get_history(session_id: str):
    """
    Lấy lịch sử trò chuyện cho một session_id cụ thể.
    """
    logger.info(f"Đang lấy lịch sử chat cho session: {session_id}")
    messages = get_chat_history(session_id)
    return {"messages": messages}





# Dependency injection
def get_rag_chain() -> RAGChain:
    """Dependency để lấy RAG chain"""
    rag_chain = app_state.get("rag_chain")
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG Chain chưa sẵn sàng")
    return rag_chain

def get_milvus_collection() -> Collection:
    """Dependency để lấy Milvus collection"""
    collection = app_state.get("milvus_collection")
    if not collection:
        raise HTTPException(status_code=503, detail="Mất kết nối đến Milvus")
    return collection

# --- API Endpoints ---

@app.post("/chat", response_model=QueryResponse)
async def chat(query_input: QueryInput, rag_chain: RAGChain = Depends(get_rag_chain)):
    """Chat endpoint với RAG"""
    session_id = query_input.session_id or str(uuid.uuid4())
    logger.info(f"Session: {session_id}, Query: {query_input.question}")

    try:
        # Process query
        result = rag_chain.process(query=query_input.question)
        answer = (result.answer or "Lỗi: Không có câu trả lời được sinh ra.")
        
        # Log interaction
        insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
        
        return QueryResponse(
            answer=answer, 
            session_id=session_id, 
            model=query_input.model
        )
    except Exception as e:
        logger.error(f"Lỗi xử lý chat: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@app.get("/chat-history")
async def chat_history(session_id: str, limit: int = 50):
    return {
        "session_id": session_id,
        "messages": get_chat_history(session_id, limit)
    }    

@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """Liệt kê tất cả các session chat đã có."""
    return get_all_sessions()

@app.delete("/session/{session_id}", status_code=200)
async def delete_chat_session(session_id: str):
    """Xóa toàn bộ lịch sử của một session chat."""
    logger.info(f"Yêu cầu xóa session: {session_id}")
    success = delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session không tìm thấy.")
    return {"message": "Session đã được xóa thành công."}

@app.put("/session/{session_id}", status_code=200)
async def rename_chat_session(session_id: str, request: RenameSessionRequest):
    """Đổi tên một session chat."""
    logger.info(f"Yêu cầu đổi tên session: {session_id} thành '{request.new_title}'")
    success = rename_session(session_id, request.new_title)
    if not success:
        raise HTTPException(status_code=404, detail="Session không tìm thấy hoặc không thể đổi tên.")
    return {"message": "Session đã được đổi tên thành công."}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "rag_chain": "ready" if app_state.get("rag_chain") else "not_ready",
            "milvus": "connected" if app_state.get("milvus_collection") else "disconnected"
        }
    }