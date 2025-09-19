from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    GEMMA_3_1B_IT = "gemma-3-1b-it"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GEMMA_3_1B_IT)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class SessionInfo(BaseModel):
    session_id: str
    last_message_time: datetime
    first_query: str

class RenameSessionRequest(BaseModel):
    new_title: str