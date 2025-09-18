import sqlite3
from datetime import datetime
import os
from typing import List, Dict

DATA_PATH = "/data" 
DB_NAME = os.path.join(DATA_PATH, "rag_app.db")

def get_db_connection():
    os.makedirs(DATA_PATH, exist_ok=True)
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    try:
        conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         session_id TEXT,
                         user_query TEXT,
                         gpt_response TEXT,
                         model TEXT,
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        conn.execute('''CREATE INDEX IF NOT EXISTS idx_app_logs_session_created
                        ON application_logs(session_id, created_at)''')
        
        conn.commit()
    finally:
        conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                     (session_id, user_query, gpt_response, model))
        conn.commit()
    finally:
        conn.close()

def get_chat_history(session_id: str, limit: int = 10):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Fixed the SQL query to include LIMIT and proper ORDER BY
        cursor.execute('''SELECT user_query, gpt_response 
                         FROM application_logs 
                         WHERE session_id = ? 
                         ORDER BY created_at DESC 
                         LIMIT ?''', (session_id, limit))
        rows = cursor.fetchall()
        
        # Reverse to get chronological order (oldest first)
        messages = []
        for row in reversed(rows):
            messages.append({"role": "user", "content": row["user_query"]})
            messages.append({"role": "assistant", "content": row["gpt_response"]})
        
        return messages
    finally:
        conn.close()

def get_all_sessions() -> List[Dict]:
    """Lấy danh sách các session_id duy nhất và thông tin mới nhất."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
        SELECT
            session_id,
            MAX(created_at) as last_message_time,
            (SELECT user_query FROM application_logs AS l2
             WHERE l2.session_id = l1.session_id
             ORDER BY created_at ASC LIMIT 1) as first_query
        FROM
            application_logs AS l1
        GROUP BY
            session_id
        ORDER BY
            last_message_time DESC
        """
        cursor.execute(query)
        sessions = cursor.fetchall()
        return [dict(session) for session in sessions]

def delete_session(session_id: str) -> bool:
    """Xóa tất cả các bản ghi của một session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM application_logs WHERE session_id = ?", (session_id,))
        return cursor.rowcount > 0 # Trả về True nếu có hàng bị xóa

def rename_session(session_id: str, new_title: str) -> bool:
    """Đổi tên một session bằng cách cập nhật câu hỏi đầu tiên."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Tìm id của tin nhắn đầu tiên trong session
        cursor.execute("SELECT id FROM application_logs WHERE session_id = ? ORDER BY created_at ASC LIMIT 1", (session_id,))
        first_message = cursor.fetchone()
        
        if first_message:
            first_message_id = first_message['id']
            # Cập nhật user_query của tin nhắn đó
            cursor.execute("UPDATE application_logs SET user_query = ? WHERE id = ?", (new_title, first_message_id))
            return cursor.rowcount > 0
    return False    

# Initialize the database tables
create_application_logs()