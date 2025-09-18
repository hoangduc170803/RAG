import requests
import streamlit as st



BACKEND_URL = "http://backend:8080"
def get_api_response(question, session_id, model):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "question": question,
        "model": model
    }
    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post(f"{BACKEND_URL}/chat", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def get_chat_history(session_id: str):
    """Lấy lịch sử chat từ backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/chat-history/{session_id}")
        response.raise_for_status()
        return response.json().get("messages", [])
    except requests.RequestException as e:
        st.error(f"Không thể tải lịch sử chat: {e}")
        return []

@st.cache_data(ttl=60)    
def list_sessions():
    """Lấy danh sách các session từ backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/sessions")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Không thể tải danh sách session: {e}")
        return []

def delete_session(session_id: str):
    """Gửi yêu cầu xóa một session đến backend."""
    try:
        response = requests.delete(f"{BACKEND_URL}/session/{session_id}")
        response.raise_for_status()
        st.cache_data.clear() # Xóa cache của list_sessions để cập nhật
        return True
    except requests.RequestException as e:
        st.error(f"Lỗi khi xóa session: {e}")
        return False

def rename_session(session_id: str, new_title: str):
    """Gửi yêu cầu đổi tên một session đến backend."""
    try:
        response = requests.put(
            f"{BACKEND_URL}/session/{session_id}",
            json={"new_title": new_title}
        )
        response.raise_for_status()
        st.cache_data.clear() # Xóa cache của list_sessions để cập nhật
        return True
    except requests.RequestException as e:
        st.error(f"Lỗi khi đổi tên session: {e}")
        return False    