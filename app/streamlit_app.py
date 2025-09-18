import streamlit as st
import uuid
from sidebar import display_sidebar
from chat_interface import display_chat_interface
from api_utils import get_chat_history

st.set_page_config(page_title="Tele-Oracle RAG", layout="wide")
st.title("Tele-Oracle: Trợ lý Hỏi-Đáp")

# 1. Khởi tạo session_id duy nhất và không đổi cho mỗi phiên làm việc
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history_loaded = False # Thêm một cờ để chỉ tải lịch sử một lần

# 2. Tải lịch sử chat từ DB (chỉ chạy một lần duy nhất khi có session_id)
if not st.session_state.history_loaded:
    print(f"Lần đầu tải lịch sử cho session: {st.session_state.session_id}")
    # Gọi API để lấy lịch sử chat cũ từ database
    st.session_state.messages = get_chat_history(st.session_state.session_id)
    st.session_state.history_loaded = True # Đánh dấu là đã tải xong
    if st.session_state.messages:
        print(f"Đã tải {len(st.session_state.messages)} tin nhắn cũ.")
    else:
        st.session_state.messages = []

# Display the sidebar
display_sidebar()

# Display the chat interface
display_chat_interface()