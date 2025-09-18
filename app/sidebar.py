import streamlit as st
import uuid
from api_utils import list_sessions, delete_session, rename_session

def display_sidebar():
    """Hiển thị sidebar với các tùy chọn và lịch sử chat có thể quản lý."""
    st.sidebar.header("Cài đặt")
    model_options = ["gemma-3-12b-it"]
    st.sidebar.selectbox("Chọn Model", options=model_options, key="model")

    st.sidebar.divider()
    st.sidebar.header("Lịch sử trò chuyện")

    if st.sidebar.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.history_loaded = True
        st.rerun()

    sessions = list_sessions()
    if sessions:
        # Tạo một dictionary để quản lý trạng thái đổi tên
        if 'renaming_session' not in st.session_state:
            st.session_state.renaming_session = None

        for session in sessions:
            session_id = session.get('session_id')
            col1, col2, col3 = st.sidebar.columns([0.7, 0.15, 0.15])

            with col1:
                session_title = session.get('first_query', 'Cuộc trò chuyện')
                if len(session_title) > 25:
                    session_title = session_title[:25] + "..."
                
                # Hiển thị ô nhập liệu nếu đang ở trạng thái đổi tên
                if st.session_state.renaming_session == session_id:
                    new_title = st.text_input(
                        "Tên mới:", 
                        value=session.get('first_query'), 
                        key=f"rename_{session_id}"
                    )
                    if st.button("Lưu", key=f"save_{session_id}"):
                        if new_title:
                            rename_session(session_id, new_title)
                            st.session_state.renaming_session = None
                            st.rerun()
                else:
                    if st.button(session_title, key=session_id, use_container_width=True):
                        if st.session_state.get('session_id') != session_id:
                            st.session_state.session_id = session_id
                            st.session_state.history_loaded = False
                            st.rerun()

            with col2:
                # Nút đổi tên
                if st.button("✏️", key=f"edit_{session_id}"):
                    st.session_state.renaming_session = session_id
                    st.rerun()

            with col3:
                # Nút xóa
                if st.button("🗑️", key=f"delete_{session_id}"):
                    if delete_session(session_id):
                        # Nếu session bị xóa là session hiện tại, tạo session mới
                        if st.session_state.session_id == session_id:
                            st.session_state.session_id = str(uuid.uuid4())
                            st.session_state.messages = []
                        st.session_state.renaming_session = None
                        st.rerun()