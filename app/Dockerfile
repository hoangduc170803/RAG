FROM python:3.10-slim

WORKDIR /app

# Sửa lại tên file ở đây cho đúng
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code từ build context
COPY . .

# Mở cổng cho Streamlit
EXPOSE 8501

# Lệnh mặc định để chạy ứng dụng Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]