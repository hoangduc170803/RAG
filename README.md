# RAG System - Retrieval-Augmented Generation

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Milvus](https://img.shields.io/badge/Vector_DB-Milvus-00A1EA?logo=milvus&logoColor=white)](https://milvus.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

Hệ thống RAG (Retrieval-Augmented Generation) được xây dựng với Milvus vector database, hỗ trợ tìm kiếm ngữ nghĩa và trả lời câu hỏi thông minh dựa trên dữ liệu đã được index.

## 📋 Mục Lục

- [Tính Năng](#-tính-năng)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
- [Cài Đặt](#-cài-đặt)
- [Cấu Hình](#-cấu-hình)
- [Sử Dụng](#-sử-dụng)
- [API Documentation](#-api-documentation)
- [Đánh Giá Mô Hình](#-đánh-giá-mô-hình)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Tính Năng

- 🔍 **Tìm kiếm ngữ nghĩa**: Sử dụng Milvus vector database để tìm kiếm thông tin chính xác
- 🤖 **RAG Pipeline**: Kết hợp retrieval và generation để trả lời câu hỏi
- 💬 **Chat Interface**: API hỗ trợ conversation với session management
- 📊 **Vector Storage**: Lưu trữ và truy xuất embeddings hiệu quả
- 🐳 **Docker Support**: Triển khai dễ dàng với Docker Compose
- 🎯 **Model**: Đánh giá performance của các models (Gemma 3-4B-IT, Gemma 3-1B-IT)
- 🔄 **Data Persistence**: Dữ liệu được lưu trữ bền vững với volume mounting

## 🏗️ Kiến Trúc Hệ Thống



## 💻 Yêu Cầu Hệ Thống

### Phần Cứng
- **CPU**: 4+ cores (khuyến nghị)
- **RAM**: 8GB+ (16GB khuyến nghị cho production)
- **GPU**: NVIDIA GPU với CUDA support (tùy chọn, tăng tốc embedding và inference, nếu sử dụng mô hình 12b khuyến nghị VRAM từ 32GB trở lên)
- **Storage**: 20GB+ dung lượng trống

### Phần Mềm
- **OS**: Ubuntu 20.04/22.04 hoặc tương đương
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Driver**: 580+ (nếu sử dụng GPU)
- **NVIDIA Container Toolkit** (nếu sử dụng GPU)

# Nếu chạy trên máy cá nhân hãy bắt đầu từ bước 4 #
## 🚀 Cài Đặt Khi Deploy

### 1. Cài Đặt Dependencies

#### Cài Đặt NVIDIA Driver (Nếu sử dụng GPU)
```bash
apt-get update
apt install -y nvidia-driver-580 nvidia-utils-580
```

#### Cài Đặt Docker
```bash
# Cài đặt các gói cần thiết
apt-get install -y ca-certificates curl gnupg lsb-release

# Thêm Docker GPG key
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Thêm Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Cài đặt Docker
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Khởi động Docker
systemctl start docker
systemctl enable docker

# Thêm user vào docker group
usermod -aG docker $USER
```

#### Cài Đặt Docker Compose (Alternative)
```bash
apt update
apt install -y docker-compose
docker-compose --version
```

#### Cài Đặt NVIDIA Container Toolkit (Nếu sử dụng GPU)
```bash
# Thêm NVIDIA GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Thêm repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Cài đặt toolkit
apt update
apt install -y nvidia-container-toolkit

# Cấu hình Docker runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Kiểm tra
docker run --rm --gpus all nvidia/cuda:12.0-runtime-ubuntu20.04 nvidia-smi
```

### 2. Clone Repository
```bash
git clone https://github.com/hoangduc170803/RAG.git
cd RAG
```

### 3. Cấu Hình Firewall
```bash
# Mở các ports cần thiết
ufw allow ssh
ufw allow 8080      # API Gateway
ufw allow 8081      # Additional service
ufw allow 8000      # Application
ufw allow 9001      # MinIO Console
ufw allow 19530     # Milvus
ufw allow 9091      # Metrics
ufw allow 8001      # Service
ufw allow 8501      # Streamlit 

# Kích hoạt firewall
ufw enable
ufw status

# Kiểm tra IP công khai
curl ifconfig.me
```

### 4. Build và Khởi Động Services (nếu chạy local)

#### Build Ingest Service
```bash
docker compose --profile manual build ingest
```

#### Khởi động các services cơ bản
```bash
# Khởi động Etcd, MinIO, Milvus, và Attu
docker compose up -d etcd minio milvus attu
```

#### Chạy Data Ingestion
```bash
# Ingest dữ liệu vào Milvus
docker compose --profile manual run --rm ingest \
  --collection my_rag_collection \
  --input-json ./data/final_output.json \
  --drop-existing
```

#### Khởi động toàn bộ hệ thống
```bash
# Start tất cả services
docker compose up -d

# Kiểm tra trạng thái
docker compose ps

# Xem logs
docker compose logs -f
```

## ⚙️ Cấu Hình

### Environment Variables

Tạo file `.env` trong thư mục gốc:

```env
# Milvus Configuration
MILVUS_URI="http://milvus:19530"
INPUT_JSON=/work/data/final_output.json
DIM=1024 # bge-m3 là 1024

# Model Configuration
TEI_MODEL=/models
VLLM_MODEL=models/
TEI_MODEL_NAME=bge-m3


# Port Mappings
APP_PORT=8501
TEI_PORT=8081
VLLM_PORT=8000

# Collection Configuration
COLLECTION_NAME=my_rag_collection
```

### Data Configuration

Đặt dữ liệu JSON của bạn vào `./data/final_output.json` với format:

```json
[
  {
    "id": "doc_1",
    "content": "Nội dung tài liệu...",
    "metadata": {
      "source": "document.pdf",
      "page": 1
    }
  }
]
```

## 📖 Sử Dụng

### 1. Search Query (CLI)

```bash
# Tìm kiếm thông tin
docker compose run --rm app python search/search.py "Phí tham gia gói M2M7S1 là bao nhiêu?"
```

### 2. Chat API

#### Gửi câu hỏi qua API
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "gói cước TOUR xuất hiện ở phiếu yêu cầu nào?",
    "session_id": "test-session-123"
  }'
```

#### Response Format
```json
{
  "answer": "Câu trả lời từ RAG system...",
  "sources": [
    {
      "content": "Nội dung nguồn tham khảo",
      "score": 0.95,
      "metadata": {
        "source": "document.pdf",
        "page": 5
      }
    }
  ],
  "session_id": "test-session-123"
}
```

### 3. Truy Cập Web UI

- **Attu (Milvus Admin)**: http://localhost:8081
- **MinIO Console**: http://localhost:9001
- **API Docs**: http://localhost:8080/docs (nếu có Swagger)

## 📊 API Documentation

### Endpoints

#### POST `/chat`
Gửi câu hỏi và nhận câu trả lời từ RAG system

**Request Body:**
```json
{
  "question": "string",
  "session_id": "string (optional)",
  "max_results": "integer (optional, default: 5)"
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [
    {
      "content": "string",
      "score": "float",
      "metadata": "object"
    }
  ],
  "session_id": "string"
}
```

#### GET `/health`
Kiểm tra health status của services

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "milvus": "connected",
    "model": "loaded"
  }
}
```

## 🎯 Đánh Giá Mô Hình

Kết quả evaluation của các models:

| Model | Eval Loss |
|-------|-----------|
| Gemma 3-4B-IT | 0.01896 |
| Gemma 3-1B-IT | 0.02866 |

### Chạy Evaluation
```bash
docker compose run --rm app python evaluation/evaluate.py
```

## 📁 Cấu Trúc Dự Án

```
final_project/
├── app/                              # Frontend Streamlit Application
│   ├── streamlit_app.py             # Main Streamlit app
│   ├── chat_interface.py            # Chat UI components
│   ├── sidebar.py                   # Sidebar components
│   ├── api_utils.py                 # API communication utilities
│   ├── Dockerfile                   # Frontend container image
│   └── requirements.txt             # Frontend dependencies
│
├── backend/                          # Backend API & RAG Engine
│   ├── main.py                      # FastAPI main application
│   ├── rag_chain.py                 # RAG chain implementation
│   ├── search.py                    # Search functionality
│   ├── db_utils.py                  # Database utilities
│   ├── pydantic_models.py           # Data models
│   ├── Dockerfile                   # Backend container image
│   ├── requirements.txt             # Backend dependencies
|
│
├── data/                             # Persistent Data Storage
│   ├── final_output.json            # Input data for ingestion
│   ├── rag_app.db                   # SQLite database
│   ├── etcd/                        # Etcd data (Milvus coordination)
│   ├── milvus/                      # Milvus vector database
│   │   ├── data/
│   │   ├── rdb_data/
│   │   └── rdb_data_meta_kv/
│   └── minio/                       # MinIO object storage
│       └── a-bucket/
│
├── models/                           # Pre-downloaded AI Models
│   ├── bge-m3/                      # BGE-M3 Embedding Model
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── ...
│   ├── gemma-3-12b-it/              # Gemma 3 12B Instruct Model
│   │   ├── model-*.safetensors
│   │   ├── config.json
│   │   └── ...
│
│
├── docker-compose.yaml               # Docker Compose configuration
├── Dockerfile.ingest                 # Ingest service container
├── ingest.py                         # Data ingestion script
├── requirements-ingest.txt           # Ingest dependencies
├── download.ipynb                    # Jupyter notebook for downloading models
├── .env                              # Environment variables
├── .gitignore                        # Git ignore file
└── README.md                         # Documentation
```

## 🔧 Troubleshooting

### Lỗi thường gặp

#### 1. Container không khởi động
```bash
# Kiểm tra logs
docker compose logs -f [service_name]

# Restart services
docker compose restart
```

#### 2. Milvus connection error
```bash
# Kiểm tra Milvus đang chạy
docker compose ps milvus

# Kiểm tra logs
docker compose logs milvus
```

#### 3. Out of memory
```bash
# Tăng memory limit trong docker-compose.yaml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 8G
```

#### 4. GPU không được nhận diện
```bash
# Kiểm tra NVIDIA driver
nvidia-smi

# Kiểm tra Docker có nhận GPU
docker run --rm --gpus all nvidia/cuda:12.0-runtime-ubuntu20.04 nvidia-smi
```

### Làm sạch và Reset

```bash
# Dừng tất cả containers
docker compose down

# Xóa volumes (cẩn thận - sẽ mất dữ liệu)
docker compose down -v

# Xóa tất cả và rebuild
docker compose down -v
docker compose build --no-cache
docker compose up -d
```


## 📧 Liên Hệ

- **Author**: Hoàng Đức
- **GitHub**: [@hoangduc170803](https://github.com/hoangduc170803)
- **Repository**: [RAG](https://github.com/hoangduc170803/RAG)

## 🙏 Acknowledgments

- [Milvus](https://milvus.io/) - Vector database
- [MinIO](https://min.io/) - Object storage
- [Docker](https://www.docker.com/) - Containerization
- [Gemma Models](https://ai.google.dev/gemma) - Language models

---

⭐ Nếu bạn thấy dự án này hữu ích, hãy cho một star trên GitHub!
