### 1.Build image để ingest dữ liệu ###
docker compose --profile manual build ingest


### 2.Chạy trước những container cần thiết để ingest dữ liệu vì đã persist bằng cách mount vào folder data ###
docker compose up -d etcd minio milvus attu


### 3.chạy script ingest.py để ingest data vào milvus ###
docker compose --profile manual run --rm ingest

### 4. model ###  
eval_loss gemma-3-4b-it : 0.01895984821021557
eval_loss gemma-3-1b-it : 0.028661


### 5. Test search ###
docker compose up -d
docker compose run --rm app python search/search.py "Phí tham gia gói M2M7S1 là bao nhiêu?"  

### 6. Test backend ###
curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d @- <<-EOF
{
  "question": "gói cước TOUR xuất hiện ở phiếu yêu cầu nào ?",
  "session_id": "test-session-123"
}
EOF


### 7. nếu muốn ingest lại dataset ###  
$ docker compose --profile manual run --rm ingest \
  --collection my_rag_collection \
  --input-json ./data/final_output.json \
  --drop-existing 



### setup trên server
# sử dụng 
# Bước 1: Cập nhật package list
 apt-get update

apt install -y nvidia-driver-580 nvidia-utils-580


# Bước 2: Cài đặt các package cần thiết
 apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Bước 3: Thêm Docker's official GPG key
 mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Bước 4: Thêm Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Bước 5: Cập nhật package list lại sau khi thêm repo
 apt-get update

# Bước 6: Cài đặt Docker
 apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Bước 7: Khởi động và enable Docker service
 systemctl start docker
 systemctl enable docker

# Bước 8: Thêm user hiện tại vào group docker (tùy chọn)
 usermod -aG docker $USER

# Bước 9: Kiểm tra cài đặt
 docker --version
 docker compose version


# Cài Docker Compose từ repository Ubuntu
apt update
apt install -y docker-compose

# Kiểm tra
docker-compose --version

# Bước 1: Thêm NVIDIA repository

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg


distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Bước 2: Cập nhật và cài đặt
apt update
apt install -y nvidia-container-toolkit

# Bước 3: Cấu hình Docker
nvidia-ctk runtime configure --runtime=docker

# Bước 4: Restart Docker
systemctl restart docker

# Bước 5: Kiểm tra
docker run --rm --gpus all nvidia/cuda:12.0-runtime-ubuntu20.04 nvidia-smi


# Bước 1: Cho phép SSH trước
ufw allow ssh

# Bước 2: Mở các port cần thiết
ufw allow 8080
ufw allow 8081  
ufw allow 8000
ufw allow 9001
ufw allow 19530
ufw allow 9091
ufw allow 8001
ufw allow 8501

# Bước 3: Cuối cùng mới enable
ufw enable 

# Kiểm tra status
ufw status

# IP public 
curl ifconfig.me


# Di chuyển vào thư mục project
cd /root/final_project

# Kiểm tra file docker-compose.yml có tồn tại không
ls -la docker-compose.yaml

# Chạy tất cả services
docker compose up -d

# Kiểm tra services đang chạy
docker-compose ps

# Xem logs nếu có lỗi
docker-compose logs -f

