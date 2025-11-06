# 📦 Nong-View2 설치 가이드

## 시스템 요구사항

### 최소 요구사항
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.10 이상
- **RAM**: 8GB
- **Storage**: 20GB 여유 공간
- **GPU**: 선택사항 (CUDA 11.8+ 지원 GPU)

### 권장 사양
- **RAM**: 16GB 이상
- **GPU**: NVIDIA RTX 3060 이상 (VRAM 6GB+)
- **Storage**: SSD 50GB 이상

## 1. 기본 설치 (Windows)

### Step 1: Python 설치
```powershell
# Python 3.10+ 설치 확인
python --version

# pip 업그레이드
python -m pip install --upgrade pip
```

### Step 2: GDAL 설치 (Windows)
```powershell
# OSGeo4W 설치 (권장)
# https://trac.osgeo.org/osgeo4w/ 에서 설치

# 또는 pip로 설치 (버전 확인 필요)
pip install GDAL==3.4.3
```

### Step 3: 프로젝트 클론
```powershell
git clone https://github.com/aebonlee/Nong-View2.git
cd Nong-View2
```

### Step 4: 가상환경 설정
```powershell
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
.\venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 2. 기본 설치 (Linux/Ubuntu)

### Step 1: 시스템 패키지 설치
```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    gcc \
    g++ \
    libspatialindex-dev
```

### Step 2: 환경변수 설정
```bash
# GDAL 환경변수
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
export GDAL_VERSION=$(gdal-config --version)

# .bashrc에 추가 (영구 설정)
echo 'export CPLUS_INCLUDE_PATH=/usr/include/gdal' >> ~/.bashrc
echo 'export C_INCLUDE_PATH=/usr/include/gdal' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: 프로젝트 설치
```bash
# 프로젝트 클론
git clone https://github.com/aebonlee/Nong-View2.git
cd Nong-View2

# 가상환경 생성 및 활성화
python3.10 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install --upgrade pip
pip install GDAL==$GDAL_VERSION
pip install -r requirements.txt
```

## 3. GPU 지원 설치 (NVIDIA)

### CUDA 및 cuDNN 설치
```bash
# CUDA 11.8 설치 (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-8

# PyTorch GPU 버전 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### GPU 확인
```python
# Python에서 GPU 확인
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

## 4. Docker 설치

### Docker 이미지 빌드
```bash
# Dockerfile을 사용하여 이미지 빌드
docker build -t nongview2:latest .

# GPU 지원 Docker 이미지 (nvidia-docker 필요)
docker build -f Dockerfile.gpu -t nongview2:gpu .
```

### Docker Compose 사용
```bash
# 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 중지
docker-compose down
```

## 5. 개발 환경 설정

### VS Code 설정
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"]
}
```

### Pre-commit 훅 설정
```bash
# pre-commit 설치
pip install pre-commit

# .pre-commit-config.yaml 생성
pre-commit install

# 수동 실행
pre-commit run --all-files
```

## 6. 설치 확인

### 기본 테스트
```bash
# 설치 확인 스크립트 실행
python -c "
import gdal
import rasterio
import geopandas
import torch
from ultralytics import YOLO
print('All packages imported successfully!')
print(f'GDAL version: {gdal.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### POD 모듈 테스트
```bash
# 단위 테스트 실행
pytest tests/test_pipeline.py -v

# 개별 POD 테스트
python -c "from pods.pod1_ingestion import IngestionEngine; print('POD1 OK')"
python -c "from pods.pod2_cropping import CroppingEngine; print('POD2 OK')"
python -c "from pods.pod3_tiling import TilingEngine; print('POD3 OK')"
python -c "from pods.pod4_ai_analysis import AnalysisEngine; print('POD4 OK')"
python -c "from pods.pod5_merging import MergingEngine; print('POD5 OK')"
python -c "from pods.pod6_gpkg_export import GPKGExporter; print('POD6 OK')"
```

## 7. 문제 해결

### GDAL 설치 오류
```bash
# Windows - OSGeo4W Shell 사용
pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/path/to/gdal/include" --global-option="-L/path/to/gdal/lib"

# Linux - 빌드 도구 설치
sudo apt-get install build-essential
pip install GDAL==$(gdal-config --version)
```

### Shapely 설치 오류
```bash
# Windows
pip install shapely --no-binary shapely

# Linux
sudo apt-get install libgeos-dev
pip install shapely
```

### PyTorch 설치 오류
```bash
# CPU 버전
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 특정 CUDA 버전
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 메모리 부족 오류
```yaml
# config.yaml 수정
ai_analysis:
  batch_size: 4  # 줄이기
  device: "cpu"  # GPU 메모리 부족 시

performance:
  max_memory: "4GB"
  cache_enabled: false
```

## 8. 업데이트 및 제거

### 프로젝트 업데이트
```bash
# 최신 버전 가져오기
git pull origin main

# 의존성 업데이트
pip install -r requirements.txt --upgrade
```

### 완전 제거
```bash
# 가상환경 비활성화
deactivate

# Windows
rmdir /s venv
del /q *.pyc

# Linux/Mac
rm -rf venv
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete
```

## 9. 추가 리소스

- [Python 설치 가이드](https://www.python.org/downloads/)
- [GDAL 설치 가이드](https://gdal.org/download.html)
- [CUDA 설치 가이드](https://developer.nvidia.com/cuda-downloads)
- [Docker 설치 가이드](https://docs.docker.com/get-docker/)

## 10. 지원

설치 중 문제가 발생하면:
1. [GitHub Issues](https://github.com/aebonlee/Nong-View2/issues) 확인
2. 새 이슈 생성 시 다음 정보 포함:
   - OS 및 버전
   - Python 버전
   - 에러 메시지 전체
   - `pip list` 출력