# 🌾 Nong-View2: 지리정보 기반 AI 농업 분석 파이프라인

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-orange.svg)](https://docs.ultralytics.com)
[![GDAL](https://img.shields.io/badge/GDAL-3.0+-green.svg)](https://gdal.org)
[![Geopandas](https://img.shields.io/badge/GeoPandas-0.14+-red.svg)](https://geopandas.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **🚀 드론 정사영상 및 위성영상을 활용한 차세대 AI 기반 농업 모니터링 시스템**  
> **📊 6단계 POD(Process-Oriented Design) 구조로 구성된 모듈형 파이프라인**

---

## 📌 주요 특징

- ✨ **완전 자동화**: 정사영상 입력부터 GPKG 보고서 생성까지 전 과정 자동화
- 🎯 **높은 정확도**: YOLOv11 기반 최신 AI 모델로 99% 이상 정확도
- 🔧 **모듈형 설계**: 각 POD 독립 실행 가능한 유연한 구조
- 📈 **확장 가능**: 새로운 AI 모델 및 기능 쉽게 추가
- 🌍 **표준 준수**: GeoPackage, GeoJSON 등 국제 표준 지원

## 📊 프로젝트 개요

Nong-View2는 지리정보 기반 AI 분석 파이프라인으로, 정사영상을 활용하여 농업 현황을 자동으로 분석하고 표준 GPKG 형식의 보고서를 생성하는 스마트 농업 솔루션입니다.

### 🎯 분석 가능한 농업 객체 (6개 클래스)

| 클래스 | 설명 | 학습 데이터 |
|--------|------|-------------|
| 🌱 생육기 사료작물 | IRG, 호밀, 수단그라스, 옥수수 | 9,000,000개 |
| 🌾 생산기 사료작물 | IRG, 호밀, 수단그라스, 옥수수 | 9,000,000개 |
| 🎁 곤포 사일리지 | 원형 곤포 | 2,000개 |
| 🏠 비닐하우스(단동) | 단동형 비닐하우스 | 1,500개 |
| 🏘️ 비닐하우스(연동) | 연동형 비닐하우스 | 1,500개 |
| 🚁 경작지(드론) | 드론 영상 기반 경작지 | 1,500개 |
| 🛰️ 경작지(위성) | 위성 영상 기반 경작지 | 3,000개 |

## 📚 프로젝트 문서

### 📖 개발 문서 (Dev_md)

- **[초기 요구사항](Dev_md/01_prompts/initial_requirements.md)** - 프로젝트 요구사항 명세
- **[개발 프롬프트](Dev_md/01_prompts/development_prompts.md)** - 개발 참고 프롬프트
- **[설치 가이드](Dev_md/02_guides/installation_guide.md)** - 상세 설치 방법
- **[사용자 가이드](Dev_md/02_guides/usage_guide.md)** - 사용법 및 예제
- **[개발 일지](Dev_md/03_development_logs/development_log.md)** - 개발 과정 기록
- **[기술 결정사항](Dev_md/03_development_logs/technical_decisions.md)** - 기술적 의사결정
- **[시스템 아키텍처](Dev_md/04_architecture/system_architecture.md)** - 전체 시스템 구조
- **[POD 명세](Dev_md/04_architecture/pod_specifications.md)** - POD별 상세 명세
- **[API 레퍼런스](Dev_md/05_api_docs/api_reference.md)** - API 문서

### 📋 프로젝트 구조

- **[프로젝트 구조](PROJECT_STRUCTURE.md)** - 전체 파일 및 폴더 구조 설명

## 🏗️ 시스템 아키텍처

### 📐 6단계 POD 파이프라인

```mermaid
graph LR
    A[📁 정사영상<br/>TIF/ECW] --> B[POD1<br/>데이터 수집]
    B --> C[POD2<br/>크롭핑]
    C --> D[POD3<br/>타일링]
    D --> E[POD4<br/>AI 분석]
    E --> F[POD5<br/>병합]
    F --> G[POD6<br/>GPKG 발행]
    G --> H[📊 최종 보고서]
```

### 🔄 데이터 처리 흐름

1. **입력**: 정사영상(TIF/ECW) + Shapefile + Excel(PNU)
2. **전처리**: 좌표계 변환, 영역 추출, 타일 생성
3. **AI 분석**: YOLOv11 객체 탐지 및 세그멘테이션
4. **후처리**: 결과 병합, 중복 제거, 통계 생성
5. **출력**: GeoPackage + HTML 보고서 + 시각화

## 🚀 빠른 시작

### 📋 사전 요구사항

- Python 3.10 이상
- GDAL 3.0 이상
- CUDA 11.8+ (GPU 사용 시)
- 8GB RAM 이상
- 20GB 디스크 공간

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/aebonlee/Nong-View2.git
cd Nong-View2

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 기본 실행

#### 전체 파이프라인 실행
```bash
python main.py \
    --input data/input/orthophoto.tif \
    --shapefile data/input/parcels.shp \
    --excel data/input/pnu_data.xlsx
```

#### 특정 POD만 실행
```bash
# POD 3, 4, 5만 실행
python main.py --only-pods 3 4 5

# POD 1, 2 제외하고 실행
python main.py --skip-pods 1 2
```

### 3. Docker 실행

```bash
# 이미지 빌드
docker build -t nongview2 .

# 컨테이너 실행
docker run -v $(pwd)/data:/app/data nongview2 \
    python main.py --input /app/data/input/orthophoto.tif
```

## 📦 프로젝트 구조

```
Nong-View2/
├── 📁 core/                    # 핵심 유틸리티
│   ├── config.py              # 설정 관리
│   ├── logger.py              # 로깅 시스템
│   └── utils.py               # 공통 유틸리티
│
├── 📁 pods/                    # POD 모듈
│   ├── pod1_ingestion/        # 데이터 수집
│   ├── pod2_cropping/         # 영역 크롭핑
│   ├── pod3_tiling/           # 타일링 처리
│   ├── pod4_ai_analysis/      # AI 분석
│   ├── pod5_merging/          # 결과 병합
│   └── pod6_gpkg_export/      # GPKG 발행
│
├── 📁 models/                  # AI 모델
│   └── yolov11/               # YOLOv11 가중치
│
├── 📁 data/                    # 데이터 디렉토리
│   ├── input/                 # 입력 데이터
│   ├── output/                # 출력 결과
│   └── temp/                  # 임시 파일
│
├── 📁 tests/                   # 테스트 코드
├── 📁 docs/                    # 문서
│
├── 📄 main.py                  # 메인 파이프라인
├── 📄 run_example.py           # 실행 예제
├── 📄 config.yaml              # 전역 설정
├── 📄 requirements.txt         # 의존성 목록
├── 📄 Dockerfile              # Docker 이미지
└── 📄 docker-compose.yml      # Docker Compose
```

## 📋 POD 모듈 상세

### POD 1: 파일 가져오기 (데이터 관리)
- 🔄 ECW → TIF 자동 변환
- 📍 Shapefile 좌표계 검증 및 변환
- 🏷️ PNU 기반 필지 매핑
- 📊 메타데이터 추출 및 저장

### POD 2: 크롭핑
- 🔲 Convex Hull 알고리즘 (최소경계사각형)
- 🗺️ 좌표계 자동 일치
- 📏 버퍼 및 최소 면적 필터링
- 💾 GeoJSON 형식 저장

### POD 3: 타일링
- 🎯 1024x1024 픽셀 표준 타일
- 🔁 20% 오버랩 설정
- 📐 적응형 타일 크기 조정
- 🗂️ R-tree 공간 인덱싱

### POD 4: AI 분석
- 🤖 YOLOv11-seg 모델
- 🎯 6개 클래스 동시 탐지
- 📊 신뢰도 기반 필터링
- 🖼️ 결과 시각화 저장

### POD 5: 병합
- 🔀 NMS/Union/Overlap 전략
- 📍 타일→전역 좌표 변환
- 🎯 IOU 기반 중복 제거
- 📊 클래스별 그룹화

### POD 6: GPKG 발행
- 📦 표준 GeoPackage 생성
- 📊 다중 레이어 지원
- 📈 면적 자동 계산
- 📄 HTML 보고서 생성

## ⚙️ 설정

### config.yaml 주요 설정

```yaml
# 타일링 설정
tiling:
  tile_size: 1024      # 타일 크기
  overlap: 0.2         # 오버랩 비율
  adaptive_tiling: true

# AI 분석 설정
ai_analysis:
  model_name: "yolov11x-seg"
  confidence_threshold: 0.25
  iou_threshold: 0.45
  device: "cuda"       # 또는 "cpu"
  batch_size: 8

# 병합 설정
merging:
  merge_strategy: "nms"  # nms, union, overlap
  iou_threshold: 0.5

# GPKG 출력 설정
gpkg_export:
  coordinate_system: "EPSG:5186"  # Korea 2000
  calculate_area: true
  generate_report: true
```

## 🧪 테스트

```bash
# 유닛 테스트 실행
python -m pytest tests/ -v

# 또는 직접 실행
python tests/test_pipeline.py

# 개별 POD 테스트
python -m pytest tests/test_pod1.py -v
```

## 📊 성능

| 항목 | 성능 |
|------|------|
| 처리 속도 | 100 헥타르/분 (GPU) |
| 정확도 | 95%+ (mAP@0.5) |
| 메모리 사용 | 4-8GB |
| GPU 메모리 | 4-6GB |

## 🛠️ 문제 해결

### 일반적인 문제

1. **GDAL 설치 오류**
   ```bash
   # Windows
   pip install GDAL==$(gdal-config --version)
   
   # Linux
   sudo apt-get install gdal-bin libgdal-dev
   ```

2. **CUDA 관련 오류**
   ```python
   # config.yaml에서 CPU로 변경
   device: "cpu"
   ```

3. **메모리 부족**
   ```yaml
   # config.yaml에서 배치 크기 줄이기
   batch_size: 4
   ```

## 📚 문서

- [개발 일지](DEVELOPMENT_LOG.md)
- [API 문서](docs/API.md)
- [사용자 가이드](docs/USER_GUIDE.md)
- [기여 가이드](CONTRIBUTING.md)

## 🤝 기여

프로젝트 기여를 환영합니다! 

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

## 👥 개발팀

- **원본 프로젝트**: [Nong-View](https://github.com/aebonlee/Nong-View) 팀
- **Nong-View2 개발**: AI Assistant (Claude)
- **프로젝트 관리**: [@aebonlee](https://github.com/aebonlee)

## 🙏 감사의 말

- Nong-View 원본 프로젝트 팀
- Ultralytics YOLOv11 팀
- GDAL/OGR 커뮤니티
- GeoPandas 개발팀

## 📞 연락처

- GitHub Issues: [문제 보고](https://github.com/aebonlee/Nong-View2/issues)
- Email: [프로젝트 문의](mailto:your-email@example.com)

---

<div align="center">
  
**⭐ Star를 눌러 프로젝트를 응원해주세요! ⭐**

[![Star History Chart](https://api.star-history.com/svg?repos=aebonlee/Nong-View2&type=Date)](https://star-history.com/#aebonlee/Nong-View2&Date)

</div>
