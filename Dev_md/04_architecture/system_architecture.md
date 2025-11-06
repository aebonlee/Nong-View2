# 🏗️ Nong-View2 시스템 아키텍처

## 목차
1. [시스템 개요](#시스템-개요)
2. [아키텍처 다이어그램](#아키텍처-다이어그램)
3. [컴포넌트 설명](#컴포넌트-설명)
4. [데이터 플로우](#데이터-플로우)
5. [기술 스택](#기술-스택)
6. [배포 아키텍처](#배포-아키텍처)

---

## 시스템 개요

Nong-View2는 지리정보 기반 농업 영역 AI 분석 파이프라인으로, 6개의 독립적인 POD(Process-Oriented Design) 모듈로 구성됩니다.

### 핵심 특징
- **모듈화**: 독립적으로 실행 가능한 6개 POD
- **확장성**: 수평적 확장 가능
- **유연성**: 선택적 POD 실행
- **성능**: GPU 가속 및 병렬 처리

---

## 아키텍처 다이어그램

### 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────┐
│                     Nong-View2 Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Input   │  │  Config  │  │  Logger  │  │  Utils   │   │
│  │  Layer   │  │  Manager │  │  System  │  │  Module  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │           │
│  ┌────┴─────────────┴─────────────┴─────────────┴────┐      │
│  │                 Main Pipeline Engine               │      │
│  └────┬─────────────┬─────────────┬─────────────┬────┘      │
│       ▼             ▼             ▼             ▼           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  POD 1  │──▶│  POD 2  │──▶│  POD 3  │──▶│  POD 4  │        │
│  │Ingestion│  │Cropping │  │ Tiling  │  │   AI    │        │
│  └─────────┘  └─────────┘  └─────────┘  └────┬────┘        │
│                                              ▼              │
│                             ┌─────────┐  ┌─────────┐        │
│                             │  POD 5  │──▶│  POD 6  │        │
│                             │ Merging │  │  GPKG   │        │
│                             └─────────┘  └─────────┘        │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Output  │  │   Data   │  │  Cache   │  │ Monitor  │   │
│  │  Layer   │  │  Store   │  │  Manager │  │  Service │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### POD 상세 구조

```
┌─────────────────────────────────────┐
│            POD Structure             │
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐    │
│  │      Input Interface         │    │
│  └───────────┬─────────────────┘    │
│              ▼                       │
│  ┌─────────────────────────────┐    │
│  │      Validation Layer        │    │
│  └───────────┬─────────────────┘    │
│              ▼                       │
│  ┌─────────────────────────────┐    │
│  │    Processing Engine         │    │
│  │  ┌─────────┬──────────┐     │    │
│  │  │ Core    │ Helpers  │     │    │
│  │  │ Logic   │ & Utils  │     │    │
│  │  └─────────┴──────────┘     │    │
│  └───────────┬─────────────────┘    │
│              ▼                       │
│  ┌─────────────────────────────┐    │
│  │      Output Interface        │    │
│  └───────────┬─────────────────┘    │
│              ▼                       │
│  ┌─────────────────────────────┐    │
│  │        Data Store            │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

---

## 컴포넌트 설명

### 1. POD 1: 데이터 수집 (Ingestion)

```python
class IngestionEngine:
    """
    책임:
    - ECW/TIF 이미지 파일 처리
    - Shapefile 공간 데이터 로드
    - PNU Excel 데이터 매칭
    - 좌표계 통일 (EPSG:5186)
    """
```

**입력**:
- 정사영상 (ECW/TIF)
- Shapefile (경계 데이터)
- Excel (PNU 속성 데이터)

**출력**:
- 변환된 TIF 이미지
- 통합 GeoDataFrame
- 메타데이터 레지스트리

### 2. POD 2: 크롭핑 (Cropping)

```python
class CroppingEngine:
    """
    책임:
    - Convex Hull 알고리즘 적용
    - 최소경계사각형 생성
    - 버퍼 영역 처리
    - 이미지 클리핑
    """
```

**알고리즘**:
- Convex Hull 기반 MBR
- 좌표 변환 및 매칭
- 래스터 클리핑

### 3. POD 3: 타일링 (Tiling)

```python
class TilingEngine:
    """
    책임:
    - 이미지를 균일한 타일로 분할
    - 오버랩 처리
    - 공간 인덱스 생성
    - 빈 타일 제거
    """
```

**특징**:
- 적응형 타일 크기
- R-tree 인덱싱
- 메모리 효율적 처리

### 4. POD 4: AI 분석 (Analysis)

```python
class AnalysisEngine:
    """
    책임:
    - YOLOv11 모델 실행
    - 객체 탐지 및 세그멘테이션
    - 6개 농업 클래스 분류
    - 결과 후처리
    """
```

**AI 모델**:
- YOLOv11x-seg
- 6개 클래스 분류기
- GPU/CPU 자동 선택

### 5. POD 5: 병합 (Merging)

```python
class MergingEngine:
    """
    책임:
    - 타일별 결과 통합
    - NMS 중복 제거
    - 좌표 변환
    - 클래스별 그룹화
    """
```

**병합 전략**:
- NMS (Non-Maximum Suppression)
- Union (합집합)
- Overlap (교집합)

### 6. POD 6: GPKG 발행 (Export)

```python
class GPKGExporter:
    """
    책임:
    - GeoPackage 생성
    - 다중 레이어 구성
    - 통계 계산
    - 보고서 생성
    """
```

**출력물**:
- 표준 GeoPackage
- HTML 보고서
- 시각화 이미지

---

## 데이터 플로우

### 순차적 처리 플로우

```
[원본 데이터]
     │
     ▼
┌─────────────┐
│   POD 1     │ ◀── ECW/TIF, SHP, XLSX
│  Ingestion  │
└─────┬───────┘
      │ TIF, GeoJSON
      ▼
┌─────────────┐
│   POD 2     │ ◀── 통합 데이터
│  Cropping   │
└─────┬───────┘
      │ Cropped Images
      ▼
┌─────────────┐
│   POD 3     │ ◀── 크롭 영상
│   Tiling    │
└─────┬───────┘
      │ Tiles (1024x1024)
      ▼
┌─────────────┐
│   POD 4     │ ◀── 타일 이미지
│ AI Analysis │
└─────┬───────┘
      │ Detections JSON
      ▼
┌─────────────┐
│   POD 5     │ ◀── 개별 탐지 결과
│   Merging   │
└─────┬───────┘
      │ Merged GeoJSON
      ▼
┌─────────────┐
│   POD 6     │ ◀── 통합 결과
│ GPKG Export │
└─────┬───────┘
      │
      ▼
[최종 결과물]
- .gpkg 파일
- .html 보고서
- .png 시각화
```

### 데이터 스키마

#### POD1 출력 스키마
```json
{
  "images": [{
    "id": "string",
    "path": "string",
    "crs": "EPSG:5186",
    "bounds": [xmin, ymin, xmax, ymax]
  }],
  "parcels": [{
    "pnu": "string",
    "geometry": "GeoJSON",
    "properties": {}
  }]
}
```

#### POD4 탐지 결과 스키마
```json
{
  "detections": [{
    "class_id": 0,
    "class_name": "string",
    "confidence": 0.95,
    "bbox": [x1, y1, x2, y2],
    "polygon": [[x, y], ...],
    "tile_id": "string"
  }]
}
```

---

## 기술 스택

### 핵심 라이브러리

| 카테고리 | 라이브러리 | 버전 | 용도 |
|---------|-----------|------|------|
| 공간정보 | GDAL | 3.4+ | ECW 변환 |
| 래스터 | Rasterio | 1.3+ | 이미지 I/O |
| 벡터 | GeoPandas | 0.13+ | 공간 데이터 |
| AI | Ultralytics | 8.0+ | YOLOv11 |
| 시각화 | Matplotlib | 3.7+ | 플로팅 |
| 병렬처리 | multiprocessing | - | CPU 병렬화 |

### 시스템 요구사항

#### 최소 요구사항
```yaml
system:
  os: ["Windows 10+", "Ubuntu 20.04+"]
  python: "3.10+"
  ram: "8GB"
  storage: "20GB"
  
optional:
  gpu: "CUDA 11.8+"
  vram: "4GB+"
```

#### 권장 사양
```yaml
system:
  ram: "16GB+"
  storage: "50GB SSD"
  cpu: "8+ cores"
  
gpu:
  model: "RTX 3060+"
  vram: "6GB+"
```

---

## 배포 아키텍처

### 1. 단일 서버 배포

```
┌──────────────────────────────┐
│       Application Server      │
├──────────────────────────────┤
│  ┌────────────────────────┐  │
│  │    Docker Container    │  │
│  │  ┌──────────────────┐  │  │
│  │  │  Nong-View2 App  │  │  │
│  │  └──────────────────┘  │  │
│  │  ┌──────────────────┐  │  │
│  │  │     GDAL/GPU     │  │  │
│  │  └──────────────────┘  │  │
│  └────────────────────────┘  │
│                               │
│  ┌────────────────────────┐  │
│  │    File Storage        │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

### 2. 분산 처리 아키텍처

```
┌─────────────────────────────────────────┐
│            Load Balancer                │
└───────┬─────────────────────┬───────────┘
        │                     │
┌───────▼────────┐   ┌────────▼───────┐
│  Worker Node 1 │   │  Worker Node 2  │
│  - POD 1,2,3   │   │  - POD 1,2,3    │
└────────────────┘   └─────────────────┘
        │                     │
┌───────▼─────────────────────▼───────────┐
│           GPU Cluster (POD 4)           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │ │
│  └─────────┘  └─────────┘  └─────────┘ │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   Result Processor   │
        │    - POD 5,6         │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Object Storage     │
        │   - S3/MinIO         │
        └─────────────────────┘
```

### 3. 클라우드 네이티브 배포

```yaml
# Kubernetes 배포 예시
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nongview2
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nongview2
  template:
    metadata:
      labels:
        app: nongview2
    spec:
      containers:
      - name: nongview2
        image: nongview2:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## 성능 최적화 전략

### 1. 캐싱 전략

```python
# Redis 기반 캐싱
cache_config = {
    'type': 'redis',
    'host': 'localhost',
    'port': 6379,
    'ttl': 3600
}
```

### 2. 병렬 처리

```python
# POD별 병렬화 설정
parallelization = {
    'pod3_tiling': {'workers': 8},
    'pod4_analysis': {'batch_size': 16},
    'pod5_merging': {'threads': 4}
}
```

### 3. 리소스 관리

```python
# 동적 리소스 할당
resource_limits = {
    'max_memory': '16GB',
    'gpu_memory_fraction': 0.8,
    'cpu_cores': 8
}
```

---

## 모니터링 및 로깅

### 로깅 아키텍처

```
Application Logs ──▶ Fluentd ──▶ Elasticsearch ──▶ Kibana
     │                               │
     └── Metrics ──▶ Prometheus ──▶ Grafana
```

### 주요 메트릭

- **처리 시간**: 각 POD별 실행 시간
- **메모리 사용량**: 피크 메모리
- **GPU 사용률**: CUDA 메모리 및 사용률
- **처리량**: 시간당 처리 이미지 수
- **에러율**: POD별 실패율

---

## 보안 아키텍처

### 1. 인증 및 권한

```python
# JWT 기반 인증
authentication = {
    'type': 'JWT',
    'secret': 'environment_variable',
    'expiry': 3600
}
```

### 2. 데이터 보호

- **암호화**: TLS 1.3 전송 암호화
- **접근 제어**: RBAC 기반
- **감사 로그**: 모든 작업 기록

### 3. 입력 검증

```python
validation_rules = {
    'file_size': '10GB',
    'file_types': ['.tif', '.ecw', '.shp'],
    'coordinate_systems': ['EPSG:5186', 'EPSG:4326']
}
```

---

## 확장성 고려사항

### 수평적 확장

- **POD 레벨**: 각 POD 독립 스케일링
- **타일 레벨**: 타일 단위 분산 처리
- **GPU 클러스터**: 다중 GPU 노드

### 수직적 확장

- **메모리**: 대용량 이미지 처리
- **GPU**: 고성능 모델 사용
- **스토리지**: SSD/NVMe 활용

---

## 향후 로드맵

### Phase 1 (단기)
- WebSocket 기반 실시간 진행상황
- REST API 엔드포인트
- 웹 UI 대시보드

### Phase 2 (중기)
- Kubernetes 오케스트레이션
- 분산 처리 시스템
- 실시간 스트리밍

### Phase 3 (장기)
- AutoML 통합
- 멀티모달 AI
- 엣지 컴퓨팅

---

문서 버전: 1.0.0
최종 수정: 2025-11-06