# 📖 Nong-View2 사용자 가이드

## 목차
1. [기본 사용법](#1-기본-사용법)
2. [POD별 사용법](#2-pod별-사용법)
3. [고급 설정](#3-고급-설정)
4. [데이터 준비](#4-데이터-준비)
5. [결과 해석](#5-결과-해석)
6. [최적화 팁](#6-최적화-팁)

---

## 1. 기본 사용법

### 전체 파이프라인 실행

#### 명령줄 인터페이스
```bash
# 기본 실행 (모든 POD 실행)
python main.py \
    --input data/input/orthophoto.tif \
    --shapefile data/input/parcels.shp \
    --excel data/input/pnu_data.xlsx

# 상세 로그 출력
python main.py \
    --input data/input/orthophoto.tif \
    --shapefile data/input/parcels.shp \
    --verbose
```

#### Python 스크립트에서 사용
```python
from main import NongViewPipeline

# 파이프라인 초기화
pipeline = NongViewPipeline(config_path="config.yaml")

# 실행
results = pipeline.run(
    input_image="data/input/orthophoto.tif",
    input_shapefile="data/input/parcels.shp",
    input_excel="data/input/pnu_data.xlsx"
)

# 결과 확인
print(f"GPKG 출력: {results['pod6']['gpkg_path']}")
```

### 배치 처리
```bash
# Windows 배치 실행
run.bat full --input data/input/*.tif

# Linux/Mac 배치 실행
./run.sh full --input data/input/*.tif
```

---

## 2. POD별 사용법

### POD1: 데이터 수집

#### 단독 실행
```python
from pods.pod1_ingestion import IngestionEngine

engine = IngestionEngine()
results = engine.process(
    image_path="orthophoto.ecw",  # ECW는 자동으로 TIF 변환
    shapefile_path="parcels.shp",
    excel_path="pnu_data.xlsx"
)
```

#### 지원 파일 형식
- **이미지**: TIF, TIFF, ECW, JPG, PNG
- **벡터**: SHP, GeoJSON, GPKG
- **테이블**: XLSX, XLS, CSV

### POD2: 크롭핑

#### 단독 실행
```python
from pods.pod2_cropping import CroppingEngine

engine = CroppingEngine()
results = engine.process()  # POD1 출력 자동 사용
```

#### 커스텀 설정
```python
# config.yaml 수정 또는
engine = CroppingEngine({
    'use_convex_hull': True,
    'buffer_size': 20,  # 20m 버퍼
    'min_area': 200    # 최소 200㎡
})
```

### POD3: 타일링

#### 단독 실행
```python
from pods.pod3_tiling import TilingEngine

engine = TilingEngine()
results = engine.process()
```

#### 타일 크기 조정
```python
engine = TilingEngine({
    'tile_size': 2048,     # 큰 타일
    'overlap': 0.3,        # 30% 오버랩
    'adaptive_tiling': False  # 고정 크기
})
```

### POD4: AI 분석

#### 단독 실행
```python
from pods.pod4_ai_analysis import AnalysisEngine

engine = AnalysisEngine()
results = engine.process()
```

#### 모델 설정
```python
engine = AnalysisEngine({
    'model_name': 'yolov11x-seg',
    'confidence_threshold': 0.3,
    'device': 'cuda',
    'batch_size': 16
})
```

### POD5: 병합

#### 단독 실행
```python
from pods.pod5_merging import MergingEngine

engine = MergingEngine()
results = engine.process()
```

#### 병합 전략 변경
```python
engine = MergingEngine({
    'merge_strategy': 'union',  # nms, union, overlap
    'iou_threshold': 0.3
})
```

### POD6: GPKG 발행

#### 단독 실행
```python
from pods.pod6_gpkg_export import GPKGExporter

exporter = GPKGExporter()
results = exporter.process()
```

---

## 3. 고급 설정

### config.yaml 구조

```yaml
# 프로젝트 설정
project:
  name: "농업지역_분석"
  version: "1.0.0"

# 경로 설정
paths:
  input_dir: "data/input"
  output_dir: "data/output"
  model_dir: "models/yolov11"

# POD별 세부 설정
ingestion:
  target_crs: "EPSG:5186"  # Korea 2000
  ecw_to_tif: true

cropping:
  use_convex_hull: true
  buffer_size: 10

tiling:
  tile_size: 1024
  overlap: 0.2

ai_analysis:
  model_name: "yolov11x-seg"
  classes:
    0: "생육기_사료작물"
    1: "생산기_사료작물"
    2: "곤포_사일리지"
    3: "비닐하우스_단동"
    4: "비닐하우스_연동"
    5: "경작지_드론"
    6: "경작지_위성"
  confidence_threshold: 0.25
  device: "cuda"

merging:
  merge_strategy: "nms"
  iou_threshold: 0.5

gpkg_export:
  calculate_area: true
  generate_report: true
```

### 선택적 POD 실행

```bash
# POD 3, 4, 5만 실행
python main.py --only-pods 3 4 5

# POD 1, 2 건너뛰기
python main.py --skip-pods 1 2

# POD 4부터 끝까지
python main.py --only-pods 4 5 6
```

---

## 4. 데이터 준비

### 정사영상 준비

#### 권장 사양
- **해상도**: 10-50cm/pixel
- **형식**: GeoTIFF (압축 권장)
- **좌표계**: EPSG:5186 (Korea 2000)
- **크기**: 10GB 이하 (대용량은 분할 처리)

#### 전처리 (선택사항)
```python
# 좌표계 변환
gdal_translate -a_srs EPSG:5186 input.tif output.tif

# 압축 적용
gdal_translate -co COMPRESS=LZW -co TILED=YES input.tif output.tif
```

### Shapefile 준비

#### 필수 필드
- **geometry**: 폴리곤 형태
- **PNU**: 19자리 필지번호 (선택)
- **address**: 주소 정보 (선택)

#### 예제 구조
```python
import geopandas as gpd

# Shapefile 확인
gdf = gpd.read_file("parcels.shp")
print(gdf.columns)  # ['geometry', 'PNU', 'address', ...]
print(gdf.crs)      # EPSG:5186
```

### Excel 데이터 준비

#### 필수 컬럼
- **PNU**: 필지번호 (19자리)
- **지번**: 주소 정보

#### 예제 형식
| PNU | 지번 | 면적 | 소유자 |
|-----|------|------|--------|
| 4511010100102080000 | 전북 전주시 덕진구 123 | 1500 | 홍길동 |

---

## 5. 결과 해석

### 출력 디렉토리 구조

```
data/output/
├── pod1_output/
│   ├── images/          # 변환된 이미지
│   ├── shapefiles/      # 처리된 Shapefile
│   └── registry.json    # 메타데이터
├── pod2_output/
│   ├── cropped_images/  # 크롭된 이미지
│   └── cropped_regions.geojson
├── pod3_output/
│   ├── tiles/           # 타일 이미지
│   └── tile_index.gpkg
├── pod4_output/
│   ├── detections/      # 탐지 결과
│   ├── visualizations/  # 시각화
│   └── analysis_results.json
├── pod5_output/
│   ├── merged_detections.geojson
│   └── merging_results.json
└── pod6_output/
    ├── nongview_results_*.gpkg  # 최종 GPKG
    ├── analysis_report.html     # HTML 보고서
    └── visualization.png        # 전체 시각화
```

### GPKG 레이어 구조

```sql
-- GeoPackage 레이어
1. parcels         -- 필지 경계
2. detections      -- AI 탐지 결과
3. clipped_detections -- 필지별 클립된 결과
4. statistics      -- 통계 테이블
```

### 결과 시각화

```python
import geopandas as gpd
import matplotlib.pyplot as plt

# GPKG 읽기
gpkg_path = "data/output/pod6_output/nongview_results.gpkg"

# 레이어별 읽기
parcels = gpd.read_file(gpkg_path, layer='parcels')
detections = gpd.read_file(gpkg_path, layer='detections')

# 시각화
fig, ax = plt.subplots(figsize=(12, 10))
parcels.plot(ax=ax, color='none', edgecolor='black')
detections.plot(ax=ax, column='class_name', legend=True, alpha=0.7)
plt.show()
```

### HTML 보고서 내용

- **요약 통계**: 클래스별 탐지 개수, 면적
- **필지별 통계**: PNU별 상세 결과
- **신뢰도 분포**: 클래스별 신뢰도 통계
- **처리 메타데이터**: 처리 시간, 설정값

---

## 6. 최적화 팁

### 메모리 최적화

```yaml
# 대용량 이미지 처리
tiling:
  tile_size: 512  # 작은 타일
  
ai_analysis:
  batch_size: 4   # 작은 배치
  
performance:
  max_memory: "4GB"
  cache_enabled: false
```

### GPU 최적화

```yaml
# GPU 메모리 관리
ai_analysis:
  device: "cuda"
  batch_size: 16  # GPU에 따라 조정
  
performance:
  gpu_memory_fraction: 0.8  # GPU 메모리 80% 사용
```

### 처리 속도 향상

```bash
# 병렬 처리 활성화
python main.py --num-workers 8

# 특정 영역만 처리
python main.py --bbox "127.0 35.0 127.1 35.1"

# 저해상도 프리뷰
python main.py --preview --scale 0.25
```

### 디버깅

```python
# 상세 로그 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# 단계별 결과 저장
pipeline = NongViewPipeline()
pipeline.save_intermediate = True
```

### 배치 처리 스크립트

```python
# batch_process.py
import glob
from main import NongViewPipeline

# 여러 파일 처리
files = glob.glob("data/input/*.tif")
pipeline = NongViewPipeline()

for file in files:
    try:
        results = pipeline.run(input_image=file)
        print(f"✓ {file}: {results['pod6']['gpkg_path']}")
    except Exception as e:
        print(f"✗ {file}: {e}")
```

---

## 자주 묻는 질문 (FAQ)

**Q: ECW 파일이 자동 변환되지 않아요**
- GDAL ECW 드라이버 설치 확인
- 수동 변환: `gdal_translate input.ecw output.tif`

**Q: GPU를 사용하지 않아요**
- CUDA 설치 확인: `nvidia-smi`
- PyTorch GPU 확인: `torch.cuda.is_available()`
- config.yaml에서 `device: "cuda"` 설정

**Q: 메모리 부족 오류가 발생해요**
- 타일 크기 줄이기
- 배치 크기 줄이기
- 시스템 메모리 확인

**Q: 결과가 부정확해요**
- 신뢰도 임계값 조정
- 더 많은 학습 데이터 필요
- 이미지 품질 확인

---

## 지원 및 문의

- GitHub Issues: https://github.com/aebonlee/Nong-View2/issues
- 문서: https://github.com/aebonlee/Nong-View2/wiki
- 이메일: support@nongview.com