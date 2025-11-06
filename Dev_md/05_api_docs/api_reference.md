# 📚 Nong-View2 API Reference

## 목차
1. [Main Pipeline API](#main-pipeline-api)
2. [POD1: Ingestion API](#pod1-ingestion-api)
3. [POD2: Cropping API](#pod2-cropping-api)
4. [POD3: Tiling API](#pod3-tiling-api)
5. [POD4: Analysis API](#pod4-analysis-api)
6. [POD5: Merging API](#pod5-merging-api)
7. [POD6: GPKG Export API](#pod6-gpkg-export-api)
8. [Utility APIs](#utility-apis)

---

## Main Pipeline API

### NongViewPipeline

메인 파이프라인 클래스로 모든 POD를 통합 관리합니다.

```python
from main import NongViewPipeline
```

#### `__init__(config_path: str = "config.yaml")`

파이프라인 초기화

**Parameters:**
- `config_path` (str): 설정 파일 경로

**Example:**
```python
pipeline = NongViewPipeline("config.yaml")
```

#### `run(**kwargs) -> Dict[str, Any]`

파이프라인 실행

**Parameters:**
- `input_image` (str, optional): 입력 이미지 경로
- `input_shapefile` (str, optional): Shapefile 경로
- `input_excel` (str, optional): Excel 파일 경로
- `skip_pods` (List[int], optional): 건너뛸 POD 번호
- `only_pods` (List[int], optional): 실행할 POD 번호만

**Returns:**
- Dict: 각 POD 실행 결과

**Example:**
```python
results = pipeline.run(
    input_image="data/input/image.tif",
    input_shapefile="data/input/parcels.shp",
    only_pods=[1, 2, 3]
)
```

---

## POD1: Ingestion API

### IngestionEngine

데이터 수집 및 전처리 엔진

```python
from pods.pod1_ingestion import IngestionEngine
```

#### `__init__(config: Dict[str, Any] = None)`

엔진 초기화

**Parameters:**
- `config` (dict): POD 설정
  - `target_crs` (str): 목표 좌표계 (기본: "EPSG:5186")
  - `ecw_to_tif` (bool): ECW 자동 변환 (기본: True)
  - `output_dir` (str): 출력 디렉토리

#### `process(**kwargs) -> Dict[str, Any]`

데이터 처리 메서드

**Parameters:**
- `image_path` (str): 이미지 파일 경로
- `shapefile_path` (str): Shapefile 경로
- `excel_path` (str): Excel 파일 경로

**Returns:**
```python
{
    'images': [
        {
            'path': str,
            'crs': str,
            'bounds': tuple,
            'shape': tuple,
            'dtype': str
        }
    ],
    'parcels': GeoDataFrame,
    'metadata': dict
}
```

**Example:**
```python
engine = IngestionEngine()
result = engine.process(
    image_path="input.ecw",
    shapefile_path="parcels.shp"
)
```

#### `convert_ecw_to_tif(ecw_path: str, output_path: str = None) -> str`

ECW를 TIF로 변환

**Parameters:**
- `ecw_path` (str): ECW 파일 경로
- `output_path` (str, optional): 출력 경로

**Returns:**
- str: 변환된 TIF 파일 경로

---

## POD2: Cropping API

### CroppingEngine

이미지 크롭핑 엔진

```python
from pods.pod2_cropping import CroppingEngine
```

#### `__init__(config: Dict[str, Any] = None)`

**Parameters:**
- `config` (dict): POD 설정
  - `use_convex_hull` (bool): Convex Hull 사용 (기본: True)
  - `buffer_size` (float): 버퍼 크기 (미터)
  - `min_area` (float): 최소 면적 (㎡)

#### `process(**kwargs) -> Dict[str, Any]`

크롭핑 처리

**Parameters:**
- `image_data` (dict): POD1 출력 이미지 데이터
- `parcels` (GeoDataFrame): 필지 데이터

**Returns:**
```python
{
    'cropped_images': [
        {
            'path': str,
            'transform': Affine,
            'shape': tuple,
            'bounds': tuple
        }
    ],
    'cropped_regions': [
        {
            'id': int,
            'geometry': Polygon,
            'properties': dict
        }
    ]
}
```

#### `apply_convex_hull(geometry: Polygon) -> Polygon`

Convex Hull 적용

**Parameters:**
- `geometry` (Polygon): 입력 폴리곤

**Returns:**
- Polygon: Convex Hull 폴리곤

---

## POD3: Tiling API

### TilingEngine

이미지 타일링 엔진

```python
from pods.pod3_tiling import TilingEngine
```

#### `__init__(config: Dict[str, Any] = None)`

**Parameters:**
- `config` (dict): POD 설정
  - `tile_size` (int): 타일 크기 (픽셀)
  - `overlap` (float): 오버랩 비율 (0-1)
  - `adaptive_tiling` (bool): 적응형 타일링
  - `remove_empty` (bool): 빈 타일 제거

#### `process(cropped_data: Dict[str, Any]) -> Dict[str, Any]`

타일링 처리

**Returns:**
```python
{
    'tiles': [
        {
            'id': str,
            'path': str,
            'window': Window,
            'transform': Affine,
            'bounds': tuple
        }
    ],
    'tile_index': rtree.Index,
    'metadata': {
        'total_tiles': int,
        'tile_size': int,
        'overlap': float
    }
}
```

#### `generate_tiles(image_path: str, tile_size: int = 1024) -> List[Window]`

타일 윈도우 생성

**Parameters:**
- `image_path` (str): 이미지 경로
- `tile_size` (int): 타일 크기

**Returns:**
- List[Window]: 타일 윈도우 리스트

---

## POD4: Analysis API

### AnalysisEngine

YOLOv11 기반 AI 분석 엔진

```python
from pods.pod4_ai_analysis import AnalysisEngine
```

#### `__init__(config: Dict[str, Any] = None)`

**Parameters:**
- `config` (dict): POD 설정
  - `model_name` (str): 모델 이름
  - `confidence_threshold` (float): 신뢰도 임계값
  - `device` (str): 'cuda' 또는 'cpu'
  - `batch_size` (int): 배치 크기
  - `classes` (dict): 클래스 매핑

#### `process(tiles_data: Dict[str, Any]) -> Dict[str, Any]`

AI 분석 처리

**Returns:**
```python
{
    'detections': [
        {
            'tile_id': str,
            'class_id': int,
            'class_name': str,
            'confidence': float,
            'bbox': [x1, y1, x2, y2],
            'polygon': [[x, y], ...],
            'transform': Affine
        }
    ],
    'statistics': {
        'total_detections': int,
        'detections_per_class': dict,
        'average_confidence': float
    },
    'metadata': {
        'model': str,
        'device': str,
        'processing_time': float
    }
}
```

#### `load_model(model_path: str = None) -> YOLO`

모델 로드

**Parameters:**
- `model_path` (str): 모델 파일 경로

**Returns:**
- YOLO: 로드된 모델

#### `run_inference(images: List[np.ndarray]) -> List`

배치 추론 실행

**Parameters:**
- `images` (List[np.ndarray]): 이미지 배열 리스트

**Returns:**
- List: 추론 결과

---

## POD5: Merging API

### MergingEngine

탐지 결과 병합 엔진

```python
from pods.pod5_merging import MergingEngine
```

#### `__init__(config: Dict[str, Any] = None)`

**Parameters:**
- `config` (dict): POD 설정
  - `merge_strategy` (str): 'nms', 'union', 'overlap'
  - `iou_threshold` (float): IOU 임계값
  - `class_agnostic` (bool): 클래스 무관 병합

#### `process(detections_data: Dict[str, Any]) -> Dict[str, Any]`

병합 처리

**Returns:**
```python
{
    'merged_detections': [
        {
            'class_id': int,
            'class_name': str,
            'confidence': float,
            'bbox': [x1, y1, x2, y2],
            'polygon': [[x, y], ...],
            'area': float
        }
    ],
    'merge_statistics': {
        'original_count': int,
        'merged_count': int,
        'reduction_rate': float
    }
}
```

#### `apply_nms(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]`

NMS 적용

**Parameters:**
- `detections` (List[Dict]): 탐지 결과
- `iou_threshold` (float): IOU 임계값

**Returns:**
- List[Dict]: NMS 적용 결과

#### `calculate_iou(box1: List, box2: List) -> float`

IOU 계산

**Parameters:**
- `box1` (List): [x1, y1, x2, y2]
- `box2` (List): [x1, y1, x2, y2]

**Returns:**
- float: IOU 값

---

## POD6: GPKG Export API

### GPKGExporter

GeoPackage 발행 엔진

```python
from pods.pod6_gpkg_export import GPKGExporter
```

#### `__init__(config: Dict[str, Any] = None)`

**Parameters:**
- `config` (dict): POD 설정
  - `calculate_area` (bool): 면적 계산
  - `generate_report` (bool): 보고서 생성
  - `include_visualization` (bool): 시각화 포함

#### `process(**kwargs) -> Dict[str, Any]`

GPKG 발행 처리

**Parameters:**
- `merged_data` (dict): POD5 출력
- `parcels_data` (GeoDataFrame, optional): 필지 데이터

**Returns:**
```python
{
    'gpkg_path': str,
    'report_path': str,
    'visualization_path': str,
    'statistics': {
        'total_objects': int,
        'total_area': float,
        'class_distribution': dict
    }
}
```

#### `export_to_gpkg(gdf: GeoDataFrame, output_path: str, layer_name: str = "detections")`

GeoPackage로 내보내기

**Parameters:**
- `gdf` (GeoDataFrame): 지오데이터프레임
- `output_path` (str): 출력 경로
- `layer_name` (str): 레이어 이름

#### `generate_html_report(statistics: Dict, output_path: str = None) -> str`

HTML 보고서 생성

**Parameters:**
- `statistics` (dict): 통계 데이터
- `output_path` (str): 출력 경로

**Returns:**
- str: 보고서 파일 경로

---

## Utility APIs

### Config Manager

```python
from utils.config import ConfigManager

config = ConfigManager("config.yaml")
pod_config = config.get_pod_config("pod1")
```

#### `load_config(path: str) -> Dict`

설정 파일 로드

#### `get_pod_config(pod_name: str) -> Dict`

특정 POD 설정 반환

### Logger

```python
from utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("Processing started")
```

#### `setup_logger(name: str, level: str = "INFO") -> Logger`

로거 설정

### Coordinate Transformer

```python
from utils.coordinates import CoordinateTransformer

transformer = CoordinateTransformer("EPSG:4326", "EPSG:5186")
transformed = transformer.transform(x, y)
```

#### `transform(x: float, y: float) -> Tuple[float, float]`

좌표 변환

#### `transform_geometry(geometry: BaseGeometry) -> BaseGeometry`

지오메트리 변환

### File Utils

```python
from utils.file_utils import ensure_dir, get_file_extension

ensure_dir("output/tiles")
ext = get_file_extension("image.tif")  # ".tif"
```

#### `ensure_dir(path: str) -> Path`

디렉토리 생성 보장

#### `get_file_extension(path: str) -> str`

파일 확장자 반환

#### `list_files(directory: str, pattern: str = "*") -> List[Path]`

파일 목록 반환

---

## Error Handling

### Custom Exceptions

```python
class NongViewError(Exception):
    """Base exception class"""
    pass

class PODError(NongViewError):
    """POD-specific error"""
    pass

class ValidationError(PODError):
    """Input validation error"""
    pass

class ProcessingError(PODError):
    """Processing error"""
    pass
```

### Error Codes

| Code | Description | POD |
|------|------------|-----|
| 1001 | File not found | POD1 |
| 1002 | Invalid file format | POD1 |
| 1003 | CRS mismatch | POD1-2 |
| 2001 | Out of memory | POD3-4 |
| 2002 | GPU not available | POD4 |
| 3001 | Processing timeout | All |
| 3002 | Save failed | POD6 |

---

## Response Formats

### Success Response

```json
{
    "status": "success",
    "data": {},
    "metadata": {
        "processing_time": 123.45,
        "timestamp": "2025-11-06T12:00:00Z"
    }
}
```

### Error Response

```json
{
    "status": "error",
    "error": {
        "code": 1001,
        "message": "File not found",
        "details": "The specified file does not exist"
    },
    "metadata": {
        "timestamp": "2025-11-06T12:00:00Z"
    }
}
```

---

## Rate Limits

| Operation | Limit | Window |
|-----------|-------|--------|
| Image upload | 100MB | per request |
| Batch processing | 1000 tiles | per job |
| API calls | 100 | per minute |

---

## Authentication

현재 버전은 인증이 필요 없습니다. 향후 버전에서 JWT 기반 인증이 추가될 예정입니다.

```python
# Future implementation
headers = {
    "Authorization": "Bearer <token>"
}
```

---

## Versioning

API 버전은 URL 경로에 포함됩니다:

```
/api/v1/pipeline/run
/api/v2/pipeline/run  # Future
```

---

## Examples

### 전체 파이프라인 실행

```python
from main import NongViewPipeline

# 초기화
pipeline = NongViewPipeline("config.yaml")

# 실행
results = pipeline.run(
    input_image="data/input/orthophoto.tif",
    input_shapefile="data/input/parcels.shp",
    input_excel="data/input/pnu_data.xlsx"
)

# 결과 확인
print(f"GPKG: {results['pod6']['gpkg_path']}")
print(f"Report: {results['pod6']['report_path']}")
```

### 개별 POD 실행

```python
from pods.pod4_ai_analysis import AnalysisEngine

# AI 분석만 실행
engine = AnalysisEngine({
    'model_name': 'yolov11x-seg',
    'device': 'cuda',
    'confidence_threshold': 0.3
})

# 타일 데이터로 분석
results = engine.process(tiles_data)

# 결과 확인
for detection in results['detections']:
    print(f"Class: {detection['class_name']}, "
          f"Confidence: {detection['confidence']:.2f}")
```

### 커스텀 병합 전략

```python
from pods.pod5_merging import MergingEngine

# Union 전략으로 병합
engine = MergingEngine({
    'merge_strategy': 'union',
    'class_agnostic': False
})

merged = engine.process(detections_data)
print(f"Merged {len(detections_data['detections'])} to "
      f"{len(merged['merged_detections'])} objects")
```

---

## Support

- GitHub Issues: https://github.com/aebonlee/Nong-View2/issues
- Documentation: https://github.com/aebonlee/Nong-View2/wiki
- Email: support@nongview.com

---

API 버전: 1.0.0
최종 수정: 2025-11-06