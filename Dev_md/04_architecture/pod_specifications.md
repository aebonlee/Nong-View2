# 📐 POD 상세 명세서

## 목차
1. [POD 공통 인터페이스](#pod-공통-인터페이스)
2. [POD1: 데이터 수집](#pod1-데이터-수집)
3. [POD2: 크롭핑](#pod2-크롭핑)
4. [POD3: 타일링](#pod3-타일링)
5. [POD4: AI 분석](#pod4-ai-분석)
6. [POD5: 병합](#pod5-병합)
7. [POD6: GPKG 발행](#pod6-gpkg-발행)

---

## POD 공통 인터페이스

### 기본 인터페이스 정의

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

class BasePOD(ABC):
    """모든 POD가 구현해야 하는 기본 인터페이스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        self.output_dir = Path(config.get('output_dir'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """입력 데이터 검증"""
        pass
    
    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """메인 처리 로직"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """리소스 정리"""
        pass
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """표준 실행 플로우"""
        try:
            self.validate_input(**kwargs)
            result = self.process(**kwargs)
            return result
        except Exception as e:
            self.logger.error(f"POD 실행 실패: {e}")
            raise
        finally:
            self.cleanup()
```

### 공통 설정 스키마

```yaml
common:
  log_level: "INFO"
  output_format: "json"
  error_handling: "raise"  # raise, skip, retry
  max_retries: 3
  timeout: 3600  # seconds
```

---

## POD1: 데이터 수집

### 명세

| 속성 | 내용 |
|------|------|
| 이름 | IngestionEngine |
| 목적 | 다양한 형식의 입력 데이터를 표준화된 형식으로 변환 |
| 입력 | ECW/TIF 이미지, Shapefile, Excel |
| 출력 | 변환된 TIF, 통합 GeoDataFrame, 메타데이터 |

### 클래스 정의

```python
class IngestionEngine(BasePOD):
    """
    POD1: 데이터 수집 엔진
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_crs = config.get('target_crs', 'EPSG:5186')
        self.ecw_to_tif = config.get('ecw_to_tif', True)
        
    def validate_input(self, 
                      image_path: Optional[str] = None,
                      shapefile_path: Optional[str] = None,
                      excel_path: Optional[str] = None) -> bool:
        """
        입력 파일 검증
        - 파일 존재 여부
        - 형식 검증
        - 크기 제한 확인
        """
        validations = []
        
        if image_path:
            validations.append(self._validate_image(image_path))
        if shapefile_path:
            validations.append(self._validate_shapefile(shapefile_path))
        if excel_path:
            validations.append(self._validate_excel(excel_path))
            
        return all(validations)
    
    def process(self,
               image_path: Optional[str] = None,
               shapefile_path: Optional[str] = None,
               excel_path: Optional[str] = None) -> Dict[str, Any]:
        """
        데이터 처리
        1. ECW → TIF 변환
        2. Shapefile 로드 및 변환
        3. Excel 데이터 매칭
        4. 좌표계 통일
        """
        results = {
            'images': [],
            'parcels': None,
            'metadata': {}
        }
        
        # 이미지 처리
        if image_path:
            processed_image = self._process_image(image_path)
            results['images'].append(processed_image)
        
        # Shapefile 처리
        if shapefile_path:
            parcels = self._process_shapefile(shapefile_path)
            results['parcels'] = parcels
        
        # Excel 데이터 병합
        if excel_path and results['parcels'] is not None:
            results['parcels'] = self._merge_excel_data(
                results['parcels'], 
                excel_path
            )
        
        return results
```

### 메서드 상세

```python
def _process_image(self, image_path: str) -> Dict[str, Any]:
    """이미지 처리 로직"""
    
    if image_path.endswith('.ecw') and self.ecw_to_tif:
        # ECW to TIF 변환
        output_path = self._convert_ecw_to_tif(image_path)
    else:
        output_path = image_path
    
    # 메타데이터 추출
    with rasterio.open(output_path) as src:
        metadata = {
            'path': output_path,
            'crs': str(src.crs),
            'bounds': src.bounds,
            'shape': (src.height, src.width),
            'dtype': str(src.dtypes[0]),
            'transform': src.transform
        }
    
    return metadata

def _convert_ecw_to_tif(self, ecw_path: str) -> str:
    """ECW to TIF 변환"""
    
    output_path = str(self.output_dir / Path(ecw_path).stem) + '.tif'
    
    # GDAL 옵션 설정
    translate_options = gdal.TranslateOptions(
        format='GTiff',
        creationOptions=[
            'COMPRESS=LZW',
            'TILED=YES',
            'BLOCKXSIZE=512',
            'BLOCKYSIZE=512'
        ]
    )
    
    # 변환 실행
    gdal.Translate(output_path, ecw_path, options=translate_options)
    
    return output_path
```

### 설정 예시

```yaml
ingestion:
  target_crs: "EPSG:5186"
  ecw_to_tif: true
  max_file_size: "10GB"
  validation:
    check_crs: true
    check_nodata: true
  output_format: "COG"  # Cloud Optimized GeoTIFF
```

---

## POD2: 크롭핑

### 명세

| 속성 | 내용 |
|------|------|
| 이름 | CroppingEngine |
| 목적 | 관심 영역 추출 및 최적화 |
| 입력 | 정사영상, 경계 폴리곤 |
| 출력 | 크롭된 이미지 |

### 클래스 정의

```python
class CroppingEngine(BasePOD):
    """
    POD2: 크롭핑 엔진
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.use_convex_hull = config.get('use_convex_hull', True)
        self.buffer_size = config.get('buffer_size', 10)
        self.min_area = config.get('min_area', 100)
        
    def process(self, 
               image_data: Dict[str, Any],
               parcels: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        크롭핑 처리
        1. Convex Hull 생성
        2. 버퍼 적용
        3. 이미지 클리핑
        """
        
        results = {
            'cropped_images': [],
            'cropped_regions': []
        }
        
        for idx, parcel in parcels.iterrows():
            # Convex Hull 적용
            if self.use_convex_hull:
                geometry = parcel.geometry.convex_hull
            else:
                geometry = parcel.geometry
            
            # 버퍼 적용
            if self.buffer_size > 0:
                geometry = geometry.buffer(self.buffer_size)
            
            # 최소 면적 확인
            if geometry.area < self.min_area:
                continue
            
            # 이미지 크롭
            cropped = self._crop_image(image_data, geometry)
            
            results['cropped_images'].append(cropped)
            results['cropped_regions'].append({
                'id': idx,
                'geometry': geometry,
                'properties': parcel.to_dict()
            })
        
        return results
```

### 크롭핑 알고리즘

```python
def _crop_image(self, 
               image_data: Dict[str, Any], 
               geometry: Polygon) -> Dict[str, Any]:
    """이미지 크롭핑 구현"""
    
    with rasterio.open(image_data['path']) as src:
        # 마스크 생성
        out_image, out_transform = mask(
            src, 
            [geometry], 
            crop=True,
            all_touched=True
        )
        
        # 메타데이터 업데이트
        out_meta = src.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform,
            'compress': 'lzw'
        })
        
        # 저장
        output_path = self._generate_output_path()
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(out_image)
        
        return {
            'path': output_path,
            'transform': out_transform,
            'shape': out_image.shape,
            'bounds': geometry.bounds
        }
```

---

## POD3: 타일링

### 명세

| 속성 | 내용 |
|------|------|
| 이름 | TilingEngine |
| 목적 | 이미지를 균일한 타일로 분할 |
| 입력 | 크롭된 이미지 |
| 출력 | 타일 이미지, 타일 인덱스 |

### 클래스 정의

```python
class TilingEngine(BasePOD):
    """
    POD3: 타일링 엔진
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tile_size = config.get('tile_size', 1024)
        self.overlap = config.get('overlap', 0.2)
        self.adaptive_tiling = config.get('adaptive_tiling', True)
        self.remove_empty = config.get('remove_empty', True)
        
    def process(self, cropped_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        타일링 처리
        1. 타일 그리드 생성
        2. 타일 추출
        3. 공간 인덱스 생성
        """
        
        results = {
            'tiles': [],
            'tile_index': self._create_spatial_index()
        }
        
        for image_data in cropped_data['cropped_images']:
            tiles = self._generate_tiles(image_data)
            
            for tile in tiles:
                if self.remove_empty and self._is_empty_tile(tile):
                    continue
                    
                saved_tile = self._save_tile(tile)
                results['tiles'].append(saved_tile)
                
                # 공간 인덱스 업데이트
                self._update_spatial_index(
                    results['tile_index'], 
                    saved_tile
                )
        
        return results
```

### 타일 생성 알고리즘

```python
def _generate_tiles(self, image_data: Dict[str, Any]) -> List[Dict]:
    """적응형 타일 생성"""
    
    with rasterio.open(image_data['path']) as src:
        # 타일 크기 결정
        if self.adaptive_tiling:
            tile_size = self._calculate_optimal_tile_size(
                src.width, 
                src.height
            )
        else:
            tile_size = self.tile_size
        
        # 스트라이드 계산
        stride = int(tile_size * (1 - self.overlap))
        
        tiles = []
        for y in range(0, src.height, stride):
            for x in range(0, src.width, stride):
                # 윈도우 생성
                window = Window(
                    x, y,
                    min(tile_size, src.width - x),
                    min(tile_size, src.height - y)
                )
                
                # 타일 데이터 읽기
                tile_data = src.read(window=window)
                
                # 타일 정보 저장
                tiles.append({
                    'data': tile_data,
                    'window': window,
                    'transform': src.window_transform(window),
                    'bounds': src.window_bounds(window)
                })
        
        return tiles

def _calculate_optimal_tile_size(self, width: int, height: int) -> int:
    """최적 타일 크기 계산"""
    
    # 이미지 크기에 따른 적응형 타일링
    min_dimension = min(width, height)
    
    if min_dimension < 512:
        return 256
    elif min_dimension < 2048:
        return 512
    elif min_dimension < 4096:
        return 1024
    else:
        return 2048
```

### 공간 인덱스 구현

```python
def _create_spatial_index(self) -> index.Index:
    """R-tree 공간 인덱스 생성"""
    
    p = index.Property()
    p.dimension = 2
    p.buffering_capacity = 10
    p.dat_extension = 'dat'
    p.idx_extension = 'idx'
    
    return index.Index(properties=p)

def _update_spatial_index(self, 
                         idx: index.Index, 
                         tile: Dict[str, Any]) -> None:
    """공간 인덱스 업데이트"""
    
    # 타일 경계 추가
    idx.insert(
        id=tile['id'],
        coordinates=tile['bounds']
    )
```

---

## POD4: AI 분석

### 명세

| 속성 | 내용 |
|------|------|
| 이름 | AnalysisEngine |
| 목적 | YOLOv11 기반 객체 탐지 및 세그멘테이션 |
| 입력 | 타일 이미지 |
| 출력 | 탐지 결과 (JSON) |

### 클래스 정의

```python
class AnalysisEngine(BasePOD):
    """
    POD4: AI 분석 엔진
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'yolov11x-seg')
        self.confidence_threshold = config.get('confidence_threshold', 0.25)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.get('batch_size', 8)
        self.classes = config.get('classes', self._default_classes())
        
        # 모델 로드
        self.model = self._load_model()
        
    def _default_classes(self) -> Dict[int, str]:
        """기본 클래스 정의"""
        return {
            0: "생육기_사료작물",
            1: "생산기_사료작물",
            2: "곤포_사일리지",
            3: "비닐하우스_단동",
            4: "비닐하우스_연동",
            5: "경작지_드론",
            6: "경작지_위성"
        }
    
    def process(self, tiles_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI 분석 처리
        1. 배치 구성
        2. 모델 추론
        3. 후처리
        """
        
        results = {
            'detections': [],
            'statistics': {},
            'metadata': {}
        }
        
        # 배치 처리
        for batch in self._create_batches(tiles_data['tiles']):
            predictions = self._run_inference(batch)
            processed = self._postprocess(predictions, batch)
            results['detections'].extend(processed)
        
        # 통계 계산
        results['statistics'] = self._calculate_statistics(
            results['detections']
        )
        
        return results
```

### 추론 구현

```python
def _run_inference(self, batch: List[Dict]) -> List:
    """배치 추론 실행"""
    
    # 이미지 준비
    images = [tile['data'] for tile in batch]
    
    # 추론
    with torch.no_grad():
        predictions = self.model(
            images,
            conf=self.confidence_threshold,
            device=self.device
        )
    
    return predictions

def _postprocess(self, 
                predictions: List, 
                batch: List[Dict]) -> List[Dict]:
    """추론 결과 후처리"""
    
    processed = []
    
    for pred, tile in zip(predictions, batch):
        for detection in pred:
            processed.append({
                'tile_id': tile['id'],
                'class_id': int(detection.cls),
                'class_name': self.classes[int(detection.cls)],
                'confidence': float(detection.conf),
                'bbox': detection.xyxy.tolist(),
                'polygon': detection.masks.xy[0].tolist() if detection.masks else None,
                'transform': tile['transform']
            })
    
    return processed
```

### GPU 최적화

```python
def _optimize_batch_size(self) -> int:
    """GPU 메모리에 따른 배치 크기 최적화"""
    
    if self.device == 'cpu':
        return 1
    
    # GPU 메모리 확인
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = gpu_memory - torch.cuda.memory_allocated()
    
    # 이미지 크기와 모델 크기 고려
    image_size = self.tile_size * self.tile_size * 3 * 4  # RGB float32
    model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
    
    # 보수적인 배치 크기 계산
    optimal_batch = int(available_memory * 0.7 / (image_size + model_size))
    
    return min(max(1, optimal_batch), 32)
```

---

## POD5: 병합

### 명세

| 속성 | 내용 |
|------|------|
| 이름 | MergingEngine |
| 목적 | 타일별 탐지 결과 통합 |
| 입력 | 개별 타일 탐지 결과 |
| 출력 | 통합된 탐지 결과 |

### 클래스 정의

```python
class MergingEngine(BasePOD):
    """
    POD5: 병합 엔진
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.merge_strategy = config.get('merge_strategy', 'nms')
        self.iou_threshold = config.get('iou_threshold', 0.5)
        self.class_agnostic = config.get('class_agnostic', False)
        
    def process(self, detections_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        병합 처리
        1. 좌표 변환 (타일 → 전역)
        2. 중복 제거
        3. 결과 통합
        """
        
        # 전역 좌표로 변환
        global_detections = self._convert_to_global_coords(
            detections_data['detections']
        )
        
        # 병합 전략 적용
        if self.merge_strategy == 'nms':
            merged = self._apply_nms(global_detections)
        elif self.merge_strategy == 'union':
            merged = self._apply_union(global_detections)
        elif self.merge_strategy == 'overlap':
            merged = self._apply_overlap(global_detections)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")
        
        return {
            'merged_detections': merged,
            'merge_statistics': self._calculate_merge_stats(
                len(global_detections), 
                len(merged)
            )
        }
```

### NMS 구현

```python
def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
    """Non-Maximum Suppression 적용"""
    
    if self.class_agnostic:
        return self._nms(detections)
    else:
        # 클래스별 NMS
        merged = []
        for class_id in set(d['class_id'] for d in detections):
            class_dets = [d for d in detections if d['class_id'] == class_id]
            merged.extend(self._nms(class_dets))
        return merged

def _nms(self, detections: List[Dict]) -> List[Dict]:
    """NMS 알고리즘 구현"""
    
    if not detections:
        return []
    
    # 신뢰도 순 정렬
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while sorted_dets:
        # 가장 높은 신뢰도 선택
        best = sorted_dets.pop(0)
        keep.append(best)
        
        # IOU 계산 및 필터링
        sorted_dets = [
            d for d in sorted_dets 
            if self._calculate_iou(best['bbox'], d['bbox']) < self.iou_threshold
        ]
    
    return keep
```

### 좌표 변환

```python
def _convert_to_global_coords(self, detections: List[Dict]) -> List[Dict]:
    """타일 좌표를 전역 좌표로 변환"""
    
    global_detections = []
    
    for det in detections:
        # 변환 행렬 적용
        transform = det['transform']
        
        # 바운딩 박스 변환
        global_bbox = self._transform_bbox(det['bbox'], transform)
        
        # 폴리곤 변환 (있는 경우)
        global_polygon = None
        if det.get('polygon'):
            global_polygon = self._transform_polygon(det['polygon'], transform)
        
        global_det = det.copy()
        global_det['bbox'] = global_bbox
        global_det['polygon'] = global_polygon
        global_detections.append(global_det)
    
    return global_detections
```

---

## POD6: GPKG 발행

### 명세

| 속성 | 내용 |
|------|------|
| 이름 | GPKGExporter |
| 목적 | GeoPackage 형식으로 결과 발행 |
| 입력 | 병합된 탐지 결과 |
| 출력 | GPKG 파일, HTML 보고서 |

### 클래스 정의

```python
class GPKGExporter(BasePOD):
    """
    POD6: GPKG 발행 엔진
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.calculate_area = config.get('calculate_area', True)
        self.generate_report = config.get('generate_report', True)
        self.include_visualization = config.get('include_visualization', True)
        
    def process(self, 
               merged_data: Dict[str, Any],
               parcels_data: Optional[gpd.GeoDataFrame] = None) -> Dict[str, Any]:
        """
        GPKG 발행 처리
        1. 레이어 구성
        2. 통계 계산
        3. 보고서 생성
        """
        
        # GeoDataFrame 생성
        detections_gdf = self._create_detections_gdf(
            merged_data['merged_detections']
        )
        
        # 면적 계산
        if self.calculate_area:
            detections_gdf['area_sqm'] = detections_gdf.geometry.area
            detections_gdf['area_ha'] = detections_gdf['area_sqm'] / 10000
        
        # GPKG 파일 생성
        gpkg_path = self._export_to_gpkg(detections_gdf, parcels_data)
        
        # 보고서 생성
        report_path = None
        if self.generate_report:
            report_path = self._generate_html_report(
                detections_gdf, 
                parcels_data
            )
        
        # 시각화
        viz_path = None
        if self.include_visualization:
            viz_path = self._create_visualization(detections_gdf)
        
        return {
            'gpkg_path': gpkg_path,
            'report_path': report_path,
            'visualization_path': viz_path,
            'statistics': self._calculate_final_statistics(detections_gdf)
        }
```

### GPKG 생성

```python
def _export_to_gpkg(self, 
                   detections_gdf: gpd.GeoDataFrame,
                   parcels_gdf: Optional[gpd.GeoDataFrame]) -> str:
    """GeoPackage 파일 생성"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gpkg_path = self.output_dir / f"nongview_results_{timestamp}.gpkg"
    
    # 탐지 결과 레이어
    detections_gdf.to_file(
        gpkg_path,
        layer='detections',
        driver='GPKG'
    )
    
    # 필지 레이어 (있는 경우)
    if parcels_gdf is not None:
        parcels_gdf.to_file(
            gpkg_path,
            layer='parcels',
            driver='GPKG'
        )
        
        # 필지별 클립된 결과
        clipped = self._clip_by_parcels(detections_gdf, parcels_gdf)
        clipped.to_file(
            gpkg_path,
            layer='clipped_detections',
            driver='GPKG'
        )
    
    # 통계 테이블
    stats_df = self._create_statistics_table(detections_gdf)
    stats_df.to_file(
        gpkg_path,
        layer='statistics',
        driver='GPKG'
    )
    
    return str(gpkg_path)
```

### HTML 보고서 생성

```python
def _generate_html_report(self,
                         detections_gdf: gpd.GeoDataFrame,
                         parcels_gdf: Optional[gpd.GeoDataFrame]) -> str:
    """HTML 보고서 생성"""
    
    from jinja2 import Template
    
    template = Template('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nong-View2 분석 보고서</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; }
            th { background-color: #4CAF50; color: white; }
            .chart { width: 100%; height: 400px; }
        </style>
    </head>
    <body>
        <h1>Nong-View2 분석 보고서</h1>
        <h2>처리 정보</h2>
        <p>처리 시간: {{ timestamp }}</p>
        <p>총 탐지 개수: {{ total_detections }}</p>
        
        <h2>클래스별 통계</h2>
        <table>
            <tr>
                <th>클래스</th>
                <th>개수</th>
                <th>총 면적 (㎡)</th>
                <th>평균 신뢰도</th>
            </tr>
            {% for row in class_stats %}
            <tr>
                <td>{{ row.class_name }}</td>
                <td>{{ row.count }}</td>
                <td>{{ "%.2f"|format(row.total_area) }}</td>
                <td>{{ "%.3f"|format(row.avg_confidence) }}</td>
            </tr>
            {% endfor %}
        </table>
        
        {% if parcel_stats %}
        <h2>필지별 통계</h2>
        <table>
            <tr>
                <th>PNU</th>
                <th>탐지 개수</th>
                <th>주요 클래스</th>
            </tr>
            {% for row in parcel_stats %}
            <tr>
                <td>{{ row.pnu }}</td>
                <td>{{ row.detection_count }}</td>
                <td>{{ row.main_class }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        <h2>처리 메타데이터</h2>
        <pre>{{ metadata | tojson(indent=2) }}</pre>
    </body>
    </html>
    ''')
    
    # 통계 계산
    class_stats = detections_gdf.groupby('class_name').agg({
        'class_name': 'count',
        'area_sqm': 'sum',
        'confidence': 'mean'
    }).rename(columns={
        'class_name': 'count',
        'area_sqm': 'total_area',
        'confidence': 'avg_confidence'
    }).reset_index()
    
    # HTML 생성
    html_content = template.render(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        total_detections=len(detections_gdf),
        class_stats=class_stats.to_dict('records'),
        parcel_stats=self._calculate_parcel_stats(detections_gdf, parcels_gdf),
        metadata=self.config
    )
    
    # 파일 저장
    report_path = self.output_dir / 'analysis_report.html'
    report_path.write_text(html_content, encoding='utf-8')
    
    return str(report_path)
```

---

## 성능 지표

### 각 POD별 처리 시간

| POD | 처리량 | 평균 시간 | 메모리 사용 |
|-----|--------|-----------|------------|
| POD1 | 1GB 이미지 | 30s | 2GB |
| POD2 | 100 폴리곤 | 15s | 1GB |
| POD3 | 1000 타일 | 20s | 500MB |
| POD4 | 100 타일/분 | - | 4GB (GPU) |
| POD5 | 10000 객체 | 5s | 1GB |
| POD6 | 5000 객체 | 20s | 500MB |

---

## 에러 처리

### 공통 에러 코드

```python
class PODError(Exception):
    """POD 에러 기본 클래스"""
    
    ERROR_CODES = {
        1001: "입력 파일 없음",
        1002: "잘못된 파일 형식",
        1003: "좌표계 불일치",
        2001: "메모리 부족",
        2002: "GPU 사용 불가",
        3001: "처리 타임아웃",
        3002: "결과 저장 실패"
    }
```

### 에러 처리 전략

```python
def handle_error(self, error: Exception) -> None:
    """에러 처리 전략"""
    
    if self.config['error_handling'] == 'raise':
        raise error
    elif self.config['error_handling'] == 'skip':
        self.logger.warning(f"에러 발생, 건너뛰기: {error}")
        return None
    elif self.config['error_handling'] == 'retry':
        for attempt in range(self.config['max_retries']):
            try:
                # 재시도 로직
                pass
            except Exception as e:
                if attempt == self.config['max_retries'] - 1:
                    raise
                time.sleep(2 ** attempt)  # 지수 백오프
```

---

문서 버전: 1.0.0
최종 수정: 2024-11-06