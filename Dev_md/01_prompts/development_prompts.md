# 🤖 Nong-View2 개발 프롬프트 모음

## 1. 프로젝트 초기화 프롬프트

```
Nong-View 리포지토리를 참고하여 개선된 Nong-View2를 개발해주세요.
다음 요구사항을 충족하는 지리정보 기반 AI 분석 파이프라인을 구현하되:
1. 6단계 POD 구조로 모듈화
2. 각 POD는 독립 실행 가능
3. Python 3.10+ 타입 힌트 사용
4. 완전한 에러 처리 및 로깅
```

## 2. POD 모듈 개발 프롬프트

### POD1 - 데이터 수집
```
POD1 데이터 수집 모듈을 구현해주세요:
- ECW를 TIF로 자동 변환
- Shapefile과 Excel(PNU) 데이터 처리
- 메타데이터 추출 및 검증
- EPSG:5186 좌표계로 통일
- 파일 무결성 검사 포함
```

### POD2 - 크롭핑
```
POD2 크롭핑 모듈을 구현해주세요:
- Convex Hull 알고리즘으로 최소경계사각형 생성
- 좌표계 자동 매칭 및 변환
- 버퍼 적용 옵션 (기본 10m)
- 최소 면적 필터링 (100㎡)
- GeoJSON 형식 결과 저장
```

### POD3 - 타일링
```
POD3 타일링 모듈을 구현해주세요:
- 1024x1024 픽셀 표준 타일
- 20% 오버랩 설정
- 적응형 타일링 (이미지 크기에 따라 조정)
- R-tree 공간 인덱싱
- 빈 타일 자동 제거
```

### POD4 - AI 분석
```
POD4 AI 분석 모듈을 구현해주세요:
- YOLOv11-seg 모델 통합
- 6개 농업 클래스 분류
- 배치 처리 최적화 (기본 8)
- GPU/CPU 자동 선택
- 신뢰도 기반 필터링 (0.25)
- 결과 시각화 저장
```

### POD5 - 병합
```
POD5 병합 모듈을 구현해주세요:
- NMS/Union/Overlap 3가지 전략
- IOU 기반 중복 제거 (0.5)
- 타일 좌표를 전역 좌표로 변환
- 클래스별 그룹화
- 공간 인덱싱으로 성능 최적화
```

### POD6 - GPKG 발행
```
POD6 GPKG 발행 모듈을 구현해주세요:
- 표준 GeoPackage 형식 생성
- 다중 레이어 (parcels, detections, statistics)
- 면적 자동 계산 (㎡, ha)
- HTML 보고서 생성
- 시각화 이미지 포함
```

## 3. 통합 및 최적화 프롬프트

### 메인 파이프라인
```
모든 POD를 통합하는 메인 파이프라인을 작성해주세요:
- 순차적 POD 실행
- 선택적 POD 실행 (--skip-pods, --only-pods)
- 진행 상태 모니터링
- 결과 요약 리포트
- 에러 발생 시 graceful 종료
```

### 성능 최적화
```
다음 성능 최적화를 적용해주세요:
- 대용량 이미지 스트리밍 처리
- 메모리 효율적인 타일 처리
- GPU 배치 크기 동적 조정
- 병렬 처리 가능한 부분 식별
- 캐싱 메커니즘 구현
```

### Docker 컨테이너화
```
Docker 환경을 구성해주세요:
- Multi-stage 빌드로 이미지 크기 최적화
- GDAL 의존성 해결
- GPU 지원 설정 (nvidia-docker)
- 볼륨 마운트 설정
- 환경 변수 관리
```

## 4. 테스트 및 문서화 프롬프트

### 테스트 코드
```
포괄적인 테스트를 작성해주세요:
- 각 POD 단위 테스트
- 통합 테스트
- 에지 케이스 처리
- 성능 벤치마크
- 모의 데이터 생성
```

### 문서화
```
완전한 문서를 작성해주세요:
- README.md (설치, 사용법, 예제)
- 개발 일지 (기술적 결정사항)
- API 문서 (각 모듈 인터페이스)
- 설정 가이드 (config.yaml)
- 문제 해결 가이드
```

## 5. 고급 기능 프롬프트

### 실시간 모니터링
```
실시간 처리 모니터링 기능을 추가해주세요:
- 진행률 표시 (tqdm)
- 메모리 사용량 추적
- GPU 사용률 모니터링
- 예상 완료 시간 계산
- 웹 대시보드 (선택사항)
```

### 품질 관리
```
자동 품질 관리 시스템을 구현해주세요:
- 입력 이미지 품질 검증
- 흐림/노이즈 감지
- 좌표계 일관성 검사
- AI 결과 신뢰도 평가
- 이상치 자동 제거
```

### 확장성
```
시스템 확장성을 고려한 설계를 적용해주세요:
- 플러그인 아키텍처
- 새 AI 모델 쉽게 추가
- 커스텀 POD 지원
- REST API 엔드포인트
- 분산 처리 지원 준비
```

## 6. 배포 및 운영 프롬프트

### CI/CD 설정
```
GitHub Actions CI/CD를 설정해주세요:
- 자동 테스트 실행
- 코드 품질 검사 (pylint, black)
- Docker 이미지 자동 빌드
- 버전 태깅
- 릴리즈 노트 자동 생성
```

### 운영 가이드
```
프로덕션 운영 가이드를 작성해주세요:
- 시스템 요구사항
- 설치 체크리스트
- 모니터링 설정
- 백업 및 복구
- 트러블슈팅 가이드
```

## 7. 사용 예시 프롬프트

### 기본 사용
```
python main.py \
    --input /data/orthophoto.tif \
    --shapefile /data/parcels.shp \
    --excel /data/pnu_data.xlsx
```

### 고급 사용
```
# 특정 POD만 실행
python main.py --only-pods 3 4 5

# GPU 설정
python main.py --device cuda --batch-size 16

# 병합 전략 변경
python main.py --merge-strategy union --iou-threshold 0.3
```

### Docker 사용
```
docker run -v /local/data:/app/data nongview2 \
    python main.py --input /app/data/input.tif
```

## 8. 유용한 코드 스니펫

### 좌표계 변환
```python
def transform_crs(gdf, target_crs='EPSG:5186'):
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf
```

### 타일 생성
```python
def create_tiles(image, tile_size=1024, overlap=0.2):
    stride = int(tile_size * (1 - overlap))
    tiles = []
    for y in range(0, image.height, stride):
        for x in range(0, image.width, stride):
            window = Window(x, y, tile_size, tile_size)
            tiles.append(window)
    return tiles
```

### NMS 구현
```python
def nms(detections, iou_threshold=0.5):
    # Sort by confidence
    sorted_dets = sorted(detections, key=lambda x: x['conf'], reverse=True)
    keep = []
    while sorted_dets:
        keep.append(sorted_dets[0])
        sorted_dets = [d for d in sorted_dets[1:] 
                       if iou(d, sorted_dets[0]) < iou_threshold]
    return keep
```