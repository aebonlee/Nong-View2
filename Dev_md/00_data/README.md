# 📂 데이터 및 리소스

이 폴더는 Nong-View2 프로젝트의 데이터, 이미지, 기타 리소스를 저장하는 공간입니다.

## 📁 저장된 파일 목록

### 🖼️ 이미지 파일
<!-- 파일 업로드 후 아래 형식으로 추가
- `pipeline-flow.png` - 6단계 POD 파이프라인 플로우 다이어그램
- `pod-architecture.png` - POD 아키텍처 구조도
- `system-overview.png` - 전체 시스템 개요
-->

### 📊 샘플 데이터
<!-- 파일 업로드 후 아래 형식으로 추가
- `sample_parcels.shp` - 샘플 필지 Shapefile
- `sample_orthophoto.tif` - 샘플 정사영상 (축소판)
- `pnu_data.xlsx` - PNU 샘플 데이터
-->

### 📄 문서 자료
<!-- 파일 업로드 후 아래 형식으로 추가
- `technical_spec.pdf` - 기술 명세서
- `api_examples.json` - API 호출 예시
-->

## 🖼️ 이미지 사용 방법

### Markdown에서 이미지 참조

```markdown
![파이프라인 플로우](../00_data/pipeline-flow.png)
![POD 아키텍처](../00_data/pod-architecture.png)
![시스템 개요](../00_data/system-overview.png)
```

### HTML에서 이미지 참조

```html
<img src="00_data/pipeline-flow.png" alt="파이프라인 플로우" width="800">
```

## 📋 지원 파일 형식

### 이미지
- **PNG**: 다이어그램, 스크린샷
- **JPG/JPEG**: 사진, 정사영상 미리보기
- **SVG**: 벡터 다이어그램, 로고

### 데이터
- **JSON**: API 응답, 설정 파일
- **CSV**: 테이블 데이터
- **XLSX**: Excel 데이터

### 지리정보
- **SHP**: Shapefile (필지 경계 등)
- **GPKG**: GeoPackage
- **GeoJSON**: 웹 호환 지리 데이터
- **TIF/TIFF**: GeoTIFF 래스터 데이터

### 문서
- **PDF**: 기술 문서, 보고서
- **MD**: Markdown 문서

## 📝 파일 추가 가이드라인

1. **파일명**: 소문자와 하이픈 사용 (`sample-data.json`)
2. **크기 제한**: 대용량 파일(>10MB)은 Git LFS 사용 권장
3. **설명 추가**: 파일 추가 시 이 README에 설명 업데이트

## 🔍 빠른 참조

| 파일 유형 | 위치 | 설명 |
|----------|------|------|
| 파이프라인 다이어그램 | `pipeline-*.png` | POD 파이프라인 플로우 |
| 샘플 데이터 | `sample_*.{shp,tif,xlsx}` | 테스트용 샘플 데이터 |
| 설정 예시 | `config_*.{json,yaml}` | 설정 파일 예시 |
| 결과 예시 | `result_*.{gpkg,html}` | 처리 결과 예시 |

---

최종 업데이트: 2025-11-06