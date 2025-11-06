# 📸 이미지 리소스

이 폴더는 Dev_md 문서에서 사용되는 이미지 파일들을 저장하는 공간입니다.

## 📁 폴더 구조

```
img/
├── architecture/      # 시스템 아키텍처 다이어그램
├── screenshots/       # 애플리케이션 스크린샷
├── logos/            # 로고 및 아이콘
├── diagrams/         # 플로우차트 및 다이어그램
└── results/          # 분석 결과 예시 이미지
```

## 🖼️ 이미지 명명 규칙

- 소문자와 하이픈(-) 사용: `system-architecture.png`
- 버전 포함 시: `pod1-flow-v2.png`
- 날짜 포함 시: `result-2025-11-06.png`

## 📝 사용 방법

### Markdown에서 이미지 참조

```markdown
![대체 텍스트](../img/폴더명/파일명.png)
```

### 예시

```markdown
![시스템 아키텍처](../img/architecture/system-overview.png)
![POD1 플로우](../img/diagrams/pod1-flow.png)
```

## 📋 이미지 형식 가이드라인

- **다이어그램**: PNG 또는 SVG
- **스크린샷**: PNG
- **로고**: SVG (벡터) 또는 PNG (투명 배경)
- **사진**: JPG

## 📏 권장 크기

- **다이어그램**: 최대 1920px 너비
- **스크린샷**: 실제 크기 또는 최대 1920px
- **로고**: 512x512px (정사각형)
- **문서 내 이미지**: 800-1200px 너비

## 🎨 이미지 최적화

업로드 전 이미지 최적화:
- PNG: [TinyPNG](https://tinypng.com/)
- JPG: [JPEGmini](https://www.jpegmini.com/)
- SVG: [SVGOMG](https://jakearchibald.github.io/svgomg/)

---

최종 업데이트: 2025-11-06