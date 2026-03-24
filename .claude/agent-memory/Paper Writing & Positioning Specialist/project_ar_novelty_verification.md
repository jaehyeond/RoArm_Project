---
name: AR-Guided Data Collection Novelty Verification
description: 2026-03-24에 수행한 AR 실시간 데모 가이드 아이디어의 신규성 검증 결과 (12개 검색어)
type: project
---

## 핵심 구분: Concept A vs Concept B

**Concept A (우리 아이디어)**: AR 오버레이로 수집 중 인간 demonstrator의 물체 배치 위치를 유도 → workspace coverage 강제. Action label은 항상 정확 (실제 환경에서 실제 동작).

**Concept B (기존 연구)**: 녹화된 데모의 이미지를 사후에 편집 (배경, 물체 텍스처 변경). GenAug, Rosie, CACTI, RoboSplat, RoCoDA = 전부 Concept B.

## 검증 결과 (12개 검색어)

**Concept B 카테고리 (새로운 주장 불가):**
- GenAug (ICRA 2023): 사후 diffusion inpainting. Threat: LOW to Concept A
- Rosie (2023): 사후 text-conditioned 증강. Threat: LOW
- CACTI (2022): 카메라+컨텍스트 증강. Threat: LOW
- RoboSplat (RSS 2025): 3DGS novel view. Threat: LOW
- 전부 사후 처리 = Action label 불일치 문제 있음. 우리 아이디어와 다름.

**AR for robot interaction (HRI 영역):**
- AR for teleoperation, task specification, programming by demo = 존재함
- 하지만 "VLA imitation learning 데이터 수집의 공간 커버리지 강제"에 초점 맞춘 논문 = 미발견
- AR2-D2: project memory에 "다른 각도"로 기록됨 → 반드시 실제 내용 확인 필요 (2026-03-10 사건 교훈)

**Sim-based domain randomization:**
- Tobin et al., ADR, Real2Render2Real = 전부 simulation 기반. 우리는 simulation 없음. Threat: LOW

## 신규성 판정

| 아이디어 | 판정 | 신뢰도 |
|--------|------|--------|
| "사후 visual augmentation 신규" | FALSE | HIGH |
| "수집 중 실시간 AR 가이드 신규" | LIKELY TRUE | MEDIUM |
| "AR for workspace coverage VLA" | LIKELY TRUE | MEDIUM |

## 필수 사전 제출 검증 (라이브 검색 미수행)

1. arXiv: "AR demonstration collection robot" (2024-2026)
2. Semantic Scholar: "mixed reality data collection robot learning"
3. AR2-D2 전문 확인 (제목, arXiv ID, contribution)
4. HRI 2025, IROS 2024 proceedings 확인

**훈련 데이터 한계**: 내 지식 컷오프 Aug 2025. 이후 논문은 직접 검색 필요.

## 권장 claim 문구

"To our knowledge, we are the first to use real-time AR overlay guidance to enforce workspace coverage during robot demonstration collection, distinct from post-hoc visual augmentation (GenAug, Rosie, CACTI) in that our approach modifies human demonstrator behavior while preserving action label validity."

**절대 사용 금지**: "First to use AR in robot learning" (HRI에 수십 년 역사 있음)

## 전체 리포트 위치
`paper/AR_NOVELTY_VERIFICATION_REPORT.md`

**Why**: 2026-03-10 사건 (4/5 거짓 갭) 재발 방지. AR 아이디어는 사후 증강과 개념적으로 다르지만, "신규"라고 주장하려면 라이브 검색 먼저.
**How to apply**: CoRL 논문 Related Work 섹션에서 "collection-time" vs "post-hoc" 구분을 명확히 서술. AR2-D2 반드시 읽고 cite 또는 differentiate.
