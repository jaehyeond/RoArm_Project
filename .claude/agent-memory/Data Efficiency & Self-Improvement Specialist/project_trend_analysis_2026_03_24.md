---
name: VLA Data Efficiency Trend Analysis (2026-03-24)
description: 3DGS augmentation, self-improving loops, VLM judge, demo quality filtering, AR-guided collection 최신 트렌드 분석 + 우리 프로젝트 포지셔닝
type: project
---

# VLA 데이터 효율 + 증강 + 자기개선 트렌드 분석

**Date: 2026-03-24**

---

## 3DGS 데이터 증강 현황

### 확인된 논문
- **RoboSplat** (2504.13175, RSS 2025): Multi-view capture → novel views. Consumer setup에 직접 적용 어려움 (multi-view 필요)
- **SplatSim** (2409.10161, ICRA 2025): 3DGS render → real robot transfer. SmolVLA 관련 핵심 발견: SigLIP frozen이면 3DGS cosine dist ~0.1-0.2 → 전이 가능
- **Real2Render2Real** (2505.09601, CoRL 2025): 73% real-robot success from synthetic images

### 진짜 갭 (확신도 MEDIUM)
Single-view RGB-D (Azure Kinect 1대) → 3DGS → novel view augmentation → VLA policy training.
기존 연구 모두 multi-view 또는 sim-first. 단일 카메라 unstructured grasping에 직접 적용 사례 없음.

---

## Self-Improving Loop 지형도 (2026-03 기준)

### RL 기반 (simulator 필수, 대세)
- SimpleVLA-RL (ICLR 2026), CRL-VLA (2602.03445), Simple Recipe (2603.11653), RISE (2602.11075)

### 데이터 큐레이션 기반 (simulator 불필요, 드묾)
- SOAR (CoRL 2024): VLM success detection → 성공 rollout 재활용
- Reflection-Based (2510.12710): VLM reflection → 재수집 전략

### SOAR의 한계 (우리가 채울 공간)
1. 초기 성공률 >30-40% 요구 → zero capability에서 불가
2. binary success/fail만 사용 → demo 내 quality 정보 버림
3. WidowX ($2K+) 한정 검증
4. VLM judge 정확도 분석 부재

### Loop 안전 조건 (수학적)
- judge precision: π, 초기 성공률: p0
- 안전 조건: p0 × π > (1 - p0) × (1 - π)
- 실용: p0=0.6, π=0.85 → 성립 (safe)
- 위험 구간: p0 < 0.4 AND π < 0.75 → 노이즈 누적 가능

---

## VLM Judge 현황

### 사용 논문들
- SOAR: CLIP 기반 (정확도 수치 미공개)
- AutoRT: Gemini Pro 사용 (false positive rate 미공개)
- Reflection-Based: structured reflection (binary 판정 아님)

### 핵심 발견: judge precision 분석 논문 없음 (확신도 MEDIUM)
VLM judge를 사용하는 모든 논문들이 judge 정확도 체계적 분석을 생략함.
→ 우리 논문의 기여: "VLM judge precision threshold for loop stability" 실험

### 로컬 Qwen2.5-VL 3B 전략
- RTX 4090에서 실행 가능, API 불필요
- Calibration 필수: 기존 74ep 중 20ep GT 라벨 → precision/recall 측정
- 임계값: precision < 0.85이면 loop 가동 금지

---

## Demo Quality Filtering 현황

### 관련 논문
- RECAP (2511.14759, PI Nov 2025): post-hoc weighted sampling
- Data Scaling Laws (2410.18647, ICLR 2025 Oral): diversity >> quantity 정량화
- Seed2Scale (2603.08260): seed 데이터 품질 유지 전략

### 우리의 차별점
- Collection-time quality gate (vs RECAP의 post-hoc)
- FK Z depth classifier + gripper phase + static% = 이미 구현
- v1(0%) vs v3(100%) = 직접 empirical evidence

---

## AR-Guided Demo Collection

### 기존 연구 (teleoperation 중심)
- AR2-D2: 로봇 없이 가상 데모 수집 (우리와 다름)
- ARMADA (Apple): Vision Pro + virtual robot ($3K+)
- XRoboToolkit (ByteDance): XR teleoperation
- GROOT (NVIDIA): VR teleoperation for humanoid

### 우리 specific contribution (확신도 MEDIUM — "없다" 주의)
"Coverage enforcement for data diversity via AR target circles"
→ 기존 AR 연구는 teleoperation/collection 자체이고, spatial coverage를 강제하는 AR tool = 검색 범위 내 없음

---

## Critical Questions 답변

**Q1. 어떤 에피소드가 가장 학습에 기여하는가?**
→ 다양한 위치 분포를 가진 DEEP 에피소드들. 검증 방법: 상위 50%만으로 재학습 ablation

**Q2. 자율 rollout vs hand-guided 품질**
→ 자율은 다양성 낮음 (작동하는 경로 1-2개만). 혼합 비율 권장: 70% hand + 30% autonomous

**Q3. VLM false positive loop 오염 조건**
→ precision < 0.85 + 초기 성공률 < 40% 동시 성립 시 위험. Calibration 먼저 필수.

**Q4. Sponge → Cup transfer**
→ SigLIP frozen이라 visual feature는 이미 구분. Action-level negative transfer 가능 (다른 grasp angle). 독립 학습 vs 혼합 학습 ablation 권장.

---

## 포지셔닝 3축

1. Collection-time quality gating (vs post-hoc curation)
2. AR workspace coverage enforcement (vs AR teleoperation)
3. Quality × Coverage × Quantity 3원 상호작용 측정

---

## 즉시 할 것 (승인 전 설계)

**self_improve_vlm_judge.py** 설계:
- 입력: 배포 후 RGB 비디오
- Qwen2.5-VL 3B → SUCCESS/FAIL/UNCERTAIN 판정
- UNCERTAIN 제외, SUCCESS만 재학습 큐에 추가
- Calibration 먼저: 20ep GT 비교로 precision 측정
- precision < 0.85 → loop 가동 금지

**augment_quality_filter.py** 설계:
- 기존 74ep에서 가장 기여도 높은 에피소드 식별
- FK depth score + gripper phase diversity + position coverage → 종합 점수
- 상위/하위 50% 분리 → ablation 실험용 subset 생성
