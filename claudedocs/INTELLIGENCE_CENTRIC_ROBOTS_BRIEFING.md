# 지능중심형 로봇 트렌드 조사 브리핑
**조사일: 2026-03-24 | 3개 전문 에이전트 병렬 조사 + 교차 검증**

---

## 1. Skild AI (스킬드 AI)

### 기본 정보 [VERIFIED]
| 항목 | 내용 |
|------|------|
| 설립 | 2023, Pittsburgh PA |
| 창업자 | **Deepak Pathak + Abhinav Gupta** (CMU 교수, 전 Meta AI) |
| 총 투자 | **~$1.7B** (Seed $14.5M → Series A $300M → Series C $1.4B) |
| 밸류에이션 | **$14B+** |
| 투자자 | SoftBank(주도), NVIDIA, Jeff Bezos, LG, Schneider Electric |
| 파트너 | **ABB Robotics, Universal Robots** |

### 핵심 기술: "Skild Brain" [대부분 CLAIMED]
- **"Omni-Bodied Intelligence"** — 하나의 모델로 이족/사족/매니퓰레이터 모두 제어
- 계층적 구조: 고수준 manipulation + 저수준 joint torque
- 4가지 데이터: 시뮬레이션(trillions), 인터넷 영상(billions), 텔레옵, 실배포
- 비즈니스: B2B SaaS (Skild Cloud API), "로봇의 AWS" 포지셔닝

### 비판적 분석 — Red Flags
1. **논문 없음**: arXiv에 기술 논문 0편. Peer review 전무. 블로그 데모만
2. **"Trillions of episodes"**: 검증 불가. 구체적 수치/벤치마크 미공개
3. **밸류에이션 괴리**: 매출 ~$30M(자사 주장) vs 밸류 $14B = P/S 470배
4. **데모 편향**: 실패 케이스, 성공률, OOD 성능 일절 미공개
5. **경쟁사 저격만**: VLM 접근을 "Potemkin village"라고 비판, 자사 비교 실험은 없음

### 판정
> 창업자 학술 배경은 세계 최상급. 투자자 신뢰도 높음. **그러나 기술 실체는 curated demo 수준이며, 독립 검증 제로.** $14B는 미래 베팅이지 현재 실적 반영 아님.

---

## 2. Physical Intelligence (π)

### 기본 정보 [VERIFIED]
| 항목 | 내용 |
|------|------|
| 설립 | 2023, San Francisco |
| CEO | **Karol Hausman** (전 Google Brain) |
| 공동창업 | **Tobias Springenberg** (전 DeepMind) |
| 어드바이저 | Sergey Levine (UC Berkeley, 직함 불확실) |
| 투자 | **$400M+** (Sequoia Capital 주도) |
| 고객 | **Weave Robotics, Ultra** (실제 상업 배포 확인) |

### 핵심 기술 [VERIFIED — 논문 있음]

| 모델 | 파라미터 | 핵심 | 상태 |
|------|---------|------|------|
| **pi0** (arXiv:2410.24164) | 3B | PaliGemma VLM + Flow Matching | RSS 2025, 부분 오픈 |
| **pi0.5** (arXiv:2504.16054) | 3B | Open-world generalization | **CoRL 2025 Oral** |
| **pi0-FAST** (arXiv:2501.09747) | 3B | 추론 10배 빠름 (토큰화) | 완전 공개, RTX 4090 가능 |
| **pi\*** (pi0.6) | ? | RL self-improvement (RECAP) | 세부 미확인 |

### 실제 성과 vs 한계

**검증된 것:**
- 68개 태스크 사전학습, 다양한 로봇 동시 제어
- 새 태스크에 50-100 데모로 fine-tuning 가능 (같은 morphology)
- **유일하게 상업 배포 중인 VLA 기업**
- "Humanoid Olympics" 11개 이벤트 수행

**한계:**
- fine-tuning 없이 새 로봇 즉시 제어 = 아직 불가능
- 일반 가정 환경 zero-shot 배포 = 미달성
- 학습 데이터 비공개 → 재현 불가능
- 4×A100 필요 → 접근성 제한

### 두 기업 비교

| 항목 | Skild AI | Physical Intelligence |
|------|----------|----------------------|
| 기술 논문 | **0편** | **4편+** (RSS, CoRL Oral) |
| 상업 배포 | 파트너십 발표만 | **실제 고객 존재** |
| 오픈소스 | 완전 비공개 | 부분 공개 (openpi) |
| 투자 규모 | $1.7B, $14B 밸류 | $400M+ |
| 기술 투명성 | 블로그만 | 논문 + 코드 + weights |
| 과장 수준 | **높음** | **보통** (CEO가 한계 인정) |

---

## 3. "지능중심형 로봇" 트렌드 — 교차 검증

### 트렌드 자체는 실재 [EVIDENCE]
- ICLR 2026 VLA 제출 164편 (전년 대비 18배)
- NVIDIA GTC 2026: ABB, FANUC과 "Physical AI" 산업 통합
- CES 2026: Samsung, LG, GE가 foundation model 로봇 발표
- 투자 총액: 2024-2025 $3-5B

### "인간수준 지능" = 거짓 [COUNTER-EVIDENCE]

**핵심 반증 5가지:**
1. 카메라 5cm 이동 → 정책 완전 실패 (직접 경험)
2. 새 물체 = 수백 개 데모 필요 (pi0도 50-100 필수)
3. 배포된 산업 로봇 99%+ = 고전적 방식
4. 투자:매출 비율 100:1+ (버블 신호)
5. 역사적 반복: Shakey→ASIMO→BD→RT-2, 매번 "범용" 약속 후 좁은 배포

### 정확한 표현
> Foundation model은 로봇의 시각/언어 이해를 개선 중이지만, "인간수준 범용 지능"은 마케팅 언어. 실제 생산 현장은 99% 고전적 방식이며, 데모와 배포 사이 격차는 크고 과소평가되고 있다.

---

## 4. 종합 판정표

| 질문 | 답변 | 확신도 |
|------|------|--------|
| Foundation model → 로봇 트렌드 실재? | **예** | HIGH |
| "인간수준 지능" 달성? | **아니오**, 마케팅 | HIGH |
| 패러다임 전환? | **아니오**, 점진적 개선 | HIGH |
| 투자 버블 (휴머노이드)? | **예** | MEDIUM |
| Skild AI 실체? | 인력 우수, 기술 불투명, 논문 0편 | HIGH |
| PI 실체? | VLA 최고 수준, 유일한 실배포 | HIGH |
