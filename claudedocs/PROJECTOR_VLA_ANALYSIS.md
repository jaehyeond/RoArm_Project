# Projector-VLA 비판적 검증 보고서

> 분석 일자: 2026-03-19
> 상태: **검토 후 기각** — 메인 연구 방향으로 부적합, 보조 도구로는 가능
> 관련: ResearchPlan.md (메인 방향: Cross-Embodiment Bimanual VLA)

---

## 1. 제안 개요

### 1.1 핵심 아이디어

빔 프로젝터가 테이블에 시각적 가이드(선, 점, 화살표)를 투사 → 로봇 카메라가 이를 인식 → VLA 모델이 "물리적 시각 프롬프트"로 해석하여 학습/추론 → 수동 kinesthetic teaching(수백 회) 대신 프로젝터 가이드(수십 회)로 데이터 효율 향상.

### 1.2 제안된 시스템 구성

| 구성 요소 | 역할 |
|-----------|------|
| Unity | Digital Twin / Manager — 최적 경로 계산, 프로젝터 이미지 전달 |
| Isaac Sim/Lab | 가상 환경에서 "빔 따라가기" 사전 학습 (Sim-to-Real) |
| SmolVLA (450M) | Robot Brain — 카메라 영상 보고 관절 각도 결정 |
| Projector Beam | Physical Prompt — 테이블에 선/점 투사 |
| RoArm-M3-Pro | Physical Actor — 1대→3대 확장 계획 |
| Azure Kinect | Global Observer — 3인칭 시점 (3대) |
| ZED Mini | Local Observer — 손목 장착 1인칭 시점 (미보유, 추가 구매 필요) |

### 1.3 제안된 실험 설계

| 비교군 | 입력 데이터 | 기대 결과 |
|--------|-----------|----------|
| Baseline | RGB Image + Text 명령 | 표준 성능 |
| Numeric Prompt | RGB + (x, y) 좌표값 입력 | 정확도 향상 |
| Projected Prompt | RGB (프로젝션 포함) + Text | 성공률 급증 (주장) |

---

## 2. 팩트 체크 — 제안서 내 주장 검증

| 제안서 주장 | 판정 | 근거 |
|------------|------|------|
| "VAP 논문이 Visual Prompting 효과 입증" | **사실이지만 맥락 다름** | VAP(arXiv 2512.20014, 2025-12)은 **디지털** 오버레이(소프트웨어). 물리적 프로젝션과 근본적으로 다름 |
| "SmolVLA의 SigLIP 인코더가 Multi-view 지원" | **사실** | 카메라 수 제한 없음, shared SigLIP encoder + token concatenation |
| "SmolVLA가 프로젝터 빛을 힌트로 인지 가능" | **미검증, 고위험** | SigLIP은 인터넷 자연 이미지로 학습. 인공 투사광 = OOD 입력. 작동 여부 불명 |
| "프로젝터 가이드로 데이터 효율 향상" | **미검증** | 디지털 오버레이(AimBot 등)로는 입증. 물리적 투사로는 아무도 안 해봄 |
| "이 연구가 새롭다" | **부분적 사실** | 물리 프로젝터 → VLA 카메라 → 정책 해석 = 0편. BUT 디지털 버전은 이미 top-tier 포화 |

---

## 3. 경쟁 연구 분석 — Visual Prompting for Robot Policy (2024-2026)

**핵심 발견**: "Visual Prompting for Robot"은 2024-2026 핫토픽. 물리적 프로젝션만 안 했을 뿐, 디지털 방식은 이미 포화.

### 3.1 직접 경쟁자 (디지털 Visual Prompting)

| 논문 | arXiv ID | 학회 | 방법 | 위협 수준 |
|------|----------|------|------|----------|
| **AimBot** | — | **CoRL 2025** | 조준선/사격선 디지털 오버레이 → visuomotor policy | **CRITICAL** — 가장 직접적 경쟁자, 같은 개념의 디지털 버전 |
| **TraceVLA** | — | **ICLR 2025** | 궤적 trace 디지털 오버레이 → VLA | HIGH — VLA + visual prompt, 이미 top-tier 채택 |
| **CoA-VLA** | — | **ICCV 2025** | 좌표/바운딩박스 디지털 오버레이 → Chain-of-Affordance | HIGH |
| **RoVI** | — | **CVPR 2025** | 화살표/원/숫자 디지털 오버레이, 87.5% 성공률 | HIGH — 가장 유사한 시각 어휘 |
| **MOKA** | 2403.03174 | **RSS 2024** | 마크 기반 visual prompting, VLM keypoint 선택 | MEDIUM |
| **PIVOT** | 2402.07872 | 2024-02 | 후보 행동(화살표, 원) 이미지 오버레이 → VLM spatial reasoning | MEDIUM |
| **VAP** | 2512.20014 | 2025-12 | 개인 물체 하이라이트 + 텍스트 재작성, training-free | MEDIUM |
| **KUDA** | 2503.10546 | 2025-03 | Keypoints + visual prompting → open-vocabulary manipulation | MEDIUM |
| **Visual Prompt + ACT** | 2508.08748 | 2025-08 | Annotation-guided pick-and-place | LOW |

### 3.2 물리 프로젝터 + 로봇 (기존 연구)

| 연구 | 출처 | 용도 | 차이점 |
|------|------|------|--------|
| "Better Teaming Through Visual Cues" | ASU, CSCW journal | 프로젝터가 **사람에게** 로봇 의도 전달 | 사람용, 로봇 학습 아님 |
| "Projecting Robot Intentions" | Sonawani et al., IROS 2023 | 로봇 네비게이션 의도를 바닥에 투사 | 사람 이해용, 정책 학습 아님 |
| ProjecTA | arXiv 2601.11328, 2026-01 | 로봇 교육 어시스턴트, 가이드 투어 | HCI, manipulation 아님 |
| LightGuide (상용) | lightguidesys.com | 산업 AR — 작업자에게 조립 단계 투사 | 사람용 산업 도구 |

**결론**: 물리 프로젝터를 로봇 정책 학습에 사용한 논문 = **0편** (genuinely novel).
BUT: 기존 연구는 모두 "프로젝터 → 사람에게 보여주기"로만 사용. "프로젝터 → 로봇 카메라 → 정책 해석"은 아무도 안 한 이유가 있을 수 있음.

---

## 4. 치명적 기술 리스크 (3가지)

### Risk 1: SigLIP + 투사광 = OOD 입력 (심각도: CRITICAL)

**문제**: SmolVLA의 SigLIP 인코더는 인터넷 자연 이미지로 학습됨. 프로젝터가 테이블에 쏘는 인공 빛 패턴은 학습 분포(training distribution)에 없는 입력.

**근거**:
- SigLIP은 contrastive vision-language encoder, 웹 이미지-텍스트 쌍으로 학습
- "Physically-based Lighting Augmentation for Robotic Manipulation" (arXiv 2508.01442, 2025)에 따르면, CLIP/SigLIP 계열 인코더를 사용하는 imitation learning 정책은 **조명 변화에 취약**
- "Holistic Evaluation of Robustness in CLIP Models" (arXiv 2410.01534)도 시각 분포 변화(조명 포함)에 민감성 문서화
- SigLIP 2 (arXiv 2502.14786, 2025-02)도 자연 이미지 분포 중심

**결과**: SigLIP이 투사된 화살표/원을 "의미 있는 시각 특징"으로 인코딩할지, "조명 변화/노이즈"로 무시할지는 **완전히 열린 실험 질문**. 실패하면 전체 연구 방향 폐기.

**검증 방법 (1일짜리 go/no-go 실험)**:
```
1. 프로젝터로 테이블에 빨간 원/화살표 투사
2. Azure Kinect로 촬영 (프로젝션 있음 vs 없음)
3. SmolVLA에 두 이미지를 넣어서 SigLIP attention map 비교
4. 프로젝션 영역에 attention이 집중되면 → GO
5. 무시하면 → NO-GO → 연구 방향 폐기
```

### Risk 2: "왜 물리적이어야 하나?" (심각도: HIGH)

Reviewer가 반드시 물을 핵심 질문:

> "AimBot(CoRL 2025)이 이미 디지털로 같은 걸 했는데, 굳이 물리적 프로젝터를 쓸 이유가 뭔가? 프로젝터 캘리브레이션, 조명 간섭, 가림 문제를 떠안으면서까지?"

**디지털 vs 물리 비교**:

| 기준 | 디지털 오버레이 (AimBot) | 물리 프로젝터 (Projector-VLA) |
|------|----------------------|---------------------------|
| 캘리브레이션 | 카메라 외부 파라미터만 | 프로젝터-카메라-테이블 전부 |
| 가림(Occlusion) | 없음 (소프트웨어) | 로봇팔이 빔 가림 |
| 조명 간섭 | 없음 | 주변 조명에 따라 투사 가시성 변동 |
| 정밀도 | 픽셀 단위 | 프로젝터 해상도 + 투사 거리 의존 |
| 재현성 | 완벽 | 프로젝터 위치/밝기/표면 반사에 의존 |
| 모델 수정 | 필요 (오버레이 파이프라인) | 불필요 (카메라가 그대로 봄) |
| Camera-agnostic | 아님 (카메라별 재캘리브) | 어떤 카메라든 빔이 보임 |

**유일한 물리적 프로젝션의 잠재 장점**:
- "Camera-agnostic": 어떤 카메라를 달아도 프로젝션이 보임 → BUT 프로젝터 자체가 캘리브레이션 필요하므로 장점이 약함
- "Non-expert teaching": 비전문가가 빔으로 가리키기만 하면 됨 → HRI 각도라면 유효. BUT 이러면 CoRL이 아니라 CHI/HRI 학회 대상

### Risk 3: 프로젝터 가림(Occlusion) (심각도: HIGH)

**문제**: 로봇팔이 프로젝터 빛 경로를 물리적으로 가림.

- 팔이 위에 있을 때 → 테이블의 투사 패턴에 팔 그림자
- 정확히 grasp moment (가장 중요한 순간)에 → 힌트가 사라짐
- 3대 로봇이면 → 가림 확률 3배
- 제안서의 대책 "다른 카메라 시점으로 전환" → 그 카메라에서도 프로젝션이 가려질 수 있음

**SmolVLA에 cross-view attention이 있는가?**
- **없음.** VLM self-attention에 의한 암시적 혼합뿐.
- 가림 인식 + 뷰 전환 = 별도 모듈 직접 개발 필요.

---

## 5. 제안서의 누락 항목

| 누락 항목 | 문제 |
|-----------|------|
| **ZED Mini** | "로봇 손목 장착"이라 했는데 현재 미보유. 추가 구매 $449 |
| **Unity 필요성** | Isaac Lab이 있는데 Unity가 왜 필요? 경로 계산 + 프로젝터 이미지 전달은 Python 스크립트로도 가능 |
| **Isaac Lab 가상 프로젝터** | Isaac Lab에 프로젝터 시뮬레이션이 기본 포함되어 있는지 불명. 직접 구현 필요할 수 있음 |
| **프로젝터 해상도** | 일반 빔프로젝터(1080p)가 테이블 30cm 영역에 투사 시 → 픽셀당 ~0.5mm. 충분한가? |
| **비교 실험 공정성** | Projected Prompt 조건은 위치 힌트라는 추가 정보를 갖고 있음 → "당연히 잘 되는 거 아닌가?" |
| **AimBot과의 비교** | 가장 직접적인 경쟁자인데 비교 대상에 없음 |
| **프로젝터 하드웨어 스펙** | 어떤 프로젝터? 루멘? 해상도? throw ratio? |

---

## 6. 물리 프로젝터만의 진짜 장점

냉정하게 보면 **하나 존재**:

> **Human-in-the-loop teaching**: 사람이 프로젝터 리모컨/포인터로 "여기!"라고 빔을 쏘면, 로봇이 그걸 보고 즉시 반응. 디지털 오버레이는 사람이 직접 제어하려면 코딩 필요. 프로젝터는 비전문가도 빔으로 가리키기만 하면 됨.

이 각도로 가면:
- "Non-expert Robot Teaching via Physical Visual Prompts" = HRI 논문
- 프로그래밍 없이 빔으로 가리키기 → 로봇이 학습
- **CHI 2027 / HRI 2027**에 맞는 프레이밍
- BUT: 이건 원래 제안의 방향(데이터 효율 향상)과 다름

---

## 7. 멀티 로봇 확장 평가

| 주장 | 판정 |
|------|------|
| "3대가 빔 가이드를 공유하며 협업" | **과도한 복잡도** — 1대에서 먼저 작동해야 함 |
| "부딪히지 않고 협업" | 충돌 회피 + 프로젝션 가림 + 3대 동기화 = 3개 연구 주제 동시 해결 필요 |
| "가려지면 다른 카메라로 전환" | SmolVLA에 이 기능 없음, 직접 구현 필요 |

**판정**: 멀티 로봇은 **별도 논문 수준**. 1대로 먼저 검증 안 되면 3대는 무의미.

---

## 8. Bimanual VLA (ResearchPlan.md) vs Projector-VLA 비교

| 기준 | Bimanual VLA (현재 메인) | Projector-VLA (이 제안) |
|------|-------------------------|----------------------|
| **Novelty** | SmolVLA bimanual = 0편 (확실) | 물리 프로젝터+VLA = 0편 (확실), BUT 디지털 버전 포화 |
| **기술적 리스크** | 데이터 수집 어려움 (해결 가능) | SigLIP이 투사광 인식할지 모름 (**go/no-go 바이너리**) |
| **선행 연구 포지셔닝** | ALOHA(no VLA) vs pi0(closed) = 깔끔한 빈 슬롯 | AimBot/TraceVLA(디지털)와 구분 어려움 |
| **필요 장비** | SO-101 2대 ($460-600) | 프로젝터 + ZED Mini + Unity = 추가 비용+복잡도 |
| **실패 시 백업** | SO-101만으로도 논문 가능 | SigLIP이 빔 안 보면 → **전체 폐기** |
| **교수님 임팩트** | "2팔이 같이 움직인다" (시각적) | "프로젝터가 길을 알려준다" (시각적이긴 하지만) |
| **학회 fit** | CoRL (robot learning) 정중앙 | HRI/CHI (interaction) 쪽이 더 맞음 |
| **기존 경험 활용** | SmolVLA 배포 경험 직결 | 프로젝터/Unity 경험 = 0에서 시작 |
| **리스크 유형** | 점진적 (데이터 더 모으면 해결) | 바이너리 (되거나 안 되거나) |

---

## 9. 최종 판정

### 9.1 메인 연구 방향으로: **기각**

3가지 이유:

1. **Go/No-Go 바이너리 리스크**: SigLIP이 투사광을 의미 있게 인코딩하는지 여부가 전체 연구의 전제조건. 실패하면 전부 폐기. Bimanual은 이런 binary risk가 없음 (12-DOF은 아키텍처가 이미 지원).

2. **디지털 경쟁자가 너무 강함**: AimBot(CoRL 2025), TraceVLA(ICLR 2025), RoVI(CVPR 2025)가 이미 top-tier acceptance. "왜 물리적이어야 하나?"에 대한 답이 약하면 reviewer가 reject할 것.

3. **기존 자산 활용도 최저**: 3개월간 쌓은 SmolVLA 배포 경험, 데이터 품질 방법론, cross-embodiment 전이 증명이 Bimanual에선 직결되지만, Projector-VLA에선 거의 쓸모없음. 프로젝터/Unity/캘리브레이션은 0에서 시작.

### 9.2 보조 도구로: **가능**

Bimanual VLA 메인 연구에서 프로젝터를 **데이터 수집 보조 도구**로 활용 가능:
- 프로젝터로 grasp 위치를 테이블에 표시 → 사람이 hand-guiding할 때 가이드로 활용
- 에피소드 품질 향상 (일관된 grasp 위치)
- 학습/추론 시 프로젝터 의존성 없음 → 리스크 없음
- 논문에서 "데이터 수집 방법론"으로 한 줄 언급 가능

### 9.3 별도 후속 연구로: **조건부 가능**

Bimanual 논문 완료 후, HRI/CHI 방향으로 별도 연구:
- "Non-Expert Robot Teaching via Physical Visual Prompts"
- 전제: SigLIP go/no-go 실험 통과
- 학회: CHI 2027 / HRI 2027
- 기간: 3-4개월 (bimanual 논문 후)

---

## 10. 참고 문헌

### 10.1 Visual Prompting for Robotics

- VAP: Lee et al., "Bring My Cup! Personalizing VLAs with Visual Attentive Prompting", arXiv 2512.20014, 2025-12
- PIVOT: Nasiriany et al., "PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs", arXiv 2402.07872, 2024-02
- MOKA: Liu et al., "MOKA: Open-Vocabulary Robotic Manipulation through Mark-Based Visual Prompting", arXiv 2403.03174, 2024-03 (RSS 2024)
- KUDA: arXiv 2503.10546, 2025-03
- RoVI: CVPR 2025
- AimBot: CoRL 2025
- TraceVLA: ICLR 2025
- CoA-VLA: ICCV 2025
- Visual Prompt + ACT: arXiv 2508.08748, 2025-08

### 10.2 Physical Projector + Robot

- "Better Teaming Through Visual Cues": Ganesan et al., ASU, CSCW journal — human-robot communication
- "Projecting Robot Intentions": Sonawani et al., IROS 2023 — human-robot communication
- ProjecTA: arXiv 2601.11328, 2026-01 — HCI teaching assistant

### 10.3 Vision Encoder Robustness

- "Physically-based Lighting Augmentation for Robotic Manipulation": arXiv 2508.01442, 2025 — imitation learning policies struggle with lighting changes
- "Holistic Evaluation of Robustness in CLIP Models": arXiv 2410.01534 — CLIP/SigLIP sensitivity to visual distribution shifts
- SigLIP 2: arXiv 2502.14786, 2025-02

### 10.4 SmolVLA Architecture

- SmolVLA: Shukor et al., arXiv 2506.01844, 2025-06
- Vision encoder: SigLIP (via SmolVLM-2)
- Image resolution: 512x512, 64 visual tokens per frame (pixel shuffle)
- Action Expert: Flow Matching Transformer (~100M params)
- Total: ~450M (350M VLM + 100M Action Expert)

---

## 11. 추가 검토 이력

### 11.1 MEM/RECAP/RD-VLA 시너지 검토 (2026-03-19)

Pi의 MEM(arXiv 2603.03596), RECAP/π*₀.₆(arXiv 2511.14759), RD-VLA(arXiv 2602.07845) 3개 논문을 Projector-VLA에 결합하는 안이 제안됨. 검토 결과 **기각**:

- "프로젝터 = 물리적 메모리" → 은유일 뿐, 메모리와 프롬프트를 혼동
- "RECAP 가치 함수를 프로젝터로 시각화" → 순환 논리 (가치 함수가 있으면 프로젝터 불필요)
- "RD-VLA 적응형 추론 → 프로젝터 정밀도 조절" → 전제 3개 미충족 (불확실성 측정, 실시간 제어, SigLIP 인식)
- 3개 논문 전부 closed-source → SmolVLA에 기술적 이식 불가

### 11.2 WiFi CSI 센싱 적용 검토 (2026-03-19)

RuView 프로젝트(WiFi CSI로 벽 뒤 인체 감지)를 로보틱스에 적용하는 안이 검토됨. **별도 문서로 격리**: `claudedocs/WIFI_CSI_ROBOTICS_ANALYSIS.md`

기각 사유: WiFi CSI 공간 해상도(~0.5m)가 manipulation 정밀도(~1-5mm)와 100배 차이. Projector-VLA 및 Bimanual VLA 어느 쪽에도 적용 불가.
