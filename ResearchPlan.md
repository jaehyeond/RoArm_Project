# Research Plan: Data-Efficient VLA Adaptation on Consumer Hardware

> 졸업 논문 + CoRL 2026 연구 계획서
> 최종 업데이트: 2026-03-19
> 상태: **Data-Centric Multi-Object 방향 확정** (SmolVLA 유지)
> 타겟: CoRL 2026 (마감 2026-05-28), IROS 2026 LBR (보험, 마감 2026-07-31)
> **⚠️ 2026-03-19 방향 전환**: Bimanual → Data-Centric (70일 타임라인, SO-101 불필요)
> **⚠️ 2026-03-10 대규모 정정**: 이전 "연구 갭" 주장 4/5 거짓 판명 → 재검증 완료

---

## 0. 방향 전환 요약 (2026-03-19)

### 왜 전환하는가?
- CoRL 2026 마감까지 70일 — Bimanual(SO-101 구매+셋업 4-6주)은 시간 부족
- Bimanual의 기여가 얕음: "SmolVLA가 bimanual도 된다" = 1줄 결과
- Data-Centric은 기존 자산(74ep, 파이프라인)을 즉시 활용

### 새 연구 질문
**"새 로봇에 소형 VLA를 적용할 때, 얼마나 많은/좋은 데이터가 필요한가?"**

### 기여 4가지
1. **OOD Scaling Laws**: episodes(25-150) × quality(filtered/not) × steps(25K-200K) → 성공률
2. **Data Quality Methodology**: 7단계 검증, FK depth, gripper phase, static frame detection
3. **Multi-Object Transfer**: 4물체(sponge/cup/box/tool) × 50ep, cross-task transfer
4. **Self-Improving Loop**: 배포→VLM 판별→성공 rollout 재활용→재학습 (Seed2Scale-lite)

### 70일 타임라인
```
D-70~D-68: Agent personas + 연구 계획 확정 ← 현재
D-68~D-56: Multi-object 데이터 수집 (cup/box/tool 각 50ep)
D-56~D-46: Scaling 실험 매트릭스 (40 runs)
D-46~D-38: 배포 평가 (물체당 20 trials × 체크포인트)
D-38~D-30: Self-improving loop 구현 + 실험
D-30~D-24: Multi-task 합동 학습 + transfer 실험
D-24~D-10: 논문 작성 + 그림 제작
D-10~D-0:  마무리 + 제출 (5/28)
```

### Bimanual은?
졸업논문 후속 연구로 분리. CoRL 논문의 "Future Work"에서 언급.

---

## (이전 내용은 아래에 아카이브)

---

## 1. 배경 및 동기

### 1.1 교수님 피드백 (2026-03 랩미팅)

> "다른 데서는 공중제비 하는 로봇도 나오고, 보스턴 다이나믹스, 중국, 테슬라에서 미친듯이 로봇이 쏟아져 나오는데 넌 언제까지 로봇팔로 스펀지만 잡고 있을거냐?"

이 피드백의 본질: **"스펀지 잡기가 작다"가 아니라 "연구 프레이밍이 작다"**는 것.

### 1.2 현재까지 달성한 것

| 항목 | 내용 | 날짜 |
|------|------|------|
| SmolVLA 파이프라인 구축 | Azure Kinect → SmolVLA → RoArm-M3 | 2026-02 |
| Cross-embodiment 전이 | SO-100 pretrained → RoArm-M3 배포 | 2026-02 |
| v1 배포 실패 분석 | 50ep, gripper 편향, closed-loop drift | 2026-02-11 |
| v3 배포 성공 | 74ep, open-loop 4-chunk, 5/5 (100%) | 2026-02-25 |
| 데이터 품질 방법론 | depth 분류, gripper phase, 정지 프레임 탐지 | 2026-02 |
| Isaac Lab 셋업 | RoArm-M3 URDF→USD, RL 파이프라인 검증 | 2026-02-20 |

**핵심 성과**: SmolVLA(SO-100 pretrained)를 구조적으로 다른 로봇(RoArm-M3)에 이식해서 실제 동작 성공 — 학계에서 체계적으로 연구된 적이 거의 없는 영역.

### 1.3 왜 방향 전환이 아니라 승격인가

- 3개월간 쌓은 파이프라인, 데이터, 배포 경험을 버리고 새로 시작하면 6개월 더 걸림
- 스펀지는 테스트 오브젝트일 뿐, 실제 연구는 cross-embodiment VLA 전이
- 교수님이 원하는 건 더 큰 로봇이 아니라 **더 큰 질문**

---

## 2. 보유 자산

### 2.1 하드웨어

```
┌─────────────────────────────────────────────────────┐
│  로봇                                                │
│  ├── RoArm-M3-Pro #1 (Follower, /dev/ttyUSB0)      │
│  ├── RoArm-M3-Pro #2 (Leader, /dev/ttyUSB1)        │
│  └── RoArm-M3-Pro #3 (추가 보유)                    │
│                                                      │
│  카메라                                               │
│  ├── Azure Kinect DK #1 (RGB 720P + Depth NFOV)     │
│  ├── Azure Kinect DK #2                             │
│  └── Azure Kinect DK #3                             │
│                                                      │
│  컴퓨트                                               │
│  ├── RTX 4090 Laptop (15.6GB VRAM, CUDA 12.6)      │
│  └── Cloud GPU 임대 가능 (Vast.ai, Lambda 등)        │
└─────────────────────────────────────────────────────┘
```

### 2.2 소프트웨어 환경

| Component | Details |
|-----------|---------|
| OS | Ubuntu 22.04 (Linux) |
| GPU | RTX 4090 Laptop, Driver 580, CUDA 12.6 |
| Python | 3.11.14 (conda env `roarm`) |
| PyTorch | 2.7.1+cu126 |
| LeRobot | 0.4.4 (source install, editable) |
| SmolVLA | lerobot[smolvla] extras |
| Isaac Lab | 2.3.0 (conda env `isaaclab`, 별도) |
| roarm_sdk | 0.1.0 |
| pyk4a | 1.5.0 (libk4a 1.4.2) |

### 2.3 경험적 자산 (경쟁 우위)

1. **실제 하드웨어 배포 경험** — 대부분 VLA 논문은 sim-only이거나 같은 로봇
2. **실패→성공 체계적 데이터** — v1(50ep 실패) vs v3(74ep 성공) 직접 비교 가능
3. **데이터 품질 분석 방법론** — depth 분류, gripper phase, 정지 프레임 탐지
4. **Cross-embodiment 전이 증명** — SO-100 pretrained → RoArm-M3 동작 성공

---

## 3. 연구 방향 탐색 과정

### 3.1 광범위 조사 (2026-03-07)

12개 연구 방향을 4개 병렬 에이전트로 조사:
- Agent 1: 학회 트렌드 (RSS 2025 163편 분석, ICRA, CoRL, IROS)
- Agent 2: 산업 동향 (NVIDIA, Google, Tesla, 중국, Boston Dynamics)
- Agent 3: ROS 생태계 (352+ 채용, $179k 중위 연봉, 한국 시장)
- Agent 4: 소프트웨어 연구 방향 (Top 5 순위)

### 3.2 초기 후보 5개

| # | 방향 | 새로움 | 실현가능성 | 임팩트 | 기존 작업 활용 |
|---|------|--------|-----------|--------|--------------|
| 1 | Cross-Embodiment VLA Adaptation | HIGH | HIGH | MEDIUM | PERFECT |
| 2 | Sim-to-Real for VLA (Isaac Lab) | HIGH | MEDIUM | HIGH | PARTIAL |
| 3 | LLM-Driven Reward Generation | MEDIUM | LOW-MEDIUM | HIGH | LOW |
| 4 | World Models for Manipulation | HIGH | LOW | HIGH | LOW |
| 5 | Foundation Model + ROS 2 Deployment | LOW | HIGH | MEDIUM | MEDIUM |

### 3.3 RD-VLA 논문 심층 분석

**논문**: "Recurrent-Depth VLA" (arXiv 2602.07845, Feb 2026, Stanford/AI2/UW)

**핵심 아이디어**: Latent space에서 가변 깊이 iterative refinement — 토큰 생성 없이 사고 깊이 조절

| 항목 | 내용 |
|------|------|
| 파라미터 | 0.5B (SmolVLA와 유사) |
| 성능 | LIBERO-10: 93.0% (7B 모델 초과) |
| 아키텍처 | Prelude → Recurrent Core (weight-tied) → Coda |
| 학습 | TBPTT + Randomized Recurrence |
| VLM | Qwen2.5-0.5B (LoRA tuned) |
| Vision | DINOv2 + SigLIP fused |
| 코드 | **미공개** (GitHub 빈 repo) |
| 상태 | Under Review |

**RD-VLA vs SmolVLA 핵심 차이**:
```
SmolVLA:  이미지 → VLM(frozen) → Flow Matching(action space에서 noise→action 정제)
RD-VLA:  이미지 → VLM(LoRA) → Recurrent Core(latent space에서 이해 정제) → action 한 번 출력
```
- SmolVLA: "어떻게 움직일지"를 action space에서 직접 정제
- RD-VLA: "상황을 어떻게 이해할지"를 latent space에서 정제
- 이 둘은 근본적으로 다른 문제를 정제하고 있음

**RD-VLA 한계**:
- MSE loss → multimodal action distribution 못 잡음 (SmolVLA의 Flow Matching이 우위)
- Cross-embodiment 전이 실험 없음
- Action chunking 없음 (single-step output)
- 코드/가중치 미공개 → 재현 불가
- 실제 로봇 결과 정성적(정량 수치 없음)
- "80배 빠르다" = 가장 느린 baseline(ThinkAct 7B+CoT)과 비교한 최대값

**우리에게 주는 영감**:
- Adaptive Flow Matching: SmolVLA의 10 denoise steps를 가변으로 (쉬운 동작 3 steps, 어려운 동작 10 steps)
- Chunk-level confidence monitoring: open-loop chunk 간 action 변화량 모니터링
- 당장 적용 가능한 것: **없음** (코드 없음, 아키텍처 다름)

### 3.4 엥지유니버스 RD-VLA 영상 분석

YouTube 채널의 논문 리뷰 영상을 교차 검증:

| 항목 | 평가 |
|------|------|
| 아키텍처 설명 정확도 | 90% — 핵심 구조 충실 |
| 학습 방법 설명 | 85% — TBPTT, Randomized Recurrence 정확 |
| 결과 해석 공정성 | 50% — "80배 빠름", "스스로 판단" 등 과장 |
| 한계점 분석 | 10% — MSE, cross-embodiment, 코드 미공개 등 빠짐 |
| 종합 | 아키텍처 해설은 충실하지만 비판적 분석 부재 — 홍보성 리뷰 |

**디지털 리터러시 교훈**: 논문 리뷰 영상도 반드시 원본 논문과 교차 검증 필요.

### 3.5 검증된 팩트 (연구 방향 결정 근거)

리서치 에이전트를 통해 검증한 핵심 사실:

| 질문 | 답변 | 검증 수준 |
|------|------|----------|
| SmolVLA로 bimanual(양팔) 한 사람? | **없음.** pi0는 했지만 SmolVLA는 단일 팔만 테스트 | HIGH (논문+코드 확인) |
| SmolVLA가 12-DOF(2팔) 지원? | **YES.** `max_action_dim=32`, zero-padding 코드 확인 | HIGH (소스코드 직접 확인) |
| RGB-D/3D 입력을 VLA에 체계적으로 연구? | **거의 없음.** DP3가 point cloud 사용했지만 VLA 아님. 3D-VLA 논문 있으나 planning용 | HIGH |
| 멀티로봇 VLA 논문? | **발견 못 함.** 멀티로봇은 RL 기반만 존재 | MEDIUM (부재 증명 어려움) |
| ALOHA는 VLA? | **아님.** ACT(CVAE+Transformer, 80M) 사용. 하지만 SmolVLA에 ALOHA 호환 config 존재 | HIGH |
| pi0 bimanual? | **YES.** Bi-ARX, Bi-Trossen에서 양팔 검증. 하지만 closed-source | HIGH (논문 확인) |
| SmolVLA pretrained 데이터? | SO-100 전용 (community_dataset_v1, 128 datasets, 11,132 episodes, 전부 SO-100) | HIGH (논문+저자 확인) |

### 3.6 SmolVLA Bimanual 지원 코드 근거

```python
# lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py
max_state_dim: int = 32   # 입력 state 최대 차원
max_action_dim: int = 32  # 출력 action 최대 차원

# modeling_smolvla.py
state = pad_vector(state, self.config.max_state_dim)      # 6-DOF → 32 zero-pad
actions = pad_vector(batch[ACTION], self.config.max_action_dim)  # 6-DOF → 32 zero-pad
# ...
original_action_dim = self.config.action_feature.shape[0]
actions = actions[:, :, :original_action_dim]              # 32 → 원래 차원으로 unpad
```

- RoArm-M3 × 2 = 12-DOF → 32-DOF 패딩 후 학습 → 12-DOF unpad 출력
- 코드 수정 최소: `action_feature.shape`만 12로 설정하면 됨

### 3.7 VLA 모델 비교 분석 (2026-03-09)

RTX 4090 Laptop (15.6GB VRAM) 제약 하에서 사용 가능한 VLA 모델 전수조사:

| 모델 | 파라미터 | 로컬 학습 | 로컬 추론 | LeRobot | Bimanual | Open-source | 비고 |
|------|---------|----------|----------|---------|---------|-------------|------|
| **SmolVLA** | 450M | **YES** (9.85GB) | YES (2GB) | Native | 코드 지원, 미검증 | YES | 현재 사용 중 |
| pi0 | 3B | NO (22.5GB+) | YES (8GB) | 데이터 호환 | 검증됨 | NO | Closed-source |
| pi0.5 | 3B | NO | YES | 호환 | YES | NO | pi0 확장 |
| **π\*₀.₆** | **5B** | NO | YES (H100, 63ms) | openpi 미포함 | YES (bimanual) | **NO** | RECAP (offline RL), arXiv 2511.14759 |
| RD-VLA | 0.5B | ? | YES | 미호환 | YES (YAM) | Partial | Recurrent network depth, ICLR 2026 |
| GR00T N1.6 | 3B | NO | YES (22.8Hz) | V2 native | humanoid 중심 | Partial (비상업) | NVIDIA, 2026-01 |
| OpenVLA-OFT | 7B | NO (27GB+) | NO (16GB+) | 미호환 | 미검증 | YES | 너무 큼 |
| Octo | 93M | YES | YES | JAX only | NO | YES | 후속 없음 |
| RDT-1B | 1B | 가능 | YES | HDF5 only | **특화** | YES | LeRobot 미호환 |
| TinyVLA | <1B | YES | YES | 미호환 | 미검증 | YES | 최신, 데이터 효율적 |
| SpatialVLA | 4B | NO (4-8x A100) | NO | 미호환 | NO | YES | 3D 특화 |
| CrossFormer | 130M | YES | YES | 부분 | NO | YES | 범용 backbone |

**결론**: RTX 4090 Laptop에서 학습 가능한 VLA = **SmolVLA가 유일**. pi0는 추론만 가능, 학습은 Cloud A100 필요.

**2026 핵심 트렌드**:
- VLA = 산업 전체 수렴 중 (NVIDIA, PI, Google, Figure 모두 채택)
- ICLR 2026 VLA 제출: 9개(2025) → **164개(2026)** = 18배 폭증
- RL augmented VLA: π\*₀.₆ RECAP (offline RL + advantage conditioning), SimpleVLA-RL (ICLR'26), PLD
- 3D/Depth VLA: DepthVLA, SpatialVLA(RSS'25), PointVLA, GeoVLA, Any3D-VLA 등 8개+ 논문
- Adaptive chunking: Mixture of Horizons (MoH, 2025-11) — pi0.5에 적용, 99% LIBERO
- 순수 imitation은 ~80% 천장 → RL 보강이 frontier (LIBERO는 "사실상 해결" 95-99%)
- **데이터 품질 연구가 놀라울 정도로 부족** (ICLR 2026 메타분석) → 우리 강점

### 3.8 로봇 플랫폼 비교 분석 (2026-03-09)

| 플랫폼 | 가격/대 | 6-DOF | SmolVLA 호환 | Bimanual | LeRobot 공식 | 비고 |
|--------|--------|-------|-------------|---------|-------------|------|
| **RoArm-M3** | 보유중 (3대) | YES | OOD (150ep+, 200K steps) | 가능 | NO | SDK 버그, 워크어라운드 필요 |
| **SO-100** | $130 | YES | **In-dist** (50ep, 20K) | bi_so config | **YES** | SmolVLA 사전학습 로봇 |
| **SO-101** | $230 | YES | **In-dist** | bi_so_follower | **YES** | SO-100 후속, 더 나은 빌드 |
| Koch v1.1 | $250 | YES | OOD | 가능 | YES | Dynamixel 서보 |
| ALOHA | $20K | YES | OOD | **검증됨** | 부분 | pi0 native, 비용 과다 |
| ViperX 300s | $7.5K | 6+1 | OOD | 가능 | YES | 고급 |

**RoArm-M3의 딜레마**:
```
장점: Cross-embodiment 전이가 곧 연구 기여 (SO-100→RoArm-M3 이미 성공)
단점: SDK 워크어라운드, 데이터 3-4배 필요, 학습 4배, 공식 지원 없음
      → "연구가 아닌 엔지니어링에 시간 소비"
```

**SO-101로 전환하면**:
- In-distribution → 50ep + 20K steps 충분 (vs RoArm 150ep + 200K)
- 공식 bi_so_follower config → bimanual 설정 수분 내 완료
- LeRobot 공식 지원 → SDK 버그 걱정 없음
- BUT: cross-embodiment 연구 각도를 잃음
- BUT: 기존 3개월 RoArm 경험 활용도 감소

**권장: 하이브리드 전략**
```
Phase 1: SO-101 2대 구매 ($460-600) → bimanual 빠른 검증 (in-dist)
Phase 2: RoArm-M3 bimanual → cross-embodiment 비교 (기존 자산 활용)
논문:    "Same VLA, two different embodiments, both bimanual"
         = cross-embodiment + bimanual의 교차 기여
```

### 3.9 플랫폼 전략 의사결정

| 옵션 | 비용 | 기간 | 논문 기여도 | 리스크 |
|------|------|------|-----------|--------|
| A) RoArm-M3만 | $0 | 6개월+ | cross-embodiment only | 높음 (SDK, 데이터 오버헤드) |
| B) SO-101만 구매 | $460-600 | 4개월 | bimanual VLA only | 낮음 (in-dist 검증) |
| C) 하이브리드 (SO-101 + RoArm) | $460-600 | 5-6개월 | **cross-embodiment + bimanual** | 중간 |

**선택: Option C (하이브리드)**

이유:
1. SO-101에서 빠르게 bimanual 검증 (1-2개월 절약)
2. RoArm-M3로 cross-embodiment 비교 (기존 자산 + 경험 활용)
3. 논문 contribution이 2배 (bimanual + cross-embodiment)
4. SO-101 실패 시 → RoArm-M3로 fallback 가능
5. $460-600 투자 = 대학원생 연구비로 합리적

---

## 4. 선정된 연구 방향: Cross-Embodiment Bimanual VLA

### 4.1 논문 제목 (가안)

> **"Cross-Embodiment Bimanual VLA: Open-Source Vision-Language-Action Models for Coordinated Manipulation Across Robot Platforms"**
>
> 대안: "BimanualVLA: Language-Conditioned Bimanual Manipulation with Cross-Embodiment Transfer"

### 4.2 연구 질문

| # | Research Question | 기존 작업과의 연결 |
|---|---|---|
| RQ1 | SmolVLA(단일 팔 VLA)가 양팔 협업 조작으로 확장될 수 있는가? | SmolVLA 12-DOF 지원 확인, 단일팔 배포 성공 |
| RQ2 | 동일 VLA가 서로 다른 로봇 플랫폼(SO-101, RoArm-M3)에서 bimanual 가능한가? | SO-100→RoArm-M3 전이 이미 성공 |
| RQ3 | In-distribution(SO-101) vs OOD(RoArm-M3) embodiment 간 bimanual 성능 차이는? | 정량적 비교 → cross-embodiment 논문 기여 |
| RQ4 | Multi-view depth가 양팔 workspace에서 occlusion을 해결하는가? | Azure Kinect 3대 활용 |

### 4.3 왜 이 방향인가

| 포인트 | 설명 |
|--------|------|
| **세계 최초** | SmolVLA bimanual = 논문 어디에도 없음 (검증됨) |
| **아키텍처 이미 지원** | `max_action_dim=32`, zero-padding → 코드 수정 최소 |
| **ALOHA와 차별화** | ALOHA=ACT(80M, no language), 우리=VLA(450M, language-conditioned) |
| **pi0와 비교 가능** | pi0는 bimanual 했지만 closed-source. 우리는 open-source SmolVLA |
| **Cross-embodiment** | 동일 VLA로 SO-101(in-dist) + RoArm-M3(OOD) 양쪽 bimanual 비교 |
| **하이브리드 플랫폼** | SO-101로 빠른 검증 → RoArm-M3로 전이 실험 → 논문 기여 2배 |
| **3번째 카메라** | multi-view로 양팔 workspace 커버 |
| **확장성** | 2팔→3팔, 단순 task→조립, language grounding |

### 4.3.1 2026 산업 트렌드와의 정합성

| 트렌드 | 우리 연구와의 연결 |
|--------|------------------|
| VLA = 산업 표준 수렴 | SmolVLA (open-source VLA) 사용 → 트렌드 정중앙 |
| Cross-embodiment 전이 | SO-101 ↔ RoArm-M3 비교 → 실용성 높은 연구 |
| RL + VLA 폭증 (π\*₀.₆, SimpleVLA-RL 등 8개+) | SmolVLA용 RL은 빈 슬롯, future work 언급 가능 |
| NVIDIA GR00T + Isaac Lab | Isaac Lab 셋업 완료 → sim-to-real 확장 가능성 |
| Open-source 생태계 중심 | HuggingFace LeRobot 기반 → 재현성 + 커뮤니티 임팩트 |
| 중국 로보틱스 급성장 | 하드웨어 접근성 ↑ → 소프트웨어 논문 수요 ↑ |

### 4.4 시스템 아키텍처

#### Setup A: SO-101 Bimanual (In-Distribution, 빠른 검증)
```
[Azure Kinect #1]     [Azure Kinect #2]     [Azure Kinect #3]
   (Left view)         (Center/Top view)      (Right view)
       ↓                     ↓                     ↓
┌──────────────────────────────────────────────────────────┐
│              SmolVLA (450M, 12-DOF output)                │
│   Input:  RGB image(s) + language + 12-DOF state          │
│   Output: action chunk [50 steps × 12-DOF]                │
│   Config: bi_so_follower (LeRobot 공식)                   │
└──────────┬──────────────────────────┬────────────────────┘
           ↓                          ↓
      [SO-101 #1]               [SO-101 #2]
       (Left arm)                (Right arm)
     In-distribution           In-distribution
```

#### Setup B: RoArm-M3 Bimanual (OOD Cross-Embodiment)
```
[Azure Kinect #1]     [Azure Kinect #2]     [Azure Kinect #3]
       ↓                     ↓                     ↓
┌──────────────────────────────────────────────────────────┐
│              SmolVLA (450M, 12-DOF output)                │
│   동일 아키텍처, 동일 학습 파이프라인                       │
│   BUT: 다른 로봇 → 다른 joint 분포 → OOD 도전             │
└──────────┬──────────────────────────┬────────────────────┘
           ↓                          ↓
      [RoArm-M3 #1]             [RoArm-M3 #2]
       (Left arm)                (Right arm)
     Out-of-distribution       Out-of-distribution
```

#### 비교 실험 구조
```
                    SmolVLA (동일 모델)
                    /              \
           SO-101 bimanual    RoArm-M3 bimanual
          (in-dist, baseline)  (OOD, 핵심 실험)
                    \              /
              Cross-Embodiment 성능 비교
              → "얼마나 전이되는가?"가 핵심 연구 질문
```

### 4.5 Bimanual Tasks (계획)

| Task | 설명 | 난이도 | 우선순위 |
|------|------|--------|---------|
| **Hold + Pick** | 한 팔이 물체 고정, 다른 팔이 부품 제거 | 낮음 | 1 (첫 번째) |
| **Handover** | 팔 A → 팔 B 물체 전달 | 중간 | 2 |
| **Hold + Pour** | 한 팔이 컵 잡고, 다른 팔이 주전자로 따르기 | 중간 | 3 |
| **Hold + Insert** | 한 팔이 구멍 잡고, 다른 팔이 삽입 | 높음 | 4 |
| **Collaborative Lift** | 두 팔로 큰 물체 동시에 들기 | 높음 | 5 (선택) |

### 4.6 대안 Tier (백업)

#### Tier 1: Multi-Task + Multi-View VLA (3개월, 안전한 백업)

단일 팔로 5가지 task 확장 + depth 카메라 3대 활용.
- Task: pick, pour, stack, push, tool-use
- Ablation: 1cam vs 2cam vs 3cam, RGB vs RGB-D
- 한계: 새로움 부족, workshop급

#### Tier 3: Autonomous Learning Loop (9개월+, 장기 비전)

3팔 시스템 = Teacher + Student + Evaluator, 사람 없이 자율 학습.
```
[Arm 1: Teacher]  ──demo──→  Dataset
                                ↓
                           SmolVLA 학습
                                ↓
[Arm 2: Student]  ←─policy─  Checkpoint
       ↓
   Task 시도
       ↓
[Arm 3: Evaluator] ──성공/실패 판정──→ Feedback Loop
       ↓
   Scene Reset → 다시 시도...
```
"로봇이 스스로 사고하는" 영역에 가장 가깝지만 복잡도 높음.

---

## 5. Cloud GPU 전략

### 5.1 SmolVLA 학습 시간 (공식 기반)

> "A100에서 20K steps ≈ 4시간" — LeRobot 공식 문서 (smolvla.mdx)

| GPU | 50K steps | 100K steps | 200K steps |
|-----|-----------|------------|------------|
| RTX 4090 (로컬) | ~12-15시간 | ~25-30시간 | ~50-60시간 |
| A100 80GB (클라우드) | ~10시간 | ~20시간 | ~40시간 |
| H100 80GB | ~5-7시간 | ~10-14시간 | ~20-28시간 |

### 5.2 Cloud GPU 가격 비교 (2026-03 검증)

#### Vast.ai (마켓플레이스, 가격 변동)

| GPU | On-Demand $/hr | Interruptible $/hr |
|-----|----------------|-------------------|
| RTX 4090 (24GB) | $0.20 - $0.50 | $0.15 - $0.35 |
| A100 80GB | $0.80 - $1.50 | $0.50 - $1.00 |
| H100 80GB | $2.00 - $3.50 | $1.50 - $2.50 |

주의: 커뮤니티 클라우드, SLA 없음, 품질 편차 있음

#### Lambda Labs (verified from FSDL CSV)

| GPU | $/hr |
|-----|------|
| A100 40GB | $1.10 |
| A100 80GB | $1.50 |
| H100 80GB | **$1.99** |

주의: 자주 품절, 엔터프라이즈 중심으로 전환 중

#### RunPod (verified from runpod.io, 2026-03)

| GPU | Flex $/hr | Active $/hr |
|-----|-----------|-------------|
| RTX 4090 | $1.10 | $0.77 |
| A100 80GB | $2.72 | $2.17 |
| H100 | $4.18 | $3.35 |

#### GCP (verified from cloud.google.com, 2026-03)

| GPU | On-Demand $/hr | Spot $/hr |
|-----|----------------|-----------|
| A100 40GB | $3.67 | $1.80 |
| A100 80GB | $5.07 | $2.53 |
| H100 (8-GPU only) | $11.06/GPU | $3.38/GPU |

- $300 무료 크레딧 (신규 계정, 90일)
- 학생 연구 크레딧: 공개 프로그램 없음, 교수-영업팀 협의 필요

#### 한국 클라우드 (미검증, 추정)

| Provider | A100 추정 | 비고 |
|----------|----------|------|
| Naver Cloud | ~$7-10/hr | 미국 대비 2-3배 비쌈 |
| KT Cloud | 유사 | 엔터프라이즈 중심 |

**결론: 학습 워크로드에 한국 클라우드 비추천. Vast.ai 또는 Lambda 사용.**

### 5.3 추천 전략

```
일상 학습/프로토타입:  로컬 RTX 4090 (무료, batch_size=64 가능)
Ablation 실험:       Vast.ai A100 80GB ($0.80-1.50/hr)
최종 학습:            Vast.ai H100 ($2-3/hr) — 속도 2배
대규모 실험:          Lambda H100 ($1.99/hr) — 가용 시
```

### 5.4 예상 예산

| 항목 | 세부 | 비용 |
|------|------|------|
| **하드웨어** | | |
| SO-101 × 2대 | $230/대 (추정, 배송비 포함) | $460-600 |
| **Cloud GPU** | | |
| SO-101 bimanual 학습 (50K) | 3회 (ablation), A100 @ Vast.ai | $24-45 |
| RoArm bimanual 학습 (100K-200K) | 3회, A100/H100 @ Vast.ai | $60-100 |
| Cross-embodiment 비교 학습 | 2회, H100 @ Vast.ai | $40-70 |
| Multi-view 실험 | 2회, A100 @ Vast.ai | $16-30 |
| **총 예상 예산** | | **$600-845** |

SO-101 비용 포함해도 $1,000 이하. 대학원생 연구비로 감당 가능.
SO-101 없이 RoArm만 사용하면 $150-250 (Cloud GPU만).

---

## 6. 6개월 로드맵 (하이브리드 플랫폼)

### Phase 1: SO-101 + 기반 구축 (Month 1)

```
Week 1-2: SO-101 구매 + 하드웨어 셋업
├── SO-101 2대 주문 ($460-600, Hiwonder/AliExpress)
├── SO-101 조립 + LeRobot 공식 설정 (bi_so_follower)
├── Azure Kinect 2대 셋업 (left view + top view)
├── USB Hub 구성: Kinect×2 + SO-101×2
├── 양팔 동시 제어 테스트 (LeRobot 공식 teleop)
└── 카메라 캘리브레이션 (intrinsic + extrinsic)

Week 3-4: SO-101 단일팔 baseline + 12-DOF dry run
├── SO-101 단일팔 pick task (50ep, 20K steps) → in-dist 검증
├── SmolVLA 12-DOF config 기능 테스트 (action_dim=12 dry run)
├── bi_so_follower config로 bimanual teleop 테스트
└── 데이터 수집 워크플로우 확립 (2팔 + 2카메라 동기화)
```

**검증 기준**: SO-101 단일팔 배포 성공률 > 90% (in-dist이므로 높아야)
**비교**: RoArm-M3 단일팔 74ep → 100% vs SO-101 50ep → ?%

### Phase 2: SO-101 Bimanual (Month 2-3)

```
Week 5-6: SO-101 Bimanual 데이터 수집
├── Leader-Follower × 2 (LeRobot 공식 bimanual teleop)
├── Hold+Pick task 시범 수집 (10ep) → 품질 검증
├── 데이터 포맷: 12-DOF action, multi-view image
└── 동기화 방식 검증 (2팔 + 2카메라)

Week 7-10: 본격 수집 + 학습
├── Hold+Pick: 5위치 × 10회 = 50 episodes
├── Handover: 5위치 × 10회 = 50 episodes
├── SmolVLA bimanual 학습 (50K steps, 로컬 RTX 4090)
├── SO-101 bimanual 배포 테스트
└── Multi-view ablation (1cam vs 2cam)
```

**검증 기준**: SO-101 Bimanual Hold+Pick > 70% (in-dist 이점)
**핵심**: 여기서 bimanual이 작동하면 → Phase 3에서 cross-embodiment 실험 진행

### Phase 3: RoArm-M3 Cross-Embodiment 비교 (Month 4)

```
├── RoArm-M3 bimanual 데이터 수집 (동일 task, 동일 카메라 설정)
│   ├── Hold+Pick: 5위치 × 20회 = 100 episodes (OOD이므로 더 필요)
│   └── 토크 OFF 수동 조작 (기존 방식)
├── RoArm-M3 bimanual 학습 (100K-200K steps)
├── Cross-embodiment 정량 비교:
│   ├── SO-101 bimanual 성공률 vs RoArm-M3 bimanual 성공률
│   ├── 데이터 효율성: 같은 성능에 몇 ep 필요?
│   ├── 학습 수렴 속도: 같은 loss에 몇 step 필요?
│   └── 실패 모드 비교: coordination error 분석
├── Language conditioning ablation
│   ├── "hold the cup and pick the ball" (specific)
│   ├── "help me with the cup" (vague)
│   └── No language (action-only)
├── 3번째 카메라 추가 → 3-view ablation
└── 200K steps 학습 (cloud GPU, Vast.ai A100/H100)
```

**검증 기준**: RoArm-M3 Bimanual Hold+Pick > 50%, Cross-embodiment gap 정량화
**핵심 결과물**: "In-dist에서 X%, OOD에서 Y% → 전이 효율 Z%" 수치

### Phase 4: 논문 작성 (Month 5-6)

```
├── 결과 정리 + 시각화
│   ├── 성공률 표 (task × embodiment × condition)
│   ├── Cross-embodiment 전이 그래프 (SO-101 vs RoArm-M3)
│   ├── Data efficiency 곡선 (episodes vs success rate, per embodiment)
│   ├── Ablation 그래프 (views, language, data amount)
│   ├── 실패 모드 분류 + 대표 영상
│   └── 단일팔 vs 양팔 비교 (두 플랫폼 모두)
├── 관련 연구 정리
│   ├── ALOHA/ACT (bimanual, but not VLA)
│   ├── pi0 (bimanual VLA, but closed-source)
│   ├── SmolVLA (open-source VLA, but single-arm only)
│   ├── 3D Diffusion Policy (3D input, but not VLA)
│   └── GR00T N1.6 (cross-embodiment, but humanoid 중심)
├── 논문 작성
└── 타겟 학회 (Section 10 참조)
```

---

## 7. 기술적 과제 및 리스크

### 7.1 예상 과제

| 과제 | 심각도 | 대응 방안 |
|------|--------|----------|
| SO-101 배송 지연 | 높음 | 조기 주문, RoArm-M3로 병행 진행 |
| 2팔 동시 수동 조작 어려움 | 높음 | SO-101: LeRobot 공식 bimanual teleop / RoArm: L-F × 2 |
| 2팔 + 2카메라 동기화 | 중간 | 타임스탬프 기반 동기, Azure Kinect sync cable 활용 |
| 12-DOF 학습 수렴 어려움 | 중간 | SO-101(in-dist)에서 먼저 검증 → RoArm으로 확장 |
| RoArm-M3 OOD bimanual 실패 | 중간 | SO-101 결과만으로도 논문 가능 (RoArm은 추가 기여) |
| 2팔 충돌 회피 | 중간 | JOINT_LIMITS 강화, workspace 분리 설계 |
| Cloud GPU 가용성 | 낮음 | Vast.ai + Lambda 병행, 로컬 4090 백업 |
| SO-101과 RoArm 데이터 호환 | 낮음 | 별도 dataset, 별도 학습, 동일 아키텍처 |

### 7.2 리스크 완화 전략

1. **SO-101에서 먼저 검증** → in-dist bimanual 성공 확인 후 RoArm 진행
2. **Month 1에서 12-DOF dry run** → 아키텍처 문제 조기 발견
3. **Hold+Pick부터 시작** → 가장 간단한 양팔 task, 실패 확률 최소
4. **SO-101 실패 시 → Tier 1 (multi-task 단일팔)으로 전환 가능**
5. **RoArm 실패 시 → SO-101 결과만으로 논문 가능** (cross-embodiment는 bonus)
6. **3번째 RoArm = 단일팔 baseline** → 비교 데이터 항상 확보

---

## 8. 관련 연구 (검증됨)

### 8.1 Bimanual Manipulation

| 논문 | 모델 | 양팔 | VLA | Open-source | Cross-embodiment |
|------|------|------|-----|-------------|-----------------|
| ALOHA (Zhao et al., 2023) | ACT (80M) | YES | NO | YES | NO |
| Mobile ALOHA (Fu et al., 2024) | ACT | YES | NO | YES | NO |
| pi0 (Black et al., 2024, RSS 2025) | pi0 (3B) | YES (Bi-ARX, Bi-Trossen) | YES | **NO** | YES (주장) |
| π\*₀.₆ (PI, 2025-11, arXiv 2511.14759) | pi0.6+RECAP (5B) | YES (bimanual static) | YES | **NO** | YES |
| RD-VLA (2026-02, ICLR 2026 WS) | RD-VLA (0.5B) | YES (YAM bimanual) | YES | Partial | NO |
| GR00T N1.6 (NVIDIA, 2026-01) | GR00T (3B) | YES (humanoid) | YES | Partial | YES (humanoid 중심) |
| DexMimicGen (Jiang et al., 2024) | IL | YES | NO | YES | NO |
| **Ours (proposed)** | **SmolVLA (450M)** | **YES** | **YES** | **YES** | **YES (SO-101 + RoArm-M3)** |

**우리의 차별점**: Open-source VLA + bimanual + 두 플랫폼 cross-embodiment 비교 = 빈 슬롯

### 8.2 3D/Depth + VLA (2026-03 재검증 — 이전 "갭 없음" 주장 정정)

> **⚠️ 정정**: 이전에 "RGBD-VLA 논문이 없다"고 주장했으나 **완전히 틀림**. 8개+ 논문 존재.

#### 실제 depth sensor 사용하는 VLA:

| 논문 | 날짜 | Depth 소스 | 비고 |
|------|------|-----------|------|
| **DepthVLA** (arXiv 2510.13375) | 2025-10 | 사전학습 depth prediction | Mixture-of-Transformers |
| **GeoVLA** (arXiv 2508.09071) | 2025-08 | RealSense 실제 depth map | Point Embedding Network |
| **PointVLA** (arXiv 2503.07511) | 2025-03 | LiDAR point cloud | Lightweight 3D injection |
| **Any3D-VLA** (arXiv 2602.00807) | 2026-02 | 센서 + 추정 둘 다 | Open-source (GitHub + HF) |

#### Depth 추정 (RGB만으로 3D 이해):

| 논문 | 날짜 | Depth 소스 | 비고 |
|------|------|-----------|------|
| **SpatialVLA** (arXiv 2501.15830) | 2025-01 | ZoeDepth (단안추정) | **RSS 2025**, Ego3D Position Encoding |
| **AugVLA-3D** (arXiv 2602.10698) | 2026-02 | VGGT (단안추정) | PointNet encoder |
| **QDepth-VLA** (arXiv 2510.14836) | 2025-10 | 양자화 depth prediction | Auxiliary depth supervision |
| **StemVLA** (arXiv 2602.23721) | 2026-02 | Video-geometry transformer | 4D historical representations |

#### 관련 (VLA 아닌 3D policy):

| 논문 | 날짜 | 비고 |
|------|------|------|
| 3D Diffusion Policy (Ze et al., RSS 2024) | 2024-03 | Point Cloud + Diffusion, VLA 아님 |
| 3D-VLA (Hong et al., ICML 2024) | 2024-03 | World model, not low-level control |

**결론**: 3D/Depth VLA는 이미 **레드오션**. 우리가 depth 자체로 novelty 주장하는 것은 불가.
Azure Kinect depth는 multi-view ablation의 **부가 실험**으로만 포지셔닝.

### 8.3 Cross-Embodiment VLA

| 논문 | 전이 범위 | 실제 검증 |
|------|----------|----------|
| SmolVLA (Shukor et al., 2025) | SO-100 → SO-101 | 같은 로봇 패밀리 |
| Octo (Ghosh et al., 2024) | 9 platforms | 광범위하지만 VLA 아님 (93M) |
| pi0 (Black et al., 2024) | 8 robots | YES, but closed-source |
| **Ours (achieved)** | **SO-100 → RoArm-M3** | **다른 로봇 패밀리, 성공** |

### 8.4 VLA + RL / Self-Improvement (2026-03 재검증)

> **⚠️ 정정**: 이전에 "open-source self-improving VLA 없다"고 주장했으나 **거짓**. 8개+ 논문/도구 존재.

| 논문/도구 | 방법 | RL 유형 | Open-source | 학회 | 핵심 |
|-----------|------|---------|-------------|------|------|
| **π\*₀.₆ RECAP** (PI, 2025-11) | Advantage conditioning | Offline RL | **NO** | arXiv | Demos+rollouts+corrections, 5B |
| **SimpleVLA-RL** (2025-09) | GRPO-based | Online RL | **YES** (GitHub) | **ICLR 2026** | 1 demo → 91.7% LIBERO-Long |
| **PLD** (2025-10) | Residual RL + distill | Residual RL | YES | arXiv | 99% LIBERO, 100% real Franka |
| **VLA-RL** (2025-05) | Process reward model | On-policy RL | NO | arXiv | OpenVLA-7B 기반 |
| **TT-VLA** (2026-01) | Test-time adaptation | Test-time RL | NO | arXiv | 재학습 불필요 |
| **GigaBrain-0.5M\*** (2026-02) | World model RL (RAMP) | Model-based RL | YES | arXiv | Leaderboard #1 (2026-02) |
| **SOAR** (Berkeley, 2024) | VLM-as-judge + hindsight | Self-supervised | **YES** (GitHub) | **CoRL 2024** | 5 WidowX fleet, 30K trajectories |
| **ORBIT** (2026) | 배포 진단 + 데이터 처방 | 진단 도구 | **YES** (Apache 2.0) | 도구 | LeRobot 호환, 논문 아님 |

**핵심 관찰**:
- 모든 RL 논문이 OpenVLA(7B) 또는 pi0(3B/5B) 타겟 → **SmolVLA(450M)용 RL = 없음**
- SimpleVLA-RL은 autoregressive VLA용 → flow matching VLA(SmolVLA)에 직접 적용 불가
- π\*₀.₆가 유일한 flow matching + RL이지만 closed-source
- **Gap**: Open-source flow matching VLA + RL (기술적 매우 어려움, future work)

### 8.5 VLA Deployment Monitoring / Uncertainty (2026-03 재검증)

> **⚠️ 정정**: 이전에 "VLA deployment monitoring 도구가 없다"고 주장했으나 **거짓**.

| 논문 | 날짜 | 학회 | 방법 |
|------|------|------|------|
| **DeeR-VLA** (arXiv 2411.02359) | 2024-11 | **NeurIPS 2024** | Action consistency로 early-exit 결정 |
| **Diff-DAgger** (arXiv 2410.14868) | 2024-10 | **ICRA 2025** | Diffusion loss를 uncertainty metric으로 사용, 39% failure prediction 향상 |
| **"Uncertainty Comes for Free"** (arXiv 2503.01876) | 2025-02 | — | Denoising noise → confidence signal, 모달 간 분산 분해 |
| **HiL Confidence-Aware** (arXiv 2602.10289) | 2026-02 | **HRI 2026** | Module-level uncertainty + human intervention 비용 모델 |

### 8.6 Adaptive Action Chunking (2026-03 재검증)

> **⚠️ 정정**: 이전에 "adaptive chunking VLA가 없다"고 주장했으나 **거짓**.

**Mixture of Horizons (MoH)** (arXiv 2511.19433, 2025-11, open-source):
- Action chunk를 다른 horizon으로 분할 (short + long)
- 동적 추론: 정밀 조작 → 짧은 horizon, 이동 → 긴 horizon 자동 선택
- pi0.5에 적용 → 99% LIBERO, 2.5x throughput
- 우리의 4-chunk open-loop 전략과 관련 → MoH가 이미 더 세련된 해법 제공

---

## 9. 교수님 Pitch

### 9.1 엘리베이터 피치 (30초)

> "open-source VLA인 SmolVLA를 세계 최초로 양팔 조작에 적용하고, 서로 다른 두 로봇 플랫폼(SO-101, RoArm-M3)에서 cross-embodiment bimanual 전이가 가능한지 체계적으로 검증합니다. pi0 같은 closed-source 모델만 했던 bimanual VLA를 open-source로 재현하고, in-distribution vs OOD 전이 효율을 정량 비교합니다."

### 9.2 상세 설명 (2분)

> "현재 bimanual VLA는 pi0(closed-source)만 검증했고, open-source VLA로는 아무도 안 했습니다. SmolVLA는 아키텍처적으로 32-DOF 지원하지만, 양팔 테스트는 미검증입니다.
>
> 저는 두 단계로 접근합니다:
> 1. SO-101(in-distribution) 2대로 bimanual 가능성 빠르게 검증
> 2. RoArm-M3(OOD) 2대로 동일 실험 → cross-embodiment 전이 효율 정량 비교
>
> 핵심 연구 질문: '동일 VLA가 서로 다른 로봇에서 양팔 협업이 가능한가? 그리고 그 전이 효율은?'
>
> 비교 대상:
> - vs 단일팔 SmolVLA (우리 기존 결과: RoArm-M3 100% 성공)
> - vs ALOHA+ACT (language 없는 bimanual, open-source)
> - vs pi0 (closed-source bimanual VLA, 논문 수치만 비교)
>
> Azure Kinect 3대의 multi-view 실험도 포함합니다.
> CoRL 2026 제출을 1차 목표로 합니다."

---

## 10. 학회 동향 및 출판 전략

### 10.1 학회별 적합성 분석 (2026-03 조사)

| 학회 | 분야 | 마감 (예상) | 우리 논문 fit | 비고 |
|------|------|-----------|-------------|------|
| **CoRL 2026** | Robot Learning | ~June 2026 | **BEST** | VLA + cross-embodiment = 정중앙 |
| IROS 2026 LBR | Robotics | **Jul 31, 2026** | HIGH | Late-Breaking Results, 2-4페이지, 가장 빠른 기회 |
| HRI 2027 | Human-Robot | ~Sep 2026 | MEDIUM | Hand-guiding 각도로 HCI 프레이밍 필요 |
| CHI 2027 | HCI | ~Sep 2026 | MEDIUM | User study 필수 — "사람이 bimanual teleop을 얼마나 쉽게 하는가" |
| IROS 2027 | Robotics | ~Mar 2027 | HIGH | Full paper 백업 |
| ICRA 2028 | Robotics | ~Sep 2027 | HIGH | 안전망 |
| RSS 2027 | Robotics | ~Jan 2027 | HIGH | Top-tier but 경쟁 치열 |
| CoRL 2027 | Robot Learning | ~June 2027 | **BEST** | 결과 충분하면 full paper |
| ACCV 2026 | Computer Vision | ~Jul 2026 | LOW-MEDIUM | Vision 기여 필요 |

#### SIGGRAPH, CHI, UIST 분석 (2026-03 조사)

| 학회 | VLA 논문 수 | 로보틱스 세션 | 우리 적합성 | 판정 |
|------|------------|-------------|-----------|------|
| **SIGGRAPH** | 0 | "Robots in the World" (2025) 있지만 graphics 기여 필수 | LOW | 제외 — VLA 논문 전례 없음 |
| **CHI** | 간접적 | teleop UX 논문 있음 | MEDIUM | HCI 프레이밍 + user study 필수 |
| **UIST** | 0 | novel interaction 필요 | LOW | 제외 — 인터페이스 기여 없음 |

### 10.2 출판 전략 (타임라인)

```
2026-07: IROS 2026 LBR 제출 (2-4페이지, SO-101 bimanual 초기 결과)
         → 빠른 피드백 + 학회 경험
2026-09: CoRL 2026 제출 (full paper, cross-embodiment bimanual 전체 결과)
         → 1차 목표
2027-03: IROS 2027 제출 (확장 결과, multi-view ablation 포함)
         → CoRL 탈락 시 백업
2027-06: CoRL 2027 제출 (추가 실험 + RL 보강 결과?)
         → 최종 안전망
```

**핵심**: arXiv 프리프린트 + 데모 영상 (YouTube/X)을 학회 제출과 동시에 공개
**졸업 논문**: 학회 논문을 기반으로 확장 (추가 ablation + 관련 연구 확대)

---

## 11. 산업 트렌드 연결 (2026-03 업데이트)

### 11.1 현재 로봇 산업 지도

| 회사 | 핵심 모델 | 양팔 | Open-source | 연결점 |
|------|----------|------|-------------|--------|
| Physical Intelligence | pi0 → pi0.5 → **π\*₀.₆** (5B, RECAP) | YES | NO (openpi에 미포함) | 비교 대상, offline RL + advantage conditioning |
| NVIDIA | **GR00T N1.6** (3B, 2026-01) | YES (humanoid) | Partial (비상업) | Isaac Lab 연계, LeRobot V2 |
| Google DeepMind | **Gemini Robotics** | 미확인 | NO | 비교 참조 |
| Hugging Face | SmolVLA (450M) | 미검증 | **YES** | **우리 기반** |
| ALOHA Team | ACT (80M) | YES | YES | 하드웨어 비교 |
| Figure | Figure 02 | YES (humanoid) | NO | VLA 채택 |
| Tesla | Optimus Gen 3 | YES (humanoid) | NO | 하드웨어 위주, AI 미공개 |

### 11.2 2026 핵심 트렌드 (2026-03-10 재검증)

| 트렌드 | 상태 | 우리 연구 영향 |
|--------|------|--------------|
| **VLA 폭증** | ICLR 2026: 9→164편 (18x) | 경쟁 치열, 차별화 필수 |
| **VLA = 산업 표준** | 수렴 완료 | 정방향 — VLA 연구가 주류 |
| **RL + VLA** | π\*₀.₆, SimpleVLA-RL(ICLR'26), PLD 등 8개+ | future work 언급, SmolVLA용 RL은 gap |
| **3D/Depth VLA** | 8개+ 논문 (레드오션) | depth 자체로 novelty 불가 |
| **Adaptive chunking** | MoH (2025-11, open-source) | 우리 4-chunk 전략의 학술적 근거 |
| **Cross-embodiment** | 주장은 많지만 **bimanual 정량 비교는 없음** | **우리 핵심 기여 (여전히 유효)** |
| **데이터 품질 연구 부족** | ICLR 2026 메타분석 지적 | **우리 data quality 도구 = 차별점** |
| **Open-source 생태계** | LeRobot, HF 중심 성장 | 재현성 + 커뮤니티 임팩트 |
| **Sim-to-Real** | NVIDIA Cosmos, Isaac Lab | Isaac Lab 셋업 완료 → 확장 가능 |

### 11.3 기회 갭 (2026-03-10 재검증 — 이전 잘못된 갭 정정)

> **⚠️ 주의**: 이전에 제시한 5가지 "갭" 중 4개가 거짓이었음. 아래는 검증된 실제 갭.

#### 이전 잘못된 주장 (정정)

| 이전 주장 | 판정 | 반증 |
|-----------|------|------|
| "RGBD-VLA 논문 없음" | **FALSE** | DepthVLA, SpatialVLA, PointVLA, GeoVLA 등 8개+ |
| "VLA 배포 모니터링 없음" | **FALSE** | DeeR-VLA(NeurIPS'24), Diff-DAgger(ICRA'25) |
| "Adaptive chunking 없음" | **FALSE** | Mixture of Horizons (2025-11, open-source) |
| "Open-source self-improving VLA 없음" | **FALSE** | SOAR(CoRL'24), SimpleVLA-RL(ICLR'26), PLD |
| "Flow matching 분산 novelty" | **부분적** | Diffusion uncertainty는 연구됨, flow matching은 미세 차이 |

#### 검증된 실제 갭 (2026-03-10)

| 갭 | 확실도 | 현재 상태 | 우리 가능성 |
|----|--------|----------|-----------|
| **SmolVLA bimanual** | HIGH | 0개 논문 — SmolVLA bimanual 검증한 사람 없음 | **직접 채움** |
| **Cross-embodiment bimanual 정량 비교** | HIGH | 아무도 같은 VLA로 2개 다른 로봇에서 bimanual 비교 안 함 | **직접 채움** |
| **SmolVLA 생태계 공백** | HIGH | 모든 RL/개선 논문이 OpenVLA(7B) 또는 pi0 타겟, SmolVLA 무시 | future work |
| **Open-source flow matching VLA + RL** | MEDIUM | π\*₀.₆ RECAP이 유일하지만 closed-source | 졸업 논문 범위 초과 |
| **Consumer hardware 자기 개선 루프** | MEDIUM-HIGH | SOAR=WidowX fleet, π\*₀.₆=커스텀 시스템, 모두 고가 | $200 로봇에서 최초 가능 |
| **VLA 데이터 품질 연구** | HIGH | ICLR 2026 메타분석이 "부족하다" 지적 | 우리 data quality 도구 활용 |

### 11.4 ROS 생태계 (취업 연결)

- 352+ 활성 채용 (ROS 2 Kilted Kaiju, 2025-05)
- 중위 연봉 $179k, 소프트웨어 트랙 $194k, CUDA $218k
- 한국: KRAFTON (VLA+ROS2), 42dot (SLAM), Samsung (Physical AI)
- **Bimanual VLA + Cross-Embodiment + ROS 2 = 최적 취업 포트폴리오**

---

## 12. 핵심 교훈 (이전 실패에서)

### 12.1 학습 관련

| Rule | Why |
|------|-----|
| 커스텀 학습 스크립트 작성 금지 | 공식 파이프라인의 정규화/스케줄러/전처리 누락됨 |
| `lerobot-train` CLI만 사용 | `run_official_train.py`가 래핑 |
| `lerobot/smolvla_base` 사전학습 필수 | Action Expert가 사전학습 안 되면 평균 액션만 출력 |
| Loss ↓ ≠ 좋은 모델 | L2 error + z-score + diversity 함께 확인 |
| batch_size=64 사용 | 공식 권장값, RTX 4090에서 9.85GB (충분) |

### 12.2 배포 관련

| Rule | Why |
|------|-----|
| dataset_mean 시작 위치 | [0,0,0,0,0,0] 시작은 OOD → 소심한 동작 |
| Open-loop chunk 사용 | Closed-loop n=1 → per-step noise → drift |
| JOINT_LIMITS 절대 제거 금지 | 하드웨어 보호 |
| n_action_steps=50 (공식 기본값) | 1로 바꾸면 mean regression → drift |

### 12.3 데이터 관련

| Rule | Why |
|------|-----|
| 카메라 고정 (삼각대/클램프) | 위치 변경 시 전체 데이터 무효 |
| Azure Kinect만 사용 | VLA 데이터는 반드시 pyk4a |
| 100+ 에피소드 목표 | 51개는 부족 (OOD embodiment는 더 필요) |
| 에피소드 품질 > 수량 | v1(50ep bad)=0% vs v3(74ep good)=100% |
| 새 데이터 추가 시 stats.json 변경 | 기존 체크포인트에서 이어학습 불가 → smolvla_base부터 재학습 |

---

## 13. 참고 자료

### 13.1 핵심 논문

- SmolVLA: Shukor et al., "SmolVLA: A Small Vision-Language-Action Model for Efficient Robot Learning", arXiv 2506.01844, June 2025
- pi0: Black et al., "pi0: A Vision-Language-Action Flow Model for General Robot Control", arXiv 2410.24164, Oct 2024 (RSS 2025)
- pi0.5: Physical Intelligence, "pi0.5: a Vision-Language-Action Model with Open-World Generalization", 2025
- π\*₀.₆: Physical Intelligence, "π\*₀.₆: a VLA That Learns From Experience", arXiv 2511.14759, Nov 2025 — RECAP (offline RL, advantage conditioning), 5B, NOT open-source
- ALOHA: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", arXiv 2304.13705, April 2023
- DP3: Ze et al., "3D Diffusion Policy", arXiv 2403.03954, March 2024 (RSS 2024)
- 3D-VLA: Hong et al., "3D-VLA: A 3D Vision-Language-Action Generative World Model", arXiv 2403.09631, March 2024
- RD-VLA: "Recurrent-Depth VLA", arXiv 2602.07845, Feb 2026 (ICLR 2026 WS Under Review) — "Depth"=네트워크 연산 깊이, NOT 깊이 카메라
- GR00T N1.6: NVIDIA, Jan 2026 — LeRobot V2 native, 22.8Hz on RTX 4090, 비상업 라이선스
- GR00T N1: NVIDIA, arXiv 2503.14734, March 2025
- Octo: Ghosh et al., arXiv 2405.12213, May 2024
- OpenVLA-OFT: Kim et al., "Fine-Tuning Vision-Language-Action Models", 2024
- TinyVLA: "TinyVLA: Towards Fast and Data-Efficient Vision-Language-Action Models", 2024
- RDT-1B: "RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins", 2024
- DepthVLA: arXiv 2510.13375, Oct 2025 — Mixture-of-Transformers, depth prediction module
- SpatialVLA: arXiv 2501.15830, Jan 2025, RSS 2025 — ZoeDepth Ego3D, 1.1M episodes
- PointVLA: arXiv 2503.07511, Mar 2025 — LiDAR point cloud injection
- GeoVLA: arXiv 2508.09071, Aug 2025 — RealSense depth map + Point Embedding
- Any3D-VLA: arXiv 2602.00807, Feb 2026 — 센서+추정 통합, open-source
- SimpleVLA-RL: arXiv 2509.09674, Sep 2025, ICLR 2026 — GRPO for VLA, open-source
- PLD (Self-Improving VLA): arXiv 2511.00091, Oct 2025 — Residual RL + distill, 99% LIBERO
- SOAR: arXiv 2407.20635, Jul 2024, CoRL 2024 — VLM-as-judge, 5 WidowX fleet, open-source
- MoH (Mixture of Horizons): arXiv 2511.19433, Nov 2025 — Adaptive action chunking, open-source
- DeeR-VLA: arXiv 2411.02359, Nov 2024, NeurIPS 2024 — Dynamic early-exit confidence
- Diff-DAgger: arXiv 2410.14868, Oct 2024, ICRA 2025 — Diffusion loss uncertainty
- ORBIT: github.com/Rahillasne/Orbit, 2026 — LeRobot 배포 진단 도구, Apache 2.0

### 13.2 프로젝트 내부 문서

| 파일 | 내용 |
|------|------|
| `CLAUDE.md` | 프로젝트 전체 가이드 |
| `claudedocs/RESEARCH_DIRECTION_2026.md` | 초기 연구 방향 탐색 |
| `claudedocs/DATA_COLLECTION_STRATEGY.md` | 데이터 수집 전략 |
| `claudedocs/CONTEXT_PROMPT_DATA_COLLECTION.md` | 컨텍스트 프롬프트 |
| `claudedocs/VLA_PAPERS.md` | VLA 논문 정리 |
| `claudedocs/RESEARCH_IDEAS.md` | 연구 아이디어 모음 |

### 13.3 외부 리소스

- LeRobot: https://github.com/huggingface/lerobot
- SmolVLA docs: https://huggingface.co/docs/lerobot/en/smolvla
- RoArm M3 PR: https://github.com/huggingface/lerobot/pull/820
- Isaac Lab: https://isaac-lab.github.io/
- Vast.ai: https://vast.ai/
- Lambda Labs: https://lambdalabs.com/
- GitHub: git@github.com:jaehyeond/RoArm_Project.git
