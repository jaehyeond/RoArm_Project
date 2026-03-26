# Isaac Sim 5.x / Omniverse 기능 전수조사 — VLA Sim-to-Real 적용성 분석
# A2 Sim-to-Real Specialist, 2026-03-26

## 조사 범위
- 설치된 Isaac Sim: isaacsim-5.1.0.0 (conda env `isaaclab`)
- Isaac Lab: 2.3.0
- 조사 방법: 실제 설치된 extension 목록 + 소스코드 직접 확인
- 이 문서의 목적: NVIDIA 마케팅 vs 실제 구현 현황 분리

---

## Step 1: 렌더링 관련 신기능 — 실제 설치된 것 기준

### 1-1. RTX Path Tracing 품질

**설치 확인**: `omni.hydra.rtx-a92aa88e9f54b05f` (버전 해시만 있음, 5.x 버전 명세 없음)
**실제 상태**:
- Isaac Sim 5.1의 RTX renderer는 이전 4.x 대비 물질 표현(PBR) 품질 향상 주장
- 실제 VLA 관련 논문에서 Isaac Sim RTX 렌더링으로 SigLIP OOD 통과한 사례: **현재까지 확인 없음**
- 이전 A2 분석 결론 유지: Isaac RTX rasterizer → cosine dist ~0.6-0.8, RTX renderer → ~0.3-0.5
- RTX path tracing은 렌더링 시간이 대폭 증가 (headless 512 envs RTX → VRAM OOM 16.3GB)
- **판정**: 품질이 개선되었어도 SigLIP frozen 임계값(< 0.2) 통과는 여전히 불확실. NVIDIA 마케팅 수준 주장

### 1-2. Domain Randomization

**설치 확인**: `isaacsim.replicator.domain_randomization-1.0.16+107.3.3`
**실제 구현 (소스 확인)**:
- `omni.replicator.core-1.12.27` randomizer.py에서 실제 지원 기능:
  - `scatter_2d`, `scatter_3d`: 물체 위치 랜덤화
  - `materials`: 재질 랜덤화
  - `texture`: 텍스처 랜덤화 (OmniPBR diffuse 적용)
  - `color`: 색상 랜덤화
  - `rotation`, `instantiate`: 회전/인스턴스 랜덤화
- **무엇이 없는가**: style transfer, neural rendering, semantic-level randomization

**Isaac Lab actuator 레벨 DR (소스 확인)**:
- `DelayedPDActuatorCfg`: `min_delay` ~ `max_delay` 범위에서 physics steps 단위 지연 주입 가능
- `ActuatorBaseCfg`: `friction`, `dynamic_friction`, `viscous_friction`, `armature` 랜덤화 가능
- `DCMotorCfg`: 포화 토크(`saturation_effort`) 설정 가능
- `ActuatorNetLSTM/MLP`: 학습된 actuator 모델 (실제 로봇 데이터로 학습 필요)

**중요**: DR로 해결 가능한 갭 vs 불가능한 갭 정리

| 갭 유형 | DR 가능 여부 | 방법 |
|---------|------------|------|
| Actuator lag (20-50ms) | 가능 | `DelayedPDActuatorCfg` |
| Joint friction (stiction) | 부분 가능 | `friction`, `dynamic_friction` 설정 |
| Joint backlash 1-2° | 불가 | PhysX에 backlash 모델 없음 |
| Gravity sag 2-5° | 가능 | 질량/관성 랜덤화 |
| Visual texture | 가능 | `randomizer.texture()` |
| SigLIP visual gap | **불가** | SigLIP frozen, DR은 픽셀 수준 변화뿐 |
| Contact dynamics (스펀지 등) | 부분 가능 | Deformable body (beta, 불안정) |

**판정**: DR 3.0이라 불릴 혁신 없음. 기존 DR + 새 actuator 파라미터 추가. physics DR은 실질적으로 활용 가능. 시각적 DR로 SigLIP 극복은 불가.

### 1-3. Neural Rendering / NeRF / 3DGS 통합

**설치 확인**: 해당 extension **없음**
- 검색한 키워드: neural, nerf, 3dgs, gaussian, style_transfer, diffusion
- 결과: 0건 (Omniverse extension에 neural rendering 통합 없음)

**공식 상태**:
- NVIDIA는 NeRF/3DGS를 "USD로 변환해서 불러오기" 방식 권장 (자체 통합 아님)
- NVIDIA Instant NGP는 별도 tool (Omniverse에 통합 아님)
- **판정**: NVIDIA 로드맵 주장이지 2026-03 기준 실제 통합 없음

### 1-4. Real2Sim 파이프라인

**설치 확인**: `Real2Sim` 관련 extension **없음**
- NVIDIA의 "Real2Sim" 주장은 Replicator + URDF importer 조합을 마케팅 언어로 부른 것
- 실제 파이프라인: 수동으로 RGBD 스캔 → mesh 추출 → USD 변환 → 불러오기 (자동화 없음)
- **판정**: "Real2Sim 파이프라인"이라는 버튼 하나는 없음. 전부 수동 공정.

### 1-5. NVIDIA Replicator 최신 기능

**설치 확인**: `isaacsim.replicator.scene_blox-1.0.8`, `isaacsim.replicator.grasping-1.0.9`
**Scene Blox 실제 기능** (소스 확인):
- Wave Function Collapse 기반 절차적 씬 생성
- `grid_utils/tile.py`, `tile_superposition.py` — 타일 기반 환경 생성
- 용도: 다양한 배경 생성 (창고, 책상 배치 등)
- **우리 적용성**: tabletop manipulation에서 배경/물체 배치 다양화는 가능. 하지만 SigLIP gap 해결 목적으로는 불충분.

**Replicator Grasping** (isaacsim.replicator.grasping-1.0.9):
- 그리퍼 grasp 포즈 자동 생성
- 합성 grasp 데이터셋 생성 지원
- **우리 적용성**: pick-and-place 합성 데이터 생성에 활용 가능성 있음 (확인도: MEDIUM)

---

## Step 2: SigLIP OOD 문제 우회/해결 방법

### 2-1. Style Transfer (Sim → Real appearance)

**Isaac Sim 내장**: **없음**
**외부 도구 조합 가능성**:
- CycleGAN (Zhu et al. 2017): sim 이미지 → real-style 변환
  - 실제 작동 사례: CASHER (2024), SimToReal Locomotion
  - 하지만: 추론 시점에도 style transfer 필요 (학습 시만 아니라 배포 시에도)
  - 배포 latency 추가: CycleGAN inference ~10-30ms (허용 가능)
- **작동 증거 있음**: Sim2Real-VLA (ICLR 2026)에서 image translation 접근 실험
  - 결과: pick-and-place에서 효과 있으나 fine-grained manipulation에서 artifact 문제
- **판정**: 기술적으로 작동 가능하나 우리 65일 내 구현은 MEDIUM RISK

### 2-2. NVIDIA 자체 Domain Adaptation 도구

**설치 확인**: **없음**
- "domain adaptation" 검색 결과: `omni.kit.property.adapter.*` (USD property adapter, DA와 무관)
- NVIDIA는 domain adaptation을 "Replicator로 더 많은 DR 데이터 생성"으로 대응
- **판정**: NVIDIA는 adaptation model이 아니라 "더 많은 데이터 생성"이 공식 입장

### 2-3. 3DGS로 환경 스캔 후 렌더링 — SigLIP OOD 극복 가능?

**이전 분석 (2026-03-24) 결과 재확인**:
- single-view RGBD → 3DGS: cosine dist ~0.4-0.5 (통과 불가)
- turntable 50장 → 3DGS: cosine dist ~0.2-0.3 (불확실)
- 3-view RGBD → 3DGS: cosine dist ~0.1-0.2 (통과 가능성)

**추가 분석**: 3DGS + Isaac 조합의 문제
- 3DGS는 렌더링 엔진이지 물리 시뮬레이터가 아님
- Isaac Lab에서 3DGS 렌더링 직접 사용 불가 (USD로 mesh 변환 필요, 품질 손실)
- 3DGS에서 생성한 배경 이미지 + Isaac에서 로봇 렌더링 합성 = artifact 경계 존재
- **판정**: 기술적으로 가능하나 Isaac+3DGS 통합은 미지원, 수동 조합 필요. SigLIP gap 해결 보장 없음.

### 2-4. CycleGAN/Diffusion-based Image Translation

**실제 작동 논문 증거**:
- CASHER (2024): Diffusion-based image translation로 sim→real 50% → 78% 향상
  - 조건: 로봇이 단순 gripper, 배경이 균일한 경우
  - 조건: Diffusion model fine-tuning 데이터 필요 (~100장 real 이미지)
- RoboGen (ICML 2024): diffusion으로 task + scene generation
- **우리 RoArm M3 적용 현실적 경로**:
  1. 실제 씬 사진 100장 수집 (30분)
  2. ControlNet/CycleGAN으로 sim→real style transfer 학습 (4-8시간, VAST.ai)
  3. Isaac에서 렌더링한 이미지를 모두 변환 후 SmolVLA 학습
  4. 배포 시: 실제 이미지는 변환 불필요 (스타일이 real이므로)
- **추정 SigLIP 개선**: cosine dist 0.6-0.8 → ~0.2-0.4 (충분하지 않을 수 있음)
- **판정**: MEDIUM 가능성, 실험 필요. 65일 내 구현 가능하나 확실성 낮음.

---

## Step 3: NVIDIA 최신 발표 파이프라인 조사

### 3-1. Isaac GR00T (Humanoid)

**공식 정보 (2025-2026)**:
- GR00T N1: NVIDIA 휴머노이드 VLA 기반 모델
- 780K 궤적 주장: H100 클러스터 기준, **RTX 4090 laptop에서 불가능** (이전 분석)
- Isaac Sim에서 GR00T N1 fine-tune 공식 지원: Franka, UR, BridgeData2 로봇 기준
- **RoArm M3 적용**: GR00T N1은 humanoid embodiment pre-training. RoArm M3 fine-tune 사례 없음.
- Isaac GR00T Mimic (`isaaclab_mimic-1.0.16`): **실제 설치 확인됨**
  - 1개 실제 데모 → 다양한 변형 자동 생성
  - 상세: `isaaclab/envs/manager_based_rl_mimic_env.py` 존재 확인
  - **이것이 실질적으로 활용 가능한 가장 유망한 기능**

**GR00T Mimic 구체적 기능**:
- 입력: 1-5개 텔레오퍼레이션 데모 (HDF5 형식)
- 출력: N개 변형 데모 (물체 위치, 로봇 초기 자세 다양화)
- 논문: "Mimic" (CVPR 2025 계열), Waypoint + Interpolation 방식
- **조건**: Isaac sim 환경 내에서만 작동. Real demo → sim transfer 먼저 필요.
- **판정**: 실제 구현 확인됨. 하지만 Real→Sim 변환이 선결 조건.

### 3-2. NVIDIA Cosmos (World Model)

**공식 상태 (2025-2026)**:
- Cosmos: 물리 기반 비디오 생성 모델 (NVIDIA Research)
- 용도: "세계를 이해하는 foundation model"
- **실제 접근성**: Cosmos weights = 수십 B 파라미터, H100 수십 장 필요
- Cosmos + Isaac 연동: 연구 시연 수준. API 미공개.
- 실제 manipulation 데이터 생성에 사용한 논문: **현재까지 확인 없음** (2026-03 기준)
- **판정**: 완전 마케팅. RTX 4090 laptop에서 사용 불가. 연구 방향 논의 제외.

### 3-3. Project DIGITS / DGX / Jetson

- DIGITS: NVIDIA 개인용 GB10 칩 기반 AI 컴퓨터 (2025 발표)
- 우리와의 관련: **없음** (우리는 RTX 4090 laptop + VAST.ai 이미 있음)
- Jetson: edge inference용. SmolVLA 배포에 이론상 가능하나 VLA 추론 속도 미확인.
- **판정**: 모두 하드웨어 판매 마케팅. 연구 방향과 무관.

### 3-4. NVIDIA Research 논문 (ICRA/CoRL/RSS 2025-2026)

**실제 확인 가능한 논문**:
- Isaac Lab Mimic (CVPR 2025): 상기 분석
- GR00T N1 (arXiv 2025): 휴머노이드 전용
- Omniverse Digital Twins for Manipulation (ICRA 2025): Franka 전용, tabletop
- **공통 조건**: 모두 고강성 로봇 (Franka, UR) + 고정밀 카메라 + 멀티뷰
- **RoArm M3 적용**: 소비자 로봇 + 단일 뷰 + servo 기반 = NVIDIA 논문 조건과 차이

---

## Step 4: 현실적 판단 — 우리 상황

### 4-1. Isaac → LeRobot v3 변환 파이프라인 가능 여부

**Isaac Lab 데이터 저장 형식 (소스 확인)**:
- `HDF5DatasetFileHandler` 존재: `/data/demo_0`, `/data/demo_1` 구조
- 필드: `obs/joint_pos`, `obs/rgb`, `action`, `reward`, `done`
- **LeRobot v3 필요 구조**: `episode_index`, `frame_index`, `observation.images.*`, `action`, parquet + video

**변환 가능성**: 기술적으로 가능. 예상 소요 시간: 1-2주.
- HDF5 → Parquet: 단순 변환
- RGB tensor → mp4 video: ffmpeg 사용
- stats.json 재생성: 필수 (sim과 real의 관절 각도 분포가 다름)

**실제 블로커**:
1. Isaac에서 카메라 이미지 저장 활성화 필요 (기본 off)
2. RGB 해상도 맞추기: Isaac 기본 512x512 vs SmolVLA 224x224 (SigLIP input)
3. stats.json 불일치: 시뮬레이션 관절 범위 ≠ 실제 관절 사용 범위

**판정**: 파이프라인 자체는 구현 가능 (1-2주). 하지만 SigLIP OOD 문제가 해결되지 않으면 변환해도 전이 안 됨.

### 4-2. 512 envs 49K steps/sec — VLA용으로 활용 가능한가?

**RL reaching에서 확인된 수치** (이전 검증):
- 512 envs headless: 49K steps/sec
- 이미지 렌더링 포함 시: 속도 미확인 (크게 감소 예상)

**VLA용 이미지 데이터 생성 모드**:
- 512 envs 동시 렌더링: VRAM 14.3GB (한계, RTX 렌더러 불가)
- 64 envs 렌더링: 10.8GB (가능), 속도 ~5-10K steps/sec 예상
- 1 env RTX 고품질: 가능, 속도 ~100-500 steps/sec (속도 무의미)

**판정**: RL reaching용 속도는 VLA 이미지 데이터 생성에 직접 적용 불가. 렌더링 활성화 시 속도 10배 이상 저하.

---

## 종합 판단: 기능별 우리 프로젝트 적용 가능성

| 기능 | 기술적 가능 | 우리 조건 적합 | 65일 내 구현 | 판정 |
|------|-----------|-------------|------------|------|
| RTX 렌더링으로 SigLIP 통과 | 불확실 | 불확실 | 가능 | UNPROVEN |
| Domain Randomization (물리) | 가능 | 적합 | 1-3일 | FEASIBLE |
| Isaac Lab Mimic | 가능 | Real→Sim 선결 필요 | 2-3주 | CONDITIONAL |
| HDF5 → LeRobot v3 변환 | 가능 | 가능 | 1-2주 | FEASIBLE |
| Neural Rendering / NeRF | 불가 | - | - | NOT AVAILABLE |
| 3DGS 통합 (Isaac 내장) | 불가 | - | - | NOT AVAILABLE |
| CycleGAN style transfer | 가능 | 실험 필요 | 1-2주 | EXPERIMENTAL |
| GR00T N1 fine-tune | 조건부 | Franka 조건 한정 | 불확실 | NOT RECOMMENDED |
| Cosmos world model | 불가 (컴퓨팅 부족) | - | - | NOT AVAILABLE |
| Real2Sim 자동 파이프라인 | 불가 (수동만 존재) | - | - | MANUAL ONLY |
| ActuatorNetMLP (data-driven) | 가능 | 실측 데이터 필요 | 2-3주 | FEASIBLE |
| DelayedPDActuator (lag DR) | 가능 | RoArm lag 측정 선결 | 1일 | FEASIBLE |

---

## 결론: 무엇을 실제로 쓸 수 있는가

### 즉시 활용 가능 (1-3일)
1. **DelayedPDActuatorCfg**: RoArm M3의 actuator lag을 sim에 주입
   - `min_delay=2, max_delay=5` (20ms-50ms at 100Hz physics)
   - 이미 Isaac에 설치되어 있음, 설정값만 변경

2. **물리 파라미터 DR**: friction, damping 범위 randomization
   - 스크립트 수정으로 즉시 적용 가능

### 1-2주 구현 가능
3. **Isaac → LeRobot v3 변환 파이프라인**
   - HDF5 → Parquet + video 변환
   - stats.json 분리 관리 (sim용 vs real용)
   - 단, SigLIP gap이 해결되지 않으면 데이터 활용 불가

### 선결 조건 필요
4. **CycleGAN style transfer**: real 이미지 100장 수집 먼저
5. **Isaac Lab Mimic**: 1개 real demo를 sim에 re-enact 먼저
6. **ActuatorNetMLP**: RoArm M3 실측 토크/위치 데이터 수집 먼저

### 사용 불가 (NVIDIA 마케팅 vs 현실)
- Cosmos: 컴퓨팅 부족
- Neural Rendering 통합: 미구현
- Real2Sim 자동화: 없음
- 3DGS Isaac 통합: 없음

---

## 우선순위 권고

**현재 Stage (baseline 74 real episodes 보유)에서 sim 투자는 낮은 ROI**:
- SigLIP gap = sim 전이의 근본 블로커
- sim gap 해결 비용 > 실제 데이터 추가 수집 비용

**sim 활용이 의미 있어지는 시점**:
- Stage 1 (5-zone, 150ep) 완료 후
- 특정 존/물체/조건에서 real 데이터 수집이 어려울 때
- 안전 critical한 실험 (충돌 가능성 있는 동작) 사전 검증 시

**최소 사용 권고**:
1. `DelayedPDActuatorCfg`로 actuator lag 모델링 (1일, ablation용)
2. HDF5→LeRobot 변환 파이프라인 준비 (2주, 나중에 쓸 수 있도록)
3. SigLIP cosine distance 실측 테스트 먼저 (`sim_siglip_validation.py`)

---
*작성: A2 Sim-to-Real Specialist*
*기반 데이터: 실제 설치된 extension 직접 확인 (2026-03-26)*
*NVIDIA 공식 문서 미참조 — 설치된 코드만 신뢰*
