# NVIDIA Omniverse / Isaac Ecosystem -- "Hidden Gems" Deep Dive

> 조사일: 2026-03-26
> 목적: 로봇 manipulation + VLA 학습에 활용 가능한 NVIDIA 도구 전수조사
> 우리 환경: RTX 4090 Laptop, Ubuntu 22.04, RoArm-M3, SmolVLA/LeRobot

---

## 요약: 우리에게 가장 유용한 Top 10

| 순위 | 도구 | 활용 | 즉시 사용? |
|------|------|------|-----------|
| 1 | **Newton 1.0** | MuJoCo 대체, 475x 빠른 manipulation sim | RTX 4090 OK |
| 2 | **Isaac Lab-Arena** | VLA policy sim 평가 (Libero/RoboCasa 벤치마크 통합) | RTX 4090 OK |
| 3 | **GR00T-Mimic** | 소수 데모 -> 대량 합성 궤적 생성 | RTX 4090 OK |
| 4 | **Cosmos Transfer 2.5** | sim 데이터 photorealistic 변환 (domain gap 축소) | RTX 4090 OK |
| 5 | **cuRobo/cuMotion** | GPU 가속 모션 플래닝 (30ms 궤적 생성) | RTX 4090 OK |
| 6 | **Omniverse Replicator** | 합성 데이터 파이프라인 (카메라 뷰/조명 랜덤화) | RTX 4090 OK |
| 7 | **NuRec (3DGS)** | 실제 작업대 -> OpenUSD sim 환경 변환 | RTX 4090 OK |
| 8 | **FoundationPose** | 6D 물체 포즈 추정 (novel object, zero-shot) | RTX 4090 OK |
| 9 | **Eureka/DrEureka** | LLM으로 reward function 자동 생성 | RTX 4090 OK |
| 10 | **Isaac Lab ADR+PBT** | 자동 도메인 랜덤화 + population-based training | RTX 4090 OK |

---

## 1. Isaac Sim 5.0 / 5.1

### 무엇인가
GPU 가속 로봇 시뮬레이션 플랫폼. PhysX 5 기반, OpenUSD 네이티브.

### 5.0 (2025 중반, SIGGRAPH) 주요 신기능
- **OmniSensor USD Schema**: 센서를 USD 프리미티브로 정의 (RTX Lidar, Radar, Depth 카메라)
- **Neural Rendering (NuRec)**: 3D Gaussian Splatting으로 실제 환경 -> sim 변환
- **오픈소스 전환**: Isaac Sim extensions GitHub 공개
- **MJCF <-> OpenUSD 변환**: MuJoCo 사용자 25만명의 자산 호환

### 5.1 (2025.10) 주요 신기능
- **DGX Spark 지원**: GB10 칩에서 최적화
- **Gripper collision 개선**: articulation collision을 마지막에 풀어 그리퍼 관통 방지
- **Joint parameter tuning tutorial**: 로봇 그리퍼 튜닝 가이드 추가

### 우리에게 유용한 점
- RoArm-M3의 URDF를 Isaac Sim에 임포트 -> sim 환경 구축 가능
- OmniSensor로 Azure Kinect와 동일한 뷰포인트의 sim 카메라 배치
- 성숙도: **Production (GA)**
- RTX 4090: **OK** (최소 8GB VRAM, 권장 16GB)

---

## 2. Isaac Lab 2.2 / 2.3

### 무엇인가
Isaac Sim 위의 오픈소스 RL/IL 학습 프레임워크. 구 Orbit의 후속.

### 2.2 (2026.01) 주요 신기능
- **GR00T-Mimic 통합**: 합성 모션 생성 (소수 데모 -> 대량 궤적)
- **Omniverse Fabric 통합**: 로드 타임 단축 + 물리/센서 효율 향상
- **Tensorized suction cup gripper**: 흡착 그리퍼 시뮬레이션

### 2.3 (2025.11) 주요 신기능 ★★★
- **Automatic Domain Randomization (ADR)**: 물리 파라미터를 자동으로 점진 확대
- **Population Based Training (PBT)**: 여러 정책 병렬 학습 + 상위 선택
- **Dictionary observation space**: perception + proprioception 분리 (VLA에 적합!)
- **Dexterous manipulation env**: Kuka+Allegro 기본 제공
- **Dexterous retargeting**: 사람 손 -> 로봇 손 매핑 (텔레옵)

### 우리에게 유용한 점
- ADR: sim 데이터 다양성 자동 확보 (수동 랜덤화 불필요)
- Dictionary obs: SmolVLA의 image + state 분리 입력과 자연스럽게 매핑
- 성숙도: **Production (GA)**
- RTX 4090: **OK**

---

## 3. Newton Physics Engine 1.0 ★★★ (Hidden Gem)

### 무엇인가
NVIDIA + Google DeepMind + Disney Research가 공동 개발한 오픈소스 GPU 물리 엔진. Warp 기반.

### 왜 "숨겨진 보석"인가
- **GTC 2026 (2026.03.16)에 1.0 GA 발표** -- 10일 전!
- MuJoCo MJX 대비 **475x 빠름** (RTX PRO 6000 기준)
- Linux Foundation 관리 -> 오픈소스 보장
- 기존 Isaac Lab/Isaac Sim과 통합

### 핵심 기능
- **Vertex Block Descent solver**: 케이블, 천, 변형체 시뮬레이션
- **SDF collision library**: signed distance field 기반 충돌 감지
- **Hydroelastic contact modeling**: 접촉 역학 정밀 모델링
- **Deformable simulation**: 스펀지 같은 변형 물체 grasping sim 가능!

### 우리에게 유용한 점
- 스펀지 잡기 태스크의 sim 데이터 생성에 직접 활용 가능
- 케이블 manipulation 연구 확장 시 필수
- Skild AI가 GPU 랙 조립에, Samsung이 냉장고 조립 라인에 사용 중
- 성숙도: **Production (1.0 GA, 2026.03.16)**
- RTX 4090: **OK** (Warp 기반, CUDA 호환)

---

## 4. GR00T-Mimic ★★★ (Hidden Gem)

### 무엇인가
소수 텔레옵 데모에서 대량 합성 모션 궤적을 생성하는 NIM 마이크로서비스.

### 왜 "숨겨진 보석"인가
- 780K 합성 궤적을 11시간 만에 생성 (= 6,500시간 인간 데모)
- 실제 데이터 + 합성 데이터 결합 시 **성능 40% 향상** (GR00T N1 기준)
- DexMimicGen (2025.05): bimanual dexterous manipulation 확장

### 동작 방식
1. 소수 텔레옵 데모 녹화 (Apple Vision Pro 등)
2. Isaac Sim에서 시뮬레이션 재생
3. Key point annotation + interpolation으로 궤적 확장
4. Cosmos Transfer로 RGB 비주얼 photorealistic 변환

### 우리에게 유용한 점
- **Stage 1 데이터 부족 문제 직접 해결**: 74ep -> 수천 합성 궤적
- RoArm-M3 URDF를 sim에 넣으면 바로 사용 가능
- 단, humanoid 중심 설계 -> arm manipulation에 adaptation 필요
- 성숙도: **Production (Blueprint 공개)**
- RTX 4090: **OK** (추론/생성은 로컬, 대규모는 클라우드 권장)

---

## 5. Cosmos World Foundation Models ★★

### 무엇인가
텍스트/이미지/비디오 입력 -> 물리적으로 정확한 가상 세계 생성.

### 주요 모델 (2025-2026)
| 모델 | 역할 | 출시 |
|------|------|------|
| Cosmos Predict 2.5 | Text/Image/Video -> 미래 예측 비디오 | 2025.12 |
| Cosmos Transfer 2.5 | sim -> photorealistic 스타일 변환 | 2025.12 |
| Cosmos Reason 2 | 물리 상식 + 공간/시간 reasoning VLM | 2026.02 |
| Cosmos Curator | 학습 데이터 자동 큐레이션 | 2026.03 |
| Cosmos Evaluator | 생성 데이터 품질 자동 평가 | 2026.03 |

### 우리에게 유용한 점
- **Cosmos Transfer**: sim에서 렌더링한 작업대 이미지 -> 실제처럼 변환 (sim2real gap)
- **Cosmos Reason 2**: 로봇 행동 계획 reasoning (어디를 잡을지, 어떤 순서로)
- **Cosmos Curator + Evaluator**: 수집한 데이터 자동 품질 검증
- Skild AI, Figure AI, Uber 등 이미 채택
- 성숙도: **Production** (Predict/Transfer), **Beta** (Reason 2, Curator, Evaluator)
- RTX 4090: Transfer/Predict 추론은 가능, 학습은 클라우드 필요

---

## 6. cuRobo / cuMotion ★★

### 무엇인가
CUDA 가속 로봇 모션 플래닝 라이브러리. FK/IK, 충돌 체크, 궤적 최적화 통합.

### 핵심 성능
- CPU 대비 **60x 빠른** 궤적 생성 (100ms 이내)
- Jetson Orin NX에서도 ~100ms, **500Hz re-optimization** 가능
- v0.7.6 (2024.11): 포즈 도달 정확도 **10x 향상**
- v0.7.5: Grasp API 추가

### 우리에게 유용한 점
- **RoArm-M3 (6-DOF) 궤적 플래닝**: SmolVLA 대신 cuRobo로 safety layer 구현 가능
- nvblox와 연동: 깊이 카메라 -> 3D 장애물 맵 -> 충돌 회피 궤적
- grasping pipeline의 motion planning 단계에 직접 활용
- 성숙도: **Production**
- RTX 4090: **OK** (Isaac ROS cuMotion 패키지로 ROS 2 연동)

---

## 7. Omniverse Replicator

### 무엇인가
합성 데이터 생성 프레임워크. 카메라 뷰, 조명, 재질, 배치를 프로그래밍으로 랜덤화.

### 6대 컴포넌트
1. **Semantic Schema Editor**: 라벨링 스키마 정의
2. **Visualizer**: 생성 결과 시각화
3. **Randomizers**: 포즈/조명/재질/카메라 랜덤화
4. **Omni.syntheticdata**: 데이터 생성 API
5. **Annotators**: bbox, segmentation, depth 등 자동 라벨링
6. **Writers**: 다양한 포맷으로 출력

### 우리에게 유용한 점
- **카메라 뷰포인트 augmentation**: 실제 데이터 74ep의 카메라 각도를 sim에서 변주
- **조명 randomization**: VLA 로버스트니스 향상
- **물체 배치 randomization**: 5-zone 데이터 sim에서 대량 생성
- 성숙도: **Production**
- RTX 4090: **OK**

---

## 8. NuRec (Neural Reconstruction) ★★★ (Hidden Gem)

### 무엇인가
실제 카메라/LiDAR 데이터 -> 3D Gaussian Splatting -> OpenUSD sim 환경 자동 변환.

### 왜 "숨겨진 보석"인가
- **Real-to-Sim의 가장 빠른 경로**: 작업대를 스마트폰으로 촬영 -> Isaac Sim 환경 생성
- RTX ray tracing + 3DGS 결합 -> photorealistic + 물리 시뮬레이션 가능
- CARLA(AV sim)에서 15만 개발자가 사용, 로봇 분야는 아직 초기

### 동작 방식
1. 실제 환경 촬영 (RGB or RGB-D)
2. NuRec가 3DGS 재구성
3. USDZ 파일로 출력
4. Isaac Sim에서 물리 시뮬레이션 가능 환경으로 활용

### 우리에게 유용한 점
- **작업대 디지털 트윈**: Azure Kinect로 작업대 촬영 -> sim 환경 자동 생성
- sim에서 카메라 위치/물체 배치 변경 -> VLA 학습 데이터 대량 생성
- 기존 3DGS 연구와 직접 연결 가능
- 성숙도: **Production (GA, 2026 초)**
- RTX 4090: **OK** (3DGS 재구성 + 렌더링 모두 가능)

---

## 9. FoundationPose / FoundationStereo / FoundationGrasp

### FoundationPose
- **무엇**: 6D 물체 포즈 추정 + 트래킹 (novel object, zero-shot)
- **입력**: CAD 모델 or RGB 참조 이미지 몇 장
- **성숙도**: Production (CVPR 2024 Best Paper 후보)

### FoundationStereo
- **무엇**: 스테레오 깊이 추정 foundation model
- **특징**: 실내/실외/합성/실제 모두 zero-shot 일반화
- **성숙도**: Production (CVPR 2025 Best Paper 후보)

### FoundationGrasp
- **무엇**: transformer 기반 dense grasp prediction (미지 3D 물체)
- **성숙도**: Beta

### 우리에게 유용한 점
- **FoundationPose**: 스펀지 포즈 추정 -> 성공/실패 자동 판별 (평가 자동화)
- **FoundationStereo**: ZED Mini 스테레오 깊이를 foundation model로 개선
- cuMotion + FoundationPose = Isaac Manipulator 파이프라인
- RTX 4090: **OK** (Isaac Manipulator 패키지 내 포함)

---

## 10. Eureka / DrEureka ★★ (Hidden Gem)

### 무엇인가
GPT-4 등 LLM으로 RL reward function을 자동 생성하는 AI 에이전트.

### Eureka (ICLR 2024)
- 인간 전문가 대비 **83% 태스크에서 우위**, 평균 **52% 개선**
- task-specific 프롬프트 불필요, 환경 소스코드만 제공하면 됨
- 10개 로봇 형태, 29개 환경에서 검증

### DrEureka (2025)
- **Domain Randomization도 자동 생성**: reward + DR 파라미터 동시
- sim2real transfer 인간 설계 대비 우수

### 우리에게 유용한 점
- Isaac Lab에서 RoArm-M3 환경 구축 후, Eureka로 reward 자동 생성
- manipulation 태스크별 reward 수동 설계 부담 제거
- 성숙도: **Research** (GitHub 공개, 논문 발표)
- RTX 4090: **OK** (Isaac Lab 환경 실행, LLM은 API 호출)

---

## 11. Physical AI Data Factory Blueprint ★★★ (Hidden Gem)

### 무엇인가
학습 데이터 생성/증강/평가를 자동화하는 오픈 레퍼런스 아키텍처.

### 왜 "숨겨진 보석"인가
- **GTC 2026 (2026.03.16)에 발표** -- 10일 전!
- 3단계 파이프라인: Cosmos Curator -> Cosmos Transfer -> Cosmos Evaluator
- OSMO로 전체 워크플로우 오케스트레이션
- **Claude Code 연동** 공식 지원!
- GitHub 공개 예정: 2026년 4월

### 우리에게 유용한 점
- 데이터 수집 -> 큐레이션 -> 증강 -> 평가를 원스톱으로 자동화
- 실제 74ep 데이터를 seed로 대규모 합성 데이터 생성
- Skild AI, Uber, Teradyne Robotics 등이 얼리 어답터
- 성숙도: **Beta** (2026.04 GitHub 공개 예정)
- RTX 4090: 로컬 테스트 OK, 대규모는 클라우드

---

## 12. OSMO (Orchestration)

### 무엇인가
Physical AI 파이프라인 오케스트레이션 플랫폼. YAML 기반.

### 핵심 가치
- 학습 + 시뮬레이션 + 엣지 테스트를 단일 워크플로우로 관리
- 로컬/클라우드/온프렘 동일 워크플로우 실행
- **Claude Code 연동** 공식 지원 (2026.03)

### 우리에게 유용한 점
- VAST.ai 클라우드 학습 + 로컬 RTX 4090 추론을 하나의 파이프라인으로
- 성숙도: **Production (오픈소스)**
- RTX 4090: **OK** (로컬 노드로 활용)

---

## 13. Isaac Cortex (Behavior Trees)

### 무엇인가
Isaac Sim 내 행동 트리 기반 로봇 태스크 플래닝 프레임워크.

### 핵심 기능
- State machine + behavior tree 기반 반응형 로봇 행동
- Perception -> World model -> Skill selection 파이프라인
- collision avoidance policy 내장

### 우리에게 유용한 점
- VLA policy의 safety wrapper로 활용 가능
- Stage 3 (연속 동작) 이후 태스크 플래닝에 유용
- 성숙도: **Production** (Isaac Sim 5.0+)
- RTX 4090: **OK**

---

## 14. nvblox (3D Reconstruction)

### 무엇인가
GPU 가속 실시간 3D 복셀 맵 + ESDF(Euclidean Signed Distance Field) 생성.

### 핵심 기능
- RGB-D 카메라 -> 실시간 3D 장애물 맵
- **Deep feature fusion**: vision foundation model 피처를 3D 맵에 융합
- 멀티 카메라 지원 (2025)

### 우리에게 유용한 점
- Azure Kinect depth -> 3D 작업 공간 맵 -> cuMotion 충돌 회피
- language-guided manipulation 연구 시 semantic 3D map 활용
- 성숙도: **Production** (Isaac ROS 패키지)
- RTX 4090: **OK**

---

## 15. Warp + Newton

### Warp
- NVIDIA의 GPU 프로그래밍 프레임워크 (Python -> CUDA JIT)
- PyTorch/JAX 미분 가능 연동
- CPU 대비 **669x 속도** (일부 케이스)
- 로봇 물리 시뮬레이션 커스텀 커널 작성 가능

### Newton 1.0 (2026.03 GA)
- Warp 기반 물리 엔진
- MJX 대비 **475x 빠름**
- 변형체(deformable) + 접촉 역학 특화

### 우리에게 유용한 점
- 커스텀 물리 시뮬레이션 필요 시 Warp으로 직접 작성
- Newton으로 스펀지 변형 + 접촉 시뮬레이션
- 성숙도: Warp **Production** (v1.12), Newton **Production** (v1.0 GA)
- RTX 4090: **OK**

---

## 16. RTX Sensor Simulation

### RTX Lidar
- GPU ray tracing 기반 LiDAR 시뮬레이션
- 반사 재질, 다양한 조명 조건 대응
- ROS 2 LaserScan/PointCloud2 출력

### RTX Radar
- 레이더 센서 시뮬레이션

### Depth Camera (새 모델, Isaac Sim 5.0+)
- **Stereo disparity artifact 시뮬레이션**: 실제 스테레오 카메라처럼 disparity 노이즈 재현
- OmniSensor USD Schema로 USD 내 직접 정의

### 우리에게 유용한 점
- Azure Kinect의 depth 특성(NFOV, 노이즈 패턴)을 sim에서 재현
- 성숙도: **Production**
- RTX 4090: **OK**

---

## 17. GR00T N1 / N1.6 / N1.7 / N2

### 무엇인가
NVIDIA의 오픈 VLA foundation model (원래 humanoid 타겟, arm manipulation도 가능).

### 모델 계보
| 버전 | 시기 | 특징 |
|------|------|------|
| N1 | 2025.03 | 최초 공개, dual-system (VLM + diffusion transformer) |
| N1.6 | 2026.01 | 성능 개선, full body control, Cosmos Reason 통합 |
| N1.7 | 2026.03 | 상용 라이선스, early access |
| N2 (preview) | 2026.03 GTC | DreamZero 기반, 새 환경/태스크 성공률 2x |

### 우리에게 유용한 점
- SmolVLA/OpenVLA 외에 **비교 대상 VLA**로 활용 가능
- 단, humanoid 중심 설계 -> 6-DOF arm 직접 사용에는 adaptation 필요
- Isaac Lab-Arena에서 벤치마크 가능
- 성숙도: **Production** (N1.6 GA), **Beta** (N1.7 EA, N2 preview)
- RTX 4090: N1 추론은 가능 (2-3B 파라미터 추정), 학습은 클라우드

---

## 18. Isaac Lab-Arena ★★ (Hidden Gem)

### 무엇인가
GPU 가속 정책 평가 프레임워크. Libero, RoboCasa, RoboTwin 벤치마크 통합.

### 왜 "숨겨진 보석"인가
- **sim에서 VLA 정책 평가 자동화**: 실제 로봇 없이 수백 에피소드 평가
- LeRobot 공식 연동 (HuggingFace 블로그 발표)
- Lightwheel과 공동 개발

### 우리에게 유용한 점
- SmolVLA 체크포인트를 sim에서 먼저 평가 -> 실제 배포 전 스크리닝
- 다양한 물체 배치, 조명에서의 성공률 자동 측정
- 성숙도: **Production** (CES 2026 발표, GitHub 공개)
- RTX 4090: **OK**

---

## 19. RoboCasa365 / GenSim2

### RoboCasa365 (2026.02)
- 365개 일상 태스크, 2,500+ 주방 환경, 2,200+ 시간 데모 데이터
- mobile manipulator + humanoid + quadruped 지원
- Omniverse 고품질 렌더링

### GenSim2
- LLM으로 sim 태스크 자동 생성 (100개 articulated 태스크, 200 오브젝트)
- 실제 데이터와 co-training 시 **20% 성능 향상**

### 우리에게 유용한 점
- 테이블탑 manipulation 환경으로 활용 가능
- 다양한 물체/환경 생성 -> VLA 일반화 학습
- 성숙도: **Production** (RoboCasa), **Research** (GenSim2)
- RTX 4090: **OK** (테이블탑 환경은 가벼움)

---

## 20. PhysX 5 Deformable/Soft Body

### 무엇인가
PhysX 5에 통합된 FEM(Finite Element Method) 기반 변형체 시뮬레이션.

### 핵심 기능
- **FEM soft body**: 스펀지, 고무 등 변형 물체
- **PBD (Position Based Dynamics)**: 액체, 천, 풍선
- **Two-way coupling**: 강체-변형체 상호작용
- 34개 물체, 6,800 grasp 평가, 1.1M grasp 측정 데이터셋 공개

### 우리에게 유용한 점
- 스펀지 잡기의 물리적 정확도 향상
- deformable object grasping sim 데이터 생성
- 성숙도: **Production** (PhysX 5.5)
- RTX 4090: **OK** (GPU 가속 필수)

---

## 21. NIM (Inference Microservices)

### 무엇인가
AI 모델 추론 최적화 + 배포 마이크로서비스. 컨테이너 기반.

### 핵심 성능
- H100 대비 **2.6x throughput** (Llama 3.1 8B)
- 5분 이내 배포 (단일 컨테이너)
- DGX/RTX/Jetson 어디서나 동일 코드

### 우리에게 유용한 점
- SmolVLA/VLA 추론을 NIM으로 최적화 -> 배포 latency 감소
- 클라우드 -> 엣지 전환 시 코드 변경 불필요
- 성숙도: **Production**
- RTX 4090: **OK**

---

## 22. DGX Spark / Jetson Thor (하드웨어)

### DGX Spark
- 데스크탑 AI 슈퍼컴퓨터, GB10 칩, **1 PFLOP**, 128GB HBM3e
- 모델 개발/프로토타이핑용

### Jetson Thor
- 엣지 AI 플랫폼, **2070 TFLOPS**, 40-130W
- VLA 모델 실시간 추론 (LLM/VLM/VLA 네이티브)
- Orin 대비 **3.5x 에너지 효율**

### 우리에게 유용한 점
- 현재 RTX 4090 Laptop으로 충분하지만, 향후 배포 시 Jetson Thor 고려
- 성숙도: DGX Spark **Production**, Jetson Thor **Pre-order** (2026 중반 출하 예상)

---

## 우리 프로젝트에 적용 가능한 워크플로우

### 즉시 적용 가능 (Stage 1과 병행)
```
1. NuRec: 작업대 촬영 -> OpenUSD sim 환경 생성 (반나절)
2. Omniverse Replicator: 물체 배치/조명 randomization (반나절)
3. GR00T-Mimic: 소수 실제 데모 -> 합성 궤적 대량 생성 (하루)
4. Cosmos Transfer: sim 이미지 -> photorealistic 변환 (수시간)
5. Isaac Lab-Arena: SmolVLA 체크포인트 sim 평가 (수시간)
```

### 중기 적용 (Stage 2+)
```
6. Newton 1.0: 변형체(스펀지) 정밀 sim -> sim2real gap 축소
7. cuRobo: safety layer (관절 한계 + 충돌 회피)
8. Eureka: RL reward 자동 설계 -> manipulation policy 보조
9. FoundationPose: 물체 포즈 자동 평가 -> 성공률 자동 측정
10. Physical AI Data Factory: 전체 파이프라인 자동화
```

### 논문/연구 관점 (Stage 2+ 달성 후)
```
- NuRec + Cosmos Transfer: sim2real gap 정량 비교 논문 소재
- Newton deformable: 변형 물체 grasping sim2real 연구
- Isaac Lab-Arena + LeRobot: VLA 벤치마크 평가 자동화 논문
- ADR+PBT: 자동 도메인 랜덤화로 데이터 효율 연구
```

---

## Sources

### Isaac Sim / Isaac Lab
- [Isaac Sim 5.1 Release Notes](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html)
- [Isaac Sim 5.0 & Isaac Lab 2.2 GA Announcement](https://developer.nvidia.com/blog/isaac-sim-and-isaac-lab-are-now-available-for-early-developer-preview/)
- [Isaac Lab 2.3 Blog](https://developer.nvidia.com/blog/streamline-robot-learning-with-whole-body-control-and-enhanced-teleoperation-in-nvidia-isaac-lab-2-3/)
- [Isaac Lab Paper (2025)](https://research.nvidia.com/publication/2025-09_isaac-lab-gpu-accelerated-simulation-framework-multi-modal-robot-learning)

### Newton Physics
- [Newton 1.0 GA Blog](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/)
- [Newton GitHub](https://github.com/newton-physics/newton)
- [Newton Developer Page](https://developer.nvidia.com/newton-physics)

### GR00T
- [GR00T N1 Paper (arXiv)](https://arxiv.org/abs/2503.14734)
- [GR00T N1 Announcement](https://nvidianews.nvidia.com/news/nvidia-isaac-gr00t-n1-open-humanoid-robot-foundation-model-simulation-frameworks)
- [GR00T N1.6 GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T-Mimic Blueprint](https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation)

### Cosmos
- [Cosmos Platform](https://www.nvidia.com/en-us/ai/cosmos/)
- [Cosmos Reason 2 GitHub](https://github.com/nvidia-cosmos/cosmos-reason2)
- [Cosmos Synthetic Data Blog](https://developer.nvidia.com/blog/scale-synthetic-data-and-physical-ai-reasoning-with-nvidia-cosmos-world-foundation-models/)
- [Cosmos Reason 2 HuggingFace Blog](https://huggingface.co/blog/nvidia/nvidia-cosmos-reason-2-brings-advanced-reasoning)

### cuRobo / cuMotion / Isaac Manipulator
- [cuRobo Website](https://curobo.org/)
- [cuRobo GitHub](https://github.com/NVlabs/curobo)
- [cuMotion GitHub](https://github.com/nvidia-isaac/cumotion)
- [Isaac Manipulator](https://developer.nvidia.com/isaac/manipulator)

### Omniverse / Replicator / NuRec
- [Omniverse Replicator Docs](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html)
- [NuRec Docs](https://docs.nvidia.com/nurec/index.html)
- [NuRec GA Announcement](https://forums.developer.nvidia.com/t/nvidia-omniverse-nurec-now-generally-available/364767)
- [Omniverse Kit 110.0](https://docs.omniverse.nvidia.com/dev-guide/latest/release-notes/110_0_highlights.html)

### Eureka / DrEureka
- [Eureka Paper (ICLR 2024)](https://arxiv.org/abs/2310.12931)
- [Eureka GitHub](https://github.com/eureka-research/Eureka)
- [DrEureka VentureBeat](https://venturebeat.com/automation/nvidias-dreureka-outperforms-humans-in-training-robotics-systems/)

### Perception
- [FoundationPose / FoundationStereo Blog](https://developer.nvidia.com/blog/r2d2-building-ai-based-3d-robot-perception-and-mapping-with-nvidia-research/)
- [nvblox Isaac ROS](https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/index.html)

### Data Factory / OSMO
- [Physical AI Data Factory Blueprint](https://nvidianews.nvidia.com/news/nvidia-announces-open-physical-ai-data-factory-blueprint-to-accelerate-robotics-vision-ai-agents-and-autonomous-vehicle-development)
- [OSMO GitHub](https://github.com/NVIDIA/OSMO)
- [OSMO Developer Page](https://developer.nvidia.com/osmo)

### Isaac Lab-Arena
- [Isaac Lab-Arena Blog](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena/)
- [Isaac Lab-Arena + LeRobot (HuggingFace)](https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot)

### RoboCasa / GenSim2
- [RoboCasa GitHub](https://github.com/robocasa/robocasa)
- [RoboCasa Website](https://robocasa.ai/)
- [GenSim2 Paper](https://www.researchgate.net/publication/384680462_GenSim2_Scaling_Robot_Data_Generation_with_Multi-modal_and_Reasoning_LLMs)

### GTC / Industry
- [GTC 2025 Keynote Updates](https://blogs.nvidia.com/blog/nvidia-keynote-at-gtc-2025-ai-news-live-updates/)
- [GTC 2026 Live Updates](https://blogs.nvidia.com/blog/gtc-2026-news/)
- [GTC 2026 Robotics Highlights](https://theaiinsider.tech/2026/03/21/10-robotics-highlights-from-nvidia-gtc-2026/)
- [NVIDIA CES 2026 Robotics Announcements](https://forums.developer.nvidia.com/t/nvidia-robotics-announcements-ces-2026/356606)

### Warp / PhysX
- [Warp GitHub](https://github.com/NVIDIA/warp)
- [Warp Docs](https://nvidia.github.io/warp/)
- [PhysX 5.5 Docs](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/index.html)

### NIM / Hardware
- [NIM Developer](https://developer.nvidia.com/nim)
- [Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
- [DGX Spark Optimizations](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/)
