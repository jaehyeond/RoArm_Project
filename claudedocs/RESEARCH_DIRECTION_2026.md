# 로봇 연구 방향 종합 분석 (2026년 3월)

교수님 피드백: "언제까지 로봇팔로 스펀지만 잡고 있을거냐?"
목표: 현실적이면서 학술적 가치 있는 연구 주제 선정

---

## Part 1: 현재 로봇 분야 전체 지도

### 1.1 학회 트렌드 (2025-2026)

| 순위 | 영역 | 추세 | 근거 |
|------|------|------|------|
| 1 | **VLA (Vision-Language-Action)** | 폭발적 성장 | RSS 2025 전용 세션 10편, pi0→pi0.5→pi0.6 |
| 2 | **휴머노이드** | 폭발적 성장 | RSS 2025 전용 세션 10편, 산업계 투자 급증 |
| 3 | **데이터 인프라/스케일링** | 급성장 | MuJoCo Playground(RSS Award), RoboCrowd, ReMix |
| 4 | **Diffusion/Flow Matching** | 성숙+분화 | DP3, ReinFlow(NeurIPS 2025), 기본 패러다임화 |
| 5 | **3D Vision (3DGS)** | 급성장 | Re3Sim, SplaTAM, 451 GitHub repos |
| 6 | **Safety/Formal Methods** | 급부상 | RSS 2025 Outstanding Paper = multi-agent safety |
| 7 | **Tactile Sensing** | 성장 | Octopi-1.5, Reactive Diffusion Policy |
| 8 | **LLM for Planning** | 안정화 | 2022-23 hype 지나고 실용화 단계 |
| 9 | **Sim-to-Real** | 성숙 | Isaac Lab, MuJoCo, real-to-sim이 새 방향 |
| 10 | **단일 task RL from scratch** | 하락 | Foundation model fine-tuning으로 대체 중 |

**핵심 관찰**: "아키텍처 혁신"보다 "데이터 스케일링"이 더 중요해지고 있음. 여러 수상 논문이 모델이 아닌 데이터에 관한 것.

### 1.2 산업계 동향

| 회사 | 전략 | 핵심 제품 | 실제 배포 여부 |
|------|------|-----------|---------------|
| **NVIDIA** | 플랫폼 | GR00T N1.6, Isaac Lab, Cosmos, Jetson Thor | Isaac Lab은 production. GR00T는 open weights |
| **Google DeepMind** | VLA 최강자 | Gemini Robotics (비공개) | 연구 단계, trusted testers만 |
| **Physical Intelligence** | VLA 스타트업 | pi0→pi0.5→pi0.6 (open-source) | **실제 고객 배포** (Weave, Ultra) |
| **Boston Dynamics** | 엔터프라이즈 | Electric Atlas + DeepMind AI | **2000+ Spot/Stretch 배포**, Atlas 2026 |
| **Unitree** | 저가 하드웨어 | G1 ($16K 휴머노이드) | 쿼드러펙드 대량 판매, 휴머노이드 초기 |
| **Tesla Optimus** | ? | Optimus Gen 3 | **검증 안됨** (텔레오퍼레이션 이벤트, 논문 없음) |
| **Figure AI** | 과대 평가? | Figure 03, Helix VLA | BMW 파트너십 범위 의문 제기됨, $39B 밸류에이션 |

**현실 필터**: 실제로 로봇이 배포되어 돈을 벌고 있는 곳은 **Boston Dynamics**(Spot/Stretch)와 **Physical Intelligence**(파트너사 배포) 뿐. 나머지는 데모 또는 연구 단계.

### 1.3 핵심 기술 수렴: VLA가 기본 패러다임

모든 major player가 VLA 모델을 구축하거나 사용 중:

| 회사 | VLA 모델 | 크기 | 접근성 |
|------|----------|------|--------|
| Physical Intelligence | pi0/pi0.5/pi0.6 | 3B | **Open-source** |
| NVIDIA | GR00T N1.6 | ~2B | **Open weights** |
| Google DeepMind | Gemini Robotics | ? | Closed |
| Figure AI | Helix VLA | ? | Proprietary |
| HuggingFace | SmolVLA | 450M | **Open-source** |
| Stanford | OpenVLA | 7B | **Open-source** |

---

## Part 2: ROS 생태계 & 취업 시장

### 2.1 ROS 현황

- **ROS 1 공식 종료**: 2025년 5월 31일 (보안 패치 중단)
- **ROS 2가 80%**: 전체 ROS 다운로드의 80% (ros.org 공식, 2024.09)
- **현재 LTS**: Jazzy Jalisco (2024.05 ~ 2029.05)
- **주요 변화**: Zenoh(rmw_zenoh_cpp) Tier 1 승격 → DDS 대안, Rust IDL 생성기 기본 포함

### 2.2 취업 시장 (미국 기준, careersinrobotics.com)

| 항목 | 수치 |
|------|------|
| 활성 ROS 관련 채용 | 352개+ |
| 중위 연봉 | $179K (약 2.4억원) |
| 소프트웨어 트랙 평균 | $194K |
| 하드웨어 트랙 평균 | $127K |
| 최고 연봉 기술 | CUDA ($218K), ML ($190K), C++ ($185K) |
| 최고 연봉 회사 | NVIDIA ($270K), Waymo ($232K), Shield AI ($228K) |

**핵심**: 소프트웨어 > 하드웨어 (53% 차이). CUDA + ML + C++ 조합이 가장 고가치.

### 2.3 한국 로봇 시장

| 회사 | ROS 활동 | 비고 |
|------|----------|------|
| **ROBOTIS** | physical_ai_tools (LeRobot + ROS 2), ai_worker | 가장 활발, TurtleBot 제작사 |
| **Doosan Robotics** | doosan-robot2 (ROS 2 Humble), MoveIt 2 | 공식 ROS 2 패키지 |
| **KRAFTON** | VLA 연구자 채용 중 (서울) | 게임→로봇 AI 확장 |
| **Samsung Research** | Physical AI 인턴 (MV, CA) | VLM/VLA, ROS, MuJoCo 요구 |
| **42dot (현대)** | SLAM 엔지니어 채용 | 자율주행 쪽 |
| **Naver Labs** | 공개 ROS 활동 없음 | 독자 스택 추정 |
| **LG** | 공개 ROS 활동 없음 | CLOi 로봇, 독자 스택 |

**주의**: 한국 채용은 원티드/사람인/잡코리아에 집중. 영어 aggregator에서 보이지 않을 뿐 시장이 작은 건 아님.

### 2.4 커리어 패스 비교

| 패스 | 연봉 (미국) | 핵심 기술 | 석사 충분? |
|------|------------|-----------|-----------|
| ROS 소프트웨어 엔지니어 | $160K-$218K | C++, ROS 2, Linux | YES (67% bachelor 충분) |
| ML 로보틱스 연구자 | $176K-$190K | PyTorch, RL, VLA | PhD 선호 (but 석사도 가능) |
| Full-stack 로보틱스 | $161K-$189K | ROS + ML + 배포 | YES (가장 희소, 가장 가치) |

**디지털 리터러시 주의**: 이 연봉은 미국 Bay Area/대도시 기준. 한국 석사 취업 시 4,000-7,000만원 수준으로 크게 다름. PhD ROI가 미국에서 음수라는 데이터(석사 $186K > PhD $180K)도 한국에는 적용 안됨.

---

## Part 3: 소프트웨어 중심 연구 방향 10개 분석

| 순위 | 영역 | 성숙도 | 하이프 위험 | RTX 4090 가능? | 석사 기여 가능? |
|------|------|--------|-----------|---------------|----------------|
| 1 | **3DGS for Robotics** | 급성장 | 중간 | **YES** (분 단위 학습) | **강력 YES** |
| 2 | **Sim-to-Real (Real-to-Sim)** | 성숙+진화 | 낮음 | **YES** (Isaac Lab) | **YES** |
| 3 | **Diffusion/Flow Matching** | 급성장 | 낮음 | **YES** (단일 GPU) | **YES** |
| 4 | **LLM/VLM Planning** | 안정화 | 중간 | **YES** (API 호출) | **YES** |
| 5 | **Data-efficient Learning** | 꾸준 | 낮음 | **YES** | **YES** |
| 6 | **VLA Fine-tuning (small)** | 급성장 | 중-높 | **YES** (SmolVLA) | **YES** |
| 7 | **Safety/Alignment** | 초기 | 낮음 (underhyped) | **YES** | **WIDE OPEN** |
| 8 | **Robot Manipulation** | 성숙 | 낮음 | **YES** | **YES** |
| 9 | **Multi-modal (Tactile)** | 초기 | 낮음 | 센서 필요 | 하드웨어 의존 |
| 10 | **World Models** | 초기 | **높음** | 한계 (multi-GPU 필요) | 어려움 |

---

## Part 4: 디지털 리터러시 — 과대광고 필터

이 섹션은 수집한 정보에 대한 의심과 검증 결과.

### 검증 1: "VLA가 지배적 패러다임"
- **근거**: RSS 2025 VLA 전용 세션 10편, 모든 major company가 VLA 구축
- **반론**: RSS 2025에서 Manipulation 30편, Imitation Learning 18편 → VLA 외 접근도 건재
- **판정**: VLA는 HOT하지만 "유일한" 패러다임은 아님. **부분적 과장**

### 검증 2: "3DGS는 genuinely revolutionary"
- **근거**: 451 GitHub repos, 분 단위 학습, SplaTAM/Re3Sim
- **반론**: 451 repos 대부분 비-로봇틱스(렌더링/게임). 로봇 적용은 아직 PoC 수준
- **판정**: Promising하지만 "revolutionary"는 과장. **"rapidly emerging"이 정확**

### 검증 3: "Robot Safety는 wide open"
- **근거**: RSS 2025 Outstanding Paper = safety, VLA-specific safety 연구 거의 없음
- **반론**: Control Barrier Functions, safe RL은 수년간 연구됨 (Aaron Ames, Chuchu Fan)
- **판정**: **"VLA-specific safety"만** wide open. 일반 robot safety는 아님

### 검증 4: "100 demonstrations면 충분"
- **Google**: 자체 벤치마크, 독립 검증 없음 (Gemini Robotics 비공개)
- **PI**: pi0.5의 "100 environments" ≠ "100 demonstrations" (중요한 차이!)
- **SmolVLA 공식**: 50 episodes (SO-100 in-distribution)
- **우리 경험**: 74 episodes → 100% (단일 위치)
- **판정**: 숫자 "100"은 맥락 의존적. **과도한 일반화 위험**

### 검증 5: "Tesla Optimus is a scam"
- **근거**: 텔레오퍼레이션 이벤트, 논문 없음, 타임라인 미스
- **반론**: "scam"은 한 연구자 의견이지 산업계 합의 아님. Ashok Elluswamy 영입 후 진전 가능
- **판정**: **"skepticism warranted"가 공정**. "scam"은 과함

### 검증 6: Korean robotics market is "tiny"
- **근거**: careersinrobotics.com에서 한국 6개 채용만 표시
- **반론**: 한국 채용은 한국 플랫폼에 집중. ROBOTIS, Doosan, KRAFTON 모두 활발
- **판정**: **데이터 소스 편향**. 시장이 작은 게 아니라 visibility가 낮은 것

---

## Part 5: 연구 주제 후보 — Tier별 분류

### Tier 1: 강력 추천 (현실적 + 학술 가치 + 트렌드 부합)

#### 방향 A: "Real-to-Sim-to-Real via 3DGS for Low-Cost Arms"

```
[Azure Kinect RGB-D] → [3DGS 씬 재구성] → [Isaac Lab 학습] → [RoArm 배포]
```

| 항목 | 내용 |
|------|------|
| 핵심 아이디어 | Kinect로 실제 테이블탑 촬영 → 3DGS로 재구성 → 재구성된 환경에서 정책 학습 → 실제 배포 |
| 신규성 | Re3Sim은 Franka(산업용) 대상. 저가형 팔(RoArm) + RGB-D(Kinect)는 미탐구 |
| 데이터 해결 | 시뮬에서 무한 데이터 생성 → 74 에피소드 한계 근본 해결 |
| 하드웨어 적합 | Kinect = RGB-D 완벽, 4090 = 3DGS + Isaac Lab 충분 |
| 논문 각도 | "저가형 로봇의 Real-to-Sim-to-Real 민주화" |
| 타겟 학회 | CoRL 2026, ICRA 2027, IROS 2026 |
| 위험 | 3DGS→물리 시뮬 변환이 기술적으로 쉽지 않음 (mesh 변환, collision geometry) |
| 교수님 설득 | "스펀지 잡기"를 넘어 "시뮬에서 무한 task 학습" 가능 |

**의심**: Re3Sim이 이미 있는데 novelty가 충분한가?
→ Re3Sim은 고가 Franka ($30K+) + 고가 카메라. 저가 셋업($500 이하)에서의 검증은 없음. "로봇 학습 민주화" 관점에서 차별화 가능.

#### 방향 B: "RL Fine-tuning of Small VLAs on Real Hardware"

```
[SmolVLA BC 학습] → [behavior cloning 성공] → [ReinFlow-style RL] → [일반화 향상]
```

| 항목 | 내용 |
|------|------|
| 핵심 아이디어 | SmolVLA imitation learning 후 → online RL로 추가 학습 → 성공률/일반화 향상 |
| 신규성 | ReinFlow(NeurIPS 2025)는 pi0/GR00T 대상. SmolVLA(450M) + 실제 하드웨어는 미탐구 |
| 기존 경험 활용 | SmolVLA 파이프라인 100% 활용 |
| 하드웨어 적합 | SmolVLA 450M → forward+backward 모두 4090에서 가능 |
| 논문 각도 | "Small VLA에서의 IL→RL 전이: 행동 복제 한계 극복" |
| 타겟 학회 | NeurIPS 2026, CoRL 2026 |
| 위험 | 실제 로봇 위 RL → 느림 + 하드웨어 손상 위험 + reward 설계 어려움 |
| 교수님 설득 | "학습 후에도 스스로 개선하는 로봇" |

**의심**: SmolVLA의 flow matching이 ReinFlow와 호환되는가?
→ ReinFlow는 flow matching을 명시적으로 지원 (pi0이 flow matching 사용). 하지만 SmolVLA-specific 구현은 필요. 레시피는 있지만 engineering effort 필요.

#### 방향 C: "Safe VLA: Constrained Action Generation"

```
[VLA 정책 출력] → [안전 필터 (CBF/Projection)] → [안전한 액션] → [로봇 실행]
```

| 항목 | 내용 |
|------|------|
| 핵심 아이디어 | VLA 모델 출력에 안전 제약 (관절 한계, 속도, 작업공간) 강제 |
| 신규성 | VLA-specific safety 연구 거의 없음. Safe-MPD(diffusion)는 있지만 VLA 안전은 미탐구 |
| 동기 부여 | 실제 Wrist_R -92° 폭주 경험 = 완벽한 motivating example |
| 하드웨어 적합 | 분석 + lightweight projection/clamping → 최소 compute |
| 논문 각도 | "VLA 모델의 안전 행동 생성: 학습된 정책의 물리적 제약 보장" |
| 타겟 학회 | RSS 2026, ICRA 2027, CoRL 2026 |
| 위험 | "그냥 clamp하면 되잖아" → 충분한 formalization 필요 (CBF level) |
| 교수님 설득 | "배포 안전성 = 실제 산업 적용의 핵심 병목" |

**의심**: Reviewer가 "trivial한 clamping" 이라고 할 위험?
→ 단순 clamping이 아니라 (1) VLA 출력의 안전 위반율 분석, (2) CBF/projection layer 설계, (3) 안전 보장이 성능에 미치는 영향 정량화, (4) 실제 하드웨어 검증이 있으면 논문 수준. Safe-MPD(diffusion)와의 차별화도 필요.

### Tier 2: 고려할만 (좋지만 추가 조건 필요)

#### 방향 D: "LLM + VLA 2-Tier Cognitive-Motor Architecture"
- Cloud VLM(Gemini API) → task planning → SmolVLA → motor control
- SayCan과 차별화 어려움이 최대 약점
- Baby AI "비비" 비전과 연결 가능
- 교수님이 "인지 아키텍처" 쪽에 관심 있으면 유망

#### 방향 E: "3D Diffusion Policy (DP3) with Depth Camera"
- Azure Kinect depth → point cloud → DP3/GenDP
- 기존 연구(DP3, GenDP)가 이미 탄탄해서 novelty 확보 어려움
- "저가 RGB-D + 저가 팔"에서의 벤치마크가 차별점

#### 방향 F: "ROS 2 + VLA 통합 프레임워크"
- ROBOTIS physical_ai_tools 유사, 취업 관점 강력
- 학술 논문으로는 novelty 부족. Workshop paper 또는 오픈소스 기여 가치
- 석사 포트폴리오로는 매우 좋음

### Tier 3: 현재 상황에서 비추천

| 방향 | 비추 이유 |
|------|-----------|
| World Model 학습 | multi-GPU 필요, 단일 4090 한계 |
| 휴머노이드 | 하드웨어 없음 ($16K+ Unitree G1 최소) |
| 대규모 VLA from scratch | 8+ A100 필요 |
| Pure HRI user study | 느림, IRB 승인 필요 |
| Tactile sensing | GelSight $300 구매 가능하지만 통합 복잡 |

---

## Part 6: 최종 추천

### 교수님 피드백 대응 전략

교수님: "스펀지만 잡고 있지 말라"
→ 해석: (a) 더 범용적, (b) 더 의미있는, (c) 산업 트렌드에 맞는 연구

### 추천 조합: A + C (3DGS Real-to-Sim + Safety)

**이유**:
1. **3DGS Real-to-Sim**은 "스펀지 잡기"를 넘어 "시뮬에서 무한 task 학습"으로 확장
2. **Safety**는 배포 시 실제로 겪은 문제 → 현실적 동기 부여
3. 두 방향 모두 VLA에 국한되지 않음 → Diffusion Policy, ACT 등 다른 방법에도 적용 가능
4. 기존 SmolVLA 경험 + Isaac Lab 설치 + Azure Kinect 모두 활용
5. "스펀지"를 넘어 시뮬에서 다양한 물체/task 자동 생성 가능

### 구체적 연구 로드맵

```
Phase 1 (1-2개월): 3DGS 파이프라인 구축
  - Kinect RGB-D → 3DGS 씬 재구성
  - 3DGS → mesh 변환 → Isaac Lab 환경 구축
  - 기본 reach task RL 학습 (이미 Isaac Lab 설치됨)

Phase 2 (2-3개월): Sim-to-Real 전이
  - 시뮬에서 학습된 정책 → 실제 RoArm 배포
  - 도메인 갭 측정 및 개선
  - SmolVLA vs Diffusion Policy vs RL 비교

Phase 3 (1-2개월): Safety Layer 추가
  - VLA/정책 출력의 안전 위반율 분석
  - CBF/projection layer 설계 및 적용
  - 안전 보장 ↔ 성능 trade-off 정량화

Phase 4 (1-2개월): 논문 작성
  - 논문 1: "Real-to-Sim for Low-Cost Arms via 3DGS"
  - 논문 2 (optional): "Safe Action Generation for Learned Robot Policies"
```

### 대안: 관심사에 따른 선택

| 관심사 | 추천 방향 | 이유 |
|--------|----------|------|
| 시뮬레이션/3D | **A (3DGS Real-to-Sim)** | Kinect + Isaac Lab 시너지 |
| ML/학습 | **B (RL Fine-tuning)** | 기존 SmolVLA 100% 활용 |
| 안전/배포 | **C (Safe VLA)** | 실제 경험 기반, wide open |
| 인지/AI | **D (LLM+VLA 2-Tier)** | Baby AI 비전 연결 |
| 취업/실용 | **F (ROS 2 + VLA)** | 산업 트렌드 직결 |

---

## Part 7: ROS를 왜 많이 채용하나?

### 핵심 이유

1. **표준화**: 센서/액추에이터 인터페이스 표준. 팀 간 협업 용이
2. **생태계**: Nav2(자율주행), MoveIt 2(매니퓰레이션), ros2_control(제어) → 바퀴 재발명 불필요
3. **산업 채택**: 352+ 채용, NVIDIA/Amazon/BMW/Caterpillar 등 대기업 사용
4. **재사용성**: 한 번 만든 패키지를 다른 로봇/프로젝트에 재사용
5. **시뮬 연동**: Gazebo/Isaac Sim과 동일 인터페이스로 연결
6. **Fleet 관리**: 다수 로봇 운영 시 필수 (Nav2 fleet, Foxglove 모니터링)

### ROS가 필요 없는 경우

- 단일 팔 + 단일 카메라 + Python 스크립트 → 현재 우리 상황
- VLA 연구 자체 (LeRobot, openpi는 ROS 없이 동작)
- 시뮬레이션 전용 연구 (MuJoCo/Isaac Lab은 ROS 불필요)

### ROS가 필수인 경우

- 다중 센서 융합 (LiDAR + camera + IMU)
- 모바일 베이스 + 매니퓰레이터 조합
- Fleet 관리/모니터링
- 산업 배포/인증

### 권장

현재 SmolVLA 연구에는 ROS 불필요. 하지만 **포트폴리오/취업 관점**에서 ROS 2 기본 경험은 가치 있음. deploy_smolvla.py를 ROS 2 node로 래핑하는 것이 좋은 연습.

---

## Part 8: 한국에서의 현실적 기회

### 채용 활발한 한국 회사

| 회사 | 분야 | ROS 사용 | VLA/AI |
|------|------|----------|--------|
| ROBOTIS | 로봇 하드웨어 + AI | YES (선구자) | LeRobot + pi0 적극 도입 |
| Doosan Robotics | 협동 로봇 | YES (공식 패키지) | 전통적 제어 중심 |
| KRAFTON | 게임 → AI/로봇 | 채용 요건 | VLA 연구자 적극 채용 |
| Samsung Research | Consumer AI | 채용 요건 | Physical AI 인턴 |
| 42dot (현대) | 자율주행 | 부분적 | SLAM/자율주행 |
| Naver Labs | 실내 로봇 | 불확실 | 독자 스택 추정 |

### 석사 졸업 후 경로

1. **KRAFTON/삼성 AI 연구** — VLA + ROS 경험이면 직접 매칭
2. **ROBOTIS** — LeRobot + VLA 경험이면 physical_ai_tools 팀과 매칭
3. **해외 (미국/유럽)** — NVIDIA Isaac ROS, PI, BD 등 → PhD 없이도 ML eng 가능
4. **대학원 진학 (PhD)** — 연구 방향 A/B/C 중 논문 나오면 좋은 PhD 지원 가능

---

## Part 9: 주요 참고 자료

### 학회
- RSS 2025: https://roboticsconference.org/2025/program/
- ICRA 2025: https://2025.ieee-icra.org/
- CoRL 2024/2025: https://www.corl.org/

### 오픈소스
- LeRobot: https://github.com/huggingface/lerobot
- pi0 (open-source): https://github.com/physical-intelligence/openpi
- NVIDIA GR00T: https://developer.nvidia.com/isaac/groot
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- Re3Sim: InternRobotics (3DGS for real-to-sim)
- ReinFlow: NeurIPS 2025 (RL fine-tuning of flow matching)

### 한국 ROS 관련
- ROBOTIS physical_ai_tools: https://github.com/ROBOTIS-GIT/physical_ai_tools
- Doosan ROS 2: https://github.com/doosan-robotics/doosan-robot2
- careersinrobotics.com: 채용/연봉 데이터 (미국 중심)

### 연구 방향별 핵심 논문
- 3DGS: Re3Sim (2025), SplaTAM (CVPR 2024), MonoGS (CVPR 2024 Best Demo)
- RL Fine-tuning: ReinFlow (NeurIPS 2025), ConRFT (RSS 2025)
- Safety: Safe-MPD (2026), SELP (2024), RSS 2025 Outstanding Paper
- Diffusion: DP3, GenDP (CoRL 2024), FAST (RSS 2025 finalist)
- VLA: pi0 (RSS 2025), OpenVLA (CoRL 2024), CoT-VLA (CVPR 2025)

---

*작성일: 2026-03-06*
*소스: RSS/ICRA/CoRL/NeurIPS 학회 프로그램, NVIDIA/Google/PI/BD 공식 블로그, ros.org, careersinrobotics.com, GitHub trending/topics, Awesome-LLM-Robotics, LeRobot README*
*디지털 리터러시 검증 완료: Part 4 참조*
