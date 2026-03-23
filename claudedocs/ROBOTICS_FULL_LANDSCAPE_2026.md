# Robotics Full Landscape 2026 — 종합 정보 보고서

**작성일:** 2026-03-07
**목적:** 연구 방향 결정을 위한 현재 로보틱스 전체 기술 Landscape
**방법:** 6개 전문 agent 병렬 조사 → 교차 검증 → 디지털 리터러시 필터링

---

## 1. Executive Summary

6개 전문 agent가 2026년 3월 기준 로보틱스 전체 landscape를 조사했다.
총 **18개 VLA 모델, 15개 시뮬레이터, 40+ 학습 방법론, 30+ ROS 2 패키지, 15+ 로봇 회사, 10+ 데이터셋**을 분석.

### 한 줄 요약

> **VLA + LeRobot이 현재 mainstream이지만, "pick up X"를 넘어서는 연구는 아직 초기.
> 진짜 기회는 VLA를 실제로 안정적이고 안전하게 배포하는 것, 그리고 다중 물체/다중 작업으로 확장하는 것.**

---

## 2. 전체 기술 Landscape Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ROBOTICS TECH LANDSCAPE 2026                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─── FOUNDATION MODELS ─────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  [Closed]          [Open, Large]        [Open, Small]              │ │
│  │  pi-0/pi-0.5       OpenVLA-OFT (7B)    ★SmolVLA (450M)           │ │
│  │  RT-2 (55B)        GR00T N1.6 (3B)     Octo (93M)                │ │
│  │                    RDT-1B (1.2B)       ACT (80M)                  │ │
│  │                                                                     │ │
│  │  Action Heads: Flow Matching > Diffusion > Discrete Tokens         │ │
│  │  Trend: Dual-System (VLM + Action Expert)                          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── LEARNING METHODS ──────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  IL (주류):  BC → ACT → Diffusion → Flow Matching → VLA           │ │
│  │  RL (보조):  PPO(sim locomotion), SERL(real fine-tune)             │ │
│  │  Hybrid:     ★VLA + RL fine-tuning (가장 유망)                     │ │
│  │  World Model: DreamerV3, TD-MPC2 (아직 real 검증 부족)             │ │
│  │                                                                     │ │
│  │  ★ IL+RL Hybrid = 현재 최선의 manipulation 방법론                   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── SIMULATION ────────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  [Research]         [Industry]           [Emerging]                │ │
│  │  MuJoCo (표준)      Isaac Sim/Lab        ★Genesis (43M FPS!)      │ │
│  │  PyBullet           Omniverse            3DGS (SplatSim)          │ │
│  │  Drake              Gazebo Harmonic      Neural Physics            │ │
│  │                                                                     │ │
│  │  Sim-to-Real: Locomotion=해결, Manipulation=미해결                  │ │
│  │  3DGS: 시각적 gap 해결 86.25% zero-shot (SplatSim)                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── MIDDLEWARE & DEPLOYMENT ───────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  ROS 2 (Humble/Jazzy)  +  Zenoh (Tier 1)  +  MoveIt 2             │ │
│  │  NVIDIA Isaac ROS       +  ros2_control     +  Nav2                 │ │
│  │  ★ROBOTIS physical_ai_tools (LeRobot + ROS 2 브릿지!)              │ │
│  │  zenoh_ros2_sdk (ROS 2 없이 ROS 2 통신 가능)                       │ │
│  │                                                                     │ │
│  │  VLA + ROS 2 = 보완 관계 (경쟁 아님)                                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── HARDWARE ──────────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  [Research 표준]     [VLA 연구]          [Humanoid]                 │ │
│  │  Franka FR3 ($30K+)  ★SO-100 ($100)     Unitree G1 ($16K)        │ │
│  │  xArm 6 ($5.3K)     Koch v1.1 ($200)    1X NEO ($20K)            │ │
│  │  UR5e ($25K+)        RoArm M3 ($200)     Figure 02                │ │
│  │                                                                     │ │
│  │  LeRobot = "로보틱스의 PyTorch" → 하드웨어보다 생태계가 중요         │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── DATA ECOSYSTEM ────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  Open X-Embodiment (1M+ traj)  |  DROID (76K demos)               │ │
│  │  BridgeData V2 (60K traj)      |  LIBERO (시뮬 벤치마크)           │ │
│  │  community_dataset_v1 (SmolVLA 사전학습, SO-100 only)              │ │
│  │  HuggingFace Hub = 로보틱스 데이터 허브                             │ │
│  │                                                                     │ │
│  │  ★RoboCup 2026: 인천 송도 (6/30-7/6)                              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 6개 도메인 핵심 요약

### 3.1 VLA & Foundation Models (VLA-INTEL)

| 발견 | 의미 |
|------|------|
| RTX 4090에서 학습 가능한 VLA: SmolVLA(450M), Octo(93M), ACT(80M) | 우리 GPU로 할 수 있는 건 제한적이지만 SmolVLA가 최선 |
| OpenVLA-OFT가 LIBERO SOTA (97.1%) | QLoRA로 fine-tune 가능하지만 7B → 현실적으로 어려움 |
| GR00T N1.6 (3B, NVIDIA) 오픈소스 | RTX 4090에서 inference 가능, fine-tuning은 어려움 |
| pi-0/pi-0.5만이 real deployment 검증 | 나머지는 모두 lab 환경 |
| Flow Matching이 Diffusion을 대체하는 추세 | SmolVLA가 이미 Flow Matching 사용 → 트렌드에 맞음 |
| Dual-System (VLM + Action Expert) 패턴 수렴 | SmolVLA도 이 구조 |

### 3.2 시뮬레이터 & 디지털 트윈 (SIM-RECON)

| 발견 | 의미 |
|------|------|
| Genesis: 28.2K stars, RTX 4090에서 43M FPS | 새로운 선택지 but 물리 정확도 미검증 |
| MuJoCo: 연구 표준, MJX/MJWarp GPU 지원 | 가장 안정적인 선택 |
| Isaac Sim/Lab: 산업 표준, RTX 4080+ 최소 | 우리 환경에서 사용 가능 (이미 구축) |
| 3DGS (SplatSim): 86.25% zero-shot sim-to-real | 시각적 sim-to-real gap 해결의 새 패러다임 |
| 디지털 트윈 플랫폼 (Siemens, PTC): 우리와 무관 | Enterprise용, 학술 연구에 부적합 |
| Sim-to-Real: locomotion=해결, manipulation=미해결 | manipulation sim-to-real이 열린 연구 문제 |

### 3.3 Robot Learning 방법론 (LEARN-OPS)

| 발견 | 의미 |
|------|------|
| 현재 최선: VLA pretrain → IL fine-tune → RL robustness | 3단계 파이프라인이 표준화 중 |
| Pure RL for manipulation = 죽었음 | RL은 보조 수단으로만 사용 |
| HIL-SERL: 20 demos + 25분 RL → near-perfect | IL+RL hybrid의 강력한 증거 |
| SmolVLA OOD 최소 50ep, 권장 150ep | 우리 경험과 일치 |
| ACT: 10분 데이터로 80-90% | VLA 없이도 가능, 좋은 baseline |
| Contact-rich manipulation = 미해결 | 아직 아무도 못 풀었음 |

### 3.4 ROS 2 생태계 (ROS-SIGINT)

| 발견 | 의미 |
|------|------|
| ROS 1 완전 EOL (2025.05) | ROS 2가 유일한 선택지 |
| Zenoh = Tier 1 (Kilted Kaiju) | DDS보다 빠르고 가벼움 |
| ★ROBOTIS physical_ai_tools | LeRobot + ROS 2 브릿지 이미 존재 |
| zenoh_ros2_sdk | pip install만으로 ROS 2 통신 가능 |
| Azure Kinect ROS driver = retired | pyk4a 사용이 올바른 선택 (이미 하고 있음) |
| VLA + ROS 2 = 보완 관계 | VLA가 ROS 2를 대체하는 게 아님 |
| MoveIt 2 + VLA hybrid = 미래 방향 | VLA for perception, MoveIt for safe execution |

### 3.5 Hardware & Embodiment (BODY-HUMINT)

| 발견 | 의미 |
|------|------|
| SO-100/101 = VLA 연구 표준 ($100-200) | SmolVLA 사전학습 로봇 |
| RoArm M3 = SO-100보다 좋은 HW but 생태계 없음 | 강점: 2x workspace, 듀얼 숄더. 약점: 커뮤니티 없음 |
| Franka FR3 = 연구 1위 ($30K+) | 우리와 다른 tier |
| LeRobot = "로보틱스의 PyTorch" | 하드웨어보다 생태계 호환성이 중요 |
| 중국 공급망 지배 | Feetech 서보가 SO-100의 기반 |
| 최저가 VLA 연구: $200-400 (SO-100 + webcam + Colab) | 우리 setup($2,650+)은 중급 |

### 3.6 데이터셋/벤치마크/대회 (DATA-OSINT)

| 발견 | 의미 |
|------|------|
| Open X-Embodiment (1M+ traj) = 최대 규모 | 기여 가능 target |
| community_dataset_v1 = SmolVLA 사전학습 데이터 (SO-100 only) | 우리 RoArm 데이터는 최초 |
| LIBERO = 시뮬레이션 벤치마크 표준 | LeRobot으로 평가 가능 |
| ★RoboCup 2026: 인천 송도 6/30-7/6 | 한국 최초 개최, 참관 추천 |
| Data scaling: 50ep (in-dist), 150ep (OOD) | 우리 데이터 전략과 일치 |
| HuggingFace Hub에 RoArm M3 데이터 올리면 최초 | 학술 기여 가능 |

---

## 4. 디지털 리터러시 검증

### 4.1 높은 신뢰도 (6개 agent 교차 검증)

| 주장 | 검증 | 판정 |
|------|------|------|
| VLA가 현재 주류 패러다임 | 6/6 agent 일치 | ✅ 신뢰 |
| SmolVLA가 RTX 4090 최선 | VLA-INTEL, LEARN-OPS | ✅ 신뢰 |
| LeRobot이 de facto 표준 | 4/6 agent 언급 | ✅ 신뢰 |
| ROS 2 + VLA = 보완 관계 | ROS-SIGINT 심층 분석 | ✅ 신뢰 |
| IL+RL hybrid = 최선 | LEARN-OPS, 논문 다수 | ✅ 신뢰 |
| Manipulation sim-to-real = 미해결 | SIM-RECON, LEARN-OPS | ✅ 신뢰 |

### 4.2 의심이 필요한 주장 (하이프 경고)

| 주장 | 출처 | 문제 | 판정 |
|------|------|------|------|
| "Genesis가 43M FPS, Isaac 10-80배 빠름" | SIM-RECON | 물리 정확도 미검증, 벤치마크 조건 불명확 | ⚠️ 과장 가능 |
| "Figure AI $39B 기업가치" | BODY-HUMINT | 실제 배포 대비 과도한 투자. 버블 징후 | ⚠️ 하이프 |
| "GR00T N1.6 RTX 4090 가능" | VLA-INTEL | inference는 가능하나 fine-tuning은 현실적으로 어려움 | ⚠️ 반만 사실 |
| "HIL-SERL 25분 near-perfect" | LEARN-OPS | Franka + 구조화 환경 결과. RoArm M3에서 동일한 결과? | ⚠️ 조건부 |
| "3DGS 86.25% zero-shot transfer" | SIM-RECON | 특정 태스크/환경 결과. 범용성 미검증 | ⚠️ 제한적 |
| "SmolVLA가 7B 모델 성능에 근접" | VLA-INTEL | LIBERO 벤치마크 한정. 실제 로봇에서는 차이 클 수 있음 | ⚠️ 벤치마크 한정 |

### 4.3 확실한 하이프/과장

| 주장 | 판정 | 이유 |
|------|------|------|
| "RD-VLA 80배 빠르다" | ❌ 과장 | 0.5B vs 7B+CoT 비교 = 불공정 |
| "VLA가 모든 것을 해결" | ❌ | Contact-rich, safety, long-horizon 모두 미해결 |
| "Humanoid 시대 도래" | ❌ | 대부분 teleoperation, 자율 주행하는 humanoid 없음 |
| "디지털 트윈으로 제조 혁신" | ❌ | Enterprise 플랫폼, 학술 연구에 부적합 |

---

## 5. 교차 분석: 도메인 간 연결점

```
                    ┌──────────────┐
                    │  VLA Models  │
                    │  (SmolVLA)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
      ┌───────────┐ ┌───────────┐ ┌───────────┐
      │  Sim/Data  │ │  ROS 2    │ │  Hardware  │
      │  Genesis   │ │  MoveIt   │ │  RoArm M3  │
      │  3DGS      │ │  Safety   │ │  SO-100    │
      └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
            │              │              │
            └──────────────┼──────────────┘
                           ↓
                    ┌──────────────┐
                    │  RL Fine-tune│
                    │  (SERL)      │
                    └──────────────┘
```

**핵심 교차점 3개:**

1. **VLA + ROS 2 = Safe Deployment**
   - ROBOTIS가 이미 구현 (physical_ai_tools)
   - VLA가 perception+planning, ROS 2/MoveIt가 safety+execution
   - 이 조합이 산업에서 실제로 필요

2. **VLA + RL = Robust Policy**
   - IL만으로는 closed-loop drift 문제
   - SERL-style RL fine-tuning이 해결책
   - 하지만 RoArm M3에서 아직 아무도 안 함

3. **3DGS + Sim = Data Scaling**
   - 물리적 데이터 수집의 병목을 깨는 방법
   - 3DGS로 실제 환경 복제 → 시뮬레이션에서 대량 생성
   - 아직 초기 단계 but 가장 미래지향적

---

## 6. 연구 주제 후보

### 기준
- 교수님 피드백: "스펀지 잡기를 넘어서라"
- 현실: 석사, RoArm M3, RTX 4090, 1-2년
- 학술 가치: 논문 가능성, novelty
- 산업 가치: 취업/커리어에 도움

### Tier 1: 가장 추천 (현실적 + 임팩트)

#### A. Multi-Task Language-Conditioned VLA (다중 물체/다중 작업)

| 항목 | 내용 |
|------|------|
| **문제** | SmolVLA가 다양한 물체를 language 조건으로 구분하여 잡을 수 있는가? |
| **방법** | 스펀지/컵/펜/박스 등 4-5개 물체, 물체별 50ep, "pick up the X" task text |
| **기여** | OOD 로봇(RoArm M3)에서의 multi-task VLA 성능 최초 보고 |
| **필요 자원** | 기존 setup + 다양한 물체 + 데이터 수집 시간 |
| **기간** | 3-6개월 |
| **논문 가능** | ✅ Workshop paper 수준 |
| **교수님 답변** | "스펀지만이 아닌 다양한 물체를 다룹니다" |

#### B. VLA + ROS 2 Safe Deployment System

| 항목 | 내용 |
|------|------|
| **문제** | VLA 정책을 안전하게 실시간 배포하는 시스템 아키텍처 |
| **방법** | SmolVLA + zenoh_ros2_sdk + MoveIt 2 safety layer |
| **기여** | VLA + classical robotics 하이브리드 시스템 아키텍처 |
| **필요 자원** | ROS 2 학습 + ros2_control 구현 |
| **기간** | 4-6개월 |
| **논문 가능** | ✅ System paper (IROS/ICRA workshop) |
| **교수님 답변** | "시스템 수준의 연구로 확장했습니다" |

### Tier 2: 도전적 (높은 학술 가치)

#### C. VLA + RL Fine-tuning (IL → RL Hybrid)

| 항목 | 내용 |
|------|------|
| **문제** | SmolVLA IL 정책을 RL로 fine-tune하여 robustness 향상 |
| **방법** | SmolVLA pretrain → SERL-style RL → 성능 비교 |
| **기여** | "First VLA + RL on sub-$500 arm" |
| **필요 자원** | 자동 리셋 구현, reward 설계, RL 프레임워크 |
| **기간** | 6-9개월 |
| **논문 가능** | ✅ Main conference potential (CoRL, ICRA) |
| **위험** | SERL 구현 복잡, 자동 리셋 하드웨어 필요 |

#### D. Real-to-Sim-to-Real with 3DGS

| 항목 | 내용 |
|------|------|
| **문제** | 물리적 데이터 수집 병목을 3DGS 시뮬레이션으로 해결 |
| **방법** | Azure Kinect depth → 3DGS → Genesis/MuJoCo → SmolVLA 사전학습 보완 |
| **기여** | 3DGS 기반 데이터 증강의 실제 효과 검증 |
| **필요 자원** | 3DGS 구현, 시뮬레이터 통합, 대량 실험 |
| **기간** | 6-12개월 |
| **논문 가능** | ✅ High impact if successful (RSS, CoRL) |
| **위험** | 기술적 난이도 높음, 3DGS 경험 없음 |

### Tier 3: 부가 가치

#### E. LeRobot RoArm M3 공식 통합 + HuggingFace 데이터셋 공개

| 항목 | 내용 |
|------|------|
| **문제** | RoArm M3의 LeRobot 생태계 부재 |
| **방법** | lerobot PR 제출 + HuggingFace Hub에 데이터셋 공개 |
| **기여** | 오픈소스, 커뮤니티 |
| **기간** | 2-4주 |
| **논문 가능** | ❌ (엔지니어링) |
| **가치** | 이력서, GitHub 프로필, 커뮤니티 인지도 |

#### F. RoboCup 2026 참관/참가

| 항목 | 내용 |
|------|------|
| **문제** | 한국 최초 RoboCup, 네트워킹 기회 |
| **기간** | 6/30-7/6 인천 송도 |
| **가치** | 연구자 네트워킹, 트렌드 파악 |

---

## 7. 최종 추천

### 추천 조합: A + B (+ E 병행)

**"다양한 물체를 안전하게 다루는 VLA 시스템"**

```
Phase 1 (1-2개월): 데이터 확장
├── 기존 스펀지 데이터 → 3-4개 물체 추가
├── 물체별 50ep, language conditioning
├── HuggingFace Hub에 데이터셋 공개 (E)
└── 중간 배포 테스트

Phase 2 (2-3개월): Multi-Task VLA 학습/평가
├── multi-task SmolVLA 학습
├── 물체별 성공률 측정
├── ACT/Diffusion Policy와 비교 (baseline)
└── 논문 작성 시작

Phase 3 (3-4개월): ROS 2 통합
├── zenoh_ros2_sdk로 SmolVLA 배포
├── MoveIt 2 safety layer 추가
├── 시스템 아키텍처 논문
└── RoArm M3 LeRobot PR (E)

Phase 4 (선택, 4-6개월): RL Fine-tuning (C)
├── SERL-style RL 추가
├── multi-task robustness 향상
└── 최종 논문
```

### 왜 이 조합인가?

1. **교수님 피드백 직접 답변**: "스펀지만이 아닌 5개 물체, 시스템 수준 연구"
2. **현실적**: 기존 setup 활용, 추가 하드웨어 불필요
3. **점진적 확장**: 각 Phase가 독립적으로 가치 있음 (Phase 1만 해도 데이터 공개 가치)
4. **산업 가치**: ROS 2 + VLA = 현재 가장 높은 채용 수요
5. **학술 가치**: OOD 로봇에서 multi-task VLA + safe deployment = 아무도 안 했음
6. **한국 기회**: ROBOTIS physical_ai_tools 활용 → ROBOTIS 취업 연결 가능

---

## 8. 한국 로보틱스 기회 (2026)

| 회사 | 분야 | 관련 기술 |
|------|------|-----------|
| **ROBOTIS** | Physical AI, DYNAMIXEL | LeRobot, ROS 2, SmolVLA |
| **Doosan Robotics** | Collaborative arms | ROS 2, MoveIt |
| **Rainbow Robotics** | Cobots (KAIST 출신) | ROS 2, 제어 |
| **Samsung Research** | AI Robotics | VLM, Physical AI |
| **KRAFTON** | VLA 연구 채용 중 | VLA, Deep Learning |
| **Naver Labs** | 실내 로봇 | SLAM, Navigation |
| **LG Electronics** | 서비스 로봇 | AMR, ROS 2 |

**RoboCup 2026 인천**: 6/30-7/6, 한국 최초. 반드시 참관 추천.

---

## 9. 생성된 보고서 목록

| 보고서 | Agent | 파일 |
|--------|-------|------|
| VLA Intelligence Report | VLA-INTEL | `VLA_INTELLIGENCE_REPORT_2026.md` |
| Simulator & Digital Twin | SIM-RECON | `SIM_RECON_INTELLIGENCE_REPORT.md` |
| Robot Learning Methods | LEARN-OPS | `claudedocs/ROBOT_LEARNING_METHODS_REPORT.md` |
| ROS 2 Ecosystem | ROS-SIGINT | `claudedocs/ROS2_ECOSYSTEM_INTEL_REPORT.md` |
| Hardware & Embodiment | BODY-HUMINT | `claudedocs/ROBOT_HARDWARE_LANDSCAPE_2026.md` |
| Data Ecosystem | DATA-OSINT | `claudedocs/ROBOTICS_DATA_ECOSYSTEM_2026.md` |

**총 분량**: 6개 보고서 합계 약 4,000줄 (이 종합 보고서 제외)

---

*6개 agent × 20+ web searches each = 120+ primary sources 기반. 2026-03-07 기준.*
