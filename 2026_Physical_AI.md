# 2024-2025 Physical AI 총정리

> **Last Updated**: 2026-01-23
> **Purpose**: VLA(Vision-Language-Action) 모델과 Physical AI 발전의 핵심 흐름을 정리한 참고 문서
> **Source**: 엥지 유니버스 2025년 Physical AI 총결산 + 최신 논문/발표 자료

---

## 목차

1. [Physical AI란?](#1-physical-ai란)
2. [역사: RT-1에서 VLA까지](#2-역사-rt-1에서-vla까지)
3. [2024: 데이터 스케일링과 범용 정책](#3-2024-데이터-스케일링과-범용-정책)
4. [2025: 아키텍처 혁신](#4-2025-아키텍처-혁신)
5. [빅테크의 접근: Google vs NVIDIA](#5-빅테크의-접근-google-vs-nvidia)
6. [온디바이스 최적화](#6-온디바이스-최적화)
7. [핵심 기술 비교표](#7-핵심-기술-비교표)
8. [미래 전망](#8-미래-전망)
9. [참고 자료](#9-참고-자료)

---

## 1. Physical AI란?

**Physical AI**는 디지털 세계를 넘어 **물리적 세계에서 작동하는 AI**를 의미한다.

```
┌─────────────────────────────────────────────────────────────┐
│                      AI의 진화                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Digital AI]                    [Physical AI]              │
│  - ChatGPT (텍스트)              - 로봇 (행동)               │
│  - DALL-E (이미지)               - 자율주행 (물리 세계)       │
│  - 가상 세계에서 동작             - 실제 세계에서 동작         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 개념: VLA (Vision-Language-Action)

```
[Vision]     +     [Language]     +     [Action]
카메라 입력        언어 명령           로봇 동작 출력
    │                  │                   │
    └──────────────────┴───────────────────┘
                       │
              [VLA Model]
              통합된 하나의 모델
```

**VLA의 핵심 아이디어**: GPT가 다음 단어를 예측하듯, **로봇이 다음 행동을 예측**하게 만듦

---

## 2. 역사: RT-1에서 VLA까지

### 2022년 말: RT-1의 등장 (패러다임 전환)

Google이 제안한 **Robotics Transformer 1 (RT-1)**:
- 카메라 영상, 로봇 관절, 언어 명령을 모두 **토큰으로 변환**
- Transformer에 입력하여 다음 행동 예측
- 13만 개 데이터로 100가지 이상 작업 수행

```
기존 방식: 정교한 수식 + 제어 이론 (Boston Dynamics)
     ↓
RT-1 방식: 데이터 + Transformer (Google)
```

### RT-2: 거대 VLM + 로봇

RT-2는 **이미 학습된 거대 시각-언어 모델**에 로봇을 연결:
- "그림 속 바나나와 같은 색깔의 물건을 집어라" → 노란색 물건 집음
- 세계 지식을 동원한 **시맨틱 기반 제어**의 시작

| 세대 | 방식 | 특징 |
|------|------|------|
| 기존 | 스킬 기반 | 정해진 동작 수행 |
| RT-1/2 | 시맨틱 기반 | 맥락과 의미 이해 |

---

## 3. 2024: 데이터 스케일링과 범용 정책

### 3.1 Open X-Embodiment Dataset

[Open X-Embodiment](https://robotics-transformer-x.github.io/)는 로봇 AI의 "ImageNet":

| 항목 | 수치 |
|------|------|
| 총 에피소드 | 100만+ |
| 로봇 종류 | 22종 |
| 참여 기관 | 34개 연구실 |
| 데이터셋 수 | 60개 |

### 3.2 Octo: 오픈소스 범용 정책의 시작

[Octo](https://octo-models.github.io/) (UC Berkeley, 2024):

```python
# Octo 핵심 특징
Architecture: Transformer-based Diffusion Policy
Parameters: 27M (Small), 93M (Base)
Training Data: 800k robot episodes (Open X-Embodiment)
```

**핵심 혁신**:
- 여러 종류의 로봇 데이터를 **하나의 모델**로 통합 학습
- 언어 명령 또는 목표 이미지로 지시 가능
- 새로운 로봇에 몇 시간 만에 fine-tuning

| 성능 비교 | Octo | RT-1-X | RT-2-X |
|-----------|------|--------|--------|
| 언어 지시 성능 | ✅ 우수 | 보통 | 우수 |
| 모델 크기 | 93M | - | 55B |
| 오픈소스 | ✅ | ✅ | ❌ |

### 3.3 OpenVLA: 행동 토큰화의 표준

[OpenVLA](https://openvla.github.io/) (Stanford, 2024):

**아키텍처**:
```
[SigLIP + DinoV2] → [Projector] → [Llama 2 7B] → [Action Tokens]
   Visual Encoder      Mapping       LLM Backbone    출력
```

**핵심 혁신: 행동 토큰화 (Action Tokenization)**
```
연속적인 로봇 관절값 → 이산적인 토큰으로 변환
      [0.15, -0.23, ...]  →  [ACT_1, ACT_2, ...]
                               ↓
                    LLM이 이해하고 생성 가능
```

**성능**:
- 7B 모델로 55B RT-2-X 추월
- 970k 실제 로봇 시연 데이터로 학습
- 64x A100 GPU, 15일 학습

### 3.4 2024년의 한계

데이터 스케일링만으로는 부족한 점:
1. **미세 동작 제어** 부족
2. **실시간 제어** 어려움
3. **토큰 기반** 행동 생성의 지연

→ **2025년: 아키텍처 자체의 혁신**으로 방향 전환

---

## 4. 2025: 아키텍처 혁신

### 4.1 π₀ (Pi-Zero): Flow Matching 기반 생성 제어

[Physical Intelligence의 π₀](https://www.pi.website/blog/pi0):

**Diffusion vs Flow Matching**:
```
┌─────────────────────────────────────────────────────────────┐
│ Diffusion Policy                                            │
│ - 노이즈 → 데이터 (여러 번 디노이징)                         │
│ - 확률적 SDE 기반                                           │
│ - 반복 과정으로 느림                                         │
├─────────────────────────────────────────────────────────────┤
│ Flow Matching (π₀)                                          │
│ - 벡터장을 따라 흐름                                         │
│ - 결정론적 ODE 기반                                          │
│ - 전체 흐름 설계 → 더 빠르고 안정적                          │
└─────────────────────────────────────────────────────────────┘
```

**π₀ 특징**:
- 7개 로봇, 68개 태스크에서 사전학습
- **고주파 제어** + **롱 호라이즌 연속 제어** 동시 달성
- 세탁물 접기, 테이블 정리 등 복잡한 작업 수행

**2025년 업데이트**:
| 버전 | 날짜 | 특징 |
|------|------|------|
| π₀ | 2024.10 | 초기 버전 |
| π₀.5 | 2025.09 | 오픈월드 일반화 |
| π₀-FAST | 2025 | Autoregressive 토큰화 (4-5x 느리지만 언어 이해 우수) |
| π₀.6 | 2025.12 | 에스프레소 제조 데모 |

**오픈소스**: [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

### 4.2 CogACT: 인지와 행동의 분리

[Microsoft의 CogACT](https://cogact.github.io/):

**아키텍처 (3단계 분리)**:
```
┌────────────────────────────────────────────────────────────┐
│ 1. Vision Module                                           │
│    이미지 → Visual Tokens                                  │
├────────────────────────────────────────────────────────────┤
│ 2. Language Module (인지)                                  │
│    Visual Tokens + 언어 명령 → Cognition Feature           │
│    "어떤 순서로 무엇을 할지" 계획                           │
├────────────────────────────────────────────────────────────┤
│ 3. Diffusion Action Module (행동)                          │
│    Cognition Feature → Multi-step Actions                  │
│    실제 관절 움직임 생성                                    │
└────────────────────────────────────────────────────────────┘
```

**핵심 혁신**: 고차원 사고(System 2)와 저차원 제어(System 1) 분리
- 마치 **두뇌**와 **신체**를 나눈 것

**성능**:
- OpenVLA (7B) 대비 **+35%** (시뮬레이션)
- OpenVLA 대비 **+55%** (실제 로봇)
- RT-2-X (55B) 대비 **+18%**

**모델 크기**: ~7.6B 파라미터 (Small/Base/Large 버전)

### 4.3 RoboVLMs: VLA 설계 가이드라인

[RoboVLMs](https://robovlms.github.io/):

VLA 설계의 4가지 핵심 질문에 대한 실험적 답:

| 질문 | 발견 |
|------|------|
| VLA를 왜 쓰나? | Vision-Language 사전학습이 일반화와 데이터 효율성에 필수 |
| VLA를 어떻게 구성? | Diffusion 기반 디코더 + 계층적 융합이 최고 |
| 어떤 VLM 백본? | 8개 VLM 비교 → 작업별 최적 선택 가이드 |
| Cross-embodiment 데이터? | 사전학습/공동학습/후학습 각각의 이점 분석 |

### 4.4 Action Chunking: 실시간 제어의 핵심

**Action Chunking**이란?
```
기존: 매 타임스텝마다 1개 액션 예측
      t=1 → a₁, t=2 → a₂, t=3 → a₃ ...

Action Chunking: 한 번에 N개 액션 예측
      t=1 → [a₁, a₂, a₃, ..., aₙ] (한 번에!)
```

**장점**:
- 정책 모델을 낮은 주기로 실행 가능
- 온디바이스 환경에서 지연 감소
- Diffusion, Flow Matching 모두와 독립적으로 사용 가능

---

## 5. 빅테크의 접근: Google vs NVIDIA

### 5.1 Google Gemini Robotics

[Gemini Robotics](https://deepmind.google/models/gemini-robotics/):

**두 가지 모델**:
1. **Gemini Robotics**: VLA 모델 (Gemini 2.0 + 행동 출력)
2. **Gemini Robotics-ER**: Embodied Reasoning 모델

**Gemini Robotics-ER 1.5** (2025.09 공개):
```
핵심 능력:
├── 물체 감지 (Object Detection)
├── 포인팅 (Pointing)
├── 궤적 예측 (Trajectory Prediction)
├── 그립 예측 (Grasp Prediction)
├── 3D 바운딩 박스
└── 멀티뷰 대응
```

**핵심 철학**: TPU 인프라 + 거대 언어 모델의 **추론 능력**을 로봇에 이식
- "이 사과는 물러서 꽉 쥐면 터질 수 있다" 같은 고차원 판단

**파트너**: Boston Dynamics, Agility Robotics, Agile Robots, Enchanted Tools

### 5.2 NVIDIA GR00T & Cosmos

[NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/gr00t):

**GR00T N1** (2025.03 GTC 발표):
```
┌─────────────────────────────────────────────────────────────┐
│ Dual-System Architecture                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [System 2 - 느린 사고]          [System 1 - 빠른 반응]     │
│  Vision-Language Module          Diffusion Transformer      │
│  환경 해석 + 계획 수립           연속적인 모터 명령 생성     │
│                                                             │
│  "무엇을 할지 생각"              "어떻게 움직일지 실행"      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**학습 데이터**:
- 실제 로봇 궤적 + 인간 비디오 + **합성 데이터**
- 780,000개 합성 궤적 = 6,500시간 = **11시간 만에 생성**
- 합성 데이터 추가로 성능 **40% 향상**

**버전 히스토리**:
| 버전 | 특징 |
|------|------|
| N1 | 첫 오픈 휴머노이드 파운데이션 모델 |
| N1.5 | GR00T-Dreams로 36시간 만에 개발, 향상된 일반화 |
| N1.6 | 전신 제어, Cosmos Reason 통합 |

### 5.3 World Foundation Model (WFM)

**NVIDIA Cosmos WFM**:
```
물리적 상상력의 엔진

"이 컵을 밀면 바닥으로 떨어져 깨질 것이다"
       ↓
실제로 해보지 않고 비디오처럼 예측

[Cosmos Predict] → 이미지/텍스트/비디오에서 움직임 예측
[Cosmos Reason]  → 애매한 지시 → 단계별 계획으로 변환
[Cosmos Transfer] → 구조화된 입력 → 현실적 합성 데이터
```

**Cosmos Reason**: Hugging Face Physical Reasoning 리더보드 **1위**

### 5.4 Google vs NVIDIA 비교

| 항목 | Google | NVIDIA |
|------|--------|--------|
| **핵심 철학** | 추론 (Reasoning) | 예측 (Prediction) |
| **기반 기술** | LLM + TPU | 시뮬레이션 + GPU |
| **강점** | 언어 이해, 상징적 사고 | 물리 법칙, 합성 데이터 |
| **접근법** | 인터넷 지식 → 로봇 | 시뮬레이션 → 현실 |
| **모델** | Gemini Robotics | GR00T + Cosmos |
| **인프라** | TPU 클러스터 | Isaac Sim + Omniverse |

---

## 6. 온디바이스 최적화

### 6.1 BitVLA: 1비트 VLA

[BitVLA](https://arxiv.org/abs/2506.07530) (2025.06):

**극단적 양자화**:
```
Full Precision: 32-bit (FP32)
       ↓
BitVLA: 1-bit (Ternary: -1, 0, 1)
```

**아키텍처**:
- LLM Backbone: BitNet b1.58 2B4T (1-bit)
- Vision Encoder: SigLIP-L → 1.58-bit로 압축 (디스틸레이션)
- Connector: 2-layer MLP (Full Precision 유지)

**성능**:
| 모델 | 메모리 | LIBERO 성능 |
|------|--------|-------------|
| OpenVLA | 4.4GB | 기준 |
| OpenVLA-OFT INT4 | 4.7GB | 높음 |
| **BitVLA** | **1.4GB** | 비슷 |

→ **29.8%** 메모리만 사용하면서 동등한 성능!

### 6.2 PD-VLA: 병렬 디코딩

[PD-VLA](https://arxiv.org/abs/2503.02310) (HKUST, 2025):

**문제**: VLA 모델은 3-5Hz, 로봇 제어는 20-30Hz 필요

**해결책**: Autoregressive 디코딩 → **병렬 고정점 반복**
```
기존 (순차적):
t=1 → a₁ → t=2 → a₂ → t=3 → a₃ (느림)

PD-VLA (병렬):
t=1 → [a₁, a₂, a₃] 동시 업데이트 → 수렴까지 반복
```

**성능**:
- **2.52x** 실행 주파수 향상 (7-DOF 매니퓰레이터)
- 재학습 없이 기존 모델에 적용 가능
- Action Chunking과 자연스럽게 결합

### 6.3 Real-Time Chunking (RTC)

[Physical Intelligence RTC](https://www.pi.website/research/real_time_chunking):

**문제**: 청크 경계에서 끊김/튀는 동작

**해결책**: Inpainting 방식
```
현재 청크: [■■■■■■□□□□]  (■=실행됨, □=예정)
                ↓
다음 청크: [????■■■■■■]  (?=새로 생성)
                ↓
인페인팅: [스무스하게 연결됨■■■■■■]
```

**핵심 아이디어**:
1. 현재 청크 실행 중에 다음 청크 **백그라운드 생성**
2. 이미 실행된 부분은 **고정**
3. 나머지는 **인페인팅**처럼 자연스럽게 이어붙임

**결과**: NeurIPS 2025 발표
- 높은 지연에도 **부드러운 동작** 유지
- 성냥 켜기 같은 **정밀 작업** 성공

### 6.4 온디바이스 최적화 요약

```
┌─────────────────────────────────────────────────────────────┐
│ 온디바이스 최적화 3축                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [BitVLA]        [PD-VLA]           [RTC]                   │
│  모델 크기 축소   디코딩 속도 향상    청크 연결 부드럽게      │
│  1-bit 양자화    병렬 고정점 반복    인페인팅 기법           │
│       │              │                  │                   │
│       └──────────────┴──────────────────┘                   │
│                      │                                      │
│              온디바이스 실행 가능                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 핵심 기술 비교표

### 7.1 주요 VLA 모델 비교

| 모델 | 개발사 | 파라미터 | 학습 방식 | 오픈소스 | 특징 |
|------|--------|----------|----------|----------|------|
| RT-1 | Google | - | IL | ❌ | 첫 Robotics Transformer |
| RT-2 | Google | 55B | VLM+IL | ❌ | 세계 지식 활용 |
| RT-1-X | Google | - | IL | ✅ | X-Embodiment 학습 |
| Octo | Berkeley | 93M | Diffusion | ✅ | 범용 정책의 시작 |
| OpenVLA | Stanford | 7B | IL+토큰화 | ✅ | 행동 토큰화 표준 |
| π₀ | Physical Intelligence | - | Flow Matching | ✅ | 연속 제어 품질 |
| CogACT | Microsoft | 7.6B | Diffusion | ✅ | 인지-행동 분리 |
| SmolVLA | HuggingFace | 450M | Flow Matching | ✅ | 경량 VLA |
| GR00T N1 | NVIDIA | - | Dual-System | ✅ | 휴머노이드 특화 |
| Gemini Robotics | Google | - | VLA | ❌ | 추론 능력 강조 |

### 7.2 학습 방식 비교

| 방식 | 대표 모델 | 장점 | 단점 |
|------|----------|------|------|
| **행동 토큰화** | OpenVLA | LLM 통합 용이 | 연속 제어 어려움 |
| **Diffusion** | Octo, CogACT | 다중 모드 분포 | 느린 샘플링 |
| **Flow Matching** | π₀, SmolVLA | 빠르고 안정적 | 구현 복잡 |
| **Dual-System** | GR00T N1 | 사고+행동 분리 | 복잡한 아키텍처 |

### 7.3 데이터셋 비교

| 데이터셋 | 에피소드 | 로봇 종류 | 특징 |
|----------|----------|----------|------|
| Open X-Embodiment | 1M+ | 22종 | 가장 큰 오픈 데이터셋 |
| DROID | 다양 | Franka | 다양한 환경/태스크 |
| Bridge V2 | - | WidowX | 테이블탑 조작 |
| LIBERO | 130+ 태스크 | - | VLA 벤치마크 |

---

## 8. 미래 전망

### 8.1 2025년 Physical AI의 핵심 축

```
┌─────────────────────────────────────────────────────────────┐
│ Physical AI의 두 기둥                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [예측하는 지능]              [사고-행동 동기화]             │
│  World Foundation Model       System 1 + System 2           │
│  물리적 미래 시뮬레이션        실시간 매끄러운 제어           │
│  "상상력"                     "반응성"                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 생태계 전쟁

| 축 | 경쟁 포인트 |
|-----|------------|
| 데이터 파이프라인 | 누가 더 효율적으로 로봇 데이터 수집? |
| 월드 모델 | 누가 더 정교한 물리적 상상력 제공? |
| 인프라 | 누가 더 큰 학습/추론 인프라 보유? |
| 온디바이스 | 누가 더 작고 빠른 모델 배포? |

### 8.3 패러다임 전환

```
과거: 데이터 경험 + 인프라 체력
현재: + 아키텍처 지능 + 온디바이스 실용성
미래: 모든 것이 결합된 Physical AI
```

**결론**: 2025년은 단순히 모델이 커진 것이 아니라, **패러다임의 완전한 전환**

---

## 9. 참고 자료

### 9.1 논문 및 프로젝트

| 이름 | 링크 |
|------|------|
| Open X-Embodiment | https://robotics-transformer-x.github.io/ |
| Octo | https://octo-models.github.io/ |
| OpenVLA | https://openvla.github.io/ |
| π₀ (Pi-Zero) | https://www.pi.website/blog/pi0 |
| CogACT | https://cogact.github.io/ |
| RoboVLMs | https://robovlms.github.io/ |
| BitVLA | https://arxiv.org/abs/2506.07530 |
| GR00T N1 | https://developer.nvidia.com/isaac/gr00t |
| Gemini Robotics | https://deepmind.google/models/gemini-robotics/ |
| Real-Time Chunking | https://www.pi.website/research/real_time_chunking |

### 9.2 GitHub 리포지토리

| 프로젝트 | GitHub |
|----------|--------|
| OpenVLA | https://github.com/openvla/openvla |
| Octo | https://github.com/octo-models/octo |
| OpenPI (π₀) | https://github.com/Physical-Intelligence/openpi |
| CogACT | https://github.com/microsoft/CogACT |
| BitVLA | https://github.com/ustcwhy/BitVLA |
| LeRobot | https://github.com/huggingface/lerobot |
| RoboVLMs | https://github.com/Robot-VLAs/RoboVLMs |

### 9.3 영상 출처

- 엥지 유니버스: 2025년 Physical AI 총결산 리뷰

---

## 업데이트 히스토리

| 날짜 | 내용 |
|------|------|
| 2026-01-23 | 초기 문서 작성 |

---

> **Note**: 이 문서는 RoArm-M3-Pro 프로젝트의 VLA/Diffusion Policy 방향 전환에 참고용으로 작성됨.
> LeRobot + SmolVLA 기반 구현 시 이 문서의 개념들을 참고할 것.
