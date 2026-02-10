# Isaac Lab RL 환경 구축 계획

> RoArm-M3-Pro 강화학습 파이프라인 구축 가이드
>
> 작성일: 2024-12-02
> 젠슨황 검증 완료 ✅

---

## 1. 개요

### 1.1 목표

```
┌─────────────────────────────────────────────────────────────────┐
│                    RoArm RL Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [Isaac Sim]  →  [Isaac Lab]  →  [RL Training]  →  [Deploy]   │
│       │              │               │                │         │
│       ▼              ▼               ▼                ▼         │
│   USD Robot      RL Env          PPO/SAC         Real Robot     │
│   Physics        Obs/Act         Policy          Sim2Real       │
│   Simulation     Reward          Training        Transfer       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Isaac Lab이란?

NVIDIA Isaac Lab은 **로봇 강화학습을 위한 GPU 가속 프레임워크**이다.

| 특징 | 설명 |
|------|------|
| **GPU 병렬화** | 수천 개 환경 동시 시뮬레이션 |
| **모듈형 설계** | 환경, 로봇, 태스크 분리 |
| **다양한 로봇** | Manipulator, Quadruped, Humanoid 지원 |
| **RL 라이브러리** | RSL-RL, RL-Games, SKRL, StableBaselines3 |

### 1.3 지원 종료 프레임워크

Isaac Lab은 다음 프레임워크들을 **대체**한다:
- ~~IsaacGymEnvs~~ (deprecated)
- ~~OmniIsaacGymEnvs~~ (deprecated)
- ~~Orbit~~ (deprecated)

---

## 2. 시스템 요구사항

### 2.1 현재 환경 (확인됨)

| 항목 | 현재 | 요구사항 | 상태 |
|------|------|----------|------|
| GPU | RTX 4070 Ti SUPER | RTX 3070+ | ✅ |
| Isaac Sim | 5.1.0 | 5.0+ | ✅ |
| OS | Windows 11 | Windows/Linux | ✅ |
| Python | - | 3.11 (Isaac Sim 5.x) | ⚠️ 확인 필요 |
| CUDA | 12.8 | 12.x | ✅ |

### 2.2 드라이버 요구사항

```
NVIDIA Driver: 576.80 (현재) ✅
권장: 580.65.06 이상 (Linux)
```

---

## 3. 설치 방법

### 3.1 방법 1: Pip 패키지 설치 (권장)

```bash
# 1. 가상환경 생성 (Python 3.11 필수!)
python -m venv isaaclab_env
isaaclab_env\Scripts\activate  # Windows

# 2. PyTorch 설치 (CUDA 12.8)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Isaac Lab + Isaac Sim 설치
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
```

### 3.2 방법 2: 소스에서 설치

```bash
# 1. Isaac Lab 클론
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 2. 의존성 설치
./isaaclab.sh --install  # Linux
.\isaaclab.bat --install  # Windows
```

### 3.3 첫 실행 주의사항

- 첫 실행 시 extension 다운로드로 **10분 이상** 소요
- NVIDIA EULA 동의 필요
- 이후 실행은 캐시 사용으로 빠름

---

## 4. RoArm-M3-Pro RL 환경 설계

### 4.1 환경 구성요소

```python
class RoArmEnv:
    """
    RoArm-M3-Pro 강화학습 환경
    """

    # Observation Space (관찰)
    observation = {
        "joint_pos": (4,),        # 관절 위치 [rad]
        "joint_vel": (4,),        # 관절 속도 [rad/s]
        "ee_pos": (3,),           # End-effector 위치 [m]
        "ee_orient": (4,),        # End-effector 자세 [quat]
        "target_pos": (3,),       # 목표 위치 [m]
    }
    # Total: 18 dimensions

    # Action Space (행동)
    action = {
        "joint_targets": (4,),    # 목표 관절 각도 [-1, 1] → 스케일링
    }
    # Total: 4 dimensions

    # Reward (보상)
    reward = (
        - distance_to_target      # 목표 거리 패널티
        - action_penalty          # 큰 동작 패널티
        + success_bonus           # 성공 보너스
    )
```

### 4.2 태스크 예시

| 태스크 | 설명 | 난이도 |
|--------|------|--------|
| **Reach** | End-effector를 목표 위치로 이동 | ⭐ |
| **Pick** | 물체 집기 | ⭐⭐ |
| **Place** | 물체 놓기 | ⭐⭐ |
| **Pick-and-Place** | 집고 옮기기 | ⭐⭐⭐ |

### 4.3 Reward 설계 (Reach 태스크)

```python
def compute_reward(self):
    # 1. 거리 보상 (가까울수록 높음)
    distance = torch.norm(ee_pos - target_pos, dim=-1)
    reward_distance = -distance

    # 2. 행동 패널티 (부드러운 동작 유도)
    reward_action = -0.01 * torch.sum(actions ** 2, dim=-1)

    # 3. 성공 보너스
    success = distance < 0.02  # 2cm 이내
    reward_success = 10.0 * success.float()

    return reward_distance + reward_action + reward_success
```

---

## 5. 학습 파이프라인

### 5.1 지원 RL 알고리즘

| 라이브러리 | 알고리즘 | 특징 |
|------------|----------|------|
| **RSL-RL** | PPO | 로봇 locomotion 특화 |
| **RL-Games** | PPO, SAC | 고속 GPU 학습 |
| **SKRL** | PPO, SAC, TD3 | 모듈형 설계 |
| **StableBaselines3** | PPO, SAC, A2C | 범용, 문서 풍부 |

### 5.2 학습 실행 예시

```bash
# Reach 태스크 학습 (PPO)
python source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Reach-RoArm-v0 \
    --num_envs 4096 \
    --headless

# 학습된 정책 테스트
python source/standalone/workflows/rsl_rl/play.py \
    --task Isaac-Reach-RoArm-v0 \
    --num_envs 32
```

### 5.3 학습 시간 추정

| 환경 수 | GPU | 예상 시간 (1M steps) |
|---------|-----|---------------------|
| 1024 | RTX 4070 Ti SUPER | ~30분 |
| 4096 | RTX 4070 Ti SUPER | ~15분 |
| 8192 | RTX 4090 | ~10분 |

---

## 6. Sim2Real 전이

### 6.1 Domain Randomization

```python
# 학습 시 무작위화로 실제 환경 대응력 향상
randomization = {
    "joint_friction": [0.5, 2.0],      # 관절 마찰
    "joint_damping": [0.8, 1.2],       # 관절 감쇠
    "payload_mass": [0.0, 0.5],        # 페이로드 질량
    "observation_noise": 0.01,         # 센서 노이즈
}
```

### 6.2 실제 로봇 배포 흐름

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Isaac Lab   │     │   ROS2       │     │  RoArm-M3    │
│  Policy      │ ──▶ │   Bridge     │ ──▶ │  Real Robot  │
│  (PyTorch)   │     │   Node       │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
   Observation         Joint Cmd            실제 동작
   (시뮬레이션)        (토픽 전송)          (하드웨어)
```

---

## 7. 구현 로드맵

### Phase 1: 환경 구축 (1-2일)

- [ ] Isaac Lab 설치 및 테스트
- [ ] RoArm USD를 Isaac Lab 포맷으로 변환
- [ ] 기본 Reach 환경 생성

### Phase 2: RL 학습 (2-3일)

- [ ] Reward function 구현
- [ ] PPO로 Reach 태스크 학습
- [ ] 학습 결과 시각화 및 분석

### Phase 3: 고급 태스크 (3-5일)

- [ ] Pick-and-Place 환경 구현
- [ ] Domain Randomization 적용
- [ ] 학습 성능 최적화

### Phase 4: Sim2Real (5-7일)

- [ ] ROS2 브릿지 연동
- [ ] 실제 로봇에 정책 배포
- [ ] Fine-tuning 및 검증

---

## 8. 참고 자료

### 공식 문서

- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [NVIDIA Developer - Isaac Lab](https://developer.nvidia.com/isaac/lab)
- [Isaac Lab NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab)

### 튜토리얼

- [Getting Started with Isaac Lab](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-first-robot-with-isaac-lab/01-what-is-reinforcement-learning.html)
- [Fast-Track Robot Learning (NVIDIA Blog)](https://developer.nvidia.com/blog/fast-track-robot-learning-in-simulation-using-nvidia-isaac-lab/)
- [Sim-to-Real with Spot (NVIDIA Blog)](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/)

### 논문

- [Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning](https://d1qx31qr3h6wln.cloudfront.net/publications/Isaac%20Lab,%20A%20GPU-Accelerated%20Simulation%20Framework%20for%20Multi-Modal%20Robot%20Learning.pdf)

---

## 9. 현재 상태 요약

```
✅ 완료:
   - Isaac Sim 5.1.0 설치
   - RoArm USD 모델 준비
   - Joint Drive 제어 확인
   - 시뮬레이션 동작 검증

⏳ 다음 단계:
   - ROS2 연동 (교수님 요청)
   - Isaac Lab 설치
   - RL 환경 구축
```

---

*"The future of robotics is simulation-first" - Jensen Huang*
