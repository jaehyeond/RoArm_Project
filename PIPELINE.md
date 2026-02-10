# RoArm M3 Pro — VLA 파이프라인 실행 계획

> 작성: 2026-01-26
> 목표: **Leader-Follower 데이터 수집 → ACT 학습 → SmolVLA Fine-tuning → 실시간 추론**

---

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RoArm M3 VLA 파이프라인                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 0         Phase 1          Phase 2        Phase 3                │
│  ┌──────┐       ┌──────────┐     ┌──────────┐   ┌──────────┐          │
│  │하드웨어│──────→│데이터 수집│────→│ ACT 학습 │──→│ACT 추론  │          │
│  │ 검증  │       │ (L-F)    │     │ (v0.1.0) │   │(실제로봇) │          │
│  └──────┘       └──────────┘     └──────────┘   └──────────┘          │
│                       │                                                 │
│                       │ 동일 데이터셋 재사용                              │
│                       ▼                                                 │
│  Phase 4         Phase 5          Phase 6                               │
│  ┌──────────┐   ┌──────────┐     ┌──────────┐                         │
│  │ LeRobot  │──→│ SmolVLA  │────→│VLA 추론  │                          │
│  │ 업그레이드│   │Fine-tune │     │(언어지시) │                          │
│  └──────────┘   └──────────┘     └──────────┘                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 왜 이 순서인가?

| 순서 | 이유 |
|------|------|
| ACT 먼저 | 현재 LeRobot v0.1.0에서 바로 학습 가능, SmolVLA는 업그레이드 필요 |
| 데이터 재사용 | LeRobot 데이터 포맷(parquet+MP4)은 버전 간 호환, 한 번 수집으로 두 정책 학습 |
| 점진적 검증 | ACT로 파이프라인 검증 → SmolVLA로 언어 기능 추가 |

---

## Phase 0: 하드웨어 검증

### 목적
모든 하드웨어가 동시에 안정적으로 작동하는지 확인

### USB 허브 구성

```
PC ──USB 케이블──→ [Waveshare HUB IN1]
                        │
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
       [USB1]        [USB2]        [USB3]
          │             │             │
       카메라        팔로워         리더
      (IMX335)      (COM8)        (COM9)
```

### Step 0.1: COM 포트 확인

```
장치 관리자 → 포트(COM & LPT) → "Silicon Labs CP210x" 2개 확인
- 팔로워: COM8 (또는 변경된 포트)
- 리더: COM9 (또는 변경된 포트)
```

### Step 0.2: L-F 독립 테스트

```powershell
# test_leader_follower.py의 COM 포트 확인 후 실행
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\test_leader_follower.py
```

**검증 체크리스트:**
- [ ] 양팔 USB 연결 성공
- [ ] 리더 토크 OFF → 손으로 자유 이동
- [ ] 팔로워가 리더를 실시간 미러링 (~50Hz)
- [ ] Ctrl+C → 리더 토크 복원 및 안전 종료

### Step 0.3: 카메라 뷰 확인

```powershell
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\test_camera.py
```

**카메라 배치 요구사항:**
```
┌─────────────────────────────────┐
│         카메라 (고정)            │  ← 테이블 앞 삼각대 or 고정 마운트
│            ↓ (내려다보기)        │
│   ┌───────────────────┐        │
│   │     작업 공간      │        │
│   │                   │        │
│   │  [물체]    [상자]  │        │
│   │                   │        │
│   │      [로봇 팔]     │        │
│   └───────────────────┘        │
└─────────────────────────────────┘
```

- 카메라는 작업 공간 전체 + 로봇 팔 + 물체 + 목표 위치를 모두 포함해야 함
- Eye-in-hand(그리퍼 장착) 보다 **고정 third-person view** 권장 (VLA 학습에 더 적합)
- IMX335 175° FOV는 너무 넓을 수 있음 → 거리 조절 또는 크롭 필요

**검증 체크리스트:**
- [ ] 카메라 화면에 로봇 팔 전체 보임
- [ ] 물체와 목표 위치(상자) 명확히 보임
- [ ] 그리퍼 동작이 카메라에서 구분 가능
- [ ] 조명 일정 (그림자/반사 최소화)

### Step 0.4: 3기기 동시 연결 테스트

LeRobot을 통해 3개 디바이스 (리더 + 팔로워 + 카메라) 동시 작동 확인:

```powershell
cd E:\RoArm_Project

E:\RoArm_Project\.venv\Scripts\python.exe lerobot/scripts/control_robot.py ^
  --robot.type=roarm_m3 ^
  --robot.leader_arms="{\"main\": \"COM9\"}" ^
  --robot.follower_arms="{\"main\": \"COM8\"}" ^
  --robot.cameras="{\"front\": {\"camera_index\": 0, \"fps\": 30, \"width\": 640, \"height\": 480}}" ^
  --control.type=teleoperate ^
  --control.fps=30
```

**검증 체크리스트:**
- [ ] "Leader-Follower Mode" 배너 출력
- [ ] 리더 이동 → 팔로워 미러링
- [ ] 카메라 영상 표시 (display_cameras 활성화 시)
- [ ] 30 FPS 안정 유지
- [ ] Ctrl+C로 정상 종료 (토크 복원, 연결 해제)

---

## Phase 1: 데이터 수집 (Leader-Follower)

### 목적
LeRobot record 모드로 50+ 에피소드 시연 데이터 수집

### 작업 환경 구성

```
필요 물품:
- 작은 물체 (블록, 큐브, 작은 장난감 등)
- 목표 용기 (상자, 컵, 트레이 등)
- 깨끗한 배경 (단색 매트 권장)
- 일정한 조명 (자연광 변화 최소화)

배치:
┌─────────────────────────────────┐
│        카메라 (약 40-50cm 높이)  │
│              ↓                  │
│  ┌───────────────────────┐     │
│  │  [물체1]         [상자]│     │
│  │       ↘          ↗    │     │
│  │        [로봇 팔]       │     │
│  │  [물체2]    [물체3]    │     │
│  └───────────────────────┘     │
│                                 │
│  5개 위치 × 10 에피소드 = 50개  │
└─────────────────────────────────┘
```

### Step 1.1: HuggingFace 로그인

```powershell
# HuggingFace 계정 필요
E:\RoArm_Project\.venv\Scripts\pip.exe install huggingface_hub
E:\RoArm_Project\.venv\Scripts\huggingface-cli.exe login
# 토큰 입력 (https://huggingface.co/settings/tokens 에서 생성)

# 환경 변수 설정
$env:HF_USER = "your_username"
```

### Step 1.2: 데이터 수집 실행

```powershell
cd E:\RoArm_Project

E:\RoArm_Project\.venv\Scripts\python.exe lerobot/scripts/control_robot.py ^
  --robot.type=roarm_m3 ^
  --robot.leader_arms="{\"main\": \"COM9\"}" ^
  --robot.follower_arms="{\"main\": \"COM8\"}" ^
  --robot.cameras="{\"front\": {\"camera_index\": 0, \"fps\": 30, \"width\": 640, \"height\": 480}}" ^
  --control.type=record ^
  --control.fps=30 ^
  --control.single_task="Pick up the object and place it in the box." ^
  --control.repo_id=%HF_USER%/roarm_m3_pick_place ^
  --control.num_episodes=50 ^
  --control.episode_time_s=30 ^
  --control.reset_time_s=10 ^
  --control.warmup_time_s=3
```

### 녹화 프로토콜

**에피소드당 플로우:**
```
[워밍업 3s] → [녹화 시작] → [시연 동작 ~15-25s] → [녹화 종료] → [리셋 10s]
                  │
                  ├── 1. 그리퍼 열기 (시작 위치)
                  ├── 2. 물체 위로 이동
                  ├── 3. 물체 위치로 하강
                  ├── 4. 그리퍼 닫기 (파지)
                  ├── 5. 들어올리기
                  ├── 6. 상자 위로 이동
                  ├── 7. 그리퍼 열기 (놓기)
                  └── 8. 홈 위치 복귀
```

**수집 전략 (50 에피소드):**

| 물체 위치 | 에피소드 수 | 설명 |
|-----------|------------|------|
| 위치 1 (정면 가까이) | 10 | 가장 쉬운 파지 |
| 위치 2 (정면 멀리) | 10 | 도달 범위 테스트 |
| 위치 3 (좌측) | 10 | 좌측 일반화 |
| 위치 4 (우측) | 10 | 우측 일반화 |
| 위치 5 (다양한 각도) | 10 | 랜덤 배치 |

**핵심 규칙:**
1. 매 에피소드 **성공적으로 완료** (실패한 시연은 데이터 품질 저하)
2. **부드러운 동작** (갑작스러운 움직임 금지)
3. **일관된 속도** (너무 빠르거나 느리지 않게)
4. **그리퍼 동작 명확** (열기→닫기 확실하게)
5. 에피소드 사이 **물체 위치 변경** (다양성 확보)

### Step 1.3: 데이터셋 검증

```powershell
# 수집된 데이터 확인
E:\RoArm_Project\.venv\Scripts\python.exe -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('%HF_USER%/roarm_m3_pick_place')
print(f'Episodes: {ds.num_episodes}')
print(f'Total frames: {ds.num_frames}')
print(f'Features: {list(ds.features.keys())}')
print(f'FPS: {ds.fps}')
"
```

**검증 체크리스트:**
- [ ] 50+ 에피소드 정상 저장
- [ ] 각 에피소드 ~900 프레임 (30fps × 30s)
- [ ] observation.state shape = (6,)
- [ ] observation.images.front shape = (480, 640, 3)
- [ ] action shape = (6,)
- [ ] 비디오 파일 정상 재생

### Step 1.4: HuggingFace Hub 업로드 (선택)

```powershell
# 데이터셋 업로드
E:\RoArm_Project\.venv\Scripts\python.exe -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('%HF_USER%/roarm_m3_pick_place')
ds.push_to_hub()
"
```

### 저장되는 데이터 구조

```
data/roarm_m3_pick_place/
├── meta/
│   ├── info.json               # 메타데이터 (fps, robot_type 등)
│   ├── episodes.json           # 에피소드별 메타
│   ├── stats/                  # 통계 (min/max/mean/std)
│   └── tasks.json              # 태스크 설명
├── data/
│   ├── episode_000000.parquet  # 프레임 데이터
│   ├── episode_000001.parquet
│   └── ...
└── videos/
    ├── observation.images.front_episode_000000.mp4
    └── ...
```

**프레임 내용 (parquet):**
```python
{
    "observation.state": float32[6],                # 팔로워 관절 각도
    "observation.images.front": uint8[480, 640, 3], # 카메라 RGB
    "action": float32[6],                           # 리더→팔로워 목표 각도
    "task": "Pick up the object and place it in the box."
}
```

---

## Phase 2: ACT 정책 학습

### 목적
수집된 데이터로 ACT (Action Chunking Transformer) 정책 학습

### ACT가 적합한 이유

| 특성 | 설명 |
|------|------|
| Action Chunking | 연속 동작을 청크 단위로 예측 → 부드러운 동작 |
| VAE | 다중 모달 동작 처리 (같은 위치에서 다른 경로 가능) |
| Temporal Ensemble | 여러 예측을 시간적 평균으로 안정화 |
| 빠른 추론 | ~100Hz (SmolVLA의 ~15Hz보다 훨씬 빠름) |
| 검증된 방법 | ALOHA, SO-100 등 다수 로봇에서 실증 |

### Step 2.1: 학습 실행

```powershell
cd E:\RoArm_Project

E:\RoArm_Project\.venv\Scripts\python.exe lerobot/scripts/train.py ^
  --dataset.repo_id=%HF_USER%/roarm_m3_pick_place ^
  --policy.type=act ^
  --output_dir=outputs/train/act_roarm_pick_place ^
  --device=cuda ^
  --batch_size=64 ^
  --steps=50000 ^
  --save_freq=10000 ^
  --log_freq=100 ^
  --wandb.enable=true ^
  --wandb.project=roarm_pick_place
```

### 학습 파라미터 가이드

| 파라미터 | 권장값 | 설명 |
|---------|--------|------|
| batch_size | 64 | RTX 4070 Ti 16GB에 적합 |
| steps | 50,000 | 50 에피소드 기준 충분 |
| save_freq | 10,000 | 5개 체크포인트 저장 |
| learning_rate | 1e-5 (기본) | ACT 기본값 사용 |
| chunk_size | 100 (기본) | Action horizon |

### Step 2.2: 학습 모니터링

```powershell
# WandB 대시보드 확인
# https://wandb.ai/<your_username>/roarm_pick_place

# 또는 로컬 로그 확인
# TensorBoard (설치 필요시)
E:\RoArm_Project\.venv\Scripts\pip.exe install tensorboard
E:\RoArm_Project\.venv\Scripts\tensorboard.exe --logdir=outputs/train/act_roarm_pick_place
```

**모니터링 지표:**
```
loss (L1)    : 0.5 → 0.05 이하로 수렴해야 함
kld_loss     : VAE KL-divergence, 적절한 수준 유지
grad_norm    : 폭발하지 않는지 확인
```

### Step 2.3: 체크포인트 구조

```
outputs/train/act_roarm_pick_place/
├── checkpoints/
│   ├── 010000/
│   │   └── pretrained_model/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       └── config.yaml
│   ├── 020000/
│   ├── 030000/
│   ├── 040000/
│   └── 050000/          ← 최종
└── wandb/
```

---

## Phase 3: ACT 실시간 추론

### 목적
학습된 ACT 정책으로 실제 로봇 자율 제어

### Step 3.1: 추론 실행 (평가 녹화)

```powershell
cd E:\RoArm_Project

E:\RoArm_Project\.venv\Scripts\python.exe lerobot/scripts/control_robot.py ^
  --robot.type=roarm_m3 ^
  --robot.follower_arms="{\"main\": \"COM8\"}" ^
  --robot.cameras="{\"front\": {\"camera_index\": 0, \"fps\": 30, \"width\": 640, \"height\": 480}}" ^
  --control.type=record ^
  --control.fps=30 ^
  --control.policy.path=outputs/train/act_roarm_pick_place/checkpoints/050000/pretrained_model ^
  --control.repo_id=%HF_USER%/eval_act_roarm_pick_place ^
  --control.num_episodes=10 ^
  --control.single_task="Pick up the object and place it in the box." ^
  --control.episode_time_s=30
```

**주의:** 추론 시에는 `leader_arms` 불필요 (정책이 제어)

### 추론 데이터 흐름

```
┌──────────────────────────────────────────────────────────────────┐
│                    매 프레임 (33ms, 30Hz)                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. capture_observation()                                         │
│     ├── safe_joints_angle_get(follower)  → state: [6]            │
│     └── camera.async_read()              → image: [480,640,3]    │
│                                                                   │
│  2. predict_action(observation, policy)                           │
│     ├── image: /255 → permute(C,H,W) → unsqueeze(batch)         │
│     ├── state: unsqueeze(batch) → to(cuda)                       │
│     ├── policy.select_action(obs)                                │
│     │   ├── normalize_inputs()                                    │
│     │   ├── model forward (ACT Transformer)                      │
│     │   ├── action chunking (queue 사용)                          │
│     │   └── unnormalize_outputs()                                │
│     └── action: squeeze(batch) → to(cpu)  → [6]                 │
│                                                                   │
│  3. send_action(action)                                           │
│     ├── max_relative_target 안전 클램핑                            │
│     └── joints_angle_ctrl(angles, speed=500, acc=200)            │
│                                                                   │
│  4. FPS 타이밍 유지 (busy_wait)                                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Step 3.2: 성능 평가

| 지표 | 측정 방법 | 목표 |
|------|----------|------|
| 성공률 | 성공 에피소드 / 전체 에피소드 | ≥ 60% (첫 시도) |
| 파지 성공 | 물체를 잡는 데 성공 | ≥ 80% |
| 배치 정확도 | 상자에 정확히 놓기 | ≥ 50% |
| 추론 속도 | 실제 FPS 확인 | ≥ 20 FPS |

### Step 3.3: 성능 개선 전략

성공률이 낮을 경우:

| 원인 | 해결책 |
|------|--------|
| 데이터 부족 | 50 → 100+ 에피소드 추가 수집 |
| 카메라 뷰 불일치 | 학습/추론 시 동일한 카메라 위치 |
| 동작 불안정 | max_relative_target 설정 (안전 클램핑) |
| 과적합 | 조기 체크포인트 사용 (10000 or 20000 step) |
| 물체 위치 변화 | 더 다양한 위치로 데이터 수집 |

---

## Phase 4: LeRobot 업그레이드

### 목적
최신 LeRobot으로 업그레이드하여 SmolVLA 지원 확보

### Step 4.1: 백업

```powershell
# 기존 코드 백업
copy E:\RoArm_Project\lerobot\lerobot\common\robot_devices\robots\roarm_m3.py ^
     E:\RoArm_Project\roarm_m3_backup.py

copy E:\RoArm_Project\lerobot\lerobot\common\robot_devices\robots\configs.py ^
     E:\RoArm_Project\configs_backup.py
```

### Step 4.2: LeRobot 업데이트

```powershell
cd E:\RoArm_Project\lerobot

# 현재 변경사항 확인
git status
git diff

# 최신 버전 가져오기
git stash            # 로컬 변경 임시 저장
git pull origin main # 최신 코드
git stash pop        # 로컬 변경 복원 (충돌 시 수동 해결)

# 의존성 재설치
E:\RoArm_Project\.venv\Scripts\pip.exe install -e ".[smolvla]"
```

### Step 4.3: roarm_m3.py 포팅

최신 LeRobot API 변경사항에 맞춰 roarm_m3.py 수정이 필요할 수 있음.

**확인할 변경사항:**
1. Robot Protocol 인터페이스 변경 여부
2. Camera 설정 방식 변경 여부
3. Dataset API 변경 여부
4. CLI 명령어 변경 (`control_robot.py` → `lerobot-record` 등)

**포팅 전략:**
- 기존 Strategy Pattern 아키텍처 유지
- API 변경 부분만 어댑터 추가
- 기존 기능 (키보드/L-F) 모두 보존

### Step 4.4: 포팅 검증

```powershell
# 텔레오퍼레이션 테스트 (새 API 사용)
lerobot-record ^
  --robot.type=roarm_m3 ^
  --robot.port=COM8 ^
  --robot.id=roarm_follower ^
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" ^
  --teleop.type=roarm_m3_leader ^
  --teleop.port=COM9 ^
  --teleop.id=roarm_leader ^
  --dataset.repo_id=%HF_USER%/test_upgrade ^
  --dataset.num_episodes=3 ^
  --dataset.single_task="Pick up the block"
```

> **참고:** 새 LeRobot API는 `--robot.type`과 `--teleop.type`을 분리합니다.
> RoArm M3의 경우 별도 등록이 필요할 수 있습니다.

---

## Phase 5: SmolVLA Fine-tuning

### 목적
수집된 데이터로 SmolVLA 기반 모델 미세조정

### SmolVLA 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                     SmolVLA (450M params)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌────────────┐    ┌────────────────┐           │
│  │ SigLIP-L │    │ Llama 2    │    │ Flow Matching  │           │
│  │ (Vision) │    │ (Language) │    │ (Action Head)  │           │
│  └────┬─────┘    └─────┬──────┘    └───────┬────────┘           │
│       │                │                    │                    │
│       ▼                ▼                    ▼                    │
│  ┌─────────────────────────────────────────────────┐            │
│  │                Fusion Module                      │            │
│  │  image_embed + text_embed → fused_representation │            │
│  └─────────────────────┬───────────────────────────┘            │
│                        │                                         │
│                        ▼                                         │
│              ┌─────────────────┐                                │
│              │  action: [6]     │  ← 6-DOF 관절 각도            │
│              └─────────────────┘                                │
│                                                                  │
│  입력: RGB 이미지 + "Pick up the cube and place it in the box"  │
│  출력: [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper] │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ACT vs SmolVLA 비교

| 특성 | ACT | SmolVLA |
|------|-----|---------|
| 입력 | 이미지 + 관절 상태 | 이미지 + **텍스트** |
| 언어 지시 | ❌ | ✅ "컵을 집어", "왼쪽으로 옮겨" |
| 모델 크기 | ~120M | 450M |
| 추론 속도 | ~100Hz | ~15Hz |
| 새 물체 일반화 | 제한적 | Zero-shot 가능 |
| VRAM 필요량 | ~6GB | ~12GB |

### Step 5.1: SmolVLA 학습

```powershell
cd E:\RoArm_Project\lerobot

# SmolVLA fine-tuning
lerobot-train ^
  --policy.path=lerobot/smolvla_base ^
  --dataset.repo_id=%HF_USER%/roarm_m3_pick_place ^
  --batch_size=32 ^
  --steps=20000 ^
  --output_dir=outputs/train/smolvla_roarm_pick_place ^
  --job_name=smolvla_roarm ^
  --policy.device=cuda ^
  --wandb.enable=true
```

### 학습 파라미터

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| batch_size | 32 | 16GB VRAM에서 안정적 (64는 OOM 가능) |
| steps | 20,000 | 50 에피소드 기준 권장 |
| base model | lerobot/smolvla_base | HuggingFace 사전학습 모델 |
| learning_rate | 1e-4 (기본) | Fine-tuning 적합 |
| VRAM 사용량 | ~12GB | RTX 4070 Ti 16GB ✅ |

### Step 5.2: 학습 모니터링

```
WandB에서 확인할 지표:
- train/loss: 꾸준히 감소해야 함
- train/action_loss: 행동 예측 정확도
- learning_rate: 스케줄러 동작 확인
- GPU memory: OOM 여부 확인
```

### Step 5.3: 모델 저장 구조

```
outputs/train/smolvla_roarm_pick_place/
├── checkpoints/
│   ├── 005000/pretrained_model/
│   ├── 010000/pretrained_model/
│   ├── 015000/pretrained_model/
│   └── 020000/pretrained_model/  ← 최종
└── wandb/
```

---

## Phase 6: SmolVLA 실시간 추론

### 목적
학습된 SmolVLA로 언어 지시 기반 실제 로봇 제어

### Step 6.1: 추론 실행

```powershell
lerobot-record ^
  --robot.type=roarm_m3 ^
  --robot.port=COM8 ^
  --robot.id=roarm_follower ^
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" ^
  --dataset.single_task="Pick up the object and place it in the box." ^
  --dataset.repo_id=%HF_USER%/eval_smolvla_roarm ^
  --dataset.episode_time_s=50 ^
  --dataset.num_episodes=10 ^
  --policy.path=outputs/train/smolvla_roarm_pick_place/checkpoints/020000/pretrained_model
```

### SmolVLA 추론 데이터 흐름

```
┌────────────────────────────────────────────────────────────────────┐
│                    매 프레임 (~15Hz)                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 관측 수집                                                       │
│     └── camera.read() → image: [480,640,3]                         │
│                                                                     │
│  2. SmolVLA 추론                                                    │
│     ├── 입력: image + "Pick up the object and place it in box"     │
│     ├── SigLIP: 이미지 → 비전 임베딩                                │
│     ├── Llama: 텍스트 → 언어 임베딩                                 │
│     ├── Fusion: 비전 + 언어 → 통합 표현                             │
│     ├── Flow Matching: 통합 표현 → 행동 예측                        │
│     └── 출력: action [6] (관절 각도)                                │
│                                                                     │
│  3. 로봇 제어                                                       │
│     └── joints_angle_ctrl(angles, speed=500, acc=200)              │
│                                                                     │
│  ★ SmolVLA는 관절 상태(state) 없이도 이미지+언어만으로 동작 가능     │
│  ★ 이것이 "End-to-End VLA"의 핵심                                   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Step 6.2: 다양한 언어 지시 테스트

SmolVLA의 핵심 장점 — 동일 모델로 다양한 태스크 수행:

```python
# 학습한 태스크
"Pick up the object and place it in the box."

# 유사한 지시 (일반화 테스트)
"Grab the cube and put it in the container."
"Move the block to the box."
"Place the object into the bin."
```

### Step 6.3: ACT vs SmolVLA 비교 평가

| 평가 항목 | ACT 결과 | SmolVLA 결과 |
|----------|---------|-------------|
| 파지 성공률 | __%  | __% |
| 배치 정확도 | __% | __% |
| 추론 속도 | __Hz | __Hz |
| 새 물체 일반화 | __% | __% |
| 언어 변형 대응 | N/A | __% |

---

## 트러블슈팅 가이드

### 공통 이슈

| 문제 | 원인 | 해결 |
|------|------|------|
| `SerialException` | COM 포트 사용 중 | 다른 프로그램 종료, 로봇 재연결 |
| `joints_angle_get() = None` | SDK 타임아웃 | 자동 재시도 로직 (5회) 동작 확인 |
| `RuntimeError: CUDA OOM` | VRAM 부족 | batch_size 줄이기 (64→32→16) |
| 팔로워 미러링 지연 | USB 대역폭 | 카메라 해상도 낮추기, FPS 낮추기 |
| 카메라 검은 화면 | 권한 문제 | Windows 카메라 앱 닫기 |

### 데이터 수집 이슈

| 문제 | 해결 |
|------|------|
| 에피소드 중 로봇 이탈 | max_relative_target 설정 |
| 녹화 FPS 불안정 | cameras fps를 15로 낮추기 |
| 데이터셋 저장 실패 | 디스크 공간 확인 (에피소드당 ~100MB) |
| 물체가 카메라에서 잘 안 보임 | 카메라 각도/거리 조정, 밝은 색 물체 사용 |

### 학습 이슈

| 문제 | 해결 |
|------|------|
| Loss가 수렴하지 않음 | steps 늘리기, learning_rate 조정 |
| 과적합 | 데이터 증강, 조기 체크포인트 사용 |
| SmolVLA OOM | batch_size=16, gradient_accumulation 사용 |
| WandB 연결 실패 | `--wandb.enable=false`로 오프라인 학습 |

---

## 하드웨어 요구사항 요약

| 구성요소 | 사양 | 상태 |
|---------|------|------|
| GPU | RTX 4070 Ti Super (16GB) | ✅ ACT + SmolVLA 학습 가능 |
| RAM | 32GB 권장 | 확인 필요 |
| 디스크 | 50GB+ 여유 공간 | 확인 필요 |
| 팔로워 | RoArm M3 Pro (COM8) | ✅ 연결됨 |
| 리더 | RoArm M3 Pro (COM9) | ⚠️ 연결 테스트 필요 |
| 카메라 | IMX335 USB (640x480@30fps) | ✅ 연결됨 |
| USB 허브 | Waveshare USB3.2 | ✅ 3포트 사용 |

---

## 일정 체크리스트

| Phase | 작업 | 상태 |
|-------|------|------|
| **Phase 0** | 하드웨어 검증 | |
| 0.1 | COM 포트 확인 | ⏳ |
| 0.2 | L-F 독립 테스트 | ⏳ |
| 0.3 | 카메라 뷰 확인 | ⏳ |
| 0.4 | 3기기 동시 연결 | ⏳ |
| **Phase 1** | 데이터 수집 | |
| 1.1 | HuggingFace 로그인 | ⏳ |
| 1.2 | 50 에피소드 수집 | ⏳ |
| 1.3 | 데이터셋 검증 | ⏳ |
| 1.4 | Hub 업로드 | ⏳ |
| **Phase 2** | ACT 학습 | |
| 2.1 | 학습 실행 (50K steps) | ⏳ |
| 2.2 | WandB 모니터링 | ⏳ |
| **Phase 3** | ACT 추론 | |
| 3.1 | 실시간 추론 테스트 | ⏳ |
| 3.2 | 성능 평가 | ⏳ |
| 3.3 | 개선 (필요시 데이터 추가) | ⏳ |
| **Phase 4** | LeRobot 업그레이드 | |
| 4.1 | 백업 | ⏳ |
| 4.2 | 업데이트 | ⏳ |
| 4.3 | roarm_m3.py 포팅 | ⏳ |
| 4.4 | 포팅 검증 | ⏳ |
| **Phase 5** | SmolVLA 학습 | |
| 5.1 | Fine-tuning (20K steps) | ⏳ |
| 5.2 | 모니터링 | ⏳ |
| **Phase 6** | SmolVLA 추론 | |
| 6.1 | 실시간 추론 | ⏳ |
| 6.2 | 언어 지시 변형 테스트 | ⏳ |
| 6.3 | ACT vs SmolVLA 비교 | ⏳ |
