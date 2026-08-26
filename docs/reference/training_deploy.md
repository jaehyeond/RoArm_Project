# Reference — SmolVLA Critical Rules · LeRobot Integration · Training Lessons

> 출처: 분리 전 `AGENTS.md` **526-620행**. 원본 전체는 `docs/archive/AGENTS_full_20260825_pre_split.md`.
> 아래 본문은 원본에서 **바이트 동일**하게 이동했다 (2026-08-25).
> `## Critical Rules`의 핵심 3건은 `AGENTS.md`의 HARD RULE **#2**(lerobot-train CLI + pretrained)·
> **#5**(JOINT_LIMITS 제거 금지)·**#6**(카메라 고정)이 그대로 커버하며 그쪽이 정본이다.
> 나머지(Loss↓≠좋은 모델, dataset_mean 시작, `n_action_steps=1`, Kinect 메인, v6+sim co-training)는
> 여기에만 있다 — **SmolVLA 학습/배포를 재개할 때 이 파일을 먼저 읽을 것.**

---

## Critical Rules (절대 지켜야 할 것)

### 학습

| Rule | Why |
|------|-----|
| **커스텀 학습 스크립트 작성 금지** | 공식 파이프라인의 정규화/스케줄러/전처리가 누락됨 |
| **`lerobot-train` CLI만 사용** | `run_official_train.py`가 래핑 |
| **`lerobot/smolvla_base` 사전학습 필수** | Action Expert가 사전학습 안 되면 평균 액션만 출력 |
| **Loss ↓ ≠ 좋은 모델** | L2 error + z-score + diversity 함께 확인 |

### 배포

| Rule | Why |
|------|-----|
| **dataset_mean 시작 위치** | [0,0,0,0,0,0] 시작은 OOD → 소심한 동작 |
| **`n_action_steps=1`** | Closed-loop: 매 스텝 새 추론 |
| **JOINT_LIMITS 절대 제거 금지** | 하드웨어 보호 |

### 데이터

| Rule | Why |
|------|-----|
| **수집 세션 내 카메라 고정 (삼각대/클램프)** | 단일 세션 위치 변경 = 그 데이터 무효. 데이터셋 설계 시 다양 viewpoint는 OK |
| **Azure Kinect 메인 사용** | 본 프로젝트 v6 = pyk4a single Kinect (IMX335/ZED Mini 보유 미장착) |
| **v6 50ep + sim demos co-training** | 4/24 결정. 단순 "100+ ep 수집"보다 sim demos (Mimic 500+) co-training이 stacking에 효과적 |

## LeRobot Integration

### 데이터 수집 방식
**현재 (4/01부터): Leader-Follower (L-F) 텔레옵**:
- Leader (USB0, 팔 #1, 그리퍼 클램프) → 손으로 조작
- Follower (USB1, 팔 #3, 카메라 촬영 대상) → 동기 추종
- `collect_data_manual.py` (L-F 모드) → Azure Kinect (pyk4a)
- v6 50ep는 모두 L-F로 수집 (4/01 전환 후)

**Legacy (참고만)**: 팔 1개 + 토크 OFF 수동 모드 (v1~v5에서 사용, v6부터 L-F)

### LeRobot 백업 파일
`lerobot_backup/` 폴더에 RoArm M3 통합 코드 백업:
- `roarm_m3.py` → `lerobot/lerobot/common/robot_devices/robots/` 에 복사
- `configs.py` → 동일 경로에 복사 (RoarmRobotConfig 추가)

### Strategy Pattern Architecture

```
RoarmRobot
├── connect() → strategy.initialize()
├── teleop_step() → strategy.generate_goal_positions()
├── capture_observation() → follower 읽기 + 카메라
├── send_action() → policy 추론용
└── disconnect() → strategy.cleanup()

Strategies:
├── KeyboardTeleopStrategy   (leader_arms={})
└── LeaderFollowerTeleopStrategy (leader_arms 설정 시)
```

## Training Lessons (실패에서 배운 것)

### 커스텀 학습 3회 실패 (Windows 환경)
| Attempt | Config | Result |
|---------|--------|--------|
| 1 | batch_size=1, vlm=False | 평균 액션 |
| 2 | batch_size=8, vlm=False | 평균 액션 |
| 3 | batch_size=8, vlm=True | 평균 액션 |

### Root Causes
1. **Action Expert 랜덤 초기화**: `SmolVLAConfig()` 대신 `from_pretrained("lerobot/smolvla_base")` 사용 필수
2. **정규화 누락**: 공식은 MEAN_STD preprocessor 적용
3. **LR 스케줄러 없음**: cosine decay + warmup 필요

### 해결
```bash
# 공식 CLI (이것만 사용!)
lerobot-train \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id=roarm_m3_pick \
  --dataset.root=lerobot_dataset_v4 \
  --batch_size=8 \
  --steps=50000 \
  --output_dir=outputs/smolvla_official
```

### 배포 실패 (2026-02-11, Linux 환경)
| Attempt | Data | Steps | Result |
|---------|------|-------|--------|
| 1 | 50ep (68% SHALLOW) | 50K | 그리퍼 미작동, Wrist_R -92° 폭주 |
| 2 | 동일 | 50K | 한 방향 드리프트, 파지 동작 없음 |

### 배포 실패 Root Causes
1. **데이터 부족**: 50 에피소드, DEEP 9개뿐 → 모델이 "내려가서 잡기" 안 배움
2. **Gripper 편향**: 대부분 프레임 gripper closed → open 동작 미학습
3. **Closed-loop drift**: 작은 오차 누적 → OOD → 한 방향 드리프트
4. **오프라인 ≠ 온라인**: 오프라인 L2=2.53° 양호해도 실제 배포 실패 가능
