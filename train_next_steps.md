# 배포 실패 후 다음 단계 가이드

**Pipeline Agent - 2026-02-11**

---

## TL;DR (3줄 요약)

1. **문제**: 50K 학습 모델이 데이터셋 분산의 6-22%만 재현 (Conservative Policy)
2. **원인**: MSE loss 평균 회귀 + 37 epochs overfitting + 데이터 다양성 부족
3. **해결**: 25K 체크포인트 먼저 시도 → 실패 시 데이터 100+ 에피소드 재수집

---

## 즉시 실행: 25K 체크포인트 테스트

### Step 1: 오프라인 평가 (5분)

```bash
conda activate roarm
cd /home/cgxr/Documents/Robotics/RoArm_Project

# 25K 체크포인트 평가
python train_eval_checkpoints.py \
  --checkpoints 15000 25000 35000 50000 \
  --num-samples 50
```

**판단 기준**:
- Prediction std > 10° (최소 50% of dataset std)
- Wrist_R std > 10° (dataset 22.14°의 50%)
- Gripper std > 6° (dataset 13.65°의 50%)

### Step 2: 배포 테스트 (10분)

**25K 체크포인트로 변경**:

```bash
# deploy_smolvla.py 수정 (line 200 근처)
# 변경 전:
checkpoint_path = "outputs/smolvla_official/checkpoints/last"

# 변경 후:
checkpoint_path = "outputs/smolvla_official/checkpoints/025000/pretrained_model"
```

**Dry-run 실행**:

```bash
python deploy_smolvla.py --start-pos dataset_mean --max-steps 100 --dry-run
```

**성공 조건**:
- ✅ Gripper 열림 (10° 이상)
- ✅ Elbow 정상 extension (-20° ~ -50° 범위)
- ✅ Wrist_R 안정 (±30° 이내)
- ✅ 드리프트 < 10° (100 steps 내)

**실패 조건**:
- ❌ Gripper 안 열림 (< 5°)
- ❌ Elbow 드리프트 > 20°
- ❌ Wrist_R 폭주 (> ±50°)

→ 실패 시 즉시 Phase 2로 이동

---

## Phase 2: 데이터 재수집 (1-2일)

### Step 1: 카메라 위치 고정 확인

```bash
# 현재 카메라 위치 사진 촬영 (스마트폰)
# 위치 문서화: calibrate_azure_kinect.py 실행

python calibrate_azure_kinect.py
# → 위치 기록 저장: camera_position.txt
```

**주의**: 카메라 위치 변경 시 모든 데이터 무효!

### Step 2: 50 에피소드 추가 수집

**목표 분포**:

| 카테고리 | 목표 개수 | 조건 |
|---------|----------|-----|
| Elbow deep extension | 50개 | Elbow < -30° 구간 포함 |
| Wrist_R diverse | 30개 | Wrist_R 범위 -90° ~ +50° |
| 빠른 gripper 개폐 | 20개 | Gripper 5프레임 내 열림/닫힘 |

**수집 스크립트**:

```bash
# 토크 OFF 수동 모드
python collect_data_manual.py

# 에피소드별 저장 경로:
# collected_data_v2/episode_NNNN/
```

**품질 검증**:

```bash
# 수집 후 즉시 검증
python data_episode_quality.py --episode NNNN

# Elbow < -30° 포함 여부 확인
# Wrist_R 범위 확인
# Gripper 개폐 타이밍 확인
```

### Step 3: LeRobot v3 변환 + Validation Split

**변환 스크립트 수정** (`convert_to_lerobot_v3.py`):

```python
# Validation split 10% 추가
# 총 100 에피소드 → 90 train / 10 validation

# convert_to_lerobot_v3.py에 validation split 로직 추가
# --val-episodes 10 플래그 추가
```

**실행**:

```bash
# Train set (90 에피소드)
python convert_to_lerobot_v3.py \
  --input collected_data_v2 \
  --output lerobot_dataset_v4 \
  --task "Pick up the white box" \
  --val-split 10

# 결과:
# lerobot_dataset_v4/train/ → 90 episodes
# lerobot_dataset_v4/val/ → 10 episodes
```

### Step 4: 학습 설정 변경

**새 학습 스크립트** (`train_100k_with_val.py`):

```python
"""
100K steps training with validation split
"""
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.argv = [
    "lerobot-train",
    "--policy.type=smolvla",
    "--policy.pretrained_path=lerobot/smolvla_base",
    "--policy.push_to_hub=false",
    "--dataset.repo_id=roarm_m3_pick_v2",
    "--dataset.root=lerobot_dataset_v4",
    "--batch_size=8",
    "--steps=100000",  # 50K → 100K
    "--eval_freq=5000",  # Validation loss 모니터링
    "--save_freq=5000",  # 더 촘촘한 체크포인트
    "--log_freq=100",
    "--output_dir=outputs/smolvla_100k_val",
    "--num_workers=4",
    "--policy.device=cuda",
]

from lerobot.scripts.lerobot_train import main
main()
```

**실행**:

```bash
conda activate roarm
python train_100k_with_val.py

# 예상 학습 시간 (RTX 4090):
# 100K steps × ~0.3s/step = 8-10시간
```

**모니터링**:

```bash
# 로그 확인
tail -f outputs/smolvla_100k_val/logs/train_log.txt

# Validation loss 확인
grep "validation" outputs/smolvla_100k_val/logs/train_log.txt

# Train loss vs Val loss 비교
# Val loss 증가 시 overfitting → early checkpoint 선택
```

---

## Phase 3: 학습 후 평가 (2시간)

### Step 1: Checkpoint Evaluation

```bash
# 모든 체크포인트 평가 (5K, 10K, ..., 100K)
python train_eval_checkpoints.py \
  --output-dir outputs/smolvla_100k_val \
  --checkpoints 20000 40000 60000 80000 100000 \
  --num-samples 50 \
  --save-json checkpoint_eval_100k.json
```

**평가 메트릭**:

| 메트릭 | 목표 값 | 설명 |
|-------|---------|------|
| Overall L2 error | < 5.0° | 전체 오차 평균 |
| Wrist_R std | > 10° | 분산 확보 (dataset 22.14°의 50%) |
| Gripper std | > 6° | 분산 확보 (dataset 13.65°의 50%) |
| Elbow z-score range | [-3, +1] | 깊은 extension 가능 |
| Validation loss | 최소값 찾기 | Overfitting 방지 |

### Step 2: Best Checkpoint 선택

**선택 기준 (우선순위)**:

1. **Validation loss 최소** (overfitting 방지)
2. **Diversity 확보**: Wrist_R std > 10°, Gripper std > 6°
3. **L2 error < 5.0°** (정확도)
4. **Elbow z-score range** 포함 [-3, +1]

**예시**:

```
Checkpoint | Val Loss | Wrist_R Std | Gripper Std | L2 Error | 선택
-----------|----------|-------------|-------------|----------|------
20K        | 0.045    | 8.2°        | 4.1°        | 4.2°     | ❌ (Std 부족)
40K        | 0.032    | 12.5°       | 7.3°        | 3.8°     | ✅ (BEST)
60K        | 0.028    | 14.2°       | 8.1°        | 3.2°     | ⚠️ (Val loss OK)
80K        | 0.035    | 11.8°       | 6.9°        | 3.5°     | ❌ (Val loss 증가)
100K       | 0.042    | 9.1°        | 5.2°        | 3.1°     | ❌ (Overfitting)
```

→ **40K 체크포인트 선택**

---

## Phase 4: 신중한 배포 (1일)

### Step 1: Dry-run (10 trials)

```bash
# Best checkpoint (예: 40K) 설정
# deploy_smolvla.py 수정:
checkpoint_path = "outputs/smolvla_100k_val/checkpoints/040000/pretrained_model"

# Dry-run 10회
for i in {1..10}; do
  echo "Trial $i"
  python deploy_smolvla.py \
    --start-pos dataset_mean \
    --max-steps 100 \
    --dry-run
done
```

**분석**:

```bash
# 로그 수집
cat logs/deploy_*.csv > logs/dryrun_all.csv

# 통계 분석
python -c "
import pandas as pd
import numpy as np

df = pd.read_csv('logs/dryrun_all.csv')

# Per-joint std (10 trials 통합)
for joint in ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']:
    std = df[joint].std()
    print(f'{joint}: std={std:.2f}°')

# Drift 분석
for trial in range(10):
    trial_df = df[trial*100:(trial+1)*100]
    for joint in ['elbow', 'gripper', 'wrist_roll']:
        drift = trial_df[joint].iloc[-1] - trial_df[joint].iloc[0]
        print(f'Trial {trial+1}, {joint}: drift={drift:+.2f}°')
"
```

**PASS 조건**:
- 10 trials 중 8회 이상 성공
- Gripper 열림 (> 10°)
- Drift < 15° (100 steps)

### Step 2: Limited Deployment (5 trials, 사람 대기)

```bash
# 실제 로봇 실행 (토크 ON)
# 사람이 대기하여 긴급 정지 준비

python deploy_smolvla.py \
  --start-pos dataset_mean \
  --max-steps 300 \
  --trials 5
```

**Abort 조건** (즉시 중단):
- Elbow < -70° (하드웨어 한계)
- Gripper 물체 잡은 상태에서 열림 (낙하)
- Base > 180° (회전 과다)
- 2회 연속 실패 → 즉시 중단

**성공 조건** (5 trials):
- 3회 이상 pick-and-place 성공
- 물체 낙하 0회
- Closed-loop drift < 20° (300 steps)

---

## Abort 시나리오: Delta Action (최후 수단)

### 문제: 데이터 100+ 에피소드 + 100K steps 학습 후에도 실패

**증상**:
- Diversity 확보 (Wrist_R std > 10°)
- L2 error OK (< 5.0°)
- BUT 배포 시 closed-loop drift 여전히 발생

**원인**: Absolute position action의 누적 에러

**해결**: Delta action (상대 좌표) 재학습

### Delta Action 변환

```bash
# 데이터셋 변환 (absolute → delta)
python convert_to_delta_actions.py \
  --input lerobot_dataset_v4 \
  --output lerobot_dataset_v4_delta

# 학습 설정 변경
# train_100k_delta.py:
sys.argv = [
    "lerobot-train",
    "--policy.type=smolvla",
    "--policy.pretrained_path=lerobot/smolvla_base",
    "--policy.use_delta_joint_actions_aloha=true",  # ← Delta action
    "--dataset.repo_id=roarm_m3_pick_delta",
    "--dataset.root=lerobot_dataset_v4_delta",
    "--batch_size=8",
    "--steps=100000",
    # ... (나머지 동일)
]

# 학습 실행
python train_100k_delta.py
```

**배포 시 변경**:

```python
# deploy_smolvla.py 수정
# Prediction을 현재 위치에 더함
action = current_position + predicted_delta
```

**장점**:
- Closed-loop drift 완화
- 작은 움직임 중심 학습

**단점**:
- 전체 재학습 필요 (8-10시간)
- 초기 위치 에러 복구 어려움

---

## 파일 체크리스트

### 생성 필요 파일

- [ ] `train_100k_with_val.py` (100K steps 학습 스크립트)
- [ ] `convert_to_delta_actions.py` (delta action 변환)
- [ ] `train_validation_split.py` (90/10 split 유틸)

### 수정 필요 파일

- [ ] `convert_to_lerobot_v3.py` (validation split 로직 추가)
- [ ] `deploy_smolvla.py` (checkpoint path 25K로 변경)
- [ ] `train_eval_checkpoints.py` (diversity warning 추가) ✅ 완료

### 읽기 전용 파일

- `outputs/smolvla_official/checkpoints/*/` (기존 체크포인트)
- `lerobot_dataset_v3/` (기존 데이터셋)

---

## 예상 타임라인

| Day | 작업 | 시간 | 누적 |
|-----|-----|-----|------|
| **D+0** | 25K 체크포인트 평가 + 배포 테스트 | 1시간 | 1시간 |
| **D+0** | 실패 → 데이터 수집 계획 수립 | 0.5시간 | 1.5시간 |
| **D+1** | 카메라 고정 + 25 에피소드 수집 | 3시간 | 4.5시간 |
| **D+2** | 나머지 25 에피소드 수집 | 3시간 | 7.5시간 |
| **D+2** | v3 변환 + validation split | 0.5시간 | 8시간 |
| **D+2** | 100K steps 학습 시작 (밤새) | 8시간 | 16시간 |
| **D+3** | Checkpoint evaluation | 2시간 | 18시간 |
| **D+3** | Dry-run 10 trials | 1시간 | 19시간 |
| **D+3** | Limited deployment 5 trials | 1시간 | 20시간 |

**총 소요 시간**: 약 **3일** (작업 시간 기준)

---

## FAQ

### Q1: 25K 체크포인트도 실패하면?
A: 즉시 Phase 2 (데이터 재수집)로 이동. 25K는 "빠른 확인"이며, 근본 해결은 데이터 증강.

### Q2: 100 에피소드 수집 대신 50 에피소드 + augmentation은?
A: 이미지 augmentation은 가능하나, action augmentation은 위험 (물리적 제약 위반 가능). 실제 수집 권장.

### Q3: Delta action을 처음부터 시도하면?
A: 데이터 100+ 에피소드 먼저 확보 추천. Delta action은 closed-loop drift 완화이지 diversity 문제 해결 아님.

### Q4: Validation loss가 계속 감소하면?
A: 100K steps까지 학습 계속. 다만 diversity 메트릭 모니터링 필수 (std 감소 시 중단).

### Q5: 배포 시 gripper가 안 열리면?
A: Gripper std < 5° 확인 → 데이터 수집 시 "빠른 개폐" 에피소드 20개 추가.

---

**[PIPELINE AGENT] STATUS: READY FOR PHASE 1**
- 25K 체크포인트 테스트 준비 완료
- Phase 2-4 가이드 문서화 완료
- Lead approval 대기 중
