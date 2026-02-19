# 50K 학습 모델 배포 실패 종합 분석

**Pipeline Agent Report - 2026-02-11**

---

## 1. 학습 관점에서 현재 모델의 문제 진단

### 1.1 핵심 문제: **심각한 Under-Prediction (Conservative Policy)**

배포 vs 데이터셋 분산 비교:

```
Joint        | Dataset Std | Deploy Std | Std Ratio (%)
------------------------------------------------------------
base         |       21.75 |       1.61 |        7.4%  ← 92.6% 분산 손실
shoulder     |       26.08 |       4.07 |       15.6%  ← 84.4% 분산 손실
elbow        |       29.03 |       6.41 |       22.1%  ← 77.9% 분산 손실
wrist_pitch  |       26.00 |       4.18 |       16.1%  ← 83.9% 분산 손실
wrist_roll   |       22.14 |       1.72 |        7.8%  ← 92.2% 분산 손실
gripper      |       13.65 |       0.85 |        6.2%  ← 93.8% 분산 손실
```

**치명적 발견**: 모델이 데이터셋 분산의 **6-22%만** 재현하고 있음.

---

### 1.2 배포 로그 분석: 단방향 드리프트

**Run 3 (100 steps) 초기 vs 후반 비교**:

```
Joint        | 초기(1-10) | 후반(90-100) | 드리프트
---------------------------------------------------------
base         |     6.59°  |      11.85°  |   +5.27°  ← 일방향 증가
shoulder     |    41.91°  |      54.07°  |  +12.16°  ← 일방향 증가
elbow        |    16.89°  |      38.23°  |  +21.34°  ← 일방향 증가
wrist_pitch  |    61.06°  |      50.19°  |  -10.87°  ← 일방향 감소
wrist_roll   |    -4.60°  |     -11.24°  |   -6.64°  ← 일방향 감소
gripper      |     4.29°  |       2.61°  |   -1.68°  ← 일방향 감소
```

**Observation**:
- 모든 관절이 **한 방향으로만** 움직임 (양방향 탐색 없음)
- Closed-loop drift: 작은 에러가 누적되어 한쪽으로 쏠림
- Gripper 1-10 스텝: 평균 4.29° → 데이터셋 평균 9.61°보다 낮음 (열린 적 없음)

---

### 1.3 원인 진단

#### (A) **MSE Loss의 평균 회귀 편향**

SmolVLA는 MSE loss 사용 (`modeling_smolvla.py` line 791):
```python
loss = F.mse_loss(u_t, v_t, reduction="none")  # (B, T, 6)
losses.mean()  # 모든 차원 평균 → 평균 액션에 수렴
```

**문제점**:
- MSE는 평균을 향해 수렴하려는 경향 (분산 축소)
- 높은 분산 관절(wrist_roll std=22.14°) vs 낮은 분산 관절(gripper std=13.65°)을 동일하게 취급
- **해결 불가 (custom loss 금지)**

#### (B) **37 Epochs 과적합 위험**

학습 통계:
- 50K steps / (10803 frames / batch_size=8) = **37 epochs**
- Loss 0.126 (초기) → 0.007 (최종) = **94% 감소**
- 검증셋 없음 (모든 테스트 샘플이 학습에 사용됨)

**Overfitting 징후**:
- 학습 loss는 낮지만 배포 시 분산 손실 93.8%
- 모델이 **학습 데이터 평균을 암기**, 다양성 학습 실패

#### (C) **데이터 부족**

- **50 에피소드, 10,803 프레임** (목표: 100+ 에피소드)
- Elbow < -30° 샘플 부족 (깊은 extension 학습 안 됨)
- Wrist_R 다양성 부족 (std=22.14°이지만 편중 가능성)

---

### 1.4 오프라인 테스트 PASS vs 온라인 배포 실패 Gap

**Why offline test passed?**

`test_inference_official.py` 결과 (50K 체크포인트):
- L2 error: 2.53° (excellent)
- Overall Std: 21.55° (dataset: 21.75-29.03°)
- Elbow deep extension: -63.37° pred vs -65.39° GT (OK)

**BUT 온라인 배포 실패 이유**:

| 항목 | 오프라인 테스트 | 온라인 배포 | 설명 |
|-----|---------------|-----------|------|
| **입력** | GT 관측 (카메라+state) | 로봇 실제 관측 | Domain gap (조명, 각도, 진동) |
| **초기 상태** | 데이터셋 실제 초기 프레임 | dataset_mean 위치 | OOD 시작점 |
| **누적 에러** | 없음 (매 스텝 GT 리셋) | Closed-loop 누적 | 작은 에러 → 드리프트 |
| **n_action_steps** | 1 (테스트 시) | 1 (배포 시) | 동일 |
| **Evaluation** | L2, z-score, diversity | 실제 작업 성공률 | 메트릭 mismatch |

**핵심**: 오프라인 테스트는 **GT state에서 시작**하므로 conservative policy도 통과. 온라인은 **self-driving**이므로 작은 에러가 누적.

---

## 2. 선택지 A, B, C 분석

### 선택지 A: `--action-scale 2.0` 재시도

**장점**:
- 즉시 시도 가능 (학습 불필요)
- 분산 강제 확대 (×2.0 → std 1.61 → 3.22°)

**단점**:
- **Root cause 미해결**: MSE loss 평균 회귀 편향 그대로
- **Overshoot 위험**: 일부 관절만 폭주 가능 (Wrist_R -3° → -92° 재발 가능)
- **Gripper timing**: 2배 scale해도 열리는 타이밍 못 잡을 수 있음 (드리프트 방향 문제)
- **Closed-loop drift**: Scale만 키워도 한쪽 방향 드리프트는 해결 안 됨

**추천도**: ⚠️ **Conditional** - 1회 dry-run 시도 후 실패 시 즉시 중단

---

### 선택지 B: 데이터 추가 수집 (100+ 에피소드)

**장점**:
- **Root cause 해결**: 데이터 다양성 증가 → overfitting 완화
- 100 에피소드 → 20K 프레임 → 50K steps = **12 epochs** (현재 37 → 12로 1/3 감소)
- Elbow < -30° 50개, Wrist_R diverse 30개 → 다양성 확보
- Validation split 가능 (10 에피소드 홀드아웃)

**단점**:
- 시간 소요 (100 에피소드 = 5-8시간)
- MSE loss 평균 회귀 편향은 여전히 존재 (custom loss 불가)
- 카메라 위치 고정 필수 (변경 시 전체 무효)

**추천도**: ✅ **HIGHLY RECOMMENDED** - 가장 확실한 해결책

---

### 선택지 C: CSV 로그 상세 분석

**장점**:
- 추가 비용 없음
- Joint별 드리프트 패턴 파악 가능
- 어떤 관절이 먼저 폭주하는지 확인

**단점**:
- **실행 불가능한 인사이트**: 분석 결과로 나올 것은 "Wrist_R이 -3→-92° 폭주" 같은 증상 기술뿐
- **해결책 제시 불가**: MSE loss는 수정 불가, 데이터 부족은 수집 필요
- 시간 낭비 가능성 높음

**추천도**: ⚠️ **SKIP** - 이미 충분한 진단 완료, 추가 분석 불필요

---

## 3. 추가 선택지 제시

### 선택지 D: 다른 체크포인트 시도 (15K, 25K, 35K)

**근거**:
- 50K는 37 epochs (overfitting 위험)
- 25K = 18 epochs (덜 과적합)
- 15K = 11 epochs (최소 과적합, 하지만 underfitting 위험)

**장점**:
- 즉시 시도 가능 (학습 불필요)
- Overfitting 이전 체크포인트는 더 나은 일반화 가능

**단점**:
- 데이터 부족 문제는 동일
- Early checkpoint는 L2 error 높을 수 있음
- Closed-loop drift는 여전히 발생 가능

**추천도**: ⚠️ **CONDITIONAL** - 25K 체크포인트 1회 시도, 실패 시 B로 이동

---

### 선택지 E: Delta Action (상대 좌표)

**현재**: Absolute position action (절대 좌표)
**제안**: Delta action (현재 위치에서 상대 이동량)

```python
# 현재
action_t = [5.32, 40.78, 15.45, ...]  # 절대 좌표

# Delta
delta_t = [+2.32, +0.78, +2.45, ...]  # 이전 대비 변화량
action_t = current_pos + delta_t
```

**장점**:
- Closed-loop drift 완화 (에러 누적 감소)
- 작은 움직임 중심 학습 → 안전성 증가
- ALOHA에서 사용 (CLAUDE.md: `use_delta_joint_actions_aloha=false`)

**단점**:
- **전체 재학습 필요** (데이터 변환 + 50K steps)
- LeRobot config 변경: `policy.use_delta_joint_actions_aloha=true`
- 초기 위치 에러 시 복구 어려움

**추천도**: ⚠️ **Future Work** - B 먼저, 실패 시 재학습 단계에서 고려

---

### 선택지 F: Action Chunk Size 변경 (50 → 10)

**현재**: `chunk_size=50, n_action_steps=50` (default)
**제안**: `chunk_size=10, n_action_steps=1` (더 짧은 horizon)

**장점**:
- 짧은 horizon → 장기 예측 부담 감소
- Closed-loop 주기 짧아짐 (50 steps → 10 steps)
- Flow matching 10 denoising steps와 정렬

**단점**:
- **전체 재학습 필요**
- 짧은 chunk는 trajectory 학습 어려움 (연속성 손실)
- SmolVLA 논문 default는 50 (이유 있을 것)

**추천도**: ⚠️ **Not Recommended** - chunk_size는 검증된 default 유지

---

### 선택지 G: LR 증가 + 짧은 학습 (100K steps)

**근거**:
- 현재 LR: 0.0001 (default)
- 데이터 부족 시 높은 LR로 빠르게 학습, early stopping

**장점**:
- 학습 속도 증가
- Overfitting 전 멈춤 가능

**단점**:
- **Root cause 미해결** (데이터 부족 그대로)
- LR 높으면 불안정 (SmolVLA pretrained 망가질 수 있음)
- Validation set 없이 early stopping 불가

**추천도**: ❌ **Not Recommended** - 데이터 증강이 우선

---

## 4. 학습 관점 최적 전략 제안

### 전략: **Progressive Deployment with Data Augmentation**

#### Phase 1: 즉시 시도 (1-2시간)

**Step 1.1**: 25K 체크포인트 배포 테스트 (덜 과적합)
```bash
conda activate roarm
# deploy_smolvla.py에서 checkpoint path 변경
# outputs/smolvla_official/checkpoints/025000
python deploy_smolvla.py --start-pos dataset_mean --max-steps 100 --dry-run
```

**판단 기준**:
- 성공: Gripper 열림 + Elbow 정상 extension + Wrist_R 안정
- 실패: 25K도 동일 증상 → Phase 2로 이동

---

#### Phase 2: 데이터 재수집 (1-2일)

**Step 2.1**: 카메라 위치 고정 확인
```bash
# 삼각대/클램프로 고정, 위치 사진 문서화
python calibrate_azure_kinect.py  # 위치 기록
```

**Step 2.2**: 100+ 에피소드 수집 (타겟 분포)
```
목표:
- 총 100 에피소드 (현재 50 + 추가 50)
- Elbow < -30° 깊은 extension: 50개
- Wrist_R diverse (-90° ~ +50°): 30개
- 빠른 gripper 개폐: 20개
```

**Step 2.3**: Validation split
```python
# convert_to_lerobot_v3.py 수정
# 90 train / 10 validation split
```

**Step 2.4**: 학습 설정 변경
```bash
# run_official_train.py에서:
--steps=100000  # 50K → 100K
--eval_freq=5000  # validation loss 모니터링
--save_freq=5000  # 더 촘촘한 체크포인트
```

**예상 학습 스펙**:
- 100 에피소드 × 평균 200 프레임/ep = 20,000 프레임
- 100K steps / (20000/8) = **40 epochs** (현재 37과 비슷)
- **BUT 데이터 2배 → 다양성 확보**

---

#### Phase 3: 학습 후 평가 (1-2시간)

**Step 3.1**: Checkpoint evaluation
```bash
python train_eval_checkpoints.py \
  --checkpoints 20000,40000,60000,80000,100000 \
  --n_samples 50
```

**Step 3.2**: Per-joint diversity check
```python
# Wrist_R std > 15° (현재 1.72° → 최소 15° 목표)
# Gripper std > 5° (현재 0.85° → 최소 5° 목표)
```

**Step 3.3**: Validation loss 확인
```bash
# outputs/smolvla_official_v2/logs/
# train_loss vs val_loss 비교
# val_loss 증가 시 overfitting → early checkpoint 선택
```

---

#### Phase 4: 신중한 배포 (1일)

**Step 4.1**: Dry-run (10 trials)
```bash
python deploy_smolvla.py --start-pos dataset_mean --max-steps 100 --dry-run --trials 10
```

**Step 4.2**: Limited deployment (10 trials, 사람 대기)
```bash
python deploy_smolvla.py --start-pos dataset_mean --max-steps 300 --trials 10
```

**Abort 조건**:
- Elbow < -70° (하드웨어 한계 근접)
- Gripper 물체 잡은 상태에서 열림 (낙하)
- Base > 180° (회전 과다)
- 2회 연속 실패 → 즉시 중단

---

## 5. 최종 권장 사항

### 🔴 즉시 실행: 선택지 D (25K 체크포인트)
```bash
# 1. outputs/smolvla_official/checkpoints/025000 테스트
# 2. 1회 dry-run
# 3. 실패 시 → 데이터 수집으로 이동
```

### 🟡 우선순위 1: 선택지 B (데이터 100+ 에피소드)
```bash
# 1. 카메라 위치 고정 확인
# 2. 50 에피소드 추가 수집 (elbow<-30° 중심)
# 3. Validation split 10개
# 4. 100K steps 재학습
# 5. Checkpoint evaluation (20K, 40K, 60K, 80K, 100K)
# 6. Best checkpoint 선택 (validation loss 최소)
```

### 🟢 대안 (B 실패 시): 선택지 E (Delta Action)
```bash
# 1. 데이터 delta 변환
# 2. policy.use_delta_joint_actions_aloha=true
# 3. 100K steps 재학습
# 4. 배포 시 position controller 수정
```

### ❌ 비추천
- 선택지 A: action-scale 2.0 (overshoot 위험, root cause 미해결)
- 선택지 C: CSV 분석 (시간 낭비, 실행 가능한 인사이트 없음)
- 선택지 F: chunk_size 변경 (검증된 default 유지)
- 선택지 G: LR 증가 (데이터 부족 미해결)

---

## 6. 예상 소요 시간

| Phase | 작업 | 시간 | 누적 |
|-------|-----|-----|------|
| 1 | 25K 체크포인트 테스트 | 1시간 | 1시간 |
| 2.1 | 카메라 고정 확인 | 0.5시간 | 1.5시간 |
| 2.2 | 50 에피소드 추가 수집 | 5시간 | 6.5시간 |
| 2.3 | v3 변환 + validation split | 0.5시간 | 7시간 |
| 2.4 | 100K steps 학습 (RTX 4090) | 8시간 | 15시간 |
| 3 | Checkpoint evaluation | 2시간 | 17시간 |
| 4 | 배포 테스트 (dry-run + limited) | 2시간 | 19시간 |

**총 소요 시간**: 약 **2.5일** (작업 시간 기준)

---

## 7. 주요 학습 교훈

### 7.1 Loss ↓ ≠ Good Model
- Loss 0.007 (excellent) BUT 배포 시 분산 손실 93.8%
- **MSE loss는 평균 회귀 편향**
- L2 error, z-score만으로 평가 불충분 → **diversity (std) 필수**

### 7.2 Offline Test ≠ Online Performance
- 오프라인 테스트 PASS != 배포 성공
- GT state 시작 vs dataset_mean 시작 (OOD gap)
- Closed-loop drift 누적 (self-driving error accumulation)

### 7.3 Overfitting Without Validation Set
- 37 epochs, no validation → overfitting 감지 불가
- Validation split 10% 필수
- Early stopping 기준: validation loss 증가

### 7.4 Data Diversity > Data Size (초반)
- 50 에피소드 10,803 프레임 (충분한 양)
- BUT elbow < -30°, wrist_R diverse 샘플 부족
- **다양성 확보 후** 양 증가

### 7.5 Per-Joint Analysis 필수
- Overall metrics 숨기는 joint-specific 문제
- Wrist_R: dataset std=22.14° → deploy std=1.72° (7.8%)
- Gripper: dataset std=13.65° → deploy std=0.85° (6.2%)

---

## 8. 파일 생성/수정 제안

### 생성 파일
1. `train_config_100k.py`: 100K steps 학습 설정 (validation split 포함)
2. `train_eval_diversity.py`: Per-joint diversity 평가 스크립트
3. `train_validation_split.py`: 90/10 train/val split 유틸

### 수정 파일
1. `run_official_train.py`:
   - `--steps=100000`
   - `--eval_freq=5000` (validation loss)
2. `test_inference_official.py`:
   - Per-joint diversity 메트릭 추가
   - Wrist_R, gripper std 별도 출력

---

## 9. 결론

### 핵심 문제
**MSE Loss 평균 회귀 + 37 Epochs Overfitting + 데이터 다양성 부족** → Conservative Policy (분산 93% 손실)

### 최적 해결책
**데이터 100+ 에피소드 수집 (elbow<-30°, wrist_R diverse 중심) + 100K steps 재학습 + Validation split**

### 즉시 조치
1. 25K 체크포인트 1회 테스트 (overfitting 이전 체크포인트)
2. 실패 시 → 데이터 수집 단계 진입

### 장기 개선
- Delta action 시도 (closed-loop drift 완화)
- Validation set 기반 early stopping
- Per-joint diversity 모니터링

---

**[PIPELINE AGENT] STATUS: ANALYSIS COMPLETE**
- Next step: Phase 1 (25K checkpoint test) 승인 대기
- Lead approval required before GPU training
