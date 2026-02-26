# RoArm-M3-Pro + SmolVLA 랩미팅 발표
## 커스텀 로봇에서의 VLA 파인튜닝: 실패에서 성공까지

**발표일**: 2026-02-25
**발표 시간**: ~22분 (8개 슬라이드)

---

# 슬라이드 1. 지난 발표 리캡 + 오늘 발표 개요 (30초)

## 지난 발표에서 소개한 내용

```
[Azure Kinect DK] ──→ [SmolVLA 450M] ──→ [RoArm-M3-Pro 6-DOF]
    RGB 720P           Flow Matching         관절 위치 제어
    고정 삼각대         ~108ms/inference      STS3215 서보 x6
```

- SmolVLA = 350M VLM (frozen) + 100M Action Expert (trainable)
- RTX 4090 Laptop(16.7GB)에서 학습+추론 가능한 유일한 VLA

## 오늘 발표 내용

```
커스텀 학습 실패  →  batch_size 발견  →  데이터 개선  →  배포 성공!
   (3회 실패)         (8→64)          (v1→v3)        (5/5)
                                                       │
                                          그런데 특정 위치에서만 성공
                                                       │
                                          왜? → OOD 로봇 분석
```

### 발표자 노트
> 지난 발표에서 SmolVLA 시스템을 소개했습니다.
> 오늘은 그 이후 실험을 진행하면서 겪은 실패와 교훈, 그리고 최종 배포 성공까지의 과정을 보고합니다.

---

# 슬라이드 2. 커스텀 학습 3회 실패 (3분) — 가장 중요한 교훈

## Windows에서 3번 연속 실패

| 시도 | batch_size | VLM | 결과 |
|------|-----------|-----|------|
| 1차 | 1 | OFF | 평균 액션 출력 |
| 2차 | 8 | OFF | 평균 액션 출력 |
| 3차 | 8 | ON | 평균 액션 출력 |

**3번 모두 동일한 증상**: 어떤 이미지를 넣어도 같은 평균 액션만 출력

## 원인 분석: 3가지 결정적 실수

### 실수 1: 랜덤 초기화 (가장 치명적)
```python
# 우리가 한 것 (3번 다 이렇게 했음)
model = SmolVLAPolicy(SmolVLAConfig())     # ← Action Expert 랜덤 초기화!

# 올바른 방법
model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")  # ← 사전학습 로드
```
- `SmolVLAConfig()` → Action Expert가 **랜덤 가중치**로 시작
- `from_pretrained` → SO-100 로봇 **11,132 에피소드**로 사전학습된 가중치 로드
- 사전학습 없이는 Action Expert가 "조작이 뭔지" 자체를 모름

### 실수 2: 정규화 누락
- 공식 파이프라인: `MEAN_STD` 정규화 자동 적용 (zero-mean, unit-var)
- 커스텀 스크립트: 정규화 없이 raw 각도값 → 학습 불안정

### 실수 3: LR 스케줄러 없음
- 공식: cosine decay + 1K warmup steps
- 커스텀: 고정 learning rate → 수렴 실패

## 교훈

> **절대 커스텀 학습 스크립트 작성 금지**
> `lerobot-train` CLI가 정규화, 스케줄러, 전처리를 모두 자동 처리
> 이후 모든 학습은 공식 CLI만 사용

### 발표자 노트
> 이건 SmolVLA만의 문제가 아니라 VLA 모델 전반의 교훈입니다.
> 공식 파이프라인은 정규화, 스케줄러, 데이터 전처리가 모두 연결되어 있어서
> 한 부분만 빠져도 모델이 평균 액션만 출력합니다.
> 특히 from_pretrained은 절대 빼면 안 됩니다.

---

# 슬라이드 3. batch_size 발견 (2분)

## batch_size 8 vs 64

| 항목 | batch_size=8 (v1/v2) | batch_size=64 (v3) |
|------|---------------------|-------------------|
| VRAM 사용 | 2.03 GB (**12%**) | 9.85 GB (**59%**) |
| GPU 활용률 | 매우 낮음 | 적정 |
| Gradient 안정성 | 노이즈 큼 | 안정적 |
| 공식 권장 | X | **O (논문 기본값)** |

## GPU 활용 시각화

```
RTX 4090 Laptop VRAM (16.7GB):

batch_size=8:  [##                    ] 2.03GB (12%) ← GPU 88% 놀림!
batch_size=64: [###########           ] 9.85GB (59%) ← 적정 활용
전체 VRAM:     [######################] 16.7GB
```

## 왜 이전에 8로 했나?

- VRAM 부족이 아닌 **보수적 설정**이었음
- 논문(arXiv:2506.01844) 재확인 → 공식 기본값 = **64**
- `lerobot-train`은 gradient_accumulation 미지원 → 직접 batch_size 올려야 함

## batch_size가 왜 중요한가

- batch_size=8: 한 번에 8개 샘플로 gradient 계산 → 방향 불안정 → 위치 편향
- batch_size=64: 한 번에 64개 샘플 → gradient 안정 → 다양한 궤적 골고루 학습

### 발표자 노트
> batch_size=8은 GPU의 12%만 사용하고 있었습니다.
> 논문을 다시 확인한 결과 공식 권장값은 64였고, RTX 4090 Laptop에서 충분히 돌아갑니다.
> 이전에 8로 한 건 VRAM 부족이 아니라 단순히 보수적으로 설정한 것이었습니다.

---

# 슬라이드 4. 데이터 수집 진화 (3분)

## 3개 버전 비교

| 버전 | 에피소드 | 대상 | batch_size | 핵심 변화 |
|------|---------|------|-----------|----------|
| v1 | 50 | 흰 박스 | 8 | 기본 수집 (68% SHALLOW) |
| v2 | 43 | 흰 박스 | 8 | 7개 불량 제거, FK Z 기준 도입 |
| **v3** | **74** | **스펀지** | **64** | **다양한 위치, 공식 batch_size** |

## 핵심 발견: 깊이 측정 메트릭 오류

### 처음 — Elbow 각도로 깊이 판단 (잘못됨)

```
Elbow 각도 vs 실제 Z-height: 상관계수 r = 0.287 (거의 무관!)
→ Elbow 기준 DEEP = 9개(18%) → "데이터 심각하게 부족" 판단
```

### 수정 — Shoulder 각도 + FK Z-height 사용

```
Shoulder 각도 vs 실제 Z-height: 상관계수 r = -0.814 (강한 상관!)
→ FK Z 기준 DEEP = 51% → 데이터 품질이 과소평가되었음
```

### ESP32 Forward Kinematics 활용

```python
pose = arm.pose_get()  # → [x_mm, y_mm, z_mm, tilt°, roll°, grip°]
# z_mm = 실제 끝단 높이 (mm)
# Z < 100mm = DEEP, 100-200mm = APPROACH, > 200mm = SHALLOW
```

> Elbow 각도만 보면 데이터가 부족해 보이지만,
> 실제 FK Z-height로 보면 51%가 DEEP — 데이터 품질을 과소평가한 것

## 수집 프로토콜 7단계 (hand-guiding)

```
1. 시작 (init)       →  팔 초기 위치
2. 접근 + 열기       →  그리퍼 ~62° 열기
3. 프리그래스프       →  물체 위 위치 조정
4. 하강              →  그리퍼 열린 채로 내려감
5. 잡기              →  그리퍼 닫기 → ~24°
6. 들어올리기         →  z: 150mm → 287mm
7. 복귀              →  init 위치로
```

### 발표자 노트
> 데이터 분석에서 메트릭 선택이 중요했습니다.
> Elbow 각도와 실제 끝단 높이의 상관관계가 r=0.287밖에 안 되는데,
> 처음에 이걸로 DEEP/SHALLOW을 분류해서 데이터 품질을 크게 과소평가했습니다.
> ESP32의 FK를 이용해 실제 Z-height를 측정하니 훨씬 정확한 분류가 가능했습니다.

---

# 슬라이드 5. 배포: 실패에서 성공까지 (5분) — 핵심 스토리

## 배포 시도 타임라인

| # | 날짜 | 설정 | 결과 | 원인 |
|---|------|------|------|------|
| 1 | Feb 11 | bs=8, 50ep, dataset_mean 시작 | **Wrist_roll -92° 폭주** | 데이터 부족+편향 |
| 2 | Feb 23 | bs=8, 43ep, 50 steps chunk | **그리퍼 30-40°만** | 그리퍼 데이터 부족 |
| 3 | Feb 25 | bs=64, 74ep, closed-loop n=1 | **접근 OK, 잡기 실패** | 매 스텝 noise 변경 |
| 4 | Feb 25 | bs=64, 74ep, open-loop 1chunk | **접근+열기 OK, 시간 부족** | 50 step = 28%만 |
| **5** | **Feb 25** | **bs=64, 74ep, open-loop 4chunk** | **성공! 5/5 (100%)** | - |

## 배포 #1: Wrist_roll 폭주 (Feb 11)

```
100 steps 동안의 관절 궤적:

Wrist_R:  -3° ────────────────────→ -92°  (4σ OOD drift!)
Gripper:   2° ────────────────────→  4°   (한 번도 열리지 않음)
Base:      3° ────────────────────→ 12°   (한 방향만 이동)
Elbow:    13° ────────────────────→ 36°   (위로만 올라감)
```

- 50 에피소드 중 DEEP grasp 9개뿐
- 그리퍼 열기 데이터 부족 → 모델이 gripper open을 학습 못함
- Closed-loop에서 작은 오차 누적 → OOD → 단방향 drift

## 배포 #3: Closed-loop 잡기 실패 (Feb 25)

```
Closed-loop (n=1): 매 100ms마다 새 추론

Step 47: 그리퍼 closing (31°) ← 잡으려는 중!
Step 48: 새 추론, 새 noise → 그리퍼 opening (38°) ← 방해!
Step 49: 새 추론, 새 noise → 다시 closing (29°)
→ noise가 바뀌면서 잡기 동작이 완성되지 못함
```

## 배포 #5: Open-loop 4-chunk 성공! (Feb 25)

```
4 chunks × 50 steps = 200 steps (약 6.7초)

Chunk 1 (step 0-50):
  Base 0°→46° 회전, 스펀지 방향으로 접근

Chunk 2 (step 51-100):
  하강, 그리퍼 62°→24° (잡기 완료!)           ← 핵심!
  chunk 내 동일 noise → 50 step 동안 확실히 닫기 실행

Chunk 3 (step 101-150):
  팔 들어올리기 (z: 150mm → 287mm)

Chunk 4 (step 151-200):
  안정 유지 (grip=24°, 스펀지 보유)

결과: 5/5 (100%) 재현 가능!
```

## 실패→성공: 무엇이 바뀌었나

| 변경 사항 | 배포 #1 (실패) | 배포 #5 (성공) |
|-----------|--------------|---------------|
| batch_size | 8 | **64** |
| 에피소드 | 50 | **74** |
| 배포 방식 | closed-loop n=1 | **open-loop 4-chunk** |
| 시작 위치 | dataset_mean | **init** |

### 발표자 노트
> 이 슬라이드가 발표의 하이라이트입니다.
> 5번의 시도 중 처음 4번은 실패했지만, 각 실패에서 원인을 분석하고
> batch_size, 데이터, 배포 방식을 하나씩 개선해서 최종 성공에 도달했습니다.
> 특히 closed-loop에서 open-loop로 바꾼 것이 결정적이었습니다.

---

# 슬라이드 6. 기술적 인사이트: Closed-loop vs Open-loop (2분)

## 비교 표

| 항목 | Closed-loop (n=1) | Open-loop (4 chunks) |
|------|-------------------|---------------------|
| 재추론 빈도 | 매 100ms (10Hz) | 약 5초마다 (0.2Hz) |
| Flow Matching noise | **매번 다른 noise** → jitter | chunk 내 **동일 noise** → smooth |
| 잡기 동작 | noise가 close 신호 방해 | 50 step 동안 확실히 실행 |
| 환경 변화 대응 | 즉각 대응 | chunk 사이에서 대응 |
| 결과 | 접근 OK, 잡기 실패 | **5/5 성공** |

## Flow Matching에서 noise가 왜 문제인가

```
Flow Matching 추론:
  noise ~ N(0, I)        ← 추론할 때마다 새로 샘플링
  action = denoise(noise, observation, 10_steps)

Closed-loop (매 스텝 새 추론):
  t=47: noise_A → 그리퍼 닫기 (31°)
  t=48: noise_B → 그리퍼 열기 (38°)  ← noise 변경 → 동작 역전!
  t=49: noise_C → 그리퍼 닫기 (29°)
  → 정밀 동작 실패

Open-loop (chunk 내 동일 추론):
  t=1~50: noise_A → action_1...action_50 (일관된 궤적)
  → 그리퍼가 50 step에 걸쳐 확실히 닫힘
  → 정밀 동작 성공
```

## 설계 원칙

- SmolVLA action chunk = **50 steps = 1.67초** (30fps)
- 에피소드 평균 길이 = 178 프레임 ≈ 5.9초
- **4 chunks × 1.67초 = 6.7초** → 전체 에피소드 커버
- chunk 사이에서만 새 관측 → 장면 변화 대응 가능

> **"더 자주 관측 = 더 좋음"이 아님**
> Flow Matching 기반 VLA에서는 chunk 단위 실행이 정밀 동작에 유리

### 발표자 노트
> 이것은 Flow Matching 기반 VLA 배포의 핵심 통찰입니다.
> Diffusion이나 Flow Matching은 noise에서 action을 생성하는데,
> 매 스텝 새로 추론하면 noise가 달라져서 연속 동작이 끊깁니다.
> n_action_steps=50을 유지하면서 chunk를 이어붙이는 것이 해답이었습니다.

---

# 슬라이드 7. 왜 논문은 성공하는데 우리는 어려운가? (5분)

## 현실: VLA 배포는 논문만큼 쉽지 않다

| 모델 | 공식 주장 | 독립 검증 |
|------|----------|----------|
| SmolVLA | SO-100에서 78.3% | 커스텀 로봇 성공 사례 없음 |
| pi0 | 높은 성공률 주장 | UPenn 독립 평가: **24%** (300+ trial) |
| OpenVLA | RT-2-X 대비 +16.5% | fine-tuning 필요, ~100 demos |

## SmolVLA 78.3%의 조건

- SO-100 로봇 **(사전학습에 포함된 로봇)**
- 50 에피소드 x batch_size=64 x **200K steps**
- 카메라 위치 고정, 특정 workspace

## 우리 상황 vs 공식 조건

| 항목 | 공식 조건 | 우리 | 갭 |
|------|----------|------|-----|
| 로봇 | SO-100 (사전학습 **포함**) | RoArm-M3 (사전학습 **미포함!**) | **매우 큼** |
| 에피소드 | 50 (5위치 x 10반복) | 74 (위치 다양성 부족) | 보통 |
| batch_size | 64 | 64 | OK |
| steps | **200K** | 50K | **4배 부족** |
| 카메라 | 2-3대, 고정 | 1대 (Azure Kinect) | 보통 |

## 핵심 문제: RoArm-M3는 사전학습에 없다

### SmolVLA 사전학습 데이터 (직접 확인한 결과)

```
SmolVLA base 사전학습 데이터:
├─ 데이터셋: HuggingFaceVLA/community_dataset_v1
├─ 규모: 128개 데이터셋, 11,132 에피소드, 55명 기여자
├─ 로봇: SO-100 (단일 로봇 종류!)
└─ 확인: 논문 Section 5.1 + 저자(@danaaubakirova) 직접 확인
```

> 논문 원문 (Section 5.1 - Limitations):
> *"Our pretraining currently uses datasets collected from a **single robot type (SO100)**."*

### 사전학습 포함 vs 미포함의 차이

```
SO-100 (사전학습 포함)                    RoArm-M3 (사전학습 미포함)
┌─────────────────────┐                 ┌─────────────────────┐
│ VLM: 이미지 이해 ✓   │                 │ VLM: 이미지 이해 ✓   │
│                      │                 │                      │
│ Action Expert:       │                 │ Action Expert:       │
│  관절→위치 매핑 학습됨│ ← 이미 알고 있음 │  관절→위치 매핑 ???  │ ← 처음부터 배워야!
│  "shoulder 40°=      │                 │  RoArm의 기어비,     │
│   이 높이로 내려가기" │                 │  링크 길이, 관절 범위 │
│                      │                 │  전부 새로 학습 필요  │
└─────────────────────┘                 └─────────────────────┘
```

**전이되는 것** (사전학습에서 가져올 수 있는 것):
- "물체를 잡는다"는 일반적 개념
- 내려가기 → 열기 → 닫기 동작 시퀀스
- 이미지에서 특징 추출

**재학습 필요** (RoArm-M3 데이터로 새로 배워야 하는 것):
- 관절 각도 → 3D 끝단 위치 매핑
- 모터 특성, 기어비, 링크 길이
- 1대 카메라 viewpoint 적응

### 사전학습 효과 (SmolVLA 논문 Table 5)

| 조건 | 성공률 |
|------|-------|
| from_scratch (사전학습 없이) | 51.7% |
| **from_pretrained (사전학습 포함)** | **78.3%** |
| 차이 | **+26.6%p** |

→ 하지만 이건 **SO-100 (사전학습 로봇)** 기준!
→ RoArm-M3 같은 OOD 로봇은 사전학습 혜택이 제한적

## 데이터-성공률 곡선 (LoRA VLA 논문, SO-101 기준)

| 에피소드 수 | 성공률 | Vision 영향 점수 |
|-----------|--------|----------------|
| 20 | 18% | 0.8 (약함) |
| **50** | **45-52%** | 2.8 (보통) |
| 100 | 68-72% | 4.5 (강함) |
| 200 | 74-76% | 6.2 (매우 강함) |

> 우리 74 에피소드 ≈ **약 50% 성공률 구간**
> 그것도 **사전학습에 포함된 SO-101 기준**
> 사전학습 미포함 커스텀 로봇은 **더 많은 데이터가 필요**

## 커스텀 로봇 독립 성공 사례

| 연구자 | 로봇 | 데이터 | 결과 |
|--------|------|--------|------|
| Xavier O'Keefe (Correll Lab) | Franka + sim | 100 ep | ~40% 성공률 |
| Henry Hu | Franka + sim | 25 ep | 잡기 근처까지 (미성공) |
| **우리** | **RoArm-M3 (물리 로봇만)** | **74 ep** | **특정 위치 5/5 성공** |

> 물리 로봇에서 커스텀 VLA 배포 고성공 사례: **거의 공개되지 않음**

### 발표자 노트
> 이 슬라이드가 학술적으로 가장 중요합니다.
> SmolVLA의 78.3% 성공률은 SO-100, 즉 사전학습에 포함된 로봇에서의 결과입니다.
> 우리가 직접 확인한 결과, 사전학습 데이터에는 SO-100 단일 로봇만 포함되어 있고,
> RoArm-M3는 완전히 새로운 로봇입니다.
>
> pi0조차 독립 검증에서 24%밖에 안 나왔고,
> 커스텀 물리 로봇에서의 VLA 성공 사례는 거의 공개되지 않았습니다.
>
> 우리가 74 에피소드 + 50K steps + OOD 로봇 조건에서
> 특정 위치 5/5 성공을 달성한 것은 문헌 기준으로 의미 있는 결과입니다.

---

# 슬라이드 8. 종합 진단과 향후 계획 (2분)

## 현재 모델의 능력 평가

| 능력 | 상태 | 근거 |
|------|------|------|
| 스펀지 인식 (VLM) | **O** | frozen VLM이지만 사전학습으로 인식 가능 |
| 내려가기→열기→닫기 시퀀스 | **O** | init 시작 시 확인됨 |
| 특정 위치(Base ~45°) 잡기 | **O** | 5/5 재현 |
| **다양한 위치에서 잡기** | **X** | **평균 궤적 재생** |

## 왜 "보고 가기"가 아닌 "외운 궤적 재생"인가

- **위치 다양성 부족**: 59.5% CENTER, 25.7% FAR_RIGHT → 균등 분포 아님
- **학습 부족**: 50K steps (공식 200K의 1/4)
- **OOD 로봇**: 관절→위치 매핑을 74 에피소드로는 충분히 못 배움

## 향후 계획

| 우선순위 | 액션 | 기대 효과 |
|---------|------|----------|
| **1** | 스펀지를 모델이 가는 방향(~50°)에 놓고 추가 테스트 | 잡기 동작 전체 검증 |
| **2** | 에피소드 **150+개** 수집 (LEFT/CENTER/RIGHT 균등) | 공간 일반화 |
| **3** | **200K steps** 학습 (현재 50K의 4배) | OOD 로봇 충분 수렴 |
| **4** | scheduler_decay_steps를 학습 step과 맞추기 | LR 조기 붕괴 방지 |
| 5 | ACT(80M) 베이스라인 비교 | 데이터 vs 모델 문제 분리 |
| 6 | X-VLA(900M) 시도 — 7개 플랫폼 사전학습 | OOD 전이 성능 비교 |

## 결론

> **74 에피소드 + 50K steps + 사전학습 미포함 로봇**
> **= 현재 결과는 문헌과 완전히 일치합니다.**
>
> 모델이 망가진 게 아니라, **데이터와 학습이 아직 부족**한 것입니다.
> **150+ 에피소드 + 200K steps**가 되면 **70%+ 성공률**을 기대할 수 있습니다.

### 발표자 노트
> 결론은 두 가지입니다.
> 첫째, 현재 결과는 문헌과 일치하므로 모델에 문제가 없습니다.
> 둘째, 데이터 150+개와 200K steps 학습이 핵심 다음 단계입니다.
> 특히 X-VLA는 7개 플랫폼으로 사전학습되어 OOD 로봇에 더 유리할 수 있어서
> SmolVLA와의 비교 실험도 계획하고 있습니다.

---

# 참고문헌

1. SmolVLA: Mellou et al., "SmolVLA: A Small Vision-Language-Action Model for Affordable and Efficient Robotics", arXiv:2506.01844, 2025
2. LeRobot: github.com/huggingface/lerobot v0.4.4
3. SmolVLA Pretrained 78.3% vs Scratch 51.7%: SmolVLA paper Table 5
4. SmolVLA batch_size=64: SmolVLA paper Section 5.1
5. SmolVLA pretraining = SO-100 only: 논문 저자(@danaaubakirova) 직접 확인
6. pi0 독립 평가 24%: UPenn, 300+ trials
7. LoRA VLA 데이터-성능 곡선: SO-101 기준 실험
8. OpenVLA: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", arXiv:2406.09246, 2024
9. X-VLA: "X-VLA: Cross-Embodiment Vision-Language-Action Model", ICLR 2026

---

# 부록 (필요 시 참고)

## 전체 타임라인 요약

| 날짜 | 이벤트 | 결과 |
|------|--------|------|
| ~02-10 | Windows → Linux 이관 | 환경 구축 완료 |
| 02-11 | v1 학습 (50ep, bs=8, 50K) | 배포 실패: Wrist_roll 폭주 |
| 02-12 | v2 학습 (43ep, bs=8, 50K) | 데이터 분석, FK Z 도입 |
| 02-20 | Isaac Lab 셋업 | URDF→USD, RL 파이프라인 검증 |
| 02-24 | batch_size 8→64 발견 | 논문 확인, VRAM 실측 |
| **02-25** | **v3 학습 (74ep, bs=64, 50K)** | **배포 성공: 5/5 open-loop** |

## 하드웨어 구성

| 구성 요소 | 세부 사항 |
|-----------|---------|
| GPU | RTX 4090 Laptop (16.7 GB VRAM), CUDA 12.6 |
| OS | Ubuntu 22.04, Driver 580 |
| Python | 3.11.14 (conda env `roarm`) |
| 로봇 | RoArm-M3-Pro 6-DOF, STS3215 서보 x6, ESP32 |
| 카메라 | Azure Kinect DK (RGB 720P, pyk4a 1.5.0) |
| 프레임워크 | LeRobot 0.4.4 + PyTorch 2.7.1+cu126 |

## SO-100 vs RoArm-M3 비교

| 항목 | SO-100 (사전학습) | RoArm-M3 (우리) |
|------|-----------------|-----------------|
| DOF | 6 (5관절+그리퍼) | 6 (5관절+그리퍼) |
| 모터 | Feetech STS3215 | Waveshare 커스텀 서보 |
| 통신 | Dynamixel/Feetech 버스 | roarm_sdk (시리얼 JSON, ESP32) |
| 제어기 | Raspberry Pi | ESP32 |
| 카메라 | 2-3대 (top+wrist) | 1대 (Azure Kinect, 외부 고정) |
| LeRobot 공식 | **지원** | **미지원** (PR #820 미머지) |
| 사전학습 | **포함** | **미포함 (OOD)** |

## 대안 VLA 모델 (향후 비교 실험용)

| 모델 | 파라미터 | VRAM | 언어 | LeRobot | 사전학습 범위 |
|------|---------|------|------|---------|------------|
| SmolVLA | 450M | 10GB | O | O | SO-100만 |
| ACT | 80M | 4GB | X | O | 없음 |
| X-VLA | 900M | 14GB | O | O | **7개 플랫폼, 290K ep** |
| Diffusion | 100M | 8GB | X | O | 없음 |
