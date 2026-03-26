# BASELINE.md — 조작 능력 기반 확보 로드맵

> **최우선 과제 (#1 PRIORITY)**
> "기어다니지도 못하는데 날려고 한다" — 교수님 피드백 (2026-03-25)
> 논문/연구 방향 논의 전에 이 문서의 Stage를 순차적으로 달성해야 함.

---

## 현재 상태 (Stage 0 완료)

| 항목 | 값 |
|------|-----|
| 데이터 | 74 episodes, 1개 위치(Base ~45°), 스펀지 1개 |
| 학습 | 50K steps, batch_size=64, smolvla_base pretrained |
| 배포 | open-loop 4-chunk, **5/5 (100%) 성공** |
| 카메라 | Azure Kinect (외부, 고정) + ZED Mini (wrist, 마운트 예정) |
| 로봇 | RoArm-M3 ×2 (follower + leader) |
| 한계 | **1개 위치, 1개 물체, 1개 동작만 가능** |

---

## 로드맵 개요

```
Stage 1: 다양한 위치에서 스펀지 잡기          ← 다음 목표
Stage 2: 색상/물체 구분하여 잡기
Stage 3: 연속 동작 (잡기 → 내려놓기 → 다시 잡기)
Stage 4: 듀얼 암 협조 동작
Stage 5: 음성 명령 통합
```

각 Stage는 이전 Stage 성공이 전제조건. 건너뛰기 금지.

---

## Stage 1: Multi-Position Grasping (다양한 위치에서 잡기)

### 목표
- 작업대 5개 영역에서 스펀지를 안정적으로 잡기
- 성공 기준: **5-zone 평균 60%+ 성공률** (각 zone 최소 50%)

### 5-Zone 전략 (Radial Layout)

```
             FAR_CENTER (35ep)
            220-290mm, ±20°
           /                \
    MID_LEFT (25ep)    MID_RIGHT (25ep)
    120-220mm          120-220mm
    -90~-30°           30~90°
           \                /
         NEAR_CENTER (30ep)
          80-140mm, ±30°
               |
          OVERHEAD (15ep)
         60-160mm, 높이↑
         (place 동작용)
```

| Zone | 거리 | 각도 | 에피소드 | 주의사항 |
|------|------|------|---------|---------|
| NEAR | 80-140mm | ±30° | 30 | Shoulder singularity 주의 |
| MID_LEFT | 120-220mm | -90~-30° | 25 | |
| MID_RIGHT | 120-220mm | 30~90° | 25 | |
| FAR_CENTER | 220-290mm | ±20° | 35 | Elbow 190° 근접, 속도↓ |
| OVERHEAD | 60-160mm | 높이 100-200mm | 15 | Stage 3 place 동작 대비 |
| **합계** | | | **150** | **zone당 균형 필수** |

### 데이터 수집 규칙
- **카메라 위치 절대 변경 금지** (Azure Kinect 고정 유지)
- ZED Mini wrist 마운트 후 수집 시작 (마운트 전 수집한 데이터는 무효)
- zone별 quota 채우기 — 한 zone에 편중되면 MEAN_STD 정규화가 편향됨
- 에피소드 품질 7단계 유지: 시작→접근+열기→프리그래스프→하강→잡기→들기→복귀
- `collect_data_manual.py` 사용, 수집 중 OSD로 현재 zone 표시

### 학습 설정
```bash
# 기존 run_official_train.py 그대로 사용
# 변경: dataset path만 새 데이터셋으로
lerobot-train \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id=roarm_m3_multipos \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/smolvla_stage1
```
- **200K steps** (150ep = 더 많은 데이터 → 더 많은 학습 필요)
- smolvla_base에서 **처음부터** 학습 (기존 74ep 체크포인트 이어 학습 불가 — stats.json 변경)
- task text: `"Pick up the sponge\n"`

### 평가
- zone별 5회 × 5 zone = **25회 배포 테스트**
- CSV 로그: gripper_angle, elbow, FK_z, zone, success/fail
- zone별 성공률 heatmap 생성

### 배포 설정 변경 (deploy_smolvla.py)
- distal joint (3-5) 속도 cap: 300 (Wrist_R 폭주 방지)
- FAR zone: speed=300, acc=100
- 그리퍼 닫힘 verify step 추가 (리프트 전 gripper_angle 확인)

### 예상 소요
- 데이터 수집: 150ep × ~15초/ep = ~40분 순수 수집 + 준비 시간
- 학습: RTX 4090에서 200K steps ≈ 14-16시간
- 평가: 25회 배포 ≈ 1시간

### 성공 기준 → Stage 2 진입 조건
- [ ] 5-zone 평균 성공률 ≥ 60%
- [ ] 모든 zone에서 최소 1회 성공
- [ ] Wrist_R 폭주 0회
- [ ] 그리퍼 열림→닫힘 동작이 모든 성공 trial에서 관찰됨

---

## Stage 2: Color/Object Conditioning (색상·물체 구분)

### 목표
- 2-3가지 색상의 물체 중 지정된 것을 잡기
- 성공 기준: **올바른 물체 선택률 80%+, 파지 성공률 60%+**

### Language Conditioning 원리
SmolVLA는 VLM(SigLIP + SmolLM2) 기반이므로 언어 입력을 이미 처리할 수 있음:
- SigLIP 비전 인코더: 색상/물체 특징 이미 분리됨 (PIVOT 논문 확인)
- Action Expert: 언어+비전 조건에 따른 행동을 **학습 데이터로부터** 배워야 함

### 데이터 수집 계획

| 물체 | 색상 | 에피소드 | task text |
|------|------|---------|-----------|
| 스펀지 | 노란색 | 50 | `"Pick up the yellow sponge\n"` |
| 스펀지 | 파란색 | 50 | `"Pick up the blue sponge\n"` |
| 폼 큐브 | 빨간색 | 50 | `"Pick up the red cube\n"` |
| **합계** | | **150** | |

- 각 물체: 5 zone에 균등 분배 (10ep/zone/object)
- **Distractor 포함 수집**: 타겟 물체 옆에 다른 색 물체 배치 (50% 이상의 에피소드)
- Stage 1 데이터와 **혼합하지 않음** (task text가 다르므로 별도 데이터셋)
- 또는 Stage 1 + Stage 2 데이터를 하나의 데이터셋으로 합침 (task_index로 구분)

### 물체 선택 기준
- **그리퍼 폭 ≤55mm** (RoArm-M3 parallel gripper 한계)
- Deformable/semi-rigid 물체만 (rigid precision grasp는 tactile 필요)
- 추천: 스펀지(다양한 색), 폼 큐브, 소형 봉제인형

### 학습
```bash
lerobot-train \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id=roarm_m3_multiobj \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/smolvla_stage2
```

### 평가
- **Distractor test**: 2개 물체 동시 배치, 지정 물체만 잡는지 확인
- 물체당 × zone당 3회 = 최소 45회 테스트
- 실패 분류: 접근 실패 / 잘못된 물체 선택 / 파지 실패 / 들기 실패

### 성공 기준 → Stage 3 진입 조건
- [ ] 올바른 물체 선택률 ≥ 80% (distractor 있을 때)
- [ ] 파지 성공률 ≥ 60% (올바른 물체 선택 후)
- [ ] 3가지 물체 모두에서 최소 1회 성공

---

## Stage 3: Sequential Pick-and-Place (연속 동작)

### 목표
- 잡기 → 지정 위치에 놓기 → 다른 물체 잡기
- 성공 기준: **2-step 연속 동작 50%+ 성공률**

### SmolVLA 한계와 해결
SmolVLA는 **단일 프레임 + 단일 task text** 처리. 메모리/상태 없음.
→ **Subtask decomposition** 필수:

```python
# deploy_sequential.py 구조
subtasks = [
    {"task": "Pick up the yellow sponge\n", "stop": "gripper_closed"},
    {"task": "Place sponge on the right side\n", "stop": "gripper_opened"},
    {"task": "Pick up the red cube\n", "stop": "gripper_closed"},
]

for subtask in subtasks:
    policy.reset()  # chunk counter 리셋
    while not check_stop_condition(subtask["stop"]):
        action = policy.select_action(observation, subtask["task"])
        robot.send_action(action)
```

### Subtask 경계 감지
- **그리퍼 상태 기반**: joint[5] > threshold = 닫힘 → "pick" 완료
- **그리퍼 열림**: joint[5] < threshold = 열림 → "place" 완료
- **타임아웃**: 최대 N chunks 후 강제 다음 subtask

### 데이터 수집
- **pick-place 시퀀스를 1개 에피소드로 수집** (잡기→이동→놓기)
- LeRobot v3의 `meta/subtasks.parquet`에 구간 어노테이션
- 100 episodes (50 pick-place + 50 pick-place-pick)
- place 위치: OVERHEAD zone 활용

### 학습
- Subtask별로 별도 학습? vs 전체 시퀀스 학습?
  - **권장: subtask별 학습** — pick policy + place policy 분리
  - pi0, RT-2도 별도 policy call 사용
- 또는: 전체 시퀀스를 1개 policy로 학습하되 task text에 현재 phase 명시

### 성공 기준 → Stage 4 진입 조건
- [ ] 2-step 연속 동작 (pick → place) 성공률 ≥ 50%
- [ ] 3-step (pick → place → pick) 성공률 ≥ 30%
- [ ] Subtask 전환 시 충돌/드리프트 없음

---

## Stage 4: Dual-Arm Coordination (듀얼 암)

### 목표
- 두 팔의 협조 동작: A팔이 잡고 → 중간 지점에 놓으면 → B팔이 가져감
- 성공 기준: **handoff 성공률 40%+**

### 하드웨어 구성
```
ARM_L (leader → follower 전환): base ∈ [-90°, +10°]  ← 왼쪽 workspace
ARM_R (기존 follower):          base ∈ [-10°, +90°]  ← 오른쪽 workspace
Handoff point: 테이블 중앙 (양쪽 도달 가능)
```

### 접근 방식
- **Static workspace partition** (ALOHA 방식)
- **동시 동작 배제** — Sequential만 (충돌 위험 제로)
- 시퀀스: L arm picks → places at handoff → R arm picks from handoff

### 데이터 수집
- 각 팔 별도 수집: L arm 50ep + R arm 50ep (각자 자기 workspace)
- Handoff 시퀀스: 50ep (L pick → L place center → R pick center)
- 총 150ep

### 충돌 방지
- deploy script에서 base 각도 hard clamp (ARM_L: max +10°, ARM_R: min -10°)
- 동시 움직임 없음 — 한 팔 동작 완료 후 다음 팔 시작

### 성공 기준 → Stage 5 진입 조건
- [ ] 단일 팔 작업은 Stage 1-3 수준 유지
- [ ] Handoff 시퀀스 성공률 ≥ 40%
- [ ] 충돌 0회

---

## Stage 5: Voice Command Integration (음성 명령)

### 목표
- "빨간 큐브 잡아" → 로봇이 해당 물체 파지
- 성공 기준: **음성→텍스트→동작 파이프라인 end-to-end 80%+ 정확도**

### 아키텍처

```
음성 입력 → Whisper (STT) → task text → SmolVLA → robot action
```

### 구현
- **Whisper large-v3** (OpenAI) — 한국어/영어 모두 지원
- STT 출력을 task text 포맷으로 매핑: "빨간 큐브 잡아" → `"Pick up the red cube\n"`
- 매핑 테이블 (소수 물체) 또는 LLM 기반 자유 형식 변환

### 왜 마지막 Stage인가
- 음성은 **인터페이스**일 뿐 — 로봇 조작 능력과 무관
- Stage 1-4가 완성되어야 음성으로 트리거할 동작이 존재함
- ASR 파이프라인 자체는 1-2일이면 구현 가능

### 대안: End-to-End Speech VLA
- **VLAS** (ICLR 2025): 음성 직접 입력 VLA, 텍스트 대비 2-5% 차이
- 하지만 SmolVLA에는 음성 인코더 없음 → ASR 파이프라인이 현실적

---

## 참고 논문 & 기술 (에이전트 조사 결과)

### VLA 성공률 벤치마크

| 모델 | 크기 | Single Pick | Multi-Position | Language | 필요 데모 |
|------|------|------------|----------------|----------|----------|
| SmolVLA | 450M | ~90% (SO-101) | ±10cm 내 | 기본 지원 | 25-100 |
| OpenVLA-OFT | 7B | 97.1% (LIBERO) | 넓은 범위 | 강함 | 20-300 |
| pi0 | ~3B | 95% (학습 위치) | 24% (wild) | 강함 | 50-200 |
| GraspVLA | 1.8B | 93.3% (zero-shot) | BBox 기반 | BBox+언어 | 10/object |
| Octo | 93M | 72% (fine-tune) | 중간 | 약함 | ~100 |

### 저비용 암 성과 비교

| 플랫폼 | 가격 | 정책 | 성과 |
|--------|------|------|------|
| SO-100/101 | $110-120 | SmolVLA | 3위치 ~75% |
| Koch v1.1 | $250 | ACT | block stack 85%, color sort 72% |
| ALOHA | $20K | ACT | 6태스크 80%+ (50ep) |
| **RoArm-M3** | **$350** | **SmolVLA** | **1위치 100% (Stage 0)** |

### 핵심 참고 논문
- **GraspVLA** (CoRL 2025): synthetic pretraining → 10 demo/object로 sequential grasping
- **Long-VLA** (CoRL 2025): end-to-end long-horizon VLA
- **TwinVLA** (ICLR 2026): 두 single-arm VLA 합성 → 데이터 효율적 bimanual
- **MoS-VLA**: 1-shot adaptation 주장 (70-100%)
- **OpenVLA-OFT**: LoRA fine-tuning, LIBERO SOTA
- **VLAS** (ICLR 2025): speech-input VLA

### 데이터 효율 가이드라인
- 25-50 demos: 단일 위치 기본 동작
- 50-100 demos: 좁은 범위 다중 위치
- 100-150 demos: 안정적 도메인 성능
- 200+ demos: 다중 물체 + 다중 위치

---

## 즉시 실행할 액션 (Stage 1 시작)

### 하드웨어 준비
1. [ ] ZED Mini wrist 물리 마운트 (고정 확인)
2. [ ] Azure Kinect 위치 최종 확인 (이후 절대 변경 불가)
3. [ ] 5-zone 작업대 마킹 (테이프/마커로 영역 표시)

### 코드 수정
4. [ ] `collect_data_manual.py`: zone 자동 판별 OSD 추가 (FK 기반)
5. [ ] `deploy_smolvla.py`: distal joint speed cap 300, gripper verify step 추가
6. [ ] 배포 CSV 로그에 zone, gripper_angle, FK_z 컬럼 추가

### 데이터 수집
7. [ ] 150 episodes 수집 (zone quota 준수)
8. [ ] 수집 중 실시간 zone 분포 모니터링

### 학습 & 평가
9. [ ] 200K steps 학습 (smolvla_base pretrained)
10. [ ] 25회 배포 테스트 (5 zone × 5회)
11. [ ] zone별 성공률 heatmap 생성

---

## 실패 모드 대비

| 순위 | 실패 유형 | 확률 | 대응 |
|------|---------|------|------|
| 1 | OOD Drift | 완화됨 | open-loop 4-chunk 유지 |
| 2 | Gripper Timing | HIGH | verify step 추가, 속도↓ |
| 3 | Approach Angle Mismatch | MEDIUM | zone별 접근 각도 다양성 확보 |
| 4 | Elbow Singularity (FAR) | MEDIUM | JOINT_LIMITS + 속도↓ |
| 5 | Chunk Boundary 불연속 | LOW-MED | EMA smoothing |

---

> **원칙**: 각 Stage를 성공시킨 후에만 다음으로 진행.
> 연구 방향/논문 아이디어는 Stage 2+ 달성 후 구체화.
> "기어다닌 후에 날자."
