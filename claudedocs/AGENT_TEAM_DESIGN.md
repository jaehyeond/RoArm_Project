# RoArm Agent Team Design

## Overview

RoArm M3 SmolVLA 파이프라인 개선을 위한 3-agent team 설계.
원본 패턴(emotion engine)을 로보틱스 ML 파이프라인에 맞게 변환.

```
원본 (Emotion Engine)              RoArm 적응
─────────────────────              ──────────
Backend: emotions.py          →   Pipeline Agent (학습)
Frontend: EmotionRadar        →   Deploy Agent (배포/추론)
DB: emotion_logs columns      →   Data Agent (데이터 분석/증강)
Lead: git + orchestration     →   Lead: git + orchestration (동일)
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LEAD (Opus 4.6)                   │
│  역할: 전체 조율, 결과 종합, git 관리, 사용자 대화    │
│  도구: 모든 도구 사용 가능                            │
│  규칙: git 작업은 Lead만 수행                         │
└──────────┬──────────────┬──────────────┬────────────┘
           │              │              │
    ┌──────▼──────┐ ┌────▼──────┐ ┌────▼──────┐
    │ Data Agent  │ │ Pipeline  │ │  Deploy   │
    │  (Sonnet)   │ │  Agent    │ │  Agent    │
    │             │ │ (Sonnet)  │ │ (Sonnet)  │
    └─────────────┘ └───────────┘ └───────────┘
```

## Current Problem Context

로봇이 물체까지 충분히 내려가지 않음:
- Elbow: +31.8° (목표: -64°)에서 수렴
- Gripper: 시각 조건부 개폐 아닌 시간적 패턴만 학습
- 모델 z-score ±1.5 범위만 출력 (보수적)
- 근본 원인: 51ep/20K steps 과소학습 + Elbow<0 데이터 부족

## Agent Definitions

### Agent 1: Data Agent

```yaml
name: "data-agent"
model: sonnet
subagent_type: python-expert
mode: plan  # plan approval 필요

role: 데이터 분석, 품질 평가, 수집 전략 수립

files_owned:
  - collect_data_manual.py          # 수동 데이터 수집 스크립트
  - analyze_dataset_phases.py       # 데이터셋 분석
  - lerobot_dataset_v3/             # 데이터셋 (읽기 전용)

files_readonly:
  - deploy_smolvla.py               # 배포 스크립트 참조
  - lerobot_dataset_v3/meta/info.json

tasks:
  1. 51 에피소드 품질 분석
     - 에피소드별 elbow 최저값, gripper 타이밍
     - "좋은 에피소드" vs "나쁜 에피소드" 분류 기준 설계
     - 데이터 분포 시각화 스크립트 작성

  2. 데이터 수집 전략 수립
     - 추가 수집 필요한 에피소드 유형 정의
     - collect_data_manual.py 개선안 (가이드 오버레이 등)
     - 목표: elbow < -30° 에피소드 50개 추가 수집

  3. 데이터 증강 가능성 검토
     - 기존 에피소드 temporal augmentation
     - action noise injection
     - 에피소드 리샘플링 (elbow<0 오버샘플링)

constraints:
  - git 작업 금지 (Lead만 수행)
  - 데이터셋 원본 수정 금지 (분석만)
  - 새 파일 생성 시 접두사: data_*
```

### Agent 2: Pipeline Agent

```yaml
name: "pipeline-agent"
model: sonnet
subagent_type: python-expert
mode: plan  # plan approval 필요

role: 학습 파이프라인 최적화, 하이퍼파라미터 튜닝

files_owned:
  - run_official_train.py           # 공식 학습 래퍼
  - test_inference_official.py      # 오프라인 추론 테스트

files_readonly:
  - outputs/smolvla_official/       # 체크포인트 (읽기)
  - lerobot_dataset_v3/meta/info.json
  - models/smolvla_base/            # 사전학습 모델

tasks:
  1. 학습 설정 최적화
     - 20K → 50K/100K steps 설정 준비
     - save_freq 조정 (더 자주 저장하여 최적 체크포인트 탐색)
     - learning rate schedule 검토

  2. 평가 파이프라인 구축
     - 체크포인트별 오프라인 평가 스크립트
     - 에피소드 재현 정확도 메트릭 (L2 error per joint)
     - elbow/gripper 특정 메트릭 추가

  3. 학습 전략 개선
     - 데이터 리샘플링 weight 설정
     - loss weight per joint (elbow 가중치 높이기) 가능 여부 확인
     - 현재 체크포인트에서 이어서 학습(resume) 설정

constraints:
  - git 작업 금지 (Lead만 수행)
  - GPU 학습 실행은 Lead 승인 후 (비용/시간)
  - lerobot 소스코드 수정 최소화
  - 새 파일 생성 시 접두사: train_*
```

### Agent 3: Deploy Agent

```yaml
name: "deploy-agent"
model: sonnet
subagent_type: python-expert
mode: plan  # plan approval 필요

role: 추론 루프 개선, 실시간 모니터링, 안전 시스템

files_owned:
  - deploy_smolvla.py               # 메인 배포 스크립트

files_readonly:
  - outputs/smolvla_official/       # 체크포인트
  - lerobot/lerobot/policies/smolvla/modeling_smolvla.py  # 모델 코드

tasks:
  1. 추론 루프 개선
     - 현재 문제: step 150 이후 수렴 (plateau)
     - action scaling factor 실험 (z-score에 multiplier)
     - temperature/noise injection으로 탈출 시도
     - 적응적 action step (큰 움직임 시 n_action_steps 조절)

  2. 실시간 모니터링 강화
     - 현재 관절별 z-score 실시간 표시
     - 목표 도달도 시각화 (elbow 진행률 바)
     - 수렴 감지: delta < threshold N회 연속 시 경고

  3. 안전 및 실용성
     - 수렴 감지 시 자동 정지 or 노이즈 주입
     - 실행 로그를 CSV로 저장 (후속 분석용)
     - 다중 체크포인트 비교 모드 (--checkpoint 인자)

constraints:
  - git 작업 금지 (Lead만 수행)
  - 실제 로봇 실행은 Lead 승인 후
  - JOINT_LIMITS 클램핑 제거 금지
  - 새 파일 생성 시 접두사: deploy_*
```

## Execution Plan

### Phase 1: 병렬 분석 (모든 에이전트 동시)

```
Data Agent:     데이터셋 51ep 품질 분석 + 수집 전략
Pipeline Agent: 학습 설정 검토 + 평가 스크립트 작성
Deploy Agent:   추론 루프 수렴 문제 해결안 설계
                ↓
Lead:           3 에이전트 결과 종합, plan approval
```

### Phase 2: 순차 실행

```
Step 1: Deploy Agent 코드 적용 → 현재 모델로 재테스트
Step 2: Data Agent 수집 전략으로 추가 데이터 수집 (수동)
Step 3: Pipeline Agent 설정으로 재학습 (50K+ steps)
Step 4: 새 체크포인트로 Deploy 재테스트
```

### Phase 3: 반복 (필요 시)

```
결과 분석 → 에이전트 재배치 → 개선 반복
```

## Task Tool Invocation Pattern

```python
# Lead가 3 에이전트를 동시 실행하는 패턴

# Agent 1: Data
Task(
    description="Analyze dataset quality",
    subagent_type="python-expert",
    model="sonnet",
    mode="plan",
    prompt="""[DATA AGENT]
    You are the Data Agent for the RoArm M3 SmolVLA project.

    Context:
    - 51 episodes, 13,010 frames, 30fps
    - Dataset: E:/RoArm_Project/lerobot_dataset_v3/
    - Problem: Robot elbow stays at +31.8° (target: -64°)
    - Only 33/51 episodes have elbow < 0°
    - Elbow < -60° is only 0.4% of all frames

    Your tasks:
    1. Analyze each episode quality (elbow depth, gripper timing)
    2. Design criteria for "good" vs "bad" episodes
    3. Propose data collection strategy for 50+ new episodes
    4. Write analysis script as data_quality_analysis.py

    Files you can modify: collect_data_manual.py, data_*.py (new)
    Files read-only: lerobot_dataset_v3/, deploy_smolvla.py
    DO NOT: run git commands, modify dataset originals
    """,
    run_in_background=True,
)

# Agent 2: Pipeline
Task(
    description="Optimize training pipeline",
    subagent_type="python-expert",
    model="sonnet",
    mode="plan",
    prompt="""[PIPELINE AGENT]
    You are the Pipeline Agent for the RoArm M3 SmolVLA project.

    Context:
    - Current: 20K steps, batch_size=8, smolvla_base pretrained
    - Checkpoint: outputs/smolvla_official/checkpoints/020000/
    - Problem: Model outputs conservative z-scores (±1.5 max)
    - Need: More aggressive predictions for elbow joint

    Your tasks:
    1. Design 50K-100K step training config
    2. Create per-checkpoint evaluation script
    3. Investigate joint-specific loss weighting
    4. Set up resume-from-20K training

    Files you can modify: run_official_train.py, test_inference_official.py, train_*.py
    Files read-only: outputs/, lerobot_dataset_v3/, models/smolvla_base/
    DO NOT: run git commands, start GPU training without approval
    """,
    run_in_background=True,
)

# Agent 3: Deploy
Task(
    description="Improve inference loop",
    subagent_type="python-expert",
    model="sonnet",
    prompt="""[DEPLOY AGENT]
    You are the Deploy Agent for the RoArm M3 SmolVLA project.

    Context:
    - deploy_smolvla.py: Azure Kinect + RoArm M3 + SmolVLA
    - Problem: Robot converges at step 150, stops moving
    - Model z-scores stay within ±1.5 (too conservative)
    - Elbow stuck at +31.8° (z=+0.22), needs to reach -64° (z=-3.04)

    Your tasks:
    1. Add action scaling/temperature to break plateau
    2. Add convergence detection (delta < threshold)
    3. Add per-step CSV logging for post-analysis
    4. Add real-time z-score display in OpenCV window

    Files you can modify: deploy_smolvla.py, deploy_*.py (new)
    Files read-only: modeling_smolvla.py, outputs/
    DO NOT: run git commands, remove JOINT_LIMITS safety
    """,
    run_in_background=True,
)
```

## Dependency Graph

```
                 ┌──────────────┐
                 │   Phase 1    │
                 │  (parallel)  │
                 └──────┬───────┘
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   ┌───────────┐  ┌──────────┐  ┌──────────┐
   │   Data    │  │ Pipeline │  │  Deploy  │
   │  분석     │  │  설정    │  │  개선    │
   └─────┬─────┘  └────┬─────┘  └────┬─────┘
         │              │              │
         └──────────────┼──────────────┘
                        ▼
              ┌──────────────────┐
              │   Lead 종합      │
              │  plan approval   │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │   Phase 2        │
              │  (sequential)    │
              └────────┬─────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   Deploy 적용   데이터 수집    재학습
   (즉시 테스트)  (수동 작업)   (GPU 시간)
         │             │             │
         └─────────────┼─────────────┘
                       ▼
              ┌──────────────────┐
              │   Phase 3        │
              │  검증 + 반복     │
              └──────────────────┘
```

## Communication Protocol

### Agent → Lead

각 에이전트는 작업 완료 시 다음 형식으로 보고:

```
[AGENT_NAME] REPORT
─────────────────────
Status: DONE / BLOCKED / NEEDS_REVIEW
Files modified: [list]
Files created: [list]
Key findings: [summary]
Recommendations: [list]
Blockers: [if any]
Next steps: [suggested]
```

### Lead → Agent (재배치)

```
[LEAD → AGENT_NAME] TASK UPDATE
────────────────────────────────
Previous task status: [completed/redirected]
New task: [description]
Context update: [new findings from other agents]
Priority: [high/medium/low]
```

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Agent가 동일 파일 동시 수정 | files_owned 명확히 분리 |
| 안전하지 않은 로봇 명령 | JOINT_LIMITS 제거 금지, Lead 승인 |
| GPU 학습 비용 | Pipeline Agent는 설정만, 실행은 Lead 승인 |
| 데이터셋 원본 손상 | Data Agent는 읽기 전용 |
| git 충돌 | Lead만 git 작업 수행 |

## Success Criteria

1. **Elbow**: -30° 이하 도달 (현재 +31.8°)
2. **Gripper**: 물체 근처에서만 열림 (현재 즉시 열림)
3. **Convergence**: 300 steps 내 plateau 없이 계속 진행
4. **Pick success**: 물체 집기 1/5 성공률 이상
