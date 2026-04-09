# EAAS: Event-Augmented Action Space — Research Proposal

> **Status**: DESIGN PROPOSAL (2026-04-09). 실행 금지. Stage 1b (v6 3-zone baseline) 완료 후 재승인 대상.
> **Purpose**: 유저 "범용성 있는 연구 방향" 요청에 대한 answer. `claudedocs/RESEARCH_TRENDS_2026.md` §1 (LLM-as-Planner) + v6 gripper empirical 자산 + prior art 반증 검색 결과 종합.
> **HARD RULE #4 준수**: Phase 3의 prior art 6편 전부 arXiv abstract 직접 확인 (Bash 검증 불가한 claim 없음).
> **Baseline-first 준수**: 이 문서는 "결정 제안"이지 "실행 명령"이 아님. Stage 1b 이전에 실행 금지.

---

## 0. 한 문장 정의

> **Consumer L-F teleop 데이터셋의 contact event(grasp close / release / push commit)를 state-derivative heuristic으로 자동 탐지해 action 값을 event-aligned form으로 데이터셋 변환 시점에 재파라미터화하는 protocol. VLA 아키텍처·하드웨어·학습 파이프라인 변경 없이 contact-action tail 실패를 완화하고, 동일 protocol이 SmolVLA / Octo / π₀에 공통 적용됨을 보인다.**

---

## 1. 문제 재정의 (abstraction 한 단계 올리기)

v6 gripper 실패 (2026-04-08 분석: `experiment_log_v6_deployment.md`) 는 **"그리퍼 하나의 문제"가 아니라 일반 병리**이다. 재정의:

**L-F teleop without force feedback 조건에서 모든 contact event는 공통적으로**:
1. **시간상 희소** — 에피소드 프레임의 1-2%
2. **값 공간에서 압축됨** — 리더 mechanical limit + force sensor 부재 → demonstrator가 tail에 도달 불가능
3. **성공의 결정 인자** — 한 프레임의 commit 실패가 에피소드 전체 실패로 이어짐
4. **Continuous policy가 구조적으로 mean-reverting** — flow matching / diffusion / regression 전부 density-biased sampling → tail 미학습

그리퍼는 이 병리의 대표 사례일 뿐. release / push commit / contact switch 등 모든 boundary event가 동일한 메커니즘으로 실패.

**유저의 3점 자연 ablation**:
- v1 (bad quality 50ep) → 0%, 그리퍼 미작동 + 단방향 drift
- v3 (good quality 74ep) → 100%, **하지만 FALSE POSITIVE (base ~45° 한 구간만)** — `tech_critical_lessons.md`
- v6 (HOME 50ep L-F) → reach ✅ / grasp ❌, 5/6차원 성공

→ **같은 병리의 서로 다른 노출**. v6는 reach가 고쳐져서 grasp 병리만 순수 분리된 상태 = **연구 관찰용 이상적 샘플**.

---

## 2. LLM-as-Planner 5편 중 이 병리에 직접 처방을 주는 것 = GenCHiP 단 1편

`claudedocs/RESEARCH_TRENDS_2026.md` §1.5 분석 재확인:

| 논문 | 추상 레이어 | 이 병리 해결? |
|---|---|---|
| SayCan (2204.01691) | Multi-step plan + affordance filter | ❌ 다른 층위 |
| Code as Policies (2209.07753) | LLM이 Python API 호출 | ❌ 다른 층위 |
| ProgPrompt (2209.11302) | 프로그램 생성 플래닝 | ❌ 다른 층위 |
| BrainBody-LLM (adrr.202500072) | Brain+Body 2-LLM hierarchy | ❌ 다른 층위 (sim feedback 의존, Isaac Lab 포기 충돌) |
| **GenCHiP (2404.06645)** ⭐ | "**with the right action space**, LLMs are capable of contact-rich" | ✅ **action space 자체를 고침** |

**GenCHiP의 transferable insight**: "모델이 아니라 action space를 고쳐라". 원 논문은 LLM 코드 생성 경로를 취하지만, **insight 자체는 continuous VLA에도 이식 가능**.

**VLA로의 번역**: action space 재설계 = 데이터 표현 재설계 → event-aligned reparameterization.

---

## 3. Prior Art 반증 검증 결과 (HARD RULE #4)

### 3.1 치명적 prior art 후보 6편 (전부 arXiv abstract 직접 확인)

| # | 논문 | arXiv | venue | 상태 | 우리 아이디어 killer? |
|---|---|---|---|---|---|
| 1 | **Beyond the Majority: Long-tail Imitation Learning for Robotic Manipulation** (Zhu et al.) | **2602.06512** | **ICRA 2026 accepted** | ✅ abstract 확인 | **NO — 다른 층위**. 그들의 long-tail = **task-budget inter-task** (head task 데이터 많음 / tail task 적음). 우리 문제 = **한 task 안에서 action value distribution 의 tail**. 층위가 다름. **반드시 인용 + 명시적 table 구분 필수**. 프레임 이름 겹침이 리뷰어 첫 공격 포인트 |
| 2 | **ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation** (Yu et al.) | **2505.22159** | **NeurIPS 2025** | ✅ abstract 확인 — "external force sensing as first-class modality within VLA" | **NO — HW 불일치**. Force sensor 전제. 우리 $130 consumer HW + L-F teleop 은 force sensor 없음. **"ForceVLA가 요구하는 하드웨어가 없을 때 어떻게?"** 포지셔닝 가능 |
| 3 | **FAVLA: Force-Adaptive Fast-Slow VLA for Contact-Rich** | 2602.23648 | arXiv 2026-02 | ✅ | **NO** — 동일 (force sensor 전제) |
| 4 | **VLA-Touch: Dual-Level Tactile Feedback for VLA** (NUS) | 2507.17294 | arXiv 2025-07 | ✅ | **NO** — tactile sensor 전제 |
| 5 | **GenCHiP** (Burns et al., Google DeepMind) | 2404.06645 | arXiv 2024-04 | ✅ 재확인 abstract: *"with the right action space, LLMs are capable..."* | **NO** — LLM code-gen 경로. 우리는 VLA continuous policy. 같은 원리(action space matters), 다른 경로. **inspiration으로 명시 인용** |
| 6 | **Learning Diffusion Policy from Primitive Skills for Robot Manipulation** (Gu et al.) | 2601.01948 | arXiv 2026-01 | ✅ | **NO** — global instruction 분해용 primitive, contact event tail 아님 |

### 3.2 우리 프레임을 뒷받침하는 community signal (peer review 아님, 하지만 증거)

- **Correll Lab Medium (2025-12-15)**: *"When Fine-Tuning Hurts: Failure Modes of Visuomotor Imitation Learning on a Low-Cost Robot"* — 커뮤니티가 **같은 문제를 공개적으로 고민 중**. peer review 아니지만 research relevance 증거.
- **arXiv 2512.11921**: *"Towards Accessible Physical AI: LoRA-Based Fine-Tuning of VLA Models for Real-World Robot Control"* — 다른 축(parameter efficiency)으로 consumer 접근. 병행 인용.
- **arXiv 2506.01844**: SmolVLA 원 논문. 인프라 인용.
- **arXiv 2602.22818**: LeRobot 원 논문 (2026-02). 인프라 인용.

### 3.3 Honest 판정
- **사라진 주장**: "VLA contact-rich 최초", "long-tail imitation 최초", "force-aware VLA 최초"
- **여전히 주장 가능한 범위 (3개 한정어 교집합)**:
  1. **Consumer HW 조건** (L-F teleop, no force sensor, $130 arm, < 100 demos, single RGB)
  2. **Action-value intra-task long-tail** (vs Beyond the Majority의 task-budget inter-task)
  3. **Protocol-level intervention** (재학습 필요해도 `lerobot-train` CLI 그대로, HW 수정 없음)
- 이 세 한정어의 교집합은 반증 검색 후에도 **열려 있음**.

---

## 4. "범용성" 4축 분해 및 EAAS가 만족하는 이유

| 축 | 의미 | EAAS 대응 |
|---|---|---|
| **Task-general** | 스펀지 외 다양한 조작 | Event Detector가 phase-based (task-agnostic). sponge/box/cup 모두 close event 동일 검출 |
| **Architecture-general** | SmolVLA 외 다른 VLA | C2는 데이터 레이어 → SmolVLA / Octo / π₀ 공통 학습 가능. C3는 출력 후처리 → 구조 무관 |
| **Problem-general** | gripper 외 모든 boundary event | release / push commit / contact switch 로 자연 확장 |
| **Intervention-general** | 재학습·HW 수정 없이 적용 | C2는 preprocessing, C3는 wrapper. `lerobot-train` CLI 그대로 (HARD RULE #2 정합) |

→ **4축 전부를 하나의 protocol-level 기여로 묶음**.

---

## 5. 설계: EAAS 3-컴포넌트 구조

### 5.1 C1 — Event Detector (dataset-time, heuristic, zero ML)
**입력**: v6 raw (leader angle trajectories + state trajectories)
**출력**: 각 프레임에 event label ∈ {reach, approach, close_start, close_commit, hold, release_start, release, retreat}
**방법**:
- State derivative stall detection (|state'| < ε 윈도우)
- Action velocity sign change (close→hold 경계 식별)
- Gripper value threshold crossing (state[5] 또는 action[5] 이동 평균)
- 완전 결정적 — ML/force sensor 불필요
**검증 조건**: v6 50ep에 돌린 뒤 "episode당 close_commit ≥1 프레임" 이 적어도 N% 이상에 확인 (64% 에피소드가 gripper max<70°여도 commit **의도**는 존재해야 함 — 없으면 가설 실패)
**파일 (미래)**: `data_event_detector.py` (data-agent 소유)

### 5.2 C2 — Event-Aligned Action Re-parameterization (dataset conversion time)
**위치**: `convert_to_lerobot_v3.py` 후단에 훅
**동작**:
- close_commit 라벨 프레임의 action[5] 값을 **episode-local min→100 선형 스케일**로 saturate
- reach/approach/hold 프레임은 **절대 건드리지 않음** → reach 성공 능력 보존
- 저장: 원본 옆에 `lerobot_dataset_v6_eaas/` 생성 (**원본 불변**)
**학습**: `lerobot-train --policy.pretrained_path=lerobot/smolvla_base --dataset.root=lerobot_dataset_v6_eaas --batch_size=8 --steps=20000` 그대로 (HARD RULE #2 정합)
**파일 (미래)**: `convert_to_lerobot_v3_eaas.py` (pipeline-agent 소유, 기존 fork)

### 5.3 C3 — Event-Aware Inference Wrapper (deploy-time, optional)
**위치**: `deploy_smolvla.py:642` (현재 Stage 1 amp 패치 위치) 의 일반화
**동작**:
- Event Detector를 state에서 online 실행 (과거 k=5 프레임 버퍼)
- close_commit 검출 시 action[5] amp 적용; reach 검출 시 amp 비활성
- **data 재수집/재학습 안 한 baseline 체크포인트에도 적용 가능** → ablation 축
**파일 (미래)**: `deploy_smolvla.py` 수정 (deploy-agent 소유)

---

## 6. v6 Stage 1b와의 통합 (경쟁이 아니라 상류 개입)

- **Stage 1b 성공 (3/5 zone 이상)**: 4/8 패치 (amp 2.5x, 단일 heuristic) 가 유효했음 증명 → EAAS C3의 1개 특수 사례 → C1/C2 일반화 정당화
- **Stage 1b 실패 (< 3/5)**: inference-time heuristic만으로는 부족 → 데이터 레벨 개입 (C2) 이 필연적 다음 단계 → EAAS 우선순위 상승
- **어느 분기에서도 EAAS는 baseline-first 규칙을 위반하지 않음** (Stage 1b 완료 후 실행)

---

## 7. HARD RULES 정합성 체크

| Rule | 위반? | 해설 |
|---|---|---|
| #1 HOME start | ✅ | v6 HOME 데이터 그대로 사용 |
| #2 `lerobot-train` + smolvla_base | ✅ | C2는 데이터셋만 생성, 학습은 공식 CLI. custom training script 없음 |
| #3 VGST FAIL ≠ 실제 실패 | ✅ | open-loop 4-chunk 실제 테스트 유지 |
| #4 반증 검증 | ✅ | §3에서 6편 abstract 직접 확인, 한정어 준수 |
| #5 JOINT_LIMITS 제거 금지 | ✅ | 건드리지 않음 |
| #6 카메라 고정 | ✅ | 데이터 그대로 |
| #7 HANDOFF.md 금지 | ✅ | 생성 안 함 |
| #8 MEMORY 오버라이드 금지 | ✅ | MEMORY.md 미수정 |
| #9 SmolVLA 한정 아님 | ✅ | Octo / π₀ cross-VLA 축 포함 (RunPod 사용) |
| #10 문제-중심 | ✅ | v6 gripper 실패가 직접 출발점, "X% 향상" 프레임 아님 |

---

## 8. Novelty Defense (예상 리뷰어 공격 → 방어선)

| 공격 | 방어 |
|---|---|
| "Beyond the Majority 2602.06512 가 long-tail imitation 이미 다룸" | 그들의 long-tail = **task-budget inter-task**. 우리 = **action-value intra-task**. Table로 명시 구분 |
| "ForceVLA / FAVLA / VLA-Touch 가 contact-rich 이미 해결" | 전부 force/tactile 센서 전제. 우리는 **센서 부재 조건**에서 dataset-level protocol — 적용 영역 다름 |
| "그냥 data preprocessing hack" | Flow matching density-biased sampling + L-F mechanical limit 의 **조인트 원인 분석** → protocol은 원인에 대응하는 원리적 해결. Ablation: EAAS 없는 raw / C2 only / C2+C3 / (센서 추가 가능 시 ForceVLA 재현) |
| "GenCHiP 과 동일" | GenCHiP = LLM이 compliant primitive 호출 (code generation). 우리 = continuous VLA policy가 데이터 레이어에서 event-aligned 값 직접 학습. 같은 insight(action space matters), 다른 경로. 명시 인용 |
| "SmolVLA 하나뿐이면 chance" | **Cross-VLA**: SmolVLA + Octo(or π₀) 동일 EAAS 데이터셋 학습 → 공통 개선 증명. RunPod 덕분 |
| "왜 2022 LLM-as-Planner 인용이 필요한가" | GenCHiP 경유만 엄격 인용. SayCan/CaP/ProgPrompt/BrainBody는 related work 의 "이 경로는 다른 층위" 1 단락 |
| "reach 능력 망가지지 않나?" | C2는 reach 프레임 건드리지 않음. Ablation으로 검증 (v6 reach 성공률 vs v6_eaas reach 성공률) |

---

## 9. 실행 스케치 (Stage 1b 완료 *이후* — 지금 실행 금지)

> **경고**: 아래는 설계 윤곽일 뿐. Stage 1b 결과 확인 전 실행 금지. baseline-first 규칙 유지.

### Week 1 (baseline 보강 — 이미 다른 세션 진행 중)
- Stage 0 (leader limit 측정) → Stage 1b (패치 배포) → Stage 2 (3-zone × 5회) → baseline 공식 달성 gate

### Week 2 (EAAS C1/C2 구현)
- `data_event_detector.py` 작성, v6 50ep 돌려 event 분포 검증
- `convert_to_lerobot_v3_eaas.py` 작성, `lerobot_dataset_v6_eaas/` 생성
- Episode quality check (원본과 비교)

### Week 3 (학습)
- SmolVLA 로컬 학습 (`run_official_train.py`, 20K steps)
- RunPod에서 Octo 동일 데이터셋 학습 (cross-VLA 축)
- 4-ckpt VGST 비교 (기존 프로토콜 유지)

### Week 4 (배포 + ablation)
4조건 × 3-zone × 5회:
1. Raw v6 체크포인트 (baseline)
2. v6_eaas SmolVLA 체크포인트 (C2만)
3. v6_eaas SmolVLA + C3 wrapper (C2+C3)
4. Raw v6 체크포인트 + C3 wrapper (C3만, Stage 1b 연장)

Cross-VLA 축: 위 표에 Octo 행 추가.

추가: v1, v3 raw 데이터에도 C2 적용 → 3점 자연 ablation을 6점으로 확장 → "데이터 품질 축" 논문 스토리

### Week 5-6 (보강 태스크 + 논문)
- 2nd task (컵 밀기 또는 박스 잡기) 50ep 수집 → EAAS protocol 재사용 → **task-general 증명**
- 논문 작성 (CoRL/IROS LBR 타겟은 baseline 달성 후 재결정)

---

## 10. Deliverables (이 세션)

1. ✅ `claudedocs/EAAS_PROPOSAL.md` (이 문서)
2. ⏳ `claudedocs/RESEARCH_TRENDS_2026.md` §1.6 업데이트 (다음 write 호출)
3. ❌ 코드 수정 없음 (baseline-first 준수)
4. ❌ MEMORY.md 수정 없음 (결정 아니라 제안 단계)

---

## 11. Key References (arXiv ID / DOI 요약)

**LLM-as-Planner 경로** (GenCHiP만 직접 연결):
- arXiv 2404.06645 — GenCHiP ⭐
- arXiv 2204.01691 / 2209.07753 / 2209.11302 — SayCan / CaP / ProgPrompt (related work 배경)
- Wiley adrr.202500072 — BrainBody-LLM (related work 배경)

**직접 경쟁/병행** (prior art 6편 전부 반증 완료):
- arXiv 2602.06512 — Beyond the Majority (ICRA 2026, task-level long-tail, 층위 구분)
- arXiv 2505.22159 — ForceVLA (NeurIPS 2025, force sensor 전제)
- arXiv 2602.23648 — FAVLA (force sensor 전제)
- arXiv 2507.17294 — VLA-Touch (tactile sensor 전제)
- arXiv 2601.01948 — Primitive Diffusion Policy (global instruction용)
- arXiv 2512.11921 — LoRA Accessible VLA (parameter efficiency 축)

**인프라**:
- arXiv 2506.01844 — SmolVLA
- arXiv 2602.22818 — LeRobot

**Community signal (non-peer-reviewed)**:
- Correll Lab Medium 2025-12-15 "When Fine-Tuning Hurts: Failure Modes on a Low-Cost Robot"

---

## 12. 다음 행동 (이 세션 안에서)
- `claudedocs/RESEARCH_TRENDS_2026.md` §1.6 업데이트 (EAAS 접점 행 + 6편 prior art 각주)
- 이후 세션 종료. 구현은 Stage 1b 완료 후 재승인.
