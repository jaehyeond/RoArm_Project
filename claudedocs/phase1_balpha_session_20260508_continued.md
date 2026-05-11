# Phase 1.B-α Session — 2026-05-08 (continued from 새벽 entry)

## 세션 개요
이전 새벽 entry에서 코드 작성 + 4 결정 confirm 완료. 본 세션은 B200 transfer + 7-step 진행 시작.

## 진행 단계 (1→6 completed/in-progress)

### 1단계 — sanity test PASS
- num_envs=4, reward_phase=4, steps=400
- obs (4,28) ✓, target_in_obs[0/1] = (0.28, -0.0435, 0.011383) diff=0 ✓
- termination=0 ✓ (HARD RULE Phase 1.A)
- truncation=4 ✓ (max_ep=400)
- throughput 724 steps/s

### 2단계 — warmstart_22_to_28 ckpt 생성
- **BUG #3 발견 + fix**: normalizer `_var`/`_std` shape `(1, 22)`로 2-D였음. 기존 expand_2d는 무조건 default=0 → `var=0` 시 normalizer NaN. expand_2d에 default 인자 추가 + heuristic (`var/std`→1.0, `mean/weight`→0.0).
- ckpt path: `$ROARM_B200_ROOT/logs/roarm_rl/warmstart_phase1B_alpha.pt`
- Source: P3 model_1497 (precision_compare 98.773%)
- 8 expand 2-D: actor.0/critic.0 + actor/critic obs_normalizer mean/var/std

### 3단계 — multi-env scaling test (4096) PASS
- Throughput **487,224 steps/s** (Phase 1.A 471K **초과**)
- target_in_obs[0/1] diff=1e-6 (FP32 noise)
- AssertionError on `truncated_count > 0` = test의 결함 (steps=200 < max_ep=400, 정상)

### 4단계 — P4 baseline (500 iter, 03:04 wall) PASS
- experiment: `roarm_stack_p4_500iter_seed0_warmstart`
- Final: tcp_sponge=0.0151m, grasped=0.93, lift_success=0.77, sponge_height=0.57m, sponge_target=0.629m
- **Warm-start 검증 SUCCESS**: 500 iter 동안 tcp_sponge ≈ 0.015m 일관 → P3 22-dim policy 보존
- **BUG #4 발견 + fix**: rsl_rl 3.1.2 `OnPolicyRunner.load()` 강제 `optimizer_state_dict` 접근 (line 319). warmstart에서 dropped → KeyError. train_ppo.py에서 manual `policy.load_state_dict(strict=False)` 우회.
- **BUG #5 발견 + fix**: rsl_rl ActorCritic.load_state_dict()이 bool 반환 (PyTorch 표준 IncompatibleKeys 아님). `missing, unexpected = ...` unpack 실패 → return value isinstance check.
- 11 ckpts (model_0/50/100/.../499)

### 5단계 v1 — P5 nav baseline (500 iter) → reward imbalance 진단
- experiment: `roarm_stack_p5_500iter_seed0_resumeP4`
- Final: sponge_target_dist 0.629→0.611 (**-19mm only**)
- 진단: lift_reward=5.0×0.58m=+2.9 vs nav_reward=1.0×0.6m=-0.6 → policy가 sponge 잡고 들고 있는 데 incentive concentrate. Lift saturate 안 함 (sponge_height 0.58m 호버링).

### 5단계 v2 — Reward fix + 1500 iter
- **Fix #1**: lift = clamp(h, max=lift_success_height=0.10) — saturation
- **Fix #2**: nav_reward_scale 1.0 → 5.0
- experiment: `roarm_stack_p5v2_1500iter_seed0_resumeP4_rewardfix`
- Iter 487 mid-check: sponge_target 0.61 → **0.12 (80%↓)**, sponge_height 0.58 → 0.097 (saturation 작동)
- Final iter 1499 (09:57 wall): sponge_target **0.1047m**, sponge_height 0.0973m
- iter 487→1499 (1012 iter 추가): -15mm only — plateau (잔여 horizontal ~60mm + z-axis ~86mm)
- z-axis 86mm = sponge 잡은 채 들고 있어서. nav reward만으로는 gripper open 동기 부재 → P6 진입.

### 6단계 (in-progress) — P6 place 학습 (2000 iter)
- experiment: `roarm_stack_p6_2000iter_seed0_resumeP5v2`
- PID 1645384 launch at 09:44, ETA 13min
- **Iter 668 mid-check**: sponge_target 0.103m (plateau 지속), **place_success_rate 0** ⚠️
- 진단: P6 reward chicken-and-egg
  - place_cond = (d<25mm) AND gripper_open AND stable_50step
  - 현재 d=103mm → place_cond 절대 fire 안 함
  - gripper open 동기는 place_bonus에서만 옴 → 학습 진입 불가

## Code 수정 (이번 세션)

### roarm_rl/warmstart_22_to_28.py
- `expand_2d(w, mode, default=0.0)` — default 파라미터 추가
- 호출 시 heuristic: `var`/`_std` → 1.0, `mean`/weight → 0.0

### roarm_rl/train_ppo.py
- `--task pick|stack` flag 추가 (default=pick)
- `--reward_phase` choices 1-3 → 1-6 확장
- env cfg/id 분기 (RoArmPickEnvCfg / RoArmStackEnvCfg)
- experiment_name default `roarm_pick_p{phase}` → `roarm_{task}_p{phase}`
- `--resume` 처리: `runner.load()` 우회, `policy.load_state_dict(sd, strict=False)` 직접 호출. return value bool/tuple 둘 다 처리.

### roarm_rl/roarm_stack_env.py
- `nav_reward_scale: 1.0 → 5.0` (line 213)
- lift = clamp(h, min=0.0, **max=lift_success_height**) (line 369-376) — saturation

## 사용자 결정 대기 (P6 plateau 옵션)

| 옵션 | 변경 | 시간 | 리스크 |
|---|---|---|---|
| **A** | place_cond에서 gripper_open 제거 OR gripper_open assistance reward 추가 (sponge near target일 때 open 비례) | ~15분 | 새 reward shape, 학습 dynamics |
| **B** | place_dist_thresh 25mm → 100mm 완화 (단계적 squeeze 학습) | ~15분 | 25mm 강한 보장 약화 |
| **C** | P6 2000 iter 끝까지 — 정체 지속 예상 | ~9분 | 시간 낭비 가능 |

추천: **A**.

## P6 학습 모니터링 — 다음 세션
- PID 1645384 (학습 끝나면 자동 PID 사라짐)
- log: `$ROARM_B200_ROOT/logs/phase1Balpha/train_p6.out`
- ckpt dir: `$ROARM_B200_ROOT/logs/roarm_rl/roarm_stack_p6_2000iter_seed0_resumeP5v2/`
- 다음 세션 진입 시: ssh로 PID + 최종 결과 확인 + 옵션 A/B/C 결정 진행

## 산출물 목록
- `claudedocs/phase1_balpha_session_20260508_continued.md` (본 doc)
- B200 logs: `$ROARM_B200_ROOT/logs/phase1Balpha/train_p4_v3.{out,err}`, `train_p5.{out,err}`, `train_p5v2.{out,err}`, `train_p6.{out,err}`
- B200 ckpts: `roarm_stack_p4_500iter_seed0_warmstart/model_*.pt`, `roarm_stack_p5_500iter_seed0_resumeP4/model_*.pt` (v1, plateau), `roarm_stack_p5v2_1500iter_seed0_resumeP4_rewardfix/model_*.pt` (v2, nav 학습 SUCCESS), `roarm_stack_p6_2000iter_seed0_resumeP5v2/model_*.pt` (in-progress)

## HARD RULES 준수
- #11 /half-clone 0회 (Stop hook 85% 거부, continuation prompt 사용)
- #14 모든 ssh 명령에 `set -e` + `ROARM_B200_ROOT` + `whoami` 검증
- #15 cu128 sm_100 alive (학습 진행 중)
- #17 visual RL 시도 X (state-only 28-dim only)
- #18 사용자 명시 4 결정 (target/gravity/22→28/P4-P5-P6 점진) 모두 보존. **P5 reward weight 변경은 implementation detail (user 결정 외 영역)이라 진행** — 단순 nav_reward_scale 1→5 + lift saturation, 4 결정과 무관.
- #19/#20 sponge edge-stand 47mm + tower geometry 그대로
- #26 5/19 deadline 11일 ahead
