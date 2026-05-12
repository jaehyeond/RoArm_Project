# Phase 1.B-α P6 v5 — cliff fix (was_grasped latch + actor bias reset) — 5/11 session

## TL;DR
- ✅ **Critical re-diagnosis** via 3 agents (A2 sim2real / general-purpose 16 papers / C1 experiment):
  - cliff effect = **surface symptom**, NOT root cause
  - real root causes (3): **(R1) actor.6.bias[5]=+0.8446 PPO entropy collapse on gripper dim** (verified via B200 ckpt read), **(R2) `_grasped` latch coupled to gripper_open** → release 순간 nav+lower 게이트도 동시 OFF → release = negative advantage (C1 정밀 진단), **(R3) obs 28-dim 누락 표준 컴포넌트** (tcp_pos_w, _grasped flag, sponge_to_target_quat — A2/general-purpose 일치).
- ✅ MEMORY HARD RULE #8 archive: 5/07 late-night #1/#2/#3 → `MEMORY_archive_20260509.md` (3 entries 본문 그대로 이동, 한 줄 pointer 유지). full-body 8→5 limit 정확히 맞춤.
- ✅ B.1 patch (`roarm_rl/roarm_stack_env.py`, 5 edits): `_was_grasped` 영구 latch state tracker 추가, `_compute_intermediate_values`에서 cond로 update, `_reset_idx`에서 episode boundary에만 reset, `nav_reward` (L416) + `lower_reward` (L435) 게이트를 `_grasped` → `_was_grasped`. **`_grasped`은 physics attach 제어용으로 그대로 유지** (release 가능성 보존).
- ✅ B.2 patch (`roarm_rl/train_ppo.py`, 3 edits): `--reset_actor_bias_idx <int>` argparse + resume 후 state_dict에서 `actor.<last>.bias[idx]=0.0` 직접 수정 + post-load verify.
- ✅ B.4 logging (env L441-467): `gripper_open_rate`, `sponge_grounded_rate`, `was_grasped_rate`, `place_cond_fire_rate` 4개 신규 키 추가. 진단 정확도 ↑.
- ✅ md5 verify (local↔B200 일치): `7aea1e423c1d0c465e58de4118f67ed7 roarm_stack_env.py` / `05c9f3f2398b46022961d029857a8281 train_ppo.py`.
- ✅ Sanity 16env × 200step PASS (env 생성 + obs shape (16,28) + 28-dim target_pos correct + 200 step random rollout 정상). truncated assertion FAIL = max_ep_len 400 step 미달 (false-fail, P6v4 동일).
- ✅ **B200 P6v5 학습 launched** PID **2045744** at 20:36:57 KST. resume P6v4 model_999 + reset_std 1.30 + entropy 0.001 + **reset_actor_bias_idx 5**. ETA ~7min wall.

## Critical 비판적 발견 (코드 정독 중 발견)

### `_grasped`은 두 역할로 overloaded
- `_apply_action` (L329): `if self._grasped.any(): self._update_grasp_attach()` → **physics kinematic attach** (gripper_open으로 풀어야 함)
- `_get_rewards` (L406/425): `nav_reward * _grasped` + `lower_reward * sponge_near * _grasped` → **reward gating** (release 후에도 유지되어야 함)

P6v4의 cliff는 위 두 역할이 같은 변수를 사용한 결과. `_grasped`을 was_grasped (영구 True)로 바꾸면 attach 풀리지 않아 release 불가 = CRITICAL BUG. → 별도 `_was_grasped` flag로 분리.

### Actor bias 정량 (ssh로 ckpt 직접 읽음)
P6v4 model_999 ckpt state_dict:
- `actor.6.bias`: shape (6,) values `[-0.1085, +0.2790, +0.2740, -0.2503, -0.0183, +0.8446]` |max|=**0.8446 outlier**
- `std`: shape (6,) values `[1.3597, 1.2970, 1.3018, 1.2849, 1.3094, 1.3055]` (균일 1.30, 안정 ✓)
- P(close per step) = Φ(0.8446/1.30) = **74.2%** → P(5 step open 연속) = 0.26⁵ = **0.12%** → PPO 학습 budget 부족

bias reset 0 → P(close)/step 50% → P(5 step open) = 3.1% = **25배 ↑**.

## Patch summary

### B.1 — `roarm_rl/roarm_stack_env.py` (5 edits)

| # | 위치 | 변경 |
|---|---|---|
| 1 | L283 init | `self._was_grasped = torch.zeros(num_envs, bool)` 추가 |
| 2 | L538 `_compute_intermediate_values` | `self._was_grasped = self._was_grasped | cond` (cond = `_grasp_condition()`) |
| 3 | L416 nav_reward | `nav_reward * self._grasped` → `nav_reward * self._was_grasped` |
| 4 | L435 lower_reward | `lower_reward * sponge_near * self._grasped` → `lower_reward * sponge_near * self._was_grasped` |
| 5 | L512 `_reset_idx` | `self._was_grasped[env_ids] = False` 추가 (episode 경계에서만 reset) |

추가: L441-467 logging dict 4개 키 신규.

### B.2 — `roarm_rl/train_ppo.py` (3 edits)

| # | 위치 | 변경 |
|---|---|---|
| 1 | argparse | `--reset_actor_bias_idx <int>` flag 신규 |
| 2 | resume 후 std reset block 다음 | state_dict에서 `actor.<last>.bias[idx]=0.0` 직접 수정 (dynamic: `actor_bias_keys[-1]`) |
| 3 | verify block | post-load `target.actor.modules()` 마지막 nn.Linear bias 출력 |

## md5 / paths
- Local: `/home/cgxr/Documents/Robotics/RoArm_Project/roarm_rl/{roarm_stack_env.py,train_ppo.py}`
- md5: `7aea1e42... env.py`, `05c9f3f2... train_ppo.py`
- B200: `$ROARM_B200_ROOT/code/roarm_rl/` (transfer_to_b200.sh 사용, md5 일치)
- Resume ckpt: `$ROARM_B200_ROOT/logs/roarm_rl/p6v4_release_path_reshape_resumeP6v3/model_999.pt`
- Train logs: `$ROARM_B200_ROOT/logs/phase1Balpha/train_p6v5.{out,err}`
- Output ckpts: `$ROARM_B200_ROOT/logs/roarm_rl/p6v5_was_grasped_latch_bias_reset_resumeP6v4/`
- Launch script: `$ROARM_B200_ROOT/launch_p6v5.sh` (chmod +x, /tmp noexec 우회)

## 학습 명령 (실제 launch)
```bash
ssh JHPark
nohup $ROARM_B200_ROOT/launch_p6v5.sh \
  > $ROARM_B200_ROOT/logs/phase1Balpha/train_p6v5.out \
  2> $ROARM_B200_ROOT/logs/phase1Balpha/train_p6v5.err < /dev/null &
# PID 2045744 — 20:36:57 KST 5/11
```

내부 명령:
```bash
python -u -m roarm_rl.train_ppo \
    --task stack --num_envs 4096 --max_iterations 1000 --reward_phase 6 --seed 0 \
    --resume $ROARM_B200_ROOT/logs/roarm_rl/p6v4_release_path_reshape_resumeP6v3/model_999.pt \
    --reset_std 1.30 --entropy_coef 0.001 --reset_actor_bias_idx 5 \
    --experiment_name p6v5_was_grasped_latch_bias_reset_resumeP6v4
```

## 비판적 예측 (P6v5 결과)

| 메트릭 | P6v4 final | P6v5 가설 success | P6v5 가설 fail | 의미 |
|---|---|---|---|---|
| `gripper_open_rate` | ~0% | **>0.10** | <0.05 | bias reset 효과 — fail 시 A2 #2 log_std clip 필요 |
| `was_grasped_rate` | n/a | ≈ grasped (0.93) | < grasped | latch regression — fail 시 cond 정의 재검토 |
| `sponge_grounded_rate` | ~0% | **>0.05** | <0.01 | lower_reward 효과 — fail 시 reward magnitude 부족 |
| `place_success_rate` | 0.000 | **>0.05** | =0 | 종합 진단 — fail 시 Option A reward re-weight OR Option C obs 확장 |
| `sponge_target_dist` | 144mm | <60mm | ~144mm | nav 작동 확인 |
| `place_cond_fire_rate` | n/a (P6) | **>0.10** | <0.01 | place_cond strict 여부 정량화 |

## 다음 세션 entry — 즉시 명령

```
1) ssh JHPark "ps -p 2045744 -o pid,etime 2>&1 | head -2"
   (없으면 종료 완료, etime 7-8min 예상)

2) ssh JHPark "tail -100 /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/logs/phase1Balpha/train_p6v5.out"
   - resume bias reset verify: "[train] reset_actor_bias: actor.6.bias[5]: +0.8446 -> 0.0"
   - reset_std verify: "[train] reset_std: ckpt std [1.36...] -> [1.30...]"
   - 첫 iter (iter 0) 메트릭 6개 신규 key 출력 정상 확인
   - 최종 iter (iter ~1000) 메트릭 확인

3) ssh JHPark "ls -la \$ROARM_B200_ROOT/logs/roarm_rl/p6v5_was_grasped_latch_bias_reset_resumeP6v4/"
   - 21개 ckpt 확인 (model_0~999)

분기 (success criteria):
- gripper_open_rate >0.10 AND place_success_rate >0.05 AND sponge_height <0.05m
  → P7: A2 #2 log_std_min/max clipping 영구 적용 + place_dist_thresh 100→50→25mm curriculum squeeze
- gripper_open_rate >0.10 AND place_success_rate <0.05
  → Option A reward re-weight (lift 5→0.5, grasp 2→0.2, place 5→50) — 사용자 명시 P4-P5-P6 P6 v5 implementation detail, HARD RULE #18 OK
- gripper_open_rate <0.05 (bias 즉시 재saturate)
  → A2 #2 log_std clipping + per-dim entropy 즉시 적용 (영구 std 제어)
- sponge_grounded_rate <0.01 AND gripper_open_rate >0.10
  → lower_reward_scale 5→15 또는 30 (magnitude 부족)
- Fail 종합: Option C obs 28→32 (`sponge_to_target_quat` 4dim 추가, HARD RULE #18 사용자 confirm 필수)
```

## HARD RULES 준수
- #8 archive 5/07 late-night #1/#2/#3 → MEMORY_archive_20260509.md (8→5 limit 정확히)
- #11 /half-clone 거부 1회 (Stop hook 94% 거부, continuation prompt + claudedocs로 처리)
- #14 fail-fast guard 모든 ssh (`set -e; source env.sh; [[ -z $ROARM_B200_ROOT ]] && exit 1; [[ $(whoami) != sogang_jhki ]] && exit 1`)
- #15 cu128 sm_100 alive (P6v4 verified, P6v5 학습 진행 = 추가 검증)
- #17 visual RL X (state-only 28-dim only)
- #18 사용자 명시 4 결정 (target Y=-0.0435 / gravity / 22→28-dim / P4-P5-P6) 보존. obs 28-dim 유지 ✓. reward design (was_grasped latch fix + bias reset)은 implementation detail OK.
- #19 sponge edge-stand 47mm / #20 # tower geometry / #26 5/19 deadline 8일 ahead

## 3 Agent 결과 — Critical 종합

### A2 (Sim-to-Real)
- Isaac Lab 표준 reward = `1-tanh(d/std)` (cliff 구조적 방지). 우리 `-d + binary gate`는 reinvent.
- State obs 누락: `tcp_pos_world`, `_grasped` flag, `sponge_to_target_quat`.
- URDF: effort_limit=2.5Nm 실제 ST3215 (1.5-2.0Nm) 과대평가 의심 — sim2real gap.
- 추천: α + β 동시.

### general-purpose (16 papers cited)
- **Misspecification framing** (Lin/Zhu 2502.20396, RL-100 2510.14830): lift/grasp 누적 +658 ≫ place_bonus +5 → hover globally optimal. cliff fix해도 release incentive 없음.
- **PPO entropy collapse** (HAEPO 2508.18884): bias[5]=+0.84 canonical "structural bias on one dim". std reset 부족, log_std_min/max + per-dim entropy 또는 separate gripper head 필요.
- **State under-specified** (DrS 2404.16779, FMB 2401.08553): relative quat 없으면 release-ready 정책 표현 불가.
- 추천 ranked: α + reward re-weight (lift 5→0.5, place 5→50, time -0.05/step).

### C1 (Experiment Design)
- **`_grasped` latch bug 정밀 진단**: gripper_open=True 순간 `_grasped=False` → `lower_reward` 즉시 OFF.
- α/β/γ falsifiability: γ 이미 P6v3에서 falsified (재실험 낭비).
- δ는 actor.6.bias>0.5이면 magnitude 증가 무효.
- 빠진 critical 진단 2개: ① actor.6.bias[5] 값 확인 + 0 reset (✅ 본 세션 실행) ② gripper joint exploration budget 수치 계산 (✅ 본 doc 정량).
- P6v5 권장 한 줄: bias[5]=0 reset + P6v3 구조 1000 iter resume.

3 agent 모두 일관: **cliff = surface symptom**, root causes = bias saturation + latch coupling + obs gap. P6v5는 R1+R2 동시 fix. R3는 Option C (HARD RULE #18 confirm 필요)로 후속.
