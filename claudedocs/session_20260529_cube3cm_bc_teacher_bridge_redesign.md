# Session 2026-05-29 - Cube3cm BC Teacher Bridge Redesign Diagnostics

## Scope

- Professor cube3cm push/tap branch only.
- Local machine only: no B200, no SSH, no pull, no `.ssh` copy.
- Goal: diagnose the failed safety-aware PPO warm-start bridge and test small
  diagnostics before any teacher-off 1024 audit.
- Not Track A grasp, not Track A runtime, not full dataset generation, not 10k/100k
  learned-policy robustness.

## Boot Verification

- `git status --short --untracked-files=all --branch` initially printed only
  `## master...origin/master`.
- `START_HERE.md:234-255`, `DECISIONS.md:5520-5567`, and
  `claudedocs/session_20260528_cube3cm_safety_rl_warmstart.md:82-145` agree on
  the latest truth: `model_11.pt` is not a successful learned PPO policy.
- Original PPO warm-start logs were rechecked:
  - Training config/log path: `ppo_bc_teacher_warmstart_smoke12_seed885_stdout.out:47-78`.
  - Runtime device: `ppo_bc_teacher_warmstart_smoke12_seed885_stdout.out:80-86`.
  - Iteration 0 BC teacher metrics:
    `ppo_bc_teacher_warmstart_smoke12_seed885_stdout.out:121-155`.
  - Final timesteps/checkpoint:
    `ppo_bc_teacher_warmstart_smoke12_seed885_stdout.out:573-614`.
  - Teacher-off 1024 fail:
    `ppo_bc_teacher_warmstart_smoke12_seed886_eval1024_audit.out:1-5` and
    `ppo_bc_teacher_warmstart_smoke12_seed886_eval1024_bucket.out:1-10`.
  - Teacher-on direct-like partial recovery:
    `ppo_bc_teacher_warmstart_teacheron_directlike_seed888_eval256_audit.out:1-5`
    and `ppo_bc_teacher_warmstart_teacheron_directlike_seed888_eval256_bucket.out:1-10`.
- Citation caveat: the prior session doc says
  `ppo_bc_teacher_warmstart_smoke12_seed885_stderr.out:1-6` shows missing
  `gymnasium`, but the current local file lines 1-6 are only `requests`
  dependency warnings. `rg "gymnasium|ModuleNotFoundError|No module named"` in
  the runtime log directory found no match. Do not cite the `gymnasium` failure
  unless the missing original log is recovered.

## Code Changes

- `roarm_rl/roarm_cube_push_env.py`
  - md5 `a0483108ef0fc8ab2f27a58b6edd8c13`.
  - Added default-preserving `joint_delta_reference`.
  - Added default-preserving `bc_teacher_phase_timing`.
  - `episode_scaled` preserves the old teacher phase behavior.
  - `direct_steps` matches the direct BC rollout's step-index timing.
  - `joint_pos` reference makes env action targets closer to the direct BC rollout
    loop, which used current joint position plus BC delta.
- `roarm_rl/train_cube_push_ppo.py`
  - md5 `7032616ded5617b546149227f4c0d110`.
  - Exposed/logged the two new options.
- `roarm_rl/eval_cube_push_policy.py`
  - md5 `b10fad43cfd3b0ca543390ad6011135f`.
  - Exposed/logged/summarized the two new options.

Static checks:

- `python -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_push_ppo.py roarm_rl/eval_cube_push_policy.py sim_scripts/cube3cm_push_ppo_rollout_audit.py sim_scripts/cube3cm_push_diffik_bucket_audit.py` PASS.
- `git diff --check` PASS.

## Diagnostics

### Teacher-On Bridge, Direct-Step/Joint-Pos, Lowx 0.85

- Run: `ppo_bc_teacher_warmstart_bridge_directstep_jointpos_seed889_eval128`.
- Summary lines 1-58 confirm `episode_length_s=6.0`,
  `action_smoothing_alpha=1.0`, `contact_joint_delta_scale=1.0`,
  `fast_cube_joint_delta_scale=1.0`, `joint_delta_reference=joint_pos`,
  `bc_teacher_phase_timing=direct_steps`, and `bc_teacher_blend=1.0`.
- Audit lines 1-5: mechanism PASS, controlled `0.984375000`, impact
  `0.007812500`, low-motion `0.039062500`, success `0.601562500`.
- Bucket lines 1-10: FAIL because low_x success was only `0.133333333` despite
  zero low_x impact.

### Teacher-On Bridge, Direct-Step/Joint-Pos, Lowx 1.00

- Run: `ppo_bc_teacher_warmstart_bridge_directstep_jointpos_lowx100_seed890_eval128`.
- Summary lines 1-58 confirm the same bridge but `bc_teacher_lowx_policy_delta_scale=1.0`.
- Audit lines 1-5 PASS mechanism with controlled `0.992187500`, impact
  `0.007812500`, low-motion `0.007812500`, success `0.765625000`.
- Bucket lines 1-10 PASS small posx screen:
  - `(1,0)` controlled `1.000000000`, impact `0`, success `0.605263158`.
  - low_x controlled `1.000000000`, impact `0`, low-motion `0`, success `0.538461538`.

### Existing `model_11.pt`, Teacher-Off New Loop

- Run: `ppo_bc_teacher_warmstart_newloop_teacheroff_model11_seed891_eval128`.
- Summary lines 1-58 confirm teacher-off, new action loop, 128 envs.
- Audit lines 1-5: mechanism PASS but performance dead: controlled `0`,
  impact `0`, low-motion `1`, success `0`.
- Bucket lines 1-10: every direction and posx bucket is low-motion `1` and
  success `0`.

### New PPO Distillation Smoke8

- Run: `ppo_bc_teacher_newloop_distill_smoke8_seed892`.
- Training stdout lines 3-30 confirm 128 envs, 8 iterations, 6s horizon,
  direct-step/joint-pos action loop, `bc_teacher_blend=1.0`,
  `bc_teacher_imitation_reward_scale=5.0`, lowx scale `1.0`, and `cuda:0`.
- Checkpoint: `ppo_bc_teacher_newloop_logs/cube_push_bc_teacher_newloop_distill_smoke8_seed892/model_7.pt`.
- Checkpoint md5: `5ed5ac34dc624ac8c660d9176378b357`.
- Imitation MSE did not improve enough:
  - Iteration 0 line 102: `0.6781`.
  - Iteration 1 line 144: `0.5564`.
  - Iteration 6 line 349: `0.5850`.
- Teacher-off eval of `model_7.pt`:
  `ppo_bc_teacher_newloop_distill_smoke8_seed892_model7_teacheroff_eval128_seed893`.
- Audit lines 1-5: controlled `0`, impact `0`, low-motion `1`, success `0`.
- Bucket lines 1-10: all directions/buckets remain low-motion `1`, success `0`.

## Interpretation

- The old failure was not just "BC checkpoint is bad". The teacher-on bridge
  can recover when the environment action loop matches the direct BC rollout more
  closely: no IK endpoint reset, 6s horizon, direct step timing, joint-pos target
  reference, no smoothing/slowdown, and lowx scale `1.0`.
- The PPO actor still has not learned the teacher action. PPO reward-side
  imitation, even with `bc_teacher_blend=1.0` and scale `5.0`, stayed around
  MSE `0.56-0.59` and produced a teacher-off zero-motion policy.
- Therefore the next blocker is actor initialization/objective, not teacher
  trajectory geometry alone.

## Decision

- Do not run teacher-off 1024 from `model_11.pt` or from the new smoke8
  `model_7.pt`; both fail the smaller 128 teacher-off gate.
- Do not call any PPO checkpoint here a successful learned cube-push policy.
- Do not run PPO scale-up, 10k/100k learned robustness, dataset generation, or
  Track A runtime from these PPO checkpoints.
- Next valid work is true supervised actor/normalized-action distillation into
  the rsl_rl actor, or another direct actor initialization method, followed by a
  small teacher-off 128 audit before any 1024 audit.

