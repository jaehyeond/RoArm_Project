# Session 2026-06-29 - Cube10cm top-view D302 hard-bin actor/teacher diagnostic

## Superseded by D303

D303 re-audited the D302 hard-bin results and found that the `13/322/935`
failures reproduce only when several bins are run sequentially inside one Isaac
process. Fresh one-bin processes for ep13, ep322, and ep935 pass with useful
`1.0` and overshoot `0.0`. Do not use the D302 multi-bin actor/teacher failure
rows as standalone policy-failure evidence. Current truth is
`D303_HARD_BIN_MULTI_PROCESS_REAUDIT_SUPERSEDES_D302_NO_REPAIR_YET`.

## Scope

- Continue D301 without PPO training.
- Probe D301 failed D256 episodes `221,198,13,322,935`.
- Compare actor-only, D257 teacher-only, and actor with small action noise.
- Add future PPO collection-time per-env final trace export.
- No long PPO, PPO ladder, partial actor preservation, real actor update,
  render, cleanup, RunPod/B200/SSH, Track A, VLA fine-tuning, or RoArm
  deployment was performed.

## Code changes

- `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`
  - Added repeatable `--episode_range` for exact D256 episode probes.
  - Added `--reset_warmup_mode` with default `direct_reset`.
  - Added actor-vs-teacher direction metrics and final contact geometry
    diagnostics.
  - Fixed env stepping so Isaac/PyTorch does not reuse inference-mode tensors
    across reset/step calls.
- `roarm_rl/train_cube_push_ppo.py`
  - Added `collection_final_env_trace_iter_<N>.jsonl` export after tap10cm
    collection.
  - Each row records `env_id`, D256 reset episode, contact/useful/overshoot,
    displacement, current contact geometry, action magnitude, joint-delta cap,
    and BC teacher diagnostics.

## Runtime contract

- Non-PPO hard-bin diagnostics only.
- Checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/cube10cm_d300_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/model_0.pt`
- Hard D256 episodes: `221,198,13,322,935`.
- Common settings:
  - `num_envs=8`
  - `eval_steps=580`
  - D256 direct reset
  - `link5_collision_aabb`
  - `tap_stop_after_disp_m=0.003`
  - `tap_success_terminate=False`
  - `tap_useful_terminate=False`
  - `tap_overshoot_terminate=False`
  - `action_scale=0.04`
  - `max_joint_delta_per_step_rad=0.04`
  - `joint_target_lead_limit_rad=0.06`
  - `joint_delta_reference=joint_pos`
  - `bc_teacher_feature_target_mode=env_target`
  - `bc_teacher_phase_timing=direct_steps`
  - `bc_teacher_linear_phase_steps=579`

## Results

| mode | ep | contact | useful | overshoot | max XY m | action abs mean | policy abs mean | teacher abs mean | actor-teacher MSE | cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor deterministic | 221 | 1.0 | 1.0 | 0.0 | 0.00001493 | 0.200433 | 0.201145 | 0.245128 | 0.219819 | 0.324091 |
| actor deterministic | 198 | 1.0 | 1.0 | 0.0 | 0.01791592 | 0.104381 | 0.104381 | 0.193283 | 0.154295 | -0.612015 |
| actor deterministic | 13 | 1.0 | 0.0 | 1.0 | 0.02625378 | 0.520652 | 16.236054 | 0.492401 | 0.799077 | -0.007444 |
| actor deterministic | 322 | 0.5 | 0.5 | 0.0 | 0.00881752 | 0.156274 | 0.156999 | 0.138406 | 0.110625 | 0.215364 |
| actor deterministic | 935 | 0.0 | 0.0 | 1.0 | 0.03418441 | 0.430695 | 0.665190 | 0.529570 | 0.495086 | 0.330994 |
| D257 teacher-only | 221 | 1.0 | 1.0 | 0.0 | 0.00001158 | 0.315585 | 0.602968 | 0.315585 | 0.557610 | 0.201762 |
| D257 teacher-only | 198 | 1.0 | 1.0 | 0.0 | 0.01790871 | 0.192930 | 0.104527 | 0.192930 | 0.154032 | -0.615061 |
| D257 teacher-only | 13 | 1.0 | 0.0 | 1.0 | 0.02133680 | 0.522912 | 1.022682 | 0.522912 | 1.023433 | -0.176681 |
| D257 teacher-only | 322 | 1.0 | 0.0 | 1.0 | 0.03290845 | 0.534185 | 1.145126 | 0.534185 | 1.083987 | -0.277097 |
| D257 teacher-only | 935 | 1.0 | 0.0 | 1.0 | 0.03383858 | 0.497309 | 1.352435 | 0.497309 | 1.083822 | -0.083635 |
| actor noise 0.005 | 221 | 1.0 | 1.0 | 0.0 | 0.00001484 | 0.200630 | 0.200899 | 0.242635 | 0.222859 | 0.311429 |
| actor noise 0.005 | 198 | 1.0 | 1.0 | 0.0 | 0.01791620 | 0.104820 | 0.104391 | 0.193288 | 0.154303 | -0.611965 |
| actor noise 0.005 | 13 | 1.0 | 0.0 | 1.0 | 0.02771312 | 0.547668 | 18.503375 | 0.511217 | 0.881520 | -0.055258 |
| actor noise 0.005 | 322 | 0.375 | 0.375 | 0.0 | 0.01097843 | 0.140514 | 0.140681 | 0.125118 | 0.093406 | 0.125169 |
| actor noise 0.005 | 935 | 0.0 | 0.0 | 1.0 | 0.03408900 | 0.440494 | 0.701349 | 0.526027 | 0.459028 | 0.391007 |

Joint delta cap stayed `0.0` across the hard-bin runs.

## Interpretation

- D301's hard-bin issue is not a single no-contact failure mode.
- Episodes `13` and `935` are overshoot-heavy under actor execution.
- Episode `322` is partial coverage under actor execution and overshoots under
  teacher-only execution.
- D257 teacher-only is not a safe repair signal for the hard bins because it
  overshoots on `13/322/935`.
- Small stochastic action noise is not the main root cause.
- The next repair should be based on D256 recorded state/action alignment or a
  pre-contact action projection/approach constraint, not blind teacher-KL.

## Decision

- No long PPO.
- No PPO ladder.
- No partial actor preservation or real actor update yet.
- Do not relax the AABB contact band to hide failures.
- Future tiny PPO gates must use the new collection-final per-env trace export.
- Next work: non-PPO actor/action repair using hard-bin D256 recorded deltas or
  direction-aware pre-contact projection, followed by teacher-off/bin
  diagnostics before any tiny PPO re-gate.

## Verification

- `python -m py_compile roarm_rl/train_cube_push_ppo.py sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
- `git diff --check`

Both passed before this document was written.

## Verdict

`D302_HARD_BIN_ACTOR_TEACHER_DIAGNOSTIC_NO_PPO_NO_TEACHER_KL`
