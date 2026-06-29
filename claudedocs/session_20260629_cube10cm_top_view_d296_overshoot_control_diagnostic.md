# D296 Cube10cm Overshoot-Control Diagnostic

Date: 2026-06-29 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch only.
No PPO training, long PPO, render, cleanup, RunPod/B200/SSH, Track A,
SmolVLA/VLA fine-tuning, or RoArm deployment was performed.

## Question

D295 showed that the D290/D295 actor can move the cube when the horizon is long
enough, but saved-checkpoint teacher-off eval failed with overshoot `0.1875`.
D296 asks whether non-PPO action constraints can control that overshoot before
any further PPO runtime.

## Code Change

Updated `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py` so frozen
teacher-off eval can test the same non-PPO control options used by the runtime:

- `--tap_stop_after_disp_m`
- `--tap_contact_slowdown_use_proxy`
- `--exec_action_clip_abs`
- per-env CSV output via `--out_env_csv`

This is diagnostic plumbing only. It does not train or update the actor.

## Common Eval Contract

- Checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`
- `num_envs=32`, `eval_steps=580`
- D256 reset active, frame `0`
- BC teacher off: `bc_teacher_blend=0.0`
- Contact proxy: `link5_collision_aabb`
- Action contract:
  - `action_scale=0.04`
  - `max_joint_delta_per_step_rad=0.04`
  - `joint_target_lead_limit_rad=0.06`
  - `joint_delta_reference=joint_pos`
- Gate:
  - useful `>=0.90`
  - overshoot `<=0.05`
  - mean XY displacement `>=0.0005m`
  - max XY displacement `>=0.001m`
  - XY `>=1mm` rate `>=0.25`

## Linspace Matrix

Using `d256_reset_sample_mode=linspace`, three variants passed the conservative
gate:

| Variant | Verdict | Useful | Overshoot | Mean XY | Max XY | XY >=1mm | XY >=3mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw rerun | fail | `0.8125` | `0.1875` | `0.0106040` | `0.0590083` | `0.6875` | `0.6250` |
| exec clip `0.5` | pass | `1.0` | `0.0` | `0.0036601` | `0.0191714` | `0.4375` | `0.40625` |
| stop after `0.001m` | pass | `1.0` | `0.0` | `0.0008960` | `0.0032587` | `0.6875` | `0.03125` |
| stop after `0.003m` | pass | `1.0` | `0.0` | `0.0021452` | `0.0038778` | `0.6875` | `0.6250` |
| useful-stop / zero-action | fail | `1.0` | `0.0` | `0.0001232` | `0.0032587` | `0.03125` | `0.03125` |
| proxy slowdown `0.35` | fail | `1.0` | `0.0` | `0.0001236` | `0.0032462` | `0.03125` | `0.03125` |

Interpretation:

- Magnitude or stop constraints can make the linspace sample look safe.
- Useful-stop and proxy slowdown are too early/too strict for displacement:
  they remove overshoot but also remove meaningful movement.
- Linspace pass alone is not enough.

## Random Reset Checks

The linspace candidates were retested with `d256_reset_sample_mode=random` and
seeds `29603` / `29604`.

| Variant | Seed | Useful | Overshoot | Mean XY | Max XY | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stop after `0.003m` | `29603` | `0.75` | `0.25` | `0.0079748` | `0.0319677` | fail |
| stop after `0.003m` | `29604` | `0.75` | `0.25` | `0.0074070` | `0.0346703` | fail |
| exec clip `0.5` | `29603` | `0.75` | `0.25` | `0.0090027` | `0.0329562` | fail |
| exec clip `0.5` | `29604` | `0.625` | `0.3125` | `0.0112702` | `0.0988506` | fail |
| stop after `0.001m` | `29603` | `0.75` | `0.25` | `0.0072996` | `0.0319677` | fail |
| stop after `0.001m` | `29604` | `0.75` | `0.25` | `0.0069719` | `0.0346703` | fail |
| exec clip `0.25` | `29603` | `0.5` | `0.25` | `0.0073335` | `0.0344693` | fail |
| exec clip `0.25` | `29604` | `0.25` | `0.28125` | `0.0078595` | `0.0329829` | fail |
| clip `0.5` + stop `0.001m` | `29603` | `0.75` | `0.25` | `0.0074296` | `0.0329562` | fail |
| clip `0.5` + stop `0.001m` | `29604` | `0.6875` | `0.25` | `0.0068375` | `0.0329587` | fail |

Interpretation:

- None of the tested action constraints pass random D256 reset sampling.
- Lower action magnitude alone is not a solution: `exec_clip_abs=0.25` still
  overshoots and reduces useful rate.
- Displacement-stop alone is not robust: the random failing states still reach
  overshoot before the stop produces a safe rollout.
- Therefore do not run another PPO gate from these constraints.

## Per-Env Failure Audit

For `stop_disp003_random_seed29604_envtrace_d296`, 8 of 32 envs overshot.

Overshoot D256 episode indices:

`339, 154, 198, 668, 736, 656, 195, 606`

All eight are original D256 `train_clean_positive` / `clean_useful_tap` episodes
with camera contract pass:

| Episode | D256 max XY | D256 max along |
| ---: | ---: | ---: |
| `154` | `0.006750` | `0.003098` |
| `195` | `0.004481` | `0.004456` |
| `198` | `0.004924` | `0.004922` |
| `339` | `0.008026` | `0.007984` |
| `606` | `0.004481` | `0.004479` |
| `656` | `0.004960` | `0.004954` |
| `668` | `0.006375` | `0.006271` |
| `736` | `0.004109` | `0.004000` |

In the failed actor rollout, most overshoot rows have large XY displacement but
zero along-displacement:

| Episode | Actor max XY | Actor max along |
| ---: | ---: | ---: |
| `339` | `0.0221776` | `0.0` |
| `154` | `0.0346703` | `0.0140285` |
| `198` | `0.0227460` | `0.0` |
| `668` | `0.0211926` | `0.0` |
| `736` | `0.0216408` | `0.0` |
| `656` | `0.0329440` | `0.0` |
| `195` | `0.0265905` | `0.0` |
| `606` | `0.0288489` | `0.0` |

Interpretation:

- The original D256 labels are not the immediate problem; those episodes were
  clean in recorded data.
- The actor closed-loop rollout is creating off-axis/lateral displacement on
  random reset states.
- This explains why magnitude-only constraints do not solve the issue.

## Decision

Verdict:

`D296_ACTION_CONSTRAINT_LINSPACE_PASS_RANDOM_FAIL_NO_PPO`

Next concrete order:

1. Do not run long PPO.
2. Do not run another tiny PPO gate from D296 constraints.
3. Add/collect episode-index-preserving actor diagnostic data so actor-vs-recorded
   action error can be inspected per episode/state, not just in aggregate.
4. Repair the actor/action contract with direction-aware constraints or training:
   penalize/project lateral action relative to `push_dx/push_dy`, or train with
   recovery/state-aggregation labels that preserve episode metadata.
5. Make random D256 reset sampling a required teacher-off gate. Linspace alone
   is too weak.
6. Only after random teacher-off useful/overshoot/displacement gates pass should
   tiny PPO + TensorBoard be reconsidered.

## Artifacts

- Matrix command:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/overshoot_control_d296/run_overshoot_control_matrix_d296.sh`
- Candidate random command:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/overshoot_control_d296/run_candidate_random_checks_d296.sh`
- Conservative random command:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/overshoot_control_d296/run_conservative_random_checks_d296.sh`
- Per-env failure CSV:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/overshoot_control_d296/stop_disp003_random_seed29604_envtrace_d296/teacher_off_policy_eval_envs_stop_disp003_random_seed29604_envtrace_d296.csv`
- Updated eval script:
  `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
