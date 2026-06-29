# D291 Cube10cm Top-View Fresh-Bin Actor Diagnostic

Date: 2026-06-28 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch only.
No Track A, SmolVLA/VLA fine-tuning, RoArm deployment, RunPod/B200/SSH, render,
cleanup, PPO, or TensorBoard run was performed.

## Question

D290 said the replay-batch actor still failed D256 reset-bin closed-loop
coverage. Before training another actor or starting PPO, D291 checked whether
that failure was a real actor failure or a diagnostic artifact from reusing one
Isaac Lab env across multiple bins.

## Code Change

- Updated `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`.
- Added `--reset_pose_source manual|env_hook`.
- Added `--d256_reset_sample_mode` and `--d256_reset_frame_index`.
- The env-hook path uses the runtime D256 reset hook, forces a zero warmup step
  like the bin probe, reads the actual runtime selected episodes, and loads the
  matching D256 rows for recovery labels.

## Actor Checkpoint

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`

## Fresh Env-Hook Results

Each bin was run as a separate Isaac Lab process with:

- reset source: `env_hook`
- D256 sample mode: `linspace`
- envs: `32`
- steps: `580`
- hold steps: `3`
- action scale / max joint delta: `0.04 / 0.04`
- joint delta reference: `joint_pos`
- contact proxy: `link5_collision_aabb`
- stop after useful: enabled
- useful/overshoot termination: disabled

| Bin | Episodes | Useful | Overshoot | Max XY m | Actor Action Mean | Recovery Clip Mean | Actor-Recovery MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `1..208` | `1.0` | `0.0` | `0.000329371279804036` | `0.09431963817955091` | `0.6563254378687847` | `0.5956258373439376` |
| 1 | `209..370` | `1.0` | `0.0` | `0.0020340927876532078` | `0.103545839467953` | `0.6378017369914671` | `0.5957295808314892` |
| 2 | `371..537` | `1.0` | `0.0` | `0.011980446986854076` | `0.1254259844214238` | `0.5096551830013251` | `0.44061636433029416` |
| 3 | `538..715` | `1.0` | `0.0` | `0.0009295984636992216` | `0.11954327848065516` | `0.5175646602722077` | `0.4541968695871564` |
| 4 | `716..999` | `0.90625` | `0.0` | `0.008024415001273155` | `0.14026873634550077` | `0.5802586293169136` | `0.634303804270618` |

## Artifacts

Root:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/closed_loop_recovery_d291/tap10cm/`

Key summaries:

- `closed_loop_recovery_summary_d291_closed_loop_recovery_bin0_envhook_ep001_208.json`
- `closed_loop_recovery_summary_d291_closed_loop_recovery_bin1_envhook_ep209_370.json`
- `closed_loop_recovery_summary_d291_closed_loop_recovery_bin2_envhook_ep371_537.json`
- `closed_loop_recovery_summary_d291_closed_loop_recovery_bin3_envhook_ep538_715.json`
- `closed_loop_recovery_summary_d291_closed_loop_recovery_bin4_envhook_ep716_999.json`

Datasets were also written for possible later DAgger-style aggregation, but this
session does not recommend immediate aggregation because the recovery labels are
still heavily clipped.

## Interpretation

The D290 same-process 5-bin reset-bin failure should not be used alone as the
current PPO blocker. Fresh env-hook diagnostics over the same episode ranges
show the D290 actor can produce useful contact without overshoot under
actor-only execution.

This does not prove a learned policy is ready. It only says the previous reset
bin failure was likely polluted by diagnostic/environment reuse behavior or an
equivalent probe mismatch. PPO collection can still fail once exploration,
teacher reward, actor preservation, and TensorBoard gates are active.

## Decision

Verdict:

`D291_FRESH_ENVHOOK_ACTOR_BIN_DIAGNOSTIC_PASS_D290_REUSED_BIN_FAILURE_SUSPECT_NO_PPO_YET`

Next concrete order:

1. Replace or fix the reused-env reset-bin diagnostic with fresh-per-bin gating.
2. Do not train another aggregated actor from D291 recovery datasets unless a
   fresh-bin or teacher-off gate fails again.
3. If runtime is explicitly approved, run only a tiny PPO + TensorBoard gate
   with actor preservation, then immediately run saved-checkpoint teacher-off
   eval and actor trace.
4. Do not run long PPO and do not claim learned-policy success or RoArm
   readiness before teacher-off eval passes after PPO.

## Verification

- `python -m py_compile sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`
- `git diff --check`
- GPU returned to idle after diagnostics (`nvidia-smi` utilization `0%`).
