# D312 Cube10cm Stop-Loss Strict Useful Baseline V1

Date: 2026-07-06 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch after D311. This session did not run Isaac runtime, PPO, tiny PPO trace gates, TensorBoard training, torchrun, learned-policy updates, RoArm deployment, Track A, VLA/SmolVLA fine-tuning, B200/SSH, pull, or `.ssh` copy.

## Starting Point

D311 found that the D310/D311 non-PPO env primitive could report contact/reaction/useful/final proxy true while moving the cube only `0.7008mm` in seed `30704`, env `19`, D256 episode `700`. With opt-in `--primitive_speed_stop_min_disp_m 0.001`, the two D311 fresh32 seeds combined to `64/64` contact/reaction/useful/final proxy, `0/64` overshoot, `64/64` XY `>=1mm`, and max XY mean/max/min `0.003797/0.005945/0.003002m`.

The strategic issue is that another seed-only speed-min validation campaign would not change the decision. The `0.001m` setting was already dominant on the observed edge case and had a fallback-controlled config path. The correct next step is to freeze a baseline and run a failure-capable perturbation benchmark.

## Code Changes

- `roarm_rl/roarm_cube_push_env.py`
  - Promoted `tap_push_primitive_speed_stop_min_disp_m` default to `0.001`.
  - Added `tap_useful_min_disp_m=0.001`.
  - Made useful/success logic require contact, reaction, no overshoot, and XY displacement `>= tap_useful_min_disp_m`.
  - Applied the same floor to `tap_target_band_now`, target-band reward, stop-after-useful, useful termination, and env log scalars.

- `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`
  - Promoted `--primitive_speed_stop_min_disp_m` default to `0.001`.
  - Made step/final `tap_useful_seen` summaries strict with `tap_useful_min_disp_m`.
  - Added perturbation CLI overrides: `--cube_size_m`, `--cube_mass_kg`, `--cube_static_friction`, `--cube_dynamic_friction`.
  - Records cube size/mass/friction and override values in summary JSON.

- `roarm_rl/train_cube_push_ppo.py`
  - Collection-final `cube_tap_useful_seen_rate` now requires the 1mm useful floor.
  - Logs `CollectionFinal/cube_tap_useful_min_disp_m`.

- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
  - Raised default `--min_collection_final_tap_disp_xy_ge_1mm_rate` to `0.90`.

- Additional active diagnostics updated to avoid old useful inflation:
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
  - `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`
  - `sim_scripts/cube10cm_top_view_actor_teacher_trace.py`
  - `sim_scripts/cube10cm_top_view_distill_actor_from_teacher.py`
  - `sim_scripts/cube10cm_top_view_distill_actor_from_d256_replay.py`

- `CLAUDE.md`
  - Added the session progress rule: every research session must run a failure-capable experiment or explicitly justify why not; control hardening is reactive only; decision-invariant validation must not be run.

## Baseline Controller V1

Baseline controller v1 is:

- `rl_action_mode="tap_push_primitive"`
- `tap_push_primitive_speed_stop_min_disp_m=0.001`
- `tap_useful_min_disp_m=0.001`
- strict useful = contact + reaction + no overshoot + max XY displacement `>=1mm`

This is a baseline controller and benchmark contract, not a learned policy, not PPO promotion, not VLA/SmolVLA fine-tuning, and not RoArm/POSCO readiness.

## Why No Runtime Experiment In This Session

This session implemented the stop-loss guardrail requested after the D311 strategic review. The new rule was added during this session, so the session is explicitly documented as a code/docs contract migration rather than a research experiment. This is the first allowed exception under the new session progress rule because it migrated a known observed failure into the metric/control contract and installed the rule itself. Future sessions cannot cite D312 as permission for no-runtime hardening. The next research session must run a failure-capable perturbation evaluation or explicitly justify why that is impossible.

## Next Required Experiment

Run the D312 perturbation benchmark in `claudedocs/cube10cm_top_view_d312_perturbation_protocol.md`.

Do not run another nominal seed campaign. Do not harden the primitive again before observing a perturbation or training failure. Do not run PPO or a tiny PPO trace gate before the perturbation benchmark defines what the baseline cannot handle.

## Verification

- `python -m py_compile` passed for all touched Python files.
- `git diff --check` passed.
- `pgrep -af '[i]saaclab|[t]rain_cube_push_ppo|[t]ensorboard|[t]orchrun|[r]l_games'` returned no residual Isaac/PPO/TensorBoard/torchrun/rl_games processes.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` reported no compute apps.
- `nvidia-smi` recheck showed GPU utilization `0%` with only display `Xorg` memory entries.

## Verdict

`D312_STOPLOSS_STRICT_USEFUL_BASELINE_V1_NO_RUNTIME_NO_PPO`
