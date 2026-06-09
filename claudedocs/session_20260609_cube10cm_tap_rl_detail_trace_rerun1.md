# Session: cube10cm tap RL detail-trace rerun1

Date: 2026-06-09 KST

## Scope

- Active branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier / RL unblock diagnosis.
- Track A grasp/dataset/training remains separate and was not touched.
- B200 is disconnected/retired. No SSH/B200 reconnect/pull/copy was attempted.
- User explicitly approved one tiny local GPU repeat after D204.

## Pre-Run Contract

The approved runtime kept the same nearface x240 h580 ep608 step-clipped built-in
DiffIK contract and added only output telemetry:

- `--reach_trace_json`
- `--reach_trace_detail_json`

Unchanged:

- geometry: `fixed_cube_x_m=0.240`, `fixed_cube_y_m=0.000`
- target path: `target_path_mode=near_face_goal`
- controller: `isaac_builtin_diffik_step_clipped_direct_apply`
- horizon: `steps=580`, `closed_loop_push_steps=580`, `episode_length_s=6.08`
- step clip: `builtin_diffik_step_clip_rad=0.010`
- strict contact gate
- action wrapper knobs
- actuator knobs

## Runtime Outputs

Base:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_reachtrace_detail_rerun1`

Files:

- `_sanity.json`
- `_sanity_summary.out`
- `_trace.json`
- `_detail_trace.json`

Runtime summary:

- line 1: `status=FAIL`, local tiny GPU runtime, no dataset/training/robot/SSH/B200/Track A.
- line 2: env `RoArm-CubeTap10cm-Direct-v0`, `cuda:0`, cube size `0.1`, mass `0.72`, `episode_length_s=6.08`, `env_max_episode_length=608`.
- line 3: `num_envs=2`, `steps_executed=580`, `cube_xy=(0.24,0.0)`, push dir `(1,0)`, controller `isaac_builtin_diffik_step_clipped_direct_apply`, `target_path_mode=near_face_goal`, step clip `0.01`.
- line 5: strict contact/tap remains failed: `contact_seen=0.0`, `tap_success=0.0`, reaction context/seen `0.0`; professor weak physical reaction seen `1.0`.
- line 8: actual TCP remains precontact: face gap max `-0.018959235m`, best shortfall `0.008959235m`.
- line 9: professor physical evidence PASS but RL contact-gated positive-control FAIL.
- line 10: command target final face gap `0.005999971m`, target inside final `1.0`, but target FK error still `25.145485846mm`; direct follow final `0.010684967rad`, actual joint step final `0.001290381rad`.

## Posthoc Detail Audit

Added and ran:

`sim_scripts/cube10cm_tap_rl_reach_trace_detail_rerun1_audit.py`

Outputs:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_reach_trace_detail_rerun1_audit.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_reach_trace_detail_rerun1_audit_summary.out`

Audit summary:

- line 1: local posthoc only; no GPU runtime in audit, no dataset/training/robot/SSH/B200/Track A.
- line 2: runtime contract preserved: `status=FAIL`, `steps_executed=580`, `truncated_count=0`, `num_envs=2`, `seed=962`, x240 nearface, ep608, step clip `0.01`.
- line 3: `rl_contact_gated_positive_control=FAIL`, `professor_physical_reaction_evidence=PASS`, `contact_seen=0.0`, `tap_success=0.0`, professor weak reaction seen `1.0`.
- line 4: detail schema rows `1160`, schema length `59`, `contains_action_fields=false`, `action_teacher_dataset=false`.
- line 5: command target inside rows `714` over steps `223..579`; applied/actual inside rows remain `0/0`; applied best face gap `-0.013995274m` and shortfall `0.003995274m`; actual best face gap `-0.018957566m` and shortfall `0.008957566m`.
- line 6: target-base evidence: raw delta max `0.095429182rad`, clipped delta max `0.010000000rad`, previous-target-minus-actual max `0.010870218rad`, current-target-minus-previous max only `0.001419574rad`. This indicates previous-target-base runtime first. Counterfactual FK was not computed.
- line 7: actuator follow/saturation is real secondary: direct follow max `0.010870218rad`, actual step max `0.001367390rad`, actual/follow mean ratio `0.111843619`, computed torque p95/max ratio `1.367817116/2.075699615`, applied torque p95 `1.0`, applied torque saturation fraction `0.085862069`.
- line 8: reset/precontact bias remains lower priority but not cleared: initial command/applied/actual gaps `-0.019955199/-0.020112728/-0.021178227m`, actual-command bias `-0.001223028m`.
- line 9: next branch `DESIGN_PREVIOUS_TARGET_BASE_RUNTIME_FIRST`; contact-gate relaxation is not next; DiffIK action dataset, PPO/RL, large dataset, and RoArm remain blocked.
- line 10: verdict `DETAIL_TRACE_CONFIRMS_TARGET_BASE_FIRST_WITH_ACTUATOR_SATURATION_SECONDARY`.

## Interpretation

The detail trace does not unblock dataset/RL. It sharpens the next diagnostic:

- The command target is correct and inside the contact band.
- The applied joint-target FK still never reaches the band.
- The actual TCP still never reaches the band.
- The actual-base target generation is effectively advancing only at actual-joint-step scale, not at the intended clipped target schedule.
- Actuator/drive saturation is also real, but because applied FK itself remains outside the band, the next first branch is target-base generation, not contact-gate relaxation and not actuator tuning first.

## Blocked

- strict contact-gated positive-control
- clean DiffIK action teacher
- tiny action dataset dry run
- PPO/RL
- large dataset
- RoArm deployment
- Track A mixing
- contact-gate relaxation as first fix

## Next

Only with explicit approval: design a default-preserving previous-target-base candidate. It should be a tiny diagnostic runtime candidate, not dataset/RL, and it must preserve geometry, near-face target semantics, strict contact gate, horizon, and step clip unless the design audit explicitly justifies otherwise.
