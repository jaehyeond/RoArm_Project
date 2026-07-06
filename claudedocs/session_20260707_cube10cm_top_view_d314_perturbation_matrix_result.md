# D314 Cube10cm Perturbation Matrix Result

Date: 2026-07-07 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch after D313. This session ran the required 9-row non-PPO perturbation matrix. It did not run B200/SSH, pull, `.ssh` copy, RoArm deployment, Track A, VLA/SmolVLA fine-tuning, or long PPO.

## Protocol Check

- D313 required the next research session to execute the 9-row matrix.
- Rows: nominal, size `0.09/0.11m`, mass `0.50/1.00kg`, friction `0.8/0.6` and `2.2/1.8`, observation noise `0.005/0.015m`.
- Baseline controller v1: `exec_source=env_tap_push_primitive`, `legacy_far_face_through`, `previous_joint_target`, `start_pose`, goal `0.003m`, push steps `220`, speed stop `0.200m/s`, speed-stop min displacement `0.001m`.
- Metric/referee terms stayed on ground truth. Observation noise was injected only into the controller cube reference.

## Runtime Note

The first nominal attempt inside the sandbox failed because Isaac/PhysX could not see CUDA. That was an execution-environment failure, not a task result. The nominal row and remaining rows were rerun on the local host GPU only. No B200/SSH path was used.

## Aggregate Table

Aggregate artifacts:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/d314_perturbation_matrix_aggregate.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/d314_perturbation_matrix_aggregate.csv`

| Row | Strict useful | Useful | Contact | Reaction | Overshoot | Final proxy | XY >=1mm | XY >=3mm | XY >=7mm | XY >=20mm | Mean XY mm | Min/Max XY mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | 32/32 | 32/32 | 32/32 | 32/32 | 0/32 | 31/32 | 32/32 | 32/32 | 0/32 | 0/32 | 3.711 | 3.008 / 5.756 |
| size 0.09m | 32/32 | 32/32 | 32/32 | 32/32 | 0/32 | 32/32 | 32/32 | 32/32 | 0/32 | 0/32 | 4.014 | 3.010 / 6.261 |
| size 0.11m | 31/32 | 31/32 | 32/32 | 32/32 | 1/32 | 31/32 | 32/32 | 31/32 | 1/32 | 1/32 | 4.119 | 2.581 / 26.199 |
| mass 0.50kg | 32/32 | 32/32 | 32/32 | 32/32 | 0/32 | 32/32 | 32/32 | 32/32 | 0/32 | 0/32 | 3.996 | 3.061 / 6.443 |
| mass 1.00kg | 31/32 | 31/32 | 32/32 | 32/32 | 0/32 | 31/32 | 31/32 | 30/32 | 1/32 | 0/32 | 3.766 | 0.483 / 7.346 |
| friction 0.8/0.6 | 10/32 | 10/32 | 11/32 | 11/32 | 1/32 | 10/32 | 32/32 | 32/32 | 30/32 | 1/32 | 9.177 | 6.978 / 53.462 |
| friction 2.2/1.8 | 0/32 | 0/32 | 8/32 | 8/32 | 32/32 | 7/32 | 32/32 | 32/32 | 32/32 | 32/32 | 3768.838 | 25.654 / 11989.522 |
| obs noise 0.005m | 32/32 | 32/32 | 32/32 | 32/32 | 0/32 | 32/32 | 32/32 | 32/32 | 0/32 | 0/32 | 3.701 | 3.004 / 5.944 |
| obs noise 0.015m | 32/32 | 32/32 | 32/32 | 32/32 | 0/32 | 32/32 | 32/32 | 32/32 | 0/32 | 0/32 | 3.823 | 3.027 / 5.944 |

Observation-noise samples:

- `0.005m` row sampled abs mean/max `2.396/4.898mm`.
- `0.015m` row sampled abs mean/max `7.189/14.693mm`.

## Interpretation

- The baseline is not robust across the 9-row matrix.
- Observation noise did not break the controller in this run. The earlier concern was technically valid, but this seed shows the primitive is more tolerant to fixed XY pose bias than expected.
- The actual failure axis is friction.
- Low friction is the better first learning target: it keeps overshoot low (`1/32`) but collapses contact/reaction/useful to `10-11/32`.
- High friction is a severe dynamics/control failure: `32/32` overshoot and meter-scale runaway. It should be preserved as a hard evaluation target, not used as the first learning target.
- Since not all 9 rows passed, the one-round combined/severe escalation clause does not apply.

## Decision

Freeze this matrix result. Do not hand-add another controller condition in response to friction failure. Per D313, the next step is primitive-parameter learning on a failing axis. Use a policy action mode that actually affects the primitive target; do not train `tap_push_primitive` directly because that mode is a baseline controller and ignores policy action for target generation.

## Verdict

`D314_PERTURBATION_MATRIX_RESULT_FRICTION_BREAKS_BASELINE_NO_PROMOTION`
