# 2026-06-02 - Cube3cm Push Metric Reframe And Target-Extension Probe

## Scope

- Branch: professor cube3cm push/tap only, separate from Track A
  grasp/dataset/training.
- No B200/JHPark SSH, no reconnect, no pull, no `.ssh` copy.
- No Track A runtime was run. No dataset generation, PPO scale-up, VLA training,
  or new 1024/10k candidate audit was run in this state-update pass.
- Existing dirty/untracked worktree state was preserved, not reverted.

## Why This Update Exists

- The previous waypoint actor work advanced the canonical actor to a sharded
  10,240-row teacher-off first-episode robustness gate PASS, but the user
  correctly questioned whether the strict 3cm `success_marker` is the professor's
  actual push/tap objective.
- The answer from code/log inspection is: 3cm is a strict env success marker, not
  the only meaningful push/tap evidence. The next reporting gate should include
  smaller displacement thresholds and object-size-normalized displacement.

## Verified Code Facts

- Cube size is fixed at `0.030m` in `roarm_rl/roarm_cube_push_env.py:31`.
- Cube mass is fixed at `0.020kg` in `roarm_rl/roarm_cube_push_env.py:72`.
- Spawned cuboid dimensions use the same `CUBE_SIZE_M` on all axes in
  `roarm_rl/roarm_cube_push_env.py:60-63`.
- Current friction constants are static `1.5`, dynamic `1.2`, restitution `0.0`
  in `roarm_rl/roarm_cube_push_env.py:74-77`.
- The old strict target displacement and success displacement are separate:
  target displacement `0.040m` at `roarm_rl/roarm_cube_push_env.py:94`;
  success displacement `0.030m` at `roarm_rl/roarm_cube_push_env.py:95`.
- The reset target is start position plus `push_dir * cube_push_target_disp_m`
  in `roarm_rl/roarm_cube_push_env.py:653-655`.
- Displacement is computed as:
  `disp_xy_vec = cube_xy - start_xy` and
  `disp_along = dot(disp_xy_vec, push_dir)` in
  `roarm_rl/roarm_cube_push_env.py:691-696`.
- `controlled` requires forward displacement at least 1mm plus bounded speed,
  tip, and xy displacement in `roarm_rl/roarm_cube_push_env.py:703-708`.
- `impact` is an outlier guard on speed, displacement, or tip in
  `roarm_rl/roarm_cube_push_env.py:709-713`.
- `low_motion` is `disp_xy < cube_low_motion_disp_m` in
  `roarm_rl/roarm_cube_push_env.py:714`.
- `success_now` requires controlled, no impact, `disp_along >= 0.030m`,
  target-distance tolerance, and speed cap in
  `roarm_rl/roarm_cube_push_env.py:781-787`.
- Eval CSV records `cube_x0_m`, `cube_y0_m`, `push_dx/dy`,
  `disp_along_push_m`, `disp_xy_m`, `target_xy_dist_m`, speed, tip,
  controlled, impact, low_motion, success, and grasp markers in
  `roarm_rl/eval_cube_push_policy.py:219-240`.

## Formula For The New Push Table

- Directional displacement:
  `disp_along_push_m = dot(final_cube_xy - start_cube_xy, push_dir_xy)`.
- Threshold pass rate for threshold `t`:
  `mean(disp_along_push_m >= t)`, where `t` is
  `0.001`, `0.005`, `0.010`, `0.020`, or `0.030`.
- Normalized object displacement:
  `disp_over_object_size = disp_along_push_m / cube_size_m`.
- Stable push table should report, at minimum:
  `n`, `controlled`, `impact`, `low_motion`, `success_marker`,
  `disp_mean_m`, `disp/object_size_mean`, and
  `disp_ge_1/5/10/20/30mm`.
- Critical interpretation: 30mm equals exactly one current cube edge length,
  because `CUBE_SIZE_M=0.030`. Failing 30mm does not mean the cube was not pushed;
  it may mean the actor made a stable smaller push.

## Sharded 10k Re-Interpretation

- Single-stage 10240-env eval failed before rollout during IsaacLab stage/ground
  setup with `Stage.GetPrimAtPath(Stage, NoneType)` in
  `model_actor_waypoint_lowx130_teacheroff_eval10240_seed912_firstonly_stderr.out:1-26`.
  Treat this as env-creation failure, not policy failure.
- The 10x1024 sharded driver completed seeds 912-921 in
  `model_actor_waypoint_lowx130_teacheroff_10kshards_seed912_921_driver.out:1-20`.
- Combined 10,240-row mechanism audit PASS:
  controlled `0.927148437`, impact `0.000097656`, low-motion `0.106054687`,
  success `0.524902344`, disp_along mean `0.023250610` in
  `model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_firstonly_audit.out:1-5`.
- Combined bucket audit PASS:
  low_x success `0.406947891`, mid_x success `0.183497537`,
  high_x success `0.213625866` in
  `model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_firstonly_bucket.out:1-10`.
- Failure-mode audit shows posx failures are displacement-limited, not target,
  speed, or impact failures: low_x/mid_x/high_x failed cases all have
  `disp_lt_0p030=1.000000000` in
  `model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_failure_modes.out:5,9,13`.

## Hierarchical Threshold Evidence

- Threshold analysis source has 10,240 rows and explicitly says threshold columns
  are displacement-only, while `success_marker` also includes controlled/no-impact,
  target-distance, and speed conditions:
  `model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_threshold_analysis.out:1-2`.
- Direction `(-1,0)`:
  5mm `0.891406250`, 10mm `0.782421875`, 20mm `0.554296875`,
  30mm `0.447265625` in threshold analysis line 3.
- Direction `(0,-1)`:
  5mm `0.743217425`, 10mm `0.694688575`, 20mm `0.608330149`,
  30mm `0.560565533` in threshold analysis line 4.
- Direction `(0,1)`:
  5mm `1.000000000`, 10mm `0.981775882`, 20mm `0.886002326`,
  30mm `0.815820085` in threshold analysis line 5.
- Direction `(1,0)`:
  5mm `0.906199678`, 10mm `0.842592593`, 20mm `0.770531401`,
  30mm `0.266505636` in threshold analysis line 6.
- Posx low_x:
  5mm `0.724565757`, 10mm `0.624069479`, 20mm `0.554590571`,
  30mm `0.406947891` in threshold analysis line 7.
- Posx mid_x:
  5mm `0.986453202`, 10mm `0.911330049`, 20mm `0.779556650`,
  30mm `0.183497537` in threshold analysis line 8.
- Posx high_x:
  5mm `1.000000000`, 10mm `0.981524249`, 20mm `0.963048499`,
  30mm `0.213625866` in threshold analysis line 9.

## Corrected Interpretation

- Do not say "the cube cannot be pushed forward." More accurate:
  the canonical actor usually produces stable 5-20mm pushes in the forward
  direction, but often does not exceed the strict 30mm/one-cube-length marker.
- Do not say the other three directions are all equally solved at 30mm. At 30mm:
  `+y` is strong, `-y` is moderate, `-x` is moderate/weak, and `+x` is weak.
- For professor push/tap reporting, lead with displacement tiers and
  `disp/object_size`, while still preserving the strict `success_marker` as a
  useful task-specific gate.

## Diagnostic Trace

- Teacher-off diagnostic trace seed911 used the canonical checkpoint with teacher
  sidecar compare-only, not teacher action injection:
  `diagnostic_trace_seed911/actor_trace_analysis.out:1`.
- Contact was reached in all traced envs and effective action matched actor
  output: actor abs `0.282554212`, teacher abs `0.340359867`,
  effective-vs-actor MSE `0.000000000`, contact reached `1.000000000` in
  `diagnostic_trace_seed911/actor_trace_analysis.out:3-4`.
- The trace supports the hypothesis that the actor is slightly smaller than the
  teacher and displacement-limited, not that the action application path is dead:
  group lines show low_x disp `0.014287824`, mid_x `0.023804545`,
  high_x `0.029833396` in
  `diagnostic_trace_seed911/actor_trace_analysis.out:8-10`.

## Code Changes In This Session

- `roarm_rl/distill_cube_push_actor.py` added default-preserving weighted loss
  knobs:
  `--loss_posx_low_weight`, `--loss_posx_mid_weight`,
  `--loss_posx_high_weight`, `--loss_push_phase_weight`, and
  `--loss_post_phase_weight`. Defaults are all `1.0`.
- Weighted loss is per-sample MSE multiplied by bucket/phase weights and
  normalized by total sample weight.
- `roarm_rl/roarm_cube_push_env.py` added default-preserving BC teacher target
  overrides:
  `bc_teacher_midx_push_through_m=-1.0` and
  `bc_teacher_highx_push_through_m=-1.0`. Negative default means disabled.
- `roarm_rl/eval_cube_push_policy.py` added CLI plumbing and summary fields for
  the same mid/high push-through overrides.
- Current md5s after these code changes:
  - `roarm_rl/roarm_cube_push_env.py`
    `560ae4883bbc84d4f7ba388ab6064bc5`
  - `roarm_rl/eval_cube_push_policy.py`
    `48da8c463e3fcd55d15a5090141cb907`
  - `roarm_rl/distill_cube_push_actor.py`
    `9fb4cca936c265180065e4cf8eb6d393`
  - `roarm_rl/trace_cube_push_actor.py`
    `b71359ac44895bd368e1189498e0cf47`
  - `roarm_rl/analyze_cube_push_trace.py`
    `2e3d8d1c862a66b766cc7eaf20b66c0d`

## Weighted Mid/High Actor Candidate

- Candidate directory:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_midhighw200_post150_seed923/`.
- Checkpoint:
  `model_actor_waypoint_midhighw200_post150.pt`, md5
  `636b4a907469453497cd4c2ddacf4ef6`.
- Metrics confirm the intended weighting:
  mid/high loss weights `2.0`, post phase `1.5`, push phase `1.2`,
  sample weight mean `1.350142002105713`, max `3.0` in
  `actor_waypoint_midhighw200_post150_metrics.json:34-47`.
- One-step fit passed numerically: final val MSE `0.00023266756033990532` and
  final weighted val MSE `0.00021922370069660246` in
  `actor_waypoint_midhighw200_post150_metrics.json:22-27`.
- Teacher-off 128 seed911 mechanism audit looked better overall:
  controlled `0.960937500`, impact `0`, low-motion `0.078125000`,
  success `0.679687500` in
  `model_actor_waypoint_midhighw200_post150_teacheroff_eval128_seed911_firstonly_audit.out:1-5`.
- But bucket screen failed:
  low_x success `0.083333333`, mid_x `0.090909091`, high_x `0.833333333`,
  verdict `FAIL_POSX_BUCKET_SCREEN` in
  `model_actor_waypoint_midhighw200_post150_teacheroff_eval128_seed911_firstonly_bucket.out:7-10`.
- Verdict: reject. Do not run 1024/10k from this candidate.

## Target-Extension Probe

- Probe used the canonical checkpoint and gain, with teacher-off eval only:
  `model_actor_waypoint_lowx130.pt`, gain `0.040`, `bc_teacher_blend=0.0`.
- The only changed evaluation target semantics were mid/high BC teacher
  push-through overrides:
  mid `0.030`, high `0.035`.
- Summary records the overrides:
  `bc_teacher_midx_push_through_m=0.03` and
  `bc_teacher_highx_push_through_m=0.035` in
  `model_actor_waypoint_lowx130_targetext_m030_h035_teacheroff_eval128_seed911_firstonly_summary.json:8,11`.
- Audit:
  controlled `0.921875000`, impact `0`, low-motion `0.085937500`,
  success `0.585937500`, disp mean `0.024030716` in
  `model_actor_waypoint_lowx130_targetext_m030_h035_teacheroff_eval128_seed911_firstonly_audit.out:1-5`.
- Bucket:
  low_x success `0.166666667`, mid_x `0.272727273`, high_x `1.000000000`,
  verdict `FAIL_POSX_BUCKET_SCREEN` in
  `model_actor_waypoint_lowx130_targetext_m030_h035_teacheroff_eval128_seed911_firstonly_bucket.out:7-10`.
- Verdict: promising for mid/high on one 128-env seed, but not a gate pass. Do
  not scale it. Next work should first decide the metric/gate and low_x handling.

## Cube Size And Mass Notes

- Current sim object is a 3cm cube with 20g mass. Inferred density:
  `0.020 / 0.030^3 = 740.7 kg/m^3`.
- If `10*10*10` means 10cm cube and density is preserved, mass should be about
  `0.741kg`. Keeping 20g at 10cm would imply about `20 kg/m^3`, an extremely
  light diagnostic object.
- If `10*10*10` means 10mm cube and density is preserved, mass should be about
  `0.000741kg`. Keeping 20g at 10mm would imply about `20000 kg/m^3`, physically
  implausible for this task.
- Therefore the next design must explicitly choose one of:
  measured-object mass, density-preserving scaling, or fixed-mass diagnostic.
  Do not silently change size while keeping mass semantics ambiguous.

## Next Concrete Steps

1. Update the audit/reporting script to emit the hierarchical table:
   `1/5/10/20/30mm`, `disp/object_size`, controlled, no-impact, low-motion, and
   old `success_marker`.
2. Decide object-size/mass convention before any `10*10*10` cube experiment:
   units, mass source, density, and whether the run is physical or diagnostic.
3. Only after the metric table exists, run at most a tiny teacher-off 128 gate for
   a clearly defined candidate. Do not jump to 1024/10k.
4. Do not run dataset generation, PPO scale-up, 10k/100k learned robustness, VLA
   training, or Track A runtime without explicit approval.

