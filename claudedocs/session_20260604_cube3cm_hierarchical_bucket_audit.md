# 2026-06-04 - Cube3cm Hierarchical Bucket Audit

## Scope

- Branch: professor cube3cm push/tap only, separate from Track A
  grasp/dataset/training.
- No B200/JHPark SSH, no reconnect, no pull, no `.ssh` copy.
- No GPU/IsaacLab runtime was run. This was a local-only audit/report update over
  existing CSV/summary artifacts.
- Existing dirty/untracked worktree state was preserved, not reverted.

## Boot Evidence

- `CLAUDE.md` Current-State Protocol was read first.
- At boot, `START_HERE.md` current direction said the professor branch must stay
  separate from Track A and that metric cleanup was the next step before any
  bigger training.
- `DECISIONS.md` D115-D117 were rechecked. D116 requires hierarchical
  `1/5/10/20/30mm`, `disp/object_size`, controlled, no-impact, low-motion
  reporting. D117 blocks scaling weighted mid/high and target-extension probes.
- `MEMORY.md` recent sessions were read as complementary context only.

## Verified Inputs

- Current code fixes cube size at `CUBE_SIZE_M=0.030` and mass `0.020kg` in
  `roarm_rl/roarm_cube_push_env.py:31,60-77`.
- Strict `success_marker` requires controlled push, no impact,
  `disp_along >= 0.030m`, target tolerance, and speed cap in
  `roarm_rl/roarm_cube_push_env.py:781-787`.
- Existing sharded 10k mechanism audit:
  `model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_firstonly_audit.out:1-5`
  reports controlled `0.927148437`, impact `0.000097656`, low-motion
  `0.106054687`, success `0.524902344`, disp mean `0.023250610`.
- Existing threshold analysis:
  `model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_threshold_analysis.out:1-9`
  reports displacement-only thresholds; direction `(1,0)` is 5mm
  `0.906199678`, 10mm `0.842592593`, 20mm `0.770531401`, 30mm
  `0.266505636`.

## Audit Update

- Updated `sim_scripts/cube3cm_push_diffik_bucket_audit.py` to add:
  - `--cube_size_m` default `0.030`
  - `--cube_mass_kg` default `0.020`
  - `density_kg_m3`
  - `no_impact`
  - `disp_over_object_size_mean`
  - displacement-only `disp_ge_1mm/5mm/10mm/20mm/30mm`
- Existing bucket PASS/FAIL logic was not changed.
- `python -m py_compile sim_scripts/cube3cm_push_diffik_bucket_audit.py` passed.

## New Report Output

- Generated:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_waypoint_lowx130_seed905/model_actor_waypoint_lowx130_teacheroff_eval10240_sharded_seed912_921_hierarchical_bucket.out`
- Line 1 logs `cube_size_m=0.030000`, `cube_mass_kg=0.020000`,
  `density_kg_m3=740.741`, and says threshold columns are displacement-only.
- Line 2 overall: controlled `0.927148437`, impact `0.000097656`,
  no-impact `0.999902344`, low-motion `0.106054687`, success `0.524902344`,
  `disp_over_object_size_mean=0.775020338`, 5/10/20/30mm rates
  `0.884472656` / `0.824804688` / `0.704101562` / `0.525195312`.
- Line 6 direction `(1,0)`: controlled `0.958534622`, no-impact `1.000000000`,
  low-motion `0.087359098`, success `0.266505636`,
  `disp_over_object_size_mean=0.743007598`, 5/10/20/30mm rates
  `0.906199678` / `0.842592593` / `0.770531401` / `0.266505636`.
- Lines 7-9 show posx buckets: low_x/mid_x/high_x have
  `disp_over_object_size_mean` `0.608311269` / `0.767408781` / `0.845491985`.
  Mid/high are strong at 10/20mm but weak at 30mm; low_x remains weaker and
  high low-motion.
- Line 10 remains `PASS_POSX_BUCKET_SCREEN learned_policy=YES
  track_a_grasp_success=NO`.

## Interpretation

- The canonical `model_actor_waypoint_lowx130.pt` + gain `0.040` result is still
  a sharded 10k teacher-off learned-policy gate PASS for the professor cube3cm
  branch.
- The new table sharpens, but does not improve, the evidence: the actor often
  makes stable sub-one-cube-length pushes. It does not solve the strict 30mm
  marker uniformly.
- This is not dataset readiness, not Track A evidence, and not PPO/RL/VLA final
  success.

## Next Concrete Step

1. Before any cube size change, define units and mass coupling: measured mass,
   density-preserving mass, or fixed-mass diagnostic.
2. Only after that, and only with explicit approval, run at most a tiny 128
   teacher-off gate for a clearly defined candidate.
3. Do not run 1024/10k candidate scale-up, dataset generation, PPO scale-up, VLA
   training, or Track A runtime without explicit approval.
