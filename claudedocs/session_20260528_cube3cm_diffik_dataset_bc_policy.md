# Session 2026-05-28 - Cube3cm DiffIK Dataset v2 and BC Policy

## Scope

- Professor cube3cm push/tap branch only. This is separate from Track A grasp,
  Track A dataset/training, and PPO/VLA learning.
- No B200, no SSH, no pull, no `.ssh` copy. All runtime was local IsaacLab/GPU
  with escalated sandbox permission where required.
- Goal: v3.1 teacher-quality sweep follow-through -> state-action dataset ->
  dataset audit -> small BC training -> learned BC rollout audit.

## Code Changes

- `sim_scripts/cube3cm_push_diffik_probe.py`
  - Added `--trace_all_envs` for auditable all-env step traces.
  - Current md5: `2342f31701e91af57d0f311db4eeec87`.
- Added `sim_scripts/cube3cm_push_diffik_build_dataset.py`
  - Builds a teacher-filtered, balanced state-action dataset artifact from a
    raw DiffIK trace.
  - Current md5: `a0d18ef1b34415c96d036ba42952e37e`.
- Added `sim_scripts/cube3cm_push_diffik_dataset_v2_audit.py`
  - Audits schema, finite values, split leakage, direction balance, final teacher
    labels, source mechanism, and size gates.
  - Current md5: `f677ea14809a0a2091bc13c5254d4fae`.
- Added `sim_scripts/cube3cm_push_diffik_train_bc.py`
  - Trains and saves a supervised MLP joint-delta BC checkpoint.
  - Current md5: `df03abb00188cfd9b644b0ef410a0e14`.
- Added `sim_scripts/cube3cm_push_bc_policy_rollout.py`
  - Loads the BC checkpoint and evaluates it in IsaacLab physics without DiffIK.
  - Important fix: artifacts are written before optional simulator close; use
    `--skip_sim_close` for this local IsaacLab path because explicit
    `sim_app.close()` hung after rollout completion.
  - Current md5: `a56e9a96feaad196fef6e1081c0116ec`.
- Added `sim_scripts/cube3cm_push_bc_rollout_audit.py`
  - Audits learned BC rollout mechanism and performance.
  - Current md5: `121721561bcde8df141f56207dcab14d`.

Static checks:

- `python -m py_compile` passed for the new/updated scripts.
- `git diff --check` passed.

## Teacher Trace Source

Generated all-env step trace from conservative v3.1 conditions:

- Trace stdout lines 20-21:
  `diffik_probe_v31_datasetv2_1024_seed779_stdout.out:20-21`
  confirm local IsaacLab, `num_envs=1024`, total 1024, built-in
  `DifferentialIKController`, no RoArm-local IK loop, no training, no grasp,
  no attach/object posewrite, `trajectory_variant=v3_1`, base steps
  `220/90/40`, v31 posx/lowx steps `300/220/60`.
- Trace CSV row count:
  `wc -l` showed 148,481 lines for the trace CSV and 1,025 lines for the
  per-env CSV.
- Per-env audit:
  `diffik_probe_v31_datasetv2_1024_seed779_audit.out:1-6`
  passed mechanism with controlled `0.964843750`, impact `0.004882812`,
  low-motion `0.034179688`, success marker `0.611328125`, final TCP error
  `0.023685800`, and `dataset_ready=NO` for the raw trace.
- Posthoc:
  `diffik_probe_v31_datasetv2_1024_seed779_posthoc.out:1-17`
  confirms `(1,0)` remains weakest: line 6 has controlled `0.911111111`,
  impact `0.014814815`, low-motion `0.085185185`, success `0.340740741`.

## Dataset v2

Built dataset artifact:

- Directory:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_state_action_dataset_v2_1024_seed779/`
- Dataset CSV:
  `diffik_state_action_dataset_v2_1024_seed779.csv`
  md5 `fbe64a66f83801c5142497b299c11d1e`.
- Build log:
  `diffik_state_action_dataset_v2_1024_seed779_build.out:1-6`
  reports source trace rows `148480`, source env count `1024`, frames/env
  `145`, eligible env count `612`, selected env count `320`, selected rows
  `46400`, balanced 80 trajectories per direction, train/val/test env split
  `224/48/48`, and `full_dataset_candidate=YES`.
- Header:
  dataset CSV line 1 includes local cube/TCP/target observations, relative
  features, arm joints, joint-delta targets, final labels, split, and trajectory
  metadata.
- Dataset audit:
  `diffik_state_action_dataset_v2_1024_seed779_audit.out:1-7`
  reports rows `46400`, env count `320`, frames/env `145`, schema/finite OK,
  split leakage OK, direction coverage OK, final rates controlled `1.0`,
  impact `0.0`, low-motion `0.0`, success `1.0`, mechanism OK, size OK, and
  verdict `PASS_FULL_STATE_ACTION_DATASET_V2 full_dataset_ready=YES`.

Interpretation:

- This can be called a teacher-filtered state-action dataset v2 for cube3cm
  push/tap BC.
- It is not an image dataset and not a Track A grasp dataset.
- It intentionally filters out failed teacher trajectories, so it is not a
  full distributional replay of all v3.1 trials.

## BC Training

Trained checkpoint:

- Model:
  `diffik_state_action_dataset_v2_1024_seed779/bc_mlp_joint_delta_v1.pt`
  md5 `01beefefcddd7d291a40eed22df3ca80`.
- Training log:
  `diffik_state_action_dataset_v2_1024_seed779_bc_train.out:1-4`
  reports rows `46400`, train/val/test rows `32480/6960/6960`, 240 epochs,
  train MSE `0.906895384 -> 0.003481144`, val baseline/final
  `0.918892920 -> 0.008247175`, test baseline/final
  `0.973477006 -> 0.007494668`, mean test MAE `0.000745819rad`, max abs error
  `0.023879237rad`, verdict `PASS_BC_TRAINED_CHECKPOINT`.
- Metrics JSON lines 35-60 also confirm `full_dataset_ready=true`,
  `learned_policy=true`, `rollout_validated=false` at checkpoint stage, and the
  same test MAE/verdict.

Interpretation:

- This is a supervised BC learned checkpoint.
- Before rollout, it was not enough to claim physics policy success.

## Learned BC Rollout

First 256-env learned-policy rollout:

- Initial unpatched 256/16 attempts timed out because artifacts were written
  after `sim_app.close()`. A 4-env 20-step progress run showed rollout reached
  `line8 rollout_done` immediately, then hung in explicit close. The fixed path
  writes artifacts first and uses `--skip_sim_close`.
- 16-env fixed rollout audit:
  `bc_mlp_joint_delta_v1_rollout16_seed882_fixed_audit.out:1-6`
  passed with controlled `1.000000000`, impact `0.000000000`, low-motion
  `0.000000000`, success `0.750000000`.
- 256-env fixed rollout audit:
  `bc_mlp_joint_delta_v1_rollout256_seed882_fixed_audit.out:1-6`
  passed with `controller=BC_MLP_joint_delta_policy`,
  `learned_policy=True`, `diffik_controller_used=False`, mechanism OK,
  controlled `0.945312500`, impact `0.019531250`, low-motion `0.031250000`,
  success `0.656250000`, and verdict `PASS_LEARNED_BC_POLICY_ROLLOUT`.
  Direction `(1,0)` remains weakest: line 5 has n=67, controlled
  `0.850746269`, impact `0.044776120`, low-motion `0.059701493`, success
  `0.417910448`.

Stronger 1024-env unseen-seed rollout:

- CSV:
  `bc_mlp_joint_delta_v1_rollout1024_seed883.csv`
  md5 `8b1fa8d33dc0f043c60dca97c11c5ab1`.
- Summary JSON lines 9-19:
  controller `BC_MLP_joint_delta_policy`, `dataset_generation=false`,
  `diffik_controller_used=false`, controlled `0.9453125`, executed steps `580`.
- Audit:
  `bc_mlp_joint_delta_v1_rollout1024_seed883_audit.out:1-6`
  passed with row count 1024, mechanism OK, posewrite calls 0, grasp OK,
  controlled `0.945312500`, impact `0.012695312`, low-motion `0.026367188`,
  success `0.648437500`, final TCP error `0.023849732`, verdict
  `PASS_LEARNED_BC_POLICY_ROLLOUT`.
- Direction caveat:
  audit line 5 shows `(1,0)` remains weakest: n=258, controlled
  `0.879844961`, impact `0.038759690`, low-motion `0.038759690`, success
  `0.453488372`.

Interpretation:

- It is now fair to say: learned BC joint-delta policy checkpoint + IsaacLab
  rollout audit PASS at 1024 envs for the professor cube3cm push/tap branch.
- It is not Track A grasp success, not PPO/RL, not VLA, not a 10k/100k learned
  policy robustness claim, and not image-based dataset readiness.
- `(1,0)` is still the weak direction and must be called out.

## Next Gate

1. Preserve this as professor-branch BC v1: teacher-filtered state-action
   dataset + supervised BC policy + 1024 rollout PASS.
2. If improving quality, target `(1,0)` with v3.2 teacher/BC data rather than
   broad scaling.
3. If moving to RL, use this BC checkpoint as an initialization/baseline, then
   run only small frozen audits before any scale-up.
4. Do not mix these professor cube3cm push/tap artifacts with Track A grasp
   dataset/training gates.
