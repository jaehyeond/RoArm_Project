# Session 2026-05-28 - Cube3cm Safety-Aware BC Gate

## Scope

- Professor cube3cm push/tap branch only.
- Not Track A grasp, not Track A dataset/training, not PPO/RL/VLA success.
- No B200, no SSH, no pull, no `.ssh` copy.
- Goal: follow D107 by directly reducing learned-policy `(1,0)` low_x/high_x
  impact before any large learned-policy scaling.

## Starting Evidence

- D107 required teacher/action-distribution redesign or safety-aware BC/RL
  warm-start, followed by another 1024 frozen per-bucket audit.
- Prior BC v2 bucket rollout seed883:
  `bc_mlp_joint_delta_v2_bucket_rollout1024_seed883_audit.out:1-6`
  - controlled `0.961914062`, impact `0.015625000`, low-motion `0.016601562`,
    success `0.679687500`.
  - `(1,0)` success `0.527131783`, impact `0.042635659`.
- Prior BC v2 bucket audit:
  `bc_mlp_joint_delta_v2_bucket_rollout1024_seed883_bucket.out:1-10`
  - low_x impact `0.068493151`, low-motion `0.082191781`, success `0.410958904`.
  - high_x impact `0.061855670`, success `0.711340206`.
  - verdict `FAIL_POSX_BUCKET_SCREEN`.

## Code Changes

- Extended `sim_scripts/cube3cm_push_diffik_train_bc.py`
  - md5 `cbe7c2e8d44fe7a92cb2ba69f29b518a`
  - Added optional `--loss_mode safety_l2`.
  - Added per-bucket sample weights and action L2/excess regularization knobs.
  - Checkpoint stores `safety_config`.
- Extended `sim_scripts/cube3cm_push_bc_policy_rollout.py`
  - md5 `aa0b5ef06db903058724a71f61225f0b`
  - Added auditable `--policy_delta_scale`, `--posx_policy_delta_scale`,
    `--lowx_policy_delta_scale`, `--highx_policy_delta_scale`, and
    `--policy_delta_smoothing_alpha`.
  - Added `posx_x_bucket` to rollout CSV rows and safety config to summary JSON.
- Fixed `sim_scripts/cube3cm_push_diffik_bucket_audit.py`
  - md5 `c69ff72a4a31228868169016ab2f2d08`
  - Bucket verdict now reports `learned_policy=YES` when summary says the rollout
    is a learned policy.
- Unchanged rollout audit:
  - `sim_scripts/cube3cm_push_bc_rollout_audit.py` md5
    `121721561bcde8df141f56207dcab14d`.

## Static Checks

- `python -m py_compile sim_scripts/cube3cm_push_diffik_train_bc.py sim_scripts/cube3cm_push_bc_policy_rollout.py sim_scripts/cube3cm_push_diffik_bucket_audit.py sim_scripts/cube3cm_push_bc_rollout_audit.py` PASS.
- `git diff --check` PASS.
- Post-run process check found no active IsaacLab/conda cube-push jobs.

## Safety BC Training

Artifact:

- `diffik_state_action_dataset_v3_bucket_1024_seed779/bc_mlp_joint_delta_v3_safety_l2.pt`
- checkpoint md5 `03b159809ddca64aad6d6449b7f44876`
- metrics:
  `diffik_state_action_dataset_v3_bucket_1024_seed779/bc_mlp_joint_delta_v3_safety_l2_metrics.json`

Training log:

- `diffik_state_action_dataset_v3_bucket_1024_seed779_bc_v3_safety_l2_train.out:1-5`
  - rows `26100`, train/val/test `17980/4060/4060`, epochs `240`.
  - final train MSE `0.004168557`, val MSE `0.024972165`, test MSE
    `0.018879525`.
  - test mean MAE `0.001225158rad`, max abs error `0.019854907rad`.
  - verdict `PASS_BC_TRAINED_CHECKPOINT`.
  - safety config: `loss_mode=safety_l2`, posx sample weight `1.15`, lowx `1.20`,
    highx `1.10`, posx action L2 `60`, lowx/highx action L2 `100`, action excess
    limit `0.020rad`.

## Frozen Rollout Seed883

Runtime:

- `bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed883_progress.out:1-10`
  confirms `device=cuda:0`, `num_envs=1024`, total steps `580`.
- `bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed883_stdout.out:1-8`
  confirms AppLauncher and environment device `cuda:0`.

Overall audit:

- `bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed883_audit.out:1-6`
  - learned policy `True`, DiffIK controller `False`, training `False`,
    posewrite calls `0`.
  - controlled `0.953125000`, impact `0.004882812`, low-motion `0.030273438`,
    success `0.662109375`.
  - `(1,0)` success `0.507751938`, impact `0.011627907`.
  - verdict `PASS_LEARNED_BC_POLICY_ROLLOUT`.

Bucket audit:

- `bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed883_bucket.out:1-10`
  - overall impact `0.004882812`, success `0.662109375`.
  - `(1,0)` low_x impact `0.041095890`, low-motion `0.315068493`, success
    `0.315068493`.
  - `(1,0)` mid_x impact `0`, success `0.511363636`.
  - `(1,0)` high_x impact `0`, success `0.649484536`.
  - verdict `PASS_POSX_BUCKET_SCREEN`.

## Frozen Rollout Seed884 Cross-Check

Runtime:

- `bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed884_progress.out:1-8`
  confirms `device=cuda:0`, `num_envs=1024`, total steps `580`.
- `bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed884_stdout.out:1-8`
  confirms AppLauncher and environment device `cuda:0`.

Overall audit:

- `bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed884_audit.out:1-6`
  - learned policy `True`, DiffIK controller `False`, training `False`,
    posewrite calls `0`.
  - controlled `0.943359375`, impact `0.010742188`, low-motion `0.024414062`,
    success `0.662109375`.
  - `(1,0)` success `0.483870968`, impact `0.010752688`.
  - verdict `PASS_LEARNED_BC_POLICY_ROLLOUT`.

Bucket audit:

- `bc_mlp_joint_delta_v3_safety_l2_scale_rollout1024_seed884_bucket.out:1-10`
  - overall impact `0.010742188`, success `0.662109375`.
  - `(1,0)` low_x impact `0.035714286`, low-motion `0.261904762`, success
    `0.309523810`.
  - `(1,0)` mid_x impact `0`, success `0.516853933`.
  - `(1,0)` high_x impact `0`, success `0.594339623`.
  - verdict `PASS_POSX_BUCKET_SCREEN`.

## Interpretation

- The safety-aware BC v3 gate passes two 1024 frozen learned-policy audits.
- Compared with BC v2 seed883, the important safety change is:
  - overall impact `0.015625000 -> 0.004882812`.
  - `(1,0)` low_x impact `0.068493151 -> 0.041095890`.
  - `(1,0)` high_x impact `0.061855670 -> 0`.
- The tradeoff is real:
  - overall success `0.679687500 -> 0.662109375`.
  - low_x success `0.410958904 -> 0.315068493` on seed883.
  - low_x low-motion `0.082191781 -> 0.315068493` on seed883.
- Therefore this is a safety gate pass, not a final learned-policy robustness claim.

## Decision

- Do not run 10k/100k learned-policy scaling from this alone.
- Do not run PPO/RL without explicit approval.
- Next valid steps:
  1. tune safety-aware action/loss to recover low_x motion/success while preserving
     the impact gate, or
  2. request explicit approval for a small safety-aware RL warm-start pilot from this
     checkpoint, then repeat 1024 overall and per-bucket audits.

## Updated State

- Updated `START_HERE.md`.
- Appended `claudedocs/EXPERIMENT_LEDGER.md`.
- Appended `claudedocs/DECISIONS.md` D108.
