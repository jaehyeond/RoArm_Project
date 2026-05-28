# Session 2026-05-28 - Cube3cm v3.2 Teacher Sweep and Bucket BC Gate

## Scope

- Professor cube3cm push/tap branch only.
- Not Track A grasp, not Track A dataset/training, not PPO/RL/VLA success.
- No B200, no SSH, no pull, no `.ssh` copy. Local IsaacLab/GPU only where runtime
  was needed.
- Goal: continue after BC v1 by testing `(1,0)` teacher/BC quality before any
  larger PPO/RL step.

## Worktree / Safety

- `git status --short --untracked-files=all --branch` was run at start and again
  after interruption. Existing dirty/untracked state was preserved.
- Local process check after interruption found no active IsaacLab/conda cube-push
  process.
- Static checks passed:
  - `python -m py_compile sim_scripts/cube3cm_push_diffik_bucket_audit.py sim_scripts/cube3cm_push_diffik_build_dataset.py sim_scripts/cube3cm_push_diffik_dataset_v2_audit.py sim_scripts/cube3cm_push_diffik_train_bc.py sim_scripts/cube3cm_push_bc_policy_rollout.py sim_scripts/cube3cm_push_bc_rollout_audit.py`
  - `git diff --check`

## Code Changes

- Added `sim_scripts/cube3cm_push_diffik_bucket_audit.py`
  - md5 `596cf6193a1f27192d7574656b2e083b`
  - Audits direction plus `(1,0)` low/mid/high-x buckets.
- Extended `sim_scripts/cube3cm_push_diffik_build_dataset.py`
  - md5 `ad0ded036401ed6a449f52abdcf08b07`
  - Added `--balance_mode direction_posx_bucket`, `--posx_bucket_edges`,
    `--min_selected_per_posx_bucket`, and `--max_per_posx_bucket`.
  - Adds `posx_x_bucket` to dataset rows and bucket counts to manifest.
- Extended `sim_scripts/cube3cm_push_diffik_dataset_v2_audit.py`
  - md5 `2341c3406e27ee1f16a8dd777de82cb5`
  - Audits bucket balance and train/val/test bucket coverage when manifest
    balance mode is `direction_posx_bucket`.

Unchanged-but-used md5s:

- `cube3cm_push_diffik_train_bc.py` md5 `df03abb00188cfd9b644b0ef410a0e14`
- `cube3cm_push_bc_policy_rollout.py` md5 `a56e9a96feaad196fef6e1081c0116ec`
- `cube3cm_push_bc_rollout_audit.py` md5 `121721561bcde8df141f56207dcab14d`

## Teacher Sweep

The first candidate was a useful negative because stdout exposed a bad comparison:
`diffik_probe_v32cand_t270_p036_eval512_seed790_stdout.out:20-21` had
`base_steps=55/35/30` and `max_diffik_joint_step_rad=0.012`, not the v3.1
comparison condition. That run was rejected as condition mismatch, not as a
teacher result.

Corrected v3.1 baseline seed790:

- `diffik_probe_v31_baseline_eval512_seed790_audit.out:1-6`
  - controlled `0.966796875`, impact `0.003906250`, low-motion `0.037109375`,
    success `0.609375000`.
- Bucket audit:
  `diffik_probe_v31_baseline_eval512_seed790_bucket.out:1-10`
  - `(1,0)` overall success `0.401408451`.
  - low_x success `0.163265306`, impact `0`.

Rejected candidate t270/p036:

- `diffik_probe_v32cand_t270_p036_eval512_seed790_fixed_bucket.out:1-10`
  - overall success improved to `0.677734375`.
  - `(1,0)` success improved to `0.591549296`, but `(1,0)` impact rose to
    `0.154929577`.
  - low_x impact `0.102040816`, mid_x impact `0.073170732`, high_x impact
    `0.269230769`.
  - Verdict: reject; success was bought with impact.

Candidate t257/p034:

- seed790:
  `diffik_probe_v32cand_t257_p034_eval512_seed790_bucket.out:1-10`
  - overall impact `0.005859375`, success `0.628906250`.
  - `(1,0)` success `0.471830986`, low_x success `0.367346939`, low_x impact
    `0.020408163`.
  - Bucket screen PASS on this seed.
- seed791:
  `diffik_probe_v32cand_t257_p034_eval512_seed791_bucket.out:1-10`
  - overall impact `0.005859375`, success `0.605468750`.
  - `(1,0)` low_x success fell to `0.162790698`, low_x low-motion rose to
    `0.581395349`, low_x final TCP error `0.061915847`.
  - Bucket screen FAIL.

Interpretation:

- No robust v3.2 scripted teacher candidate was accepted.
- The weak `(1,0)` low_x pocket remains trajectory-sensitive and seed-sensitive.
- Overall metrics alone are misleading; per-bucket audit is required.

## Bucket Dataset

Built bucket-balanced dataset v3 from the existing v3.1 all-env trace, without
new physics:

- Source trace:
  `diffik_probe_v31_datasetv2_1024_seed779_trace.csv`
- Source summary:
  `diffik_probe_v31_datasetv2_1024_seed779_summary.json`
- Build:
  `diffik_state_action_dataset_v3_bucket_1024_seed779_build.out:1-6`
  - source trace rows `148480`, source env count `1024`.
  - eligible env count `612`.
  - selected env count `180`, selected rows `26100`.
  - 45 trajectories per direction.
  - `(1,0)` low/mid/high-x buckets `15/15/15`.
  - train/val/test env split `124/28/28`.
  - `full_dataset_candidate=YES`.
- Audit:
  `diffik_state_action_dataset_v3_bucket_1024_seed779_audit.out:1-8`
  - rows `26100`, env count `180`, frames/env `145`.
  - schema/finite OK.
  - split leakage OK.
  - direction coverage OK.
  - final teacher rates controlled `1.0`, impact `0.0`, low-motion `0.0`,
    success `1.0`.
  - `balance_mode=direction_posx_bucket`, bucket OK, split bucket OK.
  - verdict `PASS_FULL_STATE_ACTION_DATASET_V2 full_dataset_ready=YES`.

Interpretation:

- This is a smaller but cleaner teacher-filtered state-action dataset for BC,
  explicitly countering the old `(1,0)` high-x selection bias.
- It is still not an image dataset and not PPO/RL data by itself.

## BC v2 Bucket Training

Training log:

- `diffik_state_action_dataset_v3_bucket_1024_seed779_bc_train.out:1-4`
  - rows `26100`, train/val/test `17980/4060/4060`.
  - train MSE `0.946003509 -> 0.005588359`.
  - val MSE `1.040071607 -> 0.024632571`.
  - test MSE `1.002809405 -> 0.022629632`.
  - mean test MAE `0.001227073rad`.
  - verdict `PASS_BC_TRAINED_CHECKPOINT`.

Interpretation:

- The checkpoint learns the supervised mapping.
- The test loss is worse than dataset v2 BC because v3 bucket is smaller and harder,
  but it still passes the BC checkpoint gate.

## Learned Rollout

BC v2 bucket, default `policy_delta_clip_rad=0.040`:

- Progress:
  `bc_mlp_joint_delta_v2_bucket_rollout1024_seed883_progress.out:1-10`
  confirms `device=cuda:0`, rollout done, artifacts written, skip sim close.
- Learned rollout audit:
  `bc_mlp_joint_delta_v2_bucket_rollout1024_seed883_audit.out:1-6`
  - learned policy `True`, DiffIK controller `False`, training `False`,
    posewrite calls `0`.
  - controlled `0.961914062`, impact `0.015625000`, low-motion `0.016601562`,
    success `0.679687500`.
  - `(1,0)` success `0.527131783`, impact `0.042635659`.
  - verdict `PASS_LEARNED_BC_POLICY_ROLLOUT`.
- Bucket audit:
  `bc_mlp_joint_delta_v2_bucket_rollout1024_seed883_bucket.out:1-10`
  - low_x success `0.410958904`, impact `0.068493151`.
  - mid_x success `0.420454545`, impact `0`.
  - high_x success `0.711340206`, impact `0.061855670`.
  - bucket screen FAIL.

Comparison to BC v1:

- `bc_mlp_joint_delta_v1_rollout1024_seed883_bucket.out:1-10`
  - overall success `0.648437500`, impact `0.012695312`.
  - `(1,0)` success `0.453488372`, impact `0.038759690`.
  - low_x success `0.287671233`, impact `0.041095890`.

So bucket-balanced BC v2 improved:

- overall success `0.648437500 -> 0.679687500`.
- `(1,0)` success `0.453488372 -> 0.527131783`.
- low_x success `0.287671233 -> 0.410958904`.

But it worsened:

- overall impact `0.012695312 -> 0.015625000`.
- low_x impact `0.041095890 -> 0.068493151`.
- high_x impact stayed high at `0.061855670`.

Policy clip checks:

- Clip `0.035`:
  `bc_mlp_joint_delta_v2_bucket_clip035_rollout1024_seed883_audit.out:1-6`
  - overall success `0.689453125`, impact `0.014648438`.
  - `(1,0)` success `0.511627907`, impact `0.038759690`.
  - bucket audit still FAIL with low_x impact `0.068493151`.
- Clip `0.030`:
  `bc_mlp_joint_delta_v2_bucket_clip030_rollout1024_seed883_audit.out:1-6`
  - overall success `0.630859375`, impact `0.020507812`.
  - `(1,0)` impact `0.050387597`.
  - worse overall tradeoff.

Interpretation:

- Bucket-balanced BC is a real improvement in success, especially low_x.
- It is not a safety pass.
- Reducing output clip did not solve low_x impact; the issue is not merely a
  single clamp setting.

## Gate Decision

- PPO/RL scale-up remains blocked.
- The current best learned BC is useful evidence that the policy pipeline and
  bucket balancing help, but it fails per-bucket impact.
- Do not run 10k/100k learned-policy robustness, PPO, or RL fine-tuning from this
  checkpoint as a success-claim path.

Next valid technical step:

1. Teacher/action-distribution redesign for `(1,0)` low_x and high_x impact, or
2. Safety-aware BC/RL warm-start objective that explicitly penalizes bucket impact,
3. Then another 1024 frozen learned rollout with per-direction/per-bucket audit.

## Updated State Files

- Updated `START_HERE.md` with current truth and next gate.
- Appended `claudedocs/EXPERIMENT_LEDGER.md` row.
- Appended `claudedocs/DECISIONS.md` D107.
