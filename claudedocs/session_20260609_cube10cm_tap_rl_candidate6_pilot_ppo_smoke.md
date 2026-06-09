# Session 2026-06-09 - Candidate6 Pilot PPO Smoke

## Scope

Professor 10cm/0.72kg cube tap RL branch only.  This session did not use
Track A, B200/SSH, robot control, dataset generation, large PPO scale-up, or
RoArm deployment.

## Fixed Contract

- Env: `RoArm-CubeTap10cm-Direct-v0`
- Cube: `(0.240, 0.000)`, push dir `(+1, 0)`
- Contact proxy: `link5_collision_aabb`
- Tool proxy: `hand_tcp`
- Precontact clearance: `0.040m`
- Episode: `6.08s`, eval steps `580`
- Policy target displacement: `0.006m`
- Step clip: `0.010rad`
- Joint target lead limit: `0.060rad`
- Scripted teacher blend: `0.0`
- Robot USD:
  `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd`

## Commands

Preflight:

```bash
conda run -n isaaclab --no-capture-output python -u -m roarm_rl.train_cube_tap10cm_ppo_smoke --device cuda:0 --seed 966 --num_envs 8 --max_iterations 0 --num_steps_per_env 64 --eval_steps 580 --initial_policy_eval --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_pilot_ppo_preflight_summary.json --summary_out claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_pilot_ppo_preflight_summary.out
```

Tiny PPO smoke:

```bash
conda run -n isaaclab --no-capture-output python -u -m roarm_rl.train_cube_tap10cm_ppo_smoke --device cuda:0 --seed 966 --num_envs 8 --max_iterations 3 --num_steps_per_env 64 --eval_steps 580 --initial_policy_eval --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_pilot_ppo_smoke_summary.json --summary_out claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_pilot_ppo_smoke_summary.out
```

Corrected posthoc checkpoint eval:

```bash
conda run -n isaaclab --no-capture-output python -u -m roarm_rl.train_cube_tap10cm_ppo_smoke --device cuda:0 --seed 966 --num_envs 8 --max_iterations 0 --num_steps_per_env 64 --eval_steps 580 --load_checkpoint claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_runs/cube10cm_tap_rl_candidate6_pilot_ppo_smoke/seed966_env8_it3/model_2.pt --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_pilot_ppo_smoke_posthoc_summary.json --summary_out claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_pilot_ppo_smoke_posthoc_summary.out
```

## Results

- Preflight PASS:
  `preflight_pass=True`, contract violations `0`, zero policy and untrained
  policy finite, and no contact/success as expected.
- Tiny PPO training smoke PASS:
  `max_iterations=3`, `checkpoint_exists=True`, `training_smoke_pass=True`.
  Checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_runs/cube10cm_tap_rl_candidate6_pilot_ppo_smoke/seed966_env8_it3/model_2.pt`.
- Corrected posthoc checkpoint eval PASS candidate:
  `tap_contact_seen_max=1.0`, `reaction_seen_max=1.0`,
  `tap_success_max=1.0`, `tap_overshoot_max=0.0`,
  and `policy_task_pass=True`.
- Quality caveat:
  the pass is fixed-contract tiny RL policy evidence only.  Posthoc details
  include `tap_disp_max=3.32072377204895e-05`,
  `tcp_cube_dist_min_m=0.08132576197385788`,
  `target_lead_limit_rate_max=0.5`, and `joint_delta_cap_rate_max=0.5`.
  Do not promote this directly to large PPO/RL, dataset generation, or RoArm.

## Caveats

- The first version of the smoke script used a wrong default USD path and was
  fixed to the same local backup USD used by the Candidate6 positive-control
  harness.
- The first PPO smoke summary underreported policy task success because the
  smoke script read non-existent log keys such as `cube_tap_success` instead
  of env keys such as `cube_tap_success_rate`.  The corrected posthoc summary
  supersedes it for policy task metrics.
- One redirected `bash -lc` posthoc attempt wrote only stdout and did not update
  summary files because Isaac/Kit could not acquire CUDA in that sandboxed
  launch.  The valid corrected posthoc result is the later direct `conda run`
  run that wrote the summary at `2026-06-09 15:46 KST`.

## Next Branch

Run a fixed-contract RL policy promotion validation, not large PPO or RoArm:

- evaluate the same `model_2.pt` under independent reset seeds and small
  env-scale settings;
- keep Candidate6 geometry/contact/reward contract fixed;
- report strict env success and quality-tier metrics separately;
- keep `large_dataset_rl_roarm_unblocked=NO` until the validation ladder passes.

## D209 Follow-Up: Candidate6 Residual Action Path Bridge

The action-path critique was valid: Candidate6 PASS used built-in DiffIK direct
target application, while the PPO env default action path was raw joint-delta.
D208 PPO smoke did not transfer the Candidate6 controller into PPO's action
space.

Implemented a default-off env action mode:

- `rl_action_mode=candidate6_diffik_residual_joint`
- base controller: Candidate6 near-face built-in DiffIK, previous-target-base,
  `0.010rad` step clip, `0.060rad` lead limit
- policy output: small residual around that base target
- default `joint_delta` behavior preserved
- added `tap_success_terminate` and Candidate6 residual contract logging to the
  smoke runner

Static bridge audit:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_action_path_bridge_design_summary.out:1-8
```

First bridge preflight, no training, no success termination:

```bash
conda run -n isaaclab --no-capture-output python -u -m roarm_rl.train_cube_tap10cm_ppo_smoke --device cuda:0 --seed 966 --num_envs 8 --max_iterations 0 --num_steps_per_env 64 --eval_steps 580 --rl_action_mode candidate6_diffik_residual_joint --candidate6_diffik_residual_scale_rad 0.002 --candidate6_diffik_push_steps 580 --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_preflight_summary.json --summary_out claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_preflight_summary.out --experiment_name cube10cm_tap_rl_candidate6_diffik_residual_preflight
```

Result: bridge worked, strict quality failed after success:

- `candidate6_active_rate_max=1.0`
- `candidate6_numeric_ok_rate_min=1.0`
- `tap_success_max=1.0`
- `contact_seen_max=1.0`
- `reaction_seen_max=1.0`
- `tap_overshoot_max=0.625`

Interpretation: the controller was moved into the action path, but fixed
580-step eval without success termination keeps pushing after the tap objective
is already achieved.

Second pass-route preflight, no training, success termination enabled:

```bash
conda run -n isaaclab --no-capture-output python -u -m roarm_rl.train_cube_tap10cm_ppo_smoke --device cuda:0 --seed 966 --num_envs 8 --max_iterations 0 --num_steps_per_env 64 --eval_steps 580 --rl_action_mode candidate6_diffik_residual_joint --candidate6_diffik_residual_scale_rad 0.002 --candidate6_diffik_push_steps 580 --tap_success_terminate --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_preflight_summary.json --summary_out claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_preflight_summary.out --experiment_name cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_preflight
```

Result: strict zero-policy pass candidate:

- `contract_violations=[]`
- `candidate6_active_rate_max=1.0`
- `candidate6_numeric_ok_rate_min=1.0`
- `tap_contact_seen_max=1.0`
- `reaction_seen_max=1.0`
- `tap_success_max=1.0`
- `tap_overshoot_max=0.0`
- `zero_policy_task_pass=True`

Posthoc comparison:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_bridge_posthoc_audit_summary.out:1-6
```

Current next step:

- Run one tiny training smoke under the new action-path contract:
  `rl_action_mode=candidate6_diffik_residual_joint` plus
  `tap_success_terminate=True`.
- Do not run raw joint-delta scale-up.
- Do not generate dataset, claim action-teacher readiness, run large PPO, or
  deploy RoArm until the tiny training smoke and promotion validation pass.

## D210 Follow-Up: Candidate6 Residual Success-Terminate PPO Smoke

Ran the next local tiny PPO smoke under the D209 residual action-path contract:

```bash
conda run -n isaaclab --no-capture-output python -u -m roarm_rl.train_cube_tap10cm_ppo_smoke --device cuda:0 --seed 966 --num_envs 8 --max_iterations 3 --num_steps_per_env 64 --eval_steps 580 --initial_policy_eval --rl_action_mode candidate6_diffik_residual_joint --candidate6_diffik_residual_scale_rad 0.002 --candidate6_diffik_push_steps 580 --tap_success_terminate --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_ppo_smoke_summary.json --summary_out claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_ppo_smoke_summary.out --experiment_name cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_ppo_smoke
```

Training smoke result:

- contract: `rl_action_mode=candidate6_diffik_residual_joint`,
  `tap_success_terminate=True`, residual scale `0.002`, seed `966`,
  `num_envs=8`, `max_iterations=3`, contract violations `0`
- zero-policy pre-eval: `tap_success_max=1.0`, `contact_seen_max=1.0`,
  `reaction_seen_max=1.0`, `overshoot_max=0.0`
- initial PPO policy eval: finite, success/contact/reaction `1.0/1.0/1.0`,
  overshoot `0.0`
- training wrote `model_0.pt`, `model_1.pt`, and `model_2.pt`
- post-eval: `tap_success_max=1.0`, `contact_seen_max=1.0`,
  `reaction_seen_max=1.0`, `overshoot_max=0.0`,
  `candidate6_active_rate_max=1.0`,
  `candidate6_numeric_ok_rate_min=1.0`,
  `candidate6_residual_abs_max_max=0.00041367902304045856`,
  lead-limit rate `0.0`, joint-delta cap rate `0.0`
- summary line 7 reports `training_smoke_pass=True` and
  `policy_task_pass=True`

Loaded-checkpoint posthoc eval:

```bash
conda run -n isaaclab --no-capture-output python -u -m roarm_rl.train_cube_tap10cm_ppo_smoke --device cuda:0 --seed 966 --num_envs 8 --max_iterations 0 --num_steps_per_env 64 --eval_steps 580 --rl_action_mode candidate6_diffik_residual_joint --candidate6_diffik_residual_scale_rad 0.002 --candidate6_diffik_push_steps 580 --tap_success_terminate --load_checkpoint claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_runs/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_ppo_smoke/seed966_env8_it3/model_2.pt --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_ppo_smoke_posthoc_summary.json --summary_out claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_ppo_smoke_posthoc_summary.out --experiment_name cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_ppo_smoke
```

Posthoc result:

- checkpoint load succeeded from `model_2.pt`
- contract violations `0`
- loaded policy post-eval PASSed with success/contact/reaction
  `1.0/1.0/1.0`, overshoot `0.0`, Candidate6 active/numeric
  `1.0/1.0`, residual max `0.00041367902304045856`

Audit summary:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_ppo_smoke_audit_summary.out:1-8
```

Interpretation:

- This is the first tiny PPO evidence that the RL policy path can sit on the
  Candidate6 DiffIK controller manifold and preserve strict tap success.
- It is not proof that raw joint-delta PPO is viable; raw joint-delta scale-up
  remains the wrong next branch.
- It is not action-teacher data, large PPO readiness, dataset readiness, VLA, or
  RoArm readiness.

Next research step:

- fixed-contract promotion validation for this residual action-path PPO result;
- independent reset seeds first, then small env-scale if seeds hold;
- keep Candidate6 geometry, contact proxy, success termination, residual scale,
  target path, and episode/eval contract fixed;
- only after promotion validation passes should large PPO/RL planning be
  reconsidered.

## D211 Follow-Up: Candidate6 Residual Promotion Validation

Ran fixed-contract loaded-checkpoint promotion evals for the D210 `model_2.pt`.
No training was run during promotion validation.

Independent reset-seed validation:

- seed967, `num_envs=8`, `max_iterations=0`
- seed968, `num_envs=8`, `max_iterations=0`
- seed969, `num_envs=8`, `max_iterations=0`
- loaded checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_runs/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_ppo_smoke/seed966_env8_it3/model_2.pt`

All three seed summaries:

- line 2: same fixed contract, `violations=0`
- line 6: `tap_success_max=1.0`, `contact_seen_max=1.0`,
  `reaction_seen_max=1.0`, `overshoot_max=0.0`,
  `candidate6_active_rate_max=1.0`,
  `candidate6_numeric_ok_rate_min=1.0`,
  residual max `0.00041367902304045856`,
  lead-limit rate `0.0`, joint-delta cap rate `0.0`
- line 7: `policy_task_pass=True`,
  `large_dataset_rl_roarm_unblocked=NO`,
  `action_teacher_dataset=NO`

Small env-scale validation:

- seed966, `num_envs=16`, `max_iterations=0`
- same checkpoint and same contract
- line 2: contract violations `0`
- line 6: success/contact/reaction `1.0/1.0/1.0`, overshoot `0.0`,
  Candidate6 active/numeric `1.0/1.0`, residual max
  `0.0004136791976634413`, lead-limit and joint-delta cap rates `0.0`
- line 7: `policy_task_pass=True`,
  `large_dataset_rl_roarm_unblocked=NO`,
  `action_teacher_dataset=NO`

Audit summary:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_diffik_residual_success_terminate_promotion_validation_audit_summary.out:1-8
```

Interpretation:

- Promotion validation PASSed for the fixed Candidate6 residual action-path
  pilot PPO checkpoint.
- This is still not raw joint-delta PPO evidence and not a dataset/action
  teacher.
- Large dataset, VLA, and RoArm deployment remain blocked.

Next research branch:

- controlled residual PPO learning ladder under the same contract;
- keep success termination, Candidate6 residual action path, geometry/contact
  proxy, residual scale, target path, and episode/eval contract fixed;
- scale learning only in small steps with explicit quality/reproducibility
  gates before any large PPO/data/RoArm claim.

## D212 Follow-Up: Candidate6 Residual PPO Learning Ladder

Ran the same-contract residual PPO ladder after the D211 promotion validation.
This is the first actual learning-scale ladder on the Candidate6 residual action
path; it is not raw joint-delta PPO and not an action-teacher dataset.

Step accounting:

- L1: `16 * 64 * 20 = 20,480` steps, not 2M.
- L2: `32 * 64 * 50 = 102,400` steps.
- L3: `32 * 64 * 500 = 1,024,000` steps.

L1 default PPO noise `0.8`:

- summary line 6: post-eval success/contact/reaction fell to `0.75/0.75/0.75`
  with overshoot `0.0`;
- TensorBoard `Train/mean_reward` fell `1.41705441 -> 0.120316416`;
- residual max stayed near the `0.002rad` scale;
- classification: health-warning, not promoted.

L1b with `--ppo_init_noise_std 0.2`:

- summary line 7: success/contact/reaction `1.0/1.0/1.0`, overshoot `0.0`,
  lead-limit rate `0.0`, joint-delta cap rate `0.0`, residual max
  `0.00033067120239138603`;
- TensorBoard reward `1.81253302 -> 1.8506515`;
- classification: L1 health PASS.

L2 with `--ppo_init_noise_std 0.2`:

- summary line 7: success/contact/reaction `1.0/1.0/1.0`, overshoot `0.0`,
  lead-limit rate `0.0`, joint-delta cap rate `0.0`, residual max
  `0.00034333037910982966`;
- TensorBoard reward `1.80665839 -> 1.96147859`;
- checkpoint: `model_49.pt`;
- classification: L2 PASS.

L3 with `--ppo_init_noise_std 0.2`:

- summary line 7: success/contact/reaction `1.0/1.0/1.0`, overshoot `0.0`,
  lead-limit rate `0.0`, joint-delta cap rate `0.0`, residual max
  `0.0006555510917678475`;
- TensorBoard reward `1.81298292 -> 1.95703006`, max `2.10515165`;
- policy noise std `0.19977969 -> 0.06192378`;
- residual max `0.000659053 -> 0.000282646`;
- overshoot/lead/cap all stayed `0`;
- checkpoint: `model_499.pt`;
- classification: L3 PASS.

Interpretation:

- The pass route is PPO on the Candidate6 DiffIK residual action path with lower
  initial exploration noise, not raw joint-delta PPO.
- The learned residual is small and stable; Candidate6 remains the base
  controller manifold.
- The next valid branch is loaded-checkpoint promotion/reproducibility
  validation for L3 `model_499.pt` across independent reset seeds and small
  env-scale settings.
- Large dataset generation, action-teacher dataset claims, VLA, and RoArm
  deployment remain blocked until that validation and the next quality-tier
  gates pass.

## D213 Follow-Up: L3 model_499 Promotion Validation

Ran loaded-checkpoint validation for L3 `model_499.pt`. No training was run.
The command used the same fixed Candidate6 residual action-path contract and
`--max_iterations 0 --load_checkpoint`.

Independent seed validation:

- seed974, `num_envs=32`
- seed975, `num_envs=32`
- seed976, `num_envs=32`

All three seed summaries:

- line 3: same fixed contract, `violations=0`
- line 7: loaded-policy success/contact/reaction
  `0.90625/0.90625/0.90625`
- line 7: overshoot `0.0`, lead-limit rate `0.0`, joint-delta cap rate `0.0`
- line 7: residual max `0.0006555510917678475`
- line 8: `policy_task_pass=True`, but that is the script's weaker
  `tap_success_max > 0` criterion and is not strict all-env promotion

Env-scale validation:

- seed977, `num_envs=64`
- line 3: same fixed contract, `violations=0`
- line 7: loaded-policy success/contact/reaction
  `0.859375/0.859375/0.859375`
- line 7: overshoot/lead/cap all `0.0`
- line 7: residual max `0.0006555506261065602`

Interpretation:

- L3 `model_499.pt` is not a strict promotion PASS.
- It remains useful evidence that the learning ladder can run, but the learned
  residual is not robust enough for all-env loaded-checkpoint promotion.
- Because zero-policy pre-eval stayed `1.0`, the likely issue is the learned
  residual nudging a subset of envs away from the Candidate6 base pass.
- Do not move to dataset/RoArm/VLA from this checkpoint.
- Next step should be checkpoint selection or residual regularization: compare
  L2 `model_49.pt` and mid-L3 checkpoints under this same loaded-checkpoint
  gate before more training or larger claims.

## D214 Follow-Up: Randomized Robustness Branch

Shifted from fixed-contract checkpoint selection to the higher-value
randomization question: can RL add robustness when the Candidate6 fixed base is
not already saturated?

Implemented default-off cube position randomization in
`roarm_rl/train_cube_tap10cm_ppo_smoke.py`. With extents left at zero, fixed
seed982 preserved the old contract:

- line 3: cube x/y ranges `[0.24, 0.24]` and `[0.0, 0.0]`
- line 4: zero-policy success/contact/reaction `1.0/1.0/1.0`, overshoot `0.0`

Base-only randomization screens:

- xy +/-3cm seed978 line 4: success/contact/reaction
  `0.359375/0.359375/0.359375`, overshoot `0.046875`
- xy +/-1cm seed979 line 4: success/contact/reaction
  `0.109375/0.109375/0.109375`, overshoot `0.015625`

Small randomized residual PPO L1:

- xy +/-1cm, seed983, `32*64*20=40,960` steps
- line 4 base pre-eval: success/contact/reaction `0.125/0.125/0.125`
- line 7 post-eval: success/contact/reaction `0.0625/0.0625/0.0625`
- line 7 residual max: `0.00031939358450472355`
- line 8: `training_smoke_pass=False`, `policy_task_pass=False`

Reset IK check:

- Added tap-env reset IK metrics to the smoke summary.
- xy +/-1cm resetmetric seed985 line 4 reports `ik_reset_rate_min=1.0`,
  `ik_reset_err_mm_max=1.316048622`, Candidate6 active/numeric `1.0/1.0`.
- The same line still reports success/contact/reaction
  `0.109375/0.109375/0.109375`, so the randomized failure is not reset IK or
  numeric DiffIK activation.

Interpretation:

- Fixed-contract PPO was saturated by the Candidate6 base; randomization is the
  right research direction.
- Current residual PPO does not recover the randomized failures and should not
  be scaled blindly.
- The next pass route is a controller/trajectory robustness candidate for
  randomized cube poses, then PPO after the base manifold is less brittle.
- Large dataset generation, action-teacher claims, VLA, and RoArm deployment
  remain blocked.

## D215 Follow-Up: Candidate7 Current-Pose/Event Metrics and Joint-Residual Stop

User asked whether Candidate6 was really overfit to an absolute cube target.
Checked the target function directly:

- Candidate6 was not hard-coded to `0.25,0.0`.
- The default path used `_cube_start_w`, which is the reset/start cube pose.
- Added opt-in `candidate6_diffik_cube_reference_mode=current_pose`; default
  remains `start_pose`.

Also fixed a success-termination metric/reward ordering issue:

- With `tap_success_terminate=True`, randomized envs succeed at different
  timesteps and reset asynchronously, so old per-step `tap_success_max` could
  underreport episode success.
- `_get_dones()` can call tap-buffer updates before `_get_rewards()`, so the
  one-step `just_succeeded` reward/log event needed a pending latch.
- Added pending success-event metrics and reward-event preservation.

Corrected base screens:

- current-pose fixed seed986: success/contact/reaction `1.0/1.0/1.0`,
  overshoot `0.0`.
- current-pose xy +/-1cm seed998:
  `success_event_count=261`, `success_episode_rate=0.9775280898876404`,
  overshoot `0.015625`.
- current-pose xy +/-3cm seed999:
  `success_event_count=429`, `success_episode_rate=0.7210084033613445`,
  overshoot `0.0625`.
- deterministic corner `x=0.21,y=-0.03` seed1010:
  `success_episode_rate=1.0`, `tap_success_max=1.0`, overshoot `0.0`.

Random xy +/-3cm joint-residual PPO L1 attempts:

- seed1000, residual scale `0.002`, noise `0.2`:
  base success `0.7047619048`, post success `0.7264705882`, but overshoot
  worsened to `0.53125`.
- seed1001, residual scale `0.001`, noise `0.1`:
  base success `0.7103658537`, post success `0.6833855799`, overshoot `0.5`.
- reward-safe seed1002 (`tap_transient_disp_reward_scale=10`,
  `tap_overshoot_penalty_scale=80`, `action_penalty_scale=0.05`):
  base success `0.7037037037`, post success `0.6623376623`, overshoot
  `0.34375`.

Interpretation:

- D214's randomization direction was right, but its old near-zero success
  readout was mostly a metric artifact under success termination.
- The useful robustness task is now xy +/-3cm, where the base is partial
  success around `0.72`.
- Joint-residual PPO should not be scaled to L2 because it fails the
  base-relative safety gate. The next pass route is a lower-dimensional
  cube-relative target/waypoint residual or a controller robustness candidate.
- Dataset generation, action-teacher claims, VLA, and RoArm remain blocked.
