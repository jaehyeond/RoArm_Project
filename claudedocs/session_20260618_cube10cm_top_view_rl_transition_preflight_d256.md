# Session 2026-06-18 - Cube10cm Top-View RL Transition Preflight D256

## Scope

Active branch:

- Professor 10cm / 0.72kg cube top-view visual trajectory dataset branch.

User corrected the next objective: the dataset is not just for reporting and
should feed a reinforcement-learning path. The requested direction was:

```text
LeRobot pair -> RL transition/reward dataset -> Isaac Lab PPO with data prior
```

This session implemented the first concrete preflight step for that path. It
did not train a teacher, run PPO, launch Isaac Lab, render, start RunPod, delete
or move files, use B200/SSH, or control RoArm.

## Current-State Checks

- `git status --short --untracked-files=all --branch` was run first.
- `CLAUDE.md` confirms that `START_HERE.md`, `DECISIONS.md`,
  `EXPERIMENT_LEDGER.md`, and session logs are the current truth, while
  `HANDOFF.md` and `TASKS.md` are stale unless explicitly referenced.
- D232 says professor feedback changed the practical work from PPO promotion to
  a camera-calibrated visual trajectory dataset path, and the dataset format
  should be LeRobot-style video+parquet with state/action/object/camera metadata.
- D253 says the selected LeRobot training input was image `[4,3,720,1280]`,
  state `[4,6]`, action `[4,6]`, but no training was run.
- D255 says local `lerobot` CUDA is broken, so SmolVLA smoke was not run.

## Critical Reframe

The corrected RL interpretation is:

- Pure PPO from scratch does not use the 50GB visual pair artifact directly.
- The current pair data must first become transitions:
  `(image/state_t, action_t, reward_t, image/state_t+1, done)`.
- For the current repo's existing PPO path, the most practical data-prior route
  is:
  1. build transition/reward data from the LeRobot pair;
  2. extract a state-action teacher-prior table from clean transitions;
  3. train a small state-action teacher checkpoint;
  4. run Isaac Lab PPO with `bc_teacher_checkpoint_path` and
     `bc_teacher_imitation_reward_scale`.

This is not the same as claiming a final BC policy. BC-style fitting is used
only as the data prior/teacher bridge into PPO.

## Existing Code Hooks Verified

`roarm_rl/roarm_cube_push_env.py` already supports the data-prior PPO bridge:

- `bc_teacher_checkpoint_path`
- `bc_teacher_blend`
- `bc_teacher_imitation_reward_scale`
- teacher target count must be 5 arm deltas
- reward includes `bc_imitation_penalty`

`roarm_rl/train_cube_push_ppo.py` already exposes the corresponding CLI knobs.

## Script Added

Added:

- `sim_scripts/cube10cm_top_view_build_rl_transition_dataset.py`

What it reads:

- `frames.jsonl`
- `label_package_d248/episode_split_manifest.csv`
- `lerobot_dataset_av1_d247/meta/info.json`

What it writes:

- `rl_transitions_d256.csv`
- `ppo_actor_prior_teacher_rows_d256.csv`
- `rl_transition_preflight_summary_d256.json`
- `rl_transition_preflight_brief_d256.md`

Output root:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/rl_transition_preflight_d256
```

## Command Run

```bash
python3 -m py_compile sim_scripts/cube10cm_top_view_build_rl_transition_dataset.py
python3 -u sim_scripts/cube10cm_top_view_build_rl_transition_dataset.py
```

## Result

Status:

- `PASS`

Counts:

- frames read: `195000`
- episodes read: `1000`
- transitions: `194000`
- expected transitions: `194000`
- teacher-prior rows: `142978`

Transition counts:

- `train_positive_actor_prior`: `142978`
- `eval_positive_holdout`: `15908`
- `eval_negative_overshoot_diagnostic`: `32398`
- `quarantine_camera_excluded`: `2716`

Done-transition episode counts:

- `train_positive_actor_prior`: `737`
- `eval_positive_holdout`: `82`
- `eval_negative_overshoot_diagnostic`: `167`
- `quarantine_camera_excluded`: `14`

Output sizes:

- output root: about `397M`
- `rl_transitions_d256.csv`: `317323201` bytes
- `ppo_actor_prior_teacher_rows_d256.csv`: `98811042` bytes
- summary JSON: `5240` bytes
- brief markdown: `620` bytes

## Reward Schema

Reward version:

- `d256_candidate_v0_not_final`

Meaning:

- This is a candidate reward for preflight and audit.
- It is not final PPO reward design.

Components:

- terminal `clean_useful_tap`: `+1.0`
- terminal `contact_reaction_with_overshoot`: `-1.0`
- terminal `camera_quality_fail`: `-2.0`
- contact onset: `+0.25`
- reaction onset: `+0.25`
- overshoot onset: `-0.75`
- progress delta: `0.05 * clip(delta_disp_along / 0.006, -1, 1)`

Critical caution:

- The `0.006m` term is only a low-weight progress scale in the candidate dense
  reward, not the primary success claim.
- The research objective remains physical contact/reaction without overshoot.

## PPO Data-Prior Compatibility

The teacher-prior table uses the feature schema expected by
`roarm_rl/roarm_cube_push_env.py`:

- 27 feature columns:
  `push_dx`, `push_dy`, `phase_alpha`, cube/tcp/target local positions,
  relative vectors, arm joints, and gripper joint.
- 5 target columns:
  `joint_delta_0_rad` through `joint_delta_4_rad`.
- Target semantics:
  `joint_delta_i_rad = action_joint_target_i_t - state_i_t` for arm joints
  `0..4`.

Action-delta audit:

- `joint_delta_abs_p95`: `0.0629434585571289`
- `joint_delta_abs_p99`: `0.07576298713684082`
- `joint_delta_clip_exceed_rate` at `0.040rad`: `0.16487835051546393`

Interpretation:

- A later teacher checkpoint or PPO bridge must explicitly decide whether to
  clip, scale, or use a larger policy-delta cap.
- Blindly using the old `0.040rad` cap would truncate about `16.5%` of arm joint
  delta entries.

## Verdict

`D256_RL_TRANSITION_REWARD_PREFLIGHT_PASS_NO_TRAINING`.

The pair dataset is now converted into an RL-transition/reward preflight dataset
and a PPO-compatible state-action teacher-prior table.

This does not mean RL is complete or ready for RoArm. It means the next
implementation step is now concrete:

1. train a small state-action teacher checkpoint from
   `ppo_actor_prior_teacher_rows_d256.csv`;
2. verify the checkpoint loads in `RoArmCubePushEnv`;
3. run a tiny Isaac Lab PPO smoke with `bc_teacher_imitation_reward_scale > 0`;
4. only after that evaluate teacher-off PPO behavior.

## Still Blocked Until Explicit Approval

- Teacher checkpoint training.
- Isaac Lab PPO runtime.
- RunPod runtime.
- Additional render or dataset generation.
- Raw cleanup/delete/archive/move.
- RoArm deployment/control.
- B200/SSH/pull.
