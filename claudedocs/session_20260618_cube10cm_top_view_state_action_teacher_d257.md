# Session 2026-06-18 - Cube10cm Top-View State-Action Teacher D257

## Scope

Active branch:

- Professor 10cm / 0.72kg cube top-view visual trajectory dataset branch.

User requested the continuation path:

```text
LeRobot pair -> RL transition/reward table -> state-action teacher checkpoint -> Isaac Lab PPO with data prior
```

This session completed the state-action teacher checkpoint and PPO smoke-command
preflight step. It did not run Isaac Lab PPO, launch Isaac Lab, render, start
RunPod, delete/archive/move files, use B200/SSH/pull, or control RoArm.

## Current-State Checks

- `git status --short --untracked-files=all --branch` was run first.
- `CLAUDE.md` Current-State Protocol was re-read.
- `START_HERE.md` D256 said the next concrete step was a small state-action
  teacher checkpoint from `ppo_actor_prior_teacher_rows_d256.csv`, then a tiny
  Isaac Lab PPO smoke with data prior.
- Existing `roarm_rl/roarm_cube_push_env.py` was re-checked:
  - `bc_teacher_checkpoint_path`
  - `bc_teacher_blend`
  - `bc_teacher_imitation_reward_scale`
  - 5-arm-delta target requirement
  - runtime clamp through `bc_teacher_policy_delta_clip_rad`
- Existing `roarm_rl/train_cube_push_ppo.py` exposes the corresponding CLI
  knobs.

## Implementation

Added:

- `sim_scripts/cube10cm_top_view_train_state_action_teacher.py`

Purpose:

- read D256 `ppo_actor_prior_teacher_rows_d256.csv`;
- enforce that rows are only `train_clean_positive`;
- train a small MLP teacher compatible with `RoArmCubePushEnv`;
- save a checkpoint with keys expected by the env loader:
  `feature_columns`, `target_columns`, `hidden_dim`, `hidden_layers`,
  `model_state_dict`, `x_mean`, `x_std`, `y_mean`, `y_std`;
- write metrics, training log, summary, and a proposed next-session PPO smoke
  command;
- not run Isaac Lab, PPO, render, cleanup, or RoArm control.

## Runtime

Commands run:

```bash
python3 -m py_compile sim_scripts/cube10cm_top_view_train_state_action_teacher.py
python3 -u sim_scripts/cube10cm_top_view_train_state_action_teacher.py
```

Local `python3` torch status:

- `torch 2.10.0+cu128`
- `torch.cuda.is_available() == False`
- host GPU visible through `nvidia-smi`, but this Python environment cannot use
  CUDA due to NVML/CUDA initialization failure.

Decision:

- Train the teacher on CPU. This is acceptable because it is a small MLP over
  `142978` rows, not Isaac PPO or vision-model fine-tuning.

## Outputs

Output root:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257`

Files:

- `cube10cm_d257_state_action_teacher_clipped0040.pt`
- `state_action_teacher_metrics_d257.json`
- `state_action_teacher_training_log_d257.txt`
- `state_action_teacher_summary_d257.md`
- `ppo_data_prior_smoke_command_d257.txt`

Output size:

- about `176K`

Checkpoint:

- path:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- bytes: `155965`
- sha256:
  `f81df20278ec9ceddef141729f717abbba2412a4a2f9f3a366d88b387caa76b8`

## Metrics

Rows:

- total: `142978`
- train: `128622`
- validation: `14356`
- episodes: `737`
- validation episodes: `74`

Training:

- epochs: `80`
- hidden dim: `128`
- hidden layers: `3`
- batch size: `4096`
- device: `cpu`

Loss:

- baseline validation MSE norm: `1.047513484954834`
- first train MSE norm: `0.6349840723686552`
- final train MSE norm: `0.02590712532401085`
- final validation MSE norm: `0.10824249684810638`

Validation RMSE by joint, radians:

- joint 0: `0.003628572914749384`
- joint 1: `0.004176619462668896`
- joint 2: `0.003567545209079981`
- joint 3: `0.0034607199486345053`
- joint 4: `0.0011242240434512496`

Checkpoint validation:

- required keys present: yes
- feature count: `27`
- target count: `5`
- `x_mean` shape: `(27,)`
- `y_mean` shape: `(5,)`
- checkpoint reload smoke: `True`

## Critical Delta-Clip Decision

D256 warned that the legacy `0.040rad` cap can truncate teacher targets. D257
therefore trained the preflight teacher on targets clipped to `+-0.040rad`,
matching the current env-side `bc_teacher_policy_delta_clip_rad` action cap.

D257 train-clean-only clip audit:

- target clip rad: `0.04`
- raw target clip exceed rate: `0.14844941179761922`

Interpretation:

- This checkpoint is internally consistent with the current env clamp.
- It is not proof that `0.040rad` is the best cap.
- If the next session decides to raise the teacher policy delta cap, retrain the
  checkpoint with a matching target clip and update the PPO command.

## PPO Data-Prior Preflight

Wrote proposed next-session command:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/ppo_data_prior_smoke_command_d257.txt`

Command intent:

- tiny Isaac Lab PPO smoke only;
- `num_envs=32`;
- `max_iterations=2`;
- checkpoint path set to D257 teacher checkpoint;
- `bc_teacher_blend=1.0`;
- `bc_teacher_imitation_reward_scale=5.0`;
- `bc_teacher_policy_delta_clip_rad=0.04`;
- `bc_teacher_phase_timing=direct_steps`;
- `joint_delta_reference=joint_pos`.

Important caution:

- Prior D110/D111-style results showed that teacher-on bridge behavior can work
  while teacher-off PPO can still fail.
- Therefore the next PPO smoke should only prove that the data-prior path loads,
  logs teacher imitation metrics, and exits cleanly.
- It must not be treated as learned RoArm policy success.

## Verification

Passed:

- `python3 -m py_compile sim_scripts/cube10cm_top_view_train_state_action_teacher.py`
- checkpoint reload/key/shape check
- `git diff --check`
- active-process check found no Isaac/PPO/torchrun runtime after training

Disk:

- output root about `176K`
- filesystem after run still about `590G/530G used/30G avail/95%`

## Verdict

`D257_STATE_ACTION_TEACHER_CHECKPOINT_PASS_NO_PPO_RUNTIME`

The branch now has:

1. D247 LeRobot pair dataset;
2. D256 RL transition/reward table;
3. D257 PPO-compatible state-action teacher checkpoint;
4. a proposed D257 tiny Isaac Lab PPO data-prior smoke command.

Still not complete:

- No Isaac Lab PPO runtime was executed in D257.
- No PPO checkpoint exists from this D257 teacher.
- No teacher-off evaluation exists.
- No RoArm deployment/control was attempted.

## Next Session

Recommended next concrete action:

1. Re-read this D257 log and `START_HERE.md`.
2. Inspect `state_action_teacher_metrics_d257.json`.
3. Run the proposed tiny Isaac Lab PPO data-prior smoke command only if runtime
   is explicitly approved.
4. During/after that smoke, verify:
   - checkpoint loads through `bc_teacher_checkpoint_path`;
   - `cube_push_bc_teacher_blend_mean` is nonzero;
   - `cube_push_bc_teacher_imitation_mse` is logged;
   - no Isaac/Kit process remains after completion;
   - GPU is released;
   - no teacher-off success claim is made yet.
