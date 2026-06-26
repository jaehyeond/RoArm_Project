# D260 Cube10cm Top-view TensorBoard PPO Gate

Date: 2026-06-19 KST

Scope:

- Professor 10cm / 0.72kg cube top-view visual trajectory branch only.
- No new PPO runtime.
- No long PPO.
- No teacher-off evaluation.
- No RoArm deployment or readiness claim.
- No RunPod/B200/SSH/pull/cleanup.

## Why This Was Added

The D258 PPO data-prior smoke proved that TensorBoard logs exist and that the
D257 teacher wiring is connected. D259 then proved that teacher-only behavior is
not valid yet.

User correctly pointed out that before any long PPO, reward and policy/loss
curves should be inspected through TensorBoard. The corrected procedure is:

- use TensorBoard dashboard for live visual inspection;
- also extract the same event scalars into a JSON/markdown gate so the decision
  is reproducible in repo docs.

TensorBoard is not a substitute for teacher-only contact or teacher-off eval.
Reward improvement alone is not a policy-success claim.

## Tooling

Added:

- `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`

The script reads an existing TensorBoard event log and summarizes:

- reward:
  - `Train/mean_reward`;
  - `Train/mean_episode_length`;
- PPO/policy health:
  - `Loss/value_function`;
  - `Loss/surrogate`;
  - `Loss/entropy`;
  - `Loss/learning_rate`;
  - `Policy/mean_noise_std`;
- task behavior:
  - `Episode/cube_push_disp_along_m`;
  - `Episode/cube_push_disp_xy_m`;
  - `Episode/cube_push_tcp_cube_dist_m`;
  - `Episode/cube_push_controlled_rate`;
  - `Episode/cube_push_low_motion_rate`;
  - `Episode/cube_push_success_rate`;
  - action/joint cap metrics;
  - BC teacher blend/imitation metrics;
  - tap-contact/tap-success metrics when present.

The gate writes:

- `tensorboard_scalar_gate_d260.json`;
- `tensorboard_scalar_gate_d260.md`.

## TensorBoard Availability

System Python does not have TensorBoard.

The `isaaclab` conda env has TensorBoard:

- version `2.20.0`;
- event accumulator import works.

Dashboard command pattern:

```bash
conda run -n isaaclab tensorboard --logdir <ppo_log_dir> --host 127.0.0.1 --port 6006
```

For the existing D258 smoke:

```bash
conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d257_logs/cube10cm_d257_data_prior_smoke2 --host 127.0.0.1 --port 6006
```

## D258 Event-log Gate Result

Command:

```bash
conda run -n isaaclab python sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py --log_dir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d257_logs/cube10cm_d257_data_prior_smoke2
```

Outputs:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d257_logs/cube10cm_d257_data_prior_smoke2/tensorboard_scalar_gate_d260.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d257_logs/cube10cm_d257_data_prior_smoke2/tensorboard_scalar_gate_d260.md`

Verdict:

`TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`

Issues:

- no task success/contact signal in TensorBoard;
- `cube_push_low_motion_rate` last `0.9778646230697632`;
- `cube_push_joint_delta_cap_rate` max `0.7411024570465088`.

Warnings:

- short run: `Train/mean_reward` has only `1` point;
- `cube_push_tcp_cube_dist_m` last `0.3268700838088989`;
- `cube_push_disp_along_m` last `0.00015073080430738628`;
- `cube_push_controlled_rate` last `0.0182291679084301`.

Selected scalar values:

- `Train/mean_reward`: `-392.5340270996094`;
- `Loss/value_function`: `6711.08642578125 -> 6737.7255859375`;
- `Loss/surrogate`: `-0.011346347630023956 -> -0.012086811475455761`;
- `Loss/entropy`: `7.177923202514648 -> 7.179165840148926`;
- `Policy/mean_noise_std`: `0.8005133867263794 -> 0.8006278276443481`;
- `cube_push_success_rate`: `0.0 -> 0.0`;
- `cube_push_bc_teacher_imitation_mse`:
  `1.2104418277740479 -> 1.253436803817749`.

## Interpretation

D260 does not change the D259 blocker. It strengthens the process:

- D258 remains wiring-only.
- D259 remains behavior-fail / feature-contract-mismatch.
- TensorBoard confirms D258 is not a promotion candidate:
  reward is deeply negative, success/contact is zero, low-motion remains high,
  and action/joint saturation is high.

## Updated Next Order

Do not launch longer PPO.

Next sequence:

1. Fix feature contract and runtime contract:
   - 10cm env selection;
   - D256 `target_position_world_m` semantics vs env `_bc_teacher_tcp_target()`;
   - +x push direction or retraining for randomized directions;
   - reset/initial-joint distribution.
2. Rerun teacher-only feature-alignment probe.
3. Only if teacher-only reaches plausible contact without saturation/explosion,
   run a tiny 10cm PPO smoke.
4. During/after that tiny PPO smoke:
   - open TensorBoard dashboard;
   - run `cube10cm_top_view_tensorboard_scalar_gate.py`;
   - require reward/loss/policy health and task metrics to agree.
5. Then run teacher-off frozen eval.
6. Only after tiny smoke plus TensorBoard gate plus teacher-off eval pass should
   a short PPO ladder be considered. Long PPO remains later than that.

Do not treat TensorBoard reward increase as learned policy, teacher-off success,
or RoArm readiness.
