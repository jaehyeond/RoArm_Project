# D313 Cube10cm PPO Trigger and Observation-Noise Perturbation Contract

Date: 2026-07-07 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch after D312. This session did not run Isaac runtime, PPO, tiny PPO trace gates, TensorBoard training, torchrun, learned-policy updates, RoArm deployment, Track A, VLA/SmolVLA fine-tuning, B200/SSH, pull, or `.ssh` copy.

## Starting Point

D312 fixed the strict useful contract, but the perturbation protocol still had three concrete gaps:

- PPO had only a negative gate: do not start before perturbation.
- The perturbation matrix omitted observation noise, even though the primitive reads privileged sim cube pose.
- Severity escalation was open-ended and could replace the old seed-only loop.

## Code Changes

- `roarm_rl/roarm_cube_push_env.py`
  - Added `candidate6_diffik_cube_pose_noise_xy_m`.
  - Samples per-env XY cube pose noise on reset.
  - Applies that noise only to the controller's cube reference inside the candidate6/tap-push primitive target calculation.
  - Leaves reward/metric/referee terms on ground-truth cube state.
  - Logs cube pose noise magnitude and sampled noise mean/max.

- `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`
  - Added `--primitive_cube_pose_noise_xy_m`.
  - Wires the value into env config for primitive execution.
  - Records configured and sampled pose-noise statistics in summary JSON.

## Protocol Changes

- The perturbation matrix is now 9 rows:
  - nominal `1`
  - size `2`
  - mass `2`
  - friction `2`
  - observation noise `2`: `0.005m` and `0.015m`
- Metric/success computation remains ground truth; noise affects only the controller's cube reference.
- Primitive-parameter PPO starts immediately after the 9-row matrix completes, regardless of result.
- Severity escalation is capped at one round. No unlimited severity ladder.

## Why No Runtime Experiment In This Session

This session was a protocol/code correction requested before launching the perturbation matrix. Running the matrix before adding the observation-noise row and positive PPO trigger would preserve the exact loophole identified in the review. This is not permission for another no-runtime hardening session: the next research session must execute the 9-row perturbation matrix or explicitly justify why execution is impossible.

## Next Required Experiment

Run the D312/D313 9-row perturbation matrix. After those rows complete, start primitive-parameter PPO. If the baseline fails, train against the failing axis. If it passes, train under domain randomization and evaluate on combined/severe rows.

## Verification

- `python -m py_compile roarm_rl/roarm_cube_push_env.py sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py` passed.
- `git diff --check` passed.
- `pgrep -af '[i]saaclab|[t]rain_cube_push_ppo|[t]ensorboard|[t]orchrun|[r]l_games'` returned no residual Isaac/PPO/TensorBoard/torchrun/rl_games processes.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` reported no compute apps.
- `nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv` reported GPU utilization `0%`.

## Verdict

`D313_PPO_TRIGGER_OBS_NOISE_PROTOCOL_READY_NO_RUNTIME_NO_PPO`
