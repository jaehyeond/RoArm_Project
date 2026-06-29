# D298 - Cube10cm tiny PPO direct-reset gate

Date: 2026-06-29 KST

## Scope

- Ran exactly one explicitly approved tiny PPO + TensorBoard gate.
- This used the D297 corrected direct-reset teacher-off contract as the posthoc
  checkpoint validation path.
- No long PPO, PPO ladder, partial actor preservation, render, cleanup,
  RunPod/B200/SSH, Track A, VLA fine-tuning, or RoArm deployment was performed.

## PPO runtime

- Command:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/ppo_command_d298.txt`
- Output root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it`
- PPO exit: clean, exit code `0`.
- Saved checkpoint:
  `model_0.pt`
- Checkpoint sha256:
  `4dcbebbaaafbd50166cd40d2610b903e7209491a542fb8e041dac1cd4b1faf70`
- Runtime contract:
  - D256 reset active, random sample mode;
  - `bc_teacher_blend=0.0`;
  - `bc_teacher_imitation_reward_scale=0.0`;
  - `actor_preserve_blend=1.0`;
  - `tap_success_terminate=True`;
  - `tap_stop_after_disp_m=0.003`;
  - `link5_collision_aabb` contact proxy.

## TensorBoard gate

- Gate artifact:
  `tensorboard_scalar_gate_d298.json`
- Verdict:
  `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- Train scalars were present:
  - `Train/mean_reward=10.783509254455566`;
  - `Train/mean_episode_length=64.90697479248047`.
- Key tap scalars:
  - contact/reaction seen: `0.7029094696044922`;
  - useful seen: `0.04482758790254593`;
  - success: `0.0023168104235082865`;
  - overshoot seen: `0.7133082151412964`;
  - max displacement along push direction: `0.01091606542468071m`;
  - max displacement XY: `0.03478653356432915m`;
  - along `>=1mm` rate: `0.3975215554237366`;
  - XY `>=1mm` rate: `0.7559267282485962`;
  - D256 reset active: `1.0`;
  - BC teacher blend: `0.0`;
  - joint delta cap rate: `0.0`;
  - target lead limit rate: `0.0`;
  - stop-after-displacement hold rate: `0.04251077398657799`.

## Saved-checkpoint teacher-off direct-reset eval

- Seed `29801`:
  - verdict:
    `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`;
  - useful: `0.96875`;
  - overshoot: `0.03125`;
  - mean/max XY displacement: `0.004003090318292379/0.06263629347085953m`;
  - XY `>=1mm` / `>=3mm`: `0.5625/0.46875`;
  - mean/max along displacement: `0.0036056730896234512/0.06229519844055176m`;
  - along `>=1mm` / `>=3mm`: `0.5/0.375`;
  - joint delta cap max trace: `0.0`;
  - D256 reset active: `1.0`;
  - BC teacher blend last: `0.0`.
- Seed `29604`:
  - verdict:
    `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`;
  - useful: `1.0`;
  - overshoot: `0.0`;
  - mean/max XY displacement: `0.0011870721355080605/0.004005730152130127m`;
  - XY `>=1mm` / `>=3mm`: `0.375/0.3125`;
  - mean/max along displacement: `0.0011399425566196442/0.00400543212890625m`;
  - along `>=1mm` / `>=3mm`: `0.34375/0.25`;
  - joint delta cap max trace: `0.0`;
  - D256 reset active: `1.0`;
  - BC teacher blend last: `0.0`.

## Interpretation

- The PPO runtime path itself is wired: it produced TensorBoard events,
  checkpoint output, and no lingering Isaac/PPO/TensorBoard process.
- The collection-time TensorBoard gate failed hard because overshoot was high
  and useful/success rates were too low.
- The saved checkpoint still passes corrected direct-reset teacher-off evals,
  so D298 does not show that the actor checkpoint is destroyed.
- The mismatch points to the PPO collection-time reset/termination contract,
  especially the interaction among random D256 reset, `tap_success_terminate`,
  stop-after-displacement hold timing, and episode recycling.
- This is not a learned-policy success claim and not RoArm readiness.

## Decision

- D298 is a no-promotion result.
- Do not run long PPO.
- Do not run another PPO gate immediately.
- Next work is a non-PPO collection-time contract diagnostic comparing:
  - PPO collection path versus teacher-off direct-reset path;
  - `tap_success_terminate=True` versus a safer termination/hold contract;
  - per-env overshoot traces during collection;
  - stop-after-displacement hold timing;
  - whether reset/recycle during collection changes the state/contact cache.

## Verification

- `python -m py_compile` passed for:
  - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`;
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`;
  - `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`.
- `git diff --check` passed.
- No active Isaac/PPO/TensorBoard/torchrun process remained.
- GPU utilization was `0%`; only pre-existing small Python compute contexts
  remained.

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/EXPERIMENT_LEDGER.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/ppo_command_d298.txt`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/tensorboard_dashboard_command_d298.txt`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it/tensorboard_scalar_gate_d298.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it/teacher_off_direct_seed29801/teacher_off_policy_eval_summary_d298_direct_seed29801.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it/teacher_off_direct_seed29604/teacher_off_policy_eval_summary_d298_direct_seed29604.json`
