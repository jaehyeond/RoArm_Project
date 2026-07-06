# D308 Cube10cm Top-View Env Governor Control Repair

Date: 2026-07-01 KST

Scope: professor 10cm / 0.72kg cube top-view visual trajectory branch after
D307. This session did not run PPO, tiny PPO trace gates, TensorBoard training,
torchrun, RoArm deployment, Track A, SmolVLA/VLA fine-tuning, B200/SSH, pull,
render, or cleanup.

## Boot / Current Truth Checked

- Followed `CLAUDE.md` Current-State Protocol.
- Rechecked D307 truth from:
  - `START_HERE.md`
  - `claudedocs/DECISIONS.md`
  - `claudedocs/session_20260630_cube10cm_top_view_d307_action_governor.md`
  - `claudedocs/EXPERIMENT_LEDGER.md`
  - `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`
  - `roarm_rl/roarm_cube_push_env.py`
  - `roarm_rl/train_cube_push_ppo.py` as reference only; PPO was not run.
- D307 verified starting point:
  - `predict_stop h=0.020s, v=0.200m/s` helped the overshoot-heavy D306 ep561
    case: max XY `0.004996m`, useful `1.0`, overshoot `0.0`.
  - Failed6 with the same setting had useful `1.0`, overshoot `0.0`, cap
    `0.0`, mean/max XY `0.002727/0.007170m`, but only `4/6` envs reached
    `>=1mm`; ep991/ep29 stayed tiny.
  - Recorded-target repair improved offline metrics but collapsed runtime
    failed6 displacement.

## Code Changes

`roarm_rl/roarm_cube_push_env.py`

- Added default-off env runtime contract fields:
  - `tap_action_governor_mode`
  - `tap_action_governor_target_disp_m`
  - `tap_action_governor_predict_horizon_s`
  - `tap_action_governor_speed_stop_mps`
  - `tap_action_governor_min_contact_steps`
  - `tap_action_governor_push_scale`
  - `tap_action_governor_brake_scale`
  - `tap_action_governor_brake_steps`
- Added `_apply_tap_action_governor_if_enabled()` in the joint-delta action
  path before smoothing/delta application.
- Added governor buffers, reset handling, and reward extras:
  - stop latch rate
  - brake active rate
  - projected displacement
  - contact age
- Modes:
  - `off`: default, no action changes.
  - `predict_stop`: after contact context, zero actions if current
    displacement, `disp + speed * horizon`, or speed exceeds the threshold.
  - `predict_brake`: same stop latch plus short opposite-action brake.

`sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`

- Added `--env_action_governor_mode off|predict_stop|predict_brake`.
- Kept local `--action_governor_mode` mutually exclusive with env governor.
- Wired `action_governor_*` parameters into env config for the env governor.
- Added per-step and per-env env-governor diagnostics.
- Added current-step contact diagnostics so latched fields are not confused
  with current geometry:
  - `tap_contact_proxy_now`
  - `tap_reaction_now`
  - `tap_overshoot_now`
  - final per-env current-proxy equivalents.

## Fresh32 Runtime Diagnostics

Actor checkpoint:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`

Common runtime contract:

- `--reset_pose_source env_hook`
- `--d256_reset_sample_mode random`
- `--num_envs 32`
- `--seed 30701`
- `--steps 580`
- `--hold_steps 3`
- `--action_smoothing_alpha 0.25`
- `--max_joint_delta_per_step_rad 0.04`
- `--contact_joint_delta_scale 0.35`
- `--fast_cube_joint_delta_scale 0.2`
- `--joint_delta_reference joint_pos`
- `--tap_contact_proxy_mode link5_collision_aabb`
- `--tap_stop_after_disp_m 0.003`
- `--no-tap_stop_after_useful_seen`
- `--action_governor_predict_horizon_s 0.020`
- `--action_governor_speed_stop_mps 0.200`
- `--action_governor_target_disp_m 0.003`

### Predict Stop

Output:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_env_contract_d308/tap10cm/fresh32_random_predict_stop_h020_v200_proxy_now/`

Headline metrics:

- Useful: `0.8125`
- Overshoot: `0.1875`
- Env governor stop latch: `0.78125`
- Max XY mean/max: `0.006647647358477116 / 0.03392736241221428m`
- XY `>=1mm`: `25/32`
- XY `>=3mm`: `8/32`
- XY `>=20mm`: `6/32`
- Final current contact proxy: `19/32`
- Low displacement `<1mm`: `7/32`

Failure split:

- Overshoot envs: `3,4,9,24,27,31`.
- Env `3,4,27,31` were already over `20mm` at step `0`.
- Env `9,24` were near threshold at step `0` (`19.614mm`, `19.516mm`),
  latched stop at step `1`, then crossed `20mm` at step `2`.
- Tiny-displacement envs `<1mm`: `6,11,17,18,19,20,23`.
  - Env `11,20,23` ended current proxy false with face gap about
    `-0.086/-0.081/-0.085m`.
  - Env `6,19` ended current proxy true but never projected to the `3mm`
    target.
  - Several tiny envs kept latched contact/useful despite final current-proxy
    false, so latched useful is not sufficient as a deployment signal.

### Predict Brake

Output:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_env_contract_d308/tap10cm/fresh32_random_predict_brake_h020_v200_proxy_now/`

Headline metrics:

- Useful: `0.8125`
- Overshoot: `0.1875`
- Env governor stop latch: `0.78125`
- Max XY mean/max: `0.006538099609315395 / 0.03394607454538345m`
- XY `>=1mm`: `25/32`
- XY `>=3mm`: `8/32`
- XY `>=20mm`: `6/32`

Comparison to `predict_stop`:

- Same useful/overshoot/displacement-count result.
- Same overshoot env set.
- Same low-displacement env set.
- The short opposite-action brake is not enough for first-step impulse or
  near-threshold inertia overshoot in this actor/control contract.

## Interpretation

`env_action_governor_mode=predict_stop` means the env runtime contract, not only
the diagnostic script, zeroes actions after contact context when the state is
already near or beyond the allowed displacement/velocity envelope. It is a late
stop safety layer. It does not choose the push direction, does not guarantee
contact quality, and does not predict first-step displacement from the actor's
joint-delta command.

D308 confirms two separate blockers:

1. First-step impulse / inertia overshoot:
   The governor only sees the pre-step state. If the actor command creates
   `20mm+` displacement in one env step, `predict_stop` can latch only after the
   damage is already visible. `predict_brake` is also too weak/late here.

2. Contact-geometry / action-direction failure:
   Some envs report latched contact/useful but end with no current contact
   proxy and almost zero displacement. A latch is not an adequate control
   contract. The policy/action space still has to put the tool on the correct
   face and push in a bounded object-moving direction.

## Verdict

`D308_ENV_GOVERNOR_FRESH32_FAIL_PUSH_PRIMITIVE_NEXT_NO_PPO`

No PPO promotion. No tiny PPO trace gate. No learned policy. No RoArm readiness.

## Next Concrete Step

Stop treating scalar joint-delta stop/brake tuning as the main repair path.
Implement a non-PPO tool/object push primitive or equivalent action-space
contract with explicit:

- approach/contact face selection,
- current contact validation,
- push direction in object frame,
- bounded push displacement/velocity,
- stop/recovery handling when current contact proxy is false.

Then run a fresh multi-episode diagnostic measuring:

- contact/useful,
- overshoot <=5%,
- low cap,
- `>=1mm` displacement rate,
- current proxy at final/contact windows, not only latched contact.

## Verification

- `python -m py_compile roarm_rl/roarm_cube_push_env.py sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`: PASS.
- `git diff --check`: PASS.
- Residual process check for Isaac/PPO/TensorBoard/torchrun/rl_games: no
  matching process output.
- Final GPU query:
  `NVIDIA GeForce RTX 4090 Laptop GPU, 0, 24, 16376`.
- Final compute-app query: no output.
