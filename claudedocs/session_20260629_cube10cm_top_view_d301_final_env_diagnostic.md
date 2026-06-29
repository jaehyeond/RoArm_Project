# D301 - Cube10cm final-env non-PPO diagnostic

Date: 2026-06-29 KST

## Purpose

Diagnose the D300 final-state gate failures without running more PPO training.
The goal was to inspect failed final envs by episode index, action magnitude,
contact proxy, displacement, and overshoot.

## Code Change

- `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
  - Extended `--out_env_csv` with final per-env diagnostic columns:
    contact proxy, reaction/target/overshoot/success now flags, final
    displacement, face gap, lateral/vertical offset, TCP distance, speed, tip
    angle, final action magnitude, stop-after-displacement hold, and
    `failure_reason`.
  - Added `--out_env_step_csv` to write per-env step traces for non-PPO
    diagnostics.

## Runtime

Ran non-PPO frozen-checkpoint diagnostics only:

- action mode: `ppo_stochastic`
- training: no
- checkpoint source: D300 `model_0.pt`
- `num_envs=32`
- `eval_steps=580`
- D256 random frame-0 reset active
- `link5_collision_aabb`
- `tap_stop_after_disp_m=0.003`
- `tap_success_terminate=False`
- `bc_teacher_blend=0.0`

Initial sandbox execution failed before env creation because Isaac/PhysX could
not create a CUDA context. `nvidia-smi` showed the GPU was healthy, so the
approved Isaac Lab commands were rerun outside the sandbox.

## Results

### Seed 29801

- final contact/reaction/useful/success: `1.0/1.0/1.0/1.0`
- overshoot: `0.0`
- XY `>=1mm`: `0.53125`
- mean/max XY: `0.0020347752142697573/0.007077273912727833m`
- RSL-like all-step useful mean: `0.8638469827586207`
- verdict: fail only because all-step RSL-like useful mean is below `0.90`

This does not reproduce D300 seed `29801` collection-final failure
(`0.8125` useful, `0.03125` overshoot). That means the saved checkpoint alone
is not enough to recover the exact D300 failed final envs. The collection RNG
and reset sequence in `runner.learn(...)` still matter.

### Seed 29604

- final contact/reaction/useful/success: `0.84375/0.84375/0.84375/0.84375`
- overshoot: `0.0`
- XY `>=1mm`: `0.5`
- mean/max XY: `0.0021026856265962124/0.013731294311583042m`
- failed envs: `5/32`
- failure reasons:
  - `success`: `27`
  - `no_contact_seen`: `5`

Failed envs:

| env | D256 episode | reason | final face gap m | final TCP dist m | max XY m | action mean trace | action max trace |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 221 | no_contact_seen | -0.0154010765 | 0.1054280326 | 0.0000117707 | 0.4893094301 | 1.0 |
| 10 | 198 | no_contact_seen | -0.0263072960 | 0.1124396101 | 0.0000114466 | 0.5202108622 | 1.0 |
| 24 | 13 | no_contact_seen | -0.0380339138 | 0.1243612841 | 0.0000117311 | 0.4958575070 | 1.0 |
| 25 | 322 | no_contact_seen | -0.0131158344 | 0.1000752524 | 0.0000114466 | 0.5189525485 | 1.0 |
| 31 | 935 | no_contact_seen | -0.3355142474 | 0.4397282600 | 0.0000118498 | 0.4382028878 | 1.0 |

Per-step trace showed:

- failed envs had zero contact steps and zero useful steps;
- initial face gap was already near the contact band but outside it:
  `-0.0121556..-0.0172336m`;
- contact band is `±0.010m`, so most failures were only `2..7mm` outside the
  face band at the start;
- over time the policy action magnitude increased, but the face gap moved more
  negative instead of entering the band;
- joint delta cap stayed `0.0`, so this is not a joint-cap saturation failure;
- displacement stayed around `0.011mm`, so the cube was effectively stationary.

## Reset-State Pattern

The seed `29604` failed envs share a hard reset region:

- `cube_local_x_m`: `0.315..0.340`, mean `0.327`
- `cube_local_y_m`: edge values including `-0.100` and `0.131`
- `arm_joint_2_rad`: `2.0468..2.1532`, low relative to successful env mean
  `2.2669`
- `arm_joint_3_rad`: `0.2222..0.2846`, low relative to successful env mean
  `0.4484`

Interpretation: the remaining blocker is a hard reset/state coverage issue in
far cube / low elbow-wrist posture states. The actor often starts just outside
the AABB contact band and then moves away instead of closing the final few mm.

## Critical Notes

- This is not evidence for learned-policy success.
- Do not lower the contact band just to make these envs pass; that would hide
  an actor coverage problem.
- Useful/success can still be true with tiny displacement because reaction can
  come from non-displacement signals and the current target band is permissive.
  Therefore displacement-rate gates remain necessary.
- D301 cannot exactly recover D300 seed `29801` failed final envs from the saved
  checkpoint alone. Before any future PPO gate, collection-time per-env final
  trace should be written directly from `train_cube_push_ppo.py`.

## Decision

- No long PPO.
- No PPO ladder.
- No partial actor preservation or real actor update yet.
- Next work should be non-PPO hard-bin repair:
  - isolate D256 reset episodes with far cube / low joint-2 and joint-3 states;
  - run targeted actor-vs-teacher/action-direction diagnostics on that bin;
  - add either hard-bin supervised warm-start data or a pre-contact action
    projection/approach constraint;
  - only after the hard bin passes teacher-off/non-PPO diagnostics should a
    tiny TensorBoard gate be rerun.

## Verdict

`D301_FINAL_ENV_DIAGNOSTIC_EDGE_RESET_NO_CONTACT_NO_PPO`

## Artifacts

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/final_env_diagnostic_d301/seed29801/`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/final_env_diagnostic_d301/seed29604/`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/final_env_diagnostic_d301/seed29604_trace/`
