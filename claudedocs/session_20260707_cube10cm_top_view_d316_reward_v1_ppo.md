# 2026-07-07 - Cube10cm Top-View D316 Reward-v1 PPO

## Scope

- Branch: professor 10cm / 0.72kg cube top-view visual trajectory dataset work.
- Purpose: analyze the D315 overshoot-heavy primitive-residual PPO failure, repair the tap reward wiring, and run failure-capable PPO follow-up on the D314 low-friction target.
- Exclusions: no B200/JHPark SSH, no pull, no `.ssh` copy, no RoArm deployment, no Track A, no VLA/SmolVLA fine-tuning, no raw joint-delta revival.

## Repo Current Truth Checked First

- `START_HERE.md` before this update said D315 was the latest current truth: D314 completed the 9-row perturbation matrix and D315 started real `candidate8_diffik_target_residual` PPO, but the 5-iteration run over-pushed.
- `claudedocs/DECISIONS.md` D314 records friction as the first axis that broke the baseline: low friction strict useful `10/32`, contact/reaction `11/32`, overshoot `1/32`; high friction strict useful `0/32`, overshoot `32/32`.
- `claudedocs/DECISIONS.md` D315 records the first real learnable primitive-residual PPO on low friction: contact/reaction `64/64`, useful `32/64`, overshoot `32/64`, XY `>=20mm` `32/64`.

## Reward Audit

The D315 overshoot was not only an early-learning issue. The tap reward contract was also under-specified:

- `roarm_rl/roarm_cube_push_env.py` had tap-specific reward scales, including `tap_overshoot_penalty_scale`, but `roarm_rl/train_cube_push_ppo.py` only exposed generic `--overshoot_penalty_scale`.
- Strict useful was computed and logged, but was not directly rewarded.
- Overshoot penalty was based on current target excess; there was no latched overshoot-seen penalty.

Default-off reward knobs were added:

- `tap_overshoot_seen_penalty_scale`
- `tap_strict_useful_reward_scale`
- `tap_strict_useful_seen_reward_scale`
- `tap_control_band_reward_scale`
- `tap_target_excess_quadratic_penalty_scale`

The train CLI now exposes the tap-specific reward scales explicitly. Defaults remain unchanged unless the flags are set.

## Low-Friction Proxy Audit

Existing D314 low-friction CSV showed a mixed proxy/control failure:

- XY `>=1mm`: `32/32`
- XY `>=7mm`: `30/32`
- contact/reaction: `11/32`
- strict useful: `10/32`

So many low-friction episodes physically moved the cube but did not satisfy the contact proxy. In no-contact rows, the final face gap was often about `-12..-26mm`, outside the `+/-10mm` contact band, consistent with fast/deep low-friction interactions that bypass the proxy condition. This should not be treated as pure controller failure without a proxy-contract check.

## High-Friction Audit

The D314 high-friction row remains suspect as a hard eval target:

- mean/max XY: `3768.838/11989.522mm`
- overshoot: `32/32`
- step traces show speed reaching `10m/s` at step 0 and meter-scale displacement within early steps.

This looks like solver/runaway artifact or a severe contact-dynamics instability, not a normal push target. It should be rendered or trace-audited before being used as the first PPO objective.

## D316 PPO Runs

Both runs used local GPU only, `rl_action_mode="candidate8_diffik_target_residual"`, friction `0.8/0.6`, action space `3`, `64` envs, `64` steps/env, and no preserve-blend actor shortcut.

Reward v1 flags:

- `tap_contact_reward_scale=0.25`
- `tap_contact_proximity_reward_scale=0.2`
- `tap_reaction_reward_scale=2.0`
- `tap_transient_disp_reward_scale=80.0`
- `tap_overshoot_penalty_scale=24.0`
- `tap_overshoot_seen_penalty_scale=2.0`
- `tap_target_excess_quadratic_penalty_scale=1.0`
- `tap_strict_useful_reward_scale=8.0`
- `tap_strict_useful_seen_reward_scale=0.25`
- `tap_control_band_reward_scale=0.5`

### Short Run: 30 Iterations

Run:

- `d316_candidate8_friction_low_reward_v1_30it`
- summary:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d316/tap10cm/d316_candidate8_friction_low_reward_v1_30it/d316_candidate8_friction_low_reward_v1_30it_summary.json`

Final collection:

- contact/reaction: `64/64`
- useful: `47/64`
- overshoot: `1/64`
- XY `>=1mm`: `48/64`
- XY `>=20mm`: `1/64`
- low motion `<1mm`: `16/64`
- max XY mean/max: `6.372/62.538mm`
- joint cap max: `0.0`

Interpretation:

- Compared with D315, reward v1 strongly reduced overshoot: `32/64 -> 1/64`.
- It also improved useful: `32/64 -> 47/64`.
- But it introduced a conservative low-motion mode: `16/64` stayed below `1mm`.
- This is useful evidence, not promotion.

### Warm-Start Run: 300 More Iterations

Run:

- `d316_candidate8_friction_low_reward_v1_warm300`
- warm start:
  `d316_candidate8_friction_low_reward_v1_30it/model_29.pt`
- summary:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d316/tap10cm/d316_candidate8_friction_low_reward_v1_warm300/d316_candidate8_friction_low_reward_v1_warm300_summary.json`

Final collection:

- contact/reaction: `64/64`
- useful: `39/64`
- overshoot: `23/64`
- XY `>=1mm`: `62/64`
- XY `>=20mm`: `23/64`
- low motion `<1mm`: `2/64`
- max XY mean/max: `15.139/54.408mm`
- joint cap max: `0.0`

Interpretation:

- Longer PPO did not monotonically improve the reward-v1 policy.
- It reduced low-motion cases from `16/64` to `2/64`, but reintroduced overshoot from `1/64` to `23/64`.
- The failure mode shifted from under-push to over-push. Reward v1 is unstable under longer training.

## Decision

- D316 is a valid failure-capable PPO session under the progress rule.
- Reward-v1 is better than D315 at short horizon, but not stable enough for promotion.
- Do not return to raw scalar joint deltas.
- Do not patch another hand controller before learning analysis; the next work should be PPO reward/control-shaping, with the standard baseline-vs-policy table.
- Low friction remains the valid learning target, but the contact proxy should be audited when physical displacement and proxy contact disagree.
- High friction should be treated as a suspect hard-eval row until a render or trace confirms it is not a solver artifact.

## Verdict

`D316_REWARD_V1_SHORT_IMPROVES_LONGER_UNSTABLE_NO_PROMOTION`

