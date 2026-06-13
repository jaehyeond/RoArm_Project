# 2026-06-11 - Useful Tap Objective Reframe

## Scope

- Branch: professor 10cm / 0.72kg cube tap RL robustness/action-space branch.
- No Track A, dataset, VLA, action-teacher, RoArm deployment, SSH/B200, pull,
  or `.ssh` work.
- No GPU runtime and no PPO training in this session note.

## User Objective Update

The user clarified that the task does not need to keep the cube displacement in
the 6mm target band. The practical goal is to make a useful tap: contact the
cube and produce a physical reaction without causing an overshoot failure.

This changes the promotion logic:

- Primary useful-tap metric: contact seen + physical reaction seen + no
  overshoot seen.
- 6mm target-band and target-error remain quality-tier diagnostics.
- The current env `tap_success` flag still includes the 6mm target band, so it
  must not be used as the D229 useful-tap metric without checking the new log
  fields.

## Existing Evidence Reinterpreted

D225/D228 found pose regions that are weak for 6mm target-band quality. They do
not yet prove useful-tap base failure:

- D225 `(x=0.14,y=0.15)` base weak bin had contact/reaction `1.0`, overshoot
  `0.0`, and target-band `0.0`.
- D228 x-band base had contact/reaction `1.0`, overshoot `0.0`, and low
  target-band.

Under the D229 definition, those are useful-tap base-pass bins and only
target-quality failures.

Existing broad randomization sweeps still motivate a new search because
overshoot rises and some contact/reaction rates fall as pose perturbation grows.
That evidence is coarse; the next useful-tap claim needs pose-binned discovery.

## Code Change

Logging only:

- `roarm_rl/roarm_cube_push_env.py`
  - added `cube_tap_useful_now_rate`
  - added `cube_tap_useful_seen_rate`
  - added `cube_tap_contact_reaction_seen_rate`
  - added `cube_tap_no_overshoot_seen_rate`
- `roarm_rl/train_cube_tap10cm_ppo_smoke.py`
  - summary now reports `useful_seen_max`
  - summary now reports `contact_reaction_seen_max`
  - summary now reports `no_overshoot_seen_min`

No reward, action-space, controller, reset, geometry, termination, or PPO
behavior was changed.

## Verification

Passed:

```bash
python3 -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_tap10cm_ppo_smoke.py
git diff --check
```

## Decision

The statement "base fails in useful-tap pose regions and RL recovers useful tap"
is the right research claim shape, but it is not proven yet.

Before any PPO recovery run, find a pose-binned region where base actually fails
the D229 useful-tap metric through missing contact/reaction or overshoot. Only
then run a small residual PPO L1 against same-run base and simple constant
residual baselines.

Do not run L2/Large PPO from D225/D228 target-band weak bins as if they were
useful-tap failures.
