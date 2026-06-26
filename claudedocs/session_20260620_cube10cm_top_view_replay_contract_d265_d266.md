# 2026-06-20 Cube10cm Top-View Replay Contract D265-D266

## Scope

- Branch: professor 10cm / 0.72kg cube top-view visual trajectory dataset path.
- No PPO learning, no rendering, no RoArm deployment, no RunPod, no B200/SSH.
- Goal: after D263/D264 showed D256 pose reset improves support but still misses
  contact, separate action timing mismatch from current contact-proxy/geometry
  mismatch.

## D265 Direct D256 Action Replay Timing Sweep

Probe:

- Script: `sim_scripts/cube10cm_top_view_d256_action_replay_probe.py`
- Env: `RoArm-CubeTap10cm-Direct-v0`
- Teacher used: `False`
- PPO learning: `False`
- Reset: D256 frame-0 joint/cube/target/push-dir state
- Action replay: D256 `state + joint_delta`
- Tested `hold_steps`: `1`, `2`, `4`, `5`
- D264 `hold_steps=3` is the reference point.

Results:

| hold_steps | contact_rate | min TCP-cube dist min (m) | max disp along mean (m) |
|---:|---:|---:|---:|
| 1 | `0.0` | `0.06175459548830986` | `0.006831126753240824` |
| 2 | `0.0` | `0.061787448823451996` | `0.006762760225683451` |
| 3 | `0.0` | `0.06179572641849518` | `0.006767723709344864` |
| 4 | `0.0` | `0.06307728588581085` | `0.005708895158022642` |
| 5 | `0.0` | `0.0657862201333046` | `0.0033459109254181385` |

Interpretation:

- Simple frame-cadence/action-hold tuning does not recover contact.
- Best observed minimum distance is still about `6.75mm` outside the current
  contact threshold `tcp_cube_dist < 0.055m`.
- Slower replay reduces movement and moves farther from contact.

## D266 Recorded-State Sequence Probe

Added:

- `sim_scripts/cube10cm_top_view_d256_state_sequence_probe.py`

Probe:

- Env: `RoArm-CubeTap10cm-Direct-v0`
- Teacher used: `False`
- PPO learning: `False`
- Action replay: `False`
- Method: write D256 recorded arm joint state and cube pose into the live env at
  each frame, then measure current env `_push_terms()` contact proxy.

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_state_sequence_probe_d266/tap10cm/d256_state_sequence_summary_d266.json`

Result:

- contact rate: `0.0`
- first contact step min: `-1`
- contact threshold: `0.055`
- min TCP-cube distance mean/min/max:
  `0.07699309289455414` / `0.06270913034677505` /
  `0.1001250371336937`
- max disp along mean/min/max:
  `0.0074400329031050205` / `0.0034220367670059204` /
  `0.018024206161499023`

Interpretation:

- This is stronger than the D264 action-replay failure.
- Even recorded D256 states do not satisfy the current env contact proxy.
- The current blocker is therefore not just D257 MLP generalization or replay
  timing. It is the contract between the visual-label definition of useful tap
  and the runtime metric `tcp_cube_dist < 0.055m`, plus the tool-surface/contact
  geometry used by the current env.

## Current Verdict

`D266_D256_RECORDED_STATE_SEQUENCE_STILL_NO_CONTACT_CONTACT_PROXY_CONTRACT_BLOCKER_NO_PPO`

Do not run PPO from the current state. The next work is contact-proxy contract
diagnosis:

- Compare D256 visual label/contact definition against current `_push_terms()`
  `tcp_cube_dist` threshold.
- Check whether current TCP point should be replaced for this gate by the
  established tool-surface proxy from D231/D224.
- Measure dataset frames using the same tool-surface or AABB distance proxy that
  the env uses for reward/contact.
- Only after recorded D256 states satisfy the chosen contact proxy should
  teacher-only contact, tiny PPO smoke, TensorBoard gate, and teacher-off eval be
  reconsidered.

## Verification

- `python3 -m py_compile` passed for the new and modified scripts.
- `git diff --check` passed.
- No Python/Isaac/PPO probe process remained after runs.
- GPU returned to observed baseline around `2509MiB` used / `13436MiB` free.
