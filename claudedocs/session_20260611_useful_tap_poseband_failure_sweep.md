# 2026-06-11 Useful-Tap Pose-Band Failure Sweep

## Scope

- Branch: professor 10cm / 0.72kg cube tap RL robustness/action-space branch.
- Objective: useful tap = contact/reaction without overshoot.
- 6mm target-band remains a quality-tier diagnostic, not the primary success
  claim for this branch.
- No PPO, L2, Large PPO, dataset, VLA, action-teacher, RoArm, SSH/B200, pull,
  or Track A work was performed.

## Code Changes

- `roarm_rl/train_cube_tap10cm_ppo_smoke.py`
  - Added `useful_seen_final`, `contact_reaction_seen_final`,
    `no_overshoot_seen_final` to smoke summaries.
  - Added `--per_env_summary_json` for per-env pose/overshoot diagnostics.
  - Added `--disable_tap_overshoot_terminate` for no-reset diagnostic rollouts.
- `roarm_rl/roarm_cube_push_env.py`
  - Fixed `_get_dones()` to respect `cfg.tap_overshoot_terminate`.
  - Default remains terminate-on-overshoot; the new flag is diagnostic-only.
- `sim_scripts/cube10cm_tap_useful_posebin_sweep.py`
  - Added as a helper, but sequential Isaac env recreation hung after one bin.
    It is not used as durable evidence for D230.

Static checks passed:

```bash
python3 -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_tap10cm_ppo_smoke.py sim_scripts/cube10cm_tap_useful_posebin_sweep.py
git diff --check
```

## Metric Correction

Legacy `--tap_success_terminate` is target-band based. It can reset envs and
make `*_final` useful metrics describe a partial post-reset episode instead of
the original useful-tap attempt.

Example:

- `(x=0.34,y=-0.10)`, seed1033, with `--tap_success_terminate`:
  `useful_seen_final=0.3125`, `done_count=32`.
- Same pose, seed1034, without legacy success termination:
  useful/contact-reaction/no-overshoot final `1.0`, overshoot `0.0`.

Therefore useful-tap discovery should use no legacy success termination. For
pose localization, disable overshoot termination too and read per-env
`overshoot_seen`.

## Runtime Evidence

Fixed xy10 corners, no legacy success termination:

| Pose | Useful final | Contact/reaction final | Overshoot |
| --- | ---: | ---: | ---: |
| `(0.14,-0.10)` | `1.0` | `1.0` | `0.0` |
| `(0.34,-0.10)` | `1.0` | `1.0` | `0.0` |
| `(0.14,+0.10)` | `1.0` | `1.0` | `0.0` |
| `(0.34,+0.10)` | `1.0` | `1.0` | `0.0` |

Random xy10 no-termination per-env diagnostics:

| Seed | n | Useful final | Contact/reaction final | No-overshoot final | Overshoot |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1036` | `64` | `0.828125` | `1.0` | `0.828125` | `0.171875` |
| `1037` | `64` | `0.828125` | `1.0` | `0.828125` | `0.171875` |

Combined result:

- `22/128` overshoot.
- `106/128` clean useful-tap.
- Contact/reaction was preserved at `1.0`.
- `done_count=0` in both no-termination runs after the termination fix.

Per-env spatial bin counts, combined seeds `1036` and `1037`, x bins
`[0.14,0.19,0.24,0.29,0.34]`, y bins `[-0.10,-0.05,0,0.05,0.10]`:

| y bin | x .14-.19 | x .19-.24 | x .24-.29 | x .29-.34 |
| --- | ---: | ---: | ---: | ---: |
| `+0.05..+0.10` | `2/8` | `0/11` | `1/6` | `1/10` |
| `+0.00..+0.05` | `4/11` | `0/3` | `1/8` | `0/7` |
| `-0.05..+0.00` | `1/5` | `3/14` | `2/9` | `1/8` |
| `-0.10..-0.05` | `0/3` | `2/10` | `2/6` | `2/9` |

Representative random overshoot sample:

- Random per-env seed1036 env47: `(x=0.197617,y=-0.085433)`,
  `overshoot_seen=True`.
- Exact fixed replay of `(0.197617,-0.085433)`, seed1037 n16, no terminations:
  useful/contact-reaction/no-overshoot final `1.0`, overshoot `0.0`.

This means the failure is not a durable fixed-pose contact miss. It is a
randomized-band / trajectory-conditioning overshoot failure.

## Interpretation

- Base useful-tap failure now exists under xy +/-10cm randomization.
- The failure is overshoot after contact/reaction, not missing contact/reaction.
- Fixed corners are clean and should not be used as PPO weak bins.
- D225/D228 remain target-band quality failures, not useful-tap failures.

Research claim wording:

> In a pose perturbation band where the scripted base over-taps some cases,
> residual RL should recover clean useful tap by reducing overshoot while
> preserving contact/reaction.

This is a valid next experimental claim, not an achieved result.

## Next Gate

Before PPO:

1. Run same xy10 randomized-band constant residual baselines.
2. Use useful/no-overshoot as the primary metric.
3. If constants already solve overshoot, the RL claim is weak.

Small PPO L1 is valid only after constants:

- Same xy10 band.
- Same-run base comparison.
- Must beat same-run base and best constant on useful/no-overshoot.
- Must preserve contact/reaction `1.0`.
- Must keep finite obs/reward/action, IK/numeric OK, contract clean.

Still blocked:

- L2/Large PPO.
- Dataset generation.
- VLA/action-teacher.
- RoArm deployment.
- SSH/B200/pull/`.ssh`.
- Track A mixing.

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md` D230
- `roarm_rl/roarm_cube_push_env.py`
- `roarm_rl/train_cube_tap10cm_ppo_smoke.py`
- `sim_scripts/cube10cm_tap_useful_posebin_sweep.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_useful_random_xy10_no_terminations_seed1036_n64_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_useful_random_xy10_no_terminations_seed1036_n64_per_env.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_useful_random_xy10_no_terminations_seed1037_n64_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_useful_random_xy10_no_terminations_seed1037_n64_per_env.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_useful_posebin_x0197617_yneg0085433_no_terminations_seed1037_n16_summary.out`
