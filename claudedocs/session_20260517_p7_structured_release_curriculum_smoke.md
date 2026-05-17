# Session 2026-05-17 — P7 structured release curriculum smoke

## Scope

- Continued Branch A once, but as a falsifiable structured curriculum, not scalar
  or xy-threshold tuning.
- Did not run long training after the smoke failed the predeclared kill metric.
- Did not add scripted chain release variants.
- Did not random-search SurfaceGripper parent/offset.
- Did not revert existing worktree state.

## Pre-Code Verification

Boot followed `CLAUDE.md` Current-State Protocol and the user-requested list.

Local md5s before coding matched the requested baseline:

- `roarm_rl/chain_skills.py` = `c6e610216197994c6b7d2b6625d87560`
- `launch_chain_topdown.sh` = `b34ef3853ac993a1e2adbaddb420adab`
- `roarm_rl/roarm_stack_env.py` = `580e137a2318586a7a848664a1f2d7c1`
- `roarm_rl/train_ppo.py` = `ffecfb0b0df89c69159dabe3dd5046e7`
- `launch_p6v17_transport_release.sh` = `2acd462042d0997610fca25ff7a41e21`
- `sim_scripts/p7_action_tcp_quat_trace.py` = `e6c9424cfe7ffafdf00fe0625f0553f7`
- `sim_scripts/p7_attach_semantics_env_probe.py` = `4997a3ec058773004441b74419da114f`
- `sim_scripts/p7_attach_quat_constraint_probe.py` = `a2e16f7683856ead1a9a9eef1da8ea69`
- `sim_scripts/p7_rollout_failure_diag.py` = `a9743d74886c454b1c161a1bade3df93`

`git status --short` was clean before coding.

Requested B200 `/tmp` logs existed on B200. Key rechecked lines:

- `/tmp/p7_attach_semantics_identity_keep.out` lines 64-66:
  `identity+keep` reset a tipped attached sponge upright and kept velocity.
- `/tmp/p7_attach_semantics_preserve_zero.out` lines 64-66:
  default `preserve+zero` preserved the tipped orientation and zeroed velocity.
- `/tmp/p7v4_attach_identity_keep_model19_trace.out` lines 338-355:
  identity+keep fresh policy failed by no release/open.
- `/tmp/p7v5_identity_keep_release_guidance_model19_trace.out` lines 239-256:
  xy `0.12` release guidance released all envs but far and final flat.
- `/tmp/p7v6_identity_keep_release_guidance_xy08_model19_trace.out` lines
  338-355: xy `0.08` released closer but reintroduced attached tip and final flat.

## Code Change

Added a default-off structured P7 release curriculum:

- `roarm_rl/roarm_stack_env.py`
  - `p7_structured_release_curriculum: bool = False`
  - structured near-target release-entry joint reset, derived from IK for
    target + 29mm release entry.
  - explicit structured gate:
    near target, near release height, low velocity, upright.
  - anti-early-open penalty outside the gate.
  - released/settle reward that requires near target and upright.
  - default attach semantics remain `preserve+zero`; default P7 remains unchanged.
- `roarm_rl/train_ppo.py`
  - `--p7_structured_release_curriculum`
  - `--p7_structured_release_xy_jitter`
  - `--p7_structured_release_z_jitter`
- `sim_scripts/p7_structured_release_curriculum_probe.py`
  - policy-free B200 smoke.
  - starts from structured reset under identity+keep.
  - holds arm still, opens only the gripper, then settles.
  - prints `MECHANISM_ACTIVE` and `EARLY_KILL`.

Post-change local/B200 md5s:

- `roarm_rl/roarm_stack_env.py` = `e2748144034d5a09d6c7a0f6c0da6906`
- `roarm_rl/train_ppo.py` = `795ee48b1bfdd83e8c9735efd01f6920`
- `sim_scripts/p7_structured_release_curriculum_probe.py` =
  `41e6b48bfaa46b82f2add262903a2a5e`

Local check:

- `python -m py_compile roarm_rl/roarm_stack_env.py roarm_rl/train_ppo.py sim_scripts/p7_structured_release_curriculum_probe.py` passed.

## B200 Smoke

Run:

- `/tmp/p7v7_structured_release_smoke.out`
- `/tmp/p7v7_structured_release_smoke.err`
- out md5 `ba270b1e88b5dd1797eeea712608ea56`
- err md5 `db74058c8b0feb8aac61f1a87b853f53`

Command:

```bash
python -u sim_scripts/p7_structured_release_curriculum_probe.py \
  --num_envs 64 \
  --hold_steps 5 \
  --open_steps 12 \
  --settle_steps 80 \
  --xy_jitter 0.0 \
  --z_jitter 0.0
```

Key stdout lines:

- line 68: `attach_quat_mode=identity attach_velocity_mode=keep
  structured_release=True`.
- line 69: reset was exact near-target/upright/attached:
  `d_xy=0.0000`, `rel_z_abs=0.0000`, `sz=1.0000`,
  `grasped=1.000`, `open=0.000`.
- line 70: after five hold steps, still attached/closed and mostly upright:
  `d_xy=0.0070`, `rel_z_abs=0.0045`, `sz=0.9601`,
  `open=0.000`, `grasped=1.000`.
- lines 74-76: open/release occurred in all envs and no attached tip before release:
  `first_open=64/64`, `release_or_open=64/64`,
  `tip_while_grasped_before_release=0/64`.
- line 78: release itself was close and upright:
  `release sz=0.9720`, `d_xy=0.0089`.
- line 79: post-release settle failed upright badly:
  `final d_xy=0.0411`, `settled_z_abs=0.0041`, `sz=0.2484`,
  `success_rate=0.2344`.
- lines 80-81: `MECHANISM_ACTIVE=YES`, `EARLY_KILL=YES`.

stderr lines 1-8 were NVML/cpufreq warnings only; no Python traceback.

## Verdict

Mechanism gate PASS, Branch A smoke FAIL.

The structured reset and identity+keep path were active, and the low-motion
manual open did remove the P7v4 no-release failure without attached pre-release
tipping. But the object still tipped after release/settle from the target +29mm
entry. This fails the predeclared early-kill metric (`final sz < 0.90` and
`success_rate < 0.90`).

Do not launch long PPO on this A configuration. A learned policy should not be
asked to solve a release/settle contact problem that fails under a perfect
near-target, arm-still, hand-authored smoke. The next stronger branch is Branch B:
properly author and validate a physics gripper/constraint unit test that reaches
stable `Closed` before chain integration.
