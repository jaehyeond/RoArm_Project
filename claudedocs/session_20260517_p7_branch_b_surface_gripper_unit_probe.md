# Session 2026-05-17 — P7 Branch B SurfaceGripper unit probe

## Scope

- Continued Track A P7/Branch B only. CoRL paper track remains separate.
- Did not claim P7 success.
- Did not tune P7 scalar/threshold/release guidance.
- Did not long-train the structured A curriculum after
  `/tmp/p7v7_structured_release_smoke.out` reported `EARLY_KILL=YES`.
- Did not random-search RoArm SurfaceGripper parent/offset settings.
- Did not add scripted release variants.
- Did not revert the existing dirty worktree.

## Boot Verification

Followed `CLAUDE.md` Current-State Protocol and the user-requested boot list.

Pre-code local md5s matched the requested baseline:

- `roarm_rl/chain_skills.py` = `c6e610216197994c6b7d2b6625d87560`
- `launch_chain_topdown.sh` = `b34ef3853ac993a1e2adbaddb420adab`
- `roarm_rl/roarm_stack_env.py` = `e2748144034d5a09d6c7a0f6c0da6906`
- `roarm_rl/train_ppo.py` = `795ee48b1bfdd83e8c9735efd01f6920`
- `launch_p6v17_transport_release.sh` = `2acd462042d0997610fca25ff7a41e21`
- `sim_scripts/p7_action_tcp_quat_trace.py` = `e6c9424cfe7ffafdf00fe0625f0553f7`
- `sim_scripts/p7_attach_semantics_env_probe.py` = `4997a3ec058773004441b74419da114f`
- `sim_scripts/p7_attach_quat_constraint_probe.py` = `a2e16f7683856ead1a9a9eef1da8ea69`
- `sim_scripts/p7_rollout_failure_diag.py` = `a9743d74886c454b1c161a1bade3df93`
- `sim_scripts/p7_structured_release_curriculum_probe.py` = `41e6b48bfaa46b82f2add262903a2a5e`

`git status --short` was already dirty before coding:

- `M START_HERE.md`
- `M claudedocs/DECISIONS.md`
- `M claudedocs/EXPERIMENT_LEDGER.md`
- `M roarm_rl/roarm_stack_env.py`
- `M roarm_rl/train_ppo.py`
- untracked recent CoRL / structured-release docs and structured-release probe

Requested B200 logs existed on B200 `/tmp`. Key rechecked lines:

- `/tmp/p7v7_structured_release_smoke.out` lines 68-81: mechanism active, exact
  near-target reset, all envs opened/released, no attached tip before release,
  but final `sz=0.2484`, `success_rate=0.2344`, `EARLY_KILL=YES`.
- `/tmp/p7_attach_semantics_identity_keep.out` lines 64-66: `identity+keep`
  attach semantics active and reset a tipped attached sponge to `sz_mean=1.0000`.
- `/tmp/p7_attach_semantics_preserve_zero.out` lines 64-66: default
  `preserve+zero` preserved tipped `sz_mean=0.5000` and zeroed velocity.
- `/tmp/p7v4_attach_identity_keep_model19_trace.out` lines 338-355:
  no release/open (`0/256`), final `d_xy=0.1488`, `sz=0.9036`.
- `/tmp/p7v5_identity_keep_release_guidance_model19_trace.out` lines 239-256:
  release guidance xy `0.12` opened/released all envs but released far and
  ended flat.
- `/tmp/p7v6_identity_keep_release_guidance_xy08_model19_trace.out` lines
  338-355: xy `0.08` released closer but reintroduced attached tip and ended
  flat.

## Pre-Code Answers

1. Mechanism: a CPU-only SurfaceGripper unit rig using Isaac Lab's canonical
   `Tests/SurfaceGripper/test_gripper.usd` SurfaceGripper asset, with the project
   RoArm sponge as the rigid object. This avoids another RoArm parent/offset guess.
2. Early-kill metric: `closed_detect_step == -1`, or hold `closed_frac < 0.95`,
   or sponge drift beyond `0.020m`.
3. Required B200 proof: stdout must show `SURFACE_UNIT_SUCCESS=YES`, stable
   `state=+1.0`, `closed_frac>=0.95`, and low sponge drift before any transport.
4. Files: add only `sim_scripts/p7_branch_b_surface_gripper_unit_probe.py`; old
   env/train/chain/launcher behavior remains untouched.
5. Docs: update this session doc, `START_HERE.md`, `EXPERIMENT_LEDGER.md`, and
   append `DECISIONS.md` only if a durable lesson changes.

## Code Change

Added `sim_scripts/p7_branch_b_surface_gripper_unit_probe.py`.

Design:

- Launches Isaac headless on CPU because Isaac Lab SurfaceGripper is CPU-only.
- Spawns Isaac Lab's canonical SurfaceGripper test USD:
  `Tests/SurfaceGripper/test_gripper.usd`.
- Spawns the normal RoArm project sponge via `RoArmStackEnvCfg().sponge`.
- Places the sponge at the canonical object pose `(0, 0, 0.5)`.
- Commands close, then holds before any transport.
- Reports:
  - `closed_detect_step`
  - hold `closed_frac`
  - `gripped_positive_frac` from `get_gripped_objects` when available
  - sponge drift and velocity
  - `SURFACE_UNIT_SUCCESS`

The script intentionally does not modify:

- `roarm_rl/roarm_stack_env.py`
- `roarm_rl/train_ppo.py`
- `roarm_rl/chain_skills.py`
- any launcher default
- any reward scalar or release-guidance threshold

Post-change md5:

- `sim_scripts/p7_branch_b_surface_gripper_unit_probe.py` =
  `1d093ebbd39d2c64252545574e74ad34`

Local check:

- `python -m py_compile sim_scripts/p7_branch_b_surface_gripper_unit_probe.py`
  passed.

B200 synced script md5:

- `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/p7_branch_b_surface_gripper_unit_probe.py`
  = `1d093ebbd39d2c64252545574e74ad34`

## B200 Smoke

Run:

- `/tmp/p7_branch_b_surface_gripper_unit_smoke.out`
- `/tmp/p7_branch_b_surface_gripper_unit_smoke.err`
- out md5 `00741a38640a5b181c4b47ded870a6a3`
- err md5 `169cef9d20f7cc732693a88607a16068`

Command:

```bash
python -u sim_scripts/p7_branch_b_surface_gripper_unit_probe.py \
  --close_steps 80 \
  --hold_steps 120 \
  --max_grip_distance 0.120
```

Key stdout lines:

- line 42: script header.
- line 43: `device=cpu chain_integration=NO transport=NO`.
- line 89: canonical SurfaceGripper asset and RoArm sponge prim verified.
- line 90: reset placed the sponge at `z=0.4986`, with gripper state open
  (`state=-1.0`).
- lines 91-103: close command never reached `Closed`; state remained `0.0`
  or `-1.0`.
- line 121: aggregate failure:
  `closed_detect_step=-1`, `closed_frac=0.0000`,
  `gripped_positive_frac=1.0000`, `max_drift=0.37595`.
- line 122: final state still not closed (`state=+0.0`), sponge fell/drifted to
  `z=0.1235`.
- line 123: `SURFACE_UNIT_SUCCESS=NO`.

stderr lines 1-13 were NVML/cpufreq warnings only; no Python traceback in the
final smoke.

## Interpretation

This is a valid early-kill for the first Branch B SurfaceGripper+sponge
hypothesis:

- The unit harness ran before chain integration.
- It did not depend on RoArm `link5` / `gripper_link` parent guesses.
- It still did not reach `Closed` on the RoArm sponge.

Critical caveat:

- `gripped_count` was positive even while `state` was open/closing and the sponge
  drifted. Therefore `get_gripped_objects` is not sufficient evidence of a
  stable attach in this setup. The `Closed` state and drift/hold metrics must be
  treated as the gate.

## Verdict

Diagnostic harness PASS / canonical SurfaceGripper+sponge hypothesis FAIL.

Do not chain-integrate SurfaceGripper yet. Do not go back to random RoArm
parent/offset variants. The next Branch B step should be one of:

1. A single controlled SurfaceGripper axis/object-size diagnostic using the
   canonical rig, comparing the RoArm sponge against the canonical cuboid only to
   isolate whether failure is sponge geometry/material/scale or gripper axis.
2. Switch to an explicitly authored fixed/D6 constraint unit test with a clear
   close/release API, then test stable attached hold before transport.
