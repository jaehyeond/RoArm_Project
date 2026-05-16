# Session 2026-05-17 — P7 release-guidance diagnostics under identity+keep attach

## Scope

- Continued mechanics-first path after env-level attach semantics were proven active.
- Did not claim P7 success.
- Did not add scripted release variants.
- Did not random-search SurfaceGripper parent/offset.
- Did not revert existing dirty worktree.

## Starting Evidence

P7v4 identity+keep fixed the immediate no-upright mechanics problem only partially, then failed by no-release:

- `/tmp/p7v4_attach_identity_keep_model19_trace.out` line 44: `attach_quat_mode=identity attach_velocity_mode=keep`.
- line 96: reset `d_xy=0.1722`, `sz=1.0000`, attached.
- lines 338-340: no open/release in 60 steps (`0/256`).
- line 355: final `d_xy=0.1488`, `sz=0.9036`.

Hypothesis: with identity+keep, P7 reward/controller has too little local signal to open near a plausible release corridor.

## Code Change

Added gated P7 release-guidance diagnostics. Defaults keep P7v3/P7v4 reward unchanged:

- `roarm_rl/roarm_stack_env.py`
  - `p7_release_guidance: bool = False`
  - `p7_release_guidance_xy_thresh: float = 0.120`
  - `p7_release_guidance_z_thresh: float = 0.040`
  - `p7_release_open_bonus_scale: float = 4.0`
  - `p7_release_closed_penalty_scale: float = 4.0`
  - `p7_transport_xy_penalty_scale: float = 4.0`
- `roarm_rl/train_ppo.py`
  - `--p7_release_guidance`
  - `--p7_release_guidance_xy_thresh`
  - `--p7_release_guidance_z_thresh`

Post-change local/B200 md5s:

- `roarm_rl/roarm_stack_env.py` = `580e137a2318586a7a848664a1f2d7c1`
- `roarm_rl/train_ppo.py` = `ffecfb0b0df89c69159dabe3dd5046e7`
- `sim_scripts/p7_action_tcp_quat_trace.py` = `e6c9424cfe7ffafdf00fe0625f0553f7`
- `sim_scripts/p7_attach_semantics_env_probe.py` = `4997a3ec058773004441b74419da114f`

## P7v5: identity+keep, release guidance xy 0.12

B200:

- `/tmp/p7v5_identity_keep_release_guidance_diag20.{out,err}`
- `/tmp/p7v5_identity_keep_release_guidance_model19_trace.{out,err}`

Training key lines:

- line 44: `attach_quat_mode: preserve -> identity`
- line 45: `attach_velocity_mode: zero -> keep`
- line 46: `p7_release_guidance: True`
- line 106: iter 0 `p7_xy_offset_mean=0.1761`
- line 111: iter 0 `p7_gripper_open_rate=0.2713`
- line 116: iter 0 `p7_upright_rate=0.9354`
- line 603: iter 16 `p7_xy_offset_mean=0.1908`
- line 608: iter 16 `p7_gripper_open_rate=0.5190`
- line 613: iter 16 `p7_upright_rate=0.5172`
- line 614: iter 16 `p7_place_success_rate=0.0017`

Trace key lines:

- line 42: checkpoint `model_19.pt`
- line 44: `identity keep`
- lines 239-241: open/release `256/256`
- lines 242-245: pre-open attached tip almost gone (`first_tip_while_grasped=1/256`)
- line 248: mean open/release step `11.96`
- line 255: release `sz=0.9935`, but `d_xy=0.1522`
- line 256: final `d_xy=0.1260`, `sz=0.4126`

Interpretation: release guidance broke the no-release failure, but the policy opens too far from target and final object orientation collapses after release.

## P7v6: identity+keep, release guidance xy 0.08

B200:

- `/tmp/p7v6_identity_keep_release_guidance_xy08_diag20.{out,err}`
- `/tmp/p7v6_identity_keep_release_guidance_xy08_model19_trace.{out,err}`

Training key lines:

- line 47: `p7_release_guidance_xy_thresh: 0.12 -> 0.08`
- line 107: iter 0 `p7_xy_offset_mean=0.1777`
- line 112: iter 0 `p7_gripper_open_rate=0.4355`
- line 117: iter 0 `p7_upright_rate=0.8766`
- line 604: iter 16 `p7_xy_offset_mean=0.1711`
- line 609: iter 16 `p7_gripper_open_rate=0.5227`
- line 614: iter 16 `p7_upright_rate=0.4332`
- line 615: iter 16 `p7_place_success_rate=0.0023`

Trace key lines:

- line 44: `identity keep`
- line 96: reset `d_xy=0.1722`, `sz=1.0000`, attached.
- lines 338-340: open/release `256/256`
- lines 341-344: attached tip before open worsened to `118/256`
- lines 347-351: mean open/release step `31.89`; mean first tip any `28.18`
- line 354: release `sz=0.9575`, `d_xy=0.0849`
- line 355: final `d_xy=0.1055`, `sz=0.2840`

Interpretation: tightening the release corridor improved release XY but reintroduced attached/pre-release tipping and still ended flat. Continuing threshold-only tuning is not a sound path.

## Verdict

Release guidance answers one question: P7v4's no-release behavior was not inevitable under identity+keep. A local open signal can produce release.

But the branch is still a primitive FAIL:

- xy 0.12: opens too early/far, then final orientation collapses.
- xy 0.08: releases closer, but attached tip returns and final orientation still collapses.

Next should not be blind scalar threshold tuning. The remaining failure is release dynamics/post-release orientation stability under an attached kinematic carrier. Useful next branches:

1. Design a structured release controller/curriculum with explicit near-target transport, then a low-motion vertical release/settle phase under identity+keep.
2. Move to Branch B and author a physics gripper/constraint unit test that reaches stable `Closed` before chain integration.
