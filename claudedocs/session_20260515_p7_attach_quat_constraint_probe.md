# Session 2026-05-15 — P7 attach quaternion constraint diagnostic

## Scope

User direction:

- Continue step-by-step from the P7 action/TCP/quaternion trace.
- Keep critical/analytical posture.
- Do not change reward first.
- Do not alter `chain_skills.py`.
- Do not add scripted release variants.
- Do not random-search SurfaceGripper parent/offset.
- Do not modify `_update_grasp_attach` in the repo yet; diagnose attach
  quaternion reset/constraint behavior first.

## Baseline

Previous B200 trace:

- `/tmp/p7v3_action_tcp_quat_trace.out` line 99: reset was upright/attached:
  `d_xy=0.1722`, `sz=1.0000`, `d_sponge_tcp=0.00000`,
  `grasped=1.000`.
- Lines 245-253: baseline had `first_tip_while_grasped=256/256`,
  `tip_before_or_at_open=256/256`, and no >3cm TCP jump.
- Lines 254-260: mean first tip while grasped step `1.72`; mean open/release
  step `20.21`.
- Lines 261-264: release `sz=0.2983`, final `sz=0.0759`, so low final z/XY
  remained a lying-flat artifact.

Source semantics:

- `roarm_rl/roarm_stack_env.py` lines 1096-1110: `_update_grasp_attach`
  writes sponge xyz to TCP, preserves current sponge quaternion at line 1107,
  and zeroes root velocity at line 1110.

## Code Change

Added `sim_scripts/p7_attach_quat_constraint_probe.py`.

Design:

- Headless/state-only diagnostic.
- Loads P7v3 `model_499.pt`.
- Starts from the exact attached-start curriculum.
- Monkey-patches `_update_grasp_attach` at runtime only. It does not edit
  `roarm_stack_env.py`.
- Modes:
  - `quat_mode=preserve|identity|reset`
  - `velocity_mode=zero|keep`
- Logs the same transition family as the action/TCP/quaternion trace:
  first open, first `_grasped=False`, first tip, first tip while grasped,
  before-open/before-release tip counts, large TCP jumps, release/final pose,
  and max angular velocity while grasped.

Post-change md5:

- `sim_scripts/p7_attach_quat_constraint_probe.py` =
  `a2e16f7683856ead1a9a9eef1da8ea69`

Local check:

- `python -m py_compile sim_scripts/p7_attach_quat_constraint_probe.py` passed.

B200 synced script md5 matched:

- `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/p7_attach_quat_constraint_probe.py`
  = `a2e16f7683856ead1a9a9eef1da8ea69`

## B200 Runs

### Smoke: identity + zero

Run:

- `/tmp/p7v3_attach_quat_identity_smoke.out`
- `/tmp/p7v3_attach_quat_identity_smoke.err`

Result:

- line 45: `quat_mode=identity velocity_mode=zero`
- line 99: reset `d_xy=0.1725`, `sz=1.0000`,
  `d_sponge_tcp=0.00000`, `grasped=1.000`
- lines 141-149: `first_tip_while_grasped=13/16`,
  `tip_before_or_at_open=9/16`, one >3cm TCP jump.
- lines 151-160: mean first tip while grasped `20.46`; release `sz=0.9715`;
  final `sz=0.8634`, `d_xy=0.2294`.

Interpretation: identity constraint substantially delays tip and improves
upright release, but does not solve transport/placement. It also exposes high
angular velocity while grasped.

### Full: identity + zero

Run:

- `/tmp/p7v3_attach_quat_identity_zero.out`
- `/tmp/p7v3_attach_quat_identity_zero.err`
- out md5 `d65b875c8db71b3b2b053cd73a497acc`
- err md5 `9e38ef4a6669d5ff0bfc1f945e9ac28f`

Key stdout lines:

- line 45: `quat_mode=identity velocity_mode=zero`
- line 97: runtime patch applied.
- line 99: reset `d_xy=0.1722`, `sz=1.0000`,
  `d_sponge_tcp=0.00000`, `grasped=1.000`
- lines 100-103: sampled step 1 states stayed upright-ish
  (`sz=0.9504`, `0.9799`, `0.9960`, `0.9799`) while `open=0`,
  `grasped=1`, unlike the baseline action/TCP trace.
- lines 141-149: `first_tip_any=189/256`,
  `first_tip_while_grasped=189/256`, `tip_before_or_at_open=128/256`,
  `tip_while_grasped_before_or_at_release=128/256`,
  `first_large_tcp_jump>0.030m=6/256`.
- lines 151-160: mean first open/release `22.39`; mean first tip while grasped
  `19.74`; release `sz=0.9664`, `d_xy=0.1547`, `rel_z_abs=0.1736`;
  final `d_xy=0.2487`, `settled_z_abs=0.0613`, `sz=0.9113`.

Interpretation: identity quaternion + velocity zeroing converts the immediate
attached tip into delayed/partial tip and gives upright release/final orientation
on average, but the object is far from target and release height is wrong. This
is not a solved primitive.

### Full: preserve + keep

Run:

- `/tmp/p7v3_attach_quat_preserve_keep.out`
- `/tmp/p7v3_attach_quat_preserve_keep.err`
- out md5 `e0fd44724863bc1df82198c7d66fc4f8`
- err md5 `50c6eac92d761eb3ac34240fa0d1028e`

Key stdout lines:

- line 45: `quat_mode=preserve velocity_mode=keep`
- line 97: runtime patch applied.
- line 99: reset `d_xy=0.1722`, `sz=1.0000`,
  `d_sponge_tcp=0.00000`, `grasped=1.000`
- lines 100-103: step 1 already tips in sampled envs with `open=0`,
  `grasped=1`, e.g. `sz=0.8738`, `0.8178`, `0.9980`, `0.8178`.
- lines 141-149: `first_tip_while_grasped=256/256`,
  `tip_before_or_at_open=256/256`, no >3cm TCP jump.
- lines 151-160: mean first tip while grasped `1.67`; mean open/release
  `20.25`; release `sz=0.1561`; final `sz=0.0101`.

Interpretation: removing velocity zeroing while preserving current quaternion
does not solve the immediate collapse. Velocity zeroing is not the primary
cause of the observed baseline failure.

### Full: identity + keep

Run:

- `/tmp/p7v3_attach_quat_identity_keep.out`
- `/tmp/p7v3_attach_quat_identity_keep.err`
- out md5 `5061457e4b74784fbf36fb610eedc008`
- err md5 `b9886b9e437235af70823180bf5f66f0`

Key stdout lines:

- line 45: `quat_mode=identity velocity_mode=keep`
- line 97: runtime patch applied.
- line 99: reset `d_xy=0.1722`, `sz=1.0000`,
  `d_sponge_tcp=0.00000`, `grasped=1.000`
- lines 100-103: sampled step 1 remains upright-ish
  (`sz=0.9686`, `0.9883`, `0.9966`, `0.9883`) while `open=0`,
  `grasped=1`.
- lines 141-149: `first_tip_any=148/256`,
  `first_tip_while_grasped=77/256`, `tip_before_or_at_open=11/256`,
  `tip_while_grasped_before_or_at_release=11/256`,
  `first_large_tcp_jump>0.030m=17/256`.
- lines 151-160: mean first open/release `33.12`; mean first tip while grasped
  `27.70`; release `sz=0.9921`, `d_xy=0.1558`, `rel_z_abs=0.0359`;
  final `d_xy=0.2604`, `settled_z_abs=0.0219`, `sz=0.6434`.

Interpretation: identity quaternion without velocity zeroing best suppresses
pre-release attached tipping (`11/256` before release versus baseline
`256/256`), but final orientation still degrades after release and target
transport is poor. This suggests the old quaternion preservation was a major
failure amplifier, while the learned policy itself is not a valid transport/
release controller under the altered semantics.

stderr for all runs contained NVML/cpufreq/rsl_rl observation warnings only; no
Python traceback.

## Comparative Interpretation

| Mode | tip while grasped | tip before open | release sz | final sz | final d_xy |
|---|---:|---:|---:|---:|---:|
| baseline preserve+zero (`p7_action_tcp_quat_trace`) | 256/256 | 256/256 | 0.2983 | 0.0759 | 0.0238 |
| preserve+keep | 256/256 | 256/256 | 0.1561 | 0.0101 | 0.0802 |
| identity+zero | 189/256 | 128/256 | 0.9664 | 0.9113 | 0.2487 |
| identity+keep | 77/256 | 11/256 | 0.9921 | 0.6434 | 0.2604 |

Critical reading:

- The stale/tipped quaternion preservation is a primary failure amplifier. When
  attached quaternion is forced upright, immediate pre-release collapse mostly
  disappears.
- Velocity zeroing alone is not the primary cause. `preserve+keep` still fails
  essentially like baseline.
- Identity quaternion is not a solution. It improves upright metrics but exposes
  bad transport: release/final XY is around 25cm in the identity runs, worse
  than the flat-object baseline XY. The old low XY was partly a lying-flat/path
  artifact.
- The current P7 policy was trained under broken attach semantics. Once
  quaternion is constrained, the same policy does not know how to place the
  object near target.
- High angular velocities remain visible in identity modes, especially while
  attached. This keeps the physics-constraint branch relevant.

## Verdict

Attach quaternion constraint diagnostic PASS as a diagnosis, FAIL as a primitive.

The next valid branch is not reward scalar hacking. It is either:

1. Implement a real env-level attached-orientation semantic change as a new
   controlled experiment, then retrain/evaluate P7 from scratch under that
   semantic; or
2. Build the authored physics gripper/constraint unit test and require it to
   reach a stable attached/closed state before chain integration.

Because identity constraint changes the mechanics substantially and the old
policy does not transport correctly under it, do not claim the attach reset
diagnostic solved P7.
