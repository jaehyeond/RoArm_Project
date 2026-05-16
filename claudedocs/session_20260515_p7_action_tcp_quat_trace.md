# Session 2026-05-15 — P7 action/TCP/quaternion step trace

## Scope

User direction:

- Diagnose before changing P7 reward.
- Do not alter `chain_skills.py`, reward scalars, SurfaceGripper parent/offset,
  scripted release variants, or `_update_grasp_attach`.
- Verify file md5s and B200 log lines before claiming metrics.
- Answer when `sz_world_z` collapses relative to `_grasped` and gripper open.

## Boot / Verification

- Read `CLAUDE.md` and followed the Current-State Protocol.
- Read `START_HERE.md`.
- Read `claudedocs/DECISIONS.md` D014-D017.
- Read `claudedocs/EXPERIMENT_LEDGER.md` rows:
  - 2026-05-15 `(G2-A v10)`
  - 2026-05-15 `(G2-A v11)`
  - 2026-05-15 `(SurfaceGripper probe v2/v3)`
  - 2026-05-15 `(P7 G2-A attached transport/release)`
  - 2026-05-15 `(P7 model_499 rollout failure diag)`
- Read:
  - `claudedocs/session_20260515_p7_rollout_failure_diag.md`
  - `claudedocs/session_20260515_p7_attached_transport_learning.md`
  - `claudedocs/session_20260515_g2a_layout_source_sweep.md`
  - `claudedocs/session_20260515_g2a_scripted_release_bridge.md`
- `git status --short` was dirty before coding; existing dirty worktree was not
  reverted.

## Pre-Code md5 Verification

All requested local md5s matched:

- `roarm_rl/chain_skills.py` = `c6e610216197994c6b7d2b6625d87560`
- `launch_chain_topdown.sh` = `b34ef3853ac993a1e2adbaddb420adab`
- `roarm_rl/roarm_stack_env.py` = `996f2afce7de1b3be93ae43ddc349f8e`
- `roarm_rl/train_ppo.py` = `6b0ffdb8365c5e37ced00833c0556c19`
- `launch_p6v17_transport_release.sh` =
  `2acd462042d0997610fca25ff7a41e21`
- `sim_scripts/attached_transport_reset_probe.py` =
  `43a04e3cfca763a50d8c856185d14b99`
- `sim_scripts/surface_gripper_transport_probe.py` =
  `053fced6551ccb02d8a9ea6c04fb4a30`
- `sim_scripts/p7_rollout_failure_diag.py` =
  `a9743d74886c454b1c161a1bade3df93`

## Prior B200 Log Verification

The requested B200 logs existed on B200 `/tmp`:

- `/tmp/p7v3_rollout_failure_diag.{out,err}`
- `/tmp/p7v3_transport_release.{out,err}`
- `/tmp/p7v1_attached_reset_probe_v2.{out,err}`
- `/tmp/p7v3_diag20.{out,err}`
- `/tmp/roarm_surface_gripper_transport_probe_v2.{out,err}`
- `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.{out,err}`

Rechecked key stdout lines:

- `/tmp/p7v3_rollout_failure_diag.out` line 93:
  `completed_episodes=512`
- line 95: `C_tips_during_attached_transport: 512 (1.000)`
- line 97: reset `d_xy=0.1732`, `sz_world_z=1.0000`
- line 98: pre-release `sz_world_z=0.2667`
- line 99: release `d_xy=0.0739`, `release_z_offset=0.0788`,
  `sz_world_z=0.2851`
- line 101: final `settled_z_offset=0.0006`, `sz_world_z=0.0156`
- `/tmp/p7v3_transport_release.out` lines 14984-14994: iter 496 had
  `p7_xy_offset_mean=0.0512`, `p7_on_target_rate=0.0005`,
  `p7_upright_rate=0.0576`, `p7_place_success_rate=0.0007`
- `/tmp/p7v1_attached_reset_probe_v2.out` lines 65-68:
  `_grasped=1.000`, `_was_grasped=1.000`, sponge-TCP `0.00mm`,
  initial mean `d_xy=175.80mm`
- `/tmp/roarm_surface_gripper_transport_probe_v2.out` lines 143/152/164 and
  `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.out` lines
  144/153/165: both quick SurfaceGripper probes failed to close/attach.

## `_update_grasp_attach` Semantics

Direct source check:

- `roarm_rl/roarm_stack_env.py` lines 1096-1110:
  - selects `_grasped` envs;
  - computes TCP from `link5` pose and `_tcp_local`;
  - writes sponge position to TCP via `pose7[:, 0:3] = tcp_pos`;
  - preserves current sponge quaternion via
    `pose7[:, 3:7] = self._sponge.data.root_quat_w[env_ids]`;
  - calls `write_root_pose_to_sim`;
  - writes zero root velocity every attached step.

Interpretation: the attach model pins position but does not restore upright
orientation. Once the sponge quaternion tips during physics, later attached
steps preserve the tipped quaternion.

## Code Change

Added `sim_scripts/p7_action_tcp_quat_trace.py`.

Design:

- Headless/state-only eval script.
- Loads
  `$ROARM_B200_ROOT/logs/roarm_rl/roarm_stack_p7v3_g2a_attached_transport_release/model_499.pt`.
- Starts from exact P7 attached-start curriculum
  (`curriculum_attached_start_jitter_rad=0.0`).
- Logs per-step sample traces for selected envs:
  action vector, gripper action, gripper joint/open flag, TCP local position,
  TCP delta and finite-difference velocity, sponge local position, quaternion,
  quaternion delta, `sz_world_z`, `_grasped`, `_was_grasped`, `d_xy`,
  release-entry z offsets, sponge-TCP distance, and rigid-object linear/angular
  velocity norms.
- Aggregates first transition steps across all envs:
  first open, first `_grasped=False`, first tip, first tip while grasped, first
  large TCP jump, release pose means, and final pose means.

Post-change md5:

- `sim_scripts/p7_action_tcp_quat_trace.py` =
  `c54b7892dd06a72f31402ab8dc011b65`

Local check:

- `python -m py_compile sim_scripts/p7_action_tcp_quat_trace.py` passed.

B200 synced script md5 matched:

- `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/p7_action_tcp_quat_trace.py`
  = `c54b7892dd06a72f31402ab8dc011b65`

## B200 Smoke

Run:

- `/tmp/p7v3_action_tcp_quat_trace_smoke.out`
- `/tmp/p7v3_action_tcp_quat_trace_smoke.err`

Command used B200 Isaac env + `OMNI_KIT_ACCEPT_EULA=YES`, with
`--num_envs 16 --trace_envs 4 --max_steps 60`.

First two attempts failed before the script logic:

- one without Isaac Lab on `PYTHONPATH`;
- one without `OMNI_KIT_ACCEPT_EULA=YES`.

Final smoke completed. Key lines:

- line 42: script header.
- line 44: `num_envs=16 max_steps=60 trace_envs=4`.
- lines 237-245: all 16 envs opened/released and tipped; no large TCP jump over
  `0.030m`.
- lines 247-251: mean first open/release `20.31`, mean first tip while grasped
  `1.81`.

stderr had NVML/observation warnings only, no Python traceback.

## B200 Diagnostic Run

Run:

- `/tmp/p7v3_action_tcp_quat_trace.out`
- `/tmp/p7v3_action_tcp_quat_trace.err`

Command:

```bash
python -u sim_scripts/p7_action_tcp_quat_trace.py \
  --checkpoint "$ROARM_B200_ROOT/logs/roarm_rl/roarm_stack_p7v3_g2a_attached_transport_release/model_499.pt" \
  --num_envs 256 \
  --trace_envs 4 \
  --max_steps 60
```

Log md5:

- out = `3ecff0bf0c7a0358c053108c8f9dd504`
- err = `44b3d53f8f234b11c033f6ada4568b67`

Key stdout lines:

- line 43: checkpoint path verified.
- line 44: `num_envs=256 max_steps=60 trace_envs=4 seed=0`.
- lines 93-97: `max_episode_length=200`, `step_dt=0.0100`,
  `grasp_gripper_thresh=0.4000rad`, `gripper_joint_idx=5`, and attach
  semantics reminder.
- line 99: reset mean `d_xy=0.1722`, `sz=1.0000`,
  `d_sponge_tcp=0.00000`, `grasped=1.000`.
- lines 100-103: by step 1, three of four sampled envs already have
  `sz_world_z<0.9` while `grasped=1` and `open=0`; q deltas are large
  (`0.28010`, `0.31719`, `0.31719`) and angular velocity norms are high.
- lines 104-115: sampled envs continue tipping while `_grasped=True`; env 001
  reaches `sz=0.0221` by step 4, still `open=0`, `grasped=1`.
- line 152: env 000 first opens/releases at step 14 with `sz=0.3265`, already
  tipped.
- line 169: env 001 opens/releases at step 18 with `sz=-0.0702`, already
  tipped.
- line 244: early stop at step 36 after all envs opened/released plus five
  steps.
- lines 245-253: transition counts:
  `first_open=256/256`, `first_grasp_false=256/256`,
  `release_or_open=256/256`, `first_tip_any=256/256`,
  `first_tip_while_grasped=256/256`,
  `tip_before_or_at_open=256/256`,
  `tip_while_grasped_before_or_at_release=256/256`,
  `first_large_tcp_jump>0.030m=0/256`.
- lines 254-260: mean transition steps:
  first open/release `20.21`; first tip while grasped `1.72`; no large TCP jump.
- lines 261-264: release mean `sz=0.2983`, `d_xy=0.0714`,
  `rel_z_abs=0.0764`; final mean `d_xy=0.0238`,
  `settled_z_abs=0.0201`, but `sz=0.0759`;
  `max_tcp_delta_mean=0.0191`, `max_tcp_delta_max=0.0246`,
  `max_abs_action_mean=1.0000`.
- lines 266-269: sample transition rows confirm env 000/001/003 tip at step 1,
  release later, and final `sz=0.0000`; env 002 tips at step 10 and stays only
  marginally upright (`final_sz=0.8918`).

stderr lines 1-12 were NVML/cpufreq/rsl_rl observation warnings only; no Python
traceback.

## Analysis Questions

1. Does `sz_world_z` collapse while `_grasped=True`, before release?
   Yes. B200 line 250 shows `first_tip_while_grasped=256/256`; line 259 shows
   mean first tip while grasped at step `1.72`, while line 257 shows mean
   release/open at step `20.21`.

2. Is collapse correlated with a specific gripper action/opening step?
   No as the primary cause. The collapse precedes open/release in all envs:
   line 251 `tip_before_or_at_open=256/256`; line 252
   `tip_while_grasped_before_or_at_release=256/256`. Sample lines 100-103 show
   `open=0`, `grasped=1` while tipping has already begun.

3. Is collapse correlated with large TCP jumps or target-chasing oscillation?
   Not with a single >3cm step jump: line 253 has
   `first_large_tcp_jump>0.030m=0/256`, and line 264 has
   `max_tcp_delta_max=0.0246`. However actions are saturated
   (`max_abs_action_mean=1.0000`, line 264), and sampled TCP velocities are high
   during the first few steps. This suggests aggressive attached motion/contact
   dynamics, not a one-frame teleport.

4. Does `_update_grasp_attach` preserve a stale tipped quaternion once tipping
   begins?
   Yes by source semantics: `roarm_stack_env.py` line 1107 writes the current
   sponge quaternion back into the attached pose. The trace shows q changes
   during physics while attached, then later attached steps continue from that
   tipped quaternion.

5. Does zeroing sponge velocity while writing position create or hide a
   nonphysical state?
   It likely hides/reset-starts velocity at each attach application, but does
   not prevent physics/contact from generating angular velocity within the step.
   Sample line 100 has `ang_vel_norm=31.8279` and line 101 has `48.6703` while
   still attached. So zeroing velocity is not an upright constraint; it may make
   the state even less physically interpretable by erasing velocity history while
   preserving tilted pose.

6. Is final low settled z just a lying-flat artifact?
   Yes. The step trace reproduces the earlier failure: line 263 has final
   `settled_z_abs=0.0201` with final `sz=0.0759`, and the previous rollout
   diagnostic line 101 had even lower final settled z with `sz_world_z=0.0156`.

7. Which is the next valid fix branch?
   Best next branch: attach quaternion reset/constraint diagnostic first. The
   failure happens immediately under the current kinematic attach semantics, so
   a focused diagnostic that constrains/resets attached orientation can test
   whether P7 transport becomes mechanically meaningful before changing reward.
   In parallel or next, an authored physics gripper/constraint unit test remains
   valid. An orientation-aware learned branch should wait until the attach
   semantics are not silently preserving a tipped quaternion.

## Verdict

The dominant failure is earlier than the previous diagnostic could prove:
`model_499.pt` tips the sponge almost immediately during attached transport,
while `_grasped=True` and before the gripper opens. Release occurs later after
the object is already tipped.

This strengthens the no-reward-hack rule: the next useful experiment is not a
P7 scalar tweak, but an attach/orientation semantics diagnostic or a proper
physics gripper/constraint unit test.
