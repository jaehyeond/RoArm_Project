# START_HERE.md

Last updated: 2026-05-17 KST (post-P7 env-level attach semantics experiment)

This is the rolling project dashboard. It is overwritten as the project moves.
Do not use it as the full experiment history. Durable lessons live in
`claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

## Current Truth

Latest verified state:

- `claudedocs/session_20260517_p7_attach_semantics_env_experiment.md`
  - Chose Branch A over B for the next mechanics-first step: controlled
    env-level attach semantics are cheaper to validate now because D019 showed
    `identity+keep` suppresses pre-release attached tip, while SurfaceGripper
    v2/v3 never reached `Closed`.
  - Added gated config in `roarm_rl/roarm_stack_env.py`: `attach_quat_mode`
    default `preserve`, `attach_velocity_mode` default `zero`; defaults keep old
    behavior unchanged unless explicitly enabled.
  - Added `sim_scripts/p7_attach_semantics_env_probe.py` and extended
    `sim_scripts/p7_action_tcp_quat_trace.py` with env-level attach flags.
  - B200 smoke proved the gate:
    `/tmp/p7_attach_semantics_identity_keep.out` line 64 enabled
    `identity+keep`; line 66 reset a tipped attached sponge to
    `sz_mean=1.0000` and kept velocity (`vel_norm_mean=3.0020`).
    `/tmp/p7_attach_semantics_preserve_zero.out` line 64 used default
    `preserve+zero`; line 66 preserved the tipped `sz_mean=0.5000` and zeroed
    velocity.
  - Fresh short P7v4 identity+keep training is **not solved**:
    `/tmp/p7v4_attach_identity_keep_diag20.out` lines 44-45 confirm enabled
    semantics; `p7_xy_offset_mean` worsened from `0.1904` at line 105 to
    `0.3620` at line 586, and `p7_place_success_rate=0.0000` at line 596.
  - Fresh checkpoint trace is also **not solved**:
    `/tmp/p7v4_attach_identity_keep_model19_trace.out` lines 338-340 show
    no open/release (`0/256`), line 355 final `d_xy=0.1488`, `sz=0.9036`.
  - Verdict: env-level mechanics gate PASS, P7 primitive FAIL. Identity+keep
    improves upright mechanics but needs controller/reward/curriculum redesign,
    or switch to authored physics gripper/constraint unit test.

- `claudedocs/session_20260515_p7_attach_quat_constraint_probe.md`
  - Added `sim_scripts/p7_attach_quat_constraint_probe.py` md5
    `a2e16f7683856ead1a9a9eef1da8ea69`.
  - It monkey-patches `_update_grasp_attach` at runtime only; no env/reward/
    chain/asset source semantics were changed.
  - B200 runs:
    `/tmp/p7v3_attach_quat_identity_zero.{out,err}`,
    `/tmp/p7v3_attach_quat_preserve_keep.{out,err}`,
    `/tmp/p7v3_attach_quat_identity_keep.{out,err}`.
  - `preserve+keep` still failed like baseline:
    `first_tip_while_grasped=256/256`, `tip_before_or_at_open=256/256`,
    release `sz=0.1561`, final `sz=0.0101`; so velocity zeroing alone is not
    the primary cause.
  - `identity+zero` improved upright release/final orientation but not
    transport: `tip_before_or_at_open=128/256`, release `sz=0.9664`, final
    `sz=0.9113`, final `d_xy=0.2487`.
  - `identity+keep` best suppressed pre-release attached tip:
    `tip_before_or_at_open=11/256`, release `sz=0.9921`, but final
    `sz=0.6434` and final `d_xy=0.2604`.
  - Verdict: stale quaternion preservation is a major failure amplifier, but a
    simple upright attach constraint does not solve P7 with the old policy.

- `claudedocs/session_20260515_p7_action_tcp_quat_trace.md`
  - Boot followed `CLAUDE.md` Current-State Protocol.
  - Pre-code md5 matched requested baseline:
    `chain_skills.py=c6e610216197994c6b7d2b6625d87560`,
    launcher `b34ef3853ac993a1e2adbaddb420adab`,
    stack env `996f2afce7de1b3be93ae43ddc349f8e`,
    `train_ppo.py=6b0ffdb8365c5e37ced00833c0556c19`,
    P7 launcher `2acd462042d0997610fca25ff7a41e21`,
    reset probe `43a04e3cfca763a50d8c856185d14b99`,
    SurfaceGripper probe `053fced6551ccb02d8a9ea6c04fb4a30`,
    rollout diag `a9743d74886c454b1c161a1bade3df93`.
  - Added `sim_scripts/p7_action_tcp_quat_trace.py` md5
    `c54b7892dd06a72f31402ab8dc011b65`.
  - Confirmed `_update_grasp_attach` lines 1096-1110: writes sponge xyz to TCP,
    preserves current sponge quat at line 1107, zeroes velocity at line 1110.
  - B200 run: `/tmp/p7v3_action_tcp_quat_trace.{out,err}`.
  - Result: reset mean was upright/attached (`d_xy=0.1722`, `sz=1.0000`,
    `d_sponge_tcp=0.00000`, `grasped=1.000`), but `sz_world_z` collapsed while
    still attached and before open/release in all envs:
    `first_tip_while_grasped=256/256`, `tip_before_or_at_open=256/256`,
    mean first tip step `1.72`, mean open/release step `20.21`.
  - No one-step TCP jump >3cm (`0/256`, max `0.0246m`), but actions were
    saturated (`max_abs_action_mean=1.0000`) and early sampled angular velocity
    was high. Final low z/XY remained a lying-flat artifact (`final d_xy=0.0238`,
    `settled_z_abs=0.0201`, `sz=0.0759`).

- `claudedocs/session_20260515_p7_rollout_failure_diag.md`
  - Boot followed `CLAUDE.md` Current-State Protocol.
  - Pre-code md5 matched requested baseline:
    `chain_skills.py=c6e610216197994c6b7d2b6625d87560`,
    launcher `b34ef3853ac993a1e2adbaddb420adab`,
    stack env `996f2afce7de1b3be93ae43ddc349f8e`,
    `train_ppo.py=6b0ffdb8365c5e37ced00833c0556c19`,
    P7 launcher `2acd462042d0997610fca25ff7a41e21`,
    reset probe `43a04e3cfca763a50d8c856185d14b99`,
    SurfaceGripper probe `053fced6551ccb02d8a9ea6c04fb4a30`.
  - Requested B200 logs existed on B200 `/tmp` and key lines were checked:
    reset probe lines 65-68, P7v1 line 584/589/596, P7v3 diag line
    584/589/594/596, P7v3 full line 14984-14994, SurfaceGripper v2
    lines 143/152/164, SurfaceGripper v3 lines 144/153/165.
  - Added `sim_scripts/p7_rollout_failure_diag.py` md5
    `a9743d74886c454b1c161a1bade3df93`.
  - B200 run:
    `/tmp/p7v3_rollout_failure_diag.{out,err}`.
  - Result: 256 envs × 2 episodes, all 512 episodes classified as
    `C_tips_during_attached_transport`. Reset was upright
    (`d_xy=0.1732`, `sz_world_z=1.0000`), but pre-release upright collapsed
    (`sz_world_z=0.2667`) and release was still bad (`d_xy=0.0739`,
    `release_z_offset=0.0788`, `sz_world_z=0.2851`). Final z looked close only
    because the sponge was lying flat (`settled_z_offset=0.0006`,
    `sz_world_z=0.0156`).

- `claudedocs/session_20260515_g2a_layout_source_sweep.md`
  - Boot followed `CLAUDE.md` Current-State Protocol.
  - Pre-code local/B200 md5 matched expected v9 stable handoff baseline:
    `chain_skills.py=f9a935cbcd7102f7bc65560f231924de`,
    launcher `6013cafdd140d3d3dbdbebe1efc9f67e`,
    URDF `cb5ce1232fd3a4f5e8ee6c456577a215`,
    G2-A STL `02115511bbea2abb82814c6329ec9cea`.
  - B200 G2-A USD md5s verified:
    `roarm_m3.usd=4497024d25abab11de5c50e144124553`,
    `configuration/roarm_m3_physics.usd=5a4eb57ade18d2a4fd0676b43ac9dd12`,
    `usd/.asset_hash=b57d9fe1ac60f5a4f0562f4437783666`.
  - Prior v4-v9 B200 logs were checked before coding.
  - Implemented G2-A v10 minimal scripted release bridge.
  - B200 v10 run: `/tmp/chain_topdown_g2a_v10_scripted_release_bridge.{out,err}`.
  - Result: short-handoff release diagnostic `CHAIN_FINAL_SUCCESS=YES`.
  - Implemented G2-A v11 seed0 four-source layout-source diagnostic for the
    current single-sponge env.
  - B200 v11 run:
    `/tmp/chain_topdown_g2a_v11_layout_source_sweep.{out,err}`.
  - Result: S1 failed before release during long attached Skill 2 transport:
    `max_arm_err=253.21deg`, `tcp_err=486.5mm`,
    `CHAIN_FINAL_SUCCESS=NO`.
  - Implemented a CPU-only SurfaceGripper transport probe:
    `sim_scripts/surface_gripper_transport_probe.py`.
  - B200 SurfaceGripper probe runs:
    `/tmp/roarm_surface_gripper_transport_probe_v2.{out,err}` and
    `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.{out,err}`.
  - Result: quick dynamic SurfaceGripper retrofit did not attach
    (`close_detect_step=-1`); the robot reached transport TCP
    (`tcp_err=7.9mm`) but the sponge stayed at the source
    (`d_xy_pre_release=166.1mm`), `SURFACE_PROBE_SUCCESS=NO`.
  - Implemented P7 learned attached transport/release curriculum from realistic
    G2-A seed0 four-source starts.
  - B200 reset probe:
    `/tmp/p7v1_attached_reset_probe_v2.{out,err}` verified `_grasped=1.0`,
    sponge-TCP `0.00mm`, initial mean `d_xy=175.8mm`.
  - B200 P7v3 run:
    `/tmp/p7v3_transport_release.{out,err}`.
  - Result: partial transport improvement only. Iter 496 reached
    `p7_xy_offset_mean=51.2mm`, but `p7_on_target_rate=0.0005`,
    `p7_place_success_rate=0.0007`, `upright_rate=0.0576`.

Active pivot: **Hierarchical chain skills with G2-A collision proxy, but the
current learned transport/release failure is object orientation collapse during
attached transport/release**. The v10 minimal release bridge is valid only after a
stable near-target handoff; it does not solve source-to-target attached transport.
The quick SurfaceGripper retrofit is not a drop-in replacement: it failed to
close/attach on the current RoArm USD.

Paper-quality pivot: either redesign learned attached transport/release under
the now-gated attach semantics so the policy actually transports and opens, or
properly author and unit-test a physics gripper/constraint asset before replacing
the kinematic pose-write attach.

## Current Status

RoArm + Isaac Lab sponge stacking research. B200 = headless state-only learning/eval
(D004). Rendering off by default unless explicitly testing render/replay.

### Hierarchical Chain

`roarm_rl/chain_skills.py` = scripted Skill 0/1/2/3/4:

- Skill 0/1/2/4 use IK + `robot_dof_targets` force-set per D007.
- Skill 1b uses the G2-A collision proxy and settled-center top-down pick path.
- Skill 1c terminates immediately on `_grasped=True` per D010.
- Skill 2 uses stable short attached handoff: wrist_r held `+90°`, latch gripper
  held near 26°, arm-only break; no attached dwell per D012.
- Skill 3 is now a minimal scripted release bridge:
  open below `grasp_gripper_thresh`, let `_grasped` clear, hold/minimize robot
  motion, settle.

Latest B200 v10 metrics:

- Guard: `GUARD-OK chain_md5=4bf308b8c0026671772ca3503f4f5387`.
- Skill 1b: no top-contact stall, `stall_signature=FALSE_at_b3`.
- Skill 1c: latch after step 15, `gripper_q=23.02°`,
  `d_sponge_tcp=23.6mm`, `grasped=True`.
- Skill 2: `steps=1`, `max_arm_err=1.10°`, `tcp_err=8.0mm`,
  `grasped=True`, `sponge_z=40.1mm`.
- Skill 3 bridge: `release_step=1`, `gripper_q=21.76°`,
  `_grasped=False`; settled `d_xy=22.3mm`, `d_z=12.1mm`,
  `CHAIN_SETTLED=YES`.
- Skill 4 retreat: `final_d_xy=22.3mm`, `final_d_z=12.1mm`,
  `CHAIN_FINAL_SUCCESS=YES`.

Latest B200 v11 four-source S1 diagnostic:

- Guard: `GUARD-OK chain_md5=c6e610216197994c6b7d2b6625d87560`.
- Harness: `--layout_source_sweep` uses seed0 four-source positions, but the
  current env has only one sponge and no L1 support bodies, so L2 targets are
  intentionally skipped.
- S1 source `(0.2137,-0.1957)` -> L1.sp1 target `(0.2800,-0.0435,+0.0114)`.
- Skill 1b/1c still pass locally: latch after step 15,
  `gripper_q=23.02deg`, `d_sponge_tcp=21.2mm`, `grasped=True`.
- Failure occurs before release: Skill 2 long attached transport hits 120 steps,
  `max_arm_err=253.21deg`, `tcp_err=486.5mm`, `sponge_z=66.7mm`.
- Release bridge then opens correctly but the sponge is already far from target:
  settled `d_xy=555.0mm`, `d_z=12.1mm`, `CHAIN_FINAL_SUCCESS=NO`.
- Harness caveat: the in-process sweep only ran S1 because `run_chain_isaac`
  closes `sim_app`. S1 is still sufficient as a counterexample to the current
  primitive in four-source geometry.

Latest B200 SurfaceGripper quick-retrofit diagnostic:

- Isaac Lab SurfaceGripper is CPU-only in the installed source/tutorial.
- v2 created a SurfaceGripper under `Robot/link5/SurfaceGripper` with TCP offset.
  It failed to attach: `close_detect_step=-1`.
- v3 created a SurfaceGripper under `Robot/gripper_link/SurfaceGripper`, zero
  offset, `grip_distance=0.200`. It also failed to attach:
  `close_detect_step=-1`.
- In both runs the robot reached the transport TCP (`tcp_err=7.9mm`) but the
  sponge stayed at source (`d_xy_pre_release=166.1mm`), ending
  `SURFACE_PROBE_SUCCESS=NO`.
- Interpretation: this rejects a quick dynamic prim retrofit, not the broader
  constraint/gripper direction. A useful constraint branch needs proper asset
  authoring and a unit test that reaches `Closed` before chain integration.

Latest B200 P7 attached-learning diagnostic:

- Reset starts from realistic G2-A four-source attached handoffs:
  `_grasped=1.0`, sponge-TCP `0.00mm`, mean source-target `d_xy=175.8mm`.
- P7v1 rejected: still encouraged closed/high hold (`p7_xy_offset=239.1mm`,
  `gripper_open_rate=0.0631`, `sponge_height=143.7mm` at iter 16/20).
- P7v3 full run improved transport but failed the primitive:
  `p7_xy_offset_mean=51.2mm`, `p7_release_z_offset=32.8mm`,
  `p7_settled_z_offset=13.8mm`, but `p7_on_target_rate=0.0005`,
  `p7_place_success_rate=0.0007`, `upright_rate=0.0576`.
- Interpretation: B200 PPO learned a partial transport tendency, but release/
  settle/upright placement remains unsolved.

Latest B200 P7 rollout failure diagnostic:

- `sim_scripts/p7_rollout_failure_diag.py` evaluates `model_499.pt` state-only,
  starting from exact G2-A attached starts.
- B200 `/tmp/p7v3_rollout_failure_diag.out`:
  - line 42: checkpoint path verified.
  - line 43: `num_envs=256 episodes=2 seed=0`.
  - line 93: `completed_episodes=512`.
  - line 95: `C_tips_during_attached_transport: 512 (1.000)`.
  - line 97: reset mean `d_xy=0.1732`, `sz_world_z=1.0000`.
  - line 98: pre-release mean `sz_world_z=0.2667`.
  - line 99: release mean `d_xy=0.0739`, `release_z_offset=0.0788`,
    `sz_world_z=0.2851`.
  - line 101: final mean `settled_z_offset=0.0006` but `sz_world_z=0.0156`.
- Interpretation: dominant failure is not just wrong release height or residual
  XY. The object is already tipped/rotated before release; final z metrics can be
  misleading because the sponge is lying flat.

Latest B200 P7 action/TCP/quaternion trace:

- `sim_scripts/p7_action_tcp_quat_trace.py` evaluates `model_499.pt` state-only,
  starting from exact G2-A attached starts and printing 4 deterministic step
  traces plus 256-env aggregate transition events.
- B200 `/tmp/p7v3_action_tcp_quat_trace.out`:
  - line 43: checkpoint path verified.
  - line 44: `num_envs=256 max_steps=60 trace_envs=4 seed=0`.
  - lines 93-97: max episode length, step dt, gripper threshold, gripper idx,
    and attach semantics reminder.
  - line 99: reset mean `d_xy=0.1722`, `sz=1.0000`,
    `d_sponge_tcp=0.00000`, `grasped=1.000`.
  - lines 100-103: step 1 samples already show tipping while `open=0` and
    `grasped=1`.
  - lines 245-253: `first_tip_while_grasped=256/256`,
    `tip_before_or_at_open=256/256`, no >3cm TCP jump.
  - lines 254-260: mean first tip while grasped step `1.72`; mean open/release
    step `20.21`.
  - lines 261-264: release mean `sz=0.2983`, `d_xy=0.0714`,
    `rel_z_abs=0.0764`; final mean `d_xy=0.0238`,
    `settled_z_abs=0.0201`, `sz=0.0759`,
    `max_abs_action_mean=1.0000`.
- Interpretation: orientation collapse begins during attached motion, not at
  release. `_update_grasp_attach` preserves the tipped quaternion once physics
  has produced it. Next should be attach quaternion reset/constraint diagnostic
  or authored physics gripper/constraint unit test, not reward hacking.

Latest B200 P7 attach quaternion constraint probe:

- `sim_scripts/p7_attach_quat_constraint_probe.py` uses runtime-only
  monkey-patch modes for `_update_grasp_attach`; it does not change repository
  env/reward/chain semantics.
- `preserve+keep` B200 `/tmp/p7v3_attach_quat_preserve_keep.out`:
  - lines 141-149: `first_tip_while_grasped=256/256`,
    `tip_before_or_at_open=256/256`, no >3cm TCP jump.
  - lines 151-160: mean first tip while grasped `1.67`,
    release `sz=0.1561`, final `sz=0.0101`.
- `identity+zero` B200 `/tmp/p7v3_attach_quat_identity_zero.out`:
  - lines 141-149: `first_tip_while_grasped=189/256`,
    `tip_before_or_at_open=128/256`.
  - lines 151-160: release `sz=0.9664`, final `sz=0.9113`,
    but final `d_xy=0.2487`.
- `identity+keep` B200 `/tmp/p7v3_attach_quat_identity_keep.out`:
  - lines 141-149: `first_tip_while_grasped=77/256`,
    `tip_before_or_at_open=11/256`.
  - lines 151-160: release `sz=0.9921`, final `sz=0.6434`,
    final `d_xy=0.2604`.
- Interpretation: quaternion preservation is a primary amplifier of immediate
  tip; velocity zeroing alone is not. But constraining quaternion changes the
  mechanics enough that the old policy no longer transports to target. This is
  a diagnostic PASS, primitive FAIL.

### Critical Lessons

- D006: URDF gripper lower limit clamps negative open targets. Use
  `GRIPPER_OPEN_DEG=0.0`; do not casually change the URDF lower limit.
- D007: scripted skills must force-set `robot_dof_targets`; action-interface
  closed-loop creates PD limit cycles.
- D008/D009: original Skill 1b stall was gripper-link top-contact at settled
  sponge top; do not resume z-stage/ramping variants.
- D010: close must stop immediately on `_grasped=True`.
- D011/D012: attached transport must be short. Do not rotate wrist_r `+90°→0°`
  while attached, and do not continue closing/dwell for convergence.
- D013: v10 minimal release bridge succeeds physically from stable G2-A handoff,
  but should not be overclaimed as a learned release primitive.
- D014: four-source S1 fails upstream of release because long attached transport
  reintroduces Skill 2 runaway under `_update_grasp_attach`.
- D015: quick dynamic SurfaceGripper retrofit did not attach; do not keep trying
  arbitrary parent/offset variants. Proper USD/constraint authoring or learned
  transport/release is required.
- D016: P7 attached-learning improves XY but does not solve release/upright
  placement. Do not claim learned transport/release solved.
- D017: P7 `model_499.pt` rollout failure is dominated by object orientation
  collapse during attached transport/release. Do not run another blind reward
  variant before inspecting attached action/path/quaternion dynamics.
- D018: P7 upright collapse starts while still kinematically attached and before
  gripper open/release. Do not change P7 reward first; diagnose/repair attached
  orientation semantics or use an authored physics gripper/constraint unit test.
- D019: Runtime attach quaternion constraint suppresses immediate tipping but
  does not solve transport/release with the old policy. Do not claim attach
  reset solved P7; pick a mechanics branch before reward tuning.
- D020: Env-level identity+keep attach semantics are active and useful, but a
  fresh short P7 diagnostic still failed by no-release/poor transport. Do not
  claim attach semantics solved P7.

## Current Direction

Next concrete action:

1. Keep G2-A collision proxy and regenerated USD.
2. Keep v10 release bridge semantics as a short-handoff diagnostic only.
3. Do not add random scripted release variants for v11. The failing surface is
   long attached transport, not gripper-open release.
4. Next valid branches:
   - redesign P7 controller/reward/curriculum under env-level `identity+keep`
     semantics so it actually transports and opens, then run a longer fresh
     train/eval only if short diagnostics improve;
   - properly author a physics gripper/constraint asset, prove it reaches
     `Closed` on the sponge in a unit test, then re-test transport;
   - only after mechanics are meaningful, revisit an orientation-aware learned
     reward branch.
5. Do not keep trying arbitrary SurfaceGripper parent/offset variants without an
   authored asset/axis/API hypothesis.

Only after transport and release both remain stable in the intended four-sponge
scene should demo generation resume.

## Must Read First

1. `claudedocs/DECISIONS.md` D006-D020
2. `claudedocs/EXPERIMENT_LEDGER.md` rows:
   2026-05-14 `(δ.4)`, `(δ.5)`, `G1/G2-A`, `G2-A v4`, `G2-A v5-v9`,
   and 2026-05-15 `G2-A v10`, `G2-A v11`, `SurfaceGripper probe v2/v3`,
   `P7 G2-A attached transport/release`, `P7 model_499 rollout failure diag`,
	   `P7 action/TCP/quat trace`, `P7 attach quat constraint probe`,
	   and 2026-05-17 `P7 env attach semantics A`
3. `claudedocs/session_20260517_p7_attach_semantics_env_experiment.md`
4. `claudedocs/session_20260515_p7_attach_quat_constraint_probe.md`
5. `claudedocs/session_20260515_p7_action_tcp_quat_trace.md`
6. `claudedocs/session_20260515_p7_rollout_failure_diag.md`
7. `claudedocs/session_20260515_g2a_scripted_release_bridge.md`
8. `claudedocs/session_20260515_g2a_layout_source_sweep.md`
9. `claudedocs/session_20260515_p7_attached_transport_learning.md`
10. `claudedocs/session_20260514_alpha_prime_delta_topdown.md` APPENDIX,
   `(δ.4)`, `(δ.5)`, `(G1/G2-A)`, `(G2-A v4)`, `(G2-A v5-v9)`
11. `roarm_rl/chain_skills.py`
12. `roarm_rl/roarm_stack_env.py` `_pre_physics_step`, `_apply_action`,
   `_grasp_condition`, `_update_grasp_attach`

## Source Files To Verify Before Coding

- `roarm_rl/chain_skills.py` — md5 `c6e610216197994c6b7d2b6625d87560`
- `launch_chain_topdown.sh` — md5 `b34ef3853ac993a1e2adbaddb420adab`
- `roarm_rl/roarm_stack_env.py` — md5 `47dad11d9f99b007d2ff22ff0fbdbad7`
- `roarm_rl/train_ppo.py` — md5 `a056cb61819deea963e1368b196bf0d4`
- `launch_p6v17_transport_release.sh` — md5
  `2acd462042d0997610fca25ff7a41e21`
- `sim_scripts/attached_transport_reset_probe.py` — md5
  `43a04e3cfca763a50d8c856185d14b99`
- `sim_scripts/surface_gripper_transport_probe.py` — md5
  `053fced6551ccb02d8a9ea6c04fb4a30`
- `sim_scripts/p7_rollout_failure_diag.py` — md5
  `a9743d74886c454b1c161a1bade3df93`
- `sim_scripts/p7_action_tcp_quat_trace.py` — md5
  `e6c9424cfe7ffafdf00fe0625f0553f7`
- `sim_scripts/p7_attach_quat_constraint_probe.py` — md5
  `a2e16f7683856ead1a9a9eef1da8ea69`
- `sim_scripts/p7_attach_semantics_env_probe.py` — md5
  `4997a3ec058773004441b74419da114f`
- `local_assets/roarm_m3/urdf/roarm_m3.urdf` — md5
  `cb5ce1232fd3a4f5e8ee6c456577a215`
- `local_assets/roarm_m3/urdf/meshes/gripper_link_collision_g2a.stl` — md5
  `02115511bbea2abb82814c6329ec9cea`
- B200 USD:
  - `assets/roarm_m3/usd/roarm_m3.usd` =
    `4497024d25abab11de5c50e144124553`
  - `assets/roarm_m3/usd/configuration/roarm_m3_physics.usd` =
    `5a4eb57ade18d2a4fd0676b43ac9dd12`
  - `assets/roarm_m3/usd/.asset_hash` =
    `b57d9fe1ac60f5a4f0562f4437783666`
- Latest B200 logs:
  `/tmp/chain_topdown_g2a_v10_scripted_release_bridge.{out,err}`,
  `/tmp/chain_topdown_g2a_v11_layout_source_sweep.{out,err}`,
  `/tmp/roarm_surface_gripper_transport_probe_v2.{out,err}`,
  `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.{out,err}`,
  `/tmp/p7v1_attached_reset_probe_v2.{out,err}`,
  `/tmp/p7v3_transport_release.{out,err}`,
  `/tmp/p7v3_rollout_failure_diag.{out,err}`,
  `/tmp/p7v3_action_tcp_quat_trace.{out,err}`,
	  `/tmp/p7v3_attach_quat_identity_zero.{out,err}`,
	  `/tmp/p7v3_attach_quat_preserve_keep.{out,err}`,
	  `/tmp/p7v3_attach_quat_identity_keep.{out,err}`,
	  `/tmp/p7_attach_semantics_identity_keep.{out,err}`,
	  `/tmp/p7_attach_semantics_preserve_zero.{out,err}`,
	  `/tmp/p7v4_attach_identity_keep_diag20.{out,err}`,
	  `/tmp/p7v4_attach_identity_keep_model19_trace.{out,err}`

## Do Not Trust As Current State

- `HANDOFF.md`, `TASKS.md`
- Path D nominal success without CLEAN split
- (δ.2) original result table L138-139 Skill 0/1a step counts; use APPENDIX errata
- More Skill 1b z-stage tuning, δ.3 ramping, or N-stage descent variants
- Treating P6v14a Skill 3 as compatible with stable G2-A handoff
- Treating quick dynamic SurfaceGripper creation as solved physics attach
- Treating P7 attached-learning as solved transport/release
- Treating env-level attach identity/keep as solved P7
- Treating P7 final z/XY improvement as success when `sz_world_z` shows the
  sponge is lying flat
- Four-sponge demo generation before solving long attached transport and release
- Any memory-only metric not rechecked against files/logs
