# Session 2026-05-15 — G2-A v11 Layout-Source Sweep Diagnostic

## Scope

User direction after v10:

- Treat v10 as a diagnostic PASS only: stable G2-A handoff is compatible with
  minimal physical release, not proof that a learned release primitive is solved.
- Validate the primitive against the four-sponge layout.
- If it becomes brittle, do not add random scripted release variants; move toward
  learning from the stable handoff distribution.
- Cross-check against NVIDIA Isaac Sim / Isaac Lab documentation and available
  open-source examples.

## Baseline

Starting code after v10:

- `roarm_rl/chain_skills.py` implemented the minimal scripted release bridge.
- `launch_chain_topdown.sh` guarded that v10 md5.
- v10 B200 log:
  `/tmp/chain_topdown_g2a_v10_scripted_release_bridge.out`
  showed `CHAIN_FINAL_SUCCESS=YES`.

The v10 caveat remains binding: the release bridge succeeded only from the short,
near-target stable handoff.

## Code Changes

Implemented a minimal four-source diagnostic in `roarm_rl/chain_skills.py`:

- Added seed0 source XYs:
  - S1 `(+0.2136961687,-0.1957191958)`
  - S2 `(+0.1516527636,+0.1757251311)`
  - S3 `(+0.3906635776,-0.1324604127)`
  - S4 `(+0.4235072424,+0.1723780331)`
- Added L1 floor targets:
  - L1.sp1 `(+0.2800,-0.0435,+0.0114)`
  - L1.sp2 `(+0.2800,+0.0435,+0.0114)`
- Added `--place_xyz` so the env target and planner target are both updated.
- Added `--layout_source_sweep`.
- Kept L2 targets intentionally skipped because current `RoArmStackEnv` has only
  one sponge and no support bodies for physical stacking.
- Added launcher argument passthrough so B200 can run:
  `bash ./launch_chain_topdown.sh --layout_source_sweep`.

Post-change md5:

- `roarm_rl/chain_skills.py` =
  `c6e610216197994c6b7d2b6625d87560`
- `launch_chain_topdown.sh` =
  `b34ef3853ac993a1e2adbaddb420adab`

Local checks:

- `python -m py_compile roarm_rl/chain_skills.py` passed.
- `python roarm_rl/chain_skills.py --dry-run` passed.

## B200 v11 Run

Run:

- `/tmp/chain_topdown_g2a_v11_layout_source_sweep.out`
- `/tmp/chain_topdown_g2a_v11_layout_source_sweep.err`

Log md5:

- out = `7d5c9e0a86f89a8b3b6d602e2299a88b`
- err = `98f104f2c61752925cb48898298a1a45`

Key stdout lines:

- line 6: `GUARD-OK chain_md5=c6e610216197994c6b7d2b6625d87560`
- lines 8-9: seed0 four-source diagnostic; L2 targets skipped because the
  current env lacks support sponges.
- line 11: S1 source `(+0.2137,-0.1957)` to target
  `(+0.2800,-0.0435,+0.0114)`.
- line 132: Skill 1b no top-contact stall:
  `total_steps=29`, `per_stage=(13,8,8)`,
  `stall_signature=FALSE_at_b3`.
- line 137: Skill 1c latch detected after step 15.
- line 138: close ended with `gripper_q=23.02deg`,
  `d_sponge_tcp=21.2mm`, `grasped=True`.
- line 142: Skill 2 pre-state already shows the four-source problem:
  `arm_err=+30.69deg`, `tcp_err=155.7mm` because source-to-target transport is
  no longer a one-step near-target handoff.
- line 263: Skill 2 fails before release:
  `steps=120`, `max_arm_err=253.21deg`, `tcp_err=486.5mm`,
  `grasped=True`, `sponge_z=66.7mm`.
- line 267: Skill 3 bridge starts with the sponge already far from target:
  `d_xy=494.7mm`, `d_z=55.3mm`.
- line 269: bridge release still occurs at `release_step=2`.
- line 280: post-release settled state is
  `d_xy=555.0mm`, `d_z=12.1mm`, `CHAIN_SETTLED=NO`.
- line 283: final `CHAIN_FINAL_SUCCESS=NO`.

stderr contained Isaac/NVML/cloner warnings only. The script still produced the
diagnostic output.

Harness caveat: `--layout_source_sweep` only ran S1 in this process because
`run_chain_isaac()` closes `sim_app`. This is not a full 4/4 sweep result.
However, S1 alone is already a valid counterexample: the current primitive fails
the four-source distribution before release.

## Isaac / NVIDIA Cross-Check

Primary-source findings:

- Isaac Lab DirectRLEnv officially separates `_pre_physics_step(actions)` and
  `_apply_action()`; `_pre_physics_step` caches/processes action data before
  physics, while `_apply_action` runs before each physics step. This matches the
  local env structure, but it does not validate the local kinematic attach model.
  Source: Isaac Lab Direct Workflow RL Environment,
  https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html
  lines 532-541.
- Isaac Lab asset docs describe rigid objects as PhysX rigid bodies and
  articulations as jointed systems with position/velocity/effort commands. This
  supports using `set_joint_position_target` for the robot and rigid-object state
  writes for resets/teleports, but not as evidence that per-step object pose
  writing is a physically faithful gripper. Source:
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html
  lines 942-945 and 2844-2851.
- Isaac Sim Surface Gripper is the official extension for suction/distance-based
  grip/release. It has `close_gripper`, `open_gripper`, `get_gripped_objects`,
  and action thresholds where negative action opens and positive action closes.
  Source:
  https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.robot.surface_gripper/docs/index.html
  lines 664-667, 721-729, 763-771, 848-855, 890-898.
- Open-source Isaac Lab lift rewards optimize object lift and object-goal
  distance with a physically simulated rigid object. They do not use the local
  `_update_grasp_attach` style kinematic pose pin as a proof of transport
  validity. Source:
  https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/rewards.py
  lines 408-417 and 451-487.

Interpretation:

- The current local code follows a recognizable Isaac Lab DirectRLEnv shape.
- The failing part is not "Isaac Lab usage" in general. It is the local grasp
  simplification: while `_grasped=True`, `_update_grasp_attach` writes sponge
  root pose to the TCP and zeroes velocity every step.
- v11 shows this simplification is acceptable for short handoff + release
  diagnostics, but not for long four-source transport.

## Verdict

FAIL for four-source validation.

This is not evidence that the v10 release bridge is brittle. It is evidence that
the object never reaches a valid release state in the four-source layout because
long attached Skill 2 transport runs away first.

Do not add random scripted release variants. Next valid paths are:

1. Train a learned transport/release primitive from the stable G2-A pick state
   and realistic source-to-target distributions.
2. Replace the current kinematic pose-write attach with a proper physics
   gripper/constraint diagnostic, then re-test long transport.
3. Build a true multi-object/four-sponge env only after single-object transport
   and release are physically valid.

## SurfaceGripper Quick-Retrofit Probe

User correction after v11:

- A learned release primitive alone cannot fix v11, because the sponge is already
  about 0.5 m from target before release.
- Next valid options are either learning transport/release from realistic
  four-source attached distributions, or replacing `_update_grasp_attach` with a
  SurfaceGripper/constraint physics model and re-testing long transport.

Local / NVIDIA / Isaac cross-check:

- Installed Isaac Lab SurfaceGripper is CPU-only. Local source
  `/NHNHOME/arf/IsaacLab/source/isaaclab/isaaclab/assets/surface_gripper/surface_gripper.py`
  lines 48-50 states CPU-only support; lines 260-265 enforce a CPU device.
- The installed tutorial
  `/NHNHOME/arf/IsaacLab/scripts/tutorials/01_assets/run_surface_gripper.py`
  lines 6-15 repeats that `--device=cpu` is required because SurfaceGripper is
  currently CPU-only.
- Installed Isaac Lab tests
  `/NHNHOME/arf/IsaacLab/source/isaaclab/test/assets/test_surface_gripper.py`
  lines 203-217 expect a CUDA creation exception. So the probe was intentionally
  CPU-only.

Added `sim_scripts/surface_gripper_transport_probe.py`:

- Creates `RoArmStackEnvCfg` scene assets directly without using
  `_update_grasp_attach`.
- Dynamically creates an Isaac Sim `SurfaceGripper` prim before reset.
- Replays the v11 S1 source and L1.sp1 target.
- Measures `close_detect_step`, transport `tcp_err`, sponge/target positions,
  `d_xy_pre_release`, release step, and `SURFACE_PROBE_SUCCESS`.

Post-probe script md5:

- `sim_scripts/surface_gripper_transport_probe.py` =
  `053fced6551ccb02d8a9ea6c04fb4a30`

B200 probe v2:

- Run logs:
  `/tmp/roarm_surface_gripper_transport_probe_v2.out`,
  `/tmp/roarm_surface_gripper_transport_probe_v2.err`
- Log md5:
  - out = `1f29a440cb04638fe4c2f96f0ca8d12b`
  - err = `0de1b7acbae216cbfbde8aea24823a39`
- Key stdout lines:
  - line 89: created `/World/envs/env_0/Robot/link5/SurfaceGripper`
  - line 143: `close_detect_step=-1`
  - line 152: robot reached transport TCP (`tcp_err=7.9mm`) but sponge stayed at
    source with `d_xy_pre_release=166.1mm`
  - line 164: `SURFACE_PROBE_SUCCESS=NO`

B200 probe v3:

- Run logs:
  `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.out`,
  `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.err`
- Log md5:
  - out = `b7994e518c52f5722df024dc0d5c661a`
  - err = `4fdeea84ede1b9db0b3ed32bdc7bc96e`
- Key stdout lines:
  - line 90: created
    `/World/envs/env_0/Robot/gripper_link/SurfaceGripper` with zero offset and
    `grip_distance=0.200`
  - line 144: `close_detect_step=-1`
  - line 153: robot reached transport TCP (`tcp_err=7.9mm`) but sponge stayed at
    source with `d_xy_pre_release=166.1mm`
  - line 165: `SURFACE_PROBE_SUCCESS=NO`

Critical interpretation:

- The quick SurfaceGripper retrofit failed at close/attach, not at long attached
  transport. Since no attach occurred, the robot could move to the target TCP and
  the sponge remained at the source.
- This does not prove that a physical gripper/constraint model cannot work. It
  proves that dynamic prim creation with guessed parent/offset settings is not a
  drop-in replacement for `_update_grasp_attach` on the current RoArm USD.
- Do not keep trying random SurfaceGripper parent/offset variants. A useful
  constraint branch needs authored USD pose/axis/API semantics and a unit test
  that reaches `state=Closed` on the sponge before chain integration.

Updated verdict:

The v11 failure remains a transport/attach-model failure, not a release-only
failure. The two defensible next branches are:

1. Train a learned transport/release primitive from realistic G2-A four-source
   attached distributions.
2. Properly author and validate a SurfaceGripper/constraint asset first, then
   replace `_update_grasp_attach` and re-run long transport.
