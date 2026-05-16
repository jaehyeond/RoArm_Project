# Session 2026-05-15 — P7 model_499 rollout failure diagnostic

## Scope

User direction:

- Do not change P7 reward before diagnosing `model_499.pt`.
- Do not add scripted release variants.
- Do not random-search SurfaceGripper parent/offset.
- Do not claim P7 solved from mean XY improvement.
- Use B200 stdout line evidence, not tensorboard-only claims.

## Boot / Verification

- Followed `CLAUDE.md` Current-State Protocol.
- Read `START_HERE.md`.
- Read `claudedocs/DECISIONS.md` D014-D016.
- Read `claudedocs/EXPERIMENT_LEDGER.md` rows for:
  - 2026-05-15 `(G2-A v10)`
  - 2026-05-15 `(G2-A v11)`
  - 2026-05-15 `(SurfaceGripper probe v2/v3)`
  - 2026-05-15 `(P7 G2-A attached transport/release)`
- Read:
  - `claudedocs/session_20260515_g2a_scripted_release_bridge.md`
  - `claudedocs/session_20260515_g2a_layout_source_sweep.md`
  - `claudedocs/session_20260515_p7_attached_transport_learning.md`
- `git status --short` was dirty before coding; existing modified/untracked
  files were treated as user/session state and not reverted.

## Pre-Code md5 Verification

Local md5s matched the requested baseline:

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

## Prior B200 Log Verification

The requested B200 logs existed on B200 `/tmp`:

- `/tmp/p7v1_attached_reset_probe_v2.{out,err}`
- `/tmp/p7v1_diag20.{out,err}`
- `/tmp/p7v3_diag20.{out,err}`
- `/tmp/p7v3_transport_release.{out,err}`
- `/tmp/roarm_surface_gripper_transport_probe_v2.{out,err}`
- `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.{out,err}`

Key verified B200 stdout lines:

- Reset probe:
  - `/tmp/p7v1_attached_reset_probe_v2.out` line 65:
    `grasped_frac=1.000`
  - line 66: `was_grasped_frac=1.000`
  - line 67: `d_sponge_tcp_mean_mm=0.00`
  - line 68: `d_xy_mean_mm=175.80`
- P7v1 diagnostic:
  - `/tmp/p7v1_diag20.out` line 584: `p7_xy_offset_mean=0.2391`
  - line 589: `p7_gripper_open_rate=0.0631`
  - line 596: `p7_sponge_height_m=0.1437`
- P7v3 diagnostic:
  - `/tmp/p7v3_diag20.out` line 584: `p7_xy_offset_mean=0.1848`
  - line 589: `p7_gripper_open_rate=0.2110`
  - line 594: `p7_place_success_rate=0.0000`
  - line 596: `p7_sponge_height_m=0.0478`
- P7v3 full run:
  - `/tmp/p7v3_transport_release.out` line 14984:
    `p7_xy_offset_mean=0.0512`
  - line 14985: `p7_release_z_offset_mean=0.0328`
  - line 14986: `p7_settled_z_offset_mean=0.0138`
  - line 14989: `p7_gripper_open_rate=0.8298`
  - line 14991: `p7_on_target_rate=0.0005`
  - line 14993: `p7_upright_rate=0.0576`
  - line 14994: `p7_place_success_rate=0.0007`
- SurfaceGripper v2:
  - `/tmp/roarm_surface_gripper_transport_probe_v2.out` line 143:
    `close_detect_step=-1`
  - line 152: `tcp_err=7.9mm`, `d_xy_pre_release=166.1mm`
  - line 164: `SURFACE_PROBE_SUCCESS=NO`
- SurfaceGripper v3:
  - `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.out` line 144:
    `close_detect_step=-1`
  - line 153: `tcp_err=7.9mm`, `d_xy_pre_release=166.1mm`
  - line 165: `SURFACE_PROBE_SUCCESS=NO`

The corresponding stderr files contained Isaac/NVML warnings, but no Python
traceback for these prior runs.

## Code Change

Added `sim_scripts/p7_rollout_failure_diag.py`.

Design:

- Headless/state-only Isaac Lab script.
- Uses `RoArmStackEnvCfg` with `reward_phase=7`,
  `curriculum_attached_transport_release=True`, and exact attached starts
  (`curriculum_attached_start_jitter_rad=0.0`).
- Loads the rsl_rl policy through `RslRlVecEnvWrapper` and `OnPolicyRunner`,
  matching `eval_policy.py` / `train_ppo.py` policy-loading conventions.
- Records per episode:
  - release step: first step where `_grasped=False` or gripper opens
  - sponge/TCP/target at reset, pre-release, release, post-settle, final
  - `d_xy`, `release_z_offset`, `settled_z_offset`
  - sponge quaternion and `sz_world_z = 1 - 2(qx^2 + qy^2)`
  - failure bucket A-F.

Post-change md5:

- `sim_scripts/p7_rollout_failure_diag.py` =
  `a9743d74886c454b1c161a1bade3df93`

Local check:

- `python -m py_compile sim_scripts/p7_rollout_failure_diag.py` passed.

B200 synced script md5 matched:

- `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/p7_rollout_failure_diag.py`
  = `a9743d74886c454b1c161a1bade3df93`

## Smoke Run

B200 smoke:

- `/tmp/p7v3_rollout_failure_diag_smoke.out`
- `/tmp/p7v3_rollout_failure_diag_smoke.err`

The first smoke exposed a logging bug: automatic reset after the requested
episode created an extra reset record. The script was fixed to stop recording at
`num_envs * episodes`.

Second smoke key stdout lines:

- line 90: `max_episode_length=200`
- line 93: `completed_episodes=16`
- line 95: `C_tips_during_attached_transport: 16 (1.000)`
- line 97: reset mean `d_xy=0.1725`, `sz_world_z=1.0000`
- line 98: pre-release mean `sz_world_z=0.3435`
- line 99: release mean `d_xy=0.0751`, `release_z_offset=0.0801`,
  `sz_world_z=0.3761`
- line 100: post-settle mean `sz_world_z=0.0625`

## B200 Diagnostic Run

Run:

- `/tmp/p7v3_rollout_failure_diag.out`
- `/tmp/p7v3_rollout_failure_diag.err`

Command:

```bash
python sim_scripts/p7_rollout_failure_diag.py \
  --checkpoint /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/logs/roarm_rl/roarm_stack_p7v3_g2a_attached_transport_release/model_499.pt \
  --num_envs 256 \
  --episodes 2 \
  --sample_print 24
```

Log md5:

- out = `5e02ca5a8ca6f6b7c457a75f5aa9add8`
- err = `f22b22edb4054095c6d6a98b6281d5e7`

Key stdout lines:

- line 42: checkpoint path is
  `.../roarm_stack_p7v3_g2a_attached_transport_release/model_499.pt`
- line 43: `num_envs=256 episodes=2 seed=0`
- line 90: `max_episode_length=200`
- line 91: `grasp_gripper_thresh=0.4000rad`
- line 93: `completed_episodes=512`
- line 95: `C_tips_during_attached_transport: 512 (1.000)`
- line 97: reset mean `d_xy=0.1732`, `release_z_offset=0.0069`,
  `settled_z_offset=0.0359`, `sz_world_z=1.0000`
- line 98: pre-release mean `d_xy=0.0783`,
  `release_z_offset=0.0770`, `settled_z_offset=0.1060`,
  `sz_world_z=0.2667`
- line 99: release mean `d_xy=0.0739`,
  `release_z_offset=0.0788`, `settled_z_offset=0.1078`,
  `sz_world_z=0.2851`
- line 100: post-settle mean `d_xy=0.0346`,
  `release_z_offset=0.0291`, `settled_z_offset=0.0006`,
  `sz_world_z=0.0194`
- line 101: final mean `d_xy=0.0348`,
  `release_z_offset=0.0292`, `settled_z_offset=0.0006`,
  `sz_world_z=0.0156`
- lines 103-126: sample episodes show release steps mostly 14-31, all classified
  as `C_tips_during_attached_transport`.

stderr lines 1-12 were NVML / observation-group warnings only; no Python
traceback.

## Interpretation

Dominant failure mode: **C. object tips/rotates during attached transport**, not
a solved transport/release primitive.

Critical details:

- Reset starts are physically upright (`sz_world_z=1.0000`) and match the
  intended long-transport distribution.
- By pre-release, mean `sz_world_z` has already collapsed to `0.2667`, so the
  object is mostly sideways before the release event.
- Release is early relative to episode length (`release_step` samples around
  14-31 of 200), and occurs with mean `d_xy=73.9mm` and
  `release_z_offset=78.8mm`, so it is also not a clean near-target height
  release.
- Post-settle/final XY and z can look deceptively better in some episodes
  because the sponge is lying flat; final mean `settled_z_offset=0.6mm` is not a
  success indicator when final `sz_world_z=0.0156`.

Therefore the P7v3 training metric pattern is explained: mean XY improved, but
upright and success stayed near zero because the policy/attach dynamics allow
the sponge to rotate/tip during the attached transport and release window.

## Verdict

FAIL, now diagnosed more specifically:

- P7 `model_499.pt` does not solve attached transport/release.
- The dominant observed failure is upright/orientation collapse during attached
  transport before or at release, with all 512 diagnostic episodes classified as
  `C_tips_during_attached_transport`.
- Do not reward-hack another scalar run before deciding how attached transport
  should preserve or control object orientation.

## Next

1. Keep G2-A collision proxy and the v10 release bridge caveat.
2. Do not add scripted release variants.
3. Do not resume P6/P6v14a release-only training.
4. For the learned branch, inspect why the P7 action sequence drives attached
   orientation collapse: candidate diagnostics are per-step gripper/TCP path,
   quaternion evolution, and whether `_update_grasp_attach` preserves a stale or
   physically invalid sponge orientation during forced TCP pose writes.
5. For the physics branch, keep SurfaceGripper separate until an authored
   gripper/constraint unit test reaches `Closed`.
