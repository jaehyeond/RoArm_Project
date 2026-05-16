# Session 2026-05-17 — P7 env-level attach semantics experiment

## Scope

- Followed `CLAUDE.md` Current-State Protocol and the user-requested boot list.
- Did not use `HANDOFF.md` or `TASKS.md`.
- Did not add scripted release variants.
- Did not random-search SurfaceGripper parent/offset.
- Did not change P7 reward scalar first.
- Did not revert the existing dirty worktree.

## Boot Verification

Pre-code local md5s matched the requested baseline:

- `roarm_rl/chain_skills.py` = `c6e610216197994c6b7d2b6625d87560`
- `launch_chain_topdown.sh` = `b34ef3853ac993a1e2adbaddb420adab`
- `roarm_rl/roarm_stack_env.py` = `996f2afce7de1b3be93ae43ddc349f8e`
- `roarm_rl/train_ppo.py` = `6b0ffdb8365c5e37ced00833c0556c19`
- `launch_p6v17_transport_release.sh` = `2acd462042d0997610fca25ff7a41e21`
- `sim_scripts/attached_transport_reset_probe.py` = `43a04e3cfca763a50d8c856185d14b99`
- `sim_scripts/surface_gripper_transport_probe.py` = `053fced6551ccb02d8a9ea6c04fb4a30`
- `sim_scripts/p7_rollout_failure_diag.py` = `a9743d74886c454b1c161a1bade3df93`
- `sim_scripts/p7_action_tcp_quat_trace.py` = `c54b7892dd06a72f31402ab8dc011b65`
- `sim_scripts/p7_attach_quat_constraint_probe.py` = `a2e16f7683856ead1a9a9eef1da8ea69`

The requested B200 `/tmp` logs existed on B200. Key rechecked lines:

- `/tmp/p7v3_action_tcp_quat_trace.out` lines 245-264: baseline all-env attached tip before open (`first_tip_while_grasped=256/256`, `tip_before_or_at_open=256/256`) and final flat artifact (`sz=0.0759`).
- `/tmp/p7v3_attach_quat_identity_keep.out` lines 145-159: identity+keep reduced pre-release attached tip to `11/256`, but final `d_xy=0.2604`, `sz=0.6434`.
- `/tmp/p7v3_transport_release.out` lines 14984-14994: P7v3 was not solved (`p7_on_target_rate=0.0005`, `p7_upright_rate=0.0576`, `p7_place_success_rate=0.0007`).
- `/tmp/roarm_surface_gripper_transport_probe_v2.out` lines 143/152/164 and v3 lines 144/153/165: quick SurfaceGripper retrofit failed to close/attach.

## Branch Decision

Chose Branch A: controlled env-level attach orientation semantics.

Reason: D019 showed a direct mechanical lever. Runtime `identity+keep` suppressed pre-release attached tip from baseline `256/256` to `11/256`, but old `model_499.pt` still failed placement. That makes a gated env semantic plus fresh training/eval cheaper and more diagnostic than authored SurfaceGripper/constraint work right now. Branch B remains valid but needs asset/API authoring because quick v2/v3 never reached `Closed`.

## Code Changes

- `roarm_rl/roarm_stack_env.py`
  - Added gated config fields:
    - `attach_quat_mode: "preserve" | "identity"`, default `preserve`.
    - `attach_velocity_mode: "zero" | "keep"`, default `zero`.
  - Defaults preserve original behavior exactly.
  - `_update_grasp_attach` still writes sponge xyz to TCP. It now either preserves current quaternion or writes identity, and either zeroes velocity or leaves velocity unchanged.
- `roarm_rl/train_ppo.py`
  - Added CLI overrides `--attach_quat_mode` and `--attach_velocity_mode`.
- `sim_scripts/p7_attach_semantics_env_probe.py`
  - New state-only probe; no policy, no monkey-patch.
  - Injects a tipped quaternion while `_grasped=True`, calls real `_update_grasp_attach`, and reports resulting `sz`, TCP distance, and velocity.
- `sim_scripts/p7_action_tcp_quat_trace.py`
  - Added env-level attach semantics CLI flags for checkpoint evaluation.

Post-change local/B200 md5s:

- `roarm_rl/roarm_stack_env.py` = `47dad11d9f99b007d2ff22ff0fbdbad7`
- `roarm_rl/train_ppo.py` = `a056cb61819deea963e1368b196bf0d4`
- `sim_scripts/p7_attach_semantics_env_probe.py` = `4997a3ec058773004441b74419da114f`
- `sim_scripts/p7_action_tcp_quat_trace.py` = `e6c9424cfe7ffafdf00fe0625f0553f7`

## B200 Smoke

Logs:

- `/tmp/p7_attach_semantics_identity_keep.{out,err}`
- `/tmp/p7_attach_semantics_preserve_zero.{out,err}`

Key lines:

- identity+keep:
  - line 64: `attach_quat_mode=identity attach_velocity_mode=keep`
  - line 65: `grasped_frac=1.000`, initial TCP attach distance `0.000002`
  - line 66: after attach `sz_mean=1.0000`, `d_tcp_mean=0.000000`, `vel_norm_mean=3.0020`
- preserve+zero:
  - line 64: `attach_quat_mode=preserve attach_velocity_mode=zero`
  - line 65: `grasped_frac=1.000`, initial TCP attach distance `0.000002`
  - line 66: after attach `sz_mean=0.5000`, `d_tcp_mean=0.000000`, `vel_norm_mean=0.0000`

Interpretation: the env-level semantic gate is active and default behavior remains the old preserve+zero semantic.

## Fresh P7 Diagnostic

B200 short training:

- `/tmp/p7v4_attach_identity_keep_diag20.{out,err}`
- Checkpoint: `$ROARM_B200_ROOT/logs/roarm_rl/roarm_stack_p7v4_attach_identity_keep_diag20/model_19.pt`

Key lines:

- line 44: `attach_quat_mode: preserve -> identity`
- line 45: `attach_velocity_mode: zero -> keep`
- line 48: `max_iter=20`
- iteration 0 lines 105/112/114/115:
  - `p7_xy_offset_mean=0.1904`
  - `p7_on_target_rate=0.0000`
  - `p7_upright_rate=0.9313`
  - `p7_place_success_rate=0.0000`
- iteration 16 lines 586/593/595/596:
  - `p7_xy_offset_mean=0.3620`
  - `p7_on_target_rate=0.0022`
  - `p7_upright_rate=0.7589`
  - `p7_place_success_rate=0.0000`

Interpretation: early P7v4 under identity+keep is not promising as-is. Upright starts high but transport worsens badly; do not continue to claim this branch is solved without changing the controller/reward curriculum.

## Fresh Checkpoint Trace

B200 eval:

- `/tmp/p7v4_attach_identity_keep_model19_trace.{out,err}`

Key lines:

- line 42: checkpoint `model_19.pt`
- line 44: `attach_quat_mode=identity attach_velocity_mode=keep`
- line 94: env uses `_update_grasp_attach quat_mode=identity velocity_mode=keep`
- line 96: reset `d_xy=0.1722`, `sz=1.0000`, `grasped=1.000`
- lines 338-340: no release/open in 60 steps (`first_open=0/256`, `release_or_open=0/256`)
- lines 341-344: attached tipping still occurs in `187/256`, but much later than baseline; mean tip step lines 350-351 = `33.26`
- line 354: release pose is `nan` because there was no release
- line 355: final `d_xy=0.1488`, `sz=0.9036`

Interpretation: env-level identity+keep improves upright mechanics relative to the old immediate-collapse baseline, but the fresh 20-iter policy learned a closed attached transport/no-release behavior and does not solve P7.

## Verdict

Branch A produced a valid mechanics gate and a useful early disproof:

- The env can now switch attach quaternion/velocity semantics without changing default behavior.
- B200 smoke proves the new semantics are active.
- Fresh P7 identity+keep did not solve transport/release; early training worsened XY and checkpoint trace produced no release/open.

Next useful direction is not to claim attach reset solved P7. Either redesign the P7 controller/reward under the new mechanics to force release/target transport, or move to Branch B authored physics gripper/constraint unit test.
