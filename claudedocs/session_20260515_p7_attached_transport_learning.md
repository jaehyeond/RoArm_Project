# Session 2026-05-15 — P7 G2-A Attached Transport/Release Learning

## Scope

User direction:

- Do not add scripted release variants.
- Do not keep trying arbitrary SurfaceGripper parent/offset variants.
- Use B200 efficiently while keeping the SurfaceGripper/constraint branch as a
  separate unit-test track.
- Train transport/release from realistic G2-A four-source attached distribution.

## Code Changes

Added P6v17/P7 attached-start curriculum in `roarm_rl/roarm_stack_env.py`:

- Seed0 G2-A post-pick attached state table:
  `G2A_SEED0_ATTACHED_Q_RAD`, `G2A_SEED0_ATTACHED_TCP`,
  `G2A_SEED0_TARGETS`.
- `curriculum_attached_transport_release=True` reset mode:
  starts `_grasped=True`, `_was_grasped=True`, sponge at TCP, wrist_r `+90deg`,
  gripper latch near `26deg`, target sampled from L1.sp1/L1.sp2 layout.
- `reward_phase=7` transport/release-only reward:
  penalizes far closed holding, rewards moving attached sponge toward release
  entry above target, then rewards released settling near target.
- P7 disables the legacy `_get_dones()` P6 `_place_condition` latch, because it
  contaminated P7 `place_success_rate` despite stricter P7 `on_target` being
  near zero.

Added:

- `launch_p6v17_transport_release.sh`
- `sim_scripts/attached_transport_reset_probe.py`

Post-change local/B200 md5:

- `roarm_rl/roarm_stack_env.py` = `996f2afce7de1b3be93ae43ddc349f8e`
- `roarm_rl/train_ppo.py` = `6b0ffdb8365c5e37ced00833c0556c19`
- `sim_scripts/attached_transport_reset_probe.py` =
  `43a04e3cfca763a50d8c856185d14b99`
- `launch_p6v17_transport_release.sh` =
  `2acd462042d0997610fca25ff7a41e21`

## Reset Probe

B200:

- `/tmp/p7v1_attached_reset_probe_v2.out`
- `/tmp/p7v1_attached_reset_probe_v2.err`

Key lines:

- line 63: reset complete.
- line 65: `grasped_frac=1.000`.
- line 66: `was_grasped_frac=1.000`.
- line 67: `d_sponge_tcp_mean_mm=0.00`.
- line 68: `d_xy_mean_mm=175.80`.
- line 69: `d_z_mean_mm=35.95`.
- lines 70-77: sampled seed0 attached starts had d_xy roughly
  `150.9-208.1mm`.

Interpretation: reset distribution is the intended long transport problem, not
a release-only or near-target problem.

## B200 Training Runs

### P7v1 diagnostic

B200 `/tmp/p7v1_diag20.out`.

At iteration 16/20:

- line 584: `p7_xy_offset_mean=0.2391`.
- line 589: `p7_gripper_open_rate=0.0631`.
- line 596: `p7_sponge_height_m=0.1437`.

Verdict: FAIL. P7v1 still drifted toward closed/held behavior and lifted too
high.

### P7v3 diagnostic

B200 `/tmp/p7v3_diag20.out`.

At iteration 16/20:

- line 584: `p7_xy_offset_mean=0.1848`.
- line 589: `p7_gripper_open_rate=0.2110`.
- line 594: `p7_place_success_rate=0.0000`.
- line 596: `p7_sponge_height_m=0.0478`.

Verdict: better than P7v1, but still no complete primitive.

### P7v3 full B200 run

B200:

- `/tmp/p7v3_transport_release.out`
- `/tmp/p7v3_transport_release.err`
- md5 out = `3c2d807f2068669da4767cdc77706653`
- md5 err = `b5e057053e8a5f3af3c42a7fe19ab46e`

Artifacts:

- `$ROARM_B200_ROOT/logs/roarm_rl/roarm_stack_p7v3_g2a_attached_transport_release/model_499.pt`

Near end, iteration 496/500:

- line 14984: `p7_xy_offset_mean=0.0512`.
- line 14985: `p7_release_z_offset_mean=0.0328`.
- line 14986: `p7_settled_z_offset_mean=0.0138`.
- line 14989: `p7_gripper_open_rate=0.8298`.
- line 14991: `p7_on_target_rate=0.0005`.
- line 14993: `p7_upright_rate=0.0576`.
- line 14994: `p7_place_success_rate=0.0007`.

Verdict: FAIL as a complete primitive. It learned a partial transport/release
tendency: mean XY improved from the reset distribution (`175.8mm`) to about
`51mm`, but success stayed around `0.07%` and upright collapsed to about `5.8%`.

## Critical Interpretation

This is not a solved learned release or solved transport/release primitive.

What improved:

- B200 was useful: P7v3 reduced mean XY transport error from ~176mm reset to
  ~51mm by iteration 496.
- The reward no longer produced the previous high-lift closed-hold farm.

What failed:

- The learned behavior opens frequently but does not land upright/on target.
- `p7_on_target_rate` and `p7_place_success_rate` remain near zero.
- Low `upright_rate` indicates release/settle geometry is still physically bad,
  not just a scalar reward-reporting issue.

Next skeptical step:

1. Do not declare P7 success.
2. Do not add random scripted release variants.
3. Either add an evaluation/rollout diagnostic for `model_499.pt` to inspect
   final object pose failure modes, or move to a more structured transport
   primitive with orientation/upright stabilization.
4. In parallel, the SurfaceGripper/constraint branch still requires a real
   authored-asset unit test that reaches `Closed`; quick dynamic prim creation
   already failed.
