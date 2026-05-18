# START_HERE.md

Last updated: 2026-05-18 KST (Track A Branch B diagnostic admissible-region wrapper; no constraint integration)

This is the rolling current-state dashboard. Do not treat it as full history.
Durable lessons live in `claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

## Current Truth

The project is two-track:

- **Track A**: existing sim/lab stacking work. Current active line is P7/Branch B
  authored constraint mechanics, isolated/pre-chain units only.
- **Track B**: CoRL 2026 paper sprint. Keep separate unless the user explicitly
  asks to switch tracks.

Do **not** use `HANDOFF.md` or `TASKS.md` as current state.

## Track A Latest

Latest session:

- `claudedocs/session_20260518_p7_branch_b_preclose_clearance_strategy.md`

What changed:

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_admissible_region_probe.py`
  md5 `89ad48b6ebdec076d6f58e330a9131f9`. This is a diagnostic-only wrapper over
  existing selector logs; it does not execute training, constraints,
  SurfaceGripper, transport, release, env/train/chain default edits, or gate
  tuning. B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_admissible_region_b200.{out,err}`.
- The wrapper applies a conservative non-deployed rule:
  `min_side_margin_m=0.002000`, max below-top side depth `-0.003000`,
  final/realized outside-AABB required for below-top side-edge, zero-margin
  side-edge rejected, below-top inside-footprint rejected, exact convergence
  still under unchanged `0.003000m` gate, and far-sponge below-top kept as a
  no-contact control.
- B200 stdout lines 1-3 confirm strict diagnostic scope and the rule. Lines 4-11
  classify the compact matrix exactly as expected: 0mm boundary rejected despite
  selector acceptance; 0.1mm observed pass rejected as below the conservative
  2mm margin; 2mm/-3mm side-edge accepted; 2mm/-4mm rejected for depth/exact
  failure; top-tangent and above-top controls accepted; nominal below-top
  inside-footprint and far-sponge controls rejected. Line 12 reports
  `expected_matches=8/8`, `attach_calls_all_zero=YES`,
  `below_inside_segments_clean_all_empty=YES`, and diagnostic success `YES`.
- B200 wrapper stderr was empty. B200 process check after the run showed no
  matching `isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process.
- This is not a deployed policy and not chain integration. It only shows the
  proposed diagnostic filters explain the accumulated pass/fail evidence without
  changing selector gates.

- Reused the unchanged selector md5 `aa24ef00acbb9d8cd0aeee061b08f85f` with
  fixed `--side_margin_m 0.0020` and side top margins
  `-0.5/-1.0/-1.5/-2.0/-3.0/-4.0/-6.0mm`. No code, gate, env/train/chain
  default, constraint, SurfaceGripper, transport, release, or training change
  was made. B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_{neg0p5,neg1p0,neg1p5,neg2p0,neg3p0,neg4p0,neg6p0}_b200.{out,err}`.
- All seven runs kept the side target outside AABB on line 52 and in the final
  segment. Line 1055 stayed exact-clean through -3mm: final errors
  `0.000548/0.000702/0.001074/0.001504/0.002409m`, exact `YES`, no top clamp,
  mechanically valid `YES`, clean `YES`.
- At -4/-6mm, line 1055 still reports outside-AABB, `top_clamped=NO`, and
  mechanically valid `YES`, but exact convergence fails (`0.003346/0.005177m`,
  exact `NO`, clean `NO`); lines 1058-1059 mark diagnostic `NO`.
- Therefore D049: at 2mm outside-AABB, side-edge below-top depth is diagnostic
  exact-clean through about -3mm under the unchanged 3mm gate, and loses exact
  convergence by -4mm. This is not an inside-footprint clamp failure; it is an
  exact-convergence/residual-error limit.
- All seven stderr files line 1-4 are only known cpufreq/NVML/Fabric warnings;
  no traceback/exception was found.

- Reused the unchanged selector md5 `aa24ef00acbb9d8cd0aeee061b08f85f` with
  `--side_top_margin_m -0.0015` and fine side margins
  `0.1/0.2/0.3/0.4/0.5mm`. No code, gate, env/train/chain default,
  constraint, SurfaceGripper, transport, release, or training change was made.
  B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_{0p1,0p2,0p3,0p4,0p5}_b200.{out,err}`.
- All five stdout files line 42 confirm unchanged exact gate `0.003000m`,
  reduction gate reference-only, and `side_top_margin_m=-0.001500`; line 46
  repeats the unchanged selector rule.
- All five runs preserve controls: line 179 keeps the nominal below-top
  inside-footprint baseline clamped (`0.023923m`, `top_clamped=YES`,
  mechanically valid `NO`), line 279 keeps the far-sponge below-top no-contact
  control exact-converged but non-candidate (`0.000854m`), and lines 473/667
  keep top-tangent/above controls clean.
- Fine positive margins were clean: line 1055 for 0.1/0.2/0.3/0.4/0.5mm reports
  final errors `0.001241/0.001232/0.001222/0.001213/0.001203m`, exact
  convergence `YES`, no top clamp, mechanically valid `YES`, clean `YES`, and
  final target outside AABB. Lines 1057-1059 keep
  `below_inside_segments_clean=[]`, `attach_calls=0`, no NaN/done, and
  diagnostic success `YES`.
- Therefore D048: the observed deterministic B200 boundary is between 0.0mm and
  0.1mm for `side_top_margin_m=-0.0015`, but 0.1mm is only the minimum tested
  positive pass, not a deployment/chain margin.
- stderr lines 1-4 in all five runs are only known cpufreq/NVML/Fabric warnings;
  no Python traceback or diagnostic `NO` was found.

- Reused the unchanged selector md5 `aa24ef00acbb9d8cd0aeee061b08f85f` with
  `--side_top_margin_m -0.0015` and side margins
  `0.0/0.5/1.0/2.0/4.0/6.0mm`. No code, gate, env/train/chain default,
  constraint, SurfaceGripper, transport, release, or training change was made.
  B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_{0p0,0p5,1p0,2p0,4p0,6p0}_b200.{out,err}`.
- All six stdout files line 42 confirm unchanged exact gate `0.003000m`,
  reduction gate reference-only, and `side_top_margin_m=-0.001500`; line 46
  repeats the unchanged selector rule.
- All six runs preserve controls: line 179 keeps the nominal below-top
  inside-footprint baseline clamped (`0.023923m`, `top_clamped=YES`,
  mechanically valid `NO`), line 279 keeps the far-sponge below-top
  no-contact control exact-converged but non-candidate (`0.000854m`), and lines
  473/667 keep top-tangent/above controls clean.
- The 0.0mm boundary is the trap. Line 52 accepted the planned side target as
  outside AABB (`target_dy_sponge_m=0.011033`), but line 1055 reports the final
  target inside realized AABB with exact convergence `YES`, `top_clamped=NO`,
  mechanically valid `NO`, and clean `NO`; lines 1058-1059 mark accepted
  candidates clean `NO` and diagnostic `NO`.
- Positive margins were clean: line 1055 for 0.5/1/2/4/6mm reports final errors
  `0.001203/0.001156/0.001074/0.000899/0.000529m`, exact convergence `YES`,
  no top clamp, mechanically valid `YES`, clean `YES`, and final target outside
  AABB. Lines 1057-1058 keep `below_inside_segments_clean=[]`, `attach_calls=0`,
  no NaN/done, and no attach/release physics claim.
- Therefore D047: shallow below-top side-edge needs positive outside-AABB margin
  and realized/final AABB validation. Zero-margin boundary is not robust and must
  not be treated as a valid contact candidate.
- stderr lines 1-4 in all six runs are only known cpufreq/NVML/Fabric warnings;
  no Python traceback was found.

- Ran the unchanged selector md5 `aa24ef00acbb9d8cd0aeee061b08f85f` with
  `--side_top_margin_m -0.0015` as an outside-AABB side-edge below-top guard.
  No code changed for this diagnostic. B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_selector_side_below_guard_b200.out`
  and `.err`.
- Side-below guard stdout lines 41-46 confirm strict pre-integration scope,
  unchanged exact gate `0.003000m`, reduction gate reference-only, no MOVE
  commands, all IK targets converged, and the unchanged selector rule.
- Line 52 accepts `candidate_side_edge_margin_2mm_top_margin_neg1p5mm` only
  because it is outside AABB, despite `final_target_top_class=below`.
- Line 1055 shows that shallow outside-AABB below-top side-edge exact-converges:
  `final_target_tcp_error_m=0.001074`, `exact_converged=YES`, `top_clamped=NO`,
  `mechanically_valid_target=YES`, and clean `YES`.
- Lines 1057-1059 keep the nominal below-top inside-footprint baseline clamped,
  report `below_segments_clean` only for the outside-AABB side-edge segment,
  keep `below_inside_segments_clean=[]`, `attach_calls=0`, no NaN/done, and no
  attach/release physics claim.
- This refines the diagnostic rule without relaxing the ban: below-top inside
  footprint remains invalid; shallow below-top side-edge remains diagnostic-only
  and must stay outside AABB.

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_selector_guard_probe.py`
  md5 `e50f7dfcb5651507b0c200af1299f171`. This is a diagnostic-only wrapper
  around the existing selector that drives an adversarial near-top invalid final
  target (`top_margin_m=-0.001500`) to prove below-top/inside-footprint targets
  are rejected even if the unchanged 3mm exact gate can pass.
- Latest guard B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_selector_guard_b200.out` and
  `/tmp/p7_branch_b_roarm_chain_preclose_selector_guard_b200.err`.
- Guard stdout lines 2-4 confirm strict pre-integration scope and the intended
  guard case. Lines 43-49 show the underlying selector run, unchanged exact gate
  `0.003000m`, reduction gate reference-only, no MOVE commands, and the same
  selector rule.
- Guard line 52 rejects `candidate_top_tangent_margin_neg1p5mm` before
  interpretation as `below_top_inside_footprint_invalid`.
- Guard line 476 is the important trap: the invalid near-top below/inside target
  has `final_target_tcp_error_m=0.001268` and `exact_converged=YES`, but it is
  still `top_clamped=YES`, `mechanically_valid_target=NO`, and
  `clean_realized_without_reduction_artifact=NO`.
- Guard lines 477 and 1060 keep that exact-converged invalid candidate rejected;
  lines 1060-1062 report accepted clean candidates are only above-top and
  side-edge, `below_inside_segments_clean=[]`, `attach_calls=0`, no NaN/done, no
  attach/release physics claim, and diagnostic success.
- Guard stderr lines 1-4 are only known cpufreq/NVML/Fabric warnings.

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  md5 `aa24ef00acbb9d8cd0aeee061b08f85f`. This converts D043/D044 into a
  diagnostic-only selector/check layer; it does not integrate constraints,
  SurfaceGripper, transport, release, training, tuning, or env/train/chain
  defaults.
- Latest B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_candidate_selector_b200.out` and
  `/tmp/p7_branch_b_roarm_chain_preclose_candidate_selector_b200.err`.
- Selector stdout lines 41-46 confirm strict pre-integration scope, unchanged
  exact gate `0.003000m`, reduction gate reference-only, no MOVE commands, and
  the explicit rule: accept final above/tangent or side-edge outside-AABB;
  reject below-top inside-footprint; treat far-sponge below-top as no-contact
  control only.
- Lines 47-52 show candidate decisions before interpretation: nominal below-top
  baseline and upward-then-below are rejected as below-top inside-footprint;
  far-sponge below-top is rejected as a contact candidate; top-tangent +0.5mm,
  above-top +1.0mm, and side-edge +2mm outside-AABB are accepted.
- Baseline line 179 preserves the failure/top clamp:
  `final_target_tcp_error_m=0.023923`, `exact_converged=NO`,
  `top_clamped=YES`, `mechanically_valid_target=NO`.
- Far-sponge control line 279 exact-converges (`0.000854m`) but remains
  `mechanically_valid_target=NO` in the selector because it is no-contact.
- Accepted candidates exact-converged cleanly: top-tangent line 473
  (`0.000920m`), above-top line 667 (`0.000921m`), and side-edge line 1055
  (`0.000915m`), with no top clamp.
- Invalid upward-then-below line 861 still fails/top-clamps
  (`0.023470m`, `top_clamped=YES`) and line 862 keeps the strategy rejected.
- Aggregate lines 1057-1059 report accepted contact candidates clean, rejected
  controls rejected, `below_inside_segments_clean=[]`, `attach_calls=0`,
  no NaN/done, no attach/release physics claim, and diagnostic success.
  stderr lines 1-4 are only known cpufreq/NVML/Fabric warnings.

Previous geometry sweep:

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_geometry_sweep_probe.py`
  md5 `95b4a8a317a9fb176c7ed258229925e5`, after the earlier
  `sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py`
  md5 `5be8cfb8c1a58f6de43f431db0befff4`.
- Both diagnostics are pre-integration only: no constraint prim insertion,
  no fixed/dynamic integration, no SurfaceGripper, no attached transport, no
  transport target, no release, no scripted release variant, no P7 training/
  tuning, no diagnostic gate tuning, and no env/train/chain default edits.
- Latest B200 logs:
  `/tmp/p7_branch_b_roarm_chain_preclose_geometry_sweep_v2_b200.out` and
  `/tmp/p7_branch_b_roarm_chain_preclose_geometry_sweep_v2_b200.err`.
- Lines 41-43 confirm strict scope, unchanged 3mm exact target gate, reduction
  gate reference-only, no MOVE/transport/release, and the tested geometry ranges:
  top margins +0.2/+0.5/+1.0/+2.0mm, clearance heights +12/+24/+36mm, and side
  outside-AABB margins +2/+6/+12/+18mm.
- line 44 confirms all IK targets converged before simulation.
- The nominal below-top inside-footprint baseline still fails and top-clamps:
  line 172 has `final_target_tcp_error_m=0.023923`, `exact_converged=NO`,
  `top_clamped=YES`, `clean_realized_without_reduction_artifact=NO`, and
  `final_target_tcp_minus_sponge_oriented_top_m=-0.022821`.
- The same below-top q target realizes only when the sponge is far: line 272 has
  `final_target_tcp_error_m=0.000854`, exact convergence `YES`, and target
  outside AABB. This is explicitly a no-contact control, not permission to use
  below-top targets in nominal contact geometry.
- Final top margins exact-converged cleanly: +0.2mm line 466
  (`0.000727m`), +0.5mm line 660 (`0.000920m`), +1.0mm line 854
  (`0.000921m`), and +2.0mm line 1048 (`0.000921m`), with no top clamp.
- Clearance height mattered less than final geometry in the tested range: +12,
  +24, and +36mm clearance with final +0.5mm top margin exact-converged on lines
  1242/1436/1630.
- Side-edge outside-AABB tangent candidates exact-converged at +2/+6/+12/+18mm
  outside width on lines 1824/2018/2212/2406.
- Aggregate lines 2408-2409 report `below_inside_segments_clean=[]`,
  `below_top_inside_targets_realize_cleanly=NO`, contact candidates excluding
  far-sponge control, `far_control_is_no_contact_control=YES`, `attach_calls=0`,
  no NaN/done, and no attach/release claim. stderr lines 1-4 are only known
  cpufreq/NVML/Fabric warnings.
- Therefore D044: final pre-close target geometry dominates tested clearance
  height once the final target remains above/tangent. Below-top inside-footprint
  targets remain banned.

Previous detailed dead-zone session:

- Added and extended `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`.
  Initial md5 `7d7c3405e6be240500b7251df91f26e3`; latest diagnostic
  instrumentation md5 `e0e84e481c3be8be7777a85ef2465c57`.
- This diagnostic is local/pre-integration only: no constraint prim insertion, no
  fixed/dynamic integration, no SurfaceGripper, no attached transport, no
  transport target, no release, no P7 training/tuning, and no default edits.
- B200 logs:
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_b200.out` and
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zhi_b200.out`.
- Follow-up B200 logs:
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zfine_b200.out`,
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zmicro_b200.out`,
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_xy_b200.out`, and
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_ycheck_b200.out`.
- Pose/top-controlled B200 logs:
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_log_zboundary_b200.out`,
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zboundary_b200.out`,
  and
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zmicro2_b200.out`.
- Lines 41-44 of the default log confirm strict scope, pre-transport execution,
  and that contact sensors are unavailable so proximity/AABB/top-height proxies
  are used.
- Line 72 records controller context:
  `action_scale=0.100000`, `null_action_max_abs=0.000000`, soft limits, and arm/
  gripper drive fields.
- Nominal sponge/open gripper still fails despite target delivery:
  default line 85 reports `set_target_seen=YES`,
  `best_data_target_attr_diff_rad=0.00000004`,
  `max_realized_tcp_delta_m=0.001144`,
  `final_target_tcp_error_m=0.023952`,
  `final_shoulder_error_deg=5.035108`, `target_realized=NO`, `_grasped=NO`, and
  target TCP inside the sponge AABB about `22.771mm` below sponge top. Default
  line 97 shows direct set+sim-step also fails.
- Same robot q/target with sponge far realizes:
  default line 110 reports env-step `target_realized=YES`,
  `final_target_tcp_error_m=0.000850`, `final_shoulder_error_deg=0.114004`;
  line 122 reports direct set also realizes.
- Higher z offsets refine the boundary: default +3/+6/+12mm nominal-sponge
  variants fail or remain insufficient (lines 135, 160, 185), but the high-z
  cross-check realizes +18/+24/+30mm (lines 135/147, 160/172, 185/197). +24/+30mm
  place the target at/above the sponge top; +18mm is still slightly below top but
  realizes.
- Sub-threshold partial close at nominal pose does not rescue (default and high-z
  lines 210/222).
- High-z lines 223-224 aggregate the refined split:
  `env_realized_conditions=['far_sponge_open',
  'nominal_sponge_z_plus_18mm_open', 'nominal_sponge_z_plus_24mm_open',
  'nominal_sponge_z_plus_30mm_open']`,
  `env_failed_conditions=['nominal_sponge_open',
  'nominal_sponge_partial_close']`,
  `sponge_far_realizes_nominal_fails=YES`,
  `higher_z_realizes_nominal_fails=YES`, and `direct_set_also_fails_nominal=YES`.
- Fine/micro z follow-up tightens the boundary. zfine lines 135/147 show +13mm
  env/direct pass the reduction-based realization gate. zmicro line 43 tested
  +12.0/+12.25/+12.5/+12.75/+13.0mm; lines 136/148, 161/173, 186/198, and
  211/223 show +12.0 through +12.75mm fail, while lines 236/248 show +13.0mm
  env/direct pass. Aggregate lines 274-275 show only far sponge and +13mm realize
  among the micro-z conditions.
- Do not overread that +13mm pass: zmicro line 236 still has
  `final_target_tcp_error_m=0.011054`; the current gate is meaningful error
  reduction, not exact 3mm target convergence.
- Horizontal offset follow-up is posture-confounded. Wide xy line 499 reported
  y -25mm and y +15mm realization, but targeted y-check line 136 did not
  reproduce y -25mm. y-check line 186 repeats y +15mm realization only with a
  lower settled sponge top (`start_sponge_top_z_m=0.034500`), while y +20mm line
  211 fails with the usual top (`0.047000`). Do not infer AABB-outside success.
- Pose/top-controlled follow-up refines the boundary again. The probe now logs
  sponge quaternion/upright top/oriented top and can reassert sponge pose before
  delivery. Controlled reassert run line 42 confirms
  `reassert_sponge_before_delivery=YES`, `reassert_sponge_z_m=0.0235`; reassert
  lines 76/130/157/184 show identity quat and top `0.047000m`. In that controlled
  state, lines 141/154 show +12.75mm still fails, lines 168/181 show +12.875mm
  passes, and lines 195/208 show +13.0mm passes.
- Controlled micro2 narrows the current reduction-gate transition:
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zmicro2_b200.out`
  line 42 confirms the same pose reassert; lines 141/154 show +12.8125mm
  env/direct fail; lines 168/181 show +12.84375mm env/direct pass; lines 195/208
  show +12.875mm pass. Aggregate lines 236-237 match that split.
- Do not overread this as a robust grasp solution. The passing controlled cases
  still have about `0.011m` final TCP error; this is a thin diagnostic
  reduction-gate boundary, not exact command convergence.
- Follow-up stall trace and nudge-direction diagnostics sharpen D041 into D042.
  The probe now prints per-step joint/target/top trace fields and nudge sweeps;
  latest md5 is `e0e84e481c3be8be7777a85ef2465c57`.
- B200 stall trace
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_stall_trace_zmicro_b200.out` line 42
  confirms unchanged gates, `trace_every_step=YES`, and controlled pose reassert.
  Lines 421/552/683 show +12.8125 fail and +12.84375/+13.0 pass all finish with
  TCP clamped at the sponge oriented top (`final_tcp_minus_sponge_oriented_top_m`
  about `-0.000043/-0.000036/-0.000018`) while target TCP remains about
  `10.5-10.7mm` below top. The pass/fail split is shoulder-error reduction, not
  below-top TCP convergence. Lines 860-862 aggregate the unchanged split and no
  attach/release claim.
- B200 nudge-direction run
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_nudge_direction_b200.out` lines 42-43
  tests controlled `[-5,+2.5,+5]deg` shoulder nudges with no z-offset variants.
  Nominal `-5deg` realizes because the target is above top (line 87), while
  nominal `+2.5deg` and `+5deg` below-top targets fail and clamp near top (lines
  114 and 141). Far-sponge +2.5/+5deg still realize (lines 195 and 222).
- Therefore D042: the current blocker is local sponge-top contact equilibrium
  around nominal pre-close geometry. Reduction-gate passes are not useful
  below-top command realization.
- Therefore D038/D039/D040/D041 are refined by D042: the current blocker is local
  contact/clearance/posture around nominal sponge/grasp geometry, not broad target
  delivery, not post-latch-only, and not pure low-pose drive failure. Offset-
  preserve moving behavior remains untested.
- Previous approach-stage target-delivery probe added
  `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
  md5 `ebe8eddafd4c6f35c28e5b79a82511b3`.
- Added `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
  md5 `ebe8eddafd4c6f35c28e5b79a82511b3`.
- This approach-stage target-delivery probe is diagnostic-local only: no
  constraint prim insertion, no fixed/dynamic integration, no SurfaceGripper, no
  attached transport, no transport target, no release, no P7 training, and no
  env/train/chain default edits.
- B200 v2 log:
  `/tmp/p7_branch_b_roarm_chain_approach_target_delivery_v2_b200.out`.
- Lines 41-43 confirm strict scope and that execution remains pre-transport:
  `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Line 72 records the controller context:
  `action_scale=0.100000`, `null_action_max_abs=0.000000`, and soft limits.
- The same +5deg shoulder nudge reaches `_robot.set_joint_position_target()` and
  Articulation target fields at every tested stage (`set_target_seen=YES`,
  `best_data_target_attr_diff_rad` `0.00000004-0.00000009rad`).
- HOME/early/high/hover realize the nudge under `env.step(null_action)`:
  line 87 HOME `target_realized=YES`, final nudge error `0.109396deg`;
  line 115 early PRE_MOVE `target_realized=YES`, final nudge error `0.106804deg`;
  line 143 high `target_realized=YES`, final nudge error `0.084780deg`;
  line 171 hover `target_realized=YES`, final nudge error `0.105476deg`.
- The grasp-before-CLOSE/open-gripper stage still fails despite target delivery:
  line 187 proves the target is nonzero and within soft/analytic limits
  (`expected_tcp_delta_m=0.024271`, limits OK); line 199 reports env-step
  `set_target_seen=YES`, `best_data_target_attr_diff_rad=0.00000004`,
  `final_target_tcp_error_m=0.023947`, `final_nudge_joint_error_deg=5.042476`,
  `tcp_target_reduced=NO`, `nudge_joint_error_reduced=NO`,
  `target_realized=NO`, `grasped=NO`.
- Direct set+sim-step at the same grasp-before-CLOSE stage does not rescue:
  line 211 reports `set_target_seen=YES`, `max_realized_tcp_delta_m=0.000108`,
  `final_target_tcp_error_m=0.023927`, `final_nudge_joint_error_deg=5.044027`,
  and `target_realized=NO`.
- Line 213 aggregates `env_realized_stages=['settled_home', 'early_pre_move',
  'high', 'hover']`, `env_failed_stages=['grasp_before_close_open']`,
  `direct_rescue_stages=[]`, `home_high_realize_grasp_fails=YES`, and
  `latch_seen=NO`; line 214 reports `broader_command_realization_blocker=NO` and
  `local_grasp_pose_only_blocker=YES`.
- Therefore D037 is refined: the current blocker is not broad articulation target
  delivery/realization and not post-latch-only. It is a local grasp-pose command
  realization failure before CLOSE, with gripper open and `_grasped=NO`.
  Offset-preserve moving behavior remains untested.
- Previous target-delivery probe added
  `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
  md5 `aad6398a9d47fef5c80efbd212e619d8`.
- Its B200 v3 log
  `/tmp/p7_branch_b_roarm_chain_post_latch_target_delivery_v3_b200.out` proved
  the same grasp-pose 5deg target was delivered before CLOSE (lines 83-85), after
  CLOSE/latch (lines 110-112), and by direct set (line 134), but was not realized
  in any of those grasp-pose comparisons. The new approach-stage probe refines
  that D037 result to a grasp-pose-local realization blocker.
- Previous executor probe added
  `sim_scripts/p7_branch_b_roarm_chain_post_latch_micro_executor_probe.py`
  md5 `c74d92816df12953c26fed577656840e`.
- B200 marker-only 4mm TCP micro target was nonzero but not realized:
  `/tmp/p7_branch_b_roarm_chain_post_latch_micro_executor_marker_only_b200.out`
  line 87 reports `delta_q_norm_deg=0.790232` and
  `expected_tcp_delta_m=0.003511`; lines 88-93 show
  `robot_dof_targets` were not overwritten, but line 94 reports
  `realized_motion_seen=NO`, `executor_reached=NO`,
  `max_realized_tcp_delta_m=0.000080`, and success `NO` on lines 95-96.
- B200 marker-only 5deg joint-nudge cross-check also did not realize motion:
  `/tmp/p7_branch_b_roarm_chain_post_latch_micro_executor_joint_nudge_b200.out`
  line 87 reports `delta_q_max_abs_deg=5.000000` and
  `expected_tcp_delta_m=0.024271`; line 94 reports
  `targets_not_overwritten=YES`, but `realized_motion_seen=NO`,
  `executor_reached=NO`, `max_realized_tcp_delta_m=0.000206`,
  `min_joint_error_max_deg=4.992061`, and success `NO` on lines 95-96.
- Therefore the failed micro-motion result remains uninterpretable as
  offset-preserve moving behavior; the robot did not realize the commanded target.
- Previous micro-motion probe added
  `sim_scripts/p7_branch_b_roarm_chain_handoff_micro_motion_probe.py`
  md5 `a7ed4387e0ab1ce5b95de08f59c2eb52`.
- The probe reuses the conservative stream and gated scheduling, executes only
  `PRE_MOVE* -> CLOSE`, holds the grasp pose briefly, then attempts tiny TCP
  perturbations around the grasp pose. It compares current TCP-center
  pose-write, marker-only, and TCP-offset-preserving pose-write. It does not
  insert constraint prims, integrate fixed/dynamic constraints, attach
  SurfaceGripper, go to the transport target, run release, run P7 training, or
  edit env/train/chain defaults.

B200 evidence:

- Logs: B200
  `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_{posewrite_tcp,marker_only,offset_preserve_posewrite}_b200.{out,err}` and
  `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_offset_preserve_posewrite_d8mm_b200.{out,err}`.
- Line 41 in each stdout confirms scope: no constraint prim insertion, no
  fixed/dynamic integration, no SurfaceGripper, no attached transport, no
  release marker, no P7 training, no default edits, `transport_target=NO`,
  `micro_motion_not_transport=YES`, and `claim_attach_success=NO`.
- Line 43 confirms source stream truncation before MOVE:
  source `events_total=44`, executed events `39`, `pre_move_cmds=38`,
  `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Current TCP-center pose-write baseline still fails before micro-motion:
  `posewrite_tcp` line 83 reports `target_error_m=0.015684`,
  `tcp_step_m=0.016131`, `pose_drift_m=0.017552`,
  `sponge_speed_mps=1.696947`, `quat_angle_deg=21.267`; lines 85-87 report
  `post_latch_hold_ok=NO`, `micro_motion_ok=NO`, success `NO`.
- Marker-only passes the short stationary hold but does not reach the first 4mm
  micro target: lines 88-91 keep `target_error_m` around `0.004764-0.004765`,
  and lines 92-94 report `micro_events_done=1`, `micro_events_planned=4`,
  `micro_motion_ok=NO`, success `NO`. This remains a negative control, not
  attach evidence.
- Offset-preserving pose-write also passes the short stationary hold but does
  not reach the first 4mm micro target: lines 88-94 report
  `micro_max_target_error_m=0.004764`, `micro_motion_ok=NO`, success `NO`.
- An 8mm offset-preserve cross-check still does not reach the first micro target:
  d8mm lines 88-94 report `micro_max_target_error_m=0.008699`,
  `micro_motion_ok=NO`, success `NO`.

Interpretation:

- Current planner kinematics can provide a contract-compatible TCP event stream
  only with explicit conservative resampling.
- The existing raw planner waypoints/targets are too coarse, and exact 10mm
  resampling can still fail due to FK/IK realized-step error; use a safety
  margin before any approved integration design.
- The real Isaac/RoArm articulation can execute the conservative stream under
  a realized-TCP gated scheduler; a one-sim-step-per-command assumption remains
  false from the previous dynamics probe.
- Nominal passive approach/close timing did not show pre-close sponge push or
  pre-close env latch in this one-sponge diagnostic. CLOSE produced only an env
  kinematic latch marker at the gripper threshold step.
- The immediate latch marker step itself did not jump, but the first stationary
  post-close hold step under the current env kinematic attach boundary produced
  a large target/TCP violation, sponge pose drift, velocity spike, and quaternion
  change. Therefore the current env `_grasped` attach boundary is not a stable
  post-close handoff surface for chain transport.
- The attribution matrix points to `_update_grasp_attach` pose-write as the
  proximate trigger. Velocity mode and quaternion mode did not rescue the
  failure; disabling only pose-write while keeping the latch marker allowed the
  stationary hold to pass. This is marker-only evidence, not attach physics.
- The handoff-model matrix narrows the trigger further: snapping the sponge
  center to TCP is the bad local handoff geometry. Waiting before the same snap
  only delays failure, and one-shot center align still fails. Preserving the
  latch-time TCP-to-sponge offset avoids the stationary post-close hold failure.
- The micro-motion probe did not validate moving offset-preserve behavior:
  marker-only and offset-preserve both survived short stationary hold, but 4mm
  and 8mm post-close `plus_x` perturbation targets were not reached.
- The micro-executor probe showed target buffers were not overwritten by null
  action/action scaling. The follow-up target-delivery probe goes deeper:
  `_robot.set_joint_position_target()` and Articulation `joint_pos_target` receive
  the watched 5deg target, but the target is not realized before CLOSE, after
  CLOSE/latch, or with direct set+sim-step. Treat this as a grasp-pose target
  realization blocker, not attach physics evidence and not moving offset-preserve
  evidence.
- This is **not P7 success** and **not constraint integration**. It does not
  validate object attachment physics, release physics, attached transport, or
  constraint insertion inside the chain. Offset-preserving stationary PASS is
  only a local kinematic handoff diagnostic, and offset-preserving micro-motion
  is still unvalidated.
- Any actual RoArm chain integration still needs explicit user approval and a
  new falsifiable gate.

## Previous Track A Evidence To Preserve

- Previous mock chain-command contract passed:
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_chain_contract.md`;
  B200 lines 129-131 report target errors `0.001468`, release drop `0.338178`,
  and all gates YES.
- Previous timing dry-run remains core evidence:
  `claudedocs/session_20260517_p7_branch_b_roarm_chain_timing_resample.md`.
- Mock-TCP interface and dynamic-anchor target tracking passed in isolation:
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_interface_probe.md`,
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_target_tracking.md`.
- SurfaceGripper still must not be attached to the RoArm chain:
  `claudedocs/session_20260517_p7_branch_b_fixed_constraint_unit.md`; B200
  `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.out` lines 111-113 and
  145-149 show canonical cuboid and RoArm sponge both fail Closed gates.
- Kinematic pose-write fixed-joint micro-move is killed:
  `claudedocs/session_20260517_p7_branch_b_fixed_constraint_micro_move.md`; B200
  lines 59-71 show anchor motion while sponge stays, and lines 103-105 fail.
- Open-loop dynamic velocity anchor coupled but overshot about 2x:
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_constraint.md`.
- Previous RoArm chain-side contract dry-run remains useful evidence:
  `claudedocs/session_20260517_p7_branch_b_roarm_chain_contract_dryrun.md`.
- Previous command-stream dry-run remains core evidence:
  `claudedocs/session_20260517_p7_branch_b_roarm_chain_command_stream.md`.
- Previous RoArm articulation timing remains core evidence:
  `claudedocs/session_20260517_p7_branch_b_roarm_chain_dynamics_timing.md`.
- Previous passive close timing remains useful but is superseded by the
  post-close boundary failure:
  `claudedocs/session_20260518_p7_branch_b_passive_contact_close_timing.md`.
- Previous stationary handoff-model matrix remains the source for D036:
  `claudedocs/session_20260518_p7_branch_b_handoff_model_probe.md`.

## Track B Status

- Must-read: `claudedocs/session_20260517_corl2026_paper_track_pivot.md`.
- CoRL 2026 full paper deadline was estimated as 2026-05-28 AoE; user must verify
  on corl.org directly.
- Candidate paper pipeline remains separate from Track A unless explicitly merged.

## Do-Not-Repeat Rules

- Do not claim P7 success.
- Do not tune P7 scalar/threshold/release-guidance blindly.
- Do not run structured A curriculum long training from the killed smoke.
- Do not resume random SurfaceGripper parent/offset search.
- Do not add scripted release variants.
- Do not attach SurfaceGripper to the RoArm chain.
- Do not integrate fixed/dynamic constraints into the RoArm chain yet.
- Do not proceed from CLOSE into attached transport using the current env
  `_grasped` kinematic attach boundary as a valid handoff surface.
- Do not treat marker-only or offset-preserving stationary hold pass as attach
  physics, release physics, attached transport, or constraint validation.
- Do not treat the failed post-close micro-motion probe as evidence that
  offset-preserving attached MOVE is valid; the micro target was not reached.
- Do not treat the failed 5deg target-delivery probe as offset-preserve failure;
  the same grasp-pose nudge fails before CLOSE with `_grasped=NO`.
- Do not describe the current command-realization blocker as broad articulation
  targeting failure: HOME/early/high/hover realize the same +5deg shoulder nudge;
  the surviving failure is local to the grasp-before-CLOSE/open-gripper pose.
- Do not describe the current grasp-pose blocker as pure low-pose drive failure:
  the same q/target realizes with the sponge far, and +18/+24/+30mm local z
  variants realize. Treat it as contact/proximity-shaped around nominal sponge/
  grasp geometry until narrower evidence says otherwise.
- Do not describe horizontal offset as a clean fix: the reproducible y +15mm pass
  is confounded by a lower settled sponge top/posture, and y -25mm did not
  reproduce.
- Do not call the +12.84375/+12.875mm reduction-gate pass a solved grasp pose:
  final TCP error remains about 11mm and no attach/transport/release physics was
  exercised.
- Do not treat the +12.84375/+13.0mm reduction-gate pass as useful below-top
  command convergence: the TCP remains clamped at the sponge top while the target
  is still about 10-11mm below top.
- Do not change B200 system NVIDIA symlinks; use per-run
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

## Current Direction

Active pivot: Track A P7/Branch B, isolated/pre-integration mechanics and chain-side timing.

Next concrete action: stay pre-integration. The conservative diagnostic wrapper
now explains the latest side-edge pass/fail matrix, so the next useful work is
either a tighter read-only/log-only audit of the admissible-region rule or an
explicitly approved new pre-close diagnostic. Still no SurfaceGripper, no RoArm
chain constraint insertion, no attached transport, and no release physics claims
unless explicitly approved.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` D042-D049
4. `claudedocs/EXPERIMENT_LEDGER.md` latest Branch B rows
5. `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_chain_contract.md`
6. `claudedocs/session_20260517_p7_branch_b_roarm_chain_command_stream.md`
7. `claudedocs/session_20260517_p7_branch_b_roarm_chain_dynamics_timing.md`
8. `claudedocs/session_20260518_p7_branch_b_passive_contact_close_timing.md`
9. `claudedocs/session_20260518_p7_branch_b_post_close_latch_boundary.md`
10. `claudedocs/session_20260518_p7_branch_b_handoff_model_probe.md`
11. `claudedocs/session_20260518_p7_branch_b_handoff_micro_motion_probe.md`
12. `claudedocs/session_20260518_p7_branch_b_post_latch_micro_executor.md`
13. `claudedocs/session_20260518_p7_branch_b_post_latch_target_delivery.md`
14. `claudedocs/session_20260518_p7_branch_b_approach_target_delivery.md`
15. `claudedocs/session_20260518_p7_branch_b_grasp_pose_deadzone.md`
16. The B200/local logs cited above, with line numbers
