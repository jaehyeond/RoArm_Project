# Session 2026-05-18 - P7 Branch B pre-close clearance strategy diagnostic

## Scope Guard

- Continued Track A P7/Branch B only.
- Did not train.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not go to the transport target.
- Did not execute release or scripted release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.

## Boot / Cross-Checks

- Read `CLAUDE.md` Current-State Protocol first.
- Read `START_HERE.md`.
- Read `claudedocs/DECISIONS.md` D037-D042.
- Read latest Branch B rows in `claudedocs/EXPERIMENT_LEDGER.md`.
- Read `claudedocs/session_20260518_p7_branch_b_grasp_pose_deadzone.md`.
- Read prior context:
  `claudedocs/session_20260518_p7_branch_b_approach_target_delivery.md`,
  `claudedocs/session_20260518_p7_branch_b_post_latch_target_delivery.md`,
  `claudedocs/session_20260518_p7_branch_b_post_latch_micro_executor.md`, and
  `claudedocs/session_20260518_p7_branch_b_handoff_micro_motion_probe.md`.
- `git status --short` before coding had no output.
- Required md5s before coding all matched:
  - `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
    `e0e84e481c3be8be7777a85ef2465c57`
  - `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
    `ebe8eddafd4c6f35c28e5b79a82511b3`
  - `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
    `aad6398a9d47fef5c80efbd212e619d8`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`
- Rechecked B200 D042 logs directly, not from memory:
  - `/tmp/p7_branch_b_roarm_chain_grasp_pose_stall_trace_zmicro_b200.out`
    lines 41-42, 421, 552, 683, and 860-862.
  - `/tmp/p7_branch_b_roarm_chain_grasp_pose_nudge_direction_b200.out`
    lines 42-43, 87, 114, 141, 195, and 222.
  - Both stderr files had only known cpufreq/NVML/Fabric warnings on lines 1-4.

## Code Added

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py`
  md5 `5be8cfb8c1a58f6de43f431db0befff4`.
- The script compares:
  - nominal below-top +5deg baseline inside the sponge footprint;
  - the same below-top q target with the sponge far away;
  - upward-first clearance then above-top;
  - upward-first clearance then top-tangent;
  - upward-first clearance then below-top as a kill-control;
  - side/edge clearance then side/edge tangent outside the sponge AABB.
- It prints per-step joint positions, velocities, target buffers, shoulder/
  elbow/wrist errors, TCP z versus oriented sponge top, target top class
  (`above`, `tangent`, `below`), sponge drift/speed/tilt, and exact convergence.
- The old reduction-style gate is printed only as `reduction_gate_would_pass`;
  strategy interpretation uses exact 3mm convergence plus top-clamp checks.

Local checks:

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py`
- `python sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py --help`

## B200 Run

Logs:

- `/tmp/p7_branch_b_roarm_chain_preclose_clearance_strategy_b200.out`
- `/tmp/p7_branch_b_roarm_chain_preclose_clearance_strategy_b200.err`

Evidence:

- stdout line 41 confirms strict scope: no constraints, no SurfaceGripper, no
  attached transport, no transport target, no release marker, no scripted release
  variant, no P7 training/tuning, no diagnostic gate tuning, no default edits,
  and no attach/release physics claim.
- line 42 records the exact gate: `target_error_gate_m=0.003000`,
  `segment_steps=45`, `reassert_sponge_z_m=0.023500`, nominal top `0.047000m`,
  and `reduction_gate_reference_only=YES`.
- line 43 confirms the conservative stream remains pre-MOVE:
  `executed_pre_moves=38`, `move_cmds_executed=0`.
- line 44 shows side-edge IK was valid for this diagnostic:
  clear/tangent IK converged with errors `0.574mm` and `0.419mm`.
- Baseline nominal below-top fails: line 172 reports target inside AABB, target
  `-0.022821m` below oriented top, final TCP near top
  (`final_tcp_minus_sponge_oriented_top_m=0.000099`),
  `final_target_tcp_error_m=0.023923`, `exact_converged=NO`,
  `top_clamped=YES`, and `clean_realized_without_reduction_artifact=NO`.
- Far-sponge control realizes the same below-top q target: line 272 reports
  `final_target_tcp_error_m=0.000854`, target outside AABB, `exact_converged=YES`,
  `top_clamped=NO`, and clean realization.
- Upward-first then above-top realizes: lines 372 and 466 exact-converge with
  final TCP errors `0.000983` and `0.000950`, both above top and not clamped.
- Upward-first then top-tangent realizes: line 660 reports
  `final_target_tcp_error_m=0.000495`, `target_top_class=tangent`,
  `top_clamped=NO`, and clean realization.
- Upward-first does not rescue an unsafe final below-top target: line 854 reports
  `final_target_tcp_error_m=0.011778`, `target_top_class=below`,
  `target_xy_inside_sponge_aabb=YES`, `reduction_gate_would_pass=YES`, but
  `exact_converged=NO`, `top_clamped=YES`, and clean realization `NO`.
- Side-edge tangent realizes outside the footprint: line 1048 reports
  `final_target_tcp_error_m=0.000910`, `target_top_class=tangent`,
  `target_xy_inside_sponge_aabb=NO`, `top_clamped=NO`, and clean realization.
- Aggregate lines 1050-1052 list clean strategies:
  `far_sponge_below_top_plus5deg_control`, `upward_first_then_above_top`,
  `upward_first_then_top_tangent`, and `side_edge_tangent_approach`; clamped
  segments are the nominal below-top baseline and the upward-first-then-below-top
  kill-control. `attach_calls=0`, no NaN/done, and no attach/release claim.
- stderr lines 1-4 are only the known cpufreq/NVML/Fabric warnings. There is no
  Python traceback.

## Interpretation

- D042 is confirmed and sharpened into a strategy rule: below-top commands inside
  or near the nominal sponge footprint are mechanically invalid pre-close targets.
- Upward-first clearance is useful only if the final commanded pre-close target
  remains above or tangent to the top. If the final target goes below top, the
  TCP reclamps near the top and can still look partially improved under the old
  reduction-style metric.
- Side/edge tangent is a viable diagnostic candidate because it exact-converges
  outside the footprint while staying tangent to the top.
- The far-sponge below-top pass is a no-contact control, not a permission to use
  below-top targets in the nominal contact geometry.
- This is not P7 success, not chain-ready, not attach physics, not transport/
  release validation, not SurfaceGripper validation, and not constraint
  integration.

## Next Step

- Stay pre-integration.
- If continuing, refine the pre-close candidate around:
  - upward-first then top-tangent;
  - side/edge tangent;
  - above-top shallow pre-close.
- Do not proceed to CLOSE->MOVE transport, release, SurfaceGripper, or constraint
  integration without explicit approval.

## Verification

- Local py_compile and `--help` passed.
- B200 py_compile passed.
- Local and B200 md5 matched:
  `5be8cfb8c1a58f6de43f431db0befff4`.
- B200 final run exited 0.

## Follow-Up: Pre-Close Geometry Sweep

Scope guard remained unchanged:

- Track A P7/Branch B only.
- Did not train.
- Did not integrate constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute CLOSE->MOVE transport, transport target, release, or scripted
  release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.

Boot/cross-checks for this follow-up:

- Re-read `CLAUDE.md`, `START_HERE.md`, D037-D043 in
  `claudedocs/DECISIONS.md`, latest Branch B rows in
  `claudedocs/EXPERIMENT_LEDGER.md`, and this session file.
- `git status --short` before coding had no output.
- Required pre-coding md5s matched exactly:
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py`
    `5be8cfb8c1a58f6de43f431db0befff4`
  - `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
    `e0e84e481c3be8be7777a85ef2465c57`
  - `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
    `ebe8eddafd4c6f35c28e5b79a82511b3`
  - `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
    `aad6398a9d47fef5c80efbd212e619d8`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`
- Rechecked latest authoritative pre-close B200 log directly:
  `/tmp/p7_branch_b_roarm_chain_preclose_clearance_strategy_b200.out`
  lines 41-44, 172, 272, 372/466, 660, 854, 1048, and 1050-1052.
  stderr lines 1-4 were only known cpufreq/NVML/Fabric warnings.

Code added:

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_geometry_sweep_probe.py`
  md5 `95b4a8a317a9fb176c7ed258229925e5`.
- It reuses the diagnostic-only pre-close instrumentation, but varies only safe
  geometry:
  - final top margins +0.2/+0.5/+1.0/+2.0mm;
  - upward clearance heights +12/+24/+36mm with final +0.5mm top margin;
  - side outside-AABB margins +2/+6/+12/+18mm with final +0.5mm top margin.
- It keeps the nominal below-top inside-footprint baseline and far-sponge
  no-contact control, and prints `below_inside_segments_clean` separately so the
  far-sponge control cannot be mistaken for a contact-geometry candidate.

Local/B200 checks:

- Local `python -m py_compile` passed.
- Local `--help` passed.
- B200 `python -m py_compile` passed.
- Local and B200 md5 matched:
  `95b4a8a317a9fb176c7ed258229925e5`.

B200 v2 logs:

- `/tmp/p7_branch_b_roarm_chain_preclose_geometry_sweep_v2_b200.out`
- `/tmp/p7_branch_b_roarm_chain_preclose_geometry_sweep_v2_b200.err`

Evidence:

- stdout line 41 confirms strict pre-integration scope and no attach/release
  physics claim.
- line 42 confirms unchanged exact gate `target_error_gate_m=0.003000`, the
  reduction gate remains reference-only, and the tested margins/heights are:
  top margins `[0.000200, 0.000500, 0.001000, 0.002000]`,
  clearance margins `[0.012000, 0.024000, 0.036000]`, and side margins
  `[0.002000, 0.006000, 0.012000, 0.018000]`.
- line 43 confirms no MOVE commands were executed.
- line 44 shows all IK targets converged.
- line 172 preserves the nominal below-top inside-footprint clamp baseline:
  `final_target_tcp_error_m=0.023923`, `exact_converged=NO`,
  `top_clamped=YES`, and clean realization `NO`.
- line 272 preserves the far-sponge no-contact control:
  `final_target_tcp_error_m=0.000854`, exact convergence `YES`,
  but target outside AABB; this is not a permission to use below-top targets in
  nominal contact geometry.
- Final top-margin sweep:
  - line 466: +0.2mm top margin exact-converges, final error `0.000727`,
    tangent, no top clamp.
  - line 660: +0.5mm exact-converges, final error `0.000920`, tangent, no top
    clamp.
  - line 854: +1.0mm exact-converges, final error `0.000921`, above, no top
    clamp.
  - line 1048: +2.0mm exact-converges, final error `0.000921`, above, no top
    clamp.
- Clearance-height sweep with the same final +0.5mm top margin:
  - line 1242: +12mm clearance exact-converges, final error `0.000920`.
  - line 1436: +24mm clearance exact-converges, final error `0.000920`.
  - line 1630: +36mm clearance exact-converges, final error `0.000856`.
- Side outside-AABB sweep with final +0.5mm top margin:
  - line 1824: +2mm outside-AABB exact-converges, final error `0.000915`.
  - line 2018: +6mm exact-converges, final error `0.000912`.
  - line 2212: +12mm exact-converges, final error `0.000910`.
  - line 2406: +18mm exact-converges, final error `0.000908`.
- Aggregate lines 2408-2409 report `strategies_tested=13`,
  `below_inside_segments_clean=[]`,
  `below_top_inside_targets_realize_cleanly=NO`,
  contact candidates excluding far-sponge control,
  `far_control_is_no_contact_control=YES`, `attach_calls=0`, no NaN/done, and no
  attach/release physics claim.
- line 2410 reports diagnostic success.
- stderr lines 1-4 contain only known cpufreq/NVML/Fabric warnings and no Python
  traceback.

Interpretation:

- D043 is refined, not relaxed. Below-top targets inside/near the nominal sponge
  footprint remain banned.
- In this diagnostic range, final geometry dominates: if the final target stays
  above/tangent, +12/+24/+36mm upward clearance heights all exact-converge.
- Top/tangent candidates are now supported down to the tested +0.2mm final margin,
  and side-edge candidates down to +2mm outside the width AABB.
- This remains diagnostic-only pre-close evidence. It is not robust grasp
  success, not chain-ready, not attach physics, not transport/release validation,
  not SurfaceGripper validation, and not constraint integration.

Next step:

- Stay pre-integration.
- If continuing, turn the narrowed candidate into a diagnostic pre-close command
  selection rule only: final target must be top-tangent/above or outside-AABB
  side-edge; below-top inside-footprint remains invalid.
- Do not proceed to CLOSE->MOVE transport, release, SurfaceGripper, or constraint
  integration without explicit approval.

## Follow-Up: Pre-Close Candidate Selector

Scope guard remained unchanged:

- Track A P7/Branch B only.
- Did not train.
- Did not integrate constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute CLOSE->MOVE transport, transport target, release, or scripted
  release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.

Boot/cross-checks for this follow-up:

- Re-read `CLAUDE.md`, `START_HERE.md`, D042-D044 in
  `claudedocs/DECISIONS.md`, latest Branch B rows 60-61 in
  `claudedocs/EXPERIMENT_LEDGER.md`, and this session file.
- Rechecked latest authoritative B200 geometry sweep log directly on B200:
  `/tmp/p7_branch_b_roarm_chain_preclose_geometry_sweep_v2_b200.out` lines
  41-44, 172, 272, 466/660/854/1048, 1242/1436/1630, 1824/2018/2212/2406,
  and 2408-2409. Local `/tmp` did not have those B200 logs.
- Required pre-coding md5s matched exactly:
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_geometry_sweep_probe.py`
    `95b4a8a317a9fb176c7ed258229925e5`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py`
    `5be8cfb8c1a58f6de43f431db0befff4`
  - `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
    `e0e84e481c3be8be7777a85ef2465c57`
  - `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
    `ebe8eddafd4c6f35c28e5b79a82511b3`
  - `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
    `aad6398a9d47fef5c80efbd212e619d8`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

Code added:

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  md5 `aa24ef00acbb9d8cd0aeee061b08f85f`.
- The script is a diagnostic-only selection/check layer. It prints the selection
  decision before result interpretation:
  - reject final below-top targets inside the nominal sponge footprint;
  - reject far-sponge below-top as no-contact control, not contact candidate;
  - accept final above/tangent candidates;
  - accept side-edge outside-AABB candidates.
- It still executes rejected baseline/control cases for comparison, but rejected
  cases cannot become clean contact candidates.

Local/B200 checks:

- Local `python -m py_compile` passed.
- Local `--help` passed.
- B200 `python -m py_compile` passed.
- Local and B200 md5 matched:
  `aa24ef00acbb9d8cd0aeee061b08f85f`.

B200 logs:

- `/tmp/p7_branch_b_roarm_chain_preclose_candidate_selector_b200.out`
- `/tmp/p7_branch_b_roarm_chain_preclose_candidate_selector_b200.err`

Evidence:

- stdout line 41 confirms strict pre-integration scope and no attach/release
  physics claim.
- line 42 confirms unchanged exact gate `target_error_gate_m=0.003000`,
  reduction gate reference-only, top margin `0.000500m`, above margin
  `0.001000m`, side margin `0.002000m`, and side top margin `0.000500m`.
- line 43 confirms no MOVE commands were executed.
- line 44 shows all candidate IK targets converged before simulation.
- line 46 prints the selector rule explicitly: accept final above/tangent or
  side-edge outside-AABB; reject below-top inside-footprint; far-sponge below-top
  is no-contact control only.
- lines 47-52 print decisions before interpretation:
  - nominal below-top baseline rejected as below-top inside-footprint;
  - far-sponge below-top rejected as no-contact control;
  - top-tangent +0.5mm accepted;
  - above-top +1.0mm accepted;
  - upward-then-below invalid control rejected;
  - side-edge +2mm outside-AABB accepted.
- line 179 preserves the nominal below-top clamp baseline:
  `final_target_tcp_error_m=0.023923`, `exact_converged=NO`,
  `top_clamped=YES`, `mechanically_valid_target=NO`.
- line 279 preserves the far-sponge no-contact control:
  `final_target_tcp_error_m=0.000854`, `exact_converged=YES`, but
  `mechanically_valid_target=NO`.
- Accepted contact candidates exact-converged cleanly:
  - line 473: top-tangent +0.5mm final error `0.000920`, no clamp.
  - line 667: above-top +1.0mm final error `0.000921`, no clamp.
  - line 1055: side-edge +2mm outside-AABB final error `0.000915`, no clamp.
- Invalid upward-then-below control still reclamps: line 861 reports
  `final_target_tcp_error_m=0.023470`, `top_clamped=YES`,
  `mechanically_valid_target=NO`; line 862 keeps the strategy rejected.
- Aggregate lines 1057-1059 report accepted contact candidates clean, rejected
  controls rejected, `below_inside_segments_clean=[]`, `attach_calls=0`,
  `nan_seen=NO`, `episode_done=NO`, and no attach/release physics claim.
- stderr lines 1-4 contain only the known cpufreq/NVML/Fabric warnings and no
  Python traceback.

Interpretation:

- This enforces D043/D044 as a diagnostic selection rule. It does not relax the
  ban: below-top inside-footprint targets remain invalid even after an upward
  clearance segment.
- Far-sponge exact convergence remains useful attribution evidence only; it is
  deliberately excluded from contact candidates.
- Accepted top-tangent/above/side-edge candidates remain pre-close diagnostic
  candidates only. This is not P7 success, not chain-ready, not attach physics,
  not transport/release validation, not SurfaceGripper validation, and not
  constraint integration.

Next step:

- Stay pre-integration.
- If continuing, use this selector only as a diagnostic gate around future
  pre-close candidate probes. Do not wire it into the RoArm chain or proceed to
  CLOSE->MOVE transport/release without explicit approval.

## Follow-Up: Pre-Close Selector Guard

Scope guard remained unchanged:

- Track A P7/Branch B only.
- Did not train.
- Did not integrate constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute CLOSE->MOVE transport, transport target, release, or scripted
  release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.

Boot/cross-checks for this follow-up:

- Re-read `CLAUDE.md`, `START_HERE.md`, D042-D044 in
  `claudedocs/DECISIONS.md`, latest Branch B rows 60-62 in
  `claudedocs/EXPERIMENT_LEDGER.md`, and this session file.
- Rechecked latest authoritative selector B200 log directly on B200:
  `/tmp/p7_branch_b_roarm_chain_preclose_candidate_selector_b200.out` lines
  41-52, 179, 279, 473/667/1055, 861-862, and 1057-1059. Local `/tmp` did not
  have those B200 logs.
- Required pre-coding md5s matched exactly:
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
    `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_geometry_sweep_probe.py`
    `95b4a8a317a9fb176c7ed258229925e5`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py`
    `5be8cfb8c1a58f6de43f431db0befff4`
  - `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
    `e0e84e481c3be8be7777a85ef2465c57`
  - `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
    `ebe8eddafd4c6f35c28e5b79a82511b3`
  - `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
    `aad6398a9d47fef5c80efbd212e619d8`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`
- `git status --short` before coding was empty in this local checkout.

Code added:

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_selector_guard_probe.py`
  md5 `e50f7dfcb5651507b0c200af1299f171`.
- This is a diagnostic-only wrapper around the unchanged selector script. It
  passes `--top_margin_m -0.0015` to create a near-top invalid final target just
  below the sponge top and inside the nominal footprint.
- The purpose is to test a stricter trap than the previous invalid controls:
  unchanged 3mm exact convergence may pass, but below-top inside-footprint must
  still be rejected before interpretation.

Local/B200 checks:

- Local `python -m py_compile` passed.
- Local `--help` passed.
- B200 `python -m py_compile` passed.
- Local and B200 md5 matched:
  `e50f7dfcb5651507b0c200af1299f171`.
- Existing selector md5 remained unchanged:
  `aa24ef00acbb9d8cd0aeee061b08f85f`.

B200 logs:

- `/tmp/p7_branch_b_roarm_chain_preclose_selector_guard_b200.out`
- `/tmp/p7_branch_b_roarm_chain_preclose_selector_guard_b200.err`

Evidence:

- stdout lines 2-4 confirm the guard wrapper scope and the intended invalid
  guard case: `invalid_top_margin_m=-0.001500`, expected selector behavior
  `REJECT`, and interpretation `below_top_inside_invalid_even_if_exact_gate_passes`.
- lines 43-45 confirm the underlying selector run and unchanged scope/gates:
  strict pre-integration, no attach/release claim, `target_error_gate_m=0.003000`,
  reduction gate reference-only, and `top_margin_m=-0.001500`.
- line 46 confirms no MOVE commands were executed.
- line 47 shows all IK targets converged, including the invalid near-top target.
- line 49 repeats the selector rule.
- line 52 is the pre-interpretation decision for the guard candidate:
  `candidate_top_tangent_margin_neg1p5mm` is `REJECT` with reason
  `below_top_inside_footprint_invalid`; final target is below top by
  `-0.001164m` and inside the sponge AABB.
- line 476 is the key result: the invalid near-top final segment has
  `final_target_tcp_error_m=0.001268` and `exact_converged=YES`, but it is still
  `top_clamped=YES`, `mechanically_valid_target=NO`, and
  `clean_realized_without_reduction_artifact=NO`.
- line 477 keeps the strategy rejected with segment flags `[True, False]`; the
  clearance segment is clean, but the below-top final segment is clamped.
- lines 1060-1062 report accepted clean candidates only for above-top and
  side-edge, `below_inside_segments_clean=[]`, `attach_calls=0`, no NaN/done, no
  attach/release physics claim, and diagnostic success.
- stderr lines 1-4 contain only known cpufreq/NVML/Fabric warnings and no Python
  traceback.

Interpretation:

- This strengthens D043/D044: exact 3mm convergence is not a sufficient
  interpretation gate. A below-top inside-footprint final target can exact-
  converge numerically by clamping at the sponge top, so mechanical validity must
  be applied before calling a candidate clean.
- The selector behavior is correct in this guard case: it rejects the invalid
  candidate before interpretation and prevents it from entering
  `contact_candidate_strategies`.
- This is still not P7 success, not chain-ready, not attach physics, not
  transport/release validation, not SurfaceGripper validation, and not constraint
  integration.

Next step:

- Stay pre-integration.
- Use the selector only as a diagnostic gate around future pre-close candidate
  probes.
- A useful next diagnostic would test pose/top sensitivity of accepted
  above/tangent and side-edge candidates against explicitly controlled sponge
  pose/oriented-top variants, while keeping the nominal below-top clamp baseline.

## Follow-Up: Side-Below Outside-AABB Selector Guard

Scope guard remained unchanged:

- Track A P7/Branch B only.
- Did not train.
- Did not integrate constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute CLOSE->MOVE transport, transport target, release, or scripted
  release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.
- Did not add code for this diagnostic; reused the unchanged selector with a
  command-line parameter.

Pre-run checks:

- `git status --short` showed only the expected dirty state from this session:
  `START_HERE.md`, `claudedocs/DECISIONS.md`,
  `claudedocs/EXPERIMENT_LEDGER.md`,
  `claudedocs/session_20260518_p7_branch_b_preclose_clearance_strategy.md`, and
  untracked `sim_scripts/p7_branch_b_roarm_chain_preclose_selector_guard_probe.py`.
- B200 process check before execution showed no matching
  `isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process.
- B200 md5s before execution:
  - selector `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

Run:

- Reused `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  md5 `aa24ef00acbb9d8cd0aeee061b08f85f` with
  `--side_top_margin_m -0.0015`.
- B200 logs:
  - `/tmp/p7_branch_b_roarm_chain_preclose_selector_side_below_guard_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_selector_side_below_guard_b200.err`
- B200 exit code was 0.

Evidence:

- stdout line 41 confirms strict pre-integration scope and no attach/release
  physics claim.
- line 42 confirms unchanged exact gate `target_error_gate_m=0.003000`,
  reduction gate reference-only, no diagnostic gate tuning, and
  `side_top_margin_m=-0.001500`.
- line 43 confirms no MOVE commands were executed.
- line 44 shows all IK targets converged, including
  `selector_side_margin_2mm_top_neg1p5mm`.
- line 46 repeats the selector rule: final above/tangent or side-edge
  outside-AABB can be accepted; below-top inside-footprint is rejected; far
  sponge below-top is no-contact control only.
- line 52 accepts `candidate_side_edge_margin_2mm_top_margin_neg1p5mm` because
  it is outside AABB, even though `final_target_top_class=below` and
  `final_target_tcp_minus_nominal_top_m=-0.001162`.
- line 1055 shows the shallow below-top outside-AABB side segment exact-converges
  cleanly: `final_target_tcp_error_m=0.001074`, `exact_converged=YES`,
  `reduction_gate_would_pass=YES`, `top_clamped=NO`,
  `mechanically_valid_target=YES`, and
  `clean_realized_without_reduction_artifact=YES`.
- line 1056 keeps the full side-edge strategy accepted with clean flags
  `[True, True]` and top classes `['above', 'below']`.
- aggregate lines 1057-1059 keep the nominal below-top inside-footprint baseline
  clamped, list the side-edge below-top segment under `below_segments_clean`, keep
  `below_inside_segments_clean=[]`, report accepted contact candidates clean,
  `attach_calls=0`, no NaN/done, and no attach/release physics claim.
- stderr lines 1-4 contain only known cpufreq/NVML/Fabric warnings and no Python
  traceback.

Interpretation:

- This does not relax D045 for inside-footprint targets. The invalid near-top
  below/inside case still proves exact convergence alone is insufficient.
- It does support the selector's outside-AABB exception as a diagnostic candidate
  class: a shallow below-top side-edge target can be clean if it stays outside
  the sponge footprint and does not top-clamp.
- This remains diagnostic-only. It is not P7 success, not chain-ready, not
  attach physics, not transport/release validation, not SurfaceGripper
  validation, and not constraint integration.

Next step:

- Stay pre-integration.
- The next useful diagnostic is a side-edge margin robustness check around this
  new outside-AABB below-top finding: compare smaller/larger outside-AABB margins
  while preserving the nominal below-top inside-footprint clamp baseline and
  logging whether any below target slips inside AABB.

## Follow-Up: Side-Edge Margin Robustness

Scope guard remained unchanged:

- Track A P7/Branch B only.
- Did not train.
- Did not integrate constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute CLOSE->MOVE transport, transport target, release, or scripted
  release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.
- Did not add code for this diagnostic; reused the unchanged selector with
  command-line parameters.

Pre-run checks:

- Re-read `CLAUDE.md`, `START_HERE.md`, D042-D046 in
  `claudedocs/DECISIONS.md`, latest Branch B rows 60-64 in
  `claudedocs/EXPERIMENT_LEDGER.md`, and this session file.
- Rechecked latest authoritative guard/selector B200 logs directly:
  - `/tmp/p7_branch_b_roarm_chain_preclose_selector_guard_b200.out` lines 2-4,
    43-49, 52, 476-477, and 1060-1062; stderr lines 1-4.
  - `/tmp/p7_branch_b_roarm_chain_preclose_selector_side_below_guard_b200.out`
    lines 41-46, 52, and 1055-1059; stderr lines 1-4.
  - `/tmp/p7_branch_b_roarm_chain_preclose_candidate_selector_b200.out` lines
    41-52, 179, 279, 473, 667, and 1055-1059; stderr lines 1-4.
- Required local md5s matched:
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_selector_guard_probe.py`
    `e50f7dfcb5651507b0c200af1299f171`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
    `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_geometry_sweep_probe.py`
    `95b4a8a317a9fb176c7ed258229925e5`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py`
    `5be8cfb8c1a58f6de43f431db0befff4`
  - `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
    `e0e84e481c3be8be7777a85ef2465c57`
  - `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
    `ebe8eddafd4c6f35c28e5b79a82511b3`
  - `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
    `aad6398a9d47fef5c80efbd212e619d8`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`
- `git status --short` before this diagnostic had no output in this checkout,
  which differs from the previously expected dirty state.
- B200 process check before execution showed no matching
  `isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process.
- B200 md5s before execution matched:
  - selector `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

Run:

- Reused
  `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  md5 `aa24ef00acbb9d8cd0aeee061b08f85f`.
- Fixed `--side_top_margin_m -0.0015`.
- Swept `--side_margin_m`: `0.0000`, `0.0005`, `0.0010`, `0.0020`,
  `0.0040`, `0.0060`.
- B200 logs:
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_0p0_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_0p5_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_1p0_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_2p0_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_4p0_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_6p0_b200.out`
  - matching `.err` files.
- All six B200 runs exited 0.

Common controls:

- In all six stdout files, line 42 confirms unchanged exact gate
  `target_error_gate_m=0.003000`, reduction gate reference-only, and
  `side_top_margin_m=-0.001500`; line 46 repeats the unchanged selector rule.
- In all six stdout files, line 179 preserves the nominal below-top
  inside-footprint clamp baseline: `final_target_tcp_error_m=0.023923`,
  `exact_converged=NO`, `top_clamped=YES`, `mechanically_valid_target=NO`,
  `clean_realized_without_reduction_artifact=NO`,
  `final_target_xy_inside_sponge_aabb=YES`.
- In all six stdout files, line 279 preserves the far-sponge no-contact control:
  `final_target_tcp_error_m=0.000854`, `exact_converged=YES`, `top_clamped=NO`,
  but `mechanically_valid_target=NO`, clean `NO`, and target outside AABB.
- Top/above controls remain clean in all six runs:
  - line 473: top-tangent +0.5mm, `final_target_tcp_error_m=0.000920`,
    exact `YES`, no top clamp, mechanically valid `YES`, clean `YES`.
  - line 667: above +1.0mm, `final_target_tcp_error_m=0.000921`, exact `YES`,
    no top clamp, mechanically valid `YES`, clean `YES`.

Side-margin results:

- 0.0mm:
  - line 52: selector accepted
    `candidate_side_edge_margin_0mm_top_margin_neg1p5mm` from planned geometry:
    `reason=side_edge_target_outside_aabb`, `final_target_top_class=below`,
    `final_target_xy_inside_sponge_aabb=NO`,
    `target_dy_sponge_m=0.011033`.
  - line 1055: final segment exact-converged (`0.001251`) and did not
    top-clamp, but realized/final target AABB class flipped to inside:
    `final_target_xy_inside_sponge_aabb=YES`,
    `mechanically_valid_target=NO`,
    `clean_realized_without_reduction_artifact=NO`.
  - lines 1058-1059 report `accepted_contact_candidates_clean=NO` and
    `ROARM_PRECLOSE_CANDIDATE_SELECTOR_DIAGNOSTIC_SUCCESS=NO`.
- 0.5mm:
  - line 52: accepted outside AABB with `target_dy_sponge_m=0.011532`.
  - line 1055: `final_target_tcp_error_m=0.001203`, exact `YES`,
    `top_clamped=NO`, `mechanically_valid_target=YES`, clean `YES`,
    final target outside AABB.
  - lines 1057-1059 keep `below_inside_segments_clean=[]`, `attach_calls=0`,
    no NaN/done, and diagnostic completion `YES`.
- 1.0mm:
  - line 52: accepted outside AABB with `target_dy_sponge_m=0.012031`.
  - line 1055: `0.001156`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`, final target outside AABB.
- 2.0mm:
  - line 52: accepted outside AABB with `target_dy_sponge_m=0.013030`.
  - line 1055: `0.001074`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`, final target outside AABB. This reproduces the D046 reference
    point.
- 4.0mm:
  - line 52: accepted outside AABB with `target_dy_sponge_m=0.015028`.
  - line 1055: `0.000899`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`, final target outside AABB.
- 6.0mm:
  - line 52: accepted outside AABB with `target_dy_sponge_m=0.017025`.
  - line 1055: `0.000529`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`, final target outside AABB.

stderr / process / md5:

- stderr lines 1-4 in all six runs contain only the known cpufreq/NVML/Fabric
  warnings and no Python traceback.
- Post-run grep found no traceback/exception; the only diagnostic `NO` was the
  intended 0.0mm boundary failure.
- B200 post-run process check showed no matching
  `isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process.
- B200 post-run md5s remained unchanged:
  - selector `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

Interpretation:

- D046 is refined, not relaxed. Shallow below-top side-edge is clean only when it
  has positive outside-AABB margin and remains outside in the realized/final
  metrics.
- The zero-margin boundary is not robust: the planned selector line can call it
  outside AABB, but small realized sponge pose differences can make the final
  target inside AABB. That must be treated as invalid even when exact convergence
  passes and top clamp is not detected.
- Positive margins 0.5-6.0mm were clean in this diagnostic, with the exact error
  improving as the side margin increased. This is still diagnostic-only evidence,
  not chain integration or attach/transport/release validation.

Next step:

- Stay pre-integration.
- Treat side-edge below-top candidates as requiring a positive outside-AABB
  margin and realized/final outside-AABB validation.
- Do not proceed to CLOSE->MOVE transport, release, SurfaceGripper, or constraint
  integration without explicit approval.

## Follow-Up: Side-Edge Boundary Fine Sweep

Research direction:

- D047 exposed the important weakness: planned selector geometry can say
  outside-AABB while realized/final metrics fall back inside at the AABB
  boundary.
- The next pre-integration question was therefore narrow: is the observed
  boundary really between 0.0mm and 0.5mm, and how small a positive margin still
  stays outside in realized/final metrics under the same deterministic B200
  diagnostic?
- This is a diagnostic boundary check only. It is not a chain policy, not
  attach/transport/release validation, and not permission to integrate
  constraints.

Scope guard remained unchanged:

- Track A P7/Branch B only.
- Did not train.
- Did not integrate constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute CLOSE->MOVE transport, transport target, release, or scripted
  release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.
- Did not add code for this diagnostic; reused the unchanged selector with
  command-line parameters.

Pre-run checks:

- Re-read `START_HERE.md`, `claudedocs/DECISIONS.md` D047, latest Branch B
  ledger rows 63-65, and this session file.
- `git status --short` before this follow-up showed only current state-doc
  modifications:
  `START_HERE.md`, `claudedocs/DECISIONS.md`,
  `claudedocs/EXPERIMENT_LEDGER.md`, and this session file.
- B200 process check before execution showed no matching
  `isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process.
- Local `python -m py_compile
  sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  passed.
- B200 md5s before execution matched:
  - selector `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

Run:

- Reused
  `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  md5 `aa24ef00acbb9d8cd0aeee061b08f85f`.
- Fixed `--side_top_margin_m -0.0015`.
- Swept fine positive `--side_margin_m`: `0.0001`, `0.0002`, `0.0003`,
  `0.0004`, `0.0005`.
- B200 logs:
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_0p1_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_0p2_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_0p3_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_0p4_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_0p5_b200.out`
  - matching `.err` files.
- All five B200 runs exited 0.

Common controls:

- In all five stdout files, line 42 confirms unchanged exact gate
  `target_error_gate_m=0.003000`, reduction gate reference-only, and
  `side_top_margin_m=-0.001500`; line 46 repeats the unchanged selector rule.
- In all five stdout files, line 179 preserves the nominal below-top
  inside-footprint clamp baseline: `final_target_tcp_error_m=0.023923`,
  `exact_converged=NO`, `top_clamped=YES`, `mechanically_valid_target=NO`,
  `clean_realized_without_reduction_artifact=NO`,
  `final_target_xy_inside_sponge_aabb=YES`.
- In all five stdout files, line 279 preserves the far-sponge no-contact control:
  `final_target_tcp_error_m=0.000854`, `exact_converged=YES`, `top_clamped=NO`,
  but `mechanically_valid_target=NO`, clean `NO`, and target outside AABB.
- Top/above controls remain clean in all five runs:
  - line 473: top-tangent +0.5mm, `final_target_tcp_error_m=0.000920`,
    exact `YES`, no top clamp, mechanically valid `YES`, clean `YES`.
  - line 667: above +1.0mm, `final_target_tcp_error_m=0.000921`, exact `YES`,
    no top clamp, mechanically valid `YES`, clean `YES`.

Fine side-margin results:

- 0.1mm:
  - line 52: accepted outside AABB with `target_dy_sponge_m=0.011133`.
  - line 1055: `final_target_tcp_error_m=0.001241`, exact `YES`,
    `top_clamped=NO`, `mechanically_valid_target=YES`, clean `YES`,
    final target outside AABB.
- 0.2mm:
  - line 52: accepted outside AABB with `target_dy_sponge_m=0.011232`.
  - line 1055: `0.001232`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`, final target outside AABB.
- 0.3mm:
  - line 52: accepted outside AABB with `target_dy_sponge_m=0.011332`.
  - line 1055: `0.001222`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`, final target outside AABB.
- 0.4mm:
  - line 52: accepted outside AABB with `target_dy_sponge_m=0.011432`.
  - line 1055: `0.001213`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`, final target outside AABB.
- 0.5mm:
  - line 52: accepted outside AABB with `target_dy_sponge_m=0.011532`.
  - line 1055: `0.001203`, exact `YES`, no top clamp, mechanically valid `YES`,
    clean `YES`, final target outside AABB.
- Lines 1057-1059 in all five runs keep
  `below_inside_segments_clean=[]`, `attach_calls=0`,
  `accepted_contact_candidates_clean=YES`, `nan_seen=NO`, `episode_done=NO`,
  and diagnostic success `YES`.

stderr / process / md5:

- stderr lines 1-4 in all five runs contain only the known cpufreq/NVML/Fabric
  warnings and no Python traceback.
- Post-run grep found no traceback/exception and no diagnostic `NO`.
- B200 post-run process check showed no matching
  `isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process.
- B200 post-run md5s remained unchanged:
  - selector `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

Interpretation:

- D047 is narrowed: in this deterministic B200 diagnostic, the realized/final
  AABB boundary lies between 0.0mm and 0.1mm for
  `side_top_margin_m=-0.0015`.
- Do not overread 0.1mm. It is the smallest tested positive pass here, not a
  robust chain margin. Any diagnostic-local selector must still require
  positive outside-AABB margin and final realized/final outside-AABB validation.
- This remains diagnostic-only evidence, not P7 success, not chain-ready, not
  attach physics, not transport/release validation, not SurfaceGripper
  validation, and not constraint integration.

Next step:

- Stay pre-integration.
- A sensible next diagnostic is to hold a conservative positive side margin
  (for example 1-2mm, not 0.1mm) and vary shallow below-top depth
  (`side_top_margin_m`) to find where outside-AABB side-edge begins to top-clamp
  or lose exact convergence.
- Do not proceed to CLOSE->MOVE transport, release, SurfaceGripper, or constraint
  integration without explicit approval.

## Follow-Up: Side-Top Depth Sweep at 2mm Outside-AABB

Research direction:

- After D048, the XY side margin was no longer the immediate unknown. The next
  pre-integration question was depth: if the side target stays 2mm outside the
  AABB, how far below the top can it go before it top-clamps or loses exact
  convergence?
- This remains a selector diagnostic only. It is not a chain policy, not
  attach/transport/release validation, and not constraint integration.

Scope guard remained unchanged:

- Track A P7/Branch B only.
- Did not train.
- Did not integrate constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute CLOSE->MOVE transport, transport target, release, or scripted
  release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.
- Did not add code for this diagnostic; reused the unchanged selector with
  command-line parameters.

Pre-run checks:

- B200 process check before execution showed no matching
  `isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process.
- B200 md5s before execution matched:
  - selector `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

Run:

- Reused
  `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
  md5 `aa24ef00acbb9d8cd0aeee061b08f85f`.
- Fixed `--side_margin_m 0.0020`.
- Swept `--side_top_margin_m`: `-0.0005`, `-0.0010`, `-0.0015`,
  `-0.0020`, `-0.0030`, `-0.0040`, `-0.0060`.
- B200 logs:
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg0p5_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg1p0_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg1p5_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg2p0_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg3p0_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg4p0_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg6p0_b200.out`
  - matching `.err` files.
- All seven B200 runs exited 0.

Depth results:

- In all seven stdout files, line 52 accepts the side candidate and reports final
  target outside AABB (`final_target_xy_inside_sponge_aabb=NO`) with
  `target_dy_sponge_m` about `0.01303m`.
- Clean through -3mm:
  - -0.5mm: line 1055 `final_target_tcp_error_m=0.000548`, exact `YES`,
    `top_clamped=NO`, `mechanically_valid_target=YES`, clean `YES`,
    top class `tangent`.
  - -1.0mm: line 1055 `0.000702`, exact `YES`, no top clamp, mechanically
    valid/clean `YES`, top class `tangent`.
  - -1.5mm: line 1055 `0.001074`, exact `YES`, no top clamp, mechanically
    valid/clean `YES`, top class `below`.
  - -2.0mm: line 1055 `0.001504`, exact `YES`, no top clamp, mechanically
    valid/clean `YES`.
  - -3.0mm: line 1055 `0.002409`, exact `YES`, no top clamp, mechanically
    valid/clean `YES`.
- Loses exact convergence by -4mm:
  - -4.0mm: line 1055 `final_target_tcp_error_m=0.003346`,
    `exact_converged=NO`, `top_clamped=NO`,
    `mechanically_valid_target=YES`, clean `NO`; lines 1058-1059 report accepted
    candidates clean `NO` and diagnostic success `NO`.
  - -6.0mm: line 1055 `0.005177`, exact `NO`, top clamp `NO`,
    mechanically valid `YES`, clean `NO`; lines 1058-1059 again report
    diagnostic success `NO`.
- In all seven runs, `below_inside_segments_clean=[]`, `attach_calls=0`,
  `nan_seen=NO`, `episode_done=NO`, and no attach/release physics claim remain
  true.

stderr / process / md5:

- stderr lines 1-4 in all seven runs contain only the known cpufreq/NVML/Fabric
  warnings and no Python traceback.
- Post-run grep found no traceback/exception. The only diagnostic `NO` cases
  were the intended -4mm and -6mm depth failures.
- B200 post-run process check showed no matching
  `isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process.
- B200 post-run md5s remained unchanged:
  - selector `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

Interpretation:

- At 2mm outside-AABB, side-edge below-top exact-clean depth is about -3mm under
  the unchanged 3mm gate. By -4mm, the target remains outside AABB and not
  top-clamped, but residual error exceeds the exact gate.
- This means side-edge below-top has two independent diagnostic filters:
  realized/final outside-AABB class and exact convergence. Mechanical validity
  alone is not sufficient.
- This remains diagnostic-only evidence, not P7 success, not chain-ready, not
  attach physics, not transport/release validation, not SurfaceGripper
  validation, and not constraint integration.

Next step:

- Stay pre-integration.
- If continuing, use the diagnostic evidence to define a conservative
  diagnostic-only admissible region, for example side margin >= about 1-2mm and
  side-top depth shallower than about -3mm, but do not wire it into the RoArm
  chain without explicit approval.
- Do not proceed to CLOSE->MOVE transport, release, SurfaceGripper, or constraint
  integration without explicit approval.

## Follow-Up: Diagnostic Admissible-Region Wrapper

Research direction:

- D047-D049 define exclusions but do not by themselves prove that one explicit
  diagnostic rule classifies the accumulated pass/fail cases consistently.
- The next step stayed log-level/local-wrapper only: convert those exclusions
  into a conservative diagnostic-only admissible-region check and compare it
  against the already verified selector logs.
- This is not a deployed policy and not chain integration.

Scope guard remained unchanged:

- Track A P7/Branch B only.
- Did not train.
- Did not integrate constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute CLOSE->MOVE transport, transport target, release, or scripted
  release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.
- Used the selector only as a diagnostic gate; the wrapper parses existing
  selector logs by default.

Boot / pre-coding checks:

- Re-read `CLAUDE.md` Current-State Protocol first, then `START_HERE.md`,
  `claudedocs/DECISIONS.md` D047-D049, latest Branch B ledger rows 65-67, and
  this session file.
- `git status --short` before coding showed only expected state-doc dirty files:
  `START_HERE.md`, `claudedocs/DECISIONS.md`,
  `claudedocs/EXPERIMENT_LEDGER.md`, and this session file.
- Required local md5s matched before coding:
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_selector_guard_probe.py`
    `e50f7dfcb5651507b0c200af1299f171`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
    `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_geometry_sweep_probe.py`
    `95b4a8a317a9fb176c7ed258229925e5`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_clearance_strategy_probe.py`
    `5be8cfb8c1a58f6de43f431db0befff4`
  - `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
    `e0e84e481c3be8be7777a85ef2465c57`
  - `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
    `ebe8eddafd4c6f35c28e5b79a82511b3`
  - `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
    `aad6398a9d47fef5c80efbd212e619d8`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`
- B200 process check before wrapper work showed no matching
  `isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process.

Direct B200 evidence rechecked before coding:

- 0.0mm side boundary:
  `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_0p0_b200.out`
  line 52 accepts the planned side target as outside AABB, but line 1055 reports
  final `final_target_xy_inside_sponge_aabb=YES`,
  `mechanically_valid_target=NO`, and clean `NO`; lines 1058-1059 report accepted
  candidates clean `NO` and diagnostic `NO`.
- 0.1mm fine boundary:
  `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_0p1_b200.out`
  line 1055 reports final outside-AABB, exact `YES`, top clamp `NO`,
  mechanically valid `YES`, and clean `YES`; lines 1057-1059 keep
  `below_inside_segments_clean=[]`, `attach_calls=0`, and diagnostic `YES`.
- -4mm depth:
  `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg4p0_b200.out`
  line 1055 reports final outside-AABB and mechanically valid `YES`, but
  `final_target_tcp_error_m=0.003346`, exact `NO`, and clean `NO`; lines
  1058-1059 report diagnostic `NO`.
- Additional B200 checks reconfirmed 0.5/1/2/4/6mm side-margin positive cases
  and -0.5/-1/-1.5/-2/-3/-6mm depth cases against line 52, line 1055, and
  lines 1057-1059. The direct SSH process query after checks still showed no
  matching P7/Isaac/training process.

Code added:

- Added `sim_scripts/p7_branch_b_roarm_chain_preclose_admissible_region_probe.py`
  md5 `89ad48b6ebdec076d6f58e330a9131f9`.
- The wrapper is diagnostic-only and defaults to parsing existing selector logs
  in `/tmp`. It can optionally execute the unchanged selector variants, but this
  session used log parsing only.
- Conservative tested rule:
  - reject any below-top inside-footprint final target;
  - reject zero-margin side-edge boundary;
  - reject side-edge below-top with side margin below `0.002000m`;
  - reject side-edge below-top deeper than `-0.003000m`;
  - require final/realized outside-AABB for below-top side-edge;
  - require exact convergence under the unchanged `0.003000m` gate;
  - keep reduction gate reference-only;
  - keep far-sponge below-top separated as no-contact control.

Local / B200 checks:

- Local `python -m py_compile` passed.
- Local `--help` passed.
- Local `/tmp` run correctly reported missing B200 logs instead of a traceback.
- B200 `python -m py_compile` passed.
- Local and B200 md5 matched:
  `89ad48b6ebdec076d6f58e330a9131f9`.

B200 wrapper logs:

- `/tmp/p7_branch_b_roarm_chain_preclose_admissible_region_b200.out`
- `/tmp/p7_branch_b_roarm_chain_preclose_admissible_region_b200.err`

Evidence:

- stdout lines 1-2 confirm strict diagnostic scope: no training, no diagnostic
  gate tuning, no env/train/chain default edits, no constraints, no
  SurfaceGripper, no attached transport, no transport target, no release marker,
  no scripted release variant, and no attach/release physics claim.
- line 3 records the conservative rule:
  `min_side_margin_m=0.002000`, max below depth `-0.003000`, unchanged exact gate
  reference `0.003000m`, below-top inside-footprint reject, zero-margin reject,
  final outside-AABB requirement for below-top side-edge, and far-sponge
  no-contact separation.
- line 4 rejects the 0.0mm boundary even though the selector accepted it:
  planned AABB outside, final AABB inside, exact `YES`, mechanically valid `NO`,
  clean `NO`.
- line 5 rejects the 0.1mm observed pass as below the conservative 2mm margin
  even though it is final outside-AABB, exact `YES`, mechanically valid `YES`,
  and clean `YES`. This preserves D048: 0.1mm is not a chain/deployment margin.
- line 6 accepts the conservative 2mm/-3mm side-edge case: final outside-AABB,
  exact `YES`, mechanically valid `YES`, clean `YES`.
- line 7 rejects the 2mm/-4mm side-edge case: final outside-AABB and mechanically
  valid `YES`, but depth is beyond the conservative bound, exact convergence is
  `NO`, and clean `NO`.
- lines 8-9 accept the top-tangent and above-top controls.
- line 10 rejects the nominal below-top inside-footprint baseline: planned/final
  inside-AABB, exact `NO`, top-clamped `YES`, mechanically valid `NO`, clean
  `NO`.
- line 11 rejects the far-sponge below-top no-contact control despite exact
  convergence.
- line 12 reports `cases_tested=8`, `expected_matches=8/8`,
  `attach_calls_all_zero=YES`, `below_inside_segments_clean_all_empty=YES`, no
  attach/release/transport validation, and diagnostic success `YES`.
- B200 wrapper stderr was empty.
- B200 process check after execution showed no matching
  `isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process.

Interpretation:

- The conservative diagnostic admissible-region rule explains the accumulated
  selector pass/fail evidence without tuning gates or changing selector logic.
- The rule deliberately rejects the 0.1mm observed clean case because the point
  is conservative diagnostic classification, not maximizing accepted candidates.
- Mechanical validity alone is still insufficient: -4mm remains mechanically
  outside-AABB but fails exact convergence.
- Exact convergence alone is still insufficient: 0.0mm boundary and below-top
  inside-footprint cases can be numerically misleading and must be rejected by
  realized/final AABB and mechanical validity.
- This is not P7 success, not chain-ready, not attach physics, not
  transport/release validation, not SurfaceGripper validation, and not constraint
  integration.

Next step:

- Stay pre-integration.
- Do not wire this rule into the RoArm chain without explicit approval.
- If continuing, use this wrapper only as a read-only/log-level audit layer or
  run a new explicitly scoped diagnostic matrix; do not proceed to CLOSE->MOVE
  transport, release, SurfaceGripper, or constraint integration.
