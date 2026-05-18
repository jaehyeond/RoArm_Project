# Session 2026-05-18 - P7 Branch B grasp-pose dead-zone diagnostic

## Scope Guard

- Continued Track A P7/Branch B only.
- Did not train.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not go to the transport target.
- Did not execute release or scripted release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not edit env/train/chain defaults.

## Boot / Cross-Checks

- Read `CLAUDE.md` Current-State Protocol first.
- Read `START_HERE.md`, `claudedocs/DECISIONS.md` D024-D038,
  latest Branch B ledger rows, and the requested Branch B session docs.
- Ran `git status --short` before coding. Existing dirty state was preserved.
- Verified requested md5s before coding:
  - `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
    `ebe8eddafd4c6f35c28e5b79a82511b3`
  - `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
    `aad6398a9d47fef5c80efbd212e619d8`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`
- Rechecked latest authoritative prior B200 approach target-delivery log
  `/tmp/p7_branch_b_roarm_chain_approach_target_delivery_v2_b200.out`:
  lines 41, 43, 72, 87, 115, 143, 171, 187, 199, 211, and 213-214.

Process note:

- At 2026-05-18 13:16:52 KST, B200 had no `isaaclab.sh`, `train_ppo`,
  `torchrun`, `rl_games`, or `python .*p7_` process, but did have an unrelated
  `python code/utils/exp018_train_recovery.py ...` process. Do not state that
  B200 had no training process at all at that timestamp.

## Code Added

- Added `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
  md5 `7d7c3405e6be240500b7251df91f26e3`.
- The script reuses the conservative pre-close stream and compares the same +5deg
  shoulder nudge under diagnostic-local local variations:
  nominal sponge/open gripper, same robot q with sponge far, higher pre-grasp z
  offsets, and nominal pose with sub-threshold partial gripper closure.
- For each condition it prints set-target call counts/diffs, Articulation target
  field diffs, current joints, shoulder-specific error reduction, realized TCP
  movement, `_grasped`, gripper angle, sponge distance/proximity proxy metrics,
  sponge drift/speed, soft limits, actuator drive fields, action scale, and
  env-step vs direct set+sim-step comparisons.
- Contact sensors remain unavailable in the current env config, so contact is
  represented only by explicit proximity/AABB/top-height proxies.

Local checks:

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
- `python sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py --help`

## B200 Runs

### Default local variation matrix

Logs:

- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_b200.out`
- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_b200.err`

Evidence:

- Line 41 confirms strict scope: no constraints, no SurfaceGripper, no transport
  target, no release, no P7 training/tuning, no default edits, and no attach or
  release physics claim.
- Line 43 confirms execution remains pre-transport:
  `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Line 44 records that contact sensors are unavailable and only proximity proxy
  metrics are printed.
- Line 72 records controller/drive context:
  `action_scale=0.100000`, `null_action_max_abs=0.000000`, soft limits, arm and
  gripper stiffness `80.0`, damping `4.0`, effort limit `2.5`, velocity limit
  `3.14`.
- Nominal sponge/open gripper fails:
  line 73 target plan is the same +5deg shoulder nudge, within limits;
  line 85 env-step reports `set_target_seen=YES`,
  `best_data_target_attr_diff_rad=0.00000004`,
  `max_realized_tcp_delta_m=0.001144`,
  `final_target_tcp_error_m=0.023952`,
  `final_shoulder_error_deg=5.035108`,
  `target_realized=NO`, `_grasped=NO`, and target TCP inside sponge AABB with
  `start_target_tcp_minus_sponge_top_m=-0.022771`.
  Line 97 direct set also fails with `target_realized=NO`.
- Same robot q with sponge far realizes:
  line 98 uses the same base/target q but sponge at `(+0.800000,+0.400000)`.
  Line 110 env-step realizes with `max_realized_tcp_delta_m=0.025811`,
  `final_target_tcp_error_m=0.000850`,
  `final_shoulder_error_deg=0.114004`,
  `target_realized=YES`, and target outside sponge AABB.
  Line 122 direct set also realizes.
- Higher z offsets with nominal sponge:
  line 135 (+3mm) fails; line 160 (+6mm) fails; line 185 (+12mm) still fails by
  the diagnostic target-realized gate, although it shows partial movement
  (`max_realized_tcp_delta_m=0.009755`, `final_target_tcp_error_m=0.012066`).
  All three still have target TCP inside sponge AABB and below sponge top.
- Sub-threshold partial close does not rescue:
  line 210 env-step and line 222 direct set both fail with `_grasped=NO`.
- Lines 223-224 aggregate:
  `env_realized_conditions=['far_sponge_open']`,
  nominal and +3/+6/+12mm/partial-close fail, direct set does not rescue nominal,
  `sponge_far_realizes_nominal_fails=YES`,
  `higher_z_realizes_nominal_fails=NO`, and no attach/release physics claim.
- stderr lines 1-4 are known cpufreq/NVML/Fabric warnings only.

### Higher-z boundary cross-check

Logs:

- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zhi_b200.out`
- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zhi_b200.err`

Evidence:

- Nominal and far-sponge controls repeat the first run: lines 85/97 nominal fail,
  lines 110/122 far-sponge env/direct realize.
- +18mm realizes even though the target is still slightly below the sponge top:
  line 123 target plan has `target_tcp=...+0.041338`; line 135 env-step reports
  `target_realized=YES`, `final_target_tcp_error_m=0.006036`,
  `final_shoulder_error_deg=1.255728`, and
  `start_target_tcp_minus_sponge_top_m=-0.005662`; line 147 direct also realizes.
- +24mm realizes with target just above the sponge top:
  line 160 env-step reports `final_target_tcp_error_m=0.000757`,
  `final_shoulder_error_deg=0.078047`, `target_realized=YES`, and
  `start_target_tcp_minus_sponge_top_m=0.000859`; line 172 direct also realizes.
- +30mm realizes with target clearly above the sponge top:
  line 185 env-step reports `final_target_tcp_error_m=0.000888`,
  `final_shoulder_error_deg=0.106054`, `target_realized=YES`, and
  `start_target_tcp_minus_sponge_top_m=0.006679`; line 197 direct also realizes.
- Partial close still fails at nominal pose: lines 210 and 222.
- Lines 223-224 aggregate:
  `env_realized_conditions=['far_sponge_open',
  'nominal_sponge_z_plus_18mm_open', 'nominal_sponge_z_plus_24mm_open',
  'nominal_sponge_z_plus_30mm_open']`,
  `env_failed_conditions=['nominal_sponge_open',
  'nominal_sponge_partial_close']`,
  `sponge_far_realizes_nominal_fails=YES`,
  `higher_z_realizes_nominal_fails=YES`,
  `direct_set_also_fails_nominal=YES`, and no attach/release physics claim.
- stderr lines 1-4 are known cpufreq/NVML/Fabric warnings only.

## Interpretation

- D038 is refined: the blocker is not a broad articulation target-realization
  failure and not merely a low-grasp-pose drive limit. The same q/target realizes
  when the sponge is far.
- The nominal failure is strongly contact/proximity shaped. At nominal pose the
  +5deg shoulder nudge drives the target TCP inside the sponge AABB and about
  22.8mm below the sponge top. Moving the sponge far removes the blocker.
- Higher-z behavior is threshold-like: +3/+6/+12mm still fails or remains
  insufficient, while +18/+24/+30mm realizes. The +18mm pass means the boundary is
  not simply "target must be above the top"; contact/proximity and local geometry
  both matter.
- Direct set+sim-step mirrors env-step in every decisive condition, so env-step
  overwrite/null-action is not the cause of the nominal failure.
- This is not P7 success, not attach physics, not transport/release, not
  SurfaceGripper, and not constraint integration.

## Next Step

- Stay pre-integration.
- The next useful diagnostic is to treat +12mm to +13mm as a marginal realization
  boundary, not a solved target-reach corridor. +13mm passes the diagnostic
  reduction gate but still has centimeter-scale final TCP error.
- Horizontal offsets are not cleanly interpretable yet because some offsets
  change the settled sponge top/posture. If continuing, inspect/reset sponge
  pose/orientation state explicitly before using xy sweeps as contact evidence.
- Do not use this as approval to integrate constraints into the RoArm chain.

## Verification

- Local `py_compile` and `--help` passed.
- B200 remote md5 matched local:
  initial `7d7c3405e6be240500b7251df91f26e3`; latest diagnostic instrumentation
  `bee46b8203e9dfdd5d86b69301551af0`.
- B200 default and high-z runs both exited 0.

## Follow-up Boundary Diagnostics

### Fine z sweep

Logs:

- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zfine_b200.out`
- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zfine_b200.err`

Evidence:

- Lines 85/97 repeat nominal sponge/open failure.
- Lines 135/147 show +13mm env/direct pass the diagnostic realization gate:
  env `final_target_tcp_error_m=0.011058`,
  `final_shoulder_error_deg=2.224505`, `target_realized=YES`; direct
  `final_target_tcp_error_m=0.011020`, `final_shoulder_error_deg=2.242731`,
  `target_realized=YES`.
- Lines 160/172, 185/197, 210/222, and 235/247 show +14/+15/+16/+17mm also pass
  the same reduction-based gate.
- Lines 273-274 aggregate that far sponge and +13 through +17mm realize while
  nominal and partial-close fail; no direct-rescue split appears.
- stderr lines 1-4 contain only known cpufreq/NVML/Fabric warnings.

Interpretation:

- The previous +12 to +18 window is much narrower under this diagnostic: +13mm is
  the first tested passing point.
- This is still a reduction gate, not exact target convergence. +13mm retains
  about 11mm final TCP error, so do not call +13mm a robust grasp command solution.

### Micro z sweep

Logs:

- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zmicro_b200.out`
- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_zmicro_b200.err`

Evidence:

- Line 43 records local z offsets `[12.0, 12.25, 12.5, 12.75, 13.0]`.
- Lines 136/148 show +12.0mm env/direct still fail the realization gate.
- Lines 161/173 show +12.25mm env/direct fail.
- Lines 186/198 show +12.5mm env/direct fail.
- Lines 211/223 show +12.75mm env/direct fail.
- Lines 236/248 show +13.0mm env/direct pass.
- Lines 274-275 aggregate that only far sponge and +13mm realize; +12.0 through
  +12.75mm fail, no direct-rescue split, and no attach/release physics claim.
- stderr lines 1-4 contain only known cpufreq/NVML/Fabric warnings.

Interpretation:

- Under the current reduction gate and nominal sponge posture, the boundary is
  between +12.75mm and +13.0mm.
- The pass remains marginal: +13.0mm still has `final_target_tcp_error_m` around
  11mm. This is evidence of improved command realization, not a validated grasp
  pose.

### Horizontal y-offset cross-check

Logs:

- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_xy_b200.out`
- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_ycheck_b200.out`

Evidence:

- The wide xy sweep line 43 tested x offsets `[-35,-30,-25,-20,+20,+25,+30,+35]`
  mm and y offsets `[-25,-20,-15,+15,+20,+25]` mm.
- Wide xy sweep line 499 reported env realization for far sponge,
  `sponge_y_minus_25mm_open`, and `sponge_y_plus_15mm_open`.
- Targeted y-check line 43 retested `[-25,-20,+15,+20]` mm with extra sponge
  top/dx/dy fields.
- Targeted y-check line 136 contradicts the earlier y -25mm pass:
  `sponge_y_minus_25mm_open` fails with `final_target_tcp_error_m=0.014990`,
  `final_shoulder_error_deg=3.120213`, `target_realized=NO`,
  `start_target_xy_inside_sponge_aabb=NO`, and
  `start_sponge_top_z_m=0.047000`.
- Targeted y-check line 186 repeats y +15mm success, but the sponge has settled
  lower: `start_sponge_xyz=([+0.265269, -0.031637, +0.011000])`,
  `start_sponge_top_z_m=0.034500`,
  `start_target_tcp_minus_sponge_top_m=-0.010271`, and `target_realized=YES`.
  Direct line 198 also realizes.
- Targeted y-check line 211 shows y +20mm fails with the usual top height:
  `start_sponge_top_z_m=0.047000`,
  `start_target_tcp_minus_sponge_top_m=-0.022771`,
  `target_realized=NO`.
- Targeted y-check lines 249-250 aggregate only far sponge and y +15mm as
  realized; y -25mm, y -20mm, y +20mm, nominal, and partial close fail.
- stderr lines 1-4 contain only known cpufreq/NVML/Fabric warnings.

Interpretation:

- Horizontal offset is not yet a clean independent variable. The y +15mm pass is
  entangled with a lower settled sponge top/posture, while y -25mm was not
  reproducible across runs.
- Do not infer "AABB outside succeeds" from the xy sweep. The stronger current
  evidence remains local z clearance / contact posture around nominal grasp
  geometry.

### Pose/top controlled boundary cross-check

Logs:

- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_log_zboundary_b200.out`
- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zboundary_b200.out`
- `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zmicro2_b200.out`

Evidence:

- The probe was extended to log sponge root quaternion, `up_z`, tilt, upright top,
  and oriented top, and to optionally reassert sponge pose before each delivery.
  Latest script md5: `bee46b8203e9dfdd5d86b69301551af0`.
- Uncontrolled pose/top logging:
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_log_zboundary_b200.out`
  line 43 tests +12.75/+12.875/+13.0mm.
  Lines 86/98 show nominal sponge is nearly upright with top about `0.047000m`.
  Lines 136/148 show +12.75mm env/direct fail with
  `start_sponge_oriented_top_z_m=0.047000`,
  `start_target_tcp_minus_sponge_oriented_top_m=-0.010666`, and final TCP error
  about `0.0112m`.
  Lines 173 and 186/198 show +12.875mm and +13.0mm env/direct pass the reduction
  gate, still with final TCP error about `0.0110-0.0111m`.
  Lines 224-225 aggregate +12.875/+13.0mm realized and +12.75mm failed.
- Controlled pose/top run:
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zboundary_b200.out`
  line 42 confirms `reassert_sponge_before_delivery=YES` and
  `reassert_sponge_z_m=0.0235`.
  Reassert lines 76, 130, 157, and 184 show the requested/actual sponge pose is
  exactly `(+0.250000,-0.040000,+0.023500)` with identity quaternion and both
  upright/oriented top `0.047000m` before delivery.
  Lines 141/154 show +12.75mm env/direct still fail.
  Lines 168/181 show +12.875mm env/direct pass.
  Lines 195/208 show +13.0mm env/direct pass.
  Lines 236-237 aggregate the same split and no direct-rescue condition.
- Controlled micro2 run:
  `/tmp/p7_branch_b_roarm_chain_grasp_pose_deadzone_pose_reassert_zmicro2_b200.out`
  line 42 again confirms pose reassert at `z=0.0235`.
  Lines 130/157/184 show identity/upright top `0.047000m` before each tested z
  condition.
  Lines 141/154 show +12.8125mm env/direct fail:
  env `final_target_tcp_error_m=0.011191`,
  `final_shoulder_error_deg=2.266477`, `target_realized=NO`; direct
  `final_target_tcp_error_m=0.011147`,
  `final_shoulder_error_deg=2.290611`, `target_realized=NO`.
  Lines 168/181 show +12.84375mm env/direct pass:
  env `final_target_tcp_error_m=0.011184`,
  `final_shoulder_error_deg=2.244741`, `target_realized=YES`; direct
  `final_target_tcp_error_m=0.011148`,
  `final_shoulder_error_deg=2.262267`, `target_realized=YES`.
  Lines 195/208 show +12.875mm env/direct pass.
  Lines 236-237 aggregate +12.84375/+12.875mm realized and +12.8125mm failed.
- stderr lines 1-4 in all three pose/top runs contain only the known
  cpufreq/NVML/Fabric warnings and no Python traceback.

Interpretation:

- The +12.75 to +13.0mm split is not explained by sponge tilt or top-height
  drift. With pose/top reasserted to identity and top `0.047000m`, the split
  remains.
- The tighter controlled boundary under this reduction gate is between
  +12.8125mm and +12.84375mm.
- This is a very thin diagnostic-gate boundary, not a robust grasp solution. The
  passing cases still leave about 11mm final TCP error, and the pass/fail switch
  is driven by shoulder-error reduction crossing the gate.
- Do not tune the gate to make this look better. The useful lesson is that the
  nominal contact/clearance posture is marginal and not a validated commandable
  grasp pose.

## Follow-up Stall Trace / Contact-Equilibrium Diagnostic

### Code extension

- Extended `sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py`
  md5 `e0e84e481c3be8be7777a85ef2465c57`.
- Added diagnostic-only fields: `--trace_every_step`, `--joint_nudge_degs`,
  per-step `robot_dof_targets`, Articulation `joint_pos_target`, per-joint error,
  joint velocity, TCP z, sponge top/oriented-top deltas, and final
  TCP-vs-oriented-top fields in `delivery_result`.
- Did not change the diagnostic gate, P7 reward/scalars, env defaults, chain
  defaults, attach semantics, SurfaceGripper, constraint integration, transport,
  or release behavior.

Verification:

- Local `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_grasp_pose_deadzone_probe.py` passed.
- Local and B200 md5 matched after the final patch:
  `e0e84e481c3be8be7777a85ef2465c57`.
- B200 `py_compile` passed.

### Controlled +12.8125/+12.84375/+13.0 trace

Logs:

- `/tmp/p7_branch_b_roarm_chain_grasp_pose_stall_trace_zmicro_b200.out`
- `/tmp/p7_branch_b_roarm_chain_grasp_pose_stall_trace_zmicro_b200.err`

Evidence:

- Line 42 confirms unchanged gates and scope for this trace:
  `target_error_gate_m=0.003000`, `joint_nudge_degs=[5.0]`,
  `trace_every_step=YES`, `reassert_sponge_before_delivery=YES`, and
  `reassert_sponge_z_m=0.0235`.
- Lines 338, 469, and 600 reassert the sponge to identity/upright top
  `0.047000m` before the +12.8125, +12.84375, and +13.0mm env deliveries.
- +12.8125mm:
  - line 342 starts with target buffers correct:
    `robot_dof_targets_deg` and `data_joint_pos_target_deg` match the watched
    target, shoulder velocity is active, and TCP moves downward toward the top.
  - line 346 reaches `tcp_z_m=+0.047688`, just above top.
  - line 420 ends with `tcp_z_m=+0.047017`, `step_tcp_delta_m=0.000000`,
    `joint_vel_deg_s` near zero, shoulder error `2.266477deg`, and
    `tcp_minus_sponge_oriented_top_m=-0.000043` while target remains below top by
    about `-0.010667m`.
  - line 421 reports final env failure:
    `final_target_tcp_error_m=0.011191`, `final_shoulder_error_deg=2.266477`,
    `target_realized=NO`, `final_tcp_minus_sponge_oriented_top_m=-0.000043`,
    and `final_target_tcp_minus_sponge_oriented_top_m=-0.010667`.
  - line 466 shows direct set also fails with the same clamp pattern.
- +12.84375mm:
  - line 467 records the condition and target, still below top.
  - lines 473-481 show the same fast downward motion toward top and then near-top
    clamp.
  - line 551 ends at `tcp_z_m=+0.047008`,
    `tcp_minus_sponge_oriented_top_m=-0.000036`, target below top by
    `-0.010620m`, and shoulder error `2.244741deg`.
  - line 552 reports env `target_realized=YES`, but only because
    `shoulder_error_reduced=YES`; final TCP error is still `0.011184m`.
  - line 597 shows direct set also passes the same reduction gate with
    `final_target_tcp_error_m=0.011148` and final TCP clamped near top.
- +13.0mm:
  - line 600 reasserts the same controlled sponge pose.
  - lines 604-612 show the same approach to the top.
  - line 682 ends at `tcp_z_m=+0.047029`,
    `tcp_minus_sponge_oriented_top_m=-0.000018`, target below top by
    `-0.010475m`, and shoulder error `2.220062deg`.
  - line 683 reports env `target_realized=YES`, again with
    `final_target_tcp_error_m=0.011042`.
  - line 728 shows direct set also passes the reduction gate with
    `final_target_tcp_error_m=0.011004`.
- Lines 860-862 aggregate the same split:
  +12.8125 fails; +12.84375 and +13.0 pass; no attach/release physics claim.
- stderr lines 1-4 contain only known cpufreq/NVML/Fabric warnings and no Python
  traceback.

Interpretation:

- The TCP is not converging to the requested below-top targets. It is stalling at
  the sponge oriented top, with the target still about 10.5-10.7mm below top.
- The +12.84375/+13.0 "pass" is a reduction-gate artifact: the shoulder error
  falls just below the 50% reduction threshold, but final TCP error stays about
  11mm and the final TCP is still top-clamped.

### Nudge magnitude/direction check

Logs:

- `/tmp/p7_branch_b_roarm_chain_grasp_pose_nudge_direction_b200.out`
- `/tmp/p7_branch_b_roarm_chain_grasp_pose_nudge_direction_b200.err`

Evidence:

- Lines 42-43 confirm controlled pose reassert, no z-offset variants, and
  shoulder nudges `[-5.0, 2.5, 5.0]`.
- Nominal sponge:
  - line 87: `-5deg` shoulder nudge realizes with
    `final_target_tcp_error_m=0.000986`; its target is above top
    (`start_target_tcp_minus_sponge_oriented_top_m=0.023941`).
  - line 114: `+2.5deg` nudge fails with target below top
    (`start_target_tcp_minus_sponge_oriented_top_m=-0.011317`), final TCP remains
    near top (`final_tcp_minus_sponge_oriented_top_m=-0.000135`), and
    `target_realized=NO`.
  - line 141: `+5deg` nudge repeats the below-top stall with target below top by
    `-0.022771m`.
- Far sponge:
  - lines 195 and 222 show +2.5/+5deg realize when the sponge is far, preserving
    the contact/proximity interpretation rather than broad drive failure.
- Partial close:
  - line 249 shows the `-5deg` upward/above-top target realizes even with the
    partial-close condition.
  - lines 276 and 303 show +2.5/+5deg downward below-top targets still fail.
- stderr lines 1-4 contain only known cpufreq/NVML/Fabric warnings and no Python
  traceback.

Interpretation:

- This kills the idea that the latest pass boundary is useful below-top command
  convergence. The decisive split is whether the requested TCP target asks the
  local posture to drive through the sponge top.
- Current blocker is local contact equilibrium at the sponge top around nominal
  pre-close geometry. A future strategy should be designed as a mechanically valid
  clearance/grasp-posture diagnostic before any chain integration.
- Still no training, no constraints, no SurfaceGripper, no transport target, no
  release, and no attach physics validation.
