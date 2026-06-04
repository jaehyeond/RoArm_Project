# Session: 2026-06-04 Cube10cm DiffIK Teacher Gate Prep

## Scope

- Professor cube push/tap branch only.
- No B200/SSH/reconnect/pull or `.ssh` copy.
- No Track A grasp/dataset/training/runtime.
- One local GPU IsaacLab tiny 128 runtime was run after explicit approval.
- No 1024/10k scale-up, dataset generation, PPO scale-up, VLA training, or
  Track A runtime was run.

## Term Definitions Used In This Session

- `GPU escalated`: Codex default sandbox hides `/dev/nvidia*`; any IsaacLab/PhysX
  command that needs the local GPU must be run with
  `sandbox_permissions=require_escalated`.
- `tiny 128 gate`: a small 128-environment smoke gate, not training and not a
  dataset run. It checks whether the configured object/controller/trajectory can
  produce stable controlled pushes before any larger 1024/10240 data collection.
- `teacher`: here means the scripted IsaacLab `DifferentialIKController` path
  that produces TCP targets and joint targets from live PhysX Jacobians.
- `teacher-off`: in the learned-policy actor/eval branch, this means disabling
  teacher blending. For this DiffIK probe, the more precise term is
  `DiffIK teacher gate`, because the controller itself is the diagnostic teacher
  and the script already sets training/dataset generation false.

## Verified Context

- Current 3cm env constants remain fixed in `roarm_rl/roarm_cube_push_env.py`:
  `CUBE_SIZE_M=0.030`, mass `0.020kg`, default target displacement `0.040m`,
  strict success displacement `0.030m`.
- Existing 2026-06-04 hierarchical sharded-10k audit remains 3cm evidence only:
  it logs `cube_size_m=0.030000`, `cube_mass_kg=0.020000`, density
  `740.741kg/m^3`, and displacement-only 1/5/10/20/30mm columns.
- Professor's current request is interpreted as a separate 10cm cube/object
  diagnostic, not as Track A grasp and not as a requirement to move the object
  one full object length.

## Code Prep

- Updated `sim_scripts/cube3cm_push_diffik_probe.py` without changing its default
  3cm behavior:
  - added `--cube_size_m`, `--cube_mass_kg`, `--cube_push_target_disp_m`,
    `--cube_success_disp_m`, and `--gate_disp_m`;
  - sets the IsaacLab cuboid size, mass, and z-center from these args before
    `gym.make`;
  - logs cube size, mass, density, object-size reference, target displacement,
    success displacement, and gate displacement;
  - computes the TCP precontact/through points from the configured object size,
    not the old imported 3cm constant;
  - records `disp_over_object_size`, `disp_ge_gate_rate`, and
    1/5/10/20/30mm threshold rates.

## Local Verification

- `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py`: PASS.
- `python sim_scripts/cube3cm_push_diffik_probe.py --help`: PASS and showed the
  new size/mass/gate args.
- `git diff --check`: PASS.

## Approved Tiny 128 Runtime

After explicit approval, ran:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u \
  sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 128 \
  --episodes 1 \
  --seed 930 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_gate128_seed930.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_gate128_seed930_summary.json
```

Result: FAIL as a 1cm gate.

- Summary lines 11-25: controlled `0.1875`, `disp_along_push_mean_m=-0.0001524518520454876`,
  `disp_ge_gate_rate=0.0`.
- Summary lines 27-34: only the 1mm displacement tier reached `0.1875`; 5/10/20/30mm all `0.0`.
- Summary lines 41-52: final TCP target error mean `0.15134897292591631m`,
  low-motion `1.0`, max cube speed mean `0.0004937104492910294m/s`,
  min TCP-cube distance mean `0.13646007765782997m`.
- Summary lines 56 and 63: posewrite during rollout `0` and rollout object
  posewrite false, so the failure is not hidden object pose-writing.
- CSV had 128 rows plus header. Direction analysis showed every direction had
  `ge10mm=0.0`, `low=1.0`, and `diffik_clip_rate=1.0`; mean final TCP target
  error was roughly `0.14-0.17m` by direction.

Interpretation:

- The 10cm/0.72kg object did not fail because 1cm is too strict. It failed
  earlier: the default v1 trajectory/controller did not reliably reach/contact
  the larger object, and the object barely moved.
- Do not scale this to 1024/10k data, dataset generation, PPO/RL, VLA, or Track A.
- The next valid work is small geometry/control diagnosis: contact height,
  precontact/through distance, horizon, and DiffIK joint-step clipping for a
  10cm object, then another tiny gate only after explicit approval.

## Post-Fail Code Review Before Further Edits

After the failed 128 gate, reviewed the relevant code before making another
change.

Findings:

1. Reset z was still 3cm-specific. `sim_scripts/cube3cm_push_diffik_probe.py`
   set `env_cfg.sponge.init_state.pos` from the requested 10cm size, but
   `roarm_rl/roarm_cube_push_env.py` reset used the module constant
   `CUBE_CENTER_Z`, derived from `CUBE_SIZE_M=0.030`. For a 10cm object this is
   a 35mm z-center error.
2. The first 10cm gate mixed geometry validation with random x/y and random push
   direction. That is useful for robustness later, but it is too broad for
   debugging whether one easy, centered push works.
3. The current probe target height mode is a top-margin path. For the professor's
   "push the front face" mental model, the next diagnostic should be able to use
   side-face center height.
4. The failed run's `diffik_clip_rate_mean=1.0` and final TCP target error mean
   `0.15134897292591631m` should be interpreted as a reach/path issue before
   treating mass/friction as the limiting factor.

## Static Patch After Review

Patched without running another IsaacLab/GPU runtime:

- `roarm_rl/roarm_cube_push_env.py`
  - added config fields `cube_size_x_m`, `cube_size_y_m`, `cube_size_z_m`;
  - reset now uses `TABLE_Z + cube_size_z_m / 2` instead of the 3cm
    `CUBE_CENTER_Z` constant;
  - local precontact/teacher target helpers now compute half extents from the
    configured object size;
  - added optional fixed push direction config.
- `sim_scripts/cube3cm_push_diffik_probe.py`
  - passes the configured cube size into the env config, not just the spawn
    object;
  - added `--fixed_cube_x_m`, `--fixed_cube_y_m`, and `--fixed_push_dir`;
  - added `--tcp_height_mode top_margin|side_center`;
  - logs `table_z_m`, `cube_center_z_m`, and per-row `cube_z0_m` so the reset
    height is auditable from logs.

Static verification:

- `python -m py_compile roarm_rl/roarm_cube_push_env.py sim_scripts/cube3cm_push_diffik_probe.py`: PASS.
- `python sim_scripts/cube3cm_push_diffik_probe.py --help`: PASS and showed the
  fixed-position, fixed-direction, and TCP-height-mode args.
- `git diff --check`: PASS.
- Direct `conda run -n isaaclab python -c "from roarm_rl..."` import was not a
  valid check because IsaacLab/Omniverse modules require `SimulationApp` /
  `AppLauncher` before import. No runtime claim is made from that failed import.

## Fixed 16 Candidate Shape

This candidate was run after explicit approval with GPU escalation:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u \
  sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 931 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --fixed_cube_x_m 0.300 \
  --fixed_cube_y_m 0.000 \
  --fixed_push_dir 1 0 \
  --tcp_height_mode side_center \
  --precontact_clearance_m 0.010 \
  --push_through_m 0.010 \
  --approach_steps 220 \
  --push_steps 90 \
  --post_steps 40 \
  --max_diffik_joint_step_rad 0.035 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010
```

Interpretation:

- This is not dataset generation, not PPO/RL, not VLA, and not Track A.
- First check `cube_z0_m`, `fixed_push_dir`, `tcp_height_mode`, final TCP target
  error, min TCP-cube distance, and only then the 1cm displacement gate.
- If this fixed easy case fails, do not add randomization; debug geometry/reach
  with trace/video.

## Approved Fixed 16 Runtime

Result: FAIL as a 1cm fixed-geometry gate.

- Summary lines 11-26: controlled `0.0`, `diffik_clip_rate_mean=1.0`,
  `disp_along_push_mean_m=0.000025190412998199463`, and
  `disp_ge_gate_rate=0.0`.
- Summary lines 28-35: 1/5/10/20/30mm rates were all `0.0`; mean xy
  displacement was `0.000027741372491618677m`.
- Summary lines 42-59: final TCP target error mean
  `0.1307708825916052m`, min TCP-cube distance mean
  `0.07594270585104823m`, and min TCP target error mean
  `0.0738731506280601m`.
- Summary lines 63-76: posewrite during rollout remained `0`, rollout object
  posewrite false, `tcp_height_mode=side_center`, and trace CSV was written.
- Per-env CSV lines 2-4 still logged `cube_z0_m=0.03788299858570099`, while
  trace lines 2-5 showed the actual settled cube z was about `0.049999m`.
  Trace line 2 also showed `target_z_m=0.03788299858570099`, so the side-center
  TCP target was generated from the reset buffer, not the settled PhysX center.
- Trace lines 278-281 showed the final target x was `3.359999895095825`, but TCP
  x stayed around `3.2489-3.2494` and TCP z around `0.106-0.107m`, with only
  ~0.03mm object displacement. This is still a reach/target-generation/clipping
  failure before mass/friction conclusions.

## Post Fixed-16 Code Review And Static Patch

Reviewed the code before editing again.

Findings:

1. `roarm_rl/roarm_stack_env.py` uses a plane terrain, while `TABLE_Z` is a
   legacy sponge geometry constant. For the 10cm cuboid, PhysX contact resolution
   settled the actual cube center near `z=0.050m`, not the reset buffer
   `TABLE_Z + 0.050 = 0.037883m`.
2. `sim_scripts/cube3cm_push_diffik_probe.py` took `cube_start_w` from
   `inner._cube_start_w` after settling, so `compute_tcp_targets()` used the reset
   buffer z for side-center targeting.
3. This does not invalidate the failure verdict, but it makes the gate too broad:
   the robot was chasing a target height that did not match the actual settled
   object center.

Patched without another GPU runtime:

- After settle, the probe now copies `inner._sponge_pos_w` into `cube_start_w`
  and `inner._cube_start_w`, then updates the internal target buffer from that
  settled pose.
- Summary now logs `cube_start_z_mean_m` so requested reset z and actual
  diagnostic start z can be compared directly.

Static verification:

- `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py`: PASS.
- `python sim_scripts/cube3cm_push_diffik_probe.py --help`: PASS.
- `git diff --check`: PASS.

Next valid runtime, only after explicit approval with GPU escalation: rerun the
same fixed 16 gate with the settled-start patch and new output paths. If it still
fails with high TCP target error and `diffik_clip_rate_mean=1.0`, the next step is
not randomization or data generation; it is trace/video diagnosis of reachable TCP
height/orientation and joint-limit/clipping behavior.

## Approved Settled-Start Fixed 16 Runtime

After explicit approval, reran the same 16-env fixed-position/fixed-direction
side-center gate with the settled-start patch.

Result: FAIL as a 1cm gate, but the z mismatch was fixed.

- Summary lines 13 and 22: requested reset-derived `cube_center_z_m=0.037883`,
  but actual settled diagnostic start `cube_start_z_mean_m=0.04999994789250195`.
- Summary lines 25-36: `diffik_clip_rate_mean=1.0`,
  `disp_along_push_mean_m=0.00001109391450881958`, `disp_ge_gate_rate=0.0`,
  and all 1/5/10/20/30mm rates `0.0`.
- Summary lines 43-60: final TCP target error mean
  `0.12522956123575568m`, min TCP target error mean `0.06143945129588246m`,
  and min TCP-cube distance mean `0.07620850298553705m`.
- Summary lines 64-77: posewrite calls `0`, rollout object posewrite false, and
  trace CSV written.
- Trace lines 2-5: target z now matches the settled cube z around `0.050m`, so
  the previous reset-buffer z bug is no longer the immediate explanation.
- Trace lines 278-281: final target x is `3.3600244522094727`, but TCP x remains
  around `3.2484-3.2491` and TCP z around `0.1068-0.1075m`; object displacement
  stays around 0.008-0.012mm.

Local non-GPU kinematics check:

- `sim_scripts/roarm_kinematics.py` uses `TCP_LOCAL_OFFSET_M=0.115428m` in the
  URDF chain, consistent with the runtime TCP offset.
- Pure local DLS IK from HOME solved the side-center precontact target
  `(0.2400246,0,0.0499997)` with `err_mm=1.334`, and the through target
  `(0.3600245,0,0.0499997)` with `err_mm=0.461`.
- Therefore the remaining failure is not a proof that the point is unreachable
  in the project kinematic model. It is now narrowed to the IsaacLab
  DifferentialIK rollout path: link5 Jacobian/body target mapping, position-only
  controller convergence, joint-step clipping, actuator tracking, or collision
  constraints.

Next:

- Do not scale.
- Do not generate 10,240 data yet.
- Next code work must start with review of the DiffIK body/Jacobian mapping and
  actuator target trace. The next runtime should be a smaller trace/video
  diagnostic, not another 128/1024 gate.

## Approved Trace / Target Geometry Diagnostics

Scope:

- Still professor 10cm/0.72kg cube push/tap branch only.
- No B200, no Track A, no dataset generation, no PPO/RL/VLA, no 128/1024/10k.
- All GPU/IsaacLab diagnostics were explicit-approval local tiny 4-env runs.

Code review before edits:

- `sim_scripts/cube3cm_push_diffik_probe.py` lines 291-297 create IsaacLab
  `DifferentialIKController` with position command and DLS.
- Lines 287-289 use `link5_body_idx=5`, fixed-base `jacobi_body_idx=4`, and arm
  joint ids 0-4.
- Lines 457-480 convert TCP target to a link5 body target using current link5
  quaternion and `tcp_local`.
- Lines 539-570 now log before/after target errors, raw delta, clipped delta,
  commanded target, and actual joint position after `env.step`.
- `roarm_rl/roarm_stack_env.py` lines 173-180 show the default arm actuator is
  stiffness `80.0`, damping `4.0`, effort limit `2.5`, velocity limit `3.14`.

Static edits:

- Added `--trace_diffik_diagnostics` and a local analyzer
  `sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py`.
- Added default-preserving actuator overrides
  `--arm_stiffness_override`, `--arm_damping_override`,
  `--arm_effort_limit_sim_override`, and `--arm_velocity_limit_sim_override`.
- Added `--through_target_mode {far_face,near_face}`. Code lines 471-475 now
  make the distinction explicit:
  `far_face = cube + half + push_through`; `near_face = cube - half + push_through`.
- Static checks passed: `py_compile`, `--help`, `git diff --check`. The analyzer
  rejected an old non-extended trace as a negative control.

Runtime diagnostics:

- Fixed 4-env far-face baseline, cap `0.035`, seed932: FAIL. Summary lines
  11/25-28/43/65-68 show controlled `0.0`, gate `0.0`, clip `1.0`, final TCP
  error `0.125396907m`, low-motion `1.0`, min TCP-cube `0.076345483m`.
  Analyzer lines 97-100 classify link5 target not reached, joint-step clipping,
  and actuator target-tracking lag.
- cap `0.120`, seed932: positive control but not teacher-ready. Summary lines
  11/25-28/43/65-67 show controlled `1.0`, gate `1.0`, mean displacement
  `0.089894325m`, clip `1.0`, final TCP error `0.094675537m`, low-motion `0.0`.
  This proves motion is possible, but it overshoots the 1cm objective by a lot.
- Long approach with cap `0.035`, seed933: FAIL. Summary lines 19/21/25-28/43/65-68
  show 830 steps, controlled `0.0`, gate `0.0`, clip `1.0`, final TCP error
  `0.124230925m`, low-motion `1.0`, min TCP-cube `0.075801797m`.
- Far-face drive boost, seed934: diagnostic positive but not teacher-ready.
  Summary lines 2-11 record stiffness `400`, damping `20`, effort `25`,
  velocity `12`; lines 21/36-38/53/75-77 show controlled `0.75`, mean
  displacement `0.067951232m`, gate `0.75`, final TCP error `0.101294972m`,
  low-motion `0.25`, clip `1.0`.
- Near-face default actuator, seed935: target geometry improved error but did
  not reach contact. Summary lines 21/35-37/53/75-79/95 show controlled `0.0`,
  gate `0.0`, clip `1.0`, final TCP error `0.061259050m`, low-motion `1.0`,
  min TCP-cube `0.082745695m`, `through_target_mode=near_face`.
- Near-face long approach, seed936: still FAIL. Summary lines 19/21/35-37/53/75-79/95
  show 830 steps, gate `0.0`, final TCP error `0.061997696m`, low-motion `1.0`,
  min TCP-cube `0.083884733m`.
- Near-face drive boost, seed937: gate PASS but overshoot. Summary lines
  21/36-38/53/75-79/91/95 show controlled `1.0`, mean displacement
  `0.050082028m`, gate `1.0`, final TCP error `0.124849608m`, low-motion `0.0`,
  success marker `1.0`, near-face mode. Per-env CSV lines 2-5 all moved about
  5cm, not about 1cm.

Interpretation:

- The old far-face target is a bad 10cm-object teacher target: it asks the TCP to
  travel through the cube to far-face + push-through.
- Near-face target geometry is necessary but not sufficient. It halves the final
  target error, but the default actuator/step path still stops about 8cm from the
  cube center and never contacts.
- Strong cap/drive controls prove the object can move, but they overshoot by
  5-9cm and still show high TCP error/clipping, so they are not teacher data.

Next:

- Do not scale this.
- Design one tiny controlled near-face contact controller: approach/contact
  phases, actual-contact or displacement stop, smaller/contact-aware push
  increments, and actuator/step scheduling. It must be judged on about-1cm
  displacement and low TCP error, not just `disp_ge_gate_rate=1.0`.

## Previous Gate Candidate

This was the initial candidate and has now failed:

```bash
python sim_scripts/cube3cm_push_diffik_probe.py \
  --num_envs 128 \
  --episodes 1 \
  --seed 930 \
  --cube_size_m 0.100 0.100 0.100 \
  --cube_mass_kg 0.720 \
  --cube_push_target_disp_m 0.010 \
  --cube_success_disp_m 0.010 \
  --gate_disp_m 0.010
```

Expected interpretation:

- Primary metric is not 10cm displacement. It is whether a 10cm/0.72kg object can
  be pushed about 1cm in a controlled, no-impact, low-motion-safe way.
- Report `disp_ge_gate_rate`, 1/5/10/20/30mm rates, `disp/object_size`,
  controlled, no-impact, low-motion, TCP target error, and posewrite count.
- The failed result blocks immediate 1024/10240 scale-up, dataset generation,
  PPO scale-up, VLA training, and Track A runtime.

## Metric Correction: Reaction Event Primary For Push/Tap

User clarification:

- For the professor's 10cm/0.72kg cube push/tap objective, final displacement does
  not have to be the primary success criterion.
- If the cube is pushed/tapped, reacts, lifts slightly, and then settles back near
  the original position, that can still satisfy the practical objective.
- Therefore final 1cm displacement is a secondary relocation metric unless the
  professor explicitly asks for sustained relocation.

Static/code update:

- `sim_scripts/cube3cm_push_diffik_probe.py` lines 61-63 add explicit reaction
  thresholds: `--reaction_disp_m`, `--reaction_z_delta_m`, and
  `--reaction_speed_mps`.
- Lines 247-252 validate those thresholds.
- Lines 589-605 track transient max displacement, max z delta, max tip angle, and
  contact/stop state.
- Lines 891-897 compute `reaction_event`; lines 932-936 write max transient
  displacement/z/tip fields and `reaction_event` into each per-env row.
- Lines 1137-1147 summarize max transient displacement, transient gate rates,
  `reaction_event_rate`, max z delta, and max tip angle.
- Static checks passed after the reaction-metric update:
  `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py`,
  `python sim_scripts/cube3cm_push_diffik_probe.py --help`, and
  `git diff --check`.

Measured-stop diagnostics:

- seed938, near slowdown `contact_near_joint_step_scale=0.35`, was too
  conservative. Summary lines 21-31 show measured-stop mode but no stop; lines
  50-52 show final displacement only `1.30385160446167e-05m` and final gate
  `0.0`; lines 68-71 show final TCP error `0.06104395352303982m`, first contact
  `-1`, and first stop `-1`; lines 96-100 show measured contact `0.0` while
  near range was seen.
- seed939, no near slowdown, reached contact/stop and then retracted. Summary
  lines 21-31 show stop seen `1.0`; lines 50-52 show final displacement
  `0.0014313608407974243m` and final gate `0.0`; lines 68-71 show first contact
  mean `261.5` and first stop mean `279.75`; lines 94-100 show max speed
  `0.1447020135819912m/s`, measured contact `1.0`, and near-contact `1.0`.
- seed940, measured-stop freeze, is the key clarified-objective result. Summary
  lines 21-32 show freeze mode and stop seen `1.0`; lines 51-53 show final
  displacement `0.0014494359493255615m` and final gate `0.0`; lines 95-100 show
  max speed `0.14879385754466057m/s`, max 8-15mm transient rate `1.0`, max
  transient displacement mean `0.010990217328071594m`, no overshoot, and transient
  1cm rate `1.0`; lines 101-106 show measured contact and near-contact both
  `1.0`.
- seed940 per-env CSV lines 2-5 show all 4 fixed envs had measured contact,
  contact stop, no overshoot, final displacement about `1.38-1.56mm`, max
  transient displacement about `10.81-11.26mm`, and max speed about
  `0.138-0.164m/s`.
- seed940 trace analyzer still reports controller-quality blockers: diagnostic
  summary lines 2-29 show joint clipping; lines 97-100 list
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`.

Interpretation:

- Under the old final 1cm relocation gate, seed940 fails because the cube settles
  back to about 1.45mm final displacement.
- Under the clarified push/tap reaction criterion, seed940 is a fixed-geometry
  reaction-event pass: contact happened, the cube transiently moved about 1cm,
  no overshoot was recorded, and speed evidence is strong.
- Do not call the 0.72kg cube too heavy or the task impossible.
- Do not call this teacher/data/RL readiness either. The DiffIK path still clips
  and lags, so the next work is metric definition and tiny reaction screening,
  not direct dataset generation or PPO/RL scale-up.

Scale/randomization analysis:

- NVIDIA's Isaac Lab docs state that Isaac Lab trains with many environments in
  parallel and uses GPU-accelerated workflows; the NVIDIA Isaac Lab developer page
  describes it as GPU-accelerated, agent-ready, and designed to train policies at
  scale.
- The Isaac Gym paper reports a GPU-native simulation/training path with 2-3
  orders of magnitude speedups over conventional CPU-simulator RL. So the
  professor's intuition that large-scale IsaacLab search can find candidates is
  directionally reasonable.
- Domain randomization and ADR papers support simulation-to-real only when the
  randomized distribution is deliberately built to cover the real-world gap.
  They do not say that one selected lucky trajectory is sufficient evidence.
- Simple random search can be competitive in some RL/control settings, but the
  same paper emphasizes high variability across seeds and hyperparameters. That
  reinforces the need for held-out seeds, not cherry-picking.
- Probability check: for per-trial success probability `p`, one or more successes
  in `N=1,000,000` trials happens with probability `1-(1-p)^N`. If `p=1e-6`, this
  is about 63%; if `p=1e-7`, about 9.5%; if `p=1e-8`, about 1%. One success in
  one million can therefore mean "the event exists" but can also mean "the event
  is too rare/brittle to be useful."

Next:

- Define the professor 10cm reaction-event gate explicitly:
  measured contact or `reaction_event`, max transient displacement/z/speed, no
  posewrite, no Track A, and trace/video audit.
- Only after that, ask for approval for a tiny randomized reaction screen. Do not
  run dataset generation, PPO/RL scale-up, VLA, 1024/10k, or Track A from seed940.

Sources:

- `sim_scripts/cube3cm_push_diffik_probe.py:61-63,247-252,589-605,891-897,932-936,1137-1147`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_seed938_summary.json:21-118`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_driveboost_noslow_seed939_summary.json:21-118`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_summary.json:21-118`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940.csv:1-5`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_measuredstop_freeze_driveboost_seed940_trace_diagnostic_summary.json:2-100`
- NVIDIA Isaac Lab docs:
  https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-first-robot-with-isaac-lab/02-how-isaac-lab-accelerates-reinforcement-learning.html
- NVIDIA Isaac Lab developer page: https://developer.nvidia.com/isaac/lab
- Isaac Gym paper: https://arxiv.org/abs/2108.10470
- Domain Randomization paper: https://arxiv.org/abs/1703.06907
- OpenAI Rubik's Cube ADR paper: https://arxiv.org/abs/1910.07113
- Simple random search RL paper: https://arxiv.org/abs/1803.07055
