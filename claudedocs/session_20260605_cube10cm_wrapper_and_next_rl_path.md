# Session 2026-06-05 - Cube10cm Wrapper And Next RL Path

## Scope

- User asked why the active professor 10cm/0.72kg cube branch still uses a
  `cube3cm_*` filename and asked to proceed step-by-step toward RL readiness.
- No B200, SSH, pull, dataset generation, PPO/RL, VLA, Track A runtime, or GPU
  IsaacLab run was performed in this step.

## Verified State

- `git status --short --untracked-files=all --branch` initially showed only
  `## master...origin/master`, so there was no dirty/untracked state to preserve
  before this edit.
- `CLAUDE.md` Current-State Protocol lines 5-31 requires reading the rolling
  state docs, avoiding stale `HANDOFF.md`/`TASKS.md`, running git status, and
  verifying metrics from local log files before citing them.
- `START_HERE.md` top state says the active branch is professor 10cm/0.72kg
  cube push/tap DiffIK diagnosis, not Track A, and no dataset/RL/1024/10k scale
  is approved from seed946.
- seed946 remains the strongest object-level candidate: summary JSON lines 48-55
  show no dataset generation, clip `1.0`, final displacement
  `0.011250250041484833m`, final 1cm gate `1.0`, and normalized displacement
  `0.11250250041484833`; lines 70-80 show final TCP error
  `0.06282096705399454m`, fixed good workspace x/y, and fixed y+; lines 96-105
  show max displacement `0.011251196265220642m`, no overshoot, and measured
  contact `1.0`.
- seed946 reaction audit lines 2-8 and 17-28 show reaction/contact `1.0/1.0`,
  final relocation pass true, no posewrite, no overshoot, reaction gate pass,
  but `teacher_quality_ready=false` because final TCP error is about `0.062821m`
  and DiffIK clip is `1.0`.
- seed946 trace diagnostic lines 97-100 and 168-184 still identify
  `LINK5_BODY_TARGET_NOT_REACHED`, `JOINT_STEP_CLIPPING_DOMINANT`,
  `ACTUATOR_TARGET_TRACKING_LAG`, and worst joint-2 clipping/raw delta.

## Code Evidence

- The existing shared probe was already object-parameterized: `--cube_size_m` and
  `--cube_mass_kg` are parser args at `sim_scripts/cube3cm_push_diffik_probe.py`
  lines 39-40.
- The shared probe applies size, mass, and cube center z to the env config at
  lines 263-268.
- It computes TCP targets from cube center, half-size, push direction, and
  lateral offset at lines 499-520; therefore the 10cm workflow already uses the
  known cube center and does not blindly target the old 3cm geometry.
- It uses IsaacLab `DifferentialIKController` and live Jacobians later in the
  same shared engine; the remaining blocker is not "missing Jacobian" but
  tracking/clipping/contact robustness.

## Code Change

- Changed `sim_scripts/cube3cm_push_diffik_probe.py` so `main()` accepts optional
  argv. Lines 17, 33, and 132 now import `Sequence`, define `main(argv)`, and
  parse that argv.
- Added `sim_scripts/cube10cm_push_diffik_probe.py` as the standard 10cm
  professor branch entrypoint.
- New wrapper lines 1-7 document that the `cube3cm` filename is legacy and that
  this wrapper injects 10cm defaults.
- New wrapper lines 24-50 define 10cm/0.72kg, tap/reaction `0.001m`
  target/success/gate/contact-stop defaults, side-center, near-face measured-stop
  freeze, step/actuator, and 10cm output defaults.
- New wrapper lines 54-65 add a default only if the user did not provide that
  flag, and lines 68-70 delegate into the shared engine.

## Verification

- Passed:
  `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py sim_scripts/cube10cm_push_diffik_probe.py sim_scripts/cube10cm_reaction_event_gate_audit.py sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py`
- Passed: `git diff --check`
- Passed local helper check:
  `_with_professor_10cm_defaults(['--seed','946','--cube_mass_kg','0.800'])`
  includes `--cube_size_m 0.100 0.100 0.100`, keeps only one
  `--cube_mass_kg`, and preserves the caller's `0.800` override.
- After rechecking D123/D124, corrected the wrapper away from default 1cm
  relocation: helper checks confirm `--cube_push_target_disp_m`,
  `--cube_success_disp_m`, `--gate_disp_m`, and `--contact_stop_disp_m` default to
  `0.001`.

## Interpretation

- The answer to "why 3cm code for 10cm?" is: the file name was historical, but
  the engine had become size/mass configurable. That was technically valid but
  cognitively unsafe for a mixed 3cm/10cm research branch.
- The wrapper fixes the command surface without breaking old 3cm logs/tools or
  duplicating the physics implementation.
- The wrapper must not reintroduce final 1cm relocation as the default. Its
  defaults are now tap/reaction scale (`0.001m`); 1cm remains a secondary
  diagnostic or explicit relocation target.
- This change does not make seed946 teacher/RL ready. The next real research
  blocker is still reducing DiffIK clip/TCP tracking error or proving robustness
  of the seed946 geometry before any 10cm dataset or RL scale-up.

## Next Step

- Use `sim_scripts/cube10cm_push_diffik_probe.py` for any future professor
  10cm DiffIK command.
- The next narrow research step should be one of:
  actuator/IK tracking cleanup inside the seed946 good workspace, or a minimal
  robustness check of the same geometry after explicit GPU approval.
- Do not run 10240 dataset generation or RL until teacher quality and robustness
  gates are explicitly improved and audited.

## Objective Guard And Next-Step Audit

Purpose:

- The user caught a real regression risk: the 10cm wrapper initially looked like
  it reintroduced final 1cm relocation as the default, even though D123/D124 make
  tap/reaction primary and final relocation secondary.
- Add local scripts that make the objective contract and next research direction
  machine-checkable before any future GPU runtime.

Code changes:

- `sim_scripts/cube10cm_tap_objective_contract_audit.py` checks wrapper defaults
  without running IsaacLab/GPU/training/data/robot control. Lines 27-36 define
  the expected 10cm/0.72kg and tap `0.001m` defaults. Lines 68-81 fail accidental
  `0.010m` defaults but preserve explicit 1cm override. Lines 83-96 write JSON.
- `sim_scripts/cube10cm_next_research_step_audit.py` reads the existing seed946
  reaction/trace diagnostic JSONs only. Lines 49-91 classify next direction from
  controller/contact/overshoot/reaction/teacher-quality evidence. Lines 104-121
  write JSON.

Commands run:

```bash
python sim_scripts/cube10cm_tap_objective_contract_audit.py
python sim_scripts/cube10cm_next_research_step_audit.py
python -m py_compile sim_scripts/cube10cm_next_research_step_audit.py sim_scripts/cube10cm_tap_objective_contract_audit.py sim_scripts/cube10cm_push_diffik_probe.py sim_scripts/cube3cm_push_diffik_probe.py sim_scripts/cube10cm_reaction_event_gate_audit.py sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py
git diff --check
```

Results:

- Contract audit JSON lines 2-20 show contract
  `professor_cube10cm_tap_reaction`, cube size `0.1m`, mass `0.72kg`,
  explicit 1cm override allowed, no failures, final 1cm relocation default
  `false`, tap defaults `0.001m`, and PASS.
- Next-step audit JSON lines 2-16 show branch
  `professor_cube10cm_tap_reaction`, primary objective `tap_reaction_not_final_1cm`,
  contact evidence `1.0`, DiffIK clip `1.0`, do-not-start list, secondary final
  relocation pass, final TCP error `0.06282096705399454m`, next direction
  `NARROW_ACTUATOR_IK_TRACKING_CLEANUP_INSIDE_WORKING_TAP_GEOMETRY`, overshoot
  `0.0`, and reaction gate pass.
- Next-step audit JSON lines 19-24 give the exact blocker reasons: DiffIK clip
  `1.0 > 0.5`, final TCP error `0.062821 > 0.030`, actuator tracking lag, and
  teacher quality false.

Interpretation:

- The next research direction is now fixed by local logs: actuator/IK tracking
  cleanup inside the working tap geometry.
- It is not 1cm relocation, not dataset generation, not PPO/RL, not VLA, not Track
  A, and not 1024/10k scale-up.
- A future GPU test still needs explicit approval and must be exactly one tiny
  local IsaacLab screen with `sandbox_permissions=require_escalated`.

## Sources

- `CLAUDE.md:5-31`
- `START_HERE.md:1-40`
- `sim_scripts/cube3cm_push_diffik_probe.py:17,33,39-40,132,263-268,499-520`
- `sim_scripts/cube10cm_push_diffik_probe.py:1-74`
- `sim_scripts/cube10cm_tap_objective_contract_audit.py:1-8,27-36,68-96`
- `sim_scripts/cube10cm_next_research_step_audit.py:1-9,49-91,104-121`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_objective_contract_audit.json:1-21`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_next_research_step_seed946_audit.json:1-26`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_summary.json:48-55,70-80,96-105`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_reaction_gate_audit.json:1-43`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_trace_diagnostic_summary.json:97-100,168-184`

## Approved Stiff600 Tiny Runtime

Purpose:

- The user explicitly approved one tiny local GPU screen while restating the
  corrected objective: the branch is tap/reaction first, not fixed final 1cm
  relocation.
- Test exactly one actuator-tracking knob inside the seed946 working tap
  geometry: `arm_stiffness_override 400 -> 600`.

Pre-runtime guards:

```bash
python sim_scripts/cube10cm_tap_objective_contract_audit.py
python sim_scripts/cube10cm_next_research_step_audit.py
```

Both guards PASSed. Contract audit kept wrapper defaults at tap scale `0.001m`
and final 1cm relocation default false. Next-step audit still blocked
dataset/RL/VLA/TrackA/1024_10k and identified actuator/IK cleanup as the narrow
direction.

Runtime:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube10cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 946 \
  --fixed_cube_x_m 0.295 \
  --fixed_cube_y_m -0.044 \
  --fixed_push_dir 0 1 \
  --tcp_center_height_offset_m 0.000 \
  --base_lateral_offset_m -0.020 \
  --arm_stiffness_override 600 \
  --trace_diffik_diagnostics \
  --trace_env_ids 0 1 2 3 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946_summary.json \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946_trace.csv
```

Posthoc audits:

```bash
python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946_trace.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946_trace_diagnostic_summary.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946_reaction_gate_audit.json

python sim_scripts/cube10cm_next_research_step_audit.py \
  --reaction_audit_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946_reaction_gate_audit.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_stiff600_seed946_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_next_research_step_stiff600_seed946_audit.json
```

Results:

- Stiff600 summary JSON lines 1-10 show stiffness `600`, damping `20`, effort
  `25`, and velocity `12`.
- Summary lines 48-55 show no dataset generation, DiffIK clip `1.0`, final
  displacement `0.0038454830646514893m`, tap gate `1.0`, and no overshoot. Lines
  70-80 show final TCP error `0.04606295237317681m`, fixed y+ geometry, and tap
  gate `0.001m`. Lines 96-105 show max displacement
  `0.004667486995458603m`, max gate `1.0`, and measured contact `1.0`. Lines
  113/120/124/143 show no posewrite, reaction `1.0`, rollout posewrite false,
  and 16 trials.
- Reaction audit lines 2-8/17-27/41 show reaction/contact `1.0/1.0`, no
  posewrite, overshoot `0.0`, reaction gate PASS, final relocation secondary
  false, and teacher quality false.
- Trace diagnostic lines 97-100/168-184 still show target not reached and
  joint-step clipping; joint 2 remains the worst clipped/follow/raw-delta joint.
- Next-step audit lines 12-23 keeps the direction at
  `NARROW_ACTUATOR_IK_TRACKING_CLEANUP_INSIDE_WORKING_TAP_GEOMETRY` and keeps
  data/RL/VLA/TrackA/1024_10k blocked.

Interpretation:

- Stiff600 is a tap/reaction PASS under the corrected objective. It is not a
  failure just because final 1cm relocation is false.
- But it is not a better candidate than seed946: max displacement dropped from
  `0.011251196m` to `0.004667487m`, final displacement dropped from
  `0.011250250m` to `0.003845483m`, and clip remained `1.0`.
- The lower final TCP error is not enough. Do not start dataset generation,
  PPO/RL, VLA, Track A, 1024/10k, or broad search from this result.

## Approved Direction-Generalization Tiny Runtime

Purpose:

- The user asked to properly verify whether seed946's good y+ geometry
  generalizes to other directions/nearby workspace, or whether it is a one-point
  lucky/contact pocket.
- Step 1 is direction generalization at the same goodxy location. If this fails,
  nearby workspace expansion and 1024/10240/data are premature.

Design:

- Keep seed946's good workspace and lateral recipe:
  `fixed_cube_x_m=0.295`, `fixed_cube_y_m=-0.044`,
  `base_lateral_offset_m=-0.020`.
- Release only `fixed_push_dir`, so the 16 envs sample directions at the same
  cube pose.
- Use wrapper tap defaults (`0.001m`); final 1cm remains secondary only.

Pre-runtime guards:

```bash
python sim_scripts/cube10cm_tap_objective_contract_audit.py
python sim_scripts/cube10cm_next_research_step_audit.py
```

Both PASSed before the screen.

Runtime:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube10cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 947 \
  --fixed_cube_x_m 0.295 \
  --fixed_cube_y_m -0.044 \
  --base_lateral_offset_m -0.020 \
  --trace_diffik_diagnostics \
  --trace_all_envs \
  --trace_stride 4 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_summary.json \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_trace.csv
```

Posthoc:

```bash
python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_trace.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_trace_diagnostic_summary.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_reaction_gate_audit.json

python sim_scripts/cube10cm_next_research_step_audit.py \
  --reaction_audit_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_reaction_gate_audit.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_next_research_step_goodxy_dir16_seed947_audit.json
```

Results:

- Summary JSON lines 48-77 show no dataset generation, DiffIK clip `1.0`, final
  displacement `0.00020624324679374695m`, tap gate `0.25`, final TCP error
  `0.052680495427921414m`, fixed goodxy, and `fixed_push_dir=null`.
- Summary lines 92-105 show low-motion `0.875`, max displacement
  `0.0015010684728622437m`, max tap gate `0.5625`, measured contact `0.5625`.
  Lines 110-122 show no posewrite, reaction `1.0`, and rollout posewrite false.
- Reaction audit lines 2-28 show reaction `1.0` but contact evidence `0.5625`,
  no posewrite, no overshoot, reaction gate FAIL, DiffIK clip `1.0`, final TCP
  error `0.052680495m`, and teacher quality false.
- Direction contact audit lines 1-4 isolate the breakage:
  y+ `n=4`, contact `1.0`, controlled `1.0`, low-motion `0.75`;
  x- `n=7`, contact `0.0`, controlled `0.0`, success `0.0`;
  x+ `n=3` and y- `n=2` contact `1.0`, but controlled `0.0` and low-motion `1.0`.
- Trace diagnostic lines 97-100/168-184 still show target not reached, clipping,
  actuator lag, and worst joint 2.
- Next-step audit lines 12-23 changes the direction from actuator-only cleanup to
  `FIX_CONTACT_GEOMETRY_OR_WORKSPACE_BUCKET_FIRST`.

Interpretation:

- This is a direction-generalization FAIL. seed946 is not ready for 1024 trace,
  10240 trace, dataset build, BC, PPO/RL, VLA, or Track A.
- The good seed946 result is a y+-specific contact geometry pocket, not a
  balanced teacher.
- The next valid test, only after explicit approval, should isolate one failed
  direction at a time at fixed goodxy, with direction-specific lateral/contact
  geometry. Do not mix nearby workspace expansion until directions are stable.

## Approved Fixed X- Deeper Near-Face Tiny Runtime

Purpose:

- The user approved one tiny local screen to test the clean x- no-contact failure
  from seed947.
- The hypothesis was intentionally narrow: keep seed946/seed947 goodxy and
  lateral recipe, fix only x-, and make the near-face target deeper with
  `push_through_m=0.020`.

Pre-runtime guards:

```bash
python sim_scripts/cube10cm_tap_objective_contract_audit.py
python sim_scripts/cube10cm_next_research_step_audit.py \
  --reaction_audit_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_reaction_gate_audit.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_goodxy_dir16_latneg020_seed947_trace_diagnostic_summary.json \
  --out_json /tmp/cube10cm_seed947_next_prerun_recheck.json
```

Both PASSed the objective/guard expectations; next direction was still
`FIX_CONTACT_GEOMETRY_OR_WORKSPACE_BUCKET_FIRST`.

Runtime:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube10cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 948 \
  --fixed_cube_x_m 0.295 \
  --fixed_cube_y_m -0.044 \
  --fixed_push_dir -1 0 \
  --base_lateral_offset_m -0.020 \
  --push_through_m 0.020 \
  --trace_diffik_diagnostics \
  --trace_all_envs \
  --trace_stride 4 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_summary.json \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_trace.csv
```

Posthoc:

```bash
python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_trace.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_trace_diagnostic_summary.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_reaction_gate_audit.json

python sim_scripts/cube10cm_next_research_step_audit.py \
  --reaction_audit_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_reaction_gate_audit.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_next_research_step_xneg_through020_seed948_audit.json
```

Results:

- Summary JSON lines 48-80 show no dataset generation, DiffIK clip `1.0`,
  final displacement `-0.00016099214553833008m`, tap gate `0.0`, final TCP error
  `0.06216392223723233m`, fixed x-, and `gate_disp_m=0.001`.
- Summary lines 95-105 show low-motion `1.0`, max displacement
  `0.000009052455425262451m`, max tap gate `0.0`, and measured contact `0.0`.
  Lines 113-124 show no posewrite, `push_through_m=0.02`, reaction `1.0`, and
  rollout posewrite false.
- Reaction audit lines 2-28 show reaction `1.0`, contact evidence `0.0`, no
  posewrite, no overshoot, reaction gate FAIL, DiffIK clip `1.0`, final TCP error
  `0.062163922m`, and teacher quality false.
- Trace diagnostic lines 97-100/168-184 still show target not reached, clipping,
  actuator lag, and worst joint 2.
- Next-step audit lines 12-23 keeps `FIX_CONTACT_GEOMETRY_OR_WORKSPACE_BUCKET_FIRST`.

Interpretation:

- Deeper near-face target alone FAILs for x-. It does not create contact and does
  not justify 1024 trace, 10240 trace, dataset build, BC, PPO/RL, VLA, or Track A.
- The next useful diagnosis should separate x- reach/IK feasibility from
  lateral/contact-face geometry, rather than assuming more push-through depth fixes
  the no-contact case.

## Approved Fixed X- Height050 Reach/IK Tiny Runtime

Purpose:

- The user asked to continue the x- reach/IK feasibility diagnosis and also asked
  whether the stepwise method is the only valid route to dataset/training.
- The local runtime isolates the x- reach/height question: keep goodxy, lateral,
  and fixed x-, but raise the side-center TCP target with
  `tcp_center_height_offset_m=0.050`.

Pre-runtime guards:

```bash
python sim_scripts/cube10cm_tap_objective_contract_audit.py
python sim_scripts/cube10cm_next_research_step_audit.py \
  --reaction_audit_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_reaction_gate_audit.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_through020_seed948_trace_diagnostic_summary.json \
  --out_json /tmp/cube10cm_seed948_next_prerun_recheck.json
```

Both PASSed objective/guard expectations; next direction remained
`FIX_CONTACT_GEOMETRY_OR_WORKSPACE_BUCKET_FIRST`.

Runtime:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python -u sim_scripts/cube10cm_push_diffik_probe.py \
  --num_envs 16 \
  --episodes 1 \
  --seed 949 \
  --fixed_cube_x_m 0.295 \
  --fixed_cube_y_m -0.044 \
  --fixed_push_dir -1 0 \
  --base_lateral_offset_m -0.020 \
  --tcp_center_height_offset_m 0.050 \
  --trace_diffik_diagnostics \
  --trace_all_envs \
  --trace_stride 4 \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949_summary.json \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949_trace.csv
```

Posthoc:

```bash
python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py \
  --trace_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949_trace.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949_trace_diagnostic_summary.json

python sim_scripts/cube10cm_reaction_event_gate_audit.py \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949_summary.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949_reaction_gate_audit.json

python sim_scripts/cube10cm_next_research_step_audit.py \
  --reaction_audit_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949_reaction_gate_audit.json \
  --trace_diag_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_cube10cm_m072_fixed_xneg16_goodxy_latneg020_height050_seed949_trace_diagnostic_summary.json \
  --out_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_next_research_step_xneg_height050_seed949_audit.json
```

Results:

- Summary JSON lines 48-80 show no dataset generation, DiffIK clip
  `0.4605769254267216`, final displacement `0.001272156834602356m`, tap gate
  `1.0`, final TCP error `0.01297600264661014m`, fixed x-, and `gate_disp_m=0.001`.
- Summary lines 95-105 show low-motion `1.0`, max displacement
  `0.001294456422328949m`, max tap gate `1.0`, and measured contact `1.0`.
  Lines 113-127 show no posewrite, reaction `1.0`, rollout posewrite false, and
  `tcp_center_height_offset_m=0.05`.
- Reaction audit lines 2-26 show reaction/contact `1.0/1.0`, no posewrite, no
  overshoot, reaction gate PASS, DiffIK clip `0.460576925`, final TCP error
  `0.012976003m`, and teacher quality true.
- Trace diagnostic lines 1-7/97-100 show clip_any `0.463010204`, no single
  dominant failure mode, and much cleaner target tracking than seed948. Lines
  168-184 still keep joint 2 as the worst raw/follow joint.
- Next-step audit lines 12-23 says
  `RUN_TINY_HELDOUT_ROBUSTNESS_CHECK_BEFORE_DATASET_OR_RL`, while lines 5-10
  still block dataset/PPO/VLA/TrackA/1024_10k.

External cross-check:

- Isaac Lab Mimic official docs describe dataset scaling as collecting successful
  demonstrations, annotating subtasks, generating spatial variants, evaluating
  generated demonstrations, and adding successful ones to the output dataset.
- The same docs show small generated datasets are inspected first, then a full
  dataset can be generated with large `--num_envs`/`--generation_num_trials`; this
  supports staged validation rather than blindly scaling failed rollouts.
- MimicGen's paper reports scaling from a small number of demonstrations to large
  generated datasets, but still through adapted successful demonstrations across
  new contexts, not through unfiltered no-contact failures.

Interpretation:

- seed949 PASSes x- tap contact/reach/IK feasibility, but it remains tap-scale
  and low-motion. It is a strong clue for direction-specific target height, not
  a 1024/10240/data milestone.
- The next valid local step is a tiny heldout robustness check for height/contact
  geometry, ideally before any dataset construction logic is touched.
