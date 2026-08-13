# p16 / t3u_side_phys1 preregistration — fixed-base side-midpoint parallel PhysX grasp

Status: **BLOCKED ON `t3u_side_preflight2` / NOT RUN**  
Case: `g0b_d420`  
Canonical prefix: `t3u_side_phys1`  
Date: 2026-08-12 KST

## 1. Scientific question

Under the user-approved sim-only D419 exception, can the fixed-base RoArm-M3-Pro grasp
the upright D29 x H50, 24.83 g cylinder across its side midpoint and lift it by contact
physics, without kinematic attachment, desk/support collision, non-jaw shoving, or
self-collision?

This is a local existence test, not a workspace atlas.  One valid passing row is
sufficient for `SIDE_MIDPOINT_GRASP_PASS_IN_FIXED_SIM_CONTROLS`.  If none passes, the
specific preregistered failure or no-evidence branch in section 3 must be reported rather
than collapsed into a generic label.  Neither result establishes real hardware
performance or measured friction.

이번 case의 신규 변수: `[side-midpoint grasp point, SDG candidate pose]`.

## 2. Frozen inheritance from instrumentation preflight

Sections 2 through 8 of `t3u_side_preflight2_prereg.md` are inherited exactly except:

- this tag is scientifically authoritative after all validity, lifecycle, Rerun, and
  visual gates pass;
- canonical environment count is `64`; exactly eight p15 candidates crossed with five
  frozen pinch offsets produce exactly 40 ordered planned rows.  The active set is hard
  pinned to exactly 10 ordered rows: candidates 5 and 7, each with offsets 0..4.  This
  equality is checked before environment creation and again after transformed-hull
  clearance; any other nonempty IK set is drift and aborts rather than silently changing
  the experiment.  All 30 preregistered IK/frame rejects and padded rows remain reported
  but excluded from PhysX denominators.  The first padding slot after the 10 active rows
  is the inherited witness and is always excluded from task counts.  Zero rows after the
  independent static-clearance gate selects the explicit no-safe-planned-path verdict;
- the frozen pre-physics planner found exactly 10/40 IK/frame-feasible rows: all five
  offsets for p15 candidates 5 and 7.  Candidate 5 is the first passing candidate in
  canonical p15 order and remains the preflight/canonical visual representative; this
  selection happened before any p16 Isaac/PhysX step and may not be replaced by whichever
  row later has the best physical outcome;
- candidate order, five geometry-derived offset controls, controller, full phase
  durations, contact schemas, representative selection, and render cadence must be
  byte-for-byte equivalent to the passing preflight contract;
- every external supervisor/lifecycle filename substitutes the prefix
  `t3u_side_phys1` for `t3u_side_preflight2`; the detached no-retry contract and its
  timeouts are otherwise exact;
- the exact full SHA-256 of the passing preflight result, terminal attestation, manual
  visual inspection, all supervisor/raw-lifecycle files, RRD/RBL/NPZ/PNGs/render manifest
  and MP4, p15 handoff, both preregistration files, p16 source/supervisor, pinned p14
  helper, jaw extractor, kinematics, environment sources, URDF, and all five attempt3
  USD layers are recorded before AppLauncher and verified unchanged after physics.
  Canonical reloads the preflight result/plan/trace/render/outcome files and recomputes
  exact result, render, lifecycle, and supervisor semantic contracts with the same pure
  raw verifier used by terminal attestation.  This includes exact child schemas/commands,
  PID/PGID/SID/TTY and raw wait status, timestamps, signal/timeout/retry/cleanup fields,
  process and GPU residue, stdout/launcher/phase hashes, every outcome binding, and the
  stored attestation's exact equality to the recomputation.  A terminal attestation
  `pass=true` or arbitrary all-true map is insufficient;
- Supervisor V4 applies the same raw-zero-is-insufficient rule to both children.
  Canonical admission requires the embedded physics preclose gate and post-hoc render
  gate to equal independent recomputation.  The render gate rechecks exactly 234 frame
  files, every NPZ/state/clock mapping, and independent decoding of all PNGs plus every
  MP4 frame with count/size/fps/hash gates, zero physics,
  exact durable render phase/stdout completion, and physics-finalize/render-start/end
  dependency equality.  Physics phases have exact row schemas/finite ordered times and
  stdout PREClose is reconstructed byte-for-byte from the result.  The NPZ must encode
  exact steps `1..2340`, `step/200` times and frozen phase arrays; render sampling must be
  steps `10..2340` by tens, with the exact 491-label zero-physics clock schedule
  (250 app updates, 240 Replicator calls, one new-stage audit).  Numeric schema is strict:
  Boolean values can never stand in for zero-valued integer counts or float clocks,
  deltas, frame/phase fields, fidelity errors, render-phase observations, or decode
  count/size/fps fields.  The dependency map is
  the exact profile-specific full physics set, including every p15 bound file and, for
  canonical, all passing-preflight files—not a reduced render-only subset.  Raw-zero
  semantic failure maps to reserved status 125 and a
  render-abort attestation is always `pass=false`/non-promotable;
- raw lifecycle schema is likewise strict and shared across success, physics-abort,
  render-abort, terminal recomputation, and canonical admission.  PID/PGID/SID must be
  integers greater than one; attempt/raw-wait/exit/signal/combined-status fields must be
  exact non-Boolean integers; times must be finite ordered floats.  Recursive V4
  contract, semantic-gate, outcome, and attestation equality compares JSON type as well
  as value, so `false == 0` and `true == 1` can never authorize promotion;
- the preflight and canonical subjects both hard-gate the actual composed fixed-base
  metatype/root joint and full-step fixed-body stability, all 15 non-adjacent self pairs
  (including fixed `world` versus five non-adjacent moving bodies), and actual composed/
  runtime joint limits plus post-clamp applied-target equality to parsed URDF controls.

The failed `t3u_side_preflight1` remains immutable negative evidence: it executed zero
task-physics steps and exposed an instrumentation defect, not a grasp result.  Its
authored USD rest transforms (`q=0`) had been compared against hard-coded HOME
`q2=90 deg`; the old p10 compact chain also rounded the URDF's `0.051959 m` and
`1.5708 rad` literals.  Preflight2 and canonical therefore use the exact decimal frozen
URDF origin/RPY/axis chain for every 2,340-sample collider-clearance transform.  They
separately gate (a) default-time authored/rest USD frames against exact URDF `q=0` and
(b) same-epoch PhysX articulation joint positions against actual body poses.  p10 remains
IK provenance only and its delta is diagnostic; no clearance, contact, pose, or success
threshold was relaxed.

Any preflight failure or contract drift blocks this canonical run; it cannot be bypassed
by changing `run_label`, environment count, candidate list, offsets, phase lengths, or
camera selection.

## 3. Scientific verdict ladder

Before physics, failure of the exact 40-planned/10-active IK contract is a source or
dependency drift abort, not a scientific branch.  If the exact IK rows exist but any is
removed by the preregistered 1 mm enabled-collider/support clearance over all 2,340
command samples, return `NO_PLANNED_STATIC_CLEARANCE_FEASIBLE_SIDE_PATH`.  That means
PhysX grasp evidence is absent and may not be presented as a physical grasp failure.

If PhysX executes, every active row is assigned exactly one causal label in the progress
order below, and branch masks/counts are serialized.  Population selection is existential:
`measurement-invalid in any row` blocks all claims; otherwise `success in any row` proves
the local existence result; otherwise the **deepest populated failure branch** is the
global verdict.  Thus one early-failing row cannot hide other rows that reached bilateral
lift contact.  The ordered row branches are:

1. `MEASUREMENT_INVALID` — any row has invalid numerical/contact instrumentation or the
   run-level lifecycle, provenance, Rerun, or visual evidence fails.  A scientific task
   verdict is forbidden.
2. `SIDE_MIDPOINT_GRASP_PASS_IN_FIXED_SIM_CONTROLS` — at least one valid row passes every
   exact gate.  Report every passing candidate/offset, not only the best video.
3. `TRACKING_OR_ARRIVAL_GATE_FAIL` — no success and a valid row fails grasp/lift arm
   arrival or commanded TCP-rise tracking.  Contact cannot hide a tracking failure.
4. `PREMATURE_JAW_CONTACT_BLOCKS_SIDE_GRASP` — no earlier branch and a valid arrived row
   contacts the object with either jaw before close above `0.02 N`.
5. `NONJAW_OR_SUPPORT_COLLISION_BLOCKS_SIDE_GRASP` — no earlier branch and a row is
   blocked by moving-link/support, non-jaw/object, or one of the 15 gated self contacts.
6. `NO_BILATERAL_SIDE_CONTACT` — this row is valid, arrived, task-clear, but lacks
   same-step bilateral contact during close.  It can be the population verdict only when
   no row reaches any later branch.
7. `BILATERAL_CONTACT_LOST_BEFORE_LIFT` — this row closes bilaterally but does not retain
   same-step bilateral contact during lift.  It is selected only when no row gets farther.
8. `BILATERAL_CONTACT_BUT_NO_CORRECTED_LIFT` — bilateral contact survives into lift for
   this row, but corrected object rise does not strictly exceed `6 mm`; it is selected only
   when no row reaches a later branch.
9. `OBJECT_LIFTED_BUT_TIPPED` — corrected lift passes for this row but final tilt fails;
   it is selected only when no success or later residual exact-gate branch exists.
10. `OTHER_EXACT_GATE_FAIL` — only a residual exact-gate combination not captured above;
    masks and row labels must expose which condition remained false.

Corrected lift is the cylinder bottom-equivalent centre rise after subtracting its own
settled rest pose, not TCP motion.  Bilateral force authority is the same-step scalar
`max_t(min(F_fixed(t),F_moving(t)))`; independent per-jaw maxima cannot prove a grasp.

## 4. Required outputs and scope warning

Required prefix outputs include `_results.json`, `_plan.json`, `_trace.npz`,
`_timeline.rrd`, `_timeline.rbl`, `_rerun_validation.json`, `_inspection.png`,
`_decision_snapshot.png`, `_rgb_frames_manifest.json`, `_side_grasp.mp4`, `_script.py.txt`, `_argv.txt`, lifecycle
phase/sentinel/attestation records, and a separate manual visual inspection record.

Canonical uses the same frozen no-retry supervisor and lifecycle modes with only the
registered profile changed:

```bash
(
  set -o noclobber
  p16_case_dir=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/g0b_d420
  p16_profile=side_phys1
  p16_prefix="t3u_${p16_profile}"
  p16_suffixes=(
    results.json plan.json trace.npz timeline.rrd timeline.rbl
    rerun_validation.json decision_snapshot.png inspection.png
    rgb_frames_manifest.json side_grasp.mp4 script.py.txt argv.txt phase.jsonl
    render_phase.jsonl
    preclose_sentinel.json terminal_attestation.json manual_visual_inspection.json
    failure.json render_failure.json exit_status.txt stdout.log supervisor_launcher.log
    supervisor_pid.txt physics_python_pid.txt render_python_pid.txt pgid.txt
    supervisor_contract.json supervisor_outcome.json nvidia_smi_before.csv
    nvidia_smi_supervisor_end.csv nvidia_smi_after.csv supervisor_failure.json
  )
  for p16_suffix in "${p16_suffixes[@]}"; do
    p16_target="${p16_case_dir}/${p16_prefix}_${p16_suffix}"
    if [[ -e "${p16_target}" ]]; then
      echo "G0 existing target: ${p16_target}" >&2
      exit 3
    fi
  done
  if [[ -e "${p16_case_dir}/${p16_prefix}_rgb_frames" ]]; then
    echo "G0 existing frame directory" >&2
    exit 3
  fi
  nohup setsid /home/cgxr/miniconda3/envs/isaaclab/bin/python \
    /home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor.py \
    --profile "${p16_profile}" \
    --candidates_sha256 67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384 \
    </dev/null >"${p16_case_dir}/${p16_prefix}_supervisor_launcher.log" 2>&1 &
  echo "detached supervisor pid=$! (one attempt; no retry)"
)
```

After the supervisor and both child groups are reaped, run exactly:

```bash
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  /home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics.py \
  --terminal_attest side_phys1
```

The block above is the complete G0/no-retry wrapper, not permission to overwrite the
launcher log or any tag artifact.

The frozen detached supervisor source SHA-256 is
`527b06e5b9a090f4207c5f9ac5feb539c4b26f4c23f48ac59e4d802a153fa365` at
`sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor.py`; canonical accepts only
the exact `--profile side_phys1 --candidates_sha256
67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384` argv shown above.

Candidate index 5 (`side_sdg_005_raw_025092`) / nominal offset remains the video
representative even if another row
is scientifically successful.  Every passing row is still reported in JSON; the video
cannot select or tune the best outcome after physics.

Scope warning serialized verbatim:

> A PASS is one fixed analytic-cylinder pose under unmeasured placeholder material
> coefficients and the parsed URDF joint envelope.  It does not authorize hardware,
> does not measure real friction, does not validate an actual finite desk, does not
> reopen prior top-centre verdicts, and does not yet constitute Isaac Lab learning.
