# p16 / t3u_side_preflight2 preregistration — reactive side-grasp instrumentation preflight

Status: **PREREGISTERED / NOT RUN / STATIC RE-AUDIT REQUIRED**  
Case: `g0b_d420`  
Canonical prefix: `t3u_side_preflight2`  
Date: 2026-08-12 KST

## 1. Why this forward-only tag exists

`t3u_side_preflight1` aborted before the first task-physics step.  Its frozen failure
reported `STATIC_CLEARANCE_FK_USD_FRAME_ALIGNMENT_FAIL`: link1 matched, link3's origin
nearly matched but its rotation differed by about one matrix unit, and downstream
translations differed by `0.204475496`, `0.281169383`, and `0.357185983 m`.  Recomputing
the frozen transforms reproduced that pattern by comparing the USD's authored/rest
`q2=0 deg` pose against p16's hard-coded HOME `q2=90 deg`.  A second, much smaller
discrepancy came from the legacy p10 compact IK chain: it uses `0.05196 m` and exact
`pi/2`, whereas the frozen URDF authors `0.051959 m` and `1.5708 rad`.

This is a reactive instrumentation repair only.  It does not change the cylinder,
candidate, jaw geometry, controller, trajectory, contact threshold, clearance threshold,
lift threshold, tilt threshold, friction placeholders, or success definition.  The
scientific variable count added by this tag is zero.  The one failure-capable experiment
already performed in this research session is preflight1; preflight2 may run only after
the new bytes receive an independent static GO.

Immutable failure evidence:

- preflight1 frozen p16 source SHA-256:
  `0f253207dd5b77073cd504b8e344118575f2d6cc082eec6ae5c679ee180e31bf`;
- `t3u_side_preflight1_failure.json` SHA-256:
  `ebbb6a6109a587d1547bdba02e0d1ed28eb2af16bbaf15239ee7275009e6179f`;
- `t3u_side_preflight1_supervisor_outcome.json` SHA-256:
  `460ea9cff3a072b714c9a17d8d637ed53ff9f6f8b517613e99a046ffe7888c0c`;
- `t3u_side_preflight1_prereg.md` SHA-256:
  `4f3ba53f3f350c11962120fbfc5d57818961f6d552db00997a682c074ad0faa4`.

Those files are historical inputs only and must never be overwritten, renamed, repaired,
or used as a grasp verdict.

## 2. Frozen physical subject inherited without relaxation

Except for the four reactive instrumentation changes in sections 3 through 6, sections
2 through 7 of `t3u_side_preflight1_prereg.md` remain the physical and measurement
contract:

- fixed-base RoArm attempt3 asset; actual 64+64 enabled convex-hull jaw parts;
- upright analytic cylinder `D=0.029 m`, `H=0.050 m`, `mass=0.02483 kg`, centre
  `[0.4235072423787768,0.17237803311822986,0.025] m`;
- placeholder cylinder material static/dynamic/restitution `0.40/0.30/0.0`, with no
  effective-pair or real-friction claim;
- p15 side_sdg2 candidate index 5, ID `side_sdg_005_raw_025092`, crossed with all five
  frozen pinch offsets: exactly five planned and five active rows in eight environments;
- link5 `+X` closure, `+Y` world-up, `+Z` radial approach; physical pinch-centre offset
  `+4.474150434 mm`; q5 open `88.3099849635 deg`, sole close command `22 deg`;
- HOME -> elevated outward pregrasp -> near-side staging -> final horizontal midpoint ->
  close -> hold -> vertical lift, exactly 2,340 samples at 200 Hz;
- transformed enabled-collider minimum support clearance `>=1.0 mm` at every sample;
  final adverse pitch `<=1 deg`; no attach; parsed true URDF limits; full fixed-base,
  applied-target, reporter, support, non-jaw/object and all 15 nonadjacent self-contact
  gates;
- same-step bilateral jaw force strictly `>0.01 N`, collision/contact gate `>0.02 N`,
  corrected object lift strictly `>6 mm`, and final tilt strictly below the unchanged
  cylinder tip angle;
- full-step authoritative NPZ plus RRD/RBL/two PNGs; render is isolated, post-hoc,
  zero-physics and produces an exact frame manifest plus MP4.

Preflight2 remains non-scientific.  Even a fully valid run may promote only
instrumentation readiness, never a side-grasp conclusion.

## 3. Reactive repair A — exact URDF clearance FK and two distinct pose audits

The transformed-collider clearance authority is now parsed directly from the frozen
`local_assets/roarm_m3/urdf/roarm_m3.urdf` bytes (SHA-256
`64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2`).
P16 verifies the exact ordered joint names, parent/child links, joint types, `+Z` axes,
and decimal origin xyz/RPY values, then evaluates all 2,340 command samples with that
chain.  P10 remains the IK solver only.  Its q=0 delta from exact URDF is serialized as a
diagnostic and can never authorize a clearance transform.

Two different observations must not be conflated:

1. **Authored/rest audit.** `UsdGeom` default-time body transforms and every composed
   revolute joint's local frames are used to recover the authored joint coordinates.
   These must be q=0 within `2e-5 deg`, have pure-axis residual `<=5e-7`, and each of the
   six moving bodies must match exact-decimal URDF q=0 within the unchanged
   `1e-6 m` / `1e-6` rotation-matrix gates.  These fields must say authored/rest, never
   current PhysX or HOME.
2. **Same-epoch runtime audit.** In one articulation data timestamp, p16 reads actual
   `joint_pos`, `body_pos_w`, and `body_quat_w` for every clone.  Exact-decimal URDF FK at
   each observed joint vector must match the corresponding actual body pose, after
   subtracting that clone's environment origin, within `5e-6 m`, `1e-5` rotation-matrix
   component, and `1e-6` quaternion-norm error.  Tensor shapes, finiteness, body/joint
   order, and unchanged timestamp are hard gates.

Failure of either audit aborts before any task step.  It is instrumentation invalidity,
not desk collision or grasp failure.  No old tolerance was widened: the first audit
retains the original `1e-6` gates; the second is a new independent float-runtime
readback gate.

As an independent no-Kit/no-PhysX countercheck before freeze, the actual composed USD
joint local frames were also evaluated at every one of the five planned command
schedules: `5 x 2,340 x 6 = 70,200` body transforms.  Exact URDF FK versus composed
`L0 * Rz(q) * inverse(L1)` differed by at most `2.4063157388066466e-08 m` in translation
and `1.6798151730723632e-07` in a rotation-matrix component; all six `L1` transforms were
identity.  This demonstrates the q-nonzero motion-map equivalence for the frozen bytes;
it is a static derivation, not physics evidence or a relaxed runtime threshold.

## 4. Reactive repair B — raw exit zero is not physics success

Preflight1 proved that `SimulationApp.close()` can terminate with raw wait status zero
after p16 has already written a failure marker.  Supervisor V4 therefore defines physics
semantic success as all of the following, recomputed after the child group is reaped:

- raw ordinary exit zero, no timeout/signal action, and empty child process group;
- `failure.json` absent;
- nonempty `results.json`, `plan.json`, `trace.npz`, RRD, RBL, Rerun validation,
  decision PNG, inspection PNG, frozen source, argv, phase log, and preclose sentinel;
- result/plan/profile/argv/source identity exact;
- sentinel binds result, NPZ, Rerun validation, source, p15 and tag hashes;
- result binds plan, NPZ, RRD, RBL, validation, and both PNG hashes;
- exactly one ordered `run_claim -> results_durable -> preclose_sentinel_durable ->
  simulation_app_close_start` sequence (optionally followed only by
  `simulation_app_close_returned`), with an exact key set for every row, finite
  nondecreasing timestamps, `failure_marker_exists=false`, and source/prereg/p15/result/
  sentinel/internal-verdict fields recomputed from the actual files;
- exactly one complete stdout line reconstructed from the result rather than token
  searched: profile, `scientific_verdict_preclose_candidate`, Boolean success count,
  and the frozen active denominator (`5` here, `10` canonical) must all match byte for
  byte.

Only this combined gate starts the render child.  If raw status is zero but the semantic
gate fails, render attempt count is zero and combined exit status is reserved value
`125`; a nonzero raw child status remains its own normalized value.  The outcome embeds
the complete `T3U_PHYSICS_PRECLOSE_SEMANTIC_GATE_V1` recomputation.  Terminal attestation
and canonical promotion independently rerun the same pinned pure verifier; they do not
trust its stored boolean.

## 5. Reactive repair C — raw exit zero is not render success

The same Kit close behavior can occur in the isolated render child.  Render success is
therefore no longer raw wait status alone.  After the render child group is reaped,
Supervisor V4 independently loads p16 only as a pure file validator and requires:

- ordinary raw exit zero, no timeout/signal action, and an empty child process group;
- no `failure.json`, plus nonempty result/plan/NPZ/frozen-source/phase/manifest/MP4;
- exactly 234 uniquely named `frame_0000.png` through `frame_0233.png` files, each bound
  to its manifest row, frozen NPZ sample, clock snapshot, body/object transform and joint
  state;
- the full manifest schema and exact 20 fps cadence; then an independent supervisor-side
  decode of all 234 PNGs and every MP4 frame, checking PNG format/mode/resolution and
  MP4 count/frame-byte-length/resolution/fps in addition to hashes; zero observed physics
  callbacks/clock delta/scenes/explicit step calls; and physics-finalize ->
  render-start -> render-end dependency hashes all independently recomputed;
- the authoritative NPZ cadence is independently regenerated as physics steps
  `1..2340`, simulation time `step/200`, and the exact seven phase-id/phase-step arrays;
  the 234 rendered samples must therefore be exactly steps `10,20,...,2340`;
- the clock-audit ledger has exactly 491 ordered labels: one `context.new_stage`, 250
  `app.update`, and 240 Replicator calls, including the six warm-ups and all 234 frames.
  A shortened but internally self-consistent audit list is invalid.  Every count/step
  field is a strict JSON integer (Boolean forbidden), and every time/delta/error/fps
  field is a finite JSON float (Boolean forbidden), including all 491 before/after
  snapshots, frame indices/phase IDs/fidelity values, the render phase, and independent
  decode counts/size/fps;
- an immutable physics `phase.jsonl` ending at SimulationApp close, plus a separate
  one-row `render_phase.jsonl` containing exactly `render_trace_durable` and binding the
  manifest/MP4 hashes; and one profile/count/path-exact `RENDER_TRACE_COMPLETE` record.

The outcome embeds `T3U_RENDER_POSTHOC_SEMANTIC_GATE_V1`.  A raw-zero render missing or
contradicting any evidence becomes reserved combined status `125`, never supervisor
PASS.  A nonzero raw render status remains its normalized value.  Terminal attestation
and canonical promotion rerun the pinned pure render verifier and demand exact equality
with the stored gate; neither trusts `manifest.pass`, `outcome.pass`, or a stored map.
All supervisor-side `nvidia-smi` probes are bounded by 15 seconds.  The independent
PNG/MP4 decode is a CPU semantic check after the render child has exited; that decode is
not subject to the 15-second `nvidia-smi` timeout.  Its paired ffprobe metadata check
and ffmpeg RGB framehash decode each have a separate fixed 120-second timeout.  The
supervisor rechecks its TERM/INT flag immediately after each physics/render semantic
gate and before spawning render or writing a successful outcome; a signal observed
during a bounded decoder can therefore never produce combined PASS.

## 6. Reactive repair D — useful terminal evidence for an aborted run

If preflight2 aborts before science artifacts exist, `--terminal_attest
side_preflight2` takes the upstream-failure branch before scanning for successful-run
artifacts.  It requires and binds the failure marker, frozen source/argv/phase, raw child
wait status, semantic-gate failure, reserved nonzero combined status, render-not-started,
PID/PGID cleanup, supervisor contract/outcome, file hashes, and GPU before/end/after
inventories.  `nvidia-smi` has a 15-second hard timeout.

On valid failure evidence it writes forward-only
`t3u_side_preflight2_terminal_attestation.json` with artifact
`T3U_EXTERNAL_TERMINAL_ABORT_ATTESTATION_V2`, `attestation_valid=true`,
`promotion_allowed=false`, and `pass=false`.  Missing NPZ/RRD/PNG/MP4 is then
reported as absent science, not as an attestor crash and not as zero physics steps.
This failure attestation can never satisfy the canonical promotion schema.

If physics preclose is complete but rendering fails, p16 writes forward-only
`render_failure.json` plus a separate `render_failure` render-phase record before
calling `SimulationApp.close()` when
an exception is observable.  Supervisor semantic failure remains the independent
authority even if Kit suppresses that Python exception or a completed-looking manifest
fails recomputation.  The terminal verifier separately rechecks both child raw wait
records, the passing physics gate, failing render gate, exact phase/failure evidence,
reserved-or-raw nonzero combined status, bindings, process groups and GPU inventories.
Valid evidence produces `T3U_EXTERNAL_TERMINAL_RENDER_ABORT_ATTESTATION_V1` with
`pass=false`, `scientific_artifacts_complete=false`, and promotion forbidden.  It can
never be mistaken for the successful `T3U_EXTERNAL_TERMINAL_ATTESTATION_V4` schema.
If close/finally prevents the child from writing a render failure row, exact absence of
both render failure files is accepted only together with the recomputed failing render
gate and nonzero/reserved supervisor outcome; absence can never become render PASS.
All pure JSON/phase/raw-wait/binding and GPU-text parsing is completed before the
one-shot `nvidia_smi_after.csv` file is created.
Keeping render failure/phase bytes separate is essential: it leaves the already-passing
physics failure marker and phase hash byte-identical, so terminal recomputation can
compare the pre-render physics gate exactly rather than pretending later appends existed.

Every successful-run, physics-abort, render-abort, and canonical-promotion verifier uses
one shared strict raw-lifecycle decoder.  PID/PGID/SID are JSON integers greater than
one; attempt counts, raw wait status, exit/signal values and combined status are exact
non-Boolean integers; timestamps/durations are finite JSON floats in monotonic order;
TTY, timeout, reap and pass fields are exact Booleans.  The decoder reconstructs
`waitpid` exit/signal meaning rather than trusting stored decoded fields, and validates
the exact child command, process group, session, cleanup actions, GPU PID inventories,
and supervisor identity.  Recursive JSON comparisons require both type and value to
match, so `false` can never alias integer zero and `true` can never alias integer one in
the V4 contract, stored semantic gates, outcome, or terminal attestation.  Canonical
promotion reruns this same verifier and a second direct raw-lifecycle check.

## 7. Forward-only outputs and detached execution

All outputs use only `t3u_side_preflight2_*` under
`claudedocs/runtime_logs/grasp_track/g0b_d420/`.  Existing preflight1/p15/p14/p10 files
are read-only.  The supervisor performs one physics attempt and at most one render
attempt, with no retry or tag reuse.

Exact launch block after independent static GO:

```bash
(
  set -o noclobber
  p16_case_dir=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/g0b_d420
  p16_profile=side_preflight2
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

After the recorded supervisor and child group are reaped, run exactly:

```bash
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  /home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics.py \
  --terminal_attest side_preflight2
```

If and only if the successful V4 attestation passes, the user manually opens both PNGs,
the RRD, and the full MP4 and writes a hash-bound
`t3u_side_preflight2_manual_visual_inspection.json`.  Canonical remains blocked until all
successful-run terminal and manual fields are independently recomputed and PASS.

## 8. Frozen dependency pins

- supervisor source SHA-256:
  `527b06e5b9a090f4207c5f9ac5feb539c4b26f4c23f48ac59e4d802a153fa365`;
- p15 source: `250a3f406f83d3b0cc95be7ccdc666d043e28eb5b5c0f9fb25e450e26ee17240`;
- p15 prereg: `23acb036cd1a26f577cff8145ef4031f1c4075af3e4e60f1df28a42d86da8330`;
- p15 candidates: `67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384`;
- p14 helper: `fcaa7b1c6aeea65cd7fd335d9cd17ee5424a53d81764f67642d074a28e3e0133`;
- p10 IK/planner: `63c6b2127d969e3291da6943eab6da1037034c154a8f21fe447519cbcb2f6cff`;
- attempt3 jaw extractor: `bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3`;
- frozen URDF: `64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2`.

P16 pins the final full SHA-256 of this prereg and the canonical prereg.  It records its
own executed-source hash rather than embedding a self-reference.  One profile-specific
path function defines the complete decision dependency set: p16 source/supervisor and
both preregs, p10/p14/p15 and every bound p15 config/output/inspection/manual/exit/stdout/
PID file, the preflight1 historical witnesses, workspace witnesses, environment sources,
URDF, jaw extractor, and all five attempt3 USD layers.  Canonical adds every passing
preflight science/render/lifecycle/manual file.  The exact key set and hashes must match
physics start, physics finalize, render start, render end, and terminal-time current
rehash; editing or adding a failure marker between these boundaries aborts.
