# p16 v10 / t3u_side_preflight11 preregistration — event-latched witness safe retreat

Status: PREREGISTERED / NOT RUN / G0 BLOCKED UNTIL INDEPENDENT AUDIT
Case: g0b_d420
Run profile: side_preflight11
Canonical prefix: t3u_side_preflight11
Date: 2026-08-12 KST

This is a forward-only reactive instrumentation repair. It does not edit, replace,
reinterpret, or promote any t3u_side_preflight10 artifact. The frozen executable is
sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v10.py and the detached
supervisor is
sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v13.py. The
self-referential p16 SHA-256 value is intentionally labelled
TO_BE_FROZEN_P16_V10_SHA256; the independently frozen Supervisor V13 SHA-256 is
4a96ae775684da4d8f57be288fd1a2fcff1226f71ba850fc06c55fec260c6b78.

이번 case의 신규 변수: [side-midpoint grasp point, SDG candidate pose].

The two scientific variables above are unchanged. The event latch and witness-only
target override are a reactive observability/safety repair forced by the completed
preflight10 numerical failure; they are not a third scientific variable and never enter
the five-row scientific denominator.

## 1. Frozen inheritance and authorization boundary

Everything in t3u_side_preflight10_prereg.md is inherited byte-for-byte unless this
document explicitly replaces it. The inherited prereg SHA-256 is
d75d790d7f5fff65af966fda928b0726907778a6361d71a06701973d8e3e26ee.
The inherited canonical prereg
t3u_side_phys1_preflight10_prereg.md has SHA-256
2266679959cc670180ecf521ed0c04121b0906c893e936b6307d99f04fafeb9b.
Frozen p16 v9 and Supervisor V12 remain immutable predecessor evidence:

- p16 v9:
  3c61f041d7013592c176770432e6f22a825d7c0a5b9ea1ae3ed9dc097dbad04c;
- Supervisor V12:
  49c3ae4455e02706934ee5a8eb3e62d21d2231e724ecee9efb92e499fdb5565d;
- p15 candidate JSON:
  67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384.

The active five rows, cylinder, robot, support, materials, controller, candidate order,
command schedule, phase lengths, all scientific thresholds, camera, Rerun contract, and
the exactly 2 diagnostic plus 2,340 task-physics callbacks are unchanged. No Isaac,
render, or hardware execution is authorized by this preregistration. G0 may be consumed
only after an independent audit has accepted the four new forward-only files and their
final hashes, followed by a separate user-authorized host launch.

## 2. Exact retirement of completed preflight10

t3u_side_preflight10 is immutable completed-preclose semantic-rejection evidence, not
an Isaac crash and not grasp science. Physics returned raw and normalized exit 0 after
the full durable preclose. Supervisor V12 correctly admitted no render, recorded one
physics attempt and zero render attempts, emitted combined reserved exit 125, reaped
the physics process group, and observed no fresh NVIDIA PID. failure.json is absent.

The original historical input inventory is exactly the following 23 regular files.
Every name and full SHA-256 is normative; no reduced subset can satisfy retirement:

| File suffix | SHA-256 |
|---|---|
| argv.txt | aa4f5f293fd7b272b636b6ed828f52793e4ee28c302b9cb4593f7233f9d240ef |
| decision_snapshot.png | 48c6ae71c5fd14e21844f6f45bb6cdcd0786eecf98a67e4c07e2105f189b7f20 |
| exit_status.txt | a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca |
| inspection.png | eb9b6a1fb6eef5e75762c3834308191a3c5f2e8cd95ce99448675cfdd70f0baf |
| nvidia_smi_before.csv | 4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0 |
| nvidia_smi_supervisor_end.csv | 4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0 |
| pgid.txt | c96089507b59077c2408d85b8594b64a128a7944baf028270adddddfee8ed66f |
| phase.jsonl | e720332b92ae0cfb54716483898b061873d665f0052e92aeaf4f2cc1cd0d4fd6 |
| physics_python_pid.txt | ebb186bfadb2e13d99e7573bc593ea4aed0ffd172ad59048164495141c30a18c |
| plan.json | b71bb97485448b994400397a3b178a9321014539b1bc0a681b5b4e85e8699363 |
| preclose_sentinel.json | eef3f1608096de962689cc2ca9b1225fda8a727059c3a5df1cfa77bd64071e5c |
| prereg.md | d75d790d7f5fff65af966fda928b0726907778a6361d71a06701973d8e3e26ee |
| rerun_validation.json | 6b5f49e56a8f5d17049ab09f929791199aa47c7171e79630ba47cff3a0f81c56 |
| results.json | 38c468764aa5a844735a5b60c71713a9c9f1f2e3ca594366b0feaaa2a3d463ce |
| script.py.txt | 3c61f041d7013592c176770432e6f22a825d7c0a5b9ea1ae3ed9dc097dbad04c |
| stdout.log | 3b4f101cec4a11ca99694780df13f063a886b9ef18b2c6bc85476c16377f848a |
| supervisor_contract.json | f71fcc4849d25b5e02daf5d6c8745a0fdca981b2b4b4bde8880ddcf2ae4e7db5 |
| supervisor_launcher.log | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| supervisor_outcome.json | 1b2234553abe91e173ec7ace778e9b85da84d95ee9875c728fff355747a14ddc |
| supervisor_pid.txt | c96089507b59077c2408d85b8594b64a128a7944baf028270adddddfee8ed66f |
| timeline.rbl | ce6c018f8250289aaea1584936acbadf6976f996fd9dfb049959a988ec35fcf7 |
| timeline.rrd | cfca8a93e9a9ade7fd47c32a9d640096c2115ca081f6e19071865d0e9f67980d |
| trace.npz | e031bce6f9cc4b7c6146f621ba2fe884008818e985bb6f247d5ec4410d339930 |

The 23-file set is the immutable upstream input to the new completed-preclose terminal
branch. A later derived terminal_attestation.json, if separately authorized, is not one
of those historical inputs and must never modify or replace any of them. Before such an
output exists the prefix has no failure.json, render child PID, render failure, render
manifest, MP4, manual visual record, nvidia_smi_after.csv, terminal attestation, or
frames directory. A validator must distinguish these intentional absences from missing
preclose evidence.

The preclose phase ledger is exactly:

1. run_claim;
2. results_durable;
3. preclose_sentinel_durable;
4. simulation_app_close_start.

The nested result semantic map has exactly two false leaves:

- numeric_trace_metrics_quaternions_counts_recomputed = false;
- runtime_instrumentation_recomputed_not_trusted = false.

Every other frozen result semantic leaf is exact Boolean true. At the supervisor
preclose-gate layer, the only false semantic check is
pinned_result_semantic_validator_exact_all_true. These two levels must not be conflated.

The sole nonzero numeric failure count is
actual_joint_position_inside_parsed_urdf_limits = 874. It occurs only in excluded
instrumentation witness environment slot 5; all five active rows pass their numerical
audits. This remains an instrumentation failure because the run-level witness is part of
measurement validity. Therefore preflight10 stays pass=false,
runtime_instrumentation_pass=false, scientific_authoritative=false, and non-promotable.
Its tempting five-row contact/lift pattern may be reported only as diagnostic,
non-authoritative evidence.

## 3. Reactive repair — earliest valid contact event

### 3.1 Witness identity

The preflight has exactly eight environments. Scientific rows are exactly slots 0:5.
The excluded instrumentation witness is exactly slot 5. Slots 6 and 7 remain padding.
The witness source trial, object, support filters, contact reporter, phases, and original
command schedule are inherited unchanged.

### 3.2 Event predicate and one-way latch

For every 1-based task step s, after that step's physics callback and fresh sensor/state
read, define event_predicate(s) as the conjunction of:

- the witness moving-gripper/support force scalar is finite and strictly greater than
  0.02 N;
- the same moving-gripper/support raw contact count is an exact non-Boolean integer
  strictly greater than zero;
- all six same-step witness actual joint positions are finite;
- all six same-step witness actual joint positions are within their parsed URDF lower
  and upper limits, inclusive.

This latch predicate uses zero joint-limit tolerance. It is intentionally stricter than
the inherited full-run numerical audit described below.

The latch event is the earliest task step for which this complete predicate is true.
The latch is one-way: once true it can never clear, move, or be replaced by a later
event. Force alone, raw count alone, a non-finite value, a planned target, or a
post-hoc hand-selected step can never fire it.

The historical preflight10 trace first crossed force 0.02 N near task step 1120 and
first violated a witness joint limit at task step 1137. Those are forensic observations,
not a trigger constant. P16 v10 and its validator must independently recompute the
earliest complete event from the new trace. No code path may hardcode step 1120.

The new run must latch by 1-based task step 1136. If no complete event has latched after
step 1136, execution aborts fail-closed before issuing any command for or executing task
step 1137. There is no silent stop-check, arbitrary fallback event, reset, or best-effort
continuation.

### 3.3 Next-step target-only safe RETRACT+CLOSE

The event step itself retains the inherited authoritative planned command and applied
target readback. Beginning on exactly event_step + 1 and on every remaining task
iteration through step 2340, only the authoritative planned joint target at witness
environment slot 5 is replaced by:

[22.147551724447293,
 54.009710735468865,
 84.68324826073442,
 -26.586518474346644,
 90.0,
 66.4] degrees.

The first five values are exactly WITNESS_Q_APPROACH_DEG[:5]; the sixth is the frozen
closed gripper target q5 = 66.4 degrees. This is the preregistered target-only
RETRACT+CLOSE command. It is not a direct state correction.

The vector above is the float64 design/provenance authority. The runtime command path
is also frozen: `torch.as_tensor(design, dtype=torch.float32) * pi / 180`, followed by
the existing `torch.rad2deg` trace serialization. Its authoritative serialized
float32-degree vector is exactly:

[22.147552490234375,
 54.00971221923828,
 84.68325805664062,
 -26.586519241333008,
 90.0,
 66.40000915527344].

The corresponding six uint32 bit patterns are exactly
`[0x41b12e30, 0x425809f2, 0x42a95dd4, 0xc1d4b131, 0x42b40000,
0x4284ccce]`. The trace validator uses dtype/shape checks plus `array_equal` against
those bytes for every post-activation planned and applied witness row. No epsilon is
allowed for this safe-target gate; changing any component by one float32 ULP fails.

Forbidden operations are absolute:

- no write to joint position q or velocity qd;
- no robot, object, support, or sensor-state write;
- no teleport, reset, set-root-pose, set-root-velocity, or direct PhysX state write;
- no extra simulation, render, forward, update, or physics callback;
- no command change in active slots 0:5 or padding slots 6:8;
- no altered threshold, phase duration, counter, clock, candidate, object, or path.

If target-only control cannot keep the witness finite and strictly within every parsed
URDF limit through task step 2340, the run fails closed as instrumentation invalid. It
must not reset, clamp state, stop checking, or improvise another target.

The all-environment full-run actual-q audit retains the already frozen p10 numerical
tolerance of 1e-5 rad = 0.0005729577951308233 degrees and recomputes it directly from
the raw NPZ rather than trusting producer failure counts. This is required for logical
compatibility with the exact frozen active trace: that trace contains 3,038 active
environment-steps whose wrist_r values cross the decimal parsed lower bound by at most
0.00023520963026157915 degrees, while all are inside the inherited 1e-5 rad tolerance.
Preflight11 must therefore require zero all-environment violations under that frozen
tolerance, separately require zero strict witness violations, and report the strict
all-environment boundary-exceedance count as informational. This does not authorize a
new tolerance, command change, clamp, or removal of the all-environment audit.

## 4. Required trace and independent recomputation

The durable NPZ must record every one of the 2,340 task steps, including at minimum:

- exact 1-based task_step and frozen phase/phase-step arrays;
- all_env_joint_planned_target_deg, defined as the authoritative command after the
  witness-only override; all_env_joint_applied_target_deg, defined as the environment
  target readback after that command; and actual joint position/velocity arrays, each
  with the corresponding all-environment shape (2340, 8, 6);
- witness moving-gripper/support force and exact raw count for every step;
- event predicate, one-way latch state, and override-active Boolean for every step;
- the existing fixed-base, moving-body, object, support/object, self-contact,
  object-contact, clock, and callback arrays required by preflight10.

Stored summaries are not trusted. The pure semantic validator independently recomputes
all of the following directly from NPZ arrays and parsed URDF limits:

1. the earliest complete event and deadline compliance;
2. latch false before the event, true at and after it, with exactly one rising edge;
3. override false through the event and true from event + 1 through step 2340, with
   exactly one rising edge;
4. authoritative planned and applied-readback witness targets equal to the inherited
   schedule through the event, then both exact-equal to the safe target on every step
   thereafter using the exact serialized float32 vector and uint32 patterns pinned in
   section 3.3; one ULP fails. The pre-override nominal witness schedule is not an
   authority and need not be stored;
5. all eight environments' planned/applied/actual q and qd finiteness, zero actual-q
   violations under the frozen 1e-5 rad audit tolerance on every executed step, and
   zero strict witness violations across all 2340 steps;
6. zero state/object/reset/teleport/extra-step operations and exactly one target write
   path restricted to witness slot 5;
7. exactly 2 diagnostic callbacks, 2340 task callbacks, and 2342 combined callbacks,
   with task counters ending at 2340 and the inherited observed-clock fsum contract;
8. exact equality of the five active rows' plan, commands, thresholds, paths, target
   arrays, actual state arrays, moving-body/object trajectories, and contact evidence to
   the pinned preflight10 trace, using equal-NaN semantics only where the inherited
   contact-position schema permits NaN for raw_count=0;
9. no active-science write or counter/clock/path drift caused by the witness branch.

The validator must require exact Boolean types, exact array shapes/dtypes, no extra or
missing semantic keys, and stored-summary equality to recomputation. Any disagreement
sets numeric_trace_metrics_quaternions_counts_recomputed=false,
runtime_instrumentation_recomputed_not_trusted=false, and prevents render admission.

## 5. Supervisor V13 completed-preclose terminal branch

Supervisor V13 and p16 v10 add a distinct terminal branch before the generic abort
branch. It covers the frozen t3u_side_preflight10 case described in section 2 and any
future completed-preclose p11 semantic rejection; it never covers a child crash or
partial preclose. Its input contract requires simultaneously:

- exact original 23-file names and hashes;
- profile side_preflight10, one physics attempt, zero render attempts;
- physics raw/normalized exit 0 and full durable preclose semantic documents;
- failure.json absent and render child/artifacts absent;
- exactly the two false result leaves named in section 2, all other result leaves true,
  and the stored Supervisor V12 preclose gate exactly equal to recomputation;
- combined exit status 125;
- physics process group reaped with no remaining group member;
- no fresh NVIDIA PID at supervisor end.

The historical branch independently enforces the exact V12 schema; the forward branch
enforces the exact V13 schema. In both, the outcome top level has exactly these 22 keys
and no others: `artifact`, `profile`, `argv`, `supervisor_source_sha256`,
`p16_source_sha256`, `candidates_sha256`, `start_time_unix`, `end_time_unix`,
`elapsed_seconds`, `supervisor`, `attempts`, `physics`, `physics_artifact_gate`,
`render`, `render_artifact_gate`, `render_started_iff_physics_success`,
`combined_exit_status`, `gpu`, `bindings`, `contract`, `host_launch_context`, and
`pass`. V12's exact contract omits the two new branch keys. V13's exact contract adds
only `completed_preclose_semantic_rejection_terminal_branch=true` and
`completed_preclose_semantic_rejection_terminal_artifact=`
`T3U_EXTERNAL_TERMINAL_COMPLETED_PRECLOSE_SEMANTIC_REJECTION_ATTESTATION_V1`.

The validator also exact-binds supervisor/source/candidate hashes and paths, supervisor
argv, physics child argv, host-launch context, supervisor PID/PGID/SID identity, raw
wait-status decoding and successful child lifecycle, attempts exactly 1-0 with zero
retry, outcome times, render and render-gate null, the render-started iff field exact
true, combined exit 125, pass exact false, and every outcome file binding. Plain
truthiness or Python Boolean/integer equality is insufficient.

At the outer `physics_artifact_gate.semantic_checks` layer, all 16 keys and exact
Boolean value types are required and the false set is exactly the singleton
`{pinned_result_semantic_validator_exact_all_true}`; every other outer value is true.
The nested result map is checked separately against its complete version-matched keyset
and exact Boolean types. Historical p10 requires exactly the two section-2 inner false
keys; a future p11 rejection requires a nonempty inner false set while retaining the
same singleton outer false set. Thus concurrent stdout, sentinel, source, argv, schema,
or lifecycle failures cannot be hidden behind the expected result-semantic rejection.

This branch must not synthesize failure.json or misclassify semantic rejection as an
upstream physics crash. Its derived attestation has attestation_valid=true because it
faithfully attests the rejection, but pass=false, promotion_allowed=false,
scientific_artifacts_complete=false, and scientific_authoritative=false. It is always
non-promotable and cannot satisfy canonical admission. The existing generic abort and
render-abort branches remain separate and unchanged.

## 6. Scientific and lifecycle verdicts

Preflight11 is still an instrumentation preflight. Even if every gate passes, it proves
only that the contact and numerical instrumentation are ready for canonical execution.
It cannot claim a side-grasp success or failure.

The preclose result may proceed to render only if every exact semantic leaf is true,
including the new event/override, all-env numeric, active-state identity, provenance,
Rerun, and lifecycle gates. Any false leaf maps to reserved semantic exit 125, no render,
and a non-promotable terminal record. Raw child exit 0 is never sufficient.

Success requires the inherited render contract, terminal success attestation, and
separate human visual inspection. Canonical side_phys1 remains blocked until all three
exist, parse, hash-bind, and independently recompute true.

## 7. Forward-only files, freeze labels, and launch substitutions

Only the following four new files belong to this G0 preparation:

- sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v10.py;
- sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v13.py;
- this t3u_side_preflight11_prereg.md;
- t3u_side_phys1_preflight11_prereg.md.

P16 v10 must pin the final full SHA-256 of both prereg documents and Supervisor V13.
Supervisor V13 must invoke only p16 v10. The two prereg self-document labels are
TO_BE_FROZEN_PREFLIGHT11_PREREG_SHA256 and
TO_BE_FROZEN_CANONICAL_PREFLIGHT11_PREREG_SHA256 until that one-way freeze.

The normative detached host wrapper is exactly the preflight10 section-8 wrapper with
only these forward substitutions:

- p16_profile=side_preflight11;
- t3u_side_preflight10 prefix becomes t3u_side_preflight11;
- p16 v9 becomes frozen p16 v10;
- Supervisor V12 becomes frozen Supervisor V13;
- both prereg expected hashes become their final preflight11 values.

The wrapper remains one host-authorized, no-retry command with the inherited PID-1 and
ancestor sandbox guard, NVIDIA probe, no-TTY/session/process-group contract, bounded
liveness guard, cleanup, output absence checks, and final terminal plus manual-visual
authorization boundary. No launch command in this document is permission to run Isaac.
