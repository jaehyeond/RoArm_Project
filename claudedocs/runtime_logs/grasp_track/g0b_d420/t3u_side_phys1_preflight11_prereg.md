# p16 v10 / t3u_side_phys1 preregistration — preflight11-bound canonical PhysX grasp

Status: BLOCKED ON COMPLETE t3u_side_preflight11 PASS + TERMINAL + MANUAL VISUAL / NOT RUN
Case: g0b_d420
Canonical profile: side_phys1
Canonical prefix: t3u_side_phys1
Date: 2026-08-12 KST

This is the sole forward canonical protocol paired with
t3u_side_preflight11_prereg.md. The executable is
sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v10.py and the detached
supervisor is
sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v13.py. Their final hashes,
and both preflight11 prereg hashes, remain
TO_BE_FROZEN_P16_V10_SHA256,
4a96ae775684da4d8f57be288fd1a2fcff1226f71ba850fc06c55fec260c6b78,
TO_BE_FROZEN_PREFLIGHT11_PREREG_SHA256, and
TO_BE_FROZEN_CANONICAL_PREFLIGHT11_PREREG_SHA256 until the independent pure audit.

No canonical Isaac execution is authorized now. Only side_preflight11 may be considered
for a later separately authorized run, and this canonical remains blocked until that
preflight's complete success evidence exists.

## 1. Scientific question and frozen variables

Under the user-approved sim-only D419 exception, can the fixed-base RoArm-M3-Pro grasp
the upright D29 x H50, 24.83 g cylinder across its side midpoint and lift it by contact
physics without kinematic attachment, desk/support collision, non-jaw shoving, or
self-collision?

This is a local existence test, not a workspace atlas or real-hardware claim. One valid
passing active row is sufficient for
SIDE_MIDPOINT_GRASP_PASS_IN_FIXED_SIM_CONTROLS. If none passes, the deepest populated
frozen failure branch must be reported.

이번 case의 신규 변수: [side-midpoint grasp point, SDG candidate pose].

No new scientific variable is introduced by preflight11. Its event latch is excluded
witness instrumentation only.

## 2. Inheritance and predecessor retirement

The complete scientific protocol, geometry, controller, thresholds, causal verdict
ladder, Rerun requirements, render cadence, and host lifecycle in
t3u_side_phys1_preflight10_prereg.md are inherited except where this document explicitly
replaces preflight10 admission with preflight11 admission and p16 v9/Supervisor V12 with
p16 v10/Supervisor V13. The retired canonical prereg SHA-256 is
2266679959cc670180ecf521ed0c04121b0906c893e936b6307d99f04fafeb9b.

The forward preflight document is authoritative for the event predicate, safe target,
trace schema, p10 retirement, and completed-preclose terminal branch. Its final hash is
TO_BE_FROZEN_PREFLIGHT11_PREREG_SHA256. All exact values from its sections 2 through 7
are incorporated here.

t3u_side_preflight10 remains immutable completed-preclose semantic-rejection evidence:

- its original upstream inventory is exactly the 23 names and hashes listed in
  preflight11 section 2;
- physics raw/normalized exit was 0 after all 2 diagnostic and 2340 task callbacks;
- failure.json and render child/artifacts were absent;
- exactly
  numeric_trace_metrics_quaternions_counts_recomputed and
  runtime_instrumentation_recomputed_not_trusted were false in the result semantic map;
- the only nonzero numerical failure count was
  actual_joint_position_inside_parsed_urdf_limits = 874 in excluded witness slot 5;
- Supervisor V12 attempts were physics 1/render 0, combined exit was 125, the group was
  reaped, and no fresh NVIDIA PID appeared.

Supervisor V13's distinct completed-preclose rejection branch may validly attest those
facts, but that attestation is necessarily pass=false, promotion_allowed=false,
scientific_artifacts_complete=false, scientific_authoritative=false, and cannot satisfy
this canonical gate. It must not synthesize failure.json. A generic abort attestation or
an arbitrary all-true JSON map also cannot satisfy admission.

That branch is dispatched before generic abort and inherits the forward prereg's strict
terminal schema: exactly the 22 outcome keys listed there; exact version-specific V12
or V13 contract; pinned source/candidate/supervisor paths and hashes; exact supervisor
and physics argv; host context; supervisor/child PID, PGID, SID, raw wait, retry,
timeout, signal, reap, and time contracts; render absent with zero render attempts;
exit 125; exact bindings; and no concurrent stdout/sentinel/source/argv failure. The
outer semantic map must have the exact singleton false set
`{pinned_result_semantic_validator_exact_all_true}` with every other exact-Boolean leaf
true. Its complete nested result map is validated separately. Historical p10 permits
only its two pinned inner false leaves; any later p11 semantic rejection must still have
a nonempty inner false set without changing the outer singleton contract.

All earlier preregs and preflights remain retired historical evidence. No old result,
terminal, manual record, run label, or prereg path can substitute for a complete
t3u_side_preflight11 success.

## 3. Exact canonical population

Canonical environment count is exactly 64. Exactly eight p15 candidates crossed with
five frozen pinch offsets produce 40 ordered planned rows. Before environment creation
and again after transformed-hull clearance, the active set must be exactly 10 ordered
rows: candidates 5 and 7, each with offsets 0 through 4. Any different nonempty set is
contract drift and aborts rather than changing the experiment.

The first padding environment after the 10 active rows is the excluded instrumentation
witness:

- active_count = 10;
- active science slice = slots 0:10;
- witness_slot = active_count = 10;
- slots 11:64 are padding;
- the witness never enters a scientific count, population mask, or verdict denominator.

Candidate order, the first visual representative, all five geometry-derived controls,
the 2340-step command schedule, phases, thresholds, objects, controller, camera, contact
schemas, and render schedule are unchanged from the inherited canonical protocol.

## 4. Canonical witness target-only contract

The actual authorized repair evidence is first established in preflight11 slot 5.
If canonical is later admitted, the same p16 v10 code applies it independently to the
canonical first-padding witness at slot 10. It must never reuse the preflight's event
step or target state.

For each 1-based canonical task step, the complete event predicate is independently
recomputed after the physics callback from:

- finite moving-gripper/support witness force strictly greater than 0.02 N;
- exact non-Boolean moving-gripper/support raw contact count greater than zero;
- all six same-step witness actual joints finite;
- all six same-step witness actual joints inside parsed URDF limits, inclusive.

The latch predicate uses zero joint-limit tolerance. The full-run all-environment audit
instead retains the inherited p10 tolerance of 1e-5 rad = 0.0005729577951308233 degrees,
recomputed directly from raw NPZ actual q. This is not a relaxation: the exact frozen
preflight active trace has 3,038 wrist_r boundary exceedance environment-steps of at
most 0.00023520963026157915 degrees, all inside that pre-existing tolerance. Canonical
requires zero all-environment violations under the frozen tolerance, zero strict
witness violations across all 2340 steps, and an informational strict all-environment
boundary-exceedance count. No command, clamp, state write, or tolerance tuning follows
from that informational count.

The earliest complete event latches once and never clears. It must occur by task step
1136 or canonical aborts before any command or physics execution for task step 1137.
The event step retains the original authoritative planned command and applied target
readback. Beginning exactly at event + 1 and continuing through task step 2340, the only
overridden authoritative planned target is slot 10:

[22.147551724447293,
 54.009710735468865,
 84.68324826073442,
 -26.586518474346644,
 90.0,
 66.4] degrees.

The first five entries are exactly WITNESS_Q_APPROACH_DEG[:5], and q5 is the frozen
closed value 66.4 degrees. The observed preflight10 force crossing near step 1120 is
historical context only. Neither preflight nor canonical code may hardcode that step.

That decimal vector is the float64 design authority. The exact existing command-path
serialization (`float32` degrees to radians, then `torch.rad2deg`) is
`[22.147552490234375, 54.00971221923828, 84.68325805664062,
-26.586519241333008, 90.0, 66.40000915527344]` float32 degrees, with uint32 patterns
`[0x41b12e30, 0x425809f2, 0x42a95dd4, 0xc1d4b131, 0x42b40000,
0x4284ccce]`. Every post-activation planned/applied witness row must be `array_equal`
to those bytes. This gate has no epsilon; one float32 ULP of drift fails.

This is applied-target-only control. Direct q/qd state writes, object/support/sensor
writes, teleport, reset, root-pose/root-velocity writes, state clamping, extra forward
or physics callbacks, arbitrary fallback targets, and silent stop-checks are forbidden.
If the target alone cannot preserve all-env numeric validity through step 2340, the run
fails closed and produces no science claim.

The active slice 0:active_count must be copied and compared before and after the witness
branch on every task iteration. Its planned and applied targets must be exact-equal, and
no active target/object/path/counter/clock write may originate from the branch.

## 5. Trace, semantic, and no-interference gates

The canonical NPZ retains every inherited science array and adds the complete latch
evidence. all_env_joint_planned_target_deg is the authoritative command after the
witness-only override; all_env_joint_applied_target_deg is the environment target
readback. Those arrays plus actual q and qd have shape (2340, 64, 6). The trace also
contains 1-based task/phase arrays, witness force, exact raw count, event predicate,
latch state, and override-active flag for all 2340 steps.

Stored summaries are not trusted. The pure validator independently recomputes:

1. the earliest complete event, deadline, and exactly one latch rising edge;
2. activation exactly event + 1 and exactly one override rising edge;
3. planned and applied-readback witness targets equal to the inherited schedule through
   the event, then both exact-equal to the bit-pinned serialized safe target thereafter;
   one ULP fails, and the pre-override nominal witness schedule is not an authority and
   need not be stored;
4. actual all-environment joint finiteness and zero parsed-URDF violations under the
   frozen 1e-5 rad audit tolerance on every executed task step, plus zero strict witness
   violations through all 2340 steps;
5. active slice 0:10 planned/applied target exact equality and absence of writes from
   the witness path;
6. exact 40-planned/10-active order, object/path/threshold identity, and unchanged
   2340-step science schedule;
7. exactly two diagnostic, 2340 task, and 2342 combined callbacks, task counters ending
   at 2340, no reset, no extra step, and inherited observed-clock fsum equality;
8. all existing numerical, quaternion, raw-count/capacity, contact-position,
   fixed-base, self-filter, support-filter, causal-mask, and Rerun contracts.

Every result semantic key must exist once with exact Boolean type. No extra key, subset,
truthy replacement, forged stored summary, or raw child exit 0 can pass. Any mismatch
is runtime instrumentation failure, blocks render, maps to reserved exit 125, and remains
non-promotable.

## 6. Preflight11 admission gate

Before AppLauncher, canonical admission must pin, load, and independently recompute the
complete side_preflight11 evidence. All of the following are jointly required:

- exact frozen p16 v10, Supervisor V13, both prereg, p15, helper, environment, URDF, and
  USD dependency hashes;
- a side_preflight11 result whose complete semantic map independently recomputes all
  true, including the event/override and active-no-interference gates;
- physics raw exit 0 plus full durable preclose, followed by one successful render
  attempt and combined exit 0;
- exact Supervisor V13 contract/outcome, child command and raw lifecycle, PID/PGID/SID/
  TTY, attempt/retry/timeout/signal/cleanup, phase/stdout/hash bindings, process reaping,
  and GPU cleanup;
- exact RRD/RBL/footer validation, fixed blueprint, headless decision screenshot, full
  render manifest/MP4 decode and clock mapping;
- a terminal success attestation that independently recomputes and recursively binds
  the same raw evidence with pass=true, promotion_allowed=true, and
  scientific_artifacts_complete=true;
- a separate human manual visual inspection record bound to both required PNGs and all
  prerequisite hashes.

Missing evidence is not false science; it is an admission block. The p10 rejection
attestation, a generic abort attestation, or a preflight11 semantic-rejection attestation
can never substitute.

## 7. Frozen scientific verdict ladder

Before physics, any mismatch in the exact 40-planned/10-active set is dependency drift.
Zero rows after the inherited enabled-collider/support clearance is
NO_PLANNED_STATIC_CLEARANCE_FEASIBLE_SIDE_PATH and is absence of PhysX evidence, not a
physical grasp failure.

If physics executes, measurement invalidity in any active row blocks all science.
Otherwise success in any active row proves the local existence result. If none succeeds,
the global verdict is the deepest populated branch in this unchanged order:

1. MEASUREMENT_INVALID;
2. SIDE_MIDPOINT_GRASP_PASS_IN_FIXED_SIM_CONTROLS;
3. TRACKING_OR_ARRIVAL_GATE_FAIL;
4. PREMATURE_JAW_CONTACT_BLOCKS_SIDE_GRASP;
5. NONJAW_OR_SUPPORT_COLLISION_BLOCKS_SIDE_GRASP;
6. NO_BILATERAL_SIDE_CONTACT;
7. BILATERAL_CONTACT_LOST_BEFORE_LIFT;
8. BILATERAL_CONTACT_BUT_NO_CORRECTED_LIFT;
9. OBJECT_LIFTED_BUT_TIPPED;
10. OTHER_EXACT_GATE_FAIL.

Corrected lift remains cylinder bottom-equivalent center rise after subtracting settled
rest pose, not TCP motion. Bilateral force remains the same-step scalar
max_t(min(F_fixed(t), F_moving(t))); independent jaw maxima do not prove grasp.

## 8. Outputs, lifecycle, and execution boundary

The required canonical prefix includes all inherited physics, NPZ, RRD/RBL, validation,
PNG, frame-manifest, MP4, source/argv, phase/sentinel, supervisor, terminal, and manual
visual artifacts. Supervisor V13 uses exactly one physics attempt, at most one render
attempt admitted only by full physics semantics, no automatic retry, bounded TERM/KILL
reap, and exact GPU residue audit.

The normative host wrapper is the preflight11 wrapper with only:

- p16_profile=side_phys1;
- prefix t3u_side_preflight11 replaced by t3u_side_phys1;
- canonical prereg selected instead of preflight prereg;
- environment count 64, active_count 10, and witness_slot 10 as frozen above.

It remains one separately authorized host command using frozen p16 v10 and Supervisor
V13 hashes. This document does not authorize it. Canonical G0 remains blocked until a
fresh preflight11 has complete passing physics, render, terminal, and human visual
evidence and a later user explicitly authorizes canonical execution.
