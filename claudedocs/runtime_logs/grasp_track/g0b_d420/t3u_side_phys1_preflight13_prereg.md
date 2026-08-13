# p16 v12 / t3u_side_phys1 canonical preregistration — blocked after preflight13

Status: BLOCKED ON COMPLETE `t3u_side_preflight13` PASS + TERMINAL + MANUAL VISUAL / NOT RUN

Canonical profile: `side_phys1`

Canonical prefix: `t3u_side_phys1`

Executable: `sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v12.py`

Detached supervisor: `sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v15.py`

Supervisor V15 SHA-256:
`64cece0c57ce0b5fb713f67c69efac6724e5b31ba16f0c0d0454294442aebeb3`.
This document's final SHA-256 and the preflight13 preregistration's final SHA-256 are
pinned by p16 v12 before source freeze. The final source SHA is published in the freeze
receipt, avoiding a circular self-hash dependency.

This preregistration does not authorize canonical Isaac, PhysX, Kit, render, cloud, or
hardware execution. It fixes the canonical decision rule before any preflight13 output
exists. 이번 case의 신규 변수 remains exactly [side-midpoint grasp point, SDG candidate
pose]; validator repair is not a scientific variable.

## 1. Frozen inheritance and admission barrier

All scientific, physics, CUDA, time, lifecycle, RRD/PNG/MP4, and verdict clauses in
`t3u_side_phys1_preflight12_prereg.md` (SHA-256
`3386376e191addcafef893b9ed3698b244daa2b352349f90bc536341212893eb`)
remain exact unless this document explicitly replaces a validator clause. The p13
preflight delta is defined by `t3u_side_preflight13_prereg.md` and applies identically to
canonical validation: path-and-SHA module isolation, exact p12 five-false retirement,
command-only witness isolation, and numeric/measurement-valid separation.

Canonical G0 remains blocked until a fresh preflight13 has all of the following:

- exact frozen p16 v12, Supervisor V15, both p13 preregs, p15, p14/p10 helpers, URDF,
  environment, collision assets, and all historical dependency hashes;
- exactly one raw-zero physics child, all semantic leaves true, exactly one successful
  render child, no retries, all groups reaped, and no fresh GPU PID delta;
- exactly 2 diagnostic and 2,340 task callbacks with the frozen callback-clock contract;
- terminal success attestation, RRD/RBL/footer validation, fixed blueprint, headless
  decision snapshot, render manifest/MP4 decode and time mapping, and an explicit passing
  manual visual inspection;
- a preflight instrumentation verdict only, never a promoted grasp verdict.

Any missing, mismatched, false, or non-exact item keeps canonical blocked. Preflight12 is
historical nonpromotable evidence and cannot satisfy this barrier.

## 2. Canonical population and unchanged science

Canonical environment count is exactly 64. Exactly eight p15 candidates cross the
static-clearance gate. Exactly ten rows are scientifically active: candidates 5 and 7,
each at offsets 0 through 4, in the frozen order. Any other nonempty active set fails.

The excluded instrumentation witness is exactly slot 10; slots 11 through 63 are padding.
It never contributes to the scientific mask, count, denominator, or verdict. The active
denominator is exactly 10.

The upright analytic D29 x H50 mm, 24.83 g cylinder, fixed-base RoArm, support, gravity,
materials, collision filters, contact reporters, parsed URDF joint limits, controller,
target poses, thresholds, camera, and candidate order are unchanged. Phases remain settle
120, approach 400, stage 400, descend 400, close 400, hold 120, lift 500 at 200 Hz: exactly
2 diagnostic, 2,340 task, and 2,342 combined callbacks. Contact remains strictly greater
than 0.02 N, jaw-load 0.01 N, lift 6.0 mm, with all inherited tip, settle, contact-count,
quaternion, fixed-base, and classification gates unchanged.

The CUDA-derived post-event witness target remains exactly float32 degrees
`[22.147554397583008, 54.00971603393555, 84.68325805664062,
-26.58652114868164, 90.0, 66.40000915527344]`; its pinned six uint32 words and runtime
provenance remain unchanged.

## 3. Non-vacuous canonical command isolation

There is no `(64, 10) => true` shortcut. The canonical validator must actually recompute
the entire command-isolation contract from the 2,340-row trace and source AST:

- all ten active planned and applied targets exactly equal their active trace views;
- the five candidate-5 active rows exactly equal both frozen preflight10 and preflight12
  planned and applied command bytes; one ULP fails;
- the candidate-7 rows remain exact to their planned/applied current active trace and all
  inherited target/cadence/limit gates;
- the same-step active slice is unchanged by the witness override and active mutation
  count is exactly zero;
- AST inspection proves the only witness branch write is
  `target[witness_slot] = witness_safe_target`, followed only by its counter increment;
- the branch has no state or object write, teleport, reset, forward call, or extra step,
  and the task loop contains exactly one `env.step` call;
- the witness event predicate, one-way latch, event+1 activation, command bytes, snapshots,
  strict witness limits, all-environment limits, and planned/applied equality independently
  pass for every task step.

Cross-run realized state, velocity, force, and contact byte equality is removed from the
witness pass because those quantities are PhysX outcomes rather than evidence that the
excluded witness changed an active command. Cross-run actual differences may remain only
as a clearly marked diagnostic. All same-run numeric, limit, quaternion, contact/count,
filter, callback, and fixed-base gates remain authoritative.

## 4. Numeric and measurement-valid separation

The numeric semantic leaf requires the independently recomputed raw numeric report and
exact `metrics.numeric_integrity`; it does not consume the aggregate
`measurement_valid`. A current authoritative numeric mutation, including one ULP where
byte equality is specified, must reject the numeric leaf.

The stored aggregate `measurement_valid` is separately and exactly recomputed from
positive control, contact-buffer validity, witness command-isolation pass, and numeric
integrity. It remains required by runtime instrumentation, causal classification, and
scientific authority. Thus the dependency is acyclic without weakening a gate.

## 5. Historical cache scopes and p12 retirement

Current Supervisor V15 uses a resolved-path plus source-SHA module key and verifies
`__file__` and SHA before using p16 v12. Generic module preloads cannot redirect it.

Historical p10 and p12 retirement scopes remain separate. P10 forces frozen v9 into the
generic cache only while frozen Supervisor V12 runs. P12 forces frozen v11 into the
generic cache for the entire frozen Supervisor V14 semantic and terminal recomputation,
so nested historical behavior exactly reproduces p12's stored five-false map. Each scope
asserts path/SHA and restores the prior binding by object identity. Preload order must not
change either result.

Preflight12 retirement binds exactly 25 files, including terminal SHA-256
`6ffe9dddb7a333a27497a631b8343f224b29954662b7df7bca5771238e20008a`,
results SHA-256
`3be6849426fb46cecfae419f5b1886f7c807b0141b427b7a2b0a0d8f0d8df0dc`,
trace SHA-256
`60969f9d3359fc918b193811cb74d2e10ed0427f59f96125346c2ec7abf0fa9a`,
outcome SHA-256
`7340a600a08d875b48307987585962afafa7afc665a8630a77c15b394812c10f`,
raw physics 0, combined 125, render 0, diagnostic 2 plus task 2,340, clean GPU,
exact stored five false semantic leaves, valid nonpassing terminal attestation, no
promotion, and no scientific verdict. The full 25-file table in the p13 preflight prereg
is normative here as well.

## 6. Canonical verdict and output boundary

If separately authorized only after the admission barrier, Supervisor V15 permits exactly
one physics attempt and at most one render attempt with no retry. Raw zero is insufficient:
every current semantic leaf, render gate, terminal gate, Rerun requirement, and manual
inspection must pass.

Canonical classification remains the frozen causal partition and exact ten-row decision
ladder. The witness never alters the denominator. Any semantic, numeric, measurement,
command-isolation, lifecycle, render, Rerun, or inspection failure maps to no scientific
authority and no promotion; no fallback run or relaxed threshold is allowed.

Only p16 v12, Supervisor V15, `t3u_side_preflight13_prereg.md`, and this document are new
in this repair. All p1 through p12 source and evidence bytes are immutable. This document
authorizes no execution.
