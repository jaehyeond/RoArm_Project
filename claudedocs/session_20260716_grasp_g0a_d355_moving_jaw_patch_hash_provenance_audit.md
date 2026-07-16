# Session 2026-07-16 - D355 moving-jaw patch hash provenance audit

## 1. What and why

D354 completed the zero-step q5 closure measurement, but its moving-jaw binding
gate remained false even though the immutable authored point/count/index streams
and runtime face order were exact.  Two failures must not be conflated:

1. the raw body-local Float64-m inner/outer XZ arrays were not bit-exact, and
2. the authored-only derived vertex/triangle/patch constants did not match the
   current authored canonicalization recipe.

This case audits only where those derived bytes came from.  It does not rerun
Isaac, q5 closure, physics, or the cap/rim classifier.

`이번 case의 신규 변수: [derived_moving_jaw_patch_hash_provenance_semantics]`

`new physical variables: []`

Forward-only output:

`claudedocs/runtime_logs/grasp_track/g0a_d355/`

Harness:

`sim_scripts/cyl34_top_view_d355_moving_jaw_patch_hash_provenance_audit.py`

## 2. Frozen authority before the run

- Current Git authority is
  `HEAD == origin/master == 64aa5b2c9552a053a3a9a34551fbfd168ce644ba`.
- D354 binding / measurement / completion / zero-step attestation hashes are:
  - `548d45ec4eb1dacbb4cbdefe2b64a3ed99ce72f4f5ffaaa6a9ee1e2b38756b15`
  - `fd0d43c2a47abefad939fc0e980456cc396bbf3ba3bb104b28bcd999100f23ed`
  - `5cc70c8aa1e50532fa4ec27756496d6b9f9447156c56ef700084b44c16226f86`
  - `1975df11b13a774b89f953991d7fdac6e38d795e81a9535850b147d823740d20`
- Frozen expected authored paired-XZ is
  `917b7154601d23984db01ebbd4adccdb272771920f225d1a021540b4b03bcaf9`.
- D354 raw-derived paired-XZ is
  `98ef77e6c5080e96f763eab04c48d4d6c06c9bc1a8b79995bd0fffa32618bbae`.
- D339 full gripper body-local Float64-m vertex stream is
  `522a4f0fe91a04bf54c5c8be6492748c7490fc557fa8c0867200d97332dfa9db`.
- D354 remains immutable and is read only.

## 3. Pre-registered procedure

### 3.1 Exact current recipes

The audit first reproduces, without Isaac:

1. authored points as little-endian Float32 millimetres, counts/indices as
   little-endian Int64, C-contiguous;
2. body-local raw points through the D334/D339 `mesh local -> world -> rigid-body
   local` Gf double transform, serialized as little-endian Float64 metres;
3. D351 authored patch derivation: sorted face IDs, triangle-point
   `np.unique(axis=0, return_inverse=True)`, original face row/winding order,
   `<f4` vertices, `<i8` triangles, `face + vertex + triangle` digest;
4. D351 raw paired-XZ derivation: source vertex-ID selection, coordinate unique,
   Float64 metres, `<f8` C-bytes.

The exact current outputs must match the immutable D354/D339 hashes before any
provenance explanation is accepted.

### 3.2 Independent implementation

A second implementation uses Python tuple sorting/dictionaries and
`struct.pack`, not NumPy byte hashing.  It must reproduce the authored current
recipe, raw full stream, and raw paired-XZ exactly.

### 3.3 Declared recipe grid

The recipe grid is frozen before result computation.  It crosses only:

- coordinate source: authored/raw, metre/millimetre, Float32/Float64, and the
  exact raw `x1000 -> Float32` roundtrip;
- ascending/descending face order;
- preserved/flipped winding;
- lexicographic/stable-first unique vertex order;
- preserved, cyclic-minimum, or unoriented-sorted triangle representation;
- face-order/lexicographic triangle rows;
- signed-zero preserved/normalized;
- `<f4>/<f8` vertex and `<i4>/<i8` triangle serialization;
- the six permutations of face/vertex/triangle blob concatenation.

No recipe axis may be added after the audit begins.  Each frozen expected hash
must be reproduced by at least one declared recipe; otherwise the result is
unresolved.

### 3.4 Failure-capable negative controls

Seven registered perturbations must behave as declared:

1. metre instead of millimetre;
2. Float64 instead of Float32;
3. big-endian instead of little-endian;
4. Fortran-order bytes without C canonicalization;
5. reversed face order;
6. flipped winding;
7. `face + triangle + vertex` instead of `face + vertex + triangle`.

This satisfies the failure-capable session rule: a missed declared hash change,
an unexpected topology change, or an independent-recalculation disagreement
forces FAIL_STOP.

## 4. Pre-registered verdict grammar

- `D355_DERIVED_PATCH_HASH_PROVENANCE_LOCALIZED`
  - all immutable/current streams reproduce;
  - both implementations agree;
  - every frozen expected derived hash is reproduced within the declared grid;
  - all negative controls behave as declared.
- `D355_DERIVED_PATCH_HASH_PROVENANCE_UNRESOLVED_FAIL_STOP`
  - any one of those requirements fails.
- `D355_OFFLINE_INPUT_OR_OBSERVABILITY_FAIL_STOP`
  - an input, forward-only, RRD/RBL, screenshot, or completion contract fails.

A localized result explains byte provenance only.  It does not turn D354 into a
geometry PASS, certify barrel-first contact, reject the current pose, prove grasp
feasibility, or justify target/IK/path repair.

## 5. Visualization preregistration

The audit will emit a save-only RRD, fixed RBL, footer/exact-contract validation,
and a `4800x2800` expected HiDPI screenshot.  Its panels are:

1. readable scope/verdict guide;
2. authored Float32-mm paired patch;
3. raw-derived Float64-m paired patch;
4. raw roundtrip overlay with residual arrows explicitly magnified `x10000` for
   display only;
5. frozen D354 cylinder/clear/overlap context, copied from the existing JSON,
   with the sub-micrometre residual magnified `x10000` for display only;
6. authoritative scalar audit and boundary event text.

Rerun Float32 spatial copies are never hashed back into a gate.  The D354 panel
does not run a new cap/rim discriminator.  The screenshot must be opened at
original resolution and separately recorded as manually inspected before
completion.

## 6. Frozen scope and prohibitions

- Isaac launch, PhysX query, q5 evaluation, controlled physics step: all `0`.
- No asset/decomposition/gate/tolerance/material/mass/actuator/renderer/solver/
  physics setting mutation.
- No target/IK/path change and no new cap/rim scientific classification.
- No settle, ten-trial, G0b, RL/PPO, VLA, or ladder promotion.
- No D354 rerun/overwrite, no D334 sidecar write, no commit/push.
- The separate D342 residual-process cleanup is operational evidence only and
  cannot change D355 scientific interpretation.

## 7. Execution log

### 7.1 Prepare and the single audit invocation

`--stage prepare` passed and froze the harness as
`b1fe5bf0f42c3d30a2b56d6809e17cfe4785eb7dcb610e2cf6fc05fb57c50d46`.
The preregistration itself is
`6cfaf0efa802546bed0177b88d8467a5d8ca1055ec113907b69bb5a81606325d`.

Exactly one `--stage audit` was then invoked.  Its exclusive invocation marker
records count `1`, PID `179644`, and `no_retry=true`.  Program order was:

1. write `audit_started`;
2. enter `_source_arrays()`;
3. execute `from pxr import Gf, Usd, UsdGeom`;
4. raise `ModuleNotFoundError: No module named 'pxr'`;
5. write `audit_exception` and stop.

The invocation, phase, and exception artifacts are respectively:

- `3bbc44edbec995fcbba0093fb7d9615290dbc32ffb65a8e12fc6b86bbb75a8f2`
- `1e5808892cbda91a7e7efb5e8a530468ae7d00b06abd961462f40c78b7da138c`
- `48bcad5c5740651f7aa8157616b64a639b79f67af5033e60dfe939da4bfdebde`

### 7.2 Exact executed and unexecuted work

The stop occurred before the USD or either frozen coordinate stream was loaded.
Therefore the authoritative execution counts are:

| Operation | Count |
|---|---:|
| audit invocation | 1 |
| second audit / retry | 0 |
| USD stream load | 0 |
| recipe candidate evaluation | 0 |
| negative-control execution | 0 |
| patch-hash computation | 0 |
| q5 evaluation | 0 |
| controlled physics step | 0 |
| Isaac launch | 0 |
| new cap/rim classification | 0 |

The correct operational verdict is
`D355_OFFLINE_INPUT_OR_OBSERVABILITY_FAIL_STOP`.  The scientific provenance
result is `null`: this is neither
`D355_DERIVED_PATCH_HASH_PROVENANCE_LOCALIZED` nor
`D355_DERIVED_PATCH_HASH_PROVENANCE_UNRESOLVED_FAIL_STOP`.

The postrun operational audit binds the invocation, exception, exact zero counts,
repaired failure observability, `scientific_provenance_result=null`, and
`g0a_pass=false`; SHA-256
`9f04faecc9dc983f14224174167f40a110655fb9c0b24d8e18e7ad7da2e56acd`.

## 8. Why PXR failed here while D350/D354 Isaac succeeded

The failed assumption was that the `isaaclab` Conda Python exposed a standalone
top-level `pxr` package without starting Kit.  It does not.  The PXR libraries
are bundled in Isaac's extension cache, and `isaacsim/__init__.py` bootstraps the
Kit kernel, environment, and Python paths.  The same installed file explicitly
requires `SimulationApp` instantiation before importing other Omniverse/Isaac
modules.

D350 and D354 followed that valid order: create `SimulationApp`, let Kit expose
extensions, then import/use PXR.  D355 intentionally registered `isaac_launch=0`
and attempted the opposite route, so it stopped before reading the USD.  Adding
extension-cache paths by hand, installing another OpenUSD package, or silently
launching `SimulationApp` would each introduce an unregistered input/runtime
variable and was therefore not used as an in-place repair.

This is not evidence of an RTX, CUDA, warp, SM-utilization, PhysX, Viewer, or
Isaac-rendering regression.  No GPU science kernel or Isaac application was
launched by the audit.  The recurring Conda `RequestsDependencyWarning` came
from the base Conda launcher and is not the `pxr` exception.  No package was
installed or changed; the preregistered Isaac-compatible pins remained
`numpy==1.26.0` and `psutil==5.9.8`.

## 9. Failure observability and visual inspection

The failed audit produced no scientific RRD because it stopped before data
loading.  The user separately approved a render of the failure itself.  All
observability attempts were forward-only and did not retry the audit:

1. attempt1 stopped before render because the script-directory launch could not
   import the repo-local `roarm_rl` package;
2. attempt2 conservatively stopped in prepare because its expected attempt1
   inventory was wrong;
3. attempt3 repaired both paths and passed automated RRD validation, but manual
   original-resolution inspection found its static-scalar Dataframe visibly
   showing `Unknown timeline`; it was not finalized;
4. attempt4 replaced only that panel with static Markdown and added a fixed-camera
   D354 view containing actual-scale points plus an explicitly labelled
   `DISPLAY EXAGGERATED` ordering aid.

Attempt4 completed RRD/RBL/footer/exact-contract validation and manual inspection.
The exact output is:

- RRD: `807939f8f2d23c3eb470652afb6035351fbedb678e76edfa7e4b1745e022c8f7`
  (`148850` bytes)
- RBL: `bf1fe2163aa322e81413900ef38d87a0740e18dc82b51cd7b1e6784813a0948e`
  (`79039` bytes)
- PNG: `adc31f4bb71359013ece0a194db9dde7f2722dbb616e8dcfe53b866138dc1340`
  (`4800x2800`, `1012112` bytes)
- validation:
  `e854ed27726e093c116957672c927b73d04bb580a08ec49435bc53d3e5ea7360`
- manual inspection:
  `a3c75a5579e26649038f73ddaf7a9ae54cc6fdb9206f0295db97ef818fb2cd3b`
- completion:
  `1b1be283537927f0d6cfb9ee52d521cd24b9db3207b8112d8d13f41e75f7e5b1`

The D354 panel copies only frozen historical values: clear `z=0.045m`, adjacent
overlap `z=0.044999619394590046m`, delta
`0.0003806054099525502mm`.  The physical-scale points are visually coincident;
only the labelled display aid separates them.  New classification count is `0`,
so the panel cannot change D354's cap/rim verdict.

The completed RRD was also opened once in a persistent Rerun `0.34.1` native GUI
on `DISPLAY=:1`.  The observed native viewer PID was `253513`; its GPU allocation
was `617MiB`.  Total GPU state after launch was `2692/16376MiB` used and
`13253MiB` free.  This is inspection-only and does not execute Isaac or science.

## 10. D342 residual-process cleanup audit

The separately approved lineage check confirmed D342 worker PID `1729639` under
wrapper PID `1729610`, process group/session `1729601`.  SIGTERM was sent first
to the worker and then to the process group.  The wrapper exited, but the worker
remained alive, was reparented to PPID `1123`, retained state `Sl`, RSS
`977284KiB`, and `320MiB` GPU allocation.  Therefore the exact cleanup verdict is
`D342_RESIDUAL_SIGTERM_CLEANUP_INCOMPLETE_STOP`, not cleanup PASS.  No SIGKILL or
additional signal was authorized or sent.  This operational result changes no
D342/D354/D355 science.

Audit SHA-256:
`82b5d63a5d5e57555e76eeeb6b0b1e304c2c4e15e6056bac276fbf8037890b9b`.

## 11. Final verdict and authorization boundary

- D354 remains `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`, controlled steps `0`,
  `g0a_pass=false`.
- D355 stopped before provenance science.  Authored/raw dtype, unit, ordering,
  canonicalization, and patch-digest provenance all remain unresolved/null.
- No evidence here certifies or rejects current-pose barrel-first closure, grasp
  feasibility, or target/IK/path repair.
- No source asset, decomposition, gate, tolerance, material, mass, actuator,
  renderer, solver, physics setting, target/IK/path, D334 sidecar, or immutable
  D354 evidence changed.

The narrowest next candidate requires a new explicit authorization: a new
forward-only case that preregisters one readonly Isaac/Kit bootstrap solely to
expose PXR, then runs the frozen D355 byte-provenance contract without q5,
physics, cap/rim science, or target/IK/path changes.  Installing a standalone
OpenUSD runtime is a distinct alternative with dependency/pin impact and is not
approved.  No provenance retry is active now.

## 12. Session-progress-rule accounting

This session did run the single registered failure-capable audit.  It failed at
an earlier input prerequisite than the seven planned perturbation controls, so
those controls could not execute (`0`).  The import failure changed the decision
from “provenance audit result” to mandatory input/observability FAIL_STOP; it was
not validation that could not change a decision.  No PPO promotion was in scope.
