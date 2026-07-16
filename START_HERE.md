# START_HERE.md

Last updated: 2026-07-16 KST. D354 remains the latest completed q5 science case.
D355 is the latest executed operational/offline case and stopped before provenance
science because plain `isaaclab` Python could not import `pxr`. No research case
is currently approved.

## Current Truth

- The pivot remains cylinder grasp-track G0a (`radius=0.017m`, `height=0.090m`).
  G0b close/lift, settle, ten-trial, PPO/RL, VLA, ladder, real hardware, and B200
  are out of scope.
- q5 is fixed: URDF `q5=0` is CLOSED; frozen sim OPEN is `q5=1.5413rad`.
  D347's measured reset was HOME-near q5=0 CLOSED, not exact HOME.
- D348 proved PhysX volume must use callback polygon topology, not a new
  vertex-only Qhull. Corrected gate: `256/256` channels, `128/128` parts.
- D349 frozen-OPEN raw/live distances were link5
  `4.2726455336/4.2727365803mm` and moving gripper
  `11.1750883746/11.3402623263mm`; these were not contact/grasp proof.
- D350 measured the actual connected fixed-jaw surface and completed real Isaac
  Viewer plus 64+64 collider visualization, but `aligned_pass=null`,
  `g0a_pass=false`.
- D351 never reached q5 science. D352 localized a pending Timeline PAUSE; D353
  proved one conditional main-thread `Timeline.commit()` applies it with zero
  registered world advance.

## Latest Completed Scientific + Observability Case: D354

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d354/`.
- Verdict: `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`; `completion_pass=true`,
  `scientific_contract_pass=false`, controlled physics steps `0`,
  `g0a_pass=false`.
- Raw/live used the same Float32 q5 clear/overlap bracket
  `1.0269782543182373/1.0269775390625rad`, width
  `7.152557373046875e-7rad`. Raw signed distances were
  `+0.0010050812803802547/-0.000988475720559677mm`; live were
  `+0.0010049780471806762/-0.0009864198978583663mm`.
- Both clear endpoints were exactly cylinder-local `z=+0.045m` and classified
  `cap_or_rim_boundary`; the immediately adjacent overlap endpoints alone were
  `barrel_interior`. Full-bracket barrel-feature consensus and cap-competitor
  exclusion therefore failed.
- The moving contact patch identity was unambiguous, but full moving-surface
  binding failed derived-hash/runtime-roundtrip exactness. Immutable authored
  streams and face order were exact; authored paired-XZ SHA
  `917b7154...bcaf9` differed from raw-derived `98ef77e6...18bbae`.
- This neither certifies barrel-first/current-pose grasp nor proves grasp
  impossible or target/IK repair necessary. Do not add a post-hoc cap/rim
  tolerance.
- Completion / measurement / moving-binding / zero-step attestation SHA-256:
  - `5cc70c8aa1e50532fa4ec27756496d6b9f9447156c56ef700084b44c16226f86`
  - `fd0d43c2a47abefad939fc0e980456cc396bbf3ba3bb104b28bcd999100f23ed`
  - `548d45ec4eb1dacbb4cbdefe2b64a3ed99ce72f4f5ffaaa6a9ee1e2b38756b15`
  - `1975df11b13a774b89f953991d7fdac6e38d795e81a9535850b147d823740d20`

## Latest Executed Case: D355 offline patch-hash provenance audit

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d355/`.
- Registered variable:
  `[derived_moving_jaw_patch_hash_provenance_semantics]`; physical variables `[]`.
- Prepare passed. Exactly one audit invocation was recorded (`count=1`,
  `no_retry=true`), then `_source_arrays()` stopped at
  `from pxr import Gf, Usd, UsdGeom` with
  `ModuleNotFoundError: No module named 'pxr'`.
- Phase markers contain only `audit_started` and `audit_exception`. USD stream
  loads, recipe candidates, negative controls, patch hashes, q5 evaluations,
  physics steps, Isaac launches, and new cap/rim classifications were all `0`.
- Operational verdict: `D355_OFFLINE_INPUT_OR_OBSERVABILITY_FAIL_STOP`.
  Scientific provenance result is `null`, not localized and not a hash-science
  FAIL. D354 science is unchanged.
- Cause: D350/D354 instantiated `SimulationApp` before Omniverse/PXR imports.
  Plain Conda Python does not expose Isaac's extension-cache PXR. This is an
  input/bootstrap assumption failure, not an RTX/warp/SM/PhysX/Viewer regression.
- Input/preregistration, invocation, phase, exception SHA-256:
  - `6cfaf0efa802546bed0177b88d8467a5d8ca1055ec113907b69bb5a81606325d`
  - `3bbc44edbec995fcbba0093fb7d9615290dbc32ffb65a8e12fc6b86bbb75a8f2`
  - `1e5808892cbda91a7e7efb5e8a530468ae7d00b06abd961462f40c78b7da138c`
  - `48bcad5c5740651f7aa8157616b64a639b79f67af5033e60dfe939da4bfdebde`
- Postrun operational audit SHA-256:
  `9f04faecc9dc983f14224174167f40a110655fb9c0b24d8e18e7ad7da2e56acd`.

### Failure visualization and live inspection

- The scientific RRD could not exist because no data loaded. A separate
  failure-only Rerun path was completed without retrying the audit.
- attempt3 automatic validation passed, but manual inspection caught an
  `Unknown timeline` Dataframe. It was not finalized. attempt4 changed only the
  blueprint to static Markdown counts and a fixed-camera D354 display.
- attempt4 RRD/RBL/footer/exact-contract and original-resolution manual inspection
  passed. The `4800x2800` PNG visibly shows one invocation, all later counts `0`,
  the red PXR stop, and frozen D354 actual-scale plus explicitly
  `DISPLAY EXAGGERATED` endpoint order.
- RRD / RBL / PNG / completion SHA-256:
  - `807939f8f2d23c3eb470652afb6035351fbedb678e76edfa7e4b1745e022c8f7`
  - `bf1fe2163aa322e81413900ef38d87a0740e18dc82b51cd7b1e6784813a0948e`
  - `adc31f4bb71359013ece0a194db9dde7f2722dbb616e8dcfe53b866138dc1340`
  - `1b1be283537927f0d6cfb9ee52d521cd24b9db3207b8112d8d13f41e75f7e5b1`
- One persistent Rerun `0.34.1` GUI is open on `DISPLAY=:1` using that RRD.
  Viewer GPU allocation was `617MiB`; total GPU state after launch was
  `2692/16376MiB` used and `13253MiB` free. It is inspection-only.

## Operational residue: D342 cleanup is incomplete

- Before signal: wrapper/worker PID `1729610/1729639`, PGID/SID `1729601`.
- User-approved SIGTERM was sent to the worker and process group. The wrapper
  exited, but worker `1729639` remained, reparented to PPID `1123`, RSS
  `977284KiB`, GPU `320MiB`.
- Verdict: `D342_RESIDUAL_SIGTERM_CLEANUP_INCOMPLETE_STOP`. No SIGKILL or extra
  signal is approved. This is operational only and changes no science.

## No Active Approved Case / Next Authorization Boundary

- D355 must not be retried or overwritten.
- Narrowest candidate: a new forward-only readonly Kit-bootstrap provenance case
  that explicitly registers one `SimulationApp` bootstrap solely to expose PXR,
  then applies the frozen D355 byte recipe. It must still forbid q5, physics,
  cap/rim science, asset writes, and target/IK/path changes.
- A standalone OpenUSD install is a distinct dependency-changing alternative and
  is not approved. Any retry requires a new explicit user approval and output
  folder.

## Frozen Boundaries

- Do not change assets, decomposition, target/IK/path, q0-q5/object initial state,
  gates/tolerances, material, mass, actuator, renderer, solver, or physics settings.
- Do not run settle, ten-trial, G0b, RL/PPO, VLA, or ladder promotion.
- Do not substitute vertex-only Qhull or Rerun Float32 display data for canonical
  callback/Float64 evidence.
- `HANDOFF.md` and `TASKS.md` are stale. D338-D355 evidence is immutable.
- Do not modify user-owned
  `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- Hardware control, B200/SSH, `/half-clone`, and unapproved commit/push are forbidden.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D348-D355; ledger tail
2. `claudedocs/session_20260716_grasp_g0a_d355_moving_jaw_patch_hash_provenance_audit.md`
3. D355 postrun audit, invocation, phase, exception, attempt4 completion/manual/summary, D342
   cleanup audit, and persistent viewer audit in the D355 output folder
4. `claudedocs/session_20260716_grasp_g0a_d354_current_pose_q5_closure_science_resume.md`
5. D354 completion, measurement, moving-binding, attestation, and supervisor JSON
6. D353 and D352 sessions; D351 original/repair sessions and termination audit
7. D348-D350 sessions referenced by DECISIONS D348-D350

## Git

- Current `HEAD == origin/master ==
  64aa5b2c9552a053a3a9a34551fbfd168ce644ba`.
- D354 execution base `b7beb91997859a5ddb2b0407388e80aed45898dc` is
  historical, not current HEAD.
- Worktree was clean before D355 and is intentionally dirty only for this approved
  forward-only case. No commit or push was performed.
