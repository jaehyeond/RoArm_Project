# START_HERE.md

Last updated: 2026-07-23 KST. D377 ran the approved one-variable StageCache lifecycle
localization exactly once. The worker exited cleanly, but the preregistered workload comparator
produced a false negative from run-dependent diagnostic identifiers. The frozen formal verdict
therefore remains FAIL_STOP pending a separately approved offline authority repair.

## Current Truth

- Pivot: cylinder grasp-track G0a. Cylinder radius/diameter/height are
  `0.017/0.034/0.090m = 17/34/90mm`; `q5=0` CLOSED and frozen OPEN is `1.5413rad`.
- D362 remains the latest physical authority: the current A64 path pushed the cylinder over
  rather than holding it. D372-D377 did not run cylinder physics or supersede that result.
- D368 established `64 link5 + 64 gripper_link` as the current 64-cap reference candidate,
  not an optimum. `maxConvexHulls=32` is an automatic-decomposition schema default, not a
  manual-compound target count or engine hard limit.
- D372 built the professor's task-local P34 offline candidate, not a global optimum:
  - link5 `16`: body convex box `1`, connector/pivot support `3`, fixed-jaw contact pieces `10`,
    fixed-jaw backbones `2`.
  - gripper_link `18`: moving support `4`, moving-jaw contact pieces `12`, moving backbones `2`.
  - total `34`, versus A64 total `128`; speed, physics, tipping, and grasp equivalence are null.
- D373 materialized P34 but failed its identity contract because of comparator, unsupported
  articulation-owner instancing, traversal, and supervisor defects. D374 proved and visualized
  those defects offline.
- D375 removed whole-robot instancing and reached valid acquisition before shutdown: direct/live
  P34 `34/34`, callback protocol `34/34`, property-query collider rows `17+19` VALID, and authored
  mass/COM/inertia/axes deltas `0.0`. It then failed to exit: watchdog `900s`, SIGTERM, `20s`,
  SIGKILL, elapsed `920.3908159369603s`, return `-9`.
- D376 localized that non-exit to the terminal framework-release/process-exit boundary. The exact
  native blocker is null. Installed Isaac Sim `5.1.0.0` already used `fastShutdown=True`; both
  graceful and skip-cleanup paths reach `shutdown_and_release_framework()`. NVIDIA's later 6.0
  fix for bug 5948099 is mechanism evidence only, not exact D375 bug identity.

## D377 Verified Results

- New lifecycle variable: one `UsdUtils.StageCache.Get().Erase(stage)` immediately after the
  inherited successful PhysX detach. Python retained the `stage` reference; shutdown API,
  `fastShutdown`, asset, settings, and D375 acquisition workload were otherwise frozen.
- Preregistration checks `17/17` and negative controls `7/7` passed. Actual worker/retry `1/0`;
  bounded watchdog `120s`.
- The worker reproduced callback `34`, property queries `2`, and collider rows link5 `17` plus
  gripper_link `19`. PhysX attach/detach were `1/1`.
- Before Erase: StageCache `Contains(stage)=true` and the registered ID found the same stage.
  Erase was called exactly once and returned `true`. After Erase: `Contains=false`, the old ID
  was invalid/absent, while the Python stage reference remained retained.
- The process exited with return `0` in `6.733121555997059s`. Timeout, SIGTERM, SIGKILL,
  process-group residue, and worker GPU residue were all absent. Thus Isaac did not time out in
  D377.
- Physics step, q5 command/sample, contact query, cylinder write, public forward, reset,
  timeline play/commit, automatic decomposition sweep, and target/IK/path or physical-setting
  changes were all `0`.
- The preregistered comparator nevertheless reported workload mismatch and froze verdict
  `D377_STAGECACHE_ERASE_BEFORE_CLOSE_LOCALIZATION_FAIL_STOP`, branch
  `UPSTREAM_WORKLOAD_MISMATCH_ERASE_EFFECT_NULL`. Do not rewrite or promote this artifact.

## Post-result Authority Audit

- A read-only independent diff found exactly `68` selected-signature differences:
  - `34` callback witness hashes changed only because each witness included a runtime object
    memory address in `request_return_repr`.
  - `34` `prototype_path_diagnostic` fields changed only in run-assigned `__Prototype_N` numbers.
- Removing only those two non-authoritative run-dependent fields makes the D375 and D377
  termination workloads identical with independent canonical SHA-256
  `28aadb5ff26270039df58f7cd06080bf7afcdec001402e886a6edf1483fdfe31`.
- Callback payloads were otherwise exact `34/34`: total vertices `314`, indices `1016`, original
  polygons `262`. Property rows differed only in opaque runtime `path_id` values `38` and
  elapsed-time values `2`; mass, inertia, volume, AABB, local pose, result, and semantic path were
  exact.
- This proves a comparator false negative, not a preregistered D377 PASS. Consequently the clean
  exit strongly motivates StageCache retention as a conditional trigger in this run, but formal
  causal support remains null until a new forward-only offline authority repair is approved.
- Missing Erase is not a universal necessary cause: D373 also omitted Erase and exited `0`.
  D377 also does not prove stage-object destruction or exact NVIDIA bug 5948099 identity.

## Observability Status

- The exact decision board is `1920x1080`; save-only RRD/RBL passed automated Rerun `0.34.1`
  footer/entity/timeline/component checks. The actual HiDPI Viewer PNG is `3840x2160` physical.
- Manual completion failed because the Rerun capture's lower Korean text rendered as square
  missing glyphs. More importantly, the board truthfully reflects the frozen but now-known-false
  comparator result `workload=False`; it must not be used as the corrected professor-facing
  explanation.
- Overall D377 completion remains false. `g0a_pass=false`; P34 full live identity, physical
  equivalence/speed, tipping causality, current-pose closure, contact, and grasp feasibility are
  all null.

## Active Case / Authorization Boundary

- D377 is complete and frozen. There is no currently approved active case.
- Recommended next minimum is unapproved
  `D378 [d377_ephemeral_identifier_provenance_and_workload_authority_repair]`, offline-only.
  It would read immutable D375/D377 JSON/witnesses, preregister exactly which runtime diagnostic
  fields are excluded from termination-workload identity, independently recompute the corrected
  digest, and render an ASCII-only corrected explanation. It must not launch Isaac/PhysX or run
  q5, physics, contact, cylinder, target/IK/path, or collider regeneration.
- Only after that authority repair may a separately approved full P34 live-identity classifier
  be considered. A64/P34 cylinder physics, center-height/wrist repair, settle/hold/lift, ten-trial,
  G0b, RL/PPO/VLA each remain separately gated.

## Frozen Boundaries / Do Not Repeat

- Do not call P34's 34 parts a mathematical, global, or performance optimum.
- Do not claim D375 full live identity PASS from its preclose acquisition subresult.
- Do not claim D377 formal lifecycle PASS from its clean exit; preserve the frozen FAIL_STOP and
  separately label the post-result comparator false negative.
- Never include runtime memory addresses, generated prototype ordinals, opaque property `path_id`,
  or elapsed time in canonical geometry/workload identity unless their identity role is separately
  proven and preregistered.
- Callback protocol PASS alone is not full surface/property identity PASS. Property values are
  authority only when result is `VALID`.
- Do not repeat whole-robot instancing of dynamic articulation links, decimal-vs-typed-Float32
  over-tight comparison, default traversal without instance proxies, or raw-summary-only worker
  success without external process-exit authority.
- NVIDIA later-version release notes do not identify the installed-version failure by themselves.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, dependency install, signal, commit, push,
  physical comparison, or further live worker is authorized.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D375-D377; ledger tail
2. `claudedocs/session_20260723_grasp_g0a_d377_stagecache_erase_before_close_localization.md`
3. D377 preregistration, worker raw/preclose/supervisor, localization evidence, completion, and
   manual inspection under the single D377 attempt path
4. D375 attempt2 raw/preclose/supervisor/fail evidence for the frozen baseline
5. D376 provenance and NVIDIA-source attestation for lifecycle interpretation
6. D373 only for the no-Erase normal-exit counterexample; D372 for P34 design provenance
7. D362 physical trace only after a physical comparison is separately approved

## Git

- D377 boot and closeout verified `HEAD == origin/master ==
  e30f7f99d44252f509e383627738f3ad7967ea93`, subject `D375`.
- The worktree was clean after the user's D375 push and before approved D376 edits. It now contains
  only the approved forward-only D376 and D377 code, evidence, visualization, and state-doc work.
- D376/D377 PNGs are preserved at exact paths but may be hidden by the repository `*.png` ignore
  rule. D375/D376 paths were not overwritten; D334 sidecar remained unchanged.
- Commit/push was not authorized and was not performed.
