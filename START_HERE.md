# START_HERE.md

Last updated: 2026-07-21 KST. The user approved the professor-requested semantic compound
collider design. D372 completed the offline P34 candidate and its observability contract.
It did not author a live asset or run Isaac/PhysX, q5, physics, contact, or grasp.

## Current Truth

- Pivot: cylinder grasp-track G0a. Cylinder radius/diameter/height are
  `0.017/0.034/0.090m = 17/34/90mm`; `q5=0` CLOSED and frozen OPEN is `1.5413rad`.
- D362 remains the physical authority: the current A64 collider path pushed the cylinder over
  rather than holding it. D372 did not supersede that physical result.
- D368 established that `64 link5 + 64 gripper_link` is a current 64-cap reference candidate,
  not an optimum. D371 compared A/R64/R32/C1/C2 offline; only R32 was an eligible reduced-count
  automatic-decomposition candidate.
- D372 changed direction under explicit user approval and built the professor's semantic split:
  simple link5 body, separate fixed jaw, and separate moving jaw.
- D372 P34 exact counts:
  - link5 `16`: body box `1`, connector/pivot support `3`, fixed contact pieces `10`, fixed-jaw
    backbones `2`.
  - gripper_link `18`: proximal moving support `4`, moving contact pieces `12`, moving-jaw
    backbones `2`.
  - total `34`, which is 94 fewer parts (`73.4375%`) than A64 total `128`; this is a design
    count reduction, not a speed or optimality result.
- Installed schema `maxConvexHulls=32` is an automatic decomposition default, not a target count
  for manually authored child colliders. Installed UI `1..2048` is an authoring range, not an
  engine hard maximum or optimum.
- D372 offline raw-outside-candidate P95/max distances:
  - link5 `0.1795673203/1.0937375550mm`
  - gripper_link `0.1872935098/1.1572354710mm`
- Contact-layer plus adjacent-backbone 2D void-fill diagnostics:
  - fixed jaw `0/1.4176018901%`
  - moving open mouth/internal window `8.6018957346/27.1668822768%`
  These are not full-candidate, through-depth, or 3D-physics proofs.
- Frozen-OPEN P34 clearances are link5 `4.2726834003mm` and gripper_link
  `10.9714602318mm`; both are clear in the immutable D349 pose.
- On 61 immutable D362 poses, first offline overlap stayed at link5 A64/P34 `246/246` and
  moving A64/P34 `232/232`; transition-window max distance delta was
  `0.0270367065/0.0025090084mm`. This is a saved-pose counterfactual replay, not causal dynamics.
- Failure-capable controls passed `5/5`; prepare `8/8`; actual run/retry `1/0`; finalize
  `15/15`.
- Completion verdict:
  `D372_PROFESSOR_SEMANTIC_COMPOUND_CANDIDATE_OFFLINE_PASS_NO_PHYSICS`.
- `live_asset_identity`, actual GPU contact execution, physics equivalence, D362 causal replay
  equivalence, runtime speed, tipping causality, grasp feasibility, and global optimum remain
  `null`; `g0a_pass=false`.

## D372 Execution and Visualization

- Attempt1 is a frozen prepare-only path bug: an installed NVIDIA `database.py` was incorrectly
  converted to a repo-relative path. It produced no preregistration, run, or measurement and did
  not invoke Isaac/PhysX/q5/physics.
- Forward-only `attempt2_external_schema_path_repair` completed one run with no retry.
- Scope counters: SimulationApp/Kit, Isaac/PhysX, cook/automatic decomposition, q5, controlled
  physics step, live contact, USD/live asset write, target/IK/path and physical-setting changes
  are all `0`; immutable rows read `61`, offline hppfcl queries `10045`.
- Three exact `1920x1080` professor boards, RRD/RBL footer/entity/component/timeline validation,
  and original-resolution manual inspection passed.
- The Rerun logical window was `1920x1080` and its HiDPI physical PNG `3840x2160`.
  Three informational toasts and auxiliary event-row clipping were recorded; they do not obscure
  the decision geometry, cylinder, distance graph, or exact `d362_phase_step` timeline.
- The desktop termination seen earlier at 18:38 was a system RAM OOM with swap `0`, not a D372
  GPU/PhysX crash. Kernel evidence shows multiple GUI/AI/Python processes under cumulative memory
  pressure. D372 attempt1 occurred later at 19:11 and attempt2 ran later still.

## Current Authorization Boundary

- D351-D372 evidence paths are frozen. Do not retry, overwrite, recook, or finalize them again.
- There is no approved next experiment.
- Narrow recommended next case, requiring explicit approval:
  `D373 [p34_live_asset_identity_preflight]`.
  1. Materialize P34 once in a new forward-only candidate asset.
  2. Keep physics/q5/contact at zero.
  3. Verify live callback/readback owner, part count `16+18`, vertices, polygons, and digests
     against the D372 offline geometry.
- Only after that identity PASS, seek separate approval for physical comparison. To isolate cause,
  compare A64 against link5-only P34, gripper-only P34, and both-P34 in separate forward-only
  cases under the same pose/trajectory and physics settings.
- Target/IK/path, center-height/wrist pose repair, materials/mass/actuator/physics settings,
  settle/hold/lift, ten-trial, G0b, RL/PPO/VLA remain separate approval boundaries.

## Frozen Boundaries / Do Not Repeat

- Do not interpret P34's 34 parts as a mathematical or global optimum.
- Do not interpret schema default `32` or UI maximum `2048` as the manual target or engine limit.
- Do not replace contact-layer topology/Float64 authority with Rerun Float32 display geometry.
- Do not use the 2D jaw-void diagnostic as whole-candidate or 3D through-depth proof.
- Do not use immutable D362 replay as physical equivalence or tipping-causality proof.
- Do not modify the user-owned `claudedocs/lab_meeting/20260715/d334_collision_table/` sidecar.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, B200/SSH, dependency install, commit,
  push, or unapproved signal is authorized.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D368-D372; ledger tail
2. `claudedocs/session_20260721_grasp_g0a_d372_professor_semantic_compound_collider_design_offline.md`
3. D372 attempt2 preregistration, geometry, evidence, report, manual inspection, and completion
4. D372 attempt1 exception only to understand the preserved prepare-path failure
5. D371 comparison and D368 allocation evidence only when the next approved case needs lineage
6. D349 frozen OPEN, D350/D354 jaw authority, and D362 physical trace only when required

## Git

- Verified during D372 finalize:
  `HEAD == origin/master == 4a1120b801e808071583136e78954c78ca941dc8`, subject `370 test`.
- D371 and D372 implementation, evidence, and state-document changes are uncommitted.
- Commit/push was not authorized and was not performed.
