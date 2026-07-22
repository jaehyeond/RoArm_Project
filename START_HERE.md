# START_HERE.md

Last updated: 2026-07-22 KST. D375 ran one repaired P34 live-identity worker. Its
live acquisition/property subresult passed, but Isaac did not return from shutdown before the
bounded watchdog, so the overall case is frozen as FAIL_STOP. No physics was run.

## Current Truth

- Pivot: cylinder grasp-track G0a. Cylinder radius/diameter/height are
  `0.017/0.034/0.090m = 17/34/90mm`; `q5=0` CLOSED and frozen OPEN is `1.5413rad`.
- D362 remains the physical authority: the current A64 path pushed the cylinder over rather
  than holding it. D372-D375 did not rerun or supersede that physical result.
- D368 established `64 link5 + 64 gripper_link` as a current 64-cap reference candidate,
  not an optimum. `maxConvexHulls=32` is an automatic-decomposition schema default, not a
  manual compound target count or engine hard limit.
- D372 built the professor's task-local P34 offline candidate:
  - link5 `16`: body box-shaped convex Mesh `1`, connector/pivot support `3`, fixed-jaw
    contact pieces `10`, fixed-jaw backbones `2`.
  - gripper_link `18`: moving support `4`, moving-jaw contact pieces `12`, moving-jaw
    backbones `2`.
  - total `34`, a design-count reduction from A64 total `128`, not a speed/physics/optimality
    result.
- D373 materialized P34 and acquired limited raw data but failed its identity contract because
  of Float32 comparator, unsupported articulation owner instancing, traversal, and supervisor
  defects. D374 proved those defects and visualized the frozen D373 failure offline.
- D375 removed whole-robot instancing while reusing the exact D373 P34 asset. Before shutdown:
  - `/World/Robot`, `link5`, and `gripper_link` were non-instance/non-proxy;
  - direct/live P34 `34/34`, callback protocol `34/34` passed;
  - property queries were VALID for link5 `17/17` and gripper_link `19/19` collider rows;
  - authored mass/COM/inertia/axes deltas were `0.0`.
- D375 still ended `D375_P34_LIVE_ASSET_IDENTITY_CONTRACT_REPAIR_FAIL_STOP`: after writing
  a hash-exact raw/preclose PASS, the process did not exit in the subsequent
  `SimulationApp.close()`/interpreter-teardown window. No post-close marker exists, so the
  internal blocking call remains unlocalized.
  The supervisor timed out at `900s`, sent SIGTERM then SIGKILL, elapsed
  `920.3908159369603s`, return `-9`, effective PASS `false`.
- Therefore full authored↔callback surface/bounds/original-polygon topology-volume identity,
  physical equivalence/speed, tipping causality, and grasp feasibility remain `null`;
  `g0a_pass=false`.

## D375 Verified Results

- Attempt1 was prepare-only: sandboxed child `nvidia-smi` returned `9`; actual worker `0`.
  Direct host query proved RTX 4090 Laptop, driver `580.159.03`, VRAM
  `16376/480/15465MiB` total/used/free, compute capability `8.9`.
- Forward-only attempt2 external GPU attestation: prereg checks `21/21`, negative controls
  `4/4` PASS; actual worker/retry `1/0`.
- Installed stack: Isaac Sim `5.1.0.0`, Isaac Lab `2.3.0`, Omni PhysX/schema `107.3.26`;
  `numpy==1.26.0`, `psutil==5.9.8`.
- Frozen P34 asset was not rewritten or rematerialized. Active A64 count `0`; disabled known
  legacy count `2`; timeline stayed STOP at `0.0s`.
- `link5`: rigid result VALID, collider `17/17 VALID`, mass
  `0.015392799861729145kg`.
- `gripper_link`: rigid result VALID, collider `19/19 VALID`, mass
  `0.0028707999736070633kg`.
- Worker raw/preclose protocol `true/true`; summary SHA/counter/timeline binding exact.
  Supervisor hash authority passed, but operational/effective authority failed due timeout and
  signals.
- Simulation launch/PhysX attach-detach/callback/property query = `1/1-1/34/2`.
  Physics step, q5, contact, cylinder, public forward, reset, target/IK/path, decomposition
  sweep, material/mass/actuator/physics change, asset/USD write = all `0`.
- Fail-closed analysis did not run the full geometry classifier or create 1920×1080/RRD/RBL;
  these are `not_run_due_worker_authority_fail`, not visual PASS.
- Both D375 attempts and the user-owned D334 sidecar are frozen; do not retry or overwrite.

## Active Case / Authorization Boundary

- No case is currently approved. D375 is complete as FAIL_STOP and frozen.
- Recommended next minimum, not yet approved:
  `D376 [d375_terminal_close_provenance_and_failure_visualization]`.
  - Read immutable D375 JSON/logs only; Isaac/PhysX rerun `0`.
  - Localize the program-order boundary around preclose, `SimulationApp.close()`, watchdog,
    SIGTERM, and SIGKILL without guessing the internal hang cause.
  - Compare installed Isaac Sim 5.1 source with version-matched NVIDIA lifecycle docs.
  - Produce an exact 1920×1080 failure board plus save-only RRD/RBL and manual inspection.
- A later live lifecycle repair must choose one preregistered variable (for example the official
  immediate-exit path versus a supervisor-owned terminal contract) and obtain separate approval.
- A repaired full live-identity PASS is still required before any A64, link5-only P34,
  gripper-only P34, or both-P34 cylinder physics comparison.
- Physical comparison, center-height/wrist pose repair, target/IK/path, material/mass/actuator/
  physics changes, settle/hold/lift, ten-trial, G0b, RL/PPO/VLA each require separate approval.

## Frozen Boundaries / Do Not Repeat

- Do not call P34's 34 parts a mathematical, global, or performance optimum.
- Do not claim D375 full live identity PASS from the preclose acquisition subresult.
- Do not treat a timeout/SIGKILL worker as successful because its raw JSON says PASS.
- Do not repeat whole-robot instancing of dynamic articulation links.
- Do not repeat decimal `0.0001` versus typed Float32 with `1e-12m` after D343.
- Default traversal zero is not collider absence when instance proxies are omitted.
- Property-query values are authority only when result is `VALID`.
- Callback protocol PASS alone is not full surface/property identity PASS.
- Do not bypass fail-closed classification to manufacture presentation artifacts.
- NVIDIA release-note hang fixes do not by themselves identify this workload's hang cause.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, B200/SSH, dependency install, commit,
  push, new signal, new live worker, or physical comparison is authorized.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D373-D375; ledger tail
2. `claudedocs/session_20260722_grasp_g0a_d375_p34_live_asset_identity_contract_repair_fail_stop.md`
3. D375 attempt2 preregistration, raw summary, preclose sentinel, supervisor, fail attestation
4. D374 session/repair contract only when tracing inherited authority
5. D373 raw only when comparing instance-proxy failure to D375 VALID property rows
6. D372 geometry/completion only when tracing the frozen P34 design source
7. Version-matched NVIDIA Omni Physics 107.3 and Isaac Sim 5.1 lifecycle docs
8. D362 physical trace only after a physical comparison is separately approved

## Git

- Session boot verified `HEAD == origin/master ==
  3d71aac219ba16f3262dc94b1898a459eaa534e7`, subject
  `D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP과 g0a_pass=false`, with a clean worktree before
  D375 edits.
- Current worktree contains only the uncommitted D375 implementation, attempt1/attempt2
  evidence, and D375 state-doc updates described here.
- No D375 PNG/RRD/RBL was created because the worker-authority gate failed before
  classification/visualization.
- Commit/push was not authorized and was not performed.
