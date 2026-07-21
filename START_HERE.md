# START_HERE.md

Last updated: 2026-07-21 KST. D370 proved the production direct-script repo-root import repair and
completed one Rerun Viewer capture, then froze at phase 9/14 because the raw color heuristic missed
purple by `23<25`; original-resolution inspection also found three informational notifications
obscuring the upper card. No further visual, physics, or collider case is authorized.

## Current Truth

- Pivot: cylinder grasp-track G0a. Cylinder radius/diameter/height are
  `0.017/0.034/0.090m = 17/34/90mm`; `q5=0` CLOSED, frozen OPEN `1.5413rad`.
- D348 callback-topology gate is `256/256` channels and `128/128` parts; live connected binding is
  `64+64`. D350 static Viewer passed, but alignment/G0a remained null/false; D354 cap/rim order is
  unresolved.
- D359 recovered the historical hash ordering. D361 raised the contact capacity to `33,280`.
  D362 remains physical authority: moving jaw `31/32`, cylinder motion `41/42`, fixed link5 `45/46`,
  endpoint XY/tilt/z delta `60.6189978mm/89.9977746deg/-28.0005205mm`—pushed over, not held.
- D363-D367 are observability/control only. D367 proved zero-step PLAY commit but overall cleanup
  completion remained FAIL.
- D368 measured the existing `64 link5 + 64 gripper_link` as a **current 64-cap reference
  candidate**, not an optimum. Measurement verdict is
  `D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_MEASURED_NO_PHYSICS`.
- D368 automated Rerun validation passed, but human inspection found an empty
  `Unknown timeline` panel and overlapping text. Therefore `visualization_pass=false` and the
  overall completion verdict is `D368_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`.
- D369 preregistration passed `14/14` with targeted controls `10/10`, but its one host worker stopped
  at phase `7/12` on `ModuleNotFoundError: No module named 'roarm_rl'`. Worker/viewer/retry counts are
  `1/0/0`; no PNG/manual/finalize exists. This was before Viewer and is not an Isaac/PhysX/GPU/Rerun
  renderer failure. Verdict: `D369_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`.
- D370 prepare/import and one-shot Viewer-capture subresults PASS, but overall visual completion
  FAIL. Prepare `18/18`, controls `9/9`; worker/Viewer/retry `1/1/0`; phase prefix `9/14`.
  Viewer returned `0` and wrote a `3840x2160` PNG. No `Unknown timeline`/empty metric panel or
  in-scene label text remained, but moving-full purple sampled `23<25` and three informational
  notifications covered the upper text card. No board/manual/finalize exists. Overall verdict:
  `D370_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`.

## Latest D370 — Import Repaired, Viewer Captured, Visual Completion FAIL

- Frozen case: `D370 [d369_project_root_import_preflight_visual_resume]`.
- Output path: `claudedocs/runtime_logs/grasp_track/g0a_d370/`; do not retry, overwrite, synthesize
  the missing board, or finalize.
- 이번 case의 신규 변수: `production_command_repo_root_import_preflight` only.
- Exact direct-script preflight removed ambient `PYTHONPATH`: enabled bootstrap return `0`; disabled
  control return `86` with exact `No module named 'roarm_rl'`; helper SHA `aaafcd93...107d`.
- Frozen D369 presentation/RBL were copied bit-exact (`0f394dec...02aab`,
  `429407b1...e216`); pre-render contract passed and Viewer return was `0`.
- Raw PNG SHA `7df0231a...4a32`; exception SHA `bbfe7602...05a9`. The fixed absolute `25`-pixel
  smoke threshold is not resolution-invariant authority and may not be lowered post hoc.
- Original inspection: four views and two cards exist; `log_time` is selected; three Rerun
  informational notifications obscure the top card. Therefore visual completion is false.
- All collider/Isaac/PhysX/q5/physics/contact/target-IK-path/settings/Warp-CUDA counters stayed zero;
  five science fields remain `null`, `g0a_pass=false`.

## Latest D368 — Semantic Allocation Measured, Visual Completion FAIL

- Output `claudedocs/runtime_logs/grasp_track/g0a_d368/` is frozen; no rerun/retry/overwrite.
- New variables were `semantic_contact_patch_authority` and
  `current_64cap_part_to_patch_allocation`. Prepare/audit/retry/controls were `7/7`, `1`, `0`, `8/8`;
  Isaac/physics/q5/contact/recook/asset writes were zero.
- Certified carriers were fixed `12 faces/4 parts`, moving inner `40/17`, moving outer `36/16`;
  moving classification was `16` dual, `1` inner-only, `47` no-certified-face.
- All `128/128` callback parts met observed offline `vertices<=64`, `polygons<=64`,
  `vertices_per_polygon<=32`; this is not proof of actual GPU contact execution.
- Installed Omni Physics schema `107.3.26+107.3.3` defaults are
  `maxConvexHulls=32`, `hullVertexLimit=64`; UI ranges are `1..2048`, `8..64`.
  Project `64/64` is a reference candidate, not NVIDIA's optimum. Five withheld fields remain
  `null`; `g0a_pass=false`.

## Current Authorization Boundary

- D370 approval was consumed by one host worker and one Viewer invocation. D370 is now frozen;
  there is no active approved case.
- A narrow next observability candidate would need separate approval and at most two preregistered
  variables: bounded notification-free screenshot stability and a resolution-normalized
  semantic-presence gate over immutable D370/D369 inputs.
- D368 offline authorization was consumed by one actual audit/no retry; its output remains frozen.
- Only after a readable professor-facing visual is complete may a collider Pareto case generate and
  compare separately preregistered semantic-split candidates using identical allocation and budget
  metrics. The next visual candidate and candidate generation/recook are both unauthorized now.
- Pose-only center-height/wrist comparisons and actual physics/contact/grasp tests are later,
  separate cases. Neither D368 nor D369 may answer those questions.
- The user's conditional statement about resuming D366 after a D367 overall PASS was not
  consumed: bridge subresult PASS와 overall completion FAIL을 먼저 보고하고 새 forward-only
  경계를 다시 승인받는다.
- Any q5/physics science, cap/rim discriminator, target/IK/path repair,
  asset/physics setting change, grasp/settle/hold/lift, ten-trial, G0b, RL/PPO/VLA needs a new
  explicit approval and forward-only preregistration.

## Frozen Boundaries / Operational Residue

- Freeze D351-D370 paths. Do not modify the user-owned
  `claudedocs/lab_meeting/20260715/d334_collision_table/` sidecar.
- Do not substitute Rerun Float32 display values or vertex-only Qhull for canonical
  callback/Float64/sensor evidence.
- Historical D342 worker PID `1729639` remained under user-systemd PID `1123`. One previously
  approved SIGTERM was sent after lineage recheck, but the process remained `Sl`; no SIGKILL or
  unapproved signal was used. Its observed GPU allocation was 320MiB.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, B200/SSH, dependency install,
  commit, push, or unapproved signal was performed.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D368-D370; ledger tail
2. D370 session plus preregistration, phases, import attestation, invocation/receipt, raw PNG, and
   exception under `claudedocs/runtime_logs/grasp_track/g0a_d370/`
3. D369 session and its frozen eight-file output; D368 session, evidence, manual, and completion
4. D367 session for the zero-step PLAY bridge; D362 session and immutable physical trace
5. D361/D360/D359/D354/D350/D348 lineage only when the next approved case requires it

## Git

- Verified at D370 boot and again after the one-shot run:
  `HEAD == origin/master == 888b92b4dfdb41e56d94fdffe4c0cb4d6e303297`, subject `D369진행`;
  worktree was clean at boot. Current uncommitted changes are only this dashboard, D370 harness,
  D370 session/state rows, and `g0a_d370/` artifacts. D368-D369 outputs stayed immutable. No
  commit/push was performed or authorized.
