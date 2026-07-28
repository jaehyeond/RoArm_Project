# START_HERE.md

Last updated: 2026-07-28 KST. D401 actual one-worker attempt is frozen.
Isaac started normally, but the harness misread the PhysX extension version
before any SDF asset was authored.

## Current Truth

- Pivot: RoArm cylinder grasp-track G0a. `q5=0` CLOSED; frozen OPEN
  `1.5413rad`.
- D362 (2026-07-17) remains the last physics run. A64 pushed over the historical
  34x90mm/0.72kg cylinder (final XY `60.619mm`, tilt `89.998deg`).
- Actual product nominal dimensions are `29x50mm`; mass/tolerance/COM/inertia/
  friction remain unmeasured. It has not been created or tested in physics.
- A64 (`64+64=128`) is a reference candidate, not an optimum or NVIDIA limit.
  P34/D397 did not produce a complete live/cooked replacement.
- D385/D397 `12 vertices/child` is a project budget, not a PhysX/NVIDIA limit
  and not an SDF setting.
- D400/D401 only attempts `gripper_link A64 -> SDF resolution 256`; link5 A64
  is frozen. No collision response, contact, tipping, or grasp evidence exists.
- `scientific_or_physics_verdict=null`; `g0a_pass=false`.

## D400 Attempt1 — Frozen Pre-Worker Failure

D400 stopped before Worker spawn because its expected Git baseline was stale
and it wrote its own phase file before taking Git status. It is not
Isaac/PhysX/SDF evidence. Preserve its four runtime outputs; no overwrite or
same-path retry.

## Active Case — D401 Actual Runtime Frozen Fail-Stop

Case: `D401 [d400_runtime_freeze_snapshot_order_repair]`

Path:
`claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/`

Approved tuple:
`7097134b350cf1641f2585c150cba45bc56ba0e9792d6549f0ae9c2f9e72cd2e`

Observable order:

1. Git snapshot was captured before the first output write.
2. runtime-freeze manifest PASS.
3. package/GPU/process gate PASS; free VRAM `15887MiB`, conflicts `0`.
4. runtime negative controls `18/18 PASS`.
5. one Worker spawned; retry `0`.
6. Isaac `SimulationApp` launched on `cuda:0` and RTX 4090.
7. runtime stack probe stopped before derivative asset authoring.
8. cleanup/evidence/supervision completed without timeout, signal, or residue.

## Exact Failure

Runtime stack checks:

- Isaac Sim `5.1.0.0`: PASS
- Isaac Lab `2.3.0`: PASS
- PhysX extension ID `omni.physx-107.3.26`: resolved
- active extension root: exact
- native plugin SHA: exact
- observed `package.version`: `null`
- expected `107.3.26`: FAIL

Installed Kit `107.3.3` returns
`carb.dictionary._dictionary.Item` from `get_extension_dict()`. The frozen
Worker accepts only built-in `dict`, so it discards the real value. Installed
`extension.toml` contains `version = "107.3.26"`.

Descriptive operational label:
`D401_D400_RUNTIME_STACK_VERSION_PROBE_HARNESS_TYPE_CONTRACT_FAIL_STOP`

Canonical JSON verdict remains:
`D400_GRIPPER_LINK_SDF_RES256_PREFLIGHT_FAIL_STOP`.

This is a harness type-contract failure, not an Isaac launch, PhysX install,
GPU, SDF geometry, collision, contact, or grasp failure.

## Scope Counters and Non-Results

- Worker/SimulationApp/retry: `1/1/0`
- derivative asset/SDF writes: `0/0`
- PhysX attach/detach/property query: `0/0/0`
- SimulationContext/reset: `0/0`
- controlled physics/public forward: `0/0`
- q5 command/sample: `0/0`
- contact/cylinder: `0/0`
- target/IK/path, source geometry, link5 representation changes: all `0`
- RRD/RBL/board/manual inspection: absent because technical PASS was never
  reached and no candidate geometry existed

Worker OS return code was `0`, but raw/pre-close protocol was false. Supervisor
correctly rejected it; OS return code alone is not success authority.

## Additional Latent Control Defect

Worker memory counter order was exact, but its JSON writer uses
`sort_keys=True`. Supervisor rereads alphabetically ordered keys and incorrectly
requires physical JSON key order to equal registered order, producing
`exact_36_keys_in_order=false`. This was not the first failure but would cause a
later false FAIL after the version probe is repaired.

## Next Authorization Boundary

Unapproved candidate:
`D402 [d401_runtime_stack_item_and_counter_order_authority_repair]`

Offline/static-only variables exactly `2`:

1. Item-compatible extension-version accessor
2. serialized counter exact-set/value/registered-projection authority

D402 may write only new forward-only scripts/static attestation/tuple. No
Controller runtime, Worker, Isaac/Kit/PhysX, USD, physics, q5, contact, cylinder,
Rerun, target/IK/path, or settings changes. Actual runtime requires another
approval citing the new tuple SHA.

## Must Read First

1. `AGENTS.md`
2. this file
3. `claudedocs/session_20260728_grasp_g0a_d401_actual_runtime_stack_probe_fail_stop.md`
4. D401 runtime manifest, phase markers, Kit log, raw, pre-close, supervisor,
   and completion JSON
5. `claudedocs/session_20260728_grasp_g0a_d401_d400_runtime_freeze_snapshot_order_repair.md`
6. `claudedocs/DECISIONS.md` (D400-P0/P1/P2, D401, D401-R1)
7. `claudedocs/EXPERIMENT_LEDGER.md` tail

## Authorization and Do-Not-Repeat

- Freeze D400 attempt1, D401 actual attempt1, and all earlier frozen paths.
- Do not retry or overwrite D401 actual attempt1.
- Do not call D401 an Isaac/PhysX/SDF/GPU/collision/grasp failure.
- Do not change science/geometry while repairing the two control defects.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- Preserve user-owned untracked `codex`; do not modify/delete/rename/stage it.
- `HANDOFF.md`/`TASKS.md` are stale; `/half-clone` is forbidden.
- No hardware, dependency install, signal, commit, or push is authorized.

## Git

- `HEAD == origin/master ==
  e9fa30088be7477ce5d6305aa5fdf68323e79adc` (`D400 gpu승인전`).
- Worktree is intentionally dirty with frozen D400/D401 evidence, state files,
  scripts, and user-owned `codex`.
- Commit/push is not authorized.
