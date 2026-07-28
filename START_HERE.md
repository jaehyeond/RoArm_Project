# START_HERE.md

Last updated: 2026-07-28 KST. D402 actual attempt1 stopped at host GPU preflight.
The approved runtime path is frozen; no same-path retry is allowed.

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
- D400-D402 only prepare `gripper_link A64 -> SDF resolution 256`; link5 A64
  is frozen. No collision response, contact, tipping, or grasp evidence exists.
- D402 actual attempt1 reached only the package/GPU/process gate. `nvidia-smi`
  could not communicate with the NVIDIA driver, so Worker/Isaac/PhysX were not
  spawned. This is infrastructure fail-stop evidence, not science evidence.
- Read-only follow-up found NVIDIA 580.173.02 modules, PCI RTX 4090 identity,
  and `/proc/driver/nvidia`, but `/dev/nvidia0`, `/dev/nvidiactl`, and
  `/dev/nvidia-uvm` are absent. No recovery action was performed.
- A separate minimal direct Isaac Sim launch confirmed the same failure inside
  Kit: NVML driver-not-loaded, no CUDA-capable device, and GPU Foundation not
  initialized. The app shell reached startup then closed; no science ran.
- `scientific_or_physics_verdict=null`; `g0a_pass=false`.

## Frozen Failures — Do Not Retry or Overwrite

- D400 attempt1 stopped before Worker spawn because its Git baseline was stale
  and it wrote its own phase file before Git capture.
- D401 attempt1 repaired that order, launched Isaac normally, then stopped
  before asset authoring because the frozen harness accepted only built-in
  `dict` while Kit returned `carb.dictionary.Item`.
- D401 also exposed a latent supervisor false-fail: Worker JSON is deliberately
  `sort_keys=True`, but the supervisor treated serialized key order as schema
  meaning.
- Neither failure is Isaac/PhysX/SDF/GPU/collision/grasp evidence.

## Active Case — D402 Actual Attempt1 GPU Preflight Fail-Stop

Case:
`D402 [d401_runtime_stack_item_and_counter_order_authority_repair]`

Path:
`claudedocs/runtime_logs/grasp_track/g0a_d402/attempt1_d401_runtime_stack_item_and_counter_order_authority_repair/`

Exactly two repaired harness variables were registered and statically attested:

1. Item-compatible active `omni.physx` `package.version` accessor
2. serialized counter exact-set/value/registered-projection authority

Worker repair reads the same active extension config through
`config["package"]["version"]` after the frozen probe; the resolved path records
two reads total and accepts only exact built-in string `107.3.26`. Controller
repair keeps `sort_keys=True`, treats physical key order as diagnostic, and
requires the exact 36-key set, strict integers, unchanged values/ranges, and a
frozen-order projection digest. D401 snapshot-before-first-write is reused.

## Static Results

- Positive repair controls: `5/5 PASS`
- Approval-compatible negative fixtures: `32/32 PASS`
- Combined Item/counter/AST/provenance controls: `43/43 PASS`
- Final tuple/static schema cross-check: `49/49 PASS`
- Installed NVIDIA primary-source hashes bound: `5/5`
- Independent adversarial reviews: `3`; remaining blockers: `0`
- Controller runtime/Worker/Isaac/Kit/PhysX/GPU/USD/Rerun:
  `0/0/0/0/0/0/0/0`
- physics/public forward/q5/contact/cylinder: all `0`

Verdict:
static: `D402_RUNTIME_STACK_ITEM_AND_COUNTER_ORDER_AUTHORITY_REPAIR_STATIC_ATTESTATION_PASS_RUNTIME_NOT_APPROVED`

actual attempt1: `D402_RUNTIME_STACK_ITEM_AND_COUNTER_ORDER_AUTHORITY_REPAIR_RUNTIME_GPU_PREFLIGHT_FAIL_STOP`

Runtime evidence: `claudedocs/session_20260728_grasp_g0a_d402_actual_runtime_gpu_preflight_fail_stop.md`.

## Proposed Runtime Tuple

Four bound files:

- prereg:
  `9868b1f60035682295610ce9e38e23d8fa1df37804a69386b00aaf3cf1fdfc4e`
- attestation:
  `c112a18d51e238ec3bd8520f5dea52452a11e060840e8038e789a6a22279561d`
- Controller:
  `af1940a57b05ad9f8afdf8359fc099437360a7ff43eb97259e1ada9eb158da52`
- Worker:
  `214d6dcf8e330aa3a6da8a01a614275092462fa337bb1c1fea649c3ec0d654c3`

Tuple-file SHA-256:
`898c91551e9b724e0d8d7114128ccfb14563f16c4e6b22aa796d07e805c012ce`

## Next Authorization Boundary

The exact tuple was approved and consumed by one Controller attempt. That attempt
stopped before Worker spawn because the NVIDIA driver was unavailable to
`nvidia-smi`; the same path must not be retried.

Even if approved, the runtime remains one Controller / one Worker / no retry
and is limited to frozen D400 asset load/cook/readback preflight. Physics,
public forward, q5, contact, cylinder, target/IK/path, and settings remain 0.

This tuple is valid only while:

`HEAD == origin/master ==
9dd14ebb32421d93e3b46e2912cd3e67e6daff20`

Any future runtime requires host GPU recovery, a new forward-only Git snapshot,
and a new tuple/approval. No physics or q5 approval is implied.

## Must Read First

1. `AGENTS.md`
2. this file
3. `claudedocs/session_20260728_grasp_g0a_d402_actual_runtime_gpu_preflight_fail_stop.md`
4. `claudedocs/session_20260728_grasp_g0a_d402_host_gpu_driver_readonly_diagnostic.md`
5. `claudedocs/session_20260728_grasp_g0a_d402_direct_isaac_gpu_initialization_fail_stop.md`
6. `claudedocs/session_20260728_grasp_g0a_d402_runtime_stack_item_and_counter_order_authority_repair.md`
7. D402 preregistration, attestation, and proposed tuple
8. `claudedocs/session_20260728_grasp_g0a_d401_actual_runtime_stack_probe_fail_stop.md`
9. D401 runtime manifest, phase, Kit log, raw, pre-close, supervisor, completion
10. `claudedocs/DECISIONS.md` D400-P0/P1/P2, D401, D401-R1
11. `claudedocs/EXPERIMENT_LEDGER.md` tail

## Authorization and Do-Not-Repeat

- Freeze all D400/D401 attempts and all earlier frozen paths.
- Freeze D402 actual attempt1; do not retry its output directory.
- Do not call D401 an Isaac/PhysX/SDF/GPU/collision/grasp failure.
- Do not change science/geometry while validating the two D402 harness repairs.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `codex` is tracked-clean; do not modify/delete/rename/stage it.
- `HANDOFF.md`/`TASKS.md` are stale; `/half-clone` is forbidden.
- No hardware, dependency install, signal, commit, or push is authorized.

## Git

- At D402 case boot:
  `HEAD == origin/master ==
  9dd14ebb32421d93e3b46e2912cd3e67e6daff20`
  (`D401까지, D402는 미승인`).
- Worktree was clean before D402; it is now intentionally dirty only with the
  approved D402 static scripts/evidence and current-state documentation.
- Commit/push is not authorized.
