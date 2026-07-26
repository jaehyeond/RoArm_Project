# START_HERE.md

Last updated: 2026-07-26 KST. D388 attempt1 is frozen as an offline
partition-contract FAIL_STOP. No case is currently approved.

## Current Truth

- Pivot: RoArm cylinder grasp-track G0a.
- `q5=0` is CLOSED; frozen OPEN is `1.5413rad`.
- Historical D362 target only: radius/diameter/height `17/34/90mm`, mass
  `0.72kg`. A64 moved and tipped it but did not grasp it.
- Actual product: nominal diameter/height `29/50mm`, zelkova or walnut; mass,
  tolerance, COM, inertia, friction, bottom flatness, and roundness unmeasured.
- No `29x50mm` target has been authored, loaded, rendered, measured, or
  simulated. D362 physics and old visuals do not transfer.
- D368 A64 (`64 link5 + 64 gripper_link = 128`) is a 64-cap reference
  candidate, not an optimum or NVIDIA limit.
- D372 P34 is manual: link5 `16`, gripper_link `18`, total `34`. It is a design
  choice, not an optimum. D379 authored-to-cooked identity passed `17/34`;
  full P34 identity remains false.
- D387 completed the 11-layer failure map:
  upper `[28,null,12]`, lower `[12,null,28]`,
  fixed-left `[12,30]`, fixed-right `[13,12,12]`.

## Latest Case — D388

Case:

`D388 [two_null_moving_support_midlayer_partition_repair_design]`

이번 case의 신규 변수:

`null_middle_layer_first_blocked_triangle_reanchored_fan_graph_v1`

What was changed:

- Exact two D387 null middle layers only.
- One shared rule rotated the same CCW profile so the new fan anchor was the
  state after the frozen forward-reachable frontier.
- Derived anchors: upper `11`, lower `10`; old anchor was `0`.
- Other nine map entries and all polygon/face/surface/volume/no-overlap gates
  were inherited unchanged.

What was observed:

- Both old null-through64 graphs became finite diagnostic graphs.
- Upper: `B*=37`, `B36` no-cover, child `6`, cuts
  `[0,3,7,11,12,16,20]`; DP/exhaustive agree.
- Lower: `B*=35`, `B34` no-cover, child `7`; DP and exhaustive agree on
  minimum35 and child count but disagree on canonical cuts:
  `[0,2,5,9,10,14,18,22]` vs `[0,1,5,9,10,14,18,22]`.
- Polygon/face/surface/volume/positive-child gates passed.
- Registered Float32 overlap gate failed on every adjacent diagnostic seam:
  upper `5/15`, lower `6/21`; calculation failures `0`.
- Positive-volume sums: upper `1.0732770688656094e-14m^3`, lower
  `3.0558646686052954e-13m^3`.
- These are formal failures under the frozen 5nm halfspace-tolerance contract.
  Whether they are real physical penetration or tolerance/Float32 seam effects
  remains null.

Exact verdict:

`D388_REANCHOR_PARTITION_CONTRACT_FAIL_STOP`

Operational verdict:

`D388_ATTEMPT1_OFFLINE_WORKER_CLAIM_FAIL_STOP_NO_FINALIZE`

Plain-language verdict:

- The one-anchor change usefully converted both disconnected graphs to finite
  paths.
- It did not produce an admissible repair: neither target met B12, both
  overlap witnesses failed, and the lower canonical path contract failed.
- Do not call `37` or `35` selected/adopted budgets.

Execution and observability:

- prepare `23/23` PASS.
- worker/retry `1/0`, elapsed `4.434133296832442s`, return `1`;
  cooperative deadline exceeded=false, signal `0`, worker exited=true.
- Return1 was an intentional fail-stop after evidence/visual writes, not a
  crash or Isaac timeout.
- Exact `1920x1080` board and strict save-only RRD/RBL validation PASS.
- Manual visual inspection `8/9` FAIL: the board is readable, but Rerun
  geometry is too small to map child `6/7` one-by-one and has Korean glyph,
  proxy, and loading warnings.
- No completion summary; do not finalize, rerun, or overwrite attempt1.

Frozen nonclaims:

- global/common, selected, adopted, complete-P34 budgets `null`
- budget application `0`; complete counts `null`
- `materializable_candidate=false`
- live identity/GPU compatibility and physics/grasp `null`
- `p34_authored_to_cooked_identity_pass=false`
- `g0a_pass=false`
- other-nine evaluation/mutation, asset/USD, Isaac/Kit/PhysX, Warp/CUDA,
  cylinder, physics/q5/contact/grasp/target-IK-path/settings counters all `0`

Canonical output:

`claudedocs/runtime_logs/grasp_track/g0a_d388/attempt1_two_null_moving_support_midlayer_partition_repair_design/`

## Next Candidate — Not Approved

`D389 [d388_overlap_gate_numeric_provenance_and_canonical_tie_audit]`

Offline-only, immutable D388 evidence:

1. Reorder all lower B35 complete paths by the global canonical key and
   localize the DP tie-pruning discrepancy.
2. For the 11 adjacent seams compare pre/post-Float32 signed penetration,
   epsilon `0` intersection, and frozen `5nm` intersection; retain nonadjacent
   negative controls.

This case must not rerun D388, change the partition, relax a tolerance/gate,
select/apply a budget, or use USD/Isaac/PhysX/cylinder/physics/contact/grasp.
It requires separate explicit user approval.

## Remaining Nulls

- Whether the 11 positive seam volumes represent actual Float32 intrusion or
  only the registered 5nm numerical band/rounding.
- The globally canonical lower B35 partition.
- A complete admissible low-count compound and live authored/callback identity.
- Actual `29x50mm` geometry/readback/render and measured physical properties.
- OPEN gap, void/contact-patch identity, middle-height pose, closure/contact/
  tipping, force closure, hold/lift, grasp, and target/IK/path justification.
- `g0a_pass=false`.

## Authorization Boundary

- No active approved case.
- D388 attempt1 is consumed: no rerun, retry, overwrite, or finalize.
- Budget selection/application, partition or gate change, asset/USD
  materialization, Isaac/live identity, `29x50mm` target rebase, mass/pose,
  A64/P34 physics, q5/contact, hold/lift, G0b, RL/PPO/VLA remain unapproved.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, signal, dependency
  install, commit, or push is authorized.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D388; ledger tail
2. `claudedocs/session_20260726_grasp_g0a_d388_two_null_moving_support_midlayer_partition_repair_design.md`
3. D388 evidence, supervisor, CSV, board, Rerun validation, manual inspection
4. D387 session/evidence; D386; D385; D379 identity
5. D362 only as historical `34x90mm` cylinder evidence

## Git

- D388 approval and attempt1 fail-stop recording:
  `HEAD == origin/master == 930b41d98576a9c0bf1dce4f3eb1c0d93df8014b`,
  subject `D385`.
- D387 and D388 are expected uncommitted forward-only worktree additions and
  state-doc modifications.
- D388 execution script SHA-256:
  `7f99f80c19b4ab7e8adbae6237ed675feb738f9e1c4418049c1fa2f166c743bf`.
- D388 evidence/geometry SHA-256:
  `582368f093ba08fec0207967e8e24ac24f0a44774dfa1a7b8c82ae2b6781caba`,
  `c119ededf4400efbef55de4d89ccd6c1c8b4e33d4d3795710b6882d369f5e882`.
- Commit/push is not authorized.
