# Session 2026-05-19 - P7 Branch B next gate step plan

## Scope

- Continued Track A P7/Branch B only.
- Step-by-step next-gate planning after D052.
- Did not train.
- Did not run Isaac.
- Did not write a new runnable Isaac diagnostic script.
- Did not insert or integrate constraints into the RoArm chain.
- Did not attach SurfaceGripper.
- Did not execute transport, transport target, release, or scripted release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.
- Did not use `HANDOFF.md` or `TASKS.md`.

## Why We Are Not At Transport Yet

The next boundary is not transport/release. It is the local handoff boundary between
two separately verified facts:

1. Isolated Branch B dynamic-anchor contract works.
2. Real RoArm top-tangent 4mm CLOSE-near local signal works.

Missing evidence:

- No test has shown that an authored attach/constraint handoff can be created at
  the real RoArm top-tangent CLOSE-near pose.
- No test has shown that the handoff survives stationary hold and the same 4mm
  local micro-motion.
- No test has `attach_physics_validated=YES` in the RoArm handoff context.

Therefore the next gate is local attach/constraint handoff only, before MOVE
transport and before release.

## Source Structure Checked

Dynamic-anchor contract source:

- `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_probe.py`
  lines 1-9 define the isolated contract: CLOSE before MOVE/RELEASE, MOVE/HOLD
  require attached state, RELEASE only after target reached, no MOVE after release.
- Lines 48-55 implement `_contract_accepts`.
- Lines 58-71 implement negative-order checks.
- Lines 142-145 define isolated anchor/sponge/joint prim paths.
- Lines 146-160 create a dynamic gravity-disabled anchor.
- Lines 209-224 create a `UsdPhysics.FixedJoint` between anchor and sponge.
- Lines 233-242 implement closed-loop velocity target tracking.
- Lines 287-352 execute MOVE targets and gate final target error.
- Lines 382-413 execute release and gate release; this part is explicitly not part
  of the next RoArm local gate.

Dynamic-anchor target source:

- `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_target_probe.py` lines
  91-107 create a dynamic gravity-disabled anchor and sponge.
- Lines 136-147 create the fixed joint.
- Lines 161-170 implement velocity servo to a target.
- Lines 303-346 gate post-move hold and target error.
- Lines 357-389 execute release; this is not part of the next gate.

RoArm close-near signal source:

- `sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py` lines 1-9
  define the current no-overclaim virtual-carrier scope.
- Lines 215-244 define arguments and gates.
- Lines 246-250 enforce conservative side-edge guards; D052 means the next gate
  must not use side-edge.
- Lines 277-303 print strict no-overclaim scope and stream metadata.
- Lines 327-339 monkey-patch `_update_grasp_attach` into marker-only behavior.
- Lines 407-409 define the current virtual anchor as `tcp + tcp_to_anchor_offset`.
- Lines 410-517 execute and gate local RoArm events.
- Lines 599-644 compute success gates for prep, stationary hold, micro-motion,
  relative transform, uprightness, no hidden pose-write artifact, target error, and
  sim-step safety.

## Step-by-Step Path From Here

Step 1 - Current decision, done:

- Accept D050/D051 only as top-tangent local signal prerequisites.
- Exclude side-edge as a 4mm local signal carrier because D052 failed
  `micro_plus_x`.
- Keep side-edge only as pre-close pose/hold geometry evidence.

Step 2 - This document, done:

- Freeze the next gate as a top-tangent-only local handoff gate.
- Keep the next gate before MOVE transport and before release.
- Explicitly separate planning from approval to write/run a new Isaac diagnostic.

Step 3 - Static design target, not yet implemented:

- New diagnostic should be top-tangent only.
- It should reuse the close-near signal script's PRE_MOVE execution and top-tangent
  local event generation.
- At the top-tangent signal pose, it should create an authored local handoff
  object/constraint semantics for the sponge without using env TCP-center
  pose-write as success evidence.
- It should then test only:
  - close/handoff creation;
  - short stationary hold;
  - 4mm local `micro_plus_x`;
  - return to signal pose.
- It must stop before any MOVE transport target.
- It must not execute release.

Step 4 - Required falsifiable gates for the future diagnostic:

- `geometry=top_tangent`.
- `signal_stage` may be `just_before_close` first; `post_close_marker` is only a
  separate later variant if the first authored handoff gate passes.
- PRE_MOVE stream must complete with realized TCP gates.
- Handoff creation must have explicit state:
  - `constraint_prim_insertion=YES` only inside the diagnostic;
  - `fixed_dynamic_constraint_integration=DIAGNOSTIC_LOCAL_ONLY`;
  - `surface_gripper=NO`;
  - `transport_target=NO`;
  - `release_marker=NO`;
  - `p7_training=NO`;
  - `env_default_edits=NO`;
  - `chain_defaults_edits=NO`.
- The diagnostic must not claim success from `_grasped` or env pose-write.
- Stationary hold must pass:
  - final target error within the existing `0.003m` gate;
  - max TCP step within `0.010m`;
  - bounded sponge drift/speed;
  - upright preserved;
  - relative TCP/anchor/object transform bounded.
- 4mm local micro-motion must pass:
  - `micro_plus_x` reached;
  - return reached;
  - target error within `0.003m`;
  - no early kill;
  - no hidden kinematic pose-write artifact.
- Success must explicitly leave:
  - `attached_transport=NO`;
  - `transport_target=NO`;
  - `release_marker=NO`;
  - `release_physics_validated=NO`.

Step 5 - Go/no-go after that future diagnostic:

- If top-tangent authored local handoff fails stationary hold:
  stop. Do not transport. The attach/constraint handoff surface is invalid.
- If stationary hold passes but 4mm micro-motion fails:
  stop. Do not transport. The handoff is stationary-only, like the old
  offset-preserve limitation.
- If hold and 4mm micro-motion pass:
  still do not jump to release. The next boundary would be a separate, explicit
  attached micro-MOVE or first MOVE segment gate before any transport target.
- Release remains a later gate after attached movement evidence, not part of this
  next diagnostic.

## What Would Count As "Enough To Move On"

Enough evidence to move past local handoff would be a B200 run, explicitly approved
in a later step, where the future diagnostic reports:

- top-tangent only;
- authored attach/constraint semantics are explicit and diagnostic-local;
- no SurfaceGripper;
- no env TCP-center pose-write success mechanism;
- stationary hold OK;
- 4mm local micro-motion plus return OK;
- relative transform OK;
- upright preservation OK;
- no transport target;
- no release marker;
- stderr/process hygiene clean.

Only then is it rational to discuss the next gate. The next gate would still be
attached movement, not release.

## Current Recommendation

Do not run anything yet under the existing restrictions.

The next concrete action, if the user wants to continue implementation in this
session, is to ask for explicit approval to write a new diagnostic script only.
That script should be reviewed/compiled first and still not run Isaac until a
second explicit run approval.
