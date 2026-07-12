# Session 2026-07-12 - D335: audited target-family geometry repair

Status before runtime: `D335_PRE_REGISTERED_RUNTIME_PENDING`

이번 case의 신규 변수: `[target_family_geometry]`

The radial/tangent offset pair is one coupled target-family variable. This
section is written before the D335 run. Runtime results must be appended below
without changing the search domain, gates, outcome branches, or non-goals.

## Research question

D334 proved that the actual stage-extracted `gripper_link` raw collision mesh
overlaps the D34 x H90 cylinder at the frozen D325 command pose. Can one
HOME-seeded position-only target in the same grasp-semantic radial/tangent
family make the complete audited raw `link5 + gripper_link` tool surface clear,
while retaining every existing G0a commanded-pose alignment gate?

If and only if that zero-step geometry gate passes, does the selected target
remain clean during one D333-style sole-support static settle?

This run does not decide approach-path performance, the 10/10 G0a gate, G0b,
gripper close/lift, or collision-representation repair.

## Frozen comparison contract

- Source truth:
  `claudedocs/runtime_logs/grasp_track/g0a_d334/g0a_d334_live_collision_audit_summary.json`.
- Seed `33201`; one environment; physics `dt=0.005s`.
- Cylinder center `[0.300, 0.000, 0.032883]m`, radius `0.017m`, height
  `0.090m`, mass placeholder `0.72kg`, friction `1.5/1.2`.
- D333 sole-support scene: exact global-ground collider disabled, TapTable sole
  support, same cylinder-owned ContactSensor and zero reporter/sleep thresholds.
- Robot USD and URDF paths/hashes are locked to D334. The audited collision
  source is extracted from the live USD stage instance proxy; loading a
  differently named URDF collision STL is forbidden.
- Target family stays `position_only_tangent_minus1`: tangent sign `-1`, TCP z
  at cylinder center, HOME `[0,0,90,0,0,0]deg` seed, position-only IK
  (`max_iter=120`, `pos_tol_mm=1.0`), open gripper `q5=0`. Wrist/tool-axis,
  gripper angle, z, seed, solver, and nullspace are not variables.
- Existing commanded-pose G0a gates remain frozen: TCP error `<=5mm`, horizontal
  jaw-tangent error `<=15deg`, fixed-jaw proxy gap `[0,5mm]`, no proxy
  penetration, and contact point at least `15mm` below cylinder top.
- D334 raw clear/borderline threshold stays `+0.1mm`. No new safety margin is
  silently introduced.

## Deterministic target search

Offsets are expressed as
`TCP = center - radial * r - tangent * t`, with z reset to cylinder center.

### Grasp-semantic domain

- Radial `r = 0.00..17.00mm` inclusive, step `0.25mm`.
- Tangent `t = 9.00..14.00mm` inclusive, step `0.25mm`.
- Anti-retreat guard:
  `radial_tip_past_near_face = cylinder_radius - r >= 0`.
  Therefore a target cannot pass merely by retreating the TCP outside the
  cylinder near face. Expanding beyond `r=17mm` requires a separate user choice
  and a new grasp-depth contract.
- The tangent domain is the nominal `[0,5mm]` fixed-jaw-gap family derived from
  radius `17mm` and the frozen fixed-jaw face offset `8mm`; the live computed
  proxy gate is still applied to every candidate.

The coarse grid is complete and deterministic. Each candidate is solved from
the same HOME seed rather than continued from the previous candidate. For each
candidate, an exact-state write followed only by `sim.forward()` and
`scene.update(dt=0)` is allowed; `sim.step()` is forbidden.

If the coarse grid contains no raw-clear candidate, refine the union of
`r/t +/-0.50mm` neighborhoods around the five coarse candidates with the
largest minimum raw clearance, using a fixed `0.05mm` step, clipped to the
registered domain and de-duplicated. Top-five ranking is: larger minimum raw
clearance, then smaller Euclidean shift from old `(7,11)mm`, then numeric
`r`, then numeric `t`.

Among all passing candidates in the executed candidate set, selection is:

1. smallest Euclidean target shift from old `(7,11)mm`;
2. largest minimum raw-tool clearance;
3. numeric `r`, then numeric `t`.

No physics-result-driven second target, domain expansion, or adaptive retry is
allowed.

## Pre-physics hard gate

Probe-controlled physics-step count must remain exactly zero through this gate.

1. Reproduce the old `(7,11)mm` target as a negative control using the same
   raw stage meshes and no-step FCL path:
   - link5 raw `+4.2726455336mm`, `CLEAR`;
   - gripper raw `-5.9566769497mm`, `OVERLAP`;
   - signed-distance absolute parity tolerance `<=0.05mm`;
   - collision sign/EPA consistency must pass.
2. For a new candidate, both stage-extracted raw shapes (`link5` and
   `gripper_link`) must have `is_collision=false`, signed distance
   `>=+0.1mm`, and consistency hard check PASS against the analytic cylinder.
3. IK convergence and every frozen commanded-pose G0a gate above must pass.
4. Asset/pin/stage/source-mesh/step-counter contracts must pass.

Failure of item 1 or 4 gives
`D335_G0A_PREPHYSICS_CONTRACT_FAIL_STOP`. No passing candidate gives
`D335_G0A_TARGET_FAMILY_NO_FEASIBLE_CLEAR_STOP`. Both stop before physics.

Cooked-gripper clearance is not a target-selection gate. The D334 gripper cook
parity failure is the deferred collision-representation case; moving the target
until an uncertified cooked mirror clears would bundle variables.

## Conditional physics evaluation

Only a passing selected candidate licenses physics:

1. Reset exact HOME/object state and run the unchanged D333 200-step
   sole-support baseline. Its complete hard gate must pass.
2. Exact-write the selected target and run one 200-step static command hold.
3. Record contact, object pose/velocity, TCP, joints, support, and root at every
   step. Re-evaluate raw tool distances at pre-step, post-step-0, and final.
4. Static repair is supported only if all of the following hold:
   - root/support/runtime contracts pass;
   - link4/link5/gripper maximum filtered force is `<0.1N`;
   - no D333 disturbance: maximum object XY `<0.5mm`, tilt `<1deg`, with no
     vertical/support disturbance;
   - final existing G0a alignment gates pass and object displacement `<5mm`;
   - both raw tool shapes remain clear at post-step-0 and final.

## Pre-registered outcomes

1. `D335_G0A_PREPHYSICS_CONTRACT_FAIL_STOP`: negative-control, asset, pin,
   source-mesh, step-counter, or FCL consistency contract failed.
2. `D335_G0A_TARGET_FAMILY_NO_FEASIBLE_CLEAR_STOP`: no registered
   candidate in the executed deterministic coarse/refinement set clears the
   actual raw tool while retaining the frozen alignment gates. This is not a
   mathematical claim about every continuous offset. Stop before physics.
3. `D335_G0A_TARGET_FAMILY_STATIC_REPAIR_SUPPORTED_STOP`: pre-physics and
   static-clean gates pass. This licenses only a later frozen-target
   approach/10-trial case; `g0a_pass=false` here.
4. `D335_G0A_RAW_CLEAR_LIVE_COLLIDER_BLOCKED_STOP`: raw tool stays clear at the
   commanded/contact/final readings but live robot contact and object motion
   remain. Route to the separate collision-representation case; do not retune
   target offsets in D335.
5. `D335_G0A_STATIC_RUNTIME_MIXED_STOP`: raw overlap reappears during physics,
   alignment/runtime gates fail, or attribution is otherwise unresolved.

Every branch stops after D335. No ladder promotion or G0a PASS claim.

## Artifacts and Visualization DoD

- Forward-only output:
  `claudedocs/runtime_logs/grasp_track/g0a_d335/`.
- Candidate scan CSV/JSON, negative-control JSON, pre-physics gate JSON, summary
  JSON/Markdown; conditional baseline/target CSVs and raw-distance readings.
- Decision-time snapshot must show cylinder, raw link5/gripper surfaces,
  target-vs-commanded/actual frames, and nearest-point clearance witnesses.
- If physics runs, add first-event or final diagnostic snapshot. Total PNG count
  `1..3`; exactly one non-empty RRD; marker and artifact contracts must pass.
- A visualization/artifact failure changes the final verdict to
  `D335_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP` without erasing geometry
  or physics evidence.

## Session progress rule

The deterministic radial/tangent candidate evaluation is a perturbation
evaluation that can fail and can change the target-repair decision. If it
passes, the conditional 200-step settle is a second failable evaluation. No
training is authorized or required for this G0a geometry case.

## Non-goals

No mesh/collision-approximation rewrite, cooked-hull target compensation,
target z/wrist/nullspace/gripper-angle change, domain expansion after results,
waypoint/approach/10-trial run, close/grasp/lift, G0b, RL/PPO, randomization,
VLA, real RoArm, B200, cube, large render/video, cleanup, commit, or push.

## Pre-run amendment after independent contract review

No Isaac runtime or candidate evaluation had executed when this amendment was
written. The original artifact minimum (raw distances at pre-step,
post-step-0, and final) is retained, but the static classifier is hardened to
record and require raw `link5 + gripper_link` clearance at **every one of the
200 target-settle steps**. This prevents a transient mid-settle raw overlap
from being hidden by clear endpoint readings. It changes no physical variable,
search domain, candidate, or deferred cook policy.

The `RAW_CLEAR_LIVE_COLLIDER_BLOCKED_STOP` branch is also attribution-gated:
object disturbance and a `link5` or `gripper_link` filtered-force onset must
both exist; the audited-body onset must be no later than disturbance onset plus
one step; and any link4 onset must be absent or strictly later than both. Root
and support contracts must remain clean. Earlier/simultaneous link4 contact or
disturbance without an audited-body onset stays `STATIC_RUNTIME_MIXED_STOP`.

## Runtime result (appended after the run)

Verdict: `D335_G0A_TARGET_FAMILY_NO_FEASIBLE_CLEAR_STOP` (exit 0,
artifact contract PASS)

### 1. Contract and old-target negative control

- D334 USD/URDF hashes, live collision paths/owners, source mesh paths,
  topology counts, body-local bounds, stage/sensor contracts, and package pins
  all passed. Runtime pins remained `numpy==1.26.0`, `psutil==5.9.8`.
- Old `(radial,tangent)=(7,11)mm` negative control was bit-exact against D334:
  link5 raw `+4.2726455336mm` / `clear`, gripper raw
  `-5.9566769497mm` / `overlap`; both distance deltas `0.000000mm` and both
  consistency checks PASS.
- Probe sim counter remained `0 -> 0`; controlled physics steps before the gate
  and in the entire run were both `0`.

### 2. Deterministic target search

| Metric | Result |
|---|---:|
| coarse candidates | `1,449` |
| unique refinement candidates | `1,180` |
| total unique candidates | `2,629` |
| frozen alignment-gate pass | `2,560` |
| link5 raw CLEAR | `2,629 / 2,629` |
| gripper raw CLEAR | `0 / 2,629` |
| gripper raw OVERLAP / BORDERLINE | `2,422 / 207` |
| complete raw-tool-clear candidates | `0` |
| selected candidate | `null` |

CSV recomputation independently matched the JSON counts, refinement union,
top-five centers, ranking, and zero-pass verdict. All `2,629` candidate rows
record `sim_step_counter_unchanged=True`.

The best bounded candidate by the registered ranking was:

| Field | Result |
|---|---:|
| radial / tangent offset | `14.6 / 13.9mm` |
| tip past near face | `2.4mm` (anti-retreat PASS) |
| target shift from old `(7,11)` | `8.134494mm` |
| commanded TCP error | `0.970411mm` |
| jaw tangent error | `2.785821deg` |
| fixed-jaw proxy gap / penetration | `4.844665 / 0.000000mm` |
| contact point below top | `44.298907mm` |
| link5 raw distance/state | `+7.787464mm / CLEAR` |
| gripper raw ranking scalar/state | `-0.000121945mm / OVERLAP` |

The tiny gripper BVH distance scalar is **not** licensed as an EPA penetration
depth or physical near-miss magnitude. The decision fact is simply that the
candidate is `OVERLAP`, not `CLEAR`, and is below the registered `+0.1mm`
clearance gate.

### 3. Physics stop was correct

Pre-physics contract PASS, candidate gate FAIL, `physics_licensed=false`.
Consequently no baseline, target-settle, target raw-trace, or final-physics PNG
exists. This is required behavior, not a missing experiment. The 2,629-point
geometry perturbation evaluation itself satisfies the session progress rule
and changed the target-family decision.

### 4. Visualization and artifact audit

- Inspected decision snapshot:
  `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_prephysics_decision.png`
  (`294,522` bytes). It contains the analytic cylinder, both audited raw tool
  meshes, target/commanded/actual TCP frames, and nearest-point witnesses.
- Marker contract PASS with six frames.
- Exactly one RRD:
  `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_target_family_repair_trace.rrd`
  (`2,480,172` bytes, one decision-time trace step, actual/commanded URDF each
  eight joints, non-empty PASS).
- Summary/CSV/PNG/RRD SHA-256:
  - summary `7ca98f31d6fc23ea0942d4863d2d7dbdce561293e181d5b1bd7a451dd0064d0e`
  - candidate CSV `f7daa545c190416f1117c275c4e8b015bce721507c04500254adc074d25d5f79`
  - PNG `0518ceb1ae31cf63900e7b03e3fc09c6078ff920339ae0de946cea31b6ff9d92`
  - RRD `99ddf6e7f78f216c26ced91cf57e9f13c4279f8278e351797e587998ca861a57`

### 5. Interpretation and stop

- Scalar radial/tangent offsets in the registered grasp-semantic domain cannot
  produce an actual-raw-tool-clear target under the frozen HOME-seeded
  position-only orientation and G0a alignment gates. Do not repeat offset-only
  tuning, approach, or 10-trial runs in this family.
- Expanding beyond the anti-retreat boundary is not authorized: it changes the
  grasp-depth semantics and can turn clearance into a retreat-away-from-object
  result.
- The deferred gripper cook-representation issue is still real, but repairing
  it cannot substitute for an actual raw-tool-clear command. Target-family
  feasibility therefore remains first in precedence.
- `g0a_pass=false`; no target selected; no ladder promotion; G0b/RL remain
  blocked.

Recommended next user decision: either (A) add exactly one new target-family
variable for reachable wrist/tool orientation (reuse the same bounded r/t
domain and raw-tool pre-physics gate), or (B) explicitly redefine grasp-depth
semantics before permitting `r>17mm`. Option A is the non-retreat continuation;
collision-representation repair remains reserve until an actual raw-clear
target exists.

### Runtime evidence

- `sim_scripts/cyl34_top_view_d335_grasp_g0a_target_family_repair.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d335/g0a_d335_target_family_repair_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_candidate_scan.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_candidate_search.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_old_target_negative_control.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_prephysics_gate.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_prephysics_scene_contract.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_prephysics_decision.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_target_family_repair_trace.rrd`
