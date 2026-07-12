# Session 2026-07-13 - D337: open-jaw target gate (q5 convention repair)

Status before runtime: `D337_PRE_REGISTERED_RUNTIME_PENDING`

이번 case의 신규 변수: `[gripper_open_command]` (정확히 1개)

The single new physical variable is the commanded gripper joint value at every
D337 exact-state write: `q5 = 1.5413 rad` URDF — `98.1%` of the `1.571rad`
open limit, i.e. approximately `86.6deg` real opening under the D322 mapping
(`real max 88.3deg <-> URDF 1.571rad`); a deliberately **sub-maximum** opening
within the URDF limit `[0, 1.571]`. Everything else is frozen to
D334/D335/D336: the
`position_only_tangent_minus1` family (HOME seed, position-only IK
`max_iter=120`, `pos_tol_mm=1.0`, tangent sign `-1`, TCP z at cylinder
center), the bounded r/t domain (`r in [0,17]mm`, `t in [9,14]mm`,
anti-retreat `17mm - r >= 0`), every G0a commanded-pose alignment gate, the
`+0.1mm` raw clearance rule, seed `33201`, the D333 sole-support scene, and
the D334-locked robot USD/URDF hashes.

## Audit findings registered by this case (evidence, no asset change)

1. **q5 convention error (root-cause reclassification)**: the family
   definition since D325 says "open gripper `q5=0`", but the URDF gripper
   joint limit is `[0, 1.571]` and the D322 gripper contract states the real
   maximum opening `88.3deg` corresponds to URDF `1.571rad`. Therefore
   `q5=0` = jaws **closed**, `q5≈1.571` = open. Every D330-D336 exact write
   placed a closed moving jaw into the grasp volume. Design-time offline
   scoping (validated `0.2um`-level against the D336 runtime exact layer at
   the old target: offline `-6.4604mm` vs runtime `-6.4606mm`) shows the
   moving jaw overlaps the cylinder `-4.4..-10.7mm` at `q5=0` across family
   anchors and clears `+11.2..+12.5mm` at `q5=1.5413..1.571`, including the
   original D325 target `(7,11)mm` (gripper `+11.175mm`, link5 `+4.274mm`,
   all alignment gates green, table clearance `+85mm`).
   D334-D336 verdicts remain true as measured — for the closed-jaw
   sub-family. D336's family closure statement is hereby scoped to `q5=0`.
2. **USD/URDF gripper collision divergence (config-integrity finding)**: the
   live stage gripper collision source is the full `gripper_link.stl`
   triangle soup (`41,094` vertices / `13,698` faces — matches the D334
   stage extraction exactly), while the current URDF collision entry is
   `gripper_link_collision_g2a.stl`, a `4mm` box authored 2026-05-14 —
   one day **after** the robot USD was generated (2026-05-13). The USD is
   stale relative to the URDF file but is the **more physical** collision
   representation and remains the audited decision truth. No URDF/USD/mesh
   change is made in D337.

## Research question

With the gripper commanded open (`q5=1.5413`), does at least one target in
the same pre-registered r/t candidate set make the complete audited raw tool
surface (`link5 + gripper_link`) clear by `>=+0.1mm` while retaining every
frozen alignment gate — and does the selected target then remain clean during
one D333-style sole-support 200-step static settle?

This run does not decide approach-path performance, the 10/10 G0a trial gate
(a later case per user choice 2026-07-13), G0b close/lift, or
collision-representation repair.

## Frozen comparison contract

- Hash-pinned inputs (all verified at runtime):
  - D334 summary sha256
    `2ff44744df99c7a99d168cdd62a4f9186a5bbad6d673205282abb62b71097b26`
    (verdict `D334_G0A_ACTUAL_TOOL_OVERLAP_SUPPORTED`).
  - D335 summary sha256
    `7ca98f31d6fc23ea0942d4863d2d7dbdce561293e181d5b1bd7a451dd0064d0e`
    (verdict `D335_G0A_TARGET_FAMILY_NO_FEASIBLE_CLEAR_STOP`); D335 CSV
    sha256 `f7daa545c190416f1117c275c4e8b015bce721507c04500254adc074d25d5f79`
    (the canonical 2,629-key grid).
  - D336 summary sha256
    `f449801302bd21769aadc43e67fd6bb884071d29d32b9b1e29f0166297220f00`
    (verdict `D336_G0A_FINITE_GRID_CAVEAT_DISCHARGED_NO_CLEAR_STOP`); D336
    exact-rescore CSV sha256
    `5f76bde76cd0578883fafa952214a4345c79ba1cca0c5b685da1fd2b352a3853`
    (per-key BVH + exact-EPA parity expectations at `q5=0`).
- Robot USD/URDF hashes must match the D334 frozen contract. Audited
  collision source = live stage instance proxy, extracted exactly as in
  D335/D336 (`_build_raw_shapes` ownership/source-mesh parity contract).
- Cylinder `[0.300, 0.000, 0.032883]m`, r `0.017m`, h `0.090m`, mass
  placeholder `0.72kg`, friction `1.5/1.2`; D333 sole-support scene.
- Exact metric and judgment exactly as registered in D336: judgment =
  `is_collision=False` AND separation `>=+0.1mm` AND consistency (both
  bodies); ranking = contact-level EPA enumeration (`num_max_contacts=64`,
  cap-saturation flagged); the BVH scalar of a colliding mesh is never a
  judgment or proximity quantity.
- `q5` is written at **every** D337 exact-state write (controls at their
  registered values below; scan/physics at `1.5413`). Pins `numpy==1.26.0`,
  `psutil==5.9.8`.

## Deterministic evaluation (zero-step until the physics gate)

Every candidate: HOME-seeded position-only IK (arm joints; q5 is not an IK
variable), exact-state write (forward + `scene.update(dt=0)` only), live raw
FCL queries, live alignment gates. `sim.step()` is forbidden before the
physics gate; the sim step counter must be unchanged through every scan row.

### Controls (all before the scan; any failure = outcome 1)

1. **Closed-jaw legacy control** — old target `(7,11)mm` at `q5=0` must
   reproduce D334 bit-parity (link5 `+4.2726455336mm` CLEAR, gripper
   `-5.9566769497mm` OVERLAP, deltas `<=0.05mm`, states equal, consistency
   PASS) — identical to the D335/D336 gate.
2. **Exact-layer control** — same pose: gripper exact-EPA max depth must
   match the pinned D336 value `6.460556421930386mm` within `0.05mm`.
3. **Grid-parity control** — keys `(14.60,13.90)` and `(0.00,9.00)` at
   `q5=0`: BVH scalar, exact metric, and states must match the pinned D336
   rescore CSV rows within `0.05mm`.
4. **Open-jaw scoping cross-check** — old target at `q5=1.5413`: gripper raw
   exact clearance must be within `11.175 +/- 0.5mm` of the design-scoping
   prediction, and link5 raw distance must be unchanged from its `q5=0`
   value within `0.05mm` (link5 is q5-independent).

### Open-jaw scan

All 2,629 unique keys parsed from the pinned D335 CSV (parse count must be
2,629), evaluated at `q5=1.5413` with exact metrics and all frozen gates.
Aggregated invariant: max |link5 raw distance (D337 scan) - link5 raw
distance (D336 rescore)| over all keys `<=0.05mm` (q5-independence).

### Selection and decision revalidation (registered)

Among passing rows (both bodies clear + all frozen alignment gates + counter
unchanged): smallest Euclidean shift from old `(7,11)mm`, then largest exact
min clearance, then numeric r, then t. After the scan the decision candidate
(selected, or best if none) is re-materialized once outside the cache
(stage `decision_snapshot`); when a candidate is selected it must reproduce
pass state, an unchanged counter, and match its scan row within `0.05mm` on
BVH scalars and the exact metric with identical states — else outcome 1. The
snapshot is a repeat evaluation and is not counted in the unique executed-set
count.

## Conditional physics evaluation (only a selected candidate licenses it)

1. Exact HOME write (`q5=1.5413`) + unchanged D333 200-step sole-support
   baseline; its complete hard gate must pass.
2. Exact write of the selected target (`q5=1.5413`) + one 200-step static
   command hold. Record contact, object pose/velocity, TCP, joints, support,
   root at every step; record raw `link5 + gripper_link` clearance at every
   one of the 200 steps (D335 amendment rule).
3. Static repair is supported only if: root/support/runtime contracts pass;
   link4/link5/gripper max filtered force `<0.1N`; no disturbance (max XY
   `<0.5mm`, tilt `<1deg`); final G0a alignment gates pass with object
   displacement `<5mm`; both raw shapes clear at every recorded reading.
4. The `RAW_CLEAR_LIVE_COLLIDER_BLOCKED` branch is attribution-gated exactly
   as in the D335 amendment (audited-body onset no later than disturbance
   onset + 1 step; link4 absent or strictly later; root/support clean).

## Pre-registered outcomes

1. `D337_G0A_PREPHYSICS_CONTRACT_FAIL_STOP` — any control, hash, pin,
   parse-count, invariant, step-counter, or decision-parity contract failed.
2. `D337_G0A_OPEN_JAW_NO_FEASIBLE_CLEAR_STOP` — even with the jaw open, no
   candidate in the executed set passes. (Executed-set statement only.)
3. `D337_G0A_OPEN_JAW_STATIC_REPAIR_SUPPORTED_STOP` — pre-physics and
   static-clean gates pass. This licenses only a later frozen-target
   approach/10-trial case; `g0a_pass=false` here.
4. `D337_G0A_RAW_CLEAR_LIVE_COLLIDER_BLOCKED_STOP` — raw tool stays clear on
   all recorded readings but attributed live contact/motion remains; route
   to the deferred collision-representation case.
5. `D337_G0A_STATIC_RUNTIME_MIXED_STOP` — anything else (baseline fail,
   transient raw overlap, unattributed disturbance).
6. `D337_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP` — artifact/viz
   contract failure (does not erase geometry/physics evidence).

Every branch stops after D337. No 10-trial run, no ladder promotion, no G0a
PASS claim, no mesh/USD/URDF change.

## Artifacts and Visualization DoD

- Forward-only output: `claudedocs/runtime_logs/grasp_track/g0a_d337/`.
- `d337_frozen_contract.json`, `d337_prephysics_scene_contract.json`,
  `d337_negative_control.json` (4 controls), `d337_open_jaw_scan.csv`,
  `d337_search.json`, `d337_prephysics_gate.json`; conditional
  `d337_baseline_trace.csv`, `d337_target_static_trace.csv`,
  `d337_target_raw_distance_trace.csv`; summary JSON/MD.
- Design-scoping record copied to `g0a_d337/design_scoping/` (labeled
  design-time, not decision evidence).
- PNGs (1..3): decision snapshot (cylinder, raw meshes incl. the open moving
  jaw, TCP frames, witnesses); conditional static-final snapshot. Exactly
  one non-empty RRD (target-settle trace if physics ran, else decision
  trace); marker contract via `draw_frames`.
- Titles/captions must stay within executed-set claims (D336 wording rule).

## Session progress rule

The open-jaw scan is a perturbation evaluation that can fail (outcome 2) and
directly changes the repair decision; the conditional settle is a second
failable evaluation. No training authorized.

## Non-goals

No 10-trial/approach/waypoint run, no close/grasp/lift (the open command is
static, not a grasp), no wrist/tool-orientation variable, no URDF/USD/mesh
edit, no cook compensation, no domain expansion, no z/seed/solver change, no
G0b, no RL/PPO, no randomization, no VLA, no real RoArm, no B200, no cube,
no large render/video, no cleanup, no commit/push without explicit request.

## Pre-run amendment after independent adversarial review

No Isaac runtime, control, or scan evaluation had executed when this amendment
was written. A three-lens adversarial review (16/20 verification agents
completed; 4 were cut by a session usage limit and their low-cost defensive
fixes were adopted without re-verification) confirmed 3 MAJOR + 4 MINOR
findings, 4 refuted. Resolutions, none changing q5's numeric value, the search
set, gates, thresholds, or outcome semantics:

1. **(MAJOR, doc+harness label fix)** `q5=1.5413rad` was mislabeled as "the
   real maximum opening 88.3deg". Under the doc's own D322 mapping
   (`88.3deg real <-> 1.571rad URDF`), `1.5413rad` is `98.1%` of travel
   (~`86.6deg` real) — a sub-maximum opening. The numeric value is unchanged
   (Control 4's `11.175mm` expectation and the offline scoping are anchored
   to it); only the basis text in this doc and the harness was corrected.
2. **(MAJOR, harness fix)** The registered D335-summary hash pin and verdict
   were imported but never verified at runtime; `d335_verdict`,
   `d335_summary_sha256_pinned`, and `d334_sha256_matches_d335_record`
   checks were added to the frozen contract with the observed hash recorded.
3. **(MAJOR, harness fix)** The decision-time snapshot PNG was rendered after
   the conditional physics block, which would have mixed post-settle live
   poses with pre-physics decision data. It is now rendered immediately
   after the pre-physics gate dump, while the live stage still holds the
   decision-candidate pose (D335 ordering). The static-final PNG remains
   post-physics.
4. **(MINOR, harness fix)** Grid-parity control now also checks the link5
   BVH delta and guards every CSV/None float conversion (a missing value is
   a parity FAIL, not a crash).
5. **(bookkeeping)** Control evaluations run through the same cache as the
   scan; the `(7.00,11.00)` open-jaw control row retains stage
   `open_old_control` and the two grid-parity keys retain
   `grid_parity_control` inside `d337_open_jaw_scan.csv`. Scan completeness
   is audited by the CSV's total data-row count (`2,629 = scan_count`), not
   by `stage=='open_scan'` rows.
6. **(bookkeeping)** The design-scoping record under
   `g0a_d337/design_scoping/` is a manual pre-run copy placed before the
   harness runs and is outside the harness artifact gate.

## Runtime result (appended after the run)

Verdict: `D337_G0A_STATIC_RUNTIME_MIXED_STOP` (exit 0, artifact contract PASS,
physics executed: 200 baseline + 200 target-settle steps)

### 1. Contract and controls (all PASS)

- Frozen contract 17/17 (incl. the amended D335 summary hash/verdict checks);
  pins `numpy==1.26.0`, `psutil==5.9.8`; seed `33201`.
- Control 1 (closed-jaw legacy): bit-exact vs D334 (deltas `0.000000mm`).
- Control 2 (closed-jaw exact-EPA): matches pinned D336 `6.460556mm`.
- Control 3 (closed-jaw grid parity incl. link5 BVH): `0.000000mm` deltas.
- Control 4 (open-jaw scoping cross-check): observed `+11.175088mm` vs
  predicted `+11.175mm` (design scoping validated on the live stage);
  link5 q5-independence delta `0.0mm` (also `0.0mm` max over all 2,629 keys).

### 2. Open-jaw scan — the family is feasible with the jaw open

| Metric | Result |
|---|---:|
| scan keys | `2,629 / 2,629` |
| full-pass candidates (raw clear + all frozen gates) | `2,560` |
| legacy-alignment pass | `2,560` |
| selected candidate | `(7.00, 11.00)mm` — the original D325 target, shift `0` |
| selected exact min clearance | `+4.2726mm` (link5; gripper `+11.1751mm`) |

The q5 convention repair alone converts the family from `0/2,629` feasible
(D335/D336, jaw closed) to `2,560/2,629` feasible. Decision-parity PASS;
zero-step contract held through the scan (counter `0 -> 0` before physics).

### 3. Conditional static settle — MIXED, with a fully-resolved causal story

Gate results: baseline hard gate PASS; `raw_tool_clear_all_recorded_readings`
**PASS** (all 200 steps; min link5 `+7.498mm`, min gripper `+9.595mm`);
final G0a alignment PASS; final object displacement `2.754mm < 5mm`; root and
support contracts PASS. Failed gates: `robot_filters_max_lt_0p1n` (link5
`38.861N`), `d333_disturbance_free` (max XY `5.418mm`, tilt `4.208deg`,
disturbance from step 0), `audited_tool_contact_timing_compatible` (recorded
link5 onset `19` vs disturbance onset `0`).

Trace-level reading (`d337_target_static_trace.csv`):

- Step 0: link5 filtered force **`38.861N` impulse** (the run maximum) with
  object XY already `1.324mm` — contact and disturbance are simultaneous at
  step 0 in the trace. Steps 1-18 record `0N` while the object slides to a
  peak `5.39mm` (step ~10), then it settles back and rests against link5 with
  a steady `~1.70N` from step 19; final XY `2.754mm`, tilt `0.842deg`.
- The summary's `first_contact_step_by_link=19` reflects the sustained-episode
  onset and misses the step-0 impulse row; this onset-metric limitation is
  what broke the pre-registered attribution-timing gate (hence MIXED rather
  than COLLIDER_BLOCKED). No verdict override is claimed.
- Causal attribution (evidence, not verdict): the raw meshes never touch
  (both clear at every reading), the gripper records `0N` throughout, and
  D334 certified at exactly this commanded pose that the **link5 cooked
  convex hull** overlaps the cylinder `-6.2367mm` (mirror-cook artifact,
  volume parity `0.0498%`). The step-0 impulse + rest-against-link5 pattern
  is the expected signature of that inflated hull: **the physics collides
  with the cooked hull, not the raw mesh**.

### 4. Decision consequences

- The moving-jaw (q5 convention) blocker is repaired and verified: gripper
  contact `0N`, gripper raw clear at all times.
- The target family is feasible again (2,560 raw-clear candidates; the
  original `(7,11)` selected with margin).
- The single remaining blocker is the **collision representation** (cooked
  hull inflation, primarily link5; gripper cook parity `1.46%` FAIL remains
  from D334). The D334/D335/D336 deferral condition — "collision-
  representation repair stays deferred until an actual raw-clear target
  exists" — is now satisfied, so that repair becomes the next critical-path
  candidate (user approval required; it changes the collision asset, e.g.
  USD regeneration from the URDF with a faithful convex decomposition).
- `g0a_pass=false`; no 10-trial run; no ladder promotion; stop after D337.

### 5. Visualization and artifact audit

- Inspected decision snapshot (pre-physics, live pose = decision pose):
  `d337_decision.png` — open jaw visibly swung clear of the cylinder;
  link5 `+4.2726mm`/clear, gripper `+11.1751mm`/clear.
- Inspected static-final snapshot: `d337_static_final.png` — link5
  `+7.9549mm`, gripper `+9.6007mm`, object slightly shifted, all clear.
- Exactly one RRD with **200 settle-trace steps** (`2,979,719` bytes) —
  the first full-trajectory RRD in the grasp track; marker contract PASS.
- SHA-256:
  - summary `80df2f0b3765faee5bbeb190ded03bc326d54602fe16bf5c8fd73513fe5500d4`
  - scan CSV `0aa4adc8308ae2bf90ecb498f01f56aea86d342a97bfa13d6f74e3d0a87354ea`
  - decision PNG `52704925855393c955219c0bb9dbb01331f4fcb6291aea050fa3536b3473b9e1`
  - final PNG `85149aae3d67bcea3387743449888c58ba62fb111280936ef5141cd41be4ba22`
  - RRD `7a3785f45fafc795872791168f4d8ee0f8fe7eb56fa9ec14ba64609086464817`

### Runtime evidence

- `sim_scripts/cyl34_top_view_d337_grasp_g0a_open_jaw_target_gate.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d337/` (frozen contract, negative
  controls, scan CSV, search/gate JSON, baseline/target/raw-distance traces,
  2 PNGs, RRD, summary JSON/MD, `design_scoping/`)
