# Session 2026-07-12 - D336: finite-grid caveat discriminator (continuous/exact feasibility)

Status before runtime: `D336_PRE_REGISTERED_RUNTIME_PENDING`

이번 case의 신규 변수: `[]` (신규 물리 변수 0)

D336 introduces **zero** new physical variables. Object, friction, mass
placeholder, robot USD/URDF, HOME-seeded position-only IK family, tangent sign,
open gripper, anti-retreat boundary `r<=17mm`, every frozen G0a alignment gate,
and the `+0.1mm` raw clearance threshold are all identical to D335. The only
change is **methodological**: the feasibility search/judgment method inside the
already-registered domain. This section is written before the D336 run.
Runtime results must be appended below without changing the search method,
gates, outcome branches, or non-goals.

## Research question

D335 evaluated a finite deterministic grid (2,629 unique candidates) and found
no raw-tool-clear target. Its verdict carries a pre-registered caveat: it is a
finite executed-set result, not a statement about every continuous
radial/tangent offset. Two specific limitations create the caveat:

1. **Ranking bias**: D335 chose its five refinement basins by the raw BVH
   distance scalar (`hppfcl.distance` on a colliding `BVHModelOBBRSS`), which
   is not a physical penetration depth. The true shallowest-penetration basins
   (by contact-level EPA depth) may lie elsewhere, and were then only sampled
   at the coarse `0.25mm` step.
2. **Grid resolution**: outside the five refined neighborhoods the domain was
   sampled at `0.25mm`; a narrow clear pocket between coarse points is not
   excluded by the executed set.

Question: within the identical frozen family and domain, does a separately
pre-registered continuous/finer feasibility method — seeded and ranked by an
**exact contact-level EPA penetration metric** instead of the BVH ranking
scalar — find any candidate whose complete audited raw tool surface
(`link5 + gripper_link`) is clear by `>=+0.1mm` while every frozen G0a
alignment gate passes?

If yes, the candidate is **registered** as a physics-evaluation candidate only;
physics execution stays behind a separate later gate and does not run in D336.
If no, the finite-grid caveat is discharged to the extent of this method's
coverage, and the next decision moves to the reserve options (one new reachable
wrist/tool-orientation variable, or an explicit grasp-depth redefinition),
pending user choice.

This run does not decide approach-path performance, the 10/10 G0a gate, G0b,
gripper close/lift, or collision-representation repair.

## Frozen comparison contract

- Source truths (hash-pinned at runtime):
  - `claudedocs/runtime_logs/grasp_track/g0a_d334/g0a_d334_live_collision_audit_summary.json`
    (sha256 must equal the value recorded in D335's frozen contract:
    `2ff44744df99c7a99d168cdd62a4f9186a5bbad6d673205282abb62b71097b26`).
  - `claudedocs/runtime_logs/grasp_track/g0a_d335/g0a_d335_target_family_repair_summary.json`
    (sha256 `7ca98f31d6fc23ea0942d4863d2d7dbdce561293e181d5b1bd7a451dd0064d0e`),
    verdict must be `D335_G0A_TARGET_FAMILY_NO_FEASIBLE_CLEAR_STOP`.
  - `claudedocs/runtime_logs/grasp_track/g0a_d335/d335_candidate_scan.csv`
    (sha256 `f7daa545c190416f1117c275c4e8b015bce721507c04500254adc074d25d5f79`).
- Seed `33201`; one environment; physics `dt=0.005s`; D333 sole-support scene
  (exact global-ground collider disabled, TapTable sole support).
- Robot USD and URDF paths/hashes are locked to D334/D335. The audited
  collision source is extracted from the live USD stage instance proxy exactly
  as in D335 (`_build_raw_shapes` contract: single enabled collision prim,
  owner 1:1, source-mesh parity vs D334).
- Cylinder center `[0.300, 0.000, 0.032883]m`, radius `0.017m`, height
  `0.090m`, mass placeholder `0.72kg`, friction `1.5/1.2`.
- Target family stays `position_only_tangent_minus1`: tangent sign `-1`, TCP z
  at cylinder center, HOME `[0,0,90,0,0,0]deg` seed, position-only IK
  (`max_iter=120`, `pos_tol_mm=1.0`), open gripper `q5=0`. Wrist/tool-axis,
  gripper angle, z, seed, solver, and nullspace are not variables.
- Domain unchanged: radial `r in [0,17]mm`, tangent `t in [9,14]mm`,
  anti-retreat `17mm - r >= 0`. No expansion under any runtime result.
- Existing commanded-pose G0a gates remain frozen: TCP error `<=5mm`, live
  exact-written TCP error `<=5mm`, horizontal jaw-tangent error `<=15deg`,
  fixed-jaw proxy gap `[0,5mm]`, no proxy penetration, contact point at least
  `15mm` below cylinder top, IK convergence.
- Raw clear/borderline threshold stays `+0.1mm`
  (`d332.SIGNED_DISTANCE_BORDER_M`). No new safety margin.
- Runtime pins `numpy==1.26.0`, `psutil==5.9.8` verified before and by the
  frozen contract.

## Exact metric definition (judgment vs ranking separation)

Per audited body (`link5`, `gripper_link`), at each evaluated (r,t):

- **Judgment (unchanged, already exact)**: state `clear` requires
  `is_collision == False` (exact triangle-level BVH collide) **and** BVH
  separation distance `>= +0.1mm` (exact when not colliding) **and** the
  existing consistency hard check. The candidate `raw_tool_clear_pass`
  requires both bodies `clear`. The BVH distance scalar of a **colliding**
  query is never used as a judgment or reporting quantity for pass/fail.
- **Ranking metric (new, replaces the BVH scalar)**:
  `exact_signed_distance_mm` :=
  - if not colliding: `+` BVH separation distance (exact), else
  - if colliding: `-` max absolute contact-level EPA penetration depth over up
    to 64 enumerated contacts (`hppfcl.collide`, `enable_contact=True`,
    `num_max_contacts=64`).
  `exact_min_clearance_mm` := min over the two bodies. A colliding query with
  zero contacts fails the exact-consistency check and can only be borderline.
- **Registered limitation (no overclaim)**: contact-level EPA depth is a
  triangle-pair penetration measure. It is exact per enumerated contact but is
  a certified lower bound on solid penetration, and saturating the 64-contact
  cap is flagged (`epa_contact_cap_saturated`). It is used only for ranking
  and basin selection — never to relax the clear gate.

## Deterministic search method (pre-registered, no adaptivity beyond this)

All stages evaluate candidates through the same live path as D335: HOME-seeded
position-only IK offline, then `d332._write_exact_state` (exact state write +
`sim.forward()` + `scene.update(dt=0)` only), then live raw FCL queries and the
live alignment gates. `sim.step()` is forbidden; the sim step counter must be
unchanged across every evaluation and across the whole run.

Offsets are keyed in integer nanometers (`r_nm`, `t_nm`) so continuous-stage
records stay exact; the evaluation itself uses the float offsets.

### Stage A — exact re-scoring of the complete D335 grid

Re-evaluate all 2,629 unique (r,t) keys parsed from the hash-pinned D335 CSV
(`round(mm*1000)` micrometer keys; the parsed unique-key count must equal
2,629 or the contract fails). Every point is scored with the exact metric.
This removes the ranking-bias limitation over the full executed set.

### Stage B — continuous local maximization (Nelder-Mead)

- Seed eligibility: Stage-A rows that are exact-consistent and
  step-counter-unchanged. Rows failing either are assigned a `-1e9` sentinel
  ranking metric, which excludes them from basins (and from the "best point"
  used anywhere downstream) without deleting their records.
- Seeds: the top-5 eligible Stage-A points ranked by (larger
  `exact_min_clearance_mm`, then smaller Euclidean shift from old `(7,11)mm`,
  then numeric r, then t), plus the D335 best-ranked point `(14.6, 13.9)mm` if
  not already among them (max 6 seeds, deduplicated).
- Optimizer: `scipy.optimize.minimize`, `method='Nelder-Mead'`,
  objective `f(r_mm, t_mm) = -exact_min_clearance_mm`, bounds
  `[(0,17),(9,14)]mm` (clip-based), `xatol=1e-4mm`, `fatol=1e-5`,
  `maxfev=300` per seed, `adaptive=False`, default initial simplex.
- Every objective evaluation is nanometer-rounded, cached (repeat keys return
  the cached row without re-evaluation), and logged with stage `nm_seed_<i>`.

### Stage C — micro-grid verification

Around the single best point over Stages A+B (by `exact_min_clearance_mm`,
tie-broken as in Stage B ranking): `dr, dt in {-0.050..+0.050}mm` step
`0.005mm` (21x21 = up to 441 points), clipped to the domain, deduplicated
against the cache, stage `micro`.

### Candidate pass and selection

A candidate passes iff (a) both bodies are `clear` (judgment rule above, with
consistency), (b) every frozen alignment gate passes, (c) its sim step counter
is unchanged. Among all passing candidates in the executed evaluation set,
selection order is identical to D335: smallest Euclidean shift from old
`(7,11)mm`, then largest `exact_min_clearance_mm`, then numeric r, then t.
No physics-result-driven second target, domain expansion, or adaptive retry.

## Pre-run hard gate (controls)

Probe-controlled physics-step count must remain exactly zero for the entire
run (there is no conditional physics branch in D336 at all).

1. **Old-target negative control** — reproduce `(7,11)mm` with the unchanged
   D335 evaluator: link5 raw `+4.2726455336mm` / CLEAR, gripper raw
   `-5.9566769497mm` / OVERLAP, absolute distance parity `<=0.05mm`, state
   parity, consistency PASS (identical to the D335 gate).
2. **Exact-layer control at the old target** — the exact evaluator at
   `(7,11)mm` must report gripper `is_collision=True` with `>=1` contact and
   max-EPA depth `>=` the D334 recorded pose-A raw EPA depth minus `0.05mm`
   (read programmatically from the D334 summary `pose_a_prestep` raw gripper
   `penetration_depth_m`).
3. **Grid-parity control** — re-evaluate the D335 keys `(14.60,13.90)mm` and
   `(0.00,9.00)mm`; the BVH scalar must match the pinned D335 CSV rows within
   `0.05mm` with identical states.
4. Asset/pin/stage/sensor/source-mesh/step-counter contracts must pass
   (reused from D335 verbatim).

Failure of any control gives `D336_G0A_PREPHYSICS_CONTRACT_FAIL_STOP` before
the search result is licensed.

Bookkeeping note: control evaluations (the exact-layer old-target point and the
two grid-parity points) run through the same cached evaluator and therefore
appear in the executed evaluation set and its total count. The old target is an
overlap point and cannot pass; this changes no gate.

## Pre-registered outcomes

1. `D336_G0A_PREPHYSICS_CONTRACT_FAIL_STOP`: any control, asset, pin,
   source-mesh, parse-count, or step-counter contract failed. Search
   conclusions are not licensed.
2. `D336_G0A_FINITE_GRID_CAVEAT_DISCHARGED_NO_CLEAR_STOP`: the registered
   exact re-scoring + continuous + micro-grid method found no candidate with
   both raw tool shapes clear `>=+0.1mm` under the frozen alignment gates.
   The D335 finite-grid caveat is discharged **to this method's coverage**;
   this is still not a mathematical impossibility proof over the continuum.
   Next decision routes to the reserve options (one reachable
   wrist/tool-orientation variable — not the unreachable D323 strict-axis
   family — or explicit `r>17mm` grasp-depth redefinition), pending user
   choice. Stop without physics.
3. `D336_G0A_RAW_CLEAR_CANDIDATE_REGISTERED_STOP`: at least one candidate
   passed. The selected candidate is registered as a physics-evaluation
   candidate only. Physics execution (D333-style baseline + static settle)
   remains behind a separate later gate and did not run. `g0a_pass=false`.
4. `D336_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP`: a visualization or
   artifact contract failure changes the final verdict without erasing the
   geometry evidence.

Every branch stops after D336. No ladder promotion, no G0a PASS claim, no
mesh/collision-representation change, no physics.

## Artifacts and Visualization DoD

- Forward-only output: `claudedocs/runtime_logs/grasp_track/g0a_d336/`.
- `d336_frozen_contract.json`, `d336_prephysics_scene_contract.json`,
  `d336_negative_control.json` (three controls), `d336_exact_rescore.csv`
  (Stage A), `d336_continuous_scan.csv` (Stages B+C), `d336_search.json`,
  `d336_prephysics_gate.json`, summary JSON/MD.
- Decision-time snapshots (total PNG count 1..3, exactly one non-empty RRD):
  1. `d336_decision.png` — cylinder, both audited raw tool meshes,
     target/commanded/actual TCP frames, nearest-point witnesses at the
     selected (or best) candidate.
  2. `d336_exact_clearance_map.png` — 2D map of `exact_min_clearance_mm` over
     the Stage-A grid with Stage-B/C evaluation points and the best/selected
     point marked (decision-time diagnostic of the basin structure).
- Marker contract via `roarm_rl.viz_debug.draw_frames`; RRD via `log_rerun`
  with actual/commanded URDF joint states at the decision candidate.
- A visualization/artifact failure yields outcome 4 without erasing evidence.

## Session progress rule

The exact re-scoring plus continuous/micro search is a perturbation evaluation
that can fail and directly changes the target-repair decision (found vs
discharged). No training is authorized or required. Control-contract items are
reused, not hardened; no reactive hardening is included.

## Non-goals

No physics (`sim.step`) of any kind, no mesh/collision-approximation rewrite,
no cooked-hull target compensation, no target z/wrist/nullspace/gripper-angle
change, no domain expansion (`r>17mm` forbidden), no waypoint/approach/10-trial
run, no close/grasp/lift, no G0b, no RL/PPO, no randomization, no VLA, no real
RoArm, no B200, no cube, no large render/video, no cleanup, no commit/push
without explicit user request.

## Pre-run amendment after independent adversarial review

No Isaac runtime, control, or candidate evaluation had executed when this
amendment was written. A three-lens adversarial review (contract fidelity,
hppfcl geometry/numerics, Isaac runtime execution; every finding independently
verified, none refuted) produced one MAJOR and four MINOR findings, resolved
as follows before runtime. No search-method constant, gate threshold, domain,
or outcome semantics changed.

1. **(MAJOR, fixed in harness)** The clearance-map snapshot title in the
   no-candidate branch said "no point is >= +0.1mm" — an unqualified
   domain-wide claim this method cannot license. The title is now derived from
   the evaluated data: "(no evaluated point >= +0.1mm)" only when Stage A ran
   and no evaluated point reached `+0.1mm`; "(raw-clear point(s) found but
   none passed the frozen alignment gates)" when a raw-clear evaluated point
   exists without a full-gate pass; "(search not executed: contract fail)"
   when controls failed before Stage A.
2. **(registered gate, doc-only)** After the search, the decision candidate
   (selected candidate, or best point if none selected) is re-materialized
   once via a fresh evaluation outside the cache (stage `decision_snapshot`).
   When a candidate is selected, it must reproduce its pass state and an
   unchanged sim step counter and must match the cached row within `0.05mm`
   on both raw BVH scalars and the exact metric, with identical raw states
   and exact offsets; otherwise the run stops with outcome 1. This snapshot
   is a repeat evaluation of an already-executed key and is not counted in
   the unique executed-set count. (Same protective behavior as the D335
   harness; now declared.)
3. **(bookkeeping, doc-only)** The three control-evaluated grid keys
   (`(7.00,11.00)`, `(14.60,13.90)`, `(0.00,9.00)` mm) retain their control
   stage labels (`old_target_exact_control`, `grid_parity_control`) inside
   `d336_exact_rescore.csv`. Stage-A completeness is audited by the CSV's
   total data-row count (2,629 = `stage_a_count`), not by counting
   `stage=='rescore'` rows (which number 2,626).

## Runtime result (appended after the run)

Verdict: `D336_G0A_FINITE_GRID_CAVEAT_DISCHARGED_NO_CLEAR_STOP` (exit 0,
artifact contract PASS, controlled physics steps `0`, sim counter `0 -> 0`)

### 1. Contract and controls

- Frozen contract PASS (13/13): D334/D335 verdicts, all three input hashes
  bit-pinned (D334 summary `2ff44744…`, D335 summary `7ca98f31…`, D335 CSV
  `f7daa545…`), robot USD/URDF hashes match D334, seed `33201`, parsed grid
  key count `2,629`, method constants exact, pins `numpy==1.26.0`,
  `psutil==5.9.8`.
- Old-target negative control bit-exact vs D334: link5 raw
  `+4.2726455336mm` / clear, gripper raw `-5.9566769497mm` / overlap, both
  deltas `0.000000mm`.
- Exact-layer control PASS: at the old target, the 64-contact enumeration
  reports gripper max EPA depth `6.460556mm >= 5.863007mm` (the D334
  contact(0) value) with `is_collision=True` and 64 contacts.
- Grid-parity control PASS: `(14.60,13.90)` and `(0.00,9.00)` reproduce the
  pinned D335 BVH scalars with `0.000000mm` delta and identical states.

### 2. Search result

| Metric | Result |
|---|---:|
| Stage A exact rescore | `2,629 / 2,629` keys |
| Stage B new NM evaluations | `322` (6 seeds; nfev 69/74/77/61/105/10) |
| Stage C new micro evaluations | `230` |
| Total unique evaluations | `3,181` |
| exact-consistent / counter-unchanged | `3,181 / 3,181` |
| raw-tool-clear candidates | `0` |
| full-pass candidates | `0` |
| gripper raw state | overlap `2,974`, borderline `207`, clear `0` |
| link5 raw state | clear `3,181 / 3,181` |
| best exact clearance (any gate state) | `-4.285374mm` at `(15.3897, 9.0000)mm` |
| best exact clearance among alignment-passing | `-4.396193mm` at `(15.2774, 9.0446)mm` |
| worst exact clearance | `-11.299095mm` |

Independent CSV recomputation matched every count, the zero-pass verdict, and
the per-stage attribution (`rescore 2,626 + control-labeled 3` in the Stage-A
CSV, exactly as pre-registered in the amendment; `322 nm + 230 micro` in the
continuous CSV).

### 3. Ranking-bias finding (decision-relevant)

The exact metric relocated and re-quantified the entire basin structure:

- D335's best-ranked point `(14.6, 13.9)mm` (BVH scalar `-0.000122mm`,
  suggesting a near-miss) has an enumerated EPA contact `7.830227mm` deep.
  The D335 refinement neighborhood (`r 14-17mm`, `t 12.5-14mm`) is `-7` to
  `-8mm` deep by the exact metric.
- The true shallowest-penetration basin sits at the `t=9.0mm` domain boundary
  near `r=15.4mm`, still `-4.285mm` deep. All six Nelder-Mead runs converged
  there (five from the exact top-5 seeds; the D335-best seed moved to the
  `(17,14)` corner and stalled, nfev 10).
- Every one of the `3,181` evaluated targets has a certified EPA contact at
  least `4.285mm` deep on the gripper raw mesh (64-contact cap saturated
  everywhere, so these depths are lower bounds of solid penetration).
- The best point fails alignment only via the fixed-jaw gap gate
  (`-0.034682mm` at the `t=9.0` boundary); this does not change the verdict
  since it is not raw-clear anyway.

Interpretation: the D335 finite-grid caveat is **discharged decisively**. The
position-only radial/tangent family is not a near-miss family blocked at the
`+0.1mm` threshold; the audited gripper raw mesh penetrates the cylinder by
millimeters at every evaluated target, including the continuous optimum. A
finer grid or better optimizer cannot plausibly bridge a `>=4.29mm` certified
penetration to `+0.1mm` clearance inside this family. (Still registered as an
executed-set statement, not a continuum impossibility proof.)

### 4. Physics stop was correct

`physics_licensed=false` by construction; no baseline/settle/target artifacts
exist. The 3,181-point exact geometry perturbation evaluation satisfies the
session progress rule and changed the target-repair decision (it removed the
"maybe the grid just missed a pocket" branch from consideration).

### 5. Visualization and artifact audit

- Inspected decision snapshot:
  `claudedocs/runtime_logs/grasp_track/g0a_d336/d336_decision.png` (`294,775`
  bytes) — cylinder, both raw tool point clouds, target/commanded/actual TCP
  frames coincident, witness segments. Note: the figure subtitle prints the
  legacy BVH scalar (`gripper -3.0258mm`) for continuity with D334/D335
  figures; the exact EPA metric at the same pose is `-4.2854mm` and is the
  decision quantity.
- Inspected clearance map:
  `claudedocs/runtime_logs/grasp_track/g0a_d336/d336_exact_clearance_map.png`
  (`112,992` bytes) — full-domain exact clearance field, 552 continuous/micro
  points, decision star at `(15.3897,9.0)`, old target marked; title
  correctly scoped to "no evaluated point >= +0.1mm".
- Marker contract PASS (six frames, `/World/D336CaveatDiscriminatorFrames`);
  exactly one non-empty RRD (`2,481,881` bytes, decision-time trace,
  actual/commanded URDF eight joints each).
- SHA-256:
  - summary `f449801302bd21769aadc43e67fd6bb884071d29d32b9b1e29f0166297220f00`
  - rescore CSV `5f76bde76cd0578883fafa952214a4345c79ba1cca0c5b685da1fd2b352a3853`
  - continuous CSV `0363324d66adf81d377a535773268716d2aa5315335cafefc851a6358decc1fc`
  - decision PNG `73728fa929c7e7daabb1a3896d62c620ef4a290a36c228929733de8a8fb78c75`
  - clearance map PNG `633062e6d08066e56c1eb81115b818e3f88800806b99f1dd6bda0781bd9791c0`
  - RRD `21709b222ec778506f032420915b041edfc893ce646f0d3e8fc3e4a5daa62f92`

### 6. Stop and next decision

- `g0a_pass=false`; no candidate registered; no physics; no ladder promotion;
  G0b/RL remain blocked.
- Offset-only repair inside the HOME-seeded position-only family is now
  closed at millimeter scale, not merely at the finite-grid level. Reserve
  option (A) — exactly one new reachable wrist/tool-orientation variable
  (reusing the same bounded r/t domain and raw-tool pre-physics gate, not the
  unreachable D323 strict-axis family) — is the non-retreat continuation and
  is now quantitatively motivated: the tool needs `>=~4.4mm` of effective
  clearance change that position offsets cannot produce. Option (B) remains
  the explicit `r>17mm` grasp-depth redefinition. User choice required;
  nothing is implemented by D336.

### Runtime evidence

- `sim_scripts/cyl34_top_view_d336_grasp_g0a_finite_grid_caveat_discriminator.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d336/g0a_d336_finite_grid_caveat_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d336/d336_exact_rescore.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d336/d336_continuous_scan.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d336/d336_search.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d336/d336_negative_control.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d336/d336_prephysics_gate.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d336/d336_decision.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d336/d336_exact_clearance_map.png`
