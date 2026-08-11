# t3w — preregistration: reach-boundary sweep over radius and approach azimuth (case `g0b_d420`)

Written **before** the sweep executes. Tag `t3w_*` is new and unused; nothing under
`t3r_*` / `t3t_*` / `t3d_*` / `t3p_*` is read-write here (all are read-only or
imported as libraries).

Script: `sim_scripts/p12_g0b_t3w_reach_boundary_radius_azimuth_sweep.py`
(`p10_*` is frozen and is *imported as a library*; `p11_*` is not modified and not
executed — the two constants this run reuses from it are re-derived, not copied
blindly, see §3.)

**이번 case의 신규 변수 (Variable Ladder, D322): ① 물체 위치 반경 `r`  ② 접근 방위
`Δψ`(반경 방향 대비 상대 각).** 그 외 모든 축은 52nd(`t3p_sweep1024_*`)와 동일하게
고정한다. 물체 D29×H50은 **불변**(HARD RULE #18).

---

## 1. Why this run exists

The 52nd session measured, for the first time, how far the tool axis can tilt away
from vertical before IK fails, and used the answer to retire the "side-face grasp"
option:

| limits | max feasible θ | residual tilt at θ = 90° |
|---|---|---|
| v6 distribution clip | **45°** | 43.14° |
| "URDF hardware envelope" | **50°** | 14.67° |

It also stated the limitation of that measurement in its own words
(`session_20260811_52nd_...md:176-180`, `t3p_prereg.md:131-133`):

> **Limitation**: one workspace point (r = 0.289779 m, ψ_radial = 317.5142°) and one
> IK formulation (vertical-biased DLS). It does **not** prove side grasps are
> impossible at other radii or with a different solver. A radius sweep was attempted
> this session and abandoned when it contended with the IK pool; it is the cheapest
> open follow-up.

`START_HERE.md:414-415` (52nd판) makes it item ⓐ of 1-NEXT and calls it *the only
unresolved limitation* of the side-grasp-infeasible claim. This run closes it.

**The question, stated so it can fail:** is θ_max ≈ 45° a property of *this pose*
(r = 0.2898 m, ψ_radial = 317.51°) or a property of *the arm + its joint clip*?

---

## 2. What is swept, and what is held fixed

| | value | source |
|---|---|---|
| object | cylinder **D 29 mm × H 50 mm**, centre at z = H/2 above the support plane | frozen, HARD RULE #18 |
| support plane | **z = 0** (`SUPPORT_Z_M`), *not* `TABLE_Z = −0.012117` | `t3p` §, `p11:116-121` |
| grasp point | object **top centre** | D419 |
| planner | `p10._build_plan_from_center` verbatim (approach → descend → lift, 5-task vertical-biased DLS) | `p10:630-675` |
| gates | `target_error_gate_m = 0.003`, `plan_tilt_gate_deg = 5.0` | 52nd defaults |
| margins | `grasp_surface_margin_m = −0.0010997957078144082` (clearance-0), `approach_clearance_m = 0.040`, `lift_delta_m = 0.025` | 52nd defaults |
| gripper | `descend_open_deg = 88.30998496351378`, close target 24.0° | 52nd `reachability_scan` |
| wrist roll | `PHI_STAR_DEG = 0.0` (roll turns about the tool axis; it moves neither TCP nor axis) | `p10:300-309` |
| **swept ①** | radius **r ∈ [0.100, 0.550] m step 0.025**, plus the exact 52nd radius `0.28977932314129784` | new |
| **swept ②** | approach azimuth **Δψ ∈ {0, 30, …, 330}°** relative to the radial direction | new |
| scanned | tool tilt **θ ∈ [0, 90]° coarse step 5°, then bisected to 0.25°** in the top bracket | refines 52nd's 10-point list |

`Δψ = 0` is the 52nd convention (tool leans radially outward). `Δψ = 180°` leans the
tool back toward the base — never tested, and the joints that saturate (elbow at its
upper bound, wrist-pitch at its lower bound) are used in the opposite sense there.

Feasibility of a cell is exactly the 52nd definition:
`approach_ik_ok AND descend_ik_ok AND lift_ik_ok`.

---

## 3. Three limit sets — and a transcription defect this run has to handle

`reachability_scan` compared two limit sets. Re-deriving them from source rather than
copying found that the second one is **not the URDF**:

| joint | `v6_clip` (`sim_scripts/roarm_kinematics.py:29-36`) | 52nd `"urdf_hardware"` (`p11:267-268`) | **URDF actual** (`local_assets/roarm_m3/urdf/roarm_m3.urdf:185,194,203,212`) |
|---|---|---|---|
| base | −90 … +90 | −90 … +90 | ±3.1416 rad = **±180.0°** |
| shoulder | −30 … +75 | −110 … +110 | ±1.5708 rad = **±90.0°** |
| elbow | +5 … +135 | −70 … +190 | [−1.0, 2.95] rad = **[−57.296°, +169.023°]** |
| wrist_p | −30 … +90 | −110 … +110 | ±1.92 rad = **±110.003°** |

The 52nd row labelled "URDF hardware envelope" is in fact the **CLAUDE.md hardware
table**, and it is **wider than the URDF on shoulder (110 vs 90) and elbow (190 vs
169.02)** and narrower on base (90 vs 180). Since the v6 boundary saturates the
*elbow upper* and *wrist-pitch lower* bounds, the widened row is **optimistic**: the
true URDF envelope can only be ≤ what it reported. So this run carries **three**
limit sets:

* `v6_clip` — the operative one (the distribution the policy lives in),
* `claudemd_table` — the 52nd row verbatim, for continuity,
* `urdf_true` — **parsed from the URDF file at runtime**, never transcribed.

`JOINT_LIMITS` is never removed and no hardware is commanded (HARD RULE #5). Widening
a *distribution clip* inside a read-only offline IK probe is the same device the 52nd
used, and exists only to attribute cause.

---

## 4. Stages, in execution order

* **R0 — anchor reproduction (blocking gate).** Re-run the 52nd `reachability_scan`
  θ-list `{6, 29, 35, 40, 45, 50, 60, 70, 80, 90}` at `seed0_S1` under `v6_clip` and
  `claudemd_table`. Must reproduce: `max_feasible_theta = 45 / 50`, `ok` pattern per θ,
  residual tilt at θ = 90° = **43.1438 / 14.67** (±0.001), r = **0.28977932314129784**,
  ψ_radial = **317.5141789665032**. **Any mismatch aborts the run** — an extension that
  cannot reproduce its own baseline is not an extension.
* **R1 — limit-set audit (report, not gate).** §3's table, computed at runtime.
* **R2 — the sweep.** 3 limit sets × 20 radii × 12 Δψ = **720 cells**.
* **R3 — focus cells, dense.** θ at 1° over [0, 90] for: `seed0_S1` under all three
  limit sets, and the arg-max cell of R2 under `v6_clip`. Feeds the RRD.
* **R4 — single-joint release ablation.** At `seed0_S1` and at the R2 arg-max radius,
  release **one** joint at a time from `v6_clip` to `urdf_true`. Answers "which single
  joint is the ceiling" instead of "some joint is".
* **R5 — ψ_pos null control.** The chain is a serial revolute about world z at the
  base, so the reachable set must be rotationally symmetric except through the base
  limit. Repeat one cell at ψ_pos ∈ {ψ_S1, 0°, +45°, −80°, +85°}; θ_max must be
  identical. A difference means the sweep's symmetry assumption is wrong.

Non-monotonicity in θ is **not assumed away**: the coarse grid is scanned in full, any
feasible θ above an infeasible one is flagged per cell (`nonmonotonic`), and the
bisection bracket is always [last feasible, first infeasible above it].

---

## 5. Verdict labels (preregistered, decided by the numbers and nothing else)

Let `θ*` = max feasible θ over **all** R2 cells under **`v6_clip`**.

| θ* | label | meaning |
|---|---|---|
| ≥ 75° | `SIDE_GRASP_REACHABLE_AT_OTHER_POSES` | a true side-face approach exists somewhere in the workspace; the 52nd retirement of option ② is **pose-specific** and must be revisited |
| 60° ≤ θ* < 75° | `TILT_CEILING_IS_POSE_DEPENDENT_BUT_NOT_SIDE` | materially better elsewhere, still not a side grasp |
| < 60° | `TILT_CEILING_IS_ARM_PROPERTY` | the 45° ceiling is a property of the arm + clip, not of the pose; 52nd's conclusion **generalizes** |

75° is "within 15° of horizontal", the point at which calling it *side-face* is fair.
60° is the smallest θ the 52nd coarse list reported infeasible, so it is the smallest
value at which this sweep would have contradicted it.

The same three labels are reported separately for `claudemd_table` and `urdf_true`,
but **only the `v6_clip` label is the verdict** — the other two are attribution.

---

## 6. What a result here does and does not license

* Kinematics only. **No physics, no Isaac, no robot.** `g0a_pass` stays **false**.
* No re-adjudication of D427 / D429 / D430 / D431 / D432 / D433 / D434 / D436 / D437 /
  D438 / D439 or any `-R` revision. Gate-0 is not re-run; the fully vertical case is
  not re-run.
* Reachability of a *plan* is not grasp success. A reachable side pose says nothing
  about whether the jaws could hold anything there — D439's force result stands
  untouched either way.
* **One IK formulation.** This sweep varies the pose, not the solver. The 52nd
  limitation had two halves; this closes the *pose* half only. A different solver
  (analytic, sampling-based, or a differently weighted DLS) is still unmeasured, and
  a `TILT_CEILING_IS_ARM_PROPERTY` verdict is therefore a statement about *this
  planner on this arm*, not a proof of impossibility.
* The grasp point stays object **top centre** (D419). A real side grasp would also
  want to re-target the grasp point to the object's side; that is a separate change
  and is **not** made here, precisely so the comparison against the 52nd number is
  like-for-like.
* Cell feasibility uses the 3 mm / 5° planning gates. Those gates are inherited, not
  re-justified.

## 7. Artifacts

`g0b_d420/t3w_<label>_{results.json, grid.npz, timeline.rrd, timeline.rbl,
rerun_validation.json, inspection.png, script.py.txt, argv.txt}`

A G0 guard aborts with exit 3 if **any** already exists (this also pre-empts
`rerun_contract.py:298-303`, which refuses to overwrite an inspection PNG and would
otherwise surface as a contract error instead of the real cause).

D341 applies: the verdict depends on pose, coordinate frames and geometry, so a
replayable RRD is mandatory — plus `.rbl`, footer-verified `rrd verify`, exact
entity/timeline/component contracts, a headless 2400×1400 screenshot, and an **actual
visual inspection** whose observations are recorded in the session doc. The RRD's
decision subject is the reach boundary itself: the sampled object positions coloured
by θ_max, and the arm configuration scrubbing through θ at the focus poses.

---

# POST-RUN CORRECTIONS (appended after `t3w_reach1_*` executed — original text above is
# left unedited on purpose; these are defects found in the prereg itself, by the
# adversarial panel `wf_11d30261-508` and by my own re-derivation)

**None of these changed a gate, a threshold, or the verdict.** They are defects in what
this document *says*, and one of them is a mislabel that was inherited from the 52nd and
then made worse here.

**C-1 — §5's justification for the 60° threshold is false.** It says "60° is the smallest
θ the 52nd coarse list reported infeasible". That reads the *abridged 10-point table* in
`session_20260811_52nd_...md:156-158`, and off the hardware row. The 52nd's **frozen**
artifact `t3p_sweep1024_plan.json /reachability_scan` scanned **14** θ values
`[0,6,15,29,35,40,45,48,50,55,60,70,80,90]`, and the smallest infeasible θ is
**48.0 under `v6_clip`** and **55.0 under `claudemd_table`** — never 60 on the verdict
axis. The threshold **value** 60 stays as preregistered (moving it after seeing results
is exactly what preregistration forbids); only its stated reason was wrong, and it was
inert here because the measured θ* = 81.25° cleared the *75°* branch.

**C-2 — `CLAUDEMD_TABLE_DEG`'s base bound is not the CLAUDE.md table value.** §3 calls the
row "the CLAUDE.md hardware table". It matches that table (`AGENTS.md:392-395`) on
shoulder / elbow / wrist_p, but its base bound `(−90, +90)` is the **v6 distribution
clip** (`sim_scripts/roarm_kinematics.py:30`), not the table's `−190 … +190`. The row is
a hybrid: three table bounds plus one v6 bound. Consequences: §3's phrase "narrower on
base (90 vs 180)" describes the row, not the table — against the real table CLAUDE.md is
**wider** than the URDF on base (190 vs 180.0004); and the `note` string emitted into
`t3w_reach1_results.json` repeats the mislabel verbatim. **Inert**: base tracks ψ_pos in
every executed cell (all cells `base = −42.4858…`, |base| ≤ 85 even in R5), and the R4
`release_base` ablation reproduces the `v6_clip` ceiling exactly, so no cell's result
depends on this bound.

**C-3 — §3 mis-converts the URDF wrist-pitch limit.** Stated ±110.003°; 1.92 rad is
**±110.00789666511805°**. The code parses the URDF at runtime and used the correct value;
only the prose is wrong.

**C-4 — §3's stated cause is one joint too many.** It repeats the 52nd's "saturates elbow
upper and wrist-pitch lower". At the *boundary* the run's own R4 ablation shows only the
**elbow upper** bound binds at `seed0_S1` (releasing wrist_p alone buys 0.0000°;
releasing elbow alone buys the whole +4.22°). Both saturate only well above the ceiling
(θ ≥ 60°). Attribution is also **radius-dependent**, which §3 did not anticipate:
`elbow@hi` at r ≈ 0.29, `wrist_p@lo` at r ≈ 0.40–0.50, and **no saturation at all** at
r = 0.525 (there the limit is plain reach on the `approach` waypoint).

**C-5 — undeclared behaviours that should have been in §4.** (a) `_best()`'s tie-break is
`(theta_max_deg, −r_m)`, i.e. ties prefer the **smaller** radius, and that choice selects
the R3 focus cell, the R4 ablation cell and the arm drawn in the RRD. (b) R4 ran the
**cross-product** of {2 radii × 2 Δψ} and added an undeclared `release_all` set, rather
than the two cells §4 describes. (c) Gate R0 checks the per-θ `ok` pattern for `v6_clip`
only. *Post-run repair for (c)*: all **28** frozen rows (14 θ × 2 limit sets) were
re-derived after the run and reproduce **bit-identically** on `ok`, `descend_tilt_deg`
and `descend_err_mm` (Δ = 0.00e+00 on every row) — a strictly stronger check than the one
preregistered.

**C-6 — a verdict-emitting pilot run happened before the production run, and the script
was edited in between.** `t3w_smoke1_*` (4 radii × 2 azimuths, 15° θ grid) was run as an
end-to-end pipeline test and it printed a verdict
(`TILT_CEILING_IS_ARM_PROPERTY`, θ* = 48.52° — **superseded, do not cite**). Its artifacts
are left on disk rather than deleted. Between it and `t3w_reach1_*` the script gained:
the `theta_just_over_*` attribution fields, TextLog stamping onto the timeline, a
three-way split of the plot panels, and the z-layering of the reach point clouds. The
first is an added output channel; the rest are rendering. No gate, threshold, sampling
rule or feasibility definition changed — but the edit-after-pilot sequence is disclosed
here rather than left to be discovered.

**C-7 — an undeclared visual convention in the RRD.** Every reach marker is displaced
**12 mm in xy** along its approach azimuth so the 24 azimuth cells of one radius do not
coincide. §7 says the decision subject is "the sampled object positions", and with this
nudge the markers are *not* at the object positions. Confirmed by inspection: adjacent
azimuth markers sit ~1 mm apart while being drawn at 4.5–7.4 mm radius, so they fuse into
a dashed line and per-azimuth structure is unreadable in the render. The z-layering of
the three limit sets **is** declared, in the RRD's own summary panel.

**C-8 — the executed azimuth grid is finer than §2/§4 declare.** §2 says
`Δψ ∈ {0, 30, …, 330}` (12 values) and §4 R2 says `3 × 20 × 12 = 720 cells`. The run was
launched with `--dpsi_step 15`, giving **24 azimuths and 1440 cells**
(`t3w_reach1_argv.txt`, `results.json /grid`). This is a deviation toward *more*
sampling, decided before launch and not after seeing any cell, but it was not written
back into §2/§4 and is therefore disclosed here. The 12 declared azimuths are a strict
subset of the 24 executed, so every preregistered cell was run.
