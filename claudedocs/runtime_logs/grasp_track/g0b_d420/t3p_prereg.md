# t3p — preregistration: randomized PARALLEL grasp sweep (case `g0b_d420`)

Written **before** the full sweep executes. Tag `t3p_*` is new; nothing under
`t3r_*` / `t3t_*` / `t3d_*` is read-write here.

Script: `sim_scripts/p11_g0b_t3p_cyld29h50_randomized_parallel_grasp_sweep.py`
(`p10_*` is frozen and is *imported as a library*, never modified or executed).

---

## 1. Why this run exists

Every physics trial in this project so far is **one deterministic rollout**:
`t3t_grasp{1,2,3}` contain zero RNG draws, one spawn point, `theta = 29°`, wrist
roll `+90.000°`, one friction pair, one gain set. leg3 is a strict prefix of
leg2 (D437-R1 (12)), and the close band that carried the only interesting signal
exists **only in leg3** (D438 (6)) — so the effective replicate count is n=1, and
for that band n=0. A single point in a ≥6-dimensional space was reported as
"grasping does not work".

Meanwhile `roarm_rl/roarm_stack_env.py:126-128` has been tensorized for 4096
envs since it was written. This run uses that.

**Contact force has never been measured in this project.** Every contact claim
to date (D434, D436, D436-R1..R4) is geometric inference. Item 2 below fixes
that, and its acceptance criterion is a *nonzero measured value*, not "the
sensor was created".

---

## 2. Contact instrumentation — the 9 items of D438-R1 Implication (2)

| # | Requirement | Where it is implemented |
|---|---|---|
| ① | `ContactSensor` registered in a `_setup_scene` **override**, into `self.scene.sensors[...]`, **before PLAY** | `P11SweepEnv._setup_scene`, after `super()` (which is where `clone_environments()` runs) |
| ② | **No dependence on `cfg.…activate_contact_sensors`** (measured neither necessary nor sufficient, D437-R1 (1)) | flag left at file default; arming happens at runtime |
| ③ | runtime `activate_contact_sensors(prim, threshold=0.0)` + **read-back assert** | spawn-func wrapper `_spawn_with_zero_threshold_reporter`; raises unless the authored threshold reads back `0.0` |
| ④ | **positive control preregistered on a window where contact must exist** | see §4 |
| ⑤ | **no** Kit-log `ContactReport` check (discriminative power 0) | absent by construction |
| ⑥ | every contact number carries a **phase label + denominator window** | all contact scalars are accumulated per named phase; `contact_steps` carries its denominator |
| ⑦ | object drift recomputed over the **whole close phase** | `close_obj_drift_max_mm`, max over every close step, not phase-end records |
| ⑧ | **tilt-corrected lift-off** as the primary metric, never raw `obj_z` | §3 |
| ⑨ | script copy + argv frozen beside the results | `t3p_*_script.py.txt`, `t3p_*_argv.txt` |

Declared instrumentation side effect: `activate_contact_sensors` also authors
`sleepThreshold = 0`. D437-R1 found this a **no-op for a single rigid body**
(`shapes.py:311`), but it is recorded here rather than presented as inert.

---

## 3. Success criterion (preregistered, discrete)

An env is `success` **iff all three** hold:

1. **tilt-corrected lift-off > 6.000 mm**
   A rigid cylinder that merely tips raises its centre without leaving the
   ground — D438-R1 F11 showed the previously reported "+0.0330 mm rise" was a
   3.3237° tilt. So the metric is the rise of the object's **lowest point**:
   `z_low = z_c − ((H/2)·cos t + R·sin t)`, `lift = z_low(final) − z_low(settled)`.
   Raw `obj_z` is **banned** as a success channel.
2. **final tilt < 30.1109°** (= `atan(D/H)`, the tipping half-angle: past it the
   object is falling over, not being carried).
3. **contact force > 1e-6 N** on at least one jaw during close or lift.

Verdict labels, decided by the above and nothing else:
`LIFT_SUCCESS_OBSERVED` / `CONTACT_BUT_NO_LIFT` / `NO_CONTACT_ANYWHERE` /
`MEASUREMENT_INVALID_POSITIVE_CONTROL_FAIL`.

---

## 4. Positive control (preregistered)

Window: **`settle`** — the object is at rest on the ground, so a ground reaction
**must** exist. Expected `F_z = m·g = 0.02483 × 9.81 = 0.2435823 N`.

* PASS iff ≥ 90 % of envs read a settle-window median `F_z` within ±35 % of that.
* If it FAILS, the run is reported `MEASUREMENT_INVALID_POSITIVE_CONTROL_FAIL`
  and **no contact claim may be made from it** — an all-zero force column would
  then be uninterpretable between "no contact" and "no measurement" (D438-R1 #71).

The ground filter path is **resolved from the stage at runtime**, not guessed:
d332's own positive control failed because it filtered on `/World/ground` (the
terrain root, which carries no collider) instead of the actual collision prim.

---

## 5. The 8 randomized axes

| # | Axis | Range |
|---|---|---|
| 1 | group | `measured_window` / `high_tilt`, 50/50 among feasible |
| 2 | tool tilt θ | `measured_window` U[6, 35]°; `high_tilt` U[35, 45]° |
| 3 | approach azimuth ψ | radial + U[−8, +8]° (D432 says radial is forced — this re-measures rather than assumes it) |
| 4 | descend depth δ | clearance-0 (−1.0997957078144082 mm) + U[−3, +3] mm |
| 5 | close target q5 | `measured_window`: fraction U[0.15, 0.75] of the **frozen** per-θ positive-bite window (window top is a STEP — D433 — so the max is never commanded). `high_tilt`: U[12, 40]°, band never measured there |
| 6 | friction | static U[0.30, 0.55]; dynamic = static × U[0.60, 0.90]; per-env via `root_physx_view.set_material_properties` |
| 7 | closing speed + gains | close ramp completes at U[0.25, 1.00] of the budget; stiffness U[40, 140], damping U[2, 8] |
| 8 | spawn | xy jitter U[−15, +15] mm, yaw U[0, 360)° |

Object **D29 × H50, 24.83 g is frozen, not swept** (HARD RULE #18).

θ windows come from `t3r_n10_ctq5_results.json` `.per_theta[*].collision.positive_windows_deg`
**read at runtime**, not transcribed. **Declared approximation**: the artifact
measures 7 discrete θ in [6, 35]; θ between them uses linear interpolation of the
window bounds. Interpolation is used only to place the target strictly inside a
window with margin — never to claim a bite value. Every sampled θ is recorded so
any row is re-derivable.

---

## 6. Scope change vs the instruction, and why

The session instruction asked for a 50/50 **top-down / side-face** split. A
side grasp needs the tool axis near horizontal (θ ≈ 90°).
`reachability_scan` (run first, stored in `results.json /reachability_scan`)
measures this at the actual pose and **it is not reachable**:

| limits | max feasible θ | residual tilt at θ = 90° |
|---|---|---|
| v6 distribution clip | **45°** | 43.14° |
| URDF hardware envelope | **50°** | 14.67° |

Under the v6 clip, elbow saturates at 135.0° and wrist-pitch at −30.0° from
θ = 60° upward. `JOINT_LIMITS` is never removed (HARD RULE #5); the hardware row
only widens the *distribution* clip to separate the two causes.

⇒ The side arm is **dropped as infeasible, and that infeasibility is itself
reported as a result**. The budget goes instead to the whole reachable tilt
continuum, whose upper half (**29–45°, never executed before**) is new ground.

**Limitation**: measured at one workspace point (`seed0_S1`, r = 0.289779 m)
with the vertical-biased DLS solver. It does **not** prove side grasps are
impossible at other positions or with a different IK formulation.

---

## 7. What a PASS would and would not license

* A PASS is a **simulation** result. `g0a_pass` stays **false**; no real-robot
  claim follows, and "T1 proves grip force" remains banned.
* The kinematic grasp attach of the RL env is **structurally disabled**
  (`P11SweepEnv._apply_action` never calls `_update_grasp_attach`), so a lift
  cannot be a kinematic pin. `kinematic_attach_calls` is reported and must be 0.
* A FAIL does **not** overturn D427/D429/D430/D431/D432/D433 — none is re-run
  or re-adjudicated here.

## 8. Known limitations, stated in advance

1. **Open-loop time-scheduled trajectory.** Fixed per-phase budgets, not
   servo-to-convergence like p10. Arrival is therefore *measured*
   (`descend_arrival_mm`, `close_q5_err_deg`, `lift_tcp_rise_mm`) and reported so
   "the arm never got there" is never conflated with "the grasp failed".
2. **`clone_in_fabric` is forced to False.** `ContactSensor` resolves bodies
   through USD, so fabric-only clones are invisible to it and initialization
   dies at N > 1. This is a departure from the env file's `True`, and it means
   this run is **not bit-comparable** to the 43rd legs.
3. Contact **points** are not tracked (`track_contact_points=False`); only
   filtered net forces per jaw. No gate here uses contact points.
4. Friction values are randomized around an **unmeasured** nominal (0.40/0.30
   was never physically measured on the real object).
5. Interpolated q5 windows, per §5.
6. One workspace point, per §6.

## 9. Artifacts

`g0b_d420/t3p_<label>_{results.json, plan.json, trace.npz, timeline.rrd,
timeline.rbl, rerun_validation.json, inspection.png, script.py.txt, argv.txt}`

A G0 guard aborts with exit 3 if **any** of them already exists (this also
pre-empts `rerun_contract.py`'s refusal to overwrite an inspection PNG, which
would otherwise surface as a contract error rather than the real cause).

D341 applies: the verdict depends on trajectory, contact and pose, so a
replayable RRD is mandatory — plus `.rbl`, footer-verified `rrd verify`, exact
entity/timeline/component contracts, a headless 2400×1400 screenshot, and an
**actual visual inspection** whose observations are recorded in the session doc.
