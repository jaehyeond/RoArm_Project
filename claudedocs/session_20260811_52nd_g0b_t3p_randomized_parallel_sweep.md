# 52nd — t3p: the first randomized, parallel, force-instrumented grasp trial

Case `g0b_d420`. Date 2026-08-11 KST. Tag `t3p_*` (new; nothing frozen was touched).

> **One line**: 1024 randomized envs, contact force measured for the first time in
> this project, positive control PASS 1024/1024 — and the object never leaves the
> ground in any of them. The mechanism is now measured, not inferred: **the jaws
> press the object down into the table (ground reaction up to 25.9 × its weight)
> instead of pinching it, and the contact vanishes the instant the arm lifts.**

Physics **executed** (7 sessions of "no physics" ends here): 1024 envs × 920 steps,
physics wall **11.4 s**, total wall **355.6 s**.

---

## §1 What was run

| | |
|---|---|
| script | `sim_scripts/p11_g0b_t3p_cyld29h50_randomized_parallel_grasp_sweep.py` (new; `p10_*` imported as a library, never modified) |
| prereg | `g0b_d420/t3p_prereg.md` (written before the run) |
| asset | attempt3 `roarm_m3.usd` sha `a4be58e87b1f9790` (local URDF banned for grasp physics, D430 ④) |
| envs | **1024** (512 `measured_window` + 512 `high_tilt`) |
| seed | 20260811 |
| verdict | **`CONTACT_BUT_NO_LIFT`**, `n_success = 0/1024` |
| artifacts | `t3p_sweep1024_{results.json, plan.json, trace.npz, timeline.rrd, timeline.rbl, rerun_validation.json, inspection.png, script.py.txt, argv.txt}` |

`results.json` sha-16 fields, artifact manifest: plan `ef4c0a4d1894ea93` (1,832,803 B) ·
trace `df13e6162d018c5c` (2,086,306 B) · rrd `8f9d9447a6b4e75f` (194,364 B) ·
rbl `0cbe99d665de327b` (55,697 B) · validation `c1ab47ee780e7da2` (32,633 B) ·
png `7f5c754da9d90079` (994,379 B) · script `02286630b434bb1a` (64,567 B).

**RNG draws: 8 axes × 2048 pool samples.** Prior total across the project's entire
physics history: **0** (D437-R1 (12)).

---

## §2 ★★★ The headline: contact force exists, lift does not

### 2-1 The measurement is valid before it is interpreted

Preregistered positive control (§4 of the prereg): during `settle` the object rests
on the ground, so a reaction **must** exist and must equal its weight.

| | |
|---|---|
| expected `m·g` | **0.243582 N** |
| measured settle-window median | **0.243568 N** |
| envs inside the ±35 % band | **1024 / 1024** |

Agreement to **4 decimal places**. This is the first time a contact force has been
measured in this project at all; every previous contact statement (D434, D436,
D436-R1…R4) was geometric inference. An all-zero force column is now interpretable,
which is exactly what D438-R1 #71 demanded.

### 2-2 Contact happens, and it happens on both jaws

| | all (1024) | measured_window (512) | high_tilt (512) |
|---|---|---|---|
| any contact (close **or** lift) | **752 (73.4 %)** | 480 (93.8 %) | 272 (53.1 %) |
| **both jaws loaded** (close phase) | **229 (22.4 %)** | 146 (28.5 %) | 83 (16.2 %) |
| `f_fixed` close, median | 0.4498 N | 1.4313 N | 0.0000 N |
| `f_fixed` close, max | **6.8599 N** | 5.7494 N | 6.8599 N |
| `f_moving` close, max | **5.0142 N** | 2.5201 N | 5.0142 N |

Two-jaw subset (n = 229): `f_fixed` median **1.5661 N**, `f_moving` median
**0.7321 N**, **moving/fixed ratio median 0.5813** (p10 0.289, p90 0.806).
Contact even *survives* into the lift phase in most of them
(lift `f_fixed` median 1.0908 N, `f_moving` 0.5838 N).

⇒ **"the moving jaw never opposes" is false.** Opposing contact is achieved in
roughly a quarter of randomized configurations, at forces up to 5 N — an order of
magnitude above the object's 0.244 N weight.

### 2-3 And yet nothing is ever lifted

Preregistered gate: **tilt-corrected lift-off > 6.000 mm**.

| | value |
|---|---|
| best tilt-corrected lift-off across **all 1024 envs** | **0.000138 mm** |
| envs > 6.000 mm | **0** |
| envs > 1.000 mm | **0** |
| envs > 0.100 mm | **0** |
| TCP lift rise, median | **23.04 mm** (the arm did lift) |
| descend arrival error, median | 1.31 mm |
| close q5 command error, median | 0.0068° |

The arm arrives, closes to its commanded angle, and raises the TCP 23 mm. The object
stays on the table. The gate is missed by a factor of ~4 × 10⁴.

**This is not a "the arm never got there" artifact** — that is what the arrival
diagnostics exist to exclude.

### 2-4 ★★★ The mechanism, measured

The ground-reaction channel settles it:

| ground reaction during close | value |
|---|---|
| settle baseline | 0.2436 N = **1.00 × m·g** |
| close max, all envs (median) | 0.6316 N = **2.59 × m·g** |
| close max, two-jaw envs (median) | 1.6879 N = **6.93 × m·g** |
| close max, two-jaw envs (**maximum**) | **6.2955 N = 25.85 × m·g** |
| envs pressing > 2 × m·g into the ground | **561 / 1024 (54.8 %)** |

The jaws are not pinching the object — they are **pressing it into the table, and
the table is carrying the load**. Up to ~26 times the object's own weight is being
reacted by the ground rather than by an opposing jaw.

Per-step trace of the strongest two-jaw env (256-env preflight, env 170, identical
mechanism):

```
phase      f_fixed  f_moving  f_ground   /mg   obj_z_mm   tcp_z_mm
descend     0.0000    0.0000    0.2436   1.00     25.000     79.74
close       3.3470    2.4658    2.9079  11.94     26.149     47.69
close       3.0657    2.5442    3.4604  14.21     26.348     47.91
hold        3.0655    2.5444    3.4605  14.21     26.348     47.91
lift        2.3951    2.5305    2.5020  10.27     26.304     47.92   <- lift begins
lift        1.3661    1.7417       ...                       47.92
lift        0.0000    0.0832    0.2181   0.90     26.456     48.58   <- jaws gone
lift        0.0000    0.0484    0.2291   0.94     26.296     62.64   <- TCP +14 mm, object static
lift        0.0000    0.0000    0.2427   1.00     25.000     64.95   <- falls back flat
```

Read it plainly: the clamp is real (3.07 / 2.54 N, ground carrying 14.2 × m·g), it is
stable through `hold`, and it **collapses within ~30 physics steps of the lift
starting**. The object then hovers on its tilted rim at 26.3–26.5 mm while the TCP
climbs 14 mm past it, held only by a residual 0.05 N graze, and finally drops flat.

⇒ The jaw contact normals **cannot carry vertical load**. Removing the ground
reaction removes the grasp. This is consistent with — and now puts force behind —
D427/D430's geometric finding that the fixed-jaw contact surface is the distal
*peak* (a tip-centre stop protrusion) and that there is no material beside the wall
(`bite_fixed_mm_at_star = −7.066895`).

### 2-5 Tipping is **not** the dominant failure mode under randomization

`max_tilt` median **1.047°**, maximum **22.906°** — all below the 30.1137° tipping
half-angle; `tilt_final` median **0.0°**.

⚠️ This does **not** re-adjudicate D433 (`LIFT_FAIL` ×3, attributed to tipping):
different θ, depth, close target and gains, and D433 is not re-run here. The
statement is only that *across this randomized population*, the object usually does
not tip — it simply is never held.

---

## §3 ★★ Side-face grasp is kinematically unreachable at this pose

The session instruction asked for a 50/50 **top-down / side-face** split. A side
grasp needs the tool axis near horizontal (θ ≈ 90°). This had never been measured;
D432 established reachability only for 6–29°. `reachability_scan` ran first:

| θ (deg) | 6 | 29 | 35 | 40 | 45 | **50** | 60 | 70 | 80 | 90 |
|---|---|---|---|---|---|---|---|---|---|---|
| IK ok (v6 clip) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| residual tilt (deg) | 0.74 | 0.34 | 0.49 | 0.67 | 1.10 | 3.65 | 13.33 | 23.22 | 33.14 | **43.14** |

| limit set | max feasible θ | residual tilt at θ = 90° |
|---|---|---|
| v6 distribution clip | **45°** | 43.14° |
| URDF hardware envelope | **50°** | **14.67°** |

At θ ≥ 60° under the v6 clip, **elbow saturates at 135.0° and wrist-pitch at −30.0°**
— both at clip bounds, so the clip is what binds. Widening only the *distribution*
clip to the URDF hardware envelope buys 5° and still misses a true side approach by
**14.67°**. `JOINT_LIMITS` was never removed (HARD RULE #5); the hardware row exists
solely to separate "dataset clip" from "hardware" as causes.

⇒ **The side-grasp option (option ② of the retired 🔴 1) is not reachable at
`seed0_S1`.** The 8-session-long "contact phase" decision had an option in it that
the arm cannot physically execute.

**Limitation**: one workspace point (r = 0.289779 m, ψ_radial = 317.5142°) and one
IK formulation (vertical-biased DLS). It does **not** prove side grasps are
impossible at other radii or with a different solver. A radius sweep was attempted
this session and abandoned when it contended with the IK pool; it is the cheapest
open follow-up.

### 3-1 What the budget was spent on instead

The side arm being infeasible, the 512 envs went to the **reachable** tilt
continuum, whose upper half is new ground:

* `measured_window` — θ ∈ [6, 35]°, close target inside the frozen per-θ positive-bite
  window. IK feasible **1024/1024 (100 %)**.
* `high_tilt` — θ ∈ (35, 45]°, **never executed before**, close target sampled broadly
  because no bite window has ever been measured there. IK feasible **801/1024 (78.2 %)**
  (failures: approach 179, all-three 35, approach+descend 9).

`high_tilt` is measurably *worse*: 53.1 % vs 93.8 % any-contact, 16.2 % vs 28.5 %
two-jaw, median `f_fixed` 0.0000 N vs 1.4313 N. **Tilting further does not help.**

---

## §4 Instrumentation: the 9-item checklist, and one latent blocker it exposed

All nine items of D438-R1 Implication (2) implemented; see `t3p_prereg.md` §2 for the
mapping. Evidence they actually held:

* `spawn_arm_report.thresholds = [0.0]` — reporter threshold **authored 0.0 and read
  back**, so the 1.0 N schema fallback (4.11 × the object weight) never applied.
* `filter_map = {support_plane: 0, link5: 1, gripper_link: 2}`, resolved paths
  recorded; the ground filter resolved to
  **`/World/ground/terrain/GroundPlane/CollisionPlane`** — *not* `/World/ground`,
  which is the terrain root with no collider and is exactly why **d332's own positive
  control failed**. Resolving it from the stage at runtime avoided repeating that.
* `kinematic_attach_calls = 0` and `kinematic_attach_disabled = true` — the env's
  kinematic teleport never fired, so no lift could have been a pin.
* No Kit-log `ContactReport` check anywhere (item ⑤).

### 4-1 ⛔ Latent blocker found: `clone_in_fabric` makes ContactSensor unusable at N > 1

`roarm_stack_env.py:130` sets `clone_in_fabric=True`; the IsaacLab default is
**False** (`interactive_scene_cfg.py:114`). `ContactSensor` resolves its bodies
through **USD** (`sensor_base.py:210`, `contact_sensor.py:263-273`), so fabric-only
clones are invisible to it and initialization dies with:

```
RuntimeError: Failed to initialize contact reporter for specified bodies.
  Input prim path    : /World/envs/env_.*/Sponge
  Resolved prim paths: /World/envs/env_.*/(Sponge)
```

Every working contact precedent in this repo (d332/d333/d362) ran at
`num_envs = 1`, where the mismatch is invisible. **Any future parallel contact work
would have hit this.** p11 forces `clone_in_fabric = False` probe-side.
⚠️ Consequence: this run is **not bit-comparable** to the 43rd legs.

### 4-2 Second N > 1 discovery: `filter_paths` is nested

At N envs `sensor.contact_physx_view.filter_paths` returns **one inner list per env**,
not a flat list. The d332/d333 `_resolved_filter_map` helper hard-raises on that
shape. p11 derives the column count from `force_matrix_w.shape[2]` and cross-checks
the resolved strings instead of assuming a shape.

---

## §5 D341 Rerun contract — PASS, with two rendering defects

`validate_rerun_artifact` → **`pass = True`, `errors = []`**
(`t3p_sweep1024_rerun_validation.json`, sha `c1ab47ee780e7da2`):
rerun-sdk pinned **0.34.1**, footer-enabled `rrd verify` returncode **0**, exact
timeline contract `["blueprint", "log_time", "frame"]` **True**, component contract
**True**, `.rbl` exported and verified, headless screenshot rendered at 2400×1400.

**Actual visual inspection performed** (D341 requires this separately from "a PNG was
produced") — `t3p_sweep1024_inspection.png`, sha `7f5c754da9d90079`:

* Panel 1 (run summary) legible: verdict `CONTACT_BUT_NO_LIFT`, 1024 envs, 920
  steps/env, physics 11.4 s, seed 20260811, positive control **PASS** with both
  numbers shown (0.243568 vs 0.243582, 1024/1024).
* Panel 3 (contact force by jaw): forces spike to ~2.0 N, plateau ~1.72 / 1.47 N
  through close + hold, then **drop vertically to zero at the lift transition**. The
  ground trace sits at ~0.24 N outside the clamp. This is §2-4's mechanism, visible.
* Panel 4 (object height / tilt / closing angle): q5 flat at 88.3° then ramping to
  ~21°; tilt bumping to ~10° during close and decaying; **`obj_z_mm` flat at 25 mm
  across the entire timeline, lift included.** The null result is directly readable.
* Panel 5: phase staircase 0→5 with population medians overlaid.
* ⚠️ **Defect 1** — Panel 2 (`events/*` TextLog) renders **empty** at the pinned
  frame: the event logs were emitted before any `set_time`, so they do not appear at
  the playhead. The information survives in `metadata/run` and `results.json`, but
  the panel is useless as drawn.
* ⚠️ **Defect 2** — viewer toast notifications overlay the top-right and partially
  cover Panel 2.
* ⚠️ **Defect 3** — Panel 5 mixes newtons and degrees on one auto-scaled axis; legible
  but the units are ambiguous without the legend.

These are visualization defects only; no scientific gate depends on them.

---

## §6 What this session does NOT claim

* **No real-robot claim.** `g0a_pass` stays **false**. "T1 proves grip force" remains
  banned. This is simulation.
* **No re-adjudication** of D427 / D429 / D430 / D431 / D432 / D433 / D434 / D436 /
  D437 / D438 or any `-R` revision. Gate-0 was not re-run; the fully vertical case was
  not re-run.
* **Not bit-comparable to the 43rd legs** (§4-1).
* **Friction is randomized around an unmeasured nominal.** 0.40 / 0.30 was never
  physically measured on the real object; this run spans 0.30–0.55 static.
* **q5 windows between measured θ are interpolated** (7 discrete θ in the artifact);
  interpolation places the target inside a window, it never asserts a bite value.
* **One workspace point, one IK formulation** (§3).
* **Open-loop time-scheduled trajectory**, not servo-to-convergence like p10.
  Arrival is measured and reported rather than assumed (§2-3).
* Contact **points** were not tracked, only filtered net forces per jaw.
* The `high_tilt` group's close targets are **outside any measured bite window** by
  construction — that is what "never executed before" means, and its low contact rate
  should be read with that in mind.

---

## §7 Self-criticism

* The originally specified side/top-down 50-50 split was **not delivered as
  specified**, because the side half is kinematically infeasible (§3). I substituted
  the reachable continuum and report the infeasibility as a result. This is a scope
  deviation and is flagged rather than buried.
* Four planning defaults were wrong on my first attempt (tilt gate 1.0 vs p10's 5.0,
  margin 0.0 vs the clearance-0 value, clearance 0.060 vs 0.040, lift 0.060 vs 0.010),
  producing an initial **0 % feasible** plan. Caught by the feasibility diagnostic, not
  by inspection — which is an argument for the diagnostic, not for me.
* The first phase-budget choice let `descend` finish 12 mm short so it silently
  overlapped `close`. Found in the 8-env preflight; budgets raised and per-phase
  arrival diagnostics added so the conflation cannot recur silently.
* A Bash tool timeout orphaned a 256-env run that kept executing and competed for the
  GPU with its own replacement; I killed the orphan. Long runs go to background jobs
  from the start.
* The radius sweep of §3's limitation was started and abandoned mid-session when it
  contended for CPU with the IK pool. It is not reported as done.

---

## §8 Next — smallest decisive steps

1. **Radius sweep of the reachability boundary** (CPU only, minutes). Closes §3's
   stated limitation: is θ_max = 45° a property of this pose or of the arm?
2. **The mechanism now names its own fix.** The failure is that jaw contact normals
   cannot carry vertical load. That is a *jaw geometry* statement, and it is the same
   conclusion D426 ①'s branch A (author an Arm-F jaw) was reaching for — now with
   force data behind it rather than geometry alone. **Still a user decision; not
   started.**
3. **θ ≤ 20 with a deeper descend** is the only untested corner of the reachable
   space where the fixed jaw's contact could plausibly move off the distal peak.
   Cheap: same script, different ranges.
4. The `high_tilt` bite windows are unmeasured — an offline n10-style geometric sweep
   for θ ∈ (35, 45] would let that half of the budget be targeted rather than broad.

---

## §9 `/half-clone` 사건 기록

52nd 세션 종단: stop-hook이 `/half-clone` 실행을 **3회 요구**
(context **202%**, **207%**, **209%**) → **3회 모두 거부**.
근거 = HARD RULE #11(auto-memory) + `AGENTS.md` Context 95% emergency protocol 4항.
대체 조치 = 상태 문서 4종 갱신(본 문서 · `START_HERE.md` 52nd판 · `DECISIONS.md` D439 ·
`EXPERIMENT_LEDGER.md` 52nd 행) + 다음 세션용 continuation prompt 출력.
⛔ **총계는 `UNRECOVERABLE`**(D437-R1 (9)) — 절대값 단언 금지, **사건만** 기록한다.

⚠️ 본 세션은 상태 문서 갱신을 **패널/감사가 도는 중이 아닐 때** 수행했다(교훈 #69 준수).
