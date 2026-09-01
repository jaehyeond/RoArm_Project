# Gates: Isaac Sim PBD solid-particle feasibility for the pellet pile (track P4)

OWNS: `sim_pbd_pellet_probe.py`, `claudedocs/runtime_logs/pbd_probe/**`

Question being decided: can Isaac Sim 5.1's PBD solid particles carry the pellet
pile, so that pile, arm and depth camera live in ONE engine, or does the project
keep DEME as the granular engine?

Not a re-litigation of D457's conclusion but of its **premise**. D457 recorded
"particle state cannot be read back from the GPU pipeline, so no post-scoop
heightmap" as the decisive reason. The physical pipeline never reads particle
coordinates - it reads a depth camera. This track therefore observes the
simulated pile the same way, through a top-down depth render, and treats
particle coordinates as a cross-check only.

**THRESHOLDS BELOW WERE WRITTEN BEFORE THE SWEEP RESULTS WERE READ.** They are
pre-registered so a gate cannot be moved to fit the outcome. `--selftest` proves
the measurement can emit FAIL.

NOT MEASURED: no pellet has been procured, calipered or weighed. The 30 deg
repose target is a coordinator-chosen placeholder. PBD friction / damping /
adhesion are dimensionless PhysX solver coefficients and are never a Coulomb mu,
a rolling resistance or a surface energy. Nothing here may be cited as a
property of real polypropylene.

Interpreter: `~/miniconda3/envs/isaaclab/bin/python` - isaacsim 5.1.0.0,
isaaclab 2.3.0, warp-lang 1.11.1, numpy 1.26.0, psutil 5.9.8 (D326 pins hold;
this track installs nothing). `newton` is NOT installed, so Newton MPM is out of
scope for this decision.

DEME comparison basis (`claudedocs/runtime_logs/pellet_model/repose_sweep.json`,
27/27 cells, same `sidewall_regression` definition):
sphere 8.90-25.64 deg, clump2 22.45-37.56 deg, clump3 20.03-41.77 deg.
PBD has only isotropic spheres, no rolling or torsional friction, so the sphere
row is the fair comparison and 25.64 deg is the DEME sphere ceiling.

---

## P0 - the harness can produce a FAIL

- [x] P0: the measurement recovers known angles and rejects non-heaps
  CHECK: `~/miniconda3/envs/isaaclab/bin/python sim_pbd_pellet_probe.py --selftest`
  EXPECT: SELFTEST_OK
  EVIDENCE: exit=0; EXPECT=matched; 5 cones recovered to <0.15 deg; pancake+cliff (52.49 deg, r2=1.000) REJECTED by heap_is_conical; output-sha256=540002661dea65fda405d441df3a04dcd628b28ea0e0c4f2f32513e6f09c8c92

- [x] P1: the format contract states the definition, the geometry and the non-claims
  CHECK: `~/miniconda3/envs/isaaclab/bin/python sim_pbd_pellet_probe.py --describe`
  EXPECT: PBD_PROBE_FORMAT_OK
  EVIDENCE: exit=0; EXPECT=matched; output-sha256=d495e849f3b21b46c3151574a190187b31225bdb941477c3f98cf222b82345e9

- [x] P2: no other track's files were touched
  CHECK: `git diff --exit-code -- sim_pellet_model.py sim_deme_pile.py sim_deme_scoop.py roarm_rl/heightmap.py GATES.md START_HERE.md claudedocs/DECISIONS.md claudedocs/DECISIONS_ACTIVE.md claudedocs/EXPERIMENT_LEDGER.md claudedocs/LEDGER_RECENT.md claudedocs/relay claudedocs/runtime_logs/pellet_model && printf 'P4_PROTECTED_FILES_OK\n'`
  EXPECT: P4_PROTECTED_FILES_OK
  EVIDENCE: exit=0; EXPECT=matched; base rev 5698fe1; only new files added (sim_pbd_pellet_probe.py, sim_pbd_pellet_report.py, claudedocs/runtime_logs/pbd_probe/)

---

## Decision gates

### G1 - angle of repose

PASS if at least one PBD parameter set produces a settled heap whose
`sidewall_regression` angle is within **30 +/- 3 deg** with `measurement_pass`
true (fit r^2 >= 0.90, toe inside 0.9 * half extent, >= 90% of the material
inside 1.15 * toe radius, quadrant spread <= 15 deg), reproduced on **>= 3 pour
seeds** with a standard deviation <= 2 deg.

FAIL if the parameter sweep (friction, then damping, then adhesion, one variable
at a time) cannot reach 27 deg at all, or reaches it only with a heap that fails
`measurement_pass`.

blind_spot: 30 deg is a placeholder, not a measurement, so PASS means "PBD can be
dialled to an arbitrary target near 30 deg", NOT "PBD matches polypropylene".
Matching one scalar also says nothing about the flow rule - two materials with
the same repose angle can respond completely differently to a scoop. And the
angle is measured on a free-poured heap; a scooped face is a different boundary
condition.

### G2 - settling

PASS if, for the G1 parameter set, the pile reaches the depth-observable settle
criterion (per-probe p99 cell drift <= 0.10 particle diameters, max cell drift
<= 1.0 diameters, apex drift <= 0.10 diameters, toe-radius drift <= 0.25
diameters, sustained 1.0 s) AND then, over a further **10 s of unloaded
watching**, drifts by <= **1.0 deg** in angle, <= 0.25 diameters in toe radius
and <= 0.10 diameters in apex height.

FAIL if the pile never meets the criterion inside 20 s, or keeps spreading during
the watch window.

blind_spot: 10 s of simulated time. A creep slower than ~0.1 mm/s is invisible
here and would still ruin a minutes-long scooping episode. A statically quiet
pile can also be metastable - this gate does not perturb it.

### G3 - scoop entry

PASS if driving the kinematic scoop through the settled pile keeps
`max particle speed during entry <= 10 x the scoop tip speed`, launches nothing
above `pile apex + 2 particle diameters` outside the bucket, and grows the pile's
radial extent by <= 2 particle diameters.

FAIL on any of: a speed spike above 10x tool speed, particles thrown above the
height bound, particles leaving the measurement domain, or a NaN/solver blow-up.

blind_spot: one scoop geometry, one entry point, one speed, one pile. A slow
plunge is the easy case; a fast or rotating grab is not covered.

### G4 - time step

PASS if the pile is stable and the angle is converged at a **practical** step:
`|angle(60 Hz) - angle(240 Hz)| <= 2 deg`, with all three of 60/120/240 Hz
producing `measurement_pass` true. The largest stable step is recorded.

FAIL if 60 Hz is unstable or off the 240 Hz answer by more than 2 deg AND the
step has to go below 1/480 s to converge.

blind_spot: convergence of a STATIC settled heap says nothing about the step a
moving scoop needs; G3 is run at the G4 step but is not itself a convergence
study.

### G5 - scooped-amount repeatability

PASS if, across **5 independent pours (different seeds), same fixed scoop
trajectory**, the coefficient of variation of the depth-measured removed volume
is <= **10%**.

FAIL above 15%. Between 10% and 15% is reported as MARGINAL and does not pass.

DEME reference for context only (different engine, different particle shape):
lip-equivalent reaction coefficient of variation 1.012, particles captured
191-204.

blind_spot: the trajectory is fixed and the pile differs only through the pour
seed, so this measures pour-to-pour scatter, not sensitivity to scoop placement,
which is the variable the proposal actually wants to learn.

---

## Verdict rule

Isaac PBD replaces DEME only if **G1..G5 all PASS**. Any single FAIL keeps DEME,
and the failing numbers are recorded as the reason.

---

## Outcome (recorded 2026-09-01, thresholds unchanged from the pre-registration above)

Machine verdict: `claudedocs/runtime_logs/pbd_probe/gates_pbd_probe.json`
(`sim_pbd_pellet_report.py --gates`). Narrative + evidence: `REPORT_pbd_probe.md`.

- [x] G1 angle of repose -> **FAIL**. 25 cells evaluated, **0** were both settled and
  measurement-valid; 0 landed in 30 +/- 3 deg with a reproducible parameter family.
- [x] G2 settling -> **FAIL**. 5 of 35 cells met the settle criterion and all 5 are
  degenerate (4 one-grain monolayers under adhesion, 1 solver blow-up at 255 iterations);
  **0** survived the 10 s creep watch. Ordinary cells sit at 35.9-43.1 mm/s rms for the
  whole 22 s window, 24-29x P3's 1.5 mm/s settle gate.
- [x] G3 scoop entry -> **FAIL** on the height bound only (29.9-32.0 mm observed vs a
  21.6-22.3 mm limit). Speed ratio 1.59-2.08 (limit 10) and radial growth 0.0-0.80 mm
  (limit 8.38 mm) both pass comfortably, so this is a bow wave in front of a plough, not a
  solver explosion - the threshold was set tighter than a ploughing tool warrants. Recorded
  as FAIL because thresholds are not moved after the fact; read with that caveat, and note
  the input pile had already failed G2.
- [x] G4 time step -> **FAIL**. 60/120/240/480/960/1920 Hz give 8.74 / 7.13 / 11.02 /
  16.58 / 20.76 / 32.27 deg: monotone, +11.51 deg on the last halving, not converged at
  1/1920 s. `|angle(60) - angle(240)| = 2.28 deg` exceeds the 2 deg bound. All six rates
  failed to settle.
- [x] G5 scooped amount -> **FAIL (NOT EVALUABLE)**. All 5 trials caught **0** particles
  and the net removed volume is **-11.69 cm^3** (the pile grew). The 1.71% CoV of the
  depth-measured removed volume is the drift of an unsettled pile between two frames, not
  the scatter of a scooped amount, so it is reported and not counted as a pass.

**VERDICT: `KEEP_DEME`.**

Correction to the record: D457's stated reason ("particle state cannot be read back from
the GPU pipeline") is FALSE. Readback works, and the depth-render path that makes readback
unnecessary also works and agrees with the analytic sphere-surface operator to 0.60-1.35 mm
rms. The conclusion survives; the reason must be replaced with the G1/G2/G4 numbers above.
