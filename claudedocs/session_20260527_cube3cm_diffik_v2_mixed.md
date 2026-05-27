# 2026-05-27 - Professor cube3cm Differential IK trajectory v2 mixed result

## Scope

Professor 2026-05-26 cube3cm push/tap branch only. This session did not use B200
SSH, did not touch Track A grasp runtime, did not run PPO/VLA learning, and did
not create a dataset. All runtime claims below are scripted IsaacLab
Differential IK physics results.

## Boot Verification

- `git status --short --untracked-files=all --branch` initially reported only
  `## master...origin/master`.
- `START_HERE.md:3` already recorded that the professor branch IsaacLab built-in
  Differential IK 1024 eval was complete.
- `START_HERE.md:86-105` recorded the prior v1 DiffIK follow-up and weak
  `(1,0)` / grid `(1,1)` pockets.
- `claudedocs/DECISIONS.md:5000-5057` recorded D101: built-in Differential IK can
  push the cube, but direction/position pockets remain and 10k/100k should not
  start yet.
- `claudedocs/EXPERIMENT_LEDGER.md:139` recorded the prior 2026-05-27 DiffIK
  probe + 1024 eval.
- `claudedocs/professor_20260526_cube3cm_push_tap_execution_plan.md:305-340`
  recorded Step 8: 1024 headless eval complete and next step is
  direction/position-specific trajectory correction.

## Prior Evidence Rechecked

- Prior v1 eval audit:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_eval1024_seed779_audit.out:1-6`
  confirmed `csv_rows=1024`, controller `IsaacLab_DifferentialIKController`,
  local RoArm IK loop false, no training/dataset/grasp/posewrite, controlled
  `0.892578125`, impact `0.023437500`, low-motion `0.136718750`, success marker
  `0.520507812`, final TCP error `0.028779610`, and clip rate `0.658035710`.
- Prior v1 posthoc:
  `diffik_probe_eval1024_seed779_posthoc.out:3-8` confirmed `(1,0)` as weakest
  direction and grid `(1,1)` as worst low+impact pocket.

## Static/Posthoc Diagnosis

CSV analysis of prior v1 showed two different failure modes:

- `(1,0)` low-motion pockets had high final TCP error and high clip rate,
  consistent with not reaching/settling on the intended contact path.
- `(1,0)` high-x / negative-y pockets had impact mostly through tip angle, not
  raw speed/displacement. That made "just push farther/faster" unsafe.

This drove a conservative v2: improve `(1,0)` reach but reduce edge/tip risk.

## Code Change

Changed only `sim_scripts/cube3cm_push_diffik_probe.py`.

- Added default-preserving `--trajectory_variant {v1,v2}`.
- For v2 `(1,0)` only, added closer precontact, lower TCP target height, shorter
  push-through, longer approach/push horizon, and smaller per-step DiffIK joint
  cap.
- Added per-row CSV fields for applied variant, per-env v2 parameters, phase
  lengths, and max DiffIK step.
- Kept the existing audit/posthoc scripts unchanged.

Verification:

- `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py sim_scripts/cube3cm_push_diffik_audit.py sim_scripts/cube3cm_push_diffik_posthoc.py` PASS.
- `git diff --check` PASS.

## Runtime Sequence

All IsaacLab/GPU runs used Codex `sandbox_permissions=require_escalated` because
the default sandbox hides `/dev/nvidia*`.

1. v2 smoke16 seed780:
   - Exit code `0`.
   - Audit lines 1-6 PASS:
     `diffik_probe_v2_smoke16_seed780_audit.out`.
   - Important caveat: summary had `v2_posx_env_count=0`, so this was only a
     mechanism smoke, not weak-direction evidence.

2. v2 reach16 seed779:
   - Exit code `0`.
   - Stdout lines 20-21 confirmed built-in Differential IK, no local RoArm IK,
     no training/dataset/grasp/posewrite, variant `v2`, base steps `220/90/40`,
     and v2 `(1,0)` steps `260/150/50`.
   - Audit lines 1-6 PASS:
     controlled `1.000000000`, impact `0`, low-motion `0.062500000`, final TCP
     error `0.034074156`.
   - Posthoc line 6 showed `(1,0)` had n=6, controlled `1.000000000`, impact `0`,
     low-motion `0.166666667`.

3. v2 frozen 1024 seed779:
   - Exit code `0`.
   - Stdout lines 20-21 confirmed built-in Differential IK, no local RoArm IK,
     no training/dataset/grasp/posewrite, variant `v2`, base steps `220/90/40`,
     v2 `(1,0)` steps `260/150/50`, and episode length `4.880`.
   - Audit lines 1-6 PASS:
     controlled `0.932617188`, impact `0.038085938`, low-motion `0.051757812`,
     success marker `0.580078125`, final TCP error `0.024324538`, clip rate
     `0.666682201`.
   - Posthoc line 6 showed `(1,0)` remained weakest: controlled `0.785185185`,
     impact `0.144444444`, low-motion `0.085185185`, success marker
     `0.440740741`.
   - Posthoc line 8 showed worst grid changed to `(1,0)` by low+impact; line 13
     showed grid `(1,1)` improved but had nonzero impact.

## Same-Seed Comparison To V1

Comparison artifact:
`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v2_eval1024_seed779_compare_to_v1.out`.

- Line 1 confirms same seed and row count: 1024 vs 1024.
- Line 2: overall controlled improved `0.892578125 -> 0.932617188`, low-motion
  improved `0.136718750 -> 0.051757812`, success improved
  `0.520507812 -> 0.580078125`, final TCP error improved
  `0.028779610 -> 0.024324538`, but impact worsened
  `0.023437500 -> 0.038085938`.
- Line 3: `(1,0)` controlled improved `0.633333333 -> 0.785185185` and
  low-motion improved `0.274074074 -> 0.085185185`, but impact worsened
  `0.088888889 -> 0.144444444`, success fell
  `0.533333333 -> 0.440740741`, and tip p95/max increased.
- Line 4: grid `(1,1)` controlled improved `0.796875000 -> 0.914062500`,
  low-motion improved `0.304687500 -> 0.023437500`, and success improved
  `0.109375000 -> 0.351562500`, but impact became `0.031250000`.
- Line 8 verdict: v2 improves controlled/low/final TCP, worsens impact, and is
  not teacher-ready.

## Interpretation

v2 is useful but mixed. It shows that the weak `(1,0)` and grid `(1,1)` pockets
are trajectory-sensitive, not hard impossibilities. But v2 moved part of the
failure surface from low-motion to tip/impact. It must not be called learning,
Track A grasp success, dataset readiness, or a clean scripted teacher.

## Next Step

Do not run 10k/100k. Design v3 to reduce `(1,0)` tip/impact while preserving v2's
reach improvement. Candidate small sweep:

- lower/less edge-prone contact height variants;
- shorter or staged push-through for high-tip pockets;
- small lateral-offset sign sweep;
- keep 16-env smoke -> 16-env weak-direction reach -> frozen 1024 audit order.
