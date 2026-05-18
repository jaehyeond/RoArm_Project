# Session 2026-05-18 - P7 Branch B pre-close coverage audit

## Scope Guard

- Continued Track A P7/Branch B only.
- Read-only/log-only audit. No new Isaac run was launched.
- Did not train.
- Did not integrate constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute CLOSE->MOVE transport, transport target, release, or scripted
  release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.

## Boot / Cross-Checks

- Read `CLAUDE.md` Current-State Protocol first.
- Read `START_HERE.md`.
- Read `claudedocs/DECISIONS.md` D047-D049.
- Read latest Branch B rows 65-68 in `claudedocs/EXPERIMENT_LEDGER.md`.
- Read `claudedocs/session_20260518_p7_branch_b_preclose_clearance_strategy.md`.
- `git status --short` had no output before this audit.
- Required local md5s matched:
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_admissible_region_probe.py`
    `89ad48b6ebdec076d6f58e330a9131f9`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
    `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

## B200 Log Set Verified

Directly checked B200 `/tmp`, not local `/tmp`:

- `/tmp/p7_branch_b_roarm_chain_preclose_admissible_region_b200.{out,err}`
- `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_robustness_{0p0,0p5,1p0,2p0,4p0,6p0}_b200.{out,err}`
- `/tmp/p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_{0p1,0p2,0p3,0p4,0p5}_b200.{out,err}`
- `/tmp/p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_{neg0p5,neg1p0,neg1p5,neg2p0,neg3p0,neg4p0,neg6p0}_b200.{out,err}`

All listed files existed. Process check showed no matching
`isaaclab.sh/train_ppo/torchrun/rl_games/python .*p7_` process. Stderr scan
found no `Traceback` or `Exception`; selector stderr files contained only the
known cpufreq/NVML/Fabric warnings, and admissible wrapper stderr was empty.

## Coverage Audit Table

This table deliberately separates observed selector cleanliness from the
conservative admissible decision. The conservative rule is the non-deployed
wrapper rule from the B200 admissible stdout line 3:
`min_side_margin_m=0.002000`, `max_below_depth_m=-0.003000`, unchanged exact gate
reference `0.003000m`, reject below-top inside-footprint, reject zero-margin,
require final outside-AABB for below-top side-edge, and keep far-sponge below-top
as no-contact control.

| Case set | B200 evidence | Observed selector clean/fail | Conservative admissible decision | D047-D049 agreement | Clean but intentionally rejected |
|---|---|---|---|---|---|
| Side margin 0.0mm, depth -1.5mm | robustness `0p0`: line 42 gate/scope, line 52 selector `ACCEPT`, line 1055 final AABB inside + `mechanically_valid_target=NO` + clean `NO`, lines 1058-1059 diagnostic `NO` | FAIL | REJECT | Agrees with D047 boundary trap | NO |
| Side margins 0.1/0.2/0.3/0.4/0.5mm, depth -1.5mm | fine `0p1..0p5`: line 42 gate/scope, line 52 selector `ACCEPT`, line 1055 final outside-AABB + exact `YES` + clean `YES`, lines 1058-1059 diagnostic `YES` | CLEAN | REJECT because below conservative 2mm side margin | Agrees with D048: observed positive pass is not deployment/chain margin | YES |
| Side margin 1.0mm, depth -1.5mm | robustness `1p0`: line 52 selector `ACCEPT`, line 1055 exact `YES`, final outside-AABB, mechanically valid, clean `YES`, line 1059 diagnostic `YES` | CLEAN | REJECT because below conservative 2mm side margin | Consistent conservative subset of D047-D048 evidence | YES |
| Side margins 2/4/6mm, depth -1.5mm | robustness `2p0/4p0/6p0`: line 1055 exact `YES`, final outside-AABB, mechanically valid, clean `YES`; line 1059 diagnostic `YES` | CLEAN | ACCEPT | Agrees with D047 positive outside-AABB margin and D049 depth bound | NO |
| Depth -0.5/-1.0mm at 2mm side margin | depth `neg0p5/neg1p0`: line 52 top class tangent, line 1055 exact `YES`, clean `YES`, line 1059 diagnostic `YES` | CLEAN | ACCEPT | Agrees with D049 | NO |
| Depth -1.5/-2.0/-3.0mm at 2mm side margin | depth `neg1p5/neg2p0/neg3p0`: line 52 outside-AABB, line 1055 exact `YES`, clean `YES`, line 1059 diagnostic `YES`; -3mm final error `0.002409` | CLEAN | ACCEPT | Agrees with D049 clean through about -3mm | NO |
| Depth -4.0/-6.0mm at 2mm side margin | depth `neg4p0/neg6p0`: line 52 selector `ACCEPT`, line 1055 final outside-AABB and mechanically valid but exact `NO`, clean `NO`; lines 1058-1059 diagnostic `NO` | FAIL | REJECT | Agrees with D049 exact-convergence limit; not an inside-footprint clamp failure | NO |
| Top-tangent control | representative depth `neg3p0`: line 473 exact `YES`, top class tangent, mechanically valid, clean `YES` | CLEAN | ACCEPT | Agrees with D043-D049; outside-AABB below-top exception is not needed | NO |
| Above-top control | representative depth `neg3p0`: line 667 exact `YES`, top class above, mechanically valid, clean `YES` | CLEAN | ACCEPT | Agrees with D043-D049 | NO |
| Nominal below-top inside baseline | representative depth `neg3p0`: line 179 exact `NO`, top-clamped `YES`, final AABB inside, mechanically valid `NO`, clean `NO` | FAIL | REJECT | Agrees with D043-D045 and D047-D049 | NO |
| Far-sponge below-top control | representative depth `neg3p0`: line 279 exact `YES`, no top clamp, but no-contact/mechanically valid `NO`, clean `NO` | FAIL as contact candidate | REJECT | Agrees with D043-D049 no-contact separation | NO |

## Interpretation

- The compact admissible wrapper matrix does not conflict with D047-D049.
- The wrapper is a conservative subset/explanation of accumulated evidence, not
  an attempt to accept every observed clean side-edge case.
- The important distinction is:
  - `0.1-1.0mm` side margins are observed clean in this deterministic diagnostic,
    but intentionally rejected by the conservative `2mm` diagnostic rule.
  - `-4/-6mm` depth cases are not inside-footprint clamp failures; they remain
    outside-AABB and mechanically valid, but fail exact convergence under the
    unchanged `0.003000m` gate.
  - Exact convergence alone is insufficient for below-top inside-footprint and
    far-sponge controls.
  - Mechanical validity alone is insufficient for deeper side-edge targets.
- No audit gap was exposed. A new diagnostic matrix is not justified yet.

## Next Step

- Stay pre-integration.
- Do not wire the admissible rule into the RoArm chain.
- Do not proceed to CLOSE->MOVE transport, release, SurfaceGripper, constraints,
  or training.
- If work continues, use the audit as a read-only/log-level checklist when
  describing future pre-close evidence: always label observed selector clean/fail
  separately from conservative admissible accept/reject.

## Follow-Up: Diagnostic Interpretation Checklist

Use this checklist for future pre-close evidence reviews before proposing any
new diagnostic matrix or chain integration:

- Stay pre-integration: no training, no constraint integration, no
  SurfaceGripper attach, no CLOSE->MOVE transport, no transport target, no
  release, no scripted release variant, and no diagnostic gate tuning.
- Cite file lines and direct B200 `/tmp` log lines before making a state claim.
- Report `observed_selector_clean_fail` separately from
  `conservative_admissible_accept_reject`; do not collapse them into one pass/
  fail label.
- For every side-edge below-top case, report side margin, side top depth,
  selector line 52 decision/reason, final line 1055 AABB class, exact
  convergence, top clamp, mechanical validity, and clean realization.
- For clean-but-rejected cases, explicitly label them as intentionally rejected
  by the conservative non-deployed rule, not as selector failures.
- For exact-fail cases, distinguish exact-convergence failure from
  inside-footprint clamp failure. The -4/-6mm depth cases are outside-AABB and
  mechanically valid but fail the unchanged 0.003000m exact gate.
- Keep controls separate: top-tangent/above controls may be conservative
  accepts; nominal below-top inside-footprint and far-sponge no-contact controls
  remain conservative rejects.
- Require stderr/process hygiene in the same review: all requested B200 files
  exist, no matching P7/Isaac/training process is running, selector stderr has no
  traceback/exception, and the admissible wrapper stderr is empty.
- If this checklist finds no file/log contradiction, do not propose a new
  diagnostic matrix.
