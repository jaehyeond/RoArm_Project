# Session 2026-05-15 — G2-A scripted release bridge

## Boot / Verification

- Followed `CLAUDE.md` Current-State Protocol.
- Read `START_HERE.md`, `claudedocs/DECISIONS.md` D006-D012,
  `claudedocs/EXPERIMENT_LEDGER.md`, and
  `claudedocs/session_20260514_alpha_prime_delta_topdown.md` APPENDIX,
  `(δ.4)`, `(δ.5)`, `(G1/G2-A)`, `(G2-A v4)`, `(G2-A v5-v9)`.
- `git status --short` was dirty before coding:
  `START_HERE.md`, `claudedocs/DECISIONS.md`,
  `claudedocs/EXPERIMENT_LEDGER.md`,
  `claudedocs/session_20260514_alpha_prime_delta_topdown.md`,
  `launch_chain_topdown.sh`,
  `local_assets/roarm_m3/urdf/roarm_m3.urdf`,
  `roarm_rl/chain_skills.py`, plus untracked
  `local_assets/roarm_m3/urdf/meshes/gripper_link_collision_g2a.stl`.

## Pre-Code md5 Verification

Local expected baseline matched:

- `roarm_rl/chain_skills.py` = `f9a935cbcd7102f7bc65560f231924de`
- `launch_chain_topdown.sh` = `6013cafdd140d3d3dbdbebe1efc9f67e`
- `local_assets/roarm_m3/urdf/roarm_m3.urdf` = `cb5ce1232fd3a4f5e8ee6c456577a215`
- `local_assets/roarm_m3/urdf/meshes/gripper_link_collision_g2a.stl` =
  `02115511bbea2abb82814c6329ec9cea`

B200 expected baseline matched:

- code `roarm_rl/chain_skills.py` = `f9a935cbcd7102f7bc65560f231924de`
- launcher `launch_chain_topdown.sh` = `6013cafdd140d3d3dbdbebe1efc9f67e`
- URDF = `cb5ce1232fd3a4f5e8ee6c456577a215`
- G2-A STL = `02115511bbea2abb82814c6329ec9cea`
- `assets/roarm_m3/usd/roarm_m3.usd` =
  `4497024d25abab11de5c50e144124553`
- `assets/roarm_m3/usd/configuration/roarm_m3_physics.usd` =
  `5a4eb57ade18d2a4fd0676b43ac9dd12`
- `assets/roarm_m3/usd/.asset_hash` =
  `b57d9fe1ac60f5a4f0562f4437783666`

Note: `.asset_hash` is under `assets/roarm_m3/usd/.asset_hash`, not directly
under `assets/roarm_m3/.asset_hash`.

## Prior B200 Log Verification

All requested logs existed:

- `/tmp/chain_topdown_g2a_v4.{out,err}`
- `/tmp/chain_topdown_g2a_v5_skill2diag.{out,err}`
- `/tmp/chain_topdown_g2a_v6_holdwrist.{out,err}`
- `/tmp/chain_topdown_g2a_v7_holdwrist_fullerr.{out,err}`
- `/tmp/chain_topdown_g2a_v8_holdwrist_latchgrip.{out,err}`
- `/tmp/chain_topdown_g2a_v9_holdwrist_latchgrip_armbreak.{out,err}`

Important verified B200 lines:

- v4: `GUARD-OK` line 6; Skill 1b `(14,8,8)` and
  `stall_signature=FALSE_at_b3` line 129; latch line 134; Skill 2 failure
  `steps=120`, `max_arm_err=66.99deg`, `tcp_err=481.7mm`,
  `sponge_z=378.9mm` line 139; final `CHAIN_FINAL_SUCCESS=NO` line 145.
- v5: Skill 2 pre line 139 shows `arm_err=+90.00deg` from the old wrist target.
- v6: line 141 shows stable short Skill 2 (`steps=1`, `tcp_err=7.9mm`,
  `sponge_z=40.1mm`), but line 147 final is `NO`.
- v7: line 259/260 shows attached convergence runaway
  `max_arm_err=261.79deg`, `tcp_err=450.5mm`; final `NO` line 268.
- v8: line 259/260 shows attached convergence runaway
  `max_arm_err=272.81deg`, `tcp_err=408.5mm`; final `NO` line 268.
- v9: line 141 shows stable short handoff (`steps=1`, `tcp_err=8.0mm`,
  `sponge_z=40.1mm`), but P6v14a Skill 3 fails line 144
  (`d_xy=508.1mm`, `d_z=365.0mm`) and final line 147
  (`final_d_xy=704.8mm`, `CHAIN_FINAL_SUCCESS=NO`).

## Release Bridge Decision

Compared two options:

- A. Train a new release primitive from the stable G2-A handoff distribution
  (`wrist_r +90°`, gripper about `23-26°`, sponge below TCP after short
  attached move). Slower, but distribution-correct and cleaner for paper-quality
  autonomy.
- B. Replace Skill 3 with a minimal scripted/physics release bridge. Faster
  diagnostic, but risks brittle scripted physics if it grows variants.

Chose B only as a quick diagnostic: open gripper below `grasp_gripper_thresh`,
stop attaching immediately once `_grasped` clears, hold/minimize robot motion,
let sponge settle, and measure.

## Code Change

Changed `roarm_rl/chain_skills.py` Skill 3:

- Removed the full-chain P6v14a Skill 3 inference path for this diagnostic run.
- Added scripted release bridge:
  - start from v9 stable Skill 2 handoff,
  - freeze current arm/wrist target,
  - set gripper target to `GRIPPER_OPEN_DEG=0`,
  - detect `release_step` when `_grasped=False` and gripper is below threshold,
  - keep settling for 40 steps,
  - log TCP/sponge/target positions and `d_xy`, `d_z`.

Updated `launch_chain_topdown.sh` expected chain md5.

Post-change local md5:

- `roarm_rl/chain_skills.py` = `4bf308b8c0026671772ca3503f4f5387`
- `launch_chain_topdown.sh` = `a2c2f063ab5a5ddb3725c3cef0422714`

Checks:

- `python -m py_compile roarm_rl/chain_skills.py` passed.
- `python roarm_rl/chain_skills.py --dry-run` passed.

Synced to B200 and verified same md5s.

## B200 v10 Run

Run:

- `/tmp/chain_topdown_g2a_v10_scripted_release_bridge.out`
- `/tmp/chain_topdown_g2a_v10_scripted_release_bridge.err`

Log md5:

- out = `d6c6792c71081f87e051626b31084902`
- err = `f752cc8c100155c6cc6675837ff5080e`

Key stdout lines:

- line 6: `GUARD-OK chain_md5=4bf308b8c0026671772ca3503f4f5387`
- line 134: Skill 1c latch after step 15.
- line 135: close ended `gripper_q=23.02deg`, `d_sponge_tcp=23.6mm`,
  `grasped=True`.
- line 139: Skill 2 pre state `arm_err=+2.23deg`, `gripper_q=23.02deg`,
  `tcp_err=23.6mm`.
- line 140: Skill 2 post step 1 `arm_err=+1.10deg`,
  `gripper_q=23.56deg`, `tcp_err=8.0mm`, sponge z `40.1mm`.
- line 141: Skill 2 done `steps=1`, `tcp_err=8.0mm`, `grasped=True`,
  `sponge_z=40.1mm`.
- line 145: bridge pre state `d_xy=11.3mm`, `d_z=28.7mm`.
- line 146: `release_step=1`, `gripper_q=21.76deg`, `_grasped=False`.
- line 153: transient success at step 5: `d_xy=19.2mm`, `d_z=22.9mm`.
- line 158: terminated at step 41 (`release_step=1 + 40 settle`).
- line 159: post-release settled `d_xy=22.3mm`, `d_z=12.1mm`,
  `CHAIN_SETTLED=YES`.
- line 162: retreat final `final_d_xy=22.3mm`, `final_d_z=12.1mm`,
  `CHAIN_FINAL_SUCCESS=YES`.

stderr contained Isaac/NVML warnings only; no script failure.

## Verdict

PASS for the requested minimal release bridge diagnostic:

- Skill 1b no top-contact stall.
- Skill 1c latch succeeded without explosion.
- Skill 2 stable short handoff retained.
- Release succeeded from stable handoff.
- `CHAIN_FINAL_SUCCESS=YES`.

Critical caveat: this proves the stable G2-A handoff is physically
release-compatible under a minimal scripted release. It does not prove a learned
release primitive is solved. For paper-quality policy, train a new release
primitive from this stable handoff distribution.

## Next

1. Keep G2-A collision proxy.
2. Keep v10 minimal release semantics: open below threshold, let `_grasped`
   clear, no wrist rotation, no post-latch close/dwell.
3. Validate the v10 primitive on the next four-sponge source/target layout.
4. If the scripted bridge becomes layout-brittle, stop adding variants and train
   a release primitive from the stable G2-A handoff distribution.
