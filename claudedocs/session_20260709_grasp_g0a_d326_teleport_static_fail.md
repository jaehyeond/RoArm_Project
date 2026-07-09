# Session 2026-07-09 — D326 G0a Execution Contract Diagnosis

Verdict: `D326_G0A_TELEPORT_STATIC_FAIL_STOP`

이번 case의 신규 변수: `[]`. D326 stayed inside G0a. It did not advance the
ladder, change the D325 pose family, tune 42mm/10mm/15deg/15mm gates, close the
gripper, spawn a cylinder, grasp, lift, train RL/PPO, or use B200/RoArm/VLA.

## Step 0 - Environment

Environment manifest:

- `claudedocs/env_manifest_isaaclab_d326.txt`

Runtime version/path table was written into:

- `claudedocs/runtime_logs/grasp_track/g0a_d326/g0a_d326_execution_contract_summary.json`

Key verified pins:

- `numpy 1.26.0`
- `psutil 5.9.8`
- `rerun 0.34.1`
- Isaac Sim packages `5.1.0.0`
- Kit runtime version `107.3.3+production.229672.69cbf6ad.gl`

`CLAUDE.md` now has the D326 IsaacLab environment package rule: after any
package install into `isaaclab`, dependency impact must be recorded and
`numpy==1.26.0`, `psutil==5.9.8` must be verified/restored.

## Step 1 - Teleport Static Check

Prompt prediction:

- If teleporting the offline IK joint solution satisfies the D325 four
  conditions, the IK target is valid and runtime execution is the blocker.
- If not, stop before any execution-contract repair.

Result: teleport static check failed.

| metric | value |
|---|---:|
| TCP error | `0.349mm` |
| commanded TCP error | `0.349mm` |
| jaw tangent error | `9.174deg` |
| fixed-jaw gap | `-0.151mm` |
| fixed-jaw penetration | `0.151mm` |
| contact point below top | `49.733mm` |
| cube displacement | `0.000mm` |
| max arm joint error | `0.000000039rad` |

Pass/fail:

- TCP position: PASS.
- Jaw tangent: PASS.
- Contact height: PASS.
- Cube displacement: PASS.
- Fixed-jaw gap/no-penetration: FAIL.

The D325/D324 `position_only_tangent_minus1` IK target is therefore not yet a
valid static G0a solution under the D325 no-penetration criterion. The failure is
small (`0.151mm`) but it is on a pre-registered hard gate. Per prompt, execution
contract repair was not allowed after this result.

## Step 2 - Rerun v2

`roarm_rl/viz_debug.py` was upgraded so `log_rerun(...)` can record:

- URDF model from `local_assets/roarm_m3/urdf/roarm_m3.urdf`.
- `actual_robot` and `commanded_robot` entities.
- per-step `step` timeline joint transforms.
- target/actual/fixed-jaw/cube/contact frames.
- blueprint with a 3D view rooted at `/`.
- optional `--live_viewer` support in the D326 script path.

Because Step 1 failed, there was no motion trial to log. A one-step static v2
RRD was still written for the stop state:

- `claudedocs/runtime_logs/grasp_track/g0a_d326/d326_teleport_static_v2.rrd`

RRD status:

- URDF actual model: PASS, `8` joints.
- URDF commanded model: PASS, `8` joints.
- blueprint: PASS.
- trace steps: `1`.

Headless Rerun open check passed and saved:

- `claudedocs/runtime_logs/grasp_track/g0a_d326/d326_teleport_static_v2_rerun_open.png`

This is not a full motion-trial v2 completion. It is a static RRD load PASS;
motion playback remains blocked until teleport static G0a passes.

## Step 3/4 - Not Run

The prompt required stopping if the teleport check failed. Therefore:

- No baseline commanded-vs-actual motion trial was run.
- No x3 approach/hold diagnostic was used as a repair.
- No lead-limit, step-extension, or effort/stiffness contract repair was applied.
- No 10-trial repaired runtime gate was run.

## Interpretation

D326 corrects the D325 interpretation. D325 showed the runtime did not reach the
low-side target, but D326 shows a deeper blocker: even direct teleport to the
offline IK solution violates the fixed-jaw no-penetration gate by `0.151mm`.

The next valid G0a action is not actuator/trajectory repair. It is to recheck the
static alignment geometry: fixed-jaw proxy, tangent-minus yaw error, and how the
42mm tangent offset maps to a nonzero jaw-axis yaw error. Do not tune runtime
execution until teleport-static D325 criteria pass.

## Artifacts

- Probe: `sim_scripts/cube10cm_top_view_d326_grasp_g0a_execution_contract_probe.py`
- Summary JSON: `claudedocs/runtime_logs/grasp_track/g0a_d326/g0a_d326_execution_contract_summary.json`
- Summary MD: `claudedocs/runtime_logs/grasp_track/g0a_d326/g0a_d326_execution_contract_summary.md`
- Teleport snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d326/d326_teleport_static_check.png`
- Static Rerun v2: `claudedocs/runtime_logs/grasp_track/g0a_d326/d326_teleport_static_v2.rrd`
- Rerun open screenshot: `claudedocs/runtime_logs/grasp_track/g0a_d326/d326_teleport_static_v2_rerun_open.png`

## Verification

- `python -m py_compile roarm_rl/viz_debug.py sim_scripts/cube10cm_top_view_d326_grasp_g0a_execution_contract_probe.py` PASS.
- Rerun headless open command completed with exit `0`.
- No B200/JHPark/SSH/pull/.ssh path was used.
