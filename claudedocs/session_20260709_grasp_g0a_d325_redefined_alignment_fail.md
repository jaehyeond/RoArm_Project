# Session 2026-07-09 — D325 G0a Redefined Alignment Criterion

Verdict: `D325_G0A_REDEFINED_ALIGNMENT_FAIL`

이번 case의 신규 변수: `[]`. D325 is a G0a criterion repair based on D323
kinematics and D324 visual evidence. It does not advance the ladder, change
object/friction, close the gripper, grasp, lift, train RL/PPO, run large
renders, or start G0b.

## Step 0 - Environment

User-approved `rerun-sdk` installation was attempted:

```bash
conda run -n isaaclab pip install rerun-sdk
```

Important side effect: the latest `rerun-sdk 0.34.1` pulled `numpy 2.4.6` and
`psutil 7.2.2`, which are incompatible with Isaac Lab / Isaac Sim. The env was
immediately repaired:

```bash
conda run -n isaaclab pip install numpy==1.26.0 psutil==5.9.8 --force-reinstall
```

Final import check passed:

- `numpy 1.26.0`
- `psutil 5.9.8`
- `rerun 0.34.1`

`.gitignore` was updated so only the small decision-evidence PNGs are trackable:

- D324 strict/position-only PNGs.
- D325 trial `01/05/10` snapshot PNGs.

The existing large-render/dataset ignore rules remain in force.

## Step 1 - Redefined G0a Criterion

Adopted pose family:

- Source: D324 `position_only_tangent_minus1`.
- `link5 +z` tool axis: free; the old horizontal-radial constraint is discarded.
- `link5 +x` jaw-separation axis: tangent `-1`, horizontal tangent error
  `<=15deg`.
- TCP target keeps the D323 contract: 42mm tangent offset and 10mm radial tip
  depth. No offset tuning was done.

Pre-registered runtime criteria:

1. TCP position error `<=5mm`.
2. Jaw-separation axis horizontal tangent error `<=15deg`.
3. Fixed-jaw face to cube side: horizontal gap `<=5mm`, no penetration, and
   contact point at least `15mm` below cube top.
4. Cube XY displacement `<5mm`; all 10 trials must satisfy 1-3.

Structural precheck: the adopted family itself is not height-gate impossible.
Offline contact point clearance below cube top was `49.733mm`, above the
required `15mm`.

## Step 2 - Probe

Added:

`sim_scripts/cube10cm_top_view_d325_grasp_g0a_redefined_alignment_probe.py`

This script reuses D323's verified TCP/IK helpers but leaves the historical D323
default behavior intact, preserving old evidence paths.

Runtime command:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output \
  python sim_scripts/cube10cm_top_view_d325_grasp_g0a_redefined_alignment_probe.py
```

Output folder:

`claudedocs/runtime_logs/grasp_track/g0a_d325/`

Visualization DoD status:

- Isaac `VisualizationMarkers`: PASS, prim path `/World/D325G0aFrames/frames`.
- PNG snapshots: PASS for trials 1, 5, and 10.
- Rerun `.rrd`: PASS,
  `claudedocs/runtime_logs/grasp_track/g0a_d325/d325_trial_01_frames.rrd`.

## Step 3 - Runtime Verdict

Summary:

| metric | mean | min | max |
|---|---:|---:|---:|
| TCP position error mm | 58.096 | 56.492 | 60.673 |
| jaw tangent error deg | 10.765 | 9.941 | 11.460 |
| fixed-jaw gap mm | 11.996 | 10.087 | 12.742 |
| penetration mm | 0.000 | 0.000 | 0.000 |
| contact point below top mm | 1.175 | -1.591 | 1.921 |
| cube displacement mm | 0.026 | 0.017 | 0.044 |
| max arm joint tracking error rad | 0.127 | 0.116 | 0.137 |

Failure counts:

| condition | failures |
|---|---:|
| TCP pose | 10/10 |
| jaw tangent | 0/10 |
| fixed-jaw gap | 10/10 |
| penetration | 0/10 |
| contact height | 10/10 |
| cube displacement | 0/10 |

Trial table and machine-readable rows:

- `claudedocs/runtime_logs/grasp_track/g0a_d325/g0a_d325_alignment_summary.md`
- `claudedocs/runtime_logs/grasp_track/g0a_d325/g0a_d325_alignment_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d325/g0a_d325_alignment_trials.csv`

Snapshots:

- `claudedocs/runtime_logs/grasp_track/g0a_d325/d325_trial_01_snapshot.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d325/d325_trial_05_snapshot.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d325/d325_trial_10_snapshot.png`

## Interpretation

D325 did not complete G0a.

The adopted yaw/jaw family is viable under the new tangent criterion:
`jaw_tangent` passed `10/10`. The failure is elsewhere: the runtime motion did
not bring the TCP to the low side target. Actual TCP stayed high, around the
cube top region, creating:

- `56-61mm` TCP position error.
- `10-13mm` fixed-jaw side gap.
- contact point only `~1mm` below the cube top, failing the `15mm` edge-avoidance
  gate.
- joint tracking error around `0.12-0.14rad`, exceeding the D323 actuator
  diagnostic threshold.

This means the next valid G0a work is not G0b and not more criterion tuning. It
is a reactive actuator/trajectory contract diagnosis for why the position-only
offline IK target is not reached in the Isaac runtime controller.

## Next

Stop after this failure. Do not tune `42mm`, `10mm`, `15deg`, or `15mm`, and do
not advance to G0b. The next G0a repair should compare commanded IK joint
targets against actual joints over time and decide whether the blocker is
actuator step clipping, target hold duration, drive stiffness/limits, or an env
action override contract mismatch.

## Verification

- `python -m py_compile sim_scripts/cube10cm_top_view_d325_grasp_g0a_redefined_alignment_probe.py roarm_rl/viz_debug.py sim_scripts/cube10cm_top_view_d323_grasp_g0a_frame_repair_probe.py` PASS before runtime.
- D325 runtime produced JSON/MD/CSV, three PNG snapshots, and one `.rrd`.
