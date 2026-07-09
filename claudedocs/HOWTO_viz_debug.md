# HOWTO: `roarm_rl.viz_debug`

Purpose: keep geometry/pose/contact probes inspectable with target-vs-actual
frame markers and a single diagnostic PNG. This is a debug fixture, not a render
or dataset-generation path.

## Probe Usage

```python
from roarm_rl.viz_debug import draw_frames, frame_from_axes, snapshot

target = frame_from_axes(
    "target_tcp",
    [0.26, 0.042, 0.037883],
    x_axis=[0.0, -1.0, 0.0],
    z_axis=[1.0, 0.0, 0.0],
    role="target",
)
actual = {
    "name": "actual_tcp",
    "position": [0.2911, 0.0412, 0.0555],
    "axes": {"x": [0.076, -0.995, -0.070], "y": [-0.678, 0.0, -0.735], "z": [0.731, 0.104, -0.674]},
    "role": "actual",
}

draw_frames([target, actual])
snapshot(
    "claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/target_vs_actual.png",
    pairs=[target, actual],
    cube={"center": [0.30, 0.0, 0.037883], "size": 0.10},
    title="target vs actual TCP",
    prefer_viewport=False,
)
```

`draw_frames()` uses Isaac Lab `VisualizationMarkers` when an Isaac app is
running. `snapshot()` first tries a viewport capture if requested, then falls
back to a deterministic matplotlib frame plot when viewport capture is not
available.

Optional rerun logging:

```python
from roarm_rl.viz_debug import log_rerun

status = log_rerun("debug.rrd", frames=[target, actual], joint_state={"q_deg": [...]})
```

If `rerun-sdk` is not installed, this helper returns a failure status and the
PNG/marker path remains the required artifact.

## D322/D323 Probes

Both G0a probe scripts now expose an opt-in debug flag:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output \
  python sim_scripts/cube10cm_top_view_d323_grasp_g0a_frame_repair_probe.py \
  --viz_debug_snapshots
```

D322 has only target/actual TCP position evidence. D323 is the frame-contract
source of truth for link5, TCP, gripper link, and jaw-face orientation.

## Existing LeRobot Dataset Viewer

The local `lerobot` environment provides `lerobot-dataset-viz`:

```bash
conda run -n lerobot lerobot-dataset-viz \
  --repo-id roarm_cube10cm_top_view_d321_script_v2_low_mid \
  --root claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/render_lerobot_v1/lerobot_dataset
```

Use this for visual inspection of existing LeRobot datasets such as D321. Do
not use it as evidence for new grasp variables unless the active case explicitly
requires dataset playback.
