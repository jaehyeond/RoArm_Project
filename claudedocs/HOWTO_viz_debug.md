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

## Rerun requirement and role

Rerun is required whenever spatial or temporal evidence affects a verdict. It
is optional only for a pure file/hash/schema audit with no geometry, pose,
contact, trajectory, or synchronized-sensor judgment.

Use the modes this way:

| Work | Mode | Required Rerun subject |
|---|---|---|
| Pure hash/inventory/schema audit | May omit with written reason | None |
| Geometry/pose/frame gate | Save-only RRD | Actual compared geometry + target/actual frames |
| Cook/decomposition/representation | Save-only RRD | Source/instance/prototype/candidate meshes + Float64 deltas + events |
| Settle/contact/trajectory | Save-only or live+file | Full executed step timeline, object/tool states, scalars, contact points and force arrows |
| Interactive exploratory debug | Live+file | Same evidence as the eventual deterministic run |
| RGB/depth/real robot | Save-only or streaming+file | Synchronized image/depth/intrinsics/joints/actions/timestamps |
| Training | Sampled rollout RRD | Spatial rollout evidence; use the training tracker for optimizer-scale scalar history |

Save-only is the deterministic Isaac default. `live_viewer=False` does not mean
Rerun was skipped; it means the RRD must be inspected offline after the run. A Viewer
window is useful during exploration but is not a completion condition by itself.

Rerun is never the bit-exact authority. Original callback arrays and canonical
JSON/hashes decide equality. `Mesh3D`/`Points3D` are Float32 spatial copies for
inspection; `Scalars` preserve Float64 metrics. Do not read coordinates back
from Rerun and hash them into a scientific gate.

## Minimal recording

```python
from roarm_rl.viz_debug import log_rerun

status = log_rerun(
    "debug.rrd",
    frames=[target, actual],
    joint_state={"q_deg": [...]},
    recording_metadata={"case": "g0a_dNNN", "git_head": "..."},
)
assert status["ok"]                       # finalized RRD/RBL + footer/entity checks
assert status["archive_validation"]["pass"]
```

`log_rerun()` uses a dedicated `RecordingStream`. It attaches `FileSink` before
the first user log, flushes, exits the context to finalize the footer, writes a
fixed `.rbl`, and runs footer/entity/timeline validation. It refuses to
overwrite an existing RRD or RBL. The exact project pin is
`rerun-sdk==0.34.1`; a mismatch returns failure.

The same fixed Blueprint is embedded and made active in the RRD before user
data is logged, then exported as `.rbl`. The headless CLI screenshot is evidence
for the active embedded Blueprint. The external `.rbl` is separately verified
as a reproducible layout export; Rerun 0.34.1 does not guarantee that passing an
external RBL overrides an already-active embedded Blueprint.

The returned `ok` means the archive is complete. It deliberately leaves
`visual_inspection_complete=false` and `completion_contract_pass=false` until
the post-run screenshot has actually been reviewed and documented.

The collision-gate Blueprint uses eight independent spatial panels: four
source/instance/prototype/candidate views for link5 and the same four for the
gripper. Do not overlay variants and then claim that each was separately
visible. The registered headless default is `2400x1400`; a case may change it
only before runtime and must record the chosen display size.

## Scientific-subject schema

Cook geometry is passed separately so the Viewer can toggle each source:

```python
meshes = [
    {
        "entity_path": "cook/source/link5/parts/part_041",
        "vertices_m": authored_vertices_float64,
        "triangles": authored_triangles,
        "source_kind": "authored_x0",
        "coordinate_frame": "link5_body_local",
        "geometry_sha256": authoritative_hash,
        "color_rgba": [135, 135, 135, 80],
    },
    {
        "entity_path": "cook/instance/link5/parts/part_041",
        "vertices_m": instance_vertices_float64,
        "triangles": instance_triangles,
        "source_kind": "instance_x1",
        "coordinate_frame": "link5_body_local",
        "geometry_sha256": authoritative_hash,
        "color_rgba": [40, 120, 255, 100],
    },
]

status = log_rerun(
    "collision_gate.rrd",
    meshes=meshes,
    scalar_trace=[
        {
            "entity_path": "metrics/link5/part_041/max_coordinate_delta_m",
            "value": 0.0,
            "sequence": {"event_idx": 7, "part_idx": 5},
        }
    ],
    events=[
        {
            "entity_path": "events/cook",
            "text": "part_041 instance callback RESULT_VALID",
            "level": "INFO",
            "sequence": {"event_idx": 7, "part_idx": 5},
        }
    ],
    blueprint_mode="collision_gate",
    app_id="roarm_g0a_collision_gate",
)
```

Use these stable path families:

```text
cook/source/<body>/parts/<part>
cook/instance/<body>/parts/<part>
cook/prototype/<body>/parts/<part>
cook/candidate/<body>/parts/<part>
metrics/<body>/<part>/<metric>
events/<phase>
gate/<body>/<part>/<predicate>
contacts/points
contacts/forces
```

Geometry rows are static unless explicitly time-indexed. Cook callbacks use
`event_idx` and `part_idx`. Physics uses `sim_step` and, when available,
`sim_time`; the whole executed trajectory must be logged. Contact locations use
`Points3D`, force/normal vectors use `Arrows3D`, and numerical clearance,
displacement, tilt, impulse, and force use `Scalars`.

## Post-finalization completion gate

Run validation only after `log_rerun()` returns:

```python
from roarm_rl.rerun_contract import validate_rerun_artifact

validation = validate_rerun_artifact(
    "collision_gate.rrd",
    blueprint_path="collision_gate.rbl",
    screenshot_path="collision_gate_rerun_inspection.png",
    expected_entity_paths=[
        "cook/source/link5/parts/part_041",
        "cook/instance/link5/parts/part_041",
    ],
    expected_timeline_names=["event_idx", "part_idx"],
    exact_entity_paths=all_registered_non_system_paths,
    exact_timeline_names=["blueprint", "event_idx", "log_time", "part_idx"],
    expected_entity_components=required_components_by_path,
)
assert validation["pass"]
```

`expected_*` checks only prove that required rows are present. A completion
gate must additionally pass `exact_entity_paths`, `exact_timeline_names`, and
`expected_entity_components`; this rejects unexpected scientific entities,
timelines, and schema drift instead of silently accepting them.

Equivalent manual checks use the CLI inside the pinned environment:

```bash
conda run -n isaaclab rerun rrd verify --check-footers true collision_gate.rrd
conda run -n isaaclab rerun rrd stats collision_gate.rrd
conda run -n isaaclab rerun rrd print -v collision_gate.rrd
conda run -n isaaclab rerun --headless --window-size 2400x1400 \
  --screenshot-to collision_gate_rerun_inspection.png collision_gate.rrd
```

`verify` proves archive integrity, while the exact entity/timeline/component
contract catches semantic hierarchy errors such as the historical
`/\/actual_robot/...` URDF subtree.
The headless command proves renderability only. Open or inspect the resulting
PNG, record what was actually visible, and cite the path in the session doc
before using the word "inspected".

## Rerun v2 URDF Trace (D326~)

`log_rerun()` also supports URDF-backed actual/commanded robot traces:

```python
status = log_rerun(
    "debug_v2.rrd",
    frames=[target, actual],
    urdf_path="local_assets/roarm_m3/urdf/roarm_m3.urdf",
    cube={"center": [0.30, 0.0, 0.037883], "size": 0.10},
    joint_trace=[
        {
            "step": 0,
            "actual_joint_rad_by_name": {
                "base_link_to_link1": 0.0,
                "link1_to_link2": 0.0,
                "link2_to_link3": 1.57,
                "link3_to_link4": 0.0,
                "link4_to_link5": 0.0,
                "link5_to_gripper_link": 0.0,
            },
            "commanded_joint_rad_by_name": {...},
            "frames": [target, actual],
        },
    ],
)
```

The resulting `.rrd` contains `/actual_robot`, `/commanded_robot`, `/frames`,
and `/cube` entities. `UrdfTree.entity_path_prefix` must be `"actual_robot"`
or `"commanded_robot"` without a leading slash; a leading slash becomes a
literal escaped path segment. Frame origin points are logged at local zero
under their transform, avoiding a duplicated world translation.

D326 only validated a static one-step trace because its teleport-static gate
failed before a motion trial. A future physics verdict must log every executed
step, not copy that one-step exception.

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
