# P7 Branch B Sim/Lab Dataset Schema Note - 2026-05-20

## Scope

Track A P7/Branch B normalized 2cm cube grasp only.

This note answers the professor-facing dataset questions without claiming that
the cube grasp dataset already exists. No Isaac run, training, constraints,
SurfaceGripper, transport, release, or gate tuning was executed while writing
this note.

## Short Answer

The intended sim dataset should sample object pose variables:

- `x`: object center x position in robot/world frame.
- `y`: object center y position in robot/world frame.
- `z`: object center height/layer.
- `yaw`: object rotation around the vertical axis.

If the object footprint is generalized, the bottom-side dimensions should also
be explicit variables:

- `a`: object footprint length/width along object-frame x.
- `b`: object footprint length/width along object-frame y.
- `h`: object height.

For the current canonical cube, these are fixed:

- `object_size = (a, b, h) = (0.020, 0.020, 0.020)m`.

The current code already expresses the grasp target as:

```text
world_grasp = object_center + RotZ(yaw) @ (normalized_grasp * object_size)
```

This means `a,b,h` are not just comments. They directly change the world TCP
target whenever the object is not exactly the canonical `2cm x 2cm x 2cm` cube.

## Current Code Evidence

### Object Pose / Size Variables

- `sim_scripts/p7_branch_b_cube2cm_local_grasp_close_sweep_probe.py`
  - `--object_size_m`, `--object_xy`, `--yaw_deg`, `--normalized_grasp` are
    command-line variables.
  - `_build_plan()` computes object center from `x/y` and
    `TABLE_Z + object_size[2] / 2`.
  - `_build_plan_from_center()` computes `world_grasp` using `RotZ(yaw)` and
    `normalized_grasp * object_size`.

- `sim_scripts/p7_branch_b_cube2cm_normalized_grasp_static_probe.py`
  - Samples workspace `x/y` grid over source regions.
  - Samples yaw list with default `[0, 45, 90, 135]`.
  - Uses `object_size_m` and normalized grasp specs for static reachability
    audit.

### Isaac Lab State Available Internally

`roarm_rl/roarm_stack_env.py` defines a 28-dim policy observation:

```text
joint_pos[6]
joint_vel[6]
sponge_pos[3]
sponge_quat[4]
tcp_to_sponge[3]
target_pos_local[3]
sponge_to_target[3]
```

The env computes:

- actual joint position: `_robot.data.joint_pos`;
- actual joint velocity: `_robot.data.joint_vel`;
- object position: `_sponge.data.root_pos_w`;
- object orientation: `_sponge.data.root_quat_w`;
- TCP position from `link5` pose plus `_tcp_local`;
- relative vectors such as TCP-to-object and object-to-target.

This is available in sim/lab runtime, but it is not automatically saved as a
dataset unless a writer records it.

### Real/LeRobot Dataset Currently Saved

The real collection path currently records:

- RGB frame;
- depth `.npy`;
- follower/single-arm joint angles as `angles`;
- optional leader joint angles as `leader_angles`;
- optional SDK pose `[x_mm, y_mm, z_mm]` in raw metadata.

The current LeRobot conversion stores only:

- `observation.images.top`;
- `observation.state` = 6 joint angles;
- `action` = 6 joint angles.

It does not currently store TCP pose, object pose, torque, acceleration, or the
object-pose conditioning variables `(x,y,z,yaw,a,b,h)` as first-class dataset
features.

## Current Files / Paths Observed

### Local Workspace

Existing local dataset/demo directories:

```text
collected_data/
collected_data_v2_backup/
collected_data_v5/
collected_data_v6/
lerobot_dataset_v3/
lerobot_dataset_v4/
lerobot_dataset_v5/
lerobot_dataset_v6/
lerobot_dataset_stacking_v1/
lerobot_dataset_stacking_v2/
lerobot_dataset_stacking_v3/
sim_demos_v1/
sim_demos_v2/
sim_demos_v3/
```

Example current LeRobot layout:

```text
lerobot_dataset_v3/data/chunk-000/file-000.parquet
lerobot_dataset_v3/meta/info.json
lerobot_dataset_v3/meta/stats.json
lerobot_dataset_v3/meta/tasks.parquet
```

Example older procedural sim demo layout:

```text
sim_demos_v1/demo_0000_trajectory.csv
sim_demos_v1/demo_0000_anchors.csv
```

Those CSVs contain only 6 joint columns:

```text
base,shoulder,elbow,wrist_p,wrist_r,gripper
```

This older sim demo format is not sufficient for the new cube dataset question,
because it does not include object pose, object size, TCP, object state, or
runtime contact/telemetry fields.

### B200 Workspace

Checked B200 code root:

```text
/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code
```

Observed dataset/log-like directories there:

```text
./logs
./IsaacLab/scripts/demos
./lerobot/examples/dataset
./lerobot/examples/port_datasets
./lerobot/src/lerobot/datasets
./lerobot/tests/artifacts/datasets
./lerobot/tests/datasets
```

Only observed project log files under `logs`:

```text
logs/phase1Balpha/train_p6v14c.out
logs/phase1Balpha/train_p6v14c.err
```

Conclusion: the new P7/Branch B normalized cube sim/lab dataset is not currently
present as a B200 dataset directory. The latest B200 evidence for v4/v5 cube work
is in `/tmp` logs and generated USD/URDF assets, not in a dataset container
folder.

Important B200 runtime evidence paths:

```text
/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_collision_usd/
/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_collision_usd/
/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_close26_hold_lift_b200.out
/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_close26_hold_lift_b200.out
```

## Proposed Cube Sim/Lab Dataset Schema

The next cube dataset writer should separate per-episode condition metadata from
per-frame runtime data.

### Episode Metadata

One record per episode / generated trajectory:

```text
episode_id
seed
variant
object_size_m = [a, b, h]
object_pose_initial = [x, y, z, yaw]
normalized_grasp = [gx, gy, gz]
world_grasp
target_tcp_path = [approach_tcp, descend_tcp, close_tcp, hold_tcp, lift_tcp]
close_deg
asset_usd_path
asset_hash
success_verdict
failure_reason
```

Notes:

- `object_pose_initial` is the sampled condition.
- `object_size_m` is fixed to `[0.020,0.020,0.020]` for the canonical cube but
  should remain explicit so future `a,b,h` generalization is not a rewrite.
- `normalized_grasp` is object-frame and size-normalized.
- `world_grasp` and `target_tcp_path` are derived fields that should be saved for
  reproducibility and later audit.

### Frame Data

One row per sim/lab step:

```text
episode_id
frame_index
phase
sim_time_s
joint_pos_rad[6]
joint_vel_rad_s[6]
action_joint_target_rad[6]
actual_tcp_pos_m[3]
target_tcp_pos_m[3]
tcp_target_error_m
object_pos_m[3]
object_quat_wxyz[4]
object_linvel_m_s[3]
object_angvel_rad_s[3]
tcp_to_object_m[3]
object_to_target_m[3]
gripper_q_rad
grasped_marker
posewrite_calls_total
attach_calls_total
```

Optional diagnostic fields for the current cube grasp failure:

```text
moving_jaw_center_obj_m[3]
counter_jaw_center_obj_m[3]
moving_overlap_obj_m[3]
counter_overlap_obj_m[3]
moving_gap_obj_m[3]
counter_gap_obj_m[3]
moving_contact
counter_contact
one_sided_push
object_drift_m
object_speed_mps
tilt_deg
upright_z
```

### Torque / Acceleration

Do not claim these are currently saved.

- Torque/effort limits exist in sim config, but measured torques are not part of
  the current recorded dataset schema.
- Joint acceleration is not directly stored. If needed, it should be derived from
  joint velocity finite differences or explicitly recorded from the simulator if
  an authoritative API is chosen.

## Current Status / Next Step

The correct next step is not dataset generation yet.

The current v4/v5 B200 physics logs still fail close/latch. Before generating a
large dataset, the project needs a runtime jaw telemetry diagnostic to identify
where close-time geometry diverges from the static fixture assumptions.

Only after a canonical cube grasp primitive passes reached/hold/lift-follow
checks with no hidden pose-write artifact should the dataset writer be
implemented.
