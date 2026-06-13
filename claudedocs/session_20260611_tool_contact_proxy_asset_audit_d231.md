# 2026-06-11 Tool Contact Proxy Asset Audit D231

## Scope

- Local asset/code audit only.
- No IsaacLab runtime, GPU runtime, PPO, L2/Large PPO, dataset, VLA,
  action-teacher, RoArm, SSH/B200, pull, or Track A work.
- Goal: answer professor-feedback question: should cube10cm tap contact use
  `link5_collision_aabb`, direct `gripper_link` AABB/OBB, or a more explicit
  true tool-surface proxy?

## Evidence

Audit command:

```bash
python3 sim_scripts/cube10cm_tool_contact_proxy_asset_audit.py
```

Summary:

```text
line1 artifact=cube10cm_tool_contact_proxy_asset_audit_d231 local_asset_audit_only=YES isaaclab_runtime=NO gpu=NO training=NO robot_control=NO ssh=NO
line2 urdf_meshes link5_collision=/home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/urdf/meshes/link5.stl gripper_collision=/home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/urdf/meshes/gripper_link_collision_g2a.stl gripper_visual=/home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/urdf/meshes/gripper_link.stl
line3 usd_collision_policy config_exists=True collision_from_visuals=False collider_type=convex_hull pxr_stage_read=not_available_in_local_python
line4 native_bbox_size_m link5_collision=[0.046496015548706054, 0.035519998550415044, 0.12063513135910035] gripper_collision=[0.004, 0.004, 0.004] gripper_visual=[0.07785000038146972, 0.02523999881744385, 0.0393675656914711]
line5 link5_frame_q0 link5_collision_distal_z=0.119885620 gripper_collision_distal_z=0.054035007 gripper_visual_distal_z=0.119117587 hand_tcp_z=0.115428000
line6 gripper_collision_verdict max_size_m=0.004000029 tiny_proxy=True direct_gripper_collision_proxy_recommended=False
line7 option2_verdict option2_direction=YES_AS_TOOL_SURFACE_UNION_NOT_AS_GRIPPER_LINK_COLLISION_ONLY link5_current_runtime_proxy=True requires_env_metric_change_before_ppo=True
line8 promotion ppo_unblocked=False dataset_vla_roarm_unblocked=False
```

URDF/config evidence:

- `local_assets/roarm_m3/urdf/roarm_m3.urdf` defines:
  - `link5` collision as `meshes/link5.stl`.
  - `gripper_link` visual as `meshes/gripper_link.stl`.
  - `gripper_link` collision as `meshes/gripper_link_collision_g2a.stl`.
- `local_assets/roarm_m3/usd/config.yaml` has `collision_from_visuals: false`
  and `collider_type: convex_hull`, so the generated USD physics collision is
  expected to use the URDF collision mesh, not the gripper visual mesh.
- Local Python and `conda run -n isaaclab python` both lack `pxr`, so this
  session did not directly open the binary USD crate. The USD conclusion is
  therefore an inference from converter config plus URDF, not a stage traversal.

## Interpretation

- The cube itself has a proper physics collider; this audit is about the robot
  tool-side contact proxy used for reward/metrics.
- Directly replacing `link5_collision_aabb` with `gripper_link` collision is
  wrong under the current asset:
  - `gripper_link_collision_g2a.stl` is only about `4mm` across.
  - Its q0 distal z in link5 frame is only about `0.054035m`.
  - It is a diagnostic/tiny collision proxy, not the full fingertip/contact
    surface for cube tap.
- The full moving gripper geometry is closer to the real surface in the visual
  mesh:
  - `gripper_link.stl` visual bbox is about
    `[77.85mm, 25.24mm, 39.37mm]`.
  - At q0 its distal z in link5 frame is about `0.119118m`, close to link5
    collision distal z `0.119886m`.
  - However it is not the current physics collision asset because
    `collision_from_visuals=false`.
- The current `link5_collision_aabb` is not just an arbitrary wrist box. In this
  RoArm asset split, the fixed jaw/distal tool body lives in `link5`; the moving
  jaw lives in `gripper_link`. A professor-facing "gripper contact surface" must
  therefore be a tool-surface union, not `gripper_link` alone.

## Decision

- Option 2 is directionally right only if defined precisely:
  "use the true tool contact surface" means union of the fixed jaw/link5 distal
  surface and the moving gripper full geometry or properly authored collision
  surface.
- Option 2 is wrong if interpreted as "switch to `gripper_link_collision_g2a`
  AABB/OBB".
- Current runtime metric should not be silently switched to direct
  `gripper_link` collision.
- PPO is not unblocked by this audit. Before the next PPO/constant-baseline
  runtime, either:
  1. explicitly accept `link5_collision_aabb` as the fixed-jaw part of the tool
     surface for the current sim contract; or
  2. implement a named `tool_surface_union` contact metric and re-run a zero/base
     metric-equivalence sanity check.

## Files

- `sim_scripts/cube10cm_tool_contact_proxy_asset_audit.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tool_contact_proxy_asset_audit_d231_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tool_contact_proxy_asset_audit_d231.json`
- `local_assets/roarm_m3/urdf/roarm_m3.urdf`
- `local_assets/roarm_m3/usd/config.yaml`

## Verification

```bash
python3 -m py_compile sim_scripts/cube10cm_tool_contact_proxy_asset_audit.py
python3 sim_scripts/cube10cm_tool_contact_proxy_asset_audit.py
```
