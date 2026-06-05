"""IsaacLab built-in Differential IK probe for cube push/tap diagnostics.

This is a professor-branch diagnostic, separate from Track A grasp. It sends
end-effector targets near a configurable cube, uses IsaacLab's DifferentialIKController
and live PhysX Jacobians to compute joint targets, and lets physics decide the
cube motion. It is not training and not a learned-policy success claim; optional
trace capture is only raw material for a separate dataset builder/audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DISP_THRESHOLDS_M = (0.001, 0.005, 0.010, 0.020, 0.030)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--cube_size_m", type=float, nargs=3, default=(0.030, 0.030, 0.030))
    parser.add_argument("--cube_mass_kg", type=float, default=0.020)
    parser.add_argument("--cube_push_target_disp_m", type=float, default=None)
    parser.add_argument("--cube_success_disp_m", type=float, default=None)
    parser.add_argument("--gate_disp_m", type=float, default=0.010)
    parser.add_argument("--fixed_cube_x_m", type=float, default=None)
    parser.add_argument("--fixed_cube_y_m", type=float, default=None)
    parser.add_argument("--fixed_push_dir", type=float, nargs=2, default=None)
    parser.add_argument("--tcp_height_mode", choices=("top_margin", "side_center"), default="top_margin")
    parser.add_argument("--tcp_center_height_offset_m", type=float, default=0.0)
    parser.add_argument("--precontact_clearance_m", type=float, default=0.020)
    parser.add_argument("--tcp_top_margin_m", type=float, default=0.003)
    parser.add_argument("--push_through_m", type=float, default=0.030)
    parser.add_argument("--base_lateral_offset_m", type=float, default=0.0)
    parser.add_argument("--through_target_mode", choices=("far_face", "near_face"), default="far_face")
    parser.add_argument("--contact_controller_mode", choices=("open_loop", "measured_stop"), default="open_loop")
    parser.add_argument("--contact_stop_target_mode", choices=("retract", "freeze"), default="retract")
    parser.add_argument("--contact_detect_disp_m", type=float, default=0.001)
    parser.add_argument("--contact_stop_disp_m", type=float, default=None)
    parser.add_argument("--contact_overshoot_disp_m", type=float, default=0.020)
    parser.add_argument("--contact_near_tcp_cube_dist_m", type=float, default=0.085)
    parser.add_argument("--contact_near_joint_step_scale", type=float, default=0.50)
    parser.add_argument("--contact_stop_joint_step_scale", type=float, default=0.25)
    parser.add_argument("--contact_retract_clearance_m", type=float, default=0.020)
    parser.add_argument("--reaction_disp_m", type=float, default=0.001)
    parser.add_argument("--reaction_z_delta_m", type=float, default=0.002)
    parser.add_argument("--reaction_speed_mps", type=float, default=0.020)
    parser.add_argument("--approach_steps", type=int, default=55)
    parser.add_argument("--push_steps", type=int, default=35)
    parser.add_argument("--post_steps", type=int, default=30)
    parser.add_argument("--settle_steps", type=int, default=8)
    parser.add_argument("--max_diffik_joint_step_rad", type=float, default=0.012)
    parser.add_argument("--dls_lambda", type=float, default=0.010)
    parser.add_argument("--arm_stiffness_override", type=float, default=None)
    parser.add_argument("--arm_damping_override", type=float, default=None)
    parser.add_argument("--arm_effort_limit_sim_override", type=float, default=None)
    parser.add_argument("--arm_velocity_limit_sim_override", type=float, default=None)
    parser.add_argument("--trajectory_variant", choices=("v1", "v2", "v3", "v3_1"), default="v1")
    parser.add_argument("--v2_posx_precontact_clearance_m", type=float, default=0.012)
    parser.add_argument("--v2_posx_push_through_m", type=float, default=0.024)
    parser.add_argument("--v2_posx_tcp_top_margin_m", type=float, default=-0.004)
    parser.add_argument("--v2_posx_lateral_offset_m", type=float, default=0.0)
    parser.add_argument("--v2_posx_approach_steps", type=int, default=260)
    parser.add_argument("--v2_posx_push_steps", type=int, default=150)
    parser.add_argument("--v2_posx_post_steps", type=int, default=50)
    parser.add_argument("--v2_posx_max_diffik_joint_step_rad", type=float, default=0.028)
    parser.add_argument("--v3_posx_precontact_clearance_m", type=float, default=0.014)
    parser.add_argument("--v3_posx_push_through_m", type=float, default=0.020)
    parser.add_argument("--v3_posx_tcp_top_margin_m", type=float, default=-0.011)
    parser.add_argument("--v3_posx_lateral_offset_m", type=float, default=0.0)
    parser.add_argument("--v3_posx_approach_steps", type=int, default=300)
    parser.add_argument("--v3_posx_push_steps", type=int, default=220)
    parser.add_argument("--v3_posx_post_steps", type=int, default=60)
    parser.add_argument("--v3_posx_max_diffik_joint_step_rad", type=float, default=0.020)
    parser.add_argument("--v31_posx_precontact_clearance_m", type=float, default=0.014)
    parser.add_argument("--v31_posx_push_through_m", type=float, default=0.020)
    parser.add_argument("--v31_posx_tcp_top_margin_m", type=float, default=-0.011)
    parser.add_argument("--v31_posx_lateral_offset_m", type=float, default=0.0)
    parser.add_argument("--v31_posx_approach_steps", type=int, default=300)
    parser.add_argument("--v31_posx_push_steps", type=int, default=220)
    parser.add_argument("--v31_posx_post_steps", type=int, default=60)
    parser.add_argument("--v31_posx_max_diffik_joint_step_rad", type=float, default=0.020)
    parser.add_argument("--v31_lowx_threshold_m", type=float, default=0.240)
    parser.add_argument("--v31_lowx_precontact_clearance_m", type=float, default=0.020)
    parser.add_argument("--v31_lowx_push_through_m", type=float, default=0.030)
    parser.add_argument("--v31_lowx_tcp_top_margin_m", type=float, default=0.003)
    parser.add_argument("--v31_lowx_lateral_offset_m", type=float, default=0.0)
    parser.add_argument("--v31_lowx_approach_steps", type=int, default=300)
    parser.add_argument("--v31_lowx_push_steps", type=int, default=220)
    parser.add_argument("--v31_lowx_post_steps", type=int, default=60)
    parser.add_argument("--v31_lowx_max_diffik_joint_step_rad", type=float, default=0.020)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_env_id", type=int, default=0)
    parser.add_argument("--video_dir", type=str, default=str(LOG_DIR / "diffik_probe_video"))
    parser.add_argument("--video_name", type=str, default="diffik_probe.mp4")
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_stride", type=int, default=4)
    parser.add_argument("--video_eye_offset_m", type=float, nargs=3, default=(0.28, -0.42, 0.30))
    parser.add_argument("--video_target_push_m", type=float, default=0.025)
    parser.add_argument("--trace_env_id", type=int, default=-1)
    parser.add_argument("--trace_env_ids", type=int, nargs="*", default=None)
    parser.add_argument("--trace_all_envs", action="store_true")
    parser.add_argument("--trace_stride", type=int, default=4)
    parser.add_argument("--trace_csv", type=str, default="")
    parser.add_argument("--trace_diffik_diagnostics", action="store_true")
    parser.add_argument("--viewer_step_sleep_s", type=float, default=0.0)
    parser.add_argument("--post_run_sleep_s", type=float, default=0.0)
    parser.add_argument("--out_csv", type=str, default=str(LOG_DIR / "diffik_probe_smoke_per_env.csv"))
    parser.add_argument("--summary_json", type=str, default=str(LOG_DIR / "diffik_probe_smoke_summary.json"))
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=not args.gui, enable_cameras=bool(args.enable_cameras or args.record_video))
    sim_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import roarm_rl  # noqa: F401 - registers envs
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_rotate, subtract_frame_transforms

    from roarm_rl.roarm_cube_push_env import (
        AUDIT_DISP_XY_P99_M,
        AUDIT_SPEED_P95_MPS,
        AUDIT_SPEED_P99_MPS,
        AUDIT_TIP_P95_DEG,
        AUDIT_TIP_P99_DEG,
        RoArmCubePushEnvCfg,
    )
    from roarm_rl.roarm_stack_env import TABLE_Z

    video_writer = None
    video_annot = None
    video_camera_xf = None
    video_frame_count = 0
    video_frame_dir = None
    video_path = None
    if args.record_video:
        if not (0 <= int(args.video_env_id) < int(args.num_envs)):
            raise ValueError(f"--video_env_id must be in [0, {int(args.num_envs) - 1}]")
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image, ImageDraw
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cam_prim = UsdGeom.Camera.Define(stage, "/World/DiffIKVideoCam")
        cam_prim.CreateFocalLengthAttr(18.0)
        cam_prim.CreateHorizontalApertureAttr(22.0)
        cam_prim.CreateVerticalApertureAttr(22.0 * float(args.video_height) / float(args.video_width))
        cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.03, 5.0))
        video_camera_xf = UsdGeom.Xformable(cam_prim.GetPrim())
        video_camera_xf.ClearXformOpOrder()
        video_camera_xf.AddTransformOp()
        render_product = rep.create.render_product("/World/DiffIKVideoCam", (int(args.video_width), int(args.video_height)))
        video_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        video_annot.attach([render_product])
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / args.video_name
        video_frame_dir = video_dir / "frames"
        video_frame_dir.mkdir(parents=True, exist_ok=True)

        def set_video_camera(eye_xyz: np.ndarray, target_xyz: np.ndarray) -> None:
            forward = target_xyz - eye_xyz
            forward = forward / np.linalg.norm(forward)
            up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            right = np.cross(forward, up_guess)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            mat = np.eye(4, dtype=np.float64)
            mat[:3, 0] = right
            mat[:3, 1] = up
            mat[:3, 2] = -forward
            mat[:3, 3] = eye_xyz
            video_camera_xf.GetOrderedXformOps()[0].Set(Gf.Matrix4d(*mat.T.flatten().tolist()))

        def draw_video_overlay(rgb: np.ndarray, text: str) -> np.ndarray:
            img = Image.fromarray(rgb[:, :, :3].astype(np.uint8))
            draw = ImageDraw.Draw(img)
            draw.rectangle((12, 10, 780, 88), fill=(0, 0, 0))
            draw.text((24, 20), text, fill=(255, 255, 255))
            return np.asarray(img)

        def capture_video_frame(text: str) -> None:
            nonlocal video_frame_count
            rep.orchestrator.step()
            sim_app.update()
            rgb = video_annot.get_data()
            if rgb is None or getattr(rgb, "ndim", 0) != 3:
                raise RuntimeError(f"video rgb annotator returned {type(rgb)}")
            frame_path = video_frame_dir / f"frame_{video_frame_count:04d}.png"
            Image.fromarray(draw_video_overlay(rgb, text)).save(frame_path)
            video_frame_count += 1

    env_cfg = RoArmCubePushEnvCfg()
    cube_size = tuple(float(x) for x in args.cube_size_m)
    cube_mass_kg = float(args.cube_mass_kg)
    if any(x <= 0.0 for x in cube_size):
        raise ValueError(f"--cube_size_m values must be positive, got {cube_size}")
    if cube_mass_kg <= 0.0:
        raise ValueError(f"--cube_mass_kg must be positive, got {cube_mass_kg}")
    cube_volume_m3 = cube_size[0] * cube_size[1] * cube_size[2]
    cube_density_kg_m3 = cube_mass_kg / cube_volume_m3
    object_size_ref_m = max(cube_size)
    cube_center_z_m = float(TABLE_Z) + cube_size[2] / 2.0
    contact_stop_disp_m = (
        float(args.gate_disp_m) if args.contact_stop_disp_m is None else float(args.contact_stop_disp_m)
    )
    if float(args.contact_detect_disp_m) < 0.0:
        raise ValueError("--contact_detect_disp_m must be non-negative")
    if contact_stop_disp_m <= 0.0:
        raise ValueError("--contact_stop_disp_m must be positive")
    if float(args.contact_overshoot_disp_m) <= 0.0:
        raise ValueError("--contact_overshoot_disp_m must be positive")
    if float(args.contact_near_tcp_cube_dist_m) <= 0.0:
        raise ValueError("--contact_near_tcp_cube_dist_m must be positive")
    if not (0.0 < float(args.contact_near_joint_step_scale) <= 1.0):
        raise ValueError("--contact_near_joint_step_scale must be in (0, 1]")
    if not (0.0 < float(args.contact_stop_joint_step_scale) <= 1.0):
        raise ValueError("--contact_stop_joint_step_scale must be in (0, 1]")
    if float(args.contact_retract_clearance_m) < 0.0:
        raise ValueError("--contact_retract_clearance_m must be non-negative")
    if float(args.reaction_disp_m) < 0.0:
        raise ValueError("--reaction_disp_m must be non-negative")
    if float(args.reaction_z_delta_m) < 0.0:
        raise ValueError("--reaction_z_delta_m must be non-negative")
    if float(args.reaction_speed_mps) < 0.0:
        raise ValueError("--reaction_speed_mps must be non-negative")
    if args.contact_controller_mode == "measured_stop" and args.through_target_mode != "near_face":
        raise ValueError("--contact_controller_mode measured_stop requires --through_target_mode near_face")
    if args.fixed_push_dir is not None:
        fixed_norm = math.hypot(float(args.fixed_push_dir[0]), float(args.fixed_push_dir[1]))
        if fixed_norm <= 1.0e-6:
            raise ValueError("--fixed_push_dir must be nonzero")
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.cube_size_x_m = cube_size[0]
    env_cfg.cube_size_y_m = cube_size[1]
    env_cfg.cube_size_z_m = cube_size[2]
    env_cfg.sponge.spawn.size = cube_size
    env_cfg.sponge.spawn.mass_props.mass = cube_mass_kg
    env_cfg.sponge.init_state.pos = (0.30, 0.00, cube_center_z_m)
    if args.fixed_cube_x_m is not None:
        env_cfg.cube_x_min = float(args.fixed_cube_x_m)
        env_cfg.cube_x_max = float(args.fixed_cube_x_m)
    if args.fixed_cube_y_m is not None:
        env_cfg.cube_y_min = float(args.fixed_cube_y_m)
        env_cfg.cube_y_max = float(args.fixed_cube_y_m)
    if args.fixed_push_dir is not None:
        env_cfg.fixed_push_dir_x = float(args.fixed_push_dir[0])
        env_cfg.fixed_push_dir_y = float(args.fixed_push_dir[1])
    if args.cube_push_target_disp_m is not None:
        env_cfg.cube_push_target_disp_m = float(args.cube_push_target_disp_m)
    if args.cube_success_disp_m is not None:
        env_cfg.cube_success_disp_m = float(args.cube_success_disp_m)
    arm_actuator = env_cfg.robot.actuators["arm"]
    if args.arm_stiffness_override is not None:
        arm_actuator.stiffness = float(args.arm_stiffness_override)
    if args.arm_damping_override is not None:
        arm_actuator.damping = float(args.arm_damping_override)
    if args.arm_effort_limit_sim_override is not None:
        arm_actuator.effort_limit_sim = float(args.arm_effort_limit_sim_override)
    if args.arm_velocity_limit_sim_override is not None:
        arm_actuator.velocity_limit_sim = float(args.arm_velocity_limit_sim_override)
    env_cfg.ik_endpoint_reset = False
    env_cfg.scripted_teacher_blend = 0.0
    env_cfg.action_scale = 0.0
    base_total_steps = int(args.approach_steps + args.push_steps + args.post_steps)
    v2_posx_total_steps = int(args.v2_posx_approach_steps + args.v2_posx_push_steps + args.v2_posx_post_steps)
    v3_posx_total_steps = int(args.v3_posx_approach_steps + args.v3_posx_push_steps + args.v3_posx_post_steps)
    v31_posx_total_steps = int(args.v31_posx_approach_steps + args.v31_posx_push_steps + args.v31_posx_post_steps)
    v31_lowx_total_steps = int(args.v31_lowx_approach_steps + args.v31_lowx_push_steps + args.v31_lowx_post_steps)
    variant_posx_total_steps = base_total_steps
    if args.trajectory_variant == "v2":
        variant_posx_total_steps = v2_posx_total_steps
    elif args.trajectory_variant == "v3":
        variant_posx_total_steps = v3_posx_total_steps
    elif args.trajectory_variant == "v3_1":
        variant_posx_total_steps = max(v31_posx_total_steps, v31_lowx_total_steps)
    total_steps = max(base_total_steps, variant_posx_total_steps)
    step_dt = float(env_cfg.sim.dt) * float(env_cfg.decimation)
    min_episode_s = (int(args.settle_steps) + total_steps + 20) * step_dt
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), min_episode_s)

    env = gym.make("RoArm-CubePush-Direct-v0", cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    n = inner.num_envs

    def probe_get_dones() -> tuple[torch.Tensor, torch.Tensor]:
        no_done = torch.zeros(n, dtype=torch.bool, device=device)
        return no_done, no_done

    def probe_pre_physics_step(actions: torch.Tensor) -> None:
        inner.actions = torch.zeros_like(actions)
        inner._last_teacher_blend.zero_()
        inner._last_joint_delta_abs_mean.zero_()
        inner._last_contact_slowdown.fill_(1.0)

    inner._get_dones = probe_get_dones
    inner._pre_physics_step = probe_pre_physics_step

    arm_joint_ids, arm_joint_names = inner._robot.find_joints(
        [
            "base_link_to_link1",
            "link1_to_link2",
            "link2_to_link3",
            "link3_to_link4",
            "link4_to_link5",
        ],
        preserve_order=True,
    )
    link5_body_idx = inner.link5_idx
    jacobi_body_idx = link5_body_idx - 1 if inner._robot.is_fixed_base else link5_body_idx
    jacobi_joint_ids = arm_joint_ids if inner._robot.is_fixed_base else [idx + 6 for idx in arm_joint_ids]

    diffik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": float(args.dls_lambda)},
    )
    diffik = DifferentialIKController(diffik_cfg, num_envs=n, device=device)

    original_write_pose = inner._sponge.write_root_pose_to_sim
    counters = {"posewrite_calls_during_rollout": 0}
    posewrite_watch = {"active": False}

    def watched_write_root_pose_to_sim(*a, **kw):
        if posewrite_watch["active"]:
            counters["posewrite_calls_during_rollout"] += 1
        return original_write_pose(*a, **kw)

    inner._sponge.write_root_pose_to_sim = watched_write_root_pose_to_sim

    zero_action = torch.zeros((n, inner.cfg.action_space), device=device, dtype=torch.float32)
    tcp_local = inner._tcp_local.unsqueeze(0).repeat(n, 1)
    half_xy = torch.tensor((cube_size[0] / 2.0, cube_size[1] / 2.0), device=device, dtype=torch.float32)
    cube_top_half_z = float(cube_size[2] / 2.0)

    print(
        "[cube3cm_push_diffik_probe] "
        f"isaac_run=YES num_envs={n} episodes={args.episodes} total_trials={n * args.episodes} "
        f"cube_size_m=({cube_size[0]:.6f},{cube_size[1]:.6f},{cube_size[2]:.6f}) "
        f"cube_mass_kg={cube_mass_kg:.6f} density_kg_m3={cube_density_kg_m3:.3f} "
        f"object_size_ref_m={object_size_ref_m:.6f} table_z_m={float(TABLE_Z):.6f} "
        f"cube_center_z_m={cube_center_z_m:.6f} "
        f"cube_push_target_disp_m={float(env_cfg.cube_push_target_disp_m):.6f} "
        f"cube_success_disp_m={float(env_cfg.cube_success_disp_m):.6f} gate_disp_m={float(args.gate_disp_m):.6f} "
        f"fixed_cube_x_m={args.fixed_cube_x_m} fixed_cube_y_m={args.fixed_cube_y_m} "
        f"fixed_push_dir={args.fixed_push_dir} tcp_height_mode={args.tcp_height_mode} "
        f"base_lateral_offset_m={args.base_lateral_offset_m:.6f} "
        f"through_target_mode={args.through_target_mode} "
        f"contact_controller_mode={args.contact_controller_mode} "
        f"contact_stop_target_mode={args.contact_stop_target_mode} "
        f"contact_stop_disp_m={contact_stop_disp_m:.6f} "
        "controller=IsaacLab_DifferentialIKController ik_method=dls command_type=position "
        "local_roarm_ik_dls_control_loop=NO training=NO dataset_generation=NO "
        "grasp=NO attach_posewrite=NO rollout_object_posewrite=NO",
        flush=True,
    )
    print(
        "[cube3cm_push_diffik_probe] "
        f"robot_usd_path={args.robot_usd_path} arm_joint_names={arm_joint_names} "
        f"ee_body=link5 body_idx={link5_body_idx} jacobi_body_idx={jacobi_body_idx} "
        f"trajectory_variant={args.trajectory_variant} "
        f"base_steps={args.approach_steps}/{args.push_steps}/{args.post_steps} "
        f"v2_posx_steps={args.v2_posx_approach_steps}/{args.v2_posx_push_steps}/{args.v2_posx_post_steps} "
        f"v3_posx_steps={args.v3_posx_approach_steps}/{args.v3_posx_push_steps}/{args.v3_posx_post_steps} "
        f"v31_posx_steps={args.v31_posx_approach_steps}/{args.v31_posx_push_steps}/{args.v31_posx_post_steps} "
        f"v31_lowx_steps={args.v31_lowx_approach_steps}/{args.v31_lowx_push_steps}/{args.v31_lowx_post_steps} "
        f"max_diffik_joint_step_rad={args.max_diffik_joint_step_rad:.6f} "
        f"v2_posx_max_diffik_joint_step_rad={args.v2_posx_max_diffik_joint_step_rad:.6f} "
        f"v3_posx_max_diffik_joint_step_rad={args.v3_posx_max_diffik_joint_step_rad:.6f} "
        f"v31_posx_max_diffik_joint_step_rad={args.v31_posx_max_diffik_joint_step_rad:.6f} "
        f"v31_lowx_max_diffik_joint_step_rad={args.v31_lowx_max_diffik_joint_step_rad:.6f} "
        f"dls_lambda={args.dls_lambda:.6f} "
        f"arm_actuator_stiffness={arm_actuator.stiffness} arm_actuator_damping={arm_actuator.damping} "
        f"arm_actuator_effort_limit_sim={arm_actuator.effort_limit_sim} "
        f"arm_actuator_velocity_limit_sim={arm_actuator.velocity_limit_sim} "
        f"env_auto_reset_disabled=YES "
        f"env_joint_delta_action_loop_bypassed=YES episode_length_s={env_cfg.episode_length_s:.3f}",
        flush=True,
    )

    records: list[dict[str, float | int | str]] = []
    trace_records: list[dict[str, float | int]] = []
    trace_env_ids: list[int] = []
    if bool(args.trace_all_envs):
        trace_env_ids.extend(range(n))
    elif args.trace_env_ids:
        trace_env_ids.extend(int(idx) for idx in args.trace_env_ids)
    elif int(args.trace_env_id) >= 0:
        trace_env_ids.append(int(args.trace_env_id))
    trace_env_ids = sorted({idx for idx in trace_env_ids if 0 <= idx < n})
    t0 = time.time()
    posx_variant_env_count = 0
    posx_variant_trial_count = 0

    def build_trajectory_tensors() -> dict[str, torch.Tensor]:
        cube = inner._cube_start_w
        push_dir = inner._push_dir_xy
        posx = (push_dir[:, 0] > 0.5) & (torch.abs(push_dir[:, 1]) < 0.5)
        cube_x_local = cube[:, 0] - inner.scene.env_origins[:, 0]
        v31_lowx = posx & (cube_x_local <= float(args.v31_lowx_threshold_m))
        approach_steps = torch.full((n,), int(args.approach_steps), dtype=torch.long, device=device)
        push_steps = torch.full((n,), int(args.push_steps), dtype=torch.long, device=device)
        post_steps = torch.full((n,), int(args.post_steps), dtype=torch.long, device=device)
        max_joint_step = torch.full((n,), float(args.max_diffik_joint_step_rad), dtype=torch.float32, device=device)
        precontact = torch.full((n,), float(args.precontact_clearance_m), dtype=torch.float32, device=device)
        push_through = torch.full((n,), float(args.push_through_m), dtype=torch.float32, device=device)
        tcp_top_margin = torch.full((n,), float(args.tcp_top_margin_m), dtype=torch.float32, device=device)
        lateral_offset = torch.full((n,), float(args.base_lateral_offset_m), dtype=torch.float32, device=device)
        if args.trajectory_variant == "v2":
            approach_steps[posx] = int(args.v2_posx_approach_steps)
            push_steps[posx] = int(args.v2_posx_push_steps)
            post_steps[posx] = int(args.v2_posx_post_steps)
            max_joint_step[posx] = float(args.v2_posx_max_diffik_joint_step_rad)
            precontact[posx] = float(args.v2_posx_precontact_clearance_m)
            push_through[posx] = float(args.v2_posx_push_through_m)
            tcp_top_margin[posx] = float(args.v2_posx_tcp_top_margin_m)
            lateral_offset[posx] = float(args.v2_posx_lateral_offset_m)
        elif args.trajectory_variant == "v3":
            approach_steps[posx] = int(args.v3_posx_approach_steps)
            push_steps[posx] = int(args.v3_posx_push_steps)
            post_steps[posx] = int(args.v3_posx_post_steps)
            max_joint_step[posx] = float(args.v3_posx_max_diffik_joint_step_rad)
            precontact[posx] = float(args.v3_posx_precontact_clearance_m)
            push_through[posx] = float(args.v3_posx_push_through_m)
            tcp_top_margin[posx] = float(args.v3_posx_tcp_top_margin_m)
            lateral_offset[posx] = float(args.v3_posx_lateral_offset_m)
        elif args.trajectory_variant == "v3_1":
            approach_steps[posx] = int(args.v31_posx_approach_steps)
            push_steps[posx] = int(args.v31_posx_push_steps)
            post_steps[posx] = int(args.v31_posx_post_steps)
            max_joint_step[posx] = float(args.v31_posx_max_diffik_joint_step_rad)
            precontact[posx] = float(args.v31_posx_precontact_clearance_m)
            push_through[posx] = float(args.v31_posx_push_through_m)
            tcp_top_margin[posx] = float(args.v31_posx_tcp_top_margin_m)
            lateral_offset[posx] = float(args.v31_posx_lateral_offset_m)
            approach_steps[v31_lowx] = int(args.v31_lowx_approach_steps)
            push_steps[v31_lowx] = int(args.v31_lowx_push_steps)
            post_steps[v31_lowx] = int(args.v31_lowx_post_steps)
            max_joint_step[v31_lowx] = float(args.v31_lowx_max_diffik_joint_step_rad)
            precontact[v31_lowx] = float(args.v31_lowx_precontact_clearance_m)
            push_through[v31_lowx] = float(args.v31_lowx_push_through_m)
            tcp_top_margin[v31_lowx] = float(args.v31_lowx_tcp_top_margin_m)
            lateral_offset[v31_lowx] = float(args.v31_lowx_lateral_offset_m)
        return {
            "posx": posx,
            "v31_lowx": v31_lowx,
            "approach_steps": approach_steps,
            "push_steps": push_steps,
            "post_steps": post_steps,
            "max_joint_step": max_joint_step,
            "precontact": precontact,
            "push_through": push_through,
            "tcp_top_margin": tcp_top_margin,
            "lateral_offset": lateral_offset,
        }

    def compute_alpha(step: int, traj: dict[str, torch.Tensor]) -> torch.Tensor:
        step_v = torch.full((n,), int(step), dtype=torch.float32, device=device)
        approach_steps = traj["approach_steps"].to(dtype=torch.float32)
        push_steps = torch.clamp(traj["push_steps"].to(dtype=torch.float32), min=1.0)
        push_start = approach_steps
        push_end = approach_steps + push_steps
        raw_alpha = (step_v - push_start + 1.0) / push_steps
        alpha = torch.where(step_v < push_start, torch.zeros_like(raw_alpha), raw_alpha)
        alpha = torch.where(step_v >= push_end, torch.ones_like(alpha), alpha)
        return torch.clamp(alpha, min=0.0, max=1.0)

    def compute_tcp_targets(alpha: torch.Tensor, traj: dict[str, torch.Tensor]) -> torch.Tensor:
        cube = inner._cube_start_w
        push_dir = inner._push_dir_xy
        lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=-1)
        half_along = torch.sum(torch.abs(push_dir) * half_xy.unsqueeze(0), dim=-1)
        pre = cube.clone()
        through = cube.clone()
        if args.tcp_height_mode == "top_margin":
            z = cube[:, 2] + cube_top_half_z + traj["tcp_top_margin"]
        elif args.tcp_height_mode == "side_center":
            z = cube[:, 2] + float(args.tcp_center_height_offset_m)
        else:
            raise ValueError(f"unsupported tcp_height_mode={args.tcp_height_mode!r}")
        lateral = lateral_dir * traj["lateral_offset"].unsqueeze(-1)
        pre[:, 0:2] = cube[:, 0:2] - push_dir * (half_along + traj["precontact"]).unsqueeze(-1) + lateral
        if args.through_target_mode == "far_face":
            through[:, 0:2] = cube[:, 0:2] + push_dir * (half_along + traj["push_through"]).unsqueeze(-1) + lateral
        elif args.through_target_mode == "near_face":
            through[:, 0:2] = cube[:, 0:2] - push_dir * (half_along - traj["push_through"]).unsqueeze(-1) + lateral
        else:
            raise ValueError(f"unsupported through_target_mode={args.through_target_mode!r}")
        pre[:, 2] = z
        through[:, 2] = z
        return pre + alpha.unsqueeze(-1) * (through - pre)

    def compute_retract_tcp_targets(traj: dict[str, torch.Tensor]) -> torch.Tensor:
        cube = inner._cube_start_w
        push_dir = inner._push_dir_xy
        lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=-1)
        half_along = torch.sum(torch.abs(push_dir) * half_xy.unsqueeze(0), dim=-1)
        if args.tcp_height_mode == "top_margin":
            z = cube[:, 2] + cube_top_half_z + traj["tcp_top_margin"]
        elif args.tcp_height_mode == "side_center":
            z = cube[:, 2] + float(args.tcp_center_height_offset_m)
        else:
            raise ValueError(f"unsupported tcp_height_mode={args.tcp_height_mode!r}")
        lateral = lateral_dir * traj["lateral_offset"].unsqueeze(-1)
        target = cube.clone()
        clearance = torch.full_like(half_along, float(args.contact_retract_clearance_m))
        target[:, 0:2] = cube[:, 0:2] - push_dir * (half_along + clearance).unsqueeze(-1) + lateral
        target[:, 2] = z
        return target

    def compute_diffik_joint_target(tcp_target_w: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        root_pos_w = inner._robot.data.root_pos_w
        root_quat_w = inner._robot.data.root_quat_w
        link5_pos_w = inner._robot.data.body_pos_w[:, link5_body_idx].clone()
        link5_quat_w = inner._robot.data.body_quat_w[:, link5_body_idx].clone()
        link5_pos_b, link5_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, link5_pos_w, link5_quat_w)
        tcp_offset_w = quat_rotate(link5_quat_w, tcp_local)
        link5_target_w = tcp_target_w - tcp_offset_w
        link5_target_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, link5_target_w)
        jacobian = inner._robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids]
        base_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
        jacobian = jacobian.clone()
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        joint_pos = inner._robot.data.joint_pos[:, arm_joint_ids]
        diffik.set_command(link5_target_b, ee_pos=link5_pos_b, ee_quat=link5_quat_b)
        joint_pos_des = diffik.compute(link5_pos_b, link5_quat_b, jacobian, joint_pos)
        return joint_pos_des, {
            "link5_pos_w": link5_pos_w,
            "link5_quat_w": link5_quat_w,
            "link5_target_w": link5_target_w,
            "link5_pos_b": link5_pos_b,
            "link5_target_b": link5_target_b,
            "tcp_offset_w": tcp_offset_w,
        }

    try:
        for episode in range(int(args.episodes)):
            env.reset()
            diffik.reset()
            posewrite_watch["active"] = False
            inner._grasped[:] = False
            inner._was_grasped[:] = False

            for _ in range(int(args.settle_steps)):
                env.step(zero_action)

            inner._compute_intermediate_values()
            push_dir = inner._push_dir_xy.clone()
            # Use the settled PhysX pose as the diagnostic start. The reset buffer can
            # differ from the actual cuboid center after plane contact resolution.
            cube_start_w = inner._sponge_pos_w.clone()
            inner._cube_start_w[:] = cube_start_w
            inner._target_world[:, 0:2] = cube_start_w[:, 0:2] + push_dir * float(env_cfg.cube_push_target_disp_m)
            inner._target_world[:, 2] = cube_start_w[:, 2]
            min_tcp_cube_dist = torch.full((n,), float("inf"), device=device)
            min_tcp_target_err = torch.full((n,), float("inf"), device=device)
            final_tcp_target_err = torch.zeros((n,), device=device)
            max_cube_speed = torch.zeros((n,), device=device)
            max_disp_along = torch.full((n,), -float("inf"), device=device)
            max_cube_z_delta = torch.zeros((n,), device=device)
            max_tip_angle_deg = torch.zeros((n,), device=device)
            max_joint_delta_abs = torch.zeros((n,), device=device)
            clipped_steps = torch.zeros((n,), device=device)
            traj = build_trajectory_tensors()
            stop_tcp_target_w = torch.zeros((n, 3), dtype=torch.float32, device=device)
            near_tcp_cube_seen = torch.zeros((n,), dtype=torch.bool, device=device)
            measured_contact_seen = torch.zeros((n,), dtype=torch.bool, device=device)
            contact_stop_seen = torch.zeros((n,), dtype=torch.bool, device=device)
            contact_overshoot_seen = torch.zeros((n,), dtype=torch.bool, device=device)
            near_tcp_cube_steps = torch.zeros((n,), device=device)
            measured_contact_steps = torch.zeros((n,), device=device)
            contact_stop_steps = torch.zeros((n,), device=device)
            first_near_tcp_cube_step = torch.full((n,), -1, dtype=torch.int32, device=device)
            first_contact_step = torch.full((n,), -1, dtype=torch.int32, device=device)
            first_stop_step = torch.full((n,), -1, dtype=torch.int32, device=device)
            posx_variant_env_count = int(traj["posx"].sum().detach().cpu().item())
            posx_variant_trial_count += posx_variant_env_count
            if args.record_video and episode == 0:
                vid_idx = int(args.video_env_id)
                target_xyz = cube_start_w[vid_idx].detach().cpu().numpy().astype(np.float64)
                push_xyz = np.array(
                    [
                        float(push_dir[vid_idx, 0].detach().cpu().item()),
                        float(push_dir[vid_idx, 1].detach().cpu().item()),
                        0.0,
                    ],
                    dtype=np.float64,
                )
                target_xyz = target_xyz + push_xyz * float(args.video_target_push_m)
                target_xyz[2] += 0.035
                eye_xyz = target_xyz + np.array(args.video_eye_offset_m, dtype=np.float64)
                set_video_camera(eye_xyz, target_xyz)
                for _ in range(5):
                    sim_app.update()
                capture_video_frame(
                    "v3 scripted IsaacLab Differential IK | "
                    f"episode={episode} env={vid_idx} push=({push_xyz[0]:+.0f},{push_xyz[1]:+.0f}) | "
                    "training=NO dataset=NO"
                )
            posewrite_watch["active"] = True

            for step in range(total_steps):
                alpha = compute_alpha(step, traj)

                inner._compute_intermediate_values()
                pre_step_terms = inner._push_terms()
                step_i = torch.full((n,), int(step), dtype=torch.int32, device=device)
                near_tcp_cube_now = pre_step_terms["tcp_cube_dist"] <= float(args.contact_near_tcp_cube_dist_m)
                measured_contact_now = pre_step_terms["disp_along"] >= float(args.contact_detect_disp_m)
                stop_now = pre_step_terms["disp_along"] >= contact_stop_disp_m
                overshoot_now = pre_step_terms["disp_along"] >= float(args.contact_overshoot_disp_m)

                new_near = near_tcp_cube_now & ~near_tcp_cube_seen
                new_contact = measured_contact_now & ~measured_contact_seen
                new_stop = (stop_now | overshoot_now) & ~contact_stop_seen
                first_near_tcp_cube_step = torch.where(new_near, step_i, first_near_tcp_cube_step)
                first_contact_step = torch.where(new_contact, step_i, first_contact_step)
                first_stop_step = torch.where(new_stop, step_i, first_stop_step)
                near_tcp_cube_seen |= near_tcp_cube_now
                measured_contact_seen |= measured_contact_now
                contact_stop_seen |= stop_now | overshoot_now
                contact_overshoot_seen |= overshoot_now
                near_tcp_cube_steps += near_tcp_cube_now.float()
                measured_contact_steps += measured_contact_now.float()
                contact_stop_steps += contact_stop_seen.float()

                tcp_target_w = compute_tcp_targets(alpha, traj)
                step_scale = torch.ones((n,), dtype=torch.float32, device=device)
                if args.contact_controller_mode == "measured_stop":
                    if args.contact_stop_target_mode == "freeze":
                        stop_tcp_target_w = torch.where(new_stop.unsqueeze(-1), tcp_target_w, stop_tcp_target_w)
                        tcp_target_w = torch.where(contact_stop_seen.unsqueeze(-1), stop_tcp_target_w, tcp_target_w)
                    elif args.contact_stop_target_mode == "retract":
                        retract_target_w = compute_retract_tcp_targets(traj)
                        tcp_target_w = torch.where(contact_stop_seen.unsqueeze(-1), retract_target_w, tcp_target_w)
                    else:
                        raise ValueError(f"unsupported contact_stop_target_mode={args.contact_stop_target_mode!r}")
                    step_scale = torch.where(
                        near_tcp_cube_now | measured_contact_seen,
                        torch.full_like(step_scale, float(args.contact_near_joint_step_scale)),
                        step_scale,
                    )
                    step_scale = torch.where(
                        contact_stop_seen,
                        torch.full_like(step_scale, float(args.contact_stop_joint_step_scale)),
                        step_scale,
                    )
                tcp_pos_before_w = inner._tcp_pos_w.clone()
                joint_target_before = inner.robot_dof_targets[:, arm_joint_ids].clone()
                joint_pos_arm = inner._robot.data.joint_pos[:, arm_joint_ids].clone()
                joint_pos_des, diffik_terms = compute_diffik_joint_target(tcp_target_w)
                raw_delta = joint_pos_des - joint_pos_arm
                max_step = traj["max_joint_step"].unsqueeze(-1) * step_scale.unsqueeze(-1)
                clip_mask = torch.abs(raw_delta) > max_step
                clip_joint_count = clip_mask.sum(dim=-1)
                clipped_delta = torch.maximum(torch.minimum(raw_delta, max_step), -max_step)
                clipped_steps += (clip_joint_count > 0).float()
                max_joint_delta_abs = torch.maximum(max_joint_delta_abs, torch.max(torch.abs(clipped_delta), dim=-1).values)
                raw_delta_abs = torch.abs(raw_delta)
                clip_max_joint_idx = torch.argmax(raw_delta_abs, dim=-1)
                tcp_err_before = torch.norm(tcp_pos_before_w - tcp_target_w, p=2, dim=-1)
                link5_err_before = torch.norm(diffik_terms["link5_pos_w"] - diffik_terms["link5_target_w"], p=2, dim=-1)

                target_full = inner.robot_dof_targets.clone()
                target_full[:, arm_joint_ids] = joint_pos_arm + clipped_delta
                target_full[:, inner.gripper_joint_idx] = 0.0
                target_full = torch.clamp(target_full, inner.robot_dof_lower_limits, inner.robot_dof_upper_limits)
                target_full[:, inner.gripper_joint_idx] = 0.0
                inner.robot_dof_targets[:] = target_full

                env.step(zero_action)
                inner._compute_intermediate_values()
                post_step_terms = inner._push_terms()
                max_disp_along = torch.maximum(max_disp_along, post_step_terms["disp_along"])
                max_cube_z_delta = torch.maximum(max_cube_z_delta, inner._sponge_pos_w[:, 2] - cube_start_w[:, 2])
                max_tip_angle_deg = torch.maximum(max_tip_angle_deg, post_step_terms["tip_angle_deg"])
                link5_pos_after_w = inner._robot.data.body_pos_w[:, link5_body_idx].clone()
                joint_pos_after = inner._robot.data.joint_pos[:, arm_joint_ids].clone()
                robot_dof_targets_after = inner.robot_dof_targets[:, arm_joint_ids].clone()
                tcp_err = torch.norm(inner._tcp_pos_w - tcp_target_w, p=2, dim=-1)
                link5_err_after = torch.norm(link5_pos_after_w - diffik_terms["link5_target_w"], p=2, dim=-1)
                final_tcp_target_err = tcp_err
                min_tcp_target_err = torch.minimum(min_tcp_target_err, tcp_err)
                min_tcp_cube_dist = torch.minimum(
                    min_tcp_cube_dist,
                    torch.norm(inner._tcp_pos_w - inner._sponge_pos_w, p=2, dim=-1),
                )
                max_cube_speed = torch.maximum(max_cube_speed, torch.norm(inner._sponge.data.root_lin_vel_w, p=2, dim=-1))
                if trace_env_ids and episode == 0 and step % max(1, int(args.trace_stride)) == 0:
                    trace_terms = post_step_terms
                    trace_frame = step // max(1, int(args.trace_stride))
                    for trace_idx in trace_env_ids:
                        trace_row: dict[str, float | int | str] = {
                            "frame": int(trace_frame),
                            "step": int(step),
                            "env_id": trace_idx,
                            "trajectory_variant": args.trajectory_variant,
                            "push_dx": float(push_dir[trace_idx, 0].detach().cpu().item()),
                            "push_dy": float(push_dir[trace_idx, 1].detach().cpu().item()),
                            "cube_size_x_m": cube_size[0],
                            "cube_size_y_m": cube_size[1],
                            "cube_size_z_m": cube_size[2],
                            "cube_mass_kg": cube_mass_kg,
                            "object_size_ref_m": object_size_ref_m,
                            "tcp_height_mode": args.tcp_height_mode,
                            "tcp_center_height_offset_m": float(args.tcp_center_height_offset_m),
                            "env_origin_x_m": float(inner.scene.env_origins[trace_idx, 0].detach().cpu().item()),
                            "env_origin_y_m": float(inner.scene.env_origins[trace_idx, 1].detach().cpu().item()),
                            "env_origin_z_m": float(inner.scene.env_origins[trace_idx, 2].detach().cpu().item()),
                            "cube_x_m": float(inner._sponge_pos_w[trace_idx, 0].detach().cpu().item()),
                            "cube_y_m": float(inner._sponge_pos_w[trace_idx, 1].detach().cpu().item()),
                            "cube_z_m": float(inner._sponge_pos_w[trace_idx, 2].detach().cpu().item()),
                            "cube_qw": float(inner._sponge_quat_w[trace_idx, 0].detach().cpu().item()),
                            "cube_qx": float(inner._sponge_quat_w[trace_idx, 1].detach().cpu().item()),
                            "cube_qy": float(inner._sponge_quat_w[trace_idx, 2].detach().cpu().item()),
                            "cube_qz": float(inner._sponge_quat_w[trace_idx, 3].detach().cpu().item()),
                            "tcp_x_m": float(inner._tcp_pos_w[trace_idx, 0].detach().cpu().item()),
                            "tcp_y_m": float(inner._tcp_pos_w[trace_idx, 1].detach().cpu().item()),
                            "tcp_z_m": float(inner._tcp_pos_w[trace_idx, 2].detach().cpu().item()),
                            "target_x_m": float(tcp_target_w[trace_idx, 0].detach().cpu().item()),
                            "target_y_m": float(tcp_target_w[trace_idx, 1].detach().cpu().item()),
                            "target_z_m": float(tcp_target_w[trace_idx, 2].detach().cpu().item()),
                            "phase_alpha": float(alpha[trace_idx].detach().cpu().item()),
                            "through_target_mode": args.through_target_mode,
                            "contact_controller_mode": args.contact_controller_mode,
                            "near_tcp_cube_now": int(bool(near_tcp_cube_now[trace_idx].detach().cpu().item())),
                            "near_tcp_cube_seen": int(bool(near_tcp_cube_seen[trace_idx].detach().cpu().item())),
                            "measured_contact_now": int(bool(measured_contact_now[trace_idx].detach().cpu().item())),
                            "measured_contact_seen": int(
                                bool(measured_contact_seen[trace_idx].detach().cpu().item())
                            ),
                            "contact_stop_seen": int(bool(contact_stop_seen[trace_idx].detach().cpu().item())),
                            "contact_overshoot_seen": int(
                                bool(contact_overshoot_seen[trace_idx].detach().cpu().item())
                            ),
                            "first_near_tcp_cube_step": int(
                                first_near_tcp_cube_step[trace_idx].detach().cpu().item()
                            ),
                            "first_contact_step": int(first_contact_step[trace_idx].detach().cpu().item()),
                            "first_stop_step": int(first_stop_step[trace_idx].detach().cpu().item()),
                            "joint_step_scale": float(step_scale[trace_idx].detach().cpu().item()),
                            "contact_stop_disp_m": float(contact_stop_disp_m),
                            "disp_along_push_m": float(trace_terms["disp_along"][trace_idx].detach().cpu().item()),
                            "disp_xy_m": float(trace_terms["disp_xy"][trace_idx].detach().cpu().item()),
                            "lateral_abs_m": float(trace_terms["lateral_abs"][trace_idx].detach().cpu().item()),
                            "target_xy_dist_m": float(trace_terms["target_xy_dist"][trace_idx].detach().cpu().item()),
                            "tcp_cube_dist_m": float(trace_terms["tcp_cube_dist"][trace_idx].detach().cpu().item()),
                            "cube_speed_mps": float(trace_terms["speed"][trace_idx].detach().cpu().item()),
                            "tip_angle_deg": float(trace_terms["tip_angle_deg"][trace_idx].detach().cpu().item()),
                            "controlled_push": int(bool(trace_terms["controlled"][trace_idx].detach().cpu().item())),
                            "impact_outlier": int(bool(trace_terms["impact"][trace_idx].detach().cpu().item())),
                            "low_motion": int(bool(trace_terms["low_motion"][trace_idx].detach().cpu().item())),
                            "success_marker": int(bool(inner._push_success_flag[trace_idx].detach().cpu().item())),
                            "v31_lowx_applied": int(
                                bool(traj["v31_lowx"][trace_idx].detach().cpu().item())
                                and args.trajectory_variant == "v3_1"
                            ),
                        }
                        if args.trace_diffik_diagnostics:
                            clip_count = int(clip_joint_count[trace_idx].detach().cpu().item())
                            max_local_idx = int(clip_max_joint_idx[trace_idx].detach().cpu().item())
                            trace_row.update(
                                {
                                    "link5_body_idx": int(link5_body_idx),
                                    "jacobi_body_idx": int(jacobi_body_idx),
                                    "jacobi_joint_count": int(len(jacobi_joint_ids)),
                                    "clip_joint_count": clip_count,
                                    "clip_any": int(clip_count > 0),
                                    "clip_single_joint": int(clip_count == 1),
                                    "clip_all_joints": int(clip_count == len(arm_joint_ids)),
                                    "clip_max_joint_local_idx": max_local_idx,
                                    "clip_max_joint_name": arm_joint_names[max_local_idx],
                                    "tcp_target_err_before_m": float(tcp_err_before[trace_idx].detach().cpu().item()),
                                    "tcp_target_err_after_m": float(tcp_err[trace_idx].detach().cpu().item()),
                                    "link5_target_err_before_m": float(link5_err_before[trace_idx].detach().cpu().item()),
                                    "link5_target_err_after_m": float(link5_err_after[trace_idx].detach().cpu().item()),
                                    "link5_x_before_m": float(diffik_terms["link5_pos_w"][trace_idx, 0].detach().cpu().item()),
                                    "link5_y_before_m": float(diffik_terms["link5_pos_w"][trace_idx, 1].detach().cpu().item()),
                                    "link5_z_before_m": float(diffik_terms["link5_pos_w"][trace_idx, 2].detach().cpu().item()),
                                    "link5_target_x_m": float(diffik_terms["link5_target_w"][trace_idx, 0].detach().cpu().item()),
                                    "link5_target_y_m": float(diffik_terms["link5_target_w"][trace_idx, 1].detach().cpu().item()),
                                    "link5_target_z_m": float(diffik_terms["link5_target_w"][trace_idx, 2].detach().cpu().item()),
                                    "link5_x_after_m": float(link5_pos_after_w[trace_idx, 0].detach().cpu().item()),
                                    "link5_y_after_m": float(link5_pos_after_w[trace_idx, 1].detach().cpu().item()),
                                    "link5_z_after_m": float(link5_pos_after_w[trace_idx, 2].detach().cpu().item()),
                                    "tcp_x_before_m": float(tcp_pos_before_w[trace_idx, 0].detach().cpu().item()),
                                    "tcp_y_before_m": float(tcp_pos_before_w[trace_idx, 1].detach().cpu().item()),
                                    "tcp_z_before_m": float(tcp_pos_before_w[trace_idx, 2].detach().cpu().item()),
                                    "tcp_x_after_m": float(inner._tcp_pos_w[trace_idx, 0].detach().cpu().item()),
                                    "tcp_y_after_m": float(inner._tcp_pos_w[trace_idx, 1].detach().cpu().item()),
                                    "tcp_z_after_m": float(inner._tcp_pos_w[trace_idx, 2].detach().cpu().item()),
                                }
                            )
                        for local_idx, joint_idx in enumerate(arm_joint_ids):
                            trace_row[f"arm_joint_{local_idx}_rad"] = float(
                                inner._robot.data.joint_pos[trace_idx, joint_idx].detach().cpu().item()
                            )
                            trace_row[f"joint_target_{local_idx}_rad"] = float(
                                target_full[trace_idx, joint_idx].detach().cpu().item()
                            )
                            trace_row[f"joint_delta_{local_idx}_rad"] = float(
                                clipped_delta[trace_idx, local_idx].detach().cpu().item()
                            )
                            if args.trace_diffik_diagnostics:
                                trace_row[f"joint_pos_before_{local_idx}_rad"] = float(
                                    joint_pos_arm[trace_idx, local_idx].detach().cpu().item()
                                )
                                trace_row[f"joint_pos_des_{local_idx}_rad"] = float(
                                    joint_pos_des[trace_idx, local_idx].detach().cpu().item()
                                )
                                trace_row[f"raw_delta_{local_idx}_rad"] = float(
                                    raw_delta[trace_idx, local_idx].detach().cpu().item()
                                )
                                trace_row[f"clipped_delta_{local_idx}_rad"] = float(
                                    clipped_delta[trace_idx, local_idx].detach().cpu().item()
                                )
                                trace_row[f"clip_mask_{local_idx}"] = int(
                                    clip_mask[trace_idx, local_idx].detach().cpu().item()
                                )
                                trace_row[f"robot_dof_target_before_{local_idx}_rad"] = float(
                                    joint_target_before[trace_idx, local_idx].detach().cpu().item()
                                )
                                trace_row[f"robot_dof_target_cmd_{local_idx}_rad"] = float(
                                    target_full[trace_idx, joint_idx].detach().cpu().item()
                                )
                                trace_row[f"robot_dof_target_after_step_{local_idx}_rad"] = float(
                                    robot_dof_targets_after[trace_idx, local_idx].detach().cpu().item()
                                )
                                trace_row[f"joint_pos_after_{local_idx}_rad"] = float(
                                    joint_pos_after[trace_idx, local_idx].detach().cpu().item()
                                )
                                trace_row[f"joint_follow_err_{local_idx}_rad"] = float(
                                    (robot_dof_targets_after[trace_idx, local_idx] - joint_pos_after[trace_idx, local_idx])
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                        trace_row["gripper_joint_rad"] = float(
                            inner._robot.data.joint_pos[trace_idx, inner.gripper_joint_idx].detach().cpu().item()
                        )
                        trace_row["gripper_target_rad"] = float(
                            target_full[trace_idx, inner.gripper_joint_idx].detach().cpu().item()
                        )
                        trace_records.append(trace_row)
                if args.record_video and episode == 0 and step % max(1, int(args.video_stride)) == 0:
                    vid_idx = int(args.video_env_id)
                    capture_video_frame(
                        "v3 scripted IsaacLab Differential IK | "
                        f"step={step:03d}/{total_steps} env={vid_idx} "
                        f"push=({push_dir[vid_idx, 0].item():+.0f},{push_dir[vid_idx, 1].item():+.0f}) | "
                        "training=NO dataset=NO"
                    )
                if float(args.viewer_step_sleep_s) > 0.0:
                    time.sleep(float(args.viewer_step_sleep_s))

            posewrite_watch["active"] = False
            inner._compute_intermediate_values()
            terms = inner._push_terms()
            disp_xy_vec = inner._sponge_pos_w[:, 0:2] - cube_start_w[:, 0:2]
            lateral_vec = disp_xy_vec - terms["disp_along"].unsqueeze(-1) * push_dir
            for idx in range(n):
                reaction_event = (
                    (max_disp_along[idx] >= float(args.reaction_disp_m))
                    | (max_cube_z_delta[idx] >= float(args.reaction_z_delta_m))
                    | (max_cube_speed[idx] >= float(args.reaction_speed_mps))
                )
                records.append(
                    {
                        "trial": len(records),
                        "episode": episode,
                        "env_id": idx,
                        "cube_x0_m": float((cube_start_w[idx, 0] - inner.scene.env_origins[idx, 0]).detach().cpu().item()),
                        "cube_y0_m": float((cube_start_w[idx, 1] - inner.scene.env_origins[idx, 1]).detach().cpu().item()),
                        "cube_z0_m": float((cube_start_w[idx, 2] - inner.scene.env_origins[idx, 2]).detach().cpu().item()),
                        "push_dx": float(push_dir[idx, 0].detach().cpu().item()),
                        "push_dy": float(push_dir[idx, 1].detach().cpu().item()),
                        "cube_size_x_m": cube_size[0],
                        "cube_size_y_m": cube_size[1],
                        "cube_size_z_m": cube_size[2],
                        "cube_mass_kg": cube_mass_kg,
                        "object_size_ref_m": object_size_ref_m,
                        "tcp_height_mode": args.tcp_height_mode,
                        "tcp_center_height_offset_m": float(args.tcp_center_height_offset_m),
                        "through_target_mode": args.through_target_mode,
                        "contact_controller_mode": args.contact_controller_mode,
                        "near_tcp_cube_seen": int(bool(near_tcp_cube_seen[idx].detach().cpu().item())),
                        "measured_contact_seen": int(bool(measured_contact_seen[idx].detach().cpu().item())),
                        "contact_stop_seen": int(bool(contact_stop_seen[idx].detach().cpu().item())),
                        "contact_overshoot_seen": int(bool(contact_overshoot_seen[idx].detach().cpu().item())),
                        "first_near_tcp_cube_step": int(first_near_tcp_cube_step[idx].detach().cpu().item()),
                        "first_contact_step": int(first_contact_step[idx].detach().cpu().item()),
                        "first_stop_step": int(first_stop_step[idx].detach().cpu().item()),
                        "near_tcp_cube_step_rate": float(
                            (near_tcp_cube_steps[idx] / float(total_steps)).detach().cpu().item()
                        ),
                        "measured_contact_step_rate": float(
                            (measured_contact_steps[idx] / float(total_steps)).detach().cpu().item()
                        ),
                        "contact_stop_step_rate": float(
                            (contact_stop_steps[idx] / float(total_steps)).detach().cpu().item()
                        ),
                        "disp_along_push_m": float(terms["disp_along"][idx].detach().cpu().item()),
                        "max_disp_along_push_m": float(max_disp_along[idx].detach().cpu().item()),
                        "max_cube_z_delta_m": float(max_cube_z_delta[idx].detach().cpu().item()),
                        "max_tip_angle_deg": float(max_tip_angle_deg[idx].detach().cpu().item()),
                        "reaction_event": int(bool(reaction_event.detach().cpu().item())),
                        "disp_over_object_size": float(
                            terms["disp_along"][idx].detach().cpu().item() / object_size_ref_m
                        ),
                        "disp_xy_m": float(terms["disp_xy"][idx].detach().cpu().item()),
                        "lateral_abs_m": float(torch.norm(lateral_vec[idx], p=2).detach().cpu().item()),
                        "target_xy_dist_m": float(terms["target_xy_dist"][idx].detach().cpu().item()),
                        "max_cube_speed_mps": float(max_cube_speed[idx].detach().cpu().item()),
                        "final_speed_mps": float(terms["speed"][idx].detach().cpu().item()),
                        "tip_angle_deg": float(terms["tip_angle_deg"][idx].detach().cpu().item()),
                        "controlled_push": int(bool(terms["controlled"][idx].detach().cpu().item())),
                        "impact_outlier": int(bool(terms["impact"][idx].detach().cpu().item())),
                        "low_motion": int(bool(terms["low_motion"][idx].detach().cpu().item())),
                        "success_marker": int(bool(inner._push_success_flag[idx].detach().cpu().item())),
                        "grasped_marker": int(bool(inner._grasped[idx].detach().cpu().item())),
                        "trajectory_variant": args.trajectory_variant,
                        "posx_variant_applied": int(
                            bool(traj["posx"][idx].detach().cpu().item())
                            and args.trajectory_variant in {"v2", "v3", "v3_1"}
                        ),
                        "v2_posx_applied": int(
                            bool(traj["posx"][idx].detach().cpu().item()) and args.trajectory_variant == "v2"
                        ),
                        "v3_posx_applied": int(
                            bool(traj["posx"][idx].detach().cpu().item()) and args.trajectory_variant == "v3"
                        ),
                        "v31_posx_applied": int(
                            bool(traj["posx"][idx].detach().cpu().item()) and args.trajectory_variant == "v3_1"
                        ),
                        "v31_lowx_applied": int(
                            bool(traj["v31_lowx"][idx].detach().cpu().item()) and args.trajectory_variant == "v3_1"
                        ),
                        "precontact_clearance_m": float(traj["precontact"][idx].detach().cpu().item()),
                        "push_through_m": float(traj["push_through"][idx].detach().cpu().item()),
                        "tcp_top_margin_m": float(traj["tcp_top_margin"][idx].detach().cpu().item()),
                        "lateral_offset_m": float(traj["lateral_offset"][idx].detach().cpu().item()),
                        "phase_approach_steps": int(traj["approach_steps"][idx].detach().cpu().item()),
                        "phase_push_steps": int(traj["push_steps"][idx].detach().cpu().item()),
                        "phase_post_steps": int(traj["post_steps"][idx].detach().cpu().item()),
                        "max_diffik_joint_step_rad_cfg": float(traj["max_joint_step"][idx].detach().cpu().item()),
                        "min_tcp_cube_dist_m": float(min_tcp_cube_dist[idx].detach().cpu().item()),
                        "min_tcp_target_err_m": float(min_tcp_target_err[idx].detach().cpu().item()),
                        "final_tcp_target_err_m": float(final_tcp_target_err[idx].detach().cpu().item()),
                        "max_joint_delta_abs_rad": float(max_joint_delta_abs[idx].detach().cpu().item()),
                        "diffik_clip_rate": float((clipped_steps[idx] / float(total_steps)).detach().cpu().item()),
                    }
                )
    finally:
        if float(args.post_run_sleep_s) > 0.0:
            time.sleep(float(args.post_run_sleep_s))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    trace_csv = Path(args.trace_csv) if args.trace_csv else None
    if trace_csv is not None:
        trace_csv.parent.mkdir(parents=True, exist_ok=True)
        trace_fieldnames = list(trace_records[0].keys()) if trace_records else []
        with trace_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trace_fieldnames)
            writer.writeheader()
            writer.writerows(trace_records)

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in records) / len(records) if records else 0.0

    def rate(key: str) -> float:
        return sum(int(r[key]) for r in records) / len(records) if records else 0.0

    def threshold_rate(threshold_m: float) -> float:
        return (
            sum(float(r["disp_along_push_m"]) >= float(threshold_m) for r in records) / len(records)
            if records
            else 0.0
        )

    def max_threshold_rate(threshold_m: float) -> float:
        return (
            sum(float(r["max_disp_along_push_m"]) >= float(threshold_m) for r in records) / len(records)
            if records
            else 0.0
        )

    def disp_band_rate(low_m: float, high_m: float, *, require_controlled: bool = False) -> float:
        if not records:
            return 0.0
        good = 0
        for row in records:
            disp = float(row["disp_along_push_m"])
            in_band = float(low_m) <= disp <= float(high_m)
            if require_controlled:
                in_band = in_band and int(row["controlled_push"]) == 1 and int(row["impact_outlier"]) == 0
            good += int(in_band)
        return good / len(records)

    def max_disp_band_rate(low_m: float, high_m: float) -> float:
        if not records:
            return 0.0
        return sum(float(low_m) <= float(r["max_disp_along_push_m"]) <= float(high_m) for r in records) / len(records)

    summary = {
        "controller": "IsaacLab_DifferentialIKController",
        "ik_method": "dls",
        "command_type": "position",
        "local_roarm_ik_dls_control_loop": False,
        "training": False,
        "dataset_generation": False,
        "grasp_attach": False,
        "rollout_object_posewrite": False,
        "posewrite_calls_during_rollout": counters["posewrite_calls_during_rollout"],
        "env_auto_reset_disabled": True,
        "env_joint_delta_action_loop_bypassed": True,
        "cube_size_m": list(cube_size),
        "cube_mass_kg": cube_mass_kg,
        "cube_density_kg_m3": cube_density_kg_m3,
        "object_size_ref_m": object_size_ref_m,
        "table_z_m": float(TABLE_Z),
        "cube_center_z_m": cube_center_z_m,
        "cube_start_z_mean_m": mean("cube_z0_m"),
        "fixed_cube_x_m": None if args.fixed_cube_x_m is None else float(args.fixed_cube_x_m),
        "fixed_cube_y_m": None if args.fixed_cube_y_m is None else float(args.fixed_cube_y_m),
        "fixed_push_dir": None if args.fixed_push_dir is None else [float(x) for x in args.fixed_push_dir],
        "tcp_height_mode": args.tcp_height_mode,
        "tcp_center_height_offset_m": float(args.tcp_center_height_offset_m),
        "through_target_mode": args.through_target_mode,
        "contact_controller_mode": args.contact_controller_mode,
        "contact_stop_target_mode": args.contact_stop_target_mode,
        "contact_detect_disp_m": float(args.contact_detect_disp_m),
        "contact_stop_disp_m": float(contact_stop_disp_m),
        "contact_overshoot_disp_m": float(args.contact_overshoot_disp_m),
        "contact_near_tcp_cube_dist_m": float(args.contact_near_tcp_cube_dist_m),
        "contact_near_joint_step_scale": float(args.contact_near_joint_step_scale),
        "contact_stop_joint_step_scale": float(args.contact_stop_joint_step_scale),
        "contact_retract_clearance_m": float(args.contact_retract_clearance_m),
        "reaction_disp_m": float(args.reaction_disp_m),
        "reaction_z_delta_m": float(args.reaction_z_delta_m),
        "reaction_speed_mps": float(args.reaction_speed_mps),
        "trajectory_variant": args.trajectory_variant,
        "base_total_steps_per_trial": base_total_steps,
        "v2_posx_total_steps_per_trial": v2_posx_total_steps,
        "v3_posx_total_steps_per_trial": v3_posx_total_steps,
        "v31_posx_total_steps_per_trial": v31_posx_total_steps,
        "v31_lowx_total_steps_per_trial": v31_lowx_total_steps,
        "posx_variant_total_steps_per_trial": variant_posx_total_steps,
        "posx_variant_env_count": posx_variant_env_count,
        "posx_variant_trial_count": posx_variant_trial_count,
        "v2_posx_env_count": posx_variant_env_count if args.trajectory_variant == "v2" else 0,
        "v3_posx_env_count": posx_variant_env_count if args.trajectory_variant == "v3" else 0,
        "v31_posx_env_count": posx_variant_env_count if args.trajectory_variant == "v3_1" else 0,
        "v2_posx_trial_count": posx_variant_trial_count if args.trajectory_variant == "v2" else 0,
        "v3_posx_trial_count": posx_variant_trial_count if args.trajectory_variant == "v3" else 0,
        "v31_posx_trial_count": posx_variant_trial_count if args.trajectory_variant == "v3_1" else 0,
        "v31_lowx_threshold_m": float(args.v31_lowx_threshold_m),
        "v31_lowx_trial_count": sum(int(r.get("v31_lowx_applied", 0)) for r in records),
        "episode_length_s": float(env_cfg.episode_length_s),
        "num_envs": n,
        "episodes": int(args.episodes),
        "trials": len(records),
        "total_steps_per_trial": total_steps,
        "precontact_clearance_m": float(args.precontact_clearance_m),
        "push_through_m": float(args.push_through_m),
        "base_lateral_offset_m": float(args.base_lateral_offset_m),
        "cube_push_target_disp_m": float(env_cfg.cube_push_target_disp_m),
        "cube_success_disp_m": float(env_cfg.cube_success_disp_m),
        "gate_disp_m": float(args.gate_disp_m),
        "max_diffik_joint_step_rad": float(args.max_diffik_joint_step_rad),
        "dls_lambda": float(args.dls_lambda),
        "arm_actuator_stiffness": arm_actuator.stiffness,
        "arm_actuator_damping": arm_actuator.damping,
        "arm_actuator_effort_limit_sim": arm_actuator.effort_limit_sim,
        "arm_actuator_velocity_limit_sim": arm_actuator.velocity_limit_sim,
        "arm_actuator_overrides": {
            "stiffness": None if args.arm_stiffness_override is None else float(args.arm_stiffness_override),
            "damping": None if args.arm_damping_override is None else float(args.arm_damping_override),
            "effort_limit_sim": (
                None
                if args.arm_effort_limit_sim_override is None
                else float(args.arm_effort_limit_sim_override)
            ),
            "velocity_limit_sim": (
                None
                if args.arm_velocity_limit_sim_override is None
                else float(args.arm_velocity_limit_sim_override)
            ),
        },
        "controlled_push_rate": rate("controlled_push"),
        "impact_outlier_rate": rate("impact_outlier"),
        "low_motion_rate": rate("low_motion"),
        "success_marker_rate": rate("success_marker"),
        "grasped_marker_rate": rate("grasped_marker"),
        "disp_along_push_mean_m": mean("disp_along_push_m"),
        "disp_over_object_size_mean": mean("disp_over_object_size"),
        "disp_xy_mean_m": mean("disp_xy_m"),
        "disp_ge_gate_rate": threshold_rate(float(args.gate_disp_m)),
        "disp_5_20mm_rate": disp_band_rate(0.005, 0.020),
        "disp_8_15mm_rate": disp_band_rate(0.008, 0.015),
        "controlled_disp_8_15mm_rate": disp_band_rate(0.008, 0.015, require_controlled=True),
        "max_disp_along_push_mean_m": mean("max_disp_along_push_m"),
        "max_disp_ge_gate_rate": max_threshold_rate(float(args.gate_disp_m)),
        "max_disp_8_15mm_rate": max_disp_band_rate(0.008, 0.015),
        "disp_ge_contact_overshoot_rate": threshold_rate(float(args.contact_overshoot_disp_m)),
        "max_disp_ge_contact_overshoot_rate": max_threshold_rate(float(args.contact_overshoot_disp_m)),
        "disp_threshold_rates": {
            f"{int(threshold_m * 1000)}mm": threshold_rate(threshold_m) for threshold_m in DISP_THRESHOLDS_M
        },
        "reaction_event_rate": rate("reaction_event"),
        "max_cube_z_delta_mean_m": mean("max_cube_z_delta_m"),
        "max_tip_angle_mean_deg": mean("max_tip_angle_deg"),
        "near_tcp_cube_seen_rate": rate("near_tcp_cube_seen"),
        "measured_contact_seen_rate": rate("measured_contact_seen"),
        "contact_stop_seen_rate": rate("contact_stop_seen"),
        "contact_overshoot_seen_rate": rate("contact_overshoot_seen"),
        "first_near_tcp_cube_step_mean": mean("first_near_tcp_cube_step"),
        "first_contact_step_mean": mean("first_contact_step"),
        "first_stop_step_mean": mean("first_stop_step"),
        "near_tcp_cube_step_rate_mean": mean("near_tcp_cube_step_rate"),
        "measured_contact_step_rate_mean": mean("measured_contact_step_rate"),
        "contact_stop_step_rate_mean": mean("contact_stop_step_rate"),
        "max_cube_speed_mean_mps": mean("max_cube_speed_mps"),
        "min_tcp_cube_dist_mean_m": mean("min_tcp_cube_dist_m"),
        "min_tcp_target_err_mean_m": mean("min_tcp_target_err_m"),
        "final_tcp_target_err_mean_m": mean("final_tcp_target_err_m"),
        "diffik_clip_rate_mean": mean("diffik_clip_rate"),
        "record_video": bool(args.record_video),
        "video_env_id": int(args.video_env_id) if args.record_video else None,
        "video_path": str(video_path) if video_path is not None else None,
        "video_frame_dir": str(video_frame_dir) if video_frame_dir is not None else None,
        "video_frame_count": int(video_frame_count),
        "trace_env_id": int(args.trace_env_id),
        "trace_env_ids": trace_env_ids,
        "trace_diffik_diagnostics": bool(args.trace_diffik_diagnostics),
        "trace_csv": str(trace_csv) if trace_csv is not None else None,
        "trace_frame_count": int(len(trace_records)),
        "link5_body_idx": int(link5_body_idx),
        "jacobi_body_idx": int(jacobi_body_idx),
        "jacobi_joint_ids": [int(idx) for idx in jacobi_joint_ids],
        "audit_thresholds": {
            "speed_p95_mps": AUDIT_SPEED_P95_MPS,
            "speed_p99_mps": AUDIT_SPEED_P99_MPS,
            "tip_p95_deg": AUDIT_TIP_P95_DEG,
            "tip_p99_deg": AUDIT_TIP_P99_DEG,
            "disp_xy_p99_m": AUDIT_DISP_XY_P99_M,
        },
        "out_csv": str(out_csv),
        "elapsed_s": time.time() - t0,
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        "[cube3cm_push_diffik_probe] "
        f"summary trials={summary['trials']} controlled_push_rate={summary['controlled_push_rate']:.6f} "
        f"impact_outlier_rate={summary['impact_outlier_rate']:.6f} low_motion_rate={summary['low_motion_rate']:.6f} "
        f"disp_along_push_mean_m={summary['disp_along_push_mean_m']:.6f} "
        f"disp_ge_gate_rate={summary['disp_ge_gate_rate']:.6f} "
        f"disp_8_15mm_rate={summary['disp_8_15mm_rate']:.6f} "
        f"max_disp_along_push_mean_m={summary['max_disp_along_push_mean_m']:.6f} "
        f"max_disp_ge_gate_rate={summary['max_disp_ge_gate_rate']:.6f} "
        f"reaction_event_rate={summary['reaction_event_rate']:.6f} "
        f"overshoot_ge_{int(float(args.contact_overshoot_disp_m) * 1000)}mm_rate="
        f"{summary['disp_ge_contact_overshoot_rate']:.6f} "
        f"measured_contact_seen_rate={summary['measured_contact_seen_rate']:.6f} "
        f"contact_stop_seen_rate={summary['contact_stop_seen_rate']:.6f} "
        f"disp_xy_mean_m={summary['disp_xy_mean_m']:.6f} posewrite_calls_during_rollout={summary['posewrite_calls_during_rollout']} "
        f"grasped_marker_rate={summary['grasped_marker_rate']:.6f}",
        flush=True,
    )
    if args.record_video:
        print(
            "[cube3cm_push_diffik_probe] "
            f"video_path={video_path} video_frame_dir={video_frame_dir} video_frame_count={video_frame_count} "
            f"video_env_id={args.video_env_id}",
            flush=True,
        )

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
