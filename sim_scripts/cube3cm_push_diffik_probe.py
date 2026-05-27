"""IsaacLab built-in Differential IK probe for 3cm cube push/tap.

This is a professor-branch diagnostic, separate from Track A grasp. It sends
end-effector targets near a 3cm cube, uses IsaacLab's DifferentialIKController
and live PhysX Jacobians to compute joint targets, and lets physics decide the
cube motion. It is not training, not dataset generation, and not a success claim.
"""
from __future__ import annotations

import argparse
import csv
import json
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--precontact_clearance_m", type=float, default=0.020)
    parser.add_argument("--tcp_top_margin_m", type=float, default=0.003)
    parser.add_argument("--push_through_m", type=float, default=0.030)
    parser.add_argument("--approach_steps", type=int, default=55)
    parser.add_argument("--push_steps", type=int, default=35)
    parser.add_argument("--post_steps", type=int, default=30)
    parser.add_argument("--settle_steps", type=int, default=8)
    parser.add_argument("--max_diffik_joint_step_rad", type=float, default=0.012)
    parser.add_argument("--dls_lambda", type=float, default=0.010)
    parser.add_argument("--trajectory_variant", choices=("v1", "v2", "v3"), default="v1")
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
    parser.add_argument("--trace_stride", type=int, default=4)
    parser.add_argument("--trace_csv", type=str, default="")
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
        CUBE_SIZE_M,
        RoArmCubePushEnvCfg,
    )

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
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.ik_endpoint_reset = False
    env_cfg.scripted_teacher_blend = 0.0
    env_cfg.action_scale = 0.0
    base_total_steps = int(args.approach_steps + args.push_steps + args.post_steps)
    v2_posx_total_steps = int(args.v2_posx_approach_steps + args.v2_posx_push_steps + args.v2_posx_post_steps)
    v3_posx_total_steps = int(args.v3_posx_approach_steps + args.v3_posx_push_steps + args.v3_posx_post_steps)
    variant_posx_total_steps = base_total_steps
    if args.trajectory_variant == "v2":
        variant_posx_total_steps = v2_posx_total_steps
    elif args.trajectory_variant == "v3":
        variant_posx_total_steps = v3_posx_total_steps
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
    half = 0.5 * float(CUBE_SIZE_M)

    print(
        "[cube3cm_push_diffik_probe] "
        f"isaac_run=YES num_envs={n} episodes={args.episodes} total_trials={n * args.episodes} "
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
        f"max_diffik_joint_step_rad={args.max_diffik_joint_step_rad:.6f} "
        f"v2_posx_max_diffik_joint_step_rad={args.v2_posx_max_diffik_joint_step_rad:.6f} "
        f"v3_posx_max_diffik_joint_step_rad={args.v3_posx_max_diffik_joint_step_rad:.6f} "
        f"dls_lambda={args.dls_lambda:.6f} env_auto_reset_disabled=YES "
        f"env_joint_delta_action_loop_bypassed=YES episode_length_s={env_cfg.episode_length_s:.3f}",
        flush=True,
    )

    records: list[dict[str, float | int | str]] = []
    trace_records: list[dict[str, float | int]] = []
    trace_env_ids: list[int] = []
    if args.trace_env_ids:
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
        approach_steps = torch.full((n,), int(args.approach_steps), dtype=torch.long, device=device)
        push_steps = torch.full((n,), int(args.push_steps), dtype=torch.long, device=device)
        post_steps = torch.full((n,), int(args.post_steps), dtype=torch.long, device=device)
        max_joint_step = torch.full((n,), float(args.max_diffik_joint_step_rad), dtype=torch.float32, device=device)
        precontact = torch.full((n,), float(args.precontact_clearance_m), dtype=torch.float32, device=device)
        push_through = torch.full((n,), float(args.push_through_m), dtype=torch.float32, device=device)
        tcp_top_margin = torch.full((n,), float(args.tcp_top_margin_m), dtype=torch.float32, device=device)
        lateral_offset = torch.zeros((n,), dtype=torch.float32, device=device)
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
        return {
            "posx": posx,
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
        pre = cube.clone()
        through = cube.clone()
        z = cube[:, 2] + half + traj["tcp_top_margin"]
        lateral = lateral_dir * traj["lateral_offset"].unsqueeze(-1)
        pre[:, 0:2] = cube[:, 0:2] - push_dir * (half + traj["precontact"]).unsqueeze(-1) + lateral
        through[:, 0:2] = cube[:, 0:2] + push_dir * (half + traj["push_through"]).unsqueeze(-1) + lateral
        pre[:, 2] = z
        through[:, 2] = z
        return pre + alpha.unsqueeze(-1) * (through - pre)

    def compute_diffik_joint_target(tcp_target_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_pos_w = inner._robot.data.root_pos_w
        root_quat_w = inner._robot.data.root_quat_w
        link5_pos_w = inner._robot.data.body_pos_w[:, link5_body_idx]
        link5_quat_w = inner._robot.data.body_quat_w[:, link5_body_idx]
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
        return joint_pos_des, link5_target_b, link5_pos_b

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
            cube_start_w = inner._cube_start_w.clone()
            push_dir = inner._push_dir_xy.clone()
            min_tcp_cube_dist = torch.full((n,), float("inf"), device=device)
            min_tcp_target_err = torch.full((n,), float("inf"), device=device)
            final_tcp_target_err = torch.zeros((n,), device=device)
            max_cube_speed = torch.zeros((n,), device=device)
            max_joint_delta_abs = torch.zeros((n,), device=device)
            clipped_steps = torch.zeros((n,), device=device)
            traj = build_trajectory_tensors()
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
                tcp_target_w = compute_tcp_targets(alpha, traj)
                joint_pos_des, _link5_target_b, _link5_pos_b = compute_diffik_joint_target(tcp_target_w)
                joint_pos_arm = inner._robot.data.joint_pos[:, arm_joint_ids]
                raw_delta = joint_pos_des - joint_pos_arm
                max_step = traj["max_joint_step"].unsqueeze(-1)
                clipped_delta = torch.maximum(torch.minimum(raw_delta, max_step), -max_step)
                clipped_steps += (torch.max(torch.abs(raw_delta), dim=-1).values > traj["max_joint_step"]).float()
                max_joint_delta_abs = torch.maximum(max_joint_delta_abs, torch.max(torch.abs(clipped_delta), dim=-1).values)

                target_full = inner.robot_dof_targets.clone()
                target_full[:, arm_joint_ids] = joint_pos_arm + clipped_delta
                target_full[:, inner.gripper_joint_idx] = 0.0
                target_full = torch.clamp(target_full, inner.robot_dof_lower_limits, inner.robot_dof_upper_limits)
                target_full[:, inner.gripper_joint_idx] = 0.0
                inner.robot_dof_targets[:] = target_full

                env.step(zero_action)
                inner._compute_intermediate_values()
                tcp_err = torch.norm(inner._tcp_pos_w - tcp_target_w, p=2, dim=-1)
                final_tcp_target_err = tcp_err
                min_tcp_target_err = torch.minimum(min_tcp_target_err, tcp_err)
                min_tcp_cube_dist = torch.minimum(
                    min_tcp_cube_dist,
                    torch.norm(inner._tcp_pos_w - inner._sponge_pos_w, p=2, dim=-1),
                )
                max_cube_speed = torch.maximum(max_cube_speed, torch.norm(inner._sponge.data.root_lin_vel_w, p=2, dim=-1))
                if trace_env_ids and episode == 0 and step % max(1, int(args.trace_stride)) == 0:
                    trace_frame = step // max(1, int(args.trace_stride))
                    for trace_idx in trace_env_ids:
                        trace_row: dict[str, float | int] = {
                            "frame": int(trace_frame),
                            "step": int(step),
                            "env_id": trace_idx,
                            "push_dx": float(push_dir[trace_idx, 0].detach().cpu().item()),
                            "push_dy": float(push_dir[trace_idx, 1].detach().cpu().item()),
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
                        }
                        for local_idx, joint_idx in enumerate(arm_joint_ids):
                            trace_row[f"arm_joint_{local_idx}_rad"] = float(
                                inner._robot.data.joint_pos[trace_idx, joint_idx].detach().cpu().item()
                            )
                        trace_row["gripper_joint_rad"] = float(
                            inner._robot.data.joint_pos[trace_idx, inner.gripper_joint_idx].detach().cpu().item()
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
                records.append(
                    {
                        "trial": len(records),
                        "episode": episode,
                        "env_id": idx,
                        "cube_x0_m": float((cube_start_w[idx, 0] - inner.scene.env_origins[idx, 0]).detach().cpu().item()),
                        "cube_y0_m": float((cube_start_w[idx, 1] - inner.scene.env_origins[idx, 1]).detach().cpu().item()),
                        "push_dx": float(push_dir[idx, 0].detach().cpu().item()),
                        "push_dy": float(push_dir[idx, 1].detach().cpu().item()),
                        "disp_along_push_m": float(terms["disp_along"][idx].detach().cpu().item()),
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
                            and args.trajectory_variant in {"v2", "v3"}
                        ),
                        "v2_posx_applied": int(
                            bool(traj["posx"][idx].detach().cpu().item()) and args.trajectory_variant == "v2"
                        ),
                        "v3_posx_applied": int(
                            bool(traj["posx"][idx].detach().cpu().item()) and args.trajectory_variant == "v3"
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
        "trajectory_variant": args.trajectory_variant,
        "base_total_steps_per_trial": base_total_steps,
        "v2_posx_total_steps_per_trial": v2_posx_total_steps,
        "v3_posx_total_steps_per_trial": v3_posx_total_steps,
        "posx_variant_total_steps_per_trial": variant_posx_total_steps,
        "posx_variant_env_count": posx_variant_env_count,
        "posx_variant_trial_count": posx_variant_trial_count,
        "v2_posx_env_count": posx_variant_env_count if args.trajectory_variant == "v2" else 0,
        "v3_posx_env_count": posx_variant_env_count if args.trajectory_variant == "v3" else 0,
        "v2_posx_trial_count": posx_variant_trial_count if args.trajectory_variant == "v2" else 0,
        "v3_posx_trial_count": posx_variant_trial_count if args.trajectory_variant == "v3" else 0,
        "episode_length_s": float(env_cfg.episode_length_s),
        "num_envs": n,
        "episodes": int(args.episodes),
        "trials": len(records),
        "total_steps_per_trial": total_steps,
        "precontact_clearance_m": float(args.precontact_clearance_m),
        "push_through_m": float(args.push_through_m),
        "max_diffik_joint_step_rad": float(args.max_diffik_joint_step_rad),
        "dls_lambda": float(args.dls_lambda),
        "controlled_push_rate": rate("controlled_push"),
        "impact_outlier_rate": rate("impact_outlier"),
        "low_motion_rate": rate("low_motion"),
        "success_marker_rate": rate("success_marker"),
        "grasped_marker_rate": rate("grasped_marker"),
        "disp_along_push_mean_m": mean("disp_along_push_m"),
        "disp_xy_mean_m": mean("disp_xy_m"),
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
        "trace_csv": str(trace_csv) if trace_csv is not None else None,
        "trace_frame_count": int(len(trace_records)),
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
