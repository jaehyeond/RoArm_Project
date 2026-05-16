"""CPU-only SurfaceGripper transport probe for RoArm G2-A.

This diagnostic is intentionally separate from the production B200 chain.
Isaac Lab SurfaceGripper is CPU-only in the installed Isaac Lab/Isaac Sim stack,
so this probes whether a proper gripper constraint can survive the v11 S1 long
transport before we consider replacing the local kinematic pose-write attach.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _quat_rotate_np(q_wxyz: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    v = np.asarray(v_xyz, dtype=np.float64)
    w, x, y, z = q
    qvec = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_skill2_steps", type=int, default=160)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--grip_distance", type=float, default=0.080)
    ap.add_argument("--force_limit", type=float, default=500.0)
    ap.add_argument("--parent_body", choices=("link5", "gripper_link"), default="link5")
    ap.add_argument("--offset_mode", choices=("tcp", "zero"), default="tcp")
    args = ap.parse_args()

    # Isaac Sim must be launched before importing Isaac Lab runtime modules.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False, device="cpu")
    simulation_app = app_launcher.app

    import torch
    import omni.kit.app
    import omni.kit.commands
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, SurfaceGripper, SurfaceGripperCfg
    from isaaclab.sim import SimulationContext
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Gf, UsdGeom

    from roarm_rl.chain_skills import (
        FOUR_SPONGE_SEED0_SOURCES,
        GRIPPER_LATCH_DEG,
        GRIPPER_OPEN_DEG,
        L1_SP1,
        SPONGE_CENTER_Z,
        TCP_RELEASE_ENTRY_Z,
        TrajectoryPlanner,
    )
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, TCP_LOCAL_OFFSET_M

    print("[surface_probe] CPU-only Isaac Lab SurfaceGripper transport diagnostic", flush=True)
    print("[surface_probe] This is not the production GPU chain.", flush=True)

    sim_cfg = sim_utils.SimulationCfg(device="cpu", dt=1 / 200, render_interval=2)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([0.8, -0.7, 0.7], [0.25, -0.05, 0.05])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/Light", light_cfg)
    sim_utils.create_prim("/World/envs/env_0", "Xform")

    env_cfg = RoArmStackEnvCfg()
    robot = Articulation(env_cfg.robot.replace(prim_path="/World/envs/env_0/Robot"))
    sponge = RigidObject(env_cfg.sponge.replace(prim_path="/World/envs/env_0/Sponge"))

    # Create a surface gripper under link5 and place it at the same local TCP
    # offset used by the existing chain. The gripper primitive must exist before
    # SurfaceGripper initializes on sim.reset().
    enable_extension("isaacsim.robot.surface_gripper")
    for _ in range(3):
        omni.kit.app.get_app().update()
    parent_prim_path = f"/World/envs/env_0/Robot/{args.parent_body}"
    result, prim = omni.kit.commands.execute("CreateSurfaceGripper", prim_path=parent_prim_path)
    if not result or prim is None:
        raise RuntimeError("CreateSurfaceGripper failed")
    gripper_prim_path = prim.GetPath().pathString
    local_offset = TCP_LOCAL_OFFSET_M if args.offset_mode == "tcp" else (0.0, 0.0, 0.0)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*local_offset))
    print(
        f"[surface_probe] created gripper prim={gripper_prim_path} "
        f"parent={parent_prim_path} offset_mode={args.offset_mode} "
        f"offset=({local_offset[0]:+.4f},{local_offset[1]:+.4f},{local_offset[2]:+.4f}) "
        f"grip_distance={args.grip_distance:.3f}",
        flush=True,
    )

    gripper_cfg = SurfaceGripperCfg(
        prim_path=gripper_prim_path,
        max_grip_distance=args.grip_distance,
        coaxial_force_limit=args.force_limit,
        shear_force_limit=args.force_limit,
        retry_interval=0.1,
    )
    surface_gripper = SurfaceGripper(gripper_cfg)

    sim.reset()
    robot.reset()
    sponge.reset()
    surface_gripper.reset()
    print("[surface_probe] sim reset complete", flush=True)

    device = torch.device("cpu")
    link5_idx = robot.find_bodies("link5")[0][0]
    joint_zeros = torch.zeros_like(robot.data.default_joint_pos)

    src_xy = FOUR_SPONGE_SEED0_SOURCES[0]
    sponge_xyz = (src_xy[0], src_xy[1], SPONGE_CENTER_Z)
    place_xyz = L1_SP1
    planner = TrajectoryPlanner(sponge_xyz=sponge_xyz, place_xyz=place_xyz)
    target_transport = np.array([place_xyz[0], place_xyz[1], TCP_RELEASE_ENTRY_Z], dtype=np.float64)
    target_world = np.array(place_xyz, dtype=np.float64)

    sponge_pose = torch.zeros((1, 7), device=device)
    sponge_pose[0, :3] = torch.tensor(sponge_xyz, dtype=torch.float32)
    sponge_pose[0, 3] = 1.0
    sponge.write_root_pose_to_sim(sponge_pose)
    sponge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

    home_q = torch.tensor([[0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0]], dtype=torch.float32)
    robot.write_joint_state_to_sim(home_q, joint_zeros)
    robot.set_joint_position_target(home_q)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    sponge.update(sim.get_physics_dt())
    surface_gripper.update(sim.get_physics_dt())

    def tcp_local_np() -> np.ndarray:
        link5_pos = robot.data.body_pos_w[0, link5_idx].detach().cpu().numpy()
        link5_quat = robot.data.body_quat_w[0, link5_idx].detach().cpu().numpy()
        return link5_pos + _quat_rotate_np(link5_quat, np.array(TCP_LOCAL_OFFSET_M))

    def sponge_local_np() -> np.ndarray:
        return sponge.data.root_pos_w[0].detach().cpu().numpy()

    def q_np() -> np.ndarray:
        return robot.data.joint_pos[0].detach().cpu().numpy()

    def step_robot(target_q: np.ndarray, steps: int, label: str, tol_rad: float = 0.03) -> tuple[int, float]:
        target_t = torch.tensor(target_q, dtype=torch.float32).unsqueeze(0)
        last_err = float("inf")
        ran = 0
        for s in range(steps):
            cur_q = q_np()
            last_err = float(np.max(np.abs(target_q[:5] - cur_q[:5])))
            if last_err < tol_rad:
                break
            robot.set_joint_position_target(target_t)
            robot.write_data_to_sim()
            surface_gripper.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            sponge.update(sim.get_physics_dt())
            surface_gripper.update(sim.get_physics_dt())
            ran += 1
            if s < 5 or ((s + 1) % args.log_every == 0):
                tcp = tcp_local_np()
                sp = sponge_local_np()
                d_tcp = float(np.linalg.norm(sp - tcp))
                tcp_err = float(np.linalg.norm(target_transport - tcp))
                print(
                    f"  [{label} s={s+1:3d}] arm_err={math.degrees(last_err):.2f}deg "
                    f"gripper_state={float(surface_gripper.state[0].item()):+.1f} "
                    f"tcp=({tcp[0]*1000:+.1f},{tcp[1]*1000:+.1f},{tcp[2]*1000:+.1f})mm "
                    f"sponge=({sp[0]*1000:+.1f},{sp[1]*1000:+.1f},{sp[2]*1000:+.1f})mm "
                    f"d_sponge_tcp={d_tcp*1000:.1f}mm tcp_err={tcp_err*1000:.1f}mm",
                    flush=True,
                )
        return ran, last_err

    # Reach and top-down grasp path.
    stages = [
        ("skill0_high", planner.target_q_skill0_high(), 120, 0.03),
        ("skill1a_hover", planner.target_q_skill1a_to_hover(), 120, 0.03),
        ("skill1b1_z50", planner.target_q_skill1b1_z50(), 120, 0.03),
        ("skill1b2_z40", planner.target_q_skill1b2_z40(), 120, 0.03),
        ("skill1b3_z33", planner.target_q_skill1b3_z33(), 120, 0.03),
    ]
    for label, target, steps, tol in stages:
        ran, err = step_robot(target, steps, label, tol)
        print(f"[surface_probe] {label} done steps={ran} arm_err={math.degrees(err):.2f}deg", flush=True)

    # Close gripper joint and command SurfaceGripper to close.
    close_target = planner.target_q_skill1c_close()
    close_t = torch.tensor(close_target, dtype=torch.float32).unsqueeze(0)
    close_cmd = torch.ones((1,), dtype=torch.float32)
    surface_gripper.set_grippers_command(close_cmd)
    close_detect = -1
    for s in range(80):
        robot.set_joint_position_target(close_t)
        robot.write_data_to_sim()
        surface_gripper.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        sponge.update(sim.get_physics_dt())
        surface_gripper.update(sim.get_physics_dt())
        if s < 5 or (s + 1) % 10 == 0:
            tcp = tcp_local_np()
            sp = sponge_local_np()
            print(
                f"  [close s={s+1:3d}] gripper_state={float(surface_gripper.state[0].item()):+.1f} "
                f"q_grip={math.degrees(q_np()[5]):.2f}deg "
                f"d_sponge_tcp={np.linalg.norm(sp-tcp)*1000:.1f}mm",
                flush=True,
            )
        if close_detect < 0 and float(surface_gripper.state[0].item()) > 0.5:
            close_detect = s + 1
            break
    print(f"[surface_probe] close_detect_step={close_detect}", flush=True)

    # Long Skill 2 transport with physical gripper constraint, no pose-writing.
    print("[surface_probe] Skill2 long transport with SurfaceGripper constraint", flush=True)
    skill2_target = planner.target_q_skill2()
    ran, err = step_robot(skill2_target, args.max_skill2_steps, "skill2_surface", 0.03)
    tcp = tcp_local_np()
    sp = sponge_local_np()
    dxy_pre_release = float(np.linalg.norm(sp[:2] - target_world[:2]))
    dz_pre_release = float(abs(sp[2] - target_world[2]))
    tcp_err = float(np.linalg.norm(target_transport - tcp))
    print(
        f"[surface_probe] skill2 done steps={ran} arm_err={math.degrees(err):.2f}deg "
        f"tcp_err={tcp_err*1000:.1f}mm gripper_state={float(surface_gripper.state[0].item()):+.1f} "
        f"sponge=({sp[0]*1000:+.1f},{sp[1]*1000:+.1f},{sp[2]*1000:+.1f})mm "
        f"target=({target_world[0]*1000:+.1f},{target_world[1]*1000:+.1f},{target_world[2]*1000:+.1f})mm "
        f"d_xy_pre_release={dxy_pre_release*1000:.1f}mm d_z_pre_release={dz_pre_release*1000:.1f}mm",
        flush=True,
    )

    # Release and settle.
    open_cmd = -torch.ones((1,), dtype=torch.float32)
    surface_gripper.set_grippers_command(open_cmd)
    release_step = -1
    for s in range(60):
        robot.set_joint_position_target(torch.tensor(skill2_target, dtype=torch.float32).unsqueeze(0))
        robot.write_data_to_sim()
        surface_gripper.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        sponge.update(sim.get_physics_dt())
        surface_gripper.update(sim.get_physics_dt())
        if release_step < 0 and float(surface_gripper.state[0].item()) < -0.5:
            release_step = s + 1
        if s < 5 or (s + 1) % 10 == 0:
            sp = sponge_local_np()
            dxy = float(np.linalg.norm(sp[:2] - target_world[:2]))
            dz = float(abs(sp[2] - target_world[2]))
            print(
                f"  [release s={s+1:3d}] gripper_state={float(surface_gripper.state[0].item()):+.1f} "
                f"sponge=({sp[0]*1000:+.1f},{sp[1]*1000:+.1f},{sp[2]*1000:+.1f})mm "
                f"d_xy={dxy*1000:.1f}mm d_z={dz*1000:.1f}mm",
                flush=True,
            )

    sp = sponge_local_np()
    final_dxy = float(np.linalg.norm(sp[:2] - target_world[:2]))
    final_dz = float(abs(sp[2] - target_world[2]))
    success = final_dxy < 0.030 and final_dz < 0.025
    print(
        f"[surface_probe] release_step={release_step} final_d_xy={final_dxy*1000:.1f}mm "
        f"final_d_z={final_dz*1000:.1f}mm SURFACE_PROBE_SUCCESS={'YES' if success else 'NO'}",
        flush=True,
    )

    simulation_app.close()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
