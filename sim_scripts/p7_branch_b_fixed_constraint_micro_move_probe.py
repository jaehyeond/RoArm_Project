"""Branch-B fixed-constraint micro-move/hold/release unit probe.

This is still a pre-chain unit test. It moves a kinematic anchor by a tiny
scripted displacement while a USD fixed joint is attached to the RoArm sponge,
then checks hold stability and release. No RoArm chain, SurfaceGripper, P7
reward, or launch defaults are touched.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _tensor_to_list(value) -> list[float]:
    return [float(x) for x in value.detach().cpu().flatten().tolist()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preclose_steps", type=int, default=5)
    ap.add_argument("--initial_hold_steps", type=int, default=40)
    ap.add_argument("--move_steps", type=int, default=80)
    ap.add_argument("--post_move_hold_steps", type=int, default=80)
    ap.add_argument("--release_steps", type=int, default=120)
    ap.add_argument("--anchor_z", type=float, default=0.35)
    ap.add_argument("--move_dx", type=float, default=0.020)
    ap.add_argument("--move_dy", type=float, default=0.000)
    ap.add_argument("--move_dz", type=float, default=0.010)
    ap.add_argument("--max_close_rel_m", type=float, default=0.005)
    ap.add_argument("--max_hold_rel_m", type=float, default=0.005)
    ap.add_argument("--max_move_rel_m", type=float, default=0.005)
    ap.add_argument("--max_post_move_rel_m", type=float, default=0.005)
    ap.add_argument("--min_release_drop_m", type=float, default=0.050)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False, device="cpu")
    simulation_app = app_launcher.app

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext
    from pxr import Gf, Sdf, UsdPhysics

    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg

    print("[branch_b_micro] fixed-constraint micro-move/hold/release unit probe", flush=True)
    print(
        "[branch_b_micro] device=cpu chain_integration=NO transport=NO "
        "surface_gripper=NO p7_training=NO",
        flush=True,
    )

    sim_cfg = sim_utils.SimulationCfg(device="cpu", dt=1 / 200, render_interval=2)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([0.9, -0.8, 0.65], [0.0, 0.0, 0.25])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
    )
    sim_utils.create_prim("/World/Env_0", "Xform")

    anchor_path = "/World/Env_0/Anchor"
    sponge_path = "/World/Env_0/Sponge"
    joint_path = "/World/Env_0/AttachFixedJoint"

    anchor = RigidObject(
        RigidObjectCfg(
            prim_path=anchor_path,
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 0.02, 0.02),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args.anchor_z)),
        )
    )
    env_cfg = RoArmStackEnvCfg()
    sponge = RigidObject(env_cfg.sponge.replace(prim_path=sponge_path))

    stage = sim.stage
    sim.reset()
    anchor.reset()
    sponge.reset()

    start_anchor_pos = torch.tensor([[0.0, 0.0, args.anchor_z]], dtype=torch.float32)
    move_delta = torch.tensor([[args.move_dx, args.move_dy, args.move_dz]], dtype=torch.float32)
    target_anchor_pos = start_anchor_pos + move_delta

    def write_pose(obj: RigidObject, pos: torch.Tensor, zero_velocity: bool = True) -> None:
        pose = torch.zeros((1, 7), dtype=torch.float32)
        pose[:, :3] = pos
        pose[:, 3] = 1.0
        obj.write_root_pose_to_sim(pose)
        if zero_velocity:
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32))

    def step_all() -> None:
        sim.step()
        anchor.update(sim.get_physics_dt())
        sponge.update(sim.get_physics_dt())

    def close_constraint() -> None:
        write_pose(anchor, start_anchor_pos, zero_velocity=False)
        write_pose(sponge, start_anchor_pos)
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(anchor_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(sponge_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.GetJointEnabledAttr().Set(True)

    def release_constraint() -> None:
        if stage.GetPrimAtPath(joint_path).IsValid():
            stage.RemovePrim(joint_path)

    def rel_dist() -> float:
        return float(torch.linalg.norm(anchor.data.root_pos_w[0] - sponge.data.root_pos_w[0]).item())

    def sponge_speed() -> float:
        return float(torch.linalg.norm(sponge.data.root_vel_w[0]).item())

    write_pose(anchor, start_anchor_pos, zero_velocity=False)
    write_pose(sponge, start_anchor_pos)
    for _ in range(args.preclose_steps):
        step_all()

    pre_rel = rel_dist()
    print(
        "[branch_b_micro] reset "
        f"anchor_pos=({_tensor_to_list(anchor.data.root_pos_w[0])}) "
        f"sponge_pos=({_tensor_to_list(sponge.data.root_pos_w[0])}) "
        f"rel={pre_rel:.6f} joint_exists={stage.GetPrimAtPath(joint_path).IsValid()} "
        f"move_delta=({_tensor_to_list(move_delta[0])})",
        flush=True,
    )

    close_constraint()
    step_all()
    close_rel = rel_dist()
    close_pos = sponge.data.root_pos_w[0].detach().cpu().clone()
    print(
        "[branch_b_micro] close "
        f"rel={close_rel:.6f} joint_exists={stage.GetPrimAtPath(joint_path).IsValid()} "
        f"sponge_pos=({_tensor_to_list(sponge.data.root_pos_w[0])})",
        flush=True,
    )

    max_initial_hold_rel = close_rel
    max_initial_hold_drift = 0.0
    for step in range(1, args.initial_hold_steps + 1):
        step_all()
        r = rel_dist()
        drift = float(torch.linalg.norm(sponge.data.root_pos_w[0].detach().cpu() - close_pos).item())
        max_initial_hold_rel = max(max_initial_hold_rel, r)
        max_initial_hold_drift = max(max_initial_hold_drift, drift)
        if step <= 5 or step % args.log_every == 0:
            print(
                f"[branch_b_micro] initial_hold step={step:03d} rel={r:.6f} "
                f"drift={drift:.6f} speed_norm={sponge_speed():.6f} "
                f"joint_exists={stage.GetPrimAtPath(joint_path).IsValid()}",
                flush=True,
            )

    max_move_rel = 0.0
    max_move_speed = 0.0
    for step in range(1, args.move_steps + 1):
        alpha = step / max(args.move_steps, 1)
        desired_anchor_pos = start_anchor_pos + move_delta * alpha
        write_pose(anchor, desired_anchor_pos, zero_velocity=False)
        step_all()
        r = rel_dist()
        max_move_rel = max(max_move_rel, r)
        max_move_speed = max(max_move_speed, sponge_speed())
        if step <= 5 or step % args.log_every == 0 or step == args.move_steps:
            print(
                f"[branch_b_micro] move step={step:03d} rel={r:.6f} "
                f"anchor_pos=({_tensor_to_list(anchor.data.root_pos_w[0])}) "
                f"sponge_pos=({_tensor_to_list(sponge.data.root_pos_w[0])}) "
                f"speed_norm={sponge_speed():.6f} joint_exists={stage.GetPrimAtPath(joint_path).IsValid()}",
                flush=True,
            )

    post_move_start = sponge.data.root_pos_w[0].detach().cpu().clone()
    max_post_move_rel = 0.0
    max_post_move_drift = 0.0
    for step in range(1, args.post_move_hold_steps + 1):
        write_pose(anchor, target_anchor_pos, zero_velocity=False)
        step_all()
        r = rel_dist()
        drift = float(torch.linalg.norm(sponge.data.root_pos_w[0].detach().cpu() - post_move_start).item())
        max_post_move_rel = max(max_post_move_rel, r)
        max_post_move_drift = max(max_post_move_drift, drift)
        if step <= 5 or step % args.log_every == 0:
            print(
                f"[branch_b_micro] post_move_hold step={step:03d} rel={r:.6f} "
                f"drift={drift:.6f} speed_norm={sponge_speed():.6f} "
                f"joint_exists={stage.GetPrimAtPath(joint_path).IsValid()}",
                flush=True,
            )

    release_constraint()
    sponge.write_root_velocity_to_sim(torch.tensor([[0.0, 0.0, -0.20, 0.0, 0.0, 0.0]], dtype=torch.float32))
    release_start = sponge.data.root_pos_w[0].detach().cpu().clone()
    print(
        "[branch_b_micro] release "
        f"joint_exists={stage.GetPrimAtPath(joint_path).IsValid()} "
        f"start_pos=({_tensor_to_list(release_start)})",
        flush=True,
    )

    max_release_rel = 0.0
    for step in range(1, args.release_steps + 1):
        step_all()
        max_release_rel = max(max_release_rel, rel_dist())
        if step <= 5 or step % args.log_every == 0:
            pos = sponge.data.root_pos_w[0].detach().cpu()
            print(
                f"[branch_b_micro] release_settle step={step:03d} "
                f"rel={rel_dist():.6f} z={float(pos[2].item()):.6f} "
                f"speed_norm={sponge_speed():.6f}",
                flush=True,
            )

    final_pos = sponge.data.root_pos_w[0].detach().cpu()
    release_drop = float(release_start[2].item()) - float(final_pos[2].item())

    close_ok = close_rel <= args.max_close_rel_m
    initial_hold_ok = max_initial_hold_rel <= args.max_hold_rel_m
    move_ok = max_move_rel <= args.max_move_rel_m
    post_move_ok = max_post_move_rel <= args.max_post_move_rel_m
    release_ok = release_drop >= args.min_release_drop_m and max_release_rel > args.max_post_move_rel_m
    success = close_ok and initial_hold_ok and move_ok and post_move_ok and release_ok

    print(
        "[branch_b_micro] aggregate "
        f"pre_rel={pre_rel:.6f} close_rel={close_rel:.6f} "
        f"max_initial_hold_rel={max_initial_hold_rel:.6f} "
        f"max_initial_hold_drift={max_initial_hold_drift:.6f} "
        f"max_move_rel={max_move_rel:.6f} max_move_speed={max_move_speed:.6f} "
        f"max_post_move_rel={max_post_move_rel:.6f} "
        f"max_post_move_drift={max_post_move_drift:.6f} "
        f"release_drop={release_drop:.6f} max_release_rel={max_release_rel:.6f}",
        flush=True,
    )
    print(
        "[branch_b_micro] gates "
        f"close_ok={'YES' if close_ok else 'NO'} "
        f"initial_hold_ok={'YES' if initial_hold_ok else 'NO'} "
        f"move_ok={'YES' if move_ok else 'NO'} "
        f"post_move_ok={'YES' if post_move_ok else 'NO'} "
        f"release_ok={'YES' if release_ok else 'NO'} "
        f"final_pos=({_tensor_to_list(final_pos)})",
        flush=True,
    )
    print(f"[branch_b_micro] FIXED_MICRO_MOVE_SUCCESS={'YES' if success else 'NO'}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if success else 2)


if __name__ == "__main__":
    raise SystemExit(main())
