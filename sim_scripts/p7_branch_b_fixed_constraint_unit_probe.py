"""Branch-B explicitly authored fixed-constraint unit probe.

This is a pre-chain unit test.  It provides a tiny close/release API around a
USD fixed joint between a kinematic anchor body and the RoArm sponge.  The goal
is to prove stable attached hold before any transport, then prove release.
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
    ap.add_argument("--hold_steps", type=int, default=120)
    ap.add_argument("--release_steps", type=int, default=120)
    ap.add_argument("--anchor_z", type=float, default=0.35)
    ap.add_argument("--max_hold_rel_m", type=float, default=0.005)
    ap.add_argument("--max_hold_drift_m", type=float, default=0.010)
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

    print("[branch_b_fixed] explicit fixed-constraint unit probe", flush=True)
    print("[branch_b_fixed] device=cpu chain_integration=NO transport=NO surface_gripper=NO", flush=True)

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

    anchor_pos = torch.tensor([[0.0, 0.0, args.anchor_z]], dtype=torch.float32)
    object_pos = anchor_pos.clone()

    joint = None

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
        nonlocal joint
        write_pose(anchor, anchor_pos, zero_velocity=False)
        write_pose(sponge, object_pos)
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(anchor_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(sponge_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.GetJointEnabledAttr().Set(True)

    def release_constraint() -> None:
        nonlocal joint
        if stage.GetPrimAtPath(joint_path).IsValid():
            stage.RemovePrim(joint_path)
        joint = None

    def rel_dist() -> float:
        return float(torch.linalg.norm(anchor.data.root_pos_w[0] - sponge.data.root_pos_w[0]).item())

    def sponge_speed() -> float:
        return float(torch.linalg.norm(sponge.data.root_vel_w[0]).item())

    write_pose(anchor, anchor_pos, zero_velocity=False)
    write_pose(sponge, object_pos)
    for _ in range(args.preclose_steps):
        step_all()

    pre_rel = rel_dist()
    pre_pos = sponge.data.root_pos_w[0].detach().cpu().clone()
    print(
        "[branch_b_fixed] reset "
        f"anchor_pos=({_tensor_to_list(anchor.data.root_pos_w[0])}) "
        f"sponge_pos=({_tensor_to_list(sponge.data.root_pos_w[0])}) "
        f"rel={pre_rel:.6f} joint_exists={stage.GetPrimAtPath(joint_path).IsValid()}",
        flush=True,
    )

    close_constraint()
    step_all()
    close_rel = rel_dist()
    close_pos = sponge.data.root_pos_w[0].detach().cpu().clone()
    print(
        "[branch_b_fixed] close "
        f"rel={close_rel:.6f} joint_exists={stage.GetPrimAtPath(joint_path).IsValid()} "
        f"sponge_pos=({_tensor_to_list(sponge.data.root_pos_w[0])})",
        flush=True,
    )

    max_hold_rel = close_rel
    max_hold_drift = 0.0
    max_hold_speed = 0.0
    for step in range(1, args.hold_steps + 1):
        step_all()
        r = rel_dist()
        drift = float(torch.linalg.norm(sponge.data.root_pos_w[0].detach().cpu() - close_pos).item())
        speed = sponge_speed()
        max_hold_rel = max(max_hold_rel, r)
        max_hold_drift = max(max_hold_drift, drift)
        max_hold_speed = max(max_hold_speed, speed)
        if step <= 5 or step % args.log_every == 0:
            print(
                f"[branch_b_fixed] hold step={step:03d} rel={r:.6f} "
                f"drift={drift:.6f} speed_norm={speed:.6f} "
                f"joint_exists={stage.GetPrimAtPath(joint_path).IsValid()}",
                flush=True,
            )

    hold_ok = max_hold_rel <= args.max_hold_rel_m and max_hold_drift <= args.max_hold_drift_m
    release_constraint()
    sponge.write_root_velocity_to_sim(torch.tensor([[0.0, 0.0, -0.20, 0.0, 0.0, 0.0]], dtype=torch.float32))
    release_start = sponge.data.root_pos_w[0].detach().cpu().clone()
    print(
        "[branch_b_fixed] release "
        f"joint_exists={stage.GetPrimAtPath(joint_path).IsValid()} "
        f"start_pos=({_tensor_to_list(release_start)})",
        flush=True,
    )

    min_release_z = float(release_start[2].item())
    max_release_rel = 0.0
    for step in range(1, args.release_steps + 1):
        step_all()
        pos = sponge.data.root_pos_w[0].detach().cpu()
        min_release_z = min(min_release_z, float(pos[2].item()))
        max_release_rel = max(max_release_rel, rel_dist())
        if step <= 5 or step % args.log_every == 0:
            print(
                f"[branch_b_fixed] release_settle step={step:03d} "
                f"rel={rel_dist():.6f} z={float(pos[2].item()):.6f} speed_norm={sponge_speed():.6f}",
                flush=True,
            )

    final_pos = sponge.data.root_pos_w[0].detach().cpu()
    release_drop = float(release_start[2].item()) - float(final_pos[2].item())
    release_ok = release_drop >= args.min_release_drop_m and max_release_rel > args.max_hold_rel_m
    success = hold_ok and release_ok

    print(
        "[branch_b_fixed] aggregate "
        f"pre_rel={pre_rel:.6f} close_rel={close_rel:.6f} "
        f"max_hold_rel={max_hold_rel:.6f} max_hold_drift={max_hold_drift:.6f} "
        f"max_hold_speed={max_hold_speed:.6f} release_drop={release_drop:.6f} "
        f"max_release_rel={max_release_rel:.6f}",
        flush=True,
    )
    print(
        "[branch_b_fixed] gates "
        f"hold_ok={'YES' if hold_ok else 'NO'} release_ok={'YES' if release_ok else 'NO'} "
        f"final_pos=({_tensor_to_list(final_pos)})",
        flush=True,
    )
    print(f"[branch_b_fixed] FIXED_UNIT_SUCCESS={'YES' if success else 'NO'}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if success else 2)


if __name__ == "__main__":
    raise SystemExit(main())
