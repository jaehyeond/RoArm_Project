"""Branch-B dynamic-anchor pre-chain TCP-interface probe.

This unit stays outside the RoArm chain. It wraps the target-tracked dynamic
anchor in a tiny mock-TCP command interface so the next risk is tested before
any articulation integration: coordinate mapping plus multi-waypoint tracking.
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


def _parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected x,y,z vector, got {text!r}")
    return (parts[0], parts[1], parts[2])


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preclose_steps", type=int, default=5)
    ap.add_argument("--initial_hold_steps", type=int, default=40)
    ap.add_argument("--max_steps_per_waypoint", type=int, default=160)
    ap.add_argument("--target_settle_steps", type=int, default=5)
    ap.add_argument("--hold_steps_per_waypoint", type=int, default=20)
    ap.add_argument("--release_steps", type=int, default=120)
    ap.add_argument("--anchor_z", type=float, default=0.35)
    ap.add_argument("--anchor_mass", type=float, default=100.0)
    ap.add_argument("--tcp_to_anchor_offset", type=_parse_vec3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--waypoint", type=_parse_vec3, action="append")
    ap.add_argument("--target_kp", type=float, default=8.0)
    ap.add_argument("--max_cmd_speed", type=float, default=0.080)
    ap.add_argument("--stop_target_error_m", type=float, default=0.0015)
    ap.add_argument("--max_close_rel_m", type=float, default=0.005)
    ap.add_argument("--max_initial_hold_rel_m", type=float, default=0.005)
    ap.add_argument("--max_move_rel_m", type=float, default=0.005)
    ap.add_argument("--max_hold_rel_m", type=float, default=0.005)
    ap.add_argument("--max_final_target_error_m", type=float, default=0.003)
    ap.add_argument("--min_release_drop_m", type=float, default=0.050)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    waypoints = args.waypoint or [
        (0.010, 0.000, 0.005),
        (0.020, 0.006, 0.010),
        (0.012, -0.004, 0.012),
    ]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False, device="cpu")
    simulation_app = app_launcher.app

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext
    from pxr import Gf, Sdf, UsdPhysics

    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg

    print("[branch_b_dyn_iface] dynamic-anchor mock-TCP interface probe", flush=True)
    print(
        "[branch_b_dyn_iface] device=cpu chain_integration=NO transport=NO "
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
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=args.anchor_mass),
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
    tcp_to_anchor_offset = torch.tensor(args.tcp_to_anchor_offset, dtype=torch.float32)
    zero_lin = torch.zeros((1, 3), dtype=torch.float32)

    def write_pose(obj: RigidObject, pos: torch.Tensor, zero_velocity: bool = True) -> None:
        pose = torch.zeros((1, 7), dtype=torch.float32)
        pose[:, :3] = pos
        pose[:, 3] = 1.0
        obj.write_root_pose_to_sim(pose)
        if zero_velocity:
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32))

    def write_velocity(obj: RigidObject, lin_vel: torch.Tensor) -> None:
        vel = torch.zeros((1, 6), dtype=torch.float32)
        vel[:, :3] = lin_vel
        obj.write_root_velocity_to_sim(vel)

    def step_all() -> None:
        sim.step()
        anchor.update(sim.get_physics_dt())
        sponge.update(sim.get_physics_dt())

    def close_constraint() -> None:
        write_pose(anchor, start_anchor_pos)
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

    def obj_speed(obj: RigidObject) -> float:
        return float(torch.linalg.norm(obj.data.root_vel_w[0]).item())

    def target_error(obj: RigidObject, target_pos: torch.Tensor) -> float:
        return float(torch.linalg.norm(obj.data.root_pos_w[0].detach().cpu() - target_pos).item())

    def command_to_target(target_pos: torch.Tensor) -> tuple[float, list[float]]:
        remaining = target_pos.to(anchor.data.root_pos_w.device) - anchor.data.root_pos_w[0]
        remaining_norm = float(torch.linalg.norm(remaining).item())
        if remaining_norm <= args.stop_target_error_m:
            write_velocity(anchor, zero_lin)
            return remaining_norm, [0.0, 0.0, 0.0]
        speed = min(args.max_cmd_speed, args.target_kp * remaining_norm)
        lin_vel = (remaining / max(remaining_norm, 1.0e-9) * speed).reshape(1, 3).detach().cpu()
        write_velocity(anchor, lin_vel)
        return remaining_norm, _tensor_to_list(lin_vel[0])

    def fail_now(reason: str, **metrics: float) -> int:
        fields = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
        print(f"[branch_b_dyn_iface] EARLY_KILL reason={reason} {fields}".rstrip(), flush=True)
        print("[branch_b_dyn_iface] DYNAMIC_ANCHOR_INTERFACE_SUCCESS=NO", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)

    write_pose(anchor, start_anchor_pos)
    write_pose(sponge, start_anchor_pos)
    for _ in range(args.preclose_steps):
        step_all()

    close_constraint()
    step_all()
    close_rel = rel_dist()
    close_anchor_pos = anchor.data.root_pos_w[0].detach().cpu().clone()
    close_sponge_pos = sponge.data.root_pos_w[0].detach().cpu().clone()
    mock_tcp_start = close_anchor_pos - tcp_to_anchor_offset
    waypoint_tensors = [torch.tensor(wp, dtype=torch.float32) for wp in waypoints]
    print(
        "[branch_b_dyn_iface] close "
        f"rel={close_rel:.6f} joint_exists={stage.GetPrimAtPath(joint_path).IsValid()} "
        f"anchor_pos=({_tensor_to_list(close_anchor_pos)}) "
        f"sponge_pos=({_tensor_to_list(close_sponge_pos)}) "
        f"mock_tcp_start=({_tensor_to_list(mock_tcp_start)}) "
        f"tcp_to_anchor_offset=({_tensor_to_list(tcp_to_anchor_offset)}) "
        f"waypoints={len(waypoint_tensors)} target_kp={args.target_kp:.3f} "
        f"max_cmd_speed={args.max_cmd_speed:.3f}",
        flush=True,
    )
    if close_rel > args.max_close_rel_m:
        fail_now("close_rel", close_rel=close_rel)

    max_initial_hold_rel = close_rel
    for step in range(1, args.initial_hold_steps + 1):
        write_velocity(anchor, zero_lin)
        step_all()
        r = rel_dist()
        max_initial_hold_rel = max(max_initial_hold_rel, r)
        if step <= 5 or step % args.log_every == 0:
            print(
                f"[branch_b_dyn_iface] initial_hold step={step:03d} rel={r:.6f} "
                f"anchor_speed={obj_speed(anchor):.6f} sponge_speed={obj_speed(sponge):.6f}",
                flush=True,
            )
        if r > args.max_initial_hold_rel_m:
            fail_now("initial_hold_rel", max_initial_hold_rel=max_initial_hold_rel)

    max_move_rel = 0.0
    max_hold_rel = 0.0
    max_final_anchor_target_error = 0.0
    max_final_sponge_target_error = 0.0
    total_steps_used = 0

    for index, tcp_delta in enumerate(waypoint_tensors, start=1):
        tcp_target = mock_tcp_start + tcp_delta
        anchor_target = tcp_target + tcp_to_anchor_offset
        transform_error = float(torch.linalg.norm((tcp_target + tcp_to_anchor_offset) - anchor_target).item())
        print(
            f"[branch_b_dyn_iface] waypoint_start index={index} "
            f"tcp_delta=({_tensor_to_list(tcp_delta)}) tcp_target=({_tensor_to_list(tcp_target)}) "
            f"anchor_target=({_tensor_to_list(anchor_target)}) "
            f"transform_error={transform_error:.6f}",
            flush=True,
        )
        if transform_error > 1.0e-6:
            fail_now("interface_transform", transform_error=transform_error)

        settled_steps = 0
        steps_used = 0
        for step in range(1, args.max_steps_per_waypoint + 1):
            pre_cmd_error, cmd_vel = command_to_target(anchor_target)
            step_all()
            steps_used = step
            total_steps_used += 1
            r = rel_dist()
            anchor_err = target_error(anchor, anchor_target)
            sponge_err = target_error(sponge, anchor_target)
            max_move_rel = max(max_move_rel, r)
            if anchor_err <= args.stop_target_error_m and sponge_err <= args.stop_target_error_m:
                settled_steps += 1
            else:
                settled_steps = 0
            if step <= 5 or step % args.log_every == 0 or settled_steps >= args.target_settle_steps:
                print(
                    f"[branch_b_dyn_iface] waypoint_move index={index} step={step:03d} "
                    f"rel={r:.6f} pre_cmd_error={pre_cmd_error:.6f} "
                    f"anchor_target_error={anchor_err:.6f} sponge_target_error={sponge_err:.6f} "
                    f"cmd_vel=({cmd_vel}) settled_steps={settled_steps}",
                    flush=True,
                )
            if r > args.max_move_rel_m:
                fail_now("move_rel", waypoint=float(index), max_move_rel=max_move_rel)
            if settled_steps >= args.target_settle_steps:
                break

        write_velocity(anchor, zero_lin)
        final_anchor_error = target_error(anchor, anchor_target)
        final_sponge_error = target_error(sponge, anchor_target)
        max_final_anchor_target_error = max(max_final_anchor_target_error, final_anchor_error)
        max_final_sponge_target_error = max(max_final_sponge_target_error, final_sponge_error)
        print(
            f"[branch_b_dyn_iface] waypoint_target_stop index={index} steps_used={steps_used} "
            f"settled_steps={settled_steps} anchor_pos=({_tensor_to_list(anchor.data.root_pos_w[0])}) "
            f"sponge_pos=({_tensor_to_list(sponge.data.root_pos_w[0])}) "
            f"final_anchor_target_error={final_anchor_error:.6f} "
            f"final_sponge_target_error={final_sponge_error:.6f}",
            flush=True,
        )
        if (
            final_anchor_error > args.max_final_target_error_m
            or final_sponge_error > args.max_final_target_error_m
        ):
            fail_now(
                "waypoint_target_error",
                waypoint=float(index),
                final_anchor_target_error=final_anchor_error,
                final_sponge_target_error=final_sponge_error,
            )

        for hold_step in range(1, args.hold_steps_per_waypoint + 1):
            write_velocity(anchor, zero_lin)
            step_all()
            r = rel_dist()
            anchor_err = target_error(anchor, anchor_target)
            sponge_err = target_error(sponge, anchor_target)
            max_hold_rel = max(max_hold_rel, r)
            max_final_anchor_target_error = max(max_final_anchor_target_error, anchor_err)
            max_final_sponge_target_error = max(max_final_sponge_target_error, sponge_err)
            if hold_step <= 5 or hold_step % args.log_every == 0:
                print(
                    f"[branch_b_dyn_iface] waypoint_hold index={index} step={hold_step:03d} "
                    f"rel={r:.6f} anchor_target_error={anchor_err:.6f} "
                    f"sponge_target_error={sponge_err:.6f}",
                    flush=True,
                )
            if r > args.max_hold_rel_m:
                fail_now("waypoint_hold_rel", waypoint=float(index), max_hold_rel=max_hold_rel)
            if anchor_err > args.max_final_target_error_m or sponge_err > args.max_final_target_error_m:
                fail_now(
                    "waypoint_hold_target_error",
                    waypoint=float(index),
                    final_anchor_target_error=anchor_err,
                    final_sponge_target_error=sponge_err,
                )

    release_constraint()
    sponge.write_root_velocity_to_sim(torch.tensor([[0.0, 0.0, -0.20, 0.0, 0.0, 0.0]], dtype=torch.float32))
    release_start = sponge.data.root_pos_w[0].detach().cpu().clone()
    print(
        "[branch_b_dyn_iface] release "
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
                f"[branch_b_dyn_iface] release_settle step={step:03d} "
                f"rel={rel_dist():.6f} z={float(pos[2].item()):.6f} "
                f"sponge_speed={obj_speed(sponge):.6f}",
                flush=True,
            )

    final_pos = sponge.data.root_pos_w[0].detach().cpu()
    release_drop = float(release_start[2].item()) - float(final_pos[2].item())

    close_ok = close_rel <= args.max_close_rel_m
    initial_hold_ok = max_initial_hold_rel <= args.max_initial_hold_rel_m
    move_rel_ok = max_move_rel <= args.max_move_rel_m
    hold_rel_ok = max_hold_rel <= args.max_hold_rel_m
    anchor_target_ok = max_final_anchor_target_error <= args.max_final_target_error_m
    sponge_target_ok = max_final_sponge_target_error <= args.max_final_target_error_m
    release_ok = release_drop >= args.min_release_drop_m and max_release_rel > args.max_hold_rel_m
    success = (
        close_ok
        and initial_hold_ok
        and move_rel_ok
        and hold_rel_ok
        and anchor_target_ok
        and sponge_target_ok
        and release_ok
    )

    print(
        "[branch_b_dyn_iface] aggregate "
        f"waypoints={len(waypoint_tensors)} total_steps_used={total_steps_used} "
        f"close_rel={close_rel:.6f} max_initial_hold_rel={max_initial_hold_rel:.6f} "
        f"max_move_rel={max_move_rel:.6f} max_hold_rel={max_hold_rel:.6f} "
        f"max_final_anchor_target_error={max_final_anchor_target_error:.6f} "
        f"max_final_sponge_target_error={max_final_sponge_target_error:.6f} "
        f"target_error_threshold={args.max_final_target_error_m:.6f} "
        f"release_drop={release_drop:.6f} max_release_rel={max_release_rel:.6f}",
        flush=True,
    )
    print(
        "[branch_b_dyn_iface] gates "
        f"close_ok={_yes(close_ok)} initial_hold_ok={_yes(initial_hold_ok)} "
        f"move_rel_ok={_yes(move_rel_ok)} hold_rel_ok={_yes(hold_rel_ok)} "
        f"anchor_target_ok={_yes(anchor_target_ok)} sponge_target_ok={_yes(sponge_target_ok)} "
        f"release_ok={_yes(release_ok)} final_pos=({_tensor_to_list(final_pos)})",
        flush=True,
    )
    print(f"[branch_b_dyn_iface] DYNAMIC_ANCHOR_INTERFACE_SUCCESS={_yes(success)}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if success else 2)


if __name__ == "__main__":
    raise SystemExit(main())
