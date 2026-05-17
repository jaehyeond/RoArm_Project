"""Branch-B SurfaceGripper unit probe for the RoArm sponge.

This is intentionally a unit test, not a chain integration.  It uses Isaac
Lab's canonical SurfaceGripper test asset so the first Branch-B question is
about the gripper API/constraint itself, not another RoArm parent/offset guess.
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
    ap.add_argument("--close_steps", type=int, default=80)
    ap.add_argument("--hold_steps", type=int, default=120)
    ap.add_argument("--max_grip_distance", type=float, default=0.120)
    ap.add_argument("--force_limit", type=float, default=500.0)
    ap.add_argument("--retry_interval", type=float, default=0.05)
    ap.add_argument("--object_z_offset", type=float, default=0.0)
    ap.add_argument("--closed_frac_thresh", type=float, default=0.95)
    ap.add_argument("--max_drift_m", type=float, default=0.020)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False, device="cpu")
    simulation_app = app_launcher.app

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, SurfaceGripper, SurfaceGripperCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from pxr import Usd, UsdGeom

    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg

    print("[branch_b_unit] SurfaceGripper canonical-asset sponge unit probe", flush=True)
    print("[branch_b_unit] device=cpu chain_integration=NO transport=NO", flush=True)

    sim_cfg = sim_utils.SimulationCfg(device="cpu", dt=1 / 200, render_interval=2)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, -1.0, 0.9], [0.0, 0.0, 0.45])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
    )
    sim_utils.create_prim("/World/Env_0", "Xform")

    gripper_usd = f"{ISAACLAB_NUCLEUS_DIR}/Tests/SurfaceGripper/test_gripper.usd"
    robot_cfg = ArticulationCfg(
        prim_path="/World/Env_0/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=gripper_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            )
        },
    )
    robot = Articulation(robot_cfg)

    env_cfg = RoArmStackEnvCfg()
    sponge = RigidObject(env_cfg.sponge.replace(prim_path="/World/Env_0/Sponge"))

    gripper_path = "/World/Env_0/Robot/Gripper/SurfaceGripper"
    surface_gripper = SurfaceGripper(
        SurfaceGripperCfg(
            prim_path=gripper_path,
            max_grip_distance=args.max_grip_distance,
            coaxial_force_limit=args.force_limit,
            shear_force_limit=args.force_limit,
            retry_interval=args.retry_interval,
        )
    )

    sim.reset()
    robot.reset()
    sponge.reset()
    surface_gripper.reset()

    stage = sim.stage
    gripper_prim = stage.GetPrimAtPath(gripper_path)
    if not gripper_prim.IsValid():
        raise RuntimeError(f"missing SurfaceGripper prim: {gripper_path}")
    world_tf = UsdGeom.Xformable(gripper_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    gripper_pos = world_tf.ExtractTranslation()
    # Isaac Lab's own SurfaceGripper test places the graspable object at the
    # canonical gripper asset root height, not at the nested SurfaceGripper prim
    # transform.  Keep this concrete authored placement as the first hypothesis.
    object_pos = torch.tensor([[0.0, 0.0, 0.5 + args.object_z_offset]], dtype=torch.float32)

    sponge_pose = torch.zeros((1, 7), dtype=torch.float32)
    sponge_pose[:, :3] = object_pos
    sponge_pose[:, 3] = 1.0
    sponge.write_root_pose_to_sim(sponge_pose)
    sponge.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32))

    sim.step()
    robot.update(sim.get_physics_dt())
    sponge.update(sim.get_physics_dt())
    surface_gripper.update(sim.get_physics_dt())

    def sponge_pos() -> torch.Tensor:
        return sponge.data.root_pos_w[0].detach().cpu()

    def state_value() -> float:
        return float(surface_gripper.state[0].detach().cpu().item())

    def gripped_count() -> int:
        view = surface_gripper.gripper_view
        if not hasattr(view, "get_gripped_objects"):
            return -1
        try:
            objects = view.get_gripped_objects()
        except TypeError:
            objects = view.get_gripped_objects([0])
        except Exception:
            return -2
        if objects is None:
            return 0
        if isinstance(objects, (list, tuple)):
            return len(objects)
        try:
            return len(objects)
        except Exception:
            return -3

    initial_sponge = sponge_pos()
    print(
        "[branch_b_unit] authored_asset="
        f"{gripper_usd} gripper_path={gripper_path} sponge_prim=/World/Env_0/Sponge",
        flush=True,
    )
    print(
        "[branch_b_unit] reset "
        f"surface_gripper_prim_pos=({_tensor_to_list(torch.tensor([gripper_pos[0], gripper_pos[1], gripper_pos[2]]))}) "
        f"sponge_pos=({_tensor_to_list(initial_sponge)}) "
        f"state={state_value():+.1f} gripped_count={gripped_count()}",
        flush=True,
    )

    close_cmd = torch.ones((1,), dtype=torch.float32)
    surface_gripper.set_grippers_command(close_cmd)
    closed_detect_step = -1
    close_states: list[float] = []
    for step in range(1, args.close_steps + 1):
        surface_gripper.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        sponge.update(sim.get_physics_dt())
        surface_gripper.update(sim.get_physics_dt())
        st = state_value()
        close_states.append(st)
        if closed_detect_step < 0 and st > 0.5:
            closed_detect_step = step
        if step <= 5 or step % args.log_every == 0 or st > 0.5:
            drift = float(torch.linalg.norm(sponge_pos() - initial_sponge).item())
            print(
                f"[branch_b_unit] close step={step:03d} state={st:+.1f} "
                f"gripped_count={gripped_count()} sponge_drift={drift:.5f}",
                flush=True,
            )
        if closed_detect_step >= 0:
            break

    hold_states: list[float] = []
    hold_gripped: list[int] = []
    max_drift = 0.0
    max_speed = 0.0
    for step in range(1, args.hold_steps + 1):
        surface_gripper.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        sponge.update(sim.get_physics_dt())
        surface_gripper.update(sim.get_physics_dt())
        st = state_value()
        hold_states.append(st)
        hold_gripped.append(gripped_count())
        vel = sponge.data.root_vel_w[0].detach().cpu()
        max_speed = max(max_speed, float(torch.linalg.norm(vel).item()))
        max_drift = max(max_drift, float(torch.linalg.norm(sponge_pos() - initial_sponge).item()))
        if step <= 5 or step % args.log_every == 0:
            print(
                f"[branch_b_unit] hold step={step:03d} state={st:+.1f} "
                f"gripped_count={hold_gripped[-1]} sponge_drift={max_drift:.5f} "
                f"speed_norm={max_speed:.5f}",
                flush=True,
            )

    closed_frac = sum(1 for st in hold_states if st > 0.5) / max(1, len(hold_states))
    gripped_positive_frac = (
        sum(1 for count in hold_gripped if count > 0) / max(1, len(hold_gripped))
        if any(count >= 0 for count in hold_gripped)
        else -1.0
    )
    final_pos = sponge_pos()
    final_state = state_value()
    success = (
        closed_detect_step >= 0
        and closed_frac >= args.closed_frac_thresh
        and max_drift <= args.max_drift_m
        and (gripped_positive_frac < 0.0 or gripped_positive_frac >= args.closed_frac_thresh)
    )

    print(
        "[branch_b_unit] aggregate "
        f"closed_detect_step={closed_detect_step} closed_frac={closed_frac:.4f} "
        f"gripped_positive_frac={gripped_positive_frac:.4f} max_drift={max_drift:.5f} "
        f"max_speed_norm={max_speed:.5f}",
        flush=True,
    )
    print(
        "[branch_b_unit] final "
        f"state={final_state:+.1f} gripped_count={gripped_count()} "
        f"sponge_pos=({_tensor_to_list(final_pos)})",
        flush=True,
    )
    print(f"[branch_b_unit] SURFACE_UNIT_SUCCESS={'YES' if success else 'NO'}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # In this B200 headless stack the surface-gripper extension can hang during
    # Kit shutdown after all diagnostic lines are printed.  Exit directly so the
    # smoke has a reliable process-level verdict.
    os._exit(0 if success else 2)


if __name__ == "__main__":
    raise SystemExit(main())
