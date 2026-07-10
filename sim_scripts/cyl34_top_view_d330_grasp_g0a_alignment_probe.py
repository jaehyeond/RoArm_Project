#!/usr/bin/env python3
"""D330 G0a cylinder D34xH90 alignment probe.

This is the failable runtime probe registered in D329.  It redefines the G0a
object from the tap-track 10cm cube to the professor-directed cylinder
D34xH90, while keeping the D325 reachable tangent-minus pose family, D327 2mm
alignment standoff, fixed pose, friction, mass placeholder, and no-close
alignment-only contract.  It does not advance to G0b, close the gripper, grasp,
lift, train RL/PPO, randomize positions, render trajectories, or touch B200.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import sim_scripts.cube10cm_top_view_d327_grasp_g0a_standoff_execution_probe as d327
from roarm_rl.viz_debug import draw_frames, frame_from_axes, log_rerun, snapshot_frame_plot
from sim_scripts.cube10cm_top_view_d323_grasp_g0a_frame_repair_probe import (
    FIXED_JAW_FACE_LOCAL_M,
    HOME_DEG,
    _fk_runtime_tcp,
    _quat_wxyz_to_rot,
    _solve_runtime_ik,
)


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d330"
DEFAULT_ROBOT_USD = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"
DEFAULT_URDF = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"

ARM_JOINT_NAMES = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
]
GRIPPER_JOINT_NAME = "link5_to_gripper_link"
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
D330_CONTACT_LINKS = ["link4", "link5", "gripper_link"]

ADOPTED_TANGENT_SIGN = -1.0
ALIGNMENT_STANDOFF_M = 0.002
FIXED_JAW_FACE_OFFSET_M = 0.008
TCP_POS_GATE_M = 0.005
JAW_TANGENT_GATE_DEG = 15.0
GAP_GATE_M = 0.005
OBJECT_DISP_GATE_M = 0.005
TOP_CLEARANCE_GATE_M = 0.015


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _unit(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return arr / norm


def _unit_xy_from_object(obj_local: np.ndarray) -> np.ndarray:
    radial = np.asarray([float(obj_local[0]), float(obj_local[1]), 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(radial[:2]))
    if norm <= 1.0e-9:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return radial / norm


def _joint_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[idx]) for idx, name in enumerate(ALL_JOINT_NAMES)}


def _version_manifest() -> dict[str, Any]:
    mods = ["omni.kit.app", "pxr.Usd", "isaaclab", "isaacsim", "numpy", "psutil", "rerun"]
    out: dict[str, Any] = {"python": sys.version, "python_executable": sys.executable}
    for name in mods:
        try:
            mod = importlib.import_module(name)
            out[name] = {"file": getattr(mod, "__file__", None), "version": getattr(mod, "__version__", None)}
        except Exception as exc:
            out[name] = {"error": repr(exc)}
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        out["omni.kit.app"]["kit_version"] = app.get_build_version()
    except Exception as exc:
        out.setdefault("omni.kit.app_runtime", {})["error"] = repr(exc)
    return out


def _target_frame(target_tcp: np.ndarray, tangent_axis: np.ndarray) -> dict[str, Any]:
    return frame_from_axes(
        "d330_target_tcp",
        target_tcp,
        x_axis=tangent_axis,
        z_axis=[0.0, 0.0, 1.0],
        role="target",
        label="target TCP / jaw tangent",
        metadata={"tool_z_policy": "free", "object": "cylinder_d34_h90"},
    )


def _actual_frame(name: str, tcp: np.ndarray, rot: np.ndarray, *, label: str, role: str = "actual") -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "position": [float(v) for v in tcp.tolist()],
        "axes": {
            "x": [float(v) for v in rot[:, 0].tolist()],
            "y": [float(v) for v in rot[:, 1].tolist()],
            "z": [float(v) for v in rot[:, 2].tolist()],
        },
        "role": role,
    }


def _object_frame(obj_center: np.ndarray) -> dict[str, Any]:
    return frame_from_axes(
        "cylinder_object_frame",
        obj_center,
        x_axis=[1.0, 0.0, 0.0],
        z_axis=[0.0, 0.0, 1.0],
        role="object",
        label="cylinder D34xH90",
    )


def _fixed_jaw_frame(pos: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray) -> dict[str, Any]:
    return frame_from_axes(
        "fixed_jaw_face",
        pos,
        x_axis=x_axis,
        z_axis=z_axis,
        role="fixed_jaw",
        label="fixed jaw face",
    )


def _contact_point_frame(pos: np.ndarray, tangent_axis: np.ndarray) -> dict[str, Any]:
    return frame_from_axes(
        "cylinder_side_contact_point",
        pos,
        x_axis=tangent_axis,
        z_axis=[0.0, 0.0, 1.0],
        role="cube_face",
        label="cylinder side proxy point",
    )


def _frames_for_state(
    *,
    obj_center: np.ndarray,
    target_tcp: np.ndarray,
    tangent: np.ndarray,
    actual_tcp: np.ndarray,
    actual_rot: np.ndarray,
    fixed_jaw_face: np.ndarray,
    contact_point: np.ndarray,
    commanded_tcp: np.ndarray | None = None,
    commanded_rot: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    frames = [
        _target_frame(target_tcp, tangent),
        _actual_frame("actual_tcp_link5", actual_tcp, actual_rot, label="actual TCP/link5"),
        _fixed_jaw_frame(fixed_jaw_face, actual_rot[:, 0], actual_rot[:, 2]),
        _contact_point_frame(contact_point, tangent),
        _object_frame(obj_center),
    ]
    if commanded_tcp is not None and commanded_rot is not None:
        frames.append(
            _actual_frame(
                "commanded_tcp_link5",
                commanded_tcp,
                commanded_rot,
                label="commanded TCP/link5",
                role="candidate",
            )
        )
    return frames


def _horizontal_axis_error_deg(axis: np.ndarray, target_axis: np.ndarray) -> float:
    axis_h = np.asarray([float(axis[0]), float(axis[1]), 0.0], dtype=np.float64)
    target_h = np.asarray([float(target_axis[0]), float(target_axis[1]), 0.0], dtype=np.float64)
    if float(np.linalg.norm(axis_h)) <= 1.0e-9:
        return 180.0
    return d327._horizontal_axis_error_deg(axis_h, target_h)


def _evaluate_alignment(
    *,
    trial: int,
    obj_center: np.ndarray,
    obj_radius_m: float,
    obj_height_m: float,
    target_tcp: np.ndarray,
    tangent: np.ndarray,
    actual_tcp: np.ndarray,
    link5_rot: np.ndarray,
    obj_start_w: np.ndarray,
    obj_final_w: np.ndarray,
    target_arm: np.ndarray,
    actual_arm: np.ndarray,
    ik_failure_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj_top_z = float(obj_center[2] + 0.5 * obj_height_m)
    link5_x = link5_rot[:, 0]
    tcp_err_m = float(np.linalg.norm(actual_tcp - target_tcp))
    tangent_err_deg = float(_horizontal_axis_error_deg(link5_x, tangent))
    fixed_jaw_face = actual_tcp + link5_rot @ FIXED_JAW_FACE_LOCAL_M
    cylinder_side_plane = obj_center - tangent * obj_radius_m
    signed_gap_m = float(np.dot(cylinder_side_plane[:2] - fixed_jaw_face[:2], tangent[:2]))
    penetration_m = max(0.0, -signed_gap_m)
    contact_point = fixed_jaw_face + tangent * signed_gap_m
    top_clearance_m = float(obj_top_z - contact_point[2])
    disp_m = float(np.linalg.norm(obj_final_w[:2] - obj_start_w[:2]))
    arm_joint_err_rad = float(np.max(np.abs(actual_arm - target_arm)))

    pass_tcp = tcp_err_m <= TCP_POS_GATE_M
    pass_tangent = tangent_err_deg <= JAW_TANGENT_GATE_DEG
    pass_gap = 0.0 <= signed_gap_m <= GAP_GATE_M
    pass_pen = penetration_m <= 1.0e-6
    pass_height = top_clearance_m >= TOP_CLEARANCE_GATE_M
    pass_disp = disp_m < OBJECT_DISP_GATE_M
    pass_all = bool(pass_tcp and pass_tangent and pass_gap and pass_pen and pass_height and pass_disp)

    commanded_tcp, _link5_pos, commanded_rot = _fk_runtime_tcp(np.degrees(np.r_[target_arm, 0.0]))
    frames = _frames_for_state(
        obj_center=obj_center,
        target_tcp=target_tcp,
        tangent=tangent,
        actual_tcp=actual_tcp,
        actual_rot=link5_rot,
        fixed_jaw_face=fixed_jaw_face,
        contact_point=contact_point,
        commanded_tcp=commanded_tcp,
        commanded_rot=commanded_rot,
    )
    radius_gap_m = float(np.linalg.norm(fixed_jaw_face[:2] - obj_center[:2]) - obj_radius_m)
    row = {
        "trial": int(trial),
        "target_tcp_x_m": float(target_tcp[0]),
        "target_tcp_y_m": float(target_tcp[1]),
        "target_tcp_z_m": float(target_tcp[2]),
        "actual_tcp_x_m": float(actual_tcp[0]),
        "actual_tcp_y_m": float(actual_tcp[1]),
        "actual_tcp_z_m": float(actual_tcp[2]),
        "commanded_tcp_x_m": float(commanded_tcp[0]),
        "commanded_tcp_y_m": float(commanded_tcp[1]),
        "commanded_tcp_z_m": float(commanded_tcp[2]),
        "tcp_pose_error_mm": float(tcp_err_m * 1000.0),
        "commanded_tcp_pose_error_mm": float(np.linalg.norm(commanded_tcp - target_tcp) * 1000.0),
        "jaw_tangent_error_deg": tangent_err_deg,
        "fixed_jaw_face_gap_mm": float(signed_gap_m * 1000.0),
        "fixed_jaw_radius_gap_mm": float(radius_gap_m * 1000.0),
        "fixed_jaw_penetration_mm": float(penetration_m * 1000.0),
        "contact_point_z_m": float(contact_point[2]),
        "contact_point_below_top_mm": float(top_clearance_m * 1000.0),
        "object_disp_xy_mm": float(disp_m * 1000.0),
        "ik_failure_steps": int(ik_failure_steps),
        "arm_joint_err_max_rad": arm_joint_err_rad,
        "pass_tcp_pose": bool(pass_tcp),
        "pass_jaw_tangent": bool(pass_tangent),
        "pass_fixed_jaw_gap": bool(pass_gap),
        "pass_no_penetration": bool(pass_pen),
        "pass_contact_height": bool(pass_height),
        "pass_object_displacement": bool(pass_disp),
        "pass_all": pass_all,
    }
    return row, frames


def _write_snapshot(
    path: Path,
    *,
    obj_center: np.ndarray,
    obj_diameter_m: float,
    obj_height_m: float,
    frames: list[dict[str, Any]],
    row: dict[str, Any],
    title: str,
) -> str:
    snapshot_frame_plot(
        path,
        frames,
        cube={"center": obj_center.tolist(), "size": float(obj_diameter_m)},
        title=title,
        annotations=[
            "D330 G0a: cylinder D34xH90; small box is diameter proxy, not cylinder mesh.",
            f"cylinder height = {obj_height_m * 1000.0:.1f} mm",
            f"pos err = {row['tcp_pose_error_mm']:.3f} mm",
            f"cmd pos err = {row['commanded_tcp_pose_error_mm']:.3f} mm",
            f"jaw tangent err = {row['jaw_tangent_error_deg']:.3f} deg",
            f"plane gap = {row['fixed_jaw_face_gap_mm']:.3f} mm, radius gap = {row['fixed_jaw_radius_gap_mm']:.3f} mm",
            f"penetration = {row['fixed_jaw_penetration_mm']:.3f} mm",
            f"top clearance = {row['contact_point_below_top_mm']:.3f} mm",
            f"object disp = {row['object_disp_xy_mm']:.3f} mm",
        ],
    )
    return _rel(path)


def _configure_env_cfg(args: argparse.Namespace, num_envs: int) -> Any:
    import isaaclab.sim as sim_utils
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
    from roarm_rl.roarm_stack_env import TABLE_Z

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.robot.spawn.activate_contact_sensors = True
    env_cfg.episode_length_s = float(args.episode_length_s)
    env_cfg.cube_x_min = float(args.object_x)
    env_cfg.cube_x_max = float(args.object_x)
    env_cfg.cube_y_min = float(args.object_y)
    env_cfg.cube_y_max = float(args.object_y)
    env_cfg.cube_size_x_m = float(args.cylinder_diameter_m)
    env_cfg.cube_size_y_m = float(args.cylinder_diameter_m)
    env_cfg.cube_size_z_m = float(args.cylinder_height_m)
    old_spawn = env_cfg.sponge.spawn
    env_cfg.sponge.spawn = sim_utils.CylinderCfg(
        radius=0.5 * float(args.cylinder_diameter_m),
        height=float(args.cylinder_height_m),
        axis="Z",
        rigid_props=old_spawn.rigid_props,
        mass_props=sim_utils.MassPropertiesCfg(mass=float(args.object_mass_kg)),
        collision_props=old_spawn.collision_props,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=float(args.static_friction),
            dynamic_friction=float(args.dynamic_friction),
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.86, 0.55, 0.20),
            metallic=0.0,
        ),
    )
    env_cfg.sponge.init_state.pos = (
        float(args.object_x),
        float(args.object_y),
        TABLE_Z + 0.5 * float(args.cylinder_height_m),
    )
    env_cfg.sponge.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    env_cfg.fixed_push_dir_x = 1.0
    env_cfg.fixed_push_dir_y = 0.0
    env_cfg.ik_endpoint_reset = False
    env_cfg.rl_action_mode = "joint_delta"
    env_cfg.bc_teacher_checkpoint_path = ""
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    return env_cfg


def _make_env(args: argparse.Namespace, num_envs: int) -> tuple[Any, Any, Any]:
    import gymnasium as gym
    import torch
    import roarm_rl  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env_cfg = _configure_env_cfg(args, num_envs)
    env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    zero = torch.zeros((inner.num_envs, inner.cfg.action_space), device=inner.device)
    return env, inner, zero


def _reset_env(inner: Any, env: Any, zero: Any) -> tuple[Any, Any, np.ndarray]:
    import torch

    inner.episode_length_buf[:] = inner.max_episode_length
    env.step(zero)
    inner._compute_intermediate_values()
    env_ids = torch.arange(inner.num_envs, device=inner.device, dtype=torch.long)
    origins = inner.scene.env_origins[env_ids]
    obj_start_w = inner._sponge_pos_w.detach().clone()
    obj_start_local = (obj_start_w - origins).detach().cpu().numpy()
    return origins, obj_start_w, obj_start_local


def _target_info(obj_start_local: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    standoff_m = float(args.alignment_standoff_m)
    if not math.isclose(standoff_m, ALIGNMENT_STANDOFF_M, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("D330 alignment standoff is fixed at 0.002m; tuning is not allowed")
    radius_m = 0.5 * float(args.cylinder_diameter_m)
    infos: list[dict[str, Any]] = []
    for idx in range(obj_start_local.shape[0]):
        obj_local = np.asarray(obj_start_local[idx], dtype=np.float64)
        radial = _unit_xy_from_object(obj_local)
        tangent = np.asarray([-radial[1], radial[0], 0.0], dtype=np.float64) * ADOPTED_TANGENT_SIGN
        grasp_flush_tangent_offset_m = radius_m - FIXED_JAW_FACE_OFFSET_M
        alignment_tangent_offset_m = grasp_flush_tangent_offset_m + standoff_m
        radial_center_offset_m = radius_m - float(args.radial_tip_past_near_face_m)
        final_tcp = obj_local.copy()
        final_tcp -= radial * radial_center_offset_m
        final_tcp -= tangent * alignment_tangent_offset_m
        final_tcp[2] = obj_local[2]
        infos.append(
            {
                "radial_axis": radial,
                "tangent_axis": tangent,
                "target_x_axis": tangent,
                "target_z_axis": radial,
                "target_tcp": final_tcp,
                "tangent_center_offset_m": float(alignment_tangent_offset_m),
                "grasp_flush_tangent_offset_m": float(grasp_flush_tangent_offset_m),
                "alignment_standoff_m": float(standoff_m),
                "radial_tip_past_near_face_m": float(args.radial_tip_past_near_face_m),
                "radial_center_offset_m": float(radial_center_offset_m),
                "target_formula": "TCP = cyl_center - radial*(D/2-10mm) - tangent*(D/2-8mm+2mm); TCP z = cyl center",
            }
        )
    return infos


def _state_eval_rows(
    inner: Any,
    origins: Any,
    obj_start_w: Any,
    obj_start_local: np.ndarray,
    target_info: list[dict[str, Any]],
    target_arm_by_env: np.ndarray,
    ik_failure_counts: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    inner._compute_intermediate_values()
    body_quat_np = inner._robot.data.body_quat_w.detach().cpu().numpy()
    tcp_local = (inner._tcp_pos_w - origins).detach().cpu().numpy()
    obj_final_w = inner._sponge_pos_w.detach().clone()
    obj_final_w_np = obj_final_w.detach().cpu().numpy()
    obj_start_w_np = obj_start_w.detach().cpu().numpy()
    joint_pos = inner._robot.data.joint_pos.detach().cpu().numpy()
    actual_arm = joint_pos[:, inner._bc_arm_joint_ids]

    rows: list[dict[str, Any]] = []
    frame_sets: list[list[dict[str, Any]]] = []
    for idx, info in enumerate(target_info):
        target_tcp = np.asarray(info["target_tcp"], dtype=np.float64)
        tangent = _unit(np.asarray(info["target_x_axis"], dtype=np.float64))
        link5_quat = body_quat_np[idx, inner.link5_idx]
        link5_rot = _quat_wxyz_to_rot(link5_quat)
        row, frames = _evaluate_alignment(
            trial=idx + 1,
            obj_center=obj_start_local[idx],
            obj_radius_m=0.5 * float(args.cylinder_diameter_m),
            obj_height_m=float(args.cylinder_height_m),
            target_tcp=target_tcp,
            tangent=tangent,
            actual_tcp=tcp_local[idx],
            link5_rot=link5_rot,
            obj_start_w=obj_start_w_np[idx],
            obj_final_w=obj_final_w_np[idx],
            target_arm=target_arm_by_env[idx],
            actual_arm=actual_arm[idx],
            ik_failure_steps=int(ik_failure_counts[idx]),
        )
        rows.append(row)
        frame_sets.append(frames)
    return rows, frame_sets


def _try_make_sensor(prim_path: str, *, filters: list[str] | None = None, track_points: bool = False) -> tuple[Any | None, dict[str, Any]]:
    try:
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        cfg = ContactSensorCfg(
            prim_path=prim_path,
            history_length=1,
            update_period=0.0,
            force_threshold=0.0,
            track_contact_points=bool(track_points),
            filter_prim_paths_expr=list(filters or []),
        )
        sensor = ContactSensor(cfg)
        # This diagnostic creates sensors after the DirectRLEnv has already
        # started playing. Isaac Lab normally initializes sensors from a PLAY
        # callback, so initialize immediately for this probe-local witness.
        sensor._initialize_impl()
        sensor._is_initialized = True
        return sensor, {
            "ok": True,
            "prim_path": cfg.prim_path,
            "filter_prim_paths_expr": list(cfg.filter_prim_paths_expr),
            "body_names": list(getattr(sensor, "body_names", [])),
            "num_bodies": int(getattr(sensor, "num_bodies", 0)),
            "num_instances": int(getattr(sensor, "num_instances", 0)),
            "filter_count": int(getattr(getattr(sensor, "contact_physx_view", None), "filter_count", 0)),
        }
    except Exception as exc:
        return None, {"ok": False, "prim_path": prim_path, "filters": list(filters or []), "error": repr(exc)}


def _make_contact_witnesses(inner: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    witnesses: dict[str, Any] = {}
    status: dict[str, Any] = {}
    net_sensor, net_status = _try_make_sensor("/World/envs/env_.*/Robot/.*")
    witnesses["robot_net"] = net_sensor
    status["robot_net"] = net_status

    selected = []
    body_names = list(getattr(inner._robot, "body_names", []))
    for name in body_names:
        if name in {"link4", "link5", "gripper_link"} or "gripper" in name:
            selected.append(name)
    if not selected:
        selected = body_names[-4:]

    filter_paths = ["/World/envs/env_.*/Sponge", "/World/envs/env_.*/TapTable"]
    filtered: dict[str, Any] = {}
    status["filtered_link_sensors"] = {}
    for body_name in selected:
        sensor, sensor_status = _try_make_sensor(f"/World/envs/env_.*/Robot/{body_name}", filters=filter_paths)
        filtered[body_name] = sensor
        status["filtered_link_sensors"][body_name] = sensor_status
    witnesses["filtered_links"] = filtered
    status["filter_labels"] = ["cylinder_sponge", "tap_table"]
    return witnesses, status


def _contact_force_row(witnesses: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    net_sensor = witnesses.get("robot_net")
    if net_sensor is not None:
        try:
            net_sensor.update(0.0, force_recompute=True)
            forces = net_sensor.data.net_forces_w
            if forces is not None:
                arr = forces[0].detach().cpu().numpy().astype(np.float64)
                norms = np.linalg.norm(arr, axis=-1)
                body_names = list(getattr(net_sensor, "body_names", []))
                idx = int(np.argmax(norms)) if norms.size else -1
                out["available"] = True
                out["robot_net_max_force_n"] = float(np.max(norms)) if norms.size else 0.0
                out["robot_net_sum_force_n"] = float(np.sum(norms)) if norms.size else 0.0
                out["robot_net_argmax_body_index"] = idx
                out["robot_net_argmax_body_name"] = body_names[idx] if 0 <= idx < len(body_names) else ""
        except Exception as exc:
            out["robot_net_error"] = repr(exc)

    link_forces: dict[str, Any] = {}
    filter_labels = list(status.get("filter_labels", ["cylinder_sponge", "tap_table"]))
    for link_name, sensor in dict(witnesses.get("filtered_links", {})).items():
        if sensor is None:
            continue
        try:
            sensor.update(0.0, force_recompute=True)
            item: dict[str, Any] = {}
            if sensor.data.net_forces_w is not None:
                net = sensor.data.net_forces_w[0].detach().cpu().numpy().astype(np.float64)
                item["net_force_n"] = float(np.linalg.norm(net.reshape(-1, 3), axis=-1).max())
            matrix = sensor.data.force_matrix_w
            if matrix is not None:
                mat = matrix[0].detach().cpu().numpy().astype(np.float64).reshape(-1, len(filter_labels), 3)
                norms = np.linalg.norm(mat, axis=-1)
                item["filtered_force_by_target_n"] = {
                    filter_labels[idx]: float(np.max(norms[:, idx])) for idx in range(len(filter_labels))
                }
            link_forces[link_name] = item
            out["available"] = True
        except Exception as exc:
            link_forces[link_name] = {"error": repr(exc)}
    if link_forces:
        out["filtered_links"] = link_forces
    return out


def _contact_trace_stats(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "trace_steps": len(trace_rows),
        "robot_net_max_force_n": 0.0,
        "robot_net_first_contact_step": -1,
        "robot_net_argmax_body_name": "",
        "filtered_max_force_by_link_target_n": {},
        "filtered_first_contact_by_link_target_step": {},
    }
    for row in trace_rows:
        step = int(row.get("step", -1))
        contact = dict(row.get("contact_force", {}))
        net_force = float(contact.get("robot_net_max_force_n", 0.0) or 0.0)
        if net_force > float(stats["robot_net_max_force_n"]):
            stats["robot_net_max_force_n"] = net_force
            stats["robot_net_argmax_body_name"] = str(contact.get("robot_net_argmax_body_name", ""))
        if stats["robot_net_first_contact_step"] < 0 and net_force > 1.0e-6:
            stats["robot_net_first_contact_step"] = step
        for link_name, item in dict(contact.get("filtered_links", {})).items():
            for target_name, force in dict(item.get("filtered_force_by_target_n", {})).items():
                key = f"{link_name}->{target_name}"
                val = float(force)
                prior = float(stats["filtered_max_force_by_link_target_n"].get(key, 0.0))
                if val > prior:
                    stats["filtered_max_force_by_link_target_n"][key] = val
                if key not in stats["filtered_first_contact_by_link_target_step"] and val > 1.0e-6:
                    stats["filtered_first_contact_by_link_target_step"][key] = step
    return stats


def _trace_stats(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace_rows:
        return {"trace_steps": 0}
    first = trace_rows[0]
    mid = trace_rows[len(trace_rows) // 2]
    last = trace_rows[-1]
    pos = np.asarray([float(row["tcp_pose_error_mm"]) for row in trace_rows], dtype=np.float64)
    joint_err = np.asarray([float(row["arm_joint_err_max_rad"]) for row in trace_rows], dtype=np.float64)
    sat_rates = []
    for row in trace_rows:
        torque = row.get("torque", {})
        if isinstance(torque, dict) and "saturation_rate" in torque:
            sat_rates.append(float(torque["saturation_rate"]))
    out = {
        "trace_steps": len(trace_rows),
        "first_tcp_error_mm": float(first["tcp_pose_error_mm"]),
        "mid_tcp_error_mm": float(mid["tcp_pose_error_mm"]),
        "final_tcp_error_mm": float(last["tcp_pose_error_mm"]),
        "min_tcp_error_mm": float(np.min(pos)),
        "final_minus_mid_tcp_error_mm": float(last["tcp_pose_error_mm"] - mid["tcp_pose_error_mm"]),
        "max_joint_err_rad": float(np.max(joint_err)),
        "final_joint_err_rad": float(last["arm_joint_err_max_rad"]),
        "torque_saturation_rate_max": float(max(sat_rates)) if sat_rates else None,
        "torque_saturation_rate_final": float(sat_rates[-1]) if sat_rates else None,
    }
    out["contact"] = _contact_trace_stats(trace_rows)
    return out


def _run_gate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    env, inner, zero = _make_env(args, int(args.num_trials))
    try:
        origins, obj_start_w, obj_start_local = _reset_env(inner, env, zero)
        target_info = _target_info(obj_start_local, args)
        witnesses, contact_status = _make_contact_witnesses(inner)

        joint_targets_full = inner._robot.data.joint_pos.detach().clone()
        q_targets_kin = np.zeros((inner.num_envs, 6), dtype=np.float64)
        q_targets_kin[:, :5] = joint_targets_full[:, inner._bc_arm_joint_ids].detach().cpu().numpy().astype(np.float64)
        q_targets_kin[:, 5] = joint_targets_full[:, inner.gripper_joint_idx].detach().cpu().numpy().astype(np.float64)
        ik_failure_counts = np.zeros(inner.num_envs, dtype=np.int64)
        final_target_arm = q_targets_kin[:, :5].copy()
        trace_rows: list[dict[str, Any]] = []

        total_steps = int(args.approach_steps) + int(args.hold_steps)
        with torch.inference_mode():
            for step in range(total_steps):
                phase = "approach" if step < int(args.approach_steps) else "hold"
                alpha = min(1.0, (step + 1) / float(max(1, int(args.approach_steps))))
                for idx, info in enumerate(target_info):
                    final_tcp = np.asarray(info["target_tcp"], dtype=np.float64)
                    radial = np.asarray(info["radial_axis"], dtype=np.float64)
                    pre_tcp = final_tcp - radial * float(args.pre_clearance_m)
                    target_tcp = pre_tcp + alpha * (final_tcp - pre_tcp)
                    if phase == "hold":
                        target_tcp = final_tcp
                    result = _solve_runtime_ik(
                        target_tcp,
                        np.degrees(q_targets_kin[idx]),
                        target_x_axis=None,
                        target_z_axis=None,
                        max_iter=120,
                        pos_tol_mm=1.0,
                    )
                    if not bool(result["converged"]):
                        ik_failure_counts[idx] += 1
                    q_targets_kin[idx] = np.radians(np.asarray(result["q_deg"], dtype=np.float64))

                targets_t = joint_targets_full.detach().clone()
                targets_t[:, inner._bc_arm_joint_ids] = torch.tensor(
                    q_targets_kin[:, :5], device=inner.device, dtype=torch.float32
                )
                targets_t[:, inner.gripper_joint_idx] = 0.0
                joint_targets_full = targets_t.detach().clone()
                inner._external_joint_targets_override = targets_t
                env.step(zero)
                inner._compute_intermediate_values()
                final_target_arm = q_targets_kin[:, :5].copy()

                body_quat_np = inner._robot.data.body_quat_w.detach().cpu().numpy()
                tcp_local = (inner._tcp_pos_w - origins).detach().cpu().numpy()
                actual_all = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
                commanded_all = targets_t[0].detach().cpu().numpy().astype(np.float64)
                actual_arm = actual_all[inner._bc_arm_joint_ids]
                target_arm = commanded_all[inner._bc_arm_joint_ids]
                info = target_info[0]
                target_tcp = np.asarray(info["target_tcp"], dtype=np.float64)
                tangent = _unit(np.asarray(info["target_x_axis"], dtype=np.float64))
                link5_rot = _quat_wxyz_to_rot(body_quat_np[0, inner.link5_idx])
                row, frames = _evaluate_alignment(
                    trial=1,
                    obj_center=obj_start_local[0],
                    obj_radius_m=0.5 * float(args.cylinder_diameter_m),
                    obj_height_m=float(args.cylinder_height_m),
                    target_tcp=target_tcp,
                    tangent=tangent,
                    actual_tcp=tcp_local[0],
                    link5_rot=link5_rot,
                    obj_start_w=obj_start_w.detach().cpu().numpy()[0],
                    obj_final_w=inner._sponge_pos_w[0].detach().cpu().numpy(),
                    target_arm=target_arm,
                    actual_arm=actual_arm,
                    ik_failure_steps=int(ik_failure_counts[0]),
                )
                trace_rows.append(
                    {
                        "step": int(step),
                        "phase": phase,
                        "alpha": float(alpha),
                        "actual_joint_rad_by_name": _joint_dict(actual_all),
                        "commanded_joint_rad_by_name": _joint_dict(commanded_all),
                        "tcp_pose_error_mm": row["tcp_pose_error_mm"],
                        "commanded_tcp_pose_error_mm": row["commanded_tcp_pose_error_mm"],
                        "arm_joint_err_max_rad": row["arm_joint_err_max_rad"],
                        "torque": d327._torque_saturation(inner, 0),
                        "contact_force": _contact_force_row(witnesses, contact_status),
                        "frames": frames,
                    }
                )

        rows, frame_sets = _state_eval_rows(
            inner,
            origins,
            obj_start_w,
            obj_start_local,
            target_info,
            final_target_arm,
            ik_failure_counts,
            args,
        )
        marker_status = draw_frames(frame_sets[0], prim_path="/World/D330CylinderG0aFrames") if frame_sets else {}
        snapshots: list[dict[str, Any]] = []
        for trial_idx in (0, 4, 9):
            if trial_idx < len(rows):
                path = args.out_dir / f"d330_cyl_alignment_trial_{trial_idx + 1:02d}_snapshot.png"
                snapshots.append(
                    {
                        "trial": int(trial_idx + 1),
                        "path": _write_snapshot(
                            path,
                            obj_center=obj_start_local[trial_idx],
                            obj_diameter_m=float(args.cylinder_diameter_m),
                            obj_height_m=float(args.cylinder_height_m),
                            frames=frame_sets[trial_idx],
                            row=rows[trial_idx],
                            title=f"D330 G0a cylinder alignment trial {trial_idx + 1}",
                        ),
                    }
                )

        rrd_file = args.out_dir / "d330_cyl_alignment_trace_v2.rrd"
        rrd_status = log_rerun(
            rrd_file,
            frames=frame_sets[0],
            joint_state={
                "label": "d330_cyl_alignment",
                "approach_steps": int(args.approach_steps),
                "hold_steps": int(args.hold_steps),
                "object": "cylinder_d34_h90",
            },
            joint_trace=trace_rows,
            cube={"center": obj_start_local[0].tolist(), "size": float(args.cylinder_diameter_m)},
            urdf_path=args.urdf_path,
            live_viewer=bool(args.live_viewer),
            app_id="roarm_g0a_d330_cyl_alignment",
        )

        failure_counts = {
            "tcp_pose": sum(1 for row in rows if not row["pass_tcp_pose"]),
            "jaw_tangent": sum(1 for row in rows if not row["pass_jaw_tangent"]),
            "fixed_jaw_gap": sum(1 for row in rows if not row["pass_fixed_jaw_gap"]),
            "fixed_jaw_penetration": sum(1 for row in rows if not row["pass_no_penetration"]),
            "contact_height": sum(1 for row in rows if not row["pass_contact_height"]),
            "object_displacement": sum(1 for row in rows if not row["pass_object_displacement"]),
        }
        pass_all_count = sum(1 for row in rows if row["pass_all"])
        return {
            "label": "d330_cyl_alignment_gate",
            "num_trials": int(args.num_trials),
            "pass_all_count": int(pass_all_count),
            "failure_counts": failure_counts,
            "rows": rows,
            "snapshots": snapshots,
            "marker_status": marker_status,
            "rrd_status": rrd_status,
            "rrd_path": _rel(rrd_file) if rrd_status.get("ok") else "",
            "contact_sensor_status": contact_status,
            "trace_stats": _trace_stats(trace_rows),
        }
    finally:
        env.close()


def _write_outputs(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "g0a_d330_cyl_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    rows = summary.get("gate", {}).get("rows", [])
    if rows:
        csv_path = out_dir / "g0a_d330_cyl_alignment_trials.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary["gate"]["trial_csv"] = _rel(csv_path)
        (out_dir / "g0a_d330_cyl_alignment_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    lines = [
        "# D330 G0a Cylinder Alignment Probe",
        "",
        "이번 case의 신규 변수: `[]` -- D330 executes the D329-approved object redefinition; no extra variable.",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- pass_all: `{summary['gate']['pass_all_count']}/{summary['gate']['num_trials']}`",
        f"- output json: `{_rel(out_dir / 'g0a_d330_cyl_alignment_summary.json')}`",
        f"- trial csv: `{summary['gate'].get('trial_csv', '')}`",
        f"- rrd: `{summary['gate'].get('rrd_path', '')}`",
        "",
        "## Trial Table",
        "",
        "| trial | pos mm | cmd pos mm | tangent deg | plane gap mm | radius gap mm | top clearance mm | disp mm | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {trial} | {tcp_pose_error_mm:.3f} | {commanded_tcp_pose_error_mm:.3f} | "
            "{jaw_tangent_error_deg:.3f} | {fixed_jaw_face_gap_mm:.3f} | "
            "{fixed_jaw_radius_gap_mm:.3f} | {contact_point_below_top_mm:.3f} | "
            "{object_disp_xy_mm:.3f} | {pass_all} |".format(**row)
        )
    lines.extend(["", "## Failure Counts", ""])
    for key, value in summary["gate"]["failure_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Contact Trace", ""])
    contact = summary["gate"].get("trace_stats", {}).get("contact", {})
    lines.append(f"- robot_net_max_force_n: `{contact.get('robot_net_max_force_n', 0.0)}`")
    lines.append(f"- robot_net_first_contact_step: `{contact.get('robot_net_first_contact_step', -1)}`")
    lines.append(f"- robot_net_argmax_body_name: `{contact.get('robot_net_argmax_body_name', '')}`")
    for key, value in dict(contact.get("filtered_max_force_by_link_target_n", {})).items():
        lines.append(f"- filtered {key}: `{value}` N")
    sensor_status = summary["gate"].get("contact_sensor_status", {})
    if sensor_status:
        lines.extend(["", "## Contact Sensor Status", ""])
        robot_status = dict(sensor_status.get("robot_net", {}))
        lines.append(f"- robot_net ok: `{robot_status.get('ok', False)}`")
        if robot_status.get("error"):
            lines.append(f"- robot_net error: `{robot_status.get('error')}`")
        for link_name, item in dict(sensor_status.get("filtered_link_sensors", {})).items():
            item = dict(item)
            lines.append(f"- {link_name} ok: `{item.get('ok', False)}`")
            if item.get("error"):
                lines.append(f"  error: `{item.get('error')}`")
    if summary["gate"].get("snapshots"):
        lines.extend(["", "## Snapshots", ""])
        for item in summary["gate"]["snapshots"]:
            lines.append(f"- trial {item['trial']}: `{item['path']}`")
    (out_dir / "g0a_d330_cyl_alignment_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_ROBOT_USD)
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=33001)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--object_x", type=float, default=0.30)
    parser.add_argument("--object_y", type=float, default=0.0)
    parser.add_argument("--cylinder_diameter_m", type=float, default=0.034)
    parser.add_argument("--cylinder_height_m", type=float, default=0.090)
    parser.add_argument("--object_mass_kg", type=float, default=0.72)
    parser.add_argument("--static_friction", type=float, default=1.5)
    parser.add_argument("--dynamic_friction", type=float, default=1.2)
    parser.add_argument("--approach_steps", type=int, default=220)
    parser.add_argument("--hold_steps", type=int, default=100)
    parser.add_argument("--pre_clearance_m", type=float, default=0.040)
    parser.add_argument("--radial_tip_past_near_face_m", type=float, default=0.010)
    parser.add_argument("--alignment_standoff_m", type=float, default=ALIGNMENT_STANDOFF_M)
    parser.add_argument("--episode_length_s", type=float, default=8.0)
    parser.add_argument("--live_viewer", action="store_true")
    args = parser.parse_args()

    if int(args.num_trials) != 10:
        raise ValueError("D330 G0a is pre-registered for exactly 10 trials")
    if not math.isclose(float(args.cylinder_diameter_m), 0.034, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("D330 cylinder diameter is fixed at 0.034m")
    if not math.isclose(float(args.cylinder_height_m), 0.090, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("D330 cylinder height is fixed at 0.090m")
    if not math.isclose(float(args.object_mass_kg), 0.72, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("D330 mass placeholder is fixed at 0.72kg; do not tune in G0a")
    if not args.robot_usd_path.exists():
        raise FileNotFoundError(args.robot_usd_path)
    if not args.urdf_path.exists():
        raise FileNotFoundError(args.urdf_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    gate = _run_gate(args)
    pass_all = int(gate["pass_all_count"])
    if pass_all == 10:
        verdict = "D330_G0A_CYL_ALIGNMENT_PASS"
        interpretation = "wrong-object hypothesis supported; G0a cylinder alignment complete; G0b transition requires user approval"
    else:
        verdict = "D330_G0A_CYL_ALIGNMENT_FAIL"
        tcp_errors = [float(row["tcp_pose_error_mm"]) for row in gate.get("rows", [])]
        mean_tcp = float(np.mean(tcp_errors)) if tcp_errors else math.inf
        if mean_tcp >= 50.0:
            interpretation = "wrong-object runtime-stall explanation falsified on cylinder; resume collision/drive audit on correct object"
        else:
            interpretation = "cylinder improved runtime but did not satisfy all G0a gates; inspect gate-specific failures and contact trace"

    env_versions = _version_manifest()
    env_versions["robot_usd_path"] = _rel(args.robot_usd_path)
    env_versions["urdf_path"] = _rel(args.urdf_path)

    radius = 0.5 * float(args.cylinder_diameter_m)
    summary = {
        "artifact": "d330_g0a_cyl_alignment_probe",
        "verdict": verdict,
        "active_case": "G0a redefined cylinder D34xH90",
        "new_variable": "none in D330 runtime; executes D329-approved object geometry change",
        "registered_prediction": {
            "hypothesis": "wrong-object is primary runtime-stall cause",
            "expected": "TCP error collapses from ~72mm to single-digit mm and 10/10 PASS",
            "falsification": "cylinder reproduces ~60-70mm stall signature",
        },
        "interpretation": interpretation,
        "object_contract": {
            "shape": "cylinder",
            "radius_m": radius,
            "diameter_m": float(args.cylinder_diameter_m),
            "height_m": float(args.cylinder_height_m),
            "mass_kg": float(args.object_mass_kg),
            "mass_note": "D330 G0a placeholder to keep geometry as the only case change; G0b lift must replace with real 100-120g spec.",
            "static_friction": float(args.static_friction),
            "dynamic_friction": float(args.dynamic_friction),
            "object_xy_m": [float(args.object_x), float(args.object_y)],
            "z_contract": "TCP target z equals cylinder center height; object center z = TABLE_Z + 0.045",
        },
        "target_contract": {
            "pose_family": "D325 position_only_tangent_minus1; link5 +x tangent -1; tool +z free",
            "alignment_standoff_m": float(args.alignment_standoff_m),
            "tangent_offset_m": float(radius - FIXED_JAW_FACE_OFFSET_M + float(args.alignment_standoff_m)),
            "radial_offset_m": float(radius - float(args.radial_tip_past_near_face_m)),
            "pre_clearance_m": float(args.pre_clearance_m),
            "waypoint_policy": "d327_radial 2-waypoint only; no waypoint search",
        },
        "criteria": {
            "tcp_position_error_mm_max": TCP_POS_GATE_M * 1000.0,
            "jaw_tangent_error_deg_max": JAW_TANGENT_GATE_DEG,
            "fixed_jaw_horizontal_gap_mm_range": [0.0, GAP_GATE_M * 1000.0],
            "no_penetration": True,
            "contact_point_below_object_top_mm_min": TOP_CLEARANCE_GATE_M * 1000.0,
            "object_displacement_mm_max": OBJECT_DISP_GATE_M * 1000.0,
            "all_trials_required": 10,
        },
        "gate": gate,
        "environment": env_versions,
        "non_goals": [
            "no cube reintroduction",
            "no waypoint search",
            "no gripper close",
            "no G0b/lift",
            "no RL/PPO",
            "no randomization",
            "no large render",
            "no friction/material/mass tuning",
            "no VLA/RoArm/B200",
        ],
    }
    _write_outputs(args.out_dir, summary)
    print(
        "[d330-g0a-cylinder] "
        f"verdict={verdict} pass_all={pass_all}/10 "
        f"failures={gate['failure_counts']} "
        f"out_dir={_rel(args.out_dir)}"
    )
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
