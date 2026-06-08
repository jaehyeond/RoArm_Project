"""Local preflight for cube10cm tool/contact-proxy and orientation path.

No IsaacLab runtime, no GPU, no dataset generation, no training, no robot
control, no SSH. This audit reconstructs link5/TCP/gripper collision proxy
geometry from the existing seed962 trace and local URDF assets, then decides
whether a side-contact teacher can be specified as a physical tool proxy or
whether the next step must be instrumentation/orientation support first.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim_scripts"))

from roarm_kinematics import _CHAIN, Tmat, Trot_z, fk_full  # noqa: E402


LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_TRACE = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_trace.csv"
DEFAULT_SUMMARY = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_summary.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tool_contact_proxy_orientation_preflight.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tool_contact_proxy_orientation_preflight_summary.out"
DEFAULT_URDF = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
DEFAULT_GRIPPER_COLLISION_STL = REPO / "local_assets/roarm_m3/urdf/meshes/gripper_link_collision_g2a.stl"
DEFAULT_LINK5_COLLISION_STL = REPO / "local_assets/roarm_m3/urdf/meshes/link5.stl"

TCP_LOCAL_OFFSET_M = np.asarray([0.0, 0.0, 0.115428], dtype=np.float64)
GRIPPER_JOINT_ORIGIN_XYZ = np.asarray([0.0, 0.018821, 0.052035], dtype=np.float64)
GRIPPER_JOINT_ORIGIN_RPY = np.asarray([-math.pi / 2.0, -math.pi / 2.0, 0.0], dtype=np.float64)


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "p95": None}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": mean(values),
        "median": median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[p95_idx],
    }


def _read_trace(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for source_line, row in enumerate(reader, start=2):
            row["_source_line"] = source_line
            rows.append(row)
    return rows, fieldnames


def _first_contact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_env: dict[int, dict[str, Any]] = {}
    for row in rows:
        env_id = _i(row, "env_id")
        if env_id in by_env:
            continue
        if _i(row, "measured_contact_now") == 1:
            by_env[env_id] = row
    return [by_env[k] for k in sorted(by_env)]


def _load_stl_vertices_m(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:5].lower() == b"solid":
        verts: list[list[float]] = []
        for line in data.decode("utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] == "vertex":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if verts:
            return np.asarray(verts, dtype=np.float64) * 0.001
    if len(data) < 84:
        raise ValueError(f"STL too small: {path}")
    n_tri = struct.unpack("<I", data[80:84])[0]
    expected = 84 + 50 * n_tri
    if expected > len(data):
        raise ValueError(f"STL size mismatch: {path} expected {expected}, got {len(data)}")
    verts = []
    off = 84
    for _ in range(n_tri):
        vals = struct.unpack("<12fH", data[off:off + 50])
        verts.extend((vals[3:6], vals[6:9], vals[9:12]))
        off += 50
    return np.asarray(verts, dtype=np.float64) * 0.001


def _proxy_points_from_bbox(box_min: np.ndarray, box_max: np.ndarray) -> dict[str, np.ndarray]:
    center = (box_min + box_max) / 2.0
    points: dict[str, np.ndarray] = {"center": center}
    for axis, name in enumerate(("x", "y", "z")):
        neg = center.copy()
        pos = center.copy()
        neg[axis] = box_min[axis]
        pos[axis] = box_max[axis]
        points[f"{name}neg_face_center"] = neg
        points[f"{name}pos_face_center"] = pos
    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                p = np.asarray(
                    [
                        box_max[0] if sx else box_min[0],
                        box_max[1] if sy else box_min[1],
                        box_max[2] if sz else box_min[2],
                    ],
                    dtype=np.float64,
                )
                points[f"corner_{sx}{sy}{sz}"] = p
    return points


def _mesh_proxy_points(vertices: np.ndarray) -> dict[str, np.ndarray]:
    box_min = vertices.min(axis=0)
    box_max = vertices.max(axis=0)
    points = _proxy_points_from_bbox(box_min, box_max)
    # Keep true mesh vertices too; nearest side contact can occur on a non-AABB
    # feature. Duplicate labels are harmless and preserve the stable face/corner
    # labels for simple box meshes.
    for idx, point in enumerate(vertices):
        points[f"vertex_{idx:05d}"] = point
    return points


def _link5_transform_from_rad(joints_rad: np.ndarray) -> np.ndarray:
    q_deg = np.degrees(joints_rad)
    q_rad = np.radians(q_deg)
    transform = np.eye(4, dtype=np.float64)
    for name, xyz, rpy, qi in _CHAIN:
        if name == "link5_to_tcp":
            break
        transform = transform @ Tmat(xyz, rpy)
        if qi is not None:
            transform = transform @ Trot_z(q_rad[qi])
    return transform


def _gripper_transform_from_rad(joints_rad: np.ndarray, gripper_rad: float) -> np.ndarray:
    return (
        _link5_transform_from_rad(joints_rad)
        @ Tmat(GRIPPER_JOINT_ORIGIN_XYZ, GRIPPER_JOINT_ORIGIN_RPY)
        @ Trot_z(float(gripper_rad))
    )


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    hom = np.ones(4, dtype=np.float64)
    hom[:3] = point
    return (transform @ hom)[:3]


def _transform_points(transform: np.ndarray, points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: _transform_point(transform, point) for name, point in points.items()}


def _joint_rad(row: dict[str, str], idx: int) -> float:
    before_key = f"joint_pos_before_{idx}_rad"
    if row.get(before_key, "") != "":
        return float(row[before_key])
    return _f(row, f"arm_joint_{idx}_rad")


def _row_world_origin(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [_f(row, "env_origin_x_m"), _f(row, "env_origin_y_m"), _f(row, "env_origin_z_m")],
        dtype=np.float64,
    )


def _row_vec(row: dict[str, str], *keys: str) -> np.ndarray:
    return np.asarray([_f(row, key) for key in keys], dtype=np.float64)


def _rate(values: list[bool]) -> float:
    return sum(1 for v in values if v) / len(values) if values else 0.0


def _counter_mode(values: list[str]) -> tuple[str | None, float | None]:
    if not values:
        return None, None
    value, count = Counter(values).most_common(1)[0]
    return value, count / len(values)


def _mesh_metrics(
    mesh_name: str,
    points_world: dict[str, np.ndarray],
    side_center: np.ndarray,
    side_top: np.ndarray,
    cube_min: np.ndarray,
    cube_max: np.ndarray,
) -> dict[str, Any]:
    labels = list(points_world.keys())
    arr = np.stack([points_world[label] for label in labels], axis=0)
    side_dist_arr = np.linalg.norm(arr - side_center.reshape(1, 3), axis=1)
    top_dist_arr = np.linalg.norm(arr - side_top.reshape(1, 3), axis=1)
    idx = int(np.argmin(side_dist_arr))
    proxy_min = arr.min(axis=0)
    proxy_max = arr.max(axis=0)
    overlap_vec = np.minimum(cube_max, proxy_max) - np.maximum(cube_min, proxy_min)
    return {
        "mesh": mesh_name,
        "closest_label": labels[idx],
        "closest_point": arr[idx],
        "side_dist": float(side_dist_arr[idx]),
        "side_z_err": float(arr[idx, 2] - side_center[2]),
        "top_dist": float(top_dist_arr[idx]),
        "aabb_overlap": bool(np.all(overlap_vec > 0.0)),
    }


def build_audit(
    trace_csv: Path,
    summary_json: Path,
    urdf_path: Path,
    gripper_collision_stl: Path,
    link5_collision_stl: Path,
) -> dict[str, Any]:
    rows, fieldnames = _read_trace(trace_csv)
    if not rows:
        raise RuntimeError(f"trace has no rows: {trace_csv}")
    contact_rows = _first_contact_rows(rows)
    if not contact_rows:
        raise RuntimeError("no measured_contact_now rows found")
    summary = json.loads(summary_json.read_text())
    gripper_vertices = _load_stl_vertices_m(gripper_collision_stl)
    link5_vertices = _load_stl_vertices_m(link5_collision_stl)
    bbox_min = gripper_vertices.min(axis=0)
    bbox_max = gripper_vertices.max(axis=0)
    link5_bbox_min = link5_vertices.min(axis=0)
    link5_bbox_max = link5_vertices.max(axis=0)
    proxy_points_gripper = _mesh_proxy_points(gripper_vertices)
    proxy_points_link5 = _mesh_proxy_points(link5_vertices)

    fk_tcp_err: list[float] = []
    hand_tcp_side_dist: list[float] = []
    hand_tcp_side_z_err: list[float] = []
    hand_tcp_top_dist: list[float] = []
    closest_proxy_side_dist: list[float] = []
    closest_proxy_side_z_err: list[float] = []
    closest_proxy_top_dist: list[float] = []
    closest_proxy_labels: list[str] = []
    closest_proxy_meshes: list[str] = []
    closest_link5_proxy_side_dist: list[float] = []
    closest_link5_proxy_side_z_err: list[float] = []
    closest_link5_proxy_top_dist: list[float] = []
    closest_gripper_proxy_side_dist: list[float] = []
    closest_gripper_proxy_side_z_err: list[float] = []
    closest_gripper_proxy_top_dist: list[float] = []
    current_link5_target_err: list[float] = []
    proxy_required_link5_target_err: list[float] = []
    proxy_required_vs_current_link5_target_err_ratio: list[float] = []
    proxy_required_link5_target_z_delta: list[float] = []
    best_face_push_abs_align: list[float] = []
    best_face_push_signed_align: list[float] = []
    proxy_aabb_overlaps_cube: list[bool] = []
    link5_aabb_overlaps_cube: list[bool] = []
    gripper_aabb_overlaps_cube: list[bool] = []
    per_env: list[dict[str, Any]] = []

    face_normals = {
        "xneg": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
        "xpos": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "yneg": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
        "ypos": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "zneg": np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
        "zpos": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    }

    for row in contact_rows:
        origin = _row_world_origin(row)
        joints = np.asarray([_joint_rad(row, idx) for idx in range(5)], dtype=np.float64)
        joints6 = np.zeros(6, dtype=np.float64)
        joints6[:5] = joints
        joints6[5] = _f(row, "gripper_joint_rad")
        gripper_rad = _f(row, "gripper_joint_rad")

        fk_tcp_local, _ = fk_full(np.degrees(joints6))
        fk_tcp_world = fk_tcp_local + origin
        trace_tcp = _row_vec(row, "tcp_x_before_m", "tcp_y_before_m", "tcp_z_before_m")
        fk_err = float(np.linalg.norm(fk_tcp_world - trace_tcp))
        fk_tcp_err.append(fk_err)

        gripper_tf = _gripper_transform_from_rad(joints, gripper_rad)
        link5_tf = _link5_transform_from_rad(joints)
        link5_world_tf = link5_tf.copy()
        link5_world_tf[:3, 3] += origin
        gripper_world_tf = gripper_tf.copy()
        gripper_world_tf[:3, 3] += origin
        proxy_world = _transform_points(gripper_world_tf, proxy_points_gripper)
        link5_proxy_world = _transform_points(link5_world_tf, proxy_points_link5)

        push_xy = np.asarray([_f(row, "push_dx"), _f(row, "push_dy")], dtype=np.float64)
        push_norm = float(np.linalg.norm(push_xy))
        if push_norm <= 1.0e-9:
            raise RuntimeError("push direction has near-zero norm")
        push_xy = push_xy / push_norm
        push3 = np.asarray([push_xy[0], push_xy[1], 0.0], dtype=np.float64)

        cube = _row_vec(row, "cube_x_m", "cube_y_m", "cube_z_m")
        cube_size = _row_vec(row, "cube_size_x_m", "cube_size_y_m", "cube_size_z_m")
        half_along = abs(push_xy[0]) * cube_size[0] / 2.0 + abs(push_xy[1]) * cube_size[1] / 2.0
        side_center = cube.copy()
        side_center[:2] -= push_xy * half_along
        side_top = side_center.copy()
        side_top[2] = cube[2] + cube_size[2] / 2.0

        cube_min = cube - cube_size / 2.0
        cube_max = cube + cube_size / 2.0
        gripper_metrics = _mesh_metrics("gripper_collision", proxy_world, side_center, side_top, cube_min, cube_max)
        link5_metrics = _mesh_metrics("link5_collision", link5_proxy_world, side_center, side_top, cube_min, cube_max)
        mesh_metrics = min((gripper_metrics, link5_metrics), key=lambda item: item["side_dist"])
        closest_label = f"{mesh_metrics['mesh']}:{mesh_metrics['closest_label']}"
        closest_point = mesh_metrics["closest_point"]

        hand_tcp_side_dist.append(float(np.linalg.norm(trace_tcp - side_center)))
        hand_tcp_side_z_err.append(float(trace_tcp[2] - side_center[2]))
        hand_tcp_top_dist.append(float(np.linalg.norm(trace_tcp - side_top)))
        closest_proxy_side_dist.append(mesh_metrics["side_dist"])
        closest_proxy_side_z_err.append(float(closest_point[2] - side_center[2]))
        closest_proxy_top_dist.append(mesh_metrics["top_dist"])
        closest_proxy_labels.append(closest_label)
        closest_proxy_meshes.append(mesh_metrics["mesh"])
        closest_link5_proxy_side_dist.append(link5_metrics["side_dist"])
        closest_link5_proxy_side_z_err.append(link5_metrics["side_z_err"])
        closest_link5_proxy_top_dist.append(link5_metrics["top_dist"])
        closest_gripper_proxy_side_dist.append(gripper_metrics["side_dist"])
        closest_gripper_proxy_side_z_err.append(gripper_metrics["side_z_err"])
        closest_gripper_proxy_top_dist.append(gripper_metrics["top_dist"])

        rot = gripper_world_tf[:3, :3]
        signed = [float((rot @ normal).dot(push3)) for normal in face_normals.values()]
        best_face_push_abs_align.append(max(abs(v) for v in signed))
        best_face_push_signed_align.append(max(signed))

        overlap = bool(mesh_metrics["aabb_overlap"])
        proxy_aabb_overlaps_cube.append(bool(overlap))
        link5_aabb_overlaps_cube.append(bool(link5_metrics["aabb_overlap"]))
        gripper_aabb_overlaps_cube.append(bool(gripper_metrics["aabb_overlap"]))

        link5_pos = _row_vec(row, "link5_x_before_m", "link5_y_before_m", "link5_z_before_m")
        link5_target_err = _f(row, "link5_target_err_before_m")
        proxy_offset_world = closest_point - link5_pos
        proxy_required_link5_target = side_center - proxy_offset_world
        proxy_err = float(np.linalg.norm(link5_pos - proxy_required_link5_target))
        current_link5_target_err.append(link5_target_err)
        proxy_required_link5_target_err.append(proxy_err)
        if link5_target_err > 1.0e-9:
            proxy_required_vs_current_link5_target_err_ratio.append(proxy_err / link5_target_err)
        proxy_required_link5_target_z_delta.append(float(proxy_required_link5_target[2] - link5_pos[2]))

        per_env.append(
            {
                "env_id": _i(row, "env_id"),
                "source_line": int(row["_source_line"]),
                "step": _i(row, "step"),
                "fk_tcp_reconstruction_err_m": fk_err,
                "hand_tcp_to_side_center_dist_m": hand_tcp_side_dist[-1],
                "hand_tcp_minus_side_center_z_m": hand_tcp_side_z_err[-1],
                "hand_tcp_to_side_top_dist_m": hand_tcp_top_dist[-1],
                "closest_proxy_label": closest_label,
                "closest_proxy_mesh": mesh_metrics["mesh"],
                "closest_proxy_to_side_center_dist_m": closest_proxy_side_dist[-1],
                "closest_proxy_minus_side_center_z_m": closest_proxy_side_z_err[-1],
                "closest_proxy_to_side_top_dist_m": closest_proxy_top_dist[-1],
                "best_face_push_abs_alignment": best_face_push_abs_align[-1],
                "best_face_push_signed_alignment": best_face_push_signed_align[-1],
                "proxy_aabb_overlaps_cube": proxy_aabb_overlaps_cube[-1],
                "link5_proxy_to_side_center_dist_m": link5_metrics["side_dist"],
                "link5_proxy_minus_side_center_z_m": link5_metrics["side_z_err"],
                "link5_proxy_aabb_overlaps_cube": link5_metrics["aabb_overlap"],
                "gripper_proxy_to_side_center_dist_m": gripper_metrics["side_dist"],
                "gripper_proxy_minus_side_center_z_m": gripper_metrics["side_z_err"],
                "gripper_proxy_aabb_overlaps_cube": gripper_metrics["aabb_overlap"],
                "current_hand_tcp_link5_target_err_m": link5_target_err,
                "proxy_required_link5_target_err_m": proxy_err,
                "proxy_required_vs_current_link5_target_err_ratio": (
                    proxy_err / link5_target_err if link5_target_err > 1.0e-9 else None
                ),
                "proxy_required_link5_target_z_delta_m": proxy_required_link5_target_z_delta[-1],
            }
        )

    proxy_mode, proxy_mode_rate = _counter_mode(closest_proxy_labels)
    proxy_mesh_mode, proxy_mesh_mode_rate = _counter_mode(closest_proxy_meshes)
    fk_trustworthy = (_stats(fk_tcp_err)["p95"] or 999.0) <= 0.005
    proxy_side_near_10mm_rate = _rate([v <= 0.010 for v in closest_proxy_side_dist])
    proxy_side_near_20mm_rate = _rate([v <= 0.020 for v in closest_proxy_side_dist])
    hand_side_near_10mm_rate = _rate([v <= 0.010 for v in hand_tcp_side_dist])
    face_aligned_rate = _rate([v >= 0.90 for v in best_face_push_abs_align])
    proxy_overlap_rate = _rate(proxy_aabb_overlaps_cube)
    link5_overlap_rate = _rate(link5_aabb_overlaps_cube)
    gripper_overlap_rate = _rate(gripper_aabb_overlaps_cube)
    position_only = summary.get("command_type") == "position"
    trace_has_link5_quat = any(name.startswith("link5_q") or name.startswith("link5_quat") for name in fieldnames)
    proxy_target_reduces_link5_error = (
        (_stats(proxy_required_vs_current_link5_target_err_ratio)["mean"] or 999.0) < 0.85
    )
    stable_link5_proxy = proxy_mesh_mode == "link5_collision" and (proxy_mesh_mode_rate or 0.0) >= 0.95

    proxy_promising = bool(
        fk_trustworthy
        and proxy_side_near_20mm_rate >= 0.75
        and proxy_side_near_10mm_rate > hand_side_near_10mm_rate
        and face_aligned_rate >= 0.75
    )
    link5_proxy_candidate_promising = bool(
        fk_trustworthy
        and stable_link5_proxy
        and proxy_target_reduces_link5_error
        and face_aligned_rate >= 0.75
    )
    orientation_blocked = bool(position_only and not trace_has_link5_quat)
    dataset_unblocked = False

    if not fk_trustworthy:
        next_step = "add_runtime_proxy_quat_trace_instrumentation_before_geometry_claim"
        verdict_class = "FK_RECONSTRUCTION_NOT_TRUSTWORTHY"
    elif link5_proxy_candidate_promising and orientation_blocked:
        next_step = "local_code_preflight_for_link5_collision_corner_proxy_with_pose_or_trace_support_before_gpu"
        verdict_class = "LINK5_PROXY_REDUCES_TARGET_ERROR_BUT_ORIENTATION_AND_CONTACT_SEMANTICS_UNVALIDATED"
    elif proxy_promising and orientation_blocked:
        next_step = "local_code_preflight_for_tool_proxy_trace_columns_then_one_tiny_runtime_if_approved"
        verdict_class = "PROXY_POSITION_PROMISING_BUT_ORIENTATION_UNVALIDATED"
    elif proxy_promising:
        next_step = "one_tiny_tool_proxy_runtime_candidate_if_explicitly_approved"
        verdict_class = "PROXY_POSITION_PROMISING"
    else:
        next_step = "do_not_run_gpu_redesign_proxy_or_orientation_contract_locally"
        verdict_class = "NO_STABLE_SIDE_CONTACT_PROXY_FROM_EXISTING_TRACE"

    return {
        "artifact_type": "cube10cm_tool_contact_proxy_orientation_preflight_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_audit_only": True,
        "no_gpu_runtime_dataset_training_robot_ssh": True,
        "source": {
            "trace_csv": str(trace_csv),
            "summary_json": str(summary_json),
            "urdf_path": str(urdf_path),
            "gripper_collision_stl": str(gripper_collision_stl),
            "link5_collision_stl": str(link5_collision_stl),
            "trace_rows": len(rows),
            "first_contact_env_count": len(contact_rows),
            "contact_source_line_min": min(int(row["_source_line"]) for row in contact_rows),
            "contact_source_line_max": max(int(row["_source_line"]) for row in contact_rows),
        },
        "runtime_contract": {
            "command_type": summary.get("command_type"),
            "ik_method": summary.get("ik_method"),
            "tcp_height_mode": summary.get("tcp_height_mode"),
            "cube_size_m": summary.get("cube_size_m"),
            "cube_mass_kg": summary.get("cube_mass_kg"),
            "diffik_clip_rate_mean": summary.get("diffik_clip_rate_mean"),
            "final_tcp_target_err_mean_m": summary.get("final_tcp_target_err_mean_m"),
        },
        "asset_contract": {
            "tcp_local_offset_m": TCP_LOCAL_OFFSET_M.tolist(),
            "gripper_joint_origin_xyz_m": GRIPPER_JOINT_ORIGIN_XYZ.tolist(),
            "gripper_joint_origin_rpy_rad": GRIPPER_JOINT_ORIGIN_RPY.tolist(),
            "gripper_collision_bbox_min_m": bbox_min.tolist(),
            "gripper_collision_bbox_max_m": bbox_max.tolist(),
            "gripper_collision_bbox_size_m": (bbox_max - bbox_min).tolist(),
            "link5_collision_bbox_min_m": link5_bbox_min.tolist(),
            "link5_collision_bbox_max_m": link5_bbox_max.tolist(),
            "link5_collision_bbox_size_m": (link5_bbox_max - link5_bbox_min).tolist(),
        },
        "trace_support": {
            "has_joint_pos_before": all(f"joint_pos_before_{idx}_rad" in fieldnames for idx in range(5)),
            "has_link5_quaternion": trace_has_link5_quat,
            "can_reconstruct_fk_from_joint_trace": True,
            "fk_tcp_reconstruction_err_m": _stats(fk_tcp_err),
            "fk_reconstruction_trustworthy_5mm_p95": fk_trustworthy,
        },
        "contact_proxy_geometry": {
            "hand_tcp_to_side_center_dist_m": _stats(hand_tcp_side_dist),
            "hand_tcp_minus_side_center_z_m": _stats(hand_tcp_side_z_err),
            "hand_tcp_to_side_top_dist_m": _stats(hand_tcp_top_dist),
            "hand_tcp_side_center_near_10mm_rate": hand_side_near_10mm_rate,
            "closest_collision_proxy_to_side_center_dist_m": _stats(closest_proxy_side_dist),
            "closest_collision_proxy_minus_side_center_z_m": _stats(closest_proxy_side_z_err),
            "closest_collision_proxy_to_side_top_dist_m": _stats(closest_proxy_top_dist),
            "closest_collision_proxy_side_center_near_10mm_rate": proxy_side_near_10mm_rate,
            "closest_collision_proxy_side_center_near_20mm_rate": proxy_side_near_20mm_rate,
            "closest_proxy_label_mode": proxy_mode,
            "closest_proxy_label_mode_rate": proxy_mode_rate,
            "closest_proxy_mesh_mode": proxy_mesh_mode,
            "closest_proxy_mesh_mode_rate": proxy_mesh_mode_rate,
            "proxy_aabb_overlaps_live_cube_rate": proxy_overlap_rate,
            "link5_collision_proxy_to_side_center_dist_m": _stats(closest_link5_proxy_side_dist),
            "link5_collision_proxy_minus_side_center_z_m": _stats(closest_link5_proxy_side_z_err),
            "link5_collision_proxy_to_side_top_dist_m": _stats(closest_link5_proxy_top_dist),
            "link5_collision_aabb_overlaps_live_cube_rate": link5_overlap_rate,
            "gripper_collision_proxy_to_side_center_dist_m": _stats(closest_gripper_proxy_side_dist),
            "gripper_collision_proxy_minus_side_center_z_m": _stats(closest_gripper_proxy_side_z_err),
            "gripper_collision_proxy_to_side_top_dist_m": _stats(closest_gripper_proxy_top_dist),
            "gripper_collision_aabb_overlaps_live_cube_rate": gripper_overlap_rate,
            "current_hand_tcp_link5_target_err_m": _stats(current_link5_target_err),
            "proxy_required_link5_target_err_m": _stats(proxy_required_link5_target_err),
            "proxy_required_vs_current_link5_target_err_ratio": _stats(
                proxy_required_vs_current_link5_target_err_ratio
            ),
            "proxy_required_link5_target_z_delta_m": _stats(proxy_required_link5_target_z_delta),
            "stable_link5_proxy_label": stable_link5_proxy,
            "proxy_target_reduces_link5_error": proxy_target_reduces_link5_error,
        },
        "orientation_feasibility": {
            "current_diffik_position_only": position_only,
            "trace_has_link5_quaternion": trace_has_link5_quat,
            "best_gripper_face_push_abs_alignment": _stats(best_face_push_abs_align),
            "best_gripper_face_push_signed_alignment": _stats(best_face_push_signed_align),
            "face_abs_alignment_ge_0p90_rate": face_aligned_rate,
            "orientation_path_validated_from_trace": bool((not position_only) and trace_has_link5_quat),
            "orientation_or_proxy_runtime_support_required": bool(position_only),
        },
        "verdict": {
            "proxy_position_promising_from_existing_trace": proxy_promising,
            "link5_proxy_candidate_promising_from_existing_trace": link5_proxy_candidate_promising,
            "orientation_path_blocked_by_position_only_diffik": orientation_blocked,
            "selected_teacher_criterion": "tool_oriented_side_contact_proxy",
            "verdict_class": verdict_class,
            "dataset_rl_roarm_unblocked": dataset_unblocked,
            "next": next_step,
        },
        "per_env_first_contact": per_env,
    }


def write_summary(audit: dict[str, Any], out_summary: Path) -> None:
    src = audit["source"]
    contract = audit["runtime_contract"]
    assets = audit["asset_contract"]
    trace = audit["trace_support"]
    geom = audit["contact_proxy_geometry"]
    orient = audit["orientation_feasibility"]
    verdict = audit["verdict"]
    lines = [
        "line1 artifact=cube10cm_tool_contact_proxy_orientation_preflight_v1 "
        "local_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        "line2 source "
        f"trace_rows={src['trace_rows']} first_contact_envs={src['first_contact_env_count']} "
        f"contact_source_lines={src['contact_source_line_min']}-{src['contact_source_line_max']}",
        "line3 runtime_asset_contract "
        f"command_type={contract['command_type']} ik_method={contract['ik_method']} "
        f"tcp_height_mode={contract['tcp_height_mode']} tcp_local_offset_z={assets['tcp_local_offset_m'][2]:.6f} "
        f"gripper_collision_bbox_size={assets['gripper_collision_bbox_size_m']} "
        f"link5_collision_bbox_size={assets['link5_collision_bbox_size_m']}",
        "line4 trace_support "
        f"has_joint_pos_before={trace['has_joint_pos_before']} has_link5_quaternion={trace['has_link5_quaternion']} "
        f"fk_tcp_err_mean={trace['fk_tcp_reconstruction_err_m']['mean']:.9f} "
        f"fk_tcp_err_p95={trace['fk_tcp_reconstruction_err_m']['p95']:.9f} "
        f"fk_trustworthy_5mm_p95={trace['fk_reconstruction_trustworthy_5mm_p95']}",
        "line5 current_hand_tcp "
        f"side_center_dist_mean={geom['hand_tcp_to_side_center_dist_m']['mean']:.9f} "
        f"side_center_z_err_mean={geom['hand_tcp_minus_side_center_z_m']['mean']:.9f} "
        f"side_top_dist_mean={geom['hand_tcp_to_side_top_dist_m']['mean']:.9f} "
        f"side_center_near_10mm_rate={geom['hand_tcp_side_center_near_10mm_rate']:.9f}",
        "line6 gripper_collision_proxy "
        f"side_center_dist_mean={geom['gripper_collision_proxy_to_side_center_dist_m']['mean']:.9f} "
        f"side_center_z_err_mean={geom['gripper_collision_proxy_minus_side_center_z_m']['mean']:.9f} "
        f"side_top_dist_mean={geom['gripper_collision_proxy_to_side_top_dist_m']['mean']:.9f} "
        f"cube_aabb_overlap_rate={geom['gripper_collision_aabb_overlaps_live_cube_rate']:.9f}",
        "line7 link5_collision_proxy "
        f"side_center_dist_mean={geom['link5_collision_proxy_to_side_center_dist_m']['mean']:.9f} "
        f"side_center_z_err_mean={geom['link5_collision_proxy_minus_side_center_z_m']['mean']:.9f} "
        f"side_top_dist_mean={geom['link5_collision_proxy_to_side_top_dist_m']['mean']:.9f} "
        f"cube_aabb_overlap_rate={geom['link5_collision_aabb_overlaps_live_cube_rate']:.9f}",
        "line8 best_collision_proxy "
        f"side_center_dist_mean={geom['closest_collision_proxy_to_side_center_dist_m']['mean']:.9f} "
        f"side_center_z_err_mean={geom['closest_collision_proxy_minus_side_center_z_m']['mean']:.9f} "
        f"near10={geom['closest_collision_proxy_side_center_near_10mm_rate']:.9f} "
        f"near20={geom['closest_collision_proxy_side_center_near_20mm_rate']:.9f} "
        f"mesh_mode={geom['closest_proxy_mesh_mode']} mesh_mode_rate={geom['closest_proxy_mesh_mode_rate']:.9f} "
        f"label_mode={geom['closest_proxy_label_mode']} label_mode_rate={geom['closest_proxy_label_mode_rate']:.9f} "
        f"cube_aabb_overlap_rate={geom['proxy_aabb_overlaps_live_cube_rate']:.9f}",
        "line9 proxy_target_feasibility "
        f"current_link5_target_err_mean={geom['current_hand_tcp_link5_target_err_m']['mean']:.9f} "
        f"proxy_required_link5_target_err_mean={geom['proxy_required_link5_target_err_m']['mean']:.9f} "
        f"proxy_vs_current_err_ratio_mean={geom['proxy_required_vs_current_link5_target_err_ratio']['mean']:.9f} "
        f"proxy_required_link5_target_z_delta_mean={geom['proxy_required_link5_target_z_delta_m']['mean']:.9f} "
        f"stable_link5_proxy={geom['stable_link5_proxy_label']} "
        f"proxy_target_reduces_link5_error={geom['proxy_target_reduces_link5_error']}",
        "line10 orientation_feasibility "
        f"position_only={orient['current_diffik_position_only']} "
        f"face_abs_align_mean={orient['best_gripper_face_push_abs_alignment']['mean']:.9f} "
        f"face_align_ge_0p90_rate={orient['face_abs_alignment_ge_0p90_rate']:.9f} "
        f"orientation_path_validated_from_trace={orient['orientation_path_validated_from_trace']} "
        f"runtime_support_required={orient['orientation_or_proxy_runtime_support_required']}",
        "line11 verdict "
        f"proxy_position_promising={verdict['proxy_position_promising_from_existing_trace']} "
        f"link5_proxy_candidate_promising={verdict['link5_proxy_candidate_promising_from_existing_trace']} "
        f"orientation_blocked={verdict['orientation_path_blocked_by_position_only_diffik']} "
        f"selected_teacher_criterion={verdict['selected_teacher_criterion']} "
        f"verdict_class={verdict['verdict_class']}",
        "line12 pipeline "
        f"dataset_rl_roarm_unblocked={verdict['dataset_rl_roarm_unblocked']} "
        f"next={verdict['next']}",
    ]
    out_summary.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_csv", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--gripper_collision_stl", type=Path, default=DEFAULT_GRIPPER_COLLISION_STL)
    parser.add_argument("--link5_collision_stl", type=Path, default=DEFAULT_LINK5_COLLISION_STL)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    audit = build_audit(
        args.trace_csv,
        args.summary_json,
        args.urdf_path,
        args.gripper_collision_stl,
        args.link5_collision_stl,
    )
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_summary(audit, args.out_summary)
    print(args.out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
