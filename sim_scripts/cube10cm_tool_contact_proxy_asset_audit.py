"""Audit RoArm-M3 tool contact proxy assets for cube10cm tap.

This is a local asset/code audit only: no IsaacLab runtime, no GPU, no
training, no robot control, no SSH. It compares candidate tool proxies:

- link5 collision mesh
- gripper_link collision mesh
- gripper_link visual mesh

It also records the USD converter collision policy when available. The goal is
to decide whether the current link5 collision proxy should be replaced by a
"direct gripper" proxy.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
URDF = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
USD_CONFIG = REPO / "local_assets/roarm_m3/usd/config.yaml"
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tool_contact_proxy_asset_audit_d231.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tool_contact_proxy_asset_audit_d231_summary.out"


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
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    if text.lstrip().startswith("solid"):
        verts = []
        for line in text.splitlines():
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
        vals = struct.unpack("<12fH", data[off : off + 50])
        verts.extend((vals[3:6], vals[6:9], vals[9:12]))
        off += 50
    return np.asarray(verts, dtype=np.float64) * 0.001


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _transform(points: np.ndarray, xyz: np.ndarray, rpy: np.ndarray, joint_z_rad: float = 0.0) -> np.ndarray:
    rot = _rpy_matrix(rpy) @ _rot_z(joint_z_rad)
    return (rot @ points.T).T + xyz.reshape(1, 3)


def _bbox(points: np.ndarray) -> dict[str, Any]:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    size = mx - mn
    center = (mn + mx) * 0.5
    return {
        "min_m": mn.tolist(),
        "max_m": mx.tolist(),
        "size_m": size.tolist(),
        "center_m": center.tolist(),
        "volume_m3": float(np.prod(size)),
        "distal_z_m": float(mx[2]),
        "proximal_z_m": float(mn[2]),
    }


def _mesh_info(path: Path, frame: str, points: np.ndarray) -> dict[str, Any]:
    return {
        "path": str(path),
        "frame": frame,
        "vertex_count": int(points.shape[0]),
        "bbox": _bbox(points),
    }


def _mesh_path(root: ET.Element, link_name: str, kind: str) -> Path:
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"missing link {link_name}")
    node = link.find(f"./{kind}/geometry/mesh")
    if node is None:
        raise ValueError(f"missing {kind} mesh for link {link_name}")
    filename = node.attrib["filename"]
    return URDF.parent / filename


def _joint_origin(root: ET.Element, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    joint = root.find(f"./joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"missing joint {joint_name}")
    origin = joint.find("./origin")
    if origin is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.asarray([float(v) for v in origin.attrib.get("xyz", "0 0 0").split()], dtype=np.float64)
    rpy = np.asarray([float(v) for v in origin.attrib.get("rpy", "0 0 0").split()], dtype=np.float64)
    return xyz, rpy


def _read_usd_config(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not path.exists():
        return {"exists": False}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() == "true":
            parsed: Any = True
        elif value.lower() == "false":
            parsed = False
        elif value.lower() == "null":
            parsed = None
        else:
            parsed = value
        data[key.strip()] = parsed
    data["exists"] = True
    data["path"] = str(path)
    return data


def build_audit() -> dict[str, Any]:
    root = ET.parse(URDF).getroot()
    link5_collision_path = _mesh_path(root, "link5", "collision")
    gripper_collision_path = _mesh_path(root, "gripper_link", "collision")
    gripper_visual_path = _mesh_path(root, "gripper_link", "visual")
    joint_xyz, joint_rpy = _joint_origin(root, "link5_to_gripper_link")

    link5_collision = _load_stl_vertices_m(link5_collision_path)
    gripper_collision = _load_stl_vertices_m(gripper_collision_path)
    gripper_visual = _load_stl_vertices_m(gripper_visual_path)

    gripper_q_samples = [0.0, math.pi / 4.0, math.pi / 2.0]
    transformed: dict[str, Any] = {}
    for q in gripper_q_samples:
        suffix = f"q_{q:.6f}"
        collision_link5 = _transform(gripper_collision, joint_xyz, joint_rpy, q)
        visual_link5 = _transform(gripper_visual, joint_xyz, joint_rpy, q)
        transformed[f"gripper_collision_in_link5_{suffix}"] = _mesh_info(
            gripper_collision_path,
            f"link5_frame_gripper_q_{q:.6f}",
            collision_link5,
        )
        transformed[f"gripper_visual_in_link5_{suffix}"] = _mesh_info(
            gripper_visual_path,
            f"link5_frame_gripper_q_{q:.6f}",
            visual_link5,
        )
        union = np.concatenate([link5_collision, visual_link5], axis=0)
        transformed[f"link5_collision_plus_gripper_visual_union_{suffix}"] = {
            "frame": f"link5_frame_gripper_q_{q:.6f}",
            "bbox": _bbox(union),
            "component_meshes": [str(link5_collision_path), str(gripper_visual_path)],
        }

    link5_distal_z = float(link5_collision[:, 2].max())
    gripper_collision_distal_q0 = float(
        transformed["gripper_collision_in_link5_q_0.000000"]["bbox"]["distal_z_m"]
    )
    gripper_visual_distal_q0 = float(
        transformed["gripper_visual_in_link5_q_0.000000"]["bbox"]["distal_z_m"]
    )
    usd_config = _read_usd_config(USD_CONFIG)
    gripper_collision_size = transformed["gripper_collision_in_link5_q_0.000000"]["bbox"]["size_m"]
    gripper_collision_tiny = max(float(v) for v in gripper_collision_size) <= 0.006

    verdict = {
        "direct_gripper_collision_proxy_recommended": False,
        "direct_gripper_collision_rejection_reason": (
            "gripper_link collision mesh is a tiny g2a proxy, not the full fingertip/contact surface"
            if gripper_collision_tiny
            else "gripper_link collision needs runtime validation before replacing current proxy"
        ),
        "visual_gripper_surface_is_candidate": True,
        "visual_gripper_surface_caveat": (
            "gripper_link visual is not the physics collision asset because USD conversion has "
            "collision_from_visuals=false"
        ),
        "link5_collision_remains_current_runtime_proxy": True,
        "option2_direction": (
            "YES_AS_TOOL_SURFACE_UNION_NOT_AS_GRIPPER_LINK_COLLISION_ONLY"
        ),
        "option2_definition": (
            "Build a true tool-surface proxy from the union of the fixed jaw/link5 distal surface "
            "and the moving gripper visual or properly-authored collision surface. Do not replace "
            "the metric with gripper_link_collision_g2a alone."
        ),
        "requires_env_metric_change_before_ppo": True,
        "ppo_unblocked": False,
        "dataset_vla_roarm_unblocked": False,
    }

    return {
        "artifact_type": "cube10cm_tool_contact_proxy_asset_audit_d231",
        "scope": {
            "local_asset_audit_only": True,
            "isaaclab_runtime": False,
            "gpu": False,
            "training": False,
            "robot_control": False,
            "ssh": False,
        },
        "urdf": {
            "path": str(URDF),
            "link5_collision_mesh": str(link5_collision_path),
            "gripper_link_collision_mesh": str(gripper_collision_path),
            "gripper_link_visual_mesh": str(gripper_visual_path),
            "link5_to_gripper_link_origin_xyz_m": joint_xyz.tolist(),
            "link5_to_gripper_link_origin_rpy_rad": joint_rpy.tolist(),
        },
        "usd": {
            "config": usd_config,
            "pxr_stage_read": "not_available_in_local_python",
            "inference": (
                "USD was generated from URDF with collision_from_visuals=false, so the gripper_link "
                "physics collision is expected to come from gripper_link_collision_g2a.stl, not "
                "gripper_link.stl visual geometry."
            ),
        },
        "meshes_native_frame": {
            "link5_collision": _mesh_info(link5_collision_path, "link5", link5_collision),
            "gripper_link_collision": _mesh_info(gripper_collision_path, "gripper_link", gripper_collision),
            "gripper_link_visual": _mesh_info(gripper_visual_path, "gripper_link", gripper_visual),
        },
        "meshes_in_link5_frame": transformed,
        "key_comparison": {
            "link5_collision_distal_z_m": link5_distal_z,
            "gripper_link_collision_distal_z_m_at_q0": gripper_collision_distal_q0,
            "gripper_link_visual_distal_z_m_at_q0": gripper_visual_distal_q0,
            "hand_tcp_z_m": 0.115428,
            "link5_collision_minus_hand_tcp_m": link5_distal_z - 0.115428,
            "gripper_link_collision_max_size_m_at_q0": max(float(v) for v in gripper_collision_size),
            "gripper_link_collision_tiny_proxy": gripper_collision_tiny,
            "usd_collision_from_visuals": usd_config.get("collision_from_visuals"),
            "usd_collider_type": usd_config.get("collider_type"),
        },
        "verdict": verdict,
    }


def write_summary(audit: dict[str, Any], path: Path) -> None:
    comp = audit["key_comparison"]
    native = audit["meshes_native_frame"]
    verdict = audit["verdict"]
    usd = audit["usd"]["config"]
    lines = [
        "line1 artifact=cube10cm_tool_contact_proxy_asset_audit_d231 "
        "local_asset_audit_only=YES isaaclab_runtime=NO gpu=NO training=NO robot_control=NO ssh=NO",
        "line2 urdf_meshes "
        f"link5_collision={audit['urdf']['link5_collision_mesh']} "
        f"gripper_collision={audit['urdf']['gripper_link_collision_mesh']} "
        f"gripper_visual={audit['urdf']['gripper_link_visual_mesh']}",
        "line3 usd_collision_policy "
        f"config_exists={usd.get('exists')} collision_from_visuals={usd.get('collision_from_visuals')} "
        f"collider_type={usd.get('collider_type')} pxr_stage_read={audit['usd']['pxr_stage_read']}",
        "line4 native_bbox_size_m "
        f"link5_collision={native['link5_collision']['bbox']['size_m']} "
        f"gripper_collision={native['gripper_link_collision']['bbox']['size_m']} "
        f"gripper_visual={native['gripper_link_visual']['bbox']['size_m']}",
        "line5 link5_frame_q0 "
        f"link5_collision_distal_z={comp['link5_collision_distal_z_m']:.9f} "
        f"gripper_collision_distal_z={comp['gripper_link_collision_distal_z_m_at_q0']:.9f} "
        f"gripper_visual_distal_z={comp['gripper_link_visual_distal_z_m_at_q0']:.9f} "
        f"hand_tcp_z={comp['hand_tcp_z_m']:.9f}",
        "line6 gripper_collision_verdict "
        f"max_size_m={comp['gripper_link_collision_max_size_m_at_q0']:.9f} "
        f"tiny_proxy={comp['gripper_link_collision_tiny_proxy']} "
        f"direct_gripper_collision_proxy_recommended={verdict['direct_gripper_collision_proxy_recommended']}",
        "line7 option2_verdict "
        f"option2_direction={verdict['option2_direction']} "
        f"link5_current_runtime_proxy={verdict['link5_collision_remains_current_runtime_proxy']} "
        f"requires_env_metric_change_before_ppo={verdict['requires_env_metric_change_before_ppo']}",
        "line8 promotion "
        f"ppo_unblocked={verdict['ppo_unblocked']} dataset_vla_roarm_unblocked={verdict['dataset_vla_roarm_unblocked']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()
    audit = build_audit()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_summary(audit, args.out_summary)
    print(args.out_summary.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
