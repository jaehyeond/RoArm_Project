"""Small visual debugging helpers for RoArm geometry probes.

The functions in this module are intentionally optional-dependency tolerant:
Isaac Lab markers are used when the simulator stack is available, while
matplotlib snapshots remain available in headless/offline diagnostics.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROLE_COLORS = {
    "target": (0.95, 0.15, 0.10),
    "actual": (0.10, 0.65, 0.95),
    "link5": (0.20, 0.30, 0.95),
    "object": (0.95, 0.70, 0.10),
    "fixed_jaw": (0.85, 0.20, 0.85),
    "cube_face": (0.10, 0.10, 0.10),
    "candidate": (0.10, 0.75, 0.35),
    "other": (0.55, 0.55, 0.55),
}


def _as_np3(value: Any, *, default: Iterable[float] | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError("missing 3-vector")
        value = default
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"expected 3-vector, got shape {arr.shape}")
    return arr


def _unit(value: Any, *, fallback: Iterable[float]) -> np.ndarray:
    arr = _as_np3(value, default=fallback)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-12:
        arr = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
    return arr / max(norm, 1.0e-12)


def rot_to_quat_wxyz(rot: Any) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion in Isaac's wxyz order."""
    matrix = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(matrix)))
        if idx == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1.0e-12)


def quat_wxyz_to_rot(quat: Any) -> np.ndarray:
    """Convert a quaternion in wxyz order to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def frame_from_axes(
    name: str,
    position: Any,
    *,
    x_axis: Any,
    z_axis: Any,
    role: str = "other",
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized frame dict from x and z axes.

    The y axis is derived as z x x, then z is re-orthogonalized as x x y.
    """
    x_axis_np = _unit(x_axis, fallback=(1.0, 0.0, 0.0))
    z_axis_np = _unit(z_axis, fallback=(0.0, 0.0, 1.0))
    y_axis_np = np.cross(z_axis_np, x_axis_np)
    if float(np.linalg.norm(y_axis_np)) <= 1.0e-8:
        y_axis_np = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis_np = _unit(y_axis_np, fallback=(0.0, 1.0, 0.0))
    z_axis_np = _unit(np.cross(x_axis_np, y_axis_np), fallback=z_axis_np)
    rotation = np.column_stack([x_axis_np, y_axis_np, z_axis_np])
    return {
        "name": str(name),
        "label": label or str(name),
        "position": _as_np3(position).tolist(),
        "rotation_matrix": rotation.tolist(),
        "quat_wxyz": rot_to_quat_wxyz(rotation).tolist(),
        "role": role,
        "metadata": metadata or {},
    }


def normalize_frame(pair: Any) -> dict[str, Any]:
    """Normalize a `(name, pose)` pair or dict into a frame dictionary."""
    if isinstance(pair, tuple) and len(pair) == 2:
        name, pose = pair
        pose = dict(pose)
        pose.setdefault("name", name)
    elif isinstance(pair, dict):
        pose = dict(pair)
        pose.setdefault("name", pose.get("label", "frame"))
    else:
        raise TypeError(f"unsupported frame pair: {type(pair)!r}")

    position = _as_np3(pose.get("position", pose.get("pos_local_m", pose.get("pos"))))
    if "rotation_matrix" in pose:
        rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    elif "quat_wxyz" in pose:
        rotation = quat_wxyz_to_rot(pose["quat_wxyz"])
    elif "axes" in pose:
        axes = pose["axes"]
        rotation = np.column_stack(
            [
                _unit(axes.get("x"), fallback=(1.0, 0.0, 0.0)),
                _unit(axes.get("y"), fallback=(0.0, 1.0, 0.0)),
                _unit(axes.get("z"), fallback=(0.0, 0.0, 1.0)),
            ]
        )
    elif "x_axis" in pose and "z_axis" in pose:
        return frame_from_axes(
            str(pose["name"]),
            position,
            x_axis=pose["x_axis"],
            z_axis=pose["z_axis"],
            role=str(pose.get("role", "other")),
            label=pose.get("label"),
            metadata=pose.get("metadata"),
        )
    else:
        rotation = np.eye(3, dtype=np.float64)

    return {
        "name": str(pose["name"]),
        "label": str(pose.get("label", pose["name"])),
        "position": position.tolist(),
        "rotation_matrix": rotation.tolist(),
        "quat_wxyz": rot_to_quat_wxyz(rotation).tolist(),
        "role": str(pose.get("role", "other")),
        "metadata": dict(pose.get("metadata", {})),
    }


def draw_frames(
    pairs: Iterable[Any],
    *,
    prim_path: str = "/World/DebugFrames",
    scale: float = 0.06,
) -> dict[str, Any]:
    """Draw frame axes in an Isaac stage using VisualizationMarkers.

    This is safe to call from non-Isaac contexts: it returns an error status
    instead of raising if Isaac Lab is unavailable.
    """
    frames = [normalize_frame(pair) for pair in pairs]
    if not frames:
        return {"ok": False, "backend": "isaac_markers", "error": "no frames supplied"}
    try:
        import torch
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG
    except Exception as exc:  # pragma: no cover - depends on Isaac runtime
        return {"ok": False, "backend": "isaac_markers", "error": repr(exc)}

    try:
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = f"{prim_path}/frames"
        marker = VisualizationMarkers(marker_cfg)
        positions = torch.tensor([frame["position"] for frame in frames], dtype=torch.float32)
        orientations = torch.tensor([frame["quat_wxyz"] for frame in frames], dtype=torch.float32)
        scales = torch.full((len(frames), 3), float(scale), dtype=torch.float32)
        marker.visualize(positions, orientations, scales=scales)
        return {
            "ok": True,
            "backend": "isaac_markers",
            "prim_path": marker_cfg.prim_path,
            "frame_count": len(frames),
        }
    except Exception as exc:  # pragma: no cover - depends on Isaac runtime
        return {"ok": False, "backend": "isaac_markers", "error": repr(exc)}


def _set_axes_equal(ax: Any, points: np.ndarray, margin: float = 0.06) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) * 0.5 + margin, margin)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _cube_edges(center: np.ndarray, size: float) -> list[tuple[np.ndarray, np.ndarray]]:
    half = float(size) * 0.5
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(center + half * np.asarray([sx, sy, sz], dtype=np.float64))
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for i, a in enumerate(corners):
        for b in corners[i + 1 :]:
            if np.count_nonzero(np.isclose(np.abs(a - b), float(size))) == 1:
                edges.append((a, b))
    return edges


def snapshot_frame_plot(
    path: str | Path,
    pairs: Iterable[Any],
    *,
    cube: dict[str, Any] | None = None,
    title: str | None = None,
    annotations: Iterable[str] | None = None,
    axis_length: float = 0.045,
    view: tuple[float, float] = (25.0, -55.0),
) -> dict[str, Any]:
    """Save an offline 3D diagnostic PNG for a list of frames."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = [normalize_frame(pair) for pair in pairs]
    if not frames:
        return {"ok": False, "backend": "matplotlib", "error": "no frames supplied"}

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 10), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=float(view[0]), azim=float(view[1]))
    ax.set_title(title or "RoArm frame debug", pad=16)
    ax.set_xlabel("x / radial (m)")
    ax.set_ylabel("y / tangent (m)")
    ax.set_zlabel("z (m)")

    all_points: list[np.ndarray] = []
    if cube is not None:
        center = _as_np3(cube.get("center"), default=(0.3, 0.0, 0.04))
        size = float(cube.get("size", 0.10))
        for a, b in _cube_edges(center, size):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="0.20", linewidth=1.2)
            all_points.extend([a, b])
        ax.scatter([center[0]], [center[1]], [center[2]], color=ROLE_COLORS["object"], s=40, label="cube center")

    for frame in frames:
        pos = np.asarray(frame["position"], dtype=np.float64)
        rot = np.asarray(frame["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        role = str(frame.get("role", "other"))
        role_color = ROLE_COLORS.get(role, ROLE_COLORS["other"])
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color=role_color, s=56, depthshade=False)
        ax.text(pos[0], pos[1], pos[2], f"  {frame.get('label', frame['name'])}", fontsize=8)
        for axis_idx, axis_color, axis_name in (
            (0, "red", "x"),
            (1, "green", "y"),
            (2, "blue", "z"),
        ):
            vec = rot[:, axis_idx] * float(axis_length)
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                vec[0],
                vec[1],
                vec[2],
                color=axis_color,
                arrow_length_ratio=0.18,
                linewidth=2.0,
            )
            tip = pos + vec
            ax.text(tip[0], tip[1], tip[2], axis_name, color=axis_color, fontsize=8)
            all_points.extend([pos, tip])

    note_lines = list(annotations or [])
    if note_lines:
        fig.text(
            0.02,
            0.02,
            "\n".join(str(line) for line in note_lines),
            fontsize=10,
            family="monospace",
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "0.70", "alpha": 0.88},
        )

    all_arr = np.vstack(all_points) if all_points else np.zeros((1, 3), dtype=np.float64)
    _set_axes_equal(ax, all_arr)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {"ok": True, "backend": "matplotlib", "path": str(out_path), "frame_count": len(frames)}


def snapshot(
    path: str | Path,
    *,
    pairs: Iterable[Any] | None = None,
    cube: dict[str, Any] | None = None,
    title: str | None = None,
    annotations: Iterable[str] | None = None,
    prefer_viewport: bool = True,
) -> dict[str, Any]:
    """Save one diagnostic PNG.

    The function first tries a live viewport capture when requested.  If that is
    unavailable and frame data was supplied, it falls back to `snapshot_frame_plot`.
    """
    viewport_error: str | None = None
    if prefer_viewport:
        try:
            import omni.kit.viewport.utility as viewport_utility

            viewport = viewport_utility.get_active_viewport()
            if viewport is None:
                raise RuntimeError("no active viewport")
            capture = getattr(viewport_utility, "capture_viewport_to_file", None)
            if capture is None:
                raise RuntimeError("capture_viewport_to_file unavailable")
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            capture(viewport, str(out_path))
            return {"ok": True, "backend": "viewport", "path": str(out_path)}
        except Exception as exc:  # pragma: no cover - depends on GUI/Kit
            viewport_error = repr(exc)

    if pairs is None:
        return {"ok": False, "backend": "viewport", "error": viewport_error or "no frame data"}
    status = snapshot_frame_plot(path, pairs, cube=cube, title=title, annotations=annotations)
    if viewport_error:
        status["viewport_error"] = viewport_error
    return status


def log_rerun(
    path: str | Path,
    *,
    frames: Iterable[Any] | None = None,
    joint_state: dict[str, Any] | None = None,
    urdf_path: str | Path | None = None,
) -> dict[str, Any]:
    """Optionally write a small rerun `.rrd` geometry log."""
    try:
        import rerun as rr
    except Exception as exc:
        return {"ok": False, "backend": "rerun", "error": repr(exc)}

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rr.init("roarm_viz_debug", spawn=False)
        if urdf_path is not None:
            rr.log("urdf_path", rr.TextDocument(str(urdf_path)))
        if joint_state is not None:
            rr.log("joint_state", rr.TextDocument(str(joint_state)))
        for frame in [normalize_frame(pair) for pair in (frames or [])]:
            pos = np.asarray(frame["position"], dtype=np.float64)
            quat = np.asarray(frame["quat_wxyz"], dtype=np.float64)
            rr.log(
                f"frames/{frame['name']}",
                rr.Transform3D(translation=pos, rotation=rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]])),
            )
        rr.save(str(out_path))
        return {"ok": True, "backend": "rerun", "path": str(out_path)}
    except Exception as exc:
        return {"ok": False, "backend": "rerun", "error": repr(exc)}
