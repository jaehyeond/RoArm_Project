"""Small visual debugging helpers for RoArm geometry probes.

The functions in this module are intentionally optional-dependency tolerant:
Isaac Lab markers are used when the simulator stack is available, while
matplotlib snapshots remain available in headless/offline diagnostics.
"""
from __future__ import annotations

import math
import os
import json
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
        metadata = dict(frame.get("metadata", {}))
        role_color = ROLE_COLORS.get(role, ROLE_COLORS["other"])
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color=role_color, s=56, depthshade=False)
        label_pos = pos + _as_np3(metadata.get("label_offset"), default=(0.0, 0.0, 0.0))
        ax.text(label_pos[0], label_pos[1], label_pos[2], f"  {frame.get('label', frame['name'])}", fontsize=8)
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
            if bool(metadata.get("show_axis_labels", True)):
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
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper left", fontsize=8)
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


def _rerun_entity_path(value: Any) -> str:
    path = str(value).strip().strip("/")
    if not path or "//" in path or "\\" in path:
        raise ValueError(f"invalid Rerun entity path: {value!r}")
    return path


def _rerun_component_name(value: Any) -> str:
    text = str(value).strip().replace("/", "_").replace("\\", "_")
    return text or "value"


def build_rerun_blueprint(mode: str = "robot_geometry") -> Any:
    """Build the fixed Rerun layout used by RoArm observability cases."""
    import rerun.blueprint as rrb

    if mode == "authored_frame_contract":
        def _body_row(body: str, label: str) -> Any:
            return rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents=f"/frame_contract/direct_authored/{body}/**",
                    name=f"{label} direct authored x0 (prim-local)",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=f"/frame_contract/body_mapped_x0/{body}/**",
                    name=f"{label} body-mapped x0",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=[
                        f"/frame_contract/direct_authored/{body}/**",
                        f"/frame_contract/body_mapped_x0/{body}/**",
                    ],
                    name=f"{label} direct vs mapped x0 overlay",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=[
                        f"/frame_contract/body_mapped_x0/{body}/**",
                        f"/frame_contract/body_mapped_x1/{body}/**",
                    ],
                    name=f"{label} mapped x0 vs candidate x1",
                ),
                column_shares=[0.25, 0.25, 0.25, 0.25],
            )

        return rrb.Blueprint(
            rrb.Vertical(
                _body_row("link5", "link5"),
                _body_row("gripper_link", "gripper"),
                rrb.Horizontal(
                    rrb.DataframeView(
                        origin="/metrics",
                        contents="/metrics/**",
                        name="Float64 frame metrics",
                    ),
                    rrb.DataframeView(
                        origin="/gate",
                        contents="/gate/**",
                        name="per-part gate state",
                    ),
                    rrb.TextLogView(
                        origin="/events",
                        contents="/events/**",
                        name="authored-frame events",
                    ),
                    column_shares=[0.52, 0.18, 0.30],
                ),
                row_shares=[0.35, 0.35, 0.30],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=True,
        )
    if mode == "collision_gate":
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="/",
                        contents=["/cook/source/link5/**", "/frames/**"],
                        name="link5 authored source (x0)",
                    ),
                    rrb.Spatial3DView(
                        origin="/",
                        contents=["/cook/instance/link5/**", "/frames/**"],
                        name="link5 live instance (x1)",
                    ),
                    rrb.Spatial3DView(
                        origin="/",
                        contents=["/cook/prototype/link5/**", "/frames/**"],
                        name="link5 prototype (x1)",
                    ),
                    rrb.Spatial3DView(
                        origin="/",
                        contents=["/cook/candidate/link5/**", "/frames/**"],
                        name="link5 candidate (x1)",
                    ),
                    column_shares=[0.25, 0.25, 0.25, 0.25],
                ),
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="/",
                        contents=["/cook/source/gripper_link/**", "/frames/**"],
                        name="gripper authored source (x0)",
                    ),
                    rrb.Spatial3DView(
                        origin="/",
                        contents=["/cook/instance/gripper_link/**", "/frames/**"],
                        name="gripper live instance (x1)",
                    ),
                    rrb.Spatial3DView(
                        origin="/",
                        contents=["/cook/prototype/gripper_link/**", "/frames/**"],
                        name="gripper prototype (x1)",
                    ),
                    rrb.Spatial3DView(
                        origin="/",
                        contents=["/cook/candidate/gripper_link/**", "/frames/**"],
                        name="gripper candidate (x1)",
                    ),
                    column_shares=[0.25, 0.25, 0.25, 0.25],
                ),
                rrb.Horizontal(
                    rrb.DataframeView(origin="/metrics", contents="/metrics/**", name="Float64 gate metrics"),
                    rrb.Tabs(
                        rrb.TextLogView(origin="/events", contents="/events/**", name="cook/gate events"),
                        rrb.DataframeView(origin="/gate", contents="/gate/**", name="gate state"),
                        active_tab=0,
                    ),
                    column_shares=[0.58, 0.42],
                ),
                row_shares=[0.35, 0.35, 0.30],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=True,
        )
    if mode == "volume_semantics":
        def _volume_body_row(body: str, label: str, suffix: str = "**") -> Any:
            return rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents=[f"/cook/source/{body}/{suffix}"],
                    name=f"{label} instance: callback face topology",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=[f"/cook/instance/{body}/{suffix}"],
                    name=f"{label} instance: vertex-only Qhull envelope",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=[f"/cook/prototype/{body}/{suffix}"],
                    name=f"{label} prototype: callback face topology",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=[f"/cook/candidate/{body}/{suffix}"],
                    name=f"{label} prototype: vertex-only Qhull envelope",
                ),
                column_shares=[0.25, 0.25, 0.25, 0.25],
            )

        return rrb.Blueprint(
            rrb.Vertical(
                _volume_body_row("link5", "link5 part_045 zoom", "parts/part_045"),
                _volume_body_row("gripper_link", "gripper full body"),
                rrb.Horizontal(
                    rrb.DataframeView(
                        origin="/metrics",
                        contents="/metrics/**",
                        name="Float64 topology/property volume metrics",
                    ),
                    rrb.TextLogView(
                        origin="/events",
                        contents="/events/**",
                        name="D348 semantic gates",
                    ),
                    column_shares=[0.65, 0.35],
                ),
                row_shares=[0.35, 0.35, 0.30],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=True,
        )
    if mode == "volume_semantics_summary":
        def _summary_body_row(body: str, label: str, suffix: str = "**") -> Any:
            return rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents=[f"/cook/source/{body}/{suffix}"],
                    name=f"{label} instance: callback face topology",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=[f"/cook/instance/{body}/{suffix}"],
                    name=f"{label} instance: vertex-only Qhull envelope",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=[f"/cook/prototype/{body}/{suffix}"],
                    name=f"{label} prototype: callback face topology",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=[f"/cook/candidate/{body}/{suffix}"],
                    name=f"{label} prototype: vertex-only Qhull envelope",
                ),
                column_shares=[0.25, 0.25, 0.25, 0.25],
            )

        return rrb.Blueprint(
            rrb.Vertical(
                _summary_body_row("link5", "link5 part_045 zoom", "parts/part_045"),
                _summary_body_row("gripper_link", "gripper full body"),
                rrb.Horizontal(
                    rrb.TextDocumentView(
                        origin="/metadata/run",
                        contents="/metadata/run",
                        name="D348 static scientific summary",
                    ),
                    rrb.TextLogView(
                        origin="/events",
                        contents="/events/**",
                        name="D348 static completion event",
                    ),
                    column_shares=[0.68, 0.32],
                ),
                row_shares=[0.34, 0.34, 0.32],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=True,
        )
    if mode == "d357_beginner_result":
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial3DView(
                    origin="/",
                    contents=[
                        "/actual_robot/**",
                        "/frames/**",
                        "/geometry/**",
                    ],
                    name="D354 robot + cylinder (display-only replay)",
                    eye_controls=rrb.EyeControls3D(
                        kind=rrb.Eye3DKind.Orbital,
                        position=(0.48, -0.42, 0.30),
                        look_target=(0.22, 0.0, 0.08),
                        eye_up=(0.0, 0.0, 1.0),
                    ),
                    spatial_information=rrb.SpatialInformation(
                        target_frame="tf#/",
                        show_axes=True,
                        show_bounding_box=False,
                    ),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/metrics/d357",
                        contents="/metrics/d357/**",
                        name="q5 and signed distance",
                    ),
                    rrb.TextLogView(
                        origin="/events/d357",
                        contents="/events/d357/**",
                        name="three display poses (no force test)",
                    ),
                    column_shares=[0.55, 0.45],
                ),
                row_shares=[0.76, 0.24],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=True,
        )
    if mode == "d368_semantic_allocation":
        def _semantic_view(
            *,
            name: str,
            contents: list[str],
            position: tuple[float, float, float],
            look_target: tuple[float, float, float],
        ) -> Any:
            return rrb.Spatial3DView(
                origin="/",
                contents=contents,
                name=name,
                eye_controls=rrb.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                    position=position,
                    look_target=look_target,
                    eye_up=(0.0, 0.0, 1.0),
                ),
                spatial_information=rrb.SpatialInformation(
                    target_frame="tf#/",
                    show_axes=True,
                    show_bounding_box=False,
                ),
            )

        link5_full_contents = [
            "/semantic/source/link5/**",
            "/semantic/collider/link5/**",
            "/semantic/anchors/link5/**",
            "/semantic/normals/link5/**",
        ]
        link5_zoom_contents = [
            "/semantic/source/link5/seed_contact_plane_patch",
            "/semantic/collider/link5/certified_seed_patch_carrier/**",
            "/semantic/anchors/link5/**",
            "/semantic/normals/link5/**",
        ]
        gripper_full_contents = [
            "/semantic/source/gripper_link/**",
            "/semantic/collider/gripper_link/**",
            "/semantic/anchors/gripper_link/**",
            "/semantic/normals/gripper_link/**",
        ]
        gripper_zoom_contents = [
            "/semantic/source/gripper_link/inner_contact_patch",
            "/semantic/source/gripper_link/outer_negative_patch",
            "/semantic/collider/gripper_link/certified_inner_patch_carrier/**",
            "/semantic/collider/gripper_link/dual_inner_outer_patch_carrier/**",
            "/semantic/collider/gripper_link/outer_negative_patch_carrier/**",
            "/semantic/anchors/gripper_link/**",
            "/semantic/normals/gripper_link/**",
        ]
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    _semantic_view(
                        name="link5: all 64 current hulls",
                        contents=link5_full_contents,
                        position=(0.18, -0.22, 0.16),
                        look_target=(-0.005, 0.0, 0.065),
                    ),
                    _semantic_view(
                        name="link5: D350 seed-plane patch zoom",
                        contents=link5_zoom_contents,
                        position=(0.08, -0.10, 0.14),
                        look_target=(-0.010, 0.0, 0.100),
                    ),
                    column_shares=[0.5, 0.5],
                ),
                rrb.Horizontal(
                    _semantic_view(
                        name="moving jaw: all 64 current hulls",
                        contents=gripper_full_contents,
                        position=(0.15, -0.18, 0.08),
                        look_target=(0.030, 0.0, -0.018),
                    ),
                    _semantic_view(
                        name="moving jaw: frozen inner patch zoom",
                        contents=gripper_zoom_contents,
                        position=(0.10, -0.10, 0.04),
                        look_target=(0.045, -0.006, -0.020),
                    ),
                    column_shares=[0.5, 0.5],
                ),
                rrb.Horizontal(
                    rrb.DataframeView(
                        origin="/metrics/d368",
                        contents="/metrics/d368/**",
                        name="allocation counts and geometry budget",
                    ),
                    rrb.TextLogView(
                        origin="/events/d368_summary",
                        contents="/events/d368_summary",
                        name="legend and scope boundary",
                    ),
                    column_shares=[0.58, 0.42],
                ),
                row_shares=[0.38, 0.38, 0.24],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=True,
        )
    if mode != "robot_geometry":
        raise ValueError(f"unsupported Rerun blueprint mode: {mode!r}")
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(
                origin="/",
                contents=[
                    "/actual_robot/**",
                    "/commanded_robot/**",
                    "/frames/**",
                    "/cube/**",
                    "/geometry/**",
                    "/contacts/**",
                ],
                name="robot + decision geometry",
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/metrics", contents="/metrics/**", name="Float64 metrics"),
                rrb.TextLogView(origin="/events", contents="/events/**", name="events and gates"),
                column_shares=[0.62, 0.38],
            ),
            row_shares=[0.72, 0.28],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _normalize_rerun_meshes(meshes: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw in meshes or []:
        row = dict(raw)
        entity_path = _rerun_entity_path(row.get("entity_path"))
        if entity_path in seen_paths:
            raise ValueError(f"duplicate Rerun mesh entity path: {entity_path}")
        seen_paths.add(entity_path)
        coordinate_frame = str(row.get("coordinate_frame", "")).strip()
        if not coordinate_frame:
            raise ValueError(f"{entity_path}: coordinate_frame is required")
        vertices = np.asarray(row.get("vertices_m"), dtype=np.float64)
        triangles = np.asarray(row.get("triangles"), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
            raise ValueError(f"{entity_path}: vertices_m must have shape (N>=3, 3)")
        if not np.isfinite(vertices).all():
            raise ValueError(f"{entity_path}: vertices_m contains NaN/Inf")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] < 1:
            raise ValueError(f"{entity_path}: triangles must have shape (M>=1, 3)")
        if int(triangles.min()) < 0 or int(triangles.max()) >= int(vertices.shape[0]):
            raise ValueError(f"{entity_path}: triangle index is out of range")
        color = [int(value) for value in row.get("color_rgba", [140, 140, 140, 110])]
        if len(color) != 4 or any(value < 0 or value > 255 for value in color):
            raise ValueError(f"{entity_path}: color_rgba must contain four uint8 values")
        row.update(
            {
                "entity_path": entity_path,
                "coordinate_frame": coordinate_frame,
                "vertices_m": vertices,
                "triangles": triangles,
                "color_rgba": color,
            }
        )
        normalized.append(row)
    return normalized


def _normalize_rerun_coordinate_frames(
    coordinate_frames: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_frames: set[str] = set()
    seen_entities: set[str] = set()
    for raw in coordinate_frames or []:
        row = dict(raw)
        frame = str(row.get("frame", "")).strip()
        parent_frame = str(row.get("parent_frame", "tf#/")).strip()
        entity_path = _rerun_entity_path(row.get("entity_path", f"coordinate_frames/{frame}"))
        if not frame or not parent_frame:
            raise ValueError("coordinate frame and parent_frame must be non-empty")
        if frame in seen_frames or entity_path in seen_entities:
            raise ValueError(f"duplicate Rerun coordinate frame: {frame!r} / {entity_path!r}")
        translation = np.asarray(row.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        quaternion = np.asarray(row.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64).reshape(4)
        if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
            raise ValueError(f"{entity_path}: coordinate-frame transform contains NaN/Inf")
        quat_norm = float(np.linalg.norm(quaternion))
        if quat_norm <= 1.0e-12:
            raise ValueError(f"{entity_path}: coordinate-frame quaternion has zero norm")
        seen_frames.add(frame)
        seen_entities.add(entity_path)
        row.update(
            {
                "frame": frame,
                "parent_frame": parent_frame,
                "entity_path": entity_path,
                "translation_m": translation,
                "quaternion_xyzw": quaternion / quat_norm,
            }
        )
        normalized.append(row)
    return normalized


def _set_rerun_row_times(recording: Any, row: dict[str, Any]) -> set[str]:
    recording.reset_time()
    names: set[str] = set()
    for name, value in dict(row.get("sequence", {})).items():
        recording.set_time(str(name), sequence=int(value))
        names.add(str(name))
    for name, value in dict(row.get("timestamp", {})).items():
        recording.set_time(str(name), timestamp=float(value))
        names.add(str(name))
    return names


def log_rerun(
    path: str | Path,
    *,
    frames: Iterable[Any] | None = None,
    joint_state: dict[str, Any] | None = None,
    urdf_path: str | Path | None = None,
    joint_trace: Iterable[dict[str, Any]] | None = None,
    cube: dict[str, Any] | None = None,
    coordinate_frames: Iterable[dict[str, Any]] | None = None,
    meshes: Iterable[dict[str, Any]] | None = None,
    points: Iterable[dict[str, Any]] | None = None,
    arrows: Iterable[dict[str, Any]] | None = None,
    scalar_trace: Iterable[dict[str, Any]] | None = None,
    events: Iterable[dict[str, Any]] | None = None,
    recording_metadata: dict[str, Any] | None = None,
    recording_id: str | None = None,
    blueprint_path: str | Path | None = None,
    blueprint_mode: str = "robot_geometry",
    live_viewer: bool = False,
    app_id: str = "roarm_viz_debug",
) -> dict[str, Any]:
    """Write and finalize a replayable Rerun observability artifact.

    The file sink is attached before the first user log and is finalized via a
    dedicated ``RecordingStream`` context.  ``ok`` requires a non-empty file,
    a complete footer, a verified blueprint, and the registered entity/timeline
    paths.  Viewer geometry is Float32 observability data; authoritative hashes
    and bit-exact decisions must continue to use the original arrays/JSON.
    """
    try:
        import rerun as rr
        from roarm_rl.rerun_contract import RERUN_CONTRACT_VERSION, validate_rerun_artifact
    except Exception as exc:
        return {"ok": False, "backend": "rerun", "error": repr(exc)}

    if str(rr.__version__) != RERUN_CONTRACT_VERSION:
        return {
            "ok": False,
            "backend": "rerun",
            "error": f"rerun-sdk pin mismatch: {rr.__version__} != {RERUN_CONTRACT_VERSION}",
        }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return {"ok": False, "backend": "rerun", "error": f"refusing to overwrite {out_path}"}
    rbl_path = Path(blueprint_path) if blueprint_path is not None else out_path.with_suffix(".rbl")
    if rbl_path.exists():
        return {"ok": False, "backend": "rerun", "error": f"refusing to overwrite {rbl_path}"}

    try:
        frames_norm = [normalize_frame(pair) for pair in (frames or [])]
        trace_rows = [dict(row) for row in (joint_trace or [])]
        coordinate_frame_rows = _normalize_rerun_coordinate_frames(coordinate_frames)
        mesh_rows = _normalize_rerun_meshes(meshes)
        point_rows = [dict(row) for row in (points or [])]
        arrow_rows = [dict(row) for row in (arrows or [])]
        scalar_rows = [dict(row) for row in (scalar_trace or [])]
        event_rows = [dict(row) for row in (events or [])]
        blueprint = build_rerun_blueprint(blueprint_mode)
        declared_frames = {row["frame"] for row in coordinate_frame_rows} | {"tf#/"}
        missing_frames = sorted(
            {
                str(row["coordinate_frame"])
                for row in [*mesh_rows, *point_rows, *arrow_rows]
                if row.get("coordinate_frame")
                and str(row["coordinate_frame"]) not in declared_frames
            }
        )
        if missing_frames:
            raise ValueError(f"spatial coordinate frames were not declared: {missing_frames}")
    except Exception as exc:
        return {"ok": False, "backend": "rerun", "error": repr(exc)}

    urdf_actual_status: dict[str, Any] = {"attempted": False}
    urdf_commanded_status: dict[str, Any] = {"attempted": False}
    blueprint_status: dict[str, Any] = {"attempted": True, "ok": False, "path": str(rbl_path)}
    expected_entities: set[str] = set()
    expected_timelines: set[str] = set()
    sink_attached_before_logging = False
    sink_finalized = False
    flush_ok = False
    actual_joint_count = 0
    commanded_joint_count = 0

    try:
        with rr.RecordingStream(
            app_id,
            recording_id=recording_id,
            make_default=False,
            send_properties=True,
        ) as recording:
            if live_viewer:
                recording.spawn(connect=False)
                recording.set_sinks(
                    rr.GrpcSink(),
                    rr.FileSink(str(out_path), write_footer=True),
                )
            else:
                recording.save(str(out_path), write_footer=True)
            sink_attached_before_logging = True

            recording.send_blueprint(blueprint, make_active=True, make_default=True)
            metadata = {
                "application_id": app_id,
                "recording_id": recording_id,
                "rerun_sdk_version": str(rr.__version__),
                "scientific_authority": "original callback arrays / canonical JSON / hashes",
                "viewer_geometry_role": "Float32 spatial observability copy",
                **dict(recording_metadata or {}),
            }
            recording.log(
                "metadata/run",
                rr.TextDocument(
                    json.dumps(
                        metadata,
                        indent=2,
                        sort_keys=True,
                        default=str,
                        ensure_ascii=False,
                    )
                ),
                static=True,
            )
            expected_entities.add("metadata/run")

            for row in coordinate_frame_rows:
                quat = row["quaternion_xyzw"]
                recording.log(
                    row["entity_path"],
                    rr.Transform3D(
                        translation=row["translation_m"],
                        rotation=rr.Quaternion(xyzw=quat),
                        parent_frame=row["parent_frame"],
                        child_frame=row["frame"],
                    ),
                    static=True,
                )
                expected_entities.add(row["entity_path"])

            actual_tree = None
            commanded_tree = None
            if urdf_path is not None:
                recording.log("metadata/urdf_path", rr.TextDocument(str(urdf_path)), static=True)
                expected_entities.add("metadata/urdf_path")
                try:
                    from rerun.urdf import UrdfTree

                    actual_tree = UrdfTree.from_file_path(
                        urdf_path,
                        entity_path_prefix="actual_robot",
                        frame_prefix="actual/",
                        static_transform_entity_path="actual_robot/tf_static",
                    )
                    actual_tree.log_urdf_to_recording(recording)
                    urdf_actual_status = {
                        "attempted": True,
                        "ok": True,
                        "name": actual_tree.name,
                        "joint_count": len(actual_tree.joints()),
                    }
                    expected_entities.add(f"actual_robot/{actual_tree.name}")
                    if trace_rows:
                        commanded_tree = UrdfTree.from_file_path(
                            urdf_path,
                            entity_path_prefix="commanded_robot",
                            frame_prefix="commanded/",
                            static_transform_entity_path="commanded_robot/tf_static",
                        )
                        commanded_tree.log_urdf_to_recording(recording)
                        urdf_commanded_status = {
                            "attempted": True,
                            "ok": True,
                            "name": commanded_tree.name,
                            "joint_count": len(commanded_tree.joints()),
                        }
                        expected_entities.add(f"commanded_robot/{commanded_tree.name}")
                except Exception as exc:
                    urdf_actual_status = {"attempted": True, "ok": False, "error": repr(exc)}
                    if trace_rows:
                        urdf_commanded_status = {"attempted": True, "ok": False, "error": repr(exc)}

            if joint_state is not None:
                recording.log(
                    "metadata/joint_state",
                    rr.TextDocument(json.dumps(joint_state, sort_keys=True, default=str)),
                    static=True,
                )
                expected_entities.add("metadata/joint_state")
            if cube is not None:
                center = _as_np3(cube.get("center"), default=(0.3, 0.0, 0.04))
                size = float(cube.get("size", 0.10))
                recording.log(
                    "cube/body",
                    rr.Boxes3D(
                        centers=[center.tolist()],
                        sizes=[[size, size, size]],
                        colors=[[240, 190, 40, 70]],
                        labels=[str(cube.get("label", "cube"))],
                    ),
                    static=True,
                )
                expected_entities.add("cube/body")

            def _log_frames(frame_set: list[dict[str, Any]], *, static: bool) -> None:
                for frame in frame_set:
                    name = _rerun_component_name(frame["name"])
                    pos = np.asarray(frame["position"], dtype=np.float64)
                    quat = np.asarray(frame["quat_wxyz"], dtype=np.float64)
                    recording.log(
                        f"frames/{name}",
                        rr.Transform3D(
                            translation=pos,
                            rotation=rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]]),
                            parent_frame="tf#/",
                            child_frame=f"debug/{name}",
                        ),
                        static=static,
                    )
                    recording.log(
                        f"frames/{name}/origin",
                        rr.Points3D(
                            [[0.0, 0.0, 0.0]],
                            radii=[0.006],
                            labels=[frame.get("label", frame["name"])],
                        ),
                        rr.CoordinateFrame(f"debug/{name}"),
                        static=static,
                    )
                    expected_entities.add(f"frames/{name}")
                    expected_entities.add(f"frames/{name}/origin")

            def _log_tree_joints(tree: Any, prefix: str, joint_values: dict[str, float]) -> int:
                if tree is None:
                    return 0
                logged = 0
                for joint in tree.joints():
                    value = float(joint_values.get(joint.name, 0.0))
                    recording.log(f"{prefix}/joints/{joint.name}", joint.compute_transform(value, clamp=True))
                    logged += 1
                return logged

            if not trace_rows:
                recording.reset_time()
                _log_frames(frames_norm, static=True)
            for row in trace_rows:
                recording.reset_time()
                step = int(row.get("step", 0))
                recording.set_time("step", sequence=step)
                expected_timelines.add("step")
                actual_joints = dict(row.get("actual_joint_rad_by_name", {}))
                commanded_joints = dict(row.get("commanded_joint_rad_by_name", {}))
                actual_joint_count = max(
                    actual_joint_count,
                    _log_tree_joints(actual_tree, "actual_robot", actual_joints),
                )
                commanded_joint_count = max(
                    commanded_joint_count,
                    _log_tree_joints(commanded_tree, "commanded_robot", commanded_joints),
                )
                frame_set = [normalize_frame(pair) for pair in row.get("frames", frames_norm)]
                _log_frames(frame_set, static=False)
                diagnostics = {key: value for key, value in row.items() if key != "frames"}
                recording.log(
                    "metadata/step_diagnostics",
                    rr.TextDocument(json.dumps(diagnostics, sort_keys=True, default=str)),
                )
                expected_entities.add("metadata/step_diagnostics")
                for key, value in row.items():
                    if key == "step" or isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                        continue
                    scalar = float(value)
                    if not math.isfinite(scalar):
                        continue
                    scalar_path = f"metrics/joint_trace/{_rerun_component_name(key)}"
                    recording.log(scalar_path, rr.Scalars([scalar]))
                    expected_entities.add(scalar_path)

            if trace_rows:
                recording.reset_time()
                recording.log(
                    "metadata/joint_trace_summary",
                    rr.TextDocument(
                        json.dumps(
                            {
                                "steps": len(trace_rows),
                                "actual_joint_count": actual_joint_count,
                                "commanded_joint_count": commanded_joint_count,
                            },
                            sort_keys=True,
                        )
                    ),
                    static=True,
                )
                expected_entities.add("metadata/joint_trace_summary")

            for row in mesh_rows:
                recording.reset_time()
                expected_timelines.update(_set_rerun_row_times(recording, row))
                entity_path = row["entity_path"]
                recording.log(
                    entity_path,
                    rr.Mesh3D(
                        vertex_positions=row["vertices_m"].astype(np.float32),
                        triangle_indices=row["triangles"].astype(np.uint32),
                        albedo_factor=row["color_rgba"],
                    ),
                    rr.CoordinateFrame(row["coordinate_frame"]),
                    static=bool(row.get("static", True)),
                )
                mesh_metadata = {
                    key: value
                    for key, value in row.items()
                    if key not in {"vertices_m", "triangles", "sequence", "timestamp"}
                }
                recording.log(
                    f"metadata/meshes/{entity_path.replace('/', '__')}",
                    rr.TextDocument(json.dumps(mesh_metadata, sort_keys=True, default=str)),
                    static=bool(row.get("static", True)),
                )
                expected_entities.add(entity_path)

            for row in point_rows:
                recording.reset_time()
                expected_timelines.update(_set_rerun_row_times(recording, row))
                entity_path = _rerun_entity_path(row.get("entity_path"))
                positions = np.asarray(row.get("positions_m"), dtype=np.float64).reshape(-1, 3)
                if not np.isfinite(positions).all():
                    raise ValueError(f"{entity_path}: points contain NaN/Inf")
                recording.log(
                    entity_path,
                    rr.Points3D(
                        positions.astype(np.float32),
                        radii=row.get("radii"),
                        colors=row.get("colors"),
                        labels=row.get("labels"),
                    ),
                    *(
                        [rr.CoordinateFrame(str(row["coordinate_frame"]))]
                        if row.get("coordinate_frame")
                        else []
                    ),
                    static=bool(row.get("static", False)),
                )
                expected_entities.add(entity_path)

            for row in arrow_rows:
                recording.reset_time()
                expected_timelines.update(_set_rerun_row_times(recording, row))
                entity_path = _rerun_entity_path(row.get("entity_path"))
                vectors = np.asarray(row.get("vectors_m"), dtype=np.float64).reshape(-1, 3)
                origins = np.asarray(row.get("origins_m"), dtype=np.float64).reshape(-1, 3)
                if not np.isfinite(vectors).all() or not np.isfinite(origins).all():
                    raise ValueError(f"{entity_path}: arrows contain NaN/Inf")
                recording.log(
                    entity_path,
                    rr.Arrows3D(
                        vectors=vectors.astype(np.float32),
                        origins=origins.astype(np.float32),
                        radii=row.get("radii"),
                        colors=row.get("colors"),
                        labels=row.get("labels"),
                    ),
                    *(
                        [rr.CoordinateFrame(str(row["coordinate_frame"]))]
                        if row.get("coordinate_frame")
                        else []
                    ),
                    static=bool(row.get("static", False)),
                )
                expected_entities.add(entity_path)

            for row in scalar_rows:
                expected_timelines.update(_set_rerun_row_times(recording, row))
                entity_path = _rerun_entity_path(row.get("entity_path"))
                value = float(row.get("value"))
                if not math.isfinite(value):
                    raise ValueError(f"{entity_path}: scalar is not finite")
                recording.log(entity_path, rr.Scalars([value]), static=bool(row.get("static", False)))
                expected_entities.add(entity_path)

            for row in event_rows:
                expected_timelines.update(_set_rerun_row_times(recording, row))
                entity_path = _rerun_entity_path(row.get("entity_path", "events/run"))
                recording.log(
                    entity_path,
                    rr.TextLog(
                        str(row.get("text", "")),
                        level=str(row.get("level", "INFO")),
                        color=row.get("color"),
                    ),
                    static=bool(row.get("static", False)),
                )
                expected_entities.add(entity_path)

            recording.flush(timeout_sec=30.0)
            flush_ok = True
        sink_finalized = True
        blueprint.save(app_id, rbl_path)
        blueprint_status = {
            "attempted": True,
            "ok": rbl_path.is_file() and rbl_path.stat().st_size > 0,
            "path": str(rbl_path),
            "mode": blueprint_mode,
        }
    except Exception as exc:
        return {
            "ok": False,
            "backend": "rerun",
            "path": str(out_path),
            "sink_attached_before_logging": sink_attached_before_logging,
            "sink_finalized": sink_finalized,
            "flush_ok": flush_ok,
            "error": repr(exc),
        }

    file_nonzero = out_path.is_file() and out_path.stat().st_size > 0
    archive_validation = validate_rerun_artifact(
        out_path,
        expected_entity_paths=sorted(expected_entities),
        expected_timeline_names=sorted(expected_timelines),
        blueprint_path=rbl_path,
        expected_version=RERUN_CONTRACT_VERSION,
    )
    urdf_ok = (
        (urdf_path is None or urdf_actual_status.get("ok", False))
        and (not trace_rows or urdf_path is None or urdf_commanded_status.get("ok", False))
    )
    ok = bool(
        sink_attached_before_logging
        and sink_finalized
        and flush_ok
        and file_nonzero
        and blueprint_status.get("ok", False)
        and archive_validation.get("pass", False)
        and urdf_ok
    )
    return {
        "ok": ok,
        "backend": "rerun",
        "path": str(out_path),
        "bytes": out_path.stat().st_size if file_nonzero else 0,
        "trace_steps": len(trace_rows),
        "mesh_count": len(mesh_rows),
        "coordinate_frame_count": len(coordinate_frame_rows),
        "point_entity_count": len(point_rows),
        "arrow_entity_count": len(arrow_rows),
        "scalar_row_count": len(scalar_rows),
        "event_row_count": len(event_rows),
        "urdf_actual_status": urdf_actual_status,
        "urdf_commanded_status": urdf_commanded_status,
        "blueprint_status": blueprint_status,
        "live_viewer": bool(live_viewer),
        "rerun_sdk_version": str(rr.__version__),
        "sink_attached_before_logging": sink_attached_before_logging,
        "sink_finalized": sink_finalized,
        "flush_ok": flush_ok,
        "file_nonzero": file_nonzero,
        "archive_validation": archive_validation,
        "requires_posthoc_visual_inspection": True,
        "visual_inspection_complete": False,
        "completion_contract_pass": False,
        "scientific_authority": "original callback arrays / canonical JSON / hashes",
    }
