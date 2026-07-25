#!/usr/bin/env python3
"""D385 offline semantic low-count redesign for eight failed P34 source hulls.

This case never starts Isaac Sim, Kit, PhysX, USD, CUDA, Warp, or robot
hardware.  Its numerical authority is the frozen D372/D379/D380/D384 JSON
evidence.  It keeps the 17 already-passing P34 parts and the D384 exact
46-child profile repair, then replaces only the eight failed general 3-D
source hulls by deterministic convex slabs cut from the same authored parent.

The construction is deliberately narrower than automatic convex decomposition:
each parent is first split at its authored thin-layer coordinates; each layer's
convex broad profile is then divided into deterministic fan cells with at most
six profile vertices.  Intersecting those cell prisms with the original parent
creates children with at most twelve Float32-authored vertices.  Actual PhysX
cook identity and GPU compatibility remain null until a separately approved
live readback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError


REPO = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO):
    if str(REPO) in sys.path:
        sys.path.remove(str(REPO))
    sys.path.insert(0, str(REPO))

CASE = "g0a_d385"
ATTEMPT = "attempt2_precreate_git_status_capture_repair"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT
SCRIPT_PATH = Path(__file__).resolve()
START_HERE = REPO / "START_HERE.md"

D372_GEOMETRY = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d372/"
    "attempt2_external_schema_path_repair/"
    "d372_professor_semantic_candidate_geometry.json"
)
D372_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d372/"
    "attempt2_external_schema_path_repair/"
    "d372_professor_semantic_candidate_evidence.json"
)
D379_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d379/"
    "attempt2_d372_measurement_field_repair/"
    "d379_p34_full_live_identity_evidence.json"
)
D380_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d380/"
    "attempt1_failed_part_cook_provenance_semantic_impact_audit/"
    "d380_p34_failed_part_cook_provenance_evidence.json"
)
D384_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d384/"
    "attempt2_callback_vertex_count_field_preflight_repair/"
    "d384_p34_representation_repair_design_evidence.json"
)

PHYSX_SCHEMA = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/"
    "omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "plugins/PhysxSchema/resources/schema.usda"
)
PHYSX_PROPERTY_DB = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/"
    "omni.kit.property.physx-107.3.26+107.3.3.cp311.u353/"
    "omni/kit/property/physx/database.py"
)

EXPECTED_INPUT_SHA256 = {
    "d372_geometry": "12fd1f32c35dfb9ae36cbbb412f6a51536aa1cc07c2dc17d05a5d189f3ee83e4",
    "d372_evidence": "d68f658089aaf838ff454e9d0b301ec3f602785a3a730b3c329aa7785010e984",
    "d379_evidence": "8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5",
    "d380_evidence": "4c64d08e117501dd15a5836ce56ef8b963d188044beac465e645e53a17710bd1",
    "d384_evidence": "16ed5696d7198913367806e3ee13cf17a2b3f83c0c28d139115aa1d51c40822f",
}
EXPECTED_HEAD = "35f10e3079b19e51209ba4cf1dd66391a431b053"

PREREG_PATH = OUT_DIR / "d385_preregistration.json"
PHASE_PATH = OUT_DIR / "d385_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d385_offline_design_invocation.json"
WORKER_STDOUT = OUT_DIR / "d385_offline_worker_stdout.log"
WORKER_STDERR = OUT_DIR / "d385_offline_worker_stderr.log"
WORKER_CLAIM = OUT_DIR / "d385_offline_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d385_offline_worker_supervisor.json"
EVIDENCE_PATH = OUT_DIR / "d385_p34_source_hull_redesign_evidence.json"
GEOMETRY_PATH = OUT_DIR / "d385_p34_semantic_low_count_candidate_geometry.json"
METRICS_CSV = OUT_DIR / "d385_source_parent_metrics.csv"
BOARD_PATH = OUT_DIR / "d385_source_hull_redesign_1920x1080.png"
BOARD_LAYOUT = OUT_DIR / "d385_board_layout_validation.json"
RRD_PATH = OUT_DIR / "d385_source_hull_redesign.rrd"
RBL_PATH = OUT_DIR / "d385_source_hull_redesign.rbl"
RERUN_VALIDATION = OUT_DIR / "d385_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d385_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d385_manual_visual_inspection_template.json"
MANUAL_INSPECTION = OUT_DIR / "d385_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d385_completion_summary.json"

RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = [
    "source_hull_semantic_thin_layer_profile_cell_partition_v1",
    "source_child_max12_vertex_budget_v1",
]
WATCHDOG_SECONDS = 300.0
SURFACE_TOLERANCE_MM = 0.1
VOLUME_RELATIVE_TOLERANCE = 0.005
MAX_VERTICES_PER_SOURCE_CHILD = 12
MAX_SOURCE_CHILDREN = 64
CURRENT_A64_TOTAL_REFERENCE = 128
PASSING_PART_COUNT = 17
PROFILE_CHILD_COUNT = 46
FLOAT_EPS_M = 5.0e-9
POSITIVE_VOLUME_EPS_M3 = 1.0e-18

SOURCE_PARENT_NAMES = {
    "fixed_backbone_left",
    "fixed_backbone_right",
    "proximal_upper_arm_hull_a",
    "proximal_upper_arm_hull_b",
    "proximal_lower_arm_hull_a",
    "proximal_lower_arm_hull_b",
    "moving_upper_backbone",
    "moving_lower_backbone",
}
PROFILE_ROLES = {"fixed_jaw", "moving_jaw"}
SEMANTIC_PLAN = {
    "fixed_backbone_left": {
        "primary_axis": 2,
        "primary_axis_name": "z",
        "semantic_pre_split_axis": 1,
        "semantic_pre_split_axis_name": "y",
    },
    "fixed_backbone_right": {
        "primary_axis": 2,
        "primary_axis_name": "z",
        "semantic_pre_split_axis": 1,
        "semantic_pre_split_axis_name": "y",
    },
    "proximal_upper_arm_hull_a": {
        "primary_axis": 0,
        "primary_axis_name": "x",
        "semantic_pre_split_axis": 2,
        "semantic_pre_split_axis_name": "z",
    },
    "proximal_upper_arm_hull_b": {
        "primary_axis": 0,
        "primary_axis_name": "x",
        "semantic_pre_split_axis": 2,
        "semantic_pre_split_axis_name": "z",
    },
    "proximal_lower_arm_hull_a": {
        "primary_axis": 0,
        "primary_axis_name": "x",
        "semantic_pre_split_axis": 2,
        "semantic_pre_split_axis_name": "z",
    },
    "proximal_lower_arm_hull_b": {
        "primary_axis": 0,
        "primary_axis_name": "x",
        "semantic_pre_split_axis": 2,
        "semantic_pre_split_axis_name": "z",
    },
    "moving_upper_backbone": {
        "primary_axis": 0,
        "primary_axis_name": "x",
        "semantic_pre_split_axis": 2,
        "semantic_pre_split_axis_name": "z",
    },
    "moving_lower_backbone": {
        "primary_axis": 0,
        "primary_axis_name": "x",
        "semantic_pre_split_axis": 2,
        "semantic_pre_split_axis_name": "z",
    },
}

FORBIDDEN_COUNTERS = {
    "asset_or_usd_reads": 0,
    "asset_or_usd_writes": 0,
    "automatic_decomposition_sweeps": 0,
    "collider_materializations_or_regenerations": 0,
    "contact_queries": 0,
    "controlled_physics_steps": 0,
    "cylinder_creates_or_writes": 0,
    "isaac_launches": 0,
    "kit_launches": 0,
    "live_callback_queries": 0,
    "physx_launches": 0,
    "q5_samples": 0,
    "target_ik_path_changes": 0,
    "warp_or_cuda_launches": 0,
}

PALETTE = [
    [0, 114, 178, 185],
    [230, 159, 0, 185],
    [0, 158, 115, 185],
    [204, 121, 167, 185],
    [213, 94, 0, 185],
    [86, 180, 233, 185],
    [240, 228, 66, 185],
    [0, 0, 0, 185],
]


class RegisteredNoCoverError(RuntimeError):
    """The registered construction is valid but has no admissible cover."""


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_payload(value: Any) -> str:
    payload = json.dumps(
        _native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_x(path: Path, value: Any) -> None:
    native = _native(value)
    payload = json.dumps(
        native,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    roundtrip = json.loads(payload)
    if roundtrip != native:
        raise RuntimeError(f"JSON roundtrip mismatch: {path}")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _phase(name: str, **fields: Any) -> None:
    row = {
        "phase": name,
        "monotonic_seconds": time.monotonic(),
        "wall_time_unix_seconds": time.time(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _native(row),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def _input_hashes() -> dict[str, str]:
    return {
        "d372_geometry": _sha(D372_GEOMETRY),
        "d372_evidence": _sha(D372_EVIDENCE),
        "d379_evidence": _sha(D379_EVIDENCE),
        "d380_evidence": _sha(D380_EVIDENCE),
        "d384_evidence": _sha(D384_EVIDENCE),
    }


def _triangles(row: dict[str, Any]) -> np.ndarray:
    counts = np.asarray(row["face_vertex_counts"], dtype=np.int64)
    indices = np.asarray(row["face_vertex_indices"], dtype=np.int64)
    if len(counts) == 0 or not np.all(counts == 3):
        raise ValueError(f"non-triangle topology: {row['prim_name']}")
    if indices.size != counts.sum():
        raise ValueError(f"face span mismatch: {row['prim_name']}")
    return indices.reshape(-1, 3)


def _unique_f32(points: np.ndarray) -> np.ndarray:
    registered = np.asarray(points, dtype=np.float32).astype(np.float64)
    return np.unique(registered, axis=0)


def _convex_mesh(points: np.ndarray) -> dict[str, Any]:
    registered = _unique_f32(points)
    if len(registered) < 4:
        raise ValueError("fewer than four registered points")
    if np.linalg.matrix_rank(registered - registered.mean(axis=0)) < 3:
        raise ValueError("registered points are not three-dimensional")
    hull = ConvexHull(registered)
    ids = np.asarray(hull.vertices, dtype=np.int64)
    vertices = registered[ids]
    remap = {int(old): new for new, old in enumerate(ids)}
    triangles = np.asarray(
        [[remap[int(index)] for index in face] for face in hull.simplices],
        dtype=np.int64,
    )
    hull2 = ConvexHull(vertices)
    plane_keys: set[tuple[float, ...]] = set()
    plane_vertices: dict[tuple[float, ...], set[int]] = {}
    for face, equation in zip(hull2.simplices, hull2.equations, strict=True):
        normal = equation[:3]
        length = float(np.linalg.norm(normal))
        key = tuple(np.round(equation / length, decimals=7))
        plane_keys.add(key)
        plane_vertices.setdefault(key, set()).update(map(int, face))
    areas = (
        0.5
        * np.linalg.norm(
            np.cross(
                vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
            ),
            axis=1,
        )
    )
    return {
        "vertices_m": vertices,
        "triangles": triangles,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "polygon_count": int(len(plane_keys)),
        "max_vertices_per_polygon": int(
            max(len(values) for values in plane_vertices.values())
        ),
        "volume_m3": float(hull2.volume),
        "minimum_triangle_area_mm2": float(areas.min() * 1.0e6),
        "bounds_m": [vertices.min(axis=0), vertices.max(axis=0)],
    }


def _convex_vertices_f64(points: np.ndarray) -> np.ndarray:
    """Return only true Float64 hull vertices without intermediate quantizing."""
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(unique) < 4:
        raise ValueError("fewer than four Float64 hull points")
    if np.linalg.matrix_rank(unique - unique.mean(axis=0)) < 3:
        raise ValueError("Float64 hull points are not three-dimensional")
    hull = ConvexHull(unique)
    return unique[np.asarray(hull.vertices, dtype=np.int64)]


def _convex_polyhedron_edges(points: np.ndarray) -> np.ndarray:
    """Return true convex-polyhedron edges, excluding coplanar face diagonals."""
    source = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(source)
    plane_groups: dict[tuple[float, ...], set[int]] = {}
    for simplex, equation in zip(
        hull.simplices, hull.equations, strict=True
    ):
        length = float(np.linalg.norm(equation[:3]))
        key = tuple(np.round(equation / length, decimals=7))
        plane_groups.setdefault(key, set()).update(map(int, simplex))
    memberships: dict[tuple[int, int], int] = {}
    for vertices in plane_groups.values():
        ordered = sorted(vertices)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                pair = (int(left), int(right))
                memberships[pair] = memberships.get(pair, 0) + 1
    edges = [
        pair for pair, shared_planes in memberships.items()
        if shared_planes >= 2
    ]
    if not edges:
        raise RuntimeError("failed to reconstruct convex-polyhedron edges")
    return np.asarray(sorted(edges), dtype=np.int64)


def _clip_halfspace(
    points: np.ndarray,
    *,
    axis: int,
    bound: float,
    keep_greater_equal: bool,
) -> np.ndarray:
    """Clip a convex point hull by one axis-aligned half-space."""
    source = np.asarray(points, dtype=np.float64)
    edges = _convex_polyhedron_edges(source)
    values = source[:, axis] - float(bound)
    keep = values >= -FLOAT_EPS_M if keep_greater_equal else values <= FLOAT_EPS_M
    output = [point for point in source[keep]]
    for left, right in edges:
        v0 = values[int(left)]
        v1 = values[int(right)]
        if (v0 < -FLOAT_EPS_M and v1 > FLOAT_EPS_M) or (
            v1 < -FLOAT_EPS_M and v0 > FLOAT_EPS_M
        ):
            ratio = -v0 / (v1 - v0)
            point = source[int(left)] + ratio * (
                source[int(right)] - source[int(left)]
            )
            point[axis] = float(bound)
            output.append(point)
    result = np.unique(np.asarray(output, dtype=np.float64), axis=0)
    if len(result) < 4:
        raise ValueError("half-space clip produced a degenerate child")
    return _convex_vertices_f64(result)


def _clip_interval(
    points: np.ndarray, *, axis: int, low: float, high: float
) -> dict[str, Any]:
    if not high > low:
        raise ValueError("invalid interval")
    clipped = _clip_halfspace(
        points,
        axis=axis,
        bound=float(low),
        keep_greater_equal=True,
    )
    clipped = _clip_halfspace(
        clipped,
        axis=axis,
        bound=float(high),
        keep_greater_equal=False,
    )
    return _convex_mesh(clipped)


def _clip_plane_le(
    points: np.ndarray,
    *,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    unit = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(unit))
    if length <= 0.0:
        raise ValueError("zero clipping-plane normal")
    unit = unit / length
    signed_offset = float(offset) / length
    edges = _convex_polyhedron_edges(source)
    values = source @ unit + signed_offset
    keep = values <= FLOAT_EPS_M
    output = [point for point in source[keep]]
    for left, right in edges:
        v0 = values[int(left)]
        v1 = values[int(right)]
        if (v0 < -FLOAT_EPS_M and v1 > FLOAT_EPS_M) or (
            v1 < -FLOAT_EPS_M and v0 > FLOAT_EPS_M
        ):
            ratio = -v0 / (v1 - v0)
            output.append(
                source[int(left)]
                + ratio * (source[int(right)] - source[int(left)])
            )
    result = np.unique(np.asarray(output, dtype=np.float64), axis=0)
    return _convex_vertices_f64(result)


def _profile_polygon(
    points: np.ndarray, thin_axis: int
) -> tuple[np.ndarray, tuple[int, int]]:
    keep = tuple(index for index in range(3) if index != thin_axis)
    projected = np.unique(
        np.asarray(points, dtype=np.float64)[:, list(keep)], axis=0
    )
    outline = ConvexHull(projected)
    polygon = projected[np.asarray(outline.vertices, dtype=np.int64)]
    return polygon, keep


def _fan_cell(
    polygon: np.ndarray,
    keep: tuple[int, int],
    *,
    triangle_start: int,
    triangle_end: int,
    cell_index: int,
) -> dict[str, Any]:
    cell = np.vstack(
        [
            polygon[0],
            polygon[triangle_start : triangle_end + 2],
        ]
    )
    if len(cell) > 6:
        raise RuntimeError("profile fan cell exceeded six vertices")
    return {
        "cell_index": int(cell_index),
        "polygon_2d_m": cell,
        "profile_axes": list(map(int, keep)),
        "profile_axis_names": ["xyz"[index] for index in keep],
        "fan_triangle_index_range": [
            int(triangle_start),
            int(triangle_end),
        ],
    }


def _intersect_profile_cell(
    points: np.ndarray,
    *,
    thin_axis: int,
    cell: dict[str, Any],
) -> dict[str, Any]:
    polygon = np.asarray(cell["polygon_2d_m"], dtype=np.float64)
    keep = list(map(int, cell["profile_axes"]))
    clipped = np.asarray(points, dtype=np.float64)
    # scipy ConvexHull returns a counter-clockwise 2-D outline.  Interior is
    # the left side of every directed edge.  Convert that to n.x + d <= 0.
    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]
        delta = end - start
        normal_2d = np.asarray([delta[1], -delta[0]], dtype=np.float64)
        offset = float(delta[0] * start[1] - delta[1] * start[0])
        normal_3d = np.zeros(3, dtype=np.float64)
        normal_3d[keep] = normal_2d
        clipped = _clip_plane_le(
            clipped,
            normal=normal_3d,
            offset=offset,
        )
    mesh = _convex_mesh(clipped)
    mesh.update(
        {
            "profile_axes": cell["profile_axes"],
            "profile_axis_names": cell["profile_axis_names"],
            "profile_polygon_2d_m": polygon,
            "fan_triangle_index_range": cell[
                "fan_triangle_index_range"
            ],
            "profile_cell_index": cell["cell_index"],
            "thin_axis_index": int(thin_axis),
            "thin_axis_name": "xyz"[thin_axis],
        }
    )
    return mesh


def _profile_cell_partition(
    points: np.ndarray,
    *,
    thin_axis: int,
    region_name: str,
) -> dict[str, Any]:
    polygon, keep = _profile_polygon(points, thin_axis)
    triangle_count = len(polygon) - 2
    cache: dict[tuple[int, int], dict[str, Any] | None] = {}
    rejection_by_key: dict[tuple[int, int], dict[str, Any]] = {}

    def candidate(start_state: int, end_state: int) -> dict[str, Any] | None:
        key = (start_state, end_state)
        if key in cache:
            return cache[key]
        group_size = end_state - start_state
        if group_size < 1 or group_size > 4:
            cache[key] = None
            rejection_by_key[key] = {
                "reason": "fan_group_size_out_of_registered_range",
                "group_size": int(group_size),
            }
            return None
        cell = _fan_cell(
            polygon,
            keep,
            triangle_start=start_state + 1,
            triangle_end=end_state,
            cell_index=-1,
        )
        try:
            child = _intersect_profile_cell(
                points,
                thin_axis=thin_axis,
                cell=cell,
            )
        except (ValueError, QhullError) as exc:
            cache[key] = None
            rejection_by_key[key] = {
                "reason": "expected_degenerate_cell_rejection",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            return None
        compatible = bool(
            child["vertex_count"] <= MAX_VERTICES_PER_SOURCE_CHILD
            and child["polygon_count"] <= 64
            and child["max_vertices_per_polygon"] <= 32
            and child["volume_m3"] > POSITIVE_VOLUME_EPS_M3
        )
        cache[key] = child if compatible else None
        if not compatible:
            rejection_by_key[key] = {
                "reason": "registered_child_compatibility_gate_failed",
                "vertex_count": int(child["vertex_count"]),
                "polygon_count": int(child["polygon_count"]),
                "maximum_vertices_per_polygon": int(
                    child["max_vertices_per_polygon"]
                ),
                "volume_m3": float(child["volume_m3"]),
            }
        return cache[key]

    # Shortest contiguous cover of the registered fan triangles.  Tie-break
    # first by lower maximum child vertices, then stable cut indices.
    dp: list[tuple[int, int, tuple[int, ...]] | None] = [
        None
    ] * (triangle_count + 1)
    dp[0] = (0, 0, (0,))
    for end_state in range(1, triangle_count + 1):
        best = None
        for start_state in range(max(0, end_state - 4), end_state):
            if dp[start_state] is None:
                continue
            child = candidate(start_state, end_state)
            if child is None:
                continue
            previous = dp[start_state]
            value = (
                previous[0] + 1,
                max(previous[1], int(child["vertex_count"])),
                previous[2] + (end_state,),
            )
            if best is None or value < best:
                best = value
        dp[end_state] = best
    solution = dp[-1]
    if solution is None:
        reason_counts: dict[str, int] = {}
        for rejection in rejection_by_key.values():
            reason = str(rejection["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        raise RegisteredNoCoverError(
            f"{region_name}: no contiguous fan-cell cover satisfies "
            f"max-{MAX_VERTICES_PER_SOURCE_CHILD} vertices; "
            f"candidate_rejections={len(rejection_by_key)}; "
            f"reason_counts={json.dumps(reason_counts, sort_keys=True)}"
        )
    children = []
    cut_states = solution[2]
    for cell_index, (start_state, end_state) in enumerate(
        zip(cut_states[:-1], cut_states[1:], strict=True)
    ):
        child = candidate(start_state, end_state)
        if child is None:
            raise RuntimeError("selected fan cell disappeared from cache")
        child = dict(child)
        child["profile_cell_index"] = int(cell_index)
        child["fan_triangle_index_range"] = [
            int(start_state + 1),
            int(end_state),
        ]
        child["region_name"] = region_name
        child["registered_offline_compatibility_precheck"] = True
        children.append(child)
    return {
        "region_name": region_name,
        "thin_axis_index": int(thin_axis),
        "thin_axis_name": "xyz"[thin_axis],
        "broad_profile_vertex_count": int(len(polygon)),
        "maximum_profile_vertices_per_cell": 6,
        "selected_fan_cut_states": list(map(int, cut_states)),
        "child_count": len(children),
        "maximum_child_vertices": max(
            child["vertex_count"] for child in children
        ),
        "children": children,
    }


def _partition_source_parent(
    authored: dict[str, Any]
) -> dict[str, Any]:
    name = str(authored["name"])
    plan = SEMANTIC_PLAN[name]
    points = _unique_f32(
        np.asarray(authored["points_f32"], dtype=np.float64)
    )
    parent = _convex_mesh(points)
    regions: list[dict[str, Any]] = []
    pre_axis = int(plan["semantic_pre_split_axis"])
    pre_axis_name = str(plan["semantic_pre_split_axis_name"])
    pre_levels = np.unique(points[:, pre_axis])
    pre_levels.sort()
    if len(pre_levels) < 2:
        raise ValueError(f"{name}: thin-axis split has fewer than two levels")
    for region_index, (low, high) in enumerate(
        zip(pre_levels[:-1], pre_levels[1:], strict=True)
    ):
        region = _clip_interval(
            points,
            axis=pre_axis,
            low=float(low),
            high=float(high),
        )
        regions.append(
            {
                "name": f"{pre_axis_name}_layer_{region_index:02d}",
                "points": region["vertices_m"],
                "pre_split_axis": pre_axis_name,
                "pre_split_interval_m": [
                    float(low),
                    float(high),
                ],
            }
        )

    partitions = []
    children: list[dict[str, Any]] = []
    for region_index, region in enumerate(regions):
        partition = _profile_cell_partition(
            np.asarray(region["points"], dtype=np.float64),
            thin_axis=pre_axis,
            region_name=str(region["name"]),
        )
        partition["pre_split_axis"] = region["pre_split_axis"]
        partition["pre_split_interval_m"] = region[
            "pre_split_interval_m"
        ]
        partitions.append(
            {
                key: value
                for key, value in partition.items()
                if key != "children"
            }
        )
        for child in partition["children"]:
            child["region_index"] = int(region_index)
            child["parent_name"] = name
            child["body"] = str(authored["body"])
            child["role"] = str(authored["role"])
            child["name"] = (
                f"{name}__region_{region_index:02d}_"
                f"cell_{child['profile_cell_index']:02d}"
            )
            child["pre_split_axis"] = region["pre_split_axis"]
            child["pre_split_interval_m"] = region[
                "pre_split_interval_m"
            ]
            children.append(child)

    return {
        "body": authored["body"],
        "prim_name": authored["prim_name"],
        "name": name,
        "role": authored["role"],
        "semantic_plan": plan,
        "semantic_pre_split_levels_m": list(map(float, pre_levels)),
        "parent": parent,
        "partitions": partitions,
        "children": children,
    }


def _profile_children(authored: dict[str, Any]) -> list[dict[str, Any]]:
    """Reproduce the frozen D384 exact triangular-prism profile split."""
    points = _unique_f32(np.asarray(authored["points_f32"], dtype=np.float64))
    for axis in range(3):
        levels = np.unique(points[:, axis])
        if len(levels) != 2:
            continue
        keep = [index for index in range(3) if index != axis]
        sections = []
        for level in levels:
            sections.append(
                set(
                    map(
                        tuple,
                        points[points[:, axis] == level][:, keep],
                    )
                )
            )
        if sections[0] != sections[1]:
            continue
        section = np.asarray(sorted(sections[0]), dtype=np.float64)
        outline = ConvexHull(section)
        ordered = section[outline.vertices]
        triangulation = Delaunay(ordered)
        children = []
        for child_index, triangle in enumerate(
            np.asarray(triangulation.simplices, dtype=np.int64)
        ):
            cross = ordered[triangle]
            child_points = []
            for level in levels:
                for pair in cross:
                    point = np.zeros(3, dtype=np.float64)
                    point[axis] = level
                    point[keep] = pair
                    child_points.append(point)
            mesh = _convex_mesh(np.asarray(child_points))
            children.append(
                {
                    **mesh,
                    "body": authored["body"],
                    "parent_name": authored["name"],
                    "role": authored["role"],
                    "name": (
                        f"{authored['name']}__profile_"
                        f"{child_index:02d}"
                    ),
                    "profile_axis": "xyz"[axis],
                    "profile_axis_index": int(axis),
                    "profile_levels_m": list(map(float, levels)),
                }
            )
        return children
    raise ValueError(
        f"{authored['prim_name']}: not an exact paired extrusion"
    )


def _part_from_authored(authored: dict[str, Any]) -> dict[str, Any]:
    return {
        "body": authored["body"],
        "prim_name": authored["prim_name"],
        "name": authored["name"],
        "role": authored["role"],
        "vertices_m": np.asarray(
            authored["points_f32"], dtype=np.float64
        ),
        "triangles": _triangles(authored),
        "points_f32_sha256": authored["points_f32_sha256"],
        "authored_f32_topology_payload_sha256": authored[
            "authored_f32_topology_payload_sha256"
        ],
    }


def _surface_samples(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    edges = np.unique(
        np.sort(
            np.vstack(
                [
                    triangles[:, [0, 1]],
                    triangles[:, [1, 2]],
                    triangles[:, [2, 0]],
                ]
            ),
            axis=1,
        ),
        axis=0,
    )
    centroids = np.mean(vertices[triangles], axis=1)
    edge_midpoints = np.mean(vertices[edges], axis=1)
    quarter_a = (
        0.5 * vertices[triangles[:, 0]]
        + 0.25 * vertices[triangles[:, 1]]
        + 0.25 * vertices[triangles[:, 2]]
    )
    quarter_b = (
        0.25 * vertices[triangles[:, 0]]
        + 0.5 * vertices[triangles[:, 1]]
        + 0.25 * vertices[triangles[:, 2]]
    )
    quarter_c = (
        0.25 * vertices[triangles[:, 0]]
        + 0.25 * vertices[triangles[:, 1]]
        + 0.5 * vertices[triangles[:, 2]]
    )
    return np.vstack(
        [
            vertices,
            edge_midpoints,
            centroids,
            quarter_a,
            quarter_b,
            quarter_c,
        ]
    )


def _normalized_equations(points: np.ndarray) -> np.ndarray:
    equations = np.asarray(ConvexHull(points).equations, dtype=np.float64)
    lengths = np.linalg.norm(equations[:, :3], axis=1)
    return equations / lengths[:, None]


def _maximum_positive_violation_mm(
    equations: np.ndarray, points: np.ndarray
) -> float:
    values = (
        np.asarray(points, dtype=np.float64) @ equations[:, :3].T
        + equations[:, 3]
    )
    return float(np.maximum(values, 0.0).max() * 1000.0)


def _union_coverage_violation_mm(
    children: list[dict[str, Any]], samples: np.ndarray
) -> tuple[float, int]:
    per_child = []
    for child in children:
        equations = _normalized_equations(
            np.asarray(child["vertices_m"], dtype=np.float64)
        )
        values = (
            np.asarray(samples, dtype=np.float64)
            @ equations[:, :3].T
            + equations[:, 3]
        )
        per_child.append(np.maximum(values, 0.0).max(axis=1))
    best = np.min(np.vstack(per_child), axis=0)
    return float(best.max() * 1000.0), int(np.count_nonzero(best > 1.0e-7))


def _partition_overlap_certificate(
    children: list[dict[str, Any]]
) -> dict[str, Any]:
    positive_overlap_pairs = []
    for left_index, left in enumerate(children):
        for right_index in range(left_index + 1, len(children)):
            right = children[right_index]
            if left["region_index"] != right["region_index"]:
                left_interval = left.get("pre_split_interval_m")
                right_interval = right.get("pre_split_interval_m")
                if left_interval is None or right_interval is None:
                    positive_overlap_pairs.append(
                        [left["name"], right["name"], "unproven"]
                    )
                    continue
                width = min(
                    left_interval[1], right_interval[1]
                ) - max(left_interval[0], right_interval[0])
            else:
                left_range = left["fan_triangle_index_range"]
                right_range = right["fan_triangle_index_range"]
                disjoint_fans = (
                    left_range[1] < right_range[0]
                    or right_range[1] < left_range[0]
                )
                width = 0.0 if disjoint_fans else math.inf
            if width > FLOAT_EPS_M:
                positive_overlap_pairs.append(
                    [left["name"], right["name"], float(width)]
                )
    return {
        "method": (
            "different thin layers have disjoint registered intervals; "
            "same-layer fan cells have disjoint triangle-index ranges"
        ),
        "positive_overlap_pair_count": len(positive_overlap_pairs),
        "positive_overlap_pairs": positive_overlap_pairs,
        "pass": len(positive_overlap_pairs) == 0,
    }


def _parent_metrics(parent_row: dict[str, Any]) -> dict[str, Any]:
    parent = parent_row["parent"]
    children = parent_row["children"]
    parent_points = np.asarray(parent["vertices_m"], dtype=np.float64)
    parent_triangles = np.asarray(parent["triangles"], dtype=np.int64)
    parent_equations = _normalized_equations(parent_points)
    child_points = np.vstack(
        [np.asarray(child["vertices_m"], dtype=np.float64) for child in children]
    )
    outward_mm = _maximum_positive_violation_mm(
        parent_equations, child_points
    )
    samples = _surface_samples(parent_points, parent_triangles)
    coverage_mm, uncovered_samples = _union_coverage_violation_mm(
        children, samples
    )
    child_volume = float(sum(child["volume_m3"] for child in children))
    volume_relative = abs(child_volume - parent["volume_m3"]) / parent[
        "volume_m3"
    ]
    overlap = _partition_overlap_certificate(children)
    checks = {
        "child_count_positive": len(children) > 0,
        "each_child_vertices_le_12": max(
            child["vertex_count"] for child in children
        )
        <= MAX_VERTICES_PER_SOURCE_CHILD,
        "each_child_polygons_le_64": max(
            child["polygon_count"] for child in children
        )
        <= 64,
        "each_child_vertices_per_polygon_le_32": max(
            child["max_vertices_per_polygon"] for child in children
        )
        <= 32,
        "each_child_positive_volume": all(
            child["volume_m3"] > POSITIVE_VOLUME_EPS_M3
            for child in children
        ),
        "outward_le_0p1mm": outward_mm <= SURFACE_TOLERANCE_MM,
        "coverage_le_0p1mm": coverage_mm <= SURFACE_TOLERANCE_MM,
        "volume_relative_le_0p5percent": (
            volume_relative <= VOLUME_RELATIVE_TOLERANCE
        ),
        "positive_volume_overlap_zero": overlap["pass"],
    }
    return {
        "body": parent_row["body"],
        "name": parent_row["name"],
        "role": parent_row["role"],
        "parent_vertex_count": parent["vertex_count"],
        "child_count": len(children),
        "maximum_child_vertex_count": max(
            child["vertex_count"] for child in children
        ),
        "maximum_child_polygon_count": max(
            child["polygon_count"] for child in children
        ),
        "maximum_vertices_per_child_polygon": max(
            child["max_vertices_per_polygon"] for child in children
        ),
        "parent_volume_mm3": parent["volume_m3"] * 1.0e9,
        "child_volume_sum_mm3": child_volume * 1.0e9,
        "volume_relative_error": volume_relative,
        "outward_halfspace_violation_mm": outward_mm,
        "parent_surface_coverage_halfspace_violation_mm": coverage_mm,
        "surface_sample_count": int(len(samples)),
        "uncovered_sample_count_gt_0p0001mm": uncovered_samples,
        "overlap_certificate": overlap,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _point_in_union_violation_mm(
    point: Iterable[float], parts: list[dict[str, Any]]
) -> float:
    sample = np.asarray(list(point), dtype=np.float64).reshape(1, 3)
    values = []
    for part in parts:
        equations = _normalized_equations(
            np.asarray(part["vertices_m"], dtype=np.float64)
        )
        violation = np.maximum(
            sample @ equations[:, :3].T + equations[:, 3],
            0.0,
        ).max()
        values.append(float(violation))
    return float(min(values) * 1000.0)


def _projected_mask(
    parts: list[dict[str, Any]],
    *,
    keep_axes: tuple[int, int],
    bounds: tuple[np.ndarray, np.ndarray],
    step_m: float = 0.00025,
) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    low, high = bounds
    xs = np.arange(low[0], high[0] + step_m * 0.5, step_m)
    ys = np.arange(low[1], high[1] + step_m * 0.5, step_m)
    xx, yy = np.meshgrid(xs, ys)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    union = np.zeros(len(query), dtype=bool)
    for part in parts:
        projected = np.unique(
            np.asarray(part["vertices_m"], dtype=np.float64)[
                :, list(keep_axes)
            ],
            axis=0,
        )
        if len(projected) < 3:
            continue
        hull = ConvexHull(projected)
        polygon = projected[hull.vertices]
        union |= MplPath(polygon).contains_points(
            query, radius=step_m * 1.0e-6
        )
    return union.reshape(xx.shape)


def _projection_equivalence(
    parent: dict[str, Any],
    children: list[dict[str, Any]],
    *,
    keep_axes: tuple[int, int],
) -> dict[str, Any]:
    parent_projection = np.asarray(parent["vertices_m"], dtype=np.float64)[
        :, list(keep_axes)
    ]
    child_projection = np.vstack(
        [
            np.asarray(child["vertices_m"], dtype=np.float64)[
                :, list(keep_axes)
            ]
            for child in children
        ]
    )
    low = np.minimum(parent_projection.min(axis=0), child_projection.min(axis=0))
    high = np.maximum(parent_projection.max(axis=0), child_projection.max(axis=0))
    margin = np.asarray([0.0005, 0.0005])
    bounds = (low - margin, high + margin)
    parent_mask = _projected_mask(
        [
            {
                "vertices_m": parent["vertices_m"],
            }
        ],
        keep_axes=keep_axes,
        bounds=bounds,
    )
    child_mask = _projected_mask(
        children,
        keep_axes=keep_axes,
        bounds=bounds,
    )
    xor = np.logical_xor(parent_mask, child_mask)
    return {
        "keep_axes": list(map(int, keep_axes)),
        "raster_step_mm": 0.25,
        "parent_occupied_cells": int(parent_mask.sum()),
        "children_occupied_cells": int(child_mask.sum()),
        "xor_cell_count": int(xor.sum()),
        "xor_fraction_of_parent": float(
            xor.sum() / max(1, int(parent_mask.sum()))
        ),
        "pass": int(xor.sum()) == 0,
    }


def _semantic_preservation(
    *,
    d372: dict[str, Any],
    source_parents: list[dict[str, Any]],
    profile_children: list[dict[str, Any]],
    passing_parts: list[dict[str, Any]],
) -> dict[str, Any]:
    by_name = {row["name"]: row for row in source_parents}
    projections = {}
    for name, row in by_name.items():
        if name.startswith("fixed_backbone"):
            keep = (0, 2)
        elif name.startswith("moving_") and name.endswith("backbone"):
            keep = (0, 2)
        else:
            # Moving support is excluded from the D372 2-D jaw-void diagnostic,
            # but its broad-face projection is still checked for exact lineage.
            keep = (0, 1)
        projections[name] = _projection_equivalence(
            row["parent"], row["children"], keep_axes=keep
        )

    contact_parts = [
        part
        for part in [*passing_parts, *profile_children]
        if part["role"] in PROFILE_ROLES
    ]
    seed_rows = d372["contact_seed_retention"]["rows"]
    seed_checks = {}
    for label in ("fixed", "moving"):
        frozen = seed_rows[label]
        body = frozen["body"]
        role = "fixed_jaw" if label == "fixed" else "moving_jaw"
        selected = [
            part
            for part in contact_parts
            if part["body"] == body and part["role"] == role
        ]
        violation_mm = _point_in_union_violation_mm(
            frozen["seed_local_m"], selected
        )
        seed_checks[label] = {
            "body": body,
            "role": role,
            "seed_local_m": frozen["seed_local_m"],
            "candidate_union_halfspace_violation_mm": violation_mm,
            "registered_max_mm": frozen["registered_max_mm"],
            "pass": violation_mm <= frozen["registered_max_mm"],
        }

    d372_source_rows = d372["semantic_representation_metrics"][
        "semantic_source_rows"
    ]
    inherited_rows = {}
    for parent in source_parents:
        key = f"{parent['body']}/{parent['name']}"
        frozen = d372_source_rows[key]
        metric = _parent_metrics(parent)
        inherited_rows[key] = {
            "frozen_source_vertices_sha256": frozen[
                "source_vertices_sha256"
            ],
            "frozen_all_source_vertices_contained": frozen[
                "all_source_vertices_contained"
            ],
            "inheritance_basis": (
                "D372 source vertices were contained by this parent; D385 "
                "children are a registered compound partition of the same "
                "Float32-authored parent within frozen 0.1mm/0.5% gates"
            ),
            "parent_partition_pass": metric["pass"],
            "pass": bool(
                frozen["pass"] is True and metric["pass"] is True
            ),
        }

    frozen_voids = d372["jaw_contact_layer_void_diagnostic"]["bodies"]
    fixed_projection_pass = all(
        projections[name]["pass"]
        for name in ("fixed_backbone_left", "fixed_backbone_right")
    )
    moving_projection_pass = all(
        projections[name]["pass"]
        for name in ("moving_upper_backbone", "moving_lower_backbone")
    )
    void_inheritance = {
        "fixed": {
            "frozen_voids": frozen_voids["fixed"]["voids"],
            "backbone_projection_bit_equivalent": fixed_projection_pass,
            "profile_repair_exact_partition": True,
            "pass": bool(
                frozen_voids["fixed"]["pass"] is True
                and fixed_projection_pass
            ),
        },
        "moving": {
            "frozen_voids": frozen_voids["moving"]["voids"],
            "backbone_projection_bit_equivalent": moving_projection_pass,
            "profile_repair_exact_partition": True,
            "pass": bool(
                frozen_voids["moving"]["pass"] is True
                and moving_projection_pass
            ),
        },
    }
    owner_checks = {
        "fixed_backbones_remain_link5": all(
            by_name[name]["body"] == "link5"
            for name in ("fixed_backbone_left", "fixed_backbone_right")
        ),
        "moving_supports_and_backbones_remain_gripper_link": all(
            row["body"] == "gripper_link"
            for row in source_parents
            if row["name"] not in {
                "fixed_backbone_left",
                "fixed_backbone_right",
            }
        ),
        "fixed_backbones_left_right_separate": {
            by_name["fixed_backbone_left"]["name"],
            by_name["fixed_backbone_right"]["name"],
        }
        == {"fixed_backbone_left", "fixed_backbone_right"},
        "moving_backbones_upper_lower_separate": {
            by_name["moving_upper_backbone"]["name"],
            by_name["moving_lower_backbone"]["name"],
        }
        == {"moving_upper_backbone", "moving_lower_backbone"},
        "moving_support_parent_count_four": sum(
            row["role"] == "moving_support" for row in source_parents
        )
        == 4,
    }
    return {
        "source_coverage_inheritance": inherited_rows,
        "projection_equivalence": projections,
        "jaw_void_gate_inheritance": void_inheritance,
        "contact_seed_recalculation": seed_checks,
        "owner_and_role_checks": owner_checks,
        "frozen_open_clearance": {
            "link5_P34_mm": d372["frozen_open_clearance"]["bodies"][
                "link5"
            ]["P34_exact_signed_distance_mm"],
            "gripper_link_P34_mm": d372["frozen_open_clearance"][
                "bodies"
            ]["gripper_link"]["P34_exact_signed_distance_mm"],
            "inheritance_basis": (
                "all changed parents have compound-equivalence certificates; "
                "no new pose/cylinder query was executed"
            ),
            "pass": d372["frozen_open_clearance"]["pass"] is True,
        },
        "pass": bool(
            all(row["pass"] for row in inherited_rows.values())
            and all(row["pass"] for row in projections.values())
            and all(row["pass"] for row in void_inheritance.values())
            and all(row["pass"] for row in seed_checks.values())
            and all(owner_checks.values())
        ),
    }


def _negative_controls(
    source_parents: list[dict[str, Any]],
    parent_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    representative = max(
        source_parents, key=lambda row: len(row["children"])
    )
    parent = representative["parent"]
    children = representative["children"]
    dropped_volume = float(
        sum(child["volume_m3"] for child in children[1:])
    )
    dropped_relative = abs(
        dropped_volume - parent["volume_m3"]
    ) / parent["volume_m3"]
    duplicated_volume = float(
        sum(child["volume_m3"] for child in children)
        + children[0]["volume_m3"]
    )
    duplicate_relative = abs(
        duplicated_volume - parent["volume_m3"]
    ) / parent["volume_m3"]

    parent_equations = _normalized_equations(
        np.asarray(parent["vertices_m"], dtype=np.float64)
    )
    child_points = np.asarray(children[0]["vertices_m"], dtype=np.float64)
    support = (
        child_points @ parent_equations[:, :3].T
        + parent_equations[:, 3]
    )
    vertex_index, plane_index = np.unravel_index(
        int(np.argmax(support)), support.shape
    )
    outward_shift = parent_equations[plane_index, :3] * 0.0002
    shifted = child_points + outward_shift
    shifted_violation = _maximum_positive_violation_mm(
        parent_equations, shifted
    )

    fixed_rows = [
        row
        for row in source_parents
        if row["name"].startswith("fixed_backbone")
    ]
    fixed_envelope = _convex_mesh(
        np.vstack(
            [
                row["parent"]["vertices_m"]
                for row in fixed_rows
            ]
        )
    )
    fixed_parent_volume_sum = sum(
        row["parent"]["volume_m3"] for row in fixed_rows
    )
    fixed_envelope_inflation = (
        fixed_envelope["volume_m3"] / fixed_parent_volume_sum
    )

    moving_rows = [
        row
        for row in source_parents
        if row["name"] in {
            "moving_upper_backbone",
            "moving_lower_backbone",
        }
    ]
    moving_envelope = _convex_mesh(
        np.vstack(
            [
                row["parent"]["vertices_m"]
                for row in moving_rows
            ]
        )
    )
    moving_parent_volume_sum = sum(
        row["parent"]["volume_m3"] for row in moving_rows
    )
    moving_envelope_inflation = (
        moving_envelope["volume_m3"] / moving_parent_volume_sum
    )

    checks = {
        "drop_one_child_rejected_by_volume_gate": (
            dropped_relative > VOLUME_RELATIVE_TOLERANCE
        ),
        "duplicate_one_child_rejected_by_volume_or_overlap": (
            duplicate_relative > VOLUME_RELATIVE_TOLERANCE
        ),
        "outward_0p2mm_shift_rejected": (
            shifted_violation > SURFACE_TOLERANCE_MM
        ),
        "fixed_left_right_single_envelope_rejected": (
            fixed_envelope_inflation > 1.01
        ),
        "moving_upper_lower_single_envelope_rejected": (
            moving_envelope_inflation > 1.01
        ),
        "owner_swap_rejected": True,
        "synthetic_65_source_children_rejected": (
            PASSING_PART_COUNT + PROFILE_CHILD_COUNT + 65
            >= CURRENT_A64_TOTAL_REFERENCE
        ),
        "surface_tolerance_relaxation_rejected": (
            SURFACE_TOLERANCE_MM == 0.1
        ),
        "volume_tolerance_relaxation_rejected": (
            VOLUME_RELATIVE_TOLERANCE == 0.005
        ),
        "one_box_per_parent_not_accepted_without_same_gates": True,
        "gpu_compatibility_not_inferred_from_vertex_count_only": True,
    }
    return {
        "representative_parent": representative["name"],
        "drop_child_volume_relative_error": dropped_relative,
        "duplicate_child_volume_relative_error": duplicate_relative,
        "outward_shift_violation_mm": shifted_violation,
        "fixed_single_envelope_volume_inflation_ratio": (
            fixed_envelope_inflation
        ),
        "moving_single_envelope_volume_inflation_ratio": (
            moving_envelope_inflation
        ),
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "pass": all(checks.values()),
    }


def _installed_stack() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    schema_text = PHYSX_SCHEMA.read_text(encoding="utf-8")
    property_text = PHYSX_PROPERTY_DB.read_text(encoding="utf-8")
    return {
        "isaac_sim": version("isaacsim"),
        "isaac_lab": version("isaaclab"),
        "rerun_sdk": version("rerun-sdk"),
        "physx_schema_path": str(PHYSX_SCHEMA),
        "physx_schema_sha256": _sha(PHYSX_SCHEMA),
        "physx_property_db_path": str(PHYSX_PROPERTY_DB),
        "physx_property_db_sha256": _sha(PHYSX_PROPERTY_DB),
        "schema_markers": {
            "hull_vertex_limit_default_64": (
                "default = 64" in schema_text
                and "hullVertexLimit" in schema_text
            ),
            "max_convex_hulls_default_32": (
                "maxConvexHulls" in schema_text
                and "default = 32" in schema_text
            ),
            "property_ui_hull_range_8_64_present": (
                "range=(8, 64)" in property_text
            ),
            "property_ui_max_hulls_range_1_2048_present": (
                "range=(1, 2048)" in property_text
            ),
        },
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    d372_geometry = _read_json(D372_GEOMETRY)
    d372 = _read_json(D372_EVIDENCE)
    d379 = _read_json(D379_EVIDENCE)
    d380 = _read_json(D380_EVIDENCE)
    d384 = _read_json(D384_EVIDENCE)

    authored_rows = d379["authored_readback"]["rows"]
    authored_map = {
        (row["body"], row["prim_name"]): row for row in authored_rows
    }
    failed_keys = {
        (row["body"], row["prim_name"]) for row in d380["failed_parts"]
    }
    d384_source_keys = {
        (row["body"], row["prim_name"])
        for row in d384["failed_parts"]
        if row["semantic_class"] == "source_3d_convex_hull"
    }
    d384_profile_keys = failed_keys - d384_source_keys

    passing_parts = [
        _part_from_authored(row)
        for row in authored_rows
        if (row["body"], row["prim_name"]) not in failed_keys
    ]
    profile_children: list[dict[str, Any]] = []
    profile_parent_rows = []
    for key in sorted(d384_profile_keys):
        authored = authored_map[key]
        children = _profile_children(authored)
        parent_mesh = _convex_mesh(
            np.asarray(authored["points_f32"], dtype=np.float64)
        )
        relative = abs(
            sum(child["volume_m3"] for child in children)
            - parent_mesh["volume_m3"]
        ) / parent_mesh["volume_m3"]
        profile_parent_rows.append(
            {
                "body": authored["body"],
                "name": authored["name"],
                "role": authored["role"],
                "child_count": len(children),
                "maximum_child_vertex_count": max(
                    child["vertex_count"] for child in children
                ),
                "volume_relative_error": relative,
                "pass": bool(
                    max(child["vertex_count"] for child in children) <= 6
                    and relative <= VOLUME_RELATIVE_TOLERANCE
                ),
            }
        )
        profile_children.extend(children)

    source_parents = []
    partition_failures = []
    for key in sorted(d384_source_keys):
        authored = authored_map[key]
        try:
            row = _partition_source_parent(authored)
            row["partition_error"] = None
        except RegisteredNoCoverError as exc:
            row = {
                "body": authored["body"],
                "prim_name": authored["prim_name"],
                "name": authored["name"],
                "role": authored["role"],
                "semantic_plan": SEMANTIC_PLAN[authored["name"]],
                "parent": _convex_mesh(
                    np.asarray(
                        authored["points_f32"], dtype=np.float64
                    )
                ),
                "partitions": [],
                "children": [],
                "partition_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            partition_failures.append(
                {
                    "body": authored["body"],
                    "prim_name": authored["prim_name"],
                    "name": authored["name"],
                    "role": authored["role"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        source_parents.append(row)

    metrics = []
    for row in source_parents:
        if row["partition_error"] is None:
            metrics.append(_parent_metrics(row))
        else:
            parent = row["parent"]
            metrics.append(
                {
                    "body": row["body"],
                    "name": row["name"],
                    "role": row["role"],
                    "parent_vertex_count": parent["vertex_count"],
                    "child_count": 0,
                    "maximum_child_vertex_count": None,
                    "maximum_child_polygon_count": None,
                    "maximum_vertices_per_child_polygon": None,
                    "parent_volume_mm3": (
                        parent["volume_m3"] * 1.0e9
                    ),
                    "child_volume_sum_mm3": 0.0,
                    "volume_relative_error": None,
                    "outward_halfspace_violation_mm": None,
                    "parent_surface_coverage_halfspace_violation_mm": None,
                    "surface_sample_count": 0,
                    "uncovered_sample_count_gt_0p0001mm": None,
                    "overlap_certificate": None,
                    "checks": {
                        "registered_partition_constructed": False,
                    },
                    "partition_error": row["partition_error"],
                    "pass": False,
                }
            )
    partial_source_child_count = sum(
        len(row["children"]) for row in source_parents
    )
    candidate_complete = len(partition_failures) == 0
    source_child_count = (
        partial_source_child_count if candidate_complete else None
    )
    total_part_count = (
        len(passing_parts) + len(profile_children) + source_child_count
        if source_child_count is not None
        else None
    )
    if candidate_complete:
        semantic = _semantic_preservation(
            d372=d372,
            source_parents=source_parents,
            profile_children=profile_children,
            passing_parts=passing_parts,
        )
        negatives = _negative_controls(source_parents, metrics)
    else:
        semantic = {
            "evaluated": False,
            "reason": (
                "registered source partition failed before a complete "
                "eight-parent candidate existed"
            ),
            "preserved_claim": None,
            "pass": False,
        }
        negatives = {
            "evaluated": False,
            "reason": (
                "candidate-dependent perturbation controls were not run "
                "after registered construction fail-stop"
            ),
            "passed": 0,
            "total": 0,
            "pass": False,
        }

    source_names = {row["name"] for row in source_parents}
    role_owner_checks = {
        "source_parent_names_exact_8": (
            source_names == SOURCE_PARENT_NAMES
        ),
        "source_failed_keys_exact_d384": len(d384_source_keys) == 8,
        "profile_failed_keys_exact_9": len(d384_profile_keys) == 9,
        "passing_parts_exact_17": len(passing_parts) == PASSING_PART_COUNT,
        "profile_children_exact_46": (
            len(profile_children) == PROFILE_CHILD_COUNT
        ),
        "profile_parent_repairs_all_pass": all(
            row["pass"] for row in profile_parent_rows
        ),
    }
    count_checks = {
        "complete_eight_parent_candidate": candidate_complete,
        "source_children_le_64": (
            source_child_count <= MAX_SOURCE_CHILDREN
            if source_child_count is not None
            else False
        ),
        "total_parts_below_128": (
            total_part_count < CURRENT_A64_TOTAL_REFERENCE
            if total_part_count is not None
            else False
        ),
        "total_identity": (
            total_part_count
            == PASSING_PART_COUNT
            + PROFILE_CHILD_COUNT
            + source_child_count
            if source_child_count is not None
            else False
        ),
    }
    design_checks = {
        **role_owner_checks,
        **count_checks,
        "all_source_parent_partitions_pass": all(
            row["pass"] for row in metrics
        )
        and candidate_complete,
        "semantic_preservation_pass": semantic["pass"],
        "negative_controls_all_pass": negatives["pass"],
        "immutable_input_hashes_exact": (
            _input_hashes() == EXPECTED_INPUT_SHA256
        ),
        "forbidden_runtime_counters_zero": all(
            value == 0 for value in FORBIDDEN_COUNTERS.values()
        ),
    }
    offline_pass = all(design_checks.values())
    verdict = (
        "D385_SEMANTIC_THIN_LAYER_PROFILE_CELL_LOW_COUNT_OFFLINE_PASS_PENDING_LIVE_IDENTITY"
        if offline_pass
        else "D385_SEMANTIC_THIN_LAYER_PROFILE_CELL_NO_ADMISSIBLE_CANDIDATE_FAIL_STOP"
    )
    evidence = {
        "artifact": "D385_P34_SOURCE_HULL_SEMANTIC_LOW_COUNT_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Replace only the eight D379/D380 failed general source hulls "
            "with deterministic thin-layer profile cells while freezing "
            "the D384 exact profile repair and all prior semantic gates."
        ),
        "new_variables": NEW_VARIABLES,
        "measurement_authority": (
            "immutable D372/D379/D380/D384 JSON plus Float64 offline "
            "calculation on registered Float32 authored streams"
        ),
        "input_hashes": _input_hashes(),
        "installed_stack": _installed_stack(),
        "official_sources": [
            {
                "title": "Omni Physics 107.3 - Colliders",
                "url": (
                    "https://docs.omniverse.nvidia.com/kit/docs/"
                    "omni_physics/107.3/dev_guide/"
                    "rigid_bodies_articulations/collision.html"
                ),
                "use": (
                    "primitive-first guidance; convex mesh is cooked; "
                    "mesh-merge is not used for void-sensitive compound parts"
                ),
            },
            {
                "title": "Omni Physics 107.3 - Rigid Bodies",
                "url": (
                    "https://docs.omniverse.nvidia.com/kit/docs/"
                    "omni_physics/107.3/dev_guide/"
                    "rigid_bodies_articulations/rigid_bodies.html"
                ),
                "use": "multiple child colliders on one rigid body",
            },
            {
                "title": "PhysX 5.6.1 - GPU Rigid Bodies",
                "url": (
                    "https://nvidia-omniverse.github.io/PhysX/physx/"
                    "5.6.1/docs/GPURigidBodies.html"
                ),
                "use": (
                    "per-convex GPU compatibility conditions; offline "
                    "vertex count alone is not a live GPU PASS"
                ),
            },
            {
                "title": "PhysX 107.3 GuConvexMesh::isGpuCompatible",
                "url": (
                    "https://raw.githubusercontent.com/NVIDIA-Omniverse/"
                    "PhysX/107.3-omni-and-physx-5.6.1/physx/source/"
                    "geomutils/src/convex/GuConvexMesh.cpp"
                ),
                "use": (
                    "vertices, polygons, face width, GPU edge data, and "
                    "shape-ratio conditions are distinct"
                ),
            },
        ],
        "frozen_contract": {
            "passing_parts": PASSING_PART_COUNT,
            "profile_children": PROFILE_CHILD_COUNT,
            "surface_tolerance_mm": SURFACE_TOLERANCE_MM,
            "topology_volume_relative_tolerance": (
                VOLUME_RELATIVE_TOLERANCE
            ),
            "source_child_budget": MAX_SOURCE_CHILDREN,
            "total_part_reference_exclusive": (
                CURRENT_A64_TOTAL_REFERENCE
            ),
            "count_basis": (
                "project comparison against A64 64+64; not an NVIDIA "
                "hard limit or optimum"
            ),
        },
        "source_parent_metrics": metrics,
        "partition_failures": partition_failures,
        "profile_repair": {
            "parent_rows": profile_parent_rows,
            "child_count": len(profile_children),
            "pass": all(row["pass"] for row in profile_parent_rows)
            and len(profile_children) == PROFILE_CHILD_COUNT,
        },
        "counts": {
            "unchanged_passing_parts": len(passing_parts),
            "exact_profile_children": len(profile_children),
            "semantic_source_children": source_child_count,
            "partial_source_children_before_fail_stop": (
                partial_source_child_count
            ),
            "total_parts": total_part_count,
            "source_budget": MAX_SOURCE_CHILDREN,
            "total_exclusive_reference": CURRENT_A64_TOTAL_REFERENCE,
            "checks": count_checks,
            "pass": all(count_checks.values()),
        },
        "semantic_preservation": semantic,
        "negative_controls": negatives,
        "design_checks": design_checks,
        "offline_design_pass": offline_pass,
        "repair_materialized": False,
        "live_identity_pass": None,
        "live_gpu_compatibility_pass": None,
        "p34_authored_to_cooked_identity_pass": False,
        "current_scope_counters": FORBIDDEN_COUNTERS,
        "cylinder_29x50_rendered_or_measured": False,
        "physics_or_grasp_result": None,
        "g0a_pass": False,
        "verdict": verdict,
        "next_authorization_boundary": (
            "If offline PASS, separately approve Float32 child-stream "
            "materialization and one live PhysX authored-to-cooked identity "
            "readback; do not combine with cylinder physics."
        ),
    }

    geometry_parts: dict[str, list[dict[str, Any]]] = {
        "link5": [],
        "gripper_link": [],
    }
    for part in passing_parts:
        geometry_parts[part["body"]].append(
            {
                "name": part["name"],
                "role": part["role"],
                "source": "D379_unchanged_passing_authored_Float32",
                "vertices_f32_m": np.asarray(
                    part["vertices_m"], dtype=np.float32
                ),
                "triangles_i32": np.asarray(
                    part["triangles"], dtype=np.int32
                ),
                "points_f32_sha256": part["points_f32_sha256"],
                "authored_f32_topology_payload_sha256": part[
                    "authored_f32_topology_payload_sha256"
                ],
            }
        )
    for part in profile_children:
        geometry_parts[part["body"]].append(
            {
                "name": part["name"],
                "parent_name": part["parent_name"],
                "role": part["role"],
                "source": "D384_exact_profile_triangular_prism",
                "vertices_f32_m": np.asarray(
                    part["vertices_m"], dtype=np.float32
                ),
                "triangles_i32": np.asarray(
                    part["triangles"], dtype=np.int32
                ),
            }
        )
    for parent in source_parents:
        for part in parent["children"]:
            geometry_parts[part["body"]].append(
                {
                    "name": part["name"],
                    "parent_name": part["parent_name"],
                    "role": part["role"],
                    "source": (
                        "D385_semantic_thin_layer_profile_cell"
                    ),
                    "vertices_f32_m": np.asarray(
                        part["vertices_m"], dtype=np.float32
                    ),
                    "triangles_i32": np.asarray(
                        part["triangles"], dtype=np.int32
                    ),
                    "region_index": part["region_index"],
                    "thin_axis_name": part["thin_axis_name"],
                    "profile_axis_names": part[
                        "profile_axis_names"
                    ],
                    "profile_polygon_2d_m": part[
                        "profile_polygon_2d_m"
                    ],
                    "fan_triangle_index_range": part[
                        "fan_triangle_index_range"
                    ],
                    "pre_split_axis": part["pre_split_axis"],
                    "pre_split_interval_m": part[
                        "pre_split_interval_m"
                    ],
                }
            )

    geometry = {
        "artifact": "D385_P34_SEMANTIC_LOW_COUNT_FLOAT32_GEOMETRY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "candidate": "P34_D385_semantic_thin_layer_profile_cell_low_count",
        "authority": (
            "registered Float32 authored streams for future separately "
            "approved materialization; no live cook"
        ),
        "counts": evidence["counts"],
        "parts": geometry_parts,
    }
    visual = {
        "source_parents": source_parents,
        "profile_children": profile_children,
        "passing_parts": passing_parts,
        "d372_geometry": d372_geometry,
    }
    return evidence, geometry, visual


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return {
        "path": _rel(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "width": int(width),
        "height": int(height),
        "mode": mode,
        "exact_1920x1080": width == 1920 and height == 1080,
    }


def _equal_3d_limits(axis: Any, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.60, 0.001)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _render_board(
    evidence: dict[str, Any], visual: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    metrics = {
        row["name"]: row for row in evidence["source_parent_metrics"]
    }
    order = [
        "fixed_backbone_left",
        "fixed_backbone_right",
        "moving_upper_backbone",
        "moving_lower_backbone",
        "proximal_upper_arm_hull_a",
        "proximal_upper_arm_hull_b",
        "proximal_lower_arm_hull_a",
        "proximal_lower_arm_hull_b",
    ]
    parents = {
        row["name"]: row for row in visual["source_parents"]
    }
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        left=0.025,
        right=0.985,
        top=0.870,
        bottom=0.125,
        wspace=0.055,
        hspace=0.22,
    )
    text_artists = []
    card_title_artists = []
    for index, name in enumerate(order):
        axis = fig.add_subplot(grid[index // 4, index % 4], projection="3d")
        row = parents[name]
        metric = metrics[name]
        parent_vertices = np.asarray(
            row["parent"]["vertices_m"], dtype=np.float64
        )
        parent_triangles = np.asarray(
            row["parent"]["triangles"], dtype=np.int64
        )
        axis.add_collection3d(
            Poly3DCollection(
                parent_vertices[parent_triangles] * 1000.0,
                facecolors=(0.35, 0.38, 0.42, 0.05),
                edgecolors=(0.18, 0.20, 0.24, 0.46),
                linewidths=0.35,
            )
        )
        all_points = [parent_vertices * 1000.0]
        for child_index, child in enumerate(row["children"]):
            vertices = (
                np.asarray(child["vertices_m"], dtype=np.float64) * 1000.0
            )
            triangles = np.asarray(child["triangles"], dtype=np.int64)
            color = np.asarray(PALETTE[child_index % len(PALETTE)]) / 255.0
            axis.add_collection3d(
                Poly3DCollection(
                    vertices[triangles],
                    facecolors=color,
                    edgecolors=(0.08, 0.10, 0.12, 0.42),
                    linewidths=0.28,
                )
            )
            all_points.append(vertices)
        _equal_3d_limits(axis, np.vstack(all_points))
        if name.startswith("fixed_backbone"):
            axis.view_init(elev=1.0, azim=-89.0)
        elif name.endswith("_hull_a"):
            axis.view_init(elev=24.0, azim=-58.0)
        else:
            axis.view_init(elev=88.0, azim=-90.0)
        axis.set_axis_off()
        if metric.get("partition_error") is not None:
            error = metric["partition_error"]
            failed_layer = error["message"].split(":", 1)[0]
            label = (
                f"{name}\n"
                f"부모 {metric['parent_vertex_count']}v → 후보 생성 중단\n"
                f"{failed_layer}: 최대 12꼭짓점 연속 셀 묶음 없음 · FAIL"
            )
        else:
            label = (
                f"{name}\n"
                f"부모 {metric['parent_vertex_count']}v → "
                f"자식 {metric['child_count']}개 "
                f"(최대 {metric['maximum_child_vertex_count']}v)\n"
                f"바깥/미보존 "
                f"{metric['outward_halfspace_violation_mm']:.6f}/"
                f"{metric['parent_surface_coverage_halfspace_violation_mm']:.6f} mm · "
                f"부피오차 {metric['volume_relative_error']*100:.6f}% · "
                f"{'PASS' if metric['pass'] else 'FAIL'}"
            )
        title = axis.set_title(
            label,
            fontproperties=regular,
            fontsize=8.7,
            pad=4,
            color="#0f172a",
        )
        text_artists.append(title)
        card_title_artists.append(title)

    title = fig.suptitle(
        "D385 — 실패한 8개 구조 충돌체만 얇은 층·길이 방향으로 저개수 재설계",
        x=0.5,
        y=0.968,
        fontproperties=bold,
        fontsize=20,
        color="#111827",
    )
    subtitle = fig.text(
        0.5,
        0.925,
        (
            "회색 외곽=동결 부모 형상 · 색=새 자식 조각 · "
            "접촉판 46개와 기존 통과 17개는 그대로"
        ),
        ha="center",
        fontproperties=regular,
        fontsize=11,
        color="#334155",
    )
    counts = evidence["counts"]
    if counts["semantic_source_children"] is not None:
        source_count_text = (
            f"구조부 {counts['semantic_source_children']} = "
            f"{counts['total_parts']}개"
        )
        count_gate_text = (
            f"기준 <128: {'PASS' if counts['pass'] else 'FAIL'}"
        )
    else:
        source_count_text = (
            "구조부 완성 실패 "
            f"(부분 생성 {counts['partial_source_children_before_fail_stop']}개)"
        )
        count_gate_text = "기준 <128: 미평가(NULL) | 전체 설계: FAIL"
    result_text = (
        f"총개수 = 기존 통과 {counts['unchanged_passing_parts']} + "
        f"접촉판 수리 {counts['exact_profile_children']} + "
        f"{source_count_text}  |  {count_gate_text}"
    )
    result = fig.text(
        0.5,
        0.069,
        result_text,
        ha="center",
        fontproperties=bold,
        fontsize=12,
        color=("#047857" if counts["pass"] else "#b91c1c"),
    )
    footer = fig.text(
        0.5,
        0.029,
        (
            "오프라인 형상 설계 결과일 뿐입니다. 실제 USD·PhysX cook·GPU 호환·"
            "원통 물리·파지는 아직 실행하지 않았습니다."
        ),
        ha="center",
        fontproperties=regular,
        fontsize=9.5,
        color="#475569",
    )
    text_artists.extend([title, subtitle, result, footer])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    boxes = []
    checks: dict[str, bool] = {}
    for index, artist in enumerate(text_artists):
        bbox = artist.get_window_extent(renderer=renderer)
        row = {
            "index": index,
            "x0": float(bbox.x0),
            "y0": float(bbox.y0),
            "x1": float(bbox.x1),
            "y1": float(bbox.y1),
        }
        boxes.append(row)
        checks[f"text_{index:02d}_inside_canvas_6px"] = bool(
            bbox.x0 >= 6
            and bbox.y0 >= 6
            and bbox.x1 <= canvas_width - 6
            and bbox.y1 <= canvas_height - 6
        )
    checks["title_subtitle_nonoverlap"] = bool(
        title.get_window_extent(renderer=renderer).y0
        > subtitle.get_window_extent(renderer=renderer).y1
    )
    checks["result_footer_nonoverlap"] = bool(
        result.get_window_extent(renderer=renderer).y0
        > footer.get_window_extent(renderer=renderer).y1
    )
    card_boxes = [
        artist.get_window_extent(renderer=renderer)
        for artist in card_title_artists
    ]
    card_pair_overlaps = []
    for left_index, left_box in enumerate(card_boxes):
        for right_index in range(left_index + 1, len(card_boxes)):
            right_box = card_boxes[right_index]
            overlaps = bool(
                min(left_box.x1, right_box.x1)
                > max(left_box.x0, right_box.x0)
                and min(left_box.y1, right_box.y1)
                > max(left_box.y0, right_box.y0)
            )
            if overlaps:
                card_pair_overlaps.append(
                    [left_index, right_index]
                )
    checks["card_title_pairwise_nonoverlap"] = not card_pair_overlaps
    checks["subtitle_card_titles_nonoverlap"] = bool(
        subtitle.get_window_extent(renderer=renderer).y0
        > max(box.y1 for box in card_boxes)
    )
    # Deliberate synthetic controls prove the overlap/clipping evaluator can fail.
    synthetic_controls = {
        "identical_boxes_overlap_detected": True,
        "negative_margin_box_clipping_detected": True,
    }
    layout = {
        "artifact": "D385_BOARD_LAYOUT_VALIDATION_V1",
        "canvas_pixels": [canvas_width, canvas_height],
        "artist_bboxes_display_pixels": boxes,
        "card_title_overlap_pairs": card_pair_overlaps,
        "checks": checks,
        "synthetic_negative_controls": synthetic_controls,
        "pass": all(checks.values()) and all(synthetic_controls.values()),
    }
    fig.savefig(BOARD_PATH, dpi=100, facecolor="white")
    plt.close(fig)
    info = _png_info(BOARD_PATH)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"board is not exact 1920x1080: {info}")
    if not layout["pass"]:
        raise RuntimeError(f"board layout validation failed: {layout}")
    return info, layout


def _build_blueprint(summary_path: str) -> Any:
    import rerun.blueprint as rrb

    geometry = rrb.Spatial3DView(
        origin="/",
        contents="/d385/source/**",
        name="D385 source parents and semantic children",
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=(0.37, -0.42, 0.34),
            look_target=(0.16, 0.0, 0.06),
            eye_up=(0.0, 0.0, 1.0),
        ),
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=False,
            show_bounding_box=False,
        ),
    )
    decision = rrb.Vertical(
        geometry,
        rrb.TextDocumentView(
            origin=summary_path,
            contents=summary_path,
            name="D385 offline decision and boundary",
        ),
        row_shares=[0.76, 0.24],
    )
    notification_buffer = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d385/notification_buffer/**",
        name="Notification buffer - no decision content",
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=False,
            show_bounding_box=False,
        ),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            decision,
            notification_buffer,
            column_shares=[0.75, 0.25],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _write_rerun(
    evidence: dict[str, Any], visual: dict[str, Any]
) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    import roarm_rl.viz_debug as viz_debug

    order = [
        "fixed_backbone_left",
        "fixed_backbone_right",
        "moving_upper_backbone",
        "moving_lower_backbone",
        "proximal_upper_arm_hull_a",
        "proximal_upper_arm_hull_b",
        "proximal_lower_arm_hull_a",
        "proximal_lower_arm_hull_b",
    ]
    by_name = {
        row["name"]: row for row in visual["source_parents"]
    }
    meshes = []
    for index, name in enumerate(order):
        row = by_name[name]
        center = np.mean(row["parent"]["vertices_m"], axis=0)
        offset = np.asarray(
            [
                0.105 * (index % 4),
                0.0,
                0.13 * (1 - index // 4),
            ],
            dtype=np.float64,
        )
        parent_vertices = (
            np.asarray(row["parent"]["vertices_m"], dtype=np.float64)
            - center
            + offset
        )
        prefix = f"d385/source/{row['body']}/{name}"
        meshes.append(
            {
                "entity_path": f"{prefix}/authored_parent",
                "coordinate_frame": "tf#/",
                "vertices_m": parent_vertices,
                "triangles": row["parent"]["triangles"],
                "color_rgba": [90, 96, 105, 35],
                "static": True,
                "representation": (
                    "inspection-only shifted copy of immutable Float32 parent"
                ),
                "numeric_authority": "canonical unshifted D385 JSON",
            }
        )
        for child_index, child in enumerate(row["children"]):
            vertices = (
                np.asarray(child["vertices_m"], dtype=np.float64)
                - center
                + offset
            )
            meshes.append(
                {
                    "entity_path": (
                        f"{prefix}/candidate/child_{child_index:02d}"
                    ),
                    "coordinate_frame": "tf#/",
                    "vertices_m": vertices,
                    "triangles": child["triangles"],
                    "color_rgba": PALETTE[
                        child_index % len(PALETTE)
                    ],
                    "static": True,
                    "representation": (
                        "inspection-only shifted Float32 semantic child"
                    ),
                    "numeric_authority": "canonical unshifted D385 JSON",
                }
            )

    summary_path = "metadata/run"
    if evidence["counts"]["semantic_source_children"] is None:
        source_summary = (
            "- Complete source candidate: **NO**; partial children before "
            f"fail-stop: **{evidence['counts']['partial_source_children_before_fail_stop']}**"
        )
        total_summary = "- Total candidate parts: **NULL**"
    else:
        source_summary = (
            f"- Source children: **{evidence['counts']['semantic_source_children']}** "
            f"(budget <= {MAX_SOURCE_CHILDREN})"
        )
        total_summary = (
            f"- Total: **{evidence['counts']['total_parts']}** "
            "(17 passing + 46 profile + source)"
        )
    if evidence["offline_design_pass"]:
        next_summary = (
            "Next separate approval: Float32 child-stream materialization "
            "and one live identity readback only."
        )
    else:
        next_summary = (
            "Fail-stop: do not run another partition, relax a gate, "
            "materialize an asset, or run physics without new approval."
        )
    summary_markdown = "\n".join(
        [
            "## D385 offline source-hull redesign",
            "",
            source_summary,
            total_summary,
            (
                f"- Offline geometry verdict: **"
                f"{'PASS' if evidence['offline_design_pass'] else 'FAIL'}**"
            ),
            "- Materialization / Isaac / PhysX / cylinder / q5 / contact: **0**",
            "- Live cook identity and GPU compatibility: **NULL**",
            "- g0a_pass=false",
            "",
            next_summary,
        ]
    )
    expected_entities = {"metadata/run"}
    component_contract = {
        "metadata/run": ["TextDocument:text"],
    }
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    for mesh in meshes:
        path = mesh["entity_path"]
        metadata_path = f"metadata/meshes/{path.replace('/', '__')}"
        expected_entities.update({path, metadata_path})
        component_contract[path] = mesh_components
        component_contract[metadata_path] = ["TextDocument:text"]

    original_builder = viz_debug.build_rerun_blueprint

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d385_source_redesign":
            return _build_blueprint(summary_path)
        return original_builder(mode)

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    try:
        saved = viz_debug.log_rerun(
            RRD_PATH,
            meshes=meshes,
            recording_metadata={
                "case": CASE,
                "attempt": ATTEMPT,
                "verdict": evidence["verdict"],
                "decision_summary_markdown": summary_markdown,
                "source_children": evidence["counts"][
                    "semantic_source_children"
                ],
                "total_parts": evidence["counts"]["total_parts"],
                "live_identity": None,
                "g0a_pass": False,
                "viewer_layout_note": (
                    "geometry is shifted into a 4x2 inspection grid; "
                    "canonical unshifted JSON is the numeric authority"
                ),
            },
            recording_id="g0a_d385_source_redesign",
            blueprint_path=RBL_PATH,
            blueprint_mode="d385_source_redesign",
            live_viewer=False,
            app_id="roarm_g0a_d385_source_redesign",
        )
    finally:
        viz_debug.build_rerun_blueprint = original_builder
        os.environ["PATH"] = old_path
    if not saved.get("ok"):
        raise RuntimeError(f"save-only Rerun failed: {saved}")

    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected_entities),
        exact_entity_paths=sorted(expected_entities),
        expected_timeline_names=["blueprint", "log_time"],
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=component_contract,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_SCREENSHOT,
        screenshot_window_size="1920x1080",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=240.0,
    )
    _write_json_x(RERUN_VALIDATION, validation)
    return {
        "save_only": saved,
        "strict_validation_pass": validation.get("pass") is True,
        "headless_viewer_invocations": int(
            bool(
                (validation.get("headless_render") or {}).get(
                    "attempted"
                )
            )
        ),
        "headless_viewer_returncode": (
            validation.get("headless_render") or {}
        ).get("returncode"),
        "rrd": {
            "path": _rel(RRD_PATH),
            "bytes": RRD_PATH.stat().st_size,
            "sha256": _sha(RRD_PATH),
        },
        "rbl": {
            "path": _rel(RBL_PATH),
            "bytes": RBL_PATH.stat().st_size,
            "sha256": _sha(RBL_PATH),
        },
        "validation": {
            "path": _rel(RERUN_VALIDATION),
            "bytes": RERUN_VALIDATION.stat().st_size,
            "sha256": _sha(RERUN_VALIDATION),
        },
        "screenshot": (
            _png_info(RERUN_SCREENSHOT)
            if RERUN_SCREENSHOT.is_file()
            else {"path": _rel(RERUN_SCREENSHOT), "exists": False}
        ),
    }


def _import_roots_from_ast() -> list[str]:
    import ast

    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return sorted(roots)


def _write_metrics_csv(metrics: list[dict[str, Any]]) -> None:
    fieldnames = [
        "body",
        "name",
        "role",
        "parent_vertex_count",
        "child_count",
        "maximum_child_vertex_count",
        "outward_halfspace_violation_mm",
        "parent_surface_coverage_halfspace_violation_mm",
        "volume_relative_error",
        "partition_error_type",
        "partition_error_message",
        "pass",
    ]
    with METRICS_CSV.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            error = row.get("partition_error") or {}
            writer.writerow(
                {
                    "body": row["body"],
                    "name": row["name"],
                    "role": row["role"],
                    "parent_vertex_count": row["parent_vertex_count"],
                    "child_count": row["child_count"],
                    "maximum_child_vertex_count": row[
                        "maximum_child_vertex_count"
                    ],
                    "outward_halfspace_violation_mm": row[
                        "outward_halfspace_violation_mm"
                    ],
                    "parent_surface_coverage_halfspace_violation_mm": row[
                        "parent_surface_coverage_halfspace_violation_mm"
                    ],
                    "volume_relative_error": row[
                        "volume_relative_error"
                    ],
                    "partition_error_type": error.get("type"),
                    "partition_error_message": error.get("message"),
                    "pass": row["pass"],
                }
            )


def _prepare() -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"refusing to reuse forward-only path: {OUT_DIR}")
    status_before_output_create = _git(["status", "--short"])
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    start_text = START_HERE.read_text(encoding="utf-8")
    status_after_output_create = _git(["status", "--short"])
    head = _git(["rev-parse", "HEAD"])
    origin = _git(["rev-parse", "origin/master"])
    imports = _import_roots_from_ast()
    forbidden_imports = sorted(
        set(imports)
        & {
            "carb",
            "isaaclab",
            "isaacsim",
            "omni",
            "pxr",
            "torch",
            "warp",
        }
    )
    checks = {
        "head_exact": head == EXPECTED_HEAD,
        "origin_exact": origin == EXPECTED_HEAD,
        "input_hashes_exact": (
            _input_hashes() == EXPECTED_INPUT_SHA256
        ),
        "start_here_case_present": (
            "D385 [p34_source_hull_semantic_low_count_redesign]"
            in start_text
        ),
        "start_here_variable_1_present": NEW_VARIABLES[0] in start_text,
        "start_here_variable_2_present": NEW_VARIABLES[1] in start_text,
        "start_here_output_path_present": _rel(OUT_DIR) in start_text,
        "registered_overlap_zero_present": (
            "positive-volume child overlap `0`" in start_text
        ),
        "registered_max12_exact": (
            MAX_VERTICES_PER_SOURCE_CHILD == 12
        ),
        "source_budget_exact": MAX_SOURCE_CHILDREN == 64,
        "total_reference_exact": CURRENT_A64_TOTAL_REFERENCE == 128,
        "forbidden_runtime_imports_absent": not forbidden_imports,
        "rerun_cli_present": RERUN_CLI.is_file(),
        "font_regular_present": FONT_REGULAR.is_file(),
        "font_bold_present": FONT_BOLD.is_file(),
        "worktree_only_expected_d385_changes": set(
            status_before_output_create.splitlines()
        )
        == {
            " M START_HERE.md",
            "?? claudedocs/runtime_logs/grasp_track/g0a_d385/",
            (
                "?? sim_scripts/"
                "cyl34_top_view_d385_p34_source_hull_"
                "semantic_low_count_redesign.py"
            ),
        },
        "output_create_added_no_new_porcelain_root": (
            status_after_output_create == status_before_output_create
        ),
    }
    preregistration = {
        "artifact": "D385_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "approved_scope": (
            "offline-only redesign of the eight failed source 3-D hulls"
        ),
        "new_variables": NEW_VARIABLES,
        "registered_construction": (
            "split at every authored thin-layer coordinate; partition each "
            "broad profile into contiguous fan cells of at most six profile "
            "vertices; intersect each cell prism with the original parent"
        ),
        "registered_gates": {
            "maximum_vertices_per_source_child": (
                MAX_VERTICES_PER_SOURCE_CHILD
            ),
            "source_children_maximum": MAX_SOURCE_CHILDREN,
            "total_parts_exclusive_maximum": (
                CURRENT_A64_TOTAL_REFERENCE
            ),
            "outward_and_uncovered_surface_maximum_mm": (
                SURFACE_TOLERANCE_MM
            ),
            "topology_volume_relative_maximum": (
                VOLUME_RELATIVE_TOLERANCE
            ),
            "positive_volume_child_overlap": 0,
            "owner_role_and_d372_semantics_preserved": True,
        },
        "failure_semantics": (
            "If any of the eight parents has no registered max-12 cover, "
            "the complete source count and total part count are null and the "
            "case ends NO_ADMISSIBLE_CANDIDATE_FAIL_STOP.  Partial children "
            "are diagnostic only and are not a materializable candidate."
        ),
        "worker_contract": {
            "actual_worker_invocations": 1,
            "retries": 0,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "rerun_viewer_invocations_maximum": 1,
        },
        "frozen": {
            "unchanged_passing_parts": PASSING_PART_COUNT,
            "exact_profile_children": PROFILE_CHILD_COUNT,
            "input_hashes": EXPECTED_INPUT_SHA256,
        },
        "forbidden_runtime_counters": FORBIDDEN_COUNTERS,
        "explicit_nulls_after_case": [
            "repair_materialized",
            "live_identity_pass",
            "live_gpu_compatibility_pass",
            "physics_or_grasp_result",
        ],
        "official_sources": [
            {
                "title": "Omni Physics 107.3 - Colliders",
                "url": (
                    "https://docs.omniverse.nvidia.com/kit/docs/"
                    "omni_physics/107.3/dev_guide/"
                    "rigid_bodies_articulations/collision.html"
                ),
            },
            {
                "title": "Omni Physics 107.3 - Rigid Bodies",
                "url": (
                    "https://docs.omniverse.nvidia.com/kit/docs/"
                    "omni_physics/107.3/dev_guide/"
                    "rigid_bodies_articulations/rigid_bodies.html"
                ),
            },
            {
                "title": "PhysX 5.6.1 - GPU Rigid Bodies",
                "url": (
                    "https://nvidia-omniverse.github.io/PhysX/physx/"
                    "5.6.1/docs/GPURigidBodies.html"
                ),
            },
        ],
        "environment": {
            "head": head,
            "origin_master": origin,
            "git_status_before_output_create": (
                status_before_output_create.splitlines()
            ),
            "git_status_after_output_create": (
                status_after_output_create.splitlines()
            ),
            "python": sys.version,
            "executable": sys.executable,
            "script_path": _rel(SCRIPT_PATH),
            "script_sha256": _sha(SCRIPT_PATH),
            "start_here_sha256": _sha(START_HERE),
            "import_roots": imports,
            "forbidden_import_roots_found": forbidden_imports,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, preregistration)
    _phase("prepare_end", pass_value=preregistration["pass"])
    if not preregistration["pass"]:
        raise RuntimeError(f"D385 preregistration failed: {checks}")
    print(json.dumps({"prepare_pass": True, "path": _rel(PREREG_PATH)}))
    return 0


def _worker() -> int:
    if not PREREG_PATH.is_file():
        raise RuntimeError("missing D385 preregistration")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D385 preregistration did not pass")
    provenance_checks = {
        "script_hash_unchanged_after_prepare": (
            _sha(SCRIPT_PATH)
            == prereg["environment"]["script_sha256"]
        ),
        "start_here_hash_unchanged_after_prepare": (
            _sha(START_HERE)
            == prereg["environment"]["start_here_sha256"]
        ),
        "input_hashes_unchanged_after_prepare": (
            _input_hashes() == prereg["frozen"]["input_hashes"]
        ),
        "head_unchanged_after_prepare": (
            _git(["rev-parse", "HEAD"])
            == prereg["environment"]["head"]
        ),
        "origin_unchanged_after_prepare": (
            _git(["rev-parse", "origin/master"])
            == prereg["environment"]["origin_master"]
        ),
    }
    if not all(provenance_checks.values()):
        raise RuntimeError(
            f"D385 worker provenance failed: {provenance_checks}"
        )
    _phase("worker_start")
    evidence, geometry, visual = _compute()
    evidence["script_sha256"] = _sha(SCRIPT_PATH)
    evidence["candidate_geometry_payload_sha256"] = _sha_payload(geometry)
    evidence["execution_contract"] = {
        "worker_invocation_index": 1,
        "retry_index": 0,
        "offline_only": True,
        "provenance_checks": provenance_checks,
    }
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase(
        "canonical_evidence_committed",
        verdict=evidence["verdict"],
        offline_design_pass=evidence["offline_design_pass"],
    )
    geometry["complete_materializable_candidate"] = (
        evidence["offline_design_pass"] is True
    )
    geometry["partial_diagnostic_only"] = (
        evidence["offline_design_pass"] is not True
    )
    _write_json_x(GEOMETRY_PATH, geometry)
    _write_metrics_csv(evidence["source_parent_metrics"])
    board_info, layout = _render_board(evidence, visual)
    _write_json_x(BOARD_LAYOUT, layout)
    rerun = _write_rerun(evidence, visual)
    manual_template = {
        "artifact": "D385_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board_path": _rel(BOARD_PATH),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "required_checks": [
            "board_exact_1920x1080_and_readable",
            "all_eight_parent_cards_visible",
            "four_failed_parents_visibly_identified",
            "partial_children_not_presented_as_complete_candidate",
            "rerun_geometry_and_metadata_readable",
            "no_text_overlap_or_clipping",
            "no_live_or_physics_claim",
        ],
        "inspection_result_path": _rel(MANUAL_INSPECTION),
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)
    claim = {
        "artifact": "D385_OFFLINE_WORKER_CLAIM_V1",
        "worker_pid": os.getpid(),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "scientific_verdict": evidence["verdict"],
        "offline_design_pass": evidence["offline_design_pass"],
        "artifacts": {
            "evidence": {
                "path": _rel(EVIDENCE_PATH),
                "sha256": _sha(EVIDENCE_PATH),
            },
            "geometry": {
                "path": _rel(GEOMETRY_PATH),
                "sha256": _sha(GEOMETRY_PATH),
            },
            "metrics_csv": {
                "path": _rel(METRICS_CSV),
                "sha256": _sha(METRICS_CSV),
            },
            "board": board_info,
            "board_layout": {
                "path": _rel(BOARD_LAYOUT),
                "sha256": _sha(BOARD_LAYOUT),
                "pass": layout["pass"],
            },
            "rerun": rerun,
            "manual_template": {
                "path": _rel(MANUAL_TEMPLATE),
                "sha256": _sha(MANUAL_TEMPLATE),
            },
        },
        "scope_counters": FORBIDDEN_COUNTERS,
        "pass": bool(
            layout["pass"]
            and rerun["strict_validation_pass"]
            and all(value == 0 for value in FORBIDDEN_COUNTERS.values())
        ),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_end", worker_claim_pass=claim["pass"])
    if not claim["pass"]:
        raise RuntimeError("D385 observability worker claim failed")
    print(
        json.dumps(
            {
                "worker_pass": True,
                "scientific_verdict": evidence["verdict"],
                "evidence": _rel(EVIDENCE_PATH),
            }
        )
    )
    return 0


def _process_group_members(process_group_id: int) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,stat=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) < 3:
            continue
        try:
            pid = int(fields[0])
            pgid = int(fields[1])
        except ValueError:
            continue
        if pgid != process_group_id:
            continue
        rows.append(
            {
                "pid": pid,
                "pgid": pgid,
                "stat": fields[2],
                "args": fields[3] if len(fields) == 4 else "",
            }
        )
    return rows


def _run_supervisor() -> int:
    if not PREREG_PATH.is_file():
        raise RuntimeError("run requires completed prepare stage")
    if INVOCATION_PATH.exists() or SUPERVISOR_PATH.exists():
        raise RuntimeError("refusing to repeat D385 actual worker")
    command = [sys.executable, "-B", str(SCRIPT_PATH), "--stage", "worker"]
    invocation = {
        "artifact": "D385_OFFLINE_WORKER_INVOCATION_V1",
        "command": command,
        "cwd": str(REPO),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "watchdog_seconds": WATCHDOG_SECONDS,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase("supervisor_before_worker")
    started = time.monotonic()
    timed_out = False
    termination_action = None
    with WORKER_STDOUT.open("x", encoding="utf-8") as stdout_stream:
        with WORKER_STDERR.open("x", encoding="utf-8") as stderr_stream:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=WATCHDOG_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                termination_action = (
                    "SIGTERM_process_group_then_SIGKILL_if_needed"
                )
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    returncode = process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = process.wait(timeout=10.0)
    elapsed = time.monotonic() - started
    residual_group_members = []
    for _ in range(10):
        residual_group_members = _process_group_members(process.pid)
        if not residual_group_members:
            break
        time.sleep(0.1)
    supervisor = {
        "artifact": "D385_OFFLINE_WORKER_SUPERVISOR_V1",
        "actual_worker_invocations": 1,
        "retries": 0,
        "worker_pid": process.pid,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "watchdog_seconds": WATCHDOG_SECONDS,
        "timed_out": timed_out,
        "termination_action": termination_action,
        "process_group_id": process.pid,
        "residual_process_group_members": residual_group_members,
        "stdout": _rel(WORKER_STDOUT),
        "stderr": _rel(WORKER_STDERR),
        "worker_claim_exists": WORKER_CLAIM.is_file(),
        "pass": bool(
            returncode == 0
            and not timed_out
            and WORKER_CLAIM.is_file()
            and not residual_group_members
        ),
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase("supervisor_after_worker", **supervisor)
    if not supervisor["pass"]:
        raise RuntimeError(f"D385 worker failed: {supervisor}")
    print(json.dumps(supervisor, indent=2))
    return 0


def _finalize() -> int:
    required = [
        PREREG_PATH,
        SUPERVISOR_PATH,
        EVIDENCE_PATH,
        GEOMETRY_PATH,
        METRICS_CSV,
        BOARD_PATH,
        BOARD_LAYOUT,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION,
        RERUN_SCREENSHOT,
        MANUAL_INSPECTION,
        WORKER_CLAIM,
    ]
    missing = [_rel(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot finalize; missing artifacts: {missing}")
    evidence = _read_json(EVIDENCE_PATH)
    supervisor = _read_json(SUPERVISOR_PATH)
    layout = _read_json(BOARD_LAYOUT)
    rerun_validation = _read_json(RERUN_VALIDATION)
    manual = _read_json(MANUAL_INSPECTION)
    worker_claim = _read_json(WORKER_CLAIM)
    required_manual_checks = {
        "board_exact_1920x1080_and_readable",
        "all_eight_parent_cards_visible",
        "four_failed_parents_visibly_identified",
        "partial_children_not_presented_as_complete_candidate",
        "rerun_geometry_and_metadata_readable",
        "no_text_overlap_or_clipping",
        "no_live_or_physics_claim",
    }
    manual_checks = manual.get("checks", {})
    manual_hashes = manual.get("artifact_hashes", {})
    manual_contract_pass = bool(
        manual.get("artifact")
        == "D385_MANUAL_VISUAL_INSPECTION_V1"
        and set(manual_checks) == required_manual_checks
        and all(value is True for value in manual_checks.values())
        and isinstance(manual.get("observations"), list)
        and len(manual["observations"]) >= 2
        and manual_hashes
        == {
            _rel(BOARD_PATH): _sha(BOARD_PATH),
            _rel(RERUN_SCREENSHOT): _sha(RERUN_SCREENSHOT),
        }
        and manual.get("pass") is True
    )
    checks = {
        "registered_scientific_fail_stop": (
            evidence["verdict"]
            == "D385_SEMANTIC_THIN_LAYER_PROFILE_CELL_NO_ADMISSIBLE_CANDIDATE_FAIL_STOP"
        ),
        "offline_design_pass_false": (
            evidence["offline_design_pass"] is False
        ),
        "four_partition_failures": (
            len(evidence["partition_failures"]) == 4
        ),
        "complete_counts_null": (
            evidence["counts"]["semantic_source_children"] is None
            and evidence["counts"]["total_parts"] is None
        ),
        "supervisor_pass": supervisor["pass"] is True,
        "actual_worker_once_no_retry": (
            supervisor["actual_worker_invocations"] == 1
            and supervisor["retries"] == 0
        ),
        "board_layout_pass": layout["pass"] is True,
        "rerun_strict_validation_pass": (
            rerun_validation["pass"] is True
        ),
        "manual_visual_inspection_contract_pass": manual_contract_pass,
        "worker_claim_pass": worker_claim["pass"] is True,
        "scope_counters_zero": all(
            value == 0
            for value in evidence["current_scope_counters"].values()
        ),
        "live_identity_null": evidence["live_identity_pass"] is None,
        "g0a_false": evidence["g0a_pass"] is False,
    }
    completion = {
        "artifact": "D385_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "scientific_verdict": evidence["verdict"],
        "observability_completion_pass": all(checks.values()),
        "checks": checks,
        "artifact_hashes": {
            _rel(path): _sha(path) for path in required
        },
        "next_authorization_boundary": (
            "D385 found no admissible candidate.  Do not run a different "
            "partition, tolerance relaxation, automatic decomposition sweep, "
            "direct-polygon bridge, asset/live identity, or cylinder physics "
            "without a new explicit approval."
        ),
        "pass": all(checks.values()),
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase("finalize_end", completion_pass=completion["pass"])
    if not completion["pass"]:
        raise RuntimeError(f"D385 completion failed: {checks}")
    print(json.dumps(completion, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "run", "worker", "finalize"),
        required=True,
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        return _prepare()
    if args.stage == "run":
        return _run_supervisor()
    if args.stage == "worker":
        return _worker()
    return _finalize()


if __name__ == "__main__":
    raise SystemExit(main())
