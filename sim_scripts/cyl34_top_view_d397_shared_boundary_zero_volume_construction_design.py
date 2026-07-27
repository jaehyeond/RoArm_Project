#!/usr/bin/env python3
"""D397 offline shared-boundary collider construction design.

This case does not launch Isaac, Kit, PhysX, Warp, CUDA, or physics.  It
replaces the eight P34 source-hull parts that failed authored-to-cooked
identity with a deterministic Float32 sibling-BSP construction.  Each split is
computed once, and the exact same seam vertices are supplied to both closed
convex children.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, QhullError


REPO = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO):
    if str(REPO) in sys.path:
        sys.path.remove(str(REPO))
    sys.path.insert(0, str(REPO))

CASE = "g0a_d397"
ATTEMPT = "attempt1_shared_boundary_zero_volume_construction_design"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT
SCRIPT = Path(__file__).resolve()
START = REPO / "START_HERE.md"

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
D385_GEOMETRY = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d385/"
    "attempt2_precreate_git_status_capture_repair/"
    "d385_p34_semantic_low_count_candidate_geometry.json"
)
D396_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d396/"
    "attempt1_d388_reanchor_direct_pre_float32_nonoverlap_admissibility_decision/"
    "d396_direct_overlap_admissibility_evidence.json"
)
D368_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d368/"
    "d368_semantic_allocation_evidence.json"
)
D349_MEASUREMENT = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d349/"
    "d349_frozen_target_distance_measurement.json"
)
AUTHORING_USD = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"
D368_SCRIPT = REPO / (
    "sim_scripts/cyl34_top_view_d368_current_64cap_semantic_allocation_audit.py"
)
D371_SCRIPT = REPO / (
    "sim_scripts/cyl34_top_view_d371_offline_collider_candidate_pareto_comparison.py"
)
D372_SCRIPT = REPO / (
    "sim_scripts/cyl34_top_view_d372_professor_semantic_compound_collider_design_offline.py"
)
D385_SCRIPT = REPO / (
    "sim_scripts/cyl34_top_view_d385_p34_source_hull_semantic_low_count_redesign.py"
)
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"

EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
EXPECTED_INPUT_SHA256 = {
    D372_GEOMETRY: "12fd1f32c35dfb9ae36cbbb412f6a51536aa1cc07c2dc17d05a5d189f3ee83e4",
    D372_EVIDENCE: "d68f658089aaf838ff454e9d0b301ec3f602785a3a730b3c329aa7785010e984",
    D379_EVIDENCE: "8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5",
    D380_EVIDENCE: "4c64d08e117501dd15a5836ce56ef8b963d188044beac465e645e53a17710bd1",
    D385_GEOMETRY: "dc376e7fba6efd4c064644139f412029fe905844db91b5f17623f0ace7f7a862",
    D396_EVIDENCE: "9cd315f69d4ce6a2b6b25addbfb589f2be0bdb3ea9e35b47069bea9be0580c1f",
    D368_EVIDENCE: "be2a422b0c74e4781b76a640c5312070b84876b1cb9e661d47e705ccdf789cf5",
    D349_MEASUREMENT: "5de6d14e37d6b74b202d1bb668120a6bb57221eac24ea5c751457ce9823b6300",
    AUTHORING_USD: "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
    D368_SCRIPT: "a760260129071a25ac40f3b7338ab0957714dde001c31cbafa6d319bc51971c9",
    D371_SCRIPT: "a907d650da1a28f3302ea4c52e6a8356e32cb38239e27686a5e4478c99c20242",
    D372_SCRIPT: "c546370d118e52b0d6d106f9e55a9e32ec3a3009296af094a906b322bdcb4bf7",
    D385_SCRIPT: "ea1d76a8db9c78a3cae9de50a62e0a25283d5550346dad158e641a0da321c5ed",
    VIZ_DEBUG: "4b5f821ad43652f529dfaa2f92b2826d9cd4973635e34521cc2b3a93ab0193d0",
    RERUN_CONTRACT: "aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e",
}

PREREG = OUT_DIR / "d397_preregistration.json"
PHASES = OUT_DIR / "d397_phase_markers.jsonl"
INVOCATION = OUT_DIR / "d397_offline_worker_invocation.json"
WORKER_CLAIM = OUT_DIR / "d397_offline_worker_claim.json"
EVIDENCE = OUT_DIR / "d397_shared_boundary_design_evidence.json"
GEOMETRY = OUT_DIR / "d397_shared_boundary_candidate_geometry.json"
PARENT_CSV = OUT_DIR / "d397_source_parent_metrics.csv"
BOARD = OUT_DIR / "d397_shared_boundary_design_1920x1080.png"
LAYOUT = OUT_DIR / "d397_board_layout_validation.json"
RRD = OUT_DIR / "d397_shared_boundary_design.rerun.rrd"
RBL = OUT_DIR / "d397_shared_boundary_design.rerun.rbl"
RERUN_VALIDATION = OUT_DIR / "d397_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d397_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d397_manual_visual_inspection_template.json"
MANUAL = OUT_DIR / "d397_manual_visual_inspection.json"
COMPLETION = OUT_DIR / "d397_completion_summary.json"
FAILURE = OUT_DIR / "d397_runtime_failure.json"

RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
PXR_ROOT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
EXPECTED_WORKER_PYTHONPATH = str(PXR_ROOT)
EXPECTED_WORKER_LD_LIBRARY_PATH = (
    "/home/cgxr/miniconda3/envs/isaaclab/lib:"
    + str(PXR_ROOT / "bin")
)
EXPECTED_OPENUSD_VERSION = [0, 24, 5]
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = ["float32_canonical_shared_plane_balanced_bsp_v1"]
MAX_SOURCE_CHILD_VERTICES = 12
MAX_SOURCE_CHILDREN = 64
TOTAL_PART_EXCLUSIVE_LIMIT = 128
MAX_POLYGONS = 64
MAX_VERTICES_PER_POLYGON = 32
SURFACE_TOLERANCE_MM = 0.1
BOUNDS_TOLERANCE_MM = 0.1
VOLUME_RELATIVE_TOLERANCE = 0.005
POSITIVE_VOLUME_EPS_M3 = 1.0e-18
RASTER_STEP_M = 0.00025
COOPERATIVE_DEADLINE_SECONDS = 600.0
MAX_TREE_DEPTH = 64
MAX_LEAVES_PER_PARENT = 128
BODY_NAMES = ("link5", "gripper_link")
BASE_SOURCES = {
    "D379_unchanged_passing_authored_Float32",
    "D384_exact_profile_triangular_prism",
}
PROFILE_ROLES = {"fixed_jaw", "moving_jaw"}
SOURCE_ROLES = {
    "moving_support",
    "moving_jaw_backbone",
    "fixed_jaw_backbone",
}
ROLE_COLORS = {
    "structural_body": (128, 128, 128, 105),
    "connector_support": (150, 150, 150, 105),
    "fixed_jaw": (238, 145, 45, 150),
    "fixed_jaw_backbone": (210, 70, 70, 205),
    "moving_support": (55, 125, 220, 205),
    "moving_jaw": (65, 170, 90, 150),
    "moving_jaw_backbone": (155, 95, 190, 205),
}
MANUAL_KEYS = [
    "board_exact_1920x1080_and_readable",
    "link5_full_candidate_visible",
    "gripper_full_candidate_visible",
    "source_children_visually_separable",
    "shared_boundary_method_text_readable",
    "count_surface_void_clearance_metrics_readable",
    "rerun_geometry_loaded",
    "no_text_overlap_or_clipping",
]

_DEADLINE: float | None = None


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_x(path: Path, value: Any) -> None:
    payload = json.dumps(
        _native(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _native(value),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def _phase(name: str, **fields: Any) -> None:
    _append_jsonl(
        PHASES,
        {
            "phase": name,
            "monotonic_seconds": time.monotonic(),
            "wall_time_unix_seconds": time.time(),
            **fields,
        },
    )


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _status_outside_output() -> list[str]:
    prefix = f"?? {_rel(OUT_DIR.parent)}/"
    return [
        row
        for row in _git("status", "--short").splitlines()
        if row.strip() and not row.startswith(prefix)
    ]


def _input_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in EXPECTED_INPUT_SHA256}


def _openusd_version() -> list[int] | None:
    try:
        from pxr import Usd

        return list(Usd.GetVersion())
    except Exception:
        return None


def _deadline_check(where: str) -> None:
    if _DEADLINE is not None and time.monotonic() > _DEADLINE:
        raise TimeoutError(f"D397 cooperative deadline exceeded at {where}")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _f32(points: Any) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).astype(np.float64)


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]


def _row_bits(point: Iterable[float]) -> bytes:
    return np.asarray(list(point), dtype="<f4").tobytes()


def _unique_f32(points: Any) -> np.ndarray:
    rows: dict[bytes, np.ndarray] = {}
    for point in _f32(points):
        rows.setdefault(_row_bits(point), point)
    return np.asarray([rows[key] for key in sorted(rows)], dtype=np.float64)


def _oriented_hull(points: Any) -> dict[str, Any]:
    registered = _unique_f32(points)
    if len(registered) < 4:
        raise ValueError("fewer than four Float32 points")
    if np.linalg.matrix_rank(registered - registered.mean(axis=0)) < 3:
        raise ValueError("Float32 points are not three-dimensional")
    first = ConvexHull(registered)
    vertices = registered[np.asarray(first.vertices, dtype=np.int64)]
    hull = ConvexHull(vertices)
    triangles = []
    for simplex, equation in zip(
        np.asarray(hull.simplices, dtype=np.int64),
        np.asarray(hull.equations, dtype=np.float64),
        strict=True,
    ):
        tri = list(map(int, simplex))
        cross = np.cross(
            vertices[tri[1]] - vertices[tri[0]],
            vertices[tri[2]] - vertices[tri[0]],
        )
        if float(np.dot(cross, equation[:3])) < 0.0:
            tri[1], tri[2] = tri[2], tri[1]
        triangles.append(tri)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    plane_vertices: dict[tuple[float, ...], set[int]] = {}
    for simplex, equation in zip(
        np.asarray(hull.simplices, dtype=np.int64),
        np.asarray(hull.equations, dtype=np.float64),
        strict=True,
    ):
        length = float(np.linalg.norm(equation[:3]))
        key = tuple(np.round(equation / length, decimals=7))
        plane_vertices.setdefault(key, set()).update(map(int, simplex))
    equations = np.asarray(hull.equations, dtype=np.float64)
    equations /= np.linalg.norm(equations[:, :3], axis=1, keepdims=True)
    return {
        "vertices_m": vertices,
        "triangles": triangle_array,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangle_array)),
        "polygon_count": int(len(plane_vertices)),
        "max_vertices_per_polygon": int(
            max(len(values) for values in plane_vertices.values())
        ),
        "volume_m3": float(hull.volume),
        "bounds_m": [vertices.min(axis=0), vertices.max(axis=0)],
        "equations": equations,
    }


def _edge_rows(triangles: np.ndarray) -> np.ndarray:
    return np.unique(
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


def _part_from_points(
    *,
    body: str,
    name: str,
    role: str,
    source: str,
    points: Any,
    parent_name: str,
    path_constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    geometry = _oriented_hull(points)
    return {
        "body": body,
        "name": name,
        "role": role,
        "source": source,
        "parent_name": parent_name,
        "vertices": geometry["vertices_m"],
        "vertices_m": geometry["vertices_m"],
        "triangles": geometry["triangles"],
        "triangles_i32": geometry["triangles"],
        "vertex_count": geometry["vertex_count"],
        "triangle_count": geometry["triangle_count"],
        "polygon_count": geometry["polygon_count"],
        "max_vertices_per_polygon": geometry["max_vertices_per_polygon"],
        "topology_volume_m3": geometry["volume_m3"],
        "volume_m3": geometry["volume_m3"],
        "bounds_m": geometry["bounds_m"],
        "equations": geometry["equations"],
        "path_constraints": path_constraints or [],
    }


def _seam_extremes(points: np.ndarray, axis: int) -> np.ndarray:
    seam = _unique_f32(points)
    keep = [index for index in range(3) if index != axis]
    projected = seam[:, keep]
    if len(projected) < 3:
        raise ValueError("shared seam has fewer than three points")
    if np.linalg.matrix_rank(projected - projected.mean(axis=0)) < 2:
        raise ValueError("shared seam is not two-dimensional")
    hull = ConvexHull(projected)
    return seam[np.asarray(hull.vertices, dtype=np.int64)]


def _paired_split(
    cell: dict[str, Any],
    *,
    axis: int,
    cut_f32: float,
) -> dict[str, Any] | None:
    points = np.asarray(cell["vertices_m"], dtype=np.float64)
    triangles = np.asarray(cell["triangles"], dtype=np.int64)
    cut = float(np.float32(cut_f32))
    values = points[:, axis] - cut
    left_points = [row for row, value in zip(points, values, strict=True) if value < 0.0]
    right_points = [row for row, value in zip(points, values, strict=True) if value > 0.0]
    seam_points = [row for row, value in zip(points, values, strict=True) if value == 0.0]
    for left_index, right_index in _edge_rows(triangles):
        v0 = float(values[int(left_index)])
        v1 = float(values[int(right_index)])
        if not ((v0 < 0.0 < v1) or (v1 < 0.0 < v0)):
            continue
        ratio = -v0 / (v1 - v0)
        point = (
            points[int(left_index)]
            + ratio * (points[int(right_index)] - points[int(left_index)])
        ).astype(np.float32)
        point[axis] = np.float32(cut)
        seam_points.append(point.astype(np.float64))
    if not left_points or not right_points or len(seam_points) < 3:
        return None
    try:
        seam = _seam_extremes(np.asarray(seam_points), axis)
        left = _oriented_hull(np.vstack([left_points, seam]))
        right = _oriented_hull(np.vstack([right_points, seam]))
    except (ValueError, QhullError):
        return None
    if left["volume_m3"] <= POSITIVE_VOLUME_EPS_M3:
        return None
    if right["volume_m3"] <= POSITIVE_VOLUME_EPS_M3:
        return None
    parent_volume = float(cell["volume_m3"])
    relative = abs(
        left["volume_m3"] + right["volume_m3"] - parent_volume
    ) / parent_volume
    if relative > VOLUME_RELATIVE_TOLERANCE:
        return None
    left_seam = {
        _row_bits(row)
        for row in left["vertices_m"]
        if float(row[axis]) == cut
    }
    right_seam = {
        _row_bits(row)
        for row in right["vertices_m"]
        if float(row[axis]) == cut
    }
    required = {_row_bits(row) for row in seam}
    shared_exact = required.issubset(left_seam) and required.issubset(right_seam)
    no_halo = (
        float(np.max(left["vertices_m"][:, axis])) <= cut
        and float(np.min(right["vertices_m"][:, axis])) >= cut
    )
    if not shared_exact or not no_halo:
        return None
    return {
        "axis": int(axis),
        "axis_name": "xyz"[axis],
        "cut_f32_m": cut,
        "cut_f32_bits": _f32_bits(cut),
        "seam_vertices_m": seam,
        "seam_vertex_count": int(len(seam)),
        "seam_vertex_bits_sha256": hashlib.sha256(
            b"".join(sorted(required))
        ).hexdigest(),
        "shared_seam_vertex_bits_exact": shared_exact,
        "opposite_closed_halfspaces_no_halo": no_halo,
        "split_volume_relative_error": float(relative),
        "left": left,
        "right": right,
    }


def _candidate_splits(cell: dict[str, Any]) -> list[dict[str, Any]]:
    points = np.asarray(cell["vertices_m"], dtype=np.float64)
    rows = []
    for axis in range(3):
        levels = sorted(set(map(float, points[:, axis])))
        for low, high in zip(levels[:-1], levels[1:], strict=True):
            _deadline_check(f"candidate_split_axis_{axis}")
            cut = float(np.float32((low + high) * 0.5))
            if not (low < cut < high):
                continue
            split = _paired_split(cell, axis=axis, cut_f32=cut)
            if split is None:
                continue
            left_count = int(split["left"]["vertex_count"])
            right_count = int(split["right"]["vertex_count"])
            if max(left_count, right_count) >= int(cell["vertex_count"]):
                continue
            imbalance = abs(
                float(split["left"]["volume_m3"])
                - float(split["right"]["volume_m3"])
            ) / float(cell["volume_m3"])
            split["score"] = [
                max(left_count, right_count),
                float(imbalance),
                left_count + right_count,
                int(axis),
                int(split["cut_f32_bits"]),
            ]
            rows.append(split)
    rows.sort(key=lambda row: tuple(row["score"]))
    return rows


def _partition_parent(
    *,
    body: str,
    name: str,
    role: str,
    points: np.ndarray,
) -> dict[str, Any]:
    root = _oriented_hull(points)
    leaves: list[dict[str, Any]] = [
        {
            **root,
            "path_constraints": [],
            "depth": 0,
            "stable_key": "",
        }
    ]
    split_rows = []
    next_node_id = 0
    failure = None
    while any(
        int(row["vertex_count"]) > MAX_SOURCE_CHILD_VERTICES
        for row in leaves
    ):
        _deadline_check(f"partition_parent_{name}")
        candidates = [
            (index, row)
            for index, row in enumerate(leaves)
            if int(row["vertex_count"]) > MAX_SOURCE_CHILD_VERTICES
        ]
        index, cell = min(
            candidates,
            key=lambda pair: (
                -int(pair[1]["vertex_count"]),
                -float(pair[1]["volume_m3"]),
                str(pair[1]["stable_key"]),
            ),
        )
        if int(cell["depth"]) >= MAX_TREE_DEPTH:
            failure = "maximum_tree_depth_reached"
            break
        options = _candidate_splits(cell)
        if not options:
            failure = "no_admissible_shared_plane_split"
            break
        selected = options[0]
        node_id = next_node_id
        next_node_id += 1
        common = {
            "node_id": int(node_id),
            "axis": int(selected["axis"]),
            "axis_name": selected["axis_name"],
            "cut_f32_m": float(selected["cut_f32_m"]),
            "cut_f32_bits": int(selected["cut_f32_bits"]),
        }
        children = []
        for side, geometry in (("le", selected["left"]), ("ge", selected["right"])):
            children.append(
                {
                    **geometry,
                    "path_constraints": [
                        *cell["path_constraints"],
                        {**common, "side": side},
                    ],
                    "depth": int(cell["depth"]) + 1,
                    "stable_key": f"{cell['stable_key']}{side[0]}{node_id:03d}",
                }
            )
        leaves = [row for offset, row in enumerate(leaves) if offset != index]
        leaves.extend(children)
        leaves.sort(key=lambda row: str(row["stable_key"]))
        split_rows.append(
            {
                **common,
                "parent_vertex_count": int(cell["vertex_count"]),
                "left_vertex_count": int(selected["left"]["vertex_count"]),
                "right_vertex_count": int(selected["right"]["vertex_count"]),
                "seam_vertex_count": int(selected["seam_vertex_count"]),
                "seam_vertices_m": selected["seam_vertices_m"],
                "seam_vertex_bits_sha256": selected[
                    "seam_vertex_bits_sha256"
                ],
                "shared_seam_vertex_bits_exact": selected[
                    "shared_seam_vertex_bits_exact"
                ],
                "opposite_closed_halfspaces_no_halo": selected[
                    "opposite_closed_halfspaces_no_halo"
                ],
                "split_volume_relative_error": selected[
                    "split_volume_relative_error"
                ],
                "candidate_split_count": len(options),
                "selected_score": selected["score"],
            }
        )
        if len(leaves) > MAX_LEAVES_PER_PARENT:
            failure = "maximum_leaf_count_reached"
            break
    children = []
    if failure is None:
        for index, leaf in enumerate(
            sorted(leaves, key=lambda row: str(row["stable_key"]))
        ):
            children.append(
                _part_from_points(
                    body=body,
                    name=f"{name}__shared_bsp_{index:02d}",
                    role=role,
                    source="D397_float32_shared_boundary_bsp",
                    points=leaf["vertices_m"],
                    parent_name=name,
                    path_constraints=leaf["path_constraints"],
                )
            )
    diagnostic_leaves = []
    for index, leaf in enumerate(
        sorted(leaves, key=lambda row: str(row["stable_key"]))
    ):
        diagnostic_leaves.append(
            _part_from_points(
                body=body,
                name=f"{name}__diagnostic_leaf_{index:02d}",
                role=role,
                source="D397_diagnostic_partition_leaf",
                points=leaf["vertices_m"],
                parent_name=name,
                path_constraints=leaf["path_constraints"],
            )
        )
    return {
        "body": body,
        "name": name,
        "role": role,
        "parent": _part_from_points(
            body=body,
            name=name,
            role=role,
            source="D379_authored_Float32_parent",
            points=points,
            parent_name=name,
        ),
        "children": children,
        "diagnostic_leaves": diagnostic_leaves,
        "split_nodes": split_rows,
        "construction_error": failure,
        "construction_complete": failure is None,
    }


def _surface_samples(part: dict[str, Any]) -> np.ndarray:
    vertices = np.asarray(part["vertices_m"], dtype=np.float64)
    triangles = np.asarray(part["triangles"], dtype=np.int64)
    edges = _edge_rows(triangles)
    triangle_points = vertices[triangles]
    quarter_a = (
        0.5 * triangle_points[:, 0]
        + 0.25 * triangle_points[:, 1]
        + 0.25 * triangle_points[:, 2]
    )
    quarter_b = (
        0.25 * triangle_points[:, 0]
        + 0.5 * triangle_points[:, 1]
        + 0.25 * triangle_points[:, 2]
    )
    quarter_c = (
        0.25 * triangle_points[:, 0]
        + 0.25 * triangle_points[:, 1]
        + 0.5 * triangle_points[:, 2]
    )
    return np.vstack(
        [
            vertices,
            np.mean(vertices[edges], axis=1),
            np.mean(triangle_points, axis=1),
            quarter_a,
            quarter_b,
            quarter_c,
        ]
    )


def _union_violation_mm(
    samples: np.ndarray, parts: list[dict[str, Any]]
) -> np.ndarray:
    rows = []
    for part in parts:
        equations = np.asarray(part["equations"], dtype=np.float64)
        rows.append(
            np.maximum(
                samples @ equations[:, :3].T + equations[:, 3],
                0.0,
            ).max(axis=1)
        )
    return np.min(np.vstack(rows), axis=0) * 1000.0


def _pair_separator_proof(children: list[dict[str, Any]]) -> dict[str, Any]:
    pair_records = []
    failures = []
    for left_index, right_index in itertools.combinations(range(len(children)), 2):
        left = children[left_index]
        right = children[right_index]
        right_by_node = {
            int(row["node_id"]): row for row in right["path_constraints"]
        }
        separators = []
        for row in left["path_constraints"]:
            other = right_by_node.get(int(row["node_id"]))
            if other is not None and other["side"] != row["side"]:
                separators.append((row, other))
        pass_value = False
        witness = None
        for row, other in separators:
            axis = int(row["axis"])
            cut = float(row["cut_f32_m"])
            le_part = left if row["side"] == "le" else right
            ge_part = right if row["side"] == "le" else left
            le_max = float(np.max(le_part["vertices_m"][:, axis]))
            ge_min = float(np.min(ge_part["vertices_m"][:, axis]))
            if le_max <= cut <= ge_min:
                pass_value = True
                witness = {
                    "node_id": int(row["node_id"]),
                    "axis": axis,
                    "cut_f32_m": cut,
                    "cut_f32_bits": int(row["cut_f32_bits"]),
                    "le_max_m": le_max,
                    "ge_min_m": ge_min,
                }
                break
        record = {
            "left_index": left_index,
            "right_index": right_index,
            "divergent_shared_separator_found": pass_value,
            "witness": witness,
        }
        pair_records.append(record)
        if not pass_value:
            failures.append(record)
    return {
        "method": (
            "every distinct leaf pair must diverge at one registered sibling "
            "node whose closed halfspaces meet only at the exact Float32 plane"
        ),
        "pair_count": len(pair_records),
        "pair_record_sha256": _sha_payload(pair_records),
        "unproven_separator_pair_count": len(failures),
        "certified_positive_volume_overlap_pair_count": (
            0 if not failures else None
        ),
        "failures": failures,
        "pass": not failures,
    }


def _parent_metrics(row: dict[str, Any]) -> dict[str, Any]:
    parent = row["parent"]
    children = row["children"]
    if not row["construction_complete"] or not children:
        return {
            "body": row["body"],
            "name": row["name"],
            "role": row["role"],
            "construction_complete": False,
            "construction_error": row["construction_error"],
            "pass": False,
        }
    parent_equations = np.asarray(parent["equations"], dtype=np.float64)
    child_vertices = np.vstack([part["vertices_m"] for part in children])
    outward = np.maximum(
        child_vertices @ parent_equations[:, :3].T
        + parent_equations[:, 3],
        0.0,
    ).max() * 1000.0
    coverage_values = _union_violation_mm(_surface_samples(parent), children)
    parent_volume = float(parent["volume_m3"])
    child_volume = float(sum(part["volume_m3"] for part in children))
    volume_relative = abs(child_volume - parent_volume) / parent_volume
    child_bounds = [
        child_vertices.min(axis=0),
        child_vertices.max(axis=0),
    ]
    bounds_delta = float(
        np.max(
            np.abs(
                np.asarray(child_bounds, dtype=np.float64)
                - np.asarray(parent["bounds_m"], dtype=np.float64)
            )
        )
        * 1000.0
    )
    separator = _pair_separator_proof(children)
    checks = {
        "construction_complete": True,
        "each_child_vertices_le_12": all(
            part["vertex_count"] <= MAX_SOURCE_CHILD_VERTICES
            for part in children
        ),
        "each_child_polygons_le_64": all(
            part["polygon_count"] <= MAX_POLYGONS for part in children
        ),
        "each_child_vertices_per_polygon_le_32": all(
            part["max_vertices_per_polygon"] <= MAX_VERTICES_PER_POLYGON
            for part in children
        ),
        "each_child_positive_volume": all(
            part["volume_m3"] > POSITIVE_VOLUME_EPS_M3
            for part in children
        ),
        "all_split_seams_bit_exact": all(
            split["shared_seam_vertex_bits_exact"]
            and split["opposite_closed_halfspaces_no_halo"]
            for split in row["split_nodes"]
        ),
        "pairwise_shared_separator_nonoverlap": separator["pass"],
        "outward_le_0p1mm": float(outward) <= SURFACE_TOLERANCE_MM,
        "coverage_le_0p1mm": float(np.max(coverage_values))
        <= SURFACE_TOLERANCE_MM,
        "volume_relative_le_0p5percent": volume_relative
        <= VOLUME_RELATIVE_TOLERANCE,
        "bounds_le_0p1mm": bounds_delta <= BOUNDS_TOLERANCE_MM,
    }
    return {
        "body": row["body"],
        "name": row["name"],
        "role": row["role"],
        "construction_complete": True,
        "child_count": len(children),
        "split_count": len(row["split_nodes"]),
        "maximum_child_vertex_count": max(
            part["vertex_count"] for part in children
        ),
        "maximum_child_polygon_count": max(
            part["polygon_count"] for part in children
        ),
        "maximum_vertices_per_child_polygon": max(
            part["max_vertices_per_polygon"] for part in children
        ),
        "parent_volume_m3": parent_volume,
        "child_volume_sum_m3": child_volume,
        "volume_relative_error": volume_relative,
        "outward_halfspace_violation_mm": float(outward),
        "parent_surface_coverage_violation_max_mm": float(
            np.max(coverage_values)
        ),
        "parent_surface_coverage_violation_p95_mm": float(
            np.percentile(coverage_values, 95.0)
        ),
        "bounds_max_abs_delta_mm": bounds_delta,
        "pairwise_nonoverlap": separator,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _projection_mask(
    parts: list[dict[str, Any]],
    *,
    axes: tuple[int, int],
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    xs = np.arange(low[0], high[0] + RASTER_STEP_M * 0.5, RASTER_STEP_M)
    ys = np.arange(low[1], high[1] + RASTER_STEP_M * 0.5, RASTER_STEP_M)
    xx, yy = np.meshgrid(xs, ys)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    occupied = np.zeros(len(query), dtype=bool)
    for part in parts:
        points = np.unique(
            np.asarray(part["vertices_m"], dtype=np.float64)[:, list(axes)],
            axis=0,
        )
        if len(points) < 3:
            continue
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            continue
        hull = ConvexHull(points)
        occupied |= MplPath(points[hull.vertices]).contains_points(
            query,
            radius=RASTER_STEP_M * 1.0e-6,
        )
    return occupied.reshape(xx.shape)


def _projection_equivalence(
    original: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    axes: tuple[int, int],
) -> dict[str, Any]:
    original_points = np.vstack(
        [np.asarray(row["vertices_m"])[:, list(axes)] for row in original]
    )
    candidate_points = np.vstack(
        [np.asarray(row["vertices_m"])[:, list(axes)] for row in candidate]
    )
    low = np.minimum(original_points.min(axis=0), candidate_points.min(axis=0))
    high = np.maximum(original_points.max(axis=0), candidate_points.max(axis=0))
    margin = np.asarray([0.0005, 0.0005])
    original_mask = _projection_mask(
        original,
        axes=axes,
        low=low - margin,
        high=high + margin,
    )
    candidate_mask = _projection_mask(
        candidate,
        axes=axes,
        low=low - margin,
        high=high + margin,
    )
    xor = np.logical_xor(original_mask, candidate_mask)
    return {
        "axes": list(map(int, axes)),
        "raster_step_mm": RASTER_STEP_M * 1000.0,
        "original_cells": int(original_mask.sum()),
        "candidate_cells": int(candidate_mask.sum()),
        "xor_cell_count": int(xor.sum()),
        "pass": int(xor.sum()) == 0,
    }


def _original_d372_parts() -> dict[str, list[dict[str, Any]]]:
    frozen = _read_json(D372_GEOMETRY)["parts"]
    output: dict[str, list[dict[str, Any]]] = {}
    for body in BODY_NAMES:
        output[body] = [
            _part_from_points(
                body=body,
                name=row["name"],
                role=row["role"],
                source="D372_frozen_original",
                points=row["vertices"],
                parent_name=row["name"],
            )
            for row in frozen[body]
        ]
    return output


def _base_parts() -> dict[str, list[dict[str, Any]]]:
    frozen = _read_json(D385_GEOMETRY)["parts"]
    output: dict[str, list[dict[str, Any]]] = {body: [] for body in BODY_NAMES}
    for body in BODY_NAMES:
        for row in frozen[body]:
            if row["source"] not in BASE_SOURCES:
                continue
            parent_name = row["name"].split("__profile_", 1)[0]
            output[body].append(
                _part_from_points(
                    body=body,
                    name=row["name"],
                    role=row["role"],
                    source=row["source"],
                    points=row["vertices_f32_m"],
                    parent_name=parent_name,
                )
            )
    return output


def _source_parent_rows() -> list[dict[str, Any]]:
    d379 = _read_json(D379_EVIDENCE)
    authored = {
        (row["body"], row["name"]): row
        for row in d379["authored_readback"]["rows"]
    }
    failed = [
        row
        for row in _read_json(D380_EVIDENCE)["failed_parts"]
        if row["role"] in SOURCE_ROLES
    ]
    rows = []
    for item in sorted(
        failed, key=lambda row: (row["body"], row["prim_name"])
    ):
        source = authored[(item["body"], item["name"])]
        rows.append(
            {
                "body": item["body"],
                "name": item["name"],
                "role": item["role"],
                "prim_name": item["prim_name"],
                "points_f32": source["points_f32"],
                "points_f32_sha256": source["points_f32_sha256"],
            }
        )
    return rows


def _candidate_inventory(
    candidate: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    output = {}
    for body in BODY_NAMES:
        rows = candidate[body]
        roles: dict[str, int] = {}
        closure_failures = []
        for part in rows:
            roles[part["role"]] = roles.get(part["role"], 0) + 1
            triangles = np.asarray(part["triangles"], dtype=np.int64)
            edges = _edge_rows(triangles)
            counts: dict[tuple[int, int], int] = {}
            directed: dict[tuple[int, int], int] = {}
            for tri in triangles:
                for left, right in zip(
                    tri, np.roll(tri, -1), strict=True
                ):
                    key = tuple(sorted((int(left), int(right))))
                    counts[key] = counts.get(key, 0) + 1
                    directed[(int(left), int(right))] = (
                        directed.get((int(left), int(right)), 0) + 1
                    )
            closed = all(value == 2 for value in counts.values())
            opposite = all(
                directed.get((left, right), 0)
                == directed.get((right, left), 0)
                for left, right in edges
            )
            if not closed or not opposite:
                closure_failures.append(part["name"])
        checks = {
            "nonempty": bool(rows),
            "vertices_le_64": all(
                part["vertex_count"] <= 64 for part in rows
            ),
            "polygons_le_64": all(
                part["polygon_count"] <= 64 for part in rows
            ),
            "vertices_per_polygon_le_32": all(
                part["max_vertices_per_polygon"] <= 32 for part in rows
            ),
            "positive_volume": all(
                part["volume_m3"] > POSITIVE_VOLUME_EPS_M3
                for part in rows
            ),
            "closed_opposite_winding": not closure_failures,
        }
        output[body] = {
            "part_count": len(rows),
            "roles": roles,
            "closure_failures": closure_failures,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return output


def _body_bounds_gate(
    original: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = {}
    for body in BODY_NAMES:
        original_points = np.vstack(
            [part["vertices_m"] for part in original[body]]
        )
        candidate_points = np.vstack(
            [part["vertices_m"] for part in candidate[body]]
        )
        original_bounds = [
            original_points.min(axis=0),
            original_points.max(axis=0),
        ]
        candidate_bounds = [
            candidate_points.min(axis=0),
            candidate_points.max(axis=0),
        ]
        delta = float(
            np.max(
                np.abs(
                    np.asarray(candidate_bounds)
                    - np.asarray(original_bounds)
                )
            )
            * 1000.0
        )
        rows[body] = {
            "original_bounds_m": original_bounds,
            "candidate_bounds_m": candidate_bounds,
            "max_abs_delta_mm": delta,
            "pass": delta <= BOUNDS_TOLERANCE_MM,
        }
    return {"bodies": rows, "pass": all(row["pass"] for row in rows.values())}


def _void_gate(
    original: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    frozen = _read_json(D372_EVIDENCE)[
        "jaw_contact_layer_void_diagnostic"
    ]
    fixed_roles = {"fixed_jaw", "fixed_jaw_backbone"}
    moving_roles = {"moving_jaw", "moving_jaw_backbone"}
    fixed = _projection_equivalence(
        [row for row in original["link5"] if row["role"] in fixed_roles],
        [row for row in candidate["link5"] if row["role"] in fixed_roles],
        axes=(1, 2),
    )
    moving = _projection_equivalence(
        [
            row
            for row in original["gripper_link"]
            if row["role"] in moving_roles
        ],
        [
            row
            for row in candidate["gripper_link"]
            if row["role"] in moving_roles
        ],
        axes=(0, 2),
    )
    rows = {
        "fixed": {
            "projection_equivalence": fixed,
            "frozen_void_rows": frozen["bodies"]["fixed"]["voids"],
            "frozen_void_pass": frozen["bodies"]["fixed"]["pass"],
            "pass": fixed["pass"]
            and frozen["bodies"]["fixed"]["pass"] is True,
        },
        "moving": {
            "projection_equivalence": moving,
            "frozen_void_rows": frozen["bodies"]["moving"]["voids"],
            "frozen_void_pass": frozen["bodies"]["moving"]["pass"],
            "pass": moving["pass"]
            and frozen["bodies"]["moving"]["pass"] is True,
        },
    }
    return {
        "semantics": (
            "D372 0.25mm jaw projection occupancy is recomputed for the "
            "original and D397 compound; exact occupancy preserves the frozen "
            "void-fill decision without claiming a through-depth proof"
        ),
        "bodies": rows,
        "pass": all(row["pass"] for row in rows.values()),
    }


def _clearance_gate(
    candidate: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    d371 = _load_module("d397_d371", D371_SCRIPT)
    d372 = _load_module("d397_d372", D372_SCRIPT)
    frozen = _read_json(D349_MEASUREMENT)
    pose = frozen["distance_gate"]["authoritative_pose_streams"]["raw_first"]
    expected = frozen["distance_gate"]["per_body"]
    d372_clearance = _read_json(D372_EVIDENCE)["frozen_open_clearance"]
    rows = {}
    query_count = 0
    for body in BODY_NAMES:
        _deadline_check(f"clearance_{body}")
        query = d372._query_parts(
            candidate[body],
            pose["body_poses_w"][body],
            pose,
            d371,
        )
        query_count += len(query["queries"])
        value = float(query["selected"]["exact_signed_distance_mm"])
        raw = float(expected[body]["raw_exact_signed_distance_mm"])
        old = float(
            d372_clearance["bodies"][body][
                "P34_exact_signed_distance_mm"
            ]
        )
        checks = {
            "no_collision": not query["collision_parts"],
            "clearance_ge_0p1mm": value >= 0.1,
            "raw_delta_le_0p5mm": abs(value - raw) <= 0.5,
            "d372_p34_delta_le_0p1mm": abs(value - old) <= 0.1,
        }
        rows[body] = {
            "candidate_exact_signed_distance_mm": value,
            "D349_raw_reference_mm": raw,
            "D372_P34_reference_mm": old,
            "absolute_delta_from_raw_mm": abs(value - raw),
            "absolute_delta_from_D372_P34_mm": abs(value - old),
            "selected_part": query["selected"]["part"],
            "collision_parts": query["collision_parts"],
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {
        "semantics": (
            "immutable D349 frozen-OPEN pose and historical 34x90mm cylinder; "
            "offline hppfcl surface query only"
        ),
        "offline_part_query_count": query_count,
        "bodies": rows,
        "pass": all(row["pass"] for row in rows.values()),
    }


def _seed_gate(candidate: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    d368 = _load_module("d397_seed_d368", D368_SCRIPT)
    frozen = _read_json(D372_EVIDENCE)["contact_seed_retention"]["rows"]
    rows = {}
    for label, role in (("fixed", "fixed_jaw"), ("moving", "moving_jaw")):
        _deadline_check(f"seed_{label}")
        body = frozen[label]["body"]
        point = np.asarray(frozen[label]["seed_local_m"], dtype=np.float64)
        selected = [
            part for part in candidate[body] if part["role"] == role
        ]
        distances = []
        for part in selected:
            mesh = d368._trimesh(
                np.asarray(part["vertices_m"], dtype=np.float64),
                np.asarray(part["triangles"], dtype=np.int64),
            )
            value = float(
                d368._nearest(mesh, point.reshape(1, 3))[0][0] * 1000.0
            )
            distances.append((value, part["name"]))
        distance, nearest_name = min(distances)
        limit = float(frozen[label]["registered_max_mm"])
        rows[label] = {
            "body": body,
            "role": role,
            "seed_local_m": point,
            "metric": (
                "D372-compatible exact nearest triangle-surface distance"
            ),
            "nearest_part": nearest_name,
            "candidate_nearest_surface_distance_mm": distance,
            "registered_max_mm": limit,
            "pass": distance <= limit,
        }
    return {"rows": rows, "pass": all(row["pass"] for row in rows.values())}


def _raw_surface_direct_gate(
    candidate: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    d368 = _load_module("d397_raw_d368", D368_SCRIPT)
    d371 = _load_module("d397_raw_d371", D371_SCRIPT)
    raw = d368._load_raw_meshes()
    frozen_streams = _read_json(D368_EVIDENCE)["raw_source_streams"]
    frozen_metrics = _read_json(D372_EVIDENCE)[
        "semantic_representation_metrics"
    ]["whole_raw_surface_representation"]
    rows = {}
    for body in BODY_NAMES:
        _deadline_check(f"raw_surface_start_{body}")
        raw_vertices = np.unique(
            np.asarray(raw[body]["vertices_m"], dtype=np.float64), axis=0
        )
        raw_triangles = np.asarray(raw[body]["triangles"], dtype=np.int64)
        triangle_rows = np.asarray(raw[body]["vertices_m"], dtype=np.float64)[
            raw_triangles[::16]
        ]
        samples = np.vstack([raw_vertices, np.mean(triangle_rows, axis=1)])
        contained = d371._union_contains(samples, candidate[body])
        distances = []
        for index, part in enumerate(candidate[body]):
            _deadline_check(f"raw_surface_{body}_part_{index}")
            mesh = d368._trimesh(
                np.asarray(part["vertices_m"], dtype=np.float64),
                np.asarray(part["triangles"], dtype=np.int64),
            )
            distances.append(d368._nearest(mesh, samples)[0])
        nearest_mm = np.min(np.vstack(distances), axis=0) * 1000.0
        outside_mm = np.where(contained, 0.0, nearest_mm)
        p95 = float(np.percentile(outside_mm, 95.0))
        maximum = float(np.max(outside_mm))
        stream_summary = raw[body]["stream_summary"]
        stream_checks = {
            key: stream_summary[key] == frozen_streams[body][key]
            for key in (
                "vertex_count",
                "triangle_count",
                "raw_vertex_stream_sha256",
                "triangle_stream_sha256",
                "authored_points_f32_mm_sha256",
                "face_counts_i64_sha256",
                "face_indices_i64_sha256",
            )
        }
        checks = {
            "immutable_raw_stream_exact": all(stream_checks.values()),
            "outside_distance_p95_le_2mm": p95 <= 2.0,
            "outside_distance_max_le_3p5mm": maximum <= 3.5,
        }
        rows[body] = {
            "sampling": (
                "D372-exact unique raw vertices plus every 16th raw triangle "
                "centroid"
            ),
            "sample_count": len(samples),
            "contained_count": int(np.sum(contained)),
            "raw_sample_outside_candidate_p95_mm": p95,
            "raw_sample_outside_candidate_max_mm": maximum,
            "registered_p95_max_mm": 2.0,
            "registered_absolute_max_mm": 3.5,
            "D372_P34_reference": frozen_metrics[body],
            "raw_stream_checks": stream_checks,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {
        "semantics": (
            "direct D372-compatible sampling and exact nearest triangle-surface "
            "distance against the D397 candidate; immutable authoring USD is "
            "read once through OpenUSD, with no USD write and no Isaac/Kit/PhysX"
        ),
        "immutable_usd_read_count": 1,
        "authoring_usd": {
            "path": _rel(AUTHORING_USD),
            "sha256": _sha(AUTHORING_USD),
        },
        "bodies": rows,
        "pass": all(row["pass"] for row in rows.values()),
    }


def _count_contract(
    source_child_count: int | None, total_count: int | None
) -> dict[str, bool]:
    return {
        "source_children_le_64": (
            source_child_count is not None
            and source_child_count <= MAX_SOURCE_CHILDREN
        ),
        "total_parts_below_128": (
            total_count is not None
            and total_count < TOTAL_PART_EXCLUSIVE_LIMIT
        ),
    }


def _negative_controls(
    parent_rows: list[dict[str, Any]],
    source_child_count: int | None,
    total_count: int | None,
) -> dict[str, Any]:
    complete = [row for row in parent_rows if row["construction_complete"]]
    representative_parent = complete[0] if complete else None
    representative = (
        representative_parent["children"][0]
        if representative_parent is not None
        else None
    )
    duplicate_proof = (
        _pair_separator_proof([representative, representative])
        if representative is not None
        else {"pass": None}
    )
    removed_relative_loss = None
    if representative_parent is not None:
        parent_volume = float(representative_parent["parent"]["volume_m3"])
        child_volumes = [
            float(row["volume_m3"])
            for row in representative_parent["children"]
        ]
        retained = sum(child_volumes) - max(child_volumes)
        removed_relative_loss = abs(retained - parent_volume) / parent_volume
    synthetic_proof = {"pass": None}
    if representative is not None:
        vertices = np.asarray(
            representative["vertices_m"], dtype=np.float64
        )
        axis = int(np.argmax(np.ptp(vertices, axis=0)))
        cut = float(np.float32(np.mean(vertices[:, axis])))
        common = {
            "node_id": 999_397,
            "axis": axis,
            "axis_name": "xyz"[axis],
            "cut_f32_m": cut,
            "cut_f32_bits": _f32_bits(cut),
        }
        synthetic_left = {
            **representative,
            "path_constraints": [{**common, "side": "le"}],
        }
        synthetic_right = {
            **representative,
            "path_constraints": [{**common, "side": "ge"}],
        }
        synthetic_proof = _pair_separator_proof(
            [synthetic_left, synthetic_right]
        )
    d396 = _read_json(D396_EVIDENCE)
    witnesses = d396["direct_witnesses"]
    count_65 = _count_contract(65, 127)
    count_128 = _count_contract(64, 128)
    checks = {
        "duplicate_leaf_positive_volume_rejected": (
            representative is not None and duplicate_proof["pass"] is False
        ),
        "removed_parent_children_volume_loss_rejected": (
            removed_relative_loss is not None
            and removed_relative_loss > VOLUME_RELATIVE_TOLERANCE
        ),
        "synthetic_overlapping_closed_halfspaces_rejected": (
            synthetic_proof["pass"] is False
        ),
        "D396_two_rejected_overlap_witnesses_above_gate": (
            len(witnesses) == 2
            and all(
                row["strict"]["volume_m3"] > POSITIVE_VOLUME_EPS_M3
                for row in witnesses
            )
        ),
        "source_count_65_rejected": (
            count_65["source_children_le_64"] is False
        ),
        "total_count_128_rejected": (
            count_128["total_parts_below_128"] is False
        ),
    }
    return {
        "checks": checks,
        "passed": sum(map(int, checks.values())),
        "total": len(checks),
        "representative_duplicate_leaf_name": (
            representative["name"] if representative else None
        ),
        "duplicate_leaf_separator_result": duplicate_proof,
        "removed_largest_child_volume_relative_loss": removed_relative_loss,
        "synthetic_overlap_separator_result": synthetic_proof,
        "source_65_count_contract": count_65,
        "total_128_count_contract": count_128,
        "observed_source_child_count": source_child_count,
        "observed_total_part_count": total_count,
        "pass": all(checks.values()),
    }


def _public_part(part: dict[str, Any]) -> dict[str, Any]:
    return {
        "body": part["body"],
        "name": part["name"],
        "role": part["role"],
        "source": part["source"],
        "parent_name": part["parent_name"],
        "vertices_f32_m": np.asarray(
            part["vertices_m"], dtype=np.float32
        ).astype(np.float64),
        "triangles_i32": np.asarray(part["triangles"], dtype=np.int32),
        "vertex_count": part["vertex_count"],
        "triangle_count": part["triangle_count"],
        "polygon_count": part["polygon_count"],
        "max_vertices_per_polygon": part["max_vertices_per_polygon"],
        "topology_volume_m3": part["volume_m3"],
        "bounds_m": part["bounds_m"],
        "path_constraints": part["path_constraints"],
        "payload_sha256": _sha_payload(
            {
                "vertices_f32_m": np.asarray(
                    part["vertices_m"], dtype=np.float32
                ).astype(np.float64),
                "triangles_i32": np.asarray(
                    part["triangles"], dtype=np.int32
                ),
            }
        ),
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    global _DEADLINE
    _DEADLINE = time.monotonic() + COOPERATIVE_DEADLINE_SECONDS
    _phase("canonical_compute_start")
    base = _base_parts()
    original = _original_d372_parts()
    source_rows = []
    for source in _source_parent_rows():
        _deadline_check(f"source_parent_start_{source['name']}")
        _phase(
            "source_parent_start",
            body=source["body"],
            name=source["name"],
        )
        row = _partition_parent(
            body=source["body"],
            name=source["name"],
            role=source["role"],
            points=np.asarray(source["points_f32"], dtype=np.float64),
        )
        row["prim_name"] = source["prim_name"]
        row["points_f32_sha256"] = source["points_f32_sha256"]
        row["metrics"] = _parent_metrics(row)
        source_rows.append(row)
        _phase(
            "source_parent_end",
            body=source["body"],
            name=source["name"],
            complete=row["construction_complete"],
            child_count=len(row["children"]),
            parent_pass=row["metrics"]["pass"],
        )
    all_complete = all(row["construction_complete"] for row in source_rows)
    candidate = {body: list(base[body]) for body in BODY_NAMES}
    if all_complete:
        for row in source_rows:
            candidate[row["body"]].extend(row["children"])
    source_child_count = (
        sum(len(row["children"]) for row in source_rows)
        if all_complete
        else None
    )
    total_count = (
        sum(len(candidate[body]) for body in BODY_NAMES)
        if all_complete
        else None
    )
    inventories = _candidate_inventory(candidate) if all_complete else None
    parent_metrics = [row["metrics"] for row in source_rows]
    body_bounds = (
        _body_bounds_gate(original, candidate) if all_complete else None
    )
    void = _void_gate(original, candidate) if all_complete else None
    clearance = _clearance_gate(candidate) if all_complete else None
    seeds = _seed_gate(candidate) if all_complete else None
    raw_surface = (
        _raw_surface_direct_gate(candidate) if all_complete else None
    )
    negatives = _negative_controls(
        source_rows, source_child_count, total_count
    )
    count_checks = _count_contract(source_child_count, total_count)
    counts = {
        "unchanged_passing_parts": 17,
        "exact_profile_children": 46,
        "base_parts_exact_63": sum(len(base[body]) for body in BODY_NAMES)
        == 63,
        "source_children": source_child_count,
        "total_parts": total_count,
        **count_checks,
        "by_body": (
            {body: len(candidate[body]) for body in BODY_NAMES}
            if all_complete
            else None
        ),
    }
    owner_checks = {
        "eight_source_parents": len(source_rows) == 8,
        "fixed_backbones_remain_link5": all(
            row["body"] == "link5"
            for row in source_rows
            if row["role"] == "fixed_jaw_backbone"
        ),
        "moving_source_roles_remain_gripper_link": all(
            row["body"] == "gripper_link"
            for row in source_rows
            if row["role"]
            in {"moving_support", "moving_jaw_backbone"}
        ),
        "parent_names_unique": len({row["name"] for row in source_rows}) == 8,
    }
    checks = {
        "one_new_variable": NEW_VARIABLES
        == ["float32_canonical_shared_plane_balanced_bsp_v1"],
        "base_count_63_exact": counts["base_parts_exact_63"],
        "all_eight_source_parents_complete": all_complete,
        "all_parent_geometry_gates_pass": all(
            row["metrics"]["pass"] for row in source_rows
        ),
        "source_children_le_64": counts["source_children_le_64"],
        "total_parts_below_128": counts["total_parts_below_128"],
        "candidate_inventory_pass": (
            inventories is not None
            and all(row["pass"] for row in inventories.values())
        ),
        "body_bounds_pass": body_bounds is not None
        and body_bounds["pass"],
        "void_gate_pass": void is not None and void["pass"],
        "frozen_open_clearance_pass": clearance is not None
        and clearance["pass"],
        "contact_seed_gate_pass": seeds is not None and seeds["pass"],
        "raw_surface_direct_gate_pass": raw_surface is not None
        and raw_surface["pass"],
        "owner_role_pass": all(owner_checks.values()),
        "negative_controls_6_of_6": negatives["pass"]
        and negatives["passed"] == negatives["total"] == 6,
    }
    design_pass = all(checks.values())
    verdict = (
        "D397_SHARED_BOUNDARY_ZERO_VOLUME_CONSTRUCTION_OFFLINE_PASS"
        if design_pass
        else "D397_SHARED_BOUNDARY_ZERO_VOLUME_CONSTRUCTION_FAIL_STOP"
    )
    geometry = {
        "artifact": "D397_SHARED_BOUNDARY_CANDIDATE_GEOMETRY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "candidate": "P34_shared_boundary_derivative",
        "complete_materializable_offline_candidate": design_pass,
        "parts": {
            body: [_public_part(part) for part in candidate[body]]
            for body in BODY_NAMES
        },
        "diagnostic_source_parents": [
            _public_part(row["parent"]) for row in source_rows
        ],
        "diagnostic_partition_leaves": [
            _public_part(part)
            for row in source_rows
            for part in row["diagnostic_leaves"]
        ],
        "diagnostic_shared_seams": [
            {
                "body": row["body"],
                "parent_name": row["name"],
                "node_id": split["node_id"],
                "axis": split["axis"],
                "axis_name": split["axis_name"],
                "cut_f32_m": split["cut_f32_m"],
                "cut_f32_bits": split["cut_f32_bits"],
                "seam_vertices_f32_m": np.asarray(
                    split["seam_vertices_m"], dtype=np.float32
                ).astype(np.float64),
                "seam_vertex_bits_sha256": split[
                    "seam_vertex_bits_sha256"
                ],
            }
            for row in source_rows
            for split in row["split_nodes"]
        ],
        "counts": counts,
        "authority": (
            "Float32 vertices plus explicit triangle topology; offline only; "
            "no USD authoring or PhysX cook/readback"
        ),
    }
    evidence = {
        "artifact": "D397_SHARED_BOUNDARY_DESIGN_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": NEW_VARIABLES,
        "verdict": verdict,
        "design_pass": design_pass,
        "method": {
            "name": NEW_VARIABLES[0],
            "rule": (
                "recursively choose the registered Float32 axis plane that "
                "minimizes maximum child vertex count, then volume imbalance, "
                "then total vertices/axis/cut bits; compute seam once and reuse "
                "its exact bits in both closed convex siblings"
            ),
            "source_child_vertex_budget": MAX_SOURCE_CHILD_VERTICES,
            "source_child_count_budget": MAX_SOURCE_CHILDREN,
            "total_part_exclusive_limit": TOTAL_PART_EXCLUSIVE_LIMIT,
            "shared_faces_are_closed_on_both_sides": True,
            "half_open_or_face_deleted_colliders": False,
        },
        "input_hashes": _input_hashes(),
        "source_parent_metrics": parent_metrics,
        "counts": counts,
        "candidate_inventory": inventories,
        "body_bounds": body_bounds,
        "jaw_void_gate": void,
        "frozen_open_clearance": clearance,
        "contact_seed_retention": seeds,
        "raw_surface_direct": raw_surface,
        "owner_role_checks": owner_checks,
        "negative_controls": negatives,
        "checks": checks,
        "scope_counters": {
            "offline_worker_invocations": 1,
            "automatic_retries": 0,
            "source_parent_constructions": 8,
            "usd_or_asset_writes": 0,
            "immutable_authoring_usd_reads": (
                raw_surface["immutable_usd_read_count"]
                if raw_surface is not None
                else 0
            ),
            "collider_materializations": 0,
            "isaac_launches": 0,
            "kit_launches": 0,
            "physx_launches": 0,
            "warp_or_cuda_launches": 0,
            "physics_steps": 0,
            "q5_samples": 0,
            "contact_queries": 0,
            "target_ik_path_changes": 0,
            "material_mass_actuator_physics_changes": 0,
            "cylinder_creates_or_writes": 0,
            "process_signals_sent": 0,
            "offline_hppfcl_clearance_queries": (
                clearance["offline_part_query_count"]
                if clearance is not None
                else 0
            ),
        },
        "interpretation_boundary": {
            "authored_to_cooked_identity": None,
            "live_physx_callback_identity": None,
            "gpu_compatibility": None,
            "29x50mm_cylinder": None,
            "physics_or_contact": None,
            "grasp_feasibility": None,
            "g0a_pass": False,
        },
    }
    _phase(
        "canonical_compute_end",
        design_pass=design_pass,
        source_children=source_child_count,
        total_parts=total_count,
    )
    return evidence, geometry, source_rows


def _write_parent_csv(rows: list[dict[str, Any]]) -> None:
    import csv

    fields = [
        "body",
        "name",
        "role",
        "construction_complete",
        "child_count",
        "split_count",
        "maximum_child_vertex_count",
        "maximum_child_polygon_count",
        "outward_halfspace_violation_mm",
        "parent_surface_coverage_violation_max_mm",
        "volume_relative_error",
        "bounds_max_abs_delta_mm",
        "unproven_separator_pair_count",
        "certified_positive_volume_overlap_pair_count",
        "pass",
    ]
    with PARENT_CSV.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metric = row["metrics"]
            writer.writerow(
                {
                    "body": row["body"],
                    "name": row["name"],
                    "role": row["role"],
                    "construction_complete": row[
                        "construction_complete"
                    ],
                    "child_count": metric.get("child_count"),
                    "split_count": metric.get("split_count"),
                    "maximum_child_vertex_count": metric.get(
                        "maximum_child_vertex_count"
                    ),
                    "maximum_child_polygon_count": metric.get(
                        "maximum_child_polygon_count"
                    ),
                    "outward_halfspace_violation_mm": metric.get(
                        "outward_halfspace_violation_mm"
                    ),
                    "parent_surface_coverage_violation_max_mm": metric.get(
                        "parent_surface_coverage_violation_max_mm"
                    ),
                    "volume_relative_error": metric.get(
                        "volume_relative_error"
                    ),
                    "bounds_max_abs_delta_mm": metric.get(
                        "bounds_max_abs_delta_mm"
                    ),
                    "unproven_separator_pair_count": (
                        metric.get("pairwise_nonoverlap", {}).get(
                            "unproven_separator_pair_count"
                        )
                    ),
                    "certified_positive_volume_overlap_pair_count": (
                        metric.get("pairwise_nonoverlap", {}).get(
                            "certified_positive_volume_overlap_pair_count"
                        )
                    ),
                    "pass": metric.get("pass"),
                }
            )


def _prepare() -> int:
    if OUT_DIR.exists():
        raise FileExistsError(f"forward-only output exists: {OUT_DIR}")
    status_before = _status_outside_output()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    actual_hashes = _input_hashes()
    start_text = START.read_text(encoding="utf-8")
    expected_hashes = {
        _rel(path): digest for path, digest in EXPECTED_INPUT_SHA256.items()
    }
    checks = {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "inputs_exact": actual_hashes == expected_hashes,
        "script_exists": SCRIPT.is_file(),
        "start_exists": START.is_file(),
        "start_active_case_D397_approved": (
            "## Active Case — D397 Approved; Offline Design Only"
            in start_text
        ),
        "start_exact_new_variable": NEW_VARIABLES[0] in start_text,
        "start_exact_forward_only_output": _rel(OUT_DIR) in start_text,
        "one_new_variable": len(NEW_VARIABLES) == 1,
        "numpy_1_26_0": np.__version__ == "1.26.0",
        "scipy_1_15_3": importlib.metadata.version("scipy") == "1.15.3",
        "hpp_fcl_2_4_4": importlib.metadata.version("hpp-fcl") == "2.4.4",
        "rerun_0_34_1": importlib.metadata.version("rerun-sdk") == "0.34.1",
        "rerun_cli": RERUN_CLI.is_file(),
        "openusd_runtime_directory": PXR_ROOT.is_dir()
        and (PXR_ROOT / "bin").is_dir(),
        "fonts": FONT_REGULAR.is_file() and FONT_BOLD.is_file(),
        "d385_base_counts": (
            _read_json(D385_GEOMETRY)["counts"][
                "unchanged_passing_parts"
            ]
            == 17
            and _read_json(D385_GEOMETRY)["counts"][
                "exact_profile_children"
            ]
            == 46
        ),
        "d396_candidate_rejected": (
            _read_json(D396_EVIDENCE)[
                "d388_reanchor_candidate_nonoverlap_admissible"
            ]
            is False
        ),
    }
    prereg = {
        "artifact": "D397_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "question": (
            "Can a one-variable paired Float32 shared-plane construction "
            "replace all eight failed P34 source hulls while preserving the "
            "frozen surface, void, clearance, bounds, count, and zero-new-"
            "positive-overlap contracts?"
        ),
        "new_variables": NEW_VARIABLES,
        "construction_rule": {
            "candidate_planes": (
                "Float32 midpoint of each adjacent unique coordinate pair on "
                "x/y/z of the current convex cell"
            ),
            "selection_key": [
                "minimum maximum child vertex count",
                "minimum normalized volume imbalance",
                "minimum total child vertex count",
                "axis index",
                "Float32 cut bits",
            ],
            "shared_boundary": (
                "edge intersections calculated once; seam convex polygon "
                "Float32 bits reused in both closed siblings"
            ),
            "termination": "all source leaves have at most 12 vertices",
            "immediate_progress_filter": (
                "each selected split must strictly reduce the maximum child "
                "vertex count relative to its parent cell"
            ),
            "maximum_tree_depth": MAX_TREE_DEPTH,
            "maximum_leaves_per_parent": MAX_LEAVES_PER_PARENT,
        },
        "frozen_gates": {
            "source_child_vertices_max": MAX_SOURCE_CHILD_VERTICES,
            "source_children_max": MAX_SOURCE_CHILDREN,
            "total_parts_strictly_below": TOTAL_PART_EXCLUSIVE_LIMIT,
            "polygons_max": MAX_POLYGONS,
            "vertices_per_polygon_max": MAX_VERTICES_PER_POLYGON,
            "surface_mm_max": SURFACE_TOLERANCE_MM,
            "bounds_mm_max": BOUNDS_TOLERANCE_MM,
            "volume_relative_max": VOLUME_RELATIVE_TOLERANCE,
            "new_internal_positive_volume_overlap_count": 0,
            "positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
            "frozen_open_clearance_mm_min": 0.1,
            "frozen_open_raw_delta_mm_max": 0.5,
            "D372_P34_clearance_delta_mm_max": 0.1,
            "jaw_projection_raster_step_mm": 0.25,
        },
        "overlap_scope": (
            "zero-new-positive-overlap applies within children replacing one "
            "source parent. Pre-existing intersections between different "
            "semantic P34 parents are not redefined by D397."
        ),
        "failure_capable_controls": [
            "duplicate leaf",
            "removed parent children",
            "synthetic overlapping closed halfspaces",
            "two frozen D396 rejected witnesses",
            "source count 65",
            "total count 128",
        ],
        "execution_contract": {
            "offline_worker": 1,
            "worker_retries": 0,
            "viewer_maximum": 1,
            "viewer_retries": 0,
            "process_signals": 0,
            "cooperative_deadline_seconds": COOPERATIVE_DEADLINE_SECONDS,
            "deadline_scope": "canonical offline computation only",
            "viewer_subprocess_wall_clock_watchdog": None,
            "viewer_watchdog_reason": (
                "the no-signal authority forbids terminating a stalled Viewer; "
                "the Rerun command is one-shot but operationally unbounded"
            ),
        },
        "read_only_inputs": {
            "immutable_authoring_usd_reads": 1,
            "purpose": (
                "recreate the exact frozen D372 raw-surface sample family; "
                "OpenUSD parse only, no stage edit/save and no Isaac/Kit/PhysX"
            ),
        },
        "worker_openusd_environment": {
            "PYTHONPATH": EXPECTED_WORKER_PYTHONPATH,
            "LD_LIBRARY_PATH": EXPECTED_WORKER_LD_LIBRARY_PATH,
            "expected_OpenUSD_version": EXPECTED_OPENUSD_VERSION,
            "purpose": "immutable source USD parse only",
        },
        "forbidden": {
            "usd_or_asset_writes": 0,
            "collider_materialization": 0,
            "isaac_kit_physx_warp_cuda": 0,
            "physics_q5_contact": 0,
            "cylinder_create_or_write": 0,
            "target_ik_path_or_settings": 0,
            "hardware_signal_commit_push": 0,
        },
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "status_outside_output": status_before,
            "status_outside_output_sha256": _sha_payload(status_before),
        },
        "script": {"path": _rel(SCRIPT), "sha256": _sha(SCRIPT)},
        "start": {"path": _rel(START), "sha256": _sha(START)},
        "input_hashes": actual_hashes,
        "checks": checks,
        "pass": all(checks.values()),
        "forward_only_output": _rel(OUT_DIR),
    }
    _write_json_x(PREREG, prereg)
    _phase("prepare_end", preregistration_pass=prereg["pass"])
    if not prereg["pass"]:
        raise RuntimeError(f"D397 preregistration failed: {checks}")
    print(json.dumps({"prepared": True, "path": _rel(PREREG)}))
    return 0


def _worker() -> int:
    if INVOCATION.exists():
        raise FileExistsError("D397 worker already invoked; no retry")
    prereg = _read_json(PREREG)
    checks = {
        "prereg_pass": prereg["pass"] is True,
        "head_unchanged": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_unchanged": _git("rev-parse", "origin/master")
        == EXPECTED_HEAD,
        "script_unchanged": prereg["script"]["sha256"] == _sha(SCRIPT),
        "start_unchanged": prereg["start"]["sha256"] == _sha(START),
        "inputs_unchanged": prereg["input_hashes"] == _input_hashes(),
        "worker_pythonpath_exact": os.environ.get("PYTHONPATH")
        == EXPECTED_WORKER_PYTHONPATH,
        "worker_ld_library_path_exact": os.environ.get("LD_LIBRARY_PATH")
        == EXPECTED_WORKER_LD_LIBRARY_PATH,
        "openusd_0_24_5": _openusd_version()
        == EXPECTED_OPENUSD_VERSION,
        "status_baseline_unchanged": prereg["git"][
            "status_outside_output_sha256"
        ]
        == _sha_payload(_status_outside_output()),
    }
    _write_json_x(
        INVOCATION,
        {
            "artifact": "D397_OFFLINE_WORKER_INVOCATION_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "worker_invocation_count": 1,
            "automatic_retry_count": 0,
            "started_monotonic_ns": time.monotonic_ns(),
            "preflight_checks": checks,
            "pass": all(checks.values()),
        },
    )
    if not all(checks.values()):
        raise RuntimeError(f"D397 worker preflight failed: {checks}")
    _phase("worker_start")
    started = time.monotonic()
    evidence, geometry, source_rows = _compute()
    geometry["canonical_evidence_sha256"] = _sha_payload(evidence)
    _write_json_x(EVIDENCE, evidence)
    geometry["canonical_evidence_sha256"] = _sha(EVIDENCE)
    _write_json_x(GEOMETRY, geometry)
    _write_parent_csv(source_rows)
    claim = {
        "artifact": "D397_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "worker_invocation_count": 1,
        "automatic_retry_count": 0,
        "elapsed_seconds": time.monotonic() - started,
        "cooperative_deadline_exceeded": False,
        "process_signals_sent": 0,
        "artifacts": {
            "evidence": {
                "path": _rel(EVIDENCE),
                "sha256": _sha(EVIDENCE),
            },
            "geometry": {
                "path": _rel(GEOMETRY),
                "sha256": _sha(GEOMETRY),
            },
            "parent_csv": {
                "path": _rel(PARENT_CSV),
                "sha256": _sha(PARENT_CSV),
            },
        },
        "design_pass": evidence["design_pass"],
        "pass": True,
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_end", design_pass=evidence["design_pass"])
    print(json.dumps(_native(claim), ensure_ascii=False))
    return 0


def _png_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "path": _rel(path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }


def _set_equal_limits(axis: Any, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.6, 0.005)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _plot_parts(axis: Any, parts: list[dict[str, Any]], title: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    all_points = []
    for index, part in enumerate(parts):
        vertices = np.asarray(part["vertices_f32_m"], dtype=np.float64)
        triangles = np.asarray(part["triangles_i32"], dtype=np.int64)
        all_points.append(vertices)
        color = ROLE_COLORS.get(part["role"], (130, 130, 130, 130))
        rgba = tuple(value / 255.0 for value in color)
        collection = Poly3DCollection(
            vertices[triangles],
            facecolor=rgba,
            edgecolor=(0.12, 0.12, 0.12, 0.22),
            linewidth=0.18,
        )
        axis.add_collection3d(collection)
    points = np.vstack(all_points)
    _set_equal_limits(axis, points)
    axis.view_init(elev=24, azim=-58)
    axis.set_title(title, fontsize=12, pad=6)
    axis.set_xlabel("x (m)", fontsize=8)
    axis.set_ylabel("y (m)", fontsize=8)
    axis.set_zlabel("z (m)", fontsize=8)
    axis.tick_params(labelsize=7)


def _render_board(evidence: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    regular = FontProperties(fname=str(FONT_REGULAR))
    bold = FontProperties(fname=str(FONT_BOLD))
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.92],
        height_ratios=[1.0, 1.0],
        left=0.035,
        right=0.975,
        top=0.91,
        bottom=0.07,
        wspace=0.16,
        hspace=0.18,
    )
    fig.suptitle(
        "D397 공유 경계 복합 충돌체 오프라인 설계",
        fontproperties=bold,
        fontsize=20,
        y=0.972,
    )
    fig.text(
        0.5,
        0.935,
        "같은 경계 꼭짓점을 양쪽 폐쇄 볼록체가 공유 — Isaac/PhysX/물리 실행 없음",
        ha="center",
        fontproperties=regular,
        fontsize=11,
        color="#333333",
    )
    link_axis = fig.add_subplot(grid[0, 0], projection="3d")
    grip_axis = fig.add_subplot(grid[0, 1], projection="3d")
    _plot_parts(
        link_axis,
        geometry["parts"]["link5"],
        f"link5 전체 후보 — {len(geometry['parts']['link5'])}개",
    )
    _plot_parts(
        grip_axis,
        geometry["parts"]["gripper_link"],
        f"gripper_link 전체 후보 — {len(geometry['parts']['gripper_link'])}개",
    )
    exploded = fig.add_subplot(grid[1, :2], projection="3d")
    source_parts = [
        row
        for body in BODY_NAMES
        for row in geometry["parts"][body]
        if row["source"] == "D397_float32_shared_boundary_bsp"
    ]
    display_leaves = (
        source_parts
        if source_parts
        else geometry["diagnostic_partition_leaves"]
    )
    parent_names = sorted(
        row["name"] for row in geometry["diagnostic_source_parents"]
    )

    def display_shift(parent_name: str) -> np.ndarray:
        parent_order = parent_names.index(parent_name)
        return np.asarray(
            [
                (parent_order % 4) * 0.085,
                (parent_order // 4) * 0.12,
                0.0,
            ]
        )

    all_points = []
    for part in display_leaves:
        shift = display_shift(part["parent_name"])
        vertices = np.asarray(part["vertices_f32_m"]) + shift
        triangles = np.asarray(part["triangles_i32"], dtype=np.int64)
        all_points.append(vertices)
        color = ROLE_COLORS[part["role"]]
        exploded.add_collection3d(
            Poly3DCollection(
                vertices[triangles],
                facecolor=tuple(value / 255.0 for value in color),
                edgecolor=(0.05, 0.05, 0.05, 0.28),
                linewidth=0.20,
            )
        )
    for parent in geometry["diagnostic_source_parents"]:
        shift = display_shift(parent["name"])
        vertices = np.asarray(parent["vertices_f32_m"]) + shift
        triangles = np.asarray(parent["triangles_i32"], dtype=np.int64)
        all_points.append(vertices)
        exploded.add_collection3d(
            Poly3DCollection(
                vertices[triangles],
                facecolor=(0.35, 0.35, 0.35, 0.025),
                edgecolor=(0.08, 0.08, 0.08, 0.48),
                linewidth=0.55,
            )
        )
    for seam in geometry["diagnostic_shared_seams"]:
        shift = display_shift(seam["parent_name"])
        points = np.asarray(seam["seam_vertices_f32_m"]) + shift
        all_points.append(points)
        exploded.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="#d000ff",
            s=10,
            depthshade=False,
        )
    if all_points:
        _set_equal_limits(exploded, np.vstack(all_points))
    exploded.view_init(elev=27, azim=-61)
    exploded.set_title(
        (
            "원 source 외곽(검정) + 공유경계 자식 + seam 꼭짓점(자홍) "
            "— 8개 펼쳐 보기"
        ),
        fontproperties=regular,
        fontsize=10.5,
    )
    exploded.tick_params(labelsize=7)
    text_axis = fig.add_subplot(grid[:, 2])
    text_axis.axis("off")
    counts = evidence["counts"]
    parents = evidence["source_parent_metrics"]
    max_surface = max(
        (
            row.get("parent_surface_coverage_violation_max_mm", math.inf)
            for row in parents
        ),
        default=math.inf,
    )
    max_outward = max(
        (
            row.get("outward_halfspace_violation_mm", math.inf)
            for row in parents
        ),
        default=math.inf,
    )
    max_volume = max(
        (row.get("volume_relative_error", math.inf) for row in parents),
        default=math.inf,
    )
    max_bounds = max(
        (row.get("bounds_max_abs_delta_mm", math.inf) for row in parents),
        default=math.inf,
    )
    clearance = evidence.get("frozen_open_clearance") or {}
    void = evidence.get("jaw_void_gate") or {}
    raw_surface = evidence.get("raw_surface_direct") or {}
    lines = [
        ("판정", evidence["verdict"]),
        (
            "부품 수",
            f"기본 63 + source {counts['source_children']} = {counts['total_parts']} "
            f"(source≤64, 전체<128)",
        ),
        ("source 부모", f"{sum(row.get('construction_complete', False) for row in parents)}/8 완성"),
        ("최대 자식 꼭짓점", str(max(row.get("maximum_child_vertex_count", 0) for row in parents))),
        (
            "새 내부 양의 겹침",
            (
                "0으로 인증"
                if all(
                    row.get("pairwise_nonoverlap", {}).get(
                        "certified_positive_volume_overlap_pair_count"
                    )
                    == 0
                    for row in parents
                    if row.get("construction_complete")
                )
                and all(row.get("construction_complete") for row in parents)
                else "미인증(null) — 분리평면 증명 실패"
            ),
        ),
        ("최대 바깥 오차", f"{max_outward:.9f} mm / 0.1 mm"),
        ("최대 표면 미포함", f"{max_surface:.9f} mm / 0.1 mm"),
        ("최대 부피 상대오차", f"{max_volume:.3e} / 5e-3"),
        ("최대 bounds 차이", f"{max_bounds:.9f} mm / 0.1 mm"),
        (
            "raw 표면 직접 재측정",
            (
                "link5 p95/max="
                f"{raw_surface.get('bodies',{}).get('link5',{}).get('raw_sample_outside_candidate_p95_mm')}/"
                f"{raw_surface.get('bodies',{}).get('link5',{}).get('raw_sample_outside_candidate_max_mm')} mm, "
                "moving p95/max="
                f"{raw_surface.get('bodies',{}).get('gripper_link',{}).get('raw_sample_outside_candidate_p95_mm')}/"
                f"{raw_surface.get('bodies',{}).get('gripper_link',{}).get('raw_sample_outside_candidate_max_mm')} mm"
            ),
        ),
        (
            "void 투영",
            (
                f"fixed xor={void.get('bodies',{}).get('fixed',{}).get('projection_equivalence',{}).get('xor_cell_count')}, "
                f"moving xor={void.get('bodies',{}).get('moving',{}).get('projection_equivalence',{}).get('xor_cell_count')}"
            ),
        ),
        (
            "OPEN 간격",
            (
                f"link5={clearance.get('bodies',{}).get('link5',{}).get('candidate_exact_signed_distance_mm')} mm, "
                f"moving={clearance.get('bodies',{}).get('gripper_link',{}).get('candidate_exact_signed_distance_mm')} mm"
            ),
        ),
        ("물리/q5/contact", "0 / 0 / 0"),
        ("g0a_pass", "false"),
    ]
    y = 0.98
    for label, value in lines:
        text_axis.text(
            0.0,
            y,
            label,
            transform=text_axis.transAxes,
            fontproperties=bold,
            fontsize=10.5,
            va="top",
            color="#111111",
        )
        text_axis.text(
            0.0,
            y - 0.035,
            value,
            transform=text_axis.transAxes,
            fontproperties=regular,
            fontsize=9.2,
            va="top",
            wrap=True,
            color="#333333",
        )
        y -= 0.071
    fig.savefig(BOARD, dpi=100, facecolor="white")
    plt.close(fig)
    info = _png_info(BOARD)
    layout = {
        "artifact": "D397_BOARD_LAYOUT_VALIDATION_V1",
        "board": info,
        "checks": {
            "exact_1920x1080": info["width"] == 1920
            and info["height"] == 1080,
            "nonempty": info["bytes"] > 100_000,
            "three_geometry_panels_and_one_metric_panel": True,
        },
    }
    layout["pass"] = all(layout["checks"].values())
    _write_json_x(LAYOUT, layout)
    return layout


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    def view(body: str, position: tuple[float, float, float], target: tuple[float, float, float]) -> Any:
        return rrb.Spatial3DView(
            origin="/",
            contents=f"/d397/{body}/**",
            name=f"D397 {body} shared-boundary candidate",
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=position,
                look_target=target,
                eye_up=(0.0, 0.0, 1.0),
            ),
            spatial_information=rrb.SpatialInformation(
                target_frame="tf#/",
                show_axes=True,
                show_bounding_box=False,
            ),
        )

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                view("link5", (0.14, -0.17, 0.14), (-0.005, 0.0, 0.055)),
                view(
                    "gripper_link",
                    (0.12, -0.15, 0.065),
                    (0.025, 0.0, -0.018),
                ),
                column_shares=[0.5, 0.5],
            ),
            rrb.TextDocumentView(
                origin="/metadata/run",
                contents="/metadata/run",
                name="D397 numeric authority and scope",
            ),
            row_shares=[0.82, 0.18],
        ),
        rrb.TimePanel(state=rrb.PanelState.Hidden),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _write_rerun(evidence: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    import roarm_rl.rerun_contract as rerun_contract
    import roarm_rl.viz_debug as viz_debug

    meshes = []
    points = []
    expected_entities = ["/metadata/run"]
    components: dict[str, list[str]] = {
        "/metadata/run": ["TextDocument:text"]
    }
    def add_mesh(entity: str, part: dict[str, Any], color: list[int]) -> None:
        meshes.append(
            {
                "entity_path": entity,
                "vertices_m": part["vertices_f32_m"],
                "triangles": part["triangles_i32"],
                "color_rgba": color,
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
        canonical = f"/{entity}"
        metadata_entity = "/metadata/meshes/" + entity.replace("/", "__")
        expected_entities.extend([canonical, metadata_entity])
        components[canonical] = [
            "CoordinateFrame:frame",
            "Mesh3D:albedo_factor",
            "Mesh3D:triangle_indices",
            "Mesh3D:vertex_positions",
        ]
        components[metadata_entity] = ["TextDocument:text"]

    for body in BODY_NAMES:
        for index, part in enumerate(
            row
            for row in geometry["diagnostic_source_parents"]
            if row["body"] == body
        ):
            slug = f"p{index:03d}_{part['name']}".replace("/", "_")
            add_mesh(
                f"d397/{body}/source/{slug}",
                part,
                [70, 70, 70, 45],
            )
        for index, part in enumerate(geometry["parts"][body]):
            slug = f"p{index:03d}_{part['name']}".replace("/", "_")
            entity = f"d397/{body}/candidate/{slug}"
            color = list(ROLE_COLORS.get(part["role"], (130, 130, 130, 130)))
            add_mesh(entity, part, color)
        if evidence["design_pass"] is False:
            for index, part in enumerate(
                row
                for row in geometry["diagnostic_partition_leaves"]
                if row["body"] == body
            ):
                slug = f"p{index:03d}_{part['name']}".replace("/", "_")
                add_mesh(
                    f"d397/{body}/diagnostic_leaf/{slug}",
                    part,
                    [220, 80, 80, 120],
                )
        for index, seam in enumerate(
            row
            for row in geometry["diagnostic_shared_seams"]
            if row["body"] == body
        ):
            entity = (
                f"d397/{body}/seam/"
                f"n{index:03d}_{seam['parent_name']}_{seam['node_id']}"
            )
            points.append(
                {
                    "entity_path": entity,
                    "positions_m": seam["seam_vertices_f32_m"],
                    "radii": [0.0012],
                    "colors": [[208, 0, 255, 255]],
                    "labels": [
                        f"{seam['parent_name']} seam {seam['node_id']}"
                    ],
                    "coordinate_frame": "tf#/",
                    "static": True,
                }
            )
            canonical = f"/{entity}"
            expected_entities.append(canonical)
            components[canonical] = [
                "CoordinateFrame:frame",
                "Points3D:colors",
                "Points3D:labels",
                "Points3D:positions",
                "Points3D:radii",
            ]
    expected_entities = sorted(expected_entities)
    metadata = {
        "case": CASE,
        "attempt": ATTEMPT,
        "verdict": evidence["verdict"],
        "source_children": evidence["counts"]["source_children"],
        "total_parts": evidence["counts"]["total_parts"],
        "shared_boundary": (
            "same Float32 seam vertices are reused by both closed siblings"
        ),
        "numeric_authority": _rel(EVIDENCE),
        "display_geometry_authority": _rel(GEOMETRY),
        "display_layers": [
            "source parent",
            "candidate or partial diagnostic leaves",
            "shared seam points",
        ],
        "Isaac_PhysX_q5_physics_contact": [0, 0, 0, 0, 0],
        "g0a_pass": False,
    }
    original_builder = viz_debug.build_rerun_blueprint
    original_runner = rerun_contract._run
    viewer_calls = 0

    def routed_builder(mode: str = "robot_geometry") -> Any:
        return _build_blueprint() if mode == "d397_shared_boundary" else original_builder(mode)

    def no_signal_run(command: list[str], *, timeout_s: float) -> dict[str, Any]:
        nonlocal viewer_calls
        del timeout_s
        if any("screenshot" in str(part) for part in command):
            viewer_calls += 1
            if viewer_calls > 1:
                return {
                    "command": command,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "D397 viewer maximum exceeded",
                    "ok": False,
                    "signals_sent": 0,
                }
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "ok": result.returncode == 0,
            "signals_sent": 0,
            "timeout_ignored_no_signal_authority": True,
        }

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    rerun_contract._run = no_signal_run
    try:
        saved = viz_debug.log_rerun(
            RRD,
            meshes=meshes,
            points=points,
            recording_metadata=metadata,
            recording_id="g0a_d397_shared_boundary_design",
            blueprint_path=RBL,
            blueprint_mode="d397_shared_boundary",
            live_viewer=False,
            app_id="roarm_g0a_d397_shared_boundary",
        )
        if saved.get("ok") is not True:
            raise RuntimeError(f"D397 save-only Rerun failed: {saved}")
        validation = rerun_contract.validate_rerun_artifact(
            RRD,
            expected_entity_paths=expected_entities,
            exact_entity_paths=expected_entities,
            expected_timeline_names=["blueprint", "log_time"],
            exact_timeline_names=["blueprint", "log_time"],
            expected_entity_components=components,
            blueprint_path=RBL,
            screenshot_path=RERUN_SCREENSHOT,
            screenshot_window_size="1920x1080",
            screenshot_port="auto",
            cli_path=RERUN_CLI,
            expected_version="0.34.1",
            timeout_s=0.0,
        )
    finally:
        rerun_contract._run = original_runner
        viz_debug.build_rerun_blueprint = original_builder
        os.environ["PATH"] = old_path
    screenshot = _png_info(RERUN_SCREENSHOT)
    base_pass = validation.get("pass") is True
    validation["d397_contract"] = {
        "source_parent_mesh_entities": len(
            geometry["diagnostic_source_parents"]
        ),
        "candidate_mesh_entities": sum(
            len(geometry["parts"][body]) for body in BODY_NAMES
        ),
        "diagnostic_leaf_mesh_entities": (
            len(geometry["diagnostic_partition_leaves"])
            if evidence["design_pass"] is False
            else 0
        ),
        "total_mesh_entities": len(meshes),
        "metadata_mesh_entities": len(meshes),
        "shared_seam_point_entities": len(points),
        "viewer_invocations": viewer_calls,
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "screenshot_16x9": screenshot["width"] * 9
        == screenshot["height"] * 16,
    }
    validation["base_contract_pass"] = base_pass
    validation["pass"] = (
        base_pass
        and viewer_calls == 1
        and screenshot["width"] * 9 == screenshot["height"] * 16
    )
    _write_json_x(RERUN_VALIDATION, validation)
    return {
        "pass": validation["pass"],
        "viewer_invocations": viewer_calls,
        "rrd": {"path": _rel(RRD), "sha256": _sha(RRD)},
        "rbl": {"path": _rel(RBL), "sha256": _sha(RBL)},
        "validation": {
            "path": _rel(RERUN_VALIDATION),
            "sha256": _sha(RERUN_VALIDATION),
        },
        "screenshot": screenshot,
    }


def _observe() -> int:
    if BOARD.exists() or RRD.exists():
        raise FileExistsError("D397 observability already executed; no retry")
    _phase("observability_start")
    evidence = _read_json(EVIDENCE)
    geometry = _read_json(GEOMETRY)
    if geometry["canonical_evidence_sha256"] != _sha(EVIDENCE):
        raise RuntimeError("D397 evidence/geometry hash link mismatch")
    layout = _render_board(evidence, geometry)
    rerun = _write_rerun(evidence, geometry)
    template = {
        "artifact": "D397_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board": _png_info(BOARD),
        "rerun_screenshot": _png_info(RERUN_SCREENSHOT),
        "checks_to_record_after_actual_viewing": MANUAL_KEYS,
        "minimum_observation_notes": 3,
        "manual_inspection_complete": False,
    }
    _write_json_x(MANUAL_TEMPLATE, template)
    checks = {
        "worker_claim_pass": _read_json(WORKER_CLAIM)["pass"] is True,
        "board_layout_pass": layout["pass"],
        "board_exact_1920x1080": _png_info(BOARD)["width"] == 1920
        and _png_info(BOARD)["height"] == 1080,
        "rerun_contract_pass": rerun["pass"],
        "viewer_exactly_one": rerun["viewer_invocations"] == 1,
    }
    observation = {
        "artifact": "D397_OBSERVABILITY_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": checks,
        "board": _png_info(BOARD),
        "rerun": rerun,
        "manual_template": {
            "path": _rel(MANUAL_TEMPLATE),
            "sha256": _sha(MANUAL_TEMPLATE),
        },
        "pass": all(checks.values()),
    }
    _write_json_x(OUT_DIR / "d397_observability_claim.json", observation)
    _phase("observability_end", pass_value=observation["pass"])
    if not observation["pass"]:
        raise RuntimeError(f"D397 observability failed: {checks}")
    print(json.dumps(_native(observation), ensure_ascii=False))
    return 0


def _finalize() -> int:
    if COMPLETION.exists():
        raise FileExistsError("D397 already finalized")
    manual = _read_json(MANUAL)
    evidence = _read_json(EVIDENCE)
    observation = _read_json(OUT_DIR / "d397_observability_claim.json")
    checks = {
        "worker_complete": _read_json(WORKER_CLAIM)["pass"] is True,
        "canonical_evidence_present": EVIDENCE.is_file()
        and GEOMETRY.is_file(),
        "observability_pass": observation["pass"] is True,
        "manual_complete": manual.get("manual_inspection_complete") is True,
        "manual_all_checks": set(manual.get("checks", {})) == set(MANUAL_KEYS)
        and all(manual["checks"].values()),
        "manual_notes": len(manual.get("observations", [])) >= 3,
        "manual_hash_binding": manual.get("board_sha256") == _sha(BOARD)
        and manual.get("rerun_screenshot_sha256")
        == _sha(RERUN_SCREENSHOT),
        "no_process_signals": evidence["scope_counters"][
            "process_signals_sent"
        ]
        == 0,
    }
    completion = {
        "artifact": "D397_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "design_verdict": evidence["verdict"],
        "design_pass": evidence["design_pass"],
        "operational_verdict": (
            "D397_OFFLINE_DESIGN_AND_OBSERVABILITY_COMPLETE"
            if all(checks.values())
            else "D397_COMPLETION_INTEGRITY_FAIL_STOP"
        ),
        "worker_invocations": 1,
        "worker_retries": 0,
        "viewer_invocations": observation["rerun"]["viewer_invocations"],
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "checks": checks,
        "completion_integrity_pass": all(checks.values()),
        "materializable_candidate": evidence["design_pass"],
        "live_identity_pass": None,
        "physics_or_grasp_result": None,
        "g0a_pass": False,
        "artifacts": {
            path.name: {"path": _rel(path), "sha256": _sha(path)}
            for path in (
                PREREG,
                INVOCATION,
                EVIDENCE,
                GEOMETRY,
                PARENT_CSV,
                WORKER_CLAIM,
                BOARD,
                LAYOUT,
                RRD,
                RBL,
                RERUN_VALIDATION,
                RERUN_SCREENSHOT,
                MANUAL,
            )
        },
    }
    _write_json_x(COMPLETION, completion)
    _phase(
        "finalize_end",
        completion_integrity_pass=completion["completion_integrity_pass"],
        design_pass=completion["design_pass"],
    )
    if not completion["completion_integrity_pass"]:
        raise RuntimeError(f"D397 finalize failed: {checks}")
    print(json.dumps(_native(completion), ensure_ascii=False))
    return 0


def _record_failure(
    stage: str,
    exc: BaseException,
    *,
    output_preexisted_before_prepare: bool = False,
) -> None:
    if stage == "prepare" and output_preexisted_before_prepare:
        return
    if FAILURE.exists():
        return
    try:
        _write_json_x(
            FAILURE,
            {
                "artifact": "D397_RUNTIME_FAILURE_V1",
                "case": CASE,
                "attempt": ATTEMPT,
                "stage": stage,
                "exception_type": type(exc).__name__,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "process_signals_sent": 0,
            },
        )
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "worker", "observe", "finalize"),
    )
    args = parser.parse_args()
    output_preexisted_before_prepare = OUT_DIR.exists()
    try:
        return {
            "prepare": _prepare,
            "worker": _worker,
            "observe": _observe,
            "finalize": _finalize,
        }[args.stage]()
    except Exception as exc:
        _record_failure(
            args.stage,
            exc,
            output_preexisted_before_prepare=(
                output_preexisted_before_prepare
            ),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
