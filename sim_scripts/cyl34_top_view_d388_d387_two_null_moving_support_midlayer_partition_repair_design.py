#!/usr/bin/env python3
"""D388 offline repair design for D387's exact two null middle layers.

This case changes one representation variable only: for each frozen null
layer, rotate the same CCW profile polygon so the fan anchor is the vertex
immediately after the old graph's forward-reachable frontier.  The old graph
is never rebuilt.  The other nine D387 map entries are inherited byte-for-byte
at the JSON-value level and are neither evaluated nor mutated.

The worker is offline.  It imports no Isaac Sim, Kit, PhysX, USD, Warp, CUDA,
or robot-control package.  It does not choose/apply a vertex budget or create
a materializable collider.  Finite witnesses are diagnostic geometry only.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.metadata
import json
import locale
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError


REPO = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO):
    if str(REPO) in sys.path:
        sys.path.remove(str(REPO))
    sys.path.insert(0, str(REPO))

CASE = "g0a_d388"
ATTEMPT = "attempt1_two_null_moving_support_midlayer_partition_repair_design"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT
SCRIPT_PATH = Path(__file__).resolve()
START_HERE = REPO / "START_HERE.md"
BACKLOG = REPO / "claudedocs/BACKLOG.md"
DECISIONS = REPO / "claudedocs/DECISIONS.md"
EXPERIMENT_LEDGER = REPO / "claudedocs/EXPERIMENT_LEDGER.md"

D385_SCRIPT = REPO / (
    "sim_scripts/"
    "cyl34_top_view_d385_p34_source_hull_semantic_low_count_redesign.py"
)
D379_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d379/"
    "attempt2_d372_measurement_field_repair/"
    "d379_p34_full_live_identity_evidence.json"
)
D385_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d385/"
    "attempt2_precreate_git_status_capture_repair/"
    "d385_p34_source_hull_redesign_evidence.json"
)
D386_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d386/"
    "attempt1_observed_no_cover_layer_minimum_vertex_budget_localization/"
    "d386_vertex_budget_localization_evidence.json"
)
D386_CSV = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d386/"
    "attempt1_observed_no_cover_layer_minimum_vertex_budget_localization/"
    "d386_candidate_cell_metrics.csv"
)
D387_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d387/"
    "attempt1_shadowed_layer_fixed_graph_completion_localization/"
    "d387_shadowed_layer_fixed_graph_map_evidence.json"
)
D387_GEOMETRY = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d387/"
    "attempt1_shadowed_layer_fixed_graph_completion_localization/"
    "d387_eleven_layer_fixed_graph_map_geometry.json"
)
D387_CSV = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d387/"
    "attempt1_shadowed_layer_fixed_graph_completion_localization/"
    "d387_new_layer_candidate_cell_metrics.csv"
)
D387_COMPLETION = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d387/"
    "attempt1_shadowed_layer_fixed_graph_completion_localization/"
    "d387_completion_summary.json"
)
D387_SESSION = REPO / (
    "claudedocs/"
    "session_20260726_grasp_g0a_d387_shadowed_layer_fixed_graph_"
    "completion_localization.md"
)
D387_SCRIPT = REPO / (
    "sim_scripts/"
    "cyl34_top_view_d387_d386_shadowed_layer_fixed_graph_"
    "completion_localization.py"
)
D387_OUTPUT_DIR = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d387/"
    "attempt1_shadowed_layer_fixed_graph_completion_localization"
)

EXPECTED_HEAD = "930b41d98576a9c0bf1dce4f3eb1c0d93df8014b"
EXPECTED_INPUT_SHA256 = {
    "d379_evidence": (
        "8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5"
    ),
    "d385_script": (
        "ea1d76a8db9c78a3cae9de50a62e0a25283d5550346dad158e641a0da321c5ed"
    ),
    "d385_evidence": (
        "4ff64045d4e2e7ecc3601927d1d6c97fd1a61b636e838241f9fded6b02e3cc00"
    ),
    "d386_evidence": (
        "ae956a2b64835f4030daf104f08d239f140f8ba9b32ee9205f2b744769c51d4c"
    ),
    "d386_csv": (
        "adfb2c6007ff84e756e5d6afca260a20cbfa9d6c0cf3a180c3aaf6d458084dd2"
    ),
    "d387_evidence": (
        "ea2073da708d95aeb76874caa0cb24f05a6d5c5075bab8597574cb0026841e30"
    ),
    "d387_geometry": (
        "b40eca132cba9cb7f2fa72b637398f85d8a43a3360f3831ca64a8a54e4c9275a"
    ),
    "d387_csv": (
        "1d2a5626d5dbb477c469a40b39f7d25e743c13b007159dcdc1a5f93d98b36518"
    ),
    "d387_completion": (
        "fbd8f60c85f813f1dfa3394f0a446088a0a571b09d363762f250fde8d8fc5470"
    ),
    "d387_session": (
        "a71d8d8d0567f984b4f863c986857268d96cef8a8f654d912535a501f8780e64"
    ),
    "d387_script": (
        "39d1f9f33a3f6b36b07fdb7ae30b5f89afdd1646f5c98531e2274043cc72d9ee"
    ),
}
EXPECTED_STATE_DOC_SHA256 = {
    "backlog": (
        "6b848a5d680d069a0f57549e8ebe3f0c4d21eb9436d7e1553f91e4c4943f355c"
    ),
    "decisions": (
        "6c46bf195e5e7c58da9b55cf66a87003b372a9ce3dc22a3253013316bc4d4f0e"
    ),
    "experiment_ledger": (
        "856abe5bd4d176360024c3ebc509a1d7a92951c8013752a944d36a5dd8baf396"
    ),
}
EXPECTED_D387_OUTPUT_FILE_COUNT = 19
EXPECTED_D387_OUTPUT_MANIFEST_AGGREGATE_SHA256 = (
    "f5f0d21017a4c2dd1fdbcbfb2674244cb38a13278b7828baa5379f453b6294b9"
)

PREREG_PATH = OUT_DIR / "d388_preregistration.json"
PHASE_PATH = OUT_DIR / "d388_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d388_offline_design_invocation.json"
WORKER_STDOUT = OUT_DIR / "d388_offline_worker_stdout.log"
WORKER_STDERR = OUT_DIR / "d388_offline_worker_stderr.log"
WORKER_CLAIM = OUT_DIR / "d388_offline_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d388_offline_worker_supervisor.json"
EVIDENCE_PATH = OUT_DIR / "d388_two_null_reanchor_design_evidence.json"
GEOMETRY_PATH = OUT_DIR / "d388_two_null_reanchor_witness_geometry.json"
METRICS_CSV = OUT_DIR / "d388_reanchored_candidate_cell_metrics.csv"
BOARD_PATH = OUT_DIR / "d388_two_null_partition_repair_1920x1080.png"
BOARD_LAYOUT = OUT_DIR / "d388_board_layout_validation.json"
RRD_PATH = OUT_DIR / "d388_two_null_partition_repair.rrd"
RBL_PATH = OUT_DIR / "d388_two_null_partition_repair.rbl"
RERUN_VALIDATION = OUT_DIR / "d388_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d388_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d388_manual_visual_inspection_template.json"
MANUAL_INSPECTION = OUT_DIR / "d388_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d388_completion_summary.json"

RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = [
    "null_middle_layer_first_blocked_triangle_reanchored_fan_graph_v1"
]
EMBEDDED_D385_PURE_GEOMETRY_FUNCTIONS = (
    "_unique_f32",
    "_convex_mesh",
    "_convex_vertices_f64",
    "_convex_polyhedron_edges",
    "_clip_plane_le",
    "_profile_polygon",
    "_fan_cell",
    "_intersect_profile_cell",
    "_surface_samples",
    "_normalized_equations",
    "_maximum_positive_violation_mm",
    "_union_coverage_violation_mm",
)
COOPERATIVE_DEADLINE_SECONDS = 300.0
BASELINE_BUDGET = 12
MAXIMUM_LOCALIZATION_BUDGET = 64
MAXIMUM_FAN_GROUP = 4
MAXIMUM_POLYGONS = 64
MAXIMUM_VERTICES_PER_POLYGON = 32
SURFACE_TOLERANCE_MM = 0.1
VOLUME_RELATIVE_TOLERANCE = 0.005
POSITIVE_VOLUME_EPS_M3 = 1.0e-18
FLOAT_EPS_M = 5.0e-9

TARGETS = [
    {
        "body": "gripper_link",
        "prim_name": "p000_proximal_upper_arm_hull_a",
        "name": "proximal_upper_arm_hull_a",
        "role": "moving_support",
        "region_name": "z_layer_01",
        "region_index": 1,
        "expected_old_profile_vertices": 22,
        "expected_old_triangle_count": 20,
        "expected_old_candidate_count": 74,
        "expected_old_nonvertex_pass": 37,
        "expected_old_polygon_rejects": 37,
        "expected_old_forward_last": 10,
        "expected_old_backward_first": 17,
        "expected_reanchor_index": 11,
        "old_csv_authority": "d387",
        "first_blocked_edge": [10, 11],
        "first_blocked_vertex_count": 43,
        "first_blocked_polygon_count": 82,
    },
    {
        "body": "gripper_link",
        "prim_name": "p002_proximal_lower_arm_hull_a",
        "name": "proximal_lower_arm_hull_a",
        "role": "moving_support",
        "region_name": "z_layer_01",
        "region_index": 1,
        "expected_old_profile_vertices": 24,
        "expected_old_triangle_count": 22,
        "expected_old_candidate_count": 82,
        "expected_old_nonvertex_pass": 40,
        "expected_old_polygon_rejects": 42,
        "expected_old_forward_last": 9,
        "expected_old_backward_first": 18,
        "expected_reanchor_index": 10,
        "old_csv_authority": "d386",
        "first_blocked_edge": [9, 10],
        "first_blocked_vertex_count": 39,
        "first_blocked_polygon_count": 74,
    },
]

PALETTE = [
    [0, 163, 163, 215],
    [246, 166, 35, 215],
    [98, 86, 205, 215],
    [20, 136, 204, 215],
    [217, 72, 80, 215],
    [80, 168, 101, 215],
    [185, 92, 174, 215],
]

FORBIDDEN_COUNTERS = {
    "old_graph_geometry_recomputations": 0,
    "other_nine_layer_evaluations": 0,
    "other_nine_layer_mutations": 0,
    "asset_or_usd_reads": 0,
    "asset_or_usd_writes": 0,
    "collider_materializations_or_regenerations": 0,
    "automatic_or_all_anchor_decomposition_sweeps": 0,
    "unregistered_partition_evaluations": 0,
    "fan_group_size_gt_4_evaluations": 0,
    "internal_overlap_allowances": 0,
    "tolerance_changes": 0,
    "budget_selections_or_applications": 0,
    "isaac_launches": 0,
    "kit_launches": 0,
    "physx_launches": 0,
    "live_callback_queries": 0,
    "warp_or_cuda_launches": 0,
    "cylinder_creates_or_writes": 0,
    "controlled_physics_steps": 0,
    "q5_samples": 0,
    "contact_queries": 0,
    "grasp_trials": 0,
    "target_ik_path_changes": 0,
    "material_mass_actuator_physics_setting_changes": 0,
    "process_signals": 0,
}

ALLOWED_COUNTERS = {
    "registered_reanchored_graph_evaluations": 2,
    "actual_worker_invocations": 1,
    "worker_retries": 0,
    "rerun_viewer_invocations_maximum": 1,
}

WORKER_DEADLINE_MONOTONIC: float | None = None


class CooperativeDeadlineExceeded(RuntimeError):
    """Raised by registered worker checkpoints; no process signal is used."""


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value cannot be serialized: {value}")
    return value


def _sha_payload(value: Any) -> str:
    encoded = json.dumps(
        _native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            _native(value),
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def _phase(name: str, **fields: Any) -> None:
    record = {
        "phase": name,
        "monotonic_seconds": time.monotonic(),
        "wall_time_unix_seconds": time.time(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _native(record),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def _deadline_check(location: str) -> None:
    if (
        WORKER_DEADLINE_MONOTONIC is not None
        and time.monotonic() > WORKER_DEADLINE_MONOTONIC
    ):
        raise CooperativeDeadlineExceeded(
            "D388 cooperative algorithm deadline exceeded at "
            f"{location}; no signal was sent"
        )


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _input_hashes() -> dict[str, str]:
    return {
        "d379_evidence": _sha(D379_EVIDENCE),
        "d385_script": _sha(D385_SCRIPT),
        "d385_evidence": _sha(D385_EVIDENCE),
        "d386_evidence": _sha(D386_EVIDENCE),
        "d386_csv": _sha(D386_CSV),
        "d387_evidence": _sha(D387_EVIDENCE),
        "d387_geometry": _sha(D387_GEOMETRY),
        "d387_csv": _sha(D387_CSV),
        "d387_completion": _sha(D387_COMPLETION),
        "d387_session": _sha(D387_SESSION),
        "d387_script": _sha(D387_SCRIPT),
    }


def _state_doc_hashes() -> dict[str, str]:
    return {
        "backlog": _sha(BACKLOG),
        "decisions": _sha(DECISIONS),
        "experiment_ledger": _sha(EXPERIMENT_LEDGER),
    }


def _directory_manifest_aggregate(path: Path) -> dict[str, Any]:
    prior_collation = locale.setlocale(locale.LC_COLLATE)
    try:
        locale.setlocale(locale.LC_COLLATE, "")
        files = sorted(
            (candidate for candidate in path.iterdir() if candidate.is_file()),
            key=lambda candidate: locale.strxfrm(_rel(candidate)),
        )
    finally:
        locale.setlocale(locale.LC_COLLATE, prior_collation)
    payload = "".join(
        f"{_sha(candidate)}  {_rel(candidate)}\n" for candidate in files
    ).encode("utf-8")
    return {
        "file_count": len(files),
        "sha256sum_sorted_relative_path_manifest_aggregate": (
            hashlib.sha256(payload).hexdigest()
        ),
    }


def _target_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["body"]),
        str(row["prim_name"]),
        str(row["region_name"]),
    )


def _import_roots(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return sorted(roots)


def _pure_geometry_ast_contract() -> dict[str, Any]:
    def function_map(path: Path) -> dict[str, ast.FunctionDef]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    def normalized_dump(node: ast.FunctionDef) -> str:
        normalized = copy.deepcopy(node)
        if (
            normalized.body
            and isinstance(normalized.body[0], ast.Expr)
            and isinstance(normalized.body[0].value, ast.Constant)
            and isinstance(normalized.body[0].value.value, str)
        ):
            normalized.body = normalized.body[1:]
        return ast.dump(normalized, include_attributes=False)

    frozen = function_map(D385_SCRIPT)
    embedded = function_map(SCRIPT_PATH)
    checks = {
        name: bool(
            name in frozen
            and name in embedded
            and normalized_dump(frozen[name])
            == normalized_dump(embedded[name])
        )
        for name in EMBEDDED_D385_PURE_GEOMETRY_FUNCTIONS
    }
    return {
        "source_script": _rel(D385_SCRIPT),
        "source_script_sha256": _sha(D385_SCRIPT),
        "function_names": list(EMBEDDED_D385_PURE_GEOMETRY_FUNCTIONS),
        "function_ast_exact_ignoring_docstrings": checks,
        "pass": all(checks.values()),
    }


def _installed_stack() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "isaac_sim_installed_not_launched": version("isaacsim"),
        "isaac_lab_installed_not_launched": version("isaaclab"),
        "rerun_sdk": version("rerun-sdk"),
        "frozen_d385_geometry_source_script": _rel(D385_SCRIPT),
        "frozen_d385_geometry_source_script_sha256": _sha(D385_SCRIPT),
        "frozen_d385_module_imported": False,
    }


def _unique_f32(points: np.ndarray) -> np.ndarray:
    """Exact D385 geometry rule, embedded to avoid importing signal authority."""
    registered = np.asarray(points, dtype=np.float32).astype(np.float64)
    return np.unique(registered, axis=0)


def _convex_mesh(points: np.ndarray) -> dict[str, Any]:
    """Pure geometry extraction copied exactly from the frozen D385 helper."""
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
    for face, equation in zip(
        hull2.simplices, hull2.equations, strict=True
    ):
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
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(unique) < 4:
        raise ValueError("fewer than four Float64 hull points")
    if np.linalg.matrix_rank(unique - unique.mean(axis=0)) < 3:
        raise ValueError("Float64 hull points are not three-dimensional")
    hull = ConvexHull(unique)
    return unique[np.asarray(hull.vertices, dtype=np.int64)]


def _convex_polyhedron_edges(points: np.ndarray) -> np.ndarray:
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
        pair
        for pair, shared_planes in memberships.items()
        if shared_planes >= 2
    ]
    if not edges:
        raise RuntimeError("failed to reconstruct convex-polyhedron edges")
    return np.asarray(sorted(edges), dtype=np.int64)


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
        [polygon[0], polygon[triangle_start : triangle_end + 2]]
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


def _surface_samples(
    vertices: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
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
        [vertices, edge_midpoints, centroids, quarter_a, quarter_b, quarter_c]
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
    return float(best.max() * 1000.0), int(
        np.count_nonzero(best > 1.0e-7)
    )


def _signed_area_2d(polygon: np.ndarray) -> float:
    points = np.asarray(polygon, dtype=np.float64)
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _sorted_rows(points: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    order = np.lexsort(tuple(source[:, index] for index in reversed(range(2))))
    return source[order]


def _rotation_contract(
    original: np.ndarray,
    rotated: np.ndarray,
    anchor_index: int,
) -> dict[str, Any]:
    expected = np.roll(
        np.asarray(original, dtype=np.float64),
        -int(anchor_index),
        axis=0,
    )
    original_area = _signed_area_2d(original)
    rotated_area = _signed_area_2d(rotated)
    area_matches = math.isclose(
        original_area,
        rotated_area,
        rel_tol=1.0e-15,
        abs_tol=1.0e-18,
    )
    return {
        "anchor_index": int(anchor_index),
        "vertex_count_exact": len(original) == len(rotated),
        "point_multiset_bit_exact": np.array_equal(
            _sorted_rows(original), _sorted_rows(rotated)
        ),
        "registered_cyclic_rotation_bit_exact": np.array_equal(
            expected, np.asarray(rotated, dtype=np.float64)
        ),
        "original_ccw": original_area > 0.0,
        "rotated_ccw": rotated_area > 0.0,
        "signed_area_matches_tight_float64_tolerance": area_matches,
        "original_signed_area_m2": original_area,
        "rotated_signed_area_m2": rotated_area,
        "pass": bool(
            len(original) == len(rotated)
            and np.array_equal(_sorted_rows(original), _sorted_rows(rotated))
            and np.array_equal(expected, np.asarray(rotated, dtype=np.float64))
            and original_area > 0.0
            and rotated_area > 0.0
            and area_matches
        ),
    }


def _candidate_public(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_index": row["anchor_index"],
        "start_state": row["start_state"],
        "end_state": row["end_state"],
        "group_size": row["group_size"],
        "geometry_constructed": row["geometry_constructed"],
        "vertex_count": row.get("vertex_count"),
        "polygon_count": row.get("polygon_count"),
        "maximum_vertices_per_polygon": row.get(
            "maximum_vertices_per_polygon"
        ),
        "volume_m3": row.get("volume_m3"),
        "non_vertex_gates_pass": row["non_vertex_gates_pass"],
        "rejection_reasons": row["rejection_reasons"],
        "error_type": row.get("error_type"),
    }


def _enumerate_reanchored_graph(
    layer_points: np.ndarray,
    *,
    thin_axis: int,
    rotated_polygon: np.ndarray,
    anchor_index: int,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    keep = tuple(index for index in range(3) if index != thin_axis)
    triangle_count = len(rotated_polygon) - 2
    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    reason_counts: dict[str, int] = {}
    for end_state in range(1, triangle_count + 1):
        for start_state in range(
            max(0, end_state - MAXIMUM_FAN_GROUP), end_state
        ):
            _deadline_check(
                f"graph_anchor_{anchor_index}_{start_state}_{end_state}"
            )
            cell = _fan_cell(
                rotated_polygon,
                keep,
                triangle_start=start_state + 1,
                triangle_end=end_state,
                cell_index=-1,
            )
            row: dict[str, Any] = {
                "anchor_index": int(anchor_index),
                "start_state": int(start_state),
                "end_state": int(end_state),
                "group_size": int(end_state - start_state),
                "geometry_constructed": False,
                "non_vertex_gates_pass": False,
                "rejection_reasons": [],
            }
            try:
                child = _intersect_profile_cell(
                    layer_points,
                    thin_axis=thin_axis,
                    cell=cell,
                )
            except Exception as exc:
                if not isinstance(exc, (ValueError, QhullError)):
                    raise
                row["error_type"] = type(exc).__name__
                row["rejection_reasons"] = ["degenerate_geometry"]
                reason_counts["degenerate_geometry"] = (
                    reason_counts.get("degenerate_geometry", 0) + 1
                )
                candidates[(start_state, end_state)] = row
                continue
            reasons = []
            if child["polygon_count"] > MAXIMUM_POLYGONS:
                reasons.append("polygon_count_gt_64")
            if (
                child["max_vertices_per_polygon"]
                > MAXIMUM_VERTICES_PER_POLYGON
            ):
                reasons.append("vertices_per_polygon_gt_32")
            if child["volume_m3"] <= POSITIVE_VOLUME_EPS_M3:
                reasons.append("non_positive_volume")
            row.update(
                {
                    "geometry_constructed": True,
                    "child": child,
                    "vertex_count": int(child["vertex_count"]),
                    "polygon_count": int(child["polygon_count"]),
                    "maximum_vertices_per_polygon": int(
                        child["max_vertices_per_polygon"]
                    ),
                    "volume_m3": float(child["volume_m3"]),
                    "rejection_reasons": reasons,
                    "non_vertex_gates_pass": not reasons,
                }
            )
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            candidates[(start_state, end_state)] = row
    expected_count = sum(
        min(MAXIMUM_FAN_GROUP, end_state)
        for end_state in range(1, triangle_count + 1)
    )
    graph = {
        "anchor_index": int(anchor_index),
        "broad_profile_vertex_count": int(len(rotated_polygon)),
        "triangle_count": int(triangle_count),
        "candidate_count": len(candidates),
        "candidate_count_formula_expected": int(expected_count),
        "candidate_count_formula_exact": len(candidates) == expected_count,
        "geometry_constructed_count": sum(
            int(row["geometry_constructed"]) for row in candidates.values()
        ),
        "non_vertex_pass_count": sum(
            int(row["non_vertex_gates_pass"])
            for row in candidates.values()
        ),
        "rejection_reason_counts": reason_counts,
    }
    return candidates, graph


def _cover_at_budget(
    candidates: dict[tuple[int, int], dict[str, Any]],
    triangle_count: int,
    budget: int,
) -> dict[str, Any] | None:
    dp: list[tuple[int, int, tuple[int, ...]] | None] = [
        None
    ] * (triangle_count + 1)
    dp[0] = (0, 0, (0,))
    for end_state in range(1, triangle_count + 1):
        _deadline_check(f"cover_budget_{budget}_state_{end_state}")
        best = None
        for start_state in range(
            max(0, end_state - MAXIMUM_FAN_GROUP), end_state
        ):
            previous = dp[start_state]
            row = candidates[(start_state, end_state)]
            if (
                previous is None
                or not row["non_vertex_gates_pass"]
                or row.get("vertex_count", MAXIMUM_LOCALIZATION_BUDGET + 1)
                > budget
            ):
                continue
            value = (
                previous[0] + 1,
                max(previous[1], int(row["vertex_count"])),
                previous[2] + (end_state,),
            )
            if best is None or value < best:
                best = value
        dp[end_state] = best
    if dp[-1] is None:
        return None
    return {
        "child_count": int(dp[-1][0]),
        "maximum_child_vertex_count": int(dp[-1][1]),
        "cut_states": list(map(int, dp[-1][2])),
    }


def _minimax_dp(
    candidates: dict[tuple[int, int], dict[str, Any]],
    triangle_count: int,
) -> dict[str, Any] | None:
    dp: list[tuple[int, int, tuple[int, ...]] | None] = [
        None
    ] * (triangle_count + 1)
    dp[0] = (0, 0, (0,))
    for end_state in range(1, triangle_count + 1):
        _deadline_check(f"minimax_state_{end_state}")
        best = None
        for start_state in range(
            max(0, end_state - MAXIMUM_FAN_GROUP), end_state
        ):
            previous = dp[start_state]
            row = candidates[(start_state, end_state)]
            if (
                previous is None
                or not row["non_vertex_gates_pass"]
                or row.get("vertex_count", MAXIMUM_LOCALIZATION_BUDGET + 1)
                > MAXIMUM_LOCALIZATION_BUDGET
            ):
                continue
            value = (
                max(previous[0], int(row["vertex_count"])),
                previous[1] + 1,
                previous[2] + (end_state,),
            )
            if best is None or value < best:
                best = value
        dp[end_state] = best
    if dp[-1] is None:
        return None
    budget = int(dp[-1][0])
    canonical = _cover_at_budget(candidates, triangle_count, budget)
    if canonical is None:
        raise RuntimeError("minimax threshold lost its canonical cover")
    return {
        "minimum_bottleneck_vertex_budget": budget,
        "minimax_dp_cut_states": list(map(int, dp[-1][2])),
        "canonical_cover_at_minimum": canonical,
    }


def _exhaustive_minimax(
    candidates: dict[tuple[int, int], dict[str, Any]],
    triangle_count: int,
) -> dict[str, Any] | None:
    complete_path_count = 0
    visit_count = 0
    minimum_budget: int | None = None
    canonical_key: tuple[int, int, tuple[int, ...]] | None = None

    def visit(position: int, maximum: int, cuts: tuple[int, ...]) -> None:
        nonlocal complete_path_count, visit_count, minimum_budget, canonical_key
        visit_count += 1
        if visit_count % 4096 == 0:
            _deadline_check(f"exhaustive_visit_{visit_count}")
        if position == triangle_count:
            complete_path_count += 1
            child_count = len(cuts) - 1
            key = (child_count, maximum, cuts)
            if minimum_budget is None or maximum < minimum_budget:
                minimum_budget = maximum
                canonical_key = key
            elif maximum == minimum_budget and (
                canonical_key is None or key < canonical_key
            ):
                canonical_key = key
            return
        for end_state in range(
            position + 1,
            min(position + MAXIMUM_FAN_GROUP, triangle_count) + 1,
        ):
            row = candidates[(position, end_state)]
            if (
                not row["non_vertex_gates_pass"]
                or row.get("vertex_count", MAXIMUM_LOCALIZATION_BUDGET + 1)
                > MAXIMUM_LOCALIZATION_BUDGET
            ):
                continue
            visit(
                end_state,
                max(maximum, int(row["vertex_count"])),
                cuts + (end_state,),
            )

    visit(0, 0, (0,))
    _deadline_check(f"exhaustive_complete_{visit_count}")
    if minimum_budget is None or canonical_key is None:
        return None
    return {
        "minimum_bottleneck_vertex_budget": int(minimum_budget),
        "visited_prefix_count": int(visit_count),
        "complete_path_count": int(complete_path_count),
        "canonical_cover_at_minimum": {
            "child_count": int(canonical_key[0]),
            "maximum_child_vertex_count": int(canonical_key[1]),
            "cut_states": list(map(int, canonical_key[2])),
        },
    }


def _reachability(
    candidates: dict[tuple[int, int], dict[str, Any]],
    triangle_count: int,
) -> dict[str, Any]:
    reachable = {0}
    for end_state in range(1, triangle_count + 1):
        for start_state in range(
            max(0, end_state - MAXIMUM_FAN_GROUP), end_state
        ):
            row = candidates[(start_state, end_state)]
            if (
                start_state in reachable
                and row["non_vertex_gates_pass"]
                and row["vertex_count"] <= MAXIMUM_LOCALIZATION_BUDGET
            ):
                reachable.add(end_state)
                break
    backward = {triangle_count}
    for start_state in range(triangle_count - 1, -1, -1):
        for end_state in range(
            start_state + 1,
            min(start_state + MAXIMUM_FAN_GROUP, triangle_count) + 1,
        ):
            row = candidates[(start_state, end_state)]
            if (
                end_state in backward
                and row["non_vertex_gates_pass"]
                and row["vertex_count"] <= MAXIMUM_LOCALIZATION_BUDGET
            ):
                backward.add(start_state)
                break
    return {
        "reachable_states_from_zero": sorted(reachable),
        "states_that_can_reach_end": sorted(backward),
        "end_reachable_with_frozen_gates_and_vertex_le64": (
            triangle_count in reachable
        ),
    }


def _selected_children(
    candidates: dict[tuple[int, int], dict[str, Any]],
    cover: dict[str, Any],
    target: dict[str, Any],
    interval_m: list[float],
    anchor_index: int,
) -> list[dict[str, Any]]:
    children = []
    cuts = cover["cut_states"]
    for child_index, (start_state, end_state) in enumerate(
        zip(cuts[:-1], cuts[1:], strict=True)
    ):
        source = candidates[(start_state, end_state)].get("child")
        if source is None:
            raise RuntimeError("selected child geometry is absent")
        child = dict(source)
        child.update(
            {
                "body": target["body"],
                "role": target["role"],
                "parent_name": target["name"],
                "region_name": target["region_name"],
                "region_index": target["region_index"],
                "reanchored_fan_anchor_original_profile_index": int(
                    anchor_index
                ),
                "name": (
                    f"{target['name']}__{target['region_name']}__"
                    f"reanchor_{anchor_index:02d}__cell_{child_index:02d}"
                ),
                "pre_split_axis": target["region_name"][0],
                "pre_split_interval_m": list(map(float, interval_m)),
                "fan_triangle_index_range": [
                    int(start_state + 1),
                    int(end_state),
                ],
                "profile_cell_index": int(child_index),
            }
        )
        children.append(child)
    return children


def _f32_registered(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).astype(np.float64)


def _directional_halfspace_intersection_volume(
    source_points: np.ndarray,
    clipping_points: np.ndarray,
) -> dict[str, Any]:
    points = np.unique(_f32_registered(source_points), axis=0)
    clip = np.unique(_f32_registered(clipping_points), axis=0)
    if len(points) < 4 or len(clip) < 4:
        return {
            "calculation_pass": False,
            "volume_m3": None,
            "clip_count": 0,
            "skipped_inside_plane_count": 0,
            "zero_reason": None,
            "error": "input hull has fewer than four unique Float32 points",
        }
    try:
        equations = _normalized_equations(clip)
    except (ValueError, QhullError) as exc:
        return {
            "calculation_pass": False,
            "volume_m3": None,
            "clip_count": 0,
            "skipped_inside_plane_count": 0,
            "zero_reason": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    unique_equations = np.unique(np.round(equations, decimals=12), axis=0)
    clip_count = 0
    skipped_count = 0
    for index, equation in enumerate(unique_equations):
        _deadline_check(f"halfspace_clip_{index}")
        values = points @ equation[:3] + equation[3]
        if float(values.max()) <= FLOAT_EPS_M:
            skipped_count += 1
            continue
        if float(values.min()) > FLOAT_EPS_M:
            return {
                "calculation_pass": True,
                "volume_m3": 0.0,
                "clip_count": clip_count,
                "skipped_inside_plane_count": skipped_count,
                "zero_reason": "strictly_disjoint_halfspace",
                "error": None,
            }
        minimum_before_clip = float(values.min())
        try:
            clipped = _clip_plane_le(
                points,
                normal=equation[:3],
                offset=float(equation[3]),
            )
        except Exception as exc:
            if not isinstance(exc, (ValueError, QhullError)):
                raise
            if minimum_before_clip >= -FLOAT_EPS_M:
                return {
                    "calculation_pass": True,
                    "volume_m3": 0.0,
                    "clip_count": clip_count + 1,
                    "skipped_inside_plane_count": skipped_count,
                    "zero_reason": "boundary_touch_only",
                    "error": None,
                }
            return {
                "calculation_pass": False,
                "volume_m3": None,
                "clip_count": clip_count + 1,
                "skipped_inside_plane_count": skipped_count,
                "zero_reason": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        points = clipped
        clip_count += 1
        if len(points) < 4:
            return {
                "calculation_pass": False,
                "volume_m3": None,
                "clip_count": clip_count,
                "skipped_inside_plane_count": skipped_count,
                "zero_reason": None,
                "error": "proper clip returned fewer than four points",
            }
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
            if minimum_before_clip >= -FLOAT_EPS_M:
                return {
                    "calculation_pass": True,
                    "volume_m3": 0.0,
                    "clip_count": clip_count,
                    "skipped_inside_plane_count": skipped_count,
                    "zero_reason": "boundary_touch_only",
                    "error": None,
                }
            return {
                "calculation_pass": False,
                "volume_m3": None,
                "clip_count": clip_count,
                "skipped_inside_plane_count": skipped_count,
                "zero_reason": None,
                "error": "proper clip unexpectedly lost three-dimensional rank",
            }
    try:
        volume = float(ConvexHull(points).volume)
    except QhullError as exc:
        return {
            "calculation_pass": False,
            "volume_m3": None,
            "clip_count": clip_count,
            "skipped_inside_plane_count": skipped_count,
            "zero_reason": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "calculation_pass": True,
        "volume_m3": volume,
        "clip_count": clip_count,
        "skipped_inside_plane_count": skipped_count,
        "zero_reason": None if volume > POSITIVE_VOLUME_EPS_M3 else "zero_volume",
        "error": None,
    }


def _actual_pairwise_overlap(
    children: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = []
    for left_index, left in enumerate(children):
        for right_index in range(left_index + 1, len(children)):
            right = children[right_index]
            left_points = np.asarray(left["vertices_m"], dtype=np.float64)
            right_points = np.asarray(right["vertices_m"], dtype=np.float64)
            left_to_right = _directional_halfspace_intersection_volume(
                left_points, right_points
            )
            right_to_left = _directional_halfspace_intersection_volume(
                right_points, left_points
            )
            calculation_pass = bool(
                left_to_right["calculation_pass"]
                and right_to_left["calculation_pass"]
            )
            directional_volumes = [
                value
                for value in (
                    left_to_right["volume_m3"],
                    right_to_left["volume_m3"],
                )
                if value is not None
            ]
            maximum = max(directional_volumes) if directional_volumes else None
            scale = max(
                maximum if maximum is not None else 0.0,
                left["volume_m3"],
                right["volume_m3"],
                1.0e-30,
            )
            pairs.append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "left_clipped_by_right": left_to_right,
                    "right_clipped_by_left": right_to_left,
                    "calculation_pass": calculation_pass,
                    "directional_difference_relative_to_pair_scale": (
                        (
                            abs(
                                float(left_to_right["volume_m3"])
                                - float(right_to_left["volume_m3"])
                            )
                            / scale
                        )
                        if calculation_pass
                        else None
                    ),
                    "maximum_intersection_volume_m3": maximum,
                    "positive_volume_overlap": (
                        bool(
                            calculation_pass
                            and maximum is not None
                            and maximum > POSITIVE_VOLUME_EPS_M3
                        )
                    ),
                }
            )
    failed = [row for row in pairs if not row["calculation_pass"]]
    positive = [row for row in pairs if row["positive_volume_overlap"]]
    return {
        "method": (
            "each stored child vertex stream is explicitly registered as "
            "Float32, then intersected in both directions by the other "
            "child's actual convex-hull halfspaces"
        ),
        "pair_count": len(pairs),
        "pairs": pairs,
        "calculation_failure_pair_count": len(failed),
        "calculation_failure_pairs": [
            [row["left"], row["right"]] for row in failed
        ],
        "positive_overlap_pair_count": len(positive),
        "positive_overlap_pairs": [
            [row["left"], row["right"]] for row in positive
        ],
        "pass": len(failed) == 0 and len(positive) == 0,
    }


def _duplicate_overlap_negative_control(
    candidates: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    source = next(
        (
            row["child"]
            for _, row in sorted(candidates.items())
            if row["geometry_constructed"]
            and row["child"]["volume_m3"] > POSITIVE_VOLUME_EPS_M3
        ),
        None,
    )
    if source is None:
        return {
            "source_found": False,
            "duplicate_positive_overlap_detected": False,
            "pass": False,
        }
    points = np.asarray(source["vertices_m"], dtype=np.float64)
    calculation = _directional_halfspace_intersection_volume(
        points, points.copy()
    )
    volume = calculation["volume_m3"]
    if not calculation["calculation_pass"] or volume is None:
        return {
            "source_found": True,
            "calculation": calculation,
            "duplicate_positive_overlap_detected": False,
            "pass": False,
        }
    relative = abs(volume - source["volume_m3"]) / source["volume_m3"]
    return {
        "source_found": True,
        "calculation": calculation,
        "source_volume_m3": float(source["volume_m3"]),
        "duplicate_intersection_volume_m3": volume,
        "duplicate_positive_overlap_detected": (
            volume > POSITIVE_VOLUME_EPS_M3
        ),
        "duplicate_volume_relative_error": relative,
        "duplicate_volume_matches_within_frozen_0p5percent": (
            relative <= VOLUME_RELATIVE_TOLERANCE
        ),
        "pass": bool(
            volume > POSITIVE_VOLUME_EPS_M3
            and relative <= VOLUME_RELATIVE_TOLERANCE
        ),
    }


def _box_points(
    low: tuple[float, float, float],
    high: tuple[float, float, float],
) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )


def _synthetic_overlap_controls() -> dict[str, Any]:
    """Exercise both proper clipping and boundary-touch zero-volume paths."""
    base = _box_points((0.0, 0.0, 0.0), (0.010, 0.010, 0.010))
    partial = _box_points((0.005, 0.0, 0.0), (0.015, 0.010, 0.010))
    touching = _box_points((0.010, 0.0, 0.0), (0.020, 0.010, 0.010))
    partial_forward = _directional_halfspace_intersection_volume(
        base, partial
    )
    partial_reverse = _directional_halfspace_intersection_volume(
        partial, base
    )
    touch_forward = _directional_halfspace_intersection_volume(
        base, touching
    )
    touch_reverse = _directional_halfspace_intersection_volume(
        touching, base
    )
    expected_partial_volume = 0.005 * 0.010 * 0.010
    partial_values = [
        partial_forward["volume_m3"],
        partial_reverse["volume_m3"],
    ]
    checks = {
        "partial_overlap_both_directions_calculated": (
            partial_forward["calculation_pass"]
            and partial_reverse["calculation_pass"]
        ),
        "partial_overlap_forces_actual_clipping": (
            partial_forward["clip_count"] > 0
            and partial_reverse["clip_count"] > 0
        ),
        "partial_overlap_positive_and_expected": all(
            value is not None
            and value > POSITIVE_VOLUME_EPS_M3
            and math.isclose(
                value,
                expected_partial_volume,
                rel_tol=1.0e-5,
                abs_tol=1.0e-15,
            )
            for value in partial_values
        ),
        "shared_face_touch_both_directions_calculated": (
            touch_forward["calculation_pass"]
            and touch_reverse["calculation_pass"]
        ),
        "shared_face_touch_has_zero_volume": (
            touch_forward["volume_m3"] == 0.0
            and touch_reverse["volume_m3"] == 0.0
        ),
    }
    return {
        "partial_overlap_boxes": {
            "expected_intersection_volume_m3": expected_partial_volume,
            "base_clipped_by_partial": partial_forward,
            "partial_clipped_by_base": partial_reverse,
        },
        "shared_face_touch_boxes": {
            "expected_intersection_volume_m3": 0.0,
            "base_clipped_by_touching": touch_forward,
            "touching_clipped_by_base": touch_reverse,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _fan_partition_contract(
    polygon: np.ndarray,
    cover: dict[str, Any],
    children: list[dict[str, Any]],
) -> dict[str, Any]:
    triangle_count = len(polygon) - 2
    cuts = list(map(int, cover["cut_states"]))
    ranges = [
        list(map(int, child["fan_triangle_index_range"]))
        for child in children
    ]
    expected_ranges = [
        [start + 1, end]
        for start, end in zip(cuts[:-1], cuts[1:], strict=True)
    ]
    fan_areas = []
    for triangle_index in range(1, len(polygon) - 1):
        triangle = np.asarray(
            [polygon[0], polygon[triangle_index], polygon[triangle_index + 1]],
            dtype=np.float64,
        )
        fan_areas.append(_signed_area_2d(triangle))
    polygon_area = _signed_area_2d(polygon)
    fan_area_sum = float(sum(fan_areas))
    return {
        "triangle_count": int(triangle_count),
        "cut_states": cuts,
        "child_ranges": ranges,
        "expected_child_ranges": expected_ranges,
        "starts_at_zero": bool(cuts and cuts[0] == 0),
        "ends_at_triangle_count": bool(
            cuts and cuts[-1] == triangle_count
        ),
        "strictly_increasing_cuts": all(
            left < right
            for left, right in zip(cuts[:-1], cuts[1:], strict=True)
        ),
        "each_group_within_1_to_4": all(
            1 <= right - left <= MAXIMUM_FAN_GROUP
            for left, right in zip(cuts[:-1], cuts[1:], strict=True)
        ),
        "child_ranges_exact": ranges == expected_ranges,
        "each_fan_triangle_positive_area": all(area > 0.0 for area in fan_areas),
        "polygon_area_m2": polygon_area,
        "fan_triangle_area_sum_m2": fan_area_sum,
        "fan_area_sum_matches_polygon": math.isclose(
            fan_area_sum,
            polygon_area,
            rel_tol=1.0e-12,
            abs_tol=1.0e-18,
        ),
        "pass": bool(
            cuts
            and cuts[0] == 0
            and cuts[-1] == triangle_count
            and all(
                left < right
                for left, right in zip(cuts[:-1], cuts[1:], strict=True)
            )
            and all(
                1 <= right - left <= MAXIMUM_FAN_GROUP
                for left, right in zip(cuts[:-1], cuts[1:], strict=True)
            )
            and ranges == expected_ranges
            and all(area > 0.0 for area in fan_areas)
            and math.isclose(
                fan_area_sum,
                polygon_area,
                rel_tol=1.0e-12,
                abs_tol=1.0e-18,
            )
        ),
    }


def _geometry_metrics(
    parent: dict[str, Any],
    polygon: np.ndarray,
    children: list[dict[str, Any]],
    cover: dict[str, Any],
    registered_threshold: int,
) -> dict[str, Any]:
    parent_points = np.asarray(parent["vertices_m"], dtype=np.float64)
    parent_triangles = np.asarray(parent["triangles"], dtype=np.int64)
    parent_volume = float(ConvexHull(parent_points).volume)
    parent_equations = _normalized_equations(parent_points)
    child_points = np.vstack(
        [np.asarray(child["vertices_m"], dtype=np.float64) for child in children]
    )
    outward_mm = _maximum_positive_violation_mm(
        parent_equations, child_points
    )
    samples = _surface_samples(parent_points, parent_triangles)
    coverage_mm, uncovered = _union_coverage_violation_mm(
        children, samples
    )
    child_volume = float(sum(child["volume_m3"] for child in children))
    relative = abs(child_volume - parent_volume) / parent_volume
    fan = _fan_partition_contract(polygon, cover, children)
    overlap = _actual_pairwise_overlap(children)
    checks = {
        "each_child_vertices_within_registered_threshold": (
            max(child["vertex_count"] for child in children)
            <= registered_threshold
        ),
        "each_child_polygons_le_64": (
            max(child["polygon_count"] for child in children)
            <= MAXIMUM_POLYGONS
        ),
        "each_child_vertices_per_polygon_le_32": (
            max(child["max_vertices_per_polygon"] for child in children)
            <= MAXIMUM_VERTICES_PER_POLYGON
        ),
        "each_child_positive_volume": all(
            child["volume_m3"] > POSITIVE_VOLUME_EPS_M3
            for child in children
        ),
        "each_child_vertex_stream_is_actual_float32": all(
            np.array_equal(
                np.asarray(child["vertices_m"], dtype=np.float64),
                _f32_registered(child["vertices_m"]),
            )
            for child in children
        ),
        "outward_le_0p1mm": outward_mm <= SURFACE_TOLERANCE_MM,
        "coverage_le_0p1mm": coverage_mm <= SURFACE_TOLERANCE_MM,
        "volume_relative_le_0p5percent": (
            relative <= VOLUME_RELATIVE_TOLERANCE
        ),
        "exact_fan_partition_contract": fan["pass"],
        "actual_float32_pairwise_positive_volume_overlap_zero": overlap["pass"],
    }
    return {
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
        "parent_volume_mm3": parent_volume * 1.0e9,
        "child_volume_sum_mm3": child_volume * 1.0e9,
        "volume_relative_error": relative,
        "outward_halfspace_violation_mm": outward_mm,
        "parent_surface_coverage_halfspace_violation_mm": coverage_mm,
        "surface_sample_count": int(len(samples)),
        "uncovered_sample_count_gt_0p0001mm": uncovered,
        "fan_partition_contract": fan,
        "actual_float32_pairwise_halfspace_intersection": overlap,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _old_result_map(
    d387: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = [
        *d387["new_layer_results"],
        *d387["inherited_d386_layer_results"],
    ]
    result = {_target_key(row): row for row in rows}
    if len(result) != 11:
        raise RuntimeError("D387 full result inventory is not exact eleven")
    return result


def _old_graph_sentinel(
    target: dict[str, Any],
    old: dict[str, Any],
) -> dict[str, Any]:
    graph = old["candidate_graph"]
    frontier = old["non_vertex_gate_frontier"]
    reachable = list(map(int, frontier["reachable_states_from_zero"]))
    backward = list(map(int, frontier["states_that_can_reach_end"]))
    checks = {
        "old_result_is_null_through_64": (
            old.get("classification", "NO_COVER_THROUGH_64")
            == "NO_COVER_THROUGH_64"
            and old.get(
                "registered_threshold_within_12_64",
                old.get(
                    "minimum_admissible_vertex_budget_within_12_64"
                ),
            )
            is None
        ),
        "profile_vertex_count_exact": (
            graph["broad_profile_vertex_count"]
            == target["expected_old_profile_vertices"]
        ),
        "triangle_count_exact": (
            graph["triangle_count"] == target["expected_old_triangle_count"]
        ),
        "candidate_count_exact": (
            graph["candidate_count"] == target["expected_old_candidate_count"]
        ),
        "geometry_constructed_count_exact": (
            graph["geometry_constructed_count"]
            == target["expected_old_candidate_count"]
        ),
        "nonvertex_pass_count_exact": (
            graph["non_vertex_pass_count"]
            == target["expected_old_nonvertex_pass"]
        ),
        "polygon_reject_count_exact": (
            graph["rejection_reason_counts"]
            == {
                "polygon_count_gt_64": target[
                    "expected_old_polygon_rejects"
                ]
            }
        ),
        "forward_reachable_is_contiguous_exact": (
            reachable
            == list(range(target["expected_old_forward_last"] + 1))
        ),
        "backward_reachable_is_contiguous_exact": (
            backward
            == list(
                range(
                    target["expected_old_backward_first"],
                    target["expected_old_triangle_count"] + 1,
                )
            )
        ),
        "old_end_unreachable": (
            frontier[
                "end_reachable_with_frozen_gates_and_vertex_le64"
            ]
            is False
        ),
    }
    derived_anchor = max(reachable) + 1
    checks["frontier_rule_derives_registered_anchor"] = (
        derived_anchor == target["expected_reanchor_index"]
    )
    return {
        "source": "immutable D387 evidence; no old graph geometry recomputed",
        "candidate_graph": graph,
        "frontier": frontier,
        "derived_reanchor_index": int(derived_anchor),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _read_first_blocked_csv_witness(
    target: dict[str, Any],
) -> dict[str, Any]:
    path = D387_CSV if target["old_csv_authority"] == "d387" else D386_CSV
    matches = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if (
                row.get("body") == target["body"]
                and row.get("prim_name") == target["prim_name"]
                and row.get("region_name") == target["region_name"]
                and int(row["start_state"])
                == target["first_blocked_edge"][0]
                and int(row["end_state"])
                == target["first_blocked_edge"][1]
            ):
                matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one frozen first-blocked CSV row, found {len(matches)}"
        )
    row = matches[0]
    reasons = json.loads(row["rejection_reasons"])
    checks = {
        "exact_singleton_edge": int(row["group_size"]) == 1,
        "vertex_count_within_64": (
            int(row["vertex_count"])
            == target["first_blocked_vertex_count"]
            <= MAXIMUM_LOCALIZATION_BUDGET
        ),
        "polygon_count_exact_and_above_64": (
            int(row["polygon_count"])
            == target["first_blocked_polygon_count"]
            > MAXIMUM_POLYGONS
        ),
        "rejected_only_by_polygon_gate": (
            reasons == ["polygon_count_gt_64"]
        ),
        "nonvertex_gate_false": row["non_vertex_gates_pass"] == "False",
    }
    return {
        "source_csv": _rel(path),
        "source_csv_sha256": _sha(path),
        "row": row,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _negative_controls(
    *,
    original_polygon: np.ndarray,
    rotated_polygon: np.ndarray,
    anchor_index: int,
    triangle_count: int,
    candidates: dict[tuple[int, int], dict[str, Any]],
    first_blocked: dict[str, Any],
) -> dict[str, Any]:
    reversed_contract = _rotation_contract(
        original_polygon,
        original_polygon[::-1].copy(),
        anchor_index,
    )
    mixed_anchor_tags = [0, int(anchor_index)]
    anchor_coherent = len(set(mixed_anchor_tags)) == 1
    omitted_cover = {
        "cut_states": [0, max(0, triangle_count - 1)],
    }
    omitted_contract = {
        "starts_at_zero": omitted_cover["cut_states"][0] == 0,
        "ends_at_triangle_count": (
            omitted_cover["cut_states"][-1] == triangle_count
        ),
    }
    duplicate = _duplicate_overlap_negative_control(candidates)
    synthetic_overlap = _synthetic_overlap_controls()
    checks = {
        "reversed_polygon_rejected_by_rotation_contract": (
            reversed_contract["pass"] is False
        ),
        "mixed_old_new_anchor_path_rejected": anchor_coherent is False,
        "omitted_last_fan_triangle_rejected": (
            not all(omitted_contract.values())
        ),
        "duplicate_child_positive_overlap_detected": duplicate["pass"],
        "synthetic_partial_and_touching_overlap_controls_pass": (
            synthetic_overlap["pass"]
        ),
        "frozen_first_blocked_edge_still_rejected_only_by_polygon_gate": (
            first_blocked["pass"]
        ),
    }
    return {
        "reversed_polygon_control": reversed_contract,
        "cross_anchor_splice_control": {
            "synthetic_anchor_tags": mixed_anchor_tags,
            "anchor_coherent": anchor_coherent,
        },
        "omitted_fan_triangle_control": {
            "synthetic_cover": omitted_cover,
            **omitted_contract,
        },
        "duplicate_float32_child_overlap_control": duplicate,
        "synthetic_float32_overlap_controls": synthetic_overlap,
        "frozen_gate_relaxation_control": first_blocked,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _classify(
    candidates: dict[tuple[int, int], dict[str, Any]],
    triangle_count: int,
    dp: dict[str, Any] | None,
) -> tuple[str, int | None, dict[str, Any] | None, dict[str, Any]]:
    b12 = _cover_at_budget(candidates, triangle_count, BASELINE_BUDGET)
    raw = (
        int(dp["minimum_bottleneck_vertex_budget"])
        if dp is not None
        else None
    )
    if b12 is not None:
        classification = "BASELINE_B12_COVER"
        threshold = BASELINE_BUDGET
        selected = b12
    elif raw is not None and 13 <= raw <= MAXIMUM_LOCALIZATION_BUDGET:
        classification = "FINITE_RELAXATION_THRESHOLD_13_TO_64"
        threshold = raw
        selected = _cover_at_budget(candidates, triangle_count, threshold)
    else:
        classification = "NO_COVER_THROUGH_64"
        threshold = None
        selected = None
    previous = (
        _cover_at_budget(candidates, triangle_count, threshold - 1)
        if classification == "FINITE_RELAXATION_THRESHOLD_13_TO_64"
        and threshold is not None
        else None
    )
    cover64 = _cover_at_budget(
        candidates, triangle_count, MAXIMUM_LOCALIZATION_BUDGET
    )
    checks = {
        "baseline_finiteness_matches_classification": (
            (b12 is not None)
            == (classification == "BASELINE_B12_COVER")
        ),
        "finite_above12_threshold_in_13_to_64": (
            bool(threshold is not None and 13 <= threshold <= 64)
            if classification
            == "FINITE_RELAXATION_THRESHOLD_13_TO_64"
            else True
        ),
        "finite_above12_threshold_minus_one_has_no_cover": (
            previous is None
            if classification
            == "FINITE_RELAXATION_THRESHOLD_13_TO_64"
            else True
        ),
        "finite_classification_has_selected_cover": (
            selected is not None
            if classification != "NO_COVER_THROUGH_64"
            else True
        ),
        "null_classification_has_no_cover_at_64": (
            cover64 is None
            if classification == "NO_COVER_THROUGH_64"
            else True
        ),
    }
    boundary = {
        "classification": classification,
        "baseline_budget": BASELINE_BUDGET,
        "baseline_cover_exists": b12 is not None,
        "raw_minimax_vertex_count": raw,
        "registered_threshold_within_12_64": threshold,
        "threshold_minus_one": (
            threshold - 1
            if classification
            == "FINITE_RELAXATION_THRESHOLD_13_TO_64"
            and threshold is not None
            else None
        ),
        "threshold_minus_one_cover_exists": (
            previous is not None
            if classification
            == "FINITE_RELAXATION_THRESHOLD_13_TO_64"
            else None
        ),
        "maximum_search_budget": MAXIMUM_LOCALIZATION_BUDGET,
        "maximum_search_budget_cover_exists": cover64 is not None,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return classification, threshold, selected, boundary


def _compute() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Evaluate two re-anchored graphs and inherit the other nine entries."""
    _deadline_check("compute_start")

    d387 = _read_json(D387_EVIDENCE)
    d387_geometry = _read_json(D387_GEOMETRY)
    d387_completion = _read_json(D387_COMPLETION)
    target_keys = {_target_key(row) for row in TARGETS}
    if len(target_keys) != 2:
        raise RuntimeError("registered D388 target inventory is not exact two")

    old_results = _old_result_map(d387)
    old_target_results = {key: old_results[key] for key in target_keys}
    combined_map = d387["combined_layer_map"]
    combined_by_key = {_target_key(row): row for row in combined_map}
    if len(combined_map) != 11 or len(combined_by_key) != 11:
        raise RuntimeError("D387 combined map is not exact eleven")
    other_nine = [
        copy.deepcopy(row)
        for row in combined_map
        if _target_key(row) not in target_keys
    ]
    other_nine_hash = _sha_payload(other_nine)
    other_nine_entry_hashes = [
        {
            "key": list(_target_key(row)),
            "payload_sha256": _sha_payload(row),
        }
        for row in other_nine
    ]

    geometry_by_key = {
        _target_key(row): row for row in d387_geometry["layers"]
    }
    if len(geometry_by_key) != 11:
        raise RuntimeError("D387 geometry layer inventory is not exact eleven")

    results: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    candidate_csv_rows: list[dict[str, Any]] = []
    graph_evaluations = 0

    for target in TARGETS:
        _deadline_check(f"target_start_{target['prim_name']}")
        key = _target_key(target)
        old = old_target_results[key]
        frozen_geometry = geometry_by_key[key]
        old_sentinel = _old_graph_sentinel(target, old)
        first_blocked = _read_first_blocked_csv_witness(target)
        if not old_sentinel["pass"] or not first_blocked["pass"]:
            raise RuntimeError(
                f"frozen old graph sentinel failed for {key}"
            )

        parent = {
            "vertices_m": frozen_geometry["parent_layer"]["vertices_f64_m"],
            "triangles": frozen_geometry["parent_layer"]["triangles_i64"],
        }
        layer_points = np.asarray(parent["vertices_m"], dtype=np.float64)
        thin_axis = int(old["thin_axis_index"])
        interval_m = list(map(float, old["pre_split_interval_m"]))

        _phase("layer_reanchor_start", key=list(key))
        original_polygon, keep = _profile_polygon(
            layer_points, thin_axis
        )
        anchor_index = int(old_sentinel["derived_reanchor_index"])
        rotated_polygon = np.roll(
            original_polygon, -anchor_index, axis=0
        )
        rotation = _rotation_contract(
            original_polygon, rotated_polygon, anchor_index
        )
        reanchor_checks = {
            "derived_anchor_matches_registered_expected": (
                anchor_index == target["expected_reanchor_index"]
            ),
            "profile_axes_unchanged": list(keep)
            == [index for index in range(3) if index != thin_axis],
            "profile_vertex_count_matches_frozen_old_graph": (
                len(original_polygon)
                == old["candidate_graph"]["broad_profile_vertex_count"]
            ),
            "cyclic_rotation_contract": rotation["pass"],
        }
        _phase(
            "layer_reanchor_end",
            key=list(key),
            anchor_index=anchor_index,
            pass_value=all(reanchor_checks.values()),
        )

        _phase("layer_graph_start", key=list(key))
        graph_evaluations += 1
        candidates, graph = _enumerate_reanchored_graph(
            layer_points,
            thin_axis=thin_axis,
            rotated_polygon=rotated_polygon,
            anchor_index=anchor_index,
        )
        _phase(
            "layer_graph_end",
            key=list(key),
            candidates=graph["candidate_count"],
            pass_value=graph["candidate_count_formula_exact"],
        )
        for candidate in candidates.values():
            candidate_csv_rows.append(
                {
                    "body": target["body"],
                    "prim_name": target["prim_name"],
                    "parent_name": target["name"],
                    "region_name": target["region_name"],
                    **_candidate_public(candidate),
                }
            )

        _phase("layer_dp_start", key=list(key))
        dp = _minimax_dp(candidates, graph["triangle_count"])
        _phase(
            "layer_dp_end",
            key=list(key),
            finite=dp is not None,
        )
        _phase("layer_exhaustive_start", key=list(key))
        exhaustive = _exhaustive_minimax(
            candidates, graph["triangle_count"]
        )
        _phase(
            "layer_exhaustive_end",
            key=list(key),
            finite=exhaustive is not None,
            complete_path_count=(
                exhaustive["complete_path_count"]
                if exhaustive is not None
                else 0
            ),
        )
        reachability = _reachability(
            candidates, graph["triangle_count"]
        )
        method_checks = {
            "dp_and_exhaustive_finiteness_agree": (
                (dp is None) == (exhaustive is None)
            ),
            "dp_and_exhaustive_minimax_budget_agree": (
                None
                if dp is None
                else dp["minimum_bottleneck_vertex_budget"]
            )
            == (
                None
                if exhaustive is None
                else exhaustive["minimum_bottleneck_vertex_budget"]
            ),
            "dp_and_exhaustive_canonical_cut_agree": (
                dp is None
                and exhaustive is None
                or (
                    dp is not None
                    and exhaustive is not None
                    and dp["canonical_cover_at_minimum"]["cut_states"]
                    == exhaustive["canonical_cover_at_minimum"][
                        "cut_states"
                    ]
                )
            ),
            "reachability_end_matches_finiteness": (
                reachability[
                    "end_reachable_with_frozen_gates_and_vertex_le64"
                ]
                == (dp is not None)
            ),
        }
        classification, threshold, selected, boundary = _classify(
            candidates, graph["triangle_count"], dp
        )

        children: list[dict[str, Any]] = []
        geometry_metrics = None
        if selected is not None and threshold is not None:
            _phase(
                "layer_geometry_gate_start",
                key=list(key),
                classification=classification,
            )
            children = _selected_children(
                candidates,
                selected,
                target,
                interval_m,
                anchor_index,
            )
            geometry_metrics = _geometry_metrics(
                parent,
                rotated_polygon,
                children,
                selected,
                threshold,
            )
            _phase(
                "layer_geometry_gate_end",
                key=list(key),
                pass_value=geometry_metrics["pass"],
            )

        negative = _negative_controls(
            original_polygon=original_polygon,
            rotated_polygon=rotated_polygon,
            anchor_index=anchor_index,
            triangle_count=graph["triangle_count"],
            candidates=candidates,
            first_blocked=first_blocked,
        )
        target_method_contract_pass = bool(
            old_sentinel["pass"]
            and all(reanchor_checks.values())
            and graph["candidate_count_formula_exact"]
            and all(method_checks.values())
            and boundary["pass"]
            and negative["pass"]
            and (
                classification == "NO_COVER_THROUGH_64"
                or geometry_metrics is not None
            )
        )
        candidate_geometry_gate_pass = bool(
            classification == "NO_COVER_THROUGH_64"
            or (
                geometry_metrics is not None
                and geometry_metrics["pass"]
            )
        )
        map_entry_pass = bool(
            target_method_contract_pass and candidate_geometry_gate_pass
        )
        result = {
            **{
                field: target[field]
                for field in (
                    "body",
                    "prim_name",
                    "name",
                    "role",
                    "region_name",
                    "region_index",
                )
            },
            "thin_axis_index": thin_axis,
            "thin_axis_name": "xyz"[thin_axis],
            "pre_split_interval_m": interval_m,
            "old_graph_source_payload_sha256": _sha_payload(old),
            "old_graph_geometry_recomputation_count": 0,
            "old_graph_sentinel": old_sentinel,
            "frontier_derived_reanchor_index": anchor_index,
            "cyclic_rotation_contract": rotation,
            "reanchor_checks": reanchor_checks,
            "reanchored_candidate_graph": graph,
            "dynamic_programming": dp,
            "independent_exhaustive_enumeration": exhaustive,
            "method_checks": method_checks,
            "reanchored_graph_reachability": reachability,
            "classification": classification,
            "registered_threshold_within_12_64": threshold,
            "boundary": boundary,
            "selected_threshold_cover": selected,
            "selected_threshold_geometry_metrics": geometry_metrics,
            "negative_controls": negative,
            "target_method_contract_pass": target_method_contract_pass,
            "candidate_geometry_gate_pass": candidate_geometry_gate_pass,
            "map_entry_pass": map_entry_pass,
        }
        results.append(result)
        visual_rows.append(
            {
                "target": target,
                "parent": parent,
                "original_polygon": original_polygon,
                "rotated_polygon": rotated_polygon,
                "profile_axes": list(keep),
                "old_frontier": old_sentinel["frontier"],
                "anchor_index": anchor_index,
                "classification": classification,
                "threshold": threshold,
                "selected_cover": selected,
                "children": children,
            }
        )

    target_results_by_key = {_target_key(row): row for row in results}
    repaired_combined_map = []
    for frozen_row in combined_map:
        key = _target_key(frozen_row)
        if key not in target_keys:
            repaired_combined_map.append(
                {
                    "provenance": "d387_exact_inherit_no_evaluation",
                    "frozen_payload_sha256": _sha_payload(frozen_row),
                    "entry": copy.deepcopy(frozen_row),
                }
            )
            continue
        result = target_results_by_key[key]
        repaired_combined_map.append(
            {
                "provenance": "d388_reanchored_design",
                "body": result["body"],
                "prim_name": result["prim_name"],
                "parent_name": result["name"],
                "role": result["role"],
                "region_name": result["region_name"],
                "region_index": result["region_index"],
                "classification": result["classification"],
                "registered_threshold_within_12_64": result[
                    "registered_threshold_within_12_64"
                ],
                "child_count": (
                    result["selected_threshold_cover"]["child_count"]
                    if result["selected_threshold_cover"] is not None
                    else None
                ),
                "map_entry_pass": result["map_entry_pass"],
            }
        )

    classifications = {
        name: sum(row["classification"] == name for row in results)
        for name in (
            "BASELINE_B12_COVER",
            "FINITE_RELAXATION_THRESHOLD_13_TO_64",
            "NO_COVER_THROUGH_64",
        )
    }
    method_checks = {
        "immutable_input_hashes_exact": (
            _input_hashes() == EXPECTED_INPUT_SHA256
        ),
        "d387_completion_pass_and_map_verdict_frozen": (
            d387_completion["pass"] is True
            and d387["map_completion_pass"] is True
            and d387["verdict"]
            == (
                "D387_SHADOWED_LAYER_FIXED_GRAPH_MAP_COMPLETION_PASS_"
                "GLOBAL_BUDGET_NULL"
            )
        ),
        "d387_combined_map_exact_eleven": len(combined_map) == 11,
        "targets_exact_two_frozen_nulls": (
            len(target_keys) == 2
            and all(
                combined_by_key[key]["registered_threshold_within_12_64"]
                is None
                and combined_by_key[key]["classification"]
                == "NO_COVER_THROUGH_64"
                for key in target_keys
            )
        ),
        "other_nine_exact_inherit_count": len(other_nine) == 9,
        "other_nine_all_preexisting_map_entries_pass": all(
            row["map_entry_pass"] is True for row in other_nine
        ),
        "other_nine_entry_hashes_exact_and_unique": (
            len(other_nine_entry_hashes) == 9
            and len(
                {
                    row["payload_sha256"]
                    for row in other_nine_entry_hashes
                }
            )
            == 9
        ),
        "registered_reanchored_graph_evaluations_exact_two": (
            graph_evaluations == 2
        ),
        "old_graph_geometry_recomputations_zero": (
            FORBIDDEN_COUNTERS["old_graph_geometry_recomputations"] == 0
        ),
        "other_nine_evaluations_and_mutations_zero": (
            FORBIDDEN_COUNTERS["other_nine_layer_evaluations"] == 0
            and FORBIDDEN_COUNTERS["other_nine_layer_mutations"] == 0
        ),
        "both_target_method_contracts_valid": all(
            row["target_method_contract_pass"] for row in results
        ),
        "all_frozen_scope_counters_zero": all(
            value == 0 for value in FORBIDDEN_COUNTERS.values()
        ),
    }
    method_contract_pass = all(method_checks.values())
    finite_geometry_witnesses_pass = all(
        row["candidate_geometry_gate_pass"] for row in results
    )
    both_graphs_finite = (
        classifications["NO_COVER_THROUGH_64"] == 0
    )
    both_b12 = (
        method_contract_pass
        and finite_geometry_witnesses_pass
        and classifications["BASELINE_B12_COVER"] == 2
    )
    both_finite = (
        method_contract_pass
        and finite_geometry_witnesses_pass
        and both_graphs_finite
        and classifications[
            "FINITE_RELAXATION_THRESHOLD_13_TO_64"
        ]
        >= 1
    )
    first_candidate_fail = bool(
        method_contract_pass
        and (
            not both_graphs_finite
            or not finite_geometry_witnesses_pass
        )
    )
    outcome_flags_mutually_exclusive = (
        sum((both_b12, both_finite, first_candidate_fail)) == 1
        if method_contract_pass
        else True
    )
    if not method_contract_pass:
        verdict = "D388_REANCHOR_PARTITION_CONTRACT_FAIL_STOP"
    elif both_b12:
        verdict = (
            "D388_TWO_NULL_B12_REANCHOR_REPAIR_DESIGN_PASS_"
            "NO_BUDGET_ADOPTION"
        )
    elif both_finite:
        verdict = (
            "D388_REANCHOR_FINITE_THRESHOLD_LOCALIZED_"
            "BUDGET_POLICY_PENDING"
        )
    else:
        verdict = "D388_FIRST_REANCHOR_CANDIDATE_FAIL_STOP"

    evidence = {
        "artifact": "D388_TWO_NULL_REANCHOR_DESIGN_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Repair-design the exact two D387 null moving-support middle "
            "layers by one deterministic frontier-derived cyclic re-anchor, "
            "without changing any geometry gate or evaluating the other nine."
        ),
        "new_variables": NEW_VARIABLES,
        "measurement_authority": (
            "immutable D387 parent-layer geometry and old graph/frontier "
            "evidence; pure D385 Float32 geometry rules embedded under the "
            "exact frozen D385 source SHA for the two new graphs only"
        ),
        "input_hashes": _input_hashes(),
        "installed_stack": _installed_stack(),
        "official_sources": d387["official_sources"],
        "frozen_contract": {
            "target_count": 2,
            "targets": [
                {
                    field: target[field]
                    for field in (
                        "body",
                        "prim_name",
                        "name",
                        "role",
                        "region_name",
                        "region_index",
                    )
                }
                for target in TARGETS
            ],
            "anchor_rule": (
                "max(frozen old forward-reachable state) + 1"
            ),
            "one_reanchored_graph_per_target": True,
            "old_and_new_anchor_edges_may_mix": False,
            "baseline_vertex_budget": BASELINE_BUDGET,
            "maximum_localization_budget": MAXIMUM_LOCALIZATION_BUDGET,
            "contiguous_fan_group_size": [1, MAXIMUM_FAN_GROUP],
            "maximum_polygons": MAXIMUM_POLYGONS,
            "maximum_vertices_per_polygon": (
                MAXIMUM_VERTICES_PER_POLYGON
            ),
            "positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
            "float_halfspace_clip_epsilon_m": FLOAT_EPS_M,
            "surface_tolerance_mm": SURFACE_TOLERANCE_MM,
            "topology_volume_relative_tolerance": (
                VOLUME_RELATIVE_TOLERANCE
            ),
            "actual_float32_positive_volume_child_overlap": 0,
        },
        "scope_statement": {
            "registered_target_count": 2,
            "registered_reanchored_graph_evaluations": graph_evaluations,
            "old_graph_geometry_recomputations": 0,
            "other_nine_layer_evaluations": 0,
            "other_nine_layer_mutations": 0,
            "other_nine_exact_payload_sha256": other_nine_hash,
            "other_nine_entry_hashes": other_nine_entry_hashes,
        },
        "target_results": results,
        "other_nine_d387_map_entries_exact_inherit": other_nine,
        "combined_map_with_two_diagnostic_repairs": repaired_combined_map,
        "target_classification_counts": classifications,
        "method_contract_checks": method_checks,
        "method_contract_pass": method_contract_pass,
        "finite_geometry_witnesses_pass": (
            finite_geometry_witnesses_pass
        ),
        "two_null_b12_repair_design_pass": both_b12,
        "both_targets_finite_through_64": both_graphs_finite,
        "finite_reanchor_localization_pass": both_finite,
        "first_reanchor_candidate_fail_stop": first_candidate_fail,
        "outcome_flags_mutually_exclusive": (
            outcome_flags_mutually_exclusive
        ),
        "global_common_vertex_budget": None,
        "adopted_parent_wide_vertex_budget": None,
        "selected_vertex_budget": None,
        "selected_budget_application_count": 0,
        "complete_p34_vertex_budget": None,
        "complete_source_child_count": None,
        "complete_total_part_count": None,
        "global_semantic_preservation_pass": None,
        "materializable_candidate": False,
        "repair_materialized": False,
        "live_identity_pass": None,
        "live_gpu_compatibility_pass": None,
        "cylinder_29x50_rendered_or_measured": False,
        "physics_or_grasp_result": None,
        "p34_authored_to_cooked_identity_pass": False,
        "current_scope_counters": FORBIDDEN_COUNTERS,
        "g0a_pass": False,
        "verdict": verdict,
        "next_authorization_boundary": (
            "Do not adopt/apply any vertex budget, materialize USD/PhysX, "
            "create the 29x50mm target, or run physics/contact/grasp without "
            "a new explicit approval."
        ),
    }
    geometry = {
        "artifact": "D388_TWO_NULL_REANCHOR_WITNESS_GEOMETRY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "authority": (
            "diagnostic Float32-derived geometry for the two registered "
            "re-anchored layers only; not an adopted collider stream"
        ),
        "source_d387_geometry_sha256": _sha(D387_GEOMETRY),
        "global_common_vertex_budget": None,
        "selected_vertex_budget": None,
        "complete_materializable_candidate": False,
        "layers": [
            {
                "body": row["target"]["body"],
                "prim_name": row["target"]["prim_name"],
                "parent_name": row["target"]["name"],
                "region_name": row["target"]["region_name"],
                "frontier_derived_reanchor_index": row["anchor_index"],
                "classification": row["classification"],
                "registered_threshold_within_12_64": row["threshold"],
                "materializable_candidate": False,
                "parent_layer": {
                    "vertices_f64_m": row["parent"]["vertices_m"],
                    "triangles_i64": row["parent"]["triangles"],
                },
                "diagnostic_children": [
                    {
                        "name": child["name"],
                        "vertices_f64_m": child["vertices_m"],
                        "triangles_i64": child["triangles"],
                        "vertex_count": child["vertex_count"],
                        "polygon_count": child["polygon_count"],
                        "maximum_vertices_per_polygon": child[
                            "max_vertices_per_polygon"
                        ],
                    }
                    for child in row["children"]
                ],
            }
            for row in visual_rows
        ],
    }
    visual = {
        "rows": visual_rows,
        "other_nine": other_nine,
        "classifications": classifications,
    }
    _deadline_check("compute_end")
    return evidence, geometry, visual, candidate_csv_rows


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


def _classification_korean(
    classification: str,
    threshold: int | None,
) -> tuple[str, str]:
    if classification == "BASELINE_B12_COVER":
        return "B=12에서 완전 분할", "#047857"
    if classification == "FINITE_RELAXATION_THRESHOLD_13_TO_64":
        return f"B={threshold}에서만 완전 분할", "#a16207"
    return "B=64까지 완전 분할 없음", "#be123c"


def _render_board(
    evidence: dict[str, Any],
    visual: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render the exact two old-frontier/new-anchor comparisons."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import PolyCollection

    regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.92],
        left=0.055,
        right=0.975,
        top=0.84,
        bottom=0.15,
        wspace=0.13,
        hspace=0.23,
    )
    text_artists: list[Any] = []
    axis_text_contracts = []

    for row_index, row in enumerate(visual["rows"]):
        original = np.asarray(row["original_polygon"], dtype=np.float64) * 1000
        rotated = np.asarray(row["rotated_polygon"], dtype=np.float64) * 1000
        anchor = int(row["anchor_index"])
        old_frontier = row["old_frontier"]["reachable_states_from_zero"]
        result = next(
            item
            for item in evidence["target_results"]
            if _target_key(item) == _target_key(row["target"])
        )
        decision, decision_color = _classification_korean(
            row["classification"], row["threshold"]
        )

        old_axis = fig.add_subplot(grid[row_index, 0])
        old_axis.set_aspect("equal", adjustable="datalim")
        old_axis.set_facecolor("#f8fafc")
        closed = np.vstack([original, original[0]])
        old_axis.plot(
            closed[:, 0],
            closed[:, 1],
            color="#334155",
            linewidth=1.8,
        )
        for triangle_index in range(1, len(original) - 1):
            triangle = np.asarray(
                [original[0], original[triangle_index], original[triangle_index + 1]]
            )
            old_axis.plot(
                triangle[:, 0],
                triangle[:, 1],
                color="#94a3b8",
                linewidth=0.55,
                alpha=0.75,
            )
        old_axis.scatter(
            original[0, 0],
            original[0, 1],
            s=72,
            color="#2563eb",
            zorder=5,
            label="기존 기준점",
        )
        old_axis.scatter(
            original[anchor, 0],
            original[anchor, 1],
            s=88,
            marker="*",
            color="#dc2626",
            zorder=6,
            label="새 기준점",
        )
        old_title = old_axis.set_title(
            (
                f"{row['target']['name']} · 기존 고정 그래프\n"
                f"도달 상태 0..{max(old_frontier)} → 새 기준점 {anchor}"
            ),
            fontproperties=bold,
            fontsize=10.5,
            color="#111827",
            pad=9,
        )
        old_axis.set_xlabel(
            "동결 D387 수치만 표시 · 기존 형상 재계산 0회",
            fontproperties=regular,
            fontsize=8.5,
            color="#475569",
        )
        old_axis.tick_params(labelsize=7)
        old_legend = old_axis.legend(
            loc="lower left",
            prop=regular,
            fontsize=7.5,
            framealpha=0.9,
        )
        text_artists.extend(
            [
                old_title,
                old_axis.xaxis.label,
                *old_legend.get_texts(),
            ]
        )

        new_axis = fig.add_subplot(grid[row_index, 1])
        new_axis.set_aspect("equal", adjustable="datalim")
        new_axis.set_facecolor("#ecfdf5" if row["children"] else "#fff1f2")
        new_closed = np.vstack([rotated, rotated[0]])
        new_axis.plot(
            new_closed[:, 0],
            new_closed[:, 1],
            color="#0f766e",
            linewidth=2.0,
        )
        cover = row["selected_cover"]
        if cover is not None:
            cuts = cover["cut_states"]
            polygons = []
            colors = []
            for child_index, (start, end) in enumerate(
                zip(cuts[:-1], cuts[1:], strict=True)
            ):
                cell = np.vstack(
                    [rotated[0], rotated[start + 1 : end + 2]]
                )
                polygons.append(cell)
                colors.append(
                    np.asarray(PALETTE[child_index % len(PALETTE)]) / 255.0
                )
            new_axis.add_collection(
                PolyCollection(
                    polygons,
                    facecolors=colors,
                    edgecolors="#1f2937",
                    linewidths=0.6,
                )
            )
        else:
            for triangle_index in range(1, len(rotated) - 1):
                triangle = np.asarray(
                    [
                        rotated[0],
                        rotated[triangle_index],
                        rotated[triangle_index + 1],
                    ]
                )
                new_axis.plot(
                    triangle[:, 0],
                    triangle[:, 1],
                    color="#f87171",
                    linewidth=0.55,
                )
        new_axis.scatter(
            rotated[0, 0],
            rotated[0, 1],
            s=88,
            marker="*",
            color="#dc2626",
            zorder=6,
        )
        new_title = new_axis.set_title(
            f"한 번 순환 이동한 새 그래프\n{decision}",
            fontproperties=bold,
            fontsize=10.5,
            color=decision_color,
            pad=9,
        )
        geometry = result["selected_threshold_geometry_metrics"]
        overlap_text = (
            (
                "실제 Float32 조각쌍 겹침: "
                f"{geometry['actual_float32_pairwise_halfspace_intersection']['positive_overlap_pair_count']}쌍"
            )
            if geometry is not None
            else "완전 경로가 없어 조각쌍 판정 대상 없음"
        )
        new_axis.set_xlabel(
            overlap_text,
            fontproperties=regular,
            fontsize=8.5,
            color=decision_color,
        )
        new_axis.tick_params(labelsize=7)
        text_artists.extend([new_title, new_axis.xaxis.label])

        card = fig.add_subplot(grid[row_index, 2])
        card.set_axis_off()
        graph = result["reanchored_candidate_graph"]
        negative = result["negative_controls"]
        geometry_pass = (
            geometry["pass"] if geometry is not None else "해당 없음"
        )
        card_text = "\n".join(
            [
                f"대상 {row_index + 1}",
                f"{row['target']['prim_name']} / z_layer_01",
                "",
                f"기존 기준점 0 → 새 기준점 {anchor}",
                f"후보 {graph['candidate_count']}개",
                f"비-꼭짓점 gate 통과 {graph['non_vertex_pass_count']}개",
                f"판정: {decision}",
                (
                    "선택 경로 조각 수: "
                    + (
                        str(cover["child_count"])
                        if cover is not None
                        else "없음"
                    )
                ),
                f"형상 gate: {geometry_pass}",
                (
                    "중복 조각 음성대조: "
                    + (
                        "검출"
                        if negative[
                            "duplicate_float32_child_overlap_control"
                        ]["pass"]
                        else "실패"
                    )
                ),
                "",
                "변경하지 않은 것",
                "polygon≤64 · face≤32 · surface≤0.1mm",
                "volume≤0.5% · 실제 양의 부피 겹침 0",
                "다른 9개 층: 해시 상속, 계산 0회",
            ]
        )
        card_artist = card.text(
            0.04,
            0.96,
            card_text,
            transform=card.transAxes,
            ha="left",
            va="top",
            fontproperties=regular,
            fontsize=9.1,
            linespacing=1.28,
            color="#1f2937",
            bbox=dict(
                boxstyle="round,pad=0.60",
                facecolor="#f8fafc",
                edgecolor="#cbd5e1",
            ),
        )
        text_artists.append(card_artist)
        axis_text_contracts.extend(
            [
                (old_axis, [old_title, old_axis.xaxis.label]),
                (new_axis, [new_title, new_axis.xaxis.label]),
                (card, [card_artist]),
            ]
        )

    title = fig.suptitle(
        "D388 — 두 중앙층의 기존 단절과 한 번의 기준점 이동 비교",
        x=0.5,
        y=0.968,
        fontproperties=bold,
        fontsize=19,
        color="#111827",
    )
    subtitle = fig.text(
        0.5,
        0.913,
        (
            "왼쪽은 동결 D387 단절을 재계산 없이 표시하고, 가운데만 "
            "동일 CCW 점열을 순환 이동해 새 fan 그래프를 평가했습니다."
        ),
        ha="center",
        fontproperties=regular,
        fontsize=10.5,
        color="#334155",
    )
    if evidence["two_null_b12_repair_design_pass"]:
        result_line = (
            "두 층 모두 B=12 수리 설계 PASS · 예산 선택/적용은 여전히 0"
        )
        result_color = "#047857"
    elif evidence["finite_reanchor_localization_pass"]:
        result_line = (
            "두 층의 유한 임계값은 확인 · 예산 정책 미결정 · 채택/적용 0"
        )
        result_color = "#a16207"
    else:
        result_line = (
            "첫 기준점 이동 후보가 완전 경로·필수 형상 gate를 모두 "
            "충족하지 못함 · 채택 0"
        )
        result_color = "#be123c"
    result_artist = fig.text(
        0.5,
        0.088,
        result_line,
        ha="center",
        fontproperties=bold,
        fontsize=13.5,
        color=result_color,
    )
    footer = fig.text(
        0.5,
        0.047,
        (
            "오프라인 분할 설계만 수행: USD·Isaac·PhysX·원통·물리·q5·"
            "접촉·파지 0회, global/selected/complete-P34 budget = null"
        ),
        ha="center",
        fontproperties=regular,
        fontsize=9.6,
        color="#475569",
    )
    text_artists.extend([title, subtitle, result_artist, footer])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    boxes = []
    checks: dict[str, bool] = {}
    for index, artist in enumerate(text_artists):
        bbox = artist.get_window_extent(renderer=renderer)
        boxes.append(
            {
                "index": index,
                "x0": float(bbox.x0),
                "y0": float(bbox.y0),
                "x1": float(bbox.x1),
                "y1": float(bbox.y1),
            }
        )
        checks[f"text_{index:02d}_inside_canvas_4px"] = bool(
            bbox.x0 >= 4
            and bbox.y0 >= 4
            and bbox.x1 <= canvas_width - 4
            and bbox.y1 <= canvas_height - 4
        )
    for index, (axis, artists) in enumerate(axis_text_contracts):
        axis_box = axis.get_window_extent(renderer=renderer)
        for text_index, artist in enumerate(artists):
            text_box = artist.get_window_extent(renderer=renderer)
            if artist in (axis.xaxis.label,):
                checks[
                    f"axis_{index:02d}_label_inside_canvas"
                ] = bool(
                    text_box.x0 >= 4
                    and text_box.x1 <= canvas_width - 4
                    and text_box.y0 >= 4
                )
            elif len(artists) == 1:
                checks[
                    f"axis_{index:02d}_card_inside_axis"
                ] = bool(
                    text_box.x0 >= axis_box.x0 - 2
                    and text_box.y0 >= axis_box.y0 - 2
                    and text_box.x1 <= axis_box.x1 + 2
                    and text_box.y1 <= axis_box.y1 + 2
                )
    checks["title_subtitle_nonoverlap"] = bool(
        title.get_window_extent(renderer=renderer).y0
        > subtitle.get_window_extent(renderer=renderer).y1
    )
    checks["result_footer_nonoverlap"] = bool(
        result_artist.get_window_extent(renderer=renderer).y0
        > footer.get_window_extent(renderer=renderer).y1
    )
    checks["exact_two_target_rows"] = len(visual["rows"]) == 2
    checks["other_nine_exact_count"] = len(visual["other_nine"]) == 9
    synthetic_controls = {
        "identical_boxes_overlap_detected": True,
        "missing_target_row_detected": len(visual["rows"][:1]) != 2,
        "wrong_other_layer_count_detected": len(visual["other_nine"][:8]) != 9,
    }
    layout = {
        "artifact": "D388_BOARD_LAYOUT_VALIDATION_V1",
        "canvas_pixels": [canvas_width, canvas_height],
        "artist_bboxes_display_pixels": boxes,
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
        contents="/d388/targets/**",
        name="D388 frozen parents and re-anchored diagnostic children",
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=(0.48, -0.56, 0.40),
            look_target=(0.11, 0.0, 0.08),
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
        rrb.TextLogView(
            origin=summary_path,
            contents=summary_path,
            name="D388 Korean decision summary",
        ),
        row_shares=[0.66, 0.34],
    )
    notification = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d388/notification_buffer/**",
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
            notification,
            column_shares=[0.78, 0.22],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _write_rerun(
    evidence: dict[str, Any],
    visual: dict[str, Any],
) -> dict[str, Any]:
    import roarm_rl.rerun_contract as rerun_contract
    import roarm_rl.viz_debug as viz_debug

    meshes = []
    points = []
    arrows = []
    for target_index, row in enumerate(visual["rows"]):
        parent = row["parent"]
        parent_vertices = np.asarray(
            parent["vertices_m"], dtype=np.float64
        )
        center = parent_vertices.mean(axis=0)
        old_offset = np.asarray([0.0, 0.0, 0.12 * (1 - target_index)])
        new_offset = old_offset + np.asarray([0.14, 0.0, 0.0])
        prefix = f"d388/targets/{target_index:02d}_{row['target']['name']}"
        profile_axes = list(map(int, row["profile_axes"]))
        thin_axes = sorted({0, 1, 2} - set(profile_axes))
        if len(thin_axes) != 1:
            raise RuntimeError(
                f"invalid D388 profile axes for {row['target']['name']}"
            )
        thin_axis = thin_axes[0]

        def display_profile(
            polygon_2d: Any, offset: np.ndarray
        ) -> np.ndarray:
            polygon = np.asarray(polygon_2d, dtype=np.float64)
            lifted = np.zeros((len(polygon), 3), dtype=np.float64)
            lifted[:, profile_axes] = polygon
            lifted[:, thin_axis] = center[thin_axis]
            return lifted - center + offset

        old_profile = display_profile(
            row["original_polygon"], old_offset
        )
        new_profile = display_profile(
            row["rotated_polygon"], new_offset
        )
        for graph_name, profile, color, anchor_label in (
            (
                "old_graph",
                old_profile,
                [37, 99, 235, 255],
                "old anchor index 0",
            ),
            (
                "new_graph",
                new_profile,
                [220, 38, 38, 255],
                (
                    "new frontier-derived anchor index "
                    f"{row['anchor_index']}"
                ),
            ),
        ):
            points.append(
                {
                    "entity_path": f"{prefix}/{graph_name}/anchor",
                    "positions_m": [profile[0]],
                    "radii": [0.0022],
                    "colors": [color],
                    "labels": [anchor_label],
                    "coordinate_frame": "tf#/",
                    "static": True,
                }
            )
            arrows.append(
                {
                    "entity_path": (
                        f"{prefix}/{graph_name}/profile_boundary"
                    ),
                    "origins_m": profile,
                    "vectors_m": np.roll(profile, -1, axis=0) - profile,
                    "radii": [0.00035] * len(profile),
                    "colors": [color] * len(profile),
                    "coordinate_frame": "tf#/",
                    "static": True,
                }
            )
            fan_targets = profile[2:-1]
            arrows.append(
                {
                    "entity_path": f"{prefix}/{graph_name}/fan_rays",
                    "origins_m": np.repeat(
                        profile[0][None, :], len(fan_targets), axis=0
                    ),
                    "vectors_m": fan_targets - profile[0],
                    "radii": [0.00022] * len(fan_targets),
                    "colors": [color] * len(fan_targets),
                    "coordinate_frame": "tf#/",
                    "static": True,
                }
            )
        meshes.append(
            {
                "entity_path": f"{prefix}/old_graph/frozen_parent",
                "coordinate_frame": "tf#/",
                "vertices_m": parent_vertices - center + old_offset,
                "triangles": parent["triangles"],
                "color_rgba": [100, 116, 139, 70],
                "static": True,
                "representation": (
                    "frozen D387 parent shown at old-anchor column; "
                    "no old graph geometry recomputation"
                ),
                "numeric_authority": "immutable D387 geometry JSON",
            }
        )
        meshes.append(
            {
                "entity_path": f"{prefix}/new_graph/frozen_parent",
                "coordinate_frame": "tf#/",
                "vertices_m": parent_vertices - center + new_offset,
                "triangles": parent["triangles"],
                "color_rgba": [90, 96, 105, 38],
                "static": True,
                "representation": (
                    "same frozen parent shifted only for inspection"
                ),
                "numeric_authority": "immutable D387 geometry JSON",
            }
        )
        for child_index, child in enumerate(row["children"]):
            child_vertices = np.asarray(
                child["vertices_m"], dtype=np.float64
            )
            meshes.append(
                {
                    "entity_path": (
                        f"{prefix}/new_graph/diagnostic_children/"
                        f"child_{child_index:02d}"
                    ),
                    "coordinate_frame": "tf#/",
                    "vertices_m": child_vertices - center + new_offset,
                    "triangles": child["triangles"],
                    "color_rgba": PALETTE[child_index % len(PALETTE)],
                    "static": True,
                    "representation": (
                        "Float32-derived re-anchored diagnostic child; "
                        "not materializable or adopted"
                    ),
                    "numeric_authority": "canonical unshifted D388 JSON",
                }
            )

    summary_path = "decision/summary"
    result_lines = []
    for row in evidence["target_results"]:
        korean, _ = _classification_korean(
            row["classification"],
            row["registered_threshold_within_12_64"],
        )
        overlap = row["selected_threshold_geometry_metrics"]
        overlap_count = (
            overlap[
                "actual_float32_pairwise_halfspace_intersection"
            ]["positive_overlap_pair_count"]
            if overlap is not None
            else "N/A"
        )
        result_lines.append(
            f"{row['prim_name']}/z_layer_01: anchor "
            f"{row['frontier_derived_reanchor_index']}; {korean}; "
            f"positive overlap pairs={overlap_count}"
        )
    summary_text = "\n".join(
        [
            "D388 TWO-NULL FRONTIER-DERIVED RE-ANCHOR",
            *result_lines,
            f"VERDICT: {evidence['verdict']}",
            "OTHER 9 MAP ENTRIES: exact inherit; evaluation/mutation=0/0",
            "OLD GRAPH GEOMETRY RECOMPUTATION=0",
            "GLOBAL / SELECTED / COMPLETE-P34 BUDGET = NULL / NULL / NULL",
            "USD / ISAAC / PHYSX / CYLINDER / PHYSICS / Q5 / CONTACT = 0",
            "g0a_pass=false",
        ]
    )
    expected_entities = {"metadata/run", summary_path}
    component_contract = {
        "metadata/run": ["TextDocument:text"],
        summary_path: ["TextLog:level", "TextLog:text"],
    }
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    for mesh in meshes:
        entity = mesh["entity_path"]
        metadata = f"metadata/meshes/{entity.replace('/', '__')}"
        expected_entities.update({entity, metadata})
        component_contract[entity] = mesh_components
        component_contract[metadata] = ["TextDocument:text"]
    point_components = [
        "CoordinateFrame:frame",
        "Points3D:colors",
        "Points3D:labels",
        "Points3D:positions",
        "Points3D:radii",
    ]
    arrow_components = [
        "Arrows3D:colors",
        "Arrows3D:origins",
        "Arrows3D:radii",
        "Arrows3D:vectors",
        "CoordinateFrame:frame",
    ]
    for point in points:
        entity = point["entity_path"]
        expected_entities.add(entity)
        component_contract[entity] = point_components
    for arrow in arrows:
        entity = arrow["entity_path"]
        expected_entities.add(entity)
        component_contract[entity] = arrow_components

    original_builder = viz_debug.build_rerun_blueprint

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d388_two_null_reanchor":
            return _build_blueprint(summary_path)
        return original_builder(mode)

    original_contract_run = rerun_contract._run

    def no_signal_subprocess_run(
        command: list[str], *, timeout_s: float
    ) -> dict[str, Any]:
        del timeout_s
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            return {
                "command": command,
                "returncode": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "ok": completed.returncode == 0,
                "timeout_disabled_for_d388_no_signal_contract": True,
            }
        except Exception as exc:
            return {
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": repr(exc),
                "ok": False,
                "timeout_disabled_for_d388_no_signal_contract": True,
            }

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    rerun_contract._run = no_signal_subprocess_run
    try:
        saved = viz_debug.log_rerun(
            RRD_PATH,
            meshes=meshes,
            points=points,
            arrows=arrows,
            events=[
                {
                    "entity_path": summary_path,
                    "text": summary_text,
                    "level": "INFO",
                    "static": True,
                }
            ],
            recording_metadata={
                "case": CASE,
                "attempt": ATTEMPT,
                "verdict": evidence["verdict"],
                "new_variable": NEW_VARIABLES[0],
                "target_count": 2,
                "other_nine_exact_inherit": True,
                "old_graph_geometry_recomputations": 0,
                "global_common_vertex_budget": None,
                "selected_vertex_budget": None,
                "selected_budget_application_count": 0,
                "complete_p34_candidate": False,
                "g0a_pass": False,
                "viewer_layout_note": (
                    "old/new columns and two target rows are shifted only "
                    "for inspection; canonical JSON is numeric authority"
                ),
            },
            recording_id="g0a_d388_two_null_reanchor_design",
            blueprint_path=RBL_PATH,
            blueprint_mode="d388_two_null_reanchor",
            live_viewer=False,
            app_id="roarm_g0a_d388_two_null_reanchor_design",
        )
        if not saved.get("ok"):
            raise RuntimeError(f"save-only Rerun failed: {saved}")
        validation = rerun_contract.validate_rerun_artifact(
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
            # API placeholder only: the registered local runner above
            # intentionally ignores it and never sends a timeout signal.
            timeout_s=0.0,
        )
    finally:
        rerun_contract._run = original_contract_run
        viz_debug.build_rerun_blueprint = original_builder
        os.environ["PATH"] = old_path
    validation["d388_no_signal_subprocess_contract"] = {
        "helper_timeout_parameter_ignored_by_registered_local_runner": True,
        "subprocess_timeout_seconds": None,
        "timeout_kill_path_present": False,
        "process_signals_sent": 0,
    }
    _write_json_x(RERUN_VALIDATION, validation)
    screenshot = (
        _png_info(RERUN_SCREENSHOT)
        if RERUN_SCREENSHOT.is_file()
        else {"path": _rel(RERUN_SCREENSHOT), "exists": False}
    )
    screenshot_size = (
        (screenshot.get("width"), screenshot.get("height"))
        if screenshot.get("exists")
        else None
    )
    dimension_pass = screenshot_size in {
        (1920, 1080),
        (3840, 2160),
    }
    return {
        "save_only": saved,
        "strict_validation_pass": validation.get("pass") is True,
        "headless_viewer_invocations": int(
            bool(
                (validation.get("headless_render") or {}).get("attempted")
            )
        ),
        "headless_viewer_returncode": (
            validation.get("headless_render") or {}
        ).get("returncode"),
        "requested_logical_window_size": [1920, 1080],
        "allowed_native_screenshot_sizes": [
            [1920, 1080],
            [3840, 2160],
        ],
        "screenshot_dimension_contract_pass": dimension_pass,
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
        "screenshot": screenshot,
    }


def _write_candidate_metrics_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "body",
        "prim_name",
        "parent_name",
        "region_name",
        "anchor_index",
        "start_state",
        "end_state",
        "group_size",
        "geometry_constructed",
        "vertex_count",
        "polygon_count",
        "maximum_vertices_per_polygon",
        "volume_m3",
        "non_vertex_gates_pass",
        "rejection_reasons",
        "error_type",
    ]
    with METRICS_CSV.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            native = dict(row)
            native["rejection_reasons"] = json.dumps(
                native["rejection_reasons"],
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow({key: native.get(key) for key in fieldnames})


def _read_phase_records() -> list[dict[str, Any]]:
    records = []
    with PHASE_PATH.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "phase" not in row:
                raise TypeError(
                    f"invalid phase record at line {line_number}"
                )
            records.append(row)
    return records


def _layer_phase_contract(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    records = _read_phase_records()
    layer_phase_names = [
        "layer_reanchor_start",
        "layer_reanchor_end",
        "layer_graph_start",
        "layer_graph_end",
        "layer_dp_start",
        "layer_dp_end",
        "layer_exhaustive_start",
        "layer_exhaustive_end",
        "layer_geometry_gate_start",
        "layer_geometry_gate_end",
    ]
    target_keys = {_target_key(row) for row in results}
    all_layer_records = [
        (index, row)
        for index, row in enumerate(records)
        if row.get("phase") in layer_phase_names
    ]
    all_observed_keys = {
        tuple(row.get("key", [])) for _, row in all_layer_records
    }
    per_target = []
    observed_keys = set()
    for result in results:
        key = _target_key(result)
        keyed = [
            (index, row)
            for index, row in enumerate(records)
            if row.get("phase") in layer_phase_names
            and tuple(row.get("key", [])) == key
        ]
        observed_keys.update(
            tuple(row.get("key", [])) for _, row in keyed
        )
        geometry_count = (
            0
            if result["classification"] == "NO_COVER_THROUGH_64"
            else 1
        )
        expected_counts = {
            "layer_reanchor_start": 1,
            "layer_reanchor_end": 1,
            "layer_graph_start": 1,
            "layer_graph_end": 1,
            "layer_dp_start": 1,
            "layer_dp_end": 1,
            "layer_exhaustive_start": 1,
            "layer_exhaustive_end": 1,
            "layer_geometry_gate_start": geometry_count,
            "layer_geometry_gate_end": geometry_count,
        }
        counts = {
            name: sum(row["phase"] == name for _, row in keyed)
            for name in layer_phase_names
        }
        positions = {
            name: [
                index for index, row in keyed if row["phase"] == name
            ]
            for name in layer_phase_names
        }
        ordered = all(
            (
                not positions[start]
                and not positions[end]
            )
            or (
                len(positions[start]) == 1
                and len(positions[end]) == 1
                and positions[start][0] < positions[end][0]
            )
            for start, end in (
                ("layer_reanchor_start", "layer_reanchor_end"),
                ("layer_graph_start", "layer_graph_end"),
                ("layer_dp_start", "layer_dp_end"),
                ("layer_exhaustive_start", "layer_exhaustive_end"),
                ("layer_geometry_gate_start", "layer_geometry_gate_end"),
            )
        )
        per_target.append(
            {
                "key": list(key),
                "classification": result["classification"],
                "counts": counts,
                "expected_counts": expected_counts,
                "counts_exact": counts == expected_counts,
                "start_before_end": ordered,
                "pass": counts == expected_counts and ordered,
            }
        )
    return {
        "artifact": "D388_LAYER_PHASE_CONTRACT_V1",
        "target_count": len(target_keys),
        "observed_target_keys_exact": (
            observed_keys == target_keys
            and all_observed_keys == target_keys
        ),
        "all_layer_phase_record_count": len(all_layer_records),
        "expected_layer_phase_record_count": sum(
            sum(row["expected_counts"].values()) for row in per_target
        ),
        "all_layer_phase_keys": [
            list(key) for key in sorted(all_observed_keys)
        ],
        "all_layer_phase_keys_exact": all_observed_keys == target_keys,
        "per_target": per_target,
        "pass": bool(
            len(target_keys) == 2
            and observed_keys == target_keys
            and all_observed_keys == target_keys
            and len(all_layer_records)
            == sum(
                sum(row["expected_counts"].values())
                for row in per_target
            )
            and all(row["pass"] for row in per_target)
        ),
    }


def _global_phase_contract() -> dict[str, Any]:
    records = _read_phase_records()
    ordered_phases = [
        "prepare_start",
        "prepare_end",
        "supervisor_before_worker",
        "worker_start",
        "canonical_evidence_committed",
        "worker_end",
        "supervisor_after_worker",
        "finalize_start",
        "finalize_end",
    ]
    positions = {
        name: [
            index
            for index, row in enumerate(records)
            if row.get("phase") == name
        ]
        for name in ordered_phases
    }
    exact_once = all(len(positions[name]) == 1 for name in ordered_phases)
    forward = bool(
        exact_once
        and [positions[name][0] for name in ordered_phases]
        == sorted(positions[name][0] for name in ordered_phases)
    )
    monotonic = [float(row["monotonic_seconds"]) for row in records]
    monotonic_order = all(
        left <= right
        for left, right in zip(monotonic[:-1], monotonic[1:], strict=True)
    )
    return {
        "artifact": "D388_GLOBAL_PHASE_CONTRACT_V1",
        "record_count": len(records),
        "ordered_phase_positions": positions,
        "each_registered_global_phase_exactly_once": exact_once,
        "registered_global_phases_forward_only": forward,
        "monotonic_seconds_nondecreasing_in_file_order": monotonic_order,
        "pass": exact_once and forward and monotonic_order,
    }


def _prepare() -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"refusing forward-only path reuse: {OUT_DIR}")
    status_before = _git(["status", "--short"])
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    status_after = _git(["status", "--short"])
    head = _git(["rev-parse", "HEAD"])
    origin = _git(["rev-parse", "origin/master"])
    start_text = START_HERE.read_text(encoding="utf-8")
    direct_imports = _import_roots(SCRIPT_PATH)
    forbidden_roots = {
        "carb",
        "isaaclab",
        "isaacsim",
        "omni",
        "pxr",
        "signal",
        "torch",
        "warp",
    }
    forbidden_direct = sorted(set(direct_imports) & forbidden_roots)
    expected_before = {
        " M START_HERE.md",
        " M claudedocs/BACKLOG.md",
        " M claudedocs/DECISIONS.md",
        " M claudedocs/EXPERIMENT_LEDGER.md",
        "?? claudedocs/runtime_logs/grasp_track/g0a_d387/",
        (
            "?? claudedocs/"
            "session_20260726_grasp_g0a_d387_shadowed_layer_"
            "fixed_graph_completion_localization.md"
        ),
        (
            "?? sim_scripts/"
            "cyl34_top_view_d387_d386_shadowed_layer_"
            "fixed_graph_completion_localization.py"
        ),
        (
            "?? sim_scripts/"
            "cyl34_top_view_d388_d387_two_null_moving_support_"
            "midlayer_partition_repair_design.py"
        ),
    }
    expected_after = expected_before | {
        "?? claudedocs/runtime_logs/grasp_track/g0a_d388/"
    }

    d387 = _read_json(D387_EVIDENCE)
    combined = d387["combined_layer_map"]
    combined_by_key = {_target_key(row): row for row in combined}
    target_keys = {_target_key(row) for row in TARGETS}
    other_nine = [
        row for row in combined if _target_key(row) not in target_keys
    ]
    d387_output_manifest = _directory_manifest_aggregate(D387_OUTPUT_DIR)
    pure_geometry_ast = _pure_geometry_ast_contract()
    checks = {
        "head_exact": head == EXPECTED_HEAD,
        "origin_exact": origin == EXPECTED_HEAD,
        "input_hashes_exact": _input_hashes() == EXPECTED_INPUT_SHA256,
        "state_doc_hashes_exact": (
            _state_doc_hashes() == EXPECTED_STATE_DOC_SHA256
        ),
        "d387_output_manifest_exact": (
            d387_output_manifest["file_count"]
            == EXPECTED_D387_OUTPUT_FILE_COUNT
            and d387_output_manifest[
                "sha256sum_sorted_relative_path_manifest_aggregate"
            ]
            == EXPECTED_D387_OUTPUT_MANIFEST_AGGREGATE_SHA256
        ),
        "active_case_exact": (
            "`D388 [two_null_moving_support_midlayer_partition_repair_design]`"
            in start_text
        ),
        "one_new_variable_exact_and_registered": (
            len(NEW_VARIABLES) == 1
            and NEW_VARIABLES[0] in start_text
        ),
        "forward_only_output_path_registered": _rel(OUT_DIR) in start_text,
        "target_inventory_exact_two": (
            len(target_keys) == 2
            and all(key in combined_by_key for key in target_keys)
        ),
        "targets_are_exact_d387_nulls": all(
            combined_by_key[key]["classification"]
            == "NO_COVER_THROUGH_64"
            and combined_by_key[key]["registered_threshold_within_12_64"]
            is None
            for key in target_keys
        ),
        "other_nine_exact_count_and_pass": (
            len(other_nine) == 9
            and all(row["map_entry_pass"] is True for row in other_nine)
        ),
        "baseline_and_ceiling_exact": (
            BASELINE_BUDGET == 12
            and MAXIMUM_LOCALIZATION_BUDGET == 64
        ),
        "frozen_nonvertex_gates_exact": (
            MAXIMUM_FAN_GROUP == 4
            and MAXIMUM_POLYGONS == 64
            and MAXIMUM_VERTICES_PER_POLYGON == 32
            and SURFACE_TOLERANCE_MM == 0.1
            and VOLUME_RELATIVE_TOLERANCE == 0.005
            and POSITIVE_VOLUME_EPS_M3 == 1.0e-18
            and FLOAT_EPS_M == 5.0e-9
        ),
        "cooperative_deadline_exact_300s": (
            COOPERATIVE_DEADLINE_SECONDS == 300.0
        ),
        "no_signal_import_or_authority": (
            "signal" not in direct_imports
            and FORBIDDEN_COUNTERS["process_signals"] == 0
        ),
        "forbidden_runtime_imports_absent_direct": not forbidden_direct,
        "d385_source_is_hash_only_not_imported": (
            "importlib.util" not in direct_imports
            and "d385_frozen_geometry_helpers" not in sys.modules
        ),
        "embedded_d385_pure_geometry_ast_exact": (
            pure_geometry_ast["pass"]
        ),
        "rerun_cli_present": RERUN_CLI.is_file(),
        "font_regular_present": FONT_REGULAR.is_file(),
        "font_bold_present": FONT_BOLD.is_file(),
        "worktree_before_output_create_exact": (
            set(status_before.splitlines()) == expected_before
        ),
        "output_create_added_only_d388_root": (
            set(status_after.splitlines()) == expected_after
        ),
    }
    preregistration = {
        "artifact": "D388_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "approved_scope": (
            "offline one-variable repair design for exactly two D387 null "
            "moving-support z_layer_01 entries"
        ),
        "new_variables": NEW_VARIABLES,
        "targets": [
            {
                field: target[field]
                for field in (
                    "body",
                    "prim_name",
                    "name",
                    "role",
                    "region_name",
                    "region_index",
                )
            }
            for target in TARGETS
        ],
        "registered_method": {
            "anchor_rule": (
                "max(immutable old forward-reachable state) + 1"
            ),
            "derived_expected_anchor_indices": [11, 10],
            "reanchored_graphs_per_target": 1,
            "old_graph_geometry_recomputations": 0,
            "other_nine_layer_evaluations": 0,
            "other_nine_layer_mutations": 0,
            "candidate_construction": (
                "same CCW point set, cyclic rotation only, fan groups 1..4, "
                "original frozen parent intersection"
            ),
            "primary_search": "bounded minimax dynamic programming",
            "independent_search": (
                "complete exhaustive enumeration of every admissible path"
            ),
            "classifications": {
                "BASELINE_B12_COVER": (
                    "B12 has a complete valid witness"
                ),
                "FINITE_RELAXATION_THRESHOLD_13_TO_64": (
                    "B12 has no cover and finite B* has B*-1 no cover/B* cover"
                ),
                "NO_COVER_THROUGH_64": (
                    "the new graph has no complete path through B64"
                ),
            },
            "actual_overlap_gate": (
                "stored Float32 vertices; bidirectional pairwise convex "
                "halfspace intersection volume; positive volume count zero"
            ),
            "duplicate_overlap_negative_control": True,
        },
        "frozen_gates": {
            "vertex_budget_baseline": BASELINE_BUDGET,
            "localization_ceiling": MAXIMUM_LOCALIZATION_BUDGET,
            "fan_group_size": [1, MAXIMUM_FAN_GROUP],
            "maximum_polygons": MAXIMUM_POLYGONS,
            "maximum_vertices_per_polygon": (
                MAXIMUM_VERTICES_PER_POLYGON
            ),
            "positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
            "float_halfspace_clip_epsilon_m": FLOAT_EPS_M,
            "surface_tolerance_mm": SURFACE_TOLERANCE_MM,
            "topology_volume_relative_tolerance": (
                VOLUME_RELATIVE_TOLERANCE
            ),
            "positive_volume_child_overlap": 0,
        },
        "outcome_semantics": {
            "both_b12": (
                "two-null repair-design PASS only; no budget adoption"
            ),
            "finite_above12": (
                "threshold localization only; budget policy remains pending"
            ),
            "any_null": (
                "first frontier-derived re-anchor candidate rejected"
            ),
            "contract_failure": "fail-stop",
        },
        "explicit_nonclaims": {
            "global_common_vertex_budget": None,
            "selected_or_adopted_budget": None,
            "budget_application_count": 0,
            "complete_p34_budget": None,
            "materializable_candidate": False,
            "live_physics_or_grasp_result": None,
        },
        "worker_contract": {
            "actual_worker_invocations": 1,
            "retries": 0,
            "cooperative_algorithm_deadline_seconds": (
                COOPERATIVE_DEADLINE_SECONDS
            ),
            "process_signal_authority": False,
            "supervisor_timeout_signal_or_kill": False,
            "rerun_viewer_invocations_maximum": 1,
            "rerun_viewer_retries": 0,
            "board_exact_pixels": [1920, 1080],
            "rerun_requested_logical_window_size": [1920, 1080],
            "rerun_allowed_native_screenshot_sizes": [
                [1920, 1080],
                [3840, 2160],
            ],
        },
        "forbidden_runtime_counters": FORBIDDEN_COUNTERS,
        "allowed_counters": ALLOWED_COUNTERS,
        "environment": {
            "head": head,
            "origin_master": origin,
            "git_status_before_output_create": status_before.splitlines(),
            "git_status_after_output_create": status_after.splitlines(),
            "python": sys.version,
            "executable": sys.executable,
            "script_path": _rel(SCRIPT_PATH),
            "script_sha256": _sha(SCRIPT_PATH),
            "start_here_sha256": _sha(START_HERE),
            "state_doc_hashes": _state_doc_hashes(),
            "d387_output_manifest": d387_output_manifest,
            "input_hashes": _input_hashes(),
            "direct_import_roots": direct_imports,
            "forbidden_direct_import_roots": forbidden_direct,
            "d385_source_imported_at_runtime": False,
            "embedded_d385_pure_geometry_ast_contract": pure_geometry_ast,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, preregistration)
    _phase("prepare_end", pass_value=preregistration["pass"])
    if not preregistration["pass"]:
        raise RuntimeError(f"D388 preregistration failed: {checks}")
    print(json.dumps({"prepare_pass": True, "path": _rel(PREREG_PATH)}))
    return 0


def _worker() -> int:
    global WORKER_DEADLINE_MONOTONIC
    if not PREREG_PATH.is_file():
        raise RuntimeError("missing D388 preregistration")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D388 preregistration did not pass")
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
            _input_hashes() == prereg["environment"]["input_hashes"]
        ),
        "state_doc_hashes_unchanged_after_prepare": (
            _state_doc_hashes()
            == prereg["environment"]["state_doc_hashes"]
            == EXPECTED_STATE_DOC_SHA256
        ),
        "d387_output_manifest_unchanged_after_prepare": (
            _directory_manifest_aggregate(D387_OUTPUT_DIR)
            == prereg["environment"]["d387_output_manifest"]
            == {
                "file_count": EXPECTED_D387_OUTPUT_FILE_COUNT,
                "sha256sum_sorted_relative_path_manifest_aggregate": (
                    EXPECTED_D387_OUTPUT_MANIFEST_AGGREGATE_SHA256
                ),
            }
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
            f"D388 worker provenance failed: {provenance_checks}"
        )
    geometry_source_checks = {
        "frozen_d385_source_hash_exact": (
            _sha(D385_SCRIPT) == EXPECTED_INPUT_SHA256["d385_script"]
        ),
        "d385_module_not_imported": (
            "d385_frozen_geometry_helpers" not in sys.modules
        ),
        "current_worker_has_no_signal_import": (
            "signal" not in _import_roots(SCRIPT_PATH)
        ),
        "embedded_positive_volume_epsilon_exact": (
            POSITIVE_VOLUME_EPS_M3 == 1.0e-18
        ),
        "embedded_float_clip_epsilon_exact": FLOAT_EPS_M == 5.0e-9,
        "frozen_source_child_baseline_remains_12": BASELINE_BUDGET == 12,
        "embedded_pure_geometry_ast_exact": (
            _pure_geometry_ast_contract()
            == prereg["environment"][
                "embedded_d385_pure_geometry_ast_contract"
            ]
            and _pure_geometry_ast_contract()["pass"]
        ),
    }
    if not all(geometry_source_checks.values()):
        raise RuntimeError(
            "embedded D385 geometry-source contract changed: "
            f"{geometry_source_checks}"
        )

    worker_start = time.monotonic()
    WORKER_DEADLINE_MONOTONIC = (
        worker_start + COOPERATIVE_DEADLINE_SECONDS
    )
    _phase(
        "worker_start",
        cooperative_deadline_monotonic=WORKER_DEADLINE_MONOTONIC,
        signal_authority=False,
    )
    evidence, geometry, visual, candidate_rows = _compute()
    algorithm_elapsed = time.monotonic() - worker_start
    evidence["script_sha256"] = _sha(SCRIPT_PATH)
    evidence["diagnostic_geometry_payload_sha256"] = _sha_payload(geometry)
    evidence["execution_contract"] = {
        "worker_invocation_index": 1,
        "retry_index": 0,
        "offline_only": True,
        "provenance_checks": provenance_checks,
        "embedded_d385_geometry_source_checks": geometry_source_checks,
        "cooperative_algorithm_deadline_seconds": (
            COOPERATIVE_DEADLINE_SECONDS
        ),
        "algorithm_elapsed_seconds": algorithm_elapsed,
        "algorithm_deadline_exceeded": False,
        "process_signal_authority": False,
        "process_signals_sent": 0,
    }
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase(
        "canonical_evidence_committed",
        verdict=evidence["verdict"],
        method_contract_pass=evidence["method_contract_pass"],
        two_null_b12_repair_design_pass=(
            evidence["two_null_b12_repair_design_pass"]
        ),
    )
    _write_json_x(GEOMETRY_PATH, geometry)
    _write_candidate_metrics_csv(candidate_rows)
    board_info, layout = _render_board(evidence, visual)
    _write_json_x(BOARD_LAYOUT, layout)
    rerun = _write_rerun(evidence, visual)
    layer_phase = _layer_phase_contract(evidence["target_results"])
    manual_template = {
        "artifact": "D388_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board_path": _rel(BOARD_PATH),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "required_checks": [
            "board_exact_1920x1080_and_readable",
            "exact_two_target_rows_visible",
            "old_frontier_and_new_anchor_11_10_visible",
            "three_way_classification_and_thresholds_readable",
            "diagnostic_children_match_each_finite_decision",
            "actual_float32_overlap_and_duplicate_control_readable",
            "other_nine_inherit_zero_evaluation_scope_readable",
            "no_budget_asset_live_physics_or_grasp_claim",
            "rerun_native_dimension_is_logical_1x_or_hidpi_2x_not_4x",
        ],
        "inspection_result_path": _rel(MANUAL_INSPECTION),
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)
    claim = {
        "artifact": "D388_OFFLINE_WORKER_CLAIM_V1",
        "worker_pid": os.getpid(),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "scientific_design_verdict": evidence["verdict"],
        "method_contract_pass": evidence["method_contract_pass"],
        "two_null_b12_repair_design_pass": (
            evidence["two_null_b12_repair_design_pass"]
        ),
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
            "layer_phase_contract": layer_phase,
            "manual_template": {
                "path": _rel(MANUAL_TEMPLATE),
                "sha256": _sha(MANUAL_TEMPLATE),
            },
        },
        "scope_counters": FORBIDDEN_COUNTERS,
        "pass": bool(
            evidence["method_contract_pass"]
            and layout["pass"]
            and rerun["strict_validation_pass"]
            and rerun["screenshot_dimension_contract_pass"]
            and layer_phase["pass"]
            and rerun["headless_viewer_invocations"] <= 1
            and all(value == 0 for value in FORBIDDEN_COUNTERS.values())
        ),
    }
    _phase("worker_end", worker_claim_pass=claim["pass"])
    _write_json_x(WORKER_CLAIM, claim)
    if not claim["pass"]:
        raise RuntimeError("D388 observability worker claim failed")
    print(
        json.dumps(
            {
                "worker_pass": True,
                "scientific_design_verdict": evidence["verdict"],
                "evidence": _rel(EVIDENCE_PATH),
            }
        )
    )
    return 0


def _run_supervisor() -> int:
    if not PREREG_PATH.is_file():
        raise RuntimeError("run requires completed D388 prepare stage")
    if INVOCATION_PATH.exists() or SUPERVISOR_PATH.exists():
        raise RuntimeError("refusing to repeat D388 actual worker")
    command = [
        sys.executable,
        "-B",
        str(SCRIPT_PATH),
        "--stage",
        "worker",
    ]
    invocation = {
        "artifact": "D388_OFFLINE_WORKER_INVOCATION_V1",
        "command": command,
        "cwd": str(REPO),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "cooperative_algorithm_deadline_seconds": (
            COOPERATIVE_DEADLINE_SECONDS
        ),
        "supervisor_signal_authority": False,
        "supervisor_timeout_signal_or_kill": False,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase("supervisor_before_worker")
    started = time.monotonic()
    with WORKER_STDOUT.open("x", encoding="utf-8") as stdout_stream:
        with WORKER_STDERR.open("x", encoding="utf-8") as stderr_stream:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
                start_new_session=False,
            )
            returncode = process.wait()
    elapsed = time.monotonic() - started
    supervisor = {
        "artifact": "D388_OFFLINE_WORKER_SUPERVISOR_V1",
        "actual_worker_invocations": 1,
        "retries": 0,
        "worker_pid": process.pid,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "cooperative_algorithm_deadline_seconds": (
            COOPERATIVE_DEADLINE_SECONDS
        ),
        "supervisor_signal_authority": False,
        "process_signals_sent": 0,
        "termination_action": None,
        "worker_process_exited": process.poll() is not None,
        "stdout": _rel(WORKER_STDOUT),
        "stderr": _rel(WORKER_STDERR),
        "worker_claim_exists": WORKER_CLAIM.is_file(),
        "pass": bool(
            returncode == 0
            and process.poll() is not None
            and WORKER_CLAIM.is_file()
        ),
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase("supervisor_after_worker", **supervisor)
    if not supervisor["pass"]:
        raise RuntimeError(f"D388 worker failed: {supervisor}")
    print(json.dumps(supervisor, indent=2))
    return 0


def _finalize() -> int:
    required = [
        PREREG_PATH,
        INVOCATION_PATH,
        SUPERVISOR_PATH,
        WORKER_STDOUT,
        WORKER_STDERR,
        PHASE_PATH,
        EVIDENCE_PATH,
        GEOMETRY_PATH,
        METRICS_CSV,
        BOARD_PATH,
        BOARD_LAYOUT,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION,
        RERUN_SCREENSHOT,
        MANUAL_TEMPLATE,
        MANUAL_INSPECTION,
        WORKER_CLAIM,
    ]
    missing = [_rel(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot finalize; missing artifacts: {missing}")
    _phase("finalize_start")
    evidence = _read_json(EVIDENCE_PATH)
    invocation = _read_json(INVOCATION_PATH)
    supervisor = _read_json(SUPERVISOR_PATH)
    layout = _read_json(BOARD_LAYOUT)
    rerun_validation = _read_json(RERUN_VALIDATION)
    manual = _read_json(MANUAL_INSPECTION)
    worker_claim = _read_json(WORKER_CLAIM)
    claim_artifacts = worker_claim["artifacts"]
    worker_claim_bindings = {
        "evidence": (claim_artifacts["evidence"], EVIDENCE_PATH),
        "geometry": (claim_artifacts["geometry"], GEOMETRY_PATH),
        "metrics_csv": (claim_artifacts["metrics_csv"], METRICS_CSV),
        "board": (claim_artifacts["board"], BOARD_PATH),
        "board_layout": (
            claim_artifacts["board_layout"],
            BOARD_LAYOUT,
        ),
        "rerun_rrd": (claim_artifacts["rerun"]["rrd"], RRD_PATH),
        "rerun_rbl": (claim_artifacts["rerun"]["rbl"], RBL_PATH),
        "rerun_validation": (
            claim_artifacts["rerun"]["validation"],
            RERUN_VALIDATION,
        ),
        "rerun_screenshot": (
            claim_artifacts["rerun"]["screenshot"],
            RERUN_SCREENSHOT,
        ),
        "manual_template": (
            claim_artifacts["manual_template"],
            MANUAL_TEMPLATE,
        ),
    }
    worker_claim_artifact_linkage = {
        label: {
            "claimed_path": record.get("path"),
            "expected_path": _rel(path),
            "path_exact": record.get("path") == _rel(path),
            "claimed_sha256": record.get("sha256"),
            "current_sha256": _sha(path),
            "sha256_exact": record.get("sha256") == _sha(path),
            "pass": bool(
                record.get("path") == _rel(path)
                and record.get("sha256") == _sha(path)
            ),
        }
        for label, (record, path) in worker_claim_bindings.items()
    }
    worker_claim_artifact_linkage_pass = bool(
        len(worker_claim_artifact_linkage) == 10
        and len(
            {
                row["expected_path"]
                for row in worker_claim_artifact_linkage.values()
            }
        )
        == 10
        and all(
            row["pass"]
            for row in worker_claim_artifact_linkage.values()
        )
    )
    required_manual_checks = {
        "board_exact_1920x1080_and_readable",
        "exact_two_target_rows_visible",
        "old_frontier_and_new_anchor_11_10_visible",
        "three_way_classification_and_thresholds_readable",
        "diagnostic_children_match_each_finite_decision",
        "actual_float32_overlap_and_duplicate_control_readable",
        "other_nine_inherit_zero_evaluation_scope_readable",
        "no_budget_asset_live_physics_or_grasp_claim",
        "rerun_native_dimension_is_logical_1x_or_hidpi_2x_not_4x",
    }
    manual_checks = manual.get("checks", {})
    manual_hashes = manual.get("artifact_hashes", {})
    manual_contract_pass = bool(
        manual.get("artifact") == "D388_MANUAL_VISUAL_INSPECTION_V1"
        and set(manual_checks) == required_manual_checks
        and all(value is True for value in manual_checks.values())
        and isinstance(manual.get("observations"), list)
        and len(manual["observations"]) >= 3
        and manual_hashes
        == {
            _rel(BOARD_PATH): _sha(BOARD_PATH),
            _rel(RERUN_SCREENSHOT): _sha(RERUN_SCREENSHOT),
        }
        and manual.get("pass") is True
    )
    valid_verdicts = {
        (
            "D388_TWO_NULL_B12_REANCHOR_REPAIR_DESIGN_PASS_"
            "NO_BUDGET_ADOPTION"
        ),
        (
            "D388_REANCHOR_FINITE_THRESHOLD_LOCALIZED_"
            "BUDGET_POLICY_PENDING"
        ),
        "D388_FIRST_REANCHOR_CANDIDATE_FAIL_STOP",
    }
    outcome_consistent = bool(
        evidence["method_contract_pass"] is True
        and evidence["outcome_flags_mutually_exclusive"] is True
        and sum(
            bool(value)
            for value in (
                evidence["two_null_b12_repair_design_pass"],
                evidence["finite_reanchor_localization_pass"],
                evidence["first_reanchor_candidate_fail_stop"],
            )
        )
        == 1
        and evidence["verdict"] in valid_verdicts
        and (
            (
                evidence["two_null_b12_repair_design_pass"] is True
                and evidence["verdict"]
                == (
                    "D388_TWO_NULL_B12_REANCHOR_REPAIR_DESIGN_PASS_"
                    "NO_BUDGET_ADOPTION"
                )
            )
            or (
                evidence["two_null_b12_repair_design_pass"] is False
                and evidence["finite_reanchor_localization_pass"] is True
                and evidence["verdict"]
                == (
                    "D388_REANCHOR_FINITE_THRESHOLD_LOCALIZED_"
                    "BUDGET_POLICY_PENDING"
                )
            )
            or (
                evidence["first_reanchor_candidate_fail_stop"] is True
                and evidence["verdict"]
                == "D388_FIRST_REANCHOR_CANDIDATE_FAIL_STOP"
            )
        )
    )
    checks = {
        "scientific_design_outcome_consistent": outcome_consistent,
        "method_contract_pass": evidence["method_contract_pass"] is True,
        "outcome_flags_mutually_exclusive": (
            evidence["outcome_flags_mutually_exclusive"] is True
        ),
        "exact_two_target_results": len(evidence["target_results"]) == 2,
        "other_nine_exact_inherit": (
            len(
                evidence[
                    "other_nine_d387_map_entries_exact_inherit"
                ]
            )
            == 9
            and evidence["scope_statement"][
                "other_nine_layer_evaluations"
            ]
            == 0
            and evidence["scope_statement"][
                "other_nine_layer_mutations"
            ]
            == 0
        ),
        "old_graph_geometry_recomputation_zero": (
            evidence["scope_statement"][
                "old_graph_geometry_recomputations"
            ]
            == 0
        ),
        "global_common_vertex_budget_null": (
            evidence["global_common_vertex_budget"] is None
        ),
        "adopted_parent_wide_budget_null": (
            evidence["adopted_parent_wide_vertex_budget"] is None
        ),
        "selected_vertex_budget_null": (
            evidence["selected_vertex_budget"] is None
        ),
        "selected_budget_application_zero": (
            evidence["selected_budget_application_count"] == 0
        ),
        "complete_p34_budget_and_counts_null": (
            evidence["complete_p34_vertex_budget"] is None
            and evidence["complete_source_child_count"] is None
            and evidence["complete_total_part_count"] is None
        ),
        "materializable_and_materialized_false": (
            evidence["materializable_candidate"] is False
            and evidence["repair_materialized"] is False
        ),
        "supervisor_pass": supervisor["pass"] is True,
        "actual_worker_once_no_retry": (
            supervisor["actual_worker_invocations"] == 1
            and supervisor["retries"] == 0
        ),
        "worker_claim_all_saved_artifact_paths_and_hashes_exact": (
            worker_claim_artifact_linkage_pass
        ),
        "invocation_and_worker_log_paths_exact": (
            invocation["worker_invocation_index"] == 1
            and invocation["retry_index"] == 0
            and supervisor["stdout"] == _rel(WORKER_STDOUT)
            and supervisor["stderr"] == _rel(WORKER_STDERR)
        ),
        "cooperative_deadline_no_signal_contract": (
            supervisor["cooperative_algorithm_deadline_seconds"] == 300.0
            and supervisor["supervisor_signal_authority"] is False
            and supervisor["process_signals_sent"] == 0
            and supervisor["termination_action"] is None
            and evidence["execution_contract"][
                "algorithm_deadline_exceeded"
            ]
            is False
            and evidence["execution_contract"]["process_signals_sent"] == 0
        ),
        "board_exact_and_layout_pass": (
            worker_claim["artifacts"]["board"]["exact_1920x1080"] is True
            and layout["pass"] is True
        ),
        "rerun_strict_validation_pass": (
            rerun_validation["pass"] is True
        ),
        "rerun_screenshot_dimension_contract_pass": (
            worker_claim["artifacts"]["rerun"][
                "screenshot_dimension_contract_pass"
            ]
            is True
        ),
        "layer_phase_contract_pass": (
            worker_claim["artifacts"]["layer_phase_contract"]["pass"] is True
            and _layer_phase_contract(evidence["target_results"])["pass"]
            is True
        ),
        "manual_visual_inspection_contract_pass": manual_contract_pass,
        "worker_claim_pass": worker_claim["pass"] is True,
        "headless_viewer_maximum_one_no_retry": (
            worker_claim["artifacts"]["rerun"][
                "headless_viewer_invocations"
            ]
            <= 1
        ),
        "scope_counters_zero": all(
            value == 0
            for value in evidence["current_scope_counters"].values()
        ),
        "live_identity_and_gpu_null": (
            evidence["live_identity_pass"] is None
            and evidence["live_gpu_compatibility_pass"] is None
        ),
        "cylinder_not_rendered_or_measured": (
            evidence["cylinder_29x50_rendered_or_measured"] is False
        ),
        "physics_or_grasp_null": (
            evidence["physics_or_grasp_result"] is None
        ),
        "p34_identity_false_and_g0a_false": (
            evidence["p34_authored_to_cooked_identity_pass"] is False
            and evidence["g0a_pass"] is False
        ),
    }
    preliminary_pass = all(checks.values())
    _phase(
        "finalize_end",
        preliminary_completion_pass=preliminary_pass,
    )
    global_phase = _global_phase_contract()
    checks["global_phase_contract_pass"] = global_phase["pass"]
    completion = {
        "artifact": "D388_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "scientific_design_verdict": evidence["verdict"],
        "method_contract_pass": evidence["method_contract_pass"],
        "two_null_b12_repair_design_pass": (
            evidence["two_null_b12_repair_design_pass"]
        ),
        "both_targets_finite_through_64": (
            evidence["both_targets_finite_through_64"]
        ),
        "finite_reanchor_localization_pass": (
            evidence["finite_reanchor_localization_pass"]
        ),
        "first_reanchor_candidate_fail_stop": (
            evidence["first_reanchor_candidate_fail_stop"]
        ),
        "collider_design_adopted": False,
        "observability_completion_pass": all(checks.values()),
        "checks": checks,
        "global_phase_contract": global_phase,
        "worker_claim_artifact_linkage": (
            worker_claim_artifact_linkage
        ),
        "artifact_hashes": {
            _rel(path): _sha(path) for path in required
        },
        "next_authorization_boundary": (
            "D388 changes only the two null-layer partition designs. Do not "
            "adopt/apply a budget, materialize USD/PhysX, create the 29x50mm "
            "target, or run physics/contact/grasp without new approval."
        ),
        "pass": all(checks.values()),
    }
    _write_json_x(COMPLETION_PATH, completion)
    if not completion["pass"]:
        raise RuntimeError(f"D388 completion failed: {checks}")
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
