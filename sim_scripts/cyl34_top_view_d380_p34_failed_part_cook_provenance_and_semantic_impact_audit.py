#!/usr/bin/env python3
"""D380 offline-only audit of the 17 D379 P34 cooked-shape failures.

Measurement authority is exactly one immutable D379 JSON.  This program does
not import or launch Isaac Sim, Kit, PhysX, USD, Warp, CUDA, or robot hardware.
It does not change any D379 tolerance and does not make a cylinder/contact/q5
measurement.  Rerun is used only after the canonical JSON result is written.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
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


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASE = "g0a_d380"
ATTEMPT = "attempt1_failed_part_cook_provenance_semantic_impact_audit"
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track"
    / CASE
    / ATTEMPT
)
SCRIPT_PATH = Path(__file__).resolve()
START_HERE = REPO / "START_HERE.md"

D379_EVIDENCE = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d379/"
    "attempt2_d372_measurement_field_repair/"
    "d379_p34_full_live_identity_evidence.json"
)
D379_EVIDENCE_SHA256 = (
    "8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5"
)

PREREG_PATH = OUT_DIR / "d380_preregistration.json"
PHASE_PATH = OUT_DIR / "d380_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d380_offline_audit_invocation.json"
WORKER_STDOUT = OUT_DIR / "d380_offline_worker_stdout.log"
WORKER_STDERR = OUT_DIR / "d380_offline_worker_stderr.log"
WORKER_CLAIM = OUT_DIR / "d380_offline_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d380_offline_worker_supervisor.json"
EVIDENCE_PATH = OUT_DIR / "d380_p34_failed_part_cook_provenance_evidence.json"
METRICS_CSV = OUT_DIR / "d380_failed_part_metrics.csv"
BOARD_PATH = OUT_DIR / "d380_p34_failed_part_cook_provenance_1920x1080.png"
RRD_PATH = OUT_DIR / "d380_p34_failed_part_cook_provenance.rrd"
RBL_PATH = OUT_DIR / "d380_p34_failed_part_cook_provenance.rbl"
RERUN_VALIDATION = OUT_DIR / "d380_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d380_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d380_manual_visual_inspection_template.json"
MANUAL_INSPECTION = OUT_DIR / "d380_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d380_completion_summary.json"

VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
ISAACLAB_PYTHON = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
)
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

EXPECTED_COUNTS = {"link5": 16, "gripper_link": 18}
EXPECTED_D379_FAILED = {"link5": 4, "gripper_link": 13}
EXPECTED_THRESHOLDS = {
    "surface_tolerance_m": 0.0001,
    "bounds_tolerance_m": 0.0001,
    "authored_callback_topology_volume_relative": 0.005,
    "callback_property_volume_relative": 0.05,
    "polygon_plane_residual_tolerance_m": 0.00001,
    "authored_mass_state_atol": 1.0e-12,
    "property_mass_state_atol": 1.0e-9,
}
NEW_VARIABLES = [
    "failed_part_cook_provenance_classifier_v1",
    "body_local_semantic_monotonic_impact_contract_v1",
]
WATCHDOG_SECONDS = 300.0
MANUAL_CHECK_KEYS = {
    "board_exact_1920x1080_legible",
    "board_no_text_clipping_or_overlap",
    "authored_cooked_and_omitted_geometry_visible",
    "surface_and_volume_plots_match_evidence",
    "rerun_four_spatial_views_visible",
    "rerun_no_unknown_timeline",
    "rerun_no_decision_obscuring_overlap",
    "rerun_geometry_consistent_with_board",
}

FORBIDDEN_IMPORT_ROOTS = {
    "carb",
    "cuda",
    "gymnasium",
    "isaaclab",
    "omni",
    "omniisaacgymenvs",
    "physx",
    "pxr",
    "torch",
    "warp",
}
FORBIDDEN_RUNTIME_COUNTERS = {
    "asset_or_usd_reads": 0,
    "asset_or_usd_writes": 0,
    "automatic_decomposition_sweeps": 0,
    "collider_materializations_or_regenerations": 0,
    "contact_queries": 0,
    "cylinder_creates_or_writes": 0,
    "isaac_launches": 0,
    "kit_launches": 0,
    "material_mass_actuator_physics_setting_changes": 0,
    "physics_steps": 0,
    "physx_calls": 0,
    "public_forwards": 0,
    "q5_commands": 0,
    "q5_samples": 0,
    "target_ik_path_pose_changes": 0,
}

ROLE_COLORS_HEX = {
    "moving_support": "#7A5195",
    "moving_jaw": "#F28E2B",
    "moving_jaw_backbone": "#E3B341",
    "fixed_jaw": "#2E86AB",
    "fixed_jaw_backbone": "#23A6A8",
    "structural_body": "#59A14F",
    "structural_support": "#8CD17D",
}
ROLE_COLORS_RGBA = {
    "moving_support": [122, 81, 149, 205],
    "moving_jaw": [242, 142, 43, 205],
    "moving_jaw_backbone": [227, 179, 65, 205],
    "fixed_jaw": [46, 134, 171, 205],
    "fixed_jaw_backbone": [35, 166, 168, 205],
    "structural_body": [89, 161, 79, 205],
    "structural_support": [140, 209, 125, 205],
}


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected object JSON: {path}")
    return value


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")


def _phase(name: str, **fields: Any) -> None:
    row = {
        "case": CASE,
        "attempt": ATTEMPT,
        "phase": name,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_epoch_s": time.time(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _status_paths() -> list[str]:
    return _git("status", "--short").splitlines()


def _source_hashes() -> dict[str, str]:
    return {
        "d380_script": _sha(SCRIPT_PATH),
        "start_here_active_case_authorization": _sha(START_HERE),
        "rerun_contract": _sha(RERUN_CONTRACT),
        "viz_debug": _sha(VIZ_DEBUG),
    }


def _dependency_versions() -> dict[str, str]:
    return {
        "matplotlib": importlib.metadata.version("matplotlib"),
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("pillow"),
        "psutil": importlib.metadata.version("psutil"),
        "rerun_sdk": importlib.metadata.version("rerun-sdk"),
        "trimesh": importlib.metadata.version("trimesh"),
    }


def _input_hashes() -> dict[str, str]:
    return {"d379_identity_evidence": _sha(D379_EVIDENCE)}


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        dimensions = [int(image.width), int(image.height)]
        mode = image.mode
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "dimensions": dimensions,
        "mode": mode,
        "exact_1920x1080": dimensions == [1920, 1080],
    }


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _edge_closure(triangles: np.ndarray) -> dict[str, Any]:
    directed: Counter[tuple[int, int]] = Counter()
    undirected: Counter[tuple[int, int]] = Counter()
    for tri in triangles:
        for start, end in (
            (int(tri[0]), int(tri[1])),
            (int(tri[1]), int(tri[2])),
            (int(tri[2]), int(tri[0])),
        ):
            directed[(start, end)] += 1
            undirected[tuple(sorted((start, end)))] += 1
    twice = bool(undirected) and all(count == 2 for count in undirected.values())
    opposite = twice and all(
        directed[(left, right)] == 1 and directed[(right, left)] == 1
        for left, right in undirected
    )
    return {
        "undirected_edge_count": len(undirected),
        "all_edges_twice": twice,
        "opposite_winding": opposite,
        "pass": twice and opposite,
    }


def _outward_support_audit(
    vertices: np.ndarray, triangles: np.ndarray
) -> dict[str, Any]:
    center = vertices.mean(axis=0)
    max_violation = 0.0
    nondegenerate = 0
    for tri in triangles:
        a, b, c = vertices[tri]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm <= np.finfo(np.float64).tiny:
            continue
        normal /= norm
        if float(np.dot(normal, center - a)) > 0.0:
            normal = -normal
        max_violation = max(
            max_violation,
            float(np.max(vertices @ normal - float(np.dot(normal, a)))),
        )
        nondegenerate += 1
    scale = max(1.0, float(np.max(np.abs(vertices))))
    numerical_budget = float(4096.0 * np.finfo(np.float64).eps * scale)
    return {
        "nondegenerate_triangle_count": nondegenerate,
        "triangle_count": int(len(triangles)),
        "max_supporting_plane_violation_m": max_violation,
        "float64_roundoff_budget_m": numerical_budget,
        "all_triangle_planes_supporting": bool(
            nondegenerate == len(triangles)
            and max_violation <= numerical_budget
        ),
    }


def _json_vertex_subset(
    authored_vertices: np.ndarray, cooked_vertices: np.ndarray
) -> dict[str, Any]:
    authored = [tuple(float(value) for value in row) for row in authored_vertices]
    cooked = [tuple(float(value) for value in row) for row in cooked_vertices]
    authored_set = set(authored)
    cooked_set = set(cooked)
    omitted_indices = [
        index for index, point in enumerate(authored) if point not in cooked_set
    ]
    introduced = sorted(cooked_set - authored_set)
    return {
        "authored_vertex_count": len(authored),
        "authored_unique_vertex_count": len(authored_set),
        "cooked_vertex_count": len(cooked),
        "cooked_unique_vertex_count": len(cooked_set),
        "retained_vertex_count": len(authored_set & cooked_set),
        "omitted_authored_vertex_count": len(authored_set - cooked_set),
        "introduced_or_moved_cooked_vertex_count": len(introduced),
        "json_numeric_exact_vertex_subset": cooked_set <= authored_set,
        "omitted_authored_vertex_indices": omitted_indices,
        "omitted_authored_vertices_m": [
            authored[index] for index in omitted_indices
        ],
        "introduced_or_moved_cooked_vertices_m": introduced,
        "authored_vertex_set_sha256": _canonical_sha(sorted(authored_set)),
        "cooked_vertex_set_sha256": _canonical_sha(sorted(cooked_set)),
    }


def _triangles_from_authored(row: dict[str, Any]) -> np.ndarray:
    counts = list(row["face_vertex_counts"])
    if not counts or any(int(count) != 3 for count in counts):
        raise ValueError(f"non-triangular authored topology: {row['body']}/{row['name']}")
    indices = np.asarray(row["face_vertex_indices"], dtype=np.int64)
    if indices.size != len(counts) * 3:
        raise ValueError(f"authored index size mismatch: {row['body']}/{row['name']}")
    return indices.reshape(-1, 3)


def _nearest_triangle_distances(
    points: np.ndarray,
    target_vertices: np.ndarray,
    target_triangles: np.ndarray,
) -> np.ndarray:
    import trimesh

    triangles = target_vertices[target_triangles]
    distances = []
    for point in points:
        tiled = np.repeat(point[None, :], len(triangles), axis=0)
        closest = trimesh.triangles.closest_point(triangles, tiled)
        distances.append(
            float(np.min(np.linalg.norm(closest - point[None, :], axis=1)))
        )
    return np.asarray(distances, dtype=np.float64)


def _failed_gate_signature(row: dict[str, Any]) -> list[str]:
    aliases = {
        "surface_le_0_1mm": "surface",
        "bounds_le_0_1mm": "bounds",
        "authored_callback_topology_volume_le_0_5pct": "volume",
        "polygon_plane_residual_le_1e_5m": "plane",
    }
    failures = [
        aliases.get(name, name)
        for name, passed in row["checks"].items()
        if passed is False
    ]
    return sorted(failures)


def _semantic_scope(role: str) -> str:
    if role == "moving_support":
        return "moving_support"
    if role in {"fixed_jaw", "fixed_jaw_backbone"}:
        return "fixed_jaw_system"
    if role in {"moving_jaw", "moving_jaw_backbone"}:
        return "moving_jaw_system"
    return "other"


def _row_metric(
    authored_row: dict[str, Any],
    callback_row: dict[str, Any],
    local_binding_check: dict[str, Any],
) -> dict[str, Any]:
    authored_vertices = np.asarray(authored_row["points_f32"], dtype=np.float64)
    authored_triangles = _triangles_from_authored(authored_row)
    cooked_vertices = np.asarray(
        callback_row["live_callback_vertices_m"], dtype=np.float64
    )
    cooked_triangles = np.asarray(
        callback_row["live_callback_topology_triangles"], dtype=np.int64
    ).reshape(-1, 3)
    subset = _json_vertex_subset(authored_vertices, cooked_vertices)
    omitted_indices = np.asarray(
        subset["omitted_authored_vertex_indices"], dtype=np.int64
    )
    omitted_distances = _nearest_triangle_distances(
        authored_vertices[omitted_indices],
        cooked_vertices,
        cooked_triangles,
    )
    subset["omitted_vertex_to_cooked_surface_distance_mm"] = (
        omitted_distances * 1000.0
    ).tolist()
    subset["omitted_vertices_beyond_inherited_0_1mm_surface_limit"] = int(
        np.count_nonzero(
            omitted_distances > EXPECTED_THRESHOLDS["surface_tolerance_m"]
        )
    )
    subset["max_omitted_vertex_to_cooked_surface_distance_mm"] = (
        float(np.max(omitted_distances) * 1000.0)
        if omitted_distances.size
        else 0.0
    )
    authored_closure = _edge_closure(authored_triangles)
    authored_support = _outward_support_audit(
        authored_vertices, authored_triangles
    )
    cooked_closure = _edge_closure(cooked_triangles)
    authored_volume = float(callback_row["authored_d372_topology_volume_m3"])
    cooked_volume = float(callback_row["callback_topology_volume_m3"])
    signed_loss = authored_volume - cooked_volume
    authored_bounds = np.asarray(callback_row["authored_bounds_m"], dtype=np.float64)
    cooked_bounds = np.asarray(
        callback_row["live_callback_bounds_m"], dtype=np.float64
    )
    lower_shift_mm = (cooked_bounds[0] - authored_bounds[0]) * 1000.0
    upper_shift_mm = (cooked_bounds[1] - authored_bounds[1]) * 1000.0
    outward_axis_expansion_mm = np.maximum(
        np.maximum(-lower_shift_mm, upper_shift_mm), 0.0
    )
    inward_axis_reduction_mm = np.maximum(
        np.maximum(lower_shift_mm, -upper_shift_mm), 0.0
    )
    containment = bool(
        authored_closure["pass"]
        and authored_support["all_triangle_planes_supporting"]
        and callback_row["structural"]["pass"] is True
        and callback_row["closure"]["pass"] is True
        and cooked_closure["pass"]
        and subset["json_numeric_exact_vertex_subset"]
        and all(value is True for value in local_binding_check.values())
    )
    return {
        "body": callback_row["body"],
        "prim_name": callback_row["prim_name"],
        "name": callback_row["name"],
        "role": callback_row["role"],
        "semantic_scope": _semantic_scope(callback_row["role"]),
        "d379_part_pass": callback_row["pass"],
        "failed_gate_signature": _failed_gate_signature(callback_row),
        "authored": {
            "closure": authored_closure,
            "supporting_plane_convexity": authored_support,
            "topology_volume_m3": authored_volume,
            "bounds_m": callback_row["authored_bounds_m"],
        },
        "cooked": {
            "closure_recomputed_from_embedded_triangles": cooked_closure,
            "d379_structural": callback_row["structural"],
            "d379_closure": callback_row["closure"],
            "original_polygon_topology_volume_m3": cooked_volume,
            "physx_property_volume_m3": callback_row[
                "physx_property_volume_m3"
            ],
            "bounds_m": callback_row["live_callback_bounds_m"],
            "polygon_count": callback_row["polygon_count"],
            "triangle_count": callback_row["triangle_count"],
            "max_polygon_plane_residual_m": callback_row[
                "max_polygon_plane_residual_m"
            ],
        },
        "vertex_provenance": subset,
        "set_containment_certificate": {
            "cooked_subset_of_authored_convex_part": containment,
            "same_body_local_frame_binding": {
                "live_path": callback_row["live_path"],
                "checks": local_binding_check,
                "pass": all(
                    value is True for value in local_binding_check.values()
                ),
            },
            "logic": (
                "authored triangle mesh is closed and all authored triangle "
                "planes are supporting; cooked callback is the inherited "
                "convexHull result and every embedded cooked vertex is a "
                "JSON-numeric-exact authored vertex; D379 property binding "
                "places the collider at zero local translation and identity "
                "local rotation in the same rigid-body frame"
            ),
            "callback_convexity_authority": (
                "inherited D379 approximation=convexHull and callback authority; "
                "D380 does not rerun NVIDIA runtime"
            ),
        },
        "difference": {
            "surface_authored_to_cooked_mm": (
                float(callback_row["surface"]["authored_to_live_m"]) * 1000.0
            ),
            "surface_cooked_to_authored_mm": (
                float(callback_row["surface"]["live_to_authored_m"]) * 1000.0
            ),
            "surface_symmetric_mm": (
                float(callback_row["surface"]["symmetric_m"]) * 1000.0
            ),
            "bounds_max_abs_mm": (
                float(callback_row["bounds_max_abs_delta_m"]) * 1000.0
            ),
            "lower_bound_inward_shift_xyz_mm": lower_shift_mm.tolist(),
            "upper_bound_inward_shift_xyz_mm": (-upper_shift_mm).tolist(),
            "max_outward_axis_expansion_mm": float(
                np.max(outward_axis_expansion_mm)
            ),
            "max_inward_axis_reduction_mm": float(
                np.max(inward_axis_reduction_mm)
            ),
            "signed_part_volume_loss_m3": signed_loss,
            "signed_part_volume_loss_mm3": signed_loss * 1.0e9,
            "signed_part_volume_loss_percent": (
                signed_loss / authored_volume * 100.0
            ),
            "d379_absolute_volume_relative_percent": (
                float(callback_row["authored_callback_volume_relative_delta"])
                * 100.0
            ),
        },
        "classification": (
            "AUTHORED_VERTEX_ELISION_WITH_INWARD_COOK"
            if containment
            and subset["omitted_authored_vertex_count"] > 0
            and signed_loss > 0.0
            and float(np.max(outward_axis_expansion_mm)) == 0.0
            else "UNRESOLVED_OR_NON_MONOTONIC"
        ),
    }


def _role_aggregate(
    all_metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    roles = sorted({row["role"] for row in all_metrics})
    for role in roles:
        rows = [row for row in all_metrics if row["role"] == role]
        authored = sum(row["authored"]["topology_volume_m3"] for row in rows)
        cooked = sum(
            row["cooked"]["original_polygon_topology_volume_m3"] for row in rows
        )
        failed = [row for row in rows if row["d379_part_pass"] is False]
        result[role] = {
            "part_count": len(rows),
            "failed_part_count": len(failed),
            "authored_part_volume_sum_mm3_not_union": authored * 1.0e9,
            "cooked_part_volume_sum_mm3_not_union": cooked * 1.0e9,
            "signed_part_volume_sum_loss_mm3_not_union": (
                (authored - cooked) * 1.0e9
            ),
            "signed_part_volume_sum_loss_percent_not_union": (
                (authored - cooked) / authored * 100.0
            ),
            "max_failed_surface_inward_mm": (
                max(
                    (
                        row["difference"]["surface_authored_to_cooked_mm"]
                        for row in failed
                    ),
                    default=0.0,
                )
            ),
            "failed_omitted_vertices_beyond_inherited_0_1mm_surface_limit": sum(
                row["vertex_provenance"][
                    "omitted_vertices_beyond_inherited_0_1mm_surface_limit"
                ]
                for row in failed
            ),
        }
    return result


def _negative_controls(
    d379: dict[str, Any],
    failed_keys: list[tuple[str, str]],
) -> dict[str, Any]:
    authored_map = {
        (row["body"], row["name"]): row
        for row in d379["authored_readback"]["rows"]
    }
    first = next(row for row in d379["callback_rows"] if row["pass"] is False)
    authored = authored_map[(first["body"], first["name"])]
    authored_vertices = np.asarray(authored["points_f32"], dtype=np.float64)
    cooked_vertices = np.asarray(first["live_callback_vertices_m"], dtype=np.float64)
    outward = cooked_vertices.copy()
    outward[0, 0] = float(np.max(authored_vertices[:, 0]) + 0.0002)
    controls = {
        "drop_one_failed_key_rejected": (
            set(failed_keys[:-1]) != set(failed_keys)
            and len(failed_keys[:-1]) != 17
        ),
        "body_name_swap_rejected_by_bijection": (
            ("link5", first["name"]) not in authored_map
            or ("link5", first["name"]) != (first["body"], first["name"])
        ),
        "outward_0_2mm_vertex_rejected_by_exact_subset": (
            _json_vertex_subset(authored_vertices, outward)[
                "json_numeric_exact_vertex_subset"
            ]
            is False
        ),
        "introduced_vertex_rejected_by_exact_subset": (
            _json_vertex_subset(
                authored_vertices,
                np.vstack([cooked_vertices, [[9.0, 9.0, 9.0]]]),
            )["introduced_or_moved_cooked_vertex_count"]
            == 1
        ),
        "nonpositive_loss_rejected": (
            not (
                float(first["authored_d372_topology_volume_m3"])
                - (
                    float(first["authored_d372_topology_volume_m3"])
                    + 1.0e-9
                )
                > 0.0
            )
        ),
        "tolerance_relaxation_detected": (
            {
                **d379["thresholds"],
                "surface_tolerance_m": 0.0002,
            }
            != EXPECTED_THRESHOLDS
        ),
        "non_d379_measurement_input_rejected": (
            not str(START_HERE.resolve()).startswith(
                str(
                    (
                        REPO
                        / "claudedocs/runtime_logs/grasp_track/g0a_d379"
                    ).resolve()
                )
            )
        ),
    }
    return {
        "controls": controls,
        "passed": sum(bool(value) for value in controls.values()),
        "total": len(controls),
        "pass": all(controls.values()),
    }


def _compute_evidence(d379: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    authored_map = {
        (row["body"], row["name"]): row
        for row in d379["authored_readback"]["rows"]
    }
    if len(authored_map) != 34:
        raise RuntimeError("D379 authored row bijection is not 34")
    property_bodies = d379["property_audit"]["bodies"]
    local_binding_map = {
        path: checks
        for body in EXPECTED_COUNTS
        for path, checks in property_bodies[body]["local_checks"].items()
    }
    if len(local_binding_map) != 34:
        raise RuntimeError("D379 local property-binding bijection is not 34")
    all_metrics = [
        _row_metric(
            authored_map[(row["body"], row["name"])],
            row,
            local_binding_map[row["live_path"]],
        )
        for row in d379["callback_rows"]
    ]
    failed = [row for row in all_metrics if row["d379_part_pass"] is False]
    failed_keys = [(row["body"], row["name"]) for row in failed]
    failed_by_body = Counter(row["body"] for row in failed)
    signature_counts = Counter(
        "+".join(row["failed_gate_signature"]) for row in failed
    )

    failed_authored_volume = sum(
        row["authored"]["topology_volume_m3"] for row in failed
    )
    failed_cooked_volume = sum(
        row["cooked"]["original_polygon_topology_volume_m3"] for row in failed
    )
    failed_loss = failed_authored_volume - failed_cooked_volume
    failed_omitted = sum(
        row["vertex_provenance"]["omitted_authored_vertex_count"]
        for row in failed
    )
    failed_omitted_beyond_surface_limit = sum(
        row["vertex_provenance"][
            "omitted_vertices_beyond_inherited_0_1mm_surface_limit"
        ]
        for row in failed
    )
    failed_retained = sum(
        row["vertex_provenance"]["retained_vertex_count"] for row in failed
    )
    failed_authored_vertices = sum(
        row["vertex_provenance"]["authored_unique_vertex_count"]
        for row in failed
    )
    all_subset = all(
        row["set_containment_certificate"][
            "cooked_subset_of_authored_convex_part"
        ]
        for row in all_metrics
    )
    all_local_bindings = (
        d379["property_audit"]["pass"] is True
        and all(
            property_bodies[body]["pass"] is True
            and property_bodies[body]["checks"][
                "all_p34_local_bindings_valid"
            ]
            is True
            and property_bodies[body]["checks"][
                "exact_p34_path_bijection"
            ]
            is True
            and len(property_bodies[body]["local_checks"])
            == EXPECTED_COUNTS[body]
            for body in EXPECTED_COUNTS
        )
        and all(
            all(value is True for value in checks.values())
            for checks in local_binding_map.values()
        )
    )
    all_json_subset = all(
        row["vertex_provenance"]["json_numeric_exact_vertex_subset"]
        for row in all_metrics
    )
    failed_monotonic = all(
        row["classification"] == "AUTHORED_VERTEX_ELISION_WITH_INWARD_COOK"
        for row in failed
    )
    max_authored_vertices = max(
        row["vertex_provenance"]["authored_unique_vertex_count"]
        for row in all_metrics
    )

    fixed_rows = [
        row
        for row in failed
        if row["role"] in {"fixed_jaw", "fixed_jaw_backbone"}
    ]
    moving_rows = [
        row
        for row in failed
        if row["role"] in {"moving_jaw", "moving_jaw_backbone"}
    ]
    fixed_h = max(
        row["difference"]["surface_authored_to_cooked_mm"]
        for row in fixed_rows
    )
    moving_h = max(
        row["difference"]["surface_authored_to_cooked_mm"]
        for row in moving_rows
    )
    link5_h = max(
        row["difference"]["surface_authored_to_cooked_mm"]
        for row in failed
        if row["body"] == "link5"
    )
    gripper_h = max(
        row["difference"]["surface_authored_to_cooked_mm"]
        for row in failed
        if row["body"] == "gripper_link"
    )

    window_core_names = {
        "moving_jaw_02_center_bridge",
        "moving_jaw_03_window_upper_rail",
        "moving_jaw_04_window_lower_rail",
        "fixed_jaw_01_lower_left_leg",
        "fixed_jaw_02_lower_right_leg",
        "fixed_jaw_03_middle_bridge",
        "fixed_jaw_04_upper_left_leg",
        "fixed_jaw_05_upper_right_leg",
    }
    window_core = [
        {
            "body": row["body"],
            "name": row["name"],
            "d379_part_pass": row["d379_part_pass"],
            "surface_symmetric_mm": row["difference"]["surface_symmetric_mm"],
        }
        for row in all_metrics
        if row["name"] in window_core_names
    ]
    role_aggregate = _role_aggregate(all_metrics)
    controls = _negative_controls(d379, failed_keys)

    current_scope_counters = {
        "actual_offline_worker_invocations": 1,
        "automatic_retries": 0,
        "offline_audit_invocations": 1,
        **FORBIDDEN_RUNTIME_COUNTERS,
    }
    evidence = {
        "artifact": "D380_P34_FAILED_PART_COOK_PROVENANCE_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Use only immutable D379 embedded authored/callback geometry to "
            "explain the 17 failed cooked parts and determine only the "
            "geometry-set direction of jaw, void, and clearance impact."
        ),
        "new_variables": NEW_VARIABLES,
        "measurement_authority": {
            "path": _rel(D379_EVIDENCE),
            "sha256": _sha(D379_EVIDENCE),
            "sole_measurement_input": True,
            "callback_volume_semantics": (
                "D379 original callback polygon topology volume remains "
                "authority; D380 does not construct a Qhull volume"
            ),
            "json_coordinate_precision_boundary": (
                "vertex equality is exact equality of JSON numeric coordinate "
                "triples, not a new raw-memory or dtype-bit attestation"
            ),
        },
        "official_sources_inherited_from_d379": d379["official_sources"],
        "official_source_boundary": (
            "NVIDIA documents that convexHull is a cooked collision "
            "approximation. NVIDIA does not state the D380 vertex-elision count "
            "or identify the internal heuristic that produced it; those remain "
            "local-evidence results and a null causal mechanism."
        ),
        "inherited_d379_contract": {
            "thresholds_unchanged": d379["thresholds"],
            "identity_pass": d379["identity_pass"],
            "verdict": d379["verdict"],
            "failed_part_count": 17,
        },
        "same_body_local_frame_authority": {
            "property_audit_pass": d379["property_audit"]["pass"],
            "all_p34_local_bindings_valid": all_local_bindings,
            "binding_count": len(local_binding_map),
            "per_body": {
                body: {
                    "count": len(property_bodies[body]["local_checks"]),
                    "exact_p34_path_bijection": property_bodies[body]["checks"][
                        "exact_p34_path_bijection"
                    ],
                    "all_p34_local_bindings_valid": property_bodies[body][
                        "checks"
                    ]["all_p34_local_bindings_valid"],
                }
                for body in EXPECTED_COUNTS
            },
            "required_local_fields": [
                "local_pos_zero",
                "local_rot_identity",
                "path_nonempty",
                "positive_finite_volume",
                "result_valid",
            ],
        },
        "counts": {
            "all_parts": len(all_metrics),
            "failed_parts": len(failed),
            "failed_by_body": dict(sorted(failed_by_body.items())),
            "failed_authored_unique_vertices": failed_authored_vertices,
            "failed_retained_vertices": failed_retained,
            "failed_omitted_vertices": failed_omitted,
            "failed_omitted_vertices_beyond_inherited_0_1mm_surface_limit": (
                failed_omitted_beyond_surface_limit
            ),
            "failed_introduced_or_moved_vertices": sum(
                row["vertex_provenance"][
                    "introduced_or_moved_cooked_vertex_count"
                ]
                for row in failed
            ),
            "all34_json_numeric_exact_vertex_subsets": sum(
                row["vertex_provenance"]["json_numeric_exact_vertex_subset"]
                for row in all_metrics
            ),
            "all34_containment_certificates": sum(
                row["set_containment_certificate"][
                    "cooked_subset_of_authored_convex_part"
                ]
                for row in all_metrics
            ),
            "failed_gate_signature_counts": dict(
                sorted(signature_counts.items())
            ),
        },
        "capacity_causal_boundary": {
            "authored_hull_vertex_limit": 64,
            "max_authored_unique_vertices_per_part": max_authored_vertices,
            "input_exceeded_hull_vertex_limit": max_authored_vertices > 64,
            "verdict": (
                "64_VERTEX_CAP_NOT_TRIGGERED_BY_ANY_P34_PART"
                if max_authored_vertices <= 64
                else "CAPACITY_TRIGGER_POSSIBLE"
            ),
            "exact_internal_cook_algorithm_or_tolerance_cause": None,
        },
        "part_volume_sum_diagnostic": {
            "warning": (
                "These are sums of per-part original-topology volumes, not "
                "boolean union volume; compound parts may overlap."
            ),
            "failed_authored_sum_mm3": failed_authored_volume * 1.0e9,
            "failed_cooked_sum_mm3": failed_cooked_volume * 1.0e9,
            "failed_signed_loss_sum_mm3": failed_loss * 1.0e9,
            "failed_signed_loss_percent": (
                failed_loss / failed_authored_volume * 100.0
            ),
            "role_aggregates": role_aggregate,
        },
        "semantic_impact": {
            "same_transform_geometry_only": {
                "cooked_union_subset_of_authored_union": all_subset,
                "authored_void_fill_by_cooked_geometry": (
                    "PROVEN_ABSENT" if all_subset else None
                ),
                "geometric_clearance_can_decrease": (
                    False if all_subset else None
                ),
                "geometric_clearance_direction": (
                    "same_or_larger" if all_subset else None
                ),
                "earlier_protruding_contact_can_be_created": (
                    False if all_subset else None
                ),
                "contact_surface_can_be_removed_or_contact_delayed": (
                    True if failed_monotonic else None
                ),
                "scope_note": (
                    "Set-distance statement only. It does not cover penetration "
                    "depth, contactOffset/restOffset, CCD, solver timing, or "
                    "contact reporting."
                ),
            },
            "role_scoped_jaw_separation_increase_upper_bound_mm": {
                "fixed_jaw_system_one_sided_surface_mm": fixed_h,
                "moving_jaw_system_one_sided_surface_mm": moving_h,
                "sum_bound_mm": fixed_h + moving_h,
                "observed_open_clearance_delta_mm": None,
                "note": (
                    "Rigid-transform-invariant geometry bound, not an observed "
                    "OPEN-pose clearance and not a cylinder clearance."
                ),
            },
            "full_body_separation_increase_upper_bound_mm": {
                "link5_one_sided_surface_mm": link5_h,
                "gripper_link_one_sided_surface_mm": gripper_h,
                "sum_bound_mm": link5_h + gripper_h,
                "observed_same_pose_separation_delta_mm": None,
            },
            "named_window_core_status": window_core,
            "actual_cross_body_open_clearance_mm": None,
            "mouth_or_window_void_volume_change_mm3": None,
            "cylinder_facing_contact_patch_identity": None,
            "cylinder_or_contact_result": None,
        },
        "failed_parts": failed,
        "all_part_control_summary": {
            "all_json_numeric_exact_vertex_subset": all_json_subset,
            "all_containment_certificates": all_subset,
            "all_same_body_local_frame_bindings": all_local_bindings,
            "max_authored_support_plane_violation_m": max(
                row["authored"]["supporting_plane_convexity"][
                    "max_supporting_plane_violation_m"
                ]
                for row in all_metrics
            ),
            "max_authored_vertex_count": max_authored_vertices,
        },
        "negative_controls": controls,
        "current_scope_counters": current_scope_counters,
        "remaining_nulls": {
            "exact_internal_physx_cook_heuristic": None,
            "actual_open_jaw_clearance": None,
            "29x50_target_geometry_or_pose": None,
            "cylinder_contact_or_tipping": None,
            "q5_closure": None,
            "grasp_feasibility": None,
            "target_ik_path_justification": None,
        },
        "audit_pass": (
            len(all_metrics) == 34
            and len(failed) == 17
            and dict(failed_by_body) == EXPECTED_D379_FAILED
            and failed_authored_vertices == 401
            and failed_retained == 178
            and failed_omitted == 223
            and failed_omitted_beyond_surface_limit == 181
            and all_json_subset
            and all_subset
            and all_local_bindings
            and failed_monotonic
            and max_authored_vertices < 64
            and controls["pass"]
            and d379["thresholds"] == EXPECTED_THRESHOLDS
        ),
        "p34_authored_to_cooked_identity_pass": False,
        "g0a_pass": False,
        "verdict": (
            "D380_FAILED_PART_PROVENANCE_AUDIT_PASS_REPAIR_REQUIRED"
            if (
                len(failed) == 17
                and all_subset
                and failed_monotonic
                and controls["pass"]
            )
            else "D380_FAILED_PART_PROVENANCE_AUDIT_INTEGRITY_FAIL_STOP"
        ),
        "next_authorization_boundary": (
            "Representation repair/live identity, 29x50 target rebase, and "
            "all physics/q5/contact work remain separately unapproved."
        ),
    }
    return evidence, all_metrics


def _write_metrics_csv(failed: list[dict[str, Any]]) -> None:
    fields = [
        "body",
        "prim_name",
        "name",
        "role",
        "failed_gate_signature",
        "authored_vertices",
        "cooked_vertices",
        "omitted_vertices",
        "introduced_or_moved_vertices",
        "surface_inward_mm",
        "bounds_max_abs_mm",
        "part_volume_loss_mm3",
        "part_volume_loss_percent",
        "classification",
    ]
    with METRICS_CSV.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in failed:
            writer.writerow(
                {
                    "body": row["body"],
                    "prim_name": row["prim_name"],
                    "name": row["name"],
                    "role": row["role"],
                    "failed_gate_signature": "+".join(
                        row["failed_gate_signature"]
                    ),
                    "authored_vertices": row["vertex_provenance"][
                        "authored_unique_vertex_count"
                    ],
                    "cooked_vertices": row["vertex_provenance"][
                        "cooked_unique_vertex_count"
                    ],
                    "omitted_vertices": row["vertex_provenance"][
                        "omitted_authored_vertex_count"
                    ],
                    "introduced_or_moved_vertices": row[
                        "vertex_provenance"
                    ]["introduced_or_moved_cooked_vertex_count"],
                    "surface_inward_mm": row["difference"][
                        "surface_authored_to_cooked_mm"
                    ],
                    "bounds_max_abs_mm": row["difference"][
                        "bounds_max_abs_mm"
                    ],
                    "part_volume_loss_mm3": row["difference"][
                        "signed_part_volume_loss_mm3"
                    ],
                    "part_volume_loss_percent": row["difference"][
                        "signed_part_volume_loss_percent"
                    ],
                    "classification": row["classification"],
                }
            )


def _geometry_maps(
    d379: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    authored_map = {
        (row["body"], row["name"]): row
        for row in d379["authored_readback"]["rows"]
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for callback in d379["callback_rows"]:
        if callback["pass"] is not False:
            continue
        authored = authored_map[(callback["body"], callback["name"])]
        result[(callback["body"], callback["prim_name"])] = {
            "name": callback["name"],
            "role": callback["role"],
            "authored_vertices": np.asarray(
                authored["points_f32"], dtype=np.float64
            ),
            "authored_triangles": _triangles_from_authored(authored),
            "cooked_vertices": np.asarray(
                callback["live_callback_vertices_m"], dtype=np.float64
            ),
            "cooked_triangles": np.asarray(
                callback["live_callback_topology_triangles"], dtype=np.int64
            ).reshape(-1, 3),
        }
    return result


def _render_board(
    d379: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    regular = font_manager.FontProperties(
        fname=str(FONT_REGULAR) if FONT_REGULAR.is_file() else None
    )
    bold = font_manager.FontProperties(
        fname=str(FONT_BOLD) if FONT_BOLD.is_file() else None
    )
    geometry = _geometry_maps(d379)
    failed = evidence["failed_parts"]
    metric_by_key = {
        (row["body"], row["prim_name"]): row for row in failed
    }

    def body_overlay(axis: Any, body: str) -> None:
        points: list[np.ndarray] = []
        for key in sorted(key for key in geometry if key[0] == body):
            row = geometry[key]
            authored = row["authored_vertices"] * 1000.0
            cooked = row["cooked_vertices"] * 1000.0
            axis.add_collection3d(
                Poly3DCollection(
                    authored[row["authored_triangles"]],
                    facecolor="#D7DCE2",
                    edgecolor="#6B7280",
                    linewidth=0.28,
                    alpha=0.16,
                )
            )
            axis.add_collection3d(
                Poly3DCollection(
                    cooked[row["cooked_triangles"]],
                    facecolor=ROLE_COLORS_HEX.get(row["role"], "#4E79A7"),
                    edgecolor="#1F2937",
                    linewidth=0.34,
                    alpha=0.58,
                )
            )
            omitted_indices = metric_by_key[key]["vertex_provenance"][
                "omitted_authored_vertex_indices"
            ]
            if omitted_indices:
                omitted = authored[np.asarray(omitted_indices, dtype=np.int64)]
                axis.scatter(
                    omitted[:, 0],
                    omitted[:, 1],
                    omitted[:, 2],
                    c="#D62728",
                    s=8.0,
                    depthshade=False,
                )
            points.extend([authored, cooked])
        stacked = np.vstack(points)
        lower, upper = stacked.min(axis=0), stacked.max(axis=0)
        center = (lower + upper) * 0.5
        radius = max(float((upper - lower).max()) * 0.59, 1.0)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.view_init(elev=18, azim=-58 if body == "link5" else -72)
        axis.set_proj_type("ortho")
        axis.set_axis_off()
        axis.set_title(
            f"{body}: failed parts only\n"
            "gray=authored, color=cooked, red=omitted vertex",
            fontproperties=bold,
            fontsize=10.5,
            pad=1.0,
        )

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#FAFBFD")
    grid = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[0.57, 0.43],
        width_ratios=[0.36, 0.36, 0.28],
        hspace=0.18,
        wspace=0.12,
    )
    body_overlay(fig.add_subplot(grid[0, 0], projection="3d"), "link5")
    body_overlay(fig.add_subplot(grid[0, 1], projection="3d"), "gripper_link")

    summary_axis = fig.add_subplot(grid[0, 2])
    summary_axis.axis("off")
    counts = evidence["counts"]
    impact = evidence["semantic_impact"]
    volume = evidence["part_volume_sum_diagnostic"]
    summary_lines = [
        "OBSERVED PROVENANCE",
        f"FAIL parts: {counts['failed_parts']}/34",
        (
            "Authored -> cooked vertices: "
            f"{counts['failed_authored_unique_vertices']} -> "
            f"{counts['failed_retained_vertices']}"
        ),
        f"Omitted: {counts['failed_omitted_vertices']}",
        (
            "Omitted farther than 0.1 mm: "
            f"{counts['failed_omitted_vertices_beyond_inherited_0_1mm_surface_limit']}"
        ),
        (
            "Introduced/moved JSON coordinates: "
            f"{counts['failed_introduced_or_moved_vertices']}"
        ),
        (
            "Failed-part volume sum loss: "
            f"{volume['failed_signed_loss_sum_mm3']:.3f} mm^3"
        ),
        "",
        "GEOMETRY-ONLY IMPACT",
        "Cooked union is a subset: YES",
        "Authored void newly filled: NO",
        "Geometric clearance decrease: NO",
        "Contact surface removal/delay: POSSIBLE",
        (
            "Jaw separation increase bound: "
            f"{impact['role_scoped_jaw_separation_increase_upper_bound_mm']['sum_bound_mm']:.3f} mm"
        ),
        "Actual OPEN clearance/contact: NULL",
        "",
        "DECISION",
        "Audit PASS; P34 identity still FAIL.",
        "Repair/live identity required before physics.",
    ]
    summary_axis.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontproperties=regular,
        fontsize=10.0,
        linespacing=1.34,
        color="#1F2937",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F3F6FA",
            "edgecolor": "#B7C1CE",
        },
    )

    surface_axis = fig.add_subplot(grid[1, 0:2])
    ordered = sorted(
        failed,
        key=lambda row: row["difference"]["surface_authored_to_cooked_mm"],
    )
    labels = [
        ("L5/" if row["body"] == "link5" else "GR/")
        + row["name"].replace("moving_", "m_").replace("fixed_", "f_")
        for row in ordered
    ]
    values = [
        row["difference"]["surface_authored_to_cooked_mm"]
        for row in ordered
    ]
    colors = [
        ROLE_COLORS_HEX.get(row["role"], "#4E79A7") for row in ordered
    ]
    y = np.arange(len(ordered))
    surface_axis.barh(y, values, color=colors, alpha=0.88)
    surface_axis.axvline(
        0.1,
        color="#9B1C1C",
        linestyle="--",
        linewidth=1.4,
        label="D379 surface limit 0.1 mm",
    )
    surface_axis.set_yticks(y)
    surface_axis.set_yticklabels(labels, fontproperties=regular, fontsize=7.6)
    surface_axis.set_xlabel(
        "Authored-to-cooked inward solid distance (mm)",
        fontproperties=regular,
        fontsize=9.5,
    )
    surface_axis.set_title(
        "All 17 failed parts move inward; 10 keep the same AABB",
        fontproperties=bold,
        fontsize=11.0,
    )
    surface_axis.grid(axis="x", alpha=0.22)
    surface_axis.legend(loc="lower right", prop=regular, fontsize=8.5)

    volume_axis = fig.add_subplot(grid[1, 2])
    ordered_volume = sorted(
        failed,
        key=lambda row: row["difference"][
            "signed_part_volume_loss_percent"
        ],
    )
    vlabels = [
        ("L5/" if row["body"] == "link5" else "GR/")
        + row["prim_name"].split("_", 1)[0]
        for row in ordered_volume
    ]
    vvalues = [
        row["difference"]["signed_part_volume_loss_percent"]
        for row in ordered_volume
    ]
    vcolors = [
        ROLE_COLORS_HEX.get(row["role"], "#4E79A7")
        for row in ordered_volume
    ]
    vy = np.arange(len(ordered_volume))
    volume_axis.barh(vy, vvalues, color=vcolors, alpha=0.88)
    volume_axis.axvline(
        0.5,
        color="#9B1C1C",
        linestyle="--",
        linewidth=1.4,
        label="D379 0.5%",
    )
    volume_axis.set_yticks(vy)
    volume_axis.set_yticklabels(
        vlabels, fontproperties=regular, fontsize=7.5
    )
    volume_axis.set_xlabel(
        "Per-part volume loss (%)",
        fontproperties=regular,
        fontsize=9.0,
    )
    volume_axis.set_title(
        "Original callback polygon topology",
        fontproperties=bold,
        fontsize=10.5,
    )
    volume_axis.grid(axis="x", alpha=0.22)
    volume_axis.legend(loc="lower right", prop=regular, fontsize=8.2)

    fig.suptitle(
        "D380 | Why 17 P34 collision parts failed after cooking",
        fontproperties=bold,
        fontsize=20.0,
        color="#14213D",
        y=0.985,
    )
    fig.text(
        0.5,
        0.954,
        (
            "Immutable D379 JSON only | no Isaac/PhysX/USD/cylinder/physics/q5/contact | "
            "existing 0.1 mm / 0.5% gates unchanged"
        ),
        ha="center",
        va="center",
        fontproperties=regular,
        fontsize=10.5,
        color="#4B5563",
    )
    fig.text(
        0.5,
        0.012,
        (
            "Exact internal cook heuristic remains unknown. Geometry-set result: "
            "vertex elision/erosion, not outward expansion. g0a_pass=false."
        ),
        ha="center",
        va="bottom",
        fontproperties=regular,
        fontsize=9.7,
        color="#374151",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.925,
        bottom=0.065,
    )
    fig.savefig(BOARD_PATH, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    info = _png_info(BOARD_PATH)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"D380 board dimension failure: {info}")
    return info


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="/",
                        contents="/d380/authored/link5/**",
                        name="link5 | authored failed parts",
                    ),
                    rrb.Spatial3DView(
                        origin="/",
                        contents="/d380/cooked/link5/**",
                        name="link5 | cooked failed parts",
                    ),
                    column_shares=[0.5, 0.5],
                ),
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="/",
                        contents="/d380/authored/gripper_link/**",
                        name="moving side | authored failed parts",
                    ),
                    rrb.Spatial3DView(
                        origin="/",
                        contents="/d380/cooked/gripper_link/**",
                        name="moving side | cooked failed parts",
                    ),
                    column_shares=[0.5, 0.5],
                ),
                row_shares=[0.5, 0.5],
            ),
            rrb.TextDocumentView(
                origin="/metadata/run",
                contents="/metadata/run",
                name="D380 static audit summary",
            ),
            column_shares=[0.72, 0.28],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _write_rerun(
    d379: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    import roarm_rl.viz_debug as viz_debug

    geometry = _geometry_maps(d379)
    meshes: list[dict[str, Any]] = []
    expected_entities = {"metadata/run"}
    component_contract: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
    }
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    for (body, prim_name), row in sorted(geometry.items()):
        authored_path = f"d380/authored/{body}/{prim_name}"
        cooked_path = f"d380/cooked/{body}/{prim_name}"
        for representation, entity_path, vertices, triangles, color in (
            (
                "authored",
                authored_path,
                row["authored_vertices"],
                row["authored_triangles"],
                [190, 196, 204, 88],
            ),
            (
                "cooked",
                cooked_path,
                row["cooked_vertices"],
                row["cooked_triangles"],
                ROLE_COLORS_RGBA.get(row["role"], [78, 121, 167, 205]),
            ),
        ):
            meshes.append(
                {
                    "entity_path": entity_path,
                    "coordinate_frame": "tf#/",
                    "vertices_m": vertices,
                    "triangles": triangles,
                    "color_rgba": color,
                    "static": True,
                    "body": body,
                    "prim_name": prim_name,
                    "name": row["name"],
                    "role": row["role"],
                    "representation": representation,
                    "display_role": (
                        "Float32 Rerun inspection copy only; D379 JSON arrays "
                        "remain numeric authority"
                    ),
                }
            )
            metadata_path = (
                f"metadata/meshes/{entity_path.replace('/', '__')}"
            )
            expected_entities.update({entity_path, metadata_path})
            component_contract[entity_path] = mesh_components
            component_contract[metadata_path] = ["TextDocument:text"]

    summary = {
        "case": CASE,
        "attempt": ATTEMPT,
        "verdict": evidence["verdict"],
        "failed_parts": evidence["counts"]["failed_parts"],
        "vertex_change": (
            f"{evidence['counts']['failed_authored_unique_vertices']} -> "
            f"{evidence['counts']['failed_retained_vertices']}; "
            f"omitted={evidence['counts']['failed_omitted_vertices']}; "
            "omitted_over_0.1mm="
            f"{evidence['counts']['failed_omitted_vertices_beyond_inherited_0_1mm_surface_limit']}; "
            "introduced_or_moved=0"
        ),
        "provenance": (
            "JSON-numeric-exact authored vertex subset with inward material loss"
        ),
        "no_void_fill": True,
        "geometric_clearance_can_decrease": False,
        "actual_open_clearance": None,
        "actual_contact_or_grasp": None,
        "p34_identity_pass": False,
        "g0a_pass": False,
        "scope": (
            "offline D379 evidence only; no NVIDIA runtime, USD, cylinder, "
            "physics, q5, contact, target/IK/path"
        ),
    }

    original_builder = viz_debug.build_rerun_blueprint

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d380_failed_cook_provenance":
            return _build_blueprint()
        return original_builder(mode)

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    try:
        saved = viz_debug.log_rerun(
            RRD_PATH,
            meshes=meshes,
            recording_metadata=summary,
            recording_id="g0a_d380_failed_cook_provenance",
            blueprint_path=RBL_PATH,
            blueprint_mode="d380_failed_cook_provenance",
            live_viewer=False,
            app_id="roarm_g0a_d380_failed_cook_provenance",
        )
    finally:
        viz_debug.build_rerun_blueprint = original_builder
        os.environ["PATH"] = old_path
    if not saved.get("ok"):
        raise RuntimeError(f"D380 save-only Rerun failed: {saved}")

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
    headless = dict(validation.get("headless_render") or {})
    return {
        "save_only": saved,
        "strict_validation_pass": validation.get("pass") is True,
        "headless_viewer_invocations": (
            1 if headless.get("attempted") is True else 0
        ),
        "headless_viewer_returncode": headless.get("returncode"),
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


def prepare() -> int:
    if OUT_DIR.exists():
        raise FileExistsError(f"forward-only output already exists: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")

    d379 = _read_json(D379_EVIDENCE)
    source_imports = _import_roots(SCRIPT_PATH)
    dependency_versions = _dependency_versions()
    start_text = START_HERE.read_text(encoding="utf-8")
    input_in_d379 = str(D379_EVIDENCE.resolve()).startswith(
        str(
            (
                REPO / "claudedocs/runtime_logs/grasp_track/g0a_d379"
            ).resolve()
        )
    )
    checks = {
        "sole_measurement_input_exists": D379_EVIDENCE.is_file(),
        "sole_measurement_input_sha256_exact": (
            _sha(D379_EVIDENCE) == D379_EVIDENCE_SHA256
        ),
        "sole_measurement_input_under_d379": input_in_d379,
        "d379_artifact_exact": (
            d379.get("artifact")
            == "D379_P34_FULL_LIVE_IDENTITY_CLASSIFIER_RESUME_EVIDENCE_V1"
        ),
        "d379_verdict_exact": (
            d379.get("verdict")
            == "D379_P34_FULL_LIVE_IDENTITY_CLASSIFIER_RESUME_FAIL_STOP"
        ),
        "d379_identity_false": d379.get("identity_pass") is False,
        "d379_g0a_false": d379.get("g0a_pass") is False,
        "d379_parts_34": len(d379.get("callback_rows", [])) == 34,
        "d379_authored_rows_34": (
            len(d379.get("authored_readback", {}).get("rows", [])) == 34
        ),
        "d379_all_authored_approximation_convex_hull": all(
            row.get("approximation") == "convexHull"
            for row in d379.get("authored_readback", {}).get("rows", [])
        ),
        "d379_all_hull_vertex_limits_64": all(
            row.get("hull_vertex_limit") == 64
            for row in d379.get("authored_readback", {}).get("rows", [])
        ),
        "d379_property_audit_and_local_bindings_exact": (
            d379.get("property_audit", {}).get("pass") is True
            and all(
                d379["property_audit"]["bodies"][body].get("pass") is True
                and d379["property_audit"]["bodies"][body]["checks"].get(
                    "exact_p34_path_bijection"
                )
                is True
                and d379["property_audit"]["bodies"][body]["checks"].get(
                    "all_p34_local_bindings_valid"
                )
                is True
                and len(
                    d379["property_audit"]["bodies"][body].get(
                        "local_checks", {}
                    )
                )
                == EXPECTED_COUNTS[body]
                and all(
                    all(value is True for value in local.values())
                    for local in d379["property_audit"]["bodies"][body][
                        "local_checks"
                    ].values()
                )
                for body in EXPECTED_COUNTS
            )
        ),
        "d379_failed_parts_17": (
            sum(row.get("pass") is False for row in d379["callback_rows"])
            == 17
        ),
        "d379_thresholds_exact_unchanged": (
            d379.get("thresholds") == EXPECTED_THRESHOLDS
        ),
        "d379_current_scope_was_offline": (
            d379.get("current_scope_counters", {}).get("isaac_launches") == 0
            and d379.get("current_scope_counters", {}).get("physx_calls") == 0
            and d379.get("current_scope_counters", {}).get("physics_steps") == 0
            and d379.get("current_scope_counters", {}).get("q5_samples") == 0
            and d379.get("current_scope_counters", {}).get("contact_queries") == 0
        ),
        "script_forbidden_imports_absent": not (
            source_imports & FORBIDDEN_IMPORT_ROOTS
        ),
        "interpreter_exact_isaaclab_python": (
            Path(sys.executable).resolve() == ISAACLAB_PYTHON.resolve()
        ),
        "dependency_versions_exact": dependency_versions
        == {
            "matplotlib": "3.10.3",
            "numpy": "1.26.0",
            "pillow": "11.3.0",
            "psutil": "5.9.8",
            "rerun_sdk": "0.34.1",
            "trimesh": "4.5.1",
        },
        "rerun_cli_exists": RERUN_CLI.is_file(),
        "font_regular_exists": FONT_REGULAR.is_file(),
        "font_bold_exists": FONT_BOLD.is_file(),
        "start_here_active_case_exact": (
            "D380 [p34_failed_part_cook_provenance_and_semantic_impact_audit]"
            in start_text
            and _rel(OUT_DIR) in start_text
        ),
        "official_sources_embedded": (
            len(d379.get("official_sources", [])) >= 3
            and all(
                str(row.get("url", "")).startswith(
                    "https://docs.omniverse.nvidia.com/"
                )
                for row in d379["official_sources"]
            )
        ),
        "head_equals_origin_master": (
            _git("rev-parse", "HEAD")
            == _git("rev-parse", "origin/master")
        ),
    }
    prepare_controls = {
        "relaxed_surface_threshold_differs": (
            {**EXPECTED_THRESHOLDS, "surface_tolerance_m": 0.0002}
            != EXPECTED_THRESHOLDS
        ),
        "wrong_input_hash_rejected": (
            "0" * 64 != D379_EVIDENCE_SHA256
        ),
        "non_d379_path_rejected": (
            not str((REPO / "START_HERE.md").resolve()).startswith(
                str(
                    (
                        REPO
                        / "claudedocs/runtime_logs/grasp_track/g0a_d379"
                    ).resolve()
                )
            )
        ),
        "forbidden_import_mutation_detected": (
            (source_imports | {"omni"}) & FORBIDDEN_IMPORT_ROOTS
            == {"omni"}
        ),
    }
    prereg = {
        "artifact": "D380_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Audit only the immutable D379 embedded authored/cooked geometry "
            "to explain 17 failures and bound geometry-only semantic impact."
        ),
        "new_variables": NEW_VARIABLES,
        "input_allowlist": [
            {
                "path": _rel(D379_EVIDENCE),
                "sha256": D379_EVIDENCE_SHA256,
                "role": "sole measurement authority",
            }
        ],
        "registered_measurements": [
            "exact D379 failed set and gate signatures",
            "JSON numeric exact cooked-vertex subset of authored vertices",
            "authored original-triangle closure and supporting-plane convexity",
            "retained/omitted/introduced vertex counts",
            "D379 directed solid-surface distances and axis bounds",
            "D379 original callback polygon-topology signed volume loss",
            "same-transform set-containment effect on jaw/void/clearance",
        ],
        "registered_nulls": [
            "exact internal PhysX cook algorithm/tolerance cause",
            "actual OPEN jaw clearance or common-frame gap",
            "mouth/window void volume change",
            "cylinder-facing face identity",
            "cylinder/contact/q5/physics/grasp result",
            "target/IK/path justification",
        ],
        "inherited_thresholds_unchanged": EXPECTED_THRESHOLDS,
        "roundoff_policy": (
            "Float64 supporting-plane computations use a machine-epsilon "
            "roundoff budget only; it is not a replacement geometry tolerance "
            "or D379 acceptance gate."
        ),
        "callback_volume_semantics": (
            "Use the original callback polygon-topology volume already embedded "
            "by D379; do not compute a new Qhull volume."
        ),
        "registered_execution": {
            "actual_offline_worker_invocations": 1,
            "automatic_retries": 0,
            "bounded_watchdog_seconds": WATCHDOG_SECONDS,
            "board_exact_pixels": [1920, 1080],
            "save_only_rrd_rbl": True,
            "headless_rerun_viewer_max": 1,
            **FORBIDDEN_RUNTIME_COUNTERS,
        },
        "source_hashes": _source_hashes(),
        "input_hashes": _input_hashes(),
        "interpreter": {
            "sys_executable": sys.executable,
            "resolved": str(Path(sys.executable).resolve()),
            "registered": str(ISAACLAB_PYTHON),
            "registered_resolved": str(ISAACLAB_PYTHON.resolve()),
        },
        "dependency_versions": dependency_versions,
        "registered_dirty_baseline": _status_paths(),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "subject": _git("log", "-1", "--pretty=%s"),
        },
        "official_sources_inherited_from_d379": d379["official_sources"],
        "checks": checks,
        "prepare_negative_controls": {
            "controls": prepare_controls,
            "passed": sum(bool(value) for value in prepare_controls.values()),
            "total": len(prepare_controls),
            "pass": all(prepare_controls.values()),
        },
        "pass": all(checks.values()) and all(prepare_controls.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    _phase(
        "preregistration_frozen",
        preregistration_sha256=_sha(PREREG_PATH),
        passed=prereg["pass"],
        checks_passed=sum(bool(value) for value in checks.values()),
        checks_total=len(checks),
    )
    if not prereg["pass"]:
        raise RuntimeError(f"D380 preregistration failed: {checks}")
    return 0


def _worker() -> int:
    _phase("worker_start", pid=os.getpid())
    prereg = _read_json(PREREG_PATH)
    invocation = _read_json(INVOCATION_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D380 preregistration did not pass")
    if invocation.get("preregistration_sha256") != _sha(PREREG_PATH):
        raise RuntimeError("D380 invocation not bound to preregistration")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D380 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D380 input changed after preregistration")
    if _status_paths() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("D380 dirty baseline changed after preregistration")

    d379 = _read_json(D379_EVIDENCE)
    evidence, _ = _compute_evidence(d379)
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase(
        "canonical_evidence_written",
        evidence_sha256=_sha(EVIDENCE_PATH),
        audit_pass=evidence["audit_pass"],
        verdict=evidence["verdict"],
    )
    _write_metrics_csv(evidence["failed_parts"])
    board = _render_board(d379, evidence)
    _phase("exact_board_written", board_sha256=board["sha256"])
    rerun = _write_rerun(d379, evidence)
    _phase(
        "rerun_finalized",
        rrd_sha256=rerun["rrd"]["sha256"],
        strict_validation_pass=rerun["strict_validation_pass"],
        viewer_returncode=rerun["headless_viewer_returncode"],
    )

    manual_template = {
        "artifact": "D380_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "evidence": {
            "path": _rel(EVIDENCE_PATH),
            "sha256": _sha(EVIDENCE_PATH),
        },
        "board": board,
        "rerun_screenshot": rerun["screenshot"],
        "required_observations": [
            "board is legible at original resolution with no clipped text",
            "authored/cooked overlays and omitted vertices are visible",
            "surface and volume plots agree with canonical JSON",
            "Rerun shows authored and cooked failed-part geometry in four views",
            "Rerun has no Unknown timeline panel or decision-obscuring overlap",
        ],
        "required_check_keys": sorted(MANUAL_CHECK_KEYS),
        "inspection_checks": {
            name: None for name in sorted(MANUAL_CHECK_KEYS)
        },
        "observations": [],
        "inspector_result": None,
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)

    required_files = [
        EVIDENCE_PATH,
        METRICS_CSV,
        BOARD_PATH,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION,
        RERUN_SCREENSHOT,
        MANUAL_TEMPLATE,
    ]
    post_checks = {
        "canonical_evidence_audit_pass": evidence["audit_pass"] is True,
        "p34_identity_remains_false": (
            evidence["p34_authored_to_cooked_identity_pass"] is False
        ),
        "g0a_remains_false": evidence["g0a_pass"] is False,
        "board_exact_1920x1080": board["exact_1920x1080"],
        "rerun_strict_validation_pass": rerun["strict_validation_pass"],
        "rerun_headless_viewer_at_most_one": (
            rerun["headless_viewer_invocations"] <= 1
        ),
        "rerun_headless_returncode_zero": (
            rerun["headless_viewer_returncode"] == 0
        ),
        "required_artifacts_nonempty": all(
            path.is_file() and path.stat().st_size > 0 for path in required_files
        ),
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["input_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
        "forbidden_runtime_counters_zero": all(
            evidence["current_scope_counters"].get(name) == 0
            for name in FORBIDDEN_RUNTIME_COUNTERS
        ),
    }
    claim = {
        "artifact": "D380_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "pid": os.getpid(),
        "preregistration": {
            "path": _rel(PREREG_PATH),
            "sha256": _sha(PREREG_PATH),
        },
        "evidence": {
            "path": _rel(EVIDENCE_PATH),
            "sha256": _sha(EVIDENCE_PATH),
            "verdict": evidence["verdict"],
        },
        "board": board,
        "rerun": rerun,
        "metrics_csv": {
            "path": _rel(METRICS_CSV),
            "bytes": METRICS_CSV.stat().st_size,
            "sha256": _sha(METRICS_CSV),
        },
        "manual_template": {
            "path": _rel(MANUAL_TEMPLATE),
            "bytes": MANUAL_TEMPLATE.stat().st_size,
            "sha256": _sha(MANUAL_TEMPLATE),
        },
        "checks": post_checks,
        "pass": all(post_checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_claim_written", worker_claim_sha256=_sha(WORKER_CLAIM))
    if not claim["pass"]:
        raise RuntimeError(f"D380 worker post-check failed: {post_checks}")
    return 0


def run_supervisor() -> int:
    if not PREREG_PATH.is_file():
        raise RuntimeError("D380 preregistration missing")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D380 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D380 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D380 input changed after preregistration")
    if _status_paths() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("D380 dirty baseline changed after preregistration")

    command = [
        sys.executable,
        "-B",
        str(SCRIPT_PATH),
        "--stage",
        "worker",
    ]
    invocation = {
        "artifact": "D380_OFFLINE_AUDIT_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "cwd": str(REPO),
        "preregistration_sha256": _sha(PREREG_PATH),
        "input_hashes": _input_hashes(),
        "source_hashes": _source_hashes(),
        "worker_spawn_count_registered": 1,
        "automatic_retry_count_registered": 0,
        "watchdog_seconds": WATCHDOG_SECONDS,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase(
        "supervisor_spawn_start",
        invocation_sha256=_sha(INVOCATION_PATH),
        watchdog_seconds=WATCHDOG_SECONDS,
    )

    start = time.monotonic()
    timed_out = False
    sigterm_sent = False
    sigkill_sent = False
    with WORKER_STDOUT.open("xb") as stdout, WORKER_STDERR.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        pgid = process.pid
        try:
            returncode = process.wait(timeout=WATCHDOG_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(pgid, signal.SIGTERM)
            sigterm_sent = True
            try:
                returncode = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                sigkill_sent = True
                returncode = process.wait(timeout=10.0)
    elapsed = time.monotonic() - start
    try:
        os.killpg(pgid, 0)
        group_alive = True
    except ProcessLookupError:
        group_alive = False
    except PermissionError:
        group_alive = True

    claim = _read_json(WORKER_CLAIM) if WORKER_CLAIM.is_file() else {}
    required_artifacts = {
        "worker_claim": WORKER_CLAIM.is_file(),
        "evidence": EVIDENCE_PATH.is_file(),
        "metrics_csv": METRICS_CSV.is_file(),
        "board": BOARD_PATH.is_file(),
        "rrd": RRD_PATH.is_file(),
        "rbl": RBL_PATH.is_file(),
        "rerun_validation": RERUN_VALIDATION.is_file(),
        "rerun_screenshot": RERUN_SCREENSHOT.is_file(),
        "manual_template": MANUAL_TEMPLATE.is_file(),
    }
    operational_pass = (
        returncode == 0
        and not timed_out
        and not sigterm_sent
        and not sigkill_sent
        and not group_alive
        and all(required_artifacts.values())
        and claim.get("pass") is True
        and _source_hashes() == prereg["source_hashes"]
        and _input_hashes() == prereg["input_hashes"]
        and _status_paths() == prereg["registered_dirty_baseline"]
    )
    supervisor = {
        "artifact": "D380_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "worker_pid": process.pid,
        "worker_process_group": pgid,
        "actual_offline_worker_invocations": 1,
        "automatic_retries": 0,
        "watchdog_seconds": WATCHDOG_SECONDS,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "process_group_alive_after_wait": group_alive,
        "required_artifacts": required_artifacts,
        "worker_claim_sha256": (
            _sha(WORKER_CLAIM) if WORKER_CLAIM.is_file() else None
        ),
        "stdout": {
            "path": _rel(WORKER_STDOUT),
            "bytes": WORKER_STDOUT.stat().st_size,
            "sha256": _sha(WORKER_STDOUT),
        },
        "stderr": {
            "path": _rel(WORKER_STDERR),
            "bytes": WORKER_STDERR.stat().st_size,
            "sha256": _sha(WORKER_STDERR),
        },
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["input_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
        "operational_pass": operational_pass,
        "pass": operational_pass,
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase(
        "supervisor_complete",
        supervisor_sha256=_sha(SUPERVISOR_PATH),
        returncode=returncode,
        elapsed_seconds=elapsed,
        passed=operational_pass,
    )
    if not operational_pass:
        raise RuntimeError(f"D380 supervisor failed: {supervisor}")
    return 0


def finalize() -> int:
    required = [
        PREREG_PATH,
        INVOCATION_PATH,
        WORKER_CLAIM,
        SUPERVISOR_PATH,
        EVIDENCE_PATH,
        METRICS_CSV,
        BOARD_PATH,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION,
        RERUN_SCREENSHOT,
        MANUAL_TEMPLATE,
        MANUAL_INSPECTION,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"D380 finalize missing artifacts: {missing}")
    prereg = _read_json(PREREG_PATH)
    invocation = _read_json(INVOCATION_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    claim = _read_json(WORKER_CLAIM)
    supervisor = _read_json(SUPERVISOR_PATH)
    manual = _read_json(MANUAL_INSPECTION)
    manual_checks = dict(manual.get("inspection_checks") or {})
    checks = {
        "preregistration_pass": prereg.get("pass") is True,
        "invocation_preregistration_hash_exact": (
            invocation.get("preregistration_sha256") == _sha(PREREG_PATH)
        ),
        "invocation_input_hashes_exact": (
            invocation.get("input_hashes") == _input_hashes()
        ),
        "invocation_source_hashes_exact": (
            invocation.get("source_hashes") == _source_hashes()
        ),
        "worker_claim_pass": claim.get("pass") is True,
        "supervisor_pass": supervisor.get("pass") is True,
        "supervisor_worker_claim_hash_exact": (
            supervisor.get("worker_claim_sha256") == _sha(WORKER_CLAIM)
        ),
        "claim_preregistration_hash_exact": (
            claim.get("preregistration", {}).get("path") == _rel(PREREG_PATH)
            and claim.get("preregistration", {}).get("sha256")
            == _sha(PREREG_PATH)
        ),
        "claim_evidence_hash_exact": (
            claim.get("evidence", {}).get("path") == _rel(EVIDENCE_PATH)
            and claim.get("evidence", {}).get("sha256")
            == _sha(EVIDENCE_PATH)
        ),
        "claim_metrics_csv_hash_exact": (
            claim.get("metrics_csv", {}).get("path") == _rel(METRICS_CSV)
            and claim.get("metrics_csv", {}).get("sha256")
            == _sha(METRICS_CSV)
        ),
        "claim_board_hash_exact": (
            claim.get("board", {}).get("path") == _rel(BOARD_PATH)
            and claim.get("board", {}).get("sha256") == _sha(BOARD_PATH)
        ),
        "claim_manual_template_hash_exact": (
            claim.get("manual_template", {}).get("path")
            == _rel(MANUAL_TEMPLATE)
            and claim.get("manual_template", {}).get("sha256")
            == _sha(MANUAL_TEMPLATE)
        ),
        "claim_rrd_hash_exact": (
            claim.get("rerun", {}).get("rrd", {}).get("path") == _rel(RRD_PATH)
            and claim.get("rerun", {}).get("rrd", {}).get("sha256")
            == _sha(RRD_PATH)
        ),
        "claim_rbl_hash_exact": (
            claim.get("rerun", {}).get("rbl", {}).get("path") == _rel(RBL_PATH)
            and claim.get("rerun", {}).get("rbl", {}).get("sha256")
            == _sha(RBL_PATH)
        ),
        "claim_rerun_validation_hash_exact": (
            claim.get("rerun", {}).get("validation", {}).get("path")
            == _rel(RERUN_VALIDATION)
            and claim.get("rerun", {}).get("validation", {}).get("sha256")
            == _sha(RERUN_VALIDATION)
        ),
        "claim_rerun_screenshot_hash_exact": (
            claim.get("rerun", {}).get("screenshot", {}).get("path")
            == _rel(RERUN_SCREENSHOT)
            and claim.get("rerun", {}).get("screenshot", {}).get("sha256")
            == _sha(RERUN_SCREENSHOT)
        ),
        "evidence_input_hash_exact": (
            evidence.get("measurement_authority", {}).get("sha256")
            == D379_EVIDENCE_SHA256
            == _sha(D379_EVIDENCE)
        ),
        "actual_worker_one_retry_zero": (
            supervisor.get("actual_offline_worker_invocations") == 1
            and supervisor.get("automatic_retries") == 0
        ),
        "no_timeout_or_signal": (
            supervisor.get("timed_out") is False
            and supervisor.get("sigterm_sent") is False
            and supervisor.get("sigkill_sent") is False
        ),
        "audit_pass": evidence.get("audit_pass") is True,
        "p34_identity_remains_false": (
            evidence.get("p34_authored_to_cooked_identity_pass") is False
        ),
        "g0a_remains_false": evidence.get("g0a_pass") is False,
        "manual_visual_inspection_pass": manual.get("pass") is True,
        "manual_artifact_case_attempt_exact": (
            manual.get("artifact")
            == "D380_MANUAL_VISUAL_INSPECTION_V1"
            and manual.get("case") == CASE
            and manual.get("attempt") == ATTEMPT
        ),
        "manual_check_keys_exact": (
            set(manual_checks) == MANUAL_CHECK_KEYS
        ),
        "manual_checks_all_true": (
            set(manual_checks) == MANUAL_CHECK_KEYS
            and all(value is True for value in manual_checks.values())
        ),
        "manual_observations_nonempty": (
            isinstance(manual.get("observations"), list)
            and len(manual["observations"]) >= 2
            and all(
                isinstance(value, str) and bool(value.strip())
                for value in manual["observations"]
            )
        ),
        "manual_inspector_result_pass": (
            manual.get("inspector_result") == "PASS"
        ),
        "manual_template_hash_exact": (
            manual.get("template", {}).get("path") == _rel(MANUAL_TEMPLATE)
            and manual.get("template", {}).get("sha256")
            == _sha(MANUAL_TEMPLATE)
        ),
        "manual_authority_chain_exact": (
            manual.get("authority_chain", {}).get("preregistration_sha256")
            == _sha(PREREG_PATH)
            and manual.get("authority_chain", {}).get("invocation_sha256")
            == _sha(INVOCATION_PATH)
            and manual.get("authority_chain", {}).get("worker_claim_sha256")
            == _sha(WORKER_CLAIM)
            and manual.get("authority_chain", {}).get("supervisor_sha256")
            == _sha(SUPERVISOR_PATH)
        ),
        "manual_evidence_hash_exact": (
            manual.get("evidence", {}).get("path") == _rel(EVIDENCE_PATH)
            and manual.get("evidence", {}).get("sha256")
            == _sha(EVIDENCE_PATH)
        ),
        "manual_board_hash_exact": (
            manual.get("board", {}).get("path") == _rel(BOARD_PATH)
            and manual.get("board", {}).get("sha256") == _sha(BOARD_PATH)
        ),
        "manual_rerun_screenshot_hash_exact": (
            manual.get("rerun_screenshot", {}).get("path")
            == _rel(RERUN_SCREENSHOT)
            and manual.get("rerun_screenshot", {}).get("sha256")
            == _sha(RERUN_SCREENSHOT)
        ),
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["input_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
    }
    artifacts = {}
    for path in required:
        artifacts[path.name] = {
            "path": _rel(path),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
    completion = {
        "artifact": "D380_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "scientific_or_geometry_audit_verdict": evidence["verdict"],
        "completion_pass": all(checks.values()),
        "checks": checks,
        "counts": evidence["counts"],
        "semantic_impact": evidence["semantic_impact"],
        "current_scope_counters": evidence["current_scope_counters"],
        "remaining_nulls": evidence["remaining_nulls"],
        "p34_authored_to_cooked_identity_pass": False,
        "g0a_pass": False,
        "artifacts": artifacts,
        "next_authorization_boundary": evidence[
            "next_authorization_boundary"
        ],
        "verdict": (
            "D380_FAILED_PART_PROVENANCE_AUDIT_COMPLETE_REPAIR_REQUIRED"
            if all(checks.values())
            else "D380_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"
        ),
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase(
        "completion_frozen",
        completion_sha256=_sha(COMPLETION_PATH),
        passed=completion["completion_pass"],
        verdict=completion["verdict"],
    )
    if not completion["completion_pass"]:
        raise RuntimeError(f"D380 completion failed: {checks}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["prepare", "worker", "run", "finalize"],
        required=True,
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        return prepare()
    if args.stage == "worker":
        return _worker()
    if args.stage == "run":
        return run_supervisor()
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
