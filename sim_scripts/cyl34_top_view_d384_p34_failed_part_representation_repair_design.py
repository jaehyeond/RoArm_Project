#!/usr/bin/env python3
"""D384 offline-only design for repairing the 17 failed P34 convex parts.

The numeric authority is the immutable D379 authored/live identity JSON plus
the immutable D380 provenance JSON/CSV.  This program never imports or starts
Isaac Sim, Kit, PhysX, USD, Warp, CUDA, or robot hardware.  It only:

1. checks whether the 17 authored convex polyhedra satisfy the low-level PhysX
   points+polygons input preconditions documented by NVIDIA;
2. measures the size of a public-convexHull exact-cell fallback;
3. rejects any fallback that exceeds the frozen A64 total-count reference;
4. records a design, not a repaired asset or a live-identity result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASE = "g0a_d384"
ATTEMPT = "attempt1_p34_failed_part_representation_repair_design"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT
SCRIPT_PATH = Path(__file__).resolve()
START_HERE = REPO / "START_HERE.md"

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
D380_CSV = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d380/"
    "attempt1_failed_part_cook_provenance_semantic_impact_audit/"
    "d380_failed_part_metrics.csv"
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
    "d379_evidence": "8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5",
    "d380_evidence": "4c64d08e117501dd15a5836ce56ef8b963d188044beac465e645e53a17710bd1",
    "d380_csv": "885806a2164c0703d8ecf2594ff19afacd86a11fdb648bb593415e6281ec1d9c",
}

PREREG_PATH = OUT_DIR / "d384_preregistration.json"
PHASE_PATH = OUT_DIR / "d384_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d384_offline_design_invocation.json"
WORKER_STDOUT = OUT_DIR / "d384_offline_worker_stdout.log"
WORKER_STDERR = OUT_DIR / "d384_offline_worker_stderr.log"
WORKER_CLAIM = OUT_DIR / "d384_offline_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d384_offline_worker_supervisor.json"
EVIDENCE_PATH = OUT_DIR / "d384_p34_representation_repair_design_evidence.json"
METRICS_CSV = OUT_DIR / "d384_repair_design_part_metrics.csv"
BOARD_PATH = OUT_DIR / "d384_p34_representation_repair_design_1920x1080.png"
RRD_PATH = OUT_DIR / "d384_p34_representation_repair_design.rrd"
RBL_PATH = OUT_DIR / "d384_p34_representation_repair_design.rbl"
RERUN_VALIDATION = OUT_DIR / "d384_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d384_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d384_manual_visual_inspection_template.json"
MANUAL_INSPECTION = OUT_DIR / "d384_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d384_completion_summary.json"

RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = [
    "failed_profile_prism_authored_subpartition_v1",
    "failed_source_hull_authored_recursive_partition_v1",
]
WATCHDOG_SECONDS = 300.0
FROZEN_SURFACE_TOLERANCE_MM = 0.1
FROZEN_TOPOLOGY_VOLUME_RELATIVE = 0.005
CURRENT_A64_TOTAL_REFERENCE = 128
TETRA_POSITIVE_VOLUME_EPS_M3 = 1.0e-18
PROFILE_ROLES = {"fixed_jaw", "moving_jaw"}

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
}

ROLE_COLORS = {
    "fixed_jaw": [242, 142, 43, 205],
    "fixed_jaw_backbone": [225, 87, 89, 205],
    "moving_support": [78, 121, 167, 205],
    "moving_jaw": [89, 161, 79, 205],
    "moving_jaw_backbone": [176, 122, 161, 205],
}


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_x(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
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
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        )


def _status_paths() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _input_hashes() -> dict[str, str]:
    return {
        "d379_evidence": _sha(D379_EVIDENCE),
        "d380_evidence": _sha(D380_EVIDENCE),
        "d380_csv": _sha(D380_CSV),
    }


def _source_hashes() -> dict[str, str]:
    return {"script": _sha(SCRIPT_PATH)}


def _triangles(row: dict[str, Any]) -> np.ndarray:
    counts = np.asarray(row["face_vertex_counts"], dtype=np.int64)
    if len(counts) == 0 or not np.all(counts == 3):
        raise ValueError(f"non-triangle authored topology: {row['prim_name']}")
    indices = np.asarray(row["face_vertex_indices"], dtype=np.int64)
    if indices.size != counts.sum():
        raise ValueError(f"face span mismatch: {row['prim_name']}")
    return indices.reshape(-1, 3)


def _tetra_volumes(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    tetra = points[simplices]
    matrices = np.stack(
        (
            tetra[:, 1] - tetra[:, 0],
            tetra[:, 2] - tetra[:, 0],
            tetra[:, 3] - tetra[:, 0],
        ),
        axis=1,
    )
    return np.abs(np.linalg.det(matrices)) / 6.0


def _direct_polygon_preconditions(
    authored: dict[str, Any], failed: dict[str, Any]
) -> dict[str, Any]:
    points = np.asarray(authored["points_f32"], dtype=np.float64)
    triangles = _triangles(authored)
    areas = (
        np.linalg.norm(
            np.cross(
                points[triangles[:, 1]] - points[triangles[:, 0]],
                points[triangles[:, 2]] - points[triangles[:, 0]],
            ),
            axis=1,
        )
        * 0.5
    )
    incidence = np.bincount(
        triangles.reshape(-1), minlength=len(points)
    )
    unique_points = np.unique(points, axis=0)
    closure = failed["authored"]["closure"]
    convexity = failed["authored"]["supporting_plane_convexity"]
    checks = {
        "points_finite": bool(np.isfinite(points).all()),
        "points_unique": len(unique_points) == len(points),
        "vertices_4_to_64": 4 <= len(points) <= 64,
        "face_indices_in_range": bool(
            triangles.min() >= 0 and triangles.max() < len(points)
        ),
        "triangles_nonempty": len(triangles) > 0,
        "triangles_nondegenerate": bool(np.all(areas > 0.0)),
        "all_vertices_used": bool(np.all(incidence > 0)),
        "at_least_three_neighbor_polygons_per_vertex": bool(
            np.all(incidence >= 3)
        ),
        "closed_all_edges_twice": closure["all_edges_twice"] is True,
        "closed_opposite_winding": closure["opposite_winding"] is True,
        "all_triangle_planes_supporting": (
            convexity["all_triangle_planes_supporting"] is True
        ),
        "positive_topology_volume": (
            failed["authored"]["topology_volume_m3"] > 0.0
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "vertex_count": len(points),
        "triangle_polygon_count": len(triangles),
        "min_neighbor_polygon_count": int(incidence.min()),
        "max_neighbor_polygon_count": int(incidence.max()),
        "min_triangle_area_mm2": float(areas.min() * 1.0e6),
    }


def _paired_profile(points: np.ndarray) -> dict[str, Any]:
    unique = np.unique(points, axis=0)
    for axis in range(3):
        levels = np.unique(unique[:, axis])
        if len(levels) != 2:
            continue
        other = [index for index in range(3) if index != axis]
        sections = []
        for level in levels:
            section = unique[unique[:, axis] == level][:, other]
            sections.append(set(map(tuple, section)))
        if sections[0] != sections[1]:
            continue
        section = np.asarray(sorted(sections[0]), dtype=np.float64)
        hull = ConvexHull(section)
        ordered = section[hull.vertices]
        triangulation = Delaunay(ordered)
        triangles_2d = np.asarray(triangulation.simplices, dtype=np.int64)
        child_volumes = []
        child_meshes = []
        for child_index, tri in enumerate(triangles_2d):
            cross = ordered[tri]
            area = abs(
                np.cross(cross[1] - cross[0], cross[2] - cross[0])
            ) * 0.5
            child_volumes.append(float(area * abs(levels[1] - levels[0])))
            child_points = []
            for level in levels:
                for pair in cross:
                    point = np.zeros(3, dtype=np.float64)
                    point[axis] = level
                    point[other] = pair
                    child_points.append(point)
            child_points_array = np.asarray(child_points)
            child_hull = ConvexHull(child_points_array)
            child_meshes.append(
                {
                    "child_index": child_index,
                    "vertices_m": child_points_array,
                    "triangles": np.asarray(
                        child_hull.simplices, dtype=np.int64
                    ),
                }
            )
        return {
            "axis": "xyz"[axis],
            "axis_index": axis,
            "levels_m": [float(levels[0]), float(levels[1])],
            "cross_section_point_count": len(ordered),
            "all_cross_section_points_extreme": (
                len(hull.vertices) == len(section)
            ),
            "child_triangular_prism_count": len(triangles_2d),
            "child_volume_sum_m3": float(sum(child_volumes)),
            "child_meshes": child_meshes,
        }
    raise ValueError("failed profile part is not an exact paired extrusion")


def _source_hull_tetra_fallback(
    points: np.ndarray, authored_volume_m3: float
) -> dict[str, Any]:
    unique = np.unique(points, axis=0)
    triangulation = Delaunay(unique)
    simplices = np.asarray(triangulation.simplices, dtype=np.int64)
    volumes = _tetra_volumes(unique, simplices)
    keep = volumes > TETRA_POSITIVE_VOLUME_EPS_M3
    positive = simplices[keep]
    positive_volumes = volumes[keep]
    used = set(map(int, np.unique(positive)))
    return {
        "raw_delaunay_simplex_count": len(simplices),
        "positive_tetra_count": len(positive),
        "zero_or_sliver_rejected_count": int(np.count_nonzero(~keep)),
        "positive_volume_sum_m3": float(positive_volumes.sum()),
        "relative_volume_error": float(
            abs(positive_volumes.sum() - authored_volume_m3)
            / authored_volume_m3
        ),
        "input_vertex_count": len(unique),
        "input_vertices_used_by_positive_tetra": len(used),
        "all_input_vertices_used_by_positive_tetra": len(used) == len(unique),
        "minimum_positive_tetra_volume_mm3": float(
            positive_volumes.min() * 1.0e9
        ),
        "maximum_child_vertex_count": 4,
        "positive_simplices": positive,
        "unique_points": unique,
    }


def _greedy_merge_tetra_cells(
    points: np.ndarray,
    positive_simplices: np.ndarray,
    *,
    maximum_vertices_per_child: int,
) -> dict[str, Any]:
    """Merge face-adjacent tetrahedra only when their exact union is convex."""

    points = np.asarray(points, dtype=np.float64)
    simplices = np.asarray(positive_simplices, dtype=np.int64)
    volumes = _tetra_volumes(points, simplices)
    cells = [
        {
            "vertices": frozenset(map(int, simplex)),
            "volume_m3": float(volume),
        }
        for simplex, volume in zip(simplices, volumes, strict=True)
    ]
    merge_count = 0
    while True:
        best = None
        for left_index in range(len(cells)):
            for right_index in range(left_index + 1, len(cells)):
                left = cells[left_index]
                right = cells[right_index]
                shared = left["vertices"] & right["vertices"]
                if len(shared) < 3:
                    continue
                union = left["vertices"] | right["vertices"]
                if len(union) > maximum_vertices_per_child:
                    continue
                try:
                    hull = ConvexHull(points[list(union)])
                except Exception:
                    continue
                summed = left["volume_m3"] + right["volume_m3"]
                relative = abs(float(hull.volume) - summed) / max(
                    summed, 1.0e-30
                )
                if relative > 2.0e-6:
                    continue
                # Deterministic preference: fewer union vertices, then larger
                # merged volume, then stable cell indices.
                score = (
                    len(union),
                    -summed,
                    left_index,
                    right_index,
                )
                if best is None or score < best[0]:
                    best = (
                        score,
                        left_index,
                        right_index,
                        frozenset(union),
                        float(hull.volume),
                    )
        if best is None:
            break
        _, left_index, right_index, union, volume = best
        cells = [
            cell
            for index, cell in enumerate(cells)
            if index not in (left_index, right_index)
        ]
        cells.append({"vertices": union, "volume_m3": volume})
        merge_count += 1
    return {
        "maximum_vertices_per_child": maximum_vertices_per_child,
        "merged_child_count": len(cells),
        "merge_count": merge_count,
        "maximum_observed_child_vertices": max(
            len(cell["vertices"]) for cell in cells
        ),
        "minimum_observed_child_vertices": min(
            len(cell["vertices"]) for cell in cells
        ),
        "merged_volume_sum_m3": float(
            sum(cell["volume_m3"] for cell in cells)
        ),
    }


def _schema_capability() -> dict[str, Any]:
    schema_text = PHYSX_SCHEMA.read_text(encoding="utf-8")
    class_start = schema_text.index('class "PhysxConvexHullCollisionAPI"')
    class_end = schema_text.index(
        'class "PhysxConvexDecompositionCollisionAPI"', class_start
    )
    block = schema_text[class_start:class_end]
    declared = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith(("int physx", "float physx", "token physx")):
            declared.append(stripped.split("=", 1)[0].strip())
    selector_markers = [
        "directPolygon",
        "providedPolygons",
        "computeConvex",
        "disableComputeConvex",
    ]
    return {
        "installed_schema_path": str(PHYSX_SCHEMA),
        "installed_schema_sha256": _sha(PHYSX_SCHEMA),
        "installed_property_db_path": str(PHYSX_PROPERTY_DB),
        "installed_property_db_sha256": _sha(PHYSX_PROPERTY_DB),
        "convex_hull_api_declared_fields": declared,
        "direct_polygon_selector_markers_searched": selector_markers,
        "public_usd_direct_polygon_selector_found": any(
            marker in block for marker in selector_markers
        ),
        "interpretation": (
            "The installed public PhysxConvexHullCollisionAPI exposes "
            "hullVertexLimit and minThickness only. This does not disprove a "
            "lower-level/private API, but no public USD selector for the "
            "PhysX points+polygons direct path is registered here."
        ),
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any]]:
    d379 = _read_json(D379_EVIDENCE)
    d380 = _read_json(D380_EVIDENCE)
    authored_map = {
        (row["body"], row["prim_name"]): row
        for row in d379["authored_readback"]["rows"]
    }
    callback_map = {
        (row["body"], row["prim_name"]): row
        for row in d379["callback_rows"]
    }
    failed_rows = []
    profile_children = []
    profile_count = 0
    source_count = 0
    source_tetra_count = 0
    source_merged_count = 0
    source_zero_slivers = 0
    direct_pass_count = 0
    representative_profile = None
    representative_source = None

    for failed in d380["failed_parts"]:
        key = (failed["body"], failed["prim_name"])
        authored = authored_map[key]
        callback = callback_map[key]
        points = np.asarray(authored["points_f32"], dtype=np.float64)
        triangles = _triangles(authored)
        direct = _direct_polygon_preconditions(authored, failed)
        direct_pass_count += int(direct["pass"])
        semantic_class = (
            "manual_profile_prism"
            if failed["role"] in PROFILE_ROLES
            else "source_3d_convex_hull"
        )
        row = {
            "body": failed["body"],
            "prim_name": failed["prim_name"],
            "name": failed["name"],
            "role": failed["role"],
            "semantic_class": semantic_class,
            "authored_vertices": len(points),
            "authored_triangle_polygons": len(triangles),
            "cooked_vertices": callback["live_callback_vertex_count"],
            "omitted_vertices": failed["vertex_provenance"][
                "omitted_authored_vertex_count"
            ],
            "surface_inward_mm": failed["difference"][
                "surface_authored_to_cooked_mm"
            ],
            "volume_loss_percent": failed["difference"][
                "signed_part_volume_loss_percent"
            ],
            "direct_polygon_preconditions": direct,
        }
        if semantic_class == "manual_profile_prism":
            profile = _paired_profile(points)
            profile_count += profile["child_triangular_prism_count"]
            profile_children.extend(profile["child_meshes"])
            row["exact_cell_fallback"] = {
                key: value
                for key, value in profile.items()
                if key != "child_meshes"
            }
            row["exact_cell_fallback"]["relative_volume_error"] = float(
                abs(
                    profile["child_volume_sum_m3"]
                    - failed["authored"]["topology_volume_m3"]
                )
                / failed["authored"]["topology_volume_m3"]
            )
            row["exact_cell_fallback"][
                "maximum_child_vertex_count"
            ] = 6
            if representative_profile is None:
                representative_profile = {
                    "row": row,
                    "vertices_m": points,
                    "triangles": triangles,
                    "children": profile["child_meshes"],
                }
        else:
            fallback = _source_hull_tetra_fallback(
                points, failed["authored"]["topology_volume_m3"]
            )
            merged = _greedy_merge_tetra_cells(
                fallback["unique_points"],
                fallback["positive_simplices"],
                maximum_vertices_per_child=8,
            )
            merged["relative_volume_error"] = float(
                abs(
                    merged["merged_volume_sum_m3"]
                    - failed["authored"]["topology_volume_m3"]
                )
                / failed["authored"]["topology_volume_m3"]
            )
            source_count += fallback["positive_tetra_count"]
            source_tetra_count += fallback["positive_tetra_count"]
            source_merged_count += merged["merged_child_count"]
            source_zero_slivers += fallback[
                "zero_or_sliver_rejected_count"
            ]
            row["exact_cell_fallback"] = {
                key: value
                for key, value in fallback.items()
                if key not in {"positive_simplices", "unique_points"}
            }
            row["recursive_exact_partition"] = merged
            if (
                representative_source is None
                or row["surface_inward_mm"]
                > representative_source["row"]["surface_inward_mm"]
            ):
                representative_source = {
                    "row": row,
                    "vertices_m": points,
                    "triangles": triangles,
                }
        failed_rows.append(row)

    profile_rows = [
        row
        for row in failed_rows
        if row["semantic_class"] == "manual_profile_prism"
    ]
    source_rows = [
        row
        for row in failed_rows
        if row["semantic_class"] == "source_3d_convex_hull"
    ]
    passing_parts = 34 - len(failed_rows)
    full_exact_cell_total = passing_parts + profile_count + source_count
    recursive_partition_total = (
        passing_parts + profile_count + source_merged_count
    )
    schema = _schema_capability()

    negative_controls = {
        "relax_surface_gate_to_0_7mm_rejected": (
            FROZEN_SURFACE_TOLERANCE_MM == 0.1
        ),
        "relax_volume_gate_to_7percent_rejected": (
            FROZEN_TOPOLOGY_VOLUME_RELATIVE == 0.005
        ),
        "reuse_cooked_vertices_as_new_authored_rejected": all(
            row["omitted_vertices"] > 0 for row in failed_rows
        ),
        "pretend_hull_vertex_overflow_rejected": max(
            row["authored_vertices"] for row in failed_rows
        )
        < 64,
        "full_exact_cell_fallback_over_budget_rejected": (
            full_exact_cell_total > CURRENT_A64_TOTAL_REFERENCE
        ),
        "claim_public_usd_direct_selector_rejected": (
            schema["public_usd_direct_polygon_selector_found"] is False
        ),
        "drop_one_profile_child_breaks_volume": (
            profile_count > 0
            and min(
                row["exact_cell_fallback"]["child_volume_sum_m3"]
                for row in profile_rows
            )
            > 0.0
        ),
        "include_zero_or_sliver_tetra_rejected": source_zero_slivers > 0,
    }

    direct_reserve = {
        "name": "R0_direct_points_plus_polygons",
        "changed_failed_parts": len(failed_rows),
        "unchanged_passing_parts": passing_parts,
        "total_collider_parts_if_supported": 34,
        "authored_input_preconditions_passed": direct_pass_count,
        "authored_input_preconditions_total": len(failed_rows),
        "authored_input_preconditions_pass": (
            direct_pass_count == len(failed_rows)
        ),
        "public_usd_selector_found": schema[
            "public_usd_direct_polygon_selector_found"
        ],
        "live_runtime_capability": None,
        "live_identity": None,
        "materialization_ready": False,
        "scope_fit": False,
        "scope_fit_reason": (
            "No public USD selector is installed; a C++ or opaque cooked-data "
            "bridge would be a new representation pipeline, not this case's "
            "minimum authored-child split."
        ),
        "status": "RESERVE_ONLY_NEW_BRIDGE_REQUIRED",
    }
    recursive_candidate = {
        "name": "R1_authored_recursive_exact_partition",
        "unchanged_passing_parts": passing_parts,
        "profile_failed_parts": len(profile_rows),
        "profile_triangular_prism_children": profile_count,
        "source_failed_parts": len(source_rows),
        "source_merged_convex_children_max8_vertices": source_merged_count,
        "total_collider_parts": recursive_partition_total,
        "count_budget": CURRENT_A64_TOTAL_REFERENCE,
        "count_budget_basis": (
            "project-authored do-not-exceed reference: current A64 has "
            "64 link5 + 64 gripper_link = 128; this is not an NVIDIA limit"
        ),
        "count_budget_pass": (
            recursive_partition_total < CURRENT_A64_TOTAL_REFERENCE
        ),
        "geometry_live_identity": None,
        "status": (
            "OFFLINE_ADMISSIBLE_PENDING_LIVE"
            if recursive_partition_total < CURRENT_A64_TOTAL_REFERENCE
            else "REJECTED_PART_COUNT_ABOVE_A64_REFERENCE"
        ),
    }
    tetra_upper_bound = {
        "name": "R2_exact_tetra_upper_bound_control",
        "profile_triangular_prism_children": profile_count,
        "source_positive_tetra_children": source_tetra_count,
        "source_zero_or_sliver_tetra_rejected": source_zero_slivers,
        "total_collider_parts": full_exact_cell_total,
        "count_budget": CURRENT_A64_TOTAL_REFERENCE,
        "count_budget_pass": (
            full_exact_cell_total < CURRENT_A64_TOTAL_REFERENCE
        ),
        "status": "REJECTED_PART_COUNT_EXPLOSION",
    }

    design_checks = {
        "immutable_input_hashes_exact": (
            _input_hashes() == EXPECTED_INPUT_SHA256
        ),
        "failed_parts_exact_17": len(failed_rows) == 17,
        "profile_prisms_exact_9": len(profile_rows) == 9,
        "source_hulls_exact_8": len(source_rows) == 8,
        "direct_polygon_input_preconditions_17_of_17": (
            direct_pass_count == 17
        ),
        "profile_exact_cell_count_46": profile_count == 46,
        "source_positive_tetra_count_495": source_tetra_count == 495,
        "source_recursive_max8_child_count_205": (
            source_merged_count == 205
        ),
        "recursive_partition_total_268": (
            recursive_partition_total == 268
        ),
        "recursive_partition_rejected_above_128": (
            recursive_partition_total > CURRENT_A64_TOTAL_REFERENCE
        ),
        "full_exact_cell_total_558": full_exact_cell_total == 558,
        "full_exact_cell_fallback_rejected": (
            tetra_upper_bound["count_budget_pass"] is False
        ),
        "public_usd_direct_selector_not_claimed": (
            schema["public_usd_direct_polygon_selector_found"] is False
        ),
        "direct_reserve_live_identity_remains_null": (
            direct_reserve["live_identity"] is None
        ),
        "frozen_surface_gate_unchanged": (
            FROZEN_SURFACE_TOLERANCE_MM == 0.1
        ),
        "frozen_volume_gate_unchanged": (
            FROZEN_TOPOLOGY_VOLUME_RELATIVE == 0.005
        ),
        "negative_controls_all_pass": all(negative_controls.values()),
        "forbidden_runtime_counters_zero": all(
            value == 0 for value in FORBIDDEN_COUNTERS.values()
        ),
    }
    evidence = {
        "artifact": "D384_P34_FAILED_PART_REPRESENTATION_REPAIR_DESIGN_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Design the smallest non-gate-relaxing representation repair for "
            "the 17 P34 convex parts that D379/D380 proved were cooked inward."
        ),
        "new_variables": NEW_VARIABLES,
        "measurement_authority": {
            "inputs": {
                "d379_evidence": {
                    "path": _rel(D379_EVIDENCE),
                    "sha256": _sha(D379_EVIDENCE),
                },
                "d380_evidence": {
                    "path": _rel(D380_EVIDENCE),
                    "sha256": _sha(D380_EVIDENCE),
                },
                "d380_csv": {
                    "path": _rel(D380_CSV),
                    "sha256": _sha(D380_CSV),
                },
            },
            "rerun_float32_is_inspection_only": True,
        },
        "installed_stack": {
            "isaac_sim": "5.1.0.0 (inherited D379 inventory; not launched)",
            "omni_physx_schema": "107.3.26",
            "kit": "107.3.3",
            "rerun_sdk": importlib.metadata.version("rerun-sdk"),
        },
        "official_sources": [
            {
                "title": "Omni Physics 107.3 - Colliders",
                "url": (
                    "https://docs.omniverse.nvidia.com/kit/docs/"
                    "omni_physics/107.3/dev_guide/"
                    "rigid_bodies_articulations/collision.html"
                ),
                "use": (
                    "primitive colliders map precisely; mesh convexHull and "
                    "convexDecomposition are cooked approximations"
                ),
            },
            {
                "title": "PhysX SDK 5.6.1 - Geometry",
                "url": (
                    "https://nvidia-omniverse.github.io/PhysX/"
                    "physx/5.6.1/docs/Geometry.html"
                ),
                "version_boundary": (
                    "official supporting SDK documentation; installed exact "
                    "PhysX SDK semver is not independently exposed"
                ),
                "use": (
                    "Quickhull returns an authored-vertex subset and may drop "
                    "vertices within planeTolerance; points+polygons can be "
                    "validated and cooked directly at the low-level SDK"
                ),
            },
        ],
        "installed_public_schema_capability": schema,
        "classification": {
            "failed_total": len(failed_rows),
            "manual_profile_prisms": len(profile_rows),
            "source_3d_convex_hulls": len(source_rows),
        },
        "repair_candidates": {
            "registered_recursive_partition": recursive_candidate,
            "exact_tetra_upper_bound": tetra_upper_bound,
            "direct_polygon_bridge_reserve": direct_reserve,
        },
        "failed_parts": failed_rows,
        "negative_controls": {
            "checks": negative_controls,
            "passed": sum(bool(value) for value in negative_controls.values()),
            "total": len(negative_controls),
            "pass": all(negative_controls.values()),
        },
        "current_scope_counters": FORBIDDEN_COUNTERS,
        "design_checks": design_checks,
        "design_audit_pass": all(design_checks.values()),
        "admissible_low_count_candidate_found": (
            recursive_candidate["count_budget_pass"] is True
        ),
        "repair_design_pass": False,
        "p34_authored_to_cooked_identity_pass": False,
        "repair_materialized": False,
        "live_identity_pass": None,
        "cylinder_29x50_rendered_or_measured": False,
        "g0a_pass": False,
        "next_authorization_boundary": (
            "No asset materialization is justified. A separate approved "
            "offline case must redesign the eight source hulls with semantic "
            "primitive/low-count parts while preserving the D372 void, source "
            "coverage, and contact-seed gates. Direct-polygon C++ bridging is "
            "reserve-only. Cylinder physics remains separate."
        ),
        "verdict": (
            "D384_REPRESENTATION_REPAIR_DESIGN_NO_ADMISSIBLE_LOW_COUNT_CANDIDATE_FAIL_STOP"
            if all(design_checks.values())
            and recursive_partition_total > CURRENT_A64_TOTAL_REFERENCE
            else "D384_REPAIR_DESIGN_INTEGRITY_FAIL_STOP"
        ),
    }
    visual = {
        "representative_profile": representative_profile,
        "representative_source": representative_source,
    }
    return evidence, visual


def _write_csv(evidence: dict[str, Any]) -> None:
    fields = [
        "body",
        "prim_name",
        "role",
        "semantic_class",
        "authored_vertices",
        "cooked_vertices",
        "omitted_vertices",
        "surface_inward_mm",
        "volume_loss_percent",
        "direct_polygon_preconditions_pass",
        "exact_cell_child_count",
        "exact_cell_relative_volume_error",
    ]
    with METRICS_CSV.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in evidence["failed_parts"]:
            fallback = row["exact_cell_fallback"]
            child_count = fallback.get(
                "child_triangular_prism_count",
                fallback.get("positive_tetra_count"),
            )
            writer.writerow(
                {
                    "body": row["body"],
                    "prim_name": row["prim_name"],
                    "role": row["role"],
                    "semantic_class": row["semantic_class"],
                    "authored_vertices": row["authored_vertices"],
                    "cooked_vertices": row["cooked_vertices"],
                    "omitted_vertices": row["omitted_vertices"],
                    "surface_inward_mm": row["surface_inward_mm"],
                    "volume_loss_percent": row["volume_loss_percent"],
                    "direct_polygon_preconditions_pass": row[
                        "direct_polygon_preconditions"
                    ]["pass"],
                    "exact_cell_child_count": child_count,
                    "exact_cell_relative_volume_error": fallback[
                        "relative_volume_error"
                    ],
                }
            )


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "width": width,
        "height": height,
        "exact_1920x1080": width == 1920 and height == 1080,
    }


def _render_board(
    evidence: dict[str, Any], visual: dict[str, Any]
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
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#F8FAFC")
    grid = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[0.56, 0.44],
        width_ratios=[0.37, 0.34, 0.29],
        hspace=0.20,
        wspace=0.14,
    )

    profile = visual["representative_profile"]
    axis_profile = fig.add_subplot(grid[0, 0], projection="3d")
    parent = profile["vertices_m"] * 1000.0
    axis_profile.add_collection3d(
        Poly3DCollection(
            parent[profile["triangles"]],
            facecolor="#D9DEE7",
            edgecolor="#475569",
            alpha=0.18,
            linewidth=0.35,
        )
    )
    palette = plt.cm.viridis(
        np.linspace(0.10, 0.90, len(profile["children"]))
    )
    for color, child in zip(palette, profile["children"], strict=True):
        vertices = child["vertices_m"] * 1000.0
        axis_profile.add_collection3d(
            Poly3DCollection(
                vertices[child["triangles"]],
                facecolor=color,
                edgecolor="#0F172A",
                alpha=0.46,
                linewidth=0.28,
            )
        )
    lower, upper = parent.min(axis=0), parent.max(axis=0)
    center = (lower + upper) * 0.5
    radius = max(float((upper - lower).max()) * 0.65, 1.0)
    axis_profile.set_xlim(center[0] - radius, center[0] + radius)
    axis_profile.set_ylim(center[1] - radius, center[1] + radius)
    axis_profile.set_zlim(center[2] - radius, center[2] + radius)
    axis_profile.set_box_aspect((1, 1, 1))
    axis_profile.view_init(elev=23, azim=-57)
    axis_profile.set_axis_off()
    axis_profile.set_title(
        "Profile-prism exact-cell control\n"
        f"{profile['row']['prim_name']} -> "
        f"{len(profile['children'])} triangular prisms",
        fontproperties=bold,
        fontsize=10.5,
    )

    source = visual["representative_source"]
    axis_source = fig.add_subplot(grid[0, 1], projection="3d")
    source_vertices = source["vertices_m"] * 1000.0
    axis_source.add_collection3d(
        Poly3DCollection(
            source_vertices[source["triangles"]],
            facecolor="#4E79A7",
            edgecolor="#172554",
            alpha=0.56,
            linewidth=0.38,
        )
    )
    lower, upper = source_vertices.min(axis=0), source_vertices.max(axis=0)
    center = (lower + upper) * 0.5
    radius = max(float((upper - lower).max()) * 0.62, 1.0)
    axis_source.set_xlim(center[0] - radius, center[0] + radius)
    axis_source.set_ylim(center[1] - radius, center[1] + radius)
    axis_source.set_zlim(center[2] - radius, center[2] + radius)
    axis_source.set_box_aspect((1, 1, 1))
    axis_source.view_init(elev=18, azim=-66)
    axis_source.set_axis_off()
    fallback = source["row"]["exact_cell_fallback"]
    recursive = source["row"]["recursive_exact_partition"]
    axis_source.set_title(
        "Worst source-hull witness\n"
        f"{source['row']['prim_name']}: "
        f"{fallback['positive_tetra_count']} tetra -> "
        f"{recursive['merged_child_count']} max-8-vertex cells",
        fontproperties=bold,
        fontsize=10.5,
    )

    decision = fig.add_subplot(grid[0, 2])
    decision.axis("off")
    candidates = evidence["repair_candidates"]
    decision_lines = [
        "DESIGN DECISION",
        "",
        "R0 | direct points + polygons (reserve)",
        "  total parts: 34",
        "  input preconditions: 17/17 PASS",
        "  public USD selector: NOT FOUND",
        "  new C++/opaque bridge required",
        "",
        "R1 | registered authored partition",
        "  profile children: 46",
        "  source max-8 cells: 205",
        f"  total: "
        f"{candidates['registered_recursive_partition']['total_collider_parts']}",
        "  budget <128: FAIL -> REJECT",
        "",
        "R2 | exact tetra upper bound",
        "  profile/source children: 46/495",
        f"  total: "
        f"{candidates['exact_tetra_upper_bound']['total_collider_parts']}",
        "  budget <128: FAIL -> REJECT",
        "",
        "NO ASSET WAS BUILT",
        "P34 live identity remains false.",
    ]
    decision.text(
        0.02,
        0.98,
        "\n".join(decision_lines),
        va="top",
        ha="left",
        fontproperties=regular,
        fontsize=10.0,
        linespacing=1.28,
        color="#1E293B",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#EFF6FF",
            "edgecolor": "#93C5FD",
        },
    )

    rows = evidence["failed_parts"]
    axis_bar = fig.add_subplot(grid[1, 0:2])
    ordered = sorted(rows, key=lambda item: item["surface_inward_mm"])
    labels = [
        ("L5/" if row["body"] == "link5" else "GR/")
        + row["prim_name"].split("_", 1)[0]
        + ("/P" if row["semantic_class"] == "manual_profile_prism" else "/H")
        for row in ordered
    ]
    values = [row["surface_inward_mm"] for row in ordered]
    colors = [
        "#F28E2B"
        if row["semantic_class"] == "manual_profile_prism"
        else "#4E79A7"
        for row in ordered
    ]
    y = np.arange(len(ordered))
    axis_bar.barh(y, values, color=colors, alpha=0.90)
    axis_bar.axvline(
        0.1, color="#991B1B", linestyle="--", linewidth=1.5
    )
    axis_bar.set_yticks(y)
    axis_bar.set_yticklabels(labels, fontproperties=regular, fontsize=8.0)
    axis_bar.set_xlabel(
        "Observed authored-to-cooked inward distance (mm)",
        fontproperties=regular,
        fontsize=9.6,
    )
    axis_bar.set_title(
        "Frozen D380 failures | P=9 profile prisms, H=8 source hulls",
        fontproperties=bold,
        fontsize=11.0,
    )
    axis_bar.grid(axis="x", alpha=0.20)

    checks_axis = fig.add_subplot(grid[1, 2])
    checks_axis.axis("off")
    checks = evidence["design_checks"]
    check_lines = [
        "OFFLINE GATES",
        f"Direct input: 17/17 PASS",
        f"Profile/source split: 9/8",
        f"Frozen surface gate: 0.1 mm",
        f"Frozen volume gate: 0.5%",
        f"Negative controls: "
        f"{evidence['negative_controls']['passed']}/"
        f"{evidence['negative_controls']['total']}",
        f"Forbidden runtime counters: all 0",
        "",
        "What PASS means",
        "The repair design is auditable.",
        "It does NOT mean PhysX preserves it.",
        "",
        "Next separate approval",
        "Offline semantic low-count redesign",
        "of the eight source hulls.",
        "",
        f"Design checks: {sum(checks.values())}/{len(checks)}",
        "g0a_pass=false",
    ]
    checks_axis.text(
        0.03,
        0.98,
        "\n".join(check_lines),
        va="top",
        ha="left",
        fontproperties=regular,
        fontsize=9.9,
        linespacing=1.30,
        color="#1F2937",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#F0FDF4",
            "edgecolor": "#86EFAC",
        },
    )

    fig.suptitle(
        "D384 | P34 failed-part representation repair design",
        fontproperties=bold,
        fontsize=20.0,
        color="#0F172A",
        y=0.986,
    )
    fig.text(
        0.5,
        0.953,
        (
            "Immutable D379/D380 evidence only | no Isaac/PhysX/USD/cylinder/"
            "physics/q5/contact | direct-path capability remains untested"
        ),
        ha="center",
        fontproperties=regular,
        fontsize=10.2,
        color="#475569",
    )
    fig.text(
        0.5,
        0.012,
        (
            "No admissible low-count repair was found: registered exact "
            "partition=268 parts, exact tetra upper bound=558, and the "
            "34-part direct path requires a new bridge."
        ),
        ha="center",
        fontproperties=regular,
        fontsize=9.6,
        color="#334155",
    )
    fig.subplots_adjust(
        left=0.055, right=0.985, top=0.92, bottom=0.068
    )
    fig.savefig(BOARD_PATH, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    info = _png_info(BOARD_PATH)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"board dimension failure: {info}")
    return info


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/",
                contents="/d384/profile/**",
                name="Profile prism: parent and exact children",
            ),
            rrb.Spatial3DView(
                origin="/",
                contents="/d384/source/**",
                name="Source hull: recursive-partition witness",
            ),
            rrb.TextDocumentView(
                origin="/metadata/run",
                contents="/metadata/run",
                name="D384 design decision",
            ),
            column_shares=[0.36, 0.36, 0.28],
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

    profile = visual["representative_profile"]
    source = visual["representative_source"]
    meshes = [
        {
            "entity_path": "d384/profile/authored_parent",
            "coordinate_frame": "tf#/",
            "vertices_m": profile["vertices_m"],
            "triangles": profile["triangles"],
            "color_rgba": [190, 196, 204, 72],
            "static": True,
            "representation": "immutable authored parent",
        },
        {
            "entity_path": "d384/source/authored_parent",
            "coordinate_frame": "tf#/",
            "vertices_m": source["vertices_m"],
            "triangles": source["triangles"],
            "color_rgba": [78, 121, 167, 210],
            "static": True,
            "representation": "immutable authored source hull; not live-cooked",
        },
    ]
    for child in profile["children"]:
        meshes.append(
            {
                "entity_path": (
                    "d384/profile/exact_children/"
                    f"child_{child['child_index']:02d}"
                ),
                "coordinate_frame": "tf#/",
                "vertices_m": child["vertices_m"],
                "triangles": child["triangles"],
                "color_rgba": [89, 161, 79, 118],
                "static": True,
                "representation": "offline exact-cell fallback",
            }
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
        metadata = f"metadata/meshes/{path.replace('/', '__')}"
        expected_entities.update({path, metadata})
        component_contract[path] = mesh_components
        component_contract[metadata] = ["TextDocument:text"]

    summary = {
        "case": CASE,
        "verdict": evidence["verdict"],
        "registered_partition": "46 profile + 205 source cells; 268 total; rejected",
        "exact_upper_bound": "46 profile + 495 tetra; 558 total; rejected",
        "direct_bridge_reserve": "34 total theoretical; public USD selector absent",
        "profile_source_failed_parts": "9 / 8",
        "asset_materialized": False,
        "live_identity": None,
        "cylinder": None,
        "g0a_pass": False,
        "scope": (
            "offline D379/D380 evidence; Rerun Float32 inspection only"
        ),
    }
    original_builder = viz_debug.build_rerun_blueprint

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d384_repair_design":
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
            recording_id="g0a_d384_repair_design",
            blueprint_path=RBL_PATH,
            blueprint_mode="d384_repair_design",
            live_viewer=False,
            app_id="roarm_g0a_d384_repair_design",
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
        raise FileExistsError(f"forward-only output exists: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    input_hashes = _input_hashes()
    schema_capability = _schema_capability()
    checks = {
        "input_hashes_exact": input_hashes == EXPECTED_INPUT_SHA256,
        "start_here_registers_d384": (
            "D384 [p34_failed_part_representation_repair_design]"
            in START_HERE.read_text(encoding="utf-8")
        ),
        "new_variables_exact_two": len(NEW_VARIABLES) == 2,
        "surface_gate_frozen_0_1mm": (
            FROZEN_SURFACE_TOLERANCE_MM == 0.1
        ),
        "volume_gate_frozen_0_5percent": (
            FROZEN_TOPOLOGY_VOLUME_RELATIVE == 0.005
        ),
        "count_budget_is_project_reference_not_nvidia_limit": (
            CURRENT_A64_TOTAL_REFERENCE == 128
        ),
        "installed_schema_exists": PHYSX_SCHEMA.is_file(),
        "installed_property_db_exists": PHYSX_PROPERTY_DB.is_file(),
        "public_usd_direct_selector_not_preclaimed": (
            schema_capability[
                "public_usd_direct_polygon_selector_found"
            ]
            is False
        ),
        "rerun_cli_exists": RERUN_CLI.is_file(),
        "forbidden_counters_registered_zero": all(
            value == 0 for value in FORBIDDEN_COUNTERS.values()
        ),
    }
    prereg = {
        "artifact": "D384_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Offline-only representation design for the 17 inward-elided "
            "P34 parts before any asset/live identity work."
        ),
        "new_variables": NEW_VARIABLES,
        "frozen_inputs": {
            "paths": {
                "d379_evidence": _rel(D379_EVIDENCE),
                "d380_evidence": _rel(D380_EVIDENCE),
                "d380_csv": _rel(D380_CSV),
            },
            "sha256": input_hashes,
        },
        "frozen_gates": {
            "surface_tolerance_mm": FROZEN_SURFACE_TOLERANCE_MM,
            "authored_topology_volume_relative": (
                FROZEN_TOPOLOGY_VOLUME_RELATIVE
            ),
            "no_gate_relaxation": True,
        },
        "candidate_contract": {
            "registered_partition": (
                "split the nine authored profile prisms exactly and merge "
                "Delaunay cells for the eight authored source hulls into "
                "face-adjacent convex children with at most eight vertices"
            ),
            "rejection_gate": (
                "reject when total parts are not below the current A64 total "
                "reference 128; this is a project gate, not an NVIDIA limit"
            ),
            "direct_polygon_bridge": "reserve only; outside current scope",
            "live_capability_and_identity_must_remain_null": True,
        },
        "worker_contract": {
            "actual_workers": 1,
            "automatic_retries": 0,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "headless_rerun_viewer_maximum": 1,
        },
        "forbidden_runtime_counters": FORBIDDEN_COUNTERS,
        "registered_dirty_baseline": _status_paths(),
        "source_hashes": _source_hashes(),
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    _phase(
        "preregistration_frozen",
        preregistration_sha256=_sha(PREREG_PATH),
        checks_passed=sum(checks.values()),
        checks_total=len(checks),
        passed=prereg["pass"],
    )
    if not prereg["pass"]:
        raise RuntimeError(f"preregistration failed: {checks}")
    return 0


def worker() -> int:
    _phase("worker_start", pid=os.getpid())
    prereg = _read_json(PREREG_PATH)
    invocation = _read_json(INVOCATION_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("preregistration not passed")
    if invocation["preregistration_sha256"] != _sha(PREREG_PATH):
        raise RuntimeError("invocation/preregistration hash mismatch")
    if _input_hashes() != prereg["frozen_inputs"]["sha256"]:
        raise RuntimeError("input hash changed after preregistration")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("script changed after preregistration")
    if _status_paths() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("dirty baseline changed after preregistration")

    evidence, visual = _compute()
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase(
        "canonical_evidence_written",
        evidence_sha256=_sha(EVIDENCE_PATH),
        design_audit_pass=evidence["design_audit_pass"],
        repair_design_pass=evidence["repair_design_pass"],
        verdict=evidence["verdict"],
    )
    _write_csv(evidence)
    board = _render_board(evidence, visual)
    _phase("exact_board_written", board_sha256=board["sha256"])
    rerun = _write_rerun(evidence, visual)
    _phase(
        "rerun_finalized",
        rrd_sha256=rerun["rrd"]["sha256"],
        strict_validation_pass=rerun["strict_validation_pass"],
        viewer_returncode=rerun["headless_viewer_returncode"],
    )
    manual_template = {
        "artifact": "D384_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "evidence": {
            "path": _rel(EVIDENCE_PATH),
            "sha256": _sha(EVIDENCE_PATH),
        },
        "board": board,
        "rerun_screenshot": rerun["screenshot"],
        "required_check_keys": [
            "board_exact_1920x1080_and_legible",
            "board_no_text_clipping_or_overlap",
            "profile_parent_and_children_visible",
            "source_hull_witness_visible",
            "candidate_counts_match_canonical_json",
            "direct_capability_null_is_visible",
            "fallback_558_rejection_is_visible",
            "rerun_no_unknown_timeline",
            "rerun_no_decision_obscuring_overlap",
        ],
        "inspection_checks": {},
        "observations": [],
        "inspector_result": None,
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)
    post_checks = {
        "canonical_design_audit_pass": (
            evidence["design_audit_pass"] is True
        ),
        "no_admissible_low_count_candidate": (
            evidence["admissible_low_count_candidate_found"] is False
        ),
        "repair_design_fail_stop_preserved": (
            evidence["repair_design_pass"] is False
        ),
        "materialization_false": evidence["repair_materialized"] is False,
        "live_identity_null": evidence["live_identity_pass"] is None,
        "p34_identity_false": (
            evidence["p34_authored_to_cooked_identity_pass"] is False
        ),
        "g0a_false": evidence["g0a_pass"] is False,
        "board_exact_1920x1080": board["exact_1920x1080"],
        "rerun_strict_validation_pass": rerun["strict_validation_pass"],
        "rerun_headless_viewer_at_most_one": (
            rerun["headless_viewer_invocations"] <= 1
        ),
        "rerun_headless_return_zero": (
            rerun["headless_viewer_returncode"] == 0
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["frozen_inputs"]["sha256"]
        ),
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
    }
    claim = {
        "artifact": "D384_OFFLINE_WORKER_CLAIM_V1",
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
        "metrics_csv": {
            "path": _rel(METRICS_CSV),
            "sha256": _sha(METRICS_CSV),
            "bytes": METRICS_CSV.stat().st_size,
        },
        "board": board,
        "rerun": rerun,
        "manual_template": {
            "path": _rel(MANUAL_TEMPLATE),
            "sha256": _sha(MANUAL_TEMPLATE),
            "bytes": MANUAL_TEMPLATE.stat().st_size,
        },
        "checks": post_checks,
        "pass": all(post_checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_claim_written", worker_claim_sha256=_sha(WORKER_CLAIM))
    if not claim["pass"]:
        raise RuntimeError(f"worker post-check failed: {post_checks}")
    return 0


def run_supervisor() -> int:
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("preregistration not passed")
    if _input_hashes() != prereg["frozen_inputs"]["sha256"]:
        raise RuntimeError("input hash changed")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("script hash changed")
    if _status_paths() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("dirty baseline changed")
    command = [
        sys.executable,
        "-B",
        str(SCRIPT_PATH),
        "--stage",
        "worker",
    ]
    invocation = {
        "artifact": "D384_OFFLINE_DESIGN_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "cwd": str(REPO),
        "preregistration_sha256": _sha(PREREG_PATH),
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
    required = {
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
        and all(required.values())
        and claim.get("pass") is True
        and _input_hashes() == prereg["frozen_inputs"]["sha256"]
        and _source_hashes() == prereg["source_hashes"]
        and _status_paths() == prereg["registered_dirty_baseline"]
    )
    supervisor = {
        "artifact": "D384_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "worker_pid": process.pid,
        "actual_offline_worker_invocations": 1,
        "automatic_retries": 0,
        "watchdog_seconds": WATCHDOG_SECONDS,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "process_group_alive_after_wait": group_alive,
        "required_artifacts": required,
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
        raise RuntimeError(f"supervisor failed: {supervisor}")
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
        raise RuntimeError(f"finalize missing artifacts: {missing}")
    prereg = _read_json(PREREG_PATH)
    invocation = _read_json(INVOCATION_PATH)
    claim = _read_json(WORKER_CLAIM)
    supervisor = _read_json(SUPERVISOR_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    manual = _read_json(MANUAL_INSPECTION)
    manual_checks = manual.get("inspection_checks") or {}
    required_manual = set(
        _read_json(MANUAL_TEMPLATE)["required_check_keys"]
    )
    checks = {
        "preregistration_pass": prereg.get("pass") is True,
        "invocation_bound_to_preregistration": (
            invocation.get("preregistration_sha256") == _sha(PREREG_PATH)
        ),
        "worker_claim_pass": claim.get("pass") is True,
        "supervisor_pass": supervisor.get("pass") is True,
        "worker_one_retry_zero": (
            supervisor.get("actual_offline_worker_invocations") == 1
            and supervisor.get("automatic_retries") == 0
        ),
        "no_timeout_signal_or_residue": (
            supervisor.get("timed_out") is False
            and supervisor.get("sigterm_sent") is False
            and supervisor.get("sigkill_sent") is False
            and supervisor.get("process_group_alive_after_wait") is False
        ),
        "canonical_design_audit_pass": (
            evidence.get("design_audit_pass") is True
        ),
        "no_admissible_low_count_candidate": (
            evidence.get("admissible_low_count_candidate_found") is False
        ),
        "repair_design_fail_stop_preserved": (
            evidence.get("repair_design_pass") is False
        ),
        "repair_not_materialized": (
            evidence.get("repair_materialized") is False
        ),
        "live_identity_null": evidence.get("live_identity_pass") is None,
        "p34_identity_false": (
            evidence.get("p34_authored_to_cooked_identity_pass") is False
        ),
        "g0a_false": evidence.get("g0a_pass") is False,
        "manual_keys_exact": set(manual_checks) == required_manual,
        "manual_checks_all_true": all(
            manual_checks.get(name) is True for name in required_manual
        ),
        "manual_pass": manual.get("pass") is True,
        "manual_evidence_hash_exact": (
            manual.get("evidence", {}).get("sha256") == _sha(EVIDENCE_PATH)
        ),
        "manual_board_hash_exact": (
            manual.get("board", {}).get("sha256") == _sha(BOARD_PATH)
        ),
        "manual_rerun_hash_exact": (
            manual.get("rerun_screenshot", {}).get("sha256")
            == _sha(RERUN_SCREENSHOT)
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["frozen_inputs"]["sha256"]
        ),
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
    }
    completion = {
        "artifact": "D384_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": checks,
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "design_verdict": evidence["verdict"],
        "operational_verdict": (
            "D384_OFFLINE_WORKER_AND_OBSERVABILITY_PASS"
            if all(checks.values())
            else "D384_COMPLETION_INTEGRITY_FAIL_STOP"
        ),
        "registered_partition": evidence["repair_candidates"][
            "registered_recursive_partition"
        ],
        "rejected_exact_upper_bound": evidence["repair_candidates"][
            "exact_tetra_upper_bound"
        ],
        "direct_polygon_bridge_reserve": evidence["repair_candidates"][
            "direct_polygon_bridge_reserve"
        ],
        "remaining_truth": {
            "repair_materialized": False,
            "live_identity_pass": None,
            "p34_identity_pass": False,
            "cylinder_29x50_rendered_or_measured": False,
            "g0a_pass": False,
        },
        "artifacts": {
            "preregistration": {
                "path": _rel(PREREG_PATH),
                "sha256": _sha(PREREG_PATH),
            },
            "evidence": {
                "path": _rel(EVIDENCE_PATH),
                "sha256": _sha(EVIDENCE_PATH),
            },
            "metrics_csv": {
                "path": _rel(METRICS_CSV),
                "sha256": _sha(METRICS_CSV),
            },
            "board": _png_info(BOARD_PATH),
            "rrd": {
                "path": _rel(RRD_PATH),
                "sha256": _sha(RRD_PATH),
                "bytes": RRD_PATH.stat().st_size,
            },
            "rbl": {
                "path": _rel(RBL_PATH),
                "sha256": _sha(RBL_PATH),
                "bytes": RBL_PATH.stat().st_size,
            },
            "rerun_screenshot": _png_info(RERUN_SCREENSHOT),
            "manual_inspection": {
                "path": _rel(MANUAL_INSPECTION),
                "sha256": _sha(MANUAL_INSPECTION),
            },
            "supervisor": {
                "path": _rel(SUPERVISOR_PATH),
                "sha256": _sha(SUPERVISOR_PATH),
            },
        },
        "next_authorization_boundary": evidence[
            "next_authorization_boundary"
        ],
        "pass": all(checks.values()),
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase(
        "completion_written",
        completion_sha256=_sha(COMPLETION_PATH),
        passed=completion["pass"],
    )
    if not completion["pass"]:
        raise RuntimeError(f"completion failed: {checks}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "run", "worker", "finalize"),
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        return prepare()
    if args.stage == "run":
        return run_supervisor()
    if args.stage == "worker":
        return worker()
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
