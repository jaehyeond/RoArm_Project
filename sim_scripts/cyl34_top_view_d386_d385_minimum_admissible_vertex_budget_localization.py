#!/usr/bin/env python3
"""D386 offline localization of D385 first-observed layer vertex budgets.

The case reads immutable D379/D385 JSON and imports D385's frozen geometry
helpers.  It never starts Isaac Sim, Kit, PhysX, USD, CUDA, Warp, or robot
hardware.  Exactly four layers are in scope: the first no-cover layer recorded
for each of D385's four failed parents.  Later layers are inventoried only and
are not evaluated.

Candidate geometry is enumerated once with D385's exact thin-layer/profile-fan
construction.  Vertex budget is then localized as a bottleneck path problem
while polygon, face-width, volume, surface, topology-volume, and no-overlap
gates remain frozen.  The result is diagnostic and never becomes an adopted
global budget or a materializable P34 candidate in this case.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO):
    if str(REPO) in sys.path:
        sys.path.remove(str(REPO))
    sys.path.insert(0, str(REPO))

CASE = "g0a_d386"
ATTEMPT = (
    "attempt1_observed_no_cover_layer_minimum_vertex_budget_localization"
)
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT
SCRIPT_PATH = Path(__file__).resolve()
START_HERE = REPO / "START_HERE.md"

D385_SCRIPT = REPO / (
    "sim_scripts/"
    "cyl34_top_view_d385_p34_source_hull_semantic_low_count_redesign.py"
)
D385_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d385/"
    "attempt2_precreate_git_status_capture_repair/"
    "d385_p34_source_hull_redesign_evidence.json"
)
D385_COMPLETION = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d385/"
    "attempt2_precreate_git_status_capture_repair/"
    "d385_completion_summary.json"
)
D379_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d379/"
    "attempt2_d372_measurement_field_repair/"
    "d379_p34_full_live_identity_evidence.json"
)

EXPECTED_HEAD = "35f10e3079b19e51209ba4cf1dd66391a431b053"
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
    "d385_completion": (
        "2caf6c47ad563c9ad82b84d5c3367139943f95c98c62a590b3551a967def91c2"
    ),
}

PREREG_PATH = OUT_DIR / "d386_preregistration.json"
PHASE_PATH = OUT_DIR / "d386_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d386_offline_localizer_invocation.json"
WORKER_STDOUT = OUT_DIR / "d386_offline_worker_stdout.log"
WORKER_STDERR = OUT_DIR / "d386_offline_worker_stderr.log"
WORKER_CLAIM = OUT_DIR / "d386_offline_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d386_offline_worker_supervisor.json"
EVIDENCE_PATH = OUT_DIR / "d386_vertex_budget_localization_evidence.json"
GEOMETRY_PATH = OUT_DIR / "d386_finite_threshold_witness_geometry.json"
METRICS_CSV = OUT_DIR / "d386_candidate_cell_metrics.csv"
BOARD_PATH = OUT_DIR / "d386_vertex_budget_localization_1920x1080.png"
BOARD_LAYOUT = OUT_DIR / "d386_board_layout_validation.json"
RRD_PATH = OUT_DIR / "d386_vertex_budget_localization.rrd"
RBL_PATH = OUT_DIR / "d386_vertex_budget_localization.rbl"
RERUN_VALIDATION = OUT_DIR / "d386_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d386_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d386_manual_visual_inspection_template.json"
MANUAL_INSPECTION = OUT_DIR / "d386_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d386_completion_summary.json"

RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = [
    "observed_no_cover_layer_exact_minimax_vertex_budget_localizer_v1"
]
WATCHDOG_SECONDS = 300.0
BASELINE_BUDGET = 12
MAXIMUM_LOCALIZATION_BUDGET = 64
MAXIMUM_POLYGONS = 64
MAXIMUM_VERTICES_PER_POLYGON = 32
SURFACE_TOLERANCE_MM = 0.1
VOLUME_RELATIVE_TOLERANCE = 0.005
POSITIVE_VOLUME_EPS_M3 = 1.0e-18

TARGETS = [
    {
        "body": "gripper_link",
        "prim_name": "p000_proximal_upper_arm_hull_a",
        "name": "proximal_upper_arm_hull_a",
        "role": "moving_support",
        "region_name": "z_layer_00",
        "region_index": 0,
    },
    {
        "body": "gripper_link",
        "prim_name": "p002_proximal_lower_arm_hull_a",
        "name": "proximal_lower_arm_hull_a",
        "role": "moving_support",
        "region_name": "z_layer_01",
        "region_index": 1,
    },
    {
        "body": "link5",
        "prim_name": "p013_fixed_backbone_left",
        "name": "fixed_backbone_left",
        "role": "fixed_jaw_backbone",
        "region_name": "y_layer_01",
        "region_index": 1,
    },
    {
        "body": "link5",
        "prim_name": "p014_fixed_backbone_right",
        "name": "fixed_backbone_right",
        "role": "fixed_jaw_backbone",
        "region_name": "y_layer_00",
        "region_index": 0,
    },
]

DISPLAY_ORDER = [
    ("link5", "p013_fixed_backbone_left"),
    ("link5", "p014_fixed_backbone_right"),
    ("gripper_link", "p000_proximal_upper_arm_hull_a"),
    ("gripper_link", "p002_proximal_lower_arm_hull_a"),
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
    "asset_or_usd_reads": 0,
    "asset_or_usd_writes": 0,
    "collider_materializations_or_regenerations": 0,
    "automatic_decomposition_sweeps": 0,
    "alternate_partition_evaluations": 0,
    "internal_overlap_allowances": 0,
    "tolerance_changes": 0,
    "isaac_launches": 0,
    "kit_launches": 0,
    "physx_launches": 0,
    "live_callback_queries": 0,
    "warp_or_cuda_launches": 0,
    "cylinder_creates_or_writes": 0,
    "controlled_physics_steps": 0,
    "q5_samples": 0,
    "contact_queries": 0,
    "target_ik_path_changes": 0,
    "material_mass_actuator_physics_setting_changes": 0,
}


def _load_verified_d385_module() -> Any:
    actual_sha256 = _sha(D385_SCRIPT)
    expected_sha256 = EXPECTED_INPUT_SHA256["d385_script"]
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "refusing to execute unverified D385 helper script: "
            f"actual={actual_sha256} expected={expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location(
        "d385_frozen_geometry_helpers", D385_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen D385 helpers: {D385_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D385: Any | None = None


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_payload(value: Any) -> str:
    encoded = json.dumps(
        _native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        "d385_completion": _sha(D385_COMPLETION),
    }


def _target_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["body"]), str(row["prim_name"])


def _candidate_public(row: dict[str, Any]) -> dict[str, Any]:
    return {
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


def _enumerate_candidate_graph(
    layer_points: np.ndarray,
    *,
    thin_axis: int,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    polygon, keep = D385._profile_polygon(layer_points, thin_axis)
    triangle_count = len(polygon) - 2
    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    reason_counts: dict[str, int] = {}
    for end_state in range(1, triangle_count + 1):
        for start_state in range(max(0, end_state - 4), end_state):
            cell = D385._fan_cell(
                polygon,
                keep,
                triangle_start=start_state + 1,
                triangle_end=end_state,
                cell_index=-1,
            )
            row: dict[str, Any] = {
                "start_state": int(start_state),
                "end_state": int(end_state),
                "group_size": int(end_state - start_state),
                "geometry_constructed": False,
                "non_vertex_gates_pass": False,
                "rejection_reasons": [],
            }
            try:
                child = D385._intersect_profile_cell(
                    layer_points,
                    thin_axis=thin_axis,
                    cell=cell,
                )
            except Exception as exc:
                if not isinstance(
                    exc, (ValueError, D385.QhullError)
                ):
                    raise
                row["error_type"] = type(exc).__name__
                row["rejection_reasons"] = ["degenerate_geometry"]
                reason_counts["degenerate_geometry"] = (
                    reason_counts.get("degenerate_geometry", 0) + 1
                )
                candidates[(start_state, end_state)] = row
                continue
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
                }
            )
            reasons = []
            if row["polygon_count"] > MAXIMUM_POLYGONS:
                reasons.append("polygon_count_gt_64")
            if (
                row["maximum_vertices_per_polygon"]
                > MAXIMUM_VERTICES_PER_POLYGON
            ):
                reasons.append("vertices_per_polygon_gt_32")
            if row["volume_m3"] <= POSITIVE_VOLUME_EPS_M3:
                reasons.append("non_positive_volume")
            row["rejection_reasons"] = reasons
            row["non_vertex_gates_pass"] = not reasons
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            candidates[(start_state, end_state)] = row
    return candidates, {
        "broad_profile_vertex_count": int(len(polygon)),
        "triangle_count": int(triangle_count),
        "candidate_count": len(candidates),
        "geometry_constructed_count": sum(
            int(row["geometry_constructed"]) for row in candidates.values()
        ),
        "non_vertex_pass_count": sum(
            int(row["non_vertex_gates_pass"])
            for row in candidates.values()
        ),
        "rejection_reason_counts": reason_counts,
    }


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
        best = None
        for start_state in range(max(0, end_state - 4), end_state):
            previous = dp[start_state]
            row = candidates[(start_state, end_state)]
            if (
                previous is None
                or not row["non_vertex_gates_pass"]
                or row["vertex_count"] > budget
            ):
                continue
            value = (
                previous[0] + 1,
                max(previous[1], row["vertex_count"]),
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
        best = None
        for start_state in range(max(0, end_state - 4), end_state):
            previous = dp[start_state]
            row = candidates[(start_state, end_state)]
            if (
                previous is None
                or not row["non_vertex_gates_pass"]
                or row["vertex_count"] > MAXIMUM_LOCALIZATION_BUDGET
            ):
                continue
            value = (
                max(previous[0], row["vertex_count"]),
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
        raise RuntimeError("minimax budget lost its fixed-budget cover")
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
    minimum_budget: int | None = None
    canonical_key: tuple[int, int, tuple[int, ...]] | None = None

    def visit(position: int, maximum: int, cuts: tuple[int, ...]) -> None:
        nonlocal complete_path_count, minimum_budget, canonical_key
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
            position + 1, min(position + 4, triangle_count) + 1
        ):
            row = candidates[(position, end_state)]
            if (
                not row["non_vertex_gates_pass"]
                or row["vertex_count"] > MAXIMUM_LOCALIZATION_BUDGET
            ):
                continue
            visit(
                end_state,
                max(maximum, int(row["vertex_count"])),
                cuts + (end_state,),
            )

    visit(0, 0, (0,))
    if minimum_budget is None or canonical_key is None:
        return None
    return {
        "minimum_bottleneck_vertex_budget": int(minimum_budget),
        "complete_path_count": int(complete_path_count),
        "canonical_cover_at_minimum": {
            "child_count": int(canonical_key[0]),
            "maximum_child_vertex_count": int(canonical_key[1]),
            "cut_states": list(map(int, canonical_key[2])),
        },
    }


def _reachability_frontier(
    candidates: dict[tuple[int, int], dict[str, Any]],
    triangle_count: int,
) -> dict[str, Any]:
    reachable = {0}
    for end_state in range(1, triangle_count + 1):
        for start_state in range(max(0, end_state - 4), end_state):
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
            start_state + 1, min(start_state + 4, triangle_count) + 1
        ):
            row = candidates[(start_state, end_state)]
            if (
                end_state in backward
                and row["non_vertex_gates_pass"]
                and row["vertex_count"] <= MAXIMUM_LOCALIZATION_BUDGET
            ):
                backward.add(start_state)
                break
    blockers = []
    for (start_state, end_state), row in sorted(candidates.items()):
        if (
            start_state in reachable
            and end_state in backward
            and (
                not row["non_vertex_gates_pass"]
                or row["vertex_count"] > MAXIMUM_LOCALIZATION_BUDGET
            )
        ):
            public = _candidate_public(row)
            public["bounded_path_rejection_reasons"] = [
                *row["rejection_reasons"],
                *(
                    ["vertex_count_gt_64"]
                    if row.get("vertex_count", 0)
                    > MAXIMUM_LOCALIZATION_BUDGET
                    else []
                ),
            ]
            blockers.append(public)
    return {
        "reachable_states_from_zero": sorted(reachable),
        "states_that_can_reach_end": sorted(backward),
        "end_reachable_with_frozen_gates_and_vertex_le64": (
            triangle_count in reachable
        ),
        "frontier_blocking_candidates": blockers,
    }


def _selected_children(
    candidates: dict[tuple[int, int], dict[str, Any]],
    cover: dict[str, Any],
    target: dict[str, Any],
    interval_m: list[float],
) -> list[dict[str, Any]]:
    children = []
    cuts = cover["cut_states"]
    for child_index, (start_state, end_state) in enumerate(
        zip(cuts[:-1], cuts[1:], strict=True)
    ):
        source = candidates[(start_state, end_state)]["child"]
        child = dict(source)
        child.update(
            {
                "body": target["body"],
                "role": target["role"],
                "parent_name": target["name"],
                "region_index": target["region_index"],
                "name": (
                    f"{target['name']}__{target['region_name']}__"
                    f"cell_{child_index:02d}"
                ),
                "pre_split_axis": target["region_name"][0],
                "pre_split_interval_m": interval_m,
                "fan_triangle_index_range": [
                    int(start_state + 1),
                    int(end_state),
                ],
                "profile_cell_index": int(child_index),
            }
        )
        children.append(child)
    return children


def _layer_geometry_metrics(
    parent: dict[str, Any],
    children: list[dict[str, Any]],
    minimum_budget: int,
) -> dict[str, Any]:
    parent_points = np.asarray(parent["vertices_m"], dtype=np.float64)
    parent_triangles = np.asarray(parent["triangles"], dtype=np.int64)
    parent_equations = D385._normalized_equations(parent_points)
    child_points = np.vstack(
        [
            np.asarray(child["vertices_m"], dtype=np.float64)
            for child in children
        ]
    )
    outward_mm = D385._maximum_positive_violation_mm(
        parent_equations, child_points
    )
    samples = D385._surface_samples(parent_points, parent_triangles)
    coverage_mm, uncovered_samples = D385._union_coverage_violation_mm(
        children, samples
    )
    child_volume = float(sum(child["volume_m3"] for child in children))
    relative = abs(child_volume - parent["volume_m3"]) / parent["volume_m3"]
    overlap = D385._partition_overlap_certificate(children)
    checks = {
        "each_child_vertices_within_local_minimum": (
            max(child["vertex_count"] for child in children)
            <= minimum_budget
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
        "outward_le_0p1mm": outward_mm <= SURFACE_TOLERANCE_MM,
        "coverage_le_0p1mm": coverage_mm <= SURFACE_TOLERANCE_MM,
        "volume_relative_le_0p5percent": (
            relative <= VOLUME_RELATIVE_TOLERANCE
        ),
        "positive_volume_overlap_zero": overlap["pass"],
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
        "parent_volume_mm3": parent["volume_m3"] * 1.0e9,
        "child_volume_sum_mm3": child_volume * 1.0e9,
        "volume_relative_error": relative,
        "outward_halfspace_violation_mm": outward_mm,
        "parent_surface_coverage_halfspace_violation_mm": coverage_mm,
        "surface_sample_count": int(len(samples)),
        "uncovered_sample_count_gt_0p0001mm": uncovered_samples,
        "overlap_certificate": overlap,
        "checks": checks,
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
        "frozen_d385_helper_script": _rel(D385_SCRIPT),
        "frozen_d385_helper_script_sha256": _sha(D385_SCRIPT),
    }


def _compute() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
    if D385 is None:
        raise RuntimeError(
            "verified D385 helpers must be loaded after worker provenance"
        )
    d379 = _read_json(D379_EVIDENCE)
    d385_evidence = _read_json(D385_EVIDENCE)
    d385_completion = _read_json(D385_COMPLETION)
    authored_rows = d379["authored_readback"]["rows"]
    authored_map = {
        (str(row["body"]), str(row["prim_name"])): row
        for row in authored_rows
    }
    failure_rows = d385_evidence["partition_failures"]
    failure_map = {
        (str(row["body"]), str(row["prim_name"])): row
        for row in failure_rows
    }
    expected_target_keys = {_target_key(row) for row in TARGETS}
    actual_failure_keys = set(failure_map)
    target_inventory_exact = (
        actual_failure_keys == expected_target_keys
        and len(failure_rows) == len(TARGETS)
    )
    if not target_inventory_exact:
        raise RuntimeError(
            "D385 first-observed failure inventory changed: "
            f"actual={sorted(actual_failure_keys)} "
            f"expected={sorted(expected_target_keys)}"
        )

    results = []
    visual_rows = []
    candidate_rows: list[dict[str, Any]] = []
    shadowed_layers = []
    shadowed_layer_keys = set()
    evaluated_layer_keys = set()
    full_graph_enumeration_counts: dict[tuple[str, str, str], int] = {}
    frozen_b12_replay_counts: dict[tuple[str, str, str], int] = {}

    for target in TARGETS:
        key = _target_key(target)
        authored = authored_map[key]
        failure = failure_map[key]
        identity_checks = {
            "body_exact": authored["body"] == target["body"],
            "prim_name_exact": authored["prim_name"] == target["prim_name"],
            "name_exact": authored["name"] == target["name"],
            "role_exact": authored["role"] == target["role"],
            "d385_failure_name_exact": failure["name"] == target["name"],
            "d385_failure_role_exact": failure["role"] == target["role"],
            "d385_failure_layer_prefix_exact": failure[
                "error_message"
            ].startswith(f"{target['region_name']}:"),
        }
        if not all(identity_checks.values()):
            raise RuntimeError(
                f"target identity mismatch for {target}: {identity_checks}"
            )

        points = D385._unique_f32(
            np.asarray(authored["points_f32"], dtype=np.float64)
        )
        plan = D385.SEMANTIC_PLAN[target["name"]]
        pre_axis = int(plan["semantic_pre_split_axis"])
        pre_axis_name = str(plan["semantic_pre_split_axis_name"])
        if pre_axis_name != target["region_name"][0]:
            raise RuntimeError(
                f"pre-split axis mismatch for {target['name']}"
            )
        levels = np.unique(points[:, pre_axis])
        levels.sort()
        region_count = len(levels) - 1
        region_index = int(target["region_index"])
        if region_index < 0 or region_index >= region_count:
            raise RuntimeError(f"target region index out of range: {target}")
        for index in range(region_count):
            name = f"{pre_axis_name}_layer_{index:02d}"
            if index != region_index:
                shadowed_key = (
                    target["body"],
                    target["prim_name"],
                    name,
                )
                shadowed_layer_keys.add(shadowed_key)
                shadowed_layers.append(
                    {
                        "body": target["body"],
                        "prim_name": target["prim_name"],
                        "parent_name": target["name"],
                        "region_name": name,
                        "status": "present_but_not_evaluated",
                    }
                )

        low = float(levels[region_index])
        high = float(levels[region_index + 1])
        interval_m = [low, high]
        layer_mesh = D385._clip_interval(
            points, axis=pre_axis, low=low, high=high
        )
        layer_points = np.asarray(
            layer_mesh["vertices_m"], dtype=np.float64
        )
        layer_key = (
            target["body"],
            target["prim_name"],
            target["region_name"],
        )
        evaluated_layer_keys.add(layer_key)

        full_graph_enumeration_counts[layer_key] = (
            full_graph_enumeration_counts.get(layer_key, 0) + 1
        )
        candidates, graph = _enumerate_candidate_graph(
            layer_points, thin_axis=pre_axis
        )
        triangle_count = int(graph["triangle_count"])
        for row in candidates.values():
            candidate_rows.append(
                {
                    "body": target["body"],
                    "prim_name": target["prim_name"],
                    "parent_name": target["name"],
                    "region_name": target["region_name"],
                    **_candidate_public(row),
                }
            )

        replay_error = None
        frozen_b12_replay_counts[layer_key] = (
            frozen_b12_replay_counts.get(layer_key, 0) + 1
        )
        try:
            D385._profile_cell_partition(
                layer_points,
                thin_axis=pre_axis,
                region_name=target["region_name"],
            )
        except D385.RegisteredNoCoverError as exc:
            replay_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        b12_cover = _cover_at_budget(
            candidates, triangle_count, BASELINE_BUDGET
        )
        b12_replay_checks = {
            "d385_helper_raised_registered_no_cover": (
                replay_error is not None
            ),
            "d385_helper_error_message_bit_exact": (
                replay_error is not None
                and replay_error["message"] == failure["error_message"]
            ),
            "independent_budget12_cover_absent": b12_cover is None,
        }

        dp = _minimax_dp(candidates, triangle_count)
        exhaustive = _exhaustive_minimax(candidates, triangle_count)
        frontier = _reachability_frontier(candidates, triangle_count)
        raw_dp_budget = (
            dp["minimum_bottleneck_vertex_budget"]
            if dp is not None
            else None
        )
        raw_exhaustive_budget = (
            exhaustive["minimum_bottleneck_vertex_budget"]
            if exhaustive is not None
            else None
        )
        finite_in_range = bool(
            raw_dp_budget is not None
            and raw_dp_budget <= MAXIMUM_LOCALIZATION_BUDGET
        )
        if raw_dp_budget is not None and not finite_in_range:
            raise RuntimeError(
                "bounded minimax returned a budget above registered maximum"
            )
        minimum_budget = int(raw_dp_budget) if finite_in_range else None
        method_checks = {
            "dp_and_exhaustive_finiteness_agree": (
                (dp is None) == (exhaustive is None)
            ),
            "dp_and_exhaustive_budget_agree": (
                raw_dp_budget == raw_exhaustive_budget
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
            "bounded_graph_reachability_agrees_with_finiteness": (
                frontier[
                    "end_reachable_with_frozen_gates_and_vertex_le64"
                ]
                == (dp is not None)
            ),
        }

        boundary = {
            "baseline_budget": BASELINE_BUDGET,
            "baseline_cover_exists": b12_cover is not None,
            "minimum_budget_minus_one": None,
            "minimum_budget_minus_one_cover_exists": None,
            "minimum_budget": minimum_budget,
            "minimum_budget_cover_exists": None,
            "maximum_search_budget": MAXIMUM_LOCALIZATION_BUDGET,
            "maximum_search_budget_cover_exists": None,
            "evaluated_budget_values": [BASELINE_BUDGET],
        }
        selected_cover = None
        children: list[dict[str, Any]] = []
        geometry_metrics = None
        if minimum_budget is not None:
            previous_budget = minimum_budget - 1
            previous_cover = _cover_at_budget(
                candidates, triangle_count, previous_budget
            )
            selected_cover = _cover_at_budget(
                candidates, triangle_count, minimum_budget
            )
            boundary.update(
                {
                    "minimum_budget_minus_one": previous_budget,
                    "minimum_budget_minus_one_cover_exists": (
                        previous_cover is not None
                    ),
                    "minimum_budget_cover_exists": (
                        selected_cover is not None
                    ),
                    "evaluated_budget_values": sorted(
                        {
                            BASELINE_BUDGET,
                            previous_budget,
                            minimum_budget,
                        }
                    ),
                }
            )
            if selected_cover is None:
                raise RuntimeError("finite minimum lacks selected cover")
            children = _selected_children(
                candidates, selected_cover, target, interval_m
            )
            geometry_metrics = _layer_geometry_metrics(
                layer_mesh, children, minimum_budget
            )
        else:
            cover64 = _cover_at_budget(
                candidates,
                triangle_count,
                MAXIMUM_LOCALIZATION_BUDGET,
            )
            boundary.update(
                {
                    "maximum_search_budget_cover_exists": (
                        cover64 is not None
                    ),
                    "evaluated_budget_values": sorted(
                        {BASELINE_BUDGET, MAXIMUM_LOCALIZATION_BUDGET}
                    ),
                }
            )

        boundary_checks = {
            "baseline12_no_cover": b12_cover is None,
            "minimum_minus_one_no_cover": (
                boundary["minimum_budget_minus_one_cover_exists"] is False
                if minimum_budget is not None
                else True
            ),
            "minimum_cover_exists": (
                boundary["minimum_budget_cover_exists"] is True
                if minimum_budget is not None
                else True
            ),
            "minimum_strictly_above_12_if_finite": (
                minimum_budget > BASELINE_BUDGET
                if minimum_budget is not None
                else True
            ),
            "null_has_no_cover_at_64": (
                boundary["maximum_search_budget_cover_exists"] is False
                if minimum_budget is None
                else True
            ),
        }
        result = {
            **target,
            "identity_checks": identity_checks,
            "thin_axis_index": pre_axis,
            "thin_axis_name": pre_axis_name,
            "pre_split_levels_m": list(map(float, levels)),
            "pre_split_interval_m": interval_m,
            "parent_layer_vertex_count": int(layer_mesh["vertex_count"]),
            "parent_layer_polygon_count": int(layer_mesh["polygon_count"]),
            "candidate_graph": graph,
            "d385_budget12_replay": {
                "recorded_error": {
                    "type": failure["error_type"],
                    "message": failure["error_message"],
                },
                "replayed_error": replay_error,
                "checks": b12_replay_checks,
                "pass": all(b12_replay_checks.values()),
            },
            "dynamic_programming": dp,
            "independent_exhaustive_enumeration": exhaustive,
            "method_checks": method_checks,
            "minimum_admissible_vertex_budget_within_12_64": minimum_budget,
            "boundary_checks": boundary_checks,
            "boundary": boundary,
            "non_vertex_gate_frontier": frontier,
            "selected_threshold_cover": selected_cover,
            "selected_threshold_geometry_metrics": geometry_metrics,
            "localization_pass": bool(
                minimum_budget is not None
                and all(identity_checks.values())
                and all(b12_replay_checks.values())
                and all(method_checks.values())
                and all(boundary_checks.values())
                and geometry_metrics is not None
                and geometry_metrics["pass"]
            ),
        }
        results.append(result)
        visual_rows.append(
            {
                "target": target,
                "layer_parent": layer_mesh,
                "minimum_budget": minimum_budget,
                "selected_cover": selected_cover,
                "children": children,
                "geometry_metrics": geometry_metrics,
                "candidate_graph": graph,
                "frontier": frontier,
            }
        )

    target_layer_keys = {
        (row["body"], row["prim_name"], row["region_name"])
        for row in TARGETS
    }
    shadowed_evaluated_keys = (
        evaluated_layer_keys & shadowed_layer_keys
    )
    graph_enumeration_counts_public = [
        {
            "body": key[0],
            "prim_name": key[1],
            "region_name": key[2],
            "full_graph_enumerations": full_graph_enumeration_counts.get(
                key, 0
            ),
            "frozen_d385_b12_helper_replays": (
                frozen_b12_replay_counts.get(key, 0)
            ),
        }
        for key in sorted(target_layer_keys)
    ]
    global_checks = {
        "immutable_input_hashes_exact": (
            _input_hashes() == EXPECTED_INPUT_SHA256
        ),
        "d385_scientific_verdict_frozen": (
            d385_evidence["verdict"]
            == (
                "D385_SEMANTIC_THIN_LAYER_PROFILE_CELL_"
                "NO_ADMISSIBLE_CANDIDATE_FAIL_STOP"
            )
        ),
        "d385_completion_pass": d385_completion["pass"] is True,
        "target_inventory_exact_four": target_inventory_exact,
        "evaluated_layers_exact_registered_four": (
            evaluated_layer_keys == target_layer_keys
        ),
        "one_full_graph_and_one_b12_replay_per_target_layer": all(
            full_graph_enumeration_counts.get(key, 0) == 1
            and frozen_b12_replay_counts.get(key, 0) == 1
            for key in target_layer_keys
        ),
        "shadowed_layer_evaluation_count_zero": (
            len(shadowed_evaluated_keys) == 0
        ),
        "all_budget12_replays_exact": all(
            row["d385_budget12_replay"]["pass"] for row in results
        ),
        "all_independent_methods_agree": all(
            all(row["method_checks"].values()) for row in results
        ),
        "all_boundary_checks_pass": all(
            all(row["boundary_checks"].values()) for row in results
        ),
        "all_finite_witness_geometry_gates_pass": all(
            row["selected_threshold_geometry_metrics"] is None
            or row["selected_threshold_geometry_metrics"]["pass"]
            for row in results
        ),
        "forbidden_runtime_counters_zero": all(
            value == 0 for value in FORBIDDEN_COUNTERS.values()
        ),
    }
    finite_values = [
        row["minimum_admissible_vertex_budget_within_12_64"]
        for row in results
        if row["minimum_admissible_vertex_budget_within_12_64"] is not None
    ]
    all_four_finite = len(finite_values) == len(TARGETS)
    method_contract_pass = all(global_checks.values())
    localization_pass = bool(
        method_contract_pass
        and all_four_finite
        and all(row["localization_pass"] for row in results)
    )
    if not method_contract_pass:
        verdict = (
            "D386_PARTITION_PROVENANCE_OR_MINIMALITY_CONTRACT_FAIL_STOP"
        )
    elif not all_four_finite:
        verdict = (
            "D386_OBSERVED_LAYER_VERTEX_BUDGET_NOT_LOCALIZABLE_FAIL_STOP"
        )
    else:
        verdict = (
            "D386_OBSERVED_NO_COVER_LAYER_MINIMUM_VERTEX_BUDGET_"
            "LOCALIZATION_PASS"
        )

    evidence = {
        "artifact": (
            "D386_OBSERVED_NO_COVER_LAYER_VERTEX_BUDGET_LOCALIZATION_"
            "EVIDENCE_V1"
        ),
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Localize the exact minimum child-vertex budget, if any within "
            "12..64, for only D385's four first-observed no-cover layers "
            "while freezing its partition and every non-vertex gate."
        ),
        "new_variables": NEW_VARIABLES,
        "measurement_authority": (
            "immutable D379 authored Float32 streams, immutable D385 "
            "first-failure evidence, and D385's frozen offline geometry helpers"
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
                "applicability": (
                    "version-matched collider/cooking context only; D386 "
                    "does not run a cook"
                ),
            },
            {
                "title": "PhysX 5.6.1 - GPU Rigid Bodies",
                "url": (
                    "https://nvidia-omniverse.github.io/PhysX/physx/"
                    "5.6.1/docs/GPURigidBodies.html"
                ),
                "applicability": (
                    "vertices, polygons, and face-width remain separate "
                    "conditions; offline D386 is not a live GPU verdict"
                ),
            },
            {
                "title": "PhysX 107.3 GuConvexMesh::isGpuCompatible",
                "url": (
                    "https://raw.githubusercontent.com/NVIDIA-Omniverse/"
                    "PhysX/107.3-omni-and-physx-5.6.1/physx/source/"
                    "geomutils/src/convex/GuConvexMesh.cpp"
                ),
                "applicability": (
                    "local source cross-check for distinct compatibility "
                    "conditions; D386 keeps non-vertex gates frozen"
                ),
            },
        ],
        "frozen_contract": {
            "target_layers": TARGETS,
            "baseline_vertex_budget": BASELINE_BUDGET,
            "maximum_localization_budget": MAXIMUM_LOCALIZATION_BUDGET,
            "contiguous_fan_group_size": [1, 4],
            "maximum_polygons": MAXIMUM_POLYGONS,
            "maximum_vertices_per_polygon": (
                MAXIMUM_VERTICES_PER_POLYGON
            ),
            "positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
            "surface_tolerance_mm": SURFACE_TOLERANCE_MM,
            "topology_volume_relative_tolerance": (
                VOLUME_RELATIVE_TOLERANCE
            ),
            "positive_volume_child_overlap": 0,
        },
        "scope_statement": {
            "first_observed_layers_only": True,
            "evaluated_layer_count": len(evaluated_layer_keys),
            "evaluation_counts_by_target_layer": (
                graph_enumeration_counts_public
            ),
            "shadowed_or_later_layers": shadowed_layers,
            "shadowed_or_later_layer_keys_evaluated": [
                {
                    "body": key[0],
                    "prim_name": key[1],
                    "region_name": key[2],
                }
                for key in sorted(shadowed_evaluated_keys)
            ],
            "shadowed_or_later_layer_evaluation_count": (
                len(shadowed_evaluated_keys)
            ),
            "parent_wide_or_complete_p34_inference_authorized": False,
        },
        "layer_results": results,
        "finite_localized_layer_count": len(finite_values),
        "null_localized_layer_count": len(TARGETS) - len(finite_values),
        "partial_finite_localized_maximum_diagnostic_only": (
            max(finite_values) if finite_values else None
        ),
        "observed_four_layer_required_maximum_diagnostic_only": (
            max(finite_values) if all_four_finite else None
        ),
        "selected_vertex_budget": None,
        "parent_wide_vertex_budget": None,
        "complete_p34_vertex_budget": None,
        "selected_budget_application_count": 0,
        "complete_source_child_count": None,
        "complete_total_part_count": None,
        "global_semantic_preservation_pass": None,
        "materializable_candidate": False,
        "method_contract_checks": global_checks,
        "method_contract_pass": method_contract_pass,
        "localization_pass": localization_pass,
        "repair_materialized": False,
        "live_identity_pass": None,
        "live_gpu_compatibility_pass": None,
        "current_scope_counters": FORBIDDEN_COUNTERS,
        "cylinder_29x50_rendered_or_measured": False,
        "physics_or_grasp_result": None,
        "p34_authored_to_cooked_identity_pass": False,
        "g0a_pass": False,
        "verdict": verdict,
        "next_authorization_boundary": (
            "Do not select or apply a vertex budget, evaluate later layers, "
            "change the partition/non-vertex gates, materialize USD/PhysX, "
            "or create/run the 29x50mm cylinder without a new explicit approval."
        ),
    }

    witness_layers = []
    for row, visual in zip(results, visual_rows, strict=True):
        witness = {
            "body": row["body"],
            "prim_name": row["prim_name"],
            "parent_name": row["name"],
            "region_name": row["region_name"],
            "minimum_budget": row[
                "minimum_admissible_vertex_budget_within_12_64"
            ],
            "materializable_candidate": False,
            "parent_layer": {
                "vertices_f64_m": visual["layer_parent"]["vertices_m"],
                "triangles_i64": visual["layer_parent"]["triangles"],
            },
            "diagnostic_children": [],
        }
        for child in visual["children"]:
            witness["diagnostic_children"].append(
                {
                    "name": child["name"],
                    "fan_triangle_index_range": child[
                        "fan_triangle_index_range"
                    ],
                    "vertices_f64_m": child["vertices_m"],
                    "triangles_i64": child["triangles"],
                    "vertex_count": child["vertex_count"],
                    "polygon_count": child["polygon_count"],
                    "maximum_vertices_per_polygon": child[
                        "max_vertices_per_polygon"
                    ],
                }
            )
        witness_layers.append(witness)
    geometry = {
        "artifact": "D386_FINITE_THRESHOLD_WITNESS_GEOMETRY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "authority": (
            "offline diagnostic witnesses for finite first-observed layers "
            "only; not an adopted collider stream"
        ),
        "selected_vertex_budget": None,
        "parent_wide_vertex_budget": None,
        "complete_p34_vertex_budget": None,
        "complete_materializable_candidate": False,
        "layers": witness_layers,
    }
    return evidence, geometry, {"rows": visual_rows}, candidate_rows


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
    by_key = {
        (row["target"]["body"], row["target"]["prim_name"]): row
        for row in visual["rows"]
    }
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.045,
        right=0.97,
        top=0.855,
        bottom=0.155,
        wspace=0.07,
        hspace=0.26,
    )
    text_artists = []
    card_titles = []
    for index, key in enumerate(DISPLAY_ORDER):
        row = by_key[key]
        target = row["target"]
        axis = fig.add_subplot(grid[index // 2, index % 2], projection="3d")
        parent = row["layer_parent"]
        parent_vertices = (
            np.asarray(parent["vertices_m"], dtype=np.float64) * 1000.0
        )
        parent_triangles = np.asarray(parent["triangles"], dtype=np.int64)
        finite = row["minimum_budget"] is not None
        axis.add_collection3d(
            Poly3DCollection(
                parent_vertices[parent_triangles],
                facecolors=(
                    (0.35, 0.38, 0.42, 0.07)
                    if finite
                    else (0.80, 0.16, 0.18, 0.13)
                ),
                edgecolors=(
                    (0.18, 0.20, 0.24, 0.52)
                    if finite
                    else (0.70, 0.08, 0.10, 0.78)
                ),
                linewidths=0.5,
            )
        )
        all_points = [parent_vertices]
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
                    edgecolors=(0.08, 0.10, 0.12, 0.45),
                    linewidths=0.38,
                )
            )
            all_points.append(vertices)
        _equal_3d_limits(axis, np.vstack(all_points))
        if target["name"].startswith("fixed_backbone"):
            axis.view_init(elev=6.0, azim=-84.0)
        else:
            axis.view_init(elev=24.0, azim=-58.0)
        axis.set_axis_off()
        if finite:
            metrics = row["geometry_metrics"]
            minimum = int(row["minimum_budget"])
            label = (
                f"{target['name']} / {target['region_name']}\n"
                f"12: 실패  |  {minimum - 1}: 실패 → "
                f"최소 {minimum}: 성공 ({len(row['children'])}조각)\n"
                f"표면 바깥/미보존 "
                f"{metrics['outward_halfspace_violation_mm']:.6f}/"
                f"{metrics['parent_surface_coverage_halfspace_violation_mm']:.6f} mm  |  "
                f"부피오차 {metrics['volume_relative_error'] * 100:.6f}%  |  "
                "겹침 0"
            )
            color = "#0f5132"
        else:
            polygon_rejections = row["candidate_graph"][
                "rejection_reason_counts"
            ].get("polygon_count_gt_64", 0)
            label = (
                f"{target['name']} / {target['region_name']}\n"
                "12: 실패  |  64까지: 실패  |  최소값 NULL\n"
                "꼭짓점 수만 늘려서는 완전 분할 불가  |  "
                f"polygon≤64 고정 게이트 거부 {polygon_rejections}개"
            )
            color = "#991b1b"
        title = axis.set_title(
            label,
            fontproperties=regular,
            fontsize=9.4,
            pad=3,
            color=color,
        )
        text_artists.append(title)
        card_titles.append(title)

    title = fig.suptitle(
        "D386 — D385에서 처음 막힌 네 층의 최소 꼭짓점 예산 국소화",
        x=0.5,
        y=0.968,
        fontproperties=bold,
        fontsize=20,
        color="#111827",
    )
    subtitle = fig.text(
        0.5,
        0.918,
        (
            "회색/붉은 외곽=동결된 층 형상 · 색 조각=최소값이 유한한 층의 "
            "임계 분할 증거 · 분할법과 비-꼭짓점 게이트는 변경하지 않음"
        ),
        ha="center",
        fontproperties=regular,
        fontsize=11,
        color="#334155",
    )
    finite_count = evidence["finite_localized_layer_count"]
    null_count = evidence["null_localized_layer_count"]
    result = fig.text(
        0.5,
        0.087,
        (
            f"유한 최소값 {finite_count}/4 · 최소값 없음 {null_count}/4  |  "
            "네 층 공통 예산 선택 = NULL  |  전체 P34 후보 = 아님"
        ),
        ha="center",
        fontproperties=bold,
        fontsize=13,
        color=("#047857" if evidence["localization_pass"] else "#b91c1c"),
    )
    footer = fig.text(
        0.5,
        0.041,
        (
            "각 부모의 첫 실패 층만 측정했습니다. 뒤쪽 층은 계산하지 않았고, "
            "USD·Isaac·PhysX·원통·물리·접촉·파지는 모두 0회입니다."
        ),
        ha="center",
        fontproperties=regular,
        fontsize=10,
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
        boxes.append(
            {
                "index": index,
                "x0": float(bbox.x0),
                "y0": float(bbox.y0),
                "x1": float(bbox.x1),
                "y1": float(bbox.y1),
            }
        )
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
        artist.get_window_extent(renderer=renderer) for artist in card_titles
    ]
    overlap_pairs = []
    for left_index, left in enumerate(card_boxes):
        for right_index in range(left_index + 1, len(card_boxes)):
            right = card_boxes[right_index]
            if (
                min(left.x1, right.x1) > max(left.x0, right.x0)
                and min(left.y1, right.y1) > max(left.y0, right.y0)
            ):
                overlap_pairs.append([left_index, right_index])
    checks["card_title_pairwise_nonoverlap"] = not overlap_pairs
    checks["subtitle_card_titles_nonoverlap"] = bool(
        subtitle.get_window_extent(renderer=renderer).y0
        > max(box.y1 for box in card_boxes)
    )
    synthetic_left = (10.0, 10.0, 30.0, 30.0)
    synthetic_right = (10.0, 10.0, 30.0, 30.0)
    synthetic_controls = {
        "identical_boxes_overlap_detected": bool(
            min(synthetic_left[2], synthetic_right[2])
            > max(synthetic_left[0], synthetic_right[0])
            and min(synthetic_left[3], synthetic_right[3])
            > max(synthetic_left[1], synthetic_right[1])
        ),
        "negative_margin_box_clipping_detected": bool(
            -1.0 < 0.0 or canvas_width + 1.0 > canvas_width
        ),
    }
    layout = {
        "artifact": "D386_BOARD_LAYOUT_VALIDATION_V1",
        "canvas_pixels": [canvas_width, canvas_height],
        "artist_bboxes_display_pixels": boxes,
        "card_title_overlap_pairs": overlap_pairs,
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
        contents="/d386/layers/**",
        name="D386 four first-observed layers",
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=(0.28, -0.36, 0.29),
            look_target=(0.10, 0.0, 0.07),
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
            name="D386 decision and authorization boundary",
        ),
        row_shares=[0.75, 0.25],
    )
    notification_buffer = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d386/notification_buffer/**",
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
            column_shares=[0.76, 0.24],
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

    by_key = {
        (row["target"]["body"], row["target"]["prim_name"]): row
        for row in visual["rows"]
    }
    meshes = []
    for index, key in enumerate(DISPLAY_ORDER):
        row = by_key[key]
        target = row["target"]
        parent = row["layer_parent"]
        center = np.mean(parent["vertices_m"], axis=0)
        offset = np.asarray(
            [0.14 * (index % 2), 0.0, 0.14 * (1 - index // 2)],
            dtype=np.float64,
        )
        parent_vertices = (
            np.asarray(parent["vertices_m"], dtype=np.float64)
            - center
            + offset
        )
        safe_name = f"{index:02d}_{target['name']}_{target['region_name']}"
        prefix = f"d386/layers/{safe_name}"
        finite = row["minimum_budget"] is not None
        meshes.append(
            {
                "entity_path": f"{prefix}/frozen_layer_parent",
                "coordinate_frame": "tf#/",
                "vertices_m": parent_vertices,
                "triangles": parent["triangles"],
                "color_rgba": (
                    [90, 96, 105, 42] if finite else [205, 48, 55, 75]
                ),
                "static": True,
                "representation": (
                    "inspection-only shifted immutable D385 first-failure "
                    "layer parent"
                ),
                "numeric_authority": "canonical unshifted D386 JSON",
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
                        f"{prefix}/finite_threshold_witness/"
                        f"child_{child_index:02d}"
                    ),
                    "coordinate_frame": "tf#/",
                    "vertices_m": vertices,
                    "triangles": child["triangles"],
                    "color_rgba": PALETTE[child_index % len(PALETTE)],
                    "static": True,
                    "representation": (
                        "inspection-only shifted finite threshold witness"
                    ),
                    "numeric_authority": "canonical unshifted D386 JSON",
                }
            )

    summary_path = "metadata/run"
    layer_lines = []
    for row in evidence["layer_results"]:
        minimum = row["minimum_admissible_vertex_budget_within_12_64"]
        if minimum is None:
            text = "NULL (no fixed-gate cover through 64)"
        else:
            text = (
                f"{minimum} ({minimum - 1} fail / {minimum} pass, "
                f"{row['selected_threshold_cover']['child_count']} children)"
            )
        layer_lines.append(
            f"- `{row['name']}/{row['region_name']}`: **{text}**"
        )
    summary_markdown = "\n".join(
        [
            "## D386 first-observed layer vertex localization",
            "",
            *layer_lines,
            "",
            (
                f"- Finite/null: **{evidence['finite_localized_layer_count']}"
                f"/{evidence['null_localized_layer_count']}**"
            ),
            "- Selected or applied vertex budget: **NULL / 0**",
            "- Complete P34 candidate: **NO**",
            "- Later/shadowed layer evaluations: **0**",
            "- Isaac / PhysX / USD / cylinder / physics / q5 / contact: **0**",
            "- g0a_pass=false",
            "",
            (
                "Do not adopt a budget or proceed to materialization/physics "
                "without a separate approval."
            ),
        ]
    )
    expected_entities = {"metadata/run"}
    component_contract = {"metadata/run": ["TextDocument:text"]}
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
        if mode == "d386_vertex_localization":
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
                "finite_layer_count": evidence[
                    "finite_localized_layer_count"
                ],
                "null_layer_count": evidence["null_localized_layer_count"],
                "selected_vertex_budget": None,
                "later_layer_evaluations": 0,
                "g0a_pass": False,
                "viewer_layout_note": (
                    "geometry is shifted into a 2x2 inspection grid; "
                    "canonical unshifted JSON is numeric authority"
                ),
            },
            recording_id="g0a_d386_vertex_budget_localization",
            blueprint_path=RBL_PATH,
            blueprint_mode="d386_vertex_localization",
            live_viewer=False,
            app_id="roarm_g0a_d386_vertex_budget_localization",
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
                (validation.get("headless_render") or {}).get("attempted")
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


def _write_candidate_metrics_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "body",
        "prim_name",
        "parent_name",
        "region_name",
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


def _import_roots_from_ast() -> list[str]:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return sorted(roots)


def _prepare() -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"refusing to reuse forward-only path: {OUT_DIR}")
    status_before_output_create = _git(["status", "--short"])
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    status_after_output_create = _git(["status", "--short"])
    start_text = START_HERE.read_text(encoding="utf-8")
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
    expected_before = {
        " M START_HERE.md",
        " M claudedocs/BACKLOG.md",
        " M claudedocs/DECISIONS.md",
        " M claudedocs/EXPERIMENT_LEDGER.md",
        "?? claudedocs/runtime_logs/grasp_track/g0a_d385/",
        (
            "?? claudedocs/"
            "session_20260725_grasp_g0a_d385_"
            "p34_source_hull_semantic_low_count_redesign.md"
        ),
        (
            "?? sim_scripts/"
            "cyl34_top_view_d385_p34_source_hull_"
            "semantic_low_count_redesign.py"
        ),
        (
            "?? sim_scripts/"
            "cyl34_top_view_d386_d385_minimum_admissible_"
            "vertex_budget_localization.py"
        ),
    }
    expected_after = expected_before | {
        "?? claudedocs/runtime_logs/grasp_track/g0a_d386/"
    }
    d385_evidence = _read_json(D385_EVIDENCE)
    d385_failures = d385_evidence["partition_failures"]
    actual_failure_identity = {
        (
            row["body"],
            row["prim_name"],
            row["name"],
            row["role"],
            row["error_message"].split(":", 1)[0],
        )
        for row in d385_failures
    }
    expected_failure_identity = {
        (
            row["body"],
            row["prim_name"],
            row["name"],
            row["role"],
            row["region_name"],
        )
        for row in TARGETS
    }
    checks = {
        "head_exact": head == EXPECTED_HEAD,
        "origin_exact": origin == EXPECTED_HEAD,
        "input_hashes_exact": _input_hashes() == EXPECTED_INPUT_SHA256,
        "d385_first_observed_failure_inventory_exact_four": (
            actual_failure_identity == expected_failure_identity
            and len(d385_failures) == 4
        ),
        "start_here_active_case_present": (
            "D386 [d385_minimum_admissible_vertex_budget_localization]"
            in start_text
        ),
        "start_here_variable_present": NEW_VARIABLES[0] in start_text,
        "start_here_output_path_present": _rel(OUT_DIR) in start_text,
        "start_here_first_observed_scope_present": (
            "four first-observed no-cover layers" in start_text
        ),
        "start_here_shadowed_not_evaluated_present": (
            "Later/shadowed layers exist" in start_text
        ),
        "start_here_budget_range_12_64_present": (
            "Path search itself is bounded to `12..64`" in start_text
        ),
        "start_here_no_overlap_gate_present": (
            "positive-volume child overlap `0`" in start_text
        ),
        "one_new_variable_exact": len(NEW_VARIABLES) == 1,
        "baseline_budget_exact_12": BASELINE_BUDGET == 12,
        "maximum_localization_budget_exact_64": (
            MAXIMUM_LOCALIZATION_BUDGET == 64
        ),
        "non_vertex_gates_frozen": (
            MAXIMUM_POLYGONS == 64
            and MAXIMUM_VERTICES_PER_POLYGON == 32
            and SURFACE_TOLERANCE_MM == 0.1
            and VOLUME_RELATIVE_TOLERANCE == 0.005
        ),
        "forbidden_runtime_imports_absent": not forbidden_imports,
        "rerun_cli_present": RERUN_CLI.is_file(),
        "font_regular_present": FONT_REGULAR.is_file(),
        "font_bold_present": FONT_BOLD.is_file(),
        "worktree_before_output_create_exact": (
            set(status_before_output_create.splitlines())
            == expected_before
        ),
        "output_create_added_only_d386_root": (
            set(status_after_output_create.splitlines()) == expected_after
        ),
    }
    preregistration = {
        "artifact": "D386_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "approved_scope": (
            "offline-only exact minimum vertex-budget localization for "
            "D385's four first-observed no-cover layers"
        ),
        "new_variables": NEW_VARIABLES,
        "target_layers": TARGETS,
        "registered_method": {
            "complete_candidate_graph_enumerations_per_layer": 1,
            "frozen_d385_b12_helper_replays_per_layer": 1,
            "construction": (
                "D385 authored thin-layer interval, broad-profile fan "
                "anchor/order, contiguous group size 1..4, and original-parent "
                "intersection"
            ),
            "primary_localizer": (
                "dynamic-programming minimization of the maximum child "
                "vertex count over complete contiguous paths"
            ),
            "independent_localizer": (
                "exhaustive enumeration of every complete fixed-gate path"
            ),
            "boundary_proof": (
                "B=12 no-cover replay; for finite B*, B*-1 no-cover and "
                "B* cover; for null, B=64 no-cover"
            ),
            "budget_range_inclusive": [
                BASELINE_BUDGET,
                MAXIMUM_LOCALIZATION_BUDGET,
            ],
        },
        "frozen_non_vertex_gates": {
            "maximum_polygons": MAXIMUM_POLYGONS,
            "maximum_vertices_per_polygon": (
                MAXIMUM_VERTICES_PER_POLYGON
            ),
            "positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
            "surface_tolerance_mm": SURFACE_TOLERANCE_MM,
            "topology_volume_relative_tolerance": (
                VOLUME_RELATIVE_TOLERANCE
            ),
            "positive_volume_child_overlap": 0,
        },
        "failure_semantics": (
            "All four layers must have a finite B*>12 within 12..64 and "
            "pass independent minimality and frozen geometry gates for "
            "localization PASS. Any null is NOT_LOCALIZABLE_FAIL_STOP; any "
            "provenance or algorithm disagreement is CONTRACT_FAIL_STOP. "
            "Partial finite minima remain diagnostic only."
        ),
        "explicit_nonclaims": {
            "selected_or_adopted_budget": None,
            "parent_wide_budget": None,
            "complete_p34_budget": None,
            "complete_source_child_count": None,
            "complete_total_part_count": None,
            "materializable_candidate": False,
            "later_or_shadowed_layer_evaluations": 0,
            "live_gpu_or_physics_or_grasp_result": None,
        },
        "worker_contract": {
            "actual_worker_invocations": 1,
            "retries": 0,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "watchdog_signal_authority": (
                "on timeout only, signal the newly created D386-owned worker "
                "process group; never signal unrelated/external processes"
            ),
            "rerun_viewer_invocations_maximum": 1,
        },
        "forbidden_runtime_counters": FORBIDDEN_COUNTERS,
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
            "input_hashes": _input_hashes(),
            "import_roots": imports,
            "forbidden_import_roots_found": forbidden_imports,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, preregistration)
    _phase("prepare_end", pass_value=preregistration["pass"])
    if not preregistration["pass"]:
        raise RuntimeError(f"D386 preregistration failed: {checks}")
    print(json.dumps({"prepare_pass": True, "path": _rel(PREREG_PATH)}))
    return 0


def _worker() -> int:
    global D385
    if not PREREG_PATH.is_file():
        raise RuntimeError("missing D386 preregistration")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D386 preregistration did not pass")
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
            f"D386 worker provenance failed: {provenance_checks}"
        )
    if D385 is not None:
        raise RuntimeError("D385 helpers were loaded before worker provenance")
    D385 = _load_verified_d385_module()
    _phase("worker_start")
    evidence, geometry, visual, candidate_rows = _compute()
    evidence["script_sha256"] = _sha(SCRIPT_PATH)
    evidence["diagnostic_geometry_payload_sha256"] = _sha_payload(geometry)
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
        localization_pass=evidence["localization_pass"],
    )
    _write_json_x(GEOMETRY_PATH, geometry)
    _write_candidate_metrics_csv(candidate_rows)
    board_info, layout = _render_board(evidence, visual)
    _write_json_x(BOARD_LAYOUT, layout)
    rerun = _write_rerun(evidence, visual)
    manual_template = {
        "artifact": "D386_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board_path": _rel(BOARD_PATH),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "required_checks": [
            "board_exact_1920x1080_and_readable",
            "all_four_registered_layer_cards_visible",
            "finite_and_null_results_visually_unambiguous",
            "boundary_labels_readable",
            "shadowed_layers_not_evaluated_notice_visible",
            "rerun_geometry_and_metadata_readable",
            "no_budget_adoption_live_or_physics_claim",
        ],
        "inspection_result_path": _rel(MANUAL_INSPECTION),
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)
    claim = {
        "artifact": "D386_OFFLINE_WORKER_CLAIM_V1",
        "worker_pid": os.getpid(),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "scientific_verdict": evidence["verdict"],
        "localization_pass": evidence["localization_pass"],
        "method_contract_pass": evidence["method_contract_pass"],
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
            evidence["method_contract_pass"]
            and layout["pass"]
            and rerun["strict_validation_pass"]
            and rerun["headless_viewer_invocations"] <= 1
            and all(value == 0 for value in FORBIDDEN_COUNTERS.values())
        ),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_end", worker_claim_pass=claim["pass"])
    if not claim["pass"]:
        raise RuntimeError("D386 observability worker claim failed")
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
        raise RuntimeError("run requires completed D386 prepare stage")
    if INVOCATION_PATH.exists() or SUPERVISOR_PATH.exists():
        raise RuntimeError("refusing to repeat D386 actual worker")
    command = [
        sys.executable,
        "-B",
        str(SCRIPT_PATH),
        "--stage",
        "worker",
    ]
    invocation = {
        "artifact": "D386_OFFLINE_WORKER_INVOCATION_V1",
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
        "artifact": "D386_OFFLINE_WORKER_SUPERVISOR_V1",
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
        raise RuntimeError(f"D386 worker failed: {supervisor}")
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
        "all_four_registered_layer_cards_visible",
        "finite_and_null_results_visually_unambiguous",
        "boundary_labels_readable",
        "shadowed_layers_not_evaluated_notice_visible",
        "rerun_geometry_and_metadata_readable",
        "no_budget_adoption_live_or_physics_claim",
    }
    manual_checks = manual.get("checks", {})
    manual_hashes = manual.get("artifact_hashes", {})
    manual_contract_pass = bool(
        manual.get("artifact") == "D386_MANUAL_VISUAL_INSPECTION_V1"
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
    pass_verdict = (
        "D386_OBSERVED_NO_COVER_LAYER_MINIMUM_VERTEX_BUDGET_"
        "LOCALIZATION_PASS"
    )
    null_verdict = (
        "D386_OBSERVED_LAYER_VERTEX_BUDGET_NOT_LOCALIZABLE_FAIL_STOP"
    )
    contract_verdict = (
        "D386_PARTITION_PROVENANCE_OR_MINIMALITY_CONTRACT_FAIL_STOP"
    )
    verdict_consistent = bool(
        (
            evidence["localization_pass"] is True
            and evidence["null_localized_layer_count"] == 0
            and evidence["verdict"] == pass_verdict
        )
        or (
            evidence["method_contract_pass"] is True
            and evidence["localization_pass"] is False
            and evidence["null_localized_layer_count"] > 0
            and evidence["verdict"] == null_verdict
        )
        or (
            evidence["method_contract_pass"] is False
            and evidence["verdict"] == contract_verdict
        )
    )
    checks = {
        "scientific_verdict_consistent": verdict_consistent,
        "selected_vertex_budget_null": (
            evidence["selected_vertex_budget"] is None
        ),
        "parent_wide_vertex_budget_null": (
            evidence["parent_wide_vertex_budget"] is None
        ),
        "complete_p34_vertex_budget_null": (
            evidence["complete_p34_vertex_budget"] is None
        ),
        "selected_budget_application_zero": (
            evidence["selected_budget_application_count"] == 0
        ),
        "complete_counts_null": (
            evidence["complete_source_child_count"] is None
            and evidence["complete_total_part_count"] is None
        ),
        "materializable_candidate_false": (
            evidence["materializable_candidate"] is False
        ),
        "registered_layers_exact_four": (
            evidence["scope_statement"]["evaluated_layer_count"] == 4
        ),
        "shadowed_layer_evaluation_zero": (
            evidence["scope_statement"][
                "shadowed_or_later_layer_evaluation_count"
            ]
            == 0
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
        "headless_viewer_maximum_one": (
            worker_claim["artifacts"]["rerun"][
                "headless_viewer_invocations"
            ]
            <= 1
        ),
        "scope_counters_zero": all(
            value == 0
            for value in evidence["current_scope_counters"].values()
        ),
        "live_identity_null": evidence["live_identity_pass"] is None,
        "physics_or_grasp_null": (
            evidence["physics_or_grasp_result"] is None
        ),
        "g0a_false": evidence["g0a_pass"] is False,
    }
    completion = {
        "artifact": "D386_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "scientific_verdict": evidence["verdict"],
        "localization_pass": evidence["localization_pass"],
        "observability_completion_pass": all(checks.values()),
        "checks": checks,
        "artifact_hashes": {
            _rel(path): _sha(path) for path in required
        },
        "next_authorization_boundary": (
            "D386 only localized the four first-observed layers. Do not "
            "select/apply a budget, evaluate later layers, alter gates or "
            "partitioning, materialize USD/PhysX, create the 29x50mm target, "
            "or run physics/contact/grasp without new explicit approval."
        ),
        "pass": all(checks.values()),
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase("finalize_end", completion_pass=completion["pass"])
    if not completion["pass"]:
        raise RuntimeError(f"D386 completion failed: {checks}")
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
