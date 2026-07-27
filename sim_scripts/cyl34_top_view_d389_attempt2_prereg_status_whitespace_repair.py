#!/usr/bin/env python3
"""D389 offline numeric-provenance and canonical-tie audit.

This program reads only immutable D388 artifacts.  It does not import or run
D388 and does not launch Isaac, Kit, PhysX, USD, Warp, CUDA, or robot code.
The two registered questions are:

1. Which lower-layer B35 path is canonical under the *whole-path* order, and
   at which state did D388's prefix-local DP discard it?
2. Are D388's eleven adjacent positive overlap reports already present in the
   reconstructed Float64 children, introduced by Float32 registration, or
   visible only under D388's frozen 5 nm numerical-band procedure?

The selected seam geometry intentionally remains D388's selected DP witness;
the newly audited global-canonical path is never substituted into geometry.
No budget, partition, collider, or scientific grasp result is adopted.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError


REPO = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO):
    if str(REPO) in sys.path:
        sys.path.remove(str(REPO))
    sys.path.insert(0, str(REPO))

CASE = "g0a_d389"
ATTEMPT = "attempt2_prereg_status_whitespace_repair"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT
SCRIPT = Path(__file__).resolve()
START = REPO / "START_HERE.md"

D389_ROOT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d389"
ATTEMPT1_DIR = (
    D389_ROOT
    / "attempt1_d388_overlap_gate_numeric_provenance_and_canonical_tie_audit"
)
ATTEMPT1_PREREG = ATTEMPT1_DIR / "d389_preregistration.json"
ATTEMPT1_PHASES = ATTEMPT1_DIR / "d389_phase_markers.jsonl"
ATTEMPT1_SCRIPT = REPO / (
    "sim_scripts/"
    "cyl34_top_view_d389_d388_overlap_gate_numeric_provenance_and_"
    "canonical_tie_audit.py"
)
ATTEMPT1_SHA256 = {
    "preregistration": (
        "a6cf72e4527ce40c7bc5cce6334608fa6a697b649bb48e5e31de2ceaa2aa8fc3"
    ),
    "phase_markers": (
        "a4deb402e09ee3afae15395dd6ea6a8797f61458e146ea39f4fe6a5b82a610ee"
    ),
    "script": (
        "c8f1e07c628ecbefe2dcf49e1a94231b9cbbf51f0b6727e3d5fb4a8083d74b6e"
    ),
}
ATTEMPT1_DOWNSTREAM_NAMES = (
    "d389_offline_audit_invocation.json",
    "d389_offline_worker_stdout.log",
    "d389_offline_worker_stderr.log",
    "d389_offline_worker_supervisor.json",
    "d389_numeric_and_tie_audit_evidence.json",
    "d389_reconstructed_seam_witness_geometry.json",
    "d389_seam_numeric_provenance.csv",
    "d389_lower_b35_complete_path_ranking.csv",
    "d389_numeric_provenance_and_tie_audit_1920x1080.png",
    "d389_board_layout_validation.json",
    "d389_numeric_provenance_and_tie_audit.rrd",
    "d389_numeric_provenance_and_tie_audit.rbl",
    "d389_rerun_validation.json",
    "d389_rerun_inspection.png",
    "d389_manual_visual_inspection_template.json",
    "d389_manual_visual_inspection.json",
    "d389_offline_worker_claim.json",
    "d389_failure_attestation.json",
    "d389_completion_summary.json",
)

D388_DIR = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d388/"
    "attempt1_two_null_moving_support_midlayer_partition_repair_design"
)
D388_EVIDENCE = D388_DIR / "d388_two_null_reanchor_design_evidence.json"
D388_GEOMETRY = D388_DIR / "d388_two_null_reanchor_witness_geometry.json"
D388_CSV = D388_DIR / "d388_reanchored_candidate_cell_metrics.csv"

EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
EXPECTED_SHA256 = {
    "d388_evidence": "582368f093ba08fec0207967e8e24ac24f0a44774dfa1a7b8c82ae2b6781caba",
    "d388_geometry": "c119ededf4400efbef55de4d89ccd6c1c8b4e33d4d3795710b6882d369f5e882",
    "d388_csv": "a4640cfb09e9a0b4b72a08fa6401f16492c63c9b4089b8e83540a52c7c355505",
}
EXPECTED_START_SHA256 = (
    "ad8baa401c34688f3e7e60f6fb8fa797799645e16d21509d997da38dcaaf9f90"
)
EXPECTED_D388_VERDICT = "D388_REANCHOR_PARTITION_CONTRACT_FAIL_STOP"
NEW_VARIABLES = [
    "lower_b35_global_canonical_path_order_and_dp_pruning_provenance_v1",
    "adjacent_seam_prepost_float32_epsilon0_vs_frozen5nm_provenance_v1",
]

FLOAT_EPS_M = 5.0e-9
POSITIVE_VOLUME_EPS_M3 = 1.0e-18
MAXIMUM_FAN_GROUP = 4
MAXIMUM_BUDGET = 64
AUDITED_BUDGET = 35
STRICT_INTERIOR_RADIUS_MM = 1.0e-10
DEADLINE_SECONDS = 300.0
EXPECTED_COMPLETE_PATHS = 151_664
EXPECTED_B35_PATHS = 22_464
EXPECTED_GLOBAL_CUTS = [0, 1, 5, 9, 10, 14, 18, 22]
EXPECTED_LOCAL_CUTS = [0, 2, 5, 9, 10, 14, 18, 22]
EXPECTED_FIRST_PRUNING_STATE = 5

PREREG = OUT_DIR / "d389_preregistration.json"
PHASES = OUT_DIR / "d389_phase_markers.jsonl"
INVOCATION = OUT_DIR / "d389_offline_audit_invocation.json"
STDOUT = OUT_DIR / "d389_offline_worker_stdout.log"
STDERR = OUT_DIR / "d389_offline_worker_stderr.log"
SUPERVISOR = OUT_DIR / "d389_offline_worker_supervisor.json"
EVIDENCE = OUT_DIR / "d389_numeric_and_tie_audit_evidence.json"
GEOMETRY = OUT_DIR / "d389_reconstructed_seam_witness_geometry.json"
SEAM_CSV = OUT_DIR / "d389_seam_numeric_provenance.csv"
PATH_CSV = OUT_DIR / "d389_lower_b35_complete_path_ranking.csv"
BOARD = OUT_DIR / "d389_numeric_provenance_and_tie_audit_1920x1080.png"
BOARD_LAYOUT = OUT_DIR / "d389_board_layout_validation.json"
RRD = OUT_DIR / "d389_numeric_provenance_and_tie_audit.rrd"
RBL = OUT_DIR / "d389_numeric_provenance_and_tie_audit.rbl"
RERUN_VALIDATION = OUT_DIR / "d389_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d389_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d389_manual_visual_inspection_template.json"
MANUAL = OUT_DIR / "d389_manual_visual_inspection.json"
WORKER_CLAIM = OUT_DIR / "d389_offline_worker_claim.json"
FAILURE_ATTESTATION = OUT_DIR / "d389_failure_attestation.json"
COMPLETION = OUT_DIR / "d389_completion_summary.json"

RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

TARGETS = [
    {
        "prim": "p000_proximal_upper_arm_hull_a",
        "cuts": [0, 3, 7, 11, 12, 16, 20],
        "anchor": 11,
        "short": "UPPER",
    },
    {
        "prim": "p002_proximal_lower_arm_hull_a",
        "cuts": EXPECTED_LOCAL_CUTS,
        "anchor": 10,
        "short": "LOWER",
    },
]

SCOPE_COUNTERS = {
    key: 0
    for key in (
        "d388_imports_or_executions",
        "d388_files_written_or_finalized",
        "partition_or_geometry_changes",
        "tolerance_or_gate_changes",
        "budget_selections_or_applications",
        "collider_or_asset_materializations",
        "isaac_kit_physx_usd_warp_cuda_launches",
        "cylinder_creates_or_writes",
        "controlled_physics_steps",
        "q5_samples",
        "contact_queries",
        "grasp_trials",
        "target_ik_path_changes",
        "material_mass_actuator_physics_setting_changes",
        "process_signals",
    )
}

_deadline_monotonic: float | None = None


class CooperativeDeadlineExceeded(RuntimeError):
    pass


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("refusing non-finite JSON scalar")
    return value


def _write_json_x(path: Path, value: Any) -> None:
    payload = json.dumps(
        _native(value), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    # Git porcelain's leading status-column space is data.  Remove only line
    # terminators; `.strip()` would turn " M path" into the different "M path".
    return result.stdout.rstrip("\r\n")


def _inputs() -> dict[str, str]:
    return {
        "d388_evidence": _sha(D388_EVIDENCE),
        "d388_geometry": _sha(D388_GEOMETRY),
        "d388_csv": _sha(D388_CSV),
    }


def _safe_hash_record(path: Path) -> dict[str, Any]:
    try:
        return {
            "path": _rel(path),
            "exists": path.is_file(),
            "sha256": _sha(path) if path.is_file() else None,
            "error": None,
        }
    except Exception as exc:
        return {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _safe_input_snapshot() -> dict[str, dict[str, Any]]:
    return {
        "d388_evidence": _safe_hash_record(D388_EVIDENCE),
        "d388_geometry": _safe_hash_record(D388_GEOMETRY),
        "d388_csv": _safe_hash_record(D388_CSV),
    }


def _attempt1_control_continuity() -> dict[str, Any]:
    expected_files = sorted(
        [_rel(ATTEMPT1_PHASES), _rel(ATTEMPT1_PREREG)]
    )
    downstream = [ATTEMPT1_DIR / name for name in ATTEMPT1_DOWNSTREAM_NAMES]
    try:
        prereg = _read_json(ATTEMPT1_PREREG)
        phase_rows = [
            json.loads(line)
            for line in ATTEMPT1_PHASES.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        actual_files = sorted(
            _rel(path)
            for path in ATTEMPT1_DIR.rglob("*")
            if path.is_file()
        )
        false_checks = sorted(
            name
            for name, passed in prereg.get("checks", {}).items()
            if passed is not True
        )
        checks = {
            "attempt1_exact_file_set_two_only": actual_files == expected_files,
            "attempt1_prereg_hash_exact": (
                _sha(ATTEMPT1_PREREG)
                == ATTEMPT1_SHA256["preregistration"]
            ),
            "attempt1_phase_hash_exact": (
                _sha(ATTEMPT1_PHASES)
                == ATTEMPT1_SHA256["phase_markers"]
            ),
            "attempt1_script_hash_exact": (
                _sha(ATTEMPT1_SCRIPT) == ATTEMPT1_SHA256["script"]
            ),
            "attempt1_prereg_false_only_git_status_checks": (
                prereg.get("pass") is False
                and false_checks
                == [
                    "git_status_after_exact_allowed",
                    "git_status_before_exact_allowed",
                ]
            ),
            "attempt1_prereg_script_lineage_exact": (
                prereg.get("script") == _rel(ATTEMPT1_SCRIPT)
                and prereg.get("script_sha256")
                == ATTEMPT1_SHA256["script"]
            ),
            "attempt1_phase_prepare_start_end_false_exact": (
                [row.get("phase") for row in phase_rows]
                == ["prepare_start", "prepare_end"]
                and len(phase_rows) == 2
                and phase_rows[1].get("pass_value") is False
            ),
            "attempt1_downstream_artifacts_all_absent": all(
                not path.exists() for path in downstream
            ),
        }
        return {
            "artifact": "D389_ATTEMPT1_CONTROL_CONTINUITY_V1",
            "operational_verdict": (
                "D389_ATTEMPT1_PRE_WORKER_GIT_PORCELAIN_"
                "LEADING_SPACE_GATE_FAIL_STOP"
            ),
            "expected_files": expected_files,
            "actual_files": actual_files,
            "false_preregistration_checks": false_checks,
            "phase_sequence": [row.get("phase") for row in phase_rows],
            "downstream_absence": {
                _rel(path): not path.exists() for path in downstream
            },
            "actual_worker_invocations": 0,
            "retries": 0,
            "headless_viewer_invocations": 0,
            "checks": checks,
            "pass": all(checks.values()),
            "error": None,
        }
    except Exception as exc:
        return {
            "artifact": "D389_ATTEMPT1_CONTROL_CONTINUITY_V1",
            "operational_verdict": (
                "D389_ATTEMPT1_CONTROL_CONTINUITY_READ_FAIL_STOP"
            ),
            "expected_files": expected_files,
            "actual_files": None,
            "actual_worker_invocations": None,
            "retries": None,
            "headless_viewer_invocations": None,
            "checks": {},
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _phase(name: str, **extra: Any) -> None:
    record = {
        "phase": name,
        "monotonic_seconds": time.monotonic(),
        **extra,
    }
    with PHASES.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_native(record), sort_keys=True) + "\n")


def _deadline(where: str) -> None:
    if _deadline_monotonic is not None and time.monotonic() > _deadline_monotonic:
        raise CooperativeDeadlineExceeded(where)


def _direct_import_roots() -> list[str]:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return sorted(roots)


def _unique_f64(
    points: np.ndarray, *, qhull_options: str | None = None
) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(unique) < 4:
        raise ValueError("fewer than four unique points")
    if np.linalg.matrix_rank(unique - unique.mean(axis=0)) < 3:
        raise ValueError("points are not three-dimensional")
    hull = ConvexHull(unique, qhull_options=qhull_options)
    return unique[np.asarray(hull.vertices, dtype=np.int64)]


def _polyhedron_edges(
    points: np.ndarray, *, qhull_options: str | None = None
) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(source, qhull_options=qhull_options)
    groups: dict[tuple[float, ...], set[int]] = {}
    for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
        length = float(np.linalg.norm(equation[:3]))
        key = tuple(np.round(equation / length, decimals=7))
        groups.setdefault(key, set()).update(map(int, simplex))
    memberships: dict[tuple[int, int], int] = {}
    for vertices in groups.values():
        ordered = sorted(vertices)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                pair = (int(left), int(right))
                memberships[pair] = memberships.get(pair, 0) + 1
    edges = sorted(pair for pair, count in memberships.items() if count >= 2)
    if not edges:
        raise RuntimeError("convex edge reconstruction failed")
    return np.asarray(edges, dtype=np.int64)


def _clip_plane(
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
    *,
    epsilon_m: float,
    qhull_options: str | None = None,
) -> np.ndarray:
    """D388-style exact-plane clip with a parameterized classification band."""
    source = np.asarray(points, dtype=np.float64)
    unit = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(unit))
    unit /= length
    signed_offset = float(offset) / length
    values = source @ unit + signed_offset
    keep = values <= epsilon_m
    output = [point for point in source[keep]]
    for left, right in _polyhedron_edges(
        source, qhull_options=qhull_options
    ):
        v0, v1 = values[int(left)], values[int(right)]
        if (v0 < -epsilon_m and v1 > epsilon_m) or (
            v1 < -epsilon_m and v0 > epsilon_m
        ):
            ratio = -v0 / (v1 - v0)
            output.append(
                source[int(left)]
                + ratio * (source[int(right)] - source[int(left)])
            )
    return _unique_f64(
        np.asarray(output, dtype=np.float64),
        qhull_options=qhull_options,
    )


def _profile(points: np.ndarray, thin_axis: int) -> tuple[np.ndarray, list[int]]:
    keep = [index for index in range(3) if index != thin_axis]
    projected = np.unique(np.asarray(points)[:, keep], axis=0)
    hull = ConvexHull(projected)
    return projected[np.asarray(hull.vertices, dtype=np.int64)], keep


def _reconstruct_child_pre_f32(
    parent_points: np.ndarray,
    thin_axis: int,
    rotated_polygon: np.ndarray,
    keep: list[int],
    start_state: int,
    end_state: int,
) -> np.ndarray:
    polygon = np.vstack(
        [
            rotated_polygon[0],
            rotated_polygon[start_state + 1 : end_state + 2],
        ]
    )
    clipped = np.asarray(parent_points, dtype=np.float64)
    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]
        delta = end - start
        normal_2d = np.asarray([delta[1], -delta[0]], dtype=np.float64)
        offset = float(delta[0] * start[1] - delta[1] * start[0])
        normal_3d = np.zeros(3, dtype=np.float64)
        normal_3d[keep] = normal_2d
        clipped = _clip_plane(
            clipped,
            normal_3d,
            offset,
            epsilon_m=FLOAT_EPS_M,
        )
    return clipped


def _post_f32_hull_vertices(points: np.ndarray) -> np.ndarray:
    registered = np.unique(
        np.asarray(points, dtype=np.float32).astype(np.float64), axis=0
    )
    hull = ConvexHull(registered)
    return registered[np.asarray(hull.vertices, dtype=np.int64)]


def _unique_planes(
    points_m: np.ndarray, *, qhull_options: str | None = None
) -> np.ndarray:
    """Return unrounded normalized halfspaces; rounding is only a dedupe key."""
    equations = np.asarray(
        ConvexHull(points_m, qhull_options=qhull_options).equations,
        dtype=np.float64,
    )
    result: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for equation in equations:
        equation = equation / np.linalg.norm(equation[:3])
        key = tuple(np.round(equation, decimals=12))
        if key not in seen:
            seen.add(key)
            result.append(equation)
    return np.asarray(result, dtype=np.float64)


def _directional_intersection_once(
    source_m: np.ndarray,
    clipping_m: np.ndarray,
    *,
    epsilon_m: float,
    quantize_planes_for_d388: bool,
    qhull_options: str | None,
) -> dict[str, Any]:
    """Run one directional clip with explicitly registered plane semantics."""
    _deadline("directional_intersection_start")
    try:
        points = np.unique(
            np.asarray(source_m, dtype=np.float64), axis=0
        )
        clipping = np.unique(
            np.asarray(clipping_m, dtype=np.float64), axis=0
        )
        if len(points) < 4 or len(clipping) < 4:
            raise ValueError("input hull has fewer than four unique points")
        if (
            np.linalg.matrix_rank(points - points.mean(axis=0)) < 3
            or np.linalg.matrix_rank(
                clipping - clipping.mean(axis=0)
            )
            < 3
        ):
            raise ValueError("input hull is not three-dimensional")
        equations = np.asarray(
            ConvexHull(
                clipping, qhull_options=qhull_options
            ).equations,
            dtype=np.float64,
        )
    except (ValueError, QhullError) as exc:
        return {
            "calculation_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "volume_m3": None,
            "positive_volume": None,
            "zero_reason": None,
            "clip_count": 0,
            "skipped_inside_plane_count": 0,
        }
    normalized = equations / np.linalg.norm(equations[:, :3], axis=1)[:, None]
    if quantize_planes_for_d388:
        selected = np.unique(np.round(normalized, decimals=12), axis=0)
    else:
        selected_rows: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for equation in normalized:
            key = tuple(np.round(equation, decimals=12))
            if key not in seen:
                seen.add(key)
                selected_rows.append(equation)
        selected = np.asarray(selected_rows, dtype=np.float64)
    clip_count = 0
    skipped = 0
    for equation in selected:
        _deadline("directional_intersection_plane")
        values = points @ equation[:3] + equation[3]
        if float(values.max()) <= epsilon_m:
            skipped += 1
            continue
        if float(values.min()) > epsilon_m:
            return {
                "calculation_pass": True,
                "error": None,
                "volume_m3": 0.0,
                "positive_volume": False,
                "zero_reason": "strictly_disjoint_halfspace",
                "clip_count": clip_count,
                "skipped_inside_plane_count": skipped,
            }
        minimum_before = float(values.min())
        try:
            points = _clip_plane(
                points,
                equation[:3],
                float(equation[3]),
                epsilon_m=epsilon_m,
                qhull_options=qhull_options,
            )
        except (ValueError, QhullError) as exc:
            if isinstance(exc, QhullError) and not quantize_planes_for_d388:
                return {
                    "calculation_pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "volume_m3": None,
                    "positive_volume": None,
                    "zero_reason": None,
                    "clip_count": clip_count + 1,
                    "skipped_inside_plane_count": skipped,
                }
            if minimum_before >= -epsilon_m:
                return {
                    "calculation_pass": True,
                    "error": None,
                    "volume_m3": 0.0,
                    "positive_volume": False,
                    "zero_reason": "boundary_touch_only",
                    "clip_count": clip_count + 1,
                    "skipped_inside_plane_count": skipped,
                }
            return {
                "calculation_pass": False,
                "error": f"{type(exc).__name__}: {exc}",
                "volume_m3": None,
                "positive_volume": None,
                "zero_reason": None,
                "clip_count": clip_count + 1,
                "skipped_inside_plane_count": skipped,
            }
        clip_count += 1
    try:
        volume = float(
            ConvexHull(points, qhull_options=qhull_options).volume
        )
    except QhullError as exc:
        return {
            "calculation_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "volume_m3": None,
            "positive_volume": None,
            "zero_reason": None,
            "clip_count": clip_count,
            "skipped_inside_plane_count": skipped,
        }
    return {
        "calculation_pass": True,
        "error": None,
        "volume_m3": volume,
        "positive_volume": volume > POSITIVE_VOLUME_EPS_M3,
        "zero_reason": None if volume > POSITIVE_VOLUME_EPS_M3 else "zero_volume",
        "clip_count": clip_count,
        "skipped_inside_plane_count": skipped,
    }


def _directional_intersection(
    source_m: np.ndarray,
    clipping_m: np.ndarray,
    *,
    epsilon_m: float,
    quantize_planes_for_d388: bool,
) -> dict[str, Any]:
    """Use D388 quantization only for replay; ε=0 retains original planes."""
    result = _directional_intersection_once(
        source_m,
        clipping_m,
        epsilon_m=epsilon_m,
        quantize_planes_for_d388=quantize_planes_for_d388,
        qhull_options=None,
    )
    fallback_used = False
    if (
        not quantize_planes_for_d388
        and result["calculation_pass"] is False
        and "QhullError" in str(result.get("error"))
    ):
        fallback_used = True
        result = _directional_intersection_once(
            source_m,
            clipping_m,
            epsilon_m=epsilon_m,
            quantize_planes_for_d388=False,
            qhull_options="Q12 Pp",
        )
    return {
        **result,
        "plane_equation_mode": (
            "D388_ROUNDED_12_DECIMAL_REPLAY"
            if quantize_planes_for_d388
            else "UNROUNDED_EPSILON0_WITH_ROUNDED_DEDUPE_KEY_ONLY"
        ),
        "qhull_fallback_used": fallback_used,
        "qhull_fallback_options": "Q12 Pp" if fallback_used else None,
    }


def _bidirectional_intersection(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon_m: float,
    quantize_planes_for_d388: bool,
) -> dict[str, Any]:
    forward = _directional_intersection(
        left,
        right,
        epsilon_m=epsilon_m,
        quantize_planes_for_d388=quantize_planes_for_d388,
    )
    reverse = _directional_intersection(
        right,
        left,
        epsilon_m=epsilon_m,
        quantize_planes_for_d388=quantize_planes_for_d388,
    )
    values = [
        row["volume_m3"]
        for row in (forward, reverse)
        if row["volume_m3"] is not None
    ]
    maximum = max(values) if values else None
    scale = max(maximum or 0.0, 1.0e-30)
    return {
        "left_clipped_by_right": forward,
        "right_clipped_by_left": reverse,
        "calculation_pass": bool(
            forward["calculation_pass"] and reverse["calculation_pass"]
        ),
        "maximum_volume_m3": maximum,
        "positive_volume": (
            maximum > POSITIVE_VOLUME_EPS_M3
            if maximum is not None
            else None
        ),
        "directional_relative_difference": (
            abs(float(forward["volume_m3"]) - float(reverse["volume_m3"]))
            / scale
            if forward["volume_m3"] is not None
            and reverse["volume_m3"] is not None
            else None
        ),
    }


def _strict_intersection_once(
    left_m: np.ndarray,
    right_m: np.ndarray,
    *,
    qhull_options: str | None,
) -> dict[str, Any]:
    """Symmetric epsilon-zero intersection in mm with a signed inradius."""
    _deadline("strict_intersection_start")
    try:
        equations_m = np.vstack(
            [
                _unique_planes(
                    np.asarray(left_m, dtype=np.float64),
                    qhull_options=qhull_options,
                ),
                _unique_planes(
                    np.asarray(right_m, dtype=np.float64),
                    qhull_options=qhull_options,
                ),
            ]
        )
        equations = equations_m.copy()
        equations[:, 3] *= 1000.0
    except (ValueError, QhullError) as exc:
        return {
            "calculation_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "chebyshev_center_m": None,
            "signed_inradius_nm": None,
            "volume_m3": None,
            "positive_volume": None,
        }
    objective = np.asarray([0.0, 0.0, 0.0, -1.0])
    a_ub = np.column_stack([equations[:, :3], np.ones(len(equations))])
    b_ub = -equations[:, 3]
    solution = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * 4,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-10,
            "dual_feasibility_tolerance": 1.0e-10,
            "ipm_optimality_tolerance": 1.0e-12,
        },
    )
    _deadline("strict_intersection_after_linprog")
    if not solution.success:
        return {
            "calculation_pass": False,
            "error": solution.message,
            "chebyshev_center_m": None,
            "signed_inradius_nm": None,
            "volume_m3": None,
            "positive_volume": None,
        }
    center = np.asarray(solution.x[:3], dtype=np.float64)
    radius_mm = float(solution.x[3])
    signed_plane_values_mm = (
        equations[:, :3] @ center + equations[:, 3]
    )
    constraint_slack_mm = -signed_plane_values_mm - radius_mm
    volume_mm3 = 0.0
    error = None
    if radius_mm > STRICT_INTERIOR_RADIUS_MM:
        try:
            vertices = HalfspaceIntersection(
                equations,
                center,
                qhull_options=qhull_options,
            ).intersections
            volume_mm3 = float(
                ConvexHull(
                    vertices, qhull_options=qhull_options
                ).volume
            )
        except (ValueError, QhullError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    _deadline("strict_intersection_end")
    objective_residual_mm = abs(float(solution.fun) + radius_mm)
    solver_contract_pass = bool(
        float(constraint_slack_mm.min()) >= -5.0e-10
        and objective_residual_mm <= 1.0e-12
    )
    if error is None and not solver_contract_pass:
        error = "solver_residual_contract_failed"
    calculation_pass = error is None and solver_contract_pass
    volume_m3 = volume_mm3 * 1.0e-9 if calculation_pass else None
    return {
        "calculation_pass": calculation_pass,
        "error": error,
        "chebyshev_center_m": center / 1000.0,
        "signed_inradius_nm": radius_mm * 1.0e6,
        "strict_interior_radius_threshold_nm": (
            STRICT_INTERIOR_RADIUS_MM * 1.0e6
        ),
        "highs_tolerances": {
            "primal_feasibility": 1.0e-10,
            "dual_feasibility": 1.0e-10,
            "ipm_optimality": 1.0e-12,
            "units_for_feasibility": "mm",
        },
        "maximum_halfspace_value_at_center_mm": float(
            signed_plane_values_mm.max()
        ),
        "minimum_constraint_slack_mm": float(
            constraint_slack_mm.min()
        ),
        "maximum_constraint_slack_mm": float(
            constraint_slack_mm.max()
        ),
        "solver_objective_residual_mm": abs(
            float(solution.fun) + radius_mm
        ),
        "solver_residual_contract_pass": solver_contract_pass,
        "volume_mm3": volume_mm3 if calculation_pass else None,
        "volume_m3": volume_m3,
        "positive_volume": (
            volume_m3 > POSITIVE_VOLUME_EPS_M3
            if volume_m3 is not None
            else None
        ),
    }


def _strict_intersection(
    left_m: np.ndarray, right_m: np.ndarray
) -> dict[str, Any]:
    """Retry only Qhull precision failures with deterministic Q12/Pp."""
    result = _strict_intersection_once(
        left_m, right_m, qhull_options=None
    )
    fallback_used = False
    if (
        result["calculation_pass"] is False
        and "QhullError" in str(result.get("error"))
    ):
        fallback_used = True
        result = _strict_intersection_once(
            left_m,
            right_m,
            qhull_options="Q12 Pp",
        )
    return {
        **result,
        "halfspace_equation_mode": (
            "UNROUNDED_METERS_CONVERTED_TO_MM_AFTER_DEDUPE"
        ),
        "qhull_fallback_used": fallback_used,
        "qhull_fallback_options": "Q12 Pp" if fallback_used else None,
    }


def _volumes_agree(
    left: float | None,
    right: float | None,
    *,
    relative: float = 1.0e-5,
    absolute_m3: float = 1.0e-20,
) -> bool:
    return bool(
        left is not None
        and right is not None
        and math.isclose(left, right, rel_tol=relative, abs_tol=absolute_m3)
    )


def _nonnegative_float64_ulp_distance(
    left: float | None, right: float | None
) -> int | None:
    if (
        left is None
        or right is None
        or left < 0.0
        or right < 0.0
        or not math.isfinite(left)
        or not math.isfinite(right)
    ):
        return None
    left_bits = struct.unpack(">Q", struct.pack(">d", float(left)))[0]
    right_bits = struct.unpack(">Q", struct.pack(">d", float(right)))[0]
    return abs(int(left_bits) - int(right_bits))


def _box_mm(low: tuple[float, ...], high: tuple[float, ...]) -> np.ndarray:
    points_mm = np.asarray(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )
    return points_mm / 1000.0


def _numeric_controls() -> dict[str, Any]:
    base = _box_mm((0, 0, 0), (1, 1, 1))
    overlap = _box_mm((0.5, 0, 0), (1.5, 1, 1))
    touch = _box_mm((1, 0, 0), (2, 1, 1))
    gap10nm = _box_mm((1.000010, 0, 0), (2.000010, 1, 1))
    gap2nm = _box_mm((1.000002, 0, 0), (2.000002, 1, 1))
    overlap2nm = _box_mm((0.999998, 0, 0), (1.999998, 1, 1))
    angle = 0.371
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cases = {
        "duplicate": (base, base),
        "known_0p5mm_partial_overlap": (base, overlap),
        "exact_shared_face": (base, touch),
        "known_10nm_gap": (base, gap10nm),
        "known_2nm_gap": (base, gap2nm),
        "known_2nm_overlap": (base, overlap2nm),
        "tilted_known_2nm_overlap": (
            base @ rotation.T,
            overlap2nm @ rotation.T,
        ),
    }
    rows = {}
    for name, (left, right) in cases.items():
        _deadline(f"numeric_control_{name}_start")
        strict = _strict_intersection(left, right)
        epsilon0 = _bidirectional_intersection(
            left,
            right,
            epsilon_m=0.0,
            quantize_planes_for_d388=False,
        )
        epsilon5 = _bidirectional_intersection(
            left,
            right,
            epsilon_m=FLOAT_EPS_M,
            quantize_planes_for_d388=True,
        )
        swapped = _strict_intersection(right, left)
        rows[name] = {
            "strict_lp_halfspace": strict,
            "directional_epsilon0": epsilon0,
            "directional_frozen5nm": epsilon5,
            "swapped_strict_lp_halfspace": swapped,
            "swap_invariant": (
                strict["positive_volume"] == swapped["positive_volume"]
                and _volumes_agree(strict["volume_m3"], swapped["volume_m3"])
            ),
            "lp_and_directional_epsilon0_agree": (
                strict["positive_volume"] == epsilon0["positive_volume"]
                and _volumes_agree(
                    strict["volume_m3"], epsilon0["maximum_volume_m3"]
                )
            ),
        }
        _deadline(f"numeric_control_{name}_end")
    degenerate = _strict_intersection(
        np.asarray(
            [[0, 0, 0], [1e-3, 0, 0], [0, 1e-3, 0], [1e-3, 1e-3, 0]],
            dtype=np.float64,
        ),
        base,
    )
    checks = {
        "duplicate_positive": rows["duplicate"]["strict_lp_halfspace"][
            "positive_volume"
        ]
        is True,
        "known_partial_overlap_positive": rows[
            "known_0p5mm_partial_overlap"
        ]["strict_lp_halfspace"]["positive_volume"]
        is True,
        "shared_face_zero": rows["exact_shared_face"]["strict_lp_halfspace"][
            "positive_volume"
        ]
        is False,
        "gap_negative_and_zero": (
            rows["known_10nm_gap"]["strict_lp_halfspace"][
                "signed_inradius_nm"
            ]
            < 0.0
            and rows["known_10nm_gap"]["strict_lp_halfspace"][
                "positive_volume"
            ]
            is False
        ),
        "two_nm_gap_epsilon0_zero_and_5nm_nonpositive": (
            rows["known_2nm_gap"]["directional_epsilon0"][
                "positive_volume"
            ]
            is False
            and rows["known_2nm_gap"]["directional_frozen5nm"][
                "positive_volume"
            ]
            is False
        ),
        "two_nm_overlap_positive": rows["known_2nm_overlap"][
            "strict_lp_halfspace"
        ]["positive_volume"]
        is True,
        "tilted_two_nm_overlap_positive": rows[
            "tilted_known_2nm_overlap"
        ]["strict_lp_halfspace"]["positive_volume"]
        is True,
        "all_calculations_pass": all(
            row["strict_lp_halfspace"]["calculation_pass"]
            and row["directional_epsilon0"]["calculation_pass"]
            and row["directional_frozen5nm"]["calculation_pass"]
            for row in rows.values()
        ),
        "all_swap_invariant": all(row["swap_invariant"] for row in rows.values()),
        "all_lp_directional_epsilon0_agree": all(
            row["lp_and_directional_epsilon0_agree"] for row in rows.values()
        ),
        "degenerate_input_fails_closed": (
            degenerate["calculation_pass"] is False
        ),
    }
    return {
        "cases": rows,
        "degenerate_input": degenerate,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _candidate_graph(
    *, reverse_source_rows: bool = False
) -> dict[tuple[int, int], dict[str, Any]]:
    with D388_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    lower = [
        row
        for row in rows
        if row["prim_name"] == "p002_proximal_lower_arm_hull_a"
        and row["region_name"] == "z_layer_01"
    ]
    if reverse_source_rows:
        lower.reverse()
    graph: dict[tuple[int, int], dict[str, Any]] = {}
    for row in lower:
        edge = (int(row["start_state"]), int(row["end_state"]))
        graph[edge] = {
            "pass": row["non_vertex_gates_pass"] == "True"
            and int(row["vertex_count"] or 999) <= MAXIMUM_BUDGET,
            "vertex_count": int(row["vertex_count"] or 999),
        }
    if len(graph) != 82:
        raise RuntimeError(f"lower graph row count is {len(graph)}, not 82")
    return graph


def _fixed_budget_global_dp(
    graph: dict[tuple[int, int], dict[str, Any]], budget: int
) -> dict[str, Any] | None:
    """Independent suffix DP: minimize full child-count, then full cuts."""
    suffix: list[tuple[int, tuple[int, ...]] | None] = [None] * 23
    suffix[22] = (0, ())
    for state in range(21, -1, -1):
        options = []
        for end in range(state + 1, min(state + MAXIMUM_FAN_GROUP, 22) + 1):
            edge = graph[(state, end)]
            future = suffix[end]
            if (
                not edge["pass"]
                or edge["vertex_count"] > budget
                or future is None
            ):
                continue
            options.append((future[0] + 1, (end,) + future[1]))
        suffix[state] = min(options) if options else None
    if suffix[0] is None:
        return None
    return {
        "child_count": suffix[0][0],
        "cuts": [0, *suffix[0][1]],
    }


def _enumerate_paths(
    graph: dict[tuple[int, int], dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    complete: list[dict[str, Any]] = []
    visited = 0

    def visit(state: int, cuts: tuple[int, ...], maximum: int) -> None:
        nonlocal visited
        visited += 1
        if visited % 4096 == 0:
            _deadline(f"path_prefix_{visited}")
        if state == 22:
            complete.append(
                {
                    "cuts": cuts,
                    "child_count": len(cuts) - 1,
                    "maximum_vertex_count": maximum,
                }
            )
            return
        for end in range(state + 1, min(state + MAXIMUM_FAN_GROUP, 22) + 1):
            edge = graph[(state, end)]
            if edge["pass"]:
                visit(
                    end,
                    cuts + (end,),
                    max(maximum, int(edge["vertex_count"])),
                )

    visit(0, (0,), 0)
    return complete, visited


def _local_dp(
    graph: dict[tuple[int, int], dict[str, Any]], budget: int
) -> tuple[list[int], list[dict[str, Any]]]:
    states: list[tuple[int, int, tuple[int, ...]] | None] = [None] * 23
    states[0] = (0, 0, (0,))
    audit: list[dict[str, Any]] = []
    for end in range(1, 23):
        options = []
        for start in range(max(0, end - MAXIMUM_FAN_GROUP), end):
            previous = states[start]
            edge = graph[(start, end)]
            if previous is None or not edge["pass"] or edge["vertex_count"] > budget:
                continue
            options.append(
                (
                    previous[0] + 1,
                    max(previous[1], edge["vertex_count"]),
                    previous[2] + (end,),
                )
            )
        options.sort()
        states[end] = options[0] if options else None
        audit.append(
            {
                "state": end,
                "candidate_prefixes": [
                    {
                        "child_count": row[0],
                        "prefix_maximum_vertex_count": row[1],
                        "cuts": list(row[2]),
                    }
                    for row in options
                ],
                "retained": list(options[0][2]) if options else None,
            }
        )
    if states[-1] is None:
        raise RuntimeError("local DP unexpectedly has no B35 path")
    return list(states[-1][2]), audit


def _tie_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    graph = _candidate_graph()
    reversed_graph = _candidate_graph(reverse_source_rows=True)
    paths, visited = _enumerate_paths(graph)
    ranked = sorted(
        paths,
        key=lambda row: (
            row["maximum_vertex_count"],
            row["child_count"],
            row["cuts"],
        ),
    )
    b35 = [row for row in ranked if row["maximum_vertex_count"] <= 35]
    local_cuts, dp_states = _local_dp(graph, 35)
    global_cuts = list(b35[0]["cuts"])
    global_rank_of_local = next(
        index + 1 for index, row in enumerate(b35) if list(row["cuts"]) == local_cuts
    )
    fixed35 = _fixed_budget_global_dp(graph, 35)
    fixed34 = _fixed_budget_global_dp(graph, 34)
    reversed35 = _fixed_budget_global_dp(reversed_graph, 35)
    minimum_child_b35 = min(row["child_count"] for row in b35)
    minimum_child_b35_count = sum(
        row["child_count"] == minimum_child_b35 for row in b35
    )
    ranked_payload = [
        [
            row["maximum_vertex_count"],
            row["child_count"],
            list(row["cuts"]),
        ]
        for row in b35
    ]
    ranked_sha = hashlib.sha256(
        json.dumps(
            ranked_payload, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    at5 = next(row for row in dp_states if row["state"] == 5)
    p_global = next(
        row
        for row in at5["candidate_prefixes"]
        if row["cuts"] == [0, 1, 5]
    )
    p_local = next(
        row
        for row in at5["candidate_prefixes"]
        if row["cuts"] == [0, 2, 5]
    )
    state_by_end = {row["state"]: row for row in dp_states}
    global_prefixes = [
        list(tuple(global_cuts[: index + 1]))
        for index in range(1, len(global_cuts))
    ]
    first_global_prefix_discard = next(
        (
            prefix[-1]
            for prefix in global_prefixes
            if state_by_end[prefix[-1]]["retained"] != prefix
        ),
        None,
    )
    checks = {
        "complete_path_count_exact": len(paths) == EXPECTED_COMPLETE_PATHS,
        "b35_path_count_exact": len(b35) == EXPECTED_B35_PATHS,
        "minimum_budget_exact_35": ranked[0]["maximum_vertex_count"] == 35,
        "global_canonical_exact": global_cuts == EXPECTED_GLOBAL_CUTS,
        "local_dp_exact": local_cuts == EXPECTED_LOCAL_CUTS,
        "global_and_local_differ": global_cuts != local_cuts,
        "first_pruning_state_exact_5": (
            first_global_prefix_discard == EXPECTED_FIRST_PRUNING_STATE
            and p_global["child_count"] == p_local["child_count"] == 2
            and p_global["prefix_maximum_vertex_count"] == 16
            and p_local["prefix_maximum_vertex_count"] == 15
        ),
        "later_b35_edge_dominates_both_prefixes": (
            graph[(10, 14)]["vertex_count"] == 35
        ),
        "independent_fixed_b35_dp_agrees_global": (
            fixed35 is not None
            and fixed35["cuts"] == EXPECTED_GLOBAL_CUTS
            and fixed35["child_count"] == 7
        ),
        "fixed_b34_has_no_cover": fixed34 is None,
        "minimum_child_count_7_and_exactly_10_paths": (
            minimum_child_b35 == 7 and minimum_child_b35_count == 10
        ),
        "reversed_csv_row_order_invariant": reversed35 == fixed35,
        "ranked_payload_sha256_present": len(ranked_sha) == 64,
    }
    payload = {
        "source": _rel(D388_CSV),
        "global_whole_path_order": [
            "maximum_vertex_count",
            "child_count",
            "cut_states_lexicographic",
        ],
        "complete_path_count": len(paths),
        "visited_prefix_count": visited,
        "b35_complete_path_count": len(b35),
        "b35_minimum_child_count": minimum_child_b35,
        "b35_minimum_child_path_count": minimum_child_b35_count,
        "b35_ranked_payload_sha256": ranked_sha,
        "global_canonical": b35[0],
        "independent_fixed_budget_dp": {
            "budget_35": fixed35,
            "budget_34": fixed34,
            "reversed_source_budget_35": reversed35,
        },
        "d388_local_dp_result": {
            "cuts": local_cuts,
            "rank_under_global_b35_order": global_rank_of_local,
        },
        "first_local_pruning": {
            "state": EXPECTED_FIRST_PRUNING_STATE,
            "globally_canonical_prefix_discarded": p_global,
            "locally_retained_prefix": p_local,
            "cause": (
                "prefix-local order preferred maximum 15 over 16 before the "
                "later common 35-vertex edge made that difference irrelevant"
            ),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    return payload, b35


def _frozen_pair_map(target: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    overlap = target["selected_threshold_geometry_metrics"][
        "actual_float32_pairwise_halfspace_intersection"
    ]
    result = {}
    for row in overlap["pairs"]:
        key = tuple(sorted((row["left"], row["right"])))
        result[key] = row
    return result


def _seam_crossing(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rotated_polygon: np.ndarray,
    keep: list[int],
    cut_state: int,
) -> dict[str, Any]:
    """Measure each child crossing of its shared fan-ray plane."""
    anchor = rotated_polygon[0]
    endpoint = rotated_polygon[cut_state + 1]
    delta = endpoint - anchor
    normal_2d = np.asarray([delta[1], -delta[0]], dtype=np.float64)
    offset = float(delta[0] * anchor[1] - delta[1] * anchor[0])
    normal = np.zeros(3, dtype=np.float64)
    normal[keep] = normal_2d
    length = float(np.linalg.norm(normal))
    normal /= length
    offset /= length
    left_values = np.asarray(left) @ normal + offset
    right_values = np.asarray(right) @ normal + offset
    if float(left_values.mean()) > float(right_values.mean()):
        normal *= -1.0
        offset *= -1.0
        left_values *= -1.0
        right_values *= -1.0
    return {
        "cut_state": cut_state,
        "plane_normal": normal,
        "plane_offset_m": offset,
        "left_max_crossing_to_right_nm": max(
            float(left_values.max()), 0.0
        )
        * 1.0e9,
        "right_max_crossing_to_left_nm": max(
            float((-right_values).max()), 0.0
        )
        * 1.0e9,
        "left_signed_range_nm": [
            float(left_values.min()) * 1.0e9,
            float(left_values.max()) * 1.0e9,
        ],
        "right_signed_range_nm": [
            float(right_values.min()) * 1.0e9,
            float(right_values.max()) * 1.0e9,
        ],
    }


def _seam_audit() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    evidence = _read_json(D388_EVIDENCE)
    geometry = _read_json(D388_GEOMETRY)
    by_prim_e = {row["prim_name"]: row for row in evidence["target_results"]}
    by_prim_g = {row["prim_name"]: row for row in geometry["layers"]}
    pairs: list[dict[str, Any]] = []
    witness_layers = []
    bit_checks = []
    for spec in TARGETS:
        _deadline(f"seam_target_{spec['short']}_start")
        target = by_prim_e[spec["prim"]]
        layer = by_prim_g[spec["prim"]]
        if target["selected_threshold_cover"]["cut_states"] != spec["cuts"]:
            raise RuntimeError(f"frozen selected cuts mismatch: {spec['prim']}")
        parent = np.asarray(layer["parent_layer"]["vertices_f64_m"], dtype=np.float64)
        thin_axis = int(target["thin_axis_index"])
        polygon, keep = _profile(parent, thin_axis)
        rotated = np.roll(polygon, -spec["anchor"], axis=0)
        pre_children = []
        stored_children = layer["diagnostic_children"]
        for index, (start, end) in enumerate(
            zip(spec["cuts"][:-1], spec["cuts"][1:], strict=True)
        ):
            _deadline(f"reconstruct_{spec['short']}_{index}")
            pre = _reconstruct_child_pre_f32(
                parent, thin_axis, rotated, keep, start, end
            )
            post = _post_f32_hull_vertices(pre)
            stored = np.asarray(
                stored_children[index]["vertices_f64_m"], dtype=np.float64
            )
            exact = np.array_equal(post, stored)
            distances = np.linalg.norm(
                pre[:, None, :] - post[None, :, :], axis=2
            )
            pre_to_post_nm = float(
                distances.min(axis=1).max() * 1.0e9
            )
            post_to_pre_nm = float(
                distances.min(axis=0).max() * 1.0e9
            )
            bit_checks.append(exact)
            pre_children.append(
                {
                    "name": stored_children[index]["name"],
                    "pre": pre,
                    "post": stored,
                    "roundtrip_exact": exact,
                    "pre_vertex_count": len(pre),
                    "post_vertex_count": len(post),
                    "pre_to_post_hausdorff_nm": pre_to_post_nm,
                    "post_to_pre_hausdorff_nm": post_to_pre_nm,
                    "symmetric_hausdorff_nm": max(
                        pre_to_post_nm, post_to_pre_nm
                    ),
                    "pre_vertices_sha256": hashlib.sha256(
                        np.asarray(pre, dtype="<f8").tobytes()
                    ).hexdigest(),
                    "post_vertices_sha256": hashlib.sha256(
                        np.asarray(post, dtype="<f8").tobytes()
                    ).hexdigest(),
                    "range": [start, end],
                }
            )
        frozen = _frozen_pair_map(target)
        for left in range(len(pre_children)):
            for right in range(left + 1, len(pre_children)):
                _deadline(
                    f"seam_pair_{spec['short']}_{left}_{right}_start"
                )
                lhs, rhs = pre_children[left], pre_children[right]
                key = tuple(sorted((lhs["name"], rhs["name"])))
                frozen_row = frozen[key]
                pre_result = _strict_intersection(lhs["pre"], rhs["pre"])
                post_result = _strict_intersection(lhs["post"], rhs["post"])
                pre_directional0 = _bidirectional_intersection(
                    lhs["pre"],
                    rhs["pre"],
                    epsilon_m=0.0,
                    quantize_planes_for_d388=False,
                )
                post_directional0 = _bidirectional_intersection(
                    lhs["post"],
                    rhs["post"],
                    epsilon_m=0.0,
                    quantize_planes_for_d388=False,
                )
                post_replay5 = _bidirectional_intersection(
                    lhs["post"],
                    rhs["post"],
                    epsilon_m=FLOAT_EPS_M,
                    quantize_planes_for_d388=True,
                )
                frozen_forward = frozen_row["left_clipped_by_right"][
                    "volume_m3"
                ]
                frozen_reverse = frozen_row["right_clipped_by_left"][
                    "volume_m3"
                ]
                frozen_maximum = frozen_row[
                    "maximum_intersection_volume_m3"
                ]
                replay_values = [
                    (
                        post_replay5["left_clipped_by_right"]["volume_m3"],
                        frozen_forward,
                    ),
                    (
                        post_replay5["right_clipped_by_left"]["volume_m3"],
                        frozen_reverse,
                    ),
                    (post_replay5["maximum_volume_m3"], frozen_maximum),
                ]
                replay_exact = [
                    left == right for left, right in replay_values
                ]
                replay_ulp = [
                    _nonnegative_float64_ulp_distance(left, right)
                    for left, right in replay_values
                ]
                epsilon0_calculations_pass = bool(
                    pre_result["calculation_pass"]
                    and post_result["calculation_pass"]
                    and pre_directional0["calculation_pass"]
                    and post_directional0["calculation_pass"]
                )
                replay_calculation_pass = bool(
                    post_replay5["calculation_pass"]
                )
                independent_agreement = bool(
                    epsilon0_calculations_pass
                    and pre_result["positive_volume"]
                    == pre_directional0["positive_volume"]
                    and post_result["positive_volume"]
                    == post_directional0["positive_volume"]
                    and _volumes_agree(
                        pre_result["volume_m3"],
                        pre_directional0["maximum_volume_m3"],
                    )
                    and _volumes_agree(
                        post_result["volume_m3"],
                        post_directional0["maximum_volume_m3"],
                    )
                )
                replay_agreement = bool(
                    replay_calculation_pass
                    and all(
                        distance is not None and distance <= 4
                        for distance in replay_ulp
                    )
                    and post_replay5["positive_volume"]
                    == frozen_row["positive_volume_overlap"]
                )
                pair_valid = bool(
                    lhs["roundtrip_exact"]
                    and rhs["roundtrip_exact"]
                    and independent_agreement
                    and replay_agreement
                )
                if not pair_valid:
                    pair_class = "INDETERMINATE_PROVENANCE_OR_SOLVER_DISAGREEMENT"
                elif pre_result["positive_volume"]:
                    pair_class = (
                        "PRE_FLOAT32_D388_5NM_CONSTRUCTION_HAS_"
                        "EPSILON0_POSITIVE_OVERLAP"
                    )
                elif post_result["positive_volume"]:
                    pair_class = (
                        "FLOAT32_REGISTRATION_INTRODUCES_"
                        "EPSILON0_POSITIVE_OVERLAP"
                    )
                elif frozen_row["positive_volume_overlap"]:
                    pair_class = (
                        "FROZEN_5NM_PROCEDURE_ONLY_NO_"
                        "EPSILON0_POSITIVE_OVERLAP"
                    )
                else:
                    pair_class = "NO_POSITIVE_OVERLAP_IN_ANY_AUDITED_PROCEDURE"
                adjacent = right == left + 1
                pairs.append(
                    {
                        "target": spec["short"],
                        "prim_name": spec["prim"],
                        "left_index": left,
                        "right_index": right,
                        "adjacent": adjacent,
                        "per_pair_classification": pair_class,
                        "pair_provenance_and_solver_pass": pair_valid,
                        "epsilon0_calculations_pass": (
                            epsilon0_calculations_pass
                        ),
                        "frozen_5nm_replay_calculation_pass": (
                            replay_calculation_pass
                        ),
                        "pre_float32_epsilon0": pre_result,
                        "pre_float32_directional_epsilon0": pre_directional0,
                        "post_float32_epsilon0": post_result,
                        "post_float32_directional_epsilon0": post_directional0,
                        "post_float32_directional_frozen5nm_replay": (
                            post_replay5
                        ),
                        "frozen_d388_5nm_procedure": {
                            "positive_volume": frozen_row["positive_volume_overlap"],
                            "left_clipped_by_right_volume_m3": frozen_forward,
                            "right_clipped_by_left_volume_m3": frozen_reverse,
                            "maximum_intersection_volume_m3": frozen_maximum,
                        },
                        "independent_lp_vs_directional_epsilon0_agreement": (
                            independent_agreement
                        ),
                        "frozen_5nm_replay_agreement": replay_agreement,
                        "frozen_5nm_replay_exact_float64": {
                            "left_clipped_by_right": replay_exact[0],
                            "right_clipped_by_left": replay_exact[1],
                            "maximum": replay_exact[2],
                            "all_exact": all(replay_exact),
                        },
                        "frozen_5nm_replay_ulp_distance": {
                            "left_clipped_by_right": replay_ulp[0],
                            "right_clipped_by_left": replay_ulp[1],
                            "maximum": replay_ulp[2],
                            "registered_gate_maximum_ulp": 4,
                        },
                        "fan_seam_crossing": (
                            {
                                "pre_float32": _seam_crossing(
                                    lhs["pre"],
                                    rhs["pre"],
                                    rotated_polygon=rotated,
                                    keep=keep,
                                    cut_state=spec["cuts"][right],
                                ),
                                "post_float32": _seam_crossing(
                                    lhs["post"],
                                    rhs["post"],
                                    rotated_polygon=rotated,
                                    keep=keep,
                                    cut_state=spec["cuts"][right],
                                ),
                            }
                            if adjacent
                            else None
                        ),
                    }
                )
                _deadline(
                    f"seam_pair_{spec['short']}_{left}_{right}_end"
                )
        witness_layers.append(
            {
                "prim_name": spec["prim"],
                "selected_d388_cuts": spec["cuts"],
                "not_replaced_by_global_canonical_path": True,
                "parent_vertices_f64_m": parent,
                "children": [
                    {
                        "name": row["name"],
                        "range": row["range"],
                        "pre_float32_vertices_f64_m": row["pre"],
                        "stored_post_float32_vertices_f64_m": row["post"],
                        "f32_roundtrip_bit_exact": row["roundtrip_exact"],
                        "pre_vertex_count": row["pre_vertex_count"],
                        "post_vertex_count": row["post_vertex_count"],
                        "pre_to_post_hausdorff_nm": row[
                            "pre_to_post_hausdorff_nm"
                        ],
                        "post_to_pre_hausdorff_nm": row[
                            "post_to_pre_hausdorff_nm"
                        ],
                        "symmetric_hausdorff_nm": row[
                            "symmetric_hausdorff_nm"
                        ],
                        "pre_vertices_sha256": row[
                            "pre_vertices_sha256"
                        ],
                        "post_vertices_sha256": row[
                            "post_vertices_sha256"
                        ],
                    }
                    for row in pre_children
                ],
            }
        )
        _deadline(f"seam_target_{spec['short']}_end")
    adjacent = [row for row in pairs if row["adjacent"]]
    nonadjacent = [row for row in pairs if not row["adjacent"]]
    pre_positive = sum(
        row["pre_float32_epsilon0"]["positive_volume"] is True for row in adjacent
    )
    post_positive = sum(
        row["post_float32_epsilon0"]["positive_volume"] is True for row in adjacent
    )
    frozen_positive = sum(
        row["frozen_d388_5nm_procedure"]["positive_volume"] is True
        for row in adjacent
    )
    all_pairs_valid = all(row["pair_provenance_and_solver_pass"] for row in pairs)
    adjacent_class_counts: dict[str, int] = {}
    for row in adjacent:
        name = row["per_pair_classification"]
        adjacent_class_counts[name] = adjacent_class_counts.get(name, 0) + 1
    if not all_pairs_valid or not all(bit_checks):
        classification = "INDETERMINATE_PROVENANCE_OR_SOLVER_DISAGREEMENT"
    elif len(adjacent_class_counts) == 1:
        classification = next(iter(adjacent_class_counts))
    else:
        classification = "MIXED_ADJACENT_PAIR_PROVENANCE"
    controls = _numeric_controls()
    _deadline("seam_audit_after_controls")
    checks = {
        "d388_verdict_frozen": evidence["verdict"] == EXPECTED_D388_VERDICT,
        "f32_roundtrip_all_children_bit_exact": all(bit_checks)
        and len(bit_checks) == 13,
        "adjacent_pair_count_exact_11": len(adjacent) == 11,
        "nonadjacent_pair_count_exact_25": len(nonadjacent) == 25,
        "frozen_adjacent_positive_exact_11": frozen_positive == 11,
        "frozen_nonadjacent_positive_zero": all(
            row["frozen_d388_5nm_procedure"]["positive_volume"] is False
            for row in nonadjacent
        ),
        "epsilon0_calculation_all_pairs_pass": all(
            row[phase]["calculation_pass"]
            for row in pairs
            for phase in ("pre_float32_epsilon0", "post_float32_epsilon0")
        ),
        "independent_lp_directional_epsilon0_all_pairs_agree": all(
            row["independent_lp_vs_directional_epsilon0_agreement"]
            for row in pairs
        ),
        "frozen_5nm_all_pairs_replay_stored_volumes": all(
            row["frozen_5nm_replay_agreement"] for row in pairs
        ),
        "all_pair_classifications_determinate": all(
            not row["per_pair_classification"].startswith("INDETERMINATE")
            for row in pairs
        ),
        "epsilon0_nonadjacent_positive_zero": all(
            row["pre_float32_epsilon0"]["positive_volume"] is False
            and row["post_float32_epsilon0"]["positive_volume"] is False
            for row in nonadjacent
        ),
        "synthetic_numeric_controls_pass": controls["pass"],
    }
    result = {
        "selected_geometry_authority": (
            "D388 selected DP cuts are frozen for this seam audit; the newly "
            "canonical lower path is not substituted"
        ),
        "float64_reconstruction": (
            "D388 parent Float64 geometry plus identical fan cells and its "
            "frozen 5 nm construction clip, before Float32 registration"
        ),
        "signed_margin_definition": (
            "Chebyshev signed inradius of the combined normalized halfspaces; "
            "positive means a full-dimensional common interior, zero means "
            "touching, and negative means disjoint under a uniform relaxation"
        ),
        "adjacent_pair_count": len(adjacent),
        "nonadjacent_negative_control_count": len(nonadjacent),
        "adjacent_pre_float32_epsilon0_positive_count": pre_positive,
        "adjacent_post_float32_epsilon0_positive_count": post_positive,
        "adjacent_frozen_d388_5nm_positive_count": frozen_positive,
        "adjacent_pair_classification_counts": adjacent_class_counts,
        "classification": classification,
        "pair_results": pairs,
        "numeric_controls": controls,
        "checks": checks,
        "pass": all(checks.values()),
    }
    witness = {
        "artifact": "D389_RECONSTRUCTED_SEAM_WITNESS_GEOMETRY_V1",
        "authority": "offline diagnostic only; no collider adoption",
        "layers": witness_layers,
    }
    return result, witness, pairs


def _write_csvs(
    seams: list[dict[str, Any]], ranked: list[dict[str, Any]]
) -> None:
    seam_fields = [
        "target",
        "prim_name",
        "left_index",
        "right_index",
        "adjacent",
        "per_pair_classification",
        "pair_provenance_and_solver_pass",
        "pre_signed_inradius_nm",
        "pre_volume_m3",
        "pre_positive",
        "post_signed_inradius_nm",
        "post_volume_m3",
        "post_positive",
        "frozen_5nm_volume_m3",
        "frozen_5nm_positive",
        "pre_left_crossing_nm",
        "pre_right_crossing_nm",
        "post_left_crossing_nm",
        "post_right_crossing_nm",
    ]
    with SEAM_CSV.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=seam_fields)
        writer.writeheader()
        for row in seams:
            writer.writerow(
                {
                    "target": row["target"],
                    "prim_name": row["prim_name"],
                    "left_index": row["left_index"],
                    "right_index": row["right_index"],
                    "adjacent": row["adjacent"],
                    "per_pair_classification": row[
                        "per_pair_classification"
                    ],
                    "pair_provenance_and_solver_pass": row[
                        "pair_provenance_and_solver_pass"
                    ],
                    "pre_signed_inradius_nm": row["pre_float32_epsilon0"][
                        "signed_inradius_nm"
                    ],
                    "pre_volume_m3": row["pre_float32_epsilon0"]["volume_m3"],
                    "pre_positive": row["pre_float32_epsilon0"]["positive_volume"],
                    "post_signed_inradius_nm": row["post_float32_epsilon0"][
                        "signed_inradius_nm"
                    ],
                    "post_volume_m3": row["post_float32_epsilon0"]["volume_m3"],
                    "post_positive": row["post_float32_epsilon0"]["positive_volume"],
                    "frozen_5nm_volume_m3": row["frozen_d388_5nm_procedure"][
                        "maximum_intersection_volume_m3"
                    ],
                    "frozen_5nm_positive": row["frozen_d388_5nm_procedure"][
                        "positive_volume"
                    ],
                    "pre_left_crossing_nm": (
                        row["fan_seam_crossing"]["pre_float32"][
                            "left_max_crossing_to_right_nm"
                        ]
                        if row["adjacent"]
                        else None
                    ),
                    "pre_right_crossing_nm": (
                        row["fan_seam_crossing"]["pre_float32"][
                            "right_max_crossing_to_left_nm"
                        ]
                        if row["adjacent"]
                        else None
                    ),
                    "post_left_crossing_nm": (
                        row["fan_seam_crossing"]["post_float32"][
                            "left_max_crossing_to_right_nm"
                        ]
                        if row["adjacent"]
                        else None
                    ),
                    "post_right_crossing_nm": (
                        row["fan_seam_crossing"]["post_float32"][
                            "right_max_crossing_to_left_nm"
                        ]
                        if row["adjacent"]
                        else None
                    ),
                }
            )
    with PATH_CSV.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "global_rank",
                "maximum_vertex_count",
                "child_count",
                "cut_states",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(ranked, 1):
            writer.writerow(
                {
                    "global_rank": index,
                    "maximum_vertex_count": row["maximum_vertex_count"],
                    "child_count": row["child_count"],
                    "cut_states": json.dumps(list(row["cuts"])),
                }
            )


def _render_board(
    tie: dict[str, Any], seam: dict[str, Any], seams: list[dict[str, Any]]
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype(str(FONT), 25)
    small = ImageFont.truetype(str(FONT), 21)
    bold = ImageFont.truetype(str(FONT_BOLD), 34)
    title = ImageFont.truetype(str(FONT_BOLD), 45)
    draw.text((55, 34), "D389 수치 계보 · 전역 경로 동률 감사", font=title, fill="#111827")
    draw.text(
        (58, 94),
        "D388 결과는 동결 — 새 경로를 형상에 적용하지 않고, 왜 판정이 갈렸는지만 검사",
        font=regular,
        fill="#374151",
    )
    panels = [(45, 145, 925, 545), (955, 145, 1875, 545), (45, 575, 1875, 1025)]
    for box in panels:
        draw.rounded_rectangle(box, radius=18, outline="#cbd5e1", width=3, fill="#f8fafc")
    draw.text((75, 170), "1. 아래쪽 층 B35 경로 동률", font=bold, fill="#0f172a")
    g = tie["global_canonical"]["cuts"]
    l = tie["d388_local_dp_result"]["cuts"]
    draw.text((80, 235), f"전역 기준 1위: {list(g)}", font=regular, fill="#166534")
    draw.text((80, 285), f"D388 국소 DP: {l}", font=regular, fill="#b45309")
    draw.text(
        (80, 345),
        "state 5에서 [0,1,5] (당시 최대 16)이\n"
        "[0,2,5] (당시 최대 15)보다 먼저 버려졌습니다.",
        font=regular,
        fill="#1f2937",
        spacing=10,
    )
    draw.text(
        (80, 445),
        "뒤의 공통 10→14 간선이 35이므로 최종 최대값은 둘 다 35.\n"
        "따라서 전체 경로 사전식 기준에서는 [0,1,5,…]가 먼저입니다.",
        font=small,
        fill="#475569",
        spacing=8,
    )
    draw.text((985, 170), "2. 11개 인접 경계의 수치 출처", font=bold, fill="#0f172a")
    draw.text(
        (995, 238),
        f"Float64 구성에서 ε=0 양의 부피: {seam['adjacent_pre_float32_epsilon0_positive_count']}/11\n"
        f"Float32 등록 후 ε=0 양의 부피: {seam['adjacent_post_float32_epsilon0_positive_count']}/11\n"
        f"D388 동결 5nm 절차 양의 부피: {seam['adjacent_frozen_d388_5nm_positive_count']}/11",
        font=regular,
        fill="#1f2937",
        spacing=14,
    )
    draw.text((995, 395), seam["classification"], font=small, fill="#7c2d12")
    draw.text(
        (995, 470),
        "비인접 대조군 25쌍 · F32 왕복 exact 13/13 · 물리/파지 0",
        font=small,
        fill="#475569",
    )
    draw.text((75, 600), "3. 인접 경계별 수치 (nm, m³)", font=bold, fill="#0f172a")
    headers = [
        "층/쌍",
        "F64 ε0 signed",
        "F64 ε0 volume",
        "F32 ε0 signed",
        "F32 ε0 volume",
        "D388 5nm volume",
        "분류",
    ]
    xs = [80, 265, 535, 800, 1065, 1330, 1600]
    column_right = [255, 525, 790, 1055, 1320, 1590, 1850]
    measured_boxes: list[list[int]] = []
    for x, header in zip(xs, headers, strict=True):
        draw.text((x, 655), header, font=small, fill="#334155")
    y = 700
    for row in [item for item in seams if item["adjacent"]]:
        pre = row["pre_float32_epsilon0"]
        post = row["post_float32_epsilon0"]
        frozen = row["frozen_d388_5nm_procedure"]
        values = [
            f"{row['target']} {row['left_index']}-{row['right_index']}",
            f"{pre['signed_inradius_nm']:+.4g}",
            f"{pre['volume_m3']:.4e}",
            f"{post['signed_inradius_nm']:+.4g}",
            f"{post['volume_m3']:.4e}",
            f"{frozen['maximum_intersection_volume_m3']:.4e}",
            {
                "PRE_FLOAT32_D388_5NM_CONSTRUCTION_HAS_EPSILON0_POSITIVE_OVERLAP": "F64구성",
                "FLOAT32_REGISTRATION_INTRODUCES_EPSILON0_POSITIVE_OVERLAP": "F32등록",
                "FROZEN_5NM_PROCEDURE_ONLY_NO_EPSILON0_POSITIVE_OVERLAP": "5nm절차",
                "NO_POSITIVE_OVERLAP_IN_ANY_AUDITED_PROCEDURE": "없음",
                "MIXED_ADJACENT_PAIR_PROVENANCE": "혼합",
                "INDETERMINATE_PROVENANCE_OR_SOLVER_DISAGREEMENT": "미결정",
            }[row["per_pair_classification"]],
        ]
        for column, (x, value) in enumerate(zip(xs, values, strict=True)):
            draw.text((x, y), value, font=small, fill="#111827")
            bbox = list(draw.textbbox((x, y), value, font=small))
            bbox.append(column)
            measured_boxes.append(bbox)
        y += 27
    draw.text(
        (75, 1000),
        "권위: 원 JSON/CSV와 독립 Float64 계산 · 이 그림/Rerun은 육안검사용 · 예산/충돌체/파지 판정은 모두 미승인·null",
        font=small,
        fill="#475569",
    )
    image.save(BOARD)
    boxes_in_columns = all(
        box[0] >= xs[box[4]]
        and box[2] <= column_right[box[4]]
        and box[1] >= 690
        and box[3] <= 1000
        for box in measured_boxes
    )
    rows_do_not_overlap = all(
        measured_boxes[index][3] <= measured_boxes[index + len(headers)][1]
        for index in range(len(measured_boxes) - len(headers))
    )
    layout_checks = {
        "exact_canvas_1920x1080": image.size == (1920, 1080),
        "three_registered_panels_in_canvas": all(
            0 <= value <= bound
            for panel in panels
            for value, bound in zip(
                panel, (1920, 1080, 1920, 1080), strict=True
            )
        ),
        "all_11_rows_rendered": len(measured_boxes) == 11 * len(headers),
        "measured_table_text_within_columns": boxes_in_columns,
        "measured_table_rows_do_not_overlap": rows_do_not_overlap,
        "last_row_above_footer": y <= 1000,
    }
    layout = {
        "artifact": "D389_BOARD_LAYOUT_VALIDATION_V1",
        "path": _rel(BOARD),
        "width": 1920,
        "height": 1080,
        "panels": 3,
        "adjacent_rows_visible": 11,
        "measured_text_box_count": len(measured_boxes),
        "checks": layout_checks,
        "pass": all(layout_checks.values()),
    }
    _write_json_x(BOARD_LAYOUT, layout)
    return {
        "path": _rel(BOARD),
        "sha256": _sha(BOARD),
        "bytes": BOARD.stat().st_size,
        "exact_1920x1080": True,
    }


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        return {
            "path": _rel(path),
            "exists": True,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }


def _build_blueprint(summary_path: str) -> Any:
    import rerun.blueprint as rrb

    spatial = rrb.Spatial3DView(
        origin="/",
        contents="/d389/geometry/**",
        name="D389 selected D388 seam geometry",
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=(0.42, -0.48, 0.32),
            look_target=(0.075, 0.0, 0.075),
            eye_up=(0.0, 0.0, 1.0),
        ),
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=False,
            show_bounding_box=False,
        ),
    )
    decision = rrb.Vertical(
        spatial,
        rrb.TextLogView(
            origin=summary_path,
            contents=summary_path,
            name="D389 decision summary",
        ),
        row_shares=[0.68, 0.32],
    )
    notification = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d389/notification_buffer/**",
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
            column_shares=[0.80, 0.20],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _write_rerun(
    evidence: dict[str, Any],
    geometry: dict[str, Any],
    seams: list[dict[str, Any]],
) -> dict[str, Any]:
    """Save and strictly validate one inspection recording; never show live."""
    import roarm_rl.rerun_contract as rerun_contract
    import roarm_rl.viz_debug as viz_debug

    palette = [
        [0, 163, 163, 175],
        [246, 166, 35, 175],
        [30, 136, 229, 175],
        [229, 57, 53, 175],
        [126, 87, 194, 175],
        [67, 160, 71, 175],
        [109, 76, 65, 175],
    ]
    meshes: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    expected_entities = {"metadata/run", "decision/summary"}
    components: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "decision/summary": ["TextLog:level", "TextLog:text"],
    }
    display_transforms: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for target_index, layer in enumerate(geometry["layers"]):
        parent = np.asarray(layer["parent_vertices_f64_m"], dtype=np.float64)
        center = parent.mean(axis=0)
        row_offset = np.asarray([0.0, 0.0, 0.14 * (1 - target_index)])
        for column_index, representation in enumerate(
            ("pre_float32", "post_float32")
        ):
            offset = row_offset + np.asarray([0.15 * column_index, 0.0, 0.0])
            display_transforms[(layer["prim_name"], representation)] = (
                center,
                offset,
            )
            for child_index, child in enumerate(layer["children"]):
                key = (
                    "pre_float32_vertices_f64_m"
                    if representation == "pre_float32"
                    else "stored_post_float32_vertices_f64_m"
                )
                vertices = np.asarray(child[key], dtype=np.float64)
                triangles = np.asarray(
                    ConvexHull(vertices).simplices, dtype=np.int64
                )
                entity = (
                    f"d389/geometry/{target_index:02d}_"
                    f"{layer['prim_name']}/{representation}/"
                    f"child_{child_index:02d}"
                )
                meshes.append(
                    {
                        "entity_path": entity,
                        "coordinate_frame": "tf#/",
                        "vertices_m": vertices - center + offset,
                        "triangles": triangles,
                        "color_rgba": palette[child_index % len(palette)],
                        "static": True,
                        "representation": (
                            "reconstructed pre-Float32 D388-selected child"
                            if representation == "pre_float32"
                            else "immutable stored post-Float32 D388 child"
                        ),
                        "numeric_authority": (
                            "canonical D389 JSON; Rerun spatial copy is "
                            "inspection-only"
                        ),
                    }
                )
                metadata = f"metadata/meshes/{entity.replace('/', '__')}"
                expected_entities.update({entity, metadata})
                components[entity] = [
                    "CoordinateFrame:frame",
                    "Mesh3D:albedo_factor",
                    "Mesh3D:triangle_indices",
                    "Mesh3D:vertex_positions",
                ]
                components[metadata] = ["TextDocument:text"]
    for row in seams:
        if not row["adjacent"]:
            continue
        target_index = 0 if row["target"] == "UPPER" else 1
        for representation, source_key, color in (
            (
                "pre_float32",
                "pre_float32_epsilon0",
                [220, 38, 38, 255],
            ),
            (
                "post_float32",
                "post_float32_epsilon0",
                [37, 99, 235, 255],
            ),
        ):
            center_m = row[source_key]["chebyshev_center_m"]
            if center_m is None:
                continue
            center, offset = display_transforms[
                (row["prim_name"], representation)
            ]
            entity = (
                f"d389/geometry/{target_index:02d}_{row['prim_name']}/"
                f"{representation}/seams/"
                f"seam_{row['left_index']:02d}_{row['right_index']:02d}"
            )
            points.append(
                {
                    "entity_path": entity,
                    "positions_m": [
                        np.asarray(center_m, dtype=np.float64) - center + offset
                    ],
                    "radii": [0.0015],
                    "colors": [color],
                    "labels": [
                        (
                            f"{row['target']} {row['left_index']}-"
                            f"{row['right_index']} "
                            f"{row['per_pair_classification']}"
                        )
                    ],
                    "coordinate_frame": "tf#/",
                    "static": True,
                }
            )
            expected_entities.add(entity)
            components[entity] = [
                "CoordinateFrame:frame",
                "Points3D:colors",
                "Points3D:labels",
                "Points3D:positions",
                "Points3D:radii",
            ]
    tie = evidence["canonical_tie_audit"]
    numeric = evidence["seam_numeric_provenance_audit"]
    summary_text = "\n".join(
        [
            "D389 OFFLINE NUMERIC PROVENANCE + CANONICAL TIE AUDIT",
            f"GLOBAL B35: {tie['global_canonical']['cuts']}",
            (
                "D388 LOCAL DP: "
                f"{tie['d388_local_dp_result']['cuts']} "
                "(not substituted into geometry)"
            ),
            "FIRST PRUNING: state=5, [0,1,5] max16 vs [0,2,5] max15",
            (
                "ADJACENT POSITIVE COUNTS PRE/F32/FROZEN5NM: "
                f"{numeric['adjacent_pre_float32_epsilon0_positive_count']}/"
                f"{numeric['adjacent_post_float32_epsilon0_positive_count']}/"
                f"{numeric['adjacent_frozen_d388_5nm_positive_count']}"
            ),
            f"CLASSIFICATION: {numeric['classification']}",
            "NONADJACENT CONTROLS: 25; D388 VERDICT REMAINS FAIL_STOP",
            "BUDGET/PARTITION/ASSET/ISAAC/PHYSICS/Q5/CONTACT/GRASP: 0",
            "SELECTED/ADOPTED BUDGET: NULL/NULL; g0a_pass=false",
        ]
    )
    original_builder = viz_debug.build_rerun_blueprint
    original_runner = rerun_contract._run

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d389_numeric_tie":
            return _build_blueprint("decision/summary")
        return original_builder(mode)

    viewer_call_count = 0

    def no_signal_runner(
        command: list[str], *, timeout_s: float
    ) -> dict[str, Any]:
        nonlocal viewer_call_count
        del timeout_s
        if any("screenshot" in str(part) for part in command):
            viewer_call_count += 1
            if viewer_call_count > 1:
                return {
                    "command": command,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "D389 viewer maximum one exceeded",
                    "ok": False,
                }
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            return {
                "command": command,
                "returncode": int(result.returncode),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "ok": result.returncode == 0,
                "d389_timeout_parameter_ignored_no_signal": True,
            }
        except Exception as exc:
            return {
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": repr(exc),
                "ok": False,
                "d389_timeout_parameter_ignored_no_signal": True,
            }

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    rerun_contract._run = no_signal_runner
    try:
        saved = viz_debug.log_rerun(
            RRD,
            meshes=meshes,
            points=points,
            events=[
                {
                    "entity_path": "decision/summary",
                    "text": summary_text,
                    "level": "INFO",
                    "static": True,
                }
            ],
            recording_metadata={
                "case": CASE,
                "attempt": ATTEMPT,
                "verdict": evidence["verdict"],
                "global_canonical_lower_b35": tie["global_canonical"]["cuts"],
                "d388_selected_lower_geometry_cuts": EXPECTED_LOCAL_CUTS,
                "d388_verdict_frozen": True,
                "selected_vertex_budget": None,
                "adopted_vertex_budget": None,
                "physics_or_grasp_result": None,
                "g0a_pass": False,
                "spatial_numeric_authority": (
                    "inspection only; canonical JSON and CSV decide"
                ),
            },
            recording_id="g0a_d389_numeric_provenance_and_tie_audit",
            blueprint_path=RBL,
            blueprint_mode="d389_numeric_tie",
            live_viewer=False,
            app_id="roarm_g0a_d389_numeric_tie_audit",
        )
        if not saved.get("ok"):
            raise RuntimeError(f"D389 save-only Rerun failed: {saved}")
        validation = rerun_contract.validate_rerun_artifact(
            RRD,
            expected_entity_paths=sorted(expected_entities),
            exact_entity_paths=sorted(expected_entities),
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
    validation["d389_no_signal_and_viewer_contract"] = {
        "subprocess_timeout_seconds": None,
        "timeout_kill_path_present": False,
        "process_signals_sent": 0,
        "headless_viewer_invocations": viewer_call_count,
        "viewer_maximum": 1,
        "viewer_retry": 0,
    }
    _write_json_x(RERUN_VALIDATION, validation)
    screenshot = (
        _png_info(RERUN_SCREENSHOT)
        if RERUN_SCREENSHOT.is_file()
        else {"path": _rel(RERUN_SCREENSHOT), "exists": False}
    )
    dimension_pass = (
        (screenshot.get("width"), screenshot.get("height"))
        in {(1920, 1080), (3840, 2160)}
        if screenshot.get("exists")
        else False
    )
    return {
        "save_only": saved,
        "strict_validation_pass": validation.get("pass") is True,
        "headless_viewer_invocations": viewer_call_count,
        "viewer_maximum_one_no_retry": viewer_call_count <= 1,
        "screenshot_dimension_contract_pass": dimension_pass,
        "rrd": {
            "path": _rel(RRD),
            "sha256": _sha(RRD),
            "bytes": RRD.stat().st_size,
        },
        "rbl": {
            "path": _rel(RBL),
            "sha256": _sha(RBL),
            "bytes": RBL.stat().st_size,
        },
        "validation": {
            "path": _rel(RERUN_VALIDATION),
            "sha256": _sha(RERUN_VALIDATION),
        },
        "screenshot": screenshot,
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tie, ranked = _tie_audit()
    _deadline("compute_after_tie_audit")
    seam, geometry, seams = _seam_audit()
    _deadline("compute_after_seam_audit")
    checks = {
        "input_hashes_exact": _inputs() == EXPECTED_SHA256,
        "tie_audit_pass": tie["pass"],
        "seam_numeric_provenance_pass": seam["pass"],
        "scope_counters_zero": all(value == 0 for value in SCOPE_COUNTERS.values()),
    }
    evidence = {
        "artifact": "D389_NUMERIC_AND_CANONICAL_TIE_AUDIT_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Localize D388's lower B35 canonical-path disagreement and the "
            "numeric provenance of its eleven positive adjacent seam volumes."
        ),
        "new_variables": NEW_VARIABLES,
        "input_hashes": _inputs(),
        "immutable_d388_verdict": EXPECTED_D388_VERDICT,
        "canonical_tie_audit": tie,
        "seam_numeric_provenance_audit": seam,
        "checks": checks,
        "audit_contract_pass": all(checks.values()),
        "d388_retroactive_pass": False,
        "d388_verdict_modified": False,
        "selected_vertex_budget": None,
        "adopted_vertex_budget": None,
        "selected_budget_application_count": 0,
        "partition_changed": False,
        "tolerance_or_gate_changed": False,
        "materializable_candidate": False,
        "live_identity_pass": None,
        "physics_or_grasp_result": None,
        "g0a_pass": False,
        "scope_counters": SCOPE_COUNTERS,
        "verdict": (
            "D389_NUMERIC_PROVENANCE_AND_GLOBAL_CANONICAL_TIE_AUDIT_PASS_"
            "NO_REPAIR_ADOPTION"
            if all(checks.values())
            else "D389_AUDIT_CONTRACT_FAIL_STOP"
        ),
    }
    return evidence, geometry, seams, ranked


def _phase_contract() -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in PHASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = [
        "prepare_start",
        "prepare_end",
        "supervisor_before_worker",
        "worker_start",
        "canonical_numeric_evidence_committed",
        "worker_end",
        "supervisor_after_worker",
        "finalize_start",
        "finalize_end",
    ]
    observed = [row["phase"] for row in records]
    monotonic = [row["monotonic_seconds"] for row in records]
    checks = {
        "exact_phase_sequence": observed == expected,
        "each_phase_exactly_once": all(
            observed.count(name) == 1 for name in expected
        ),
        "monotonic_seconds_nondecreasing": all(
            left <= right
            for left, right in zip(
                monotonic[:-1], monotonic[1:], strict=True
            )
        ),
    }
    return {
        "expected": expected,
        "observed": observed,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _prepare() -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"refusing forward-only reuse: {OUT_DIR}")
    status_before = _git(
        "status", "--porcelain=v1", "--untracked-files=all"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    start = START.read_text(encoding="utf-8")
    imports = _direct_import_roots()
    forbidden = sorted(
        set(imports)
        & {"carb", "isaaclab", "isaacsim", "omni", "pxr", "signal", "torch", "warp"}
    )
    d388 = _read_json(D388_EVIDENCE)
    attempt1_continuity = _attempt1_control_continuity()
    expected_status_before = (
        " M START_HERE.md",
        (
            "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
            "attempt1_d388_overlap_gate_numeric_provenance_and_canonical_"
            "tie_audit/d389_phase_markers.jsonl"
        ),
        (
            "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
            "attempt1_d388_overlap_gate_numeric_provenance_and_canonical_"
            "tie_audit/d389_preregistration.json"
        ),
        (
            "?? sim_scripts/"
            "cyl34_top_view_d389_attempt2_prereg_status_whitespace_repair.py"
        ),
        (
            "?? sim_scripts/"
            "cyl34_top_view_d389_d388_overlap_gate_numeric_provenance_and_"
            "canonical_tie_audit.py"
        ),
    )
    expected_status_after = (
        *expected_status_before[:3],
        (
            "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
            "attempt2_prereg_status_whitespace_repair/"
            "d389_phase_markers.jsonl"
        ),
        *expected_status_before[3:],
    )
    status_after = _git(
        "status", "--porcelain=v1", "--untracked-files=all"
    )
    installed = {
        "numpy": np.__version__,
        "psutil": importlib.metadata.version("psutil"),
        "rerun_sdk": importlib.metadata.version("rerun-sdk"),
    }
    checks = {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "input_hashes_exact": _inputs() == EXPECTED_SHA256,
        "start_here_preregistered_sha256_exact": (
            _sha(START) == EXPECTED_START_SHA256
        ),
        "git_status_before_exact_allowed": (
            tuple(status_before.splitlines()) == expected_status_before
        ),
        "git_status_after_exact_allowed": (
            tuple(status_after.splitlines()) == expected_status_after
        ),
        "attempt1_control_lineage_exact_and_worker0": (
            attempt1_continuity["pass"] is True
            and attempt1_continuity["actual_worker_invocations"] == 0
            and attempt1_continuity["retries"] == 0
            and attempt1_continuity["headless_viewer_invocations"] == 0
        ),
        "case_aggregate_worker_maximum_one_registered": (
            attempt1_continuity["actual_worker_invocations"] == 0
        ),
        "installed_compatibility_pins_exact": installed
        == {
            "numpy": "1.26.0",
            "psutil": "5.9.8",
            "rerun_sdk": "0.34.1",
        },
        "d388_verdict_exact_and_frozen": d388["verdict"] == EXPECTED_D388_VERDICT,
        "active_case_registered": (
            "`D389 [d388_overlap_gate_numeric_provenance_and_canonical_tie_audit]`"
            in start
        ),
        "two_variables_exact_and_registered": (
            len(NEW_VARIABLES) == 2
            and all(variable in start for variable in NEW_VARIABLES)
        ),
        "forward_only_path_registered": _rel(OUT_DIR) in start,
        "direct_forbidden_imports_zero": forbidden == [],
        "only_registered_d388_artifacts_are_read": (
            set(_inputs()) == {"d388_evidence", "d388_geometry", "d388_csv"}
        ),
        "attempt1_reads_are_control_lineage_not_scientific_inputs": (
            set(_inputs()) == {"d388_evidence", "d388_geometry", "d388_csv"}
            and attempt1_continuity["artifact"]
            == "D389_ATTEMPT1_CONTROL_CONTINUITY_V1"
        ),
        "worker_one_retry_zero_registered": (
            "actual worker `1`, retry `0`" in start
        ),
        "cooperative_deadline_no_signal": DEADLINE_SECONDS == 300.0
        and "signal" not in imports,
    }
    prereg = {
        "artifact": "D389_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "script": _rel(SCRIPT),
        "script_sha256": _sha(SCRIPT),
        "input_hashes": _inputs(),
        "installed_stack": installed,
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "status_before_output_create": status_before,
            "status_after_output_create": status_after,
        },
        "prior_attempt_control_continuity": attempt1_continuity,
        "new_variables": NEW_VARIABLES,
        "frozen_constants": {
            "d388_selected_upper_cuts": TARGETS[0]["cuts"],
            "d388_selected_lower_dp_cuts": TARGETS[1]["cuts"],
            "float_construction_epsilon_m": FLOAT_EPS_M,
            "positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
            "audit_budget_only_not_adopted": AUDITED_BUDGET,
        },
        "registered_outcomes": [
            "pre-Float32 epsilon0 positive overlap",
            "post-Float32-only epsilon0 positive overlap",
            "frozen-5nm-procedure-only positive overlap",
            "calculation/provenance fail-stop",
        ],
        "execution": {
            "prior_actual_workers": 0,
            "planned_current_actual_workers": 1,
            "case_aggregate_actual_worker_maximum": 1,
            "prior_retries": 0,
            "planned_current_retry_index": 0,
            "case_aggregate_retries": 0,
            "prior_headless_viewer_invocations": 0,
            "current_viewer_maximum": 1,
            "case_aggregate_viewer_maximum": 1,
            "cooperative_deadline_seconds": DEADLINE_SECONDS,
            "process_signal_authority": False,
            "viewer_retry": 0,
        },
        "nonclaims": {
            "d388_retroactive_pass": False,
            "budget_selection_or_application": False,
            "partition_or_gate_change": False,
            "asset_or_physics_or_grasp": False,
        },
        "direct_import_roots": imports,
        "forbidden_direct_import_roots": forbidden,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _phase("prepare_end", pass_value=prereg["pass"])
    _write_json_x(PREREG, prereg)
    if not prereg["pass"]:
        raise RuntimeError(f"D389 preregistration failed: {checks}")
    print(json.dumps({"prepare_pass": True, "prereg": _rel(PREREG)}))
    return 0


def _worker_inner() -> int:
    global _deadline_monotonic
    prereg = _read_json(PREREG)
    if prereg["pass"] is not True:
        raise RuntimeError("D389 preregistration is not PASS")
    if prereg["script_sha256"] != _sha(SCRIPT) or _inputs() != EXPECTED_SHA256:
        raise RuntimeError("script or immutable input changed after prepare")
    attempt1_continuity = _attempt1_control_continuity()
    if attempt1_continuity["pass"] is not True:
        raise RuntimeError(
            f"D389 attempt1 control continuity changed: {attempt1_continuity}"
        )
    forbidden_existing = [
        path
        for path in (
            EVIDENCE,
            GEOMETRY,
            SEAM_CSV,
            PATH_CSV,
            BOARD,
            WORKER_CLAIM,
            FAILURE_ATTESTATION,
        )
        if path.exists()
    ]
    if forbidden_existing:
        raise RuntimeError(f"worker artifact reuse: {forbidden_existing}")
    started = time.monotonic()
    _deadline_monotonic = started + DEADLINE_SECONDS
    _phase("worker_start", signal_authority=False)
    evidence, geometry, seams, ranked = _compute()
    _deadline("worker_before_canonical_evidence_write")
    algorithm_elapsed = time.monotonic() - started
    if algorithm_elapsed > DEADLINE_SECONDS:
        raise CooperativeDeadlineExceeded(
            "worker_algorithm_elapsed_contract"
        )
    evidence["script_sha256"] = _sha(SCRIPT)
    evidence["execution"] = {
        "prior_attempt_actual_worker_invocations": 0,
        "worker_invocation_index": 1,
        "case_aggregate_actual_worker_invocations": 1,
        "retry_index": 0,
        "case_aggregate_retries": 0,
        "offline_only": True,
        "cooperative_deadline_seconds": DEADLINE_SECONDS,
        "algorithm_elapsed_seconds": algorithm_elapsed,
        "deadline_exceeded": (
            _deadline_monotonic is not None
            and time.monotonic() > _deadline_monotonic
        ),
        "process_signals_sent": 0,
    }
    _write_json_x(EVIDENCE, evidence)
    _phase("canonical_numeric_evidence_committed", verdict=evidence["verdict"])
    _write_json_x(GEOMETRY, geometry)
    _write_csvs(seams, ranked)
    board = _render_board(
        evidence["canonical_tie_audit"],
        evidence["seam_numeric_provenance_audit"],
        seams,
    )
    rerun = _write_rerun(evidence, geometry, seams)
    manual_template = {
        "artifact": "D389_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "required_checks": [
            "board_exact_1920x1080_and_readable",
            "global_vs_local_paths_and_state5_pruning_visible",
            "all_11_adjacent_seam_rows_readable",
            "pre_post_float32_and_frozen5nm_columns_distinct",
            "rerun_decision_subject_visible_and_readable",
            "no_budget_partition_asset_physics_or_grasp_claim",
        ],
        "board_path": _rel(BOARD),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "inspection_result_path": _rel(MANUAL),
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)
    claim = {
        "artifact": "D389_OFFLINE_WORKER_CLAIM_V1",
        "prior_attempt_actual_worker_invocations": 0,
        "worker_invocation_index": 1,
        "case_aggregate_actual_worker_invocations": 1,
        "retry_index": 0,
        "case_aggregate_retries": 0,
        "prior_attempt_control_continuity_pass": (
            attempt1_continuity["pass"] is True
        ),
        "numeric_audit_verdict": evidence["verdict"],
        "artifacts": {
            "evidence": {"path": _rel(EVIDENCE), "sha256": _sha(EVIDENCE)},
            "geometry": {"path": _rel(GEOMETRY), "sha256": _sha(GEOMETRY)},
            "seam_csv": {"path": _rel(SEAM_CSV), "sha256": _sha(SEAM_CSV)},
            "path_csv": {"path": _rel(PATH_CSV), "sha256": _sha(PATH_CSV)},
            "board": board,
            "board_layout": {
                "path": _rel(BOARD_LAYOUT),
                "sha256": _sha(BOARD_LAYOUT),
            },
            "rerun": rerun,
            "manual_template": {
                "path": _rel(MANUAL_TEMPLATE),
                "sha256": _sha(MANUAL_TEMPLATE),
            },
        },
        "scope_counters": SCOPE_COUNTERS,
        "pass": bool(
            evidence["audit_contract_pass"]
            and evidence["execution"]["deadline_exceeded"] is False
            and evidence["execution"]["algorithm_elapsed_seconds"]
            <= DEADLINE_SECONDS
            and _read_json(BOARD_LAYOUT)["pass"]
            and rerun["strict_validation_pass"]
            and rerun["viewer_maximum_one_no_retry"]
            and rerun["screenshot_dimension_contract_pass"]
            and attempt1_continuity["pass"] is True
            and all(value == 0 for value in SCOPE_COUNTERS.values())
        ),
    }
    _phase("worker_end", worker_claim_pass=claim["pass"])
    _write_json_x(WORKER_CLAIM, claim)
    if not claim["pass"]:
        raise RuntimeError("D389 worker claim failed")
    print(
        json.dumps(
            {
                "worker_pass": True,
                "audit_verdict": evidence["verdict"],
                "evidence": _rel(EVIDENCE),
            }
        )
    )
    return 0


def _worker() -> int:
    started = time.monotonic()
    try:
        return _worker_inner()
    except Exception as exc:
        elapsed = time.monotonic() - started
        failure = {
            "artifact": "D389_FAILURE_ATTESTATION_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "stage": "worker",
            "prior_attempt_actual_worker_invocations": 0,
            "worker_invocation_index": 1,
            "case_aggregate_actual_worker_invocations": 1,
            "retry_index": 0,
            "case_aggregate_retries": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": elapsed,
            "cooperative_deadline_seconds": DEADLINE_SECONDS,
            "deadline_exceeded": bool(
                _deadline_monotonic is not None
                and time.monotonic() > _deadline_monotonic
            ),
            "script": _safe_hash_record(SCRIPT),
            "input_snapshot": _safe_input_snapshot(),
            "scope_counters": SCOPE_COUNTERS,
            "process_signals_sent": 0,
            "d388_verdict_modified": False,
            "selected_or_adopted_budget": None,
            "materializable_candidate": False,
            "physics_or_grasp_result": None,
            "g0a_pass": False,
            "pass": False,
        }
        if not FAILURE_ATTESTATION.exists():
            _write_json_x(FAILURE_ATTESTATION, failure)
        if not WORKER_CLAIM.exists():
            _write_json_x(
                WORKER_CLAIM,
                {
                    "artifact": "D389_OFFLINE_WORKER_CLAIM_V1",
                    "prior_attempt_actual_worker_invocations": 0,
                    "worker_invocation_index": 1,
                    "case_aggregate_actual_worker_invocations": 1,
                    "retry_index": 0,
                    "case_aggregate_retries": 0,
                    "failure_attestation": {
                        "path": _rel(FAILURE_ATTESTATION),
                        "sha256": _sha(FAILURE_ATTESTATION),
                    },
                    "scope_counters": SCOPE_COUNTERS,
                    "pass": False,
                },
            )
        _phase(
            "worker_fail_stop",
            error_type=type(exc).__name__,
            deadline_exceeded=failure["deadline_exceeded"],
        )
        raise


def _run() -> int:
    if not PREREG.is_file():
        raise RuntimeError("prepare must precede run")
    if INVOCATION.exists() or SUPERVISOR.exists():
        raise RuntimeError("refusing a second D389 worker")
    command = [sys.executable, "-B", str(SCRIPT), "--stage", "worker"]
    _write_json_x(
        INVOCATION,
        {
            "artifact": "D389_OFFLINE_AUDIT_INVOCATION_V1",
            "command": command,
            "cwd": str(REPO),
            "prior_attempt_actual_worker_invocations": 0,
            "worker_invocation_index": 1,
            "case_aggregate_actual_worker_invocations_if_spawned": 1,
            "retry_index": 0,
            "case_aggregate_retries": 0,
            "cooperative_deadline_seconds": DEADLINE_SECONDS,
            "supervisor_signal_authority": False,
        },
    )
    _phase("supervisor_before_worker")
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    returncode: int | None = None
    supervisor_error: str | None = None
    try:
        with STDOUT.open("x", encoding="utf-8") as stdout:
            with STDERR.open("x", encoding="utf-8") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    start_new_session=False,
                )
                returncode = process.wait()
    except Exception as exc:
        supervisor_error = f"{type(exc).__name__}: {exc}"
    worker_claim_pass = False
    worker_claim_error: str | None = None
    if WORKER_CLAIM.is_file():
        try:
            worker_claim_pass = (
                _read_json(WORKER_CLAIM).get("pass") is True
            )
        except Exception as exc:
            worker_claim_error = f"{type(exc).__name__}: {exc}"
    record = {
        "artifact": "D389_OFFLINE_WORKER_SUPERVISOR_V1",
        "prior_attempt_actual_worker_invocations": 0,
        "actual_worker_invocations": int(process is not None),
        "case_aggregate_actual_worker_invocations": int(process is not None),
        "retries": 0,
        "case_aggregate_retries": 0,
        "worker_pid": process.pid if process is not None else None,
        "returncode": returncode,
        "supervisor_error": supervisor_error,
        "elapsed_seconds": time.monotonic() - started,
        "cooperative_deadline_seconds": DEADLINE_SECONDS,
        "supervisor_signal_authority": False,
        "process_signals_sent": 0,
        "termination_action": None,
        "worker_process_exited": (
            process.poll() is not None if process is not None else False
        ),
        "stdout": _rel(STDOUT),
        "stderr": _rel(STDERR),
        "worker_claim_exists": WORKER_CLAIM.is_file(),
        "worker_claim_pass": worker_claim_pass,
        "worker_claim_read_error": worker_claim_error,
        "pass": (
            process is not None
            and supervisor_error is None
            and returncode == 0
            and WORKER_CLAIM.is_file()
            and worker_claim_pass
            and worker_claim_error is None
        ),
    }
    _write_json_x(SUPERVISOR, record)
    _phase("supervisor_after_worker", returncode=returncode, pass_value=record["pass"])
    if not record["pass"]:
        raise RuntimeError(f"D389 worker failed: {record}")
    return 0


def _finalize() -> int:
    required = [
        PREREG,
        PHASES,
        INVOCATION,
        SUPERVISOR,
        STDOUT,
        STDERR,
        EVIDENCE,
        GEOMETRY,
        SEAM_CSV,
        PATH_CSV,
        BOARD,
        BOARD_LAYOUT,
        RRD,
        RBL,
        RERUN_VALIDATION,
        RERUN_SCREENSHOT,
        MANUAL_TEMPLATE,
        MANUAL,
        WORKER_CLAIM,
    ]
    missing = [_rel(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot finalize; missing: {missing}")
    _phase("finalize_start")
    evidence = _read_json(EVIDENCE)
    supervisor = _read_json(SUPERVISOR)
    worker = _read_json(WORKER_CLAIM)
    validation = _read_json(RERUN_VALIDATION)
    manual = _read_json(MANUAL)
    attempt1_continuity = _attempt1_control_continuity()
    current_viewer_invocations = validation.get(
        "d389_no_signal_and_viewer_contract", {}
    ).get("headless_viewer_invocations")
    case_aggregate = {
        "actual_worker_invocations": (
            attempt1_continuity.get("actual_worker_invocations", -1)
            + supervisor["actual_worker_invocations"]
        ),
        "retries": (
            attempt1_continuity.get("retries", -1)
            + supervisor["retries"]
        ),
        "headless_viewer_invocations": (
            attempt1_continuity.get("headless_viewer_invocations", -1)
            + (
                current_viewer_invocations
                if isinstance(current_viewer_invocations, int)
                else -1
            )
        ),
    }
    required_manual = set(_read_json(MANUAL_TEMPLATE)["required_checks"])
    manual_pass = (
        set(manual.get("checks", {})) == required_manual
        and all(value is True for value in manual["checks"].values())
        and manual.get("pass") is True
        and len(manual.get("observations", [])) >= 3
        and manual.get("artifact_hashes")
        == {
            _rel(BOARD): _sha(BOARD),
            _rel(RERUN_SCREENSHOT): _sha(RERUN_SCREENSHOT),
        }
    )
    claim_artifacts = worker["artifacts"]
    bindings = {
        "evidence": (claim_artifacts["evidence"], EVIDENCE),
        "geometry": (claim_artifacts["geometry"], GEOMETRY),
        "seam_csv": (claim_artifacts["seam_csv"], SEAM_CSV),
        "path_csv": (claim_artifacts["path_csv"], PATH_CSV),
        "board": (claim_artifacts["board"], BOARD),
        "board_layout": (claim_artifacts["board_layout"], BOARD_LAYOUT),
        "rrd": (claim_artifacts["rerun"]["rrd"], RRD),
        "rbl": (claim_artifacts["rerun"]["rbl"], RBL),
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
    linkage = {
        label: {
            "path_exact": record.get("path") == _rel(path),
            "hash_exact": record.get("sha256") == _sha(path),
            "claimed_path": record.get("path"),
            "expected_path": _rel(path),
            "claimed_sha256": record.get("sha256"),
            "current_sha256": _sha(path),
        }
        for label, (record, path) in bindings.items()
    }
    linkage_pass = all(
        row["path_exact"] and row["hash_exact"] for row in linkage.values()
    )
    checks = {
        "audit_contract_pass": evidence["audit_contract_pass"] is True,
        "worker_once_retry_zero": supervisor["actual_worker_invocations"] == 1
        and supervisor["retries"] == 0,
        "attempt1_continuity_exact_and_worker_viewer_zero": (
            attempt1_continuity["pass"] is True
            and attempt1_continuity["actual_worker_invocations"] == 0
            and attempt1_continuity["retries"] == 0
            and attempt1_continuity["headless_viewer_invocations"] == 0
        ),
        "case_aggregate_worker_one_retry_zero_viewer_maximum_one": (
            case_aggregate["actual_worker_invocations"] == 1
            and case_aggregate["retries"] == 0
            and 0 <= case_aggregate["headless_viewer_invocations"] <= 1
        ),
        "supervisor_and_worker_claim_pass": supervisor["pass"] is True
        and worker["pass"] is True,
        "strict_rerun_validation_pass": validation.get("pass") is True,
        "worker_claim_artifact_paths_and_hashes_exact": linkage_pass,
        "manual_visual_inspection_pass": manual_pass,
        "d388_frozen_and_no_retroactive_pass": (
            evidence["immutable_d388_verdict"] == EXPECTED_D388_VERDICT
            and evidence["d388_retroactive_pass"] is False
            and evidence["d388_verdict_modified"] is False
        ),
        "budgets_null_no_application": evidence["selected_vertex_budget"] is None
        and evidence["adopted_vertex_budget"] is None
        and evidence["selected_budget_application_count"] == 0,
        "no_materializable_physics_or_grasp_claim": (
            evidence["materializable_candidate"] is False
            and evidence["physics_or_grasp_result"] is None
            and evidence["g0a_pass"] is False
        ),
        "scope_counters_zero": all(
            value == 0 for value in evidence["scope_counters"].values()
        ),
    }
    _phase("finalize_end", pass_value=all(checks.values()))
    phase_contract = _phase_contract()
    checks["global_phase_contract_pass"] = phase_contract["pass"]
    completion = {
        "artifact": "D389_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "numeric_audit_verdict": evidence["verdict"],
        "checks": checks,
        "global_phase_contract": phase_contract,
        "prior_attempt_control_continuity": attempt1_continuity,
        "case_aggregate_execution": case_aggregate,
        "worker_claim_artifact_linkage": linkage,
        "artifact_hashes": {_rel(path): _sha(path) for path in required},
        "collider_or_budget_adopted": False,
        "physics_or_grasp_result": None,
        "g0a_pass": False,
        "pass": all(checks.values()),
    }
    _write_json_x(COMPLETION, completion)
    if not completion["pass"]:
        raise RuntimeError(f"D389 completion failed: {checks}")
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("prepare", "run", "worker", "finalize"), required=True
    )
    stage = parser.parse_args().stage
    if stage == "prepare":
        return _prepare()
    if stage == "run":
        return _run()
    if stage == "worker":
        return _worker()
    return _finalize()


if __name__ == "__main__":
    raise SystemExit(main())
