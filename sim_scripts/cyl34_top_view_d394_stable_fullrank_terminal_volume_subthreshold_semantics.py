#!/usr/bin/env python3
"""D394: exact-volume semantics for the ten stable FULL D390 terminals.

This is an offline-only, forward-only case.  It binds the immutable D389-D393
artifacts, reads exactly the ten D392 calls classified stable
FULL_DIMENSIONAL, and asks whether their binary64 point clouds have positive
exact-dyadic volume while remaining provably below the frozen D389 volume and
strict-interior gates.

No NVIDIA runtime, asset, collider, USD, cylinder, physics, q5, contact, grasp,
target, IK, or path is imported or executed.  Rerun receives centered and
magnified inspection copies only; canonical JSON/Fraction arithmetic is the
numeric authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASE = "D394"
ATTEMPT = "attempt1_stable_fullrank_terminal_volume_subthreshold_semantics"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d394" / ATTEMPT
START = REPO / "START_HERE.md"
EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
EXPECTED_START_SHA256 = (
    "4a6229369b7771131b66da9d5b1a79f20f9023bbeb167e0440dfbebdcff1fc00"
)

D389_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d389"
    / "attempt2_prereg_status_whitespace_repair"
)
D390_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d390"
    / "attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization"
)
D391_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d391"
    / "attempt1_d390_rank_basis_and_clip_input_immutability_repair"
)
D392_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d392"
    / "attempt1_d391_remaining35_same_authority_coverage_audit"
)
D393_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d393"
    / "attempt1_call29_third_direction_provenance_localization"
)

D389_PREREG = D389_DIR / "d389_preregistration.json"
D389_EVIDENCE = D389_DIR / "d389_numeric_and_tie_audit_evidence.json"
D389_GEOMETRY = D389_DIR / "d389_reconstructed_seam_witness_geometry.json"
D389_CSV = D389_DIR / "d389_seam_numeric_provenance.csv"
D390_EVIDENCE = D390_DIR / "d390_boundary_collapse_localization_evidence.json"
D390_GEOMETRY = D390_DIR / "d390_terminal_candidate_geometry.json"
D390_CSV = D390_DIR / "d390_failed_directional_call_trace.csv"
D391_EVIDENCE = D391_DIR / "d391_rank_and_plane_immutability_evidence.json"
D392_EVIDENCE = D392_DIR / "d392_remaining35_rank_evidence.json"
D392_COMPLETION = D392_DIR / "d392_completion_summary.json"
D393_EVIDENCE = D393_DIR / "d393_call29_provenance_evidence.json"
D393_GEOMETRY = D393_DIR / "d393_call29_lineage_geometry.json"
D393_COMPLETION = D393_DIR / "d393_completion_summary.json"

EXPECTED_INPUT_SHA256 = {
    D389_PREREG: "f4b38c4c5db311412c5700f792a66be805bbe06abf9be89b533002e1860ce780",
    D389_EVIDENCE: "9423e870c0a218606781943abd2f5c48cb1e5d53cbbf9fb1212294b4ef5bb5dd",
    D389_GEOMETRY: "66042a93389cb8d0e6c867be87382566c753cd965ceda619e947e73de4a607be",
    D389_CSV: "1fdbaac1c756983c8bd2d2d8e8eabed36a4530393b5f3e3491678335d778f66f",
    D390_EVIDENCE: "3014610b7b2fd953740d239b91d9d9dce8aa917be67c6a1be15cf6ac052d9975",
    D390_GEOMETRY: "73fc986043b976bec26e1cc92643b8aab281a529f1c71c2918163ba7b98475c7",
    D390_CSV: "5a8295c85b62552459806c8a51a2ec485c20248365e8d1382857c19ebb527591",
    D391_EVIDENCE: "d76bbd88b2a6f188f9c46c382adc7292e56267e6e4a3beb8b9004b70e34b80cc",
    D392_EVIDENCE: "9ced175925c6c528d47bf94e5ae224e65bae1d4d6c88fc236952343de0c72102",
    D392_COMPLETION: "01046290ca840960290198a4136cc7ed818cfbeccc7b745139a9e419dd421f70",
    D393_EVIDENCE: "537537d89a2204987eebfa9bf668968801247e7cef70b7b694434b16b98883a9",
    D393_GEOMETRY: "c674a43f3eca216b94bd4cc328f2f93cffd2bc558df1d3a44442da40244c9bed",
    D393_COMPLETION: "cafac7a1f79b785592204300dc98c49ed1cd0ae0ba424d7091d641e7aa4cfab9",
}

FULL_CALL_INDICES = [1, 6, 13, 14, 16, 21, 23, 26, 34, 39]
FACE_CONTROL_INDEX = 2
LINE_CONTROL_INDEX = 31
CALL29_INDEX = 29
POSITIVE_VOLUME_EPS_M3 = Fraction(1, 10**18)
STRICT_INTERIOR_RADIUS_M = Fraction(1, 10**13)
VARIABLES = [
    "stable_fullrank_terminal_exact_dyadic_volume_sandwich_v1",
    "frozen_volume_gate_monotone_early_stop_semantics_v1",
]
EXPECTED_USER_APPROVAL_NORMALIZED = (
    "D390 나머지35 동결 D391 기준 적용, call29 미세 제3방향 국소화, "
    "D389 seam 반영, 이후 collider/USD/Isaac/PhysX/29x50mm 원통 물리를 "
    "각 선행 결과 동결 뒤 순차 진행 승인"
)
COOPERATIVE_DEADLINE_SECONDS = 300.0
EXECUTION_AUTHORITY_ENV = "D394_EXECUTION_AUTHORITY_SHA256"
WORKER_AUTHORIZATION_ENV = "D394_WORKER_AUTHORIZATION_SHA256"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

EXECUTION_AUTHORITY = OUT_DIR / "d394_execution_authority.json"
PREREGISTRATION = OUT_DIR / "d394_preregistration.json"
PHASES = OUT_DIR / "d394_phase_markers.jsonl"
INVOCATION = OUT_DIR / "d394_offline_worker_invocation.json"
AUTHORIZATION = OUT_DIR / "d394_worker_authorization.json"
SENTINEL = OUT_DIR / "d394_worker_start_sentinel.json"
STDOUT = OUT_DIR / "d394_offline_worker_stdout.log"
STDERR = OUT_DIR / "d394_offline_worker_stderr.log"
PROGRESS = OUT_DIR / "d394_full10_progress.jsonl"
EVIDENCE = OUT_DIR / "d394_full10_volume_semantics_evidence.json"
GEOMETRY = OUT_DIR / "d394_full10_display_geometry.json"
CSV_PATH = OUT_DIR / "d394_full10_volume_semantics.csv"
WORKER_CLAIM = OUT_DIR / "d394_offline_worker_claim.json"
SUPERVISOR = OUT_DIR / "d394_offline_worker_supervisor.json"
BOARD = OUT_DIR / "d394_full10_volume_semantics_1920x1080.png"
LAYOUT = OUT_DIR / "d394_board_layout_validation.json"
RRD = OUT_DIR / "d394_full10_volume_semantics.rerun.rrd"
RBL = OUT_DIR / "d394_full10_volume_semantics.rerun.rbl"
RERUN_VALIDATION = OUT_DIR / "d394_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d394_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d394_manual_visual_inspection_template.json"
OBSERVABILITY = OUT_DIR / "d394_observability_claim.json"
MANUAL = OUT_DIR / "d394_manual_visual_inspection.json"
FAILURE = OUT_DIR / "d394_failure_attestation.json"
COMPLETION = OUT_DIR / "d394_completion_summary.json"

PHASE_ORDER = [
    "prepare_start",
    "prepare_end",
    "supervisor_before_worker",
    "worker_start",
    "full10_manifest_bound",
    "full10_exact_sandwich_computed",
    "controls_computed",
    "canonical_numeric_evidence_committed",
    "worker_end",
    "supervisor_after_worker",
    "observability_start",
    "observability_end",
    "finalize_start",
    "finalize_end",
]
MANUAL_CHECK_KEYS = [
    "board_exact_1920x1080",
    "board_all_ten_calls_present",
    "board_positive_witness_and_upper_bounds_readable",
    "board_two_frozen_thresholds_distinguished",
    "board_monotone_subset_semantics_readable",
    "board_call29_and_seam_nonclaims_readable",
    "board_no_text_overlap_or_clipping",
    "rerun_all_ten_magnified_clouds_visible",
    "rerun_magnification_marked_inspection_only",
    "rerun_screenshot_matches_canonical_counts_and_verdict",
]


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _native(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
            "float": float(value),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _write_json_x(path: Path, value: Any) -> None:
    payload = json.dumps(
        _native(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, value: Any) -> None:
    payload = json.dumps(
        _native(value),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _array_sha(points: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(points, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"|float64|C|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\r\n")


def _phase(name: str, **details: Any) -> None:
    if name not in PHASE_ORDER:
        raise RuntimeError(f"unknown D394 phase: {name}")
    existing: list[str] = []
    if PHASES.is_file():
        with PHASES.open("r", encoding="utf-8") as stream:
            existing = [json.loads(line)["phase"] for line in stream if line.strip()]
    expected = PHASE_ORDER[len(existing)]
    if name != expected:
        raise RuntimeError(f"D394 phase order mismatch: expected {expected}, got {name}")
    _append_jsonl(
        PHASES,
        {
            "ordinal": len(existing),
            "phase": name,
            "wall_time_unix": time.time(),
            **details,
        },
    )


def _frac_point(point: Sequence[float]) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(Fraction.from_float(float(value)) for value in point)  # type: ignore[return-value]


def _sub(
    left: Sequence[Fraction], right: Sequence[Fraction]
) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(left[i] - right[i] for i in range(3))  # type: ignore[return-value]


def _cross(
    left: Sequence[Fraction], right: Sequence[Fraction]
) -> tuple[Fraction, Fraction, Fraction]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((left[i] * right[i] for i in range(3)), Fraction(0))


def _det3(rows: Sequence[Sequence[Fraction]]) -> Fraction:
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _tetra_det6(
    points: Sequence[Sequence[Fraction]], indices: Sequence[int]
) -> Fraction:
    anchor = points[indices[0]]
    return _det3([_sub(points[index], anchor) for index in indices[1:]])


def _max_tetra(
    points: Sequence[Sequence[Fraction]],
) -> tuple[tuple[int, int, int, int], Fraction]:
    best_indices: tuple[int, int, int, int] | None = None
    best = Fraction(-1)
    for raw in itertools.combinations(range(len(points)), 4):
        indices = tuple(raw)
        value = abs(_tetra_det6(points, indices)) / 6
        if value > best or (value == best and (best_indices is None or indices < best_indices)):
            best_indices = indices
            best = value
    if best_indices is None:
        raise RuntimeError("fewer than four points in max-tetra calculation")
    return best_indices, best


def _projected_hull(
    points: Sequence[Sequence[Fraction]],
    vertex_indices: Sequence[int],
    outward_normal: Sequence[Fraction],
) -> list[int]:
    drop = max(range(3), key=lambda axis: abs(outward_normal[axis]))
    keep = [axis for axis in range(3) if axis != drop]
    projected = [
        (points[index][keep[0]], points[index][keep[1]], index)
        for index in vertex_indices
    ]
    projected.sort()

    def turn(a: tuple[Fraction, Fraction, int], b: tuple[Fraction, Fraction, int],
             c: tuple[Fraction, Fraction, int]) -> Fraction:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower: list[tuple[Fraction, Fraction, int]] = []
    for point in projected:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[Fraction, Fraction, int]] = []
    for point in reversed(projected):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    order = [row[2] for row in lower[:-1] + upper[:-1]]
    if len(order) < 3:
        raise RuntimeError("support facet projected to fewer than three hull vertices")
    normal = _cross(
        _sub(points[order[1]], points[order[0]]),
        _sub(points[order[2]], points[order[0]]),
    )
    if _dot(normal, outward_normal) < 0:
        order = [order[0], *reversed(order[1:])]
    return order


def _exact_hull(
    points: Sequence[Sequence[Fraction]],
) -> dict[str, Any]:
    count = len(points)
    if count < 4:
        raise RuntimeError("exact hull needs at least four points")
    centroid = tuple(
        sum((point[axis] for point in points), Fraction(0)) / count
        for axis in range(3)
    )
    facet_sets: dict[tuple[int, ...], tuple[Fraction, Fraction, Fraction]] = {}
    for i, j, k in itertools.combinations(range(count), 3):
        normal = _cross(_sub(points[j], points[i]), _sub(points[k], points[i]))
        if normal == (0, 0, 0):
            continue
        signed = [_dot(normal, _sub(point, points[i])) for point in points]
        if all(value <= 0 for value in signed):
            outward = normal
        elif all(value >= 0 for value in signed):
            outward = tuple(-value for value in normal)
        else:
            continue
        coplanar = tuple(index for index, value in enumerate(signed) if value == 0)
        if len(coplanar) >= 3:
            facet_sets.setdefault(coplanar, outward)  # exact grouping
    if len(facet_sets) < 4:
        raise RuntimeError(f"too few exact supporting facets: {len(facet_sets)}")

    triangles: list[tuple[int, int, int]] = []
    facet_orders: list[list[int]] = []
    for vertices, normal in sorted(facet_sets.items()):
        order = _projected_hull(points, vertices, normal)
        facet_orders.append(order)
        triangles.extend(
            (order[0], order[offset], order[offset + 1])
            for offset in range(1, len(order) - 1)
        )

    volume = Fraction(0)
    edge_counter: Counter[tuple[int, int]] = Counter()
    for i, j, k in triangles:
        volume += abs(
            _det3(
                [
                    _sub(points[i], centroid),
                    _sub(points[j], centroid),
                    _sub(points[k], centroid),
                ]
            )
        ) / 6
        for edge in ((i, j), (j, k), (k, i)):
            edge_counter[tuple(sorted(edge))] += 1

    support_pass = True
    for vertices, normal in facet_sets.items():
        anchor = points[vertices[0]]
        if any(_dot(normal, _sub(point, anchor)) > 0 for point in points):
            support_pass = False
    manifold_pass = all(value == 2 for value in edge_counter.values())
    used_vertices = set(itertools.chain.from_iterable(triangles))
    # The edge counter includes fan diagonals, so use the triangulated face
    # count here (not the coplanar polygon-facet count).
    euler = len(used_vertices) - len(edge_counter) + len(triangles)
    return {
        "volume": volume,
        "facet_vertex_sets": [list(item) for item in sorted(facet_sets)],
        "facet_orders": facet_orders,
        "triangles": [list(item) for item in triangles],
        "support_halfspace_pass": support_pass,
        "triangulated_boundary_edge_twice_pass": manifold_pass,
        "euler_characteristic": euler,
        "euler_pass": euler == 2,
    }


def _qhull_exact_reintegration(
    points_float: np.ndarray,
    points_exact: Sequence[Sequence[Fraction]],
) -> dict[str, Any]:
    anchor = points_float[0]
    centered = points_float - anchor
    maximum = float(np.max(np.abs(centered)))
    if maximum <= 0:
        raise RuntimeError("zero point-cloud span")
    exponent = math.frexp(maximum)[1]
    scale_exponent = -exponent + 8
    scale = math.ldexp(1.0, scale_exponent)
    normalized = centered * scale
    hull = ConvexHull(normalized, qhull_options="Qx Q12 Pp")
    centroid = tuple(
        sum((point[axis] for point in points_exact), Fraction(0)) / len(points_exact)
        for axis in range(3)
    )
    volume = Fraction(0)
    simplices = sorted(tuple(sorted(int(value) for value in row)) for row in hull.simplices)
    for i, j, k in simplices:
        volume += abs(
            _det3(
                [
                    _sub(points_exact[i], centroid),
                    _sub(points_exact[j], centroid),
                    _sub(points_exact[k], centroid),
                ]
            )
        ) / 6
    return {
        "power_of_two_scale_exponent": scale_exponent,
        "qhull_options": "Qx Q12 Pp",
        "qhull_simplices": [list(item) for item in simplices],
        "reintegrated_exact_volume": volume,
        "qhull_float_volume_diagnostic_m3": float(hull.volume) / (scale**3),
    }


def _aabb_volume(points: Sequence[Sequence[Fraction]]) -> Fraction:
    widths = [
        max(point[axis] for point in points) - min(point[axis] for point in points)
        for axis in range(3)
    ]
    return widths[0] * widths[1] * widths[2]


def _diameter_squared(points: Sequence[Sequence[Fraction]]) -> Fraction:
    return max(
        sum(
            ((points[left][axis] - points[right][axis]) ** 2 for axis in range(3)),
            Fraction(0),
        )
        for left, right in itertools.combinations(range(len(points)), 2)
    )


def _diameter_float(points: np.ndarray) -> float:
    return max(
        float(np.linalg.norm(points[left] - points[right]))
        for left, right in itertools.combinations(range(len(points)), 2)
    )


def _record_map() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    d392 = _read_json(D392_EVIDENCE)
    d390_geometry = _read_json(D390_GEOMETRY)
    return (
        {int(row["call_index"]): row for row in d392["records"]},
        {int(row["call_index"]): row for row in d390_geometry["records"]},
    )


def _strict_record_for_call(call_id: str) -> dict[str, Any]:
    evidence = _read_json(D390_EVIDENCE)
    for row in evidence["failed_call_records"]:
        if row["call_id"] == call_id:
            return row["stored_strict_halfspace_relation"]
    raise KeyError(call_id)


def _source_scale(points: np.ndarray) -> dict[str, float]:
    exact = [_frac_point(point) for point in points]
    return {
        "aabb_diagonal_m": float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))),
        "aabb_volume_m3": float(_aabb_volume(exact)),
        "convex_hull_volume_m3_float64_diagnostic": float(
            ConvexHull(points, qhull_options="Q12 Pp").volume
        ),
    }


def _analyze_call(
    d392_row: dict[str, Any], geometry_row: dict[str, Any]
) -> dict[str, Any]:
    points_float = np.asarray(
        d392_row["rank_authority"]["canonical_unique_points_f64_m"],
        dtype=np.float64,
    )
    points = [_frac_point(point) for point in points_float]
    max_indices, witness = _max_tetra(points)
    exact_hull = _exact_hull(points)
    qhull = _qhull_exact_reintegration(points_float, points)
    aabb = _aabb_volume(points)
    diameter_sq = _diameter_squared(points)
    diameter = _diameter_float(points_float)
    diameter_cube_upper = diameter**3
    widths = [
        max(point[axis] for point in points) - min(point[axis] for point in points)
        for axis in range(3)
    ]
    inradius_upper = min(widths) / 2
    strict = _strict_record_for_call(d392_row["call_id"])

    order_control = True
    expected_pair = (max_indices, witness)
    for permutation in itertools.permutations(range(len(points))):
        permuted = [points[index] for index in permutation]
        _, candidate = _max_tetra(permuted)
        if candidate != witness or _aabb_volume(permuted) != aabb:
            order_control = False
            break
    translation = (
        Fraction(1, 2**8),
        -Fraction(1, 2**9),
        Fraction(1, 2**10),
    )
    translated = [
        tuple(point[axis] + translation[axis] for axis in range(3))
        for point in points
    ]
    _, translated_witness = _max_tetra(translated)
    translated_hull = _exact_hull(translated)["volume"]
    scale = Fraction(2**20)
    scaled = [tuple(value * scale for value in point) for point in points]
    _, scaled_witness = _max_tetra(scaled)
    scaled_hull = _exact_hull(scaled)["volume"]
    mutated = points_float.copy()
    mutated[0, 0] = np.nextafter(mutated[0, 0], math.inf)

    checks = {
        "call_index_registered": d392_row["call_index"] in FULL_CALL_INDICES,
        "d392_points_sha_exact": _array_sha(points_float) == d392_row["points_sha256"],
        "d390_terminal_sha_exact": (
            _array_sha(points_float)
            == geometry_row["terminal_candidate_unique_points_sha256"]
        ),
        "d392_stable_full_rank3_exact": (
            d392_row["stable"] is True
            and d392_row["ambiguous"] is False
            and d392_row["rank_authority"]["status"] == "STABLE"
            and d392_row["rank_authority"]["authoritative_rank"] == 3
            and d392_row["rank_authority"]["authoritative_class"] == "FULL_DIMENSIONAL"
            and d392_row["rank_authority"]["exact_dyadic_rank"] == 3
        ),
        "positive_exact_tetra_witness": witness > 0,
        "exact_hull_positive": exact_hull["volume"] > 0,
        "witness_le_exact_hull": witness <= exact_hull["volume"],
        "exact_hull_le_aabb": exact_hull["volume"] <= aabb,
        "aabb_below_frozen_volume_gate": aabb <= POSITIVE_VOLUME_EPS_M3,
        "diameter_cube_below_frozen_volume_gate_exact_no_sqrt": (
            diameter_sq**3 < POSITIVE_VOLUME_EPS_M3**2
        ),
        "inradius_width_upper_below_strict_gate": (
            inradius_upper < STRICT_INTERIOR_RADIUS_M
        ),
        "diameter_radius_upper_below_strict_gate_exact_no_sqrt": (
            diameter_sq < (2 * STRICT_INTERIOR_RADIUS_M) ** 2
        ),
        "exact_support_halfspaces": exact_hull["support_halfspace_pass"],
        "exact_boundary_manifold": exact_hull["triangulated_boundary_edge_twice_pass"],
        "exact_euler_two": exact_hull["euler_pass"],
        "independent_qhull_topology_reintegration_exact": (
            qhull["reintegrated_exact_volume"] == exact_hull["volume"]
        ),
        "max_tetra_and_aabb_point_order_exhaustive_invariant": order_control,
        "exact_translation_invariant": (
            translated_witness == witness and translated_hull == exact_hull["volume"]
        ),
        "power_two_scale_restores_exact": (
            scaled_witness / (scale**3) == witness
            and scaled_hull / (scale**3) == exact_hull["volume"]
        ),
        "single_bit_mutation_hash_negative": _array_sha(mutated) != _array_sha(points_float),
        "stored_strict_calculation_pass": strict["calculation_pass"] is True,
        "stored_strict_positive_false": strict["positive_volume"] is False,
        "stored_strict_gate_zero": strict["volume_m3"] == 0.0,
        "stored_strict_threshold_exact": (
            strict["strict_interior_radius_threshold_nm"] == 0.0001
        ),
    }
    source = _source_scale(
        np.asarray(geometry_row["source_vertices_f64_m"], dtype=np.float64)
    )
    clipping = _source_scale(
        np.asarray(geometry_row["clipping_vertices_f64_m"], dtype=np.float64)
    )
    return {
        "call_index": d392_row["call_index"],
        "call_id": d392_row["call_id"],
        "target": geometry_row["target"],
        "source_child_index": geometry_row["source_child_index"],
        "clipping_child_index": geometry_row["clipping_child_index"],
        "stage": geometry_row["stage"],
        "direction": geometry_row["direction"],
        "adjacent": abs(
            int(geometry_row["source_child_index"])
            - int(geometry_row["clipping_child_index"])
        ) == 1,
        "point_count": len(points),
        "points_sha256": _array_sha(points_float),
        "max_tetra_indices": list(max_indices),
        "max_tetra_volume_m3": witness,
        "exact_convex_hull_volume_m3": exact_hull["volume"],
        "exact_aabb_volume_upper_m3": aabb,
        "diameter_m": diameter,
        "diameter_cube_upper_m3": diameter_cube_upper,
        "diameter_squared_exact_m2": diameter_sq,
        "inradius_coordinate_width_upper_m": inradius_upper,
        "frozen_positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
        "strict_interior_radius_threshold_m": STRICT_INTERIOR_RADIUS_M,
        "volume_gate_over_aabb_upper_ratio": float(POSITIVE_VOLUME_EPS_M3 / aabb),
        "strict_radius_over_width_bound_ratio": float(
            STRICT_INTERIOR_RADIUS_M / inradius_upper
        ),
        "source_scale_diagnostic": source,
        "clipping_scale_diagnostic": clipping,
        "terminal_exact_hull_to_source_aabb_volume_ratio_diagnostic": (
            float(exact_hull["volume"]) / source["aabb_volume_m3"]
        ),
        "exact_hull": exact_hull,
        "independent_normalized_qhull_topology": qhull,
        "registered_semantics": (
            "FULL_DIMENSIONAL_BUT_PROVABLY_SUBTHRESHOLD_TERMINAL_MICRO_VOLUME"
        ),
        "forward_only_gate_semantics": {
            "original_calculation_pass": False,
            "propagated_gate_decision_available": True,
            "propagated_positive_volume": False,
            "derived_gate_volume_m3": 0.0,
            "exact_final_intersection_volume_m3": None,
            "reason": "EXACT_POSITIVE_BUT_BELOW_FROZEN_VOLUME_GATE",
            "monotone_basis": (
                "every remaining halfspace clipping result is a subset of this "
                "terminal candidate, so its volume cannot increase"
            ),
        },
        "checks": checks,
        "pass": all(checks.values()),
        "_display_points": points_float,
    }


def _control_record(
    d392_row: dict[str, Any], expected_class: str
) -> dict[str, Any]:
    points_float = np.asarray(
        d392_row["rank_authority"]["canonical_unique_points_f64_m"],
        dtype=np.float64,
    )
    points = [_frac_point(point) for point in points_float]
    maximum = max(
        (abs(_tetra_det6(points, indices)) for indices in itertools.combinations(range(len(points)), 4)),
        default=Fraction(0),
    )
    return {
        "call_index": d392_row["call_index"],
        "call_id": d392_row["call_id"],
        "expected_class": expected_class,
        "stored_class": d392_row["rank_authority"]["authoritative_class"],
        "maximum_abs_tetra_det6": maximum,
        "unique_point_count": len(points),
        "fewer_than_four_points": len(points) < 4,
        "points_sha256_exact": _array_sha(points_float) == d392_row["points_sha256"],
        "pass": (
            d392_row["rank_authority"]["authoritative_class"] == expected_class
            and len(points) < 4
            and maximum == 0
            and _array_sha(points_float) == d392_row["points_sha256"]
        ),
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    d392_map, geometry_map = _record_map()
    full_rows = [
        row
        for row in d392_map.values()
        if row["rank_authority"]["authoritative_class"] == "FULL_DIMENSIONAL"
        and row["stable"] is True
    ]
    manifest_indices = sorted(int(row["call_index"]) for row in full_rows)
    _phase("full10_manifest_bound", call_indices=manifest_indices)
    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    display_records: list[dict[str, Any]] = []
    for row in sorted(full_rows, key=lambda item: int(item["call_index"])):
        index = int(row["call_index"])
        result = _analyze_call(row, geometry_map[index])
        points = result.pop("_display_points")
        records.append(result)
        csv_rows.append(
            {
                "call_index": index,
                "call_id": result["call_id"],
                "target": result["target"],
                "stage": result["stage"],
                "direction": result["direction"],
                "point_count": result["point_count"],
                "witness_volume_m3": float(result["max_tetra_volume_m3"]),
                "exact_hull_volume_m3": float(result["exact_convex_hull_volume_m3"]),
                "aabb_upper_m3": float(result["exact_aabb_volume_upper_m3"]),
                "diameter_m": result["diameter_m"],
                "diameter_cube_upper_m3": result["diameter_cube_upper_m3"],
                "gate_over_aabb_ratio": result["volume_gate_over_aabb_upper_ratio"],
                "strict_over_width_ratio": result["strict_radius_over_width_bound_ratio"],
                "pass": result["pass"],
            }
        )
        display_records.append(
            {
                "call_index": index,
                "call_id": result["call_id"],
                "canonical_points_f64_m": points,
                "canonical_points_sha256": result["points_sha256"],
            }
        )
        _append_jsonl(
            PROGRESS,
            {
                "ordinal": len(records) - 1,
                "call_index": index,
                "call_id": result["call_id"],
                "exact_hull_volume_m3": result["exact_convex_hull_volume_m3"],
                "pass": result["pass"],
            },
        )
    _phase("full10_exact_sandwich_computed", passing=sum(row["pass"] for row in records))

    face = _control_record(d392_map[FACE_CONTROL_INDEX], "FACE_LIKE")
    line = _control_record(d392_map[LINE_CONTROL_INDEX], "LINE")
    below_side = Fraction(1, 2**20)
    above_side = Fraction(1, 2**19)
    below_volume = below_side**3 / 6
    above_volume = above_side**3 / 6
    below_fixture = [
        (Fraction(0), Fraction(0), Fraction(0)),
        (below_side, Fraction(0), Fraction(0)),
        (Fraction(0), below_side, Fraction(0)),
        (Fraction(0), Fraction(0), below_side),
    ]
    above_fixture = [
        (Fraction(0), Fraction(0), Fraction(0)),
        (above_side, Fraction(0), Fraction(0)),
        (Fraction(0), above_side, Fraction(0)),
        (Fraction(0), Fraction(0), above_side),
    ]
    below_pipeline_volume = _exact_hull(below_fixture)["volume"]
    above_pipeline_volume = _exact_hull(above_fixture)["volume"]
    controls = {
        "face_exact_zero": face,
        "line_exact_zero": line,
        "threshold_straddle": {
            "below_side_m": below_side,
            "below_volume_m3": below_volume,
            "above_side_m": above_side,
            "above_volume_m3": above_volume,
            "below_pipeline_exact_hull_volume_m3": below_pipeline_volume,
            "above_pipeline_exact_hull_volume_m3": above_pipeline_volume,
            "pass": (
                below_pipeline_volume == below_volume
                and above_pipeline_volume == above_volume
                and below_volume < POSITIVE_VOLUME_EPS_M3
                and above_volume > POSITIVE_VOLUME_EPS_M3
            ),
        },
    }
    controls["pass"] = (
        face["pass"] and line["pass"] and controls["threshold_straddle"]["pass"]
    )
    _phase("controls_computed", pass_value=controls["pass"])

    counts = {
        "stable_full_calls": len(records),
        "passing": sum(row["pass"] for row in records),
        "upper": sum(row["target"] == "UPPER" for row in records),
        "lower": sum(row["target"] == "LOWER" for row in records),
        "pre_float32": sum(row["stage"] == "pre_float32" for row in records),
        "post_float32": sum(row["stage"] == "post_float32" for row in records),
        "left_clipped_by_right": sum(
            row["direction"] == "left_clipped_by_right" for row in records
        ),
        "right_clipped_by_left": sum(
            row["direction"] == "right_clipped_by_left" for row in records
        ),
        "adjacent": sum(row["adjacent"] for row in records),
        "nonadjacent": sum(not row["adjacent"] for row in records),
        "unique_pair_contexts": len(
            {
                (
                    row["target"],
                    min(row["source_child_index"], row["clipping_child_index"]),
                    max(row["source_child_index"], row["clipping_child_index"]),
                )
                for row in records
            }
        ),
    }
    scientific_pass = (
        manifest_indices == FULL_CALL_INDICES
        and len(records) == 10
        and all(row["pass"] for row in records)
        and controls["pass"]
        and counts
        == {
            "stable_full_calls": 10,
            "passing": 10,
            "upper": 5,
            "lower": 5,
            "pre_float32": 4,
            "post_float32": 6,
            "left_clipped_by_right": 3,
            "right_clipped_by_left": 7,
            "adjacent": 0,
            "nonadjacent": 10,
            "unique_pair_contexts": 9,
        }
    )
    verdict = (
        "D394_FULL10_EXACT_POSITIVE_BUT_FROZEN_SUBTHRESHOLD_MONOTONIC_EARLY_STOP_PASS"
        if scientific_pass
        else "D394_FULL10_VOLUME_SEMANTICS_INTEGRITY_FAIL_STOP"
    )
    evidence = {
        "artifact": "D394_FULL10_VOLUME_SEMANTICS_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "what_and_why": (
            "Distinguish exact binary64 rank-3 micro-volume from a physical or "
            "frozen-gate-positive overlap for D392's ten stable FULL terminals."
        ),
        "frozen_thresholds": {
            "positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
            "strict_interior_radius_threshold_m": STRICT_INTERIOR_RADIUS_M,
            "strict_interior_radius_threshold_nm": 0.0001,
        },
        "manifest_call_indices": manifest_indices,
        "counts": counts,
        "records": records,
        "controls": controls,
        "aggregate_bounds": {
            "maximum_exact_hull_volume_m3": max(
                row["exact_convex_hull_volume_m3"] for row in records
            ),
            "maximum_exact_aabb_upper_m3": max(
                row["exact_aabb_volume_upper_m3"] for row in records
            ),
            "maximum_diameter_cube_upper_m3_float64_diagnostic": max(
                row["diameter_cube_upper_m3"] for row in records
            ),
            "minimum_gate_over_aabb_upper_ratio": min(
                row["volume_gate_over_aabb_upper_ratio"] for row in records
            ),
            "minimum_strict_radius_over_width_bound_ratio": min(
                row["strict_radius_over_width_bound_ratio"] for row in records
            ),
        },
        "monotone_early_stop_proof": {
            "statement": (
                "Every remaining frozen halfspace clip is an intersection with "
                "the current terminal candidate; the result is a subset and "
                "Lebesgue volume cannot increase."
            ),
            "precondition": "terminal exact hull volume <= frozen positive-volume epsilon",
            "applies_to_calls": FULL_CALL_INDICES,
            "does_not_apply_to_call29_without_separate_bound": True,
            "does_not_update_pair_or_seam_verdict_here": True,
            "pass": all(
                row["exact_convex_hull_volume_m3"] <= POSITIVE_VOLUME_EPS_M3
                for row in records
            ),
        },
        "input_hashes": {_rel(path): _sha(path) for path in EXPECTED_INPUT_SHA256},
        "algorithm_elapsed_seconds": time.monotonic() - started,
        "numeric_verdict": verdict,
        "diagnostic_conclusion": (
            "FULL_DIMENSIONAL_BUT_PROVABLY_SUBTHRESHOLD_TERMINAL_MICRO_VOLUME"
            if scientific_pass
            else None
        ),
        "call29_rank": None,
        "call29_class": None,
        "all_41_aggregate_updated": False,
        "pair_or_seam_verdict_updated": False,
        "scope_counters": {
            "calls_evaluated": 10,
            "all_41_replay": 0,
            "pair_sweeps": 0,
            "call29_reclassification": 0,
            "seam_updates": 0,
            "collider_asset_usd_isaac_physx_warp_cuda": 0,
            "cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_changes": 0,
        },
        "g0a_pass": False,
        "pass": scientific_pass,
    }
    geometry = {
        "artifact": "D394_FULL10_DISPLAY_GEOMETRY_V1",
        "case": CASE,
        "role": (
            "canonical Float64 points for audit; Rerun receives independently "
            "centered/magnified Float32 inspection copies"
        ),
        "records": display_records,
    }
    return evidence, geometry, csv_rows


def _write_csv(rows: list[dict[str, Any]]) -> None:
    with CSV_PATH.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _frozen_checks() -> dict[str, bool]:
    checks = {
        f"input_{path.name}_exact": path.is_file() and _sha(path) == expected
        for path, expected in EXPECTED_INPUT_SHA256.items()
    }
    checks.update(
        {
            "start_exact": START.is_file() and _sha(START) == EXPECTED_START_SHA256,
            "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
            "origin_master_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        }
    )
    return checks


def _prepare() -> int:
    if not OUT_DIR.is_dir() or set(path.name for path in OUT_DIR.iterdir()) != {
        EXECUTION_AUTHORITY.name
    }:
        raise RuntimeError("D394 prepare requires authority-only forward directory")
    external = os.environ.get(EXECUTION_AUTHORITY_ENV)
    if external != _sha(EXECUTION_AUTHORITY):
        raise RuntimeError("D394 external execution authority hash missing/mismatched")
    authority = _read_json(EXECUTION_AUTHORITY)
    frozen = _frozen_checks()
    environment = {
        "numpy_1_26_0": np.__version__ == "1.26.0",
        "rerun_sdk_0_34_1": importlib.metadata.version("rerun-sdk") == "0.34.1",
        "rerun_cli_exists": RERUN_CLI.is_file(),
        "font_exists": FONT_PATH.is_file(),
        "python_no_bytecode_requested": sys.dont_write_bytecode,
        "no_nvidia_module_imported": not any(
            name.split(".", 1)[0]
            in {"isaaclab", "isaacsim", "omni", "pxr", "warp", "torch"}
            for name in sys.modules
        ),
    }
    authority_checks = {
        "artifact_exact": authority.get("artifact") == "D394_EXTERNAL_EXECUTION_AUTHORITY_V1",
        "case_attempt_exact": (
            authority.get("case") == CASE and authority.get("attempt") == ATTEMPT
        ),
        "script_hash_exact": authority.get("script", {}).get("sha256") == _sha(SCRIPT),
        "start_hash_exact": (
            authority.get("start", {}).get("sha256") == EXPECTED_START_SHA256
        ),
        "variables_exact": authority.get("approval", {}).get("new_variables") == VARIABLES,
        "user_text_hash_exact": (
            authority.get("approval", {}).get("normalized_user_text_sha256")
            == _text_sha(EXPECTED_USER_APPROVAL_NORMALIZED)
        ),
        "inputs_exact": authority.get("inputs")
        == {_rel(path): expected for path, expected in EXPECTED_INPUT_SHA256.items()},
        "forward_path_exact": authority.get("output", {}).get("path") == _rel(OUT_DIR),
        "git_status_exact": authority.get("git", {}).get("status_sha256")
        == _text_sha(_git("status", "--porcelain=v1", "--untracked-files=all") + "\n"),
    }
    if not all(frozen.values()):
        raise RuntimeError(f"D394 frozen input failure: {frozen}")
    if not all(environment.values()):
        raise RuntimeError(f"D394 environment failure: {environment}")
    if not all(authority_checks.values()):
        raise RuntimeError(f"D394 authority failure: {authority_checks}")
    _phase("prepare_start")
    prereg = {
        "artifact": "D394_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "question": (
            "Do the ten stable FULL terminals contain exact-dyadic positive "
            "micro-volume, and is each provably below both frozen gates so "
            "remaining clips may stop monotonically?"
        ),
        "exact_numeric_method": [
            "Fraction.from_float exact dyadic coordinates",
            "all 4-point determinants with lexicographic maximum witness",
            "exact supporting facets and pyramid-sum convex-hull volume",
            "translated power-of-two normalized Qhull topology reintegrated with Fractions",
            "exact AABB and diameter-squared upper bounds",
        ],
        "pass_sandwich": (
            "0 < max tetra <= exact convex hull <= exact AABB <= 1e-18 m^3"
        ),
        "frozen_thresholds": {
            "positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
            "strict_interior_radius_threshold_m": STRICT_INTERIOR_RADIUS_M,
        },
        "manifest_call_indices": FULL_CALL_INDICES,
        "controls": [
            "frozen FACE call2 exact zero",
            "frozen LINE call31 exact zero",
            "2^-20m and 2^-19m tetra threshold straddle",
            "all point permutations for max-tetra and AABB invariance",
            "exact dyadic translation",
            "power-of-two scaling",
            "single-bit hash mutation negative",
        ],
        "execution_contract": {
            "offline_worker": 1,
            "worker_retries": 0,
            "process_signals": 0,
            "hard_watchdog_seconds": None,
            "post_exit_elapsed_budget_seconds": COOPERATIVE_DEADLINE_SECONDS,
            "deadline_note": (
                "No signal authority: the worker exits cooperatively; this is "
                "a post-exit elapsed audit, not a kill watchdog."
            ),
            "viewer_maximum": 1,
            "viewer_retries": 0,
            "numeric_before_observability": True,
        },
        "outcome_policy": {
            "pass": (
                "D394_FULL10_EXACT_POSITIVE_BUT_FROZEN_SUBTHRESHOLD_"
                "MONOTONIC_EARLY_STOP_PASS"
            ),
            "failure": "D394_FULL10_VOLUME_SEMANTICS_INTEGRITY_FAIL_STOP",
            "call29_rank_class": None,
            "pair_or_seam_update": None,
        },
        "forbidden": {
            "all_41_or_36_pair_sweep": 0,
            "call29_reclassification": 0,
            "pair_seam_geometry_update": 0,
            "gate_tolerance_qj_jitter_change": 0,
            "collider_asset_usd_isaac_physx_warp_cuda": 0,
            "cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_hardware_signal_commit_push": 0,
        },
        "authority_sha256": _sha(EXECUTION_AUTHORITY),
        "script_sha256": _sha(SCRIPT),
        "start_sha256": _sha(START),
        "frozen_checks": frozen,
        "environment_checks": environment,
        "authority_checks": authority_checks,
        "forward_only_output": _rel(OUT_DIR),
    }
    _write_json_x(PREREGISTRATION, prereg)
    _phase("prepare_end", preregistration_sha256=_sha(PREREGISTRATION))
    print(json.dumps({"case": CASE, "prepared": True}, ensure_ascii=False))
    return 0


def _worker() -> int:
    authorization = _read_json(AUTHORIZATION)
    invocation = _read_json(INVOCATION)
    frozen = _frozen_checks()
    if os.environ.get(WORKER_AUTHORIZATION_ENV) != _sha(AUTHORIZATION):
        raise RuntimeError("D394 worker authorization hash mismatch")
    checks = {
        "parent_pid_exact": os.getppid() == authorization["supervisor_pid"],
        "worker_index_one": authorization["worker_invocation_index"] == 1,
        "retry_zero": authorization["retry_index"] == 0,
        "script_exact": authorization["script_sha256"] == _sha(SCRIPT),
        "prereg_exact": authorization["preregistration_sha256"] == _sha(PREREGISTRATION),
        "authority_exact": authorization["execution_authority_sha256"]
        == _sha(EXECUTION_AUTHORITY),
        "invocation_file_hash_exact": (
            authorization["invocation_sha256"] == _sha(INVOCATION)
        ),
        "invocation_script_and_chain_exact": (
            invocation["script_sha256"] == _sha(SCRIPT)
            and invocation["preregistration_sha256"] == _sha(PREREGISTRATION)
            and invocation["execution_authority_sha256"] == _sha(EXECUTION_AUTHORITY)
        ),
        "authorization_input_hashes_exact": authorization["input_hashes"]
        == {_rel(path): expected for path, expected in EXPECTED_INPUT_SHA256.items()},
        "invocation_input_hashes_exact": invocation["input_hashes"]
        == {_rel(path): expected for path, expected in EXPECTED_INPUT_SHA256.items()},
        "start_hash_bound": (
            authorization["start_sha256"] == EXPECTED_START_SHA256
            and invocation["start_sha256"] == EXPECTED_START_SHA256
        ),
        "head_bound": (
            authorization["head"] == EXPECTED_HEAD
            and invocation["head"] == EXPECTED_HEAD
        ),
        "frozen_inputs_rechecked_in_worker": all(frozen.values()),
        "signals_false": authorization["signal_authority"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"D394 worker authorization failed: {checks}")
    _write_json_x(
        SENTINEL,
        {
            "artifact": "D394_WORKER_START_SENTINEL_V1",
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "authorization_sha256": _sha(AUTHORIZATION),
            "checks": checks,
            "start_monotonic": time.monotonic(),
        },
    )
    _phase("worker_start", worker_pid=os.getpid())
    evidence, geometry, rows = _compute()
    _write_json_x(EVIDENCE, evidence)
    _write_json_x(GEOMETRY, geometry)
    _write_csv(rows)
    _phase(
        "canonical_numeric_evidence_committed",
        numeric_verdict=evidence["numeric_verdict"],
        evidence_sha256=_sha(EVIDENCE),
    )
    claim = {
        "artifact": "D394_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "script_sha256": _sha(SCRIPT),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "authorization_sha256": _sha(AUTHORIZATION),
        "sentinel_sha256": _sha(SENTINEL),
        "progress_sha256": _sha(PROGRESS),
        "evidence": _artifact(EVIDENCE),
        "geometry": _artifact(GEOMETRY),
        "csv": _artifact(CSV_PATH),
        "numeric_verdict": evidence["numeric_verdict"],
        "actual_worker_invocations": 1,
        "retries": 0,
        "process_signals_sent": 0,
        "pass": evidence["pass"],
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_end", pass_value=claim["pass"])
    print(json.dumps(claim, ensure_ascii=False))
    return 0 if claim["pass"] else 1


def _run() -> int:
    required = {EXECUTION_AUTHORITY.name, PREREGISTRATION.name, PHASES.name}
    if set(path.name for path in OUT_DIR.iterdir()) != required:
        raise RuntimeError("D394 run requires exact prepared artifact set")
    if not all(_frozen_checks().values()):
        raise RuntimeError("D394 frozen inputs changed after prepare")
    invocation = {
        "artifact": "D394_OFFLINE_WORKER_INVOCATION_V1",
        "command": [sys.executable, "-B", _rel(SCRIPT), "--stage", "worker"],
        "cwd": str(REPO),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "script_sha256": _sha(SCRIPT),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "input_hashes": {
            _rel(path): expected for path, expected in EXPECTED_INPUT_SHA256.items()
        },
        "start_sha256": EXPECTED_START_SHA256,
        "head": EXPECTED_HEAD,
        "signal_authority": False,
    }
    _write_json_x(INVOCATION, invocation)
    authorization = {
        "artifact": "D394_WORKER_AUTHORIZATION_V1",
        "supervisor_pid": os.getpid(),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "script_sha256": _sha(SCRIPT),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "invocation_sha256": _sha(INVOCATION),
        "input_hashes": {
            _rel(path): expected for path, expected in EXPECTED_INPUT_SHA256.items()
        },
        "start_sha256": EXPECTED_START_SHA256,
        "head": EXPECTED_HEAD,
        "signal_authority": False,
    }
    _write_json_x(AUTHORIZATION, authorization)
    _phase("supervisor_before_worker", supervisor_pid=os.getpid())
    environment = os.environ.copy()
    environment[WORKER_AUTHORIZATION_ENV] = _sha(AUTHORIZATION)
    started = time.monotonic()
    with STDOUT.open("x", encoding="utf-8") as stdout, STDERR.open(
        "x", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            invocation["command"],
            cwd=REPO,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        returncode = process.wait()
    elapsed = time.monotonic() - started
    worker = _read_json(WORKER_CLAIM) if WORKER_CLAIM.is_file() else None
    supervisor = {
        "artifact": "D394_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "script_sha256": _sha(SCRIPT),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "invocation_sha256": _sha(INVOCATION),
        "authorization_sha256": _sha(AUTHORIZATION),
        "sentinel_sha256": _sha(SENTINEL) if SENTINEL.is_file() else None,
        "worker_claim_sha256": _sha(WORKER_CLAIM) if WORKER_CLAIM.is_file() else None,
        "returncode": returncode,
        "worker_exited": process.poll() is not None,
        "elapsed_seconds": elapsed,
        "post_exit_elapsed_budget_seconds": COOPERATIVE_DEADLINE_SECONDS,
        "hard_watchdog_seconds": None,
        "deadline_semantics": (
            "post-exit elapsed audit only; supervisor has no signal authority"
        ),
        "post_exit_elapsed_budget_exceeded": elapsed > COOPERATIVE_DEADLINE_SECONDS,
        "actual_worker_invocations": 1,
        "retries": 0,
        "process_signals_sent": 0,
        "pass": (
            returncode == 0
            and worker is not None
            and worker["pass"] is True
            and elapsed <= COOPERATIVE_DEADLINE_SECONDS
        ),
    }
    _write_json_x(SUPERVISOR, supervisor)
    if WORKER_CLAIM.is_file():
        phase_prefix = [
            json.loads(line)["phase"]
            for line in PHASES.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if phase_prefix and phase_prefix[-1] == "worker_end":
            _phase(
                "supervisor_after_worker",
                returncode=returncode,
                pass_value=supervisor["pass"],
            )
    if not supervisor["pass"]:
        raise RuntimeError(f"D394 worker failed: {supervisor}")
    print(json.dumps(supervisor, ensure_ascii=False))
    return 0


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _render_board(evidence: dict[str, Any]) -> dict[str, Any]:
    image = Image.new("RGB", (1920, 1080), (246, 248, 250))
    draw = ImageDraw.Draw(image)
    title = _font(40)
    subtitle = _font(25)
    body = _font(22)
    small = _font(18)
    draw.rectangle((0, 0, 1920, 100), fill=(30, 42, 56))
    draw.text((55, 24), "D394 · FULL 10건의 ‘미세 3차원 잔차’ 의미 검증", font=title, fill="white")
    draw.text(
        (55, 112),
        "정확히 0은 아니지만, 동결된 겹침 기준보다 충분히 작은지 독립 계산으로 확인",
        font=subtitle,
        fill=(35, 50, 65),
    )
    headers = ["호출", "점", "최대 4점 하한 (m³)", "정확 hull (m³)", "AABB 상한 (m³)", "기준/AABB"]
    xs = [55, 185, 245, 550, 850, 1150]
    y0 = 178
    draw.rounded_rectangle((40, y0 - 12, 1505, y0 + 46), radius=10, fill=(220, 228, 236))
    for x, text in zip(xs, headers, strict=True):
        draw.text((x, y0), text, font=small, fill=(25, 38, 50))
    row_y = y0 + 62
    for ordinal, row in enumerate(evidence["records"]):
        fill = (255, 255, 255) if ordinal % 2 == 0 else (235, 241, 246)
        draw.rectangle((40, row_y - 6, 1505, row_y + 34), fill=fill)
        values = [
            f"C{row['call_index']:02d}",
            str(row["point_count"]),
            f"{row['max_tetra_volume_m3']['float']:.3e}",
            f"{row['exact_convex_hull_volume_m3']['float']:.3e}",
            f"{row['exact_aabb_volume_upper_m3']['float']:.3e}",
            f"{row['volume_gate_over_aabb_upper_ratio']:.2e}×",
        ]
        for x, text in zip(xs, values, strict=True):
            draw.text((x, row_y), text, font=small, fill=(22, 44, 60))
        draw.text((1330, row_y), "PASS" if row["pass"] else "FAIL", font=small,
                  fill=(20, 135, 80) if row["pass"] else (200, 45, 55))
        row_y += 46

    panel_y = 740
    draw.rounded_rectangle((40, panel_y, 930, 1025), radius=16, fill=(226, 242, 234))
    draw.text((65, panel_y + 22), "판정 원리", font=subtitle, fill=(20, 95, 60))
    lines = [
        "1) 4점 사면체 하한 > 0  → binary64 점들은 정확히 3차원",
        "2) exact hull ≤ AABB ≤ 10⁻¹⁸ m³  → 동결 부피 gate에서는 음성",
        "3) 남은 halfspace 절단은 부분집합  → 이후 부피는 증가할 수 없음",
        "4) 내부반지름 상한도 10⁻¹³ m보다 작음 (별도 길이 gate)",
    ]
    for offset, line in enumerate(lines):
        draw.text((65, panel_y + 72 + offset * 47), line, font=body, fill=(25, 55, 43))

    draw.rounded_rectangle((960, panel_y, 1880, 1025), radius=16, fill=(252, 235, 230))
    draw.text((985, panel_y + 22), "이 단계가 말하지 않는 것", font=subtitle, fill=(145, 55, 40))
    nonclaims = [
        "• FULL10은 모두 비인접 쌍: 9개 seam 결론은 아직 미변경",
        "• call29의 rank/class는 계속 null",
        "• 제조 두께·PhysX 접촉·파지 가능성의 증거가 아님",
        "• collider/USD/Isaac/원통 물리 실행은 0",
    ]
    for offset, line in enumerate(nonclaims):
        draw.text((985, panel_y + 75 + offset * 47), line, font=body, fill=(85, 40, 32))

    image.save(BOARD)
    boxes = [
        {"name": "title", "box": [0, 0, 1920, 100]},
        {"name": "table", "box": [40, 166, 1505, 700]},
        {"name": "principle", "box": [40, panel_y, 930, 1025]},
        {"name": "nonclaims", "box": [960, panel_y, 1880, 1025]},
    ]

    def overlaps(left: list[int], right: list[int]) -> bool:
        return not (
            left[2] <= right[0]
            or right[2] <= left[0]
            or left[3] <= right[1]
            or right[3] <= left[1]
        )

    panel_pairs = list(itertools.combinations(boxes, 2))
    panels_nonoverlapping = all(
        not overlaps(left["box"], right["box"]) for left, right in panel_pairs
    )
    content_inside = all(
        0 <= box["box"][0] < box["box"][2] <= 1920
        and 0 <= box["box"][1] < box["box"][3] <= 1080
        for box in boxes
    )
    report = {
        "artifact": "D394_BOARD_LAYOUT_VALIDATION_V1",
        "board": _artifact(BOARD),
        "width": 1920,
        "height": 1080,
        "boxes": boxes,
        "checks": {
            "exact_1920x1080": Image.open(BOARD).size == (1920, 1080),
            "all_ten_rows_present": len(evidence["records"]) == 10,
            "registered_panels_nonoverlapping": panels_nonoverlapping,
            "content_inside_canvas": content_inside,
        },
    }
    report["pass"] = all(report["checks"].values())
    _write_json_x(LAYOUT, report)
    return report


def _blueprint() -> Any:
    import rerun.blueprint as rrb

    decision = rrb.Vertical(
        rrb.Spatial3DView(
            origin="/",
            contents="/d394/**",
            name="D394 magnified FULL10 atlas (inspection only)",
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=(4.0, -18.0, 17.0),
                look_target=(0.0, 0.0, 0.0),
                eye_up=(0.0, 0.0, 1.0),
            ),
            spatial_information=rrb.SpatialInformation(
                target_frame="tf#/", show_axes=True, show_bounding_box=False
            ),
        ),
        rrb.TextDocumentView(
            origin="/metadata/run",
            contents="/metadata/run",
            name="Canonical numeric authority and nonclaims",
        ),
        row_shares=[0.78, 0.22],
    )
    notification = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d394/notification_buffer/**",
        name="Notification buffer - no decision evidence",
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/", show_axes=False, show_bounding_box=False
        ),
    )
    return rrb.Blueprint(
        rrb.Horizontal(decision, notification, column_shares=[0.78, 0.22]),
        rrb.TimePanel(state=rrb.PanelState.Hidden),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "path": _rel(path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }


def _write_rerun(evidence: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    import roarm_rl.rerun_contract as rerun_contract
    import roarm_rl.viz_debug as viz_debug

    points_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(geometry["records"]):
        points = np.asarray(row["canonical_points_f64_m"], dtype=np.float64)
        centered = points - points.mean(axis=0)
        span = float(np.max(np.linalg.norm(centered, axis=1)))
        scale = 0.75 / span if span > 0 else 1.0
        offset = np.asarray([(ordinal % 5) * 2.6 - 5.2, (1 - ordinal // 5) * 3.0 - 1.5, 0.0])
        display = centered * scale + offset
        points_rows.append(
            {
                "entity_path": f"d394/full10/c{row['call_index']:02d}",
                "positions_m": display,
                "radii": [0.075] * len(display),
                "colors": [[225, 120, 35, 255]] * len(display),
                "labels": [f"C{row['call_index']:02d}"] + [""] * (len(display) - 1),
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
    metadata = {
        "case": CASE,
        "numeric_verdict": evidence["numeric_verdict"],
        "canonical_evidence_sha256": _sha(EVIDENCE),
        "display_geometry_sha256": _sha(GEOMETRY),
        "viewer_role": "centered and magnified Float32 inspection atlas only",
        "magnification_is_not_physical_scale": True,
        "call29_rank": None,
        "call29_class": None,
        "pair_or_seam_updated": False,
        "g0a_pass": False,
    }
    original_builder = viz_debug.build_rerun_blueprint
    original_runner = rerun_contract._run
    viewer_calls = 0

    def routed(mode: str = "robot_geometry") -> Any:
        return _blueprint() if mode == "d394_full10_static_atlas" else original_builder(mode)

    def no_signal(command: list[str], *, timeout_s: float) -> dict[str, Any]:
        nonlocal viewer_calls
        del timeout_s
        if any("screenshot" in str(part) for part in command):
            viewer_calls += 1
            if viewer_calls > 1:
                return {
                    "command": command,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "D394 Viewer maximum one exceeded",
                    "ok": False,
                }
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        return {
            "command": command,
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "ok": result.returncode == 0,
            "timeout_parameter_ignored_no_signal_authority": True,
        }

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed
    rerun_contract._run = no_signal
    try:
        saved = viz_debug.log_rerun(
            RRD,
            points=points_rows,
            recording_metadata=metadata,
            recording_id="g0a_d394_full10_volume_semantics",
            blueprint_path=RBL,
            blueprint_mode="d394_full10_static_atlas",
            live_viewer=False,
            app_id="roarm_g0a_d394_full10_volume_semantics",
        )
        if not saved.get("ok"):
            raise RuntimeError(f"D394 Rerun save failed: {saved}")
        entities = ["metadata/run"] + sorted(row["entity_path"] for row in points_rows)
        components = {
            "metadata/run": ["TextDocument:text"],
            **{
                row["entity_path"]: [
                    "CoordinateFrame:frame",
                    "Points3D:colors",
                    "Points3D:labels",
                    "Points3D:positions",
                    "Points3D:radii",
                ]
                for row in points_rows
            },
        }
        validation = rerun_contract.validate_rerun_artifact(
            RRD,
            expected_entity_paths=entities,
            exact_entity_paths=entities,
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
    screenshot = _png(RERUN_SCREENSHOT)
    dimensions = (
        screenshot["width"] in {1920, 3840}
        and screenshot["height"] in {1080, 2160}
        and screenshot["width"] * 9 == screenshot["height"] * 16
    )
    validation["d394_execution_contract"] = {
        "static_atlas": True,
        "viewer_calls": viewer_calls,
        "viewer_maximum": 1,
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "screenshot_dimension_contract_pass": dimensions,
    }
    validation["base_contract_pass"] = validation.get("pass") is True
    validation["pass"] = validation["base_contract_pass"] and viewer_calls == 1 and dimensions
    _write_json_x(RERUN_VALIDATION, validation)
    return {
        "pass": validation["pass"],
        "viewer_calls": viewer_calls,
        "rrd": _artifact(RRD),
        "rbl": _artifact(RBL),
        "validation": _artifact(RERUN_VALIDATION),
        "screenshot": screenshot,
    }


def _observe() -> int:
    supervisor = _read_json(SUPERVISOR)
    evidence = _read_json(EVIDENCE)
    geometry = _read_json(GEOMETRY)
    if supervisor["pass"] is not True or evidence["pass"] is not True:
        raise RuntimeError("D394 observe requires passing frozen numeric worker")
    _phase("observability_start")
    layout = _render_board(evidence)
    rerun = _write_rerun(evidence, geometry)
    template = {
        "artifact": "D394_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board_path": _rel(BOARD),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "artifact_hashes": {
            "board_sha256": _sha(BOARD),
            "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
            "evidence_sha256": _sha(EVIDENCE),
        },
        "required_checks": MANUAL_CHECK_KEYS,
        "instructions": (
            "Inspect both PNGs, copy this template to d394_manual_visual_inspection.json "
            "with every check as a boolean, observations, inspector, and pass."
        ),
    }
    _write_json_x(MANUAL_TEMPLATE, template)
    claim = {
        "artifact": "D394_OBSERVABILITY_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "numeric_evidence_sha256": _sha(EVIDENCE),
        "board": _artifact(BOARD),
        "layout": _artifact(LAYOUT),
        "rerun": rerun,
        "manual_template": _artifact(MANUAL_TEMPLATE),
        "layout_pass": layout["pass"],
        "rerun_pass": rerun["pass"],
        "viewer_invocations": rerun["viewer_calls"],
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "pass": layout["pass"] and rerun["pass"] and rerun["viewer_calls"] == 1,
    }
    _write_json_x(OBSERVABILITY, claim)
    _phase("observability_end", pass_value=claim["pass"])
    if not claim["pass"]:
        raise RuntimeError(f"D394 observability failed: {claim}")
    print(json.dumps(claim, ensure_ascii=False))
    return 0


def _finalize() -> int:
    if not MANUAL.is_file():
        raise RuntimeError("D394 manual visual inspection is required")
    _phase("finalize_start")
    manual = _read_json(MANUAL)
    evidence = _read_json(EVIDENCE)
    supervisor = _read_json(SUPERVISOR)
    observability = _read_json(OBSERVABILITY)
    checks = {
        "manual_artifact_exact": manual.get("artifact") == "D394_MANUAL_VISUAL_INSPECTION_V1",
        "manual_paths_exact": (
            manual.get("board_path") == _rel(BOARD)
            and manual.get("rerun_screenshot_path") == _rel(RERUN_SCREENSHOT)
        ),
        "manual_hashes_exact": manual.get("artifact_hashes")
        == {
            "board_sha256": _sha(BOARD),
            "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
            "evidence_sha256": _sha(EVIDENCE),
        },
        "manual_keys_exact": sorted(manual.get("checks", {})) == sorted(MANUAL_CHECK_KEYS),
        "manual_all_pass": (
            all(manual.get("checks", {}).values()) and manual.get("pass") is True
        ),
        "numeric_pass": evidence["pass"] is True,
        "supervisor_pass": supervisor["pass"] is True,
        "observability_pass": observability["pass"] is True,
        "one_worker_no_retry_signal": (
            supervisor["actual_worker_invocations"] == 1
            and supervisor["retries"] == 0
            and supervisor["process_signals_sent"] == 0
        ),
        "one_viewer_no_retry": (
            observability["viewer_invocations"] == 1
            and observability["viewer_retries"] == 0
        ),
        "call29_and_seam_unchanged": (
            evidence["call29_rank"] is None
            and evidence["call29_class"] is None
            and evidence["pair_or_seam_verdict_updated"] is False
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D394 finalize checks failed: {checks}")
    _phase("finalize_end", pass_value=True)
    completion = {
        "artifact": "D394_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "checks": checks,
        "numeric_verdict": evidence["numeric_verdict"],
        "diagnostic_conclusion": evidence["diagnostic_conclusion"],
        "records_pass": evidence["counts"]["passing"],
        "records_total": evidence["counts"]["stable_full_calls"],
        "execution": {
            "actual_worker_invocations": 1,
            "worker_retries": 0,
            "viewer_invocations": 1,
            "viewer_retries": 0,
            "process_signals_sent": 0,
        },
        "artifacts": {
            "script": _artifact(SCRIPT),
            "execution_authority": _artifact(EXECUTION_AUTHORITY),
            "preregistration": _artifact(PREREGISTRATION),
            "evidence": _artifact(EVIDENCE),
            "geometry": _artifact(GEOMETRY),
            "csv": _artifact(CSV_PATH),
            "supervisor": _artifact(SUPERVISOR),
            "board": _artifact(BOARD),
            "rrd": _artifact(RRD),
            "rbl": _artifact(RBL),
            "rerun_validation": _artifact(RERUN_VALIDATION),
            "rerun_screenshot": _artifact(RERUN_SCREENSHOT),
            "manual": _artifact(MANUAL),
            "phases": _artifact(PHASES),
        },
        "call29_rank": None,
        "call29_class": None,
        "pair_or_seam_verdict_updated": False,
        "materializable_candidate": False,
        "physics_or_grasp_result": None,
        "operational_verdict": (
            "D394_FULL10_EXACT_VOLUME_SEMANTICS_COMPLETE_NO_PAIR_OR_SEAM_ADOPTION"
        ),
        "g0a_pass": False,
        "pass": True,
    }
    _write_json_x(COMPLETION, completion)
    print(json.dumps(completion, ensure_ascii=False))
    return 0


def _failure(stage: str, exc: BaseException) -> None:
    if FAILURE.exists():
        return
    try:
        _write_json_x(
            FAILURE,
            {
                "artifact": "D394_FAILURE_ATTESTATION_V1",
                "case": CASE,
                "attempt": ATTEMPT,
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
                "script_sha256": _sha(SCRIPT),
                "phase_prefix": [
                    json.loads(line)["phase"]
                    for line in PHASES.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if PHASES.is_file()
                else [],
                "actual_worker_invocations": (
                    1 if INVOCATION.is_file() else 0
                ),
                "process_signals_sent": 0,
                "verdict": "D394_OPERATIONAL_OR_NUMERIC_INTEGRITY_FAIL_STOP",
                "g0a_pass": False,
            },
        )
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "run", "worker", "observe", "finalize"),
        required=True,
    )
    args = parser.parse_args()
    functions = {
        "prepare": _prepare,
        "run": _run,
        "worker": _worker,
        "observe": _observe,
        "finalize": _finalize,
    }
    try:
        return functions[args.stage]()
    except Exception as exc:
        if OUT_DIR.is_dir():
            _failure(args.stage, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
