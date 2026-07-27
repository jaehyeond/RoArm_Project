#!/usr/bin/env python3
"""D393: localize call29's microscopic third direction in frozen provenance.

This case is deliberately offline-only and call29-only.  It independently
reconstructs the two frozen D389 LOWER child streams and replays only the
frozen-order call29 plane lineage.  It does not repair D389/D390, change the
D391 rank policy, decide overlap volume, or invoke any NVIDIA runtime.

The numerical authority is canonical Float64/exact-rational JSON.  Rerun is
inspection-only and is produced only after the numeric worker has exited.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
import time
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

CASE = "D393"
ATTEMPT = "attempt1_call29_third_direction_provenance_localization"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d393" / ATTEMPT
START = REPO / "START_HERE.md"
EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
EXPECTED_START_SHA256 = (
    "8d1c36a8e7145b0ed9cff7614a3561bb982a6aca3c78f5efd32bece6ccd9237a"
)

D388_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d388"
    / "attempt1_two_null_moving_support_midlayer_partition_repair_design"
)
D388_EVIDENCE = D388_DIR / "d388_two_null_reanchor_design_evidence.json"
D388_GEOMETRY = D388_DIR / "d388_two_null_reanchor_witness_geometry.json"
D389_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d389"
    / "attempt2_prereg_status_whitespace_repair"
)
D389_EVIDENCE = D389_DIR / "d389_numeric_and_tie_audit_evidence.json"
D389_WITNESS = D389_DIR / "d389_reconstructed_seam_witness_geometry.json"
D389_CSV = D389_DIR / "d389_seam_numeric_provenance.csv"
D389_SCRIPT = (
    REPO
    / "sim_scripts"
    / "cyl34_top_view_d389_attempt2_prereg_status_whitespace_repair.py"
)
D390_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d390"
    / "attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization"
)
D390_EVIDENCE = D390_DIR / "d390_boundary_collapse_localization_evidence.json"
D390_GEOMETRY = D390_DIR / "d390_terminal_candidate_geometry.json"
D390_CSV = D390_DIR / "d390_failed_directional_call_trace.csv"
D391_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d391"
    / "attempt1_d390_rank_basis_and_clip_input_immutability_repair"
)
D391_EVIDENCE = D391_DIR / "d391_rank_and_plane_immutability_evidence.json"
D391_COMPLETION = D391_DIR / "d391_completion_summary.json"
D392_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d392"
    / "attempt1_d391_remaining35_same_authority_coverage_audit"
)
D392_EVIDENCE = D392_DIR / "d392_remaining35_rank_evidence.json"
D392_COMPLETION = D392_DIR / "d392_completion_summary.json"

EXPECTED_INPUT_SHA256 = {
    D388_EVIDENCE: "582368f093ba08fec0207967e8e24ac24f0a44774dfa1a7b8c82ae2b6781caba",
    D388_GEOMETRY: "c119ededf4400efbef55de4d89ccd6c1c8b4e33d4d3795710b6882d369f5e882",
    D389_EVIDENCE: "9423e870c0a218606781943abd2f5c48cb1e5d53cbbf9fb1212294b4ef5bb5dd",
    D389_WITNESS: "66042a93389cb8d0e6c867be87382566c753cd965ceda619e947e73de4a607be",
    D389_CSV: "1fdbaac1c756983c8bd2d2d8e8eabed36a4530393b5f3e3491678335d778f66f",
    D390_EVIDENCE: "3014610b7b2fd953740d239b91d9d9dce8aa917be67c6a1be15cf6ac052d9975",
    D390_GEOMETRY: "73fc986043b976bec26e1cc92643b8aab281a529f1c71c2918163ba7b98475c7",
    D390_CSV: "5a8295c85b62552459806c8a51a2ec485c20248365e8d1382857c19ebb527591",
    D391_EVIDENCE: "d76bbd88b2a6f188f9c46c382adc7292e56267e6e4a3beb8b9004b70e34b80cc",
    D391_COMPLETION: "9d09e55e6cf7cb7e6d60b1fb1e7722dff09a4da3487e299f3304ad4310d085ea",
    D392_EVIDENCE: "9ced175925c6c528d47bf94e5ae224e65bae1d4d6c88fc236952343de0c72102",
    D392_COMPLETION: "01046290ca840960290198a4136cc7ed818cfbeccc7b745139a9e419dd421f70",
}
EXPECTED_D389_SCRIPT_SHA256 = (
    "105b2403b8d49d5baf80390bfc319c446786055e395e141c6918dc55a4d983cb"
)

CALL_INDEX = 29
CALL_ID = "lower_01_02_pre_float32_lbr"
CALL_DIRECTION = "left_clipped_by_right"
CALL_STAGE = "pre_float32"
EXPECTED_SOURCE_SHA256 = (
    "f387f7353249ed411345c71fd6a5f26e8bda1a146ca443cb127fc3a0e5cd18bf"
)
EXPECTED_CLIPPING_SHA256 = (
    "7996b3485b923f993787aaaa69577415fabed3339c409173f95fb233d08a893f"
)
EXPECTED_TERMINAL_SHA256 = (
    "dcd4590e77d929d5abd4edb15f594d5956a9472f9ee099724b39544a7fdfddc6"
)
EXPECTED_ACTIVE_PLANES = [3, 15, 20]
EXPECTED_PLANE_TRACE_SHA256 = (
    "2f702df1c7747ad8ae572031d06a8ea7f5c062f55eae9ee78a78a963c72a5cc9"
)
SOURCE_QUARTET_INDICES = [0, 3, 4, 6]
VARIABLES = [
    "call29_d389_shared_fan_seam_exact_rational_point_lineage_v1",
    "call29_d390_clip_carry_and_near_duplicate_decomposition_v1",
]
EXPECTED_USER_APPROVAL_NORMALIZED = (
    "D390 나머지35 동결 D391 기준 적용, call29 미세 제3방향 국소화, "
    "D389 seam 반영, 이후 collider/USD/Isaac/PhysX/29x50mm 원통 물리를 "
    "각 선행 결과 동결 뒤 순차 진행 승인"
)

COOPERATIVE_DEADLINE_SECONDS = 300.0
EXECUTION_AUTHORITY_SHA256_ENV = "D393_EXECUTION_AUTHORITY_SHA256"
WORKER_AUTHORIZATION_SHA256_ENV = "D393_WORKER_AUTHORIZATION_SHA256"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

EXECUTION_AUTHORITY = OUT_DIR / "d393_execution_authority.json"
PREREGISTRATION = OUT_DIR / "d393_preregistration.json"
PHASES = OUT_DIR / "d393_phase_markers.jsonl"
INVOCATION = OUT_DIR / "d393_offline_worker_invocation.json"
AUTHORIZATION = OUT_DIR / "d393_worker_authorization.json"
SENTINEL = OUT_DIR / "d393_worker_start_sentinel.json"
STDOUT = OUT_DIR / "d393_offline_worker_stdout.log"
STDERR = OUT_DIR / "d393_offline_worker_stderr.log"
PROGRESS = OUT_DIR / "d393_call29_lineage_progress.jsonl"
EVIDENCE = OUT_DIR / "d393_call29_provenance_evidence.json"
GEOMETRY = OUT_DIR / "d393_call29_lineage_geometry.json"
CSV_PATH = OUT_DIR / "d393_call29_point_lineage.csv"
WORKER_CLAIM = OUT_DIR / "d393_offline_worker_claim.json"
WORKER_FAILURE = OUT_DIR / "d393_worker_failure_claim.json"
SUPERVISOR = OUT_DIR / "d393_offline_worker_supervisor.json"
BOARD = OUT_DIR / "d393_call29_provenance_1920x1080.png"
LAYOUT = OUT_DIR / "d393_board_layout_validation.json"
RRD = OUT_DIR / "d393_call29_provenance.rerun.rrd"
RBL = OUT_DIR / "d393_call29_provenance.rerun.rbl"
RERUN_VALIDATION = OUT_DIR / "d393_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d393_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d393_manual_visual_inspection_template.json"
OBSERVABILITY_CLAIM = OUT_DIR / "d393_observability_claim.json"
MANUAL = OUT_DIR / "d393_manual_visual_inspection.json"
FAILURE = OUT_DIR / "d393_failure_attestation.json"
COMPLETION = OUT_DIR / "d393_completion_summary.json"

MANUAL_CHECK_KEYS = [
    "board_exact_1920x1080",
    "board_two_source_children_readable",
    "board_three_active_clip_stages_and_terminal_six_visible",
    "board_terminal_lineage_labels_readable",
    "board_numbers_match_canonical_json",
    "board_null_and_nonclaims_readable",
    "board_no_text_overlap_or_clipping",
    "rerun_source_stage_terminal_entities_visible",
    "rerun_residual_magnification_clearly_inspection_only",
    "rerun_screenshot_matches_numeric_evidence_and_notifications_obscure_nothing",
]
PHASE_ORDER = [
    "prepare_start",
    "prepare_end",
    "supervisor_before_worker",
    "worker_start",
    "call29_source_streams_bound",
    "call29_active_clip_1_lineage_committed",
    "call29_active_clip_2_lineage_committed",
    "call29_terminal_lineage_committed",
    "canonical_numeric_evidence_committed",
    "worker_end",
    "supervisor_after_worker",
    "observability_start",
    "observability_end",
    "finalize_start",
    "finalize_end",
]

AUTHORITY_NAMES = {EXECUTION_AUTHORITY.name}
PREPARED_NAMES = AUTHORITY_NAMES | {PREREGISTRATION.name, PHASES.name}
PRE_WORKER_NAMES = PREPARED_NAMES | {
    INVOCATION.name,
    AUTHORIZATION.name,
    STDOUT.name,
    STDERR.name,
}
POST_WORKER_NAMES = PRE_WORKER_NAMES | {
    SENTINEL.name,
    PROGRESS.name,
    EVIDENCE.name,
    GEOMETRY.name,
    CSV_PATH.name,
    WORKER_CLAIM.name,
    SUPERVISOR.name,
}
POST_OBSERVE_NAMES = POST_WORKER_NAMES | {
    BOARD.name,
    LAYOUT.name,
    RRD.name,
    RBL.name,
    RERUN_VALIDATION.name,
    RERUN_SCREENSHOT.name,
    MANUAL_TEMPLATE.name,
    OBSERVABILITY_CLAIM.name,
}
FINAL_NAMES = POST_OBSERVE_NAMES | {MANUAL.name, COMPLETION.name}

_deadline_monotonic: float | None = None
_worker_started = False
_worker_pid: int | None = None


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Fraction):
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
            "float": float(value),
        }
    if isinstance(value, Path):
        return _rel(value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _native(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _array_sha(points: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(points, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"|float64|C|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _status_lines() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _out_names() -> set[str]:
    if not OUT_DIR.is_dir():
        return set()
    return {path.name for path in OUT_DIR.iterdir() if path.is_file()}


def _require_names(expected: set[str], stage: str) -> None:
    observed = _out_names()
    if observed != expected:
        raise RuntimeError(
            f"{stage} output inventory mismatch: "
            f"missing={sorted(expected-observed)} "
            f"unexpected={sorted(observed-expected)}"
        )


def _phase(name: str, **details: Any) -> None:
    if name not in PHASE_ORDER:
        raise RuntimeError(f"unregistered D393 phase: {name}")
    _append_jsonl(
        PHASES,
        {
            "ordinal": sum(
                1
                for line in (
                    PHASES.read_text(encoding="utf-8").splitlines()
                    if PHASES.exists()
                    else []
                )
                if line.strip()
            ),
            "phase": name,
            "monotonic_ns": time.monotonic_ns(),
            "time_ns": time.time_ns(),
            **details,
        },
    )


def _deadline(label: str) -> None:
    if _deadline_monotonic is None:
        return
    if time.monotonic() > _deadline_monotonic:
        raise RuntimeError(f"D393 cooperative deadline exceeded at {label}")


def _direct_import_roots() -> set[str]:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".")[0])
    return result


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frac(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _frac_point(point: Sequence[float]) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(_frac(value) for value in point)  # type: ignore[return-value]


def _det3(rows: Sequence[Sequence[Fraction]]) -> Fraction:
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _tetra_det(points: Sequence[Sequence[float]]) -> Fraction:
    rows = [_frac_point(point) for point in points]
    base = rows[0]
    diffs = [
        tuple(row[index] - base[index] for index in range(3))
        for row in rows[1:]
    ]
    return _det3(diffs)


def _tetra_det_fraction(
    points: Sequence[Sequence[Fraction]],
) -> Fraction:
    base = points[0]
    diffs = [
        tuple(row[index] - base[index] for index in range(3))
        for row in points[1:]
    ]
    return _det3(diffs)


def _max_tetra(points: np.ndarray) -> dict[str, Any]:
    rows = np.asarray(points, dtype=np.float64)
    best_indices: tuple[int, int, int, int] | None = None
    best_det = Fraction(0)
    for indices in itertools.combinations(range(len(rows)), 4):
        _deadline("tetra_combination")
        determinant = _tetra_det(rows[list(indices)])
        if abs(determinant) > abs(best_det):
            best_det = determinant
            best_indices = indices
    return {
        "indices": list(best_indices) if best_indices is not None else None,
        "determinant_m3": best_det,
        "tetra_volume_m3": abs(best_det) / 6,
        "exact_rank3_witness": best_det != 0,
    }


def _polyhedron_edges(points: np.ndarray, options: str | None) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(source, qhull_options=options)
    groups: dict[tuple[float, ...], set[int]] = {}
    for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
        length = float(np.linalg.norm(equation[:3]))
        key = tuple(np.round(equation / length, decimals=7))
        groups.setdefault(key, set()).update(map(int, simplex))
    memberships: dict[tuple[int, int], int] = {}
    for vertices in groups.values():
        ordered = sorted(vertices)
        for position, left in enumerate(ordered):
            for right in ordered[position + 1 :]:
                pair = (left, right)
                memberships[pair] = memberships.get(pair, 0) + 1
    edges = sorted(pair for pair, count in memberships.items() if count >= 2)
    if not edges:
        raise RuntimeError("D393 edge reconstruction returned no edges")
    return np.asarray(edges, dtype=np.int64)


def _unique_hull_with_tokens(
    raw: np.ndarray,
    tokens: list[dict[str, Any]],
    *,
    options: str | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    array = np.asarray(raw, dtype=np.float64).reshape(-1, 3)
    unique, first = np.unique(array, axis=0, return_index=True)
    unique_tokens = [tokens[int(index)] for index in first]
    if len(unique) < 4:
        return unique, unique_tokens
    hull = ConvexHull(unique, qhull_options=options)
    indices = np.asarray(hull.vertices, dtype=np.int64)
    return unique[indices], [unique_tokens[int(index)] for index in indices]


def _clip_with_tokens(
    points: np.ndarray,
    tokens: list[dict[str, Any]],
    equation: np.ndarray,
    *,
    epsilon_m: float,
    options: str | None,
    clip_tag: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    source = np.asarray(points, dtype=np.float64)
    if len(tokens) != len(source):
        raise RuntimeError("token/point length mismatch")
    equation_copy = np.asarray(equation, dtype=np.float64).copy()
    equation_before = _array_sha(equation_copy.reshape(1, 4))
    normal = equation_copy[:3].copy()
    length = float(np.linalg.norm(normal))
    normal /= length
    offset = float(equation_copy[3]) / length
    # The frozen D389/D390 evidence was produced by this deterministic
    # component-order contraction on the installed NumPy stack.  BLAS-backed
    # matmul may change the final bit at ~1e-18 m and therefore cannot be used
    # to replay a branch boundary as bit-exact provenance.
    values = (
        np.einsum("ij,j->i", source, normal, optimize=False) + offset
    )
    keep = values <= epsilon_m
    output = [point.copy() for point in source[keep]]
    output_tokens = [
        {**tokens[index], "carried_by": clip_tag}
        for index in np.flatnonzero(keep)
    ]
    crossing_rows: list[dict[str, Any]] = []
    edges = _polyhedron_edges(source, options)
    for left_raw, right_raw in edges:
        left, right = int(left_raw), int(right_raw)
        v0, v1 = float(values[left]), float(values[right])
        if (v0 < -epsilon_m and v1 > epsilon_m) or (
            v1 < -epsilon_m and v0 > epsilon_m
        ):
            ratio = -v0 / (v1 - v0)
            point = source[left] + ratio * (source[right] - source[left])
            lineage = {
                "kind": "intersection",
                "clip_tag": clip_tag,
                "left_index": left,
                "right_index": right,
                "ratio_f64": ratio,
                "left_point_f64_m": source[left],
                "right_point_f64_m": source[right],
                "left_lineage": tokens[left],
                "right_lineage": tokens[right],
            }
            output.append(point)
            output_tokens.append(lineage)
            crossing_rows.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "ratio_f64": ratio,
                    "point_f64_m": point,
                }
            )
    equation_after = _array_sha(equation_copy.reshape(1, 4))
    return (
        np.asarray(output, dtype=np.float64).reshape(-1, 3),
        output_tokens,
        {
            "clip_tag": clip_tag,
            "source_count": len(source),
            "kept_count": int(np.count_nonzero(keep)),
            "edge_count": len(edges),
            "crossing_count": len(crossing_rows),
            "minimum_signed_value_m": float(values.min()),
            "maximum_signed_value_m": float(values.max()),
            "equation_sha256_before": equation_before,
            "equation_sha256_after": equation_after,
            "equation_input_immutable": equation_before == equation_after,
            "signed_distance_evaluation": (
                "numpy.einsum('ij,j->i', optimize=False) plus offset"
            ),
            "crossings": crossing_rows,
        },
    )


def _child_reconstruction_with_lineage(
    parent: np.ndarray,
    thin_axis: int,
    rotated_polygon: np.ndarray,
    keep_axes: list[int],
    start_state: int,
    end_state: int,
    *,
    child_name: str,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    polygon = np.vstack(
        [
            rotated_polygon[0],
            rotated_polygon[start_state + 1 : end_state + 2],
        ]
    )
    points = np.asarray(parent, dtype=np.float64)
    tokens = [
        {"kind": "parent", "parent_index": index}
        for index in range(len(points))
    ]
    events: list[dict[str, Any]] = []
    for clip_index in range(len(polygon)):
        _deadline(f"{child_name}_clip_{clip_index}")
        start = polygon[clip_index]
        end = polygon[(clip_index + 1) % len(polygon)]
        delta = end - start
        normal_2d = np.asarray([delta[1], -delta[0]], dtype=np.float64)
        offset = float(delta[0] * start[1] - delta[1] * start[0])
        normal_3d = np.zeros(3, dtype=np.float64)
        normal_3d[keep_axes] = normal_2d
        equation = np.r_[normal_3d, offset]
        raw, raw_tokens, metadata = _clip_with_tokens(
            points,
            tokens,
            equation,
            epsilon_m=5.0e-9,
            options=None,
            clip_tag=f"{child_name}_fan_clip_{clip_index}",
        )
        points, tokens = _unique_hull_with_tokens(
            raw, raw_tokens, options=None
        )
        events.append(
            {
                "clip_index": clip_index,
                "polygon_start_f64": start,
                "polygon_end_f64": end,
                "raw_normal_2d": normal_2d,
                "raw_offset": offset,
                "output_count": len(points),
                "output_sha256": _array_sha(points),
                "metadata": metadata,
            }
        )
    return points, tokens, events


def _find_call29_records() -> dict[str, Any]:
    d389 = _read_json(D389_EVIDENCE)
    d390_geometry = _read_json(D390_GEOMETRY)
    d391 = _read_json(D391_EVIDENCE)
    d392 = _read_json(D392_EVIDENCE)
    d390_records = [
        row
        for row in d390_geometry["records"]
        if row["call_index"] == CALL_INDEX and row["call_id"] == CALL_ID
    ]
    d391_records = [
        row
        for row in d391["disputed_records"]
        if row["call_index"] == CALL_INDEX and row["call_id"] == CALL_ID
    ]
    d392_records = [
        row
        for row in d392["same_rank_authority_41_coverage"][
            "combined_authority_vector"
        ]
        if row["call_index"] == CALL_INDEX and row["call_id"] == CALL_ID
    ]
    pair_rows = [
        row
        for row in d389["seam_numeric_provenance_audit"]["pair_results"]
        if row["target"] == "LOWER"
        and row["left_index"] == 1
        and row["right_index"] == 2
    ]
    if not (
        len(d390_records)
        == len(d391_records)
        == len(d392_records)
        == len(pair_rows)
        == 1
    ):
        raise RuntimeError("call29 frozen identity is not unique")
    return {
        "d390": d390_records[0],
        "d391": d391_records[0],
        "d392": d392_records[0],
        "d389_pair": pair_rows[0],
    }


def _exact_intersection_on_plane(
    left: Sequence[float],
    right: Sequence[float],
    normal: Sequence[float],
    offset: float,
) -> tuple[Fraction, Fraction, Fraction]:
    lhs = _frac_point(left)
    rhs = _frac_point(right)
    n = tuple(_frac(value) for value in normal)
    d = _frac(offset)
    left_value = sum(n[index] * lhs[index] for index in range(3)) + d
    denominator = sum(
        n[index] * (rhs[index] - lhs[index]) for index in range(3)
    )
    if denominator == 0:
        raise RuntimeError("exact shadow intersection denominator is zero")
    ratio = -left_value / denominator
    return tuple(
        lhs[index] + ratio * (rhs[index] - lhs[index])
        for index in range(3)
    )  # type: ignore[return-value]


def _source_children_and_seam() -> dict[str, Any]:
    d388_evidence = _read_json(D388_EVIDENCE)
    d388_geometry = _read_json(D388_GEOMETRY)
    module = _load_module(D389_SCRIPT, "_d393_frozen_d389")
    spec = next(row for row in module.TARGETS if row["short"] == "LOWER")
    target = next(
        row
        for row in d388_evidence["target_results"]
        if row["prim_name"] == spec["prim"]
    )
    layer = next(
        row
        for row in d388_geometry["layers"]
        if row["prim_name"] == spec["prim"]
    )
    if target["selected_threshold_cover"]["cut_states"] != spec["cuts"]:
        raise RuntimeError("D393 frozen LOWER cut states changed")
    parent = np.asarray(
        layer["parent_layer"]["vertices_f64_m"], dtype=np.float64
    )
    thin_axis = int(target["thin_axis_index"])
    polygon, keep_axes = module._profile(parent, thin_axis)
    rotated = np.roll(polygon, -int(spec["anchor"]), axis=0)
    cuts = list(map(int, spec["cuts"]))
    child1, tokens1, events1 = _child_reconstruction_with_lineage(
        parent,
        thin_axis,
        rotated,
        keep_axes,
        cuts[1],
        cuts[2],
        child_name="lower_child1",
    )
    child2, tokens2, events2 = _child_reconstruction_with_lineage(
        parent,
        thin_axis,
        rotated,
        keep_axes,
        cuts[2],
        cuts[3],
        child_name="lower_child2",
    )
    frozen = _find_call29_records()["d390"]
    source = np.asarray(frozen["source_vertices_f64_m"], dtype=np.float64)
    clipping = np.asarray(
        frozen["clipping_vertices_f64_m"], dtype=np.float64
    )
    source_sorted = np.unique(child1, axis=0)
    clipping_sorted = np.unique(child2, axis=0)
    if not np.array_equal(source_sorted, source):
        raise RuntimeError("D393 child1 does not reproduce D390 source stream")
    if not np.array_equal(clipping_sorted, clipping):
        raise RuntimeError(
            "D393 child2 does not reproduce D390 clipping stream"
        )

    source_tokens: list[dict[str, Any]] = []
    for point in source_sorted:
        matches = [
            index
            for index, candidate in enumerate(child1)
            if np.array_equal(candidate, point)
        ]
        if len(matches) != 1:
            raise RuntimeError("D393 source point lineage is not unique")
        source_tokens.append(tokens1[matches[0]])
    clipping_tokens: list[dict[str, Any]] = []
    for point in clipping_sorted:
        matches = [
            index
            for index, candidate in enumerate(child2)
            if np.array_equal(candidate, point)
        ]
        if len(matches) != 1:
            raise RuntimeError("D393 clipping point lineage is not unique")
        clipping_tokens.append(tokens2[matches[0]])

    quartet = source_sorted[SOURCE_QUARTET_INDICES]
    quartet_tokens = [source_tokens[index] for index in SOURCE_QUARTET_INDICES]
    final_event = events1[-1]
    final_tag = f"lower_child1_fan_clip_{len(events1)-1}"
    kinds = [
        "final_intersection"
        if token.get("kind") == "intersection"
        and token.get("clip_tag") == final_tag
        else "parent_or_carried"
        for token in quartet_tokens
    ]
    if kinds.count("final_intersection") != 3:
        raise RuntimeError(
            "D393 expected exactly three final fan-clip intersections"
        )
    if kinds.count("parent_or_carried") != 1:
        raise RuntimeError("D393 expected one exact carried parent point")

    normal_3d = np.zeros(3, dtype=np.float64)
    normal_3d[keep_axes] = np.asarray(
        final_event["raw_normal_2d"], dtype=np.float64
    )
    offset = float(final_event["raw_offset"])
    normal_frac = tuple(_frac(value) for value in normal_3d)
    offset_frac = _frac(offset)
    residuals: list[Fraction] = []
    analytic_quartet: list[tuple[Fraction, Fraction, Fraction]] = []
    for point, token, kind in zip(
        quartet, quartet_tokens, kinds, strict=True
    ):
        point_frac = _frac_point(point)
        residual = (
            sum(normal_frac[index] * point_frac[index] for index in range(3))
            + offset_frac
        )
        residuals.append(residual)
        if kind == "final_intersection":
            analytic_quartet.append(
                _exact_intersection_on_plane(
                    token["left_point_f64_m"],
                    token["right_point_f64_m"],
                    normal_3d,
                    offset,
                )
            )
        else:
            if residual != 0:
                raise RuntimeError(
                    "D393 carried seam parent is not exactly on raw plane"
                )
            analytic_quartet.append(point_frac)
    analytic_det = _tetra_det_fraction(analytic_quartet)
    stored_det = _tetra_det(quartet)
    length = float(np.linalg.norm(normal_3d))
    normalized_abs_residuals_m = [
        abs(float(value)) / length for value in residuals
    ]
    source_max = _max_tetra(source_sorted)
    if not source_max["exact_rank3_witness"]:
        raise RuntimeError("D393 non-seam source-child rank3 control failed")
    return {
        "parent": parent,
        "thin_axis": thin_axis,
        "keep_axes": keep_axes,
        "rotated_polygon": rotated,
        "cuts": cuts,
        "child1": child1,
        "child2": child2,
        "source_sorted": source_sorted,
        "clipping_sorted": clipping_sorted,
        "source_tokens": source_tokens,
        "clipping_tokens": clipping_tokens,
        "child1_events": events1,
        "child2_events": events2,
        "quartet": quartet,
        "quartet_tokens": quartet_tokens,
        "quartet_kinds": kinds,
        "final_fan_clip_tag": final_tag,
        "raw_seam_plane": {
            "normal_f64": normal_3d,
            "offset_f64": offset,
            "normal_exact": normal_frac,
            "offset_exact": offset_frac,
            "stored_quartet_residuals_exact": residuals,
            "stored_quartet_normalized_abs_residuals_m": (
                normalized_abs_residuals_m
            ),
            "maximum_normalized_abs_residual_m": max(
                normalized_abs_residuals_m
            ),
        },
        "stored_quartet_determinant_m3": stored_det,
        "stored_quartet_tetra_volume_m3": abs(stored_det) / 6,
        "analytic_shadow_quartet": analytic_quartet,
        "analytic_shadow_determinant_m3": analytic_det,
        "analytic_shadow_tetra_volume_m3": abs(analytic_det) / 6,
        "source_child_max_tetra": source_max,
        "source_child_convex_hull_volume_m3": float(
            ConvexHull(source_sorted).volume
        ),
        "checks": {
            "child1_stream_exact": _array_sha(source_sorted)
            == EXPECTED_SOURCE_SHA256,
            "child2_stream_exact": _array_sha(clipping_sorted)
            == EXPECTED_CLIPPING_SHA256,
            "source_quartet_exact_rank3": stored_det != 0,
            "registered_witness_contains_three_final_fan_clip_intersections": (
                kinds.count("final_intersection") == 3
                and kinds.count("parent_or_carried") == 1
            ),
            "analytic_rational_shadow_rank2_zero_tetra": analytic_det == 0,
            "non_seam_source_child_remains_exact_rank3": source_max[
                "exact_rank3_witness"
            ],
            "all_child_plane_inputs_immutable": all(
                event["metadata"]["equation_input_immutable"]
                for event in [*events1, *events2]
            ),
        },
    }


def _replay_call29(
    source_info: dict[str, Any],
) -> dict[str, Any]:
    frozen = _find_call29_records()["d390"]
    trace = frozen["plane_trace"]
    if len(trace) != 21 or _canonical_sha(trace) != EXPECTED_PLANE_TRACE_SHA256:
        raise RuntimeError("D393 frozen call29 plane trace identity failed")
    points = np.asarray(source_info["source_sorted"], dtype=np.float64)
    tokens = [
        {
            "kind": "d389_source_vertex",
            "source_sorted_index": index,
            "source_lineage": token,
        }
        for index, token in enumerate(source_info["source_tokens"])
    ]
    active_events: list[dict[str, Any]] = []
    skip_diagnostics: list[dict[str, Any]] = []
    trace_input_checks: list[dict[str, Any]] = []
    terminal: np.ndarray | None = None
    terminal_tokens: list[dict[str, Any]] | None = None
    for row in trace:
        plane_index = int(row["selected_plane_index_zero_based"])
        _deadline(f"call29_plane_{plane_index}")
        equation = np.asarray(
            row["plane_equation_f64_m"], dtype=np.float64
        )
        trace_input_check = {
            "plane_index": plane_index,
            "points_before_count_exact": (
                len(points) == row["points_before_count"]
            ),
            "points_before_sha_exact": (
                _array_sha(points) == row["points_before_sha256"]
            ),
            "plane_equation_sha_exact": (
                _array_sha(equation.reshape(1, 4))
                == row["plane_equation_sha256"]
            ),
        }
        trace_input_check["pass"] = all(
            value
            for key, value in trace_input_check.items()
            if key != "plane_index"
        )
        trace_input_checks.append(trace_input_check)
        if not trace_input_check["pass"]:
            raise RuntimeError(
                f"D393 frozen trace input changed at plane {plane_index}"
            )
        values = points @ equation[:3] + equation[3]
        if row["branch"] == "SKIP_INSIDE":
            deterministic_values = (
                np.einsum(
                    "ij,j->i",
                    points,
                    equation[:3],
                    optimize=False,
                )
                + equation[3]
            )
            skip_diagnostics.append(
                {
                    "plane_index": plane_index,
                    "frozen_branch_followed": "SKIP_INSIDE",
                    "frozen_minimum_signed_value_m": row[
                        "minimum_signed_value_m"
                    ],
                    "frozen_maximum_signed_value_m": row[
                        "maximum_signed_value_m"
                    ],
                    "current_matmul_minimum_signed_value_m": float(
                        values.min()
                    ),
                    "current_matmul_maximum_signed_value_m": float(
                        values.max()
                    ),
                    "current_deterministic_einsum_minimum_m": float(
                        deterministic_values.min()
                    ),
                    "current_deterministic_einsum_maximum_m": float(
                        deterministic_values.max()
                    ),
                    "branch_not_redecided_from_recomputed_sign": True,
                    "diagnostic_only": True,
                }
            )
            continue
        if row["branch"] not in {"CLIP_CONTINUE", "COLLAPSE"}:
            raise RuntimeError(
                f"D393 unexpected frozen branch {row['branch']}"
            )
        active_ordinal = len(active_events) + 1
        raw, raw_tokens, metadata = _clip_with_tokens(
            points,
            tokens,
            equation,
            epsilon_m=0.0,
            options="Q12 Pp",
            clip_tag=f"d390_active_{active_ordinal}_plane_{plane_index}",
        )
        unique, first = np.unique(raw, axis=0, return_index=True)
        unique_tokens = [raw_tokens[int(index)] for index in first]
        candidate_sha = _array_sha(unique)
        event = {
            "active_ordinal": active_ordinal,
            "plane_index": plane_index,
            "branch": row["branch"],
            "input_count": len(points),
            "input_sha256": _array_sha(points),
            "candidate_count": len(unique),
            "candidate_sha256": candidate_sha,
            "candidate_points_f64_m": unique,
            "candidate_immediate_lineages": unique_tokens,
            "frozen_candidate_sha256": row[
                "candidate_unique_points_sha256"
            ],
            "frozen_minimum_signed_value_m": row[
                "minimum_signed_value_m"
            ],
            "frozen_maximum_signed_value_m": row[
                "maximum_signed_value_m"
            ],
            "current_blas_matmul_minimum_signed_value_m_diagnostic": (
                float(values.min())
            ),
            "current_blas_matmul_maximum_signed_value_m_diagnostic": (
                float(values.max())
            ),
            "candidate_replay_signed_distance_contract": (
                "einsum optimize=False; BLAS matmul shown only as "
                "allocation-sensitive diagnostic"
            ),
            "metadata": metadata,
        }
        if candidate_sha != row["candidate_unique_points_sha256"]:
            raise RuntimeError(
                f"D393 active candidate mismatch at plane {plane_index}"
            )
        active_events.append(event)
        if row["branch"] == "COLLAPSE":
            terminal = unique
            terminal_tokens = unique_tokens
            break
        hull = ConvexHull(unique, qhull_options="Q12 Pp")
        hull_indices = np.asarray(hull.vertices, dtype=np.int64)
        points = unique[hull_indices]
        tokens = [unique_tokens[int(index)] for index in hull_indices]
    if terminal is None or terminal_tokens is None:
        raise RuntimeError("D393 call29 did not reach frozen terminal")
    frozen_terminal = np.asarray(
        frozen["terminal_candidate_unique_points_f64_m"], dtype=np.float64
    )
    if not np.array_equal(terminal, frozen_terminal):
        raise RuntimeError("D393 terminal points are not bit-exact")
    active_planes = [row["plane_index"] for row in active_events]
    if active_planes != EXPECTED_ACTIVE_PLANES:
        raise RuntimeError("D393 active plane sequence changed")
    immediate_kinds = [
        (
            "terminal_intersection"
            if token.get("kind") == "intersection"
            and token.get("clip_tag") == "d390_active_3_plane_20"
            else "terminal_carried"
        )
        for token in terminal_tokens
    ]
    pair_rows: list[dict[str, Any]] = []
    for left, right in ((1, 2), (3, 4)):
        delta = terminal[left] - terminal[right]
        pair_rows.append(
            {
                "indices": [left, right],
                "delta_f64_m": delta,
                "distance_m": float(np.linalg.norm(delta)),
                "delta_exact": [
                    _frac(terminal[left, axis])
                    - _frac(terminal[right, axis])
                    for axis in range(3)
                ],
            }
        )
    terminal_max = _max_tetra(terminal)
    if terminal_max["indices"] is None:
        raise RuntimeError("D393 terminal max tetra witness is absent")

    def source_indices(token: dict[str, Any]) -> set[int]:
        if token.get("kind") == "d389_source_vertex":
            return {int(token["source_sorted_index"])}
        if token.get("kind") == "intersection":
            return (
                source_indices(token["left_lineage"])
                | source_indices(token["right_lineage"])
            )
        raise RuntimeError(
            f"D393 unknown D390 terminal lineage token: {token.get('kind')}"
        )

    terminal_source_ancestry = [
        sorted(source_indices(token)) for token in terminal_tokens
    ]
    terminal_max_ancestry = sorted(
        set().union(
            *(
                set(terminal_source_ancestry[index])
                for index in terminal_max["indices"]
            )
        )
    )
    final_tag = source_info["final_fan_clip_tag"]
    final_fan_indices = sorted(
        index
        for index in terminal_max_ancestry
        if source_info["source_tokens"][index].get("kind")
        == "intersection"
        and source_info["source_tokens"][index].get("clip_tag")
        == final_tag
    )
    plane = source_info["raw_seam_plane"]
    normal_exact = plane["normal_exact"]
    offset_exact = plane["offset_exact"]
    exact_seam_candidates: list[int] = []
    for index in terminal_max_ancestry:
        point = _frac_point(source_info["source_sorted"][index])
        residual = (
            sum(
                normal_exact[axis] * point[axis]
                for axis in range(3)
            )
            + offset_exact
        )
        if residual == 0 and index not in final_fan_indices:
            exact_seam_candidates.append(index)
    selected_anchor = (
        min(exact_seam_candidates)
        if exact_seam_candidates
        else None
    )
    derived_registered_quartet = (
        [selected_anchor, *final_fan_indices]
        if selected_anchor is not None
        else []
    )
    source_volume = float(source_info["source_child_convex_hull_volume_m3"])
    # _max_tetra returns an exact Fraction until _native serialization.
    terminal_volume = float(terminal_max["tetra_volume_m3"])
    return {
        "terminal": terminal,
        "terminal_tokens": terminal_tokens,
        "terminal_immediate_kinds": immediate_kinds,
        "active_events": active_events,
        "skip_diagnostics": skip_diagnostics,
        "trace_input_checks": trace_input_checks,
        "near_duplicate_pairs": pair_rows,
        "terminal_max_tetra": terminal_max,
        "terminal_source_ancestry_by_index": terminal_source_ancestry,
        "terminal_max_tetra_source_ancestry_union": (
            terminal_max_ancestry
        ),
        "terminal_max_ancestry_final_fan_intersection_indices": (
            final_fan_indices
        ),
        "terminal_max_ancestry_exact_seam_carried_candidates": (
            exact_seam_candidates
        ),
        "registered_witness_anchor_selection": (
            "minimum frozen source-sorted index among exact-seam carried "
            "candidates in terminal-max ancestry"
        ),
        "derived_registered_quartet_indices": (
            derived_registered_quartet
        ),
        "terminal_max_tetra_to_source_child_volume_ratio": (
            terminal_volume / source_volume
        ),
        "checks": {
            "plane_trace_exact_21": len(trace) == 21,
            "all_trace_inputs_hash_and_count_exact": all(
                row["pass"] for row in trace_input_checks
            )
            and len(trace_input_checks) == 21,
            "frozen_skip_branches_followed_without_sign_redecision": (
                len(skip_diagnostics) == 18
                and all(
                    row["branch_not_redecided_from_recomputed_sign"]
                    for row in skip_diagnostics
                )
            ),
            "active_planes_exact_3_15_20": active_planes
            == EXPECTED_ACTIVE_PLANES,
            "active_candidate_hashes_exact": all(
                row["candidate_sha256"] == row["frozen_candidate_sha256"]
                for row in active_events
            ),
            "terminal_sha_exact": _array_sha(terminal)
            == EXPECTED_TERMINAL_SHA256,
            "terminal_immediate_four_carried_two_intersections": (
                immediate_kinds.count("terminal_carried") == 4
                and immediate_kinds.count("terminal_intersection") == 2
            ),
            "near_duplicate_pair_deltas_bit_exact_equal": np.array_equal(
                terminal[1] - terminal[2], terminal[3] - terminal[4]
            ),
            "each_near_duplicate_pair_is_one_carried_one_intersection": all(
                {
                    immediate_kinds[left],
                    immediate_kinds[right],
                }
                == {"terminal_carried", "terminal_intersection"}
                for left, right in ((1, 2), (3, 4))
            ),
            "terminal_max_tetra_indices_exact": (
                terminal_max["indices"] == [0, 1, 4, 5]
            ),
            "terminal_max_ancestry_final_fan_indices_exact": (
                final_fan_indices == [3, 4, 6]
            ),
            "terminal_max_ancestry_exact_seam_candidates_exact": (
                exact_seam_candidates == [0, 16]
            ),
            "derived_registered_quartet_exact": (
                derived_registered_quartet == SOURCE_QUARTET_INDICES
            ),
            "all_d390_plane_inputs_immutable": all(
                row["metadata"]["equation_input_immutable"]
                for row in active_events
            ),
        },
    }


def _scope_counters() -> dict[str, Any]:
    return {
        "source_child_streams_reconstructed": 2,
        "call_ids_replayed": 1,
        "replayed_call_indices": [CALL_INDEX],
        "frozen_plane_rows_checked": 21,
        "active_clip_events": 3,
        "other_calls_replayed": 0,
        "pair_sweeps": 0,
        "global_path_enumerations_or_applications": 0,
        "strict_lp_or_overlap_volume_solver_calls": 0,
        "rank_or_class_adoptions": 0,
        "seam_verdict_updates": 0,
        "partition_budget_geometry_changes": 0,
        "collider_asset_usd_materializations": 0,
        "isaac_kit_physx_warp_cuda_launches": 0,
        "cylinder_physics_q5_contact_grasp": 0,
        "target_ik_path_changes": 0,
        "process_signals_sent": 0,
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    records = _find_call29_records()
    source = _source_children_and_seam()
    _append_jsonl(
        PROGRESS,
        {
            "ordinal": 0,
            "event": "source_streams_bound",
            "source_sha256": _array_sha(source["source_sorted"]),
            "clipping_sha256": _array_sha(source["clipping_sorted"]),
            "pass": all(source["checks"].values()),
        },
    )
    _phase(
        "call29_source_streams_bound",
        source_sha256=_array_sha(source["source_sorted"]),
        clipping_sha256=_array_sha(source["clipping_sorted"]),
    )
    replay = _replay_call29(source)
    for event in replay["active_events"]:
        ordinal = int(event["active_ordinal"])
        _append_jsonl(
            PROGRESS,
            {
                "ordinal": ordinal,
                "event": f"active_clip_{ordinal}",
                "plane_index": event["plane_index"],
                "candidate_sha256": event["candidate_sha256"],
                "pass": (
                    event["candidate_sha256"]
                    == event["frozen_candidate_sha256"]
                ),
            },
        )
        if ordinal in {1, 2}:
            _phase(
                f"call29_active_clip_{ordinal}_lineage_committed",
                plane_index=event["plane_index"],
                candidate_sha256=event["candidate_sha256"],
            )
        else:
            _phase(
                "call29_terminal_lineage_committed",
                plane_index=event["plane_index"],
                terminal_sha256=event["candidate_sha256"],
            )

    d391_authority = records["d391"]["rank_authority"]
    d392_authority = records["d392"]
    authority_preserved = {
        "d391_status": d391_authority["status"],
        "d391_authoritative_rank": d391_authority[
            "authoritative_rank"
        ],
        "d391_authoritative_class": d391_authority[
            "authoritative_class"
        ],
        "d392_status": d392_authority["status"],
        "d392_authoritative_rank": d392_authority[
            "authoritative_rank"
        ],
        "d392_authoritative_class": d392_authority[
            "authoritative_class"
        ],
        "all_41_aggregate": _read_json(D392_EVIDENCE)[
            "same_rank_authority_41_coverage"
        ]["authoritative_all_41_class_aggregate"],
    }
    authority_null = (
        authority_preserved["d391_status"]
        == "NUMERICALLY_AMBIGUOUS_BASIS"
        and authority_preserved["d392_status"]
        == "NUMERICALLY_AMBIGUOUS_BASIS"
        and authority_preserved["d391_authoritative_rank"] is None
        and authority_preserved["d391_authoritative_class"] is None
        and authority_preserved["d392_authoritative_rank"] is None
        and authority_preserved["d392_authoritative_class"] is None
        and authority_preserved["all_41_aggregate"] is None
    )
    pair = records["d389_pair"]
    frozen_pair_nonclaim = {
        "pair_index": 21,
        "target": pair["target"],
        "left_index": pair["left_index"],
        "right_index": pair["right_index"],
        "pre_float32_strict_volume_m3": pair["pre_float32_epsilon0"][
            "volume_m3"
        ],
        "pre_float32_strict_positive_volume": pair[
            "pre_float32_epsilon0"
        ]["positive_volume"],
        "pair_classification_remains": pair[
            "per_pair_classification"
        ],
        "d393_does_not_update_pair": True,
    }
    source_checks = dict(source["checks"])
    replay_checks = dict(replay["checks"])
    checks = {
        "input_hashes_exact": all(
            path.is_file() and _sha(path) == expected
            for path, expected in EXPECTED_INPUT_SHA256.items()
        ),
        "d389_script_exact": _sha(D389_SCRIPT)
        == EXPECTED_D389_SCRIPT_SHA256,
        "call_identity_exact": (
            records["d390"]["call_index"] == CALL_INDEX
            and records["d390"]["call_id"] == CALL_ID
            and records["d390"]["target"] == "LOWER"
            and records["d390"]["stage"] == CALL_STAGE
            and records["d390"]["direction"] == CALL_DIRECTION
            and records["d390"]["source_child_index"] == 1
            and records["d390"]["clipping_child_index"] == 2
            and records["d389_pair"]["target"] == "LOWER"
            and records["d389_pair"]["left_index"] == 1
            and records["d389_pair"]["right_index"] == 2
        ),
        "source_provenance_pass": all(source_checks.values()),
        "call29_replay_pass": all(replay_checks.values()),
        "d391_d392_null_authority_preserved": authority_null,
        "analytic_shadow_negative_control_zero": source[
            "analytic_shadow_determinant_m3"
        ]
        == 0,
        "non_seam_source_rank3_negative_control": source[
            "source_child_max_tetra"
        ]["exact_rank3_witness"],
        "scope_exact": _scope_counters()
        == {
            "source_child_streams_reconstructed": 2,
            "call_ids_replayed": 1,
            "replayed_call_indices": [29],
            "frozen_plane_rows_checked": 21,
            "active_clip_events": 3,
            "other_calls_replayed": 0,
            "pair_sweeps": 0,
            "global_path_enumerations_or_applications": 0,
            "strict_lp_or_overlap_volume_solver_calls": 0,
            "rank_or_class_adoptions": 0,
            "seam_verdict_updates": 0,
            "partition_budget_geometry_changes": 0,
            "collider_asset_usd_materializations": 0,
            "isaac_kit_physx_warp_cuda_launches": 0,
            "cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_changes": 0,
            "process_signals_sent": 0,
        },
    }
    pass_contract = all(checks.values())
    if pass_contract:
        verdict = (
            "D393_CALL29_REGISTERED_MICRO_THIRD_DIRECTION_WITNESS_ALREADY_"
            "PRESENT_AFTER_D389_FINAL_FAN_CLIP_PASS"
        )
        conclusion = (
            "The registered call29 source-child witness is already "
            "non-coplanar after the D389 final fan clip that creates three "
            "of its points. The exact-rational raw semantic-plane shadow is "
            "planar. This supports residue from the full Float64 fan-clip "
            "intersection pipeline; it does not prove the earliest micro-"
            "rank3 event across every possible quartet or physical "
            "manufacturing thickness."
        )
    else:
        verdict = "D393_PROVENANCE_LOCALIZATION_FAIL_STOP"
        conclusion = None
    evidence = {
        "artifact": "D393_CALL29_PROVENANCE_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "call_identity": {
            "call_index": CALL_INDEX,
            "call_id": CALL_ID,
            "target": "LOWER",
            "pair_index": 21,
            "left_child": 1,
            "right_child": 2,
            "stage": CALL_STAGE,
            "direction": CALL_DIRECTION,
        },
        "d389_shared_fan_seam_lineage": {
            "source_vertex_count": len(source["source_sorted"]),
            "clipping_vertex_count": len(source["clipping_sorted"]),
            "source_sha256": _array_sha(source["source_sorted"]),
            "clipping_sha256": _array_sha(source["clipping_sorted"]),
            "source_quartet_indices": SOURCE_QUARTET_INDICES,
            "source_quartet_f64_m": source["quartet"],
            "source_quartet_lineage_kind": source["quartet_kinds"],
            "stored_quartet_determinant_m3": source[
                "stored_quartet_determinant_m3"
            ],
            "stored_quartet_tetra_volume_m3": source[
                "stored_quartet_tetra_volume_m3"
            ],
            "raw_seam_plane": source["raw_seam_plane"],
            "analytic_shadow_determinant_m3": source[
                "analytic_shadow_determinant_m3"
            ],
            "analytic_shadow_tetra_volume_m3": source[
                "analytic_shadow_tetra_volume_m3"
            ],
            "registered_witness_first_defined_at_event": (
                "D389 lower child1 final shared fan-ray clip; three of the "
                "four registered witness points are created there, so this "
                "specific quartet cannot exist earlier"
            ),
            "registered_witness_selection": {
                "indices_in_frozen_sorted_call29_source": (
                    SOURCE_QUARTET_INDICES
                ),
                "unique_or_maximal_among_all_quartets": False,
                "earliest_micro_rank3_across_every_possible_quartet": None,
            },
            "source_child_max_tetra": source[
                "source_child_max_tetra"
            ],
            "source_child_convex_hull_volume_m3": source[
                "source_child_convex_hull_volume_m3"
            ],
            "checks": source_checks,
        },
        "d390_carry_and_near_duplicate_decomposition": {
            "plane_trace_count": 21,
            "active_plane_indices": [
                row["plane_index"] for row in replay["active_events"]
            ],
            "active_events": replay["active_events"],
            "skip_recomputation_diagnostics": replay[
                "skip_diagnostics"
            ],
            "trace_input_checks": replay["trace_input_checks"],
            "terminal_sha256": _array_sha(replay["terminal"]),
            "terminal_points_f64_m": replay["terminal"],
            "terminal_immediate_lineage_kind": replay[
                "terminal_immediate_kinds"
            ],
            "near_duplicate_pairs": replay["near_duplicate_pairs"],
            "terminal_max_tetra": replay["terminal_max_tetra"],
            "terminal_source_ancestry_by_index": replay[
                "terminal_source_ancestry_by_index"
            ],
            "terminal_max_tetra_source_ancestry_union": replay[
                "terminal_max_tetra_source_ancestry_union"
            ],
            "terminal_max_ancestry_final_fan_intersection_indices": replay[
                "terminal_max_ancestry_final_fan_intersection_indices"
            ],
            "terminal_max_ancestry_exact_seam_carried_candidates": replay[
                "terminal_max_ancestry_exact_seam_carried_candidates"
            ],
            "registered_witness_anchor_selection": replay[
                "registered_witness_anchor_selection"
            ],
            "derived_registered_quartet_indices": replay[
                "derived_registered_quartet_indices"
            ],
            "terminal_max_tetra_to_source_child_volume_ratio": replay[
                "terminal_max_tetra_to_source_child_volume_ratio"
            ],
            "interpretation": (
                "The registered non-coplanar D389 source witness already "
                "exists before D390. D390 carries that source lineage and "
                "produces two near-duplicate crossing points. This does not "
                "claim the earliest micro-rank3 event among every quartet."
            ),
            "checks": replay_checks,
        },
        "frozen_rank_authority": authority_preserved,
        "frozen_pair_nonclaim": frozen_pair_nonclaim,
        "scope_counters": _scope_counters(),
        "case_checks": checks,
        "numeric_integrity_pass": pass_contract,
        "diagnostic_conclusion": conclusion,
        "numeric_verdict": verdict,
        "algorithm_elapsed_seconds": time.monotonic() - started,
        "nonclaims": {
            "physical_manufacturing_thickness_measured": False,
            "float64_interpolation_alone_isolated_as_single_cause": False,
            "earliest_micro_rank3_across_all_child_clip_quartets": None,
            "registered_witness_is_unique_or_maximal": False,
            "call29_rank_or_class_adopted": None,
            "authoritative_all_41_aggregate": None,
            "d389_or_d390_repaired_or_retroactively_passed": False,
            "seam_or_pair_verdict_updated": False,
            "positive_overlap_or_zero_volume_adopted": None,
            "collider_usd_isaac_physx_cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_changes": 0,
            "g0a_pass": False,
        },
    }
    geometry = {
        "artifact": "D393_CALL29_LINEAGE_GEOMETRY_V1",
        "source_child1_f64_m": source["source_sorted"],
        "clipping_child2_f64_m": source["clipping_sorted"],
        "d389_source_quartet_f64_m": source["quartet"],
        "d389_analytic_shadow_quartet_f64_m": np.asarray(
            [
                [float(value) for value in point]
                for point in source["analytic_shadow_quartet"]
            ],
            dtype=np.float64,
        ),
        "d390_terminal_six_f64_m": replay["terminal"],
        "d390_active_stages": [
            {
                "active_ordinal": row["active_ordinal"],
                "plane_index": row["plane_index"],
                "branch": row["branch"],
                "candidate_points_f64_m": row[
                    "candidate_points_f64_m"
                ],
                "candidate_sha256": row["candidate_sha256"],
            }
            for row in replay["active_events"]
        ],
        "terminal_lineages": replay["terminal_tokens"],
        "raw_seam_plane_normal_f64": source["raw_seam_plane"][
            "normal_f64"
        ],
        "raw_seam_plane_offset_f64": source["raw_seam_plane"][
            "offset_f64"
        ],
        "canonical_evidence_sha256": None,
        "viewer_role": (
            "Float32 inspection copies only; never hash back into numeric gate"
        ),
    }
    csv_rows: list[dict[str, Any]] = []
    for index, (point, token, kind) in enumerate(
        zip(
            replay["terminal"],
            replay["terminal_tokens"],
            replay["terminal_immediate_kinds"],
            strict=True,
        )
    ):
        csv_rows.append(
            {
                "terminal_index": index,
                "x_m": float(point[0]),
                "y_m": float(point[1]),
                "z_m": float(point[2]),
                "immediate_lineage": kind,
                "clip_tag": token.get("clip_tag"),
                "kind": token.get("kind"),
            }
        )
    return evidence, geometry, csv_rows


def _status_path(line: str) -> str:
    path = line[3:] if len(line) >= 4 else ""
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"')


def _status_scope_pass(
    current: Sequence[str],
    baseline: Sequence[str],
    *,
    allowed_output_names: set[str],
) -> bool:
    baseline_set = set(baseline)
    if not baseline_set.issubset(set(current)):
        return False
    allowed_paths = {
        f"{_rel(OUT_DIR)}/{name}" for name in allowed_output_names
    }
    extras = [line for line in current if line not in baseline_set]
    return all(
        line.startswith("?? ") and _status_path(line) in allowed_paths
        for line in extras
    )


def _safe_read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return _read_json(path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _artifact_reference_exact(reference: Any, path: Path) -> bool:
    return (
        isinstance(reference, dict)
        and path.is_file()
        and reference == _artifact(path)
    )


def _expected_authority_inputs() -> dict[str, str]:
    result = {
        _rel(path): expected
        for path, expected in EXPECTED_INPUT_SHA256.items()
    }
    result[_rel(D389_SCRIPT)] = EXPECTED_D389_SCRIPT_SHA256
    return result


def _frozen_checks() -> dict[str, bool]:
    return {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_master_exact": (
            _git("rev-parse", "origin/master") == EXPECTED_HEAD
        ),
        "start_exact": (
            START.is_file()
            and _sha(START) == EXPECTED_START_SHA256
        ),
        "active_case_marker_exact": (
            "## Active Case — D393 Call29 Provenance Localization"
            in START.read_text(encoding="utf-8")
        ),
        "all_input_hashes_exact": all(
            path.is_file() and _sha(path) == expected
            for path, expected in EXPECTED_INPUT_SHA256.items()
        ),
        "d389_script_exact": (
            D389_SCRIPT.is_file()
            and _sha(D389_SCRIPT) == EXPECTED_D389_SCRIPT_SHA256
        ),
    }


def _schema_checks() -> dict[str, bool]:
    records = _find_call29_records()
    frozen = records["d390"]
    trace = frozen["plane_trace"]
    active = [
        row["selected_plane_index_zero_based"]
        for row in trace
        if row["branch"] != "SKIP_INSIDE"
    ]
    d391 = records["d391"]["rank_authority"]
    d392 = records["d392"]
    return {
        "call_index_id_exact": (
            frozen["call_index"] == CALL_INDEX
            and frozen["call_id"] == CALL_ID
        ),
        "stage_direction_exact": (
            frozen["stage"] == CALL_STAGE
            and frozen["direction"] == CALL_DIRECTION
        ),
        "source_sha_exact": (
            frozen["source_vertices_sha256"]
            == EXPECTED_SOURCE_SHA256
        ),
        "clipping_sha_exact": (
            frozen["clipping_vertices_sha256"]
            == EXPECTED_CLIPPING_SHA256
        ),
        "terminal_sha_exact": (
            frozen["terminal_candidate_unique_points_sha256"]
            == EXPECTED_TERMINAL_SHA256
        ),
        "trace_exact_21": (
            len(trace) == 21
            and _canonical_sha(trace) == EXPECTED_PLANE_TRACE_SHA256
        ),
        "active_planes_exact": active == EXPECTED_ACTIVE_PLANES,
        "d391_null_authority_exact": (
            d391["status"] == "NUMERICALLY_AMBIGUOUS_BASIS"
            and d391["authoritative_rank"] is None
            and d391["authoritative_class"] is None
        ),
        "d392_null_authority_exact": (
            d392["status"] == "NUMERICALLY_AMBIGUOUS_BASIS"
            and d392["authoritative_rank"] is None
            and d392["authoritative_class"] is None
        ),
    }


def _execution_authority_checks(
    authority: dict[str, Any], *, exact_status: bool
) -> dict[str, bool]:
    baseline = authority.get("git", {}).get(
        "status_with_execution_authority", []
    )
    current = _status_lines()
    authority_line = f"?? {_rel(EXECUTION_AUTHORITY)}"
    return {
        "artifact_exact": (
            authority.get("artifact")
            == "D393_EXTERNAL_EXECUTION_AUTHORITY_V1"
        ),
        "case_attempt_exact": (
            authority.get("case") == CASE
            and authority.get("attempt") == ATTEMPT
        ),
        "approval_exact": (
            authority.get("approval", {}).get("normalized_user_text")
            == EXPECTED_USER_APPROVAL_NORMALIZED
            and authority.get("approval", {}).get(
                "normalized_user_text_sha256"
            )
            == _text_sha(EXPECTED_USER_APPROVAL_NORMALIZED)
            and authority.get("approval", {}).get("interpretation")
            == (
                "conditional sequential approval; D393 is the only active "
                "case and each later case must freeze the prior result and "
                "preregister one or two variables"
            )
            and authority.get("approval", {}).get("new_variables")
            == VARIABLES
        ),
        "script_exact": (
            authority.get("script", {}).get("path") == _rel(SCRIPT)
            and authority.get("script", {}).get("sha256") == _sha(SCRIPT)
        ),
        "start_exact": (
            authority.get("start", {}).get("path") == _rel(START)
            and authority.get("start", {}).get("sha256")
            == _sha(START)
            == EXPECTED_START_SHA256
        ),
        "head_origin_exact": (
            authority.get("git", {}).get("head") == EXPECTED_HEAD
            and authority.get("git", {}).get("origin_master")
            == EXPECTED_HEAD
        ),
        "status_self_inclusive_exact": (
            authority_line in baseline
            and len(baseline)
            == authority.get("git", {}).get(
                "status_with_execution_authority_line_count"
            )
            and authority.get("git", {}).get(
                "status_with_execution_authority_sha256"
            )
            == _text_sha("\n".join(baseline))
        ),
        "status_scope_exact": (
            current == baseline
            if exact_status
            else _status_scope_pass(
                current,
                baseline,
                allowed_output_names=_out_names(),
            )
        ),
        "inputs_exact": (
            authority.get("inputs") == _expected_authority_inputs()
        ),
        "output_exact": (
            authority.get("output", {}).get("path") == _rel(OUT_DIR)
            and authority.get("output", {}).get("forward_only") is True
        ),
    }


def _preregistered_chain_checks() -> dict[str, bool]:
    if not PREREGISTRATION.is_file() or not EXECUTION_AUTHORITY.is_file():
        return {
            "preregistration_exists": PREREGISTRATION.is_file(),
            "execution_authority_exists": EXECUTION_AUTHORITY.is_file(),
        }
    prereg = _read_json(PREREGISTRATION)
    authority = _read_json(EXECUTION_AUTHORITY)
    case_authority = prereg.get("case_authority", {})
    phase_rows: list[dict[str, Any]] = []
    if PHASES.is_file():
        try:
            phase_rows = [
                json.loads(line)
                for line in PHASES.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line
            ]
        except Exception:
            phase_rows = []
    prepare_end = [
        row for row in phase_rows
        if row.get("phase") == "prepare_end"
    ]
    return {
        "case_attempt_exact": (
            prereg.get("case") == CASE
            and prereg.get("attempt") == ATTEMPT
        ),
        "variables_exact": prereg.get("new_variables") == VARIABLES,
        "script_chain_exact": (
            case_authority.get("script_path") == _rel(SCRIPT)
            and case_authority.get("script_sha256")
            == _sha(SCRIPT)
            == authority.get("script", {}).get("sha256")
        ),
        "start_chain_exact": (
            case_authority.get("start_path") == _rel(START)
            and case_authority.get("start_sha256")
            == _sha(START)
            == EXPECTED_START_SHA256
            == authority.get("start", {}).get("sha256")
        ),
        "execution_authority_chain_exact": (
            case_authority.get("execution_authority_path")
            == _rel(EXECUTION_AUTHORITY)
            and case_authority.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
        ),
        "inputs_exact": (
            prereg.get("frozen_inputs")
            == _expected_authority_inputs()
        ),
        "call_scope_exact": (
            prereg.get("evaluation_scope", {}).get("call_indices")
            == [CALL_INDEX]
            and prereg.get("evaluation_scope", {}).get(
                "active_plane_indices"
            )
            == EXPECTED_ACTIVE_PLANES
        ),
        "prepare_end_hash_exact": (
            len(prepare_end) == 1
            and prepare_end[0].get("preregistration_sha256")
            == _sha(PREREGISTRATION)
        ),
        "authority_still_valid": all(
            _execution_authority_checks(
                authority, exact_status=False
            ).values()
        ),
    }


def _validate_progress_prefix() -> dict[str, Any]:
    expected = [
        ("source_streams_bound", None),
        ("active_clip_1", 3),
        ("active_clip_2", 15),
        ("active_clip_3", 20),
    ]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if PROGRESS.is_file():
        for index, line in enumerate(
            PROGRESS.read_text(encoding="utf-8").splitlines()
        ):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("progress row is not an object")
                rows.append(value)
            except Exception as exc:
                errors.append(
                    f"row {index}: {type(exc).__name__}: {exc}"
                )
    checks: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        event, plane = expected[index] if index < len(expected) else (None, None)
        row_checks = {
            "ordinal_exact": row.get("ordinal") == index,
            "event_exact": row.get("event") == event,
            "pass_true": row.get("pass") is True,
            "plane_exact": (
                plane is None or row.get("plane_index") == plane
            ),
        }
        checks.append(
            {
                "ordinal": index,
                "checks": row_checks,
                "pass": all(row_checks.values()),
            }
        )
    return {
        "row_count": len(rows),
        "parse_errors": errors,
        "rows": checks,
        "prefix_exact": (
            not errors
            and len(rows) <= len(expected)
            and all(row["pass"] for row in checks)
        ),
        "complete_exact": (
            not errors
            and len(rows) == len(expected)
            and all(row["pass"] for row in checks)
        ),
    }


def _write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "terminal_index",
        "x_m",
        "y_m",
        "z_m",
        "immediate_lineage",
        "clip_tag",
        "kind",
    ]
    with CSV_PATH.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _prepare() -> int:
    _require_names(AUTHORITY_NAMES, "before_prepare")
    authority = _read_json(EXECUTION_AUTHORITY)
    external = os.environ.get(EXECUTION_AUTHORITY_SHA256_ENV)
    if external is None or external != _sha(EXECUTION_AUTHORITY):
        raise RuntimeError(
            "D393 prepare requires exact external execution-authority SHA"
        )
    authority_checks = _execution_authority_checks(
        authority, exact_status=True
    )
    frozen = _frozen_checks()
    schema = _schema_checks()
    environment = {
        "numpy_1_26_0": np.__version__ == "1.26.0",
        "rerun_sdk_0_34_1": (
            importlib.metadata.version("rerun-sdk") == "0.34.1"
        ),
        "rerun_cli_exists": RERUN_CLI.is_file(),
        "font_exists": FONT_PATH.is_file(),
        "repo_import_bootstrap_exact_once": (
            sys.path.count(str(REPO)) == 1
        ),
        "no_direct_nvidia_imports": not any(
            name
            in {
                "isaaclab",
                "isaacsim",
                "omni",
                "pxr",
                "warp",
                "torch",
            }
            for name in _direct_import_roots()
        ),
        "python_no_bytecode_requested": sys.dont_write_bytecode,
    }
    if not all(authority_checks.values()):
        raise RuntimeError(
            f"D393 execution authority failed: {authority_checks}"
        )
    if not all(frozen.values()):
        raise RuntimeError(f"D393 frozen inputs changed: {frozen}")
    if not all(schema.values()):
        raise RuntimeError(f"D393 schema preflight failed: {schema}")
    if not all(environment.values()):
        raise RuntimeError(
            f"D393 environment preflight failed: {environment}"
        )
    _phase("prepare_start")
    prereg = {
        "artifact": "D393_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "case_authority": {
            "script_path": _rel(SCRIPT),
            "script_sha256": _sha(SCRIPT),
            "start_path": _rel(START),
            "start_sha256": _sha(START),
            "execution_authority_path": _rel(EXECUTION_AUTHORITY),
            "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
            "normalized_user_approval_text": (
                EXPECTED_USER_APPROVAL_NORMALIZED
            ),
            "normalized_user_approval_text_sha256": _text_sha(
                EXPECTED_USER_APPROVAL_NORMALIZED
            ),
        },
        "question": (
            "Is the registered call29 microscopic third-direction witness "
            "already present when D389 creates its shared fan-seam points, "
            "or is that registered witness first created by a D390 clip?"
        ),
        "frozen_inputs": _expected_authority_inputs(),
        "evaluation_scope": {
            "source_child_streams": 2,
            "call_indices": [CALL_INDEX],
            "call_id": CALL_ID,
            "frozen_plane_rows": 21,
            "active_plane_indices": EXPECTED_ACTIVE_PLANES,
            "other_calls": 0,
            "pair_sweeps": 0,
            "volume_or_overlap_solver_calls": 0,
            "frozen_signed_distance_replay_contract": (
                "numpy.einsum('ij,j->i', optimize=False) plus offset; "
                "BLAS matmul is diagnostic only because it can differ at "
                "the ~1e-18 m boundary"
            ),
        },
        "outcome_policy": {
            "lineage_and_controls_all_pass": (
                "D393_CALL29_REGISTERED_MICRO_THIRD_DIRECTION_WITNESS_"
                "ALREADY_PRESENT_AFTER_D389_FINAL_FAN_CLIP_PASS"
            ),
            "any_hash_schema_lineage_or_control_failure": (
                "D393_PROVENANCE_LOCALIZATION_FAIL_STOP"
            ),
            "rank_or_class_adoption": None,
            "all_41_aggregate": None,
            "seam_or_overlap_verdict_update": None,
        },
        "execution_contract": {
            "numeric_worker": 1,
            "worker_retries": 0,
            "process_signals_authorized": 0,
            "hard_watchdog_seconds": None,
            "cooperative_deadline_seconds": (
                COOPERATIVE_DEADLINE_SECONDS
            ),
            "viewer_maximum": 1,
            "viewer_retries": 0,
            "numeric_committed_before_observability": True,
            "progress_append_only": True,
        },
        "visualization_contract": {
            "board": "exact 1920x1080",
            "required_subjects": [
                "both frozen source children",
                "D389 stored quartet and analytic rational shadow",
                "D390 active planes 3,15,20 and terminal six",
                "near-duplicate connectors and explicit null nonclaims",
            ],
            "rrd_rbl": "save-only; Float32 inspection copies",
            "viewer_maximum": 1,
            "manual_inspection_required": True,
        },
        "forbidden": {
            "other_call_or_pair_solver_replay": 0,
            "rank_class_or_seam_adoption": 0,
            "partition_budget_geometry_change": 0,
            "collider_asset_usd_isaac_physx_warp_cuda": 0,
            "cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_change": 0,
            "commit_push_signal_hardware": 0,
        },
        "frozen_checks": frozen,
        "schema_checks": schema,
        "environment_checks": environment,
        "status_baseline": authority["git"][
            "status_with_execution_authority"
        ],
        "forward_only_output": _rel(OUT_DIR),
    }
    _write_json_x(PREREGISTRATION, prereg)
    _phase(
        "prepare_end",
        preregistration_sha256=_sha(PREREGISTRATION),
    )
    _require_names(PREPARED_NAMES, "after_prepare")
    print(json.dumps({"prepared": True, "case": CASE}, ensure_ascii=False))
    return 0


def _authorization_checks() -> dict[str, bool]:
    authority = _read_json(EXECUTION_AUTHORITY)
    authorization = _read_json(AUTHORIZATION)
    external = os.environ.get(WORKER_AUTHORIZATION_SHA256_ENV)
    return {
        "external_authorization_exact": (
            external is not None and external == _sha(AUTHORIZATION)
        ),
        "parent_supervisor_pid_exact": (
            os.getppid() == authorization.get("supervisor_pid")
        ),
        "worker_index_exact": (
            authorization.get("worker_invocation_index") == 1
        ),
        "retry_zero": authorization.get("retry_index") == 0,
        "script_exact": (
            authorization.get("script_sha256") == _sha(SCRIPT)
        ),
        "preregistration_exact": (
            authorization.get("preregistration_sha256")
            == _sha(PREREGISTRATION)
        ),
        "execution_authority_exact": (
            authorization.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
        ),
        "invocation_exact": (
            authorization.get("invocation_sha256")
            == _sha(INVOCATION)
        ),
        "frozen_inputs_exact": all(_frozen_checks().values()),
        "schema_exact": all(_schema_checks().values()),
        "preregistered_chain_exact": all(
            _preregistered_chain_checks().values()
        ),
        "status_scope_exact": _status_scope_pass(
            _status_lines(),
            authority["git"]["status_with_execution_authority"],
            allowed_output_names=_out_names(),
        ),
    }


def _worker_inner() -> int:
    global _deadline_monotonic
    started = time.monotonic()
    _deadline_monotonic = started + COOPERATIVE_DEADLINE_SECONDS
    _require_names(PRE_WORKER_NAMES, "worker_before_sentinel")
    authorization_checks = _authorization_checks()
    if not all(authorization_checks.values()):
        raise RuntimeError(
            f"D393 worker authorization failed: {authorization_checks}"
        )
    _write_json_x(
        SENTINEL,
        {
            "artifact": "D393_WORKER_START_SENTINEL_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "worker_pid": os.getpid(),
            "parent_supervisor_pid": os.getppid(),
            "worker_invocation_index": 1,
            "retry_index": 0,
            "script_sha256": _sha(SCRIPT),
            "preregistration_sha256": _sha(PREREGISTRATION),
            "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
            "authorization_sha256": _sha(AUTHORIZATION),
            "wall_time_ns": time.time_ns(),
        },
    )
    _phase("worker_start", worker_pid=os.getpid())
    evidence, geometry, csv_rows = _compute()
    _write_json_x(EVIDENCE, evidence)
    geometry["canonical_evidence_sha256"] = _sha(EVIDENCE)
    _write_json_x(GEOMETRY, geometry)
    _write_csv(csv_rows)
    _phase(
        "canonical_numeric_evidence_committed",
        evidence_sha256=_sha(EVIDENCE),
        geometry_sha256=_sha(GEOMETRY),
        csv_sha256=_sha(CSV_PATH),
        progress_sha256=_sha(PROGRESS),
    )
    progress = _validate_progress_prefix()
    worker_checks = {
        "authorization_checks_pass": all(
            authorization_checks.values()
        ),
        "numeric_integrity_pass": (
            evidence["numeric_integrity_pass"] is True
        ),
        "progress_exact_four": progress["complete_exact"],
        "evidence_geometry_link_exact": (
            _read_json(GEOMETRY)["canonical_evidence_sha256"]
            == _sha(EVIDENCE)
        ),
        "frozen_inputs_remain_exact": all(_frozen_checks().values()),
        "scope_counters_exact": (
            evidence["scope_counters"] == _scope_counters()
        ),
        "within_cooperative_deadline": (
            time.monotonic() <= _deadline_monotonic
        ),
    }
    claim = {
        "artifact": "D393_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "actual_worker_invocations": 1,
        "worker_invocation_index": 1,
        "retries": 0,
        "process_signals_sent": 0,
        "hard_watchdog_seconds": None,
        "cooperative_deadline_seconds": COOPERATIVE_DEADLINE_SECONDS,
        "worker_elapsed_seconds": time.monotonic() - started,
        "script_sha256": _sha(SCRIPT),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "invocation_sha256": _sha(INVOCATION),
        "authorization_sha256": _sha(AUTHORIZATION),
        "worker_start_sentinel_sha256": _sha(SENTINEL),
        "authorization_checks": authorization_checks,
        "progress_validation": progress,
        "checks": worker_checks,
        "numeric_verdict": evidence["numeric_verdict"],
        "diagnostic_conclusion": evidence["diagnostic_conclusion"],
        "artifacts": {
            "progress": _artifact(PROGRESS),
            "evidence": _artifact(EVIDENCE),
            "geometry": _artifact(GEOMETRY),
            "csv": _artifact(CSV_PATH),
        },
        "pass": all(worker_checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_end", worker_claim_pass=claim["pass"])
    if not claim["pass"]:
        raise RuntimeError(f"D393 worker checks failed: {worker_checks}")
    print(
        json.dumps(
            {
                "worker_pass": True,
                "numeric_verdict": evidence["numeric_verdict"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _worker_failure_claim(exc: BaseException) -> None:
    if WORKER_FAILURE.exists():
        return
    progress = _validate_progress_prefix()
    _write_json_x(
        WORKER_FAILURE,
        {
            "artifact": "D393_WORKER_FAILURE_CLAIM_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "worker_pid": os.getpid(),
            "parent_supervisor_pid": os.getppid(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "progress_prefix_validation": progress,
            "numeric_evidence_committed": EVIDENCE.is_file(),
            "numeric_evidence_sha256": (
                _sha(EVIDENCE) if EVIDENCE.is_file() else None
            ),
            "script_sha256": _sha(SCRIPT),
            "preregistration_sha256": (
                _sha(PREREGISTRATION)
                if PREREGISTRATION.is_file()
                else None
            ),
            "execution_authority_sha256": (
                _sha(EXECUTION_AUTHORITY)
                if EXECUTION_AUTHORITY.is_file()
                else None
            ),
            "authorization_sha256": (
                _sha(AUTHORIZATION)
                if AUTHORIZATION.is_file()
                else None
            ),
            "sentinel_sha256": (
                _sha(SENTINEL) if SENTINEL.is_file() else None
            ),
            "retries": 0,
            "process_signals_sent": 0,
            "hard_watchdog_seconds": None,
        },
    )


def _failure(stage: str, exc: BaseException) -> None:
    if FAILURE.exists():
        return
    supervisor, supervisor_error = (
        _safe_read_json(SUPERVISOR)
        if SUPERVISOR.is_file()
        else (None, None)
    )
    evidence, evidence_error = (
        _safe_read_json(EVIDENCE)
        if EVIDENCE.is_file()
        else (None, None)
    )
    actual_worker_started = bool(
        _worker_started
        or SENTINEL.is_file()
        or WORKER_FAILURE.is_file()
        or (
            isinstance(supervisor, dict)
            and supervisor.get("actual_worker_invocations") == 1
        )
    )
    before = sorted(_out_names() - {FAILURE.name})
    artifacts = {
        path.name: _artifact(path)
        for path in sorted(OUT_DIR.iterdir())
        if path.is_file() and path != FAILURE
    }
    _write_json_x(
        FAILURE,
        {
            "artifact": "D393_FAILURE_ATTESTATION_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "actual_worker_started": actual_worker_started,
            "actual_worker_invocations": int(actual_worker_started),
            "worker_pid": _worker_pid,
            "retries": 0,
            "process_signals_sent": 0,
            "hard_watchdog_seconds": None,
            "progress_prefix_validation": _validate_progress_prefix(),
            "numeric_evidence_committed": EVIDENCE.is_file(),
            "numeric_evidence_read_error": evidence_error,
            "numeric_verdict": (
                evidence.get("numeric_verdict")
                if isinstance(evidence, dict)
                else None
            ),
            "supervisor_read_error": supervisor_error,
            "script_sha256": _sha(SCRIPT),
            "start_sha256": _sha(START),
            "execution_authority_sha256": (
                _sha(EXECUTION_AUTHORITY)
                if EXECUTION_AUTHORITY.is_file()
                else None
            ),
            "artifact_hashes_before_attestation": artifacts,
            "output_inventory_before_attestation": before,
            "expected_final_inventory": sorted(
                set(before) | {FAILURE.name}
            ),
            "verdict": "D393_OPERATIONAL_OR_PROVENANCE_FAIL_STOP",
        },
    )


def _run() -> int:
    global _worker_started, _worker_pid
    _require_names(PREPARED_NAMES, "before_run")
    if FAILURE.exists():
        raise RuntimeError("D393 failure attestation already exists")
    authority = _read_json(EXECUTION_AUTHORITY)
    if not all(_frozen_checks().values()):
        raise RuntimeError("D393 frozen inputs changed before run")
    if not all(_preregistered_chain_checks().values()):
        raise RuntimeError("D393 preregistered chain changed before run")
    if not _status_scope_pass(
        _status_lines(),
        authority["git"]["status_with_execution_authority"],
        allowed_output_names=_out_names(),
    ):
        raise RuntimeError("D393 worktree scope changed before run")
    command = [sys.executable, "-B", str(SCRIPT), "--stage", "worker"]
    invocation = {
        "artifact": "D393_OFFLINE_WORKER_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "cwd": str(REPO),
        "actual_worker_maximum": 1,
        "worker_invocation_index": 1,
        "retry_index": 0,
        "retries": 0,
        "hard_watchdog_seconds": None,
        "cooperative_deadline_seconds": COOPERATIVE_DEADLINE_SECONDS,
        "supervisor_signal_authority": False,
        "process_signals_authorized": 0,
        "script_sha256": _sha(SCRIPT),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
    }
    _write_json_x(INVOCATION, invocation)
    _write_json_x(
        AUTHORIZATION,
        {
            "artifact": "D393_WORKER_AUTHORIZATION_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "supervisor_pid": os.getpid(),
            "worker_invocation_index": 1,
            "retry_index": 0,
            "retries": 0,
            "script_sha256": _sha(SCRIPT),
            "preregistration_sha256": _sha(PREREGISTRATION),
            "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
            "invocation_sha256": _sha(INVOCATION),
            "hard_watchdog_seconds": None,
            "process_signals_authorized": 0,
        },
    )
    _phase("supervisor_before_worker", supervisor_pid=os.getpid())
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    returncode: int | None = None
    error: str | None = None
    env = dict(os.environ)
    env[WORKER_AUTHORIZATION_SHA256_ENV] = _sha(AUTHORIZATION)
    try:
        with STDOUT.open("x", encoding="utf-8") as stdout, STDERR.open(
            "x", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                stdout=stdout,
                stderr=stderr,
                text=True,
                env=env,
            )
            _worker_started = True
            _worker_pid = process.pid
            returncode = process.wait()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    claim, claim_error = (
        _safe_read_json(WORKER_CLAIM)
        if WORKER_CLAIM.is_file()
        else (None, "worker claim absent")
    )
    evidence, evidence_error = (
        _safe_read_json(EVIDENCE)
        if EVIDENCE.is_file()
        else (None, "evidence absent")
    )
    artifacts = (
        claim.get("artifacts", {})
        if isinstance(claim, dict)
        else {}
    )
    post_checks = {
        "process_created": process is not None,
        "process_exited": (
            process is not None and process.poll() is not None
        ),
        "return_zero": returncode == 0,
        "claim_readable": claim_error is None,
        "evidence_readable": evidence_error is None,
        "claim_pass": (
            isinstance(claim, dict) and claim.get("pass") is True
        ),
        "progress_complete": _validate_progress_prefix()[
            "complete_exact"
        ],
        "artifact_refs_exact": (
            _artifact_reference_exact(
                artifacts.get("progress"), PROGRESS
            )
            and _artifact_reference_exact(
                artifacts.get("evidence"), EVIDENCE
            )
            and _artifact_reference_exact(
                artifacts.get("geometry"), GEOMETRY
            )
            and _artifact_reference_exact(
                artifacts.get("csv"), CSV_PATH
            )
        ),
        "no_worker_failure_claim": not WORKER_FAILURE.exists(),
        "no_supervisor_error": error is None,
    }
    supervisor = {
        "artifact": "D393_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "actual_worker_invocations": int(_worker_started),
        "worker_pid": _worker_pid,
        "worker_returncode": returncode,
        "worker_exited": (
            process is not None and process.poll() is not None
        ),
        "worker_runtime_seconds": time.monotonic() - started,
        "retries": 0,
        "process_signals_sent": 0,
        "hard_watchdog_seconds": None,
        "cooperative_deadline_seconds": COOPERATIVE_DEADLINE_SECONDS,
        "supervisor_error": error,
        "worker_claim_read_error": claim_error,
        "evidence_read_error": evidence_error,
        "script_sha256": _sha(SCRIPT),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "invocation_sha256": _sha(INVOCATION),
        "authorization_sha256": _sha(AUTHORIZATION),
        "sentinel_sha256": (
            _sha(SENTINEL) if SENTINEL.is_file() else None
        ),
        "worker_claim_sha256": (
            _sha(WORKER_CLAIM) if WORKER_CLAIM.is_file() else None
        ),
        "progress_sha256": (
            _sha(PROGRESS) if PROGRESS.is_file() else None
        ),
        "evidence_sha256": (
            _sha(EVIDENCE) if EVIDENCE.is_file() else None
        ),
        "geometry_sha256": (
            _sha(GEOMETRY) if GEOMETRY.is_file() else None
        ),
        "csv_sha256": (
            _sha(CSV_PATH) if CSV_PATH.is_file() else None
        ),
        "post_worker_chain_checks": post_checks,
        "numeric_verdict": (
            evidence.get("numeric_verdict")
            if isinstance(evidence, dict)
            else None
        ),
        "pass": all(post_checks.values()),
    }
    _write_json_x(SUPERVISOR, supervisor)
    _phase(
        "supervisor_after_worker",
        worker_returncode=returncode,
        supervisor_pass=supervisor["pass"],
    )
    if not supervisor["pass"]:
        raise RuntimeError(f"D393 numeric worker failed: {supervisor}")
    _require_names(POST_WORKER_NAMES, "after_run")
    print(json.dumps(supervisor, ensure_ascii=False))
    return 0


def _numeric_chain_checks() -> dict[str, bool]:
    authority = _read_json(EXECUTION_AUTHORITY)
    invocation = _read_json(INVOCATION)
    authorization = _read_json(AUTHORIZATION)
    sentinel = _read_json(SENTINEL)
    worker = _read_json(WORKER_CLAIM)
    supervisor = _read_json(SUPERVISOR)
    evidence = _read_json(EVIDENCE)
    geometry = _read_json(GEOMETRY)
    artifacts = worker.get("artifacts", {})
    return {
        "frozen_inputs_exact": all(_frozen_checks().values()),
        "schema_exact": all(_schema_checks().values()),
        "preregistered_chain_exact": all(
            _preregistered_chain_checks().values()
        ),
        "status_scope_exact": _status_scope_pass(
            _status_lines(),
            authority["git"]["status_with_execution_authority"],
            allowed_output_names=_out_names(),
        ),
        "invocation_chain_exact": (
            invocation.get("script_sha256") == _sha(SCRIPT)
            and invocation.get("preregistration_sha256")
            == _sha(PREREGISTRATION)
            and invocation.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
            and invocation.get("worker_invocation_index") == 1
            and invocation.get("retry_index") == 0
        ),
        "authorization_chain_exact": (
            authorization.get("script_sha256") == _sha(SCRIPT)
            and authorization.get("preregistration_sha256")
            == _sha(PREREGISTRATION)
            and authorization.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
            and authorization.get("invocation_sha256")
            == _sha(INVOCATION)
            and authorization.get("worker_invocation_index") == 1
            and authorization.get("retry_index") == 0
        ),
        "sentinel_chain_exact": (
            sentinel.get("script_sha256") == _sha(SCRIPT)
            and sentinel.get("preregistration_sha256")
            == _sha(PREREGISTRATION)
            and sentinel.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
            and sentinel.get("authorization_sha256")
            == _sha(AUTHORIZATION)
            and sentinel.get("worker_invocation_index") == 1
            and sentinel.get("retry_index") == 0
            and sentinel.get("parent_supervisor_pid")
            == authorization.get("supervisor_pid")
        ),
        "worker_chain_exact": (
            worker.get("script_sha256") == _sha(SCRIPT)
            and worker.get("preregistration_sha256")
            == _sha(PREREGISTRATION)
            and worker.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
            and worker.get("invocation_sha256") == _sha(INVOCATION)
            and worker.get("authorization_sha256")
            == _sha(AUTHORIZATION)
            and worker.get("worker_start_sentinel_sha256")
            == _sha(SENTINEL)
            and worker.get("pass") is True
        ),
        "worker_artifacts_exact": (
            _artifact_reference_exact(
                artifacts.get("progress"), PROGRESS
            )
            and _artifact_reference_exact(
                artifacts.get("evidence"), EVIDENCE
            )
            and _artifact_reference_exact(
                artifacts.get("geometry"), GEOMETRY
            )
            and _artifact_reference_exact(
                artifacts.get("csv"), CSV_PATH
            )
        ),
        "geometry_evidence_link_exact": (
            geometry.get("canonical_evidence_sha256")
            == _sha(EVIDENCE)
        ),
        "progress_complete_exact": _validate_progress_prefix()[
            "complete_exact"
        ],
        "supervisor_chain_exact": (
            supervisor.get("script_sha256") == _sha(SCRIPT)
            and supervisor.get("preregistration_sha256")
            == _sha(PREREGISTRATION)
            and supervisor.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
            and supervisor.get("invocation_sha256")
            == _sha(INVOCATION)
            and supervisor.get("authorization_sha256")
            == _sha(AUTHORIZATION)
            and supervisor.get("sentinel_sha256") == _sha(SENTINEL)
            and supervisor.get("worker_claim_sha256")
            == _sha(WORKER_CLAIM)
            and supervisor.get("progress_sha256") == _sha(PROGRESS)
            and supervisor.get("evidence_sha256") == _sha(EVIDENCE)
            and supervisor.get("geometry_sha256") == _sha(GEOMETRY)
            and supervisor.get("csv_sha256") == _sha(CSV_PATH)
            and supervisor.get("actual_worker_invocations") == 1
            and supervisor.get("retries") == 0
            and supervisor.get("process_signals_sent") == 0
            and supervisor.get("pass") is True
        ),
        "numeric_integrity_pass": (
            evidence.get("numeric_integrity_pass") is True
        ),
        "null_authority_preserved": (
            evidence.get("frozen_rank_authority", {}).get(
                "d391_authoritative_rank"
            )
            is None
            and evidence.get("frozen_rank_authority", {}).get(
                "d392_authoritative_rank"
            )
            is None
            and evidence.get("nonclaims", {}).get(
                "authoritative_all_41_aggregate"
            )
            is None
        ),
    }


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _render_board(
    evidence: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    image = Image.new("RGB", (1920, 1080), (246, 249, 252))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(33)
    subtitle_font = _load_font(19)
    header_font = _load_font(22)
    body_font = _load_font(17)
    small_font = _load_font(15)
    text_boxes: list[dict[str, Any]] = []

    def put(
        xy: tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        color: tuple[int, int, int],
        tag: str,
        owner: Sequence[int],
    ) -> None:
        draw.text(xy, text, font=font, fill=color)
        bbox = list(draw.textbbox(xy, text, font=font))
        text_boxes.append(
            {
                "tag": tag,
                "text": text,
                "bbox": bbox,
                "owner": list(owner),
                "inside_owner": (
                    owner[0] <= bbox[0]
                    and owner[1] <= bbox[1]
                    and bbox[2] <= owner[2]
                    and bbox[3] <= owner[3]
                ),
            }
        )

    put(
        (42, 18),
        "D393: call29에 등록한 극미세 세 번째 방향 증거는 D390 전부터 있었나?",
        title_font,
        (20, 35, 55),
        "title",
        [35, 10, 1885, 62],
    )
    put(
        (44, 66),
        (
            "동결된 D389 자식 생성 → D390 평면 3·15·20 절단만 추적 "
            "· 아래 확대 형상은 관찰용이며 수치 판정은 JSON의 Float64/정확분수"
        ),
        subtitle_font,
        (55, 70, 90),
        "subtitle",
        [35, 60, 1885, 98],
    )

    left = [40, 115, 920, 655]
    right = [960, 115, 1880, 655]
    lower_left = [40, 680, 920, 1045]
    lower_right = [960, 680, 1880, 1045]
    for bounds in (left, right, lower_left, lower_right):
        draw.rounded_rectangle(
            bounds,
            radius=15,
            fill=(255, 255, 255),
            outline=(155, 172, 190),
            width=2,
        )
    put(
        (62, 130),
        "1) D389의 두 자식 형상과 공유 경계면 잔차",
        header_font,
        (25, 60, 90),
        "left_header",
        [55, 124, 905, 165],
    )
    put(
        (982, 130),
        "2) D390에서 실제로 작동한 절단 3단계",
        header_font,
        (25, 60, 90),
        "right_header",
        [975, 124, 1865, 165],
    )

    def plot_points(
        points: np.ndarray,
        bounds: Sequence[int],
        color: tuple[int, int, int],
        *,
        radius: int = 5,
        axes: tuple[int, int] = (0, 2),
    ) -> list[tuple[int, int]]:
        array = np.asarray(points, dtype=np.float64)
        projected = array[:, list(axes)]
        minimum = projected.min(axis=0)
        maximum = projected.max(axis=0)
        span = np.maximum(maximum - minimum, 1.0e-15)
        normalized = (projected - minimum) / span
        x0, y0, x1, y1 = bounds
        result: list[tuple[int, int]] = []
        for row in normalized:
            x = int(x0 + 18 + row[0] * (x1 - x0 - 36))
            y = int(y1 - 18 - row[1] * (y1 - y0 - 36))
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(25, 35, 45),
                width=1,
            )
            result.append((x, y))
        return result

    child1 = np.asarray(
        geometry["source_child1_f64_m"], dtype=np.float64
    )
    child2 = np.asarray(
        geometry["clipping_child2_f64_m"], dtype=np.float64
    )
    plot_points(child1, [65, 185, 455, 390], (25, 145, 205))
    plot_points(child2, [505, 185, 895, 390], (230, 145, 40))
    put(
        (80, 176),
        f"왼쪽 자식 1 · {len(child1)}점",
        body_font,
        (20, 105, 160),
        "child1_label",
        [65, 168, 455, 205],
    )
    put(
        (520, 176),
        f"오른쪽 자식 2 · {len(child2)}점",
        body_font,
        (175, 95, 20),
        "child2_label",
        [505, 168, 895, 205],
    )
    seam = evidence["d389_shared_fan_seam_lineage"]
    stored_quartet = np.asarray(
        geometry["d389_source_quartet_f64_m"], dtype=np.float64
    )
    analytic_quartet = np.asarray(
        geometry["d389_analytic_shadow_quartet_f64_m"],
        dtype=np.float64,
    )
    plot_points(
        stored_quartet,
        [70, 455, 330, 575],
        (195, 55, 80),
        radius=6,
    )
    plot_points(
        analytic_quartet,
        [365, 455, 625, 575],
        (35, 165, 105),
        radius=6,
    )
    put(
        (78, 430),
        "저장 Float64 4점",
        small_font,
        (145, 40, 60),
        "stored_quartet_label",
        [68, 421, 330, 454],
    )
    put(
        (373, 430),
        "정확분수 교차 shadow 4점",
        small_font,
        (25, 125, 80),
        "analytic_quartet_label",
        [363, 421, 650, 454],
    )
    residuals = seam["raw_seam_plane"][
        "stored_quartet_normalized_abs_residuals_m"
    ]
    max_residual = max(residuals) if residuals else 1.0
    base_y = 575
    for index, residual in enumerate(residuals):
        height = int(85 * residual / max_residual) if max_residual else 0
        x = 675 + index * 52
        draw.rectangle(
            (x, base_y - height, x + 32, base_y),
            fill=(190, 55, 75),
            outline=(90, 35, 45),
        )
        put(
            (x + 5, 583),
            f"Q{index}",
            small_font,
            (55, 60, 70),
            f"quartet_{index}",
            [x, 579, x + 42, 615],
        )
    put(
        (665, 430),
        (
            "경계면 잔차 세로 확대"
        ),
        small_font,
        (110, 45, 60),
        "residual_label",
        [655, 421, 900, 454],
    )
    put(
        (665, 612),
        f"최대 {max_residual:.3e} m",
        small_font,
        (110, 45, 60),
        "residual_value",
        [655, 605, 905, 640],
    )

    stages = geometry["d390_active_stages"]
    stage_bounds = [
        [985, 190, 1260, 500],
        [1280, 190, 1555, 500],
        [1575, 190, 1850, 500],
    ]
    stage_colors = [(55, 130, 205), (70, 165, 120), (210, 85, 90)]
    for index, (stage, bounds, color) in enumerate(
        zip(stages, stage_bounds, stage_colors, strict=True)
    ):
        points = np.asarray(
            stage["candidate_points_f64_m"], dtype=np.float64
        )
        plot_points(points, bounds, color, radius=6)
        put(
            (bounds[0] + 5, 172),
            (
                f"평면 {stage['plane_index']} · "
                f"{len(points)}점"
            ),
            small_font,
            color,
            f"stage_{index}",
            [bounds[0], 166, bounds[2], 200],
        )
    terminal = evidence[
        "d390_carry_and_near_duplicate_decomposition"
    ]
    pair_distance = terminal["near_duplicate_pairs"][0]["distance_m"]
    put(
        (987, 530),
        (
            "최종 6점: 기존점 4 + 평면20 교차점 2 · "
            f"근접쌍 간격 {pair_distance:.3e} m"
        ),
        body_font,
        (105, 45, 55),
        "terminal_summary",
        [980, 520, 1860, 558],
    )
    put(
        (987, 570),
        (
            "계보: "
            + " | ".join(
                f"T{index}="
                f"{'유지점' if kind == 'terminal_carried' else '평면20 교차점'}"
                for index, kind in enumerate(
                    terminal["terminal_immediate_lineage_kind"]
                )
            )
        ),
        small_font,
        (80, 45, 55),
        "terminal_lineage_legend",
        [980, 562, 1860, 594],
    )
    put(
        (987, 610),
        "중요: 등록한 4점 증거는 D390 전에 이미 존재하며, 평면20은 그 계보를 운반",
        body_font,
        (150, 45, 45),
        "stage_nonorigin",
        [980, 600, 1860, 640],
    )

    put(
        (62, 697),
        "3) 정확한 수치 계보",
        header_font,
        (25, 60, 90),
        "numeric_header",
        [55, 690, 905, 730],
    )
    stored_det = seam["stored_quartet_determinant_m3"]["float"]
    stored_volume = seam["stored_quartet_tetra_volume_m3"]["float"]
    analytic_det = seam["analytic_shadow_determinant_m3"]["float"]
    terminal_volume = terminal["terminal_max_tetra"][
        "tetra_volume_m3"
    ]["float"]
    numeric_lines = [
        f"D389 저장 Float64 4점 determinant: {stored_det:.6e} m³",
        f"D389 저장 4점 사면체 부피: {stored_volume:.6e} m³",
        f"동일 선분의 정확분수 교차 shadow determinant: {analytic_det:.1f} m³",
        f"D390 최종 6점의 최대 미세 사면체 부피: {terminal_volume:.6e} m³",
        (
            "등록 4점이 처음 정의되는 단계: D389 lower child1의 마지막 "
            "공유 fan-ray 교차 계산(전체 조합 중 최초라는 뜻은 아님)"
        ),
    ]
    for index, line in enumerate(numeric_lines):
        put(
            (66, 744 + index * 48),
            line,
            body_font,
            (45, 60, 75),
            f"numeric_{index}",
            [58, 735 + index * 48, 905, 780 + index * 48],
        )

    put(
        (982, 697),
        "4) 이 결과가 말하는 것 / 말하지 않는 것",
        header_font,
        (25, 60, 90),
        "boundary_header",
        [975, 690, 1865, 730],
    )
    boundary_lines = [
        "✓ 저장된 Float64 계보에서는 경계면 교차 반올림 잔차 가설이 강하게 지지됨",
        "✓ 정확분수로 같은 교차를 계산하면 해당 4점의 미세 부피는 정확히 0",
        "✗ 실제 제작물에 물리적 두께가 있다는 증거가 아님",
        "✗ call29 rank/class는 계속 null · all-41 집계도 계속 null",
        "✗ D389 seam/겹침 판정·충돌체·USD·Isaac·원통 물리는 변경/실행 0",
        "✗ g0a_pass=false · ‘원통을 잡는다’는 결론과 무관",
    ]
    for index, line in enumerate(boundary_lines):
        put(
            (988, 744 + index * 46),
            line,
            body_font,
            (30, 105, 70) if index < 2 else (155, 45, 50),
            f"boundary_{index}",
            [978, 735 + index * 46, 1865, 778 + index * 46],
        )
    image.save(BOARD)

    overlaps: list[dict[str, str]] = []
    for left_index, first in enumerate(text_boxes):
        fx0, fy0, fx1, fy1 = first["bbox"]
        for second in text_boxes[left_index + 1 :]:
            sx0, sy0, sx1, sy1 = second["bbox"]
            if max(fx0, sx0) < min(fx1, sx1) and max(fy0, sy0) < min(
                fy1, sy1
            ):
                overlaps.append(
                    {"left": first["tag"], "right": second["tag"]}
                )
    layout = {
        "artifact": "D393_BOARD_LAYOUT_VALIDATION_V1",
        "path": _rel(BOARD),
        "width": 1920,
        "height": 1080,
        "required_subjects": {
            "source_children": 2,
            "stored_and_analytic_quartets": 2,
            "active_clip_stages": [
                row["plane_index"] for row in stages
            ],
            "terminal_point_count": len(
                geometry["d390_terminal_six_f64_m"]
            ),
            "explicit_null_nonclaims": True,
        },
        "text_boxes": text_boxes,
        "text_bounds_pass": all(
            0 <= row["bbox"][0]
            and 0 <= row["bbox"][1]
            and row["bbox"][2] <= 1920
            and row["bbox"][3] <= 1080
            for row in text_boxes
        ),
        "text_owner_bounds_pass": all(
            row["inside_owner"] for row in text_boxes
        ),
        "text_overlap_count": len(overlaps),
        "text_overlaps": overlaps,
        "pass": (
            [row["plane_index"] for row in stages]
            == EXPECTED_ACTIVE_PLANES
            and len(geometry["d390_terminal_six_f64_m"]) == 6
            and all(row["inside_owner"] for row in text_boxes)
            and not overlaps
        ),
    }
    _write_json_x(LAYOUT, layout)
    return layout


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    decision = rrb.Vertical(
        rrb.Spatial3DView(
            origin="/",
            contents="/d393/**",
            name="D393 call29 lineage atlas (inspection only)",
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=(4.0, -18.0, 17.0),
                look_target=(0.0, 0.0, 0.0),
                eye_up=(0.0, 0.0, 1.0),
            ),
            spatial_information=rrb.SpatialInformation(
                target_frame="tf#/",
                show_axes=True,
                show_bounding_box=False,
            ),
        ),
        rrb.TextDocumentView(
            origin="/metadata/run",
            contents="/metadata/run",
            name="Numeric authority and nonclaims",
        ),
        row_shares=[0.78, 0.22],
    )
    notification = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d393/notification_buffer/**",
        name="Notification buffer - no decision evidence",
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=False,
            show_bounding_box=False,
        ),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            decision, notification, column_shares=[0.78, 0.22]
        ),
        rrb.TimePanel(state=rrb.PanelState.Hidden),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


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


def _write_rerun(
    evidence: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    import roarm_rl.rerun_contract as rerun_contract
    import roarm_rl.viz_debug as viz_debug

    def atlas(
        points: Sequence[Sequence[float]],
        offset: Sequence[float],
    ) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        center = array.mean(axis=0)
        centered = array - center
        span = float(np.max(np.linalg.norm(centered, axis=1)))
        scale = 1.2 / span if span > 0.0 else 1.0
        return centered * scale + np.asarray(offset, dtype=np.float64)

    groups: list[tuple[str, np.ndarray, Sequence[float], list[int]]] = [
        (
            "d393/source/child1",
            np.asarray(geometry["source_child1_f64_m"]),
            [-4.5, 3.0, 0.0],
            [30, 145, 215, 255],
        ),
        (
            "d393/source/child2",
            np.asarray(geometry["clipping_child2_f64_m"]),
            [-1.5, 3.0, 0.0],
            [235, 145, 35, 255],
        ),
        (
            "d393/d389/stored_quartet",
            np.asarray(geometry["d389_source_quartet_f64_m"]),
            [1.5, 3.0, 0.0],
            [205, 55, 80, 255],
        ),
        (
            "d393/d389/analytic_shadow",
            np.asarray(
                geometry["d389_analytic_shadow_quartet_f64_m"]
            ),
            [4.5, 3.0, 0.0],
            [35, 175, 110, 255],
        ),
    ]
    stage_offsets = [[-4.0, -2.0, 0.0], [0.0, -2.0, 0.0], [4.0, -2.0, 0.0]]
    stage_colors = [
        [55, 130, 210, 255],
        [65, 170, 120, 255],
        [215, 70, 80, 255],
    ]
    for stage, offset, color in zip(
        geometry["d390_active_stages"],
        stage_offsets,
        stage_colors,
        strict=True,
    ):
        groups.append(
            (
                f"d393/d390/plane{stage['plane_index']:02d}",
                np.asarray(stage["candidate_points_f64_m"]),
                offset,
                color,
            )
        )
    points_rows: list[dict[str, Any]] = []
    atlas_by_path: dict[str, np.ndarray] = {}
    for path, points, offset, color in groups:
        display = atlas(points, offset)
        atlas_by_path[path] = display
        if path == "d393/d390/plane20":
            kinds = evidence[
                "d390_carry_and_near_duplicate_decomposition"
            ]["terminal_immediate_lineage_kind"]
            labels = [
                (
                    f"T{index}:carried"
                    if kind == "terminal_carried"
                    else f"T{index}:plane20_intersection"
                )
                for index, kind in enumerate(kinds)
            ]
        elif path == "d393/d389/stored_quartet":
            labels = [f"Q{index}:stored_f64" for index in range(len(display))]
        elif path == "d393/d389/analytic_shadow":
            labels = [
                f"A{index}:exact_rational_shadow"
                for index in range(len(display))
            ]
        else:
            labels = (
                [path.rsplit("/", 1)[-1]]
                + [""] * (len(display) - 1)
            )
        points_rows.append(
            {
                "entity_path": path,
                "positions_m": display,
                "radii": [0.07] * len(display),
                "colors": [color] * len(display),
                "labels": labels,
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
    terminal_display = atlas_by_path["d393/d390/plane20"]
    arrows: list[dict[str, Any]] = []
    origins: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    for left, right in ((1, 2), (3, 4)):
        direction = terminal_display[right] - terminal_display[left]
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-15:
            raw = np.asarray(
                geometry["d390_terminal_six_f64_m"], dtype=np.float64
            )
            direction = raw[right] - raw[left]
            length = float(np.linalg.norm(direction))
        vector = (
            direction / length * 0.55
            if length > 0.0
            else np.asarray([0.55, 0.0, 0.0])
        )
        origins.append(terminal_display[left])
        vectors.append(vector)
    arrows.append(
        {
            "entity_path": "d393/d390/near_duplicate_connectors",
            "origins_m": np.asarray(origins),
            "vectors_m": np.asarray(vectors),
            "radii": [0.025, 0.025],
            "colors": [[250, 220, 40, 255]] * 2,
            "labels": ["magnified pair 1-2", "magnified pair 3-4"],
            "coordinate_frame": "tf#/",
            "static": True,
        }
    )
    metadata = {
        "case": CASE,
        "attempt": ATTEMPT,
        "numeric_verdict": evidence["numeric_verdict"],
        "canonical_evidence_sha256": _sha(EVIDENCE),
        "display_geometry_sha256": _sha(GEOMETRY),
        "call29_rank": None,
        "call29_class": None,
        "all_41_aggregate": None,
        "viewer_geometry_role": (
            "centered, scaled, offset Float32 inspection atlas only"
        ),
        "near_duplicate_arrows": (
            "direction-normalized magnification for visibility only"
        ),
        "seam_overlap_physics_verdict_updated": False,
        "g0a_pass": False,
    }
    original_builder = viz_debug.build_rerun_blueprint
    original_runner = rerun_contract._run
    viewer_calls = 0

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d393_static_lineage_atlas":
            return _build_blueprint()
        return original_builder(mode)

    def no_signal_runner(
        command: list[str], *, timeout_s: float
    ) -> dict[str, Any]:
        nonlocal viewer_calls
        del timeout_s
        if any("screenshot" in str(part) for part in command):
            viewer_calls += 1
            if viewer_calls > 1:
                return {
                    "command": command,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "D393 Viewer maximum one exceeded",
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
                "timeout_parameter_ignored_no_signal_authority": True,
            }
        except Exception as exc:
            return {
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": repr(exc),
                "ok": False,
                "timeout_parameter_ignored_no_signal_authority": True,
            }

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    rerun_contract._run = no_signal_runner
    try:
        saved = viz_debug.log_rerun(
            RRD,
            points=points_rows,
            arrows=arrows,
            recording_metadata=metadata,
            recording_id="g0a_d393_call29_provenance",
            blueprint_path=RBL,
            blueprint_mode="d393_static_lineage_atlas",
            live_viewer=False,
            app_id="roarm_g0a_d393_call29_provenance",
        )
        if not saved.get("ok"):
            raise RuntimeError(f"D393 save-only Rerun failed: {saved}")
        expected_entities = ["metadata/run"] + sorted(
            [row[0] for row in groups]
            + ["d393/d390/near_duplicate_connectors"]
        )
        components: dict[str, list[str]] = {
            "metadata/run": ["TextDocument:text"]
        }
        for path, *_ in groups:
            components[path] = [
                "CoordinateFrame:frame",
                "Points3D:colors",
                "Points3D:labels",
                "Points3D:positions",
                "Points3D:radii",
            ]
        components["d393/d390/near_duplicate_connectors"] = [
            "Arrows3D:colors",
            "Arrows3D:labels",
            "Arrows3D:origins",
            "Arrows3D:radii",
            "Arrows3D:vectors",
            "CoordinateFrame:frame",
        ]
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
    dimensions = (
        screenshot["width"] in {1920, 3840}
        and screenshot["height"] in {1080, 2160}
        and screenshot["width"] * 9
        == screenshot["height"] * 16
    )
    validation["d393_execution_contract"] = {
        "static_atlas": True,
        "time_panel_hidden": True,
        "notification_buffer_share": 0.22,
        "headless_viewer_invocations": viewer_calls,
        "viewer_maximum": 1,
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "subprocess_timeout_seconds": None,
        "screenshot_dimension_contract_pass": dimensions,
    }
    validation["base_contract_pass"] = validation.get("pass") is True
    validation["pass"] = (
        validation["base_contract_pass"]
        and viewer_calls == 1
        and dimensions
    )
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
    _require_names(POST_WORKER_NAMES, "before_observability")
    if FAILURE.exists():
        raise RuntimeError("D393 failure attestation forbids observability")
    supervisor = _read_json(SUPERVISOR)
    worker = _read_json(WORKER_CLAIM)
    numeric_chain = _numeric_chain_checks()
    if supervisor.get("pass") is not True or worker.get("pass") is not True:
        raise RuntimeError("D393 numeric authority is not complete")
    if not all(numeric_chain.values()):
        raise RuntimeError(
            f"D393 numeric artifact chain changed: {numeric_chain}"
        )
    _phase("observability_start")
    started = time.monotonic()
    evidence = _read_json(EVIDENCE)
    geometry = _read_json(GEOMETRY)
    layout = _render_board(evidence, geometry)
    rerun = _write_rerun(evidence, geometry)
    template = {
        "artifact": "D393_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board_path": _rel(BOARD),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "artifact_hashes_to_bind_after_actual_viewing": {
            "board_sha256": _sha(BOARD),
            "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
        },
        "checks_to_record_after_actual_viewing": MANUAL_CHECK_KEYS,
        "minimum_nonempty_observations": 3,
        "manual_inspection_complete": False,
    }
    _write_json_x(MANUAL_TEMPLATE, template)
    checks = {
        "numeric_worker_complete_before_observability": (
            EVIDENCE.is_file()
            and WORKER_CLAIM.is_file()
            and SUPERVISOR.is_file()
        ),
        "numeric_artifact_chain_exact": all(numeric_chain.values()),
        "board_layout_pass": layout["pass"],
        "board_exact_1920x1080": (
            _png_info(BOARD)["width"] == 1920
            and _png_info(BOARD)["height"] == 1080
        ),
        "rerun_contract_pass": rerun["pass"],
        "viewer_exactly_one_no_retry": rerun["viewer_calls"] == 1,
    }
    claim = {
        "artifact": "D393_OBSERVABILITY_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "script_sha256": _sha(SCRIPT),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "invocation_sha256": _sha(INVOCATION),
        "authorization_sha256": _sha(AUTHORIZATION),
        "sentinel_sha256": _sha(SENTINEL),
        "progress_sha256": _sha(PROGRESS),
        "evidence_sha256": _sha(EVIDENCE),
        "geometry_sha256": _sha(GEOMETRY),
        "csv_sha256": _sha(CSV_PATH),
        "worker_claim_sha256": _sha(WORKER_CLAIM),
        "supervisor_sha256": _sha(SUPERVISOR),
        "numeric_chain_checks": numeric_chain,
        "checks": checks,
        "artifacts": {
            "board": _artifact(BOARD),
            "layout": _artifact(LAYOUT),
            "rerun": rerun,
            "manual_template": _artifact(MANUAL_TEMPLATE),
        },
        "elapsed_seconds": time.monotonic() - started,
        "process_signals_sent": 0,
        "viewer_invocations": rerun["viewer_calls"],
        "viewer_retries": 0,
        "pass": all(checks.values()),
    }
    _write_json_x(OBSERVABILITY_CLAIM, claim)
    _phase("observability_end", observability_pass=claim["pass"])
    if not claim["pass"]:
        raise RuntimeError(f"D393 observability failed: {checks}")
    _require_names(POST_OBSERVE_NAMES, "after_observability")
    print(json.dumps(claim, ensure_ascii=False))
    return 0


def _observability_chain_checks() -> dict[str, bool]:
    claim = _read_json(OBSERVABILITY_CLAIM)
    artifacts = claim.get("artifacts", {})
    rerun = artifacts.get("rerun", {})
    template = _read_json(MANUAL_TEMPLATE)
    layout = _read_json(LAYOUT)
    validation = _read_json(RERUN_VALIDATION)
    return {
        "numeric_chain_exact": all(_numeric_chain_checks().values()),
        "claim_provenance_exact": (
            claim.get("script_sha256") == _sha(SCRIPT)
            and claim.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
            and claim.get("preregistration_sha256")
            == _sha(PREREGISTRATION)
            and claim.get("invocation_sha256") == _sha(INVOCATION)
            and claim.get("authorization_sha256")
            == _sha(AUTHORIZATION)
            and claim.get("sentinel_sha256") == _sha(SENTINEL)
            and claim.get("progress_sha256") == _sha(PROGRESS)
            and claim.get("evidence_sha256") == _sha(EVIDENCE)
            and claim.get("geometry_sha256") == _sha(GEOMETRY)
            and claim.get("csv_sha256") == _sha(CSV_PATH)
            and claim.get("worker_claim_sha256")
            == _sha(WORKER_CLAIM)
            and claim.get("supervisor_sha256") == _sha(SUPERVISOR)
        ),
        "board_layout_template_refs_exact": (
            _artifact_reference_exact(artifacts.get("board"), BOARD)
            and _artifact_reference_exact(artifacts.get("layout"), LAYOUT)
            and _artifact_reference_exact(
                artifacts.get("manual_template"), MANUAL_TEMPLATE
            )
        ),
        "rerun_refs_exact": (
            _artifact_reference_exact(rerun.get("rrd"), RRD)
            and _artifact_reference_exact(rerun.get("rbl"), RBL)
            and _artifact_reference_exact(
                rerun.get("validation"), RERUN_VALIDATION
            )
            and rerun.get("screenshot") == _png_info(RERUN_SCREENSHOT)
        ),
        "layout_pass": layout.get("pass") is True,
        "rerun_validation_pass": validation.get("pass") is True,
        "template_identity_exact": (
            template.get("artifact")
            == "D393_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1"
            and template.get("case") == CASE
            and template.get("attempt") == ATTEMPT
            and template.get("board_path") == _rel(BOARD)
            and template.get("rerun_screenshot_path")
            == _rel(RERUN_SCREENSHOT)
            and template.get("checks_to_record_after_actual_viewing")
            == MANUAL_CHECK_KEYS
            and template.get("minimum_nonempty_observations") == 3
            and template.get("artifact_hashes_to_bind_after_actual_viewing")
            == {
                "board_sha256": _sha(BOARD),
                "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
            }
            and template.get("manual_inspection_complete") is False
        ),
        "claim_pass": claim.get("pass") is True,
        "viewer_once_no_retry_no_signal": (
            claim.get("viewer_invocations") == 1
            and claim.get("viewer_retries") == 0
            and claim.get("process_signals_sent") == 0
        ),
    }


def _phase_contract() -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in PHASES.read_text(encoding="utf-8").splitlines()
        if line
    ]
    observed = [row["phase"] for row in rows]
    return {
        "observed": observed,
        "expected": PHASE_ORDER,
        "exact": observed == PHASE_ORDER,
        "ordinal_exact": (
            [row["ordinal"] for row in rows]
            == list(range(len(rows)))
        ),
        "monotonic_time_forward": all(
            rows[index]["monotonic_ns"]
            <= rows[index + 1]["monotonic_ns"]
            for index in range(len(rows) - 1)
        ),
        "wall_time_forward": all(
            rows[index]["time_ns"] <= rows[index + 1]["time_ns"]
            for index in range(len(rows) - 1)
        ),
    }


def _finalize() -> int:
    _require_names(FINAL_NAMES - {COMPLETION.name}, "before_finalize")
    if FAILURE.exists():
        raise RuntimeError("D393 failure attestation forbids finalize")
    _phase("finalize_start")
    evidence = _read_json(EVIDENCE)
    supervisor = _read_json(SUPERVISOR)
    worker = _read_json(WORKER_CLAIM)
    observability = _read_json(OBSERVABILITY_CLAIM)
    template = _read_json(MANUAL_TEMPLATE)
    manual = _read_json(MANUAL)
    observability_chain = _observability_chain_checks()
    expected_keys = set(MANUAL_CHECK_KEYS)
    manual_checks = manual.get("checks", {})
    expected_hashes = {
        "board_sha256": _sha(BOARD),
        "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
    }
    prechecks = {
        "frozen_inputs_exact": all(_frozen_checks().values()),
        "supervisor_pass": supervisor.get("pass") is True,
        "worker_pass": worker.get("pass") is True,
        "numeric_integrity_pass": (
            evidence.get("numeric_integrity_pass") is True
        ),
        "call29_rank_class_null": (
            evidence["nonclaims"]["call29_rank_or_class_adopted"]
            is None
            and evidence["nonclaims"][
                "authoritative_all_41_aggregate"
            ]
            is None
        ),
        "observability_pass": observability.get("pass") is True,
        "observability_chain_exact": all(
            observability_chain.values()
        ),
        "manual_identity_exact": (
            manual.get("artifact")
            == "D393_MANUAL_VISUAL_INSPECTION_V1"
            and manual.get("case") == CASE
            and manual.get("attempt") == ATTEMPT
        ),
        "manual_paths_exact": (
            manual.get("board_path") == _rel(BOARD)
            and manual.get("rerun_screenshot_path")
            == _rel(RERUN_SCREENSHOT)
        ),
        "manual_keys_exact": set(manual_checks) == expected_keys,
        "manual_checks_all_true": (
            set(manual_checks) == expected_keys
            and all(value is True for value in manual_checks.values())
        ),
        "manual_hashes_exact": (
            manual.get("artifact_hashes") == expected_hashes
            == template[
                "artifact_hashes_to_bind_after_actual_viewing"
            ]
        ),
        "manual_authority_links_exact": (
            manual.get("manual_template_sha256")
            == _sha(MANUAL_TEMPLATE)
            and manual.get("observability_claim_sha256")
            == _sha(OBSERVABILITY_CLAIM)
        ),
        "manual_observations_minimum_three": (
            isinstance(manual.get("observations"), list)
            and len(manual["observations"]) >= 3
            and all(
                isinstance(value, str) and value.strip()
                for value in manual["observations"]
            )
        ),
        "manual_complete": (
            manual.get("manual_inspection_complete") is True
        ),
        "one_worker_zero_retry_zero_signal": (
            supervisor.get("actual_worker_invocations") == 1
            and supervisor.get("retries") == 0
            and supervisor.get("process_signals_sent") == 0
        ),
        "one_viewer_zero_retry": (
            observability.get("viewer_invocations") == 1
            and observability.get("viewer_retries") == 0
        ),
        "status_scope_pass": _status_scope_pass(
            _status_lines(),
            _read_json(EXECUTION_AUTHORITY)["git"][
                "status_with_execution_authority"
            ],
            allowed_output_names=_out_names(),
        ),
    }
    if not all(prechecks.values()):
        raise RuntimeError(f"D393 finalize prechecks failed: {prechecks}")
    _phase("finalize_end")
    phase_contract = _phase_contract()
    if not all(
        phase_contract[key]
        for key in (
            "exact",
            "ordinal_exact",
            "monotonic_time_forward",
            "wall_time_forward",
        )
    ):
        raise RuntimeError(
            f"D393 phase contract failed: {phase_contract}"
        )
    completion = {
        "artifact": "D393_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "prechecks": prechecks,
        "phase_contract": phase_contract,
        "provenance_artifacts": {
            "script": _artifact(SCRIPT),
            "start": _artifact(START),
            "execution_authority": _artifact(EXECUTION_AUTHORITY),
            "preregistration": _artifact(PREREGISTRATION),
            "invocation": _artifact(INVOCATION),
            "authorization": _artifact(AUTHORIZATION),
            "sentinel": _artifact(SENTINEL),
            "progress": _artifact(PROGRESS),
            "evidence": _artifact(EVIDENCE),
            "geometry": _artifact(GEOMETRY),
            "csv": _artifact(CSV_PATH),
            "worker_claim": _artifact(WORKER_CLAIM),
            "supervisor": _artifact(SUPERVISOR),
            "board": _artifact(BOARD),
            "layout": _artifact(LAYOUT),
            "rrd": _artifact(RRD),
            "rbl": _artifact(RBL),
            "rerun_validation": _artifact(RERUN_VALIDATION),
            "rerun_screenshot": _artifact(RERUN_SCREENSHOT),
            "manual_template": _artifact(MANUAL_TEMPLATE),
            "observability_claim": _artifact(OBSERVABILITY_CLAIM),
            "manual_inspection": _artifact(MANUAL),
            "phase_markers": _artifact(PHASES),
        },
        "observability_chain_checks": observability_chain,
        "numeric_verdict": evidence["numeric_verdict"],
        "diagnostic_conclusion": evidence["diagnostic_conclusion"],
        "call29_authoritative_rank": None,
        "call29_authoritative_class": None,
        "authoritative_all_41_aggregate": None,
        "seam_or_overlap_verdict_updated": False,
        "actual_worker_invocations": 1,
        "worker_retries": 0,
        "viewer_invocations": 1,
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "failure_attestation_exists": False,
        "operational_verdict": (
            "D393_CALL29_FLOAT64_INTERSECTION_RESIDUE_PROVENANCE_"
            "LOCALIZATION_COMPLETE_NO_SEAM_ADOPTION"
        ),
        "g0a_pass": False,
        "pass": True,
    }
    _write_json_x(COMPLETION, completion)
    _require_names(FINAL_NAMES, "after_finalize")
    print(json.dumps(completion, ensure_ascii=False))
    return 0


def _dispatch(stage: str) -> int:
    if stage == "prepare":
        return _prepare()
    if stage == "run":
        return _run()
    if stage == "worker":
        return _worker_inner()
    if stage == "observe":
        return _observe()
    if stage == "finalize":
        return _finalize()
    raise ValueError(stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "run", "worker", "observe", "finalize"),
        required=True,
    )
    args = parser.parse_args()
    try:
        return _dispatch(args.stage)
    except Exception as exc:
        if OUT_DIR.is_dir():
            if args.stage == "worker":
                _worker_failure_claim(exc)
            else:
                _failure(args.stage, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
