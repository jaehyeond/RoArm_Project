#!/usr/bin/env python3
"""D391: repair rank-basis authority and clip-input immutability offline.

This case reads only the immutable D390 terminal-geometry JSON as its
scientific input.  It does not replay D389/D390 clipping, alter a tolerance,
select a vertex budget, materialize a collider, or invoke Isaac/PhysX.

The two registered variables are:

1. translation_and_order_stable_terminal_affine_rank_authority_v1
2. clip_plane_fixture_input_immutability_v1

The supervisor launches exactly one worker with no retry, timeout, or signal
authority.  D390 remains frozen and is never modified or finalized here.
"""

from __future__ import annotations

import argparse
import ast
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
START = REPO / "START_HERE.md"
CASE = "D391"
ATTEMPT = "attempt1_d390_rank_basis_and_clip_input_immutability_repair"
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d391"
    / ATTEMPT
)
D390_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d390"
    / "attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization"
)
D390_GEOMETRY = D390_DIR / "d390_terminal_candidate_geometry.json"
EXPECTED_D390_GEOMETRY_SHA256 = (
    "73fc986043b976bec26e1cc92643b8aab281a529f1c71c2918163ba7b98475c7"
)
EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
EXPECTED_START_SHA256 = (
    "315c8af4f9fe5e7a97c4746dd0a6700641436d3e3038128044bb82205e23b90b"
)
EXPECTED_D390_DIRECTORY_MANIFEST_SHA256 = (
    "8ceb1aa2b3d8ec6f543d4f9bccadb363164c2422a06285dfffdb3932d535209e"
)
EXPECTED_START_APPROVAL_MARKER = (
    "## Active Case — D391 Approved, Not Yet Executed"
)
EXECUTION_AUTHORITY_SHA256_ENV = (
    "D391_EXECUTION_AUTHORITY_SHA256"
)

VARIABLES = [
    "translation_and_order_stable_terminal_affine_rank_authority_v1",
    "clip_plane_fixture_input_immutability_v1",
]
DISPUTED_MANIFEST = [
    {
        "catalog_index": 0,
        "call_index": 3,
        "call_id": "upper_00_03_post_float32_rbl",
        "unique_point_count": 3,
        "points_sha256": (
            "54b85f19f64fc424f5f4c2eab1d184a3ad0cd5425704ef7a07316b3112cb8f2f"
        ),
    },
    {
        "catalog_index": 1,
        "call_index": 7,
        "call_id": "upper_01_03_pre_float32_rbl",
        "unique_point_count": 3,
        "points_sha256": (
            "2425f243071ff4acfe5f7552341f2fce91781dc7d989cdef66ef07e12e27e7fa"
        ),
    },
    {
        "catalog_index": 2,
        "call_index": 9,
        "call_id": "upper_01_03_post_float32_rbl",
        "unique_point_count": 3,
        "points_sha256": (
            "2425f243071ff4acfe5f7552341f2fce91781dc7d989cdef66ef07e12e27e7fa"
        ),
    },
    {
        "catalog_index": 3,
        "call_index": 12,
        "call_id": "upper_01_04_post_float32_rbl",
        "unique_point_count": 6,
        "points_sha256": (
            "90df50455942edfaf95254133f933719c88f88090cf3149f121d4ac370445214"
        ),
    },
    {
        "catalog_index": 4,
        "call_index": 27,
        "call_id": "lower_00_05_pre_float32_rbl",
        "unique_point_count": 2,
        "points_sha256": (
            "486743c9d7cd8d24d2c1ac429eb8802b194dc4457e92b4b813e7ab8da54952f8"
        ),
    },
    {
        "catalog_index": 5,
        "call_index": 29,
        "call_id": "lower_01_02_pre_float32_lbr",
        "unique_point_count": 6,
        "points_sha256": (
            "dcd4590e77d929d5abd4edb15f594d5956a9472f9ee099724b39544a7fdfddc6"
        ),
    },
]

FLOAT64_EPSILON = float(np.finfo(np.float64).eps)
THRESHOLD_ALPHAS = (0.5, 1.0, 2.0)
SCALE_EXPONENTS = (-20, -10, 10, 20)
COOPERATIVE_DEADLINE_SECONDS = 300.0
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
PLANE_FIXTURE_VALUES = {
    "EMPTY": [1.0, 0.0, 0.0, 1.0],
    "POINT": [1.0, 1.0, 1.0, 0.0],
    "LINE": [1.0, 1.0, 0.0, 0.0],
    "FACE_LIKE": [1.0, 0.0, 0.0, 0.0],
    "FULL_DIMENSIONAL": [1.0, 0.0, 0.0, -0.5],
}

PREREG = OUT_DIR / "d391_preregistration.json"
EXECUTION_AUTHORITY = OUT_DIR / "d391_execution_authority.json"
PHASES = OUT_DIR / "d391_phase_markers.jsonl"
INVOCATION = OUT_DIR / "d391_offline_worker_invocation.json"
AUTHORIZATION = OUT_DIR / "d391_worker_authorization.json"
SENTINEL = OUT_DIR / "d391_worker_start_sentinel.json"
STDOUT = OUT_DIR / "d391_offline_worker_stdout.log"
STDERR = OUT_DIR / "d391_offline_worker_stderr.log"
EVIDENCE = OUT_DIR / "d391_rank_and_plane_immutability_evidence.json"
GEOMETRY = OUT_DIR / "d391_disputed_terminal_geometry.json"
CSV_PATH = OUT_DIR / "d391_disputed_rank_catalog.csv"
BOARD = OUT_DIR / "d391_rank_basis_and_input_immutability_1920x1080.png"
LAYOUT = OUT_DIR / "d391_board_layout_validation.json"
RRD = OUT_DIR / "d391_rank_basis_and_input_immutability.rrd"
RBL = OUT_DIR / "d391_rank_basis_and_input_immutability.rbl"
RERUN_VALIDATION = OUT_DIR / "d391_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d391_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d391_manual_visual_inspection_template.json"
MANUAL = OUT_DIR / "d391_manual_visual_inspection.json"
WORKER_CLAIM = OUT_DIR / "d391_offline_worker_claim.json"
FAILURE = OUT_DIR / "d391_failure_attestation.json"
SUPERVISOR = OUT_DIR / "d391_offline_worker_supervisor.json"
COMPLETION = OUT_DIR / "d391_completion_summary.json"

AUTHORITY_INVENTORY = {EXECUTION_AUTHORITY.name}
PREPARE_INVENTORY = AUTHORITY_INVENTORY | {PREREG.name, PHASES.name}
PRE_WORKER_INVENTORY = PREPARE_INVENTORY | {
    INVOCATION.name,
    AUTHORIZATION.name,
    STDOUT.name,
    STDERR.name,
}
WORKER_START_INVENTORY = PRE_WORKER_INVENTORY | {SENTINEL.name}
PRE_CLAIM_INVENTORY = WORKER_START_INVENTORY | {
    EVIDENCE.name,
    GEOMETRY.name,
    CSV_PATH.name,
    BOARD.name,
    LAYOUT.name,
    RRD.name,
    RBL.name,
    RERUN_VALIDATION.name,
    RERUN_SCREENSHOT.name,
    MANUAL_TEMPLATE.name,
}
POST_WORKER_INVENTORY = PRE_CLAIM_INVENTORY | {WORKER_CLAIM.name}
POST_RUN_INVENTORY = POST_WORKER_INVENTORY | {SUPERVISOR.name}
PRE_FINALIZE_INVENTORY = POST_RUN_INVENTORY | {MANUAL.name}
POST_FINALIZE_INVENTORY = PRE_FINALIZE_INVENTORY | {COMPLETION.name}

_deadline_monotonic: float | None = None
_supervisor_process_started = False
_supervisor_worker_pid: int | None = None


class CooperativeDeadlineExceeded(RuntimeError):
    """Raised by the worker itself; the supervisor has no signal authority."""


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


def _array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(b"|float64|C|")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        _native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return _read_json(path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            _native(value),
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\r\n")


def _status_lines() -> list[str]:
    text = _git("status", "--porcelain=v1", "--untracked-files=all")
    return text.splitlines() if text else []


def _status_manifest_sha256(lines: list[str]) -> str:
    return _text_sha("\n".join(lines))


def _status_path(line: str) -> str:
    if len(line) < 4:
        return ""
    path = line[3:]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"')


def _status_scope_pass(
    current: list[str], baseline: list[str], *, allow_output: bool
) -> bool:
    if not allow_output:
        return current == baseline
    baseline_set = set(baseline)
    if not baseline_set.issubset(set(current)):
        return False
    for line in current:
        if line in baseline_set:
            continue
        if (
            line.startswith("?? ")
            and _status_path(line).startswith(
                "claudedocs/runtime_logs/grasp_track/g0a_d391/"
            )
        ):
            continue
        return False
    return True


def _directory_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": _rel(item),
            "bytes": item.stat().st_size,
            "sha256": _sha(item),
        }
        for item in sorted(path.iterdir())
        if item.is_file()
    ]


def _out_names() -> set[str]:
    if not OUT_DIR.is_dir():
        return set()
    return {path.name for path in OUT_DIR.iterdir()}


def _require_inventory(expected: set[str], stage: str) -> None:
    observed = _out_names()
    if observed != expected:
        raise RuntimeError(
            f"D391 output inventory mismatch at {stage}: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _phase(name: str, **payload: Any) -> None:
    row = {
        "phase": name,
        "monotonic_seconds": time.monotonic(),
        "wall_time_ns": time.time_ns(),
        **payload,
    }
    with PHASES.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _native(row),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def _deadline(where: str) -> None:
    if (
        _deadline_monotonic is not None
        and time.monotonic() > _deadline_monotonic
    ):
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


def _geometry_schema_checks(geometry: dict[str, Any]) -> dict[str, bool]:
    records = geometry.get("records", [])
    indexed = {row.get("call_index"): row for row in records}
    observed_manifest: list[dict[str, Any]] = []
    for expected in DISPUTED_MANIFEST:
        row = indexed.get(expected["call_index"], {})
        points = np.asarray(
            row.get("terminal_candidate_unique_points_f64_m", []),
            dtype=np.float64,
        ).reshape(-1, 3)
        observed_manifest.append(
            {
                "catalog_index": expected["catalog_index"],
                "call_index": row.get("call_index"),
                "call_id": row.get("call_id"),
                "unique_point_count": len(points),
                "points_sha256": (
                    _array_sha(_canonical_unique(points))
                    if len(points)
                    else _array_sha(np.empty((0, 3), dtype=np.float64))
                ),
            }
        )
    trace_rows = [
        trace
        for record in records
        for trace in record.get("plane_trace", [])
    ]
    trace_hash_matches = 0
    for trace in trace_rows:
        equation = np.asarray(
            trace.get("plane_equation_f64_m", []), dtype=np.float64
        )
        if (
            equation.shape == (4,)
            and _array_sha(equation.reshape(1, 4))
            == trace.get("plane_equation_sha256")
        ):
            trace_hash_matches += 1
    return {
        "artifact_exact": geometry.get("artifact")
        == "D390_TERMINAL_CANDIDATE_GEOMETRY_V1",
        "record_count_exact_41": geometry.get("record_count") == 41
        and len(records) == 41,
        "disputed_manifest_exact": observed_manifest == DISPUTED_MANIFEST,
        "stored_trace_count_exact_351": len(trace_rows) == 351,
        "stored_trace_plane_hashes_exact_351": trace_hash_matches == 351,
    }


def _frozen_checks(
    prereg: dict[str, Any], *, allow_output: bool
) -> dict[str, bool]:
    authority = _read_json(EXECUTION_AUTHORITY)
    baseline = authority.get("git", {}).get(
        "status_before_preregistration", []
    )
    start_text = START.read_text(encoding="utf-8")
    parent_manifest = _directory_manifest(D390_DIR)
    return {
        "execution_authority_checks_pass": all(
            _execution_authority_checks(
                authority, require_current_status_exact=False
            ).values()
        ),
        "execution_authority_hash_bound": prereg.get(
            "execution_authority", {}
        ).get("sha256")
        == _sha(EXECUTION_AUTHORITY),
        "execution_authority_path_bound": prereg.get(
            "execution_authority", {}
        ).get("path")
        == _rel(EXECUTION_AUTHORITY),
        "head_exact": _git("rev-parse", "HEAD")
        == EXPECTED_HEAD
        == prereg.get("git", {}).get("head"),
        "origin_master_exact": _git("rev-parse", "origin/master")
        == EXPECTED_HEAD
        == prereg.get("git", {}).get("origin_master"),
        "head_equals_origin": _git("rev-parse", "HEAD")
        == _git("rev-parse", "origin/master"),
        "script_hash_exact": _sha(SCRIPT)
        == authority.get("script", {}).get("sha256")
        == prereg.get("script", {}).get("sha256"),
        "start_here_hash_exact": _sha(START)
        == EXPECTED_START_SHA256
        == prereg.get("authorization", {}).get("sha256"),
        "start_here_approval_marker_exact": (
            EXPECTED_START_APPROVAL_MARKER in start_text
            and "`D391 [d390_rank_basis_and_clip_input_immutability_repair]`"
            in start_text
        ),
        "d390_geometry_hash_exact": _sha(D390_GEOMETRY)
        == EXPECTED_D390_GEOMETRY_SHA256
        == prereg.get("scientific_input", {}).get("sha256"),
        "d390_full_directory_manifest_exact": (
            parent_manifest == prereg.get("frozen_parent_manifest")
            and _canonical_sha(parent_manifest)
            == EXPECTED_D390_DIRECTORY_MANIFEST_SHA256
        ),
        "variables_exact": prereg.get("new_variables") == VARIABLES,
        "prepare_status_manifest_exact": (
            isinstance(baseline, list)
            and baseline
            == prereg.get("git", {}).get(
                "status_before_preregistration"
            )
            and len(baseline)
            == authority.get("git", {}).get("status_line_count")
            and _status_manifest_sha256(baseline)
            == authority.get("git", {}).get(
                "status_manifest_sha256"
            )
        ),
        "current_git_scope_exact": _status_scope_pass(
            _status_lines(), baseline, allow_output=allow_output
        ),
        "preregistration_hash_bound_to_prepare_end": (
            _prepare_prereg_hash() == _sha(PREREG)
        ),
    }


def _prepare_prereg_hash() -> str | None:
    if not PHASES.is_file():
        return None
    matches = [
        json.loads(line)
        for line in PHASES.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line).get("phase") == "prepare_end"
    ]
    if len(matches) != 1:
        return None
    return matches[0].get("preregistration_sha256")


def _execution_authority_checks(
    authority: dict[str, Any], *, require_current_status_exact: bool
) -> dict[str, bool]:
    status = authority.get("git", {}).get(
        "status_before_preregistration", []
    )
    parent_manifest = _directory_manifest(D390_DIR)
    start_text = START.read_text(encoding="utf-8")
    return {
        "artifact_exact": authority.get("artifact")
        == "D391_EXECUTION_AUTHORITY_V1",
        "case_attempt_exact": authority.get("case") == CASE
        and authority.get("attempt") == ATTEMPT,
        "approval_exact": authority.get("user_approval")
        == "D391 [d390_rank_basis_and_clip_input_immutability_repair]",
        "variables_exact": authority.get("new_variables") == VARIABLES,
        "script_path_exact": authority.get("script", {}).get("path")
        == _rel(SCRIPT),
        "script_sha_exact": authority.get("script", {}).get("sha256")
        == _sha(SCRIPT),
        "start_path_exact": authority.get("start_here", {}).get("path")
        == _rel(START),
        "start_sha_exact": authority.get("start_here", {}).get("sha256")
        == _sha(START)
        == EXPECTED_START_SHA256,
        "start_approval_marker_exact": (
            authority.get("start_here", {}).get("approval_marker")
            == EXPECTED_START_APPROVAL_MARKER
            and EXPECTED_START_APPROVAL_MARKER in start_text
        ),
        "head_origin_exact": authority.get("git", {}).get("head")
        == authority.get("git", {}).get("origin_master")
        == _git("rev-parse", "HEAD")
        == _git("rev-parse", "origin/master")
        == EXPECTED_HEAD,
        "status_manifest_internal_exact": (
            isinstance(status, list)
            and authority.get("git", {}).get("status_line_count")
            == len(status)
            and authority.get("git", {}).get("status_manifest_sha256")
            == _status_manifest_sha256(status)
        ),
        "current_status_exact_when_required": (
            not require_current_status_exact or _status_lines() == status
        ),
        "d390_geometry_exact": authority.get("d390", {}).get(
            "terminal_geometry_sha256"
        )
        == _sha(D390_GEOMETRY)
        == EXPECTED_D390_GEOMETRY_SHA256,
        "d390_parent_manifest_exact": authority.get("d390", {}).get(
            "directory_file_count"
        )
        == len(parent_manifest)
        == 22
        and authority.get("d390", {}).get("directory_manifest_sha256")
        == _canonical_sha(parent_manifest)
        == EXPECTED_D390_DIRECTORY_MANIFEST_SHA256,
        "output_path_exact": authority.get("output", {}).get("path")
        == _rel(OUT_DIR),
        "forward_only_exact": authority.get("output", {}).get(
            "forward_only_no_overwrite"
        )
        is True,
    }


def _canonical_unique(points: np.ndarray) -> np.ndarray:
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(raw) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("rank points contain NaN/Inf")
    rows = sorted({tuple(float(value) for value in row) for row in raw})
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def _fraction_point(point: Sequence[float]) -> tuple[Fraction, ...]:
    return tuple(Fraction.from_float(float(value)) for value in point)


def _exact_difference(
    left: Sequence[float], right: Sequence[float]
) -> tuple[Fraction, Fraction, Fraction]:
    left_f = _fraction_point(left)
    right_f = _fraction_point(right)
    return tuple(
        right_f[index] - left_f[index] for index in range(3)
    )  # type: ignore[return-value]


def _fraction_rank(rows: Sequence[Sequence[Fraction]]) -> int:
    matrix = [list(row) for row in rows if any(value != 0 for value in row)]
    if not matrix:
        return 0
    row_count = len(matrix)
    column_count = len(matrix[0])
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (
                index
                for index in range(pivot_row, row_count)
                if matrix[index][column] != 0
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        pivot_value = matrix[pivot_row][column]
        matrix[pivot_row] = [
            value / pivot_value for value in matrix[pivot_row]
        ]
        for index in range(row_count):
            if index == pivot_row:
                continue
            factor = matrix[index][column]
            if factor == 0:
                continue
            matrix[index] = [
                matrix[index][j] - factor * matrix[pivot_row][j]
                for j in range(column_count)
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def _basis_rows(
    unique: np.ndarray, *, anchor_index: int | None
) -> list[tuple[Fraction, Fraction, Fraction]]:
    count = len(unique)
    if anchor_index is None:
        return [
            _exact_difference(unique[left], unique[right])
            for left in range(count)
            for right in range(left + 1, count)
        ]
    return [
        _exact_difference(unique[anchor_index], unique[index])
        for index in range(count)
        if index != anchor_index
    ]


def _basis_diagnostic(
    rows: list[tuple[Fraction, Fraction, Fraction]],
    *,
    basis_name: str,
    unique_count: int,
    hard_cap: int,
) -> dict[str, Any]:
    matrix = (
        np.asarray(
            [[float(value) for value in row] for row in rows],
            dtype=np.float64,
        ).reshape(-1, 3)
        if rows
        else np.empty((0, 3), dtype=np.float64)
    )
    singular_raw = (
        np.linalg.svd(matrix, compute_uv=False, full_matrices=False)
        if len(matrix)
        else np.empty(0, dtype=np.float64)
    )
    singular = np.pad(
        singular_raw, (0, max(0, 3 - len(singular_raw)))
    )[:3]
    sigma_max = float(singular[0]) if len(singular) else 0.0
    tau = sigma_max * float(max(unique_count, 3)) * FLOAT64_EPSILON
    ranks: dict[str, dict[str, int]] = {}
    for alpha in THRESHOLD_ALPHAS:
        raw_rank = int(np.count_nonzero(singular > alpha * tau))
        ranks[str(alpha)] = {
            "raw_svd_rank": raw_rank,
            "bounded_affine_rank": min(raw_rank, hard_cap),
        }
    return {
        "basis_name": basis_name,
        "row_count": len(rows),
        "matrix_shape": list(matrix.shape),
        "matrix_f64_sha256": _array_sha(matrix),
        "exact_fraction_rows_sha256": _canonical_sha(rows),
        "exact_rank": _fraction_rank(rows),
        "singular_values": singular,
        "tau": tau,
        "tau_formula": (
            "sigma_max * max(unique_point_count,3) * float64_epsilon"
        ),
        "threshold_alphas": list(THRESHOLD_ALPHAS),
        "ranks": ranks,
        "hard_cap": hard_cap,
    }


def _class_for_rank(unique_count: int, rank: int | None) -> str | None:
    if rank is None:
        return None
    if unique_count == 0:
        return "EMPTY"
    return {
        0: "POINT",
        1: "LINE",
        2: "FACE_LIKE",
        3: "FULL_DIMENSIONAL",
    }.get(rank, "INVALID")


def _rank_core(points: np.ndarray) -> dict[str, Any]:
    unique = _canonical_unique(points)
    count = len(unique)
    hard_cap = min(3, max(0, count - 1))
    bases = [
        _basis_diagnostic(
            _basis_rows(unique, anchor_index=index),
            basis_name=f"anchor_{index}",
            unique_count=count,
            hard_cap=hard_cap,
        )
        for index in range(count)
    ]
    bases.append(
        _basis_diagnostic(
            _basis_rows(unique, anchor_index=None),
            basis_name="all_unordered_pairs",
            unique_count=count,
            hard_cap=hard_cap,
        )
    )
    exact_ranks = {row["exact_rank"] for row in bases}
    raw_ranks_by_alpha = {
        str(alpha): {
            row["ranks"][str(alpha)]["raw_svd_rank"] for row in bases
        }
        for alpha in THRESHOLD_ALPHAS
    }
    ranks_by_alpha = {
        str(alpha): {
            row["ranks"][str(alpha)]["bounded_affine_rank"]
            for row in bases
        }
        for alpha in THRESHOLD_ALPHAS
    }
    nominal = ranks_by_alpha["1.0"]
    basis_agreement = len(nominal) == 1
    threshold_agreement = (
        basis_agreement
        and all(values == nominal for values in ranks_by_alpha.values())
    )
    exact_agreement = len(exact_ranks) == 1
    exact_rank = next(iter(exact_ranks)) if exact_agreement else None
    nominal_rank = next(iter(nominal)) if basis_agreement else None
    bounded_rank_within_exact = (
        nominal_rank is not None
        and exact_rank is not None
        and nominal_rank <= exact_rank
    )
    raw_numeric_over_hard_cap_at_alpha_1 = any(
        rank > hard_cap for rank in raw_ranks_by_alpha["1.0"]
    )
    raw_numeric_over_exact_at_alpha_1 = (
        exact_rank is not None
        and any(
            rank > exact_rank for rank in raw_ranks_by_alpha["1.0"]
        )
    )
    hard_cap_correction_applied = (
        raw_numeric_over_hard_cap_at_alpha_1
        and basis_agreement
        and threshold_agreement
        and bounded_rank_within_exact
    )
    if not exact_agreement:
        status = "IMPLEMENTATION_FAIL_EXACT_BASIS"
        final_rank = None
    elif not basis_agreement:
        status = "NUMERICALLY_AMBIGUOUS_BASIS"
        final_rank = None
    elif not threshold_agreement:
        status = "NUMERICALLY_AMBIGUOUS_THRESHOLD"
        final_rank = None
    elif not bounded_rank_within_exact:
        status = "IMPLEMENTATION_FAIL_NUMERIC_GT_EXACT"
        final_rank = None
    else:
        status = (
            "STABLE_HARD_CAP_CORRECTED"
            if hard_cap_correction_applied
            else "STABLE"
        )
        final_rank = nominal_rank
    return {
        "raw_point_count": int(len(np.asarray(points).reshape(-1, 3))),
        "unique_point_count": count,
        "canonical_unique_points_f64_m": unique,
        "canonical_unique_points_sha256": _array_sha(unique),
        "hard_affine_rank_cap": hard_cap,
        "hard_cap_formula": "min(3,max(0,unique_point_count-1))",
        "exact_dyadic_rank": exact_rank,
        "basis_diagnostics": bases,
        "raw_ranks_by_alpha": {
            key: sorted(values)
            for key, values in raw_ranks_by_alpha.items()
        },
        "bounded_ranks_by_alpha": {
            key: sorted(values) for key, values in ranks_by_alpha.items()
        },
        "basis_agreement_at_alpha_1": basis_agreement,
        "threshold_band_agreement": threshold_agreement,
        "bounded_rank_within_exact": bounded_rank_within_exact,
        "raw_numeric_over_hard_cap_at_alpha_1": (
            raw_numeric_over_hard_cap_at_alpha_1
        ),
        "raw_numeric_over_exact_at_alpha_1": (
            raw_numeric_over_exact_at_alpha_1
        ),
        "hard_cap_correction_applied": hard_cap_correction_applied,
        "status": status,
        "authoritative_rank": final_rank,
        "authoritative_class": _class_for_rank(count, final_rank),
        "ambiguity_preserved": status.startswith("NUMERICALLY_AMBIGUOUS"),
        "final_rank_respects_hard_cap": (
            final_rank is None or final_rank <= hard_cap
        ),
    }


def _rank_signature(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "authoritative_rank": result["authoritative_rank"],
        "authoritative_class": result["authoritative_class"],
        "hard_affine_rank_cap": result["hard_affine_rank_cap"],
        "exact_dyadic_rank": result["exact_dyadic_rank"],
        "raw_numeric_over_hard_cap_at_alpha_1": result[
            "raw_numeric_over_hard_cap_at_alpha_1"
        ],
        "raw_numeric_over_exact_at_alpha_1": result[
            "raw_numeric_over_exact_at_alpha_1"
        ],
        "hard_cap_correction_applied": result[
            "hard_cap_correction_applied"
        ],
        "basis_nominal_bounded_ranks": [
            row["ranks"]["1.0"]["bounded_affine_rank"]
            for row in result["basis_diagnostics"]
        ],
        "basis_half_bounded_ranks": [
            row["ranks"]["0.5"]["bounded_affine_rank"]
            for row in result["basis_diagnostics"]
        ],
        "basis_double_bounded_ranks": [
            row["ranks"]["2.0"]["bounded_affine_rank"]
            for row in result["basis_diagnostics"]
        ],
    }


def _order_controls(
    raw_points: np.ndarray, baseline: dict[str, Any]
) -> dict[str, Any]:
    raw = np.asarray(raw_points, dtype=np.float64).reshape(-1, 3)
    signatures: set[str] = set()
    hashes: set[str] = set()
    permutation_count = 0
    for order in itertools.permutations(range(len(raw))):
        result = _rank_core(raw[list(order)])
        signatures.add(_canonical_sha(_rank_signature(result)))
        hashes.add(result["canonical_unique_points_sha256"])
        permutation_count += 1
    return {
        "exhaustive_permutation_count": permutation_count,
        "expected_factorial_count": math.factorial(len(raw)),
        "unique_rank_signature_count": len(signatures),
        "unique_canonical_points_hash_count": len(hashes),
        "baseline_signature_sha256": _canonical_sha(
            _rank_signature(baseline)
        ),
        "observed_signature_sha256": sorted(signatures),
        "pass": (
            permutation_count == math.factorial(len(raw))
            and signatures
            == {_canonical_sha(_rank_signature(baseline))}
            and hashes == {baseline["canonical_unique_points_sha256"]}
        ),
    }


def _scale_controls(
    raw_points: np.ndarray, baseline: dict[str, Any]
) -> dict[str, Any]:
    raw = np.asarray(raw_points, dtype=np.float64).reshape(-1, 3)
    base_unique = _canonical_unique(raw)
    rows: list[dict[str, Any]] = []
    for exponent in SCALE_EXPONENTS:
        scaled = np.ldexp(raw, exponent)
        restored = np.ldexp(scaled, -exponent)
        result = _rank_core(scaled)
        scaled_unique = _canonical_unique(scaled)
        exact_factor = Fraction(2) ** exponent
        exact_scale_checks: list[bool] = []
        singular_scale_checks: list[bool] = []
        tau_scale_checks: list[bool] = []
        for basis_index, base_diagnostic in enumerate(
            baseline["basis_diagnostics"]
        ):
            anchor = (
                basis_index
                if basis_index < len(base_unique)
                else None
            )
            base_rows = _basis_rows(base_unique, anchor_index=anchor)
            scaled_rows = _basis_rows(
                scaled_unique, anchor_index=anchor
            )
            exact_scale_checks.append(
                scaled_rows
                == [
                    tuple(value * exact_factor for value in row)
                    for row in base_rows
                ]
            )
            scaled_diagnostic = result["basis_diagnostics"][basis_index]
            singular_scale_checks.append(
                bool(
                    np.allclose(
                        np.asarray(
                            scaled_diagnostic["singular_values"],
                            dtype=np.float64,
                        ),
                        np.ldexp(
                            np.asarray(
                                base_diagnostic["singular_values"],
                                dtype=np.float64,
                            ),
                            exponent,
                        ),
                        rtol=128.0 * FLOAT64_EPSILON,
                        atol=0.0,
                    )
                )
            )
            tau_scale_checks.append(
                math.isclose(
                    float(scaled_diagnostic["tau"]),
                    math.ldexp(float(base_diagnostic["tau"]), exponent),
                    rel_tol=128.0 * FLOAT64_EPSILON,
                    abs_tol=0.0,
                )
            )
        checks = {
            "scaled_finite": bool(np.isfinite(scaled).all()),
            "inverse_scale_bit_exact": _array_sha(restored)
            == _array_sha(raw),
            "rank_signature_exact": _rank_signature(result)
            == _rank_signature(baseline),
            "every_exact_basis_row_scales_by_power_of_two": all(
                exact_scale_checks
            ),
            "every_singular_value_scales_within_128eps": all(
                singular_scale_checks
            ),
            "every_tau_scales_within_128eps": all(tau_scale_checks),
        }
        rows.append(
            {
                "power_of_two_exponent": exponent,
                "scaled_points_sha256": _array_sha(scaled),
                "restored_points_sha256": _array_sha(restored),
                "rank_signature": _rank_signature(result),
                "basis_exact_scale_checks": exact_scale_checks,
                "basis_singular_scale_checks": singular_scale_checks,
                "basis_tau_scale_checks": tau_scale_checks,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    return {
        "exponents": list(SCALE_EXPONENTS),
        "rows": rows,
        "pass": all(row["pass"] for row in rows),
    }


def _translation_control(
    raw_points: np.ndarray, baseline: dict[str, Any]
) -> dict[str, Any]:
    unique = _canonical_unique(raw_points)
    translation = (
        Fraction(1, 2**20),
        Fraction(-1, 2**30),
        Fraction(1, 2**10),
    )
    translated = [
        tuple(_fraction_point(point)[axis] + translation[axis] for axis in range(3))
        for point in unique
    ]
    basis_rows: list[dict[str, Any]] = []
    for basis_index, baseline_diagnostic in enumerate(
        baseline["basis_diagnostics"]
    ):
        anchor = basis_index if basis_index < len(unique) else None
        baseline_rows = _basis_rows(unique, anchor_index=anchor)
        if anchor is None:
            translated_rows = [
                tuple(
                    translated[right][axis] - translated[left][axis]
                    for axis in range(3)
                )
                for left in range(len(translated))
                for right in range(left + 1, len(translated))
            ]
        else:
            translated_rows = [
                tuple(
                    translated[index][axis] - translated[anchor][axis]
                    for axis in range(3)
                )
                for index in range(len(translated))
                if index != anchor
            ]
        translated_diagnostic = _basis_diagnostic(
            translated_rows,
            basis_name=baseline_diagnostic["basis_name"],
            unique_count=len(unique),
            hard_cap=baseline["hard_affine_rank_cap"],
        )
        checks = {
            "exact_fraction_rows_equal": translated_rows == baseline_rows,
            "float64_matrix_hash_equal": (
                translated_diagnostic["matrix_f64_sha256"]
                == baseline_diagnostic["matrix_f64_sha256"]
            ),
            "exact_rank_equal": (
                translated_diagnostic["exact_rank"]
                == baseline_diagnostic["exact_rank"]
            ),
            "threshold_rank_map_equal": (
                translated_diagnostic["ranks"]
                == baseline_diagnostic["ranks"]
            ),
        }
        basis_rows.append(
            {
                "basis_name": baseline_diagnostic["basis_name"],
                "baseline_fraction_sha256": _canonical_sha(
                    baseline_rows
                ),
                "translated_fraction_sha256": _canonical_sha(
                    translated_rows
                ),
                "translated_float64_matrix_sha256": (
                    translated_diagnostic["matrix_f64_sha256"]
                ),
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    return {
        "translation_exact_dyadic": translation,
        "rank_basis_uses_exact_differences_before_float64_svd": True,
        "baseline_signature_sha256": _canonical_sha(
            _rank_signature(baseline)
        ),
        "basis_rows": basis_rows,
        "pass": all(row["pass"] for row in basis_rows),
    }


def _polyhedron_edges(points: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(source)
    groups: dict[tuple[float, ...], set[int]] = {}
    for simplex, equation in zip(
        hull.simplices, hull.equations, strict=True
    ):
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
    return np.asarray(
        sorted(pair for pair, count in memberships.items() if count >= 2),
        dtype=np.int64,
    )


def _normalize_plane_copy(equation: np.ndarray) -> np.ndarray:
    source = np.asarray(equation, dtype=np.float64)
    if source.shape != (4,):
        raise ValueError("plane equation must have shape (4,)")
    unit = np.array(source[:3], dtype=np.float64, copy=True)
    length = float(np.linalg.norm(unit))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("plane normal is invalid")
    unit /= length
    output = np.empty(4, dtype=np.float64)
    output[:3] = unit
    output[3] = float(source[3]) / length
    return output


def _clip_candidate_copy(
    points: np.ndarray, equation: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(points, dtype=np.float64)
    original = np.asarray(equation, dtype=np.float64)
    normalized = _normalize_plane_copy(original)
    values = source @ normalized[:3] + float(normalized[3])
    keep = values <= 0.0
    output = [point for point in source[keep]]
    edges = _polyhedron_edges(source)
    crossing_count = 0
    for left, right in edges:
        value_left = values[int(left)]
        value_right = values[int(right)]
        if (value_left < 0.0 < value_right) or (
            value_right < 0.0 < value_left
        ):
            ratio = -value_left / (value_right - value_left)
            output.append(
                source[int(left)]
                + ratio * (source[int(right)] - source[int(left)])
            )
            crossing_count += 1
    raw = np.asarray(output, dtype=np.float64).reshape(-1, 3)
    return raw, {
        "source_point_count": len(source),
        "kept_source_vertex_count": int(np.count_nonzero(keep)),
        "source_edge_count": len(edges),
        "strict_crossing_edge_count": crossing_count,
        "minimum_signed_value": float(values.min()),
        "maximum_signed_value": float(values.max()),
        "normalized_working_copy": normalized,
    }


def _class_and_threshold_controls() -> dict[str, Any]:
    class_fixtures = {
        "EMPTY": np.empty((0, 3), dtype=np.float64),
        "POINT": np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
        ),
        "LINE": np.asarray(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        "FACE_LIKE": np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "FULL_DIMENSIONAL": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    }
    class_rows: dict[str, Any] = {}
    for expected, points in class_fixtures.items():
        base = _rank_core(points)
        transformed = (
            points[:, [2, 0, 1]] * 8.0
            + np.asarray([0.125, -0.25, 0.5], dtype=np.float64)
            if len(points)
            else points
        )
        if len(transformed):
            transformed = np.vstack(
                [transformed[::-1], transformed[:1], transformed[:1]]
            )
        altered = _rank_core(transformed)
        checks = {
            "base_stable_expected_class": (
                base["status"].startswith("STABLE")
                and base["authoritative_class"] == expected
            ),
            "transformed_stable_expected_class": (
                altered["status"].startswith("STABLE")
                and altered["authoritative_class"] == expected
            ),
            "base_hard_cap_respected": base[
                "final_rank_respects_hard_cap"
            ],
            "transformed_hard_cap_respected": altered[
                "final_rank_respects_hard_cap"
            ],
        }
        class_rows[expected] = {
            "base": _rank_signature(base),
            "transformed": _rank_signature(altered),
            "checks": checks,
            "pass": all(checks.values()),
        }

    def thin_tetra(height: float) -> np.ndarray:
        return np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, height],
            ],
            dtype=np.float64,
        )

    low_height = math.ldexp(1.0, -60)
    high_height = math.ldexp(1.0, -40)
    boundary_height = math.ldexp(1.0, -50)
    low = _rank_core(thin_tetra(low_height))
    high = _rank_core(thin_tetra(high_height))
    boundary = _rank_core(thin_tetra(boundary_height))
    direct_boundary_rows = [
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1, 2**50)),
    ]
    direct_boundary = _basis_diagnostic(
        direct_boundary_rows,
        basis_name="registered_threshold_boundary_diagonal",
        unique_count=4,
        hard_cap=3,
    )
    direct_boundary_bounded_ranks = {
        alpha: direct_boundary["ranks"][alpha]["bounded_affine_rank"]
        for alpha in ("0.5", "1.0", "2.0")
    }
    direct_boundary_status = (
        "NUMERICALLY_AMBIGUOUS_THRESHOLD"
        if len(set(direct_boundary_bounded_ranks.values())) > 1
        else "STABLE"
    )
    threshold_checks = {
        "low_stable_face_like": (
            low["status"].startswith("STABLE")
            and low["authoritative_class"] == "FACE_LIKE"
        ),
        "high_stable_full_dimensional": (
            high["status"].startswith("STABLE")
            and high["authoritative_class"] == "FULL_DIMENSIONAL"
        ),
        "boundary_point_fixture_preserves_ambiguity": (
            boundary["status"]
            in {
                "NUMERICALLY_AMBIGUOUS_BASIS",
                "NUMERICALLY_AMBIGUOUS_THRESHOLD",
            }
        ),
        "direct_threshold_policy_control_is_ambiguous": (
            direct_boundary_status
            == "NUMERICALLY_AMBIGUOUS_THRESHOLD"
        ),
        "low_hard_cap_respected": low["final_rank_respects_hard_cap"],
        "high_hard_cap_respected": high["final_rank_respects_hard_cap"],
    }
    return {
        "class_controls": class_rows,
        "rank_threshold_straddle_controls": {
            "low_height_power_of_two": low_height,
            "high_height_power_of_two": high_height,
            "boundary_height_power_of_two": boundary_height,
            "low": _rank_signature(low),
            "high": _rank_signature(high),
            "boundary": _rank_signature(boundary),
            "direct_threshold_boundary": {
                "basis_diagnostic": direct_boundary,
                "bounded_ranks_by_alpha": direct_boundary_bounded_ranks,
                "derived_status": direct_boundary_status,
            },
            "checks": threshold_checks,
            "pass": all(threshold_checks.values()),
        },
        "pass": all(row["pass"] for row in class_rows.values())
        and all(threshold_checks.values()),
    }


def _plane_controls(geometry: dict[str, Any]) -> dict[str, Any]:
    cube = np.asarray(
        [
            [x, y, z]
            for x in (0.0, 1.0)
            for y in (0.0, 1.0)
            for z in (0.0, 1.0)
        ],
        dtype=np.float64,
    )
    fixtures = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in PLANE_FIXTURE_VALUES.items()
    }
    preregistered_fixture_hashes = _read_json(PREREG)["method"][
        "plane_input"
    ]["synthetic_fixture_sha256"]
    fixture_rows: dict[str, Any] = {}
    for expected, equation in fixtures.items():
        authored_before = equation.copy()
        before = _array_sha(equation.reshape(1, 4))
        candidate, clip_meta = _clip_candidate_copy(cube, equation)
        after = _array_sha(equation.reshape(1, 4))
        rank = _rank_core(candidate)
        binary_exact_before = all(
            float(value).is_integer() or value in {-0.5, 0.5}
            for value in authored_before
        )
        binary_exact_after = all(
            float(value).is_integer() or value in {-0.5, 0.5}
            for value in equation
        )
        checks = {
            "caller_plane_pre_post_hash_exact": before == after,
            "runtime_fixture_values_exact_registered": (
                equation.tolist() == PLANE_FIXTURE_VALUES[expected]
            ),
            "runtime_fixture_hash_exact_preregistered": (
                before == preregistered_fixture_hashes[expected]
            ),
            "binary_exact_fixture_before": binary_exact_before,
            "binary_exact_fixture_after": binary_exact_after,
            "working_copy_does_not_share_memory": not np.shares_memory(
                equation, clip_meta["normalized_working_copy"]
            ),
            "candidate_finite": bool(np.isfinite(candidate).all()),
            "stable_expected_class": rank["status"].startswith("STABLE")
            and rank["authoritative_class"] == expected,
        }
        fixture_rows[expected] = {
            "input_plane": equation,
            "input_sha256_before": before,
            "input_sha256_after": after,
            "clip_metadata": clip_meta,
            "candidate_sha256": _array_sha(candidate),
            "rank_signature": _rank_signature(rank),
            "checks": checks,
            "pass": all(checks.values()),
        }

    regression_input = np.asarray(
        [2.0, 0.0, 0.0, -1.0], dtype=np.float64
    )
    regression_before = _array_sha(regression_input.reshape(1, 4))
    regression_normalized = _normalize_plane_copy(regression_input)
    regression_after = _array_sha(regression_input.reshape(1, 4))
    regression_checks = {
        "caller_pre_post_hash_exact": (
            regression_before == regression_after
        ),
        "working_copy_no_shared_memory": not np.shares_memory(
            regression_input, regression_normalized
        ),
        "normal_and_offset_both_normalized_exact": np.array_equal(
            regression_normalized,
            np.asarray([1.0, 0.0, 0.0, -0.5], dtype=np.float64),
        ),
    }
    normalization_offset_regression = {
        "input_plane": regression_input,
        "input_sha256_before": regression_before,
        "input_sha256_after": regression_after,
        "normalized_working_copy": regression_normalized,
        "checks": regression_checks,
        "pass": all(regression_checks.values()),
    }

    real_rows: list[dict[str, Any]] = []
    for record in geometry["records"]:
        for trace in record["plane_trace"]:
            equation = np.asarray(
                trace["plane_equation_f64_m"], dtype=np.float64
            )
            before = _array_sha(equation.reshape(1, 4))
            working = _normalize_plane_copy(equation)
            after = _array_sha(equation.reshape(1, 4))
            real_rows.append(
                {
                    "call_id": record["call_id"],
                    "selected_plane_index_zero_based": trace[
                        "selected_plane_index_zero_based"
                    ],
                    "stored_sha256": trace["plane_equation_sha256"],
                    "before_sha256": before,
                    "after_sha256": after,
                    "working_copy_sha256": _array_sha(
                        working.reshape(1, 4)
                    ),
                    "stored_before_after_exact": (
                        trace["plane_equation_sha256"] == before == after
                    ),
                    "working_copy_no_shared_memory": not np.shares_memory(
                        equation, working
                    ),
                    "working_copy_finite": bool(np.isfinite(working).all()),
                }
            )
    real_pass_count = sum(
        row["stored_before_after_exact"]
        and row["working_copy_no_shared_memory"]
        and row["working_copy_finite"]
        for row in real_rows
    )
    return {
        "synthetic_fixture_count": len(fixture_rows),
        "synthetic_fixtures": fixture_rows,
        "synthetic_pass_count": sum(
            row["pass"] for row in fixture_rows.values()
        ),
        "nonunit_normal_nonzero_offset_regression": (
            normalization_offset_regression
        ),
        "real_trace_policy": (
            "copy-normalization/hash only; no D389/D390 clipping replay"
        ),
        "real_trace_plane_count": len(real_rows),
        "real_trace_plane_pass_count": real_pass_count,
        "real_trace_rows": real_rows,
        "pass": (
            all(row["pass"] for row in fixture_rows.values())
            and normalization_offset_regression["pass"]
            and len(real_rows) == 351
            and real_pass_count == 351
        ),
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    geometry = _read_json(D390_GEOMETRY)
    schema_checks = _geometry_schema_checks(geometry)
    if not all(schema_checks.values()):
        raise RuntimeError(f"D390 geometry schema failed: {schema_checks}")
    by_index = {row["call_index"]: row for row in geometry["records"]}
    records: list[dict[str, Any]] = []
    display_records: list[dict[str, Any]] = []
    for expected in DISPUTED_MANIFEST:
        _deadline(f"before_rank_call_{expected['catalog_index']}")
        source = by_index[expected["call_index"]]
        points = np.asarray(
            source["terminal_candidate_unique_points_f64_m"],
            dtype=np.float64,
        )
        rank = _rank_core(points)
        order = _order_controls(points, rank)
        scale = _scale_controls(points, rank)
        translation = _translation_control(points, rank)
        scientific_checks = {
            "source_manifest_exact": (
                source["call_id"] == expected["call_id"]
                and source["terminal_candidate_unique_points_sha256"]
                == expected["points_sha256"]
                and len(points) == expected["unique_point_count"]
            ),
            "canonical_points_hash_matches_source": (
                rank["canonical_unique_points_sha256"]
                == expected["points_sha256"]
            ),
            "final_rank_respects_n_minus_one": rank[
                "final_rank_respects_hard_cap"
            ],
            "terminal_status_admissible": rank["status"]
            in {
                "STABLE",
                "STABLE_HARD_CAP_CORRECTED",
                "NUMERICALLY_AMBIGUOUS_BASIS",
                "NUMERICALLY_AMBIGUOUS_THRESHOLD",
            },
            "order_control_pass": order["pass"],
            "translation_control_pass": translation["pass"],
            "scale_control_pass": scale["pass"],
        }
        record = {
            **expected,
            "d390_historical_rank": source["affine_rank"],
            "d390_historical_class": source["affine_class"],
            "d390_historical_values_are_not_answer_key": True,
            "rank_authority": rank,
            "order_control": order,
            "translation_control": translation,
            "scale_control": scale,
            "scientific_checks": scientific_checks,
            "pass": all(scientific_checks.values()),
        }
        records.append(record)

        unique = rank["canonical_unique_points_f64_m"]
        if len(unique):
            center = unique.mean(axis=0)
            centered = unique - center
            span = float(np.max(np.linalg.norm(centered, axis=1)))
            display_scale = 1.0 / span if span > 0.0 else 1.0
            displayed = centered * display_scale
        else:
            center = np.zeros(3, dtype=np.float64)
            display_scale = 1.0
            displayed = unique.copy()
        display_records.append(
            {
                "catalog_index": expected["catalog_index"],
                "call_index": expected["call_index"],
                "call_id": expected["call_id"],
                "canonical_terminal_points_f64_m": unique,
                "canonical_terminal_points_sha256": expected[
                    "points_sha256"
                ],
                "display_center_f64_m": center,
                "display_scale_inspection_only": display_scale,
                "display_points_f64": displayed,
                "display_points_sha256_not_scientific": _array_sha(
                    displayed
                ),
                "display_role": (
                    "centered/scaled Float32 inspection copy only; never "
                    "hashed back into the rank gate"
                ),
            }
        )
    class_controls = _class_and_threshold_controls()
    plane_controls = _plane_controls(geometry)
    stable_count = sum(
        row["rank_authority"]["status"].startswith("STABLE")
        for row in records
    )
    hard_cap_corrected_count = sum(
        row["rank_authority"]["status"]
        == "STABLE_HARD_CAP_CORRECTED"
        for row in records
    )
    ambiguous_count = sum(
        row["rank_authority"]["ambiguity_preserved"] for row in records
    )
    case_checks = {
        "d390_geometry_schema_pass": all(schema_checks.values()),
        "six_disputed_only": len(records) == 6,
        "six_rank_contracts_pass": all(row["pass"] for row in records),
        "every_call_stable_or_explicitly_ambiguous": (
            stable_count + ambiguous_count == 6
        ),
        "class_and_threshold_controls_pass": class_controls["pass"],
        "clip_plane_input_immutability_controls_pass": plane_controls["pass"],
        "real_trace_baseline_exact_351_of_351": (
            plane_controls["real_trace_plane_count"] == 351
            and plane_controls["real_trace_plane_pass_count"] == 351
        ),
    }
    evidence = {
        "artifact": "D391_RANK_AND_CLIP_INPUT_IMMUTABILITY_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "scientific_input": {
            "path": _rel(D390_GEOMETRY),
            "sha256": _sha(D390_GEOMETRY),
            "role": "only scientific input",
        },
        "rank_authority": {
            "canonical_order": "numeric lexicographic unique Float64 rows",
            "difference_arithmetic": (
                "exact dyadic Fraction differences before Float64 SVD"
            ),
            "bases": "all anchors plus all unordered point pairs",
            "threshold_formula": (
                "sigma_max * max(unique_point_count,3) * float64_epsilon"
            ),
            "threshold_alphas": list(THRESHOLD_ALPHAS),
            "hard_cap": "min(3,max(0,unique_point_count-1))",
            "ambiguity_policy": (
                "basis/threshold disagreement remains explicitly ambiguous"
            ),
            "exact_rank_role": (
                "bit-geometry upper authority and implementation check; "
                "not sole physical-dimensionality authority"
            ),
        },
        "schema_checks": schema_checks,
        "disputed_records": records,
        "disputed_summary": {
            "count": len(records),
            "stable_count": stable_count,
            "hard_cap_corrected_count": hard_cap_corrected_count,
            "ambiguous_count": ambiguous_count,
            "status_counts": {
                status: sum(
                    row["rank_authority"]["status"] == status
                    for row in records
                )
                for status in sorted(
                    {row["rank_authority"]["status"] for row in records}
                )
            },
            "stable_class_counts": {
                class_name: sum(
                    row["rank_authority"]["authoritative_class"]
                    == class_name
                    for row in records
                )
                for class_name in (
                    "EMPTY",
                    "POINT",
                    "LINE",
                    "FACE_LIKE",
                    "FULL_DIMENSIONAL",
                )
            },
        },
        "class_and_threshold_controls": class_controls,
        "plane_input_immutability": plane_controls,
        "case_checks": case_checks,
        "numeric_pass": all(case_checks.values()),
        "numeric_verdict": (
            (
                "D391_RANK_CONTRACT_PASS_WITH_EXPLICIT_AMBIGUITY_"
                "CLIP_INPUT_IMMUTABILITY_PASS"
                if ambiguous_count > 0
                else "D391_RANK_BASIS_AND_CLIP_INPUT_IMMUTABILITY_PASS"
            )
            if all(case_checks.values())
            else "D391_RANK_BASIS_OR_CLIP_INPUT_IMMUTABILITY_FAIL_STOP"
        ),
        "nonclaims": {
            "d390_repaired": False,
            "d390_retroactive_pass": False,
            "authoritative_all_41_aggregate": None,
            "d389_or_d390_clipping_reexecuted": 0,
            "qj_or_random_jitter": 0,
            "epsilon_or_5nm_or_tolerance_changes": 0,
            "partition_or_budget_or_geometry_changes": 0,
            "selected_vertex_budget": None,
            "adopted_vertex_budget": None,
            "collider_or_asset_or_usd_materialization": 0,
            "isaac_kit_physx_warp_cuda": 0,
            "cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_or_settings_changes": 0,
            "physics_or_grasp_result": None,
            "g0a_pass": False,
        },
        "algorithm_elapsed_seconds": time.monotonic() - started,
    }
    geometry_output = {
        "artifact": "D391_DISPUTED_TERMINAL_GEOMETRY_V1",
        "scientific_authority": (
            "canonical D390 Float64 points and D391 JSON hashes"
        ),
        "rerun_role": (
            "centered/scaled Float32 inspection copy; not scientific authority"
        ),
        "records": display_records,
    }
    return evidence, geometry_output, records


def _write_csv(records: list[dict[str, Any]]) -> None:
    fields = [
        "catalog_index",
        "call_index",
        "call_id",
        "unique_point_count",
        "hard_rank_cap",
        "d390_historical_rank",
        "d390_historical_class",
        "d391_status",
        "d391_authoritative_rank",
        "d391_authoritative_class",
        "exact_dyadic_rank",
        "order_permutations",
        "order_pass",
        "translation_pass",
        "scale_pass",
        "record_pass",
    ]
    with CSV_PATH.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            rank = row["rank_authority"]
            writer.writerow(
                {
                    "catalog_index": row["catalog_index"],
                    "call_index": row["call_index"],
                    "call_id": row["call_id"],
                    "unique_point_count": row["unique_point_count"],
                    "hard_rank_cap": rank["hard_affine_rank_cap"],
                    "d390_historical_rank": row["d390_historical_rank"],
                    "d390_historical_class": row["d390_historical_class"],
                    "d391_status": rank["status"],
                    "d391_authoritative_rank": rank[
                        "authoritative_rank"
                    ],
                    "d391_authoritative_class": rank[
                        "authoritative_class"
                    ],
                    "exact_dyadic_rank": rank["exact_dyadic_rank"],
                    "order_permutations": row["order_control"][
                        "exhaustive_permutation_count"
                    ],
                    "order_pass": row["order_control"]["pass"],
                    "translation_pass": row["translation_control"]["pass"],
                    "scale_pass": row["scale_control"]["pass"],
                    "record_pass": row["pass"],
                }
            )


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _project_points(
    points: np.ndarray, axes: tuple[int, int], box: tuple[int, int, int, int]
) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(source):
        return []
    plane = source[:, list(axes)]
    low = plane.min(axis=0)
    high = plane.max(axis=0)
    span = np.maximum(high - low, 1.0e-30)
    normalized = (plane - low) / span
    if np.all(high == low):
        normalized[:] = 0.5
    elif high[0] == low[0]:
        normalized[:, 0] = 0.5
    elif high[1] == low[1]:
        normalized[:, 1] = 0.5
    pad = 14
    return [
        (
            x0 + pad + float(row[0]) * (x1 - x0 - 2 * pad),
            y1 - pad - float(row[1]) * (y1 - y0 - 2 * pad),
        )
        for row in normalized
    ]


def _render_board(
    evidence: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    image = Image.new("RGB", (1920, 1080), (247, 249, 252))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(38)
    subtitle_font = _load_font(23)
    card_title_font = _load_font(22)
    body_font = _load_font(18)
    tiny_font = _load_font(15)
    text_boxes: list[dict[str, Any]] = []

    def text_at(
        xy: tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: tuple[int, int, int],
        *,
        tag: str,
    ) -> None:
        draw.text(xy, text, font=font, fill=fill)
        bbox = draw.textbbox(xy, text, font=font)
        text_boxes.append({"tag": tag, "bbox": list(bbox), "text": text})

    text_at(
        (52, 24),
        "D391: 순위 계산 기준과 평면 입력 불변성 검증",
        title_font,
        (20, 35, 55),
        tag="title",
    )
    text_at(
        (54, 75),
        "빨강=동결 점, 청록=모든 점쌍 연결 · 각 투영은 육안검사용 확대이며 수치 판정에는 사용하지 않음",
        subtitle_font,
        (55, 70, 90),
        tag="subtitle",
    )
    display_by_index = {
        row["catalog_index"]: row for row in geometry["records"]
    }
    cards: list[dict[str, Any]] = []
    card_w, card_h = 590, 350
    x_values = (45, 665, 1285)
    y_values = (120, 495)
    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    for row in evidence["disputed_records"]:
        index = row["catalog_index"]
        column = index % 3
        line = index // 3
        x0, y0 = x_values[column], y_values[line]
        x1, y1 = x0 + card_w, y0 + card_h
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=18,
            fill=(255, 255, 255),
            outline=(175, 185, 198),
            width=2,
        )
        rank = row["rank_authority"]
        result_color = (
            (20, 125, 80)
            if rank["status"].startswith("STABLE")
            else (190, 105, 15)
        )
        text_at(
            (x0 + 18, y0 + 12),
            f"C{index} · {row['call_id']}",
            card_title_font,
            (20, 35, 55),
            tag=f"c{index}_title",
        )
        result_text = (
            f"결과: {rank['authoritative_class']} / rank "
            f"{rank['authoritative_rank']} ({rank['status']})"
            if rank["status"].startswith("STABLE")
            else f"결과: {rank['status']}"
        )
        text_at(
            (x0 + 18, y0 + 47),
            result_text,
            body_font,
            result_color,
            tag=f"c{index}_result",
        )
        text_at(
            (x0 + 18, y0 + 75),
            (
                f"점 {row['unique_point_count']} · 수학 상한 "
                f"{rank['hard_affine_rank_cap']} · exact "
                f"{rank['exact_dyadic_rank']} · D390 기록 "
                f"{row['d390_historical_rank']}"
            ),
            tiny_font,
            (70, 80, 95),
            tag=f"c{index}_metrics",
        )
        display = np.asarray(
            display_by_index[index]["display_points_f64"],
            dtype=np.float64,
        )
        panel_y0, panel_y1 = y0 + 115, y1 - 20
        panel_width = 175
        for projection_index, (axis_a, axis_b, label) in enumerate(
            projections
        ):
            px0 = x0 + 18 + projection_index * 188
            px1 = px0 + panel_width
            draw.rectangle(
                (px0, panel_y0, px1, panel_y1),
                fill=(247, 251, 252),
                outline=(205, 214, 222),
                width=1,
            )
            text_at(
                (px0 + 6, panel_y0 + 4),
                label,
                tiny_font,
                (75, 85, 100),
                tag=f"c{index}_{label}",
            )
            projected = _project_points(
                display,
                (axis_a, axis_b),
                (px0 + 4, panel_y0 + 25, px1 - 4, panel_y1 - 4),
            )
            for left in range(len(projected)):
                for right in range(left + 1, len(projected)):
                    draw.line(
                        (projected[left], projected[right]),
                        fill=(20, 145, 145),
                        width=2,
                    )
            for point in projected:
                radius = 6
                draw.ellipse(
                    (
                        point[0] - radius,
                        point[1] - radius,
                        point[0] + radius,
                        point[1] + radius,
                    ),
                    fill=(220, 45, 55),
                    outline=(120, 15, 25),
                    width=1,
                )
        cards.append(
            {
                "catalog_index": index,
                "bbox": [x0, y0, x1, y1],
                "status": rank["status"],
            }
        )
    footer_y = 875
    draw.rounded_rectangle(
        (45, footer_y, 1875, 1045),
        radius=18,
        fill=(232, 241, 250),
        outline=(135, 160, 185),
        width=2,
    )
    summary = evidence["disputed_summary"]
    plane = evidence["plane_input_immutability"]
    text_at(
        (68, footer_y + 18),
        (
            f"6개 재판정: 안정 {summary['stable_count']} · 모호함 보존 "
            f"{summary['ambiguous_count']} · 평면 대조군 "
            f"{plane['synthetic_pass_count']}/{plane['synthetic_fixture_count']} "
            f"· 동결 실측 평면 {plane['real_trace_plane_pass_count']}/"
            f"{plane['real_trace_plane_count']}"
        ),
        subtitle_font,
        (20, 60, 100),
        tag="footer_summary",
    )
    text_at(
        (68, footer_y + 62),
        (
            "판정 권위: 정확한 점 차이 + 모든 기준점/점쌍 + 순서 전수 + "
            "2의 거듭제곱 배율 대조"
        ),
        body_font,
        (40, 65, 90),
        tag="footer_method",
    )
    text_at(
        (68, footer_y + 96),
        (
            "비채택: D390 소급수리 아님 · 41개 전체 합계 없음 · 예산/충돌체/"
            "Isaac/물리/q5/접촉/파지 0 · g0a_pass=false"
        ),
        body_font,
        (165, 45, 45),
        tag="footer_nonclaim",
    )
    image.save(BOARD)
    bounds_pass = all(
        0 <= bbox["bbox"][0]
        and 0 <= bbox["bbox"][1]
        and bbox["bbox"][2] <= 1920
        and bbox["bbox"][3] <= 1080
        for bbox in text_boxes
    )
    card_region_by_prefix = {
        f"c{row['catalog_index']}_": row["bbox"] for row in cards
    }
    owner_region_checks: list[dict[str, Any]] = []
    for item in text_boxes:
        if item["tag"] in {"title", "subtitle"}:
            owner = [45, 0, 1875, 115]
        elif item["tag"].startswith("footer_"):
            owner = [45, footer_y, 1875, 1045]
        else:
            owner = next(
                (
                    region
                    for prefix, region in card_region_by_prefix.items()
                    if item["tag"].startswith(prefix)
                ),
                [0, 0, 1920, 1080],
            )
        bbox = item["bbox"]
        inside = (
            owner[0] <= bbox[0]
            and owner[1] <= bbox[1]
            and bbox[2] <= owner[2]
            and bbox[3] <= owner[3]
        )
        owner_region_checks.append(
            {
                "tag": item["tag"],
                "owner_region": owner,
                "bbox": bbox,
                "inside": inside,
            }
        )
    owner_region_bounds_pass = all(
        row["inside"] for row in owner_region_checks
    )
    overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(text_boxes):
        lx0, ly0, lx1, ly1 = left["bbox"]
        for right in text_boxes[left_index + 1 :]:
            rx0, ry0, rx1, ry1 = right["bbox"]
            if max(lx0, rx0) < min(lx1, rx1) and max(ly0, ry0) < min(
                ly1, ry1
            ):
                overlaps.append(
                    {"left": left["tag"], "right": right["tag"]}
                )
    layout = {
        "artifact": "D391_BOARD_LAYOUT_VALIDATION_V1",
        "path": _rel(BOARD),
        "width": 1920,
        "height": 1080,
        "card_count": len(cards),
        "cards": cards,
        "text_bboxes": text_boxes,
        "text_bbox_count": len(text_boxes),
        "text_bounds_pass": bounds_pass,
        "text_owner_region_bounds_pass": owner_region_bounds_pass,
        "text_owner_region_checks": owner_region_checks,
        "text_overlap_count": len(overlaps),
        "text_overlaps": overlaps,
        "all_six_cards_present": len(cards) == 6,
        "pass": (
            bounds_pass
            and owner_region_bounds_pass
            and not overlaps
            and len(cards) == 6
        ),
    }
    _write_json_x(LAYOUT, layout)
    return layout


def _build_blueprint(evidence: dict[str, Any]) -> Any:
    import rerun.blueprint as rrb

    views = []
    for row in evidence["disputed_records"]:
        index = row["catalog_index"]
        rank = row["rank_authority"]
        result = (
            f"{rank['authoritative_class']} r={rank['authoritative_rank']}"
            if rank["status"].startswith("STABLE")
            else "AMBIGUOUS"
        )
        views.append(
            rrb.Spatial3DView(
                origin="/",
                contents=f"/d391/calls/c{index}/**",
                name=f"C{index} | n={row['unique_point_count']} | {result}",
                eye_controls=rrb.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                    position=(2.4, -2.4, 2.0),
                    look_target=(0.0, 0.0, 0.0),
                    eye_up=(0.0, 0.0, 1.0),
                ),
                spatial_information=rrb.SpatialInformation(
                    target_frame="tf#/",
                    show_axes=True,
                    show_bounding_box=False,
                ),
            )
        )
    decision = rrb.Vertical(
            rrb.Horizontal(*views[:3], column_shares=[1 / 3] * 3),
            rrb.Horizontal(*views[3:], column_shares=[1 / 3] * 3),
            rrb.TextDocumentView(
                origin="/metadata/run",
                contents="/metadata/run",
                name="D391 authority, input integrity, and nonclaims",
            ),
            row_shares=[0.38, 0.38, 0.24],
    )
    notification_buffer = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d391/notification_buffer/**",
        name="알림 전용 여백 | 판정 내용 없음",
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
            column_shares=[0.78, 0.22],
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


def _artifact_reference_exact(reference: Any, path: Path) -> bool:
    return (
        isinstance(reference, dict)
        and reference.get("path") == _rel(path)
        and reference.get("sha256") == _sha(path)
    )


def _write_rerun(
    evidence: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    import roarm_rl.rerun_contract as rerun_contract
    import roarm_rl.viz_debug as viz_debug

    points_rows: list[dict[str, Any]] = []
    arrow_rows: list[dict[str, Any]] = []
    for row in geometry["records"]:
        index = row["catalog_index"]
        points = np.asarray(row["display_points_f64"], dtype=np.float64)
        origins: list[np.ndarray] = []
        vectors: list[np.ndarray] = []
        for left in range(len(points)):
            for right in range(left + 1, len(points)):
                origins.append(points[left])
                vectors.append(points[right] - points[left])
        points_rows.append(
            {
                "entity_path": f"d391/calls/c{index}/terminal_points",
                "positions_m": points,
                "radii": [0.055] * len(points),
                "colors": [[220, 45, 55, 255]] * len(points),
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
        arrow_rows.append(
            {
                "entity_path": f"d391/calls/c{index}/all_pair_chords",
                "origins_m": np.asarray(origins, dtype=np.float64).reshape(
                    -1, 3
                ),
                "vectors_m": np.asarray(vectors, dtype=np.float64).reshape(
                    -1, 3
                ),
                "radii": 0.012,
                "colors": [[20, 145, 145, 210]] * len(origins),
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
    summary = evidence["disputed_summary"]
    plane = evidence["plane_input_immutability"]
    metadata = {
        "00_legend": "red=terminal points; teal=all unordered pair chords",
        "01_scope": "six static D390 disputed sets only; no timeline",
        "02_numeric": (
            f"stable={summary['stable_count']}; "
            f"ambiguous={summary['ambiguous_count']}"
        ),
        "03_plane_input": (
            f"fixtures={plane['synthetic_pass_count']}/"
            f"{plane['synthetic_fixture_count']}; "
            f"real={plane['real_trace_plane_pass_count']}/"
            f"{plane['real_trace_plane_count']}"
        ),
        "04_authority": (
            "canonical Float64 JSON; centered/scaled Float32 display only"
        ),
        "05_nonclaim": (
            "D390 not repaired; all41 aggregate=null; budget/collider/"
            "Isaac/physics/q5/contact/grasp=0; g0a_pass=false"
        ),
        "case": CASE,
        "attempt": ATTEMPT,
        "canonical_evidence_sha256": _sha(EVIDENCE),
        "canonical_geometry_sha256": _sha(GEOMETRY),
        "d390_repaired": False,
        "d390_retroactive_pass": False,
        "g0a_pass": False,
    }
    original_builder = viz_debug.build_rerun_blueprint
    original_runner = rerun_contract._run
    viewer_calls = 0

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d391_static_six_catalog":
            return _build_blueprint(evidence)
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
                    "stderr": "D391 Viewer maximum one exceeded",
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
            arrows=arrow_rows,
            recording_metadata=metadata,
            recording_id="g0a_d391_rank_basis_and_input_immutability",
            blueprint_path=RBL,
            blueprint_mode="d391_static_six_catalog",
            live_viewer=False,
            app_id="roarm_g0a_d391_rank_basis_and_input_immutability",
        )
        if not saved.get("ok"):
            raise RuntimeError(f"D391 save-only Rerun failed: {saved}")
        expected_entities = ["metadata/run"]
        component_contract: dict[str, list[str]] = {
            "metadata/run": ["TextDocument:text"]
        }
        for index in range(6):
            point_path = f"d391/calls/c{index}/terminal_points"
            chord_path = f"d391/calls/c{index}/all_pair_chords"
            expected_entities.extend([point_path, chord_path])
            component_contract[point_path] = [
                "CoordinateFrame:frame",
                "Points3D:colors",
                "Points3D:positions",
                "Points3D:radii",
            ]
            component_contract[chord_path] = [
                "Arrows3D:colors",
                "Arrows3D:origins",
                "Arrows3D:radii",
                "Arrows3D:vectors",
                "CoordinateFrame:frame",
            ]
        expected_entities = sorted(expected_entities)
        validation = rerun_contract.validate_rerun_artifact(
            RRD,
            expected_entity_paths=expected_entities,
            exact_entity_paths=expected_entities,
            expected_timeline_names=["blueprint", "log_time"],
            exact_timeline_names=["blueprint", "log_time"],
            expected_entity_components=component_contract,
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
    dimensions_pass = (
        screenshot["width"] in {1920, 3840}
        and screenshot["height"] in {1080, 2160}
        and screenshot["width"] * 9 == screenshot["height"] * 16
    )
    validation["d391_execution_contract"] = {
        "static_catalog_no_decision_sequence_timeline": True,
        "system_log_time_only": True,
        "time_panel_registered_hidden": True,
        "notification_buffer_column_share": 0.22,
        "notification_buffer_contains_decision_data": False,
        "headless_viewer_invocations": viewer_calls,
        "viewer_maximum": 1,
        "viewer_retry": 0,
        "process_signals_sent": 0,
        "subprocess_timeout_seconds": None,
        "d390_presentation_parser_reused": False,
        "d390_repaired": False,
        "screenshot_dimension_contract_pass": dimensions_pass,
    }
    validation["base_rerun_contract_pass"] = validation.get("pass") is True
    validation["pass"] = (
        validation["base_rerun_contract_pass"]
        and viewer_calls == 1
        and dimensions_pass
    )
    _write_json_x(RERUN_VALIDATION, validation)
    return {
        "pass": validation["pass"],
        "viewer_calls": viewer_calls,
        "rrd": {
            "path": _rel(RRD),
            "bytes": RRD.stat().st_size,
            "sha256": _sha(RRD),
        },
        "rbl": {
            "path": _rel(RBL),
            "bytes": RBL.stat().st_size,
            "sha256": _sha(RBL),
        },
        "validation": {
            "path": _rel(RERUN_VALIDATION),
            "sha256": _sha(RERUN_VALIDATION),
        },
        "screenshot": screenshot,
    }


def _prepare() -> int:
    if not OUT_DIR.is_dir():
        raise RuntimeError(
            "D391 execution authority must be created before prepare"
        )
    _require_inventory(AUTHORITY_INVENTORY, "before_prepare")
    authority = _read_json(EXECUTION_AUTHORITY)
    external_authority_sha256 = os.environ.get(
        EXECUTION_AUTHORITY_SHA256_ENV
    )
    if (
        external_authority_sha256 is None
        or external_authority_sha256 != _sha(EXECUTION_AUTHORITY)
    ):
        raise RuntimeError(
            "prepare requires the externally supplied exact D391 "
            "execution-authority SHA-256"
        )
    authority_checks = _execution_authority_checks(
        authority, require_current_status_exact=True
    )
    if not all(authority_checks.values()):
        raise RuntimeError(
            f"D391 execution authority failed: {authority_checks}"
        )
    if _sha(D390_GEOMETRY) != EXPECTED_D390_GEOMETRY_SHA256:
        raise RuntimeError("immutable D390 geometry hash changed")
    if _sha(START) != EXPECTED_START_SHA256:
        raise RuntimeError("approved START_HERE authority hash changed")
    start_text = START.read_text(encoding="utf-8")
    if (
        EXPECTED_START_APPROVAL_MARKER not in start_text
        or "`D391 [d390_rank_basis_and_clip_input_immutability_repair]`"
        not in start_text
    ):
        raise RuntimeError("D391 approval marker is absent from START_HERE")
    geometry = _read_json(D390_GEOMETRY)
    schema_checks = _geometry_schema_checks(geometry)
    if not all(schema_checks.values()):
        raise RuntimeError(f"D390 geometry preflight failed: {schema_checks}")
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/master")
    if head != origin or head != EXPECTED_HEAD:
        raise RuntimeError("HEAD/origin do not match registered D391 head")
    baseline_status = authority["git"]["status_before_preregistration"]
    parent_manifest = _directory_manifest(D390_DIR)
    if (
        len(parent_manifest) != 22
        or _canonical_sha(parent_manifest)
        != EXPECTED_D390_DIRECTORY_MANIFEST_SHA256
    ):
        raise RuntimeError("frozen D390 22-file manifest changed")
    environment_checks = {
        "numpy_pin_1_26_0": np.__version__ == "1.26.0",
        "rerun_sdk_pin_0_34_1": importlib.metadata.version("rerun-sdk")
        == "0.34.1",
        "rerun_cli_exists": RERUN_CLI.is_file(),
        "korean_font_exists": FONT_PATH.is_file(),
        "repo_root_import_bootstrap_exact_once": (
            sys.path.count(str(REPO)) == 1
        ),
        "no_isaac_or_physx_import_root": not any(
            name
            in {
                "isaaclab",
                "isaacsim",
                "omni",
                "pxr",
                "warp",
            }
            for name in _direct_import_roots()
        ),
    }
    if not all(environment_checks.values()):
        raise RuntimeError(
            f"D391 environment preflight failed: {environment_checks}"
        )
    _phase("prepare_start")
    prereg = {
        "artifact": "D391_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "execution_authority": {
            "path": _rel(EXECUTION_AUTHORITY),
            "sha256": _sha(EXECUTION_AUTHORITY),
            "external_prepare_environment_variable": (
                EXECUTION_AUTHORITY_SHA256_ENV
            ),
            "externally_supplied_sha256": external_authority_sha256,
            "checks": authority_checks,
        },
        "scientific_question": (
            "Do a translation/order-stable exact-difference rank authority "
            "classify each of the six D390 disputes stably or preserve it "
            "explicitly as ambiguous, while copy-before-normalization "
            "preserves caller-owned plane inputs?"
        ),
        "scientific_input": {
            "path": _rel(D390_GEOMETRY),
            "sha256": EXPECTED_D390_GEOMETRY_SHA256,
            "only_scientific_input": True,
        },
        "disputed_manifest": DISPUTED_MANIFEST,
        "frozen_parent_manifest": parent_manifest,
        "frozen_parent_manifest_sha256": (
            EXPECTED_D390_DIRECTORY_MANIFEST_SHA256
        ),
        "method": {
            "rank": {
                "canonical_unique_order": (
                    "numeric lexicographic Float64 tuple order"
                ),
                "difference_arithmetic": (
                    "Fraction.from_float exact dyadic differences, then "
                    "one Float64 conversion for SVD"
                ),
                "bases": "every anchor plus all unordered pairs",
                "hard_rank_cap": (
                    "min(3,max(0,unique_point_count-1))"
                ),
                "threshold": (
                    "sigma_max * max(unique_point_count,3) * eps64"
                ),
                "threshold_alphas": list(THRESHOLD_ALPHAS),
                "final_policy": (
                    "stable only if every bounded basis rank is unchanged "
                    "for alpha 0.5/1/2 and bounded numeric rank <= exact "
                    "dyadic rank; impossible raw ranks remain recorded as "
                    "STABLE_HARD_CAP_CORRECTED when the mathematical cap "
                    "alone resolves them; otherwise preserve explicit "
                    "ambiguity"
                ),
                "order_control": (
                    "exhaustive permutations: n! for each of six sets"
                ),
                "ambiguity_controls": {
                    "point_fixture_height": "2^-50",
                    "point_fixture_expected_status": (
                        "NUMERICALLY_AMBIGUOUS_BASIS or "
                        "NUMERICALLY_AMBIGUOUS_THRESHOLD"
                    ),
                    "direct_diagonal_rows": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, math.ldexp(1.0, -50)],
                    ],
                    "direct_expected_status": (
                        "NUMERICALLY_AMBIGUOUS_THRESHOLD"
                    ),
                },
                "translation_control": (
                    "exact dyadic translation before exact difference; "
                    "every anchor and unordered-pair Fraction row, "
                    "Float64 matrix hash, exact rank, and alpha rank map "
                    "must be identical"
                ),
                "scale_control_exponents": list(SCALE_EXPONENTS),
                "scale_control_gates": (
                    "inverse Float64 point hash exact; every Fraction "
                    "basis row scales by 2^k; every singular value and "
                    "tau scale within relative 128*eps64; rank signature "
                    "exact"
                ),
                "random_jitter_or_qj": False,
            },
            "plane_input": {
                "repair": (
                    "np.array(equation[:3],float64,copy=True) before "
                    "in-place normalization"
                ),
                "synthetic_fixtures": PLANE_FIXTURE_VALUES,
                "synthetic_fixture_sha256": {
                    name: _array_sha(
                        np.asarray(values, dtype=np.float64).reshape(1, 4)
                    )
                    for name, values in PLANE_FIXTURE_VALUES.items()
                },
                "binary_exact_contract": (
                    "all fixture values remain integer or +/-0.5 both "
                    "before and after the helper call"
                ),
                "nonunit_normal_nonzero_offset_regression": {
                    "input": [2.0, 0.0, 0.0, -1.0],
                    "expected_normalized_copy": [1.0, 0.0, 0.0, -0.5],
                },
                "real_trace_check": (
                    "351 copy-normalizations and pre/post hashes only; "
                    "no clipping replay"
                ),
            },
        },
        "observability": {
            "d391_static_six_catalog_only": True,
            "d390_parser_cursor_layout_repair": False,
            "d390_repaired": False,
            "rrd_rbl_and_headless_screenshot_required": True,
            "viewer_maximum": 1,
            "viewer_retry": 0,
            "board_dimensions": [1920, 1080],
            "rerun_sdk_cli_pin": "0.34.1",
            "manual_visual_inspection_required_before_finalize": True,
        },
        "execution": {
            "actual_worker_maximum": 1,
            "worker_retry": 0,
            "cooperative_deadline_seconds": COOPERATIVE_DEADLINE_SECONDS,
            "hard_watchdog_seconds": None,
            "supervisor_wait_bounded": False,
            "supervisor_signal_authority": False,
            "process_signals_authorized": 0,
        },
        "frozen_nonclaims": {
            "d390_reexecution_or_repair": 0,
            "d389_or_d390_clipping_replay": 0,
            "epsilon_5nm_tolerance_gate_partition_budget_geometry_change": 0,
            "selected_or_adopted_budget": None,
            "collider_asset_usd_isaac_physx": 0,
            "cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_settings": 0,
            "g0a_pass": False,
        },
        "authorization": {
            "path": _rel(START),
            "sha256": _sha(START),
            "user_approval": (
                "D391 [d390_rank_basis_and_clip_input_immutability_repair]"
            ),
        },
        "script": {
            "path": _rel(SCRIPT),
            "sha256": _sha(SCRIPT),
            "direct_import_roots": _direct_import_roots(),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "pillow": importlib.metadata.version("Pillow"),
            "rerun_sdk": importlib.metadata.version("rerun-sdk"),
            "rerun_cli": str(RERUN_CLI),
        },
        "git": {
            "head": head,
            "origin_master": origin,
            "status_before_preregistration": baseline_status,
            "status_line_count": len(baseline_status),
            "status_manifest_sha256": _status_manifest_sha256(
                baseline_status
            ),
        },
        "preflight_schema_checks": schema_checks,
        "environment_preflight_checks": environment_checks,
    }
    _write_json_x(PREREG, prereg)
    _phase(
        "prepare_end",
        prepare_pass=True,
        preregistration_sha256=_sha(PREREG),
    )
    _require_inventory(PREPARE_INVENTORY, "after_prepare")
    print(
        json.dumps(
            {
                "prepare_pass": True,
                "preregistration": _rel(PREREG),
                "sha256": _sha(PREREG),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _authorization_checks() -> dict[str, bool]:
    prereg = _read_json(PREREG)
    invocation = _read_json(INVOCATION)
    authorization = _read_json(AUTHORIZATION)
    expected_command = [
        sys.executable,
        "-B",
        str(SCRIPT),
        "--stage",
        "worker",
    ]
    return {
        "frozen_authority_exact": all(
            _frozen_checks(prereg, allow_output=True).values()
        ),
        "invocation_artifact_exact": invocation.get("artifact")
        == "D391_OFFLINE_WORKER_INVOCATION_V1",
        "authorization_artifact_exact": authorization.get("artifact")
        == "D391_WORKER_AUTHORIZATION_V1",
        "command_exact": invocation.get("command") == expected_command,
        "cwd_exact": invocation.get("cwd") == str(REPO),
        "worker_index_one": invocation.get("worker_invocation_index") == 1
        and authorization.get("worker_invocation_index") == 1,
        "retry_index_zero": invocation.get("retry_index") == 0
        and authorization.get("retry_index") == 0,
        "retry_zero": invocation.get("retries") == 0
        and authorization.get("retries") == 0,
        "actual_worker_maximum_one": invocation.get(
            "actual_worker_maximum"
        )
        == 1
        and authorization.get("actual_worker_maximum") == 1,
        "hard_watchdog_null": invocation.get("hard_watchdog_seconds")
        is None
        and authorization.get("hard_watchdog_seconds") is None,
        "signal_authority_false": invocation.get(
            "supervisor_signal_authority"
        )
        is False
        and authorization.get("supervisor_signal_authority") is False,
        "authorized_signals_zero": invocation.get(
            "process_signals_authorized"
        )
        == 0
        and authorization.get("process_signals_authorized") == 0,
        "script_hash_exact": invocation.get("script_sha256") == _sha(SCRIPT)
        and authorization.get("script_sha256") == _sha(SCRIPT),
        "input_hash_exact": invocation.get("scientific_input_sha256")
        == EXPECTED_D390_GEOMETRY_SHA256
        and authorization.get("scientific_input_sha256")
        == EXPECTED_D390_GEOMETRY_SHA256,
        "preregistration_hash_exact": invocation.get(
            "preregistration_sha256"
        )
        == _sha(PREREG)
        and authorization.get("preregistration_sha256") == _sha(PREREG),
        "execution_authority_hash_exact": invocation.get(
            "execution_authority_sha256"
        )
        == _sha(EXECUTION_AUTHORITY)
        and authorization.get("execution_authority_sha256")
        == _sha(EXECUTION_AUTHORITY),
        "invocation_hash_bound": authorization.get("invocation_sha256")
        == _sha(INVOCATION),
        "parent_supervisor_exact": authorization.get("supervisor_pid")
        == os.getppid(),
        "stdout_stderr_precreated": STDOUT.is_file() and STDERR.is_file(),
        "sentinel_absent": not SENTINEL.exists(),
    }


def _worker_inner() -> int:
    global _deadline_monotonic
    started = time.monotonic()
    _deadline_monotonic = started + COOPERATIVE_DEADLINE_SECONDS
    _require_inventory(PRE_WORKER_INVENTORY, "worker_before_sentinel")
    authorization_checks = _authorization_checks()
    if not all(authorization_checks.values()):
        raise RuntimeError(
            f"D391 worker authorization failed: {authorization_checks}"
        )
    _write_json_x(
        SENTINEL,
        {
            "artifact": "D391_WORKER_START_SENTINEL_V1",
            "worker_pid": os.getpid(),
            "parent_supervisor_pid": os.getppid(),
            "worker_invocation_index": 1,
            "retry": 0,
            "preregistration_sha256": _sha(PREREG),
            "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
            "script_sha256": _sha(SCRIPT),
            "scientific_input_sha256": _sha(D390_GEOMETRY),
            "wall_time_ns": time.time_ns(),
        },
    )
    _phase("worker_start", worker_pid=os.getpid())
    _require_inventory(WORKER_START_INVENTORY, "worker_start")
    evidence, geometry, records = _compute()
    _write_json_x(EVIDENCE, evidence)
    _write_json_x(GEOMETRY, geometry)
    _write_csv(records)
    _phase(
        "canonical_numeric_evidence_committed",
        evidence_sha256=_sha(EVIDENCE),
        geometry_sha256=_sha(GEOMETRY),
        csv_sha256=_sha(CSV_PATH),
    )
    _deadline("before_board")
    layout = _render_board(evidence, geometry)
    _deadline("after_board")
    rerun = _write_rerun(evidence, geometry)
    _deadline("after_rerun")
    manual_template = {
        "artifact": "D391_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board_path": _rel(BOARD),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "artifact_hashes_to_bind_after_actual_viewing": {
            "board_sha256": _sha(BOARD),
            "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
        },
        "checks_to_record_after_actual_viewing": [
            "board_six_cards_visible",
            "board_text_readable_no_overlap_or_clipping",
            "board_red_points_and_teal_pair_chords_visible",
            "rerun_six_spatial_views_visible",
            "rerun_legend_and_numeric_summary_readable",
            "rerun_no_decision_subject_obscured",
            "rerun_time_panel_hidden_no_decision_cursor",
            "stable_or_ambiguous_result_matches_canonical_json",
            "d390_frozen_and_all_nonclaims_visible",
        ],
        "minimum_nonempty_observations": 3,
        "manual_inspection_complete": False,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)
    _require_inventory(PRE_CLAIM_INVENTORY, "before_worker_claim")
    worker_checks = {
        "numeric_pass": evidence["numeric_pass"],
        "board_layout_pass": layout["pass"],
        "rerun_automatic_contract_pass": rerun["pass"],
        "viewer_exactly_one": rerun["viewer_calls"] == 1,
        "worker_within_cooperative_deadline": (
            time.monotonic() <= _deadline_monotonic
        ),
        "authorization_checks_pass": all(authorization_checks.values()),
        "d390_remains_unmodified": _directory_manifest(D390_DIR)
        == _read_json(PREREG)["frozen_parent_manifest"],
    }
    claim = {
        "artifact": "D391_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "actual_worker_invocations": 1,
        "worker_invocation_index": 1,
        "retries": 0,
        "process_signals_sent": 0,
        "cooperative_deadline_seconds": COOPERATIVE_DEADLINE_SECONDS,
        "cooperative_deadline_exceeded": (
            time.monotonic() > _deadline_monotonic
        ),
        "hard_watchdog_seconds": None,
        "worker_elapsed_seconds": time.monotonic() - started,
        "preregistration_sha256": _sha(PREREG),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "worker_start_sentinel_sha256": _sha(SENTINEL),
        "authorization_checks": authorization_checks,
        "checks": worker_checks,
        "artifacts": {
            "evidence": {"path": _rel(EVIDENCE), "sha256": _sha(EVIDENCE)},
            "geometry": {"path": _rel(GEOMETRY), "sha256": _sha(GEOMETRY)},
            "csv": {"path": _rel(CSV_PATH), "sha256": _sha(CSV_PATH)},
            "board": {"path": _rel(BOARD), "sha256": _sha(BOARD)},
            "layout": {"path": _rel(LAYOUT), "sha256": _sha(LAYOUT)},
            "rerun": rerun,
            "manual_template": {
                "path": _rel(MANUAL_TEMPLATE),
                "sha256": _sha(MANUAL_TEMPLATE),
            },
        },
        "pass": all(worker_checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_end", worker_claim_pass=claim["pass"])
    if not claim["pass"]:
        raise RuntimeError(f"D391 worker claim failed: {worker_checks}")
    _require_inventory(POST_WORKER_INVENTORY, "after_worker_claim")
    print(
        json.dumps(
            {
                "worker_pass": True,
                "numeric_verdict": evidence["numeric_verdict"],
                "worker_claim": _rel(WORKER_CLAIM),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _worker() -> int:
    try:
        return _worker_inner()
    except Exception as exc:
        evidence_value, evidence_read_error = (
            _safe_read_json(EVIDENCE)
            if EVIDENCE.is_file()
            else (None, None)
        )
        inventory_before_attestation = sorted(_out_names())
        expected_inventory_after_attestation = sorted(
            set(inventory_before_attestation) | {FAILURE.name}
        )
        expected_final_failure_inventory = sorted(
            set(expected_inventory_after_attestation) | {SUPERVISOR.name}
        )
        failure = {
            "artifact": "D391_FAILURE_ATTESTATION_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "stage": "worker",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "actual_worker_invocations": 1,
            "retries": 0,
            "process_signals_sent": 0,
            "script_sha256": _sha(SCRIPT),
            "scientific_input_sha256": _sha(D390_GEOMETRY),
            "preregistration_sha256": (
                _sha(PREREG) if PREREG.is_file() else None
            ),
            "execution_authority_sha256": (
                _sha(EXECUTION_AUTHORITY)
                if EXECUTION_AUTHORITY.is_file()
                else None
            ),
            "invocation_sha256": (
                _sha(INVOCATION) if INVOCATION.is_file() else None
            ),
            "authorization_sha256": (
                _sha(AUTHORIZATION) if AUTHORIZATION.is_file() else None
            ),
            "sentinel_sha256": (
                _sha(SENTINEL) if SENTINEL.is_file() else None
            ),
            "cooperative_deadline_exceeded": (
                _deadline_monotonic is not None
                and time.monotonic() > _deadline_monotonic
            ),
            "hard_watchdog_seconds": None,
            "evidence_exists": EVIDENCE.is_file(),
            "evidence_sha256": _sha(EVIDENCE) if EVIDENCE.is_file() else None,
            "evidence_read_error": evidence_read_error,
            "worker_claim_exists": WORKER_CLAIM.is_file(),
            "output_inventory_before_attestation": (
                inventory_before_attestation
            ),
            "expected_output_inventory_after_attestation": (
                expected_inventory_after_attestation
            ),
            "expected_final_failure_inventory_after_supervisor": (
                expected_final_failure_inventory
            ),
            "verdict": (
                "D391_RANK_BASIS_OR_CLIP_INPUT_IMMUTABILITY_FAIL_STOP"
                if isinstance(evidence_value, dict)
                and not evidence_value.get("numeric_pass", False)
                else "D391_OFFLINE_WORKER_OR_OBSERVABILITY_INTEGRITY_FAIL_STOP"
            ),
        }
        if not FAILURE.exists():
            _write_json_x(FAILURE, failure)
        failure_inventory_exact = (
            sorted(_out_names()) == expected_inventory_after_attestation
        )
        _phase(
            "worker_fail_stop",
            error_type=type(exc).__name__,
            failure_attestation_sha256=_sha(FAILURE),
            failure_inventory_exact=failure_inventory_exact,
        )
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1


def _run_inner() -> int:
    global _supervisor_process_started, _supervisor_worker_pid
    if not PREREG.is_file():
        raise RuntimeError("D391 prepare must precede run")
    _require_inventory(PREPARE_INVENTORY, "supervisor_before_invocation")
    prereg = _read_json(PREREG)
    frozen = _frozen_checks(prereg, allow_output=True)
    if not all(frozen.values()):
        raise RuntimeError(f"D391 frozen authority changed: {frozen}")
    command = [sys.executable, "-B", str(SCRIPT), "--stage", "worker"]
    invocation = {
        "artifact": "D391_OFFLINE_WORKER_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "cwd": str(REPO),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "actual_worker_maximum": 1,
        "retries": 0,
        "script_sha256": _sha(SCRIPT),
        "scientific_input_sha256": _sha(D390_GEOMETRY),
        "preregistration_sha256": _sha(PREREG),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "cooperative_deadline_seconds": COOPERATIVE_DEADLINE_SECONDS,
        "hard_watchdog_seconds": None,
        "supervisor_wait_bounded": False,
        "supervisor_signal_authority": False,
        "process_signals_authorized": 0,
    }
    _write_json_x(INVOCATION, invocation)
    STDOUT.open("x").close()
    STDERR.open("x").close()
    _write_json_x(
        AUTHORIZATION,
        {
            "artifact": "D391_WORKER_AUTHORIZATION_V1",
            "supervisor_pid": os.getpid(),
            "worker_invocation_index": 1,
            "retry_index": 0,
            "actual_worker_maximum": 1,
            "retries": 0,
            "script_sha256": _sha(SCRIPT),
            "scientific_input_sha256": _sha(D390_GEOMETRY),
            "preregistration_sha256": _sha(PREREG),
            "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
            "invocation_sha256": _sha(INVOCATION),
            "hard_watchdog_seconds": None,
            "supervisor_signal_authority": False,
            "process_signals_authorized": 0,
        },
    )
    _phase("supervisor_before_worker", supervisor_pid=os.getpid())
    _require_inventory(PRE_WORKER_INVENTORY, "supervisor_before_popen")
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    supervisor_error: str | None = None
    returncode: int | None = None
    try:
        with STDOUT.open("w", encoding="utf-8") as stdout, STDERR.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            _supervisor_process_started = True
            _supervisor_worker_pid = process.pid
            returncode = process.wait()
    except Exception as exc:
        supervisor_error = f"{type(exc).__name__}: {exc}"
    claim, claim_read_error = (
        _safe_read_json(WORKER_CLAIM)
        if WORKER_CLAIM.is_file()
        else (None, None)
    )
    worker_pass = bool(claim and claim.get("pass") is True)
    supervisor = {
        "artifact": "D391_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "supervisor_pid": os.getpid(),
        "actual_worker_invocations": int(process is not None),
        "worker_pid": process.pid if process is not None else None,
        "worker_returncode": returncode,
        "worker_process_exited": (
            process is not None and process.poll() is not None
        ),
        "worker_claim_exists": WORKER_CLAIM.is_file(),
        "worker_claim_pass": worker_pass,
        "worker_claim_read_error": claim_read_error,
        "failure_attestation_exists": FAILURE.is_file(),
        "retries": 0,
        "process_signals_sent": 0,
        "supervisor_signal_authority": False,
        "hard_watchdog_seconds": None,
        "supervisor_wait_bounded": False,
        "supervisor_error": supervisor_error,
        "supervisor_elapsed_seconds": time.monotonic() - started,
        "preregistration_sha256": _sha(PREREG),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "invocation_sha256": _sha(INVOCATION),
        "authorization_sha256": _sha(AUTHORIZATION),
        "sentinel_sha256": _sha(SENTINEL) if SENTINEL.is_file() else None,
        "worker_claim_sha256": (
            _sha(WORKER_CLAIM) if WORKER_CLAIM.is_file() else None
        ),
        "pass": (
            process is not None
            and returncode == 0
            and process.poll() is not None
            and worker_pass
            and not FAILURE.exists()
            and supervisor_error is None
        ),
    }
    _write_json_x(SUPERVISOR, supervisor)
    _phase(
        "supervisor_after_worker",
        worker_returncode=returncode,
        supervisor_pass=supervisor["pass"],
    )
    if not supervisor["pass"]:
        raise RuntimeError(f"D391 worker failed: {supervisor}")
    _require_inventory(POST_RUN_INVENTORY, "after_supervisor")
    print(json.dumps(supervisor, ensure_ascii=False))
    return 0


def _run() -> int:
    try:
        return _run_inner()
    except Exception as exc:
        if OUT_DIR.is_dir():
            if not SUPERVISOR.exists():
                _write_json_x(
                    SUPERVISOR,
                    {
                        "artifact": "D391_OFFLINE_WORKER_SUPERVISOR_V1",
                        "case": CASE,
                        "attempt": ATTEMPT,
                        "stage": "supervisor_fail_stop",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "actual_worker_invocations": int(
                            _supervisor_process_started
                        ),
                        "worker_pid": _supervisor_worker_pid,
                        "worker_process_exited": None,
                        "worker_claim_exists": WORKER_CLAIM.is_file(),
                        "worker_claim_pass": False,
                        "retries": 0,
                        "process_signals_sent": 0,
                        "supervisor_signal_authority": False,
                        "hard_watchdog_seconds": None,
                        "script_sha256": _sha(SCRIPT),
                        "scientific_input_sha256": _sha(D390_GEOMETRY),
                        "preregistration_sha256": (
                            _sha(PREREG) if PREREG.is_file() else None
                        ),
                        "execution_authority_sha256": (
                            _sha(EXECUTION_AUTHORITY)
                            if EXECUTION_AUTHORITY.is_file()
                            else None
                        ),
                        "invocation_sha256": (
                            _sha(INVOCATION)
                            if INVOCATION.is_file()
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
                        "pass": False,
                    },
                )
            if not FAILURE.exists():
                inventory_before = sorted(_out_names())
                expected_after = sorted(
                    set(inventory_before) | {FAILURE.name}
                )
                _write_json_x(
                    FAILURE,
                    {
                        "artifact": "D391_FAILURE_ATTESTATION_V1",
                        "case": CASE,
                        "attempt": ATTEMPT,
                        "stage": "supervisor",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "actual_worker_invocations": int(
                            _supervisor_process_started
                        ),
                        "worker_pid": _supervisor_worker_pid,
                        "retries": 0,
                        "process_signals_sent": 0,
                        "hard_watchdog_seconds": None,
                        "script_sha256": _sha(SCRIPT),
                        "scientific_input_sha256": _sha(D390_GEOMETRY),
                        "preregistration_sha256": (
                            _sha(PREREG) if PREREG.is_file() else None
                        ),
                        "execution_authority_sha256": (
                            _sha(EXECUTION_AUTHORITY)
                            if EXECUTION_AUTHORITY.is_file()
                            else None
                        ),
                        "invocation_sha256": (
                            _sha(INVOCATION)
                            if INVOCATION.is_file()
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
                        "output_inventory_before_attestation": (
                            inventory_before
                        ),
                        "expected_output_inventory_after_attestation": (
                            expected_after
                        ),
                        "supervisor_sha256": _sha(SUPERVISOR),
                        "verdict": (
                            "D391_OFFLINE_WORKER_OR_PROVENANCE_"
                            "INTEGRITY_FAIL_STOP"
                        ),
                    },
                )
            failure_value, failure_read_error = _safe_read_json(FAILURE)
            expected_failure_inventory = (
                failure_value.get(
                    "expected_final_failure_inventory_after_supervisor"
                )
                if isinstance(failure_value, dict)
                else None
            )
            if (
                expected_failure_inventory is None
                and isinstance(failure_value, dict)
            ):
                immediate_inventory = failure_value.get(
                    "expected_output_inventory_after_attestation"
                )
                if isinstance(immediate_inventory, list):
                    expected_failure_inventory = sorted(
                        set(immediate_inventory) | {SUPERVISOR.name}
                    )
            _phase(
                "supervisor_fail_stop",
                error_type=type(exc).__name__,
                supervisor_sha256=_sha(SUPERVISOR),
                failure_attestation_sha256=_sha(FAILURE),
                failure_read_error=failure_read_error,
                failure_inventory_exact=(
                    sorted(_out_names())
                    == expected_failure_inventory
                ),
            )
        raise


def _phase_contract() -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in PHASES.read_text(encoding="utf-8").splitlines()
        if line
    ]
    names = [row.get("phase") for row in rows]
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
    return {
        "observed": names,
        "expected": expected,
        "exact": names == expected,
        "forward_wall_time": all(
            rows[index]["wall_time_ns"] <= rows[index + 1]["wall_time_ns"]
            for index in range(len(rows) - 1)
        ),
    }


def _finalize() -> int:
    if FAILURE.exists():
        raise RuntimeError("D391 failure attestation forbids finalize")
    _require_inventory(PRE_FINALIZE_INVENTORY, "before_finalize")
    prereg = _read_json(PREREG)
    frozen = _frozen_checks(prereg, allow_output=True)
    if not all(frozen.values()):
        raise RuntimeError(f"D391 finalize frozen authority changed: {frozen}")
    supervisor = _read_json(SUPERVISOR)
    worker = _read_json(WORKER_CLAIM)
    invocation = _read_json(INVOCATION)
    authorization = _read_json(AUTHORIZATION)
    sentinel = _read_json(SENTINEL)
    manual = _read_json(MANUAL)
    evidence = _read_json(EVIDENCE)
    validation = _read_json(RERUN_VALIDATION)
    layout = _read_json(LAYOUT)
    manual_checks = manual.get("checks", {})
    manual_template = _read_json(MANUAL_TEMPLATE)
    expected_manual_keys = set(
        manual_template["checks_to_record_after_actual_viewing"]
    )
    manual_observations = manual.get("observations", [])
    expected_manual_hashes = {
        "board_sha256": _sha(BOARD),
        "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
    }
    worker_artifacts = worker.get("artifacts", {})
    worker_rerun = worker_artifacts.get("rerun", {})
    worker_artifact_linkage = {
        "evidence": _artifact_reference_exact(
            worker_artifacts.get("evidence"), EVIDENCE
        ),
        "geometry": _artifact_reference_exact(
            worker_artifacts.get("geometry"), GEOMETRY
        ),
        "csv": _artifact_reference_exact(
            worker_artifacts.get("csv"), CSV_PATH
        ),
        "board": _artifact_reference_exact(
            worker_artifacts.get("board"), BOARD
        ),
        "layout": _artifact_reference_exact(
            worker_artifacts.get("layout"), LAYOUT
        ),
        "rrd": _artifact_reference_exact(worker_rerun.get("rrd"), RRD),
        "rbl": _artifact_reference_exact(worker_rerun.get("rbl"), RBL),
        "rerun_validation": _artifact_reference_exact(
            worker_rerun.get("validation"), RERUN_VALIDATION
        ),
        "rerun_screenshot": _artifact_reference_exact(
            worker_rerun.get("screenshot"), RERUN_SCREENSHOT
        ),
        "manual_template": _artifact_reference_exact(
            worker_artifacts.get("manual_template"), MANUAL_TEMPLATE
        ),
    }
    provenance_linkage = {
        "preregistration_execution_authority": prereg.get(
            "execution_authority", {}
        ).get("sha256")
        == _sha(EXECUTION_AUTHORITY),
        "invocation_preregistration": invocation.get(
            "preregistration_sha256"
        )
        == _sha(PREREG),
        "invocation_execution_authority": invocation.get(
            "execution_authority_sha256"
        )
        == _sha(EXECUTION_AUTHORITY),
        "authorization_preregistration": authorization.get(
            "preregistration_sha256"
        )
        == _sha(PREREG),
        "authorization_execution_authority": authorization.get(
            "execution_authority_sha256"
        )
        == _sha(EXECUTION_AUTHORITY),
        "authorization_invocation": authorization.get(
            "invocation_sha256"
        )
        == _sha(INVOCATION),
        "sentinel_preregistration": sentinel.get(
            "preregistration_sha256"
        )
        == _sha(PREREG),
        "sentinel_execution_authority": sentinel.get(
            "execution_authority_sha256"
        )
        == _sha(EXECUTION_AUTHORITY),
        "sentinel_script": sentinel.get("script_sha256") == _sha(SCRIPT),
        "sentinel_input": sentinel.get("scientific_input_sha256")
        == _sha(D390_GEOMETRY),
        "worker_preregistration": worker.get("preregistration_sha256")
        == _sha(PREREG),
        "worker_execution_authority": worker.get(
            "execution_authority_sha256"
        )
        == _sha(EXECUTION_AUTHORITY),
        "worker_sentinel": worker.get("worker_start_sentinel_sha256")
        == _sha(SENTINEL),
        "supervisor_preregistration": supervisor.get(
            "preregistration_sha256"
        )
        == _sha(PREREG),
        "supervisor_execution_authority": supervisor.get(
            "execution_authority_sha256"
        )
        == _sha(EXECUTION_AUTHORITY),
        "supervisor_invocation": supervisor.get("invocation_sha256")
        == _sha(INVOCATION),
        "supervisor_authorization": supervisor.get(
            "authorization_sha256"
        )
        == _sha(AUTHORIZATION),
        "supervisor_sentinel": supervisor.get("sentinel_sha256")
        == _sha(SENTINEL),
        "supervisor_worker_claim": supervisor.get(
            "worker_claim_sha256"
        )
        == _sha(WORKER_CLAIM),
        "supervisor_pid_chain": (
            supervisor.get("supervisor_pid")
            == authorization.get("supervisor_pid")
            == sentinel.get("parent_supervisor_pid")
        ),
        "worker_index_chain": (
            invocation.get("worker_invocation_index")
            == authorization.get("worker_invocation_index")
            == sentinel.get("worker_invocation_index")
            == worker.get("worker_invocation_index")
            == 1
        ),
        "retry_index_and_count_chain": (
            invocation.get("retry_index") == 0
            and authorization.get("retry_index") == 0
            and invocation.get("retries") == 0
            and authorization.get("retries") == 0
            and worker.get("retries") == 0
            and supervisor.get("retries") == 0
        ),
    }
    prechecks = {
        "supervisor_pass": supervisor.get("pass") is True,
        "worker_claim_pass": worker.get("pass") is True,
        "numeric_pass": evidence.get("numeric_pass") is True,
        "layout_pass": layout.get("pass") is True,
        "rerun_validation_pass": validation.get("pass") is True,
        "manual_artifact_exact": manual.get("artifact")
        == "D391_MANUAL_VISUAL_INSPECTION_V1",
        "manual_paths_exact": manual.get("board_path") == _rel(BOARD)
        and manual.get("rerun_screenshot_path")
        == _rel(RERUN_SCREENSHOT),
        "manual_check_keys_exact": set(manual_checks)
        == expected_manual_keys,
        "manual_all_checks_pass": (
            set(manual_checks) == expected_manual_keys
            and all(value is True for value in manual_checks.values())
        ),
        "manual_artifact_hashes_exact_current_pngs": (
            manual.get("artifact_hashes") == expected_manual_hashes
            == manual_template.get(
                "artifact_hashes_to_bind_after_actual_viewing"
            )
        ),
        "manual_observations_minimum_three_nonempty": (
            isinstance(manual_observations, list)
            and len(manual_observations) >= 3
            and all(
                isinstance(value, str) and bool(value.strip())
                for value in manual_observations
            )
        ),
        "manual_inspection_complete": manual.get(
            "manual_inspection_complete"
        )
        is True,
        "worker_once_retry_zero_no_signal": (
            supervisor.get("actual_worker_invocations") == 1
            and supervisor.get("retries") == 0
            and supervisor.get("process_signals_sent") == 0
        ),
        "invocation_authorization_sentinel_claim_supervisor_hash_chain": all(
            provenance_linkage.values()
        ),
        "worker_artifact_paths_and_hashes_exact": all(
            worker_artifact_linkage.values()
        ),
        "viewer_exactly_one_no_retry": validation.get(
            "d391_execution_contract", {}
        ).get("headless_viewer_invocations")
        == 1
        and validation.get("d391_execution_contract", {}).get(
            "viewer_retry"
        )
        == 0,
        "d390_parent_manifest_exact": _directory_manifest(D390_DIR)
        == prereg["frozen_parent_manifest"],
        "all_frozen_checks_pass": all(frozen.values()),
    }
    if not all(prechecks.values()):
        raise RuntimeError(f"D391 finalize prechecks failed: {prechecks}")
    _phase("finalize_start")
    phase_before_end = [
        json.loads(line).get("phase")
        for line in PHASES.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_before_end = [
        "prepare_start",
        "prepare_end",
        "supervisor_before_worker",
        "worker_start",
        "canonical_numeric_evidence_committed",
        "worker_end",
        "supervisor_after_worker",
        "finalize_start",
    ]
    if phase_before_end != expected_before_end:
        raise RuntimeError(
            f"D391 phase order before finalize_end failed: {phase_before_end}"
        )
    _phase("finalize_end", pass_value=True)
    phase_contract = _phase_contract()
    checks = {
        **prechecks,
        "phase_contract_exact": phase_contract["exact"]
        and phase_contract["forward_wall_time"],
        "success_inventory_exact_before_completion": (
            _out_names() == PRE_FINALIZE_INVENTORY
        ),
    }
    completion = {
        "artifact": "D391_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "numeric_verdict": evidence["numeric_verdict"],
        "operational_verdict": (
            "D391_RANK_BASIS_AND_CLIP_INPUT_IMMUTABILITY_PASS_"
            "NO_D390_REPAIR"
        ),
        "disputed_summary": evidence["disputed_summary"],
        "plane_input_summary": {
            "synthetic_pass_count": evidence["plane_input_immutability"][
                "synthetic_pass_count"
            ],
            "synthetic_fixture_count": evidence[
                "plane_input_immutability"
            ]["synthetic_fixture_count"],
            "real_trace_plane_pass_count": evidence[
                "plane_input_immutability"
            ]["real_trace_plane_pass_count"],
            "real_trace_plane_count": evidence[
                "plane_input_immutability"
            ]["real_trace_plane_count"],
        },
        "worker_retry_viewer_signal": [1, 0, 1, 0],
        "checks": checks,
        "provenance_linkage": provenance_linkage,
        "worker_artifact_linkage": worker_artifact_linkage,
        "phase_contract": phase_contract,
        "manual_visual_inspection": {
            "path": _rel(MANUAL),
            "sha256": _sha(MANUAL),
        },
        "artifacts": {
            label: {
                "path": _rel(path),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
            for label, path in (
                ("execution_authority", EXECUTION_AUTHORITY),
                ("preregistration", PREREG),
                ("evidence", EVIDENCE),
                ("geometry", GEOMETRY),
                ("csv", CSV_PATH),
                ("board", BOARD),
                ("layout", LAYOUT),
                ("rrd", RRD),
                ("rbl", RBL),
                ("rerun_validation", RERUN_VALIDATION),
                ("rerun_screenshot", RERUN_SCREENSHOT),
                ("worker_claim", WORKER_CLAIM),
                ("supervisor", SUPERVISOR),
                ("manual_inspection", MANUAL),
            )
        },
        "nonclaims": evidence["nonclaims"],
        "pass": all(checks.values()),
    }
    if not completion["pass"]:
        raise RuntimeError(f"D391 completion checks failed: {checks}")
    _write_json_x(COMPLETION, completion)
    _require_inventory(POST_FINALIZE_INVENTORY, "after_finalize")
    print(json.dumps(completion, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "run", "worker", "finalize"),
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
