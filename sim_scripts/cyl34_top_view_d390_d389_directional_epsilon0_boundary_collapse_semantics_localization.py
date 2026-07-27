#!/usr/bin/env python3
"""D390 offline localization of D389 directional epsilon-zero collapse states.

This program reads exactly three immutable D389 artifacts: the numeric
evidence, reconstructed seam geometry, and seam CSV.  It does not import or
execute D389.  It independently reconstructs only the 41 directional calls
that D389 recorded as failed and preserves the candidate point set immediately
before the original 3-D-only hull contract would reject it.

The single registered variable is the diagnostic affine-rank classification
contract.  Its SVD threshold formalizes NumPy 1.26's default matrix-rank
semantics and is never reused as an overlap, tolerance, gate, or physics
threshold.  The program does not continue clipping after a collapse, repair
D389, change geometry, select a collider budget, or launch Isaac/PhysX.
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

CASE = "g0a_d390"
ATTEMPT = (
    "attempt1_d389_directional_epsilon0_boundary_collapse_"
    "semantics_localization"
)
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT
SCRIPT = Path(__file__).resolve()
START = REPO / "START_HERE.md"

D389_DIR = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d389/"
    "attempt2_prereg_status_whitespace_repair"
)
D389_EVIDENCE = D389_DIR / "d389_numeric_and_tie_audit_evidence.json"
D389_GEOMETRY = D389_DIR / "d389_reconstructed_seam_witness_geometry.json"
D389_CSV = D389_DIR / "d389_seam_numeric_provenance.csv"

EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
EXPECTED_START_SHA256 = (
    "90cf1f95619f4be21a04e1b11f1a38b4b7ca9e2f9724c419eb6c3b51e644d1a3"
)
EXPECTED_INPUT_SHA256 = {
    "d389_evidence": (
        "9423e870c0a218606781943abd2f5c48cb1e5d53cbbf9fb1212294b4ef5bb5dd"
    ),
    "d389_geometry": (
        "66042a93389cb8d0e6c867be87382566c753cd965ceda619e947e73de4a607be"
    ),
    "d389_csv": (
        "1fdbaac1c756983c8bd2d2d8e8eabed36a4530393b5f3e3491678335d778f66f"
    ),
}
EXPECTED_D389_VERDICT = "D389_AUDIT_CONTRACT_FAIL_STOP"
EXPECTED_PAIR_COUNT = 36
EXPECTED_CALL_COUNT = 144
EXPECTED_FAILED_CALL_COUNT = 41
EXPECTED_SUCCESS_CALL_COUNT = 103
EXPECTED_AFFECTED_PAIR_COUNT = 26
EXPECTED_FAILURE_BREAKDOWN = {
    "FEWER_THAN_FOUR_UNIQUE_POINTS": 12,
    "AFFINE_RANK_LT_3": 17,
    "QH6154_FLAT_OR_COPLANAR": 12,
}
EXPECTED_STAGE_BREAKDOWN = {"pre_float32": 24, "post_float32": 17}
EXPECTED_DIRECTION_BREAKDOWN = {
    "left_clipped_by_right": 16,
    "right_clipped_by_left": 25,
}
EXPECTED_ADJACENCY_BREAKDOWN = {"adjacent": 13, "nonadjacent": 28}
EXPECTED_PREPARE_STATUS_LINES = [
    " M START_HERE.md",
    " M claudedocs/DECISIONS.md",
    " M claudedocs/EXPERIMENT_LEDGER.md",
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt1_d388_overlap_gate_numeric_provenance_and_canonical_tie_audit/"
        "d389_phase_markers.jsonl"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt1_d388_overlap_gate_numeric_provenance_and_canonical_tie_audit/"
        "d389_preregistration.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_board_layout_validation.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_failure_attestation.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_manual_visual_inspection.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_manual_visual_inspection_template.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_numeric_and_tie_audit_evidence.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_numeric_provenance_and_tie_audit.rbl"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_numeric_provenance_and_tie_audit.rrd"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_offline_audit_invocation.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_offline_worker_claim.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_offline_worker_supervisor.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_phase_markers.jsonl"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_preregistration.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_reconstructed_seam_witness_geometry.json"
    ),
    (
        "?? claudedocs/runtime_logs/grasp_track/g0a_d389/"
        "attempt2_prereg_status_whitespace_repair/"
        "d389_rerun_validation.json"
    ),
    (
        "?? claudedocs/session_20260726_grasp_g0a_d389_d388_overlap_gate_"
        "numeric_provenance_and_canonical_tie_audit.md"
    ),
    (
        "?? sim_scripts/cyl34_top_view_d389_attempt2_prereg_status_"
        "whitespace_repair.py"
    ),
    (
        "?? sim_scripts/cyl34_top_view_d389_d388_overlap_gate_numeric_"
        "provenance_and_canonical_tie_audit.py"
    ),
    (
        "?? sim_scripts/cyl34_top_view_d390_d389_directional_epsilon0_"
        "boundary_collapse_semantics_localization.py"
    ),
]
NEW_VARIABLES = [
    "directional_epsilon0_terminal_affine_rank_boundary_classification_v1"
]
DEADLINE_SECONDS = 300.0
FLOAT64_EPSILON = float(np.finfo(np.float64).eps)

PREREG = OUT_DIR / "d390_preregistration.json"
PHASES = OUT_DIR / "d390_phase_markers.jsonl"
INVOCATION = OUT_DIR / "d390_offline_localization_invocation.json"
STDOUT = OUT_DIR / "d390_offline_worker_stdout.log"
STDERR = OUT_DIR / "d390_offline_worker_stderr.log"
WORKER_AUTHORIZATION = OUT_DIR / "d390_worker_authorization.json"
WORKER_SENTINEL = OUT_DIR / "d390_worker_start_sentinel.json"
SUPERVISOR = OUT_DIR / "d390_offline_worker_supervisor.json"
EVIDENCE = OUT_DIR / "d390_boundary_collapse_localization_evidence.json"
TRACE_GEOMETRY = OUT_DIR / "d390_terminal_candidate_geometry.json"
TRACE_CSV = OUT_DIR / "d390_failed_directional_call_trace.csv"
BOARD = OUT_DIR / "d390_boundary_collapse_localization_1920x1080.png"
BOARD_LAYOUT = OUT_DIR / "d390_board_layout_validation.json"
RRD = OUT_DIR / "d390_boundary_collapse_localization.rrd"
RBL = OUT_DIR / "d390_boundary_collapse_localization.rbl"
RERUN_VALIDATION = OUT_DIR / "d390_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d390_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d390_manual_visual_inspection_template.json"
MANUAL = OUT_DIR / "d390_manual_visual_inspection.json"
WORKER_CLAIM = OUT_DIR / "d390_offline_worker_claim.json"
FAILURE_ATTESTATION = OUT_DIR / "d390_failure_attestation.json"
COMPLETION = OUT_DIR / "d390_completion_summary.json"

PREPARE_INVENTORY = {PREREG.name, PHASES.name}
PRE_WORKER_INVENTORY = PREPARE_INVENTORY | {
    INVOCATION.name,
    STDOUT.name,
    STDERR.name,
    WORKER_AUTHORIZATION.name,
}
WORKER_START_INVENTORY = PRE_WORKER_INVENTORY | {WORKER_SENTINEL.name}
PRE_CLAIM_SUCCESS_INVENTORY = WORKER_START_INVENTORY | {
    EVIDENCE.name,
    TRACE_GEOMETRY.name,
    TRACE_CSV.name,
    BOARD.name,
    BOARD_LAYOUT.name,
    RRD.name,
    RBL.name,
    RERUN_VALIDATION.name,
    RERUN_SCREENSHOT.name,
    MANUAL_TEMPLATE.name,
}
POST_WORKER_SUCCESS_INVENTORY = PRE_CLAIM_SUCCESS_INVENTORY | {
    WORKER_CLAIM.name
}
PRE_FINALIZE_INVENTORY = POST_WORKER_SUCCESS_INVENTORY | {
    SUPERVISOR.name,
    MANUAL.name,
}
POST_FINALIZE_INVENTORY = PRE_FINALIZE_INVENTORY | {COMPLETION.name}

RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

SCOPE_COUNTERS = {
    key: 0
    for key in (
        "d389_imports_or_executions",
        "d389_files_written_or_finalized",
        "d389_successful_directional_calls_recomputed",
        "d389_strict_halfspace_calls_recomputed",
        "d389_frozen_5nm_calls_recomputed",
        "qj_or_random_jitter_uses",
        "epsilon_5nm_tolerance_or_gate_changes",
        "partition_budget_or_geometry_changes",
        "collider_asset_or_usd_materializations",
        "isaac_kit_physx_warp_cuda_launches",
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
    """Raised by the worker itself; the supervisor has no signal authority."""


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(b"|float64|C|")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha(value: Any) -> str:
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
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _status_path(line: str) -> str:
    if len(line) < 4:
        return ""
    path = line[3:]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"')


def _status_scope_pass(lines: list[str], *, allow_d390_output: bool) -> bool:
    if not allow_d390_output:
        return lines == EXPECTED_PREPARE_STATUS_LINES
    baseline = set(EXPECTED_PREPARE_STATUS_LINES)
    if not baseline.issubset(set(lines)):
        return False
    for line in lines:
        if line in baseline:
            continue
        if (
            line.startswith("?? ")
            and _status_path(line).startswith(
                "claudedocs/runtime_logs/grasp_track/g0a_d390/"
            )
        ):
            continue
        return False
    return True


def _out_names() -> set[str]:
    if not OUT_DIR.is_dir():
        return set()
    return {path.name for path in OUT_DIR.iterdir()}


def _require_out_inventory(expected: set[str], stage: str) -> None:
    observed = _out_names()
    if observed != expected:
        raise RuntimeError(
            f"D390 output inventory mismatch at {stage}: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _input_hashes() -> dict[str, str]:
    return {
        "d389_evidence": _sha(D389_EVIDENCE),
        "d389_geometry": _sha(D389_GEOMETRY),
        "d389_csv": _sha(D389_CSV),
    }


def _prepare_end_preregistration_sha256() -> str | None:
    if not PHASES.is_file():
        return None
    rows = [
        json.loads(line)
        for line in PHASES.read_text(encoding="utf-8").splitlines()
        if line
    ]
    matches = [
        row
        for row in rows
        if row.get("phase") == "prepare_end"
    ]
    if len(matches) != 1:
        return None
    value = matches[0].get("preregistration_sha256")
    return str(value) if value is not None else None


def _frozen_authority_checks(
    prereg: dict[str, Any], *, allow_d390_output: bool
) -> dict[str, bool]:
    return {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_master_exact": _git("rev-parse", "origin/master")
        == EXPECTED_HEAD,
        "start_here_hash_exact": _sha(START) == EXPECTED_START_SHA256,
        "start_here_hash_matches_preregistration": prereg.get(
            "authorization"
        )
        == {
            "path": _rel(START),
            "sha256": EXPECTED_START_SHA256,
        },
        "immutable_input_hashes_exact": _input_hashes()
        == EXPECTED_INPUT_SHA256,
        "script_hash_matches_preregistration": prereg.get("script", {}).get(
            "sha256"
        )
        == _sha(SCRIPT),
        "preregistration_hash_matches_prepare_end_marker": (
            _prepare_end_preregistration_sha256() == _sha(PREREG)
        ),
        "prepare_status_manifest_exact": prereg.get("git", {}).get(
            "status_before_output_create"
        )
        == EXPECTED_PREPARE_STATUS_LINES,
        "current_git_scope_exact": _status_scope_pass(
            _status_lines(), allow_d390_output=allow_d390_output
        ),
    }


def _worker_authorization_checks() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, bool]
]:
    invocation = _read_json(INVOCATION)
    authorization = _read_json(WORKER_AUTHORIZATION)
    expected_command = [
        sys.executable,
        "-B",
        str(SCRIPT),
        "--stage",
        "worker",
    ]
    checks = {
        "invocation_artifact_exact": invocation.get("artifact")
        == "D390_OFFLINE_LOCALIZATION_INVOCATION_V1",
        "authorization_artifact_exact": authorization.get("artifact")
        == "D390_WORKER_AUTHORIZATION_V1",
        "command_exact": invocation.get("command") == expected_command,
        "cwd_exact": invocation.get("cwd") == str(REPO),
        "worker_index_exact": invocation.get("worker_invocation_index") == 1
        and authorization.get("worker_invocation_index") == 1,
        "retry_zero": invocation.get("retries") == 0
        and authorization.get("retries") == 0,
        "script_hash_exact": invocation.get("script_sha256") == _sha(SCRIPT)
        and authorization.get("script_sha256") == _sha(SCRIPT),
        "input_hashes_exact": invocation.get("input_hashes")
        == EXPECTED_INPUT_SHA256
        and authorization.get("input_hashes") == EXPECTED_INPUT_SHA256,
        "start_here_hash_exact": invocation.get("start_here_sha256")
        == EXPECTED_START_SHA256
        and authorization.get("start_here_sha256")
        == EXPECTED_START_SHA256,
        "invocation_hash_bound": authorization.get("invocation_sha256")
        == _sha(INVOCATION),
        "preregistration_hash_bound": invocation.get(
            "preregistration_sha256"
        )
        == _sha(PREREG)
        and authorization.get("preregistration_sha256") == _sha(PREREG),
        "parent_is_authorizing_supervisor": authorization.get(
            "supervisor_pid"
        )
        == os.getppid(),
        "stdout_stderr_precreated": STDOUT.is_file() and STDERR.is_file(),
        "worker_start_sentinel_absent": not WORKER_SENTINEL.exists(),
    }
    return invocation, authorization, checks


def _phase(name: str, **payload: Any) -> None:
    record = {
        "phase": name,
        "monotonic_seconds": time.monotonic(),
        "wall_time_ns": time.time_ns(),
        **payload,
    }
    with PHASES.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _native(record),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


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


def _failure_family(error: str | None) -> str | None:
    text = str(error or "")
    if "fewer than four unique points" in text:
        return "FEWER_THAN_FOUR_UNIQUE_POINTS"
    if "points are not three-dimensional" in text:
        return "AFFINE_RANK_LT_3"
    if "QH6154" in text:
        return "QH6154_FLAT_OR_COPLANAR"
    return None


def _affine_diagnostics(points: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    unique = np.unique(raw, axis=0)
    count = int(len(unique))
    if count == 0:
        singular = np.zeros(3, dtype=np.float64)
        anchor_singular = np.zeros(3, dtype=np.float64)
        threshold = 0.0
        anchor_threshold = 0.0
        rank = 0
        anchor_rank = 0
        numpy_rank = 0
    else:
        centered = unique - unique.mean(axis=0)
        anchor_centered = unique - unique[0]
        singular_raw = np.linalg.svd(
            centered, compute_uv=False, full_matrices=False
        )
        anchor_raw = np.linalg.svd(
            anchor_centered, compute_uv=False, full_matrices=False
        )
        singular = np.pad(singular_raw, (0, 3 - len(singular_raw)))
        anchor_singular = np.pad(anchor_raw, (0, 3 - len(anchor_raw)))
        threshold = (
            float(singular[0])
            * float(max(count, 3))
            * FLOAT64_EPSILON
        )
        anchor_threshold = (
            float(anchor_singular[0])
            * float(max(count, 3))
            * FLOAT64_EPSILON
        )
        rank = int(np.count_nonzero(singular > threshold))
        anchor_rank = int(
            np.count_nonzero(anchor_singular > anchor_threshold)
        )
        numpy_rank = int(np.linalg.matrix_rank(centered))
    if count == 0:
        affine_class = "EMPTY"
    elif rank == 0:
        affine_class = "POINT"
    elif rank == 1:
        affine_class = "LINE"
    elif rank == 2:
        affine_class = "FACE_LIKE"
    elif rank == 3:
        affine_class = "FULL_DIMENSIONAL"
    else:
        affine_class = "NUMERIC_INDETERMINATE"
    largest = float(singular[0])
    ratios = [
        float(value / largest) if largest > 0.0 else 0.0
        for value in singular
    ]
    return {
        "raw_point_count": int(len(raw)),
        "unique_point_count": count,
        "raw_points_sha256": _array_sha(raw),
        "unique_points_sha256": _array_sha(unique),
        "singular_values_m": singular,
        "singular_value_ratios": ratios,
        "rank_threshold_m": threshold,
        "rank_threshold_formula": (
            "sigma_max * max(unique_point_count,3) * float64_epsilon"
        ),
        "float64_epsilon": FLOAT64_EPSILON,
        "affine_rank": rank,
        "numpy_matrix_rank": numpy_rank,
        "first_point_anchor_singular_values_m": anchor_singular,
        "first_point_anchor_rank_threshold_m": anchor_threshold,
        "first_point_anchor_affine_rank": anchor_rank,
        "independent_rank_checks_agree": (
            rank == numpy_rank == anchor_rank
        ),
        "affine_class": affine_class,
        "unique_points_f64_m": unique,
    }


def _polyhedron_edges(
    points: np.ndarray, *, qhull_options: str | None
) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(source, qhull_options=qhull_options)
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
    edges = sorted(pair for pair, count in memberships.items() if count >= 2)
    if not edges:
        raise RuntimeError("convex edge reconstruction failed")
    return np.asarray(edges, dtype=np.int64)


def _selected_unrounded_planes(
    clipping: np.ndarray, *, qhull_options: str | None
) -> np.ndarray:
    equations = np.asarray(
        ConvexHull(clipping, qhull_options=qhull_options).equations,
        dtype=np.float64,
    )
    normalized = equations / np.linalg.norm(
        equations[:, :3], axis=1
    )[:, None]
    selected: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for equation in normalized:
        key = tuple(np.round(equation, decimals=12))
        if key not in seen:
            seen.add(key)
            selected.append(equation)
    return np.asarray(selected, dtype=np.float64)


def _clip_candidate(
    points: np.ndarray,
    equation: np.ndarray,
    *,
    qhull_options: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(points, dtype=np.float64)
    unit = np.asarray(equation[:3], dtype=np.float64)
    length = float(np.linalg.norm(unit))
    unit /= length
    signed_offset = float(equation[3]) / length
    values = source @ unit + signed_offset
    keep = values <= 0.0
    output = [point for point in source[keep]]
    edges = _polyhedron_edges(source, qhull_options=qhull_options)
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
        "source_point_count": int(len(source)),
        "kept_source_vertex_count": int(np.count_nonzero(keep)),
        "source_edge_count": int(len(edges)),
        "strict_crossing_edge_count": crossing_count,
        "minimum_signed_value_m": float(values.min()),
        "maximum_signed_value_m": float(values.max()),
    }


def _trace_failed_call(
    source: np.ndarray,
    clipping: np.ndarray,
    recorded: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fallback_used = bool(recorded["qhull_fallback_used"])
    recorded_fallback_options = recorded.get("qhull_fallback_options")
    fallback_branch_contract = (
        (not fallback_used and recorded_fallback_options is None)
        or (
            fallback_used
            and recorded_fallback_options == "Q12 Pp"
        )
    )
    if not fallback_branch_contract:
        raise RuntimeError(
            "D390 recorded fallback branch is neither None nor Q12 Pp"
        )
    qhull_options = "Q12 Pp" if fallback_used else None
    source_unique = np.unique(
        np.asarray(source, dtype=np.float64), axis=0
    )
    clipping_unique = np.unique(
        np.asarray(clipping, dtype=np.float64), axis=0
    )
    source_diag = _affine_diagnostics(source_unique)
    clipping_diag = _affine_diagnostics(clipping_unique)
    if (
        len(source_unique) < 4
        or len(clipping_unique) < 4
        or source_diag["affine_rank"] < 3
        or clipping_diag["affine_rank"] < 3
    ):
        raise RuntimeError("D390 received a non-3D D389 input child")
    equations = _selected_unrounded_planes(
        clipping_unique, qhull_options=qhull_options
    )
    points = source_unique
    active_clip_ordinal = 0
    skipped = 0
    plane_records: list[dict[str, Any]] = []
    collapse: dict[str, Any] | None = None
    for plane_index, equation in enumerate(equations):
        _deadline("trace_failed_call_plane")
        values = points @ equation[:3] + equation[3]
        minimum = float(values.min())
        maximum = float(values.max())
        base = {
            "selected_plane_index_zero_based": plane_index,
            "plane_equation_f64_m": equation,
            "plane_equation_sha256": _array_sha(equation.reshape(1, 4)),
            "points_before_sha256": _array_sha(points),
            "points_before_count": int(len(points)),
            "minimum_signed_value_m": minimum,
            "maximum_signed_value_m": maximum,
        }
        if maximum <= 0.0:
            skipped += 1
            plane_records.append({**base, "branch": "SKIP_INSIDE"})
            continue
        if minimum > 0.0:
            plane_records.append(
                {
                    **base,
                    "branch": "STRICTLY_DISJOINT",
                    "active_clip_ordinal_one_based": active_clip_ordinal,
                }
            )
            collapse = {
                "failure_location": "STRICTLY_DISJOINT_BEFORE_RECORDED_FAILURE",
                "selected_plane_index_zero_based": plane_index,
                "active_clip_ordinal_one_based": active_clip_ordinal,
                "candidate": _affine_diagnostics(
                    np.empty((0, 3), dtype=np.float64)
                ),
                "reconstructed_error_family": None,
                "reconstructed_error": None,
            }
            break
        active_clip_ordinal += 1
        try:
            raw, clip_meta = _clip_candidate(
                points, equation, qhull_options=qhull_options
            )
        except (ValueError, QhullError, RuntimeError) as exc:
            current = _affine_diagnostics(points)
            family = _failure_family(f"{type(exc).__name__}: {exc}")
            plane_records.append(
                {
                    **base,
                    "branch": "EDGE_OR_PRE_CANDIDATE_FAILURE",
                    "active_clip_ordinal_one_based": active_clip_ordinal,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            collapse = {
                "failure_location": "EDGE_RECONSTRUCTION_BEFORE_CANDIDATE",
                "selected_plane_index_zero_based": plane_index,
                "active_clip_ordinal_one_based": active_clip_ordinal,
                "candidate": current,
                "reconstructed_error_family": family,
                "reconstructed_error": f"{type(exc).__name__}: {exc}",
            }
            break
        candidate = _affine_diagnostics(raw)
        candidate_unique = np.asarray(
            candidate["unique_points_f64_m"], dtype=np.float64
        )
        reconstructed_error: str | None = None
        reconstructed_family: str | None = None
        next_points: np.ndarray | None = None
        if candidate["unique_point_count"] < 4:
            reconstructed_error = "ValueError: fewer than four unique points"
            reconstructed_family = "FEWER_THAN_FOUR_UNIQUE_POINTS"
        elif candidate["affine_rank"] < 3:
            reconstructed_error = "ValueError: points are not three-dimensional"
            reconstructed_family = "AFFINE_RANK_LT_3"
        else:
            try:
                hull = ConvexHull(
                    candidate_unique, qhull_options=qhull_options
                )
                next_points = candidate_unique[
                    np.asarray(hull.vertices, dtype=np.int64)
                ]
            except QhullError as exc:
                reconstructed_error = f"QhullError: {exc}"
                reconstructed_family = _failure_family(reconstructed_error)
        plane_records.append(
            {
                **base,
                **clip_meta,
                "branch": (
                    "COLLAPSE"
                    if reconstructed_error is not None
                    else "CLIP_CONTINUE"
                ),
                "active_clip_ordinal_one_based": active_clip_ordinal,
                "candidate_raw_point_count": candidate["raw_point_count"],
                "candidate_unique_point_count": candidate[
                    "unique_point_count"
                ],
                "candidate_affine_rank": candidate["affine_rank"],
                "candidate_affine_class": candidate["affine_class"],
                "candidate_unique_points_sha256": candidate[
                    "unique_points_sha256"
                ],
                "reconstructed_error_family": reconstructed_family,
            }
        )
        if reconstructed_error is not None:
            collapse = {
                "failure_location": "ACTIVE_CLIP_CANDIDATE_HULL",
                "selected_plane_index_zero_based": plane_index,
                "active_clip_ordinal_one_based": active_clip_ordinal,
                "candidate": candidate,
                "reconstructed_error_family": reconstructed_family,
                "reconstructed_error": reconstructed_error,
            }
            break
        if next_points is None:
            raise RuntimeError("D390 internal next-points contract failed")
        points = next_points
    if collapse is None:
        final_candidate = _affine_diagnostics(points)
        try:
            ConvexHull(points, qhull_options=qhull_options)
        except QhullError as exc:
            reconstructed_error = f"QhullError: {exc}"
            collapse = {
                "failure_location": "AFTER_LAST_PLANE_FINAL_HULL",
                "selected_plane_index_zero_based": None,
                "active_clip_ordinal_one_based": active_clip_ordinal,
                "candidate": final_candidate,
                "reconstructed_error_family": _failure_family(
                    reconstructed_error
                ),
                "reconstructed_error": reconstructed_error,
            }
    if collapse is None:
        raise RuntimeError("recorded failed call independently completed")
    recorded_family = _failure_family(recorded.get("error"))
    collapse["recorded_error_family"] = recorded_family
    collapse["recorded_error"] = recorded.get("error")
    collapse["recorded_clip_count"] = int(recorded["clip_count"])
    collapse["recorded_skipped_inside_plane_count"] = int(
        recorded["skipped_inside_plane_count"]
    )
    collapse["qhull_fallback_used"] = fallback_used
    collapse["qhull_options"] = qhull_options
    collapse[
        "recorded_fallback_branch_schema_valid_and_replayed"
    ] = fallback_branch_contract and qhull_options == recorded_fallback_options
    collapse["error_family_matches_recorded"] = (
        collapse["reconstructed_error_family"] == recorded_family
    )
    collapse["active_clip_count_matches_recorded"] = (
        collapse["active_clip_ordinal_one_based"]
        == int(recorded["clip_count"])
    )
    collapse["skipped_count_through_collapse"] = skipped
    collapse["skipped_count_matches_recorded"] = (
        skipped == int(recorded["skipped_inside_plane_count"])
    )
    collapse["rank_contract_pass"] = collapse["candidate"][
        "independent_rank_checks_agree"
    ]
    summary = {
        key: value
        for key, value in collapse.items()
        if key != "candidate"
    }
    summary["candidate"] = {
        key: value
        for key, value in collapse["candidate"].items()
        if key != "unique_points_f64_m"
    }
    summary["plane_trace_count"] = len(plane_records)
    summary["plane_trace_sha256"] = _canonical_sha(plane_records)
    geometry = {
        "source_vertices_f64_m": source_unique,
        "clipping_vertices_f64_m": clipping_unique,
        "source_vertices_sha256": _array_sha(source_unique),
        "clipping_vertices_sha256": _array_sha(clipping_unique),
        "terminal_candidate_unique_points_f64_m": collapse["candidate"][
            "unique_points_f64_m"
        ],
        "terminal_candidate_unique_points_sha256": collapse["candidate"][
            "unique_points_sha256"
        ],
        "collapse_plane_equation_f64_m": (
            equations[int(collapse["selected_plane_index_zero_based"])]
            if collapse["selected_plane_index_zero_based"] is not None
            else None
        ),
        "plane_trace": plane_records,
    }
    return summary, geometry


def _strict_relation(strict: dict[str, Any]) -> dict[str, Any]:
    if strict.get("calculation_pass") is not True:
        return {
            "classification": "STRICT_AUTHORITY_UNAVAILABLE",
            "pass": False,
        }
    radius = strict.get("signed_inradius_nm")
    threshold = strict.get("strict_interior_radius_threshold_nm")
    if radius is None or threshold is None:
        return {
            "classification": "STRICT_AUTHORITY_UNAVAILABLE",
            "pass": False,
        }
    if strict.get("positive_volume") is True:
        classification = "STRICT_FULL_DIMENSIONAL_POSITIVE"
    elif float(radius) < -float(threshold):
        classification = "STRICT_INFEASIBLE_GAP"
    else:
        classification = (
            "STRICT_NONPOSITIVE_OR_SUBTHRESHOLD_WITHIN_SOLVER_BAND"
        )
    return {
        "classification": classification,
        "calculation_pass": True,
        "positive_volume": strict.get("positive_volume"),
        "volume_m3": strict.get("volume_m3"),
        "signed_inradius_nm": radius,
        "strict_interior_radius_threshold_nm": threshold,
        "source": "immutable D389 stored strict-halfspace result; not recomputed",
        "pass": True,
    }


def _semantic_relation(
    affine_class: str, strict_class: str
) -> tuple[str, bool]:
    rank_limited = affine_class in {
        "EMPTY",
        "POINT",
        "LINE",
        "FACE_LIKE",
    }
    if strict_class == "STRICT_FULL_DIMENSIONAL_POSITIVE":
        if rank_limited:
            return "STRICT_POSITIVE_VS_DIRECTIONAL_COLLAPSE_OBSERVED", True
        return "STRICT_POSITIVE_WITH_QHULL_NUMERIC_FAILURE", True
    if (
        strict_class
        == "STRICT_NONPOSITIVE_OR_SUBTHRESHOLD_WITHIN_SOLVER_BAND"
    ):
        if rank_limited:
            return (
                "LOWER_DIMENSIONAL_FIRST_COLLAPSE_WITH_STRICT_SOLVER_BAND",
                True,
            )
        return (
            "STRICT_SOLVER_BAND_WITH_FULL_RANK_QHULL_PRECISION_FAILURE",
            True,
        )
    if strict_class == "STRICT_INFEASIBLE_GAP":
        if rank_limited:
            return "INTERMEDIATE_COLLAPSE_BEFORE_STORED_GAP_PROOF", True
        return "GAP_WITH_FULL_RANK_QHULL_PRECISION_FAILURE", True
    return "STRICT_AUTHORITY_UNAVAILABLE", False


def _synthetic_controls() -> dict[str, Any]:
    controls = {
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
    rows: dict[str, Any] = {}
    for expected, points in controls.items():
        base = _affine_diagnostics(points)
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
        altered = _affine_diagnostics(transformed)
        checks = {
            "base_class_exact": base["affine_class"] == expected,
            "transformed_permuted_duplicated_class_exact": (
                altered["affine_class"] == expected
            ),
            "base_rank_checks_agree": base[
                "independent_rank_checks_agree"
            ],
            "transformed_rank_checks_agree": altered[
                "independent_rank_checks_agree"
            ],
        }
        rows[expected] = {
            "base": {
                key: value
                for key, value in base.items()
                if key != "unique_points_f64_m"
            },
            "power_of_two_scale_translation_axis_row_duplicate_control": {
                key: value
                for key, value in altered.items()
                if key != "unique_points_f64_m"
            },
            "checks": checks,
            "pass": all(checks.values()),
        }
    cube = np.asarray(
        [
            [x, y, z]
            for x in (0.0, 1.0)
            for y in (0.0, 1.0)
            for z in (0.0, 1.0)
        ],
        dtype=np.float64,
    )
    plane_fixtures = {
        "EMPTY": np.asarray([1.0, 0.0, 0.0, 1.0]),
        "POINT": np.asarray([1.0, 1.0, 1.0, 0.0]),
        "LINE": np.asarray([1.0, 1.0, 0.0, 0.0]),
        "FACE_LIKE": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "FULL_DIMENSIONAL": np.asarray([1.0, 0.0, 0.0, -0.5]),
    }
    pipeline_rows: dict[str, Any] = {}
    for expected, equation in plane_fixtures.items():
        raw, clip_meta = _clip_candidate(
            cube, equation, qhull_options=None
        )
        diagnostic = _affine_diagnostics(raw)
        checks = {
            "single_plane_candidate_class_exact": diagnostic[
                "affine_class"
            ]
            == expected,
            "rank_checks_agree": diagnostic[
                "independent_rank_checks_agree"
            ],
            "candidate_points_finite": bool(np.isfinite(raw).all()),
            "plane_equation_binary_exact_fixture": all(
                float(value).is_integer()
                or value in {-0.5, 0.5}
                for value in equation
            ),
        }
        pipeline_rows[expected] = {
            "plane_equation": equation,
            "clip_metadata": clip_meta,
            "diagnostic": {
                key: value
                for key, value in diagnostic.items()
                if key != "unique_points_f64_m"
            },
            "checks": checks,
            "pass": all(checks.values()),
        }
    low_height = math.ldexp(1.0, -60)
    high_height = math.ldexp(1.0, -40)

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

    thin_low = _affine_diagnostics(thin_tetra(low_height))
    thin_high = _affine_diagnostics(thin_tetra(high_height))
    threshold_checks = {
        "binary_exact_low_fixture_classifies_face_like": thin_low[
            "affine_class"
        ]
        == "FACE_LIKE",
        "binary_exact_high_fixture_classifies_full_dimensional": thin_high[
            "affine_class"
        ]
        == "FULL_DIMENSIONAL",
        "low_fixture_rank_checks_agree": thin_low[
            "independent_rank_checks_agree"
        ],
        "high_fixture_rank_checks_agree": thin_high[
            "independent_rank_checks_agree"
        ],
    }
    pipeline_pass = all(row["pass"] for row in pipeline_rows.values())
    threshold_pass = all(threshold_checks.values())
    return {
        "coordinate_policy": (
            "deterministic exact binary values; no random jitter or QJ"
        ),
        "class_controls": rows,
        "single_plane_clip_candidate_controls": pipeline_rows,
        "rank_threshold_straddle_controls": {
            "low_height_power_of_two": low_height,
            "high_height_power_of_two": high_height,
            "low": {
                key: value
                for key, value in thin_low.items()
                if key != "unique_points_f64_m"
            },
            "high": {
                key: value
                for key, value in thin_high.items()
                if key != "unique_points_f64_m"
            },
            "checks": threshold_checks,
            "pass": threshold_pass,
        },
        "control_count": len(rows) + len(pipeline_rows) + 2,
        "checks": {
            "all_five_classes_present": set(rows)
            == {
                "EMPTY",
                "POINT",
                "LINE",
                "FACE_LIKE",
                "FULL_DIMENSIONAL",
            },
            "all_controls_pass": all(row["pass"] for row in rows.values()),
            "all_single_plane_clip_candidate_controls_pass": pipeline_pass,
            "rank_threshold_straddle_controls_pass": threshold_pass,
        },
        "pass": (
            all(row["pass"] for row in rows.values())
            and pipeline_pass
            and threshold_pass
        ),
    }


def _csv_lineage(
    evidence_pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    with D389_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    evidence_keys = [
        (
            str(row["target"]),
            int(row["left_index"]),
            int(row["right_index"]),
            bool(row["adjacent"]),
        )
        for row in evidence_pairs
    ]
    csv_keys = [
        (
            str(row["target"]),
            int(row["left_index"]),
            int(row["right_index"]),
            row["adjacent"] == "True",
        )
        for row in rows
    ]
    checks = {
        "row_count_exact_36": len(rows) == EXPECTED_PAIR_COUNT,
        "pair_order_and_adjacency_exact": csv_keys == evidence_keys,
        "header_has_required_fields": {
            "target",
            "prim_name",
            "left_index",
            "right_index",
            "adjacent",
            "per_pair_classification",
            "pre_signed_inradius_nm",
            "post_signed_inradius_nm",
        }.issubset(set(rows[0])) if rows else False,
    }
    return {
        "row_count": len(rows),
        "ordered_pair_manifest_sha256": _canonical_sha(csv_keys),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _build_failed_manifest(
    evidence: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = evidence["seam_numeric_provenance_audit"]["pair_results"]
    stages = [
        (
            "pre_float32",
            "pre_float32_directional_epsilon0",
            "pre_float32_epsilon0",
        ),
        (
            "post_float32",
            "post_float32_directional_epsilon0",
            "post_float32_epsilon0",
        ),
    ]
    directions = [
        "left_clipped_by_right",
        "right_clipped_by_left",
    ]
    failed: list[dict[str, Any]] = []
    successful = 0
    for pair_index, pair in enumerate(pairs):
        for stage_short, directional_key, strict_key in stages:
            for direction in directions:
                recorded = pair[directional_key][direction]
                if recorded["calculation_pass"] is True:
                    successful += 1
                    continue
                call_id = (
                    f"{pair['target'].lower()}_"
                    f"{int(pair['left_index']):02d}_"
                    f"{int(pair['right_index']):02d}_"
                    f"{stage_short}_"
                    f"{'lbr' if direction == 'left_clipped_by_right' else 'rbl'}"
                )
                failed.append(
                    {
                        "call_id": call_id,
                        "pair_index": pair_index,
                        "target": pair["target"],
                        "prim_name": pair["prim_name"],
                        "left_index": int(pair["left_index"]),
                        "right_index": int(pair["right_index"]),
                        "adjacent": bool(pair["adjacent"]),
                        "stage": stage_short,
                        "direction": direction,
                        "directional_key": directional_key,
                        "strict_key": strict_key,
                        "recorded_directional_failure": recorded,
                        "stored_strict_halfspace": pair[strict_key],
                    }
                )
    affected = {
        (
            row["target"],
            row["left_index"],
            row["right_index"],
        )
        for row in failed
    }
    breakdown_family: dict[str, int] = {}
    breakdown_stage: dict[str, int] = {}
    breakdown_direction: dict[str, int] = {}
    breakdown_adjacency: dict[str, int] = {}
    for row in failed:
        family = _failure_family(
            row["recorded_directional_failure"].get("error")
        )
        breakdown_family[family or "UNKNOWN"] = (
            breakdown_family.get(family or "UNKNOWN", 0) + 1
        )
        breakdown_stage[row["stage"]] = (
            breakdown_stage.get(row["stage"], 0) + 1
        )
        breakdown_direction[row["direction"]] = (
            breakdown_direction.get(row["direction"], 0) + 1
        )
        adjacency = "adjacent" if row["adjacent"] else "nonadjacent"
        breakdown_adjacency[adjacency] = (
            breakdown_adjacency.get(adjacency, 0) + 1
        )
    ids = [row["call_id"] for row in failed]
    checks = {
        "d389_verdict_exact": evidence.get("verdict")
        == EXPECTED_D389_VERDICT,
        "pair_count_exact_36": len(pairs) == EXPECTED_PAIR_COUNT,
        "total_directional_call_count_exact_144": (
            len(pairs) * 2 * 2 == EXPECTED_CALL_COUNT
        ),
        "failed_call_count_exact_41": len(failed)
        == EXPECTED_FAILED_CALL_COUNT,
        "successful_call_count_exact_103": successful
        == EXPECTED_SUCCESS_CALL_COUNT,
        "affected_pair_count_exact_26": len(affected)
        == EXPECTED_AFFECTED_PAIR_COUNT,
        "call_ids_unique": len(set(ids)) == len(ids),
        "failure_family_breakdown_exact": breakdown_family
        == EXPECTED_FAILURE_BREAKDOWN,
        "stage_breakdown_exact": breakdown_stage
        == EXPECTED_STAGE_BREAKDOWN,
        "direction_breakdown_exact": breakdown_direction
        == EXPECTED_DIRECTION_BREAKDOWN,
        "adjacency_breakdown_exact": breakdown_adjacency
        == EXPECTED_ADJACENCY_BREAKDOWN,
    }
    return failed, {
        "pair_count": len(pairs),
        "total_directional_call_count": len(pairs) * 4,
        "failed_call_count": len(failed),
        "successful_call_count": successful,
        "affected_pair_count": len(affected),
        "failure_family_breakdown": breakdown_family,
        "stage_breakdown": breakdown_stage,
        "direction_breakdown": breakdown_direction,
        "adjacency_breakdown": breakdown_adjacency,
        "ordered_call_ids": ids,
        "ordered_call_manifest_sha256": _text_sha("\n".join(ids) + "\n"),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source_evidence = _read_json(D389_EVIDENCE)
    source_geometry = _read_json(D389_GEOMETRY)
    failed, manifest = _build_failed_manifest(source_evidence)
    csv_lineage = _csv_lineage(
        source_evidence["seam_numeric_provenance_audit"]["pair_results"]
    )
    layers = {
        str(layer["prim_name"]): layer
        for layer in source_geometry["layers"]
    }
    records: list[dict[str, Any]] = []
    geometry_records: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    strict_counts: dict[str, int] = {}
    semantic_counts: dict[str, int] = {}
    for call_index, row in enumerate(failed):
        _deadline("compute_failed_call")
        layer = layers[row["prim_name"]]
        left = layer["children"][row["left_index"]]
        right = layer["children"][row["right_index"]]
        vertex_key = (
            "pre_float32_vertices_f64_m"
            if row["stage"] == "pre_float32"
            else "stored_post_float32_vertices_f64_m"
        )
        left_points = np.asarray(left[vertex_key], dtype=np.float64)
        right_points = np.asarray(right[vertex_key], dtype=np.float64)
        if row["direction"] == "left_clipped_by_right":
            source_points, clipping_points = left_points, right_points
            source_child_index, clipping_child_index = (
                row["left_index"],
                row["right_index"],
            )
        else:
            source_points, clipping_points = right_points, left_points
            source_child_index, clipping_child_index = (
                row["right_index"],
                row["left_index"],
            )
        trace, trace_geometry = _trace_failed_call(
            source_points,
            clipping_points,
            row["recorded_directional_failure"],
        )
        strict = _strict_relation(row["stored_strict_halfspace"])
        affine_class = trace["candidate"]["affine_class"]
        relation, relation_pass = _semantic_relation(
            affine_class, strict["classification"]
        )
        record_checks = {
            "error_family_matches_recorded": trace[
                "error_family_matches_recorded"
            ],
            "active_clip_count_matches_recorded": trace[
                "active_clip_count_matches_recorded"
            ],
            "skipped_plane_count_matches_recorded": trace[
                "skipped_count_matches_recorded"
            ],
            "recorded_fallback_branch_schema_valid_and_replayed": trace[
                "recorded_fallback_branch_schema_valid_and_replayed"
            ],
            "rank_contract_pass": trace["rank_contract_pass"],
            "strict_authority_bound": strict["pass"],
            "semantic_relation_not_contradictory": relation_pass,
            "candidate_points_finite": bool(
                np.isfinite(
                    trace_geometry[
                        "terminal_candidate_unique_points_f64_m"
                    ]
                ).all()
            ),
        }
        record = {
            "call_index": call_index,
            "call_id": row["call_id"],
            "target": row["target"],
            "prim_name": row["prim_name"],
            "pair_index": row["pair_index"],
            "left_index": row["left_index"],
            "right_index": row["right_index"],
            "adjacent": row["adjacent"],
            "stage": row["stage"],
            "direction": row["direction"],
            "source_child_index": source_child_index,
            "clipping_child_index": clipping_child_index,
            "recorded_failure": {
                "error": row["recorded_directional_failure"]["error"],
                "error_family": _failure_family(
                    row["recorded_directional_failure"]["error"]
                ),
                "clip_count": row["recorded_directional_failure"][
                    "clip_count"
                ],
                "skipped_inside_plane_count": row[
                    "recorded_directional_failure"
                ]["skipped_inside_plane_count"],
                "qhull_fallback_used": row[
                    "recorded_directional_failure"
                ]["qhull_fallback_used"],
                "qhull_fallback_options": row[
                    "recorded_directional_failure"
                ]["qhull_fallback_options"],
            },
            "independent_reconstruction": trace,
            "stored_strict_halfspace_relation": strict,
            "semantic_relation": relation,
            "first_collapse_is_implementation_order_provenance_not_geometry_invariant": True,
            "first_collapse_is_not_final_intersection_dimension": True,
            "checks": record_checks,
            "pass": all(record_checks.values()),
        }
        trace_geometry.update(
            {
                "call_index": call_index,
                "call_id": row["call_id"],
                "target": row["target"],
                "stage": row["stage"],
                "direction": row["direction"],
                "source_child_index": source_child_index,
                "clipping_child_index": clipping_child_index,
                "affine_class": affine_class,
                "affine_rank": trace["candidate"]["affine_rank"],
                "strict_relation": strict["classification"],
                "semantic_relation": relation,
            }
        )
        records.append(record)
        geometry_records.append(trace_geometry)
        class_counts[affine_class] = class_counts.get(affine_class, 0) + 1
        strict_class = strict["classification"]
        strict_counts[strict_class] = strict_counts.get(strict_class, 0) + 1
        semantic_counts[relation] = semantic_counts.get(relation, 0) + 1
    controls = _synthetic_controls()
    checks = {
        "immutable_input_hashes_exact": _input_hashes()
        == EXPECTED_INPUT_SHA256,
        "manifest_contract_pass": manifest["pass"],
        "csv_lineage_pass": csv_lineage["pass"],
        "all_41_record_reconstructions_pass": (
            len(records) == EXPECTED_FAILED_CALL_COUNT
            and all(row["pass"] for row in records)
        ),
        "class_count_sum_exact_41": sum(class_counts.values())
        == EXPECTED_FAILED_CALL_COUNT,
        "strict_count_sum_exact_41": sum(strict_counts.values())
        == EXPECTED_FAILED_CALL_COUNT,
        "synthetic_controls_pass": controls["pass"],
        "scope_counters_zero": all(
            value == 0 for value in SCOPE_COUNTERS.values()
        ),
    }
    localization_pass = all(checks.values())
    verdict = (
        "D390_DIRECTIONAL_EPSILON0_TERMINAL_AFFINE_RANK_LOCALIZATION_PASS_NO_D389_REPAIR"
        if localization_pass
        else "D390_TERMINAL_CLASSIFICATION_OR_TRACE_IDENTITY_FAIL_STOP"
    )
    evidence = {
        "artifact": "D390_BOUNDARY_COLLAPSE_LOCALIZATION_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Independently reconstruct only D389's 41 recorded failed "
            "directional epsilon-zero calls and classify the candidate point "
            "set at the first 3-D-only hull rejection."
        ),
        "new_variables": NEW_VARIABLES,
        "input_hashes": _input_hashes(),
        "immutable_d389_verdict": source_evidence["verdict"],
        "method_scope": {
            "runtime_inputs": [
                _rel(D389_EVIDENCE),
                _rel(D389_GEOMETRY),
                _rel(D389_CSV),
            ],
            "d389_script_imported_or_executed": False,
            "failed_calls_only": True,
            "successful_calls_recomputed": 0,
            "strict_halfspace_recomputed": 0,
            "frozen_5nm_recomputed": 0,
            "epsilon_m": 0.0,
            "plane_equation_mode": (
                "unrounded normalized Float64 equations; rounded 12-decimal "
                "key used only for deduplication"
            ),
            "row_uniqueness": "exact np.unique Float64 rows; no rounding merge",
            "lower_dimensional_continuation": False,
            "qj_or_random_jitter": False,
            "first_collapse_claim": (
                "independent reconstruction order; not an intermediate trace "
                "stored by D389 and not a geometry-invariant ordering"
            ),
        },
        "rank_contract": {
            "classification": {
                "EMPTY": "zero unique points",
                "POINT": "affine rank 0",
                "LINE": "affine rank 1",
                "FACE_LIKE": "affine rank 2; not a contact-manifold claim",
                "FULL_DIMENSIONAL": "affine rank 3",
            },
            "threshold_formula": (
                "sigma_max * max(unique_point_count,3) * float64_epsilon"
            ),
            "float64_epsilon": FLOAT64_EPSILON,
            "role": (
                "diagnostic rank semantics only; not overlap epsilon, 5nm, "
                "tolerance, gate, or physics setting"
            ),
        },
        "d389_failed_call_manifest": manifest,
        "d389_csv_lineage": csv_lineage,
        "synthetic_controls": controls,
        "aggregate": {
            "affine_class_counts": class_counts,
            "strict_relation_counts": strict_counts,
            "semantic_relation_counts": semantic_counts,
            "localized_call_count": len(records),
            "all_classes_sum_to_41": sum(class_counts.values())
            == EXPECTED_FAILED_CALL_COUNT,
        },
        "failed_call_records": records,
        "checks": checks,
        "localization_contract_pass": localization_pass,
        "verdict": verdict,
        "d389_repaired_or_recomputed": False,
        "d389_retroactive_pass": False,
        "directional_contract_repaired": False,
        "all_36_pairs_reaudited": False,
        "selected_vertex_budget": None,
        "adopted_vertex_budget": None,
        "selected_budget_application_count": 0,
        "materializable_candidate": False,
        "live_identity_pass": None,
        "physics_or_grasp_result": None,
        "g0a_pass": False,
        "scope_counters": SCOPE_COUNTERS,
    }
    geometry = {
        "artifact": "D390_TERMINAL_CANDIDATE_GEOMETRY_V1",
        "authority": (
            "canonical Float64 diagnostic geometry; Rerun spatial copies are "
            "inspection-only"
        ),
        "input_geometry_sha256": EXPECTED_INPUT_SHA256["d389_geometry"],
        "ordered_call_manifest_sha256": manifest[
            "ordered_call_manifest_sha256"
        ],
        "record_count": len(geometry_records),
        "records": geometry_records,
    }
    return evidence, geometry, records


def _write_trace_csv(records: list[dict[str, Any]]) -> None:
    fields = [
        "call_index",
        "call_id",
        "target",
        "left_index",
        "right_index",
        "adjacent",
        "stage",
        "direction",
        "recorded_error_family",
        "recorded_clip_count",
        "qhull_fallback_used",
        "collapse_plane_index",
        "active_clip_ordinal",
        "failure_location",
        "unique_point_count",
        "affine_rank",
        "affine_class",
        "sigma0_m",
        "sigma1_m",
        "sigma2_m",
        "sigma1_over_sigma0",
        "sigma2_over_sigma0",
        "rank_threshold_m",
        "strict_classification",
        "strict_signed_inradius_nm",
        "semantic_relation",
        "record_pass",
    ]
    with TRACE_CSV.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            trace = row["independent_reconstruction"]
            candidate = trace["candidate"]
            singular = candidate["singular_values_m"]
            ratios = candidate["singular_value_ratios"]
            strict = row["stored_strict_halfspace_relation"]
            writer.writerow(
                {
                    "call_index": row["call_index"],
                    "call_id": row["call_id"],
                    "target": row["target"],
                    "left_index": row["left_index"],
                    "right_index": row["right_index"],
                    "adjacent": row["adjacent"],
                    "stage": row["stage"],
                    "direction": row["direction"],
                    "recorded_error_family": row["recorded_failure"][
                        "error_family"
                    ],
                    "recorded_clip_count": row["recorded_failure"][
                        "clip_count"
                    ],
                    "qhull_fallback_used": row["recorded_failure"][
                        "qhull_fallback_used"
                    ],
                    "collapse_plane_index": trace[
                        "selected_plane_index_zero_based"
                    ],
                    "active_clip_ordinal": trace[
                        "active_clip_ordinal_one_based"
                    ],
                    "failure_location": trace["failure_location"],
                    "unique_point_count": candidate["unique_point_count"],
                    "affine_rank": candidate["affine_rank"],
                    "affine_class": candidate["affine_class"],
                    "sigma0_m": singular[0],
                    "sigma1_m": singular[1],
                    "sigma2_m": singular[2],
                    "sigma1_over_sigma0": ratios[1],
                    "sigma2_over_sigma0": ratios[2],
                    "rank_threshold_m": candidate["rank_threshold_m"],
                    "strict_classification": strict["classification"],
                    "strict_signed_inradius_nm": strict.get(
                        "signed_inradius_nm"
                    ),
                    "semantic_relation": row["semantic_relation"],
                    "record_pass": row["pass"],
                }
            )


def _short_class(value: str) -> str:
    return {
        "EMPTY": "빈집합",
        "POINT": "점",
        "LINE": "선",
        "FACE_LIKE": "면형",
        "FULL_DIMENSIONAL": "3차원",
        "NUMERIC_INDETERMINATE": "미결",
    }.get(value, value)


def _short_strict(value: str) -> str:
    return {
        "STRICT_NONPOSITIVE_OR_SUBTHRESHOLD_WITHIN_SOLVER_BAND": "비양성밴드",
        "STRICT_INFEASIBLE_GAP": "분리",
        "STRICT_FULL_DIMENSIONAL_POSITIVE": "양의부피",
    }.get(value, "미결")


def _render_board(
    evidence: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(str(FONT_BOLD), 43)
    subtitle = ImageFont.truetype(str(FONT), 24)
    panel_title = ImageFont.truetype(str(FONT_BOLD), 25)
    regular = ImageFont.truetype(str(FONT), 20)
    compact = ImageFont.truetype(str(FONT), 18)
    compact_bold = ImageFont.truetype(str(FONT_BOLD), 18)
    footer_font = ImageFont.truetype(str(FONT), 15)
    title_text = "D390 방향성 교차 계산의 경계 차원 국소화"
    subtitle_text = (
        "오프라인 수치 진단 — 충돌체·분할·예산·물리·파지 채택이 아닙니다"
    )
    draw.text(
        (45, 25),
        title_text,
        font=title,
        fill="#111827",
    )
    draw.text(
        (48, 82),
        subtitle_text,
        font=subtitle,
        fill="#475569",
    )
    panels = [
        (40, 125, 600, 340),
        (620, 125, 1225, 340),
        (1245, 125, 1880, 340),
        (40, 365, 1880, 1005),
    ]
    for box in panels:
        draw.rounded_rectangle(
            box, radius=16, outline="#cbd5e1", width=2, fill="#f8fafc"
        )
    manifest = evidence["d389_failed_call_manifest"]
    aggregate = evidence["aggregate"]
    draw.text((65, 145), "1. D389에서 물려받은 분모", font=panel_title, fill="#0f172a")
    inherited_lines = [
        (
            f"전체 방향 호출 {manifest['total_directional_call_count']} = "
            f"성공 {manifest['successful_call_count']} + 실패 "
            f"{manifest['failed_call_count']}"
        ),
        f"영향 쌍 {manifest['affected_pair_count']}/36",
        (
            f"전/후 Float32 실패 "
            f"{manifest['stage_breakdown']['pre_float32']}/"
            f"{manifest['stage_breakdown']['post_float32']}"
        ),
        (
            "오류: 점<4 "
            f"{manifest['failure_family_breakdown']['FEWER_THAN_FOUR_UNIQUE_POINTS']}, "
            "rank<3 "
            f"{manifest['failure_family_breakdown']['AFFINE_RANK_LT_3']}, "
            "QH6154 "
            f"{manifest['failure_family_breakdown']['QH6154_FLAT_OR_COPLANAR']}"
        ),
    ]
    y = 190
    for line in inherited_lines:
        draw.text((70, y), line, font=regular, fill="#1f2937")
        y += 34
    draw.text((645, 145), "2. 이번 분류의 뜻", font=panel_title, fill="#0f172a")
    legend = [
        ("빈집합", "고유점 0"),
        ("점", "affine rank 0"),
        ("선", "affine rank 1"),
        ("면형", "affine rank 2 — 실제 접촉면 단정 아님"),
        ("3차원", "affine rank 3 — Qhull 실패는 별도"),
    ]
    y = 188
    for name, meaning in legend:
        draw.text((650, y), f"{name:<5}  {meaning}", font=regular, fill="#1f2937")
        y += 29
    draw.text(
        (650, 310),
        "색·차원은 파지 성공/실패를 뜻하지 않습니다.",
        font=compact_bold,
        fill="#7c2d12",
    )
    draw.text((1270, 145), "3. D390 집계", font=panel_title, fill="#0f172a")
    class_order = ["EMPTY", "POINT", "LINE", "FACE_LIKE", "FULL_DIMENSIONAL"]
    class_text = " · ".join(
        f"{_short_class(name)} {aggregate['affine_class_counts'].get(name, 0)}"
        for name in class_order
    )
    draw.text((1275, 192), class_text, font=regular, fill="#1f2937")
    strict_text = " · ".join(
        f"{_short_strict(name)} {count}"
        for name, count in sorted(
            aggregate["strict_relation_counts"].items()
        )
    )
    draw.text((1275, 235), strict_text, font=regular, fill="#1f2937")
    draw.text(
        (1275, 278),
        f"분류 합계 {sum(aggregate['affine_class_counts'].values())}/41 · "
        f"계약 {'PASS' if evidence['localization_contract_pass'] else 'FAIL'}",
        font=regular,
        fill="#334155",
    )
    draw.text(
        (65, 385),
        "4. 실패 호출 전수 목록 (왼쪽 21개 + 오른쪽 20개)",
        font=panel_title,
        fill="#0f172a",
    )
    draw.text(
        (1235, 391),
        "E: V=점 수/차원 ValueError, Q=QH6154 정밀도 오류",
        font=compact_bold,
        fill="#475569",
    )
    roster_header = (
        "ID | A/N | 전/후 | 방향 | 고유점 n | rank | 차원 | "
        "σ2/σ1 | σ3/σ1 | strict | E"
    )
    draw.text(
        (65, 423),
        roster_header,
        font=compact_bold,
        fill="#334155",
    )
    draw.text(
        (985, 423),
        roster_header,
        font=compact_bold,
        fill="#334155",
    )
    roster_y0 = 455
    line_height = 26
    column_x = [65, 985]
    row_boxes: list[tuple[int, int, int, int, int]] = []
    displayed_ids: list[str] = []
    for index, row in enumerate(records):
        column = 0 if index < 21 else 1
        local_index = index if column == 0 else index - 21
        x = column_x[column]
        y = roster_y0 + local_index * line_height
        trace = row["independent_reconstruction"]
        candidate = trace["candidate"]
        ratios = candidate["singular_value_ratios"]
        stage = "전" if row["stage"] == "pre_float32" else "후"
        direction = "L←R" if row["direction"] == "left_clipped_by_right" else "R←L"
        adjacency = "A" if row["adjacent"] else "N"
        error_badge = (
            "Q"
            if row["recorded_failure"]["error_family"]
            == "QH6154_FLAT_OR_COPLANAR"
            else "V"
        )
        text = (
            f"{index:02d} {row['call_id'][:31]:31} | {adjacency} {stage} "
            f"{direction:3} | n={candidate['unique_point_count']:2d} "
            f"r={candidate['affine_rank']} {_short_class(candidate['affine_class']):3} "
            f"| {ratios[1]:.1e} {ratios[2]:.1e} "
            f"| {_short_strict(row['stored_strict_halfspace_relation']['classification']):4} "
            f"| {error_badge}"
        )
        draw.text((x, y), text, font=compact, fill="#111827")
        bbox = draw.textbbox((x, y), text, font=compact)
        row_boxes.append((*bbox, column))
        displayed_ids.append(row["call_id"])
    footer = (
        f"41/41 표시 · manifest {manifest['ordered_call_manifest_sha256'][:12]}… · "
        "selected/adopted budget=null · materializable=false · physics/grasp=null · "
        "g0a_pass=false · Rerun 공간복사본은 육안검사용"
    )
    footer_position = (55, 1030)
    draw.text(footer_position, footer, font=footer_font, fill="#475569")
    image.save(BOARD)
    left_rows = [box for box in row_boxes if box[4] == 0]
    right_rows = [box for box in row_boxes if box[4] == 1]
    row_nonoverlap = all(
        rows[index][3] <= rows[index + 1][1]
        for rows in (left_rows, right_rows)
        for index in range(len(rows) - 1)
    )
    within_columns = all(
        box[0] >= column_x[box[4]]
        and box[2] <= (955 if box[4] == 0 else 1870)
        and box[1] >= roster_y0
        and box[3] <= 1005
        for box in row_boxes
    )
    fixed_text_boxes = [
        draw.textbbox((45, 25), title_text, font=title),
        draw.textbbox((48, 82), subtitle_text, font=subtitle),
        draw.textbbox(
            (65, 385),
            "4. 실패 호출 전수 목록 (왼쪽 21개 + 오른쪽 20개)",
            font=panel_title,
        ),
        draw.textbbox((65, 423), roster_header, font=compact_bold),
        draw.textbbox((985, 423), roster_header, font=compact_bold),
        draw.textbbox(footer_position, footer, font=footer_font),
    ]
    fixed_text_inside_canvas = all(
        0 <= box[0] < box[2] <= 1920
        and 0 <= box[1] < box[3] <= 1080
        for box in fixed_text_boxes
    )
    footer_tokens = [
        "selected/adopted budget=null",
        "materializable=false",
        "physics/grasp=null",
        "g0a_pass=false",
    ]
    layout_checks = {
        "exact_canvas_1920x1080": image.size == (1920, 1080),
        "four_registered_regions_inside_canvas": all(
            0 <= value <= bound
            for panel in panels
            for value, bound in zip(
                panel, (1920, 1080, 1920, 1080), strict=True
            )
        ),
        "roster_exact_41_unique_ids": len(displayed_ids) == 41
        and len(set(displayed_ids)) == 41,
        "roster_order_exact_manifest": displayed_ids
        == manifest["ordered_call_ids"],
        "roster_split_exact_21_20": len(left_rows) == 21
        and len(right_rows) == 20,
        "roster_rows_do_not_overlap": row_nonoverlap,
        "roster_text_within_columns": within_columns,
        "registered_title_headers_footer_inside_canvas": (
            fixed_text_inside_canvas
        ),
        "class_counts_sum_41": sum(
            aggregate["affine_class_counts"].values()
        )
        == 41,
        "fixed_nonclaim_footer_tokens_present": all(
            token in footer for token in footer_tokens
        ),
    }
    layout = {
        "artifact": "D390_BOARD_LAYOUT_VALIDATION_V1",
        "path": _rel(BOARD),
        "width": 1920,
        "height": 1080,
        "registered_regions": 4,
        "roster_rows": len(displayed_ids),
        "roster_split": [len(left_rows), len(right_rows)],
        "minimum_roster_font_px": 18,
        "line_height_px": line_height,
        "checks": layout_checks,
        "pass": all(layout_checks.values()),
    }
    _write_json_x(BOARD_LAYOUT, layout)
    return {
        "path": _rel(BOARD),
        "sha256": _sha(BOARD),
        "bytes": BOARD.stat().st_size,
        "exact_1920x1080": True,
        "layout_validation": {
            "path": _rel(BOARD_LAYOUT),
            "sha256": _sha(BOARD_LAYOUT),
            "pass": layout["pass"],
        },
    }


def _hull_edge_arrows(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(source, qhull_options="Q12 Pp")
    edges: set[tuple[int, int]] = set()
    for triangle in hull.simplices:
        for left, right in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edges.add(tuple(sorted((int(left), int(right)))))
    ordered = sorted(edges)
    origins = np.asarray([source[left] for left, _ in ordered])
    vectors = np.asarray(
        [source[right] - source[left] for left, right in ordered]
    )
    return origins, vectors


def _plane_outline(
    equation: np.ndarray | None,
    reference: np.ndarray,
    extent: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    if equation is None:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty, empty, empty, {
            "plane_present": False,
            "arbitrary_plane_drawn": False,
            "maximum_corner_equation_residual": None,
            "pass": True,
        }
    local_equation = np.asarray(equation, dtype=np.float64)
    if local_equation.shape != (4,):
        raise RuntimeError("collapse plane equation must contain four values")
    raw_normal = local_equation[:3]
    normal_norm = float(np.linalg.norm(raw_normal))
    if not math.isfinite(normal_norm) or normal_norm <= 0.0:
        raise RuntimeError("collapse plane normal is invalid")
    normal = raw_normal / normal_norm
    reference = np.asarray(reference, dtype=np.float64)
    center = reference - (
        (
            float(np.dot(raw_normal, reference))
            + float(local_equation[3])
        )
        / float(np.dot(raw_normal, raw_normal))
    ) * raw_normal
    helper = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(float(normal[0])) < 0.8
        else np.asarray([0.0, 1.0, 0.0])
    )
    axis_u = np.cross(normal, helper)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    corners = np.asarray(
        [
            center - extent * axis_u - extent * axis_v,
            center + extent * axis_u - extent * axis_v,
            center + extent * axis_u + extent * axis_v,
            center - extent * axis_u + extent * axis_v,
        ]
    )
    origins = corners
    vectors = np.roll(corners, -1, axis=0) - corners
    normal_origin = center.reshape(1, 3)
    normal_vector = (normal * extent * 0.6).reshape(1, 3)
    residuals = corners @ raw_normal + float(local_equation[3])
    maximum_residual = float(np.max(np.abs(residuals)))
    metadata = {
        "plane_present": True,
        "arbitrary_plane_drawn": False,
        "local_equation_f64_m": local_equation,
        "projected_center_local_m": center,
        "maximum_corner_equation_residual": maximum_residual,
        "residual_gate": 1.0e-12,
        "pass": maximum_residual <= 1.0e-12,
    }
    return origins, vectors, normal_origin, normal_vector, metadata


def _build_blueprint() -> Any:
    import rerun as rr
    import rerun.blueprint as rrb

    spatial = rrb.Spatial3DView(
        origin="/",
        contents="/d390/geometry/**",
        name="D390 selected failed-call collapse geometry",
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=(0.06, -0.08, 0.055),
            look_target=(0.0, 0.0, 0.0),
            eye_up=(0.0, 0.0, 1.0),
        ),
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=True,
            show_bounding_box=False,
        ),
    )
    metrics = rrb.TimeSeriesView(
        origin="/metrics/d390",
        contents="/metrics/d390/**",
        name="Affine rank and singular ratios by failed_call_index",
    )
    text = rrb.TextLogView(
        origin="/events/d390",
        contents="/events/d390/**",
        name="Selected call and frozen nonclaims",
    )
    decision = rrb.Horizontal(
        spatial,
        rrb.Vertical(metrics, text, row_shares=[0.52, 0.48]),
        column_shares=[0.64, 0.36],
    )
    buffer = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d390/notification_buffer/**",
        name="Notification buffer - no decision subject",
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=False,
            show_bounding_box=False,
        ),
    )
    return rrb.Blueprint(
        rrb.Horizontal(decision, buffer, column_shares=[0.78, 0.22]),
        rrb.TimePanel(
            timeline="failed_call_index",
            play_state="Paused",
            time_selection=rr.datatypes.AbsoluteTimeRange(40, 40),
            state=rrb.PanelState.Collapsed,
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    if not path.is_file():
        return {"path": _rel(path), "exists": False}
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


def _write_rerun(
    evidence: dict[str, Any],
    geometry: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    import roarm_rl.rerun_contract as rerun_contract
    import roarm_rl.viz_debug as viz_debug

    geometry_by_id = {
        row["call_id"]: row for row in geometry["records"]
    }
    class_priority = {
        "FACE_LIKE": 0,
        "LINE": 1,
        "POINT": 2,
        "EMPTY": 3,
        "FULL_DIMENSIONAL": 4,
        "NUMERIC_INDETERMINATE": 5,
    }
    ranked_presentation = sorted(
        records,
        key=lambda row: (
            class_priority.get(
                row["independent_reconstruction"]["candidate"][
                    "affine_class"
                ],
                9,
            ),
            0 if row["adjacent"] else 1,
            row["call_id"],
        ),
    )
    representative = ranked_presentation[0]
    presentation_order = [
        *ranked_presentation[1:],
        representative,
    ]
    points: list[dict[str, Any]] = []
    arrows: list[dict[str, Any]] = []
    scalars: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    plane_display_checks: list[dict[str, Any]] = []
    for presentation_index, record in enumerate(presentation_order):
        row = geometry_by_id[record["call_id"]]
        source = np.asarray(row["source_vertices_f64_m"], dtype=np.float64)
        clipping = np.asarray(
            row["clipping_vertices_f64_m"], dtype=np.float64
        )
        terminal = np.asarray(
            row["terminal_candidate_unique_points_f64_m"],
            dtype=np.float64,
        ).reshape(-1, 3)
        combined = np.vstack([source, clipping])
        display_center = (
            terminal.mean(axis=0)
            if len(terminal)
            else combined.mean(axis=0)
        )
        source_local = source - display_center
        clipping_local = clipping - display_center
        terminal_local = terminal - display_center
        span = float(np.ptp(combined, axis=0).max())
        extent = max(0.004, min(0.035, span * 0.65))
        source_origins, source_vectors = _hull_edge_arrows(source_local)
        clipping_origins, clipping_vectors = _hull_edge_arrows(
            clipping_local
        )
        plane_equation = row["collapse_plane_equation_f64_m"]
        plane_equation_local: np.ndarray | None = None
        if plane_equation is not None:
            plane_equation_world = np.asarray(
                plane_equation, dtype=np.float64
            )
            plane_equation_local = plane_equation_world.copy()
            plane_equation_local[3] += float(
                np.dot(plane_equation_world[:3], display_center)
            )
        plane_center_local = (
            terminal_local.mean(axis=0)
            if len(terminal_local)
            else np.zeros(3, dtype=np.float64)
        )
        (
            plane_origins,
            plane_vectors,
            normal_origin,
            normal_vector,
            plane_display,
        ) = _plane_outline(
            plane_equation_local,
            plane_center_local,
            extent,
        )
        plane_display_checks.append(
            {
                "failed_call_index": presentation_index,
                "call_id": record["call_id"],
                "world_equation_present": plane_equation is not None,
                "display_center_f64_m": display_center,
                **plane_display,
            }
        )
        sequence = {"failed_call_index": presentation_index}
        points.extend(
            [
                {
                    "entity_path": "d390/geometry/source_vertices",
                    "positions_m": source_local,
                    "radii": [0.00065] * len(source_local),
                    "colors": [[13, 148, 136, 220]] * len(source_local),
                    "coordinate_frame": "tf#/",
                    "sequence": sequence,
                    "static": False,
                },
                {
                    "entity_path": "d390/geometry/clipping_vertices",
                    "positions_m": clipping_local,
                    "radii": [0.00065] * len(clipping_local),
                    "colors": [[245, 158, 11, 220]] * len(clipping_local),
                    "coordinate_frame": "tf#/",
                    "sequence": sequence,
                    "static": False,
                },
                {
                    "entity_path": "d390/geometry/terminal_points",
                    "positions_m": terminal_local,
                    "radii": [0.0012] * len(terminal_local),
                    "colors": [[220, 38, 38, 255]] * len(terminal_local),
                    "coordinate_frame": "tf#/",
                    "sequence": sequence,
                    "static": False,
                },
            ]
        )
        arrows.extend(
            [
                {
                    "entity_path": "d390/geometry/source_edges",
                    "origins_m": source_origins,
                    "vectors_m": source_vectors,
                    "radii": 0.00018,
                    "colors": [[13, 148, 136, 150]]
                    * len(source_origins),
                    "coordinate_frame": "tf#/",
                    "sequence": sequence,
                    "static": False,
                },
                {
                    "entity_path": "d390/geometry/clipping_edges",
                    "origins_m": clipping_origins,
                    "vectors_m": clipping_vectors,
                    "radii": 0.00018,
                    "colors": [[245, 158, 11, 150]]
                    * len(clipping_origins),
                    "coordinate_frame": "tf#/",
                    "sequence": sequence,
                    "static": False,
                },
                {
                    "entity_path": "d390/geometry/collapse_plane_outline",
                    "origins_m": plane_origins,
                    "vectors_m": plane_vectors,
                    "radii": 0.00022,
                    "colors": [[99, 102, 241, 180]]
                    * len(plane_origins),
                    "coordinate_frame": "tf#/",
                    "sequence": sequence,
                    "static": False,
                },
                {
                    "entity_path": "d390/geometry/collapse_plane_normal",
                    "origins_m": normal_origin,
                    "vectors_m": normal_vector,
                    "radii": 0.00032,
                    "colors": [[99, 102, 241, 255]]
                    * len(normal_origin),
                    "coordinate_frame": "tf#/",
                    "sequence": sequence,
                    "static": False,
                },
            ]
        )
        candidate = record["independent_reconstruction"]["candidate"]
        ratios = candidate["singular_value_ratios"]
        for path, value in (
            ("affine_rank", candidate["affine_rank"]),
            ("unique_point_count", candidate["unique_point_count"]),
            ("sigma1_over_sigma0", ratios[1]),
            ("sigma2_over_sigma0", ratios[2]),
        ):
            scalars.append(
                {
                    "entity_path": f"metrics/d390/{path}",
                    "value": float(value),
                    "sequence": sequence,
                    "static": False,
                }
            )
        events.append(
            {
                "entity_path": "events/d390/selected_call",
                "text": (
                    f"call={record['call_id']} | presentation "
                    f"{presentation_index + 1}/41 | "
                    f"class={candidate['affine_class']} | "
                    f"n={candidate['unique_point_count']} | "
                    f"rank={candidate['affine_rank']} | "
                    f"sigma2/sigma1={ratios[1]:.3e} | "
                    f"sigma3/sigma1={ratios[2]:.3e} | "
                    f"strict={record['stored_strict_halfspace_relation']['classification']} | "
                    f"plane={'shown' if plane_equation is not None else 'none'} | "
                    "colors: teal=source, orange=clipper, red=terminal, "
                    "purple=collapse plane | local display is translated "
                    "inspection copy; no scale change"
                ),
                "level": "INFO",
                "sequence": sequence,
                "static": False,
            }
        )
        events.append(
            {
                "entity_path": "events/d390/nonclaims",
                "text": (
                    "D389 remains FAIL_STOP; directional contract not repaired; "
                    "budget/partition/asset/Isaac/physics/q5/contact/grasp=0; "
                    "selected/adopted budget=NULL; g0a_pass=false"
                ),
                "level": "WARN",
                "sequence": sequence,
                "static": False,
            }
        )
    original_builder = viz_debug.build_rerun_blueprint
    original_runner = rerun_contract._run

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d390_boundary_collapse":
            return _build_blueprint()
        return original_builder(mode)

    viewer_calls = 0

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
                    "stderr": "D390 Viewer maximum one exceeded",
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
            points=points,
            arrows=arrows,
            scalar_trace=scalars,
            events=events,
            recording_metadata={
                "case": CASE,
                "attempt": ATTEMPT,
                "verdict": evidence["verdict"],
                "timeline_role": (
                    "failed_call_index is catalog order, not physics time"
                ),
                "ordered_call_manifest_sha256": evidence[
                    "d389_failed_call_manifest"
                ]["ordered_call_manifest_sha256"],
                "presentation_order_call_ids": [
                    row["call_id"] for row in presentation_order
                ],
                "default_last_representative_call_id": representative[
                    "call_id"
                ],
                "canonical_evidence_sha256": _sha(EVIDENCE),
                "canonical_trace_geometry_sha256": _sha(TRACE_GEOMETRY),
                "canonical_trace_csv_sha256": _sha(TRACE_CSV),
                "scientific_authority": (
                    "canonical Float64 JSON/CSV; Rerun Float32 copy is "
                    "inspection-only"
                ),
                "selected_vertex_budget": None,
                "adopted_vertex_budget": None,
                "physics_or_grasp_result": None,
                "g0a_pass": False,
            },
            recording_id="g0a_d390_boundary_collapse_localization",
            blueprint_path=RBL,
            blueprint_mode="d390_boundary_collapse",
            live_viewer=False,
            app_id="roarm_g0a_d390_boundary_collapse_localization",
        )
        if not saved.get("ok"):
            raise RuntimeError(f"D390 save-only Rerun failed: {saved}")
        expected_entities = sorted(
            {
                "metadata/run",
                "d390/geometry/source_vertices",
                "d390/geometry/clipping_vertices",
                "d390/geometry/terminal_points",
                "d390/geometry/source_edges",
                "d390/geometry/clipping_edges",
                "d390/geometry/collapse_plane_outline",
                "d390/geometry/collapse_plane_normal",
                "metrics/d390/affine_rank",
                "metrics/d390/unique_point_count",
                "metrics/d390/sigma1_over_sigma0",
                "metrics/d390/sigma2_over_sigma0",
                "events/d390/selected_call",
                "events/d390/nonclaims",
            }
        )
        component_contract = {
            "metadata/run": ["TextDocument:text"],
            "d390/geometry/source_vertices": [
                "CoordinateFrame:frame",
                "Points3D:colors",
                "Points3D:positions",
                "Points3D:radii",
            ],
            "d390/geometry/clipping_vertices": [
                "CoordinateFrame:frame",
                "Points3D:colors",
                "Points3D:positions",
                "Points3D:radii",
            ],
            "d390/geometry/terminal_points": [
                "CoordinateFrame:frame",
                "Points3D:colors",
                "Points3D:positions",
                "Points3D:radii",
            ],
            "d390/geometry/source_edges": [
                "Arrows3D:colors",
                "Arrows3D:origins",
                "Arrows3D:radii",
                "Arrows3D:vectors",
                "CoordinateFrame:frame",
            ],
            "d390/geometry/clipping_edges": [
                "Arrows3D:colors",
                "Arrows3D:origins",
                "Arrows3D:radii",
                "Arrows3D:vectors",
                "CoordinateFrame:frame",
            ],
            "d390/geometry/collapse_plane_outline": [
                "Arrows3D:colors",
                "Arrows3D:origins",
                "Arrows3D:radii",
                "Arrows3D:vectors",
                "CoordinateFrame:frame",
            ],
            "d390/geometry/collapse_plane_normal": [
                "Arrows3D:colors",
                "Arrows3D:origins",
                "Arrows3D:radii",
                "Arrows3D:vectors",
                "CoordinateFrame:frame",
            ],
            "metrics/d390/affine_rank": ["Scalars:scalars"],
            "metrics/d390/unique_point_count": ["Scalars:scalars"],
            "metrics/d390/sigma1_over_sigma0": ["Scalars:scalars"],
            "metrics/d390/sigma2_over_sigma0": ["Scalars:scalars"],
            "events/d390/selected_call": [
                "TextLog:level",
                "TextLog:text",
            ],
            "events/d390/nonclaims": [
                "TextLog:level",
                "TextLog:text",
            ],
        }
        validation = rerun_contract.validate_rerun_artifact(
            RRD,
            expected_entity_paths=expected_entities,
            exact_entity_paths=expected_entities,
            expected_timeline_names=[
                "blueprint",
                "failed_call_index",
                "log_time",
            ],
            exact_timeline_names=[
                "blueprint",
                "failed_call_index",
                "log_time",
            ],
            expected_entity_components=component_contract,
            blueprint_path=RBL,
            screenshot_path=RERUN_SCREENSHOT,
            screenshot_window_size="1920x1080",
            screenshot_port="auto",
            cli_path=RERUN_CLI,
            expected_version="0.34.1",
            timeout_s=0.0,
        )
        timeline_probe_raw = no_signal_runner(
            [
                str(RERUN_CLI),
                "rrd",
                "print",
                "-vvv",
                "--entity",
                "/metrics/d390/affine_rank",
                str(RRD),
            ],
            timeout_s=0.0,
        )
    finally:
        rerun_contract._run = original_runner
        viz_debug.build_rerun_blueprint = original_builder
        os.environ["PATH"] = old_path
    timeline_row_lines = [
        line
        for line in timeline_probe_raw.get("stdout", "").splitlines()
        if "│ │ row_" in line
    ]
    timeline_values: list[int] = []
    timeline_parse_error: str | None = None
    try:
        timeline_values = [
            int(line.split("┆")[2].strip()) for line in timeline_row_lines
        ]
    except Exception as exc:
        timeline_parse_error = f"{type(exc).__name__}: {exc}"
    timeline_data_contract = {
        "entity": "/metrics/d390/affine_rank",
        "command": timeline_probe_raw.get("command"),
        "returncode": timeline_probe_raw.get("returncode"),
        "stderr": timeline_probe_raw.get("stderr"),
        "stdout_sha256": _text_sha(timeline_probe_raw.get("stdout", "")),
        "row_count": len(timeline_row_lines),
        "failed_call_index_values": timeline_values,
        "parse_error": timeline_parse_error,
        "exact_0_through_40": timeline_values == list(range(41)),
        "pass": (
            timeline_probe_raw.get("ok") is True
            and timeline_parse_error is None
            and len(timeline_row_lines) == 41
            and timeline_values == list(range(41))
        ),
    }
    validation["d390_execution_contract"] = {
        "headless_viewer_invocations": viewer_calls,
        "viewer_maximum": 1,
        "viewer_retry": 0,
        "process_signals_sent": 0,
        "subprocess_timeout_seconds": None,
        "presentation_order_call_ids": [
            row["call_id"] for row in presentation_order
        ],
        "default_last_representative_call_id": representative["call_id"],
        "collapse_plane_display_checks": plane_display_checks,
        "collapse_plane_display_contract_pass": all(
            row["pass"] for row in plane_display_checks
        ),
        "failed_call_index_data_contract": timeline_data_contract,
    }
    validation["base_rerun_contract_pass"] = (
        validation.get("pass") is True
    )
    validation["pass"] = (
        validation["base_rerun_contract_pass"]
        and validation["d390_execution_contract"][
            "collapse_plane_display_contract_pass"
        ]
        and validation["d390_execution_contract"][
            "failed_call_index_data_contract"
        ]["pass"]
    )
    _write_json_x(RERUN_VALIDATION, validation)
    screenshot = _png_info(RERUN_SCREENSHOT)
    dimension_pass = (
        screenshot.get("width") in {1920, 3840}
        and screenshot.get("height") in {1080, 2160}
        and screenshot.get("width") == 16 * screenshot.get("height") // 9
    )
    strict_pass = validation.get("pass") is True
    return {
        "strict_validation_pass": strict_pass,
        "viewer_maximum_one_no_retry": viewer_calls == 1,
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
        "timeline_steps": len(presentation_order),
        "presentation_order_call_ids": [
            row["call_id"] for row in presentation_order
        ],
    }


def _prepare() -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"refusing existing D390 output: {OUT_DIR}")
    status_before = _status_lines()
    imports = _direct_import_roots()
    forbidden = sorted(
        set(imports)
        & {
            "isaaclab",
            "omni",
            "pxr",
            "warp",
            "torch",
            "roarm_sdk",
            "serial",
        }
    )
    checks = {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_master_exact": _git("rev-parse", "origin/master")
        == EXPECTED_HEAD,
        "start_here_authorization_hash_exact": _sha(START)
        == EXPECTED_START_SHA256,
        "immutable_input_hashes_exact": _input_hashes()
        == EXPECTED_INPUT_SHA256,
        "d390_output_absent_before_prepare": not OUT_DIR.exists(),
        "preexisting_dirty_scope_only_d389_state_and_d390_registration": (
            bool(status_before)
            and _status_scope_pass(
                status_before, allow_d390_output=False
            )
        ),
        "no_forbidden_direct_import_roots": not forbidden,
        "isaaclab_pins_preserved": importlib.metadata.version("numpy")
        == "1.26.0"
        and importlib.metadata.version("psutil") == "5.9.8",
        "rerun_sdk_pin_exact": importlib.metadata.version("rerun-sdk")
        == "0.34.1",
        "font_files_exist": FONT.is_file() and FONT_BOLD.is_file(),
        "rerun_cli_exists": RERUN_CLI.is_file(),
        "scope_counters_zero": all(
            value == 0 for value in SCOPE_COUNTERS.values()
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    prereg = {
        "artifact": "D390_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "approved_scope": (
            "offline-only localization of terminal affine rank for the 41 "
            "D389-recorded failed directional epsilon-zero calls"
        ),
        "new_variables": NEW_VARIABLES,
        "authorization": {
            "path": _rel(START),
            "sha256": EXPECTED_START_SHA256,
        },
        "runtime_inputs": {
            "d389_evidence": {
                "path": _rel(D389_EVIDENCE),
                "sha256": EXPECTED_INPUT_SHA256["d389_evidence"],
            },
            "d389_geometry": {
                "path": _rel(D389_GEOMETRY),
                "sha256": EXPECTED_INPUT_SHA256["d389_geometry"],
            },
            "d389_csv": {
                "path": _rel(D389_CSV),
                "sha256": EXPECTED_INPUT_SHA256["d389_csv"],
            },
        },
        "method_contract": {
            "independent_reconstruction_not_d389_import_or_execution": True,
            "failed_calls_only": 41,
            "epsilon_m": 0.0,
            "plane_semantics": (
                "unrounded normalized Float64 equations; rounded 12-decimal "
                "dedupe key only"
            ),
            "exact_float64_row_uniqueness": True,
            "rank_threshold": (
                "sigma_max * max(unique_point_count,3) * float64_epsilon"
            ),
            "classification": [
                "EMPTY",
                "POINT",
                "LINE",
                "FACE_LIKE",
                "FULL_DIMENSIONAL",
            ],
            "stop_at_first_collapse_no_lower_dimensional_continuation": True,
            "first_collapse_ordinal_is_independent_implementation_provenance": True,
            "first_collapse_not_claimed_as_d389_stored_intermediate_trace": True,
            "qj_random_jitter": False,
            "epsilon_5nm_tolerance_gate_change": False,
        },
        "registered_denominators": {
            "pairs": EXPECTED_PAIR_COUNT,
            "directional_calls": EXPECTED_CALL_COUNT,
            "failed_calls": EXPECTED_FAILED_CALL_COUNT,
            "successful_calls_not_recomputed": EXPECTED_SUCCESS_CALL_COUNT,
            "affected_pairs": EXPECTED_AFFECTED_PAIR_COUNT,
            "failure_breakdown": EXPECTED_FAILURE_BREAKDOWN,
            "stage_breakdown": EXPECTED_STAGE_BREAKDOWN,
            "direction_breakdown": EXPECTED_DIRECTION_BREAKDOWN,
            "adjacency_breakdown": EXPECTED_ADJACENCY_BREAKDOWN,
        },
        "registered_pass_gates": [
            "input hashes exact",
            "36 pairs / 144 calls / 41 failures / 103 successes / 26 affected exact",
            "all 41 failed-call keys unique and trace error-family/clip/skip/fallback exact",
            "mean, NumPy matrix_rank, and first-point-anchor ranks agree",
            "stored strict result bound for every call without recomputation",
            "five class, five single-plane clip, and rank-threshold controls pass",
            "class and strict aggregates each sum to 41",
            "canonical JSON/geometry/CSV precede presentation",
            "exact 1920x1080 board and strict save-only RRD/RBL contract pass",
            "scope counters remain zero",
        ],
        "registered_verdicts": {
            "pass": (
                "D390_DIRECTIONAL_EPSILON0_TERMINAL_AFFINE_RANK_"
                "LOCALIZATION_PASS_NO_D389_REPAIR"
            ),
            "numeric_fail": (
                "D390_TERMINAL_CLASSIFICATION_OR_TRACE_IDENTITY_FAIL_STOP"
            ),
            "operational_fail": (
                "D390_OFFLINE_WORKER_OR_OBSERVABILITY_INTEGRITY_FAIL_STOP"
            ),
        },
        "execution": {
            "actual_worker_maximum": 1,
            "retry": 0,
            "cooperative_compute_and_presentation_check_deadline_seconds": (
                DEADLINE_SECONDS
            ),
            "hard_wall_clock_watchdog_seconds": None,
            "supervisor_wait_is_bounded": False,
            "no_hard_watchdog_reason": (
                "no supervisor process-signal authority was approved"
            ),
            "supervisor_signal_authority": False,
            "process_signals": 0,
            "save_only_rrd": True,
            "headless_viewer_maximum": 1,
            "viewer_retry": 0,
        },
        "forward_only_output_inventory_contract": {
            "after_prepare": sorted(PREPARE_INVENTORY),
            "before_worker_start_sentinel": sorted(PRE_WORKER_INVENTORY),
            "at_worker_start": sorted(WORKER_START_INVENTORY),
            "before_success_claim": sorted(PRE_CLAIM_SUCCESS_INVENTORY),
            "after_success_worker": sorted(
                POST_WORKER_SUCCESS_INVENTORY
            ),
            "before_finalize": sorted(PRE_FINALIZE_INVENTORY),
            "after_finalize": sorted(POST_FINALIZE_INVENTORY),
        },
        "frozen_nonclaims": {
            "d389_repaired_or_recomputed": False,
            "d389_retroactive_pass": False,
            "directional_contract_repaired": False,
            "all_36_pairs_reaudited": False,
            "partition_or_budget_or_geometry_change": False,
            "asset_or_collider_or_usd_materialization": False,
            "isaac_or_physx_or_cylinder_or_physics_or_q5_or_contact_or_grasp": False,
            "selected_or_adopted_budget": None,
            "materializable_candidate": False,
            "physics_or_grasp_result": None,
            "g0a_pass": False,
        },
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "status_before_output_create": status_before,
            "status_before_sha256": _text_sha(
                "\n".join(status_before) + "\n"
            ),
        },
        "script": {
            "path": _rel(SCRIPT),
            "sha256": _sha(SCRIPT),
            "direct_import_roots": imports,
            "forbidden_direct_import_roots": forbidden,
        },
        "environment": {
            "python": sys.version,
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "psutil": importlib.metadata.version("psutil"),
            "rerun_sdk": importlib.metadata.version("rerun-sdk"),
        },
        "scope_counters": SCOPE_COUNTERS,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG, prereg)
    _phase(
        "prepare_end",
        pass_value=prereg["pass"],
        preregistration_sha256=_sha(PREREG),
    )
    _require_out_inventory(PREPARE_INVENTORY, "after_prepare")
    if not prereg["pass"]:
        raise RuntimeError(f"D390 preregistration failed: {checks}")
    print(
        json.dumps(
            {"prepare_pass": True, "preregistration": _rel(PREREG)}
        )
    )
    return 0


def _worker_inner() -> int:
    global _deadline_monotonic
    prereg = _read_json(PREREG)
    if prereg.get("pass") is not True:
        raise RuntimeError("D390 preregistration is not PASS")
    authority_checks = _frozen_authority_checks(
        prereg, allow_d390_output=True
    )
    if not all(authority_checks.values()):
        raise RuntimeError(
            f"D390 frozen authority changed: {authority_checks}"
        )
    _require_out_inventory(PRE_WORKER_INVENTORY, "worker_before_sentinel")
    invocation, authorization, authorization_checks = (
        _worker_authorization_checks()
    )
    if not all(authorization_checks.values()):
        raise RuntimeError(
            f"D390 worker authorization failed: {authorization_checks}"
        )
    _write_json_x(
        WORKER_SENTINEL,
        {
            "artifact": "D390_WORKER_START_SENTINEL_V1",
            "worker_pid": os.getpid(),
            "parent_supervisor_pid": os.getppid(),
            "worker_invocation_index": 1,
            "retry_index": 0,
            "retries": 0,
            "invocation_sha256": _sha(INVOCATION),
            "authorization_sha256": _sha(WORKER_AUTHORIZATION),
            "preregistration_sha256": _sha(PREREG),
            "script_sha256": _sha(SCRIPT),
            "input_hashes": _input_hashes(),
            "start_here_sha256": _sha(START),
            "authorization_checks": authorization_checks,
            "pass": True,
        },
    )
    _require_out_inventory(WORKER_START_INVENTORY, "worker_start")
    started = time.monotonic()
    _deadline_monotonic = started + DEADLINE_SECONDS
    _phase(
        "worker_start",
        signal_authority=False,
        invocation_sha256=_sha(INVOCATION),
        authorization_sha256=_sha(WORKER_AUTHORIZATION),
        sentinel_sha256=_sha(WORKER_SENTINEL),
    )
    evidence, geometry, records = _compute()
    algorithm_elapsed = time.monotonic() - started
    _deadline("worker_before_canonical_commit")
    evidence["script_sha256"] = _sha(SCRIPT)
    evidence["execution"] = {
        "actual_worker_invocations": 1,
        "worker_invocation_index": 1,
        "retry_index": 0,
        "retries": 0,
        "offline_only": True,
        "cooperative_compute_and_presentation_check_deadline_seconds": (
            DEADLINE_SECONDS
        ),
        "hard_wall_clock_watchdog_seconds": None,
        "numeric_algorithm_elapsed_seconds": algorithm_elapsed,
        "cooperative_deadline_exceeded": (
            algorithm_elapsed > DEADLINE_SECONDS
        ),
        "process_signals_sent": 0,
    }
    _write_json_x(EVIDENCE, evidence)
    _write_json_x(TRACE_GEOMETRY, geometry)
    _write_trace_csv(records)
    _phase(
        "canonical_numeric_evidence_committed",
        verdict=evidence["verdict"],
        evidence_sha256=_sha(EVIDENCE),
        geometry_sha256=_sha(TRACE_GEOMETRY),
        csv_sha256=_sha(TRACE_CSV),
    )
    _deadline("worker_before_board")
    board = _render_board(evidence, records)
    _deadline("worker_after_board")
    _deadline("worker_before_rerun")
    rerun = _write_rerun(evidence, geometry, records)
    _deadline("worker_after_rerun")
    manual_template = {
        "artifact": "D390_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "required_checks": [
            "board_exact_1920x1080_and_title_footer_readable",
            "board_all_41_rows_visible_in_21_20_split",
            "board_no_text_overlap_or_clipping",
            "board_dimension_legend_not_confused_with_grasp_verdict",
            "rerun_source_clipping_terminal_and_plane_subject_visible",
            "rerun_selected_call_metrics_and_nonclaims_readable",
            "rerun_failed_call_index_timeline_paused_at_40",
            "rerun_color_legend_matches_teal_orange_red_purple_geometry",
            "rerun_no_decision_subject_obscured_by_warning_or_notification",
            "no_budget_collider_physics_or_grasp_adoption_claim",
        ],
        "board_path": _rel(BOARD),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "inspection_result_path": _rel(MANUAL),
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)
    _require_out_inventory(
        PRE_CLAIM_SUCCESS_INVENTORY, "worker_before_success_claim"
    )
    worker_elapsed = time.monotonic() - started
    worker_checks = {
        "numeric_localization_contract_pass": evidence[
            "localization_contract_pass"
        ],
        "cooperative_deadline_not_exceeded": (
            worker_elapsed <= DEADLINE_SECONDS
        ),
        "board_layout_pass": _read_json(BOARD_LAYOUT)["pass"] is True,
        "strict_rerun_validation_pass": rerun[
            "strict_validation_pass"
        ],
        "viewer_maximum_one_no_retry": rerun[
            "viewer_maximum_one_no_retry"
        ],
        "screenshot_dimension_contract_pass": rerun[
            "screenshot_dimension_contract_pass"
        ],
        "scope_counters_zero": all(
            value == 0 for value in SCOPE_COUNTERS.values()
        ),
        "worker_authorization_exact": all(
            authorization_checks.values()
        ),
        "frozen_authority_exact": all(authority_checks.values()),
    }
    claim = {
        "artifact": "D390_OFFLINE_WORKER_CLAIM_V1",
        "actual_worker_invocations": 1,
        "worker_invocation_index": 1,
        "retry_index": 0,
        "retries": 0,
        "numeric_verdict": evidence["verdict"],
        "worker_elapsed_seconds": worker_elapsed,
        "cooperative_compute_and_presentation_check_deadline_seconds": (
            DEADLINE_SECONDS
        ),
        "hard_wall_clock_watchdog_seconds": None,
        "invocation": {
            "path": _rel(INVOCATION),
            "sha256": _sha(INVOCATION),
        },
        "preregistration": {
            "path": _rel(PREREG),
            "sha256": _sha(PREREG),
        },
        "authorization": {
            "path": _rel(WORKER_AUTHORIZATION),
            "sha256": _sha(WORKER_AUTHORIZATION),
        },
        "worker_start_sentinel": {
            "path": _rel(WORKER_SENTINEL),
            "sha256": _sha(WORKER_SENTINEL),
        },
        "worker_authorization_checks": authorization_checks,
        "frozen_authority_checks": authority_checks,
        "checks": worker_checks,
        "artifacts": {
            "evidence": {
                "path": _rel(EVIDENCE),
                "sha256": _sha(EVIDENCE),
            },
            "geometry": {
                "path": _rel(TRACE_GEOMETRY),
                "sha256": _sha(TRACE_GEOMETRY),
            },
            "csv": {
                "path": _rel(TRACE_CSV),
                "sha256": _sha(TRACE_CSV),
            },
            "board": board,
            "rerun": rerun,
            "manual_template": {
                "path": _rel(MANUAL_TEMPLATE),
                "sha256": _sha(MANUAL_TEMPLATE),
            },
        },
        "scope_counters": SCOPE_COUNTERS,
        "pass": all(worker_checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_end", worker_claim_pass=claim["pass"])
    _require_out_inventory(
        POST_WORKER_SUCCESS_INVENTORY, "worker_after_success_claim"
    )
    if not claim["pass"]:
        raise RuntimeError(
            "D390 numeric or observability worker claim failed"
        )
    print(
        json.dumps(
            {
                "worker_pass": True,
                "verdict": evidence["verdict"],
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
            "artifact": "D390_FAILURE_ATTESTATION_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "stage": "worker",
            "actual_worker_invocations": 1,
            "retry_index": 0,
            "retries": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": elapsed,
            "cooperative_compute_and_presentation_check_deadline_seconds": (
                DEADLINE_SECONDS
            ),
            "hard_wall_clock_watchdog_seconds": None,
            "cooperative_deadline_exceeded": bool(
                _deadline_monotonic is not None
                and time.monotonic() > _deadline_monotonic
            ),
            "script": {
                "path": _rel(SCRIPT),
                "sha256": _sha(SCRIPT),
            },
            "preregistration": {
                "path": _rel(PREREG),
                "sha256": _sha(PREREG),
            },
            "input_hashes": _input_hashes(),
            "authorization": (
                {
                    "path": _rel(WORKER_AUTHORIZATION),
                    "sha256": _sha(WORKER_AUTHORIZATION),
                }
                if WORKER_AUTHORIZATION.is_file()
                else None
            ),
            "worker_start_sentinel": (
                {
                    "path": _rel(WORKER_SENTINEL),
                    "sha256": _sha(WORKER_SENTINEL),
                }
                if WORKER_SENTINEL.is_file()
                else None
            ),
            "scope_counters": SCOPE_COUNTERS,
            "process_signals_sent": 0,
            "d389_modified_repaired_or_recomputed": False,
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
                    "artifact": "D390_OFFLINE_WORKER_CLAIM_V1",
                    "actual_worker_invocations": 1,
                    "retry_index": 0,
                    "retries": 0,
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
            cooperative_deadline_exceeded=failure[
                "cooperative_deadline_exceeded"
            ],
        )
        raise


def _run() -> int:
    if not PREREG.is_file():
        raise RuntimeError("D390 prepare must precede run")
    _require_out_inventory(PREPARE_INVENTORY, "supervisor_before_invocation")
    if INVOCATION.exists() or SUPERVISOR.exists():
        raise RuntimeError("refusing a second D390 worker")
    prereg = _read_json(PREREG)
    authority_checks = _frozen_authority_checks(
        prereg, allow_d390_output=True
    )
    if not all(authority_checks.values()):
        raise RuntimeError(
            f"D390 supervisor frozen authority changed: {authority_checks}"
        )
    command = [sys.executable, "-B", str(SCRIPT), "--stage", "worker"]
    _write_json_x(
        INVOCATION,
        {
            "artifact": "D390_OFFLINE_LOCALIZATION_INVOCATION_V1",
            "command": command,
            "cwd": str(REPO),
            "actual_worker_maximum": 1,
            "worker_invocation_index": 1,
            "retry_index": 0,
            "retries": 0,
            "script_sha256": _sha(SCRIPT),
            "preregistration_sha256": _sha(PREREG),
            "input_hashes": _input_hashes(),
            "start_here_sha256": _sha(START),
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "prepare_status_sha256": prereg["git"][
                "status_before_sha256"
            ],
            "cooperative_compute_and_presentation_check_deadline_seconds": (
                DEADLINE_SECONDS
            ),
            "hard_wall_clock_watchdog_seconds": None,
            "supervisor_signal_authority": False,
            "process_signals_sent": 0,
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
                _write_json_x(
                    WORKER_AUTHORIZATION,
                    {
                        "artifact": "D390_WORKER_AUTHORIZATION_V1",
                        "supervisor_pid": os.getpid(),
                        "worker_invocation_index": 1,
                        "retry_index": 0,
                        "retries": 0,
                        "invocation_sha256": _sha(INVOCATION),
                        "preregistration_sha256": _sha(PREREG),
                        "script_sha256": _sha(SCRIPT),
                        "input_hashes": _input_hashes(),
                        "start_here_sha256": _sha(START),
                        "supervisor_signal_authority": False,
                        "process_signals_sent": 0,
                    },
                )
                _require_out_inventory(
                    PRE_WORKER_INVENTORY, "supervisor_before_popen"
                )
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
    claim_pass = False
    claim_error: str | None = None
    if WORKER_CLAIM.is_file():
        try:
            claim_pass = _read_json(WORKER_CLAIM).get("pass") is True
        except Exception as exc:
            claim_error = f"{type(exc).__name__}: {exc}"
    success_inventory_pass = (
        _out_names() == POST_WORKER_SUCCESS_INVENTORY
    )
    provenance = {
        label: (
            {
                "path": _rel(path),
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
            }
            if path.is_file()
            else None
        )
        for label, path in {
            "preregistration": PREREG,
            "invocation": INVOCATION,
            "authorization": WORKER_AUTHORIZATION,
            "worker_start_sentinel": WORKER_SENTINEL,
            "stdout": STDOUT,
            "stderr": STDERR,
            "worker_claim": WORKER_CLAIM,
            "failure_attestation": FAILURE_ATTESTATION,
        }.items()
    }
    record = {
        "artifact": "D390_OFFLINE_WORKER_SUPERVISOR_V1",
        "actual_worker_invocations": int(process is not None),
        "retries": 0,
        "worker_pid": process.pid if process is not None else None,
        "returncode": returncode,
        "supervisor_error": supervisor_error,
        "elapsed_seconds": time.monotonic() - started,
        "cooperative_compute_and_presentation_check_deadline_seconds": (
            DEADLINE_SECONDS
        ),
        "hard_wall_clock_watchdog_seconds": None,
        "supervisor_wait_is_bounded": False,
        "no_hard_watchdog_reason": (
            "no supervisor process-signal authority was approved"
        ),
        "supervisor_signal_authority": False,
        "process_signals_sent": 0,
        "termination_action": None,
        "worker_process_exited": (
            process.poll() is not None if process is not None else False
        ),
        "artifact_provenance": provenance,
        "worker_claim_exists": WORKER_CLAIM.is_file(),
        "worker_claim_pass": claim_pass,
        "worker_claim_read_error": claim_error,
        "failure_attestation_exists": FAILURE_ATTESTATION.is_file(),
        "success_output_inventory_exact": success_inventory_pass,
        "frozen_authority_checks": authority_checks,
        "pass": (
            process is not None
            and supervisor_error is None
            and returncode == 0
            and claim_pass
            and claim_error is None
            and not FAILURE_ATTESTATION.exists()
            and success_inventory_pass
            and all(authority_checks.values())
        ),
    }
    _write_json_x(SUPERVISOR, record)
    _phase(
        "supervisor_after_worker",
        returncode=returncode,
        pass_value=record["pass"],
    )
    if record["pass"]:
        _require_out_inventory(
            POST_WORKER_SUCCESS_INVENTORY | {SUPERVISOR.name},
            "supervisor_after_success_worker",
        )
    if not record["pass"]:
        raise RuntimeError(f"D390 worker failed: {record}")
    return 0


def _phase_contract() -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in PHASES.read_text(encoding="utf-8").splitlines()
        if line
    ]
    names = [row["phase"] for row in rows]
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
        "forward_only_wall_time": all(
            rows[index]["wall_time_ns"]
            <= rows[index + 1]["wall_time_ns"]
            for index in range(len(rows) - 1)
        ),
        "pass": names == expected
        and all(
            rows[index]["wall_time_ns"]
            <= rows[index + 1]["wall_time_ns"]
            for index in range(len(rows) - 1)
        ),
    }


def _finalize() -> int:
    required = [
        PREREG,
        PHASES,
        INVOCATION,
        STDOUT,
        STDERR,
        WORKER_AUTHORIZATION,
        WORKER_SENTINEL,
        SUPERVISOR,
        EVIDENCE,
        TRACE_GEOMETRY,
        TRACE_CSV,
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
        raise RuntimeError(f"cannot finalize D390; missing: {missing}")
    if COMPLETION.exists():
        raise RuntimeError("refusing to overwrite D390 completion")
    _require_out_inventory(PRE_FINALIZE_INVENTORY, "before_finalize")
    prereg = _read_json(PREREG)
    frozen_authority_checks = _frozen_authority_checks(
        prereg, allow_d390_output=True
    )
    if not all(frozen_authority_checks.values()):
        raise RuntimeError(
            "D390 finalize frozen authority changed: "
            f"{frozen_authority_checks}"
        )
    if FAILURE_ATTESTATION.exists():
        raise RuntimeError("D390 failure attestation forbids finalize")
    _phase("finalize_start")
    evidence = _read_json(EVIDENCE)
    supervisor = _read_json(SUPERVISOR)
    worker = _read_json(WORKER_CLAIM)
    validation = _read_json(RERUN_VALIDATION)
    manual_template = _read_json(MANUAL_TEMPLATE)
    manual = _read_json(MANUAL)
    required_manual = set(manual_template["required_checks"])
    observations = manual.get("observations", [])
    observations_valid = (
        isinstance(observations, list)
        and len(observations) >= 3
        and all(
            isinstance(observation, str) and bool(observation.strip())
            for observation in observations
        )
    )
    manual_pass = (
        set(manual.get("checks", {})) == required_manual
        and all(value is True for value in manual["checks"].values())
        and manual.get("pass") is True
        and observations_valid
        and manual.get("artifact_hashes")
        == {
            _rel(BOARD): _sha(BOARD),
            _rel(RERUN_SCREENSHOT): _sha(RERUN_SCREENSHOT),
        }
    )
    claim_artifacts = worker["artifacts"]
    bindings = {
        "evidence": (claim_artifacts["evidence"], EVIDENCE),
        "geometry": (claim_artifacts["geometry"], TRACE_GEOMETRY),
        "csv": (claim_artifacts["csv"], TRACE_CSV),
        "board": (claim_artifacts["board"], BOARD),
        "board_layout": (
            claim_artifacts["board"]["layout_validation"],
            BOARD_LAYOUT,
        ),
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
        }
        for label, (record, path) in bindings.items()
    }
    supervisor_provenance_bindings = {
        "preregistration": PREREG,
        "invocation": INVOCATION,
        "authorization": WORKER_AUTHORIZATION,
        "worker_start_sentinel": WORKER_SENTINEL,
        "stdout": STDOUT,
        "stderr": STDERR,
        "worker_claim": WORKER_CLAIM,
    }
    supervisor_provenance = supervisor.get("artifact_provenance", {})
    supervisor_linkage = {
        label: {
            "record_present": isinstance(
                supervisor_provenance.get(label), dict
            ),
            "path_exact": (
                supervisor_provenance.get(label, {}).get("path")
                == _rel(path)
            ),
            "hash_exact": (
                supervisor_provenance.get(label, {}).get("sha256")
                == _sha(path)
            ),
        }
        for label, path in supervisor_provenance_bindings.items()
    }
    invocation_record = _read_json(INVOCATION)
    authorization_record = _read_json(WORKER_AUTHORIZATION)
    sentinel_record = _read_json(WORKER_SENTINEL)
    preregistration_sha256 = _sha(PREREG)
    preregistration_chain = {
        "preregistration_sha256": preregistration_sha256,
        "invocation_bound": invocation_record.get(
            "preregistration_sha256"
        )
        == preregistration_sha256,
        "authorization_bound": authorization_record.get(
            "preregistration_sha256"
        )
        == preregistration_sha256,
        "worker_start_sentinel_bound": sentinel_record.get(
            "preregistration_sha256"
        )
        == preregistration_sha256,
        "worker_claim_bound": worker.get("preregistration")
        == {
            "path": _rel(PREREG),
            "sha256": preregistration_sha256,
        },
        "supervisor_provenance_bound": (
            supervisor_provenance.get("preregistration", {}).get("path")
            == _rel(PREREG)
            and supervisor_provenance.get(
                "preregistration", {}
            ).get("sha256")
            == preregistration_sha256
        ),
    }
    preregistration_chain["pass"] = all(
        value is True
        for key, value in preregistration_chain.items()
        if key not in {"preregistration_sha256", "pass"}
    )
    checks = {
        "numeric_localization_pass": evidence[
            "localization_contract_pass"
        ]
        is True,
        "worker_once_retry_zero_no_signal": (
            supervisor["actual_worker_invocations"] == 1
            and supervisor["retries"] == 0
            and supervisor["process_signals_sent"] == 0
        ),
        "supervisor_and_worker_claim_pass": supervisor["pass"] is True
        and worker["pass"] is True,
        "strict_rerun_validation_pass": validation.get("pass") is True,
        "manual_visual_inspection_pass": manual_pass,
        "manual_observations_nonempty": observations_valid,
        "artifact_paths_and_hashes_exact": all(
            row["path_exact"] and row["hash_exact"]
            for row in linkage.values()
        ),
        "supervisor_provenance_paths_and_hashes_exact": all(
            row["record_present"]
            and row["path_exact"]
            and row["hash_exact"]
            for row in supervisor_linkage.values()
        ),
        "preregistration_full_hash_chain_exact": preregistration_chain[
            "pass"
        ],
        "failure_attestation_absent": not FAILURE_ATTESTATION.exists(),
        "frozen_authority_rechecked_exact": all(
            frozen_authority_checks.values()
        ),
        "d389_frozen_no_repair_or_retroactive_pass": (
            evidence["immutable_d389_verdict"] == EXPECTED_D389_VERDICT
            and evidence["d389_repaired_or_recomputed"] is False
            and evidence["d389_retroactive_pass"] is False
            and evidence["directional_contract_repaired"] is False
            and evidence["all_36_pairs_reaudited"] is False
        ),
        "budgets_null_no_application": (
            evidence["selected_vertex_budget"] is None
            and evidence["adopted_vertex_budget"] is None
            and evidence["selected_budget_application_count"] == 0
        ),
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
        "artifact": "D390_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "numeric_verdict": evidence["verdict"],
        "checks": checks,
        "global_phase_contract": phase_contract,
        "execution": {
            "actual_worker_invocations": 1,
            "retries": 0,
            "headless_viewer_invocations": validation[
                "d390_execution_contract"
            ]["headless_viewer_invocations"],
            "viewer_retry": 0,
            "process_signals_sent": 0,
        },
        "worker_claim_artifact_linkage": linkage,
        "supervisor_artifact_linkage": supervisor_linkage,
        "preregistration_full_hash_chain": preregistration_chain,
        "frozen_authority_checks": frozen_authority_checks,
        "artifact_hashes": {
            _rel(path): _sha(path) for path in required
        },
        "directional_contract_repaired": False,
        "collider_or_budget_adopted": False,
        "physics_or_grasp_result": None,
        "g0a_pass": False,
        "pass": all(checks.values()),
    }
    _write_json_x(COMPLETION, completion)
    _require_out_inventory(POST_FINALIZE_INVENTORY, "after_finalize")
    if not completion["pass"]:
        raise RuntimeError(f"D390 completion failed: {checks}")
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "run", "worker", "finalize"),
        required=True,
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
