#!/usr/bin/env python3
"""D395: certify exact-subset gate semantics for 36 pairs / 144 directions.

Offline-only.  This script never reruns D389 clipping.  It copies the 103
completed directional records from immutable D389 and gives each of the 41
failed directions a rank-agnostic *upper-bound* certificate computed from the
immutable D390 terminal point set.  An upper bound is not an intersection
volume: certified failed records therefore keep ``derived_volume_m3=None``.

This certificate is valid for the ideal exact-halfspace subset relation only.
It is not a bit-exact replay of the remaining Float64 clipping operations and
does not bound any roundoff-induced point motion outside ``conv(P)``.  Thus all
41 actual frozen-solver results remain failed/null; only their exact-subset
semantic Boolean is certified.  The call29 rank and class stay null.  D389
evidence and verdict are never modified.  Rerun is an inspection copy;
canonical JSON and exact Fractions are the numerical authority.
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
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASE = "D395"
ATTEMPT = "attempt1_all36_pair_144direction_gate_semantics_propagation"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d395" / ATTEMPT
START = REPO / "START_HERE.md"
EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
VARIABLES = [
    "all36_pair_144direction_hybrid_registered_gate_semantics_v1",
    "failed41_rank_agnostic_exact_tetra_sum_upper_bound_v1",
]
VOLUME_GATE = Fraction(1, 10**18)
CALL29_INDEX = 29
CALL29_ID = "lower_01_02_pre_float32_lbr"
PAIR_COUNT = 36
DIRECTION_COUNT = 144
FAILED_COUNT = 41
SUCCESS_COUNT = 103
EXPECTED_TOTAL_TETRA_COMBINATIONS = 42928
EXPECTED_POSITIVE_BOUND_COUNT = 23
EXPECTED_ZERO_BOUND_COUNT = 18
EXPECTED_CALL29_BOUND = Fraction(
    53541976163091329624703586759,
    287342913912354160942190067590682971928513585409425408,
)
EXPECTED_MAX_BOUND_CALL_INDEX = 35
EXPECTED_MAX_BOUND = Fraction(
    721947539299282420838041739364374335,
    49039857307708443467467104868809893875799651909875269632,
)
EXPECTED_PRE_AND_POST_POSITIVE_PAIRS = frozenset(
    {("UPPER", 1, 2), ("LOWER", 2, 3)}
)
EXPECTED_POST_ONLY_POSITIVE_PAIRS = frozenset(
    {
        ("UPPER", 0, 1),
        ("UPPER", 2, 3),
        ("UPPER", 3, 4),
        ("UPPER", 4, 5),
        ("LOWER", 0, 1),
        ("LOWER", 1, 2),
        ("LOWER", 3, 4),
        ("LOWER", 4, 5),
        ("LOWER", 5, 6),
    }
)
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
AUTHORITY_ENV = "D395_EXECUTION_AUTHORITY_SHA256"
WORKER_AUTHORITY_ENV = "D395_WORKER_AUTHORIZATION_SHA256"

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
D394_NUMERIC_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d394"
    / "attempt2_gate_numeric_null_semantics_repair"
)
D394_COMPLETE_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d394"
    / "attempt3_ascii_exponent_visual_repair"
)

D389_EVIDENCE = D389_DIR / "d389_numeric_and_tie_audit_evidence.json"
D389_WITNESS = D389_DIR / "d389_reconstructed_seam_witness_geometry.json"
D389_PREREG = D389_DIR / "d389_preregistration.json"
D390_EVIDENCE = D390_DIR / "d390_boundary_collapse_localization_evidence.json"
D390_GEOMETRY = D390_DIR / "d390_terminal_candidate_geometry.json"
D392_EVIDENCE = D392_DIR / "d392_remaining35_rank_evidence.json"
D393_EVIDENCE = D393_DIR / "d393_call29_provenance_evidence.json"
D393_COMPLETION = D393_DIR / "d393_completion_summary.json"
D394_EVIDENCE = D394_NUMERIC_DIR / "d394_full10_volume_semantics_evidence.json"
D394_COMPLETION = D394_COMPLETE_DIR / "d394_completion_summary.json"

INPUT_HASHES = {
    D389_EVIDENCE: "9423e870c0a218606781943abd2f5c48cb1e5d53cbbf9fb1212294b4ef5bb5dd",
    D389_WITNESS: "66042a93389cb8d0e6c867be87382566c753cd965ceda619e947e73de4a607be",
    D389_PREREG: "f4b38c4c5db311412c5700f792a66be805bbe06abf9be89b533002e1860ce780",
    D390_EVIDENCE: "3014610b7b2fd953740d239b91d9d9dce8aa917be67c6a1be15cf6ac052d9975",
    D390_GEOMETRY: "73fc986043b976bec26e1cc92643b8aab281a529f1c71c2918163ba7b98475c7",
    D392_EVIDENCE: "9ced175925c6c528d47bf94e5ae224e65bae1d4d6c88fc236952343de0c72102",
    D393_EVIDENCE: "537537d89a2204987eebfa9bf668968801247e7cef70b7b694434b16b98883a9",
    D393_COMPLETION: "cafac7a1f79b785592204300dc98c49ed1cd0ae0ba424d7091d641e7aa4cfab9",
    D394_EVIDENCE: "7672f208cc704bd9c3a51bc0b60040e2a121335cf12dcfaa5fd851484dd089a1",
    D394_COMPLETION: "0b04651f1f984f71d880eb52b3fa968154dd9970bbbeffcf410ae17eb4d21e30",
}

AUTHORITY = OUT_DIR / "d395_execution_authority.json"
PREREG = OUT_DIR / "d395_preregistration.json"
PHASES = OUT_DIR / "d395_phase_markers.jsonl"
INVOCATION = OUT_DIR / "d395_offline_worker_invocation.json"
WORKER_AUTH = OUT_DIR / "d395_worker_authorization.json"
SENTINEL = OUT_DIR / "d395_worker_start_sentinel.json"
STDOUT = OUT_DIR / "d395_offline_worker_stdout.log"
STDERR = OUT_DIR / "d395_offline_worker_stderr.log"
PROGRESS = OUT_DIR / "d395_failed41_progress.jsonl"
EVIDENCE = OUT_DIR / "d395_all36_gate_semantics_evidence.json"
CSV_PATH = OUT_DIR / "d395_all144_direction_semantics.csv"
GEOMETRY = OUT_DIR / "d395_pair_atlas_geometry.json"
WORKER_CLAIM = OUT_DIR / "d395_offline_worker_claim.json"
SUPERVISOR = OUT_DIR / "d395_offline_worker_supervisor.json"
BOARD = OUT_DIR / "d395_all36_gate_semantics_1920x1080.png"
LAYOUT = OUT_DIR / "d395_board_layout_validation.json"
RRD = OUT_DIR / "d395_all36_gate_semantics.rerun.rrd"
RBL = OUT_DIR / "d395_all36_gate_semantics.rerun.rbl"
RERUN_VALIDATION = OUT_DIR / "d395_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d395_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d395_manual_visual_inspection_template.json"
MANUAL = OUT_DIR / "d395_manual_visual_inspection.json"
OBSERVABILITY = OUT_DIR / "d395_observability_claim.json"
FAILURE = OUT_DIR / "d395_failure_attestation.json"
COMPLETION = OUT_DIR / "d395_completion_summary.json"

MANUAL_KEYS = [
    "board_exact_1920x1080",
    "board_all_36_pairs_visible",
    "board_11_adjacent_and_25_nonadjacent_visible",
    "board_pre_post_classification_readable",
    "board_failed41_upper_bound_semantics_readable",
    "board_call29_null_nonclaim_readable",
    "board_derived_volume_null_readable",
    "board_float64_clip_rounding_not_replayed_readable",
    "board_no_text_overlap_or_clipping",
    "rerun_36_pair_entities_visible",
    "rerun_failed41_terminal_cloud_entities_visible",
    "rerun_terminal_clouds_are_normalized_inspection_copies",
    "rerun_pair_grid_is_auxiliary",
    "rerun_metadata_matches_canonical_json_and_notifications_obscure_nothing",
]
PHASE_ORDER = [
    "prepare_start",
    "prepare_end",
    "supervisor_before_worker",
    "worker_start",
    "d389_144_manifest_bound",
    "failed41_upper_bounds_committed",
    "all36_pair_semantics_committed",
    "canonical_numeric_evidence_committed",
    "worker_end",
    "supervisor_after_worker",
    "observability_start",
    "observability_end",
    "finalize_start",
    "finalize_end",
]


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
            "float": float(value),
        }
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    return value


def _write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _native(value), indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_native(value), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": _rel(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _artifact_matches(reference: Any, path: Path) -> bool:
    return (
        isinstance(reference, dict)
        and reference.get("path") == _rel(path)
        and reference.get("sha256") == _sha(path)
        and reference.get("bytes") == path.stat().st_size
    )


def _array_sha(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(value.shape).encode("ascii"))
    h.update(b"|float64|C|")
    h.update(value.tobytes(order="C"))
    return h.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        _native(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True,
    )
    return result.stdout.rstrip("\r\n")


def _status_outside_output() -> list[str]:
    prefix = _rel(OUT_DIR) + "/"
    rows = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return [row for row in rows if not row[3:].startswith(prefix)]


def _inventory() -> set[str]:
    if not OUT_DIR.exists():
        return set()
    return {path.name for path in OUT_DIR.iterdir() if path.is_file()}


def _require_inventory(expected: set[str], where: str) -> None:
    actual = _inventory()
    if actual != expected:
        raise RuntimeError(
            f"{where}: D395 inventory mismatch; "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )


def _phase(name: str, **extra: Any) -> None:
    if name not in PHASE_ORDER:
        raise ValueError(name)
    rows = []
    if PHASES.exists():
        rows = [
            json.loads(line) for line in PHASES.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    expected = PHASE_ORDER[len(rows)]
    if name != expected:
        raise RuntimeError(f"D395 phase {name!r}, expected {expected!r}")
    _append(
        PHASES,
        {
            "ordinal": len(rows),
            "phase": name,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_ns": time.time_ns(),
            **extra,
        },
    )


def _phase_contract() -> dict[str, Any]:
    rows = [
        json.loads(line) for line in PHASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = [row["phase"] for row in rows]
    return {
        "count": len(rows),
        "expected_count": len(PHASE_ORDER),
        "exact": names == PHASE_ORDER,
        "ordinals_exact": [row["ordinal"] for row in rows] == list(range(len(rows))),
        "monotonic_forward": all(
            rows[i]["monotonic_ns"] < rows[i + 1]["monotonic_ns"]
            for i in range(len(rows) - 1)
        ),
        "wall_forward": all(
            rows[i]["wall_time_ns"] <= rows[i + 1]["wall_time_ns"]
            for i in range(len(rows) - 1)
        ),
    }


def _frozen_checks() -> dict[str, bool]:
    checks = {f"sha::{_rel(path)}": path.is_file() and _sha(path) == digest
              for path, digest in INPUT_HASHES.items()}
    checks.update(
        {
            "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
            "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
            "d389_threshold_exact": (
                _read(D389_PREREG)["frozen_contract"]["positive_volume_epsilon_m3"]
                == 1.0e-18
            ),
            "d393_complete_and_call29_null": (
                _read(D393_COMPLETION).get("pass") is True
                and _read(D393_COMPLETION).get("call29_authoritative_rank") is None
                and _read(D393_COMPLETION).get("call29_authoritative_class") is None
            ),
            "d394_numeric_pass": (
                _read(D394_EVIDENCE).get("pass") is True
                and _read(D394_EVIDENCE).get("pair_or_seam_verdict_updated") is False
                and _read(D394_EVIDENCE).get("call29_rank") is None
                and _read(D394_EVIDENCE).get("call29_class") is None
            ),
            "d394_complete": _read(D394_COMPLETION).get("pass") is True,
        }
    )
    return checks


def _authority_core_checks(authority: dict[str, Any]) -> dict[str, bool]:
    return {
        "artifact": authority.get("artifact") == "D395_EXTERNAL_EXECUTION_AUTHORITY_V1",
        "case_attempt": authority.get("case") == CASE
        and authority.get("attempt") == ATTEMPT,
        "script": authority.get("script_path") == _rel(SCRIPT)
        and authority.get("script_sha256") == _sha(SCRIPT),
        "start": authority.get("start_path") == _rel(START)
        and authority.get("start_sha256") == _sha(START),
        "variables": authority.get("new_variables") == VARIABLES,
        "output": authority.get("output_dir") == _rel(OUT_DIR),
        "inputs": authority.get("input_sha256")
        == {_rel(path): digest for path, digest in INPUT_HASHES.items()},
        "input_files_current": all(
            path.is_file() and _sha(path) == digest
            for path, digest in INPUT_HASHES.items()
        ),
        "git": authority.get("git", {}).get("head") == EXPECTED_HEAD
        and authority.get("git", {}).get("origin_master") == EXPECTED_HEAD
        and _git("rev-parse", "HEAD") == EXPECTED_HEAD
        and _git("rev-parse", "origin/master") == EXPECTED_HEAD
        and authority.get("git", {}).get("status_outside_output_dir")
        == _status_outside_output(),
        "execution": authority.get("execution_contract")
        == {
            "worker_invocations": 1,
            "worker_retries": 0,
            "process_signals": 0,
            "viewer_maximum": 1,
            "viewer_retries": 0,
        },
    }


def _authority_checks(authority: dict[str, Any]) -> dict[str, bool]:
    checks = _authority_core_checks(authority)
    external = os.environ.get(AUTHORITY_ENV)
    checks["external_sha"] = (
        external is not None and external == _sha(AUTHORITY)
    )
    return checks


def _frac_point(row: Sequence[float]) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(Fraction.from_float(float(value)) for value in row)  # type: ignore[return-value]


def _sub(
    left: Sequence[Fraction], right: Sequence[Fraction]
) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(left[i] - right[i] for i in range(3))  # type: ignore[return-value]


def _det3(
    a: Sequence[Fraction], b: Sequence[Fraction], c: Sequence[Fraction]
) -> Fraction:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _tetra6(
    a: Sequence[Fraction],
    b: Sequence[Fraction],
    c: Sequence[Fraction],
    d: Sequence[Fraction],
) -> Fraction:
    return _det3(_sub(b, a), _sub(c, a), _sub(d, a))


def _tetra_sum_upper(points_f64: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Exact sum of all vertex-tetra volumes for an exact-subset certificate.

    Every finite 3-D point hull admits a triangulation whose tetrahedra use
    hull vertices.  The non-negative sum over *all* 4-vertex subsets therefore
    upper-bounds any such triangulation and hence the hull volume.  This holds
    without first deciding affine rank.  It is deliberately an upper bound,
    never a derived volume.  The monotone continuation claim applies only to
    ideal exact halfspace intersections.  It is not a bit-exact replay or an
    error bound for the remaining Float64 clipping arithmetic.
    """
    canonical_f64 = sorted(set(tuple(float(v) for v in row) for row in points_f64))
    points = [_frac_point(row) for row in canonical_f64]
    total = Fraction(0)
    maximum = Fraction(0)
    nonzero = 0
    for indices in itertools.combinations(range(len(points)), 4):
        value = abs(_tetra6(*(points[index] for index in indices))) / 6
        total += value
        maximum = max(maximum, value)
        nonzero += int(value > 0)
    return {
        "unique_point_count": len(points),
        "combination_count": math.comb(len(points), 4) if len(points) >= 4 else 0,
        "nonzero_tetra_count": nonzero,
        "maximum_tetra_m3": maximum,
        "tetra_sum_upper_bound_m3": total,
        "below_or_equal_frozen_gate": total <= VOLUME_GATE,
        "derived_volume_m3": None,
        "proof": (
            "A vertex-only tetrahedralization of conv(P) uses a subset of "
            "the 4-point tetrahedra enumerated here; summing every absolute "
            "tetra volume is therefore a rank-agnostic upper bound.  Under "
            "ideal exact halfspace intersection, every continuation K is a "
            "subset of conv(P), so volume(K) cannot exceed this bound."
        ),
        "certificate_scope": "EXACT_HALFSPACE_SUBSET_SEMANTICS_ONLY",
        "not_bit_exact_remaining_float64_clip_replay": True,
        "float64_roundoff_point_motion_outside_conv_p_bounded": False,
    }


def _call_id(
    target: str, left: int, right: int, stage: str, direction: str
) -> str:
    suffix = "lbr" if direction == "left_clipped_by_right" else "rbl"
    return f"{target.lower()}_{left:02d}_{right:02d}_{stage}_{suffix}"


def _direction_slots(pair: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    rows = []
    for stage in ("pre_float32", "post_float32"):
        block = pair[f"{stage}_directional_epsilon0"]
        for direction in ("left_clipped_by_right", "right_clipped_by_left"):
            rows.append((stage, direction, block[direction]))
    return rows


def _compute() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    d389 = _read(D389_EVIDENCE)
    d390 = _read(D390_EVIDENCE)
    geometry = _read(D390_GEOMETRY)
    pair_rows = d389["seam_numeric_provenance_audit"]["pair_results"]
    failed_records = d390["failed_call_records"]
    failed_geometry = geometry["records"]
    if len(pair_rows) != PAIR_COUNT or len(failed_records) != FAILED_COUNT:
        raise RuntimeError("D395 frozen denominator mismatch")
    failed_by_id = {row["call_id"]: row for row in failed_records}
    geometry_by_id = {row["call_id"]: row for row in failed_geometry}
    if set(failed_by_id) != set(geometry_by_id):
        raise RuntimeError("D395 failed-call geometry identity mismatch")
    _phase("d389_144_manifest_bound")

    propagated: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    failed_bounds: list[dict[str, Any]] = []
    ordinal = 0
    for pair_index, pair in enumerate(pair_rows):
        for stage, direction, stored in _direction_slots(pair):
            call_id = _call_id(
                pair["target"], pair["left_index"], pair["right_index"],
                stage, direction,
            )
            base = {
                "direction_index": ordinal,
                "pair_index": pair_index,
                "call_id": call_id,
                "target": pair["target"],
                "left_index": pair["left_index"],
                "right_index": pair["right_index"],
                "adjacent": pair["adjacent"],
                "stage": stage,
                "direction": direction,
                "stored_d389_record_sha256": _canonical_sha(stored),
                "stored_calculation_pass": stored["calculation_pass"],
                "stored_positive_volume": stored["positive_volume"],
                "stored_volume_m3": stored["volume_m3"],
            }
            if stored["calculation_pass"]:
                result = {
                    **base,
                    "semantic_source": (
                        "IMMUTABLE_D389_ACTUAL_FLOAT64_QHULL_SOLVER_OUTPUT"
                    ),
                    "actual_solver_calculation_pass": True,
                    "actual_solver_result": "IMMUTABLE_D389_STORED_OUTPUT",
                    "actual_solver_positive_volume": stored["positive_volume"],
                    "ideal_exact_subset_certificate_applicable": False,
                    "ideal_exact_subset_certificate_pass": None,
                    "ideal_exact_subset_certificate_positive_volume": None,
                    "hybrid_registered_gate_positive_volume": stored[
                        "positive_volume"
                    ],
                    "derived_volume_m3": None,
                    "terminal_upper_bound_m3": None,
                    "rank_or_class_used": False,
                    "failed_call_index": None,
                }
            else:
                failed = failed_by_id.get(call_id)
                terminal = geometry_by_id.get(call_id)
                if failed is None or terminal is None:
                    raise RuntimeError(f"missing failed lineage for {call_id}")
                points = terminal["terminal_candidate_unique_points_f64_m"]
                if (
                    terminal["terminal_candidate_unique_points_sha256"]
                    != _array_sha(points)
                    or terminal["terminal_candidate_unique_points_sha256"]
                    != failed["independent_reconstruction"]["candidate"][
                        "unique_points_sha256"
                    ]
                ):
                    raise RuntimeError(f"terminal hash mismatch: {call_id}")
                bound = _tetra_sum_upper(points)
                bound_row = {
                    "call_index": failed["call_index"],
                    "call_id": call_id,
                    "terminal_points_sha256": _array_sha(points),
                    "actual_solver_result": None,
                    "actual_solver_positive_volume": None,
                    "ideal_exact_subset_certificate_positive_volume": (
                        False if bound["below_or_equal_frozen_gate"] else None
                    ),
                    **bound,
                }
                failed_bounds.append(bound_row)
                _append(
                    PROGRESS,
                    {
                        "ordinal": len(failed_bounds) - 1,
                        "call_index": failed["call_index"],
                        "call_id": call_id,
                        "upper_bound_m3": float(
                            bound["tetra_sum_upper_bound_m3"]
                        ),
                        "below_gate": bound["below_or_equal_frozen_gate"],
                    },
                )
                result = {
                    **base,
                    "semantic_source": (
                        "D390_TERMINAL_EXACT_HALFSPACE_SUBSET_TETRA_SUM_"
                        "UPPER_BOUND"
                    ),
                    "ideal_exact_subset_certificate_applicable": True,
                    "ideal_exact_subset_certificate_pass": bound[
                        "below_or_equal_frozen_gate"
                    ],
                    "actual_solver_calculation_pass": False,
                    "actual_solver_result": None,
                    "actual_solver_positive_volume": None,
                    "ideal_exact_subset_certificate_positive_volume": (
                        False if bound["below_or_equal_frozen_gate"] else None
                    ),
                    "hybrid_registered_gate_positive_volume": (
                        False if bound["below_or_equal_frozen_gate"] else None
                    ),
                    "derived_volume_m3": None,
                    "terminal_upper_bound_m3": bound[
                        "tetra_sum_upper_bound_m3"
                    ],
                    "rank_or_class_used": False,
                    "failed_call_index": failed["call_index"],
                }
            propagated.append(result)
            csv_rows.append(
                {
                    "direction_index": ordinal,
                    "pair_index": pair_index,
                    "call_id": call_id,
                    "adjacent": pair["adjacent"],
                    "stage": stage,
                    "direction": direction,
                    "stored_pass": stored["calculation_pass"],
                    "source": result["semantic_source"],
                    "actual_solver_positive": result[
                        "actual_solver_positive_volume"
                    ],
                    "ideal_exact_subset_certificate_positive": result[
                        "ideal_exact_subset_certificate_positive_volume"
                    ],
                    "hybrid_registered_gate_positive": result[
                        "hybrid_registered_gate_positive_volume"
                    ],
                    "derived_volume_m3": "",
                    "upper_bound_m3": (
                        "" if result["terminal_upper_bound_m3"] is None
                        else f"{float(result['terminal_upper_bound_m3']):.17e}"
                    ),
                }
            )
            ordinal += 1
    _phase("failed41_upper_bounds_committed")

    pair_semantics: list[dict[str, Any]] = []
    by_pair = {
        index: [row for row in propagated if row["pair_index"] == index]
        for index in range(PAIR_COUNT)
    }
    for pair_index, pair in enumerate(pair_rows):
        stage_rows: dict[str, Any] = {}
        for stage in ("pre_float32", "post_float32"):
            directions = [row for row in by_pair[pair_index] if row["stage"] == stage]
            positives = [
                row["hybrid_registered_gate_positive_volume"]
                for row in directions
            ]
            semantic_positive = (
                any(value is True for value in positives)
                if all(value is not None for value in positives)
                else None
            )
            strict = pair[f"{stage}_epsilon0"]["positive_volume"]
            stage_rows[stage] = {
                "hybrid_direction_gate_values": positives,
                "hybrid_registered_gate_positive_volume": semantic_positive,
                "strict_positive_volume": strict,
                "strict_vs_hybrid_registered_boolean_agreement": (
                    semantic_positive == strict
                ),
                "certificate_direction_count": sum(
                    row["stored_calculation_pass"] is False for row in directions
                ),
                "derived_volume_m3": None,
            }
        pre = stage_rows["pre_float32"][
            "hybrid_registered_gate_positive_volume"
        ]
        post = stage_rows["post_float32"][
            "hybrid_registered_gate_positive_volume"
        ]
        classification = {
            (True, True): "PRE_AND_POST_GATE_POSITIVE_REGISTERED_STREAM",
            (False, True): "POST_ONLY_GATE_POSITIVE_REGISTERED_STREAM",
            (False, False): "PRE_AND_POST_GATE_NONPOSITIVE_REGISTERED_STREAM",
        }.get((pre, post), "UNRESOLVED_REGISTERED_STREAM_FAIL_STOP")
        pair_semantics.append(
            {
                "pair_index": pair_index,
                "target": pair["target"],
                "prim_name": pair["prim_name"],
                "left_index": pair["left_index"],
                "right_index": pair["right_index"],
                "adjacent": pair["adjacent"],
                "pre_float32": stage_rows["pre_float32"],
                "post_float32": stage_rows["post_float32"],
                "hybrid_registered_gate_pair_pattern": classification,
                "actual_float64_solver_pair_classification": None,
                "ideal_exact_subset_only_pair_classification": None,
                "classification_role": (
                    "mixed-authority diagnostic: D389 actual solver for 103 "
                    "directions plus ideal exact-subset certificates for 41; "
                    "not adopted, no Float32 causation, physics, or "
                    "manufacturability claim"
                ),
                "derived_volume_m3": None,
                "d389_pair_record_sha256": _canonical_sha(pair),
            }
        )
    _phase("all36_pair_semantics_committed")

    adjacent = [row for row in pair_semantics if row["adjacent"]]
    nonadjacent = [row for row in pair_semantics if not row["adjacent"]]
    call29 = next(row for row in failed_bounds if row["call_index"] == CALL29_INDEX)
    max_bound = max(
        failed_bounds, key=lambda row: row["tetra_sum_upper_bound_m3"]
    )
    positive_bound_count = sum(
        row["tetra_sum_upper_bound_m3"] > 0 for row in failed_bounds
    )
    zero_bound_count = sum(
        row["tetra_sum_upper_bound_m3"] == 0 for row in failed_bounds
    )
    total_tetra_combinations = sum(
        row["combination_count"] for row in failed_bounds
    )
    pre_and_post_pair_ids = {
        (row["target"], row["left_index"], row["right_index"])
        for row in pair_semantics
        if row["hybrid_registered_gate_pair_pattern"]
        == "PRE_AND_POST_GATE_POSITIVE_REGISTERED_STREAM"
    }
    post_only_pair_ids = {
        (row["target"], row["left_index"], row["right_index"])
        for row in pair_semantics
        if row["hybrid_registered_gate_pair_pattern"]
        == "POST_ONLY_GATE_POSITIVE_REGISTERED_STREAM"
    }
    nonpositive_pair_ids = {
        (row["target"], row["left_index"], row["right_index"])
        for row in pair_semantics
        if row["hybrid_registered_gate_pair_pattern"]
        == "PRE_AND_POST_GATE_NONPOSITIVE_REGISTERED_STREAM"
    }
    all_pair_ids = {
        (row["target"], row["left_index"], row["right_index"])
        for row in pair_semantics
    }
    checks = {
        "pair_count_36": len(pair_semantics) == 36,
        "direction_count_144": len(propagated) == 144,
        "direction_call_ids_unique_144": (
            len({row["call_id"] for row in propagated}) == 144
        ),
        "direction_indices_exact_0_143": [
            row["direction_index"] for row in propagated
        ] == list(range(144)),
        "exact_four_directions_per_pair": all(
            len(by_pair[index]) == 4 for index in range(PAIR_COUNT)
        ),
        "failed_record_indices_unique_range41": (
            {row["call_index"] for row in failed_records} == set(range(41))
            and len(failed_records) == 41
        ),
        "failed_geometry_indices_unique_range41": (
            {row["call_index"] for row in failed_geometry} == set(range(41))
            and len(failed_geometry) == 41
        ),
        "stored_failed_id_set_matches_d390": (
            {
                row["call_id"] for row in propagated
                if row["stored_calculation_pass"] is False
            }
            == {row["call_id"] for row in failed_records}
        ),
        "stored_success_103": sum(row["stored_calculation_pass"] for row in propagated)
        == SUCCESS_COUNT,
        "failed_bound_41": len(failed_bounds) == FAILED_COUNT,
        "failed41_all_below_gate": all(
            row["below_or_equal_frozen_gate"] for row in failed_bounds
        ),
        "failed41_derived_volume_all_null": all(
            row["derived_volume_m3"] is None for row in failed_bounds
        ),
        "failed41_actual_solver_positive_all_null": all(
            row["actual_solver_positive_volume"] is None
            for row in propagated
            if row["stored_calculation_pass"] is False
        ),
        "failed41_actual_solver_result_all_null": all(
            row["actual_solver_result"] is None
            for row in propagated
            if row["stored_calculation_pass"] is False
        ),
        "failed41_actual_solver_calculation_stays_failed": all(
            row["actual_solver_calculation_pass"] is False
            for row in propagated
            if row["stored_calculation_pass"] is False
        ),
        "hybrid_registered_gate_144_populated": all(
            row["hybrid_registered_gate_positive_volume"] is not None
            for row in propagated
        ),
        "frozen_gate_positive_nonpositive_26_118": (
            sum(
                row["hybrid_registered_gate_positive_volume"] is True
                for row in propagated
            )
            == 26
            and sum(
                row["hybrid_registered_gate_positive_volume"] is False
                for row in propagated
            )
            == 118
        ),
        "strict_vs_hybrid_registered_72_agree": all(
            stage["strict_vs_hybrid_registered_boolean_agreement"]
            for pair in pair_semantics
            for stage in (pair["pre_float32"], pair["post_float32"])
        ),
        "adjacent_11_nonadjacent_25": len(adjacent) == 11 and len(nonadjacent) == 25,
        "adjacent_registered_pattern_counts_2_9_no_causation": (
            sum(
                row["hybrid_registered_gate_pair_pattern"]
                == "PRE_AND_POST_GATE_POSITIVE_REGISTERED_STREAM"
                for row in adjacent
            )
            == 2
            and sum(
                row["hybrid_registered_gate_pair_pattern"]
                == "POST_ONLY_GATE_POSITIVE_REGISTERED_STREAM"
                for row in adjacent
            )
            == 9
        ),
        "nonadjacent_25_nonpositive": all(
            row["hybrid_registered_gate_pair_pattern"]
            == "PRE_AND_POST_GATE_NONPOSITIVE_REGISTERED_STREAM"
            for row in nonadjacent
        ),
        "registered_pair_identity_sets_exact": (
            pre_and_post_pair_ids == EXPECTED_PRE_AND_POST_POSITIVE_PAIRS
            and post_only_pair_ids == EXPECTED_POST_ONLY_POSITIVE_PAIRS
            and nonpositive_pair_ids
            == (
                all_pair_ids
                - EXPECTED_PRE_AND_POST_POSITIVE_PAIRS
                - EXPECTED_POST_ONLY_POSITIVE_PAIRS
            )
            and len(nonpositive_pair_ids) == 25
        ),
        "call29_identity": call29["call_id"] == CALL29_ID,
        "call29_bound_below_gate": call29["below_or_equal_frozen_gate"],
        "call29_bound_expected": (
            call29["tetra_sum_upper_bound_m3"] == EXPECTED_CALL29_BOUND
        ),
        "failed41_total_tetra_combinations_42928": (
            total_tetra_combinations == EXPECTED_TOTAL_TETRA_COMBINATIONS
        ),
        "failed41_positive_zero_bounds_23_18": (
            positive_bound_count == EXPECTED_POSITIVE_BOUND_COUNT
            and zero_bound_count == EXPECTED_ZERO_BOUND_COUNT
        ),
        "maximum_bound_call35_expected": (
            max_bound["call_index"] == EXPECTED_MAX_BOUND_CALL_INDEX
            and max_bound["tetra_sum_upper_bound_m3"]
            == EXPECTED_MAX_BOUND
        ),
        "maximum_bound_below_gate": max_bound[
            "tetra_sum_upper_bound_m3"
        ] <= VOLUME_GATE,
        "call29_rank_class_not_read_or_adopted": True,
        "all_pair_derived_volumes_null": all(
            row["derived_volume_m3"] is None for row in pair_semantics
        ),
        "all_actual_float64_solver_pair_classification_null": all(
            row["actual_float64_solver_pair_classification"] is None
            for row in pair_semantics
        ),
        "all_ideal_exact_subset_only_pair_classification_null": all(
            row["ideal_exact_subset_only_pair_classification"] is None
            for row in pair_semantics
        ),
    }
    numeric_pass = all(checks.values())
    evidence = {
        "artifact": "D395_ALL36_GATE_SEMANTICS_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "what_and_why": (
            "Build a non-adopted mixed-authority registered gate table: copy "
            "103 immutable D389 actual solver Booleans and add 41 ideal "
            "exact-halfspace subset certificates without inventing solver "
            "results or volumes for the failed calls."
        ),
        "input_hashes": {_rel(path): digest for path, digest in INPUT_HASHES.items()},
        "frozen_volume_gate_m3": VOLUME_GATE,
        "method": {
            "stored_successes": (
                "copy immutable D389 actual Float64/Qhull solver Boolean; "
                "not an exact-subset certificate; no clipping rerun"
            ),
            "stored_failures": (
                "exact rank-agnostic sum of all 4-point tetra volumes upper-"
                "bounds the terminal hull; an ideal exact-halfspace "
                "continuation is a subset"
            ),
            "certificate_scope": "EXACT_HALFSPACE_SUBSET_SEMANTICS_ONLY",
            "combined_table_authority": (
                "HYBRID_103_ACTUAL_SOLVER_PLUS_41_IDEAL_EXACT_SUBSET_"
                "CERTIFICATES"
            ),
            "hybrid_table_adopted": False,
            "not_bit_exact_remaining_float64_clip_replay": True,
            "float64_roundoff_point_motion_outside_conv_p_bounded": False,
            "gate_nonpositive_definition": (
                "not greater than the frozen 1e-18 m3 gate; not a claim of "
                "geometric zero"
            ),
            "failed_actual_solver_positive_volume_policy": None,
            "failed_derived_volume_policy": None,
            "call29_rank_class_policy": None,
        },
        "counts": {
            "pairs": len(pair_semantics),
            "directions": len(propagated),
            "stored_successes": sum(
                row["stored_calculation_pass"] for row in propagated
            ),
            "ideal_exact_subset_certified_failures": len(failed_bounds),
            "adjacent": len(adjacent),
            "nonadjacent": len(nonadjacent),
            "frozen_gate_positive_directions": sum(
                row["hybrid_registered_gate_positive_volume"] is True
                for row in propagated
            ),
            "frozen_gate_nonpositive_directions": sum(
                row["hybrid_registered_gate_positive_volume"] is False
                for row in propagated
            ),
            "failed_upper_bound_positive": positive_bound_count,
            "failed_upper_bound_zero": zero_bound_count,
            "failed_total_tetra_combinations": total_tetra_combinations,
        },
        "registered_expected_extrema": {
            "call29_upper_bound_m3": call29["tetra_sum_upper_bound_m3"],
            "maximum_upper_bound_call_index": max_bound["call_index"],
            "maximum_upper_bound_m3": max_bound[
                "tetra_sum_upper_bound_m3"
            ],
            "gate_to_maximum_bound_ratio": (
                VOLUME_GATE / max_bound["tetra_sum_upper_bound_m3"]
            ),
        },
        "checks": checks,
        "failed41_upper_bounds": failed_bounds,
        "hybrid_direction_table": propagated,
        "pair_semantics": pair_semantics,
        "call29_rank": None,
        "call29_class": None,
        "authoritative_all_41_rank_class_aggregate": None,
        "failed41_actual_solver_positive_volume": None,
        "failed41_actual_solver_result": None,
        "derived_failed_direction_volumes_m3": None,
        "d389_modified": False,
        "d389_retroactive_pass": False,
        "d389_solver_repaired": False,
        "d389_semantic_certificate_adopted": False,
        "hybrid_table_adopted": False,
        "pair_or_seam_geometry_modified": False,
        "numeric_verdict": (
            "D395_HYBRID_103_ACTUAL_41_IDEAL_CERTIFICATE_TABLE_PASS_"
            "NO_SOLVER_REPAIR_NO_ADOPTION"
            if numeric_pass
            else "D395_HYBRID_REGISTERED_GATE_TABLE_FAIL_STOP"
        ),
        "scope_counters": {
            "d389_clipping_reexecutions": 0,
            "remaining_float64_clip_replays": 0,
            "rank_or_class_evaluations": 0,
            "pair_geometry_updates": 0,
            "collider_usd_isaac_physx_warp_cuda": 0,
            "cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_settings_changes": 0,
        },
        "algorithm_elapsed_seconds": time.monotonic() - started,
        "g0a_pass": False,
        "pass": numeric_pass,
    }
    atlas = {
        "artifact": "D395_TERMINAL_CLOUD_AND_PAIR_ATLAS_GEOMETRY_V1",
        "canonical_evidence_sha256": None,
        "canonical_float64_authority": _artifact(D390_GEOMETRY),
        "failed_terminal_clouds": [],
        "pairs": [
            {
                "pair_index": row["pair_index"],
                "target": row["target"],
                "left_index": row["left_index"],
                "right_index": row["right_index"],
                "adjacent": row["adjacent"],
                "classification": row[
                    "hybrid_registered_gate_pair_pattern"
                ],
                "position": [
                    float((row["pair_index"] % 6) * 2.2),
                    float(-(row["pair_index"] // 6) * 2.0),
                    -3.5,
                ],
            }
            for row in pair_semantics
        ],
        "viewer_role": (
            "41 normalized terminal-cloud inspection copies are the decision "
            "subject; the 36-pair mixed-authority pattern grid is auxiliary"
        ),
        "display_geometry_authority": (
            "Float32 inspection only; never hashed back into the exact "
            "Fraction/Float64 gate"
        ),
    }
    bound_by_call = {row["call_index"]: row for row in failed_bounds}
    for terminal in sorted(failed_geometry, key=lambda row: row["call_index"]):
        call_index = terminal["call_index"]
        source_points = np.asarray(
            terminal["terminal_candidate_unique_points_f64_m"],
            dtype=np.float64,
        )
        center = source_points.mean(axis=0)
        centered = source_points - center
        radius = float(np.max(np.linalg.norm(centered, axis=1)))
        scale = 0.75 / radius if radius > 0.0 else 1.0
        offset = np.asarray(
            [
                float((call_index % 7) * 2.4),
                float(-(call_index // 7) * 2.4),
                0.0,
            ],
            dtype=np.float64,
        )
        display = np.asarray(
            centered * scale + offset, dtype=np.float32
        )
        bound = bound_by_call[call_index]
        atlas["failed_terminal_clouds"].append(
            {
                "call_index": call_index,
                "call_id": terminal["call_id"],
                "source_point_count": len(source_points),
                "source_terminal_points_sha256": terminal[
                    "terminal_candidate_unique_points_sha256"
                ],
                "source_float64_points_path": (
                    f"records[call_index={call_index}]."
                    "terminal_candidate_unique_points_f64_m"
                ),
                "display_points_f32_inspection_only": display.tolist(),
                "display_transform": {
                    "source_center_f64_m": center.tolist(),
                    "uniform_scale_inspection_only": scale,
                    "atlas_offset_inspection_only": offset.tolist(),
                },
                "tetra_sum_upper_bound_m3": bound[
                    "tetra_sum_upper_bound_m3"
                ],
                "upper_bound_is_zero": (
                    bound["tetra_sum_upper_bound_m3"] == 0
                ),
                "is_call29": call_index == CALL29_INDEX,
            }
        )
    atlas["failed_terminal_cloud_count"] = len(
        atlas["failed_terminal_clouds"]
    )
    atlas["failed_terminal_point_count"] = sum(
        row["source_point_count"] for row in atlas["failed_terminal_clouds"]
    )
    return evidence, atlas, csv_rows


PREPARED = {AUTHORITY.name, PREREG.name, PHASES.name}
PRE_WORKER = PREPARED | {
    INVOCATION.name, WORKER_AUTH.name, STDOUT.name, STDERR.name,
}
POST_WORKER = PRE_WORKER | {
    SENTINEL.name, PROGRESS.name, EVIDENCE.name, CSV_PATH.name, GEOMETRY.name,
    WORKER_CLAIM.name, SUPERVISOR.name,
}
POST_OBSERVE = POST_WORKER | {
    BOARD.name, LAYOUT.name, RRD.name, RBL.name, RERUN_VALIDATION.name,
    RERUN_SCREENSHOT.name, MANUAL_TEMPLATE.name, OBSERVABILITY.name,
}
FINAL = POST_OBSERVE | {MANUAL.name, COMPLETION.name}


def _write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "direction_index", "pair_index", "call_id", "adjacent", "stage",
        "direction", "stored_pass", "source", "actual_solver_positive",
        "ideal_exact_subset_certificate_positive",
        "hybrid_registered_gate_positive",
        "derived_volume_m3", "upper_bound_m3",
    ]
    with CSV_PATH.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _prepare() -> int:
    _require_inventory({AUTHORITY.name}, "before_prepare")
    authority = _read(AUTHORITY)
    authority_checks = _authority_checks(authority)
    frozen = _frozen_checks()
    d389 = _read(D389_EVIDENCE)
    d390 = _read(D390_EVIDENCE)
    schema = {
        "d389_pairs_36": len(
            d389["seam_numeric_provenance_audit"]["pair_results"]
        ) == PAIR_COUNT,
        "d390_failures_41": len(d390["failed_call_records"]) == FAILED_COUNT,
        "d390_geometry_41": _read(D390_GEOMETRY)["record_count"] == FAILED_COUNT,
        "d389_success_failure_103_41": (
            d390["d389_failed_call_manifest"][
                "total_directional_call_count"
            ]
            == DIRECTION_COUNT
            and d390["d389_failed_call_manifest"]["successful_call_count"]
            == SUCCESS_COUNT
            and d390["d389_failed_call_manifest"]["failed_call_count"]
            == FAILED_COUNT
        ),
        "call29_exact": d390["failed_call_records"][CALL29_INDEX]["call_id"]
        == CALL29_ID,
        "numpy_1_26_0": np.__version__ == "1.26.0",
        "rerun_0_34_1": importlib.metadata.version("rerun-sdk") == "0.34.1",
        "rerun_cli": RERUN_CLI.is_file(),
        "font": FONT.is_file(),
        "python_no_bytecode": sys.dont_write_bytecode,
    }
    if not all(authority_checks.values()):
        raise RuntimeError(f"D395 authority failed: {authority_checks}")
    if not all(frozen.values()):
        raise RuntimeError(f"D395 frozen input failed: {frozen}")
    if not all(schema.values()):
        raise RuntimeError(f"D395 schema failed: {schema}")
    _phase("prepare_start")
    prereg = {
        "artifact": "D395_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "question": (
            "Can a non-adopted hybrid table copy 103 immutable D389 actual "
            "solver Booleans and add 41 ideal exact-halfspace subset "
            "certificates while every failed Float64 solver result stays null?"
        ),
        "scientific_inputs": {
            _rel(path): digest for path, digest in INPUT_HASHES.items()
        },
        "denominator": {
            "pairs": 36,
            "directions_per_pair": 4,
            "directions": 144,
            "stored_successes": 103,
            "stored_failures": 41,
            "adjacent_pairs": 11,
            "nonadjacent_pairs": 25,
        },
        "method_contract": {
            "failed_call_upper_bound": (
                "sum exact absolute tetra volumes over all 4-point subsets"
            ),
            "monotone_subset_rule": (
                "only ideal exact halfspace intersections K subset conv(P) "
                "cannot increase volume"
            ),
            "certificate_scope": "EXACT_HALFSPACE_SUBSET_SEMANTICS_ONLY",
            "combined_table_authority": (
                "HYBRID_103_ACTUAL_SOLVER_PLUS_41_IDEAL_EXACT_SUBSET_"
                "CERTIFICATES"
            ),
            "stored103_are_exact_subset_certificates": False,
            "hybrid_table_adopted": False,
            "not_bit_exact_remaining_float64_clip_replay": True,
            "float64_roundoff_point_motion_outside_conv_p_bounded": False,
            "failed_actual_solver_positive_volume": None,
            "failed_actual_solver_result": None,
            "frozen_positive_volume_gate_m3": VOLUME_GATE,
            "failed_direction_derived_volume_m3": None,
            "call29_rank": None,
            "call29_class": None,
            "d389_mutation": False,
            "d389_solver_repair_or_adoption": False,
            "pair_classification_role": (
                "mixed-authority registered pre/post pattern only; not "
                "adopted; no Float32 causation, physics, or manufacturing "
                "interpretation"
            ),
            "gate_nonpositive_definition": (
                "not greater than the frozen 1e-18 m3 gate; not a claim of "
                "geometric zero"
            ),
        },
        "registered_expected_aggregate": {
            "failed_total_tetra_combinations": (
                EXPECTED_TOTAL_TETRA_COMBINATIONS
            ),
            "failed_positive_upper_bounds": EXPECTED_POSITIVE_BOUND_COUNT,
            "failed_zero_upper_bounds": EXPECTED_ZERO_BOUND_COUNT,
            "call29_upper_bound_m3_exact": EXPECTED_CALL29_BOUND,
            "maximum_upper_bound_call_index": EXPECTED_MAX_BOUND_CALL_INDEX,
            "maximum_upper_bound_m3_exact": EXPECTED_MAX_BOUND,
            "frozen_gate_positive_directions": 26,
            "frozen_gate_nonpositive_directions": 118,
        },
        "outcomes": {
            "pass": (
                "D395_HYBRID_103_ACTUAL_41_IDEAL_CERTIFICATE_TABLE_PASS_"
                "NO_SOLVER_REPAIR_NO_ADOPTION"
            ),
            "any_failed_upper_bound_above_gate": (
                "D395_FAILED_DIRECTION_REQUIRES_EXACT_CLIP_CONTINUATION_"
                "FAIL_STOP"
            ),
            "manifest_boolean_or_strict_disagreement": (
                "D395_HYBRID_REGISTERED_GATE_TABLE_FAIL_STOP"
            ),
            "any_actual_solver_result_or_volume_filled": (
                "D395_FLOAT64_SOLVER_NULL_CONTRACT_FAIL_STOP"
            ),
            "aggregate_or_extrema_mismatch": (
                "D395_REGISTERED_AGGREGATE_MISMATCH_FAIL_STOP"
            ),
        },
        "execution_contract": {
            "external_numeric_worker": 1,
            "worker_retries": 0,
            "process_signals": 0,
            "hard_watchdog": None,
            "viewer_maximum": 1,
            "viewer_retries": 0,
            "numeric_before_observability": True,
        },
        "forbidden": {
            "d389_clipping_rerun_or_mutation": 0,
            "remaining_float64_clip_replay": 0,
            "rank_class_evaluation_or_call29_adoption": 0,
            "derived_failed_direction_volume": None,
            "collider_usd_isaac_physx_warp_cuda": 0,
            "cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_settings": 0,
            "signals_hardware_commit_push": 0,
        },
        "authority": _artifact(AUTHORITY),
        "script": _artifact(SCRIPT),
        "start": _artifact(START),
        "authority_checks": authority_checks,
        "frozen_checks": frozen,
        "schema_checks": schema,
        "forward_only_output": _rel(OUT_DIR),
    }
    _write_json_x(PREREG, prereg)
    _phase("prepare_end", preregistration_sha256=_sha(PREREG))
    _require_inventory(PREPARED, "after_prepare")
    print(json.dumps({"prepared": True, "case": CASE}))
    return 0


def _worker_authorized() -> dict[str, bool]:
    authorization = _read(WORKER_AUTH)
    external = os.environ.get(WORKER_AUTHORITY_ENV)
    authority = _read(AUTHORITY)
    return {
        "external": external is not None and external == _sha(WORKER_AUTH),
        "parent": authorization.get("supervisor_pid") == os.getppid(),
        "worker_one": authorization.get("worker_invocation_index") == 1,
        "retry_zero": authorization.get("retry_index") == 0,
        "script": authorization.get("script_sha256") == _sha(SCRIPT),
        "prereg": authorization.get("preregistration_sha256") == _sha(PREREG),
        "authority": authorization.get("execution_authority_sha256")
        == _sha(AUTHORITY),
        "invocation": authorization.get("invocation_sha256") == _sha(INVOCATION),
        "outside_status": authority["git"]["status_outside_output_dir"]
        == _status_outside_output(),
        "authority_core": all(_authority_core_checks(authority).values()),
        "frozen": all(_frozen_checks().values()),
    }


def _worker() -> int:
    _require_inventory(PRE_WORKER, "worker_before_sentinel")
    checks = _worker_authorized()
    if not all(checks.values()):
        raise RuntimeError(f"D395 worker authorization failed: {checks}")
    started = time.monotonic()
    _write_json_x(
        SENTINEL,
        {
            "artifact": "D395_WORKER_START_SENTINEL_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "worker_pid": os.getpid(),
            "supervisor_pid": os.getppid(),
            "worker_invocation_index": 1,
            "retry_index": 0,
            "script_sha256": _sha(SCRIPT),
            "authorization_sha256": _sha(WORKER_AUTH),
            "wall_time_ns": time.time_ns(),
        },
    )
    _phase("worker_start", worker_pid=os.getpid())
    evidence, geometry, rows = _compute()
    _write_json_x(EVIDENCE, evidence)
    geometry["canonical_evidence_sha256"] = _sha(EVIDENCE)
    _write_json_x(GEOMETRY, geometry)
    _write_csv(rows)
    _phase("canonical_numeric_evidence_committed", evidence_sha256=_sha(EVIDENCE))
    frozen_after = _frozen_checks()
    claim_checks = {
        "numeric_pass": evidence["pass"] is True,
        "frozen_inputs_after": all(frozen_after.values()),
        "call29_null": evidence["call29_rank"] is None
        and evidence["call29_class"] is None,
        "derived_volume_null": evidence["derived_failed_direction_volumes_m3"]
        is None,
        "failed_actual_solver_null": (
            evidence["failed41_actual_solver_positive_volume"] is None
            and evidence["failed41_actual_solver_result"] is None
        ),
        "d389_not_modified": evidence["d389_modified"] is False,
        "d389_not_repaired_or_adopted": (
            evidence["d389_solver_repaired"] is False
            and evidence["d389_semantic_certificate_adopted"] is False
            and evidence["hybrid_table_adopted"] is False
        ),
        "progress_41": len(PROGRESS.read_text(encoding="utf-8").splitlines())
        == FAILED_COUNT,
        "csv_144": len(rows) == DIRECTION_COUNT,
    }
    claim = {
        "artifact": "D395_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "authorization_checks": checks,
        "checks": claim_checks,
        "artifacts": {
            "progress": _artifact(PROGRESS),
            "evidence": _artifact(EVIDENCE),
            "geometry": _artifact(GEOMETRY),
            "csv": _artifact(CSV_PATH),
        },
        "elapsed_seconds": time.monotonic() - started,
        "process_signals_sent": 0,
        "pass": all(claim_checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_end", worker_claim_pass=claim["pass"])
    if not claim["pass"]:
        return 1
    print(json.dumps(claim, ensure_ascii=False))
    return 0


def _run() -> int:
    _require_inventory(PREPARED, "before_run")
    authority = _read(AUTHORITY)
    authority_core = _authority_core_checks(authority)
    if not all(authority_core.values()):
        raise RuntimeError(
            f"D395 execution authority changed before worker: {authority_core}"
        )
    invocation = {
        "artifact": "D395_OFFLINE_WORKER_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": [
            sys.executable, "-B", _rel(SCRIPT), "--stage", "worker",
        ],
        "worker_invocation_index": 1,
        "retry_index": 0,
        "signals_authorized": 0,
        "hard_watchdog": None,
        "script_sha256": _sha(SCRIPT),
        "preregistration_sha256": _sha(PREREG),
        "execution_authority_sha256": _sha(AUTHORITY),
    }
    _write_json_x(INVOCATION, invocation)
    authorization = {
        "artifact": "D395_WORKER_AUTHORIZATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "supervisor_pid": os.getpid(),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "script_sha256": _sha(SCRIPT),
        "preregistration_sha256": _sha(PREREG),
        "execution_authority_sha256": _sha(AUTHORITY),
        "invocation_sha256": _sha(INVOCATION),
    }
    _write_json_x(WORKER_AUTH, authorization)
    _phase("supervisor_before_worker", supervisor_pid=os.getpid())
    started = time.monotonic()
    env = dict(os.environ)
    env[WORKER_AUTHORITY_ENV] = _sha(WORKER_AUTH)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with STDOUT.open("x", encoding="utf-8") as stdout, STDERR.open(
        "x", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            [sys.executable, "-B", str(SCRIPT), "--stage", "worker"],
            cwd=REPO, env=env, stdout=stdout, stderr=stderr, text=True,
        )
        returncode = process.wait()
    supervisor = {
        "artifact": "D395_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "actual_worker_invocations": 1,
        "retries": 0,
        "process_signals_sent": 0,
        "hard_watchdog": None,
        "worker_pid": process.pid,
        "worker_exited": True,
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "worker_claim_exists": WORKER_CLAIM.is_file(),
        "worker_claim_pass": (
            WORKER_CLAIM.is_file() and _read(WORKER_CLAIM).get("pass") is True
        ),
        "artifacts": (
            {
                "invocation": _artifact(INVOCATION),
                "worker_authorization": _artifact(WORKER_AUTH),
                "sentinel": _artifact(SENTINEL),
                "stdout": _artifact(STDOUT),
                "stderr": _artifact(STDERR),
                "worker_claim": _artifact(WORKER_CLAIM),
            }
            if WORKER_CLAIM.is_file() and SENTINEL.is_file()
            else None
        ),
    }
    supervisor["pass"] = (
        returncode == 0
        and supervisor["worker_claim_exists"]
        and supervisor["worker_claim_pass"]
    )
    _write_json_x(SUPERVISOR, supervisor)
    _phase("supervisor_after_worker", returncode=returncode, pass_=supervisor["pass"])
    if supervisor["pass"]:
        _require_inventory(POST_WORKER, "after_run")
    print(json.dumps(supervisor, ensure_ascii=False))
    return 0 if supervisor["pass"] else 1


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size)


def _render_board(evidence: dict[str, Any]) -> dict[str, Any]:
    image = Image.new("RGB", (1920, 1080), (247, 249, 252))
    draw = ImageDraw.Draw(image)
    title_font = _font(32)
    header_font = _font(20)
    body_font = _font(16)
    small_font = _font(14)
    draw.text(
        (28, 18),
        "D395 | HYBRID table: 103 actual + 41 ideal certificates",
        fill=(18, 28, 44), font=title_font,
    )
    draw.text(
        (30, 62),
        "Mixed authority, non-adopted | U>0:23, U=0:18, "
        "tetra subsets:42,928 | failed solver results:null",
        fill=(55, 68, 86), font=header_font,
    )
    colors = {
        "PRE_AND_POST_GATE_POSITIVE_REGISTERED_STREAM": (252, 221, 180),
        "POST_ONLY_GATE_POSITIVE_REGISTERED_STREAM": (255, 239, 166),
        "PRE_AND_POST_GATE_NONPOSITIVE_REGISTERED_STREAM": (211, 236, 222),
        "UNRESOLVED_REGISTERED_STREAM_FAIL_STOP": (246, 188, 188),
    }
    short = {
        "PRE_AND_POST_GATE_POSITIVE_REGISTERED_STREAM": "PRE+POST GATE +",
        "POST_ONLY_GATE_POSITIVE_REGISTERED_STREAM": "POST-ONLY GATE +",
        "PRE_AND_POST_GATE_NONPOSITIVE_REGISTERED_STREAM": "PRE+POST GATE NON+",
        "UNRESOLVED_REGISTERED_STREAM_FAIL_STOP": "UNRESOLVED",
    }
    boxes: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    for row in evidence["pair_semantics"]:
        index = row["pair_index"]
        col, line = index % 6, index // 6
        x0, y0 = 22 + col * 316, 104 + line * 142
        x1, y1 = x0 + 302, y0 + 132
        owner = (x0, y0, x1, y1)
        classification = row["hybrid_registered_gate_pair_pattern"]
        fill = colors[classification]
        draw.rounded_rectangle(owner, radius=9, fill=fill, outline=(92, 105, 120), width=2)
        pre = row["pre_float32"]["hybrid_registered_gate_positive_volume"]
        post = row["post_float32"]["hybrid_registered_gate_positive_volume"]
        certified = (
            row["pre_float32"]["certificate_direction_count"]
            + row["post_float32"]["certificate_direction_count"]
        )
        lines = [
            (
                f"#{index:02d} {row['target'][0]} "
                f"{row['left_index']}-{row['right_index']} "
                f"{'ADJ' if row['adjacent'] else 'NON'}",
                header_font, (25, 34, 48), y0 + 7,
            ),
            (
                f"pre {int(bool(pre))} -> post {int(bool(post))} | cert {certified}/4",
                body_font, (35, 46, 61), y0 + 43,
            ),
            (short[classification], body_font, (25, 55, 55), y0 + 70),
            ("failed solver/volume: null", small_font, (80, 58, 65), y0 + 99),
        ]
        for text, font, color, y in lines:
            draw.text((x0 + 10, y), text, font=font, fill=color)
            bbox = draw.textbbox((x0 + 10, y), text, font=font)
            boxes.append(
                {
                    "pair_index": index,
                    "bbox": list(bbox),
                    "owner": list(owner),
                    "inside_owner": (
                        bbox[0] >= x0 and bbox[1] >= y0
                        and bbox[2] <= x1 and bbox[3] <= y1
                    ),
                }
            )
        cards.append(
            {
                "pair_index": index,
                "bbox": list(owner),
                "classification": classification,
            }
        )
    bottom_y = 974
    draw.rectangle((0, bottom_y, 1920, 1080), fill=(25, 35, 52))
    draw.text(
        (28, 987),
        "Registered pattern: adjacent 2 pre+post, 9 post-only; "
        "nonadjacent 25 gate-nonpositive.",
        font=_font(18), fill=(244, 247, 251),
    )
    draw.text(
        (28, 1018),
        "Failed-41 certificate = ideal exact-halfspace subset only; remaining "
        "Float64 clip rounding was NOT replayed/bounded.",
        font=_font(15), fill=(255, 218, 144),
    )
    draw.text(
        (28, 1045),
        "Failed 41 actual solver result/volume remain null; call29 rank/class "
        "null; hybrid pattern NOT adopted; D389 unchanged; no physics claim.",
        font=_font(15), fill=(255, 218, 144),
    )
    image.save(BOARD)
    overlaps = []
    for left, right in itertools.combinations(boxes, 2):
        if left["pair_index"] != right["pair_index"]:
            continue
        a, b = left["bbox"], right["bbox"]
        if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
            overlaps.append([left["pair_index"], a, b])
    layout = {
        "artifact": "D395_BOARD_LAYOUT_VALIDATION_V1",
        "path": _rel(BOARD),
        "width": 1920,
        "height": 1080,
        "cards": cards,
        "card_count": len(cards),
        "text_box_count": len(boxes),
        "text_overlap_count": len(overlaps),
        "text_overlaps": overlaps,
        "all_text_inside_owner": all(row["inside_owner"] for row in boxes),
        "all_36_pair_indices_exact": [row["pair_index"] for row in cards]
        == list(range(36)),
        "pass": (
            len(cards) == 36
            and not overlaps
            and all(row["inside_owner"] for row in boxes)
        ),
    }
    _write_json_x(LAYOUT, layout)
    return layout


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    decision = rrb.Vertical(
        rrb.Spatial3DView(
            origin="/",
            contents="/d395/**",
            name="D395 terminal clouds (41) + auxiliary pair grid",
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=(7.2, -25.0, 24.0),
                look_target=(7.2, -6.0, -1.0),
                eye_up=(0.0, 0.0, 1.0),
            ),
            spatial_information=rrb.SpatialInformation(
                target_frame="tf#/", show_axes=True, show_bounding_box=False,
            ),
        ),
        rrb.TextDocumentView(
            origin="/metadata/run", contents="/metadata/run",
            name="Boolean authority and frozen nonclaims",
        ),
        row_shares=[0.78, 0.22],
    )
    buffer_view = rrb.Spatial3DView(
        origin="/", contents="/presentation/d395/notification_buffer/**",
        name="Notification buffer - no decision evidence",
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/", show_axes=False, show_bounding_box=False,
        ),
    )
    return rrb.Blueprint(
        rrb.Horizontal(decision, buffer_view, column_shares=[0.78, 0.22]),
        rrb.TimePanel(state=rrb.PanelState.Hidden),
        auto_layout=False, auto_views=False, collapse_panels=True,
    )


def _png_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "path": _rel(path), "width": image.width, "height": image.height,
            "mode": image.mode, "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }


def _write_rerun(evidence: dict[str, Any], atlas: dict[str, Any]) -> dict[str, Any]:
    import roarm_rl.rerun_contract as rerun_contract
    import roarm_rl.viz_debug as viz_debug

    color_by_class = {
        "PRE_AND_POST_GATE_POSITIVE_REGISTERED_STREAM": [235, 135, 45, 255],
        "POST_ONLY_GATE_POSITIVE_REGISTERED_STREAM": [235, 200, 45, 255],
        "PRE_AND_POST_GATE_NONPOSITIVE_REGISTERED_STREAM": [55, 170, 105, 255],
        "UNRESOLVED_REGISTERED_STREAM_FAIL_STOP": [220, 55, 65, 255],
    }
    points: list[dict[str, Any]] = []
    for row in atlas["failed_terminal_clouds"]:
        if row["is_call29"]:
            color = [220, 55, 75, 255]
        elif row["upper_bound_is_zero"]:
            color = [70, 135, 230, 255]
        else:
            color = [235, 145, 45, 255]
        count = row["source_point_count"]
        points.append(
            {
                "entity_path": (
                    f"d395/failed_terminal_sets/call{row['call_index']:02d}"
                ),
                "positions_m": row[
                    "display_points_f32_inspection_only"
                ],
                "radii": [0.07] * count,
                "colors": [color] * count,
                "labels": (
                    [f"C{row['call_index']:02d}"] + [""] * (count - 1)
                ),
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
    for row in atlas["pairs"]:
        points.append(
            {
                "entity_path": f"d395/pairs/p{row['pair_index']:02d}",
                "positions_m": [row["position"]],
                "radii": [0.34],
                "colors": [color_by_class[row["classification"]]],
                "labels": [
                    f"p{row['pair_index']:02d} {row['target'][0]} "
                    f"{row['left_index']}-{row['right_index']} "
                    f"{'A' if row['adjacent'] else 'N'}"
                ],
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
    metadata = {
        "00_decision_subject": (
            "41 registered D390 terminal point clouds; centered/scaled "
            "Float32 inspection copies bound to canonical Float64 hashes"
        ),
        "01_auxiliary": (
            "36-pair HYBRID registered pattern: actual solver 103 + ideal "
            "subset certificates 41; diagnostic and non-adopted"
        ),
        "02_certificate": (
            "ideal exact-halfspace subset semantics apply only to failed 41"
        ),
        "03_nonclaim": (
            "remaining Float64 clips not replayed; roundoff motion outside "
            "conv(P) not bounded; failed solver results/volumes remain null"
        ),
        "case": CASE,
        "attempt": ATTEMPT,
        "numeric_verdict": evidence["numeric_verdict"],
        "pairs": 36,
        "directions": 144,
        "failed_direction_derived_volume_m3": None,
        "call29_rank": None,
        "call29_class": None,
        "d389_modified": False,
        "viewer_role": (
            "terminal cloud atlas is the decision subject; pair grid is "
            "auxiliary mixed-authority pattern; all spatial values are "
            "Float32 inspection copies"
        ),
        "canonical_evidence_sha256": _sha(EVIDENCE),
        "canonical_d390_terminal_geometry_sha256": _sha(D390_GEOMETRY),
        "display_atlas_sha256": _sha(GEOMETRY),
        "g0a_pass": False,
    }
    expected_entities = ["metadata/run"] + [
        f"d395/failed_terminal_sets/call{index:02d}"
        for index in range(41)
    ] + [
        f"d395/pairs/p{index:02d}" for index in range(36)
    ]
    expected_entities = sorted(expected_entities)
    components = {"metadata/run": ["TextDocument:text"]}
    for index in range(41):
        components[f"d395/failed_terminal_sets/call{index:02d}"] = [
            "CoordinateFrame:frame", "Points3D:colors", "Points3D:labels",
            "Points3D:positions", "Points3D:radii",
        ]
    for index in range(36):
        components[f"d395/pairs/p{index:02d}"] = [
            "CoordinateFrame:frame", "Points3D:colors", "Points3D:labels",
            "Points3D:positions", "Points3D:radii",
        ]
    original_builder = viz_debug.build_rerun_blueprint
    original_runner = rerun_contract._run
    viewer_calls = 0

    def routed_builder(mode: str = "robot_geometry") -> Any:
        return _build_blueprint() if mode == "d395_pair_grid" else original_builder(mode)

    def no_signal_run(command: list[str], *, timeout_s: float) -> dict[str, Any]:
        nonlocal viewer_calls
        del timeout_s
        if any("screenshot" in str(part) for part in command):
            viewer_calls += 1
            if viewer_calls > 1:
                return {
                    "command": command, "returncode": None, "stdout": "",
                    "stderr": "D395 viewer maximum exceeded", "ok": False,
                }
        try:
            result = subprocess.run(
                command, check=False, capture_output=True, text=True,
            )
            return {
                "command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "ok": result.returncode == 0,
                "signals_sent": 0, "timeout_ignored_no_signal_authority": True,
            }
        except Exception as exc:
            return {
                "command": command, "returncode": None, "stdout": "",
                "stderr": repr(exc), "ok": False, "signals_sent": 0,
            }

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    rerun_contract._run = no_signal_run
    try:
        saved = viz_debug.log_rerun(
            RRD, points=points, recording_metadata=metadata,
            recording_id="g0a_d395_all36_gate_semantics",
            blueprint_path=RBL, blueprint_mode="d395_pair_grid",
            live_viewer=False, app_id="roarm_g0a_d395_gate_semantics",
        )
        if saved.get("ok") is not True:
            raise RuntimeError(f"D395 save-only Rerun failed: {saved}")
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
    aspect = (
        screenshot["width"] in {1920, 3840}
        and screenshot["height"] in {1080, 2160}
        and screenshot["width"] * 9 == screenshot["height"] * 16
    )
    base = validation.get("pass") is True
    validation["d395_contract"] = {
        "pair_entities": 36,
        "failed_terminal_cloud_entities": 41,
        "failed_terminal_points": atlas["failed_terminal_point_count"],
        "terminal_clouds_are_normalized_float32_inspection_copies": True,
        "canonical_float64_terminal_geometry_sha256": _sha(D390_GEOMETRY),
        "pair_grid_role": "AUXILIARY",
        "viewer_invocations": viewer_calls,
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "time_panel_hidden": True,
        "screenshot_aspect_pass": aspect,
    }
    validation["base_contract_pass"] = base
    validation["pass"] = (
        base
        and viewer_calls == 1
        and aspect
        and atlas["failed_terminal_cloud_count"] == 41
        and atlas["failed_terminal_point_count"] == 307
    )
    _write_json_x(RERUN_VALIDATION, validation)
    return {
        "pass": validation["pass"],
        "viewer_invocations": viewer_calls,
        "rrd": _artifact(RRD), "rbl": _artifact(RBL),
        "validation": _artifact(RERUN_VALIDATION),
        "screenshot": screenshot,
    }


def _observe() -> int:
    _require_inventory(POST_WORKER, "before_observe")
    authority_core = _authority_core_checks(_read(AUTHORITY))
    if not all(authority_core.values()):
        raise RuntimeError(
            f"D395 authority changed before observe: {authority_core}"
        )
    supervisor = _read(SUPERVISOR)
    worker = _read(WORKER_CLAIM)
    if supervisor.get("pass") is not True or worker.get("pass") is not True:
        raise RuntimeError("D395 numeric worker is not authoritative")
    if not all(_frozen_checks().values()):
        raise RuntimeError("D395 frozen input changed before observe")
    _phase("observability_start")
    started = time.monotonic()
    evidence = _read(EVIDENCE)
    atlas = _read(GEOMETRY)
    if atlas["canonical_evidence_sha256"] != _sha(EVIDENCE):
        raise RuntimeError("D395 evidence/atlas link changed")
    frozen_terminal_by_index = {
        row["call_index"]: row for row in _read(D390_GEOMETRY)["records"]
    }
    layout = _render_board(evidence)
    rerun = _write_rerun(evidence, atlas)
    template = {
        "artifact": "D395_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board_path": _rel(BOARD),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "artifact_hashes_to_bind_after_actual_viewing": {
            "board_sha256": _sha(BOARD),
            "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
        },
        "checks_to_record_after_actual_viewing": MANUAL_KEYS,
        "minimum_nonempty_observations": 3,
        "manual_inspection_complete": False,
    }
    _write_json_x(MANUAL_TEMPLATE, template)
    checks = {
        "numeric_evidence_pass": evidence["pass"] is True,
        "evidence_exact": _sha(EVIDENCE)
        == worker["artifacts"]["evidence"]["sha256"],
        "csv_exact": _sha(CSV_PATH) == worker["artifacts"]["csv"]["sha256"],
        "geometry_exact": _sha(GEOMETRY)
        == worker["artifacts"]["geometry"]["sha256"],
        "terminal_clouds_41_points_307": (
            atlas["failed_terminal_cloud_count"] == 41
            and atlas["failed_terminal_point_count"] == 307
        ),
        "terminal_cloud_source_hashes_exact": all(
            row["source_terminal_points_sha256"]
            == frozen_terminal_by_index[row["call_index"]][
                "terminal_candidate_unique_points_sha256"
            ]
            for row in atlas["failed_terminal_clouds"]
        ),
        "board_layout_pass": layout["pass"] is True,
        "board_exact_1920x1080": (
            _png_info(BOARD)["width"] == 1920
            and _png_info(BOARD)["height"] == 1080
        ),
        "rerun_pass": rerun["pass"] is True,
        "viewer_one": rerun["viewer_invocations"] == 1,
    }
    claim = {
        "artifact": "D395_OBSERVABILITY_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "numeric_artifacts": {
            "evidence": _artifact(EVIDENCE),
            "csv": _artifact(CSV_PATH),
            "geometry": _artifact(GEOMETRY),
            "worker_claim": _artifact(WORKER_CLAIM),
            "supervisor": _artifact(SUPERVISOR),
        },
        "artifacts": {
            "board": _artifact(BOARD), "layout": _artifact(LAYOUT),
            "rerun": rerun, "manual_template": _artifact(MANUAL_TEMPLATE),
        },
        "checks": checks,
        "viewer_invocations": rerun["viewer_invocations"],
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "elapsed_seconds": time.monotonic() - started,
        "pass": all(checks.values()),
    }
    _write_json_x(OBSERVABILITY, claim)
    _phase("observability_end", observability_pass=claim["pass"])
    if not claim["pass"]:
        raise RuntimeError(f"D395 observability failed: {checks}")
    _require_inventory(POST_OBSERVE, "after_observe")
    print(json.dumps(claim, ensure_ascii=False))
    return 0


def _finalize() -> int:
    _require_inventory(FINAL - {COMPLETION.name}, "before_finalize")
    _phase("finalize_start")
    evidence = _read(EVIDENCE)
    worker = _read(WORKER_CLAIM)
    supervisor = _read(SUPERVISOR)
    observability = _read(OBSERVABILITY)
    template = _read(MANUAL_TEMPLATE)
    manual = _read(MANUAL)
    invocation = _read(INVOCATION)
    worker_authorization = _read(WORKER_AUTH)
    phase_rows = [
        json.loads(line)
        for line in PHASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    phase_by_name = {row["phase"]: row for row in phase_rows}
    expected_hashes = {
        "board_sha256": _sha(BOARD),
        "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
    }
    manual_checks = manual.get("checks", {})
    rerun_refs = observability["artifacts"]["rerun"]
    checks = {
        "authority_core": all(
            _authority_core_checks(_read(AUTHORITY)).values()
        ),
        "frozen_inputs": all(_frozen_checks().values()),
        "outside_status": _read(AUTHORITY)["git"]["status_outside_output_dir"]
        == _status_outside_output(),
        "numeric_pass": evidence.get("pass") is True,
        "worker_pass": worker.get("pass") is True,
        "supervisor_pass": supervisor.get("pass") is True,
        "observability_pass": observability.get("pass") is True,
        "worker_artifact_chain": (
            _artifact_matches(worker["artifacts"]["progress"], PROGRESS)
            and
            _artifact_matches(worker["artifacts"]["evidence"], EVIDENCE)
            and _artifact_matches(worker["artifacts"]["csv"], CSV_PATH)
            and _artifact_matches(worker["artifacts"]["geometry"], GEOMETRY)
        ),
        "preregistration_embedded_links": (
            invocation["preregistration_sha256"] == _sha(PREREG)
            and worker_authorization["preregistration_sha256"]
            == _sha(PREREG)
            and phase_by_name["prepare_end"]["preregistration_sha256"]
            == _sha(PREREG)
        ),
        "execution_authority_embedded_links": (
            invocation["script_sha256"] == _sha(SCRIPT)
            and invocation["execution_authority_sha256"] == _sha(AUTHORITY)
            and worker_authorization["script_sha256"] == _sha(SCRIPT)
            and worker_authorization["execution_authority_sha256"]
            == _sha(AUTHORITY)
            and worker_authorization["invocation_sha256"]
            == _sha(INVOCATION)
        ),
        "numeric_phase_evidence_link": (
            phase_by_name["canonical_numeric_evidence_committed"][
                "evidence_sha256"
            ]
            == _sha(EVIDENCE)
        ),
        "supervisor_artifact_chain": (
            _artifact_matches(
                supervisor["artifacts"]["invocation"], INVOCATION
            )
            and _artifact_matches(
                supervisor["artifacts"]["worker_authorization"], WORKER_AUTH
            )
            and _artifact_matches(
                supervisor["artifacts"]["sentinel"], SENTINEL
            )
            and _artifact_matches(
                supervisor["artifacts"]["stdout"], STDOUT
            )
            and _artifact_matches(
                supervisor["artifacts"]["stderr"], STDERR
            )
            and _artifact_matches(
                supervisor["artifacts"]["worker_claim"], WORKER_CLAIM
            )
        ),
        "observability_numeric_chain": (
            _artifact_matches(
                observability["numeric_artifacts"]["evidence"], EVIDENCE
            )
            and _artifact_matches(
                observability["numeric_artifacts"]["csv"], CSV_PATH
            )
            and _artifact_matches(
                observability["numeric_artifacts"]["geometry"], GEOMETRY
            )
            and _artifact_matches(
                observability["numeric_artifacts"]["worker_claim"],
                WORKER_CLAIM,
            )
            and _artifact_matches(
                observability["numeric_artifacts"]["supervisor"], SUPERVISOR
            )
        ),
        "observability_artifact_chain": (
            _artifact_matches(observability["artifacts"]["board"], BOARD)
            and _artifact_matches(observability["artifacts"]["layout"], LAYOUT)
            and _artifact_matches(
                observability["artifacts"]["manual_template"],
                MANUAL_TEMPLATE,
            )
            and _artifact_matches(rerun_refs["rrd"], RRD)
            and _artifact_matches(rerun_refs["rbl"], RBL)
            and _artifact_matches(
                rerun_refs["validation"], RERUN_VALIDATION
            )
            and _artifact_matches(
                rerun_refs["screenshot"], RERUN_SCREENSHOT
            )
        ),
        "manual_identity": manual.get("artifact")
        == "D395_MANUAL_VISUAL_INSPECTION_V1"
        and manual.get("case") == CASE and manual.get("attempt") == ATTEMPT,
        "manual_paths": manual.get("board_path") == _rel(BOARD)
        and manual.get("rerun_screenshot_path") == _rel(RERUN_SCREENSHOT),
        "manual_keys": set(manual_checks) == set(MANUAL_KEYS),
        "manual_all_true": set(manual_checks) == set(MANUAL_KEYS)
        and all(value is True for value in manual_checks.values()),
        "manual_hashes": manual.get("artifact_hashes") == expected_hashes
        == template["artifact_hashes_to_bind_after_actual_viewing"],
        "manual_links": manual.get("manual_template_sha256")
        == _sha(MANUAL_TEMPLATE)
        and manual.get("observability_claim_sha256") == _sha(OBSERVABILITY),
        "manual_observations": isinstance(manual.get("observations"), list)
        and len(manual["observations"]) >= 3
        and all(
            isinstance(value, str) and value.strip()
            for value in manual["observations"]
        ),
        "manual_complete": manual.get("manual_inspection_complete") is True,
        "worker_one_retry_signal_zero": (
            supervisor["actual_worker_invocations"] == 1
            and supervisor["retries"] == 0
            and supervisor["process_signals_sent"] == 0
        ),
        "viewer_one_retry_zero": (
            observability["viewer_invocations"] == 1
            and observability["viewer_retries"] == 0
        ),
        "call29_null": evidence["call29_rank"] is None
        and evidence["call29_class"] is None,
        "derived_volume_null": evidence["derived_failed_direction_volumes_m3"]
        is None,
        "failed_actual_solver_null": (
            evidence["failed41_actual_solver_positive_volume"] is None
            and evidence["failed41_actual_solver_result"] is None
        ),
        "d389_immutable": evidence["d389_modified"] is False
        and evidence["d389_retroactive_pass"] is False
        and evidence["d389_solver_repaired"] is False
        and evidence["d389_semantic_certificate_adopted"] is False
        and evidence["hybrid_table_adopted"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"D395 finalize prechecks failed: {checks}")
    _phase("finalize_end")
    phase_contract = _phase_contract()
    if not all(
        phase_contract[key]
        for key in ("exact", "ordinals_exact", "monotonic_forward", "wall_forward")
    ):
        raise RuntimeError(f"D395 phase contract failed: {phase_contract}")
    completion = {
        "artifact": "D395_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": checks,
        "phase_contract": phase_contract,
        "artifacts": {
            "script": _artifact(SCRIPT), "start": _artifact(START),
            "authority": _artifact(AUTHORITY), "preregistration": _artifact(PREREG),
            "invocation": _artifact(INVOCATION), "worker_authorization": _artifact(WORKER_AUTH),
            "sentinel": _artifact(SENTINEL), "progress": _artifact(PROGRESS),
            "evidence": _artifact(EVIDENCE), "csv": _artifact(CSV_PATH),
            "geometry": _artifact(GEOMETRY), "worker_claim": _artifact(WORKER_CLAIM),
            "supervisor": _artifact(SUPERVISOR), "board": _artifact(BOARD),
            "layout": _artifact(LAYOUT), "rrd": _artifact(RRD),
            "rbl": _artifact(RBL), "rerun_validation": _artifact(RERUN_VALIDATION),
            "rerun_screenshot": _artifact(RERUN_SCREENSHOT),
            "manual_template": _artifact(MANUAL_TEMPLATE),
            "observability": _artifact(OBSERVABILITY), "manual": _artifact(MANUAL),
            "phases": _artifact(PHASES),
        },
        "numeric_verdict": evidence["numeric_verdict"],
        "operational_verdict": (
            "D395_HYBRID_103_ACTUAL_41_IDEAL_CERTIFICATE_TABLE_COMPLETE_"
            "NO_FLOAT64_SOLVER_REPAIR_NO_ADOPTION"
        ),
        "pairs": 36,
        "directions": 144,
        "stored_successes": 103,
        "ideal_exact_subset_certified_failures": 41,
        "actual_float64_solver_failures_remaining": 41,
        "actual_float64_solver_successes_inherited": 103,
        "combined_table_authority": (
            "HYBRID_103_ACTUAL_SOLVER_PLUS_41_IDEAL_EXACT_SUBSET_"
            "CERTIFICATES"
        ),
        "hybrid_table_adopted": False,
        "failed_total_tetra_combinations": EXPECTED_TOTAL_TETRA_COMBINATIONS,
        "failed_positive_upper_bounds": EXPECTED_POSITIVE_BOUND_COUNT,
        "failed_zero_upper_bounds": EXPECTED_ZERO_BOUND_COUNT,
        "certificate_scope": "EXACT_HALFSPACE_SUBSET_SEMANTICS_ONLY",
        "remaining_float64_clip_replayed": False,
        "float64_roundoff_outside_conv_p_bounded": False,
        "call29_rank": None,
        "call29_class": None,
        "derived_failed_direction_volume_m3": None,
        "d389_modified": False,
        "actual_worker_invocations": 1,
        "worker_retries": 0,
        "viewer_invocations": 1,
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "g0a_pass": False,
        "pass": True,
    }
    _write_json_x(COMPLETION, completion)
    _require_inventory(FINAL, "after_finalize")
    print(json.dumps(completion, ensure_ascii=False))
    return 0


def _record_failure(stage: str, exc: BaseException) -> None:
    if FAILURE.exists():
        return
    try:
        _write_json_x(
            FAILURE,
            {
                "artifact": "D395_FAILURE_ATTESTATION_V1",
                "case": CASE, "attempt": ATTEMPT, "stage": stage,
                "error_type": type(exc).__name__, "error": str(exc),
                "worker_started": SENTINEL.exists(),
                "process_signals_sent": 0,
                "call29_rank": None, "call29_class": None,
                "derived_volume_m3": None, "d389_modified": False,
                "wall_time_ns": time.time_ns(),
            },
        )
    except Exception:
        pass


def _dispatch(stage: str) -> int:
    if stage == "prepare":
        return _prepare()
    if stage == "run":
        return _run()
    if stage == "worker":
        return _worker()
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
        if OUT_DIR.exists():
            _record_failure(args.stage, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
