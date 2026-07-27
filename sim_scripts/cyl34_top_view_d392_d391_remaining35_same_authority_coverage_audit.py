#!/usr/bin/env python3
"""D392: apply the frozen D391 rank authority to D390's remaining 35 calls.

This is an offline-only, forward-only case.  It imports the frozen D391
implementation by exact SHA-256 and reuses its rank, translation, and scale
functions without copying or changing the numerical policy.

The D391 exhaustive permutation control is only tractable for small point
sets.  D392 includes sets with up to 25 points, so it preregisters a different
but explicit order-invariance contract:

* prove from the exact frozen source that rank input is sorted unique rows;
* independently reconstruct that canonical row set bit-exactly;
* reject numerically-equal rows with different Float64 encodings;
* exhaust all permutations for n <= 6; and
* for larger n, check reversal, every cyclic rotation, and every adjacent
  transposition while recording that this is not an n! enumeration.

The numerical worker runs once with no retry.  Presentation is a later host
stage so a Viewer failure cannot erase or weaken committed numerical evidence.
Isaac, PhysX, USD, colliders, cylinders, physics, q5, contact, and grasp are
never imported or executed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
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


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASE = "D392"
ATTEMPT = "attempt1_d391_remaining35_same_authority_coverage_audit"
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d392"
    / ATTEMPT
)
START = REPO / "START_HERE.md"
D390_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d390"
    / "attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization"
)
D390_GEOMETRY = D390_DIR / "d390_terminal_candidate_geometry.json"
D391_SCRIPT = (
    REPO
    / "sim_scripts"
    / "cyl34_top_view_d391_d390_rank_basis_and_clip_input_immutability_repair.py"
)
D391_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d391"
    / "attempt1_d390_rank_basis_and_clip_input_immutability_repair"
)
D391_EVIDENCE = D391_DIR / "d391_rank_and_plane_immutability_evidence.json"
D391_EXECUTION_AUTHORITY = D391_DIR / "d391_execution_authority.json"
D391_PREREGISTRATION = D391_DIR / "d391_preregistration.json"
D391_COMPLETION = D391_DIR / "d391_completion_summary.json"

EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
EXPECTED_START_SHA256 = (
    "ed9545eee3d01180b07df2b7822ceb156990ff71a1884590cb9a0fbad43be48d"
)
EXPECTED_D390_GEOMETRY_SHA256 = (
    "73fc986043b976bec26e1cc92643b8aab281a529f1c71c2918163ba7b98475c7"
)
EXPECTED_D390_DIRECTORY_MANIFEST_SHA256 = (
    "8ceb1aa2b3d8ec6f543d4f9bccadb363164c2422a06285dfffdb3932d535209e"
)
EXPECTED_D391_SCRIPT_SHA256 = (
    "2e92e03bd7174622c3e010a88c23d84cb70159bcb7265876a526ec81e99b37b1"
)
EXPECTED_D391_DIRECTORY_MANIFEST_SHA256 = (
    "b39d874c39e6a4303c2ee4013855e378016a3e9154403061192c163fca83080e"
)
EXPECTED_D391_EVIDENCE_SHA256 = (
    "d76bbd88b2a6f188f9c46c382adc7292e56267e6e4a3beb8b9004b70e34b80cc"
)
EXPECTED_D391_EXECUTION_AUTHORITY_SHA256 = (
    "af148b1a392399f9918ad01b62fd7bf2839056ab8fea57c2442985738a15958f"
)
EXPECTED_D391_PREREGISTRATION_SHA256 = (
    "0617fd65d747fbd9695b076e5323b5c15283d8e5b0826b7b6861f150cde6e618"
)
EXPECTED_D391_COMPLETION_SHA256 = (
    "9d09e55e6cf7cb7e6d60b1fb1e7722dff09a4da3487e299f3304ad4310d085ea"
)
EXPECTED_REMAINING35_MANIFEST_SHA256 = (
    "a77a6e200f63f4ff0a58e576370ef1d29c4e8de32d2f8955e717b46aeb955870"
)
EXPECTED_D391_FUNCTION_SOURCE_SHA256 = {
    "_array_sha": (
        "2405dfba3b5b6355c40bdbe2d2f0adf694ed965c39328b3c4f16cf3ff58619f0"
    ),
    "_canonical_sha": (
        "f25006d64366255b1d442bf3d3ccd2a112fa0bbe844ff3325b76380788fca90e"
    ),
    "_canonical_unique": (
        "5fb3a72690bb81f15ce06974acd4cefc98564e9297504dcc3c86aec5449e294d"
    ),
    "_rank_core": (
        "b4dbc59d6aafc9ae27b7044e1252f4e5b5d38ac32d52cc8672894a95d3ce30c0"
    ),
    "_rank_signature": (
        "c1e9a1f37c0e38fab5dfe141fbd71b38a3dbcb3af1f2d66db91ccf725d56fac0"
    ),
    "_scale_controls": (
        "de531d0f69458326e1636cf709084a1c242622a2a81f09dc635cd98d211fbd5d"
    ),
    "_translation_control": (
        "a3f747ae3f7b595f852aad0f57fc60b281eadb9d4eacba1c5f123fff4a36261c"
    ),
}
EXPECTED_APPROVAL_MARKER = "## Active Case — D392 Approved, Not Yet Executed"
EXPECTED_USER_APPROVAL_NORMALIZED = (
    "D390의 나머지 35건에 동결된 D391 기준 적용 → call29의 미세한 "
    "세 번째 방향 발생 지점 국소화 → D389 미확정 seam 반영 → 그 "
    "뒤 충돌체·USD·Isaac/PhysX·29×50mm 원통 물리 검토; 모두 순차 승인"
)
EXECUTION_AUTHORITY_SHA256_ENV = "D392_EXECUTION_AUTHORITY_SHA256"
WORKER_AUTHORIZATION_SHA256_ENV = "D392_WORKER_AUTHORIZATION_SHA256"

VARIABLES = [
    "d390_remaining35_frozen_d391_rank_authority_evaluation_set_v1",
    "factorial_free_canonical_order_invariance_proof_v1",
]
MANUAL_CHECK_KEYS = [
    "board_exact_35_rows_visible_18_plus_17",
    "board_full_call_ids_readable",
    "board_no_text_overlap_or_clipping",
    "board_summary_matches_canonical_json",
    "rerun_all_35_atlas_groups_visible",
    "rerun_class_colors_and_axes_distinguishable",
    "rerun_metadata_and_nonclaims_readable",
    "rerun_notification_does_not_obscure_decision_subject",
    "rerun_time_panel_hidden",
    "observed_result_matches_numeric_evidence",
]
DISPUTED_CALL_INDICES = (3, 7, 9, 12, 27, 29)
SMALL_EXHAUSTIVE_MAX_POINTS = 6
COOPERATIVE_DEADLINE_SECONDS = 300.0
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

EXECUTION_AUTHORITY = OUT_DIR / "d392_execution_authority.json"
PREREGISTRATION = OUT_DIR / "d392_preregistration.json"
PHASES = OUT_DIR / "d392_phase_markers.jsonl"
INVOCATION = OUT_DIR / "d392_offline_worker_invocation.json"
AUTHORIZATION = OUT_DIR / "d392_worker_authorization.json"
SENTINEL = OUT_DIR / "d392_worker_start_sentinel.json"
STDOUT = OUT_DIR / "d392_offline_worker_stdout.log"
STDERR = OUT_DIR / "d392_offline_worker_stderr.log"
PROGRESS = OUT_DIR / "d392_call_progress.jsonl"
EVIDENCE = OUT_DIR / "d392_remaining35_rank_evidence.json"
GEOMETRY = OUT_DIR / "d392_remaining35_display_geometry.json"
CSV_PATH = OUT_DIR / "d392_remaining35_rank_catalog.csv"
WORKER_CLAIM = OUT_DIR / "d392_offline_worker_claim.json"
WORKER_FAILURE_CLAIM = OUT_DIR / "d392_worker_failure_claim.json"
SUPERVISOR = OUT_DIR / "d392_offline_worker_supervisor.json"
BOARD = OUT_DIR / "d392_remaining35_rank_coverage_1920x1080.png"
LAYOUT = OUT_DIR / "d392_board_layout_validation.json"
RRD = OUT_DIR / "d392_remaining35_rank_coverage.rerun.rrd"
RBL = OUT_DIR / "d392_remaining35_rank_coverage.rerun.rbl"
RERUN_VALIDATION = OUT_DIR / "d392_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d392_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d392_manual_visual_inspection_template.json"
OBSERVABILITY_CLAIM = OUT_DIR / "d392_observability_claim.json"
MANUAL = OUT_DIR / "d392_manual_visual_inspection.json"
FAILURE = OUT_DIR / "d392_failure_attestation.json"
COMPLETION = OUT_DIR / "d392_completion_summary.json"

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
POST_OBSERVABILITY_NAMES = POST_WORKER_NAMES | {
    BOARD.name,
    LAYOUT.name,
    RRD.name,
    RBL.name,
    RERUN_VALIDATION.name,
    RERUN_SCREENSHOT.name,
    MANUAL_TEMPLATE.name,
    OBSERVABILITY_CLAIM.name,
}
PRE_FINALIZE_NAMES = POST_OBSERVABILITY_NAMES | {MANUAL.name}
POST_FINALIZE_NAMES = PRE_FINALIZE_NAMES | {COMPLETION.name}

_deadline_monotonic: float | None = None
_worker_started = False
_worker_pid: int | None = None


class CooperativeDeadlineExceeded(RuntimeError):
    """Raised by this process; no external timeout or signal is authorized."""


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
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _native(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


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
    value = _git("status", "--porcelain=v1", "--untracked-files=all")
    return value.splitlines() if value else []


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
    # Ignored artifacts (notably *.log and *.png) do not appear in Git
    # porcelain.  Exact disk inventory is enforced separately by
    # _require_names(); this gate only rejects visible Git changes outside the
    # registered attempt.
    return all(
        line.startswith("?? ") and _status_path(line) in allowed_paths
        for line in extras
    )


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
    return {item.name for item in OUT_DIR.iterdir()}


def _require_names(expected: set[str], stage: str) -> None:
    observed = _out_names()
    if observed != expected:
        raise RuntimeError(
            f"D392 output inventory mismatch at {stage}: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _phase(name: str, **details: Any) -> None:
    _append_jsonl(
        PHASES,
        {
            "phase": name,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **details,
        },
    )


def _deadline(label: str) -> None:
    if (
        _deadline_monotonic is not None
        and time.monotonic() > _deadline_monotonic
    ):
        raise CooperativeDeadlineExceeded(
            f"D392 cooperative deadline exceeded at {label}"
        )


def _direct_import_roots(path: Path = SCRIPT) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _load_d391() -> Any:
    if _sha(D391_SCRIPT) != EXPECTED_D391_SCRIPT_SHA256:
        raise RuntimeError("frozen D391 script SHA-256 changed")
    spec = importlib.util.spec_from_file_location(
        "d391_frozen_rank_authority", D391_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen D391 script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = {
        name: _text_sha(inspect.getsource(getattr(module, name)))
        for name in EXPECTED_D391_FUNCTION_SOURCE_SHA256
    }
    if observed != EXPECTED_D391_FUNCTION_SOURCE_SHA256:
        raise RuntimeError(
            f"frozen D391 function bundle changed: {observed}"
        )
    return module


D391: Any | None = None


def _d391() -> Any:
    if D391 is None:
        raise RuntimeError("frozen D391 authority was not loaded")
    return D391


def _frozen_checks() -> dict[str, bool]:
    return {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "start_exact": _sha(START) == EXPECTED_START_SHA256,
        "approval_marker_present": (
            EXPECTED_APPROVAL_MARKER
            in START.read_text(encoding="utf-8")
        ),
        "d390_geometry_exact": (
            _sha(D390_GEOMETRY) == EXPECTED_D390_GEOMETRY_SHA256
        ),
        "d390_directory_manifest_exact": (
            _canonical_sha(_directory_manifest(D390_DIR))
            == EXPECTED_D390_DIRECTORY_MANIFEST_SHA256
        ),
        "d391_script_exact": (
            _sha(D391_SCRIPT) == EXPECTED_D391_SCRIPT_SHA256
        ),
        "d391_directory_manifest_exact": (
            _canonical_sha(_directory_manifest(D391_DIR))
            == EXPECTED_D391_DIRECTORY_MANIFEST_SHA256
        ),
        "d391_evidence_exact": (
            _sha(D391_EVIDENCE) == EXPECTED_D391_EVIDENCE_SHA256
        ),
        "d391_execution_authority_exact": (
            _sha(D391_EXECUTION_AUTHORITY)
            == EXPECTED_D391_EXECUTION_AUTHORITY_SHA256
        ),
        "d391_preregistration_exact": (
            _sha(D391_PREREGISTRATION)
            == EXPECTED_D391_PREREGISTRATION_SHA256
        ),
        "d391_completion_exact": (
            _sha(D391_COMPLETION)
            == EXPECTED_D391_COMPLETION_SHA256
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
            == "D392_EXTERNAL_EXECUTION_AUTHORITY_V1"
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
                "conditional sequential approval; D392 is the only active "
                "case and later cases require prior-result freeze plus "
                "their own one-or-two-variable preregistration"
            )
            and authority.get("approval", {}).get("active_case") == CASE
            and authority.get("approval", {}).get("new_variables")
            == VARIABLES
        ),
        "script_exact": (
            authority.get("script", {}).get("path") == _rel(SCRIPT)
            and authority.get("script", {}).get("sha256") == _sha(SCRIPT)
        ),
        "start_exact": (
            authority.get("start", {}).get("path") == _rel(START)
            and authority.get("start", {}).get("sha256") == _sha(START)
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
        "d390_d391_manifests_exact": (
            authority.get("git", {}).get(
                "status_with_execution_authority_sha256"
            )
            == _text_sha("\n".join(baseline))
            and authority.get("inputs", {}).get(
                "d390_directory_manifest_sha256"
            )
            == EXPECTED_D390_DIRECTORY_MANIFEST_SHA256
            and authority.get("inputs", {}).get(
                "d391_directory_manifest_sha256"
            )
            == EXPECTED_D391_DIRECTORY_MANIFEST_SHA256
        ),
        "input_hashes_exact": (
            authority.get("inputs", {}).get("d390_geometry_sha256")
            == EXPECTED_D390_GEOMETRY_SHA256
            and authority.get("inputs", {}).get("d391_script_sha256")
            == EXPECTED_D391_SCRIPT_SHA256
            and authority.get("inputs", {}).get("d391_evidence_sha256")
            == EXPECTED_D391_EVIDENCE_SHA256
        ),
        "output_exact": (
            authority.get("output", {}).get("path") == _rel(OUT_DIR)
            and authority.get("output", {}).get("forward_only") is True
        ),
    }


def _preregistered_chain_checks() -> dict[str, bool]:
    if not PREREGISTRATION.is_file() or not EXECUTION_AUTHORITY.is_file():
        return {
            "preregistration_exists": False,
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
    prepare_end_rows = [
        row for row in phase_rows
        if row.get("phase") == "prepare_end"
    ]
    return {
        "preregistration_exists": True,
        "execution_authority_exists": True,
        "case_attempt_exact": (
            prereg.get("case") == CASE
            and prereg.get("attempt") == ATTEMPT
        ),
        "variables_exact": prereg.get("new_variables") == VARIABLES,
        "script_path_sha_exact": (
            case_authority.get("script_path") == _rel(SCRIPT)
            and case_authority.get("script_sha256") == _sha(SCRIPT)
            == authority.get("script", {}).get("sha256")
        ),
        "start_path_sha_exact": (
            case_authority.get("start_path") == _rel(START)
            and case_authority.get("start_sha256") == _sha(START)
            == authority.get("start", {}).get("sha256")
            == EXPECTED_START_SHA256
        ),
        "execution_authority_path_sha_exact": (
            case_authority.get("execution_authority_path")
            == _rel(EXECUTION_AUTHORITY)
            and case_authority.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
        ),
        "authority_approval_exact": (
            authority.get("approval", {}).get("normalized_user_text")
            == EXPECTED_USER_APPROVAL_NORMALIZED
            and authority.get("approval", {}).get("new_variables")
            == VARIABLES
        ),
        "prepare_end_preregistration_hash_exact": (
            len(prepare_end_rows) == 1
            and prepare_end_rows[0].get("preregistration_sha256")
            == _sha(PREREGISTRATION)
        ),
        "d390_d391_input_hashes_exact": (
            case_authority.get("d390_geometry_sha256")
            == _sha(D390_GEOMETRY)
            == EXPECTED_D390_GEOMETRY_SHA256
            and case_authority.get("d391_script_sha256")
            == _sha(D391_SCRIPT)
            == EXPECTED_D391_SCRIPT_SHA256
            and case_authority.get("d391_evidence_sha256")
            == _sha(D391_EVIDENCE)
            == EXPECTED_D391_EVIDENCE_SHA256
        ),
        "authority_structure_still_valid": all(
            _execution_authority_checks(
                authority, exact_status=False
            ).values()
        ),
    }


def _remaining_manifest(
    geometry: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in geometry["records"]:
        if int(source["call_index"]) in DISPUTED_CALL_INDICES:
            continue
        rows.append(
            {
                "call_index": int(source["call_index"]),
                "call_id": str(source["call_id"]),
                "unique_point_count": len(
                    source["terminal_candidate_unique_points_f64_m"]
                ),
                "points_sha256": str(
                    source["terminal_candidate_unique_points_sha256"]
                ),
            }
        )
    return rows


def _d391_six_identity_checks(
    geometry: dict[str, Any]
) -> dict[str, bool]:
    by_index = {
        int(row.get("call_index", -1)): row
        for row in geometry.get("records", [])
    }
    expected = list(_d391().DISPUTED_MANIFEST)
    evidence = _read_json(D391_EVIDENCE)
    observed = evidence.get("disputed_records", [])
    observed_by_index = {
        int(row.get("call_index", -1)): row for row in observed
    }
    expected_indices = [
        int(row["call_index"]) for row in expected
    ]
    expected_join: list[bool] = []
    evidence_join: list[bool] = []
    for row in expected:
        index = int(row["call_index"])
        source = by_index.get(index, {})
        result = observed_by_index.get(index, {})
        expected_join.append(
            source.get("call_id") == row["call_id"]
            and source.get("terminal_candidate_unique_points_sha256")
            == row["points_sha256"]
            and len(
                source.get(
                    "terminal_candidate_unique_points_f64_m", []
                )
            )
            == row["unique_point_count"]
        )
        evidence_join.append(
            result.get("call_id") == row["call_id"]
            and result.get("points_sha256") == row["points_sha256"]
            and result.get("unique_point_count")
            == row["unique_point_count"]
        )
    stable = [
        row
        for row in observed
        if row.get("rank_authority", {}).get("status", "").startswith(
            "STABLE"
        )
    ]
    ambiguous = [
        row
        for row in observed
        if row.get("rank_authority", {}).get("status", "").startswith(
            "NUMERICALLY_AMBIGUOUS"
        )
    ]
    call29 = ambiguous[0] if len(ambiguous) == 1 else {}
    call29_rank = call29.get("rank_authority", {})
    return {
        "frozen_manifest_indices_exact": (
            expected_indices == list(DISPUTED_CALL_INDICES)
        ),
        "d390_index_id_point_sha_count_join_exact_6": all(
            expected_join
        )
        and len(expected_join) == 6,
        "d391_evidence_index_id_point_sha_count_join_exact_6": all(
            evidence_join
        )
        and len(evidence_join) == 6,
        "d391_evidence_indices_exact": (
            sorted(observed_by_index) == list(DISPUTED_CALL_INDICES)
        ),
        "d391_five_stable": len(stable) == 5,
        "d391_only_ambiguity_is_call29": (
            len(ambiguous) == 1
            and call29.get("call_index") == 29
            and call29.get("call_id")
            == "lower_01_02_pre_float32_lbr"
            and call29_rank.get("status")
            == "NUMERICALLY_AMBIGUOUS_BASIS"
            and call29_rank.get("authoritative_rank") is None
            and call29_rank.get("authoritative_class") is None
        ),
        "d391_numeric_evidence_pass": (
            evidence.get("numeric_pass") is True
        ),
    }


def _input_schema_checks(
    geometry: dict[str, Any]
) -> dict[str, bool]:
    records = geometry.get("records", [])
    indices = [int(row.get("call_index", -1)) for row in records]
    manifest = _remaining_manifest(geometry) if len(records) == 41 else []
    disputed = [
        row for row in records
        if int(row.get("call_index", -1)) in DISPUTED_CALL_INDICES
    ]
    source_hash_checks: list[bool] = []
    for row in records:
        points = np.asarray(
            row.get("terminal_candidate_unique_points_f64_m", []),
            dtype=np.float64,
        ).reshape(-1, 3)
        source_hash_checks.append(
            bool(np.isfinite(points).all())
            and _d391()._array_sha(points)
            == row.get("terminal_candidate_unique_points_sha256")
        )
    d391_identity = _d391_six_identity_checks(geometry)
    return {
        "artifact_exact": (
            geometry.get("artifact")
            == "D390_TERMINAL_CANDIDATE_GEOMETRY_V1"
        ),
        "record_count_41": (
            geometry.get("record_count") == len(records) == 41
        ),
        "indices_exact_0_to_40": indices == list(range(41)),
        "remaining_count_35": len(manifest) == 35,
        "disputed_count_6": len(disputed) == 6,
        "sets_disjoint": (
            not set(row["call_index"] for row in manifest)
            .intersection(DISPUTED_CALL_INDICES)
        ),
        "sets_union_0_to_40": (
            sorted(
                [row["call_index"] for row in manifest]
                + list(DISPUTED_CALL_INDICES)
            )
            == list(range(41))
        ),
        "remaining_manifest_exact": (
            _canonical_sha(manifest)
            == EXPECTED_REMAINING35_MANIFEST_SHA256
        ),
        "source_point_hashes_exact_41": all(source_hash_checks),
        "d391_six_identity_exact": all(d391_identity.values()),
    }


def _row_bits(point: Sequence[float]) -> str:
    return np.asarray(point, dtype=np.float64).reshape(1, 3).tobytes().hex()


def _numeric_equal_bitwise_different_pairs(
    points: np.ndarray,
) -> list[dict[str, Any]]:
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rows: list[dict[str, Any]] = []
    for left in range(len(raw)):
        for right in range(left + 1, len(raw)):
            if bool(np.equal(raw[left], raw[right]).all()):
                left_bits = _row_bits(raw[left])
                right_bits = _row_bits(raw[right])
                if left_bits != right_bits:
                    rows.append(
                        {
                            "left": left,
                            "right": right,
                            "left_bits": left_bits,
                            "right_bits": right_bits,
                        }
                    )
    return rows


def _independent_canonical_unique(points: np.ndarray) -> np.ndarray:
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(raw):
        return np.empty((0, 3), dtype=np.float64)
    return np.unique(raw, axis=0).astype(np.float64, copy=False)


def _orders_for_control(count: int) -> tuple[str, list[tuple[int, ...]]]:
    if count <= SMALL_EXHAUSTIVE_MAX_POINTS:
        return "exhaustive_n_factorial", list(
            itertools.permutations(range(count))
        )
    identity = tuple(range(count))
    rows: list[tuple[int, ...]] = [identity, tuple(reversed(identity))]
    rows.extend(
        tuple(identity[offset:] + identity[:offset])
        for offset in range(1, count)
    )
    for index in range(count - 1):
        swapped = list(identity)
        swapped[index], swapped[index + 1] = (
            swapped[index + 1],
            swapped[index],
        )
        rows.append(tuple(swapped))
    return "structural_proof_plus_generator_smoke", list(dict.fromkeys(rows))


def _canonical_order_proof(
    points: np.ndarray, baseline: dict[str, Any]
) -> dict[str, Any]:
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    canonical = np.asarray(
        baseline["canonical_unique_points_f64_m"], dtype=np.float64
    )
    independent = _independent_canonical_unique(raw)
    alias_pairs = _numeric_equal_bitwise_different_pairs(raw)
    mode, orders = _orders_for_control(len(raw))
    identity = tuple(range(len(raw)))
    expected_adjacent: set[tuple[int, ...]] = set()
    for index in range(max(0, len(raw) - 1)):
        swapped = list(identity)
        swapped[index], swapped[index + 1] = (
            swapped[index + 1],
            swapped[index],
        )
        expected_adjacent.add(tuple(swapped))
    expected_rotations = {
        tuple(identity[offset:] + identity[:offset])
        for offset in range(1, len(raw))
    }
    expected_large_orders = (
        {identity, tuple(reversed(identity))}
        | expected_rotations
        | expected_adjacent
    )
    registered_order_set = set(orders)
    signature_sha = _canonical_sha(_d391()._rank_signature(baseline))
    transform_rows: list[dict[str, Any]] = []
    for ordinal, order in enumerate(orders):
        reordered = raw[list(order)]
        reordered_canonical = _d391()._canonical_unique(reordered)
        reordered_rank = _d391()._rank_core(reordered)
        row = {
            "ordinal": ordinal,
            "order_sha256": _canonical_sha(order),
            "canonical_sha256": _d391()._array_sha(reordered_canonical),
            "rank_signature_sha256": _canonical_sha(
                _d391()._rank_signature(reordered_rank)
            ),
            "canonical_exact": (
                _d391()._array_sha(reordered_canonical)
                == baseline["canonical_unique_points_sha256"]
            ),
            "rank_signature_exact": (
                _canonical_sha(_d391()._rank_signature(reordered_rank))
                == signature_sha
            ),
        }
        row["pass"] = row["canonical_exact"] and row["rank_signature_exact"]
        transform_rows.append(row)
    reverse_raw_differs = (
        len(raw) <= 1
        or _d391()._array_sha(raw[::-1]) != _d391()._array_sha(raw)
    )
    dropped = _d391()._canonical_unique(raw[:-1])
    dropped_hash_differs = (
        len(raw) == 0
        or _d391()._array_sha(dropped)
        != baseline["canonical_unique_points_sha256"]
    )
    structural_checks = {
        "source_function_sha_exact": (
            _text_sha(inspect.getsource(_d391()._canonical_unique))
            == EXPECTED_D391_FUNCTION_SOURCE_SHA256["_canonical_unique"]
        ),
        "source_uses_sorted_set_rows": (
            "rows = sorted({tuple(float(value) for value in row) for row in raw})"
            in inspect.getsource(_d391()._canonical_unique)
        ),
        "rank_core_source_sha_exact": (
            _text_sha(inspect.getsource(_d391()._rank_core))
            == EXPECTED_D391_FUNCTION_SOURCE_SHA256["_rank_core"]
        ),
        "rank_core_factors_through_canonical_unique": (
            "unique = _canonical_unique(points)"
            in inspect.getsource(_d391()._rank_core)
            and "_basis_rows(unique, anchor_index=index)"
            in inspect.getsource(_d391()._rank_core)
            and "_basis_rows(unique, anchor_index=None)"
            in inspect.getsource(_d391()._rank_core)
        ),
        "finite_shape_n_by_3": (
            raw.ndim == 2
            and raw.shape[1:] == (3,)
            and bool(np.isfinite(raw).all())
        ),
        "source_already_unique": len(raw) == len(canonical),
        "no_numeric_equal_bitwise_different_rows": not alias_pairs,
        "independent_numpy_unique_bit_exact": (
            _d391()._array_sha(independent)
            == baseline["canonical_unique_points_sha256"]
        ),
        "every_registered_transform_canonical_exact": all(
            row["canonical_exact"] for row in transform_rows
        ),
        "every_registered_transform_rank_signature_exact": all(
            row["rank_signature_exact"] for row in transform_rows
        ),
        "small_set_exhaustive_count_exact": (
            mode != "exhaustive_n_factorial"
            or len(orders) == math.factorial(len(raw))
        ),
        "large_set_registered_order_set_exact": (
            mode != "structural_proof_plus_generator_smoke"
            or registered_order_set == expected_large_orders
        ),
        "large_set_adjacent_swap_set_exact": (
            mode != "structural_proof_plus_generator_smoke"
            or expected_adjacent.issubset(registered_order_set)
        ),
        "large_set_rotation_set_exact": (
            mode != "structural_proof_plus_generator_smoke"
            or expected_rotations.issubset(registered_order_set)
        ),
        "negative_no_sort_reverse_detected": reverse_raw_differs,
        "negative_drop_row_detected": dropped_hash_differs,
    }
    return {
        "mode": mode,
        "explicit_nonclaim": (
            None
            if mode == "exhaustive_n_factorial"
            else "not an exhaustive n-factorial permutation enumeration"
        ),
        "canonicalization_source_proof": (
            "the exact frozen rank core first factors every input through "
            "sorted(set(Float64 row tuples)); every downstream basis consumes "
            "that canonical array"
        ),
        "generator_smoke_only": (
            "reverse, rotations, and adjacent swaps are finite smoke controls "
            "and are not the mathematical proof or an exhaustive n! test"
        ),
        "raw_point_count": len(raw),
        "registered_order_count": len(orders),
        "factorial_order_count": str(math.factorial(len(raw))),
        "baseline_canonical_sha256": (
            baseline["canonical_unique_points_sha256"]
        ),
        "independent_canonical_sha256": _d391()._array_sha(independent),
        "numeric_equal_bitwise_different_pairs": alias_pairs,
        "transform_rows": transform_rows,
        "structural_checks": structural_checks,
        "pass": all(structural_checks.values()),
    }


def _signed_zero_negative_control() -> dict[str, Any]:
    fixture = np.asarray(
        [[0.0, 1.0, 2.0], [-0.0, 1.0, 2.0]], dtype=np.float64
    )
    aliases = _numeric_equal_bitwise_different_pairs(fixture)
    return {
        "fixture_f64": fixture,
        "fixture_array_sha256": _d391()._array_sha(fixture),
        "numeric_equal_bitwise_different_pairs": aliases,
        "gate_rejects_fixture": len(aliases) == 1,
        "pass": len(aliases) == 1,
    }


def _safe_read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return _read_json(path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _validate_progress_prefix(
    manifest: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected_keys = {
        "catalog_index",
        "call_index",
        "call_id",
        "input_points_sha256",
        "rank_status",
        "rank_signature_sha256",
        "record_sha256",
        "integrity_pass",
    }
    rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    if PROGRESS.is_file():
        for ordinal, line in enumerate(
            PROGRESS.read_text(encoding="utf-8").splitlines()
        ):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("progress row is not an object")
                rows.append(value)
            except Exception as exc:
                parse_errors.append(
                    f"row {ordinal}: {type(exc).__name__}: {exc}"
                )
    prefix_checks: list[dict[str, Any]] = []
    record_by_catalog = {
        int(row["catalog_index"]): row for row in (records or [])
    }
    for ordinal, row in enumerate(rows):
        expected = manifest[ordinal] if ordinal < len(manifest) else {}
        record = record_by_catalog.get(ordinal)
        checks = {
            "keys_exact": set(row) == expected_keys,
            "catalog_index_exact": row.get("catalog_index") == ordinal,
            "call_index_exact": (
                row.get("call_index") == expected.get("call_index")
            ),
            "call_id_exact": (
                row.get("call_id") == expected.get("call_id")
            ),
            "input_sha_exact": (
                row.get("input_points_sha256")
                == expected.get("points_sha256")
            ),
            "integrity_bool": isinstance(
                row.get("integrity_pass"), bool
            ),
        }
        if record is not None:
            checks.update(
                {
                    "rank_status_exact_committed_record": (
                        row.get("rank_status")
                        == record["rank_authority"]["status"]
                    ),
                    "rank_signature_exact_committed_record": (
                        row.get("rank_signature_sha256")
                        == _canonical_sha(
                            _d391()._rank_signature(
                                record["rank_authority"]
                            )
                        )
                    ),
                    "record_digest_exact_committed_record": (
                        row.get("record_sha256")
                        == _canonical_sha(record)
                    ),
                    "integrity_exact_committed_record": (
                        row.get("integrity_pass")
                        == record["integrity_pass"]
                    ),
                }
            )
        prefix_checks.append(
            {
                "ordinal": ordinal,
                "call_index": row.get("call_index"),
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    return {
        "path": _rel(PROGRESS),
        "exists": PROGRESS.is_file(),
        "sha256": _sha(PROGRESS) if PROGRESS.is_file() else None,
        "row_count": len(rows),
        "manifest_count": len(manifest),
        "parse_errors": parse_errors,
        "prefix_rows": prefix_checks,
        "prefix_exact": (
            not parse_errors
            and len(rows) <= len(manifest)
            and all(row["pass"] for row in prefix_checks)
        ),
        "complete_exact": (
            not parse_errors
            and len(rows) == len(manifest)
            and all(row["pass"] for row in prefix_checks)
        ),
    }


def _display_record(
    catalog_index: int,
    manifest: dict[str, Any],
    rank: dict[str, Any],
) -> dict[str, Any]:
    unique = np.asarray(
        rank["canonical_unique_points_f64_m"], dtype=np.float64
    )
    center = unique.mean(axis=0) if len(unique) else np.zeros(3)
    centered = unique - center
    span = (
        float(np.max(np.linalg.norm(centered, axis=1)))
        if len(centered)
        else 0.0
    )
    scale = 0.8 / span if span > 0.0 else 1.0
    normalized = centered * scale
    column = catalog_index % 7
    row = catalog_index // 7
    offset = np.asarray(
        [3.0 * column, 3.0 * (4 - row), 0.0], dtype=np.float64
    )
    atlas = normalized + offset
    if len(centered):
        _, singular, vh = np.linalg.svd(
            centered, full_matrices=False
        )
        stable_rank = (
            rank["authoritative_rank"]
            if rank["status"].startswith("STABLE")
            else None
        )
        usable = (
            min(int(stable_rank), len(vh))
            if stable_rank is not None
            else min(3, len(vh))
        )
        sigma_max = float(singular[0]) if len(singular) else 0.0
        axes = np.zeros((usable, 3), dtype=np.float64)
        for index in range(usable):
            relative = (
                float(singular[index]) / sigma_max
                if sigma_max > 0.0
                else 0.0
            )
            axes[index] = vh[index] * 0.65 * relative
    else:
        singular = np.zeros(3)
        axes = np.zeros((0, 3), dtype=np.float64)
    return {
        "catalog_index": catalog_index,
        **manifest,
        "canonical_points_f64_m": unique,
        "canonical_points_sha256": manifest["points_sha256"],
        "display_center_f64_m": center,
        "display_scale_inspection_only": scale,
        "atlas_offset_inspection_only": offset,
        "atlas_points_f64_inspection_only": atlas,
        "atlas_points_sha256_not_scientific": _d391()._array_sha(atlas),
        "principal_axis_singular_values_inspection_only": singular,
        "principal_axis_count_inspection_only": len(axes),
        "principal_axis_vectors_atlas_inspection_only": axes,
        "principal_axis_display_policy": (
            "stable sets show authoritative_rank axes; ambiguous sets show "
            "up to three axes; each length is proportional to sigma_i/sigma_1"
        ),
        "display_role": (
            "centered/scaled/offset Float32 inspection copy only; never "
            "hashed back into the numerical gate"
        ),
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    geometry = _read_json(D390_GEOMETRY)
    schema_checks = _input_schema_checks(geometry)
    if not all(schema_checks.values()):
        raise RuntimeError(f"D392 input schema failed: {schema_checks}")
    manifest = _remaining_manifest(geometry)
    by_index = {
        int(row["call_index"]): row for row in geometry["records"]
    }
    records: list[dict[str, Any]] = []
    display_records: list[dict[str, Any]] = []
    for catalog_index, expected in enumerate(manifest):
        _deadline(f"before_call_{expected['call_index']}")
        source = by_index[expected["call_index"]]
        points = np.asarray(
            source["terminal_candidate_unique_points_f64_m"],
            dtype=np.float64,
        )
        rank = _d391()._rank_core(points)
        order = _canonical_order_proof(points, rank)
        translation = _d391()._translation_control(points, rank)
        scale = _d391()._scale_controls(points, rank)
        implementation_status = rank["status"].startswith(
            "IMPLEMENTATION_FAIL"
        )
        admissible_status = (
            rank["status"].startswith("STABLE")
            or rank["status"].startswith("NUMERICALLY_AMBIGUOUS")
        )
        checks = {
            "source_manifest_exact": (
                source["call_id"] == expected["call_id"]
                and len(points) == expected["unique_point_count"]
                and source["terminal_candidate_unique_points_sha256"]
                == expected["points_sha256"]
            ),
            "canonical_points_hash_matches_source": (
                rank["canonical_unique_points_sha256"]
                == expected["points_sha256"]
            ),
            "rank_status_admissible": admissible_status,
            "no_implementation_failure": not implementation_status,
            "final_rank_respects_hard_cap": rank[
                "final_rank_respects_hard_cap"
            ],
            "canonical_order_proof_pass": order["pass"],
            "translation_control_pass": translation["pass"],
            "scale_control_pass": scale["pass"],
        }
        record = {
            "catalog_index": catalog_index,
            **expected,
            "d390_historical_rank": source["affine_rank"],
            "d390_historical_class": source["affine_class"],
            "d390_historical_values_are_diagnostic_not_answer_key": True,
            "rank_authority": rank,
            "canonical_order_proof": order,
            "translation_control": translation,
            "scale_control": scale,
            "scientific_checks": checks,
            "integrity_pass": all(checks.values()),
            "stable": rank["status"].startswith("STABLE"),
            "ambiguous": rank["status"].startswith(
                "NUMERICALLY_AMBIGUOUS"
            ),
        }
        records.append(record)
        display_records.append(
            _display_record(catalog_index, expected, rank)
        )
        _append_jsonl(
            PROGRESS,
            {
                "catalog_index": catalog_index,
                "call_index": expected["call_index"],
                "call_id": expected["call_id"],
                "input_points_sha256": expected["points_sha256"],
                "rank_status": rank["status"],
                "rank_signature_sha256": _canonical_sha(
                    _d391()._rank_signature(rank)
                ),
                "record_sha256": _canonical_sha(record),
                "integrity_pass": record["integrity_pass"],
            },
        )
        _deadline(f"after_call_{expected['call_index']}")
    signed_zero = _signed_zero_negative_control()
    progress_validation = _validate_progress_prefix(manifest, records)
    stable_count = sum(row["stable"] for row in records)
    ambiguous_count = sum(row["ambiguous"] for row in records)
    integrity_pass = (
        len(records) == 35
        and all(row["integrity_pass"] for row in records)
        and signed_zero["pass"]
    )
    coverage_pass = integrity_pass and stable_count == 35
    status_counts = {
        status: sum(
            row["rank_authority"]["status"] == status for row in records
        )
        for status in sorted(
            {row["rank_authority"]["status"] for row in records}
        )
    }
    class_counts = {
        name: sum(
            row["rank_authority"]["authoritative_class"] == name
            for row in records
        )
        for name in (
            "EMPTY",
            "POINT",
            "LINE",
            "FACE_LIKE",
            "FULL_DIMENSIONAL",
        )
    }
    historical_match_count = sum(
        row["rank_authority"]["authoritative_rank"]
        == row["d390_historical_rank"]
        and row["rank_authority"]["authoritative_class"]
        == row["d390_historical_class"]
        for row in records
    )
    d391_evidence = _read_json(D391_EVIDENCE)
    d391_records = d391_evidence["disputed_records"]
    d391_stable = [
        row
        for row in d391_records
        if row["rank_authority"]["status"].startswith("STABLE")
    ]
    d391_ambiguous = [
        row
        for row in d391_records
        if row["rank_authority"]["status"].startswith(
            "NUMERICALLY_AMBIGUOUS"
        )
    ]
    resolved_class_counts = {
        name: class_counts[name]
        + sum(
            row["rank_authority"]["authoritative_class"] == name
            for row in d391_stable
        )
        for name in class_counts
    }
    d392_by_index = {
        int(row["call_index"]): row for row in records
    }
    d391_by_index = {
        int(row["call_index"]): row for row in d391_records
    }
    combined_authority_vector: list[dict[str, Any]] = []
    for call_index in range(41):
        if call_index in d392_by_index:
            row = d392_by_index[call_index]
            source_case = "D392"
        else:
            row = d391_by_index[call_index]
            source_case = "D391"
        rank = row["rank_authority"]
        combined_authority_vector.append(
            {
                "call_index": call_index,
                "call_id": row["call_id"],
                "points_sha256": row["points_sha256"],
                "authority_source_case": source_case,
                "status": rank["status"],
                "authoritative_rank": rank["authoritative_rank"],
                "authoritative_class": rank["authoritative_class"],
                "stable": rank["status"].startswith("STABLE"),
                "ambiguous": rank["status"].startswith(
                    "NUMERICALLY_AMBIGUOUS"
                ),
            }
        )
    combined_call29 = combined_authority_vector[29]
    d392_ambiguous_rows = [
        row for row in records if row["ambiguous"]
    ]
    combined_checks = {
        "indices_exact_0_to_40": (
            [row["call_index"] for row in combined_authority_vector]
            == list(range(41))
        ),
        "source_counts_exact_35_plus_6": (
            sum(
                row["authority_source_case"] == "D392"
                for row in combined_authority_vector
            )
            == 35
            and sum(
                row["authority_source_case"] == "D391"
                for row in combined_authority_vector
            )
            == 6
        ),
        "call29_exact_ambiguity_null": (
            combined_call29["call_id"]
            == "lower_01_02_pre_float32_lbr"
            and combined_call29["authority_source_case"] == "D391"
            and combined_call29["status"]
            == "NUMERICALLY_AMBIGUOUS_BASIS"
            and combined_call29["authoritative_rank"] is None
            and combined_call29["authoritative_class"] is None
        ),
        "resolved_class_sum_equals_resolved_stable": (
            sum(resolved_class_counts.values())
            == stable_count + len(d391_stable)
        ),
    }
    case_checks = {
        "input_schema_pass": all(schema_checks.values()),
        "remaining35_exact": len(records) == 35,
        "all_record_integrity_pass": all(
            row["integrity_pass"] for row in records
        ),
        "signed_zero_alias_negative_control_pass": signed_zero["pass"],
        "progress_prefix_exact_35": progress_validation[
            "complete_exact"
        ],
        "frozen_d391_five_stable_one_ambiguous": (
            len(d391_stable) == 5 and len(d391_ambiguous) == 1
        ),
        "combined_41_authority_vector_exact": all(
            combined_checks.values()
        ),
    }
    if not all(case_checks.values()):
        integrity_pass = False
        coverage_pass = False
    if not integrity_pass:
        verdict = "D392_FROZEN_AUTHORITY_OR_INTEGRITY_FAIL_STOP"
    elif coverage_pass:
        verdict = (
            "D392_REMAINING35_FROZEN_RANK_AUTHORITY_WITH_"
            "SCALABLE_ORDER_PROOF_PASS"
        )
    else:
        verdict = "D392_REMAINING35_COVERAGE_INCOMPLETE_STOP"
    evidence = {
        "artifact": "D392_REMAINING35_RANK_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "scientific_input": {
            "path": _rel(D390_GEOMETRY),
            "sha256": _sha(D390_GEOMETRY),
            "remaining35_manifest": manifest,
            "remaining35_manifest_sha256": _canonical_sha(manifest),
        },
        "frozen_d391_authority": {
            "script_path": _rel(D391_SCRIPT),
            "script_sha256": _sha(D391_SCRIPT),
            "function_source_sha256": (
                EXPECTED_D391_FUNCTION_SOURCE_SHA256
            ),
            "evidence_path": _rel(D391_EVIDENCE),
            "evidence_sha256": _sha(D391_EVIDENCE),
            "rank_core_reimplemented_in_d392": False,
            "rank_core_imported_read_only": True,
            "same_numeric_policy": {
                "exact_dyadic_differences": True,
                "all_anchors_plus_all_unordered_pairs": True,
                "tau_formula": (
                    "sigma_max * max(unique_point_count,3) * "
                    "float64_epsilon"
                ),
                "threshold_alphas": [0.5, 1.0, 2.0],
                "hard_cap": (
                    "min(3,max(0,unique_point_count-1))"
                ),
                "translation_control": "frozen D391 function",
                "scale_exponents": [-20, -10, 10, 20],
                "ambiguity_forced_to_class": False,
            },
            "order_control_change_disclosed": (
                "n! retained for n<=6; larger sets use source/bit-set "
                "structural proof plus reversal, every rotation, and every "
                "adjacent-transposition smoke controls"
            ),
            "six_call_identity_checks": (
                _d391_six_identity_checks(geometry)
            ),
        },
        "schema_checks": schema_checks,
        "signed_zero_negative_control": signed_zero,
        "progress_validation": progress_validation,
        "records": records,
        "remaining35_summary": {
            "count": len(records),
            "stable_count": stable_count,
            "ambiguous_count": ambiguous_count,
            "status_counts": status_counts,
            "stable_class_counts": class_counts,
            "d390_historical_rank_class_match_count_diagnostic_only": (
                historical_match_count
            ),
        },
        "same_rank_authority_41_coverage": {
            "d392_stable_count": stable_count,
            "d391_stable_count": len(d391_stable),
            "resolved_stable_count": stable_count + len(d391_stable),
            "explicit_ambiguous_count": (
                ambiguous_count + len(d391_ambiguous)
            ),
            "resolved_class_counts": resolved_class_counts,
            "combined_authority_vector": combined_authority_vector,
            "combined_authority_vector_sha256": _canonical_sha(
                combined_authority_vector
            ),
            "combined_authority_checks": combined_checks,
            "authoritative_all_41_class_aggregate": None,
            "reason_all_41_aggregate_null": (
                (
                    f"D392 remaining35 contains {ambiguous_count} new "
                    "ambiguity/ambiguities in addition to frozen call29: "
                    + ", ".join(
                        row["call_id"] for row in d392_ambiguous_rows
                    )
                )
                if ambiguous_count > 0
                else (
                    "frozen call29 remains explicitly ambiguous"
                    if len(d391_ambiguous) == 1
                    else "coverage is incomplete"
                )
            ),
            "later_call29_only_case_admissible": (
                coverage_pass
                and ambiguous_count == 0
                and combined_checks["call29_exact_ambiguity_null"]
            ),
        },
        "case_checks": case_checks,
        "numeric_integrity_pass": integrity_pass,
        "coverage_pass": coverage_pass,
        "numeric_verdict": verdict,
        "nonclaims": {
            "d389_d390_d391_repaired_or_retroactively_passed": False,
            "d390_clipping_reexecuted": 0,
            "d391_plane_contract_retested": 0,
            "call29_recomputed_or_forced": 0,
            "d389_seam_propagation": 0,
            "threshold_epsilon_tolerance_or_jitter_changes": 0,
            "partition_budget_or_geometry_changes": 0,
            "selected_or_adopted_budget": None,
            "collider_asset_usd_materialization": 0,
            "isaac_kit_physx_warp_cuda": 0,
            "cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_or_settings_changes": 0,
            "physics_or_grasp_result": None,
            "g0a_pass": False,
        },
        "algorithm_elapsed_seconds": time.monotonic() - started,
    }
    display = {
        "artifact": "D392_REMAINING35_DISPLAY_GEOMETRY_V1",
        "scientific_authority": (
            "D390 Float64 terminal points and D392 canonical JSON hashes"
        ),
        "rerun_role": (
            "centered/scaled/offset Float32 inspection copies only"
        ),
        "record_count": len(display_records),
        "records": display_records,
    }
    return evidence, display, records


def _write_csv(records: list[dict[str, Any]]) -> None:
    fields = [
        "catalog_index",
        "call_index",
        "call_id",
        "unique_point_count",
        "d390_rank",
        "d390_class",
        "d392_status",
        "d392_rank",
        "d392_class",
        "exact_dyadic_rank",
        "order_mode",
        "registered_order_count",
        "order_pass",
        "translation_pass",
        "scale_pass",
        "historical_match_diagnostic",
        "integrity_pass",
    ]
    with CSV_PATH.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            rank = row["rank_authority"]
            order = row["canonical_order_proof"]
            writer.writerow(
                {
                    "catalog_index": row["catalog_index"],
                    "call_index": row["call_index"],
                    "call_id": row["call_id"],
                    "unique_point_count": row["unique_point_count"],
                    "d390_rank": row["d390_historical_rank"],
                    "d390_class": row["d390_historical_class"],
                    "d392_status": rank["status"],
                    "d392_rank": rank["authoritative_rank"],
                    "d392_class": rank["authoritative_class"],
                    "exact_dyadic_rank": rank["exact_dyadic_rank"],
                    "order_mode": order["mode"],
                    "registered_order_count": order[
                        "registered_order_count"
                    ],
                    "order_pass": order["pass"],
                    "translation_pass": row["translation_control"]["pass"],
                    "scale_pass": row["scale_control"]["pass"],
                    "historical_match_diagnostic": (
                        rank["authoritative_rank"]
                        == row["d390_historical_rank"]
                        and rank["authoritative_class"]
                        == row["d390_historical_class"]
                    ),
                    "integrity_pass": row["integrity_pass"],
                }
            )
        stream.flush()
        os.fsync(stream.fileno())


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _artifact_reference_exact(reference: Any, path: Path) -> bool:
    return (
        isinstance(reference, dict)
        and path.is_file()
        and reference == _artifact(path)
    )


def _authorization_checks() -> dict[str, bool]:
    prereg = _read_json(PREREGISTRATION)
    authority = _read_json(EXECUTION_AUTHORITY)
    authorization = _read_json(AUTHORIZATION)
    external = os.environ.get(WORKER_AUTHORIZATION_SHA256_ENV)
    return {
        "external_authorization_sha_exact": (
            external is not None
            and external == _sha(AUTHORIZATION)
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
        "input_exact": (
            authorization.get("scientific_input_sha256")
            == _sha(D390_GEOMETRY)
        ),
        "preregistration_exact": (
            authorization.get("preregistration_sha256")
            == _sha(PREREGISTRATION)
        ),
        "execution_authority_exact": (
            authorization.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
        ),
        "preregistered_chain_exact": all(
            _preregistered_chain_checks().values()
        ),
        "frozen_checks_pass": all(_frozen_checks().values()),
        "status_scope_pass": _status_scope_pass(
            _status_lines(),
            authority["git"]["status_with_execution_authority"],
            allowed_output_names=_out_names(),
        ),
        "prereg_manifest_exact": (
            prereg["evaluation_set"]["manifest_sha256"]
            == EXPECTED_REMAINING35_MANIFEST_SHA256
        ),
    }


def _worker_inner() -> int:
    global _deadline_monotonic
    started = time.monotonic()
    _deadline_monotonic = started + COOPERATIVE_DEADLINE_SECONDS
    _require_names(PRE_WORKER_NAMES, "worker_before_sentinel")
    checks = _authorization_checks()
    if not all(checks.values()):
        raise RuntimeError(f"D392 worker authorization failed: {checks}")
    _write_json_x(
        SENTINEL,
        {
            "artifact": "D392_WORKER_START_SENTINEL_V1",
            "worker_pid": os.getpid(),
            "parent_supervisor_pid": os.getppid(),
            "worker_invocation_index": 1,
            "retry_index": 0,
            "script_sha256": _sha(SCRIPT),
            "scientific_input_sha256": _sha(D390_GEOMETRY),
            "preregistration_sha256": _sha(PREREGISTRATION),
            "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
            "authorization_sha256": _sha(AUTHORIZATION),
            "wall_time_ns": time.time_ns(),
        },
    )
    _phase("worker_start", worker_pid=os.getpid())
    evidence, geometry, records = _compute()
    _write_json_x(EVIDENCE, evidence)
    _write_json_x(GEOMETRY, geometry)
    _write_csv(records)
    _phase(
        "canonical_numeric_evidence_committed",
        evidence_sha256=_sha(EVIDENCE),
        geometry_sha256=_sha(GEOMETRY),
        csv_sha256=_sha(CSV_PATH),
        progress_sha256=_sha(PROGRESS),
    )
    progress_validation = _validate_progress_prefix(
        _remaining_manifest(_read_json(D390_GEOMETRY)),
        records,
    )
    worker_checks = {
        "authorization_checks_pass": all(checks.values()),
        "numeric_integrity_pass": evidence["numeric_integrity_pass"],
        "progress_exact_35": progress_validation["complete_exact"],
        "preregistered_chain_exact": all(
            _preregistered_chain_checks().values()
        ),
        "within_cooperative_deadline": (
            time.monotonic() <= _deadline_monotonic
        ),
        "d390_manifest_remains_exact": (
            _canonical_sha(_directory_manifest(D390_DIR))
            == EXPECTED_D390_DIRECTORY_MANIFEST_SHA256
        ),
        "d391_manifest_remains_exact": (
            _canonical_sha(_directory_manifest(D391_DIR))
            == EXPECTED_D391_DIRECTORY_MANIFEST_SHA256
        ),
    }
    claim = {
        "artifact": "D392_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "actual_worker_invocations": 1,
        "worker_invocation_index": 1,
        "retries": 0,
        "process_signals_sent": 0,
        "cooperative_deadline_seconds": COOPERATIVE_DEADLINE_SECONDS,
        "hard_watchdog_seconds": None,
        "worker_elapsed_seconds": time.monotonic() - started,
        "script_sha256": _sha(SCRIPT),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "invocation_sha256": _sha(INVOCATION),
        "authorization_sha256": _sha(AUTHORIZATION),
        "worker_start_sentinel_sha256": _sha(SENTINEL),
        "authorization_checks": checks,
        "progress_validation": progress_validation,
        "checks": worker_checks,
        "scientific_outcome_may_be_negative_without_worker_failure": True,
        "numeric_verdict": evidence["numeric_verdict"],
        "coverage_pass": evidence["coverage_pass"],
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
        raise RuntimeError(f"D392 worker claim failed: {worker_checks}")
    print(
        json.dumps(
            {
                "worker_pass": True,
                "numeric_verdict": evidence["numeric_verdict"],
                "coverage_pass": evidence["coverage_pass"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _worker_failure_claim(exc: BaseException) -> None:
    if WORKER_FAILURE_CLAIM.exists():
        return
    geometry_value, geometry_error = _safe_read_json(D390_GEOMETRY)
    manifest = (
        _remaining_manifest(geometry_value)
        if isinstance(geometry_value, dict)
        else []
    )
    progress = _validate_progress_prefix(manifest)
    _write_json_x(
        WORKER_FAILURE_CLAIM,
        {
            "artifact": "D392_WORKER_FAILURE_CLAIM_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "worker_pid": os.getpid(),
            "parent_supervisor_pid": os.getppid(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "geometry_read_error": geometry_error,
            "progress_prefix_validation": progress,
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
                _sha(AUTHORIZATION) if AUTHORIZATION.is_file() else None
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
    geometry_value, geometry_error = _safe_read_json(D390_GEOMETRY)
    manifest = (
        _remaining_manifest(geometry_value)
        if isinstance(geometry_value, dict)
        else []
    )
    evidence_value, evidence_error = (
        _safe_read_json(EVIDENCE)
        if EVIDENCE.is_file()
        else (None, None)
    )
    progress = _validate_progress_prefix(
        manifest,
        evidence_value.get("records", [])
        if isinstance(evidence_value, dict)
        else None,
    )
    supervisor_value, supervisor_error = (
        _safe_read_json(SUPERVISOR)
        if SUPERVISOR.is_file()
        else (None, None)
    )
    sentinel_value, sentinel_error = (
        _safe_read_json(SENTINEL)
        if SENTINEL.is_file()
        else (None, None)
    )
    worker_failure_value, worker_failure_error = (
        _safe_read_json(WORKER_FAILURE_CLAIM)
        if WORKER_FAILURE_CLAIM.is_file()
        else (None, None)
    )
    actual_worker_started = bool(
        _worker_started
        or SENTINEL.is_file()
        or WORKER_FAILURE_CLAIM.is_file()
        or (
            isinstance(supervisor_value, dict)
            and supervisor_value.get("actual_worker_invocations") == 1
        )
    )
    worker_pid = (
        supervisor_value.get("worker_pid")
        if isinstance(supervisor_value, dict)
        else None
    ) or (
        sentinel_value.get("worker_pid")
        if isinstance(sentinel_value, dict)
        else None
    ) or (
        worker_failure_value.get("worker_pid")
        if isinstance(worker_failure_value, dict)
        else None
    ) or _worker_pid
    before_names = sorted(_out_names() - {FAILURE.name})
    expected_final_names = sorted(set(before_names) | {FAILURE.name})
    artifact_hashes_before = {
        item.name: _artifact(item)
        for item in sorted(OUT_DIR.iterdir())
        if item.is_file() and item != FAILURE
    }
    failure = {
        "artifact": "D392_FAILURE_ATTESTATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "actual_worker_started": actual_worker_started,
        "actual_worker_invocations": int(actual_worker_started),
        "worker_pid": worker_pid,
        "retries": 0,
        "process_signals_sent": 0,
        "hard_watchdog_seconds": None,
        "progress_prefix_validation": progress,
        "numeric_evidence_committed": EVIDENCE.is_file(),
        "numeric_evidence_read_error": evidence_error,
        "numeric_evidence_sha256": (
            _sha(EVIDENCE) if EVIDENCE.is_file() else None
        ),
        "numeric_verdict": (
            evidence_value.get("numeric_verdict")
            if isinstance(evidence_value, dict)
            else None
        ),
        "geometry_read_error": geometry_error,
        "supervisor_read_error": supervisor_error,
        "sentinel_read_error": sentinel_error,
        "worker_failure_claim_read_error": worker_failure_error,
        "script_sha256": _sha(SCRIPT),
        "start_sha256": _sha(START),
        "execution_authority_sha256": (
            _sha(EXECUTION_AUTHORITY)
            if EXECUTION_AUTHORITY.is_file()
            else None
        ),
        "preregistration_sha256": (
            _sha(PREREGISTRATION)
            if PREREGISTRATION.is_file()
            else None
        ),
        "supervisor_sha256": (
            _sha(SUPERVISOR) if SUPERVISOR.is_file() else None
        ),
        "worker_failure_claim_sha256": (
            _sha(WORKER_FAILURE_CLAIM)
            if WORKER_FAILURE_CLAIM.is_file()
            else None
        ),
        "artifact_hashes_before_attestation": artifact_hashes_before,
        "output_inventory_before_attestation": before_names,
        "expected_final_inventory": expected_final_names,
        "verdict": "D392_OPERATIONAL_OR_PROVENANCE_FAIL_STOP",
    }
    _write_json_x(FAILURE, failure)
    if sorted(_out_names()) != expected_final_names:
        raise RuntimeError(
            "D392 failure final inventory diverged after attestation"
        )


def _prepare() -> int:
    _require_names(AUTHORITY_NAMES, "before_prepare")
    authority = _read_json(EXECUTION_AUTHORITY)
    external = os.environ.get(EXECUTION_AUTHORITY_SHA256_ENV)
    if external is None or external != _sha(EXECUTION_AUTHORITY):
        raise RuntimeError(
            "D392 prepare requires external exact execution authority SHA"
        )
    authority_checks = _execution_authority_checks(
        authority, exact_status=True
    )
    frozen = _frozen_checks()
    geometry = _read_json(D390_GEOMETRY)
    schema = _input_schema_checks(geometry)
    d391_six_identity = _d391_six_identity_checks(geometry)
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
        "no_nvidia_direct_imports_in_wrapper_or_frozen_d391": not any(
            name
            in {
                "isaaclab",
                "isaacsim",
                "omni",
                "pxr",
                "warp",
                "torch",
            }
            for name in (
                _direct_import_roots(SCRIPT)
                | _direct_import_roots(D391_SCRIPT)
            )
        ),
        "python_no_bytecode_requested": sys.dont_write_bytecode,
    }
    if not all(authority_checks.values()):
        raise RuntimeError(
            f"D392 execution authority failed: {authority_checks}"
        )
    if not all(frozen.values()):
        raise RuntimeError(f"D392 frozen inputs changed: {frozen}")
    if not all(schema.values()):
        raise RuntimeError(f"D392 schema preflight failed: {schema}")
    if not all(environment.values()):
        raise RuntimeError(f"D392 environment preflight failed: {environment}")
    _phase("prepare_start")
    manifest = _remaining_manifest(geometry)
    prereg = {
        "artifact": "D392_PREREGISTRATION_V1",
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
            "d390_geometry_sha256": _sha(D390_GEOMETRY),
            "d391_script_sha256": _sha(D391_SCRIPT),
            "d391_evidence_sha256": _sha(D391_EVIDENCE),
            "normalized_user_approval_text": (
                EXPECTED_USER_APPROVAL_NORMALIZED
            ),
            "normalized_user_approval_text_sha256": _text_sha(
                EXPECTED_USER_APPROVAL_NORMALIZED
            ),
        },
        "execution_authority": {
            "path": _rel(EXECUTION_AUTHORITY),
            "sha256": _sha(EXECUTION_AUTHORITY),
            "external_environment_variable": (
                EXECUTION_AUTHORITY_SHA256_ENV
            ),
            "checks": authority_checks,
        },
        "scientific_question": (
            "Do D390's remaining 35 terminal point sets stay stable when "
            "the exact frozen D391 rank core, translation controls, and "
            "scale controls are applied, with a disclosed scalable "
            "canonical-order proof?"
        ),
        "evaluation_set": {
            "source_path": _rel(D390_GEOMETRY),
            "source_sha256": _sha(D390_GEOMETRY),
            "excluded_d391_call_indices": list(DISPUTED_CALL_INDICES),
            "manifest": manifest,
            "manifest_sha256": _canonical_sha(manifest),
            "count": len(manifest),
            "disjoint_union_with_d391_six_is_0_to_40": True,
        },
        "frozen_d391_authority": {
            "script_path": _rel(D391_SCRIPT),
            "script_sha256": _sha(D391_SCRIPT),
            "function_source_sha256": (
                EXPECTED_D391_FUNCTION_SOURCE_SHA256
            ),
            "evidence_path": _rel(D391_EVIDENCE),
            "evidence_sha256": _sha(D391_EVIDENCE),
            "rank_core_reimplementation": 0,
            "threshold_or_hard_cap_change": 0,
            "translation_or_scale_control_change": 0,
            "six_call_identity_checks": d391_six_identity,
        },
        "order_invariance_contract": {
            "why_d391_n_factorial_cannot_scale": (
                "remaining sets contain up to 25 points; 25! = "
                "15511210043330985984000000"
            ),
            "small_sets_n_le_6": "all n! permutations",
            "large_sets": (
                "exact frozen sorted(set(Float64 row tuples)) source/hash "
                "proof plus independent NumPy canonical reconstruction, "
                "reversal, every cyclic rotation, and every adjacent "
                "transposition"
            ),
            "large_set_nonclaim": (
                "finite reverse/rotation/adjacent controls are smoke only; "
                "they are neither the source-bound proof nor exhaustive n!"
            ),
            "proof_authority": (
                "exact frozen _rank_core source factors input through exact "
                "frozen sorted-set _canonical_unique before all bases"
            ),
            "adjacent_transposition_generation_theorem_diagnostic_only": True,
            "signed_zero_or_equal_numeric_different_bits_rejected": True,
        },
        "outcome_policy": {
            "all35_stable": (
                "D392_REMAINING35_FROZEN_RANK_AUTHORITY_WITH_"
                "SCALABLE_ORDER_PROOF_PASS"
            ),
            "admissible_ambiguity": (
                "D392_REMAINING35_COVERAGE_INCOMPLETE_STOP"
            ),
            "hash_schema_function_or_proof_failure": (
                "D392_FROZEN_AUTHORITY_OR_INTEGRITY_FAIL_STOP"
            ),
            "historical_d390_class_is_answer_key": False,
            "exact_dyadic_rank_must_equal_numeric_rank": False,
            "call29_forced_or_recomputed": False,
        },
        "execution_contract": {
            "numeric_worker": 1,
            "worker_retries": 0,
            "viewer_maximum": 1,
            "viewer_retries": 0,
            "hard_watchdog_seconds": None,
            "process_signals_authorized": 0,
            "cooperative_deadline_seconds": (
                COOPERATIVE_DEADLINE_SECONDS
            ),
            "numeric_committed_before_observability": True,
            "per_call_prefix_append_only": True,
        },
        "visualization_contract": {
            "board": "exact 1920x1080 two-column 18+17 row table",
            "rerun": "single 7x5 atlas containing all 35 point sets",
            "rerun_float32_role": "inspection only",
            "rrd_rbl_footer_entity_component_contracts": True,
            "manual_inspection_required_after_actual_viewing": True,
        },
        "forbidden": {
            "d390_clipping_replay": 0,
            "d391_plane_retest": 0,
            "call29_localization": 0,
            "d389_seam_propagation": 0,
            "collider_usd_isaac_physx_cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_or_settings_changes": 0,
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
    _phase("prepare_end", preregistration_sha256=_sha(PREREGISTRATION))
    _require_names(PREPARED_NAMES, "after_prepare")
    print(json.dumps({"prepared": True, "case": CASE}, ensure_ascii=False))
    return 0


def _run() -> int:
    global _worker_started, _worker_pid
    _require_names(PREPARED_NAMES, "before_run")
    if FAILURE.exists():
        raise RuntimeError("D392 failure attestation already exists")
    authority = _read_json(EXECUTION_AUTHORITY)
    if not all(_frozen_checks().values()):
        raise RuntimeError("D392 frozen inputs changed before run")
    if not all(_preregistered_chain_checks().values()):
        raise RuntimeError("D392 preregistered chain changed before run")
    if not _status_scope_pass(
        _status_lines(),
        authority["git"]["status_with_execution_authority"],
        allowed_output_names=_out_names(),
    ):
        raise RuntimeError("D392 worktree scope changed before run")
    command = [sys.executable, "-B", str(SCRIPT), "--stage", "worker"]
    invocation = {
        "artifact": "D392_OFFLINE_WORKER_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "cwd": str(REPO),
        "actual_worker_maximum": 1,
        "worker_invocation_index": 1,
        "retry_index": 0,
        "retries": 0,
        "hard_watchdog_seconds": None,
        "supervisor_signal_authority": False,
        "process_signals_authorized": 0,
        "script_sha256": _sha(SCRIPT),
        "scientific_input_sha256": _sha(D390_GEOMETRY),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
    }
    _write_json_x(INVOCATION, invocation)
    _write_json_x(
        AUTHORIZATION,
        {
            "artifact": "D392_WORKER_AUTHORIZATION_V1",
            "supervisor_pid": os.getpid(),
            "worker_invocation_index": 1,
            "retry_index": 0,
            "retries": 0,
            "script_sha256": _sha(SCRIPT),
            "scientific_input_sha256": _sha(D390_GEOMETRY),
            "preregistration_sha256": _sha(PREREGISTRATION),
            "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
            "invocation_sha256": _sha(INVOCATION),
            "hard_watchdog_seconds": None,
            "process_signals_authorized": 0,
        },
    )
    _phase("supervisor_before_worker", supervisor_pid=os.getpid())
    started = time.monotonic()
    returncode: int | None = None
    error: str | None = None
    process: subprocess.Popen[str] | None = None
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
    claim, claim_read_error = (
        _safe_read_json(WORKER_CLAIM)
        if WORKER_CLAIM.is_file()
        else (None, None)
    )
    evidence_value, evidence_read_error = (
        _safe_read_json(EVIDENCE)
        if EVIDENCE.is_file()
        else (None, None)
    )
    progress_validation = (
        _validate_progress_prefix(
            _remaining_manifest(_read_json(D390_GEOMETRY)),
            evidence_value.get("records", [])
            if isinstance(evidence_value, dict)
            else None,
        )
        if PROGRESS.is_file()
        else {"complete_exact": False}
    )
    claim_artifacts = (
        claim.get("artifacts", {}) if isinstance(claim, dict) else {}
    )
    post_worker_chain_checks = {
        "claim_artifacts_exact": (
            _artifact_reference_exact(
                claim_artifacts.get("progress"), PROGRESS
            )
            and _artifact_reference_exact(
                claim_artifacts.get("evidence"), EVIDENCE
            )
            and _artifact_reference_exact(
                claim_artifacts.get("geometry"), GEOMETRY
            )
            and _artifact_reference_exact(
                claim_artifacts.get("csv"), CSV_PATH
            )
        ),
        "claim_provenance_exact": bool(
            isinstance(claim, dict)
            and claim.get("script_sha256") == _sha(SCRIPT)
            and claim.get("preregistration_sha256")
            == _sha(PREREGISTRATION)
            and claim.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
            and claim.get("invocation_sha256") == _sha(INVOCATION)
            and claim.get("authorization_sha256")
            == _sha(AUTHORIZATION)
            and claim.get("worker_start_sentinel_sha256")
            == (_sha(SENTINEL) if SENTINEL.is_file() else None)
        ),
        "progress_complete_exact": progress_validation.get(
            "complete_exact"
        )
        is True,
        "preregistered_chain_exact": all(
            _preregistered_chain_checks().values()
        ),
    }
    supervisor = {
        "artifact": "D392_OFFLINE_WORKER_SUPERVISOR_V1",
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
        "worker_claim_pass": bool(
            isinstance(claim, dict) and claim.get("pass") is True
        ),
        "worker_claim_read_error": claim_read_error,
        "evidence_read_error": evidence_read_error,
        "numeric_evidence_committed": EVIDENCE.is_file(),
        "numeric_verdict": (
            evidence_value.get("numeric_verdict")
            if isinstance(evidence_value, dict)
            else None
        ),
        "retries": 0,
        "process_signals_sent": 0,
        "hard_watchdog_seconds": None,
        "supervisor_error": error,
        "elapsed_seconds": time.monotonic() - started,
        "script_sha256": _sha(SCRIPT),
        "preregistration_sha256": _sha(PREREGISTRATION),
        "execution_authority_sha256": _sha(EXECUTION_AUTHORITY),
        "invocation_sha256": _sha(INVOCATION),
        "authorization_sha256": _sha(AUTHORIZATION),
        "sentinel_sha256": _sha(SENTINEL) if SENTINEL.is_file() else None,
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
        "post_worker_chain_checks": post_worker_chain_checks,
        "pass": (
            process is not None
            and returncode == 0
            and process.poll() is not None
            and bool(
                isinstance(claim, dict) and claim.get("pass") is True
            )
            and EVIDENCE.is_file()
            and claim_read_error is None
            and evidence_read_error is None
            and all(post_worker_chain_checks.values())
            and error is None
            and not FAILURE.exists()
        ),
    }
    _write_json_x(SUPERVISOR, supervisor)
    _phase(
        "supervisor_after_worker",
        worker_returncode=returncode,
        supervisor_pass=supervisor["pass"],
    )
    if not supervisor["pass"]:
        raise RuntimeError(f"D392 numeric worker failed: {supervisor}")
    _require_names(POST_WORKER_NAMES, "after_run")
    print(json.dumps(supervisor, ensure_ascii=False))
    return 0


def _numeric_chain_checks() -> dict[str, bool]:
    invocation = _read_json(INVOCATION)
    authorization = _read_json(AUTHORIZATION)
    sentinel = _read_json(SENTINEL)
    worker = _read_json(WORKER_CLAIM)
    supervisor = _read_json(SUPERVISOR)
    evidence = _read_json(EVIDENCE)
    worker_artifacts = worker.get("artifacts", {})
    progress = _validate_progress_prefix(
        _remaining_manifest(_read_json(D390_GEOMETRY)),
        evidence.get("records", []),
    )
    authority = _read_json(EXECUTION_AUTHORITY)
    return {
        "frozen_checks_exact": all(_frozen_checks().values()),
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
            and invocation.get("scientific_input_sha256")
            == _sha(D390_GEOMETRY)
            and invocation.get("preregistration_sha256")
            == _sha(PREREGISTRATION)
            and invocation.get("execution_authority_sha256")
            == _sha(EXECUTION_AUTHORITY)
            and invocation.get("worker_invocation_index") == 1
            and invocation.get("retry_index") == 0
        ),
        "authorization_chain_exact": (
            authorization.get("script_sha256") == _sha(SCRIPT)
            and authorization.get("scientific_input_sha256")
            == _sha(D390_GEOMETRY)
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
            and sentinel.get("scientific_input_sha256")
            == _sha(D390_GEOMETRY)
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
                worker_artifacts.get("progress"), PROGRESS
            )
            and _artifact_reference_exact(
                worker_artifacts.get("evidence"), EVIDENCE
            )
            and _artifact_reference_exact(
                worker_artifacts.get("geometry"), GEOMETRY
            )
            and _artifact_reference_exact(
                worker_artifacts.get("csv"), CSV_PATH
            )
        ),
        "progress_complete_exact": progress["complete_exact"],
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
        "numeric_evidence_integrity_pass": (
            evidence.get("numeric_integrity_pass") is True
        ),
    }


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _render_board(evidence: dict[str, Any]) -> dict[str, Any]:
    image = Image.new("RGB", (1920, 1080), (246, 249, 252))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(35)
    subtitle_font = _load_font(20)
    header_font = _load_font(15)
    row_font = _load_font(13)
    footer_font = _load_font(17)
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

    title_owner = [35, 10, 1885, 70]
    put(
        (45, 20),
        "D392: 남은 35건에 D391의 동결 핵심 순위 판정식 적용",
        title_font,
        (20, 35, 55),
        "title",
        title_owner,
    )
    put(
        (47, 70),
        (
            "D390 기록은 비교용이며 정답표가 아님 · O=순서 증명, "
            "T=평행이동, S=2의 거듭제곱 배율 대조"
        ),
        subtitle_font,
        (55, 70, 90),
        "subtitle",
        [35, 65, 1885, 103],
    )
    summary = evidence["remaining35_summary"]
    coverage = evidence["same_rank_authority_41_coverage"]
    put(
        (47, 107),
        (
            f"D392 안정 {summary['stable_count']}/35 · 모호 "
            f"{summary['ambiguous_count']} · D391과 합친 해결 범위 "
            f"{coverage['resolved_stable_count']}/41 · call29는 아직 별도"
        ),
        subtitle_font,
        (20, 95, 70),
        "summary",
        [35, 102, 1885, 140],
    )

    table_specs = [(40, evidence["records"][:18]), (980, evidence["records"][18:])]
    table_y = 150
    table_width = 900
    header_h = 36
    row_h = 39
    column_widths = [42, 420, 38, 116, 116, 125]
    headers = ["C", "전체 call ID", "n", "D390(참고)", "D392", "대조 O/T/S"]
    row_count = 0
    for table_index, (x0, rows) in enumerate(table_specs):
        x_positions = [x0]
        for width in column_widths:
            x_positions.append(x_positions[-1] + width)
        x_positions[-1] = x0 + table_width
        draw.rounded_rectangle(
            (x0, table_y, x0 + table_width, table_y + header_h + row_h * len(rows)),
            radius=10,
            fill=(255, 255, 255),
            outline=(165, 180, 195),
            width=2,
        )
        draw.rectangle(
            (x0, table_y, x0 + table_width, table_y + header_h),
            fill=(223, 233, 243),
        )
        for column, header in enumerate(headers):
            owner = [
                x_positions[column],
                table_y,
                x_positions[column + 1],
                table_y + header_h,
            ]
            put(
                (owner[0] + 5, owner[1] + 7),
                header,
                header_font,
                (30, 50, 70),
                f"t{table_index}_h{column}",
                owner,
            )
        for local_index, row in enumerate(rows):
            row_count += 1
            y0 = table_y + header_h + local_index * row_h
            y1 = y0 + row_h
            fill = (247, 251, 249) if local_index % 2 == 0 else (238, 247, 243)
            draw.rectangle((x0, y0, x0 + table_width, y1), fill=fill)
            rank = row["rank_authority"]
            history = (
                f"{str(row['d390_historical_class']).replace('_DIMENSIONAL','')}"
                f" r{row['d390_historical_rank']}"
            )
            current = (
                f"{str(rank['authoritative_class']).replace('_DIMENSIONAL','')}"
                f" r{rank['authoritative_rank']}"
                if rank["authoritative_class"] is not None
                else "모호함"
            )
            values = [
                str(row["call_index"]),
                row["call_id"],
                str(row["unique_point_count"]),
                history,
                current,
                (
                    f"{'✓' if row['canonical_order_proof']['pass'] else '×'}/"
                    f"{'✓' if row['translation_control']['pass'] else '×'}/"
                    f"{'✓' if row['scale_control']['pass'] else '×'}"
                ),
            ]
            for column, value in enumerate(values):
                owner = [
                    x_positions[column],
                    y0,
                    x_positions[column + 1],
                    y1,
                ]
                put(
                    (owner[0] + 5, owner[1] + 8),
                    value,
                    row_font,
                    (
                        (25, 115, 75)
                        if column in {4, 5} and row["stable"]
                        else (50, 65, 80)
                    ),
                    f"t{table_index}_r{local_index}_c{column}",
                    owner,
                )
        for x in x_positions[1:-1]:
            draw.line(
                (x, table_y, x, table_y + header_h + row_h * len(rows)),
                fill=(195, 205, 215),
                width=1,
            )
        for local_index in range(len(rows) + 1):
            y = table_y + header_h + local_index * row_h
            draw.line((x0, y, x0 + table_width, y), fill=(205, 214, 222), width=1)

    footer_y = 905
    draw.rounded_rectangle(
        (40, footer_y, 1880, 1048),
        radius=14,
        fill=(232, 241, 250),
        outline=(135, 160, 185),
        width=2,
    )
    footer_lines = [
        (
            "방법: 정확한 이진분수 점 차이 + 모든 기준점/모든 점쌍 + "
            "D391 임계값(0.5/1/2)·hard cap 그대로"
        ),
        (
            "순서: n≤6은 전순열, n>6은 frozen sorted-set 구조 증명 + "
            "독립 재구성 + reverse/rotation/adjacent-swap (n! 전수 아님)"
        ),
        (
            "비채택: call29 강제분류·seam 반영·충돌체/USD/Isaac/PhysX/"
            "29×50mm 원통 물리/q5/접촉/파지 모두 0 · g0a_pass=false"
        ),
    ]
    for index, line in enumerate(footer_lines):
        put(
            (62, footer_y + 17 + index * 38),
            line,
            footer_font,
            (35, 65, 90) if index < 2 else (165, 45, 45),
            f"footer_{index}",
            [40, footer_y, 1880, 1048],
        )
    image.save(BOARD)
    overlaps: list[dict[str, str]] = []
    for left_index, left in enumerate(text_boxes):
        lx0, ly0, lx1, ly1 = left["bbox"]
        for right in text_boxes[left_index + 1 :]:
            rx0, ry0, rx1, ry1 = right["bbox"]
            if max(lx0, rx0) < min(lx1, rx1) and max(ly0, ry0) < min(ly1, ry1):
                overlaps.append(
                    {"left": left["tag"], "right": right["tag"]}
                )
    layout = {
        "artifact": "D392_BOARD_LAYOUT_VALIDATION_V1",
        "path": _rel(BOARD),
        "width": 1920,
        "height": 1080,
        "table_row_count": row_count,
        "left_row_count": 18,
        "right_row_count": 17,
        "observed_call_ids": [
            row["call_id"] for row in evidence["records"]
        ],
        "expected_call_ids": [
            row["call_id"]
            for row in evidence["scientific_input"][
                "remaining35_manifest"
            ]
        ],
        "call_ids_exact_in_evidence_order": (
            [
                row["call_id"] for row in evidence["records"]
            ]
            == [
                row["call_id"]
                for row in evidence["scientific_input"][
                    "remaining35_manifest"
                ]
            ]
        ),
        "summary_values": {
            "stable_count": summary["stable_count"],
            "ambiguous_count": summary["ambiguous_count"],
            "resolved_stable_count": coverage[
                "resolved_stable_count"
            ],
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
            row_count == 35
            and [
                row["call_id"] for row in evidence["records"]
            ]
            == [
                row["call_id"]
                for row in evidence["scientific_input"][
                    "remaining35_manifest"
                ]
            ]
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
            contents="/d392/calls/**",
            name=(
                "D392 35개 terminal 형상의 inspection-only "
                "정규화 atlas"
            ),
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=(9.0, -20.0, 20.0),
                look_target=(9.0, 6.0, 0.0),
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
            name="판정 기준·수치 요약·비채택",
        ),
        row_shares=[0.78, 0.22],
    )
    notification = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d392/notification_buffer/**",
        name="알림 전용 여백 · 판정 내용 없음",
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

    colors = {
        "LINE": [55, 120, 220, 255],
        "FACE_LIKE": [20, 155, 120, 255],
        "FULL_DIMENSIONAL": [225, 120, 35, 255],
        None: [210, 55, 70, 255],
    }
    points_rows: list[dict[str, Any]] = []
    arrow_rows: list[dict[str, Any]] = []
    evidence_by_call = {
        row["call_index"]: row for row in evidence["records"]
    }
    for row in geometry["records"]:
        call_index = row["call_index"]
        authority = evidence_by_call[call_index]["rank_authority"]
        points = np.asarray(
            row["atlas_points_f64_inspection_only"], dtype=np.float64
        )
        origin = np.asarray(
            row["atlas_offset_inspection_only"], dtype=np.float64
        )
        axes = np.asarray(
            row["principal_axis_vectors_atlas_inspection_only"],
            dtype=np.float64,
        )
        point_color = colors.get(
            authority["authoritative_class"], [120, 120, 120, 255]
        )
        points_rows.append(
            {
                "entity_path": (
                    f"d392/calls/c{call_index:02d}/terminal_points"
                ),
                "positions_m": points,
                "radii": [0.065] * len(points),
                "colors": [point_color] * len(points),
                "labels": (
                    [f"C{call_index:02d}"] + [""] * (len(points) - 1)
                    if len(points)
                    else []
                ),
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
        axis_colors = [
            [220, 60, 60, 220],
            [40, 165, 80, 220],
            [50, 95, 220, 220],
        ][: len(axes)]
        arrow_rows.append(
            {
                "entity_path": (
                    f"d392/calls/c{call_index:02d}/principal_axes"
                ),
                "origins_m": np.repeat(
                    origin.reshape(1, 3), len(axes), axis=0
                ),
                "vectors_m": axes,
                "radii": 0.025,
                "colors": axis_colors,
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
    summary = evidence["remaining35_summary"]
    coverage = evidence["same_rank_authority_41_coverage"]
    metadata = {
        "00_legend": (
            "blue=LINE, teal=FACE_LIKE, orange=FULL_DIMENSIONAL; "
            "RGB arrows=inspection-only principal axes"
        ),
        "01_scope": "35 static D390 terminal sets; no science timeline",
        "02_result": (
            f"stable={summary['stable_count']}/35; "
            f"ambiguous={summary['ambiguous_count']}; "
            f"resolved_with_D391={coverage['resolved_stable_count']}/41"
        ),
        "03_order": (
            "n<=6 exhaustive; n>6 frozen sorted-set proof plus finite "
            "generators; not exhaustive n!"
        ),
        "04_authority": (
            "canonical Float64 JSON only; atlas Float32 is inspection-only"
        ),
        "05_nonclaim": (
            "call29/seams/collider/USD/Isaac/PhysX/cylinder/physics/q5/"
            "contact/grasp=0; g0a_pass=false"
        ),
        "case": CASE,
        "attempt": ATTEMPT,
        "numeric_verdict": evidence["numeric_verdict"],
        "canonical_evidence_sha256": _sha(EVIDENCE),
        "canonical_d390_geometry_sha256": _sha(D390_GEOMETRY),
        "display_geometry_sha256": _sha(GEOMETRY),
        "g0a_pass": False,
    }
    original_builder = viz_debug.build_rerun_blueprint
    original_runner = rerun_contract._run
    viewer_calls = 0

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d392_static_35_atlas":
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
                    "stderr": "D392 Viewer maximum one exceeded",
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
            recording_id="g0a_d392_remaining35_rank_coverage",
            blueprint_path=RBL,
            blueprint_mode="d392_static_35_atlas",
            live_viewer=False,
            app_id="roarm_g0a_d392_remaining35_rank_coverage",
        )
        if not saved.get("ok"):
            raise RuntimeError(f"D392 save-only Rerun failed: {saved}")
        expected_entities = ["metadata/run"]
        components: dict[str, list[str]] = {
            "metadata/run": ["TextDocument:text"]
        }
        for row in geometry["records"]:
            call_index = row["call_index"]
            point_path = (
                f"d392/calls/c{call_index:02d}/terminal_points"
            )
            axis_path = f"d392/calls/c{call_index:02d}/principal_axes"
            expected_entities.extend([point_path, axis_path])
            components[point_path] = [
                "CoordinateFrame:frame",
                "Points3D:colors",
                "Points3D:labels",
                "Points3D:positions",
                "Points3D:radii",
            ]
            components[axis_path] = [
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
    dimension_pass = (
        screenshot["width"] in {1920, 3840}
        and screenshot["height"] in {1080, 2160}
        and screenshot["width"] * 9 == screenshot["height"] * 16
    )
    validation["d392_execution_contract"] = {
        "static_atlas_no_decision_sequence_timeline": True,
        "time_panel_hidden": True,
        "notification_buffer_share": 0.22,
        "headless_viewer_invocations": viewer_calls,
        "viewer_maximum": 1,
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "subprocess_timeout_seconds": None,
        "screenshot_dimension_contract_pass": dimension_pass,
    }
    validation["base_contract_pass"] = validation.get("pass") is True
    validation["pass"] = (
        validation["base_contract_pass"]
        and viewer_calls == 1
        and dimension_pass
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
        raise RuntimeError("D392 failure attestation forbids observability")
    supervisor = _read_json(SUPERVISOR)
    worker = _read_json(WORKER_CLAIM)
    numeric_chain = _numeric_chain_checks()
    if supervisor.get("pass") is not True or worker.get("pass") is not True:
        raise RuntimeError("D392 numeric authority is not complete")
    if not all(numeric_chain.values()):
        raise RuntimeError(
            f"D392 numeric artifact chain changed: {numeric_chain}"
        )
    _phase("observability_start")
    started = time.monotonic()
    evidence = _read_json(EVIDENCE)
    geometry = _read_json(GEOMETRY)
    layout = _render_board(evidence)
    rerun = _write_rerun(evidence, geometry)
    manual_template = {
        "artifact": "D392_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
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
    _write_json_x(MANUAL_TEMPLATE, manual_template)
    checks = {
        "numeric_worker_complete_before_observability": (
            EVIDENCE.is_file()
            and WORKER_CLAIM.is_file()
            and SUPERVISOR.is_file()
        ),
        "numeric_artifact_chain_exact": all(numeric_chain.values()),
        "board_layout_pass": layout["pass"],
        "rerun_contract_pass": rerun["pass"],
        "viewer_exactly_one_no_retry": rerun["viewer_calls"] == 1,
    }
    claim = {
        "artifact": "D392_OBSERVABILITY_CLAIM_V1",
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
        raise RuntimeError(f"D392 observability failed: {checks}")
    _require_names(POST_OBSERVABILITY_NAMES, "after_observability")
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
            == "D392_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1"
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
    expected = [
        "prepare_start",
        "prepare_end",
        "supervisor_before_worker",
        "worker_start",
        "canonical_numeric_evidence_committed",
        "worker_end",
        "supervisor_after_worker",
        "observability_start",
        "observability_end",
        "finalize_start",
        "finalize_end",
    ]
    return {
        "observed": observed,
        "expected": expected,
        "exact": observed == expected,
        "monotonic_time_forward": all(
            rows[index]["monotonic_ns"]
            <= rows[index + 1]["monotonic_ns"]
            for index in range(len(rows) - 1)
        ),
        "wall_time_forward": all(
            rows[index]["wall_time_ns"]
            <= rows[index + 1]["wall_time_ns"]
            for index in range(len(rows) - 1)
        ),
    }


def _finalize() -> int:
    _require_names(PRE_FINALIZE_NAMES, "before_finalize")
    if FAILURE.exists():
        raise RuntimeError("D392 failure attestation forbids finalize")
    _phase("finalize_start")
    evidence = _read_json(EVIDENCE)
    supervisor = _read_json(SUPERVISOR)
    worker = _read_json(WORKER_CLAIM)
    observability = _read_json(OBSERVABILITY_CLAIM)
    manual_template = _read_json(MANUAL_TEMPLATE)
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
        "observability_pass": observability.get("pass") is True,
        "observability_artifact_chain_exact": all(
            observability_chain.values()
        ),
        "manual_artifact_exact": (
            manual.get("artifact")
            == "D392_MANUAL_VISUAL_INSPECTION_V1"
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
            == manual_template[
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
        raise RuntimeError(f"D392 finalize prechecks failed: {prechecks}")
    _phase("finalize_end")
    phase_contract = _phase_contract()
    if (
        not phase_contract["exact"]
        or not phase_contract["monotonic_time_forward"]
        or not phase_contract["wall_time_forward"]
    ):
        raise RuntimeError(
            f"D392 phase contract failed: {phase_contract}"
        )
    completion = {
        "artifact": "D392_COMPLETION_SUMMARY_V1",
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
        "coverage_pass": evidence["coverage_pass"],
        "remaining35_summary": evidence["remaining35_summary"],
        "same_rank_authority_41_coverage": (
            evidence["same_rank_authority_41_coverage"]
        ),
        "actual_worker_invocations": 1,
        "worker_retries": 0,
        "viewer_invocations": 1,
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "failure_attestation_exists": False,
        "manual_inspection": _artifact(MANUAL),
        "operational_verdict": (
            "D392_REMAINING35_FROZEN_RANK_AUTHORITY_"
            "SCALABLE_ORDER_PROOF_COVERAGE_COMPLETE"
            if evidence["coverage_pass"]
            else "D392_REMAINING35_COVERAGE_INCOMPLETE_STOP"
        ),
        "g0a_pass": False,
        "pass": True,
    }
    _write_json_x(COMPLETION, completion)
    _require_names(POST_FINALIZE_NAMES, "after_finalize")
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
    global D391
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "run", "worker", "observe", "finalize"),
        required=True,
    )
    args = parser.parse_args()
    try:
        D391 = _load_d391()
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
