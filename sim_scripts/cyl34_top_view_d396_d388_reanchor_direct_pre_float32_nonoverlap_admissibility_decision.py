#!/usr/bin/env python3
"""D396: decide D388 candidate admissibility from two completed D389 overlaps.

Offline-only.  The decision authority is limited to the two D389 pre-Float32
pair records whose strict and both directional epsilon-zero calculations all
completed and reported positive volume.  D395 is an integrity/background input
only: its mixed-authority table is never used to decide the candidate.

The microscopic overlaps are a frozen *design-contract* rejection.  They are
not evidence of visible manufacturing penetration, a cooked PhysX shape, or a
physical contact.  No clipping, geometry, collider, USD, Isaac, PhysX, cylinder,
q5, contact, target, IK, or path operation is performed here.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASE_PATH = (
    REPO
    / "sim_scripts"
    / "cyl34_top_view_d395_all36_pair_144direction_gate_semantics_propagation.py"
)
SPEC = importlib.util.spec_from_file_location("d396_frozen_helpers", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen helper module")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

CASE = "D396"
ATTEMPT = (
    "attempt1_d388_reanchor_direct_pre_float32_nonoverlap_"
    "admissibility_decision"
)
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d396" / ATTEMPT
START = REPO / "START_HERE.md"
EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
VARIABLES = [
    "d388_direct_pre_float32_positive_pair_nonoverlap_admissibility_decision_v1"
]
GATE_M3 = 1.0e-18
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
AUTHORITY_ENV = "D396_EXECUTION_AUTHORITY_SHA256"
WORKER_AUTHORITY_ENV = "D396_WORKER_AUTHORIZATION_SHA256"

D388_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d388"
    / "attempt1_two_null_moving_support_midlayer_partition_repair_design"
)
D389_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d389"
    / "attempt2_prereg_status_whitespace_repair"
)
D395_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d395"
    / "attempt2_d389_prereg_threshold_path_repair"
)

D388_PREREG = D388_DIR / "d388_preregistration.json"
D388_EVIDENCE = D388_DIR / "d388_two_null_reanchor_design_evidence.json"
D388_GEOMETRY = D388_DIR / "d388_two_null_reanchor_witness_geometry.json"
D389_PREREG = D389_DIR / "d389_preregistration.json"
D389_EVIDENCE = D389_DIR / "d389_numeric_and_tie_audit_evidence.json"
D389_GEOMETRY = D389_DIR / "d389_reconstructed_seam_witness_geometry.json"
D395_EVIDENCE = D395_DIR / "d395_all36_gate_semantics_evidence.json"
D395_COMPLETION = D395_DIR / "d395_completion_summary.json"

INPUT_HASHES = {
    BASE_PATH: "238ddb98bd524b893be981c540f872f4beb6040906839141a451ea3c716d01ce",
    D388_PREREG: "57c4870ee74a19457fe7ec262f3d7d094ddc89833fe0e75b4cc3e4a4d8839eb9",
    D388_EVIDENCE: "582368f093ba08fec0207967e8e24ac24f0a44774dfa1a7b8c82ae2b6781caba",
    D388_GEOMETRY: "c119ededf4400efbef55de4d89ccd6c1c8b4e33d4d3795710b6882d369f5e882",
    D389_PREREG: "f4b38c4c5db311412c5700f792a66be805bbe06abf9be89b533002e1860ce780",
    D389_EVIDENCE: "9423e870c0a218606781943abd2f5c48cb1e5d53cbbf9fb1212294b4ef5bb5dd",
    D389_GEOMETRY: "66042a93389cb8d0e6c867be87382566c753cd965ceda619e947e73de4a607be",
    D395_EVIDENCE: "e44b250b6177aed1089dda3627fc7719f7f8d3b43f0377e8e2109f09e25b7dae",
    D395_COMPLETION: "3aa48cebadaf9922ef32aacc3c29f74ad6f3735a9ba0f986c5fdd66634fc3396",
}

AUTHORITY = OUT_DIR / "d396_execution_authority.json"
PREREG = OUT_DIR / "d396_preregistration.json"
PHASES = OUT_DIR / "d396_phase_markers.jsonl"
INVOCATION = OUT_DIR / "d396_offline_worker_invocation.json"
WORKER_AUTH = OUT_DIR / "d396_worker_authorization.json"
SENTINEL = OUT_DIR / "d396_worker_start_sentinel.json"
STDOUT = OUT_DIR / "d396_offline_worker_stdout.log"
STDERR = OUT_DIR / "d396_offline_worker_stderr.log"
EVIDENCE = OUT_DIR / "d396_direct_overlap_admissibility_evidence.json"
GEOMETRY = OUT_DIR / "d396_direct_overlap_display_geometry.json"
WORKER_CLAIM = OUT_DIR / "d396_offline_worker_claim.json"
SUPERVISOR = OUT_DIR / "d396_offline_worker_supervisor.json"
BOARD = OUT_DIR / "d396_direct_overlap_admissibility_1920x1080.png"
LAYOUT = OUT_DIR / "d396_board_layout_validation.json"
RRD = OUT_DIR / "d396_direct_overlap_admissibility.rerun.rrd"
RBL = OUT_DIR / "d396_direct_overlap_admissibility.rerun.rbl"
RERUN_VALIDATION = OUT_DIR / "d396_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d396_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d396_manual_visual_inspection_template.json"
MANUAL = OUT_DIR / "d396_manual_visual_inspection.json"
OBSERVABILITY = OUT_DIR / "d396_observability_claim.json"
COMPLETION = OUT_DIR / "d396_completion_summary.json"
FAILURE = OUT_DIR / "d396_failure_attestation.json"

MANUAL_KEYS = [
    "board_exact_1920x1080",
    "board_upper_direct_overlap_readable",
    "board_lower_direct_overlap_readable",
    "board_candidate_rejection_readable",
    "board_masked_control_null_readable",
    "board_physical_nonclaim_readable",
    "board_no_text_overlap_or_clipping",
    "rerun_upper_pair_children_visible",
    "rerun_lower_pair_children_visible",
    "rerun_centers_are_display_markers_not_overlap_extent",
    "rerun_metadata_matches_canonical_evidence",
]
PHASE_ORDER = [
    "prepare_start",
    "prepare_end",
    "supervisor_before_worker",
    "worker_start",
    "canonical_evidence_committed",
    "worker_end",
    "supervisor_after_worker",
    "observability_start",
    "observability_end",
    "finalize_start",
    "finalize_end",
]

PREPARED = {AUTHORITY.name, PREREG.name, PHASES.name}
PRE_WORKER = PREPARED | {
    INVOCATION.name,
    WORKER_AUTH.name,
    STDOUT.name,
    STDERR.name,
}
POST_WORKER = PRE_WORKER | {
    SENTINEL.name,
    EVIDENCE.name,
    GEOMETRY.name,
    WORKER_CLAIM.name,
    SUPERVISOR.name,
}
POST_OBSERVE = POST_WORKER | {
    BOARD.name,
    LAYOUT.name,
    RRD.name,
    RBL.name,
    RERUN_VALIDATION.name,
    RERUN_SCREENSHOT.name,
    MANUAL_TEMPLATE.name,
    OBSERVABILITY.name,
}
FINAL = POST_OBSERVE | {MANUAL.name, COMPLETION.name}


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    return BASE._sha(path)


def _read(path: Path) -> Any:
    return BASE._read(path)


def _write_json_x(path: Path, value: Any) -> None:
    BASE._write_json_x(path, value)


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": _rel(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _artifact_matches(reference: Any, path: Path) -> bool:
    return (
        isinstance(reference, dict)
        and reference.get("path") == _rel(path)
        and reference.get("sha256") == _sha(path)
        and reference.get("bytes") == path.stat().st_size
    )


def _git(*args: str) -> str:
    return BASE._git(*args)


def _status_outside_output() -> list[str]:
    prefix = _rel(OUT_DIR) + "/"
    rows = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return [row for row in rows if not row[3:].startswith(prefix)]


def _status_outside_output_sha256() -> str:
    return BASE._canonical_sha(_status_outside_output())


def _inventory() -> set[str]:
    if not OUT_DIR.exists():
        return set()
    return {path.name for path in OUT_DIR.iterdir() if path.is_file()}


def _require_inventory(expected: set[str], where: str) -> None:
    actual = _inventory()
    if actual != expected:
        raise RuntimeError(
            f"{where}: D396 inventory mismatch; "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )


def _append(path: Path, value: Any) -> None:
    BASE._append(path, value)


def _phase(name: str, **extra: Any) -> None:
    if name not in PHASE_ORDER:
        raise ValueError(name)
    rows = []
    if PHASES.exists():
        rows = [
            json.loads(line)
            for line in PHASES.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    expected = PHASE_ORDER[len(rows)]
    if name != expected:
        raise RuntimeError(f"D396 phase {name!r}, expected {expected!r}")
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
        json.loads(line)
        for line in PHASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = [row["phase"] for row in rows]
    return {
        "count": len(rows),
        "expected_count": len(PHASE_ORDER),
        "exact": names == PHASE_ORDER,
        "ordinals_exact": [row["ordinal"] for row in rows]
        == list(range(len(rows))),
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
    checks = {
        f"sha::{_rel(path)}": path.is_file() and _sha(path) == digest
        for path, digest in INPUT_HASHES.items()
    }
    d388 = _read(D388_PREREG)
    d395_evidence = _read(D395_EVIDENCE)
    d395_completion = _read(D395_COMPLETION)
    checks.update(
        {
            "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
            "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
            "d388_gate_exact": (
                d388["frozen_gates"]["positive_volume_child_overlap"] == 0
                and d388["frozen_gates"]["positive_volume_epsilon_m3"]
                == GATE_M3
            ),
            "d395_integrity_background_only": (
                d395_evidence.get("pass") is True
                and d395_evidence.get("hybrid_table_adopted") is False
                and d395_completion.get("pass") is True
                and d395_completion.get("g0a_pass") is False
            ),
        }
    )
    return checks


def _authority_core_checks(authority: dict[str, Any]) -> dict[str, bool]:
    return {
        "artifact": authority.get("artifact")
        == "D396_EXTERNAL_EXECUTION_AUTHORITY_V1",
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
        "git": (
            authority.get("git", {}).get("head") == EXPECTED_HEAD
            and authority.get("git", {}).get("origin_master") == EXPECTED_HEAD
            and _git("rev-parse", "HEAD") == EXPECTED_HEAD
            and _git("rev-parse", "origin/master") == EXPECTED_HEAD
            and authority.get("git", {}).get(
                "status_outside_output_sha256"
            )
            == _status_outside_output_sha256()
            and authority.get("git", {}).get("status_outside_output_count")
            == len(_status_outside_output())
        ),
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
    checks["external_sha"] = (
        os.environ.get(AUTHORITY_ENV) == _sha(AUTHORITY)
    )
    return checks


def _prepare() -> int:
    _require_inventory({AUTHORITY.name}, "before_prepare")
    authority = _read(AUTHORITY)
    authority_checks = _authority_checks(authority)
    frozen = _frozen_checks()
    d389 = _read(D389_EVIDENCE)
    pairs = d389["seam_numeric_provenance_audit"]["pair_results"]
    schema = {
        "d389_pair_count_36": len(pairs) == 36,
        "upper_pair5_identity": (
            pairs[5]["target"] == "UPPER"
            and pairs[5]["left_index"] == 1
            and pairs[5]["right_index"] == 2
        ),
        "lower_pair26_identity": (
            pairs[26]["target"] == "LOWER"
            and pairs[26]["left_index"] == 2
            and pairs[26]["right_index"] == 3
        ),
        "numpy_1_26_0": np.__version__ == "1.26.0",
        "rerun_0_34_1": importlib.metadata.version("rerun-sdk") == "0.34.1",
        "rerun_cli": RERUN_CLI.is_file(),
        "font": FONT.is_file(),
        "python_no_bytecode": sys.dont_write_bytecode,
    }
    if not all(authority_checks.values()):
        raise RuntimeError(f"D396 authority failed: {authority_checks}")
    if not all(frozen.values()):
        raise RuntimeError(f"D396 frozen input failed: {frozen}")
    if not all(schema.values()):
        raise RuntimeError(f"D396 schema failed: {schema}")
    _phase("prepare_start")
    prereg = {
        "artifact": "D396_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "question": (
            "Do the two completed D389 pre-Float32 epsilon-zero positive "
            "overlaps independently make the frozen D388 re-anchor candidate "
            "inadmissible under its zero-positive-overlap contract?"
        ),
        "decision_authority": {
            "source": _rel(D389_EVIDENCE),
            "pair_indices": [5, 26],
            "required": (
                "strict LP plus both directional Float64/Qhull calculations "
                "pass, positive, above 1e-18m3, no Qhull fallback"
            ),
            "d395_hybrid_table_used": False,
            "d395_role": "INPUT_INTEGRITY_AND_BACKGROUND_ONLY",
        },
        "registered_controls": {
            "upper_only": False,
            "lower_only": False,
            "both_enabled": False,
            "both_masked": None,
            "d389_integrity_failure": None,
            "d395_generic_outcome_must_not_change_decision": True,
        },
        "outcomes": {
            "pass": (
                "D396_D388_REANCHOR_DIRECT_PRE_FLOAT32_NONOVERLAP_"
                "INADMISSIBILITY_CERTIFICATE_PASS"
            ),
            "identity_or_numeric_mismatch": (
                "D396_DIRECT_OVERLAP_AUTHORITY_MISMATCH_FAIL_STOP"
            ),
            "no_direct_witness": (
                "D396_D388_ADMISSIBILITY_REMAINS_NULL_FAIL_STOP"
            ),
        },
        "frozen_nonclaims": {
            "d388_verdict_modified": False,
            "d389_unresolved_seams_modified": False,
            "d395_hybrid_table_adopted": False,
            "physical_or_manufacturing_penetration": None,
            "cooked_or_live_physx_overlap": None,
            "materializable_candidate": False,
            "g0a_pass": False,
        },
        "execution_contract": {
            "offline_worker": 1,
            "worker_retries": 0,
            "process_signals": 0,
            "viewer_maximum": 1,
            "viewer_retries": 0,
        },
        "forbidden": {
            "clipping_or_solver_replay": 0,
            "pair_seam_partition_budget_geometry_change": 0,
            "collider_usd_isaac_physx_warp_cuda": 0,
            "cylinder_physics_q5_contact_grasp": 0,
            "target_ik_path_settings": 0,
            "signals_hardware_commit_push": 0,
        },
        "authority": _artifact(AUTHORITY),
        "script": _artifact(SCRIPT),
        "start": _artifact(START),
        "input_sha256": {
            _rel(path): digest for path, digest in INPUT_HASHES.items()
        },
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


def _decision(
    witnesses: list[dict[str, Any]],
    enabled: set[str],
    *,
    d389_integrity: bool,
) -> bool | None:
    if not d389_integrity:
        return None
    active = [
        row for row in witnesses
        if row["witness_id"] in enabled and row["authority_pass"]
    ]
    return False if active else None


def _extract_witness(
    pair: dict[str, Any],
    *,
    pair_index: int,
    target: str,
    left: int,
    right: int,
) -> dict[str, Any]:
    strict = pair["pre_float32_epsilon0"]
    directional = pair["pre_float32_directional_epsilon0"]
    lbr = directional["left_clipped_by_right"]
    rbl = directional["right_clipped_by_left"]
    identity = (
        pair["target"] == target
        and pair["left_index"] == left
        and pair["right_index"] == right
        and pair["adjacent"] is True
    )
    authority_pass = (
        identity
        and pair["pair_provenance_and_solver_pass"] is True
        and strict["calculation_pass"] is True
        and strict["positive_volume"] is True
        and strict["volume_m3"] > GATE_M3
        and strict["solver_residual_contract_pass"] is True
        and strict["qhull_fallback_used"] is False
        and directional["calculation_pass"] is True
        and directional["positive_volume"] is True
        and lbr["calculation_pass"] is True
        and rbl["calculation_pass"] is True
        and lbr["positive_volume"] is True
        and rbl["positive_volume"] is True
        and lbr["volume_m3"] > GATE_M3
        and rbl["volume_m3"] > GATE_M3
        and lbr["qhull_fallback_used"] is False
        and rbl["qhull_fallback_used"] is False
    )
    return {
        "witness_id": f"{target.lower()}_{left}_{right}",
        "pair_index": pair_index,
        "target": target,
        "left_index": left,
        "right_index": right,
        "adjacent": pair["adjacent"],
        "identity_exact": identity,
        "authority_pass": authority_pass,
        "strict": {
            "calculation_pass": strict["calculation_pass"],
            "positive_volume": strict["positive_volume"],
            "volume_m3": strict["volume_m3"],
            "volume_minus_gate_m3": strict["volume_m3"] - GATE_M3,
            "volume_to_gate_ratio": strict["volume_m3"] / GATE_M3,
            "signed_inradius_nm": strict["signed_inradius_nm"],
            "strict_interior_radius_threshold_nm": strict[
                "strict_interior_radius_threshold_nm"
            ],
            "inradius_to_threshold_ratio": (
                strict["signed_inradius_nm"]
                / strict["strict_interior_radius_threshold_nm"]
            ),
            "chebyshev_center_m": strict["chebyshev_center_m"],
            "solver_residual_contract_pass": strict[
                "solver_residual_contract_pass"
            ],
            "qhull_fallback_used": strict["qhull_fallback_used"],
        },
        "directional": {
            "calculation_pass": directional["calculation_pass"],
            "positive_volume": directional["positive_volume"],
            "relative_difference": directional[
                "directional_relative_difference"
            ],
            "left_clipped_by_right": {
                "calculation_pass": lbr["calculation_pass"],
                "positive_volume": lbr["positive_volume"],
                "volume_m3": lbr["volume_m3"],
                "qhull_fallback_used": lbr["qhull_fallback_used"],
            },
            "right_clipped_by_left": {
                "calculation_pass": rbl["calculation_pass"],
                "positive_volume": rbl["positive_volume"],
                "volume_m3": rbl["volume_m3"],
                "qhull_fallback_used": rbl["qhull_fallback_used"],
            },
        },
        "physical_or_manufacturing_penetration_claim": None,
        "cooked_or_live_physx_claim": None,
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any]]:
    d388_prereg = _read(D388_PREREG)
    d388_evidence = _read(D388_EVIDENCE)
    d389 = _read(D389_EVIDENCE)
    source_geometry = _read(D389_GEOMETRY)
    pairs = d389["seam_numeric_provenance_audit"]["pair_results"]
    witnesses = [
        _extract_witness(
            pairs[5], pair_index=5, target="UPPER", left=1, right=2
        ),
        _extract_witness(
            pairs[26], pair_index=26, target="LOWER", left=2, right=3
        ),
    ]
    witness_ids = {row["witness_id"] for row in witnesses}
    controls = {
        "upper_only": _decision(
            witnesses, {"upper_1_2"}, d389_integrity=True
        ),
        "lower_only": _decision(
            witnesses, {"lower_2_3"}, d389_integrity=True
        ),
        "both_enabled": _decision(
            witnesses, witness_ids, d389_integrity=True
        ),
        "both_masked": _decision(witnesses, set(), d389_integrity=True),
        "d389_integrity_failure": _decision(
            witnesses, witness_ids, d389_integrity=False
        ),
        "d395_pass_background": _decision(
            witnesses, witness_ids, d389_integrity=True
        ),
        "d395_generic_fail_background": _decision(
            witnesses, witness_ids, d389_integrity=True
        ),
    }
    candidate_admissible = controls["both_enabled"]
    checks = {
        "two_witnesses": len(witnesses) == 2,
        "both_authoritative": all(row["authority_pass"] for row in witnesses),
        "strict_above_gate": all(
            row["strict"]["volume_m3"] > GATE_M3 for row in witnesses
        ),
        "both_directions_above_gate": all(
            row["directional"]["left_clipped_by_right"]["volume_m3"] > GATE_M3
            and row["directional"]["right_clipped_by_left"]["volume_m3"]
            > GATE_M3
            for row in witnesses
        ),
        "no_qhull_fallback": all(
            row["strict"]["qhull_fallback_used"] is False
            and row["directional"]["left_clipped_by_right"][
                "qhull_fallback_used"
            ]
            is False
            and row["directional"]["right_clipped_by_left"][
                "qhull_fallback_used"
            ]
            is False
            for row in witnesses
        ),
        "d388_zero_positive_overlap_gate": (
            d388_prereg["frozen_gates"]["positive_volume_child_overlap"] == 0
            and d388_prereg["frozen_gates"]["positive_volume_epsilon_m3"]
            == GATE_M3
        ),
        "candidate_rejected": candidate_admissible is False,
        "either_witness_sufficient": (
            controls["upper_only"] is False
            and controls["lower_only"] is False
        ),
        "absence_not_promoted_to_pass": controls["both_masked"] is None,
        "d389_integrity_failure_blocks": (
            controls["d389_integrity_failure"] is None
        ),
        "d395_outcome_does_not_decide": (
            controls["d395_pass_background"] is False
            and controls["d395_generic_fail_background"] is False
        ),
        "d388_already_nonmaterializable": (
            d388_evidence["materializable_candidate"] is False
        ),
    }
    evidence = {
        "artifact": "D396_DIRECT_OVERLAP_ADMISSIBILITY_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": VARIABLES,
        "frozen_positive_volume_gate_m3": GATE_M3,
        "frozen_required_positive_overlap_count": 0,
        "direct_witnesses": witnesses,
        "controls": controls,
        "checks": checks,
        "d388_reanchor_candidate_nonoverlap_admissible": candidate_admissible,
        "d388_verdict_modified": False,
        "d388_original_verdict": d388_evidence["verdict"],
        "d389_unresolved_seam_records_modified": False,
        "d395_predecessor_outcome_used_as_decision_authority": False,
        "d395_hybrid_table_adopted": False,
        "physical_or_manufacturing_penetration": None,
        "cooked_or_live_physx_overlap": None,
        "materializable_candidate": False,
        "g0a_pass": False,
        "scope_counters": {
            "clipping_or_solver_replays": 0,
            "pair_seam_partition_budget_geometry_changes": 0,
            "collider_or_usd_operations": 0,
            "isaac_physx_warp_cuda_operations": 0,
            "cylinder_operations": 0,
            "physics_steps": 0,
            "q5_samples": 0,
            "contact_or_grasp_evaluations": 0,
            "target_ik_path_settings_changes": 0,
            "hardware_operations": 0,
            "process_signals_sent": 0,
        },
        "numeric_verdict": (
            "D396_D388_REANCHOR_DIRECT_PRE_FLOAT32_NONOVERLAP_"
            "INADMISSIBILITY_CERTIFICATE_PASS"
        ),
        "pass": all(checks.values()),
    }
    layers = source_geometry["layers"]
    geometry = {
        "artifact": "D396_DIRECT_OVERLAP_DISPLAY_GEOMETRY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "source_geometry_path": _rel(D389_GEOMETRY),
        "source_geometry_sha256": _sha(D389_GEOMETRY),
        "display_only": True,
        "decision_numeric_authority": False,
        "pairs": [
            {
                "witness_id": "upper_1_2",
                "target": "UPPER",
                "left": {
                    "child_index": 1,
                    "source_name": layers[0]["children"][1]["name"],
                    "source_points_sha256": layers[0]["children"][1][
                        "pre_vertices_sha256"
                    ],
                    "points_f64_m": layers[0]["children"][1][
                        "pre_float32_vertices_f64_m"
                    ],
                },
                "right": {
                    "child_index": 2,
                    "source_name": layers[0]["children"][2]["name"],
                    "source_points_sha256": layers[0]["children"][2][
                        "pre_vertices_sha256"
                    ],
                    "points_f64_m": layers[0]["children"][2][
                        "pre_float32_vertices_f64_m"
                    ],
                },
                "strict_center_m": witnesses[0]["strict"][
                    "chebyshev_center_m"
                ],
                "marker_radius_m": 0.00025,
                "marker_is_not_overlap_extent": True,
            },
            {
                "witness_id": "lower_2_3",
                "target": "LOWER",
                "left": {
                    "child_index": 2,
                    "source_name": layers[1]["children"][2]["name"],
                    "source_points_sha256": layers[1]["children"][2][
                        "pre_vertices_sha256"
                    ],
                    "points_f64_m": layers[1]["children"][2][
                        "pre_float32_vertices_f64_m"
                    ],
                },
                "right": {
                    "child_index": 3,
                    "source_name": layers[1]["children"][3]["name"],
                    "source_points_sha256": layers[1]["children"][3][
                        "pre_vertices_sha256"
                    ],
                    "points_f64_m": layers[1]["children"][3][
                        "pre_float32_vertices_f64_m"
                    ],
                },
                "strict_center_m": witnesses[1]["strict"][
                    "chebyshev_center_m"
                ],
                "marker_radius_m": 0.00025,
                "marker_is_not_overlap_extent": True,
            },
        ],
    }
    return evidence, geometry


def _worker_authorized() -> dict[str, bool]:
    authorization = _read(WORKER_AUTH)
    authority = _read(AUTHORITY)
    return {
        "external": os.environ.get(WORKER_AUTHORITY_ENV) == _sha(WORKER_AUTH),
        "parent": authorization.get("supervisor_pid") == os.getppid(),
        "worker_one": authorization.get("worker_invocation_index") == 1,
        "retry_zero": authorization.get("retry_index") == 0,
        "script": authorization.get("script_sha256") == _sha(SCRIPT),
        "prereg": authorization.get("preregistration_sha256") == _sha(PREREG),
        "authority": authorization.get("execution_authority_sha256")
        == _sha(AUTHORITY),
        "invocation": authorization.get("invocation_sha256")
        == _sha(INVOCATION),
        "outside_status": (
            authority["git"]["status_outside_output_sha256"]
            == _status_outside_output_sha256()
            and authority["git"]["status_outside_output_count"]
            == len(_status_outside_output())
        ),
        "authority_core": all(_authority_core_checks(authority).values()),
        "frozen": all(_frozen_checks().values()),
    }


def _worker() -> int:
    _require_inventory(PRE_WORKER, "worker_before_sentinel")
    authorization = _worker_authorized()
    if not all(authorization.values()):
        raise RuntimeError(f"D396 worker authorization failed: {authorization}")
    started = time.monotonic()
    _write_json_x(
        SENTINEL,
        {
            "artifact": "D396_WORKER_START_SENTINEL_V1",
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
    evidence, geometry = _compute()
    _write_json_x(EVIDENCE, evidence)
    geometry["canonical_evidence_sha256"] = _sha(EVIDENCE)
    _write_json_x(GEOMETRY, geometry)
    _phase("canonical_evidence_committed", evidence_sha256=_sha(EVIDENCE))
    frozen_after = _frozen_checks()
    checks = {
        "numeric_pass": evidence["pass"] is True,
        "candidate_rejected": evidence[
            "d388_reanchor_candidate_nonoverlap_admissible"
        ]
        is False,
        "d395_not_authority": evidence[
            "d395_predecessor_outcome_used_as_decision_authority"
        ]
        is False,
        "frozen_inputs_after": all(frozen_after.values()),
        "scope_zero": all(
            value == 0 for value in evidence["scope_counters"].values()
        ),
    }
    claim = {
        "artifact": "D396_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "authorization_checks": authorization,
        "checks": checks,
        "artifacts": {
            "evidence": _artifact(EVIDENCE),
            "geometry": _artifact(GEOMETRY),
        },
        "elapsed_seconds": time.monotonic() - started,
        "process_signals_sent": 0,
        "pass": all(checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_end", worker_claim_pass=claim["pass"])
    print(json.dumps(claim, ensure_ascii=False))
    return 0 if claim["pass"] else 1


def _run() -> int:
    _require_inventory(PREPARED, "before_run")
    authority = _read(AUTHORITY)
    authority_core = _authority_core_checks(authority)
    if not all(authority_core.values()):
        raise RuntimeError(
            f"D396 execution authority changed before worker: {authority_core}"
        )
    invocation = {
        "artifact": "D396_OFFLINE_WORKER_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": [sys.executable, "-B", _rel(SCRIPT), "--stage", "worker"],
        "worker_invocation_index": 1,
        "retry_index": 0,
        "signals_authorized": 0,
        "script_sha256": _sha(SCRIPT),
        "preregistration_sha256": _sha(PREREG),
        "execution_authority_sha256": _sha(AUTHORITY),
    }
    _write_json_x(INVOCATION, invocation)
    authorization = {
        "artifact": "D396_WORKER_AUTHORIZATION_V1",
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
            cwd=REPO,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        returncode = process.wait()
    supervisor = {
        "artifact": "D396_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "actual_worker_invocations": 1,
        "retries": 0,
        "process_signals_sent": 0,
        "worker_pid": process.pid,
        "worker_exited": True,
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "worker_claim_exists": WORKER_CLAIM.is_file(),
        "worker_claim_pass": (
            WORKER_CLAIM.is_file()
            and _read(WORKER_CLAIM).get("pass") is True
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
    _phase(
        "supervisor_after_worker",
        returncode=returncode,
        pass_=supervisor["pass"],
    )
    if supervisor["pass"]:
        _require_inventory(POST_WORKER, "after_run")
    print(json.dumps(supervisor, ensure_ascii=False))
    return 0 if supervisor["pass"] else 1


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size)


def _render_board(evidence: dict[str, Any]) -> dict[str, Any]:
    image = Image.new("RGB", (1920, 1080), (246, 249, 252))
    draw = ImageDraw.Draw(image)
    title = _font(36)
    header = _font(25)
    body = _font(20)
    small = _font(17)
    draw.text(
        (42, 26),
        "D396 | D388 candidate rejected by two completed direct overlaps",
        fill=(20, 31, 48),
        font=title,
    )
    draw.text(
        (44, 82),
        "Decision authority: D389 pre-Float32 strict + both directions; "
        "D395 hybrid table is not used",
        fill=(58, 72, 92),
        font=header,
    )
    cards = []
    text_boxes = []
    for index, row in enumerate(evidence["direct_witnesses"]):
        x0 = 42 + index * 936
        y0, x1, y1 = 145, x0 + 900, 650
        owner = [x0, y0, x1, y1]
        draw.rounded_rectangle(
            owner,
            radius=18,
            fill=(255, 232, 224),
            outline=(182, 62, 55),
            width=4,
        )
        strict = row["strict"]
        directional = row["directional"]
        lines = [
            (
                f"{row['target']} child {row['left_index']}-"
                f"{row['right_index']} | D389 pair #{row['pair_index']}",
                header,
                (84, 24, 28),
            ),
            (
                f"strict volume: {strict['volume_m3']:.16e} m^3",
                body,
                (34, 43, 58),
            ),
            (
                f"gate ratio: {strict['volume_to_gate_ratio']:.6f} x "
                f"(gate = 1e-18 m^3)",
                body,
                (34, 43, 58),
            ),
            (
                f"signed inradius: {strict['signed_inradius_nm']:.15f} nm "
                f"({strict['inradius_to_threshold_ratio']:.3f} x threshold)",
                body,
                (34, 43, 58),
            ),
            (
                "left-by-right: "
                f"{directional['left_clipped_by_right']['volume_m3']:.16e} m^3",
                body,
                (34, 43, 58),
            ),
            (
                "right-by-left: "
                f"{directional['right_clipped_by_left']['volume_m3']:.16e} m^3",
                body,
                (34, 43, 58),
            ),
            (
                "strict + both directions PASS; positive; Qhull fallback 0",
                body,
                (25, 105, 67),
            ),
            (
                "One witness alone is sufficient for contract rejection.",
                small,
                (120, 48, 45),
            ),
        ]
        for line_index, (text, font, color) in enumerate(lines):
            y = y0 + 28 + line_index * 55
            draw.text((x0 + 26, y), text, fill=color, font=font)
            box = draw.textbbox((x0 + 26, y), text, font=font)
            text_boxes.append(
                {
                    "card": index,
                    "bbox": list(box),
                    "inside_owner": (
                        box[0] >= x0
                        and box[1] >= y0
                        and box[2] <= x1
                        and box[3] <= y1
                    ),
                }
            )
        cards.append({"index": index, "bbox": owner})
    decision_box = [42, 690, 1878, 864]
    draw.rounded_rectangle(
        decision_box,
        radius=18,
        fill=(215, 235, 221),
        outline=(39, 121, 72),
        width=4,
    )
    draw.text(
        (70, 718),
        "Registered decision: D388 candidate non-overlap admissible = FALSE",
        fill=(21, 89, 50),
        font=_font(30),
    )
    draw.text(
        (70, 770),
        "Controls: upper only -> FALSE | lower only -> FALSE | "
        "both masked -> NULL (never promoted to PASS)",
        fill=(31, 50, 67),
        font=body,
    )
    draw.text(
        (70, 815),
        "D388 verdict unchanged; D389 unresolved records unchanged; "
        "candidate remains non-materializable.",
        fill=(31, 50, 67),
        font=body,
    )
    draw.rectangle((0, 900, 1920, 1080), fill=(29, 39, 56))
    footer = [
        "Scope: offline contract decision only. No clipping replay, geometry "
        "change, collider, USD, Isaac, PhysX, cylinder, q5, contact, or grasp.",
        "The 0.070 nm / 0.026 nm inradii are microscopic numerical geometry. "
        "This is NOT a visible manufacturing or live-PhysX penetration claim.",
        "Next: design a new shared-boundary construction with guaranteed "
        "zero-volume overlap before any materialization or physics.",
    ]
    for index, text in enumerate(footer):
        draw.text(
            (42, 920 + index * 45),
            text,
            fill=(247, 228, 176) if index else (240, 245, 250),
            font=small,
        )
    image.save(BOARD)
    layout = {
        "artifact": "D396_BOARD_LAYOUT_VALIDATION_V1",
        "path": _rel(BOARD),
        "width": 1920,
        "height": 1080,
        "cards": cards,
        "text_box_count": len(text_boxes),
        "all_text_inside_owner": all(
            row["inside_owner"] for row in text_boxes
        ),
        "pass": (
            len(cards) == 2
            and all(row["inside_owner"] for row in text_boxes)
        ),
    }
    _write_json_x(LAYOUT, layout)
    return layout


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(
                origin="/",
                contents="/d396/**",
                name="Two direct pre-Float32 overlap witnesses",
                eye_controls=rrb.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                    position=(0.08, -0.14, 0.10),
                    look_target=(0.02, -0.01, -0.02),
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
                name="Authority and nonclaims",
            ),
            row_shares=[0.78, 0.22],
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
        "left": [55, 130, 220, 220],
        "right": [235, 145, 45, 220],
        "center": [220, 45, 65, 255],
    }
    points = []
    expected_entities = ["metadata/run"]
    components = {"metadata/run": ["TextDocument:text"]}
    for pair_index, pair in enumerate(geometry["pairs"]):
        pair_name = "upper" if pair_index == 0 else "lower"
        translation = np.array(
            [0.0, 0.0, 0.04 if pair_index == 0 else -0.04],
            dtype=np.float64,
        )
        for side in ("left", "right"):
            source = np.asarray(
                pair[side]["points_f64_m"], dtype=np.float64
            )
            display = (source + translation).astype(np.float32)
            entity = f"d396/{pair_name}/{side}_child"
            points.append(
                {
                    "entity_path": entity,
                    "positions_m": display.tolist(),
                    "radii": [0.00035] * len(display),
                    "colors": [colors[side]] * len(display),
                    "labels": [f"{pair_name} {side}"]
                    + [""] * (len(display) - 1),
                    "coordinate_frame": "tf#/",
                    "static": True,
                }
            )
            expected_entities.append(entity)
            components[entity] = [
                "CoordinateFrame:frame",
                "Points3D:colors",
                "Points3D:labels",
                "Points3D:positions",
                "Points3D:radii",
            ]
        center = (
            np.asarray(pair["strict_center_m"], dtype=np.float64)
            + translation
        ).astype(np.float32)
        center_entity = f"d396/{pair_name}/strict_center_display_marker"
        points.append(
            {
                "entity_path": center_entity,
                "positions_m": [center.tolist()],
                "radii": [pair["marker_radius_m"]],
                "colors": [colors["center"]],
                "labels": [f"{pair_name} strict center (marker only)"],
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
        expected_entities.append(center_entity)
        components[center_entity] = [
            "CoordinateFrame:frame",
            "Points3D:colors",
            "Points3D:labels",
            "Points3D:positions",
            "Points3D:radii",
        ]
    expected_entities = sorted(expected_entities)
    metadata = {
        "case": CASE,
        "attempt": ATTEMPT,
        "decision": (
            "D388 candidate non-overlap admissible=false from D389 actual "
            "completed pair5 and pair26 pre-Float32 records"
        ),
        "d395_hybrid_table_used_as_authority": False,
        "display": (
            "source child vertices are Float32 inspection copies; upper/lower "
            "pairs are translated by +/-0.04m; red marker radius is not "
            "physical overlap extent"
        ),
        "physical_or_manufacturing_penetration": None,
        "cooked_or_live_physx_overlap": None,
        "canonical_evidence_sha256": _sha(EVIDENCE),
        "canonical_display_geometry_sha256": _sha(GEOMETRY),
        "g0a_pass": False,
    }
    original_builder = viz_debug.build_rerun_blueprint
    original_runner = rerun_contract._run
    viewer_calls = 0

    def routed_builder(mode: str = "robot_geometry") -> Any:
        return _build_blueprint() if mode == "d396_direct_overlap" else original_builder(mode)

    def no_signal_run(command: list[str], *, timeout_s: float) -> dict[str, Any]:
        nonlocal viewer_calls
        del timeout_s
        if any("screenshot" in str(part) for part in command):
            viewer_calls += 1
            if viewer_calls > 1:
                return {
                    "command": command,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "D396 viewer maximum exceeded",
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
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "ok": result.returncode == 0,
                "signals_sent": 0,
                "timeout_ignored_no_signal_authority": True,
            }
        except Exception as exc:
            return {
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": repr(exc),
                "ok": False,
                "signals_sent": 0,
            }

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    rerun_contract._run = no_signal_run
    try:
        saved = viz_debug.log_rerun(
            RRD,
            points=points,
            recording_metadata=metadata,
            recording_id="g0a_d396_direct_overlap_admissibility",
            blueprint_path=RBL,
            blueprint_mode="d396_direct_overlap",
            live_viewer=False,
            app_id="roarm_g0a_d396_direct_overlap",
        )
        if saved.get("ok") is not True:
            raise RuntimeError(f"D396 save-only Rerun failed: {saved}")
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
    base_pass = validation.get("pass") is True
    validation["d396_contract"] = {
        "source_child_entities": 4,
        "strict_center_marker_entities": 2,
        "markers_are_not_overlap_extent": True,
        "upper_lower_display_translation_m": [0.04, -0.04],
        "viewer_invocations": viewer_calls,
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "screenshot_aspect_pass": aspect,
    }
    validation["base_contract_pass"] = base_pass
    validation["pass"] = base_pass and viewer_calls == 1 and aspect
    _write_json_x(RERUN_VALIDATION, validation)
    return {
        "pass": validation["pass"],
        "viewer_invocations": viewer_calls,
        "rrd": _artifact(RRD),
        "rbl": _artifact(RBL),
        "validation": _artifact(RERUN_VALIDATION),
        "screenshot": screenshot,
    }


def _observe() -> int:
    _require_inventory(POST_WORKER, "before_observe")
    if not all(_authority_core_checks(_read(AUTHORITY)).values()):
        raise RuntimeError("D396 authority changed before observe")
    if not all(_frozen_checks().values()):
        raise RuntimeError("D396 frozen input changed before observe")
    if _read(SUPERVISOR).get("pass") is not True:
        raise RuntimeError("D396 worker is not authoritative")
    _phase("observability_start")
    started = time.monotonic()
    evidence = _read(EVIDENCE)
    geometry = _read(GEOMETRY)
    if geometry["canonical_evidence_sha256"] != _sha(EVIDENCE):
        raise RuntimeError("D396 evidence/geometry link changed")
    layout = _render_board(evidence)
    rerun = _write_rerun(evidence, geometry)
    template = {
        "artifact": "D396_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
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
        "worker_artifact_chain": (
            _artifact_matches(
                _read(WORKER_CLAIM)["artifacts"]["evidence"], EVIDENCE
            )
            and _artifact_matches(
                _read(WORKER_CLAIM)["artifacts"]["geometry"], GEOMETRY
            )
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
        "artifact": "D396_OBSERVABILITY_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "numeric_artifacts": {
            "evidence": _artifact(EVIDENCE),
            "geometry": _artifact(GEOMETRY),
            "worker_claim": _artifact(WORKER_CLAIM),
            "supervisor": _artifact(SUPERVISOR),
        },
        "artifacts": {
            "board": _artifact(BOARD),
            "layout": _artifact(LAYOUT),
            "rerun": rerun,
            "manual_template": _artifact(MANUAL_TEMPLATE),
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
        raise RuntimeError(f"D396 observability failed: {checks}")
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
        "outside_status": (
            _read(AUTHORITY)["git"]["status_outside_output_sha256"]
            == _status_outside_output_sha256()
            and _read(AUTHORITY)["git"]["status_outside_output_count"]
            == len(_status_outside_output())
        ),
        "numeric_pass": evidence.get("pass") is True,
        "worker_pass": worker.get("pass") is True,
        "supervisor_pass": supervisor.get("pass") is True,
        "observability_pass": observability.get("pass") is True,
        "worker_artifact_chain": (
            _artifact_matches(worker["artifacts"]["evidence"], EVIDENCE)
            and _artifact_matches(worker["artifacts"]["geometry"], GEOMETRY)
        ),
        "preregistration_links": (
            invocation["preregistration_sha256"] == _sha(PREREG)
            and worker_authorization["preregistration_sha256"] == _sha(PREREG)
            and phase_by_name["prepare_end"]["preregistration_sha256"]
            == _sha(PREREG)
        ),
        "execution_authority_links": (
            invocation["script_sha256"] == _sha(SCRIPT)
            and invocation["execution_authority_sha256"] == _sha(AUTHORITY)
            and worker_authorization["script_sha256"] == _sha(SCRIPT)
            and worker_authorization["execution_authority_sha256"]
            == _sha(AUTHORITY)
            and worker_authorization["invocation_sha256"] == _sha(INVOCATION)
        ),
        "numeric_phase_link": (
            phase_by_name["canonical_evidence_committed"][
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
                observability["artifacts"]["manual_template"], MANUAL_TEMPLATE
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
        "manual_identity": (
            manual.get("artifact") == "D396_MANUAL_VISUAL_INSPECTION_V1"
            and manual.get("case") == CASE
            and manual.get("attempt") == ATTEMPT
        ),
        "manual_paths": (
            manual.get("board_path") == _rel(BOARD)
            and manual.get("rerun_screenshot_path")
            == _rel(RERUN_SCREENSHOT)
        ),
        "manual_keys": set(manual_checks) == set(MANUAL_KEYS),
        "manual_all_true": (
            set(manual_checks) == set(MANUAL_KEYS)
            and all(value is True for value in manual_checks.values())
        ),
        "manual_hashes": (
            manual.get("artifact_hashes") == expected_hashes
            == template["artifact_hashes_to_bind_after_actual_viewing"]
        ),
        "manual_links": (
            manual.get("manual_template_sha256") == _sha(MANUAL_TEMPLATE)
            and manual.get("observability_claim_sha256")
            == _sha(OBSERVABILITY)
        ),
        "manual_observations": (
            isinstance(manual.get("observations"), list)
            and len(manual["observations"]) >= 3
            and all(
                isinstance(value, str) and value.strip()
                for value in manual["observations"]
            )
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
        "candidate_rejected": evidence[
            "d388_reanchor_candidate_nonoverlap_admissible"
        ]
        is False,
        "d395_not_authority": evidence[
            "d395_predecessor_outcome_used_as_decision_authority"
        ]
        is False,
        "scope_zero": all(
            value == 0 for value in evidence["scope_counters"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D396 finalize prechecks failed: {checks}")
    _phase("finalize_end")
    phase_contract = _phase_contract()
    if not all(
        phase_contract[key]
        for key in (
            "exact",
            "ordinals_exact",
            "monotonic_forward",
            "wall_forward",
        )
    ):
        raise RuntimeError(f"D396 phase contract failed: {phase_contract}")
    completion = {
        "artifact": "D396_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": checks,
        "phase_contract": phase_contract,
        "artifacts": {
            "script": _artifact(SCRIPT),
            "start": _artifact(START),
            "authority": _artifact(AUTHORITY),
            "preregistration": _artifact(PREREG),
            "invocation": _artifact(INVOCATION),
            "worker_authorization": _artifact(WORKER_AUTH),
            "sentinel": _artifact(SENTINEL),
            "evidence": _artifact(EVIDENCE),
            "geometry": _artifact(GEOMETRY),
            "worker_claim": _artifact(WORKER_CLAIM),
            "supervisor": _artifact(SUPERVISOR),
            "board": _artifact(BOARD),
            "layout": _artifact(LAYOUT),
            "rrd": _artifact(RRD),
            "rbl": _artifact(RBL),
            "rerun_validation": _artifact(RERUN_VALIDATION),
            "rerun_screenshot": _artifact(RERUN_SCREENSHOT),
            "manual_template": _artifact(MANUAL_TEMPLATE),
            "observability": _artifact(OBSERVABILITY),
            "manual": _artifact(MANUAL),
            "phases": _artifact(PHASES),
        },
        "numeric_verdict": evidence["numeric_verdict"],
        "operational_verdict": (
            "D396_D388_REANCHOR_NONOVERLAP_INADMISSIBILITY_"
            "DECISION_COMPLETE_NO_MATERIALIZATION"
        ),
        "direct_authoritative_witnesses": 2,
        "d388_reanchor_candidate_nonoverlap_admissible": False,
        "d388_verdict_modified": False,
        "d389_unresolved_seam_records_modified": False,
        "d395_hybrid_table_used_as_decision_authority": False,
        "materializable_candidate": False,
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
                "artifact": "D396_FAILURE_ATTESTATION_V1",
                "case": CASE,
                "attempt": ATTEMPT,
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "worker_started": SENTINEL.exists(),
                "process_signals_sent": 0,
                "materializable_candidate": False,
                "g0a_pass": False,
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
