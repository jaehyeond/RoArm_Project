#!/usr/bin/env python3
"""D394 attempt2: pre-worker semantic-field repair.

Attempt1 was stopped after prepare and before any worker because static review
found one over-specific output value: the final numeric intersection volume was
written as 0.0 even though the registered proof establishes only the Boolean
gate result (not positive) and leaves the final numeric volume unknown.

This wrapper freezes the attempt1 implementation and changes exactly that
derived output field to ``None``.  All exact geometry, thresholds, controls,
execution contracts, and forbidden-scope counters remain inherited.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BASE_PATH = (
    REPO
    / "sim_scripts"
    / "cyl34_top_view_d394_stable_fullrank_terminal_volume_subthreshold_semantics.py"
)
SPEC = importlib.util.spec_from_file_location("d394_attempt1_frozen_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen D394 attempt1 base")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

base.SCRIPT = Path(__file__).resolve()
base.ATTEMPT = "attempt2_gate_numeric_null_semantics_repair"
base.OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d394"
    / base.ATTEMPT
)

_OUTPUT_NAMES = {
    "EXECUTION_AUTHORITY": "d394_execution_authority.json",
    "PREREGISTRATION": "d394_preregistration.json",
    "PHASES": "d394_phase_markers.jsonl",
    "INVOCATION": "d394_offline_worker_invocation.json",
    "AUTHORIZATION": "d394_worker_authorization.json",
    "SENTINEL": "d394_worker_start_sentinel.json",
    "STDOUT": "d394_offline_worker_stdout.log",
    "STDERR": "d394_offline_worker_stderr.log",
    "PROGRESS": "d394_full10_progress.jsonl",
    "EVIDENCE": "d394_full10_volume_semantics_evidence.json",
    "GEOMETRY": "d394_full10_display_geometry.json",
    "CSV_PATH": "d394_full10_volume_semantics.csv",
    "WORKER_CLAIM": "d394_offline_worker_claim.json",
    "SUPERVISOR": "d394_offline_worker_supervisor.json",
    "BOARD": "d394_full10_volume_semantics_1920x1080.png",
    "LAYOUT": "d394_board_layout_validation.json",
    "RRD": "d394_full10_volume_semantics.rerun.rrd",
    "RBL": "d394_full10_volume_semantics.rerun.rbl",
    "RERUN_VALIDATION": "d394_rerun_validation.json",
    "RERUN_SCREENSHOT": "d394_rerun_inspection.png",
    "MANUAL_TEMPLATE": "d394_manual_visual_inspection_template.json",
    "OBSERVABILITY": "d394_observability_claim.json",
    "MANUAL": "d394_manual_visual_inspection.json",
    "FAILURE": "d394_failure_attestation.json",
    "COMPLETION": "d394_completion_summary.json",
}
for _name, _filename in _OUTPUT_NAMES.items():
    setattr(base, _name, base.OUT_DIR / _filename)

ATTEMPT1 = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d394"
    / "attempt1_stable_fullrank_terminal_volume_subthreshold_semantics"
)
base.EXPECTED_INPUT_SHA256 = {
    **base.EXPECTED_INPUT_SHA256,
    BASE_PATH: "4595bee98e87a192c2e55a57f9f545e5a12548a40301ac0b897e3f2b2bbeeac7",
    ATTEMPT1 / "d394_execution_authority.json": (
        "7bbb98363c23f6a99629b729a6a15f7b9451fdf7de4844a8f8df50ee6ed2913f"
    ),
    ATTEMPT1 / "d394_preregistration.json": (
        "f794cf2b6bdf2df405e776ee209a13e85f571824a93879c78dddcd7f947dd6f5"
    ),
    ATTEMPT1 / "d394_phase_markers.jsonl": (
        "1decb15f393e8f8b2d823b06ce8a193d63d7156ffd5ee5bf1e5d30deb494b9b8"
    ),
    ATTEMPT1 / "d394_pre_worker_semantic_review_stop.json": (
        "b20e2dd0290fc9f42c14772f1060e556488b0ab8e0e5c93d850a97500080e978"
    ),
}

_frozen_analyze_call = base._analyze_call


def _analyze_call_numeric_null_repair(
    d392_row: dict[str, Any], geometry_row: dict[str, Any]
) -> dict[str, Any]:
    result = _frozen_analyze_call(d392_row, geometry_row)
    semantics = result["forward_only_gate_semantics"]
    if semantics != {
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
    }:
        raise RuntimeError("D394 attempt1 semantic source shape changed")
    semantics["derived_gate_volume_m3"] = None
    semantics["numeric_volume_nonclaim"] = (
        "The final intersection volume was not calculated; only the Boolean "
        "frozen-gate decision is proven by an upper bound."
    )
    return result


base._analyze_call = _analyze_call_numeric_null_repair


if __name__ == "__main__":
    raise SystemExit(base.main())
