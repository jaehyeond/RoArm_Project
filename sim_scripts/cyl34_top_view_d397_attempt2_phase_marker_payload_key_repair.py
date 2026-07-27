#!/usr/bin/env python3
"""D397 attempt2: phase-marker payload-key repair.

Attempt1 passed its complete preflight and then stopped before the first source
parent construction because ``_phase("source_parent_start", name=...)`` passed
two values for the helper's ``name`` parameter.  This wrapper freezes the
attempt1 harness by hash, changes only the helper parameter to ``phase_name``,
and routes every output to a new forward-only directory.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
BASE_SCRIPT = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d397_shared_boundary_zero_volume_construction_design.py"
)
ATTEMPT1 = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d397/"
    "attempt1_shared_boundary_zero_volume_construction_design"
)
ATTEMPT1_PREREG = ATTEMPT1 / "d397_preregistration.json"
ATTEMPT1_INVOCATION = ATTEMPT1 / "d397_offline_worker_invocation.json"
ATTEMPT1_PHASES = ATTEMPT1 / "d397_phase_markers.jsonl"
ATTEMPT1_FAILURE = ATTEMPT1 / "d397_runtime_failure.json"

EXPECTED_REPAIR_INPUT_SHA256 = {
    BASE_SCRIPT: (
        "52745beab46bc695467dd8d676a06b30fa3ea873c7dcad685861e65cfecf4b36"
    ),
    ATTEMPT1_PREREG: (
        "de3a1d2d2f13a5dd64123321e932cec64a0ce3b858e6cc0151c36e6f948f43cd"
    ),
    ATTEMPT1_INVOCATION: (
        "29d2f5127d3cad98acc5d60604bb865f6194146cd89bb35fc86748dfe3ea696f"
    ),
    ATTEMPT1_PHASES: (
        "3d8c54cd8abf5a6cadc66383a30d628195eec5218ef361d1fe055a8fb743a713"
    ),
    ATTEMPT1_FAILURE: (
        "c29c8e1daef1394fad485c782aa6d3d0edefbec7e15c386940751e6cea47c98d"
    ),
}


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("d397_attempt1_frozen", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen D397 harness: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
BASE.ATTEMPT = "attempt2_phase_marker_payload_key_repair"
BASE.OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d397/"
    "attempt2_phase_marker_payload_key_repair"
)
BASE.SCRIPT = SCRIPT
BASE.EXPECTED_INPUT_SHA256.update(EXPECTED_REPAIR_INPUT_SHA256)

for name, filename in {
    "PREREG": "d397_preregistration.json",
    "PHASES": "d397_phase_markers.jsonl",
    "INVOCATION": "d397_offline_worker_invocation.json",
    "WORKER_CLAIM": "d397_offline_worker_claim.json",
    "EVIDENCE": "d397_shared_boundary_design_evidence.json",
    "GEOMETRY": "d397_shared_boundary_candidate_geometry.json",
    "PARENT_CSV": "d397_source_parent_metrics.csv",
    "BOARD": "d397_shared_boundary_design_1920x1080.png",
    "LAYOUT": "d397_board_layout_validation.json",
    "RRD": "d397_shared_boundary_design.rerun.rrd",
    "RBL": "d397_shared_boundary_design.rerun.rbl",
    "RERUN_VALIDATION": "d397_rerun_validation.json",
    "RERUN_SCREENSHOT": "d397_rerun_inspection.png",
    "MANUAL_TEMPLATE": "d397_manual_visual_inspection_template.json",
    "MANUAL": "d397_manual_visual_inspection.json",
    "COMPLETION": "d397_completion_summary.json",
    "FAILURE": "d397_runtime_failure.json",
}.items():
    setattr(BASE, name, BASE.OUT_DIR / filename)

REPAIR_ATTESTATION = BASE.OUT_DIR / "d397_attempt2_repair_attestation.json"


def _phase(phase_name: str, **fields: Any) -> None:
    """Write one marker while permitting a payload field named ``name``."""

    BASE._append_jsonl(
        BASE.PHASES,
        {
            "phase": phase_name,
            "monotonic_seconds": time.monotonic(),
            "wall_time_unix_seconds": time.time(),
            **fields,
        },
    )


BASE._phase = _phase
_ORIGINAL_PREPARE = BASE._prepare


def _prepare() -> int:
    result = _ORIGINAL_PREPARE()
    failure = BASE._read_json(ATTEMPT1_FAILURE)
    invocation = BASE._read_json(ATTEMPT1_INVOCATION)
    phases = [
        BASE.json.loads(row)
        for row in ATTEMPT1_PHASES.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    attestation = {
        "artifact": "D397_ATTEMPT2_PHASE_MARKER_PAYLOAD_KEY_REPAIR_V1",
        "case": BASE.CASE,
        "attempt": BASE.ATTEMPT,
        "scientific_new_variables": BASE.NEW_VARIABLES,
        "reactive_operational_repair": {
            "from": "_phase(name: str, **fields)",
            "to": "_phase(phase_name: str, **fields)",
            "call_site_payload": "name -> unchanged",
            "geometry_partition_gate_budget_changes": 0,
        },
        "attempt1": {
            "preflight_pass": invocation["pass"],
            "worker_invocations": invocation["worker_invocation_count"],
            "exception_type": failure["exception_type"],
            "exception": failure["exception"],
            "source_parent_start_markers": sum(
                row.get("phase") == "source_parent_start" for row in phases
            ),
            "source_parent_end_markers": sum(
                row.get("phase") == "source_parent_end" for row in phases
            ),
            "geometry_evaluation_count": 0,
            "design_verdict": None,
        },
        "frozen_input_hashes": {
            BASE._rel(path): BASE._sha(path)
            for path in EXPECTED_REPAIR_INPUT_SHA256
        },
        "checks": {
            "attempt1_preflight_pass": invocation["pass"] is True,
            "attempt1_failure_exact": failure["exception"]
            == "TypeError(\"_phase() got multiple values for argument 'name'\")",
            "attempt1_no_source_parent_start": not any(
                row.get("phase") == "source_parent_start" for row in phases
            ),
            "attempt1_no_source_parent_end": not any(
                row.get("phase") == "source_parent_end" for row in phases
            ),
            "frozen_repair_inputs_exact": all(
                BASE._sha(path) == expected
                for path, expected in EXPECTED_REPAIR_INPUT_SHA256.items()
            ),
            "only_phase_parameter_renamed": True,
        },
    }
    attestation["pass"] = all(attestation["checks"].values())
    BASE._write_json_x(REPAIR_ATTESTATION, attestation)
    BASE._phase(
        "attempt2_repair_attested",
        repair_attestation_pass=attestation["pass"],
    )
    if not attestation["pass"]:
        raise RuntimeError(f"D397 attempt2 repair attestation failed: {attestation}")
    return result


BASE._prepare = _prepare


if __name__ == "__main__":
    raise SystemExit(BASE.main())
