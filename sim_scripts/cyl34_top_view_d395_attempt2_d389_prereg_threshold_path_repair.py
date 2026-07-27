#!/usr/bin/env python3
"""D395 attempt2: repair only the immutable D389 threshold schema path.

Attempt1 stopped during prepare before any worker because it looked for
``frozen_contract.positive_volume_epsilon_m3``.  The immutable D389
preregistration stores the same registered value under
``frozen_constants.positive_volume_epsilon_m3``.  This wrapper preserves the
attempt1 script and all scientific/observability logic, redirects every output
to a new forward-only path, and replaces only that schema lookup.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BASE_PATH = (
    REPO
    / "sim_scripts"
    / "cyl34_top_view_d395_all36_pair_144direction_gate_semantics_propagation.py"
)
SPEC = importlib.util.spec_from_file_location("d395_attempt1_frozen", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen D395 attempt1 module")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

BASE.SCRIPT = Path(__file__).resolve()
BASE.ATTEMPT = "attempt2_d389_prereg_threshold_path_repair"
BASE.OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d395"
    / BASE.ATTEMPT
)
ATTEMPT1_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d395"
    / "attempt1_all36_pair_144direction_gate_semantics_propagation"
)
BASE.INPUT_HASHES = dict(BASE.INPUT_HASHES)
BASE.INPUT_HASHES.update(
    {
        BASE_PATH: (
            "238ddb98bd524b893be981c540f872f4beb6040906839141a451ea3c716d01ce"
        ),
        ATTEMPT1_DIR / "d395_execution_authority.json": (
            "452cc6f529351e779207d946a4e93d041a96deb61ec118798edc512b80df3bb0"
        ),
        ATTEMPT1_DIR / "d395_failure_attestation.json": (
            "3ee158a06c53ab40640c28c83bc269e08fc3dc546620b57befaa2037a9fe30b0"
        ),
    }
)

_OUTPUTS = {
    "AUTHORITY": "d395_execution_authority.json",
    "PREREG": "d395_preregistration.json",
    "PHASES": "d395_phase_markers.jsonl",
    "INVOCATION": "d395_offline_worker_invocation.json",
    "WORKER_AUTH": "d395_worker_authorization.json",
    "SENTINEL": "d395_worker_start_sentinel.json",
    "STDOUT": "d395_offline_worker_stdout.log",
    "STDERR": "d395_offline_worker_stderr.log",
    "PROGRESS": "d395_failed41_progress.jsonl",
    "EVIDENCE": "d395_all36_gate_semantics_evidence.json",
    "CSV_PATH": "d395_all144_direction_semantics.csv",
    "GEOMETRY": "d395_pair_atlas_geometry.json",
    "WORKER_CLAIM": "d395_offline_worker_claim.json",
    "SUPERVISOR": "d395_offline_worker_supervisor.json",
    "BOARD": "d395_all36_gate_semantics_1920x1080.png",
    "LAYOUT": "d395_board_layout_validation.json",
    "RRD": "d395_all36_gate_semantics.rerun.rrd",
    "RBL": "d395_all36_gate_semantics.rerun.rbl",
    "RERUN_VALIDATION": "d395_rerun_validation.json",
    "RERUN_SCREENSHOT": "d395_rerun_inspection.png",
    "MANUAL_TEMPLATE": "d395_manual_visual_inspection_template.json",
    "MANUAL": "d395_manual_visual_inspection.json",
    "OBSERVABILITY": "d395_observability_claim.json",
    "FAILURE": "d395_failure_attestation.json",
    "COMPLETION": "d395_completion_summary.json",
}
for _name, _filename in _OUTPUTS.items():
    setattr(BASE, _name, BASE.OUT_DIR / _filename)


def _repaired_frozen_checks() -> dict[str, bool]:
    checks = {
        f"sha::{BASE._rel(path)}": path.is_file() and BASE._sha(path) == digest
        for path, digest in BASE.INPUT_HASHES.items()
    }
    d389_prereg = BASE._read(BASE.D389_PREREG)
    checks.update(
        {
            "head_exact": BASE._git("rev-parse", "HEAD")
            == BASE.EXPECTED_HEAD,
            "origin_exact": BASE._git("rev-parse", "origin/master")
            == BASE.EXPECTED_HEAD,
            "d389_threshold_exact": (
                d389_prereg["frozen_constants"][
                    "positive_volume_epsilon_m3"
                ]
                == 1.0e-18
            ),
            "d389_threshold_schema_path_repaired": (
                "frozen_contract" not in d389_prereg
                and "frozen_constants" in d389_prereg
            ),
            "d393_complete_and_call29_null": (
                BASE._read(BASE.D393_COMPLETION).get("pass") is True
                and BASE._read(BASE.D393_COMPLETION).get(
                    "call29_authoritative_rank"
                )
                is None
                and BASE._read(BASE.D393_COMPLETION).get(
                    "call29_authoritative_class"
                )
                is None
            ),
            "d394_numeric_pass": (
                BASE._read(BASE.D394_EVIDENCE).get("pass") is True
                and BASE._read(BASE.D394_EVIDENCE).get(
                    "pair_or_seam_verdict_updated"
                )
                is False
                and BASE._read(BASE.D394_EVIDENCE).get("call29_rank") is None
                and BASE._read(BASE.D394_EVIDENCE).get("call29_class") is None
            ),
            "d394_complete": BASE._read(BASE.D394_COMPLETION).get("pass")
            is True,
        }
    )
    return checks


BASE._frozen_checks = _repaired_frozen_checks

BASE.PREPARED = {BASE.AUTHORITY.name, BASE.PREREG.name, BASE.PHASES.name}
BASE.PRE_WORKER = BASE.PREPARED | {
    BASE.INVOCATION.name,
    BASE.WORKER_AUTH.name,
    BASE.STDOUT.name,
    BASE.STDERR.name,
}
BASE.POST_WORKER = BASE.PRE_WORKER | {
    BASE.SENTINEL.name,
    BASE.PROGRESS.name,
    BASE.EVIDENCE.name,
    BASE.CSV_PATH.name,
    BASE.GEOMETRY.name,
    BASE.WORKER_CLAIM.name,
    BASE.SUPERVISOR.name,
}
BASE.POST_OBSERVE = BASE.POST_WORKER | {
    BASE.BOARD.name,
    BASE.LAYOUT.name,
    BASE.RRD.name,
    BASE.RBL.name,
    BASE.RERUN_VALIDATION.name,
    BASE.RERUN_SCREENSHOT.name,
    BASE.MANUAL_TEMPLATE.name,
    BASE.OBSERVABILITY.name,
}
BASE.FINAL = BASE.POST_OBSERVE | {BASE.MANUAL.name, BASE.COMPLETION.name}


def main() -> int:
    return BASE.main()


if __name__ == "__main__":
    raise SystemExit(main())
