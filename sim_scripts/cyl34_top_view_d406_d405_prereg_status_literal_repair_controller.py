#!/usr/bin/env python3
"""D406 thin Controller wrapper: pure rebind, zero code repairs.

The D405 actual attempt1 passed every infrastructure and antecedent gate
(approval tuple, freeze manifest with all 47 dirty paths exactly allowlisted,
environment gate, Isaac launch, and the D405 pre-delegation observability
probe proving repair R1 live) and then fail-stopped inside the frozen D400
worker's preregistration admission (worker.py:2517-2518), because the D405
preregistration authored a "more descriptive" status string instead of the
frozen literal ``PREREGISTERED_NOT_EXECUTED`` the frozen consumer requires.
That was an authoring defect of the preregistration DOCUMENT, not of any
code in the chain (DECISIONS D405).

This module therefore carries NO new repair and NO new variable: it rebinds
the D406 case paths and provenance (including the D406 preregistration,
whose status literal is now derived from the frozen worker source) onto the
hash-pinned frozen D405 controller.  The D405 controller's own delegation
then installs, unchanged, the three D405 observability repairs plus the
fail-closed pre-delegation probe, the four D404 gate-contract repairs, the
two D402 harness repairs, the D403 host-boundary gate, and the D401
pre-write git snapshot on the eventually loaded frozen D400 preflight
(D406 -> D405 -> D404 -> D403 -> D402 -> D401 -> D400, all byte-for-byte
unchanged).  Importing this module does not import or launch Isaac, Kit,
PhysX, Warp, CUDA, Rerun, or the Worker.  The single runtime invocation is
authorized by the user's 2026-07-28 explicit D406 directive and is consumed
by this attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import traceback
from pathlib import Path
from types import ModuleType


REPO = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = Path(__file__).resolve()
WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d406_d405_prereg_status_literal_repair_worker.py"
)
D405_CONTROLLER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d405_d404_observability_import_path_repair_controller.py"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d406/"
    "attempt1_d405_prereg_status_literal_repair"
)
PREREG_PATH = OUT_DIR / "d406_preregistration.json"
ATTESTATION_PATH = OUT_DIR / "d406_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d406_proposed_runtime_hash_tuple.json"
RUNTIME_MANIFEST_PATH = OUT_DIR / "d400_runtime_freeze_manifest.json"
PHASE_PATH = OUT_DIR / "d400_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d400_worker_invocation.json"
CLAIM_PATH = OUT_DIR / "d400_worker_claim.json"
KIT_LOG_PATH = OUT_DIR / "d400_kit_log.txt"
RAW_PATH = OUT_DIR / "d400_worker_raw_summary.json"
PRECLOSE_PATH = OUT_DIR / "d400_worker_preclose_sentinel.json"
OWNER_EVIDENCE_PATH = OUT_DIR / "d400_live_configuration_owner_evidence.json"
SUPERVISOR_PATH = OUT_DIR / "d400_worker_supervisor.json"
COMPLETION_PATH = OUT_DIR / "d400_completion_summary.json"
RRD_PATH = OUT_DIR / "d400_sdf_preflight.rrd"
RBL_PATH = OUT_DIR / "d400_sdf_preflight.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d400_rerun_validation.json"
BOARD_PATH = OUT_DIR / "d400_decision_board_1920x1080.png"
RERUN_SCREENSHOT_PATH = OUT_DIR / "d400_rerun_viewer_1920x1080.png"
RERUN_RECEIPT_PATH = OUT_DIR / "d400_rerun_render_receipt.json"
MANUAL_INSPECTION_PATH = OUT_DIR / "d400_manual_visual_inspection.json"
COLLISION_ASSET_ROOT = OUT_DIR / "collision_asset"

EXPECTED_PREREG_SHA256 = (
    "c49801577f44590774927ca2b74a23be233a536025db2d784d042faf01c4c7de"
)
EXPECTED_D405_CONTROLLER_SHA256 = (
    "eb54b29025270363d18cbcc42ed7f248304bbd543e2741df95c1b5fa3b8d6365"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_d405_controller() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D406 controller requires python -B before loading the frozen "
            "D405 controller"
        )
    observed = _sha(D405_CONTROLLER_PATH)
    if observed != EXPECTED_D405_CONTROLLER_SHA256:
        raise RuntimeError(
            "frozen D405 controller hash drift: "
            f"{observed} != {EXPECTED_D405_CONTROLLER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d406_frozen_d405_controller",
        D405_CONTROLLER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D405 controller import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_d405_paths(d405: ModuleType) -> None:
    bindings = {
        "CONTROLLER_PATH": CONTROLLER_PATH,
        "WORKER_PATH": WORKER_PATH,
        "OUT_DIR": OUT_DIR,
        "PREREG_PATH": PREREG_PATH,
        "ATTESTATION_PATH": ATTESTATION_PATH,
        "TUPLE_PATH": TUPLE_PATH,
        "RUNTIME_MANIFEST_PATH": RUNTIME_MANIFEST_PATH,
        "PHASE_PATH": PHASE_PATH,
        "INVOCATION_PATH": INVOCATION_PATH,
        "CLAIM_PATH": CLAIM_PATH,
        "KIT_LOG_PATH": KIT_LOG_PATH,
        "RAW_PATH": RAW_PATH,
        "PRECLOSE_PATH": PRECLOSE_PATH,
        "OWNER_EVIDENCE_PATH": OWNER_EVIDENCE_PATH,
        "SUPERVISOR_PATH": SUPERVISOR_PATH,
        "COMPLETION_PATH": COMPLETION_PATH,
        "RRD_PATH": RRD_PATH,
        "RBL_PATH": RBL_PATH,
        "RERUN_VALIDATION_PATH": RERUN_VALIDATION_PATH,
        "BOARD_PATH": BOARD_PATH,
        "RERUN_SCREENSHOT_PATH": RERUN_SCREENSHOT_PATH,
        "RERUN_RECEIPT_PATH": RERUN_RECEIPT_PATH,
        "MANUAL_INSPECTION_PATH": MANUAL_INSPECTION_PATH,
        "COLLISION_ASSET_ROOT": COLLISION_ASSET_ROOT,
        "EXPECTED_PREREG_SHA256": EXPECTED_PREREG_SHA256,
    }
    for name, value in bindings.items():
        setattr(d405, name, value)


def run_runtime(approved_tuple_sha256: str) -> int:
    """Rebind the D406 case onto the frozen D405 controller and delegate."""

    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D406 controller must be launched with python -B; runtime "
            "refused before frozen D405 module load or any forward-only "
            "write"
        )
    d405 = _load_frozen_d405_controller()
    _configure_d405_paths(d405)
    return int(d405.run_runtime(approved_tuple_sha256))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved-tuple-sha256",
        required=True,
        help=(
            "Exact SHA-256 of d406_proposed_runtime_hash_tuple.json under "
            "the user's 2026-07-28 explicit D406 directive."
        ),
    )
    args = parser.parse_args()
    try:
        return run_runtime(args.approved_tuple_sha256)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
