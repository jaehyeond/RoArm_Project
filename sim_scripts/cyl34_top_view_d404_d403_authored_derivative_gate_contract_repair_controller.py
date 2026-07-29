#!/usr/bin/env python3
"""D404 thin controller wrapper: authored-derivative gate contract repair.

D403 actual attempt1 fail-stopped at the frozen D400 authored-derivative gate
(worker.py:1378) because of four contract defects of the gate code itself —
Isaac, PhysX, SDF cooking, and the GPU driver recorded zero failures.  The
four repairs are worker-side function-object replacements carried by the D404
Worker wrapper; this controller wrapper changes nothing behavioral and only
rebinds D404 case paths and provenance onto the hash-pinned frozen D403
controller, which itself contributes the fail-closed host-boundary gate and
delegates through D402 (harness authority repairs) and D401 (pre-write Git
snapshot) to the frozen D400 supervision flow.

Importing this module does not import or launch Isaac, Kit, PhysX, Warp,
CUDA, Rerun, or the Worker.  Actual runtime remains separately unapproved
until the user explicitly approves the exact SHA-256 of the D404 four-hash
tuple (the 2026-07-28 sequential fast-execution directive was consumed by the
D403 attempt).
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
    "cyl34_top_view_d404_d403_authored_derivative_gate_contract_repair_worker.py"
)
D403_CONTROLLER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d403_d402_host_boundary_git_repin_rerun_controller.py"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d404/"
    "attempt1_d403_authored_derivative_gate_contract_repair"
)
PREREG_PATH = OUT_DIR / "d404_preregistration.json"
ATTESTATION_PATH = OUT_DIR / "d404_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d404_proposed_runtime_hash_tuple.json"
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
    "4514e824a93902e1b69715df923d43a6c8b86790777b913f3e8c72434b254db0"
)
EXPECTED_D403_CONTROLLER_SHA256 = (
    "187d12f50415d8a33ead42c8cc851adea6614fed9ff777807a7378f757a99d22"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_d403_controller() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D404 controller requires python -B before loading the frozen "
            "D403 controller"
        )
    observed = _sha(D403_CONTROLLER_PATH)
    if observed != EXPECTED_D403_CONTROLLER_SHA256:
        raise RuntimeError(
            "frozen D403 controller hash drift: "
            f"{observed} != {EXPECTED_D403_CONTROLLER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d404_frozen_d403_controller",
        D403_CONTROLLER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D403 controller import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_d403_paths(d403: ModuleType) -> None:
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
        setattr(d403, name, value)


def run_runtime(approved_tuple_sha256: str) -> int:
    """Delegate to the frozen D403 flow (host-boundary gate included)."""

    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D404 controller must be launched with python -B; runtime refused "
            "before frozen D403 module load or any forward-only write"
        )
    d403 = _load_frozen_d403_controller()
    _configure_d403_paths(d403)
    return int(d403.run_runtime(approved_tuple_sha256))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved-tuple-sha256",
        required=True,
        help=(
            "Exact SHA-256 of d404_proposed_runtime_hash_tuple.json explicitly "
            "approved by the user in a later, separate runtime approval."
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
