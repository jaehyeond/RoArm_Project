#!/usr/bin/env python3
"""Path/provenance wrapper for the frozen D400 one-shot worker.

The D400 worker implementation remains byte-for-byte frozen.  This wrapper
rebinds only the new D401 controller, preregistration, output, and artifact
paths before delegating to the original guarded ``main`` entry point.  Merely
parsing or importing this wrapper does not import or launch Isaac, Kit, PhysX,
Warp, or CUDA and does not create a USD or runtime artifact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO = Path(__file__).resolve().parents[1]
WORKER_PATH = Path(__file__).resolve()
CONTROLLER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d401_d400_runtime_freeze_snapshot_order_repair_controller.py"
)
BASE_D400_WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_worker.py"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d401/"
    "attempt1_d400_runtime_freeze_snapshot_order_repair"
)
PREREG_PATH = OUT_DIR / "d401_preregistration.json"
ATTESTATION_PATH = OUT_DIR / "d401_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d401_proposed_runtime_hash_tuple.json"
RUNTIME_MANIFEST_PATH = OUT_DIR / "d400_runtime_freeze_manifest.json"
INVOCATION_PATH = OUT_DIR / "d400_worker_invocation.json"

EXPECTED_PREREG_SHA256 = (
    "c010578e7307c21e305a3db499fa25204297ceb08192233dbc42023bfd5de5c8"
)
EXPECTED_BASE_D400_WORKER_SHA256 = (
    "e5b4b764012258757a9086edb840af40bcc1637586bc05934fec2674ffbd0f0a"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_d400_worker() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D401 worker requires python -B before loading the frozen "
            "D400 module"
        )
    observed = _sha(BASE_D400_WORKER_PATH)
    if observed != EXPECTED_BASE_D400_WORKER_SHA256:
        raise RuntimeError(
            "frozen D400 worker hash drift: "
            f"{observed} != {EXPECTED_BASE_D400_WORKER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d401_frozen_d400_worker",
        BASE_D400_WORKER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D400 worker import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_base_paths(base: ModuleType) -> None:
    """Rebind only control/provenance paths and artifact basenames."""

    base.WORKER_PATH = WORKER_PATH
    base.CONTROLLER_PATH = CONTROLLER_PATH
    base.PREREG_PATH = PREREG_PATH
    base.OUT_DIR = OUT_DIR
    base.ATTESTATION_PATH = ATTESTATION_PATH
    base.TUPLE_PATH = TUPLE_PATH
    base.RUNTIME_MANIFEST_PATH = RUNTIME_MANIFEST_PATH
    base.INVOCATION_PATH = INVOCATION_PATH
    base.EXPECTED_PREREG_SHA256 = EXPECTED_PREREG_SHA256
    base.CLAIM_NAME = "d400_worker_claim.json"
    base.RAW_SUMMARY_NAME = "d400_worker_raw_summary.json"
    base.PRECLOSE_NAME = "d400_worker_preclose_sentinel.json"
    base.PHASE_NAME = "d400_phase_markers.jsonl"
    base.OWNER_EVIDENCE_NAME = "d400_live_configuration_owner_evidence.json"


def main() -> int:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D401 worker must be launched with python -B before frozen-module "
            "import"
        )
    base = _load_frozen_d400_worker()
    _configure_base_paths(base)
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
