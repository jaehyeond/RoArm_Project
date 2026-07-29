#!/usr/bin/env python3
"""D406 thin Worker wrapper: pure rebind, zero worker-side changes.

The D405 failure was an authoring defect of the D405 preregistration
document (its status field was not the frozen admission literal), observed
inside the frozen D400 worker's admission check (worker.py:2517-2518).  No
worker code was ever at fault; the entire technical chain live-passed in
the frozen D404 attempt.  This module therefore carries NO repair: it
rebinds the D406 case paths and provenance (including the D406
preregistration whose status literal is derived from the frozen worker
source) onto the hash-pinned frozen D405 worker wrapper, whose delegation
installs the four D404 gate-contract repairs and the two D402 harness
repairs on the eventually loaded frozen D400 worker module (D406 -> D405 ->
D404 -> D403 -> D402 -> D401 -> D400, all byte-for-byte unchanged).
Importing this module does not import or launch Isaac, Kit, PhysX, Warp,
CUDA, Rerun, or the actual Worker.
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
    "cyl34_top_view_d406_d405_prereg_status_literal_repair_controller.py"
)
D405_WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d405_d404_observability_import_path_repair_worker.py"
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
INVOCATION_PATH = OUT_DIR / "d400_worker_invocation.json"

EXPECTED_PREREG_SHA256 = (
    "c49801577f44590774927ca2b74a23be233a536025db2d784d042faf01c4c7de"
)
EXPECTED_D405_WORKER_SHA256 = (
    "938af5dc2981da26e3e2a5b60b92df7f5ba99ce52f78d2512f99715277743912"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_d405_worker() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D406 worker requires python -B before loading the frozen D405 "
            "worker"
        )
    observed = _sha(D405_WORKER_PATH)
    if observed != EXPECTED_D405_WORKER_SHA256:
        raise RuntimeError(
            "frozen D405 worker hash drift: "
            f"{observed} != {EXPECTED_D405_WORKER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d406_frozen_d405_worker",
        D405_WORKER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D405 worker import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_d405_paths(d405: ModuleType) -> None:
    bindings = {
        "WORKER_PATH": WORKER_PATH,
        "CONTROLLER_PATH": CONTROLLER_PATH,
        "OUT_DIR": OUT_DIR,
        "PREREG_PATH": PREREG_PATH,
        "ATTESTATION_PATH": ATTESTATION_PATH,
        "TUPLE_PATH": TUPLE_PATH,
        "RUNTIME_MANIFEST_PATH": RUNTIME_MANIFEST_PATH,
        "INVOCATION_PATH": INVOCATION_PATH,
        "EXPECTED_PREREG_SHA256": EXPECTED_PREREG_SHA256,
    }
    for name, value in bindings.items():
        setattr(d405, name, value)


def main() -> int:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D406 worker must be launched with python -B before frozen D405 "
            "module load"
        )
    d405 = _load_frozen_d405_worker()
    _configure_d405_paths(d405)
    return int(d405.main())


if __name__ == "__main__":
    raise SystemExit(main())
