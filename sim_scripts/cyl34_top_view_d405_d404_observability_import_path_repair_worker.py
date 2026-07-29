#!/usr/bin/env python3
"""D405 thin Worker wrapper: pure rebind, zero worker-side changes.

The D404 observability failure was a controller-process import defect only;
the worker subprocess never imports ``roarm_rl`` and its entire technical
chain live-passed in the frozen D404 attempt (repairs 1-4, SDF cook drain,
PhysX property queries, mass gate).  This module therefore carries NO new
repair: it rebinds D405 case paths and provenance onto the hash-pinned
frozen D404 Worker wrapper, whose own delegation installs the four D404
gate-contract repairs and the two D402 harness repairs on the eventually
loaded frozen D400 Worker module (D404 -> D403 -> D402 -> D401 -> D400, all
byte-for-byte unchanged).  Importing this module does not import or launch
Isaac, Kit, PhysX, Warp, CUDA, Rerun, or the actual Worker.
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
    "cyl34_top_view_d405_d404_observability_import_path_repair_controller.py"
)
D404_WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d404_d403_authored_derivative_gate_contract_repair_worker.py"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d405/"
    "attempt1_d404_observability_import_path_repair"
)
PREREG_PATH = OUT_DIR / "d405_preregistration.json"
ATTESTATION_PATH = OUT_DIR / "d405_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d405_proposed_runtime_hash_tuple.json"
RUNTIME_MANIFEST_PATH = OUT_DIR / "d400_runtime_freeze_manifest.json"
INVOCATION_PATH = OUT_DIR / "d400_worker_invocation.json"

EXPECTED_PREREG_SHA256 = (
    "f63e6c69953926697cbb87202fbbb24bd751c897d2dca370373157dd1f4195b2"
)
EXPECTED_D404_WORKER_SHA256 = (
    "baa1e889ef324307bab695188ef3e163a7427a3f28f97150a4392c4f58ef3e82"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_d404_worker() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D405 worker requires python -B before loading the frozen D404 "
            "worker"
        )
    observed = _sha(D404_WORKER_PATH)
    if observed != EXPECTED_D404_WORKER_SHA256:
        raise RuntimeError(
            "frozen D404 worker hash drift: "
            f"{observed} != {EXPECTED_D404_WORKER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d405_frozen_d404_worker",
        D404_WORKER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D404 worker import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_d404_paths(d404: ModuleType) -> None:
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
        setattr(d404, name, value)


def main() -> int:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D405 worker must be launched with python -B before frozen D404 "
            "module load"
        )
    d404 = _load_frozen_d404_worker()
    _configure_d404_paths(d404)
    return int(d404.main())


if __name__ == "__main__":
    raise SystemExit(main())
