#!/usr/bin/env python3
"""D403 thin Worker wrapper: host-boundary rerun of the D402 Worker contract.

The frozen D402, D401, and D400 Workers remain byte-for-byte unchanged.  This
module only rebinds D403 case paths and provenance onto the hash-pinned frozen
D402 Worker wrapper, then delegates.  Both D402 harness repairs (Item-
compatible ``package.version`` access and serialized-counter registered-
projection authority) are inherited unchanged.

Importing this module does not import or launch Isaac, Kit, PhysX, Warp, CUDA,
Rerun, or the actual Worker.
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
    "cyl34_top_view_d403_d402_host_boundary_git_repin_rerun_controller.py"
)
D402_WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d402_d401_runtime_stack_item_and_counter_order_authority_repair_worker.py"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d403/"
    "attempt1_d402_host_boundary_git_repin_rerun"
)
PREREG_PATH = OUT_DIR / "d403_preregistration.json"
ATTESTATION_PATH = OUT_DIR / "d403_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d403_proposed_runtime_hash_tuple.json"
RUNTIME_MANIFEST_PATH = OUT_DIR / "d400_runtime_freeze_manifest.json"
INVOCATION_PATH = OUT_DIR / "d400_worker_invocation.json"

EXPECTED_PREREG_SHA256 = (
    "fd403c6633ddd9f0f01615c4da35463547ade2319f0860af5b2db5cfe7e919f0"
)
EXPECTED_D402_WORKER_SHA256 = (
    "214d6dcf8e330aa3a6da8a01a614275092462fa337bb1c1fea649c3ec0d654c3"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_d402_worker() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D403 worker requires python -B before loading the frozen D402 "
            "worker"
        )
    observed = _sha(D402_WORKER_PATH)
    if observed != EXPECTED_D402_WORKER_SHA256:
        raise RuntimeError(
            "frozen D402 worker hash drift: "
            f"{observed} != {EXPECTED_D402_WORKER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d403_frozen_d402_worker",
        D402_WORKER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D402 worker import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_d402_paths(d402: ModuleType) -> None:
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
        setattr(d402, name, value)


def main() -> int:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D403 worker must be launched with python -B before frozen D402 "
            "module load"
        )
    d402 = _load_frozen_d402_worker()
    _configure_d402_paths(d402)
    return int(d402.main())


if __name__ == "__main__":
    raise SystemExit(main())
