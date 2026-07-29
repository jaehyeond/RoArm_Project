#!/usr/bin/env python3
"""D403 thin controller wrapper: host-boundary rerun of the D402 contract.

D402 actual attempt1 fail-stopped because its execution environment was a
sandbox with a private ``/dev`` (no NVIDIA device nodes) and a private PID
namespace (controller self-recorded pid 2).  The host driver was continuously
healthy in the same boot.  This wrapper changes exactly two things:

1. the registered Git baseline is repinned to the user-pushed current HEAD
   (``a69a96d``), carried by the D403 preregistration;
2. a fail-closed host-boundary gate runs before any frozen module load so a
   sandboxed invocation stops before any forward-only write.

All science, Isaac/PhysX behavior, both D402 harness repairs, the D401
pre-write Git snapshot, and the D400 SDF preflight remain byte-for-byte frozen
through hash-pinned delegation (D403 -> D402 -> D401 -> D400).  Importing this
module does not import or launch Isaac, Kit, PhysX, Warp, CUDA, Rerun, or the
Worker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType


REPO = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = Path(__file__).resolve()
WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d403_d402_host_boundary_git_repin_rerun_worker.py"
)
D402_CONTROLLER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d402_d401_runtime_stack_item_and_counter_order_authority_repair_controller.py"
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
    "fd403c6633ddd9f0f01615c4da35463547ade2319f0860af5b2db5cfe7e919f0"
)
EXPECTED_D402_CONTROLLER_SHA256 = (
    "af1940a57b05ad9f8afdf8359fc099437360a7ff43eb97259e1ada9eb158da52"
)

GPU_DEVICE_NODES = ("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm")
NAMESPACE_LOW_PID_BOUND = 10


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _host_boundary_gate() -> None:
    """Fail closed before any frozen import or forward-only write."""

    nodes = {node: Path(node).exists() for node in GPU_DEVICE_NODES}
    pid = os.getpid()
    checks = {
        "all_gpu_device_nodes_visible": all(nodes.values()),
        "pid_above_namespace_low_range": pid > NAMESPACE_LOW_PID_BOUND,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "D403 host-boundary gate failed (sandboxed or GPU-less "
            f"environment): nodes={nodes}, pid={pid}, checks={checks}; "
            "nothing was written, no attempt was consumed"
        )


def _load_frozen_d402_controller() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D403 controller requires python -B before loading the frozen "
            "D402 controller"
        )
    observed = _sha(D402_CONTROLLER_PATH)
    if observed != EXPECTED_D402_CONTROLLER_SHA256:
        raise RuntimeError(
            "frozen D402 controller hash drift: "
            f"{observed} != {EXPECTED_D402_CONTROLLER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d403_frozen_d402_controller",
        D402_CONTROLLER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D402 controller import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_d402_paths(d402: ModuleType) -> None:
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
        setattr(d402, name, value)


def run_runtime(approved_tuple_sha256: str) -> int:
    """Delegate to the frozen D402 flow after the host-boundary gate."""

    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D403 controller must be launched with python -B; runtime refused "
            "before frozen D402 module load or any forward-only write"
        )
    _host_boundary_gate()
    d402 = _load_frozen_d402_controller()
    _configure_d402_paths(d402)
    return int(d402.run_runtime(approved_tuple_sha256))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved-tuple-sha256",
        required=True,
        help=(
            "Exact SHA-256 of d403_proposed_runtime_hash_tuple.json under the "
            "user's 2026-07-28 standing sequential-execution approval."
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
