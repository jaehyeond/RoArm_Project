#!/usr/bin/env python3
"""D402 thin controller wrapper for two D401 harness-authority repairs.

This wrapper reuses the hash-frozen D401 pre-write Git snapshot and one-shot
supervision flow.  It changes only the external supervisor counter gate:
serialized JSON object order is diagnostic, while the exact key set, strict
integer values, and a frozen registered-order projection are authoritative.

Importing this module does not import or launch Isaac, Kit, PhysX, Warp, CUDA,
Rerun, or the Worker.  Actual runtime remains separately unapproved until the
user cites the exact SHA-256 of the D402 four-hash tuple.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = Path(__file__).resolve()
WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d402_d401_runtime_stack_item_and_counter_order_authority_repair_worker.py"
)
D401_CONTROLLER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d401_d400_runtime_freeze_snapshot_order_repair_controller.py"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d402/"
    "attempt1_d401_runtime_stack_item_and_counter_order_authority_repair"
)
PREREG_PATH = OUT_DIR / "d402_preregistration.json"
ATTESTATION_PATH = OUT_DIR / "d402_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d402_proposed_runtime_hash_tuple.json"
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
    "9868b1f60035682295610ce9e38e23d8fa1df37804a69386b00aaf3cf1fdfc4e"
)
EXPECTED_D401_CONTROLLER_SHA256 = (
    "2807353bb36f3309ed7592bdd3b24f4214ebde8b204ab3e253443f51bf63296e"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counter_gate_registered_projection(
    base: ModuleType,
    counters: Any,
) -> dict[str, Any]:
    """Validate counter meaning without trusting serialized object order."""

    is_mapping = isinstance(counters, dict)
    registered_keys = tuple(base.COUNTER_KEYS)
    serialized_keys = list(counters) if is_mapping else []
    exact_key_set = bool(
        is_mapping
        and len(counters) == len(registered_keys)
        and set(counters) == set(registered_keys)
    )
    projection = [
        [key, counters[key] if is_mapping and key in counters else None]
        for key in registered_keys
    ]
    projection_complete = bool(
        exact_key_set
        and len(projection) == len(registered_keys)
        and [row[0] for row in projection] == list(registered_keys)
    )
    types = (
        {key: type(value).__name__ for key, value in counters.items()}
        if is_mapping
        else {}
    )
    integer_types = bool(
        exact_key_set
        and all(type(counters[key]) is int for key in registered_keys)
    )
    exact = {
        key: bool(
            is_mapping
            and key in counters
            and type(counters[key]) is int
            and counters[key] == expected
        )
        for key, expected in base.EXACT_COUNTERS.items()
    }
    zeros = {
        key: bool(
            is_mapping
            and key in counters
            and type(counters[key]) is int
            and counters[key] == 0
        )
        for key in base.ZERO_COUNTERS
    }
    pump = bool(
        is_mapping
        and "simulation_app_update_pumps" in counters
        and type(counters["simulation_app_update_pumps"]) is int
        and 1
        <= counters["simulation_app_update_pumps"]
        <= base.MAX_APP_UPDATE_PUMPS
    )
    checks = {
        "mapping": is_mapping,
        "exact_36_key_set": exact_key_set,
        "registered_order_projection_complete": projection_complete,
        "all_values_exact_int_not_bool": integer_types,
        "exact_14": all(exact.values()) and len(exact) == 14,
        "zero_21": all(zeros.values()) and len(zeros) == 21,
        "one_range_pump": pump,
    }
    return {
        "serialized_iteration_order_diagnostic_only": serialized_keys,
        "serialized_order_used_as_pass_authority": False,
        "registered_order_projection": projection,
        "registered_order_projection_sha256": base._json_sha256(projection),
        "types": types,
        "exact": exact,
        "zero": zeros,
        "simulation_app_update_pumps": (
            counters.get("simulation_app_update_pumps")
            if is_mapping
            else None
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _load_frozen_d401_controller() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D402 controller requires python -B before loading the frozen "
            "D401 controller"
        )
    observed = _sha(D401_CONTROLLER_PATH)
    if observed != EXPECTED_D401_CONTROLLER_SHA256:
        raise RuntimeError(
            "frozen D401 controller hash drift: "
            f"{observed} != {EXPECTED_D401_CONTROLLER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d402_frozen_d401_controller",
        D401_CONTROLLER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D401 controller import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_d401_paths(d401: ModuleType) -> None:
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
        setattr(d401, name, value)


def _install_counter_authority_repair(d401: ModuleType) -> None:
    frozen_loader = d401._load_frozen_d400_controller

    def load_d400_with_repaired_counter_gate() -> ModuleType:
        base = frozen_loader()

        def repaired_counter_gate(counters: Any) -> dict[str, Any]:
            return _counter_gate_registered_projection(base, counters)

        base._counter_gate = repaired_counter_gate
        return base

    d401._load_frozen_d400_controller = load_d400_with_repaired_counter_gate


def _install_nvidia_primary_source_freeze_overlay(d401: ModuleType) -> None:
    """Bind the installed API evidence used by the Item-accessor repair."""

    frozen_merge = d401._merged_runtime_preregistration

    def merge_with_d402_installed_sources(base: ModuleType) -> dict[str, Any]:
        merged = frozen_merge(base)
        repair = base._read_json(PREREG_PATH)
        records = repair["installed_nvidia_primary_sources"]
        if not isinstance(records, dict) or not records:
            raise RuntimeError("D402 installed NVIDIA source records missing")
        installed = merged["installed_primary_source_hashes"]
        added_paths = []
        for record in records.values():
            if (
                not isinstance(record, dict)
                or type(record.get("path")) is not str
                or type(record.get("sha256")) is not str
            ):
                raise RuntimeError(
                    "D402 installed NVIDIA source record is malformed"
                )
            installed[record["path"]] = record["sha256"]
            added_paths.append(record["path"])
        merged["d402_installed_primary_source_freeze_overlay"] = {
            "source": base._rel(PREREG_PATH),
            "count": len(added_paths),
            "paths": sorted(added_paths),
        }
        return merged

    d401._merged_runtime_preregistration = (
        merge_with_d402_installed_sources
    )


def run_runtime(approved_tuple_sha256: str) -> int:
    """Delegate only after binding D402 paths and the one counter repair."""

    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D402 controller must be launched with python -B; runtime refused "
            "before frozen D401 module load or any forward-only write"
        )
    d401 = _load_frozen_d401_controller()
    _configure_d401_paths(d401)
    _install_counter_authority_repair(d401)
    _install_nvidia_primary_source_freeze_overlay(d401)
    return int(d401.run_runtime(approved_tuple_sha256))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved-tuple-sha256",
        required=True,
        help=(
            "Exact SHA-256 of d402_proposed_runtime_hash_tuple.json cited by "
            "the user in a later, separate runtime approval."
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
