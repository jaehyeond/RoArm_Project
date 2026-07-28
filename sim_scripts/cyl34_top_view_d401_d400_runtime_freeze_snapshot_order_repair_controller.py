#!/usr/bin/env python3
"""D401 control-only repair wrapper for the frozen D400 SDF preflight.

This module changes exactly two pieces of control provenance:

1. the registered Git baseline is repinned to the user-approved current HEAD;
2. Git HEAD/origin/status and dirty-file hashes are captured in memory before
   the first runtime artifact is written.

The frozen D400 controller supplies all later environment, watchdog, worker,
Isaac/PhysX, SDF, property-query, cleanup, and Rerun behavior.  Merely parsing
or importing this wrapper does not import or launch Isaac, Kit, PhysX, Warp, or
the worker.  Runtime remains separately unapproved until a user cites the exact
SHA-256 of the D401 four-hash tuple.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import secrets
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = Path(__file__).resolve()
WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d401_d400_runtime_freeze_snapshot_order_repair_worker.py"
)
BASE_D400_CONTROLLER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_preflight.py"
)
BASE_D400_WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_worker.py"
)
BASE_D400_PREREG_PATH = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d400/"
    "attempt1_gripper_link_sdf_res256_live_cook_articulation_preflight/"
    "d400_preregistration.json"
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
    "c010578e7307c21e305a3db499fa25204297ceb08192233dbc42023bfd5de5c8"
)
EXPECTED_BASE_D400_CONTROLLER_SHA256 = (
    "6d1f5014535fdffa1e9e63973b1037c14fb1c228a4d22f6a23f980a961ab3b17"
)
EXPECTED_BASE_D400_WORKER_SHA256 = (
    "e5b4b764012258757a9086edb840af40bcc1637586bc05934fec2674ffbd0f0a"
)
EXPECTED_BASE_D400_PREREG_SHA256 = (
    "fc689cb1afd6108a326a73f22b8117dfdefc0bb4d8caee5bcb7470c362e96c93"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _load_frozen_d400_controller() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D401 controller requires python -B before loading the frozen "
            "D400 module"
        )
    observed = _sha(BASE_D400_CONTROLLER_PATH)
    if observed != EXPECTED_BASE_D400_CONTROLLER_SHA256:
        raise RuntimeError(
            "frozen D400 controller hash drift: "
            f"{observed} != {EXPECTED_BASE_D400_CONTROLLER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d401_frozen_d400_controller",
        BASE_D400_CONTROLLER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D400 controller import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_base_paths(base: ModuleType) -> None:
    """Rebind only path/provenance globals; retain D400 science behavior."""

    path_bindings = {
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
        "CONTROLLER": CONTROLLER_PATH,
        "WORKER": WORKER_PATH,
    }
    for name, value in path_bindings.items():
        setattr(base, name, value)
    base.EXPECTED_PREREG_SHA256 = EXPECTED_PREREG_SHA256


def _merged_runtime_preregistration(base: ModuleType) -> dict[str, Any]:
    """Overlay only the D401 Git-freeze repair onto the frozen D400 contract."""

    if _sha(BASE_D400_PREREG_PATH) != EXPECTED_BASE_D400_PREREG_SHA256:
        raise RuntimeError("frozen D400 preregistration hash drift")
    if _sha(BASE_D400_WORKER_PATH) != EXPECTED_BASE_D400_WORKER_SHA256:
        raise RuntimeError("frozen D400 worker hash drift")
    if _sha(PREREG_PATH) != EXPECTED_PREREG_SHA256:
        raise RuntimeError("D401 preregistration hash drift")

    inherited = base._read_json(BASE_D400_PREREG_PATH)
    repair = base._read_json(PREREG_PATH)
    merged = copy.deepcopy(inherited)
    merged["git_baseline"] = copy.deepcopy(repair["git_baseline"])
    runtime_freeze = merged["planned_runtime_contract_for_separate_approval"][
        "runtime_freeze_manifest"
    ]
    runtime_freeze["allowed_dirty_paths"] = copy.deepcopy(
        repair["runtime_overlay_contract"]["allowed_dirty_paths"]
    )
    merged["frozen_input_hashes"].update(
        repair["runtime_overlay_contract"]["additional_frozen_repo_inputs"]
    )
    for record in repair["inherited_science_contract"].values():
        if (
            isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and isinstance(record.get("sha256"), str)
        ):
            merged["frozen_input_hashes"][record["path"]] = record["sha256"]
    merged["d401_control_repair_overlay"] = {
        "repair_preregistration_path": base._rel(PREREG_PATH),
        "repair_preregistration_sha256": _sha(PREREG_PATH),
        "new_variables": copy.deepcopy(repair["new_variables"]),
        "git_snapshot_contract": copy.deepcopy(
            repair["git_snapshot_contract"]
        ),
    }
    return merged


def _runtime_paths(base: ModuleType) -> tuple[Path, ...]:
    return (
        base.RUNTIME_MANIFEST_PATH,
        base.PHASE_PATH,
        base.INVOCATION_PATH,
        base.CLAIM_PATH,
        base.KIT_LOG_PATH,
        base.RAW_PATH,
        base.PRECLOSE_PATH,
        base.OWNER_EVIDENCE_PATH,
        base.SUPERVISOR_PATH,
        base.COMPLETION_PATH,
        base.COLLISION_ASSET_ROOT,
        base.RRD_PATH,
        base.RBL_PATH,
        base.RERUN_VALIDATION_PATH,
        base.BOARD_PATH,
        base.RERUN_SCREENSHOT_PATH,
        base.RERUN_RECEIPT_PATH,
        base.MANUAL_INSPECTION_PATH,
    )


def _capture_prewrite_git_snapshot(base: ModuleType) -> dict[str, Any]:
    """Capture all Git evidence before any forward-only runtime path is written."""

    capture_started_ns = time.monotonic_ns()
    status_rows = base._git_status_rows()
    status_paths = [row[3:] for row in status_rows if len(row) >= 4]
    dirty_files: dict[str, dict[str, Any]] = {}
    for relative in status_paths:
        path = REPO / relative
        if path.is_file():
            dirty_files[relative] = {
                "sha256": base._sha(path),
                "bytes": path.stat().st_size,
            }
    existing_runtime_paths = [
        base._rel(path) for path in _runtime_paths(base) if path.exists()
    ]
    head = base._git_value("rev-parse", "HEAD")
    origin_master = base._git_value("rev-parse", "origin/master")
    capture_completed_ns = time.monotonic_ns()
    return {
        "artifact": "D401_PREWRITE_GIT_SNAPSHOT_IN_MEMORY_V1",
        "capture_started_monotonic_ns": capture_started_ns,
        "capture_completed_monotonic_ns": capture_completed_ns,
        "head": head,
        "origin_master": origin_master,
        "status_command": "git status --short --untracked-files=all",
        "status_rows": status_rows,
        "status_paths": status_paths,
        "dirty_regular_files": dirty_files,
        "existing_future_runtime_paths": existing_runtime_paths,
    }


def _runtime_freeze_manifest_from_snapshot(
    base: ModuleType,
    approval: dict[str, Any],
    prereg: dict[str, Any],
    launch_authority: dict[str, Any],
    worker_command: list[str],
    snapshot: dict[str, Any],
    first_phase_row: dict[str, Any],
) -> dict[str, Any]:
    """Write the manifest from the immutable pre-write snapshot only."""

    if base.RUNTIME_MANIFEST_PATH.exists():
        raise RuntimeError("D401 runtime manifest already exists; retry refused")

    status = list(snapshot["status_rows"])
    status_paths = list(snapshot["status_paths"])
    allowed = set(
        prereg["planned_runtime_contract_for_separate_approval"][
            "runtime_freeze_manifest"
        ]["allowed_dirty_paths"]
    )
    unexpected = sorted(path for path in status_paths if path not in allowed)

    frozen_repo: dict[str, dict[str, Any]] = {}
    for relative, expected in prereg["frozen_input_hashes"].items():
        path = REPO / relative
        observed = base._sha(path) if path.is_file() else None
        frozen_repo[relative] = {
            "expected": expected,
            "observed": observed,
            "pass": observed == expected,
        }

    installed: dict[str, dict[str, Any]] = {}
    for absolute, expected in prereg["installed_primary_source_hashes"].items():
        path = Path(absolute)
        observed = base._sha(path) if path.is_file() else None
        installed[absolute] = {
            "expected": expected,
            "observed": observed,
            "pass": observed == expected,
        }

    sidecar: dict[str, dict[str, Any]] = {}
    for relative, frozen in prereg["d334_sidecar_before"]["files"].items():
        path = REPO / relative
        observed_sha = base._sha(path) if path.is_file() else None
        observed_bytes = path.stat().st_size if path.is_file() else None
        sidecar[relative] = {
            "expected": frozen,
            "observed": {
                "sha256": observed_sha,
                "bytes": observed_bytes,
            },
            "pass": (
                observed_sha == frozen["sha256"]
                and observed_bytes == frozen["bytes"]
            ),
        }

    expected_git = prereg["git_baseline"]
    phase_relative = base._rel(base.PHASE_PATH)
    dirty_observations_complete = all(
        relative in snapshot["dirty_regular_files"]
        for relative in status_paths
    )
    dirty_regular_files_recheck: dict[str, dict[str, Any]] = {}
    for relative, captured in snapshot["dirty_regular_files"].items():
        path = REPO / relative
        observed = (
            {
                "sha256": base._sha(path),
                "bytes": path.stat().st_size,
            }
            if path.is_file()
            else {"sha256": None, "bytes": None}
        )
        dirty_regular_files_recheck[relative] = {
            "captured": captured,
            "observed_before_manifest": observed,
            "pass": observed == captured,
        }
    checks = {
        "head_exact": snapshot["head"] == expected_git["head"],
        "origin_master_exact": (
            snapshot["origin_master"] == expected_git["origin_master"]
        ),
        "head_equals_origin_master": (
            snapshot["head"] == snapshot["origin_master"]
        ),
        "no_unexpected_dirty_paths": not unexpected,
        "all_frozen_repo_hashes_exact": all(
            row["pass"] for row in frozen_repo.values()
        ),
        "all_installed_primary_hashes_exact": all(
            row["pass"] for row in installed.values()
        ),
        "d334_sidecar_untouched": all(
            row["pass"] for row in sidecar.values()
        ),
        "approval_tuple_gate_pass": approval["pass"] is True,
        "snapshot_captured_before_first_phase_write": (
            type(snapshot["capture_started_monotonic_ns"]) is int
            and type(snapshot["capture_completed_monotonic_ns"]) is int
            and type(first_phase_row.get("monotonic_ns")) is int
            and snapshot["capture_started_monotonic_ns"]
            <= snapshot["capture_completed_monotonic_ns"]
            and snapshot["capture_completed_monotonic_ns"]
            < first_phase_row["monotonic_ns"]
        ),
        "snapshot_runtime_paths_were_fresh": (
            snapshot["existing_future_runtime_paths"] == []
        ),
        "phase_path_absent_from_captured_status": (
            phase_relative not in status_paths
        ),
        "captured_dirty_regular_file_observations_complete": (
            dirty_observations_complete
            and set(snapshot["dirty_regular_files"]) == set(status_paths)
        ),
        "captured_dirty_regular_files_still_exact_before_manifest": all(
            row["pass"] for row in dirty_regular_files_recheck.values()
        ),
        "first_phase_is_supervisor_preflight_start": (
            first_phase_row.get("ordinal") == 1
            and first_phase_row.get("phase") == "supervisor_preflight_start"
            and first_phase_row.get("owner") == "controller"
            and first_phase_row.get(
                "git_snapshot_captured_before_this_write"
            )
            is True
        ),
    }
    manifest = {
        "artifact": "D401_RUNTIME_FREEZE_MANIFEST_V1",
        "control_repair": "d400_runtime_freeze_snapshot_order_repair",
        "created_before_single_worker_spawn": True,
        "approved_tuple": approval,
        "git_snapshot_provenance": {
            "captured_before_first_runtime_write": True,
            "captured_snapshot_canonical_sha256": base._json_sha256(snapshot),
            "capture_started_monotonic_ns": snapshot[
                "capture_started_monotonic_ns"
            ],
            "capture_completed_monotonic_ns": snapshot[
                "capture_completed_monotonic_ns"
            ],
            "first_runtime_write": phase_relative,
            "first_runtime_write_monotonic_ns": first_phase_row[
                "monotonic_ns"
            ],
            "manifest_used_live_git_requery_after_phase": False,
            "existing_future_runtime_paths_at_capture": snapshot[
                "existing_future_runtime_paths"
            ],
        },
        "git": {
            "head": snapshot["head"],
            "origin_master": snapshot["origin_master"],
            "status_command": snapshot["status_command"],
            "status_rows": status,
            "status_paths": status_paths,
            "allowed_dirty_paths": sorted(allowed),
            "unexpected_dirty_paths": unexpected,
            "dirty_regular_files_at_capture": snapshot[
                "dirty_regular_files"
            ],
            "dirty_regular_files_recheck_before_manifest": (
                dirty_regular_files_recheck
            ),
        },
        "frozen_repo_inputs": frozen_repo,
        "installed_primary_sources": installed,
        "d334_sidecar": sidecar,
        "output_root": base._rel(base.OUT_DIR),
        "worker_command": worker_command,
        "worker_launch_authority": {
            "controller_pid": launch_authority["controller_pid"],
            "approved_tuple_sha256": launch_authority[
                "approved_tuple_sha256"
            ],
            "one_shot_nonce": launch_authority["one_shot_nonce"],
            "invocation_path": base._rel(base.INVOCATION_PATH),
            "invocation_sha256_transport": (
                "D400_INVOCATION_SHA256 environment variable, populated "
                "only after the invocation file is exclusively written"
            ),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    base._write_json_x(base.RUNTIME_MANIFEST_PATH, manifest)
    return manifest


def run_runtime(approved_tuple_sha256: str) -> int:
    """Run the inherited D400 controller only after the repaired snapshot gate."""

    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D401 controller must be launched with python -B; runtime refused "
            "before frozen-module import or any forward-only write"
        )
    base = _load_frozen_d400_controller()
    _configure_base_paths(base)
    approval = base._validate_approval_tuple(approved_tuple_sha256)
    base._assert_fresh_runtime_paths()
    prereg = _merged_runtime_preregistration(base)
    launch_authority = {
        "controller_pid": base.os.getpid(),
        "approved_tuple_sha256": approved_tuple_sha256,
        "one_shot_nonce": secrets.token_hex(32),
    }
    command = base._worker_command(launch_authority)

    snapshot = _capture_prewrite_git_snapshot(base)
    base._phase(
        "supervisor_preflight_start",
        approved_tuple_sha256=approved_tuple_sha256,
        git_snapshot_captured_before_this_write=True,
    )
    first_phase_row = base._phase_rows()[0]

    failed_stage = "runtime_freeze_manifest"
    try:
        manifest = _runtime_freeze_manifest_from_snapshot(
            base,
            approval,
            prereg,
            launch_authority,
            command,
            snapshot,
            first_phase_row,
        )
        base._phase(
            "runtime_freeze_manifest_gate_end",
            passed=manifest["pass"],
            sha256=base._sha(base.RUNTIME_MANIFEST_PATH),
        )
        if not manifest["pass"]:
            raise RuntimeError(
                f"D401 runtime freeze manifest failed: {manifest['checks']}"
            )

        failed_stage = "package_gpu_and_existing_process_gate"
        environment = base._environment_gate(prereg)
        base._phase(
            "package_gpu_and_existing_process_gate_end",
            passed=environment["pass"],
            free_vram_mib=(
                environment["gpu"]["gpus"][0]["memory_free_mib"]
                if environment["gpu"]["gpus"]
                else None
            ),
            d400_process_conflicts=len(environment["d400_conflicts"]),
        )
        if not environment["pass"]:
            raise RuntimeError(
                "D401 package/GPU/process gate failed: "
                f"{environment['checks']}"
            )

        failed_stage = "offline_negative_controls"
        negative = base._offline_negative_controls()
        base._phase(
            "offline_negative_controls_end",
            passed=negative["pass"],
            passed_count=negative["passed"],
            total=negative["total"],
        )
        if not negative["pass"]:
            raise RuntimeError(
                "D401 inherited runtime offline negative controls failed"
            )

        failed_stage = "single_worker_supervision"
        supervisor = base._supervise_worker(
            environment,
            approval,
            manifest,
            negative,
            launch_authority,
            command,
        )
    except Exception as error:
        completion = base._write_controller_fail_stop(
            failed_stage=failed_stage,
            error=error,
        )
        print(
            json.dumps(completion, indent=2, sort_keys=True),
            flush=True,
        )
        return 1

    if not supervisor["technical_pass"]:
        completion = base._write_completion(
            supervisor=supervisor,
            board=None,
            rerun=None,
            manual=None,
        )
        print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
        return 1

    board = None
    rerun = None
    manual = None
    observability_error = None
    try:
        base._phase("technical_pass_branch_start")
        raw = base._read_json(base.RAW_PATH)
        evidence = base._read_json(base.OWNER_EVIDENCE_PATH)
        board = base._write_decision_board(evidence, raw)
        rerun = base._write_rerun(evidence, raw)
        base._phase(
            "rerun_save_only_finalize_end_if_technical_pass",
            rrd_sha256=base._sha(base.RRD_PATH),
            rbl_sha256=base._sha(base.RBL_PATH),
        )
        manual = base._wait_for_manual_inspection()
        base._phase(
            "rerun_verify_and_visual_inspection_end_if_technical_pass",
            validation_pass=rerun["pass_before_manual"],
            manual_pass=manual["pass"],
            screenshot_sha256=(
                base._sha(base.RERUN_SCREENSHOT_PATH)
                if base.RERUN_SCREENSHOT_PATH.is_file()
                else None
            ),
        )
    except Exception as error:
        observability_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    completion = base._write_completion(
        supervisor=supervisor,
        board=board,
        rerun=rerun,
        manual=manual,
        observability_error=observability_error,
    )
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return 0 if completion["pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved-tuple-sha256",
        required=True,
        help=(
            "Exact SHA-256 of d401_proposed_runtime_hash_tuple.json explicitly "
            "cited by the user in a later, separate runtime approval."
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
