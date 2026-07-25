#!/usr/bin/env python3
"""Forward-only D384 attempt5 project-root import bootstrap repair.

Attempt4 proved the JSON-native scalar repair and board layout, then stopped
before RRD projection because a direct script invocation sets ``sys.path[0]``
to ``sim_scripts`` rather than the repository root.  This wrapper freezes that
failure, inherits the attempt4 serializer, and inserts the exact repository
root once before the unchanged presentation worker imports ``roarm_rl``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ATTEMPT4_SCRIPT = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d384_attempt4_json_native_layout_serialization_repair.py"
)
SPEC = importlib.util.spec_from_file_location(
    "d384_attempt4_frozen",
    ATTEMPT4_SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load frozen attempt4: {ATTEMPT4_SCRIPT}")
frozen4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen4)
base = frozen4.base

ATTEMPT = "attempt5_project_root_import_bootstrap_repair"
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d384"
    / ATTEMPT
)
SCRIPT_PATH = Path(__file__).resolve()
ATTEMPT4_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d384"
    / "attempt4_json_native_layout_serialization_repair"
)
ATTEMPT4_STDERR = ATTEMPT4_DIR / "d384_attempt4_worker_stderr.log"
ATTEMPT4_SUPERVISOR = (
    ATTEMPT4_DIR / "d384_attempt4_worker_supervisor.json"
)
ATTEMPT4_LAYOUT = (
    ATTEMPT4_DIR / "d384_attempt4_board_layout_validation.json"
)
PREFLIGHT_PATH = OUT_DIR / "d384_attempt5_import_bootstrap_preflight.json"
BOOTSTRAP_RECEIPT = OUT_DIR / "d384_attempt5_import_bootstrap_receipt.json"

EXPECTED_ATTEMPT4_HASHES = {
    "script": (
        "06aa92d18fe4ba7362cd048f92be19b5033ca70f7c71d76c0784dd8eafd1966a"
    ),
    "stderr": (
        "10109caf2a129688472f13453b917d387f537ed4263452fbad48848fe065a038"
    ),
    "supervisor": (
        "007e8bd0445eeeef892f4c178d976341d92bf37523129fd78b85217cae5c6817"
    ),
    "layout": (
        "3324793be322232740778f5b9ed6ed84bd92298a55695b9b213bbf8c5219ff7a"
    ),
}

base.ATTEMPT = ATTEMPT
base.OUT_DIR = OUT_DIR
base.SCRIPT_PATH = SCRIPT_PATH

_PATH_NAMES = {
    "PREREG_PATH": "d384_attempt5_preregistration.json",
    "PHASE_PATH": "d384_attempt5_phase_markers.jsonl",
    "INVOCATION_PATH": "d384_attempt5_worker_invocation.json",
    "WORKER_STDOUT": "d384_attempt5_worker_stdout.log",
    "WORKER_STDERR": "d384_attempt5_worker_stderr.log",
    "WORKER_CLAIM": "d384_attempt5_worker_claim.json",
    "SUPERVISOR_PATH": "d384_attempt5_worker_supervisor.json",
    "BOARD_PATH": "d384_attempt5_presentation_board_1920x1080.png",
    "BOARD_LAYOUT": "d384_attempt5_board_layout_validation.json",
    "RECORDING_ONLY": "d384_attempt5_recording_only.rrd",
    "OVERLAY_RRD": "d384_attempt5_summary_overlay.rrd",
    "RBL_PATH": "d384_attempt5_presentation.rbl",
    "PRESENTATION_RRD": "d384_attempt5_presentation.rrd",
    "RECORDING_EQUIVALENCE": "d384_attempt5_recording_equivalence.json",
    "RERUN_VALIDATION": "d384_attempt5_rerun_validation.json",
    "VIEWER_RECEIPT": "d384_attempt5_viewer_receipt.json",
    "RERUN_SCREENSHOT": "d384_attempt5_rerun_inspection.png",
    "MANUAL_TEMPLATE": "d384_attempt5_manual_visual_inspection_template.json",
    "MANUAL_INSPECTION": "d384_attempt5_manual_visual_inspection.json",
    "COMPLETION_PATH": "d384_attempt5_completion_summary.json",
}
for name, filename in _PATH_NAMES.items():
    setattr(base, name, OUT_DIR / filename)


def _source_hashes() -> dict[str, str]:
    return {
        "attempt5_wrapper": base._sha(SCRIPT_PATH),
        "frozen_attempt4_wrapper": base._sha(ATTEMPT4_SCRIPT),
        "frozen_attempt3_module": base._sha(frozen4.BASE_SCRIPT),
    }


base._source_hashes = _source_hashes


def _attempt5_preflight() -> dict[str, Any]:
    supervisor = base._read_json(ATTEMPT4_SUPERVISOR)
    layout = base._read_json(ATTEMPT4_LAYOUT)
    stderr = ATTEMPT4_STDERR.read_text(encoding="utf-8")
    observed_hashes = {
        "script": base._sha(ATTEMPT4_SCRIPT),
        "stderr": base._sha(ATTEMPT4_STDERR),
        "supervisor": base._sha(ATTEMPT4_SUPERVISOR),
        "layout": base._sha(ATTEMPT4_LAYOUT),
    }
    checks = {
        "attempt4_hashes_exact": (
            observed_hashes == EXPECTED_ATTEMPT4_HASHES
        ),
        "attempt4_worker_one_retry_zero": (
            supervisor.get("actual_offline_worker_invocations") == 1
            and supervisor.get("automatic_retries") == 0
        ),
        "attempt4_return_one": supervisor.get("returncode") == 1,
        "attempt4_no_timeout_signal_or_residue": (
            supervisor.get("timed_out") is False
            and supervisor.get("sigterm_sent") is False
            and supervisor.get("sigkill_sent") is False
            and supervisor.get("process_group_alive_after_wait") is False
        ),
        "attempt4_json_and_layout_repair_passed": (
            layout.get("pass") is True
            and layout["board"]["exact_1920x1080"] is True
        ),
        "attempt4_stopped_before_rerun_projection": (
            supervisor["required_artifacts"]["recording_only"] is False
            and supervisor["required_artifacts"]["rerun_screenshot"] is False
        ),
        "exact_import_exception_classified": (
            "ModuleNotFoundError: No module named 'roarm_rl'" in stderr
        ),
        "repair_is_project_root_bootstrap_only": True,
        "attempt4_not_rerun_or_overwritten": True,
    }
    return {
        "artifact": "D384_ATTEMPT5_PROJECT_ROOT_IMPORT_PREFLIGHT_V1",
        "attempt4_failure": {
            "classification": (
                "DIRECT_SCRIPT_REPO_ROOT_IMPORT_MISSING_BEFORE_RRD_PROJECTION"
            ),
            "observed_hashes": observed_hashes,
            "attempt4_rerun_or_overwrite": False,
        },
        "repair": {
            "repo_root": str(REPO),
            "operation": (
                "insert exact repository root at sys.path[0] once before "
                "calling the frozen presentation worker"
            ),
            "new_scientific_or_design_variables": 0,
            "presentation_variables_inherited": list(base.NEW_VARIABLES),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _bootstrap_project_root() -> dict[str, Any]:
    repo = str(REPO)
    count_before = sys.path.count(repo)
    if count_before > 1:
        raise RuntimeError(
            f"repository root duplicated before bootstrap: {count_before}"
        )
    if count_before == 0:
        sys.path.insert(0, repo)
        inserted = True
    else:
        inserted = False
        index = sys.path.index(repo)
        if index != 0:
            sys.path.pop(index)
            sys.path.insert(0, repo)
    count_after = sys.path.count(repo)
    receipt = {
        "artifact": "D384_ATTEMPT5_PROJECT_ROOT_IMPORT_RECEIPT_V1",
        "repo_root": repo,
        "count_before": count_before,
        "inserted": inserted,
        "count_after": count_after,
        "index_after": sys.path.index(repo),
        "ambient_pythonpath": None,
        "checks": {
            "count_before_zero_or_one": count_before in {0, 1},
            "count_after_exactly_one": count_after == 1,
            "repo_root_at_sys_path_zero": sys.path[0] == repo,
        },
    }
    receipt["pass"] = all(receipt["checks"].values())
    return receipt


def prepare() -> int:
    preflight = _attempt5_preflight()
    if preflight["pass"] is not True:
        raise RuntimeError(f"attempt5 preflight failed: {preflight}")
    result = base.prepare()
    base._write_json_x(PREFLIGHT_PATH, preflight)
    base._phase(
        "attempt5_project_root_import_preflight_frozen",
        preflight_sha256=base._sha(PREFLIGHT_PATH),
        passed=True,
    )
    return result


def worker() -> int:
    expected = _attempt5_preflight()
    frozen = base._read_json(PREFLIGHT_PATH)
    if frozen != expected or frozen.get("pass") is not True:
        raise RuntimeError("attempt5 import preflight drifted")
    receipt = _bootstrap_project_root()
    if receipt["pass"] is not True:
        raise RuntimeError(f"attempt5 bootstrap failed: {receipt}")
    base._write_json_x(BOOTSTRAP_RECEIPT, receipt)
    base._phase(
        "attempt5_project_root_inserted_once",
        receipt_sha256=base._sha(BOOTSTRAP_RECEIPT),
        count_after=receipt["count_after"],
        index_after=receipt["index_after"],
    )
    return base.worker()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "run", "worker", "finalize"),
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        return prepare()
    if args.stage == "run":
        return base.run_supervisor()
    if args.stage == "worker":
        return worker()
    return base.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
