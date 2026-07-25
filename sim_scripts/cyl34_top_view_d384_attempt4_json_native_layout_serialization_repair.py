#!/usr/bin/env python3
"""Forward-only D384 attempt4 harness-only JSON scalar repair.

Attempt3 stopped after creating its fixed board because a NumPy ``bool_`` from
the layout audit was not converted to a JSON-native ``bool``.  This wrapper
freezes attempt3, verifies that exact failure, normalizes NumPy scalar values
only at JSON serialization, and otherwise reuses the frozen attempt3
presentation implementation unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BASE_SCRIPT = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d384_attempt3_presentation_contract_repair.py"
)
SPEC = importlib.util.spec_from_file_location(
    "d384_attempt3_frozen",
    BASE_SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load frozen attempt3: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

ATTEMPT = "attempt4_json_native_layout_serialization_repair"
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d384"
    / ATTEMPT
)
SCRIPT_PATH = Path(__file__).resolve()
ATTEMPT3_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d384"
    / "attempt3_presentation_contract_repair"
)
ATTEMPT3_STDERR = ATTEMPT3_DIR / "d384_attempt3_worker_stderr.log"
ATTEMPT3_SUPERVISOR = (
    ATTEMPT3_DIR / "d384_attempt3_worker_supervisor.json"
)
PREFLIGHT_PATH = (
    OUT_DIR / "d384_attempt4_json_native_scalar_preflight.json"
)

EXPECTED_ATTEMPT3_HASHES = {
    "script": (
        "c0354cbb8e537a787d5b0657c9ac843070613470f7331e06314e146b35d72278"
    ),
    "stderr": (
        "dd8de6dc111c809815b0c9b9f7c9f29e6b2397399bea10ac7ecbcd6dc0d2ba90"
    ),
    "supervisor": (
        "cba82f6e03ec49825a6ed31a8e05f52454fb4efbdf6e810b02dfc30dbb08bd43"
    ),
}

base.ATTEMPT = ATTEMPT
base.OUT_DIR = OUT_DIR
base.SCRIPT_PATH = SCRIPT_PATH

_PATH_NAMES = {
    "PREREG_PATH": "d384_attempt4_preregistration.json",
    "PHASE_PATH": "d384_attempt4_phase_markers.jsonl",
    "INVOCATION_PATH": "d384_attempt4_worker_invocation.json",
    "WORKER_STDOUT": "d384_attempt4_worker_stdout.log",
    "WORKER_STDERR": "d384_attempt4_worker_stderr.log",
    "WORKER_CLAIM": "d384_attempt4_worker_claim.json",
    "SUPERVISOR_PATH": "d384_attempt4_worker_supervisor.json",
    "BOARD_PATH": "d384_attempt4_presentation_board_1920x1080.png",
    "BOARD_LAYOUT": "d384_attempt4_board_layout_validation.json",
    "RECORDING_ONLY": "d384_attempt4_recording_only.rrd",
    "OVERLAY_RRD": "d384_attempt4_summary_overlay.rrd",
    "RBL_PATH": "d384_attempt4_presentation.rbl",
    "PRESENTATION_RRD": "d384_attempt4_presentation.rrd",
    "RECORDING_EQUIVALENCE": "d384_attempt4_recording_equivalence.json",
    "RERUN_VALIDATION": "d384_attempt4_rerun_validation.json",
    "VIEWER_RECEIPT": "d384_attempt4_viewer_receipt.json",
    "RERUN_SCREENSHOT": "d384_attempt4_rerun_inspection.png",
    "MANUAL_TEMPLATE": "d384_attempt4_manual_visual_inspection_template.json",
    "MANUAL_INSPECTION": "d384_attempt4_manual_visual_inspection.json",
    "COMPLETION_PATH": "d384_attempt4_completion_summary.json",
}
for name, filename in _PATH_NAMES.items():
    setattr(base, name, OUT_DIR / filename)


def _source_hashes() -> dict[str, str]:
    return {
        "attempt4_wrapper": base._sha(SCRIPT_PATH),
        "frozen_attempt3_module": base._sha(BASE_SCRIPT),
    }


base._source_hashes = _source_hashes


def _json_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_native(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def _write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            _json_native(value),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


base._write_json_x = _write_json_x


def _attempt4_preflight() -> dict[str, Any]:
    supervisor = base._read_json(ATTEMPT3_SUPERVISOR)
    stderr = ATTEMPT3_STDERR.read_text(encoding="utf-8")
    observed_hashes = {
        "script": base._sha(BASE_SCRIPT),
        "stderr": base._sha(ATTEMPT3_STDERR),
        "supervisor": base._sha(ATTEMPT3_SUPERVISOR),
    }
    checks = {
        "attempt3_hashes_exact": (
            observed_hashes == EXPECTED_ATTEMPT3_HASHES
        ),
        "attempt3_worker_one_retry_zero": (
            supervisor.get("actual_offline_worker_invocations") == 1
            and supervisor.get("automatic_retries") == 0
        ),
        "attempt3_return_one": supervisor.get("returncode") == 1,
        "attempt3_no_timeout_signal_or_residue": (
            supervisor.get("timed_out") is False
            and supervisor.get("sigterm_sent") is False
            and supervisor.get("sigkill_sent") is False
            and supervisor.get("process_group_alive_after_wait") is False
        ),
        "attempt3_failed_before_rerun_projection": (
            supervisor["required_artifacts"]["board"] is True
            and supervisor["required_artifacts"]["recording_only"] is False
            and supervisor["required_artifacts"]["rerun_screenshot"] is False
        ),
        "exact_exception_classified": (
            "TypeError: Object of type bool_ is not JSON serializable"
            in stderr
        ),
        "repair_is_json_boundary_only": True,
        "attempt3_not_rerun_or_overwritten": True,
    }
    return {
        "artifact": "D384_ATTEMPT4_JSON_NATIVE_SCALAR_PREFLIGHT_V1",
        "attempt3_failure": {
            "classification": (
                "HARNESS_JSON_NUMPY_BOOL_SERIALIZATION_ERROR_AFTER_BOARD"
            ),
            "observed_hashes": observed_hashes,
            "attempt3_rerun_or_overwrite": False,
        },
        "repair": {
            "scope": (
                "recursive NumPy scalar to JSON-native scalar conversion at "
                "the write boundary only"
            ),
            "new_scientific_or_design_variables": 0,
            "presentation_variables_inherited": list(base.NEW_VARIABLES),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def prepare() -> int:
    preflight = _attempt4_preflight()
    if preflight["pass"] is not True:
        raise RuntimeError(f"attempt4 preflight failed: {preflight}")
    result = base.prepare()
    base._write_json_x(PREFLIGHT_PATH, preflight)
    base._phase(
        "attempt4_json_native_scalar_preflight_frozen",
        preflight_sha256=base._sha(PREFLIGHT_PATH),
        passed=True,
    )
    return result


def worker() -> int:
    expected = _attempt4_preflight()
    frozen = base._read_json(PREFLIGHT_PATH)
    if frozen != expected or frozen.get("pass") is not True:
        raise RuntimeError("attempt4 scalar preflight drifted")
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
