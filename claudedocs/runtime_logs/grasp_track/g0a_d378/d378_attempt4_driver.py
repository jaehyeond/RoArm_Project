#!/usr/bin/env python3
"""D378 attempt4 forward-only driver for the failed attempt3 preregistration.

Attempt3 stopped before board generation because START_HERE line-wrapped the
registered output path. This driver keeps the attempt3 implementation frozen,
redirects its writes to a new attempt4 folder, records the failed attempt3
inventory, and changes only the registration string plus the already
preregistered vertical board layout repair.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
FROZEN_ATTEMPT3_HARNESS = (
    REPO
    / "sim_scripts/cyl34_top_view_d378_attempt3_ascii_board_layout_repair.py"
)
FAILED_ATTEMPT3 = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d378/"
    "attempt3_ascii_board_layout_repair"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d378/"
    "attempt4_start_here_registration_string_repair"
)
DRIVER = Path(__file__).resolve()
EXPECTED_FROZEN_ATTEMPT3_HARNESS_SHA = (
    "0cc64d148c36a9a2b13fdf0793bc7f2a6f1b24f2658120cefa8a667a00dc8b88"
)
EXPECTED_FAILED_ATTEMPT3_HASHES = {
    "d378_attempt3_phase_markers.jsonl": (
        "253f9f4820407a9aae55c67efc68ced5018b132d3bacec5ce5d932e6a474c686"
    ),
    "d378_attempt3_preregistration.json": (
        "73a5effc87acd900bb4bbae932b7df8327c2e4c63f3b6b593abcacf013a63f9b"
    ),
    "d378_attempt3_runtime_exception.json": (
        "e9acbc31e869c43fd2b6407d344bf49acd2d7d577b165deb94c6fe9689ca010c"
    ),
}


def _load_frozen_attempt3() -> Any:
    spec = importlib.util.spec_from_file_location(
        "d378_frozen_attempt3_board_repair",
        FROZEN_ATTEMPT3_HARNESS,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen D378 attempt3 harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_frozen_attempt3()
FAILED_ATTEMPT3_INVENTORY = mod._inventory(FAILED_ATTEMPT3)

mod.OUT_DIR = OUT_DIR
mod.PREREG_PATH = OUT_DIR / "d378_attempt4_preregistration.json"
mod.PHASE_PATH = OUT_DIR / "d378_attempt4_phase_markers.jsonl"
mod.INVOCATION_PATH = OUT_DIR / "d378_attempt4_invocation.json"
mod.REPAIR_EVIDENCE_PATH = OUT_DIR / "d378_attempt4_visual_repair_evidence.json"
mod.BOARD_PATH = (
    OUT_DIR / "d378_corrected_workload_authority_repaired_1920x1080.png"
)
mod.LAYOUT_PATH = OUT_DIR / "d378_attempt4_layout_validation.json"
mod.MANUAL_TEMPLATE_PATH = (
    OUT_DIR / "d378_attempt4_manual_visual_inspection_template.json"
)
mod.MANUAL_PATH = OUT_DIR / "d378_attempt4_manual_visual_inspection.json"
mod.COMPLETION_PATH = OUT_DIR / "d378_final_completion_summary.json"
mod.EXCEPTION_PATH = OUT_DIR / "d378_attempt4_runtime_exception.json"
mod.REPAIR_CHANGE = (
    "start_here_exact_path_registration_plus_inherited_vertical_layout_only"
)
mod.VERDICT_FAIL = "D378_ATTEMPT4_ASCII_BOARD_LAYOUT_REPAIR_FAIL_STOP"


def _source_hashes() -> dict[str, str]:
    return {
        "attempt4_driver": mod._sha(DRIVER),
        "frozen_attempt3_board_harness": mod._sha(FROZEN_ATTEMPT3_HARNESS),
        "frozen_attempt2_authority_harness": mod._sha(
            mod.ORIGINAL_D378_HARNESS
        ),
    }


mod._source_hashes = _source_hashes
_original_write_json_x = mod._write_json_x
_original_read_json = mod._read_json


def _translate_attempt_label(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("ATTEMPT3", "ATTEMPT4")
    if isinstance(value, dict):
        return {
            key: _translate_attempt_label(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_translate_attempt_label(child) for child in value]
    return value


def _attempt3_still_immutable() -> bool:
    current_hashes = {
        path.name: mod._sha(path)
        for path in FAILED_ATTEMPT3.iterdir()
        if path.is_file()
    }
    return bool(
        mod._sha(FROZEN_ATTEMPT3_HARNESS)
        == EXPECTED_FROZEN_ATTEMPT3_HARNESS_SHA
        and current_hashes == EXPECTED_FAILED_ATTEMPT3_HASHES
        and mod._inventory(FAILED_ATTEMPT3) == FAILED_ATTEMPT3_INVENTORY
    )


def _patched_write_json_x(path: Path, payload: dict[str, Any]) -> None:
    translated = _translate_attempt_label(payload)
    payload.clear()
    payload.update(translated)

    if path == mod.PREREG_PATH:
        payload["checks"]["start_here_attempt4_registered"] = payload[
            "checks"
        ].pop("start_here_attempt3_registered")
        payload["registration_repair"] = {
            "failed_attempt3_reason": (
                "START_HERE split the exact output path across two code spans"
            ),
            "failed_attempt3_actual_board_regenerations": 0,
            "failed_attempt3_actual_authority_audits": 0,
            "failed_attempt3_inventory_before": FAILED_ATTEMPT3_INVENTORY,
            "changed_contract_field": (
                "one contiguous exact START_HERE output-path registration"
            ),
            "finalize_compatibility_bridge": (
                "read the on-disk ATTEMPT4 manual label once as the frozen "
                "ATTEMPT3 literal, then attest the actual ATTEMPT4 label in "
                "the completion record"
            ),
        }
        payload["checks"]["failed_attempt3_immutable_at_preregistration"] = (
            _attempt3_still_immutable()
        )
        payload["checks"]["failed_attempt3_exact_three_file_failure_artifact"] = (
            len(FAILED_ATTEMPT3_INVENTORY["files"]) == 3
            and _attempt3_still_immutable()
        )
        payload["pass"] = all(payload["checks"].values())
    elif path == mod.INVOCATION_PATH:
        payload["failed_attempt3_inventory_sha256"] = (
            FAILED_ATTEMPT3_INVENTORY["inventory_sha256"]
        )
    elif path == mod.REPAIR_EVIDENCE_PATH:
        payload["failed_attempt3_preregistration_preserved"] = {
            "inventory": FAILED_ATTEMPT3_INVENTORY,
            "actual_board_regenerations": 0,
            "actual_authority_audits": 0,
        }
        payload["checks"]["failed_attempt3_immutable"] = (
            _attempt3_still_immutable()
        )
        payload["pass"] = all(payload["checks"].values())
        payload["verdict"] = (
            mod.VERDICT_PASS if payload["pass"] else mod.VERDICT_FAIL
        )
    elif path == mod.COMPLETION_PATH:
        payload["attempt4_repair"] = payload.pop("attempt3_repair")
        payload["checks"][
            "manual_artifact_attempt4_exact_via_compatibility_bridge"
        ] = payload["checks"].pop("manual_artifact_exact") and (
            _original_read_json(mod.MANUAL_PATH).get("artifact")
            == "D378_ATTEMPT4_MANUAL_VISUAL_INSPECTION_V1"
        )
        payload["finalize_compatibility_bridge"] = {
            "reason": (
                "the frozen attempt3 finalize literal expects the attempt3 "
                "manual artifact label"
            ),
            "in_memory_legacy_label_reads": 1,
            "on_disk_manual_artifact": (
                "D378_ATTEMPT4_MANUAL_VISUAL_INSPECTION_V1"
            ),
        }
        payload["attempt3_preregistration_failure_preserved"] = {
            "inventory": FAILED_ATTEMPT3_INVENTORY,
            "actual_board_regenerations": 0,
            "actual_authority_audits": 0,
        }
        payload["checks"]["failed_attempt3_immutable"] = (
            _attempt3_still_immutable()
        )
        payload["pass"] = all(payload["checks"].values())
        payload["verdict"] = (
            mod.VERDICT_PASS if payload["pass"] else mod.VERDICT_FAIL
        )

    _original_write_json_x(path, payload)


mod._write_json_x = _patched_write_json_x


def _patched_read_json(path: Path) -> dict[str, Any]:
    payload = _original_read_json(path)
    if (
        path == mod.MANUAL_PATH
        and payload.get("artifact")
        == "D378_ATTEMPT4_MANUAL_VISUAL_INSPECTION_V1"
    ):
        payload["artifact"] = "D378_ATTEMPT3_MANUAL_VISUAL_INSPECTION_V1"
    return payload


mod._read_json = _patched_read_json


def _check_failed_attempt3_against_preregistration() -> None:
    prereg = mod._read_json(mod.PREREG_PATH)
    expected = prereg["registration_repair"][
        "failed_attempt3_inventory_before"
    ]
    if mod._inventory(FAILED_ATTEMPT3) != expected:
        raise RuntimeError("frozen attempt3 inventory drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "run", "finalize"),
    )
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            mod._prepare()
        elif args.stage == "run":
            _check_failed_attempt3_against_preregistration()
            mod._run()
        else:
            _check_failed_attempt3_against_preregistration()
            mod._finalize()
        return 0
    except Exception as exc:
        payload = {
            "artifact": "D378_ATTEMPT4_RUNTIME_EXCEPTION_V1",
            "stage": args.stage,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "verdict": mod.VERDICT_FAIL,
        }
        try:
            if OUT_DIR.exists() and not mod.EXCEPTION_PATH.exists():
                mod._write_json_x(mod.EXCEPTION_PATH, payload)
            if OUT_DIR.exists():
                mod._phase(
                    "exception",
                    stage=args.stage,
                    exception_type=type(exc).__name__,
                )
        except Exception:
            pass
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
