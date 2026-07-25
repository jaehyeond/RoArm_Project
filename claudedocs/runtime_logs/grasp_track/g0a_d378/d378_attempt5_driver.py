#!/usr/bin/env python3
"""D378 attempt5: measured one-text inset repair, forward-only.

Attempt4 rendered the correct 1920x1080 board but missed its own 6.48-pixel
containment margin by 0.08 pixel for exactly one text block. This driver keeps
attempt4 immutable and shifts only that block upward by 0.002 normalized Y
(2.16 pixels). It does not recompute authority or invoke Rerun/Isaac/PhysX.
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
FROZEN_ATTEMPT4_DRIVER = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d378/d378_attempt4_driver.py"
)
FROZEN_ATTEMPT3_HARNESS = (
    REPO
    / "sim_scripts/cyl34_top_view_d378_attempt3_ascii_board_layout_repair.py"
)
FAILED_ATTEMPT4 = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d378/"
    "attempt4_start_here_registration_string_repair"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d378/"
    "attempt5_measured_outcome_inset_repair"
)
DRIVER = Path(__file__).resolve()

EXPECTED_FROZEN_ATTEMPT4_DRIVER_SHA = (
    "11f101e524d05e791e47d656eeb0298d4efc20fd8ce9b0b57bf3e9b075d3c0bf"
)
EXPECTED_FROZEN_ATTEMPT3_HARNESS_SHA = (
    "0cc64d148c36a9a2b13fdf0793bc7f2a6f1b24f2658120cefa8a667a00dc8b88"
)
EXPECTED_FAILED_ATTEMPT4_HASHES = {
    "d378_attempt4_invocation.json": (
        "f61098673ff0f5d8508c80e8d51ec4aa21181631e4fc86dd20256adc7122927a"
    ),
    "d378_attempt4_layout_validation.json": (
        "3f78b5fc2c6cc9ee226351cf0c5de4b0f017d05e5aa3e12b0d63ebe2c65b0c02"
    ),
    "d378_attempt4_manual_visual_inspection_template.json": (
        "8965781908f8aca37b0b963c1b93f5bd5ea1bf3671928bd5ecfd475f6cc07871"
    ),
    "d378_attempt4_phase_markers.jsonl": (
        "d2211076ff542592c9e02599e68c0a0384d53b467d88b98258fc337e75cbd532"
    ),
    "d378_attempt4_preregistration.json": (
        "0d7f2a07c458aa39b9a877d0523c6783a686e3398509168bec3331574ff23471"
    ),
    "d378_attempt4_runtime_exception.json": (
        "21e70ca9498adc69c21a017684d1cada9f05490127f2b2c7204ddaaaa85fd48d"
    ),
    "d378_attempt4_visual_repair_evidence.json": (
        "d37439d70a4dfb82728e7835dcf832096a7d5d55eaf58fc6d0261bf1aad0fe62"
    ),
    "d378_corrected_workload_authority_repaired_1920x1080.png": (
        "2aa6ae1f193d0643e14544d664ca0dcf412b3f59f5b36e33086b344e3c283931"
    ),
}
OUTCOME_Y_SHIFT_NORMALIZED = 0.002
REGISTERED_MARGIN_NORMALIZED = 0.006
REGISTERED_MARGIN_PX = 6.48
ATTEMPT4_OBSERVED_MARGIN_PX = 6.40
ATTEMPT4_SHORTFALL_PX = 0.08


def _load_frozen_attempt4() -> Any:
    spec = importlib.util.spec_from_file_location(
        "d378_frozen_attempt4_driver",
        FROZEN_ATTEMPT4_DRIVER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen D378 attempt4 driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a4 = _load_frozen_attempt4()
mod = a4.mod
FAILED_ATTEMPT4_INVENTORY = mod._inventory(FAILED_ATTEMPT4)

mod.OUT_DIR = OUT_DIR
mod.PREREG_PATH = OUT_DIR / "d378_attempt5_preregistration.json"
mod.PHASE_PATH = OUT_DIR / "d378_attempt5_phase_markers.jsonl"
mod.INVOCATION_PATH = OUT_DIR / "d378_attempt5_invocation.json"
mod.REPAIR_EVIDENCE_PATH = OUT_DIR / "d378_attempt5_visual_repair_evidence.json"
mod.BOARD_PATH = (
    OUT_DIR / "d378_corrected_workload_authority_repaired_1920x1080.png"
)
mod.LAYOUT_PATH = OUT_DIR / "d378_attempt5_layout_validation.json"
mod.MANUAL_TEMPLATE_PATH = (
    OUT_DIR / "d378_attempt5_manual_visual_inspection_template.json"
)
mod.MANUAL_PATH = OUT_DIR / "d378_attempt5_manual_visual_inspection.json"
mod.COMPLETION_PATH = OUT_DIR / "d378_final_completion_summary.json"
mod.EXCEPTION_PATH = OUT_DIR / "d378_attempt5_runtime_exception.json"
mod.REPAIR_CHANGE = "outcome_body_vertical_shift_plus_0p002_normalized_only"
mod.VERDICT_FAIL = "D378_ATTEMPT5_MEASURED_OUTCOME_INSET_REPAIR_FAIL_STOP"


def _source_hashes() -> dict[str, str]:
    return {
        "attempt5_driver": mod._sha(DRIVER),
        "frozen_attempt4_driver": mod._sha(FROZEN_ATTEMPT4_DRIVER),
        "frozen_attempt3_board_harness": mod._sha(FROZEN_ATTEMPT3_HARNESS),
        "frozen_attempt2_authority_harness": mod._sha(
            mod.ORIGINAL_D378_HARNESS
        ),
    }


def _attempt4_still_immutable() -> bool:
    current_hashes = {
        path.name: mod._sha(path)
        for path in FAILED_ATTEMPT4.iterdir()
        if path.is_file()
    }
    return bool(
        mod._sha(FROZEN_ATTEMPT4_DRIVER)
        == EXPECTED_FROZEN_ATTEMPT4_DRIVER_SHA
        and mod._sha(FROZEN_ATTEMPT3_HARNESS)
        == EXPECTED_FROZEN_ATTEMPT3_HARNESS_SHA
        and current_hashes == EXPECTED_FAILED_ATTEMPT4_HASHES
        and mod._inventory(FAILED_ATTEMPT4) == FAILED_ATTEMPT4_INVENTORY
    )


def _translate_attempt_label(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("ATTEMPT3", "ATTEMPT5")
    if isinstance(value, dict):
        return {
            key: _translate_attempt_label(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_translate_attempt_label(child) for child in value]
    return value


_raw_write_json_x = a4._original_write_json_x
_raw_read_json = a4._original_read_json


def _patched_write_json_x(path: Path, payload: dict[str, Any]) -> None:
    translated = _translate_attempt_label(payload)
    payload.clear()
    payload.update(translated)

    if path == mod.PREREG_PATH:
        payload["checks"]["start_here_attempt5_registered"] = payload[
            "checks"
        ].pop("start_here_attempt3_registered")
        payload["checks"]["failed_attempt4_exact_eight_file_artifact"] = (
            len(FAILED_ATTEMPT4_INVENTORY["files"]) == 8
            and _attempt4_still_immutable()
        )
        payload["registration_repair"] = {
            "failed_attempt4_reason": (
                "outcome_body inset 6.40px was 0.08px below the frozen "
                "6.48px programmatic margin"
            ),
            "failed_attempt4_actual_board_regenerations": 1,
            "failed_attempt4_actual_authority_audits": 0,
            "failed_attempt4_inventory_before": FAILED_ATTEMPT4_INVENTORY,
            "single_changed_value": {
                "field": "outcome_body_y",
                "delta_normalized": OUTCOME_Y_SHIFT_NORMALIZED,
                "delta_pixels_at_1080": 2.16,
                "predicted_bottom_inset_pixels": 8.56,
            },
            "unchanged_gate": {
                "margin_normalized": REGISTERED_MARGIN_NORMALIZED,
                "margin_pixels_at_1080": REGISTERED_MARGIN_PX,
            },
        }
        payload["pass"] = all(payload["checks"].values())
    elif path == mod.INVOCATION_PATH:
        payload["failed_attempt4_inventory_sha256"] = (
            FAILED_ATTEMPT4_INVENTORY["inventory_sha256"]
        )
    elif path == mod.REPAIR_EVIDENCE_PATH:
        layout = _raw_read_json(mod.LAYOUT_PATH)
        payload["failed_attempt4_layout_failure_preserved"] = {
            "inventory": FAILED_ATTEMPT4_INVENTORY,
            "actual_board_regenerations": 1,
            "actual_authority_audits": 0,
            "observed_margin_pixels": ATTEMPT4_OBSERVED_MARGIN_PX,
            "registered_margin_pixels": REGISTERED_MARGIN_PX,
            "shortfall_pixels": ATTEMPT4_SHORTFALL_PX,
        }
        payload["checks"]["failed_attempt4_immutable"] = (
            _attempt4_still_immutable()
        )
        payload["checks"]["same_box_title_body_checks_pass"] = all(
            layout["same_box_title_body_checks"].values()
        )
        payload["pass"] = all(payload["checks"].values())
        payload["verdict"] = (
            mod.VERDICT_PASS if payload["pass"] else mod.VERDICT_FAIL
        )
    elif path == mod.COMPLETION_PATH:
        payload["attempt5_repair"] = payload.pop("attempt3_repair")
        payload["checks"][
            "manual_artifact_attempt5_exact_via_compatibility_bridge"
        ] = payload["checks"].pop("manual_artifact_exact") and (
            _raw_read_json(mod.MANUAL_PATH).get("artifact")
            == "D378_ATTEMPT5_MANUAL_VISUAL_INSPECTION_V1"
        )
        payload["checks"]["failed_attempt4_immutable"] = (
            _attempt4_still_immutable()
        )
        payload["attempt4_layout_failure_preserved"] = {
            "inventory": FAILED_ATTEMPT4_INVENTORY,
            "actual_board_regenerations": 1,
            "actual_authority_audits": 0,
        }
        payload["finalize_compatibility_bridge"] = {
            "reason": (
                "the frozen attempt3 finalize literal expects the attempt3 "
                "manual artifact label"
            ),
            "in_memory_legacy_label_reads": 1,
            "on_disk_manual_artifact": (
                "D378_ATTEMPT5_MANUAL_VISUAL_INSPECTION_V1"
            ),
        }
        payload["pass"] = all(payload["checks"].values())
        payload["verdict"] = (
            mod.VERDICT_PASS if payload["pass"] else mod.VERDICT_FAIL
        )

    _raw_write_json_x(path, payload)


def _patched_read_json(path: Path) -> dict[str, Any]:
    payload = _raw_read_json(path)
    if (
        path == mod.MANUAL_PATH
        and payload.get("artifact")
        == "D378_ATTEMPT5_MANUAL_VISUAL_INSPECTION_V1"
    ):
        payload["artifact"] = "D378_ATTEMPT3_MANUAL_VISUAL_INSPECTION_V1"
    return payload


_frozen_render_repaired_board = mod._render_repaired_board


def _render_repaired_board_with_measured_shift(
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    original_text = Figure.text
    shift_counter = {"count": 0}

    def shifted_text(
        figure: Figure,
        x: float,
        y: float,
        text: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        is_outcome_body = bool(
            isinstance(text, str)
            and text.startswith("D375: no explicit Erase | timeout | return ")
            and "Conditional trigger support for this pair: PASS" in text
            and "Universal necessity and exact native root cause: NOT PROVEN"
            in text
        )
        if is_outcome_body:
            y += OUTCOME_Y_SHIFT_NORMALIZED
            shift_counter["count"] += 1
        return original_text(figure, x, y, text, *args, **kwargs)

    Figure.text = shifted_text
    try:
        board, layout = _frozen_render_repaired_board(evidence)
    finally:
        Figure.text = original_text

    same_box_checks = {
        f"{key}_title_body_separated": not mod._intersects(
            layout["text_bboxes_normalized"][f"{key}_title"],
            layout["text_bboxes_normalized"][f"{key}_body"],
        )
        for key in layout["containers_normalized"]
    }
    outcome = layout["containers_normalized"]["outcome"]
    outcome_body = layout["text_bboxes_normalized"]["outcome_body"]
    measured_margin_px = (outcome_body[1] - outcome[1]) * 1080.0
    layout["attempt5_measured_shift"] = {
        "target_text_matches": shift_counter["count"],
        "delta_normalized_y": OUTCOME_Y_SHIFT_NORMALIZED,
        "delta_pixels_at_1080": OUTCOME_Y_SHIFT_NORMALIZED * 1080.0,
        "measured_bottom_inset_pixels": measured_margin_px,
        "registered_bottom_inset_pixels": REGISTERED_MARGIN_PX,
        "registered_margin_unchanged": True,
    }
    layout["same_box_title_body_checks"] = same_box_checks
    layout["pass"] = bool(
        layout["pass"]
        and shift_counter["count"] == 1
        and measured_margin_px >= REGISTERED_MARGIN_PX
        and all(same_box_checks.values())
    )
    return board, layout


mod._source_hashes = _source_hashes
mod._write_json_x = _patched_write_json_x
mod._read_json = _patched_read_json
mod._render_repaired_board = _render_repaired_board_with_measured_shift


def _check_attempt4_against_preregistration() -> None:
    prereg = mod._read_json(mod.PREREG_PATH)
    expected = prereg["registration_repair"][
        "failed_attempt4_inventory_before"
    ]
    if mod._inventory(FAILED_ATTEMPT4) != expected:
        raise RuntimeError("frozen attempt4 inventory drift")
    if not _attempt4_still_immutable():
        raise RuntimeError("frozen attempt4 exact hashes drift")


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
            _check_attempt4_against_preregistration()
            mod._run()
        else:
            _check_attempt4_against_preregistration()
            mod._finalize()
        return 0
    except Exception as exc:
        payload = {
            "artifact": "D378_ATTEMPT5_RUNTIME_EXCEPTION_V1",
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
