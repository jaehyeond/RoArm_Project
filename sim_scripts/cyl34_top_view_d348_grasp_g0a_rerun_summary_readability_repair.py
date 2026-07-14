#!/usr/bin/env python3
"""D348 attempt4: repair only the readability of the frozen Rerun summary.

The attempt2 scientific evidence is not recomputed.  This wrapper reuses the
attempt3 exact entity/count/raster gates, changes the output directory, and
requires the preserved attempt3 manual screenshot failure as its reactive
trigger.  ``roarm_rl.viz_debug`` now writes UTF-8 metadata without JSON unicode
escapes, so the HOME-near versus offline distinction is readable in the actual
screenshot.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import sim_scripts.cyl34_top_view_d348_grasp_g0a_rerun_observability_repair as base  # noqa: E402


ATTEMPT3 = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt3_observability"
ATTEMPT3_FAIL = ATTEMPT3 / "d348_observability_manual_visual_inspection_fail.json"
ATTEMPT3_FAIL_MD = ATTEMPT3 / "d348_observability_manual_visual_inspection_fail.md"
ATTEMPT3_FAIL_SHA256 = "217ae03114d82dbd72e79ab0d50588c515e0fc375c91d8ee76321e03c08a5bb8"
ATTEMPT3_FAIL_MD_SHA256 = "e9d4e226325ffc857a8fc6fb4a04c59ebc4c314950e1ccbfe127ed04f0918d86"

ATTEMPT3_FROZEN_HASHES = {
    ATTEMPT3 / "d348_observability_preregistration.json": (
        "d679882235571455ccdce382c77b0f51cb787d725f0c5363c6db049fed728a3d"
    ),
    ATTEMPT3 / "d348_observability_parameter_freeze.json": (
        "7199955510aecdd3fa4386603c45105a8df7210b5b2a31ea37d4a0a952348504"
    ),
    ATTEMPT3 / "d348_volume_semantics_summary.rrd": (
        "3080c7ff7a7262a007534c95a16fd07e88265d68957a51313ababeef988ca436"
    ),
    ATTEMPT3 / "d348_volume_semantics_summary.rbl": (
        "a2bf3fc816532e7c704cf2ce9dc7c74b6d6ddf79a9b7177ef9a33146b820d88e"
    ),
    ATTEMPT3 / "d348_volume_semantics_summary_rerun.png": (
        "1da11da5d0b6a95e80413ebd9dd111011cd2ea894f62bce2ca9d6f07c0e8be51"
    ),
    ATTEMPT3 / "d348_observability_rerun_validation.json": (
        "8d790e7d47a38fa8d6fdd973e4ae3e487849b8f992ad68981d23dc203d8677fc"
    ),
    ATTEMPT3 / "d348_observability_automated_summary.json": (
        "d04e3891b2ff67e5b9c39b666e17fdc3040f8479f9aa9131e67b72a04ef94e39"
    ),
    ATTEMPT3_FAIL: ATTEMPT3_FAIL_SHA256,
    ATTEMPT3_FAIL_MD: ATTEMPT3_FAIL_MD_SHA256,
}

VISIBLE_SUMMARY = {
    "00_RESULT": "SUPPORTED",
    "01_TOLERANCE": "5% FROZEN",
    "02_CALLBACKS": "256/256 PASS",
    "03_PARTS": "128/128 PASS",
    "04_RAW_PAIRS": "128/128 EXACT",
    "05_CLOSED_ORIENTED": "256/256 PASS",
    "06_D347_START": "HOME-near; q5=0 CLOSED",
    "07_D347_START_KO": "HOME 근방; q5=0 닫힘",
    "08_D347_PHYSICS": "0 steps",
    "09_D348_MODE": "OFFLINE evidence replay",
    "10_D348_MODE_KO": "오프라인 재판독",
    "11_G0A": False,
    "12_PART045_TOPOLOGY_REL": "3.01548618356e-8",
    "13_PART045_QHULL_REL": "0.273316720525",
    "scientific_authority": "attempt2 immutable JSON",
    "viewer_geometry_role": "Float32 display copy",
}
VISIBLE_EVENT = "PASS | D347 HOME-near | D348 offline | G0a=false"

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt4_readability"


def _configure_attempt4() -> None:
    old_fail = base.ATTEMPT2_MANUAL_FAIL
    old_fail_md = base.ATTEMPT2_MANUAL_FAIL_MD

    base.OUT_DIR = OUT_DIR
    base.PREREG_PATH = OUT_DIR / "d348_readability_preregistration.json"
    base.PARAMETER_PATH = OUT_DIR / "d348_readability_parameter_freeze.json"
    base.RRD_PATH = OUT_DIR / "d348_volume_semantics_readable.rrd"
    base.RBL_PATH = OUT_DIR / "d348_volume_semantics_readable.rbl"
    base.SCREENSHOT_PATH = OUT_DIR / "d348_volume_semantics_readable_rerun.png"
    base.VALIDATION_PATH = OUT_DIR / "d348_readability_rerun_validation.json"
    base.AUTOMATED_PATH = OUT_DIR / "d348_readability_automated_summary.json"
    base.AUTOMATED_MD_PATH = OUT_DIR / "d348_readability_automated_report.md"
    base.MANUAL_PATH = OUT_DIR / "d348_readability_manual_visual_inspection.json"
    base.MANUAL_MD_PATH = OUT_DIR / "d348_readability_manual_visual_inspection.md"
    base.COMPLETION_PATH = OUT_DIR / "d348_completion_summary.json"
    base.COMPLETION_MD_PATH = OUT_DIR / "d348_completion_report.md"
    base.HARNESS_PATH = Path(__file__).resolve()

    # The new reactive trigger is the preserved attempt3 human-screen failure.
    base.ATTEMPT2_MANUAL_FAIL = ATTEMPT3_FAIL
    base.ATTEMPT2_MANUAL_FAIL_MD = ATTEMPT3_FAIL_MD
    hashes = dict(base.INPUT_HASHES)
    hashes.pop(old_fail)
    hashes.pop(old_fail_md)
    hashes[ATTEMPT3_FAIL] = ATTEMPT3_FAIL_SHA256
    hashes[ATTEMPT3_FAIL_MD] = ATTEMPT3_FAIL_MD_SHA256
    hashes.update(ATTEMPT3_FROZEN_HASHES)
    base.INPUT_HASHES = hashes

    original_input_guard = base._input_guard

    def attempt4_input_guard() -> dict[str, Any]:
        guard = original_input_guard()
        guard = copy.deepcopy(guard)
        guard["artifact"] = "D348_READABILITY_INPUT_GUARD_V1"
        checks = guard["checks"]
        rename = {
            "attempt2_manual_visual_fail_observed": "attempt3_manual_visual_fail_observed",
            "attempt2_manual_fail_did_not_override_science": (
                "attempt3_manual_fail_did_not_override_science"
            ),
            "attempt2_g0a_false": "attempt3_g0a_false",
        }
        for old, new in rename.items():
            checks[new] = checks.pop(old)
        guard["pass"] = all(checks.values())
        return guard

    base._input_guard = attempt4_input_guard

    original_write_json = base._write_json
    original_write_text = base._write_text
    original_log_rerun = base.log_rerun

    def attempt4_log_rerun(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        events = [dict(row) for row in kwargs.get("events", [])]
        found_summary = False
        for row in events:
            if row.get("entity_path") == "events/d348_summary":
                row["text"] = VISIBLE_EVENT
                found_summary = True
        if not found_summary:
            raise RuntimeError("attempt4 summary event missing before Rerun write")
        kwargs["events"] = events
        kwargs["recording_metadata"] = dict(VISIBLE_SUMMARY)
        kwargs["recording_id"] = "g0a_d348_a4"
        kwargs["app_id"] = "roarm_d348"
        return original_log_rerun(path, **kwargs)

    def attempt4_write_json(path: Path, value: Any) -> None:
        value = copy.deepcopy(value)
        if path == base.PARAMETER_PATH:
            value["artifact"] = "D348_READABILITY_PARAMETER_FREEZE_V1"
            value["only_changes"] = [
                "metadata TextDocument preserves UTF-8 Korean instead of JSON unicode escapes",
                "attempt4 output directory and immutable attempt3 visual-failure trigger",
            ]
        elif path == base.PREREG_PATH:
            value["artifact"] = "D348_READABILITY_PREREGISTRATION_V1"
            value["attempt"] = "attempt4_readability"
            value["manual_contract"]["static_summary_must_show"] = [
                "frozen tolerance 5%",
                "topology/property 256/256",
                "part gate 128/128",
                "readable Korean: D347 HOME-near and D348 offline distinction",
                "g0a_pass=false",
            ]
        elif path == base.AUTOMATED_PATH:
            value["artifact"] = "D348_READABILITY_AUTOMATED_SUMMARY_V1"
            value["static_summary"] = dict(VISIBLE_SUMMARY)
            value["static_completion_event"] = VISIBLE_EVENT
        elif path == base.COMPLETION_PATH:
            value["artifact"] = "D348_COMPLETION_SUMMARY_V3"
            value["attempt_history"] = {
                "prepare_attempt1": "FAIL preserved; git-status leading-space parser only",
                "scientific_attempt2": (
                    "256/256 and 128/128 numeric PASS; blank/truncated Rerun summary FAIL preserved"
                ),
                "observability_attempt3": (
                    "machine PASS; HOME Korean unicode escapes and truncated event manual FAIL preserved"
                ),
                "readability_attempt4": (
                    "UTF-8 static summary, exact-count/HiDPI gates, and original-image inspection"
                ),
            }
            guards = value.get("scope_guards", {})
            guards["attempt4_scientific_recomputation"] = guards.pop(
                "attempt3_scientific_recomputation", 0
            )
            evidence = value.get("scientific_evidence", {})
            evidence["recomputed_in_attempt4"] = evidence.pop("recomputed_in_attempt3", False)
        original_write_json(path, value)

    def attempt4_write_text(path: Path, value: str) -> None:
        value = value.replace("observability attempt3", "readability attempt4")
        value = value.replace("observability repair", "readability repair")
        original_write_text(path, value)

    base._write_json = attempt4_write_json
    base._write_text = attempt4_write_text
    base.log_rerun = attempt4_log_rerun


def main() -> int:
    _configure_attempt4()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
