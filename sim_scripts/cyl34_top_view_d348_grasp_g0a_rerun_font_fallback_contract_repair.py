#!/usr/bin/env python3
"""D348 attempt5: freeze an ASCII Rerun contract after Korean glyph failure.

Rerun 0.34.1 displayed Korean metadata as missing-glyph boxes even though the
RRD stored UTF-8 correctly.  This reactive attempt removes Korean from the
machine-viewer completion gate, keeps a short unambiguous ASCII duplicate, and
leaves Korean explanation to the session documents and user briefing.  It does
not run PhysX or recompute any scientific quantity.
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

import sim_scripts.cyl34_top_view_d348_grasp_g0a_rerun_summary_readability_repair as a4  # noqa: E402


ATTEMPT4 = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt4_readability"
ATTEMPT4_FAIL = ATTEMPT4 / "d348_readability_manual_visual_inspection_fail.json"
ATTEMPT4_FAIL_MD = ATTEMPT4 / "d348_readability_manual_visual_inspection_fail.md"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt5_ascii_contract"

ATTEMPT4_FROZEN_HASHES = {
    ATTEMPT4 / "d348_readability_preregistration.json": (
        "d0add8d02b4d367bddf2c2c902b94e32bb4d9d68339699d4f70b2363d472c4ff"
    ),
    ATTEMPT4 / "d348_readability_parameter_freeze.json": (
        "09f73750f8ac2f066fcab29312cd6d1d120142c1b8e37ee4064e1da13a297680"
    ),
    ATTEMPT4 / "d348_volume_semantics_readable.rrd": (
        "91b531ac1d7d6bfceba0db7bb4fb509d666c88d96577e3af5413f0f498a17579"
    ),
    ATTEMPT4 / "d348_volume_semantics_readable.rbl": (
        "6c50a279f0f73b97e13c2b27abc81c79ca7f7477a0c8b88142331b42f01acd9c"
    ),
    ATTEMPT4 / "d348_volume_semantics_readable_rerun.png": (
        "6b984eaab4a4600b58a64a2e86f0a54e82fe12a9e63f7f2e9b71a11bd9ae1cca"
    ),
    ATTEMPT4 / "d348_readability_rerun_validation.json": (
        "c95605a5b99b362e9d558fd67f8308372da6c07ef205715d61a3e7e291419bb9"
    ),
    ATTEMPT4 / "d348_readability_automated_summary.json": (
        "d492350943c6119d2b5682501247580c28b4956344283cdbb1aac31932a80a60"
    ),
    ATTEMPT4_FAIL: "7d48de90a82945cf4d623c2b6bc5a9cd8a43692ac974effeabdacf3a12bbb82f",
    ATTEMPT4_FAIL_MD: "3be74e333411aeb4da8fad7a84b4791d937bcce1aabfcc6cac2bd953b996c56a",
}

HELPER_HASHES = {
    REPO / "sim_scripts/cyl34_top_view_d348_grasp_g0a_rerun_summary_readability_repair.py": (
        "fa433899dcaf632d62bee08f291466572d6ed029309348ebea5b948420337d2b"
    ),
    REPO / "sim_scripts/cyl34_top_view_d348_grasp_g0a_rerun_observability_repair.py": (
        "20904aaf1369ef8ee7d06dbd981c7a91cb84f0a75e802e7a50cf6d52011ead02"
    ),
}

ASCII_SUMMARY = {
    "00_RESULT": "SUPPORTED",
    "01_TOLERANCE": "5% FROZEN",
    "02_CALLBACKS": "256/256 PASS",
    "03_PARTS": "128/128 PASS",
    "04_RAW_PAIRS": "128/128 EXACT",
    "05_CLOSED_ORIENTED": "256/256 PASS",
    "06_D347_START": "HOME-near; q5=0 CLOSED",
    "07_D347_PHYSICS": "0 steps",
    "08_D348_MODE": "OFFLINE evidence replay",
    "09_G0A": False,
    "10_PART045_TOPOLOGY_REL": "3.01548618356e-8",
    "11_PART045_QHULL_REL": "0.273316720525",
    "scientific_authority": "attempt2 immutable JSON",
    "viewer_geometry_role": "Float32 display copy",
}
ASCII_EVENT = "PASS | D347 HOME-near | D348 offline | G0a=false"


def _configure_attempt5() -> None:
    a4.OUT_DIR = OUT_DIR
    a4.ATTEMPT3_FAIL = ATTEMPT4_FAIL
    a4.ATTEMPT3_FAIL_MD = ATTEMPT4_FAIL_MD
    a4.ATTEMPT3_FAIL_SHA256 = ATTEMPT4_FROZEN_HASHES[ATTEMPT4_FAIL]
    a4.ATTEMPT3_FAIL_MD_SHA256 = ATTEMPT4_FROZEN_HASHES[ATTEMPT4_FAIL_MD]
    a4.ATTEMPT3_FROZEN_HASHES = dict(ATTEMPT4_FROZEN_HASHES)
    a4.VISIBLE_SUMMARY = dict(ASCII_SUMMARY)
    a4.VISIBLE_EVENT = ASCII_EVENT
    a4._configure_attempt4()

    base = a4.base
    base.OUT_DIR = OUT_DIR
    base.PREREG_PATH = OUT_DIR / "d348_ascii_preregistration.json"
    base.PARAMETER_PATH = OUT_DIR / "d348_ascii_parameter_freeze.json"
    base.RRD_PATH = OUT_DIR / "d348_volume_semantics_ascii.rrd"
    base.RBL_PATH = OUT_DIR / "d348_volume_semantics_ascii.rbl"
    base.SCREENSHOT_PATH = OUT_DIR / "d348_volume_semantics_ascii_rerun.png"
    base.VALIDATION_PATH = OUT_DIR / "d348_ascii_rerun_validation.json"
    base.AUTOMATED_PATH = OUT_DIR / "d348_ascii_automated_summary.json"
    base.AUTOMATED_MD_PATH = OUT_DIR / "d348_ascii_automated_report.md"
    base.MANUAL_PATH = OUT_DIR / "d348_ascii_manual_visual_inspection.json"
    base.MANUAL_MD_PATH = OUT_DIR / "d348_ascii_manual_visual_inspection.md"
    base.COMPLETION_PATH = OUT_DIR / "d348_completion_summary.json"
    base.COMPLETION_MD_PATH = OUT_DIR / "d348_completion_report.md"
    base.HARNESS_PATH = Path(__file__).resolve()
    base.INPUT_HASHES.update(HELPER_HASHES)

    prior_guard = base._input_guard

    def attempt5_input_guard() -> dict[str, Any]:
        guard = copy.deepcopy(prior_guard())
        guard["artifact"] = "D348_ASCII_INPUT_GUARD_V1"
        checks = guard["checks"]
        rename = {
            "attempt3_manual_visual_fail_observed": "attempt4_manual_visual_fail_observed",
            "attempt3_manual_fail_did_not_override_science": (
                "attempt4_manual_fail_did_not_override_science"
            ),
            "attempt3_g0a_false": "attempt4_g0a_false",
        }
        for old, new in rename.items():
            checks[new] = checks.pop(old)
        guard["pass"] = all(checks.values())
        return guard

    def write_json(path: Path, value: Any) -> None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        value = copy.deepcopy(value)
        if path == base.PARAMETER_PATH:
            value["artifact"] = "D348_ASCII_PARAMETER_FREEZE_V1"
            value["only_changes"] = [
                "remove unsupported Korean glyph lines from the Rerun machine-viewer gate",
                "retain the same meaning in short ASCII fields and Korean session reporting",
                "freeze both imported observability helper scripts",
            ]
        elif path == base.PREREG_PATH:
            value["artifact"] = "D348_ASCII_PREREGISTRATION_V1"
            value["attempt"] = "attempt5_ascii_contract"
            value["manual_contract"]["static_summary_must_show"] = [
                "5% FROZEN",
                "256/256 PASS",
                "128/128 PASS",
                "D347 HOME-near, q5=0 CLOSED, 0 steps",
                "D348 OFFLINE evidence replay",
                "G0A=false",
                "no unicode escapes or missing-glyph boxes",
                "completion event fully visible through G0a=false",
            ]
        elif path == base.AUTOMATED_PATH:
            value["artifact"] = "D348_ASCII_AUTOMATED_SUMMARY_V1"
            value["static_summary"] = dict(ASCII_SUMMARY)
            value["static_completion_event"] = ASCII_EVENT
        elif path == base.COMPLETION_PATH:
            value["artifact"] = "D348_COMPLETION_SUMMARY_V4"
            value["attempt_history"] = {
                "prepare_attempt1": "FAIL preserved; git-status leading-space parser only",
                "scientific_attempt2": "numeric PASS; initial Rerun completion view FAIL preserved",
                "observability_attempt3": "summary visible; long/escaped HOME text manual FAIL preserved",
                "readability_attempt4": "ASCII complete; Korean missing-glyph boxes manual FAIL preserved",
                "ascii_contract_attempt5": "short ASCII viewer contract plus Korean session translation",
            }
            guards = value.get("scope_guards", {})
            guards["attempt5_scientific_recomputation"] = guards.pop(
                "attempt3_scientific_recomputation", 0
            )
            evidence = value.get("scientific_evidence", {})
            evidence["recomputed_in_attempt5"] = evidence.pop(
                "recomputed_in_attempt3", False
            )
            value["rerun_language_contract"] = (
                "ASCII in Rerun 0.34.1; Korean translation in session docs and user briefing"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    def write_text(path: Path, value: str) -> None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        value = value.replace("observability attempt3", "ASCII-contract attempt5")
        value = value.replace("observability repair", "ASCII-contract repair")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    base._input_guard = attempt5_input_guard
    base._write_json = write_json
    base._write_text = write_text


def main() -> int:
    _configure_attempt5()
    return a4.base.main()


if __name__ == "__main__":
    raise SystemExit(main())
