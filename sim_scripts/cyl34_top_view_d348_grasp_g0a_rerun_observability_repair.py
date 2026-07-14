#!/usr/bin/env python3
"""Reactive D348 Rerun-only repair after attempt2 manual screenshot failure.

This script never recomputes callback geometry or scientific metrics.  It reads
the immutable, already-passing D348 attempt2 evidence and emits a new Rerun
recording whose static summary and static completion event remain visible on
every selected timeline.  It also registers a logical-window versus HiDPI
raster contract for manual inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rerun as rr


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.rerun_contract import (  # noqa: E402
    RERUN_CONTRACT_VERSION,
    validate_rerun_artifact,
)
from roarm_rl.viz_debug import log_rerun  # noqa: E402


D348_MAIN_PATH = (
    REPO
    / "sim_scripts/cyl34_top_view_d348_grasp_g0a_physx_property_query_volume_semantics.py"
)
spec = importlib.util.spec_from_file_location("d348_scientific", D348_MAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load D348 scientific harness: {D348_MAIN_PATH}")
d348 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d348)


CASE = "g0a_d348"
EXPECTED_HEAD = "d452921e04b7d5082c20d4edcfcc44bcefc7c34d"
NEW_VARIABLES = [
    "physx_property_query_volume_semantics",
    "rerun_static_summary_and_hidpi_contract",
]
NEW_PHYSICAL_VARIABLES: list[str] = []
LOGICAL_SCREENSHOT_SIZE = [2400, 1400]
ALLOWED_DEVICE_PIXEL_RATIOS = [1.0, 2.0]

ATTEMPT2 = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt3_observability"
PREREG_PATH = OUT_DIR / "d348_observability_preregistration.json"
PARAMETER_PATH = OUT_DIR / "d348_observability_parameter_freeze.json"
RRD_PATH = OUT_DIR / "d348_volume_semantics_summary.rrd"
RBL_PATH = OUT_DIR / "d348_volume_semantics_summary.rbl"
SCREENSHOT_PATH = OUT_DIR / "d348_volume_semantics_summary_rerun.png"
VALIDATION_PATH = OUT_DIR / "d348_observability_rerun_validation.json"
AUTOMATED_PATH = OUT_DIR / "d348_observability_automated_summary.json"
AUTOMATED_MD_PATH = OUT_DIR / "d348_observability_automated_report.md"
MANUAL_PATH = OUT_DIR / "d348_observability_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d348_observability_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d348_completion_summary.json"
COMPLETION_MD_PATH = OUT_DIR / "d348_completion_report.md"

EVIDENCE_PATH = ATTEMPT2 / "d348_callback_topology_volume_evidence.json"
CONTROLS_PATH = ATTEMPT2 / "d348_matched_controls.json"
ATTEMPT2_AUTOMATED = ATTEMPT2 / "d348_automated_summary.json"
ATTEMPT2_VALIDATION = ATTEMPT2 / "d348_rerun_validation.json"
ATTEMPT2_PREREG = ATTEMPT2 / "d348_preregistration.json"
ATTEMPT2_PARAMETER = ATTEMPT2 / "d348_parameter_freeze_audit.json"
ATTEMPT2_SOURCE = ATTEMPT2 / "d348_source_semantics.json"
ATTEMPT2_HOME = ATTEMPT2 / "d348_home_start_contract.json"
ATTEMPT2_DECISION = ATTEMPT2 / "d348_volume_semantics_decision.png"
ATTEMPT2_RRD = ATTEMPT2 / "d348_volume_semantics.rrd"
ATTEMPT2_RBL = ATTEMPT2 / "d348_volume_semantics.rbl"
ATTEMPT2_SCREENSHOT = ATTEMPT2 / "d348_volume_semantics_rerun.png"
ATTEMPT2_MANUAL_FAIL = ATTEMPT2 / "d348_manual_visual_inspection_fail.json"
ATTEMPT2_MANUAL_FAIL_MD = ATTEMPT2 / "d348_manual_visual_inspection_fail.md"
VIZ_DEBUG_PATH = REPO / "roarm_rl/viz_debug.py"
HARNESS_PATH = Path(__file__).resolve()

INPUT_HASHES: dict[Path, str] = {
    ATTEMPT2_PREREG: "1141e5914344af8942babd0bdedc288456ffc588d9cc074bd8f03008c0c3f699",
    ATTEMPT2_PARAMETER: "8b8c4883619c21ecfd2be103b826dcd31304140c07405d4fc8b2e20c5c0ad604",
    ATTEMPT2_SOURCE: "49fb43522c319f0df8a884234902de825c6728f4f93466e95bfa13ef6d92390b",
    ATTEMPT2_HOME: "bd4fcb39ffbe8bc5dfb9bc2797f3ba73b1c669f3bad79cee070ebdd40f8816df",
    EVIDENCE_PATH: "83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6",
    CONTROLS_PATH: "35bf839e2e3efe2c64d64819d53db9d4e98dd906bc937961d342eed994a17965",
    ATTEMPT2_AUTOMATED: "68035a0a3f2c1b852a99b78fd1860e9af1e42e88950623ee3b0cc6de7db96d9e",
    ATTEMPT2_VALIDATION: "2061f97edecdbb834b80ef9335f81fd83c006624f95dfd1deb65ae063273e73d",
    ATTEMPT2_DECISION: "00a77861296b048c02c91f18086cf9ff02a728305d923f8c7b751bbdfee46db9",
    ATTEMPT2_RRD: "4a37e383bb8b87177b9607cf326cb031c92461c3a9cd9425e0269a01218c729b",
    ATTEMPT2_RBL: "824aed746f3df8080b34533f3aecf1188a55fcd7ddb55eef0c298a6a11d86da1",
    ATTEMPT2_SCREENSHOT: "9a3049e5c8670543c1bb49350c1e4a4039a40a7e80e5f70fcd7fec85bbfa94b7",
    ATTEMPT2_MANUAL_FAIL: "b4d56ea545248aae3f9098ad1e7e5416333629e18a8dfa91c0ac37ef470fce52",
    ATTEMPT2_MANUAL_FAIL_MD: "ddeec9767c1b083f52092b7779b31f11a39fe53868dc92a729e29f7e1f66b6ab",
    D348_MAIN_PATH: "444cbad5faa878a69252accd2e6923d39fc0cbc0715dd727600638cb4613acba",
}

EXPECTED_COUNTS = {
    "coordinate_frame_count": 2,
    "mesh_count": 512,
    "scalar_row_count": 1280,
    "event_row_count": 133,
    "exact_non_system_entity_count": 2309,
}

VERDICT_MACHINE_PASS = "D348_RERUN_STATIC_SUMMARY_MACHINE_PASS_MANUAL_PENDING"
VERDICT_OBSERVABILITY_FAIL = "D348_RERUN_OBSERVABILITY_INCOMPLETE_STOP"
VERDICT_COMPLETE = "D348_PHYSX_PROPERTY_QUERY_TOPOLOGY_SEMANTICS_SUPPORTED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _input_guard() -> dict[str, Any]:
    rows = []
    for path, expected in INPUT_HASHES.items():
        actual = _sha256(path) if path.is_file() else None
        rows.append(
            {
                "path": _relative(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "pass": actual == expected,
            }
        )
    attempt2 = _json(ATTEMPT2_AUTOMATED)
    evidence = _json(EVIDENCE_PATH)
    controls = _json(CONTROLS_PATH)
    manual_fail = _json(ATTEMPT2_MANUAL_FAIL)
    checks = {
        "hashes_exact": all(row["pass"] for row in rows),
        "attempt2_science_automated_pass": attempt2.get("automated_pass") is True,
        "attempt2_science_manual_pending": attempt2.get("manual_visual_inspection_pending")
        is True,
        "attempt2_evidence_pass": evidence.get("pass") is True,
        "attempt2_controls_pass": controls.get("pass") is True,
        "attempt2_manual_visual_fail_observed": manual_fail.get(
            "manual_visual_inspection_pass"
        )
        is False,
        "attempt2_manual_fail_did_not_override_science": manual_fail.get(
            "scientific_verdict_override"
        )
        is False,
        "attempt2_g0a_false": manual_fail.get("g0a_pass") is False,
    }
    return {
        "artifact": "D348_OBSERVABILITY_INPUT_GUARD_V1",
        "rows": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _expected_contract() -> tuple[list[str], dict[str, list[str]]]:
    entities, components = d348._expected_rrd_contract()
    entity_set = set(entities)
    entity_set.add("events/d348_summary")
    component_copy = {key: list(value) for key, value in components.items()}
    component_copy["events/d348_summary"] = ["TextLog:level", "TextLog:text"]
    return sorted(entity_set), component_copy


def _contract_digest() -> str:
    entities, components = _expected_contract()
    payload = {
        "exact_non_system_entity_paths": entities,
        "exact_timeline_names": ["blueprint", "event_idx", "log_time", "part_idx"],
        "required_components_by_path": components,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _png_dimensions(path: Path) -> list[int] | None:
    if not path.is_file():
        return None
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return [int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")]


def _run_prepare(_args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise FileExistsError(f"observability output already exists: {OUT_DIR}")
    guard = _input_guard()
    manual_fail = _json(ATTEMPT2_MANUAL_FAIL)
    package_versions = {
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "rerun": rr.__version__,
    }
    checks = {
        "input_guard_pass": guard["pass"],
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "two_variables_total": len(NEW_VARIABLES) == 2,
        "new_physical_variables_zero": NEW_PHYSICAL_VARIABLES == [],
        "numpy_pin_exact": package_versions["numpy"] == "1.26.0",
        "psutil_pin_exact": package_versions["psutil"] == "5.9.8",
        "rerun_pin_exact": package_versions["rerun"] == RERUN_CONTRACT_VERSION,
        "original_screenshot_hidpi_2x_observed": manual_fail["rerun_screenshot"].get(
            "observed_device_pixel_ratio"
        )
        == 2.0,
        "exact_entity_count_2309": len(_expected_contract()[0]) == 2309,
    }
    parameter = {
        "artifact": "D348_OBSERVABILITY_PARAMETER_FREEZE_V1",
        "case": CASE,
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "reactive_trigger": {
            "path": _relative(ATTEMPT2_MANUAL_FAIL),
            "sha256": _sha256(ATTEMPT2_MANUAL_FAIL),
            "failed_fields": [
                key
                for key, value in manual_fail["observations"].items()
                if value is False
            ],
        },
        "frozen_science": {
            "evidence_path": _relative(EVIDENCE_PATH),
            "evidence_sha256": _sha256(EVIDENCE_PATH),
            "science_recomputation_allowed": False,
            "property_relative_tolerance": 0.05,
            "part_gate": "128/128",
            "callback_gate": "256/256",
        },
        "only_changes": [
            "static metadata summary in the Rerun blueprint",
            "one static completion TextLog event",
            "manual raster contract accepts registered 1x or 2x device pixel ratio",
        ],
        "logical_screenshot_size": LOGICAL_SCREENSHOT_SIZE,
        "allowed_device_pixel_ratios": ALLOWED_DEVICE_PIXEL_RATIOS,
        "package_versions": package_versions,
        "checks": checks,
        "pass": all(checks.values()),
    }
    OUT_DIR.mkdir(parents=True)
    _write_json(PARAMETER_PATH, parameter)
    prereg_checks = {
        "parameter_pass": parameter["pass"],
        "input_guard_pass": guard["pass"],
        "entity_count_2309": len(_expected_contract()[0]) == 2309,
        "mesh_count_512": EXPECTED_COUNTS["mesh_count"] == 512,
        "scalar_count_1280": EXPECTED_COUNTS["scalar_row_count"] == 1280,
        "event_count_133": EXPECTED_COUNTS["event_row_count"] == 133,
    }
    prereg = {
        "artifact": "D348_OBSERVABILITY_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": "attempt3_observability",
        "input_guard": guard,
        "parameter_path": _relative(PARAMETER_PATH),
        "parameter_sha256": _sha256(PARAMETER_PATH),
        "harness_sha256": _sha256(HARNESS_PATH),
        "scientific_harness_sha256": _sha256(D348_MAIN_PATH),
        "viz_debug_sha256": _sha256(VIZ_DEBUG_PATH),
        "contract_digest": _contract_digest(),
        "expected_counts": EXPECTED_COUNTS,
        "exact_timelines": ["blueprint", "event_idx", "log_time", "part_idx"],
        "blueprint_mode": "volume_semantics_summary",
        "manual_contract": {
            "logical_window": LOGICAL_SCREENSHOT_SIZE,
            "allowed_device_pixel_ratios": ALLOWED_DEVICE_PIXEL_RATIOS,
            "actual_screenshot_view_image_required": True,
            "static_summary_must_show": [
                "frozen tolerance 5%",
                "topology/property 256/256",
                "part gate 128/128",
                "HOME-near D347 source and D348 offline distinction",
                "g0a_pass=false",
            ],
        },
        "scope_guards": {
            "scientific_recomputation": 0,
            "physics_steps": 0,
            "cook_requests": 0,
            "asset_writes": 0,
            "target_queries": 0,
            "g0a_pass": False,
        },
        "checks": prereg_checks,
        "pass": all(prereg_checks.values()),
    }
    _write_json(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"]}, sort_keys=True))
    return 0 if prereg["pass"] else 1


def _run_record(_args: argparse.Namespace) -> int:
    prereg = _json(PREREG_PATH)
    guard = _input_guard()
    evidence = _json(EVIDENCE_PATH)
    prepare_checks = {
        "prereg_pass": prereg.get("pass") is True,
        "input_guard_pass": guard["pass"],
        "parameter_hash_unchanged": prereg.get("parameter_sha256") == _sha256(PARAMETER_PATH),
        "harness_hash_unchanged": prereg.get("harness_sha256") == _sha256(HARNESS_PATH),
        "scientific_harness_hash_unchanged": prereg.get("scientific_harness_sha256")
        == _sha256(D348_MAIN_PATH),
        "viz_debug_hash_unchanged": prereg.get("viz_debug_sha256") == _sha256(VIZ_DEBUG_PATH),
        "contract_digest_unchanged": prereg.get("contract_digest") == _contract_digest(),
        "evidence_pass_without_recomputation": evidence.get("pass") is True,
    }
    if not all(prepare_checks.values()):
        raise RuntimeError(f"observability prereg/input failure: {prepare_checks}")

    rows = evidence["rows"]
    coordinate_frames, meshes, scalars, events = d348._rerun_rows(rows, evidence)
    root = evidence["part045_root_cause"]
    home = _json(ATTEMPT2_HOME)["bounded_answer_ko"]
    static_summary = {
        "00_CASE": CASE,
        "01_FINAL_NUMERIC_VERDICT": "CALLBACK_TOPOLOGY_SEMANTICS_SUPPORTED",
        "02_FROZEN_PROPERTY_TOLERANCE": "5% (unchanged)",
        "03_TOPOLOGY_PROPERTY_GATE": "256/256 PASS",
        "04_PART_GATE": "128/128 PASS",
        "05_RAW_INSTANCE_PROTOTYPE_PAIRS": "128/128 EXACT",
        "06_CLOSED_ORIENTED_CALLBACKS": "256/256 PASS",
        "07_PART045_PROPERTY_VOLUME_M3": root["property_volume_m3"],
        "08_PART045_TOPOLOGY_VOLUME_M3": root["callback_topology_volume_m3"],
        "09_PART045_TOPOLOGY_REL_ERROR": root["callback_topology_relative_error"],
        "10_PART045_VERTEX_ONLY_QHULL_REL_ERROR": root["vertex_only_qhull_relative_error"],
        "11_HOME_CONTRACT_KO": home,
        "12_D348_RUNTIME": "offline re-analysis; no Isaac/PhysX reset or start pose",
        "13_PHYSICS_STEPS": 0,
        "14_ASSET_WRITES": 0,
        "15_G0A_PASS": False,
        "16_NEXT": "target/settle remains blocked pending a separate approved case",
    }
    events.append(
        {
            "entity_path": "events/d348_summary",
            "text": (
                "D348 SUPPORTED | frozen tolerance=5% | topology/property=256/256 | "
                "part gate=128/128 | D347 source=HOME-near q5=0 closed, zero physics steps | "
                "D348=offline re-analysis | g0a_pass=false"
            ),
            "level": "INFO",
            "static": True,
        }
    )
    exact_entities, expected_components = _expected_contract()
    log_status = log_rerun(
        RRD_PATH,
        coordinate_frames=coordinate_frames,
        meshes=meshes,
        scalar_trace=scalars,
        events=events,
        recording_metadata=static_summary,
        recording_id="g0a_d348_volume_semantics_static_summary",
        blueprint_path=RBL_PATH,
        blueprint_mode="volume_semantics_summary",
        live_viewer=False,
        app_id="roarm_g0a_volume_semantics",
    )
    validation = (
        validate_rerun_artifact(
            RRD_PATH,
            expected_entity_paths=[
                "metadata/run",
                "cook/source/link5/parts/part_045",
                "cook/instance/link5/parts/part_045",
                "cook/prototype/gripper_link/parts/part_058",
                "cook/candidate/gripper_link/parts/part_058",
                "events/d348_summary",
            ],
            expected_timeline_names=["event_idx", "part_idx"],
            exact_entity_paths=exact_entities,
            exact_timeline_names=["blueprint", "event_idx", "log_time", "part_idx"],
            expected_entity_components=expected_components,
            blueprint_path=RBL_PATH,
            screenshot_path=SCREENSHOT_PATH,
            screenshot_window_size="2400x1400",
        )
        if log_status.get("ok")
        else {"pass": False, "errors": ["Rerun recording/finalization failed"]}
    )
    observed_count = len(
        validation.get("entity_path_contract", {}).get("observed_non_system", [])
    )
    count_checks = {
        "coordinate_frame_count": log_status.get("coordinate_frame_count") == 2,
        "mesh_count": log_status.get("mesh_count") == 512,
        "scalar_row_count": log_status.get("scalar_row_count") == 1280,
        "event_row_count": log_status.get("event_row_count") == 133,
        "exact_non_system_entity_count": observed_count == 2309,
        "trace_steps_zero": log_status.get("trace_steps") == 0,
    }
    screenshot_dimensions = _png_dimensions(SCREENSHOT_PATH)
    ratio_x = (
        screenshot_dimensions[0] / LOGICAL_SCREENSHOT_SIZE[0]
        if screenshot_dimensions is not None
        else None
    )
    ratio_y = (
        screenshot_dimensions[1] / LOGICAL_SCREENSHOT_SIZE[1]
        if screenshot_dimensions is not None
        else None
    )
    raster_checks = {
        "png_dimensions_available": screenshot_dimensions is not None,
        "uniform_device_pixel_ratio": ratio_x is not None and ratio_x == ratio_y,
        "device_pixel_ratio_registered": ratio_x in ALLOWED_DEVICE_PIXEL_RATIOS,
    }
    validation_report = {
        **validation,
        "d348_observability_log_status": log_status,
        "d348_observability_count_checks": count_checks,
        "d348_observability_raster_checks": raster_checks,
        "d348_observed_non_system_entity_count": observed_count,
        "d348_contract_digest": _contract_digest(),
        "pass": bool(validation.get("pass"))
        and all(count_checks.values())
        and all(raster_checks.values()),
    }
    _write_json(VALIDATION_PATH, validation_report)
    machine_pass = validation_report["pass"]
    pixel_ratio = ratio_x
    automated = {
        "artifact": "D348_OBSERVABILITY_AUTOMATED_SUMMARY_V1",
        "case": CASE,
        "verdict": VERDICT_MACHINE_PASS if machine_pass else VERDICT_OBSERVABILITY_FAIL,
        "machine_pass": machine_pass,
        "manual_visual_inspection_pending": machine_pass,
        "prepare_checks": prepare_checks,
        "science_recomputed": False,
        "scientific_evidence": {
            "path": _relative(EVIDENCE_PATH),
            "sha256": _sha256(EVIDENCE_PATH),
            "pass": evidence["pass"],
        },
        "rerun": {
            "validation_path": _relative(VALIDATION_PATH),
            "validation_sha256": _sha256(VALIDATION_PATH),
            "rrd_path": _relative(RRD_PATH),
            "rrd_sha256": _sha256(RRD_PATH) if RRD_PATH.is_file() else None,
            "rbl_path": _relative(RBL_PATH),
            "rbl_sha256": _sha256(RBL_PATH) if RBL_PATH.is_file() else None,
            "screenshot_path": _relative(SCREENSHOT_PATH),
            "screenshot_sha256": _sha256(SCREENSHOT_PATH) if SCREENSHOT_PATH.is_file() else None,
            "logical_window": LOGICAL_SCREENSHOT_SIZE,
            "raster_dimensions": screenshot_dimensions,
            "observed_device_pixel_ratio": pixel_ratio,
        },
        "static_summary": static_summary,
        "scope_guards": {
            "scientific_recomputation": 0,
            "physics_steps": 0,
            "cook_requests": 0,
            "asset_writes": 0,
            "target_queries": 0,
            "g0a_pass": False,
        },
        "input_guard": guard,
    }
    _write_json(AUTOMATED_PATH, automated)
    _write_text(
        AUTOMATED_MD_PATH,
        "# D348 observability attempt3 자동 결과\n\n"
        f"- 판정: `{automated['verdict']}`\n"
        "- 과학 재계산: `0회`\n"
        "- 정적 요약: 5%, 256/256, 128/128, HOME 계약, g0a=false 기록\n"
        f"- 논리 창: `{LOGICAL_SCREENSHOT_SIZE[0]}x{LOGICAL_SCREENSHOT_SIZE[1]}`\n"
        f"- 실제 raster: `{screenshot_dimensions}` (DPR={pixel_ratio})\n"
        "- 다음: 실제 screenshot 원본 수동 검사\n",
    )
    print(
        json.dumps(
            {
                "stage": "record",
                "verdict": automated["verdict"],
                "manual_pending": automated["manual_visual_inspection_pending"],
                "raster_dimensions": screenshot_dimensions,
            },
            sort_keys=True,
        )
    )
    return 0 if machine_pass else 1


def _manual_checks(manual: dict[str, Any], automated: dict[str, Any]) -> dict[str, bool]:
    shot = manual.get("rerun_screenshot", {})
    observed = _png_dimensions(SCREENSHOT_PATH)
    ratio_x = observed[0] / LOGICAL_SCREENSHOT_SIZE[0] if observed else None
    ratio_y = observed[1] / LOGICAL_SCREENSHOT_SIZE[1] if observed else None
    observations = manual.get("observations", {})
    return {
        "artifact_exact": manual.get("artifact")
        == "D348_OBSERVABILITY_MANUAL_VISUAL_INSPECTION_V1",
        "case_exact": manual.get("case") == CASE,
        "inspection_date_exact": manual.get("inspection_date_kst") == "2026-07-14",
        "inspection_method_original_view_image": manual.get("inspection_method")
        == "original_resolution_view_image",
        "screenshot_path_exact": shot.get("path") == _relative(SCREENSHOT_PATH),
        "screenshot_sha_exact": SCREENSHOT_PATH.is_file()
        and shot.get("sha256") == _sha256(SCREENSHOT_PATH),
        "screenshot_bytes_exact": SCREENSHOT_PATH.is_file()
        and shot.get("bytes") == SCREENSHOT_PATH.stat().st_size,
        "raster_dimensions_exact": shot.get("raster_dimensions") == observed,
        "logical_window_exact": shot.get("requested_logical_window")
        == LOGICAL_SCREENSHOT_SIZE,
        "uniform_device_pixel_ratio": ratio_x is not None and ratio_x == ratio_y,
        "device_pixel_ratio_registered": ratio_x in ALLOWED_DEVICE_PIXEL_RATIOS
        and shot.get("observed_device_pixel_ratio") == ratio_x,
        "required_observations_true": all(
            observations.get(key) is True
            for key in (
                "part045_four_geometry_views_visible",
                "gripper_four_geometry_views_visible",
                "static_summary_panel_populated_and_readable",
                "summary_contains_frozen_5pct",
                "summary_contains_256_of_256",
                "summary_contains_128_of_128",
                "summary_distinguishes_d347_home_near_from_d348_offline",
                "summary_contains_g0a_false",
                "static_completion_event_visible",
                "no_blank_completion_panel",
            )
        ),
        "attempt2_decision_png_reinspected": manual.get(
            "attempt2_decision_png_reinspected"
        )
        is True,
        "manual_pass_true": manual.get("manual_visual_inspection_pass") is True,
        "scientific_override_false": manual.get("scientific_verdict_override") is False,
        "g0a_false": manual.get("g0a_pass") is False,
        "manual_markdown_nonzero": MANUAL_MD_PATH.is_file() and MANUAL_MD_PATH.stat().st_size > 0,
        "automated_machine_pass": automated.get("machine_pass") is True,
    }


def _run_finalize(_args: argparse.Namespace) -> int:
    prereg = _json(PREREG_PATH)
    automated = _json(AUTOMATED_PATH)
    validation = _json(VALIDATION_PATH)
    manual = _json(MANUAL_PATH)
    guard = _input_guard()
    manual_checks = _manual_checks(manual, automated)
    artifact_checks = {
        "input_guard_pass": guard["pass"],
        "prereg_harness_hash_unchanged": prereg.get("harness_sha256") == _sha256(HARNESS_PATH),
        "scientific_harness_hash_unchanged": prereg.get("scientific_harness_sha256")
        == _sha256(D348_MAIN_PATH),
        "viz_debug_hash_unchanged": prereg.get("viz_debug_sha256") == _sha256(VIZ_DEBUG_PATH),
        "parameter_hash_unchanged": prereg.get("parameter_sha256") == _sha256(PARAMETER_PATH),
        "validation_hash_unchanged": automated["rerun"]["validation_sha256"]
        == _sha256(VALIDATION_PATH),
        "rrd_hash_unchanged": automated["rerun"]["rrd_sha256"] == _sha256(RRD_PATH),
        "rbl_hash_unchanged": automated["rerun"]["rbl_sha256"] == _sha256(RBL_PATH),
        "screenshot_hash_unchanged": automated["rerun"]["screenshot_sha256"]
        == _sha256(SCREENSHOT_PATH),
        "machine_validation_pass": validation.get("pass") is True,
        "science_recomputed_false": automated.get("science_recomputed") is False,
    }
    completion_pass = all(artifact_checks.values()) and all(manual_checks.values())
    final_verdict = VERDICT_COMPLETE if completion_pass else VERDICT_OBSERVABILITY_FAIL
    completion = {
        "artifact": "D348_COMPLETION_SUMMARY_V2",
        "case": CASE,
        "final_verdict": final_verdict,
        "completion_contract_pass": completion_pass,
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "attempt_history": {
            "prepare_attempt1": "FAIL preserved; git-status leading-space parser only",
            "scientific_attempt2": "256/256 and 128/128 numeric PASS; manual Rerun screenshot FAIL preserved",
            "observability_attempt3": "static summary and HiDPI contract",
        },
        "scientific_evidence": {
            "path": _relative(EVIDENCE_PATH),
            "sha256": _sha256(EVIDENCE_PATH),
            "pass": _json(EVIDENCE_PATH)["pass"],
            "recomputed_in_attempt3": False,
        },
        "manual_evidence": {
            "path": _relative(MANUAL_PATH),
            "sha256": _sha256(MANUAL_PATH),
            "checks": manual_checks,
            "pass": all(manual_checks.values()),
        },
        "rerun_evidence": automated["rerun"],
        "artifact_checks": artifact_checks,
        "interpretation_ko": (
            "D347의 유일한 27.33% 불일치는 충돌 자산 실패가 아니라 콜백 면 목록을 "
            "버리고 꼭짓점을 새로 볼록껍질화한 비교기 오류였다. 이 결론은 PhysX "
            "107.3.26과 보존된 D347 256개 콜백에 한정된다."
        ),
        "home_answer_ko": _json(ATTEMPT2_HOME)["bounded_answer_ko"],
        "scope_guards": {
            "scientific_attempts": 1,
            "attempt3_scientific_recomputation": 0,
            "physics_steps": 0,
            "cook_requests": 0,
            "asset_writes": 0,
            "target_queries": 0,
            "g0a_pass": False,
            "g0b_rl_ladder_blocked": True,
        },
        "next_case_requires_separate_approval": True,
    }
    _write_json(COMPLETION_PATH, completion)
    evidence = _json(EVIDENCE_PATH)
    root = evidence["part045_root_cause"]
    _write_text(
        COMPLETION_MD_PATH,
        "# D348 최종 완료 보고\n\n"
        f"- 최종 판정: `{final_verdict}`\n"
        "- 콜백 면 ↔ PhysX 부피: `256/256 PASS`\n"
        "- 조각 게이트: `128/128 PASS`\n"
        f"- 최대 상대 오차: `{evidence['aggregate']['max_topology_property_relative_error']:.12g}`\n"
        f"- part_045 기존 Qhull 오차: `{root['vertex_only_qhull_relative_error'] * 100.0:.9f}%`\n"
        f"- part_045 면 부피 오차: `{root['callback_topology_relative_error'] * 100.0:.12g}%`\n"
        "- D348 observability repair의 과학 재계산/물리/cook/asset/target: `전부 0회`\n"
        "- HOME: D347 source는 HOME 근방 q5=0 닫힘, D348은 offline 재판독\n"
        "- G0a: `false` 유지\n",
    )
    print(json.dumps({"stage": "finalize", "final_verdict": final_verdict}, sort_keys=True))
    return 0 if completion_pass else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "record", "finalize"), required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.stage == "prepare":
        return _run_prepare(args)
    if args.stage == "record":
        return _run_record(args)
    return _run_finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
