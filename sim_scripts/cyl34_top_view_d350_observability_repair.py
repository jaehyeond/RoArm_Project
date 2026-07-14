#!/usr/bin/env python3
"""D350 attempt2: artifact-only reactive observability repair.

This harness never imports or launches Isaac.  It preserves D350 attempt1,
revalidates the six delayed/finalized Viewer PNGs, merges the immutable full
RRD with exact per-mesh metadata bindings for the originally required
``part_idx`` timeline, validates the unchanged five-timeline contract, renders
one Rerun inspection screenshot, and combines results only after manual review.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import secrets
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.rerun_contract import sha256_file, validate_rerun_artifact  # noqa: E402
from sim_scripts import cyl34_top_view_d350_fixed_jaw_geometry_viewer as d350  # noqa: E402


CASE = "g0a_d350"
ATTEMPT1_DIR = d350.OUT_DIR
REPAIR_DIR = ATTEMPT1_DIR / "attempt2_observability_repair"
ATTEMPT1_MANIFEST_PATH = REPAIR_DIR / "d350_attempt1_immutability_manifest.json"
PREREG_PATH = REPAIR_DIR / "d350_observability_repair_preregistration.json"
CAPTURE_REVALIDATION_PATH = REPAIR_DIR / "d350_viewer_capture_postclose_revalidation.json"
RERUN_REVALIDATION_PATH = REPAIR_DIR / "d350_rerun_reactive_revalidation.json"
PART_IDX_WITNESS_PATH = REPAIR_DIR / "d350_part_idx_temporal_witness.rrd"
PART_IDX_MANIFEST_PATH = REPAIR_DIR / "d350_part_idx_temporal_witness_manifest.json"
REPAIRED_RRD_PATH = REPAIR_DIR / "d350_fixed_jaw_geometry_5timeline_repaired.rrd"
REPAIRED_RBL_PATH = REPAIR_DIR / "d350_fixed_jaw_geometry_5timeline_repaired.rbl"
RERUN_SCREENSHOT_PATH = REPAIR_DIR / "d350_fixed_jaw_geometry_rerun_repair.png"
REPAIR_SUMMARY_PATH = REPAIR_DIR / "d350_observability_repair_summary.json"
REPAIR_REPORT_PATH = REPAIR_DIR / "d350_observability_repair_report.md"
MANUAL_PATH = REPAIR_DIR / "d350_manual_visual_inspection.json"
MANUAL_MD_PATH = REPAIR_DIR / "d350_manual_visual_inspection.md"
COMPLETION_PATH = REPAIR_DIR / "d350_completion_summary.json"
COMPLETION_MD_PATH = REPAIR_DIR / "d350_completion_report.md"

HARNESS = Path(__file__).resolve()
ORIGINAL_HARNESS = Path(d350.__file__).resolve()
START_HERE = REPO / "START_HERE.md"
ATTEMPT1_SESSION = d350.SESSION_DOC
REPAIR_SESSION = REPO / "claudedocs/session_20260714_grasp_g0a_d350_observability_repair.md"
EXPECTED_HEAD = d350.EXPECTED_HEAD
EXPECTED_ORIGINAL_HARNESS_SHA256 = (
    "99a9b558754c9c4ebf83b265e4bcc70744e1981786066d1343c96cd046d4c538"
)
ORIGINAL_RERUN_CONTRACT_SHA256 = (
    "a2b1ddb4a6fa55b30b7b277c7a1f37fa4b1d01995fa25b4e3385105442aa98de"
)
ORIGINAL_TIMELINES = [
    "blueprint",
    "event_idx",
    "log_time",
    "measurement_idx",
    "part_idx",
]
PART_IDX_MAPPING_SHA256 = (
    "b822ffcf91cf332ab3e26e7b19fc23ca180ae07fe16a8881bdc142d419c6269f"
)
EXPECTED_RECORDING_APP_ID = "roarm_g0a_fixed_jaw_geometry"
EXPECTED_RECORDING_ID = "g0a_d350_fixed_jaw_geometry"
EXPECTED_RERUN_RASTER_DIMENSIONS = "4800x2800"
EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
EXPECTED_RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
EXPECTED_RERUN_VERSION = "0.34.1"
NEW_VARIABLES: list[str] = []
NEW_PHYSICAL_VARIABLES: list[str] = []

EXPECTED_ATTEMPT1_HASHES = {
    "d350_automated_report.md": "ed3307dfafb22e1acfc218ef9aa89865cdba1cbc61b7cb5c2af9a0fb7ab74bbe",
    "d350_automated_summary.json": "a79b26ebcfde9590788e11c59f61cfe27b6d47ececf5b48977f4af140fc49048",
    "d350_fixed_jaw_geometry_measurement.json": "4fe91e4cd37f5b0f064c7e9c91480881973ca51e651132af2c8feb57750e8446",
    "d350_fixed_jaw_geometry.rbl": "7256c3a04655b7665ba423b940073d28fec06c9a3391d472492901f2a23f0576",
    "d350_fixed_jaw_geometry.rrd": "3d0b978d86e7ccff0f02bdadb41ce0f9c09ba24eee05fb489e479f4d6f95ef52",
    "d350_fixed_jaw_semantic_binding.json": "1ec1c309461357eeae89204fa55a498b64d2d216708ab6e6c7dfdd3d0b878c12",
    "d350_live_topology_runtime_binding.json": "9bc8d1c95f3c235816eb1c3c11516f3f27416e45b302cf8b6f9d5ee01ad6ec05",
    "d350_parameter_freeze_audit.json": "d377e58cc72b5b61513b6a121fc783bd3f39e7ff54524687094ece0336ab3c60",
    "d350_preregistration.json": "f1f6ef465b02c69fb3c95fd8ea01fa6ea6c7a57abccba6657b1fe35c9d0a26d2",
    "d350_rerun_validation.json": "e2ecbb9715189d18c289523265a9291c24f2df06c8b0a3db8ae0a88f353a3751",
    "d350_tool_oblique_actual_physx_colliders.png": "a23a0f83bce878a542ca165922be515fc7371d0b78e2cabcec263bdf97f32094",
    "d350_tool_oblique_colored_64plus64.png": "3c5982c3e858cbb6cd30fbff13e1e54db168ee7c185ebf6e72274b20830909a6",
    "d350_tool_side_colored_64plus64.png": "d0e2f6d9aafb888272d0cb34e5cfcfec24dda7a7e8cebd7cf49731e3e29859f5",
    "d350_tool_top_colored_64plus64.png": "cc90af396be2f93508049c5d1894841a0afbf893a6f416887efc40a6afb325b8",
    "d350_validate_preflight.json": "165964d50b00f2879b265c2edaed5945c39f2b52a503ad5501f88cd65276a0b2",
    "d350_viewer_capture_contract.json": "9e43b105cb5b12635a28a9fdfd2748a7d07de81c41a35120b6b8989ab257e9b6",
    "d350_viewer_overlay_contract.json": "efd961cf714c4307cc3db2d27e5f5495e455a76cc32b4957488b7d3c0b1c7592",
    "d350_whole_oblique_actual_physx_colliders.png": "209fdae70a89106624c2576526542dc28b58bc6aa8fe7ecb1d9acad223135b10",
    "d350_whole_oblique_colored_64plus64.png": "38c0102c80cc89ef2d2775a5eb1cde22504246176649c633caf95c0a5a204fab",
}

IMMEDIATE_SUCCESS_NAMES = {"whole_oblique_physx", "tool_top", "tool_oblique"}
DELAYED_FINALIZATION_NAMES = {
    "tool_oblique_physx",
    "whole_oblique_colored",
    "tool_side",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha(path: Path) -> str:
    return sha256_file(path)


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_status() -> dict[str, str]:
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    fields = raw.decode("utf-8", errors="surrogateescape").split("\0")
    result: dict[str, str] = {}
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        status, path = field[:2], field[3:]
        if status[0] in "RC" and index < len(fields):
            path = fields[index]
            index += 1
        result[path] = status
    return result


def _status_scope_pass(status: dict[str, str]) -> bool:
    allowed = {
        _rel(START_HERE),
        _rel(ATTEMPT1_SESSION),
        _rel(REPAIR_SESSION),
        _rel(ORIGINAL_HARNESS),
        _rel(HARNESS),
        *(_rel(ATTEMPT1_DIR / name) for name in EXPECTED_ATTEMPT1_HASHES),
        *d350.EXTERNAL_DIRTY_BASELINE,
    }
    repair_prefix = _rel(REPAIR_DIR) + "/"
    return all(path in allowed or path.startswith(repair_prefix) for path in status)


def _attempt1_top_level_inventory(*, repair_dir_expected: bool) -> dict[str, Any]:
    files = sorted(path.name for path in ATTEMPT1_DIR.iterdir() if path.is_file())
    directories = sorted(path.name for path in ATTEMPT1_DIR.iterdir() if path.is_dir())
    expected_directories = [REPAIR_DIR.name] if repair_dir_expected else []
    checks = {
        "top_level_files_exact": files == sorted(EXPECTED_ATTEMPT1_HASHES),
        "top_level_directories_exact": directories == expected_directories,
    }
    return {
        "files": files,
        "directories": directories,
        "expected_files": sorted(EXPECTED_ATTEMPT1_HASHES),
        "expected_directories": expected_directories,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _isaac_modules_loaded() -> list[str]:
    prefixes = ("isaaclab", "isaacsim", "omni.physx", "omni.isaac")
    return sorted(name for name in sys.modules if name.startswith(prefixes))


def _runtime_environment_contract() -> dict[str, Any]:
    import importlib.metadata
    import rerun as rr
    import rerun_bindings

    cli = shutil.which("rerun")
    version = (
        subprocess.run(
            [cli, "--version"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if cli is not None
        else None
    )
    checks = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "rerun_sdk_exact": str(rr.__version__) == EXPECTED_RERUN_VERSION,
        "rerun_cli_path_exact": cli is not None
        and Path(cli).resolve() == EXPECTED_RERUN_CLI.resolve(),
        "rerun_cli_version_exact": version is not None
        and version.returncode == 0
        and f"rerun-cli {EXPECTED_RERUN_VERSION}" in version.stdout,
        "rerun_bindings_from_isaaclab": str(Path(rerun_bindings.__file__).resolve()).startswith(
            str(EXPECTED_PYTHON.parent.parent.resolve()) + "/"
        ),
        "numpy_pin": importlib.metadata.version("numpy") == "1.26.0",
        "psutil_pin": importlib.metadata.version("psutil") == "5.9.8",
    }
    return {
        "python": sys.executable,
        "rerun_sdk": str(rr.__version__),
        "rerun_cli": cli,
        "rerun_cli_version_stdout": version.stdout if version is not None else None,
        "rerun_bindings": str(Path(rerun_bindings.__file__).resolve()),
        "numpy": importlib.metadata.version("numpy"),
        "psutil": importlib.metadata.version("psutil"),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _png_dimensions(path: Path) -> str | None:
    if not path.is_file():
        return None
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return f"{width}x{height}"


def _attempt1_hashes() -> dict[str, str]:
    rows = {}
    for name in EXPECTED_ATTEMPT1_HASHES:
        path = ATTEMPT1_DIR / name
        if path.is_file():
            rows[name] = _sha(path)
    return rows


def _attempt1_hashes_exact() -> bool:
    return _attempt1_hashes() == EXPECTED_ATTEMPT1_HASHES


def _original_contract_digest() -> str:
    return d350._rrd_digest()


def _mesh_part_mapping_paths() -> list[dict[str, Any]]:
    mesh_paths = [
        *(f"geometry/live_parts/link5/part_{index:03d}" for index in range(64)),
        *(f"geometry/live_parts/gripper_link/part_{index:03d}" for index in range(64)),
        "geometry/fixed_jaw/raw_connected_component",
        "geometry/target/cylinder_collider",
    ]
    return [
        {
            "part_idx": part_idx,
            "mesh_entity_path": mesh_path,
            "metadata_entity_path": f"metadata/meshes/{mesh_path.replace('/', '__')}",
        }
        for part_idx, mesh_path in enumerate(mesh_paths)
    ]


def _recording_chunks(path: Path) -> tuple[list[Any], list[dict[str, str]]]:
    import rerun_bindings as rb

    reader = rb.RrdReaderInternal(str(path))
    stores = [
        {
            "kind": str(entry.kind),
            "application_id": str(entry.application_id),
            "recording_id": str(entry.recording_id),
        }
        for entry in reader.store_entries()
    ]
    return reader.stream().to_chunks(), stores


def _all_store_chunks(path: Path) -> list[dict[str, Any]]:
    import rerun_bindings as rb

    reader = rb.RrdReaderInternal(str(path))
    rows = []
    for entry in reader.store_entries():
        rows.append(
            {
                "store": {
                    "kind": str(entry.kind),
                    "application_id": str(entry.application_id),
                    "recording_id": str(entry.recording_id),
                },
                "chunks": reader.stream(entry).to_chunks(),
            }
        )
    return rows


def _component_text(chunk: Any) -> str:
    values = chunk.to_record_batch().to_pydict().get("TextDocument:text")
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError(f"invalid TextDocument row at {chunk.entity_path}: {values!r}")
    row = values[0]
    if not isinstance(row, list) or len(row) != 1 or not isinstance(row[0], str):
        raise RuntimeError(f"invalid TextDocument component at {chunk.entity_path}: {row!r}")
    return row[0]


def _original_part_mapping() -> tuple[list[dict[str, Any]], dict[str, str]]:
    chunks, _stores = _recording_chunks(d350.RRD_PATH)
    by_path: dict[str, list[Any]] = {}
    for chunk in chunks:
        by_path.setdefault(str(chunk.entity_path).lstrip("/"), []).append(chunk)
    rows = []
    payloads: dict[str, str] = {}
    for expected in _mesh_part_mapping_paths():
        metadata_path = expected["metadata_entity_path"]
        candidates = [
            chunk
            for chunk in by_path.get(metadata_path, [])
            if bool(chunk.is_static) and list(chunk.timeline_names) == []
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one immutable metadata row at {metadata_path}, got {len(candidates)}"
            )
        text_value = _component_text(candidates[0])
        payloads[metadata_path] = text_value
        rows.append(
            {
                **expected,
                "original_text_sha256": hashlib.sha256(text_value.encode()).hexdigest(),
            }
        )
    return rows, payloads


def _part_mapping_digest() -> str:
    rows, _payloads = _original_part_mapping()
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _attempt1_zero_scope_contract() -> dict[str, Any]:
    automated = _json(d350.AUTOMATED_PATH)
    capture = _json(d350.CAPTURE_PATH)
    measurement = _json(d350.MEASUREMENT_PATH)
    execution = automated.get("execution_order", [])
    hold = capture.get("interactive_hold", {})
    counter = capture.get("counter", {})
    timeline = capture.get("timeline", {})
    scope = measurement.get("scope_guards", {})
    checks = {
        "execution_order_nonempty_all_counter_zero": bool(execution)
        and all(row.get("counter") == 0 for row in execution),
        "controlled_physics_steps_zero": automated.get("controlled_physics_steps") == 0,
        "capture_counter_before_after_delta_zero": counter
        == {"before": 0, "after": 0, "delta": 0},
        "timeline_initial_pause_only": timeline.get("interventions") == 1
        and timeline.get("playing_after") is False
        and timeline.get("time_before") == timeline.get("time_after"),
        "interactive_hold_zero_step": hold.get("pass") is True
        and hold.get("counter_before") == 0
        and hold.get("counter_after") == 0
        and hold.get("timeline_interventions") == 0
        and hold.get("timeline_time_before") == hold.get("timeline_time_after")
        and hold.get("state_guard", {}).get("max_abs_delta") == 0.0,
        "target_guard_after_interactive": capture.get(
            "target_guard_after_interactive", {}
        ).get("pass")
        is True,
        "measurement_scope_exact": all(
            scope.get(name) is False
            for name in (
                "asset_write",
                "decomposition_change",
                "g0a_pass",
                "g0b",
                "ladder_promotion",
                "material_mass_actuator_physics_change",
                "rl_or_ppo",
                "settle",
                "target_change",
                "ten_trial",
                "tolerance_change",
            )
        )
        and scope.get("controlled_physics_steps") == 0
        and scope.get("fresh_cook_callback_or_property_query") == 0,
        "promotion_fields_false": all(
            automated.get(name) is False
            for name in (
                "g0a_pass",
                "settle_executed",
                "ten_trial_run",
                "g0b_run",
                "rl_run",
                "ladder_promoted",
            )
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


def _attempt1_science_contract() -> dict[str, Any]:
    binding = _json(d350.BINDING_PATH)
    measurement = _json(d350.MEASUREMENT_PATH)
    automated = _json(d350.AUTOMATED_PATH)
    capture = _json(d350.CAPTURE_PATH)
    checks = {
        "binding_pass": binding.get("pass") is True,
        "measurement_pass": measurement.get("pass") is True,
        "frozen_input_reproduction_pass": measurement.get("frozen_input_reproduction_pass")
        is True,
        "measured_not_aligned": measurement.get("aligned_pass") is None
        and measurement.get("verdict_semantics") == "MEASURED, not ALIGNED_PASS",
        "scientific_verdict_pending_only_for_manual": automated.get("scientific_verdict")
        == d350.VERDICT_PENDING,
        "semantic_and_measurement_summary_pass": automated.get("semantic_binding_pass")
        is True
        and automated.get("measurement_pass") is True,
        "target_bits_exact_before_after": automated.get("target_guard", {}).get("pass")
        is True
        and automated.get("target_guard_after_interactive", {}).get("pass") is True,
        "controlled_physics_steps_zero": automated.get("controlled_physics_steps") == 0,
        "capture_zero_step_hold": capture.get("interactive_hold", {}).get("pass") is True,
        "no_promotion": all(
            automated.get(name) is False
            for name in (
                "g0a_pass",
                "settle_executed",
                "ten_trial_run",
                "g0b_run",
                "rl_run",
                "ladder_promoted",
            )
        ),
    }
    return {
        "checks": checks,
        "metrics": measurement.get("metrics", {}),
        "distance_rows": measurement.get("distances", {}),
        "pass": all(checks.values()),
    }


def _attempt1_failure_contract() -> dict[str, Any]:
    capture = _json(d350.CAPTURE_PATH)
    rerun = _json(d350.RERUN_VALIDATION_PATH)
    automated = _json(d350.AUTOMATED_PATH)
    archive = rerun.get("log_status", {}).get("archive_validation", {})
    entity = archive.get("entity_path_contract", {})
    timeline = archive.get("timeline_contract", {})
    component = archive.get("component_contract", {})
    rows = capture.get("captures", {})
    immediate = {name for name, row in rows.items() if row.get("ok") is True}
    delayed = {name for name, row in rows.items() if row.get("ok") is False}
    immutable = automated.get("immutability", {})
    checks = {
        "attempt1_automated_false": automated.get("automated_pass") is False,
        "attempt1_observability_fail": automated.get("observability_verdict")
        == d350.VERDICT_VISUAL,
        "capture_v1_false": capture.get("pass") is False
        and capture.get("checks", {}).get("all_six_captures") is False,
        "capture_tokens_six_true": len(rows) == 6
        and all(row.get("capture_result") is True for row in rows.values()),
        "immediate_success_set_exact": immediate == IMMEDIATE_SUCCESS_NAMES,
        "delayed_finalization_set_exact": delayed == DELAYED_FINALIZATION_NAMES,
        "rerun_v1_false": rerun.get("pass") is False,
        "rrd_footer_valid": archive.get("verify", {}).get("ok") is True,
        "rbl_footer_valid": archive.get("blueprint_verify", {}).get("ok") is True,
        "entity_296_intact": entity.get("pass") is True
        and len(entity.get("observed_non_system", [])) == 296,
        "legacy_empty_component_contract_pass_preserved": component.get("pass") is True
        and component.get("checks") == {},
        "timeline_failure_only_part_idx": timeline.get("observed")
        == [name for name in ORIGINAL_TIMELINES if name != "part_idx"]
        and timeline.get("checks", {}).get("part_idx") is False
        and archive.get("errors") == ["RRD timeline contract failed"],
        "immutability_inputs_true": all(
            immutable.get(name) is True
            for name in (
                "source_inventories_exact",
                "input_hashes_exact",
                "external_user_files_exact",
            )
        ),
        "asset_write_expected_false": immutable.get("asset_write") is False,
        "original_aggregate_false": immutable.get("pass") is False,
    }
    return {"checks": checks, "pass": all(checks.values())}


def _run_prepare(args: argparse.Namespace) -> int:
    if REPAIR_DIR.exists():
        raise RuntimeError(f"forward-only repair directory already exists: {REPAIR_DIR}")
    science = _attempt1_science_contract()
    failure = _attempt1_failure_contract()
    zero_scope = _attempt1_zero_scope_contract()
    runtime_environment = _runtime_environment_contract()
    status = _git_status()
    mapping_rows, _mapping_payloads = _original_part_mapping()
    _original_chunks, original_stores = _recording_chunks(d350.RRD_PATH)
    recording_stores = [row for row in original_stores if row["kind"] == "recording"]
    blueprint_stores = [row for row in original_stores if row["kind"] == "blueprint"]
    prewrite_checks = {
        "science_contract_pass": science["pass"],
        "observed_failure_contract_pass": failure["pass"],
        "attempt1_zero_scope_pass": zero_scope["pass"],
        "head_exact": _git_head() == EXPECTED_HEAD,
        "status_scope": _status_scope_pass(status),
        "attempt1_inventory_exact_before_repair": _attempt1_top_level_inventory(
            repair_dir_expected=False
        )["pass"],
        "all_attempt1_hashes_exact": _attempt1_hashes_exact(),
        "original_harness_exact": _sha(ORIGINAL_HARNESS)
        == EXPECTED_ORIGINAL_HARNESS_SHA256,
        "external_user_files_unchanged": d350._external_baseline()["pass"],
        "original_inputs_exact": d350._input_hashes() == d350.EXPECTED_INPUT_HASHES,
        "repair_variables_zero": NEW_VARIABLES == [] and NEW_PHYSICAL_VARIABLES == [],
        "original_contract_digest_exact": _original_contract_digest()
        == ORIGINAL_RERUN_CONTRACT_SHA256,
        "original_five_timelines_retained": ORIGINAL_TIMELINES
        == ["blueprint", "event_idx", "log_time", "measurement_idx", "part_idx"],
        "part_mapping_count_130": len(mapping_rows) == 130,
        "part_mapping_digest_exact": _part_mapping_digest()
        == PART_IDX_MAPPING_SHA256,
        "original_recording_store_exact": recording_stores
        == [
            {
                "kind": "recording",
                "application_id": EXPECTED_RECORDING_APP_ID,
                "recording_id": EXPECTED_RECORDING_ID,
            }
        ],
        "original_blueprint_store_exactly_one": len(blueprint_stores) == 1
        and blueprint_stores[0]["application_id"] == EXPECTED_RECORDING_APP_ID,
        "active_case_registered": "attempt2_observability_repair"
        in START_HERE.read_text(encoding="utf-8"),
        "repair_session_present": REPAIR_SESSION.is_file(),
        "no_isaac_modules_loaded": _isaac_modules_loaded() == [],
        "runtime_environment_exact": runtime_environment["pass"],
    }
    if not all(prewrite_checks.values()):
        raise RuntimeError(f"D350 repair prepare prewrite checks failed: {prewrite_checks}")
    original = {
        "artifact": "D350_ATTEMPT1_IMMUTABILITY_MANIFEST_V1",
        "case": CASE,
        "attempt": "attempt1",
        "original_harness": {
            "path": _rel(ORIGINAL_HARNESS),
            "sha256": _sha(ORIGINAL_HARNESS),
        },
        "attempt1_files": {
            name: {
                "path": _rel(ATTEMPT1_DIR / name),
                "bytes": (ATTEMPT1_DIR / name).stat().st_size,
                "sha256": _sha(ATTEMPT1_DIR / name),
            }
            for name in sorted(EXPECTED_ATTEMPT1_HASHES)
        },
        "top_level_inventory_before_repair": _attempt1_top_level_inventory(
            repair_dir_expected=False
        ),
        "checks": {
            "original_harness_exact": _sha(ORIGINAL_HARNESS)
            == EXPECTED_ORIGINAL_HARNESS_SHA256,
            "all_attempt1_hashes_exact": _attempt1_hashes_exact(),
            "top_level_inventory_exact": _attempt1_top_level_inventory(
                repair_dir_expected=False
            )["pass"],
        },
    }
    original["pass"] = all(original["checks"].values())
    _write_json(ATTEMPT1_MANIFEST_PATH, original)
    checks = {
        **prewrite_checks,
        "manifest_pass": original["pass"],
        "repair_output_contains_manifest_only": sorted(
            path.name for path in REPAIR_DIR.iterdir()
        )
        == [ATTEMPT1_MANIFEST_PATH.name],
    }
    prereg = {
        "artifact": "D350_OBSERVABILITY_REPAIR_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": "attempt2_observability_repair",
        "repair_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "git_head": _git_head(),
        "git_status": status,
        "state_hashes": {
            "start_here": _sha(START_HERE),
            "repair_session": _sha(REPAIR_SESSION),
        },
        "harness_sha256": _sha(HARNESS),
        "original_harness_sha256": _sha(ORIGINAL_HARNESS),
        "attempt1_manifest_sha256": _sha(ATTEMPT1_MANIFEST_PATH),
        "attempt1_hashes": _attempt1_hashes(),
        "attempt1_science_contract": science,
        "attempt1_failure_contract": failure,
        "attempt1_zero_scope_contract": zero_scope,
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "original_rerun_contract_sha256": _original_contract_digest(),
        "original_exact_timelines": ORIGINAL_TIMELINES,
        "part_idx_mapping": mapping_rows,
        "part_idx_mapping_sha256": _part_mapping_digest(),
        "recording_store_contract": {
            "application_id": EXPECTED_RECORDING_APP_ID,
            "recording_id": EXPECTED_RECORDING_ID,
            "stores": original_stores,
        },
        "runtime_environment": runtime_environment,
        "repair_method": (
            "merge immutable full attempt1 RRD with 130 exact per-mesh metadata "
            "temporal bindings; no Mesh3D reconstruction"
        ),
        "scope_guards": {
            "isaac_launch": False,
            "physics_steps": 0,
            "target_write": False,
            "geometry_remeasurement": False,
            "attempt1_overwrite": False,
            "asset_write": False,
            "settle": False,
            "ten_trial": False,
            "g0b": False,
            "rl_or_ppo": False,
            "ladder_promotion": False,
            "g0a_pass": False,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"]}, sort_keys=True))
    return 0 if prereg["pass"] else 2


def _decode_png_source(
    source: Path | io.BytesIO,
    *,
    expected_mode: str,
    expected_size: list[int],
) -> dict[str, Any]:
    verify_ok = False
    load_ok = False
    mode = None
    size = None
    error = None
    try:
        with Image.open(source) as image:
            image.verify()
        verify_ok = True
        if isinstance(source, io.BytesIO):
            source.seek(0)
        with Image.open(source) as image:
            image.load()
            mode = image.mode
            size = list(image.size)
        load_ok = True
    except Exception as exc:  # pragma: no cover - recorded negative path
        error = f"{type(exc).__name__}: {exc}"
    return {
        "verify_ok": verify_ok,
        "load_ok": load_ok,
        "mode": mode,
        "size": size,
        "error": error,
        "pass": verify_ok
        and load_ok
        and mode == expected_mode
        and size == expected_size,
    }


def _decode_viewer_png(path: Path) -> dict[str, Any]:
    return _decode_png_source(path, expected_mode="RGBA", expected_size=[1280, 720])


def _decode_rerun_png(path: Path) -> dict[str, Any]:
    return _decode_png_source(path, expected_mode="RGBA", expected_size=[4800, 2800])


def _sample_pngs() -> dict[str, Any]:
    return {
        name: {
            "bytes": path.stat().st_size if path.is_file() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
            "sha256": _sha(path) if path.is_file() else None,
        }
        for name, path in d350.VIEWER_PNGS.items()
    }


def _png_decoder_negative_controls() -> dict[str, bool]:
    truncated = io.BytesIO(next(iter(d350.VIEWER_PNGS.values())).read_bytes()[:64])
    wrong_mode = io.BytesIO()
    Image.new("RGB", (1280, 720), color=(0, 0, 0)).save(wrong_mode, format="PNG")
    wrong_mode.seek(0)
    wrong_size = io.BytesIO()
    Image.new("RGBA", (1, 1), color=(0, 0, 0, 255)).save(wrong_size, format="PNG")
    wrong_size.seek(0)
    return {
        "truncated_rejected_by_production_decoder": not _decode_png_source(
            truncated, expected_mode="RGBA", expected_size=[1280, 720]
        )["pass"],
        "valid_wrong_mode_rejected_by_production_decoder": not _decode_png_source(
            wrong_mode, expected_mode="RGBA", expected_size=[1280, 720]
        )["pass"],
        "valid_wrong_size_rejected_by_production_decoder": not _decode_png_source(
            wrong_size, expected_mode="RGBA", expected_size=[1280, 720]
        )["pass"],
    }


def _capture_revalidation() -> dict[str, Any]:
    original = _json(d350.CAPTURE_PATH)
    preflight = _json(d350.PREFLIGHT_PATH)
    automated = _json(d350.AUTOMATED_PATH)
    overlay = _json(d350.OVERLAY_PATH)
    validate_pid = int(preflight["validate_process_identity"]["pid"])
    process_absent_before_sampling = not psutil.pid_exists(validate_pid)
    sample_a = _sample_pngs()
    time.sleep(1.0)
    sample_b = _sample_pngs()
    time.sleep(1.0)
    sample_c = _sample_pngs()
    rows = {}
    for name, path in d350.VIEWER_PNGS.items():
        original_row = original["captures"][name]
        rows[name] = {
            "path": _rel(path),
            "original_capture_token": original_row.get("capture_result"),
            "original_immediate_ok": original_row.get("ok"),
            "samples": [sample_a[name], sample_b[name], sample_c[name]],
            "decode": _decode_viewer_png(path),
            "png_dimensions": _png_dimensions(path),
            "mtime_before_capture_contract": bool(
                path.is_file() and path.stat().st_mtime_ns < d350.CAPTURE_PATH.stat().st_mtime_ns
            ),
            "mtime_before_automated_summary": bool(
                path.is_file() and path.stat().st_mtime_ns < d350.AUTOMATED_PATH.stat().st_mtime_ns
            ),
            "original_path_exact": original_row.get("path") == _rel(path),
            "original_success_exact": bool(
                name not in IMMEDIATE_SUCCESS_NAMES
                or (
                    original_row.get("bytes") == sample_c[name]["bytes"]
                    and original_row.get("sha256") == sample_c[name]["sha256"]
                    and original_row.get("dimensions") == _png_dimensions(path)
                )
            ),
        }
        rows[name]["post_decode_sample"] = _sample_pngs()[name]
        rows[name]["stable"] = (
            sample_a[name]
            == sample_b[name]
            == sample_c[name]
            == rows[name]["post_decode_sample"]
        )
        rows[name]["pass"] = all(
            [
                rows[name]["original_capture_token"] is True,
                rows[name]["stable"],
                bool(sample_c[name]["bytes"] and sample_c[name]["bytes"] > 0),
                rows[name]["decode"]["pass"],
                rows[name]["png_dimensions"] == "1280x720",
                rows[name]["mtime_before_capture_contract"],
                rows[name]["mtime_before_automated_summary"],
                rows[name]["original_path_exact"],
                rows[name]["original_success_exact"],
            ]
        )
    hashes = [sample_c[name]["sha256"] for name in sorted(sample_c)]
    process_absent_after_sampling = not psutil.pid_exists(validate_pid)
    negative_controls = _png_decoder_negative_controls()
    checks = {
        "attempt1_capture_contract_false_preserved": original.get("pass") is False,
        "expected_six_paths_exact": set(original.get("captures", {}))
        == set(d350.VIEWER_PNGS),
        "capture_tokens_six_true": all(
            row.get("capture_result") is True for row in original["captures"].values()
        ),
        "immediate_and_delayed_sets_exact": {
            name for name, row in original["captures"].items() if row.get("ok") is True
        }
        == IMMEDIATE_SUCCESS_NAMES
        and {
            name for name, row in original["captures"].items() if row.get("ok") is False
        }
        == DELAYED_FINALIZATION_NAMES,
        "preflight_runtime_outputs_absent": preflight.get("checks", {}).get(
            "runtime_outputs_absent"
        )
        is True,
        "attempt1_process_absent_before_sampling": process_absent_before_sampling,
        "attempt1_process_absent_after_sampling": process_absent_after_sampling,
        "all_six_stable_decodable_exact": all(row["pass"] for row in rows.values()),
        "six_hashes_pairwise_distinct": len(set(hashes)) == 6,
        "production_decoder_negative_controls": all(negative_controls.values()),
        "zero_step_target_state_contract": all(
            [
                original.get("checks", {}).get("counter_zero_unchanged") is True,
                original.get("checks", {}).get("timeline_time_unchanged") is True,
                original.get("checks", {}).get("state_guard") is True,
                original.get("checks", {}).get("interactive_hold_zero_step") is True,
                original.get("checks", {}).get("target_bits_exact_after_interactive") is True,
                original.get("checks", {}).get("persistent_settings_restored") is True,
                original.get("checks", {}).get("session_only_guides_after_capture") is True,
                automated.get("controlled_physics_steps") == 0,
                overlay.get("pass") is True,
            ]
        ),
    }
    return {
        "artifact": "D350_VIEWER_CAPTURE_POSTCLOSE_REVALIDATION_V1",
        "case": CASE,
        "attempt1_capture_contract_sha256": _sha(d350.CAPTURE_PATH),
        "interpretation": (
            "capture tokens were 6/6; three immediate stat checks raced the asynchronous "
            "file sink; the immutable attempt1 false is preserved"
        ),
        "rows": rows,
        "negative_controls": negative_controls,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _verify_rrd(path: Path) -> dict[str, Any]:
    cli = shutil.which("rerun")
    if cli is None or Path(cli).resolve() != EXPECTED_RERUN_CLI.resolve():
        return {"ok": False, "error": "rerun CLI not found"}
    completed = subprocess.run(
        [cli, "rrd", "verify", "--check-footers", "true", str(path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "command": [cli, "rrd", "verify", "--check-footers", "true", str(path)],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "ok": completed.returncode == 0,
    }


def _write_part_idx_witness() -> dict[str, Any]:
    import rerun as rr

    if str(rr.__version__) != EXPECTED_RERUN_VERSION:
        raise RuntimeError(
            f"Rerun SDK mismatch: {rr.__version__} != {EXPECTED_RERUN_VERSION}"
        )
    if PART_IDX_WITNESS_PATH.exists():
        raise RuntimeError(f"refusing to overwrite {PART_IDX_WITNESS_PATH}")
    mapping_rows, payloads = _original_part_mapping()
    with rr.RecordingStream(
        EXPECTED_RECORDING_APP_ID,
        recording_id=EXPECTED_RECORDING_ID,
        make_default=False,
        send_properties=False,
    ) as recording:
        recording.save(str(PART_IDX_WITNESS_PATH), write_footer=True)
        for row in mapping_rows:
            recording.reset_time()
            recording.set_time("part_idx", sequence=int(row["part_idx"]))
            recording.log(
                row["metadata_entity_path"],
                rr.TextDocument(payloads[row["metadata_entity_path"]]),
                static=False,
            )
        recording.flush(timeout_sec=30.0)
    verify = _verify_rrd(PART_IDX_WITNESS_PATH)
    return {
        "path": _rel(PART_IDX_WITNESS_PATH),
        "bytes": PART_IDX_WITNESS_PATH.stat().st_size,
        "sha256": _sha(PART_IDX_WITNESS_PATH),
        "mapping": mapping_rows,
        "mapping_sha256": _part_mapping_digest(),
        "footer_verify": verify,
        "pass": bool(
            len(mapping_rows) == 130
            and _part_mapping_digest() == PART_IDX_MAPPING_SHA256
            and verify.get("ok") is True
        ),
    }


def _merge_full_rerun_archive() -> dict[str, Any]:
    cli = shutil.which("rerun")
    if cli is None or Path(cli).resolve() != EXPECTED_RERUN_CLI.resolve():
        return {"ok": False, "error": "rerun CLI not found"}
    if REPAIRED_RRD_PATH.exists() or REPAIRED_RBL_PATH.exists():
        raise RuntimeError("refusing to overwrite repaired RRD/RBL")
    command = [
        cli,
        "rrd",
        "merge",
        "-o",
        str(REPAIRED_RRD_PATH),
        str(d350.RRD_PATH),
        str(PART_IDX_WITNESS_PATH),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode == 0 and REPAIRED_RRD_PATH.is_file():
        shutil.copyfile(d350.RBL_PATH, REPAIRED_RBL_PATH)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "rrd_nonzero": REPAIRED_RRD_PATH.is_file()
        and REPAIRED_RRD_PATH.stat().st_size > 0,
        "rbl_exact_forward_copy": REPAIRED_RBL_PATH.is_file()
        and _sha(REPAIRED_RBL_PATH) == _sha(d350.RBL_PATH),
        "ok": completed.returncode == 0
        and REPAIRED_RRD_PATH.is_file()
        and REPAIRED_RRD_PATH.stat().st_size > 0
        and REPAIRED_RBL_PATH.is_file()
        and _sha(REPAIRED_RBL_PATH) == _sha(d350.RBL_PATH),
    }


def _rrd_structural_delta() -> dict[str, Any]:
    original_chunks, original_stores = _recording_chunks(d350.RRD_PATH)
    repaired_chunks, repaired_stores = _recording_chunks(REPAIRED_RRD_PATH)
    original_all_stores = _all_store_chunks(d350.RRD_PATH)
    repaired_all_stores = _all_store_chunks(REPAIRED_RRD_PATH)
    original_id_list = [str(chunk.id) for chunk in original_chunks]
    repaired_id_list = [str(chunk.id) for chunk in repaired_chunks]
    original_ids = set(original_id_list)
    repaired_by_id = {str(chunk.id): chunk for chunk in repaired_chunks}
    added = [chunk for chunk in repaired_chunks if str(chunk.id) not in original_ids]
    mapping_rows, payloads = _original_part_mapping()
    expected_by_path = {
        "/" + row["metadata_entity_path"]: row for row in mapping_rows
    }
    added_by_path: dict[str, list[Any]] = {}
    for chunk in added:
        added_by_path.setdefault(str(chunk.entity_path), []).append(chunk)
    mapping_checks = []
    for path, expected in expected_by_path.items():
        candidates = added_by_path.get(path, [])
        row_check: dict[str, Any] = {
            "part_idx": expected["part_idx"],
            "metadata_entity_path": expected["metadata_entity_path"],
            "mesh_entity_path": expected["mesh_entity_path"],
            "candidate_count": len(candidates),
        }
        if len(candidates) == 1:
            chunk = candidates[0]
            batch = chunk.to_record_batch().to_pydict()
            component_names = sorted(
                name
                for name in batch
                if name not in {"rerun.controls.RowId", "log_time", "part_idx"}
            )
            observed_part_idx = batch.get("part_idx")
            observed_text = _component_text(chunk)
            row_check.update(
                {
                    "is_static": bool(chunk.is_static),
                    "timelines": sorted(str(name) for name in chunk.timeline_names),
                    "components": component_names,
                    "observed_part_idx": observed_part_idx,
                    "observed_text_sha256": hashlib.sha256(
                        observed_text.encode()
                    ).hexdigest(),
                    "expected_text_sha256": hashlib.sha256(
                        payloads[expected["metadata_entity_path"]].encode()
                    ).hexdigest(),
                }
            )
            row_check["pass"] = bool(
                chunk.is_static is False
                and set(chunk.timeline_names) == {"log_time", "part_idx"}
                and component_names == ["TextDocument:text"]
                and observed_part_idx == [expected["part_idx"]]
                and observed_text == payloads[expected["metadata_entity_path"]]
            )
        else:
            row_check["pass"] = False
        mapping_checks.append(row_check)

    def has_component(chunk: Any, prefix: str) -> bool:
        return any(
            str(name).startswith(prefix)
            for name in chunk.to_record_batch().to_pydict()
        )

    original_paths = {str(chunk.entity_path) for chunk in original_chunks}
    repaired_paths = {str(chunk.entity_path) for chunk in repaired_chunks}
    original_payload_checks = []
    for chunk in original_chunks:
        chunk_id = str(chunk.id)
        repaired_chunk = repaired_by_id.get(chunk_id)
        exact = bool(
            repaired_chunk is not None
            and str(repaired_chunk.entity_path) == str(chunk.entity_path)
            and bool(repaired_chunk.is_static) == bool(chunk.is_static)
            and sorted(str(name) for name in repaired_chunk.timeline_names)
            == sorted(str(name) for name in chunk.timeline_names)
            and chunk.to_record_batch().equals(
                repaired_chunk.to_record_batch(), check_metadata=True
            )
        )
        original_payload_checks.append(
            {
                "chunk_id": chunk_id,
                "entity_path": str(chunk.entity_path),
                "exact": exact,
            }
        )
    original_blueprint_chunks = next(
        row["chunks"]
        for row in original_all_stores
        if row["store"]["kind"] == "blueprint"
    )
    repaired_blueprint_chunks = next(
        row["chunks"]
        for row in repaired_all_stores
        if row["store"]["kind"] == "blueprint"
    )
    repaired_blueprint_by_id = {
        str(chunk.id): chunk for chunk in repaired_blueprint_chunks
    }
    blueprint_payload_checks = []
    for chunk in original_blueprint_chunks:
        chunk_id = str(chunk.id)
        repaired_chunk = repaired_blueprint_by_id.get(chunk_id)
        exact = bool(
            repaired_chunk is not None
            and str(repaired_chunk.entity_path) == str(chunk.entity_path)
            and bool(repaired_chunk.is_static) == bool(chunk.is_static)
            and sorted(str(name) for name in repaired_chunk.timeline_names)
            == sorted(str(name) for name in chunk.timeline_names)
            and chunk.to_record_batch().equals(
                repaired_chunk.to_record_batch(), check_metadata=True
            )
        )
        blueprint_payload_checks.append(
            {
                "chunk_id": chunk_id,
                "entity_path": str(chunk.entity_path),
                "exact": exact,
            }
        )
    recording_stores = [row for row in repaired_stores if row["kind"] == "recording"]
    blueprint_stores = [row for row in repaired_stores if row["kind"] == "blueprint"]
    checks = {
        "original_chunk_ids_unique": len(original_id_list) == len(original_ids),
        "repaired_chunk_ids_unique": len(repaired_id_list) == len(set(repaired_id_list)),
        "original_chunk_ids_retained": original_ids.issubset(repaired_by_id),
        "original_payloads_297_exact": len(original_payload_checks) == 297
        and all(row["exact"] for row in original_payload_checks),
        "embedded_blueprint_payloads_12_exact": len(original_blueprint_chunks) == 12
        and len(repaired_blueprint_chunks) == 12
        and len({str(chunk.id) for chunk in original_blueprint_chunks}) == 12
        and len({str(chunk.id) for chunk in repaired_blueprint_chunks}) == 12
        and all(row["exact"] for row in blueprint_payload_checks),
        "recording_chunks_297_plus_130": len(original_chunks) == 297
        and len(repaired_chunks) == 427
        and len(added) == 130,
        "static_chunks_279_unchanged": sum(bool(chunk.is_static) for chunk in original_chunks)
        == sum(bool(chunk.is_static) for chunk in repaired_chunks)
        == 279,
        "entity_paths_unchanged": original_paths == repaired_paths,
        "added_paths_exact_expected_metadata": set(added_by_path) == set(expected_by_path),
        "mapping_0_through_129_exact": len(mapping_checks) == 130
        and all(row["pass"] for row in mapping_checks)
        and [row["part_idx"] for row in mapping_checks] == list(range(130)),
        "added_mesh_chunks_zero": not any(
            has_component(chunk, "Mesh3D:") for chunk in added
        ),
        "mesh_chunks_130_unchanged": sum(
            has_component(chunk, "Mesh3D:") for chunk in original_chunks
        )
        == sum(has_component(chunk, "Mesh3D:") for chunk in repaired_chunks)
        == 130,
        "text_document_chunks_131_plus_130": sum(
            has_component(chunk, "TextDocument:") for chunk in original_chunks
        )
        == 131
        and sum(has_component(chunk, "TextDocument:") for chunk in repaired_chunks)
        == 261,
        "recording_store_exact": recording_stores
        == [
            {
                "kind": "recording",
                "application_id": EXPECTED_RECORDING_APP_ID,
                "recording_id": EXPECTED_RECORDING_ID,
            }
        ],
        "blueprint_store_exactly_one": len(blueprint_stores) == 1
        and blueprint_stores[0]["application_id"] == EXPECTED_RECORDING_APP_ID,
        "store_set_preserved": repaired_stores == original_stores,
        "mapping_digest_exact": _part_mapping_digest() == PART_IDX_MAPPING_SHA256,
    }
    return {
        "original_stores": original_stores,
        "repaired_stores": repaired_stores,
        "original_recording_chunk_count": len(original_chunks),
        "repaired_recording_chunk_count": len(repaired_chunks),
        "added_chunk_count": len(added),
        "original_payload_exact_count": sum(
            row["exact"] for row in original_payload_checks
        ),
        "original_payload_checks": original_payload_checks,
        "embedded_blueprint_payload_exact_count": sum(
            row["exact"] for row in blueprint_payload_checks
        ),
        "embedded_blueprint_payload_checks": blueprint_payload_checks,
        "mapping_rows": mapping_checks,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _rerun_revalidation() -> dict[str, Any]:
    original = _json(d350.RERUN_VALIDATION_PATH)
    original_archive = original["log_status"]["archive_validation"]
    original_timeline = original_archive.get("timeline_contract", {})
    witness = _write_part_idx_witness()
    merge = _merge_full_rerun_archive()
    if not witness["pass"] or not merge.get("ok"):
        raise RuntimeError(f"D350 offline Rerun generation failed: {witness=} {merge=}")
    structural = _rrd_structural_delta()
    witness_manifest = {
        "artifact": "D350_PART_IDX_TEMPORAL_WITNESS_MANIFEST_V1",
        "case": CASE,
        "role": (
            "exact per-mesh metadata temporal binding merged into the immutable full "
            "attempt1 archive; not a standalone decision artifact"
        ),
        "original_rrd": {
            "path": _rel(d350.RRD_PATH),
            "sha256": _sha(d350.RRD_PATH),
        },
        "witness": witness,
        "merged_rrd": {
            "path": _rel(REPAIRED_RRD_PATH),
            "bytes": REPAIRED_RRD_PATH.stat().st_size,
            "sha256": _sha(REPAIRED_RRD_PATH),
        },
        "merged_rbl": {
            "path": _rel(REPAIRED_RBL_PATH),
            "bytes": REPAIRED_RBL_PATH.stat().st_size,
            "sha256": _sha(REPAIRED_RBL_PATH),
            "original_byte_exact": _sha(REPAIRED_RBL_PATH) == _sha(d350.RBL_PATH),
        },
        "merge": merge,
        "structural_delta": structural,
        "part_idx_mapping_sha256": _part_mapping_digest(),
        "pass": bool(witness["pass"] and merge.get("ok") and structural["pass"]),
    }
    _write_json(PART_IDX_MANIFEST_PATH, witness_manifest)

    entities, components = d350._expected_rrd_contract()
    validation = validate_rerun_artifact(
        REPAIRED_RRD_PATH,
        expected_entity_paths=[
            "geometry/live_parts/link5/part_000",
            "geometry/live_parts/gripper_link/part_063",
            "geometry/fixed_jaw/raw_connected_component",
            "geometry/target/cylinder_collider",
            "geometry/arrows/fixed_jaw_centerline",
        ],
        expected_timeline_names=["part_idx", "measurement_idx", "event_idx"],
        exact_entity_paths=entities,
        exact_timeline_names=ORIGINAL_TIMELINES,
        expected_entity_components=components,
        blueprint_path=REPAIRED_RBL_PATH,
        screenshot_path=RERUN_SCREENSHOT_PATH,
        cli_path=EXPECTED_RERUN_CLI,
        expected_version=EXPECTED_RERUN_VERSION,
    )
    observed_non_system = validation.get("entity_path_contract", {}).get(
        "observed_non_system", []
    )
    component_rows = validation.get("component_contract", {}).get("checks", {})
    exact_components = len(component_rows) == 296 and all(
        row.get("pass") is True and row.get("observed") == row.get("required")
        for row in component_rows.values()
    )
    screenshot_decode = (
        _decode_rerun_png(RERUN_SCREENSHOT_PATH)
        if RERUN_SCREENSHOT_PATH.is_file()
        else {"pass": False}
    )
    checks = {
        "attempt1_rerun_false_preserved": original.get("pass") is False,
        "original_failure_only_part_idx": original_archive.get("errors")
        == ["RRD timeline contract failed"]
        and original_timeline.get("observed")
        == [name for name in ORIGINAL_TIMELINES if name != "part_idx"]
        and original_timeline.get("checks", {}).get("part_idx") is False,
        "original_contract_digest_exact": _original_contract_digest()
        == ORIGINAL_RERUN_CONTRACT_SHA256,
        "witness_manifest_pass": witness_manifest["pass"],
        "structural_delta_pass": structural["pass"],
        "validation_pass": validation.get("pass") is True,
        "exact_entities_296": len(observed_non_system) == 296
        and validation.get("entity_path_contract", {}).get("pass") is True,
        "part_entity_paths_128_exact": all(
            f"/geometry/live_parts/{body}/part_{index:03d}" in observed_non_system
            for body in ("link5", "gripper_link")
            for index in range(64)
        ),
        "exact_components_296": exact_components,
        "exact_original_five_timelines": validation.get("timeline_contract", {}).get(
            "observed"
        )
        == ORIGINAL_TIMELINES
        and validation.get("timeline_contract", {}).get("pass") is True,
        "footer_verify_pass": validation.get("verify", {}).get("ok") is True,
        "rbl_verify_pass": validation.get("blueprint_verify", {}).get("ok") is True,
        "headless_render_hash_bound": validation.get("headless_render", {}).get("ok")
        is True
        and RERUN_SCREENSHOT_PATH.is_file()
        and validation.get("headless_render", {}).get("sha256")
        == _sha(RERUN_SCREENSHOT_PATH),
        "screenshot_exact_decode": screenshot_decode.get("pass") is True
        and _png_dimensions(RERUN_SCREENSHOT_PATH)
        == EXPECTED_RERUN_RASTER_DIMENSIONS,
        "original_rrd_rbl_still_exact": _attempt1_hashes_exact(),
    }
    return {
        "artifact": "D350_RERUN_ORIGINAL_CONTRACT_REVALIDATION_V1",
        "case": CASE,
        "original_rrd_path": _rel(d350.RRD_PATH),
        "original_rrd_sha256": _sha(d350.RRD_PATH),
        "repaired_rrd_path": _rel(REPAIRED_RRD_PATH),
        "repaired_rrd_sha256": _sha(REPAIRED_RRD_PATH),
        "repaired_rbl_path": _rel(REPAIRED_RBL_PATH),
        "repaired_rbl_sha256": _sha(REPAIRED_RBL_PATH),
        "witness_path": _rel(PART_IDX_WITNESS_PATH),
        "witness_sha256": _sha(PART_IDX_WITNESS_PATH),
        "witness_manifest_path": _rel(PART_IDX_MANIFEST_PATH),
        "witness_manifest_sha256": _sha(PART_IDX_MANIFEST_PATH),
        "original_contract_sha256": _original_contract_digest(),
        "original_exact_timelines": ORIGINAL_TIMELINES,
        "validation": validation,
        "structural_delta": structural,
        "screenshot_decode": screenshot_decode,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run_repair(args: argparse.Namespace) -> int:
    prereg = _json(PREREG_PATH)
    runtime_environment = _runtime_environment_contract()
    runtime_outputs = [
        CAPTURE_REVALIDATION_PATH,
        RERUN_REVALIDATION_PATH,
        PART_IDX_WITNESS_PATH,
        PART_IDX_MANIFEST_PATH,
        REPAIRED_RRD_PATH,
        REPAIRED_RBL_PATH,
        RERUN_SCREENSHOT_PATH,
        REPAIR_SUMMARY_PATH,
        REPAIR_REPORT_PATH,
        MANUAL_PATH,
        MANUAL_MD_PATH,
        COMPLETION_PATH,
        COMPLETION_MD_PATH,
    ]
    prechecks = {
        "prereg_pass": prereg.get("pass") is True,
        "fresh_process_pid": prereg["repair_process_identity"]["pid"] != os.getpid(),
        "fresh_process_nonce": prereg["repair_process_identity"]["nonce"]
        != args.process_nonce,
        "head_exact": _git_head() == prereg.get("git_head") == EXPECTED_HEAD,
        "state_hashes_exact": prereg.get("state_hashes")
        == {"start_here": _sha(START_HERE), "repair_session": _sha(REPAIR_SESSION)},
        "harness_hash_exact": prereg.get("harness_sha256") == _sha(HARNESS),
        "original_harness_exact": prereg.get("original_harness_sha256")
        == _sha(ORIGINAL_HARNESS)
        == EXPECTED_ORIGINAL_HARNESS_SHA256,
        "attempt1_manifest_exact": prereg.get("attempt1_manifest_sha256")
        == _sha(ATTEMPT1_MANIFEST_PATH),
        "attempt1_hashes_exact": prereg.get("attempt1_hashes")
        == _attempt1_hashes()
        == EXPECTED_ATTEMPT1_HASHES,
        "attempt1_inventory_exact": _attempt1_top_level_inventory(
            repair_dir_expected=True
        )["pass"],
        "repair_dir_prereg_only": sorted(path.name for path in REPAIR_DIR.iterdir())
        == sorted([ATTEMPT1_MANIFEST_PATH.name, PREREG_PATH.name]),
        "original_contract_exact": prereg.get("original_rerun_contract_sha256")
        == _original_contract_digest()
        == ORIGINAL_RERUN_CONTRACT_SHA256,
        "original_timelines_exact": prereg.get("original_exact_timelines")
        == ORIGINAL_TIMELINES,
        "part_mapping_exact": prereg.get("part_idx_mapping_sha256")
        == _part_mapping_digest()
        == PART_IDX_MAPPING_SHA256,
        "attempt1_zero_scope_pass": _attempt1_zero_scope_contract()["pass"],
        "status_scope": _status_scope_pass(_git_status()),
        "external_user_files_unchanged": d350._external_baseline()["pass"],
        "runtime_outputs_absent": all(not path.exists() for path in runtime_outputs),
        "no_isaac_modules_loaded": _isaac_modules_loaded() == [],
        "runtime_environment_exact": runtime_environment["pass"]
        and prereg.get("runtime_environment") == runtime_environment,
    }
    if not all(prechecks.values()):
        raise RuntimeError(f"D350 repair precheck failed: {prechecks}")
    hashes_before = _attempt1_hashes()
    capture = _capture_revalidation()
    _write_json(CAPTURE_REVALIDATION_PATH, capture)
    rerun = _rerun_revalidation()
    _write_json(RERUN_REVALIDATION_PATH, rerun)
    hashes_after = _attempt1_hashes()
    automated = _json(d350.AUTOMATED_PATH)
    immutable = automated["immutability"]
    aggregation_checks = {
        "source_inventories_exact_true": immutable.get("source_inventories_exact") is True,
        "input_hashes_exact_true": immutable.get("input_hashes_exact") is True,
        "external_user_files_exact_true": immutable.get("external_user_files_exact") is True,
        "asset_write_is_false": immutable.get("asset_write") is False,
        "original_false_aggregate_preserved": immutable.get("pass") is False,
    }
    science = _attempt1_science_contract()
    failure = _attempt1_failure_contract()
    zero_scope = _attempt1_zero_scope_contract()
    scope_checks = {
        "preregistered_scope_exact": prereg.get("scope_guards")
        == {
            "isaac_launch": False,
            "physics_steps": 0,
            "target_write": False,
            "geometry_remeasurement": False,
            "attempt1_overwrite": False,
            "asset_write": False,
            "settle": False,
            "ten_trial": False,
            "g0b": False,
            "rl_or_ppo": False,
            "ladder_promotion": False,
            "g0a_pass": False,
        },
        "attempt1_zero_scope_recomputed": zero_scope["pass"],
        "no_isaac_modules_before": prechecks["no_isaac_modules_loaded"],
        "no_isaac_modules_after": _isaac_modules_loaded() == [],
        "attempt1_hashes_unchanged": hashes_before
        == hashes_after
        == EXPECTED_ATTEMPT1_HASHES,
        "source_inputs_unchanged": d350._input_hashes() == d350.EXPECTED_INPUT_HASHES,
        "external_user_files_unchanged": d350._external_baseline()["pass"],
        "new_variables_zero": NEW_VARIABLES == [] and NEW_PHYSICAL_VARIABLES == [],
        "offline_rerun_only": rerun.get("pass") is True
        and rerun.get("original_rrd_sha256") == EXPECTED_ATTEMPT1_HASHES[
            "d350_fixed_jaw_geometry.rrd"
        ],
    }
    scope_evidence = {
        "method": (
            "offline Pillow reads plus Rerun metadata witness/merge/validation; "
            "no Isaac import, launch, target write, or geometry function"
        ),
        "isaac_modules_before": [],
        "isaac_modules_after": _isaac_modules_loaded(),
        "attempt1_zero_scope": zero_scope,
        "checks": scope_checks,
        "pass": all(scope_checks.values()),
    }
    checks = {
        "prechecks": all(prechecks.values()),
        "attempt1_hashes_unchanged": hashes_before
        == hashes_after
        == EXPECTED_ATTEMPT1_HASHES,
        "science_contract_pass": science["pass"],
        "observed_failure_contract_pass": failure["pass"],
        "capture_revalidation_pass": capture["pass"],
        "rerun_revalidation_pass": rerun["pass"],
        "corrected_immutability_aggregate_pass": all(aggregation_checks.values()),
        "new_variables_zero": NEW_VARIABLES == [] and NEW_PHYSICAL_VARIABLES == [],
        "physics_and_promotion_zero": scope_evidence["pass"],
    }
    summary = {
        "artifact": "D350_OBSERVABILITY_REPAIR_SUMMARY_V1",
        "case": CASE,
        "attempt": "attempt2_observability_repair",
        "repair_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "scientific_result": "D350_FROZEN_FIXED_JAW_GEOMETRY_MEASURED",
        "observability_result": (
            "D350_VIEWER_AND_RERUN_REACTIVE_REPAIR_PASS_MANUAL_PENDING"
            if all(checks.values())
            else d350.VERDICT_VISUAL
        ),
        "repair_pass": all(checks.values()),
        "attempt1_science": science,
        "attempt1_failure": failure,
        "capture_revalidation": {
            "path": _rel(CAPTURE_REVALIDATION_PATH),
            "sha256": _sha(CAPTURE_REVALIDATION_PATH),
            "pass": capture["pass"],
        },
        "rerun_revalidation": {
            "path": _rel(RERUN_REVALIDATION_PATH),
            "sha256": _sha(RERUN_REVALIDATION_PATH),
            "screenshot_path": _rel(RERUN_SCREENSHOT_PATH),
            "screenshot_sha256": _sha(RERUN_SCREENSHOT_PATH)
            if RERUN_SCREENSHOT_PATH.is_file()
            else None,
            "pass": rerun["pass"],
            "repaired_rrd_path": _rel(REPAIRED_RRD_PATH),
            "repaired_rrd_sha256": _sha(REPAIRED_RRD_PATH),
            "repaired_rbl_path": _rel(REPAIRED_RBL_PATH),
            "repaired_rbl_sha256": _sha(REPAIRED_RBL_PATH),
            "witness_path": _rel(PART_IDX_WITNESS_PATH),
            "witness_sha256": _sha(PART_IDX_WITNESS_PATH),
            "witness_manifest_path": _rel(PART_IDX_MANIFEST_PATH),
            "witness_manifest_sha256": _sha(PART_IDX_MANIFEST_PATH),
        },
        "corrected_immutability_aggregation": {
            "checks": aggregation_checks,
            "pass": all(aggregation_checks.values()),
        },
        "attempt1_hashes_before": hashes_before,
        "attempt1_hashes_after": hashes_after,
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "scope_evidence": scope_evidence,
        "scope": {
            "isaac_launches": 0,
            "physics_steps": 0,
            "target_writes": 0,
            "geometry_remeasurements": 0,
            "attempt1_overwrites": 0,
            "g0a_pass": False,
            "settle": False,
            "ten_trial": False,
            "g0b": False,
            "rl_or_ppo": False,
            "ladder_promotion": False,
        },
        "prechecks": prechecks,
        "checks": checks,
        "manual_visual_inspection_pending": True,
    }
    _write_json(REPAIR_SUMMARY_PATH, summary)
    _write_text(
        REPAIR_REPORT_PATH,
        "# D350 reactive observability repair\n\n"
        f"- repair pass: `{summary['repair_pass']}`\n"
        f"- scientific result: `{summary['scientific_result']}`\n"
        f"- observability: `{summary['observability_result']}`\n"
        f"- capture post-close revalidation: `{capture['pass']}`\n"
        f"- Rerun original 5-timeline contract validation: `{rerun['pass']}`\n"
        "- immutable Mesh3D 130 retained; per-mesh metadata part_idx 0..129 added\n"
        "- Isaac launches / physics steps / target writes: `0 / 0 / 0`\n"
        "- attempt1 files were not overwritten\n",
    )
    print(json.dumps({"stage": "repair", "pass": summary["repair_pass"]}, sort_keys=True))
    return 0 if summary["repair_pass"] else 2


def _manual_checks(manual: dict[str, Any]) -> dict[str, bool]:
    expected_images = {**d350.VIEWER_PNGS, "rerun": RERUN_SCREENSHOT_PATH}
    images = manual.get("images", {})
    image_checks = {}
    for name, path in expected_images.items():
        row = images.get(name, {})
        decoder = _decode_rerun_png(path) if name == "rerun" else _decode_viewer_png(path)
        image_checks[f"{name}_path"] = row.get("path") == _rel(path)
        image_checks[f"{name}_sha"] = path.is_file() and row.get("sha256") == _sha(path)
        image_checks[f"{name}_bytes"] = path.is_file() and row.get("bytes") == path.stat().st_size
        image_checks[f"{name}_dimensions"] = row.get("raster_dimensions") == _png_dimensions(path)
        image_checks[f"{name}_decoder"] = decoder.get("pass") is True
    observations = manual.get("observations", {})
    required = (
        "actual_isaac_viewer_was_observed_live",
        "physx_whole_view_shows_robot_tool_and_target_colliders",
        "physx_close_view_shows_jaw_and_target_collider_wire",
        "colored_whole_view_is_nonblank_and_assembled",
        "colored_top_view_is_nonblank_and_pose_consistent",
        "colored_side_view_is_nonblank_and_pose_consistent",
        "colored_oblique_view_is_nonblank_and_pose_consistent",
        "two_colored_body_families_and_target_are_visible",
        "fixed_jaw_surface_region_is_visible",
        "rerun_128_part_entities_and_target_are_visible",
        "rerun_frames_arrows_and_metrics_are_visible",
        "no_blank_corrupt_swapped_or_decision_obscuring_image",
        "exact_64_plus_64_count_attributed_to_machine_contract_not_eyeballing",
        "all_seven_pngs_opened_at_original_resolution",
    )
    method = str(manual.get("inspection_method", "")).lower()
    checks = {
        "artifact": manual.get("artifact")
        == "D350_REACTIVE_REPAIR_MANUAL_VISUAL_INSPECTION_V1",
        "case": manual.get("case") == CASE,
        "attempt": manual.get("attempt") == "attempt2_observability_repair",
        "inspection_date": manual.get("inspection_date") == "2026-07-14",
        "inspection_method": "view_image" in method and "original" in method,
        "exact_seven_image_rows": set(images) == set(expected_images),
        "observations": all(observations.get(name) is True for name in required),
        "bounded_interpretation": len(manual.get("bounded_interpretation", [])) >= 4,
        "manual_pass": manual.get("manual_visual_inspection_pass") is True,
        "scientific_override_false": manual.get("scientific_verdict_override") is False,
        "g0a_false": manual.get("g0a_pass") is False,
        "settle_false": manual.get("settle_executed") is False,
        "markdown_nonzero": MANUAL_MD_PATH.is_file() and MANUAL_MD_PATH.stat().st_size > 0,
        **image_checks,
    }
    return checks


def _capture_finalize_contract(
    capture: dict[str, Any], repair: dict[str, Any]
) -> dict[str, Any]:
    rows = capture.get("rows", {})
    current = _sample_pngs()
    expected_check_keys = {
        "attempt1_capture_contract_false_preserved",
        "expected_six_paths_exact",
        "capture_tokens_six_true",
        "immediate_and_delayed_sets_exact",
        "preflight_runtime_outputs_absent",
        "attempt1_process_absent_before_sampling",
        "attempt1_process_absent_after_sampling",
        "all_six_stable_decodable_exact",
        "six_hashes_pairwise_distinct",
        "production_decoder_negative_controls",
        "zero_step_target_state_contract",
    }
    expected_negative_keys = {
        "truncated_rejected_by_production_decoder",
        "valid_wrong_mode_rejected_by_production_decoder",
        "valid_wrong_size_rejected_by_production_decoder",
    }
    recomputed_negative_controls = _png_decoder_negative_controls()
    checks = {
        "artifact_exact": capture.get("artifact")
        == "D350_VIEWER_CAPTURE_POSTCLOSE_REVALIDATION_V1",
        "case_exact": capture.get("case") == CASE,
        "report_pass": capture.get("pass") is True
        and set(capture.get("checks", {})) == expected_check_keys
        and all(capture.get("checks", {}).values()),
        "summary_binding_exact": repair.get("capture_revalidation")
        == {
            "path": _rel(CAPTURE_REVALIDATION_PATH),
            "sha256": _sha(CAPTURE_REVALIDATION_PATH),
            "pass": True,
        },
        "six_rows_exact": set(rows) == set(d350.VIEWER_PNGS)
        and all(row.get("pass") is True for row in rows.values()),
        "current_files_match_post_decode_sample": all(
            rows[name].get("post_decode_sample") == current[name]
            for name in d350.VIEWER_PNGS
        ),
        "all_current_decoders_pass": all(
            _decode_viewer_png(path).get("pass") is True
            for path in d350.VIEWER_PNGS.values()
        ),
        "negative_controls_pass": set(capture.get("negative_controls", {}))
        == expected_negative_keys
        and capture.get("negative_controls") == recomputed_negative_controls
        and all(recomputed_negative_controls.values()),
        "original_capture_hash_exact": capture.get("attempt1_capture_contract_sha256")
        == _sha(d350.CAPTURE_PATH)
        == EXPECTED_ATTEMPT1_HASHES["d350_viewer_capture_contract.json"],
    }
    return {"checks": checks, "pass": all(checks.values())}


def _rerun_finalize_contract(
    rerun: dict[str, Any], repair: dict[str, Any]
) -> dict[str, Any]:
    entities, components = d350._expected_rrd_contract()
    current_validation = validate_rerun_artifact(
        REPAIRED_RRD_PATH,
        expected_entity_paths=[
            "geometry/live_parts/link5/part_000",
            "geometry/live_parts/gripper_link/part_063",
            "geometry/fixed_jaw/raw_connected_component",
            "geometry/target/cylinder_collider",
            "geometry/arrows/fixed_jaw_centerline",
        ],
        expected_timeline_names=["part_idx", "measurement_idx", "event_idx"],
        exact_entity_paths=entities,
        exact_timeline_names=ORIGINAL_TIMELINES,
        expected_entity_components=components,
        blueprint_path=REPAIRED_RBL_PATH,
        cli_path=EXPECTED_RERUN_CLI,
        expected_version=EXPECTED_RERUN_VERSION,
    )
    validation = rerun.get("validation", {})
    entity = validation.get("entity_path_contract", {})
    timeline = validation.get("timeline_contract", {})
    component_rows = validation.get("component_contract", {}).get("checks", {})
    current_component_rows = current_validation.get("component_contract", {}).get(
        "checks", {}
    )
    structural_now = _rrd_structural_delta()
    manifest = _json(PART_IDX_MANIFEST_PATH)
    summary_row = repair.get("rerun_revalidation", {})
    current_binding = {
        "path": _rel(RERUN_REVALIDATION_PATH),
        "sha256": _sha(RERUN_REVALIDATION_PATH),
        "screenshot_path": _rel(RERUN_SCREENSHOT_PATH),
        "screenshot_sha256": _sha(RERUN_SCREENSHOT_PATH),
        "pass": True,
        "repaired_rrd_path": _rel(REPAIRED_RRD_PATH),
        "repaired_rrd_sha256": _sha(REPAIRED_RRD_PATH),
        "repaired_rbl_path": _rel(REPAIRED_RBL_PATH),
        "repaired_rbl_sha256": _sha(REPAIRED_RBL_PATH),
        "witness_path": _rel(PART_IDX_WITNESS_PATH),
        "witness_sha256": _sha(PART_IDX_WITNESS_PATH),
        "witness_manifest_path": _rel(PART_IDX_MANIFEST_PATH),
        "witness_manifest_sha256": _sha(PART_IDX_MANIFEST_PATH),
    }
    checks = {
        "artifact_exact": rerun.get("artifact")
        == "D350_RERUN_ORIGINAL_CONTRACT_REVALIDATION_V1",
        "case_exact": rerun.get("case") == CASE,
        "report_pass": rerun.get("pass") is True
        and all(rerun.get("checks", {}).values()),
        "current_archive_direct_revalidation": current_validation.get("pass") is True
        and current_validation.get("entity_path_contract", {}).get("pass") is True
        and len(
            current_validation.get("entity_path_contract", {}).get(
                "observed_non_system", []
            )
        )
        == 296
        and current_validation.get("timeline_contract", {}).get("observed")
        == ORIGINAL_TIMELINES
        and current_validation.get("timeline_contract", {}).get("exact_match") is True
        and current_validation.get("component_contract", {}).get("pass") is True
        and len(current_component_rows) == 296
        and all(
            row.get("pass") is True and row.get("observed") == row.get("required")
            for row in current_component_rows.values()
        )
        and current_validation.get("version", {}).get("expected_version_match") is True,
        "summary_binding_exact": summary_row == current_binding,
        "report_artifact_hashes_exact": all(
            [
                rerun.get("original_rrd_sha256") == _sha(d350.RRD_PATH),
                rerun.get("repaired_rrd_sha256") == _sha(REPAIRED_RRD_PATH),
                rerun.get("repaired_rbl_sha256") == _sha(REPAIRED_RBL_PATH),
                rerun.get("witness_sha256") == _sha(PART_IDX_WITNESS_PATH),
                rerun.get("witness_manifest_sha256") == _sha(PART_IDX_MANIFEST_PATH),
            ]
        ),
        "original_contract_exact": rerun.get("original_contract_sha256")
        == _original_contract_digest()
        == ORIGINAL_RERUN_CONTRACT_SHA256,
        "exact_entities_296": entity.get("pass") is True
        and entity.get("exact_non_system_match") is True
        and len(entity.get("observed_non_system", [])) == 296,
        "exact_components_296": validation.get("component_contract", {}).get("pass")
        is True
        and len(component_rows) == 296
        and all(
            row.get("pass") is True and row.get("observed") == row.get("required")
            for row in component_rows.values()
        ),
        "exact_original_five_timelines": timeline.get("pass") is True
        and timeline.get("exact_match") is True
        and timeline.get("observed") == ORIGINAL_TIMELINES,
        "footer_and_rbl_verified": validation.get("verify", {}).get("ok") is True
        and validation.get("blueprint_verify", {}).get("ok") is True
        and _verify_rrd(REPAIRED_RRD_PATH).get("ok") is True
        and _verify_rrd(REPAIRED_RBL_PATH).get("ok") is True
        and _verify_rrd(PART_IDX_WITNESS_PATH).get("ok") is True,
        "headless_screenshot_exact": validation.get("headless_render", {}).get("ok")
        is True
        and validation.get("headless_render", {}).get("sha256")
        == _sha(RERUN_SCREENSHOT_PATH)
        and _decode_rerun_png(RERUN_SCREENSHOT_PATH).get("pass") is True
        and _png_dimensions(RERUN_SCREENSHOT_PATH)
        == EXPECTED_RERUN_RASTER_DIMENSIONS,
        "structural_delta_recomputed": structural_now.get("pass") is True
        and structural_now.get("checks")
        == rerun.get("structural_delta", {}).get("checks"),
        "witness_manifest_exact": manifest.get("artifact")
        == "D350_PART_IDX_TEMPORAL_WITNESS_MANIFEST_V1"
        and manifest.get("case") == CASE
        and manifest.get("pass") is True
        and manifest.get("part_idx_mapping_sha256")
        == _part_mapping_digest()
        == PART_IDX_MAPPING_SHA256,
        "rbl_byte_exact_forward_copy": _sha(REPAIRED_RBL_PATH) == _sha(d350.RBL_PATH),
    }
    return {
        "current_archive_direct_revalidation": current_validation,
        "structural_delta_recomputed": structural_now,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run_finalize(_args: argparse.Namespace) -> int:
    if COMPLETION_PATH.exists() or COMPLETION_MD_PATH.exists():
        raise RuntimeError("D350 repair completion already exists")
    prereg = _json(PREREG_PATH)
    repair = _json(REPAIR_SUMMARY_PATH)
    capture = _json(CAPTURE_REVALIDATION_PATH)
    rerun = _json(RERUN_REVALIDATION_PATH)
    manual = _json(MANUAL_PATH)
    manual_checks = _manual_checks(manual)
    capture_contract = _capture_finalize_contract(capture, repair)
    rerun_contract = _rerun_finalize_contract(rerun, repair)
    science_now = _attempt1_science_contract()
    failure_now = _attempt1_failure_contract()
    zero_scope_now = _attempt1_zero_scope_contract()
    runtime_environment_now = _runtime_environment_contract()
    original_before = prereg["attempt1_hashes"]
    original_now = _attempt1_hashes()
    required_repair = [
        ATTEMPT1_MANIFEST_PATH,
        PREREG_PATH,
        CAPTURE_REVALIDATION_PATH,
        RERUN_REVALIDATION_PATH,
        PART_IDX_WITNESS_PATH,
        PART_IDX_MANIFEST_PATH,
        REPAIRED_RRD_PATH,
        REPAIRED_RBL_PATH,
        RERUN_SCREENSHOT_PATH,
        REPAIR_SUMMARY_PATH,
        REPAIR_REPORT_PATH,
        MANUAL_PATH,
        MANUAL_MD_PATH,
    ]
    expected_repair_check_keys = {
        "prechecks",
        "attempt1_hashes_unchanged",
        "science_contract_pass",
        "observed_failure_contract_pass",
        "capture_revalidation_pass",
        "rerun_revalidation_pass",
        "corrected_immutability_aggregate_pass",
        "new_variables_zero",
        "physics_and_promotion_zero",
    }
    checks = {
        "prereg_pass": prereg.get("pass") is True,
        "repair_pass_and_all_checks": repair.get("repair_pass") is True
        and set(repair.get("checks", {})) == expected_repair_check_keys
        and all(repair.get("checks", {}).values()),
        "capture_contract_recomputed": capture_contract["pass"],
        "rerun_contract_recomputed": rerun_contract["pass"],
        "manual_contract_pass": all(manual_checks.values()),
        "head_exact": _git_head() == prereg.get("git_head") == EXPECTED_HEAD,
        "state_hashes_exact": prereg.get("state_hashes")
        == {"start_here": _sha(START_HERE), "repair_session": _sha(REPAIR_SESSION)},
        "repair_harness_exact": prereg.get("harness_sha256") == _sha(HARNESS),
        "original_harness_exact": prereg.get("original_harness_sha256")
        == _sha(ORIGINAL_HARNESS)
        == EXPECTED_ORIGINAL_HARNESS_SHA256,
        "attempt1_hashes_exact": original_before
        == original_now
        == EXPECTED_ATTEMPT1_HASHES,
        "attempt1_manifest_exact": prereg.get("attempt1_manifest_sha256")
        == _sha(ATTEMPT1_MANIFEST_PATH)
        and _json(ATTEMPT1_MANIFEST_PATH).get("pass") is True,
        "attempt1_inventory_exact": _attempt1_top_level_inventory(
            repair_dir_expected=True
        )["pass"],
        "all_repair_artifacts_present": all(path.is_file() for path in required_repair),
        "repair_directory_exact_before_completion": sorted(
            path.name for path in REPAIR_DIR.iterdir()
        )
        == sorted(path.name for path in required_repair),
        "status_scope": _status_scope_pass(_git_status()),
        "external_user_files_unchanged": d350._external_baseline()["pass"],
        "original_inputs_exact": d350._input_hashes() == d350.EXPECTED_INPUT_HASHES,
        "science_still_measured": science_now["pass"]
        and repair.get("scientific_result")
        == "D350_FROZEN_FIXED_JAW_GEOMETRY_MEASURED"
        and repair.get("attempt1_science", {}).get("metrics")
        == science_now.get("metrics"),
        "attempt1_failure_still_preserved": failure_now["pass"]
        and repair.get("attempt1_failure", {}).get("pass") is True,
        "scope_evidence_recomputed": zero_scope_now["pass"]
        and repair.get("scope_evidence", {}).get("pass") is True
        and all(repair.get("scope_evidence", {}).get("checks", {}).values())
        and _isaac_modules_loaded() == []
        and NEW_VARIABLES == []
        and NEW_PHYSICAL_VARIABLES == [],
        "runtime_environment_exact": runtime_environment_now["pass"]
        and prereg.get("runtime_environment") == runtime_environment_now,
        "original_contract_and_mapping_unchanged": prereg.get(
            "original_rerun_contract_sha256"
        )
        == _original_contract_digest()
        == ORIGINAL_RERUN_CONTRACT_SHA256
        and prereg.get("part_idx_mapping_sha256")
        == _part_mapping_digest()
        == PART_IDX_MAPPING_SHA256,
        "scope_zero_values": repair.get("scope")
        == {
            "isaac_launches": 0,
            "physics_steps": 0,
            "target_writes": 0,
            "geometry_remeasurements": 0,
            "attempt1_overwrites": 0,
            "g0a_pass": False,
            "settle": False,
            "ten_trial": False,
            "g0b": False,
            "rl_or_ppo": False,
            "ladder_promotion": False,
        },
    }
    final_verdict = (
        d350.VERDICT_COMPLETE if all(checks.values()) else d350.VERDICT_VISUAL
    )
    artifact_paths = {
        **{f"attempt1/{name}": ATTEMPT1_DIR / name for name in EXPECTED_ATTEMPT1_HASHES},
        **{f"repair/{path.name}": path for path in required_repair},
    }
    artifacts = {
        name: {
            "path": _rel(path),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
            "png_dimensions": _png_dimensions(path),
        }
        for name, path in sorted(artifact_paths.items())
        if path.is_file()
    }
    completion = {
        "artifact": "D350_REACTIVE_REPAIR_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "final_verdict": final_verdict,
        "completion_pass": final_verdict == d350.VERDICT_COMPLETE and all(checks.values()),
        "attempt1_scientific_result": repair.get("scientific_result"),
        "attempt1_observability_failure_preserved": True,
        "attempt2_observability_repair": repair.get("observability_result"),
        "measurement_metrics": repair.get("attempt1_science", {}).get("metrics", {}),
        "aligned_pass": None,
        "g0a_pass": False,
        "settle_authorized": False,
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "checks": checks,
        "capture_finalize_contract": capture_contract,
        "rerun_finalize_contract": rerun_contract,
        "manual_checks": manual_checks,
        "artifacts": artifacts,
        "commit_or_push_performed": False,
    }
    _write_json(COMPLETION_PATH, completion)
    _write_text(
        COMPLETION_MD_PATH,
        "# D350 completion after reactive observability repair\n\n"
        f"- final verdict: `{final_verdict}`\n"
        f"- completion pass: `{completion['completion_pass']}`\n"
        "- scientific vocabulary: `MEASURED`, never `ALIGNED_PASS`\n"
        "- attempt1 observability FAIL preserved; attempt2 repair is separate\n"
        "- original five-timeline Rerun contract retained and satisfied\n"
        "- static Mesh3D 130 retained; exact metadata part_idx 0..129 added\n"
        "- Isaac launches / physics steps in repair: `0 / 0`\n"
        "- `g0a_pass=false`; settle/G0b/RL/ladder remain blocked\n",
    )
    print(
        json.dumps(
            {"stage": "finalize", "pass": completion["completion_pass"], "verdict": final_verdict},
            sort_keys=True,
        )
    )
    return 0 if completion["completion_pass"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "repair", "finalize"), required=True)
    parser.add_argument("--out_dir", type=Path, default=REPAIR_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if Path(args.out_dir).resolve() != REPAIR_DIR.resolve():
        raise RuntimeError("D350 repair output path is fixed and forward-only")
    args.process_nonce = secrets.token_hex(16)
    if args.stage == "prepare":
        return _run_prepare(args)
    if args.stage == "repair":
        return _run_repair(args)
    return _run_finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
