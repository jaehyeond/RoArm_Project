#!/usr/bin/env python3
"""D370: resume D369's frozen presentation after proving direct-script import bootstrap.

This is observability-only.  Domain/display inputs are immutable D368 evidence plus the eight
frozen D369 partial artifacts.  No geometry is decoded or generated and no Isaac/PhysX module is
imported.  The sole new variable is the production-command repo-root import preflight.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve()
BOOTSTRAP_ENV = "D370_REPO_ROOT_BOOTSTRAP"
BOOTSTRAP_MODE = os.environ.get(BOOTSTRAP_ENV, "enabled")


def _normalized_sys_path(entry: str) -> Path:
    return Path(entry or os.getcwd()).resolve()


if BOOTSTRAP_MODE == "enabled":
    sys.path[:] = [entry for entry in sys.path if _normalized_sys_path(entry) != REPO]
    sys.path.insert(0, str(REPO))
elif BOOTSTRAP_MODE == "disabled":
    sys.path[:] = [entry for entry in sys.path if _normalized_sys_path(entry) != REPO]
else:
    raise RuntimeError(f"unsupported {BOOTSTRAP_ENV}={BOOTSTRAP_MODE!r}")

try:
    import roarm_rl.rerun_contract as rerun_contract
    from roarm_rl.rerun_contract import _entity_paths, _is_system_entity, validate_rerun_artifact
except ModuleNotFoundError as exc:
    if BOOTSTRAP_MODE == "disabled" and "import-preflight" in sys.argv:
        print(
            json.dumps(
                {
                    "artifact": "D370_IMPORT_PREFLIGHT_NEGATIVE_V1",
                    "bootstrap_mode": BOOTSTRAP_MODE,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "error_name": exc.name,
                    "repo_root": str(REPO),
                    "repo_root_occurrences": sum(
                        _normalized_sys_path(entry) == REPO for entry in sys.path
                    ),
                    "pass": False,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(86)
    raise


OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d370"
SOURCE_D368_EVIDENCE = (
    REPO / "claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_evidence.json"
)
SOURCE_D368_RRD = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation.rrd"
SOURCE_D369_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d369"
SOURCE_D369_PREREG = SOURCE_D369_DIR / "d369_preregistration.json"
SOURCE_D369_PHASE = SOURCE_D369_DIR / "d369_phase_markers.jsonl"
SOURCE_D369_EXCEPTION = SOURCE_D369_DIR / "d369_runtime_exception.json"
SOURCE_D369_BASE = SOURCE_D369_DIR / "d369_d368_base_bitexact_copy.rrd"
SOURCE_D369_RECORDING = SOURCE_D369_DIR / "d369_d368_recording_only_display_copy.rrd"
SOURCE_D369_OVERLAY = SOURCE_D369_DIR / "d369_static_text_overlay.rrd"
SOURCE_D369_RBL = SOURCE_D369_DIR / "d369_professor_visual_contract.rbl"
SOURCE_D369_PRESENTATION = SOURCE_D369_DIR / "d369_professor_visual_contract.rrd"
RERUN_CONTRACT_HELPER = REPO / "roarm_rl/rerun_contract.py"

PREREG_PATH = OUT_DIR / "d370_preregistration.json"
IMPORT_ATTESTATION_PATH = OUT_DIR / "d370_import_preflight_attestation.json"
PHASE_PATH = OUT_DIR / "d370_phase_markers.jsonl"
PRESENTATION_COPY_PATH = OUT_DIR / "d370_d369_presentation_bitexact.rrd"
RBL_COPY_PATH = OUT_DIR / "d370_d369_blueprint_bitexact.rbl"
INVOCATION_PATH = OUT_DIR / "d370_render_invocation.json"
RECEIPT_PATH = OUT_DIR / "d370_render_receipt.json"
VALIDATION_PATH = OUT_DIR / "d370_rerun_validation.json"
RERUN_PNG_PATH = OUT_DIR / "d370_professor_visual_contract_rerun.png"
BOARD_PNG_PATH = OUT_DIR / "d370_professor_board_1920x1080.png"
AUTOMATED_PATH = OUT_DIR / "d370_automated_summary.json"
REPORT_PATH = OUT_DIR / "d370_automated_report.md"
MANUAL_JSON_PATH = OUT_DIR / "d370_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d370_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d370_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d370_runtime_exception.json"

EXPECTED_HEAD = "888b92b4dfdb41e56d94fdffe4c0cb4d6e303297"
PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
EXPECTED_RERUN_VERSION = "0.34.1"
WINDOW_SIZE = "1920x1080"
RENDER_TIMEOUT_SECONDS = 180.0
HOST_RENDER_ENV = "D370_HOST_RENDER_APPROVED"
APP_ID = "roarm_g0a_d368_semantic_allocation"
EXPECTED_BLUEPRINT_ID = "rec_4465cbb5f772481cb927929933b2b35a"

NEW_VARIABLES = ["production_command_repo_root_import_preflight"]
INHERITED_DISPLAY_VARIABLES = [
    "timeline_free_static_metric_overlay",
    "label_suppressed_professor_layout",
]

EXPECTED_SOURCE_MANIFEST = {
    "claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_evidence.json": {
        "bytes": 953696,
        "sha256": "be2a422b0c74e4781b76a640c5312070b84876b1cb9e661d47e705ccdf789cf5",
    },
    "claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation.rrd": {
        "bytes": 1339534,
        "sha256": "f66a9fe41c625e3460b341eef2bfb0e107fbccdca4bf012c28b77e694efb5af0",
    },
    "claudedocs/runtime_logs/grasp_track/g0a_d369/d369_preregistration.json": {
        "bytes": 73385,
        "sha256": "f991aaeefd88d5066773b573d2a540b8c7d5658ef4252a792d9f422378338642",
    },
    "claudedocs/runtime_logs/grasp_track/g0a_d369/d369_phase_markers.jsonl": {
        "bytes": 860,
        "sha256": "5ddae9f25f33fab446954e197b478af937fd3ded458cdf59aa8526ec01457efa",
    },
    "claudedocs/runtime_logs/grasp_track/g0a_d369/d369_runtime_exception.json": {
        "bytes": 911,
        "sha256": "0b338694f77d34f910764ed99f5be6e61d1a554cd347c53be978d0496e47d1bb",
    },
    "claudedocs/runtime_logs/grasp_track/g0a_d369/d369_d368_base_bitexact_copy.rrd": {
        "bytes": 1339534,
        "sha256": "f66a9fe41c625e3460b341eef2bfb0e107fbccdca4bf012c28b77e694efb5af0",
    },
    "claudedocs/runtime_logs/grasp_track/g0a_d369/d369_d368_recording_only_display_copy.rrd": {
        "bytes": 1237161,
        "sha256": "ce00df2fbb95630e58439e9d7fd13afd56e27d3386581da8c71edac902f2403e",
    },
    "claudedocs/runtime_logs/grasp_track/g0a_d369/d369_static_text_overlay.rrd": {
        "bytes": 9229,
        "sha256": "1df88ad1a0aad052d3ba879e49aa10164cc878a8793fc61a07dea8dda79d9cc2",
    },
    "claudedocs/runtime_logs/grasp_track/g0a_d369/d369_professor_visual_contract.rbl": {
        "bytes": 105470,
        "sha256": "429407b11120167655c059085e8f3f4ef81191d49f4b5728dc20cfdfda45e216",
    },
    "claudedocs/runtime_logs/grasp_track/g0a_d369/d369_professor_visual_contract.rrd": {
        "bytes": 1347605,
        "sha256": "0f394dec88ad1d253d5c4e0996e80a01752b272b3a4e04ff0a0f0de439302aab",
    },
}

SOURCE_PATHS = [
    SOURCE_D368_EVIDENCE,
    SOURCE_D368_RRD,
    SOURCE_D369_PREREG,
    SOURCE_D369_PHASE,
    SOURCE_D369_EXCEPTION,
    SOURCE_D369_BASE,
    SOURCE_D369_RECORDING,
    SOURCE_D369_OVERLAY,
    SOURCE_D369_RBL,
    SOURCE_D369_PRESENTATION,
]

SCOPE_GUARDS = {
    "collider_regeneration_or_recook": 0,
    "isaac_or_kit_or_physx": 0,
    "simulation_app": 0,
    "q5_science_or_target": 0,
    "physics_steps": 0,
    "contact_queries": 0,
    "target_ik_path_changes": 0,
    "usd_or_asset_writes": 0,
    "material_mass_actuator_physics_changes": 0,
    "warp_or_cuda_compute": 0,
    "nvidia_smi": 0,
    "rerun_display_render_allowed": 1,
}

FORBIDDEN_MODULE_PREFIXES = (
    "omni",
    "isaacsim",
    "isaaclab",
    "pxr",
    "warp",
    "torch",
    "trimesh",
    "scipy",
)

ALLOWED_DIRTY_PREFIXES = (
    "START_HERE.md",
    "claudedocs/DECISIONS.md",
    "claudedocs/EXPERIMENT_LEDGER.md",
    "claudedocs/runtime_logs/grasp_track/g0a_d370/",
    "claudedocs/session_20260721_grasp_g0a_d370_",
    "sim_scripts/cyl34_top_view_d370_",
)

EXPECTED_D369_PHASES = [
    "render_started",
    "host_loopback_bind_capability_pass",
    "frozen_evidence_fields_copied",
    "d368_rrd_bitexact_copy_complete",
    "static_text_overlay_and_blueprint_finalized",
    "d368_recording_only_display_copy_finalized",
    "single_presentation_archive_finalized",
]

EXPECTED_PHASE_SEQUENCE = [
    "render_worker_started",
    "production_command_import_attestation_rechecked",
    "frozen_d369_manifest_verified",
    "frozen_d368_authority_verified",
    "active_rrd_rbl_bitexact_copies_written",
    "pre_render_artifact_contract_pass",
    "host_loopback_bind_capability_pass",
    "one_shot_viewer_invocation_recorded",
    "one_shot_viewer_returned",
    "raw_png_automated_gate_pass",
    "professor_board_written",
    "automated_validation_finalized",
    "manual_visual_inspection_contract_pass",
    "completion_gate_pass",
]

MANUAL_CHECK_KEYS = [
    "opened_both_pngs_original_resolution",
    "raw_four_named_spatial_views_nonblank_unclipped",
    "raw_two_text_cards_visible_and_legible",
    "no_unknown_timeline_or_empty_metric_panel",
    "no_in_scene_label_overlap",
    "no_error_toast_or_dialog",
    "raw_colors_match_patch_meaning",
    "board_exact_1920x1080",
    "board_geometry_not_distorted_or_clipped",
    "board_text_no_overlap_clip_or_ellipsis",
    "counts_and_inner_outer_relationship_exact",
    "five_null_g0a_false_core_runtime_scope_zero_visible",
    "d368_authority_and_d369_display_roles_visible",
    "d370_resume_lineage_hash_bound_outside_frozen_board_pixels",
]


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected JSON object row: {path}")
                rows.append(value)
    return rows


def _write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def _copy_x(source: Path, destination: Path) -> None:
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _run(
    command: list[str], *, timeout: float = 60.0, env: dict[str, str] | None = None
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_seconds": time.monotonic() - started,
            "ok": completed.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
            "elapsed_seconds": time.monotonic() - started,
            "ok": False,
        }


def _git(*args: str) -> str:
    result = _run(["git", *args])
    if not result["ok"]:
        raise RuntimeError(f"git {' '.join(args)} failed: {result['stderr']}")
    return str(result["stdout"]).strip()


def _git_status_paths() -> list[str]:
    result = _run(["git", "status", "--short"])
    if not result["ok"]:
        raise RuntimeError(f"git status failed: {result['stderr']}")
    paths: list[str] = []
    for line in str(result["stdout"]).splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def _dirty_scope_allowed(paths: list[str]) -> bool:
    return all(
        any(path == prefix or path.startswith(prefix) for prefix in ALLOWED_DIRTY_PREFIXES)
        for path in paths
    )


def _source_manifest() -> dict[str, Any]:
    return {_rel(path): {"bytes": path.stat().st_size, "sha256": _sha(path)} for path in SOURCE_PATHS}


def _dynamic_manifest() -> dict[str, str]:
    return {_rel(HARNESS): _sha(HARNESS), _rel(RERUN_CONTRACT_HELPER): _sha(RERUN_CONTRACT_HELPER)}


def _forbidden_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_MODULE_PREFIXES)
    )


def _phase(name: str, **fields: Any) -> None:
    rows = _read_jsonl(PHASE_PATH) if PHASE_PATH.exists() else []
    ordinal = len(rows) + 1
    if ordinal > len(EXPECTED_PHASE_SEQUENCE) or EXPECTED_PHASE_SEQUENCE[ordinal - 1] != name:
        raise RuntimeError(f"phase order mismatch at {ordinal}: {name}")
    row = {
        "ordinal": ordinal,
        "phase": name,
        "monotonic_seconds": time.monotonic(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _import_attestation() -> dict[str, Any]:
    helper_path = Path(rerun_contract.__file__).resolve()
    root_occurrences = sum(_normalized_sys_path(entry) == REPO for entry in sys.path)
    symbols = {
        "_entity_paths": callable(_entity_paths),
        "_is_system_entity": callable(_is_system_entity),
        "validate_rerun_artifact": callable(validate_rerun_artifact),
    }
    checks = {
        "bootstrap_mode_enabled": BOOTSTRAP_MODE == "enabled",
        "repo_root_present_exactly_once": root_occurrences == 1,
        "repo_root_is_sys_path_0": bool(sys.path)
        and _normalized_sys_path(sys.path[0]) == REPO,
        "ambient_pythonpath_absent": "PYTHONPATH" not in os.environ,
        "python_executable_exact": Path(sys.executable).resolve() == PYTHON.resolve(),
        "direct_script_exact": Path(sys.argv[0]).resolve() == HARNESS,
        "repo_cwd_exact": Path.cwd().resolve() == REPO,
        "bytecode_disabled": sys.dont_write_bytecode is True,
        "helper_path_exact": helper_path == RERUN_CONTRACT_HELPER,
        "helper_hash_exact": _sha(helper_path)
        == "aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e",
        "required_symbols_callable": all(symbols.values()),
        "contract_version_exact": rerun_contract.RERUN_CONTRACT_VERSION == EXPECTED_RERUN_VERSION,
    }
    return {
        "artifact": "D370_IMPORT_PREFLIGHT_V1",
        "bootstrap_mode": BOOTSTRAP_MODE,
        "repo_root": str(REPO),
        "repo_root_occurrences": root_occurrences,
        "sys_path_0": sys.path[0] if sys.path else None,
        "ambient_pythonpath_present": "PYTHONPATH" in os.environ,
        "python_executable": str(Path(sys.executable).resolve()),
        "script_path": str(Path(sys.argv[0]).resolve()),
        "cwd": str(Path.cwd().resolve()),
        "dont_write_bytecode": sys.dont_write_bytecode,
        "helper_path": str(helper_path),
        "helper_sha256": _sha(helper_path),
        "symbols": symbols,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _parse_single_json(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one JSON stdout line, got {len(lines)}")
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise TypeError("preflight stdout must be a JSON object")
    return value


def _production_preflight_commands() -> tuple[list[str], list[str]]:
    prefix = [str(PYTHON), "-B", str(HARNESS), "--stage"]
    return [*prefix, "import-preflight"], [*prefix, "render"]


def _run_production_import_preflight(mode: str) -> dict[str, Any]:
    command, _ = _production_preflight_commands()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop(HOST_RENDER_ENV, None)
    environment[BOOTSTRAP_ENV] = mode
    result = _run(command, timeout=30.0, env=environment)
    try:
        payload = _parse_single_json(str(result["stdout"]))
    except Exception as exc:
        payload = {"parse_error": repr(exc), "raw_stdout": result["stdout"]}
    return {
        "mode": mode,
        "ambient_pythonpath_removed": "PYTHONPATH" not in environment,
        "host_render_gate_removed": HOST_RENDER_ENV not in environment,
        "result": result,
        "payload": payload,
    }


def _candidate_import_contract(
    attestation: dict[str, Any], *, worker_count: int = 1, viewer_count: int = 1
) -> dict[str, Any]:
    checks = {
        "bootstrap_mode_enabled": attestation.get("bootstrap_mode") == "enabled",
        "repo_root_present_exactly_once": attestation.get("repo_root_occurrences") == 1,
        "repo_root_is_sys_path_0": attestation.get("sys_path_0") == str(REPO),
        "python_executable_exact": attestation.get("python_executable")
        == str(PYTHON.resolve()),
        "direct_script_exact": attestation.get("script_path") == str(HARNESS),
        "repo_cwd_exact": attestation.get("cwd") == str(REPO),
        "bytecode_disabled": attestation.get("dont_write_bytecode") is True,
        "helper_path_exact": attestation.get("helper_path") == str(RERUN_CONTRACT_HELPER),
        "helper_hash_exact": attestation.get("helper_sha256")
        == "aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e",
        "ambient_pythonpath_absent": attestation.get("ambient_pythonpath_present") is False,
        "one_worker_no_retry": worker_count == 1,
        "viewer_at_most_one": viewer_count <= 1,
    }
    return {"checks": checks, "pass": all(checks.values())}


def _negative_controls(
    baseline_run: dict[str, Any], disabled_run: dict[str, Any], source_manifest: dict[str, Any]
) -> dict[str, Any]:
    baseline_payload = baseline_run["payload"]
    baseline_contract = _candidate_import_contract(baseline_payload)
    wrong_path = dict(baseline_payload)
    wrong_path["helper_path"] = "/tmp/not-the-helper.py"
    wrong_hash = dict(baseline_payload)
    wrong_hash["helper_sha256"] = "0" * 64
    missing_root = dict(baseline_payload)
    missing_root["repo_root_occurrences"] = 0
    ambient = dict(baseline_payload)
    ambient["ambient_pythonpath_present"] = True
    tampered_manifest = dict(source_manifest)
    tampered_manifest[_rel(SOURCE_D369_PRESENTATION)] = {
        **tampered_manifest[_rel(SOURCE_D369_PRESENTATION)],
        "sha256": "f" * 64,
    }
    disabled_payload = disabled_run["payload"]
    checks = {
        "baseline_accepts": baseline_run["result"]["returncode"] == 0
        and baseline_payload.get("pass") is True
        and baseline_contract["pass"],
        "disabled_bootstrap_reproduces_d369_failure": disabled_run["result"]["returncode"] == 86
        and disabled_payload.get("error_type") == "ModuleNotFoundError"
        and disabled_payload.get("error") == "No module named 'roarm_rl'"
        and disabled_payload.get("error_name") == "roarm_rl"
        and disabled_payload.get("repo_root_occurrences") == 0,
        "wrong_helper_path_rejects": not _candidate_import_contract(wrong_path)["pass"],
        "wrong_helper_hash_rejects": not _candidate_import_contract(wrong_hash)["pass"],
        "missing_repo_root_rejects": not _candidate_import_contract(missing_root)["pass"],
        "ambient_pythonpath_dependence_rejects": not _candidate_import_contract(ambient)["pass"],
        "d369_source_hash_tamper_rejects": tampered_manifest != EXPECTED_SOURCE_MANIFEST
        and source_manifest == EXPECTED_SOURCE_MANIFEST,
        "second_host_worker_rejects": not _candidate_import_contract(
            baseline_payload, worker_count=2
        )["pass"],
        "second_viewer_rejects": not _candidate_import_contract(
            baseline_payload, viewer_count=2
        )["pass"],
    }
    return {
        "baseline": baseline_run,
        "disabled_bootstrap": disabled_run,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "pass": all(checks.values()),
    }


def _d369_lineage_checks() -> tuple[dict[str, Any], dict[str, bool]]:
    prereg = _read_json(SOURCE_D369_PREREG)
    exception = _read_json(SOURCE_D369_EXCEPTION)
    phases = _read_jsonl(SOURCE_D369_PHASE)
    phase_names = [row.get("phase") for row in phases]
    phase_times = [row.get("monotonic_seconds") for row in phases]
    science = prereg.get("science_boundary", {})
    expected_inventory = {path.name for path in SOURCE_PATHS if path.parent == SOURCE_D369_DIR}
    checks = {
        "d369_partial_inventory_exact_8": {path.name for path in SOURCE_D369_DIR.iterdir()}
        == expected_inventory
        and len(expected_inventory) == 8,
        "d369_preregistration_exact": prereg.get("artifact") == "D369_PREREGISTRATION_V1"
        and prereg.get("case") == "g0a_d369"
        and prereg.get("pass") is True,
        "d369_display_variables_exact": prereg.get("new_variables") == INHERITED_DISPLAY_VARIABLES
        and prereg.get("new_physical_variables") == [],
        "d369_phase_prefix_exact_7": phase_names == EXPECTED_D369_PHASES,
        "d369_phase_times_forward_only": all(
            isinstance(value, (int, float)) for value in phase_times
        )
        and all(first <= second for first, second in zip(phase_times, phase_times[1:])),
        "d369_exception_exact": exception.get("artifact") == "D369_RUNTIME_EXCEPTION_V1"
        and exception.get("case") == "g0a_d369"
        and exception.get("error") == 'ModuleNotFoundError("No module named \'roarm_rl\'")'
        and exception.get("render_invocation_exists") is False
        and exception.get("render_retry_forbidden") is True
        and exception.get("scope_guards") == SCOPE_GUARDS,
        "d369_blueprint_spec_exact": prereg.get("blueprint_spec", {}).get("spatial_view_count") == 4
        and prereg.get("blueprint_spec", {}).get("text_document_view_count") == 2
        and prereg.get("blueprint_spec", {}).get("dataframe_view_count") == 0
        and prereg.get("blueprint_spec", {}).get("time_series_view_count") == 0
        and prereg.get("blueprint_spec", {}).get("label_entity_paths_included") == 0,
        "d369_science_boundary_exact": all(
            science.get(key) is None
            for key in (
                "current_64cap_optimal",
                "physics_equivalence",
                "collider_count_tipping_causality",
                "actual_gpu_contact_execution",
                "grasp_feasibility",
            )
        )
        and science.get("g0a_pass") is False,
        "d369_base_copy_equals_d368_rrd": _sha(SOURCE_D369_BASE) == _sha(SOURCE_D368_RRD)
        and SOURCE_D369_BASE.stat().st_size == SOURCE_D368_RRD.stat().st_size,
    }
    return prereg, checks


def _strict_index_names(stats_text: str) -> set[str]:
    match = re.search(
        r"Num chunks per index\n-+\n(?P<body>.*?)(?=\nNum chunks per |\nSize \(|\Z)",
        stats_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("missing Rerun index section")
    return {
        line.split(":", 1)[0].strip()
        for line in match.group("body").splitlines()
        if ":" in line
    }


def _blueprint_activation_commands(text: str) -> list[dict[str, Any]]:
    rows = re.findall(
        r'BlueprintActivationCommand\(StoreId\(Blueprint,\s*"([^"]+)",\s*"([^"]+)"\),'
        r"\s*make_active:\s*(true|false),\s*make_default:\s*(true|false)\)",
        text,
        flags=re.DOTALL,
    )
    return [
        {
            "application_id": app_id,
            "blueprint_id": blueprint_id,
            "make_active": active == "true",
            "make_default": default == "true",
        }
        for app_id, blueprint_id, active, default in rows
    ]


def _loopback_bind_probe() -> dict[str, Any]:
    started = time.monotonic()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        handle.listen(1)
        address, port = handle.getsockname()
    return {
        "attempt_count": 1,
        "bind_succeeded": True,
        "listen_succeeded": True,
        "address": address,
        "ephemeral_port": port,
        "elapsed_seconds": time.monotonic() - started,
        "pass": address == "127.0.0.1" and isinstance(port, int) and port > 0,
    }


def _png_dimensions(path: Path) -> list[int]:
    from PIL import Image

    with Image.open(path) as image:
        return [int(image.width), int(image.height)]


def _spatial_panel_diagnostics(path: Path) -> dict[str, Any]:
    from PIL import Image, ImageStat

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        top = max(1, int(round(height * 0.022)))
        left_width = int(round(width * 0.72))
        middle_x = left_width // 2
        middle_y = top + (height - top) // 2
        boxes = {
            "link5_full": (0, top, middle_x, middle_y),
            "link5_zoom": (middle_x, top, left_width, middle_y),
            "moving_full": (0, middle_y, middle_x, height),
            "moving_zoom": (middle_x, middle_y, left_width, height),
        }
        required = {
            "link5_full": ["cyan", "green", "blue"],
            "link5_zoom": ["cyan", "green"],
            "moving_full": ["cyan", "yellow", "purple", "blue"],
            "moving_zoom": ["cyan", "yellow", "purple"],
        }
        rows: dict[str, Any] = {}
        for name, box in boxes.items():
            crop = rgb.crop(box)
            variance = sum(ImageStat.Stat(crop).var) / 3.0
            inner = crop.crop(
                (
                    int(crop.width * 0.04),
                    int(crop.height * 0.07),
                    int(crop.width * 0.96),
                    int(crop.height * 0.96),
                )
            )
            counts = {key: 0 for key in ("cyan", "green", "yellow", "purple", "blue")}
            stride = max(1, min(inner.width, inner.height) // 260)
            for y in range(0, inner.height, stride):
                for x in range(0, inner.width, stride):
                    red, green, blue = inner.getpixel((x, y))
                    hue, saturation, value = colorsys.rgb_to_hsv(
                        red / 255.0, green / 255.0, blue / 255.0
                    )
                    if saturation < 0.45 or value < 0.35:
                        continue
                    if 0.47 <= hue < 0.57:
                        counts["cyan"] += 1
                    if 0.25 <= hue < 0.44:
                        counts["green"] += 1
                    if 0.10 <= hue < 0.20:
                        counts["yellow"] += 1
                    if 0.72 <= hue < 0.90:
                        counts["purple"] += 1
                    if 0.57 <= hue < 0.72:
                        counts["blue"] += 1
            color_checks = {color: counts[color] >= 25 for color in required[name]}
            rows[name] = {
                "box": list(box),
                "mean_channel_variance": variance,
                "nonblank": variance > 20.0,
                "semantic_color_sample_stride": stride,
                "semantic_color_pixels": counts,
                "required_semantic_colors": required[name],
                "semantic_color_checks": color_checks,
                "semantic_content_signature_pass": all(color_checks.values()),
            }
        dimensions = [width, height]
        return {
            "image_dimensions": dimensions,
            "allowed_dimensions": [[1920, 1080], [3840, 2160], [4800, 2800]],
            "dimensions_allowed": dimensions in ([1920, 1080], [3840, 2160], [4800, 2800]),
            "panels": rows,
            "all_four_nonblank": all(row["nonblank"] for row in rows.values()),
            "all_four_semantic_content_signatures_pass": all(
                row["semantic_content_signature_pass"] for row in rows.values()
            ),
        }


def _font_manifest() -> dict[str, dict[str, Any]]:
    paths = {
        "regular": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        "bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    }
    return {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
        for name, path in paths.items()
    }


def _frozen_board_plan_contract(d369_prereg: dict[str, Any]) -> dict[str, Any]:
    plan = d369_prereg.get("negative_controls", {}).get("baseline", {}).get("layout", {})
    items = plan.get("items", []) if isinstance(plan, dict) else []
    required_lines = {
        "Separate hullVertexLimit: 64 (schema 64 | UI 8..64)",
        "Installed schema extension: 107.3.26+107.3.3",
        "Frozen D368 controls: 8/8 PASS",
        "Actual GPU contact execution: NULL (not instrumented)",
    }
    item_texts = {
        item.get("text") for item in items if isinstance(item, dict) and isinstance(item.get("text"), str)
    }
    item_shapes_exact = all(
        isinstance(item, dict)
        and set(item) == {"bbox", "style", "text"}
        and item.get("style") in {"section", "body"}
        and isinstance(item.get("text"), str)
        and isinstance(item.get("bbox"), list)
        and len(item["bbox"]) == 4
        and all(isinstance(value, int) for value in item["bbox"])
        for item in items
    )
    checks = {
        "d369_negative_control_baseline_pass": d369_prereg.get("negative_controls", {})
        .get("baseline", {})
        .get("pass")
        is True,
        "frozen_layout_dimensions_exact": plan.get("width") == 540
        and plan.get("height") == 790,
        "frozen_layout_summary_exact": plan.get("pass") is True
        and plan.get("text_bbox_within_zone") is True
        and plan.get("text_intersection_count") == 0
        and plan.get("intersections") == []
        and plan.get("y_end") == 548,
        "frozen_layout_items_exact_21": len(items) == 21 and item_shapes_exact,
        "required_professor_lines_preserved": required_lines <= item_texts,
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return {
        "layout_sha256": hashlib.sha256(canonical).hexdigest(),
        "item_count": len(items),
        "required_lines": sorted(required_lines),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _draw_professor_board(
    d369_prereg: dict[str, Any], expected_fonts: dict[str, Any]
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if BOARD_PNG_PATH.exists():
        raise FileExistsError(BOARD_PNG_PATH)
    current_fonts = _font_manifest()
    if current_fonts != expected_fonts:
        raise RuntimeError("font manifest changed after preregistration")
    plan_contract = _frozen_board_plan_contract(d369_prereg)
    if not plan_contract["pass"]:
        raise RuntimeError(f"frozen D369 board plan failed: {plan_contract['checks']}")
    facts = d369_prereg["facts"]
    plan = d369_prereg["negative_controls"]["baseline"]["layout"]
    regular = current_fonts["regular"]["path"]
    bold = current_fonts["bold"]["path"]
    title_font = ImageFont.truetype(bold, 32)
    subtitle_font = ImageFont.truetype(regular, 17)
    section_font = ImageFont.truetype(bold, 20)
    body_font = ImageFont.truetype(regular, 17)
    small_font = ImageFont.truetype(regular, 15)

    canvas = Image.new("RGB", (1920, 1080), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    header_zone = (0, 0, 1920, 88)
    geometry_zone = (34, 108, 1300, 930)
    card_zone = (1324, 108, 1890, 930)
    footer_zone = (30, 962, 1900, 1070)
    draw.rectangle(header_zone, fill="#15202b")
    draw.rounded_rectangle(geometry_zone, radius=12, fill="#111820", outline="#627180", width=2)
    draw.rounded_rectangle(card_zone, radius=12, fill="white", outline="#a8b2bd", width=2)
    footer_top = 956
    draw.line((34, footer_top, 1890, footer_top), fill="#84919d", width=2)

    text_items: list[dict[str, Any]] = []

    def add_text(x: int, y: int, text: str, font: Any, fill: str, zone: str) -> None:
        bbox = list(map(int, draw.textbbox((x, y), text, font=font)))
        draw.text((x, y), text, font=font, fill=fill)
        text_items.append({"text": text, "bbox": bbox, "zone": zone})

    title = (
        f"D369 | Current {facts['max_convex_hulls']['project_input']}-cap reference: "
        "certified semantic-patch carrier allocation"
    )
    subtitle = (
        "Frozen D368 evidence replay | OFFLINE display repair | no Isaac, PhysX, q5, or physics step"
    )
    add_text(34, 14, title, title_font, "white", "header")
    add_text(
        36,
        57,
        subtitle,
        subtitle_font,
        "#dce6ef",
        "header",
    )

    with Image.open(RERUN_PNG_PATH) as source_image:
        source_rgb = source_image.convert("RGB")
        top = max(1, int(round(source_rgb.height * 0.022)))
        left_width = int(round(source_rgb.width * 0.72))
        crop_box = (0, top, left_width, source_rgb.height)
        spatial_group = source_rgb.crop(crop_box)
        fitted = ImageOps.contain(spatial_group, (1242, 798), method=Image.Resampling.LANCZOS)
        paste_x = geometry_zone[0] + (geometry_zone[2] - geometry_zone[0] - fitted.width) // 2
        paste_y = geometry_zone[1] + (geometry_zone[3] - geometry_zone[1] - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))
        source_aspect = spatial_group.width / spatial_group.height
        fitted_aspect = fitted.width / fitted.height

    classes = facts["classification_counts"]["gripper_link"]
    dual_inner_outer = classes["mixed_certified:inner_contact_patch+outer_negative_patch"]
    inner_only = classes["certified:inner_contact_patch"]
    boundary = facts["interpretation_boundary"]
    for item in plan["items"]:
        font = section_font if item["style"] == "section" else body_font
        local = item["bbox"]
        add_text(
            card_zone[0] + 13 + local[0],
            card_zone[1] + 3 + local[1],
            item["text"],
            font,
            "#17212b",
            "card",
        )

    null_text = lambda value: "NULL" if value is None else str(value)
    unresolved_values = " / ".join(
        null_text(boundary[key])
        for key in (
            "current_64cap_optimal",
            "physics_equivalence",
            "collider_count_tipping_causality",
            "actual_gpu_contact_execution",
            "grasp_feasibility",
        )
    )
    add_text(
        44,
        970,
        "CYAN raw patch | GREEN certified | YELLOW dual inner+outer | BLUE other | PURPLE outer patch",
        small_font,
        "#28343f",
        "footer",
    )
    add_text(
        44,
        1001,
        f"{facts['max_convex_hulls']['project_input']} optimal / physics-equivalent / "
        "toppling-cause / GPU-contact / grasp = "
        f"{unresolved_values}. G0a = {str(boundary['g0a_pass']).lower()}.",
        small_font,
        "#28343f",
        "footer",
    )
    add_text(
        44,
        1032,
        "Authority: immutable D368 JSON + RRD. This board is display evidence only.",
        small_font,
        "#56636f",
        "footer",
    )

    zone_bounds = {
        "header": (20, 8, 1900, 84),
        "card": (card_zone[0] + 8, card_zone[1] + 8, card_zone[2] - 8, card_zone[3] - 8),
        "footer": footer_zone,
    }
    containment: dict[str, bool] = {}
    for index, item in enumerate(text_items):
        zone = zone_bounds[item["zone"]]
        box = item["bbox"]
        containment[str(index)] = (
            box[0] >= zone[0]
            and box[1] >= zone[1]
            and box[2] <= zone[2]
            and box[3] <= zone[3]
        )
    overlaps: list[dict[str, Any]] = []
    for i, first in enumerate(text_items):
        for j in range(i + 1, len(text_items)):
            second = text_items[j]
            if first["zone"] != second["zone"]:
                continue
            x0 = max(first["bbox"][0], second["bbox"][0])
            y0 = max(first["bbox"][1], second["bbox"][1])
            x1 = min(first["bbox"][2], second["bbox"][2])
            y1 = min(first["bbox"][3], second["bbox"][3])
            if x1 > x0 and y1 > y0:
                overlaps.append({"i": i, "j": j, "area_px2": (x1 - x0) * (y1 - y0)})

    encoded = io.BytesIO()
    canvas.save(encoded, format="PNG")
    with BOARD_PNG_PATH.open("xb") as handle:
        handle.write(encoded.getvalue())
    checks = {
        "exact_1920x1080": _png_dimensions(BOARD_PNG_PATH) == [1920, 1080],
        "all_text_within_registered_zones": all(containment.values()),
        "drawn_text_bbox_overlap_zero": not overlaps,
        "zones_disjoint": geometry_zone[2] < card_zone[0]
        and geometry_zone[3] < footer_top
        and card_zone[3] < footer_top,
        "aspect_ratio_preserved": abs(source_aspect - fitted_aspect) <= 0.003,
        "frozen_d369_fact_card_plan_pass": plan_contract["pass"],
        "frozen_d369_fact_card_items_rendered_exact": [
            item["text"] for item in text_items if item["zone"] == "card"
        ]
        == [item["text"] for item in plan["items"]],
        "science_boundary_exact": all(
            boundary[key] is None
            for key in (
                "current_64cap_optimal",
                "physics_equivalence",
                "collider_count_tipping_causality",
                "actual_gpu_contact_execution",
                "grasp_feasibility",
            )
        )
        and boundary["g0a_pass"] is False,
        "inner_outer_relationship_exact": facts["moving_inner"]["part_count"] == 17
        and dual_inner_outer == 16
        and inner_only == 1
        and facts["moving_outer"]["part_count"] == 16,
        "scope_boundary_exact": SCOPE_GUARDS["rerun_display_render_allowed"] == 1
        and all(
            value == 0
            for key, value in SCOPE_GUARDS.items()
            if key != "rerun_display_render_allowed"
        ),
    }
    return {
        **_file_record(BOARD_PNG_PATH),
        "dimensions": _png_dimensions(BOARD_PNG_PATH),
        "source_raw_png": _file_record(RERUN_PNG_PATH),
        "frozen_d369_board_plan_contract": plan_contract,
        "source_crop_box": list(crop_box),
        "source_crop_dimensions": [spatial_group.width, spatial_group.height],
        "fitted_dimensions": [fitted.width, fitted.height],
        "paste_bbox": [paste_x, paste_y, paste_x + fitted.width, paste_y + fitted.height],
        "source_aspect_ratio": source_aspect,
        "fitted_aspect_ratio": fitted_aspect,
        "zones": {
            "header": list(header_zone),
            "geometry": list(geometry_zone),
            "fact_card": list(card_zone),
            "footer": list(footer_zone),
        },
        "text_items": text_items,
        "containment": containment,
        "overlaps": overlaps,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _pre_render_rerun_contract(prereg: dict[str, Any]) -> dict[str, Any]:
    verify_paths = [
        SOURCE_D369_BASE,
        SOURCE_D369_RECORDING,
        SOURCE_D369_OVERLAY,
        RBL_COPY_PATH,
        PRESENTATION_COPY_PATH,
    ]
    verifies = {
        _rel(path): _run(
            [str(RERUN_CLI), "rrd", "verify", "--check-footers", "true", str(path)],
            timeout=90.0,
        )
        for path in verify_paths
    }
    compare = _run(
        [
            str(RERUN_CLI),
            "rrd",
            "compare",
            "--unordered",
            str(SOURCE_D369_BASE),
            str(SOURCE_D369_RECORDING),
        ],
        timeout=90.0,
    )
    recording_stats = _run(
        [str(RERUN_CLI), "rrd", "stats", str(SOURCE_D369_RECORDING)], timeout=90.0
    )
    overlay_stats = _run(
        [str(RERUN_CLI), "rrd", "stats", str(SOURCE_D369_OVERLAY)], timeout=90.0
    )
    presentation_stats = _run(
        [str(RERUN_CLI), "rrd", "stats", str(PRESENTATION_COPY_PATH)], timeout=90.0
    )
    recording_entities = {
        path
        for path in _entity_paths(f"{recording_stats['stdout']}\n{recording_stats['stderr']}")
        if not _is_system_entity(path)
    }
    overlay_entities = {
        path
        for path in _entity_paths(f"{overlay_stats['stdout']}\n{overlay_stats['stderr']}")
        if not _is_system_entity(path)
    }
    presentation_entities = {
        path
        for path in _entity_paths(f"{presentation_stats['stdout']}\n{presentation_stats['stderr']}")
        if not _is_system_entity(path)
    }
    expected_presentation_entities = recording_entities | overlay_entities
    validation = validate_rerun_artifact(
        PRESENTATION_COPY_PATH,
        expected_entity_paths=sorted(expected_presentation_entities),
        exact_entity_paths=sorted(expected_presentation_entities),
        expected_timeline_names=["blueprint", "log_time"],
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components={
            "/presentation/d369/allocation": ["TextDocument:text", "TextDocument:media_type"],
            "/presentation/d369/scope_and_64_basis": [
                "TextDocument:text",
                "TextDocument:media_type",
            ],
        },
        cli_path=RERUN_CLI,
        expected_version=EXPECTED_RERUN_VERSION,
        timeout_s=90.0,
    )
    rbl_print = _run([str(RERUN_CLI), "rrd", "print", "-vvv", str(RBL_COPY_PATH)], timeout=90.0)
    presentation_print = _run(
        [str(RERUN_CLI), "rrd", "print", "-vvv", str(PRESENTATION_COPY_PATH)], timeout=90.0
    )
    rbl_text = f"{rbl_print['stdout']}\n{rbl_print['stderr']}"
    presentation_text = f"{presentation_print['stdout']}\n{presentation_print['stderr']}"
    rbl_activation = _blueprint_activation_commands(rbl_text)
    presentation_activation = _blueprint_activation_commands(presentation_text)
    expected_activation = [
        {
            "application_id": APP_ID,
            "blueprint_id": EXPECTED_BLUEPRINT_ID,
            "make_active": True,
            "make_default": True,
        }
    ]
    checks = {
        "five_footer_verifies_pass": len(verifies) == 5
        and all(result["ok"] for result in verifies.values()),
        "recording_compare_unordered_pass": compare["ok"] and compare["returncode"] == 0,
        "recording_entity_count_exact_284": len(recording_entities) == 284,
        "overlay_entity_set_exact": overlay_entities
        == {"/presentation/d369/allocation", "/presentation/d369/scope_and_64_basis"},
        "presentation_entity_union_exact_286": len(presentation_entities) == 286
        and presentation_entities == expected_presentation_entities,
        "presentation_validation_pass": validation["pass"] is True,
        "presentation_indexes_exact": presentation_stats["ok"]
        and _strict_index_names(
            f"{presentation_stats['stdout']}\n{presentation_stats['stderr']}"
        )
        == {"blueprint", "log_time"},
        "rbl_views_exact_4_spatial_2_text": rbl_print["ok"]
        and rbl_text.count("[3D]") == 4
        and rbl_text.count("[TextDocument]") == 2
        and rbl_text.count("[Dataframe]") == 0
        and rbl_text.count("[TimeSeries]") == 0
        and rbl_text.count("[TextLog]") == 0,
        "rbl_text_queries_exact": "/presentation/d369/allocation" in rbl_text
        and "/presentation/d369/scope_and_64_basis" in rbl_text,
        "rbl_label_queries_absent": "/semantic/anchors" not in rbl_text
        and "/semantic/normals" not in rbl_text,
        "activation_exact_and_matching": rbl_activation
        == presentation_activation
        == expected_activation,
        "active_copies_bitexact": _sha(PRESENTATION_COPY_PATH)
        == prereg["source_manifest"][_rel(SOURCE_D369_PRESENTATION)]["sha256"]
        and _sha(RBL_COPY_PATH) == prereg["source_manifest"][_rel(SOURCE_D369_RBL)]["sha256"],
    }
    return {
        "cli_version": _run([str(RERUN_CLI), "--version"]),
        "verifies": verifies,
        "recording_compare": compare,
        "recording_stats": recording_stats,
        "overlay_stats": overlay_stats,
        "presentation_stats": presentation_stats,
        "recording_entity_count": len(recording_entities),
        "overlay_entities": sorted(overlay_entities),
        "presentation_entity_count": len(presentation_entities),
        "validation": validation,
        "rbl_print": rbl_print,
        "presentation_print": presentation_print,
        "rbl_activation": rbl_activation,
        "presentation_activation": presentation_activation,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _prepare() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if any(OUT_DIR.iterdir()):
        raise RuntimeError("D370 output must be empty before prepare")
    source_manifest = _source_manifest()
    d369_prereg, lineage_checks = _d369_lineage_checks()
    frozen_board_plan_contract = _frozen_board_plan_contract(d369_prereg)
    baseline_run = _run_production_import_preflight("enabled")
    disabled_run = _run_production_import_preflight("disabled")
    preflight_output_remained_empty = not any(OUT_DIR.iterdir())
    negative = _negative_controls(baseline_run, disabled_run, source_manifest)
    baseline_payload = baseline_run["payload"]
    _, production_render_command = _production_preflight_commands()
    cli_version = _run([str(RERUN_CLI), "--version"])
    dirty_paths = _git_status_paths()
    fonts = _font_manifest()
    checks = {
        "output_empty_before_preregistration": True,
        "head_origin_expected": _git("rev-parse", "HEAD")
        == _git("rev-parse", "origin/master")
        == EXPECTED_HEAD,
        "dirty_scope_allowed": _dirty_scope_allowed(dirty_paths),
        "source_manifest_exact": source_manifest == EXPECTED_SOURCE_MANIFEST,
        "d369_lineage_exact": all(lineage_checks.values()),
        "frozen_d369_board_plan_pass": frozen_board_plan_contract["pass"],
        "new_variable_exactly_one": NEW_VARIABLES
        == ["production_command_repo_root_import_preflight"],
        "no_new_physical_variables": True,
        "production_preflight_return_zero": baseline_run["result"]["returncode"] == 0,
        "production_preflight_payload_pass": baseline_payload.get("pass") is True,
        "production_preflight_ambient_pythonpath_absent": baseline_payload.get(
            "ambient_pythonpath_present"
        )
        is False,
        "preflight_output_remained_empty": preflight_output_remained_empty,
        "production_and_render_command_prefix_exact": baseline_run["result"]["command"][:-1]
        == production_render_command[:-1],
        "failure_capable_controls_9_of_9": negative["pass"]
        and negative["passed"] == 9
        and negative["total"] == 9,
        "scope_guards_exact": SCOPE_GUARDS["rerun_display_render_allowed"] == 1
        and all(
            value == 0
            for key, value in SCOPE_GUARDS.items()
            if key != "rerun_display_render_allowed"
        ),
        "rerun_version_exact": cli_version["ok"]
        and EXPECTED_RERUN_VERSION in f"{cli_version['stdout']}\n{cli_version['stderr']}",
        "isaac_compatible_pins_unchanged": importlib.metadata.version("numpy") == "1.26.0"
        and importlib.metadata.version("psutil") == "5.9.8",
        "fonts_present_and_hash_bound": set(fonts) == {"regular", "bold"}
        and all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in fonts.values()),
        "forbidden_modules_absent": not _forbidden_modules(),
    }
    prereg = {
        "artifact": "D370_PREREGISTRATION_V1",
        "case": "g0a_d370",
        "head": EXPECTED_HEAD,
        "new_variables": NEW_VARIABLES,
        "inherited_display_variables": INHERITED_DISPLAY_VARIABLES,
        "new_physical_variables": [],
        "source_manifest": source_manifest,
        "dynamic_source_manifest": _dynamic_manifest(),
        "d369_lineage_checks": lineage_checks,
        "frozen_d369_board_plan_contract": frozen_board_plan_contract,
        "production_import_preflight": baseline_run,
        "disabled_bootstrap_control": disabled_run,
        "failure_capable_controls": negative,
        "production_render_command": production_render_command,
        "render_contract": {
            "host_worker_count": 1,
            "viewer_invocation_count": 1,
            "automatic_retry_count": 0,
            "host_gate": {HOST_RENDER_ENV: "1"},
            "viewer_input": _rel(PRESENTATION_COPY_PATH),
            "window_size": WINDOW_SIZE,
            "allowed_raw_png_dimensions": [[1920, 1080], [3840, 2160], [4800, 2800]],
            "board_dimensions": [1920, 1080],
        },
        "expected_phase_sequence": EXPECTED_PHASE_SEQUENCE,
        "fonts": fonts,
        "facts": d369_prereg["facts"],
        "science_boundary": d369_prereg["science_boundary"],
        "scope_guards": SCOPE_GUARDS,
        "checks": checks,
        "pass": all(checks.values()),
        "decision_rule": (
            "Only after preregistration PASS may one host render worker run. Any worker/Viewer "
            "failure freezes D370 with no retry. Visual PASS cannot change D368 science."
        ),
    }
    _write_json_x(PREREG_PATH, prereg)


def _render() -> None:
    if os.environ.get(HOST_RENDER_ENV) != "1":
        raise RuntimeError(f"set {HOST_RENDER_ENV}=1 only on the approved host worker")
    if {path.name for path in OUT_DIR.iterdir()} != {PREREG_PATH.name}:
        raise RuntimeError("forward-only D370 pre-render inventory mismatch")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D370 preregistration did not pass")
    if _git("rev-parse", "HEAD") != EXPECTED_HEAD or _git("rev-parse", "origin/master") != EXPECTED_HEAD:
        raise RuntimeError("Git base changed after D370 preregistration")
    if not _dirty_scope_allowed(_git_status_paths()):
        raise RuntimeError("dirty worktree escaped D370 allowed scope")
    if _dynamic_manifest() != prereg["dynamic_source_manifest"]:
        raise RuntimeError("D370 harness/helper changed after preregistration")
    if _source_manifest() != prereg["source_manifest"]:
        raise RuntimeError("frozen D368/D369 source changed after preregistration")
    if _forbidden_modules():
        raise RuntimeError(f"forbidden modules imported before render: {_forbidden_modules()}")

    PHASE_PATH.touch(exist_ok=False)
    _phase("render_worker_started")
    try:
        attestation = _import_attestation()
        if not attestation["pass"]:
            raise RuntimeError(f"in-process import attestation failed: {attestation}")
        _write_json_x(IMPORT_ATTESTATION_PATH, attestation)
        _phase("production_command_import_attestation_rechecked")

        current_manifest = _source_manifest()
        d369_prereg, lineage_checks = _d369_lineage_checks()
        if not all(lineage_checks.values()):
            raise RuntimeError(f"D369 lineage changed: {lineage_checks}")
        if _frozen_board_plan_contract(d369_prereg) != prereg[
            "frozen_d369_board_plan_contract"
        ]:
            raise RuntimeError("frozen D369 board plan changed after preregistration")
        _phase("frozen_d369_manifest_verified")
        if not (
            current_manifest == prereg["source_manifest"] == EXPECTED_SOURCE_MANIFEST
            and _sha(SOURCE_D369_BASE) == _sha(SOURCE_D368_RRD)
        ):
            raise RuntimeError("D368 authority chain changed")
        _phase("frozen_d368_authority_verified")

        _copy_x(SOURCE_D369_PRESENTATION, PRESENTATION_COPY_PATH)
        _copy_x(SOURCE_D369_RBL, RBL_COPY_PATH)
        if _sha(PRESENTATION_COPY_PATH) != _sha(SOURCE_D369_PRESENTATION) or _sha(
            RBL_COPY_PATH
        ) != _sha(SOURCE_D369_RBL):
            raise RuntimeError("D370 active copies are not bit-exact")
        _phase("active_rrd_rbl_bitexact_copies_written")

        pre_render = _pre_render_rerun_contract(prereg)
        if not pre_render["pass"]:
            raise RuntimeError(f"pre-render Rerun contract failed: {pre_render['checks']}")
        _phase("pre_render_artifact_contract_pass")

        capability = _loopback_bind_probe()
        if not capability["pass"]:
            raise RuntimeError(f"host loopback capability failed: {capability}")
        _phase("host_loopback_bind_capability_pass")

        staging_png = Path(f"/tmp/roarm_d370_{os.getpid()}_rerun.png")
        if staging_png.exists():
            raise FileExistsError(staging_png)
        render_command = [
            str(RERUN_CLI),
            "--headless",
            "--hide-welcome-screen",
            "--port",
            "auto",
            "--window-size",
            WINDOW_SIZE,
            "--screenshot-to",
            str(staging_png),
            str(PRESENTATION_COPY_PATH),
        ]
        invocation = {
            "artifact": "D370_ONE_SHOT_VIEWER_INVOCATION_V1",
            "case": "g0a_d370",
            "host_worker_count": 1,
            "viewer_invocation_count": 1,
            "automatic_retry_count": 0,
            "no_retry": True,
            "command": render_command,
            "input": _file_record(PRESENTATION_COPY_PATH),
            "preregistration_sha256": _sha(PREREG_PATH),
            "import_attestation_sha256": _sha(IMPORT_ATTESTATION_PATH),
            "host_gate": {HOST_RENDER_ENV: os.environ.get(HOST_RENDER_ENV)},
            "host_capability": capability,
            "staging_screenshot_path": str(staging_png),
            "final_screenshot_path": _rel(RERUN_PNG_PATH),
            "scope_guards": SCOPE_GUARDS,
        }
        _write_json_x(INVOCATION_PATH, invocation)
        _phase("one_shot_viewer_invocation_recorded")
        result = _run(render_command, timeout=RENDER_TIMEOUT_SECONDS)
        if result["ok"] and staging_png.is_file():
            _copy_x(staging_png, RERUN_PNG_PATH)
        receipt = {
            "artifact": "D370_ONE_SHOT_VIEWER_RECEIPT_V1",
            "case": "g0a_d370",
            **result,
            "staging_screenshot_exists": staging_png.is_file(),
            "screenshot_exists": RERUN_PNG_PATH.is_file(),
            "screenshot_bytes": RERUN_PNG_PATH.stat().st_size if RERUN_PNG_PATH.is_file() else 0,
            "screenshot_sha256": _sha(RERUN_PNG_PATH) if RERUN_PNG_PATH.is_file() else None,
            "unknown_timeline_console_text_absent": "Unknown timeline"
            not in f"{result['stdout']}\n{result['stderr']}",
            "message_proxy_operation_not_permitted_pair_absent": not (
                "message proxy server crashed" in f"{result['stdout']}\n{result['stderr']}"
                and "Operation not permitted" in f"{result['stdout']}\n{result['stderr']}"
            ),
        }
        _write_json_x(RECEIPT_PATH, receipt)
        if not (result["ok"] and RERUN_PNG_PATH.is_file()):
            raise RuntimeError(f"one-shot D370 Viewer failed: {receipt}")
        _phase("one_shot_viewer_returned", returncode=result["returncode"])

        diagnostics = _spatial_panel_diagnostics(RERUN_PNG_PATH)
        if not (
            diagnostics["dimensions_allowed"]
            and diagnostics["all_four_nonblank"]
            and diagnostics["all_four_semantic_content_signatures_pass"]
        ):
            raise RuntimeError(f"raw PNG automated diagnostics failed: {diagnostics}")
        _phase("raw_png_automated_gate_pass")

        board = _draw_professor_board(d369_prereg, prereg["fonts"])
        if not board["pass"]:
            raise RuntimeError(f"professor board automated layout failed: {board['checks']}")
        _phase("professor_board_written", sha256=board["sha256"])

        source_after = _source_manifest()
        automated_checks = {
            "import_attestation_pass": attestation["pass"],
            "source_inputs_unchanged": source_after == prereg["source_manifest"],
            "active_copies_bitexact": _sha(PRESENTATION_COPY_PATH)
            == _sha(SOURCE_D369_PRESENTATION)
            and _sha(RBL_COPY_PATH) == _sha(SOURCE_D369_RBL),
            "pre_render_contract_pass": pre_render["pass"],
            "viewer_returncode_zero": result["returncode"] == 0,
            "viewer_invocation_exactly_one": invocation["viewer_invocation_count"] == 1,
            "automatic_retry_zero": invocation["automatic_retry_count"] == 0,
            "raw_png_diagnostics_pass": diagnostics["dimensions_allowed"]
            and diagnostics["all_four_nonblank"]
            and diagnostics["all_four_semantic_content_signatures_pass"],
            "board_layout_pass": board["pass"],
            "unknown_timeline_console_signature_absent": receipt[
                "unknown_timeline_console_text_absent"
            ],
            "message_proxy_error_pair_absent": receipt[
                "message_proxy_operation_not_permitted_pair_absent"
            ],
            "forbidden_modules_absent": not _forbidden_modules(),
            "scope_guards_preserved": prereg["scope_guards"] == SCOPE_GUARDS,
        }
        automated = {
            "artifact": "D370_AUTOMATED_SUMMARY_V1",
            "case": "g0a_d370",
            "facts": d369_prereg["facts"],
            "science_boundary": prereg["science_boundary"],
            "scope_guards": SCOPE_GUARDS,
            "import_attestation": attestation,
            "presentation_copy": _file_record(PRESENTATION_COPY_PATH),
            "rbl_copy": _file_record(RBL_COPY_PATH),
            "pre_render_validation": pre_render,
            "render_invocation": _file_record(INVOCATION_PATH),
            "render_receipt": _file_record(RECEIPT_PATH),
            "raw_png": {**_file_record(RERUN_PNG_PATH), "dimensions": _png_dimensions(RERUN_PNG_PATH)},
            "raw_png_diagnostics": diagnostics,
            "professor_board": board,
            "checks": automated_checks,
            "pass": all(automated_checks.values()),
        }
        report = "\n".join(
            [
                "# D370 automated visual-resume report",
                "",
                f"- import preflight: {'PASS' if attestation['pass'] else 'FAIL'}",
                f"- inherited Rerun contract: {'PASS' if pre_render['pass'] else 'FAIL'}",
                f"- Viewer invocation/retry: {invocation['viewer_invocation_count']}/"
                f"{invocation['automatic_retry_count']}",
                f"- raw PNG: `{_rel(RERUN_PNG_PATH)}` `{_sha(RERUN_PNG_PATH)}`",
                f"- board: `{_rel(BOARD_PNG_PATH)}` `{_sha(BOARD_PNG_PATH)}`",
                "- Scientific recomputation: none; five fields NULL; G0a false.",
                "- Human original-resolution inspection is still required before finalize.",
                "",
            ]
        )
        _write_text_x(REPORT_PATH, report)
        validation_files = {
            _rel(path): _file_record(path)
            for path in (
                PREREG_PATH,
                IMPORT_ATTESTATION_PATH,
                PRESENTATION_COPY_PATH,
                RBL_COPY_PATH,
                INVOCATION_PATH,
                RECEIPT_PATH,
                RERUN_PNG_PATH,
                BOARD_PNG_PATH,
                REPORT_PATH,
            )
        }
        validation_checks = {
            "automated_checks_pass": automated["pass"],
            "pre_render_checks_pass": pre_render["pass"],
            "source_manifest_unchanged": source_after == prereg["source_manifest"],
            "presentation_record_current": validation_files[_rel(PRESENTATION_COPY_PATH)]
            == _file_record(PRESENTATION_COPY_PATH),
            "rbl_record_current": validation_files[_rel(RBL_COPY_PATH)] == _file_record(RBL_COPY_PATH),
        }
        validation = {
            "artifact": "D370_RERUN_VALIDATION_V1",
            "case": "g0a_d370",
            "files": validation_files,
            "checks": validation_checks,
            "pass": all(validation_checks.values()),
        }
        _write_json_x(VALIDATION_PATH, validation)
        _write_json_x(AUTOMATED_PATH, automated)
        if not (validation["pass"] and automated["pass"]):
            raise RuntimeError("D370 automated validation failed")
        _phase("automated_validation_finalized")
    except Exception as exc:
        if not EXCEPTION_PATH.exists():
            _write_json_x(
                EXCEPTION_PATH,
                {
                    "artifact": "D370_RUNTIME_EXCEPTION_V1",
                    "case": "g0a_d370",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "host_worker_count": 1,
                    "viewer_invocation_exists": INVOCATION_PATH.is_file(),
                    "viewer_retry_forbidden": True,
                    "scope_guards": SCOPE_GUARDS,
                },
            )
        raise


def _record_matches(record: Any, path: Path) -> bool:
    return isinstance(record, dict) and record == _file_record(path)


def _finalize() -> None:
    if COMPLETION_PATH.exists():
        raise FileExistsError(COMPLETION_PATH)
    if EXCEPTION_PATH.exists():
        raise RuntimeError("cannot finalize a D370 runtime exception")
    prereg = _read_json(PREREG_PATH)
    attestation = _read_json(IMPORT_ATTESTATION_PATH)
    invocation = _read_json(INVOCATION_PATH)
    receipt = _read_json(RECEIPT_PATH)
    validation = _read_json(VALIDATION_PATH)
    automated = _read_json(AUTOMATED_PATH)
    manual = _read_json(MANUAL_JSON_PATH)
    manual_md = MANUAL_MD_PATH.read_text(encoding="utf-8")
    expected_inventory = {
        PREREG_PATH.name,
        IMPORT_ATTESTATION_PATH.name,
        PHASE_PATH.name,
        PRESENTATION_COPY_PATH.name,
        RBL_COPY_PATH.name,
        INVOCATION_PATH.name,
        RECEIPT_PATH.name,
        VALIDATION_PATH.name,
        RERUN_PNG_PATH.name,
        BOARD_PNG_PATH.name,
        AUTOMATED_PATH.name,
        REPORT_PATH.name,
        MANUAL_JSON_PATH.name,
        MANUAL_MD_PATH.name,
    }
    manual_checks = manual.get("checks", {})
    manual_files = manual.get("files", [])
    expected_manual_paths = {_rel(RERUN_PNG_PATH), _rel(BOARD_PNG_PATH)}
    observed_manual_paths = {
        row.get("path") for row in manual_files if isinstance(row, dict)
    }
    manual_paths_safe = len(manual_files) == 2 and all(
        isinstance(row, dict) and row.get("path") in expected_manual_paths for row in manual_files
    )
    manual_records_match = manual_paths_safe and all(
        isinstance(row.get("observations"), list)
        and len(row["observations"]) >= 3
        and sum(len(str(value)) for value in row["observations"]) >= 160
        and row.get("dimensions") == _png_dimensions(REPO / row["path"])
        and row.get("bytes") == (REPO / row["path"]).stat().st_size
        and row.get("sha256") == _sha(REPO / row["path"])
        for row in manual_files
    )
    phase_rows = _read_jsonl(PHASE_PATH)
    phase_names = [row.get("phase") for row in phase_rows]
    phase_ordinals = [row.get("ordinal") for row in phase_rows]
    phase_times = [row.get("monotonic_seconds") for row in phase_rows]
    pre_manual_checks = {
        "precompletion_inventory_exact": {path.name for path in OUT_DIR.iterdir()}
        == expected_inventory,
        "head_origin_preregistered_exact": _git("rev-parse", "HEAD")
        == _git("rev-parse", "origin/master")
        == prereg.get("head")
        == EXPECTED_HEAD,
        "dirty_scope_allowed": _dirty_scope_allowed(_git_status_paths()),
        "source_manifest_unchanged": _source_manifest() == prereg.get("source_manifest"),
        "dynamic_manifest_unchanged": _dynamic_manifest()
        == prereg.get("dynamic_source_manifest"),
        "preregistration_pass": prereg.get("pass") is True
        and all(prereg.get("checks", {}).values())
        and prereg.get("failure_capable_controls", {}).get("passed") == 9
        and prereg.get("failure_capable_controls", {}).get("total") == 9,
        "import_attestation_current": attestation.get("pass") is True
        and attestation == _import_attestation()
        and automated.get("import_attestation") == attestation
        and invocation.get("import_attestation_sha256") == _sha(IMPORT_ATTESTATION_PATH),
        "automated_and_validation_pass": automated.get("pass") is True
        and validation.get("pass") is True
        and all(automated.get("checks", {}).values())
        and all(validation.get("checks", {}).values()),
        "one_worker_one_viewer_no_retry": invocation.get("host_worker_count") == 1
        and invocation.get("viewer_invocation_count") == 1
        and invocation.get("automatic_retry_count") == 0
        and invocation.get("no_retry") is True,
        "receipt_command_and_png_current": receipt.get("command") == invocation.get("command")
        and receipt.get("returncode") == 0
        and receipt.get("screenshot_exists") is True
        and receipt.get("screenshot_bytes") == RERUN_PNG_PATH.stat().st_size
        and receipt.get("screenshot_sha256") == _sha(RERUN_PNG_PATH),
        "active_rrd_rbl_bitexact": _sha(PRESENTATION_COPY_PATH) == _sha(SOURCE_D369_PRESENTATION)
        and _sha(RBL_COPY_PATH) == _sha(SOURCE_D369_RBL)
        and _record_matches(automated.get("presentation_copy"), PRESENTATION_COPY_PATH)
        and _record_matches(automated.get("rbl_copy"), RBL_COPY_PATH),
        "manual_artifact_case_exact": manual.get("artifact")
        == "D370_MANUAL_VISUAL_INSPECTION_V1"
        and manual.get("case") == "g0a_d370",
        "manual_keys_exact_and_all_true": sorted(manual_checks) == sorted(MANUAL_CHECK_KEYS)
        and all(manual_checks.get(key) is True for key in MANUAL_CHECK_KEYS),
        "manual_exact_two_paths_and_records": observed_manual_paths == expected_manual_paths
        and manual_records_match,
        "manual_markdown_hash_bound": len(manual_md.strip()) >= 400
        and all(path in manual_md for path in expected_manual_paths)
        and _sha(RERUN_PNG_PATH) in manual_md
        and _sha(BOARD_PNG_PATH) in manual_md,
        "manual_pass": manual.get("pass") is True,
        "phase_prefix_exact_forward": phase_names == EXPECTED_PHASE_SEQUENCE[:12]
        and phase_ordinals == list(range(1, 13))
        and all(isinstance(value, (int, float)) for value in phase_times)
        and all(first <= second for first, second in zip(phase_times, phase_times[1:])),
        "board_exact_1920x1080": _png_dimensions(BOARD_PNG_PATH) == [1920, 1080],
        "science_boundary_preserved": automated.get("science_boundary")
        == prereg.get("science_boundary")
        and all(
            prereg["science_boundary"][key] is None
            for key in (
                "current_64cap_optimal",
                "physics_equivalence",
                "collider_count_tipping_causality",
                "actual_gpu_contact_execution",
                "grasp_feasibility",
            )
        )
        and prereg["science_boundary"]["g0a_pass"] is False,
        "scope_guards_exact": automated.get("scope_guards")
        == prereg.get("scope_guards")
        == SCOPE_GUARDS,
        "forbidden_modules_absent": not _forbidden_modules(),
    }
    if not all(pre_manual_checks.values()):
        raise RuntimeError(f"D370 precompletion gate failed: {pre_manual_checks}")
    _phase("manual_visual_inspection_contract_pass")
    phase_after_manual = _read_jsonl(PHASE_PATH)
    if [row.get("phase") for row in phase_after_manual] != EXPECTED_PHASE_SEQUENCE[:13]:
        raise RuntimeError("D370 manual phase sequence mismatch")
    _phase("completion_gate_pass")
    final_phases = _read_jsonl(PHASE_PATH)
    final_phase_times = [row.get("monotonic_seconds") for row in final_phases]
    if not (
        [row.get("phase") for row in final_phases] == EXPECTED_PHASE_SEQUENCE
        and [row.get("ordinal") for row in final_phases]
        == list(range(1, len(EXPECTED_PHASE_SEQUENCE) + 1))
        and all(isinstance(value, (int, float)) for value in final_phase_times)
        and all(
            first <= second for first, second in zip(final_phase_times, final_phase_times[1:])
        )
    ):
        raise RuntimeError("D370 final phase sequence mismatch")
    artifacts = {
        _rel(path): {"bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(OUT_DIR.iterdir())
    }
    completion = {
        "artifact": "D370_COMPLETION_SUMMARY_V1",
        "case": "g0a_d370",
        "checks": pre_manual_checks,
        "visualization_pass": True,
        "pass": True,
        "verdict": "D370_FROZEN_D369_PROFESSOR_VISUAL_CONTRACT_RESUMED_OBSERVABILITY_ONLY",
        "d368_measurement_recomputed": False,
        "d369_failure_rewritten": False,
        "science_boundary": prereg["science_boundary"],
        "scope_guards": SCOPE_GUARDS,
        "host_worker_count": 1,
        "viewer_invocation_count": 1,
        "automatic_retry_count": 0,
        "phase_sequence": [row["phase"] for row in final_phases],
        "artifacts_before_completion": artifacts,
        "next_science_or_collider_case_requires_new_approval": True,
        "commit_or_push_performed": False,
    }
    _write_json_x(COMPLETION_PATH, completion)


def _import_preflight() -> None:
    payload = _import_attestation()
    print(json.dumps(payload, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit(87)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("import-preflight", "prepare", "render", "finalize"), required=True
    )
    args = parser.parse_args()
    if args.stage == "import-preflight":
        _import_preflight()
    elif args.stage == "prepare":
        _prepare()
    elif args.stage == "render":
        _render()
    else:
        _finalize()


if __name__ == "__main__":
    main()
