#!/usr/bin/env python3
"""D353: prove the pending PAUSE -> explicit commit zero-step bridge, once.

This forward-only control-repair harness inherits D352's real-GUI localization,
marker, watchdog, telemetry, raw/live binding, and fail-closed science boundary.
Its only new intervention is a main-thread ``Timeline.commit()`` immediately
after an existing pause block actually issued at least one pause request.

The worker returns before q5 science.  It never calls a Kit/frame update,
forward/rewind, commit_silently, render, or physics step.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import json
import os
import secrets
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim_scripts import (  # noqa: E402
    cyl34_top_view_d352_d351_validate_phase_localization_watchdog as d352,
)


CASE = "g0a_d353"
CASE_NAME = "timeline_pause_pending_state_commit_bridge"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d353"
HARNESS = Path(__file__).resolve()
SESSION_DOC = (
    REPO
    / "claudedocs/session_20260715_grasp_g0a_d353_timeline_pause_pending_state_commit_bridge.md"
)
START_HERE = REPO / "START_HERE.md"
EXPECTED_HEAD = "1f235b8a310afeb9f4f6734d69aba2a5430b7602"
SEED = 33201

INACTIVITY_WATCHDOG_S = 120.0
TOTAL_WATCHDOG_S = 300.0
FAULT_DUMP_GRACE_S = 5.0
TERM_GRACE_S = 30.0
GPU_SAMPLE_PERIOD_S = 1.0
SUPERVISOR_PID_ENV = "D353_SUPERVISOR_PID"
WORKER_LAUNCH_TOKEN_ENV = "D353_WORKER_LAUNCH_TOKEN"

NEW_OPERATIONAL_VARIABLES = ["explicit_timeline_commit_after_pause"]
NEW_SCIENTIFIC_VARIABLES: list[str] = []
NEW_PHYSICAL_VARIABLES: list[str] = []

PARAMETER_PATH = OUT_DIR / "d353_parameter_freeze_audit.json"
GPU_HARDWARE_PATH = OUT_DIR / "d353_gpu_hardware_contract.json"
PREREG_PATH = OUT_DIR / "d353_preregistration.json"
SUPERVISOR_PREFLIGHT_PATH = OUT_DIR / "d353_supervisor_preflight.json"
WORKER_PREFLIGHT_PATH = OUT_DIR / "d353_validate_preflight.json"
MARKER_PATH = OUT_DIR / "d353_phase_markers.jsonl"
WORKER_LOG_PATH = OUT_DIR / "d353_worker_stdout_stderr.log"
GPU_TELEMETRY_PATH = OUT_DIR / "d353_gpu_cpu_telemetry.jsonl"
FAULT_PATH = OUT_DIR / "d353_faulthandler.txt"
LIVE_BINDING_PATH = OUT_DIR / "d353_live_topology_runtime_binding.json"
RAW_CONTRACT_PATH = OUT_DIR / "d353_raw_source_contract.json"
BRIDGE_PATH = OUT_DIR / "d353_zero_step_bridge_contract.json"
COMMIT_CONTRACT_PATH = OUT_DIR / "d353_timeline_commit_event_contract.json"
LOCALIZATION_PATH = OUT_DIR / "d353_timeline_commit_bridge_summary.json"
BRIDGE_ATTESTATION_PATH = OUT_DIR / "d353_timeline_commit_bridge_attestation.json"
RUNTIME_EXCEPTION_PATH = OUT_DIR / "d353_runtime_exception.json"
WATCHDOG_PROC_PATH = OUT_DIR / "d353_watchdog_proc_snapshot.json"
RAW_SUPERVISOR_PATH = OUT_DIR / "d353_inherited_supervisor_raw_audit.json"
SUPERVISOR_AUDIT_PATH = OUT_DIR / "d353_supervisor_audit.json"

USER_SIDECAR_DIR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
USER_SIDECAR_INVENTORY = [
    "README.md",
    "d334_collision_table_academic.html",
    "d334_collision_table_academic.png",
]

D352_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d352"
D352_IMMUTABLE_HASHES = {
    "sim_scripts/cyl34_top_view_d352_d351_validate_phase_localization_watchdog.py": (
        "3f770200d3ca729f106a74ba9e22390d4db5fdd4547e7091cf36ad46a59f65d0"
    ),
    "claudedocs/session_20260715_grasp_g0a_d352_d351_validate_phase_localization_watchdog.md": (
        "19e67399cf73c2379eb7fc8afba8a28c0bb95665bb4572bd10171a058a20fb29"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_faulthandler.txt": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_gpu_cpu_telemetry.jsonl": (
        "04dc29f902afc1ef3f142d424d0a98918defc735aa70ce5cfb698824b5684a2c"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_gpu_hardware_contract.json": (
        "77770159639b7ab2b0b2a9aae3e109a95564e51d93e2cd491c9259a706b15696"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_live_topology_runtime_binding.json": (
        "8e607e1248c0670a7bed7ab65ccffd0b94ecd506259cd445efe0f7d09e366afb"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_localization_summary.json": (
        "2548cadc18b098680b8c5500237e4e363258df99ef96bd29d8327f0955c47d60"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_parameter_freeze_audit.json": (
        "3c41109a92398fa9b5f19cb22cb07095a52863686f6f628b57a6bca34e3bb39e"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_phase_markers.jsonl": (
        "09371d795a3e0214e3ddf335e7ff4a9bf78955c5b51ad85048e7b1700de389ce"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_postrun_classification_audit.json": (
        "92c186a7a4175101e7a3890f6bedf4cb6125bc5a78f13f38b79004a9b6035594"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_preregistration.json": (
        "37843ebbf2708cd664bfa2b0b418ccad76e681da401c90a0d098e18a54430e8f"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_raw_source_contract.json": (
        "325004fdc98f01bc01e5534d96ce1e2abe410b47d21029f5961446f2b53f243b"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_supervisor_audit.json": (
        "b1bacc589d63d5dff60746c4030635ad100f9a86b701598c1dece159004f70be"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_supervisor_preflight.json": (
        "d49bca00f90ff8230506613a73b6b97988d5b7ac3d556a36e6cbf9a2a6942285"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_validate_preflight.json": (
        "98942f35dd51ac67539e2bb866beb4b704985e2cf3fc20031603984edc5852bc"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_worker_stdout_stderr.log": (
        "16e3e0bdfc01dc8203d3a7a89173e9635d6da442ee708d59535ab2512293de77"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d352/d352_zero_step_bridge_contract.json": (
        "26a05d8c76ceaf83a0ebf57324b50c7853d38ce2e6bd58c25e4484c13f9a0036"
    ),
}
D352_ROOT_INVENTORY = sorted(
    [
        "d352_faulthandler.txt",
        "d352_gpu_cpu_telemetry.jsonl",
        "d352_gpu_hardware_contract.json",
        "d352_live_topology_runtime_binding.json",
        "d352_localization_summary.json",
        "d352_parameter_freeze_audit.json",
        "d352_phase_markers.jsonl",
        "d352_postrun_classification_audit.json",
        "d352_preregistration.json",
        "d352_raw_source_contract.json",
        "d352_supervisor_audit.json",
        "d352_supervisor_preflight.json",
        "d352_validate_preflight.json",
        "d352_worker_stdout_stderr.log",
        "d352_zero_step_bridge_contract.json",
    ]
)

TIMELINE_ROOT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.timeline-1.0.14+69cbf6ad.lx64.r.cp311"
)
CARB_EVENTS_STUB = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/"
    "isaacsim/kit/kernel/py/carb/events/_events.pyi"
)
TIMELINE_API_HASHES = {
    TIMELINE_ROOT / "docs/USAGE_PYTHON.md": (
        "3e3ad641192893c47d9bf09c73b3a1f220dfbefb5b8f754fc9b83c8e3f85a6dd"
    ),
    TIMELINE_ROOT / "docs/FRAME_INTEGRITY.md": (
        "1db84fba636fa743bcf98a38561132323ad13fcb89dc91629a06812b123b2e37"
    ),
    TIMELINE_ROOT / "docs/TIME_STEPPING.md": (
        "8741959b4d2d80cd0d5296a5d6ab20aa0bb659ef5b04b4aa6abb5ad3edd61059"
    ),
    TIMELINE_ROOT / "omni/timeline/_timeline.pyi": (
        "c5a431d83c24de23aefca0912ef819ae2f3322418264b81aba5279d4fe4ac35e"
    ),
    TIMELINE_ROOT / "omni/timeline/__init__.py": (
        "4657e8a3cc5f1e8214a131f21f2f065e526de2d421b126ea144d69c3f51838a0"
    ),
    TIMELINE_ROOT / "omni/timeline/tests/tests.py": (
        "570b36d310e3f3a307c8a35c38ba277da051e6e1e8fc25da6889e794ad270638"
    ),
    TIMELINE_ROOT / "config/extension.toml": (
        "3258f271440bb2f5240f5e222f25cd2e90ed4c2ecc86a4dd3239c9ad344e781c"
    ),
    CARB_EVENTS_STUB: (
        "3dd85850696acfbbad92f7df19557df8401500bf4cb5b3151252de55ae925c55"
    ),
}

_ORIGINAL_WRITE_JSON = d352._write_json
_ORIGINAL_IMMUTABLE_D351 = d352._immutable_d351_contract
_ORIGINAL_GPU_HARDWARE = d352._gpu_hardware_contract
_ORIGINAL_SAMPLE_GPU_CPU = d352._sample_gpu_cpu
_ORIGINAL_SUPERVISOR_PREFLIGHT = d352._supervisor_preflight
_ORIGINAL_WORKER_PREFLIGHT = d352._worker_preflight
_ORIGINAL_RUN_SUPERVISOR = d352._run_supervisor
_ORIGINAL_RUN_WORKER = d352._run_worker
_ORIGINAL_USER_SIDECAR = d352._user_sidecar_contract

_CONFIGURED = False
_COMMIT_CALL_COUNT = 0
_COMMIT_ATTEMPT_COUNT = 0
_D353_BRIDGE_COMPLETE = False
_TIMELINE_EVENT_ROWS: list[dict[str, Any]] = []
_TIMELINE_EVENT_TYPE_NAMES: dict[int, str] = {}
_ACTIVE_COMMIT_PHASE: str | None = None
_TIMELINE: Any | None = None
_TIMELINE_SUBSCRIPTION: Any | None = None


def _replace_artifact(payload: Any, artifact: str | None = None) -> Any:
    row = copy.deepcopy(payload)
    if not isinstance(row, dict):
        return row
    if artifact is not None:
        row["artifact"] = artifact
    elif isinstance(row.get("artifact"), str) and row["artifact"].startswith("D352_"):
        row["artifact"] = "D353_" + row["artifact"][len("D352_") :]
    return row


def _write_json(path: Path, payload: Any) -> None:
    artifact = None
    if path == RAW_SUPERVISOR_PATH:
        artifact = "D353_INHERITED_D352_SUPERVISOR_RAW_V1"
    row = _replace_artifact(payload, artifact)
    if path == RUNTIME_EXCEPTION_PATH and isinstance(row, dict):
        inherited_q5 = int(row.pop("d352_q5_evaluation_count", 0) or 0)
        row.pop("d352_controlled_physics_steps", None)
        row.update(
            {
                "artifact": "D353_RUNTIME_EXCEPTION_STOP_V1",
                "d353_q5_science_evaluation_count": 0,
                "d353_q5_science_state_write_count": 0,
                "d353_q5_trap_invocation_count": inherited_q5,
                "d353_controlled_physics_steps": (
                    0 if _D353_BRIDGE_COMPLETE else None
                ),
                "inherited_d351_attempt2_q5_evaluation_count": 0,
                "inherited_d351_attempt2_controlled_physics_steps": None,
                "inherited_d352_q5_evaluation_count": 0,
                "inherited_d352_controlled_physics_steps": None,
            }
        )
    _ORIGINAL_WRITE_JSON(path, row)


def _gpu_hardware_contract() -> dict[str, Any]:
    return _replace_artifact(_ORIGINAL_GPU_HARDWARE(), "D353_GPU_HARDWARE_CONTRACT_V1")


def _sample_gpu_cpu(
    worker_pid: int,
    sample_index: int,
    worker_process: Any | None,
) -> dict[str, Any]:
    row = _ORIGINAL_SAMPLE_GPU_CPU(worker_pid, sample_index, worker_process)
    return _replace_artifact(row, "D353_GPU_CPU_TELEMETRY_SAMPLE_V1")


def _user_sidecar_contract() -> dict[str, Any]:
    row = copy.deepcopy(_ORIGINAL_USER_SIDECAR())
    inventory = sorted(
        path.name for path in USER_SIDECAR_DIR.iterdir() if path.is_file()
    )
    non_file_inventory = sorted(
        path.name for path in USER_SIDECAR_DIR.iterdir() if not path.is_file()
    )
    row["root_inventory"] = inventory
    row["expected_root_inventory"] = USER_SIDECAR_INVENTORY
    row["non_file_inventory"] = non_file_inventory
    row["checks"]["root_inventory_exact"] = inventory == USER_SIDECAR_INVENTORY
    row["checks"]["non_file_inventory_empty"] = not non_file_inventory
    row["pass"] = all(row["checks"].values())
    return row


def _marker(actor: str, phase: str, event: str, details: Any | None = None) -> dict[str, Any]:
    d352._MARKER_SEQUENCE += 1
    row = {
        "artifact": "D353_DURABLE_PHASE_MARKER_V1",
        "run_nonce": d352._RUN_NONCE,
        "actor": actor,
        "pid": os.getpid(),
        "sequence": d352._MARKER_SEQUENCE,
        "phase": phase,
        "event": event,
        "wall_time_ns": time.time_ns(),
        "wall_time_iso": dt.datetime.now().astimezone().isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "process_elapsed_s": (
            time.monotonic_ns() - d352._PROCESS_START_MONOTONIC_NS
        )
        / 1.0e9,
        "q5_trap_invocation_count": d352._Q5_TRAP_INVOCATION_COUNT,
        "timeline_commit_attempt_count": _COMMIT_ATTEMPT_COUNT,
        "timeline_commit_call_count": _COMMIT_CALL_COUNT,
        "details": details,
    }
    d352._append_jsonl(MARKER_PATH, row)
    return row


def _immutable_prior_contract() -> dict[str, Any]:
    d351_contract = _ORIGINAL_IMMUTABLE_D351()
    rows: dict[str, Any] = {}
    for relative, expected in D352_IMMUTABLE_HASHES.items():
        path = REPO / relative
        rows[relative] = {
            "exists": path.is_file(),
            "sha256": d352._sha(path) if path.is_file() else None,
            "expected_sha256": expected,
        }
    inventory = sorted(path.name for path in D352_DIR.iterdir() if path.is_file())
    d352_checks = {
        "all_hashes_exact": all(
            row["exists"] and row["sha256"] == row["expected_sha256"]
            for row in rows.values()
        ),
        "root_inventory_exact_15": inventory == D352_ROOT_INVENTORY,
    }
    d352_contract = {
        "rows": rows,
        "root_inventory": inventory,
        "expected_root_inventory": D352_ROOT_INVENTORY,
        "checks": d352_checks,
        "pass": all(d352_checks.values()),
    }
    return {
        "d351": d351_contract,
        "d352": d352_contract,
        "pass": d351_contract["pass"] and d352_contract["pass"],
    }


def _frozen_run_hash_contract() -> dict[str, Any]:
    prereg = d352._json(PREREG_PATH) if PREREG_PATH.is_file() else {}
    current_state_hashes = {
        "start_here": d352._sha(START_HERE),
        "session_doc": d352._sha(SESSION_DOC),
    }
    checks = {
        "prereg_available_and_pass": prereg.get("pass") is True,
        "harness_hash_exact": HARNESS.is_file()
        and d352._sha(HARNESS) == prereg.get("harness_sha256"),
        "state_hashes_exact": current_state_hashes == prereg.get("state_hashes"),
        "parameter_hash_exact": PARAMETER_PATH.is_file()
        and d352._sha(PARAMETER_PATH) == prereg.get("parameter_sha256"),
        "gpu_hash_exact": GPU_HARDWARE_PATH.is_file()
        and d352._sha(GPU_HARDWARE_PATH) == prereg.get("gpu_hardware_sha256"),
        "timeline_api_exact": _timeline_api_contract() == prereg.get("timeline_api"),
        "d351_d352_immutable": _immutable_prior_contract()["pass"],
        "user_sidecar_read_only_exact_inventory": _user_sidecar_contract()["pass"],
    }
    return {
        "harness_sha256": d352._sha(HARNESS) if HARNESS.is_file() else None,
        "state_hashes": current_state_hashes,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _timeline_api_contract() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for path, expected in TIMELINE_API_HASHES.items():
        rows[str(path)] = {
            "exists": path.is_file(),
            "sha256": d352._sha(path) if path.is_file() else None,
            "expected_sha256": expected,
        }
    usage = (TIMELINE_ROOT / "docs/USAGE_PYTHON.md").read_text(encoding="utf-8")
    frame = (TIMELINE_ROOT / "docs/FRAME_INTEGRITY.md").read_text(encoding="utf-8")
    stub = (TIMELINE_ROOT / "omni/timeline/_timeline.pyi").read_text(encoding="utf-8")
    init = (TIMELINE_ROOT / "omni/timeline/__init__.py").read_text(encoding="utf-8")
    tests = (TIMELINE_ROOT / "omni/timeline/tests/tests.py").read_text(encoding="utf-8")
    carb_events_stub = CARB_EVENTS_STUB.read_text(encoding="utf-8")
    checks = {
        "all_hashes_exact": all(
            row["exists"] and row["sha256"] == row["expected_sha256"]
            for row in rows.values()
        ),
        "usage_deferred_next_frame": "state changes are applied in the next frame" in usage,
        "frame_commit_immediate": "`commit()` function" in frame
        and "applies changes instantly" in frame,
        "frame_commit_main_thread_caution": "not thread-safe" in frame
        and "calling thread" in frame,
        "stub_commit_callbacks": "Applies all pending state changes and invokes all callbacks" in stub,
        "stub_commit_silently_distinct": "does not invoke any callbacks" in stub,
        "get_timeline_returns_timeline": "return get_timeline_interface.timeline.get_timeline" in init,
        "tests_pause_commit_discriminator": "self._timeline.pause()" in tests
        and "self._timeline.commit()" in tests
        and "self.assertFalse(self._timeline.is_stopped())" in tests,
        "stub_tentative_time_getter_present": "def get_tentative_time(" in stub,
        "carb_pop_subscription_signature_pinned": (
            "def create_subscription_to_pop(" in carb_events_stub
            and "fn: typing.Callable[[IEvent], None]" in carb_events_stub
        ),
    }
    return {
        "artifact": "D353_TIMELINE_API_CONTRACT_V1",
        "extension": "omni.timeline-1.0.14+69cbf6ad.lx64.r.cp311",
        "rows": rows,
        "semantics": {
            "pause_request": "pending until next frame or explicit commit",
            "commit": "apply all pending state changes and invoke callbacks on calling thread",
            "commit_silently": "forbidden because callbacks would be suppressed",
            "forward_one_frame": "forbidden because it advances a frame",
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _parameter_freeze_audit() -> dict[str, Any]:
    scope = {
        "asset_write_or_recook": False,
        "decomposition_change": False,
        "fresh_property_or_cook_query": False,
        "q0_q5_or_object_pose_change": False,
        "q5_science_sample": False,
        "moving_surface_measurement": False,
        "geometry_or_current_pose_verdict": None,
        "target_ik_or_path_change": False,
        "gate_or_tolerance_change": False,
        "material_mass_actuator_physics_solver_or_renderer_change": False,
        "authorized_timeline_commit_after_existing_pause": True,
        "other_new_playback_or_timeline_control": False,
        "controlled_physics_steps_planned": 0,
        "simulation_app_or_kit_update": False,
        "forward_or_rewind_one_frame": False,
        "commit_silently": False,
        "settle": False,
        "ten_trial": False,
        "g0b": False,
        "rl_or_ppo": False,
        "vla": False,
        "ladder_promotion": False,
        "g0a_pass": False,
    }
    checks = {
        "operational_variable_exact_one": NEW_OPERATIONAL_VARIABLES
        == ["explicit_timeline_commit_after_pause"],
        "scientific_variables_zero": NEW_SCIENTIFIC_VARIABLES == [],
        "physical_variables_zero": NEW_PHYSICAL_VARIABLES == [],
        "seed_frozen": SEED == 33201,
        "watchdogs_frozen_from_d352": INACTIVITY_WATCHDOG_S == 120.0
        and TOTAL_WATCHDOG_S == 300.0
        and FAULT_DUMP_GRACE_S == 5.0
        and TERM_GRACE_S == 30.0,
        "gpu_telemetry_frozen_from_d352": GPU_SAMPLE_PERIOD_S == 1.0,
        "only_authorized_true_scope_flag": not any(
            value is True
            for key, value in scope.items()
            if key not in {"authorized_timeline_commit_after_existing_pause"}
        ),
        "planned_zero_not_result_authority": scope["controlled_physics_steps_planned"] == 0,
    }
    return {
        "artifact": "D353_PARAMETER_FREEZE_AUDIT_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "new_operational_variables": NEW_OPERATIONAL_VARIABLES,
        "new_scientific_variables": NEW_SCIENTIFIC_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "inherited_controls": {
            "durable_phase_marker_stream": (
                "D352 append/fsync mechanism; preregistered D353 read-only commit fields"
            ),
            "external_bounded_wall_clock_watchdog": "D352 frozen 120s/300s",
            "gpu_cpu_telemetry": "D352 frozen informational observer",
        },
        "read_only_measurement_channels": [
            "timeline all-event callback delta",
            "timeline metadata exact snapshots",
            "custom counter and SimulationContext clock",
            "joint/object mutation sentinels",
        ],
        "watchdog": {
            "marker_inactivity_s": INACTIVITY_WATCHDOG_S,
            "total_worker_wall_clock_s": TOTAL_WATCHDOG_S,
            "faulthandler_signal": "SIGUSR1",
            "faulthandler_grace_s": FAULT_DUMP_GRACE_S,
            "termination_signal": "SIGTERM",
            "termination_grace_s": TERM_GRACE_S,
            "escalation_if_still_alive": "SIGKILL",
            "automatic_retry": False,
        },
        "commit_application": {
            "registered_boundaries": [
                "after_reset_initial_pause",
                "after_live_topology_binding",
                "after_raw_shape_binding_before_prerequisites",
            ],
            "call_rule": "one commit only when the inherited boundary issued pause request(s)",
            "already_paused_rule": "no redundant pause and no bare commit",
            "minimum_discriminating_transitions": 1,
            "main_thread_only": True,
        },
        "result_semantics": {
            "full_bridge": "controlled_physics_steps=0",
            "incomplete_or_failed_bridge": "controlled_physics_steps=null",
            "authority_sequence": [
                "summary O_EXCL/fsync with candidate only",
                "summary exact reread",
                "separate attestation O_EXCL/fsync with authoritative result",
            ],
            "historical_d351_d352_values_unchanged": True,
            "all_outcomes": {
                "scientific_verdict": None,
                "geometry_pass_or_fail": None,
                "current_pose_support_or_rejection": None,
                "target_ik_repair_justification": None,
                "g0a_pass": False,
            },
        },
        "rerun_omission": (
            "Timeline transaction and exact mutation-sentinel verdict only; no geometry, pose, "
            "contact, trajectory, or synchronized sensor-time interpretation"
        ),
        "scope_guards": scope,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _ast_scope_contract() -> dict[str, Any]:
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"), filename=str(HARNESS))
    call_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    calls = [d352._call_name(node.func) for node in call_nodes]
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    imports.extend(
        str(node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden_exact = {
        "d352.d351._evaluate_q5",
        "d352.d351._run_validate",
        "d352.d351a2._ORIGINAL_EVALUATE_Q5",
        "d352.d351a2._evaluate_q5_counted",
        "d352.d351a2._run_validate_with_reactive_repairs",
        "simulation_app.update",
        "inner.step",
        "inner.sim.step",
        "inner.sim.render",
        "timeline.commit_silently",
        "timeline.forward_one_frame",
        "timeline.rewind_one_frame",
        "d352.d351._write_exact_state",
    }
    found = sorted(set(calls) & forbidden_exact)
    forbidden_fragments = (
        "next_update",
        "forward_one_frame",
        "rewind_one_frame",
        "commit_silently",
        "moving_surface",
        "evaluate_q5",
        "write_exact_state",
    )
    fragment_hits = sorted(
        name
        for name in set(calls)
        if any(fragment in name for fragment in forbidden_fragments)
    )
    forbidden_method_names = {
        "play",
        "stop",
        "forward_one_frame",
        "rewind_one_frame",
        "commit_silently",
        "set_current_time",
        "set_auto_update",
        "set_looping",
        "set_director",
        "set_start_time",
        "set_end_time",
        "set_time_codes_per_second",
        "set_target_framerate",
        "set_ticks_per_frame",
        "set_fast_mode",
        "set_play_every_frame",
        "set_prerolling",
        "set_zoom",
        "step",
        "render",
        "write_joint_state_to_sim",
        "write_root_state_to_sim",
        "write_root_pose_to_sim",
        "write_root_velocity_to_sim",
        "set_joint_position_target",
        "set_joint_velocity_target",
        "set_joint_effort_target",
        "set_world_pose",
        "set_local_pose",
    }
    forbidden_method_hits = sorted(
        {
            d352._call_name(node.func)
            for node in call_nodes
            if isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_method_names
        }
    )
    update_alias_hits = sorted(
        {
            d352._call_name(node.func)
            for node in call_nodes
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and d352._call_name(node.func)
            in {
                "simulation_app.update",
                "app.update",
                "kit_app.update",
                "world.update",
                "sim.update",
                "inner.update",
                "inner.sim.update",
            }
        }
    )
    pause_method_count = sum(
        isinstance(node.func, ast.Attribute) and node.func.attr == "pause"
        for node in call_nodes
    )
    commit_method_count = sum(
        isinstance(node.func, ast.Attribute) and node.func.attr == "commit"
        for node in call_nodes
    )
    checks = {
        "forbidden_exact_calls_absent": not found,
        "forbidden_fragment_calls_absent": not fragment_hits,
        "forbidden_control_physics_state_methods_absent": not forbidden_method_hits,
        "forbidden_app_update_aliases_absent": not update_alias_hits,
        "explicit_timeline_commit_source_call_site_exact_one": calls.count("_TIMELINE.commit") == 1,
        "all_commit_method_calls_accounted_exact_one": commit_method_count == 1,
        "initial_timeline_pause_source_call_site_exact_one": calls.count("_TIMELINE.pause") == 1,
        "all_pause_method_calls_accounted_exact_one": pause_method_count == 1,
        "inherited_pause_helper_source_call_site_exact_two": calls.count(
            "d352.d351a2._pause_without_update"
        )
        == 2,
        "q5_grid_generation_absent": "np.linspace" not in calls,
        "viewer_rerun_import_absent": not any(
            name == "rerun" or name.startswith("rerun.") for name in imports
        ),
        "hardware_control_absent": not any(
            name.endswith(("joints_angle_ctrl", "torque_set", "move_init"))
            for name in calls
        ),
    }
    return {
        "artifact": "D353_AST_SCOPE_CONTRACT_V1",
        "new_intervention_call_site": "timeline.commit",
        "timeline_commit_call_site_count": calls.count("_TIMELINE.commit"),
        "inherited_pause_helper_call_site_count": calls.count(
            "d352.d351a2._pause_without_update"
        ),
        "found_forbidden_calls": found,
        "found_forbidden_fragment_calls": fragment_hits,
        "found_forbidden_method_calls": forbidden_method_hits,
        "found_forbidden_update_alias_calls": update_alias_hits,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run_prepare(args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"forward-only D353 output path already exists: {OUT_DIR}")
    status = d352.d351._git_status()
    parameter = _parameter_freeze_audit()
    gpu = _gpu_hardware_contract()
    immutable = _immutable_prior_contract()
    sidecar = d352._user_sidecar_contract()
    line_map = d352._live_trace_line_map()
    ast_scope = _ast_scope_contract()
    timeline_api = _timeline_api_contract()
    inputs = d352.d351._input_hashes()
    start_text = START_HERE.read_text(encoding="utf-8")
    session_text = SESSION_DOC.read_text(encoding="utf-8") if SESSION_DOC.is_file() else ""
    prechecks = {
        "head_exact": d352._git_head() == EXPECTED_HEAD,
        "origin_master_exact": d352._git_head("origin/master") == EXPECTED_HEAD,
        "git_scope_only_d353": d352._status_scope_pass(status),
        "start_here_active_case_exact": "Active Case: D353" in start_text
        and CASE_NAME in start_text
        and "claudedocs/runtime_logs/grasp_track/g0a_d353/" in start_text,
        "start_here_git_truth_repaired": EXPECTED_HEAD in start_text
        and "prior `c2cfa5f...`" in start_text,
        "session_doc_exists": SESSION_DOC.is_file(),
        "session_declares_single_variable": all(
            value in session_text for value in NEW_OPERATIONAL_VARIABLES
        )
        and "신규 과학 변수는 `[]`" in session_text,
        "session_declares_q5_forbidden": "q5 science" in session_text
        and "별도" in session_text,
        "d351_inputs_exact": inputs == d352.d351.EXPECTED_INPUT_HASHES,
        "d351_d352_immutable": immutable["pass"],
        "user_sidecar_read_only_exact": sidecar["pass"],
        "gpu_hardware": gpu["pass"],
        "parameter_freeze": parameter["pass"],
        "trace_line_map": line_map["pass"],
        "ast_scope": ast_scope["pass"],
        "timeline_api": timeline_api["pass"],
    }
    if not all(prechecks.values()):
        raise RuntimeError(f"D353 prepare STOP: {prechecks}")

    _write_json(PARAMETER_PATH, parameter)
    _write_json(GPU_HARDWARE_PATH, gpu)
    run_nonce = secrets.token_hex(16)
    prereg = {
        "artifact": "D353_PREREGISTRATION_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "run_nonce": run_nonce,
        "git_head": d352._git_head(),
        "origin_master": d352._git_head("origin/master"),
        "git_status_before_prepare_outputs": status,
        "prepare_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "output_root": d352._rel(OUT_DIR),
        "harness_sha256": d352._sha(HARNESS),
        "state_hashes": {
            "start_here": d352._sha(START_HERE),
            "session_doc": d352._sha(SESSION_DOC),
        },
        "parameter_path": d352._rel(PARAMETER_PATH),
        "parameter_sha256": d352._sha(PARAMETER_PATH),
        "gpu_hardware_path": d352._rel(GPU_HARDWARE_PATH),
        "gpu_hardware_sha256": d352._sha(GPU_HARDWARE_PATH),
        "input_hashes": inputs,
        "prior_immutability": immutable,
        "user_sidecar": sidecar,
        "trace_line_map": line_map,
        "ast_scope": ast_scope,
        "timeline_api": timeline_api,
        "watchdog": parameter["watchdog"],
        "controlled_steps_authority": {
            "summary_path": d352._rel(LOCALIZATION_PATH),
            "attestation_path": d352._rel(BRIDGE_ATTESTATION_PATH),
            "summary_must_keep_current_value_null": True,
            "attestation_only_after_summary_fsync_and_exact_reread": True,
        },
        "single_effective_validate": True,
        "automatic_retry": False,
        "q5_science_authorized_inside_d353": False,
        "post_d353_q5_boundary": "requires result briefing and separate explicit approval",
        "prechecks": prechecks,
        "pass": all(prechecks.values()),
    }
    _write_json(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": True, "run_nonce": run_nonce}))
    return 0


def _supervisor_preflight(args: argparse.Namespace) -> dict[str, Any]:
    row = _replace_artifact(
        _ORIGINAL_SUPERVISOR_PREFLIGHT(args), "D353_SUPERVISOR_PREFLIGHT_V1"
    )
    prereg = d352._json(PREREG_PATH)
    extra = {
        "prior_d351_d352_immutable": _immutable_prior_contract()["pass"],
        "timeline_api_exact": _timeline_api_contract() == prereg.get("timeline_api"),
        "final_supervisor_artifact_absent": not SUPERVISOR_AUDIT_PATH.exists(),
        "commit_contract_absent": not COMMIT_CONTRACT_PATH.exists(),
        "authoritative_summary_absent": not LOCALIZATION_PATH.exists(),
        "bridge_attestation_absent": not BRIDGE_ATTESTATION_PATH.exists(),
        "runtime_exception_absent": not RUNTIME_EXCEPTION_PATH.exists(),
        "d353_bridge_authority_closed": not _D353_BRIDGE_COMPLETE,
    }
    row["checks"].update(extra)
    row["pass"] = all(row["checks"].values())
    return row


def _worker_preflight(args: argparse.Namespace) -> dict[str, Any]:
    row = _replace_artifact(_ORIGINAL_WORKER_PREFLIGHT(args), "D353_VALIDATE_PREFLIGHT_V1")
    prereg = d352._json(PREREG_PATH)
    extra = {
        "prior_d351_d352_immutable": _immutable_prior_contract()["pass"],
        "timeline_api_exact": _timeline_api_contract() == prereg.get("timeline_api"),
        "attempt2_saved_q5_alias_trapped": d352.d351a2._ORIGINAL_EVALUATE_Q5
        is _forbidden_q5,
        "commit_count_zero_before_launcher": _COMMIT_CALL_COUNT == 0,
        "commit_attempt_count_zero_before_launcher": _COMMIT_ATTEMPT_COUNT == 0,
        "d353_bridge_authority_closed": not _D353_BRIDGE_COMPLETE,
        "inherited_d352_bridge_authority_closed": not d352._BRIDGE_COMPLETE,
        "commit_contract_absent": not COMMIT_CONTRACT_PATH.exists(),
        "bridge_attestation_absent": not BRIDGE_ATTESTATION_PATH.exists(),
        "final_supervisor_artifact_absent": not SUPERVISOR_AUDIT_PATH.exists(),
    }
    row["checks"].update(extra)
    row["pass"] = all(row["checks"].values())
    return row


def _forbidden_q5(*_args: Any, **_kwargs: Any) -> Any:
    d352._Q5_TRAP_INVOCATION_COUNT += 1
    _marker(
        "worker",
        "q5_fail_closed_boundary",
        "breach_trapped",
        {"message": "both D351 q5 evaluator entry paths were trapped before science"},
    )
    raise RuntimeError("D353 q5 scope breach trapped before science")


def _float64_bits(value: float) -> str:
    return struct.pack("<d", float(value)).hex()


def _timeline_metadata(timeline: Any) -> dict[str, Any]:
    float_getters = {
        "start_time": timeline.get_start_time,
        "end_time": timeline.get_end_time,
        "tentative_time": timeline.get_tentative_time,
        "target_framerate": timeline.get_target_framerate,
        "ticks_per_second": timeline.get_ticks_per_second,
        "time_codes_per_second": timeline.get_time_codes_per_second,
        "zoom_start_time": timeline.get_zoom_start_time,
        "zoom_end_time": timeline.get_zoom_end_time,
    }
    return {
        "float64_bits": {
            name: _float64_bits(getter()) for name, getter in float_getters.items()
        },
        "current_tick": int(timeline.get_current_tick()),
        "ticks_per_frame": int(timeline.get_ticks_per_frame()),
        "fast_mode": bool(timeline.get_fast_mode()),
        "play_every_frame": bool(timeline.get_play_every_frame()),
        "auto_updating": bool(timeline.is_auto_updating()),
        "looping": bool(timeline.is_looping()),
        "prerolling": bool(timeline.is_prerolling()),
        "zoomed": bool(timeline.is_zoomed()),
        "director_none": timeline.get_director() is None,
    }


def _detailed_snapshot(inner: Any, phase: str) -> dict[str, Any]:
    if _TIMELINE is None:
        raise RuntimeError("D353 timeline snapshot requested before interface binding")
    row = d352.d351a2._bridge_snapshot(inner, phase)
    row.update(
        {
            "timeline_stopped": bool(_TIMELINE.is_stopped()),
            "timeline_time_float64_bits": _float64_bits(row["timeline_time"]),
            "simulation_context_clock_float64_bits": {
                "current_time": _float64_bits(
                    row["simulation_context_clock"]["current_time"]
                )
            },
            "timeline_metadata": _timeline_metadata(_TIMELINE),
            "q5_science_evaluation_count": 0,
            "q5_science_state_write_count": 0,
            "q5_trap_invocation_count": d352._Q5_TRAP_INVOCATION_COUNT,
            "timeline_commit_call_count": _COMMIT_CALL_COUNT,
            "thread": {
                "ident": threading.get_ident(),
                "name": threading.current_thread().name,
                "is_main_thread": threading.current_thread() is threading.main_thread(),
            },
        }
    )
    return row


def _canonical_snapshot(inner: Any, phase: str) -> dict[str, Any]:
    _marker("worker", "zero_step_bridge_snapshot", "start", {"name": phase})
    row = _detailed_snapshot(inner, phase)
    _marker(
        "worker",
        "zero_step_bridge_snapshot",
        "end",
        {
            "name": phase,
            "custom_step_counter": row["custom_step_counter"],
            "timeline_playing": row["timeline_playing"],
            "timeline_stopped": row["timeline_stopped"],
            "timeline_time": row["timeline_time"],
            "simulation_context_clock": row["simulation_context_clock"],
        },
    )
    return row


def _timeline_event_callback(event: Any) -> None:
    if _TIMELINE is None:
        return
    event_type = int(event.type)
    _TIMELINE_EVENT_ROWS.append(
        {
            "sequence": len(_TIMELINE_EVENT_ROWS),
            "phase": _ACTIVE_COMMIT_PHASE,
            "event_type": event_type,
            "event_name": _TIMELINE_EVENT_TYPE_NAMES.get(event_type, f"UNKNOWN_{event_type}"),
            "callback_monotonic_ns": time.monotonic_ns(),
            "callback_thread_ident": threading.get_ident(),
            "callback_thread_name": threading.current_thread().name,
            "callback_is_main_thread": threading.current_thread() is threading.main_thread(),
            "timeline_playing": bool(_TIMELINE.is_playing()),
            "timeline_stopped": bool(_TIMELINE.is_stopped()),
            "timeline_time_float64_bits": _float64_bits(_TIMELINE.get_current_time()),
            "timeline_current_tick": int(_TIMELINE.get_current_tick()),
            "q5_trap_invocation_count": d352._Q5_TRAP_INVOCATION_COUNT,
        }
    )


def _mutation_checks(rows: list[dict[str, Any]]) -> dict[str, bool]:
    first = rows[0] if rows else {}
    payloads = ("joint_float32", "object_position_float32", "object_quaternion_float32")
    return {
        "rows_present": bool(rows),
        "custom_step_counter_zero": bool(rows)
        and all(row.get("custom_step_counter") == 0 for row in rows),
        "timeline_time_float64_bits_exact": bool(rows)
        and all(
            row.get("timeline_time_float64_bits")
            == first.get("timeline_time_float64_bits")
            for row in rows
        ),
        "simulation_context_clock_exact": bool(rows)
        and all(
            row.get("simulation_context_clock") == first.get("simulation_context_clock")
            for row in rows
        ),
        "simulation_context_time_float64_bits_exact": bool(rows)
        and all(
            row.get("simulation_context_clock_float64_bits")
            == first.get("simulation_context_clock_float64_bits")
            for row in rows
        ),
        "timeline_metadata_exact": bool(rows)
        and all(
            row.get("timeline_metadata") == first.get("timeline_metadata") for row in rows
        ),
        "joint_object_float32_bits_exact": bool(rows)
        and all(
            row.get(name, {}).get("bits_hex")
            == first.get(name, {}).get("bits_hex")
            for row in rows
            for name in payloads
        ),
        "play_simulations_false": bool(rows)
        and all(
            row.get("play_simulations_setting", {}).get("readable") is True
            and row.get("play_simulations_setting", {}).get("value") is False
            for row in rows
        ),
        "geometry_evaluation_count_zero": bool(rows)
        and all(row.get("geometry_evaluation_call_count") == 0 for row in rows),
        "q5_science_count_zero": bool(rows)
        and all(row.get("q5_science_evaluation_count") == 0 for row in rows),
        "q5_science_state_write_count_zero": bool(rows)
        and all(row.get("q5_science_state_write_count") == 0 for row in rows),
        "q5_trap_count_zero": bool(rows)
        and all(row.get("q5_trap_invocation_count") == 0 for row in rows),
        "main_thread_snapshots": bool(rows)
        and all(row.get("thread", {}).get("is_main_thread") is True for row in rows),
        "director_none": bool(rows)
        and all(row.get("timeline_metadata", {}).get("director_none") is True for row in rows),
    }


def _matched_d352_initial_snapshot_contract(
    current_pending: dict[str, Any],
) -> dict[str, Any]:
    source_path = D352_DIR / "d352_zero_step_bridge_contract.json"
    source = d352._json(source_path)
    baseline = source.get("snapshots", [])[0] if source.get("snapshots") else {}
    payloads = ("joint_float32", "object_position_float32", "object_quaternion_float32")
    baseline_clock = baseline.get("simulation_context_clock", {})
    current_clock = current_pending.get("simulation_context_clock", {})
    checks = {
        "source_hash_frozen_exact": d352._sha(source_path)
        == D352_IMMUTABLE_HASHES[
            "claudedocs/runtime_logs/grasp_track/g0a_d352/"
            "d352_zero_step_bridge_contract.json"
        ],
        "d352_first_snapshot_was_pending_play": baseline.get("timeline_playing") is True,
        "d353_pending_is_same_play_not_stop": current_pending.get("timeline_playing") is True
        and current_pending.get("timeline_stopped") is False,
        "custom_step_counter_exact": current_pending.get("custom_step_counter")
        == baseline.get("custom_step_counter")
        == 0,
        "geometry_count_exact_zero": current_pending.get("geometry_evaluation_call_count")
        == baseline.get("geometry_evaluation_call_count")
        == 0,
        "timeline_time_float64_bits_exact": _float64_bits(
            current_pending.get("timeline_time")
        )
        == _float64_bits(baseline.get("timeline_time")),
        "simulation_context_clock_exact": current_clock == baseline_clock,
        "simulation_context_current_time_float64_bits_exact": _float64_bits(
            current_clock.get("current_time")
        )
        == _float64_bits(baseline_clock.get("current_time")),
        "play_simulations_setting_exact": current_pending.get(
            "play_simulations_setting"
        )
        == baseline.get("play_simulations_setting")
        == {"readable": True, "value": False, "error": None},
        "joint_object_float32_bits_exact": all(
            current_pending.get(name, {}).get("bits_hex")
            == baseline.get(name, {}).get("bits_hex")
            for name in payloads
        ),
    }
    return {
        "source_path": d352._rel(source_path),
        "source_sha256": d352._sha(source_path),
        "baseline_phase": baseline.get("phase"),
        "current_phase": current_pending.get("phase"),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _assert_boundary_preconditions(snapshot: dict[str, Any], phase: str) -> None:
    if snapshot.get("thread", {}).get("is_main_thread") is not True:
        raise RuntimeError(
            f"D353 timeline commit precondition STOP at {phase}: not main thread"
        )
    if snapshot.get("timeline_metadata", {}).get("director_none") is not True:
        raise RuntimeError(
            f"D353 timeline commit precondition STOP at {phase}: director is not None"
        )
    setting = snapshot.get("play_simulations_setting", {})
    if setting.get("readable") is not True or setting.get("value") is not False:
        raise RuntimeError(
            f"D353 timeline commit precondition STOP at {phase}: "
            "/app/player/playSimulations is not exact readable false"
        )


def _commit_pending_pause(
    inner: Any,
    phase: str,
    before: dict[str, Any],
    pending: dict[str, Any],
    pause_interventions: int,
    callback_start_index: int,
    inherited_pause_event: dict[str, Any],
) -> dict[str, Any]:
    global _ACTIVE_COMMIT_PHASE, _COMMIT_ATTEMPT_COUNT, _COMMIT_CALL_COUNT

    if _TIMELINE is None:
        raise RuntimeError("D353 commit requested before timeline binding")
    _assert_boundary_preconditions(before, f"{phase}:before_pause")
    _assert_boundary_preconditions(pending, f"{phase}:post_pause_pre_commit")

    commit_called = False
    commit_return_is_none: bool | None = None
    commit_start_monotonic_ns: int | None = None
    commit_end_monotonic_ns: int | None = None
    commit_caller_thread_ident: int | None = None
    callbacks_before_commit = copy.deepcopy(
        _TIMELINE_EVENT_ROWS[callback_start_index:]
    )
    if callbacks_before_commit:
        raise RuntimeError(
            f"D353 timeline commit precondition STOP at {phase}: "
            "timeline callback arrived before explicit commit"
        )
    direct_playing = bool(_TIMELINE.is_playing())
    direct_stopped = bool(_TIMELINE.is_stopped())
    if pause_interventions > 0:
        pending_pause_exact = (
            before.get("timeline_playing") is True
            and before.get("timeline_stopped") is False
            and pending.get("timeline_playing") is True
            and pending.get("timeline_stopped") is False
            and direct_playing is True
            and direct_stopped is False
        )
        if not pending_pause_exact:
            raise RuntimeError(
                f"D353 timeline commit precondition STOP at {phase}: "
                "requested PAUSE is not the exact still-pending PLAY/not-STOP state"
            )
    else:
        already_paused_exact = (
            before.get("timeline_playing") is False
            and before.get("timeline_stopped") is False
            and pending.get("timeline_playing") is False
            and pending.get("timeline_stopped") is False
            and direct_playing is False
            and direct_stopped is False
        )
        if not already_paused_exact:
            raise RuntimeError(
                f"D353 timeline commit precondition STOP at {phase}: "
                "zero-request boundary is not exact PAUSE/not-STOP no-op"
            )
    if pause_interventions > 0:
        _marker(
            "worker",
            "timeline_commit",
            "start",
            {"phase": phase, "pause_interventions": pause_interventions},
        )
        _COMMIT_ATTEMPT_COUNT += 1
        commit_caller_thread_ident = threading.get_ident()
        _ACTIVE_COMMIT_PHASE = phase
        commit_start_monotonic_ns = time.monotonic_ns()
        try:
            result = _TIMELINE.commit()
        except Exception as error:
            _marker(
                "worker",
                "timeline_commit",
                "error",
                {
                    "phase": phase,
                    "commit_attempt_count": _COMMIT_ATTEMPT_COUNT,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        finally:
            commit_end_monotonic_ns = time.monotonic_ns()
            _ACTIVE_COMMIT_PHASE = None
        _COMMIT_CALL_COUNT += 1
        commit_called = True
        commit_return_is_none = result is None
        _marker(
            "worker",
            "timeline_commit",
            "end",
            {
                "phase": phase,
                "commit_attempt_count": _COMMIT_ATTEMPT_COUNT,
                "commit_call_count": _COMMIT_CALL_COUNT,
            },
        )

    after = _detailed_snapshot(inner, f"{phase}_post_commit")
    callbacks = copy.deepcopy(_TIMELINE_EVENT_ROWS[callback_start_index:])
    mutation = _mutation_checks([before, pending, after])
    intervention_expected = pause_interventions > 0
    if intervention_expected:
        transition = (
            before["timeline_playing"] is True
            and before["timeline_stopped"] is False
            and pending["timeline_playing"] is True
            and pending["timeline_stopped"] is False
            and after["timeline_playing"] is False
            and after["timeline_stopped"] is False
        )
        callback_exact = (
            1 <= len(callbacks) <= pause_interventions
            and (pause_interventions != 1 or len(callbacks) == 1)
            and all(row["event_type"] == 1 for row in callbacks)
            and all(row["event_name"] == "PAUSE" for row in callbacks)
            and all(row["phase"] == phase for row in callbacks)
            and all(row["callback_is_main_thread"] is True for row in callbacks)
            and all(
                row["callback_thread_ident"] == commit_caller_thread_ident
                for row in callbacks
            )
            and all(row["timeline_playing"] is False for row in callbacks)
            and all(row["timeline_stopped"] is False for row in callbacks)
            and all(
                row["timeline_time_float64_bits"]
                == before["timeline_time_float64_bits"]
                for row in callbacks
            )
            and all(
                row["timeline_current_tick"]
                == before["timeline_metadata"]["current_tick"]
                for row in callbacks
            )
            and commit_start_monotonic_ns is not None
            and commit_end_monotonic_ns is not None
            and all(
                commit_start_monotonic_ns
                <= row["callback_monotonic_ns"]
                <= commit_end_monotonic_ns
                for row in callbacks
            )
        )
    else:
        transition = (
            before["timeline_playing"] is False
            and before["timeline_stopped"] is False
            and pending["timeline_playing"] is False
            and pending["timeline_stopped"] is False
            and after["timeline_playing"] is False
            and after["timeline_stopped"] is False
        )
        callback_exact = len(callbacks) == 0
    checks = {
        "pause_block_and_commit_pairing": commit_called is intervention_expected,
        "no_callback_before_explicit_commit": not callbacks_before_commit,
        "commit_return_none_when_called": (
            commit_return_is_none is True if intervention_expected else commit_return_is_none is None
        ),
        "pending_to_pause_or_already_paused_noop": transition,
        "callback_delta_exact": callback_exact,
        "mutation_sentinels_exact": all(mutation.values()),
        "post_boundary_pause_not_stop": after["timeline_playing"] is False
        and after["timeline_stopped"] is False,
        "q5_trap_zero": d352._Q5_TRAP_INVOCATION_COUNT == 0,
        "commit_attempt_success_pairing": (
            _COMMIT_ATTEMPT_COUNT == _COMMIT_CALL_COUNT
        ),
    }
    return {
        "phase": phase,
        "pause_interventions": pause_interventions,
        "commit_called": commit_called,
        "commit_attempt_count_after": _COMMIT_ATTEMPT_COUNT,
        "commit_call_count_after": _COMMIT_CALL_COUNT,
        "commit_window": {
            "caller_thread_ident": commit_caller_thread_ident,
            "start_monotonic_ns": commit_start_monotonic_ns,
            "end_monotonic_ns": commit_end_monotonic_ns,
        },
        "before_pause": before,
        "post_pause_pre_commit": pending,
        "post_commit": after,
        "timeline_callbacks": callbacks,
        "inherited_pause_event": inherited_pause_event,
        "mutation_checks": mutation,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run_commit_bridge(args: argparse.Namespace, launcher_report: dict[str, Any]) -> int:
    global _D353_BRIDGE_COMPLETE, _TIMELINE, _TIMELINE_SUBSCRIPTION

    import omni.timeline

    snapshots: list[dict[str, Any]] = []
    commit_events: list[dict[str, Any]] = []

    _marker("worker", "source_inventory", "start")
    source_before = d352.d351.d349._source_inventories()
    _marker("worker", "source_inventory", "end")
    _marker("worker", "input_hashes", "start")
    input_before = d352.d351._input_hashes()
    _marker("worker", "input_hashes", "end")

    args.robot_usd_path = d352.d351.VARIANT_ROBOT_USD
    _marker("worker", "_make_runtime_env", "start")
    inner = d352.d351.d333._make_runtime_env(args)
    d352._ACTIVE_INNER = inner
    _marker("worker", "_make_runtime_env", "end")
    _TIMELINE = omni.timeline.get_timeline_interface()
    event_types = (
        "PLAY",
        "PAUSE",
        "STOP",
        "CURRENT_TIME_CHANGED",
        "CURRENT_TIME_TICKED_PERMANENT",
        "CURRENT_TIME_TICKED",
        "LOOP_MODE_CHANGED",
        "START_TIME_CHANGED",
        "END_TIME_CHANGED",
        "TIME_CODE_PER_SECOND_CHANGED",
        "AUTO_UPDATE_CHANGED",
        "PREROLLING_CHANGED",
        "TENTATIVE_TIME_CHANGED",
        "TICKS_PER_FRAME_CHANGED",
        "FAST_MODE_CHANGED",
        "PLAY_EVERY_FRAME_CHANGED",
        "TARGET_FRAMERATE_CHANGED",
        "DIRECTOR_CHANGED",
        "ZOOM_CHANGED",
    )
    _TIMELINE_EVENT_TYPE_NAMES.clear()
    for name in event_types:
        value = getattr(omni.timeline.TimelineEventType, name, None)
        if value is not None:
            raw = value.value if hasattr(value, "value") else value
            event_value = int(raw)
            if event_value in _TIMELINE_EVENT_TYPE_NAMES:
                _TIMELINE_EVENT_TYPE_NAMES[event_value] += f"|{name}"
            else:
                _TIMELINE_EVENT_TYPE_NAMES[event_value] = name
    if _TIMELINE_EVENT_TYPE_NAMES.get(1) != "PAUSE":
        raise RuntimeError(
            "D353 timeline commit precondition STOP: runtime PAUSE enum is not exact value 1"
        )
    _marker("worker", "reset", "start")
    inner.reset(seed=SEED)
    inner.sim.set_setting("/app/player/playSimulations", False)
    # The observer window starts only after reset.  Reset's own already-completed
    # PLAY/tick traffic is baseline setup, not part of the commit callback delta.
    _TIMELINE_EVENT_ROWS.clear()
    _TIMELINE_SUBSCRIPTION = (
        _TIMELINE.get_timeline_event_stream().create_subscription_to_pop(
            _timeline_event_callback, name="D353 commit-only timeline observer"
        )
    )
    initial_before = _detailed_snapshot(inner, "after_reset_before_initial_pause")
    _assert_boundary_preconditions(initial_before, "after_reset_initial_pause:before")
    callback_start = len(_TIMELINE_EVENT_ROWS)
    initial_pause_interventions = 0
    if _TIMELINE.is_playing():
        _TIMELINE.pause()
        initial_pause_interventions = 1
    initial_pending = _detailed_snapshot(inner, "after_reset_pause_before_commit")
    matched_d352_initial = _matched_d352_initial_snapshot_contract(initial_pending)
    _marker(
        "worker",
        "matched_d352_initial_pending_snapshot",
        "end",
        {"pass": matched_d352_initial["pass"]},
    )
    if not matched_d352_initial["pass"]:
        raise RuntimeError(
            "D353 timeline commit precondition STOP: D352 matched pending-state baseline drift"
        )
    initial_pause_event = {
        "phase": "after_reset_initial_pause",
        "playing_before": initial_before["timeline_playing"],
        "playing_pre_commit": initial_pending["timeline_playing"],
        "pause_interventions": initial_pause_interventions,
        "inherited_setting_writes": 1,
    }
    initial_event = _commit_pending_pause(
        inner,
        "after_reset_initial_pause",
        initial_before,
        initial_pending,
        initial_pause_interventions,
        callback_start,
        initial_pause_event,
    )
    commit_events.append(initial_event)
    counter_after_reset = int(inner._sim_step_counter)
    _marker(
        "worker",
        "reset",
        "end",
        {
            "counter_after_reset": counter_after_reset,
            "timeline_playing": bool(_TIMELINE.is_playing()),
            "timeline_stopped": bool(_TIMELINE.is_stopped()),
            "initial_pause_interventions": initial_pause_interventions,
            "commit_called": initial_event["commit_called"],
        },
    )

    _marker("worker", "corrected_audit", "start")
    corrected = d352.d351.d349._corrected_live_audit()
    _marker("worker", "corrected_audit", "end", {"pass": corrected.get("pass")})

    snapshots.append(
        _canonical_snapshot(inner, "after_reset_initial_pause_before_live_binding")
    )
    topology_parts, live_binding, trace_contract = d352._build_live_with_markers(inner)

    _marker("worker", "live_payload_hash", "start")
    original_payload_sha256 = d352.d351a2._payload_sha(live_binding)
    d352_live_reference = d352._json(D352_DIR / "d352_live_topology_runtime_binding.json")
    _marker(
        "worker", "live_payload_hash", "end", {"sha256": original_payload_sha256}
    )
    snapshots.append(_canonical_snapshot(inner, "after_live_binding_before_repause"))
    live_before = _detailed_snapshot(inner, "after_live_binding_before_pause_request")
    _assert_boundary_preconditions(live_before, "after_live_topology_binding:before")
    callback_start = len(_TIMELINE_EVENT_ROWS)
    _marker("worker", "inherited_live_repause", "start")
    inherited_live_event = d352.d351a2._pause_without_update(
        inner, "after_live_topology_binding"
    )
    _marker("worker", "inherited_live_repause", "end", inherited_live_event)
    live_pending = _detailed_snapshot(inner, "after_live_repause_before_commit")
    live_event = _commit_pending_pause(
        inner,
        "after_live_topology_binding",
        live_before,
        live_pending,
        int(inherited_live_event["pause_interventions"]),
        callback_start,
        inherited_live_event,
    )
    commit_events.append(live_event)
    snapshots.append(_canonical_snapshot(inner, "after_live_binding_after_repause"))
    live_binding["d353_inherited_precommit_pause_diagnostic"] = inherited_live_event
    live_binding["d353_committed_pause_contract"] = live_event
    live_binding["d353_original_live_binding_payload_sha256"] = original_payload_sha256
    live_binding["checks"]["d353_committed_pause_contract"] = live_event["pass"]
    live_binding["checks"]["d353_original_payload_reproduces_attempt1"] = (
        original_payload_sha256
        == d352.d351a2.ATTEMPT1_ROOT_HASHES[
            "d351_live_topology_runtime_binding.json"
        ]
    )
    live_binding["checks"]["d353_original_payload_matches_d352_matched_control"] = (
        original_payload_sha256
        == d352_live_reference.get("attempt2_original_live_binding_payload_sha256")
    )
    live_binding["pass"] = all(live_binding["checks"].values())

    _marker("worker", "live_payload_deepcopy_serialize_write", "start")
    live_payload = d352._payload_bytes(copy.deepcopy(live_binding))
    d352._write_bytes(LIVE_BINDING_PATH, live_payload)
    _marker(
        "worker",
        "live_payload_deepcopy_serialize_write",
        "end",
        {"bytes": len(live_payload), "sha256": hashlib.sha256(live_payload).hexdigest()},
    )

    _marker("worker", "raw_binding", "start")
    _raw_shapes, raw_contract = d352.d351a2._ORIGINAL_BUILD_RAW(
        inner, d352._json(d352.d351.D334_SUMMARY)
    )
    d352_raw_reference = d352._json(D352_DIR / "d352_raw_source_contract.json")
    raw_reproduces_d352 = raw_contract == d352_raw_reference
    _marker("worker", "raw_binding", "end", {"pass": raw_contract.get("pass")})
    snapshots.append(
        _canonical_snapshot(inner, "after_live_and_raw_binding_before_final_repause")
    )
    raw_before = _detailed_snapshot(inner, "after_raw_binding_before_pause_request")
    _assert_boundary_preconditions(
        raw_before, "after_raw_shape_binding_before_prerequisites:before"
    )
    callback_start = len(_TIMELINE_EVENT_ROWS)
    _marker("worker", "inherited_final_repause", "start")
    inherited_raw_event = d352.d351a2._pause_without_update(
        inner, "after_raw_shape_binding_before_prerequisites"
    )
    _marker("worker", "inherited_final_repause", "end", inherited_raw_event)
    raw_pending = _detailed_snapshot(inner, "after_raw_repause_before_commit")
    raw_event = _commit_pending_pause(
        inner,
        "after_raw_shape_binding_before_prerequisites",
        raw_before,
        raw_pending,
        int(inherited_raw_event["pause_interventions"]),
        callback_start,
        inherited_raw_event,
    )
    commit_events.append(raw_event)
    snapshots.append(
        _canonical_snapshot(inner, "after_live_and_raw_binding_after_final_repause")
    )

    d352.d351a2._BRIDGE_SNAPSHOTS.clear()
    d352.d351a2._BRIDGE_SNAPSHOTS.extend(copy.deepcopy(snapshots))
    bridge = d352.d351a2._bridge_contract()
    bridge["artifact"] = "D353_ZERO_STEP_BINDING_BRIDGE_V1"
    all_detailed_rows = [
        row
        for event in commit_events
        for row in (
            event["before_pause"],
            event["post_pause_pre_commit"],
            event["post_commit"],
        )
    ]
    all_mutation_rows = all_detailed_rows + snapshots
    global_mutation_checks = _mutation_checks(all_mutation_rows)
    bridge["checks"]["pause_not_stop_at_every_primary_snapshot"] = all(
        row["timeline_playing"] is False and row["timeline_stopped"] is False
        for row in snapshots
    )
    bridge["checks"]["primary_snapshot_metadata_exact"] = all(
        row["timeline_metadata"] == snapshots[0]["timeline_metadata"] for row in snapshots
    )
    bridge["checks"]["q5_counts_zero_at_every_primary_snapshot"] = all(
        row["q5_science_evaluation_count"] == 0
        and row["q5_science_state_write_count"] == 0
        and row["q5_trap_invocation_count"] == 0
        for row in snapshots
    )
    bridge["checks"]["all_detailed_and_canonical_snapshot_count_exact_14"] = (
        len(all_mutation_rows) == 14
    )
    bridge["checks"]["all_detailed_and_canonical_mutation_sentinels_exact"] = all(
        global_mutation_checks.values()
    )
    bridge["checks"]["matched_d352_initial_pending_snapshot"] = (
        matched_d352_initial["pass"]
    )
    bridge["pass"] = all(bridge["checks"].values())
    bridge["d353_scope"] = {
        "control_repair_only": True,
        "single_new_variable": NEW_OPERATIONAL_VARIABLES,
        "q5_science_evaluation_count": 0,
        "q5_trap_invocation_count": d352._Q5_TRAP_INVOCATION_COUNT,
        "timeline_commit_call_count": _COMMIT_CALL_COUNT,
        "timeline_commit_attempt_count": _COMMIT_ATTEMPT_COUNT,
        "commit_events": commit_events,
        "matched_d352_initial_pending_snapshot": matched_d352_initial,
        "global_mutation_checks": global_mutation_checks,
        "global_mutation_snapshot_count": len(all_mutation_rows),
    }
    _write_json(RAW_CONTRACT_PATH, raw_contract)
    raw_file_sha_reproduces_d352 = (
        d352._sha(RAW_CONTRACT_PATH)
        == D352_IMMUTABLE_HASHES[
            "claudedocs/runtime_logs/grasp_track/g0a_d352/"
            "d352_raw_source_contract.json"
        ]
    )
    _write_json(BRIDGE_PATH, bridge)

    discriminating = [
        event
        for event in commit_events
        if event["pause_interventions"] > 0
        and event["before_pause"]["timeline_playing"] is True
        and event["post_pause_pre_commit"]["timeline_playing"] is True
        and event["post_commit"]["timeline_playing"] is False
    ]
    boundary_callback_rows = [
        row for event in commit_events for row in event["timeline_callbacks"]
    ]
    commit_checks = {
        "exact_three_registered_boundaries": [event["phase"] for event in commit_events]
        == [
            "after_reset_initial_pause",
            "after_live_topology_binding",
            "after_raw_shape_binding_before_prerequisites",
        ],
        "all_boundary_contracts_pass": all(event["pass"] for event in commit_events),
        "at_least_one_discriminating_pending_transition": len(discriminating) >= 1,
        "commit_count_matches_pause_blocks": _COMMIT_CALL_COUNT
        == sum(event["pause_interventions"] > 0 for event in commit_events),
        "commit_count_at_least_one": _COMMIT_CALL_COUNT >= 1,
        "commit_attempt_count_matches_success_count": _COMMIT_ATTEMPT_COUNT
        == _COMMIT_CALL_COUNT,
        "all_callback_rows_exactly_accounted_by_boundaries": (
            boundary_callback_rows == _TIMELINE_EVENT_ROWS
        ),
        "all_callbacks_are_exact_pause": bool(_TIMELINE_EVENT_ROWS)
        and all(
            row["event_type"] == 1 and row["event_name"] == "PAUSE"
            for row in _TIMELINE_EVENT_ROWS
        ),
        "all_callbacks_main_thread": all(
            row["callback_is_main_thread"] is True for row in _TIMELINE_EVENT_ROWS
        ),
        "canonical_five_snapshot_bridge": bridge["pass"],
        "global_detailed_plus_canonical_mutation_contract": all(
            global_mutation_checks.values()
        ),
        "matched_d352_initial_pending_snapshot": matched_d352_initial["pass"],
        "raw_payload_reproduces_d352": raw_reproduces_d352,
        "raw_serialized_sha_reproduces_d352": raw_file_sha_reproduces_d352,
        "q5_counts_zero": d352._Q5_TRAP_INVOCATION_COUNT == 0,
        "timeline_api_runtime_exact": _timeline_api_contract()["pass"],
        "runtime_event_type_map_complete_0_through_17": set(
            _TIMELINE_EVENT_TYPE_NAMES
        )
        == set(range(18))
        and _TIMELINE_EVENT_TYPE_NAMES.get(1) == "PAUSE",
        "subscription_retained": _TIMELINE_SUBSCRIPTION is not None,
    }
    commit_contract = {
        "artifact": "D353_TIMELINE_COMMIT_EVENT_CONTRACT_V1",
        "case": CASE,
        "question": "Does explicit commit apply pending PAUSE without advancing timeline/world state?",
        "single_new_variable": NEW_OPERATIONAL_VARIABLES,
        "commit_call_count": _COMMIT_CALL_COUNT,
        "commit_attempt_count": _COMMIT_ATTEMPT_COUNT,
        "boundary_events": commit_events,
        "all_timeline_callback_rows": copy.deepcopy(_TIMELINE_EVENT_ROWS),
        "timeline_event_type_map": copy.deepcopy(_TIMELINE_EVENT_TYPE_NAMES),
        "matched_d352_initial_pending_snapshot": matched_d352_initial,
        "global_mutation_checks": global_mutation_checks,
        "raw_reproduces_d352": raw_reproduces_d352,
        "raw_file_sha_reproduces_d352": raw_file_sha_reproduces_d352,
        "discriminating_transition_count": len(discriminating),
        "canonical_bridge_path": d352._rel(BRIDGE_PATH),
        "canonical_bridge_sha256": d352._sha(BRIDGE_PATH),
        "checks": commit_checks,
        "pass": all(commit_checks.values()),
    }
    _write_json(COMMIT_CONTRACT_PATH, commit_contract)

    _marker("worker", "post_bridge_source_inventory", "start")
    source_after = d352.d351.d349._source_inventories()
    input_after = d352.d351._input_hashes()
    immutable_after = _immutable_prior_contract()
    sidecar_after = _user_sidecar_contract()
    frozen_hashes_after = _frozen_run_hash_contract()
    _marker("worker", "post_bridge_source_inventory", "end")

    prerequisites = {
        "launcher": launcher_report["pass"],
        "counter_after_reset_zero": counter_after_reset == 0,
        "corrected_d348_128_of_128": corrected["pass"],
        "live_trace_128_start_end": trace_contract["pass"],
        "live_binding_64_plus_64_and_commit": live_binding["pass"],
        "raw_source_contract": raw_contract["pass"],
        "canonical_five_snapshot_bridge": bridge["pass"],
        "timeline_commit_event_contract": commit_contract["pass"],
        "source_inventory_unchanged": source_after == source_before,
        "input_hashes_unchanged": input_after == input_before,
        "d351_d352_immutable_after": immutable_after["pass"],
        "user_sidecar_read_only_exact_inventory_after": sidecar_after["pass"],
        "preregistered_run_hashes_exact_after": frozen_hashes_after["pass"],
        "matched_d352_initial_pending_snapshot": matched_d352_initial["pass"],
        "raw_payload_reproduces_d352": raw_reproduces_d352,
        "raw_serialized_sha_reproduces_d352": raw_file_sha_reproduces_d352,
        "q5_base_and_saved_alias_trapped": d352.d351._evaluate_q5 is _forbidden_q5
        and d352.d351a2._ORIGINAL_EVALUATE_Q5 is _forbidden_q5,
        "q5_trap_invocation_count_zero": d352._Q5_TRAP_INVOCATION_COUNT == 0,
        "q5_science_evaluation_count_zero": True,
        "q5_science_state_write_count_zero": True,
    }
    controlled_steps = 0 if all(prerequisites.values()) else None
    summary = {
        "artifact": "D353_TIMELINE_COMMIT_BRIDGE_SUMMARY_V1",
        "case": CASE,
        "operational_verdict": (
            "D353_TIMELINE_COMMIT_BRIDGE_CONTRACT_COMPLETE_PENDING_ATTESTATION"
            if all(prerequisites.values())
            else "D353_TIMELINE_COMMIT_BRIDGE_CONTRACT_FAIL_STOP"
        ),
        "question": "commit-only pending-state PAUSE zero-step bridge",
        "prerequisites": prerequisites,
        "trace_contract": trace_contract,
        "commit_contract_path": d352._rel(COMMIT_CONTRACT_PATH),
        "commit_contract_sha256": d352._sha(COMMIT_CONTRACT_PATH),
        "bridge_path": d352._rel(BRIDGE_PATH),
        "bridge_sha256": d352._sha(BRIDGE_PATH),
        "live_binding_path": d352._rel(LIVE_BINDING_PATH),
        "live_binding_sha256": d352._sha(LIVE_BINDING_PATH),
        "raw_contract_path": d352._rel(RAW_CONTRACT_PATH),
        "raw_contract_sha256": d352._sha(RAW_CONTRACT_PATH),
        "d353_q5_science_evaluation_count": 0,
        "d353_q5_science_state_write_count": 0,
        "d353_q5_trap_invocation_count": d352._Q5_TRAP_INVOCATION_COUNT,
        "d353_controlled_physics_steps": None,
        "d353_controlled_physics_steps_candidate": controlled_steps,
        "controlled_steps_authority": (
            "pending separate attestation after this summary is fsynced and reread exact"
        ),
        "d353_timeline_commit_call_count": _COMMIT_CALL_COUNT,
        "d353_timeline_commit_attempt_count": _COMMIT_ATTEMPT_COUNT,
        "inherited_d351_attempt2_q5_evaluation_count": 0,
        "inherited_d351_attempt2_controlled_physics_steps": None,
        "inherited_d352_q5_evaluation_count": 0,
        "inherited_d352_controlled_physics_steps": None,
        "historical_d352_current_value_aliases_absent": True,
        "matched_d352_initial_pending_snapshot": matched_d352_initial,
        "frozen_run_hashes_after": frozen_hashes_after,
        "user_sidecar_read_only_after": sidecar_after,
        "scientific_verdict": None,
        "geometry_pass_or_fail": None,
        "current_pose_support_or_rejection": None,
        "grasp_feasibility": None,
        "target_ik_repair_justification": None,
        "moving_surface_measurement": None,
        "q5_sweep": None,
        "viewer_rerun_rrd_rbl": None,
        "g0a_pass": False,
        "automatic_retry": False,
        "commit_or_push_performed": False,
        "pass": all(prerequisites.values()),
    }
    _write_json(LOCALIZATION_PATH, summary)
    persisted_summary = d352._json(LOCALIZATION_PATH)
    if persisted_summary != summary:
        raise RuntimeError("D353 authoritative summary durable re-read mismatch")
    attestation = {
        "artifact": "D353_TIMELINE_COMMIT_BRIDGE_ATTESTATION_V1",
        "case": CASE,
        "summary_path": d352._rel(LOCALIZATION_PATH),
        "summary_sha256": d352._sha(LOCALIZATION_PATH),
        "summary_fsync_and_reread_exact": True,
        "operational_verdict": (
            "D353_TIMELINE_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE"
            if summary["pass"] and controlled_steps == 0
            else "D353_TIMELINE_COMMIT_BRIDGE_CONTRACT_FAIL_STOP"
        ),
        "d353_q5_science_evaluation_count": 0,
        "d353_q5_science_state_write_count": 0,
        "d353_q5_trap_invocation_count": d352._Q5_TRAP_INVOCATION_COUNT,
        "d353_controlled_physics_steps": controlled_steps,
        "inherited_d351_attempt2_controlled_physics_steps": None,
        "inherited_d352_controlled_physics_steps": None,
        "scientific_verdict": None,
        "geometry_pass_or_fail": None,
        "current_pose_support_or_rejection": None,
        "grasp_feasibility": None,
        "target_ik_repair_justification": None,
        "g0a_pass": False,
        "pass": bool(summary["pass"] and controlled_steps == 0),
    }
    _write_json(BRIDGE_ATTESTATION_PATH, attestation)
    persisted_attestation = d352._json(BRIDGE_ATTESTATION_PATH)
    if persisted_attestation != attestation:
        raise RuntimeError("D353 bridge attestation durable re-read mismatch")
    _D353_BRIDGE_COMPLETE = bool(
        attestation["pass"] and attestation["d353_controlled_physics_steps"] == 0
    )
    _marker(
        "worker",
        "timeline_commit_bridge_boundary",
        "complete",
        {
            "pass": attestation["pass"],
            "q5_count": 0,
            "commit_count": _COMMIT_CALL_COUNT,
            "controlled_physics_steps": attestation["d353_controlled_physics_steps"],
            "attestation_sha256": d352._sha(BRIDGE_ATTESTATION_PATH),
        },
    )
    _marker(
        "worker",
        "timeline_event_observer",
        "end",
        {"recorded_rows": len(_TIMELINE_EVENT_ROWS), "release_before_cleanup": True},
    )
    _TIMELINE_SUBSCRIPTION = None
    _TIMELINE = None
    return 0 if attestation["pass"] else 2


def _last_pre_cleanup_marker(markers: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [
        row
        for row in markers
        if row.get("actor") == "worker"
        and row.get("phase") not in {"inner.close", "SimulationApp.close"}
    ]
    return rows[-1] if rows else None


def _output_inventory_contract() -> dict[str, Any]:
    allowed = {
        path.name
        for path in (
            PARAMETER_PATH,
            GPU_HARDWARE_PATH,
            PREREG_PATH,
            SUPERVISOR_PREFLIGHT_PATH,
            WORKER_PREFLIGHT_PATH,
            MARKER_PATH,
            WORKER_LOG_PATH,
            GPU_TELEMETRY_PATH,
            FAULT_PATH,
            LIVE_BINDING_PATH,
            RAW_CONTRACT_PATH,
            BRIDGE_PATH,
            COMMIT_CONTRACT_PATH,
            LOCALIZATION_PATH,
            BRIDGE_ATTESTATION_PATH,
            RUNTIME_EXCEPTION_PATH,
            WATCHDOG_PROC_PATH,
            RAW_SUPERVISOR_PATH,
            SUPERVISOR_AUDIT_PATH,
        )
    }
    normal_success_before_final = {
        path.name
        for path in (
            PARAMETER_PATH,
            GPU_HARDWARE_PATH,
            PREREG_PATH,
            SUPERVISOR_PREFLIGHT_PATH,
            WORKER_PREFLIGHT_PATH,
            MARKER_PATH,
            WORKER_LOG_PATH,
            GPU_TELEMETRY_PATH,
            FAULT_PATH,
            LIVE_BINDING_PATH,
            RAW_CONTRACT_PATH,
            BRIDGE_PATH,
            COMMIT_CONTRACT_PATH,
            LOCALIZATION_PATH,
            BRIDGE_ATTESTATION_PATH,
            RAW_SUPERVISOR_PATH,
        )
    }
    actual = {
        path.name for path in OUT_DIR.iterdir() if path.is_file()
    } if OUT_DIR.is_dir() else set()
    non_files = sorted(
        path.name for path in OUT_DIR.iterdir() if not path.is_file()
    ) if OUT_DIR.is_dir() else []
    return {
        "actual_before_final_audit": sorted(actual),
        "allowed": sorted(allowed),
        "unexpected": sorted(actual - allowed),
        "missing_normal_success_before_final": sorted(normal_success_before_final - actual),
        "extra_vs_normal_success_before_final": sorted(actual - normal_success_before_final),
        "expected_normal_success_after_final_audit": sorted(
            normal_success_before_final | {SUPERVISOR_AUDIT_PATH.name}
        ),
        "non_file_inventory": non_files,
        "checks": {
            "only_registered_files": not (actual - allowed),
            "non_file_inventory_empty": not non_files,
            "normal_success_before_final_exact": actual == normal_success_before_final,
        },
    }


def _write_supervisor_exception_audit(error: Exception) -> int:
    preflight = (
        d352._json(SUPERVISOR_PREFLIGHT_PATH)
        if SUPERVISOR_PREFLIGHT_PATH.is_file()
        else None
    )
    localization = (
        d352._json(LOCALIZATION_PATH) if LOCALIZATION_PATH.is_file() else None
    )
    attestation = (
        d352._json(BRIDGE_ATTESTATION_PATH)
        if BRIDGE_ATTESTATION_PATH.is_file()
        else None
    )
    worker_preflight = (
        d352._json(WORKER_PREFLIGHT_PATH) if WORKER_PREFLIGHT_PATH.is_file() else None
    )
    runtime_exception = (
        d352._json(RUNTIME_EXCEPTION_PATH) if RUNTIME_EXCEPTION_PATH.is_file() else None
    )
    raw = d352._json(RAW_SUPERVISOR_PATH) if RAW_SUPERVISOR_PATH.is_file() else None
    markers, invalid_markers = (
        d352._read_markers(d352._RUN_NONCE) if MARKER_PATH.is_file() else ([], [])
    )
    marker_q5 = sum(
        row.get("phase") == "q5_fail_closed_boundary" for row in markers
    )
    q5_sources = {
        "localization": (
            localization.get("d353_q5_trap_invocation_count")
            if localization is not None
            else None
        ),
        "runtime_exception": (
            runtime_exception.get("d353_q5_trap_invocation_count")
            if runtime_exception is not None
            else None
        ),
        "bridge_attestation": (
            attestation.get("d353_q5_trap_invocation_count")
            if attestation is not None
            else None
        ),
        "durable_marker_count": marker_q5,
    }
    q5_trap_count = max(
        int(value) for value in q5_sources.values() if value is not None
    )
    controlled_steps = (
        attestation.get("d353_controlled_physics_steps")
        if attestation is not None
        else runtime_exception.get("d353_controlled_physics_steps")
        if runtime_exception is not None
        else None
    )
    bridge_authority_persisted = bool(
        localization is not None
        and localization.get("pass") is True
        and attestation is not None
        and attestation.get("pass") is True
        and attestation.get("summary_fsync_and_reread_exact") is True
        and attestation.get("summary_sha256") == d352._sha(LOCALIZATION_PATH)
        and controlled_steps == 0
    )
    runtime_error_text = str((runtime_exception or {}).get("error", ""))
    watchdog_marker_seen = any(
        row.get("actor") == "supervisor"
        and row.get("phase") == "watchdog"
        and row.get("event") == "deadline"
        for row in markers
    )
    if q5_trap_count:
        verdict = "D353_Q5_SCOPE_BREACH_STOP"
    elif watchdog_marker_seen:
        verdict = "D353_PHASE_WATCHDOG_STOP"
    elif (preflight is not None and preflight.get("pass") is False) or (
        worker_preflight is not None and worker_preflight.get("pass") is False
    ):
        verdict = "D353_VALIDATE_PREFLIGHT_STOP"
    elif bridge_authority_persisted:
        verdict = "D353_POST_BRIDGE_OBSERVABILITY_STOP"
    elif "precondition STOP" in runtime_error_text:
        verdict = "D353_TIMELINE_COMMIT_PRECONDITION_STOP"
    else:
        verdict = "D353_RUNTIME_EXCEPTION_STOP"
    inventory = _output_inventory_contract()
    audit = {
        "artifact": "D353_SUPERVISOR_AUDIT_V1",
        "case": CASE,
        "operational_verdict": verdict,
        "supervisor_wrapper_exception": f"{type(error).__name__}: {error}",
        "supervisor_preflight": preflight,
        "worker_preflight": worker_preflight,
        "inherited_supervisor_raw": (
            {
                "path": d352._rel(RAW_SUPERVISOR_PATH),
                "sha256": d352._sha(RAW_SUPERVISOR_PATH),
                "raw_pass": raw.get("pass"),
                "classifier_is_non_authoritative_for_d353_taxonomy": True,
            }
            if raw is not None
            else None
        ),
        "worker": raw.get("worker") if raw is not None else None,
        "watchdog": raw.get("watchdog") if raw is not None else None,
        "markers": {
            "path": d352._rel(MARKER_PATH) if MARKER_PATH.is_file() else None,
            "sha256": d352._sha(MARKER_PATH) if MARKER_PATH.is_file() else None,
            "valid_count": len(markers),
            "invalid_rows": invalid_markers,
            "last_valid": markers[-1] if markers else None,
            "last_pre_cleanup": _last_pre_cleanup_marker(markers),
        },
        "telemetry": raw.get("telemetry") if raw is not None else None,
        "localization_summary": {
            "path": d352._rel(LOCALIZATION_PATH) if localization is not None else None,
            "sha256": d352._sha(LOCALIZATION_PATH) if localization is not None else None,
            "pass": localization.get("pass") if localization is not None else None,
        },
        "bridge_attestation": {
            "path": (
                d352._rel(BRIDGE_ATTESTATION_PATH) if attestation is not None else None
            ),
            "sha256": (
                d352._sha(BRIDGE_ATTESTATION_PATH) if attestation is not None else None
            ),
            "pass": attestation.get("pass") if attestation is not None else None,
        },
        "runtime_exception": runtime_exception,
        "q5_count_sources": q5_sources,
        "bridge_authority_persisted": bridge_authority_persisted,
        "watchdog_deadline_marker_seen": watchdog_marker_seen,
        "output_inventory_before_final_audit": inventory,
        "frozen_run_hashes_after": _frozen_run_hash_contract(),
        "prior_d351_d352_immutable_after": _immutable_prior_contract(),
        "user_sidecar_read_only_after": _user_sidecar_contract(),
        "d353_q5_science_evaluation_count": 0,
        "d353_q5_science_state_write_count": 0,
        "d353_q5_trap_invocation_count": q5_trap_count,
        "d353_controlled_physics_steps": controlled_steps,
        "inherited_d351_attempt2_controlled_physics_steps": None,
        "inherited_d352_controlled_physics_steps": None,
        "scientific_verdict": None,
        "geometry_pass_or_fail": None,
        "current_pose_support_or_rejection": None,
        "grasp_feasibility": None,
        "target_ik_repair_justification": None,
        "moving_surface_measurement": None,
        "q5_sweep": None,
        "viewer_rerun_rrd_rbl": None,
        "g0a_pass": False,
        "automatic_retry": False,
        "commit_or_push_performed": False,
        "pass": False,
    }
    _write_json(SUPERVISOR_AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "stage": "validate",
                "operational_verdict": verdict,
                "pass": False,
                "controlled_physics_steps": controlled_steps,
                "q5_science_evaluation_count": 0,
            },
            sort_keys=True,
        )
    )
    return 2


def _run_supervisor(args: argparse.Namespace) -> int:
    try:
        _raw_return_code = _ORIGINAL_RUN_SUPERVISOR(args)
    except Exception as error:
        return _write_supervisor_exception_audit(error)

    raw = d352._json(RAW_SUPERVISOR_PATH)
    localization = d352._json(LOCALIZATION_PATH) if LOCALIZATION_PATH.is_file() else None
    attestation = (
        d352._json(BRIDGE_ATTESTATION_PATH)
        if BRIDGE_ATTESTATION_PATH.is_file()
        else None
    )
    runtime_exception = (
        d352._json(RUNTIME_EXCEPTION_PATH) if RUNTIME_EXCEPTION_PATH.is_file() else None
    )
    markers, invalid_markers = d352._read_markers(d352._RUN_NONCE)
    q5_breaches = sum(
        row.get("phase") == "q5_fail_closed_boundary" for row in markers
    )
    q5_count_sources = {
        "localization": (
            localization.get("d353_q5_trap_invocation_count")
            if localization is not None
            else None
        ),
        "runtime_exception": (
            runtime_exception.get("d353_q5_trap_invocation_count")
            if runtime_exception is not None
            else None
        ),
        "bridge_attestation": (
            attestation.get("d353_q5_trap_invocation_count")
            if attestation is not None
            else None
        ),
        "durable_marker_count": q5_breaches,
    }
    q5_trap_count = max(
        int(value)
        for value in q5_count_sources.values()
        if value is not None
    )
    controlled_steps = (
        attestation.get("d353_controlled_physics_steps")
        if attestation is not None
        else runtime_exception.get("d353_controlled_physics_steps")
        if runtime_exception is not None
        else None
    )
    error_text = str((runtime_exception or {}).get("error", ""))
    supervisor_preflight = d352._json(SUPERVISOR_PREFLIGHT_PATH)
    worker_preflight = (
        d352._json(WORKER_PREFLIGHT_PATH) if WORKER_PREFLIGHT_PATH.is_file() else None
    )
    bridge_authority_persisted = bool(
        localization is not None
        and localization.get("pass") is True
        and attestation is not None
        and attestation.get("pass") is True
        and attestation.get("summary_fsync_and_reread_exact") is True
        and attestation.get("summary_sha256") == d352._sha(LOCALIZATION_PATH)
        and controlled_steps == 0
    )
    verdict = "D353_ABNORMAL_EXIT_STOP"
    if q5_trap_count:
        verdict = "D353_Q5_SCOPE_BREACH_STOP"
    elif raw.get("watchdog", {}).get("triggered"):
        verdict = "D353_PHASE_WATCHDOG_STOP"
    elif supervisor_preflight.get("pass") is False or (
        worker_preflight is not None and worker_preflight.get("pass") is False
    ):
        verdict = "D353_VALIDATE_PREFLIGHT_STOP"
    elif runtime_exception is not None and bridge_authority_persisted:
        verdict = "D353_POST_BRIDGE_OBSERVABILITY_STOP"
    elif runtime_exception is not None and "precondition STOP" in error_text:
        verdict = "D353_TIMELINE_COMMIT_PRECONDITION_STOP"
    elif runtime_exception is not None:
        verdict = "D353_RUNTIME_EXCEPTION_STOP"
    elif localization is not None and localization.get("pass") is False:
        verdict = "D353_TIMELINE_COMMIT_BRIDGE_CONTRACT_FAIL_STOP"
    elif localization is not None and localization.get("pass") is True:
        verdict = "D353_BRIDGE_COMPLETE_PENDING_FINAL_GATES"
    worker = raw.get("worker", {})
    process_terminal = bool(
        worker.get("process_absent_after_reap")
        and worker.get("process_group_absent_after_cleanup")
    )
    cleanup_rows = [
        row
        for row in markers
        if row.get("phase") in {"inner.close", "SimulationApp.close"}
    ]
    prior_after = _immutable_prior_contract()
    sidecar_after = d352._user_sidecar_contract()
    frozen_hashes_after = _frozen_run_hash_contract()
    inventory = _output_inventory_contract()
    telemetry = raw.get("telemetry") or {}
    raw_completion_checks = {
        "supervisor_preflight_pass": supervisor_preflight.get("pass") is True,
        "worker_preflight_pass": worker_preflight is not None
        and worker_preflight.get("pass") is True,
        "worker_exit_zero": worker.get("exit_code") == 0,
        "worker_and_process_group_terminal": process_terminal,
        "watchdog_not_triggered": raw.get("watchdog", {}).get("triggered") is False,
        "marker_rows_valid": not invalid_markers,
        "runtime_exception_absent": runtime_exception is None,
        "telemetry_samples_valid": telemetry.get("sample_count", 0) > 0
        and telemetry.get("valid_gpu_sample_count", 0) > 0
        and telemetry.get("invalid_gpu_sample_count") == 0
        and telemetry.get("uuid_mismatch_sample_count") == 0,
        "telemetry_thread_joined": raw.get("telemetry_thread_alive_after_join") is False,
        "forbidden_science_outputs_absent": not raw.get(
            "forbidden_science_or_viewer_outputs", []
        ),
        "prior_d351_d352_immutable": prior_after["pass"],
        "user_sidecar_read_only_exact_inventory": sidecar_after["pass"],
        "preregistered_run_hashes_exact": frozen_hashes_after["pass"],
        "normal_success_output_inventory_exact": inventory["checks"][
            "normal_success_before_final_exact"
        ],
        "authoritative_attestation_pass": bridge_authority_persisted,
        "q5_authorities_all_zero": q5_trap_count == 0
        and all(value in (None, 0) for value in q5_count_sources.values()),
    }
    raw_completion_pass = all(raw_completion_checks.values())
    if verdict == "D353_BRIDGE_COMPLETE_PENDING_FINAL_GATES":
        verdict = (
            "D353_TIMELINE_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE"
            if raw_completion_pass
            else "D353_POST_BRIDGE_OBSERVABILITY_STOP"
        )

    final_pass = bool(
        verdict == "D353_TIMELINE_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE"
        and localization
        and localization.get("pass")
        and raw_completion_pass
        and not invalid_markers
        and q5_trap_count == 0
        and controlled_steps == 0
        and process_terminal
        and prior_after["pass"]
        and sidecar_after["pass"]
    )
    audit = {
        "artifact": "D353_SUPERVISOR_AUDIT_V1",
        "case": CASE,
        "operational_verdict": verdict,
        "inherited_supervisor_raw": {
            "path": d352._rel(RAW_SUPERVISOR_PATH),
            "sha256": d352._sha(RAW_SUPERVISOR_PATH),
            "return_code": _raw_return_code,
            "classifier_is_non_authoritative_for_d353_taxonomy": True,
            "raw_operational_verdict": raw.get("operational_verdict"),
            "raw_pass": raw.get("pass"),
            "raw_pass_not_used_for_d353": True,
            "reason": (
                "Inherited D352 classifier expects d352_* current-case aliases; "
                "D353 deliberately omits them to preserve historical D352 null."
            ),
        },
        "worker": worker,
        "watchdog": raw.get("watchdog"),
        "markers": {
            "path": d352._rel(MARKER_PATH),
            "sha256": d352._sha(MARKER_PATH),
            "valid_count": len(markers),
            "invalid_rows": invalid_markers,
            "last_valid": markers[-1] if markers else None,
            "last_pre_cleanup": _last_pre_cleanup_marker(markers),
        },
        "cleanup_interpretation": {
            "process_terminal": process_terminal,
            "simulation_app_close_start_seen": any(
                row.get("phase") == "SimulationApp.close" and row.get("event") == "start"
                for row in cleanup_rows
            ),
            "simulation_app_close_return_marker_seen": any(
                row.get("phase") == "SimulationApp.close" and row.get("event") == "end"
                for row in cleanup_rows
            ),
            "cleanup_error_marker_count": sum(
                row.get("event") == "error" for row in cleanup_rows
            ),
            "active_stall_location": (
                raw.get("markers", {}).get("localized_boundary")
                if raw.get("watchdog", {}).get("triggered")
                else None
            ),
            "missing_close_return_marker_is_not_alone_failure_authority": True,
        },
        "telemetry": telemetry,
        "supervisor_preflight": supervisor_preflight,
        "worker_preflight": worker_preflight,
        "localization_summary": {
            "path": d352._rel(LOCALIZATION_PATH) if LOCALIZATION_PATH.is_file() else None,
            "sha256": d352._sha(LOCALIZATION_PATH) if LOCALIZATION_PATH.is_file() else None,
            "pass": localization.get("pass") if localization else None,
        },
        "bridge_attestation": {
            "path": (
                d352._rel(BRIDGE_ATTESTATION_PATH) if attestation is not None else None
            ),
            "sha256": (
                d352._sha(BRIDGE_ATTESTATION_PATH) if attestation is not None else None
            ),
            "pass": attestation.get("pass") if attestation is not None else None,
        },
        "runtime_exception": runtime_exception,
        "q5_count_sources": q5_count_sources,
        "bridge_authority_persisted": bridge_authority_persisted,
        "raw_completion_checks": raw_completion_checks,
        "output_inventory_before_final_audit": inventory,
        "d353_q5_science_evaluation_count": 0,
        "d353_q5_science_state_write_count": 0,
        "d353_q5_trap_invocation_count": q5_trap_count,
        "d353_controlled_physics_steps": controlled_steps,
        "inherited_d351_attempt2_controlled_physics_steps": None,
        "inherited_d352_controlled_physics_steps": None,
        "scientific_verdict": None,
        "geometry_pass_or_fail": None,
        "current_pose_support_or_rejection": None,
        "grasp_feasibility": None,
        "target_ik_repair_justification": None,
        "moving_surface_measurement": None,
        "q5_sweep": None,
        "viewer_rerun_rrd_rbl": None,
        "g0a_pass": False,
        "prior_d351_d352_immutable_after": prior_after,
        "user_sidecar_read_only_after": sidecar_after,
        "frozen_run_hashes_after": frozen_hashes_after,
        "automatic_retry": False,
        "commit_or_push_performed": False,
        "pass": final_pass,
    }
    _write_json(SUPERVISOR_AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "stage": "validate",
                "operational_verdict": verdict,
                "pass": final_pass,
                "controlled_physics_steps": controlled_steps,
                "q5_science_evaluation_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if final_pass else 2


def _run_worker(args: argparse.Namespace) -> int:
    global _ACTIVE_COMMIT_PHASE, _TIMELINE, _TIMELINE_SUBSCRIPTION

    d352.d351a2._ORIGINAL_EVALUATE_Q5 = _forbidden_q5
    d352._BRIDGE_COMPLETE = False
    try:
        return _ORIGINAL_RUN_WORKER(args)
    finally:
        _ACTIVE_COMMIT_PHASE = None
        _TIMELINE_SUBSCRIPTION = None
        _TIMELINE = None


def _configure_base() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    replacements = {
        "CASE": CASE,
        "CASE_NAME": CASE_NAME,
        "OUT_DIR": OUT_DIR,
        "HARNESS": HARNESS,
        "SESSION_DOC": SESSION_DOC,
        "START_HERE": START_HERE,
        "EXPECTED_HEAD": EXPECTED_HEAD,
        "SEED": SEED,
        "INACTIVITY_WATCHDOG_S": INACTIVITY_WATCHDOG_S,
        "TOTAL_WATCHDOG_S": TOTAL_WATCHDOG_S,
        "FAULT_DUMP_GRACE_S": FAULT_DUMP_GRACE_S,
        "TERM_GRACE_S": TERM_GRACE_S,
        "GPU_SAMPLE_PERIOD_S": GPU_SAMPLE_PERIOD_S,
        "SUPERVISOR_PID_ENV": SUPERVISOR_PID_ENV,
        "WORKER_LAUNCH_TOKEN_ENV": WORKER_LAUNCH_TOKEN_ENV,
        "NEW_OPERATIONAL_VARIABLES": NEW_OPERATIONAL_VARIABLES,
        "NEW_SCIENTIFIC_VARIABLES": NEW_SCIENTIFIC_VARIABLES,
        "NEW_PHYSICAL_VARIABLES": NEW_PHYSICAL_VARIABLES,
        "PARAMETER_PATH": PARAMETER_PATH,
        "GPU_HARDWARE_PATH": GPU_HARDWARE_PATH,
        "PREREG_PATH": PREREG_PATH,
        "SUPERVISOR_PREFLIGHT_PATH": SUPERVISOR_PREFLIGHT_PATH,
        "WORKER_PREFLIGHT_PATH": WORKER_PREFLIGHT_PATH,
        "MARKER_PATH": MARKER_PATH,
        "WORKER_LOG_PATH": WORKER_LOG_PATH,
        "GPU_TELEMETRY_PATH": GPU_TELEMETRY_PATH,
        "FAULT_PATH": FAULT_PATH,
        "LIVE_BINDING_PATH": LIVE_BINDING_PATH,
        "RAW_CONTRACT_PATH": RAW_CONTRACT_PATH,
        "BRIDGE_PATH": BRIDGE_PATH,
        "LOCALIZATION_PATH": LOCALIZATION_PATH,
        "RUNTIME_EXCEPTION_PATH": RUNTIME_EXCEPTION_PATH,
        "WATCHDOG_PROC_PATH": WATCHDOG_PROC_PATH,
        "SUPERVISOR_AUDIT_PATH": RAW_SUPERVISOR_PATH,
    }
    for name, value in replacements.items():
        setattr(d352, name, value)
    d352._write_json = _write_json
    d352._gpu_hardware_contract = _gpu_hardware_contract
    d352._sample_gpu_cpu = _sample_gpu_cpu
    d352._marker = _marker
    d352._user_sidecar_contract = _user_sidecar_contract
    d352._immutable_d351_contract = _immutable_prior_contract
    d352._parameter_freeze_audit = _parameter_freeze_audit
    d352._ast_scope_contract = _ast_scope_contract
    d352._supervisor_preflight = _supervisor_preflight
    d352._worker_preflight = _worker_preflight
    d352._forbidden_q5 = _forbidden_q5
    d352._run_localization = _run_commit_bridge
    _CONFIGURED = True


def main() -> int:
    _configure_base()
    stage_probe = argparse.ArgumentParser(add_help=False)
    stage_probe.add_argument(
        "--stage", choices=("prepare", "validate", "_worker"), required=True
    )
    stage_args, _ = stage_probe.parse_known_args()
    args = d352._parser(stage_args.stage).parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D353 output path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D353 seed drift")
    if float(args.inactivity_watchdog_s) != INACTIVITY_WATCHDOG_S:
        raise RuntimeError("D353 inactivity watchdog drift")
    if float(args.total_watchdog_s) != TOTAL_WATCHDOG_S:
        raise RuntimeError("D353 total watchdog drift")
    args.process_nonce = secrets.token_hex(16)
    if args.stage == "prepare":
        return _run_prepare(args)
    if args.stage == "validate":
        return _run_supervisor(args)

    args.headless = False
    args.livestream = 0
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    if hasattr(args, "xr"):
        args.xr = False
    args.device = "cuda:0"
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
