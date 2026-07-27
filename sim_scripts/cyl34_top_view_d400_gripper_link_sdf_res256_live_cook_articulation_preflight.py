#!/usr/bin/env python3
"""Hash-gated D400 controller for a future, separately approved runtime.

Merely parsing or importing this module does not start Isaac, Kit, PhysX, a
GPU job, a subprocess, or a USD write.  The ``run`` stage first verifies the
user-cited four-hash tuple, freezes the exact dirty worktree and installed
stack, runs failure-capable offline controls, and only then spawns the
registered worker once in its own process group.  It supervises total and
inactivity watchdogs, audits the authoritative Kit-log window, and generates
Rerun observability only after the worker technical gate passes.

The D400 runtime is not authorized by the existence of this file.  A future
command must provide ``--approved-tuple-sha256`` equal to the tuple-file SHA
explicitly cited in a later user approval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import tomllib
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d400/"
    "attempt1_gripper_link_sdf_res256_live_cook_articulation_preflight"
)
PREREG_PATH = OUT_DIR / "d400_preregistration.json"
ATTESTATION_PATH = OUT_DIR / "d400_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d400_proposed_runtime_hash_tuple.json"
RUNTIME_MANIFEST_PATH = OUT_DIR / "d400_runtime_freeze_manifest.json"
PHASE_PATH = OUT_DIR / "d400_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d400_worker_invocation.json"
CLAIM_PATH = OUT_DIR / "d400_worker_claim.json"
KIT_LOG_PATH = OUT_DIR / "d400_kit_log.txt"
RAW_PATH = OUT_DIR / "d400_worker_raw_summary.json"
PRECLOSE_PATH = OUT_DIR / "d400_worker_preclose_sentinel.json"
OWNER_EVIDENCE_PATH = OUT_DIR / "d400_live_configuration_owner_evidence.json"
SUPERVISOR_PATH = OUT_DIR / "d400_worker_supervisor.json"
COMPLETION_PATH = OUT_DIR / "d400_completion_summary.json"

RRD_PATH = OUT_DIR / "d400_sdf_preflight.rrd"
RBL_PATH = OUT_DIR / "d400_sdf_preflight.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d400_rerun_validation.json"
BOARD_PATH = OUT_DIR / "d400_decision_board_1920x1080.png"
RERUN_SCREENSHOT_PATH = OUT_DIR / "d400_rerun_viewer_1920x1080.png"
RERUN_RECEIPT_PATH = OUT_DIR / "d400_rerun_render_receipt.json"
MANUAL_INSPECTION_PATH = OUT_DIR / "d400_manual_visual_inspection.json"
COLLISION_ASSET_ROOT = OUT_DIR / "collision_asset"

CONTROLLER = Path(__file__).resolve()
WORKER = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_worker.py"
)
ISAAC_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
PHYSX_NATIVE_PLUGIN = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/"
    "omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "bin/libomni.physx.plugin.so"
)
EXPECTED_PHYSX_NATIVE_PLUGIN_SHA256 = (
    "03fbf17e6f0dc3f9006c8c00aa0ca572a72fd69498874df6dd900dac726c9909"
)

EXPECTED_PREREG_SHA256 = (
    "fc689cb1afd6108a326a73f22b8117dfdefc0bb4d8caee5bcb7470c362e96c93"
)
TUPLE_FIELDS = (
    "preregistration_sha256",
    "reviewed_script_attestation_sha256",
    "controller_script_sha256",
    "worker_script_sha256",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

TOTAL_WATCHDOG_S = 300.0
INACTIVITY_WATCHDOG_S = 60.0
SIGTERM_GRACE_S = 20.0
SIGKILL_WAIT_S = 20.0
MANUAL_INSPECTION_WAIT_S = 300.0
MINIMUM_FREE_VRAM_MIB = 8192
MAX_APP_UPDATE_PUMPS = 300000
RERUN_VERSION = "0.34.1"
REGISTERED_STATIC_NEGATIVE_IDS = (
    "source_mesh_hash_perturbation_rejected",
    "source_stream_bit_or_order_perturbation_rejected",
    "approximation_convex_hull_rejected",
    "required_api_missing_rejected",
    "resolution_255_257_512_rejected",
    "remeshing_true_rejected",
    "triangle_reduction_not_one_rejected",
    "sdf_api_on_xform_rejected",
    "instance_proxy_owner_rejected",
    "one_gripper_a64_still_active_rejected",
    "sdf_mesh_collision_disabled_rejected",
    "link5_active_count_not_64_rejected",
    "property_query_contract_perturbation_rejected",
    "cook_zero_to_zero_false_pass_rejected",
    "mass_com_or_inertia_perturbation_rejected",
    "semantic_runtime_ids_excluded_but_semantic_change_rejected",
    "worker_internal_fail_return_zero_rejected",
    "truncated_rrd_footer_rejected",
)

RESULT_PASS = (
    "D400_GRIPPER_LINK_SDF_RES256_CONFIGURATION_LOAD_ADMISSION_"
    "OWNER_ENUMERATION_PREFLIGHT_PASS_NO_PHYSICS"
)
RESULT_TECHNICAL_FAIL = "D400_GRIPPER_LINK_SDF_RES256_PREFLIGHT_FAIL_STOP"
RESULT_OBSERVABILITY_FAIL = "D400_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"

MANDATORY_PHASE_ORDER = (
    "supervisor_preflight_start",
    "runtime_freeze_manifest_gate_end",
    "package_gpu_and_existing_process_gate_end",
    "offline_negative_controls_end",
    "worker_spawn_once",
    "simulation_app_launch_start",
    "simulation_app_launch_end",
    "derivative_copy_start",
    "derivative_sdf_opinion_write_end",
    "typed_authored_readback_gate_end",
    "live_stage_create_end",
    "live_owner_inventory_gate_end",
    "cooking_baseline_capture_start",
    "cooking_baseline_capture_end",
    "physx_stage_attach_start",
    "physx_stage_attach_end",
    "sdf_cooking_wait_start",
    "sdf_cooking_wait_end",
    "property_query_link5_end",
    "property_query_gripper_link_end",
    "mass_invariance_gate_end",
    "cleanup_start",
    "physx_stage_detach_start",
    "physx_stage_detach_end",
    "stagecache_erase_end",
    "safe_to_close_app",
    "worker_raw_summary_written",
    "worker_preclose_sentinel_written",
    "simulation_app_close_start",
    "external_kit_log_audit_end",
    "external_supervisor_end",
    "technical_pass_branch_start",
    "rerun_save_only_finalize_end_if_technical_pass",
    "rerun_verify_and_visual_inspection_end_if_technical_pass",
    "completion_summary_end",
)
OPTIONAL_PHASES = ("simulation_app_close_returned",)
CONTROLLER_PRE_PHASES = (
    "supervisor_preflight_start",
    "runtime_freeze_manifest_gate_end",
    "package_gpu_and_existing_process_gate_end",
    "offline_negative_controls_end",
    "worker_spawn_once",
)
WORKER_PHASES = (
    "simulation_app_launch_start",
    "simulation_app_launch_end",
    "derivative_copy_start",
    "derivative_sdf_opinion_write_end",
    "typed_authored_readback_gate_end",
    "live_stage_create_end",
    "live_owner_inventory_gate_end",
    "cooking_baseline_capture_start",
    "cooking_baseline_capture_end",
    "physx_stage_attach_start",
    "physx_stage_attach_end",
    "sdf_cooking_wait_start",
    "sdf_cooking_wait_end",
    "property_query_link5_end",
    "property_query_gripper_link_end",
    "mass_invariance_gate_end",
    "cleanup_start",
    "physx_stage_detach_start",
    "physx_stage_detach_end",
    "stagecache_erase_end",
    "safe_to_close_app",
    "worker_raw_summary_written",
    "worker_preclose_sentinel_written",
    "simulation_app_close_start",
)
CONTROLLER_POST_PHASES = (
    "external_kit_log_audit_end",
    "external_supervisor_end",
    "technical_pass_branch_start",
    "rerun_save_only_finalize_end_if_technical_pass",
    "rerun_verify_and_visual_inspection_end_if_technical_pass",
    "completion_summary_end",
)
PHASE_OWNER = {
    **{name: "controller" for name in CONTROLLER_PRE_PHASES},
    **{name: "worker" for name in WORKER_PHASES},
    **{name: "controller" for name in CONTROLLER_POST_PHASES},
    "simulation_app_close_returned": "worker",
}

COUNTER_KEYS = (
    "actual_worker_invocations",
    "automatic_retries",
    "simulation_app_launches",
    "derivative_asset_materializations",
    "collision_scope_instanceable_false_writes",
    "gripper_a64_collision_enable_changes",
    "gripper_sdf_mesh_collision_enable_true_writes",
    "gripper_sdf_api_apply_sets",
    "sdf_parameter_attr_writes",
    "link5_collision_representation_changes",
    "source_geometry_stream_changes",
    "p34_or_d397_geometry_reads_for_materialization",
    "automatic_decomposition_sweeps",
    "sdf_resolution_sweeps",
    "sdf_remesh_operations",
    "simulation_context_constructions",
    "resets",
    "timeline_play_requests",
    "timeline_commit_requests",
    "timeline_raw_stop_time_zero_checks",
    "physx_stage_attaches",
    "physx_stage_detaches",
    "physx_property_queries",
    "stagecache_erase_calls",
    "simulation_app_update_pumps",
    "controlled_physics_steps",
    "public_forwards",
    "q5_commands",
    "q5_samples",
    "contact_queries",
    "cylinder_creates_or_writes",
    "target_ik_path_pose_changes",
    "unregistered_nonrepresentation_material_mass_actuator_scene_solver_setting_changes",
    "physx_convex_callback_requests",
    "sdf_tensor_or_distance_queries",
    "isaac_hydra_renders",
)
EXACT_COUNTERS = {
    "actual_worker_invocations": 1,
    "automatic_retries": 0,
    "simulation_app_launches": 1,
    "derivative_asset_materializations": 1,
    "collision_scope_instanceable_false_writes": 2,
    "gripper_a64_collision_enable_changes": 64,
    "gripper_sdf_mesh_collision_enable_true_writes": 1,
    "gripper_sdf_api_apply_sets": 1,
    "sdf_parameter_attr_writes": 7,
    "physx_stage_attaches": 1,
    "physx_stage_detaches": 1,
    "physx_property_queries": 2,
    "stagecache_erase_calls": 1,
    "timeline_raw_stop_time_zero_checks": 2,
}
ZERO_COUNTERS = (
    "link5_collision_representation_changes",
    "source_geometry_stream_changes",
    "p34_or_d397_geometry_reads_for_materialization",
    "automatic_decomposition_sweeps",
    "sdf_resolution_sweeps",
    "sdf_remesh_operations",
    "simulation_context_constructions",
    "resets",
    "timeline_play_requests",
    "timeline_commit_requests",
    "controlled_physics_steps",
    "public_forwards",
    "q5_commands",
    "q5_samples",
    "contact_queries",
    "cylinder_creates_or_writes",
    "target_ik_path_pose_changes",
    "unregistered_nonrepresentation_material_mass_actuator_scene_solver_setting_changes",
    "physx_convex_callback_requests",
    "sdf_tensor_or_distance_queries",
    "isaac_hydra_renders",
)

LINK5_RERUN_PATHS = tuple(
    f"/d400/inspection/link5_a64/part_{index:03d}" for index in range(64)
)
RERUN_PHASE_ASSIGNMENTS = {
    0: (
        "/d400/phase/source_baseline",
        "/d400/inspection/source_gripper_mesh",
    ),
    1: (
        "/d400/phase/live_configuration",
        "/d400/inspection/live_sdf_input_mesh",
        "/d400/status/api_token_attributes",
        "/d400/status/inventory_owner_query",
        *LINK5_RERUN_PATHS,
    ),
    2: (
        "/d400/phase/post_query_decision",
        "/d400/status/cook_queue",
        "/d400/status/mass_counters_instance_state",
    ),
}
RERUN_ENTITY_PATHS = tuple(
    path
    for phase in (0, 1, 2)
    for path in RERUN_PHASE_ASSIGNMENTS[phase]
)
EXPECTED_RERUN_ENTITY_SHA256 = (
    "7ce6480db2b6d337bba9b0c1ad681e5a43549e3ca14805f3cc0828a6054ea8a1"
)
EXPECTED_LINK5_RERUN_SHA256 = (
    "2fdc7021f06850791892070188dfe54dcec14748598533855fbee96aa3019699"
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def _path_set_sha(paths: tuple[str, ...] | list[str]) -> str:
    return _sha_bytes(("".join(f"{path}\n" for path in sorted(paths))).encode("utf-8"))


def _json_no_duplicates(text: str) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject)


def _read_json(path: Path) -> dict[str, Any]:
    value = _json_no_duplicates(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _phase(name: str, **fields: Any) -> None:
    if name not in PHASE_OWNER:
        raise ValueError(f"unregistered D400 phase: {name}")
    if PHASE_OWNER[name] != "controller":
        raise ValueError(f"controller attempted worker-owned phase: {name}")
    ordinal = 1
    if PHASE_PATH.is_file():
        ordinal = (
            sum(
                1
                for line in PHASE_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            + 1
        )
    row = {
        "ordinal": ordinal,
        "phase": name,
        "owner": "controller",
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _phase_rows() -> list[dict[str, Any]]:
    if not PHASE_PATH.is_file():
        return []
    return [
        _json_no_duplicates(line)
        for line in PHASE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(
    command: list[str],
    *,
    timeout_s: float = 30.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "pass": completed.returncode == 0,
        }
    except Exception as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
            "pass": False,
        }


def _validate_approval_tuple(approved_sha256: str) -> dict[str, Any]:
    if SHA256_RE.fullmatch(approved_sha256) is None:
        raise RuntimeError(
            "--approved-tuple-sha256 must be exactly 64 lowercase hex characters"
        )
    required = (PREREG_PATH, ATTESTATION_PATH, TUPLE_PATH, CONTROLLER, WORKER)
    missing = [_rel(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"D400 approval files missing: {missing}")
    tuple_sha = _sha(TUPLE_PATH)
    if tuple_sha != approved_sha256:
        raise RuntimeError(
            f"user-approved tuple SHA mismatch: {approved_sha256} != {tuple_sha}"
        )
    tuple_value = _json_no_duplicates(
        TUPLE_PATH.read_text(encoding="utf-8")
    )
    if tuple(tuple_value) != TUPLE_FIELDS or set(tuple_value) != set(TUPLE_FIELDS):
        raise RuntimeError(
            f"D400 tuple must have exact ordered fields {TUPLE_FIELDS}: {tuple_value}"
        )
    observed = {
        "preregistration_sha256": _sha(PREREG_PATH),
        "reviewed_script_attestation_sha256": _sha(ATTESTATION_PATH),
        "controller_script_sha256": _sha(CONTROLLER),
        "worker_script_sha256": _sha(WORKER),
    }
    if tuple_value != observed:
        raise RuntimeError(
            f"D400 tuple current-file hash mismatch: expected={tuple_value}, observed={observed}"
        )
    if observed["preregistration_sha256"] != EXPECTED_PREREG_SHA256:
        raise RuntimeError("D400 preregistration hash no longer matches reviewed V2")
    attestation = _json_no_duplicates(
        ATTESTATION_PATH.read_text(encoding="utf-8")
    )
    required_static_true = (
        "static_ast_parse_pass",
        "top_level_runtime_side_effect_static_scan_pass",
        "counter_key_alignment_pass",
        "phase_order_and_owner_static_contract_pass",
        "implementation_static_attestation_pass",
    )
    failed_static_fields = [
        field
        for field in required_static_true
        if attestation.get(field) is not True
    ]
    negative_static = attestation.get("negative_static_fixture_results")
    negative_static_fixtures = (
        negative_static.get("fixtures")
        if isinstance(negative_static, dict)
        else None
    )
    negative_static_ids = (
        [
            row.get("id")
            for row in negative_static_fixtures
            if isinstance(row, dict)
        ]
        if isinstance(negative_static_fixtures, list)
        else []
    )
    negative_static_exact = bool(
        isinstance(negative_static, dict)
        and negative_static.get("pass") is True
        and type(negative_static.get("passed")) is int
        and type(negative_static.get("total")) is int
        and negative_static["total"] >= 30
        and negative_static["passed"] == negative_static["total"]
        and isinstance(negative_static_fixtures, list)
        and len(negative_static_fixtures) == negative_static["total"]
        and len(negative_static_ids) == len(set(negative_static_ids))
        and all(
            isinstance(row, dict)
            and isinstance(row.get("id"), str)
            and bool(row["id"])
            and row.get("expected") == "reject"
            and row.get("observed") == "rejected"
            and row.get("pass") is True
            for row in negative_static_fixtures
        )
        and {
            row["id"] for row in negative_static_fixtures
        }.issuperset(REGISTERED_STATIC_NEGATIVE_IDS)
    )
    zero_stage_counters = attestation.get("static_stage_zero_counters")
    expected_zero_stage_counters = {
        "script_imports": 0,
        "isaac_kit_physx_launches": 0,
        "simulation_app_launches": 0,
        "derivative_asset_materializations": 0,
        "usd_stage_creations_or_writes": 0,
        "gpu_runtime_jobs": 0,
        "physics_steps": 0,
        "q5_samples": 0,
        "contact_queries": 0,
        "cylinder_creates_or_writes": 0,
    }
    if (
        failed_static_fields
        or not negative_static_exact
        or zero_stage_counters != expected_zero_stage_counters
    ):
        raise RuntimeError(
            "D400 static attestation required-field gate failed: "
            f"false_or_missing={failed_static_fields}, "
            f"negative_static_exact={negative_static_exact}, "
            "static_stage_zero_counters_exact="
            f"{zero_stage_counters == expected_zero_stage_counters}"
        )
    controller_binding = attestation.get("controller_script_path_and_sha256")
    worker_binding = attestation.get("worker_script_path_and_sha256")
    if controller_binding != {
        "path": _rel(CONTROLLER),
        "sha256": observed["controller_script_sha256"],
    }:
        raise RuntimeError("D400 controller attestation binding mismatch")
    if worker_binding != {
        "path": _rel(WORKER),
        "sha256": observed["worker_script_sha256"],
    }:
        raise RuntimeError("D400 worker attestation binding mismatch")
    if attestation.get("preregistration_sha256") != EXPECTED_PREREG_SHA256:
        raise RuntimeError("D400 attestation preregistration binding mismatch")
    return {
        "approved_tuple_sha256": approved_sha256,
        "tuple_path": _rel(TUPLE_PATH),
        "tuple_sha256": tuple_sha,
        "tuple": tuple_value,
        "observed": observed,
        "attestation_static_pass": True,
        "pass": True,
    }


def _git_status_rows() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _runtime_freeze_manifest(
    approval: dict[str, Any],
    prereg: dict[str, Any],
    launch_authority: dict[str, Any],
    worker_command: list[str],
) -> dict[str, Any]:
    if RUNTIME_MANIFEST_PATH.exists():
        raise RuntimeError("D400 runtime manifest already exists; retry refused")
    status = _git_status_rows()
    allowed = set(
        prereg["planned_runtime_contract_for_separate_approval"][
            "runtime_freeze_manifest"
        ]["allowed_dirty_paths"]
    )
    status_paths = [row[3:] for row in status if len(row) >= 4]
    unexpected = sorted(path for path in status_paths if path not in allowed)
    dirty_hashes = {
        path: _sha(REPO / path)
        for path in status_paths
        if (REPO / path).is_file()
    }
    frozen_repo = {}
    for relative, expected in prereg["frozen_input_hashes"].items():
        path = REPO / relative
        observed = _sha(path) if path.is_file() else None
        frozen_repo[relative] = {
            "expected": expected,
            "observed": observed,
            "pass": observed == expected,
        }
    installed = {}
    for absolute, expected in prereg["installed_primary_source_hashes"].items():
        path = Path(absolute)
        observed = _sha(path) if path.is_file() else None
        installed[absolute] = {
            "expected": expected,
            "observed": observed,
            "pass": observed == expected,
        }
    sidecar = {}
    for relative, frozen in prereg["d334_sidecar_before"]["files"].items():
        path = REPO / relative
        observed_sha = _sha(path) if path.is_file() else None
        observed_bytes = path.stat().st_size if path.is_file() else None
        sidecar[relative] = {
            "expected": frozen,
            "observed": {
                "sha256": observed_sha,
                "bytes": observed_bytes,
            },
            "pass": observed_sha == frozen["sha256"]
            and observed_bytes == frozen["bytes"],
        }
    head = _git_value("rev-parse", "HEAD")
    origin_master = _git_value("rev-parse", "origin/master")
    expected_git = prereg["git_baseline"]
    checks = {
        "head_exact": head == expected_git["head"],
        "origin_master_exact": origin_master == expected_git["origin_master"],
        "head_equals_origin_master": head == origin_master,
        "no_unexpected_dirty_paths": not unexpected,
        "all_frozen_repo_hashes_exact": all(
            row["pass"] for row in frozen_repo.values()
        ),
        "all_installed_primary_hashes_exact": all(
            row["pass"] for row in installed.values()
        ),
        "d334_sidecar_untouched": all(row["pass"] for row in sidecar.values()),
        "approval_tuple_gate_pass": approval["pass"] is True,
    }
    manifest = {
        "artifact": "D400_RUNTIME_FREEZE_MANIFEST_V1",
        "created_immediately_before_single_worker_spawn": True,
        "approved_tuple": approval,
        "git": {
            "head": head,
            "origin_master": origin_master,
            "status_command": "git status --short --untracked-files=all",
            "status_rows": status,
            "status_paths": status_paths,
            "allowed_dirty_paths": sorted(allowed),
            "unexpected_dirty_paths": unexpected,
            "dirty_file_sha256": dirty_hashes,
        },
        "frozen_repo_inputs": frozen_repo,
        "installed_primary_sources": installed,
        "d334_sidecar": sidecar,
        "output_root": _rel(OUT_DIR),
        "worker_command": worker_command,
        "worker_launch_authority": {
            "controller_pid": launch_authority["controller_pid"],
            "approved_tuple_sha256": launch_authority[
                "approved_tuple_sha256"
            ],
            "one_shot_nonce": launch_authority["one_shot_nonce"],
            "invocation_path": _rel(INVOCATION_PATH),
            "invocation_sha256_transport": (
                "D400_INVOCATION_SHA256 environment variable, populated "
                "only after the invocation file is exclusively written"
            ),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(RUNTIME_MANIFEST_PATH, manifest)
    return manifest


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _process_snapshot() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,pgid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) < 4:
            continue
        rows.append(
            {
                "pid": int(fields[0]),
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "command": fields[3],
            }
        )
    return rows


def _ancestor_pids(rows: list[dict[str, Any]]) -> set[int]:
    parent = {row["pid"]: row["ppid"] for row in rows}
    result = {os.getpid()}
    cursor = os.getpid()
    while cursor in parent and parent[cursor] > 0 and parent[cursor] not in result:
        cursor = parent[cursor]
        result.add(cursor)
    return result


def _gpu_snapshot() -> dict[str, Any]:
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    parsed = []
    if gpu["pass"]:
        for line in gpu["stdout"].splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 6:
                parsed.append(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "memory_total_mib": int(fields[2]),
                        "memory_used_mib": int(fields[3]),
                        "memory_free_mib": int(fields[4]),
                        "compute_capability": fields[5],
                    }
                )
    return {
        "gpu_query": gpu,
        "compute_process_query": processes,
        "gpus": parsed,
        "pass": bool(
            gpu["pass"]
            and parsed
            and parsed[0]["memory_free_mib"] >= MINIMUM_FREE_VRAM_MIB
        ),
    }


def _extension_package_version(path: Path) -> str | None:
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
        version = value.get("package", {}).get("version")
        return str(version) if version is not None else None
    except (OSError, tomllib.TOMLDecodeError, TypeError, AttributeError):
        return None


def _os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(
            encoding="utf-8"
        ).splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key] = value.strip().strip('"')
    except OSError:
        pass
    return result


def _environment_gate(prereg: dict[str, Any]) -> dict[str, Any]:
    process_rows = _process_snapshot()
    excluded = _ancestor_pids(process_rows)
    controller_name = CONTROLLER.name
    worker_name = WORKER.name
    output_text = str(OUT_DIR)
    conflicts = [
        row
        for row in process_rows
        if row["pid"] not in excluded
        and (
            controller_name in row["command"]
            or worker_name in row["command"]
            or output_text in row["command"]
        )
    ]
    unrelated_isaac = [
        row
        for row in process_rows
        if row["pid"] not in excluded
        and re.search(r"(?i)(isaac|kit(?:\\.kit)?|omni)", row["command"])
        and row not in conflicts
    ]
    gpu = _gpu_snapshot()
    packages = {
        "isaacsim": _package_version("isaacsim"),
        "isaaclab": _package_version("isaaclab"),
        "numpy": _package_version("numpy"),
        "psutil": _package_version("psutil"),
        "rerun-sdk": _package_version("rerun-sdk"),
    }
    installed_paths = [
        Path(path) for path in prereg["installed_primary_source_hashes"]
    ]
    omni_physx_toml = next(
        (
            path
            for path in installed_paths
            if "extscache/omni.physx-" in str(path)
            and str(path).endswith("/config/extension.toml")
        ),
        None,
    )
    physx_schema_toml = next(
        (
            path
            for path in installed_paths
            if "extscache/omni.usd.schema.physx-" in str(path)
            and str(path).endswith("/config/extension.toml")
        ),
        None,
    )
    extension_versions = {
        "omni_physx": (
            _extension_package_version(omni_physx_toml)
            if omni_physx_toml is not None
            else None
        ),
        "physx_schema": (
            _extension_package_version(physx_schema_toml)
            if physx_schema_toml is not None
            else None
        ),
    }
    expected = prereg[
        "installed_stack_expected_and_to_be_reprobed_at_runtime"
    ]
    os_release = _os_release()
    native_plugin_bytes = (
        PHYSX_NATIVE_PLUGIN.read_bytes()
        if PHYSX_NATIVE_PLUGIN.is_file()
        else b""
    )
    physx_engine_provenance = {
        "classification": (
            "native_plugin_embedded_evidence_not_public_runtime_api"
        ),
        "path": str(PHYSX_NATIVE_PLUGIN),
        "sha256": (
            _sha(PHYSX_NATIVE_PLUGIN)
            if PHYSX_NATIVE_PLUGIN.is_file()
            else None
        ),
        "expected_sha256": EXPECTED_PHYSX_NATIVE_PLUGIN_SHA256,
        "embedded_5_6_1_marker_count": native_plugin_bytes.count(b"5.6.1"),
        "public_python_runtime_getter_available": False,
        "engine_version_claim": "5.6.1",
        "authority_limit": (
            "Pinned native-binary provenance only. The supported runtime "
            "extension-manager probe reports omni.physx 107.3.26, not the "
            "internal PhysX SDK version."
        ),
    }
    rerun_version = _run([str(RERUN_CLI), "--version"])
    checks = {
        "isaac_python_exact": ISAAC_PYTHON.is_file(),
        "rerun_cli_exact_path": RERUN_CLI.is_file(),
        "ubuntu_22_04_exact": os_release.get("VERSION_ID") == "22.04",
        "isaac_sim_pin": packages["isaacsim"] == expected["isaac_sim"],
        "isaac_lab_pin": packages["isaaclab"] == expected["isaac_lab"],
        "omni_physx_extension_pin": extension_versions["omni_physx"]
        == expected["omni_physx"],
        "physx_schema_extension_pin": extension_versions["physx_schema"]
        == expected["physx_schema"],
        "physx_native_plugin_hash_exact": (
            physx_engine_provenance["sha256"]
            == EXPECTED_PHYSX_NATIVE_PLUGIN_SHA256
        ),
        "physx_native_plugin_embeds_5_6_1": (
            physx_engine_provenance["embedded_5_6_1_marker_count"] > 0
        ),
        "numpy_pin": packages["numpy"] == "1.26.0",
        "psutil_pin": packages["psutil"] == "5.9.8",
        "rerun_sdk_pin": packages["rerun-sdk"] == RERUN_VERSION,
        "rerun_cli_pin": rerun_version["pass"]
        and RERUN_VERSION
        in (rerun_version["stdout"] + rerun_version["stderr"]),
        "no_preexisting_d400_process": not conflicts,
        "minimum_free_vram": gpu["pass"],
        "gpu_model_exact": bool(
            gpu["gpus"]
            and gpu["gpus"][0]["name"]
            == "NVIDIA GeForce RTX 4090 Laptop GPU"
        ),
        "gpu_compute_capability_exact": bool(
            gpu["gpus"]
            and gpu["gpus"][0]["compute_capability"] == "8.9"
        ),
    }
    return {
        "os_release": os_release,
        "packages": packages,
        "extension_versions": extension_versions,
        "expected_stack": expected,
        "physx_engine_provenance": physx_engine_provenance,
        "launched_worker_probe_boundary": (
            "The worker must re-probe the supported omni.physx extension "
            "version through Kit's extension manager. It must not relabel "
            "107.3.26 as SDK 5.6.1."
        ),
        "rerun_cli_version": rerun_version,
        "gpu": gpu,
        "process_snapshot": process_rows,
        "excluded_current_and_ancestor_pids": sorted(excluded),
        "d400_conflicts": conflicts,
        "unrelated_isaac_kit_diagnostic_only": unrelated_isaac,
        "unrelated_processes_signaled": 0,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _counter_gate(counters: Any) -> dict[str, Any]:
    is_mapping = isinstance(counters, dict)
    keys = tuple(counters) if is_mapping else ()
    types = (
        {key: type(value).__name__ for key, value in counters.items()}
        if is_mapping
        else {}
    )
    integer_types = bool(
        is_mapping and all(type(value) is int for value in counters.values())
    )
    exact = {
        key: bool(
            is_mapping
            and key in counters
            and type(counters[key]) is int
            and counters[key] == expected
        )
        for key, expected in EXACT_COUNTERS.items()
    }
    zeros = {
        key: bool(
            is_mapping
            and key in counters
            and type(counters[key]) is int
            and counters[key] == 0
        )
        for key in ZERO_COUNTERS
    }
    pump = bool(
        is_mapping
        and "simulation_app_update_pumps" in counters
        and type(counters["simulation_app_update_pumps"]) is int
        and 1 <= counters["simulation_app_update_pumps"] <= MAX_APP_UPDATE_PUMPS
    )
    checks = {
        "mapping": is_mapping,
        "exact_36_keys_in_order": keys == COUNTER_KEYS,
        "all_values_exact_int_not_bool": integer_types,
        "exact_14": all(exact.values()) and len(exact) == 14,
        "zero_21": all(zeros.values()) and len(zeros) == 21,
        "one_range_pump": pump,
    }
    return {
        "keys": list(keys),
        "types": types,
        "exact": exact,
        "zero": zeros,
        "simulation_app_update_pumps": (
            counters.get("simulation_app_update_pumps")
            if is_mapping
            else None
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _phase_audit(
    rows: list[dict[str, Any]],
    *,
    technical_pass_branch: bool,
    observability_pass: bool,
) -> dict[str, Any]:
    names = [row.get("phase") for row in rows]
    canonical_order = list(MANDATORY_PHASE_ORDER)
    close_index = canonical_order.index("simulation_app_close_start") + 1
    canonical_order[close_index:close_index] = list(OPTIONAL_PHASES)
    canonical_rank = {
        name: index for index, name in enumerate(canonical_order)
    }
    pre_observed = [name for name in names if name in CONTROLLER_PRE_PHASES]
    worker_observed = [name for name in names if name in WORKER_PHASES]
    optional_observed = [name for name in names if name in OPTIONAL_PHASES]
    rerun_order = (
        "rerun_save_only_finalize_end_if_technical_pass",
        "rerun_verify_and_visual_inspection_end_if_technical_pass",
    )
    rerun_observed = [name for name in names if name in rerun_order]
    required = [
        "supervisor_preflight_start",
        "external_kit_log_audit_end",
        "external_supervisor_end",
        "completion_summary_end",
    ]
    if technical_pass_branch:
        required.extend(CONTROLLER_PRE_PHASES)
        required.extend(WORKER_PHASES)
        required.append("technical_pass_branch_start")
    if observability_pass:
        required.extend(rerun_order)
    required = [
        name for name in canonical_order if name in set(required)
    ]
    ranks = [
        canonical_rank[name]
        for name in names
        if name in canonical_rank
    ]
    positions = [
        names.index(name) if name in names else None for name in required
    ]
    required_counts = {name: names.count(name) for name in required}
    required_owner_checks = {
        name: all(
            row.get("owner") == PHASE_OWNER[name]
            for row in rows
            if row.get("phase") == name
        )
        for name in required
    }
    optional_counts = {name: names.count(name) for name in OPTIONAL_PHASES}
    rerun_names = set(rerun_order)
    monotonic_values = [row.get("monotonic_ns") for row in rows]
    controller_pre_prefix = (
        pre_observed == list(CONTROLLER_PRE_PHASES[: len(pre_observed)])
    )
    worker_requires_spawn = (
        not worker_observed or "worker_spawn_once" in pre_observed
    )
    technical_full_prefix = bool(
        pre_observed == list(CONTROLLER_PRE_PHASES)
        and worker_observed == list(WORKER_PHASES)
    )
    technical_fail_tail_exact = bool(
        not technical_pass_branch
        and names[-3:]
        == [
            "external_kit_log_audit_end",
            "external_supervisor_end",
            "completion_summary_end",
        ]
    )
    checks = {
        "rows_are_objects": all(isinstance(row, dict) for row in rows),
        "row_scalar_types_exact": all(
            type(row.get("ordinal")) is int
            and isinstance(row.get("phase"), str)
            and isinstance(row.get("owner"), str)
            and type(row.get("pid")) is int
            and row["pid"] > 0
            and type(row.get("monotonic_ns")) is int
            and row["monotonic_ns"] > 0
            for row in rows
        ),
        "ordinals_contiguous": [row.get("ordinal") for row in rows]
        == list(range(1, len(rows) + 1)),
        "monotonic_ns_strictly_increasing": all(
            type(left) is int
            and type(right) is int
            and left < right
            for left, right in zip(
                monotonic_values, monotonic_values[1:]
            )
        ),
        "all_observed_phases_in_registered_order": ranks
        == sorted(ranks),
        "all_observed_phases_at_most_once": all(
            names.count(name) <= 1 for name in PHASE_OWNER
        ),
        "required_exactly_once": all(
            count == 1 for count in required_counts.values()
        ),
        "required_owners_exact": all(required_owner_checks.values()),
        "all_observed_owners_exact": all(
            row.get("phase") in PHASE_OWNER
            and row.get("owner") == PHASE_OWNER[row.get("phase")]
            for row in rows
        ),
        "optional_max_once": all(
            count <= 1 for count in optional_counts.values()
        ),
        "no_unknown_phase": all(
            name in PHASE_OWNER for name in names
        ),
        "controller_pre_is_forward_prefix": controller_pre_prefix,
        "worker_phases_require_spawn": worker_requires_spawn,
        "worker_failure_phases_are_forward_subsequence": worker_observed
        == sorted(worker_observed, key=lambda name: canonical_rank[name]),
        "optional_close_return_requires_close_start": (
            not optional_observed
            or "simulation_app_close_start" in worker_observed
        ),
        "technical_pass_requires_full_pre_and_worker": (
            not technical_pass_branch or technical_full_prefix
        ),
        "technical_fail_has_exact_controller_tail": (
            technical_pass_branch or technical_fail_tail_exact
        ),
        "technical_markers_match_branch": technical_pass_branch
        == ("technical_pass_branch_start" in names),
        "rerun_markers_absent_on_technical_fail": (
            technical_pass_branch
            or rerun_names.isdisjoint(names)
        ),
        "rerun_markers_are_forward_prefix": rerun_observed
        == list(rerun_order[: len(rerun_observed)]),
        "rerun_markers_complete_on_observability_pass": (
            not observability_pass or rerun_names.issubset(names)
        ),
        "observability_cannot_pass_without_technical_pass": (
            not observability_pass or technical_pass_branch
        ),
    }
    return {
        "names": names,
        "required": required,
        "required_positions": positions,
        "required_counts": required_counts,
        "owner_checks": required_owner_checks,
        "optional_counts": optional_counts,
        "checks": checks,
        "technical_pass_branch": technical_pass_branch,
        "observability_pass": observability_pass,
        "pass": all(checks.values()),
    }


def _pre_rerun_phase_audit(
    rows_with_projected_supervisor_end: list[dict[str, Any]],
) -> dict[str, Any]:
    names = [row.get("phase") for row in rows_with_projected_supervisor_end]
    expected = [
        *CONTROLLER_PRE_PHASES,
        *WORKER_PHASES,
    ]
    if "simulation_app_close_returned" in names:
        insert_at = expected.index("simulation_app_close_start") + 1
        expected[insert_at:insert_at] = ["simulation_app_close_returned"]
    expected.extend(
        ("external_kit_log_audit_end", "external_supervisor_end")
    )
    owners = {
        row.get("phase"): row.get("owner")
        for row in rows_with_projected_supervisor_end
    }
    monotonic_values = [
        row.get("monotonic_ns")
        for row in rows_with_projected_supervisor_end
    ]
    checks = {
        "exact_technical_prefix_through_supervisor": names == expected,
        "ordinals_contiguous": [
            row.get("ordinal") for row in rows_with_projected_supervisor_end
        ]
        == list(range(1, len(rows_with_projected_supervisor_end) + 1)),
        "all_owners_exact": all(
            owners.get(name) == PHASE_OWNER[name] for name in expected
        ),
        "row_scalar_types_exact": all(
            type(row.get("ordinal")) is int
            and type(row.get("pid")) is int
            and row["pid"] > 0
            and type(row.get("monotonic_ns")) is int
            and row["monotonic_ns"] > 0
            for row in rows_with_projected_supervisor_end
        ),
        "monotonic_ns_strictly_increasing": all(
            type(left) is int
            and type(right) is int
            and left < right
            for left, right in zip(
                monotonic_values, monotonic_values[1:]
            )
        ),
        "no_technical_or_rerun_or_completion_marker_yet": not any(
            name
            in {
                "technical_pass_branch_start",
                "rerun_save_only_finalize_end_if_technical_pass",
                "rerun_verify_and_visual_inspection_end_if_technical_pass",
                "completion_summary_end",
            }
            for name in names
        ),
    }
    return {
        "names": names,
        "expected": expected,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _offline_negative_controls() -> dict[str, Any]:
    baseline_counters = {
        key: (
            1
            if key == "simulation_app_update_pumps"
            else EXACT_COUNTERS.get(key, 0)
        )
        for key in COUNTER_KEYS
    }
    valid_rows = [
        {
            "ordinal": index + 1,
            "phase": name,
            "owner": PHASE_OWNER[name],
            "pid": 1234,
            "monotonic_ns": index + 1,
        }
        for index, name in enumerate(MANDATORY_PHASE_ORDER)
    ]
    baseline_counter_gate = _counter_gate(baseline_counters)
    baseline_phase_audit = _phase_audit(
        valid_rows,
        technical_pass_branch=True,
        observability_pass=True,
    )
    controls = []

    def add(identifier: str, rejected: bool, observed: Any) -> None:
        controls.append(
            {
                "id": identifier,
                "expected": "reject",
                "rejected": bool(rejected),
                "observed": observed,
                "pass": bool(rejected),
            }
        )

    missing = dict(baseline_counters)
    missing.pop("q5_samples")
    add("counter_missing_key", not _counter_gate(missing)["pass"], _counter_gate(missing))
    extra = dict(baseline_counters)
    extra["unregistered"] = 0
    add("counter_extra_key", not _counter_gate(extra)["pass"], _counter_gate(extra))
    boolean = dict(baseline_counters)
    boolean["q5_samples"] = False
    add("counter_bool_not_int", not _counter_gate(boolean)["pass"], _counter_gate(boolean))
    wrong = dict(baseline_counters)
    wrong["physx_stage_attaches"] = 2
    add("counter_wrong_exact", not _counter_gate(wrong)["pass"], _counter_gate(wrong))
    pump_zero = dict(baseline_counters)
    pump_zero["simulation_app_update_pumps"] = 0
    add("counter_pump_zero", not _counter_gate(pump_zero)["pass"], _counter_gate(pump_zero))
    pump_high = dict(baseline_counters)
    pump_high["simulation_app_update_pumps"] = MAX_APP_UPDATE_PUMPS + 1
    add("counter_pump_overflow", not _counter_gate(pump_high)["pass"], _counter_gate(pump_high))
    forbidden = dict(baseline_counters)
    forbidden["controlled_physics_steps"] = 1
    add("counter_physics_nonzero", not _counter_gate(forbidden)["pass"], _counter_gate(forbidden))
    missing_phase = valid_rows[:-1]
    add(
        "phase_missing",
        not _phase_audit(
            missing_phase,
            technical_pass_branch=True,
            observability_pass=True,
        )["pass"],
        _phase_audit(
            missing_phase,
            technical_pass_branch=True,
            observability_pass=True,
        ),
    )
    duplicate_phase = valid_rows + [dict(valid_rows[-1], ordinal=len(valid_rows) + 1)]
    add(
        "phase_duplicate",
        not _phase_audit(
            duplicate_phase,
            technical_pass_branch=True,
            observability_pass=True,
        )["pass"],
        _phase_audit(
            duplicate_phase,
            technical_pass_branch=True,
            observability_pass=True,
        ),
    )
    reordered = list(valid_rows)
    reordered[10], reordered[11] = reordered[11], reordered[10]
    add(
        "phase_reordered",
        not _phase_audit(
            reordered,
            technical_pass_branch=True,
            observability_pass=True,
        )["pass"],
        _phase_audit(
            reordered,
            technical_pass_branch=True,
            observability_pass=True,
        ),
    )
    wrong_owner = [dict(row) for row in valid_rows]
    wrong_owner[6]["owner"] = "controller"
    add(
        "phase_wrong_owner",
        not _phase_audit(
            wrong_owner,
            technical_pass_branch=True,
            observability_pass=True,
        )["pass"],
        _phase_audit(
            wrong_owner,
            technical_pass_branch=True,
            observability_pass=True,
        ),
    )
    add(
        "approval_hash_uppercase",
        SHA256_RE.fullmatch("A" * 64) is None,
        {"value": "A" * 64},
    )
    add(
        "approval_hash_short",
        SHA256_RE.fullmatch("0" * 63) is None,
        {"length": 63},
    )
    try:
        _json_no_duplicates('{"a":1,"a":2}')
        duplicate_rejected = False
        duplicate_observed = None
    except ValueError as error:
        duplicate_rejected = True
        duplicate_observed = str(error)
    add("tuple_duplicate_json_key", duplicate_rejected, duplicate_observed)
    missing_tuple = {key: "0" * 64 for key in TUPLE_FIELDS[:-1]}
    add(
        "tuple_missing_field",
        tuple(missing_tuple) != TUPLE_FIELDS,
        list(missing_tuple),
    )
    extra_tuple = {key: "0" * 64 for key in TUPLE_FIELDS}
    extra_tuple["extra"] = "0" * 64
    add(
        "tuple_extra_field",
        set(extra_tuple) != set(TUPLE_FIELDS),
        list(extra_tuple),
    )
    add(
        "attestation_string_true",
        "true" is not True,
        {"implementation_static_attestation_pass": "true"},
    )
    add(
        "cook_zero_to_zero_false_pass",
        not (0 > 0 and 0 == 0 and 0 == 0),
        {"scheduled_delta": 0, "finished_delta": 0, "running": 0},
    )
    return {
        "artifact": "D400_RUNTIME_OFFLINE_NEGATIVE_CONTROLS_V1",
        "positive_baselines": {
            "counter_gate": baseline_counter_gate,
            "phase_audit": baseline_phase_audit,
            "pass": (
                baseline_counter_gate["pass"]
                and baseline_phase_audit["pass"]
            ),
        },
        "controls": controls,
        "passed": sum(row["pass"] for row in controls),
        "total": len(controls),
        "pass": bool(
            controls
            and baseline_counter_gate["pass"]
            and baseline_phase_audit["pass"]
            and all(row["pass"] for row in controls)
        ),
    }


def _worker_command(
    launch_authority: dict[str, Any],
) -> list[str]:
    return [
        str(ISAAC_PYTHON),
        "-B",
        str(WORKER),
        "--out-dir",
        str(OUT_DIR),
        "--prereg",
        str(PREREG_PATH),
        "--invocation",
        str(INVOCATION_PATH),
        "--controller-pid",
        str(launch_authority["controller_pid"]),
        "--approved-tuple-sha256",
        launch_authority["approved_tuple_sha256"],
        "--one-shot-nonce",
        launch_authority["one_shot_nonce"],
        "--headless",
    ]


def _process_group_residue(pgid: int) -> list[dict[str, Any]]:
    return [row for row in _process_snapshot() if row["pgid"] == pgid]


def _safe_process_group_residue(
    pgid: int,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        return _process_group_residue(pgid), None
    except Exception as error:
        return [], f"{type(error).__name__}: {error}"


def _gpu_compute_rows() -> list[dict[str, Any]]:
    result = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    if result["pass"]:
        for line in result["stdout"].splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 3 and fields[0].isdigit():
                rows.append(
                    {
                        "pid": int(fields[0]),
                        "process_name": fields[1],
                        "used_memory_mib": fields[2],
                    }
                )
    return rows


def _prefix_hash_check(preclose: dict[str, Any]) -> bool:
    byte_count = preclose.get("phase_prefix_bytes")
    expected = preclose.get("phase_prefix_sha256")
    if type(byte_count) is not int or byte_count <= 0 or not isinstance(expected, str):
        return False
    payload = PHASE_PATH.read_bytes()
    return len(payload) >= byte_count and _sha_bytes(payload[:byte_count]) == expected


def _audit_kit_log() -> dict[str, Any]:
    text = KIT_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    parsed_stdout_phase_rows = []
    stdout_phase_parse_errors = []
    for line_number, line in enumerate(lines, start=1):
        if not line.startswith("D400_PHASE "):
            continue
        try:
            value = _json_no_duplicates(line[len("D400_PHASE ") :])
            parsed_stdout_phase_rows.append(value)
        except Exception as error:
            stdout_phase_parse_errors.append(
                {
                    "line_number": line_number,
                    "error": f"{type(error).__name__}: {error}",
                    "text": line,
                }
            )
    phase_file_rows = _phase_rows()
    boundary_names = (
        "physx_stage_attach_start",
        "worker_preclose_sentinel_written",
    )
    boundary_comparisons = {}
    for name in boundary_names:
        file_matches = [
            row for row in phase_file_rows if row.get("phase") == name
        ]
        stdout_matches = [
            row
            for row in parsed_stdout_phase_rows
            if row.get("phase") == name
        ]
        boundary_comparisons[name] = {
            "phase_file_count": len(file_matches),
            "stdout_count": len(stdout_matches),
            "phase_file_row": (
                file_matches[0] if len(file_matches) == 1 else None
            ),
            "stdout_row": (
                stdout_matches[0] if len(stdout_matches) == 1 else None
            ),
            "exact": bool(
                len(file_matches) == 1
                and len(stdout_matches) == 1
                and file_matches[0] == stdout_matches[0]
            ),
        }
    starts = [
        index
        for index, line in enumerate(lines)
        if '"phase": "physx_stage_attach_start"' in line
    ]
    ends = [
        index
        for index, line in enumerate(lines)
        if '"phase": "worker_preclose_sentinel_written"' in line
    ]
    valid_window = bool(
        len(starts) == 1 and len(ends) == 1 and starts[0] <= ends[0]
    )
    window = lines[starts[0] : ends[0] + 1] if valid_window else []
    window_start = starts[0] if valid_window else 0

    def with_context(index: int, line: str) -> dict[str, Any]:
        lower = max(0, index - 2)
        upper = min(len(window), index + 3)
        return {
            "line_number": window_start + index + 1,
            "text": line,
            "context_start_line": window_start + lower + 1,
            "context_end_line": window_start + upper,
            "context": window[lower:upper],
        }

    errors = [
        with_context(index, line)
        for index, line in enumerate(window)
        if re.search(r"(?i)\b(?:ERROR|FATAL)\b", line)
    ] if valid_window else []
    warning_subject = re.compile(
        r"(?i)(sdf|cook(?:ing)?|collision|fallback|triangle.?mesh|"
        r"rigid(?:body)?|articulation|material|/World/Robot/gripper_link|"
        r"node_STL_BINARY_)"
    )
    warnings = [
        with_context(index, line)
        for index, line in enumerate(window)
        if re.search(r"(?i)\bWARN(?:ING)?\b", line)
        and warning_subject.search(line)
    ] if valid_window else []
    start_context_row = boundary_comparisons[
        "physx_stage_attach_start"
    ]["phase_file_row"]
    context_checks = {
        "case_output_path_exact": isinstance(start_context_row, dict)
        and start_context_row.get("case_output_path") == _rel(OUT_DIR),
        "link5_rigid_prim_exact": isinstance(start_context_row, dict)
        and start_context_row.get("link5_rigid_prim")
        == "/World/Robot/link5",
        "gripper_rigid_prim_exact": isinstance(start_context_row, dict)
        and start_context_row.get("gripper_rigid_prim")
        == "/World/Robot/gripper_link",
        "sdf_mesh_prim_exact": isinstance(start_context_row, dict)
        and start_context_row.get("sdf_mesh_prim")
        == (
            "/World/Robot/gripper_link/collisions/gripper_link/"
            "node_STL_BINARY_/mesh"
        ),
        "all_failure_matches_preserve_surrounding_context": all(
            len(row["context"]) >= 1 for row in [*errors, *warnings]
        ),
    }
    checks = {
        "window_markers_exactly_once_and_ordered": valid_window,
        "stdout_phase_json_parse_exact": not stdout_phase_parse_errors,
        "boundary_phase_file_stdout_rows_exact": all(
            row["exact"] for row in boundary_comparisons.values()
        ),
        "no_error_or_fatal": not errors,
        "no_registered_subject_warning": not warnings,
        "case_path_and_prim_context_exact": all(
            context_checks.values()
        ),
    }
    return {
        "artifact": "D400_AUTHORITATIVE_KIT_LOG_WINDOW_AUDIT_V1",
        "path": _rel(KIT_LOG_PATH),
        "sha256": _sha(KIT_LOG_PATH),
        "line_count": len(lines),
        "window_start_indices": starts,
        "window_end_indices": ends,
        "window_line_count": len(window),
        "stdout_phase_row_count": len(parsed_stdout_phase_rows),
        "stdout_phase_parse_errors": stdout_phase_parse_errors,
        "boundary_phase_file_stdout_comparisons": boundary_comparisons,
        "error_or_fatal_lines": errors,
        "matching_warning_lines": warnings,
        "case_path_and_prim_context_checks": context_checks,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _post_worker_binding_gate(
    *,
    approval: dict[str, Any],
    manifest: dict[str, Any],
    invocation: dict[str, Any],
    environment: dict[str, Any],
    negative: dict[str, Any],
    invocation_sha256_at_write: str,
) -> dict[str, Any]:
    current_tuple_files = {
        "preregistration_sha256": (
            _sha(PREREG_PATH) if PREREG_PATH.is_file() else None
        ),
        "reviewed_script_attestation_sha256": (
            _sha(ATTESTATION_PATH) if ATTESTATION_PATH.is_file() else None
        ),
        "controller_script_sha256": (
            _sha(CONTROLLER) if CONTROLLER.is_file() else None
        ),
        "worker_script_sha256": _sha(WORKER) if WORKER.is_file() else None,
    }
    frozen_repo_current = {
        relative: _sha(REPO / relative)
        if (REPO / relative).is_file()
        else None
        for relative in manifest["frozen_repo_inputs"]
    }
    installed_current = {
        absolute: _sha(Path(absolute)) if Path(absolute).is_file() else None
        for absolute in manifest["installed_primary_sources"]
    }
    sidecar_current = {
        relative: {
            "sha256": (
                _sha(REPO / relative)
                if (REPO / relative).is_file()
                else None
            ),
            "bytes": (
                (REPO / relative).stat().st_size
                if (REPO / relative).is_file()
                else None
            ),
        }
        for relative in manifest["d334_sidecar"]
    }
    bindings = invocation.get("runtime_authority_bindings", {})
    invocation_file_payload = (
        _read_json(INVOCATION_PATH)
        if INVOCATION_PATH.is_file()
        else None
    )
    checks = {
        "approved_tuple_still_exact": (
            TUPLE_PATH.is_file()
            and _sha(TUPLE_PATH) == approval["approved_tuple_sha256"]
        ),
        "tuple_member_files_still_exact": current_tuple_files
        == approval["observed"],
        "runtime_manifest_file_still_exact": (
            RUNTIME_MANIFEST_PATH.is_file()
            and _sha(RUNTIME_MANIFEST_PATH)
            == bindings.get("runtime_manifest_sha256")
        ),
        "runtime_manifest_payload_still_exact": _json_sha256(manifest)
        == bindings.get("runtime_manifest_payload_sha256"),
        "invocation_file_hash_still_exact": (
            INVOCATION_PATH.is_file()
            and _sha(INVOCATION_PATH) == invocation_sha256_at_write
        ),
        "invocation_file_payload_still_exact": invocation_file_payload
        == invocation,
        "invocation_environment_payload_exact": (
            invocation.get("pre_spawn_package_gpu_and_process_attestation")
            == environment
            and _json_sha256(environment)
            == bindings.get("environment_payload_sha256")
        ),
        "invocation_negative_controls_payload_exact": (
            invocation.get("pre_spawn_offline_negative_controls") == negative
            and _json_sha256(negative)
            == bindings.get("negative_controls_payload_sha256")
        ),
        "manifest_pass_was_true": manifest.get("pass") is True,
        "approval_pass_was_true": approval.get("pass") is True,
        "head_still_exact": _git_value("rev-parse", "HEAD")
        == manifest["git"]["head"],
        "origin_master_still_exact": _git_value(
            "rev-parse", "origin/master"
        )
        == manifest["git"]["origin_master"],
        "all_frozen_repo_inputs_still_exact": all(
            frozen_repo_current[relative] == row["observed"]
            for relative, row in manifest["frozen_repo_inputs"].items()
        ),
        "all_installed_primary_sources_still_exact": all(
            installed_current[absolute] == row["observed"]
            for absolute, row in manifest[
                "installed_primary_sources"
            ].items()
        ),
        "d334_sidecar_still_exact": all(
            sidecar_current[relative] == row["observed"]
            for relative, row in manifest["d334_sidecar"].items()
        ),
    }
    return {
        "current_tuple_member_hashes": current_tuple_files,
        "invocation_sha256_at_write": invocation_sha256_at_write,
        "invocation_sha256_current": (
            _sha(INVOCATION_PATH)
            if INVOCATION_PATH.is_file()
            else None
        ),
        "frozen_repo_current": frozen_repo_current,
        "installed_primary_sources_current": installed_current,
        "d334_sidecar_current": sidecar_current,
        "runtime_authority_bindings": bindings,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _bounded_case_group_cleanup(
    process: subprocess.Popen[str],
    pgid: int,
    *,
    reason: str,
) -> dict[str, Any]:
    before, before_error = _safe_process_group_residue(pgid)
    record: dict[str, Any] = {
        "reason": reason,
        "pid": process.pid,
        "pgid": pgid,
        "pgid_equals_spawned_pid": pgid == process.pid,
        "before": before,
        "residue_query_errors": (
            [{"stage": "before", "error": before_error}]
            if before_error is not None
            else []
        ),
        "sigterm_sent": False,
        "sigkill_sent": False,
        "term_error": None,
        "kill_error": None,
        "parent_reaped": False,
        "after": None,
        "pass": False,
    }
    if pgid != process.pid:
        record["term_error"] = (
            "refused signal because start_new_session invariant "
            "pgid==spawned pid failed"
        )
        after, after_error = _safe_process_group_residue(pgid)
        record["after"] = after
        if after_error is not None:
            record["residue_query_errors"].append(
                {"stage": "refused_after", "error": after_error}
            )
        return record

    if record["before"] or process.poll() is None:
        try:
            os.killpg(pgid, signal.SIGTERM)
            record["sigterm_sent"] = True
        except ProcessLookupError:
            pass
        except Exception as error:
            record["term_error"] = f"{type(error).__name__}: {error}"

    term_deadline = time.monotonic() + SIGTERM_GRACE_S
    while time.monotonic() < term_deadline:
        residue, residue_error = _safe_process_group_residue(pgid)
        if residue_error is not None:
            record["residue_query_errors"].append(
                {"stage": "term_wait", "error": residue_error}
            )
        if process.poll() is not None and not residue and residue_error is None:
            break
        time.sleep(0.1)

    residue_after_term, residue_after_term_error = (
        _safe_process_group_residue(pgid)
    )
    if residue_after_term_error is not None:
        record["residue_query_errors"].append(
            {"stage": "after_term", "error": residue_after_term_error}
        )
    if (
        residue_after_term
        or process.poll() is None
        or residue_after_term_error is not None
    ):
        try:
            os.killpg(pgid, signal.SIGKILL)
            record["sigkill_sent"] = True
        except ProcessLookupError:
            pass
        except Exception as error:
            record["kill_error"] = f"{type(error).__name__}: {error}"

    kill_deadline = time.monotonic() + SIGKILL_WAIT_S
    while time.monotonic() < kill_deadline:
        residue, residue_error = _safe_process_group_residue(pgid)
        if residue_error is not None:
            record["residue_query_errors"].append(
                {"stage": "kill_wait", "error": residue_error}
            )
        if process.poll() is not None and not residue and residue_error is None:
            break
        time.sleep(0.1)

    if process.poll() is None:
        try:
            process.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            pass
    else:
        process.wait()
    record["parent_reaped"] = process.poll() is not None
    after, after_error = _safe_process_group_residue(pgid)
    record["after"] = after
    if after_error is not None:
        record["residue_query_errors"].append(
            {"stage": "after", "error": after_error}
        )
    record["pass"] = bool(
        record["pgid_equals_spawned_pid"]
        and record["term_error"] is None
        and record["kill_error"] is None
        and record["parent_reaped"]
        and not record["after"]
        and not record["residue_query_errors"]
    )
    return record


def _supervise_worker(
    environment: dict[str, Any],
    approval: dict[str, Any],
    manifest: dict[str, Any],
    negative: dict[str, Any],
    launch_authority: dict[str, Any],
    command: list[str],
) -> dict[str, Any]:
    for path in (
        INVOCATION_PATH,
        CLAIM_PATH,
        KIT_LOG_PATH,
        RAW_PATH,
        PRECLOSE_PATH,
        OWNER_EVIDENCE_PATH,
        SUPERVISOR_PATH,
        COMPLETION_PATH,
        COLLISION_ASSET_ROOT,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION_PATH,
        BOARD_PATH,
        RERUN_SCREENSHOT_PATH,
        RERUN_RECEIPT_PATH,
        MANUAL_INSPECTION_PATH,
    ):
        if path.exists():
            raise RuntimeError(f"D400 one-shot runtime path already claimed: {_rel(path)}")
    invocation = {
        "artifact": "D400_SINGLE_WORKER_SPAWN_REQUEST_V1",
        "command": command,
        "cwd": str(REPO),
        "controller_pid": launch_authority["controller_pid"],
        "one_shot_nonce": launch_authority["one_shot_nonce"],
        "worker_sha256": _sha(WORKER),
        "controller_sha256": _sha(CONTROLLER),
        "preregistration_sha256": _sha(PREREG_PATH),
        "approved_tuple_sha256": _sha(TUPLE_PATH),
        "worker_spawn_request_budget": 1,
        "actual_worker_invocations_before_popen": 0,
        "automatic_retries": 0,
        "start_new_session": True,
        "headless": True,
        "total_watchdog_s": TOTAL_WATCHDOG_S,
        "inactivity_watchdog_s": INACTIVITY_WATCHDOG_S,
        "sigterm_grace_s": SIGTERM_GRACE_S,
        "sigkill_wait_s": SIGKILL_WAIT_S,
        "activity_definition": (
            "strict increase in d400_phase_markers.jsonl bytes or d400_kit_log.txt bytes"
        ),
        "environment_overrides": {"OMNI_KIT_ACCEPT_EULA": "YES"},
        "pre_spawn_package_gpu_and_process_attestation": environment,
        "pre_spawn_offline_negative_controls": negative,
        "runtime_authority_bindings": {
            "approved_tuple_sha256": approval[
                "approved_tuple_sha256"
            ],
            "approved_tuple_members": approval["observed"],
            "runtime_manifest_sha256": _sha(RUNTIME_MANIFEST_PATH),
            "runtime_manifest_payload_sha256": _json_sha256(manifest),
            "environment_payload_sha256": _json_sha256(environment),
            "negative_controls_payload_sha256": _json_sha256(negative),
        },
        "invocation_sha256_transport": (
            "D400_INVOCATION_SHA256 environment variable set after this "
            "exclusive file write; the digest is not self-embedded"
        ),
    }
    _write_json_x(INVOCATION_PATH, invocation)
    invocation_sha256_at_write = _sha(INVOCATION_PATH)
    env = os.environ.copy()
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["D400_INVOCATION_SHA256"] = invocation_sha256_at_write
    _phase(
        "worker_spawn_once",
        invocation_sha256=_sha(INVOCATION_PATH),
        marker_semantics=(
            "single Popen request boundary; actual spawn is attested by "
            "the post-Popen supervisor"
        ),
    )
    start = time.monotonic()
    last_activity = start
    last_sizes = (
        PHASE_PATH.stat().st_size if PHASE_PATH.is_file() else 0,
        0,
    )
    timeout_cleanup = None
    exception_cleanup = None
    residue_cleanup = None
    with KIT_LOG_PATH.open("x", encoding="utf-8") as kit_log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=kit_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        pgid = process.pid
        observed_pgid = None
        observed_pgid_error = None
        try:
            observed_pgid = os.getpgid(process.pid)
        except ProcessLookupError as error:
            observed_pgid_error = f"{type(error).__name__}: {error}"
        if observed_pgid is not None and observed_pgid != pgid:
            cleanup = _bounded_case_group_cleanup(
                process,
                pgid,
                reason="post_popen_start_new_session_pgid_mismatch",
            )
            raise RuntimeError(
                "D400 start_new_session PGID invariant failed: "
                f"expected={pgid}, observed={observed_pgid}, "
                f"cleanup_pass={cleanup['pass']}"
            )
        timed_out = False
        timeout_kind = None
        timeout_last_phase = None
        sigterm_sent = False
        sigkill_sent = False
        kill_wait_expired = False
        try:
            while process.poll() is None:
                now = time.monotonic()
                elapsed = now - start
                phase_size = (
                    PHASE_PATH.stat().st_size
                    if PHASE_PATH.is_file()
                    else 0
                )
                log_size = (
                    KIT_LOG_PATH.stat().st_size
                    if KIT_LOG_PATH.is_file()
                    else 0
                )
                sizes = (phase_size, log_size)
                if sizes != last_sizes:
                    last_sizes = sizes
                    last_activity = now
                inactivity = now - last_activity
                if (
                    elapsed >= TOTAL_WATCHDOG_S
                    or inactivity >= INACTIVITY_WATCHDOG_S
                ):
                    timed_out = True
                    timeout_kind = (
                        "total"
                        if elapsed >= TOTAL_WATCHDOG_S
                        else "inactivity"
                    )
                    try:
                        phase_rows = _phase_rows()
                        timeout_last_phase = (
                            phase_rows[-1].get("phase")
                            if phase_rows
                            else None
                        )
                    except Exception as error:
                        timeout_last_phase = (
                            "phase-read-error:"
                            f"{type(error).__name__}:{error}"
                        )
                    timeout_cleanup = _bounded_case_group_cleanup(
                        process,
                        pgid,
                        reason=f"{timeout_kind}_watchdog",
                    )
                    sigterm_sent = timeout_cleanup["sigterm_sent"]
                    sigkill_sent = timeout_cleanup["sigkill_sent"]
                    kill_wait_expired = not timeout_cleanup["pass"]
                    break
                time.sleep(0.1)
            returncode = process.poll()
            if returncode is None and not kill_wait_expired:
                returncode = process.wait()
        except Exception as error:
            exception_cleanup = _bounded_case_group_cleanup(
                process,
                pgid,
                reason=(
                    "unexpected_supervisor_exception:"
                    f"{type(error).__name__}"
                ),
            )
            kit_log.write(
                "D400_SUPERVISOR_EXCEPTION_CLEANUP "
                + json.dumps(
                    exception_cleanup,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
            kit_log.flush()
            os.fsync(kit_log.fileno())
            raise RuntimeError(
                "D400 supervisor exception after Popen; case process-group "
                f"cleanup pass={exception_cleanup['pass']}"
            ) from error
        finally:
            kit_log.flush()
            os.fsync(kit_log.fileno())
    elapsed_s = time.monotonic() - start
    time.sleep(1.0)
    (
        group_residue_before_cleanup,
        group_residue_before_cleanup_error,
    ) = _safe_process_group_residue(pgid)
    if (
        group_residue_before_cleanup
        or group_residue_before_cleanup_error is not None
    ):
        residue_cleanup = _bounded_case_group_cleanup(
            process,
            pgid,
            reason=(
                "post_exit_process_group_residue_or_query_error"
            ),
        )
        sigterm_sent = (
            sigterm_sent or residue_cleanup["sigterm_sent"]
        )
        sigkill_sent = (
            sigkill_sent or residue_cleanup["sigkill_sent"]
        )
        kill_wait_expired = (
            kill_wait_expired or not residue_cleanup["pass"]
        )
    group_residue, group_residue_error = _safe_process_group_residue(
        pgid
    )
    gpu_rows_after = _gpu_compute_rows()
    worker_gpu_residue = [
        row for row in gpu_rows_after if row["pid"] == process.pid
    ]
    required = {
        "claim": CLAIM_PATH.is_file(),
        "raw_summary": RAW_PATH.is_file(),
        "preclose": PRECLOSE_PATH.is_file(),
        "phase_markers": PHASE_PATH.is_file(),
        "owner_evidence": OWNER_EVIDENCE_PATH.is_file(),
    }
    raw = _read_json(RAW_PATH) if required["raw_summary"] else {}
    preclose = _read_json(PRECLOSE_PATH) if required["preclose"] else {}
    counter_gate = _counter_gate(raw.get("counters"))
    hash_checks = {
        "raw_worker_protocol_pass": raw.get("worker_protocol_pass") is True,
        "launched_runtime_stack_probe_pass": raw.get(
            "runtime_stack_probe", {}
        ).get("pass")
        is True,
        "launched_omni_physx_extension_exact": raw.get(
            "runtime_stack_probe", {}
        )
        .get("supported_runtime_probe", {})
        .get("omni_physx_extension_version")
        == "107.3.26",
        "launched_active_extension_root_exact": raw.get(
            "runtime_stack_probe", {}
        )
        .get("checks", {})
        .get("active_extension_root_exact")
        is True,
        "pinned_active_extension_native_plugin_provenance_hash_exact": raw.get(
            "runtime_stack_probe", {}
        )
        .get("physx_sdk_engine_provenance", {})
        .get("active_extension_native_plugin_sha256")
        == EXPECTED_PHYSX_NATIVE_PLUGIN_SHA256,
        "preclose_worker_protocol_pass": preclose.get("worker_protocol_pass")
        is True,
        "preclose_raw_path_exact": preclose.get("raw_summary_path")
        == _rel(RAW_PATH),
        "preclose_summary_hash_exact": required["raw_summary"]
        and preclose.get("summary_sha256") == _sha(RAW_PATH),
        "preclose_counters_exact": preclose.get("counters")
        == raw.get("counters"),
        "preclose_counter_gate_exact": preclose.get("counter_gate")
        == raw.get("counter_gate"),
        "preclose_stagecache_erase_exact": preclose.get("stagecache_erase")
        == raw.get("stagecache_erase"),
        "preclose_phase_prefix_exact": required["phase_markers"]
        and _prefix_hash_check(preclose),
        "safe_to_close_app": preclose.get("safe_to_close_app") is True,
        "owner_evidence_hash_exact": required["owner_evidence"]
        and raw.get("owner_evidence", {}).get("sha256")
        == _sha(OWNER_EVIDENCE_PATH),
    }
    operational = {
        "returncode_zero": returncode == 0,
        "no_watchdog_timeout": not timed_out,
        "no_sigterm": not sigterm_sent,
        "no_sigkill": not sigkill_sent,
        "kill_wait_not_expired": not kill_wait_expired,
        "process_reaped": process.poll() is not None,
        "process_group_residue_queries_succeeded": (
            group_residue_before_cleanup_error is None
            and group_residue_error is None
        ),
        "no_process_group_residue": not group_residue,
        "no_worker_gpu_pid_residue": not worker_gpu_residue,
        "all_required_artifacts": all(required.values()),
        "counter_vector_exact": counter_gate["pass"],
    }
    kit_log_audit = _audit_kit_log()
    _phase(
        "external_kit_log_audit_end",
        passed=kit_log_audit["pass"],
        error_count=len(kit_log_audit["error_or_fatal_lines"]),
        warning_count=len(kit_log_audit["matching_warning_lines"]),
    )
    binding_gate = _post_worker_binding_gate(
        approval=approval,
        manifest=manifest,
        invocation=invocation,
        environment=environment,
        negative=negative,
        invocation_sha256_at_write=invocation_sha256_at_write,
    )
    phase_rows_before_supervisor = _phase_rows()
    projected_supervisor_row = {
        "ordinal": len(phase_rows_before_supervisor) + 1,
        "phase": "external_supervisor_end",
        "owner": "controller",
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
    }
    pre_rerun_phase_audit = _pre_rerun_phase_audit(
        [*phase_rows_before_supervisor, projected_supervisor_row]
    )
    technical_pass = bool(
        all(operational.values())
        and all(hash_checks.values())
        and kit_log_audit["pass"]
        and binding_gate["pass"]
        and pre_rerun_phase_audit["pass"]
    )
    supervisor = {
        "artifact": "D400_HASH_BOUND_PROCESS_GROUP_SUPERVISOR_V1",
        "pid": process.pid,
        "pgid": pgid,
        "observed_pgid_after_popen": observed_pgid,
        "observed_pgid_probe_error": observed_pgid_error,
        "actual_worker_invocations": 1,
        "automatic_retries": 0,
        "returncode": returncode,
        "elapsed_s": elapsed_s,
        "watchdogs": {
            "total_s": TOTAL_WATCHDOG_S,
            "inactivity_s": INACTIVITY_WATCHDOG_S,
            "activity_definition": invocation["activity_definition"],
            "timed_out": timed_out,
            "timeout_kind": timeout_kind,
            "timeout_last_phase": timeout_last_phase,
            "sigterm_sent": sigterm_sent,
            "sigkill_sent": sigkill_sent,
            "kill_wait_expired": kill_wait_expired,
            "timeout_cleanup": timeout_cleanup,
            "exception_cleanup": exception_cleanup,
            "post_exit_residue_cleanup": residue_cleanup,
        },
        "group_residue_before_cleanup": group_residue_before_cleanup,
        "group_residue_before_cleanup_error": (
            group_residue_before_cleanup_error
        ),
        "group_residue": group_residue,
        "group_residue_error": group_residue_error,
        "gpu_rows_after": gpu_rows_after,
        "worker_gpu_residue": worker_gpu_residue,
        "required_artifacts": required,
        "counter_gate": counter_gate,
        "hash_checks": hash_checks,
        "operational_checks": operational,
        "kit_log_audit": kit_log_audit,
        "post_worker_runtime_authority_binding_gate": binding_gate,
        "pre_rerun_phase_audit_with_projected_supervisor_marker": (
            pre_rerun_phase_audit
        ),
        "unrelated_processes_signaled": 0,
        "technical_pass": technical_pass,
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase(
        "external_supervisor_end",
        passed=technical_pass,
        returncode=returncode,
        timed_out=timed_out,
        signaled=sigterm_sent or sigkill_sent,
    )
    actual_pre_rerun_phase_audit = _pre_rerun_phase_audit(_phase_rows())
    if (
        actual_pre_rerun_phase_audit["checks"]
        != pre_rerun_phase_audit["checks"]
    ):
        raise RuntimeError(
            "D400 actual pre-Rerun phase audit differs from projection"
        )
    return supervisor


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        size = list(image.size)
        mode = image.mode
    return {
        "path": _rel(path),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        "size": size,
        "mode": mode,
        "exact_1920x1080": size == [1920, 1080],
    }


def _write_decision_board(
    evidence: dict[str, Any], raw: dict[str, Any]
) -> dict[str, Any]:
    if BOARD_PATH.exists():
        raise RuntimeError(f"D400 decision board already exists: {_rel(BOARD_PATH)}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def draw(ax: Any, rows: list[dict[str, Any]], color: str, title: str) -> None:
        all_vertices = []
        for row in rows:
            vertices = np.asarray(row["vertices_m"], dtype=np.float64)
            triangles = np.asarray(row["triangles"], dtype=np.int64)
            all_vertices.append(vertices)
            faces = vertices[triangles]
            collection = Poly3DCollection(
                faces,
                facecolors=color,
                edgecolors="#1f2937",
                linewidths=0.08,
                alpha=0.42,
            )
            ax.add_collection3d(collection)
        stacked = np.vstack(all_vertices)
        lower = stacked.min(axis=0)
        upper = stacked.max(axis=0)
        center = 0.5 * (lower + upper)
        radius = max(float(np.max(upper - lower)) * 0.58, 0.01)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.view_init(elev=24, azim=-58)

    source = evidence["source_gripper_mesh"]
    live = evidence["live_sdf_input_mesh"]
    link5 = evidence["link5_a64"]
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="white")
    draw(
        fig.add_subplot(1, 3, 1, projection="3d"),
        [source],
        "#3b82f6",
        "Frozen source gripper Mesh",
    )
    draw(
        fig.add_subplot(1, 3, 2, projection="3d"),
        [live],
        "#f59e0b",
        "Live SDF input Mesh (not cooked surface)",
    )
    draw(
        fig.add_subplot(1, 3, 3, projection="3d"),
        link5,
        "#10b981",
        "Frozen link5 A64 (64 parts)",
    )
    cooking = raw["cooking"]
    footer = (
        "D400 zero-step configuration/load preflight | "
        f"cook scheduled/finished delta={cooking['delta']['scheduled']}/"
        f"{cooking['delta']['finished']} | property rows link5/gripper="
        f"{len(raw['property_queries']['link5']['collider_callbacks'])}/"
        f"{len(raw['property_queries']['gripper_link']['collider_callbacks'])} | "
        "physics step=0, q5=0, contact=0, cylinder=0"
    )
    fig.suptitle(
        "D400 gripper_link A64 -> SDF res256: registered inspection subjects",
        fontsize=20,
        y=0.975,
    )
    fig.text(0.5, 0.025, footer, ha="center", va="center", fontsize=10)
    fig.tight_layout(rect=(0.01, 0.06, 0.99, 0.94))
    fig.savefig(BOARD_PATH, dpi=120, facecolor="white")
    plt.close(fig)
    info = _png_info(BOARD_PATH)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"D400 decision board size mismatch: {info}")
    return info


def _rerun_component_contract() -> dict[str, list[str]]:
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
        "Transform3D:child_frame",
        "Transform3D:parent_frame",
        "Transform3D:quaternion",
        "Transform3D:translation",
    ]
    result = {
        "/d400/phase/source_baseline": ["TextDocument:text"],
        "/d400/phase/live_configuration": ["TextDocument:text"],
        "/d400/phase/post_query_decision": ["TextDocument:text"],
        "/d400/inspection/source_gripper_mesh": mesh_components,
        "/d400/inspection/live_sdf_input_mesh": mesh_components,
        "/d400/status/api_token_attributes": ["TextDocument:text"],
        "/d400/status/inventory_owner_query": ["TextDocument:text"],
        "/d400/status/cook_queue": [
            "Scalars:scalars",
            "TextDocument:text",
        ],
        "/d400/status/mass_counters_instance_state": ["TextDocument:text"],
    }
    for path in LINK5_RERUN_PATHS:
        result[path] = mesh_components
    return result


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    spatial = rrb.Spatial3DView(
        origin="/",
        contents="/d400/inspection/**",
        name="D400 source, live SDF input, and frozen link5 A64",
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=(0.28, -0.34, 0.22),
            look_target=(0.03, 0.0, -0.01),
            eye_up=(0.0, 0.0, 1.0),
        ),
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=True,
            show_bounding_box=False,
        ),
    )
    status = rrb.TextDocumentView(
        origin="/d400/status",
        contents="/d400/status/**",
        name="API, owner, cook queue, and invariance gates",
    )
    phases = rrb.TextDocumentView(
        origin="/d400/phase",
        contents="/d400/phase/**",
        name="Registered preflight phases",
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            spatial,
            rrb.Vertical(status, phases, row_shares=[0.75, 0.25]),
            column_shares=[0.72, 0.28],
        ),
        rrb.TimePanel(state=rrb.PanelState.Hidden),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _is_rerun_system_entity(path: str) -> bool:
    return bool(
        path == "/__properties"
        or path.startswith("/__properties/")
        or path
        in {
            "/viewport",
            "/blueprint_panel",
            "/selection_panel",
            "/time_panel",
        }
        or path.startswith("/container/")
        or path.startswith("/view/")
    )


def _actual_rerun_row_contract(
    intent_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from rerun.experimental import RrdReader

    expected_phase_by_path = {
        path: phase
        for phase, paths in RERUN_PHASE_ASSIGNMENTS.items()
        for path in paths
    }
    reader = RrdReader(RRD_PATH)
    recordings = reader.recordings()
    recording_rows = [
        {
            "kind": row.kind,
            "application_id": row.application_id,
            "recording_id": row.recording_id,
        }
        for row in recordings
    ]
    parse_errors: list[dict[str, Any]] = []
    unexpected_non_system: set[str] = set()
    observed_target_paths: set[str] = set()
    timeless_target_rows: set[tuple[str, str]] = set()
    logical_rows: dict[tuple[str, str], set[int]] = {}
    row_id_owners: dict[str, set[str]] = {}
    chunk_rows: list[dict[str, Any]] = []
    footer_store_opened = False
    if len(recordings) == 1:
        store = reader.store(store=recordings[0])
        footer_store_opened = True
        for chunk_index, chunk in enumerate(store.stream()):
            path = f"/{str(chunk.entity_path).strip('/')}"
            if _is_rerun_system_entity(path):
                continue
            if path not in expected_phase_by_path:
                unexpected_non_system.add(path)
                continue
            observed_target_paths.add(path)
            batch = chunk.to_record_batch()
            fields = list(batch.schema)
            row_id_fields = [
                (index, field)
                for index, field in enumerate(fields)
                if field.name == "rerun.controls.RowId"
                and (field.metadata or {}).get(b"rerun:kind")
                == b"control"
                and (field.metadata or {}).get(
                    b"ARROW:extension:name"
                )
                == b"rerun.datatypes.TUID"
                and (field.metadata or {}).get(
                    b"ARROW:extension:metadata"
                )
                == b'{"namespace":"row"}'
            ]
            phase_fields = [
                (index, field)
                for index, field in enumerate(fields)
                if (field.metadata or {}).get(b"rerun:kind") == b"index"
                and (field.metadata or {}).get(b"rerun:index_name")
                == b"preflight_phase"
            ]
            chunk_record = {
                "chunk_index": chunk_index,
                "entity_path": path,
                "num_rows": int(chunk.num_rows),
                "record_batch_num_rows": int(batch.num_rows),
                "is_static": bool(chunk.is_static),
                "timeline_names": list(chunk.timeline_names),
                "row_id_field_count": len(row_id_fields),
                "phase_field_count": len(phase_fields),
                "phase_field_type": (
                    str(phase_fields[0][1].type)
                    if len(phase_fields) == 1
                    else None
                ),
            }
            chunk_rows.append(chunk_record)
            if batch.num_rows != chunk.num_rows:
                parse_errors.append(
                    {
                        "chunk_index": chunk_index,
                        "entity_path": path,
                        "error": "record_batch_row_count_mismatch",
                    }
                )
                continue
            if len(row_id_fields) != 1:
                parse_errors.append(
                    {
                        "chunk_index": chunk_index,
                        "entity_path": path,
                        "error": "row_id_field_contract_mismatch",
                        "count": len(row_id_fields),
                    }
                )
                continue
            row_id_array = batch.column(row_id_fields[0][0])
            if row_id_array.null_count != 0:
                parse_errors.append(
                    {
                        "chunk_index": chunk_index,
                        "entity_path": path,
                        "error": "row_id_null",
                    }
                )
                continue
            row_ids = [
                bytes(value).hex()
                for value in row_id_array.to_pylist()
            ]
            if any(len(row_id) != 32 for row_id in row_ids):
                parse_errors.append(
                    {
                        "chunk_index": chunk_index,
                        "entity_path": path,
                        "error": "row_id_not_exactly_16_bytes",
                    }
                )
                continue
            if len(set(row_ids)) != len(row_ids):
                parse_errors.append(
                    {
                        "chunk_index": chunk_index,
                        "entity_path": path,
                        "error": "duplicate_row_id_inside_chunk",
                    }
                )
                continue
            if chunk.is_static:
                if chunk.timeline_names or phase_fields:
                    parse_errors.append(
                        {
                            "chunk_index": chunk_index,
                            "entity_path": path,
                            "error": "contradictory_static_chunk",
                        }
                    )
                for row_id in row_ids:
                    timeless_target_rows.add((path, row_id))
                continue
            if (
                "preflight_phase" not in chunk.timeline_names
                or len(phase_fields) != 1
                or str(phase_fields[0][1].type) != "int64"
            ):
                parse_errors.append(
                    {
                        "chunk_index": chunk_index,
                        "entity_path": path,
                        "error": "phase_index_field_contract_mismatch",
                    }
                )
                continue
            phase_array = batch.column(phase_fields[0][0])
            if (
                phase_array.null_count != 0
                or len(phase_array) != len(row_ids)
            ):
                parse_errors.append(
                    {
                        "chunk_index": chunk_index,
                        "entity_path": path,
                        "error": "phase_array_dense_length_mismatch",
                    }
                )
                continue
            for row_id, phase in zip(
                row_ids, phase_array.to_pylist(), strict=True
            ):
                logical_rows.setdefault((path, row_id), set()).add(
                    int(phase)
                )
                row_id_owners.setdefault(row_id, set()).add(path)
    observed_rows = [
        {
            "entity_path": path,
            "row_id_hex": row_id,
            "preflight_phase_values": sorted(phases),
        }
        for (path, row_id), phases in sorted(logical_rows.items())
    ]
    rows_by_path = {
        path: [
            row
            for row in observed_rows
            if row["entity_path"] == path
        ]
        for path in expected_phase_by_path
    }
    checks = {
        "rerun_sdk_version_exact": _package_version("rerun-sdk")
        == RERUN_VERSION,
        "exactly_one_recording_store": len(recordings) == 1,
        "recording_application_id_exact": len(recordings) == 1
        and recordings[0].application_id
        == "roarm_g0a_d400_sdf_preflight",
        "recording_id_exact": len(recordings) == 1
        and recordings[0].recording_id == "g0a_d400_sdf_preflight",
        "footer_manifest_store_opened": footer_store_opened,
        "no_chunk_parse_error": not parse_errors,
        "target_path_set_exact": observed_target_paths
        == set(expected_phase_by_path),
        "no_unexpected_non_system_entity": not unexpected_non_system,
        "zero_timeless_target_rows": not timeless_target_rows,
        "exactly_one_unique_logical_row_per_target": all(
            len(rows) == 1 for rows in rows_by_path.values()
        ),
        "each_target_phase_exact": all(
            len(rows_by_path[path]) == 1
            and rows_by_path[path][0]["preflight_phase_values"]
            == [expected_phase]
            for path, expected_phase in expected_phase_by_path.items()
        ),
        "total_unique_target_rows_exact_73": len(observed_rows) == 73,
        "row_id_not_shared_across_entities": all(
            len(owners) == 1 for owners in row_id_owners.values()
        ),
        "intent_assignment_matches_registered_contract": intent_rows
        == [
            {"path": path, "phase": phase}
            for phase in (0, 1, 2)
            for path in RERUN_PHASE_ASSIGNMENTS[phase]
        ],
    }
    return {
        "artifact": "D400_ACTUAL_RRD_ROW_TIMELINE_READBACK_V1",
        "reader": (
            "rerun.experimental.RrdReader explicit recording store; "
            "footer manifest required"
        ),
        "recording_stores": recording_rows,
        "expected_phase_by_path": expected_phase_by_path,
        "intent_rows_diagnostic_only": intent_rows,
        "chunks": chunk_rows,
        "parse_errors": parse_errors,
        "unexpected_non_system_entities": sorted(
            unexpected_non_system
        ),
        "timeless_target_rows": [
            {"entity_path": path, "row_id_hex": row_id}
            for path, row_id in sorted(timeless_target_rows)
        ],
        "observed_logical_rows": observed_rows,
        "observed_logical_rows_sha256": _json_sha256(observed_rows),
        "row_id_owner_sets": {
            row_id: sorted(owners)
            for row_id, owners in sorted(row_id_owners.items())
        },
        "phase_counts": {
            str(phase): sum(
                row["preflight_phase_values"] == [phase]
                for row in observed_rows
            )
            for phase in (0, 1, 2)
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _write_rerun(
    evidence: dict[str, Any], raw: dict[str, Any]
) -> dict[str, Any]:
    preexisting = [
        _rel(path)
        for path in (
            RRD_PATH,
            RBL_PATH,
            RERUN_VALIDATION_PATH,
            RERUN_SCREENSHOT_PATH,
            RERUN_RECEIPT_PATH,
            MANUAL_INSPECTION_PATH,
        )
        if path.exists()
    ]
    if preexisting:
        raise RuntimeError(
            f"D400 Rerun outputs already exist; overwrite refused: {preexisting}"
        )
    import numpy as np
    import rerun as rr
    from roarm_rl.rerun_contract import validate_rerun_artifact

    if str(rr.__version__) != RERUN_VERSION:
        raise RuntimeError(f"D400 Rerun SDK version drift: {rr.__version__}")
    if len(RERUN_ENTITY_PATHS) != 73:
        raise RuntimeError("D400 Rerun entity count literal drift")
    if _path_set_sha(RERUN_ENTITY_PATHS) != EXPECTED_RERUN_ENTITY_SHA256:
        raise RuntimeError("D400 Rerun entity path-set hash drift")
    if _path_set_sha(LINK5_RERUN_PATHS) != EXPECTED_LINK5_RERUN_SHA256:
        raise RuntimeError("D400 link5 Rerun family path-set hash drift")
    blueprint = _build_blueprint()
    logged_rows = []

    def set_phase(recording: Any, phase: int) -> None:
        recording.reset_time()
        recording.set_time("preflight_phase", sequence=phase)

    def log_document(
        recording: Any, path: str, phase: int, payload: Any
    ) -> None:
        set_phase(recording, phase)
        recording.log(
            path.lstrip("/"),
            rr.TextDocument(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
            ),
        )
        logged_rows.append({"path": path, "phase": phase})

    def log_mesh(
        recording: Any,
        path: str,
        phase: int,
        row: dict[str, Any],
        color: list[int],
        translation: list[float],
        frame: str,
    ) -> None:
        set_phase(recording, phase)
        recording.log(
            path.lstrip("/"),
            rr.Mesh3D(
                vertex_positions=np.asarray(row["vertices_m"], dtype=np.float32),
                triangle_indices=np.asarray(row["triangles"], dtype=np.uint32),
                albedo_factor=color,
            ),
            rr.CoordinateFrame(frame),
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
                parent_frame="tf#/",
                child_frame=frame,
            ),
        )
        logged_rows.append({"path": path, "phase": phase})

    app_id = "roarm_g0a_d400_sdf_preflight"
    with rr.RecordingStream(
        app_id,
        recording_id="g0a_d400_sdf_preflight",
        make_default=False,
        send_properties=True,
    ) as recording:
        recording.save(str(RRD_PATH), write_footer=True)
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        log_document(
            recording,
            "/d400/phase/source_baseline",
            0,
            {
                "phase": 0,
                "meaning": "source_baseline",
                "authority": "frozen D344 source mesh",
            },
        )
        log_mesh(
            recording,
            "/d400/inspection/source_gripper_mesh",
            0,
            evidence["source_gripper_mesh"],
            [59, 130, 246, 180],
            [-0.1, 0.0, 0.0],
            "d400/source_gripper",
        )
        log_document(
            recording,
            "/d400/phase/live_configuration",
            1,
            {
                "phase": 1,
                "meaning": "live_configuration",
                "guard": "live USD SDF input Mesh, not the cooked SDF surface",
            },
        )
        log_mesh(
            recording,
            "/d400/inspection/live_sdf_input_mesh",
            1,
            evidence["live_sdf_input_mesh"],
            [245, 158, 11, 180],
            [0.0, 0.0, 0.0],
            "d400/live_gripper_sdf_input",
        )
        log_document(
            recording,
            "/d400/status/api_token_attributes",
            1,
            raw["live_inventory"]["sdf_readback"],
        )
        log_document(
            recording,
            "/d400/status/inventory_owner_query",
            1,
            {
                "inventory": evidence["active_inventory"],
                "query_counts": {
                    body: len(query["collider_callbacks"])
                    for body, query in raw["property_queries"].items()
                },
                "query_path_hashes": {
                    body: query["sorted_path_set_sha256"]
                    for body, query in raw["property_queries"].items()
                },
            },
        )
        for index, row in enumerate(evidence["link5_a64"]):
            log_mesh(
                recording,
                LINK5_RERUN_PATHS[index],
                1,
                row,
                [16, 185, 129, 150],
                [0.1, 0.0, 0.0],
                f"d400/link5_a64_{index:03d}",
            )
        log_document(
            recording,
            "/d400/phase/post_query_decision",
            2,
            {
                "phase": 2,
                "meaning": "post_query_decision",
                "worker_protocol_pass": raw["worker_protocol_pass"],
                "scientific_or_physics_verdict": None,
                "g0a_pass": False,
            },
        )
        set_phase(recording, 2)
        recording.log(
            "d400/status/cook_queue",
            rr.Scalars([float(raw["cooking"]["delta"]["scheduled"])]),
            rr.TextDocument(
                json.dumps(raw["cooking"], indent=2, sort_keys=True)
            ),
        )
        logged_rows.append({"path": "/d400/status/cook_queue", "phase": 2})
        log_document(
            recording,
            "/d400/status/mass_counters_instance_state",
            2,
            {
                "authored_mass": raw["derivative_asset"]["authored_mass_gate"],
                "live_mass": raw["live_mass_com_inertia_gate"],
                "counters": raw["counters"],
                "instance_state": raw["live_inventory"]["structure"],
            },
        )
        recording.flush(timeout_sec=30.0)
    blueprint.save(app_id, RBL_PATH)
    row_contract = _actual_rerun_row_contract(logged_rows)
    if not row_contract["pass"]:
        _write_json_x(
            RERUN_VALIDATION_PATH,
            {
                "artifact": "D400_RERUN_ARTIFACT_VALIDATION_FAIL_STOP_V1",
                "headless_viewer_invocations": 0,
                "headless_viewer_skipped_reason": (
                    "actual finalized RRD row/timeline readback failed"
                ),
                "d400_exact_row_assignment_contract": row_contract,
                "pass": False,
            },
        )
        raise RuntimeError(
            "D400 actual finalized RRD row/timeline contract failed"
        )
    old_scale = {
        key: os.environ.get(key)
        for key in (
            "GDK_SCALE",
            "GDK_DPI_SCALE",
            "QT_SCALE_FACTOR",
            "WINIT_X11_SCALE_FACTOR",
        )
    }
    os.environ["GDK_SCALE"] = "1"
    os.environ["GDK_DPI_SCALE"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["WINIT_X11_SCALE_FACTOR"] = "1"
    try:
        validation = validate_rerun_artifact(
            RRD_PATH,
            expected_entity_paths=RERUN_ENTITY_PATHS,
            exact_entity_paths=RERUN_ENTITY_PATHS,
            expected_timeline_names=["preflight_phase"],
            exact_timeline_names=["blueprint", "log_time", "preflight_phase"],
            expected_entity_components=_rerun_component_contract(),
            blueprint_path=RBL_PATH,
            screenshot_path=RERUN_SCREENSHOT_PATH,
            screenshot_window_size="1920x1080",
            screenshot_port="auto",
            cli_path=RERUN_CLI,
            expected_version=RERUN_VERSION,
            timeout_s=240.0,
        )
    finally:
        for key, value in old_scale.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    render_command = validation.get("headless_render", {}).get("command")
    actual_headless_viewer_invocations = int(
        isinstance(render_command, list)
        and len(render_command) >= 2
        and str(render_command[0]) == str(RERUN_CLI)
        and render_command[1] == "--headless"
        and str(RRD_PATH) in render_command
    )
    validation["d400_exact_row_assignment_contract"] = row_contract
    validation["headless_viewer_invocations"] = (
        actual_headless_viewer_invocations
    )
    validation["pass"] = bool(
        validation.get("pass") is True
        and row_contract["pass"]
        and actual_headless_viewer_invocations == 1
    )
    _write_json_x(RERUN_VALIDATION_PATH, validation)
    screenshot = (
        _png_info(RERUN_SCREENSHOT_PATH)
        if RERUN_SCREENSHOT_PATH.is_file()
        else {"path": _rel(RERUN_SCREENSHOT_PATH), "exists": False}
    )
    receipt = {
        "artifact": "D400_RERUN_RENDER_RECEIPT_V1",
        "rrd": {
            "path": _rel(RRD_PATH),
            "sha256": _sha(RRD_PATH),
            "bytes": RRD_PATH.stat().st_size,
        },
        "rbl": {
            "path": _rel(RBL_PATH),
            "sha256": _sha(RBL_PATH),
            "bytes": RBL_PATH.stat().st_size,
        },
        "validation": {
            "path": _rel(RERUN_VALIDATION_PATH),
            "sha256": _sha(RERUN_VALIDATION_PATH),
            "pass": validation["pass"],
        },
        "screenshot": screenshot,
        "headless_viewer_invocations": actual_headless_viewer_invocations,
        "manual_inspection_pending": True,
        "pass_before_manual": bool(
            validation["pass"]
            and screenshot.get("exact_1920x1080") is True
        ),
    }
    _write_json_x(RERUN_RECEIPT_PATH, receipt)
    return receipt


def _wait_for_manual_inspection() -> dict[str, Any]:
    print(
        json.dumps(
            {
                "D400_MANUAL_INSPECTION_REQUIRED": True,
                "screenshot": _rel(RERUN_SCREENSHOT_PATH),
                "screenshot_sha256": _sha(RERUN_SCREENSHOT_PATH),
                "write_once_path": _rel(MANUAL_INSPECTION_PATH),
                "required_fields": {
                    "artifact": "D400_MANUAL_ORIGINAL_RESOLUTION_INSPECTION_V1",
                    "inspection_completed": True,
                    "screenshot_path": _rel(RERUN_SCREENSHOT_PATH),
                    "screenshot_sha256": _sha(RERUN_SCREENSHOT_PATH),
                    "original_resolution": [1920, 1080],
                    "subjects_visible": {
                        "source_gripper_mesh": True,
                        "live_sdf_input_mesh": True,
                        "link5_a64": True,
                        "api_token_attributes": True,
                        "cook_queue_and_owner_status": True,
                    },
                    "text_overlap_or_clipping_observed": False,
                    "observations": ["non-empty human/agent observation"],
                },
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        flush=True,
    )
    start = time.monotonic()
    last_notice = start
    while time.monotonic() - start < MANUAL_INSPECTION_WAIT_S:
        if MANUAL_INSPECTION_PATH.is_file():
            value = _read_json(MANUAL_INSPECTION_PATH)
            subjects = value.get("subjects_visible", {})
            checks = {
                "artifact_exact": value.get("artifact")
                == "D400_MANUAL_ORIGINAL_RESOLUTION_INSPECTION_V1",
                "inspection_completed_true": value.get("inspection_completed")
                is True,
                "screenshot_path_exact": value.get("screenshot_path")
                == _rel(RERUN_SCREENSHOT_PATH),
                "screenshot_sha_exact": value.get("screenshot_sha256")
                == _sha(RERUN_SCREENSHOT_PATH),
                "original_resolution_exact": value.get("original_resolution")
                == [1920, 1080],
                "all_subjects_visible": subjects
                == {
                    "source_gripper_mesh": True,
                    "live_sdf_input_mesh": True,
                    "link5_a64": True,
                    "api_token_attributes": True,
                    "cook_queue_and_owner_status": True,
                },
                "no_text_overlap_or_clipping": value.get(
                    "text_overlap_or_clipping_observed"
                )
                is False,
                "nonempty_observations": isinstance(
                    value.get("observations"), list
                )
                and bool(value["observations"])
                and all(
                    isinstance(item, str) and item.strip()
                    for item in value["observations"]
                ),
            }
            return {
                "path": _rel(MANUAL_INSPECTION_PATH),
                "sha256": _sha(MANUAL_INSPECTION_PATH),
                "value": value,
                "checks": checks,
                "pass": all(checks.values()),
            }
        now = time.monotonic()
        if now - last_notice >= 5.0:
            print(
                json.dumps(
                    {
                        "D400_MANUAL_INSPECTION_WAIT": round(now - start, 3),
                        "remaining_s": round(
                            MANUAL_INSPECTION_WAIT_S - (now - start), 3
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_notice = now
        time.sleep(0.25)
    return {
        "path": _rel(MANUAL_INSPECTION_PATH),
        "timeout_s": MANUAL_INSPECTION_WAIT_S,
        "checks": {"file_arrived_within_bound": False},
        "pass": False,
    }


def _write_completion(
    *,
    supervisor: dict[str, Any],
    board: dict[str, Any] | None,
    rerun: dict[str, Any] | None,
    manual: dict[str, Any] | None,
    observability_error: dict[str, str] | None = None,
) -> dict[str, Any]:
    if COMPLETION_PATH.exists():
        raise RuntimeError("D400 completion summary already exists")
    technical = supervisor.get("technical_pass") is True
    observability = bool(
        technical
        and board
        and board.get("exact_1920x1080") is True
        and rerun
        and rerun.get("pass_before_manual") is True
        and manual
        and manual.get("pass") is True
    )
    verdict = (
        RESULT_PASS
        if technical and observability
        else RESULT_OBSERVABILITY_FAIL
        if technical
        else RESULT_TECHNICAL_FAIL
    )
    current_phase_rows = _phase_rows()
    projected_phase_rows = [
        *current_phase_rows,
        {
            "ordinal": len(current_phase_rows) + 1,
            "phase": "completion_summary_end",
            "owner": "controller",
            "pid": os.getpid(),
            "monotonic_ns": time.monotonic_ns(),
        },
    ]
    phase_audit = _phase_audit(
        projected_phase_rows,
        technical_pass_branch=technical,
        observability_pass=observability,
    )
    summary = {
        "artifact": "D400_COMPLETION_SUMMARY_V1",
        "verdict": verdict,
        "technical_pass": technical,
        "observability_pass": observability,
        "runtime_preflight_pass": (
            technical and observability and phase_audit["pass"]
        ),
        "scientific_or_physics_verdict": None,
        "g0a_pass": False,
        "supervisor": {
            "path": _rel(SUPERVISOR_PATH),
            "sha256": _sha(SUPERVISOR_PATH),
        },
        "board": board,
        "rerun": rerun,
        "manual_inspection": manual,
        "observability_error": observability_error,
        "phase_audit_with_projected_completion_marker": phase_audit,
        "scope_counters": (
            _read_json(RAW_PATH).get("counters") if RAW_PATH.is_file() else None
        ),
        "authority_limit": (
            "Configuration, stage-load admission, global cook-queue drain, and "
            "rigid-owner/property enumeration only. No per-prim SDF internal "
            "identity, collision participation, contact, tipping, or grasp claim."
        ),
        "pass": technical and observability and phase_audit["pass"],
    }
    if not phase_audit["pass"]:
        summary["verdict"] = RESULT_OBSERVABILITY_FAIL if technical else RESULT_TECHNICAL_FAIL
    _write_json_x(COMPLETION_PATH, summary)
    _phase(
        "completion_summary_end",
        passed=summary["pass"],
        verdict=summary["verdict"],
        sha256=_sha(COMPLETION_PATH),
    )
    actual_phase_audit = _phase_audit(
        _phase_rows(),
        technical_pass_branch=technical,
        observability_pass=observability,
    )
    if actual_phase_audit["checks"] != phase_audit["checks"]:
        raise RuntimeError(
            "D400 emitted phase audit differs from the write-once projected audit"
        )
    return summary


def _write_controller_fail_stop(
    *,
    failed_stage: str,
    error: Exception,
) -> dict[str, Any]:
    names = [row.get("phase") for row in _phase_rows()]
    if "external_kit_log_audit_end" not in names:
        _phase(
            "external_kit_log_audit_end",
            passed=False,
            audit_skipped=not KIT_LOG_PATH.is_file(),
            failure_stage=failed_stage,
        )
    failure = {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    supervisor = {
        "artifact": "D400_CONTROLLER_FAIL_STOP_SUPERVISOR_V1",
        "failed_stage": failed_stage,
        "failure": failure,
        "worker_spawn_request_recorded": INVOCATION_PATH.is_file(),
        "worker_claim_observed": CLAIM_PATH.is_file(),
        "actual_worker_invocations": (
            1 if CLAIM_PATH.is_file() else None
        ),
        "actual_worker_invocation_authority": (
            "worker claim observed"
            if CLAIM_PATH.is_file()
            else "unknown at controller exception boundary; request is not spawn proof"
        ),
        "invocation": (
            {
                "path": _rel(INVOCATION_PATH),
                "sha256": _sha(INVOCATION_PATH),
            }
            if INVOCATION_PATH.is_file()
            else None
        ),
        "kit_log": (
            {
                "path": _rel(KIT_LOG_PATH),
                "sha256": _sha(KIT_LOG_PATH),
            }
            if KIT_LOG_PATH.is_file()
            else None
        ),
        "unrelated_processes_signaled": 0,
        "technical_pass": False,
    }
    if SUPERVISOR_PATH.exists():
        raise RuntimeError(
            "D400 fail-stop supervisor path was already claimed"
        ) from error
    _write_json_x(SUPERVISOR_PATH, supervisor)
    if "external_supervisor_end" not in [
        row.get("phase") for row in _phase_rows()
    ]:
        _phase(
            "external_supervisor_end",
            passed=False,
            failed_stage=failed_stage,
        )
    completion = _write_completion(
        supervisor=supervisor,
        board=None,
        rerun=None,
        manual=None,
        observability_error=failure,
    )
    return completion


def _assert_fresh_runtime_paths() -> None:
    runtime_paths = (
        RUNTIME_MANIFEST_PATH,
        PHASE_PATH,
        INVOCATION_PATH,
        CLAIM_PATH,
        KIT_LOG_PATH,
        RAW_PATH,
        PRECLOSE_PATH,
        OWNER_EVIDENCE_PATH,
        SUPERVISOR_PATH,
        COMPLETION_PATH,
        COLLISION_ASSET_ROOT,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION_PATH,
        BOARD_PATH,
        RERUN_SCREENSHOT_PATH,
        RERUN_RECEIPT_PATH,
        MANUAL_INSPECTION_PATH,
    )
    claimed = [_rel(path) for path in runtime_paths if path.exists()]
    if claimed:
        raise RuntimeError(
            "D400 forward-only runtime paths already claimed; retry refused: "
            f"{claimed}"
        )


def run_runtime(approved_tuple_sha256: str) -> int:
    approval = _validate_approval_tuple(approved_tuple_sha256)
    _assert_fresh_runtime_paths()
    prereg = _read_json(PREREG_PATH)
    launch_authority = {
        "controller_pid": os.getpid(),
        "approved_tuple_sha256": approved_tuple_sha256,
        "one_shot_nonce": secrets.token_hex(32),
    }
    command = _worker_command(launch_authority)
    _phase(
        "supervisor_preflight_start",
        approved_tuple_sha256=approved_tuple_sha256,
    )
    failed_stage = "runtime_freeze_manifest"
    try:
        manifest = _runtime_freeze_manifest(
            approval,
            prereg,
            launch_authority,
            command,
        )
        _phase(
            "runtime_freeze_manifest_gate_end",
            passed=manifest["pass"],
            sha256=_sha(RUNTIME_MANIFEST_PATH),
        )
        if not manifest["pass"]:
            raise RuntimeError(
                f"D400 runtime freeze manifest failed: {manifest['checks']}"
            )
        failed_stage = "package_gpu_and_existing_process_gate"
        environment = _environment_gate(prereg)
        _phase(
            "package_gpu_and_existing_process_gate_end",
            passed=environment["pass"],
            free_vram_mib=(
                environment["gpu"]["gpus"][0]["memory_free_mib"]
                if environment["gpu"]["gpus"]
                else None
            ),
            d400_process_conflicts=len(environment["d400_conflicts"]),
        )
        if not environment["pass"]:
            raise RuntimeError(
                "D400 package/GPU/process gate failed: "
                f"{environment['checks']}"
            )
        failed_stage = "offline_negative_controls"
        negative = _offline_negative_controls()
        _phase(
            "offline_negative_controls_end",
            passed=negative["pass"],
            passed_count=negative["passed"],
            total=negative["total"],
        )
        if not negative["pass"]:
            raise RuntimeError(
                "D400 runtime offline negative controls failed"
            )
        failed_stage = "single_worker_supervision"
        supervisor = _supervise_worker(
            environment,
            approval,
            manifest,
            negative,
            launch_authority,
            command,
        )
    except Exception as error:
        completion = _write_controller_fail_stop(
            failed_stage=failed_stage,
            error=error,
        )
        print(
            json.dumps(completion, indent=2, sort_keys=True),
            flush=True,
        )
        return 1
    if not supervisor["technical_pass"]:
        completion = _write_completion(
            supervisor=supervisor,
            board=None,
            rerun=None,
            manual=None,
        )
        print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
        return 1
    board = None
    rerun = None
    manual = None
    observability_error = None
    try:
        _phase("technical_pass_branch_start")
        raw = _read_json(RAW_PATH)
        evidence = _read_json(OWNER_EVIDENCE_PATH)
        board = _write_decision_board(evidence, raw)
        rerun = _write_rerun(evidence, raw)
        _phase(
            "rerun_save_only_finalize_end_if_technical_pass",
            rrd_sha256=_sha(RRD_PATH),
            rbl_sha256=_sha(RBL_PATH),
        )
        manual = _wait_for_manual_inspection()
        _phase(
            "rerun_verify_and_visual_inspection_end_if_technical_pass",
            validation_pass=rerun["pass_before_manual"],
            manual_pass=manual["pass"],
            screenshot_sha256=(
                _sha(RERUN_SCREENSHOT_PATH)
                if RERUN_SCREENSHOT_PATH.is_file()
                else None
            ),
        )
    except Exception as error:
        observability_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    completion = _write_completion(
        supervisor=supervisor,
        board=board,
        rerun=rerun,
        manual=manual,
        observability_error=observability_error,
    )
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return 0 if completion["pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved-tuple-sha256",
        required=True,
        help=(
            "Exact SHA-256 of d400_proposed_runtime_hash_tuple.json explicitly "
            "cited by the user in a separate runtime approval."
        ),
    )
    args = parser.parse_args()
    try:
        return run_runtime(args.approved_tuple_sha256)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
