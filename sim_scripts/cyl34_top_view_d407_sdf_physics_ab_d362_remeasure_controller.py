#!/usr/bin/env python3
"""D407 controller: SDF-vs-A64 two-live-leg A/B physics remeasure supervisor.

Host-side only — this file never imports Isaac, Kit, PhysX, pxr, or omni.
It validates the user-approved 4-sha tuple (D400:462-595 pattern), runs the
fail-closed admission contract with zero writes, writes the runtime freeze
manifest as the first forward-only artifact, then launches the D407 worker
once per leg (A = D344 attempt3 A64 base asset, B = D406 attempt1 SDF res256
derivative) under the D362 supervisor protocol (exclusive log create,
start_new_session, 1 s polling, per-leg watchdog 300 s inactivity / 900 s
total, SIGTERM then SIGKILL, retry 0).  Leg A must pass its full per-leg
verdict (operational AND observability) before leg B is launched (design
section 3.7 B2 branch rule).  After both legs it re-runs the asset freeze
contract, builds the B-minus-A delta summary and the Korean A/B comparison
sheet, runs the single 300 s live manual inspection, and writes the
completion summary.  It never classifies force closure, grasp, cap/rim
order, SDF general superiority, or G0a success; g0a_pass stays False.
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
import traceback
from pathlib import Path
from typing import Any

if not sys.dont_write_bytecode:
    raise RuntimeError(
        "D407 controller requires python -B (sys.dont_write_bytecode) before "
        "any third-party or project-local import"
    )

import psutil

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Safe pre-AppLauncher: the worker module's top level imports only
# d351/d361/numpy/psutil — no Isaac, Kit, pxr, or omni modules.
from sim_scripts import cyl34_top_view_d407_sdf_physics_ab_d362_remeasure_worker as w  # noqa: E402


CONTROLLER = Path(__file__).resolve()
WORKER = Path(w.__file__).resolve()
ATTEMPT_DIR = w.ATTEMPT_DIR
PREREG_PATH = w.PREREG_PATH
EXPECTED_PREREG_SHA256 = "6deb6779a18619f547952de9119eee599ea5dd40ac466d57d6a813988afb1269"

ATTESTATION_PATH = ATTEMPT_DIR / "d407_reviewed_script_attestation.json"
TUPLE_PATH = ATTEMPT_DIR / "d407_proposed_runtime_hash_tuple.json"
STATIC_RESULTS_PATH = ATTEMPT_DIR / "d407_static_fixture_results.json"
FREEZE_MANIFEST_PATH = ATTEMPT_DIR / "d407_runtime_freeze_manifest.json"
SUPERVISOR_SUMMARY_PATH = ATTEMPT_DIR / "d407_supervisor_summary.json"
DELTA_PATH = ATTEMPT_DIR / "d407_ab_delta_summary.json"
MANUAL_PATH = ATTEMPT_DIR / "d407_manual_visual_inspection.json"
COMPLETION_PATH = ATTEMPT_DIR / "d407_completion_summary.json"
AB_SHEET_PATH = ATTEMPT_DIR / "d407_ab_comparison_sheet_ko.png"
CONTROLLER_PHASE_PATH = ATTEMPT_DIR / "d407_controller_phase_markers.jsonl"

TUPLE_FIELDS = (
    "preregistration_sha256",
    "reviewed_script_attestation_sha256",
    "controller_script_sha256",
    "worker_script_sha256",
)
REGISTERED_STATIC_NEGATIVE_IDS = frozenset(
    {
        "prereg_hash_drift_rejected",
        "prereg_status_literal_tampered_rejected",
        "leg_asset_pin_tampered_rejected",
        "leg_asset_extra_file_rejected",
        "leg_asset_ab_root_identical_rejected",
        "capacity_formula_perturbed_rejected",
        "science_source_mutation_rejected",
        "session_doc_harness_sha_count_rejected",
        "dirty_path_outside_allowlist_rejected",
        "python_without_dash_b_rejected",
        "d361_prefix_header_forgery_rejected",
        "truncated_rrd_footer_rejected",
    }
)
EXPECTED_ZERO_STAGE_COUNTERS = {
    "script_imports": 0,
    "isaac_kit_physx_launches": 0,
    "simulation_app_launches": 0,
    "physics_worker_launches": 0,
    "usd_stage_creations_or_writes": 0,
    "gpu_runtime_jobs": 0,
    "physics_steps": 0,
    "q5_samples": 0,
    "contact_queries": 0,
    "cylinder_creates_or_writes": 0,
}
REQUIRED_STATIC_TRUE = (
    "static_ast_parse_pass",
    "top_level_runtime_side_effect_static_scan_pass",
    "science_source_identity_pass",
    "phase_order_static_contract_pass",
    "implementation_static_attestation_pass",
)

INTER_LEG_SETTLE_POLL_S = 5.0
INTER_LEG_SETTLE_MAX_S = 180.0
MANUAL_INSPECTION_WAIT_S = 300.0
MANUAL_INSPECTION_POLL_S = 0.25
GPU_DEVICE_NODES = ("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm")
NAMESPACE_LOW_PID_BOUND = 10
SHA256_RE = re.compile(r"[0-9a-f]{64}")

GPU_MODEL_EXACT = "NVIDIA GeForce RTX 4090 Laptop GPU"
GPU_COMPUTE_CAPABILITY_EXACT = "8.9"
SESSION_DOC_DESIGN_HEADING = "## 3. D407 확정 설계"

MEASURED_VERDICT = "D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_MEASURED"
FAIL_STOP_VERDICT = "D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP"
MANUAL_ARTIFACT = "D407_MANUAL_VISUAL_INSPECTION_V1"
MANUAL_LEG_SUBJECT_KEYS = (
    "jaw_or_gripper_visible",
    "cylinder_visible",
    "timeseries_legible",
    "no_text_overlap",
)
MANUAL_AB_SHEET_KEYS = ("delta_table_legible", "no_text_overlap")
AB_SHEET_SIZE = [3840, 1080]

MUST_REMAIN_NULL = {
    "force_closure": None,
    "stable_grasp": None,
    "cap_rim_or_barrel_order": None,
    "exact_manifold_or_face": None,
    "grasp_feasibility": None,
    "transfer_to_29x50_cylinders": None,
    "sdf_general_superiority": None,
    "per_prim_cooked_sdf_internal_identity": None,
    "collider_count_tipping_causality": None,
}
SCIENTIFIC_CAVEATS = {
    "s2_sdf_capacity_assumption": (
        "the 256 contacts/pair capacity term for the SDF pair is a project "
        "assumption inherited from the installed PhysX 5.6.1 convex-pair "
        "evidence, not a documented engine limit; the contact overflow audit "
        "is the fail-capable verifier"
    ),
    "s3_instanceable_link5_scope": (
        "instanceable=false is authored on the link5 collision scope of the "
        "leg B derivative too, so link5-side B-minus-A deltas are not solely "
        "attributable to the gripper representation variable"
    ),
    "f5_first_live_boundary": (
        "leg B live ContactSensor binding, SDF shape inventory, and the "
        "property-to-sensor path are a first live observation with no prior "
        "guarantee; D406 evidence covers property enumeration only"
    ),
}

_PHASE_SEQUENCE = 0


def _marker(phase: str, event: str, details: dict[str, Any] | None = None) -> None:
    """Controller-side durable phase marker (fsync per row).

    Never called before the runtime freeze manifest exists: the manifest is
    the first forward-only write of the attempt.
    """
    global _PHASE_SEQUENCE
    _PHASE_SEQUENCE += 1
    row = {
        "sequence": _PHASE_SEQUENCE,
        "utc": w._utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "phase": phase,
        "event": event,
        "details": details or {},
    }
    CONTROLLER_PHASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONTROLLER_PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key: {key}")
        seen[key] = value
    return seen


def _json_no_duplicates(text: str) -> dict[str, Any]:
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def _host_boundary_gate() -> None:
    """Fail closed before any gate evaluation or forward-only write (D403)."""
    nodes = {node: Path(node).exists() for node in GPU_DEVICE_NODES}
    pid = os.getpid()
    checks = {
        "all_gpu_device_nodes_visible": all(nodes.values()),
        "pid_above_namespace_low_range": pid > NAMESPACE_LOW_PID_BOUND,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "D407 host-boundary gate failed (sandboxed or GPU-less "
            f"environment): nodes={nodes}, pid={pid}, checks={checks}; "
            "nothing was written, no attempt was consumed"
        )


def _validate_approval_tuple(approved_sha256: str) -> dict[str, Any]:
    if SHA256_RE.fullmatch(approved_sha256) is None:
        raise RuntimeError(
            "--approved-tuple-sha256 must be exactly 64 lowercase hex characters"
        )
    if EXPECTED_PREREG_SHA256 != w.EXPECTED_PREREG_SHA256:
        raise RuntimeError(
            "D407 controller/worker EXPECTED_PREREG_SHA256 embeds disagree"
        )
    required = (PREREG_PATH, ATTESTATION_PATH, TUPLE_PATH, CONTROLLER, WORKER)
    missing = [w._rel(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"D407 approval files missing: {missing}")
    tuple_sha = w._sha(TUPLE_PATH)
    if tuple_sha != approved_sha256:
        raise RuntimeError(
            f"user-approved tuple SHA mismatch: {approved_sha256} != {tuple_sha}"
        )
    tuple_value = _json_no_duplicates(TUPLE_PATH.read_text(encoding="utf-8"))
    if tuple(tuple_value) != TUPLE_FIELDS or set(tuple_value) != set(TUPLE_FIELDS):
        raise RuntimeError(
            f"D407 tuple must have exact ordered fields {TUPLE_FIELDS}: {tuple_value}"
        )
    observed = {
        "preregistration_sha256": w._sha(PREREG_PATH),
        "reviewed_script_attestation_sha256": w._sha(ATTESTATION_PATH),
        "controller_script_sha256": w._sha(CONTROLLER),
        "worker_script_sha256": w._sha(WORKER),
    }
    if tuple_value != observed:
        raise RuntimeError(
            f"D407 tuple current-file hash mismatch: expected={tuple_value}, observed={observed}"
        )
    if observed["preregistration_sha256"] != EXPECTED_PREREG_SHA256:
        raise RuntimeError("D407 preregistration hash no longer matches reviewed V2")
    attestation = _json_no_duplicates(ATTESTATION_PATH.read_text(encoding="utf-8"))
    failed_static_fields = [
        field
        for field in REQUIRED_STATIC_TRUE
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
    if (
        failed_static_fields
        or not negative_static_exact
        or zero_stage_counters != EXPECTED_ZERO_STAGE_COUNTERS
    ):
        raise RuntimeError(
            "D407 static attestation required-field gate failed: "
            f"false_or_missing={failed_static_fields}, "
            f"negative_static_exact={negative_static_exact}, "
            "static_stage_zero_counters_exact="
            f"{zero_stage_counters == EXPECTED_ZERO_STAGE_COUNTERS}"
        )
    controller_binding = attestation.get("controller_script_path_and_sha256")
    worker_binding = attestation.get("worker_script_path_and_sha256")
    if controller_binding != {
        "path": w._rel(CONTROLLER),
        "sha256": observed["controller_script_sha256"],
    }:
        raise RuntimeError("D407 controller attestation binding mismatch")
    if worker_binding != {
        "path": w._rel(WORKER),
        "sha256": observed["worker_script_sha256"],
    }:
        raise RuntimeError("D407 worker attestation binding mismatch")
    if attestation.get("preregistration_sha256") != EXPECTED_PREREG_SHA256:
        raise RuntimeError("D407 attestation preregistration binding mismatch")
    return {
        "approved_tuple_sha256": approved_sha256,
        "tuple_path": w._rel(TUPLE_PATH),
        "tuple_sha256": tuple_sha,
        "tuple": tuple_value,
        "observed": observed,
        "attestation_static_pass": True,
        "pass": True,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _rerun_cli_report() -> dict[str, Any]:
    if not w.RERUN_CLI.is_file():
        return {"exists": False, "output": None, "pass": False}
    result = subprocess.run(
        [str(w.RERUN_CLI), "--version"], text=True, capture_output=True, check=False
    )
    output = (result.stdout + result.stderr).strip()
    return {
        "exists": True,
        "returncode": result.returncode,
        "output": output,
        "pass": result.returncode == 0 and w.RERUN_VERSION in output,
    }


def _conflicting_process_scan() -> dict[str, Any]:
    own = psutil.Process(os.getpid())
    allowed = {own.pid}
    for parent in own.parents():
        allowed.add(parent.pid)
    matches: dict[str, list[int]] = {}
    for pattern in ("isaac", "omni.kit", "rerun", "d407"):
        result = subprocess.run(
            ["pgrep", "-f", pattern], text=True, capture_output=True, check=False
        )
        pids = [
            int(line)
            for line in result.stdout.split()
            if line.strip().isdigit()
        ]
        matches[pattern] = sorted(pid for pid in pids if pid not in allowed)
    return {
        "non_ancestor_matches": matches,
        "pass": all(not pids for pids in matches.values()),
    }


def _runtime_output_paths() -> list[Path]:
    return [
        FREEZE_MANIFEST_PATH,
        SUPERVISOR_SUMMARY_PATH,
        DELTA_PATH,
        MANUAL_PATH,
        COMPLETION_PATH,
        AB_SHEET_PATH,
        CONTROLLER_PHASE_PATH,
        ATTEMPT_DIR / w.LEG_OUT_SUBDIRS[w.LEG_A],
        ATTEMPT_DIR / w.LEG_OUT_SUBDIRS[w.LEG_B],
    ]


def _admission(tuple_gate: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed admission contract.  Zero writes: any failure raises before
    the first forward-only artifact exists, so the attempt is not consumed."""
    head = w._git_head()
    origin = w._git_head("origin/master")
    gpu = w._gpu_snapshot()
    cli = _rerun_cli_report()
    prereg = w._json(PREREG_PATH)
    prereg_sha = w._sha(PREREG_PATH)
    frozen_values = w._frozen_value_contract()
    frozen_science = w._frozen_d362_science_source_contract()
    asset_contract = w._asset_dir_contract()
    session_text = w.SESSION_DOC.read_text(encoding="utf-8")
    controller_sha = w._sha(CONTROLLER)
    worker_sha = w._sha(WORKER)
    dirty_paths = w._status_paths()
    allowed_dirty = set(
        prereg.get("runtime_overlay_contract", {}).get("allowed_dirty_paths", [])
    )
    preexisting = [
        w._rel(path) for path in _runtime_output_paths() if path.exists()
    ]
    process_scan = _conflicting_process_scan()
    checks = {
        "head_origin_base_exact": head == origin == w.BASE_GIT,
        "registered_python": Path(sys.executable).resolve()
        == Path(w.REGISTERED_PYTHON).resolve(),
        "numpy_pin_1_26_0": _package_version("numpy") == "1.26.0",
        "psutil_pin_5_9_8": _package_version("psutil") == "5.9.8",
        "rerun_sdk_pin_0_34_1": _package_version("rerun-sdk") == w.RERUN_VERSION,
        "rerun_cli_pin_0_34_1": cli["pass"] is True,
        "korean_font_exists": w.FONT_PATH.is_file(),
        "gpu_model_exact": gpu.get("name") == GPU_MODEL_EXACT,
        "gpu_compute_capability_exact": gpu.get("compute_capability")
        == GPU_COMPUTE_CAPABILITY_EXACT,
        "gpu_free_vram_gate": int(gpu.get("memory_free_mib", 0))
        >= w.MIN_GPU_FREE_MIB,
        "ram_available_gate": int(gpu.get("ram_available_bytes", 0))
        >= w.MIN_RAM_AVAILABLE_BYTES,
        "prereg_sha_exact": prereg_sha == EXPECTED_PREREG_SHA256,
        "prereg_status_frozen": prereg.get("status")
        == "PREREGISTERED_NOT_EXECUTED",
        "prereg_case_exact": prereg.get("case") == w.CASE
        and prereg.get("case_name") == w.CASE_NAME,
        "prereg_new_variables_exact": prereg.get("new_variables")
        == w.NEW_VARIABLES,
        "prereg_legs_exact": isinstance(prereg.get("legs"), dict)
        and set(prereg["legs"]) == {w.LEG_A, w.LEG_B},
        "prereg_git_baseline_exact": prereg.get("git_baseline", {}).get("head")
        == w.BASE_GIT,
        "prereg_run_nonce_present": isinstance(prereg.get("run_nonce"), str)
        and bool(prereg.get("run_nonce")),
        "frozen_value_contract": frozen_values["pass"] is True,
        "frozen_d362_science_source_contract": frozen_science["pass"] is True,
        "leg_asset_freeze_contract": asset_contract["pass"] is True,
        "input_hashes_exact": w._input_hashes()
        == prereg.get("frozen_input_hashes"),
        "d334_sidecar_exact_before_first_write": w._sidecar_hashes()
        == prereg.get("d334_sidecar_before"),
        "registered_base_git_in_session_doc": session_text.count(w.BASE_GIT) >= 1,
        "controller_sha_in_session_doc_exactly_once": session_text.count(
            controller_sha
        )
        == 1,
        "worker_sha_in_session_doc_exactly_once": session_text.count(worker_sha)
        == 1,
        "session_doc_design_heading": SESSION_DOC_DESIGN_HEADING in session_text,
        "git_dirty_subset_of_allowlist": set(dirty_paths) <= allowed_dirty,
        "no_preexisting_runtime_outputs": not preexisting,
        "no_conflicting_processes": process_scan["pass"] is True,
    }
    admission = {
        "checks": checks,
        "pass": all(checks.values()),
        "tuple_gate": tuple_gate,
        "git_head": head,
        "git_origin_master": origin,
        "gpu_and_ram": gpu,
        "rerun_cli": cli,
        "controller_script_sha256": controller_sha,
        "worker_script_sha256": worker_sha,
        "preregistration_sha256": prereg_sha,
        "frozen_value_contract": frozen_values,
        "frozen_d362_science_source_contract": frozen_science,
        "leg_asset_dir_contract": asset_contract,
        "conflicting_process_scan": process_scan,
        "preexisting_runtime_outputs": preexisting,
        "dirty_paths": dirty_paths,
        "allowed_dirty_paths": sorted(allowed_dirty),
    }
    if not admission["pass"]:
        failed = sorted(name for name, ok in checks.items() if not ok)
        raise RuntimeError(
            "D407 admission contract failed before any write "
            f"(attempt not consumed): {failed}"
        )
    return admission


def _write_freeze_manifest(
    approved_sha256: str, admission: dict[str, Any]
) -> dict[str, Any]:
    """First forward-only write of the attempt (D401 snapshot semantics,
    simplified): capture the exact dirty state plus per-dirty-file hashes
    before any worker launch can change anything."""
    status_rows = w._git_status()
    # One git-status snapshot is the authority for both the preserved rows and
    # their hashed paths.  A second status command here could observe a
    # different filesystem moment and make the freeze manifest self-inconsistent.
    dirty_paths = [row[3:] for row in status_rows if len(row) > 3]
    dirty_file_sha256: dict[str, str | None] = {}
    for path in dirty_paths:
        target = w.REPO / path
        dirty_file_sha256[path] = w._sha(target) if target.is_file() else None
    unexpected = sorted(
        set(dirty_paths) - set(admission["allowed_dirty_paths"])
    )
    manifest = {
        "artifact": "D407_RUNTIME_FREEZE_MANIFEST_V1",
        "case": w.CASE,
        "utc": w._utc_now(),
        "controller_pid": os.getpid(),
        "controller_path": w._rel(CONTROLLER),
        "worker_path": w._rel(WORKER),
        "approved_tuple_sha256": approved_sha256,
        "tuple_gate": admission["tuple_gate"],
        "admission_checks": admission["checks"],
        "admission_pass": admission["pass"],
        "admission_evidence": {
            "gpu_and_ram": admission["gpu_and_ram"],
            "rerun_cli": admission["rerun_cli"],
            "frozen_value_contract": admission["frozen_value_contract"],
            "frozen_d362_science_source_contract": admission[
                "frozen_d362_science_source_contract"
            ],
            "leg_asset_dir_contract": admission["leg_asset_dir_contract"],
            "conflicting_process_scan": admission["conflicting_process_scan"],
        },
        "static_fixture_results_sha256": w._sha(STATIC_RESULTS_PATH)
        if STATIC_RESULTS_PATH.is_file()
        else None,
        "git": {
            "head": admission["git_head"],
            "origin_master": admission["git_origin_master"],
            "status_rows": status_rows,
            "allowed_dirty_paths": admission["allowed_dirty_paths"],
            "unexpected": unexpected,
            "dirty_file_sha256": dirty_file_sha256,
        },
        "checks": {
            "admission_pass": admission["pass"],
            "no_unexpected_dirty_paths": not unexpected,
        },
        "pass": admission["pass"] and not unexpected,
    }
    w._write_json_x(FREEZE_MANIFEST_PATH, manifest)
    if not manifest["pass"]:
        raise RuntimeError(
            f"D407 runtime freeze manifest checks failed: {manifest['checks']}"
        )
    return manifest


def _run_worker_leg(
    leg: str,
    prereg: dict[str, Any],
    active_process_groups: set[int],
) -> dict[str, Any]:
    """One supervised worker invocation for one leg (D362 _run pattern via the
    worker module's own per-leg audit helpers).  Retry budget is zero."""
    w._configure_leg(leg)
    leg_dir = w.OUT_DIR
    leg_dir.mkdir(parents=True, exist_ok=False)
    token = secrets.token_hex(32)
    invocation = {
        "artifact": "D407_SINGLE_ISAAC_INVOCATION_MARKER_V1",
        "utc": w._utc_now(),
        "leg": leg,
        "run_nonce": prereg["run_nonce"],
        "invocation_index": 1,
        "supervisor_pid": os.getpid(),
        "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "preregistration_sha256": w._sha(PREREG_PATH),
        "automatic_retry": False,
    }
    w._write_json_x(w.INVOCATION_PATH, invocation)
    command = [
        w.REGISTERED_PYTHON,
        "-B",
        str(WORKER),
        "--stage",
        "_worker",
        "--leg",
        leg,
        "--out_dir",
        str(w.ATTEMPT_DIR),
        "--seed",
        str(w.SEED),
    ]
    env = os.environ.copy()
    env.update(
        {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONUNBUFFERED": "1",
            w.WORKER_TOKEN_ENV: token,
            w.SUPERVISOR_PID_ENV: str(os.getpid()),
            "PATH": f"{w.RERUN_CLI.parent}:{env.get('PATH', '')}",
        }
    )
    log_path = w.WORKER_LOG_PATH
    phase_path = w.PHASE_PATH
    prefix_path = w.PREFIX_PATH
    start = time.monotonic()
    last_progress = start
    last_sizes = (-1, -1, -1)
    watchdog_triggered = False
    watchdog_reason = None
    process_group_cleanup: dict[str, Any] | None = None
    telemetry: list[dict[str, Any]] = []
    process: subprocess.Popen[Any] | None = None
    exit_code: int | None = None
    _marker("leg_worker", "start", {"leg": leg, "command": command})
    with log_path.open("xb") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            # Register immediately after Popen, before any stat/telemetry wait
            # can fail.  The outer controller exception boundary can therefore
            # still clean up a launched group if this function's own cleanup
            # path raises unexpectedly.
            active_process_groups.add(process.pid)
            while process.poll() is None:
                sizes = (
                    log_path.stat().st_size if log_path.exists() else 0,
                    phase_path.stat().st_size if phase_path.exists() else 0,
                    prefix_path.stat().st_size if prefix_path.exists() else 0,
                )
                if sizes != last_sizes:
                    last_sizes = sizes
                    last_progress = time.monotonic()
                now = time.monotonic()
                elapsed = now - start
                idle = now - last_progress
                try:
                    sample = w._gpu_snapshot()
                except Exception as error:
                    sample = {
                        "telemetry_error": f"{type(error).__name__}: {error}"
                    }
                telemetry.append(
                    {"elapsed_seconds": elapsed, "idle_seconds": idle, **sample}
                )
                if (
                    elapsed > w.TOTAL_WATCHDOG_S
                    or idle > w.INACTIVITY_WATCHDOG_S
                ):
                    watchdog_triggered = True
                    watchdog_reason = (
                        "total"
                        if elapsed > w.TOTAL_WATCHDOG_S
                        else "inactivity"
                    )
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=20.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=10.0)
                    break
                time.sleep(1.0)
            exit_code = process.wait()
        finally:
            if process is not None:
                process_group_cleanup = _ensure_process_group_gone(process.pid)
                if process_group_cleanup.get("pass") is True:
                    active_process_groups.discard(process.pid)
                if process.poll() is None:
                    try:
                        exit_code = process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        exit_code = None
            log.flush()
            os.fsync(log.fileno())
    if process is None or exit_code is None or process_group_cleanup is None:
        raise RuntimeError(
            "D407 worker supervision ended without a reaped leader and "
            "whole-process-group cleanup record"
        )
    contact_overflow = w._write_contact_overflow_warning_audit()
    recovery = w._write_supervisor_prefix_recovery_audit(exit_code)
    worker_summary_file = w._safe_json_file(w.WORKER_SUMMARY_PATH)
    payload = (
        worker_summary_file["payload"]
        if worker_summary_file["parse_pass"] is True
        else None
    )
    trace_rows: Any = None
    try:
        trace_rows = w._json(w.TRACE_JSON_PATH)
    except Exception:
        trace_rows = None
    trace_500_exact = bool(
        isinstance(trace_rows, list)
        and len(trace_rows) == w.BASELINE_STEPS + w.CLOSURE_MAX_STEPS
        and trace_rows
        and int(trace_rows[-1].get("global_step", -1))
        == w.BASELINE_STEPS + w.CLOSURE_MAX_STEPS
    )
    inventory = w._output_file_inventory()
    expected_inventory = w._expected_postworker_inventory()
    missing = sorted(set(expected_inventory) - set(inventory))
    unexpected = sorted(set(inventory) - set(expected_inventory))
    core_missing = sorted(w._core_postworker_inventory() - set(inventory))
    inventory_pass = not missing and not unexpected and not core_missing
    hash_manifest = w._inventory_hashes(inventory)
    phase_contract = (
        w._phase_contract(payload)
        if payload is not None and phase_path.is_file()
        else {"pass": False, "error": "worker summary or phase stream missing"}
    )
    observability_ok = bool(
        payload is not None and payload.get("observability_artifact_pass") is True
    )
    operational_checks = {
        "exit_code_zero": exit_code == 0,
        "watchdog_not_triggered": not watchdog_triggered,
        "worker_summary_parses_and_pass": worker_summary_file["parse_pass"] is True
        and payload is not None
        and payload.get("pass") is True,
        "frozen_open_baseline_pass": payload is not None
        and payload.get("baseline", {}).get("pass") is True,
        "controlled_500_rows_exact": trace_500_exact
        and payload is not None
        and payload.get("controlled_physics_steps")
        == w.BASELINE_STEPS + w.CLOSURE_MAX_STEPS,
        "no_worker_exception_file": not w.WORKER_EXCEPTION_PATH.exists(),
        "supervisor_prefix_recovery_success_path": recovery.get(
            "success_path_pass"
        )
        is True,
        "contact_overflow_warning_absent": contact_overflow.get("pass") is True,
        "postworker_inventory_integrity_pass": inventory_pass,
        "phase_contract_pass": phase_contract.get("pass") is True,
        "no_process_group_residue": process_group_cleanup.get("pass") is True,
        "no_forced_process_group_cleanup_on_success": process_group_cleanup.get(
            "cleanup_actions"
        )
        == [],
    }
    operational_pass = all(operational_checks.values())
    if not operational_pass:
        classification = f"leg_{leg}_operational"
    elif not observability_ok:
        classification = f"leg_{leg}_observability"
    else:
        classification = None
    record = {
        "leg": leg,
        "out_dir": w._rel(leg_dir),
        "command": command,
        "worker_pid": process.pid,
        "process_group_id": process.pid,
        "process_group_cleanup": process_group_cleanup,
        "worker_exit_code": exit_code,
        "elapsed_seconds": time.monotonic() - start,
        "watchdog_triggered": watchdog_triggered,
        "watchdog_reason": watchdog_reason,
        "automatic_retry": False,
        "telemetry_sample_count": len(telemetry),
        "telemetry": telemetry,
        "worker_summary_file": worker_summary_file,
        "worker_exception_exists": w.WORKER_EXCEPTION_PATH.is_file(),
        "supervisor_prefix_recovery_audit": recovery,
        "supervisor_prefix_recovery_audit_sha256": w._sha(
            w.SUPERVISOR_PREFIX_AUDIT_PATH
        ),
        "contact_overflow_warning_audit": contact_overflow,
        "contact_overflow_warning_audit_sha256": w._sha(
            w.CONTACT_OVERFLOW_WARNING_AUDIT_PATH
        ),
        "expected_postworker_inventory": expected_inventory,
        "postworker_inventory": inventory,
        "postworker_missing_expected_artifacts": missing,
        "postworker_unexpected_artifacts": unexpected,
        "postworker_core_missing": core_missing,
        "postworker_inventory_integrity_pass": inventory_pass,
        "postworker_hash_manifest": hash_manifest,
        "phase_contract": phase_contract,
        "operational_checks": operational_checks,
        "operational_pass": operational_pass,
        "observability_artifact_pass": observability_ok,
        "classification": classification,
        "physical_sub_verdict": (
            payload.get("closure", {}).get("verdict")
            if payload is not None
            else None
        ),
        "pass": operational_pass and observability_ok,
    }
    _marker(
        "leg_worker",
        "complete",
        {
            "leg": leg,
            "exit_code": exit_code,
            "pass": record["pass"],
            "classification": classification,
        },
    )
    return record


def _process_group_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _process_group_members(pgid: int) -> list[int]:
    members: list[int] = []
    for process in psutil.process_iter(["pid"]):
        pid = int(process.info["pid"])
        try:
            if os.getpgid(pid) == pgid:
                members.append(pid)
        except (ProcessLookupError, PermissionError, psutil.Error):
            continue
    return sorted(members)


def _ensure_process_group_gone(pgid: int) -> dict[str, Any]:
    """Bounded whole-process-group cleanup for every worker exit path."""
    initial_members = _process_group_members(pgid)
    actions: list[str] = []

    def wait_until_gone(timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if _process_group_gone(pgid):
                return True
            time.sleep(0.1)
        return _process_group_gone(pgid)

    if not _process_group_gone(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
            actions.append("SIGTERM")
        except ProcessLookupError:
            pass
        wait_until_gone(5.0)
    if not _process_group_gone(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
            actions.append("SIGKILL")
        except ProcessLookupError:
            pass
        wait_until_gone(5.0)
    final_members = _process_group_members(pgid)
    return {
        "process_group_id": pgid,
        "initial_members_after_leader_wait": initial_members,
        "cleanup_actions": actions,
        "final_members": final_members,
        "pass": _process_group_gone(pgid) and not final_members,
    }


def _inter_leg_settle(previous_group_pid: int) -> dict[str, Any]:
    """Design section 3.6.5 (2): poll every 5 s for at most 180 s until the
    exact host/GPU identity, leg A process group, VRAM, and RAM have all
    settled for two consecutive samples."""
    start = time.monotonic()
    samples: list[dict[str, Any]] = []
    consecutive_passes = 0
    while True:
        gpu = w._gpu_snapshot()
        device_nodes = {
            node: Path(node).exists() for node in GPU_DEVICE_NODES
        }
        conditions = {
            "all_gpu_device_nodes_visible": all(device_nodes.values()),
            "gpu_model_exact": gpu.get("name") == GPU_MODEL_EXACT,
            "gpu_compute_capability_exact": gpu.get("compute_capability")
            == GPU_COMPUTE_CAPABILITY_EXACT,
            "gpu_free_vram_recovered": int(gpu.get("memory_free_mib", 0))
            >= w.MIN_GPU_FREE_MIB,
            "leg_a_process_group_gone": _process_group_gone(previous_group_pid),
            "ram_available_recovered": int(gpu.get("ram_available_bytes", 0))
            >= w.MIN_RAM_AVAILABLE_BYTES,
        }
        consecutive_passes = consecutive_passes + 1 if all(conditions.values()) else 0
        samples.append(
            {
                "utc": w._utc_now(),
                "elapsed_seconds": time.monotonic() - start,
                "gpu_model": gpu.get("name"),
                "gpu_compute_capability": gpu.get("compute_capability"),
                "memory_free_mib": gpu.get("memory_free_mib"),
                "ram_available_bytes": gpu.get("ram_available_bytes"),
                "gpu_device_nodes": device_nodes,
                "consecutive_passes": consecutive_passes,
                "conditions": conditions,
            }
        )
        if consecutive_passes >= 2:
            return {
                "pass": True,
                "elapsed_seconds": time.monotonic() - start,
                "poll_interval_seconds": INTER_LEG_SETTLE_POLL_S,
                "max_wait_seconds": INTER_LEG_SETTLE_MAX_S,
                "required_consecutive_passes": 2,
                "samples": samples,
            }
        if time.monotonic() - start >= INTER_LEG_SETTLE_MAX_S:
            return {
                "pass": False,
                "elapsed_seconds": time.monotonic() - start,
                "poll_interval_seconds": INTER_LEG_SETTLE_POLL_S,
                "max_wait_seconds": INTER_LEG_SETTLE_MAX_S,
                "required_consecutive_passes": 2,
                "samples": samples,
            }
        time.sleep(INTER_LEG_SETTLE_POLL_S)


def _leg_final_metrics(trace_path: Path) -> dict[str, Any]:
    rows = w._json(trace_path)
    final = rows[-1]
    tilt = float(final["object_tilt_delta_from_reference_deg"])
    return {
        "row_count": len(rows),
        "final_row_index_zero_based": len(rows) - 1,
        "final_global_step": int(final["global_step"]),
        "disp_xy_mm": float(final["object_disp_xy_mm"]),
        "tilt_delta_deg": tilt,
        "z_delta_mm": float(final["object_z_delta_mm"]),
        "toppled_tilt_gt_45_descriptive": tilt > 45.0,
    }


def _leg_events(closure: dict[str, Any]) -> dict[str, Any]:
    events: dict[str, Any] = {}
    for label in ("link4", "link5", "gripper_link"):
        onset = int(closure["first_contact_step_by_filter"][label])
        events[f"{label}_contact"] = {
            "present": onset >= 0,
            "onset_phase_step": onset if onset >= 0 else None,
            "confirmation_phase_step": onset + 1 if onset >= 0 else None,
        }
    motion = int(closure["first_object_motion_step"])
    events["object_motion"] = {
        "present": motion >= 0,
        "onset_phase_step": motion if motion >= 0 else None,
        "confirmation_phase_step": motion + 1 if motion >= 0 else None,
    }
    return events


def _build_delta_summary(
    leg_summaries: dict[str, dict[str, Any]],
    leg_paths: dict[str, dict[str, Path]],
) -> dict[str, Any]:
    per_leg: dict[str, Any] = {}
    for leg in (w.LEG_A, w.LEG_B):
        closure = leg_summaries[leg]["closure"]
        finals = _leg_final_metrics(leg_paths[leg]["trace"])
        per_leg[leg] = {
            "collision_representation": (
                "a64_convex_64_plus_64"
                if leg == w.LEG_A
                else "link5_a64_64_plus_gripper_sdf_res256_mesh_1"
            ),
            "final_row": finals,
            "events": _leg_events(closure),
            "peak_force_by_filter": closure["peak_force_by_filter"],
            "physical_sub_verdict": closure["verdict"],
            "toppled_tilt_gt_45_descriptive": finals[
                "toppled_tilt_gt_45_descriptive"
            ],
        }
    a_final = per_leg[w.LEG_A]["final_row"]
    b_final = per_leg[w.LEG_B]["final_row"]
    peak_deltas: dict[str, Any] = {}
    for label in ("link4", "link5", "gripper_link"):
        a_peak = per_leg[w.LEG_A]["peak_force_by_filter"][label]["force_norm_n"]
        b_peak = per_leg[w.LEG_B]["peak_force_by_filter"][label]["force_norm_n"]
        peak_deltas[label] = (
            float(b_peak) - float(a_peak)
            if a_peak is not None and b_peak is not None
            else None
        )
    event_deltas: dict[str, Any] = {}
    for name in per_leg[w.LEG_A]["events"]:
        a_event = per_leg[w.LEG_A]["events"][name]
        b_event = per_leg[w.LEG_B]["events"][name]
        both = a_event["present"] and b_event["present"]
        event_deltas[name] = {
            "present_pair": {
                w.LEG_A: a_event["present"],
                w.LEG_B: b_event["present"],
            },
            "delta_onset_step": (
                b_event["onset_phase_step"] - a_event["onset_phase_step"]
                if both
                else None
            ),
            "delta_confirmation_step": (
                b_event["confirmation_phase_step"]
                - a_event["confirmation_phase_step"]
                if both
                else None
            ),
        }
    prerequisite_records = {
        leg: w._json(leg_paths[leg]["prerequisites"])
        for leg in (w.LEG_A, w.LEG_B)
    }
    cylinder_payloads = {
        leg: prerequisite_records[leg].get("cylinder_runtime_geometry")
        for leg in (w.LEG_A, w.LEG_B)
    }
    physics_setting_payloads = {
        leg: prerequisite_records[leg]
        .get("runtime_physics_settings", {})
        .get("canonical_comparison_payload")
        for leg in (w.LEG_A, w.LEG_B)
    }
    d362_final = _leg_final_metrics(w.D362_TRACE_JSON)
    d362_worker = w._json(w.D362_WORKER_SUMMARY)
    d362_events = _leg_events(d362_worker["closure"])
    a_events = per_leg[w.LEG_A]["events"]
    delta = {
        "artifact": "D407_AB_DELTA_SUMMARY_V1",
        "case": w.CASE,
        "utc": w._utc_now(),
        "delta_definition": (
            "each scalar delta = leg B value minus leg A value; event-step "
            "deltas only when both legs observed the event, else null with "
            "the presence flag pair"
        ),
        "per_leg": per_leg,
        "b_minus_a": {
            "final_disp_xy_mm": b_final["disp_xy_mm"] - a_final["disp_xy_mm"],
            "final_tilt_delta_deg": b_final["tilt_delta_deg"]
            - a_final["tilt_delta_deg"],
            "final_z_delta_mm": b_final["z_delta_mm"] - a_final["z_delta_mm"],
            "peak_force_n_by_filter": peak_deltas,
            "event_step_deltas": event_deltas,
        },
        "record_only_cross_leg_checks": {
            "gate": False,
            "cylinder_runtime_geometry_payloads": cylinder_payloads,
            "cylinder_runtime_geometry_equal": (
                cylinder_payloads[w.LEG_A] == cylinder_payloads[w.LEG_B]
            ),
            "runtime_physics_settings_payloads": physics_setting_payloads,
            "runtime_physics_settings_equal": (
                physics_setting_payloads[w.LEG_A]
                == physics_setting_payloads[w.LEG_B]
            ),
        },
        "leg_a_vs_frozen_d362_descriptive_only": {
            "gate": False,
            "authority": (
                "frozen D362 canonical trace/worker summary; differences are "
                "descriptive and do not decide D407 admission or verdict"
            ),
            "d362_final_row": d362_final,
            "a_minus_d362_final": {
                "disp_xy_mm": a_final["disp_xy_mm"] - d362_final["disp_xy_mm"],
                "tilt_delta_deg": (
                    a_final["tilt_delta_deg"] - d362_final["tilt_delta_deg"]
                ),
                "z_delta_mm": a_final["z_delta_mm"] - d362_final["z_delta_mm"],
            },
            "event_presence_and_a_minus_d362": {
                name: {
                    "a_present": a_events[name]["present"],
                    "d362_present": d362_events[name]["present"],
                    "a_minus_d362_onset_step": (
                        a_events[name]["onset_phase_step"]
                        - d362_events[name]["onset_phase_step"]
                        if a_events[name]["present"]
                        and d362_events[name]["present"]
                        else None
                    ),
                    "a_minus_d362_confirmation_step": (
                        a_events[name]["confirmation_phase_step"]
                        - d362_events[name]["confirmation_phase_step"]
                        if a_events[name]["present"]
                        and d362_events[name]["present"]
                        else None
                    ),
                }
                for name in a_events
            },
            "physical_sub_verdicts": {
                "leg_a": per_leg[w.LEG_A]["physical_sub_verdict"],
                "d362": d362_worker["closure"]["verdict"],
                "dxxx_normalized_equal": per_leg[w.LEG_A][
                    "physical_sub_verdict"
                ].replace("D407", "DXXX")
                == d362_worker["closure"]["verdict"].replace("D362", "DXXX"),
            },
        },
        "toppled_definition_descriptive": "final tilt_delta_deg > 45.0 (not a gate)",
        "science_null_boundary": dict(MUST_REMAIN_NULL),
        "g0a_pass": False,
    }
    w._write_json_x(DELTA_PATH, delta)
    return delta


def _fmt(value: Any, pattern: str = "{:.3f}") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return pattern.format(value)


def _build_ab_comparison_sheet(
    leg_paths: dict[str, dict[str, Path]], delta: dict[str, Any]
) -> dict[str, Any]:
    """Korean A/B comparison sheet, 3840x1080 PIL.

    Composition choice (registered in the authoring spec, step 6): instead of
    re-rendering canonical-trace frames (which would need the per-leg
    topology_parts that are never persisted to disk), the two per-leg beginner
    sheets — already validated 3840x1720 canonical-trace renders — are scaled
    to 1900x850 and placed side by side, with a Korean B-minus-A delta table
    strip at the bottom.  This is the simpler, robust composition.
    """
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (AB_SHEET_SIZE[0], AB_SHEET_SIZE[1]), (9, 13, 19))
    for index, leg in enumerate((w.LEG_A, w.LEG_B)):
        with Image.open(leg_paths[leg]["sheet"]) as sheet:
            scaled = sheet.convert("RGB").resize((1900, 850))
        canvas.paste(scaled, (13 + index * 1914, 10))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(w.FONT_PATH), 24)
    a_row = delta["per_leg"][w.LEG_A]["final_row"]
    b_row = delta["per_leg"][w.LEG_B]["final_row"]
    b_minus_a = delta["b_minus_a"]
    a_peak = delta["per_leg"][w.LEG_A]["peak_force_by_filter"]["gripper_link"]
    b_peak = delta["per_leg"][w.LEG_B]["peak_force_by_filter"]["gripper_link"]
    lines = [
        "왼쪽 = Leg A (A64 convex 64+64, control) / 오른쪽 = Leg B (link5 A64 64 + gripper SDF res256 mesh 1) — seed 33201, 200+300 step 동결 D362 계약",
        f"최종 XY 변위(mm): A={_fmt(a_row['disp_xy_mm'])} / B={_fmt(b_row['disp_xy_mm'])} / Δ(B−A)={_fmt(b_minus_a['final_disp_xy_mm'], '{:+.3f}')}",
        f"최종 기울기 변화(°): A={_fmt(a_row['tilt_delta_deg'])} / B={_fmt(b_row['tilt_delta_deg'])} / Δ(B−A)={_fmt(b_minus_a['final_tilt_delta_deg'], '{:+.3f}')}",
        f"최종 z 변화(mm): A={_fmt(a_row['z_delta_mm'])} / B={_fmt(b_row['z_delta_mm'])} / Δ(B−A)={_fmt(b_minus_a['final_z_delta_mm'], '{:+.3f}')}",
        f"gripper_link 최대 힘(N): A={_fmt(a_peak['force_norm_n'])} (step {a_peak['phase_step']}) / B={_fmt(b_peak['force_norm_n'])} (step {b_peak['phase_step']}) / Δ(B−A)={_fmt(b_minus_a['peak_force_n_by_filter']['gripper_link'], '{:+.3f}')}",
        f"전도 여부(tilt>45°, descriptive): A={delta['per_leg'][w.LEG_A]['toppled_tilt_gt_45_descriptive']} / B={delta['per_leg'][w.LEG_B]['toppled_tilt_gt_45_descriptive']}",
        "개선 게이트 사전등록 없음 — 측정 전용 case, g0a_pass=false (force closure/grasp/원인 주장 없음)",
    ]
    y = 872
    for line in lines:
        draw.text((26, y), line, font=font, fill=(226, 233, 243))
        y += 29
    with AB_SHEET_PATH.open("xb") as stream:
        canvas.save(stream, format="PNG")
    return w._png_report(AB_SHEET_PATH, AB_SHEET_SIZE)


def _manual_screenshot_bindings(
    leg_paths: dict[str, dict[str, Path]],
) -> dict[str, dict[str, str]]:
    return {
        "leg_a_rerun_screenshot": {
            "path": w._rel(leg_paths[w.LEG_A]["rerun_png"]),
            "sha256": w._sha(leg_paths[w.LEG_A]["rerun_png"]),
        },
        "leg_a_beginner_sheet": {
            "path": w._rel(leg_paths[w.LEG_A]["sheet"]),
            "sha256": w._sha(leg_paths[w.LEG_A]["sheet"]),
        },
        "leg_b_rerun_screenshot": {
            "path": w._rel(leg_paths[w.LEG_B]["rerun_png"]),
            "sha256": w._sha(leg_paths[w.LEG_B]["rerun_png"]),
        },
        "leg_b_beginner_sheet": {
            "path": w._rel(leg_paths[w.LEG_B]["sheet"]),
            "sha256": w._sha(leg_paths[w.LEG_B]["sheet"]),
        },
        "ab_comparison_sheet": {
            "path": w._rel(AB_SHEET_PATH),
            "sha256": w._sha(AB_SHEET_PATH),
        },
    }


def _run_manual_inspection(
    screenshots: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Single live manual inspection (D406 operating contract): prompt marker
    on stdout, 300 s window, 0.25 s polling, first read wins."""
    prompt = {
        "marker": "D407_MANUAL_INSPECTION_REQUIRED",
        "manual_path": w._rel(MANUAL_PATH),
        "window_seconds": MANUAL_INSPECTION_WAIT_S,
        "poll_interval_seconds": MANUAL_INSPECTION_POLL_S,
        "required_fields": {
            "artifact": MANUAL_ARTIFACT,
            "case": w.CASE,
            "subjects_visible": {
                w.LEG_A: list(MANUAL_LEG_SUBJECT_KEYS),
                w.LEG_B: list(MANUAL_LEG_SUBJECT_KEYS),
            },
            "ab_sheet": list(MANUAL_AB_SHEET_KEYS),
            "screenshots": screenshots,
            "pass": "true only when every subject check is honestly true",
        },
    }
    print("D407_MANUAL_INSPECTION_REQUIRED", flush=True)
    print(json.dumps(prompt, ensure_ascii=False), flush=True)
    _marker("manual_inspection", "prompted", {"manual_path": prompt["manual_path"]})
    deadline = time.monotonic() + MANUAL_INSPECTION_WAIT_S
    payload: dict[str, Any] | None = None
    read_error: str | None = None
    while time.monotonic() < deadline:
        if MANUAL_PATH.exists():
            # First read wins: the atomic writer renames within the attempt
            # filesystem, so existence implies a complete document.
            try:
                candidate = w._json(MANUAL_PATH)
                if not isinstance(candidate, dict):
                    raise TypeError("manual inspection JSON must be an object")
                payload = candidate
            except Exception as error:
                read_error = f"{type(error).__name__}: {error}"
            break
        time.sleep(MANUAL_INSPECTION_POLL_S)
    if payload is None:
        return {
            "received": False,
            "timeout": read_error is None,
            "read_error": read_error,
            "checks": {},
            "payload": None,
            "manual_inspection_sha256": w._sha(MANUAL_PATH)
            if MANUAL_PATH.is_file()
            else None,
            "pass": False,
        }
    subjects = payload.get("subjects_visible")
    ab_sheet = payload.get("ab_sheet")
    checks = {
        "artifact_exact": payload.get("artifact") == MANUAL_ARTIFACT,
        "case_exact": payload.get("case") == w.CASE,
        "leg_subjects_exact_keys_boolean_true": isinstance(subjects, dict)
        and set(subjects) == {w.LEG_A, w.LEG_B}
        and all(
            isinstance(subjects[leg], dict)
            and set(subjects[leg]) == set(MANUAL_LEG_SUBJECT_KEYS)
            and all(subjects[leg][key] is True for key in MANUAL_LEG_SUBJECT_KEYS)
            for leg in (w.LEG_A, w.LEG_B)
        ),
        "ab_sheet_exact_keys_boolean_true": isinstance(ab_sheet, dict)
        and set(ab_sheet) == set(MANUAL_AB_SHEET_KEYS)
        and all(ab_sheet[key] is True for key in MANUAL_AB_SHEET_KEYS),
        "screenshot_paths_and_sha256_verbatim": payload.get("screenshots")
        == screenshots,
        "manual_pass": payload.get("pass") is True,
    }
    return {
        "received": True,
        "timeout": False,
        "read_error": None,
        "checks": checks,
        "payload": payload,
        "manual_inspection_sha256": w._sha(MANUAL_PATH),
        "pass": all(checks.values()),
    }


def _controller_phase_contract() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    parse_error = None
    try:
        for line in CONTROLLER_PHASE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("controller phase row is not an object")
                rows.append(row)
    except Exception as error:
        parse_error = f"{type(error).__name__}: {error}"
    checks = {
        "parse_pass": parse_error is None,
        "nonempty": bool(rows),
        "sequence_exact": [row.get("sequence") for row in rows]
        == list(range(1, len(rows) + 1)),
        "monotonic_ns_nondecreasing": all(
            int(rows[index]["monotonic_ns"])
            <= int(rows[index + 1]["monotonic_ns"])
            for index in range(len(rows) - 1)
        ),
        "last_marker_completion_about_to_write": bool(rows)
        and rows[-1].get("phase") == "completion"
        and rows[-1].get("event") == "about_to_write",
    }
    return {
        "row_count": len(rows),
        "parse_error": parse_error,
        "checks": checks,
        "sha256": w._sha(CONTROLLER_PHASE_PATH)
        if CONTROLLER_PHASE_PATH.is_file()
        else None,
        "pass": all(checks.values()),
    }


def run_runtime(approved_tuple_sha256: str) -> int:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D407 controller requires python -B (sys.dont_write_bytecode) "
            "before any gate evaluation or write"
        )
    _host_boundary_gate()
    tuple_gate = _validate_approval_tuple(approved_tuple_sha256)
    admission = _admission(tuple_gate)
    prereg = w._json(PREREG_PATH)

    leg_blocks: dict[str, dict[str, Any]] = {}
    leg_paths: dict[str, dict[str, Path]] = {}
    leg_summaries: dict[str, dict[str, Any]] = {}
    settle_record: dict[str, Any] | None = None
    postrun_asset: dict[str, Any] | None = None
    latest_postrun_integrity: dict[str, Any] | None = None
    postrun_integrity_rechecks: list[dict[str, Any]] = []
    controller_exception: dict[str, Any] | None = None
    active_classification = "postrun_integrity"
    supervisor_state: dict[str, Any] = {"written": False, "sha256": None}
    active_process_groups: set[int] = set()
    emergency_cleanup_records: list[dict[str, Any]] = []
    failure_state: dict[str, Any] = {
        "initial_classification": None,
        "integrity_override": False,
    }
    delta_sha_at_creation: str | None = None
    root_artifact_integrity: dict[str, Any] | None = None

    def _cleanup_registered_worker_groups(reason: str) -> bool:
        all_pass = True
        for pgid in sorted(active_process_groups):
            try:
                record = _ensure_process_group_gone(pgid)
            except Exception as error:
                record = {
                    "process_group_id": pgid,
                    "pass": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            record = {"reason": reason, **record}
            emergency_cleanup_records.append(record)
            if record.get("pass") is True:
                active_process_groups.discard(pgid)
            else:
                all_pass = False
        return all_pass

    def _write_supervisor_summary(classification: str | None) -> None:
        if supervisor_state["written"]:
            return
        supervisor = {
            "artifact": "D407_SUPERVISOR_SUMMARY_V1",
            "case": w.CASE,
            "utc": w._utc_now(),
            "controller_pid": os.getpid(),
            "approved_tuple_sha256": approved_tuple_sha256,
            "freeze_manifest_sha256": (
                w._sha(FREEZE_MANIFEST_PATH)
                if FREEZE_MANIFEST_PATH.is_file()
                else None
            ),
            "legs": leg_blocks,
            "inter_leg_settle": settle_record,
            "postrun_asset_dir_contract": postrun_asset,
            "latest_postrun_integrity": latest_postrun_integrity,
            "postrun_integrity_rechecks": postrun_integrity_rechecks,
            "root_artifact_integrity": root_artifact_integrity,
            "emergency_process_group_cleanup": emergency_cleanup_records,
            "initial_failure_classification": failure_state[
                "initial_classification"
            ],
            "postrun_integrity_override": failure_state["integrity_override"],
            "controller_exception": controller_exception,
            "failure_classification": classification,
            "automatic_retry": False,
            "pass": classification is None
            and all(block["pass"] for block in leg_blocks.values())
            and len(leg_blocks) == 2,
        }
        _marker(
            "supervisor_summary",
            "about_to_write",
            {"pass": supervisor["pass"], "classification": classification},
        )
        w._write_json_x(SUPERVISOR_SUMMARY_PATH, supervisor)
        supervisor_state["written"] = True
        supervisor_state["sha256"] = w._sha(SUPERVISOR_SUMMARY_PATH)

    def _physical_by_leg() -> dict[str, Any]:
        return {
            leg: (
                leg_blocks[leg]["physical_sub_verdict"]
                if leg in leg_blocks
                else None
            )
            for leg in (w.LEG_A, w.LEG_B)
        }

    def _write_completion(
        final_verdict: str,
        classification: str | None,
        physical: dict[str, Any] | None,
        manual_record: dict[str, Any] | None,
        ab_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        phase_contract = _controller_phase_contract()
        if final_verdict == MEASURED_VERDICT and not phase_contract["pass"]:
            final_verdict = FAIL_STOP_VERDICT
            classification = "postrun_integrity"
            physical = None
            if failure_state["initial_classification"] is None:
                failure_state["initial_classification"] = classification
        completion = {
            "artifact": "D407_COMPLETION_SUMMARY_V1",
            "case": w.CASE,
            "utc": w._utc_now(),
            "final_verdict": final_verdict,
            "failure_classification": classification,
            "initial_failure_classification": failure_state[
                "initial_classification"
            ],
            "postrun_integrity_override": failure_state["integrity_override"],
            "per_leg_physical_sub_verdicts": physical,
            "freeze_manifest_sha256": (
                w._sha(FREEZE_MANIFEST_PATH)
                if FREEZE_MANIFEST_PATH.is_file()
                else None
            ),
            "supervisor_summary_sha256": supervisor_state["sha256"],
            "delta_summary_sha256": w._sha(DELTA_PATH)
            if DELTA_PATH.is_file()
            else None,
            "ab_comparison_sheet_report": ab_report,
            "manual_inspection": (
                {
                    "received": manual_record["received"],
                    "timeout": manual_record["timeout"],
                    "read_error": manual_record["read_error"],
                    "checks": manual_record["checks"],
                    "pass": manual_record["pass"],
                }
                if manual_record is not None
                else None
            ),
            "manual_inspection_sha256": (
                manual_record.get("manual_inspection_sha256")
                if manual_record is not None
                else None
            ),
            "must_remain_null": dict(MUST_REMAIN_NULL),
            "scientific_caveats": dict(SCIENTIFIC_CAVEATS),
            "controller_phase_contract": phase_contract,
            "root_artifact_integrity": root_artifact_integrity,
            "emergency_process_group_cleanup": emergency_cleanup_records,
            "controller_exception": controller_exception,
            "g0a_pass": False,
            "pass": final_verdict == MEASURED_VERDICT,
        }
        w._write_json_x(COMPLETION_PATH, completion)
        # Completion is the final file write.  A closed stdout pipe after that
        # point must not turn a durable PASS completion into process rc=1.
        try:
            print(
                json.dumps(
                    {
                        "stage": "runtime",
                        "pass": completion["pass"],
                        "verdict": final_verdict,
                        "classification": classification,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except (BrokenPipeError, OSError):
            pass
        return completion

    def _ensure_postrun_asset_contract() -> dict[str, Any] | None:
        nonlocal postrun_asset
        if postrun_asset is None:
            postrun_asset = w._asset_dir_contract()
            _marker(
                "postrun_asset_contract",
                "complete",
                {"pass": postrun_asset["pass"]},
            )
        return postrun_asset

    def _recheck_postworker_integrity(stage: str) -> dict[str, Any]:
        nonlocal latest_postrun_integrity
        per_leg: dict[str, Any] = {}
        for leg, block in leg_blocks.items():
            w._configure_leg(leg)
            inventory = w._output_file_inventory()
            hashes = w._inventory_hashes(inventory)
            payload = block.get("worker_summary_file", {}).get("payload")
            phase = (
                w._phase_contract(payload)
                if block.get("pass") is True
                and isinstance(payload, dict)
                and w.PHASE_PATH.is_file()
                else None
            )
            checks = {
                "inventory_unchanged": inventory
                == block.get("postworker_inventory"),
                "hash_manifest_unchanged": hashes
                == block.get("postworker_hash_manifest"),
                "successful_leg_phase_contract_still_passes": (
                    phase is not None and phase.get("pass") is True
                    if block.get("pass") is True
                    else True
                ),
                "successful_leg_phase_sha_unchanged": (
                    phase is not None
                    and phase.get("phase_sha256")
                    == block.get("phase_contract", {}).get("phase_sha256")
                    if block.get("pass") is True
                    else True
                ),
            }
            per_leg[leg] = {
                "checks": checks,
                "inventory": inventory,
                "hash_manifest": hashes,
                "phase_contract": phase,
                "pass": all(checks.values()),
            }
        allowed_dirty = set(
            prereg.get("runtime_overlay_contract", {}).get(
                "allowed_dirty_paths", []
            )
        )
        tuple_recheck: dict[str, Any] | None = None
        tuple_recheck_error: str | None = None
        try:
            tuple_recheck = _validate_approval_tuple(approved_tuple_sha256)
        except Exception as error:
            tuple_recheck_error = f"{type(error).__name__}: {error}"
        global_checks = {
            "all_completed_leg_artifacts_unchanged": all(
                record["pass"] for record in per_leg.values()
            ),
            "frozen_inputs_unchanged": w._input_hashes()
            == prereg.get("frozen_input_hashes"),
            "d334_sidecar_unchanged": w._sidecar_hashes()
            == prereg.get("d334_sidecar_before"),
            "postrun_asset_contract_pass": postrun_asset is not None
            and postrun_asset.get("pass") is True,
            "git_dirty_subset_of_allowlist": set(w._status_paths())
            <= allowed_dirty,
            "head_origin_base_unchanged": w._git_head()
            == w._git_head("origin/master")
            == w.BASE_GIT,
            "prereg_sha_unchanged": w._sha(PREREG_PATH)
            == EXPECTED_PREREG_SHA256,
            "approval_tuple_and_bound_files_unchanged": tuple_recheck is not None
            and tuple_recheck.get("pass") is True,
        }
        report = {
            "stage": stage,
            "sequence": len(postrun_integrity_rechecks) + 1,
            "per_leg": per_leg,
            "global_checks": global_checks,
            "tuple_recheck": tuple_recheck,
            "tuple_recheck_error": tuple_recheck_error,
            "pass": all(global_checks.values()),
        }
        postrun_integrity_rechecks.append(report)
        latest_postrun_integrity = report
        return report

    def _fail_stop(
        classification: str,
        *,
        physical_null: bool = False,
        manual_record: dict[str, Any] | None = None,
        ab_report: dict[str, Any] | None = None,
    ) -> int:
        nonlocal controller_exception
        if failure_state["initial_classification"] is None:
            failure_state["initial_classification"] = classification
        cleanup_pass = _cleanup_registered_worker_groups(
            f"before_{classification}_fail_stop"
        )
        if not cleanup_pass:
            physical_null = True
            failure_state["integrity_override"] = True
        _marker("fail_stop", "start", {"classification": classification})
        _ensure_postrun_asset_contract()
        integrity = _recheck_postworker_integrity(
            f"before_{classification}_completion"
        )
        if not integrity["pass"]:
            physical_null = True
            failure_state["integrity_override"] = True
        _write_supervisor_summary(classification)
        physical = None if physical_null else _physical_by_leg()
        _marker(
            "completion",
            "about_to_write",
            {"pass": False, "classification": classification},
        )
        _write_completion(
            FAIL_STOP_VERDICT, classification, physical, manual_record, ab_report
        )
        return 1

    try:
        # The canonical exception boundary starts before the first runtime
        # write.  Admission above is read-only; freeze manifest is the attempt's
        # first forward-only runtime artifact.
        _write_freeze_manifest(approved_tuple_sha256, admission)
        _marker(
            "freeze_manifest",
            "complete",
            {
                "path": w._rel(FREEZE_MANIFEST_PATH),
                "approved": approved_tuple_sha256,
            },
        )

        # Leg loop, fixed order A then B (design section 3.7 B2 branch rule:
        # leg A must fully pass — operational AND observability — before leg B
        # is ever launched; any leg failure consumes the attempt).
        for leg in (w.LEG_A, w.LEG_B):
            active_classification = "postrun_integrity"
            leg_tuple_gate = _validate_approval_tuple(approved_tuple_sha256)
            _marker(
                "approval_tuple_recheck",
                "complete",
                {"leg": leg, "pass": leg_tuple_gate["pass"]},
            )
            active_classification = f"leg_{leg}_operational"
            block = _run_worker_leg(leg, prereg, active_process_groups)
            leg_blocks[leg] = block
            leg_paths[leg] = {
                "dir": w.OUT_DIR,
                "summary": w.WORKER_SUMMARY_PATH,
                "trace": w.TRACE_JSON_PATH,
                "prerequisites": w.PREREQUISITE_PATH,
                "sheet": w.SHEET_PATH,
                "rerun_png": w.RERUN_PNG_PATH,
            }
            if not block["pass"]:
                active_classification = block["classification"]
                return _fail_stop(block["classification"])
            leg_summaries[leg] = block["worker_summary_file"]["payload"]
            if leg == w.LEG_A:
                active_classification = "inter_leg_settle"
                settle_record = _inter_leg_settle(block["process_group_id"])
                _marker(
                    "inter_leg_settle",
                    "complete",
                    {
                        "pass": settle_record["pass"],
                        "elapsed_seconds": settle_record["elapsed_seconds"],
                    },
                )
                if not settle_record["pass"]:
                    return _fail_stop("inter_leg_settle")

        # Post-run cross-leg integrity: re-run the asset freeze contract once
        # after the workers and recheck the exact per-leg inventories/hashes
        # before any trace is consumed by the delta layer.
        active_classification = "postrun_integrity"
        _ensure_postrun_asset_contract()
        before_delta = _recheck_postworker_integrity("before_delta")
        if postrun_asset is None or postrun_asset["pass"] is not True:
            return _fail_stop("postrun_integrity", physical_null=True)
        if not before_delta["pass"]:
            return _fail_stop("postrun_integrity", physical_null=True)

        delta = _build_delta_summary(leg_summaries, leg_paths)
        delta_sha_at_creation = w._sha(DELTA_PATH)
        _marker("delta_summary", "complete", {"path": w._rel(DELTA_PATH)})
        active_classification = "manual_inspection"
        ab_report = _build_ab_comparison_sheet(leg_paths, delta)
        _marker("ab_comparison_sheet", "complete", {"pass": ab_report["pass"]})
        if ab_report["pass"] is not True:
            # The comparison sheet is the manual-inspection subject; a failed
            # sheet makes the inspection impossible, so it is classified under
            # the manual_inspection observability branch.
            return _fail_stop("manual_inspection", ab_report=ab_report)

        screenshots = _manual_screenshot_bindings(leg_paths)
        manual_record = _run_manual_inspection(screenshots)
        _marker(
            "manual_inspection",
            "complete",
            {"received": manual_record["received"], "pass": manual_record["pass"]},
        )
        if not manual_record["pass"]:
            return _fail_stop(
                "manual_inspection",
                manual_record=manual_record,
                ab_report=ab_report,
            )

        active_classification = "postrun_integrity"
        before_completion = _recheck_postworker_integrity("before_completion")
        if not before_completion["pass"]:
            return _fail_stop(
                "postrun_integrity",
                physical_null=True,
                manual_record=manual_record,
                ab_report=ab_report,
            )
        current_ab_report = w._png_report(AB_SHEET_PATH, AB_SHEET_SIZE)
        current_screenshots = _manual_screenshot_bindings(leg_paths)
        root_checks = {
            "manual_inspection_file_sha_unchanged": MANUAL_PATH.is_file()
            and w._sha(MANUAL_PATH)
            == manual_record["manual_inspection_sha256"],
            "delta_summary_sha_unchanged": DELTA_PATH.is_file()
            and delta_sha_at_creation is not None
            and w._sha(DELTA_PATH) == delta_sha_at_creation,
            "ab_sheet_report_unchanged": current_ab_report == ab_report,
            "manual_screenshot_bindings_unchanged": current_screenshots
            == screenshots,
        }
        root_artifact_integrity = {
            "checks": root_checks,
            "delta_sha256_at_creation": delta_sha_at_creation,
            "current_delta_sha256": (
                w._sha(DELTA_PATH) if DELTA_PATH.is_file() else None
            ),
            "ab_report_at_creation": ab_report,
            "current_ab_report": current_ab_report,
            "screenshots_at_inspection": screenshots,
            "current_screenshots": current_screenshots,
            "pass": all(root_checks.values()),
        }
        if not root_artifact_integrity["pass"]:
            return _fail_stop(
                "postrun_integrity",
                physical_null=True,
                manual_record=manual_record,
                ab_report=ab_report,
            )

        # Final success binding, immediately before supervisor/completion
        # writes, closes the manual-inspection window against script/tuple drift.
        _validate_approval_tuple(approved_tuple_sha256)
        _write_supervisor_summary(None)
        _marker(
            "completion",
            "about_to_write",
            {"pass": True, "classification": None},
        )
        completion = _write_completion(
            MEASURED_VERDICT,
            None,
            _physical_by_leg(),
            manual_record,
            ab_report,
        )
        return 0 if completion["pass"] else 1
    except Exception as error:
        _cleanup_registered_worker_groups("outer_exception")
        if COMPLETION_PATH.exists():
            try:
                print(traceback.format_exc())
            except (BrokenPipeError, OSError):
                pass
            return 1
        controller_exception = {
            "classification_at_exception": active_classification,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        try:
            return _fail_stop(active_classification, physical_null=True)
        except Exception as finalization_error:
            # Last-resort canonical attempt closure.  It intentionally performs
            # no write after completion and never retries a worker.
            if not COMPLETION_PATH.exists():
                fallback = {
                    "artifact": "D407_COMPLETION_SUMMARY_V1",
                    "case": w.CASE,
                    "utc": w._utc_now(),
                    "final_verdict": FAIL_STOP_VERDICT,
                    "failure_classification": active_classification,
                    "initial_failure_classification": (
                        failure_state["initial_classification"]
                        or active_classification
                    ),
                    "postrun_integrity_override": failure_state[
                        "integrity_override"
                    ],
                    "per_leg_physical_sub_verdicts": None,
                    "freeze_manifest_sha256": (
                        w._sha(FREEZE_MANIFEST_PATH)
                        if FREEZE_MANIFEST_PATH.is_file()
                        else None
                    ),
                    "supervisor_summary_sha256": (
                        w._sha(SUPERVISOR_SUMMARY_PATH)
                        if SUPERVISOR_SUMMARY_PATH.is_file()
                        else None
                    ),
                    "delta_summary_sha256": (
                        w._sha(DELTA_PATH) if DELTA_PATH.is_file() else None
                    ),
                    "controller_exception": controller_exception,
                    "emergency_process_group_cleanup": emergency_cleanup_records,
                    "canonical_finalization_exception": (
                        f"{type(finalization_error).__name__}: "
                        f"{finalization_error}"
                    ),
                    "must_remain_null": dict(MUST_REMAIN_NULL),
                    "scientific_caveats": dict(SCIENTIFIC_CAVEATS),
                    "g0a_pass": False,
                    "pass": False,
                }
                w._write_json_x(COMPLETION_PATH, fallback)
            try:
                print(traceback.format_exc())
            except (BrokenPipeError, OSError):
                pass
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-tuple-sha256", required=True)
    args = parser.parse_args()
    try:
        return run_runtime(args.approved_tuple_sha256)
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
