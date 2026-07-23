#!/usr/bin/env python3
"""D377 StageCache erase-before-close lifecycle localization controller.

This case reproduces the frozen D375 live acquisition workload and changes one
lifecycle operation only: after the inherited PhysX detach returns, erase the
same in-memory USD stage from the singleton StageCache exactly once.  The
controller owns preregistration, one process-group-isolated worker invocation,
a bounded watchdog, lifecycle-only analysis, and offline observability.

It intentionally does not run the D375 authored/callback geometry classifier
and does not make a P34 identity, physics, contact, q5, cylinder, tipping, or
grasp judgment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

CASE_ROOT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d377"
ATTEMPT = "attempt1_stagecache_erase_before_close_localization"
OUT_DIR = CASE_ROOT / ATTEMPT
PREREG_PATH = OUT_DIR / "d377_preregistration.json"
PHASE_PATH = OUT_DIR / "d377_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d377_worker_invocation.json"
SUPERVISOR_PATH = OUT_DIR / "d377_worker_supervisor.json"
STDOUT_PATH = OUT_DIR / "d377_worker_stdout.log"
STDERR_PATH = OUT_DIR / "d377_worker_stderr.log"
CLAIM_PATH = OUT_DIR / "d377_worker_claim.json"
RAW_PATH = OUT_DIR / "d377_worker_raw_summary.json"
PRECLOSE_PATH = OUT_DIR / "d377_worker_preclose_sentinel.json"
WORKER_EXCEPTION_PATH = OUT_DIR / "d377_worker_exception.json"
EVIDENCE_PATH = OUT_DIR / "d377_stagecache_erase_localization_evidence.json"
BOARD_PATH = OUT_DIR / "d377_d375_vs_d377_stagecache_erase_1920x1080.png"
RRD_PATH = OUT_DIR / "d377_stagecache_erase_localization.rrd"
RBL_PATH = OUT_DIR / "d377_stagecache_erase_localization.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d377_rerun_validation.json"
RERUN_PNG_PATH = OUT_DIR / "d377_rerun_inspection.png"
AUTOMATED_PATH = OUT_DIR / "d377_automated_summary.json"
MANUAL_TEMPLATE_PATH = OUT_DIR / "d377_manual_visual_inspection_template.json"
MANUAL_PATH = OUT_DIR / "d377_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d377_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d377_controller_exception.json"

HARNESS = Path(__file__).resolve()
WORKER = REPO / "sim_scripts/cyl34_top_view_d377_d375_stagecache_erase_before_close_localization_worker.py"
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
D375_WORKER = REPO / "sim_scripts/cyl34_top_view_d375_p34_live_asset_identity_contract_repair_worker.py"
D375_CONTROLLER = REPO / "sim_scripts/cyl34_top_view_d375_p34_live_asset_identity_contract_repair.py"
D375_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d375/"
    "attempt2_external_gpu_attestation_repair"
)
D375_RAW = D375_DIR / "d375_worker_raw_summary.json"
D375_PRECLOSE = D375_DIR / "d375_worker_preclose_sentinel.json"
D375_SUPERVISOR = D375_DIR / "d375_worker_supervisor.json"
D375_FAIL = D375_DIR / "d375_fail_stop_attestation.json"
D376_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d376/"
    "attempt1_d375_terminal_close_provenance_and_failure_visualization"
)
D376_EVIDENCE = D376_DIR / "d376_terminal_close_provenance_evidence.json"
D376_COMPLETION = D376_DIR / "d376_completion_summary.json"
D376_OFFICIAL = D376_DIR / "d376_nvidia_official_source_attestation.json"
D376_KIT_LOG = D376_DIR / "d376_frozen_d375_kit_log.txt"
D372_GEOMETRY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d372/"
    "attempt2_external_schema_path_repair/"
    "d372_professor_semantic_candidate_geometry.json"
)
D373_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d373/"
    "attempt1_p34_live_asset_identity_preflight"
)
D373_RAW = D373_DIR / "d373_worker_raw_summary.json"
D373_ASSET_DIR = D373_DIR / "collision_asset/roarm_m3_p34_semantic_compound"
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
PHYSX_HELPER = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "omni/physx/scripts/utils.py"
)
SIMULATION_APP_SOURCE = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py"
)
ISAAC_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

EXPECTED_HEAD = "e30f7f99d44252f509e383627738f3ad7967ea93"
WORKER_TIMEOUT_S = 120.0
TERM_GRACE_S = 20.0
KILL_WAIT_S = 20.0
NEW_VARIABLES = ["explicit_stagecache_erase_after_physx_detach_v1"]
VERDICT_PASS = "D377_STAGECACHE_ERASE_BEFORE_CLOSE_LIFECYCLE_PASS_NO_SCIENCE"
VERDICT_FAIL = "D377_STAGECACHE_ERASE_BEFORE_CLOSE_LOCALIZATION_FAIL_STOP"

EXPECTED_FIXED_HASHES = {
    D375_CONTROLLER: "70cb1bf9b0d518117fe90c848a9cefbdb95c8a0cabd4f3fa0fc5a37aae29c5e1",
    D375_WORKER: "4b2bbef3cf445ef4c9c9a8de2bac8a01087180502ef134129f9bcfb444020fa4",
    D375_RAW: "74f959b765860d06ca1d892823d47dc395cad3aea92d0250e21ff706263fc21e",
    D375_PRECLOSE: "1352d49f63b1ba58c75c1e5ad4d0bcb2d000510f1fc060938d672c53288d5203",
    D375_SUPERVISOR: "69f5f8ec5760e7804f3d076c377fc0ea597bde902f3d8ec7d941f36208f4f51c",
    D375_FAIL: "c3fb645ae9ca918e433bdf1734561504aab01a63d97d50086393c16b5d6f8fc7",
    D376_EVIDENCE: "8a29d8df80ac769cffde3fae785d4e14dd8567d716ea0b5c091c16049e871264",
    D376_COMPLETION: "0ab9b55b79b8cd0351c2d174244d32068738bdb4bcfb1d1a79f1de9bfdf3b5a7",
    D376_OFFICIAL: "df253c95c5aa552363cf631c4e832ae292dd9f1704a3402468990f5b84b56691",
    D376_KIT_LOG: "6522efde45e776fabf3186ddf362d509a6a3b04f999adc5024c28f41dce1ccc9",
    PHYSX_HELPER: "d7e62f14d065257032a5f0d7d54f24840df25babb51311c5d779aa459e20dcc1",
    SIMULATION_APP_SOURCE: "7cbaa6f00e935a6f14bf1c28ec0db089fd924e931f3b0deee07a822f9b7d0090",
}

OFFICIAL_SOURCES = [
    {
        "title": "Isaac Sim 5.1.0 SimulationApp API",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html",
        "applicable_version": "installed Isaac Sim 5.1.0.0 / SimulationApp extension 2.12.2",
        "use": "graceful versus immediate close and fast_shutdown default semantics",
    },
    {
        "title": "Isaac Sim 5.1.0 Release Notes",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html",
        "applicable_version": "installed Isaac Sim 5.1.0.0",
        "use": "version-matched shutdown context; no exact D375 root-cause claim",
    },
    {
        "title": "Kit SDK 107.3.1 UsdUtils StageCache API",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/107.3.1/pxr/UsdUtils.html",
        "applicable_version": "near-match to installed Kit 107.3.3 / USD 107.3.1 docs",
        "use": "singleton StageCache API identity",
    },
    {
        "title": "Kit SDK 107.3.1 Usd StageCache API",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/107.3.1/pxr/Usd.html",
        "applicable_version": "near-match to installed Kit 107.3.3 / USD 107.3.1 docs",
        "use": "GetId, Find, Contains, and Erase semantics",
    },
    {
        "title": "Isaac Sim 6.0.0 Release Notes",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html",
        "applicable_version": "later-version mechanism evidence only; not installed version",
        "use": "bug 5948099 replaced shutdown_and_release_framework to avoid GIL/tasking teardown deadlock",
    },
]


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


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha_bytes(payload)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _phase(name: str, **fields: Any) -> None:
    ordinal = 1
    if PHASE_PATH.is_file():
        ordinal = sum(
            1 for line in PHASE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
        ) + 1
    row = {
        "ordinal": ordinal,
        "phase": name,
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_worker_module() -> Any:
    spec = importlib.util.spec_from_file_location("d377_worker_source_audit", WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import D377 worker source-audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _process_snapshot() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,pgid=,stat=,comm=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 5)
        if len(parts) < 6:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "pgid": int(parts[2]),
                "stat": parts[3],
                "comm": parts[4],
                "args": parts[5],
            }
        )
    return rows


def _active_isaac_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    own_pid = os.getpid()
    needles = (
        "isaac-sim",
        "isaacsim.exp.full",
        "cyl34_top_view_d375_p34_live_asset_identity_contract_repair_worker.py",
        "cyl34_top_view_d377_d375_stagecache_erase_before_close_localization_worker.py",
    )
    return [
        row
        for row in rows
        if row["pid"] != own_pid
        and (row["comm"] in {"kit", "kit-bin"} or any(token in row["args"] for token in needles))
    ]


def _gpu_attestation() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    parsed = None
    if completed.returncode == 0 and completed.stdout.strip():
        fields = [field.strip() for field in completed.stdout.strip().splitlines()[0].split(",")]
        if len(fields) == 5:
            parsed = {
                "name": fields[0],
                "driver_version": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_free_mib": int(fields[3]),
                "compute_capability": fields[4],
            }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "parsed": parsed,
        "pass": bool(
            parsed
            and parsed["name"] == "NVIDIA GeForce RTX 4090 Laptop GPU"
            and parsed["driver_version"] == "580.159.03"
            and parsed["memory_total_mib"] == 16376
            and parsed["compute_capability"] == "8.9"
        ),
    }


def _gpu_compute_rows() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    rows = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
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


def _logic_gate(
    *,
    workload_equivalent: bool,
    erase_exact: bool,
    hash_authority: bool,
    close_start: bool,
    returncode: int,
    timed_out: bool,
    signaled: bool,
    reaped: bool,
    residue: bool,
) -> bool:
    return bool(
        workload_equivalent
        and erase_exact
        and hash_authority
        and close_start
        and returncode == 0
        and not timed_out
        and not signaled
        and reaped
        and not residue
    )


def _negative_controls() -> dict[str, Any]:
    base = {
        "workload_equivalent": True,
        "erase_exact": True,
        "hash_authority": True,
        "close_start": True,
        "returncode": 0,
        "timed_out": False,
        "signaled": False,
        "reaped": True,
        "residue": False,
    }
    cases = {
        "baseline_accepts_without_optional_close_return_marker": (base, True),
        "mutated_frozen_or_summary_hash_rejected": ({**base, "hash_authority": False}, False),
        "cache_before_absent_or_erase_postcondition_rejected": ({**base, "erase_exact": False}, False),
        "erase_counter_two_rejected_by_erase_exact": ({**base, "erase_exact": False}, False),
        "spoofed_return_zero_with_timeout_rejected": ({**base, "timed_out": True}, False),
        "spoofed_return_zero_with_signal_rejected": ({**base, "signaled": True}, False),
        "clean_exit_with_process_or_gpu_residue_rejected": ({**base, "residue": True}, False),
    }
    rows = {}
    for name, (payload, expected) in cases.items():
        observed = _logic_gate(**payload)
        rows[name] = {"expected": expected, "observed": observed, "pass": observed is expected}
    return {"rows": rows, "pass": all(row["pass"] for row in rows.values())}


def _source_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in (HARNESS, WORKER, VIZ_DEBUG, RERUN_CONTRACT)}


def prepare() -> int:
    if CASE_ROOT.exists():
        raise RuntimeError(f"forward-only D377 path already exists: {_rel(CASE_ROOT)}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    frozen_hash_checks = {
        _rel(path): path.is_file() and _sha(path) == expected
        for path, expected in EXPECTED_FIXED_HASHES.items()
    }
    d373_raw = _read_json(D373_RAW)
    registered_assets = d373_raw["asset"]["variant_file_hashes"]
    actual_assets = _inventory(D373_ASSET_DIR)
    worker_attestation = _load_worker_module().source_attestation()
    negatives = _negative_controls()
    gpu = _gpu_attestation()
    active_isaac = _active_isaac_rows(_process_snapshot())
    inputs = {
        _rel(path): _sha(path)
        for path in (
            D372_GEOMETRY,
            D373_RAW,
            D375_RAW,
            D375_PRECLOSE,
            D375_SUPERVISOR,
            D375_FAIL,
            D376_EVIDENCE,
            D376_COMPLETION,
            D376_OFFICIAL,
            D376_KIT_LOG,
            PHYSX_HELPER,
            SIMULATION_APP_SOURCE,
        )
    }
    checks = {
        "git_head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_master_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "one_new_variable": len(NEW_VARIABLES) == 1,
        "all_frozen_hashes_exact": all(frozen_hash_checks.values()),
        "d373_asset_inventory_exact": actual_assets == registered_assets,
        "worker_source_transform_pass": worker_attestation["pass"] is True,
        "negative_controls_pass": negatives["pass"] is True,
        "no_preexisting_isaac_or_kit_worker": not active_isaac,
        "gpu_attestation_pass": gpu["pass"] is True,
        "isaacsim_version_exact": _package_version("isaacsim") == "5.1.0.0",
        "isaaclab_version_exact": _package_version("isaaclab") == "2.3.0",
        "numpy_pin_exact": _package_version("numpy") == "1.26.0",
        "psutil_pin_exact": _package_version("psutil") == "5.9.8",
        "rerun_sdk_pin_exact": _package_version("rerun-sdk") == "0.34.1",
        "isaac_python_exact": ISAAC_PYTHON.is_file(),
        "rerun_cli_exact": RERUN_CLI.is_file(),
        "fonts_exist": FONT_REGULAR.is_file() and FONT_BOLD.is_file(),
    }
    prereg = {
        "artifact": "D377_STAGECACHE_ERASE_BEFORE_CLOSE_PREREGISTRATION_V1",
        "case": "g0a_d377",
        "attempt": ATTEMPT,
        "what_and_why": (
            "Test whether explicit removal of the already-detached D375 in-memory stage from "
            "the singleton StageCache is sufficient for one clean process exit."
        ),
        "new_variables": NEW_VARIABLES,
        "single_variable_definition": {
            "before": "D375 successful PhysX detach leaves the custom stage registered in StageCache",
            "after": "the same detach is followed by exactly one StageCache.Erase(stage)",
            "stage_python_reference": "retained; object destruction is not claimed",
            "everything_else": "frozen D375 workload and installed shutdown path",
        },
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "status_short": _git("status", "--short", "--untracked-files=all"),
            "clean_claim": False,
            "note": "approved uncommitted D376 plus forward-only D377 files are present",
        },
        "installed_stack": {
            "isaac_sim": _package_version("isaacsim"),
            "isaac_lab": _package_version("isaaclab"),
            "simulation_app_extension": "2.12.2",
            "kit": "107.3.3",
            "omni_physx": "107.3.26",
            "gpu": gpu,
        },
        "official_sources": OFFICIAL_SOURCES,
        "official_vs_inference": {
            "installed_source_fact": "PhysX release_memory_stage performs detach_stage then cache.Erase(stage)",
            "official_semantic_limit": "Erase removes cache membership; retained stage references may keep the object alive",
            "project_hypothesis": "D375's retained cache membership may be a conditional trigger for terminal non-exit",
            "not_claimed": [
                "StageCache omission is necessary or sufficient in general",
                "D375 is exact NVIDIA bug 5948099",
                "all USD stage references are destroyed",
            ],
        },
        "inputs": inputs,
        "source_hashes": _source_hashes(),
        "worker_transform_attestation": worker_attestation,
        "frozen_hash_checks": frozen_hash_checks,
        "d373_asset_file_hashes": registered_assets,
        "d373_asset_inventory_before": actual_assets,
        "d376_evidence_inventory_before": _inventory(D376_DIR),
        "d334_sidecar_before": _inventory(D334_SIDECAR),
        "preexisting_isaac_rows": active_isaac,
        "preexisting_gpu_compute_rows": _gpu_compute_rows(),
        "registered_execution": {
            "actual_worker": 1,
            "automatic_retry": 0,
            "start_new_session_process_group": True,
            "bounded_watchdog_s": WORKER_TIMEOUT_S,
            "sigterm_grace_s": TERM_GRACE_S,
            "sigkill_wait_s": KILL_WAIT_S,
            "simulation_app_launches": 1,
            "physx_attach_detach": [1, 1],
            "stagecache_erase_calls": 1,
            "callback_requests_for_workload_reproduction": 34,
            "property_queries_for_workload_reproduction": 2,
            "rerun_save_only": 1,
            "rerun_viewer_max": 1,
        },
        "registered_phase_order": [
            "simulation_app_launch_start",
            "simulation_app_launch_end",
            "worker_execution_start",
            "noninstance_inspection_stage_create_start",
            "noninstance_owner_and_inventory_gate_end",
            "physx_stage_attach_start",
            "physx_stage_attach_end",
            "callback_progress x34",
            "property_query_start/end x2",
            "worker_cleanup_start",
            "physx_stage_detach_start",
            "physx_stage_detach_end",
            "stagecache_erase_before",
            "stagecache_erase_call_start",
            "stagecache_erase_call_end",
            "stagecache_erase_after",
            "durable raw and preclose",
            "worker_cleanup_end",
            "simulation_app_close_start",
            "external worker_process_exit_observed",
        ],
        "registered_pass_formula": (
            "D375 workload selected-canonical equivalence AND exact erase pre/post/counter AND "
            "raw-preclose-prefix hash authority AND close_start AND external return0/no timeout/no signal/"
            "reaped/no process-group-or-worker-GPU residue"
        ),
        "strict_zero_scope": [
            "physics_steps",
            "public_forwards",
            "q5_commands",
            "q5_samples",
            "contact_queries",
            "cylinder_creates_or_writes",
            "target_ik_path_pose_changes",
            "material_mass_actuator_physics_setting_changes",
            "automatic_convex_decomposition_sweeps",
            "simulation_app_update_pumps",
        ],
        "result_branches": {
            "clean_exit": "conditional trigger support in this one run only",
            "upstream_workload_mismatch": "erase effect null",
            "erase_contract_failure": "erase effect null",
            "post_erase_terminal_nonexit": "erase alone insufficient in this run; native cause null",
            "terminal_nonzero_or_residue": "erase causality null",
        },
        "prepare_negative_controls": negatives,
        "checks": checks,
        "pass": all(checks.values()),
        "promotion_boundary": (
            "No full P34 identity, physics, q5, contact, cylinder, tipping, or grasp promotion. "
            "g0a_pass remains false; repaired full identity and physics need separate approval."
        ),
    }
    _write_json_x(PREREG_PATH, prereg)
    _phase("preregistration_frozen", prereg_sha256=_sha(PREREG_PATH), passed=prereg["pass"])
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"], "checks": checks}, sort_keys=True))
    return 0 if prereg["pass"] else 1


def _phase_rows() -> list[dict[str, Any]]:
    if not PHASE_PATH.is_file():
        return []
    return [json.loads(line) for line in PHASE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prefix_hash_check(preclose: dict[str, Any]) -> bool:
    byte_count = preclose.get("phase_prefix_bytes")
    expected = preclose.get("phase_prefix_sha256")
    if not isinstance(byte_count, int) or byte_count <= 0 or not isinstance(expected, str):
        return False
    payload = PHASE_PATH.read_bytes()
    return len(payload) >= byte_count and _sha_bytes(payload[:byte_count]) == expected


def _process_group_residue(pgid: int) -> list[dict[str, Any]]:
    return [row for row in _process_snapshot() if row["pgid"] == pgid]


def run_worker() -> int:
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D377 preregistration did not pass")
    for path in (INVOCATION_PATH, SUPERVISOR_PATH, CLAIM_PATH, RAW_PATH, PRECLOSE_PATH):
        if path.exists():
            raise RuntimeError(f"D377 one-shot path already claimed: {_rel(path)}")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D377 source changed after preregistration")
    if _active_isaac_rows(_process_snapshot()):
        raise RuntimeError("a pre-existing Isaac/Kit worker appeared after preregistration")
    command = [
        str(ISAAC_PYTHON),
        "-B",
        str(WORKER),
        "--out-dir",
        str(OUT_DIR),
        "--prereg",
        str(PREREG_PATH),
        "--headless",
    ]
    invocation = {
        "artifact": "D377_SINGLE_ACTUAL_WORKER_INVOCATION_V1",
        "command": command,
        "cwd": str(REPO),
        "worker_sha256": _sha(WORKER),
        "controller_sha256": _sha(HARNESS),
        "preregistration_sha256": _sha(PREREG_PATH),
        "actual_worker_count": 1,
        "automatic_retry_count": 0,
        "start_new_session": True,
        "bounded_watchdog_seconds": WORKER_TIMEOUT_S,
        "term_grace_seconds": TERM_GRACE_S,
        "kill_wait_seconds": KILL_WAIT_S,
        "environment_overrides": {"OMNI_KIT_ACCEPT_EULA": "YES"},
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase("supervisor_spawn_start", invocation_sha256=_sha(INVOCATION_PATH))
    env = os.environ.copy()
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    start = time.monotonic()
    with STDOUT_PATH.open("x", encoding="utf-8") as stdout, STDERR_PATH.open(
        "x", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        pgid = os.getpgid(process.pid)
        timed_out = False
        timeout_phase = None
        sigterm_sent = False
        sigkill_sent = False
        kill_wait_expired = False
        close_start_first_seen_s = None
        while process.poll() is None:
            elapsed = time.monotonic() - start
            names = [row["phase"] for row in _phase_rows()]
            if close_start_first_seen_s is None and "simulation_app_close_start" in names:
                close_start_first_seen_s = elapsed
            if elapsed >= WORKER_TIMEOUT_S:
                timed_out = True
                timeout_phase = names[-1] if names else None
                os.killpg(pgid, signal.SIGTERM)
                sigterm_sent = True
                try:
                    process.wait(timeout=TERM_GRACE_S)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    sigkill_sent = True
                    try:
                        process.wait(timeout=KILL_WAIT_S)
                    except subprocess.TimeoutExpired:
                        kill_wait_expired = True
                break
            time.sleep(0.1)
        returncode = process.poll()
        if returncode is None and not kill_wait_expired:
            returncode = process.wait()
        stdout.flush(); os.fsync(stdout.fileno())
        stderr.flush(); os.fsync(stderr.fileno())
    elapsed_s = time.monotonic() - start
    time.sleep(1.0)
    group_residue = _process_group_residue(pgid)
    gpu_rows_after = _gpu_compute_rows()
    worker_gpu_residue = [row for row in gpu_rows_after if row["pid"] == process.pid]
    required = {
        "claim": CLAIM_PATH.is_file(),
        "raw_summary": RAW_PATH.is_file(),
        "preclose": PRECLOSE_PATH.is_file(),
        "phase_markers": PHASE_PATH.is_file(),
    }
    raw = _read_json(RAW_PATH) if required["raw_summary"] else {}
    preclose = _read_json(PRECLOSE_PATH) if required["preclose"] else {}
    phase_names = [row["phase"] for row in _phase_rows()]
    hash_checks = {
        "raw_worker_protocol_pass": raw.get("worker_protocol_pass") is True,
        "preclose_worker_protocol_pass": preclose.get("worker_protocol_pass") is True,
        "preclose_summary_path_exact": preclose.get("summary_path") == _rel(RAW_PATH),
        "preclose_summary_sha_exact": required["raw_summary"]
        and preclose.get("summary_sha256") == _sha(RAW_PATH),
        "preclose_counters_exact": preclose.get("counters") == raw.get("counters"),
        "preclose_timeline_exact": preclose.get("timeline_after") == raw.get("timeline_after"),
        "preclose_erase_exact": preclose.get("stagecache_erase") == raw.get("stagecache_erase"),
        "preclose_phase_prefix_sha_exact": required["phase_markers"]
        and _prefix_hash_check(preclose),
        "safe_to_close_app": preclose.get("safe_to_close_app") is True,
        "close_start_marker_present": "simulation_app_close_start" in phase_names,
    }
    operational_checks = {
        "returncode_zero": returncode == 0,
        "no_watchdog_timeout": not timed_out,
        "no_sigterm": not sigterm_sent,
        "no_sigkill": not sigkill_sent,
        "kill_wait_did_not_expire": not kill_wait_expired,
        "process_reaped": process.poll() is not None,
        "no_process_group_residue": not group_residue,
        "no_worker_gpu_pid_residue": not worker_gpu_residue,
        "all_required_artifacts": all(required.values()),
    }
    erase_record = raw.get("stagecache_erase", {})
    erase_exact = bool(
        raw.get("counters", {}).get("stagecache_erase_calls") == 1
        and erase_record.get("pass") is True
    )
    effective_preanalysis_pass = _logic_gate(
        workload_equivalent=True,
        erase_exact=erase_exact,
        hash_authority=all(hash_checks.values()),
        close_start=hash_checks["close_start_marker_present"],
        returncode=returncode,
        timed_out=timed_out,
        signaled=sigterm_sent or sigkill_sent,
        reaped=operational_checks["process_reaped"],
        residue=bool(group_residue or worker_gpu_residue),
    ) and operational_checks["all_required_artifacts"]
    supervisor = {
        "artifact": "D377_HASH_BOUND_PROCESS_GROUP_SUPERVISOR_V1",
        "pid": process.pid,
        "pgid": pgid,
        "returncode": returncode,
        "elapsed_s": elapsed_s,
        "watchdog_s": WORKER_TIMEOUT_S,
        "timed_out": timed_out,
        "timeout_last_phase": timeout_phase,
        "close_start_first_seen_s": close_start_first_seen_s,
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "kill_wait_expired": kill_wait_expired,
        "worker_spawn_count": 1,
        "automatic_retry_count": 0,
        "required_artifacts": required,
        "hash_authority_checks": hash_checks,
        "hash_authority_pass": all(hash_checks.values()),
        "operational_checks": operational_checks,
        "operational_pass": all(operational_checks.values()),
        "erase_exact_preanalysis": erase_exact,
        "process_group_residue": group_residue,
        "gpu_compute_rows_after": gpu_rows_after,
        "worker_gpu_residue": worker_gpu_residue,
        "stdout": {"path": _rel(STDOUT_PATH), "sha256": _sha(STDOUT_PATH)},
        "stderr": {"path": _rel(STDERR_PATH), "sha256": _sha(STDERR_PATH)},
        "effective_preanalysis_pass": effective_preanalysis_pass,
        "pass": effective_preanalysis_pass,
        "note": "D375 workload equivalence is independently evaluated only in analyze",
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase(
        "worker_process_exit_observed",
        returncode=returncode,
        elapsed_s=elapsed_s,
        timed_out=timed_out,
        sigterm_sent=sigterm_sent,
        sigkill_sent=sigkill_sent,
        process_group_residue_count=len(group_residue),
        worker_gpu_residue_count=len(worker_gpu_residue),
    )
    print(
        json.dumps(
            {
                "stage": "run",
                "pass": effective_preanalysis_pass,
                "returncode": returncode,
                "elapsed_s": elapsed_s,
                "timed_out": timed_out,
            },
            sort_keys=True,
        )
    )
    return 0 if effective_preanalysis_pass else 1


def _without(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def _workload_signature(raw: dict[str, Any]) -> dict[str, Any]:
    properties = {}
    for body, value in sorted(raw.get("property_queries", {}).items()):
        properties[body] = {
            "body": value.get("body"),
            "body_path": value.get("body_path"),
            "expected_collider_count_including_disabled_legacy": value.get(
                "expected_collider_count_including_disabled_legacy"
            ),
            "finished": value.get("finished"),
            "pass": value.get("pass"),
            "errors": value.get("errors"),
            "simulation_app_update_pumps": value.get("simulation_app_update_pumps"),
            "rigid_body": _without(value.get("rigid_body", {}), "path_id"),
            "colliders": [
                _without(row, "path_id") for row in value.get("colliders", [])
            ],
        }
    callbacks = []
    for row in raw.get("callback_rows", []):
        callback = row.get("callback", {})
        callbacks.append(
            {
                "body": row.get("body"),
                "name": row.get("name"),
                "role": row.get("role"),
                "prim_name": row.get("prim_name"),
                "live_path": row.get("live_path"),
                "authored_f32_topology_payload_sha256": row.get(
                    "authored_f32_topology_payload_sha256"
                ),
                "protocol_pass": row.get("protocol_pass"),
                "callback": _without(callback, "witness_path"),
            }
        )
    common_counter_keys = (
        "worker_invocations",
        "automatic_retries",
        "simulation_app_launches",
        "derivative_asset_materializations",
        "usd_stage_file_writes",
        "physx_stage_attaches",
        "physx_stage_detaches",
        "physx_property_queries",
        "physx_callback_requests",
        "simulation_context_constructions",
        "resets",
        "timeline_play_requests",
        "timeline_commit_requests",
        "physics_steps",
        "public_forwards",
        "q5_commands",
        "q5_samples",
        "contact_queries",
        "cylinder_creates_or_writes",
        "target_ik_path_pose_changes",
        "automatic_convex_decomposition_sweeps",
        "material_mass_actuator_physics_setting_changes",
        "isaac_hydra_renders",
        "simulation_app_update_pumps",
        "d343_contract_retests",
    )
    counters = raw.get("counters", {})
    signature = {
        "asset_reuse": raw.get("asset_reuse"),
        "authored_readback": raw.get("authored_readback"),
        "callback_rows": callbacks,
        "canonical_outside_collision_subtree_diff": raw.get(
            "canonical_outside_collision_subtree_diff"
        ),
        "common_counters": {key: counters.get(key) for key in common_counter_keys},
        "live_inventory": raw.get("live_inventory"),
        "mass_api_base_vs_derivative": raw.get("mass_api_base_vs_derivative"),
        "mass_api_inspection_stage": raw.get("mass_api_inspection_stage"),
        "owner_structure": raw.get("owner_structure"),
        "physx_stage_attach_return": raw.get("physx_stage_attach_return"),
        "property_queries": properties,
        "timeline_before": raw.get("timeline_before"),
        "timeline_after": raw.get("timeline_after"),
    }
    return {"payload": signature, "sha256": _canonical_sha(signature)}


def _phase_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = [row["phase"] for row in rows]
    required = [
        "simulation_app_launch_start",
        "simulation_app_launch_end",
        "worker_execution_start",
        "noninstance_inspection_stage_create_start",
        "noninstance_owner_and_inventory_gate_end",
        "physx_stage_attach_start",
        "physx_stage_attach_end",
        "worker_cleanup_start",
        "physx_stage_detach_start",
        "physx_stage_detach_end",
        "stagecache_erase_before",
        "stagecache_erase_call_start",
        "stagecache_erase_call_end",
        "stagecache_erase_after",
        "worker_cleanup_end",
        "simulation_app_close_start",
        "worker_process_exit_observed",
    ]
    indices = {}
    cursor = -1
    ordered = True
    for name in required:
        try:
            cursor = names.index(name, cursor + 1)
            indices[name] = cursor
        except ValueError:
            indices[name] = None
            ordered = False
    checks = {
        "required_subsequence_order": ordered,
        "callback_progress_exactly_34": names.count("callback_progress") == 34,
        "property_query_start_exactly_2": names.count("property_query_start") == 2,
        "property_query_end_exactly_2": names.count("property_query_end") == 2,
        "erase_before_exactly_1": names.count("stagecache_erase_before") == 1,
        "erase_call_start_exactly_1": names.count("stagecache_erase_call_start") == 1,
        "erase_call_end_exactly_1": names.count("stagecache_erase_call_end") == 1,
        "erase_after_exactly_1": names.count("stagecache_erase_after") == 1,
        "close_start_exactly_1": names.count("simulation_app_close_start") == 1,
        "close_return_marker_optional_max_1": names.count(
            "simulation_app_close_returned_optional"
        ) <= 1,
    }
    return {
        "phase_count": len(rows),
        "names": names,
        "required_indices": indices,
        "close_returned_optional_observed": "simulation_app_close_returned_optional" in names,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _erase_audit(raw: dict[str, Any]) -> dict[str, Any]:
    erase = raw.get("stagecache_erase", {})
    counters = raw.get("counters", {})
    before = erase.get("before", {})
    after = erase.get("after", {})
    checks = {
        "call_counter_exactly_one": counters.get("stagecache_erase_calls") == 1,
        "placement_exact": erase.get("placement")
        == "immediately_after_successful_physx_detach",
        "before_contains_true": before.get("contains_stage") is True,
        "before_id_valid": before.get("id_valid") is True,
        "before_id_matches_registered": before.get("id_matches_registered_stage_id") is True,
        "before_find_matches_stage": before.get("find_old_id_matches_stage") is True,
        "erase_return_true": erase.get("erase_return") is True,
        "after_contains_false": after.get("contains_stage") is False,
        "after_id_invalid": after.get("id_valid") is False,
        "after_find_old_id_absent": after.get("find_old_id_present") is False,
        "python_stage_reference_retained": erase.get("python_stage_reference_retained") is True,
        "worker_erase_pass": erase.get("pass") is True,
    }
    return {"record": erase, "checks": checks, "pass": all(checks.values())}


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        size = list(image.size)
        mode = image.mode
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "dimensions": size,
        "mode": mode,
        "exact_1920x1080": size == [1920, 1080],
    }


def _render_board(evidence: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch

    regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#F7F8FA")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    passed = evidence["lifecycle_localization_pass"]
    result_color = "#147D5B" if passed else "#B42318"
    result_fill = "#E5F5EC" if passed else "#FDE8E8"
    result_title = "정상 종료 관찰" if passed else "종료 복구 미확인"
    supervisor = evidence["supervisor"]
    erase = evidence["erase_audit"]
    erase_record = erase.get("record", {})
    before_contains = erase_record.get("before", {}).get("contains_stage")
    after_contains = erase_record.get("after", {}).get("contains_stage")
    erase_return = erase_record.get("erase_return")
    fig.text(0.5, 0.955, "D377 · D375 종료 멈춤의 StageCache 조건을 한 변수로 확인", ha="center", va="center", fontproperties=bold, fontsize=23, color="#14213D")
    fig.text(0.5, 0.918, "같은 P34 읽기 작업 + PhysX detach 뒤 StageCache.Erase(stage) 정확히 1회 · 물리/q5/접촉 0", ha="center", va="center", fontproperties=regular, fontsize=12.5, color="#4B5563")

    lanes = [
        (0.66, "D375 기준선", "측정 완료 → detach 완료 → Erase 없음 → close/framework release 뒤 920.391초 강제 종료", "#FDE8E8", "#B42318"),
        (0.43, "D377 단일변수", f"동일 workload={evidence['workload_equivalence']['pass']} → Erase 전/후={erase['pass']} → return={supervisor['returncode']} · {supervisor['elapsed_s']:.3f}초", result_fill, result_color),
    ]
    for y, title, text_value, fill, edge in lanes:
        ax.add_patch(FancyBboxPatch((0.055, y), 0.89, 0.15, boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=1.6, edgecolor=edge, facecolor=fill, transform=ax.transAxes))
        fig.text(0.078, y + 0.105, title, ha="left", va="center", fontproperties=bold, fontsize=15, color=edge)
        fig.text(0.078, y + 0.05, text_value, ha="left", va="center", fontproperties=regular, fontsize=11.5, color="#1F2937")

    cards = [
        (0.055, 0.13, 0.275, 0.21, "실제로 바꾼 것", "PhysX detach는 그대로 유지\n그 직후 cache.Erase(stage) 1회\nPython stage 참조는 그대로 유지\n종료 API·fastShutdown 변경 없음", "#E8F1FB", "#1769AA"),
        (0.363, 0.13, 0.275, 0.21, "이번 실행에서 확인한 것", f"callback={evidence['observed_counts']['callbacks']} · property={evidence['observed_counts']['property_queries']}\nErase 전 Contains={before_contains}\nErase 반환={erase_return} · 이후 Contains={after_contains}\ntimeout={supervisor['timed_out']} · signal={supervisor['sigterm_sent'] or supervisor['sigkill_sent']}", "#FFF4D8", "#B7791F"),
        (0.671, 0.13, 0.275, 0.21, "해석 한계", "캐시 등록 제거만 확인\nstage 객체 전체 파괴는 아님\nNVIDIA 5948099 동일성은 미확정\nP34 identity·물리·파지는 여전히 null", "#F3ECFA", "#71429B"),
    ]
    for x, y, w, h, title, body, fill, edge in cards:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.016", linewidth=1.3, edgecolor=edge, facecolor=fill, transform=ax.transAxes))
        fig.text(x + 0.017, y + h - 0.045, title, ha="left", va="center", fontproperties=bold, fontsize=13.5, color=edge)
        fig.text(x + 0.017, y + h - 0.092, body, ha="left", va="top", fontproperties=regular, fontsize=10.7, color="#1F2937", linespacing=1.42)
    fig.text(0.5, 0.07, f"판정: {result_title} · {evidence['causal_interpretation']} · g0a_pass=false", ha="center", va="center", fontproperties=bold, fontsize=13, color=result_color)
    fig.savefig(BOARD_PATH, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    info = _png_info(BOARD_PATH)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"D377 board dimension failure: {info}")
    return info


def _write_rerun(evidence: dict[str, Any]) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.viz_debug import log_rerun

    rows = _phase_rows()
    launch_rows = [row for row in rows if row["phase"] == "simulation_app_launch_start"]
    base_ns = launch_rows[0]["monotonic_ns"] if launch_rows else rows[0]["monotonic_ns"]

    def elapsed_s(row: dict[str, Any]) -> float:
        return max(0.0, (row["monotonic_ns"] - base_ns) / 1.0e9)

    before_rows = [row for row in rows if row["phase"] == "stagecache_erase_before"]
    call_end_rows = [row for row in rows if row["phase"] == "stagecache_erase_call_end"]
    after_rows = [row for row in rows if row["phase"] == "stagecache_erase_after"]
    exit_rows = [row for row in rows if row["phase"] == "worker_process_exit_observed"]
    final_elapsed = elapsed_s(exit_rows[-1]) if exit_rows else float(evidence["supervisor"]["elapsed_s"])
    scalars = [
        {"entity_path": "metrics/d377/stagecache_contains__1_true_0_false_minus1_unknown", "value": -1.0, "duration": {"lifecycle_elapsed_s": 0.0}},
        {"entity_path": "metrics/d377/erase_return__1_true_0_false_minus1_unknown", "value": -1.0, "duration": {"lifecycle_elapsed_s": 0.0}},
        {"entity_path": "metrics/d377/process_terminal_state__1_clean_0_running_minus1_forced_or_fail", "value": 0.0, "duration": {"lifecycle_elapsed_s": 0.0}},
    ]
    if before_rows:
        row = before_rows[-1]
        scalars.append({"entity_path": "metrics/d377/stagecache_contains__1_true_0_false_minus1_unknown", "value": 1.0 if row.get("contains_stage") is True else 0.0, "duration": {"lifecycle_elapsed_s": elapsed_s(row)}})
    if call_end_rows:
        row = call_end_rows[-1]
        scalars.append({"entity_path": "metrics/d377/erase_return__1_true_0_false_minus1_unknown", "value": 1.0 if row.get("erase_return") is True else 0.0, "duration": {"lifecycle_elapsed_s": elapsed_s(row)}})
    if after_rows:
        row = after_rows[-1]
        scalars.append({"entity_path": "metrics/d377/stagecache_contains__1_true_0_false_minus1_unknown", "value": 1.0 if row.get("contains_stage") is True else 0.0, "duration": {"lifecycle_elapsed_s": elapsed_s(row)}})
    scalars.append({"entity_path": "metrics/d377/process_terminal_state__1_clean_0_running_minus1_forced_or_fail", "value": 1.0 if evidence["lifecycle_localization_pass"] else -1.0, "duration": {"lifecycle_elapsed_s": final_elapsed}})

    critical_rows = []
    for row in rows:
        phase = row["phase"]
        if phase in {
            "simulation_app_launch_start",
            "simulation_app_launch_end",
            "physx_stage_attach_end",
            "physx_stage_detach_end",
            "stagecache_erase_before",
            "stagecache_erase_after",
            "simulation_app_close_start",
            "worker_process_exit_observed",
        }:
            critical_rows.append(row)
        elif phase == "callback_progress" and row.get("part_progress") == 34:
            critical_rows.append(row)
        elif phase == "property_query_end" and row.get("body") == "gripper_link":
            critical_rows.append(row)
    events = []
    for row in critical_rows:
        detail = ""
        if row["phase"] == "callback_progress":
            detail = f" {row.get('part_progress')}/{row.get('part_total')}"
        elif row["phase"] == "property_query_end":
            detail = f" both bodies done; final={row.get('body')} colliders={row.get('collider_count')}"
        elif row["phase"] in {"stagecache_erase_before", "stagecache_erase_after"}:
            detail = f" Contains={row.get('contains_stage')} id_valid={row.get('id_valid')}"
        elif row["phase"] == "worker_process_exit_observed":
            detail = f" return={row.get('returncode')} timeout={row.get('timed_out')}"
        events.append({"entity_path": "events/d377/timeline", "text": row["phase"] + detail, "level": "INFO", "duration": {"lifecycle_elapsed_s": elapsed_s(row)}})
    events.extend(
        [
            {"entity_path": "events/d377/verdict", "text": evidence["verdict"], "level": "INFO" if evidence["lifecycle_localization_pass"] else "WARN", "static": True},
            {"entity_path": "events/d377/boundary", "text": "D375 baseline: StageCache Erase absent; framework-release reached at 5.523s; forced stop at 920.391s. " + evidence["causal_interpretation"], "level": "WARN", "static": True},
            {"entity_path": "events/d377/scope", "text": "Lifecycle-only: physics/q5/contact/cylinder/target-IK-path=0; g0a_pass=false.", "level": "WARN", "static": True},
        ]
    )
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    try:
        saved = log_rerun(
            RRD_PATH,
            scalar_trace=scalars,
            events=events,
            recording_metadata={
                "case": "g0a_d377",
                "verdict": evidence["verdict"],
                "source": "D375 baseline plus one actual D377 lifecycle worker",
                "scientific_authority": "raw/preclose/phase/supervisor JSON and process return",
                "physics_steps": 0,
                "q5_samples": 0,
                "contact_queries": 0,
                "g0a_pass": False,
                "display_role": "inspection only",
            },
            recording_id="g0a_d377_stagecache_erase_localization",
            blueprint_path=RBL_PATH,
            blueprint_mode="d377_stagecache_erase_localization",
            live_viewer=False,
            app_id="roarm_g0a_d377_stagecache_erase",
        )
    finally:
        os.environ["PATH"] = old_path
    if saved.get("ok") is not True:
        raise RuntimeError(f"D377 save-only Rerun failed: {saved}")
    entities = {
        "metadata/run",
        "metrics/d377/stagecache_contains__1_true_0_false_minus1_unknown",
        "metrics/d377/erase_return__1_true_0_false_minus1_unknown",
        "metrics/d377/process_terminal_state__1_clean_0_running_minus1_forced_or_fail",
        "events/d377/timeline",
        "events/d377/verdict",
        "events/d377/boundary",
        "events/d377/scope",
    }
    components = {
        "metadata/run": ["TextDocument:text"],
        "metrics/d377/stagecache_contains__1_true_0_false_minus1_unknown": ["Scalars:scalars"],
        "metrics/d377/erase_return__1_true_0_false_minus1_unknown": ["Scalars:scalars"],
        "metrics/d377/process_terminal_state__1_clean_0_running_minus1_forced_or_fail": ["Scalars:scalars"],
        "events/d377/timeline": ["TextLog:text", "TextLog:level"],
        "events/d377/verdict": ["TextLog:text", "TextLog:level"],
        "events/d377/boundary": ["TextLog:text", "TextLog:level"],
        "events/d377/scope": ["TextLog:text", "TextLog:level"],
    }
    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(entities),
        exact_entity_paths=sorted(entities),
        expected_timeline_names=["blueprint", "lifecycle_elapsed_s", "log_time"],
        exact_timeline_names=["blueprint", "lifecycle_elapsed_s", "log_time"],
        expected_entity_components=components,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG_PATH,
        screenshot_window_size="1920x1080",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=240.0,
    )
    _write_json_x(RERUN_VALIDATION_PATH, validation)
    headless = validation.get("headless_render") or {}
    return {
        "save_only": saved,
        "strict_validation_pass": validation.get("pass") is True,
        "rrd": {"path": _rel(RRD_PATH), "bytes": RRD_PATH.stat().st_size, "sha256": _sha(RRD_PATH)},
        "rbl": {"path": _rel(RBL_PATH), "bytes": RBL_PATH.stat().st_size, "sha256": _sha(RBL_PATH)},
        "viewer_invocations": 1 if headless.get("attempted") is True else 0,
        "viewer_returncode": headless.get("returncode"),
        "requested_logical_window_size": "1920x1080",
        "screenshot": _png_info(RERUN_PNG_PATH) if RERUN_PNG_PATH.is_file() else {"path": _rel(RERUN_PNG_PATH), "exists": False},
    }


def analyze() -> int:
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D377 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D377 source changed after preregistration")
    if EVIDENCE_PATH.exists():
        raise RuntimeError("D377 analyze is forward-only and already claimed")
    supervisor = _read_json(SUPERVISOR_PATH)
    raw = _read_json(RAW_PATH) if RAW_PATH.is_file() else {}
    preclose = _read_json(PRECLOSE_PATH) if PRECLOSE_PATH.is_file() else {}
    d375_raw = _read_json(D375_RAW)
    baseline_signature = _workload_signature(d375_raw)
    current_signature = _workload_signature(raw)
    workload_checks = {
        "selected_canonical_sha_exact": current_signature["sha256"] == baseline_signature["sha256"],
        "worker_protocol_pass": raw.get("worker_protocol_pass") is True,
        "callback_count_34": raw.get("counters", {}).get("physx_callback_requests") == 34,
        "property_query_count_2": raw.get("counters", {}).get("physx_property_queries") == 2,
        "property_collider_counts_17_19": {
            body: len(raw.get("property_queries", {}).get(body, {}).get("colliders", []))
            for body in ("link5", "gripper_link")
        } == {"link5": 17, "gripper_link": 19},
        "properties_valid": all(
            value.get("pass") is True
            for value in raw.get("property_queries", {}).values()
        ) and len(raw.get("property_queries", {})) == 2,
        "timeline_stop_time_zero_exact": raw.get("timeline_before") == raw.get("timeline_after")
        and raw.get("timeline_after", {}).get("is_stopped") is True
        and raw.get("timeline_after", {}).get("current_time_s") == 0.0,
        "asset_immutable": raw.get("asset_reuse", {}).get("immutable_before_after") is True,
    }
    workload = {
        "baseline_selected_canonical_sha256": baseline_signature["sha256"],
        "d377_selected_canonical_sha256": current_signature["sha256"],
        "differing_top_level_fields": sorted(
            key
            for key in set(baseline_signature["payload"]) | set(current_signature["payload"])
            if baseline_signature["payload"].get(key) != current_signature["payload"].get(key)
        ),
        "checks": workload_checks,
        "pass": all(workload_checks.values()),
        "role": "termination-workload equivalence only; no geometry identity promotion",
    }
    erase = _erase_audit(raw)
    phase = _phase_audit(_phase_rows())
    hash_authority = bool(
        supervisor.get("hash_authority_pass") is True
        and preclose.get("summary_sha256") == (_sha(RAW_PATH) if RAW_PATH.is_file() else None)
        and preclose.get("stagecache_erase") == raw.get("stagecache_erase")
        and _prefix_hash_check(preclose)
    )
    external_clean = bool(
        supervisor.get("returncode") == 0
        and supervisor.get("timed_out") is False
        and supervisor.get("sigterm_sent") is False
        and supervisor.get("sigkill_sent") is False
        and supervisor.get("operational_checks", {}).get("process_reaped") is True
        and not supervisor.get("process_group_residue")
        and not supervisor.get("worker_gpu_residue")
    )
    registered_input_checks = {}
    for registered_path, expected_sha in prereg["inputs"].items():
        path = Path(registered_path)
        if not path.is_absolute():
            path = REPO / path
        registered_input_checks[registered_path] = bool(
            path.is_file() and _sha(path) == expected_sha
        )
    frozen_inventory_checks = {
        "registered_inputs_exist_and_hash_exact": all(registered_input_checks.values()),
        "D375_and_installed_fixed_hashes_exact": all(
            path.is_file() and _sha(path) == expected
            for path, expected in EXPECTED_FIXED_HASHES.items()
        ),
        "D373_asset_unchanged": _inventory(D373_ASSET_DIR)
        == prereg["d373_asset_inventory_before"],
        "D376_evidence_unchanged": _inventory(D376_DIR)
        == prereg["d376_evidence_inventory_before"],
        "D334_sidecar_unchanged": _inventory(D334_SIDECAR)
        == prereg["d334_sidecar_before"],
    }
    frozen_inventory_pass = all(frozen_inventory_checks.values())
    zero_keys = prereg["strict_zero_scope"]
    zero_scope = {
        key: raw.get("counters", {}).get(key) == 0 for key in zero_keys
    }
    zero_scope_pass = all(zero_scope.values())
    lifecycle_pass = _logic_gate(
        workload_equivalent=workload["pass"],
        erase_exact=erase["pass"],
        hash_authority=hash_authority,
        close_start=phase["checks"]["close_start_exactly_1"],
        returncode=supervisor.get("returncode"),
        timed_out=supervisor.get("timed_out") is True,
        signaled=supervisor.get("sigterm_sent") is True or supervisor.get("sigkill_sent") is True,
        reaped=supervisor.get("operational_checks", {}).get("process_reaped") is True,
        residue=bool(supervisor.get("process_group_residue") or supervisor.get("worker_gpu_residue")),
    ) and bool(
        phase["pass"]
        and supervisor.get("pass") is True
        and supervisor.get("operational_checks", {}).get("all_required_artifacts") is True
        and frozen_inventory_pass
        and zero_scope_pass
    )
    if not frozen_inventory_pass:
        branch = "FROZEN_INPUT_OR_SIDECAR_DRIFT_ERASE_EFFECT_NULL"
        causal = "동결 입력 또는 sidecar가 달라져 Erase 효과는 판정할 수 없음(null)"
    elif not workload["pass"]:
        branch = "UPSTREAM_WORKLOAD_MISMATCH_ERASE_EFFECT_NULL"
        causal = "D375 workload가 같지 않아 Erase 효과는 판정할 수 없음(null)"
    elif not erase["pass"]:
        branch = "ERASE_CONTRACT_FAIL_EFFECT_NULL"
        causal = "Erase 전후 계약이 성립하지 않아 효과는 판정할 수 없음(null)"
    elif not zero_scope_pass:
        branch = "FORBIDDEN_SCOPE_COUNTER_NONZERO_ERASE_EFFECT_NULL"
        causal = "금지된 실행 범위가 0이 아니어서 Erase 효과는 판정할 수 없음(null)"
    elif supervisor.get("timed_out") is True:
        branch = "POST_ERASE_TERMINAL_NONEXIT_ERASE_ALONE_INSUFFICIENT"
        causal = "이 실행에서는 Erase 1회만으로 종료 문제를 해결하지 못함"
    elif (
        not external_clean
        or not hash_authority
        or not phase["pass"]
        or supervisor.get("pass") is not True
        or supervisor.get("operational_checks", {}).get("all_required_artifacts") is not True
    ):
        branch = "TERMINAL_NONZERO_OR_RESIDUE_ERASE_CAUSALITY_NULL"
        causal = "외부 종료 권위가 깨져 Erase 인과는 판정할 수 없음(null)"
    else:
        branch = "STAGECACHE_ERASE_PATH_CLEAN_EXIT_SUPPORTED_THIS_RUN"
        causal = "이 1회에서는 StageCache 등록 제거를 포함한 경로가 정상 종료됨"
    evidence = {
        "artifact": "D377_STAGECACHE_ERASE_BEFORE_CLOSE_LOCALIZATION_EVIDENCE_V1",
        "case": "g0a_d377",
        "attempt": ATTEMPT,
        "what_and_why": prereg["what_and_why"],
        "new_variables": NEW_VARIABLES,
        "worker_and_retry": [1, 0],
        "observed_counts": {
            "callbacks": raw.get("counters", {}).get("physx_callback_requests"),
            "property_queries": raw.get("counters", {}).get("physx_property_queries"),
            "stagecache_erase_calls": raw.get("counters", {}).get("stagecache_erase_calls"),
        },
        "workload_equivalence": workload,
        "erase_audit": erase,
        "phase_audit": phase,
        "hash_authority_pass": hash_authority,
        "external_clean_exit": external_clean,
        "supervisor": supervisor,
        "zero_scope_checks": zero_scope,
        "lifecycle_localization_pass": lifecycle_pass,
        "result_branch": branch,
        "causal_interpretation": causal,
        "verdict": VERDICT_PASS if lifecycle_pass else VERDICT_FAIL,
        "scientific_and_promotion_boundaries": {
            "exact_nvidia_bug_5948099_identity": None,
            "stage_object_destroyed": None,
            "full_p34_authored_callback_identity": None,
            "physics_equivalence": None,
            "q5_closure": None,
            "contact_or_tipping": None,
            "grasp_feasibility": None,
            "g0a_pass": False,
        },
        "registered_input_hash_checks": registered_input_checks,
        "frozen_inventory_checks": frozen_inventory_checks,
        "frozen_inventory_pass": frozen_inventory_pass,
        "observability_role": "offline display of lifecycle evidence; raw JSON/process return remain authority",
    }
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase("lifecycle_evidence_frozen", verdict=evidence["verdict"], branch=branch)
    board = _render_board(evidence)
    _phase("exact_1920x1080_board_complete", board_sha256=board["sha256"])
    rerun = _write_rerun(evidence)
    _phase("save_only_rerun_and_headless_inspection_complete", strict=rerun["strict_validation_pass"])
    automated_checks = {
        "lifecycle_value_preserved": evidence["lifecycle_localization_pass"] is lifecycle_pass,
        "board_exact_1920x1080": board["exact_1920x1080"],
        "rerun_save_only_ok": rerun["save_only"].get("ok") is True,
        "rerun_strict_validation_pass": rerun["strict_validation_pass"],
        "viewer_exactly_one": rerun["viewer_invocations"] == 1,
        "viewer_return_zero": rerun["viewer_returncode"] == 0,
        "viewer_screenshot_exists": RERUN_PNG_PATH.is_file() and RERUN_PNG_PATH.stat().st_size > 0,
    }
    automated = {
        "artifact": "D377_AUTOMATED_OBSERVABILITY_SUMMARY_V1",
        "evidence_path": _rel(EVIDENCE_PATH),
        "evidence_sha256": _sha(EVIDENCE_PATH),
        "board": board,
        "rerun": rerun,
        "checks": automated_checks,
        "pass": all(automated_checks.values()),
        "note": "manual original-resolution inspection is still required",
    }
    _write_json_x(AUTOMATED_PATH, automated)
    template = {
        "artifact": "D377_MANUAL_VISUAL_INSPECTION_V1",
        "reviewed_original_resolution": False,
        "board": {"path": board["path"], "sha256": board["sha256"], "dimensions": board["dimensions"]},
        "rerun_screenshot": {
            "path": rerun["screenshot"].get("path"),
            "sha256": rerun["screenshot"].get("sha256"),
            "dimensions": rerun["screenshot"].get("dimensions"),
            "requested_logical_window": "1920x1080",
        },
        "observations": [],
        "no_text_overlap_or_clipping": False,
        "d375_vs_d377_difference_readable": False,
        "rerun_timeline_and_boundary_readable": False,
        "pass": False,
    }
    _write_json_x(MANUAL_TEMPLATE_PATH, template)
    print(json.dumps({"stage": "analyze", "lifecycle_pass": lifecycle_pass, "observability_automated_pass": automated["pass"], "verdict": evidence["verdict"]}, sort_keys=True))
    return 0 if automated["pass"] else 1


def finalize() -> int:
    if COMPLETION_PATH.exists():
        raise RuntimeError("D377 completion already exists")
    evidence = _read_json(EVIDENCE_PATH)
    automated = _read_json(AUTOMATED_PATH)
    manual = _read_json(MANUAL_PATH)
    prereg = _read_json(PREREG_PATH)
    expected_board = automated["board"]
    expected_rerun = automated["rerun"]["screenshot"]
    artifact_checks = {
        "source_hashes_still_exact": _source_hashes() == prereg["source_hashes"],
        "evidence_sha_still_exact": _sha(EVIDENCE_PATH) == automated["evidence_sha256"],
        "board_exists_and_sha_exact": BOARD_PATH.is_file()
        and _sha(BOARD_PATH) == expected_board["sha256"],
        "rrd_exists_and_sha_exact": RRD_PATH.is_file()
        and _sha(RRD_PATH) == automated["rerun"]["rrd"]["sha256"],
        "rbl_exists_and_sha_exact": RBL_PATH.is_file()
        and _sha(RBL_PATH) == automated["rerun"]["rbl"]["sha256"],
        "rerun_screenshot_exists_and_sha_exact": RERUN_PNG_PATH.is_file()
        and _sha(RERUN_PNG_PATH) == expected_rerun.get("sha256"),
    }
    manual_checks = {
        "reviewed_original_resolution": manual.get("reviewed_original_resolution") is True,
        "board_path_exact": manual.get("board", {}).get("path") == expected_board["path"],
        "board_sha_exact": manual.get("board", {}).get("sha256") == expected_board["sha256"],
        "board_dimensions_exact": manual.get("board", {}).get("dimensions") == [1920, 1080],
        "rerun_path_exact": manual.get("rerun_screenshot", {}).get("path") == expected_rerun.get("path"),
        "rerun_sha_exact": manual.get("rerun_screenshot", {}).get("sha256") == expected_rerun.get("sha256"),
        "no_text_overlap_or_clipping": manual.get("no_text_overlap_or_clipping") is True,
        "d375_vs_d377_difference_readable": manual.get("d375_vs_d377_difference_readable") is True,
        "rerun_timeline_and_boundary_readable": manual.get("rerun_timeline_and_boundary_readable") is True,
        "observations_present": len(manual.get("observations", [])) >= 2,
        "manual_pass": manual.get("pass") is True,
    }
    visualization_pass = bool(
        automated.get("pass") is True
        and all(artifact_checks.values())
        and all(manual_checks.values())
    )
    completion = {
        "artifact": "D377_COMPLETION_SUMMARY_V1",
        "case": "g0a_d377",
        "attempt": ATTEMPT,
        "verdict": evidence["verdict"],
        "result_branch": evidence["result_branch"],
        "lifecycle_localization_pass": evidence["lifecycle_localization_pass"],
        "causal_interpretation": evidence["causal_interpretation"],
        "automated_observability_pass": automated["pass"],
        "artifact_integrity_checks": artifact_checks,
        "manual_checks": manual_checks,
        "visualization_completion_pass": visualization_pass,
        "overall_case_completion_pass": visualization_pass,
        "worker_and_retry": [1, 0],
        "physics_q5_contact_cylinder": [0, 0, 0, 0],
        "g0a_pass": False,
        "evidence": {"path": _rel(EVIDENCE_PATH), "sha256": _sha(EVIDENCE_PATH)},
        "board": expected_board,
        "rrd": automated["rerun"]["rrd"],
        "rbl": automated["rerun"]["rbl"],
        "manual_inspection": {"path": _rel(MANUAL_PATH), "sha256": _sha(MANUAL_PATH)},
        "next_authorization_boundary": (
            "A repaired full P34 live-identity classifier run remains separately approved work. "
            "No cylinder physics comparison is authorized by D377."
        ),
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase("completion_frozen", pass_=completion["overall_case_completion_pass"])
    print(json.dumps({"stage": "finalize", "pass": completion["overall_case_completion_pass"], "verdict": completion["verdict"]}, sort_keys=True))
    return 0 if completion["overall_case_completion_pass"] else 1


def selftest() -> int:
    worker = _load_worker_module().source_attestation()
    negatives = _negative_controls()
    checks = {
        "worker_transform": worker["pass"] is True,
        "negative_controls": negatives["pass"] is True,
        "one_variable": len(NEW_VARIABLES) == 1,
        "watchdog_bounded": WORKER_TIMEOUT_S == 120.0 and TERM_GRACE_S == 20.0,
    }
    print(json.dumps({"stage": "selftest", "checks": checks, "pass": all(checks.values())}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("selftest", "prepare", "run", "analyze", "finalize"), required=True)
    args = parser.parse_args()
    try:
        return {
            "selftest": selftest,
            "prepare": prepare,
            "run": run_worker,
            "analyze": analyze,
            "finalize": finalize,
        }[args.stage]()
    except Exception as error:
        traceback.print_exc()
        if OUT_DIR.is_dir() and not EXCEPTION_PATH.exists():
            _write_json_x(
                EXCEPTION_PATH,
                {
                    "artifact": "D377_CONTROLLER_EXCEPTION_V1",
                    "stage": args.stage,
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
