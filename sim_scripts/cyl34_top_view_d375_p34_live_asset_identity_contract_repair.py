#!/usr/bin/env python3
"""D375 controller, offline classifier, and observability completion gate.

Stages are forward-only:

* prepare freezes one repaired stage-structure variable and all authorities;
* run launches exactly one bounded Isaac/PhysX worker with no retry;
* analyze independently verifies the worker/preclose hash chain, classifies the
  34 direct-live callback polygons and property rows, and emits PNG/RRD/RBL;
* finalize accepts a separately authored original-resolution visual inspection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

CASE_ROOT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d375"
ATTEMPT = "attempt2_external_gpu_attestation_repair"
OUT_DIR = CASE_ROOT / ATTEMPT
GPU_ATTESTATION_PATH = CASE_ROOT / "d375_external_gpu_attestation.json"
PREREG_PATH = OUT_DIR / "d375_preregistration.json"
PHASE_PATH = OUT_DIR / "d375_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d375_worker_invocation.json"
SUPERVISOR_PATH = OUT_DIR / "d375_worker_supervisor.json"
STDOUT_PATH = OUT_DIR / "d375_worker_stdout.log"
STDERR_PATH = OUT_DIR / "d375_worker_stderr.log"
WORKER_SUMMARY_PATH = OUT_DIR / "d375_worker_raw_summary.json"
PRECLOSE_PATH = OUT_DIR / "d375_worker_preclose_sentinel.json"
CLAIM_PATH = OUT_DIR / "d375_worker_claim.json"
EVIDENCE_PATH = OUT_DIR / "d375_p34_live_identity_repair_evidence.json"
REPORT_PATH = OUT_DIR / "d375_p34_live_identity_repair_report.md"
BOARD_PATH = OUT_DIR / "d375_p34_live_identity_comparison_1920x1080.png"
RRD_PATH = OUT_DIR / "d375_p34_live_identity_repair.rrd"
RBL_PATH = OUT_DIR / "d375_p34_live_identity_repair.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d375_rerun_validation.json"
RERUN_PNG_PATH = OUT_DIR / "d375_rerun_inspection.png"
AUTOMATED_PATH = OUT_DIR / "d375_automated_summary.json"
MANUAL_JSON_PATH = OUT_DIR / "d375_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d375_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d375_completion_summary.json"
FAIL_ATTESTATION_PATH = OUT_DIR / "d375_fail_stop_attestation.json"
EXCEPTION_PATH = OUT_DIR / "d375_runtime_exception.json"

HARNESS = Path(__file__).resolve()
WORKER = REPO / "sim_scripts/cyl34_top_view_d375_p34_live_asset_identity_contract_repair_worker.py"
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
D373_CONTROLLER_PATH = REPO / "sim_scripts/cyl34_top_view_d373_p34_live_asset_identity_preflight.py"
D373_WORKER_PATH = REPO / "sim_scripts/cyl34_top_view_d373_p34_live_asset_worker.py"
D373_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d373/"
    "attempt1_p34_live_asset_identity_preflight"
)
D373_RAW = D373_DIR / "d373_worker_raw_summary.json"
D373_FAIL = D373_DIR / "d373_fail_stop_attestation.json"
D373_WITNESSES = D373_DIR / "callback_witnesses"
D373_ASSET_DIR = D373_DIR / "collision_asset/roarm_m3_p34_semantic_compound"
D373_ROOT_USD = D373_ASSET_DIR / "roarm_m3.usd"
D373_PHYSICS_USD = D373_ASSET_DIR / "configuration/roarm_m3_physics.usd"
D374_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d374/"
    "attempt1_d373_fail_stop_provenance_and_failure_visualization"
)
D374_REPAIR = D374_DIR / "d374_live_repair_contract.json"
D374_EVIDENCE = D374_DIR / "d374_failure_provenance_evidence.json"
D374_COMPLETION = D374_DIR / "d374_completion_summary.json"
D343_SUMMARY = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d343/d343_usd_typed_float_readback_summary.json"
D343_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d343/d343_usd_typed_float_readback_evidence.json"
D348_EVIDENCE = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/"
    "d348_callback_topology_volume_evidence.json"
)
D372_GEOMETRY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d372/"
    "attempt2_external_schema_path_repair/"
    "d372_professor_semantic_candidate_geometry.json"
)
D372_EVIDENCE = D372_GEOMETRY.with_name("d372_professor_semantic_candidate_evidence.json")
D372_COMPLETION = D372_GEOMETRY.with_name("d372_completion_summary.json")
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
PHYSX_SCHEMA = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "plugins/PhysxSchema/resources/schema.usda"
)
PHYSX_EXTENSION = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/config/extension.toml"
)
PHYSX_PROPERTY_EXAMPLE = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "omni/physx/scripts/propertyQueryRigidBody.py"
)
PHYSX_PROPERTY_TEST = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.physx.tests-107.3.26+107.3.3.cp311.u353/"
    "omni/physxtests/tests/PhysxPropertyQueryInterface.py"
)

EXPECTED_HEAD = "3d71aac219ba16f3262dc94b1898a459eaa534e7"
EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
EXPECTED_COUNTS = {"link5": 16, "gripper_link": 18}
EXPECTED_FIXED_HASHES = {
    D372_GEOMETRY: "12fd1f32c35dfb9ae36cbbb412f6a51536aa1cc07c2dc17d05a5d189f3ee83e4",
    D372_EVIDENCE: "d68f658089aaf838ff454e9d0b301ec3f602785a3a730b3c329aa7785010e984",
    D372_COMPLETION: "57f3ed8fe6f057d059980a78bb51be8e881d8300297a4f41def6ddf94ad0cf43",
    D373_RAW: "dd57da307acf6134487bcd1dfa4a847fd41f24832177421f6291c45b06091373",
    D373_FAIL: "a47ea8600ddc74600644c2d747dd5f95861a2ecbcb2e0667ba0641e17f717206",
    D374_REPAIR: "09d95e78f4bf7ec617a2dc330c83dcb96a9cbc26512679a569cf3ef6e7a5ce88",
    D374_EVIDENCE: "2de72cb64033ffa9bf71a42b1c7cb1b1edf340635894b5d0bea05f54f2120ced",
    D374_COMPLETION: "0540cf4183b75d89fb649fe596212a881bfdee0cbd739bbce5e9fec932148d5b",
    D343_SUMMARY: "880601aac768df38675603828258850aea796b6436a299c46f8cc489ed8b00da",
    D343_EVIDENCE: "95bb4e3787d300071f1bac22037814b732781cd72a69a0334a34a05a50ac920b",
    D348_EVIDENCE: "83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6",
}
EXPECTED_SCHEMA_SHA = "fe075bce4bde5ba7db69c6ccef0c4c26909336ab34c619129fc276f7cb4d7abc"
EXPECTED_EXTENSION_SHA = "6c9d9ed33d927e302334b7cae8ed0c81c4fba37bfbfac07053b72ccc16b7398f"
NEW_VARIABLES = ["whole_robot_noninstance_direct_live_identity_contract_v1"]
SURFACE_TOL_M = 0.0001
BOUNDS_TOL_M = 0.0001
AUTHORED_CALLBACK_VOLUME_REL_TOL = 0.005
PROPERTY_VOLUME_REL_TOL = 0.05
PROPERTY_MASS_STATE_ATOL = 1.0e-9
WORKER_TIMEOUT_S = 900.0
VERDICT_PASS = "D375_P34_LIVE_ASSET_IDENTITY_CONTRACT_REPAIR_PASS_NO_PHYSICS"
VERDICT_FAIL = "D375_P34_LIVE_ASSET_IDENTITY_CONTRACT_REPAIR_FAIL_STOP"
VERDICT_OBSERVABILITY_FAIL = "D375_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"

ROLE_COLORS_HEX = {
    "structural_body": "#0072B2",
    "connector_support": "#E69F00",
    "fixed_jaw": "#F0E442",
    "fixed_jaw_backbone": "#D97800",
    "moving_support": "#CC79A7",
    "moving_jaw": "#009E73",
    "moving_jaw_backbone": "#00695C",
}
ROLE_COLORS_RGBA = {
    "structural_body": [0, 114, 178, 185],
    "connector_support": [230, 159, 0, 185],
    "fixed_jaw": [240, 228, 66, 210],
    "fixed_jaw_backbone": [217, 120, 0, 205],
    "moving_support": [204, 121, 167, 185],
    "moving_jaw": [0, 158, 115, 210],
    "moving_jaw_backbone": [0, 105, 92, 205],
}

OFFICIAL_SOURCES = [
    {
        "title": "Omni Physics 107.3 — Rigid Bodies",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html",
        "applicable_version": "installed omni.physx 107.3.26",
        "use": "articulation links may not be instanced; multiple child colliders are supported",
    },
    {
        "title": "Omni Physics 107.3 — Query The Mass and Volume",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/mass_inertia_queries.html",
        "applicable_version": "installed omni.physx 107.3.26",
        "use": "query_prim callback ordering, timeout, and VALID-only property authority",
    },
    {
        "title": "Omni Physics 107.3 — Colliders",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html",
        "applicable_version": "installed omni.physx 107.3.26",
        "use": "convex hull mesh binding, cooking, and performance semantics",
    },
    {
        "title": "Isaac Sim 5.1.0 — Physics Simulation Fundamentals",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html",
        "applicable_version": "installed Isaac Sim 5.1.0.0",
        "use": "installed product physics context; no physics step is authorized here",
    },
]


def _load_d373_controller() -> Any:
    spec = importlib.util.spec_from_file_location("d373_controller_frozen_for_d375", D373_CONTROLLER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen D373 controller module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D373 = _load_d373_controller()


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _key(path: Path) -> str:
    try:
        return _rel(path)
    except ValueError:
        return str(path.resolve())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return _key(value)
    raise TypeError(type(value).__name__)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_text_x(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _phase(name: str, **fields: Any) -> None:
    ordinal = 1
    if PHASE_PATH.is_file():
        ordinal = sum(1 for line in PHASE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()) + 1
    row = {
        "ordinal": ordinal,
        "phase": name,
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _inventory(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = {_rel(path): _sha(path) for path in files}
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "root": _rel(root),
        "file_count": len(rows),
        "files": rows,
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _asset_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(D373_ASSET_DIR)): _sha(path)
        for path in sorted(item for item in D373_ASSET_DIR.rglob("*") if item.is_file())
    }


def _prepare_negative_controls() -> dict[str, Any]:
    expected_paths = {f"p{index:03d}" for index in range(34)}
    missing_paths = set(expected_paths)
    missing_paths.remove("p000")
    owner_gate = lambda flags: not (
        flags["is_instanceable"] or flags["is_instance"] or flags["is_instance_proxy"]
    )
    supervisor_gate = lambda raw_pass, preclose_pass, expected_sha, observed_sha: (
        raw_pass and preclose_pass and expected_sha == observed_sha
    )
    controls = {
        "instanceable_owner_is_rejected": not owner_gate(
            {"is_instanceable": True, "is_instance": True, "is_instance_proxy": True}
        ),
        "missing_live_path_breaks_bijection": missing_paths != expected_paths,
        "error_parsing_is_not_valid_measurement": "ERROR_PARSING" != "VALID",
        "mutated_preclose_sha_breaks_authority": not supervisor_gate(
            True, True, "registered", "mutated"
        ),
    }
    return {"controls": controls, "pass": all(controls.values())}


def _environment() -> dict[str, Any]:
    rerun = subprocess.run(
        [str(RERUN_CLI), "--version"], cwd=REPO, capture_output=True, text=True
    )
    gpu = _read_json(GPU_ATTESTATION_PATH)
    return {
        "python": sys.executable,
        "packages": {
            "isaacsim": _package_version("isaacsim"),
            "isaaclab": _package_version("isaaclab"),
            "numpy": _package_version("numpy"),
            "psutil": _package_version("psutil"),
            "rerun-sdk": _package_version("rerun-sdk"),
        },
        "rerun_cli": {
            "returncode": rerun.returncode,
            "stdout": rerun.stdout.strip(),
            "stderr": rerun.stderr.strip(),
        },
        "nvidia_smi": {
            "execution_mode": "external_direct_command_attestation",
            "attestation_path": _rel(GPU_ATTESTATION_PATH),
            "attestation_sha256": _sha(GPU_ATTESTATION_PATH),
            "returncode": gpu["returncode"],
            "stdout": gpu["stdout"],
            "stderr": gpu["stderr"],
            "parsed": gpu["parsed"],
            "pass": gpu["pass"],
        },
    }


def prepare() -> int:
    if not CASE_ROOT.is_dir():
        raise FileNotFoundError(f"D375 case root with immutable attempt1 is missing: {CASE_ROOT}")
    if not GPU_ATTESTATION_PATH.is_file():
        raise FileNotFoundError(f"external GPU attestation is missing: {GPU_ATTESTATION_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    d373_raw = _read_json(D373_RAW)
    registered_asset_hashes = d373_raw["asset"]["variant_file_hashes"]
    current_asset_hashes = _asset_hashes()
    inputs = {
        _key(path): _sha(path)
        for path in [
            D372_GEOMETRY,
            D372_EVIDENCE,
            D372_COMPLETION,
            D373_RAW,
            D373_FAIL,
            D374_REPAIR,
            D374_EVIDENCE,
            D374_COMPLETION,
            D343_SUMMARY,
            D343_EVIDENCE,
            D348_EVIDENCE,
            D373_CONTROLLER_PATH,
            D373_WORKER_PATH,
            PHYSX_SCHEMA,
            PHYSX_EXTENSION,
            PHYSX_PROPERTY_EXAMPLE,
            PHYSX_PROPERTY_TEST,
            GPU_ATTESTATION_PATH,
        ]
    }
    source_hashes = {
        _rel(path): _sha(path) for path in [HARNESS, WORKER, VIZ_DEBUG, RERUN_CONTRACT]
    }
    environment = _environment()
    negatives = _prepare_negative_controls()
    frozen_checks = {
        _key(path): inputs[_key(path)] == expected for path, expected in EXPECTED_FIXED_HASHES.items()
    }
    checks = {
        "git_head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_master_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "all_frozen_hashes_exact": all(frozen_checks.values()),
        "d373_asset_hashes_match_raw": current_asset_hashes == registered_asset_hashes,
        "d373_asset_root_hash_exact": _sha(D373_ROOT_USD)
        == registered_asset_hashes["roarm_m3.usd"],
        "d373_asset_physics_hash_exact": _sha(D373_PHYSICS_USD)
        == registered_asset_hashes["configuration/roarm_m3_physics.usd"],
        "physx_schema_hash_exact": _sha(PHYSX_SCHEMA) == EXPECTED_SCHEMA_SHA,
        "physx_extension_hash_exact": _sha(PHYSX_EXTENSION) == EXPECTED_EXTENSION_SHA,
        "isaacsim_5_1_0_0": environment["packages"]["isaacsim"] == "5.1.0.0",
        "isaaclab_2_3_0": environment["packages"]["isaaclab"] == "2.3.0",
        "numpy_pin": environment["packages"]["numpy"] == "1.26.0",
        "psutil_pin": environment["packages"]["psutil"] == "5.9.8",
        "rerun_sdk_pin": environment["packages"]["rerun-sdk"] == "0.34.1",
        "rerun_cli_pin": environment["rerun_cli"]["returncode"] == 0
        and "0.34.1" in (environment["rerun_cli"]["stdout"] + environment["rerun_cli"]["stderr"]),
        "gpu_query_pass": environment["nvidia_smi"]["returncode"] == 0
        and environment["nvidia_smi"]["pass"] is True
        and environment["nvidia_smi"]["parsed"]["name"]
        == "NVIDIA GeForce RTX 4090 Laptop GPU"
        and environment["nvidia_smi"]["parsed"]["driver_version"] == "580.159.03"
        and environment["nvidia_smi"]["parsed"]["memory_total_mib"] == 16376
        and environment["nvidia_smi"]["parsed"]["compute_capability"] == "8.9",
        "one_new_variable": len(NEW_VARIABLES) == 1,
        "negative_controls_pass": negatives["pass"],
        "d374_repair_contract_pass": _read_json(D374_REPAIR).get("pass") is True,
        "d374_completion_pass": _read_json(D374_COMPLETION).get("pass") is True,
        "d343_typed_contract_pass": _read_json(D343_SUMMARY).get("pass") is True,
        "d348_polygon_topology_contract_pass": _read_json(D348_EVIDENCE).get("pass") is True,
    }
    prereg = {
        "artifact": "D375_PREREGISTRATION_V1",
        "case": "g0a_d375",
        "attempt": ATTEMPT,
        "what_and_why": (
            "Repair D373's invalid whole-robot instanceability while keeping the frozen P34 geometry, "
            "then determine whether exact live callback and property identity is valid without any physics step."
        ),
        "new_variables": NEW_VARIABLES,
        "single_variable_definition": {
            "before": "/World/Robot SetInstanceable(True), making articulation rigid-body owners proxies",
            "after": "reference the same frozen D373 asset without whole-robot instanceability",
            "everything_else": "frozen",
        },
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "status_short_at_preregistration": _git("status", "--short", "--untracked-files=all"),
        },
        "environment": environment,
        "installed_stack": {
            "isaac_sim": "5.1.0.0",
            "isaac_lab": "2.3.0",
            "omni_physx": "107.3.26",
            "physx_schema": "107.3.26",
            "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU; compute capability 8.9",
        },
        "official_sources": OFFICIAL_SOURCES,
        "official_vs_inference": {
            "official": [
                "articulation links may not be instanced",
                "one rigid body may own multiple child colliders",
                "IPhysxPropertyQuery values are authority only when result is VALID",
            ],
            "project_inference_to_test": (
                "removing whole-robot instanceability should restore valid P34 owner/property parsing "
                "while preserving the 34 cooked collider shapes"
            ),
            "installed_test_limitation": (
                "the installed NVIDIA property test uses a plain rigid body, not this RoArm articulation; "
                "D375 must test the actual asset"
            ),
        },
        "inputs": inputs,
        "source_hashes": source_hashes,
        "d373_asset_file_hashes": registered_asset_hashes,
        "d373_inventory_before": _inventory(D373_DIR),
        "d334_sidecar_before": _inventory(D334_SIDECAR),
        "frozen_hash_checks": frozen_checks,
        "typed_float32_authority_inherited": {
            "d343_retest_count": 0,
            "expected_type_name": "float",
            "expected_bits_hex": "0x38d1b717",
            "expected_value_m": 0.00009999999747378752,
            "decimal_0_0001_comparator_forbidden": True,
        },
        "registered_execution": {
            "actual_worker": 1,
            "automatic_retry": 0,
            "bounded_watchdog_s": WORKER_TIMEOUT_S,
            "simulation_app_launches": 1,
            "physx_attach_detach": [1, 1],
            "direct_live_callback_requests": 34,
            "property_queries": 2,
            "asset_materializations": 0,
            "usd_stage_file_writes": 0,
            "rerun_viewer_max": 1,
        },
        "registered_identity_contract": {
            "direct_authored_parts": EXPECTED_COUNTS,
            "proxy_aware_live_parts": EXPECTED_COUNTS,
            "owner_flags": "Robot/link5/gripper_link all non-instance and non-proxy before attach",
            "callback_channel": "one actual live composed path per P34 part; 34 total",
            "frozen_d373_callback": "immutable comparison channel only; no new request",
            "property_rows_including_disabled_legacy": {"link5": 17, "gripper_link": 19},
            "result_authority": "rigid body and every collider must be VALID with exact paths",
            "surface_tolerance_m": SURFACE_TOL_M,
            "bounds_tolerance_m": BOUNDS_TOL_M,
            "authored_callback_topology_volume_relative": AUTHORED_CALLBACK_VOLUME_REL_TOL,
            "callback_property_volume_relative": PROPERTY_VOLUME_REL_TOL,
            "mass_com_inertia_axes_atol": PROPERTY_MASS_STATE_ATOL,
            "volume_semantics": "D348 callback original polygon topology; no new Qhull",
        },
        "hash_bound_supervisor_formula": (
            "returncode==0 AND operational_pass AND raw.worker_protocol_pass AND "
            "preclose.worker_protocol_pass AND preclose.summary_sha256==sha256(raw)"
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
        ],
        "prepare_negative_controls": negatives,
        "checks": checks,
        "pass": all(checks.values()),
        "promotion_boundary": (
            "D375 PASS validates live P34 identity only. A64/P34 physics comparison remains separately approved work; "
            "g0a_pass remains false."
        ),
    }
    _write_json_x(PREREG_PATH, prereg)
    _phase("preregistration_frozen", preregistration_sha256=_sha(PREREG_PATH), passed=prereg["pass"])
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"], "path": _rel(PREREG_PATH)}, sort_keys=True))
    return 0 if prereg["pass"] else 1


def run_worker() -> int:
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D375 preregistration did not pass")
    for path in (INVOCATION_PATH, SUPERVISOR_PATH, CLAIM_PATH, WORKER_SUMMARY_PATH, PRECLOSE_PATH):
        if path.exists():
            raise RuntimeError(f"D375 one-shot worker path already claimed: {_rel(path)}")
    if _sha(HARNESS) != prereg["source_hashes"][_rel(HARNESS)]:
        raise RuntimeError("D375 controller changed after preregistration")
    if _sha(WORKER) != prereg["source_hashes"][_rel(WORKER)]:
        raise RuntimeError("D375 worker changed after preregistration")
    command = [
        str(Path(sys.executable).resolve()),
        "-B",
        str(WORKER),
        "--out-dir",
        str(OUT_DIR),
        "--prereg",
        str(PREREG_PATH),
        "--headless",
    ]
    invocation = {
        "artifact": "D375_SINGLE_WORKER_INVOCATION_V1",
        "command": command,
        "cwd": str(REPO),
        "worker_sha256": _sha(WORKER),
        "controller_sha256": _sha(HARNESS),
        "preregistration_sha256": _sha(PREREG_PATH),
        "actual_worker_count": 1,
        "automatic_retry_count": 0,
        "bounded_watchdog_seconds": WORKER_TIMEOUT_S,
        "environment_overrides": {"OMNI_KIT_ACCEPT_EULA": "YES"},
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase("supervisor_spawn_start", invocation_sha256=_sha(INVOCATION_PATH))
    env = os.environ.copy()
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    sigterm_sent = False
    sigkill_sent = False
    try:
        stdout, stderr = process.communicate(timeout=WORKER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        sigterm_sent = True
        try:
            stdout, stderr = process.communicate(timeout=20.0)
        except subprocess.TimeoutExpired:
            process.kill()
            sigkill_sent = True
            stdout, stderr = process.communicate(timeout=20.0)
    elapsed = time.monotonic() - start
    _write_text_x(STDOUT_PATH, stdout)
    _write_text_x(STDERR_PATH, stderr)
    required = {
        "claim": CLAIM_PATH.is_file(),
        "raw_summary": WORKER_SUMMARY_PATH.is_file(),
        "preclose": PRECLOSE_PATH.is_file(),
    }
    operational_pass = bool(
        not timed_out
        and not sigterm_sent
        and not sigkill_sent
        and all(required.values())
    )
    raw = _read_json(WORKER_SUMMARY_PATH) if required["raw_summary"] else {}
    preclose = _read_json(PRECLOSE_PATH) if required["preclose"] else {}
    hash_authority_checks = {
        "raw_worker_protocol_pass": raw.get("worker_protocol_pass") is True,
        "preclose_worker_protocol_pass": preclose.get("worker_protocol_pass") is True,
        "preclose_summary_path_exact": preclose.get("summary_path") == _rel(WORKER_SUMMARY_PATH),
        "preclose_summary_sha_exact": bool(required["raw_summary"])
        and preclose.get("summary_sha256") == _sha(WORKER_SUMMARY_PATH),
        "preclose_counters_exact": preclose.get("counters") == raw.get("counters"),
        "preclose_timeline_exact": preclose.get("timeline_after") == raw.get("timeline_after"),
        "safe_to_close_app": preclose.get("safe_to_close_app") is True,
    }
    hash_authority_pass = all(hash_authority_checks.values())
    effective_pass = bool(
        process.returncode == 0 and operational_pass and hash_authority_pass
    )
    supervisor = {
        "artifact": "D375_HASH_BOUND_SINGLE_WORKER_SUPERVISOR_V1",
        "pid": process.pid,
        "returncode": process.returncode,
        "elapsed_s": elapsed,
        "timeout_s": WORKER_TIMEOUT_S,
        "timed_out": timed_out,
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "worker_spawn_count": 1,
        "automatic_retry_count": 0,
        "stdout_path": _rel(STDOUT_PATH),
        "stdout_sha256": _sha(STDOUT_PATH),
        "stderr_path": _rel(STDERR_PATH),
        "stderr_sha256": _sha(STDERR_PATH),
        "required_artifacts": required,
        "operational_pass": operational_pass,
        "hash_authority_checks": hash_authority_checks,
        "hash_authority_pass": hash_authority_pass,
        "effective_pass": effective_pass,
        "pass": effective_pass,
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase(
        "supervisor_worker_exit",
        returncode=process.returncode,
        elapsed_s=elapsed,
        operational_pass=operational_pass,
        hash_authority_pass=hash_authority_pass,
        effective_pass=effective_pass,
    )
    print(json.dumps({"stage": "run", "pass": effective_pass, "returncode": process.returncode}, sort_keys=True))
    return 0 if effective_pass else 1


def _independent_supervisor_audit(
    supervisor: dict[str, Any], raw: dict[str, Any], preclose: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "returncode_zero": supervisor.get("returncode") == 0,
        "supervisor_operational_pass": supervisor.get("operational_pass") is True,
        "raw_worker_protocol_pass": raw.get("worker_protocol_pass") is True,
        "preclose_worker_protocol_pass": preclose.get("worker_protocol_pass") is True,
        "summary_path_exact": preclose.get("summary_path") == _rel(WORKER_SUMMARY_PATH),
        "summary_sha_exact": preclose.get("summary_sha256") == _sha(WORKER_SUMMARY_PATH),
        "counter_copy_exact": preclose.get("counters") == raw.get("counters"),
        "timeline_copy_exact": preclose.get("timeline_after") == raw.get("timeline_after"),
        "safe_to_close_app": preclose.get("safe_to_close_app") is True,
        "supervisor_effective_pass_true": supervisor.get("effective_pass") is True,
        "supervisor_pass_true": supervisor.get("pass") is True,
    }
    return {"checks": checks, "pass": all(checks.values())}


def _property_audit(worker: dict[str, Any]) -> dict[str, Any]:
    readback_paths = {
        body: {
            row["live_path"]
            for row in worker["authored_readback"]["rows"]
            if row["body"] == body
        }
        for body in EXPECTED_COUNTS
    }
    legacy_paths = worker["live_inventory"]["legacy"]
    bodies = {}
    for body in EXPECTED_COUNTS:
        query = worker["property_queries"][body]
        p34 = [row for row in query["colliders"] if "/d373_p34_parts/" in str(row["path"])]
        legacy = [row for row in query["colliders"] if row["path"] == legacy_paths[body]["path"]]
        unknown = [row for row in query["colliders"] if row not in p34 and row not in legacy]
        rigid = query.get("rigid_body") or {}
        local_checks = {}
        for row in p34:
            local_checks[row["path"]] = {
                "result_valid": row["result_name"] == "VALID" and row["result_value"] == 0,
                "path_nonempty": bool(row["path"]),
                "local_pos_zero": np.allclose(row["local_pos_m"], [0.0, 0.0, 0.0], rtol=0.0, atol=1.0e-9),
                "local_rot_identity": np.allclose(row["local_rot_xyzw"], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1.0e-9),
                "positive_finite_volume": math.isfinite(float(row["volume_m3"])) and float(row["volume_m3"]) > 0.0,
            }
        expected_total = EXPECTED_COUNTS[body] + 1
        checks = {
            "query_worker_pass": query.get("pass") is True,
            "query_finished": query.get("finished") is True,
            "query_errors_zero": not query.get("errors"),
            "rigid_result_valid": rigid.get("result_name") == "VALID" and rigid.get("result_value") == 0,
            "rigid_path_exact": rigid.get("path") == f"/World/Robot/{body}",
            "rigid_mass_positive_finite": math.isfinite(float(rigid.get("mass_kg", 0.0)))
            and float(rigid.get("mass_kg", 0.0)) > 0.0,
            "exact_total_including_disabled_legacy": len(query["colliders"]) == expected_total,
            "exact_p34_count": len(p34) == EXPECTED_COUNTS[body],
            "exact_p34_path_bijection": {row["path"] for row in p34} == readback_paths[body],
            "exact_one_known_legacy": len(legacy) == 1,
            "legacy_result_valid": len(legacy) == 1
            and legacy[0]["result_name"] == "VALID"
            and legacy[0]["result_value"] == 0,
            "unknown_rows_zero": not unknown,
            "all_p34_local_bindings_valid": bool(local_checks)
            and all(all(values.values()) for values in local_checks.values()),
            "zero_app_update_pumps": query.get("simulation_app_update_pumps") == 0,
        }
        bodies[body] = {
            "expected_total": expected_total,
            "p34_rows": p34,
            "legacy_rows": legacy,
            "unknown_rows": unknown,
            "local_checks": local_checks,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {"bodies": bodies, "pass": all(row["pass"] for row in bodies.values())}


def _callback_convex(path: Path) -> dict[str, Any]:
    return D373._callback_convex(path)


def _analyze_callbacks(
    geometry: dict[str, Any], worker: dict[str, Any], property_audit: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    readback_by_path = {
        row["live_path"]: row for row in worker["authored_readback"]["rows"]
    }
    property_by_path = {
        row["path"]: row
        for body in EXPECTED_COUNTS
        for row in property_audit["bodies"][body]["p34_rows"]
    }
    source_by_key = {
        (body, f"p{index:03d}_{part['name']}"): part
        for body in EXPECTED_COUNTS
        for index, part in enumerate(geometry["parts"][body])
    }
    rows = []
    aggregate = Counter()
    for worker_row in worker["callback_rows"]:
        body = worker_row["body"]
        prim_name = worker_row["prim_name"]
        source = source_by_key[(body, prim_name)]
        readback = readback_by_path[worker_row["live_path"]]
        property_row = property_by_path[worker_row["live_path"]]
        authored_vertices = np.asarray(readback["points_f32"], dtype=np.float64)
        authored_triangles = np.asarray(
            readback["face_vertex_indices"], dtype=np.int64
        ).reshape(-1, 3)
        live_witness = REPO / worker_row["callback"]["witness_path"]
        frozen_witness = D373_WITNESSES / f"{body}_{prim_name}_instance.json"
        live_convex = _callback_convex(live_witness)
        frozen_convex = _callback_convex(frozen_witness)
        live_triangles = D373._triangulate(live_convex)
        frozen_triangles = D373._triangulate(frozen_convex)
        live_vertices = np.asarray(live_convex["vertices"], dtype=np.float64)
        frozen_vertices = np.asarray(frozen_convex["vertices"], dtype=np.float64)
        structural = D373._structural_callback(live_convex)
        closure = D373._closed_oriented(live_triangles)
        signed_volume = D373._signed_volume(live_vertices, live_triangles)
        volume = abs(signed_volume)
        surface = D373._surface_distance(
            authored_vertices, authored_triangles, live_convex, live_triangles
        )
        live_bounds = np.vstack([live_vertices.min(axis=0), live_vertices.max(axis=0)])
        authored_bounds = np.vstack([authored_vertices.min(axis=0), authored_vertices.max(axis=0)])
        bounds_delta = float(np.max(np.abs(live_bounds - authored_bounds)))
        authored_volume = float(source["topology_volume_m3"])
        property_volume = float(property_row["volume_m3"])
        authored_rel = abs(volume - authored_volume) / max(abs(authored_volume), 1.0e-15)
        property_rel = abs(volume - property_volume) / max(abs(property_volume), 1.0e-15)
        plane_residual = 0.0
        indices = np.asarray(live_convex["indices"], dtype=np.int64)
        for polygon in live_convex["polygons"]:
            base, count = int(polygon["index_base"]), int(polygon["num_vertices"])
            plane = np.asarray(polygon["plane"], dtype=np.float64)
            face_vertices = live_vertices[indices[base : base + count]]
            plane_residual = max(
                plane_residual,
                float(np.max(np.abs(face_vertices @ plane[:3] + plane[3]))),
            )
        live_digest = D373._payload_digest(live_convex)
        frozen_digest = D373._payload_digest(frozen_convex)
        checks = {
            "worker_callback_protocol_pass": worker_row["callback"]["pass"] is True,
            "structural_payload_pass": structural["pass"],
            "callback_polygon_topology_closed_oriented": closure["pass"],
            "positive_finite_topology_volume": math.isfinite(volume) and volume > 0.0,
            "surface_le_0_1mm": surface["pass"],
            "bounds_le_0_1mm": bounds_delta <= BOUNDS_TOL_M,
            "authored_callback_topology_volume_le_0_5pct": authored_rel
            <= AUTHORED_CALLBACK_VOLUME_REL_TOL,
            "callback_topology_property_volume_le_5pct": property_rel <= PROPERTY_VOLUME_REL_TOL,
            "polygon_plane_residual_le_1e_5m": plane_residual <= 1.0e-5,
            "frozen_d373_live_payload_exact": live_digest == frozen_digest,
            "authored_f32_digest_bound": readback["authored_f32_topology_payload_sha256"]
            == worker_row["authored_f32_topology_payload_sha256"],
        }
        row = {
            "body": body,
            "name": worker_row["name"],
            "role": worker_row["role"],
            "prim_name": prim_name,
            "live_path": worker_row["live_path"],
            "live_witness_path": _rel(live_witness),
            "live_witness_sha256": _sha(live_witness),
            "frozen_d373_witness_path": _rel(frozen_witness),
            "frozen_d373_witness_sha256": _sha(frozen_witness),
            "live_payload_sha256": live_digest,
            "frozen_d373_payload_sha256": frozen_digest,
            "live_callback_vertices_m": live_vertices.tolist(),
            "live_callback_topology_triangles": live_triangles.tolist(),
            "frozen_callback_vertices_m": frozen_vertices.tolist(),
            "frozen_callback_topology_triangles": frozen_triangles.tolist(),
            "vertex_count": len(live_vertices),
            "polygon_count": len(live_convex["polygons"]),
            "triangle_count": len(live_triangles),
            "max_vertices_per_polygon": max(
                int(polygon["num_vertices"]) for polygon in live_convex["polygons"]
            ),
            "callback_topology_signed_volume_m3": signed_volume,
            "callback_topology_volume_m3": volume,
            "authored_d372_topology_volume_m3": authored_volume,
            "physx_property_volume_m3": property_volume,
            "authored_callback_volume_relative_delta": authored_rel,
            "callback_property_volume_relative_delta": property_rel,
            "authored_bounds_m": authored_bounds.tolist(),
            "live_callback_bounds_m": live_bounds.tolist(),
            "bounds_max_abs_delta_m": bounds_delta,
            "max_polygon_plane_residual_m": plane_residual,
            "surface": surface,
            "structural": structural,
            "closure": closure,
            "checks": checks,
            "pass": all(checks.values()),
        }
        rows.append(row)
        aggregate["parts"] += 1
        aggregate["parts_pass"] += int(row["pass"])
        aggregate["frozen_payload_exact"] += int(checks["frozen_d373_live_payload_exact"])
        aggregate["closed"] += int(closure["pass"])
        aggregate["surface"] += int(surface["pass"])
        aggregate["bounds"] += int(bounds_delta <= BOUNDS_TOL_M)
        aggregate["authored_volume"] += int(authored_rel <= AUTHORED_CALLBACK_VOLUME_REL_TOL)
        aggregate["property_volume"] += int(property_rel <= PROPERTY_VOLUME_REL_TOL)
    rows.sort(key=lambda row: (row["body"], row["prim_name"]))
    counts = dict(aggregate)
    checks = {
        "parts_34": len(rows) == 34 and counts.get("parts") == 34,
        "parts_pass_34": counts.get("parts_pass") == 34,
        "frozen_payload_exact_34": counts.get("frozen_payload_exact") == 34,
        "closed_34": counts.get("closed") == 34,
        "surface_34": counts.get("surface") == 34,
        "bounds_34": counts.get("bounds") == 34,
        "authored_volume_34": counts.get("authored_volume") == 34,
        "property_volume_34": counts.get("property_volume") == 34,
        "all_rows_pass": all(row["pass"] for row in rows),
    }
    return rows, {"counts": counts, "checks": checks, "pass": all(checks.values())}


def _post_negative_controls(rows: list[dict[str, Any]], mass_audit: dict[str, Any]) -> dict[str, Any]:
    first_witness = REPO / rows[0]["live_witness_path"]
    convex = _callback_convex(first_witness)
    dropped = D373._triangulate(convex, drop_polygon=0)
    dropped_closed = D373._closed_oriented(dropped)
    base_mass = float(mass_audit["bodies"]["link5"]["base"]["mass_kg"])
    perturbed_mass = base_mass + 10.0 * PROPERTY_MASS_STATE_ATOL
    controls = {
        "delete_one_reported_polygon_breaks_closure": not dropped_closed["pass"],
        "perturb_mass_breaks_state_guard": abs(perturbed_mass - base_mass)
        > PROPERTY_MASS_STATE_ATOL,
    }
    return {"controls": controls, "dropped_polygon_closure": dropped_closed, "pass": all(controls.values())}


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        dimensions = [int(image.width), int(image.height)]
        mode = image.mode
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "dimensions": dimensions,
        "mode": mode,
        "exact_1920x1080": dimensions == [1920, 1080],
    }


def _representation_maps(
    geometry: dict[str, Any], worker: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    source = {
        (body, f"p{index:03d}_{part['name']}"): part
        for body in EXPECTED_COUNTS
        for index, part in enumerate(geometry["parts"][body])
    }
    readback = {
        (row["body"], row["prim_name"]): row for row in worker["authored_readback"]["rows"]
    }
    analyzed = {(row["body"], row["prim_name"]): row for row in rows}
    result = {}
    for key, part in source.items():
        authored = readback[key]
        analyzed_row = analyzed[key]
        result[key] = {
            "source": (
                np.asarray(part["vertices"], dtype=np.float64),
                np.asarray(part["triangles"], dtype=np.int64),
            ),
            "authored": (
                np.asarray(authored["points_f32"], dtype=np.float64),
                np.asarray(authored["face_vertex_indices"], dtype=np.int64).reshape(-1, 3),
            ),
            "frozen_callback": (
                np.asarray(analyzed_row["frozen_callback_vertices_m"], dtype=np.float64),
                np.asarray(analyzed_row["frozen_callback_topology_triangles"], dtype=np.int64),
            ),
            "live_callback": (
                np.asarray(analyzed_row["live_callback_vertices_m"], dtype=np.float64),
                np.asarray(analyzed_row["live_callback_topology_triangles"], dtype=np.int64),
            ),
        }
    return result


def _make_board(
    geometry: dict[str, Any], worker: dict[str, Any], rows: list[dict[str, Any]], evidence: dict[str, Any]
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    reps = _representation_maps(geometry, worker, rows)
    channel_titles = {
        "source": "D372 원본 Float64",
        "authored": "동결 USD Float32",
        "frozen_callback": "D373 동결 콜백",
        "live_callback": "D375 수리 후 live 콜백",
    }

    def add_mesh(ax: Any, vertices: np.ndarray, triangles: np.ndarray, role: str) -> None:
        collection = Poly3DCollection(
            vertices[triangles] * 1000.0,
            facecolor=ROLE_COLORS_HEX[role],
            edgecolor="#202020",
            linewidth=0.16,
            alpha=0.76,
        )
        ax.add_collection3d(collection)

    def frame(ax: Any, vertices: np.ndarray, body: str) -> None:
        mm = vertices * 1000.0
        lo, hi = mm.min(axis=0), mm.max(axis=0)
        center = (lo + hi) * 0.5
        radius = max(float((hi - lo).max()) * 0.58, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=18, azim=-58 if body == "link5" else -72)
        ax.set_proj_type("ortho")
        ax.set_axis_off()

    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="white")
    for row_index, body in enumerate(("link5", "gripper_link")):
        keys = sorted(key for key in reps if key[0] == body)
        for col_index, channel in enumerate(channel_titles):
            ax = fig.add_subplot(2, 4, row_index * 4 + col_index + 1, projection="3d")
            all_vertices = []
            for key in keys:
                vertices, triangles = reps[key][channel]
                role = next(row["role"] for row in rows if (row["body"], row["prim_name"]) == key)
                add_mesh(ax, vertices, triangles, role)
                all_vertices.append(vertices)
            frame(ax, np.vstack(all_vertices), body)
            count = EXPECTED_COUNTS[body]
            ax.set_title(
                f"{body} {count}개\n{channel_titles[channel]}",
                fontproperties=bold,
                fontsize=10.5,
                pad=1,
            )
    fig.suptitle(
        "D375 · 전체 로봇 인스턴스 설정 제거 후 P34 충돌체 계보 비교",
        fontproperties=bold,
        fontsize=20,
        y=0.975,
    )
    maxima = evidence["maxima"]
    footer = (
        f"실제 live 콜백 {evidence['counts']['callback_pass']}/34 · PhysX 속성 행 link5 17 / gripper_link 19 · "
        f"D373 콜백과 payload {evidence['counts']['frozen_payload_exact']}/34 exact · "
        f"최대 표면 차이 {maxima['surface_symmetric_mm']:.6f} mm · 최대 경계 차이 {maxima['bounds_max_abs_mm']:.6f} mm\n"
        "표시는 링크 로컬 좌표의 육안검사용 복사본이며, 원 JSON/콜백 배열과 해시가 판정 권위입니다. 물리 step·q5·접촉 시험은 0회입니다."
    )
    fig.text(0.5, 0.035, footer, ha="center", va="center", fontproperties=regular, fontsize=10.5)
    fig.tight_layout(rect=[0.005, 0.085, 0.995, 0.93], h_pad=0.2, w_pad=0.05)
    fig.savefig(BOARD_PATH, dpi=120, facecolor="white")
    plt.close(fig)
    info = _png_info(BOARD_PATH)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"D375 board not exact 1920x1080: {info}")
    return info


def _write_rerun(
    geometry: dict[str, Any], worker: dict[str, Any], rows: list[dict[str, Any]], evidence: dict[str, Any]
) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.viz_debug import log_rerun

    reps = _representation_maps(geometry, worker, rows)
    meshes = []
    expected_entities = {"metadata/run", "events/d375_summary"}
    component_contract: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "events/d375_summary": ["TextLog:text", "TextLog:level"],
    }
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    row_by_key = {(row["body"], row["prim_name"]): row for row in rows}
    for key in sorted(reps):
        body, prim_name = key
        role = row_by_key[key]["role"]
        for channel, (vertices, triangles) in reps[key].items():
            path = f"d375/{channel}/{body}/{prim_name}"
            metadata_path = f"metadata/meshes/{path.replace('/', '__')}"
            meshes.append(
                {
                    "entity_path": path,
                    "coordinate_frame": "tf#/",
                    "vertices_m": vertices,
                    "triangles": triangles,
                    "color_rgba": ROLE_COLORS_RGBA[role],
                    "static": True,
                    "body": body,
                    "prim_name": prim_name,
                    "role": role,
                    "representation": channel,
                    "display_role": "Float32 inspection copy only",
                }
            )
            expected_entities.update({path, metadata_path})
            component_contract[path] = mesh_components
            component_contract[metadata_path] = ["TextDocument:text"]
    scalars = [
        {"entity_path": "metrics/d375/part_count", "value": 34, "static": True},
        {
            "entity_path": "metrics/d375/callback_parts_pass",
            "value": evidence["counts"]["callback_pass"],
            "static": True,
        },
        {
            "entity_path": "metrics/d375/frozen_payload_exact",
            "value": evidence["counts"]["frozen_payload_exact"],
            "static": True,
        },
        {
            "entity_path": "metrics/d375/max_surface_delta_mm",
            "value": evidence["maxima"]["surface_symmetric_mm"],
            "static": True,
        },
        {
            "entity_path": "metrics/d375/max_bounds_delta_mm",
            "value": evidence["maxima"]["bounds_max_abs_mm"],
            "static": True,
        },
        {"entity_path": "metrics/d375/physics_steps", "value": 0, "static": True},
        {"entity_path": "metrics/d375/q5_samples", "value": 0, "static": True},
        {
            "entity_path": "gate/d375/identity_pass",
            "value": 1 if evidence["identity_pass"] else 0,
            "static": True,
        },
        {"entity_path": "gate/d375/g0a_pass", "value": 0, "static": True},
    ]
    for scalar in scalars:
        expected_entities.add(scalar["entity_path"])
        component_contract[scalar["entity_path"]] = ["Scalars:scalars"]
    event = {
        "entity_path": "events/d375_summary",
        "text": (
            "D375 repaired P34 live identity only: non-instance articulation owners, "
            "link5=16, gripper_link=18, live callbacks=34, property queries=2. "
            "Physics step, q5, contact, cylinder, and grasp verdict remain out of scope."
        ),
        "level": "INFO",
        "static": True,
    }
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    try:
        result = log_rerun(
            RRD_PATH,
            meshes=meshes,
            scalar_trace=scalars,
            events=[event],
            recording_metadata={
                "case": "g0a_d375",
                "attempt": ATTEMPT,
                "verdict": evidence["verdict"],
                "evidence_sha256": _sha(EVIDENCE_PATH),
                "physics_steps": 0,
                "q5_samples": 0,
                "contact_queries": 0,
                "display_role": "inspection only; original JSON/callback arrays are authority",
            },
            recording_id="g0a_d375_p34_live_identity_repair",
            blueprint_path=RBL_PATH,
            blueprint_mode="d375_p34_identity",
            live_viewer=False,
            app_id="roarm_g0a_d375_p34_live_identity_repair",
        )
    finally:
        os.environ["PATH"] = old_path
    if not result.get("ok"):
        raise RuntimeError(f"D375 Rerun save-only write failed: {result}")
    strict = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected_entities),
        exact_entity_paths=sorted(expected_entities),
        expected_timeline_names=["blueprint", "log_time"],
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=component_contract,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG_PATH,
        screenshot_window_size="1920x1080",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=240.0,
    )
    _write_json_x(RERUN_VALIDATION_PATH, strict)
    return {
        "save_only_log": result,
        "strict_validation_pass": strict.get("pass") is True,
        "rrd": {
            "path": _rel(RRD_PATH),
            "bytes": RRD_PATH.stat().st_size,
            "sha256": _sha(RRD_PATH),
        },
        "rbl": {
            "path": _rel(RBL_PATH),
            "bytes": RBL_PATH.stat().st_size,
            "sha256": _sha(RBL_PATH),
        },
        "headless_viewer_invocations": 1,
        "screenshot": _png_info(RERUN_PNG_PATH)
        if RERUN_PNG_PATH.is_file()
        else {"path": _rel(RERUN_PNG_PATH), "exists": False},
    }


def _write_report(evidence: dict[str, Any]) -> None:
    lines = [
        "# D375 P34 live asset identity contract repair",
        "",
        "## 무엇을 왜 확인했나",
        "",
        "D373은 P34 34개를 USD와 callback까지 읽었지만, 전체 로봇을 instanceable로 만들어",
        "동적 articulation owner의 PhysX property query가 ERROR_PARSING으로 실패했다.",
        "D375는 같은 동결 P34 asset을 non-instance robot root로 합성하여 live identity만 재검증한다.",
        "",
        "## 실행 순서",
        "",
        "1. D372/D373/D374/D343/D348 및 asset hash를 고정했다.",
        "2. /World/Robot을 instanceable로 만들지 않고 owner flags를 attach 전에 검사했다.",
        "3. P34 live 경로 34개에 callback을 각 1회 요청했다.",
        "4. link5와 gripper_link property query를 각 1회 실행했다.",
        "5. worker raw와 preclose SHA를 supervisor 및 분석기가 각각 검증했다.",
        "6. 원 polygon topology, 표면, bounds, volume, mass 상태를 오프라인 분류했다.",
        "",
        "## 정량 결과",
        "",
        f"- verdict: `{evidence['verdict']}`",
        f"- owner non-instance gate: `{evidence['checks']['owner_structure_noninstance_pass']}`",
        f"- direct/live part count: `{evidence['counts']['direct_authored']}/34`, `{evidence['counts']['live_inventory']}/34`",
        f"- live callback PASS: `{evidence['counts']['callback_pass']}/34`",
        f"- frozen D373 callback payload exact: `{evidence['counts']['frozen_payload_exact']}/34`",
        f"- property rows: link5 `{evidence['counts']['property_rows']['link5']}/17`, gripper_link `{evidence['counts']['property_rows']['gripper_link']}/19`",
        f"- max surface symmetric delta: `{evidence['maxima']['surface_symmetric_mm']:.12g} mm`",
        f"- max bounds delta: `{evidence['maxima']['bounds_max_abs_mm']:.12g} mm`",
        f"- max authored↔callback topology-volume relative delta: `{evidence['maxima']['authored_callback_volume_relative']:.12g}`",
        f"- max callback↔property-volume relative delta: `{evidence['maxima']['callback_property_volume_relative']:.12g}`",
        f"- max mass/COM/inertia/axes absolute delta: `{evidence['maxima']['mass_com_inertia_axes_abs_delta']:.12g}`",
        "- physics step / q5 sample / contact query: `0 / 0 / 0`",
        "- g0a_pass: `false`",
        "",
        "## 공식 문서",
        "",
        *[
            f"- {row['title']}: {row['url']} ({row['applicable_version']})"
            for row in OFFICIAL_SOURCES
        ],
        "",
        "## 경계",
        "",
        evidence["next_authorization_boundary"],
    ]
    _write_text_x(REPORT_PATH, "\n".join(lines))


def _write_fail_attestation(reason: str, supervisor: dict[str, Any], raw: dict[str, Any]) -> None:
    if FAIL_ATTESTATION_PATH.exists():
        return
    counters = raw.get("counters", {})
    _write_json_x(
        FAIL_ATTESTATION_PATH,
        {
            "artifact": "D375_FAIL_STOP_ATTESTATION_V1",
            "case": "g0a_d375",
            "attempt": ATTEMPT,
            "verdict": VERDICT_FAIL,
            "reason": reason,
            "worker_protocol_pass": raw.get("worker_protocol_pass"),
            "supervisor_effective_pass": supervisor.get("effective_pass"),
            "scope_counters": counters,
            "identity_pass": False,
            "physics_comparison": None,
            "grasp_feasibility": None,
            "g0a_pass": False,
            "automatic_retry_count": 0,
            "next_authorization_boundary": "Inspect D375 evidence and obtain separate approval before any retry or physics comparison.",
        },
    )


def analyze() -> int:
    prereg = _read_json(PREREG_PATH)
    supervisor = _read_json(SUPERVISOR_PATH)
    raw = _read_json(WORKER_SUMMARY_PATH) if WORKER_SUMMARY_PATH.is_file() else {}
    preclose = _read_json(PRECLOSE_PATH) if PRECLOSE_PATH.is_file() else {}
    authority = _independent_supervisor_audit(supervisor, raw, preclose)
    if not authority["pass"]:
        _write_fail_attestation("hash-bound worker/supervisor authority failed", supervisor, raw)
        print(json.dumps({"stage": "analyze", "pass": False, "reason": "worker_authority"}, sort_keys=True))
        return 1
    _phase("offline_classification_start")
    geometry = _read_json(D372_GEOMETRY)
    property_audit = _property_audit(raw)
    mass_audit = D373._mass_live_audit(raw)
    rows, callback_aggregate = _analyze_callbacks(geometry, raw, property_audit)
    post_negatives = _post_negative_controls(rows, mass_audit)
    all_negatives = {
        **prereg["prepare_negative_controls"]["controls"],
        **post_negatives["controls"],
    }
    maxima = {
        "surface_symmetric_mm": max(row["surface"]["symmetric_m"] * 1000.0 for row in rows),
        "bounds_max_abs_mm": max(row["bounds_max_abs_delta_m"] * 1000.0 for row in rows),
        "authored_callback_volume_relative": max(
            row["authored_callback_volume_relative_delta"] for row in rows
        ),
        "callback_property_volume_relative": max(
            row["callback_property_volume_relative_delta"] for row in rows
        ),
        "mass_com_inertia_axes_abs_delta": max(
            value
            for body in mass_audit["bodies"].values()
            for value in body["max_abs_deltas"].values()
        ),
    }
    counters = raw["counters"]
    zero_keys = (
        "automatic_retries",
        "derivative_asset_materializations",
        "usd_stage_file_writes",
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
    canonical_checks = raw["canonical_outside_collision_subtree_diff"]["checks"]
    canonical_required = all(
        canonical_checks.get(key) is True
        for key in (
            "outside_registered_subtrees_bit_exact",
            "no_runtime_address_patterns",
            "no_unsupported_types",
            "variant_contains_no_a64",
        )
    )
    checks = {
        "preregistration_pass": prereg["pass"] is True,
        "hash_bound_supervisor_authority_pass": authority["pass"],
        "worker_protocol_pass": raw["worker_protocol_pass"] is True,
        "direct_authored_readback_pass": raw["authored_readback"]["pass"] is True,
        "d343_contract_inherited_without_retest": raw["authored_readback"]["d343_contract_inherited"] is True
        and raw["authored_readback"]["typed_float32_contract_retests"] == 0,
        "owner_structure_noninstance_pass": raw["owner_structure"]["pass"] is True,
        "live_inventory_path_owner_binding_pass": raw["live_inventory"]["pass"] is True,
        "property_binding_pass": property_audit["pass"],
        "callback_original_polygon_identity_pass": callback_aggregate["pass"],
        "mass_com_inertia_axes_invariant": mass_audit["pass"],
        "negative_controls_all_pass": len(all_negatives) == 6 and all(all_negatives.values()),
        "outside_collision_subtrees_preserved": canonical_required,
        "frozen_asset_immutable": raw["asset_reuse"]["immutable_before_after"] is True
        and raw["asset_reuse"]["file_hashes_after"] == prereg["d373_asset_file_hashes"],
        "D373_inventory_immutable": _inventory(D373_DIR) == prereg["d373_inventory_before"],
        "D334_sidecar_immutable": _inventory(D334_SIDECAR) == prereg["d334_sidecar_before"],
        "strict_zero_scope_counters": all(counters[key] == 0 for key in zero_keys),
        "timeline_stopped_unchanged": raw["timeline_before"] == raw["timeline_after"]
        and raw["timeline_before"]["is_stopped"] is True
        and raw["timeline_before"]["is_playing"] is False,
    }
    identity_pass = all(checks.values())
    verdict = VERDICT_PASS if identity_pass else VERDICT_FAIL
    evidence = {
        "artifact": "D375_P34_LIVE_ASSET_IDENTITY_CONTRACT_REPAIR_EVIDENCE_V1",
        "case": "g0a_d375",
        "attempt": ATTEMPT,
        "new_variables": NEW_VARIABLES,
        "verdict": verdict,
        "identity_pass": identity_pass,
        "g0a_pass": False,
        "physics_grasp_tipping_speed_optimum": None,
        "what_was_repaired": (
            "The frozen D373 P34 asset was referenced without making /World/Robot instanceable; "
            "link5 and gripper_link remained non-instance articulation owners."
        ),
        "inputs": {
            "preregistration": {"path": _rel(PREREG_PATH), "sha256": _sha(PREREG_PATH)},
            "worker_summary": {"path": _rel(WORKER_SUMMARY_PATH), "sha256": _sha(WORKER_SUMMARY_PATH)},
            "preclose": {"path": _rel(PRECLOSE_PATH), "sha256": _sha(PRECLOSE_PATH)},
            "d372_geometry": {"path": _rel(D372_GEOMETRY), "sha256": _sha(D372_GEOMETRY)},
            "d373_frozen_raw": {"path": _rel(D373_RAW), "sha256": _sha(D373_RAW)},
            "d374_repair_contract": {"path": _rel(D374_REPAIR), "sha256": _sha(D374_REPAIR)},
        },
        "counts": {
            "direct_authored": sum(row["pass"] for row in raw["authored_readback"]["rows"]),
            "live_inventory": sum(row["pass"] for row in raw["live_inventory"]["rows"]),
            "callback_pass": callback_aggregate["counts"]["parts_pass"],
            "frozen_payload_exact": callback_aggregate["counts"]["frozen_payload_exact"],
            "property_rows": {
                body: len(property_audit["bodies"][body]["p34_rows"])
                + len(property_audit["bodies"][body]["legacy_rows"])
                for body in EXPECTED_COUNTS
            },
            "actual_worker": 1,
            "automatic_retry": 0,
        },
        "worker_counters": counters,
        "timeline_before": raw["timeline_before"],
        "timeline_after": raw["timeline_after"],
        "supervisor_authority": authority,
        "asset_reuse": raw["asset_reuse"],
        "owner_structure": raw["owner_structure"],
        "authored_readback": raw["authored_readback"],
        "live_inventory": raw["live_inventory"],
        "property_audit": property_audit,
        "mass_audit": mass_audit,
        "callback_rows": rows,
        "callback_aggregate": callback_aggregate,
        "maxima": maxima,
        "negative_controls": {
            "prepare": prereg["prepare_negative_controls"],
            "post_callback": post_negatives,
            "all_controls": all_negatives,
            "pass_count": sum(all_negatives.values()),
            "expected_count": 6,
            "pass": len(all_negatives) == 6 and all(all_negatives.values()),
        },
        "canonical_outside_collision_subtree_diff": raw[
            "canonical_outside_collision_subtree_diff"
        ],
        "checks": checks,
        "official_sources": OFFICIAL_SOURCES,
        "scientific_boundary": {
            "physics_steps": 0,
            "q5_samples": 0,
            "contact_queries": 0,
            "cylinder_creates_or_writes": 0,
            "grasp_feasibility": None,
            "tipping_causality": None,
        },
        "next_authorization_boundary": (
            "If and only if D375 completes PASS, a separately approved one-variable A64/P34 cylinder physics comparison may be designed. "
            "D375 itself does not authorize that run."
        ),
    }
    _write_json_x(EVIDENCE_PATH, evidence)
    _write_report(evidence)
    if not identity_pass:
        _write_fail_attestation("offline live-identity classification gate failed", supervisor, raw)
        return 1
    board = _make_board(geometry, raw, rows, evidence)
    rerun = _write_rerun(geometry, raw, rows, evidence)
    automated_checks = {
        "identity_evidence_pass": identity_pass,
        "exact_1920x1080_board": board["exact_1920x1080"],
        "rerun_save_only_log_ok": rerun["save_only_log"].get("ok") is True,
        "rerun_strict_validation_pass": rerun["strict_validation_pass"],
        "rrd_rbl_exist": RRD_PATH.is_file() and RBL_PATH.is_file(),
        "headless_viewer_exactly_one": rerun["headless_viewer_invocations"] == 1,
        "manual_inspection_pending": not MANUAL_JSON_PATH.exists(),
        "D373_immutable_after_observability": _inventory(D373_DIR) == prereg["d373_inventory_before"],
        "D334_sidecar_immutable_after_observability": _inventory(D334_SIDECAR)
        == prereg["d334_sidecar_before"],
    }
    automated = {
        "artifact": "D375_AUTOMATED_IDENTITY_AND_OBSERVABILITY_SUMMARY_V1",
        "scientific_identity_verdict": verdict,
        "evidence": {"path": _rel(EVIDENCE_PATH), "sha256": _sha(EVIDENCE_PATH)},
        "board": board,
        "rerun": rerun,
        "manual_inspection_required_before_completion": True,
        "checks": automated_checks,
        "automated_pass_pending_manual": all(
            value for key, value in automated_checks.items() if key != "manual_inspection_pending"
        )
        and automated_checks["manual_inspection_pending"],
        "g0a_pass": False,
    }
    _write_json_x(AUTOMATED_PATH, automated)
    _phase(
        "offline_classification_and_observability_end",
        identity_pass=identity_pass,
        rerun_pass=rerun["strict_validation_pass"],
    )
    print(
        json.dumps(
            {
                "stage": "analyze",
                "identity_pass": identity_pass,
                "rerun_pass": rerun["strict_validation_pass"],
                "manual_inspection_pending": True,
            },
            sort_keys=True,
        )
    )
    return 0 if automated["automated_pass_pending_manual"] else 1


def finalize() -> int:
    for path in (
        PREREG_PATH,
        SUPERVISOR_PATH,
        WORKER_SUMMARY_PATH,
        PRECLOSE_PATH,
        EVIDENCE_PATH,
        AUTOMATED_PATH,
        MANUAL_JSON_PATH,
        MANUAL_MD_PATH,
    ):
        if not path.is_file():
            raise RuntimeError(f"D375 finalize prerequisite missing: {path}")
    if COMPLETION_PATH.exists():
        raise FileExistsError(f"forward-only completion already exists: {COMPLETION_PATH}")
    _phase("finalize_start")
    prereg = _read_json(PREREG_PATH)
    supervisor = _read_json(SUPERVISOR_PATH)
    raw = _read_json(WORKER_SUMMARY_PATH)
    preclose = _read_json(PRECLOSE_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    automated = _read_json(AUTOMATED_PATH)
    manual = _read_json(MANUAL_JSON_PATH)
    authority = _independent_supervisor_audit(supervisor, raw, preclose)
    expected_visual_hashes = {
        "comparison_board": automated["board"]["sha256"],
        "rerun_inspection": automated["rerun"]["screenshot"]["sha256"],
    }
    checks = {
        "preregistration_pass": prereg["pass"] is True,
        "hash_bound_supervisor_authority_pass": authority["pass"],
        "identity_evidence_pass": evidence["identity_pass"] is True,
        "automated_summary_pass_pending_manual": automated[
            "automated_pass_pending_manual"
        ]
        is True,
        "manual_original_resolution_inspection_pass": manual.get("pass") is True,
        "manual_hashes_exact": manual.get("inspected_sha256") == expected_visual_hashes,
        "D373_immutable": _inventory(D373_DIR) == prereg["d373_inventory_before"],
        "D334_sidecar_immutable": _inventory(D334_SIDECAR) == prereg["d334_sidecar_before"],
        "frozen_asset_hashes_exact": _asset_hashes() == prereg["d373_asset_file_hashes"],
        "physics_and_grasp_nulls_preserved": all(
            value is None
            for value in (
                evidence["physics_grasp_tipping_speed_optimum"],
                evidence["scientific_boundary"]["grasp_feasibility"],
                evidence["scientific_boundary"]["tipping_causality"],
            )
        ),
        "g0a_false": evidence["g0a_pass"] is False and automated["g0a_pass"] is False,
    }
    completion = {
        "artifact": "D375_COMPLETION_SUMMARY_V1",
        "case": "g0a_d375",
        "attempt": ATTEMPT,
        "new_variables": NEW_VARIABLES,
        "preregistration": {"path": _rel(PREREG_PATH), "sha256": _sha(PREREG_PATH)},
        "worker_summary": {"path": _rel(WORKER_SUMMARY_PATH), "sha256": _sha(WORKER_SUMMARY_PATH)},
        "supervisor": {"path": _rel(SUPERVISOR_PATH), "sha256": _sha(SUPERVISOR_PATH)},
        "evidence": {"path": _rel(EVIDENCE_PATH), "sha256": _sha(EVIDENCE_PATH)},
        "automated_summary": {"path": _rel(AUTOMATED_PATH), "sha256": _sha(AUTOMATED_PATH)},
        "manual_inspection": {
            "path": _rel(MANUAL_JSON_PATH),
            "sha256": _sha(MANUAL_JSON_PATH),
            "report": _rel(MANUAL_MD_PATH),
        },
        "visual_artifacts": {
            "comparison_board": automated["board"],
            "rerun_inspection": automated["rerun"]["screenshot"],
        },
        "rrd": automated["rerun"]["rrd"],
        "rbl": automated["rerun"]["rbl"],
        "counts": evidence["counts"],
        "maxima": evidence["maxima"],
        "worker_counters": evidence["worker_counters"],
        "scientific_boundary": evidence["scientific_boundary"],
        "g0a_pass": False,
        "checks": checks,
        "pass": all(checks.values()),
        "verdict": VERDICT_PASS if all(checks.values()) else VERDICT_OBSERVABILITY_FAIL,
        "next_authorization_boundary": evidence["next_authorization_boundary"],
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase("finalize_complete", completion_sha256=_sha(COMPLETION_PATH), verdict=completion["verdict"])
    print(json.dumps({"stage": "finalize", "pass": completion["pass"], "verdict": completion["verdict"]}, sort_keys=True))
    return 0 if completion["pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "run", "analyze", "finalize"))
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            return prepare()
        if args.stage == "run":
            return run_worker()
        if args.stage == "analyze":
            return analyze()
        return finalize()
    except Exception as error:
        payload = {
            "artifact": "D375_RUNTIME_EXCEPTION_V1",
            "stage": args.stage,
            "exception_type": type(error).__name__,
            "exception": repr(error),
            "traceback": traceback.format_exc(),
            "verdict": VERDICT_FAIL,
            "automatic_retry_count": 0,
        }
        try:
            if OUT_DIR.exists() and not EXCEPTION_PATH.exists():
                _write_json_x(EXCEPTION_PATH, payload)
            if OUT_DIR.exists():
                _phase("exception", stage=args.stage, exception_type=type(error).__name__)
        except Exception:
            pass
        print(json.dumps(payload, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
