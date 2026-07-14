#!/usr/bin/env python3
"""D349 frozen open-jaw raw/live target-distance gate.

The case re-materializes the frozen D337 target in one fresh Isaac process,
queries the retained raw triangle meshes first, and then queries the immutable
D347 live-callback geometry reconstructed with the D348 polygon topology.  It
does not request a new cook callback, mutate an asset, or execute a physics
step.  Settle, trials, G0b, RL, and ladder promotion remain forbidden.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import secrets
import struct
import subprocess
import sys
import traceback
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
    sha256_file,
    validate_rerun_artifact,
)
from roarm_rl.viz_debug import draw_frames, log_rerun  # noqa: E402
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d332_grasp_g0a_static_collision_discriminator as d332,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d333_grasp_g0a_sole_support_static_retest as d333,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d334_grasp_g0a_live_collision_shape_ownership_audit as d334,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d336_grasp_g0a_finite_grid_caveat_discriminator as d336,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d337_grasp_g0a_open_jaw_target_gate as d337,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair as d339,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d344_grasp_g0a_attempt3_fixed_point_collision_geometry as d344,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d347_grasp_g0a_asset_validator_activation_order_repair as d347,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d348_grasp_g0a_physx_property_query_volume_semantics as d348,
)


OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d349"
PREREG_PATH = OUT_DIR / "d349_preregistration.json"
PARAMETER_AUDIT_PATH = OUT_DIR / "d349_parameter_freeze_audit.json"
PREFLIGHT_PATH = OUT_DIR / "d349_validate_preflight.json"
HOME_PATH = OUT_DIR / "d349_home_start_contract.json"
CORRECTED_AUDIT_PATH = OUT_DIR / "d349_d348_corrected_live_topology_audit.json"
BINDING_PATH = OUT_DIR / "d349_live_topology_runtime_binding.json"
CONTROLS_PATH = OUT_DIR / "d349_d337_frozen_controls.json"
MEASUREMENT_PATH = OUT_DIR / "d349_frozen_target_distance_measurement.json"
DECISION_PNG = OUT_DIR / "d349_raw_live_target_distance_decision.png"
RRD_PATH = OUT_DIR / "d349_frozen_target_distance.rrd"
RBL_PATH = OUT_DIR / "d349_frozen_target_distance.rbl"
RERUN_SCREENSHOT_PATH = OUT_DIR / "d349_frozen_target_distance_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d349_rerun_validation.json"
AUTOMATED_SUMMARY_PATH = OUT_DIR / "d349_automated_summary.json"
AUTOMATED_REPORT_PATH = OUT_DIR / "d349_automated_report.md"
RUNTIME_EXCEPTION_PATH = OUT_DIR / "d349_runtime_exception.json"
MANUAL_PATH = OUT_DIR / "d349_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d349_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d349_completion_summary.json"
COMPLETION_MD_PATH = OUT_DIR / "d349_completion_report.md"

SESSION_DOC = REPO / (
    "claudedocs/session_20260714_grasp_g0a_d349_"
    "frozen_open_jaw_target_live_distance_gate.md"
)
START_HERE = REPO / "START_HERE.md"
HARNESS = Path(__file__).resolve()

D344_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d344"
ATTEMPT3_DIR = D344_DIR / "collision_asset/attempt3"
VARIANT_DIR = ATTEMPT3_DIR / "roarm_m3_fullmesh_fixed_point_parts"
VARIANT_ROBOT_USD = VARIANT_DIR / "roarm_m3.usd"
VARIANT_PHYSICS_USD = VARIANT_DIR / "configuration/roarm_m3_physics.usd"
CORE_MANIFEST = ATTEMPT3_DIR / "d340_attempt3_asset_manifest.json"
D334_SUMMARY = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d334/"
    "g0a_d334_live_collision_audit_summary.json"
)
D336_SUMMARY = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d336/"
    "g0a_d336_finite_grid_caveat_summary.json"
)
D337_SEARCH = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d337/d337_search.json"
D337_PREPHYSICS = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d337/d337_prephysics_gate.json"
)
D347_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d347"
D347_COMPLETION = D347_DIR / "d347_completion_summary.json"
D347_LIVE_AUDIT = D347_DIR / "d347_fresh_live_representation_audit.json"
D347_RAW = D347_DIR / "d347_raw_live_measurement.json"
D347_WITNESS_MANIFEST = D347_DIR / "d347_validate_cook_witness_manifest.json"
D347_WITNESS_DIR = D347_DIR / "d347_validate_cook_witnesses"
D348_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348"
D348_COMPLETION = (
    D348_DIR / "attempt5_ascii_contract/d348_completion_summary.json"
)
D348_EVIDENCE = D348_DIR / "attempt2/d348_callback_topology_volume_evidence.json"
D348_HOME = D348_DIR / "attempt2/d348_home_start_contract.json"

STORAGE_BASELINE = {
    REPO / "claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/ARCHIVE_PLAN.md": {
        "status": " M",
        "bytes": 9843,
        "sha256": "1b664a25a9f38158d23feeed6c20dd4feca687d9e81e0e6a45bcc64983168d9e",
    },
    REPO
    / "claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/"
    "raw_local_cleanup_receipt_20260714.json": {
        "status": "??",
        "bytes": 3421,
        "sha256": "1b143a65068b794351777b2de9391bb835f2875f7a2629ddf5de612068ab23d4",
    },
    REPO
    / "claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/"
    "raw_predelete_manifest_20260714.tsv": {
        "status": "??",
        "bytes": 21255018,
        "sha256": "462418736b7dfe3542138441edec710dbc472da60a61fe065b4b91ff58427750",
    },
    REPO / "claudedocs/session_20260714_cube10cm_0_999_windows_archive_local_raw_cleanup.md": {
        "status": "??",
        "bytes": 3711,
        "sha256": "ce2a0776caed78ff313442e682aa8fe5357a381c7a0f2a30b43ec1f12a8e21cf",
    },
}

EXPECTED_HEAD = "25f085a388a29c18baffa5789cc0d47f713a4728"
EXPECTED_CRITICAL_HASHES = {
    "variant_robot_usd": "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
    "variant_physics_usd": "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503",
    "core_manifest": "52bc2e40f586cd8401aa0fcabf23e5e733776b9c373082bd49fbda01190c2bb5",
    "d337_search": "15d471fce6646db6c587abe4fff991e01efe6bb0d9ea73b8472fef0096a0df34",
    "d337_prephysics": "2d63a84be48e121a8566bf7364cbd99551587501b445bcdb83d6a38196d90c15",
    "d337_harness": "081d0a77c91b27373eadca51fd0d9aa530d14fcdf0f64b24797a9cf8e3109489",
    "d347_completion": "93ae7a6daea4d8ba9af6fa09d01deb6c72017925375195a53804b0d55286d65e",
    "d347_live_audit": "e652b16063cc0d7f9370df7e597ba6dcff9813f260c897b3b58b8b6c4d1b96ab",
    "d347_raw": "2b2306862b4fc0cb22ffc6ed41c179f542b4a07f7014db17290c2003e99dfb9a",
    "d347_witness_manifest": "a57bcd32b60c65ead4313a8914c8c2d61efd3fb7d620b993ba29af6967791438",
    "d348_completion": "bc93b77fbfbeee074b1241b8f48c0317745b62ff5bca5e2196da00d25eb28697",
    "d348_evidence": "83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6",
    "d348_home": "bd4fcb39ffbe8bc5dfb9bc2797f3ba73b1c669f3bad79cee070ebdd40f8816df",
    "d348_harness": "444cbad5faa878a69252accd2e6923d39fc0cbc0715dd727600638cb4613acba",
    "viz_debug": "622b7197afe8cdeb1bb5411f2a961aa9a9a5c58aaf248417fc145639374577c5",
}

NEW_VARIABLES = ["frozen_open_jaw_target_live_distance_gate"]
SEED = 33201
Q5_OPEN_RAD = d339.Q5_OPEN_RAD
RADIAL_NM = d339.OLD_RADIAL_NM
TANGENT_NM = d339.OLD_TANGENT_NM
CLEAR_GATE_MM = d339.CLEAR_GATE_MM
FIDELITY_TOL_MM = d339.TASK_FIDELITY_TOL_MM
DIAGNOSTIC_DISTANCE_DELTA_REFERENCE_MM = d337.PARITY_TOL_MM
PROPERTY_REL_TOL = d339.PROPERTY_VOLUME_BINDING_REL_TOL
VOLUME_REPRO_ABS_TOL_M3 = d348.TRANSLATION_VOLUME_ABS_TOL_M3
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"

VERDICT_PREREQUISITE = "D349_RUNTIME_OR_FROZEN_INPUT_CONTRACT_FAIL_STOP"
VERDICT_TARGET_FAIL = "D349_FROZEN_OPEN_JAW_TARGET_DISTANCE_FAIL_STOP"
VERDICT_PENDING = "D349_FROZEN_OPEN_JAW_TARGET_LIVE_DISTANCE_SUPPORTED_MANUAL_PENDING"
VERDICT_OBSERVABILITY = "D349_RERUN_OBSERVABILITY_INCOMPLETE_STOP"
VERDICT_COMPLETE = "D349_FROZEN_OPEN_JAW_TARGET_LIVE_DISTANCE_SUPPORTED"

EXPECTED_RERUN_COUNTS = {
    "frame_count": 6,
    "coordinate_frame_count": 2,
    "mesh_count": 522,
    "point_entity_count": 4,
    "arrow_entity_count": 4,
    "scalar_row_count": 1040,
    "event_row_count": 136,
    "exact_non_system_entity_count": 2112,
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> dict[str, str]:
    output = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result: dict[str, str] = {}
    for line in output.splitlines():
        status = line[:2]
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        result[value] = status
    return result


def _allowed_status_paths() -> set[str]:
    return {
        "START_HERE.md",
        _relative(SESSION_DOC),
        _relative(HARNESS),
        *(_relative(path) for path in STORAGE_BASELINE),
    }


def _status_scope_pass(status: dict[str, str]) -> bool:
    prefix = _relative(OUT_DIR) + "/"
    return all(path in _allowed_status_paths() or path.startswith(prefix) for path in status)


def _critical_paths() -> dict[str, Path]:
    return {
        "variant_robot_usd": VARIANT_ROBOT_USD,
        "variant_physics_usd": VARIANT_PHYSICS_USD,
        "core_manifest": CORE_MANIFEST,
        "d337_search": D337_SEARCH,
        "d337_prephysics": D337_PREPHYSICS,
        "d337_harness": REPO / "sim_scripts/cyl34_top_view_d337_grasp_g0a_open_jaw_target_gate.py",
        "d347_completion": D347_COMPLETION,
        "d347_live_audit": D347_LIVE_AUDIT,
        "d347_raw": D347_RAW,
        "d347_witness_manifest": D347_WITNESS_MANIFEST,
        "d348_completion": D348_COMPLETION,
        "d348_evidence": D348_EVIDENCE,
        "d348_home": D348_HOME,
        "d348_harness": REPO
        / "sim_scripts/cyl34_top_view_d348_grasp_g0a_physx_property_query_volume_semantics.py",
        "viz_debug": REPO / "roarm_rl/viz_debug.py",
    }


def _critical_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, path in _critical_paths().items()}


def _external_baseline_report() -> dict[str, Any]:
    status = _git_status()
    rows = []
    for path, expected in STORAGE_BASELINE.items():
        row = {
            "path": _relative(path),
            "exists": path.is_file(),
            "status": status.get(_relative(path)),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256(path) if path.is_file() else None,
            "expected": expected,
        }
        row["pass"] = bool(
            row["exists"]
            and row["status"] == expected["status"]
            and row["bytes"] == expected["bytes"]
            and row["sha256"] == expected["sha256"]
        )
        rows.append(row)
    return {"rows": rows, "pass": all(row["pass"] for row in rows)}


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": _relative(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _inventory_digest(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_inventories() -> dict[str, dict[str, Any]]:
    roots = {"d344_attempt3": ATTEMPT3_DIR, "d347_witnesses": D347_WITNESS_DIR}
    result = {}
    for name, root in roots.items():
        rows = _inventory(root)
        result[name] = {
            "root": _relative(root),
            "file_count": len(rows),
            "digest": _inventory_digest(rows),
            "rows": rows,
        }
    return result


def _png_dimensions(path: Path) -> str | None:
    if not path.is_file():
        return None
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return f"{width}x{height}"


def _expected_rrd_contract() -> tuple[list[str], dict[str, list[str]]]:
    entities, components = d344._expected_rrd_contract()
    entities = ["events/d349" if path == "events/d344" else path for path in entities]
    components = {
        ("events/d349" if path == "events/d344" else path): list(values)
        for path, values in components.items()
    }
    for name in ("status", "link5", "gripper", "scope"):
        path = f"events/d349_summary/{name}"
        entities.append(path)
        components[path] = ["TextLog:level", "TextLog:text"]
    point_components = [
        "CoordinateFrame:frame",
        "Points3D:colors",
        "Points3D:labels",
        "Points3D:positions",
        "Points3D:radii",
    ]
    arrow_components = [
        "Arrows3D:colors",
        "Arrows3D:labels",
        "Arrows3D:origins",
        "Arrows3D:radii",
        "Arrows3D:vectors",
        "CoordinateFrame:frame",
    ]
    for representation in ("source", "instance"):
        for body in d334.BODY_LABELS:
            base = f"cook/{representation}/{body}/distance_witness"
            entities.extend([f"{base}/endpoints", f"{base}/vector"])
            components[f"{base}/endpoints"] = point_components
            components[f"{base}/vector"] = arrow_components
    return sorted(entities), components


def _rrd_contract_digest() -> str:
    entities, components = _expected_rrd_contract()
    payload = {
        "exact_non_system_entity_paths": entities,
        "exact_timeline_names": ["blueprint", "event_idx", "log_time", "part_idx"],
        "required_components_by_path": components,
        "exact_observation_counts": EXPECTED_RERUN_COUNTS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parameter_audit() -> dict[str, Any]:
    checks = {
        "new_variable_count_one": len(NEW_VARIABLES) == 1,
        "new_physical_variables_zero": True,
        "target_frozen": bool(
            RADIAL_NM == 7_000_000
            and TANGENT_NM == 11_000_000
            and math.isclose(Q5_OPEN_RAD, 1.5413, rel_tol=0.0, abs_tol=0.0)
            and d332.ADOPTED_TANGENT_SIGN == -1.0
            and SEED == 33201
        ),
        "distance_gates_frozen": bool(
            CLEAR_GATE_MM == 0.1
            and FIDELITY_TOL_MM == 0.5
            and DIAGNOSTIC_DISTANCE_DELTA_REFERENCE_MM == 0.05
            and PROPERTY_REL_TOL == 0.05
        ),
        "object_frozen": bool(
            d332.CYLINDER_RADIUS_M == 0.017
            and d332.CYLINDER_HEIGHT_M == 0.09
            and d332.OBJECT_MASS_KG == 0.72
            and d332.STATIC_FRICTION == 1.5
            and d332.DYNAMIC_FRICTION == 1.2
        ),
        "decomposition_frozen": copy.deepcopy(d339.DECOMPOSITION_PARAMS)
        == copy.deepcopy(d347.DECOMPOSITION_PARAMS),
    }
    return {
        "artifact": "D349_PARAMETER_FREEZE_AUDIT_V1",
        "new_variables": NEW_VARIABLES,
        "new_variable_count": 1,
        "new_physical_variables": [],
        "parameters_added": [],
        "parameters_removed": [],
        "parameters_increased": [],
        "parameters_decreased": [],
        "parameters_changed": [],
        "thresholds_relaxed": [],
        "asset_changes": [],
        "decomposition_changes": [],
        "target_changes": [],
        "material_changes": [],
        "actuator_changes": [],
        "physics_setting_changes": [],
        "decomposition_parameters": copy.deepcopy(d339.DECOMPOSITION_PARAMS),
        "target": {
            "radial_offset_mm": RADIAL_NM / 1.0e6,
            "tangent_offset_mm": TANGENT_NM / 1.0e6,
            "tangent_sign": d332.ADOPTED_TANGENT_SIGN,
            "q5_rad": Q5_OPEN_RAD,
            "seed": SEED,
            "ik": "HOME-seeded position-only",
        },
        "tolerances": {
            "clear_gate_mm": CLEAR_GATE_MM,
            "raw_live_fidelity_mm": FIDELITY_TOL_MM,
            "diagnostic_distance_delta_reference_mm": {
                "value": DIAGNOSTIC_DISTANCE_DELTA_REFERENCE_MM,
                "verdict_authority": False,
                "provenance": "frozen D337 parity tolerance; display context only in D349",
            },
            "property_volume_relative": PROPERTY_REL_TOL,
            "topology_volume_reproduction_abs_m3": VOLUME_REPRO_ABS_TOL_M3,
        },
        "scope_guards": {
            "asset_write": False,
            "cook_callback_requests": 0,
            "physx_property_queries": 0,
            "controlled_physics_steps": 0,
            "settle": False,
            "ten_trial": False,
            "g0b": False,
            "rl": False,
            "ladder_promotion": False,
            "g0a_pass": False,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run_prepare(args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"D349 output already exists: {OUT_DIR}")
    status = _git_status()
    critical = _critical_hashes()
    external = _external_baseline_report()
    if _git_head() != EXPECTED_HEAD:
        raise RuntimeError("D349 HEAD drift before preregistration")
    if critical != EXPECTED_CRITICAL_HASHES:
        raise RuntimeError("D349 critical frozen-input hash mismatch")
    if not _status_scope_pass(status):
        raise RuntimeError(f"D349 git status has out-of-scope paths: {status}")
    if not external["pass"]:
        raise RuntimeError("D349 external storage baseline changed")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    parameter = _parameter_audit()
    _write_json(PARAMETER_AUDIT_PATH, parameter)
    inventories = _source_inventories()
    state_hashes = {
        "start_here": _sha256(START_HERE),
        "session_doc": _sha256(SESSION_DOC),
    }
    prereg_checks = {
        "parameter_audit_pass": parameter["pass"],
        "numpy_pin": str(np.__version__) == "1.26.0",
        "psutil_pin": str(psutil.__version__) == "5.9.8",
        "rerun_pin": str(rr.__version__) == RERUN_CONTRACT_VERSION == "0.34.1",
        "registered_python": str(Path(sys.executable).resolve())
        == str(Path(REGISTERED_PYTHON).resolve()),
        "head_exact": _git_head() == EXPECTED_HEAD,
        "status_scope": _status_scope_pass(_git_status()),
        "external_baseline": _external_baseline_report()["pass"],
        "d348_completion_pass": _json(D348_COMPLETION).get("completion_contract_pass")
        is True,
        "d348_verdict_exact": _json(D348_COMPLETION).get("final_verdict")
        == "D348_PHYSX_PROPERTY_QUERY_TOPOLOGY_SEMANTICS_SUPPORTED",
        "d348_evidence_128": _json(D348_EVIDENCE).get("aggregate", {}).get("part_count")
        == 128,
        "d348_evidence_pass": _json(D348_EVIDENCE).get("pass") is True,
    }
    prereg = {
        "artifact": "D349_PREREGISTRATION_V1",
        "case": "g0a_d349",
        "git_head": _git_head(),
        "prepare_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "user_authorization": {
            "authorized": True,
            "source": (
                "current user turn: D349 step-by-step으로 확인하면서 순차적으로 진행"
            ),
            "scope": "one measurement-only frozen open-jaw raw/live distance gate",
        },
        "registered_scientific_execution_count": 1,
        "registered_stages": ["prepare", "validate", "manual_inspection", "finalize"],
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "critical_hashes": critical,
        "state_hashes": state_hashes,
        "harness_sha256": _sha256(HARNESS),
        "parameter_audit_sha256": _sha256(PARAMETER_AUDIT_PATH),
        "source_inventories": inventories,
        "external_storage_baseline": external,
        "measurement_semantics": {
            "raw": "retained D334 source triangle BVH",
            "live_authority": (
                "D347 callback vertices plus D348 polygon-topology triangles, "
                "BVHModelOBBRSS triangulated face-surface 64-part union"
            ),
            "diagnostic_1": (
                "hppfcl.Convex(points, topology triangles): D348 volume-binding witness; "
                "its GJK distance has no D349 verdict authority"
            ),
            "diagnostic_2": (
                "vertex-only Convex.convexHull distance; no D349 verdict authority"
            ),
            "live_distance_scope": (
                "callback-face surface proxy for the active collider, not a direct PhysX "
                "narrowphase distance query"
            ),
            "direct_physx_narrowphase_distance_api": False,
            "diagnostics_can_veto_authoritative_gate": False,
        },
        "query_order": [
            "raw_mesh_first",
            "raw_witness_repeat",
            "live_callback_topology_surface_authority",
            "non_authoritative_convex_and_qhull_diagnostics",
        ],
        "rerun_contract": {
            "profile": "MEASURED_AUTHORITY",
            "counts": EXPECTED_RERUN_COUNTS,
            "contract_sha256": _rrd_contract_digest(),
            "prerequisite_failure_policy": (
                "case incomplete with runtime_exception evidence; no scientific geometry "
                "verdict and no falsely registered PREQUERY_STOP RRD"
            ),
        },
        "scope_guards": parameter["scope_guards"],
        "checks": prereg_checks,
        "pass": all(prereg_checks.values()),
    }
    _write_json(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"]}, sort_keys=True))
    return 0 if prereg["pass"] else 2


def _app_launcher_checks(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "headless": args.headless is True,
        "livestream_zero": int(args.livestream) == 0,
        "enable_cameras_false": getattr(args, "enable_cameras", False) is False,
        "xr_false": getattr(args, "xr", False) is False,
        "device_cuda0": str(args.device) == "cuda:0",
        "cpu_false": getattr(args, "cpu", False) is False,
        "experience_empty": str(getattr(args, "experience", "")) == "",
        "kit_args_empty": str(getattr(args, "kit_args", "")) == "",
    }


def _prepare_validate_preflight(args: argparse.Namespace) -> bool:
    import torch

    prereg = _json(PREREG_PATH)
    parameter = _json(PARAMETER_AUDIT_PATH)
    current_inventories = _source_inventories()
    checks = {
        "prereg_pass": prereg.get("pass") is True,
        "fresh_process_pid": prereg.get("prepare_process_identity", {}).get("pid")
        != os.getpid(),
        "fresh_process_nonce": prereg.get("prepare_process_identity", {}).get("nonce")
        != args.process_nonce,
        "head_exact": _git_head() == prereg.get("git_head") == EXPECTED_HEAD,
        "status_scope": _status_scope_pass(_git_status()),
        "critical_hashes": _critical_hashes()
        == prereg.get("critical_hashes")
        == EXPECTED_CRITICAL_HASHES,
        "state_hashes": {
            "start_here": _sha256(START_HERE),
            "session_doc": _sha256(SESSION_DOC),
        }
        == prereg.get("state_hashes"),
        "harness_hash": _sha256(HARNESS) == prereg.get("harness_sha256"),
        "parameter_hash": _sha256(PARAMETER_AUDIT_PATH)
        == prereg.get("parameter_audit_sha256"),
        "parameter_pass": parameter.get("pass") is True,
        "source_inventories": current_inventories == prereg.get("source_inventories"),
        "external_storage_baseline": _external_baseline_report()["pass"],
        "numpy_pin": str(np.__version__) == "1.26.0",
        "psutil_pin": str(psutil.__version__) == "5.9.8",
        "rerun_pin": str(rr.__version__) == RERUN_CONTRACT_VERSION == "0.34.1",
        "registered_python": str(Path(sys.executable).resolve())
        == str(Path(REGISTERED_PYTHON).resolve()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_4090": bool(
            torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)
        ),
        "app_launcher": all(_app_launcher_checks(args).values()),
        "manual_pythonpath_absent": os.environ.get("PYTHONPATH") in (None, ""),
        "output_runtime_absent": all(
            not path.exists()
            for path in (
                HOME_PATH,
                CORRECTED_AUDIT_PATH,
                BINDING_PATH,
                CONTROLS_PATH,
                MEASUREMENT_PATH,
                DECISION_PNG,
                RRD_PATH,
                RBL_PATH,
                RERUN_SCREENSHOT_PATH,
                RERUN_VALIDATION_PATH,
                AUTOMATED_SUMMARY_PATH,
                AUTOMATED_REPORT_PATH,
                RUNTIME_EXCEPTION_PATH,
                MANUAL_PATH,
                MANUAL_MD_PATH,
                COMPLETION_PATH,
                COMPLETION_MD_PATH,
            )
        ),
    }
    report = {
        "artifact": "D349_VALIDATE_PREFLIGHT_V1",
        "case": "g0a_d349",
        "preregistration_sha256": _sha256(PREREG_PATH),
        "parameter_audit_sha256": _sha256(PARAMETER_AUDIT_PATH),
        "validate_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "git_status": _git_status(),
        "critical_hashes": _critical_hashes(),
        "source_inventories": current_inventories,
        "external_storage_baseline": _external_baseline_report(),
        "environment": {
            "numpy": str(np.__version__),
            "psutil": str(psutil.__version__),
            "rerun": str(rr.__version__),
            "python": str(Path(sys.executable).resolve()),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "app_launcher_checks": _app_launcher_checks(args),
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json(PREFLIGHT_PATH, report)
    return bool(report["pass"])


def _topology_canonical(channel: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(channel["vertices_m"], dtype=np.float64)
    triangles = np.asarray(channel["topology_triangles"], dtype=np.int64)
    payload = {
        "vertices_m": vertices.tolist(),
        "triangles": triangles.tolist(),
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
    }
    payload["geometry_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["topology_sha256"] = hashlib.sha256(
        triangles.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    payload["vertex_stream_sha256"] = hashlib.sha256(
        vertices.astype("<f8", copy=False).tobytes()
    ).hexdigest()
    payload["bounds_m"] = [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]
    payload["centroid_m"] = vertices.mean(axis=0).tolist()
    payload["volume_m3"] = float(channel["volume_origin_m3"])
    payload["live_polygon_count"] = int(channel["polygon_count"])
    return payload


def _corrected_live_audit() -> dict[str, Any]:
    audit = copy.deepcopy(_json(D347_LIVE_AUDIT))
    evidence = _json(D348_EVIDENCE)
    evidence_by_key = {(row["body"], row["name"]): row for row in evidence["rows"]}
    for body in d334.BODY_LABELS:
        for row in audit["per_body"][body]["part_checks"]:
            source = evidence_by_key[(body, row["name"])]
            for channel_name in ("instance", "prototype"):
                row["channel_consensus"][channel_name]["canonical"] = _topology_canonical(
                    source[channel_name]
                )
            row["channel_consensus"]["consensus"] = _topology_canonical(source["instance"])
            row["property_vs_channel_volume_relative_difference"] = {
                "instance": source["topology_instance_relative_error"],
                "prototype": source["topology_prototype_relative_error"],
            }
            row["max_property_volume_relative_error"] = max(
                float(source["topology_instance_relative_error"]),
                float(source["topology_prototype_relative_error"]),
            )
            row["checks"]["property_vs_both_channel_volume_le_5pct"] = bool(
                source["checks"]["topology_vs_property_both_le_5pct"]
            )
            row["checks"]["d348_callback_topology_preserved"] = bool(source["pass"])
            row["pass"] = all(row["checks"].values())
        body_checks = audit["per_body"][body]["checks"]
        body_checks["all_corrected_parts_pass"] = all(
            row["pass"] for row in audit["per_body"][body]["part_checks"]
        )
        body_checks["certified_parts_64"] = (
            sum(bool(row["pass"]) for row in audit["per_body"][body]["part_checks"])
            == 64
        )
        audit["per_body"][body]["pass"] = all(body_checks.values())
    audit["artifact"] = "D349_D348_CORRECTED_LIVE_TOPOLOGY_AUDIT_V1"
    audit["source_artifact"] = "D347_FRESH_LIVE_REPRESENTATION_AUDIT_V1"
    audit["d348_evidence"] = {
        "path": _relative(D348_EVIDENCE),
        "sha256": _sha256(D348_EVIDENCE),
        "pass": evidence["pass"],
    }
    audit["checks"].update(
        {
            "all_parts_corrected_pass_128_of_128": all(
                row["pass"]
                for body in d334.BODY_LABELS
                for row in audit["per_body"][body]["part_checks"]
            ),
            "both_bodies_pass": all(
                audit["per_body"][body]["pass"] for body in d334.BODY_LABELS
            ),
            "property_volume_bound_128_of_128": sum(
                bool(row["checks"]["property_vs_both_channel_volume_le_5pct"])
                for body in d334.BODY_LABELS
                for row in audit["per_body"][body]["part_checks"]
            )
            == 128,
        }
    )
    audit["pass"] = all(audit["checks"].values())
    return audit


def _home_start_contract(inner: Any) -> dict[str, Any]:
    actual = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    home = np.radians(np.asarray(d332.HOME_DEG, dtype=np.float64)).astype(np.float32)
    jitter = actual - home
    q5_idx = list(inner._robot.joint_names).index(d332.GRIPPER_JOINT_NAME)
    limit = float(np.nextafter(np.float32(0.02), np.float32(np.inf)))
    checks = {
        "joint_count_six": len(actual) == 6,
        "finite": bool(np.isfinite(actual).all()),
        "arm_jitter_within_plus_minus_0p02rad": bool(
            np.max(np.abs(np.delete(jitter, q5_idx))) <= limit
        ),
        "q5_exact_zero_closed": bool(actual[q5_idx].tobytes() == np.float32(0.0).tobytes()),
        "classified_home_near_not_exact_home": bool(not np.array_equal(actual, home)),
        "counter_zero": int(inner._sim_step_counter) == 0,
    }
    return {
        "artifact": "D349_HOME_START_CONTRACT_V1",
        "seed": SEED,
        "nominal_home_deg": np.asarray(d332.HOME_DEG, dtype=np.float64).tolist(),
        "actual_reset_joint_rad_float32": actual.tolist(),
        "reset_jitter_rad_float32": jitter.tolist(),
        "q5_index": q5_idx,
        "q5_convention": "0=CLOSED; 1.5413=OPEN target",
        "classification": "HOME-near deterministic jitter, q5=0 CLOSED",
        "exact_home": False,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _build_live_topology_parts(
    inner: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import hppfcl

    evidence = _json(D348_EVIDENCE)
    historical = _json(D347_LIVE_AUDIT)
    historical_by_key = {
        (body, row["name"]): row
        for body in d334.BODY_LABELS
        for row in historical["per_body"][body]["part_checks"]
    }
    inventories = {body: d334._usd_collision_inventory(inner, body) for body in d334.BODY_LABELS}
    inventory_by_path = {
        body: {row["path"]: row for row in inventories[body]} for body in d334.BODY_LABELS
    }
    parts: dict[str, list[dict[str, Any]]] = {body: [] for body in d334.BODY_LABELS}
    public_rows = []
    for source in evidence["rows"]:
        body, name = source["body"], source["name"]
        historical_row = historical_by_key[(body, name)]
        path = historical_row["path"]
        current = inventory_by_path[body].get(path)
        instance = source["instance"]
        vertices = np.asarray(instance["vertices_m"], dtype=np.float64)
        triangles = np.asarray(instance["topology_triangles"], dtype=np.int64)
        points = d332._fcl_points(hppfcl, vertices)
        topology_triangles = d332._fcl_triangles(hppfcl, triangles)
        volume_witness_convex = hppfcl.Convex(points, topology_triangles)
        topology_surface_bvh = d332._build_raw_bvh(hppfcl, vertices, triangles)
        try:
            qhull_diagnostic = hppfcl.Convex.convexHull(
                d332._fcl_points(hppfcl, vertices), False, ""
            )
            qhull_error = None
        except Exception as error:  # diagnostic-only channel cannot veto D349 science
            qhull_diagnostic = None
            qhull_error = f"{type(error).__name__}: {error}"
        direct_volume = float(volume_witness_convex.computeVolume())
        identity = d339._prim_to_body_transform(inner, path, body)
        instance_witness = REPO / instance["witness_path"]
        prototype_witness = REPO / source["prototype"]["witness_path"]
        checks = {
            "d348_row_pass": source["pass"] is True,
            "instance_prototype_payload_exact": source["checks"][
                "raw_instance_prototype_payload_exact"
            ]
            is True,
            "topology_closed_oriented": source["checks"][
                "both_topologies_closed_and_oriented"
            ]
            is True,
            "topology_property_le_5pct": source["checks"][
                "topology_vs_property_both_le_5pct"
            ]
            is True,
            "witness_instance_hash_exact": instance_witness.is_file()
            and _sha256(instance_witness) == instance["witness_sha256"],
            "witness_prototype_hash_exact": prototype_witness.is_file()
            and _sha256(prototype_witness) == source["prototype"]["witness_sha256"],
            "runtime_path_present": current is not None,
            "runtime_collision_enabled": bool(current and current["collision_enabled"]),
            "runtime_owner_matches": bool(
                current and current["nearest_rigid_body_ancestor"] == d334.BODY_PATHS[body]
            ),
            "runtime_piece_to_body_identity": identity["identity_pass"],
            "volume_witness_point_count_exact": int(volume_witness_convex.num_points)
            == len(vertices),
            "volume_witness_triangle_count_exact": int(volume_witness_convex.num_polygons)
            == len(triangles),
            "volume_witness_reproduces_d348_topology": abs(
                direct_volume - float(instance["volume_origin_m3"])
            )
            <= VOLUME_REPRO_ABS_TOL_M3,
            "volume_witness_geometry_nonnull": volume_witness_convex is not None,
            "topology_surface_authority_nonnull": topology_surface_bvh is not None,
        }
        part = {
            "body": body,
            "name": name,
            "path": path,
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
            "topology_volume_m3": float(instance["volume_origin_m3"]),
            "primary_compute_volume_m3": direct_volume,
            "property_volume_m3": float(source["property_volume_m3"]),
            "topology_property_relative_error": float(
                source["topology_instance_relative_error"]
            ),
            "identity_max_abs_delta": identity["identity_max_abs_delta"],
            "diagnostics": {
                "qhull_available": qhull_diagnostic is not None,
                "qhull_error": qhull_error,
                "verdict_authority": False,
            },
            "checks": checks,
            "pass": all(checks.values()),
            "_vertices": vertices,
            "_triangles": triangles,
            "_geometry_topology_surface_authority": topology_surface_bvh,
            "_geometry_volume_witness_convex": volume_witness_convex,
            "_geometry_qhull_diagnostic": qhull_diagnostic,
        }
        parts[body].append(part)
        public_rows.append({key: value for key, value in part.items() if not key.startswith("_")})
    body_checks = {}
    for body in d334.BODY_LABELS:
        enabled = sorted(row["path"] for row in inventories[body] if row["collision_enabled"])
        expected = sorted(row["path"] for row in parts[body])
        legacy = [
            row for row in inventories[body] if row["path"] == d339.LIVE_OLD_COLLIDER_PATHS[body]
        ]
        body_checks[body] = {
            "part_count_64": len(parts[body]) == 64,
            "all_parts_pass": all(row["pass"] for row in parts[body]),
            "enabled_paths_exact": enabled == expected,
            "disabled_legacy_exact_one": len(legacy) == 1
            and legacy[0]["collision_enabled"] is False,
        }
    report = {
        "artifact": "D349_LIVE_TOPOLOGY_RUNTIME_BINDING_V1",
        "measurement_geometry": {
            "authority": "BVHModelOBBRSS on D348 callback-topology triangles",
            "volume_witness": (
                "hppfcl.Convex(points, D348 topology triangles); computeVolume only"
            ),
            "distance_diagnostics_no_verdict_authority": [
                "hppfcl.Convex(points, D348 topology triangles)",
                "Convex.convexHull(points)",
            ],
        },
        "d348_evidence_path": _relative(D348_EVIDENCE),
        "d348_evidence_sha256": _sha256(D348_EVIDENCE),
        "runtime_inventories": inventories,
        "parts": public_rows,
        "body_checks": body_checks,
        "checks": {
            "d348_evidence_pass": evidence["pass"] is True,
            "part_rows_128": len(public_rows) == 128,
            "all_part_rows_pass": all(row["pass"] for row in public_rows),
            "both_body_bindings_pass": all(all(row.values()) for row in body_checks.values()),
        },
    }
    report["pass"] = all(report["checks"].values())
    return parts, report


def _epa_contacts_with_witness(
    hppfcl: Any, geometry: Any, transform: Any, cylinder: Any, cylinder_tf: Any
) -> dict[str, Any]:
    request = hppfcl.CollisionRequest()
    request.enable_contact = True
    request.num_max_contacts = d336.EPA_MAX_CONTACTS
    result = hppfcl.CollisionResult()
    hppfcl.collide(geometry, transform, cylinder, cylinder_tf, request, result)
    contacts = []
    for index in range(result.numContacts()):
        contact = result.getContact(index)
        depth_m = abs(float(contact.penetration_depth))
        position = np.asarray(contact.pos, dtype=np.float64)
        normal = np.asarray(contact.normal, dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        normal_unit = normal / norm if math.isfinite(norm) and norm > 0.0 else normal
        contacts.append(
            {
                "index": index,
                "depth_m": depth_m,
                "depth_mm": depth_m * 1000.0,
                "position_m": position.tolist(),
                "normal_unit": normal_unit.tolist(),
                "finite": bool(
                    math.isfinite(depth_m)
                    and np.isfinite(position).all()
                    and np.isfinite(normal_unit).all()
                ),
            }
        )
    selected = max(contacts, key=lambda row: row["depth_m"]) if contacts else None
    return {
        "is_collision": bool(result.isCollision()),
        "num_contacts": int(result.numContacts()),
        "cap_saturated": bool(int(result.numContacts()) >= d336.EPA_MAX_CONTACTS),
        "max_abs_depth_mm": None if selected is None else selected["depth_mm"],
        "selected_contact": selected,
        "contacts_top8_desc": sorted(
            contacts, key=lambda row: row["depth_m"], reverse=True
        )[:8],
    }


def _union_distances(
    inner: Any,
    parts: dict[str, list[dict[str, Any]]],
    geometry_key: str,
    representation: str,
) -> dict[str, Any]:
    import hppfcl

    cylinder = hppfcl.Cylinder(d332.CYLINDER_RADIUS_M, d332.CYLINDER_HEIGHT_M)
    obj_pos, obj_quat = d334._object_pose_w(inner)
    cyl_tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(obj_quat), obj_pos)
    rows = []
    body_poses = {}
    for body in d334.BODY_LABELS:
        pos, quat = d334._body_pose_w(inner, body)
        body_poses[body] = {"pos_m": pos.tolist(), "quat_wxyz": quat.tolist()}
        body_tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(quat), pos)
        queries = []
        for part in parts[body]:
            geometry = part[geometry_key]
            query = d332._fcl_query(hppfcl, geometry, body_tf, cylinder, cyl_tf)
            epa = _epa_contacts_with_witness(
                hppfcl, geometry, body_tf, cylinder, cyl_tf
            )
            if query["is_collision"]:
                selected = epa["selected_contact"]
                selected_valid = bool(selected is not None and selected["finite"])
                exact = (
                    -float(epa["max_abs_depth_mm"])
                    if epa["num_contacts"] and selected_valid
                    else None
                )
                consistent = bool(
                    epa["is_collision"] and epa["num_contacts"] > 0 and selected_valid
                )
                if selected_valid:
                    witness_0 = np.asarray(selected["position_m"], dtype=np.float64)
                    witness_1 = witness_0 + np.asarray(
                        selected["normal_unit"], dtype=np.float64
                    ) * float(selected["depth_m"])
                else:
                    witness_0 = np.asarray(
                        query["nearest_point_geometry_m"], dtype=np.float64
                    )
                    witness_1 = np.asarray(
                        query["nearest_point_cylinder_m"], dtype=np.float64
                    )
                witness_kind = (
                    "epa_penetration" if selected_valid else "collision_inconsistent"
                )
            else:
                exact = float(query["signed_distance_mm"])
                consistent = bool(not epa["is_collision"])
                witness_0 = np.asarray(query["nearest_point_geometry_m"], dtype=np.float64)
                witness_1 = np.asarray(query["nearest_point_cylinder_m"], dtype=np.float64)
                witness_kind = "clear_separation"
            queries.append(
                {
                    "path": part["path"],
                    "is_collision": bool(query["is_collision"]),
                    "exact_signed_distance_mm": exact,
                    "exact_consistent": consistent,
                    "epa_contact_count": int(epa["num_contacts"]),
                    "epa_cap_saturated": bool(epa["cap_saturated"]),
                    "epa_selected_contact": epa["selected_contact"],
                    "nearest_point_geometry_m": query["nearest_point_geometry_m"],
                    "nearest_point_cylinder_m": query["nearest_point_cylinder_m"],
                    "witness_kind": witness_kind,
                    "witness_endpoint_0_m": witness_0.tolist(),
                    "witness_endpoint_1_m": witness_1.tolist(),
                }
            )
        collisions = [row for row in queries if row["is_collision"]]
        if collisions:
            eligible = [row for row in collisions if row["exact_signed_distance_mm"] is not None]
            witness = min(eligible or collisions, key=lambda row: float(row.get("exact_signed_distance_mm") or 0.0))
            exact_value = witness["exact_signed_distance_mm"]
            exact_consistent = bool(
                len(eligible) == len(collisions)
                and all(row["exact_consistent"] for row in collisions)
                and all(not row["epa_cap_saturated"] for row in collisions)
            )
            state = "overlap" if exact_consistent else "borderline"
        else:
            witness = min(queries, key=lambda row: float(row["exact_signed_distance_mm"]))
            exact_value = float(witness["exact_signed_distance_mm"])
            exact_consistent = bool(
                all(row["exact_consistent"] for row in queries)
                and all(not row["epa_cap_saturated"] for row in queries)
            )
            state = "clear" if exact_value >= CLEAR_GATE_MM else "borderline"
        rows.append(
            {
                "body": body,
                "representation": representation,
                "part_count": len(queries),
                "is_collision": bool(collisions),
                "exact_signed_distance_mm": exact_value,
                "exact_consistent": exact_consistent,
                "epa_cap_saturated_any": any(row["epa_cap_saturated"] for row in queries),
                "overlap_state": state,
                "clear_pass": bool(
                    not collisions
                    and exact_value is not None
                    and exact_value >= CLEAR_GATE_MM
                    and exact_consistent
                ),
                "distance_nearest_point_geometry_m": witness["nearest_point_geometry_m"],
                "distance_nearest_point_cylinder_m": witness["nearest_point_cylinder_m"],
                "nearest_point_geometry_m": witness["witness_endpoint_0_m"],
                "nearest_point_cylinder_m": witness["witness_endpoint_1_m"],
                "witness_kind": witness["witness_kind"],
                "witness_endpoint_0_m": witness["witness_endpoint_0_m"],
                "witness_endpoint_1_m": witness["witness_endpoint_1_m"],
                "witness_part_path": witness["path"],
                "parts": queries,
            }
        )
    return {
        "representation": representation,
        "object_pos_w_m": obj_pos.tolist(),
        "object_quat_wxyz": obj_quat.tolist(),
        "body_poses_w": body_poses,
        "queries": rows,
    }


def _diagnostic_union_distances(
    inner: Any,
    parts: dict[str, list[dict[str, Any]]],
    geometry_key: str,
    representation: str,
) -> dict[str, Any]:
    missing = [
        f"{part['body']}/{part['name']}"
        for body in d334.BODY_LABELS
        for part in parts[body]
        if part.get(geometry_key) is None
    ]
    if missing:
        return {
            "representation": representation,
            "available": False,
            "error": "diagnostic geometry unavailable",
            "missing_parts": missing,
            "verdict_authority": False,
            "queries": [],
        }
    try:
        result = _union_distances(inner, parts, geometry_key, representation)
        result.update({"available": True, "error": None, "verdict_authority": False})
        return result
    except Exception as error:  # diagnostics may never veto the authoritative gate
        return {
            "representation": representation,
            "available": False,
            "error": f"{type(error).__name__}: {error}",
            "missing_parts": [],
            "verdict_authority": False,
            "queries": [],
        }


def _by_body(distance_set: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["body"]: row for row in distance_set["queries"]}


def _target_state_guard(inner: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    commanded = candidate["_command"][0].detach().cpu().numpy().astype(np.float32)
    actual = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    object_pos, object_quat = d334._object_pose_w(inner)
    expected_pos = (
        inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float32)
        + np.asarray(d332.OBJECT_CENTER_LOCAL_M, dtype=np.float32)
    )
    q5_idx = list(inner._robot.joint_names).index(d332.GRIPPER_JOINT_NAME)
    checks = {
        "joint_float32_bit_exact": np.array_equal(actual, commanded),
        "q5_float32_bit_exact": actual[q5_idx].tobytes() == commanded[q5_idx].tobytes(),
        "q5_command_matches_frozen": commanded[q5_idx].tobytes()
        == np.float32(Q5_OPEN_RAD).tobytes(),
        "object_position_float32_bit_exact": np.array_equal(
            object_pos.astype(np.float32), expected_pos
        ),
        "object_quaternion_identity_float32": np.array_equal(
            object_quat.astype(np.float32), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ),
    }
    return {
        "commanded_joint_rad_float32": commanded.tolist(),
        "actual_joint_rad_float32": actual.tolist(),
        "q5_index": q5_idx,
        "object_pos_w_m": object_pos.tolist(),
        "expected_object_pos_w_float32_m": expected_pos.tolist(),
        "object_quat_wxyz": object_quat.tolist(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _distance_gate(
    candidate: dict[str, Any],
    raw_witness_repeat: dict[str, Any],
    controls: dict[str, Any],
    corrected_audit: dict[str, Any],
    binding: dict[str, Any],
    live_authority: dict[str, Any],
    live_convex_diagnostic: dict[str, Any],
    live_qhull_diagnostic: dict[str, Any],
    target_state: dict[str, Any],
) -> dict[str, Any]:
    raw = _by_body(candidate["_distance_set"])
    raw_repeat = _by_body(raw_witness_repeat)
    live = _by_body(live_authority)
    convex = (
        _by_body(live_convex_diagnostic)
        if live_convex_diagnostic.get("available")
        else {}
    )
    qhull = (
        _by_body(live_qhull_diagnostic)
        if live_qhull_diagnostic.get("available")
        else {}
    )
    per_body = {}
    for body in d334.BODY_LABELS:
        raw_value = raw[body]["exact_signed_distance_mm"]
        raw_repeat_value = raw_repeat[body]["exact_signed_distance_mm"]
        live_value = live[body]["exact_signed_distance_mm"]
        values = (raw_value, live_value)
        finite = all(value is not None and math.isfinite(float(value)) for value in values)
        raw_live_delta = abs(float(raw_value) - float(live_value)) if finite else math.inf
        raw_repeat_exact = bool(
            raw_value == raw_repeat_value
            and raw[body]["is_collision"] == raw_repeat[body]["is_collision"]
            and raw[body]["exact_consistent"] == raw_repeat[body]["exact_consistent"]
        )

        def _diagnostic_row(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
            row = rows.get(body)
            value = None if row is None else row.get("exact_signed_distance_mm")
            value_finite = bool(value is not None and math.isfinite(float(value)))
            return {
                "available": row is not None,
                "exact_signed_distance_mm": value,
                "is_collision": None if row is None else row.get("is_collision"),
                "delta_from_topology_surface_authority_mm": (
                    abs(float(value) - float(live_value))
                    if value_finite and finite
                    else None
                ),
                "reference_mm": DIAGNOSTIC_DISTANCE_DELTA_REFERENCE_MM,
                "verdict_authority": False,
            }

        diagnostics = {
            "callback_vertex_convex_support": _diagnostic_row(convex),
            "vertex_only_qhull": _diagnostic_row(qhull),
        }
        checks = {
            "authoritative_values_finite": finite,
            "raw_exact_consistent": bool(raw[body]["exact_consistent"]),
            "raw_epa_not_saturated": not bool(raw[body]["epa_cap_saturated"]),
            "raw_not_collision": not bool(raw[body]["is_collision"]),
            "raw_clear_ge_0p1mm": bool(finite and float(raw_value) >= CLEAR_GATE_MM),
            "raw_witness_repeat_exact": raw_repeat_exact,
            "live_exact_consistent": bool(live[body]["exact_consistent"]),
            "live_epa_not_saturated": not bool(live[body]["epa_cap_saturated_any"]),
            "live_not_collision": not bool(live[body]["is_collision"]),
            "live_clear_ge_0p1mm": bool(finite and float(live_value) >= CLEAR_GATE_MM),
            "raw_live_delta_le_0p5mm": raw_live_delta <= FIDELITY_TOL_MM,
            "live_part_count_64": live[body]["part_count"] == 64,
        }
        per_body[body] = {
            "raw_exact_signed_distance_mm": raw_value,
            "cooked_exact_signed_distance_mm": live_value,
            "live_topology_exact_signed_distance_mm": live_value,
            "live_authority": "D348 callback-topology triangle-surface BVH union",
            "raw_witness_repeat_exact_signed_distance_mm": raw_repeat_value,
            "absolute_delta_mm": raw_live_delta,
            "tolerance_mm": FIDELITY_TOL_MM,
            "diagnostics": diagnostics,
            "checks": checks,
            "pass": all(checks.values()),
        }
    anchor_checks = d339._frozen_anchor_checks(candidate)
    authoritative_pose_streams = {
        "raw_first": {
            "object_pos_w_m": candidate["_distance_set"]["object_pos_w_m"],
            "object_quat_wxyz": candidate["_distance_set"]["object_quat_wxyz"],
            "body_poses_w": candidate["_distance_set"]["body_poses_w"],
        },
        "raw_witness_repeat": {
            "object_pos_w_m": raw_witness_repeat["object_pos_w_m"],
            "object_quat_wxyz": raw_witness_repeat["object_quat_wxyz"],
            "body_poses_w": raw_witness_repeat["body_poses_w"],
        },
        "live_topology_surface": {
            "object_pos_w_m": live_authority["object_pos_w_m"],
            "object_quat_wxyz": live_authority["object_quat_wxyz"],
            "body_poses_w": live_authority["body_poses_w"],
        },
    }
    pose_stream_exact = (
        authoritative_pose_streams["raw_first"]
        == authoritative_pose_streams["raw_witness_repeat"]
        == authoritative_pose_streams["live_topology_surface"]
    )
    contract_checks = {
        "d337_controls": controls.get("pass") is True,
        "d348_corrected_audit_128": corrected_audit.get("pass") is True,
        "live_runtime_binding_128": binding.get("pass") is True,
        "frozen_anchor": all(anchor_checks.values()),
        "target_state_float32_exact": target_state.get("pass") is True,
        "authoritative_query_pose_streams_exact": pose_stream_exact,
    }
    return {
        "artifact": "D349_FROZEN_TARGET_DISTANCE_GATE_V1",
        "measurement_semantics": {
            "raw": "retained raw triangle BVH",
            "live_authority": "callback-topology triangle-surface BVH 64-part union",
            "live_authority_scope": (
                "D347 callback-face surface proxy for the active collider; not a direct "
                "PhysX narrowphase distance query"
            ),
            "distance_diagnostics_no_verdict_authority": [
                "callback-vertex Convex support mapping",
                "vertex-only Qhull support mapping",
            ],
            "direct_physx_narrowphase_distance_api": False,
        },
        "anchor_checks": anchor_checks,
        "authoritative_pose_streams": authoritative_pose_streams,
        "diagnostic_pose_streams": {
            "callback_vertex_convex_support": {
                "available": live_convex_diagnostic.get("available"),
                "matches_authority": bool(
                    live_convex_diagnostic.get("available")
                    and {
                        "object_pos_w_m": live_convex_diagnostic["object_pos_w_m"],
                        "object_quat_wxyz": live_convex_diagnostic["object_quat_wxyz"],
                        "body_poses_w": live_convex_diagnostic["body_poses_w"],
                    }
                    == authoritative_pose_streams["live_topology_surface"]
                ),
                "verdict_authority": False,
            },
            "vertex_only_qhull": {
                "available": live_qhull_diagnostic.get("available"),
                "matches_authority": bool(
                    live_qhull_diagnostic.get("available")
                    and {
                        "object_pos_w_m": live_qhull_diagnostic["object_pos_w_m"],
                        "object_quat_wxyz": live_qhull_diagnostic["object_quat_wxyz"],
                        "body_poses_w": live_qhull_diagnostic["body_poses_w"],
                    }
                    == authoritative_pose_streams["live_topology_surface"]
                ),
                "verdict_authority": False,
            },
        },
        "checks": contract_checks,
        "per_body": per_body,
        "contract_pass": all(contract_checks.values()),
        "target_clear_and_faithful": all(row["pass"] for row in per_body.values()),
        "prephysics_support_eligible": bool(
            all(contract_checks.values())
            and all(row["pass"] for row in per_body.values())
        ),
        "separate_settle_case_eligible_after_full_completion": False,
        "physics_licensed": False,
        "controlled_physics_steps": 0,
    }


def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if not key.startswith("_")}


def _rerun_rows(
    inner: Any,
    corrected_audit: dict[str, Any],
    gate: dict[str, Any],
    candidate: dict[str, Any],
    raw_set: dict[str, Any] | None,
    live_set: dict[str, Any] | None,
) -> tuple[list[Any], ...]:
    coordinate_frames, meshes, scalars, events = d344._rerun_rows(
        inner,
        _json(CORE_MANIFEST),
        d339._build_retained_raw_shapes(inner, _json(D334_SUMMARY))[0],
        corrected_audit,
        gate,
    )
    transformed_events = []
    for event in events:
        row = dict(event)
        if row.get("entity_path") == "events/d344":
            row["entity_path"] = "events/d349"
        row["text"] = str(row.get("text", "")).replace("D344", "D349")
        transformed_events.append(row)
    measured = bool(
        raw_set
        and live_set
        and len(raw_set.get("queries", [])) == 2
        and len(live_set.get("queries", [])) == 2
    )
    if not measured:
        raise RuntimeError("D349 Rerun requires both authoritative distance sets")
    points: list[dict[str, Any]] = []
    arrows: list[dict[str, Any]] = []
    origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    for representation, distance_set in (("source", raw_set), ("instance", live_set)):
        for row in distance_set["queries"]:
            body = row["body"]
            p0 = np.asarray(row["witness_endpoint_0_m"], dtype=np.float64) - origin
            p1 = np.asarray(row["witness_endpoint_1_m"], dtype=np.float64) - origin
            kind = str(row["witness_kind"])
            is_penetration = kind == "epa_penetration"
            base = f"cook/{representation}/{body}/distance_witness"
            base_color = [50, 190, 255] if representation == "source" else [255, 80, 190]
            if is_penetration:
                color = [245, 55, 45]
                label = "EPA max-depth penetration witness (display only)"
            elif kind == "clear_separation":
                color = base_color
                label = "clear separation witness (display only)"
            else:
                color = [245, 135, 35]
                label = "inconsistent collision witness; scientific STOP"
            points.append(
                {
                    "entity_path": f"{base}/endpoints",
                    "positions_m": [p0.tolist(), p1.tolist()],
                    "coordinate_frame": "tf#/",
                    "radii": [0.0025, 0.0025],
                    "colors": [color, [250, 185, 30]],
                    "labels": [f"{representation}:{body}:{kind}", "target cylinder"],
                    "static": True,
                }
            )
            arrows.append(
                {
                    "entity_path": f"{base}/vector",
                    "origins_m": [p0.tolist()],
                    "vectors_m": [(p1 - p0).tolist()],
                    "coordinate_frame": "tf#/",
                    "radii": [0.0012],
                    "colors": [color],
                    "labels": [label],
                    "static": True,
                }
            )
    body = gate["per_body"]

    def _ascii_mm(value: Any) -> str:
        if value is None:
            return "NA"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "NA"
        if math.isnan(number):
            return "NAN"
        if math.isinf(number):
            return "INF" if number > 0.0 else "-INF"
        return f"{number:.4f}"

    summary_rows = {
        "status": "OPEN q5=1.5413 | target=(7,11)mm | steps=0",
        "link5": (
            "L5 raw/live/delta="
            f"{_ascii_mm(body['link5']['raw_exact_signed_distance_mm'])}/"
            f"{_ascii_mm(body['link5']['live_topology_exact_signed_distance_mm'])}/"
            f"{_ascii_mm(body['link5']['absolute_delta_mm'])}mm"
        ),
        "gripper": (
            "GR raw/live/delta="
            f"{_ascii_mm(body['gripper_link']['raw_exact_signed_distance_mm'])}/"
            f"{_ascii_mm(body['gripper_link']['live_topology_exact_signed_distance_mm'])}/"
            f"{_ascii_mm(body['gripper_link']['absolute_delta_mm'])}mm"
        ),
        "scope": "gates=0.1/0.5mm | G0a=false | settle=separate",
    }
    for name, summary in summary_rows.items():
        transformed_events.append(
            {
                "entity_path": f"events/d349_summary/{name}",
                "text": summary,
                "level": "INFO",
                "static": True,
            }
        )
    return coordinate_frames, meshes, points, arrows, scalars, transformed_events


def _rerun_contract(
    frames: list[dict[str, Any]],
    coordinate_frames: list[dict[str, Any]],
    meshes: list[dict[str, Any]],
    points: list[dict[str, Any]],
    arrows: list[dict[str, Any]],
    scalars: list[dict[str, Any]],
    events: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if len(points) != 4 or len(arrows) != 4:
        raise RuntimeError("D349 requires four authoritative endpoint/vector witnesses")
    expected_counts = EXPECTED_RERUN_COUNTS
    entities, components = _expected_rrd_contract()
    log_status = log_rerun(
        RRD_PATH,
        frames=frames,
        coordinate_frames=coordinate_frames,
        meshes=meshes,
        points=points,
        arrows=arrows,
        scalar_trace=scalars,
        events=events,
        recording_metadata=metadata,
        recording_id="g0a_d349_frozen_open_jaw_target_distance",
        blueprint_path=RBL_PATH,
        blueprint_mode="collision_gate",
        live_viewer=False,
        app_id="roarm_g0a_collision_gate",
    )
    validation = (
        validate_rerun_artifact(
            RRD_PATH,
            expected_entity_paths=[
                "cook/source/link5/raw_reference",
                "cook/instance/gripper_link/parts/part_057",
                "cook/instance/link5/target/cylinder",
                "events/d349_summary/status",
            ],
            expected_timeline_names=["event_idx", "part_idx"],
            exact_entity_paths=entities,
            exact_timeline_names=["blueprint", "event_idx", "log_time", "part_idx"],
            expected_entity_components=components,
            blueprint_path=RBL_PATH,
            screenshot_path=RERUN_SCREENSHOT_PATH,
        )
        if log_status.get("ok")
        else {"pass": False, "errors": ["Rerun logging/finalization failed"]}
    )
    observed_counts = {
        "frame_count": len(frames),
        "coordinate_frame_count": len(coordinate_frames),
        "mesh_count": len(meshes),
        "point_entity_count": len(points),
        "arrow_entity_count": len(arrows),
        "scalar_row_count": len(scalars),
        "event_row_count": len(events),
        "exact_non_system_entity_count": len(
            validation.get("entity_path_contract", {}).get("observed_non_system", [])
        ),
    }
    count_checks = {
        name: observed_counts.get(name) == expected for name, expected in expected_counts.items()
    }
    log_count_checks = {
        "joint_trace_steps_zero": log_status.get("trace_steps") == 0,
        "coordinate_frame_count": log_status.get("coordinate_frame_count")
        == expected_counts["coordinate_frame_count"],
        "mesh_count": log_status.get("mesh_count") == expected_counts["mesh_count"],
        "point_entity_count": log_status.get("point_entity_count")
        == expected_counts["point_entity_count"],
        "arrow_entity_count": log_status.get("arrow_entity_count")
        == expected_counts["arrow_entity_count"],
        "scalar_row_count": log_status.get("scalar_row_count")
        == expected_counts["scalar_row_count"],
        "event_row_count": log_status.get("event_row_count")
        == expected_counts["event_row_count"],
    }
    report = {
        "artifact": "D349_RERUN_MACHINE_VALIDATION_V1",
        "profile": "MEASURED_AUTHORITY",
        "log_status": log_status,
        "archive_validation": validation,
        "observed_counts": observed_counts,
        "expected_counts": expected_counts,
        "count_checks": count_checks,
        "log_count_checks": log_count_checks,
        "rrd_contract_sha256": _rrd_contract_digest(),
        "pass": bool(
            log_status.get("ok")
            and validation.get("pass")
            and all(count_checks.values())
            and all(log_count_checks.values())
        ),
    }
    _write_json(RERUN_VALIDATION_PATH, report)
    return report


def _write_automated_report(summary: dict[str, Any]) -> None:
    gate = summary["distance_gate"]
    body = gate.get("per_body", {})
    lines = [
        "# D349 automated report",
        "",
        f"- scientific verdict: `{summary['scientific_verdict']}`",
        f"- automated pass: `{summary['automated_pass']}`",
        f"- physics steps: `{summary['controlled_physics_steps']}`",
        f"- Rerun machine pass: `{summary['rerun']['pass']}`",
    ]
    for name in d334.BODY_LABELS:
        row = body.get(name, {})
        lines.append(
            f"- {name}: raw/live/delta = `{row.get('raw_exact_signed_distance_mm')}` / "
            f"`{row.get('live_topology_exact_signed_distance_mm')}` / "
            f"`{row.get('absolute_delta_mm')}` mm"
        )
    lines.extend(["", "`g0a_pass=false`; settle requires a separate approved case.", ""])
    _write_text(AUTOMATED_REPORT_PATH, "\n".join(lines))


def _run_validate(args: argparse.Namespace, simulation_app: Any, resolved_launcher: dict[str, Any]) -> int:
    prereg = _json(PREREG_PATH)
    before_inventories = _source_inventories()
    before_external = _external_baseline_report()
    args.robot_usd_path = VARIANT_ROBOT_USD
    inner = d333._make_runtime_env(args)
    try:
        inner.reset(seed=SEED)
        counters = [{"phase": "after_reset", "counter": int(inner._sim_step_counter)}]
        home = _home_start_contract(inner)
        _write_json(HOME_PATH, home)
        stage_contract = d333._stage_contract(inner)
        sensor_contract, _ = d333._sensor_contract(inner)
        from pxr import UsdGeom

        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(inner.scene.stage))
        raw_shapes, raw_contract = d339._build_retained_raw_shapes(inner, _json(D334_SUMMARY))
        corrected_audit = _corrected_live_audit()
        _write_json(CORRECTED_AUDIT_PATH, corrected_audit)
        topology_parts, binding = _build_live_topology_parts(inner)
        _write_json(BINDING_PATH, binding)
        runtime_prerequisites = {
            "home_start": home["pass"],
            "stage_contract": stage_contract["hard_contract_pass"],
            "sensor_contract": sensor_contract["hard_contract_pass"],
            "meters_per_unit_one": math.isclose(meters_per_unit, 1.0, rel_tol=0.0, abs_tol=0.0),
            "raw_source_contract": raw_contract["pass"],
            "d348_corrected_live_audit": corrected_audit["pass"],
            "live_topology_runtime_binding": binding["pass"],
            "resolved_app_launcher": resolved_launcher.get("pass") is True,
        }
        if not all(runtime_prerequisites.values()):
            raise RuntimeError(f"D349 runtime prerequisite STOP: {runtime_prerequisites}")
        counters.append({"phase": "before_d337_controls", "counter": int(inner._sim_step_counter)})
        cache = d337._Cache(inner, raw_shapes)
        controls = d337._negative_controls(
            inner,
            raw_shapes,
            _json(D334_SUMMARY),
            _json(D336_SUMMARY),
            d337._load_d336_rescore(),
            cache,
        )
        _write_json(CONTROLS_PATH, controls)
        counters.append({"phase": "after_d337_controls", "counter": int(inner._sim_step_counter)})
        if not controls["pass"]:
            raise RuntimeError("D349 D337 frozen controls failed; target query forbidden")
        counters.append({"phase": "before_target_exact_write", "counter": int(inner._sim_step_counter)})
        candidate = d337._evaluate_candidate(
            inner,
            raw_shapes,
            RADIAL_NM,
            TANGENT_NM,
            Q5_OPEN_RAD,
            stage="d349_frozen_open_jaw_raw_first",
        )
        counters.append({"phase": "after_raw_target_query", "counter": int(inner._sim_step_counter)})
        raw_set = candidate["_distance_set"]
        target_state = _target_state_guard(inner, candidate)
        raw_parts = {body: [] for body in d334.BODY_LABELS}
        for shape in raw_shapes:
            raw_parts[shape["body"]].append(
                {
                    "body": shape["body"],
                    "name": "retained_raw_mesh",
                    "path": shape["collider_path"],
                    "_geometry_raw": shape["_geom_raw"],
                }
            )
        raw_witness_repeat = _union_distances(
            inner, raw_parts, "_geometry_raw", "retained_raw_triangle_bvh_witness_repeat"
        )
        counters.append(
            {"phase": "after_raw_witness_repeat", "counter": int(inner._sim_step_counter)}
        )
        live_authority = _union_distances(
            inner,
            topology_parts,
            "_geometry_topology_surface_authority",
            "live_callback_topology_surface_bvh_authority",
        )
        counters.append(
            {"phase": "after_live_authority_query", "counter": int(inner._sim_step_counter)}
        )
        live_convex_diagnostic = _diagnostic_union_distances(
            inner,
            topology_parts,
            "_geometry_volume_witness_convex",
            "live_callback_vertex_convex_support_diagnostic",
        )
        live_qhull_diagnostic = _diagnostic_union_distances(
            inner,
            topology_parts,
            "_geometry_qhull_diagnostic",
            "live_vertex_qhull_diagnostic",
        )
        counters.append(
            {"phase": "after_non_authoritative_diagnostics", "counter": int(inner._sim_step_counter)}
        )
        gate = _distance_gate(
            candidate,
            raw_witness_repeat,
            controls,
            corrected_audit,
            binding,
            live_authority,
            live_convex_diagnostic,
            live_qhull_diagnostic,
            target_state,
        )
        counter_values = [row["counter"] for row in counters]
        counter_unchanged = len(set(counter_values)) == 1 and counter_values[0] == 0
        gate["global_sim_counter"] = {
            "start": counter_values[0],
            "end": counter_values[-1],
            "delta": counter_values[-1] - counter_values[0],
            "unchanged": counter_unchanged,
        }
        gate["checks"]["global_sim_counter_zero_unchanged"] = counter_unchanged
        gate["contract_pass"] = bool(gate["contract_pass"] and counter_unchanged)
        gate["prephysics_support_eligible"] = bool(
            gate["contract_pass"]
            and gate["target_clear_and_faithful"]
        )
        gate["controlled_physics_steps"] = 0 if counter_unchanged else max(counter_values[-1], 0)
        measurement = {
            "artifact": "D349_FROZEN_OPEN_JAW_TARGET_DISTANCE_MEASUREMENT_V1",
            "case": "g0a_d349",
            "new_variables": NEW_VARIABLES,
            "measurement_semantics": gate["measurement_semantics"],
            "query_order": [
                "raw_mesh_first",
                "raw_witness_repeat",
                "live_callback_topology_surface_authority",
                "non_authoritative_convex_and_qhull_diagnostics",
            ],
            "execution_order": counters,
            "home_start_contract": home,
            "runtime_prerequisites": runtime_prerequisites,
            "target_contract": {
                "radial_offset_mm": RADIAL_NM / 1.0e6,
                "tangent_offset_mm": TANGENT_NM / 1.0e6,
                "q5_rad": Q5_OPEN_RAD,
                "tangent_sign": d332.ADOPTED_TANGENT_SIGN,
                "seed": SEED,
                "ik": "HOME-seeded position-only",
            },
            "target_state_guard": target_state,
            "d337_controls": controls,
            "frozen_candidate": _public_candidate(candidate),
            "frozen_candidate_alignment": candidate["_alignment"],
            "raw_mesh": raw_set,
            "raw_witness_repeat": raw_witness_repeat,
            "live_topology_surface_authority": live_authority,
            "live_callback_vertex_convex_support_diagnostic": live_convex_diagnostic,
            "live_vertex_qhull_diagnostic": live_qhull_diagnostic,
            "distance_gate": gate,
            "scope_guards": {
                "asset_write": False,
                "cook_callback_requests": 0,
                "physx_property_queries": 0,
                "controlled_physics_steps": gate["controlled_physics_steps"],
                "physics_licensed": False,
                "settle_executed": False,
                "ten_trial_run": False,
                "g0b_run": False,
                "rl_run": False,
                "ladder_promoted": False,
                "g0a_pass": False,
            },
        }
        _write_json(MEASUREMENT_PATH, measurement)
        try:
            d339._write_representation_figure(
                DECISION_PNG,
                "D349 frozen OPEN target: raw vs callback-topology live union",
                inner,
                raw_shapes,
                topology_parts,
                raw_witness_repeat,
                live_authority,
                candidate["_canonical"],
            )
            decision_ok = DECISION_PNG.is_file() and DECISION_PNG.stat().st_size > 0
            decision_error = None
        except Exception as error:
            decision_ok = False
            decision_error = f"{type(error).__name__}: {error}"
        try:
            markers = draw_frames(candidate["_frames"], prim_path="/World/D349TargetFrames")
        except Exception as error:
            markers = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        coordinate_frames, meshes, points, arrows, scalars, events = _rerun_rows(
            inner,
            corrected_audit,
            gate,
            candidate,
            raw_witness_repeat,
            live_authority,
        )
        rerun = _rerun_contract(
            candidate["_frames"],
            coordinate_frames,
            meshes,
            points,
            arrows,
            scalars,
            events,
            {
                "case": "g0a_d349",
                "purpose": "frozen OPEN raw/live callback-topology target-distance gate",
                "git_head": _git_head(),
                "new_variables": NEW_VARIABLES,
                "q5": "1.5413 OPEN",
                "target": "radial=7mm tangent=11mm sign=-1",
                "physics": "forbidden; controlled steps 0",
                "scientific_authority": "Float64 JSON and immutable callback topology/hashes",
                "viewer_role": "Float32 one-way observability copy",
            },
        )
        after_inventories = _source_inventories()
        after_external = _external_baseline_report()
        immutability = {
            "source_before": before_inventories,
            "source_after": after_inventories,
            "source_exact": before_inventories == after_inventories == prereg["source_inventories"],
            "external_before": before_external,
            "external_after": after_external,
            "external_exact": before_external == after_external and after_external["pass"],
        }
        immutability["pass"] = immutability["source_exact"] and immutability["external_exact"]
        if not gate["contract_pass"]:
            scientific_verdict = VERDICT_PREREQUISITE
        elif not gate["target_clear_and_faithful"]:
            scientific_verdict = VERDICT_TARGET_FAIL
        else:
            scientific_verdict = VERDICT_PENDING
        artifact_checks = {
            "decision_png": decision_ok,
            "frame_markers": bool(markers.get("ok")),
            "rerun_machine": rerun["pass"],
            "immutability": immutability["pass"],
        }
        summary = {
            "artifact": "D349_AUTOMATED_SUMMARY_V1",
            "case": "g0a_d349",
            "scientific_verdict": scientific_verdict,
            "observability_verdict": (
                "D349_RERUN_MACHINE_CONTRACT_PASS_MANUAL_INSPECTION_PENDING"
                if all(artifact_checks.values())
                else VERDICT_OBSERVABILITY
            ),
            "automated_pass": bool(
                scientific_verdict == VERDICT_PENDING and all(artifact_checks.values())
            ),
            "manual_visual_inspection_pending": bool(all(artifact_checks.values())),
            "new_variables": NEW_VARIABLES,
            "new_physical_variables": [],
            "runtime_prerequisites": runtime_prerequisites,
            "home_start_contract": home,
            "target_state_guard": target_state,
            "distance_gate": gate,
            "controlled_physics_steps": gate["controlled_physics_steps"],
            "visualization": {
                "decision_png": _relative(DECISION_PNG) if DECISION_PNG.is_file() else None,
                "decision_png_sha256": _sha256(DECISION_PNG) if DECISION_PNG.is_file() else None,
                "decision_png_error": decision_error,
                "frame_markers": markers,
            },
            "rerun": rerun,
            "immutability": immutability,
            "artifact_checks": artifact_checks,
            "environment": {
                "numpy": str(np.__version__),
                "psutil": str(psutil.__version__),
                "rerun": str(rr.__version__),
                "python": str(Path(sys.executable).resolve()),
                "resolved_app_launcher": resolved_launcher,
            },
            "single_execution_contract": {
                "scientific_execution_count": 1,
                "validate_preflight_path": _relative(PREFLIGHT_PATH),
                "validate_preflight_sha256": _sha256(PREFLIGHT_PATH),
            },
            "outcome_guards": {
                "g0a_pass": False,
                "settle_executed": False,
                "ten_trial_run": False,
                "g0b_run": False,
                "rl_run": False,
                "ladder_promoted": False,
                "separate_settle_case_requires_new_approval": scientific_verdict
                == VERDICT_PENDING,
            },
        }
        _write_json(AUTOMATED_SUMMARY_PATH, summary)
        _write_automated_report(summary)
        print(
            json.dumps(
                {
                    "stage": "validate",
                    "scientific_verdict": scientific_verdict,
                    "automated_pass": summary["automated_pass"],
                },
                sort_keys=True,
            )
        )
        return 0 if summary["automated_pass"] else 2
    finally:
        inner.close()


def _manual_checks(manual: dict[str, Any]) -> dict[str, bool]:
    screenshot = manual.get("rerun_screenshot", {})
    decision = manual.get("decision_png", {})
    observations = manual.get("observations", {})
    required = (
        "eight_spatial_panels_visible_and_nonempty",
        "raw_mesh_visible_for_both_bodies",
        "live_64_part_union_visible_for_both_bodies",
        "target_cylinder_visible",
        "target_and_actual_frames_visible",
        "four_distance_endpoint_pairs_visible",
        "four_distance_vectors_visible",
        "ascii_summary_distances_and_deltas_legible",
        "ascii_summary_thresholds_open_q5_zero_steps_g0a_false_legible",
        "prototype_and_candidate_panels_independent",
        "required_content_not_obscured",
        "decision_png_opened_and_legible",
    )
    method = str(manual.get("inspection_method", "")).lower()
    return {
        "artifact": manual.get("artifact") == "D349_RERUN_MANUAL_VISUAL_INSPECTION_V1",
        "case": manual.get("case") == "g0a_d349",
        "inspection_date": manual.get("inspection_date") == "2026-07-14",
        "screenshot_path": screenshot.get("path") == _relative(RERUN_SCREENSHOT_PATH),
        "screenshot_sha": RERUN_SCREENSHOT_PATH.is_file()
        and screenshot.get("sha256") == _sha256(RERUN_SCREENSHOT_PATH),
        "screenshot_bytes": RERUN_SCREENSHOT_PATH.is_file()
        and screenshot.get("bytes") == RERUN_SCREENSHOT_PATH.stat().st_size,
        "screenshot_dimensions": screenshot.get("raster_dimensions")
        == _png_dimensions(RERUN_SCREENSHOT_PATH),
        "decision_path": decision.get("path") == _relative(DECISION_PNG),
        "decision_sha": DECISION_PNG.is_file()
        and decision.get("sha256") == _sha256(DECISION_PNG),
        "decision_bytes": DECISION_PNG.is_file()
        and decision.get("bytes") == DECISION_PNG.stat().st_size,
        "decision_dimensions": decision.get("raster_dimensions") == _png_dimensions(DECISION_PNG),
        "inspection_method": "view_image" in method and "original" in method,
        "observations": all(observations.get(name) is True for name in required),
        "bounded_interpretation": len(manual.get("bounded_interpretation", [])) >= 3,
        "manual_pass": manual.get("manual_visual_inspection_pass") is True,
        "scientific_override_false": manual.get("scientific_verdict_override") is False,
        "g0a_false": manual.get("g0a_pass") is False,
        "settle_false": manual.get("settle_executed") is False,
        "markdown_nonzero": MANUAL_MD_PATH.is_file() and MANUAL_MD_PATH.stat().st_size > 0,
    }


def _run_finalize(_args: argparse.Namespace) -> int:
    if COMPLETION_PATH.exists() or COMPLETION_MD_PATH.exists():
        raise RuntimeError("D349 completion already exists")
    automated = _json(AUTOMATED_SUMMARY_PATH)
    rerun = _json(RERUN_VALIDATION_PATH)
    manual = _json(MANUAL_PATH)
    prereg = _json(PREREG_PATH)
    preflight = _json(PREFLIGHT_PATH)
    manual_checks = _manual_checks(manual)
    current_inventories = _source_inventories()
    input_checks = {
        "head": _git_head() == prereg["git_head"] == EXPECTED_HEAD,
        "status_scope": _status_scope_pass(_git_status()),
        "critical_hashes": _critical_hashes() == EXPECTED_CRITICAL_HASHES,
        "state_hashes": {
            "start_here": _sha256(START_HERE),
            "session_doc": _sha256(SESSION_DOC),
        }
        == prereg.get("state_hashes"),
        "harness_hash": _sha256(HARNESS) == prereg.get("harness_sha256"),
        "preregistration_hash": _sha256(PREREG_PATH)
        == preflight.get("preregistration_sha256"),
        "parameter_audit_hash": _sha256(PARAMETER_AUDIT_PATH)
        == prereg.get("parameter_audit_sha256")
        == preflight.get("parameter_audit_sha256"),
        "source_inventories": current_inventories == prereg["source_inventories"],
        "external_baseline": _external_baseline_report()["pass"],
        "preflight_hash": _sha256(PREFLIGHT_PATH)
        == automated["single_execution_contract"]["validate_preflight_sha256"],
        "rrd_hash": RRD_PATH.is_file()
        and _sha256(RRD_PATH) == rerun["archive_validation"].get("sha256"),
        "rbl_hash": RBL_PATH.is_file()
        and _sha256(RBL_PATH)
        == rerun["archive_validation"].get("blueprint_verify", {}).get("sha256"),
        "screenshot_hash": RERUN_SCREENSHOT_PATH.is_file()
        and _sha256(RERUN_SCREENSHOT_PATH)
        == rerun["archive_validation"].get("headless_render", {}).get("sha256"),
        "decision_hash": DECISION_PNG.is_file()
        and _sha256(DECISION_PNG) == automated["visualization"]["decision_png_sha256"],
        "rerun_report_exact_to_automated": rerun == automated.get("rerun"),
        "rerun_profile_registered": rerun.get("profile")
        == prereg.get("rerun_contract", {}).get("profile"),
        "rerun_digest_registered": rerun.get("rrd_contract_sha256")
        == prereg.get("rerun_contract", {}).get("contract_sha256"),
    }
    completion_pass = bool(
        automated["automated_pass"]
        and automated["scientific_verdict"] == VERDICT_PENDING
        and rerun["pass"]
        and all(manual_checks.values())
        and all(input_checks.values())
        and automated["controlled_physics_steps"] == 0
    )
    if completion_pass:
        final_verdict = VERDICT_COMPLETE
    elif automated["scientific_verdict"] in {
        VERDICT_PREREQUISITE,
        VERDICT_TARGET_FAIL,
    }:
        final_verdict = automated["scientific_verdict"]
    else:
        final_verdict = VERDICT_OBSERVABILITY
    completion = {
        "artifact": "D349_COMPLETION_SUMMARY_V1",
        "case": "g0a_d349",
        "final_verdict": final_verdict,
        "completion_contract_pass": completion_pass,
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "automated_evidence": {
            "path": _relative(AUTOMATED_SUMMARY_PATH),
            "sha256": _sha256(AUTOMATED_SUMMARY_PATH),
            "scientific_verdict": automated["scientific_verdict"],
        },
        "measurement_evidence": {
            "path": _relative(MEASUREMENT_PATH),
            "sha256": _sha256(MEASUREMENT_PATH),
            "per_body": automated["distance_gate"]["per_body"],
            "live_authority": automated["distance_gate"]["measurement_semantics"][
                "live_authority"
            ],
            "diagnostics_have_verdict_authority": False,
        },
        "rerun_evidence": {
            "validation_path": _relative(RERUN_VALIDATION_PATH),
            "validation_sha256": _sha256(RERUN_VALIDATION_PATH),
            "rrd_path": _relative(RRD_PATH),
            "rrd_sha256": _sha256(RRD_PATH) if RRD_PATH.is_file() else None,
            "rbl_path": _relative(RBL_PATH),
            "rbl_sha256": _sha256(RBL_PATH) if RBL_PATH.is_file() else None,
            "screenshot_path": _relative(RERUN_SCREENSHOT_PATH),
            "screenshot_sha256": (
                _sha256(RERUN_SCREENSHOT_PATH)
                if RERUN_SCREENSHOT_PATH.is_file()
                else None
            ),
            "machine_contract_pass": rerun["pass"],
            "profile": rerun["profile"],
        },
        "manual_evidence": {
            "json_path": _relative(MANUAL_PATH),
            "json_sha256": _sha256(MANUAL_PATH) if MANUAL_PATH.is_file() else None,
            "markdown_path": _relative(MANUAL_MD_PATH),
            "markdown_sha256": _sha256(MANUAL_MD_PATH) if MANUAL_MD_PATH.is_file() else None,
            "checks": manual_checks,
            "pass": all(manual_checks.values()),
        },
        "input_checks": input_checks,
        "scope_guards": {
            "controlled_physics_steps": automated["controlled_physics_steps"],
            "asset_write": False,
            "cook_callback_requests": 0,
            "physx_property_queries": 0,
            "settle_executed": False,
            "ten_trial_run": False,
            "g0b_run": False,
            "rl_run": False,
            "ladder_promoted": False,
            "g0a_pass": False,
        },
        "separate_settle_case_eligible": completion_pass,
        "next_case_requires_separate_approval": "settle evaluation" if completion_pass else None,
        "interpretation": (
            "D349 certifies only the zero-step frozen OPEN target distance of the D347 "
            "callback-face surface proxy when complete; it is not a direct PhysX "
            "narrowphase distance result and never certifies settle, grasp, G0a, G0b, "
            "RL, or ladder promotion."
        ),
    }
    _write_json(COMPLETION_PATH, completion)
    body = automated["distance_gate"]["per_body"]
    _write_text(
        COMPLETION_MD_PATH,
        "# D349 완료 보고\n\n"
        f"- 최종 판정: `{final_verdict}`\n"
        f"- 완료 계약: `{completion_pass}`\n"
        f"- link5 raw/live/delta: `{body['link5']['raw_exact_signed_distance_mm']}` / "
        f"`{body['link5']['live_topology_exact_signed_distance_mm']}` / "
        f"`{body['link5']['absolute_delta_mm']}` mm\n"
        f"- gripper raw/live/delta: `{body['gripper_link']['raw_exact_signed_distance_mm']}` / "
        f"`{body['gripper_link']['live_topology_exact_signed_distance_mm']}` / "
        f"`{body['gripper_link']['absolute_delta_mm']}` mm\n"
        f"- 물리 step: `{automated['controlled_physics_steps']}`\n"
        "- `g0a_pass=false`\n\n"
        "PASS해도 settle은 사용자의 별도 승인 case다.\n",
    )
    print(json.dumps({"stage": "finalize", "final_verdict": final_verdict}, sort_keys=True))
    return 0 if completion_pass else 2


def _parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "validate", "finalize"), required=True)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--urdf_path", type=Path, default=d333.DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=SEED)
    if stage == "validate":
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    if "--print-rerun-contract" in sys.argv:
        print(
            json.dumps(
                {
                    "MEASURED_AUTHORITY": {
                        "counts": EXPECTED_RERUN_COUNTS,
                        "digest": _rrd_contract_digest(),
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    stage_probe = argparse.ArgumentParser(add_help=False)
    stage_probe.add_argument("--stage", choices=("prepare", "validate", "finalize"), required=True)
    stage_args, _ = stage_probe.parse_known_args()
    args = _parser(stage_args.stage).parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D349 output path is forward-only and fixed")
    if Path(args.urdf_path).resolve() != Path(d333.DEFAULT_URDF).resolve():
        raise RuntimeError("D349 URDF path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D349 seed drift")
    args.process_nonce = secrets.token_hex(16)
    if args.stage == "prepare":
        return _run_prepare(args)
    if args.stage == "finalize":
        return _run_finalize(args)
    if not all(_app_launcher_checks(args).values()):
        raise RuntimeError(f"D349 AppLauncher drift: {_app_launcher_checks(args)}")
    if not _prepare_validate_preflight(args):
        return 2

    from isaaclab.app import AppLauncher

    args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    launcher = AppLauncher(copy.deepcopy(args))
    simulation_app = launcher.app
    resolved_launcher = d347._resolved_launcher_report(launcher)

    try:
        return _run_validate(args, simulation_app, resolved_launcher)
    except Exception as error:
        if not RUNTIME_EXCEPTION_PATH.exists():
            _write_json(
                RUNTIME_EXCEPTION_PATH,
                {
                    "artifact": "D349_RUNTIME_EXCEPTION_STOP",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "controlled_physics_steps": None,
                    "zero_step_claim_withheld_after_exception": True,
                    "g0a_pass": False,
                },
            )
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
