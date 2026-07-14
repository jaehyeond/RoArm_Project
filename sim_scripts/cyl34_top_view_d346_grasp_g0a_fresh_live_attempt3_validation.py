#!/usr/bin/env python3
"""D346 fresh-process validation of immutable D344 attempt3.

This is a validation-only case.  It never builds, copies, rewrites, or recooks a
collision asset on disk.  A fresh Isaac process records 256 independent PhysX
callback witnesses for 128 active convex pieces, applies the repaired D342/D343
classification contract, conditionally evaluates the frozen open-jaw target,
and completes the D341 Rerun lifecycle.  Controlled physics, settle, trials,
G0b, RL, and ladder promotion are forbidden.
"""
from __future__ import annotations

import argparse
import atexit
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
    cyl34_top_view_d335_grasp_g0a_target_family_repair as d335,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d336_grasp_g0a_finite_grid_caveat_discriminator as d336,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d337_grasp_g0a_open_jaw_target_gate as d337,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d338_grasp_g0a_collision_representation_repair as d338,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair as d339,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d340_grasp_g0a_fixed_point_live_authoring_repair as d340,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d344_grasp_g0a_attempt3_fixed_point_collision_geometry as d344,
)


OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d346"
PREREG_PATH = OUT_DIR / "d346_preregistration.json"
PARAMETER_AUDIT_PATH = OUT_DIR / "d346_parameter_freeze_audit.json"
ELIGIBILITY_PATH = OUT_DIR / "d346_effective_eligibility.json"
VALIDATE_PREFLIGHT_PATH = OUT_DIR / "d346_validate_preflight.json"
REACTIVE_AMENDMENT_PATH = OUT_DIR / "d346_gpu_visibility_reactive_amendment.json"
EFFECTIVE_VALIDATE_PREFLIGHT_PATH = OUT_DIR / "d346_validate_preflight_effective.json"
RAW_LIVE_MEASUREMENT_PATH = OUT_DIR / "d346_raw_live_measurement.json"
LIVE_AUDIT_PATH = OUT_DIR / "d346_fresh_live_representation_audit.json"
REPRESENTATION_GATE_PATH = OUT_DIR / "d346_zero_step_representation_gate.json"
CONTROLS_PATH = OUT_DIR / "d346_d337_frozen_controls.json"
WITNESS_DIR = OUT_DIR / "d346_validate_cook_witnesses"
WITNESS_MANIFEST_PATH = OUT_DIR / "d346_validate_cook_witness_manifest.json"
DECISION_PNG = OUT_DIR / "d346_fresh_live_representation_decision.png"
RRD_PATH = OUT_DIR / "d346_attempt3_live_representation.rrd"
RBL_PATH = OUT_DIR / "d346_attempt3_live_representation.rbl"
RERUN_SCREENSHOT_PATH = OUT_DIR / "d346_attempt3_live_representation_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d346_rerun_validation.json"
AUTOMATED_SUMMARY_PATH = OUT_DIR / "d346_automated_summary.json"
AUTOMATED_REPORT_PATH = OUT_DIR / "d346_automated_report.md"
RUNTIME_EXCEPTION_PATH = OUT_DIR / "d346_runtime_exception.json"
EFFECTIVE_RUNTIME_EXCEPTION_PATH = OUT_DIR / "d346_runtime_exception_effective.json"
MANUAL_INSPECTION_PATH = OUT_DIR / "d346_manual_visual_inspection.json"
MANUAL_INSPECTION_MD_PATH = OUT_DIR / "d346_manual_visual_inspection.md"
COMPLETION_SUMMARY_PATH = OUT_DIR / "d346_completion_summary.json"
COMPLETION_REPORT_PATH = OUT_DIR / "d346_completion_report.md"

D344_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d344"
D345_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d345"
ATTEMPT3_DIR = D344_DIR / "collision_asset/attempt3"
VARIANT_DIR = ATTEMPT3_DIR / "roarm_m3_fullmesh_fixed_point_parts"
VARIANT_ROBOT_USD = VARIANT_DIR / "roarm_m3.usd"
VARIANT_PHYSICS_USD = VARIANT_DIR / "configuration/roarm_m3_physics.usd"
D344_BUILD_SUMMARY = D344_DIR / "d344_attempt3_build_summary.json"
D344_OUTER_MANIFEST = ATTEMPT3_DIR / "d344_attempt3_asset_manifest.json"
D344_CORE_MANIFEST = ATTEMPT3_DIR / "d340_attempt3_asset_manifest.json"
D344_ROOT_CAUSE = D344_DIR / "d344_postrun_root_cause_audit.json"
D345_SUMMARY = D345_DIR / "d345_deterministic_usd_metadata_summary.json"
D345_EVIDENCE = D345_DIR / "d345_deterministic_usd_metadata_evidence.json"
D340_CANDIDATES = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d340/"
    "d340_capture_fixed_point_candidates.json"
)
D334_SUMMARY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d334/"
    "g0a_d334_live_collision_audit_summary.json"
)
D336_SUMMARY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d336/"
    "g0a_d336_finite_grid_caveat_summary.json"
)
START_HERE = REPO / "START_HERE.md"
SESSION_DOC = (
    REPO
    / "claudedocs/session_20260714_grasp_g0a_d346_fresh_live_attempt3_validation.md"
)

NEW_VARIABLES = ["attempt3_fresh_live_representation_validation"]
DECOMPOSITION_PARAMS = copy.deepcopy(d339.DECOMPOSITION_PARAMS)
Q5_OPEN_RAD = d339.Q5_OPEN_RAD
OLD_RADIAL_NM = d339.OLD_RADIAL_NM
OLD_TANGENT_NM = d339.OLD_TANGENT_NM
SEED = 33201
EXPECTED_MIN_THICKNESS_BITS = 0x38D1B717
EXPECTED_MIN_THICKNESS_HEX = "0x38d1b717"
FIXED_POINT_TOL_M = 1.0e-9
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
APP_LAUNCHER_SOURCE = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/"
    "source/isaaclab/isaaclab/app/app_launcher.py"
)

EXPECTED_HEAD = "b09b62e0ffad919b9bdc1bb6155de2f662f2ab5c"
EXPECTED_D345_SUMMARY_SHA = (
    "d7cd4d4b0cb4c5a010b8673b47cb010103c337642b6e9b33df1fe577ca73bba5"
)
EXPECTED_D345_EVIDENCE_SHA = (
    "68652b51c7a0667a63c5d4b1e812e43868af07a93b4b91838322f81ac4cb4379"
)
EXPECTED_D345_CANONICAL_SHA = (
    "3f85d121439060ef5c6deb49cab7860dbc72eb94e23e54617c4ac2b1f7cdcd09"
)
EXPECTED_ATTEMPT3_INVENTORY_DIGEST = (
    "ea6965199ff1f195a6d19d9c55febfe44cc9838f12651570c80d5bb97fa6caf1"
)
EXPECTED_D344_INVENTORY_DIGEST = (
    "bd07acc09a8c89f4f8da78da7e9492546a8f2974864e3e42c9069a211195a3c8"
)
EXPECTED_D345_INVENTORY_DIGEST = (
    "b859f9ad44de22cd9bb93e3f16bb3ee669721e3c788c45ec366c27cdcb4bb147"
)
EXPECTED_D339_ATTEMPT2_DIGEST = (
    "0dae41fd3937a0a8aea18488019c74f097d32f7b8de916943ff31334e30464a1"
)
EXPECTED_VARIANT_ROOT_SHA = (
    "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff"
)
EXPECTED_VARIANT_PHYSICS_SHA = (
    "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503"
)
EXPECTED_OUTER_MANIFEST_SHA = (
    "c318f7ef9e7f384db3e3f68785917bc58c731d21d8047d74dbc8d21bdd53985f"
)
EXPECTED_CORE_MANIFEST_SHA = (
    "52bc2e40f586cd8401aa0fcabf23e5e733776b9c373082bd49fbda01190c2bb5"
)
ATTEMPT0_HARNESS_SHA = "d609a717cb96247fcddd58460fa164784395c0caf14c07aea2d3dc4aa5df55a3"
ATTEMPT0_PREFLIGHT_SHA = "e0759345507dbf48df80a490cb4c9da05383658c4903a1a6a07beb940529515f"
ATTEMPT0_EXCEPTION_SHA = "21ecca9860192601b2b813a2bee43c51552470f490f346a7676b7f6aa4e9b624"

VERDICT_LIVE_FAIL = "D346_G0A_FRESH_LIVE_REPRESENTATION_FAIL_STOP"
VERDICT_PREREQUISITE_FAIL = "D346_G0A_RUNTIME_PREREQUISITE_CONTRACT_FAIL_STOP"
VERDICT_TARGET_FAIL = "D346_G0A_COOKED_TARGET_FIDELITY_FAIL_STOP"
VERDICT_PREPHYSICS_PENDING = (
    "D346_G0A_PREPHYSICS_COLLISION_REPRESENTATION_SUPPORTED_"
    "MANUAL_INSPECTION_PENDING"
)
VERDICT_OBSERVABILITY_FAIL = "D346_RERUN_OBSERVABILITY_INCOMPLETE_STOP"
VERDICT_COMPLETE = "D346_G0A_PREPHYSICS_COLLISION_REPRESENTATION_SUPPORTED"

EXPECTED_RERUN_COUNTS = {
    "frame_count": 6,
    "coordinate_frame_count": 2,
    "mesh_count": 522,
    "scalar_row_count": 1040,
    "event_row_count": 132,
    "exact_non_system_entity_count": 2100,
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


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_status_paths() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[str] = []
    for line in completed.stdout.splitlines():
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value)
    return sorted(paths)


def _status_scope_pass(paths: list[str]) -> bool:
    exact = {
        "START_HERE.md",
        _relative(SESSION_DOC),
        _relative(Path(__file__).resolve()),
    }
    output_prefix = _relative(OUT_DIR) + "/"
    return all(path in exact or path.startswith(output_prefix) for path in paths)


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": _relative(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _inventory_digest(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_inventories() -> dict[str, dict[str, Any]]:
    roots = {
        "d338_attempt1": REPO
        / "claudedocs/runtime_logs/grasp_track/g0a_d338/collision_asset/attempt1",
        "d339_attempt2": REPO
        / "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2",
        "d340": REPO / "claudedocs/runtime_logs/grasp_track/g0a_d340",
        "d342": REPO / "claudedocs/runtime_logs/grasp_track/g0a_d342",
        "d343": REPO / "claudedocs/runtime_logs/grasp_track/g0a_d343",
        "d344_attempt3": ATTEMPT3_DIR,
        "d344_all": D344_DIR,
        "d345_all": D345_DIR,
    }
    result: dict[str, dict[str, Any]] = {}
    for name, root in roots.items():
        rows = _inventory(root)
        result[name] = {
            "root": _relative(root),
            "file_count": len(rows),
            "digest": _inventory_digest(rows),
            "rows": rows,
        }
    return result


def _source_hashes() -> dict[str, str]:
    paths = {
        "d346_harness": Path(__file__).resolve(),
        "d346_session": SESSION_DOC,
        "start_here": START_HERE,
        "d332_harness": Path(d332.__file__).resolve(),
        "d333_harness": Path(d333.__file__).resolve(),
        "d334_harness": Path(d334.__file__).resolve(),
        "d335_harness": Path(d335.__file__).resolve(),
        "d336_harness": Path(d336.__file__).resolve(),
        "d344_harness": Path(d344.__file__).resolve(),
        "d340_harness": Path(d340.__file__).resolve(),
        "d339_harness": Path(d339.__file__).resolve(),
        "d338_harness": Path(d338.__file__).resolve(),
        "d337_harness": Path(d337.__file__).resolve(),
        "frozen_urdf": Path(d333.DEFAULT_URDF).resolve(),
        "isaaclab_app_launcher": APP_LAUNCHER_SOURCE,
        "d344_build_summary": D344_BUILD_SUMMARY,
        "d344_outer_manifest": D344_OUTER_MANIFEST,
        "d344_core_manifest": D344_CORE_MANIFEST,
        "d344_root_cause": D344_ROOT_CAUSE,
        "d345_summary": D345_SUMMARY,
        "d345_evidence": D345_EVIDENCE,
        "d340_candidates": D340_CANDIDATES,
        "d334_summary": D334_SUMMARY,
        "d336_summary": D336_SUMMARY,
        "viz_debug": REPO / "roarm_rl/viz_debug.py",
        "rerun_contract": REPO / "roarm_rl/rerun_contract.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _f32_bits(value: float | np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _expected_rrd_contract() -> tuple[list[str], dict[str, list[str]]]:
    d344_entities, d344_components = d344._expected_rrd_contract()
    entities = ["events/d346" if path == "events/d344" else path for path in d344_entities]
    components = {
        ("events/d346" if path == "events/d344" else path): list(values)
        for path, values in d344_components.items()
    }
    return sorted(entities), components


def _rrd_contract_digest() -> str:
    entities, components = _expected_rrd_contract()
    payload = {
        "exact_non_system_entity_paths": entities,
        "exact_timeline_names": ["blueprint", "event_idx", "log_time", "part_idx"],
        "required_components_by_path": components,
        "exact_observation_counts": EXPECTED_RERUN_COUNTS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _png_dimensions(path: Path) -> str | None:
    if not path.is_file():
        return None
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return f"{width}x{height}"


def _parameter_checks() -> dict[str, bool]:
    return {
        "one_measurement_variable": len(NEW_VARIABLES) == 1,
        "decomposition_exact_d339": DECOMPOSITION_PARAMS == d339.DECOMPOSITION_PARAMS,
        "decomposition_exact_d338_helper": DECOMPOSITION_PARAMS
        == d338.DECOMPOSITION_PARAMS,
        "decomposition_exact_d340_helper": DECOMPOSITION_PARAMS
        == d340.DECOMPOSITION_PARAMS,
        "decomposition_exact_d344_helper": DECOMPOSITION_PARAMS
        == d344.DECOMPOSITION_PARAMS,
        "hull_vertex_limit_64": DECOMPOSITION_PARAMS["hull_vertex_limit"] == 64,
        "max_convex_hulls_64": DECOMPOSITION_PARAMS["max_convex_hulls"] == 64,
        "voxel_resolution_one_million": DECOMPOSITION_PARAMS["voxel_resolution"]
        == 1_000_000,
        "error_percentage_one": DECOMPOSITION_PARAMS["error_percentage"] == 1.0,
        "min_thickness_requested_0p0001m": DECOMPOSITION_PARAMS["min_thickness_m"]
        == 0.0001,
        "min_thickness_float32_bits_exact": _f32_bits(
            DECOMPOSITION_PARAMS["min_thickness_m"]
        )
        == EXPECTED_MIN_THICKNESS_BITS,
        "shrink_wrap_true": DECOMPOSITION_PARAMS["shrink_wrap"] is True,
        "surface_tolerance_0p1mm": d339.LIVE_SURFACE_PARITY_TOL_M == 0.0001,
        "property_volume_tolerance_5pct": d339.PROPERTY_VOLUME_BINDING_REL_TOL
        == 0.05,
        "target_clear_gate_0p1mm": d339.CLEAR_GATE_MM == 0.1,
        "target_delta_gate_0p5mm": d339.TASK_FIDELITY_TOL_MM == 0.5,
        "raw_anchor_tolerance_0p05mm": d339.RAW_ANCHOR_TOL_MM == 0.05,
        "d337_parity_tolerance_0p05mm": d337.PARITY_TOL_MM == 0.05,
        "d337_scoping_open_tolerance_0p5mm": d337.SCOPING_OPEN_TOL_MM == 0.5,
        "d337_link5_invariant_tolerance_0p05mm": d337.LINK5_INVARIANT_TOL_MM
        == 0.05,
        "fixed_point_tolerance_1e_9m": FIXED_POINT_TOL_M == 1.0e-9,
        "fixed_point_tolerance_exact_d339_helper": d339.COLD_COOK_COORD_TOL_M
        == FIXED_POINT_TOL_M,
        "fixed_point_tolerance_exact_d340_helper": d340.FIXED_POINT_COORD_TOL_M
        == FIXED_POINT_TOL_M,
        "fixed_point_tolerance_exact_d344_helper": d344.FIXED_POINT_TOL_M
        == FIXED_POINT_TOL_M,
        "typed_bits_exact_d344_helper": d344.EXPECTED_MIN_THICKNESS_BITS
        == EXPECTED_MIN_THICKNESS_BITS,
        "typed_bits_hex_exact_d344_helper": d344.EXPECTED_MIN_THICKNESS_HEX
        == EXPECTED_MIN_THICKNESS_HEX,
        "compatibility_tolerance_exact_d344_helper": d344.FROZEN_COMPATIBILITY_TOL_M
        == 1.0e-10,
        "q5_open_1p5413rad": math.isclose(
            Q5_OPEN_RAD, 1.5413, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "target_radial_7mm": OLD_RADIAL_NM == 7_000_000,
        "target_tangent_11mm": OLD_TANGENT_NM == 11_000_000,
        "target_tangent_sign_minus_one": d332.ADOPTED_TANGENT_SIGN == -1.0,
        "seed_33201": SEED == 33201,
        "cylinder_radius_17mm": d332.CYLINDER_RADIUS_M == 0.017,
        "cylinder_height_90mm": d332.CYLINDER_HEIGHT_M == 0.090,
        "object_mass_0p72kg": d332.OBJECT_MASS_KG == 0.72,
        "static_friction_1p5": d332.STATIC_FRICTION == 1.5,
        "dynamic_friction_1p2": d332.DYNAMIC_FRICTION == 1.2,
    }


def _app_launcher_checks(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "headless_true": getattr(args, "headless", None) is True,
        "livestream_zero": getattr(args, "livestream", None) == 0,
        "cameras_disabled": getattr(args, "enable_cameras", None) is False,
        "xr_disabled": getattr(args, "xr", None) is False,
        "device_cuda_zero": getattr(args, "device", None) == "cuda:0",
        "cpu_flag_false": getattr(args, "cpu", None) is False,
        "experience_default_empty": getattr(args, "experience", None) == "",
        "kit_args_empty": getattr(args, "kit_args", None) == "",
        "rendering_mode_default": getattr(args, "rendering_mode", None) is None,
        "verbose_disabled": getattr(args, "verbose", None) is False,
        "info_disabled": getattr(args, "info", None) is False,
        "animation_recording_disabled": getattr(args, "anim_recording_enabled", None)
        is False,
    }


def _resolved_launcher_report(launcher: Any) -> dict[str, Any]:
    values = {
        "headless": bool(launcher._headless),
        "livestream": int(launcher._livestream),
        "enable_cameras": bool(launcher._enable_cameras),
        "xr": bool(launcher._xr),
        "offscreen_render": bool(launcher._offscreen_render),
        "device_id": int(launcher.device_id),
        "experience": str(launcher._sim_experience_file),
    }
    checks = {
        "headless_true": values["headless"] is True,
        "livestream_zero": values["livestream"] == 0,
        "cameras_disabled": values["enable_cameras"] is False,
        "xr_disabled": values["xr"] is False,
        "offscreen_render_disabled": values["offscreen_render"] is False,
        "device_id_zero": values["device_id"] == 0,
        "resolved_headless_experience": Path(values["experience"]).name
        == "isaaclab.python.headless.kit",
    }
    return {"values": values, "checks": checks, "pass": all(checks.values())}


def _effective_eligibility() -> dict[str, Any]:
    d344_summary = _json(D344_BUILD_SUMMARY)
    outer = _json(D344_OUTER_MANIFEST)
    core = _json(D344_CORE_MANIFEST)
    d345_summary = _json(D345_SUMMARY)
    d345_evidence = _json(D345_EVIDENCE)
    inventories = _source_inventories()
    core_false = sorted(name for name, passed in core["checks"].items() if not passed)
    outer_false = sorted(name for name, passed in outer["checks"].items() if not passed)
    checks = {
        "d344_historical_fail_retained": d344_summary.get("verdict")
        == "D344_G0A_ATTEMPT3_AUTHORING_CONTRACT_FAIL_STOP"
        and d344_summary.get("pass") is False,
        "d344_core_only_address_comparator_gate_false": core_false
        == ["whole_physics_semantic_allowlist_exact"],
        "d344_outer_only_core_gate_false": outer_false == ["core_build_pass"],
        "d344_all_128_part_audits_pass": len(core.get("part_audits", [])) == 128
        and all(row.get("pass") is True for row in core["part_audits"]),
        "d344_exact_13_changed": core.get("changed_part_count") == 13,
        "d344_exact_115_preserved": core.get("preserved_part_count") == 115,
        "d344_typed_scalar_128_pass": outer.get("checks", {}).get(
            "typed_scalar_128_pass"
        )
        is True,
        "d345_summary_pass": d345_summary.get("pass") is True,
        "d345_verdict_exact": d345_summary.get("verdict")
        == "D345_DETERMINISTIC_USD_METADATA_COMPARATOR_PASS",
        "d345_canonical_hash_exact": d345_summary.get("canonical_sha256")
        == EXPECTED_D345_CANONICAL_SHA,
        "d345_four_way_exact": d345_summary.get("four_way_canonical_hash_exact")
        is True,
        "d345_immutability_pass": d345_summary.get("immutability_pass") is True,
        "d345_evidence_pass": d345_evidence.get("pass") is True,
        "d345_summary_sha_exact": _sha256(D345_SUMMARY)
        == EXPECTED_D345_SUMMARY_SHA,
        "d345_evidence_sha_exact": _sha256(D345_EVIDENCE)
        == EXPECTED_D345_EVIDENCE_SHA,
        "attempt3_inventory_9_exact": inventories["d344_attempt3"]["file_count"] == 9,
        "attempt3_inventory_digest_exact": inventories["d344_attempt3"]["digest"]
        == EXPECTED_ATTEMPT3_INVENTORY_DIGEST,
        "d344_inventory_19_exact": inventories["d344_all"]["file_count"] == 19,
        "d344_inventory_digest_exact": inventories["d344_all"]["digest"]
        == EXPECTED_D344_INVENTORY_DIGEST,
        "d345_inventory_8_exact": inventories["d345_all"]["file_count"] == 8,
        "d345_inventory_digest_exact": inventories["d345_all"]["digest"]
        == EXPECTED_D345_INVENTORY_DIGEST,
        "d339_attempt2_inventory_18_exact": inventories["d339_attempt2"]["file_count"]
        == 18,
        "d339_attempt2_digest_exact": inventories["d339_attempt2"]["digest"]
        == EXPECTED_D339_ATTEMPT2_DIGEST,
        "variant_root_usd_hash_exact": _sha256(VARIANT_ROBOT_USD)
        == EXPECTED_VARIANT_ROOT_SHA,
        "variant_physics_usd_hash_exact": _sha256(VARIANT_PHYSICS_USD)
        == EXPECTED_VARIANT_PHYSICS_SHA,
        "outer_manifest_hash_exact": _sha256(D344_OUTER_MANIFEST)
        == EXPECTED_OUTER_MANIFEST_SHA,
        "core_manifest_hash_exact": _sha256(D344_CORE_MANIFEST)
        == EXPECTED_CORE_MANIFEST_SHA,
        "core_manifest_decomposition_exact": core.get("decomposition_parameters")
        == DECOMPOSITION_PARAMS,
        "outer_manifest_decomposition_exact": outer.get("decomposition_parameters")
        == DECOMPOSITION_PARAMS,
    }
    return {
        "artifact": "D346_EFFECTIVE_ELIGIBILITY_V1",
        "historical_d344_verdict_retained": d344_summary.get("verdict"),
        "d345_verdict": d345_summary.get("verdict"),
        "core_false_checks": core_false,
        "outer_false_checks": outer_false,
        "critical_inventories": {
            name: inventories[name]
            for name in ("d339_attempt2", "d344_attempt3", "d344_all", "d345_all")
        },
        "checks": checks,
        "pass": all(checks.values()),
        "interpretation": (
            "D344 remains historical FAIL. D345 independently licenses only a fresh "
            "read-only live validation of the unchanged attempt3 asset."
        ),
    }


def _run_prepare(args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"D346 output already exists; refusing prepare rerun: {OUT_DIR}")
    if _git_head() != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected HEAD before D346 prepare: {_git_head()}")
    if not _status_scope_pass(_git_status_paths()):
        raise RuntimeError("git status contains paths outside the registered D346 scope")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    eligibility = _effective_eligibility()
    _write_json(ELIGIBILITY_PATH, eligibility)
    parameter_checks = _parameter_checks()
    parameter_audit = {
        "artifact": "D346_PARAMETER_FREEZE_AUDIT_V1",
        "new_variables": NEW_VARIABLES,
        "new_variable_count": len(NEW_VARIABLES),
        "new_physical_variables": [],
        "existing_parameters_increased": [],
        "existing_parameters_changed": [],
        "thresholds_relaxed": [],
        "decomposition_parameters": DECOMPOSITION_PARAMS,
        "target": {
            "q5_rad": Q5_OPEN_RAD,
            "radial_offset_mm": OLD_RADIAL_NM / 1.0e6,
            "tangent_offset_mm": OLD_TANGENT_NM / 1.0e6,
            "tangent_sign": d332.ADOPTED_TANGENT_SIGN,
            "seed": SEED,
            "ik": "HOME-seeded position-only",
        },
        "tolerances": {
            "live_surface_m": d339.LIVE_SURFACE_PARITY_TOL_M,
            "property_volume_relative": d339.PROPERTY_VOLUME_BINDING_REL_TOL,
            "target_clear_mm": d339.CLEAR_GATE_MM,
            "raw_live_delta_mm": d339.TASK_FIDELITY_TOL_MM,
            "fixed_point_m": FIXED_POINT_TOL_M,
            "min_thickness_bits": EXPECTED_MIN_THICKNESS_HEX,
        },
        "scene_constants": {
            "cylinder_radius_m": d332.CYLINDER_RADIUS_M,
            "cylinder_height_m": d332.CYLINDER_HEIGHT_M,
            "mass_kg": d332.OBJECT_MASS_KG,
            "static_friction": d332.STATIC_FRICTION,
            "dynamic_friction": d332.DYNAMIC_FRICTION,
            "restitution": 0.0,
        },
        "measurement_only_runtime_operations": [
            "cache-isolated synchronous prototype/instance cook requests",
            "local and runtime mesh cache release before each request",
            "temporary cooking instrumentation settings restored after every request",
            "asset-validator extension enable if absent",
        ],
        "registered_app_launcher": {
            "headless": True,
            "livestream": 0,
            "enable_cameras": False,
            "xr": False,
            "device": "cuda:0",
            "experience": "",
            "kit_args": "",
            "rendering_mode": None,
            "animation_recording": False,
        },
        "checks": parameter_checks,
        "pass": all(parameter_checks.values()),
    }
    _write_json(PARAMETER_AUDIT_PATH, parameter_audit)
    inventories = _source_inventories()
    exact_entities, _components = _expected_rrd_contract()
    prereg = {
        "artifact": "D346_PREREGISTRATION_V1",
        "case": "g0a_d346",
        "git_head": _git_head(),
        "prepare_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "user_authorization": {
            "authorized": True,
            "source": "current user turn: D346진행해",
            "scope": "fresh live validation of immutable D344 attempt3",
        },
        "registered_scientific_execution_count": 1,
        "registered_stages": ["prepare", "validate", "manual_inspection", "finalize"],
        "new_variables": NEW_VARIABLES,
        "new_variable_count": 1,
        "new_physical_variables": [],
        "source_hashes": _source_hashes(),
        "source_inventory_counts": {
            name: row["file_count"] for name, row in inventories.items()
        },
        "source_inventory_digests": {
            name: row["digest"] for name, row in inventories.items()
        },
        "source_inventories": inventories,
        "eligibility_path": _relative(ELIGIBILITY_PATH),
        "eligibility_sha256": _sha256(ELIGIBILITY_PATH),
        "parameter_audit_path": _relative(PARAMETER_AUDIT_PATH),
        "parameter_audit_sha256": _sha256(PARAMETER_AUDIT_PATH),
        "callback_contract": {
            "active_parts": 128,
            "channels": ["prototype", "instance"],
            "request_order": ["prototype", "instance"],
            "requests": 256,
            "callbacks_per_request": 1,
            "convexes_per_callback": 1,
            "result": "RESULT_VALID(0)",
            "serialization_errors": 0,
            "witness_persisted_before_classification": True,
        },
        "rerun_contract_sha256": _rrd_contract_digest(),
        "rerun_subject_counts": EXPECTED_RERUN_COUNTS,
        "exact_non_system_entity_count": len(exact_entities),
        "exact_timelines": ["blueprint", "event_idx", "log_time", "part_idx"],
        "scope_guards": {
            "collision_asset_build": False,
            "collision_asset_write": False,
            "recook_to_disk": False,
            "controlled_physics_steps": 0,
            "settle": False,
            "ten_trial": False,
            "g0b": False,
            "rl": False,
            "ladder_promotion": False,
            "g0a_pass": False,
        },
        "stop_order": [
            "preflight before Isaac launch",
            "256 callback witnesses",
            "128-part corrected classification",
            "conditional frozen target query",
            "zero controlled physics",
            "Rerun machine validation",
            "actual screenshot inspection",
        ],
        "pass": bool(eligibility["pass"] and parameter_audit["pass"]),
    }
    _write_json(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"]}, sort_keys=True))
    return 0 if prereg["pass"] else 2


def _run_reactive_amendment(args: argparse.Namespace) -> int:
    if REACTIVE_AMENDMENT_PATH.exists():
        raise RuntimeError("D346 reactive amendment already exists; refusing overwrite")
    prereg = _json(PREREG_PATH)
    attempt0_preflight = _json(VALIDATE_PREFLIGHT_PATH)
    attempt0_exception = _json(RUNTIME_EXCEPTION_PATH)
    current_hashes = _source_hashes()
    original_hashes = prereg.get("source_hashes", {})
    current_inventories = _source_inventories()
    unchanged_source_checks = {
        name: current_hashes.get(name) == expected
        for name, expected in original_hashes.items()
        if name != "d346_harness"
    }
    scientific_outputs = (
        RAW_LIVE_MEASUREMENT_PATH,
        LIVE_AUDIT_PATH,
        REPRESENTATION_GATE_PATH,
        CONTROLS_PATH,
        WITNESS_DIR,
        WITNESS_MANIFEST_PATH,
        DECISION_PNG,
        RRD_PATH,
        RBL_PATH,
        RERUN_SCREENSHOT_PATH,
        RERUN_VALIDATION_PATH,
        AUTOMATED_SUMMARY_PATH,
        AUTOMATED_REPORT_PATH,
    )
    gpu_probe = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    checks = {
        "original_preregistration_pass": prereg.get("pass") is True,
        "attempt0_preflight_pass": attempt0_preflight.get("pass") is True,
        "attempt0_preflight_hash_exact": _sha256(VALIDATE_PREFLIGHT_PATH)
        == ATTEMPT0_PREFLIGHT_SHA,
        "attempt0_exception_hash_exact": _sha256(RUNTIME_EXCEPTION_PATH)
        == ATTEMPT0_EXCEPTION_SHA,
        "attempt0_exception_no_cuda_exact": attempt0_exception.get("error")
        == "RuntimeError: No CUDA GPUs are available",
        "attempt0_harness_hash_was_registered": original_hashes.get("d346_harness")
        == ATTEMPT0_HARNESS_SHA,
        "current_harness_is_reactive_revision": current_hashes.get("d346_harness")
        != ATTEMPT0_HARNESS_SHA,
        "all_non_harness_sources_unchanged": all(unchanged_source_checks.values()),
        "source_inventory_counts_still_exact": {
            name: row["file_count"] for name, row in current_inventories.items()
        }
        == prereg.get("source_inventory_counts"),
        "source_inventory_digests_still_exact": {
            name: row["digest"] for name, row in current_inventories.items()
        }
        == prereg.get("source_inventory_digests"),
        "no_callback_or_scientific_outputs": not any(path.exists() for path in scientific_outputs),
        "host_gpu_probe_pass": gpu_probe.returncode == 0
        and "RTX 4090" in gpu_probe.stdout,
        "git_head_unchanged": _git_head() == prereg.get("git_head") == EXPECTED_HEAD,
        "git_status_scope_only": _status_scope_pass(_git_status_paths()),
    }
    amendment = {
        "artifact": "D346_GPU_VISIBILITY_REACTIVE_AMENDMENT_V1",
        "case": "g0a_d346",
        "process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "reason": (
            "The managed command sandbox hid CUDA from Isaac after preflight. The run "
            "stopped while constructing SimulationContext, before env return/reset, callback, "
            "target query, Rerun, or controlled physics."
        ),
        "attempt0_classification": {
            "scientific_execution": False,
            "entered_callback_harness": False,
            "callback_count": 0,
            "controlled_physics_steps": 0,
            "asset_write": False,
            "parameter_change": False,
            "eligible_as_effective_execution": False,
            "preflight_path": _relative(VALIDATE_PREFLIGHT_PATH),
            "preflight_sha256": _sha256(VALIDATE_PREFLIGHT_PATH),
            "exception_path": _relative(RUNTIME_EXCEPTION_PATH),
            "exception_sha256": _sha256(RUNTIME_EXCEPTION_PATH),
            "validate_process_identity": attempt0_preflight.get("validate_process_identity"),
        },
        "reactive_change": (
            "Preserve attempt0 evidence and permit exactly one effective validate preflight/output "
            "path; require torch CUDA visibility before launching Isaac and run outside the "
            "device-hiding sandbox. Scientific variables and thresholds are unchanged."
        ),
        "retry_contract": {
            "effective_scientific_execution_count": 1,
            "requires_unsandboxed_gpu_visibility": True,
            "preflight_path": _relative(EFFECTIVE_VALIDATE_PREFLIGHT_PATH),
            "runtime_exception_path": _relative(EFFECTIVE_RUNTIME_EXCEPTION_PATH),
            "same_callback_target_rerun_contract": True,
            "new_variables": NEW_VARIABLES,
            "new_physical_variables": [],
            "parameters_increased": [],
            "parameters_changed": [],
            "thresholds_relaxed": [],
        },
        "host_gpu_probe": {
            "command": [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            "returncode": gpu_probe.returncode,
            "stdout": gpu_probe.stdout,
            "stderr": gpu_probe.stderr,
        },
        "original_source_hashes": original_hashes,
        "amended_source_hashes": current_hashes,
        "source_inventories": current_inventories,
        "unchanged_non_harness_source_checks": unchanged_source_checks,
        "checks": checks,
        "pass": all(checks.values()),
        "g0a_pass": False,
    }
    _write_json(REACTIVE_AMENDMENT_PATH, amendment)
    print(json.dumps({"stage": "amend", "pass": amendment["pass"]}, sort_keys=True))
    return 0 if amendment["pass"] else 2


def _runtime_outputs_absent() -> bool:
    paths = (
        EFFECTIVE_VALIDATE_PREFLIGHT_PATH,
        RAW_LIVE_MEASUREMENT_PATH,
        LIVE_AUDIT_PATH,
        REPRESENTATION_GATE_PATH,
        CONTROLS_PATH,
        WITNESS_DIR,
        WITNESS_MANIFEST_PATH,
        DECISION_PNG,
        RRD_PATH,
        RBL_PATH,
        RERUN_SCREENSHOT_PATH,
        RERUN_VALIDATION_PATH,
        AUTOMATED_SUMMARY_PATH,
        AUTOMATED_REPORT_PATH,
        EFFECTIVE_RUNTIME_EXCEPTION_PATH,
        MANUAL_INSPECTION_PATH,
        MANUAL_INSPECTION_MD_PATH,
        COMPLETION_SUMMARY_PATH,
        COMPLETION_REPORT_PATH,
    )
    return not any(path.exists() for path in paths)


def _prepare_validate_preflight(args: argparse.Namespace) -> bool:
    prereg = _json(PREREG_PATH) if PREREG_PATH.is_file() else {}
    amendment = _json(REACTIVE_AMENDMENT_PATH) if REACTIVE_AMENDMENT_PATH.is_file() else {}
    attempt0_preflight = _json(VALIDATE_PREFLIGHT_PATH) if VALIDATE_PREFLIGHT_PATH.is_file() else {}
    parameter_audit = _json(PARAMETER_AUDIT_PATH) if PARAMETER_AUDIT_PATH.is_file() else {}
    eligibility = _effective_eligibility()
    recorded_eligibility = _json(ELIGIBILITY_PATH) if ELIGIBILITY_PATH.is_file() else {}
    inventories = _source_inventories()
    status_paths = _git_status_paths()
    source_hashes = _source_hashes()
    try:
        import torch

        torch_cuda_available = bool(torch.cuda.is_available())
        torch_cuda_device_name = (
            str(torch.cuda.get_device_name(0)) if torch_cuda_available else ""
        )
    except Exception:
        torch_cuda_available = False
        torch_cuda_device_name = ""
    expected_retry_contract = {
        "effective_scientific_execution_count": 1,
        "requires_unsandboxed_gpu_visibility": True,
        "preflight_path": _relative(EFFECTIVE_VALIDATE_PREFLIGHT_PATH),
        "runtime_exception_path": _relative(EFFECTIVE_RUNTIME_EXCEPTION_PATH),
        "same_callback_target_rerun_contract": True,
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "parameters_increased": [],
        "parameters_changed": [],
        "thresholds_relaxed": [],
    }
    checks = {
        "preregistration_pass": prereg.get("pass") is True,
        "user_authorized": prereg.get("user_authorization", {}).get("authorized") is True,
        "one_registered_scientific_execution": prereg.get(
            "registered_scientific_execution_count"
        )
        == 1,
        "new_variables_exact": prereg.get("new_variables") == NEW_VARIABLES,
        "new_physical_variables_empty": prereg.get("new_physical_variables") == [],
        "git_head_exact": prereg.get("git_head") == _git_head() == EXPECTED_HEAD,
        "git_status_scope_only": _status_scope_pass(status_paths),
        "reactive_amendment_pass": amendment.get("pass") is True,
        "attempt0_non_scientific": amendment.get("attempt0_classification", {}).get(
            "scientific_execution"
        )
        is False,
        "attempt0_preflight_hash_still_exact": _sha256(VALIDATE_PREFLIGHT_PATH)
        == ATTEMPT0_PREFLIGHT_SHA,
        "attempt0_exception_hash_still_exact": _sha256(RUNTIME_EXCEPTION_PATH)
        == ATTEMPT0_EXCEPTION_SHA,
        "amendment_retry_contract_exact": amendment.get("retry_contract")
        == expected_retry_contract,
        "amendment_retry_count_exact": amendment.get("retry_contract", {}).get(
            "effective_scientific_execution_count"
        )
        == 1,
        "amendment_requires_gpu_visibility": amendment.get(
            "retry_contract", {}
        ).get("requires_unsandboxed_gpu_visibility")
        is True,
        "amendment_same_scientific_contract": amendment.get(
            "retry_contract", {}
        ).get("same_callback_target_rerun_contract")
        is True,
        "amendment_no_parameter_drift": amendment.get("retry_contract", {}).get(
            "new_physical_variables"
        )
        == []
        and amendment.get("retry_contract", {}).get("parameters_increased") == []
        and amendment.get("retry_contract", {}).get("parameters_changed") == []
        and amendment.get("retry_contract", {}).get("thresholds_relaxed") == [],
        "source_hashes_exact": amendment.get("amended_source_hashes") == source_hashes,
        "source_inventory_counts_exact": prereg.get("source_inventory_counts")
        == {name: row["file_count"] for name, row in inventories.items()},
        "source_inventory_digests_exact": prereg.get("source_inventory_digests")
        == {name: row["digest"] for name, row in inventories.items()},
        "eligibility_recomputed_pass": eligibility.get("pass") is True,
        "eligibility_record_exact": recorded_eligibility == eligibility,
        "eligibility_hash_exact": prereg.get("eligibility_sha256")
        == _sha256(ELIGIBILITY_PATH),
        "parameter_audit_pass": parameter_audit.get("pass") is True,
        "parameter_checks_runtime_pass": all(_parameter_checks().values()),
        "parameter_audit_hash_exact": prereg.get("parameter_audit_sha256")
        == _sha256(PARAMETER_AUDIT_PATH),
        "parameters_increased_empty": parameter_audit.get(
            "existing_parameters_increased"
        )
        == [],
        "parameters_changed_empty": parameter_audit.get("existing_parameters_changed")
        == [],
        "thresholds_relaxed_empty": parameter_audit.get("thresholds_relaxed") == [],
        "fresh_process_pid": prereg.get("prepare_process_identity", {}).get("pid")
        != os.getpid(),
        "fresh_process_nonce": prereg.get("prepare_process_identity", {}).get("nonce")
        != args.process_nonce,
        "fresh_from_attempt0_pid": attempt0_preflight.get(
            "validate_process_identity", {}
        ).get("pid")
        != os.getpid(),
        "fresh_from_attempt0_nonce": attempt0_preflight.get(
            "validate_process_identity", {}
        ).get("nonce")
        != args.process_nonce,
        "fresh_from_amendment_pid": amendment.get("process_identity", {}).get("pid")
        != os.getpid(),
        "fresh_from_amendment_nonce": amendment.get("process_identity", {}).get("nonce")
        != args.process_nonce,
        "runtime_outputs_absent": _runtime_outputs_absent(),
        "variant_robot_exists": VARIANT_ROBOT_USD.is_file(),
        "variant_physics_exists": VARIANT_PHYSICS_USD.is_file(),
        "numpy_pin": str(np.__version__) == "1.26.0",
        "psutil_pin": str(psutil.__version__) == "5.9.8",
        "rerun_sdk_pin": str(rr.__version__) == RERUN_CONTRACT_VERSION == "0.34.1",
        "registered_python_executable": str(Path(sys.executable).resolve())
        == str(Path(REGISTERED_PYTHON).resolve()),
        "torch_cuda_available_before_isaac": torch_cuda_available,
        "torch_cuda_device_zero_4090": "4090" in torch_cuda_device_name,
        "app_launcher_arguments_frozen": all(_app_launcher_checks(args).values()),
        "enable_cameras_environment_zero": os.environ.get("ENABLE_CAMERAS", "0")
        == "0",
        "xr_environment_zero": os.environ.get("XR", "0") == "0",
        "frozen_urdf_path_exact": Path(args.urdf_path).resolve()
        == Path(d333.DEFAULT_URDF).resolve(),
        "rrd_contract_digest_exact": prereg.get("rerun_contract_sha256")
        == _rrd_contract_digest(),
        "rerun_counts_exact": prereg.get("rerun_subject_counts")
        == EXPECTED_RERUN_COUNTS,
        "physics_forbidden": prereg.get("scope_guards", {}).get(
            "controlled_physics_steps"
        )
        == 0,
        "settle_forbidden": prereg.get("scope_guards", {}).get("settle") is False,
        "g0a_stays_false": prereg.get("scope_guards", {}).get("g0a_pass") is False,
    }
    report = {
        "artifact": "D346_VALIDATE_PREFLIGHT_EFFECTIVE_V1",
        "case": "g0a_d346",
        "git_head": _git_head(),
        "git_status_paths": status_paths,
        "validate_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "prepare_process_identity": prereg.get("prepare_process_identity"),
        "attempt0_validate_process_identity": attempt0_preflight.get(
            "validate_process_identity"
        ),
        "reactive_amendment": {
            "path": _relative(REACTIVE_AMENDMENT_PATH),
            "sha256": _sha256(REACTIVE_AMENDMENT_PATH),
        },
        "torch_cuda_preflight": {
            "available": torch_cuda_available,
            "device_name": torch_cuda_device_name,
        },
        "source_hashes": source_hashes,
        "source_inventories": inventories,
        "recomputed_eligibility": eligibility,
        "parameter_checks": _parameter_checks(),
        "app_launcher_checks": _app_launcher_checks(args),
        "app_launcher_environment": {
            "ENABLE_CAMERAS": os.environ.get("ENABLE_CAMERAS"),
            "XR": os.environ.get("XR"),
            "LIVESTREAM": os.environ.get("LIVESTREAM"),
            "HEADLESS": os.environ.get("HEADLESS"),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json(EFFECTIVE_VALIDATE_PREFLIGHT_PATH, report)
    return bool(report["pass"])


def _witness_manifest(raw_audit: dict[str, Any]) -> dict[str, Any]:
    expected = {
        f"{body}_part_{index:03d}_{channel}.json"
        for body in d334.BODY_LABELS
        for index in range(64)
        for channel in ("prototype", "instance")
    }
    observed_paths = sorted(WITNESS_DIR.glob("*.json")) if WITNESS_DIR.is_dir() else []
    observed = {path.name for path in observed_paths}
    rows: list[dict[str, Any]] = []
    ordinals: list[int] = []
    for path in observed_paths:
        witness = _json(path)
        stem = path.stem
        channel = stem.rsplit("_", 1)[1]
        body_and_part = stem[: -(len(channel) + 1)]
        body = "gripper_link" if body_and_part.startswith("gripper_link_") else "link5"
        part_name = body_and_part[len(body) + 1 :]
        events = witness.get("events", [])
        event = events[0] if len(events) == 1 else {}
        ordinal = witness.get("request_order_ordinal")
        if isinstance(ordinal, int):
            ordinals.append(ordinal)
        cache = witness.get("cache_release", {})
        settings = witness.get("isolated_cooking_settings", {})
        checks = {
            "filename_identity": witness.get("body") == body
            and witness.get("part_name") == part_name
            and witness.get("channel") == channel,
            "callback_exactly_once": witness.get("callback_count") == 1
            and len(events) == 1,
            "callback_inline": event.get("callback_during_synchronous_request") is True,
            "result_valid": event.get("result_name") == "RESULT_VALID"
            and event.get("result_value") == 0,
            "one_convex": event.get("convex_count") == 1
            and len(event.get("convexes", [])) == 1,
            "serialization_errors_zero": event.get("serialization_errors") == [],
            "payload_persisted_before_classification": witness.get(
                "callback_payload_persisted_before_classification"
            )
            is True
            and witness.get("classification_performed") is False,
            "cache_release_complete": bool(cache) and all(cache.values()),
            "settings_isolated_and_restored": d339._isolated_settings_pass(settings),
            "request_no_exception": witness.get("request_exception") is None,
            "channel_valid": channel in {"prototype", "instance"},
            "prototype_path_distinct": channel != "prototype"
            or witness.get("cook_prim_path") != witness.get("instance_path"),
        }
        rows.append(
            {
                "filename": path.name,
                "body": body,
                "part_name": part_name,
                "channel": channel,
                "request_order_ordinal": ordinal,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    row_by_key = {(row["body"], row["part_name"], row["channel"]): row for row in rows}
    order_checks = []
    for body in d334.BODY_LABELS:
        for index in range(64):
            name = f"part_{index:03d}"
            prototype = row_by_key.get((body, name, "prototype"))
            instance = row_by_key.get((body, name, "instance"))
            order_checks.append(
                bool(
                    prototype
                    and instance
                    and isinstance(prototype["request_order_ordinal"], int)
                    and isinstance(instance["request_order_ordinal"], int)
                    and prototype["request_order_ordinal"]
                    < instance["request_order_ordinal"]
                )
            )
    raw_parts = [
        row
        for body in raw_audit.get("per_body", {}).values()
        for row in body.get("part_checks", [])
    ]
    checks = {
        "exact_256_filenames": observed == expected and len(observed_paths) == 256,
        "all_nonzero": len(observed_paths) == 256
        and all(path.stat().st_size > 0 for path in observed_paths),
        "all_unique_sha256": len({row["sha256"] for row in rows}) == 256,
        "all_witness_rows_pass": len(rows) == 256 and all(row["pass"] for row in rows),
        "ordinals_exact_1_through_256": sorted(ordinals) == list(range(1, 257)),
        "prototype_before_instance_128_of_128": len(order_checks) == 128
        and all(order_checks),
        "raw_audit_request_order_exact": raw_audit.get("request_order")
        == ["prototype", "instance"],
        "raw_audit_part_count_128": len(raw_parts) == 128,
        "raw_audit_channel_requests_256": sum(
            2
            for row in raw_parts
            if set(row.get("channel_consensus", {})) >= {"instance", "prototype"}
        )
        == 256,
    }
    return {
        "artifact": "D346_FRESH_COOK_WITNESS_MANIFEST_V1",
        "expected_count": 256,
        "observed_count": len(observed_paths),
        "expected_filenames": sorted(expected),
        "observed_filenames": sorted(observed),
        "rows": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _rerun_rows(
    inner: Any,
    core_manifest: dict[str, Any],
    raw_shapes: list[dict[str, Any]],
    live_audit: dict[str, Any],
    representation_gate: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    coordinate_frames, meshes, scalars, events = d344._rerun_rows(
        inner, core_manifest, raw_shapes, live_audit, representation_gate
    )
    transformed_events = []
    for event in events:
        transformed = dict(event)
        if transformed.get("entity_path") == "events/d344":
            transformed["entity_path"] = "events/d346"
        transformed["text"] = str(transformed.get("text", "")).replace("D344", "D346")
        transformed_events.append(transformed)
    return coordinate_frames, meshes, scalars, transformed_events


def _rerun_contract(
    frames: list[dict[str, Any]],
    coordinate_frames: list[dict[str, Any]],
    meshes: list[dict[str, Any]],
    scalars: list[dict[str, Any]],
    events: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    exact_entities, expected_components = _expected_rrd_contract()
    observed_counts = {
        "frame_count": len(frames),
        "coordinate_frame_count": len(coordinate_frames),
        "mesh_count": len(meshes),
        "scalar_row_count": len(scalars),
        "event_row_count": len(events),
    }
    log_status = log_rerun(
        RRD_PATH,
        frames=frames,
        coordinate_frames=coordinate_frames,
        meshes=meshes,
        scalar_trace=scalars,
        events=events,
        recording_metadata=metadata,
        recording_id="g0a_d346_attempt3_fresh_live_representation",
        blueprint_path=RBL_PATH,
        blueprint_mode="collision_gate",
        live_viewer=False,
        app_id="roarm_g0a_collision_gate",
    )
    validation = (
        validate_rerun_artifact(
            RRD_PATH,
            expected_entity_paths=[
                "cook/source/link5/parts/part_000",
                "cook/candidate/link5/parts/part_045",
                "cook/instance/gripper_link/parts/part_057",
                "cook/prototype/gripper_link/target/cylinder",
                "events/d346",
            ],
            expected_timeline_names=["event_idx", "part_idx"],
            exact_entity_paths=exact_entities,
            exact_timeline_names=["blueprint", "event_idx", "log_time", "part_idx"],
            expected_entity_components=expected_components,
            blueprint_path=RBL_PATH,
            screenshot_path=RERUN_SCREENSHOT_PATH,
        )
        if log_status.get("ok", False)
        else {"pass": False, "errors": ["Rerun recording/finalization failed"]}
    )
    observed_counts["exact_non_system_entity_count"] = len(
        validation.get("entity_path_contract", {}).get("observed_non_system", [])
    )
    count_checks = {
        name: observed_counts.get(name) == expected
        for name, expected in EXPECTED_RERUN_COUNTS.items()
    }
    log_count_checks = {
        "coordinate_frame_count": log_status.get("coordinate_frame_count") == 2,
        "mesh_count": log_status.get("mesh_count") == 522,
        "scalar_row_count": log_status.get("scalar_row_count") == 1040,
        "event_row_count": log_status.get("event_row_count") == 132,
        "trace_steps_zero": log_status.get("trace_steps") == 0,
    }
    report = {
        "artifact": "D346_RERUN_MACHINE_VALIDATION_V1",
        "log_status": log_status,
        "archive_validation": validation,
        "observed_counts": observed_counts,
        "expected_counts": EXPECTED_RERUN_COUNTS,
        "count_checks": count_checks,
        "log_count_checks": log_count_checks,
        "rrd_contract_sha256": _rrd_contract_digest(),
        "manual_visual_inspection_required": True,
        "manual_visual_inspection_pending": bool(validation.get("pass")),
        "pass": bool(
            log_status.get("ok")
            and validation.get("pass")
            and all(count_checks.values())
            and all(log_count_checks.values())
        ),
    }
    _write_json(RERUN_VALIDATION_PATH, report)
    return report


def _immutability_report(before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    after = _source_inventories()
    checks = {
        name: before[name]["rows"] == after[name]["rows"]
        and before[name]["digest"] == after[name]["digest"]
        for name in before
    }
    return {
        "before": before,
        "after": after,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if not key.startswith("_")}


def _aggregate_live_counts(live_audit: dict[str, Any]) -> dict[str, int]:
    rows = [
        row
        for body in live_audit.get("per_body", {}).values()
        for row in body.get("part_checks", [])
    ]
    return {
        "observed_parts": len(rows),
        "part_pass": sum(bool(row.get("pass")) for row in rows),
        "surface_pass": sum(
            bool(row.get("checks", {}).get("both_channel_surface_le_0p1mm"))
            for row in rows
        ),
        "volume_pass": sum(
            bool(row.get("checks", {}).get("property_vs_both_channel_volume_le_5pct"))
            for row in rows
        ),
        "owner_pass": sum(
            bool(row.get("checks", {}).get("owner_matches")) for row in rows
        ),
        "gpu_compatible_pass": sum(
            bool(row.get("checks", {}).get("physx_gpu_convex_compatible"))
            for row in rows
        ),
        "typed_bits_pass": sum(
            bool(row.get("typed_scalar", {}).get("pass")) for row in rows
        ),
        "fixed_point_or_preserved_pass": sum(
            bool(row.get("fixed_point", {}).get("pass")) for row in rows
        ),
    }


def _write_automated_report(summary: dict[str, Any]) -> None:
    counts = summary["live_counts"]
    gate = summary["representation_gate"]
    rerun = summary["rerun"]
    _write_text(
        AUTOMATED_REPORT_PATH,
        "# D346 자동 검사 보고\n\n"
        f"- 수치 판정: `{summary['scientific_verdict']}`\n"
        f"- callback witness: `{summary['witness_manifest']['observed_count']}/256`\n"
        f"- 실제 충돌 조각: `{counts['part_pass']}/128`\n"
        f"- 목표 자세 거리 검사: `{gate.get('target_clear_and_faithful', False)}`\n"
        f"- Rerun 기계 검사: `{rerun.get('pass', False)}`\n"
        f"- 측정 구간 물리 진행 횟수: `{summary['controlled_physics_steps']}`\n"
        f"- 기존 파라미터 증가/변경/허용값 완화: `0/0/0`\n\n"
        "이 자동 결과만으로 D346 완료라고 부르지 않는다. Rerun 화면과 decision PNG를 "
        "실제로 열어 본 별도 기록이 있어야 완료 요약을 만들 수 있다.\n",
    )


def _run_validate(
    args: argparse.Namespace,
    simulation_app: Any,
    resolved_launcher: dict[str, Any],
) -> int:
    preflight = _json(EFFECTIVE_VALIDATE_PREFLIGHT_PATH)
    if preflight.get("pass") is not True:
        raise RuntimeError("D346 validate preflight did not pass before Isaac launch")
    if resolved_launcher.get("pass") is not True:
        raise RuntimeError(f"D346 resolved AppLauncher contract failed: {resolved_launcher}")
    prereg = _json(PREREG_PATH)
    before_inventories = _source_inventories()
    core_manifest = _json(D344_CORE_MANIFEST)
    capture = _json(D340_CANDIDATES)
    d334_summary = _json(D334_SUMMARY)
    d336_summary = _json(D336_SUMMARY)
    args.robot_usd_path = VARIANT_ROBOT_USD
    inner = d333._make_runtime_env(args)
    try:
        inner.reset(seed=int(args.seed))
        counter_start = int(inner._sim_step_counter)
        stage_contract = d333._stage_contract(inner)
        sensor_contract, _filter_map = d333._sensor_contract(inner)
        from pxr import UsdGeom

        runtime_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(inner.scene.stage))
        try:
            raw_shapes, raw_source_contract = d339._build_retained_raw_shapes(
                inner, d334_summary
            )
        except Exception as error:
            raw_shapes = []
            raw_source_contract = {
                "pass": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }

        WITNESS_DIR.mkdir(parents=True, exist_ok=False)
        try:
            _helper_cooked, raw_audit = d340._validate_live_attempt3(
                inner,
                simulation_app,
                core_manifest,
                capture,
                WITNESS_DIR,
            )
        except Exception as error:
            raw_audit = {
                "artifact": "D346_FRESH_LIVE_MEASUREMENT_EXCEPTION_STOP",
                "pass": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "per_body": {},
            }
        raw_audit["d346_wrapper_artifact"] = "D346_RAW_LIVE_MEASUREMENT_V1"
        _write_json(RAW_LIVE_MEASUREMENT_PATH, raw_audit)
        witness_manifest = _witness_manifest(raw_audit)
        _write_json(WITNESS_MANIFEST_PATH, witness_manifest)

        try:
            cooked_by_body, live_audit = d344._reclassify_live_measurement(
                inner, core_manifest, raw_audit, capture
            )
            live_audit["helper_artifact"] = live_audit.get("artifact")
            live_audit["artifact"] = "D346_FRESH_LIVE_REPRESENTATION_AUDIT_V1"
        except Exception as error:
            cooked_by_body = {body: [] for body in d334.BODY_LABELS}
            live_audit = {
                "artifact": "D346_FRESH_LIVE_RECLASSIFICATION_EXCEPTION_STOP",
                "pass": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "per_body": {},
            }
        _write_json(LIVE_AUDIT_PATH, live_audit)

        scene_checks = {
            "stage_contract": bool(stage_contract["hard_contract_pass"]),
            "sensor_contract": bool(sensor_contract["hard_contract_pass"]),
            "runtime_meters_per_unit_one": math.isclose(
                runtime_meters_per_unit, 1.0, rel_tol=0.0, abs_tol=0.0
            ),
            "resolved_app_launcher_contract": bool(resolved_launcher["pass"]),
            "retained_raw_source_contract": bool(raw_source_contract["pass"]),
            "witness_manifest_256": bool(witness_manifest["pass"]),
            "fresh_live_audit_128": bool(live_audit["pass"]),
        }
        all_live_prerequisites = all(scene_checks.values())
        if all_live_prerequisites:
            d336_rescore = d337._load_d336_rescore()
            cache = d337._Cache(inner, raw_shapes)
            controls = d337._negative_controls(
                inner,
                raw_shapes,
                d334_summary,
                d336_summary,
                d336_rescore,
                cache,
            )
            if controls.get("pass"):
                candidate = d337._evaluate_candidate(
                    inner,
                    raw_shapes,
                    OLD_RADIAL_NM,
                    OLD_TANGENT_NM,
                    Q5_OPEN_RAD,
                    stage="d346_frozen_open_jaw_target",
                )
            else:
                candidate = d339._fallback_candidate_without_raw(
                    inner,
                    "D346 target query forbidden because the frozen controls failed",
                )
                candidate["stage"] = "d346_target_query_not_run_control_stop"
        else:
            controls = {
                "artifact": "D346_D337_CONTROLS_SKIPPED",
                "pass": False,
                "reason": "256-callback/128-part/stage/sensor/raw prerequisite failed",
            }
            candidate = d339._fallback_candidate_without_raw(
                inner,
                "D346 target query forbidden until the full 128-part live contract passes",
            )
            candidate["stage"] = "d346_target_query_not_run"
        _write_json(CONTROLS_PATH, controls)

        target_prerequisites = bool(all_live_prerequisites and controls.get("pass"))
        if target_prerequisites:
            representation_gate, decision_raw, decision_live = d339._representation_gate(
                inner,
                raw_shapes,
                cooked_by_body,
                candidate,
                controls,
                live_audit,
            )
            representation_gate["artifact"] = "D346_ZERO_STEP_REPRESENTATION_GATE_V1"
            representation_gate["prephysics_support_eligible"] = bool(
                representation_gate["contract_pass"]
                and representation_gate["target_clear_and_faithful"]
            )
            representation_gate["physics_licensed"] = False
            representation_gate["physics_forbidden_in_d346"] = True
        else:
            decision_raw = {"pose": "target_query_not_run", "queries": []}
            decision_live = {"pose": "live_union_not_queried", "queries": []}
            representation_gate = {
                "artifact": "D346_ZERO_STEP_REPRESENTATION_GATE_V1",
                "checks": {**scene_checks, "d337_controls": bool(controls.get("pass"))},
                "per_body": {},
                "contract_pass": False,
                "target_clear_and_faithful": False,
                "prephysics_support_eligible": False,
                "physics_licensed": False,
                "physics_forbidden_in_d346": True,
                "structured_stop_reason": (
                    "callback/live prerequisite failed; raw/live target union not queried"
                ),
            }

        counter_end = int(inner._sim_step_counter)
        counter_delta = counter_end - counter_start
        counter_unchanged = counter_start == counter_end
        representation_gate["global_sim_counter"] = {
            "start": counter_start,
            "end": counter_end,
            "delta": counter_delta,
            "unchanged": counter_unchanged,
        }
        representation_gate["controlled_physics_steps"] = max(counter_delta, 0)
        representation_gate.setdefault("checks", {})[
            "global_sim_counter_unchanged"
        ] = counter_unchanged
        if not counter_unchanged:
            representation_gate["contract_pass"] = False
            representation_gate["target_clear_and_faithful"] = False
            representation_gate["prephysics_support_eligible"] = False
        _write_json(REPRESENTATION_GATE_PATH, representation_gate)

        base_runtime_prerequisites = bool(
            scene_checks["stage_contract"]
            and scene_checks["sensor_contract"]
            and scene_checks["runtime_meters_per_unit_one"]
            and scene_checks["resolved_app_launcher_contract"]
            and scene_checks["retained_raw_source_contract"]
        )
        if not base_runtime_prerequisites:
            scientific_verdict = VERDICT_PREREQUISITE_FAIL
            interpretation = "장면·센서·원본 기준 형상 사전조건 중 하나 이상이 실패해 목표 검사를 실행하지 않았다"
        elif not live_audit.get("pass") or not witness_manifest.get("pass"):
            scientific_verdict = VERDICT_LIVE_FAIL
            interpretation = "256개 callback 또는 128개 실제 충돌 조각 검사 중 하나 이상이 실패했다"
        elif not controls.get("pass"):
            scientific_verdict = VERDICT_PREREQUISITE_FAIL
            interpretation = "128개 조각은 통과했지만 고정 D337 대조 검사가 실패해 목표 검사를 실행하지 않았다"
        elif not representation_gate.get("contract_pass") or not representation_gate.get(
            "target_clear_and_faithful"
        ):
            scientific_verdict = VERDICT_TARGET_FAIL
            interpretation = "128개 조각은 통과했지만 고정 목표 자세의 거리·충실도 검사가 실패했다"
        else:
            scientific_verdict = VERDICT_PREPHYSICS_PENDING
            interpretation = (
                "새 실제 충돌 형상과 고정 목표 자세가 사전 물리 검사를 통과했다; "
                "Rerun 실제 화면 확인 뒤에도 settle은 별도 사례다"
            )

        try:
            if decision_raw.get("queries") and decision_live.get("queries"):
                d339._write_representation_figure(
                    DECISION_PNG,
                    "D346: frozen zero-step raw vs fresh live convex union",
                    inner,
                    raw_shapes,
                    cooked_by_body,
                    decision_raw,
                    decision_live,
                    candidate["_canonical"],
                )
            else:
                d339._write_contract_stop_figure(
                    DECISION_PNG,
                    title="D346 pre-physics contract STOP",
                    scene_checks=scene_checks,
                    candidate=candidate,
                )
            decision_png_ok = DECISION_PNG.is_file() and DECISION_PNG.stat().st_size > 0
            decision_png_error = None
        except Exception as error:
            decision_png_ok = False
            decision_png_error = f"{type(error).__name__}: {error}"
        try:
            marker_status = draw_frames(
                candidate["_frames"], prim_path="/World/D346FreshLiveFrames"
            )
        except Exception as error:
            marker_status = {"ok": False, "error": f"{type(error).__name__}: {error}"}

        coordinate_frames, meshes, scalars, events = _rerun_rows(
            inner,
            core_manifest,
            raw_shapes,
            live_audit,
            representation_gate,
        )
        frame_names = tuple(str(row["name"]) for row in candidate["_frames"])
        if frame_names != d344.FRAME_NAMES:
            raise RuntimeError(
                f"D346 target frame schema drift: {frame_names} != {d344.FRAME_NAMES}"
            )
        rerun = _rerun_contract(
            candidate["_frames"],
            coordinate_frames,
            meshes,
            scalars,
            events,
            {
                "case": "g0a_d346",
                "purpose": "immutable attempt3 fresh live representation and frozen target audit",
                "git_head": _git_head(),
                "new_variables": NEW_VARIABLES,
                "scientific_authority": (
                    "callback arrays, direct authored arrays, JSON metrics, exact hashes, "
                    "and Float64 distances; Rerun is observability only"
                ),
                "viewer_geometry_role": "Float32 one-way spatial observability copy",
                "q5_convention": "0=CLOSED; 1.5413rad=OPEN",
                "target": "radial=7mm tangent=11mm tangent_sign=-1",
                "physics": "forbidden / controlled steps 0",
                "candidate_panel_role": (
                    "immutable attempt3 authored geometry; changed 13 are D340 fixed points"
                ),
            },
        )
        immutability = _immutability_report(before_inventories)
        artifact_checks = {
            "decision_png_nonzero": decision_png_ok,
            "frame_markers_ok": bool(marker_status.get("ok")),
            "rerun_completion_machine_pass": bool(rerun["pass"]),
            "witness_manifest_pass": bool(witness_manifest["pass"]),
            "source_immutability_pass": bool(immutability["pass"]),
        }
        automated_artifact_pass = all(artifact_checks.values())
        observability_verdict = (
            "D346_RERUN_MACHINE_CONTRACT_PASS_MANUAL_INSPECTION_PENDING"
            if automated_artifact_pass
            else VERDICT_OBSERVABILITY_FAIL
        )
        live_counts = _aggregate_live_counts(live_audit)
        summary = {
            "artifact": "D346_AUTOMATED_SUMMARY_V1",
            "case": "g0a_d346",
            "scientific_verdict": scientific_verdict,
            "observability_verdict": observability_verdict,
            "scientific_verdict_before_observability_gate": scientific_verdict,
            "automated_pass": bool(
                scientific_verdict == VERDICT_PREPHYSICS_PENDING
                and automated_artifact_pass
            ),
            "manual_visual_inspection_pending": bool(rerun["pass"]),
            "interpretation": interpretation,
            "new_variables": NEW_VARIABLES,
            "new_physical_variables": [],
            "parameters_increased": [],
            "parameters_changed": [],
            "thresholds_relaxed": [],
            "prepare_process_identity": prereg.get("prepare_process_identity"),
            "reactive_amendment": {
                "path": _relative(REACTIVE_AMENDMENT_PATH),
                "sha256": _sha256(REACTIVE_AMENDMENT_PATH),
                "attempt0_non_scientific": True,
                "effective_scientific_execution_count": 1,
                "effective_preflight_path": _relative(
                    EFFECTIVE_VALIDATE_PREFLIGHT_PATH
                ),
                "effective_preflight_sha256": _sha256(
                    EFFECTIVE_VALIDATE_PREFLIGHT_PATH
                ),
            },
            "validate_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
            "environment": {
                "numpy": str(np.__version__),
                "psutil": str(psutil.__version__),
                "rerun_sdk": str(rr.__version__),
                "python": str(Path(sys.executable).resolve()),
                "app_launcher": {
                    "headless": args.headless,
                    "livestream": args.livestream,
                    "enable_cameras": args.enable_cameras,
                    "xr": args.xr,
                    "device": args.device,
                    "cpu": args.cpu,
                    "experience": args.experience,
                    "kit_args": args.kit_args,
                    "rendering_mode": args.rendering_mode,
                    "verbose": args.verbose,
                    "info": args.info,
                    "anim_recording_enabled": args.anim_recording_enabled,
                },
                "resolved_app_launcher": resolved_launcher,
            },
            "loaded_asset": {
                "root_usd": _relative(VARIANT_ROBOT_USD),
                "root_usd_sha256": _sha256(VARIANT_ROBOT_USD),
                "physics_usd": _relative(VARIANT_PHYSICS_USD),
                "physics_usd_sha256": _sha256(VARIANT_PHYSICS_USD),
                "attempt3_inventory_digest": before_inventories["d344_attempt3"][
                    "digest"
                ],
            },
            "parameter_contract": _json(PARAMETER_AUDIT_PATH),
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "runtime_meters_per_unit": runtime_meters_per_unit,
            "raw_source_contract": raw_source_contract,
            "scene_checks": scene_checks,
            "witness_manifest": witness_manifest,
            "fresh_live_audit": live_audit,
            "live_counts": live_counts,
            "d337_controls": controls,
            "frozen_candidate": _public_candidate(candidate),
            "frozen_candidate_alignment": candidate.get("_alignment"),
            "representation_gate": representation_gate,
            "decision_raw": decision_raw,
            "decision_live": decision_live,
            "controlled_physics_steps": max(counter_delta, 0),
            "simulation_counter": {
                "start": counter_start,
                "end": counter_end,
                "delta": counter_delta,
                "unchanged": counter_unchanged,
            },
            "visualization": {
                "decision_png": _relative(DECISION_PNG) if DECISION_PNG.is_file() else None,
                "decision_png_sha256": _sha256(DECISION_PNG)
                if DECISION_PNG.is_file()
                else None,
                "decision_png_error": decision_png_error,
                "frame_markers": marker_status,
            },
            "rerun": rerun,
            "immutability": immutability,
            "artifact_checks": artifact_checks,
            "outcome_guards": {
                "g0a_pass": False,
                "physics_executed": counter_delta != 0,
                "settle_executed": False,
                "ten_trial_run": False,
                "g0b_run": False,
                "rl_run": False,
                "ladder_promoted": False,
                "canonical_asset_changed": False,
                "d344_attempt3_changed": not bool(
                    immutability.get("checks", {}).get("d344_attempt3")
                ),
                "separate_settle_case_required": scientific_verdict
                == VERDICT_PREPHYSICS_PENDING,
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
                    "manual_visual_inspection_pending": summary[
                        "manual_visual_inspection_pending"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0 if summary["automated_pass"] else 2
    finally:
        inner.close()


def _manual_inspection_checks(manual: dict[str, Any]) -> dict[str, bool]:
    screenshot = manual.get("rerun_screenshot", {})
    decision = manual.get("decision_png", {})
    observations = manual.get("observations", {})
    required_observations = (
        "eight_independent_spatial_panels_visible",
        "link5_four_variants_nonempty",
        "gripper_four_variants_nonempty",
        "target_cylinder_visible_in_all_eight_panels",
        "registered_frames_visible",
        "metric_table_visible",
        "event_table_visible",
        "required_content_not_obscured",
        "decision_png_opened_and_legible",
    )
    inspection_method = str(manual.get("inspection_method", "")).lower()
    return {
        "artifact_exact": manual.get("artifact")
        == "D346_RERUN_MANUAL_VISUAL_INSPECTION_V1",
        "case_exact": manual.get("case") == "g0a_d346",
        "rerun_screenshot_path_exact": screenshot.get("path")
        == _relative(RERUN_SCREENSHOT_PATH),
        "rerun_screenshot_sha_exact": RERUN_SCREENSHOT_PATH.is_file()
        and screenshot.get("sha256") == _sha256(RERUN_SCREENSHOT_PATH),
        "rerun_screenshot_bytes_exact": RERUN_SCREENSHOT_PATH.is_file()
        and screenshot.get("bytes") == RERUN_SCREENSHOT_PATH.stat().st_size,
        "rerun_screenshot_dimensions_exact": screenshot.get("raster_dimensions")
        == _png_dimensions(RERUN_SCREENSHOT_PATH),
        "decision_png_path_exact": decision.get("path") == _relative(DECISION_PNG),
        "decision_png_sha_exact": DECISION_PNG.is_file()
        and decision.get("sha256") == _sha256(DECISION_PNG),
        "decision_png_bytes_exact": DECISION_PNG.is_file()
        and decision.get("bytes") == DECISION_PNG.stat().st_size,
        "decision_png_dimensions_exact": decision.get("raster_dimensions")
        == _png_dimensions(DECISION_PNG),
        "inspection_date_exact": manual.get("inspection_date_kst") == "2026-07-14",
        "inspection_method_original_view_image": "view_image" in inspection_method
        and "original" in inspection_method,
        "required_observations_true": all(
            observations.get(name) is True for name in required_observations
        ),
        "bounded_interpretation_present": len(manual.get("bounded_interpretation", []))
        >= 3,
        "manual_pass_true": manual.get("manual_visual_inspection_pass") is True,
        "scientific_override_false": manual.get("scientific_verdict_override") is False,
        "d344_fail_retained": manual.get("d344_verdict_retained")
        == "D344_G0A_ATTEMPT3_AUTHORING_CONTRACT_FAIL_STOP",
        "g0a_false": manual.get("g0a_pass") is False,
        "manual_markdown_nonzero": MANUAL_INSPECTION_MD_PATH.is_file()
        and MANUAL_INSPECTION_MD_PATH.stat().st_size > 0,
    }


def _run_finalize(_args: argparse.Namespace) -> int:
    if COMPLETION_SUMMARY_PATH.exists() or COMPLETION_REPORT_PATH.exists():
        raise RuntimeError("D346 completion artifacts already exist; refusing overwrite")
    automated = _json(AUTOMATED_SUMMARY_PATH)
    manual = _json(MANUAL_INSPECTION_PATH)
    rerun_validation = _json(RERUN_VALIDATION_PATH)
    prereg = _json(PREREG_PATH)
    amendment = _json(REACTIVE_AMENDMENT_PATH)
    manual_checks = _manual_inspection_checks(manual)
    machine_report_pass = bool(rerun_validation.get("pass"))
    scientific_supported = automated.get("scientific_verdict") == VERDICT_PREPHYSICS_PENDING
    current_inventories = _source_inventories()
    finalize_input_checks = {
        "git_head_exact": _git_head() == prereg.get("git_head") == EXPECTED_HEAD,
        "git_status_scope_only": _status_scope_pass(_git_status_paths()),
        "attempt0_preflight_hash_still_exact": _sha256(VALIDATE_PREFLIGHT_PATH)
        == ATTEMPT0_PREFLIGHT_SHA,
        "attempt0_exception_hash_still_exact": _sha256(RUNTIME_EXCEPTION_PATH)
        == ATTEMPT0_EXCEPTION_SHA,
        "amendment_hash_matches_automated_summary": _sha256(
            REACTIVE_AMENDMENT_PATH
        )
        == automated.get("reactive_amendment", {}).get("sha256"),
        "effective_preflight_hash_matches_automated_summary": _sha256(
            EFFECTIVE_VALIDATE_PREFLIGHT_PATH
        )
        == automated.get("reactive_amendment", {}).get(
            "effective_preflight_sha256"
        ),
        "source_hashes_still_exact": _source_hashes()
        == amendment.get("amended_source_hashes"),
        "source_inventory_counts_still_exact": {
            name: row["file_count"] for name, row in current_inventories.items()
        }
        == prereg.get("source_inventory_counts"),
        "source_inventory_digests_still_exact": {
            name: row["digest"] for name, row in current_inventories.items()
        }
        == prereg.get("source_inventory_digests"),
        "matches_validate_after_inventory": current_inventories
        == automated.get("immutability", {}).get("after"),
        "rrd_present": RRD_PATH.is_file(),
        "rbl_present": RBL_PATH.is_file(),
        "rerun_screenshot_present": RERUN_SCREENSHOT_PATH.is_file(),
        "decision_png_present": DECISION_PNG.is_file(),
        "rrd_hash_still_matches_machine_validation": RRD_PATH.is_file()
        and _sha256(RRD_PATH)
        == rerun_validation.get("archive_validation", {}).get("sha256"),
        "rbl_hash_still_matches_machine_validation": RBL_PATH.is_file()
        and _sha256(RBL_PATH)
        == rerun_validation.get("archive_validation", {})
        .get("blueprint_verify", {})
        .get("sha256"),
        "screenshot_hash_still_matches_machine_validation": RERUN_SCREENSHOT_PATH.is_file()
        and _sha256(RERUN_SCREENSHOT_PATH)
        == rerun_validation.get("archive_validation", {})
        .get("headless_render", {})
        .get("sha256"),
        "decision_png_hash_still_matches_automated_summary": DECISION_PNG.is_file()
        and _sha256(DECISION_PNG)
        == automated.get("visualization", {}).get("decision_png_sha256"),
    }
    machine_pass = bool(
        machine_report_pass
        and finalize_input_checks["rrd_hash_still_matches_machine_validation"]
        and finalize_input_checks["rbl_hash_still_matches_machine_validation"]
        and finalize_input_checks["screenshot_hash_still_matches_machine_validation"]
    )
    source_immutable = bool(
        automated.get("immutability", {}).get("pass") is True
        and all(finalize_input_checks.values())
    )
    completion_pass = bool(
        automated.get("automated_pass")
        and scientific_supported
        and machine_pass
        and all(manual_checks.values())
        and source_immutable
        and automated.get("controlled_physics_steps") == 0
    )
    if completion_pass:
        final_verdict = VERDICT_COMPLETE
    elif automated.get("scientific_verdict") in {
        VERDICT_PREREQUISITE_FAIL,
        VERDICT_LIVE_FAIL,
        VERDICT_TARGET_FAIL,
    }:
        final_verdict = automated["scientific_verdict"]
    else:
        final_verdict = VERDICT_OBSERVABILITY_FAIL
    representation = automated.get("representation_gate", {})
    body_distances = {
        body: representation.get("per_body", {}).get(body)
        for body in d334.BODY_LABELS
    }
    completion = {
        "artifact": "D346_COMPLETION_SUMMARY_V1",
        "case": "g0a_d346",
        "final_verdict": final_verdict,
        "completion_contract_pass": completion_pass,
        "new_variables": NEW_VARIABLES,
        "automated_evidence": {
            "path": _relative(AUTOMATED_SUMMARY_PATH),
            "sha256": _sha256(AUTOMATED_SUMMARY_PATH),
            "scientific_verdict": automated.get("scientific_verdict"),
            "automated_pass": automated.get("automated_pass"),
            "preserved_without_overwrite": True,
        },
        "manual_evidence": {
            "json_path": _relative(MANUAL_INSPECTION_PATH),
            "json_sha256": _sha256(MANUAL_INSPECTION_PATH),
            "markdown_path": _relative(MANUAL_INSPECTION_MD_PATH),
            "markdown_sha256": _sha256(MANUAL_INSPECTION_MD_PATH),
            "checks": manual_checks,
            "manual_visual_inspection_pass": all(manual_checks.values()),
        },
        "rerun_evidence": {
            "validation_path": _relative(RERUN_VALIDATION_PATH),
            "validation_sha256": _sha256(RERUN_VALIDATION_PATH),
            "rrd_path": _relative(RRD_PATH),
            "rrd_bytes": RRD_PATH.stat().st_size if RRD_PATH.is_file() else None,
            "rrd_sha256": _sha256(RRD_PATH) if RRD_PATH.is_file() else None,
            "rbl_path": _relative(RBL_PATH),
            "rbl_bytes": RBL_PATH.stat().st_size if RBL_PATH.is_file() else None,
            "rbl_sha256": _sha256(RBL_PATH) if RBL_PATH.is_file() else None,
            "screenshot_path": _relative(RERUN_SCREENSHOT_PATH),
            "screenshot_bytes": RERUN_SCREENSHOT_PATH.stat().st_size
            if RERUN_SCREENSHOT_PATH.is_file()
            else None,
            "screenshot_sha256": _sha256(RERUN_SCREENSHOT_PATH)
            if RERUN_SCREENSHOT_PATH.is_file()
            else None,
            "machine_contract_pass": machine_pass,
            "expected_counts": EXPECTED_RERUN_COUNTS,
            "exact_timelines": ["blueprint", "event_idx", "log_time", "part_idx"],
        },
        "scientific_evidence": {
            "callback_witnesses": automated.get("witness_manifest", {}).get(
                "observed_count"
            ),
            "callback_contract_pass": automated.get("witness_manifest", {}).get("pass"),
            "active_part_counts": automated.get("live_counts"),
            "fresh_live_audit_pass": automated.get("fresh_live_audit", {}).get("pass"),
            "target_clear_and_faithful": representation.get(
                "target_clear_and_faithful"
            ),
            "body_distances": body_distances,
        },
        "source_immutability_pass": source_immutable,
        "finalize_input_checks": finalize_input_checks,
        "reactive_amendment": {
            "path": _relative(REACTIVE_AMENDMENT_PATH),
            "sha256": _sha256(REACTIVE_AMENDMENT_PATH),
            "attempt0_scientific_execution": False,
            "effective_scientific_execution_count": 1,
        },
        "parameter_change_audit": {
            "new_physical_variables": 0,
            "existing_parameter_increases": 0,
            "existing_parameter_changes": 0,
            "threshold_relaxations": 0,
            "decomposition_parameter_changes": 0,
        },
        "scope_guards": {
            "controlled_physics_steps": automated.get("controlled_physics_steps"),
            "settle_executed": False,
            "ten_trial_run": False,
            "g0b_run": False,
            "rl_run": False,
            "ladder_promoted": False,
            "d344_historical_verdict_retained": (
                "D344_G0A_ATTEMPT3_AUTHORING_CONTRACT_FAIL_STOP"
            ),
            "g0a_pass": False,
        },
        "next_case_requires_separate_approval": (
            "fresh settle evaluation" if completion_pass else None
        ),
        "interpretation": (
            "D346 certifies only the fresh pre-physics collision representation and frozen "
            "target when completion_contract_pass is true. It never certifies settle or G0a."
        ),
    }
    _write_json(COMPLETION_SUMMARY_PATH, completion)
    _write_text(
        COMPLETION_REPORT_PATH,
        "# D346 완료 보고\n\n"
        f"- 최종 판정: `{final_verdict}`\n"
        f"- 전체 완료 계약: `{completion_pass}`\n"
        f"- callback: `{completion['scientific_evidence']['callback_witnesses']}/256`\n"
        f"- 실제 충돌 조각: `{automated.get('live_counts', {}).get('part_pass')}/128`\n"
        f"- 고정 목표 거리 검사: `{representation.get('target_clear_and_faithful')}`\n"
        f"- Rerun 기계/육안 검사: `{machine_pass}/{all(manual_checks.values())}`\n"
        f"- 측정 구간 물리 진행: `{automated.get('controlled_physics_steps')}`\n"
        f"- g0a_pass: `false`\n\n"
        "통과해도 다음 단계는 자동 settle이 아니라 별도 승인 사례다.\n",
    )
    print(json.dumps({"stage": "finalize", "final_verdict": final_verdict}, sort_keys=True))
    return 0 if completion_pass else 2


def _parser_for_stage(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "amend", "validate", "finalize"), required=True
    )
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--urdf_path", type=Path, default=d333.DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=SEED)
    if stage == "validate":
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    if "--print-rerun-contract" in sys.argv:
        entities, _components = _expected_rrd_contract()
        print(
            json.dumps(
                {
                    "digest": _rrd_contract_digest(),
                    "entity_count": len(entities),
                    "counts": EXPECTED_RERUN_COUNTS,
                },
                sort_keys=True,
            )
        )
        return 0
    stage_probe = argparse.ArgumentParser(add_help=False)
    stage_probe.add_argument(
        "--stage", choices=("prepare", "amend", "validate", "finalize"), required=True
    )
    stage_args, _unknown = stage_probe.parse_known_args()
    parser = _parser_for_stage(stage_args.stage)
    args = parser.parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D346 output path is forward-only and fixed by preregistration")
    if int(args.seed) != SEED:
        raise RuntimeError("D346 seed is frozen at 33201")
    if Path(args.urdf_path).resolve() != Path(d333.DEFAULT_URDF).resolve():
        raise RuntimeError("D346 URDF path is frozen at the registered D333 default")
    if args.stage == "validate" and not all(_app_launcher_checks(args).values()):
        false_checks = [name for name, passed in _app_launcher_checks(args).items() if not passed]
        raise RuntimeError(f"D346 AppLauncher arguments drifted: {false_checks}")
    args.process_nonce = secrets.token_hex(16)
    if args.stage == "prepare":
        return _run_prepare(args)
    if args.stage == "amend":
        return _run_reactive_amendment(args)
    if args.stage == "finalize":
        return _run_finalize(args)

    from isaaclab.app import AppLauncher

    if not _prepare_validate_preflight(args):
        return 2
    args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    launcher = AppLauncher(copy.deepcopy(args))
    simulation_app = launcher.app
    resolved_launcher = _resolved_launcher_report(launcher)

    def _close_app_at_exit() -> None:
        if simulation_app is not None:
            simulation_app.close()

    atexit.register(_close_app_at_exit)
    try:
        return _run_validate(args, simulation_app, resolved_launcher)
    except Exception as error:
        if not EFFECTIVE_RUNTIME_EXCEPTION_PATH.exists():
            _write_json(
                EFFECTIVE_RUNTIME_EXCEPTION_PATH,
                {
                    "artifact": "D346_RUNTIME_EXCEPTION_STOP",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "g0a_pass": False,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
