#!/usr/bin/env python3
"""D344 forward-only attempt3 authoring and fresh live-shape validation.

The case has two registered processes.  ``--stage build`` copies immutable
D339 attempt2 and authors the already-captured D340 fixed-point geometry on
exactly 13 parts.  ``--stage validate`` opens that derivative in a fresh Isaac
process, captures prototype and instance cook results for all 128 enabled
parts, applies the corrected D342 coordinate-domain and D343 typed-float
contracts, and conditionally evaluates the frozen zero-step target.  Physics,
settle, trials, G0b, RL, and ladder promotion are forbidden.
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
    cyl34_top_view_d336_grasp_g0a_finite_grid_caveat_discriminator as d336,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d337_grasp_g0a_open_jaw_target_gate as d337,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair as d339,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d340_grasp_g0a_fixed_point_live_authoring_repair as d340,
)


OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d344"
PREREG_PATH = OUT_DIR / "d344_preregistration.json"
PARAMETER_AUDIT_PATH = OUT_DIR / "d344_parameter_freeze_audit.json"
ELIGIBILITY_PATH = OUT_DIR / "d344_candidate_eligibility.json"
BUILD_PREFLIGHT_PATH = OUT_DIR / "d344_build_preflight.json"
BUILD_SUMMARY_PATH = OUT_DIR / "d344_attempt3_build_summary.json"
BUILD_REPORT_PATH = OUT_DIR / "d344_attempt3_build_report.md"
CORE_BUILD_MANIFEST_PATH = (
    OUT_DIR / "collision_asset/attempt3/d340_attempt3_asset_manifest.json"
)
D344_BUILD_MANIFEST_PATH = (
    OUT_DIR / "collision_asset/attempt3/d344_attempt3_asset_manifest.json"
)
VARIANT_DIR = (
    OUT_DIR
    / "collision_asset/attempt3/roarm_m3_fullmesh_fixed_point_parts"
)
VARIANT_ROBOT_USD = VARIANT_DIR / "roarm_m3.usd"
VARIANT_PHYSICS_USD = VARIANT_DIR / "configuration/roarm_m3_physics.usd"

VALIDATE_PREFLIGHT_PATH = OUT_DIR / "d344_validate_preflight.json"
RAW_LIVE_MEASUREMENT_PATH = OUT_DIR / "d344_raw_live_measurement.json"
LIVE_AUDIT_PATH = OUT_DIR / "d344_fresh_live_representation_audit.json"
REPRESENTATION_GATE_PATH = OUT_DIR / "d344_zero_step_representation_gate.json"
CONTROLS_PATH = OUT_DIR / "d344_d337_frozen_controls.json"
WITNESS_DIR = OUT_DIR / "d344_validate_cook_witnesses"
WITNESS_MANIFEST_PATH = OUT_DIR / "d344_validate_cook_witness_manifest.json"
DECISION_PNG = OUT_DIR / "d344_fixed_point_representation_decision.png"
RRD_PATH = OUT_DIR / "d344_attempt3_live_representation.rrd"
RBL_PATH = OUT_DIR / "d344_attempt3_live_representation.rbl"
RERUN_SCREENSHOT_PATH = OUT_DIR / "d344_attempt3_live_representation_rerun.png"
AUTOMATED_SUMMARY_PATH = OUT_DIR / "d344_automated_summary.json"
AUTOMATED_REPORT_PATH = OUT_DIR / "d344_automated_report.md"

D340_CANDIDATES = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d340/"
    "d340_capture_fixed_point_candidates.json"
)
D342_EVIDENCE = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d342/"
    "d342_authored_coordinate_stream_evidence.json"
)
D343_EVIDENCE = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d343/"
    "d343_usd_typed_float_readback_evidence.json"
)
D343_SUMMARY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d343/"
    "d343_usd_typed_float_readback_summary.json"
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
    / "claudedocs/session_20260714_grasp_g0a_d344_attempt3_fixed_point_collision_geometry.md"
)

NEW_VARIABLES = ["attempt3_fixed_point_collision_geometry"]
FAILING_PARTS = {
    "link5": (
        "part_011",
        "part_018",
        "part_023",
        "part_024",
        "part_040",
        "part_041",
        "part_045",
        "part_054",
    ),
    "gripper_link": (
        "part_000",
        "part_035",
        "part_036",
        "part_048",
        "part_057",
    ),
}
PART_KEYS = tuple(
    (body, f"part_{index:03d}")
    for body in ("link5", "gripper_link")
    for index in range(64)
)
CHANGED_KEYS = {
    (body, name) for body, names in FAILING_PARTS.items() for name in names
}

DECOMPOSITION_PARAMS = copy.deepcopy(d339.DECOMPOSITION_PARAMS)
EXPECTED_MIN_THICKNESS_BITS = 0x38D1B717
EXPECTED_MIN_THICKNESS_HEX = "0x38d1b717"
EXPECTED_MIN_THICKNESS_TYPED_M = float(np.float32(0.0001))
FROZEN_COMPATIBILITY_TOL_M = 1.0e-10
FIXED_POINT_TOL_M = 1.0e-9
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
REGISTERED_USD_CORE = (
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
REGISTERED_BUILD_LD_LIBRARY_PATH = ":".join(
    (
        "/home/cgxr/miniconda3/envs/isaaclab/lib",
        f"{REGISTERED_USD_CORE}/bin",
    )
)
Q5_OPEN_RAD = d339.Q5_OPEN_RAD
OLD_RADIAL_NM = d339.OLD_RADIAL_NM
OLD_TANGENT_NM = d339.OLD_TANGENT_NM
SEED = 33201

VERDICT_BUILD_FAIL = "D344_G0A_ATTEMPT3_AUTHORING_CONTRACT_FAIL_STOP"
VERDICT_LIVE_FAIL = "D344_G0A_FRESH_LIVE_REPRESENTATION_FAIL_STOP"
VERDICT_TARGET_FAIL = "D344_G0A_COOKED_TARGET_FIDELITY_FAIL_STOP"
VERDICT_PREPHYSICS_PENDING = (
    "D344_G0A_PREPHYSICS_COLLISION_REPRESENTATION_SUPPORTED_OBSERVABILITY_PENDING"
)
VERDICT_OBSERVABILITY_FAIL = "D344_RERUN_OBSERVABILITY_INCOMPLETE_STOP"

SOURCE_COLORS = {
    "source": [145, 145, 145, 52],
    "instance": [35, 120, 255, 72],
    "prototype": [235, 65, 200, 72],
    "candidate": [35, 205, 90, 92],
}
PART_METRICS = (
    "measurement_available",
    "max_surface_error_m",
    "max_property_volume_relative_error",
    "owner_matches",
    "collision_enabled",
    "min_thickness_bits_exact",
    "fixed_point_or_preserved",
    "part_pass",
)
REPRESENTATION_METRICS = (
    "query_ran",
    "raw_signed_distance_mm",
    "raw_value_finite",
    "live_signed_distance_mm",
    "live_value_finite",
    "absolute_delta_mm",
    "delta_finite",
    "body_pass",
)
FRAME_NAMES = (
    "d330_target_tcp",
    "actual_tcp_link5",
    "fixed_jaw_face",
    "cylinder_side_contact_point",
    "cylinder_object_frame",
    "commanded_tcp_link5",
)


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
    path.write_text(text, encoding="utf-8")


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
    paths = []
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
        "roarm_rl/viz_debug.py",
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
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_inventories() -> dict[str, dict[str, Any]]:
    roots = {
        "d338_attempt1": REPO / "claudedocs/runtime_logs/grasp_track/g0a_d338/collision_asset/attempt1",
        "d339_attempt2": REPO
        / "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2",
        "d340": REPO / "claudedocs/runtime_logs/grasp_track/g0a_d340",
        "d342": REPO / "claudedocs/runtime_logs/grasp_track/g0a_d342",
        "d343": REPO / "claudedocs/runtime_logs/grasp_track/g0a_d343",
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
        "d344_harness": Path(__file__).resolve(),
        "d344_session": SESSION_DOC,
        "start_here": START_HERE,
        "parameter_audit": PARAMETER_AUDIT_PATH,
        "d340_candidates": D340_CANDIDATES,
        "d342_evidence": D342_EVIDENCE,
        "d343_evidence": D343_EVIDENCE,
        "d343_summary": D343_SUMMARY,
        "d340_harness": Path(d340.__file__).resolve(),
        "d339_harness": Path(d339.__file__).resolve(),
        "viz_debug": REPO / "roarm_rl/viz_debug.py",
        "rerun_contract": REPO / "roarm_rl/rerun_contract.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _f32_bits(value: float | np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _f32_sha(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _i8_sha(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _parameter_checks(parameter_audit: dict[str, Any]) -> dict[str, bool]:
    return {
        "one_new_variable": parameter_audit.get("new_variables") == NEW_VARIABLES,
        "new_variable_count_one": parameter_audit.get("new_variable_count") == 1,
        "parameters_increased_empty": parameter_audit.get("parameters_increased") == [],
        "parameters_changed_empty": parameter_audit.get("parameters_changed") == [],
        "thresholds_relaxed_empty": parameter_audit.get("thresholds_relaxed") == [],
        "decomposition_exact_d339": parameter_audit.get("decomposition_parameters")
        == DECOMPOSITION_PARAMS,
        "hull_vertex_limit_64": DECOMPOSITION_PARAMS["hull_vertex_limit"] == 64,
        "max_convex_hulls_64": DECOMPOSITION_PARAMS["max_convex_hulls"] == 64,
        "voxel_resolution_one_million": DECOMPOSITION_PARAMS["voxel_resolution"]
        == 1_000_000,
        "error_percentage_one": DECOMPOSITION_PARAMS["error_percentage"] == 1.0,
        "min_thickness_requested_0p0001": DECOMPOSITION_PARAMS["min_thickness_m"]
        == 0.0001,
        "shrink_wrap_true": DECOMPOSITION_PARAMS["shrink_wrap"] is True,
        "min_thickness_typed_bits": _f32_bits(DECOMPOSITION_PARAMS["min_thickness_m"])
        == EXPECTED_MIN_THICKNESS_BITS,
        "surface_tolerance_0p1mm": d339.LIVE_SURFACE_PARITY_TOL_M == 0.0001,
        "target_delta_tolerance_0p5mm": d339.TASK_FIDELITY_TOL_MM == 0.5,
        "target_clear_gate_0p1mm": d339.CLEAR_GATE_MM == 0.1,
        "frozen_target_7_11mm": OLD_RADIAL_NM == 7_000_000
        and OLD_TANGENT_NM == 11_000_000,
        "q5_open_1p5413": math.isclose(Q5_OPEN_RAD, 1.5413, rel_tol=0.0, abs_tol=1e-12),
        "seed_33201": SEED == 33201,
    }


def _expected_rrd_contract() -> tuple[list[str], dict[str, list[str]]]:
    entities = {"metadata/run", "events/d344"}
    components: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "events/d344": ["TextLog:level", "TextLog:text"],
    }
    transform_components = [
        "Transform3D:child_frame",
        "Transform3D:parent_frame",
        "Transform3D:quaternion",
        "Transform3D:translation",
    ]
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    for body in ("link5", "gripper_link"):
        path = f"coordinate_frames/{body}/body_local"
        entities.add(path)
        components[path] = transform_components
    for name in FRAME_NAMES:
        frame_path = f"frames/{name}"
        origin_path = f"frames/{name}/origin"
        entities.update({frame_path, origin_path})
        components[frame_path] = transform_components
        components[origin_path] = [
            "CoordinateFrame:frame",
            "Points3D:labels",
            "Points3D:positions",
            "Points3D:radii",
        ]
    for body, name in PART_KEYS:
        for variant in ("source", "instance", "prototype", "candidate"):
            path = f"cook/{variant}/{body}/parts/{name}"
            metadata = f"metadata/meshes/{path.replace('/', '__')}"
            entities.update({path, metadata})
            components[path] = mesh_components
            components[metadata] = ["TextDocument:text"]
        for metric in PART_METRICS:
            path = f"metrics/{body}/{name}/{metric}"
            entities.add(path)
            components[path] = ["Scalars:scalars"]
    for body in ("link5", "gripper_link"):
        raw_path = f"cook/source/{body}/raw_reference"
        raw_metadata = f"metadata/meshes/{raw_path.replace('/', '__')}"
        entities.update({raw_path, raw_metadata})
        components[raw_path] = mesh_components
        components[raw_metadata] = ["TextDocument:text"]
        for variant in ("source", "instance", "prototype", "candidate"):
            path = f"cook/{variant}/{body}/target/cylinder"
            metadata = f"metadata/meshes/{path.replace('/', '__')}"
            entities.update({path, metadata})
            components[path] = mesh_components
            components[metadata] = ["TextDocument:text"]
        for metric in REPRESENTATION_METRICS:
            path = f"metrics/representation/{body}/{metric}"
            entities.add(path)
            components[path] = ["Scalars:scalars"]
    return sorted(entities), components


def _rrd_contract_digest() -> str:
    entities, components = _expected_rrd_contract()
    payload = {
        "exact_non_system_entity_paths": entities,
        "exact_timeline_names": ["blueprint", "event_idx", "log_time", "part_idx"],
        "required_components_by_path": components,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prereg_checks(stage: str) -> tuple[dict[str, Any], dict[str, bool]]:
    prereg = _json(PREREG_PATH)
    parameter_audit = _json(PARAMETER_AUDIT_PATH)
    inventories = _source_inventories()
    status_paths = _git_status_paths()
    exact_entities, _components = _expected_rrd_contract()
    expected_subject = {
        "part_mesh_entities": len(PART_KEYS) * 4,
        "raw_reference_mesh_entities": 2,
        "cylinder_mesh_entities": 8,
        "total_mesh_entities": len(PART_KEYS) * 4 + 10,
        "part_scalar_entities": len(PART_KEYS) * len(PART_METRICS),
        "representation_scalar_entities": 2 * len(REPRESENTATION_METRICS),
        "frame_entities": len(FRAME_NAMES) * 2,
        "exact_non_system_entities": len(exact_entities),
    }
    checks = {
        "artifact": prereg.get("artifact") == "D344_PREREGISTRATION_V1",
        "stage_registered": stage in prereg.get("registered_stages", []),
        "user_authorized": prereg.get("user_authorization", {}).get("authorized") is True,
        "new_variables_exact": prereg.get("new_variables") == NEW_VARIABLES,
        "new_variable_count_one": prereg.get("new_variable_count") == 1,
        "git_head_exact": prereg.get("git_head") == _git_head(),
        "git_status_scope_only": _status_scope_pass(status_paths),
        "source_hashes_exact": prereg.get("source_hashes") == _source_hashes(),
        "source_inventory_digests_exact": prereg.get("source_inventory_digests")
        == {name: row["digest"] for name, row in inventories.items()},
        "source_inventory_counts_exact": prereg.get("source_inventory_counts")
        == {name: row["file_count"] for name, row in inventories.items()},
        "parameter_audit_self_pass": parameter_audit.get("pass") is True,
        "parameter_audit_runtime_pass": all(_parameter_checks(parameter_audit).values()),
        "numpy_pin": str(np.__version__) == "1.26.0",
        "psutil_pin": str(psutil.__version__) == "5.9.8",
        "rerun_sdk_pin": str(rr.__version__) == RERUN_CONTRACT_VERSION == "0.34.1",
        "registered_python_executable": str(Path(sys.executable).resolve())
        == str(Path(REGISTERED_PYTHON).resolve()),
        "registered_build_core_environment_if_build": bool(
            stage != "build"
            or (
                os.environ.get("PYTHONPATH") == REGISTERED_USD_CORE
                and os.environ.get("LD_LIBRARY_PATH")
                == REGISTERED_BUILD_LD_LIBRARY_PATH
            )
        ),
        "rrd_contract_digest_exact": prereg.get("rrd_contract_sha256")
        == _rrd_contract_digest(),
        "rrd_subject_counts_exact": prereg.get("rerun_subject_counts") == expected_subject,
        "exact_changed_part_set": prereg.get("changed_parts")
        == {body: list(FAILING_PARTS[body]) for body in ("link5", "gripper_link")},
        "physics_forbidden": prereg.get("scope_guards", {}).get("physics_steps") == 0,
        "settle_forbidden": prereg.get("scope_guards", {}).get("settle") is False,
        "g0a_stays_false": prereg.get("scope_guards", {}).get("g0a_pass") is False,
    }
    report = {
        "artifact": f"D344_{stage.upper()}_PREFLIGHT",
        "stage": stage,
        "git_head": _git_head(),
        "git_status_paths": status_paths,
        "source_inventories": inventories,
        "source_hashes": _source_hashes(),
        "parameter_checks": _parameter_checks(parameter_audit),
        "checks": checks,
        "pass": all(checks.values()),
    }
    return report, checks


def _candidate_eligibility() -> tuple[dict[str, Any], dict[str, Any]]:
    capture = _json(D340_CANDIDATES)
    d342 = _json(D342_EVIDENCE)
    d343 = _json(D343_EVIDENCE)
    d343_summary = _json(D343_SUMMARY)
    capture_by_key = {(row["body"], row["name"]): row for row in capture["parts"]}
    d342_by_key = {(row["body"], row["name"]): row for row in d342["parts"]}
    d343_by_key = {(row["body"], row["name"]): row for row in d343["parts"]}
    rows = []
    adapter = copy.deepcopy(capture)
    adapter_by_key = {(row["body"], row["name"]): row for row in adapter["parts"]}
    for key in sorted(CHANGED_KEYS):
        d340_row = capture_by_key.get(key)
        d342_row = d342_by_key.get(key)
        d343_row = d343_by_key.get(key)
        direct_false = (
            sorted(name for name, passed in d342_row["direct_checks"].items() if not passed)
            if d342_row is not None
            else ["missing_d342_row"]
        )
        d340_false = (
            sorted(name for name, passed in d340_row["checks"].items() if not passed)
            if d340_row is not None
            else ["missing_d340_row"]
        )
        checks = {
            "d340_row_found": d340_row is not None,
            "d340_historical_row_fail_retained": bool(d340_row is not None and not d340_row["pass"]),
            "d340_only_mixed_stream_hash_false": d340_false
            == ["authored_hash_matches_d339_manifest"],
            "d340_fixed_point_subevidence_pass": bool(
                d340_row is not None and d340_row["fixed_point"]["pass"]
            ),
            "d340_candidate_present": bool(
                d340_row is not None and d340_row["fixed_point"].get("candidate_x1")
            ),
            "d342_row_found": d342_row is not None,
            "d342_only_unregistered_scalar_comparator_false": direct_false
            == ["min_thickness_frozen_1e_4m"],
            "d342_numeric_pass": bool(d342_row is not None and d342_row["numeric_pass"]),
            "d342_legacy_mixed_hash_rejected": bool(
                d342_row is not None and d342_row["negative_domain_control"]["rejected"]
            ),
            "d343_row_found": d343_row is not None,
            "d343_exact_typed_scalar_pass": bool(d343_row is not None and d343_row["pass"]),
            "d343_bits_exact": bool(
                d343_row is not None
                and d343_row["actual_float32_bits_hex"] == EXPECTED_MIN_THICKNESS_HEX
            ),
        }
        row_pass = all(checks.values())
        rows.append(
            {
                "body": key[0],
                "name": key[1],
                "historical_d340_row_pass": None if d340_row is None else d340_row["pass"],
                "historical_d342_row_pass": None if d342_row is None else d342_row["pass"],
                "checks": checks,
                "pass": row_pass,
            }
        )
        if row_pass:
            adapter_by_key[key]["pass"] = True
    global_checks = {
        "exact_13_rows": len(rows) == 13,
        "all_rows_eligible": all(row["pass"] for row in rows),
        "d340_historical_top_fail_retained": capture.get("pass") is False,
        "d340_only_all_rows_gate_false": sorted(
            name for name, passed in capture.get("checks", {}).items() if not passed
        )
        == ["all_13_capture_candidates_pass"],
        "d342_historical_fail_retained": d342.get("scientific_pass") is False,
        "d343_full_contract_pass": d343.get("pass") is True,
        "d343_typed_128_of_128": d343.get("typed_attribute_pass_count") == 128,
        "d343_subset_13_of_13": d343.get("d342_subset_anchor_pass_count") == 13,
        "d343_separate_approval_eligible": d343_summary.get(
            "d344_attempt3_eligible_for_separate_approval"
        )
        is True,
        "current_prereg_user_authorized": _json(PREREG_PATH)
        .get("user_authorization", {})
        .get("authorized")
        is True,
    }
    eligible = all(global_checks.values())
    adapter["historical_artifact"] = adapter.get("artifact")
    adapter["artifact"] = "D344_ELIGIBILITY_ADAPTER_IN_MEMORY_ONLY"
    adapter["historical_d340_top_pass"] = capture.get("pass")
    adapter["pass"] = eligible
    report = {
        "artifact": "D344_CANDIDATE_ELIGIBILITY_V1",
        "new_variables": NEW_VARIABLES,
        "historical_verdicts_unchanged": {
            "d340_pass": capture.get("pass"),
            "d342_scientific_pass": d342.get("scientific_pass"),
            "d343_pass": d343.get("pass"),
        },
        "source_hashes": {
            "d340_candidates": _sha256(D340_CANDIDATES),
            "d342_evidence": _sha256(D342_EVIDENCE),
            "d343_evidence": _sha256(D343_EVIDENCE),
            "d343_summary": _sha256(D343_SUMMARY),
        },
        "parts": rows,
        "global_checks": global_checks,
        "pass": eligible,
        "interpretation": (
            "D340/D342 failures remain historical facts. Only their independently repaired "
            "fixed-point, direct-coordinate, and typed-float subevidence is joined here."
        ),
    }
    return report, adapter


def _typed_scalar_audit(physics_path: Path) -> dict[str, Any]:
    from pxr import Sdf, Usd, UsdGeom

    layer = Sdf.Layer.FindOrOpen(str(physics_path))
    stage = Usd.Stage.Open(str(physics_path), load=Usd.Stage.LoadAll)
    if layer is None or stage is None:
        raise RuntimeError(f"failed to open typed-scalar audit source: {physics_path}")
    rows = []
    for body, name in PART_KEYS:
        prim_path = d340._part_spec_path(body, name)
        attr_path = Sdf.Path(f"{prim_path}.physxConvexHullCollision:minThickness")
        prim_spec = layer.GetPrimAtPath(prim_path)
        api_list_op = (
            prim_spec.GetInfo("apiSchemas")
            if prim_spec is not None and prim_spec.HasInfo("apiSchemas")
            else None
        )
        api_schemas = (
            [str(item) for item in api_list_op.GetAppliedItems()]
            if api_list_op is not None
            else []
        )
        spec = layer.GetAttributeAtPath(attr_path)
        prim = stage.GetPrimAtPath(prim_path)
        attr = prim.GetAttribute("physxConvexHullCollision:minThickness")
        value = attr.Get() if attr.IsValid() else None
        direct_value = None if spec is None else spec.default
        bits = None if value is None else _f32_bits(value)
        direct_bits = None if direct_value is None else _f32_bits(direct_value)
        resolve_info = attr.GetResolveInfo() if attr.IsValid() else None
        resolve_source = resolve_info.GetSource() if resolve_info is not None else None
        property_stack = list(attr.GetPropertyStack()) if attr.IsValid() else []
        property_stack_exact = bool(
            len(property_stack) == 1
            and Path(property_stack[0].layer.realPath).resolve() == physics_path.resolve()
            and str(property_stack[0].path) == str(attr_path)
            and str(property_stack[0].typeName) == "float"
            and property_stack[0].HasInfo(Sdf.AttributeSpec.DefaultValueKey)
            and _f32_bits(
                property_stack[0].GetInfo(Sdf.AttributeSpec.DefaultValueKey)
            )
            == EXPECTED_MIN_THICKNESS_BITS
        )
        checks = {
            "direct_prim_spec_exists": prim_spec is not None,
            "direct_api_schema_authored": "PhysxConvexHullCollisionAPI" in api_schemas,
            "direct_spec_exists": spec is not None,
            "direct_type_float": bool(spec is not None and str(spec.typeName) == "float"),
            "direct_default_authored": bool(
                spec is not None and spec.HasInfo(Sdf.AttributeSpec.DefaultValueKey)
            ),
            "direct_bits_exact": direct_bits == EXPECTED_MIN_THICKNESS_BITS,
            "composed_prim_valid": prim.IsValid(),
            "composed_attr_valid": attr.IsValid(),
            "composed_type_float": bool(attr.IsValid() and str(attr.GetTypeName()) == "float"),
            "composed_authored_opinion": bool(attr.IsValid() and attr.HasAuthoredValueOpinion()),
            "composed_authored_value": bool(attr.IsValid() and attr.HasAuthoredValue()),
            "composed_bits_exact": bits == EXPECTED_MIN_THICKNESS_BITS,
            "resolve_source_authored_default": resolve_source == Usd.ResolveInfoSourceDefault,
            "resolve_source_not_fallback": resolve_source != Usd.ResolveInfoSourceFallback,
            "resolved_value_not_blocked": bool(
                resolve_info is not None and not resolve_info.ValueIsBlocked()
            ),
            "zero_time_samples": bool(attr.IsValid() and attr.GetNumTimeSamples() == 0),
            "not_time_varying": bool(
                attr.IsValid() and not attr.ValueMightBeTimeVarying()
            ),
            "property_stack_exactly_one_direct_spec": property_stack_exact,
            "compatibility_tolerance": bool(
                value is not None
                and math.isclose(
                    float(value),
                    DECOMPOSITION_PARAMS["min_thickness_m"],
                    rel_tol=0.0,
                    abs_tol=FROZEN_COMPATIBILITY_TOL_M,
                )
            ),
        }
        rows.append(
            {
                "body": body,
                "name": name,
                "prim_path": prim_path,
                "value_m": None if value is None else float(value),
                "bits_hex": None if bits is None else f"0x{bits:08x}",
                "direct_bits_hex": None if direct_bits is None else f"0x{direct_bits:08x}",
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    checks = {
        "part_count_128": len(rows) == 128,
        "all_parts_pass": all(row["pass"] for row in rows),
        "unique_bits_exact": sorted({row["bits_hex"] for row in rows})
        == [EXPECTED_MIN_THICKNESS_HEX],
        "stage_meters_per_unit_one": math.isclose(
            float(UsdGeom.GetStageMetersPerUnit(stage)), 1.0, rel_tol=0.0, abs_tol=0.0
        ),
    }
    return {
        "artifact": "D344_ATTEMPT3_TYPED_FLOAT_AUDIT",
        "expected_bits_hex": EXPECTED_MIN_THICKNESS_HEX,
        "expected_typed_value_m": EXPECTED_MIN_THICKNESS_TYPED_M,
        "rows": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _part_layer_record_core(physics_path: Path, body: str, name: str) -> dict[str, Any]:
    """D340 part record without importing the Kit-only PhysxSchema module."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(physics_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open physics layer {physics_path}")
    prim = stage.GetPrimAtPath(d340._part_spec_path(body, name))
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"missing authored part {d340._part_spec_path(body, name)}")
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(
        [[float(value) for value in point] for point in list(mesh.GetPointsAttr().Get() or [])],
        dtype=np.float32,
    )
    counts = np.asarray(list(mesh.GetFaceVertexCountsAttr().Get() or []), dtype=np.int64)
    indices = np.asarray(list(mesh.GetFaceVertexIndicesAttr().Get() or []), dtype=np.int64)
    canonical = d339._canonical_convex(points.astype(np.float64))
    mesh_api = UsdPhysics.MeshCollisionAPI(prim)
    collision = UsdPhysics.CollisionAPI(prim)
    collision_enabled = collision.GetCollisionEnabledAttr().Get()
    hull_vertex_limit = prim.GetAttribute(
        "physxConvexHullCollision:hullVertexLimit"
    ).Get()
    min_thickness = prim.GetAttribute("physxConvexHullCollision:minThickness").Get()
    return {
        "path": d340._part_spec_path(body, name),
        "points_f32_sha256": _f32_sha(points),
        "points_f32": points.astype(np.float64).tolist(),
        "face_vertex_counts": counts.tolist(),
        "face_vertex_indices": indices.tolist(),
        "face_vertex_counts_sha256": _i8_sha(counts),
        "face_vertex_indices_sha256": _i8_sha(indices),
        "canonical": d340._public_convex(canonical),
        "subdivision_scheme": str(mesh.GetSubdivisionSchemeAttr().Get()),
        "double_sided": bool(mesh.GetDoubleSidedAttr().Get()),
        "collision_enabled": True if collision_enabled is None else bool(collision_enabled),
        "approximation": str(mesh_api.GetApproximationAttr().Get()),
        "hull_vertex_limit": int(hull_vertex_limit),
        "min_thickness_m": float(min_thickness),
    }


def _run_build(args: argparse.Namespace) -> int:
    preflight, checks = _prereg_checks("build")
    checks.update(
        {
            "attempt3_absent": not (OUT_DIR / "collision_asset/attempt3").exists(),
            "build_summary_absent": not BUILD_SUMMARY_PATH.exists(),
            "eligibility_absent": not ELIGIBILITY_PATH.exists(),
        }
    )
    preflight["checks"] = checks
    preflight["pass"] = all(checks.values())
    _write_json(BUILD_PREFLIGHT_PATH, preflight)
    if not preflight["pass"]:
        return 2

    eligibility, adapter = _candidate_eligibility()
    _write_json(ELIGIBILITY_PATH, eligibility)
    if not eligibility["pass"]:
        summary = {
            "artifact": "D344_BUILD_SUMMARY_V1",
            "verdict": VERDICT_BUILD_FAIL,
            "pass": False,
            "reason": "candidate eligibility contract failed before asset creation",
            "controlled_physics_steps": 0,
            "runtime_env_created": False,
        }
        _write_json(BUILD_SUMMARY_PATH, summary)
        return 2

    args.out_dir = OUT_DIR
    args.process_nonce = secrets.token_hex(16)
    original_part_layer_record = d340._part_layer_record
    d340._part_layer_record = _part_layer_record_core
    try:
        core_manifest = d340._author_attempt3(args, adapter)
    finally:
        d340._part_layer_record = original_part_layer_record
    typed_scalar = _typed_scalar_audit(VARIANT_PHYSICS_USD)
    checks = {
        "core_build_pass": core_manifest.get("pass") is True,
        "eligibility_pass": eligibility["pass"] is True,
        "typed_scalar_128_pass": typed_scalar["pass"] is True,
        "exact_13_changed": core_manifest.get("changed_part_count") == 13,
        "exact_115_preserved": core_manifest.get("preserved_part_count") == 115,
        "parameters_increased_empty": core_manifest.get("parameters_increased") == [],
        "decomposition_parameters_frozen": core_manifest.get("decomposition_parameters")
        == DECOMPOSITION_PARAMS,
        "variant_robot_exists": VARIANT_ROBOT_USD.is_file(),
        "variant_physics_exists": VARIANT_PHYSICS_USD.is_file(),
        "d338_attempt1_immutable": d339._d338_attempt1_integrity()["pass"],
        "d339_attempt2_immutable": d340._attempt2_integrity()["pass"],
    }
    d344_manifest = {
        "artifact": "D344_ATTEMPT3_ASSET_MANIFEST_V1",
        "new_variables": NEW_VARIABLES,
        "parameters_increased": [],
        "parameters_changed": [],
        "thresholds_relaxed": [],
        "source_attempt2_dir": _relative(d340.D339_ASSET_DIR),
        "variant_asset_dir": _relative(VARIANT_DIR),
        "variant_robot_usd": _relative(VARIANT_ROBOT_USD),
        "variant_physics_usd": _relative(VARIANT_PHYSICS_USD),
        "core_helper_manifest": _relative(CORE_BUILD_MANIFEST_PATH),
        "core_helper_manifest_sha256": _sha256(CORE_BUILD_MANIFEST_PATH),
        "core_helper_lineage": (
            "D340 low-level one-write/preservation helper; D340 historical verdict is unchanged"
        ),
        "eligibility": eligibility,
        "typed_scalar_audit": typed_scalar,
        "decomposition_parameters": DECOMPOSITION_PARAMS,
        "changed_parts": {body: list(FAILING_PARTS[body]) for body in FAILING_PARTS},
        "changed_part_count": core_manifest.get("changed_part_count"),
        "preserved_part_count": core_manifest.get("preserved_part_count"),
        "parts": core_manifest.get("parts"),
        "part_audits": core_manifest.get("part_audits"),
        "source_tool_mass_semantics": core_manifest.get("source_tool_mass_semantics"),
        "variant_tool_mass_semantics": core_manifest.get("variant_tool_mass_semantics"),
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json(D344_BUILD_MANIFEST_PATH, d344_manifest)
    summary = {
        "artifact": "D344_BUILD_SUMMARY_V1",
        "verdict": "D344_ATTEMPT3_BUILD_PASS_VALIDATE_PENDING"
        if d344_manifest["pass"]
        else VERDICT_BUILD_FAIL,
        "pass": d344_manifest["pass"],
        "new_variables": NEW_VARIABLES,
        "process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "attempt3_manifest": _relative(D344_BUILD_MANIFEST_PATH),
        "attempt3_manifest_sha256": _sha256(D344_BUILD_MANIFEST_PATH),
        "core_manifest": _relative(CORE_BUILD_MANIFEST_PATH),
        "core_manifest_sha256": _sha256(CORE_BUILD_MANIFEST_PATH),
        "changed_part_count": d344_manifest["changed_part_count"],
        "preserved_part_count": d344_manifest["preserved_part_count"],
        "typed_scalar_pass_count": sum(
            row["pass"] for row in typed_scalar["rows"]
        ),
        "controlled_physics_steps": 0,
        "runtime_env_created": False,
        "g0a_pass": False,
    }
    _write_json(BUILD_SUMMARY_PATH, summary)
    _write_text(
        BUILD_REPORT_PATH,
        "# D344 자산 생성 자동 보고\n\n"
        f"- 판정: `{summary['verdict']}`\n"
        f"- 바꾼 조각: `{summary['changed_part_count']}`\n"
        f"- 그대로 보존한 조각: `{summary['preserved_part_count']}`\n"
        f"- 최소 두께 정확 판독: `{summary['typed_scalar_pass_count']}/128`\n"
        "- 물리 진행: `0`\n",
    )
    return 0 if d344_manifest["pass"] else 2


def _direct_record_checks(
    build_row: dict[str, Any],
    candidate_row: dict[str, Any] | None,
    *,
    changed: bool,
) -> dict[str, bool]:
    variant = build_row["variant"]
    expected = build_row["source"] if not changed else variant
    checks = {
        "build_part_audit_pass": build_row.get("pass") is True,
        "variant_points_f32_sha_consistent": variant["points_f32_sha256"]
        == _f32_sha(variant["points_f32"]),
        "variant_face_counts_i8_sha_consistent": variant["face_vertex_counts_sha256"]
        == _i8_sha(variant["face_vertex_counts"]),
        "variant_face_indices_i8_sha_consistent": variant["face_vertex_indices_sha256"]
        == _i8_sha(variant["face_vertex_indices"]),
        "variant_geometry_matches_manifest": variant["canonical"]["geometry_sha256"]
        == expected["canonical"]["geometry_sha256"],
    }
    if changed:
        candidate = None if candidate_row is None else candidate_row["fixed_point"]["candidate_x1"]
        candidate_points = (
            np.empty((0, 3), dtype=np.float32)
            if candidate is None
            else np.asarray(candidate["vertices_m"], dtype=np.float32)
        )
        candidate_triangles = (
            np.empty((0, 3), dtype=np.int64)
            if candidate is None
            else np.asarray(candidate["triangles"], dtype=np.int64)
        )
        checks.update(
            {
                "candidate_present": candidate is not None,
                "direct_points_exact_candidate": bool(
                    candidate is not None
                    and np.array_equal(
                        np.asarray(variant["points_f32"], dtype=np.float32),
                        candidate_points,
                    )
                ),
                "direct_face_counts_exact_candidate": bool(
                    candidate is not None
                    and np.array_equal(
                        np.asarray(variant["face_vertex_counts"], dtype=np.int64),
                        np.full(len(candidate_triangles), 3, dtype=np.int64),
                    )
                ),
                "direct_face_indices_exact_candidate": bool(
                    candidate is not None
                    and np.array_equal(
                        np.asarray(variant["face_vertex_indices"], dtype=np.int64),
                        candidate_triangles.reshape(-1),
                    )
                ),
            }
        )
    else:
        source = build_row["source"]
        checks.update(
            {
                "preserved_points_f32_exact_source": variant["points_f32_sha256"]
                == source["points_f32_sha256"],
                "preserved_face_counts_exact_source": variant["face_vertex_counts_sha256"]
                == source["face_vertex_counts_sha256"],
                "preserved_face_indices_exact_source": variant["face_vertex_indices_sha256"]
                == source["face_vertex_indices_sha256"],
                "preserved_canonical_exact_source": variant["canonical"] == source["canonical"],
            }
        )
    return checks


def _canonical_from_public(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    vertices = np.asarray(value["vertices_m"], dtype=np.float64)
    triangles = np.asarray(value["triangles"], dtype=np.int64)
    canonical = d339._canonical_convex(vertices)
    if not np.array_equal(canonical["triangles"], triangles):
        # The JSON topology is still authoritative for display.  Scientific
        # solid distances use the canonical convex reconstructed from points.
        canonical["reported_triangles"] = triangles
    return canonical


def _fixed_point_checks(
    body: str,
    name: str,
    candidate_row: dict[str, Any] | None,
    mapped_authored: dict[str, Any] | None,
    instance: dict[str, Any] | None,
    prototype: dict[str, Any] | None,
    consensus: dict[str, Any] | None,
    direct_checks: dict[str, bool],
) -> tuple[dict[str, Any], bool]:
    changed = (body, name) in CHANGED_KEYS
    if not changed:
        checks = {
            "registered_preserved_part": True,
            "direct_stream_preserved": all(direct_checks.values()),
        }
        return {"kind": "preserved", "checks": checks, "pass": all(checks.values())}, all(
            checks.values()
        )

    x0 = None if candidate_row is None else candidate_row["fixed_point"]["authored_x0"]
    x1 = None if candidate_row is None else candidate_row["fixed_point"]["candidate_x1"]
    x0_convex = _canonical_from_public(x0)
    x1_convex = _canonical_from_public(x1)

    def _surface(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float:
        if a is None or b is None:
            return math.inf
        return max(
            float(d339._directed_convex_solid_distance_m(a["vertices"], b)),
            float(d339._directed_convex_solid_distance_m(b["vertices"], a)),
        )

    mapped_to_x1 = _surface(mapped_authored, x1_convex)
    instance_to_x1 = _surface(instance, x1_convex)
    prototype_to_x1 = _surface(prototype, x1_convex)
    consensus_to_x1 = _surface(consensus, x1_convex)
    x0_to_consensus = _surface(x0_convex, consensus)
    checks = {
        "candidate_row_found": candidate_row is not None,
        "direct_authored_stream_exact_candidate": all(direct_checks.values()),
        "mapped_attempt3_matches_x1_le_1e_9m": mapped_to_x1 <= FIXED_POINT_TOL_M,
        "instance_Fx1_matches_x1_le_1e_9m": instance_to_x1 <= FIXED_POINT_TOL_M,
        "prototype_Fx1_matches_x1_le_1e_9m": prototype_to_x1 <= FIXED_POINT_TOL_M,
        "consensus_Fx1_matches_x1_le_1e_9m": consensus_to_x1 <= FIXED_POINT_TOL_M,
        "no_second_vertex_decrease": bool(
            consensus is not None
            and x1_convex is not None
            and consensus["vertex_count"] >= x1_convex["vertex_count"]
        ),
        "not_two_cycle_back_to_x0": bool(
            consensus is not None
            and x0_convex is not None
            and (
                consensus["vertex_count"] != x0_convex["vertex_count"]
                or x0_to_consensus > FIXED_POINT_TOL_M
            )
        ),
        "single_authoring_application": True,
        "iterative_retry_forbidden": True,
    }
    payload = {
        "kind": "changed_fixed_point",
        "mapped_attempt3_vs_x1_surface_m": mapped_to_x1,
        "instance_vs_x1_surface_m": instance_to_x1,
        "prototype_vs_x1_surface_m": prototype_to_x1,
        "consensus_vs_x1_surface_m": consensus_to_x1,
        "x0_vs_consensus_surface_m": x0_to_consensus,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return payload, payload["pass"]


def _reclassify_live_measurement(
    inner: Any,
    core_manifest: dict[str, Any],
    raw_audit: dict[str, Any],
    capture: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import hppfcl
    from pxr import PhysxSchema

    build_by_key = {
        (row["body"], row["name"]): row for row in core_manifest["part_audits"]
    }
    candidate_by_key = {(row["body"], row["name"]): row for row in capture["parts"]}
    cooked_by_body: dict[str, list[dict[str, Any]]] = {body: [] for body in d334.BODY_LABELS}
    audit: dict[str, Any] = {
        "artifact": "D344_FRESH_LIVE_REPRESENTATION_AUDIT_V1",
        "classification_contract": {
            "direct_authored_domain": "ordered Vec3f/count/index bytes before transform",
            "mapped_geometry_domain": "body-local solid distance only; no exact hash",
            "typed_scalar_identity": EXPECTED_MIN_THICKNESS_HEX,
            "surface_tolerance_m": d339.LIVE_SURFACE_PARITY_TOL_M,
            "property_volume_relative_tolerance": d339.PROPERTY_VOLUME_BINDING_REL_TOL,
        },
        "raw_measurement_helper": (
            "D340 callback/property acquisition only; its historical mixed-domain classification "
            "is not reused"
        ),
        "per_body": {},
    }
    for body in d334.BODY_LABELS:
        raw_body = raw_audit.get("per_body", {}).get(body, {})
        live_inventory = {
            Path(row["path"]).name: row
            for row in raw_body.get("usd_inventory", [])
            if row.get("collision_enabled")
        }
        corrected_rows = []
        for raw_row in raw_body.get("part_checks", []):
            name = raw_row["name"]
            key = (body, name)
            build_row = build_by_key.get(key)
            candidate_row = candidate_by_key.get(key)
            inventory_row = live_inventory.get(name)
            direct_checks = (
                {"build_row_found": False}
                if build_row is None
                else _direct_record_checks(
                    build_row,
                    candidate_row,
                    changed=key in CHANGED_KEYS,
                )
            )
            try:
                source = d334._source_mesh_body_local(inner, inventory_row, body)
                mapped_authored = d339._canonical_convex(source["_verts_body"])
                mapped_error = None
            except Exception as error:
                mapped_authored = None
                mapped_error = f"{type(error).__name__}: {error}"

            consensus_public = raw_row.get("channel_consensus", {}).get("consensus")
            instance_public = raw_row.get("channel_consensus", {}).get("instance", {}).get(
                "canonical"
            )
            prototype_public = raw_row.get("channel_consensus", {}).get("prototype", {}).get(
                "canonical"
            )
            consensus = _canonical_from_public(consensus_public)
            instance = _canonical_from_public(instance_public)
            prototype = _canonical_from_public(prototype_public)
            fixed_point, fixed_pass = _fixed_point_checks(
                body,
                name,
                candidate_row,
                mapped_authored,
                instance,
                prototype,
                consensus,
                direct_checks,
            )
            prim = inner.scene.stage.GetPrimAtPath(raw_row["path"])
            attr = prim.GetAttribute("physxConvexHullCollision:minThickness")
            value = attr.Get() if attr.IsValid() else None
            bits = None if value is None else _f32_bits(value)
            typed_scalar = {
                "type_name": str(attr.GetTypeName()) if attr.IsValid() else None,
                "value_m": None if value is None else float(value),
                "bits_hex": None if bits is None else f"0x{bits:08x}",
                "checks": {
                    "attr_valid": attr.IsValid(),
                    "type_float": bool(attr.IsValid() and str(attr.GetTypeName()) == "float"),
                    "authored_opinion": bool(attr.IsValid() and attr.HasAuthoredValueOpinion()),
                    "bits_exact": bits == EXPECTED_MIN_THICKNESS_BITS,
                    "compatibility_tolerance": bool(
                        value is not None
                        and math.isclose(
                            float(value),
                            0.0001,
                            rel_tol=0.0,
                            abs_tol=FROZEN_COMPATIBILITY_TOL_M,
                        )
                    ),
                },
            }
            typed_scalar["pass"] = all(typed_scalar["checks"].values())
            inherited_checks = {
                key: value
                for key, value in raw_row.get("checks", {}).items()
                if key
                not in {
                    "authored_hash_matches_attempt3_manifest",
                    "min_thickness_frozen_0p0001m",
                    "changed_part_fixed_point_or_preserved",
                }
            }
            corrected_checks = {
                **inherited_checks,
                "direct_authored_stream_contract": all(direct_checks.values()),
                "typed_min_thickness_exact_bits": typed_scalar["pass"],
                "fixed_point_or_preserved_contract": fixed_pass,
            }
            part_pass = all(corrected_checks.values())
            if part_pass and consensus is not None:
                points = d332._fcl_points(hppfcl, consensus["vertices"])
                geometry = hppfcl.Convex.convexHull(points, False, "")
                if geometry is None:
                    raise RuntimeError(f"hppfcl reconstruction failed: {body}/{name}")
                cooked_by_body[body].append(
                    {
                        "body": body,
                        "path": raw_row["path"],
                        "_vertices": consensus["vertices"],
                        "_triangles": consensus["triangles"],
                        "_geometry": geometry,
                    }
                )
            surface_values = [
                value.get("symmetric_m")
                for value in raw_row.get("channel_surfaces", {}).values()
                if value.get("symmetric_m") is not None
            ]
            volume_values = [
                float(value)
                for value in raw_row.get(
                    "property_vs_channel_volume_relative_difference", {}
                ).values()
                if math.isfinite(float(value))
            ]
            corrected_rows.append(
                {
                    "body": body,
                    "name": name,
                    "path": raw_row["path"],
                    "collision_enabled": bool(
                        inventory_row is not None and inventory_row["collision_enabled"]
                    ),
                    "mapped_authored_error": mapped_error,
                    "direct_stream": direct_checks,
                    "typed_scalar": typed_scalar,
                    "fixed_point": fixed_point,
                    "channel_consensus": raw_row.get("channel_consensus"),
                    "channel_surfaces": raw_row.get("channel_surfaces"),
                    "property_vs_channel_volume_relative_difference": raw_row.get(
                        "property_vs_channel_volume_relative_difference"
                    ),
                    "max_surface_error_m": max(surface_values) if surface_values else None,
                    "max_property_volume_relative_error": (
                        max(volume_values) if volume_values else None
                    ),
                    "checks": corrected_checks,
                    "pass": part_pass,
                }
            )
        inventory = raw_body.get("usd_inventory", [])
        enabled = [row for row in inventory if row.get("collision_enabled")]
        disabled = [row for row in inventory if not row.get("collision_enabled")]
        expected_names = [row["name"] for row in core_manifest["parts"][body]]
        expected_paths = d340._expected_live_paths(core_manifest, body)
        corrected_names = [row["name"] for row in corrected_rows]
        corrected_paths = [row["path"] for row in corrected_rows]
        enabled_paths = sorted(str(row["path"]) for row in enabled)
        property_paths = sorted(
            str(row["path"])
            for row in raw_body.get("property_query", {}).get("colliders", [])
        )
        expected_property_paths = sorted(
            [*expected_paths, d339.LIVE_OLD_COLLIDER_PATHS[body]]
        )
        body_checks = {
            "raw_measurement_body_present": bool(raw_body),
            "usd_inventory_exact_65": len(inventory) == 65,
            "enabled_exact_64": len(enabled) == 64,
            "disabled_exact_known_legacy": len(disabled) == 1
            and disabled[0]["path"] == d339.LIVE_OLD_COLLIDER_PATHS[body],
            "property_query_pass": bool(raw_body.get("property_query", {}).get("pass")),
            "property_inventory_exact_65": len(
                raw_body.get("property_query", {}).get("colliders", [])
            )
            == 65,
            "corrected_part_rows_64": len(corrected_rows) == 64,
            "corrected_names_exact": corrected_names == expected_names,
            "corrected_names_unique": len(set(corrected_names)) == 64,
            "corrected_paths_exact": corrected_paths == expected_paths,
            "corrected_paths_unique": len(set(corrected_paths)) == 64,
            "enabled_paths_exact": enabled_paths == expected_paths,
            "enabled_paths_unique": len(set(enabled_paths)) == 64,
            "property_paths_exact_64_plus_disabled_legacy": property_paths
            == expected_property_paths,
            "property_paths_unique": len(set(property_paths)) == 65,
            "all_corrected_parts_pass": bool(corrected_rows)
            and all(row["pass"] for row in corrected_rows),
            "certified_parts_64": len(cooked_by_body[body]) == 64,
            "changed_fixed_points_exact": sum(
                row["fixed_point"]["kind"] == "changed_fixed_point"
                and row["fixed_point"]["pass"]
                for row in corrected_rows
            )
            == len(FAILING_PARTS[body]),
            "mass_semantics_preserved": bool(
                raw_body.get("checks", {}).get("live_mass_semantics_equal")
            ),
            "property_mass_preserved": bool(
                raw_body.get("checks", {}).get("property_query_mass_equal")
            ),
        }
        audit["per_body"][body] = {
            "checks": body_checks,
            "part_checks": corrected_rows,
            "usd_inventory": inventory,
            "property_query": raw_body.get("property_query"),
            "pass": all(body_checks.values()),
        }
    state_guard = raw_audit.get("asset_validator_state_guard", {})
    checks = {
        "raw_callback_measurement_has_two_bodies": sorted(raw_audit.get("per_body", {}))
        == sorted(d334.BODY_LABELS),
        "raw_callback_request_order_inverted_from_capture": raw_audit.get("request_order")
        == ["prototype", "instance"],
        "asset_validator_state_guard": not bool(state_guard.get("violated", True)),
        "surface_certified_128_of_128": sum(
            row["checks"].get("both_channel_surface_le_0p1mm", False)
            for body in audit["per_body"].values()
            for row in body["part_checks"]
        )
        == 128,
        "property_volume_bound_128_of_128": sum(
            row["checks"].get("property_vs_both_channel_volume_le_5pct", False)
            for body in audit["per_body"].values()
            for row in body["part_checks"]
        )
        == 128,
        "typed_scalar_bits_128_of_128": sum(
            row["typed_scalar"]["pass"]
            for body in audit["per_body"].values()
            for row in body["part_checks"]
        )
        == 128,
        "all_parts_corrected_pass_128_of_128": sum(
            row["pass"]
            for body in audit["per_body"].values()
            for row in body["part_checks"]
        )
        == 128,
        "both_bodies_pass": all(body["pass"] for body in audit["per_body"].values()),
    }
    audit["checks"] = checks
    audit["pass"] = all(checks.values())
    return cooked_by_body, audit


def _witness_manifest() -> dict[str, Any]:
    expected = {
        f"{body}_part_{index:03d}_{channel}.json"
        for body in d334.BODY_LABELS
        for index in range(64)
        for channel in ("prototype", "instance")
    }
    observed_paths = sorted(WITNESS_DIR.glob("*.json")) if WITNESS_DIR.is_dir() else []
    observed = {path.name for path in observed_paths}
    checks = {
        "exact_256_filenames": observed == expected,
        "all_nonzero": all(path.stat().st_size > 0 for path in observed_paths),
        "all_unique_sha256": len({_sha256(path) for path in observed_paths}) == len(observed_paths),
    }
    return {
        "artifact": "D344_FRESH_COOK_WITNESS_MANIFEST_V1",
        "expected_count": len(expected),
        "observed_count": len(observed),
        "expected_filenames": sorted(expected),
        "observed_filenames": sorted(observed),
        "sha256": {path.name: _sha256(path) for path in observed_paths},
        "checks": checks,
        "pass": all(checks.values()),
    }


def _cylinder_world_mesh(inner: Any, *, sides: int = 64) -> tuple[np.ndarray, np.ndarray]:
    radius = float(d332.CYLINDER_RADIUS_M)
    half_height = 0.5 * float(d332.CYLINDER_HEIGHT_M)
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    local = []
    for z in (-half_height, half_height):
        local.extend([[radius * math.cos(a), radius * math.sin(a), z] for a in angles])
    local.extend([[0.0, 0.0, -half_height], [0.0, 0.0, half_height]])
    triangles = []
    bottom_center = 2 * sides
    top_center = 2 * sides + 1
    for index in range(sides):
        nxt = (index + 1) % sides
        triangles.extend(
            [
                [index, nxt, sides + nxt],
                [index, sides + nxt, sides + index],
                [bottom_center, nxt, index],
                [top_center, sides + index, sides + nxt],
            ]
        )
    obj_pos_w, obj_quat = d334._object_pose_w(inner)
    origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    rotation = d332._quat_wxyz_to_rot(obj_quat)
    world = (rotation @ np.asarray(local, dtype=np.float64).T).T + obj_pos_w - origin
    return world, np.asarray(triangles, dtype=np.int64)


def _rerun_rows(
    inner: Any,
    core_manifest: dict[str, Any],
    raw_shapes: list[dict[str, Any]],
    live_audit: dict[str, Any],
    representation_gate: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    coordinate_frames = []
    for body in d334.BODY_LABELS:
        pos, quat = d334._body_pose_w(inner, body)
        coordinate_frames.append(
            {
                "frame": f"{body}_body_local",
                "parent_frame": "tf#/",
                "entity_path": f"coordinate_frames/{body}/body_local",
                "translation_m": (pos - origin).tolist(),
                "quaternion_xyzw": [
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                    float(quat[0]),
                ],
            }
        )
    build_by_key = {
        (row["body"], row["name"]): row for row in core_manifest["part_audits"]
    }
    live_by_key = {
        (body, row["name"]): row
        for body, body_row in live_audit.get("per_body", {}).items()
        for row in body_row.get("part_checks", [])
    }
    meshes: list[dict[str, Any]] = []
    scalars: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = [
        {
            "entity_path": "events/d344",
            "text": "D344 fresh-process 128-part live-representation audit started",
            "level": "INFO",
            "sequence": {"event_idx": 0, "part_idx": 0},
        }
    ]
    event_idx = 1
    for part_idx, key in enumerate(PART_KEYS):
        body, name = key
        build_row = build_by_key[key]
        live_row = live_by_key.get(key)
        source = build_row["source"]
        candidate = build_row["variant"]
        instance = (
            None
            if live_row is None
            else live_row.get("channel_consensus", {}).get("instance", {}).get("canonical")
        )
        prototype = (
            None
            if live_row is None
            else live_row.get("channel_consensus", {}).get("prototype", {}).get("canonical")
        )
        geometry_rows = {
            "source": {
                "vertices_m": source["points_f32"],
                "triangles": np.asarray(source["face_vertex_indices"], dtype=np.int64).reshape(-1, 3),
                "geometry_sha256": source["canonical"]["geometry_sha256"],
            },
            "candidate": {
                "vertices_m": candidate["points_f32"],
                "triangles": np.asarray(candidate["face_vertex_indices"], dtype=np.int64).reshape(-1, 3),
                "geometry_sha256": candidate["canonical"]["geometry_sha256"],
            },
            "instance": instance,
            "prototype": prototype,
        }
        for variant, geometry in geometry_rows.items():
            if geometry is None:
                continue
            meshes.append(
                {
                    "entity_path": f"cook/{variant}/{body}/parts/{name}",
                    "vertices_m": geometry["vertices_m"],
                    "triangles": geometry["triangles"],
                    "coordinate_frame": f"{body}_body_local",
                    "body": body,
                    "part": name,
                    "source_kind": variant,
                    "geometry_sha256": geometry.get("geometry_sha256"),
                    "changed_part": key in CHANGED_KEYS,
                    "candidate_equals_authored_attempt3": variant == "candidate",
                    "color_rgba": SOURCE_COLORS[variant],
                }
            )
        available = live_row is not None and instance is not None and prototype is not None
        surface = -1.0 if live_row is None or live_row["max_surface_error_m"] is None else float(
            live_row["max_surface_error_m"]
        )
        volume = (
            -1.0
            if live_row is None or live_row["max_property_volume_relative_error"] is None
            else float(live_row["max_property_volume_relative_error"])
        )
        metric_values = {
            "measurement_available": float(available),
            "max_surface_error_m": surface,
            "max_property_volume_relative_error": volume,
            "owner_matches": float(
                bool(live_row and live_row["checks"].get("owner_matches"))
            ),
            "collision_enabled": float(bool(live_row and live_row["collision_enabled"])),
            "min_thickness_bits_exact": float(
                bool(live_row and live_row["typed_scalar"]["pass"])
            ),
            "fixed_point_or_preserved": float(
                bool(live_row and live_row["fixed_point"]["pass"])
            ),
            "part_pass": float(bool(live_row and live_row["pass"])),
        }
        if tuple(metric_values) != PART_METRICS:
            raise RuntimeError("D344 per-part Rerun metric schema drift")
        for metric, value in metric_values.items():
            scalars.append(
                {
                    "entity_path": f"metrics/{body}/{name}/{metric}",
                    "value": value,
                    "sequence": {"event_idx": event_idx, "part_idx": part_idx},
                }
            )
        events.append(
            {
                "entity_path": "events/d344",
                "text": (
                    f"{body}/{name}: live={'PASS' if live_row and live_row['pass'] else 'FAIL'}; "
                    f"surface_m={surface}; property_volume_relative={volume}"
                ),
                "level": "INFO" if live_row and live_row["pass"] else "WARN",
                "sequence": {"event_idx": event_idx, "part_idx": part_idx},
            }
        )
        event_idx += 1

    cylinder_vertices, cylinder_triangles = _cylinder_world_mesh(inner)
    raw_by_body = {row["body"]: row for row in raw_shapes}
    for body_idx, body in enumerate(d334.BODY_LABELS):
        raw = raw_by_body.get(body)
        if raw is not None:
            meshes.append(
                {
                    "entity_path": f"cook/source/{body}/raw_reference",
                    "vertices_m": raw["_raw_verts"],
                    "triangles": raw["_triangles"],
                    "coordinate_frame": f"{body}_body_local",
                    "body": body,
                    "source_kind": "raw_stl_reference",
                    "color_rgba": [245, 245, 245, 28],
                }
            )
        for variant in ("source", "instance", "prototype", "candidate"):
            meshes.append(
                {
                    "entity_path": f"cook/{variant}/{body}/target/cylinder",
                    "vertices_m": cylinder_vertices,
                    "triangles": cylinder_triangles,
                    "coordinate_frame": "tf#/",
                    "body": body,
                    "source_kind": "frozen_target_cylinder",
                    "color_rgba": [250, 185, 30, 42],
                }
            )
        body_gate = representation_gate.get("per_body", {}).get(body)
        query_ran = body_gate is not None
        raw_value = None if not query_ran else body_gate.get("raw_exact_signed_distance_mm")
        live_value = None if not query_ran else body_gate.get("cooked_exact_signed_distance_mm")
        delta_value = None if not query_ran else body_gate.get("absolute_delta_mm")

        def _finite(value: Any) -> bool:
            try:
                return bool(value is not None and math.isfinite(float(value)))
            except (TypeError, ValueError):
                return False

        def _display_value(value: Any) -> float:
            return float(value) if _finite(value) else 0.0

        representation_values = {
            "query_ran": float(query_ran),
            "raw_signed_distance_mm": _display_value(raw_value),
            "raw_value_finite": float(_finite(raw_value)),
            "live_signed_distance_mm": _display_value(live_value),
            "live_value_finite": float(_finite(live_value)),
            "absolute_delta_mm": _display_value(delta_value),
            "delta_finite": float(_finite(delta_value)),
            "body_pass": float(bool(query_ran and body_gate["pass"])),
        }
        if tuple(representation_values) != REPRESENTATION_METRICS:
            raise RuntimeError("D344 representation Rerun metric schema drift")
        for metric, value in representation_values.items():
            scalars.append(
                {
                    "entity_path": f"metrics/representation/{body}/{metric}",
                    "value": value,
                    "sequence": {
                        "event_idx": event_idx,
                        "part_idx": 64 * body_idx,
                    },
                }
            )
        events.append(
            {
                "entity_path": "events/d344",
                "text": (
                    f"{body} frozen-target representation: "
                    + (
                        f"raw={body_gate['raw_exact_signed_distance_mm']}mm, "
                        f"live={body_gate['cooked_exact_signed_distance_mm']}mm, "
                        f"delta={body_gate['absolute_delta_mm']}mm, pass={body_gate['pass']}"
                        if query_ran
                        else "not queried because the 128-part live contract did not pass"
                    )
                ),
                "level": "INFO" if query_ran and body_gate["pass"] else "WARN",
                "sequence": {"event_idx": event_idx, "part_idx": 64 * body_idx},
            }
        )
        event_idx += 1
    events.append(
        {
            "entity_path": "events/d344",
            "text": "D344 STOP boundary: zero physics steps; settle/10-trial/G0b/RL forbidden",
            "level": "WARN",
            "sequence": {"event_idx": event_idx, "part_idx": 127},
        }
    )
    return coordinate_frames, meshes, scalars, events


def _rerun_contract(
    frames: list[dict[str, Any]],
    meshes: list[dict[str, Any]],
    scalars: list[dict[str, Any]],
    events: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    exact_entities, expected_components = _expected_rrd_contract()
    observed_counts = {
        "frame_count": len(frames),
        "mesh_count": len(meshes),
        "scalar_count": len(scalars),
        "event_count": len(events),
    }
    log_status = log_rerun(
        RRD_PATH,
        frames=frames,
        coordinate_frames=metadata.pop("coordinate_frames"),
        meshes=meshes,
        scalar_trace=scalars,
        events=events,
        recording_metadata=metadata,
        recording_id="g0a_d344_attempt3_fresh_live_representation",
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
                "events/d344",
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
    return {
        "log_status": log_status,
        "validation": validation,
        "observed_counts": observed_counts,
        "expected_exact_entity_count": len(exact_entities),
        "rrd_contract_sha256": _rrd_contract_digest(),
        "manual_visual_inspection_required": True,
        "manual_visual_inspection_pending": bool(validation.get("pass")),
        "pass": bool(log_status.get("ok") and validation.get("pass")),
    }


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


def _write_automated_report(summary: dict[str, Any]) -> None:
    live = summary.get("fresh_live_audit", {})
    gate = summary.get("representation_gate", {})
    rerun = summary.get("rerun", {})
    _write_text(
        AUTOMATED_REPORT_PATH,
        "# D344 자동 검사 보고\n\n"
        f"- 수치 판정: `{summary['automated_verdict']}`\n"
        f"- Rerun 완료 판정: `{summary['observability_verdict']}`\n"
        f"- 실제 충돌 조각 검사: `{sum(row['pass'] for body in live.get('per_body', {}).values() for row in body.get('part_checks', []))}/128`\n"
        f"- 목표 자세의 원본/실제 충돌체 거리 검사: `{gate.get('target_clear_and_faithful', False)}`\n"
        f"- Rerun 기계 완료 검사: `{rerun.get('pass', False)}`\n"
        f"- 물리 진행 횟수: `{summary.get('controlled_physics_steps')}`\n\n"
        "Rerun 화면 파일은 생성·구조 검사만 끝난 상태입니다. 실제 화면을 열어 본 별도 "
        "기록이 있어야 D344 완료 판정을 만들 수 있습니다.\n",
    )


def _prepare_validate_preflight(args: argparse.Namespace) -> bool:
    preflight, checks = _prereg_checks("validate")
    build_summary = _json(BUILD_SUMMARY_PATH) if BUILD_SUMMARY_PATH.is_file() else {}
    build_manifest = _json(D344_BUILD_MANIFEST_PATH) if D344_BUILD_MANIFEST_PATH.is_file() else {}
    core_manifest = _json(CORE_BUILD_MANIFEST_PATH) if CORE_BUILD_MANIFEST_PATH.is_file() else {}
    checks.update(
        {
            "build_summary_pass": build_summary.get("pass") is True,
            "d344_build_manifest_pass": build_manifest.get("pass") is True,
            "core_build_manifest_pass": core_manifest.get("pass") is True,
            "build_manifest_hash_exact": bool(
                D344_BUILD_MANIFEST_PATH.is_file()
                and build_summary.get("attempt3_manifest_sha256")
                == _sha256(D344_BUILD_MANIFEST_PATH)
            ),
            "core_manifest_hash_exact": bool(
                CORE_BUILD_MANIFEST_PATH.is_file()
                and build_summary.get("core_manifest_sha256")
                == _sha256(CORE_BUILD_MANIFEST_PATH)
            ),
            "fresh_process_pid": build_summary.get("process_identity", {}).get("pid")
            != os.getpid(),
            "fresh_process_nonce": build_summary.get("process_identity", {}).get("nonce")
            != args.process_nonce,
            "validate_outputs_absent": not any(
                path.exists()
                for path in (
                    RAW_LIVE_MEASUREMENT_PATH,
                    LIVE_AUDIT_PATH,
                    REPRESENTATION_GATE_PATH,
                    RRD_PATH,
                    AUTOMATED_SUMMARY_PATH,
                )
            ),
            "variant_robot_exists": VARIANT_ROBOT_USD.is_file(),
            "variant_physics_exists": VARIANT_PHYSICS_USD.is_file(),
        }
    )
    preflight["checks"] = checks
    preflight["pass"] = all(checks.values())
    _write_json(VALIDATE_PREFLIGHT_PATH, preflight)
    return bool(preflight["pass"])


def _run_validate(args: argparse.Namespace, simulation_app: Any) -> int:
    preflight = _json(VALIDATE_PREFLIGHT_PATH)
    if preflight.get("pass") is not True:
        raise RuntimeError("D344 validate preflight did not pass before Isaac launch")
    build_summary = _json(BUILD_SUMMARY_PATH)
    core_manifest = _json(CORE_BUILD_MANIFEST_PATH)

    before_inventories = _source_inventories()
    capture = _json(D340_CANDIDATES)
    args.robot_usd_path = VARIANT_ROBOT_USD
    d334_summary = _json(D334_SUMMARY)
    d336_summary = _json(D336_SUMMARY)
    inner = d333._make_runtime_env(args)
    try:
        inner.reset(seed=int(args.seed))
        counter_start = int(inner._sim_step_counter)
        stage_contract = d333._stage_contract(inner)
        sensor_contract, _filter_map = d333._sensor_contract(inner)
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
                "artifact": "D344_FRESH_LIVE_MEASUREMENT_EXCEPTION_STOP",
                "pass": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "per_body": {},
            }
        _write_json(RAW_LIVE_MEASUREMENT_PATH, raw_audit)
        witness_manifest = _witness_manifest()
        _write_json(WITNESS_MANIFEST_PATH, witness_manifest)

        try:
            cooked_by_body, live_audit = _reclassify_live_measurement(
                inner, core_manifest, raw_audit, capture
            )
        except Exception as error:
            cooked_by_body = {body: [] for body in d334.BODY_LABELS}
            live_audit = {
                "artifact": "D344_FRESH_LIVE_RECLASSIFICATION_EXCEPTION_STOP",
                "pass": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "per_body": {},
            }
        _write_json(LIVE_AUDIT_PATH, live_audit)

        scene_checks = {
            "stage_contract": bool(stage_contract["hard_contract_pass"]),
            "sensor_contract": bool(sensor_contract["hard_contract_pass"]),
            "retained_raw_source_contract": bool(raw_source_contract["pass"]),
            "witness_manifest_256": bool(witness_manifest["pass"]),
            "fresh_live_audit_128": bool(live_audit["pass"]),
        }
        raw_prerequisites = bool(
            scene_checks["stage_contract"]
            and scene_checks["sensor_contract"]
            and scene_checks["retained_raw_source_contract"]
        )
        if raw_prerequisites:
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
            candidate = d337._evaluate_candidate(
                inner,
                raw_shapes,
                OLD_RADIAL_NM,
                OLD_TANGENT_NM,
                Q5_OPEN_RAD,
                stage="d344_frozen_open_jaw_target",
            )
        else:
            controls = {
                "artifact": "D344_D337_CONTROLS_SKIPPED",
                "pass": False,
                "reason": "stage/sensor/raw prerequisite failed",
            }
            candidate = d339._fallback_candidate_without_raw(
                inner, "D344 retained raw-source contract failed"
            )
        _write_json(CONTROLS_PATH, controls)

        if all(scene_checks.values()) and controls.get("pass"):
            representation_gate, decision_raw, decision_live = d339._representation_gate(
                inner,
                raw_shapes,
                cooked_by_body,
                candidate,
                controls,
                live_audit,
            )
            representation_gate["artifact"] = "D344_ZERO_STEP_REPRESENTATION_GATE_V1"
            representation_gate["prephysics_support_eligible"] = bool(
                representation_gate["contract_pass"]
                and representation_gate["target_clear_and_faithful"]
            )
            representation_gate["physics_licensed"] = False
            representation_gate["physics_forbidden_in_d344"] = True
        else:
            decision_raw = (
                d336._exact_raw_metrics(inner, raw_shapes, "d344_raw_only_contract_stop")
                if raw_prerequisites
                else {"pose": "raw_not_certified", "queries": []}
            )
            decision_live = {
                "pose": "live_union_not_queried",
                "queries": [],
                "invalid_reason": (
                    "all 128 parts and prerequisite controls must pass before any union query"
                ),
            }
            representation_gate = {
                "artifact": "D344_ZERO_STEP_REPRESENTATION_GATE_V1",
                "checks": {**scene_checks, "d337_controls": bool(controls.get("pass"))},
                "per_body": {},
                "contract_pass": False,
                "target_clear_and_faithful": False,
                "prephysics_support_eligible": False,
                "physics_licensed": False,
                "physics_forbidden_in_d344": True,
                "structured_stop_reason": (
                    "fresh 128-part live contract or prerequisite failed; live union not queried"
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

        if not live_audit.get("pass"):
            scientific_verdict = VERDICT_LIVE_FAIL
            interpretation = "128개 실제 충돌 조각 검사 중 하나 이상이 실패했다"
        elif not representation_gate.get("contract_pass") or not representation_gate.get(
            "target_clear_and_faithful"
        ):
            scientific_verdict = VERDICT_TARGET_FAIL
            interpretation = "128개 조각은 통과했지만 고정 목표 자세의 거리·충실도 검사가 실패했다"
        else:
            scientific_verdict = VERDICT_PREPHYSICS_PENDING
            interpretation = (
                "새 실제 충돌 형상과 고정 목표 자세가 사전 물리 검사를 통과했다; "
                "Rerun 실제 화면 확인 뒤에도 물리 실험은 별도 사례다"
            )

        try:
            if decision_raw.get("queries") and decision_live.get("queries"):
                d339._write_representation_figure(
                    DECISION_PNG,
                    "D344 attempt3: frozen zero-step raw vs fresh live convex union",
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
                    title="D344 pre-physics contract STOP",
                    scene_checks=scene_checks,
                    candidate=candidate,
                )
            decision_png_ok = DECISION_PNG.is_file() and DECISION_PNG.stat().st_size > 0
        except Exception as error:
            decision_png_ok = False
            decision_png_error = f"{type(error).__name__}: {error}"
        else:
            decision_png_error = None
        try:
            marker_status = draw_frames(
                candidate["_frames"], prim_path="/World/D344FixedPointFrames"
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
        if frame_names != FRAME_NAMES:
            raise RuntimeError(f"D344 target frame schema drift: {frame_names} != {FRAME_NAMES}")
        rerun = _rerun_contract(
            candidate["_frames"],
            meshes,
            scalars,
            events,
            {
                "coordinate_frames": coordinate_frames,
                "case": "g0a_d344",
                "purpose": "attempt3 fresh live collision representation and frozen target audit",
                "git_head": _git_head(),
                "new_variables": NEW_VARIABLES,
                "scientific_authority": (
                    "callback arrays, direct authored arrays, JSON metrics, and exact hashes; "
                    "Rerun is observability only"
                ),
                "viewer_geometry_role": "Float32 one-way spatial observability copy",
                "q5_convention": "0=CLOSED; 1.5413rad=OPEN",
                "target": "radial=7mm tangent=11mm",
                "physics": "forbidden / 0 steps",
                "candidate_panel_role": (
                    "attempt3 authored geometry; changed 13 are byte-exact D340 candidates"
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
            "D344_RERUN_MACHINE_CONTRACT_PASS_MANUAL_INSPECTION_PENDING"
            if automated_artifact_pass
            else VERDICT_OBSERVABILITY_FAIL
        )
        summary = {
            "artifact": "D344_AUTOMATED_SUMMARY_V1",
            "automated_verdict": scientific_verdict,
            "observability_verdict": observability_verdict,
            "scientific_verdict_before_observability_gate": scientific_verdict,
            "automated_pass": bool(
                scientific_verdict == VERDICT_PREPHYSICS_PENDING
                and automated_artifact_pass
            ),
            "manual_visual_inspection_pending": bool(rerun["pass"]),
            "interpretation": interpretation,
            "new_variables": NEW_VARIABLES,
            "parameters_increased": [],
            "parameters_changed": [],
            "process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
            "build_process_identity": build_summary.get("process_identity"),
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "raw_source_contract": raw_source_contract,
            "scene_checks": scene_checks,
            "fresh_live_audit": live_audit,
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
            "witness_manifest": witness_manifest,
            "visualization": {
                "decision_png": _relative(DECISION_PNG) if DECISION_PNG.is_file() else None,
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
                "d339_attempt2_changed": False,
                "separate_settle_case_required": scientific_verdict
                == VERDICT_PREPHYSICS_PENDING,
            },
        }
        _write_json(AUTOMATED_SUMMARY_PATH, summary)
        _write_automated_report(summary)
        return 0 if summary["automated_pass"] else 2
    finally:
        inner.close()


def _parser_for_stage(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "validate"), required=True)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--urdf_path", type=Path, default=d333.DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=SEED)
    if stage == "validate":
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    stage_probe = argparse.ArgumentParser(add_help=False)
    stage_probe.add_argument("--stage", choices=("build", "validate"), required=True)
    stage_args, _unknown = stage_probe.parse_known_args()
    if "--print-rerun-contract" in sys.argv:
        print(
            json.dumps(
                {
                    "digest": _rrd_contract_digest(),
                    "entity_count": len(_expected_rrd_contract()[0]),
                },
                sort_keys=True,
            )
        )
        return 0
    parser = _parser_for_stage(stage_args.stage)
    args = parser.parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D344 output path is forward-only and fixed by preregistration")
    if int(args.seed) != SEED:
        raise RuntimeError("D344 seed is frozen at 33201")
    args.process_nonce = secrets.token_hex(16)
    if args.stage == "build":
        return _run_build(args)

    from isaaclab.app import AppLauncher

    if not _prepare_validate_preflight(args):
        return 2
    args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    def _close_app_at_exit() -> None:
        if simulation_app is not None:
            simulation_app.close()

    atexit.register(_close_app_at_exit)
    return _run_validate(args, simulation_app)


if __name__ == "__main__":
    raise SystemExit(main())
