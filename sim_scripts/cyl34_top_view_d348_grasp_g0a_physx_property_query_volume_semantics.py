#!/usr/bin/env python3
"""D348 read-only discriminator for PhysX property-query volume semantics.

The scientific input is the immutable D347 256-callback witness set.  D348
does not launch Isaac Sim, request a cook, edit an asset, or advance physics.
It compares two interpretations of each callback:

1. the historical vertex-only SciPy convex-hull envelope; and
2. the polygon topology explicitly returned by the PhysX cook callback.

Only the second interpretation can repair the D347 comparator, and only when
all 128 instance/prototype pairs pass the preregistered structural, translation,
property-volume, negative-control, and Rerun observability gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rerun as rr
import scipy
from scipy.spatial import ConvexHull


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.rerun_contract import (  # noqa: E402
    RERUN_CONTRACT_VERSION,
    validate_rerun_artifact,
)
from roarm_rl.viz_debug import log_rerun  # noqa: E402


CASE = "g0a_d348"
EXPECTED_HEAD = "d452921e04b7d5082c20d4edcfcc44bcefc7c34d"
NEW_VARIABLES = ["physx_property_query_volume_semantics"]
PROPERTY_REL_TOL = 0.05
VOLUME_DENOMINATOR_FLOOR_M3 = 1.0e-12
TRANSLATION_VOLUME_ABS_TOL_M3 = 1.0e-18
EXPECTED_PARTS_PER_BODY = 64
EXPECTED_PART_COUNT = 128
EXPECTED_CALLBACK_COUNT = 256

CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348"
OUT_DIR = CASE_DIR / "attempt2"
PREREG_PATH = OUT_DIR / "d348_preregistration.json"
PARAMETER_FREEZE_PATH = OUT_DIR / "d348_parameter_freeze_audit.json"
SOURCE_SEMANTICS_PATH = OUT_DIR / "d348_source_semantics.json"
HOME_CONTRACT_PATH = OUT_DIR / "d348_home_start_contract.json"
EVIDENCE_PATH = OUT_DIR / "d348_callback_topology_volume_evidence.json"
CONTROLS_PATH = OUT_DIR / "d348_matched_controls.json"
DECISION_PNG = OUT_DIR / "d348_volume_semantics_decision.png"
RRD_PATH = OUT_DIR / "d348_volume_semantics.rrd"
RBL_PATH = OUT_DIR / "d348_volume_semantics.rbl"
RERUN_SCREENSHOT_PATH = OUT_DIR / "d348_volume_semantics_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d348_rerun_validation.json"
AUTOMATED_SUMMARY_PATH = OUT_DIR / "d348_automated_summary.json"
AUTOMATED_REPORT_PATH = OUT_DIR / "d348_automated_report.md"
MANUAL_INSPECTION_PATH = OUT_DIR / "d348_manual_visual_inspection.json"
MANUAL_INSPECTION_MD_PATH = OUT_DIR / "d348_manual_visual_inspection.md"
COMPLETION_SUMMARY_PATH = OUT_DIR / "d348_completion_summary.json"
COMPLETION_REPORT_PATH = OUT_DIR / "d348_completion_report.md"

ATTEMPT1_PREPARE_HASHES: dict[Path, str] = {
    CASE_DIR / "d348_home_start_contract.json": (
        "bd4fcb39ffbe8bc5dfb9bc2797f3ba73b1c669f3bad79cee070ebdd40f8816df"
    ),
    CASE_DIR / "d348_parameter_freeze_audit.json": (
        "5d68abd4c0a029f1aec217de033de61226c0fdfa8621e8af291559a654e9e41b"
    ),
    CASE_DIR / "d348_preregistration.json": (
        "28e52d1b17442651dd2824777b5fa7d294df09bf542087630445ae3620b712ef"
    ),
    CASE_DIR / "d348_source_semantics.json": (
        "49fb43522c319f0df8a884234902de825c6728f4f93466e95bfa13ef6d92390b"
    ),
}

D347_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d347"
D347_COMPLETION = D347_DIR / "d347_completion_summary.json"
D347_AUDIT = D347_DIR / "d347_fresh_live_representation_audit.json"
D347_WITNESS_MANIFEST = D347_DIR / "d347_validate_cook_witness_manifest.json"
D347_WITNESS_DIR = D347_DIR / "d347_validate_cook_witnesses"
D347_RAW = D347_DIR / "d347_raw_live_measurement.json"
D347_PARAMETER = D347_DIR / "d347_parameter_freeze_audit.json"
D347_PREREG = D347_DIR / "d347_preregistration.json"
D347_ZERO_STEP = D347_DIR / "d347_zero_step_representation_gate.json"
D339_LIVE = (
    REPO / "claudedocs/runtime_logs/grasp_track/g0a_d339/d339_live_collider_audit.json"
)
D339_COOK_MANIFEST = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2/"
    "d339_cook_witness_manifest.json"
)

D339_HARNESS = REPO / "sim_scripts/cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair.py"
D340_HARNESS = REPO / "sim_scripts/cyl34_top_view_d340_grasp_g0a_fixed_point_live_authoring_repair.py"
D347_HARNESS = REPO / "sim_scripts/cyl34_top_view_d347_grasp_g0a_asset_validator_activation_order_repair.py"
D348_HARNESS = Path(__file__).resolve()
RERUN_CONTRACT_SOURCE = REPO / "roarm_rl/rerun_contract.py"
VIZ_DEBUG_SOURCE = REPO / "roarm_rl/viz_debug.py"
CUBE_ENV_SOURCE = REPO / "roarm_rl/roarm_cube_push_env.py"
STACK_ENV_SOURCE = REPO / "roarm_rl/roarm_stack_env.py"
D323_HARNESS = REPO / "sim_scripts/cube10cm_top_view_d323_grasp_g0a_frame_repair_probe.py"
D332_HARNESS = REPO / "sim_scripts/cyl34_top_view_d332_grasp_g0a_static_collision_discriminator.py"

PHYSX_PYI = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/omni/physx/"
    "bindings/_physx.pyi"
)
PHYSX_PROPERTY_TEST = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.physx.tests-107.3.26+107.3.3.cp311.u353/omni/physxtests/"
    "tests/PhysxPropertyQueryInterface.py"
)

INPUT_HASHES: dict[Path, str] = {
    D347_COMPLETION: "93ae7a6daea4d8ba9af6fa09d01deb6c72017925375195a53804b0d55286d65e",
    D347_AUDIT: "e652b16063cc0d7f9370df7e597ba6dcff9813f260c897b3b58b8b6c4d1b96ab",
    D347_WITNESS_MANIFEST: "a57bcd32b60c65ead4313a8914c8c2d61efd3fb7d620b993ba29af6967791438",
    D347_RAW: "2b2306862b4fc0cb22ffc6ed41c179f542b4a07f7014db17290c2003e99dfb9a",
    D347_PARAMETER: "417ca99f2c56d276f18ec455e1f3c499b0870796c57fe40379001a696022b669",
    D347_PREREG: "ca5f0c31e7974520f21ef51d765c8a78f78f276b720edc0d8019048f8fd50655",
    D347_ZERO_STEP: "ebc6fa2e6ba708721a7ad8d8786f3a8e83fcb81c1f530391bfc37c8f0c7748d9",
    D339_LIVE: "6148252b654a6250faf78a1ebcde4caa57870e800fa1d3c45b93c803fdf882cb",
    D339_COOK_MANIFEST: "7d0a82842af141c1e194ffcb5f9947777b8087c8fd56c72e13f684cf61481e81",
    PHYSX_PYI: "ff13abb83480dcc707ac2ad60062306aef7a33f885d32ed4c8ee6dfea2008e79",
    PHYSX_PROPERTY_TEST: "4c22d665fef5dce39bec2e1fb06c259c66d70e79764dac5c0ca3fe89fe07f108",
    D339_HARNESS: "fd307cb573699f8a08df1ab580789188774158877b8abf0a05cc4c60ef6562d6",
    D340_HARNESS: "1bdea659fadd7801b0f5749cca6286c7eb90c95a857ed516e09e34ceb12a023a",
    D347_HARNESS: "57e75de4807237121747ff00035402474aa18ab624134c2fa4698bb17cb18d0b",
    D332_HARNESS: "3ab551232b9c3e2a3886578e5f4baa4589d578567758a351203c2260a1428ad4",
    RERUN_CONTRACT_SOURCE: "90559c931bc753be97def463841d41426a2f1bd8e5ddd15a2a2ab08fb54a2e60",
    CUBE_ENV_SOURCE: "673c7cd415a2bcd1243ac100401977e8767aeb18880b3628e0f66947c0dcc057",
    STACK_ENV_SOURCE: "726a57f4be83276fda1bb5b3eaf07a17f56d5c16cdbc0441bc14fb5c794a697d",
    D323_HARNESS: "088208552bf1aa448c82756f13b598039632a603e67d115b8b9a4b489df18393",
}

METRIC_NAMES = (
    "property_volume_m3",
    "topology_instance_volume_m3",
    "topology_prototype_volume_m3",
    "topology_instance_relative_error",
    "topology_prototype_relative_error",
    "qhull_volume_m3",
    "qhull_relative_error",
    "raw_payload_equal",
    "closed_oriented",
    "part_pass",
)
EXPECTED_RERUN_COUNTS = {
    "coordinate_frame_count": 2,
    "mesh_count": 512,
    "scalar_row_count": 1280,
    "event_row_count": 132,
    "exact_non_system_entity_count": 2308,
}

VERDICT_NUMERIC_FAIL = "D348_PHYSX_PROPERTY_QUERY_VOLUME_SEMANTICS_FAIL_STOP"
VERDICT_MANUAL_PENDING = (
    "D348_PHYSX_PROPERTY_QUERY_TOPOLOGY_SEMANTICS_SUPPORTED_MANUAL_INSPECTION_PENDING"
)
VERDICT_OBSERVABILITY_FAIL = "D348_RERUN_OBSERVABILITY_INCOMPLETE_STOP"
VERDICT_COMPLETE = "D348_PHYSX_PROPERTY_QUERY_TOPOLOGY_SEMANTICS_SUPPORTED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _git_status_short_lines() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


def _attempt1_prepare_guard() -> dict[str, Any]:
    rows = []
    for path, expected in ATTEMPT1_PREPARE_HASHES.items():
        actual = _sha256(path) if path.is_file() else None
        rows.append(
            {
                "path": _relative(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "pass": actual == expected,
            }
        )
    failed_prereg = _json(CASE_DIR / "d348_preregistration.json")
    failed_parameter = _json(CASE_DIR / "d348_parameter_freeze_audit.json")
    checks = {
        "attempt1_files_4_of_4_hash_exact": len(rows) == 4 and all(row["pass"] for row in rows),
        "attempt1_prereg_failed": failed_prereg.get("pass") is False,
        "attempt1_only_parameter_freeze_failed": [
            key for key, value in failed_prereg.get("checks", {}).items() if not value
        ]
        == ["parameter_freeze_pass"],
        "attempt1_only_git_status_scope_failed": [
            key for key, value in failed_parameter.get("checks", {}).items() if not value
        ]
        == ["git_status_scope_only"],
        "attempt1_scientific_analysis_not_run": not any(
            (CASE_DIR / name).exists()
            for name in (
                "d348_callback_topology_volume_evidence.json",
                "d348_automated_summary.json",
                "d348_volume_semantics.rrd",
            )
        ),
    }
    return {
        "artifact": "D348_PREPARE_ATTEMPT1_PRESERVATION_GUARD_V1",
        "failure_class": "git_status_leading_space_parser_bug",
        "scientific_execution_consumed": False,
        "rows": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _input_guard() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path, expected in INPUT_HASHES.items():
        actual = _sha256(path) if path.is_file() else None
        rows.append(
            {
                "path": _relative(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "pass": actual == expected,
            }
        )
    manifest = _json(D347_WITNESS_MANIFEST)
    witness_rows = []
    for row in manifest.get("rows", []):
        path = D347_WITNESS_DIR / str(row.get("filename"))
        actual = _sha256(path) if path.is_file() else None
        witness_rows.append(
            {
                "filename": path.name,
                "expected_sha256": row.get("sha256"),
                "actual_sha256": actual,
                "pass": actual == row.get("sha256"),
            }
        )
    checks = {
        "registered_input_hashes_exact": len(rows) == len(INPUT_HASHES)
        and all(row["pass"] for row in rows),
        "manifest_historical_pass": manifest.get("pass") is True,
        "witness_count_256": len(witness_rows) == EXPECTED_CALLBACK_COUNT,
        "witness_hashes_256_of_256": len(witness_rows) == EXPECTED_CALLBACK_COUNT
        and all(row["pass"] for row in witness_rows),
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
    }
    return {
        "artifact": "D348_IMMUTABLE_INPUT_GUARD_V1",
        "registered_inputs": rows,
        "witness_rows": witness_rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _source_lines(path: Path, start: int, end: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1)]


def _source_semantics() -> dict[str, Any]:
    pyi_lines = _source_lines(PHYSX_PYI, 3414, 3418)
    test_lines = _source_lines(PHYSX_PROPERTY_TEST, 305, 319)
    zero = _json(D347_ZERO_STEP)
    checks = {
        "pyi_calls_value_collider_volume": any(
            "Volume of the collider" in line for line in pyi_lines
        ),
        "test_calls_volume_global_quantity": any(
            "global quantity" in line for line in test_lines
        ),
        "test_calls_aabb_local_quantity": any("local quantity" in line for line in test_lines),
        "test_volume_assertion_is_one_sided": any(
            "collider_info.volume - expected_volume <" in line for line in test_lines
        ),
        "d347_runtime_meters_per_unit_one": zero.get("checks", {}).get(
            "runtime_meters_per_unit_one"
        )
        is True,
    }
    return {
        "artifact": "D348_PHYSX_LOCAL_SOURCE_SEMANTICS_V1",
        "physx_version_scope": "107.3.26+107.3.3 / Isaac Sim 5.1 local installation",
        "binding_declaration": {
            "path": str(PHYSX_PYI),
            "sha256": _sha256(PHYSX_PYI),
            "lines": pyi_lines,
        },
        "bundled_test": {
            "path": str(PHYSX_PROPERTY_TEST),
            "sha256": _sha256(PHYSX_PROPERTY_TEST),
            "lines": test_lines,
            "caveat": (
                "The bundled test documents scale semantics, but its volume assertion is "
                "one-sided. D348 therefore treats the 256 frozen measurements, not this test, "
                "as the decisive evidence."
            ),
        },
        "registered_interpretation": (
            "For identity local transforms and stage meters-per-unit 1, compare the query "
            "volume to the closed polygon topology returned for that collider."
        ),
        "runtime_unit_evidence": {
            "path": _relative(D347_ZERO_STEP),
            "sha256": _sha256(D347_ZERO_STEP),
            "runtime_meters_per_unit_one": zero.get("checks", {}).get(
                "runtime_meters_per_unit_one"
            ),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _home_contract() -> dict[str, Any]:
    zero = _json(D347_ZERO_STEP)
    refs = [
        {
            "role": "nominal_home_definition",
            "path": _relative(D323_HARNESS),
            "sha256": _sha256(D323_HARNESS),
            "lines": _source_lines(D323_HARNESS, 39, 42),
        },
        {
            "role": "articulation_authored_initial_state",
            "path": _relative(STACK_ENV_SOURCE),
            "sha256": _sha256(STACK_ENV_SOURCE),
            "lines": _source_lines(STACK_ENV_SOURCE, 161, 169),
        },
        {
            "role": "runtime_reset_home_jitter_and_closed_q5",
            "path": _relative(CUBE_ENV_SOURCE),
            "sha256": _sha256(CUBE_ENV_SOURCE),
            "lines": _source_lines(CUBE_ENV_SOURCE, 1632, 1645),
        },
        {
            "role": "d347_reset_call",
            "path": _relative(D347_HARNESS),
            "sha256": _sha256(D347_HARNESS),
            "lines": _source_lines(D347_HARNESS, 1903, 1907),
        },
        {
            "role": "fallback_exact_state_write_without_step",
            "path": _relative(D339_HARNESS),
            "sha256": _sha256(D339_HARNESS),
            "lines": _source_lines(D339_HARNESS, 2408, 2422),
        },
        {
            "role": "exact_state_write_forward_and_zero_dt_update",
            "path": _relative(D332_HARNESS),
            "sha256": _sha256(D332_HARNESS),
            "lines": _source_lines(D332_HARNESS, 578, 602),
        },
    ]
    checks = {
        "nominal_home_exact_deg": "HOME_DEG = np.array([0.0, 0.0, 90.0, 0.0, 0.0, 0.0]"
        in D323_HARNESS.read_text(encoding="utf-8"),
        "authored_initial_state_exact_home": all(
            token in STACK_ENV_SOURCE.read_text(encoding="utf-8")
            for token in (
                '"base_link_to_link1": 0.0',
                '"link2_to_link3": math.pi / 2',
                '"link5_to_gripper_link": 0.0',
            )
        ),
        "runtime_reset_adds_plus_minus_0p02_rad": "sample_uniform(-0.02, 0.02"
        in CUBE_ENV_SOURCE.read_text(encoding="utf-8"),
        "runtime_reset_forces_q5_zero": "joint_pos[:, self.gripper_joint_idx] = 0.0"
        in CUBE_ENV_SOURCE.read_text(encoding="utf-8"),
        "d347_controlled_physics_steps_zero": zero.get("controlled_physics_steps") == 0,
        "d347_sim_counter_unchanged": zero.get("global_sim_counter", {}).get("unchanged")
        is True,
        "d348_does_not_change_reset": True,
    }
    return {
        "artifact": "D348_HOME_START_CONTRACT_V1",
        "nominal_home_deg": [0.0, 0.0, 90.0, 0.0, 0.0, 0.0],
        "d347_callback_measurement_start": {
            "classification": "HOME-near deterministic jitter, q5=0 closed",
            "seed": 33201,
            "per_joint_reset_jitter_rad": [-0.02, 0.02],
            "per_joint_reset_jitter_deg_approx": [-1.1459155903, 1.1459155903],
            "q5_after_reset_rad": 0.0,
            "exact_absolute_joint_vector_persisted": False,
        },
        "d347_post_failure_observability_pose": {
            "q5_rad": 1.5413,
            "method": "exact state write plus sim.forward and zero-dt update; controlled physics step 0",
            "physical_home_to_target_motion": False,
        },
        "bounded_answer_ko": (
            "D347 실측은 정확한 HOME이 아니라 HOME 근방의 닫힌 그리퍼 자세에서 "
            "충돌 API를 검사했다. 목표 열린 자세는 물리 step 없이 관찰용으로만 "
            "기록했다. D348은 PhysX를 다시 실행하지 않고 그 D347 실측을 재판독한다."
        ),
        "source_references": refs,
        "d347_zero_step_source": {
            "path": _relative(D347_ZERO_STEP),
            "sha256": _sha256(D347_ZERO_STEP),
            "controlled_physics_steps": zero.get("controlled_physics_steps"),
            "global_sim_counter": zero.get("global_sim_counter"),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _expected_rrd_contract() -> tuple[list[str], dict[str, list[str]]]:
    entities = {"metadata/run", "events/d348"}
    components: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "events/d348": ["TextLog:level", "TextLog:text"],
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
        for idx in range(EXPECTED_PARTS_PER_BODY):
            name = f"part_{idx:03d}"
            for variant in ("source", "instance", "prototype", "candidate"):
                mesh_path = f"cook/{variant}/{body}/parts/{name}"
                metadata_path = f"metadata/meshes/{mesh_path.replace('/', '__')}"
                entities.update({mesh_path, metadata_path})
                components[mesh_path] = mesh_components
                components[metadata_path] = ["TextDocument:text"]
            for metric in METRIC_NAMES:
                metric_path = f"metrics/{body}/{name}/{metric}"
                entities.add(metric_path)
                components[metric_path] = ["Scalars:scalars"]
    return sorted(entities), components


def _rrd_contract_digest() -> str:
    entities, components = _expected_rrd_contract()
    payload = {
        "exact_non_system_entity_paths": entities,
        "exact_timeline_names": ["blueprint", "event_idx", "log_time", "part_idx"],
        "required_components_by_path": components,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_prepare(_args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise FileExistsError(f"D348 output already exists: {OUT_DIR}")
    input_guard = _input_guard()
    source_semantics = _source_semantics()
    home_contract = _home_contract()
    attempt1_guard = _attempt1_prepare_guard()
    d347_audit = _json(D347_AUDIT)
    d347_zero_step = _json(D347_ZERO_STEP)
    package_versions = {
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "rerun": rr.__version__,
        "scipy": scipy.__version__,
    }
    status_before = _git_status_short_lines()
    status_rows = [
        {"status_code": line[:2], "path": line[3:] if len(line) >= 4 else ""}
        for line in status_before
    ]
    allowed_exact_paths = {
        "roarm_rl/viz_debug.py",
        "claudedocs/session_20260714_grasp_g0a_d348_physx_property_query_volume_semantics.md",
        "sim_scripts/cyl34_top_view_d348_grasp_g0a_physx_property_query_volume_semantics.py",
        "claudedocs/runtime_logs/grasp_track/g0a_d348/",
    }
    status_scope_only = all(
        row["status_code"] in {" M", "M ", "??"}
        and row["path"] in allowed_exact_paths
        for row in status_rows
    )
    parameter_checks = {
        "one_new_variable": NEW_VARIABLES == ["physx_property_query_volume_semantics"],
        "no_new_physical_variables": True,
        "property_tolerance_still_0p05": d347_audit.get("classification_contract", {}).get(
            "property_volume_relative_tolerance"
        )
        == PROPERTY_REL_TOL,
        "part_gate_still_128_of_128": EXPECTED_PART_COUNT == 128,
        "runtime_meters_per_unit_one": d347_zero_step.get("checks", {}).get(
            "runtime_meters_per_unit_one"
        )
        is True,
        "numpy_pin_exact": package_versions["numpy"] == "1.26.0",
        "psutil_pin_exact": package_versions["psutil"] == "5.9.8",
        "rerun_pin_exact": package_versions["rerun"] == RERUN_CONTRACT_VERSION,
        "scipy_qhull_reader_pin_exact": package_versions["scipy"] == "1.15.3",
        "input_guard_pass": input_guard["pass"],
        "source_semantics_pass": source_semantics["pass"],
        "home_contract_pass": home_contract["pass"],
        "attempt1_failure_preserved_and_science_not_run": attempt1_guard["pass"],
        "git_status_scope_only": status_scope_only,
    }
    parameter_freeze = {
        "artifact": "D348_PARAMETER_FREEZE_AUDIT_V1",
        "case": CASE,
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "frozen": {
            "collision_assets": "D344 attempt3 immutable",
            "decomposition_settings": "unchanged",
            "part_count": EXPECTED_PART_COUNT,
            "target_family": {"radial_mm": 7.0, "tangent_mm": 11.0, "q5_rad": 1.5413},
            "property_relative_tolerance": PROPERTY_REL_TOL,
            "physics_steps": 0,
            "asset_writes": 0,
            "cook_requests": 0,
        },
        "measurement_parameters": {
            "volume_denominator_floor_m3": VOLUME_DENOMINATOR_FLOOR_M3,
            "translation_volume_abs_tolerance_m3": TRANSLATION_VOLUME_ABS_TOL_M3,
        },
        "package_versions": package_versions,
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_before_prepare": status_before,
        "git_status_parsed": status_rows,
        "prepare_attempt1_guard": attempt1_guard,
        "reactive_repair": {
            "attempt": 2,
            "reason": (
                "attempt1 used _git(...).strip(), which removed the leading space from the "
                "first porcelain status code before scope classification"
            ),
            "scientific_execution_consumed_before_attempt2": False,
            "attempt1_overwritten_or_deleted": False,
        },
        "checks": parameter_checks,
        "pass": all(parameter_checks.values()),
    }
    OUT_DIR.mkdir(parents=True)
    _write_json(PARAMETER_FREEZE_PATH, parameter_freeze)
    _write_json(SOURCE_SEMANTICS_PATH, source_semantics)
    _write_json(HOME_CONTRACT_PATH, home_contract)
    prereg_checks = {
        "parameter_freeze_pass": parameter_freeze["pass"],
        "input_guard_pass": input_guard["pass"],
        "attempt1_preservation_guard_pass": attempt1_guard["pass"],
        "exact_rrd_entity_count_2308": len(_expected_rrd_contract()[0]) == 2308,
        "registered_mesh_count_512": EXPECTED_RERUN_COUNTS["mesh_count"] == 512,
        "registered_scalar_count_1280": EXPECTED_RERUN_COUNTS["scalar_row_count"] == 1280,
        "registered_event_count_132": EXPECTED_RERUN_COUNTS["event_row_count"] == 132,
    }
    prereg = {
        "artifact": "D348_PREREGISTRATION_V1",
        "case": CASE,
        "git_head": _git("rev-parse", "HEAD"),
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "scientific_question": (
            "Does the PhysX property-query volume match the callback's explicit polygon "
            "topology, rather than a new convex hull inferred from callback vertices?"
        ),
        "registered_stages": ["prepare", "analyze", "manual inspection", "finalize"],
        "single_scientific_execution": True,
        "input_guard": input_guard,
        "prepare_attempt1_guard": attempt1_guard,
        "prepare_attempt": 2,
        "attempt1_overwritten_or_deleted": False,
        "parameter_freeze_path": _relative(PARAMETER_FREEZE_PATH),
        "parameter_freeze_sha256": _sha256(PARAMETER_FREEZE_PATH),
        "source_semantics_path": _relative(SOURCE_SEMANTICS_PATH),
        "source_semantics_sha256": _sha256(SOURCE_SEMANTICS_PATH),
        "home_contract_path": _relative(HOME_CONTRACT_PATH),
        "home_contract_sha256": _sha256(HOME_CONTRACT_PATH),
        "harness_sha256": _sha256(D348_HARNESS),
        "viz_debug_sha256": _sha256(VIZ_DEBUG_SOURCE),
        "comparators": {
            "property_relative_tolerance": PROPERTY_REL_TOL,
            "volume_denominator_floor_m3": VOLUME_DENOMINATOR_FLOOR_M3,
            "translation_volume_abs_tolerance_m3": TRANSLATION_VOLUME_ABS_TOL_M3,
            "required_topology_property_pass_count": 256,
            "required_pair_pass_count": 128,
            "negative_control_must_fail_closed_oriented_gate": True,
        },
        "matched_controls": {
            "all_historical_d347_passes": 127,
            "same_7_vertex_10_triangle_group": [
                "gripper_link/part_004",
                "gripper_link/part_058",
                "link5/part_004",
                "link5/part_008",
                "link5/part_009",
                "link5/part_023",
                "link5/part_024",
            ],
            "nearest_qhull_volume": "link5/part_000",
            "nearest_minimum_extent": "link5/part_039",
            "negative_control": "in-memory removal of first part_045 polygon",
        },
        "rerun_contract": {
            "counts": EXPECTED_RERUN_COUNTS,
            "exact_timelines": ["blueprint", "event_idx", "log_time", "part_idx"],
            "digest": _rrd_contract_digest(),
            "blueprint_mode": "volume_semantics",
            "manual_actual_screenshot_inspection_required": True,
        },
        "stop_rules": {
            "numeric_failure": VERDICT_NUMERIC_FAIL,
            "observability_failure": VERDICT_OBSERVABILITY_FAIL,
            "success": VERDICT_COMPLETE,
            "target_query_forbidden": True,
            "g0a_pass_remains_false": True,
        },
        "checks": prereg_checks,
        "pass": all(prereg_checks.values()),
    }
    _write_json(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"]}, sort_keys=True))
    return 0 if prereg["pass"] else 1


def _callback_convex(path: Path) -> dict[str, Any]:
    witness = _json(path)
    events = witness.get("events", [])
    if witness.get("callback_count") != 1 or len(events) != 1:
        raise ValueError(f"{path}: expected exactly one callback event")
    convexes = events[0].get("convexes", [])
    if events[0].get("convex_count") != 1 or len(convexes) != 1:
        raise ValueError(f"{path}: expected exactly one convex")
    convex = convexes[0]
    vertices = np.asarray(convex.get("vertices"), dtype=np.float64)
    indices = np.asarray(convex.get("indices"), dtype=np.int64)
    polygons = list(convex.get("polygons", []))
    if vertices.shape != (int(convex.get("vertex_count", -1)), 3):
        raise ValueError(f"{path}: vertex count mismatch")
    if indices.shape != (int(convex.get("index_count", -1)),):
        raise ValueError(f"{path}: index count mismatch")
    if len(polygons) != int(convex.get("polygon_count", -1)):
        raise ValueError(f"{path}: polygon count mismatch")
    if not np.isfinite(vertices).all() or indices.min() < 0 or indices.max() >= len(vertices):
        raise ValueError(f"{path}: invalid vertex/index payload")
    return convex


def _payload_digest(convex: dict[str, Any]) -> str:
    payload = {
        "vertices": convex["vertices"],
        "indices": convex["indices"],
        "polygons": convex["polygons"],
        "vertex_count": convex["vertex_count"],
        "index_count": convex["index_count"],
        "polygon_count": convex["polygon_count"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _triangulate(convex: dict[str, Any], *, drop_polygon: int | None = None) -> np.ndarray:
    indices = np.asarray(convex["indices"], dtype=np.int64)
    triangles: list[list[int]] = []
    covered = 0
    for polygon_idx, polygon in enumerate(convex["polygons"]):
        base = int(polygon["index_base"])
        count = int(polygon["num_vertices"])
        if base != covered or count < 3 or base + count > len(indices):
            raise ValueError("polygon spans do not exactly and validly partition index buffer")
        face = indices[base : base + count]
        covered += count
        if polygon_idx == drop_polygon:
            continue
        for idx in range(1, count - 1):
            triangles.append([int(face[0]), int(face[idx]), int(face[idx + 1])])
    if covered != len(indices) or not triangles:
        raise ValueError("polygon spans do not cover index buffer or produced no triangles")
    return np.asarray(triangles, dtype=np.int64)


def _closed_oriented(triangles: np.ndarray) -> dict[str, Any]:
    directed: Counter[tuple[int, int]] = Counter()
    undirected: Counter[tuple[int, int]] = Counter()
    for tri in triangles.tolist():
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            directed[(int(a), int(b))] += 1
            undirected[tuple(sorted((int(a), int(b))))] += 1
    count_two = all(count == 2 for count in undirected.values())
    opposite = all(
        directed[(edge[0], edge[1])] == 1 and directed[(edge[1], edge[0])] == 1
        for edge in undirected
    )
    return {
        "undirected_edge_count": len(undirected),
        "all_undirected_edges_exactly_twice": count_two,
        "all_edge_pairs_opposite_direction": opposite,
        "pass": count_two and opposite,
    }


def _signed_volume(vertices: np.ndarray, triangles: np.ndarray, origin: np.ndarray) -> float:
    shifted = vertices - np.asarray(origin, dtype=np.float64).reshape(1, 3)
    a = shifted[triangles[:, 0]]
    b = shifted[triangles[:, 1]]
    c = shifted[triangles[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _plane_residual(convex: dict[str, Any]) -> float:
    vertices = np.asarray(convex["vertices"], dtype=np.float64)
    indices = np.asarray(convex["indices"], dtype=np.int64)
    residual = 0.0
    for polygon in convex["polygons"]:
        base = int(polygon["index_base"])
        count = int(polygon["num_vertices"])
        plane = np.asarray(polygon["plane"], dtype=np.float64)
        face_vertices = vertices[indices[base : base + count]]
        residual = max(residual, float(np.max(np.abs(face_vertices @ plane[:3] + plane[3]))))
    return residual


def _property_volumes() -> dict[tuple[str, str], dict[str, Any]]:
    raw = _json(D347_RAW)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for body in ("link5", "gripper_link"):
        for row in raw["per_body"][body]["property_query"]["colliders"]:
            name = str(row["path"]).rsplit("/", 1)[-1]
            if "/d338_convex_parts/" not in str(row["path"]):
                continue
            result[(body, name)] = dict(row)
    return result


def _historical_qhull_volumes() -> dict[tuple[str, str], float]:
    audit = _json(D347_AUDIT)
    result: dict[tuple[str, str], float] = {}
    for body in ("link5", "gripper_link"):
        for row in audit["per_body"][body]["part_checks"]:
            result[(body, row["name"])] = float(
                row["channel_consensus"]["consensus"]["volume_m3"]
            )
    return result


def _analyze_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    properties = _property_volumes()
    historical_qhull = _historical_qhull_volumes()
    rows: list[dict[str, Any]] = []
    for body in ("link5", "gripper_link"):
        for idx in range(EXPECTED_PARTS_PER_BODY):
            name = f"part_{idx:03d}"
            key = (body, name)
            channels: dict[str, dict[str, Any]] = {}
            for channel in ("instance", "prototype"):
                witness_path = D347_WITNESS_DIR / f"{body}_{name}_{channel}.json"
                convex = _callback_convex(witness_path)
                vertices = np.asarray(convex["vertices"], dtype=np.float64)
                triangles = _triangulate(convex)
                closure = _closed_oriented(triangles)
                signed_origin = _signed_volume(vertices, triangles, np.zeros(3))
                signed_center = _signed_volume(vertices, triangles, vertices.mean(axis=0))
                qhull = ConvexHull(vertices)
                channels[channel] = {
                    "witness_path": _relative(witness_path),
                    "witness_sha256": _sha256(witness_path),
                    "payload_sha256": _payload_digest(convex),
                    "vertex_count": int(len(vertices)),
                    "polygon_count": int(len(convex["polygons"])),
                    "triangle_count": int(len(triangles)),
                    "vertices_m": vertices.tolist(),
                    "topology_triangles": triangles.tolist(),
                    "qhull_triangles": np.asarray(qhull.simplices, dtype=np.int64).tolist(),
                    "closure": closure,
                    "signed_volume_origin_m3": signed_origin,
                    "signed_volume_centroid_shift_m3": signed_center,
                    "volume_origin_m3": abs(signed_origin),
                    "volume_centroid_shift_m3": abs(signed_center),
                    "translation_abs_delta_m3": abs(signed_origin - signed_center),
                    "qhull_volume_m3": float(qhull.volume),
                    "max_polygon_plane_residual_m": _plane_residual(convex),
                }
            prop = properties[key]
            property_volume = float(prop["volume_m3"])
            instance = channels["instance"]
            prototype = channels["prototype"]
            denom = max(abs(property_volume), VOLUME_DENOMINATOR_FLOOR_M3)
            instance_rel = abs(instance["volume_origin_m3"] - property_volume) / denom
            prototype_rel = abs(prototype["volume_origin_m3"] - property_volume) / denom
            qhull_rel = abs(instance["qhull_volume_m3"] - property_volume) / denom
            raw_equal = instance["payload_sha256"] == prototype["payload_sha256"]
            closure_pass = instance["closure"]["pass"] and prototype["closure"]["pass"]
            translation_pass = (
                instance["translation_abs_delta_m3"] <= TRANSLATION_VOLUME_ABS_TOL_M3
                and prototype["translation_abs_delta_m3"] <= TRANSLATION_VOLUME_ABS_TOL_M3
            )
            topology_property_pass = (
                instance_rel <= PROPERTY_REL_TOL and prototype_rel <= PROPERTY_REL_TOL
            )
            qhull_reproduces_d347 = math.isclose(
                instance["qhull_volume_m3"],
                historical_qhull[key],
                rel_tol=1.0e-12,
                abs_tol=1.0e-18,
            )
            checks = {
                "property_row_valid": prop.get("result") == "PhysxPropertyQueryResult.VALID",
                "property_transform_identity": prop.get("local_pos_m") == [0.0, 0.0, 0.0]
                and prop.get("local_rot_xyzw") == [0.0, 0.0, 0.0, 1.0],
                "raw_instance_prototype_payload_exact": raw_equal,
                "both_topologies_closed_and_oriented": closure_pass,
                "both_signed_volumes_nonzero_same_sign": instance["signed_volume_origin_m3"]
                * prototype["signed_volume_origin_m3"]
                > 0.0,
                "translation_independent_within_1e_18_m3": translation_pass,
                "topology_vs_property_both_le_5pct": topology_property_pass,
                "qhull_reproduces_d347_registered_value": qhull_reproduces_d347,
            }
            rows.append(
                {
                    "body": body,
                    "name": name,
                    "global_part_idx": len(rows),
                    "property_query": prop,
                    "property_volume_m3": property_volume,
                    "historical_d347_qhull_volume_m3": historical_qhull[key],
                    "instance": instance,
                    "prototype": prototype,
                    "topology_instance_relative_error": instance_rel,
                    "topology_prototype_relative_error": prototype_rel,
                    "qhull_relative_error": qhull_rel,
                    "checks": checks,
                    "pass": all(checks.values()),
                }
            )
    part045 = next(row for row in rows if row["body"] == "link5" and row["name"] == "part_045")
    negative_triangles = _triangulate(
        _callback_convex(D347_WITNESS_DIR / "link5_part_045_instance.json"),
        drop_polygon=0,
    )
    negative = _closed_oriented(negative_triangles)
    negative["method"] = "removed polygon 0 in memory; immutable witness not modified"
    negative["expected_pass"] = False
    negative["control_pass"] = negative["pass"] is False
    d347_audit = _json(D347_AUDIT)
    historical_status: dict[str, bool] = {}
    historical_minimum_extent_m: dict[str, float] = {}
    for body in ("link5", "gripper_link"):
        for old_row in d347_audit["per_body"][body]["part_checks"]:
            key = f"{body}/{old_row['name']}"
            historical_status[key] = bool(old_row["pass"])
            vertices = np.asarray(
                old_row["channel_consensus"]["consensus"]["vertices_m"],
                dtype=np.float64,
            )
            historical_minimum_extent_m[key] = float(
                np.min(vertices.max(axis=0) - vertices.min(axis=0))
            )
    historical_passes = sorted(key for key, passed in historical_status.items() if passed)
    historical_failures = sorted(key for key, passed in historical_status.items() if not passed)
    target_key = "link5/part_045"
    target_qhull = part045["historical_d347_qhull_volume_m3"]
    target_minimum_extent = historical_minimum_extent_m[target_key]
    computed_nearest_qhull = min(
        (
            abs(row["historical_d347_qhull_volume_m3"] - target_qhull),
            f"{row['body']}/{row['name']}",
        )
        for row in rows
        if f"{row['body']}/{row['name']}" != target_key
    )[1]
    computed_nearest_minimum_extent = min(
        (abs(value - target_minimum_extent), key)
        for key, value in historical_minimum_extent_m.items()
        if key != target_key
    )[1]
    controls = {
        "historical_d347_status": historical_status,
        "all_historical_d347_passes": historical_passes,
        "historical_d347_failures": historical_failures,
        "same_7_vertex_10_triangle_group": [
            "gripper_link/part_004",
            "gripper_link/part_058",
            "link5/part_004",
            "link5/part_008",
            "link5/part_009",
            "link5/part_023",
            "link5/part_024",
        ],
        "registered_nearest_qhull_volume": "link5/part_000",
        "computed_nearest_qhull_volume": computed_nearest_qhull,
        "registered_nearest_minimum_extent": "link5/part_039",
        "computed_nearest_minimum_extent": computed_nearest_minimum_extent,
        "historical_minimum_extent_m": historical_minimum_extent_m,
        "negative_dropped_face": negative,
    }
    return rows, controls


def _row_lookup(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    body, name = key.split("/")
    return next(row for row in rows if row["body"] == body and row["name"] == name)


def _matched_controls(rows: list[dict[str, Any]], controls: dict[str, Any]) -> dict[str, Any]:
    same_group = [_row_lookup(rows, key) for key in controls["same_7_vertex_10_triangle_group"]]
    selected_keys = [
        "link5/part_045",
        controls["registered_nearest_qhull_volume"],
        controls["registered_nearest_minimum_extent"],
        *controls["same_7_vertex_10_triangle_group"],
    ]
    selected = []
    for key in dict.fromkeys(selected_keys):
        row = _row_lookup(rows, key)
        selected.append(
            {
                "key": key,
                "vertex_count": row["instance"]["vertex_count"],
                "triangle_count": row["instance"]["triangle_count"],
                "property_volume_m3": row["property_volume_m3"],
                "topology_volume_m3": row["instance"]["volume_origin_m3"],
                "qhull_volume_m3": row["instance"]["qhull_volume_m3"],
                "topology_relative_error": row["topology_instance_relative_error"],
                "qhull_relative_error": row["qhull_relative_error"],
                "pass": row["pass"],
            }
        )
    checks = {
        "historical_control_count_127": len(controls["all_historical_d347_passes"]) == 127,
        "historical_d347_fail_exactly_part045": controls["historical_d347_failures"]
        == ["link5/part_045"],
        "historical_d347_status_covers_128": len(controls["historical_d347_status"]) == 128,
        "historical_controls_topology_pass_127_of_127": all(
            _row_lookup(rows, key)["pass"] for key in controls["all_historical_d347_passes"]
        ),
        "registered_nearest_qhull_is_computed_minimum": controls[
            "registered_nearest_qhull_volume"
        ]
        == controls["computed_nearest_qhull_volume"],
        "registered_nearest_extent_is_computed_minimum": controls[
            "registered_nearest_minimum_extent"
        ]
        == controls["computed_nearest_minimum_extent"],
        "same_topology_group_exact_7": len(same_group) == 7,
        "same_topology_group_7_vertices_10_triangles": all(
            row["instance"]["vertex_count"] == 7
            and row["instance"]["triangle_count"] == 10
            for row in same_group
        ),
        "negative_control_rejected": controls["negative_dropped_face"]["control_pass"],
    }
    return {
        "artifact": "D348_MATCHED_CONTROLS_V1",
        "selection_registered_before_analysis": True,
        "historical_d347_failures": controls["historical_d347_failures"],
        "nearest_control_recalculation": {
            "registered_nearest_qhull_volume": controls["registered_nearest_qhull_volume"],
            "computed_nearest_qhull_volume": controls["computed_nearest_qhull_volume"],
            "registered_nearest_minimum_extent": controls[
                "registered_nearest_minimum_extent"
            ],
            "computed_nearest_minimum_extent": controls[
                "computed_nearest_minimum_extent"
            ],
        },
        "selected_rows": selected,
        "negative_control": controls["negative_dropped_face"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _evidence(rows: list[dict[str, Any]], controls: dict[str, Any]) -> dict[str, Any]:
    part045 = _row_lookup(rows, "link5/part_045")
    topology_errors = [
        value
        for row in rows
        for value in (
            row["topology_instance_relative_error"],
            row["topology_prototype_relative_error"],
        )
    ]
    translation_deltas = [
        channel["translation_abs_delta_m3"]
        for row in rows
        for channel in (row["instance"], row["prototype"])
    ]
    checks = {
        "parts_128": len(rows) == EXPECTED_PART_COUNT,
        "callback_channels_256": len(rows) * 2 == EXPECTED_CALLBACK_COUNT,
        "raw_pair_equality_128_of_128": all(
            row["checks"]["raw_instance_prototype_payload_exact"] for row in rows
        ),
        "closed_oriented_256_of_256": all(
            channel["closure"]["pass"]
            for row in rows
            for channel in (row["instance"], row["prototype"])
        ),
        "translation_invariant_256_of_256": max(translation_deltas)
        <= TRANSLATION_VOLUME_ABS_TOL_M3,
        "topology_property_256_of_256_le_5pct": max(topology_errors) <= PROPERTY_REL_TOL,
        "all_part_checks_128_of_128": all(row["pass"] for row in rows),
        "part045_old_qhull_still_gt_5pct": part045["qhull_relative_error"] > PROPERTY_REL_TOL,
        "part045_topology_now_le_5pct": part045["topology_instance_relative_error"]
        <= PROPERTY_REL_TOL,
        "negative_dropped_face_rejected": controls["negative_dropped_face"]["control_pass"],
    }
    return {
        "artifact": "D348_CALLBACK_TOPOLOGY_VOLUME_EVIDENCE_V1",
        "case": CASE,
        "scientific_scope": (
            "Immutable D347 callbacks and property-query scalars; no new cook, asset write, "
            "target query, or physics step."
        ),
        "comparators": {
            "property_relative_tolerance": PROPERTY_REL_TOL,
            "volume_denominator_floor_m3": VOLUME_DENOMINATOR_FLOOR_M3,
            "translation_volume_abs_tolerance_m3": TRANSLATION_VOLUME_ABS_TOL_M3,
        },
        "aggregate": {
            "part_count": len(rows),
            "callback_channel_count": len(rows) * 2,
            "topology_property_pass_count": sum(
                int(value <= PROPERTY_REL_TOL) for value in topology_errors
            ),
            "max_topology_property_relative_error": max(topology_errors),
            "median_topology_property_relative_error": float(np.median(topology_errors)),
            "max_translation_volume_abs_delta_m3": max(translation_deltas),
            "raw_pair_equal_count": sum(
                int(row["checks"]["raw_instance_prototype_payload_exact"]) for row in rows
            ),
            "closed_oriented_channel_count": sum(
                int(channel["closure"]["pass"])
                for row in rows
                for channel in (row["instance"], row["prototype"])
            ),
        },
        "part045_root_cause": {
            "property_volume_m3": part045["property_volume_m3"],
            "callback_topology_volume_m3": part045["instance"]["volume_origin_m3"],
            "callback_topology_relative_error": part045["topology_instance_relative_error"],
            "vertex_only_qhull_volume_m3": part045["instance"]["qhull_volume_m3"],
            "vertex_only_qhull_relative_error": part045["qhull_relative_error"],
            "max_polygon_plane_residual_m": part045["instance"][
                "max_polygon_plane_residual_m"
            ],
            "explanation": (
                "The callback polygon is not exactly coplanar at Float32 precision. Re-Qhulling "
                "the vertex cloud replaces the reported faces with a different envelope."
            ),
        },
        "negative_control": controls["negative_dropped_face"],
        "rows": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _decision_png(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if DECISION_PNG.exists():
        raise FileExistsError(f"refusing to overwrite {DECISION_PNG}")
    x = np.arange(len(rows))
    topology = np.asarray([row["topology_instance_relative_error"] for row in rows]) * 100.0
    qhull = np.asarray([row["qhull_relative_error"] for row in rows]) * 100.0
    part045 = _row_lookup(rows, "link5/part_045")
    vertices = np.asarray(part045["instance"]["vertices_m"], dtype=np.float64)
    topology_tri = np.asarray(part045["instance"]["topology_triangles"], dtype=np.int64)
    qhull_tri = np.asarray(part045["instance"]["qhull_triangles"], dtype=np.int64)

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(grid[0, :])
    ax0.semilogy(x, np.maximum(topology, 1.0e-12), label="callback face topology", lw=2)
    ax0.semilogy(x, np.maximum(qhull, 1.0e-12), label="vertex-only Qhull", lw=1.3)
    ax0.axhline(PROPERTY_REL_TOL * 100.0, color="red", ls="--", label="frozen 5% gate")
    ax0.axvline(45, color="black", alpha=0.3)
    ax0.set_xlabel("part index: link5 0–63, gripper_link 64–127")
    ax0.set_ylabel("relative error to PhysX property volume (%)")
    ax0.set_title("D348 full 128-part semantic discriminator")
    ax0.grid(True, which="both", alpha=0.25)
    ax0.legend()

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for axis, triangles, title, color in (
        (fig.add_subplot(grid[1, 0], projection="3d"), topology_tri, "callback face topology", "#33bb77"),
        (fig.add_subplot(grid[1, 1], projection="3d"), qhull_tri, "vertex-only Qhull envelope", "#ee6677"),
    ):
        collection = Poly3DCollection(vertices[triangles], alpha=0.55, facecolor=color, edgecolor="black")
        axis.add_collection3d(collection)
        mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float((maxs - mins).max()) / 2.0, 1.0e-4)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_title(f"link5/part_045 — {title}")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")

    root = evidence["part045_root_cause"]
    fig.suptitle(
        "D348: property={:.9e} m³ | topology={:.9e} ({:.3g}%) | Qhull={:.9e} ({:.3f}%)".format(
            root["property_volume_m3"],
            root["callback_topology_volume_m3"],
            root["callback_topology_relative_error"] * 100.0,
            root["vertex_only_qhull_volume_m3"],
            root["vertex_only_qhull_relative_error"] * 100.0,
        ),
        fontsize=15,
    )
    fig.savefig(DECISION_PNG, dpi=120)
    plt.close(fig)


def _rerun_rows(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> tuple[list[Any], ...]:
    coordinate_frames = []
    for body in ("link5", "gripper_link"):
        coordinate_frames.append(
            {
                "entity_path": f"coordinate_frames/{body}/body_local",
                "frame": f"body/{body}",
                "parent_frame": "tf#/",
                "translation_m": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        )
    meshes = []
    scalars = []
    events = [
        {
            "entity_path": "events/d348",
            "text": "D348 START: immutable D347 callbacks; 5% and 128/128 gates frozen",
            "level": "INFO",
            "sequence": {"event_idx": 0},
        }
    ]
    colors = {
        "source": [45, 190, 115, 150],
        "instance": [238, 102, 119, 105],
        "prototype": [80, 145, 225, 150],
        "candidate": [230, 170, 50, 105],
    }
    for row in rows:
        part_idx = int(row["global_part_idx"])
        body = row["body"]
        name = row["name"]
        for variant, channel, triangle_key in (
            ("source", "instance", "topology_triangles"),
            ("instance", "instance", "qhull_triangles"),
            ("prototype", "prototype", "topology_triangles"),
            ("candidate", "prototype", "qhull_triangles"),
        ):
            channel_row = row[channel]
            meshes.append(
                {
                    "entity_path": f"cook/{variant}/{body}/parts/{name}",
                    "coordinate_frame": f"body/{body}",
                    "vertices_m": channel_row["vertices_m"],
                    "triangles": channel_row[triangle_key],
                    "color_rgba": colors[variant],
                    "sequence": {"part_idx": part_idx},
                    "static": True,
                    "scientific_source": channel_row["witness_path"],
                    "semantic_variant": variant,
                }
            )
        metric_values = {
            "property_volume_m3": row["property_volume_m3"],
            "topology_instance_volume_m3": row["instance"]["volume_origin_m3"],
            "topology_prototype_volume_m3": row["prototype"]["volume_origin_m3"],
            "topology_instance_relative_error": row["topology_instance_relative_error"],
            "topology_prototype_relative_error": row["topology_prototype_relative_error"],
            "qhull_volume_m3": row["instance"]["qhull_volume_m3"],
            "qhull_relative_error": row["qhull_relative_error"],
            "raw_payload_equal": float(row["checks"]["raw_instance_prototype_payload_exact"]),
            "closed_oriented": float(row["checks"]["both_topologies_closed_and_oriented"]),
            "part_pass": float(row["pass"]),
        }
        for metric, value in metric_values.items():
            scalars.append(
                {
                    "entity_path": f"metrics/{body}/{name}/{metric}",
                    "value": float(value),
                    "sequence": {"part_idx": part_idx},
                }
            )
        events.append(
            {
                "entity_path": "events/d348",
                "text": (
                    f"{body}/{name}: topology_rel={row['topology_instance_relative_error']:.12g}, "
                    f"qhull_rel={row['qhull_relative_error']:.12g}, pass={row['pass']}"
                ),
                "level": "INFO" if row["pass"] else "WARN",
                "sequence": {"event_idx": part_idx + 1},
            }
        )
    root = evidence["part045_root_cause"]
    events.extend(
        [
            {
                "entity_path": "events/d348",
                "text": (
                    "ROOT CAUSE part_045: callback faces match property; vertex-only re-Qhull "
                    f"changes volume by {root['vertex_only_qhull_relative_error'] * 100.0:.6f}%"
                ),
                "level": "WARN",
                "sequence": {"event_idx": 129},
            },
            {
                "entity_path": "events/d348",
                "text": (
                    "HOME CONTRACT: D347 source callback audit started HOME-near with ±0.02 rad "
                    "reset jitter and q5=0 closed; zero physics steps; D348 is offline re-analysis"
                ),
                "level": "INFO",
                "sequence": {"event_idx": 130},
            },
            {
                "entity_path": "events/d348",
                "text": (
                    f"NUMERIC VERDICT: {'SUPPORTED' if evidence['pass'] else 'FAIL'}; "
                    f"topology/property={evidence['aggregate']['topology_property_pass_count']}/256, "
                    f"part_gate={sum(int(row['pass']) for row in rows)}/128, frozen_tol=5%; "
                    "g0a_pass=false; target/settle require a separate case"
                ),
                "level": "INFO" if evidence["pass"] else "ERROR",
                "sequence": {"event_idx": 131},
            },
        ]
    )
    return coordinate_frames, meshes, scalars, events


def _run_rerun(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
    coordinate_frames, meshes, scalars, events = _rerun_rows(rows, evidence)
    exact_entities, expected_components = _expected_rrd_contract()
    log_status = log_rerun(
        RRD_PATH,
        coordinate_frames=coordinate_frames,
        meshes=meshes,
        scalar_trace=scalars,
        events=events,
        recording_metadata={
            "case": CASE,
            "scientific_authority": _relative(EVIDENCE_PATH),
            "property_relative_tolerance": PROPERTY_REL_TOL,
            "part_count": EXPECTED_PART_COUNT,
            "home_contract": _json(HOME_CONTRACT_PATH)["bounded_answer_ko"],
            "g0a_pass": False,
        },
        recording_id="g0a_d348_physx_property_query_volume_semantics",
        blueprint_path=RBL_PATH,
        blueprint_mode="volume_semantics",
        live_viewer=False,
        app_id="roarm_g0a_volume_semantics",
    )
    validation = (
        validate_rerun_artifact(
            RRD_PATH,
            expected_entity_paths=[
                "cook/source/link5/parts/part_045",
                "cook/instance/link5/parts/part_045",
                "cook/prototype/gripper_link/parts/part_058",
                "cook/candidate/gripper_link/parts/part_058",
                "metrics/link5/part_045/property_volume_m3",
                "events/d348",
            ],
            expected_timeline_names=["event_idx", "part_idx"],
            exact_entity_paths=exact_entities,
            exact_timeline_names=["blueprint", "event_idx", "log_time", "part_idx"],
            expected_entity_components=expected_components,
            blueprint_path=RBL_PATH,
            screenshot_path=RERUN_SCREENSHOT_PATH,
        )
        if log_status.get("ok")
        else {"pass": False, "errors": ["Rerun recording/finalization failed"]}
    )
    observed_count = len(
        validation.get("entity_path_contract", {}).get("observed_non_system", [])
    )
    count_checks = {
        "coordinate_frame_count": log_status.get("coordinate_frame_count") == 2,
        "mesh_count": log_status.get("mesh_count") == 512,
        "scalar_row_count": log_status.get("scalar_row_count") == 1280,
        "event_row_count": log_status.get("event_row_count") == 132,
        "exact_non_system_entity_count": observed_count == 2308,
        "trace_steps_zero": log_status.get("trace_steps") == 0,
    }
    report = {
        **validation,
        "d348_log_status": log_status,
        "d348_count_checks": count_checks,
        "d348_observed_non_system_entity_count": observed_count,
        "d348_contract_digest": _rrd_contract_digest(),
        "pass": bool(validation.get("pass")) and all(count_checks.values()),
    }
    _write_json(RERUN_VALIDATION_PATH, report)
    return report


def _run_analyze(_args: argparse.Namespace) -> int:
    if not PREREG_PATH.is_file():
        raise FileNotFoundError("run --stage prepare first")
    prereg = _json(PREREG_PATH)
    input_guard = _input_guard()
    prepare_checks = {
        "prereg_pass": prereg.get("pass") is True,
        "harness_hash_unchanged": prereg.get("harness_sha256") == _sha256(D348_HARNESS),
        "viz_debug_hash_unchanged": prereg.get("viz_debug_sha256") == _sha256(VIZ_DEBUG_SOURCE),
        "parameter_freeze_hash_unchanged": prereg.get("parameter_freeze_sha256")
        == _sha256(PARAMETER_FREEZE_PATH),
        "source_semantics_hash_unchanged": prereg.get("source_semantics_sha256")
        == _sha256(SOURCE_SEMANTICS_PATH),
        "home_contract_hash_unchanged": prereg.get("home_contract_sha256")
        == _sha256(HOME_CONTRACT_PATH),
        "attempt1_prepare_artifacts_still_preserved": _attempt1_prepare_guard()["pass"],
        "input_guard_pass": input_guard["pass"],
        "rrd_contract_digest_unchanged": prereg.get("rerun_contract", {}).get("digest")
        == _rrd_contract_digest(),
    }
    if not all(prepare_checks.values()):
        raise RuntimeError(f"D348 prepare/input contract failed: {prepare_checks}")
    rows, raw_controls = _analyze_rows()
    matched_controls = _matched_controls(rows, raw_controls)
    evidence = _evidence(rows, raw_controls)
    _write_json(EVIDENCE_PATH, evidence)
    _write_json(CONTROLS_PATH, matched_controls)
    _decision_png(rows, evidence)
    rerun = _run_rerun(rows, evidence)
    scientific_pass = evidence["pass"] and matched_controls["pass"]
    scientific_verdict = VERDICT_MANUAL_PENDING if scientific_pass else VERDICT_NUMERIC_FAIL
    observability_pass = rerun["pass"]
    automated_pass = scientific_pass and observability_pass
    summary = {
        "artifact": "D348_AUTOMATED_SUMMARY_V1",
        "case": CASE,
        "new_variables": NEW_VARIABLES,
        "prepare_checks": prepare_checks,
        "scientific_verdict": scientific_verdict,
        "observability_verdict": (
            "D348_RERUN_MACHINE_CONTRACT_PASS_MANUAL_PENDING"
            if observability_pass
            else VERDICT_OBSERVABILITY_FAIL
        ),
        "automated_pass": automated_pass,
        "manual_visual_inspection_pending": automated_pass,
        "scientific_evidence": {
            "path": _relative(EVIDENCE_PATH),
            "sha256": _sha256(EVIDENCE_PATH),
            "pass": evidence["pass"],
            "aggregate": evidence["aggregate"],
            "part045_root_cause": evidence["part045_root_cause"],
        },
        "matched_controls": {
            "path": _relative(CONTROLS_PATH),
            "sha256": _sha256(CONTROLS_PATH),
            "pass": matched_controls["pass"],
        },
        "decision_png": {
            "path": _relative(DECISION_PNG),
            "sha256": _sha256(DECISION_PNG),
            "bytes": DECISION_PNG.stat().st_size,
        },
        "rerun": {
            "validation_path": _relative(RERUN_VALIDATION_PATH),
            "validation_sha256": _sha256(RERUN_VALIDATION_PATH),
            "pass": rerun["pass"],
            "rrd_path": _relative(RRD_PATH),
            "rrd_sha256": _sha256(RRD_PATH) if RRD_PATH.is_file() else None,
            "rbl_path": _relative(RBL_PATH),
            "rbl_sha256": _sha256(RBL_PATH) if RBL_PATH.is_file() else None,
            "screenshot_path": _relative(RERUN_SCREENSHOT_PATH),
            "screenshot_sha256": (
                _sha256(RERUN_SCREENSHOT_PATH) if RERUN_SCREENSHOT_PATH.is_file() else None
            ),
        },
        "scope_guards": {
            "new_cook_requests": 0,
            "asset_writes": 0,
            "physics_steps": 0,
            "target_queries": 0,
            "settle_runs": 0,
            "g0a_pass": False,
            "g0b_rl_ladder_blocked": True,
        },
        "input_guard": input_guard,
    }
    _write_json(AUTOMATED_SUMMARY_PATH, summary)
    root = evidence["part045_root_cause"]
    report = (
        "# D348 자동 판독 결과\n\n"
        f"- 수치 판정: `{scientific_verdict}`\n"
        f"- Rerun 기계 계약: `{summary['observability_verdict']}`\n"
        f"- 256개 면-부피 비교 중 통과: "
        f"`{evidence['aggregate']['topology_property_pass_count']}/256`\n"
        f"- 최대 면-부피 상대 오차: "
        f"`{evidence['aggregate']['max_topology_property_relative_error']:.12g}`\n"
        f"- part_045 PhysX 부피: `{root['property_volume_m3']:.16g} m^3`\n"
        f"- part_045 콜백 면 부피: `{root['callback_topology_volume_m3']:.16g} m^3`\n"
        f"- part_045 기존 Qhull 오차: `{root['vertex_only_qhull_relative_error'] * 100.0:.9f}%`\n"
        "- 물리/자산/목표 변경: `0건`\n"
        "- G0a: `false` 유지\n"
        "- 다음 단계: 실제 Rerun 스크린샷과 결정 PNG를 원본 해상도로 검사한 뒤 finalize\n"
    )
    _write_text(AUTOMATED_REPORT_PATH, report)
    print(
        json.dumps(
            {
                "stage": "analyze",
                "scientific_verdict": scientific_verdict,
                "rerun_pass": rerun["pass"],
                "manual_pending": summary["manual_visual_inspection_pending"],
            },
            sort_keys=True,
        )
    )
    return 0 if automated_pass else 1


def _png_dimensions(path: Path) -> list[int] | None:
    if not path.is_file():
        return None
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return [int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")]


def _manual_checks(manual: dict[str, Any]) -> dict[str, bool]:
    decision = manual.get("decision_png", {})
    screenshot = manual.get("rerun_screenshot", {})
    observations = manual.get("observations", {})
    return {
        "artifact_exact": manual.get("artifact") == "D348_MANUAL_VISUAL_INSPECTION_V1",
        "case_exact": manual.get("case") == CASE,
        "inspection_date_exact": manual.get("inspection_date_kst") == "2026-07-14",
        "inspection_method_original_view_image": manual.get("inspection_method")
        == "original_resolution_view_image",
        "decision_png_path_exact": decision.get("path") == _relative(DECISION_PNG),
        "decision_png_sha_exact": DECISION_PNG.is_file()
        and decision.get("sha256") == _sha256(DECISION_PNG),
        "decision_png_bytes_exact": DECISION_PNG.is_file()
        and decision.get("bytes") == DECISION_PNG.stat().st_size,
        "decision_png_dimensions_exact": decision.get("raster_dimensions")
        == _png_dimensions(DECISION_PNG),
        "rerun_screenshot_path_exact": screenshot.get("path")
        == _relative(RERUN_SCREENSHOT_PATH),
        "rerun_screenshot_sha_exact": RERUN_SCREENSHOT_PATH.is_file()
        and screenshot.get("sha256") == _sha256(RERUN_SCREENSHOT_PATH),
        "rerun_screenshot_bytes_exact": RERUN_SCREENSHOT_PATH.is_file()
        and screenshot.get("bytes") == RERUN_SCREENSHOT_PATH.stat().st_size,
        "rerun_screenshot_dimensions_2400x1400": screenshot.get("raster_dimensions")
        == [2400, 1400]
        == _png_dimensions(RERUN_SCREENSHOT_PATH),
        "required_observations_true": all(
            observations.get(key) is True
            for key in (
                "both_bodies_visible",
                "four_semantic_panels_distinguishable",
                "part045_difference_readable",
                "frozen_5pct_and_128_of_128_readable",
                "home_contract_and_verdict_event_readable",
            )
        ),
        "manual_pass_true": manual.get("manual_visual_inspection_pass") is True,
        "scientific_override_false": manual.get("scientific_verdict_override") is False,
        "g0a_false": manual.get("g0a_pass") is False,
        "bounded_interpretation_present": bool(manual.get("bounded_interpretation")),
        "manual_markdown_nonzero": MANUAL_INSPECTION_MD_PATH.is_file()
        and MANUAL_INSPECTION_MD_PATH.stat().st_size > 0,
    }


def _run_finalize(_args: argparse.Namespace) -> int:
    automated = _json(AUTOMATED_SUMMARY_PATH)
    manual = _json(MANUAL_INSPECTION_PATH)
    rerun_validation = _json(RERUN_VALIDATION_PATH)
    prereg = _json(PREREG_PATH)
    input_guard = _input_guard()
    manual_checks = _manual_checks(manual)
    artifact_checks = {
        "input_guard_pass": input_guard["pass"],
        "harness_hash_unchanged": prereg.get("harness_sha256") == _sha256(D348_HARNESS),
        "viz_debug_hash_unchanged": prereg.get("viz_debug_sha256")
        == _sha256(VIZ_DEBUG_SOURCE),
        "parameter_freeze_hash_unchanged": prereg.get("parameter_freeze_sha256")
        == _sha256(PARAMETER_FREEZE_PATH),
        "source_semantics_hash_unchanged": prereg.get("source_semantics_sha256")
        == _sha256(SOURCE_SEMANTICS_PATH),
        "home_contract_hash_unchanged": prereg.get("home_contract_sha256")
        == _sha256(HOME_CONTRACT_PATH),
        "attempt1_prepare_artifacts_still_preserved": _attempt1_prepare_guard()["pass"],
        "evidence_hash_unchanged": automated["scientific_evidence"]["sha256"]
        == _sha256(EVIDENCE_PATH),
        "controls_hash_unchanged": automated["matched_controls"]["sha256"]
        == _sha256(CONTROLS_PATH),
        "rerun_validation_hash_unchanged": automated["rerun"]["validation_sha256"]
        == _sha256(RERUN_VALIDATION_PATH),
        "rrd_hash_unchanged": automated["rerun"]["rrd_sha256"] == _sha256(RRD_PATH),
        "rbl_hash_unchanged": automated["rerun"]["rbl_sha256"] == _sha256(RBL_PATH),
        "screenshot_hash_unchanged": automated["rerun"]["screenshot_sha256"]
        == _sha256(RERUN_SCREENSHOT_PATH),
        "decision_png_hash_unchanged": automated["decision_png"]["sha256"]
        == _sha256(DECISION_PNG),
        "rerun_machine_pass": rerun_validation.get("pass") is True,
    }
    scientific_supported = automated.get("scientific_verdict") == VERDICT_MANUAL_PENDING
    if scientific_supported and all(artifact_checks.values()) and all(manual_checks.values()):
        final_verdict = VERDICT_COMPLETE
    elif automated.get("scientific_verdict") == VERDICT_NUMERIC_FAIL:
        final_verdict = VERDICT_NUMERIC_FAIL
    else:
        final_verdict = VERDICT_OBSERVABILITY_FAIL
    completion_pass = final_verdict == VERDICT_COMPLETE
    completion = {
        "artifact": "D348_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "final_verdict": final_verdict,
        "completion_contract_pass": completion_pass,
        "new_variables": NEW_VARIABLES,
        "scientific_evidence": automated["scientific_evidence"],
        "matched_controls": automated["matched_controls"],
        "rerun_evidence": automated["rerun"],
        "manual_evidence": {
            "path": _relative(MANUAL_INSPECTION_PATH),
            "sha256": _sha256(MANUAL_INSPECTION_PATH),
            "checks": manual_checks,
            "pass": all(manual_checks.values()),
        },
        "artifact_checks": artifact_checks,
        "input_guard": input_guard,
        "interpretation_ko": (
            "D347의 유일한 27.33% 불일치는 실제 충돌 자산 실패가 아니라, 콜백이 준 "
            "면 목록을 버리고 꼭짓점을 새로 볼록껍질화한 비교기 오류였다. 이 결론은 "
            "현재 PhysX 버전과 불변 D347 256개 콜백에 한정된다."
        ),
        "home_answer_ko": _json(HOME_CONTRACT_PATH)["bounded_answer_ko"],
        "scope_guards": {
            "physics_steps": 0,
            "asset_writes": 0,
            "cook_requests": 0,
            "target_queries": 0,
            "g0a_pass": False,
            "g0b_rl_ladder_blocked": True,
        },
        "next_case_requires_separate_approval": True,
        "next_case_eligible_only_if_complete": completion_pass,
    }
    _write_json(COMPLETION_SUMMARY_PATH, completion)
    root = automated["scientific_evidence"]["part045_root_cause"]
    report = (
        "# D348 완료 보고\n\n"
        f"- 최종 판정: `{final_verdict}`\n"
        f"- 256개 콜백 면-부피 비교: "
        f"`{automated['scientific_evidence']['aggregate']['topology_property_pass_count']}/256`\n"
        f"- 최대 상대 오차: "
        f"`{automated['scientific_evidence']['aggregate']['max_topology_property_relative_error']:.12g}`\n"
        f"- part_045 기존 Qhull 상대 오차: "
        f"`{root['vertex_only_qhull_relative_error'] * 100.0:.9f}%`\n"
        f"- part_045 콜백 면 상대 오차: "
        f"`{root['callback_topology_relative_error'] * 100.0:.12g}%`\n"
        "- HOME 답: 정확한 HOME이 아니라 HOME 근방의 닫힌 q5=0 reset 자세에서 검사; "
        "물리 step 0회\n"
        "- 자산·분해·목표·허용값·물리 변경: `0건`\n"
        "- G0a: `false` 유지\n"
        "- 다음: 별도 승인 case에서만 동결 목표 거리 게이트를 다시 실행할 수 있음\n"
    )
    _write_text(COMPLETION_REPORT_PATH, report)
    print(json.dumps({"stage": "finalize", "final_verdict": final_verdict}, sort_keys=True))
    return 0 if completion_pass else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "analyze", "finalize"), required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.stage == "prepare":
        return _run_prepare(args)
    if args.stage == "analyze":
        return _run_analyze(args)
    return _run_finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
