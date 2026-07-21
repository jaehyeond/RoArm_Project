#!/usr/bin/env python3
"""D372 professor-style semantic compound collider design, offline only.

This case constructs a forward-only candidate with a simple link5 housing,
separate fixed-jaw prisms, and separate moving-jaw prisms.  It reads immutable
raw meshes and prior evidence.  It does not launch Isaac/Kit/PhysX, author a USD,
step physics, evaluate q5, query live contacts, or change target/IK/path.
"""

from __future__ import annotations

import argparse
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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASE_ROOT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d372"
ATTEMPT_NAME = "attempt2_external_schema_path_repair"
OUT_DIR = CASE_ROOT / ATTEMPT_NAME
ATTEMPT1_EXCEPTION_PATH = CASE_ROOT / "d372_runtime_exception.json"
PREREG_PATH = OUT_DIR / "d372_preregistration.json"
PHASE_PATH = OUT_DIR / "d372_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d372_invocation.json"
CANDIDATE_PATH = OUT_DIR / "d372_professor_semantic_candidate_geometry.json"
EVIDENCE_PATH = OUT_DIR / "d372_professor_semantic_candidate_evidence.json"
REPORT_PATH = OUT_DIR / "d372_professor_semantic_candidate_report.md"
OWNERSHIP_PNG = OUT_DIR / "d372_ownership_and_split_1920x1080.png"
JAWS_PNG = OUT_DIR / "d372_jaw_void_preservation_1920x1080.png"
CLEARANCE_PNG = OUT_DIR / "d372_frozen_open_clearance_1920x1080.png"
RRD_PATH = OUT_DIR / "d372_professor_semantic_candidate.rrd"
RBL_PATH = OUT_DIR / "d372_professor_semantic_candidate.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d372_rerun_validation.json"
RERUN_PNG = OUT_DIR / "d372_rerun_inspection.png"
AUTOMATED_PATH = OUT_DIR / "d372_automated_summary.json"
MANUAL_JSON_PATH = OUT_DIR / "d372_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d372_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d372_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d372_runtime_exception.json"

HARNESS = Path(__file__).resolve()
D368_HARNESS = REPO / "sim_scripts/cyl34_top_view_d368_current_64cap_semantic_allocation_audit.py"
D371_HARNESS = REPO / "sim_scripts/cyl34_top_view_d371_offline_collider_candidate_pareto_comparison.py"
AUTHORING_USD = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"
URDF_PATH = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
D349_MEASUREMENT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d349/d349_frozen_target_distance_measurement.json"
D350_BINDING = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d350/d350_fixed_jaw_semantic_binding.json"
D350_MEASUREMENT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d350/d350_fixed_jaw_geometry_measurement.json"
D354_BINDING = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354/d354_moving_jaw_surface_binding.json"
D368_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_evidence.json"
D371_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d371/d371_offline_collider_comparison_evidence.json"
D362_TRACE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d362/d362_physics_trace.json"
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"

EXPECTED_HEAD = "4a1120b801e808071583136e78954c78ca941dc8"
EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
PXR_ROOT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
PHYSX_SCHEMA = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "plugins/PhysxSchema/resources/schema.usda"
)
PHYSX_EXTENSION_TOML = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/config/extension.toml"
)
PHYSX_PROPERTY_DATABASE = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.kit.property.physx-107.3.26+107.3.3.cp311.u353/"
    "omni/kit/property/physx/database.py"
)
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = [
    "semantic_owner_region_partition_v1",
    "manual_compound_primitive_budget_v1",
]
BODY_NAMES = ("link5", "gripper_link")
RASTER_STEP_M = 0.00025
MAJOR_VOID_MIN_AREA_MM2 = 20.0
MISSING_COMPONENT_MIN_AREA_MM2 = 2.0
FIXED_EXTRA_MAJOR_VOID_FILL_LIMIT_MM2 = 1.0
FIXED_EXTRA_GHOST_LIMIT_MM2 = 10.0
OPEN_CLEAR_GATE_MM = 0.1
OPEN_RAW_DELTA_GATE_MM = 0.5
PATCH_MAX_GHOST_RATIO = {"fixed": 0.10, "moving": 0.15}
PATCH_MAX_MISSING_RATIO = {"fixed": 0.01, "moving": 0.005}
PATCH_MAX_GHOST_DISTANCE_MM = {"fixed": 2.0, "moving": 2.0}
RAW_TO_CANDIDATE_P95_MAX_MM = 2.0
RAW_TO_CANDIDATE_MAX_MM = 3.5
TRACE_TRANSITION_WINDOW_DELTA_MAX_MM = 0.1
TRACE_PHASE = "q5_close_observation"
TRACE_PHASE_STEP_MAX = 60
EXPECTED_TRACE_FIRST_GLOBAL_STEP = {"link5": 246, "gripper_link": 232}
JAW_THICKNESS_TARGET_M = 0.0015
JAW_THICKNESS_TOL_M = 1.0e-9
MAJOR_VOID_MAX_FILL = {
    "fixed_primary": 0.05,
    "fixed_secondary": 0.05,
    "moving_open_mouth": 0.10,
    "moving_internal_window_diagnostic": 0.30,
}
EXPECTED_PART_COUNTS = {"link5": 16, "gripper_link": 18, "total": 34}
EXPECTED_ROLE_COUNTS = {
    "link5": {"connector_support": 3, "fixed_jaw": 10, "fixed_jaw_backbone": 2, "structural_body": 1},
    "gripper_link": {"moving_jaw": 12, "moving_jaw_backbone": 2, "moving_support": 4},
}
REQUIRED_FIXED_BASE_ZONES = {
    "lower_bridge", "lower_left_leg", "lower_right_leg", "middle_bridge",
    "upper_left_leg", "upper_right_leg", "roof_bridge",
}
REQUIRED_MOVING_BASE_ZONES = {
    "proximal_upper_rail", "proximal_lower_rail", "center_bridge",
    "window_upper_rail", "window_lower_rail", "distal_nose_bridge",
}
VERDICT_PASS = "D372_PROFESSOR_SEMANTIC_COMPOUND_CANDIDATE_OFFLINE_PASS_NO_PHYSICS"
VERDICT_FAIL = "D372_PROFESSOR_SEMANTIC_COMPOUND_CANDIDATE_OFFLINE_FAIL_STOP"
VERDICT_VIZ_FAIL = "D372_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"

OFFICIAL_SOURCES = [
    {
        "title": "Omni Physics 107.3 Rigid Bodies",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html",
        "use": "one rigid body may own multiple child colliders",
    },
    {
        "title": "Omni Physics 107.3 Colliders",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html",
        "use": "primitives first, convex meshes when needed",
    },
    {
        "title": "Isaac Sim 5.1 Physics Simulation Fundamentals",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html",
        "use": "multiple convex shapes preserve concave openings",
    },
    {
        "title": "Isaac Sim 5.1 Performance Optimization Handbook",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html",
        "use": "use the simplest collision approximation that satisfies precision",
    },
    {
        "title": "PhysX 5.6.1 GPU Rigid Bodies",
        "url": "https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html",
        "use": "supplementary convex geometry limits; not an installed SDK identity claim",
    },
    {
        "title": "Omni Physics 107.3 PhysX Schema API",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/schemas/physxschema.html",
        "use": "version-matched schema API defaults for hullVertexLimit and maxConvexHulls",
    },
]


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blob(value: Any, dtype: str) -> bytes:
    return np.ascontiguousarray(np.asarray(value, dtype=dtype)).tobytes(order="C")


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
        return _rel(value)
    raise TypeError(type(value).__name__)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default)
        stream.write("\n")


def _write_text_x(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _phase(name: str, **fields: Any) -> None:
    ordinal = 1
    if PHASE_PATH.is_file():
        ordinal = sum(1 for line in PHASE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()) + 1
    row = {"ordinal": ordinal, "phase": name, "monotonic_ns": time.monotonic_ns(), **fields}
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sidecar_snapshot() -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in D334_SIDECAR.rglob("*") if item.is_file()):
        rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha(path)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"file_count": len(rows), "inventory_sha256": _sha_bytes(canonical), "files": rows}


def _input_paths() -> list[Path]:
    return [
        D368_HARNESS,
        D371_HARNESS,
        AUTHORING_USD,
        URDF_PATH,
        D349_MEASUREMENT,
        D350_BINDING,
        D350_MEASUREMENT,
        D354_BINDING,
        D368_EVIDENCE,
        D371_EVIDENCE,
        D362_TRACE,
        VIZ_DEBUG,
        RERUN_CONTRACT,
        ATTEMPT1_EXCEPTION_PATH,
    ]


def _input_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in _input_paths()}


def _forbidden_modules() -> list[str]:
    prefixes = ("isaacsim", "isaaclab", "omni", "carb", "physx", "warp", "torch")
    return sorted(name for name in sys.modules if any(name == p or name.startswith(p + ".") for p in prefixes))


def _environment() -> dict[str, Any]:
    from pxr import Usd
    import hppfcl
    import matplotlib
    import rerun as rr
    import scipy
    import trimesh

    cli = subprocess.run([str(RERUN_CLI), "--version"], capture_output=True, text=True, check=False, timeout=30)
    report = {
        "python": str(Path(sys.executable).resolve()),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "path_first": os.environ.get("PATH", "").split(os.pathsep)[0],
        "openusd_version": list(Usd.GetVersion()),
        "versions": {
            "numpy": np.__version__,
            "psutil": importlib.metadata.version("psutil"),
            "rerun": str(rr.__version__),
            "scipy": str(scipy.__version__),
            "trimesh": str(trimesh.__version__),
            "hppfcl": getattr(hppfcl, "__version__", "unknown"),
            "matplotlib": str(matplotlib.__version__),
        },
        "rerun_cli": {"returncode": cli.returncode, "stdout": cli.stdout.strip(), "stderr": cli.stderr.strip()},
        "forbidden_modules": _forbidden_modules(),
    }
    checks = {
        "python_exact": Path(report["python"]) == EXPECTED_PYTHON.resolve(),
        "numpy_1_26_0": report["versions"]["numpy"] == "1.26.0",
        "psutil_5_9_8": report["versions"]["psutil"] == "5.9.8",
        "rerun_0_34_1": report["versions"]["rerun"] == "0.34.1",
        "openusd_0_24_5": report["openusd_version"] == [0, 24, 5],
        "rerun_cli_0_34_1": cli.returncode == 0 and "0.34.1" in (cli.stdout + cli.stderr),
        "no_isaac_kit_physx_warp_torch_modules": report["forbidden_modules"] == [],
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def _schema_facts() -> dict[str, Any]:
    import re

    text = PHYSX_SCHEMA.read_text(encoding="utf-8")
    extension_text = PHYSX_EXTENSION_TOML.read_text(encoding="utf-8")
    property_text = PHYSX_PROPERTY_DATABASE.read_text(encoding="utf-8")
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', extension_text, flags=re.MULTILINE)
    report = {
        "installed_isaac_sim": importlib.metadata.version("isaacsim"),
        "installed_isaac_sim_source": "importlib.metadata",
        "installed_omni_physx_extension": version_match.group(1) if version_match else None,
        "installed_omni_physx_extension_source": str(PHYSX_EXTENSION_TOML),
        "installed_physx_sdk_semver": None,
        "schema_path": str(PHYSX_SCHEMA),
        "schema_sha256": _sha(PHYSX_SCHEMA),
        "property_database_path": str(PHYSX_PROPERTY_DATABASE),
        "property_database_sha256": _sha(PHYSX_PROPERTY_DATABASE),
        "schema_hull_vertex_limit_line": 886,
        "schema_max_convex_hulls_line": 895,
        "extension_version_line": 5,
        "property_database_info_data_line": 887,
        "property_database_ranges_line": 954,
        "default_max_convex_hulls_32": "physxConvexDecompositionCollision:maxConvexHulls = 32" in text,
        "default_hull_vertex_limit_64": "physxConvexDecompositionCollision:hullVertexLimit = 64" in text,
        "ui_hull_vertex_limit_range_8_64": '"physxConvexDecompositionCollision:hullVertexLimit": InfoData(8, 64, 1)' in property_text,
        "ui_max_convex_hulls_range_1_2048": '"physxConvexDecompositionCollision:maxConvexHulls": InfoData(1, 2048, 1)' in property_text,
        "manual_compound_note": "32 is an automatic decomposition default, not a target count for child primitives",
        "ui_range_note": "2048 is an installed property-editor authoring maximum, not an engine hard limit or optimum",
        "official_sources": OFFICIAL_SOURCES,
    }
    report["checks"] = {
        "isaac_sim_5_1_0_0": report["installed_isaac_sim"] == "5.1.0.0",
        "omni_physx_107_3_26": report["installed_omni_physx_extension"] == "107.3.26",
        "schema_default_32": report["default_max_convex_hulls_32"],
        "schema_vertex_limit_64": report["default_hull_vertex_limit_64"],
        "ui_hull_vertex_limit_range_8_64": report["ui_hull_vertex_limit_range_8_64"],
        "ui_max_convex_hulls_range_1_2048": report["ui_max_convex_hulls_range_1_2048"],
    }
    report["pass"] = all(report["checks"].values())
    return report


def _prior_evidence_and_owner_contract(
    candidate: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Recheck the inherited evidence contents and derive ownership from URDF.

    Input hashes alone prove immutability, not that the frozen files contain the
    PASS/subresult semantics D372 needs.  This contract therefore reads those
    fields explicitly and independently derives link ownership from the q5
    parent/child joint in the frozen URDF.
    """

    d368 = _read_json(D368_EVIDENCE)
    d371 = _read_json(D371_EVIDENCE)
    d350 = _read_json(D350_BINDING)
    d354 = _read_json(D354_BINDING)

    root = ET.parse(URDF_PATH).getroot()
    joint = next(
        item
        for item in root.findall("joint")
        if item.attrib.get("name") == "link5_to_gripper_link"
    )
    parent = joint.find("parent").attrib["link"]
    child = joint.find("child").attrib["link"]
    joint_type = joint.attrib.get("type")
    expected_manifest = {"fixed_jaw": parent, "moving_jaw": child}

    fixed_roles = {"fixed_jaw", "fixed_jaw_backbone"}
    moving_roles = {"moving_jaw", "moving_jaw_backbone", "moving_support"}
    fixed_owners = sorted(
        {
            body
            for body, parts in candidate.items()
            for part in parts
            if part["role"] in fixed_roles
        }
    )
    moving_owners = sorted(
        {
            body
            for body, parts in candidate.items()
            for part in parts
            if part["role"] in moving_roles
        }
    )
    candidate_manifest = {
        "fixed_jaw": fixed_owners[0] if len(fixed_owners) == 1 else fixed_owners,
        "moving_jaw": moving_owners[0] if len(moving_owners) == 1 else moving_owners,
    }
    all_part_body_fields_match_container = all(
        part.get("body") == body
        for body, parts in candidate.items()
        for part in parts
    )

    d368_lineage = d368.get("lineage_checks", {})
    d371_r32 = d371.get("R64_R32_cap_isolation", {}).get("actual_counts", {}).get("R32")
    d371_boundary = d371.get("interpretation_boundary", {})
    d350_joint = d350.get("joint_owner_negative_control", {})
    d354_joint = d354.get("joint_semantics", {})
    checks = {
        "D368_artifact_exact": d368.get("artifact")
        == "D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_EVIDENCE_V1",
        "D368_measurement_pass": d368.get("measurement_pass") is True,
        "D368_verdict_exact": d368.get("verdict")
        == "D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_MEASURED_NO_PHYSICS",
        "D368_lineage_checks_all_true": bool(d368_lineage)
        and all(value is True for value in d368_lineage.values()),
        "D368_lineage_gate_true": d368.get("checks", {}).get("lineage_checks_pass") is True,
        "D368_g0a_false_preserved": d368.get("interpretation_boundary", {}).get("g0a_pass") is False,
        "D371_artifact_exact": d371.get("artifact") == "D371_OFFLINE_COLLIDER_COMPARISON_EVIDENCE_V1",
        "D371_measurement_pass": d371.get("measurement_pass") is True,
        "D371_verdict_exact": d371.get("verdict") == "D371_OFFLINE_COLLIDER_PARETO_MEASURED_NO_PHYSICS",
        "D371_current_A_128_pass": d371.get("current_A_inventory", {}).get("pass") is True
        and d371.get("current_A_inventory", {}).get("part_counts")
        == {"link5": 64, "gripper_link": 64}
        and d371.get("current_A_inventory", {}).get("total_parts") == 128,
        "D371_R32_32_plus_32_exact": d371_r32 == {"link5": 32, "gripper_link": 32},
        "D371_C1_C2_not_exact_partition_preserved": d371_boundary.get(
            "C1_C2_are_exact_body_jaw_partition"
        )
        is False,
        "D371_live_or_physics_still_requires_approval": d371_boundary.get(
            "next_live_authoring_or_physics_requires_new_approval"
        )
        is True,
        "D350_fixed_owner_subresult_pass": d350.get("pass") is True
        and d350.get("checks", {}).get("owner_link5") is True,
        "D350_joint_matches_URDF": d350_joint.get("pass") is True
        and d350_joint.get("parent") == parent
        and d350_joint.get("child") == child
        and d350_joint.get("joint") == joint.attrib.get("name"),
        "D354_overall_false_preserved": d354.get("pass") is False,
        "D354_moving_owner_subresult_pass": d354.get("checks", {}).get("owner_gripper_link") is True
        and d354.get("checks", {}).get("q5_child_gripper_link") is True
        and d354.get("checks", {}).get("link5_parent_negative_control_rejected") is True,
        "D354_joint_matches_URDF": d354_joint.get("pass") is True
        and d354_joint.get("parent") == parent
        and d354_joint.get("child") == child
        and d354_joint.get("joint") == joint.attrib.get("name"),
        "URDF_joint_revolute_parent_link5_child_gripper_link": joint_type == "revolute"
        and parent == "link5"
        and child == "gripper_link",
        "candidate_part_body_fields_match_container": all_part_body_fields_match_container,
        "candidate_owner_manifest_matches_URDF": candidate_manifest == expected_manifest,
    }
    return {
        "sources": {
            _rel(path): {"sha256": _sha(path)}
            for path in (D368_EVIDENCE, D371_EVIDENCE, D350_BINDING, D354_BINDING, URDF_PATH)
        },
        "owner_contract": {
            "joint": joint.attrib.get("name"),
            "joint_type": joint_type,
            "urdf_parent": parent,
            "urdf_child": child,
            "expected_manifest": expected_manifest,
            "candidate_manifest": candidate_manifest,
            "fixed_role_owners": fixed_owners,
            "moving_role_owners": moving_owners,
        },
        "candidate_count_lineage": {
            "current_A64_automatic_decomposition": {"link5": 64, "gripper_link": 64, "total": 128},
            "D371_R32_automatic_decomposition": {"link5": 32, "gripper_link": 32, "total": 64},
            "D372_P34_manual_semantic_compound": EXPECTED_PART_COUNTS,
            "D371_C1_C2_exact_body_jaw_partition": False,
            "count_alone_proves_speed_physics_or_optimality": False,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _prepare() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"refusing nonempty forward-only path: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/master")
    schema = _schema_facts()
    checks = {
        "head_expected": head == EXPECTED_HEAD,
        "head_equals_origin_master": head == origin,
        "new_variable_count_two": len(NEW_VARIABLES) == 2,
        "environment_pass": environment["pass"],
        "installed_nvidia_contract_pass": schema["pass"],
        "all_inputs_exist": all(path.is_file() for path in _input_paths()),
        "output_empty_before_preregistration": not any(OUT_DIR.iterdir()),
        "attempt1_prepare_failure_preserved_without_run": ATTEMPT1_EXCEPTION_PATH.is_file()
        and not (CASE_ROOT / "d372_preregistration.json").exists()
        and not (CASE_ROOT / "d372_invocation.json").exists(),
    }
    prereg = {
        "artifact": "D372_PREREGISTRATION_V1",
        "case": "g0a_d372",
        "attempt": ATTEMPT_NAME,
        "preceding_prepare_failure": {
            "path": _rel(ATTEMPT1_EXCEPTION_PATH),
            "sha256": _sha(ATTEMPT1_EXCEPTION_PATH),
            "semantics": "prepare-only external installed-file relative-path bug; run invocation 0; no Isaac/PhysX/q5/physics",
        },
        "title": "professor_semantic_compound_collider_design_offline",
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "head": head,
        "origin_master": origin,
        "git_status_before_prepare": _git("status", "--short").splitlines(),
        "harness_sha256": _sha(HARNESS),
        "input_hashes": _input_hashes(),
        "d334_sidecar_before": _sidecar_snapshot(),
        "environment": environment,
        "nvidia_contract": schema,
        "design_contract": {
            "ownership": {
                "link5": "one rigid body; one lower-housing box + connector/pivot support + fixed-jaw contact prisms and side backbones",
                "gripper_link": "separate q5 child rigid body; proximal support hulls + moving-jaw contact prisms and separated upper/lower backbones",
            },
            "raster_step_mm": RASTER_STEP_M * 1000.0,
            "major_void_min_area_mm2": MAJOR_VOID_MIN_AREA_MM2,
            "ignored_missing_component_below_mm2": MISSING_COMPONENT_MIN_AREA_MM2,
            "fixed_jaw_partition": "seven functional bands around two large openings, then split omitted >=2mm2 components when a large opening would gain >1mm2 false fill",
            "moving_jaw_partition": "six functional bands around the open mouth and internal window, then one convex prism per omitted >=2mm2 connected brace",
            "required_fixed_base_zone_names": sorted(REQUIRED_FIXED_BASE_ZONES),
            "required_moving_base_zone_names": sorted(REQUIRED_MOVING_BASE_ZONES),
            "structural_primitives": {
                "link5": [
                    "lower_housing_box",
                    "neck_connector_box",
                    "pivot_support_box",
                    "lower_pivot_hub",
                    "fixed_backbone_left",
                    "fixed_backbone_right",
                ],
                "gripper_link": [
                    "proximal_upper_arm_hull_a",
                    "proximal_upper_arm_hull_b",
                    "proximal_lower_arm_hull_a",
                    "proximal_lower_arm_hull_b",
                    "moving_upper_backbone",
                    "moving_lower_backbone",
                ],
            },
            "semantic_selection_formulas": {
                "fixed_backbone_left": "raw broad component; y in [-0.0177465,-0.0162360], x in [-0.023250,-0.011525], z in [0.016085,0.107613] m",
                "fixed_backbone_right": "raw broad component; y in [0.015211,0.016722], x in [-0.023250,-0.011525], z in [0.016085,0.107613] m",
                "lower_pivot_hub": "raw broad component; x in [-0.0311,-0.0100], y in [-0.0096,0.0086], z in [0.0790,0.0878] m",
                "moving_proximal_support": "raw gripper vertices at x <= D354 inner-contact-patch x minimum; split by mouth z bounds and each branch median y",
                "moving_upper_backbone": "raw gripper vertices at x >= D354 inner-patch x minimum, y >= D354 outer plane, z >= D354 inner-patch top edge",
                "moving_lower_backbone": "raw gripper vertices at x >= D354 inner-patch x minimum, y >= D354 outer plane, z <= D354 inner-patch bottom edge",
            },
            "expected_part_counts": EXPECTED_PART_COUNTS,
            "expected_role_counts": EXPECTED_ROLE_COUNTS,
            "mass_inertia_policy": "preserve existing explicit URDF/USD mass and inertia in any later asset; no mass/inertia write in D372",
        },
        "registered_gates": {
            "part_counts_exact": EXPECTED_PART_COUNTS,
            "per_convex_project_authored_gpu_eligibility_gate": {
                "vertices_max": 64,
                "polygons_max": 64,
                "vertices_per_polygon_max": 32,
                "semantics": "project gate informed by the cited PhysX GPU guide; not a schema default or proof of runtime GPU execution",
            },
            "patch_max_ghost_ratio": PATCH_MAX_GHOST_RATIO,
            "patch_max_missing_ratio": PATCH_MAX_MISSING_RATIO,
            "patch_max_ghost_distance_mm": PATCH_MAX_GHOST_DISTANCE_MM,
            "structural_source_vertices_must_all_be_contained": True,
            "whole_raw_surface_containment": "diagnostic only; point-count/tessellation density is not a pass authority",
            "raw_surface_sample_to_candidate_p95_max_mm": RAW_TO_CANDIDATE_P95_MAX_MM,
            "raw_surface_sample_to_candidate_max_mm": RAW_TO_CANDIDATE_MAX_MM,
            "raw_semantic_face_classification_overlap_and_missing": 0,
            "major_void_max_fill_fraction": MAJOR_VOID_MAX_FILL,
            "major_void_gate_scope": "2-D contact-skin plus adjacent jaw-backbone diagnostic only; connector/support parts excluded; not a full-candidate or through-depth proof",
            "frozen_open_clearance_min_mm": OPEN_CLEAR_GATE_MM,
            "absolute_delta_from_D349_raw_max_mm": OPEN_RAW_DELTA_GATE_MM,
            "contact_seed_to_candidate_max_mm": RASTER_STEP_M * 1000.0,
            "immutable_D362_trace_phase": TRACE_PHASE,
            "immutable_D362_trace_phase_steps": [0, TRACE_PHASE_STEP_MAX],
            "A64_and_P34_first_collision_global_steps_exact": EXPECTED_TRACE_FIRST_GLOBAL_STEP,
            "transition_window_A64_vs_P34_abs_distance_delta_max_mm": TRACE_TRANSITION_WINDOW_DELTA_MAX_MM,
        },
        "negative_controls": [
            "full_link5_AABB must be rejected by frozen-open collision/clearance",
            "single fixed-jaw envelope must fill both major openings and be rejected",
            "single moving-jaw envelope must fill the open mouth and be rejected",
            "removing fixed and moving contact layers must lose the D350 and D354 certified seeds",
            "owner swap must be rejected against URDF parent/child plus D350/D354 owner subresults",
        ],
        "decision_rule": {
            "pass": VERDICT_PASS,
            "fail": VERDICT_FAIL,
            "visualization_fail": VERDICT_VIZ_FAIL,
            "pass_means": "offline candidate geometry is internally eligible for a later live-asset identity preflight; not physics equivalence, speed, grasp, or optimality",
        },
        "scope_guards": {
            "simulation_app_or_kit": 0,
            "isaac_or_physx": 0,
            "cook_or_automatic_decomposition": 0,
            "usd_or_live_asset_writes": 0,
            "physics_steps": 0,
            "q5_samples": 0,
            "live_contact_queries": 0,
            "target_ik_path_changes": 0,
            "material_mass_actuator_physics_changes": 0,
            "offline_hppfcl_static_geometry_queries_allowed": True,
            "immutable_D362_trace_read_only_replay_allowed": True,
            "rerun_save_and_headless_screenshot_allowed": True,
        },
        "single_run_contract": {"run_invocation_count": 1, "automatic_retry_count": 0, "forward_only": True},
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    _phase("prepare_complete", passed=prereg["pass"])
    if not prereg["pass"]:
        raise RuntimeError(f"D372 prepare failed: {checks}")
    print(json.dumps({"stage": "prepare", "pass": True, "path": _rel(PREREG_PATH)}, ensure_ascii=False))


def _polygon_mask(poly: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    polygon = np.asarray(poly, dtype=np.float64)
    closed = np.vstack([polygon, polygon[0]])
    xgrid, ygrid = np.meshgrid(xs, ys, indexing="xy")
    query = np.column_stack([xgrid.ravel(), ygrid.ravel()])
    return MplPath(closed, closed=True).contains_points(query, radius=1.0e-12).reshape(len(ys), len(xs))


def _convex_polygon(points: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import ConvexHull

    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(points) < 3:
        raise RuntimeError("convex polygon needs at least three points")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        half = min(float(xs[1] - xs[0]), float(ys[1] - ys[0])) * 0.5
        lo, hi = points.min(axis=0) - half, points.max(axis=0) + half
        poly = np.asarray([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    else:
        hull = ConvexHull(points)
        poly = points[hull.vertices]
    return poly, _polygon_mask(poly, xs, ys)


def _raster_patch(patch: dict[str, Any]) -> dict[str, Any]:
    from matplotlib.path import Path as MplPath

    axis = int(patch["axis"])
    keep = [value for value in range(3) if value != axis]
    vertices = np.asarray(patch["vertices"], dtype=np.float64)[:, keep]
    triangles = np.asarray(patch["triangles"], dtype=np.int64)
    lo, hi = vertices.min(axis=0) - RASTER_STEP_M, vertices.max(axis=0) + RASTER_STEP_M
    xs = np.arange(lo[0] + RASTER_STEP_M / 2.0, hi[0], RASTER_STEP_M)
    ys = np.arange(lo[1] + RASTER_STEP_M / 2.0, hi[1], RASTER_STEP_M)
    xgrid, ygrid = np.meshgrid(xs, ys, indexing="xy")
    query = np.column_stack([xgrid.ravel(), ygrid.ravel()])
    flat = np.zeros(len(query), dtype=bool)
    for triangle in triangles:
        poly = vertices[triangle]
        closed = np.vstack([poly, poly[0]])
        flat |= MplPath(closed, closed=True).contains_points(query, radius=1.0e-12)
    mask = flat.reshape(len(ys), len(xs))
    occupied = np.argwhere(mask)
    outer_poly, outer_mask = _convex_polygon(np.column_stack([xs[occupied[:, 1]], ys[occupied[:, 0]]]), xs, ys)
    return {"axis": axis, "keep": keep, "vertices": vertices, "xs": xs, "ys": ys, "mask": mask, "outer_poly": outer_poly, "outer_mask": outer_mask}


def _major_voids(raster: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from scipy.ndimage import label

    void_mask = raster["outer_mask"] & ~raster["mask"]
    labels, count = label(void_mask)
    rows = []
    for identity in range(1, count + 1):
        cells = np.argwhere(labels == identity)
        area_mm2 = len(cells) * (RASTER_STEP_M * 1000.0) ** 2
        if area_mm2 < MAJOR_VOID_MIN_AREA_MM2:
            continue
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        rows.append(
            {
                "label": identity,
                "area_mm2": area_mm2,
                "row_bounds": [int(r0), int(r1)],
                "column_bounds": [int(c0), int(c1)],
                "coordinate_bounds_m": [
                    [float(raster["xs"][c0]), float(raster["ys"][r0])],
                    [float(raster["xs"][c1]), float(raster["ys"][r1])],
                ],
            }
        )
    rows.sort(key=lambda item: item["area_mm2"], reverse=True)
    return labels, rows


def _semantic_zone_cells(label: str, raster: dict[str, Any], voids: list[dict[str, Any]]) -> list[tuple[str, np.ndarray]]:
    occupied = np.argwhere(raster["mask"])
    if len(voids) < 2:
        raise RuntimeError(f"{label}: expected at least two major voids")
    if label == "fixed":
        lower, upper = sorted(voids[:2], key=lambda item: item["row_bounds"][0])
        lr0, lr1 = lower["row_bounds"]
        lc0, lc1 = lower["column_bounds"]
        ur0, ur1 = upper["row_bounds"]
        uc0, uc1 = upper["column_bounds"]
        definitions = [
            ("lower_bridge", occupied[:, 0] < lr0),
            ("lower_left_leg", (occupied[:, 0] >= lr0) & (occupied[:, 0] <= lr1) & (occupied[:, 1] < lc0)),
            ("lower_right_leg", (occupied[:, 0] >= lr0) & (occupied[:, 0] <= lr1) & (occupied[:, 1] > lc1)),
            ("middle_bridge", (occupied[:, 0] > lr1) & (occupied[:, 0] < ur0)),
            ("upper_left_leg", (occupied[:, 0] >= ur0) & (occupied[:, 0] <= ur1) & (occupied[:, 1] < uc0)),
            ("upper_right_leg", (occupied[:, 0] >= ur0) & (occupied[:, 0] <= ur1) & (occupied[:, 1] > uc1)),
            ("roof_bridge", occupied[:, 0] > ur1),
        ]
    else:
        mouth, window = sorted(voids[:2], key=lambda item: item["column_bounds"][0])
        mr0, mr1 = mouth["row_bounds"]
        _, mc1 = mouth["column_bounds"]
        wr0, wr1 = window["row_bounds"]
        wc0, wc1 = window["column_bounds"]
        definitions = [
            ("proximal_upper_rail", (occupied[:, 1] <= mc1) & (occupied[:, 0] < mr0)),
            ("proximal_lower_rail", (occupied[:, 1] <= mc1) & (occupied[:, 0] > mr1)),
            ("center_bridge", (occupied[:, 1] > mc1) & (occupied[:, 1] < wc0)),
            ("window_upper_rail", (occupied[:, 1] >= wc0) & (occupied[:, 1] <= wc1) & (occupied[:, 0] < wr0)),
            ("window_lower_rail", (occupied[:, 1] >= wc0) & (occupied[:, 1] <= wc1) & (occupied[:, 0] > wr1)),
            ("distal_nose_bridge", occupied[:, 1] > wc1),
        ]
    return [(name, occupied[selector]) for name, selector in definitions if int(np.sum(selector)) >= 3]


def _semantic_polygons(label: str, patch: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from scipy.ndimage import distance_transform_edt, label as component_label

    raster = _raster_patch(patch)
    void_labels, voids = _major_voids(raster)
    named_cells = _semantic_zone_cells(label, raster, voids)
    polygons: list[dict[str, Any]] = []
    union = np.zeros_like(raster["mask"])

    def append_polygon(name: str, cells: np.ndarray) -> None:
        points = np.column_stack([raster["xs"][cells[:, 1]], raster["ys"][cells[:, 0]]])
        poly, poly_mask = _convex_polygon(points, raster["xs"], raster["ys"])
        polygons.append({"name": name, "polygon_2d_m": poly, "cell_count": len(cells)})
        nonlocal union
        union |= poly_mask

    for name, cells in named_cells:
        append_polygon(name, cells)

    missing = raster["mask"] & ~union
    missing_labels, missing_count = component_label(missing)
    raw_distance = distance_transform_edt(~raster["mask"]) * RASTER_STEP_M

    def fixed_extra(name: str, cells: np.ndarray) -> None:
        points = np.column_stack([raster["xs"][cells[:, 1]], raster["ys"][cells[:, 0]]])
        poly, poly_mask = _convex_polygon(points, raster["xs"], raster["ys"])
        ghost = poly_mask & ~raster["mask"]
        max_void_fill_mm2 = max(
            [float(np.sum(poly_mask & (void_labels == item["label"]))) * (RASTER_STEP_M * 1000.0) ** 2 for item in voids]
            or [0.0]
        )
        ghost_mm2 = float(np.sum(ghost)) * (RASTER_STEP_M * 1000.0) ** 2
        if (
            max_void_fill_mm2 <= FIXED_EXTRA_MAJOR_VOID_FILL_LIMIT_MM2
            and ghost_mm2 <= FIXED_EXTRA_GHOST_LIMIT_MM2
        ) or len(cells) < 12:
            append_polygon(name, cells)
            return
        spans = np.ptp(cells, axis=0)
        axis = int(np.argmax(spans))
        split = float(np.median(cells[:, axis]))
        left, right = cells[cells[:, axis] <= split], cells[cells[:, axis] > split]
        if len(left) < 3 or len(right) < 3:
            append_polygon(name, cells)
            return
        fixed_extra(name + "_a", left)
        fixed_extra(name + "_b", right)

    def moving_extra(name: str, cells: np.ndarray) -> None:
        points = np.column_stack([raster["xs"][cells[:, 1]], raster["ys"][cells[:, 0]]])
        poly, poly_mask = _convex_polygon(points, raster["xs"], raster["ys"])
        ghost = poly_mask & ~raster["mask"]
        max_ghost_distance_mm = float(raw_distance[ghost].max() * 1000.0) if np.any(ghost) else 0.0
        if max_ghost_distance_mm <= PATCH_MAX_GHOST_DISTANCE_MM["moving"] or len(cells) < 12:
            append_polygon(name, cells)
            return
        spans = np.ptp(cells, axis=0)
        axis = int(np.argmax(spans))
        split = float(np.median(cells[:, axis]))
        left, right = cells[cells[:, axis] <= split], cells[cells[:, axis] > split]
        if len(left) < 3 or len(right) < 3:
            append_polygon(name, cells)
            return
        moving_extra(name + "_a", left)
        moving_extra(name + "_b", right)

    extra_index = 0
    for identity in range(1, missing_count + 1):
        cells = np.argwhere(missing_labels == identity)
        area_mm2 = len(cells) * (RASTER_STEP_M * 1000.0) ** 2
        if area_mm2 < MISSING_COMPONENT_MIN_AREA_MM2:
            continue
        extra_index += 1
        if label == "fixed":
            fixed_extra(f"fixed_brace_{extra_index:02d}", cells)
        else:
            moving_extra(f"moving_brace_{extra_index:02d}", cells)

    final_union = np.zeros_like(raster["mask"])
    for row in polygons:
        final_union |= _polygon_mask(row["polygon_2d_m"], raster["xs"], raster["ys"])
    ghost = final_union & ~raster["mask"]
    under = raster["mask"] & ~final_union
    occupied_count = int(np.sum(raster["mask"]))
    distance = raw_distance
    if label == "fixed":
        semantic_voids = sorted(voids[:2], key=lambda item: item["row_bounds"][0])
        semantic_roles = ["fixed_primary", "fixed_secondary"]
    else:
        semantic_voids = sorted(voids[:2], key=lambda item: item["column_bounds"][0])
        semantic_roles = ["moving_open_mouth", "moving_internal_window_diagnostic"]
    void_rows = []
    for item, role in zip(semantic_voids, semantic_roles, strict=True):
        target = void_labels == item["label"]
        filled = int(np.sum(final_union & target))
        void_rows.append(
            {
                **item,
                "role": role,
                "candidate_filled_cell_count": filled,
                "candidate_fill_fraction": filled / max(1, int(np.sum(target))),
                "registered_max_fill_fraction": MAJOR_VOID_MAX_FILL[role],
                "pass": filled / max(1, int(np.sum(target))) <= MAJOR_VOID_MAX_FILL[role],
            }
        )
    metrics = {
        "raster_step_mm": RASTER_STEP_M * 1000.0,
        "raw_occupied_area_mm2": occupied_count * (RASTER_STEP_M * 1000.0) ** 2,
        "candidate_occupied_area_mm2": int(np.sum(final_union)) * (RASTER_STEP_M * 1000.0) ** 2,
        "ghost_area_mm2": int(np.sum(ghost)) * (RASTER_STEP_M * 1000.0) ** 2,
        "missing_area_mm2": int(np.sum(under)) * (RASTER_STEP_M * 1000.0) ** 2,
        "ghost_ratio": int(np.sum(ghost)) / max(1, occupied_count),
        "missing_ratio": int(np.sum(under)) / max(1, occupied_count),
        "maximum_ghost_distance_mm": float(distance[ghost].max() * 1000.0) if np.any(ghost) else 0.0,
        "major_voids": void_rows,
        "polygon_count": len(polygons),
        "max_polygon_vertices": max(len(row["polygon_2d_m"]) for row in polygons),
    }
    metrics["checks"] = {
        "ghost_ratio_within_gate": metrics["ghost_ratio"] <= PATCH_MAX_GHOST_RATIO[label],
        "missing_ratio_within_gate": metrics["missing_ratio"] <= PATCH_MAX_MISSING_RATIO[label],
        "major_voids_within_gates": len(void_rows) == 2 and all(row["pass"] for row in void_rows),
        "exactly_two_registered_major_voids": len(voids) == 2 and len(void_rows) == 2,
        "maximum_ghost_distance_within_gate": metrics["maximum_ghost_distance_mm"]
        <= PATCH_MAX_GHOST_DISTANCE_MM[label],
        "polygon_vertices_le_32": metrics["max_polygon_vertices"] <= 32,
    }
    metrics["pass"] = all(metrics["checks"].values())
    private = {"raster": raster, "void_labels": void_labels, "candidate_mask": final_union}
    return polygons, metrics, private


def _make_prism_part(
    *, body: str, name: str, role: str, polygon_2d_m: np.ndarray, axis: int, low_m: float, high_m: float
) -> dict[str, Any]:
    polygon = np.asarray(polygon_2d_m, dtype=np.float64)
    if high_m <= low_m:
        raise RuntimeError(f"invalid prism thickness for {name}")
    keep = [value for value in range(3) if value != axis]
    basis = np.eye(3, dtype=np.float64)
    orientation = float(np.dot(np.cross(basis[keep[0]], basis[keep[1]]), basis[axis]))
    if orientation < 0.0:
        # The generic cap winding assumes 2-D CCW maps to +extrusion-axis.
        # x-z coordinates map to -y, so axis=1 must reverse its 2-D order.
        polygon = polygon[::-1].copy()
    n = len(polygon)
    if n < 3 or n > 32:
        raise RuntimeError(f"invalid prism polygon size {n} for {name}")
    vertices = np.zeros((2 * n, 3), dtype=np.float64)
    vertices[:n, axis] = low_m
    vertices[n:, axis] = high_m
    vertices[:n, keep] = polygon
    vertices[n:, keep] = polygon
    triangles: list[list[int]] = []
    for index in range(1, n - 1):
        triangles.append([0, index + 1, index])
        triangles.append([n, n + index, n + index + 1])
    for index in range(n):
        nxt = (index + 1) % n
        triangles.extend([[index, nxt, n + nxt], [index, n + nxt, n + index]])
    tri = np.asarray(triangles, dtype=np.int64)
    area = 0.5 * abs(float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))))
    payload = _sha_bytes(_blob(vertices, "<f8") + _blob(tri, "<i8"))
    return {
        "body": body,
        "name": name,
        "role": role,
        "source": "D372_manual_semantic_compound",
        "vertices": vertices,
        "triangles": tri,
        "vertex_count": len(vertices),
        "polygon_count": n + 2,
        # A triangular prism has quad side faces, so n=3 still means a
        # four-vertex polygon maximum.  The stored topology metric must describe
        # the convex polyhedron rather than only its two cap polygons.
        "max_vertices_per_polygon": max(n, 4),
        "triangle_count": len(tri),
        "topology_volume_m3": area * (high_m - low_m),
        "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "payload_sha256": payload,
    }


def _make_box_part(*, body: str, name: str, role: str, lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
    polygon = np.asarray([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]], dtype=np.float64)
    return _make_prism_part(body=body, name=name, role=role, polygon_2d_m=polygon, axis=2, low_m=float(lo[2]), high_m=float(hi[2]))


def _make_source_hull_part(*, body: str, name: str, role: str, source_vertices: np.ndarray, d368: Any) -> dict[str, Any]:
    """Build one deterministic 3-D convex proxy from a semantic source cluster.

    Qhull triangulates planar faces.  The PhysX GPU constraint is stated in
    convex vertices/polygons, not display triangles, so coplanar Qhull facets
    are grouped by their outward plane equation before recording polygon counts.
    """
    from scipy.spatial import ConvexHull

    source = np.unique(np.asarray(source_vertices, dtype=np.float64), axis=0)
    if len(source) < 4 or np.linalg.matrix_rank(source - source.mean(axis=0)) < 3:
        raise RuntimeError(f"{name}: source cluster is not three-dimensional")
    hull = ConvexHull(source)
    original_vertex_ids = np.asarray(hull.vertices, dtype=np.int64)
    remap = {int(old): index for index, old in enumerate(original_vertex_ids)}
    vertices = source[original_vertex_ids]
    triangles = np.asarray(
        [[remap[int(value)] for value in face] for face in hull.simplices],
        dtype=np.int64,
    )
    mesh = d368._trimesh(vertices, triangles)
    mesh.fix_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces, dtype=np.int64)
    plane_keys = np.round(np.asarray(hull.equations, dtype=np.float64), decimals=8)
    plane_groups: dict[tuple[float, ...], set[int]] = {}
    for face, key in zip(hull.simplices, map(tuple, plane_keys), strict=True):
        plane_groups.setdefault(key, set()).update(map(int, face))
    payload = _sha_bytes(_blob(vertices, "<f8") + _blob(triangles, "<i8"))
    return {
        "body": body,
        "name": name,
        "role": role,
        "source": "D372_semantic_source_3d_convex_hull",
        "vertices": vertices,
        "triangles": triangles,
        "vertex_count": len(vertices),
        "polygon_count": len(plane_groups),
        "max_vertices_per_polygon": max(map(len, plane_groups.values())),
        "triangle_count": len(triangles),
        "topology_volume_m3": float(mesh.volume),
        "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "payload_sha256": payload,
        "coplanar_plane_grouping_decimals": 8,
    }


def _make_source_profile_part(
    *, body: str, name: str, role: str, source_vertices: np.ndarray, axis: int
) -> dict[str, Any]:
    from scipy.spatial import ConvexHull

    source = np.unique(np.asarray(source_vertices, dtype=np.float64), axis=0)
    keep = [value for value in range(3) if value != axis]
    profile_points = np.unique(source[:, keep], axis=0)
    if len(profile_points) < 3 or np.linalg.matrix_rank(profile_points - profile_points.mean(axis=0)) < 2:
        raise RuntimeError(f"{name}: source profile is not two-dimensional")
    hull = ConvexHull(profile_points)
    return _make_prism_part(
        body=body,
        name=name,
        role=role,
        polygon_2d_m=profile_points[hull.vertices],
        axis=axis,
        low_m=float(source[:, axis].min()),
        high_m=float(source[:, axis].max()),
    )


def _raw_components(raw_body: dict[str, Any], d368: Any) -> list[dict[str, Any]]:
    triangles = np.asarray(raw_body["triangles"], dtype=np.int64)
    unseen = set(range(len(triangles)))
    rows = []
    while unseen:
        seed = min(unseen)
        component = d368._vertex_connected_component(raw_body["vertices_m"], triangles, seed, reverse=False)
        face_ids = set(int(value) for value in component["face_ids"])
        unseen -= face_ids
        rows.append(component)
    rows.sort(key=lambda item: len(item["face_ids"]), reverse=True)
    return rows


def _bounds_of(components: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.vstack([np.asarray(item["vertices"], dtype=np.float64) for item in components])
    return vertices.min(axis=0), vertices.max(axis=0)


def _build_candidate(raw: dict[str, dict[str, Any]], patches: dict[str, Any], d368: Any) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    fixed_polygons, fixed_metrics, fixed_private = _semantic_polygons("fixed", patches["fixed"])
    moving_polygons, moving_metrics, moving_private = _semantic_polygons("moving", patches["inner"])
    moving_outer_raster = _raster_patch(patches["outer"])
    moving_projection_checks = {
        "axis_exact": moving_private["raster"]["axis"] == moving_outer_raster["axis"] == 1,
        "xs_bit_exact": np.array_equal(moving_private["raster"]["xs"], moving_outer_raster["xs"]),
        "ys_bit_exact": np.array_equal(moving_private["raster"]["ys"], moving_outer_raster["ys"]),
        "raw_inner_outer_footprint_mask_bit_exact": np.array_equal(moving_private["raster"]["mask"], moving_outer_raster["mask"]),
    }
    moving_projection = {
        "inner_mask_sha256": _sha_bytes(_blob(moving_private["raster"]["mask"], "|b1")),
        "outer_mask_sha256": _sha_bytes(_blob(moving_outer_raster["mask"], "|b1")),
        "candidate_mask_sha256": _sha_bytes(_blob(moving_private["candidate_mask"], "|b1")),
        "outer_gate_inherits_inner_metrics_only_if_bit_exact": True,
        "checks": moving_projection_checks,
        "pass": all(moving_projection_checks.values()) and moving_metrics["pass"],
    }
    components = _raw_components(raw["link5"], d368)
    by_count = {len(item["face_ids"]): item for item in components}
    if set(by_count) != {7250, 5276, 1054, 288, 192, 32}:
        raise RuntimeError(f"unexpected link5 components: {sorted(by_count)}")

    housing_source = np.vstack([by_count[5276]["vertices"], by_count[288]["vertices"], by_count[32]["vertices"]])
    pivot_source = np.vstack([by_count[1054]["vertices"], by_count[192]["vertices"]])
    housing_hi = housing_source.max(axis=0)
    pivot_lo, pivot_hi = _bounds_of([by_count[1054], by_count[192]])
    broad_vertices = np.asarray(by_count[7250]["vertices"], dtype=np.float64)
    neck_selection = broad_vertices[
        (broad_vertices[:, 2] >= housing_hi[2] - 1.0e-9)
        & (broad_vertices[:, 2] <= float(patches["fixed"]["vertices"].min(axis=0)[2]) + 1.0e-9)
    ]
    if len(neck_selection) < 4:
        raise RuntimeError("link5 neck selection empty")
    neck_lo, neck_hi = neck_selection.min(axis=0), neck_selection.max(axis=0)

    parts: dict[str, list[dict[str, Any]]] = {"link5": [], "gripper_link": []}
    structural_link5 = [
        _make_box_part(body="link5", name="lower_housing_box", role="structural_body", lo=housing_source.min(axis=0), hi=housing_source.max(axis=0)),
        _make_box_part(body="link5", name="neck_connector_box", role="connector_support", lo=neck_lo, hi=neck_hi),
        _make_box_part(body="link5", name="pivot_support_box", role="connector_support", lo=pivot_lo, hi=pivot_hi),
    ]
    for part, source_vertices in zip(structural_link5, (housing_source, neck_selection, pivot_source), strict=True):
        part["_source_vertices"] = np.unique(np.asarray(source_vertices, dtype=np.float64), axis=0)
    parts["link5"].extend(structural_link5)
    fixed_plane = float(patches["fixed"]["plane_m"])
    for index, row in enumerate(fixed_polygons):
        parts["link5"].append(
            _make_prism_part(
                body="link5",
                name=f"fixed_jaw_{index:02d}_{row['name']}",
                role="fixed_jaw",
                polygon_2d_m=row["polygon_2d_m"],
                axis=0,
                low_m=fixed_plane - JAW_THICKNESS_TARGET_M,
                high_m=fixed_plane,
            )
        )

    # The fixed contact skin is only the front 1.5 mm plate.  Preserve the two
    # thin side backbones and the lower pivot hub explicitly, without joining
    # the A-frame openings into one convex envelope.
    broad_unique = np.unique(broad_vertices, axis=0)
    fixed_backbone_left = broad_unique[
        (broad_unique[:, 1] >= -0.0177465)
        & (broad_unique[:, 1] <= -0.0162360)
        & (broad_unique[:, 0] >= -0.023250)
        & (broad_unique[:, 0] <= -0.011525)
        & (broad_unique[:, 2] >= 0.016085)
        & (broad_unique[:, 2] <= 0.107613)
    ]
    fixed_backbone_right = broad_unique[
        (broad_unique[:, 1] >= 0.015211)
        & (broad_unique[:, 1] <= 0.016722)
        & (broad_unique[:, 0] >= -0.023250)
        & (broad_unique[:, 0] <= -0.011525)
        & (broad_unique[:, 2] >= 0.016085)
        & (broad_unique[:, 2] <= 0.107613)
    ]
    lower_pivot_hub = broad_unique[
        (broad_unique[:, 0] >= -0.0311)
        & (broad_unique[:, 0] <= -0.0100)
        & (broad_unique[:, 1] >= -0.0096)
        & (broad_unique[:, 1] <= 0.0086)
        & (broad_unique[:, 2] >= 0.0790)
        & (broad_unique[:, 2] <= 0.0878)
    ]
    if min(map(len, (fixed_backbone_left, fixed_backbone_right, lower_pivot_hub))) < 4:
        raise RuntimeError("link5 fixed-jaw backbone semantic selection empty")
    fixed_backbone_parts = [
        _make_source_hull_part(
            body="link5", name="fixed_backbone_left", role="fixed_jaw_backbone",
            source_vertices=fixed_backbone_left, d368=d368,
        ),
        _make_source_hull_part(
            body="link5", name="fixed_backbone_right", role="fixed_jaw_backbone",
            source_vertices=fixed_backbone_right, d368=d368,
        ),
        _make_source_profile_part(
            body="link5", name="lower_pivot_hub", role="connector_support",
            source_vertices=lower_pivot_hub, axis=0,
        ),
    ]
    for part, source_vertices in zip(
        fixed_backbone_parts,
        (fixed_backbone_left, fixed_backbone_right, lower_pivot_hub),
        strict=True,
    ):
        part["_source_vertices"] = np.unique(np.asarray(source_vertices, dtype=np.float64), axis=0)
    parts["link5"].extend(fixed_backbone_parts)

    mouth = sorted(moving_metrics["major_voids"], key=lambda item: item["coordinate_bounds_m"][0][0])[0]
    mouth_z_min = float(mouth["coordinate_bounds_m"][0][1])
    mouth_z_max = float(mouth["coordinate_bounds_m"][1][1])
    gripper_vertices = np.asarray(raw["gripper_link"]["vertices_m"], dtype=np.float64)
    patch_x_min = float(patches["inner"]["vertices"].min(axis=0)[0])
    proximal = gripper_vertices[gripper_vertices[:, 0] <= patch_x_min + 1.0e-9]
    upper = proximal[proximal[:, 2] >= mouth_z_max - 1.0e-9]
    lower = proximal[proximal[:, 2] <= mouth_z_min + 1.0e-9]
    if min(len(upper), len(lower)) < 4:
        raise RuntimeError("moving proximal arm selection empty")
    upper_cut = float(np.median(upper[:, 1])); lower_cut = float(np.median(lower[:, 1]))
    upper_sources = [upper[upper[:, 1] <= upper_cut], upper[upper[:, 1] > upper_cut]]
    lower_sources = [lower[lower[:, 1] <= lower_cut], lower[lower[:, 1] > lower_cut]]
    if min(map(len, [*upper_sources, *lower_sources])) < 4:
        raise RuntimeError("moving support semantic split empty")
    structural_gripper = [
        _make_source_hull_part(body="gripper_link", name="proximal_upper_arm_hull_a", role="moving_support", source_vertices=upper_sources[0], d368=d368),
        _make_source_hull_part(body="gripper_link", name="proximal_upper_arm_hull_b", role="moving_support", source_vertices=upper_sources[1], d368=d368),
        _make_source_hull_part(body="gripper_link", name="proximal_lower_arm_hull_a", role="moving_support", source_vertices=lower_sources[0], d368=d368),
        _make_source_hull_part(body="gripper_link", name="proximal_lower_arm_hull_b", role="moving_support", source_vertices=lower_sources[1], d368=d368),
    ]
    for part, source_vertices in zip(structural_gripper, (*upper_sources, *lower_sources), strict=True):
        part["_source_vertices"] = np.unique(np.asarray(source_vertices, dtype=np.float64), axis=0)
    parts["gripper_link"].extend(structural_gripper)
    inner_plane = float(patches["inner"]["plane_m"])
    outer_plane = float(patches["outer"]["plane_m"])
    for index, row in enumerate(moving_polygons):
        parts["gripper_link"].append(
            _make_prism_part(
                body="gripper_link",
                name=f"moving_jaw_{index:02d}_{row['name']}",
                role="moving_jaw",
                polygon_2d_m=row["polygon_2d_m"],
                axis=1,
                low_m=inner_plane,
                high_m=outer_plane,
            )
        )

    # The D354 inner/outer patches describe the 1.5 mm vertical contact skin.
    # The remaining distal structure is not a thick copy of that skin: the raw
    # mesh contains one thin upper rail and one thin lower rail extending behind
    # the outer plane.  Build one actual 3-D source hull for each rail.  Keeping
    # the rails separate preserves the mouth and internal window in X-Z instead
    # of filling them with an axis-extruded profile.
    gripper_unique = np.unique(gripper_vertices, axis=0)
    moving_top_z = float(np.max(patches["inner"]["vertices"][:, 2]))
    moving_bottom_z = float(np.min(patches["inner"]["vertices"][:, 2]))
    moving_upper_backbone = gripper_unique[
        (gripper_unique[:, 0] >= patch_x_min - 1.0e-9)
        & (gripper_unique[:, 1] >= outer_plane - 1.0e-9)
        & (gripper_unique[:, 2] >= moving_top_z - 1.0e-9)
    ]
    moving_lower_backbone = gripper_unique[
        (gripper_unique[:, 0] >= patch_x_min - 1.0e-9)
        & (gripper_unique[:, 1] >= outer_plane - 1.0e-9)
        & (gripper_unique[:, 2] <= moving_bottom_z + 1.0e-9)
    ]
    if min(map(len, (moving_upper_backbone, moving_lower_backbone))) < 4:
        raise RuntimeError("moving-jaw backbone semantic selection empty")
    moving_backbone_parts = [
        _make_source_hull_part(
            body="gripper_link", name="moving_upper_backbone", role="moving_jaw_backbone",
            source_vertices=moving_upper_backbone, d368=d368,
        ),
        _make_source_hull_part(
            body="gripper_link", name="moving_lower_backbone", role="moving_jaw_backbone",
            source_vertices=moving_lower_backbone, d368=d368,
        ),
    ]
    for part, source_vertices in zip(
        moving_backbone_parts,
        (moving_upper_backbone, moving_lower_backbone),
        strict=True,
    ):
        part["_source_vertices"] = np.unique(np.asarray(source_vertices, dtype=np.float64), axis=0)
    parts["gripper_link"].extend(moving_backbone_parts)

    for body in BODY_NAMES:
        for part in parts[body]:
            part["_mesh"] = d368._trimesh(part["vertices"], part["triangles"])
    fixed_face_ids = set(range(1740, 2007))
    link_groups = {
        "lower_housing_source": set().union(*(set(map(int, by_count[count]["face_ids"])) for count in (5276, 288, 32))),
        "pivot_support_source": set().union(*(set(map(int, by_count[count]["face_ids"])) for count in (1054, 192))),
        "fixed_jaw_contact_source": fixed_face_ids,
        "broad_connector_remainder_source": set(map(int, by_count[7250]["face_ids"])) - fixed_face_ids,
    }
    moving_inner = set(map(int, d368.INNER_FACE_IDS))
    moving_outer = set(map(int, d368.OUTER_FACE_IDS))
    moving_groups = {
        "moving_inner_contact_source": moving_inner,
        "moving_outer_contact_source": moving_outer,
        "moving_structural_remainder_source": set(range(len(raw["gripper_link"]["triangles"]))) - moving_inner - moving_outer,
    }

    def partition_report(groups: dict[str, set[int]], total: int) -> dict[str, Any]:
        names = list(groups)
        overlaps = sum(len(groups[names[i]] & groups[names[j]]) for i in range(len(names)) for j in range(i + 1, len(names)))
        union = set().union(*groups.values())
        return {
            "counts": {name: len(values) for name, values in groups.items()},
            "overlap_count": overlaps,
            "missing_count": total - len(union),
            "out_of_range_count": len({value for value in union if value < 0 or value >= total}),
            "union_count": len(union),
            "total_face_count": total,
            "pass": overlaps == 0 and union == set(range(total)),
        }

    def source_record(values: np.ndarray, criteria: str) -> dict[str, Any]:
        source = np.unique(np.asarray(values, dtype=np.float64), axis=0)
        return {
            "criteria": criteria,
            "vertex_count": len(source),
            "bounds_m": [source.min(axis=0).tolist(), source.max(axis=0).tolist()],
            "vertices_sha256": _sha_bytes(_blob(source, "<f8")),
        }

    construction = {
        "component_face_counts": [len(item["face_ids"]) for item in components],
        "structural_bounds_m": {
            **{part["name"]: part["bounds_m"] for part in structural_link5},
            **{part["name"]: part["bounds_m"] for part in fixed_backbone_parts},
            **{part["name"]: part["bounds_m"] for part in structural_gripper},
            **{part["name"]: part["bounds_m"] for part in moving_backbone_parts},
        },
        "semantic_source_selections": {
            "lower_housing": source_record(housing_source, "three disconnected lower-housing components: face counts 5276, 288, 32"),
            "neck_connector": source_record(neck_selection, "broad-shell vertices from lower-housing top through fixed-patch z minimum"),
            "pivot_support": source_record(pivot_source, "two disconnected pivot components: face counts 1054, 192"),
            "fixed_backbone_left": source_record(fixed_backbone_left, "thin raw side rail outside fixed contact patch negative-y edge"),
            "fixed_backbone_right": source_record(fixed_backbone_right, "thin raw side rail outside fixed contact patch positive-y edge"),
            "lower_pivot_hub": source_record(lower_pivot_hub, "localized broad-shell pivot hub; y-z convex profile extruded along x"),
            "proximal_upper_support_a": source_record(upper_sources[0], "proximal upper support, negative/local-middle y semantic half"),
            "proximal_upper_support_b": source_record(upper_sources[1], "proximal upper support, positive-y semantic half"),
            "proximal_lower_support_a": source_record(lower_sources[0], "proximal lower support, negative/local-middle y semantic half"),
            "proximal_lower_support_b": source_record(lower_sources[1], "proximal lower support, positive-y semantic half"),
            "moving_upper_backbone": source_record(moving_upper_backbone, "raw vertices at/behind D354 outer plane, distal of the contact-patch x minimum, and at/above the inner-patch top edge"),
            "moving_lower_backbone": source_record(moving_lower_backbone, "raw vertices at/behind D354 outer plane, distal of the contact-patch x minimum, and at/below the inner-patch bottom edge"),
        },
        "fixed_patch": fixed_metrics,
        "moving_patch": moving_metrics,
        "semantic_zone_manifest": {
            "fixed_base_required": sorted(REQUIRED_FIXED_BASE_ZONES),
            "fixed_observed": [row["name"] for row in fixed_polygons],
            "fixed_base_present": REQUIRED_FIXED_BASE_ZONES.issubset({row["name"] for row in fixed_polygons}),
            "moving_base_required": sorted(REQUIRED_MOVING_BASE_ZONES),
            "moving_observed": [row["name"] for row in moving_polygons],
            "moving_base_present": REQUIRED_MOVING_BASE_ZONES.issubset({row["name"] for row in moving_polygons}),
        },
        "moving_inner_outer_projection": moving_projection,
        "raw_semantic_face_classification_partition": {
            "link5": partition_report(link_groups, len(raw["link5"]["triangles"])),
            "gripper_link": partition_report(moving_groups, len(raw["gripper_link"]["triangles"])),
        },
        "jaw_plane_and_thickness": {
            "fixed_contact_plane_m": fixed_plane,
            "fixed_thickness_m": JAW_THICKNESS_TARGET_M,
            "fixed_all_exact": all(
                abs(part["bounds_m"][1][0] - fixed_plane) <= 1.0e-12
                and abs(
                    (part["bounds_m"][1][0] - part["bounds_m"][0][0])
                    - JAW_THICKNESS_TARGET_M
                ) <= 1.0e-12
                for part in parts["link5"] if part["role"] == "fixed_jaw"
            ),
            "moving_inner_plane_m": inner_plane,
            "moving_outer_plane_m": outer_plane,
            "moving_thickness_m": outer_plane - inner_plane,
            "moving_all_exact": all(
                abs(part["bounds_m"][0][1] - inner_plane) <= 1.0e-12
                and abs(part["bounds_m"][1][1] - outer_plane) <= 1.0e-12
                for part in parts["gripper_link"] if part["role"] == "moving_jaw"
            ),
        },
    }
    private = {"fixed": fixed_private, "moving": moving_private}
    return parts, construction, private


def _public_part(part: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in part.items() if not key.startswith("_")}


def _inventory(parts: list[dict[str, Any]]) -> dict[str, Any]:
    topology = []
    for part in parts:
        mesh = part["_mesh"]
        topology.append(
            {
                "name": part["name"],
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "convex": bool(mesh.is_convex),
                "minimum_triangle_double_area_m2": float(
                    np.min(
                        np.linalg.norm(
                            np.cross(
                                part["vertices"][part["triangles"]][:, 1] - part["vertices"][part["triangles"]][:, 0],
                                part["vertices"][part["triangles"]][:, 2] - part["vertices"][part["triangles"]][:, 0],
                            ),
                            axis=1,
                        )
                    )
                ),
                "signed_volume_m3": float(mesh.volume),
                "registered_volume_m3": float(part["topology_volume_m3"]),
                "volume_delta_m3": abs(float(mesh.volume) - float(part["topology_volume_m3"])),
            }
        )
    checks = {
        "nonempty": bool(parts),
        "vertices_le_64": bool(parts) and all(part["vertex_count"] <= 64 for part in parts),
        "polygons_le_64": bool(parts) and all(part["polygon_count"] <= 64 for part in parts),
        "vertices_per_polygon_le_32": bool(parts) and all(part["max_vertices_per_polygon"] <= 32 for part in parts),
        "finite": bool(parts) and all(np.isfinite(part["vertices"]).all() for part in parts),
        "all_watertight": bool(topology) and all(row["watertight"] for row in topology),
        "all_winding_consistent": bool(topology) and all(row["winding_consistent"] for row in topology),
        "all_convex": bool(topology) and all(row["convex"] for row in topology),
        "all_triangles_nondegenerate": bool(topology) and all(row["minimum_triangle_double_area_m2"] > 0.0 for row in topology),
        "all_positive_signed_volume": bool(topology) and all(row["signed_volume_m3"] > 0.0 for row in topology),
        "signed_volume_matches_registered": bool(topology)
        and all(row["volume_delta_m3"] <= max(1.0e-15, row["registered_volume_m3"] * 1.0e-9) for row in topology),
    }
    return {
        "part_count": len(parts),
        "roles": {role: sum(part["role"] == role for part in parts) for role in sorted({part["role"] for part in parts})},
        "vertex_count_sum": sum(part["vertex_count"] for part in parts),
        "polygon_count_sum": sum(part["polygon_count"] for part in parts),
        "triangle_count_sum": sum(part["triangle_count"] for part in parts),
        "whole_part_volume_sum_m3": sum(part["topology_volume_m3"] for part in parts),
        "volume_semantics": "overlap-prone part sum; not mass or unique material volume",
        "topology": topology,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _quat_wxyz_to_rot(quaternion: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _query_parts(parts: list[dict[str, Any]], body_pose: dict[str, Any], cylinder_pose: dict[str, Any], d371: Any) -> dict[str, Any]:
    import hppfcl

    body_tf = hppfcl.Transform3f(_quat_wxyz_to_rot(body_pose["quat_wxyz"]), np.asarray(body_pose["pos_m"], dtype=np.float64))
    cylinder_tf = hppfcl.Transform3f(_quat_wxyz_to_rot(cylinder_pose["object_quat_wxyz"]), np.asarray(cylinder_pose["object_pos_w_m"], dtype=np.float64))
    cylinder = hppfcl.Cylinder(0.017, 0.090)
    rows = []
    for part in parts:
        query = d371._fcl_part_query(hppfcl, d371._build_bvh(hppfcl, part), body_tf, cylinder, cylinder_tf)
        rows.append({"part": part["name"], **query})
    collisions = [row for row in rows if row["is_collision"]]
    selected = min(collisions or rows, key=lambda row: row["exact_signed_distance_mm"])
    return {"queries": rows, "selected": selected, "collision_parts": [row["part"] for row in collisions]}


def _clearance(candidate: dict[str, list[dict[str, Any]]], current: dict[str, list[dict[str, Any]]], d371: Any) -> dict[str, Any]:
    frozen = _read_json(D349_MEASUREMENT)
    pose = frozen["distance_gate"]["authoritative_pose_streams"]["raw_first"]
    expected = frozen["distance_gate"]["per_body"]
    output = {}
    query_count = 0
    for body in BODY_NAMES:
        p34 = _query_parts(candidate[body], pose["body_poses_w"][body], pose, d371)
        a64 = _query_parts(current[body], pose["body_poses_w"][body], pose, d371)
        query_count += len(p34["queries"]) + len(a64["queries"])
        value = float(p34["selected"]["exact_signed_distance_mm"])
        raw_value = float(expected[body]["raw_exact_signed_distance_mm"])
        checks = {
            "no_collision": not p34["collision_parts"],
            "clearance_ge_0p1mm": value >= OPEN_CLEAR_GATE_MM,
            "raw_delta_le_0p5mm": abs(value - raw_value) <= OPEN_RAW_DELTA_GATE_MM,
            "A64_reproduces_D349_live_within_1e6mm": abs(
                float(a64["selected"]["exact_signed_distance_mm"])
                - float(expected[body]["live_topology_exact_signed_distance_mm"])
            ) <= 1.0e-6,
        }
        output[body] = {
            "P34_exact_signed_distance_mm": value,
            "D349_raw_reference_mm": raw_value,
            "absolute_delta_from_raw_mm": abs(value - raw_value),
            "P34_selected_part": p34["selected"]["part"],
            "P34_collision_parts": p34["collision_parts"],
            "A64_exact_signed_distance_mm": float(a64["selected"]["exact_signed_distance_mm"]),
            "checks": checks,
            "pass": all(checks.values()),
            "P34_selected_witness": p34["selected"],
        }
    return {
        "semantics": "immutable D349 pose, offline hppfcl surface query; no q5, physics, or live contact",
        "offline_part_query_count": query_count,
        "bodies": output,
        "pass": all(row["pass"] for row in output.values()),
    }


def _seed_metrics(candidate: dict[str, list[dict[str, Any]]], d368: Any) -> dict[str, Any]:
    seeds = {
        "fixed": ("link5", np.asarray(_read_json(D350_BINDING)["seed_local_m"], dtype=np.float64), "fixed_jaw"),
        "moving": ("gripper_link", np.asarray(_read_json(D354_BINDING)["seed_local_m"], dtype=np.float64), "moving_jaw"),
    }
    output = {}
    for label, (body, seed, role) in seeds.items():
        selected_parts = [part for part in candidate[body] if part["role"] == role]
        distances = []
        for part in selected_parts:
            value = float(d368._nearest(part["_mesh"], seed.reshape(1, 3))[0][0] * 1000.0)
            distances.append((value, part["name"]))
        distance, name = min(distances)
        output[label] = {
            "body": body,
            "seed_local_m": seed.tolist(),
            "nearest_part": name,
            "distance_mm": distance,
            "registered_max_mm": RASTER_STEP_M * 1000.0,
            "pass": distance <= RASTER_STEP_M * 1000.0,
        }
    return {"rows": output, "pass": all(row["pass"] for row in output.values())}


def _representation_metrics(
    candidate: dict[str, list[dict[str, Any]]], raw: dict[str, dict[str, Any]], patches: dict[str, Any], d368: Any, d371: Any
) -> dict[str, Any]:
    source_rows: dict[str, Any] = {}
    candidate_surface_rows: dict[str, Any] = {}
    for body in BODY_NAMES:
        raw_mesh = d368._trimesh(raw[body]["vertices_m"], raw[body]["triangles"])
        for part in candidate[body]:
            triangle_ids = np.asarray(part["triangles"], dtype=np.int64)
            edges = np.unique(
                np.sort(
                    np.vstack(
                        [triangle_ids[:, [0, 1]], triangle_ids[:, [1, 2]], triangle_ids[:, [2, 0]]]
                    ),
                    axis=1,
                ),
                axis=0,
            )
            surface_samples = np.vstack(
                [
                    np.asarray(part["vertices"], dtype=np.float64),
                    np.mean(part["vertices"][edges], axis=1),
                    np.mean(part["vertices"][triangle_ids], axis=1),
                ]
            )
            surface_distances_mm = d368._nearest(raw_mesh, surface_samples)[0] * 1000.0
            candidate_surface_rows[f"{body}/{part['name']}"] = {
                "body": body,
                "name": part["name"],
                "role": part["role"],
                "sampling": "vertices plus all unique edge midpoints plus all triangle centroids",
                "sample_count": len(surface_samples),
                "candidate_surface_to_raw_p95_mm": float(np.percentile(surface_distances_mm, 95.0)),
                "candidate_surface_to_raw_max_mm": float(np.max(surface_distances_mm)),
                "diagnostic_only": True,
                "reason_not_a_gate": "a valid simplified proxy may place a collider boundary inside raw solid material",
            }
            if "_source_vertices" not in part:
                continue
            source = np.asarray(part["_source_vertices"], dtype=np.float64)
            contained = d371._convex_contains(source, part)
            source_rows[f"{body}/{part['name']}"] = {
                "body": body,
                "name": part["name"],
                "role": part["role"],
                "source_vertex_count": len(source),
                "source_vertices_sha256": _sha_bytes(_blob(source, "<f8")),
                "source_vertices_contained_count": int(np.sum(contained)),
                "all_source_vertices_contained": bool(np.all(contained)),
                "pass": bool(np.all(contained)),
            }
    moving_patch_x_min = float(np.min(patches["inner"]["vertices"][:, 0]))
    access_checks = {
        "exactly_one_lower_housing_box": sum(
            part["name"] == "lower_housing_box" for part in candidate["link5"]
        ) == 1,
        "moving_support_stays_proximal_to_contact_patch": all(
            float(part["bounds_m"][1][0]) <= moving_patch_x_min + 1.0e-9
            for part in candidate["gripper_link"] if part["role"] == "moving_support"
        ),
        "fixed_backbones_are_two_separate_parts": sum(
            part["role"] == "fixed_jaw_backbone" for part in candidate["link5"]
        ) == 2,
        "moving_backbones_are_upper_lower_separate": {
            part["name"] for part in candidate["gripper_link"] if part["role"] == "moving_jaw_backbone"
        } == {"moving_upper_backbone", "moving_lower_backbone"},
        "no_structural_part_is_full_link_AABB": all(
            not (
                np.allclose(part["bounds_m"][0], raw[body]["vertices_m"].min(axis=0), atol=1.0e-12)
                and np.allclose(part["bounds_m"][1], raw[body]["vertices_m"].max(axis=0), atol=1.0e-12)
            )
            for body in BODY_NAMES
            for part in candidate[body]
            if part["role"] not in {"fixed_jaw", "moving_jaw"}
        ),
    }
    whole_body = {}
    for body in BODY_NAMES:
        raw_vertices = np.unique(np.asarray(raw[body]["vertices_m"], dtype=np.float64), axis=0)
        triangle_rows = raw[body]["vertices_m"][raw[body]["triangles"][::16]]
        samples = np.vstack([raw_vertices, np.mean(triangle_rows, axis=1)])
        contained = d371._union_contains(samples, candidate[body])
        surface_distance_mm = np.min(
            np.vstack([d368._nearest(part["_mesh"], samples)[0] for part in candidate[body]]),
            axis=0,
        ) * 1000.0
        outside_distance_mm = np.where(contained, 0.0, surface_distance_mm)
        checks = {
            "outside_distance_p95_le_registered": float(np.percentile(outside_distance_mm, 95.0))
            <= RAW_TO_CANDIDATE_P95_MAX_MM,
            "outside_distance_max_le_registered": float(np.max(outside_distance_mm))
            <= RAW_TO_CANDIDATE_MAX_MM,
        }
        whole_body[body] = {
            "sampling": "unique raw vertices plus every 16th raw triangle centroid",
            "sample_count": len(samples),
            "contained_count": int(np.sum(contained)),
            "containment_fraction": float(np.mean(contained)),
            "containment_authority": "diagnostic only; point-count fraction depends on tessellation density",
            "raw_sample_outside_candidate_p95_mm": float(np.percentile(outside_distance_mm, 95.0)),
            "raw_sample_outside_candidate_max_mm": float(np.max(outside_distance_mm)),
            "registered_p95_max_mm": RAW_TO_CANDIDATE_P95_MAX_MM,
            "registered_absolute_max_mm": RAW_TO_CANDIDATE_MAX_MM,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {
        "semantic_source_rows": source_rows,
        "candidate_surface_to_raw_diagnostic": candidate_surface_rows,
        "access_checks": access_checks,
        "whole_raw_surface_representation": whole_body,
        "pass": bool(source_rows)
        and all(row["pass"] for row in source_rows.values())
        and all(access_checks.values())
        and all(row["pass"] for row in whole_body.values()),
    }


def _jaw_contact_layer_void_diagnostic(
    candidate: dict[str, list[dict[str, Any]]], construction: dict[str, Any]
) -> dict[str, Any]:
    """Diagnose the 2-D opening of contact skins and adjacent backbones only.

    Connector/support parts can sit behind the contact plane and overlap a
    frontal projection without blocking the cylinder in 3-D.  They are
    intentionally excluded here; D349 clearance and the D362 stored-pose replay
    are the task-local 3-D corridor checks.  This diagnostic must therefore not
    be described as a complete-candidate or through-depth void proof.
    """
    from scipy.spatial import ConvexHull

    output = {}
    for label, body in (("fixed", "link5"), ("moving", "gripper_link")):
        private = construction["_private"][label]
        raster = private["raster"]
        union = np.zeros_like(raster["mask"])
        part_rows = []
        jaw_roles = {"fixed_jaw", "fixed_jaw_backbone"} if label == "fixed" else {"moving_jaw", "moving_jaw_backbone"}
        for part in candidate[body]:
            if part["role"] not in jaw_roles:
                continue
            projected = np.unique(np.asarray(part["vertices"], dtype=np.float64)[:, raster["keep"]], axis=0)
            if len(projected) < 3 or np.linalg.matrix_rank(projected - projected.mean(axis=0)) < 2:
                continue
            hull = ConvexHull(projected)
            polygon = projected[hull.vertices]
            union |= _polygon_mask(polygon, raster["xs"], raster["ys"])
            part_rows.append({"name": part["name"], "role": part["role"], "projected_polygon_vertices": len(polygon)})
        void_rows = []
        for void in construction[f"{label}_patch"]["major_voids"]:
            target = private["void_labels"] == void["label"]
            fill_fraction = float(np.sum(union & target)) / max(1, int(np.sum(target)))
            void_rows.append(
                {
                    "role": void["role"],
                    "fill_fraction": fill_fraction,
                    "registered_max_fill_fraction": void["registered_max_fill_fraction"],
                    "pass": fill_fraction <= void["registered_max_fill_fraction"],
                }
            )
        output[label] = {
            "semantics": (
                "2-D projection of contact-skin plus immediately adjacent jaw-backbone parts only; "
                "connector/moving-support parts excluded; not a full-candidate or through-depth proof"
            ),
            "excluded_roles": ["connector_support", "moving_support", "structural_body"],
            "projected_parts": part_rows,
            "voids": void_rows,
            "pass": len(void_rows) == 2 and all(row["pass"] for row in void_rows),
        }
    return {"bodies": output, "pass": all(row["pass"] for row in output.values())}


def _immutable_d362_transition_replay(
    candidate: dict[str, list[dict[str, Any]]], current: dict[str, list[dict[str, Any]]], d371: Any,
    expected_sha256: str,
) -> dict[str, Any]:
    trace = _read_json(D362_TRACE)
    selected_rows = [
        row for row in trace
        if row.get("phase") == TRACE_PHASE and int(row.get("phase_step", -1)) <= TRACE_PHASE_STEP_MAX
    ]
    phase_steps = [int(row["phase_step"]) for row in selected_rows]
    rows_by_body: dict[str, list[dict[str, Any]]] = {}
    query_count = 0
    for body in BODY_NAMES:
        body_rows = []
        role_by_name = {part["name"]: part["role"] for part in candidate[body]}
        for trace_row in selected_rows:
            if body == "link5":
                body_pose = {
                    "pos_m": trace_row["link5_pos_w_m"],
                    "quat_wxyz": trace_row["link5_quat_wxyz"],
                }
            else:
                body_pose = {
                    "pos_m": trace_row["gripper_pos_w_m"],
                    "quat_wxyz": trace_row["gripper_quat_wxyz"],
                }
            cylinder_pose = {
                "object_pos_w_m": trace_row["object_pos_w_m"],
                "object_quat_wxyz": trace_row["object_quat_wxyz"],
            }
            a64 = _query_parts(current[body], body_pose, cylinder_pose, d371)
            p34 = _query_parts(candidate[body], body_pose, cylinder_pose, d371)
            query_count += len(a64["queries"]) + len(p34["queries"])
            body_rows.append(
                {
                    "global_step": int(trace_row["global_step"]),
                    "phase_step": int(trace_row["phase_step"]),
                    "q5_actual_rad_from_immutable_trace": float(trace_row["q5_actual_rad"]),
                    "A64_exact_signed_distance_mm": float(a64["selected"]["exact_signed_distance_mm"]),
                    "P34_exact_signed_distance_mm": float(p34["selected"]["exact_signed_distance_mm"]),
                    "absolute_distance_delta_mm": abs(
                        float(a64["selected"]["exact_signed_distance_mm"])
                        - float(p34["selected"]["exact_signed_distance_mm"])
                    ),
                    "A64_collision": bool(a64["collision_parts"]),
                    "P34_collision": bool(p34["collision_parts"]),
                    "A64_selected_part": a64["selected"]["part"],
                    "P34_selected_part": p34["selected"]["part"],
                    "P34_selected_role": role_by_name[p34["selected"]["part"]],
                    "body_pose": body_pose,
                    "cylinder_pose": cylinder_pose,
                }
            )
        rows_by_body[body] = body_rows
    bodies = {}
    for body, rows in rows_by_body.items():
        first_a64 = next((row for row in rows if row["A64_collision"]), None)
        first_p34 = next((row for row in rows if row["P34_collision"]), None)
        if first_a64 is None or first_p34 is None:
            transition_rows: list[dict[str, Any]] = []
        else:
            transition_rows = [
                row for row in rows
                if first_a64["phase_step"] - 2 <= row["phase_step"] <= first_a64["phase_step"]
            ]
        expected_role = "fixed_jaw" if body == "link5" else "moving_jaw"
        checks = {
            "A64_first_collision_matches_registered_global_step": first_a64 is not None
            and first_a64["global_step"] == EXPECTED_TRACE_FIRST_GLOBAL_STEP[body],
            "P34_first_collision_matches_A64_exactly": first_p34 is not None
            and first_a64 is not None
            and first_p34["global_step"] == first_a64["global_step"],
            "no_P34_collision_before_A64": first_p34 is not None
            and first_a64 is not None
            and first_p34["global_step"] >= first_a64["global_step"],
            "P34_first_overlap_is_contact_skin_not_backbone": first_p34 is not None
            and first_p34["P34_selected_role"] == expected_role,
            "transition_window_distance_delta_le_registered": len(transition_rows) == 3
            and max(row["absolute_distance_delta_mm"] for row in transition_rows)
            <= TRACE_TRANSITION_WINDOW_DELTA_MAX_MM,
        }
        bodies[body] = {
            "timeline": rows,
            "first_A64_collision": first_a64,
            "first_P34_collision": first_p34,
            "transition_window": transition_rows,
            "transition_window_max_abs_distance_delta_mm": max(
                [row["absolute_distance_delta_mm"] for row in transition_rows] or [math.inf]
            ),
            "whole_replay_max_abs_distance_delta_mm_diagnostic": max(
                row["absolute_distance_delta_mm"] for row in rows
            ),
            "checks": checks,
            "pass": all(checks.values()),
        }
    trace_checks = {
        "trace_sha256_matches_preregistered_input": _sha(D362_TRACE) == expected_sha256,
        "phase_row_count_61": len(selected_rows) == TRACE_PHASE_STEP_MAX + 1,
        "phase_steps_exact_0_through_60": phase_steps == list(range(TRACE_PHASE_STEP_MAX + 1)),
    }
    return {
        "source": _rel(D362_TRACE),
        "source_sha256": _sha(D362_TRACE),
        "semantics": "read-only counterfactual geometry replay on immutable D362 poses; no new q5 command, physics, or causal dynamics claim",
        "phase": TRACE_PHASE,
        "phase_step_range": [0, TRACE_PHASE_STEP_MAX],
        "row_count": len(selected_rows),
        "offline_part_query_count": query_count,
        "trace_checks": trace_checks,
        "bodies": bodies,
        "pass": all(trace_checks.values()) and all(row["pass"] for row in bodies.values()),
    }


def _owner_manifest_pass(manifest: dict[str, str], expected: dict[str, str]) -> bool:
    return manifest == expected


def _negative_controls(
    candidate: dict[str, list[dict[str, Any]]],
    raw: dict[str, dict[str, Any]],
    construction: dict[str, Any],
    owner_contract: dict[str, Any],
    d368: Any,
    d371: Any,
) -> dict[str, Any]:
    frozen = _read_json(D349_MEASUREMENT)["distance_gate"]["authoritative_pose_streams"]["raw_first"]
    link_lo = np.asarray(raw["link5"]["vertices_m"].min(axis=0), dtype=np.float64)
    link_hi = np.asarray(raw["link5"]["vertices_m"].max(axis=0), dtype=np.float64)
    full_box = _make_box_part(body="link5", name="negative_full_link5_AABB", role="negative", lo=link_lo, hi=link_hi)
    full_box["_mesh"] = d368._trimesh(full_box["vertices"], full_box["triangles"])
    box_query = _query_parts([full_box], frozen["body_poses_w"]["link5"], frozen, d371)
    envelope_controls = {}
    for label in ("fixed", "moving"):
        private = construction["_private"][label]
        raster = private["raster"]
        envelope_mask = _polygon_mask(raster["outer_poly"], raster["xs"], raster["ys"])
        rows = []
        for void in construction[f"{label}_patch"]["major_voids"]:
            target = private["void_labels"] == void["label"]
            fraction = float(np.sum(envelope_mask & target)) / max(1, int(np.sum(target)))
            rows.append(
                {
                    "role": void["role"],
                    "fill_fraction": fraction,
                    "registered_max": void["registered_max_fill_fraction"],
                    "rejected": fraction > void["registered_max_fill_fraction"],
                }
            )
        envelope_controls[label] = {
            "polygon_2d_m": raster["outer_poly"].tolist(),
            "voids": rows,
            "rejected": bool(rows) and all(row["rejected"] for row in rows),
        }
    fixed_parts = [part for part in candidate["link5"] if part["role"] == "fixed_jaw"]
    fixed_seed = np.asarray(_read_json(D350_BINDING)["seed_local_m"], dtype=np.float64)
    without_fixed = [part for part in candidate["link5"] if part["role"] != "fixed_jaw"]
    fixed_lost_distance = min(
        float(d368._nearest(part["_mesh"], fixed_seed.reshape(1, 3))[0][0] * 1000.0)
        for part in without_fixed
    )
    moving_parts = [part for part in candidate["gripper_link"] if part["role"] == "moving_jaw"]
    moving_seed = np.asarray(_read_json(D354_BINDING)["seed_local_m"], dtype=np.float64)
    without_moving = [part for part in candidate["gripper_link"] if part["role"] != "moving_jaw"]
    moving_lost_distance = min(
        float(d368._nearest(part["_mesh"], moving_seed.reshape(1, 3))[0][0] * 1000.0)
        for part in without_moving
    )
    expected_owner_manifest = dict(owner_contract["owner_contract"]["expected_manifest"])
    valid_owner_manifest = dict(owner_contract["owner_contract"]["candidate_manifest"])
    swapped_owner_manifest = {
        "fixed_jaw": expected_owner_manifest["moving_jaw"],
        "moving_jaw": expected_owner_manifest["fixed_jaw"],
    }
    checks = {
        "full_link5_AABB_rejected": bool(box_query["collision_parts"]) or float(box_query["selected"]["exact_signed_distance_mm"]) < OPEN_CLEAR_GATE_MM,
        "single_fixed_envelope_rejected": envelope_controls["fixed"]["rejected"],
        "single_moving_envelope_rejected": envelope_controls["moving"]["rejected"],
        "jaw_contact_layer_removal_loses_both_certified_seeds": fixed_lost_distance
        > RASTER_STEP_M * 1000.0
        and moving_lost_distance > RASTER_STEP_M * 1000.0,
        "owner_swap_rejected_against_URDF": _owner_manifest_pass(
            valid_owner_manifest, expected_owner_manifest
        )
        and not _owner_manifest_pass(swapped_owner_manifest, expected_owner_manifest),
    }
    return {
        "full_link5_AABB": {"bounds_m": [link_lo.tolist(), link_hi.tolist()], "query": box_query["selected"], "collision_parts": box_query["collision_parts"]},
        "single_envelope_controls": envelope_controls,
        "single_envelope_semantics": "actual projected outer-envelope polygons were rasterized and evaluated by the same registered void-fill gates",
        "owner_control": {
            "authority": "URDF link5_to_gripper_link parent/child plus D350/D354 owner subresults",
            "expected_manifest_from_URDF": expected_owner_manifest,
            "valid_manifest": valid_owner_manifest,
            "valid_manifest_pass": _owner_manifest_pass(valid_owner_manifest, expected_owner_manifest),
            "swapped_manifest": swapped_owner_manifest,
            "swapped_manifest_pass": _owner_manifest_pass(swapped_owner_manifest, expected_owner_manifest),
        },
        "fixed_seed_distance_without_fixed_jaw_mm": fixed_lost_distance,
        "moving_seed_distance_without_moving_jaw_mm": moving_lost_distance,
        "fixed_jaw_part_count": len(fixed_parts),
        "moving_jaw_part_count": len(moving_parts),
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "pass": all(checks.values()),
    }


def _transform(vertices: np.ndarray, pose: dict[str, Any]) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float64) @ _quat_wxyz_to_rot(pose["quat_wxyz"]).T + np.asarray(pose["pos_m"], dtype=np.float64)


def _cylinder_mesh(pose: dict[str, Any], segments: int = 48) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    local = []
    for z in (-0.045, 0.045):
        local.extend([[0.017 * math.cos(a), 0.017 * math.sin(a), z] for a in angles])
    local.extend([[0.0, 0.0, -0.045], [0.0, 0.0, 0.045]])
    faces = []
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.extend([[index, nxt, segments + nxt], [index, segments + nxt, segments + index]])
        faces.extend([[2 * segments, nxt, index], [2 * segments + 1, segments + index, segments + nxt]])
    vertices = np.asarray(local, dtype=np.float64) @ _quat_wxyz_to_rot(pose["object_quat_wxyz"]).T + np.asarray(pose["object_pos_w_m"], dtype=np.float64)
    return vertices, np.asarray(faces, dtype=np.int64)


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        dimensions = [int(image.width), int(image.height)]
        mode = image.mode
    return {"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha(path), "dimensions": dimensions, "mode": mode, "exact_1920x1080": dimensions == [1920, 1080]}


def _render_boards(
    raw: dict[str, dict[str, Any]],
    candidate: dict[str, list[dict[str, Any]]],
    patches: dict[str, Any],
    construction: dict[str, Any],
    clearance: dict[str, Any],
    transition_replay: dict[str, Any],
    jaw_void_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    role_colors = {
        "structural_body": "#0072B2",
        "connector_support": "#E69F00",
        "fixed_jaw": "#F0E442",
        "fixed_jaw_backbone": "#D98C00",
        "moving_support": "#CC79A7",
        "moving_jaw": "#009E73",
        "moving_jaw_backbone": "#00796B",
    }

    def add_mesh(ax: Any, vertices: np.ndarray, triangles: np.ndarray, color: str, alpha: float, edge: str = "#202020", width: float = 0.05) -> None:
        mesh = Poly3DCollection(np.asarray(vertices)[np.asarray(triangles, dtype=np.int64)] * 1000.0, facecolor=color, edgecolor=edge, linewidth=width, alpha=alpha)
        ax.add_collection3d(mesh)

    def frame(ax: Any, vertices: np.ndarray, *, elev: float, azim: float) -> None:
        mm = np.asarray(vertices) * 1000.0
        lo, hi = mm.min(axis=0), mm.max(axis=0)
        span = np.maximum(hi - lo, 1.0)
        margin = np.maximum(span * 0.08, 2.0)
        ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0]); ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1]); ax.set_zlim(lo[2] - margin[2], hi[2] + margin[2])
        ax.set_box_aspect(tuple(np.maximum(span, 1.0))); ax.view_init(elev=elev, azim=azim); ax.set_proj_type("ortho"); ax.set_axis_off()

    # Board 1: owner and structure split.
    fig = plt.figure(figsize=(16, 9), dpi=120)
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    for row, body in enumerate(BODY_NAMES):
        raw_ax, candidate_ax, contact_ax = axes[row * 3 : row * 3 + 3]
        add_mesh(raw_ax, raw[body]["vertices_m"], raw[body]["triangles"], "#9A9A9A", 0.55)
        frame(raw_ax, raw[body]["vertices_m"], elev=18, azim=-62 if body == "link5" else -90)
        raw_ax.set_title(("link5 원본 메시" if body == "link5" else "gripper_link 원본 메시"), fontproperties=bold, fontsize=12)
        add_mesh(candidate_ax, raw[body]["vertices_m"], raw[body]["triangles"], "#A0A0A0", 0.08, "#606060", 0.025)
        for part in candidate[body]:
            add_mesh(candidate_ax, part["vertices"], part["triangles"], role_colors[part["role"]], 0.62)
        frame(candidate_ax, raw[body]["vertices_m"], elev=18, azim=-62 if body == "link5" else -90)
        candidate_ax.set_title(f"원본 외곽선 + 새 충돌체 · {len(candidate[body])}개", fontproperties=bold, fontsize=12)
        add_mesh(contact_ax, raw[body]["vertices_m"], raw[body]["triangles"], "#A0A0A0", 0.10, "#505050", 0.02)
        role = "fixed_jaw" if body == "link5" else "moving_jaw"
        jaw_roles = {role, f"{role}_backbone"}
        for part in candidate[body]:
            if part["role"] in jaw_roles:
                add_mesh(contact_ax, part["vertices"], part["triangles"], role_colors[part["role"]], 0.72)
        patch = patches["fixed" if body == "link5" else "inner"]
        add_mesh(contact_ax, patch["vertices"], patch["triangles"], "#00BFC4", 0.22, "#101010", 0.04)
        jaw_vertices = [patch["vertices"]] + [part["vertices"] for part in candidate[body] if part["role"] in jaw_roles]
        frame(contact_ax, np.vstack(jaw_vertices), elev=8, azim=0 if body == "link5" else -90)
        contact_ax.set_title("턱만 분리 확대", fontproperties=bold, fontsize=12)
    fig.suptitle("D372 교수님 안 — 몸통·고정 턱·움직이는 턱을 소유 링크별로 분리", fontproperties=bold, fontsize=20, y=0.96)
    fig.text(0.5, 0.035, "파랑=몸통 박스 · 주황=고정 턱 연결/뒷면 지지대 · 노랑=고정 접촉면 · 보라=움직이는 연결부 · 초록=움직이는 접촉면/위·아래 지지대", ha="center", fontproperties=bold, fontsize=12)
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.92]); fig.savefig(OWNERSHIP_PNG, dpi=120, facecolor="white"); plt.close(fig)

    # Board 2: projected void preservation.
    fig, axes2 = plt.subplots(2, 2, figsize=(16, 9), dpi=120)
    for row, (label, display) in enumerate((("fixed", "고정 턱"), ("moving", "움직이는 턱"))):
        private = construction["_private"][label]
        raster = private["raster"]
        extent = [raster["xs"][0] * 1000.0, raster["xs"][-1] * 1000.0, raster["ys"][0] * 1000.0, raster["ys"][-1] * 1000.0]
        axes2[row, 0].imshow(raster["mask"], origin="lower", extent=extent, cmap="Greys", interpolation="nearest")
        axes2[row, 0].set_title(f"{display} 원본 접촉판", fontproperties=bold, fontsize=13)
        axes2[row, 1].imshow(private["candidate_mask"], origin="lower", extent=extent, cmap="viridis", interpolation="nearest")
        axes2[row, 1].contour(raster["mask"].astype(float), levels=[0.5], colors=["black"], linewidths=0.8, origin="lower", extent=extent)
        metric = construction[f"{label}_patch"]
        compound = jaw_void_diagnostic["bodies"][label]["voids"]
        axes2[row, 1].set_title(
            f"접촉층 {metric['polygon_count']}조각 · 접촉층+인접 지지대 2D 빈 공간 채움 "
            + "/".join(f"{item['fill_fraction']*100:.1f}%" for item in compound),
            fontproperties=bold,
            fontsize=12,
        )
        for col in range(2):
            axes2[row, col].set_aspect("equal"); axes2[row, col].set_xlabel("가로 (mm)", fontproperties=regular); axes2[row, col].set_ylabel("세로 (mm)", fontproperties=regular)
    fig.suptitle("D372 턱 분할 근거 — 접촉층을 통째 볼록껍질로 만들지 않아 큰 빈 공간을 남김", fontproperties=bold, fontsize=20, y=0.97)
    fig.text(0.5, 0.025, f"격자 {RASTER_STEP_M*1000:.2f}mm · 고정 턱 {construction['fixed_patch']['polygon_count']}조각 · 움직이는 턱 {construction['moving_patch']['polygon_count']}조각", ha="center", fontproperties=bold, fontsize=12)
    fig.tight_layout(rect=[0.03, 0.06, 0.97, 0.93]); fig.savefig(JAWS_PNG, dpi=120, facecolor="white"); plt.close(fig)

    # Board 3: frozen-open pose plus the immutable D362 contact transition.
    frozen = _read_json(D349_MEASUREMENT)["distance_gate"]["authoritative_pose_streams"]["raw_first"]
    cylinder_vertices, cylinder_triangles = _cylinder_mesh(frozen)
    fig = plt.figure(figsize=(16, 9), dpi=120)
    axes3 = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    for ax, body in zip(axes3[:2], BODY_NAMES):
        body_pose = frozen["body_poses_w"][body]
        all_world = []
        for part in candidate[body]:
            world = _transform(part["vertices"], body_pose); all_world.append(world)
            add_mesh(ax, world, part["triangles"], role_colors[part["role"]], 0.65)
        add_mesh(ax, cylinder_vertices, cylinder_triangles, "#D55E00", 0.38, "#8B2500", 0.08)
        witness = clearance["bodies"][body]["P34_selected_witness"]
        witness_points = np.asarray(
            [witness["nearest_point_geometry_m"], witness["nearest_point_cylinder_m"]],
            dtype=np.float64,
        ) * 1000.0
        ax.plot(witness_points[:, 0], witness_points[:, 1], witness_points[:, 2], color="#00BFC4", linewidth=3.0)
        ax.scatter(witness_points[:, 0], witness_points[:, 1], witness_points[:, 2], color="#00BFC4", s=24)
        frame(ax, np.vstack([*all_world, cylinder_vertices]), elev=20, azim=-55)
        value = clearance["bodies"][body]["P34_exact_signed_distance_mm"]
        ax.set_title(("link5·고정 턱" if body == "link5" else "gripper_link·움직이는 턱") + f"\n정적 열린 간격 {value:.3f}mm", fontproperties=bold, fontsize=14)
    for ax, body in zip(axes3[2:], BODY_NAMES):
        first = transition_replay["bodies"][body]["first_P34_collision"]
        cylinder_at_contact = {
            "object_pos_w_m": first["cylinder_pose"]["object_pos_w_m"],
            "object_quat_wxyz": first["cylinder_pose"]["object_quat_wxyz"],
        }
        cylinder_contact_vertices, cylinder_contact_triangles = _cylinder_mesh(cylinder_at_contact)
        all_world = []
        for part in candidate[body]:
            world = _transform(part["vertices"], first["body_pose"]); all_world.append(world)
            add_mesh(ax, world, part["triangles"], role_colors[part["role"]], 0.65)
        add_mesh(ax, cylinder_contact_vertices, cylinder_contact_triangles, "#D55E00", 0.38, "#8B2500", 0.08)
        frame(ax, np.vstack([*all_world, cylinder_contact_vertices]), elev=20, azim=-55)
        contact_name = "고정 턱 접촉층" if body == "link5" else "움직이는 턱 접촉층"
        ax.set_title(
            ("고정 턱" if body == "link5" else "움직이는 턱")
            + f" · 저장 trace 첫 겹침\nphase {first['phase_step']} / global {first['global_step']} · {contact_name}",
            fontproperties=bold,
            fontsize=11,
        )
    fig.suptitle("D372 정적 형상 확인 — 열린 간격과 저장된 D362 첫 접촉 시점", fontproperties=bold, fontsize=20, y=0.97)
    fig.text(0.5, 0.035, "D349/D362 저장 좌표만 오프라인 재계산 · 새 q5 구동 0 · 물리 스텝 0 · 실제 접촉 시험 0", ha="center", fontproperties=bold, fontsize=12)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.92]); fig.savefig(CLEARANCE_PNG, dpi=120, facecolor="white"); plt.close(fig)
    infos = {"ownership": _png_info(OWNERSHIP_PNG), "jaws": _png_info(JAWS_PNG), "clearance": _png_info(CLEARANCE_PNG)}
    if not all(row["exact_1920x1080"] for row in infos.values()):
        raise RuntimeError(f"board dimensions failed: {infos}")
    return infos


def _write_rerun(raw: dict[str, dict[str, Any]], candidate: dict[str, list[dict[str, Any]]], patches: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.viz_debug import log_rerun

    colors = {
        "structural_body": [0, 114, 178, 150],
        "connector_support": [230, 159, 0, 150],
        "fixed_jaw": [240, 228, 66, 190],
        "fixed_jaw_backbone": [217, 140, 0, 175],
        "moving_support": [204, 121, 167, 150],
        "moving_jaw": [0, 158, 115, 180],
        "moving_jaw_backbone": [0, 121, 107, 170],
    }
    meshes = []
    points = []
    arrows = []
    expected = []
    for body in BODY_NAMES:
        raw_path = f"semantic/source/{body}/raw_surface"
        meshes.append({"entity_path": raw_path, "coordinate_frame": "tf#/", "vertices_m": raw[body]["vertices_m"], "triangles": raw[body]["triangles"], "color_rgba": [130, 130, 130, 40], "static": True, "authority": "raw Float64; Rerun display-only"})
        expected.extend([raw_path, f"metadata/meshes/{raw_path.replace('/', '__')}"])
        for part in candidate[body]:
            path = f"semantic/collider/{body}/{part['name']}"
            meshes.append({"entity_path": path, "coordinate_frame": "tf#/", "vertices_m": part["vertices"], "triangles": part["triangles"], "color_rgba": colors[part["role"]], "static": True, "role": part["role"], "payload_sha256": part["payload_sha256"], "authority": "D372 Float64 candidate; Rerun display-only Float32 copy"})
            expected.extend([path, f"metadata/meshes/{path.replace('/', '__')}"])
        patch_label = "fixed" if body == "link5" else "inner"
        patch_path = f"semantic/source/{body}/{'seed_contact_plane_patch' if body == 'link5' else 'inner_contact_patch'}"
        patch = patches[patch_label]
        meshes.append({"entity_path": patch_path, "coordinate_frame": "tf#/", "vertices_m": patch["vertices"], "triangles": patch["triangles"], "color_rgba": [0, 205, 255, 110], "static": True, "authority": "raw certified patch"})
        expected.extend([patch_path, f"metadata/meshes/{patch_path.replace('/', '__')}"])
    frozen = _read_json(D349_MEASUREMENT)["distance_gate"]["authoritative_pose_streams"]["raw_first"]
    cylinder_vertices, cylinder_triangles = _cylinder_mesh(frozen)
    cylinder_path = "d372/world/cylinder"
    meshes.append({"entity_path": cylinder_path, "coordinate_frame": "tf#/", "vertices_m": cylinder_vertices, "triangles": cylinder_triangles, "color_rgba": [213, 94, 0, 115], "static": True, "authority": "immutable D349 cylinder pose; display only"})
    expected.extend([cylinder_path, f"metadata/meshes/{cylinder_path.replace('/', '__')}"])
    for body in BODY_NAMES:
        pose = frozen["body_poses_w"][body]
        for part in candidate[body]:
            path = f"d372/world/{body}/{part['name']}"
            meshes.append({"entity_path": path, "coordinate_frame": "tf#/", "vertices_m": _transform(part["vertices"], pose), "triangles": part["triangles"], "color_rgba": colors[part["role"]], "static": True, "role": part["role"], "authority": "immutable D349 pose plus D372 candidate; display only"})
            expected.extend([path, f"metadata/meshes/{path.replace('/', '__')}"])
        witness = evidence["frozen_open_clearance"]["bodies"][body]["P34_selected_witness"]
        geometry_point = np.asarray(witness["nearest_point_geometry_m"], dtype=np.float64)
        cylinder_point = np.asarray(witness["nearest_point_cylinder_m"], dtype=np.float64)
        point_path = f"d372/world/witness/{body}/nearest_points"
        arrow_path = f"d372/world/witness/{body}/clearance_vector"
        points.append({"entity_path": point_path, "positions_m": [geometry_point, cylinder_point], "radii": [0.0012, 0.0012], "colors": [[0, 255, 255, 255], [255, 255, 255, 255]], "coordinate_frame": "tf#/", "static": True})
        arrows.append({"entity_path": arrow_path, "origins_m": [geometry_point], "vectors_m": [cylinder_point - geometry_point], "radii": [0.00035], "colors": [[0, 255, 255, 255]], "coordinate_frame": "tf#/", "static": True})
        expected.extend([point_path, arrow_path])
    scalar_values = {
        "link5_parts": evidence["actual_part_counts"]["link5"],
        "gripper_link_parts": evidence["actual_part_counts"]["gripper_link"],
        "total_parts": evidence["actual_part_counts"]["total"],
        "fixed_jaw_prisms": evidence["construction"]["fixed_patch"]["polygon_count"],
        "moving_jaw_prisms": evidence["construction"]["moving_patch"]["polygon_count"],
        "physics_steps": 0,
        "q5_samples": 0,
        "link5_open_clearance_mm": evidence["frozen_open_clearance"]["bodies"]["link5"]["P34_exact_signed_distance_mm"],
        "moving_open_clearance_mm": evidence["frozen_open_clearance"]["bodies"]["gripper_link"]["P34_exact_signed_distance_mm"],
    }
    scalars = [
        {"entity_path": f"metrics/d372/{name}", "value": value, "static": True}
        for name, value in scalar_values.items()
    ]
    for body in BODY_NAMES:
        for row in evidence["immutable_D362_transition_replay"]["bodies"][body]["timeline"]:
            scalars.extend(
                [
                    {
                        "entity_path": f"metrics/d372_trace/{body}/A64_signed_distance_mm",
                        "value": row["A64_exact_signed_distance_mm"],
                        "sequence": {"d362_phase_step": row["phase_step"]},
                    },
                    {
                        "entity_path": f"metrics/d372_trace/{body}/P34_signed_distance_mm",
                        "value": row["P34_exact_signed_distance_mm"],
                        "sequence": {"d362_phase_step": row["phase_step"]},
                    },
                ]
            )
    events = [
        {
            "entity_path": "events/d372_summary",
            "text": (
                "P34 parts: link5=16, gripper_link=18, total=34. "
                "Blue=body; orange=fixed support; yellow=fixed contact; purple=moving support; green=moving jaw. "
                "Offline stored-pose replay only: Isaac=0 PhysX=0 q5=0 steps=0. Not grasp/physics/optimum proof."
            ),
            "level": "INFO",
            "static": True,
        }
    ]
    expected.extend([*(f"metrics/d372/{name}" for name in scalar_values), "events/d372_summary"])
    expected.extend(sorted({row["entity_path"] for row in scalars if row["entity_path"].startswith("metrics/d372_trace/")}))
    result = log_rerun(
        RRD_PATH,
        meshes=meshes,
        points=points,
        arrows=arrows,
        scalar_trace=scalars,
        events=events,
        recording_metadata={"case": "g0a_d372", "attempt": ATTEMPT_NAME, "verdict": evidence["verdict"], "evidence_sha256": _sha(EVIDENCE_PATH), "physics_steps": 0, "q5_samples": 0, "contact_queries": 0, "display_role": "inspection only"},
        recording_id="g0a_d372_professor_semantic_compound",
        blueprint_path=RBL_PATH,
        blueprint_mode="d372_semantic_compound",
        live_viewer=False,
        app_id="roarm_g0a_d372_semantic_compound",
    )
    if not result.get("ok"):
        raise RuntimeError(f"Rerun write failed: {result}")
    expected.append("metadata/run")
    component_contract: dict[str, list[str]] = {}
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    for row in meshes:
        component_contract[row["entity_path"]] = mesh_components
        component_contract[f"metadata/meshes/{row['entity_path'].replace('/', '__')}"] = ["TextDocument:text"]
    for row in points:
        component_contract[row["entity_path"]] = ["Points3D:positions"]
    for row in arrows:
        component_contract[row["entity_path"]] = ["Arrows3D:vectors"]
    for row in scalars:
        component_contract[row["entity_path"]] = ["Scalars:scalars"]
    component_contract["events/d372_summary"] = ["TextLog:text", "TextLog:level"]
    component_contract["metadata/run"] = ["TextDocument:text"]
    strict = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected),
        exact_entity_paths=sorted(expected),
        expected_timeline_names=["blueprint", "d362_phase_step", "log_time"],
        exact_timeline_names=["blueprint", "d362_phase_step", "log_time"],
        expected_entity_components=component_contract,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG,
        screenshot_window_size="1920x1080",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=240.0,
    )
    _write_json_x(RERUN_VALIDATION_PATH, strict)
    screenshot = (
        _png_info(RERUN_PNG)
        if RERUN_PNG.is_file()
        else {"path": _rel(RERUN_PNG), "exists": False, "bytes": 0, "sha256": None, "dimensions": None, "exact_1920x1080": False}
    )
    return {"log": result, "validation_pass": strict.get("pass") is True, "rrd_sha256": _sha(RRD_PATH), "rbl_sha256": _sha(RBL_PATH), "screenshot": screenshot}


def _write_report(evidence: dict[str, Any]) -> None:
    inventory = evidence["candidate_inventory"]
    fixed = evidence["construction"]["fixed_patch"]
    moving = evidence["construction"]["moving_patch"]
    clearance = evidence["frozen_open_clearance"]["bodies"]
    replay = evidence["immutable_D362_transition_replay"]["bodies"]
    jaw_voids = evidence["jaw_contact_layer_void_diagnostic"]["bodies"]
    lineage = evidence["prior_evidence_and_owner_lineage"]["candidate_count_lineage"]
    nvidia = evidence["nvidia_contract"]
    official_lines = [
        f"- {row['title']} ({row['url']}) — {row['use']}"
        for row in nvidia["official_sources"]
    ]
    lines = [
        "# D372 교수님 안: 의미 기반 복합 충돌체",
        "",
        "이 단계는 충돌체 후보를 오프라인에서 만든 것입니다. Isaac/PhysX 실행, 물리 스텝, q5 구동, 실제 접촉·파지는 모두 0회입니다.",
        "",
        "## 만든 구조",
        "",
        f"- link5: 몸통 박스 1개 + 연결/회전축 3개 + 고정 접촉판 {fixed['polygon_count']}개 + 고정 턱 뒷면 지지대 2개 = {inventory['link5']['part_count']}개",
        f"- gripper_link: 근위 지지 볼록껍질 4개 + 움직이는 접촉판 {moving['polygon_count']}개 + 위/아래 뒷면 지지대 2개 = {inventory['gripper_link']['part_count']}개",
        f"- 합계: {inventory['link5']['part_count'] + inventory['gripper_link']['part_count']}개 (현재 기준 128개)",
        "",
        "## 개수 비교와 의미",
        "",
        "| 후보 | 생성 방식 | link5 | gripper_link | 합계 | 해석 |",
        "|---|---:|---:|---:|---:|---|",
        f"| 현재 A64 | 자동 convex decomposition | {lineage['current_A64_automatic_decomposition']['link5']} | {lineage['current_A64_automatic_decomposition']['gripper_link']} | {lineage['current_A64_automatic_decomposition']['total']} | 현재 64-cap 기준 후보 |",
        f"| D371 R32 | 자동 convex decomposition | {lineage['D371_R32_automatic_decomposition']['link5']} | {lineage['D371_R32_automatic_decomposition']['gripper_link']} | {lineage['D371_R32_automatic_decomposition']['total']} | maxConvexHulls만 32로 바꾼 비교 후보 |",
        f"| D372 P34 | 의미 기반 수동 복합 충돌체 | {lineage['D372_P34_manual_semantic_compound']['link5']} | {lineage['D372_P34_manual_semantic_compound']['gripper_link']} | {lineage['D372_P34_manual_semantic_compound']['total']} | 몸통·고정 턱·움직이는 턱을 역할별 분리한 G0a 후보 |",
        "",
        "- 설치 스키마의 `maxConvexHulls=32`는 자동 분해의 기본값이지, 수동 child collider의 목표 개수가 아닙니다.",
        "- D371 C1/C2는 몸통과 턱의 정확한 분할이 아니었으므로 D372 설계의 직접 대안으로 채택하지 않습니다.",
        "- 부품 수만으로 속도, 물리 동등성, 전도 개선, 최적성을 증명할 수 없습니다.",
        "",
        "## 빈 공간과 열린 자세",
        "",
        f"- 고정 턱 접촉층+인접 뒷면 지지대의 2D 큰 빈 공간 채움: {jaw_voids['fixed']['voids'][0]['fill_fraction']*100:.2f}% / {jaw_voids['fixed']['voids'][1]['fill_fraction']*100:.2f}%",
        f"- 움직이는 턱 접촉층+위·아래 지지대의 2D 열린 입구 채움: {jaw_voids['moving']['voids'][0]['fill_fraction']*100:.2f}%",
        f"- 움직이는 턱 접촉층+위·아래 지지대의 2D 내부 창 채움(진단): {jaw_voids['moving']['voids'][1]['fill_fraction']*100:.2f}%",
        "- 위 2D 수치는 뒤쪽 connector/moving-support를 제외한 접촉층 진단입니다. 전체 P34의 관통 공간이나 3D 물리를 증명하지 않습니다.",
        f"- 동결 OPEN 정적 간격: link5 {clearance['link5']['P34_exact_signed_distance_mm']:.6f}mm, 움직이는 턱 {clearance['gripper_link']['P34_exact_signed_distance_mm']:.6f}mm",
        f"- 저장된 D362 좌표 재계산의 첫 겹침 global step: link5 A64/P34={replay['link5']['first_A64_collision']['global_step']}/{replay['link5']['first_P34_collision']['global_step']}, 움직이는 턱 A64/P34={replay['gripper_link']['first_A64_collision']['global_step']}/{replay['gripper_link']['first_P34_collision']['global_step']}",
        "",
        "## NVIDIA 공식 근거와 설치 버전",
        "",
        f"- 설치 제품: Isaac Sim {nvidia['installed_isaac_sim']}, Omni PhysX extension {nvidia['installed_omni_physx_extension']}",
        f"- 설치 스키마: `{nvidia['schema_path']}:{nvidia['schema_hull_vertex_limit_line']}`의 hullVertexLimit=64, `:{nvidia['schema_max_convex_hulls_line']}`의 maxConvexHulls=32 기본값",
        f"- 설치 속성 편집 UI: `{nvidia['property_database_path']}:{nvidia['property_database_ranges_line']}`에서 hullVertexLimit 8~64, maxConvexHulls 1~2048. 이는 UI 입력 범위이며 엔진 절대 한계나 최적값이 아닙니다.",
        "- D372의 convex당 64 vertices/64 polygons/32 vertices-per-polygon는 프로젝트가 사전등록한 GPU 적격성 gate입니다. 실제 GPU 실행을 뜻하지 않습니다.",
        *official_lines,
        "",
        "## 해석 경계",
        "",
        "- 이 후보는 G0a 원통 경로를 위한 task-local 후보이며, 일반 목적 로봇 충돌 모델의 최적값이 아닙니다.",
        "- D362 재계산은 저장 좌표에 새 형상을 대입한 오프라인 검사일 뿐, 새 형상이 같은 동역학을 만든다는 인과 증명이 아닙니다.",
        "- 이 후보는 다음 live asset identity 검사 대상으로 적격인지까지만 판정합니다.",
        "- 아직 실제 속도, 접촉 순서, 원통 전도, 파지 성공, 전역 최적성을 판정하지 않습니다.",
        "- 후속 물리 비교에서는 link5만 교체한 경우와 gripper_link만 교체한 경우를 분리해야 원인을 구분할 수 있습니다.",
        "",
        "## 시각자료",
        "",
        f"- 소유 링크와 분할: `{_rel(OWNERSHIP_PNG)}`",
        f"- 턱 빈 공간: `{_rel(JAWS_PNG)}`",
        f"- 동결 OPEN 정적 간격: `{_rel(CLEARANCE_PNG)}`",
        f"- Rerun 기록: `{_rel(RRD_PATH)}`",
    ]
    _write_text_x(REPORT_PATH, "\n".join(lines) + "\n")


def _run() -> None:
    if INVOCATION_PATH.exists():
        raise FileExistsError("D372 actual run already invoked; no retry")
    prereg = _read_json(PREREG_PATH)
    invocation = {"artifact": "D372_INVOCATION_V1", "attempt": ATTEMPT_NAME, "run_invocation_count": 1, "automatic_retry_count": 0, "started_monotonic_ns": time.monotonic_ns()}
    _write_json_x(INVOCATION_PATH, invocation)
    _phase("run_started")
    checks_pre = {
        "prereg_pass": prereg.get("pass") is True,
        "head_unchanged": _git("rev-parse", "HEAD") == prereg["head"],
        "origin_unchanged": _git("rev-parse", "origin/master") == prereg["origin_master"],
        "harness_unchanged": _sha(HARNESS) == prereg["harness_sha256"],
        "inputs_unchanged": _input_hashes() == prereg["input_hashes"],
        "sidecar_unchanged": _sidecar_snapshot() == prereg["d334_sidecar_before"],
    }
    if not all(checks_pre.values()):
        raise RuntimeError(f"D372 run preflight failed: {checks_pre}")
    d368 = _load_module("d372_d368", D368_HARNESS)
    d371 = _load_module("d372_d371", D371_HARNESS)
    raw = d368._load_raw_meshes()
    patches = d371._raw_patch_context(raw, d368)
    current, current_inventory = d368._load_current_parts()
    candidate, construction, private = _build_candidate(raw, patches, d368)
    construction["_private"] = private
    prior_lineage = _prior_evidence_and_owner_contract(candidate)
    inventories = {body: _inventory(candidate[body]) for body in BODY_NAMES}
    seed = _seed_metrics(candidate, d368)
    representation = _representation_metrics(candidate, raw, patches, d368, d371)
    jaw_void_diagnostic = _jaw_contact_layer_void_diagnostic(candidate, construction)
    clearance = _clearance(candidate, current, d371)
    transition_replay = _immutable_d362_transition_replay(
        candidate,
        current,
        d371,
        prereg["input_hashes"][_rel(D362_TRACE)],
    )
    negatives = _negative_controls(candidate, raw, construction, prior_lineage, d368, d371)
    actual_counts = {body: inventories[body]["part_count"] for body in BODY_NAMES}
    actual_counts["total"] = sum(actual_counts.values())
    checks = {
        **checks_pre,
        "current_A64_lineage_pass": current_inventory["pass"],
        "prior_evidence_and_owner_lineage_pass": prior_lineage["pass"],
        "candidate_part_counts_exact": actual_counts == EXPECTED_PART_COUNTS,
        "candidate_role_counts_exact": all(inventories[body]["roles"] == EXPECTED_ROLE_COUNTS[body] for body in BODY_NAMES),
        "candidate_inventories_pass": all(row["pass"] for row in inventories.values()),
        "fixed_patch_pass": construction["fixed_patch"]["pass"],
        "moving_patch_pass": construction["moving_patch"]["pass"],
        "moving_inner_outer_projection_pass": construction["moving_inner_outer_projection"]["pass"],
        "required_semantic_base_zones_present": construction["semantic_zone_manifest"]["fixed_base_present"]
        and construction["semantic_zone_manifest"]["moving_base_present"],
        "raw_semantic_face_classification_partition_exact": all(
            row["pass"]
            for row in construction["raw_semantic_face_classification_partition"].values()
        ),
        "jaw_planes_and_thickness_exact": construction["jaw_plane_and_thickness"]["fixed_all_exact"]
        and construction["jaw_plane_and_thickness"]["moving_all_exact"]
        and abs(
            construction["jaw_plane_and_thickness"]["moving_thickness_m"] - JAW_THICKNESS_TARGET_M
        ) <= JAW_THICKNESS_TOL_M,
        "semantic_representation_metrics_pass": representation["pass"],
        "jaw_contact_layer_void_diagnostic_pass": jaw_void_diagnostic["pass"],
        "contact_seeds_retained": seed["pass"],
        "frozen_open_clearance_pass": clearance["pass"],
        "immutable_D362_transition_replay_pass": transition_replay["pass"],
        "negative_controls_5_of_5": negatives["pass"]
        and negatives["passed"] == negatives["total"] == 5,
        "forbidden_modules_absent_before_visualization": _forbidden_modules() == [],
        "d334_sidecar_unchanged_after_measurement": _sidecar_snapshot() == prereg["d334_sidecar_before"],
    }
    measurement_pass = all(checks.values())
    verdict = VERDICT_PASS if measurement_pass else VERDICT_FAIL
    candidate_payload = {
        "artifact": "D372_PROFESSOR_SEMANTIC_CANDIDATE_GEOMETRY_V1",
        "case": "g0a_d372",
        "attempt": ATTEMPT_NAME,
        "candidate": "P34_professor_semantic_compound",
        "owner_contract": prior_lineage["owner_contract"],
        "parts": {body: [_public_part(part) for part in candidate[body]] for body in BODY_NAMES},
        "authority": "Float64 vertices plus explicit triangle topology; no USD/live/PhysX authoring in D372",
    }
    _write_json_x(CANDIDATE_PATH, candidate_payload)
    evidence_construction = {key: value for key, value in construction.items() if key != "_private"}
    evidence = {
        "artifact": "D372_PROFESSOR_SEMANTIC_CANDIDATE_EVIDENCE_V1",
        "case": "g0a_d372",
        "attempt": ATTEMPT_NAME,
        "candidate": "P34_professor_semantic_compound",
        "verdict": verdict,
        "measurement_pass": measurement_pass,
        "new_variables": NEW_VARIABLES,
        "input_hashes": prereg["input_hashes"],
        "candidate_geometry": {"path": _rel(CANDIDATE_PATH), "sha256": _sha(CANDIDATE_PATH)},
        "candidate_inventory": inventories,
        "actual_part_counts": actual_counts,
        "current_A64_inventory": current_inventory,
        "prior_evidence_and_owner_lineage": prior_lineage,
        "nvidia_contract": prereg["nvidia_contract"],
        "construction": evidence_construction,
        "semantic_representation_metrics": representation,
        "jaw_contact_layer_void_diagnostic": jaw_void_diagnostic,
        "contact_seed_retention": seed,
        "frozen_open_clearance": clearance,
        "immutable_D362_transition_replay": transition_replay,
        "negative_controls": negatives,
        "checks": checks,
        "scope_guards": {
            "simulation_app_or_kit": 0,
            "isaac_or_physx": 0,
            "cook_or_automatic_decomposition": 0,
            "usd_or_live_asset_writes": 0,
            "physics_steps": 0,
            "q5_samples": 0,
            "live_contact_queries": 0,
            "target_ik_path_changes": 0,
            "material_mass_actuator_physics_changes": 0,
            "offline_hppfcl_static_part_queries": clearance["offline_part_query_count"]
            + transition_replay["offline_part_query_count"]
            + 1,
            "immutable_D362_trace_rows_read": transition_replay["row_count"],
            "run_invocation_count": 1,
            "automatic_retry_count": 0,
        },
        "interpretation_boundary": {
            "live_asset_identity": None,
            "actual_gpu_contact_execution": None,
            "physics_equivalence": None,
            "D362_replay_causal_equivalence": None,
            "runtime_speed": None,
            "tipping_causality": None,
            "grasp_feasibility": None,
            "global_optimum": None,
            "g0a_pass": False,
        },
    }
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase("measurement_committed", measurement_pass=measurement_pass, verdict=verdict)
    _write_report(evidence)
    boards = _render_boards(
        raw,
        candidate,
        patches,
        {**construction, "_private": private},
        clearance,
        transition_replay,
        jaw_void_diagnostic,
    )
    rerun = _write_rerun(raw, candidate, patches, evidence)
    post_visualization_forbidden_modules = _forbidden_modules()
    automated_checks = {
        "measurement_pass": measurement_pass,
        "three_1920x1080_boards": len(boards) == 3 and all(row["exact_1920x1080"] for row in boards.values()),
        "rerun_validation_pass": rerun["validation_pass"],
        "forbidden_modules_absent_after_visualization": post_visualization_forbidden_modules == [],
        "report_exists": REPORT_PATH.is_file(),
    }
    authority_artifacts = {
        _rel(path): {"bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in (CANDIDATE_PATH, EVIDENCE_PATH, REPORT_PATH, RERUN_VALIDATION_PATH)
    }
    automated = {
        "artifact": "D372_AUTOMATED_SUMMARY_V1",
        "attempt": ATTEMPT_NAME,
        "checks": automated_checks,
        "pass": all(automated_checks.values()),
        "boards": boards,
        "rerun": rerun,
        "post_visualization_forbidden_modules": post_visualization_forbidden_modules,
        "authority_artifacts": authority_artifacts,
        "manual_inspection_pending": True,
    }
    _write_json_x(AUTOMATED_PATH, automated)
    _phase("run_complete", measurement_pass=measurement_pass, automated_pass=automated["pass"])
    print(json.dumps({"stage": "run", "measurement_pass": measurement_pass, "verdict": verdict, "automated_pass": automated["pass"], "part_counts": actual_counts}, ensure_ascii=False))
    if not measurement_pass:
        raise RuntimeError(f"D372 measurement failed: {checks}")
    if not automated["pass"]:
        raise RuntimeError(f"D372 visualization contract failed after committed measurement: {automated_checks}")


def _finalize() -> None:
    if COMPLETION_PATH.exists():
        raise FileExistsError(COMPLETION_PATH)
    for path in [PREREG_PATH, INVOCATION_PATH, CANDIDATE_PATH, EVIDENCE_PATH, REPORT_PATH, OWNERSHIP_PNG, JAWS_PNG, CLEARANCE_PNG, RRD_PATH, RBL_PATH, RERUN_VALIDATION_PATH, RERUN_PNG, AUTOMATED_PATH, MANUAL_JSON_PATH, MANUAL_MD_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)
    prereg = _read_json(PREREG_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    automated = _read_json(AUTOMATED_PATH)
    manual = _read_json(MANUAL_JSON_PATH)
    manual_paths = {_rel(path) for path in (OWNERSHIP_PNG, JAWS_PNG, CLEARANCE_PNG, RERUN_PNG)}
    observed_paths = {row.get("path") for row in manual.get("files", [])}
    automated_board_hashes_current = all(
        (REPO / row["path"]).is_file() and _sha(REPO / row["path"]) == row["sha256"]
        for row in automated.get("boards", {}).values()
    )
    automated_rerun_hashes_current = (
        _sha(RRD_PATH) == automated.get("rerun", {}).get("rrd_sha256")
        and _sha(RBL_PATH) == automated.get("rerun", {}).get("rbl_sha256")
        and _sha(RERUN_PNG) == automated.get("rerun", {}).get("screenshot", {}).get("sha256")
    )
    authority_artifacts_current = all(
        (REPO / path).is_file()
        and _sha(REPO / path) == row.get("sha256")
        and (REPO / path).stat().st_size == row.get("bytes")
        for path, row in automated.get("authority_artifacts", {}).items()
    ) and len(automated.get("authority_artifacts", {})) == 4
    candidate_hash_matches_evidence = _sha(CANDIDATE_PATH) == evidence.get("candidate_geometry", {}).get("sha256")
    manual_files_current = all(
        (REPO / row["path"]).is_file()
        and _sha(REPO / row["path"]) == row.get("sha256")
        and (REPO / row["path"]).stat().st_size == row.get("bytes")
        for row in manual.get("files", [])
    )
    checks = {
        "measurement_pass": evidence.get("measurement_pass") is True and evidence.get("verdict") == VERDICT_PASS,
        "automated_pass": automated.get("pass") is True,
        "manual_inspection_performed": manual.get("inspection_performed") is True,
        "manual_pass": manual.get("pass") is True,
        "manual_exact_paths": observed_paths == manual_paths,
        "manual_file_hashes_and_sizes_current": manual_files_current and len(manual.get("files", [])) == 4,
        "manual_markdown_hash": manual.get("markdown_sha256") == _sha(MANUAL_MD_PATH),
        "automated_board_hashes_current": automated_board_hashes_current,
        "automated_rerun_hashes_current": automated_rerun_hashes_current,
        "authority_artifacts_current": authority_artifacts_current,
        "candidate_hash_matches_evidence": candidate_hash_matches_evidence,
        "single_run_no_retry": _read_json(INVOCATION_PATH).get("run_invocation_count") == 1 and _read_json(INVOCATION_PATH).get("automatic_retry_count") == 0,
        "head_origin_unchanged": _git("rev-parse", "HEAD") == _git("rev-parse", "origin/master") == prereg["head"],
        "frozen_inputs_unchanged": _input_hashes() == prereg["input_hashes"],
        "sidecar_unchanged": _sidecar_snapshot() == prereg["d334_sidecar_before"],
    }
    overall = all(checks.values())
    completion = {
        "artifact": "D372_COMPLETION_SUMMARY_V1",
        "case": "g0a_d372",
        "attempt": ATTEMPT_NAME,
        "measurement_verdict": evidence["verdict"],
        "completion_verdict": VERDICT_PASS if overall else VERDICT_VIZ_FAIL,
        "measurement_pass": evidence["measurement_pass"],
        "visualization_pass": automated["pass"] and manual.get("pass") is True,
        "pass": overall,
        "checks": checks,
        "part_counts": evidence["actual_part_counts"],
        "scope_guards": evidence["scope_guards"],
        "interpretation_boundary": evidence["interpretation_boundary"],
        "next_live_asset_or_physics_requires_new_approval": True,
        "g0a_pass": False,
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase("finalize_complete", passed=overall, completion_verdict=completion["completion_verdict"])
    print(json.dumps({"stage": "finalize", "pass": overall, "completion_verdict": completion["completion_verdict"]}, ensure_ascii=False))
    if not overall:
        raise RuntimeError(f"D372 finalize failed: {checks}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            _prepare()
        elif args.stage == "run":
            _run()
        else:
            _finalize()
        return 0
    except Exception as error:
        if OUT_DIR.is_dir() and not EXCEPTION_PATH.exists():
            preserved_measurement_pass = False
            preserved_measurement_verdict = None
            if EVIDENCE_PATH.is_file():
                try:
                    preserved = _read_json(EVIDENCE_PATH)
                    preserved_measurement_pass = preserved.get("measurement_pass") is True
                    preserved_measurement_verdict = preserved.get("verdict")
                except Exception:
                    preserved_measurement_pass = False
            _write_json_x(
                EXCEPTION_PATH,
                {
                    "artifact": "D372_RUNTIME_EXCEPTION_V1",
                    "stage": args.stage,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "measurement_evidence_exists": EVIDENCE_PATH.is_file(),
                    "preserved_measurement_pass": preserved_measurement_pass,
                    "preserved_measurement_verdict": preserved_measurement_verdict,
                    "verdict": VERDICT_VIZ_FAIL if preserved_measurement_pass else VERDICT_FAIL,
                    "g0a_pass": False,
                },
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
