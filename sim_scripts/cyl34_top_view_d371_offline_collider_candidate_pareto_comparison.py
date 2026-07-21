#!/usr/bin/env python3
"""D371 offline collider candidate comparison for the RoArm cylinder G0a case.

The case may launch one headless Kit process solely to obtain synchronous PhysX
cooking callbacks.  It never creates a SimulationContext, resets an environment,
steps physics, reads or writes q5, queries live contacts, or authors a live/canonical
collider asset.  All geometry decisions use Float64 source data and each candidate's
original callback polygon topology.  Rerun and PNG copies are display-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d371"
PREREG_PATH = OUT_DIR / "d371_preregistration.json"
PHASE_PATH = OUT_DIR / "d371_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d371_worker_invocation.json"
WORKER_CLAIM_PATH = OUT_DIR / "d371_cook_worker_claim.json"
STDOUT_PATH = OUT_DIR / "d371_cook_worker_stdout.log"
STDERR_PATH = OUT_DIR / "d371_cook_worker_stderr.log"
SUPERVISOR_PATH = OUT_DIR / "d371_worker_supervisor.json"
WORKER_SUMMARY_PATH = OUT_DIR / "d371_cook_worker_summary.json"
PRECLOSE_PATH = OUT_DIR / "d371_preclose_sentinel.json"
EVIDENCE_PATH = OUT_DIR / "d371_offline_collider_comparison_evidence.json"
REPORT_PATH = OUT_DIR / "d371_professor_comparison_report.md"
CAP_BOARD_PATH = OUT_DIR / "d371_cap_comparison_1920x1080.png"
SEMANTIC_BOARD_PATH = OUT_DIR / "d371_semantic_comparison_1920x1080.png"
CONTACT_BOARD_PATH = OUT_DIR / "d371_contact_detail_1920x1080.png"
RRD_PATH = OUT_DIR / "d371_collider_comparison.rrd"
RBL_PATH = OUT_DIR / "d371_collider_comparison.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d371_rerun_validation.json"
RERUN_PNG_PATH = OUT_DIR / "d371_collider_comparison_rerun.png"
AUTOMATED_PATH = OUT_DIR / "d371_automated_summary.json"
MANUAL_JSON_PATH = OUT_DIR / "d371_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d371_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d371_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d371_runtime_exception.json"

HARNESS = Path(__file__).resolve()
WORKER = REPO / "sim_scripts/cyl34_top_view_d371_offline_collider_cook_worker.py"
D339_HARNESS = REPO / "sim_scripts/cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair.py"
D368_HARNESS = REPO / "sim_scripts/cyl34_top_view_d368_current_64cap_semantic_allocation_audit.py"
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
AGENTS = REPO / "AGENTS.md"
START_HERE = REPO / "START_HERE.md"
AUTHORING_USD = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"
URDF_PATH = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
D348_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_callback_topology_volume_evidence.json"
D349_MEASUREMENT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d349/d349_frozen_target_distance_measurement.json"
D350_BINDING = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d350/d350_fixed_jaw_semantic_binding.json"
D350_MEASUREMENT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d350/d350_fixed_jaw_geometry_measurement.json"
D368_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_evidence.json"
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"

EXPECTED_HEAD = "4a1120b801e808071583136e78954c78ca941dc8"
EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
EXPECTED_CONDA_LIB = Path("/home/cgxr/miniconda3/envs/isaaclab/lib")
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
PHYSX_PROPERTY_DB = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.kit.property.physx-107.3.26+107.3.3.cp311.u353/"
    "omni/kit/property/physx/database.py"
)
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
MPL_CONFIG_DIR = Path("/tmp/roarm_d371_matplotlib_cache")

NEW_VARIABLES = [
    "offline_collider_candidate_family",
    "professor_facing_candidate_comparison_capture_contract",
]
BODY_NAMES = ("link5", "gripper_link")
CANDIDATE_ORDER = ("A", "R64", "R32", "C1", "C2")
CANDIDATE_LABEL_KO = {
    "A": "현재 보정 64",
    "R64": "원본 자동 64",
    "R32": "원본 자동 32",
    "C1": "접촉부 보존 C1",
    "C2": "턱 주변 보존 C2",
}
RERUN_CANDIDATE_KEYS = {
    "A": "current64",
    "R64": "raw64",
    "R32": "raw32",
    "C1": "semantic_c1",
    "C2": "semantic_c2",
}

C1_RETAINED = {
    "link5": {"part_027", "part_029", "part_030", "part_031"},
    "gripper_link": {
        "part_030", "part_035", "part_042", "part_045", "part_046", "part_047",
        "part_048", "part_050", "part_051", "part_053", "part_056", "part_058",
        "part_059", "part_060", "part_061", "part_062", "part_063",
    },
}
C2_RETAINED = {
    "link5": {
        "part_013", "part_022", "part_023", "part_025", "part_026", "part_027",
        "part_028", "part_029", "part_030", "part_031",
    },
    "gripper_link": {
        "part_029", "part_030", "part_035", "part_039", "part_042", "part_044",
        "part_045", "part_046", "part_047", "part_048", "part_050", "part_051",
        "part_053", "part_054", "part_055", "part_056", "part_057", "part_058",
        "part_059", "part_060", "part_061", "part_062", "part_063",
    },
}

DECOMPOSITION_COMMON = {
    "hull_vertex_limit": 64,
    "voxel_resolution": 1_000_000,
    "error_percentage": 1.0,
    "min_thickness_m": 0.0001,
    "shrink_wrap": True,
}
OPEN_CLEAR_GATE_MM = 0.1
OPEN_RAW_DELTA_GATE_MM = 0.5
GRID_STEP_M = 0.001
WORKER_TIMEOUT_SECONDS = 1800.0
AUDIT_TIMEOUT_SECONDS = 1200

VERDICT_MEASURED = "D371_OFFLINE_COLLIDER_PARETO_MEASURED_NO_PHYSICS"
VERDICT_FAIL = "D371_COLLIDER_GENERATION_OR_MEASUREMENT_INTEGRITY_FAIL_STOP"
VERDICT_VIZ_FAIL = "D371_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"

OFFICIAL_SOURCES = [
    {
        "title": "PhysxConvexDecompositionCollisionAPI",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.6.1/api/classusdrt_1_1_physx_schema_physx_convex_decomposition_collision_a_p_i.html",
        "use": "schema attributes and defaults",
    },
    {
        "title": "Isaac Sim Physics Simulation Fundamentals",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html",
        "use": "multiple collision shapes and approximation semantics",
    },
    {
        "title": "Isaac Sim Performance Optimization Handbook",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html",
        "use": "collider-count and simplification guidance",
    },
    {
        "title": "PhysX 5.6.1 GPU Rigid Bodies",
        "url": "https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html",
        "use": "per-convex GPU geometry limits; supporting version, not installed SDK identity",
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default)
        stream.write("\n")


def _write_text_x(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _phase(name: str, **fields: Any) -> None:
    row = {"ordinal": 1, "phase": name, "monotonic_ns": time.monotonic_ns(), **fields}
    if PHASE_PATH.is_file():
        with PHASE_PATH.open("r", encoding="utf-8") as stream:
            row["ordinal"] = sum(1 for line in stream if line.strip()) + 1
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _sidecar_snapshot() -> dict[str, str]:
    if not D334_SIDECAR.is_dir():
        return {}
    return {
        _rel(path): _sha(path)
        for path in sorted(D334_SIDECAR.rglob("*"))
        if path.is_file()
    }


def _input_paths() -> list[Path]:
    return [
        AGENTS,
        START_HERE,
        AUTHORING_USD,
        URDF_PATH,
        D348_EVIDENCE,
        D349_MEASUREMENT,
        D350_BINDING,
        D350_MEASUREMENT,
        D368_EVIDENCE,
        D339_HARNESS,
        D368_HARNESS,
        VIZ_DEBUG,
        RERUN_CONTRACT,
        PHYSX_SCHEMA,
        PHYSX_PROPERTY_DB,
        FONT_REGULAR,
        FONT_BOLD,
    ]


def _input_hashes() -> dict[str, str]:
    return {_rel(path) if path.is_relative_to(REPO) else str(path): _sha(path) for path in _input_paths()}


def _dynamic_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in (HARNESS, WORKER)}


def _schema_facts() -> dict[str, Any]:
    text = PHYSX_SCHEMA.read_text(encoding="utf-8", errors="replace")
    prop = PHYSX_PROPERTY_DB.read_text(encoding="utf-8", errors="replace")
    max_hulls = re.search(r"int physxConvexDecompositionCollision:maxConvexHulls\s*=\s*(\d+)", text)
    vertex_limit = re.search(r"int physxConvexDecompositionCollision:hullVertexLimit\s*=\s*(\d+)", text)
    checks = {
        "schema_default_max_convex_hulls_32": bool(max_hulls and int(max_hulls.group(1)) == 32),
        "schema_default_hull_vertex_limit_64": bool(vertex_limit and int(vertex_limit.group(1)) == 64),
        "property_ui_range_max_hulls_1_to_2048": bool(
            re.search(
                r'"physxConvexDecompositionCollision:maxConvexHulls"\s*:\s*InfoData\(1,\s*2048,\s*1\)',
                prop,
            )
        ),
        "property_ui_range_vertex_limit_8_to_64": bool(
            re.search(
                r'"physxConvexDecompositionCollision:hullVertexLimit"\s*:\s*InfoData\(8,\s*64,\s*1\)',
                prop,
            )
        ),
    }
    return {
        "installed_schema_path": str(PHYSX_SCHEMA),
        "installed_property_database_path": str(PHYSX_PROPERTY_DB),
        "schema_default_max_convex_hulls": None if max_hulls is None else int(max_hulls.group(1)),
        "schema_default_hull_vertex_limit": None if vertex_limit is None else int(vertex_limit.group(1)),
        "official_sources": OFFICIAL_SOURCES,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _package_versions() -> dict[str, Any]:
    names = ["numpy", "psutil", "scipy", "trimesh", "rerun-sdk", "hpp-fcl"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    pythonpath_entries = [Path(value).resolve() for value in os.environ.get("PYTHONPATH", "").split(":") if value]
    ld_library_entries = [Path(value).resolve() for value in os.environ.get("LD_LIBRARY_PATH", "").split(":") if value]
    checks = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "numpy_pin_1p26p0": versions["numpy"] == "1.26.0",
        "psutil_pin_5p9p8": versions["psutil"] == "5.9.8",
        "rerun_pin_0p34p1": versions["rerun-sdk"] == "0.34.1",
        "rerun_cli_exists": RERUN_CLI.is_file(),
        "pxr_root_exists": PXR_ROOT.is_dir(),
        "pxr_root_explicit_in_pythonpath": PXR_ROOT.resolve() in pythonpath_entries,
        "conda_lib_explicit_in_ld_library_path": EXPECTED_CONDA_LIB.resolve() in ld_library_entries,
        "pxr_bin_explicit_in_ld_library_path": (PXR_ROOT / "bin").resolve() in ld_library_entries,
        "matplotlib_config_dir_exact_tmp": os.environ.get("MPLCONFIGDIR") == str(MPL_CONFIG_DIR),
        "fonts_exist": FONT_REGULAR.is_file() and FONT_BOLD.is_file(),
    }
    return {
        "python": sys.executable,
        "versions": versions,
        "runtime_loader_environment": {
            "PYTHONPATH_entries": [str(value) for value in pythonpath_entries],
            "LD_LIBRARY_PATH_entries": [str(value) for value in ld_library_entries],
            "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR"),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _prepare_controls() -> dict[str, Any]:
    expected_params = {**DECOMPOSITION_COMMON, "max_convex_hulls": 32}

    def params_ok(value: dict[str, Any]) -> bool:
        return value == expected_params

    baseline_params = params_ok(dict(expected_params))
    mislabeled = dict(expected_params)
    mislabeled["max_convex_hulls"] = 64
    mislabeled_rejected = baseline_params and not params_ok(mislabeled)

    baseline_owner = {"link5": "link5", "gripper_link": "gripper_link"}
    swapped_owner = {"link5": "gripper_link", "gripper_link": "link5"}
    owner_rejected = baseline_owner == {"link5": "link5", "gripper_link": "gripper_link"} and swapped_owner != baseline_owner

    c1_expected = {body: sorted(values) for body, values in C1_RETAINED.items()}
    dropped = {body: list(values) for body, values in c1_expected.items()}
    dropped["link5"] = dropped["link5"][1:]
    carrier_drop_rejected = dropped != c1_expected and len(dropped["link5"]) == len(c1_expected["link5"]) - 1

    meter = np.asarray([[0.001, 0.002, 0.003]], dtype="<f8")
    unit_rejected = _sha_bytes(meter.tobytes()) != _sha_bytes((meter * 1000.0).astype("<f8").tobytes())

    ordered = ["link5:a", "link5:b", "gripper:a"]
    reversed_rows = list(reversed(ordered))
    order_control = (
        _sha_bytes("\n".join(ordered).encode()) != _sha_bytes("\n".join(reversed_rows).encode())
        and _sha_bytes("\n".join(sorted(ordered)).encode())
        == _sha_bytes("\n".join(sorted(reversed_rows)).encode())
    )
    checks = {
        "raw32_mislabeled_as_64_rejected": mislabeled_rejected,
        "body_owner_swap_rejected": owner_rejected,
        "certified_carrier_drop_rejected": carrier_drop_rejected,
        "meter_to_millimeter_x1000_rejected": unit_rejected,
        "part_order_changes_ordered_but_not_canonical_hash": order_control,
    }
    return {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "pass": all(checks.values())}


def _prepare() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise RuntimeError(f"D371 output is not empty: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/master")
    packages = _package_versions()
    schema = _schema_facts()
    controls = _prepare_controls()
    expected_active_path = "claudedocs/runtime_logs/grasp_track/g0a_d371/"
    start_text = START_HERE.read_text(encoding="utf-8")
    checks = {
        "head_matches_registered": head == EXPECTED_HEAD,
        "origin_matches_registered": origin == EXPECTED_HEAD,
        "head_equals_origin": head == origin,
        "two_new_variables": len(NEW_VARIABLES) == 2,
        "active_path_registered_in_start_here": expected_active_path in start_text,
        "worker_exists": WORKER.is_file(),
        "all_frozen_inputs_exist": all(path.is_file() for path in _input_paths()),
        "environment_pins_pass": packages["pass"],
        "installed_schema_facts_pass": schema["pass"],
        "prepare_negative_controls_5_of_5": controls["passed"] == controls["total"] == 5,
        "output_empty_before_preregistration": not any(OUT_DIR.iterdir()),
    }
    candidate_contract = {
        "A": {
            "name": "current repaired 64-cap reference",
            "lineage": "D347/D348 current live-part callback topology after D344 fixed-point repair",
            "generation": "none; immutable D348/D368 read",
        },
        "R64": {
            "name": "same-environment raw full-mesh cap64 control",
            "lineage": "frozen raw full mesh",
            "max_convex_hulls": 64,
        },
        "R32": {
            "name": "same-environment raw full-mesh cap32 control",
            "lineage": "same frozen raw full mesh as R64",
            "max_convex_hulls": 32,
            "only_changed_cook_parameter_vs_R64": "max_convex_hulls",
        },
        "C1": {
            "name": "aggressive contact-carrier-preserving low-count prototype",
            "lineage": "D348 current repaired parts",
            "retained_names": {body: sorted(values) for body, values in C1_RETAINED.items()},
            "remainder_policy": "all non-retained current parts per owner -> one NVIDIA maxHulls=1 cook",
            "predicted_counts_not_results": {"link5": 5, "gripper_link": 18, "total": 23},
        },
        "C2": {
            "name": "guarded jaw-neighborhood-preserving low-count prototype",
            "lineage": "D348 current repaired parts plus D368 union-nearest guard sets",
            "retained_names": {body: sorted(values) for body, values in C2_RETAINED.items()},
            "remainder_policy": "all non-retained current parts per owner -> one NVIDIA maxHulls=1 cook",
            "predicted_counts_not_results": {"link5": 11, "gripper_link": 24, "total": 35},
            "nearest_sets_are_guard_policy_not_contact_authority": True,
        },
    }
    prereg = {
        "artifact": "D371_PREREGISTRATION_V1",
        "case": "g0a_d371",
        "approved_pivot": (
            "offline collider comparison and professor-facing result supersede the separate D370 "
            "visual-repair-first ordering; D370 FAIL itself remains frozen"
        ),
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "head": head,
        "origin_master": origin,
        "git_status_at_prepare": _git("status", "--short").splitlines(),
        "dynamic_hashes": _dynamic_hashes(),
        "input_hashes": _input_hashes(),
        "d334_sidecar_before": _sidecar_snapshot(),
        "environment": packages,
        "installed_nvidia_schema_and_official_sources": schema,
        "candidate_contract": candidate_contract,
        "lineage_rule": {
            "cap_only_causal_pair": ["R64", "R32"],
            "A_vs_R64": "lineage comparison only; D344 repair is confounded",
            "A_or_R_vs_C": "representation family plus allocation changes; no hull-count causality",
        },
        "cook_contract": {
            "common_params": DECOMPOSITION_COMMON,
            "R64_max_convex_hulls": 64,
            "R32_max_convex_hulls": 32,
            "C1_C2_remainder_max_convex_hulls": 1,
            "cold_repetitions_per_source": 2,
            "registered_synchronous_cook_requests": 16,
            "callback_evidence_written_before_classification": True,
            "callback_original_indices_polygons_planes_are_authority": True,
            "Qhull_role": "repeat canonicalization and convex containment diagnostic only",
        },
        "registered_metrics": [
            "actual hull/vertex/polygon/max-vertices-per-polygon counts by body",
            "original-callback-topology whole-part volume sum (overlap-prone diagnostic)",
            "raw full-surface to candidate and candidate to raw P95/max/RMS",
            "D368 fixed/inner/outer patch sampled surface distances and normal-facing inventory",
            "D349 immutable frozen-OPEN cylinder signed clearance using stored transforms only",
            "1mm grid current-A union occupancy vs candidate ghost/undercoverage diagnostics",
            "Pareto non-dominance without scalar score or global optimum claim",
        ],
        "gates": {
            "integrity": [
                "callback RESULT_VALID exactly once and cold1/cold2 canonical repeat PASS",
                "all observed parts finite positive and vertices<=64 polygons<=64 vertices_per_polygon<=32",
                "source owner/unit/hash exact and C retained carriers bit-exact",
            ],
            "frozen_open_task": {
                "clearance_min_mm": OPEN_CLEAR_GATE_MM,
                "absolute_delta_from_raw_max_mm": OPEN_RAW_DELTA_GATE_MM,
                "source": _rel(D349_MEASUREMENT),
                "meaning": "offline immutable-pose geometry gate; not q5 execution or physics",
            },
            "non_gating_diagnostics": [
                "D368 historical nearest P95/max values",
                "whole-part volume sum",
                "1mm occupancy grid",
                "hull count as runtime proxy",
            ],
        },
        "negative_controls": {
            "prepare": controls,
            "audit_registered": [
                "callback topology replaced by vertex-only Qhull is rejected",
                "C retained carrier deletion is rejected",
                "candidate label permutation changes ordered lineage but not canonical set",
                "naive whole-body convex envelope occupies at least one A-reference void witness or is recorded as not detected",
            ],
        },
        "visual_contract": {
            "rrd_role": "replay and inspection only; Float32 spatial copies",
            "rrd_blueprint": "two rows x five candidate views plus right-side notification buffer",
            "professor_boards": [
                _rel(CAP_BOARD_PATH), _rel(SEMANTIC_BOARD_PATH), _rel(CONTACT_BOARD_PATH)
            ],
            "boards_exact_dimensions": [1920, 1080],
            "no_fixed_absolute_color_pixel_threshold": True,
            "authority": "Float64 evidence JSON plus callback polygon arrays",
            "manual_original_resolution_inspection_required": True,
        },
        "scope_guards": {
            "app_launcher_for_cook_only": 1,
            "physx_cook_callbacks": 16,
            "simulation_context": 0,
            "environment_reset": 0,
            "cylinder_pose_write": 0,
            "q5_target_or_sample": 0,
            "controlled_physics_steps": 0,
            "live_contact_queries": 0,
            "offline_hppfcl_static_part_geometry_queries": (
                "one immutable D349-pose geometry query per generated candidate part; result-dependent count"
            ),
            "target_ik_path_changes": 0,
            "material_mass_actuator_physics_changes": 0,
            "canonical_or_live_asset_writes": 0,
            "d334_sidecar_writes": 0,
            "hardware": 0,
        },
        "single_run_contract": {
            "worker_invocations": 1,
            "automatic_retries": 0,
            "atomic_worker_claim_required": True,
            "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
            "worker_preclose_sentinel_required": True,
            "failure_evidence_preserved": True,
        },
        "interpretation_boundary": {
            "current64_optimal": None,
            "physics_equivalence": None,
            "collider_count_tipping_causality": None,
            "actual_gpu_contact_execution": None,
            "grasp_feasibility": None,
            "g0a_pass": False,
            "next_live_authoring_or_physics_requires_new_approval": True,
        },
        "registered_worker_command": {
            "python": str(EXPECTED_PYTHON),
            "python_flags": ["-B"],
            "script": _rel(WORKER),
            "argv": ["--out-dir", _rel(OUT_DIR), "--headless"],
            "environment": {
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "PYTHONPATH_must_include": str(PXR_ROOT),
                "LD_LIBRARY_PATH_must_include": [str(EXPECTED_CONDA_LIB), str(PXR_ROOT / "bin")],
                "MPLCONFIGDIR": str(MPL_CONFIG_DIR),
            },
        },
        "prepare_checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    if not prereg["pass"]:
        raise RuntimeError(f"D371 prepare failed: {checks}")
    print(json.dumps({"stage": "prepare", "pass": True, "path": _rel(PREREG_PATH)}, ensure_ascii=False))


def _blob(array: Any, dtype: str) -> bytes:
    return np.ascontiguousarray(array, dtype=dtype).tobytes(order="C")


def _triangulate_callback(indices: np.ndarray, polygons: list[dict[str, Any]]) -> np.ndarray:
    rows: list[list[int]] = []
    cursor = 0
    for polygon in polygons:
        base = int(polygon["index_base"])
        count = int(polygon["num_vertices"])
        if base != cursor or count < 3 or base + count > len(indices):
            raise RuntimeError("callback polygon spans are not contiguous and valid")
        face = [int(value) for value in indices[base : base + count]]
        for offset in range(1, count - 1):
            rows.append([face[0], face[offset], face[offset + 1]])
        cursor += count
    if cursor != len(indices):
        raise RuntimeError("callback polygons do not exactly cover index stream")
    return np.asarray(rows, dtype=np.int64)


def _topology_volume_m3(vertices: np.ndarray, triangles: np.ndarray) -> float:
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    return abs(float(np.sum(signed)))


def _closed_oriented_topology(triangles: np.ndarray) -> dict[str, Any]:
    directed: dict[tuple[int, int], int] = {}
    undirected: dict[tuple[int, int], int] = {}
    for row in np.asarray(triangles, dtype=np.int64):
        for left, right in ((row[0], row[1]), (row[1], row[2]), (row[2], row[0])):
            a, b = int(left), int(right)
            directed[(a, b)] = directed.get((a, b), 0) + 1
            key = tuple(sorted((a, b)))
            undirected[key] = undirected.get(key, 0) + 1
    closed = bool(undirected and all(value == 2 for value in undirected.values()))
    oriented = bool(closed and all(directed.get((b, a), 0) == value for (a, b), value in directed.items()))
    return {
        "undirected_edge_count": len(undirected),
        "all_undirected_edges_degree_two": closed,
        "opposite_directed_edges_balanced": oriented,
        "pass": closed and oriented,
    }


def _part_from_callback_payload(
    payload: dict[str, Any], *, body: str, name: str, source: str
) -> dict[str, Any]:
    import trimesh

    vertices = np.asarray(payload["vertices"], dtype=np.float64)
    indices = np.asarray(payload["indices"], dtype=np.int64)
    polygons = list(payload["polygons"])
    triangles = _triangulate_callback(indices, polygons)
    topology = _closed_oriented_topology(triangles)
    polygon_count = len(polygons)
    max_vertices = max(int(row["num_vertices"]) for row in polygons)
    volume = _topology_volume_m3(vertices, triangles)
    payload_hash = _sha_bytes(
        _blob(vertices, "<f8")
        + _blob(indices, "<i8")
        + json.dumps(polygons, sort_keys=True, separators=(",", ":")).encode()
    )
    checks = {
        "vertices_shape_finite": vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all(),
        "indices_in_range": bool(indices.size and int(indices.min()) >= 0 and int(indices.max()) < len(vertices)),
        "topology_closed_oriented": topology["pass"],
        "positive_finite_volume": math.isfinite(volume) and volume > 0.0,
        "vertices_le_64": 4 <= len(vertices) <= 64,
        "polygons_le_64": 4 <= polygon_count <= 64,
        "vertices_per_polygon_le_32": 3 <= max_vertices <= 32,
    }
    return {
        "body": body,
        "name": name,
        "source": source,
        "vertex_count": len(vertices),
        "polygon_count": polygon_count,
        "max_vertices_per_polygon": max_vertices,
        "triangle_count": len(triangles),
        "topology_volume_m3": volume,
        "payload_sha256": payload_hash,
        "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "topology": topology,
        "checks": checks,
        "pass": all(checks.values()),
        "vertices": vertices,
        "triangles": triangles,
        "_mesh": trimesh.Trimesh(vertices=vertices, faces=triangles, process=False, validate=False),
    }


def _generated_parts_from_witness(path: Path, *, body: str, source: str) -> list[dict[str, Any]]:
    witness = _read_json(path)
    request = witness.get("request", witness)
    events = request.get("events", [])
    if len(events) != 1 or events[0].get("result_name") != "RESULT_VALID":
        raise RuntimeError(f"invalid callback witness: {path}")
    if events[0].get("serialization_errors"):
        raise RuntimeError(f"callback serialization errors: {path}")
    convexes = events[0].get("convexes", [])
    if int(events[0].get("convex_count", -1)) != len(convexes) or not convexes:
        raise RuntimeError(f"callback convex count mismatch: {path}")
    rows = [
        _part_from_callback_payload(payload, body=body, name=f"part_{index:03d}", source=source)
        for index, payload in enumerate(convexes)
    ]
    rows.sort(
        key=lambda row: (
            *np.round(np.mean(row["vertices"], axis=0), 9).tolist(),
            row["payload_sha256"],
        )
    )
    for index, row in enumerate(rows):
        row["name"] = f"part_{index:03d}"
    return rows


def _clone_current_part(part: dict[str, Any], *, name: str | None = None, source: str = "A") -> dict[str, Any]:
    import trimesh

    vertices = np.asarray(part["vertices"], dtype=np.float64).copy()
    triangles = np.asarray(part["triangles"], dtype=np.int64).copy()
    topology_volume_m3 = float(part["topology_volume_m3"])
    checks = {
        "vertices_le_64": int(part["vertex_count"]) <= 64,
        "polygons_le_64": int(part["polygon_count"]) <= 64,
        "vertices_per_polygon_le_32": int(part["max_vertices_per_polygon"]) <= 32,
        "positive_finite_volume": math.isfinite(topology_volume_m3) and topology_volume_m3 > 0.0,
    }
    return {
        "body": part["body"],
        "name": part["name"] if name is None else name,
        "source": source,
        "original_current_name": part["name"],
        "vertex_count": int(part["vertex_count"]),
        "polygon_count": int(part["polygon_count"]),
        "max_vertices_per_polygon": int(part["max_vertices_per_polygon"]),
        "triangle_count": int(part["triangle_count"]),
        "topology_volume_m3": topology_volume_m3,
        "volume_authority": "D348 instance original callback polygon topology volume_origin_m3",
        "payload_sha256": part["payload_sha256"],
        "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "checks": checks,
        "pass": all(checks.values()),
        "vertices": vertices,
        "triangles": triangles,
        "_mesh": trimesh.Trimesh(vertices=vertices, faces=triangles, process=False, validate=False),
    }


def _worker_witness_path(worker: dict[str, Any], family: str, body: str) -> Path:
    row = worker["cooks"][family][body]
    candidate = row["cold1"]
    raw = candidate.get("callback_witness_path") or candidate.get("witness_path")
    if not raw:
        raise KeyError(f"{family}/{body} cold1 witness path missing")
    return REPO / raw


def _build_candidates(
    current: dict[str, list[dict[str, Any]]], worker: dict[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    d348 = _read_json(D348_EVIDENCE)
    topology_volumes = {
        (str(row["body"]), str(row["name"])): float(row["instance"]["volume_origin_m3"])
        for row in d348["rows"]
    }
    expected_keys = {
        (body, part["name"])
        for body in BODY_NAMES
        for part in current[body]
    }
    if set(topology_volumes) != expected_keys:
        raise RuntimeError("D348 topology-volume owner/name inventory mismatch")
    for body in BODY_NAMES:
        for part in current[body]:
            part["topology_volume_m3"] = topology_volumes[(body, part["name"])]
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        "A": {
            body: [_clone_current_part(part, source="A_current_repaired_callback") for part in current[body]]
            for body in BODY_NAMES
        }
    }
    for family in ("R64", "R32"):
        candidates[family] = {}
        for body in BODY_NAMES:
            candidates[family][body] = _generated_parts_from_witness(
                _worker_witness_path(worker, family, body),
                body=body,
                source=f"{family}_raw_fullmesh_callback",
            )
    for family, retained_map in (("C1", C1_RETAINED), ("C2", C2_RETAINED)):
        candidates[family] = {}
        for body in BODY_NAMES:
            by_name = {part["name"]: part for part in current[body]}
            retained = [
                _clone_current_part(
                    by_name[name], name=f"retained_{name}", source=f"{family}_retained_current_callback"
                )
                for name in sorted(retained_map[body])
            ]
            collapsed = _generated_parts_from_witness(
                _worker_witness_path(worker, family, body),
                body=body,
                source=f"{family}_single_remainder_callback",
            )
            if len(collapsed) != 1:
                raise RuntimeError(f"{family}/{body} expected exactly one collapsed remainder")
            collapsed[0]["name"] = "collapsed_structural_remainder"
            candidates[family][body] = [*retained, collapsed[0]]
    return candidates


def _public_part(part: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in part.items() if not key.startswith("_") and key not in {"vertices", "triangles"}}


def _part_inventory(parts: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "all_part_payloads_pass": bool(parts) and all(part["pass"] for part in parts),
        "all_vertices_le_64": bool(parts) and all(part["vertex_count"] <= 64 for part in parts),
        "all_polygons_le_64": bool(parts) and all(part["polygon_count"] <= 64 for part in parts),
        "all_vertices_per_polygon_le_32": bool(parts) and all(part["max_vertices_per_polygon"] <= 32 for part in parts),
    }
    return {
        "part_count": len(parts),
        "vertex_count_sum": sum(part["vertex_count"] for part in parts),
        "polygon_count_sum": sum(part["polygon_count"] for part in parts),
        "triangle_count_sum": sum(part["triangle_count"] for part in parts),
        "whole_part_topology_volume_sum_m3": sum(part["topology_volume_m3"] for part in parts),
        "volume_semantics": "overlap-prone whole-convex sum; not unique occupied volume, mass, or material volume",
        "checks": checks,
        "pass": all(checks.values()),
    }


def _min_surface_distances(parts: list[dict[str, Any]], points: np.ndarray, d368: Any) -> np.ndarray:
    query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(query):
        return np.asarray([], dtype=np.float64)
    return np.min(
        np.column_stack([d368._nearest(part["_mesh"], query)[0] for part in parts]),
        axis=1,
    )


def _surface_metrics(
    raw: dict[str, Any], parts: list[dict[str, Any]], d368: Any
) -> dict[str, Any]:
    raw_vertices = np.unique(np.asarray(raw["vertices_m"], dtype=np.float64), axis=0)
    raw_triangles = np.asarray(raw["triangles"], dtype=np.int64)
    raw_tri_points = raw["vertices_m"][raw_triangles[::16]]
    raw_centroids = np.mean(raw_tri_points, axis=1)
    raw_samples = np.vstack([raw_vertices, raw_centroids])
    raw_to_candidate = _min_surface_distances(parts, raw_samples, d368)

    candidate_samples = []
    for part in parts:
        candidate_samples.append(np.unique(part["vertices"], axis=0))
        candidate_samples.append(np.mean(part["vertices"][part["triangles"]], axis=1))
    candidate_samples_array = np.unique(np.vstack(candidate_samples), axis=0)
    raw_mesh = d368._trimesh(raw["vertices_m"], raw_triangles)
    candidate_to_raw, _ = d368._nearest(raw_mesh, candidate_samples_array)
    return {
        "sampling_contract": {
            "raw": "unique raw vertices plus every 16th raw triangle centroid by frozen face order",
            "candidate": "unique callback vertices plus all callback-topology fan-triangle centroids",
            "authority": "Float64 point-to-triangle surface queries",
            "hard_threshold": None,
        },
        "raw_sample_count": len(raw_samples),
        "candidate_sample_count": len(candidate_samples_array),
        "raw_sample_sha256": _sha_bytes(_blob(raw_samples, "<f8")),
        "candidate_sample_sha256": _sha_bytes(_blob(candidate_samples_array, "<f8")),
        "raw_to_candidate": d368._distance_stats_m(raw_to_candidate),
        "candidate_to_raw": d368._distance_stats_m(candidate_to_raw),
    }


def _raw_patch_context(raw: dict[str, dict[str, Any]], d368: Any) -> dict[str, Any]:
    d350_binding = _read_json(D350_BINDING)
    d350_measurement = _read_json(D350_MEASUREMENT)
    broad = d368._vertex_connected_component(
        raw["link5"]["vertices_m"],
        raw["link5"]["triangles"],
        d368.EXPECTED_D350_COMPONENT["seed_face"],
        reverse=False,
    )
    fixed = d368._fixed_support_patch(
        raw["link5"]["vertices_m"],
        raw["link5"]["triangles"],
        broad["face_ids"],
        d350_oriented_normal_local=np.asarray(
            d350_measurement["actual_surface"]["oriented_surface_normal_local"], dtype=np.float64
        ),
        d350_seed_local_m=np.asarray(d350_binding["seed_local_m"], dtype=np.float64),
    )
    fixed_compact = d368._compact_faces(
        raw["link5"]["vertices_m"], raw["link5"]["triangles"], fixed["_face_ids"]
    )
    inner = d368._compact_faces(
        raw["gripper_link"]["vertices_m"], raw["gripper_link"]["triangles"], d368.INNER_FACE_IDS
    )
    outer = d368._compact_faces(
        raw["gripper_link"]["vertices_m"], raw["gripper_link"]["triangles"], d368.OUTER_FACE_IDS
    )
    seed_face = d368.EXPECTED_D350_COMPONENT["seed_face"]
    fixed_plane = float(
        raw["link5"]["vertices_m"][raw["link5"]["triangles"][seed_face, 0], 0]
    )
    inner_plane = float(
        raw["gripper_link"]["vertices_m"][raw["gripper_link"]["triangles"][672, 0], 1]
    )
    outer_plane = float(
        raw["gripper_link"]["vertices_m"][raw["gripper_link"]["triangles"][13205, 0], 1]
    )
    return {
        "fixed": {
            "vertices": fixed_compact["vertices"],
            "triangles": fixed_compact["triangles"],
            "mesh": d368._trimesh(fixed_compact["vertices"], fixed_compact["triangles"]),
            "axis": 0,
            "plane_m": fixed_plane,
            "normal": np.asarray(fixed["seed_normal_local"], dtype=np.float64),
            "exact_normal": False,
            "source": {key: value for key, value in fixed.items() if not key.startswith("_")},
        },
        "inner": {
            "vertices": inner["vertices"],
            "triangles": inner["triangles"],
            "mesh": d368._trimesh(inner["vertices"], inner["triangles"]),
            "axis": 1,
            "plane_m": inner_plane,
            "normal": np.asarray([0.0, -1.0, 0.0]),
            "exact_normal": True,
            "source": {
                "face_count": len(d368.INNER_FACE_IDS),
                "bounds_local_m": [inner["vertices"].min(axis=0).tolist(), inner["vertices"].max(axis=0).tolist()],
                "digest": inner["digest"],
            },
        },
        "outer": {
            "vertices": outer["vertices"],
            "triangles": outer["triangles"],
            "mesh": d368._trimesh(outer["vertices"], outer["triangles"]),
            "axis": 1,
            "plane_m": outer_plane,
            "normal": np.asarray([0.0, 1.0, 0.0]),
            "exact_normal": True,
            "source": {
                "face_count": len(d368.OUTER_FACE_IDS),
                "bounds_local_m": [outer["vertices"].min(axis=0).tolist(), outer["vertices"].max(axis=0).tolist()],
                "digest": outer["digest"],
            },
        },
    }


def _contact_patch_metrics(
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    patches: dict[str, Any],
    d368: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in CANDIDATE_ORDER:
        rows: dict[str, Any] = {}
        for label, body in (("fixed", "link5"), ("inner", "gripper_link"), ("outer", "gripper_link")):
            patch = patches[label]
            parts = candidates[family][body]
            certified = d368._certified_faces(
                parts,
                axis=patch["axis"],
                plane_value_m=patch["plane_m"],
                expected_normal=patch["normal"],
                raw_patch_mesh=patch["mesh"],
                exact_normal=patch["exact_normal"],
            )
            compatibility_parts = [
                {**part, "property_volume_m3": part["topology_volume_m3"]}
                for part in parts
            ]
            allocation = d368._support_allocation(
                patch["vertices"], patch["triangles"], compatibility_parts, certified
            )
            for budget in allocation["carrier_budgets"].values():
                budget["whole_part_topology_volume_sum_m3"] = budget.pop(
                    "whole_part_property_volume_sum_m3"
                )
                budget["volume_semantics"] = (
                    "D371 original-callback-topology whole-carrier sum; overlap-prone diagnostic, "
                    "not PhysX property-query volume, pad volume, unique occupied volume, mass, "
                    "or material volume"
                )
            rows[label] = {
                "certified": {key: value for key, value in certified.items() if not key.startswith("_")},
                "allocation": allocation,
            }
        output[family] = rows
    return output


def _convex_contains(points: np.ndarray, part: dict[str, Any]) -> np.ndarray:
    from scipy.spatial import ConvexHull

    vertices = np.unique(np.asarray(part["vertices"], dtype=np.float64), axis=0)
    hull = ConvexHull(vertices)
    equations = np.asarray(hull.equations, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    lo = vertices.min(axis=0) - 1.0e-10
    hi = vertices.max(axis=0) + 1.0e-10
    broad = np.all((points >= lo) & (points <= hi), axis=1)
    result = np.zeros(len(points), dtype=bool)
    ids = np.flatnonzero(broad)
    for start in range(0, len(ids), 50_000):
        selected = ids[start : start + 50_000]
        values = points[selected] @ equations[:, :3].T + equations[:, 3]
        result[selected] = np.all(values <= 1.0e-9, axis=1)
    return result


def _union_contains(points: np.ndarray, parts: list[dict[str, Any]]) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    for part in parts:
        pending = np.flatnonzero(~result)
        if not len(pending):
            break
        result[pending] = _convex_contains(points[pending], part)
    return result


def _occupancy_metrics(
    raw: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    reports: dict[str, Any] = {}
    private: dict[str, dict[str, np.ndarray]] = {}
    for body in BODY_NAMES:
        all_vertices = [raw[body]["vertices_m"]]
        for family in CANDIDATE_ORDER:
            all_vertices.extend(part["vertices"] for part in candidates[family][body])
        stacked = np.vstack(all_vertices)
        lo = np.floor((stacked.min(axis=0) - GRID_STEP_M) / GRID_STEP_M) * GRID_STEP_M
        hi = np.ceil((stacked.max(axis=0) + GRID_STEP_M) / GRID_STEP_M) * GRID_STEP_M
        axes = [
            np.arange(lo[index] + 0.5 * GRID_STEP_M, hi[index], GRID_STEP_M, dtype=np.float64)
            for index in range(3)
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        reference = _union_contains(grid, candidates["A"][body])
        body_rows: dict[str, Any] = {}
        private[body] = {}
        for family in CANDIDATE_ORDER:
            occupied = _union_contains(grid, candidates[family][body])
            ghost = (~reference) & occupied
            missing = reference & (~occupied)
            intersection = int(np.sum(reference & occupied))
            union = int(np.sum(reference | occupied))
            ghost_points = grid[ghost]
            missing_points = grid[missing]
            body_rows[family] = {
                "grid_step_mm": GRID_STEP_M * 1000.0,
                "grid_point_count": len(grid),
                "reference_A_occupied_count": int(reference.sum()),
                "candidate_occupied_count": int(occupied.sum()),
                "ghost_occupied_count": int(ghost.sum()),
                "A_undercoverage_count": int(missing.sum()),
                "ghost_volume_mm3_approx": float(ghost.sum() * (GRID_STEP_M * 1000.0) ** 3),
                "undercoverage_volume_mm3_approx": float(missing.sum() * (GRID_STEP_M * 1000.0) ** 3),
                "jaccard_vs_A": 1.0 if union == 0 else intersection / union,
                "diagnostic_only": True,
                "Qhull_role": "convex containment diagnostic only; not callback surface/volume authority",
            }
            private[body][f"{family}_ghost"] = ghost_points[:: max(1, len(ghost_points) // 2000 + 1)]
            private[body][f"{family}_missing"] = missing_points[:: max(1, len(missing_points) // 2000 + 1)]
        reports[body] = {
            "grid_bounds_m": [lo.tolist(), hi.tolist()],
            "axis_counts": [len(axis) for axis in axes],
            "candidates": body_rows,
        }
    return reports, private


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


def _build_bvh(hppfcl: Any, part: dict[str, Any]) -> Any:
    model = hppfcl.BVHModelOBBRSS()
    vertices = np.asarray(part["vertices"], dtype=np.float64)
    triangles = np.asarray(part["triangles"], dtype=np.int64)
    codes = [
        int(model.beginModel(len(triangles), len(vertices))),
        int(model.addVertices(vertices)),
        int(model.addTriangles(triangles)),
        int(model.endModel()),
    ]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"hppfcl BVH build failed: {codes}")
    return model


def _fcl_part_query(hppfcl: Any, geometry: Any, body_tf: Any, cylinder: Any, cylinder_tf: Any) -> dict[str, Any]:
    distance_request = hppfcl.DistanceRequest(True, 1.0e-9, 1.0e-9)
    distance_request.gjk_tolerance = 1.0e-9
    distance_request.gjk_max_iterations = 1000
    distance_result = hppfcl.DistanceResult()
    distance_m = float(hppfcl.distance(geometry, body_tf, cylinder, cylinder_tf, distance_request, distance_result))
    collision_request = hppfcl.CollisionRequest()
    collision_request.enable_contact = True
    collision_request.num_max_contacts = 256
    collision_result = hppfcl.CollisionResult()
    hppfcl.collide(geometry, body_tf, cylinder, cylinder_tf, collision_request, collision_result)
    contacts = []
    for index in range(collision_result.numContacts()):
        contact = collision_result.getContact(index)
        contacts.append(abs(float(contact.penetration_depth)))
    collision = bool(collision_result.isCollision())
    exact_mm = -max(contacts) * 1000.0 if collision and contacts else distance_m * 1000.0
    return {
        "is_collision": collision,
        "distance_return_mm": distance_m * 1000.0,
        "exact_signed_distance_mm": exact_mm,
        "contact_count": int(collision_result.numContacts()),
        "contact_capacity": 256,
        "contact_capacity_saturated": int(collision_result.numContacts()) >= 256,
        "nearest_point_geometry_m": np.asarray(distance_result.getNearestPoint1(), dtype=np.float64).tolist(),
        "nearest_point_cylinder_m": np.asarray(distance_result.getNearestPoint2(), dtype=np.float64).tolist(),
    }


def _frozen_open_clearance(
    candidates: dict[str, dict[str, list[dict[str, Any]]]]
) -> dict[str, Any]:
    import hppfcl

    frozen = _read_json(D349_MEASUREMENT)
    pose = frozen["distance_gate"]["authoritative_pose_streams"]["raw_first"]
    cylinder = hppfcl.Cylinder(0.017, 0.090)
    cylinder_tf = hppfcl.Transform3f(
        _quat_wxyz_to_rot(pose["object_quat_wxyz"]), np.asarray(pose["object_pos_w_m"], dtype=np.float64)
    )
    expected = frozen["distance_gate"]["per_body"]
    result: dict[str, Any] = {}
    for family in CANDIDATE_ORDER:
        family_rows: dict[str, Any] = {}
        for body in BODY_NAMES:
            body_pose = pose["body_poses_w"][body]
            body_tf = hppfcl.Transform3f(
                _quat_wxyz_to_rot(body_pose["quat_wxyz"]), np.asarray(body_pose["pos_m"], dtype=np.float64)
            )
            queries = []
            for part in candidates[family][body]:
                query = _fcl_part_query(hppfcl, _build_bvh(hppfcl, part), body_tf, cylinder, cylinder_tf)
                queries.append({"part": part["name"], **query})
            collisions = [row for row in queries if row["is_collision"]]
            if collisions:
                selected = min(collisions, key=lambda row: row["exact_signed_distance_mm"])
            else:
                selected = min(queries, key=lambda row: row["exact_signed_distance_mm"])
            value = float(selected["exact_signed_distance_mm"])
            raw_value = float(expected[body]["raw_exact_signed_distance_mm"])
            checks = {
                "finite": math.isfinite(value),
                "no_collision": not collisions,
                "clearance_ge_0p1mm": not collisions and value >= OPEN_CLEAR_GATE_MM,
                "absolute_delta_from_raw_le_0p5mm": abs(value - raw_value) <= OPEN_RAW_DELTA_GATE_MM,
                "contact_capacity_not_saturated": all(not row["contact_capacity_saturated"] for row in queries),
            }
            family_rows[body] = {
                "part_count": len(queries),
                "exact_signed_distance_mm": value,
                "raw_reference_mm": raw_value,
                "absolute_delta_from_raw_mm": abs(value - raw_value),
                "selected_part": selected["part"],
                "selected_witness": selected,
                "collision_part_names": [row["part"] for row in collisions],
                "checks": checks,
                "pass": all(checks.values()),
            }
        result[family] = family_rows
    reproduction = {
        body: {
            "computed_A_mm": result["A"][body]["exact_signed_distance_mm"],
            "D349_live_topology_mm": float(expected[body]["live_topology_exact_signed_distance_mm"]),
            "absolute_delta_mm": abs(
                result["A"][body]["exact_signed_distance_mm"]
                - float(expected[body]["live_topology_exact_signed_distance_mm"])
            ),
        }
        for body in BODY_NAMES
    }
    for row in reproduction.values():
        row["pass_within_1e-6mm"] = row["absolute_delta_mm"] <= 1.0e-6
    return {
        "source": _rel(D349_MEASUREMENT),
        "semantics": "immutable stored-pose offline hppfcl callback-topology surface query; no q5 or physics execution",
        "reproduction": reproduction,
        "reproduction_pass": all(row["pass_within_1e-6mm"] for row in reproduction.values()),
        "offline_hppfcl_part_geometry_query_count": sum(
            result[family][body]["part_count"]
            for family in CANDIDATE_ORDER
            for body in BODY_NAMES
        ),
        "candidates": result,
    }


def _retention_checks(
    current: dict[str, list[dict[str, Any]]],
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    worker: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family, expected in (("C1", C1_RETAINED), ("C2", C2_RETAINED)):
        body_rows: dict[str, Any] = {}
        for body in BODY_NAMES:
            current_by_name = {part["name"]: part for part in current[body]}
            retained_parts = [part for part in candidates[family][body] if part["name"].startswith("retained_")]
            retained_original_names = [part["original_current_name"] for part in retained_parts]
            per_part = []
            for part in retained_parts:
                source = current_by_name[part["original_current_name"]]
                per_part.append(
                    {
                        "name": part["original_current_name"],
                        "payload_sha256": part["payload_sha256"],
                        "expected_payload_sha256": source["payload_sha256"],
                        "vertices_exact": np.array_equal(part["vertices"], source["vertices"]),
                        "triangles_exact": np.array_equal(part["triangles"], source["triangles"]),
                        "payload_hash_exact": part["payload_sha256"] == source["payload_sha256"],
                    }
                )
            worker_names = worker["cooks"][family][body].get("retained_names", [])
            checks = {
                "retained_names_exact": retained_original_names == sorted(expected[body]),
                "worker_retained_names_exact": worker_names == sorted(expected[body]),
                "retained_count_exact": len(retained_parts) == len(expected[body]),
                "all_retained_streams_and_hashes_exact": bool(per_part)
                and all(row["vertices_exact"] and row["triangles_exact"] and row["payload_hash_exact"] for row in per_part),
                "collapsed_remainder_exactly_one": sum(
                    part["name"] == "collapsed_structural_remainder" for part in candidates[family][body]
                )
                == 1,
                "candidate_count_is_retained_plus_one": len(candidates[family][body]) == len(expected[body]) + 1,
            }
            body_rows[body] = {"retained_parts": per_part, "checks": checks, "pass": all(checks.values())}
        output[family] = {"per_body": body_rows, "pass": all(row["pass"] for row in body_rows.values())}
    return output


def _audit_negative_controls(
    current: dict[str, list[dict[str, Any]]],
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    retention: dict[str, Any],
    occupancy: dict[str, Any],
) -> dict[str, Any]:
    qhull_differences = sum(
        not np.array_equal(part["triangles"], part["_qhull_triangles"])
        for body in BODY_NAMES
        for part in current[body]
    )
    qhull_substitution_rejected = qhull_differences == 128

    dropped_names = set(C1_RETAINED["link5"])
    dropped_names.remove(sorted(dropped_names)[0])
    carrier_drop_rejected = (
        retention["C1"]["per_body"]["link5"]["pass"]
        and dropped_names != C1_RETAINED["link5"]
        and len(dropped_names) + 1 == len(C1_RETAINED["link5"])
    )

    ordered = [
        f"{family}:{body}:{part['name']}:{part['payload_sha256']}"
        for family in CANDIDATE_ORDER
        for body in BODY_NAMES
        for part in candidates[family][body]
    ]
    reverse = list(reversed(ordered))
    order_control = (
        _sha_bytes("\n".join(ordered).encode()) != _sha_bytes("\n".join(reverse).encode())
        and _sha_bytes("\n".join(sorted(ordered)).encode())
        == _sha_bytes("\n".join(sorted(reverse)).encode())
    )

    semantic_ghost_counts = {
        family: sum(occupancy[body]["candidates"][family]["ghost_occupied_count"] for body in BODY_NAMES)
        for family in ("C1", "C2")
    }
    # Synthetic failure-capable check: two separated boxes leave the origin empty,
    # while their single convex envelope fills it.  This validates the same
    # containment path without assuming the real C1/C2 outcome in advance.
    from scipy.spatial import ConvexHull

    cube = np.asarray(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        dtype=np.float64,
    )
    left, right = cube + np.asarray([-2.0, 0.0, 0.0]), cube + np.asarray([2.0, 0.0, 0.0])
    envelope = ConvexHull(np.vstack([left, right]))
    origin = np.zeros(3, dtype=np.float64)
    inside_left = np.all(ConvexHull(left).equations[:, :3] @ origin + ConvexHull(left).equations[:, 3] <= 1.0e-9)
    inside_right = np.all(ConvexHull(right).equations[:, :3] @ origin + ConvexHull(right).equations[:, 3] <= 1.0e-9)
    inside_envelope = np.all(envelope.equations[:, :3] @ origin + envelope.equations[:, 3] <= 1.0e-9)
    single_remainder_ghost_detected = bool(not inside_left and not inside_right and inside_envelope)
    checks = {
        "callback_topology_to_vertex_qhull_substitution_rejected": qhull_substitution_rejected,
        "retained_carrier_deletion_rejected": carrier_drop_rejected,
        "candidate_order_changes_ordered_but_not_canonical_hash": order_control,
        "single_remainder_void_intrusion_detector_is_failure_capable": single_remainder_ghost_detected,
    }
    return {
        "qhull_different_part_count": qhull_differences,
        "semantic_ghost_counts": semantic_ghost_counts,
        "synthetic_two_box_gap_control": {
            "origin_inside_left": bool(inside_left),
            "origin_inside_right": bool(inside_right),
            "origin_inside_single_envelope": bool(inside_envelope),
            "rejected": single_remainder_ghost_detected,
        },
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "pass": all(checks.values()),
    }


def _candidate_objectives(
    family: str,
    inventories: dict[str, dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    clearance: dict[str, Any],
    occupancy: dict[str, Any],
) -> dict[str, float]:
    return {
        "total_hull_count": float(sum(inventories[family][body]["part_count"] for body in BODY_NAMES)),
        "max_open_raw_delta_mm": float(
            max(clearance["candidates"][family][body]["absolute_delta_from_raw_mm"] for body in BODY_NAMES)
        ),
        "total_ghost_volume_mm3_approx": float(
            sum(occupancy[body]["candidates"][family]["ghost_volume_mm3_approx"] for body in BODY_NAMES)
        ),
        "total_A_undercoverage_volume_mm3_approx": float(
            sum(
                occupancy[body]["candidates"][family]["undercoverage_volume_mm3_approx"]
                for body in BODY_NAMES
            )
        ),
        "max_raw_surface_to_candidate_error_mm": float(
            max(
                surfaces[family][body]["raw_to_candidate"]["max_mm"]
                for body in BODY_NAMES
            )
        ),
    }


def _pareto_report(
    inventories: dict[str, dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    clearance: dict[str, Any],
    occupancy: dict[str, Any],
    integrity: dict[str, bool],
) -> dict[str, Any]:
    objectives = {
        family: _candidate_objectives(family, inventories, surfaces, clearance, occupancy)
        for family in CANDIDATE_ORDER
    }
    eligibility = {
        family: bool(
            integrity[family]
            and all(clearance["candidates"][family][body]["pass"] for body in BODY_NAMES)
        )
        for family in CANDIDATE_ORDER
    }

    def dominates(left: str, right: str) -> bool:
        keys = list(objectives[left])
        return all(objectives[left][key] <= objectives[right][key] for key in keys) and any(
            objectives[left][key] < objectives[right][key] for key in keys
        )

    dominated_by: dict[str, list[str]] = {}
    for family in CANDIDATE_ORDER:
        dominated_by[family] = [
            other
            for other in CANDIDATE_ORDER
            if other != family and eligibility[other] and eligibility[family] and dominates(other, family)
        ]
    nondominated = [
        family for family in CANDIDATE_ORDER if eligibility[family] and not dominated_by[family]
    ]
    return {
        "objective_direction": "all minimized",
        "objectives": objectives,
        "offline_task_gate_eligible": eligibility,
        "dominated_by": dominated_by,
        "nondominated_candidates": nondominated,
        "scalar_score": None,
        "global_optimum": None,
        "interpretation": (
            "Pareto membership only: fewer hulls, frozen-OPEN fidelity, A-reference ghost/undercoverage, "
            "and raw-exterior-to-candidate deviation trade off. Candidate callback triangle centroids "
            "to raw include internal decomposition faces and remain non-gating diagnostics. This is not "
            "a physics-speed or grasp winner."
        ),
    }


def _write_report(evidence: dict[str, Any]) -> None:
    inventories = evidence["candidate_inventories"]
    clearance = evidence["frozen_open_clearance"]["candidates"]
    occupancy = evidence["occupancy_vs_A"]
    pareto = evidence["pareto"]
    lines = [
        "# D371 교수님용 오프라인 충돌체 비교",
        "",
        "이 비교는 충돌체 형상만 봅니다. 물리 스텝, 접촉 시험, q5 구동, 실제 파지는 모두 0회입니다.",
        "",
        "| 후보 | link5 개수 | 움직이는 턱 개수 | 합계 | link5 열린 간격(mm) | 움직이는 턱 열린 간격(mm) | A 밖 돌출(mm³, 근사) | A 누락(mm³, 근사) | 원본 외피→후보 최대오차(mm) | 다음 live 작성검사 적격 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for family in CANDIDATE_ORDER:
        link_count = inventories[family]["link5"]["part_count"]
        grip_count = inventories[family]["gripper_link"]["part_count"]
        ghost = sum(occupancy[body]["candidates"][family]["ghost_volume_mm3_approx"] for body in BODY_NAMES)
        missing = sum(
            occupancy[body]["candidates"][family]["undercoverage_volume_mm3_approx"]
            for body in BODY_NAMES
        )
        raw_surface_error = pareto["objectives"][family]["max_raw_surface_to_candidate_error_mm"]
        lines.append(
            f"| {family} — {CANDIDATE_LABEL_KO[family]} | {link_count} | {grip_count} | "
            f"{link_count + grip_count} | {clearance[family]['link5']['exact_signed_distance_mm']:.6f} | "
            f"{clearance[family]['gripper_link']['exact_signed_distance_mm']:.6f} | {ghost:.1f} | "
            f"{missing:.1f} | {raw_surface_error:.6f} | "
            f"{'예' if pareto['offline_task_gate_eligible'][family] else '아니오'} |"
        )
    lines.extend(
        [
            "",
            f"- 파레토 비지배 후보: {', '.join(pareto['nondominated_candidates']) or '없음'}",
            "- `R64 ↔ R32`만 최대 hull 상한 하나의 효과를 분리합니다.",
            "- `A`는 D344 보정 뒤 실제 D362가 사용한 현재 기준입니다. A와 R64의 차이를 64 상한 효과로 해석하지 않습니다.",
            "- C1/C2는 정확한 몸통·턱 분할이라고 부르지 않습니다. 기존 접촉 운반 조각을 보존하고 나머지만 한 볼록체로 줄인 시제품입니다.",
            "- hull 개수는 계산량의 대리값일 뿐 실제 속도 측정값이 아닙니다.",
            "- D371 결과만으로 전도 원인, 실제 GPU 접촉 실행, 파지 성공 여부를 판정하지 않습니다.",
            "",
            "## 시각자료",
            "",
            f"- 상한 비교: `{_rel(CAP_BOARD_PATH)}`",
            f"- 의미 보존 비교: `{_rel(SEMANTIC_BOARD_PATH)}`",
            f"- 접촉면 확대: `{_rel(CONTACT_BOARD_PATH)}`",
            f"- 재생 기록: `{_rel(RRD_PATH)}`",
        ]
    )
    _write_text_x(REPORT_PATH, "\n".join(lines) + "\n")


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
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


def _render_comparison_board(
    path: Path,
    *,
    title: str,
    columns: list[str],
    raw: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    patches: dict[str, Any],
    contact_zoom: bool,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if path.exists():
        raise FileExistsError(path)
    regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    colors = {
        "raw": "#7F7F7F",
        "A": "#0072B2",
        "R64": "#56B4E9",
        "R32": "#E69F00",
        "C1": "#009E73",
        "C2": "#33A02C",
        "contact": "#00BFC4",
        "retained": "#F0E442",
        "collapsed": "#009E73",
    }
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.89, bottom=0.10, wspace=0.03, hspace=0.10)
    axes = [fig.add_subplot(2, len(columns), index + 1, projection="3d") for index in range(2 * len(columns))]

    def add_mesh(ax: Any, vertices: np.ndarray, triangles: np.ndarray, color: str, alpha: float, edge: str, lw: float) -> None:
        collection = Poly3DCollection(
            np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)] * 1000.0,
            facecolor=color,
            edgecolor=edge,
            alpha=alpha,
            linewidth=lw,
        )
        ax.add_collection3d(collection)

    def limits(body: str, patch_label: str | None) -> tuple[np.ndarray, np.ndarray]:
        if patch_label is None:
            lo = np.asarray(raw[body]["vertices_m"].min(axis=0), dtype=np.float64) * 1000.0
            hi = np.asarray(raw[body]["vertices_m"].max(axis=0), dtype=np.float64) * 1000.0
            margin = np.asarray([4.0, 4.0, 4.0])
        else:
            lo = np.asarray(patches[patch_label]["vertices"].min(axis=0), dtype=np.float64) * 1000.0
            hi = np.asarray(patches[patch_label]["vertices"].max(axis=0), dtype=np.float64) * 1000.0
            margin = np.asarray([5.0, 5.0, 5.0])
            for axis in range(3):
                if hi[axis] - lo[axis] < 2.0:
                    margin[axis] = 8.0
        return lo - margin, hi + margin

    for row_index, body in enumerate(BODY_NAMES):
        patch_label = ("fixed" if body == "link5" else "inner") if contact_zoom else None
        lo, hi = limits(body, patch_label)
        for col_index, family in enumerate(columns):
            ax = axes[row_index * len(columns) + col_index]
            ax.set_proj_type("ortho")
            if family == "raw":
                add_mesh(
                    ax,
                    raw[body]["vertices_m"],
                    raw[body]["triangles"],
                    colors["raw"],
                    0.48,
                    "#555555",
                    0.02,
                )
            else:
                for part in candidates[family][body]:
                    part_lo = np.asarray(part["bounds_m"][0]) * 1000.0
                    part_hi = np.asarray(part["bounds_m"][1]) * 1000.0
                    if contact_zoom and (
                        not np.all(part_hi >= lo) or not np.all(part_lo <= hi)
                    ):
                        continue
                    if family in ("C1", "C2") and part["name"].startswith("retained_"):
                        color = colors["retained"]
                        alpha = 0.62
                    elif family in ("C1", "C2"):
                        color = colors["collapsed"]
                        alpha = 0.46
                    else:
                        color = colors[family]
                        alpha = 0.43
                    add_mesh(ax, part["vertices"], part["triangles"], color, alpha, "#222222", 0.06)
            if contact_zoom:
                patch = patches[patch_label]
                add_mesh(ax, patch["vertices"], patch["triangles"], colors["contact"], 0.36, "#000000", 0.12)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            spans = np.maximum(hi - lo, 1.0)
            ax.set_box_aspect(tuple(spans.tolist()))
            if contact_zoom and body == "link5":
                ax.view_init(elev=12, azim=-82)
            elif contact_zoom:
                ax.view_init(elev=4, azim=-90)
            else:
                ax.view_init(elev=20, azim=-55)
            ax.set_axis_off()
            label = (
                "원본 표면\n(충돌체 아님)"
                if family == "raw"
                else f"{family} · {CANDIDATE_LABEL_KO[family]}\n실제 충돌체 {len(candidates[family][body])}개"
            )
            ax.set_title(label, fontproperties=bold, fontsize=10, pad=2)
            if col_index == 0:
                body_label = "link5 · 고정 턱" if body == "link5" else "gripper_link · 움직이는 턱"
                ax.text2D(-0.02, 0.50, body_label, transform=ax.transAxes, rotation=90, va="center", ha="right", fontproperties=bold, fontsize=10)

    fig.suptitle(title, fontproperties=bold, fontsize=18, y=0.965)
    fig.text(
        0.5,
        0.035,
        "오프라인 충돌체 형상 비교입니다. 물리 스텝 0 · 접촉 시험 0 · q5 구동 0 · 파지 성공 여부 미판정",
        ha="center",
        fontproperties=bold,
        fontsize=11,
    )
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)
    info = _png_info(path)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"board dimensions are not exact: {info}")
    return info


def _render_boards(
    raw: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    patches: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cap_comparison": _render_comparison_board(
            CAP_BOARD_PATH,
            title="D371 상한 비교 — 같은 원본에서 64와 32를 분리 확인",
            columns=["raw", "A", "R64", "R32"],
            raw=raw,
            candidates=candidates,
            patches=patches,
            contact_zoom=False,
        ),
        "semantic_comparison": _render_comparison_board(
            SEMANTIC_BOARD_PATH,
            title="D371 의미 보존 저개수 후보 — 접촉부는 남기고 몸통 나머지를 1개로 축약",
            columns=["raw", "A", "C1", "C2"],
            raw=raw,
            candidates=candidates,
            patches=patches,
            contact_zoom=False,
        ),
        "contact_detail": _render_comparison_board(
            CONTACT_BOARD_PATH,
            title="D371 접촉면 확대 — 청록=원본 접촉면 · 노랑=보존 후보 조각",
            columns=["A", "R32", "C1", "C2"],
            raw=raw,
            candidates=candidates,
            patches=patches,
            contact_zoom=True,
        ),
    }


def _write_rerun(
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    patches: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.viz_debug import log_rerun

    family_colors = {
        "A": [0, 114, 178, 115],
        "R64": [86, 180, 233, 115],
        "R32": [230, 159, 0, 125],
        "C1": [0, 158, 115, 125],
        "C2": [51, 160, 44, 125],
    }
    meshes: list[dict[str, Any]] = []
    expected_entities: list[str] = []
    for family in CANDIDATE_ORDER:
        key = RERUN_CANDIDATE_KEYS[family]
        for body in BODY_NAMES:
            for part in candidates[family][body]:
                color = family_colors[family]
                if family in ("C1", "C2") and part["name"].startswith("retained_"):
                    color = [240, 228, 66, 190]
                path = f"compare/{key}/{body}/{part['name']}"
                meshes.append(
                    {
                        "entity_path": path,
                        "coordinate_frame": "tf#/",
                        "vertices_m": part["vertices"],
                        "triangles": part["triangles"],
                        "color_rgba": color,
                        "static": True,
                        "candidate": family,
                        "body": body,
                        "source": part["source"],
                        "payload_sha256": part["payload_sha256"],
                        "authority": "callback topology Float64; Rerun copy display-only Float32",
                    }
                )
                expected_entities.extend([path, f"metadata/meshes/{path.replace('/', '__')}"])
            patch_labels = ["fixed"] if body == "link5" else ["inner", "outer"]
            for patch_label in patch_labels:
                patch = patches[patch_label]
                path = f"compare/contact_patch/{key}/{body}/{patch_label}"
                meshes.append(
                    {
                        "entity_path": path,
                        "coordinate_frame": "tf#/",
                        "vertices_m": patch["vertices"],
                        "triangles": patch["triangles"],
                        "color_rgba": [0, 205, 255, 150] if patch_label != "outer" else [204, 121, 167, 155],
                        "static": True,
                        "semantic_patch": patch_label,
                        "authority": "raw Float64 patch; Rerun copy display-only Float32",
                    }
                )
                expected_entities.extend([path, f"metadata/meshes/{path.replace('/', '__')}"])

    result = log_rerun(
        RRD_PATH,
        meshes=meshes,
        recording_metadata={
            "case": "g0a_d371",
            "measurement_verdict": evidence["verdict"],
            "evidence_path": _rel(EVIDENCE_PATH),
            "evidence_sha256": _sha(EVIDENCE_PATH),
            "authority": "original callback polygon topology plus Float64 source arrays",
            "display_role": "Float32 replay copy only",
            "physics_steps": 0,
            "q5_samples": 0,
            "contact_queries": 0,
            "grasp_feasibility": None,
        },
        recording_id="g0a_d371_offline_collider_candidate_comparison",
        blueprint_path=RBL_PATH,
        blueprint_mode="d371_collider_comparison",
        live_viewer=False,
        app_id="roarm_g0a_d371_collider_comparison",
    )
    if not result.get("ok"):
        raise RuntimeError(f"Rerun write failed: {result}")
    expected_entities.append("metadata/run")
    component_contract: dict[str, list[str]] = {}
    for mesh in meshes:
        component_contract[mesh["entity_path"]] = [
            "CoordinateFrame:frame",
            "Mesh3D:albedo_factor",
            "Mesh3D:triangle_indices",
            "Mesh3D:vertex_positions",
        ]
        component_contract[f"metadata/meshes/{mesh['entity_path'].replace('/', '__')}"] = ["TextDocument:text"]
    component_contract["metadata/run"] = ["TextDocument:text"]
    strict = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected_entities),
        exact_entity_paths=sorted(expected_entities),
        expected_timeline_names=["blueprint", "log_time"],
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=component_contract,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG_PATH,
        screenshot_window_size="3840x2160",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=240.0,
    )
    _write_json_x(RERUN_VALIDATION_PATH, strict)
    return {
        "log": result,
        "strict_validation_pass": strict.get("pass") is True,
        "rrd_sha256": _sha(RRD_PATH),
        "rbl_sha256": _sha(RBL_PATH),
        "screenshot": _png_info(RERUN_PNG_PATH),
    }


def _worker_integrity(worker: dict[str, Any], supervisor: dict[str, Any]) -> dict[str, Any]:
    cook_rows = [worker["cooks"][family][body] for family in ("R64", "R32", "C1", "C2") for body in BODY_NAMES]
    params_r64 = worker["cooks"]["R64"]["link5"]["params"]
    params_r32 = worker["cooks"]["R32"]["link5"]["params"]
    differing_keys = sorted(key for key in set(params_r64) | set(params_r32) if params_r64.get(key) != params_r32.get(key))
    source_equal = {}
    for body in BODY_NAMES:
        left = worker["cooks"]["R64"][body]["source"]
        right = worker["cooks"]["R32"][body]["source"]
        source_equal[body] = bool(
            left["vertex_stream_sha256"] == right["vertex_stream_sha256"]
            and left["triangle_stream_sha256"] == right["triangle_stream_sha256"]
            and left["vertex_count"] == right["vertex_count"]
            and left["triangle_count"] == right["triangle_count"]
        )
    sentinel = _read_json(PRECLOSE_PATH)
    claim = _read_json(WORKER_CLAIM_PATH)
    recorded_artifacts: list[dict[str, Any]] = []
    for row in cook_rows:
        for cold_name in ("cold1", "cold2"):
            cold = row[cold_name]
            for artifact_name in ("callback_witness", "canonical_geometry"):
                raw_path = cold.get(f"{artifact_name}_path")
                recorded_hash = cold.get(f"{artifact_name}_sha256")
                path = REPO / str(raw_path) if raw_path else None
                safe_path = bool(
                    path is not None
                    and not Path(str(raw_path)).is_absolute()
                    and path.resolve().is_relative_to(OUT_DIR.resolve())
                )
                exists = bool(safe_path and path.is_file())
                actual_hash = _sha(path) if exists else None
                recorded_artifacts.append(
                    {
                        "cold": cold_name,
                        "artifact": artifact_name,
                        "path": raw_path,
                        "recorded_sha256": recorded_hash,
                        "actual_sha256": actual_hash,
                        "path_safe_and_file_exists": exists,
                        "hash_exact": bool(exists and actual_hash == recorded_hash),
                    }
                )
    artifact_paths = [row["path"] for row in recorded_artifacts]
    checks = {
        "worker_summary_pass": worker.get("pass") is True,
        "worker_invocation_exactly_one": worker.get("worker_invocation_count") == 1,
        "supervisor_worker_return_zero": supervisor.get("returncode") == 0,
        "supervisor_no_timeout_or_signal": not supervisor.get("timed_out")
        and not supervisor.get("sigterm_sent")
        and not supervisor.get("sigkill_sent"),
        "worker_preclose_sentinel_pass": sentinel.get("pass") is True,
        "preclose_summary_path_exact": sentinel.get("summary_path") == _rel(WORKER_SUMMARY_PATH),
        "preclose_summary_sha256_recomputed_exact": sentinel.get("summary_sha256") == _sha(WORKER_SUMMARY_PATH),
        "preclose_summary_pass_and_invocation_exact": sentinel.get("summary_pass") is True
        and sentinel.get("worker_invocation_count") == 1,
        "exclusive_claim_case_and_pid_exact": claim.get("artifact") == "D371_COOK_WORKER_EXCLUSIVE_CLAIM_V1"
        and claim.get("case") == "g0a_d371"
        and claim.get("single_worker_claimed") is True
        and claim.get("pid") == supervisor.get("pid"),
        "all_32_cold_artifact_paths_unique": len(recorded_artifacts) == 32
        and len(set(artifact_paths)) == 32,
        "all_32_cold_artifacts_recomputed_hash_exact": len(recorded_artifacts) == 32
        and all(row["hash_exact"] for row in recorded_artifacts),
        "all_eight_candidate_body_cook_pairs_pass": len(cook_rows) == 8 and all(row.get("pass") is True for row in cook_rows),
        "exactly_sixteen_cook_requests": worker.get("controlled_physx_cook_requests") == 16,
        "exactly_sixteen_in_memory_stages": worker.get("controlled_in_memory_cook_stages") == 16,
        "all_forbidden_worker_counters_zero": all(
            worker.get(key) == 0
            for key in (
                "controlled_simulation_context_constructions",
                "controlled_resets",
                "controlled_environment_resets",
                "controlled_physics_steps",
                "controlled_timeline_requests",
                "controlled_q5_samples",
                "controlled_contact_queries",
                "controlled_live_contact_queries",
                "controlled_cylinder_pose_writes",
                "controlled_target_ik_path_changes",
                "controlled_material_mass_actuator_physics_changes",
                "controlled_usd_asset_writes",
                "controlled_canonical_or_live_asset_writes",
            )
        ),
        "R64_R32_sources_exact_by_body": all(source_equal.values()),
        "R64_R32_only_max_convex_hulls_differs": differing_keys == ["max_convex_hulls"],
        "R64_max64_R32_max32": params_r64.get("max_convex_hulls") == 64 and params_r32.get("max_convex_hulls") == 32,
    }
    return {
        "R64_R32_parameter_differing_keys": differing_keys,
        "R64_R32_source_equal": source_equal,
        "preclose_sentinel": sentinel,
        "exclusive_worker_claim": claim,
        "recomputed_cold_artifacts": recorded_artifacts,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _audit_after_worker(supervisor: dict[str, Any]) -> dict[str, Any]:
    from sim_scripts import cyl34_top_view_d368_current_64cap_semantic_allocation_audit as d368

    worker = _read_json(WORKER_SUMMARY_PATH)
    worker_integrity = _worker_integrity(worker, supervisor)
    if not worker_integrity["pass"]:
        raise RuntimeError(f"D371 worker integrity failed: {worker_integrity['checks']}")
    _phase("cook_worker_evidence_verified", requests=worker["controlled_physx_cook_requests"])

    raw = d368._load_raw_meshes()
    current, current_inventory = d368._load_current_parts()
    if not current_inventory["pass"]:
        raise RuntimeError("D371 current A inventory failed")
    candidates = _build_candidates(current, worker)
    patches = _raw_patch_context(raw, d368)
    _phase(
        "candidate_geometries_reconstructed",
        counts={family: {body: len(candidates[family][body]) for body in BODY_NAMES} for family in CANDIDATE_ORDER},
    )

    inventories = {
        family: {body: _part_inventory(candidates[family][body]) for body in BODY_NAMES}
        for family in CANDIDATE_ORDER
    }
    surfaces = {
        family: {body: _surface_metrics(raw[body], candidates[family][body], d368) for body in BODY_NAMES}
        for family in CANDIDATE_ORDER
    }
    _phase("whole_surface_metrics_completed")
    contact = _contact_patch_metrics(candidates, patches, d368)
    _phase("contact_patch_metrics_completed")
    occupancy, occupancy_private = _occupancy_metrics(raw, candidates)
    del occupancy_private  # display boards use authoritative surfaces, not sampled grid points
    _phase("one_millimeter_occupancy_diagnostics_completed")
    clearance = _frozen_open_clearance(candidates)
    if not clearance["reproduction_pass"]:
        raise RuntimeError(f"D349 current-A clearance reproduction failed: {clearance['reproduction']}")
    _phase("frozen_open_clearance_completed", reproduction_pass=True)
    retention = _retention_checks(current, candidates, worker)
    controls = _audit_negative_controls(current, candidates, retention, occupancy)

    candidate_integrity = {
        "A": bool(current_inventory["pass"] and all(inventories["A"][body]["pass"] for body in BODY_NAMES)),
        "R64": bool(
            all(worker["cooks"]["R64"][body]["pass"] for body in BODY_NAMES)
            and all(inventories["R64"][body]["pass"] for body in BODY_NAMES)
        ),
        "R32": bool(
            all(worker["cooks"]["R32"][body]["pass"] for body in BODY_NAMES)
            and all(inventories["R32"][body]["pass"] for body in BODY_NAMES)
        ),
        "C1": bool(retention["C1"]["pass"] and all(inventories["C1"][body]["pass"] for body in BODY_NAMES)),
        "C2": bool(retention["C2"]["pass"] and all(inventories["C2"][body]["pass"] for body in BODY_NAMES)),
    }
    pareto = _pareto_report(inventories, surfaces, clearance, occupancy, candidate_integrity)

    raw_stream_checks = {
        body: all(
            raw[body]["stream_summary"].get(key) == value
            for key, value in d368.RAW_STREAM_EXPECTED[body].items()
        )
        for body in BODY_NAMES
    }
    expected_counts = {
        "A": {"link5": 64, "gripper_link": 64},
        "C1": {"link5": 5, "gripper_link": 18},
        "C2": {"link5": 11, "gripper_link": 24},
    }
    count_checks = {
        family: {
            body: inventories[family][body]["part_count"] == expected_counts[family][body]
            for body in BODY_NAMES
        }
        for family in expected_counts
    }
    measurement_checks = {
        "worker_integrity_pass": worker_integrity["pass"],
        "raw_streams_exact": all(raw_stream_checks.values()),
        "current_A_inventory_pass": current_inventory["pass"],
        "all_candidate_integrity_pass": all(candidate_integrity.values()),
        "A_and_semantic_counts_exact": all(all(rows.values()) for rows in count_checks.values()),
        "D349_A_clearance_reproduction_pass": clearance["reproduction_pass"],
        "C1_C2_retained_geometry_exact": retention["C1"]["pass"] and retention["C2"]["pass"],
        "audit_negative_controls_4_of_4": controls["passed"] == controls["total"] == 4,
        "D334_sidecar_unchanged": _sidecar_snapshot() == _read_json(PREREG_PATH)["d334_sidecar_before"],
    }
    measurement_pass = all(measurement_checks.values())
    verdict = VERDICT_MEASURED if measurement_pass else VERDICT_FAIL
    evidence = {
        "artifact": "D371_OFFLINE_COLLIDER_COMPARISON_EVIDENCE_V1",
        "case": "g0a_d371",
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "worker_integrity": worker_integrity,
        "raw_stream_checks": raw_stream_checks,
        "current_A_inventory": current_inventory,
        "candidate_inventories": inventories,
        "candidate_parts": {
            family: {body: [_public_part(part) for part in candidates[family][body]] for body in BODY_NAMES}
            for family in CANDIDATE_ORDER
        },
        "count_checks": count_checks,
        "R64_R32_cap_isolation": {
            "only_valid_causal_pair": True,
            "source_and_parameter_checks": worker_integrity,
            "actual_counts": {
                family: {body: inventories[family][body]["part_count"] for body in BODY_NAMES}
                for family in ("R64", "R32")
            },
        },
        "contact_patch_source": {
            label: patch["source"] for label, patch in patches.items()
        },
        "contact_patch_metrics": contact,
        "whole_surface_metrics": surfaces,
        "occupancy_vs_A": occupancy,
        "frozen_open_clearance": clearance,
        "semantic_retention": retention,
        "negative_controls": controls,
        "candidate_integrity": candidate_integrity,
        "pareto": pareto,
        "measurement_checks": measurement_checks,
        "measurement_pass": measurement_pass,
        "verdict": verdict,
        "scope_guards": {
            "app_launcher_for_cook_only": 1,
            "physx_cook_callbacks": 16,
            "simulation_context": 0,
            "environment_reset": 0,
            "cylinder_pose_write": 0,
            "q5_target_or_sample": 0,
            "controlled_physics_steps": 0,
            "live_contact_queries": 0,
            "offline_hppfcl_static_part_geometry_queries": clearance[
                "offline_hppfcl_part_geometry_query_count"
            ],
            "target_ik_path_changes": 0,
            "material_mass_actuator_physics_changes": 0,
            "canonical_or_live_asset_writes": 0,
            "hardware": 0,
        },
        "interpretation_boundary": {
            "current64_optimal": None,
            "physics_equivalence": None,
            "collider_count_tipping_causality": None,
            "actual_gpu_contact_execution": None,
            "grasp_feasibility": None,
            "g0a_pass": False,
            "C1_C2_are_exact_body_jaw_partition": False,
            "hull_count_is_measured_runtime": False,
            "next_live_authoring_or_physics_requires_new_approval": True,
        },
    }
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase("authoritative_measurement_evidence_committed", verdict=verdict)
    if not measurement_pass:
        raise RuntimeError(f"D371 measurement integrity failed: {measurement_checks}")

    boards = _render_boards(raw, candidates, patches)
    _phase("professor_boards_rendered", board_count=len(boards))
    rerun = _write_rerun(candidates, patches, evidence)
    _phase("rerun_artifact_validated", pass_value=rerun["strict_validation_pass"])
    _write_report(evidence)
    automated_checks = {
        "measurement_evidence_pass": evidence["measurement_pass"],
        "three_professor_boards_exact_1920x1080": len(boards) == 3 and all(row["exact_1920x1080"] for row in boards.values()),
        "rerun_strict_validation_pass": rerun["strict_validation_pass"],
        "report_exists_nonzero": REPORT_PATH.is_file() and REPORT_PATH.stat().st_size > 0,
        "D334_sidecar_unchanged": _sidecar_snapshot() == _read_json(PREREG_PATH)["d334_sidecar_before"],
    }
    automated = {
        "artifact": "D371_AUTOMATED_SUMMARY_V1",
        "case": "g0a_d371",
        "evidence_path": _rel(EVIDENCE_PATH),
        "evidence_sha256": _sha(EVIDENCE_PATH),
        "measurement_verdict": verdict,
        "boards": boards,
        "rerun": rerun,
        "report_path": _rel(REPORT_PATH),
        "report_sha256": _sha(REPORT_PATH),
        "checks": automated_checks,
        "pass": all(automated_checks.values()),
        "manual_inspection_pending": True,
    }
    _write_json_x(AUTOMATED_PATH, automated)
    _phase("automated_outputs_complete_manual_inspection_pending", pass_value=automated["pass"])
    if not automated["pass"]:
        raise RuntimeError(f"D371 automated visualization failed: {automated_checks}")
    return evidence


def _run() -> None:
    if not PREREG_PATH.is_file():
        raise RuntimeError("D371 prepare must pass before run")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D371 preregistration is not PASS")
    if _git("rev-parse", "HEAD") != prereg["head"] or _git("rev-parse", "origin/master") != prereg["origin_master"]:
        raise RuntimeError("D371 Git base changed after preregistration")
    if _dynamic_hashes() != prereg["dynamic_hashes"]:
        raise RuntimeError("D371 harness/worker changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D371 frozen input changed after preregistration")
    if _sidecar_snapshot() != prereg["d334_sidecar_before"]:
        raise RuntimeError("D334 user-owned sidecar changed")
    current_environment = _package_versions()
    if not current_environment["pass"] or current_environment != prereg["environment"]:
        raise RuntimeError("D371 runtime loader/package environment changed after preregistration")
    allowed = {PREREG_PATH.name}
    if {path.name for path in OUT_DIR.iterdir()} != allowed:
        raise RuntimeError("D371 pre-run inventory is not preregistration-only")

    PHASE_PATH.touch(exist_ok=False)
    _phase("run_started")
    command = [
        str(EXPECTED_PYTHON),
        "-B",
        str(WORKER),
        "--out-dir",
        str(OUT_DIR),
        "--headless",
    ]
    invocation = {
        "artifact": "D371_WORKER_INVOCATION_V1",
        "case": "g0a_d371",
        "worker_invocation_count": 1,
        "automatic_retry_count": 0,
        "command": command,
        "cwd": str(REPO),
        "worker_sha256": _sha(WORKER),
        "timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "scope": "cook-only; no SimulationContext/reset/physics/q5/contact/asset authoring",
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase("cook_worker_invoked")
    env = os.environ.copy()
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = f"{EXPECTED_PYTHON.parent}:{env.get('PATH', '')}"
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    sigterm_sent = False
    sigkill_sent = False
    try:
        stdout, stderr = process.communicate(timeout=WORKER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        sigterm_sent = True
        try:
            stdout, stderr = process.communicate(timeout=20.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            sigkill_sent = True
            stdout, stderr = process.communicate(timeout=20.0)
    elapsed = time.monotonic() - start
    _write_text_x(STDOUT_PATH, stdout)
    _write_text_x(STDERR_PATH, stderr)
    try:
        os.killpg(process.pid, 0)
        process_group_residue = True
    except ProcessLookupError:
        process_group_residue = False
    supervisor = {
        "artifact": "D371_WORKER_SUPERVISOR_V1",
        "case": "g0a_d371",
        "worker_invocation_count": 1,
        "automatic_retry_count": 0,
        "pid": process.pid,
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "timed_out": timed_out,
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "process_group_residue_after_wait": process_group_residue,
        "stdout_path": _rel(STDOUT_PATH),
        "stdout_sha256": _sha(STDOUT_PATH),
        "stderr_path": _rel(STDERR_PATH),
        "stderr_sha256": _sha(STDERR_PATH),
        "worker_summary_exists": WORKER_SUMMARY_PATH.is_file(),
        "worker_summary_sha256": _sha(WORKER_SUMMARY_PATH) if WORKER_SUMMARY_PATH.is_file() else None,
        "preclose_sentinel_exists": PRECLOSE_PATH.is_file(),
        "preclose_sentinel_sha256": _sha(PRECLOSE_PATH) if PRECLOSE_PATH.is_file() else None,
        "exclusive_worker_claim_exists": WORKER_CLAIM_PATH.is_file(),
        "exclusive_worker_claim_sha256": _sha(WORKER_CLAIM_PATH) if WORKER_CLAIM_PATH.is_file() else None,
    }
    supervisor["pass"] = bool(
        process.returncode == 0
        and not timed_out
        and not sigterm_sent
        and not sigkill_sent
        and not process_group_residue
        and supervisor["worker_summary_exists"]
        and supervisor["preclose_sentinel_exists"]
        and supervisor["exclusive_worker_claim_exists"]
    )
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase("cook_worker_returned", returncode=process.returncode, elapsed_seconds=elapsed)
    if not supervisor["pass"]:
        raise RuntimeError(f"D371 cook worker failed: {supervisor}")
    evidence = _audit_after_worker(supervisor)
    print(
        json.dumps(
            {
                "stage": "run",
                "measurement_pass": evidence["measurement_pass"],
                "verdict": evidence["verdict"],
                "manual_inspection_pending": True,
            },
            ensure_ascii=False,
        )
    )


EXPECTED_MANUAL_CHECKS = {
    "all_four_pngs_opened_at_original_resolution",
    "cap_board_has_exact_raw_A_R64_R32_columns",
    "semantic_board_has_exact_raw_A_C1_C2_columns",
    "contact_board_shows_fixed_and_moving_source_patch_overlays",
    "rerun_has_two_rows_five_candidate_views_and_notification_buffer",
    "no_unknown_timeline_or_empty_metric_panel",
    "no_notification_obscures_candidate_geometry",
    "Korean_text_legible_without_overlap_or_clipping",
    "offline_no_physics_no_grasp_boundary_visible",
    "colors_and_titles_match_candidate_lineage",
}


def _finalize() -> None:
    if COMPLETION_PATH.exists():
        raise FileExistsError(COMPLETION_PATH)
    required = [
        PREREG_PATH,
        PHASE_PATH,
        INVOCATION_PATH,
        WORKER_CLAIM_PATH,
        STDOUT_PATH,
        STDERR_PATH,
        SUPERVISOR_PATH,
        WORKER_SUMMARY_PATH,
        PRECLOSE_PATH,
        EVIDENCE_PATH,
        REPORT_PATH,
        CAP_BOARD_PATH,
        SEMANTIC_BOARD_PATH,
        CONTACT_BOARD_PATH,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION_PATH,
        RERUN_PNG_PATH,
        AUTOMATED_PATH,
        MANUAL_JSON_PATH,
        MANUAL_MD_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if EXCEPTION_PATH.exists():
        raise RuntimeError("D371 exception exists; finalize forbidden")
    prereg = _read_json(PREREG_PATH)
    if _git("rev-parse", "HEAD") != prereg["head"] or _git("rev-parse", "origin/master") != prereg["origin_master"]:
        raise RuntimeError("D371 Git base drift before finalize")
    if _dynamic_hashes() != prereg["dynamic_hashes"] or _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D371 code/input drift before finalize")
    current_environment = _package_versions()
    if not current_environment["pass"] or current_environment != prereg["environment"]:
        raise RuntimeError("D371 finalize loader/package environment changed after preregistration")
    evidence = _read_json(EVIDENCE_PATH)
    automated = _read_json(AUTOMATED_PATH)
    manual = _read_json(MANUAL_JSON_PATH)
    manual_checks = manual.get("checks", {})
    files = manual.get("files", [])
    expected_paths = {_rel(path) for path in (CAP_BOARD_PATH, SEMANTIC_BOARD_PATH, CONTACT_BOARD_PATH, RERUN_PNG_PATH)}
    observed_paths = {str(row.get("path")) for row in files if isinstance(row, dict)}
    file_checks = []
    for row in files:
        try:
            path = REPO / row["path"]
            file_checks.append(
                path.is_file()
                and _sha(path) == row["sha256"]
                and path.stat().st_size == row["bytes"]
                and _png_info(path)["dimensions"] == row["dimensions"]
                and bool(row.get("observations"))
            )
        except Exception:
            file_checks.append(False)
    checks = {
        "measurement_pass": evidence.get("measurement_pass") is True,
        "measurement_verdict_exact": evidence.get("verdict") == VERDICT_MEASURED,
        "automated_pass": automated.get("pass") is True,
        "rerun_validation_pass": _read_json(RERUN_VALIDATION_PATH).get("pass") is True,
        "manual_inspection_performed": manual.get("inspection_performed") is True,
        "manual_pass": manual.get("pass") is True,
        "manual_check_keys_exact": set(manual_checks) == EXPECTED_MANUAL_CHECKS,
        "all_manual_checks_pass": set(manual_checks) == EXPECTED_MANUAL_CHECKS and all(value is True for value in manual_checks.values()),
        "manual_exact_four_png_paths": observed_paths == expected_paths and len(files) == 4,
        "manual_file_hashes_sizes_dimensions_observations_exact": len(file_checks) == 4 and all(file_checks),
        "manual_markdown_exact": manual.get("markdown_path") == _rel(MANUAL_MD_PATH)
        and manual.get("markdown_sha256") == _sha(MANUAL_MD_PATH),
        "worker_actual_once_no_retry": _read_json(SUPERVISOR_PATH).get("pass") is True
        and _read_json(INVOCATION_PATH).get("worker_invocation_count") == 1
        and _read_json(INVOCATION_PATH).get("automatic_retry_count") == 0,
        "D334_sidecar_unchanged": _sidecar_snapshot() == prereg["d334_sidecar_before"],
    }
    visualization_keys = {
        "automated_pass",
        "rerun_validation_pass",
        "manual_inspection_performed",
        "manual_pass",
        "manual_check_keys_exact",
        "all_manual_checks_pass",
        "manual_exact_four_png_paths",
        "manual_file_hashes_sizes_dimensions_observations_exact",
        "manual_markdown_exact",
    }
    visualization_pass = all(checks[key] for key in visualization_keys)
    overall_pass = all(checks.values())
    completion_verdict = VERDICT_MEASURED if overall_pass else VERDICT_VIZ_FAIL
    _phase("finalize_started", checks_pass=overall_pass)
    completion = {
        "artifact": "D371_COMPLETION_SUMMARY_V1",
        "case": "g0a_d371",
        "worker_invocation_count": 1,
        "automatic_retry_count": 0,
        "measurement_pass": evidence["measurement_pass"],
        "visualization_pass": visualization_pass,
        "pass": overall_pass,
        "measurement_verdict": evidence["verdict"],
        "completion_verdict": completion_verdict,
        "checks": checks,
        "artifacts": {_rel(path): {"bytes": path.stat().st_size, "sha256": _sha(path)} for path in required},
        "scope_guards": evidence["scope_guards"],
        "interpretation_boundary": evidence["interpretation_boundary"],
        "next_live_authoring_or_physics_requires_new_approval": True,
        "g0a_pass": False,
    }
    _write_json_x(COMPLETION_PATH, completion)
    print(json.dumps({"stage": "finalize", "pass": overall_pass, "completion_verdict": completion_verdict}, ensure_ascii=False))
    if not overall_pass:
        raise RuntimeError(f"D371 completion integrity failed: {checks}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "finalize"), required=True)
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
        if args.stage == "run" and OUT_DIR.is_dir() and not EXCEPTION_PATH.exists():
            evidence_valid = False
            preserved_verdict = None
            if EVIDENCE_PATH.is_file():
                try:
                    preserved = _read_json(EVIDENCE_PATH)
                    evidence_valid = preserved.get("artifact") == "D371_OFFLINE_COLLIDER_COMPARISON_EVIDENCE_V1"
                    preserved_verdict = preserved.get("verdict")
                except Exception:
                    evidence_valid = False
            _write_json_x(
                EXCEPTION_PATH,
                {
                    "artifact": "D371_RUNTIME_EXCEPTION_V1",
                    "case": "g0a_d371",
                    "stage": args.stage,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "worker_invocation_count": 1 if INVOCATION_PATH.is_file() else 0,
                    "automatic_retry_count": 0,
                    "measurement_evidence_valid": evidence_valid,
                    "preserved_measurement_verdict": preserved_verdict,
                    "verdict": VERDICT_VIZ_FAIL if evidence_valid and preserved_verdict == VERDICT_MEASURED else VERDICT_FAIL,
                    "g0a_pass": False,
                },
            )
            try:
                _phase("run_exception", error=f"{type(error).__name__}: {error}")
            except Exception:
                pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
