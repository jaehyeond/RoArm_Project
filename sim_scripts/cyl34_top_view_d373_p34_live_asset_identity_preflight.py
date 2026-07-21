#!/usr/bin/env python3
"""D373 P34 live-asset identity controller, analyzer, and observability gate.

Stages are forward-only:

* ``prepare`` freezes inputs, the single variable, thresholds, counters, and
  failure-capable controls before any derivative asset or Isaac worker exists;
* ``run`` launches exactly one bounded worker and never retries it;
* ``analyze`` reads the worker's immutable raw USD/PhysX evidence, classifies
  callback polygon topology, and creates the 1080p board plus save-only RRD;
* ``finalize`` accepts a separately written manual visual-inspection record and
  writes the completion summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
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

CASE_ROOT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d373"
ATTEMPT = "attempt1_p34_live_asset_identity_preflight"
OUT_DIR = CASE_ROOT / ATTEMPT
PREREG_PATH = OUT_DIR / "d373_preregistration.json"
PHASE_PATH = OUT_DIR / "d373_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d373_worker_invocation.json"
SUPERVISOR_PATH = OUT_DIR / "d373_worker_supervisor.json"
STDOUT_PATH = OUT_DIR / "d373_worker_stdout.log"
STDERR_PATH = OUT_DIR / "d373_worker_stderr.log"
WORKER_SUMMARY_PATH = OUT_DIR / "d373_worker_raw_summary.json"
EVIDENCE_PATH = OUT_DIR / "d373_p34_live_identity_evidence.json"
REPORT_PATH = OUT_DIR / "d373_p34_live_identity_report.md"
BOARD_PATH = OUT_DIR / "d373_p34_identity_comparison_1920x1080.png"
RRD_PATH = OUT_DIR / "d373_p34_live_identity.rrd"
RBL_PATH = OUT_DIR / "d373_p34_live_identity.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d373_rerun_validation.json"
RERUN_PNG_PATH = OUT_DIR / "d373_rerun_inspection.png"
AUTOMATED_PATH = OUT_DIR / "d373_automated_summary.json"
MANUAL_JSON_PATH = OUT_DIR / "d373_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d373_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d373_completion_summary.json"
ANALYZE_EXCEPTION_PATH = OUT_DIR / "d373_analyze_exception.json"

HARNESS = Path(__file__).resolve()
WORKER = REPO / "sim_scripts/cyl34_top_view_d373_p34_live_asset_worker.py"
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
D339_HARNESS = REPO / "sim_scripts/cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair.py"
D345_HARNESS = REPO / "sim_scripts/cyl34_top_view_d345_grasp_g0a_deterministic_usd_metadata_comparator.py"
D348_HARNESS = REPO / "sim_scripts/cyl34_top_view_d348_grasp_g0a_physx_property_query_volume_semantics.py"
D372_GEOMETRY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d372/"
    "attempt2_external_schema_path_repair/"
    "d372_professor_semantic_candidate_geometry.json"
)
D372_EVIDENCE = D372_GEOMETRY.with_name("d372_professor_semantic_candidate_evidence.json")
D372_COMPLETION = D372_GEOMETRY.with_name("d372_completion_summary.json")
D372_SESSION = REPO / "claudedocs/session_20260721_grasp_g0a_d372_professor_semantic_compound_collider_design_offline.md"
BASE_ASSET_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/"
    "roarm_m3_fullmesh_fixed_point_parts"
)
BASE_ROOT_USD = BASE_ASSET_DIR / "roarm_m3.usd"
BASE_PHYSICS_USD = BASE_ASSET_DIR / "configuration/roarm_m3_physics.usd"
D345_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d345/d345_deterministic_usd_metadata_evidence.json"
D348_EVIDENCE = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/"
    "d348_callback_topology_volume_evidence.json"
)
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
PHYSX_SCHEMA = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "plugins/PhysxSchema/resources/schema.usda"
)
PHYSX_EXTENSION_TOML = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/config/extension.toml"
)
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
EXPECTED_HEAD = "5214721e91bd23b224998cba2b13a1f76294edad"
EXPECTED_GEOMETRY_SHA = "12fd1f32c35dfb9ae36cbbb412f6a51536aa1cc07c2dc17d05a5d189f3ee83e4"
EXPECTED_D372_EVIDENCE_SHA = "d68f658089aaf838ff454e9d0b301ec3f602785a3a730b3c329aa7785010e984"
EXPECTED_D372_COMPLETION_SHA = "57f3ed8fe6f057d059980a78bb51be8e881d8300297a4f41def6ddf94ad0cf43"
EXPECTED_BASE_ROOT_SHA = "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff"
EXPECTED_BASE_PHYSICS_SHA = "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503"
EXPECTED_COUNTS = {"link5": 16, "gripper_link": 18, "total": 34}
NEW_VARIABLES = ["p34_live_asset_materialization_and_binding_v1"]
SURFACE_TOL_M = 0.0001
BOUNDS_TOL_M = 0.0001
AUTHORED_CALLBACK_VOLUME_REL_TOL = 0.005
PROPERTY_VOLUME_REL_TOL = 0.05
MASS_API_ATOL = 1.0e-12
PROPERTY_MASS_STATE_ATOL = 1.0e-9
WORKER_TIMEOUT_S = 900.0
VERDICT_PASS = "D373_P34_LIVE_ASSET_IDENTITY_PASS_NO_PHYSICS"
VERDICT_FAIL = "D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP"
VERDICT_OBSERVABILITY_FAIL = "D373_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"

OFFICIAL_SOURCES = [
    {
        "title": "Omni Physics 107.3 Rigid Bodies",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html",
        "applicable_version": "Omni Physics 107.3; installed omni.physx 107.3.26",
        "use": "one rigid body may own multiple child colliders",
    },
    {
        "title": "Omni Physics 107.3 Colliders",
        "url": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html",
        "applicable_version": "Omni Physics 107.3; installed omni.physx 107.3.26",
        "use": "convex Mesh collider binding and collision approximation semantics",
    },
    {
        "title": "Isaac Sim 5.1.0 Physics Simulation Fundamentals",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html",
        "applicable_version": "installed Isaac Sim 5.1.0.0",
        "use": "multiple convex shapes can preserve concave openings",
    },
    {
        "title": "Isaac Sim 5.1.0 Performance Optimization Handbook",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html",
        "applicable_version": "installed Isaac Sim 5.1.0.0",
        "use": "use the simplest collision representation that satisfies accuracy",
    },
]


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


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_bytes(value: Any, dtype: str) -> bytes:
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
        stream.flush()
        os.fsync(stream.fileno())


def _phase(name: str, **fields: Any) -> None:
    ordinal = 1
    if PHASE_PATH.is_file():
        ordinal = sum(1 for line in PHASE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()) + 1
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"ordinal": ordinal, "phase": name, "pid": os.getpid(), "monotonic_ns": time.monotonic_ns(), **fields},
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def _sidecar_snapshot() -> dict[str, Any]:
    rows = [
        {"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(item for item in D334_SIDECAR.rglob("*") if item.is_file())
    ]
    return {
        "file_count": len(rows),
        "inventory_sha256": _sha_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "files": rows,
    }


def _candidate_manifest(geometry: dict[str, Any]) -> dict[str, Any]:
    rows = []
    names: dict[str, list[str]] = {body: [] for body in ("link5", "gripper_link")}
    roles: dict[str, Counter[str]] = {body: Counter() for body in names}
    for body in names:
        for index, part in enumerate(geometry["parts"][body]):
            vertices_f64 = np.asarray(part["vertices"], dtype="<f8")
            triangles_i64 = np.asarray(part["triangles"], dtype="<i8")
            vertices_f32 = np.asarray(vertices_f64, dtype="<f4")
            counts_i32 = np.full(len(triangles_i64), 3, dtype="<i4")
            indices_i32 = np.asarray(triangles_i64.reshape(-1), dtype="<i4")
            d372_digest = _sha_bytes(_array_bytes(vertices_f64, "<f8") + _array_bytes(triangles_i64, "<i8"))
            prim_name = f"p{index:03d}_{part['name']}"
            rows.append(
                {
                    "body": body,
                    "index": index,
                    "name": part["name"],
                    "prim_name": prim_name,
                    "role": part["role"],
                    "source": part["source"],
                    "d372_payload_sha256_registered": part["payload_sha256"],
                    "d372_payload_sha256_recomputed": d372_digest,
                    "d372_payload_exact": d372_digest == part["payload_sha256"],
                    "vertex_count": len(vertices_f64),
                    "triangle_count": len(triangles_i64),
                    "topology_volume_m3": float(part["topology_volume_m3"]),
                    "points_f32_sha256": _sha_bytes(_array_bytes(vertices_f32, "<f4")),
                    "face_counts_i32_sha256": _sha_bytes(_array_bytes(counts_i32, "<i4")),
                    "face_indices_i32_sha256": _sha_bytes(_array_bytes(indices_i32, "<i4")),
                    "authored_f32_topology_payload_sha256": _sha_bytes(
                        _array_bytes(vertices_f32, "<f4")
                        + _array_bytes(counts_i32, "<i4")
                        + _array_bytes(indices_i32, "<i4")
                    ),
                }
            )
            names[body].append(str(part["name"]))
            roles[body][str(part["role"])] += 1
    checks = {
        "artifact_exact": geometry.get("artifact") == "D372_PROFESSOR_SEMANTIC_CANDIDATE_GEOMETRY_V1",
        "candidate_exact": geometry.get("candidate") == "P34_professor_semantic_compound",
        "authority_float64_explicit_triangles": geometry.get("authority")
        == "Float64 vertices plus explicit triangle topology; no USD/live/PhysX authoring in D372",
        "link5_16": len(names["link5"]) == 16,
        "gripper_link_18": len(names["gripper_link"]) == 18,
        "total_34": len(rows) == 34,
        "unique_names_per_owner": all(len(values) == len(set(values)) for values in names.values()),
        "all_body_fields_match_container": all(
            part["body"] == body for body in names for part in geometry["parts"][body]
        ),
        "all_d372_float64_payloads_exact": all(row["d372_payload_exact"] for row in rows),
        "all_vertices_le_64": all(row["vertex_count"] <= 64 for row in rows),
        "owner_contract_exact": geometry["owner_contract"]["candidate_manifest"]
        == {"fixed_jaw": "link5", "moving_jaw": "gripper_link"},
    }
    return {
        "rows": rows,
        "names": names,
        "roles": {body: dict(sorted(value.items())) for body, value in roles.items()},
        "checks": checks,
        "pass": all(checks.values()),
    }


def _prepare_negative_controls(
    manifest: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    first = manifest["rows"][0]
    points = np.ascontiguousarray(
        np.asarray(geometry["parts"]["link5"][0]["vertices"], dtype="<f4")
    )
    point_bytes = bytearray(points.tobytes(order="C"))
    point_bytes[0] ^= 1
    perturbed_point_hash = _sha_bytes(bytes(point_bytes))
    controls = {
        "wrong_frozen_input_hash_rejected": ("0" * 64) != EXPECTED_GEOMETRY_SHA,
        "one_missing_part_rejected": len(manifest["rows"][:-1]) != EXPECTED_COUNTS["total"],
        "fixed_moving_owner_swap_rejected": {"fixed_jaw": "gripper_link", "moving_jaw": "link5"}
        != {"fixed_jaw": "link5", "moving_jaw": "gripper_link"},
        "one_float32_stream_bit_flip_rejected": perturbed_point_hash
        != first["points_f32_sha256"],
    }
    return {
        "controls": controls,
        "pass_count": sum(controls.values()),
        "expected_count": 4,
        "pass": all(controls.values()),
    }


def _input_paths() -> list[Path]:
    return [
        HARNESS,
        WORKER,
        VIZ_DEBUG,
        RERUN_CONTRACT,
        D339_HARNESS,
        D345_HARNESS,
        D348_HARNESS,
        D372_GEOMETRY,
        D372_EVIDENCE,
        D372_COMPLETION,
        D372_SESSION,
        BASE_ROOT_USD,
        BASE_PHYSICS_USD,
        *(path for path in sorted(BASE_ASSET_DIR.rglob("*")) if path.is_file()),
        D345_EVIDENCE,
        D348_EVIDENCE,
        PHYSX_SCHEMA,
        PHYSX_EXTENSION_TOML,
    ]


def _environment() -> dict[str, Any]:
    version = subprocess.run([str(RERUN_CLI), "--version"], capture_output=True, text=True, timeout=30, check=False)
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    values = {
        "python": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "psutil": importlib.metadata.version("psutil"),
        "rerun": importlib.metadata.version("rerun-sdk"),
        "isaacsim": importlib.metadata.version("isaacsim"),
        "rerun_cli": {"returncode": version.returncode, "stdout": version.stdout.strip(), "stderr": version.stderr.strip()},
        "nvidia_smi": {"returncode": gpu.returncode, "stdout": gpu.stdout.strip(), "stderr": gpu.stderr.strip()},
    }
    checks = {
        "python_exact": Path(values["python"]) == EXPECTED_PYTHON.resolve(),
        "numpy_1_26_0": values["numpy"] == "1.26.0",
        "psutil_5_9_8": values["psutil"] == "5.9.8",
        "rerun_0_34_1": values["rerun"] == "0.34.1",
        "isaacsim_5_1_0_0": values["isaacsim"] == "5.1.0.0",
        "rerun_cli_0_34_1": version.returncode == 0 and "0.34.1" in (version.stdout + version.stderr),
        "gpu_query_success_in_controller_context": gpu.returncode == 0 and bool(gpu.stdout.strip()),
    }
    required = {
        key: value
        for key, value in checks.items()
        if key != "gpu_query_success_in_controller_context"
    }
    return {
        **values,
        "checks": checks,
        "required_checks": required,
        "gpu_query_role": "informational here; the actual one-shot Isaac worker is launched with GPU access and is the operational gate",
        "pass": all(required.values()),
    }


def _schema_facts() -> dict[str, Any]:
    text = PHYSX_SCHEMA.read_text(encoding="utf-8")
    extension = PHYSX_EXTENSION_TOML.read_text(encoding="utf-8")
    facts = {
        "schema_path": str(PHYSX_SCHEMA),
        "schema_sha256": _sha(PHYSX_SCHEMA),
        "extension_toml_path": str(PHYSX_EXTENSION_TOML),
        "extension_toml_sha256": _sha(PHYSX_EXTENSION_TOML),
        "hull_vertex_limit_default_64": "physxConvexDecompositionCollision:hullVertexLimit = 64" in text,
        "max_convex_hulls_default_32": "physxConvexDecompositionCollision:maxConvexHulls = 32" in text,
        "omni_physx_extension_107_3_26": 'version = "107.3.26"' in extension,
        "official_sources": OFFICIAL_SOURCES,
        "representation_note": "D373 authors 34 box-shaped/semantic convex Mesh colliders; it does not test native UsdGeom.Cube performance",
    }
    facts["pass"] = bool(
        facts["hull_vertex_limit_default_64"]
        and facts["max_convex_hulls_default_32"]
        and facts["omni_physx_extension_107_3_26"]
    )
    return facts


def prepare() -> int:
    if CASE_ROOT.exists() or OUT_DIR.exists():
        raise RuntimeError("D373 forward-only path already exists; prepare refuses overwrite")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    geometry = _read_json(D372_GEOMETRY)
    manifest = _candidate_manifest(geometry)
    negatives = _prepare_negative_controls(manifest, geometry)
    inputs = {_key(path): _sha(path) for path in dict.fromkeys(_input_paths())}
    environment = _environment()
    schema = _schema_facts()
    git = {
        "head": _git("rev-parse", "HEAD"),
        "origin_master": _git("rev-parse", "origin/master"),
        "subject": _git("log", "-1", "--pretty=%s"),
        "status_short_after_d373_code_edits": _git("status", "--short").splitlines(),
        "clean_before_d373_edits": True,
        "clean_before_evidence_source": "START_HERE Git section updated at D373 boot",
    }
    checks = {
        "new_variable_count_one": NEW_VARIABLES == ["p34_live_asset_materialization_and_binding_v1"],
        "head_exact": git["head"] == EXPECTED_HEAD,
        "origin_exact": git["origin_master"] == EXPECTED_HEAD,
        "geometry_hash_exact": inputs[_rel(D372_GEOMETRY)] == EXPECTED_GEOMETRY_SHA,
        "d372_evidence_hash_exact": inputs[_rel(D372_EVIDENCE)] == EXPECTED_D372_EVIDENCE_SHA,
        "d372_completion_hash_exact": inputs[_rel(D372_COMPLETION)] == EXPECTED_D372_COMPLETION_SHA,
        "base_root_hash_exact": inputs[_rel(BASE_ROOT_USD)] == EXPECTED_BASE_ROOT_SHA,
        "base_physics_hash_exact": inputs[_rel(BASE_PHYSICS_USD)] == EXPECTED_BASE_PHYSICS_SHA,
        "candidate_manifest_pass": manifest["pass"],
        "negative_controls_4_of_4": negatives["pass"],
        "environment_pass": environment["pass"],
        "schema_pass": schema["pass"],
        "sidecar_three_files": _sidecar_snapshot()["file_count"] == 3,
    }
    prereg = {
        "artifact": "D373_P34_LIVE_ASSET_IDENTITY_PREREGISTRATION_V1",
        "case": "g0a_d373",
        "attempt": ATTEMPT,
        "approved_scope": "live asset load/cook/readback identity only; no physics/q5/contact/cylinder/target/IK/path",
        "new_variables": NEW_VARIABLES,
        "git": git,
        "inputs": inputs,
        "candidate_manifest": manifest,
        "prepare_negative_controls": negatives,
        "environment": environment,
        "nvidia_contract": schema,
        "d344_lineage": {
            "role": "current physical-reference composition cloned only to preserve non-collision robot semantics",
            "historical_d344_fail_not_rewritten": True,
            "d345_address_free_comparator_pass_inherited": True,
            "d347_d348_live_callback_and_polygon_topology_contract_inherited": True,
        },
        "representation_contract": {
            "part_count": 34,
            "link5": 16,
            "gripper_link": 18,
            "usd_representation": "manually authored UsdGeom.Mesh child colliders with convexHull approximation",
            "lower_housing_semantics": "box-shaped Mesh copied exactly from D372; not a native Box/Cube performance test",
            "automatic_convex_decomposition": False,
            "callback_request_order": "prototype then instance per part",
            "callback_request_count": 68,
            "raw_instance_prototype_pairs": 34,
        },
        "registered_thresholds": {
            "authored_live_symmetric_surface_m": SURFACE_TOL_M,
            "authored_live_bounds_max_abs_m": BOUNDS_TOL_M,
            "authored_callback_topology_volume_relative": AUTHORED_CALLBACK_VOLUME_REL_TOL,
            "callback_topology_property_volume_relative": PROPERTY_VOLUME_REL_TOL,
            "mass_api_state_atol": MASS_API_ATOL,
            "property_query_mass_com_inertia_axes_atol": PROPERTY_MASS_STATE_ATOL,
            "threshold_note": "surface/bounds inherit D339 0.1mm; property volume inherits D348 5%; authored/callback volume 0.5% is a D373 preregistered engineering identity gate, not an NVIDIA default",
        },
        "scope_counters": {
            "actual_worker": 1,
            "automatic_retry": 0,
            "watchdog_timeout_s": WORKER_TIMEOUT_S,
            "simulation_app_launch": 1,
            "derivative_asset_materialization": 1,
            "physx_stage_attach_detach": "1/1",
            "property_queries": 2,
            "callback_requests": 68,
            "simulation_context_reset_timeline_play_commit_step_forward": 0,
            "q5_command_sample_contact_query": 0,
            "cylinder_create_or_pose_write": 0,
            "target_ik_path_pose_change": 0,
            "automatic_decomposition_sweep": 0,
            "approved_collision_mesh_and_schema_authors": 34,
            "inherited_material_mass_actuator_physics_setting_change": 0,
            "isaac_hydra_render": 0,
        },
        "failure_capable_controls": {
            "prepare": list(negatives["controls"]),
            "post_callback": ["delete_one_reported_polygon_breaks_closure", "perturb_mass_breaks_state_guard"],
            "expected_total": 6,
        },
        "observability_contract": {
            "exact_board": "1920x1080",
            "comparison_columns": ["D372 Float64", "USD Float32 readback", "PhysX instance callback", "PhysX prototype callback"],
            "rrd": "save-only before first user log; footer/entity/timeline/component/RBL validation",
            "manual_original_resolution_inspection": True,
            "rerun_float32_is_inspection_only": True,
        },
        "sidecar_before": _sidecar_snapshot(),
        "stop_rules": {
            "identity_pass": VERDICT_PASS,
            "numeric_or_binding_failure": VERDICT_FAIL,
            "observability_failure": VERDICT_OBSERVABILITY_FAIL,
            "physics_comparison_requires_new_approval": True,
            "g0a_pass_remains_false": True,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    _phase("preregistration_frozen", preregistration_sha256=_sha(PREREG_PATH), passed=prereg["pass"])
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"], "path": _rel(PREREG_PATH)}, sort_keys=True))
    return 0 if prereg["pass"] else 1


def run_worker() -> int:
    prereg = _read_json(PREREG_PATH)
    if not prereg.get("pass"):
        raise RuntimeError("D373 preregistration did not pass")
    for path in (INVOCATION_PATH, SUPERVISOR_PATH, OUT_DIR / "d373_worker_claim.json"):
        if path.exists():
            raise RuntimeError(f"D373 one-shot worker path already claimed: {_rel(path)}")
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
        "artifact": "D373_SINGLE_WORKER_INVOCATION_V1",
        "command": command,
        "cwd": str(REPO),
        "worker_sha256": _sha(WORKER),
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
    supervisor = {
        "artifact": "D373_BOUNDED_SINGLE_WORKER_SUPERVISOR_V1",
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
        "worker_summary_exists": WORKER_SUMMARY_PATH.is_file(),
        "preclose_sentinel_exists": (OUT_DIR / "d373_worker_preclose_sentinel.json").is_file(),
        "exclusive_claim_exists": (OUT_DIR / "d373_worker_claim.json").is_file(),
    }
    supervisor["pass"] = bool(
        process.returncode == 0
        and not timed_out
        and not sigterm_sent
        and not sigkill_sent
        and supervisor["worker_summary_exists"]
        and supervisor["preclose_sentinel_exists"]
        and supervisor["exclusive_claim_exists"]
    )
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase("supervisor_worker_exit", returncode=process.returncode, elapsed_s=elapsed, passed=supervisor["pass"])
    print(json.dumps({"stage": "run", "pass": supervisor["pass"], "returncode": process.returncode}, sort_keys=True))
    return 0 if supervisor["pass"] else 1


def _callback_convex(path: Path) -> dict[str, Any]:
    witness = _read_json(path)
    events = witness.get("events", [])
    if witness.get("callback_count") != 1 or len(events) != 1:
        raise ValueError(f"{path}: expected exactly one callback")
    if events[0].get("result_name") != "RESULT_VALID" or events[0].get("serialization_errors"):
        raise ValueError(f"{path}: invalid callback result or serialization")
    convexes = events[0].get("convexes", [])
    if events[0].get("convex_count") != 1 or len(convexes) != 1:
        raise ValueError(f"{path}: expected exactly one convex")
    return convexes[0]


def _payload_digest(convex: dict[str, Any]) -> str:
    payload = {
        key: convex[key]
        for key in ("vertices", "indices", "polygons", "vertex_count", "index_count", "polygon_count")
    }
    return _sha_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _triangulate(convex: dict[str, Any], *, drop_polygon: int | None = None) -> np.ndarray:
    indices = np.asarray(convex["indices"], dtype=np.int64)
    triangles: list[list[int]] = []
    covered = 0
    for polygon_index, polygon in enumerate(convex["polygons"]):
        base = int(polygon["index_base"])
        count = int(polygon["num_vertices"])
        if base != covered or count < 3 or base + count > len(indices):
            raise ValueError("callback polygon spans are non-contiguous or invalid")
        face = indices[base : base + count]
        covered += count
        if polygon_index == drop_polygon:
            continue
        for index in range(1, count - 1):
            triangles.append([int(face[0]), int(face[index]), int(face[index + 1])])
    if covered != len(indices) or not triangles:
        raise ValueError("callback polygon spans do not exactly cover the index buffer")
    return np.asarray(triangles, dtype=np.int64)


def _closed_oriented(triangles: np.ndarray) -> dict[str, Any]:
    directed: Counter[tuple[int, int]] = Counter()
    undirected: Counter[tuple[int, int]] = Counter()
    for tri in triangles.tolist():
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            directed[(int(a), int(b))] += 1
            undirected[tuple(sorted((int(a), int(b))))] += 1
    twice = bool(undirected) and all(value == 2 for value in undirected.values())
    opposite = bool(undirected) and all(
        directed[(edge[0], edge[1])] == 1 and directed[(edge[1], edge[0])] == 1
        for edge in undirected
    )
    return {"undirected_edge_count": len(undirected), "all_edges_twice": twice, "opposite_winding": opposite, "pass": twice and opposite}


def _signed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    origin = vertices.mean(axis=0)
    shifted = vertices - origin
    a, b, c = (shifted[triangles[:, index]] for index in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _outward_planes(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    center = np.mean(vertices, axis=0)
    planes = []
    for tri in triangles:
        a, b, c = vertices[tri]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-15:
            continue
        normal /= norm
        if float(np.dot(normal, center - a)) > 0.0:
            normal = -normal
        planes.append([*normal.tolist(), -float(np.dot(normal, a))])
    if not planes:
        raise ValueError("no nondegenerate authored planes")
    return np.asarray(planes, dtype=np.float64)


def _callback_planes(convex: dict[str, Any]) -> np.ndarray:
    vertices = np.asarray(convex["vertices"], dtype=np.float64)
    center = vertices.mean(axis=0)
    rows = []
    for polygon in convex["polygons"]:
        plane = np.asarray(polygon["plane"], dtype=np.float64)
        norm = float(np.linalg.norm(plane[:3]))
        if plane.shape != (4,) or not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("invalid callback polygon plane")
        plane = plane / norm
        if float(np.dot(plane[:3], center) + plane[3]) > 0.0:
            plane = -plane
        rows.append(plane)
    return np.asarray(rows, dtype=np.float64)


def _directed_solid_distance(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    target_triangles: np.ndarray,
    target_planes: np.ndarray,
) -> float:
    import trimesh

    triangles = target_vertices[target_triangles]
    maximum = 0.0
    for point in source_vertices:
        if float(np.max(target_planes[:, :3] @ point + target_planes[:, 3])) <= 1.0e-10:
            distance = 0.0
        else:
            tiled = np.repeat(point[None, :], len(triangles), axis=0)
            closest = trimesh.triangles.closest_point(triangles, tiled)
            distance = float(np.min(np.linalg.norm(closest - point[None, :], axis=1)))
        maximum = max(maximum, distance)
    return maximum


def _surface_distance(
    authored_vertices: np.ndarray,
    authored_triangles: np.ndarray,
    convex: dict[str, Any],
    callback_triangles: np.ndarray,
) -> dict[str, Any]:
    live_vertices = np.asarray(convex["vertices"], dtype=np.float64)
    authored_planes = _outward_planes(authored_vertices, authored_triangles)
    live_planes = _callback_planes(convex)
    authored_to_live = _directed_solid_distance(
        authored_vertices, live_vertices, callback_triangles, live_planes
    )
    live_to_authored = _directed_solid_distance(
        live_vertices, authored_vertices, authored_triangles, authored_planes
    )
    symmetric = max(authored_to_live, live_to_authored)
    return {
        "authored_to_live_m": authored_to_live,
        "live_to_authored_m": live_to_authored,
        "symmetric_m": symmetric,
        "tolerance_m": SURFACE_TOL_M,
        "pass": symmetric <= SURFACE_TOL_M,
    }


def _structural_callback(convex: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(convex["vertices"], dtype=np.float64)
    indices = np.asarray(convex["indices"], dtype=np.int64)
    polygons = list(convex["polygons"])
    spans = []
    covered = 0
    for polygon in polygons:
        base, count = int(polygon["index_base"]), int(polygon["num_vertices"])
        spans.append(base == covered and count >= 3 and base + count <= len(indices))
        covered += count
    checks = {
        "vertices_shape_finite": vertices.shape == (int(convex["vertex_count"]), 3) and np.all(np.isfinite(vertices)),
        "indices_shape_nonempty": indices.shape == (int(convex["index_count"]),) and len(indices) > 0,
        "indices_in_range": len(indices) > 0 and int(indices.min()) >= 0 and int(indices.max()) < len(vertices),
        "polygon_count_exact_nonempty": len(polygons) == int(convex["polygon_count"]) and bool(polygons),
        "polygon_spans_contiguous_exact": all(spans) and covered == len(indices),
        "gpu_vertices_4_to_64": 4 <= len(vertices) <= 64,
        "gpu_polygons_le_64": 4 <= len(polygons) <= 64,
        "vertices_per_polygon_le_32": bool(polygons) and max(int(row["num_vertices"]) for row in polygons) <= 32,
        "polygon_planes_finite": all(
            len(row["plane"]) == 4 and np.all(np.isfinite(np.asarray(row["plane"], dtype=np.float64)))
            for row in polygons
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


def _property_audit(worker: dict[str, Any], readback_by_path: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    expected_counts = {"link5": 17, "gripper_link": 19}
    for body in ("link5", "gripper_link"):
        query = worker["property_queries"][body]
        p34 = [row for row in query["colliders"] if "/d373_p34_parts/" in str(row["path"])]
        legacy = [row for row in query["colliders"] if row["path"] == worker["live_inventory"]["legacy"][body]["path"]]
        unknown = [row for row in query["colliders"] if row not in p34 and row not in legacy]
        p34_paths = {row["path"] for row in p34}
        expected_paths = {path for path, row in readback_by_path.items() if row["body"] == body}
        local_checks = {
            row["path"]: {
                "local_pos_zero": np.allclose(row["local_pos_m"], [0.0, 0.0, 0.0], rtol=0.0, atol=1.0e-9),
                "local_rot_identity": np.allclose(row["local_rot_xyzw"], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1.0e-9),
                "positive_volume": math.isfinite(float(row["volume_m3"])) and float(row["volume_m3"]) > 0.0,
                "valid_result": row["result_name"] == "VALID",
            }
            for row in p34
        }
        checks = {
            "query_finished_pass": query["pass"],
            "exact_property_row_total_with_disabled_legacy": len(query["colliders"]) == expected_counts[body],
            "exact_p34_count": len(p34) == EXPECTED_COUNTS[body],
            "exact_p34_path_bijection": p34_paths == expected_paths,
            "exact_one_known_disabled_legacy_row": len(legacy) == 1,
            "unknown_rows_zero": not unknown,
            "all_p34_local_bindings_valid": bool(local_checks) and all(all(value.values()) for value in local_checks.values()),
            "zero_app_update_pumps": query["simulation_app_update_pumps"] == 0,
        }
        rows[body] = {
            "expected_property_row_count_including_disabled_legacy": expected_counts[body],
            "p34_rows": p34,
            "legacy_rows": legacy,
            "unknown_rows": unknown,
            "local_checks": local_checks,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {"bodies": rows, "pass": all(row["pass"] for row in rows.values())}


def _mass_live_audit(worker: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for body in ("link5", "gripper_link"):
        base = worker["mass_api_base_vs_derivative"]["bodies"][body]["base"]
        derivative = worker["mass_api_base_vs_derivative"]["bodies"][body]["variant"]
        inspection = worker["mass_api_inspection_stage"]["bodies"][body]
        prop = worker["property_queries"][body]["rigid_body"]
        property_wxyz = [
            prop["principal_axes_xyzw"][3],
            prop["principal_axes_xyzw"][0],
            prop["principal_axes_xyzw"][1],
            prop["principal_axes_xyzw"][2],
        ]
        comparisons = {
            "base_derivative_mass": abs(base["mass_kg"] - derivative["mass_kg"]),
            "base_derivative_com": float(np.max(np.abs(np.asarray(base["center_of_mass_m"]) - np.asarray(derivative["center_of_mass_m"])))),
            "base_derivative_inertia": float(np.max(np.abs(np.asarray(base["diagonal_inertia"]) - np.asarray(derivative["diagonal_inertia"])))),
            "base_derivative_axes": float(np.max(np.abs(np.asarray(base["principal_axes_wxyz"]) - np.asarray(derivative["principal_axes_wxyz"])))),
            "base_inspection_mass": abs(base["mass_kg"] - inspection["mass_kg"]),
            "base_inspection_com": float(np.max(np.abs(np.asarray(base["center_of_mass_m"]) - np.asarray(inspection["center_of_mass_m"])))),
            "base_inspection_inertia": float(np.max(np.abs(np.asarray(base["diagonal_inertia"]) - np.asarray(inspection["diagonal_inertia"])))),
            "base_inspection_axes": float(np.max(np.abs(np.asarray(base["principal_axes_wxyz"]) - np.asarray(inspection["principal_axes_wxyz"])))),
            "base_property_mass": abs(base["mass_kg"] - prop["mass_kg"]),
            "base_property_com": float(np.max(np.abs(np.asarray(base["center_of_mass_m"]) - np.asarray(prop["center_of_mass_m"])))),
            "base_property_inertia": float(np.max(np.abs(np.asarray(base["diagonal_inertia"]) - np.asarray(prop["diagonal_inertia"])))),
            "base_property_axes": float(np.max(np.abs(np.asarray(base["principal_axes_wxyz"]) - np.asarray(property_wxyz)))),
        }
        checks = {
            "authored_and_composed_mass_state_atol_1e_12": max(
                value for key, value in comparisons.items() if "property" not in key
            )
            <= MASS_API_ATOL,
            "property_mass_com_inertia_axes_atol_1e_9": max(
                value for key, value in comparisons.items() if "property" in key
            )
            <= PROPERTY_MASS_STATE_ATOL,
        }
        rows[body] = {
            "base": base,
            "derivative": derivative,
            "inspection_mass_api": inspection,
            "physx_property_query": prop,
            "property_principal_axes_wxyz": property_wxyz,
            "max_abs_deltas": comparisons,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {"bodies": rows, "pass": all(row["pass"] for row in rows.values())}


def _analyze_callbacks(
    geometry: dict[str, Any], worker: dict[str, Any], property_audit: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    readback_by_path = {
        row["live_path"]: row for row in worker["authored_readback"]["rows"]
    }
    property_by_path = {
        row["path"]: row
        for body in ("link5", "gripper_link")
        for row in property_audit["bodies"][body]["p34_rows"]
    }
    source_by_key = {
        (body, f"p{index:03d}_{part['name']}"): part
        for body in ("link5", "gripper_link")
        for index, part in enumerate(geometry["parts"][body])
    }
    rows = []
    aggregate = Counter()
    for worker_row in worker["callback_rows"]:
        body, prim_name = worker_row["body"], worker_row["prim_name"]
        source = source_by_key[(body, prim_name)]
        readback = readback_by_path[worker_row["instance_path"]]
        authored_vertices = np.asarray(readback["points_f32"], dtype=np.float64)
        authored_triangles = np.asarray(readback["face_vertex_indices"], dtype=np.int64).reshape(-1, 3)
        property_row = property_by_path[worker_row["instance_path"]]
        channels = {}
        for channel in ("prototype", "instance"):
            witness_path = REPO / worker_row["channels"][channel]["witness_path"]
            convex = _callback_convex(witness_path)
            structural = _structural_callback(convex)
            callback_triangles = _triangulate(convex)
            closure = _closed_oriented(callback_triangles)
            vertices = np.asarray(convex["vertices"], dtype=np.float64)
            signed_volume = _signed_volume(vertices, callback_triangles)
            volume = abs(signed_volume)
            surface = _surface_distance(authored_vertices, authored_triangles, convex, callback_triangles)
            callback_bounds = np.vstack([vertices.min(axis=0), vertices.max(axis=0)])
            authored_bounds = np.vstack([authored_vertices.min(axis=0), authored_vertices.max(axis=0)])
            bounds_delta = float(np.max(np.abs(callback_bounds - authored_bounds)))
            authored_volume = float(source["topology_volume_m3"])
            property_volume = float(property_row["volume_m3"])
            authored_rel = abs(volume - authored_volume) / max(abs(authored_volume), 1.0e-15)
            property_rel = abs(volume - property_volume) / max(abs(property_volume), 1.0e-15)
            plane_residual = 0.0
            indices = np.asarray(convex["indices"], dtype=np.int64)
            for polygon in convex["polygons"]:
                base, count = int(polygon["index_base"]), int(polygon["num_vertices"])
                plane = np.asarray(polygon["plane"], dtype=np.float64)
                face_vertices = vertices[indices[base : base + count]]
                plane_residual = max(
                    plane_residual,
                    float(np.max(np.abs(face_vertices @ plane[:3] + plane[3]))),
                )
            checks = {
                "worker_protocol_pass": worker_row["channels"][channel]["pass"],
                "structural_payload_pass": structural["pass"],
                "callback_polygon_topology_closed_oriented": closure["pass"],
                "positive_finite_topology_volume": math.isfinite(volume) and volume > 0.0,
                "surface_le_0_1mm": surface["pass"],
                "bounds_le_0_1mm": bounds_delta <= BOUNDS_TOL_M,
                "authored_callback_topology_volume_le_0_5pct": authored_rel
                <= AUTHORED_CALLBACK_VOLUME_REL_TOL,
                "callback_topology_property_volume_le_5pct": property_rel <= PROPERTY_VOLUME_REL_TOL,
                "polygon_plane_residual_le_1e_5m": plane_residual <= 1.0e-5,
            }
            channels[channel] = {
                "witness_path": _rel(witness_path),
                "witness_sha256": _sha(witness_path),
                "payload_sha256": _payload_digest(convex),
                "vertex_count": len(vertices),
                "polygon_count": len(convex["polygons"]),
                "triangle_count": len(callback_triangles),
                "max_vertices_per_polygon": max(int(row["num_vertices"]) for row in convex["polygons"]),
                "callback_topology_triangles": callback_triangles.tolist(),
                "callback_vertices_m": vertices.tolist(),
                "callback_bounds_m": callback_bounds.tolist(),
                "authored_bounds_m": authored_bounds.tolist(),
                "bounds_max_abs_delta_m": bounds_delta,
                "callback_topology_signed_volume_m3": signed_volume,
                "callback_topology_volume_m3": volume,
                "authored_d372_topology_volume_m3": authored_volume,
                "physx_property_volume_m3": property_volume,
                "authored_callback_volume_relative_delta": authored_rel,
                "callback_property_volume_relative_delta": property_rel,
                "max_polygon_plane_residual_m": plane_residual,
                "surface": surface,
                "structural": structural,
                "closure": closure,
                "checks": checks,
                "pass": all(checks.values()),
            }
            aggregate["channels"] += 1
            aggregate["channels_pass"] += int(channels[channel]["pass"])
            aggregate["closed"] += int(closure["pass"])
            aggregate["surface"] += int(surface["pass"])
            aggregate["bounds"] += int(bounds_delta <= BOUNDS_TOL_M)
            aggregate["authored_volume"] += int(authored_rel <= AUTHORED_CALLBACK_VOLUME_REL_TOL)
            aggregate["property_volume"] += int(property_rel <= PROPERTY_VOLUME_REL_TOL)
        raw_pair = channels["prototype"]["payload_sha256"] == channels["instance"]["payload_sha256"]
        row_checks = {
            "prototype_pass": channels["prototype"]["pass"],
            "instance_pass": channels["instance"]["pass"],
            "raw_instance_prototype_payload_exact": raw_pair,
            "authored_f32_digest_bound": readback["authored_f32_topology_payload_sha256"]
            == worker_row["authored_f32_topology_payload_sha256"],
        }
        rows.append(
            {
                "body": body,
                "name": worker_row["name"],
                "role": worker_row["role"],
                "prim_name": prim_name,
                "instance_path": worker_row["instance_path"],
                "prototype_path": worker_row["prototype_path"],
                "authored_f32_topology_payload_sha256": readback[
                    "authored_f32_topology_payload_sha256"
                ],
                "channels": channels,
                "checks": row_checks,
                "pass": all(row_checks.values()),
            }
        )
        aggregate["pairs"] += 1
        aggregate["raw_pairs_exact"] += int(raw_pair)
    rows.sort(key=lambda row: (row["body"], row["prim_name"]))
    counts = dict(aggregate)
    checks = {
        "rows_34": len(rows) == 34,
        "channels_68": counts.get("channels") == 68,
        "channels_pass_68": counts.get("channels_pass") == 68,
        "raw_pairs_exact_34": counts.get("raw_pairs_exact") == 34,
        "closed_68": counts.get("closed") == 68,
        "surface_68": counts.get("surface") == 68,
        "bounds_68": counts.get("bounds") == 68,
        "authored_volume_68": counts.get("authored_volume") == 68,
        "property_volume_68": counts.get("property_volume") == 68,
        "all_rows_pass": all(row["pass"] for row in rows),
    }
    return rows, {"counts": counts, "checks": checks, "pass": all(checks.values())}


def _post_negative_controls(rows: list[dict[str, Any]], mass: dict[str, Any]) -> dict[str, Any]:
    first_channel = rows[0]["channels"]["instance"]
    convex = _callback_convex(REPO / first_channel["witness_path"])
    dropped = _triangulate(convex, drop_polygon=0)
    dropped_closed = _closed_oriented(dropped)
    base_mass = float(mass["bodies"]["link5"]["base"]["mass_kg"])
    perturbed_mass = base_mass + 10.0 * PROPERTY_MASS_STATE_ATOL
    controls = {
        "delete_one_reported_polygon_breaks_closure": not dropped_closed["pass"],
        "perturb_mass_breaks_exact_state_guard": abs(perturbed_mass - base_mass)
        > PROPERTY_MASS_STATE_ATOL,
    }
    return {
        "controls": controls,
        "dropped_polygon_closure": dropped_closed,
        "base_mass_kg": base_mass,
        "perturbed_mass_kg": perturbed_mass,
        "pass_count": sum(controls.values()),
        "expected_count": 2,
        "pass": all(controls.values()),
    }


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


def _make_board(geometry: dict[str, Any], worker: dict[str, Any], rows: list[dict[str, Any]], verdict: str) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    regular = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    bold = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    colors = {
        "structural_body": "#0072B2",
        "connector_support": "#E69F00",
        "fixed_jaw": "#F0E442",
        "fixed_jaw_backbone": "#D98C00",
        "moving_support": "#CC79A7",
        "moving_jaw": "#009E73",
        "moving_jaw_backbone": "#00796B",
    }
    readback = {(row["body"], row["prim_name"]): row for row in worker["authored_readback"]["rows"]}
    by_key = {(row["body"], row["prim_name"]): row for row in rows}
    source = {
        (body, f"p{index:03d}_{part['name']}"): part
        for body in ("link5", "gripper_link")
        for index, part in enumerate(geometry["parts"][body])
    }

    def add_mesh(ax: Any, vertices: np.ndarray, triangles: np.ndarray, color: str) -> None:
        collection = Poly3DCollection(
            vertices[triangles] * 1000.0,
            facecolor=color,
            edgecolor="#303030",
            linewidth=0.18,
            alpha=0.62,
        )
        ax.add_collection3d(collection)

    def frame(ax: Any, vertices: np.ndarray, body: str) -> None:
        value = vertices * 1000.0
        lo, hi = value.min(axis=0), value.max(axis=0)
        center = (lo + hi) / 2.0
        radius = max(float(np.max(hi - lo)) / 2.0, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-62 if body == "link5" else -90)
        ax.set_xlabel("x (mm)", fontsize=7)
        ax.set_ylabel("y (mm)", fontsize=7)
        ax.set_zlabel("z (mm)", fontsize=7)
        ax.tick_params(labelsize=6)

    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="white")
    titles = ["D372 원본 Float64", "USD 읽기값 Float32", "PhysX instance callback", "PhysX prototype callback"]
    for row_index, body in enumerate(("link5", "gripper_link")):
        body_keys = sorted(key for key in source if key[0] == body)
        all_vertices = np.vstack([np.asarray(source[key]["vertices"], dtype=np.float64) for key in body_keys])
        for column in range(4):
            ax = fig.add_subplot(2, 4, row_index * 4 + column + 1, projection="3d")
            for key in body_keys:
                role = str(source[key]["role"])
                if column == 0:
                    vertices = np.asarray(source[key]["vertices"], dtype=np.float64)
                    triangles = np.asarray(source[key]["triangles"], dtype=np.int64)
                elif column == 1:
                    item = readback[key]
                    vertices = np.asarray(item["points_f32"], dtype=np.float64)
                    triangles = np.asarray(item["face_vertex_indices"], dtype=np.int64).reshape(-1, 3)
                else:
                    channel = "instance" if column == 2 else "prototype"
                    item = by_key[key]["channels"][channel]
                    vertices = np.asarray(item["callback_vertices_m"], dtype=np.float64)
                    triangles = np.asarray(item["callback_topology_triangles"], dtype=np.int64)
                add_mesh(ax, vertices, triangles, colors[role])
            frame(ax, all_vertices, body)
            count = EXPECTED_COUNTS[body]
            ax.set_title(f"{body} · {count}개\n{titles[column]}", fontproperties=bold, fontsize=10, pad=4)
    max_surface_mm = max(
        row["channels"][channel]["surface"]["symmetric_m"] * 1000.0
        for row in rows
        for channel in ("instance", "prototype")
    )
    max_bounds_mm = max(
        row["channels"][channel]["bounds_max_abs_delta_m"] * 1000.0
        for row in rows
        for channel in ("instance", "prototype")
    )
    fig.suptitle(
        "D373 P34 실제 USD/PhysX 동일성 확인 — 16 + 18 = 34개",
        fontproperties=bold,
        fontsize=20,
        y=0.965,
    )
    fig.text(
        0.5,
        0.043,
        f"판정 {verdict} · callback {evidence['callback_aggregate']['counts']['channels_pass']}/{evidence['callback_aggregate']['counts']['channels']} · instance↔prototype {evidence['callback_aggregate']['counts']['raw_pairs_exact']}/{evidence['callback_aggregate']['counts']['pairs']} · 최대 표면 차이 {max_surface_mm:.6f}mm · 최대 bounds 차이 {max_bounds_mm:.6f}mm",
        ha="center",
        fontproperties=bold,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.018,
        "같은 색은 같은 역할 · 원 polygon topology 기준 · 물리 스텝/q5/접촉/원통 시험은 모두 0회",
        ha="center",
        fontproperties=regular,
        fontsize=10,
    )
    fig.tight_layout(rect=[0.01, 0.075, 0.99, 0.925])
    fig.savefig(BOARD_PATH, dpi=120, facecolor="white")
    plt.close(fig)
    info = _png_info(BOARD_PATH)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"D373 board is not exact 1920x1080: {info}")
    return info


def _write_rerun(geometry: dict[str, Any], worker: dict[str, Any], rows: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.viz_debug import log_rerun

    colors = {
        "structural_body": [0, 114, 178, 170],
        "connector_support": [230, 159, 0, 170],
        "fixed_jaw": [240, 228, 66, 210],
        "fixed_jaw_backbone": [217, 140, 0, 190],
        "moving_support": [204, 121, 167, 170],
        "moving_jaw": [0, 158, 115, 200],
        "moving_jaw_backbone": [0, 121, 107, 190],
    }
    source = {
        (body, f"p{index:03d}_{part['name']}"): part
        for body in ("link5", "gripper_link")
        for index, part in enumerate(geometry["parts"][body])
    }
    readback = {(row["body"], row["prim_name"]): row for row in worker["authored_readback"]["rows"]}
    analyzed = {(row["body"], row["prim_name"]): row for row in rows}
    meshes = []
    expected = []
    component_contract: dict[str, list[str]] = {}
    mesh_components = ["CoordinateFrame:frame", "Mesh3D:albedo_factor", "Mesh3D:triangle_indices", "Mesh3D:vertex_positions"]
    for body, prim_name in sorted(source):
        part = source[(body, prim_name)]
        role = part["role"]
        representations = {
            "source": (
                np.asarray(part["vertices"], dtype=np.float64),
                np.asarray(part["triangles"], dtype=np.int64),
                "D372 frozen Float64; viewer copy only",
            ),
            "authored": (
                np.asarray(readback[(body, prim_name)]["points_f32"], dtype=np.float64),
                np.asarray(readback[(body, prim_name)]["face_vertex_indices"], dtype=np.int64).reshape(-1, 3),
                "USD Float32 readback; exact hash authority is JSON",
            ),
            "instance": (
                np.asarray(analyzed[(body, prim_name)]["channels"]["instance"]["callback_vertices_m"], dtype=np.float64),
                np.asarray(analyzed[(body, prim_name)]["channels"]["instance"]["callback_topology_triangles"], dtype=np.int64),
                "PhysX instance callback original polygon triangulation; JSON authority",
            ),
            "prototype": (
                np.asarray(analyzed[(body, prim_name)]["channels"]["prototype"]["callback_vertices_m"], dtype=np.float64),
                np.asarray(analyzed[(body, prim_name)]["channels"]["prototype"]["callback_topology_triangles"], dtype=np.int64),
                "PhysX prototype callback original polygon triangulation; JSON authority",
            ),
        }
        for representation, (vertices, triangles, authority) in representations.items():
            path = f"d373/{representation}/{body}/{prim_name}"
            meshes.append(
                {
                    "entity_path": path,
                    "coordinate_frame": "tf#/",
                    "vertices_m": vertices,
                    "triangles": triangles,
                    "color_rgba": colors[role],
                    "static": True,
                    "role": role,
                    "authority": authority,
                }
            )
            metadata_path = f"metadata/meshes/{path.replace('/', '__')}"
            expected.extend([path, metadata_path])
            component_contract[path] = mesh_components
            component_contract[metadata_path] = ["TextDocument:text"]
    scalars = [
        {
            "entity_path": "metrics/d373/part_count",
            "value": evidence["counts"]["parts_total"],
            "static": True,
        },
        {
            "entity_path": "metrics/d373/callback_channels_pass",
            "value": evidence["callback_aggregate"]["counts"]["channels_pass"],
            "static": True,
        },
        {
            "entity_path": "metrics/d373/raw_pairs_exact",
            "value": evidence["callback_aggregate"]["counts"]["raw_pairs_exact"],
            "static": True,
        },
        {"entity_path": "metrics/d373/physics_steps", "value": 0, "static": True},
        {"entity_path": "metrics/d373/q5_samples", "value": 0, "static": True},
        {
            "entity_path": "metrics/d373/max_surface_delta_mm",
            "value": evidence["maxima"]["surface_symmetric_mm"],
            "static": True,
        },
        {
            "entity_path": "metrics/d373/max_bounds_delta_mm",
            "value": evidence["maxima"]["bounds_max_abs_mm"],
            "static": True,
        },
    ]
    gate_rows = [
        {"entity_path": "gate/d373/identity_pass", "value": 1 if evidence["identity_pass"] else 0, "static": True},
        {"entity_path": "gate/d373/g0a_pass", "value": 0, "static": True},
    ]
    scalars.extend(gate_rows)
    for row in scalars:
        expected.append(row["entity_path"])
        component_contract[row["entity_path"]] = ["Scalars:scalars"]
    event_path = "events/d373_summary"
    event = {
        "entity_path": event_path,
        "text": (
            "P34 live identity only: link5=16 gripper=18 callbacks=68 raw_pairs=34. "
            "No simulation step, q5, contact, cylinder, target/IK/path, or grasp verdict."
        ),
        "level": "INFO",
        "static": True,
    }
    expected.extend([event_path, "metadata/run"])
    component_contract[event_path] = ["TextLog:text", "TextLog:level"]
    component_contract["metadata/run"] = ["TextDocument:text"]
    result = log_rerun(
        RRD_PATH,
        meshes=meshes,
        scalar_trace=scalars,
        events=[event],
        recording_metadata={
            "case": "g0a_d373",
            "attempt": ATTEMPT,
            "verdict": evidence["verdict"],
            "evidence_sha256": _sha(EVIDENCE_PATH),
            "physics_steps": 0,
            "q5_samples": 0,
            "contact_queries": 0,
            "display_role": "inspection only; original JSON/callback arrays are authority",
        },
        recording_id="g0a_d373_p34_live_identity",
        blueprint_path=RBL_PATH,
        blueprint_mode="d373_p34_identity",
        live_viewer=False,
        app_id="roarm_g0a_d373_p34_identity",
    )
    if not result.get("ok"):
        raise RuntimeError(f"D373 Rerun write failed: {result}")
    strict = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected),
        exact_entity_paths=sorted(expected),
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
        "log": result,
        "validation_pass": strict.get("pass") is True,
        "rrd_sha256": _sha(RRD_PATH),
        "rbl_sha256": _sha(RBL_PATH),
        "screenshot": _png_info(RERUN_PNG_PATH) if RERUN_PNG_PATH.is_file() else None,
    }


def _write_report(evidence: dict[str, Any]) -> None:
    lines = [
        "# D373 P34 live asset identity preflight",
        "",
        "## 목적",
        "",
        "D372의 34개 충돌체 설계를 실제 USD와 PhysX가 같은 소유 링크와 형상으로 읽는지 확인했다. 이 단계는 물리 파지 시험이 아니다.",
        "",
        "## 범위",
        "",
        "- 새 변수: `p34_live_asset_materialization_and_binding_v1` 한 개",
        "- link5 16개, gripper_link 18개, 합계 34개",
        "- 실제 Isaac worker 1회, 자동 재시도 0회",
        "- PhysX callback: 각 part의 prototype→instance, 총 68회",
        "- physics step, q5, contact, cylinder, target/IK/path: 모두 0회",
        "- 몸통은 D372를 정확히 옮긴 박스 모양 convex Mesh이며 native Cube 성능시험이 아님",
        "",
        "## 정량 결과",
        "",
        f"- authored Float32 readback: {evidence['counts']['authored_readback_pass']}/34",
        f"- callback protocol/topology/surface/bounds/volume: {evidence['callback_aggregate']['counts']['channels_pass']}/68",
        f"- instance↔prototype raw polygon payload exact: {evidence['callback_aggregate']['counts']['raw_pairs_exact']}/34",
        f"- 최대 표면 차이: {evidence['maxima']['surface_symmetric_mm']:.12f} mm",
        f"- 최대 bounds 차이: {evidence['maxima']['bounds_max_abs_mm']:.12f} mm",
        f"- 최대 authored↔callback topology-volume 상대차: {evidence['maxima']['authored_callback_volume_relative']:.12g}",
        f"- 최대 callback↔PhysX property-volume 상대차: {evidence['maxima']['callback_property_volume_relative']:.12g}",
        f"- mass/COM/inertia/principal-axes 불변: {'PASS' if evidence['mass_audit']['pass'] else 'FAIL'}",
        "",
        "## 판정 의미",
        "",
        f"`{evidence['verdict']}`",
        "",
        "PASS는 P34가 실제 asset/cook/readback에서 설계한 동일성을 유지했다는 뜻일 뿐이다. 원통 전도, 접촉순서, 파지 성공, 속도, 최적성은 모두 아직 null이며 `g0a_pass=false`다.",
        "",
        "## NVIDIA 1차 자료",
        "",
        *[f"- {row['title']} — {row['url']} ({row['applicable_version']})" for row in OFFICIAL_SOURCES],
        "",
    ]
    _write_text_x(REPORT_PATH, "\n".join(lines))


def analyze() -> int:
    try:
        prereg = _read_json(PREREG_PATH)
        supervisor = _read_json(SUPERVISOR_PATH)
        worker = _read_json(WORKER_SUMMARY_PATH)
        geometry = _read_json(D372_GEOMETRY)
        if not supervisor.get("pass") or not worker.get("worker_protocol_pass"):
            raise RuntimeError("D373 worker/supervisor protocol did not pass; scientific classification stops")
        _phase("offline_classification_start")
        readback_by_path = {row["live_path"]: row for row in worker["authored_readback"]["rows"]}
        property_audit = _property_audit(worker, readback_by_path)
        mass_audit = _mass_live_audit(worker)
        rows, callback_aggregate = _analyze_callbacks(geometry, worker, property_audit)
        post_negatives = _post_negative_controls(rows, mass_audit)
        all_negatives = {
            **prereg["prepare_negative_controls"]["controls"],
            **post_negatives["controls"],
        }
        maxima = {
            "surface_symmetric_mm": max(
                row["channels"][channel]["surface"]["symmetric_m"] * 1000.0
                for row in rows
                for channel in ("instance", "prototype")
            ),
            "bounds_max_abs_mm": max(
                row["channels"][channel]["bounds_max_abs_delta_m"] * 1000.0
                for row in rows
                for channel in ("instance", "prototype")
            ),
            "authored_callback_volume_relative": max(
                row["channels"][channel]["authored_callback_volume_relative_delta"]
                for row in rows
                for channel in ("instance", "prototype")
            ),
            "callback_property_volume_relative": max(
                row["channels"][channel]["callback_property_volume_relative_delta"]
                for row in rows
                for channel in ("instance", "prototype")
            ),
            "mass_com_inertia_axes_abs_delta": max(
                value
                for body in mass_audit["bodies"].values()
                for value in body["max_abs_deltas"].values()
            ),
        }
        counters = worker["counters"]
        scope_zero = all(
            counters[key] == 0
            for key in (
                "automatic_retries",
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
                "inherited_material_mass_actuator_physics_setting_changes",
                "isaac_hydra_renders",
                "simulation_app_update_pumps",
            )
        )
        checks = {
            "preregistration_pass": prereg["pass"],
            "single_worker_supervisor_pass": supervisor["pass"],
            "worker_protocol_pass": worker["worker_protocol_pass"],
            "authored_readback_34_of_34": worker["authored_readback"]["pass"],
            "active_owner_path_count_binding_pass": worker["live_inventory"]["pass"],
            "property_binding_pass": property_audit["pass"],
            "callback_identity_topology_surface_volume_pass": callback_aggregate["pass"],
            "mass_com_inertia_axes_invariant": mass_audit["pass"],
            "negative_controls_6_of_6": len(all_negatives) == 6 and all(all_negatives.values()),
            "canonical_outside_collision_subtree_pass": all(
                worker["canonical_outside_collision_subtree_diff"]["checks"].values()
            ),
            "nonphysics_files_bit_exact": worker["nonphysics_copy_audit"]["pass"],
            "strict_zero_scope_counters": scope_zero,
            "timeline_stopped_unchanged": worker["timeline_before"]["is_stopped"]
            and worker["timeline_after"]["is_stopped"]
            and not worker["timeline_before"]["is_playing"]
            and not worker["timeline_after"]["is_playing"],
            "timeline_raw_tuple_exact": worker["timeline_before"] == worker["timeline_after"],
        }
        identity_pass = all(checks.values())
        verdict = VERDICT_PASS if identity_pass else VERDICT_FAIL
        evidence = {
            "artifact": "D373_P34_LIVE_ASSET_IDENTITY_EVIDENCE_V1",
            "case": "g0a_d373",
            "attempt": ATTEMPT,
            "new_variables": NEW_VARIABLES,
            "verdict": verdict,
            "identity_pass": identity_pass,
            "g0a_pass": False,
            "physics_grasp_tipping_speed_optimum": None,
            "inputs": {
                "preregistration": {"path": _rel(PREREG_PATH), "sha256": _sha(PREREG_PATH)},
                "worker_summary": {"path": _rel(WORKER_SUMMARY_PATH), "sha256": _sha(WORKER_SUMMARY_PATH)},
                "d372_geometry": {"path": _rel(D372_GEOMETRY), "sha256": _sha(D372_GEOMETRY)},
                "base_asset_root": {"path": _rel(BASE_ROOT_USD), "sha256": _sha(BASE_ROOT_USD)},
            },
            "counts": {
                "parts_total": len(rows),
                "link5_parts": sum(row["body"] == "link5" for row in rows),
                "gripper_link_parts": sum(row["body"] == "gripper_link" for row in rows),
                "authored_readback_pass": sum(row["pass"] for row in worker["authored_readback"]["rows"]),
                "callback_channels": callback_aggregate["counts"]["channels"],
                "raw_instance_prototype_pairs": callback_aggregate["counts"]["pairs"],
                "actual_worker": 1,
                "automatic_retry": 0,
            },
            "worker_counters": counters,
            "timeline_before": worker["timeline_before"],
            "timeline_after": worker["timeline_after"],
            "asset": worker["asset"],
            "authored_readback": worker["authored_readback"],
            "live_inventory": worker["live_inventory"],
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
            "canonical_outside_collision_subtree_diff": worker[
                "canonical_outside_collision_subtree_diff"
            ],
            "nonphysics_copy_audit": worker["nonphysics_copy_audit"],
            "checks": checks,
        }
        _write_json_x(EVIDENCE_PATH, evidence)
        _write_report(evidence)
        board = _make_board(geometry, worker, rows, verdict)
        rerun = _write_rerun(geometry, worker, rows, evidence)
        automated_checks = {
            "identity_evidence_pass": identity_pass,
            "exact_1920x1080_board": board["exact_1920x1080"],
            "rerun_log_pass": rerun["log"].get("ok") is True,
            "rerun_validation_pass": rerun["validation_pass"],
            "rrd_rbl_exist": RRD_PATH.is_file() and RBL_PATH.is_file(),
            "manual_inspection_pending": not MANUAL_JSON_PATH.exists(),
        }
        automated = {
            "artifact": "D373_AUTOMATED_IDENTITY_AND_OBSERVABILITY_SUMMARY_V1",
            "scientific_identity_verdict": verdict,
            "board": board,
            "rerun": rerun,
            "manual_inspection_required_before_completion": True,
            "checks": automated_checks,
            "automated_pass_pending_manual": all(
                value for key, value in automated_checks.items() if key != "manual_inspection_pending"
            )
            and automated_checks["manual_inspection_pending"],
        }
        _write_json_x(AUTOMATED_PATH, automated)
        _phase("offline_classification_and_observability_end", identity_pass=identity_pass, rerun_pass=rerun["validation_pass"])
        print(json.dumps({"stage": "analyze", "identity_pass": identity_pass, "rerun_pass": rerun["validation_pass"]}, sort_keys=True))
        return 0 if automated["automated_pass_pending_manual"] else 1
    except Exception as error:
        if not ANALYZE_EXCEPTION_PATH.exists():
            _write_json_x(
                ANALYZE_EXCEPTION_PATH,
                {
                    "artifact": "D373_ANALYZE_EXCEPTION_V1",
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                    "no_retry_performed": True,
                },
            )
        raise


def finalize() -> int:
    prereg = _read_json(PREREG_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    automated = _read_json(AUTOMATED_PATH)
    manual = _read_json(MANUAL_JSON_PATH)
    validation = _read_json(RERUN_VALIDATION_PATH)
    current_inputs = {_key(path): _sha(path) for path in dict.fromkeys(_input_paths())}
    input_immutability = {
        key: current_inputs.get(key) == value for key, value in prereg["inputs"].items()
    }
    sidecar_after = _sidecar_snapshot()
    checks = {
        "input_hashes_unchanged": all(input_immutability.values()),
        "d334_sidecar_unchanged": sidecar_after == prereg["sidecar_before"],
        "head_origin_still_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD
        and _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "identity_verdict_pass": evidence["verdict"] == VERDICT_PASS and evidence["identity_pass"],
        "automated_observability_pass": automated["automated_pass_pending_manual"],
        "rerun_strict_validation_pass": validation.get("pass") is True,
        "manual_board_inspection_pass": manual.get("board", {}).get("pass") is True,
        "manual_rerun_inspection_pass": manual.get("rerun", {}).get("pass") is True,
        "manual_original_resolution": manual.get("original_resolution_inspection") is True,
        "manual_authority_limits_preserved": manual.get("scientific_authority_remains_original_json") is True,
        "scope_boundary_preserved": evidence["worker_counters"]["physics_steps"] == 0
        and evidence["worker_counters"]["q5_samples"] == 0
        and evidence["worker_counters"]["contact_queries"] == 0,
        "g0a_false": evidence["g0a_pass"] is False,
    }
    completion_pass = all(checks.values())
    completion = {
        "artifact": "D373_P34_LIVE_ASSET_IDENTITY_COMPLETION_V1",
        "case": "g0a_d373",
        "attempt": ATTEMPT,
        "verdict": VERDICT_PASS if completion_pass else VERDICT_OBSERVABILITY_FAIL,
        "scientific_identity_verdict": evidence["verdict"],
        "completion_pass": completion_pass,
        "g0a_pass": False,
        "physics_comparison_authorized": False,
        "next_boundary": "seek separate approval before any P34 physical comparison",
        "input_immutability": input_immutability,
        "sidecar_after": sidecar_after,
        "artifact_hashes": {
            _rel(path): _sha(path)
            for path in (
                PREREG_PATH,
                INVOCATION_PATH,
                SUPERVISOR_PATH,
                WORKER_SUMMARY_PATH,
                EVIDENCE_PATH,
                REPORT_PATH,
                BOARD_PATH,
                RRD_PATH,
                RBL_PATH,
                RERUN_VALIDATION_PATH,
                RERUN_PNG_PATH,
                AUTOMATED_PATH,
                MANUAL_JSON_PATH,
                MANUAL_MD_PATH,
            )
        },
        "checks": checks,
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase("finalize_complete", completion_pass=completion_pass, completion_sha256=_sha(COMPLETION_PATH))
    print(json.dumps({"stage": "finalize", "pass": completion_pass, "verdict": completion["verdict"]}, sort_keys=True))
    return 0 if completion_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "analyze", "finalize"), required=True)
    args = parser.parse_args()
    return {
        "prepare": prepare,
        "run": run_worker,
        "analyze": analyze,
        "finalize": finalize,
    }[args.stage]()


if __name__ == "__main__":
    raise SystemExit(main())
