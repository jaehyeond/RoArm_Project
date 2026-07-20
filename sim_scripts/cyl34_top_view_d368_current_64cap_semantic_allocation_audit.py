#!/usr/bin/env python3
"""D368 offline audit of semantic allocation in the frozen current 64+64 colliders.

This harness never launches SimulationApp, Kit, Isaac, PhysX, Warp, CUDA, or a
physics/contact/q5 path.  It reads the frozen authored mesh and the D348 original
callback polygon evidence, writes one authoritative Float64 JSON audit, and then
creates display-only Rerun/PNG copies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d368"
PREREG_PATH = OUT_DIR / "d368_preregistration.json"
INVOCATION_PATH = OUT_DIR / "d368_audit_invocation.json"
PHASE_PATH = OUT_DIR / "d368_phase_markers.jsonl"
EVIDENCE_PATH = OUT_DIR / "d368_semantic_allocation_evidence.json"
AUTOMATED_PATH = OUT_DIR / "d368_automated_summary.json"
REPORT_PATH = OUT_DIR / "d368_automated_report.md"
RRD_PATH = OUT_DIR / "d368_semantic_allocation.rrd"
RBL_PATH = OUT_DIR / "d368_semantic_allocation.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d368_rerun_validation.json"
RERUN_PNG_PATH = OUT_DIR / "d368_semantic_allocation_rerun.png"
SUMMARY_PNG_PATH = OUT_DIR / "d368_semantic_allocation_summary_1920x1080.png"
MANUAL_JSON_PATH = OUT_DIR / "d368_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d368_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d368_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d368_runtime_exception.json"

HARNESS = Path(__file__).resolve()
SESSION_DOC = REPO / "claudedocs/session_20260720_grasp_g0a_d368_current_64cap_semantic_allocation_audit.md"
START_HERE = REPO / "START_HERE.md"
AGENTS = REPO / "AGENTS.md"
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"

AUTHORING_USD = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"
URDF_PATH = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
D348_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_callback_topology_volume_evidence.json"
D350_BINDING = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d350/d350_fixed_jaw_semantic_binding.json"
D350_MEASUREMENT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d350/d350_fixed_jaw_geometry_measurement.json"
D354_BINDING = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354/d354_moving_jaw_surface_binding.json"
D359_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d359/d359_historical_hash_provenance_evidence.json"
D359_HARNESS = REPO / "sim_scripts/cyl34_top_view_d359_d351_historical_hash_generator_lineage.py"
D339_LINK5_WITNESS = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2/d339_link5_cold1_callback_witness.json"
D339_GRIPPER_WITNESS = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2/d339_gripper_link_cold1_callback_witness.json"
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"

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
PXR_ROOT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
EXPECTED_PYTHONPATH = str(PXR_ROOT)
EXPECTED_LD_LIBRARY_PATH = "/home/cgxr/miniconda3/envs/isaaclab/lib:" + str(PXR_ROOT / "bin")
EXPECTED_MPLCONFIGDIR = "/tmp/roarm_d368_matplotlib_cache"
EXPECTED_HEAD = "7c4819632bb193c8fd552372c919f8a107675b41"
EXPECTED_USD_VERSION = [0, 24, 5]
AUDIT_TIMEOUT_SECONDS = 900
PROXIMITY_TIE_M = 1.0e-9
PLANE_TIE_M = 1.0e-9
NORMAL_DOT_TOL = 1.0e-12
FROZEN_FACE_SOURCE_RESIDUAL_M = 0.0005

NEW_VARIABLES = [
    "semantic_contact_patch_authority",
    "current_64cap_part_to_patch_allocation",
]
FORBIDDEN_MODULE_PREFIXES = ("isaacsim", "omni", "physx", "carb", "warp", "torch")

EXPECTED_INPUT_HASHES = {
    AUTHORING_USD: "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
    URDF_PATH: "64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2",
    D348_EVIDENCE: "83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6",
    D350_BINDING: "1ec1c309461357eeae89204fa55a498b64d2d216708ab6e6c7dfdd3d0b878c12",
    D350_MEASUREMENT: "4fe91e4cd37f5b0f064c7e9c91480881973ca51e651132af2c8feb57750e8446",
    D354_BINDING: "548d45ec4eb1dacbb4cbdefe2b64a3ed99ce72f4f5ffaaa6a9ee1e2b38756b15",
    D359_EVIDENCE: "9a4c2aa38bfc8e26722852a328d5f228aeccba17e372b017767f4da7c281f822",
    D359_HARNESS: "961939863649f483f00ce667b347bfe79f38bb623eb713b38bad084930762ea3",
    D339_LINK5_WITNESS: "e705e7ed5d3a9b4803eaeeac67e60b6494edba37cfecf9226f86e72d60b73a43",
    D339_GRIPPER_WITNESS: "08a500ed54a5c42c02a77b981b96d629a7de9b09df593c9f25dc3698d7220a69",
}

RAW_STREAM_EXPECTED = {
    "link5": {
        "vertex_count": 42276,
        "triangle_count": 14092,
        "raw_vertex_stream_sha256": "93a8a526e7a7efaaa3df2dfacfca7a594308ac88670a13d7fdfd0797b5f4e4dc",
        "triangle_stream_sha256": "055c324299bb84cb54ca118b0f20f6de9cd6291572225613426ac670051ab2b8",
    },
    "gripper_link": {
        "vertex_count": 41094,
        "triangle_count": 13698,
        "raw_vertex_stream_sha256": "522a4f0fe91a04bf54c5c8be6492748c7490fc557fa8c0867200d97332dfa9db",
        "triangle_stream_sha256": "205a08458b895d96c6eb9593d1f04a8815629f7f972a889cce683b86955f2545",
        "authored_points_f32_mm_sha256": "b89c67e99bd253ae710e6b0a2fcacd0b27263d6ede29fe6f6334ed70247895ed",
        "face_counts_i64_sha256": "f17eac58b9b109f98f7a69efcc3b1e64b632d805ccca8cc8883cf0349e07cb6c",
    },
}

EXPECTED_D350_COMPONENT = {
    "seed_face": 1984,
    "face_count": 7250,
    "unique_vertex_count": 3519,
    "component_count": 6,
    "digest": "8f64ddb03308521ce905d0714def9b72e1e69871d2f9f13ea3bd2a3f07559a4d",
}
EXPECTED_FIXED_PATCH = {
    "face_count": 267,
    "unique_vertex_count": 255,
    "face_id_min": 1740,
    "face_id_max": 2006,
    "face_ids_sha256": "caaf745b81ce1af79ef7381cadbc533e786af58d6b706de4fb997f08b33ac06b",
    "digest": "6d896eceb21ec0d92b8bc2b8b9262b04ca601ee28954d56799b3e8a026e931fb",
}

INNER_FACE_IDS = np.arange(672, 1165, dtype=np.int64)
OUTER_FACE_IDS = np.arange(13205, 13698, dtype=np.int64)
EXPECTED_MOVING_BUNDLE = {
    "inner_vertex": "13c65ee478a2668896ec2a8f1e237a9ba7b7e6e0ef40ab08cb350087d3a74d55",
    "outer_vertex": "0d9f7f856eb66d5f749303aa7f4bac8138a595d228dff8424221a6b0b732772a",
    "inner_triangle": "5644e9a66386d68945d340a46cfa9e1507b6dd55cf0b721823ef6afb079b9e17",
    "outer_triangle": "5644e9a66386d68945d340a46cfa9e1507b6dd55cf0b721823ef6afb079b9e17",
    "inner_patch": "c927e8c628073f9f1d8fc0250d8190a71bb2b0701b97b41d7f8069b216c3531b",
    "outer_patch": "9b430c7d7e8c389eb648726014d61169aa671ec910f94a782084b467e96d6486",
    "inner_paired_xz": "917b7154601d23984db01ebbd4adccdb272771920f225d1a021540b4b03bcaf9",
    "outer_paired_xz": "917b7154601d23984db01ebbd4adccdb272771920f225d1a021540b4b03bcaf9",
}
EXPECTED_LIVE_INNER = {
    "face_count": 40,
    "part_count": 17,
    "face_key_sha256": "5bb7ad8a21826cb0709da55f85b0e3772114a782e1263483c180963aa9eccab5",
    "part_names": [
        "part_030", "part_035", "part_042", "part_045", "part_046", "part_047",
        "part_048", "part_050", "part_051", "part_053", "part_056", "part_058",
        "part_059", "part_060", "part_061", "part_062", "part_063",
    ],
}

VERDICT_MEASURED = "D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_MEASURED_NO_PHYSICS"
VERDICT_FAIL = "D368_SEMANTIC_AUTHORITY_OR_ALLOCATION_INTEGRITY_FAIL_STOP"
VERDICT_OBSERVABILITY_FAIL = "D368_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"

EXPECTED_MANUAL_CHECKS = {
    "opened_both_pngs_original_resolution",
    "summary_four_panels_nonblank",
    "rerun_four_spatial_views_nonblank",
    "link5_cyan_seed_patch_and_certified_carrier_state_visible",
    "moving_cyan_inner_patch_and_certified_carrier_state_visible",
    "purple_outer_and_yellow_dual_legend_consistent",
    "metric_and_scope_text_legible_no_overlap",
    "no_unknown_timeline_or_empty_panel",
    "display_only_and_optimality_null_label_visible",
}


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _png_dimensions(path: Path) -> list[int]:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return [int(image.width), int(image.height)]


def _blob(array: Any, dtype: str) -> bytes:
    return np.ascontiguousarray(array, dtype=dtype).tobytes(order="C")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return _rel(value) if value.resolve().is_relative_to(REPO.resolve()) else str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


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
        stream.write(text.rstrip() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _phase(name: str, **fields: Any) -> None:
    row = {"phase": name, "monotonic_ns": time.monotonic_ns(), **fields}
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _input_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in EXPECTED_INPUT_HASHES}


def _expected_input_hashes() -> dict[str, str]:
    return {_rel(path): digest for path, digest in EXPECTED_INPUT_HASHES.items()}


def _dynamic_hashes() -> dict[str, str]:
    paths = [HARNESS, SESSION_DOC, START_HERE, AGENTS, VIZ_DEBUG, RERUN_CONTRACT]
    return {_rel(path): _sha(path) for path in paths}


def _sidecar_snapshot() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if D334_SIDECAR.exists():
        for path in sorted(item for item in D334_SIDECAR.rglob("*") if item.is_file()):
            rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha(path)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"file_count": len(rows), "inventory_sha256": _sha_bytes(canonical), "files": rows}


def _forbidden_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_MODULE_PREFIXES)
    )


def _schema_facts() -> dict[str, Any]:
    schema = PHYSX_SCHEMA.read_text(encoding="utf-8")
    database = PHYSX_PROPERTY_DB.read_text(encoding="utf-8")
    facts = {
        "installed_physx_schema_extension": "107.3.26+107.3.3",
        "installed_physx_sdk_semver": None,
        "public_physx_reference_version": "5.6.1",
        "public_physx_reference_version_matches_installed_runtime": None,
        "schema_path": str(PHYSX_SCHEMA),
        "schema_sha256": _sha(PHYSX_SCHEMA),
        "property_database_path": str(PHYSX_PROPERTY_DB),
        "property_database_sha256": _sha(PHYSX_PROPERTY_DB),
        "schema_sha256_exact": _sha(PHYSX_SCHEMA)
        == "fe075bce4bde5ba7db69c6ccef0c4c26909336ab34c619129fc276f7cb4d7abc",
        "property_database_sha256_exact": _sha(PHYSX_PROPERTY_DB)
        == "4c46d4ae503f0608770a0e130f037e6eae51971007f1b0a570704c56a1fe01b5",
        "schema_default_hull_vertex_limit_64": "physxConvexDecompositionCollision:hullVertexLimit = 64" in schema,
        "schema_default_max_convex_hulls_32": "physxConvexDecompositionCollision:maxConvexHulls = 32" in schema,
        "ui_hull_vertex_range_8_64": 'physxConvexDecompositionCollision:hullVertexLimit": InfoData(8, 64, 1)' in database,
        "ui_max_convex_hulls_range_1_2048": 'physxConvexDecompositionCollision:maxConvexHulls": InfoData(1, 2048, 1)' in database,
        "official_urls": [
            "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html",
            "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/current_limitations.html",
            "https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html",
            "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html",
        ],
        "distinction": {
            "schema_default": {"maxConvexHulls": 32, "hullVertexLimit": 64},
            "ui_authoring_range": {"maxConvexHulls": [1, 2048], "hullVertexLimit": [8, 64]},
            "project_authored_candidate": {"maxConvexHulls": 64, "hullVertexLimit": 64},
            "optimality_claim": None,
            "physx_5_6_1_role": (
                "supplementary official corroboration; no claim that Isaac Sim 5.1 embeds this exact SDK semver"
            ),
        },
    }
    facts["pass"] = all(
        facts[key]
        for key in [
            "schema_default_hull_vertex_limit_64",
            "schema_default_max_convex_hulls_32",
            "ui_hull_vertex_range_8_64",
            "ui_max_convex_hulls_range_1_2048",
            "schema_sha256_exact",
            "property_database_sha256_exact",
        ]
    )
    return facts


def _environment_preflight() -> dict[str, Any]:
    before = _forbidden_modules()
    import matplotlib
    import rerun as rr
    import rtree
    import scipy
    import trimesh
    from pxr import Gf, Usd, UsdGeom
    from roarm_rl.viz_debug import build_rerun_blueprint

    smoke = Usd.Stage.CreateInMemory()
    blueprint = build_rerun_blueprint("d368_semantic_allocation")
    cli = subprocess.run(
        [str(RERUN_CLI), "--version"], check=False, capture_output=True, text=True, timeout=30
    )
    after = _forbidden_modules()
    report = {
        "python_executable": str(Path(sys.executable).resolve()),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "path_first": os.environ.get("PATH", "").split(os.pathsep)[0],
        "mplconfigdir": os.environ.get("MPLCONFIGDIR"),
        "openusd_version": list(Usd.GetVersion()),
        "module_origins": {
            "Gf": str(Path(Gf.__file__).resolve()),
            "Usd": str(Path(Usd.__file__).resolve()),
            "UsdGeom": str(Path(UsdGeom.__file__).resolve()),
        },
        "versions": {
            "numpy": np.__version__,
            "psutil": importlib.metadata.version("psutil"),
            "rerun_sdk": str(rr.__version__),
            "trimesh": str(trimesh.__version__),
            "scipy": str(scipy.__version__),
            "rtree": str(rtree.__version__),
            "matplotlib": str(matplotlib.__version__),
        },
        "rerun_cli": {
            "path": str(RERUN_CLI),
            "returncode": cli.returncode,
            "stdout": cli.stdout.strip(),
            "stderr": cli.stderr.strip(),
        },
        "forbidden_modules_before": before,
        "forbidden_modules_after": after,
        "smoke_stage_valid": smoke is not None and smoke.GetPseudoRoot().IsValid(),
        "blueprint_constructed": blueprint is not None,
    }
    checks = {
        "python_exact": Path(report["python_executable"]) == EXPECTED_PYTHON.resolve(),
        "pythonpath_exact": report["pythonpath"] == EXPECTED_PYTHONPATH,
        "ld_library_path_exact": report["ld_library_path"] == EXPECTED_LD_LIBRARY_PATH,
        "path_uses_isaaclab_bin_first": Path(report["path_first"]).resolve() == EXPECTED_PYTHON.parent.resolve(),
        "matplotlib_cache_is_registered_tmp": report["mplconfigdir"] == EXPECTED_MPLCONFIGDIR,
        "openusd_0_24_5": report["openusd_version"] == EXPECTED_USD_VERSION,
        "numpy_pin": report["versions"]["numpy"] == "1.26.0",
        "psutil_pin": report["versions"]["psutil"] == "5.9.8",
        "rerun_pin": report["versions"]["rerun_sdk"] == "0.34.1",
        "rerun_cli_pin": cli.returncode == 0 and "0.34.1" in (cli.stdout + cli.stderr),
        "pxr_modules_from_registered_root": all(
            Path(path).resolve().is_relative_to(PXR_ROOT.resolve())
            for path in report["module_origins"].values()
        ),
        "standalone_usd_smoke": report["smoke_stage_valid"],
        "d368_blueprint_constructed": report["blueprint_constructed"],
        "no_isaac_kit_physx_warp_torch_modules": before == after == [],
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def _success_inventory() -> list[str]:
    return sorted(
        path.name
        for path in [
            PREREG_PATH,
            INVOCATION_PATH,
            PHASE_PATH,
            EVIDENCE_PATH,
            AUTOMATED_PATH,
            REPORT_PATH,
            RRD_PATH,
            RBL_PATH,
            RERUN_VALIDATION_PATH,
            RERUN_PNG_PATH,
            SUMMARY_PNG_PATH,
            MANUAL_JSON_PATH,
            MANUAL_MD_PATH,
            COMPLETION_PATH,
        ]
    )


def _prepare() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"refusing nonempty forward-only output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [HARNESS, SESSION_DOC, START_HERE, AGENTS, VIZ_DEBUG, RERUN_CONTRACT, *EXPECTED_INPUT_HASHES]:
        if not path.is_file():
            raise FileNotFoundError(path)
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/master")
    environment = _environment_preflight()
    schema = _schema_facts()
    actual_inputs = _input_hashes()
    expected_inputs = _expected_input_hashes()
    checks = {
        "head_expected": head == EXPECTED_HEAD,
        "head_equals_origin_master": head == origin,
        "frozen_inputs_exact": actual_inputs == expected_inputs,
        "environment_preflight_pass": environment["pass"],
        "installed_nvidia_schema_facts_pass": schema["pass"],
        "new_variable_count_two": len(NEW_VARIABLES) == 2,
        "output_empty_before_preregistration": not any(OUT_DIR.iterdir()),
    }
    prereg = {
        "artifact": "D368_PREREGISTRATION_V1",
        "case": "g0a_d368",
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "head": head,
        "origin_master": origin,
        "git_status_before_prepare": _git("status", "--short").splitlines(),
        "dynamic_hashes": _dynamic_hashes(),
        "input_hashes": actual_inputs,
        "expected_input_hashes": expected_inputs,
        "d334_sidecar_before": _sidecar_snapshot(),
        "environment_preflight": environment,
        "nvidia_official_source_contract": schema,
        "registered_fixed_patch": EXPECTED_FIXED_PATCH,
        "registered_moving_inner": EXPECTED_LIVE_INNER,
        "semantic_rules": {
            "link5": (
                "D350 seed face 1984 support plane within 1e-9m, same oriented normal within "
                "1-dot <= 1e-12, exact-welded edge-connected, BFS/union-find traversal agreement"
            ),
            "gripper_link": (
                "frozen inner raw faces 672..1164 (-localY), outer negative faces 13205..13697 "
                "(+localY), original-point-ID identity and D354 live-inner subpartition replay"
            ),
            "current_live_authority": "D348 instance.vertices_m + original witness polygons + instance.topology_triangles",
            "forbidden_authorities": ["qhull_triangles", "D339 current convex geometry", "Rerun Float32 geometry"],
            "certified_classification_policy": (
                "preserve mixed_certified and no_certified_contact_face; never force a whole hull "
                "into a raw semantic label from nearest distance alone"
            ),
        },
        "registered_metrics": [
            "certified carriers plus recurrent/union nearest-sample attribution diagnostics",
            "callback vertices/polygons/max vertices per polygon/topology triangles",
            "overlap-prone whole-carrier part-volume sum",
            "raw-patch to live and certified-live to raw sampled max/P95/RMS",
            "normal disagreement and D350 seed witness nearest part",
            "per-part tie-inclusive nearest-region sample incidence plus mixed-certified/no-certified-face inventory",
            "offline GPU geometry limits only; actual GPU contact execution null",
        ],
        "sampling_contract": {
            "families": ["unique_vertices", "unique_edge_midpoints", "triangle_centroids", "fixed_barycentric_refined"],
            "refined_barycentric_weights": [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
            "nearest_query": "exact point-to-triangle surface query through trimesh/rtree",
            "tie_tolerance_m": PROXIMITY_TIE_M,
            "distance_role": "metric only; no new whole-patch hard fidelity threshold",
        },
        "negative_controls": [
            "owner_swap_rejected",
            "moving_outer_as_inner_rejected_by_normal_sign",
            "callback_topology_to_qhull_authority_substitution_rejected",
            "disabled_legacy_65th_part_rejected",
            "meter_to_millimeter_x1000_rejected",
            "part_order_permutation_changes_ordered_but_not_canonical_hash",
            "coordinate_and_reverse_point_id_remaps_rejected",
            "d350_seed_component_removal_worsens_anchor",
        ],
        "decision_rule": {
            "measured": VERDICT_MEASURED,
            "measurement_fail_stop": VERDICT_FAIL,
            "observability_or_completion_fail_stop": VERDICT_OBSERVABILITY_FAIL,
            "pass_semantics": "allocation inventory/provenance completed, not collider adequacy or optimality",
            "visualization_failure_does_not_override_measurement_verdict": True,
        },
        "null_boundaries": {
            "current_64cap_optimal": None,
            "physics_equivalence": None,
            "collider_count_tipping_causality": None,
            "actual_gpu_contact_execution": None,
            "grasp_feasibility": None,
            "g0a_pass": False,
        },
        "scope_guards": {
            "simulation_app_or_kit": 0,
            "isaac_or_physx": 0,
            "warp_or_cuda_compute": 0,
            "nvidia_smi": 0,
            "cook_or_decomposition": 0,
            "usd_or_asset_writes": 0,
            "q5_science": 0,
            "physics_steps": 0,
            "contact_queries": 0,
            "target_ik_path_changes": 0,
            "material_mass_actuator_physics_changes": 0,
            "d334_sidecar_writes": 0,
            "rerun_display_render_allowed": 1,
        },
        "single_run_contract": {
            "prepare_is_not_audit": True,
            "audit_invocation_count": 1,
            "manual_then_finalize_is_control_only": True,
            "no_retry_or_overwrite": True,
            "timeout_seconds": AUDIT_TIMEOUT_SECONDS,
        },
        "registered_command": {
            "python": str(EXPECTED_PYTHON),
            "python_flags": ["-B"],
            "script": _rel(HARNESS),
            "argv": ["--stage", "audit"],
            "environment": {
                "PYTHONPATH": EXPECTED_PYTHONPATH,
                "LD_LIBRARY_PATH": EXPECTED_LD_LIBRARY_PATH,
                "PATH_prefix": str(EXPECTED_PYTHON.parent),
                "MPLCONFIGDIR": EXPECTED_MPLCONFIGDIR,
            },
        },
        "expected_success_inventory": _success_inventory(),
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    if not prereg["pass"]:
        raise RuntimeError(f"D368 prepare failed: {checks}")
    print(json.dumps({"stage": "prepare", "pass": True, "path": _rel(PREREG_PATH)}))


RAW_MESH_PATHS = {
    "link5": (
        "/roarm_m3/link5/collisions/link5/node_STL_BINARY_/mesh",
        "/roarm_m3/link5",
    ),
    "gripper_link": (
        "/roarm_m3/gripper_link/collisions/gripper_link/node_STL_BINARY_/mesh",
        "/roarm_m3/gripper_link",
    ),
}


def _load_raw_meshes() -> dict[str, dict[str, Any]]:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(AUTHORING_USD), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError("failed to open frozen authoring USD")
    result: dict[str, dict[str, Any]] = {}
    for body, (mesh_path, body_path) in RAW_MESH_PATHS.items():
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        body_prim = stage.GetPrimAtPath(body_path)
        if not mesh_prim.IsValid() or not body_prim.IsValid():
            raise RuntimeError(f"missing frozen prim for {body}")
        mesh = UsdGeom.Mesh(mesh_prim)
        point_rows = list(mesh.GetPointsAttr().Get() or [])
        authored = np.asarray([[float(x) for x in row] for row in point_rows], dtype="<f4")
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype="<i8")
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype="<i8")
        if not np.all(counts == 3) or int(indices.size) != int(counts.size * 3):
            raise RuntimeError(f"{body} source is not the frozen triangle stream")
        triangles = indices.reshape(-1, 3)
        mesh_l2w = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        body_w2l = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        raw_m = np.asarray(
            [
                [float(x) for x in body_w2l.Transform(mesh_l2w.Transform(Gf.Vec3d(*[float(x) for x in row])))]
                for row in point_rows
            ],
            dtype="<f8",
        )
        result[body] = {
            "body": body,
            "mesh_path": mesh_path,
            "body_path": body_path,
            "authored_points_f32_mm": authored,
            "face_counts_i64": counts,
            "face_indices_i64": indices,
            "vertices_m": raw_m,
            "triangles": np.asarray(triangles, dtype="<i8"),
            "stream_summary": {
                "vertex_count": len(raw_m),
                "triangle_count": len(triangles),
                "raw_vertex_stream_sha256": _sha_bytes(_blob(raw_m, "<f8")),
                "triangle_stream_sha256": _sha_bytes(_blob(triangles, "<i8")),
                "authored_points_f32_mm_sha256": _sha_bytes(_blob(authored, "<f4")),
                "face_counts_i64_sha256": _sha_bytes(_blob(counts, "<i8")),
                "face_indices_i64_sha256": _sha_bytes(_blob(indices, "<i8")),
                "bounds_m": [raw_m.min(axis=0).tolist(), raw_m.max(axis=0).tolist()],
            },
        }
    return result


def _component_digest(vertices: np.ndarray, triangles: np.ndarray, face_ids: np.ndarray) -> str:
    return _sha_bytes(_blob(vertices, "<f8") + _blob(triangles, "<i8") + _blob(face_ids, "<i8"))


def _compact_faces(vertices: np.ndarray, triangles: np.ndarray, face_ids: np.ndarray) -> dict[str, Any]:
    ids = np.asarray(sorted(int(x) for x in face_ids), dtype=np.int64)
    selected = np.asarray(triangles[ids], dtype=np.int64)
    unique, inverse = np.unique(np.asarray(vertices, dtype=np.float64), axis=0, return_inverse=True)
    welded = inverse[selected]
    used = np.unique(welded.reshape(-1))
    remap = {int(old): new for new, old in enumerate(used.tolist())}
    compact_tri = np.asarray([[remap[int(x)] for x in row] for row in welded], dtype=np.int64)
    compact_vertices = np.asarray(unique[used], dtype=np.float64)
    return {
        "vertices": compact_vertices,
        "triangles": compact_tri,
        "face_ids": ids,
        "face_ids_sha256": _sha_bytes(_blob(ids, "<i8")),
        "digest": _component_digest(compact_vertices, compact_tri, ids),
    }


def _vertex_connected_component(
    vertices: np.ndarray, triangles: np.ndarray, seed_face: int, *, reverse: bool
) -> dict[str, Any]:
    unique, inverse = np.unique(np.asarray(vertices, dtype=np.float64), axis=0, return_inverse=True)
    welded = inverse[np.asarray(triangles, dtype=np.int64)]
    count = len(welded)
    parent = np.arange(count, dtype=np.int64)
    rank = np.zeros(count, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    owners: dict[int, int] = {}
    order = range(count - 1, -1, -1) if reverse else range(count)
    for face in order:
        for vertex in welded[face]:
            key = int(vertex)
            if key in owners:
                union(face, owners[key])
            else:
                owners[key] = face
    root = find(seed_face)
    face_ids = np.asarray(sorted(i for i in range(count) if find(i) == root), dtype=np.int64)
    compact = _compact_faces(vertices, triangles, face_ids)
    compact.update(
        {
            "component_count": len({find(i) for i in range(count)}),
            "welded_unique_vertex_count": len(unique),
        }
    )
    return compact


def _face_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    rows = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    cross = np.cross(rows[:, 1] - rows[:, 0], rows[:, 2] - rows[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise RuntimeError("nonfinite or degenerate source triangle")
    return cross / lengths[:, None]


def _fixed_support_patch(
    vertices: np.ndarray,
    triangles: np.ndarray,
    broad_face_ids: np.ndarray,
    *,
    d350_oriented_normal_local: np.ndarray,
    d350_seed_local_m: np.ndarray,
) -> dict[str, Any]:
    seed = EXPECTED_D350_COMPONENT["seed_face"]
    tri = np.asarray(triangles, dtype=np.int64)
    verts = np.asarray(vertices, dtype=np.float64)
    rows = verts[tri]
    normals = _face_normals(verts, tri)
    raw_seed_normal = normals[seed]
    seed_normal = np.asarray(d350_oriented_normal_local, dtype=np.float64)
    seed_normal = seed_normal / np.linalg.norm(seed_normal)
    seed_point = rows[seed, 0]
    plane_residual = np.max(np.abs(np.einsum("fvi,i->fv", rows - seed_point, seed_normal)), axis=1)
    normal_dot = np.einsum("fi,i->f", normals, seed_normal)
    candidates = set(
        int(x)
        for x in np.flatnonzero(
            (plane_residual <= PLANE_TIE_M) & (normal_dot >= 1.0 - NORMAL_DOT_TOL)
        )
    )
    unique, inverse = np.unique(verts, axis=0, return_inverse=True)
    welded = inverse[tri]
    edge_owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face, row in enumerate(welded):
        for a, b in ((row[0], row[1]), (row[1], row[2]), (row[2], row[0])):
            edge_owners[tuple(sorted((int(a), int(b))))].append(face)

    # Independent implementation A: explicit adjacency BFS.
    adjacency: dict[int, set[int]] = defaultdict(set)
    for owners in edge_owners.values():
        for left in owners:
            for right in owners:
                if left != right:
                    adjacency[left].add(right)
    bfs_seen = {seed}
    queue: deque[int] = deque([seed])
    while queue:
        face = queue.popleft()
        for neighbor in adjacency[face]:
            if neighbor in candidates and neighbor not in bfs_seen:
                bfs_seen.add(neighbor)
                queue.append(neighbor)
    bfs_ids = np.asarray(sorted(bfs_seen), dtype=np.int64)

    # Independent implementation B: union-find only over full welded edges.
    candidate_order = sorted(candidates, reverse=True)
    parent = {face: face for face in candidates}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for owners in edge_owners.values():
        filtered = [face for face in owners if face in parent]
        for face in candidate_order:
            if face not in filtered:
                continue
            for other in filtered:
                union(face, other)
    seed_root = find(seed)
    union_ids = np.asarray(sorted(face for face in candidates if find(face) == seed_root), dtype=np.int64)
    compact_a = _compact_faces(verts, tri, bfs_ids)
    compact_b = _compact_faces(verts, tri, union_ids)
    broad = set(int(x) for x in broad_face_ids)
    d350_seed_plane_residual_m = float(
        abs(np.dot(np.asarray(d350_seed_local_m, dtype=np.float64) - seed_point, seed_normal))
    )
    checks = {
        "raw_seed_winding_matches_d350_oriented_normal_exact": np.array_equal(
            raw_seed_normal, seed_normal
        ),
        "d350_seed_lies_on_registered_plane_within_1nm": d350_seed_plane_residual_m
        <= PLANE_TIE_M,
        "bfs_union_find_traversal_face_sets_exact": np.array_equal(bfs_ids, union_ids),
        "bfs_union_find_traversal_digests_exact": compact_a["digest"] == compact_b["digest"],
        "subset_of_d350_witness_component": set(map(int, bfs_ids)).issubset(broad),
        "face_count_267": len(bfs_ids) == EXPECTED_FIXED_PATCH["face_count"],
        "unique_vertex_count_255": len(compact_a["vertices"]) == EXPECTED_FIXED_PATCH["unique_vertex_count"],
        "face_range_exact": int(bfs_ids.min()) == EXPECTED_FIXED_PATCH["face_id_min"]
        and int(bfs_ids.max()) == EXPECTED_FIXED_PATCH["face_id_max"],
        "face_id_hash_exact": compact_a["face_ids_sha256"] == EXPECTED_FIXED_PATCH["face_ids_sha256"],
        "digest_exact": compact_a["digest"] == EXPECTED_FIXED_PATCH["digest"],
        "seed_face_retained": seed in set(map(int, bfs_ids)),
    }
    return {
        "artifact": "D368_D350_SEED_CONTACT_PLANE_PATCH_V1",
        "seed_face_index": seed,
        "raw_seed_winding_normal_local": raw_seed_normal.tolist(),
        "seed_normal_local": seed_normal.tolist(),
        "normal_authority": "D350 actual_surface.oriented_surface_normal_local",
        "d350_seed_local_m": np.asarray(d350_seed_local_m, dtype=np.float64).tolist(),
        "d350_seed_plane_residual_m": d350_seed_plane_residual_m,
        "support_plane_offset_m": float(np.dot(seed_normal, seed_point)),
        "candidate_face_count": len(candidates),
        "face_count": len(bfs_ids),
        "unique_vertex_count": len(compact_a["vertices"]),
        "face_id_min": int(bfs_ids.min()),
        "face_id_max": int(bfs_ids.max()),
        "face_ids_sha256": compact_a["face_ids_sha256"],
        "digest": compact_a["digest"],
        "bounds_local_m": [compact_a["vertices"].min(axis=0).tolist(), compact_a["vertices"].max(axis=0).tolist()],
        "checks": checks,
        "pass": all(checks.values()),
        "_face_ids": bfs_ids,
        "_vertices": compact_a["vertices"],
        "_triangles": compact_a["triangles"],
    }


def _patch_hash(points: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, remap: str) -> dict[str, str]:
    ids = np.sort(np.asarray(face_ids, dtype=np.int64))
    selected = np.asarray(faces[ids], dtype=np.int64)
    if remap == "original_point_id_ascending":
        old = np.unique(selected.reshape(-1))
        lookup = {int(value): index for index, value in enumerate(old.tolist())}
        vertices = np.asarray(points[old], dtype="<f4")
        triangles = np.asarray([[lookup[int(value)] for value in row] for row in selected], dtype="<i8")
    elif remap == "coordinate_lexicographic_unique":
        vertices, inverse = np.unique(points[selected].reshape(-1, 3), axis=0, return_inverse=True)
        vertices = np.asarray(vertices, dtype="<f4")
        triangles = np.asarray(inverse.reshape(-1, 3), dtype="<i8")
    elif remap == "reverse_original_point_id":
        old = np.unique(selected.reshape(-1))[::-1]
        lookup = {int(value): index for index, value in enumerate(old.tolist())}
        vertices = np.asarray(points[old], dtype="<f4")
        triangles = np.asarray([[lookup[int(value)] for value in row] for row in selected], dtype="<i8")
    else:
        raise ValueError(remap)
    face_blob = _blob(ids, "<i8")
    vertex_blob = _blob(vertices, "<f4")
    triangle_blob = _blob(triangles, "<i8")
    xz = np.asarray(np.unique(vertices[:, [0, 2]], axis=0), dtype="<f4")
    return {
        "vertex": _sha_bytes(vertex_blob),
        "triangle": _sha_bytes(triangle_blob),
        "patch": _sha_bytes(face_blob + vertex_blob + triangle_blob),
        "paired_xz": _sha_bytes(_blob(xz, "<f4")),
    }


def _moving_identity(raw: dict[str, Any]) -> dict[str, Any]:
    points = raw["authored_points_f32_mm"]
    faces = raw["triangles"]
    rows: dict[str, dict[str, str]] = {}
    for remap in ["original_point_id_ascending", "coordinate_lexicographic_unique", "reverse_original_point_id"]:
        inner = _patch_hash(points, faces, INNER_FACE_IDS, remap)
        outer = _patch_hash(points, faces, OUTER_FACE_IDS, remap)
        rows[remap] = {
            "inner_vertex": inner["vertex"],
            "outer_vertex": outer["vertex"],
            "inner_triangle": inner["triangle"],
            "outer_triangle": outer["triangle"],
            "inner_patch": inner["patch"],
            "outer_patch": outer["patch"],
            "inner_paired_xz": inner["paired_xz"],
            "outer_paired_xz": outer["paired_xz"],
        }
    checks = {
        "original_point_id_reproduces_historical_8_of_8": rows["original_point_id_ascending"] == EXPECTED_MOVING_BUNDLE,
        "coordinate_remap_matches_only_paired_xz_2_of_8": sum(
            rows["coordinate_lexicographic_unique"][key] == EXPECTED_MOVING_BUNDLE[key]
            for key in EXPECTED_MOVING_BUNDLE
        )
        == 2,
        "reverse_point_id_rejected": rows["reverse_original_point_id"] != EXPECTED_MOVING_BUNDLE,
    }
    return {"rows": rows, "expected": EXPECTED_MOVING_BUNDLE, "checks": checks, "pass": all(checks.values())}


def _find_artifact(value: Any, artifact: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("artifact") == artifact:
            return value
        for child in value.values():
            found = _find_artifact(child, artifact)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_artifact(child, artifact)
            if found is not None:
                return found
    return None


def _triangulate_polygons(indices: np.ndarray, polygons: list[dict[str, Any]]) -> np.ndarray:
    rows: list[list[int]] = []
    for polygon in polygons:
        base = int(polygon["index_base"])
        count = int(polygon["num_vertices"])
        face = [int(x) for x in indices[base : base + count]]
        for offset in range(1, count - 1):
            rows.append([face[0], face[offset], face[offset + 1]])
    return np.asarray(rows, dtype=np.int64)


def _load_current_parts() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import trimesh

    evidence = _read_json(D348_EVIDENCE)
    groups: dict[str, list[dict[str, Any]]] = {"link5": [], "gripper_link": []}
    witness_checks: list[bool] = []
    topology_checks: list[bool] = []
    prototype_checks: list[bool] = []
    for row in evidence["rows"]:
        body = str(row["body"])
        if body not in groups:
            raise RuntimeError(f"unexpected D348 body {body}")
        name = str(row["name"])
        witness_path = REPO / row["instance"]["witness_path"]
        witness_hash_ok = witness_path.is_file() and _sha(witness_path) == row["instance"]["witness_sha256"]
        witness_checks.append(witness_hash_ok)
        witness = _read_json(witness_path)
        convexes = witness["events"][0]["convexes"]
        if len(convexes) != 1:
            raise RuntimeError(f"{body}/{name}: witness convex count != 1")
        convex = convexes[0]
        vertices = np.asarray(convex["vertices"], dtype=np.float64)
        polygon_indices = np.asarray(convex["indices"], dtype=np.int64)
        polygons = list(convex["polygons"])
        triangles = _triangulate_polygons(polygon_indices, polygons)
        row_vertices = np.asarray(row["instance"]["vertices_m"], dtype=np.float64)
        row_triangles = np.asarray(row["instance"]["topology_triangles"], dtype=np.int64)
        topology_ok = np.array_equal(vertices, row_vertices) and np.array_equal(triangles, row_triangles)
        topology_checks.append(topology_ok)
        prototype_ok = (
            row["instance"]["payload_sha256"] == row["prototype"]["payload_sha256"]
            and np.array_equal(row_vertices, np.asarray(row["prototype"]["vertices_m"], dtype=np.float64))
            and np.array_equal(row_triangles, np.asarray(row["prototype"]["topology_triangles"], dtype=np.int64))
        )
        prototype_checks.append(prototype_ok)
        sizes = [int(poly["num_vertices"]) for poly in polygons]
        extents = np.ptp(vertices, axis=0)
        positive = extents[extents > 1.0e-12]
        aabb_aspect = float(np.max(positive) / np.min(positive)) if len(positive) else math.inf
        part = {
            "body": body,
            "name": name,
            "path": row["property_query"]["path"],
            "global_part_idx": int(row["global_part_idx"]),
            "vertices": vertices,
            "triangles": triangles,
            "polygon_indices": polygon_indices,
            "polygons": polygons,
            "vertex_count": len(vertices),
            "polygon_count": len(polygons),
            "max_vertices_per_polygon": max(sizes),
            "triangle_count": len(triangles),
            "property_volume_m3": float(row["property_volume_m3"]),
            "payload_sha256": row["instance"]["payload_sha256"],
            "callback_vertices_sha256": _sha_bytes(_blob(vertices, "<f8")),
            "original_polygon_indices_sha256": _sha_bytes(_blob(polygon_indices, "<i8")),
            "original_polygon_descriptors_sha256": _sha_bytes(
                json.dumps(polygons, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
            "callback_topology_sha256": _sha_bytes(_blob(triangles, "<i8")),
            "expected_callback_topology_sha256": _sha_bytes(_blob(row_triangles, "<i8")),
            "witness_path": _rel(witness_path),
            "witness_sha256": row["instance"]["witness_sha256"],
            "aabb_aspect_ratio_diagnostic": aabb_aspect,
            "offline_gpu_geometry_limits": {
                "vertices_le_64": len(vertices) <= 64,
                "polygons_le_64": len(polygons) <= 64,
                "vertices_per_polygon_le_32": max(sizes) <= 32,
                "aabb_aspect_ratio_le_100_diagnostic": aabb_aspect <= 100.0,
            },
            "_qhull_triangles": np.asarray(row["instance"]["qhull_triangles"], dtype=np.int64),
        }
        part["_mesh"] = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False, validate=False)
        groups[body].append(part)
    for body in groups:
        groups[body].sort(key=lambda item: item["name"])
    inventory = {
        "d348_aggregate_pass": evidence.get("pass") is True,
        "part_counts": {body: len(parts) for body, parts in groups.items()},
        "part_names_exact": {
            body: [part["name"] for part in parts] == [f"part_{i:03d}" for i in range(64)]
            for body, parts in groups.items()
        },
        "witness_hashes_exact_count": sum(witness_checks),
        "witness_count": len(witness_checks),
        "witness_topology_exact_count": sum(topology_checks),
        "instance_prototype_payload_exact_count": sum(prototype_checks),
        "gpu_geometry_limit_counts": {
            key: sum(part["offline_gpu_geometry_limits"][key] for body in groups.values() for part in body)
            for key in [
                "vertices_le_64",
                "polygons_le_64",
                "vertices_per_polygon_le_32",
                "aabb_aspect_ratio_le_100_diagnostic",
            ]
        },
        "total_parts": sum(len(parts) for parts in groups.values()),
    }
    inventory["checks"] = {
        "d348_corrected_pass": inventory["d348_aggregate_pass"],
        "64_plus_64": inventory["part_counts"] == {"link5": 64, "gripper_link": 64},
        "names_exact": all(inventory["part_names_exact"].values()),
        "all_witness_hashes_exact": inventory["witness_hashes_exact_count"] == inventory["witness_count"] == 128,
        "all_original_polygon_topology_reconstructed": inventory["witness_topology_exact_count"] == 128,
        "all_instance_prototype_payloads_exact": inventory["instance_prototype_payload_exact_count"] == 128,
        "all_gpu_vertex_polygon_limits_observed": all(
            inventory["gpu_geometry_limit_counts"][key] == 128
            for key in ["vertices_le_64", "polygons_le_64", "vertices_per_polygon_le_32"]
        ),
    }
    inventory["pass"] = all(inventory["checks"].values())
    return groups, inventory


def _trimesh(vertices: np.ndarray, triangles: np.ndarray) -> Any:
    import trimesh

    return trimesh.Trimesh(vertices=vertices, faces=triangles, process=False, validate=False)


def _sample_families(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    unique_vertices = np.unique(vertices[triangles].reshape(-1, 3), axis=0)
    edges = np.sort(
        np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1
    )
    edges = np.unique(edges, axis=0)
    edge_midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])
    tri_points = vertices[triangles]
    centroids = np.mean(tri_points, axis=1)
    weights = np.asarray([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]], dtype=np.float64)
    refined = np.einsum("wi,fij->fwj", weights, tri_points).reshape(-1, 3)
    return {
        "unique_vertices": unique_vertices,
        "unique_edge_midpoints": np.unique(edge_midpoints, axis=0),
        "triangle_centroids": centroids,
        "fixed_barycentric_refined": refined,
    }


def _nearest(mesh: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64)
    _, distances, face_ids = mesh.nearest.on_surface(points)
    return np.asarray(distances, dtype=np.float64), np.asarray(face_ids, dtype=np.int64)


def _distance_stats_m(distances: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(distances), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return {"count": int(values.size), "finite": False}
    mm = values * 1000.0
    return {
        "count": len(values),
        "finite": True,
        "min_mm": float(np.min(mm)),
        "mean_mm": float(np.mean(mm)),
        "p50_mm": float(np.percentile(mm, 50)),
        "p95_mm": float(np.percentile(mm, 95)),
        "max_mm": float(np.max(mm)),
        "rms_mm": float(np.sqrt(np.mean(np.square(mm)))),
    }


def _angle_stats_deg(angles: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(angles), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return {"count": int(values.size), "finite": False}
    return {
        "count": len(values),
        "finite": True,
        "min_deg": float(np.min(values)),
        "mean_deg": float(np.mean(values)),
        "p95_deg": float(np.percentile(values, 95)),
        "max_deg": float(np.max(values)),
        "rms_deg": float(np.sqrt(np.mean(np.square(values)))),
    }


def _carrier_budget(parts: list[dict[str, Any]], names: Iterable[str]) -> dict[str, Any]:
    selected_names = sorted(set(str(name) for name in names))
    selected = [part for part in parts if part["name"] in selected_names]
    return {
        "part_names": selected_names,
        "part_count": len(selected),
        "part_count_denominator": 64,
        "whole_part_vertex_count_sum": sum(part["vertex_count"] for part in selected),
        "whole_part_polygon_count_sum": sum(part["polygon_count"] for part in selected),
        "whole_part_topology_triangle_count_sum": sum(part["triangle_count"] for part in selected),
        "whole_part_property_volume_sum_m3": sum(part["property_volume_m3"] for part in selected),
        "volume_semantics": "overlap-prone whole-carrier diagnostic; not pad volume, unique occupied volume, mass, or material volume",
    }


def _certified_faces(
    parts: list[dict[str, Any]],
    *,
    axis: int,
    plane_value_m: float,
    expected_normal: np.ndarray,
    raw_patch_mesh: Any,
    exact_normal: bool,
) -> dict[str, Any]:
    keys: list[str] = []
    part_names: set[str] = set()
    triangles_out: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    source_vertices: list[np.ndarray] = []
    diagnostics: list[np.ndarray] = []
    for part in parts:
        for face_index, ids in enumerate(part["triangles"]):
            triangle = part["vertices"][ids]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            length = float(np.linalg.norm(normal))
            if not math.isfinite(length) or length <= 0.0:
                continue
            unit = normal / length
            plane_match = bool(np.max(np.abs(triangle[:, axis] - plane_value_m)) <= PLANE_TIE_M)
            normal_match = bool(
                np.array_equal(unit, expected_normal)
                if exact_normal
                else float(np.dot(unit, expected_normal)) >= 1.0 - NORMAL_DOT_TOL
            )
            if plane_match and normal_match:
                keys.append(f"{part['name']}:{face_index}")
                part_names.add(part["name"])
                triangles_out.append(triangle)
                normals.append(unit)
                source_vertices.extend([triangle[0], triangle[1], triangle[2]])
                diagnostics.extend(
                    [
                        0.5 * (triangle[0] + triangle[1]),
                        0.5 * (triangle[1] + triangle[2]),
                        0.5 * (triangle[2] + triangle[0]),
                        np.mean(triangle, axis=0),
                    ]
                )
    source = np.unique(np.asarray(source_vertices, dtype=np.float64), axis=0) if source_vertices else np.zeros((0, 3))
    diag = np.unique(np.asarray(diagnostics, dtype=np.float64), axis=0) if diagnostics else np.zeros((0, 3))
    source_dist, _ = _nearest(raw_patch_mesh, source)
    diag_dist, _ = _nearest(raw_patch_mesh, diag)
    normal_angles = []
    for unit in normals:
        dot = float(np.clip(np.dot(unit, expected_normal), -1.0, 1.0))
        normal_angles.append(math.degrees(math.acos(dot)))
    sorted_keys = sorted(keys)
    report = {
        "face_count": len(sorted_keys),
        "part_count": len(part_names),
        "part_names": sorted(part_names),
        "face_keys": sorted_keys,
        "face_key_sha256": _sha_bytes("\n".join(sorted_keys).encode()),
        "source_vertex_count": len(source),
        "source_vertex_to_raw_patch": _distance_stats_m(source_dist),
        "interior_diagnostic_point_count": len(diag),
        "interior_diagnostic_to_raw_patch": _distance_stats_m(diag_dist),
        "interior_diagnostic_verdict_authority": False,
        "normal_angle_error": _angle_stats_deg(normal_angles),
        "source_vertex_residual_within_inherited_face_identity_0p5mm": bool(
            len(source_dist) and np.isfinite(source_dist).all() and np.max(source_dist) <= FROZEN_FACE_SOURCE_RESIDUAL_M
        ),
        "_triangles": np.asarray(triangles_out, dtype=np.float64),
    }
    return report


def _support_allocation(
    raw_patch_vertices: np.ndarray,
    raw_patch_triangles: np.ndarray,
    parts: list[dict[str, Any]],
    certified: dict[str, Any],
) -> dict[str, Any]:
    families = _sample_families(raw_patch_vertices, raw_patch_triangles)
    method_rows: dict[str, Any] = {}
    method_sets: list[set[str]] = []
    all_min_distances: list[float] = []
    for method, points in families.items():
        matrix = np.column_stack([_nearest(part["_mesh"], points)[0] for part in parts])
        minima = np.min(matrix, axis=1)
        selected: set[str] = set()
        tie_count = 0
        for row, minimum in zip(matrix, minima, strict=True):
            hits = np.flatnonzero(row <= minimum + PROXIMITY_TIE_M)
            tie_count += int(len(hits) > 1)
            selected.update(parts[int(index)]["name"] for index in hits)
        method_sets.append(selected)
        all_min_distances.extend(minima.tolist())
        method_rows[method] = {
            "sample_count": len(points),
            "nearest_attributed_part_names": sorted(selected),
            "nearest_attributed_part_count": len(selected),
            "multi_part_tie_sample_count": tie_count,
            "nearest_live_surface_distance": _distance_stats_m(minima),
            "interpretation": "nearest-surface attribution diagnostic; not proof of contact support",
        }
    recurrent = set.intersection(*method_sets) if method_sets else set()
    union = set.union(*method_sets) if method_sets else set()
    certified_names = set(certified["part_names"])
    certified_triangles = np.asarray(certified["_triangles"], dtype=np.float64)
    certified_surface_stats: dict[str, Any] = {}
    if len(certified_triangles):
        compact_vertices, inverse = np.unique(certified_triangles.reshape(-1, 3), axis=0, return_inverse=True)
        compact_triangles = inverse.reshape(-1, 3)
        raw_mesh = _trimesh(raw_patch_vertices, raw_patch_triangles)
        for method, points in _sample_families(compact_vertices, compact_triangles).items():
            distances, _ = _nearest(raw_mesh, points)
            certified_surface_stats[method] = _distance_stats_m(distances)
    return {
        "sampling_methods": method_rows,
        "recurrent_nearest_sample_part_names": sorted(recurrent),
        "union_nearest_sample_part_names": sorted(union),
        "certified_surface_carrier_part_names": sorted(certified_names),
        "set_relationships": {
            "recurrent_subset_union": recurrent.issubset(union),
            "certified_intersection_recurrent": sorted(certified_names & recurrent),
            "certified_only_vs_recurrent": sorted(certified_names - recurrent),
            "recurrent_only_vs_certified": sorted(recurrent - certified_names),
            "certified_subset_union_observation": certified_names.issubset(union),
            "certified_subset_union_is_not_a_gate": True,
        },
        "raw_patch_to_nearest_live_surface_all_samples": _distance_stats_m(all_min_distances),
        "certified_live_surface_to_raw_patch_by_method": certified_surface_stats,
        "authority_note": (
            "certified_surface_carrier_part_names are authoritative under the registered face contract; "
            "recurrent/union sets are finite-sample nearest-surface diagnostics only"
        ),
        "carrier_budgets": {
            "certified": _carrier_budget(parts, certified_names),
            "recurrent_nearest_sample": _carrier_budget(parts, recurrent),
            "union_nearest_sample": _carrier_budget(parts, union),
        },
    }


def _semantic_association(
    parts: list[dict[str, Any]],
    region_meshes: dict[str, Any],
    certified_region_parts: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    region_names = sorted(region_meshes)
    for part in parts:
        families = _sample_families(part["vertices"], part["triangles"])
        method_rows: dict[str, Any] = {}
        method_region_sets: list[set[str]] = []
        all_minima: list[float] = []
        for method, points in families.items():
            matrix = np.column_stack([_nearest(region_meshes[name], points)[0] for name in region_names])
            minima = np.min(matrix, axis=1)
            hit_counts: Counter[str] = Counter()
            tied = 0
            observed: set[str] = set()
            for dist_row, minimum in zip(matrix, minima, strict=True):
                hits = np.flatnonzero(dist_row <= minimum + PROXIMITY_TIE_M)
                tied += int(len(hits) > 1)
                for index in hits:
                    name = region_names[int(index)]
                    observed.add(name)
                    hit_counts[name] += 1
            method_region_sets.append(observed)
            all_minima.extend(minima.tolist())
            method_rows[method] = {
                "sample_count": len(points),
                "nearest_observed_regions": sorted(observed),
                "nearest_region_hit_counts_including_ties": dict(sorted(hit_counts.items())),
                "nearest_region_incidence_fraction_including_ties": {
                    name: float(count / len(points)) for name, count in sorted(hit_counts.items())
                },
                "incidence_fraction_sum_may_exceed_one_due_to_ties": True,
                "multi_region_tie_sample_count": tied,
                "nearest_raw_surface_distance": _distance_stats_m(minima),
            }
        nearest_union = set.union(*method_region_sets) if method_region_sets else set()
        nearest_recurrent = set.intersection(*method_region_sets) if method_region_sets else set()
        certified_regions = sorted(
            name for name, part_names in certified_region_parts.items() if part["name"] in part_names
        )
        if not certified_regions:
            classification = "no_certified_contact_face"
        elif len(certified_regions) == 1:
            classification = "certified:" + certified_regions[0]
        else:
            classification = "mixed_certified:" + "+".join(certified_regions)
        rows.append(
            {
                "name": part["name"],
                "classification": classification,
                "classification_authority": "registered certified callback-face contracts only",
                "certified_regions": certified_regions,
                "nearest_region_profile": {
                    "recurrent_regions": sorted(nearest_recurrent),
                    "union_regions": sorted(nearest_union),
                    "methods": method_rows,
                    "semantic_classification_authority": False,
                },
                "all_samples_nearest_raw_surface_distance": _distance_stats_m(all_minima),
            }
        )
    return rows


def _region_inventory(raw: dict[str, Any], region_face_ids: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    public: dict[str, Any] = {}
    meshes: dict[str, Any] = {}
    for name, face_ids in region_face_ids.items():
        compact = _compact_faces(raw["vertices_m"], raw["triangles"], face_ids)
        tri_points = compact["vertices"][compact["triangles"]]
        area = 0.5 * np.linalg.norm(
            np.cross(tri_points[:, 1] - tri_points[:, 0], tri_points[:, 2] - tri_points[:, 0]), axis=1
        )
        public[name] = {
            "face_count": len(compact["face_ids"]),
            "unique_vertex_count": len(compact["vertices"]),
            "face_ids_sha256": compact["face_ids_sha256"],
            "digest": compact["digest"],
            "surface_area_m2": float(np.sum(area)),
            "bounds_local_m": [compact["vertices"].min(axis=0).tolist(), compact["vertices"].max(axis=0).tolist()],
        }
        meshes[name] = {
            "vertices": compact["vertices"],
            "triangles": compact["triangles"],
            "mesh": _trimesh(compact["vertices"], compact["triangles"]),
        }
    return public, meshes


def _part_public(part: dict[str, Any], association: dict[str, Any]) -> dict[str, Any]:
    return {
        "body": part["body"],
        "name": part["name"],
        "path": part["path"],
        "global_part_idx": part["global_part_idx"],
        "payload_sha256": part["payload_sha256"],
        "callback_vertices_sha256": part["callback_vertices_sha256"],
        "original_polygon_indices_sha256": part["original_polygon_indices_sha256"],
        "original_polygon_descriptors_sha256": part["original_polygon_descriptors_sha256"],
        "callback_topology_sha256": part["callback_topology_sha256"],
        "witness_path": part["witness_path"],
        "witness_sha256": part["witness_sha256"],
        "vertex_count": part["vertex_count"],
        "polygon_count": part["polygon_count"],
        "max_vertices_per_polygon": part["max_vertices_per_polygon"],
        "topology_triangle_count": part["triangle_count"],
        "triangle_count_is_not_polygon_count": True,
        "property_volume_m3": part["property_volume_m3"],
        "aabb_aspect_ratio_diagnostic": part["aabb_aspect_ratio_diagnostic"],
        "offline_gpu_geometry_limits": part["offline_gpu_geometry_limits"],
        "semantic_allocation_profile": association,
    }


def _negative_controls(
    raw: dict[str, dict[str, Any]],
    parts: dict[str, list[dict[str, Any]]],
    moving_identity: dict[str, Any],
    broad: dict[str, Any],
    fixed_patch: dict[str, Any],
) -> dict[str, Any]:
    del fixed_patch  # the seed-removal perturbation is evaluated against the broader frozen component

    def owner_contract(candidate_parent: str, candidate_child: str) -> bool:
        return candidate_parent == "link5" and candidate_child == "gripper_link"

    def closing_normal_contract(candidate: np.ndarray) -> bool:
        unit = np.asarray(candidate, dtype=np.float64)
        unit = unit / np.linalg.norm(unit)
        return float(np.dot(unit, np.asarray([0.0, -1.0, 0.0]))) >= 1.0 - NORMAL_DOT_TOL

    def callback_topology_contract(selector: str) -> bool:
        for body_parts in parts.values():
            for part in body_parts:
                triangles = part["triangles"] if selector == "callback" else part["_qhull_triangles"]
                if _sha_bytes(_blob(triangles, "<i8")) != part["expected_callback_topology_sha256"]:
                    return False
        return True

    def inventory_contract(candidate: dict[str, list[str]]) -> bool:
        expected = [f"part_{index:03d}" for index in range(64)]
        return set(candidate) == {"link5", "gripper_link"} and all(
            candidate[body] == expected for body in ["link5", "gripper_link"]
        )

    def meter_stream_contract(candidate: np.ndarray) -> bool:
        return _sha_bytes(_blob(candidate, "<f8")) == RAW_STREAM_EXPECTED["link5"]["raw_vertex_stream_sha256"]

    def seed_anchor_contract(mesh: Any, seed_point: np.ndarray) -> tuple[bool, float]:
        distance = float(_nearest(mesh, seed_point.reshape(1, 3))[0][0])
        return distance <= PROXIMITY_TIE_M, distance

    root = ET.parse(URDF_PATH).getroot()
    joint = next(item for item in root.findall("joint") if item.attrib.get("name") == "link5_to_gripper_link")
    parent = joint.find("parent").attrib["link"]
    child = joint.find("child").attrib["link"]
    owner_baseline = owner_contract(parent, child)
    owner_perturbed = owner_contract(child, parent)
    owner_swap = owner_baseline and not owner_perturbed

    inner_expected = np.asarray([0.0, -1.0, 0.0])
    outer_expected = np.asarray([0.0, 1.0, 0.0])
    normal_baseline = closing_normal_contract(inner_expected)
    normal_perturbed = closing_normal_contract(outer_expected)
    outer_as_inner_rejected = normal_baseline and not normal_perturbed

    callback_baseline = callback_topology_contract("callback")
    qhull_perturbed = callback_topology_contract("qhull")
    qhull_differences = sum(
        part["callback_topology_sha256"]
        != _sha_bytes(_blob(part["_qhull_triangles"], "<i8"))
        for body_parts in parts.values()
        for part in body_parts
    )
    qhull_rejected = callback_baseline and not qhull_perturbed and qhull_differences > 0

    baseline_inventory = {body: [part["name"] for part in parts[body]] for body in ["link5", "gripper_link"]}
    fake_inventory = {body: list(names) for body, names in baseline_inventory.items()}
    fake_inventory["link5"].append("legacy_disabled_raw_mesh")
    inventory_baseline = inventory_contract(baseline_inventory)
    inventory_perturbed = inventory_contract(fake_inventory)
    inventory_65_rejected = inventory_baseline and not inventory_perturbed

    original_vertices = raw["link5"]["vertices_m"]
    scaled = original_vertices * 1000.0
    unit_baseline = meter_stream_contract(original_vertices)
    unit_perturbed = meter_stream_contract(scaled)
    scale_original = _sha_bytes(_blob(original_vertices, "<f8"))
    scale_perturbed = _sha_bytes(_blob(scaled, "<f8"))
    scale_rejected = unit_baseline and not unit_perturbed

    ordered_rows = [f"{body}:{part['name']}:{part['payload_sha256']}" for body in ["link5", "gripper_link"] for part in parts[body]]
    reversed_rows = list(reversed(ordered_rows))
    ordered_hash = _sha_bytes("\n".join(ordered_rows).encode())
    reversed_ordered_hash = _sha_bytes("\n".join(reversed_rows).encode())
    canonical_hash = _sha_bytes("\n".join(sorted(ordered_rows)).encode())
    reversed_canonical_hash = _sha_bytes("\n".join(sorted(reversed_rows)).encode())
    order_control = ordered_hash != reversed_ordered_hash and canonical_hash == reversed_canonical_hash

    point_rows = moving_identity["rows"]
    point_baseline = point_rows["original_point_id_ascending"] == EXPECTED_MOVING_BUNDLE
    coordinate_perturbed = point_rows["coordinate_lexicographic_unique"] == EXPECTED_MOVING_BUNDLE
    reverse_perturbed = point_rows["reverse_original_point_id"] == EXPECTED_MOVING_BUNDLE
    point_id_rejected = point_baseline and not coordinate_perturbed and not reverse_perturbed

    seed = np.asarray(_read_json(D350_BINDING)["seed_local_m"], dtype=np.float64)
    full_mesh = _trimesh(raw["link5"]["vertices_m"], raw["link5"]["triangles"])
    seed_baseline, baseline_distance = seed_anchor_contract(full_mesh, seed)
    remaining_faces = np.asarray(
        sorted(set(range(len(raw["link5"]["triangles"]))) - set(map(int, broad["face_ids"]))), dtype=np.int64
    )
    remaining = _compact_faces(raw["link5"]["vertices_m"], raw["link5"]["triangles"], remaining_faces)
    removed_mesh = _trimesh(remaining["vertices"], remaining["triangles"])
    seed_perturbed, removed_distance = seed_anchor_contract(removed_mesh, seed)
    seed_removal = seed_baseline and not seed_perturbed and removed_distance > baseline_distance + PROXIMITY_TIE_M

    checks = {
        "owner_swap_rejected": owner_swap,
        "moving_outer_as_inner_rejected_by_normal_sign": outer_as_inner_rejected,
        "callback_topology_to_qhull_authority_substitution_rejected": qhull_rejected,
        "disabled_legacy_65th_part_rejected": inventory_65_rejected,
        "meter_to_millimeter_x1000_rejected": scale_rejected,
        "part_order_permutation_invariant_only_after_canonical_sort": order_control,
        "coordinate_and_reverse_point_id_remaps_rejected": point_id_rejected,
        "d350_seed_component_removal_worsens_anchor": seed_removal,
    }
    return {
        "rows": {
            "owner_swap": {
                "baseline_pass": owner_baseline,
                "perturbed_pass": owner_perturbed,
                "urdf_parent": parent,
                "urdf_child": child,
                "rejected": owner_swap,
            },
            "outer_as_inner": {
                "baseline_pass": normal_baseline,
                "perturbed_pass": normal_perturbed,
                "outer_dot_inner": float(np.dot(outer_expected, inner_expected)),
                "rejected": outer_as_inner_rejected,
            },
            "qhull_substitution": {
                "baseline_pass": callback_baseline,
                "perturbed_pass": qhull_perturbed,
                "different_part_count": qhull_differences,
                "rejected": qhull_rejected,
            },
            "legacy_65th": {
                "baseline_pass": inventory_baseline,
                "perturbed_pass": inventory_perturbed,
                "perturbed_count": len(fake_inventory["link5"]),
                "rejected": inventory_65_rejected,
            },
            "unit_x1000": {
                "baseline_pass": unit_baseline,
                "perturbed_pass": unit_perturbed,
                "original_sha256": scale_original,
                "perturbed_sha256": scale_perturbed,
                "rejected": scale_rejected,
            },
            "part_order": {
                "ordered_sha256": ordered_hash,
                "reverse_ordered_sha256": reversed_ordered_hash,
                "canonical_sha256": canonical_hash,
                "reverse_canonical_sha256": reversed_canonical_hash,
                "pass": order_control,
            },
            "point_id_remaps": {
                "baseline_pass": point_baseline,
                "coordinate_perturbed_pass": coordinate_perturbed,
                "reverse_perturbed_pass": reverse_perturbed,
                "rejected": point_id_rejected,
            },
            "seed_component_removal": {
                "baseline_pass": seed_baseline,
                "perturbed_pass": seed_perturbed,
                "baseline_distance_mm": baseline_distance * 1000.0,
                "removed_component_distance_mm": removed_distance * 1000.0,
                "rejected": seed_removal,
            },
        },
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "pass": all(checks.values()),
    }


def _seed_to_live(parts: list[dict[str, Any]], seed: np.ndarray) -> dict[str, Any]:
    distances = np.asarray([_nearest(part["_mesh"], seed.reshape(1, 3))[0][0] for part in parts])
    minimum = float(np.min(distances))
    names = [parts[int(i)]["name"] for i in np.flatnonzero(distances <= minimum + PROXIMITY_TIE_M)]
    return {
        "seed_local_m": seed.tolist(),
        "minimum_distance_mm": minimum * 1000.0,
        "nearest_part_names_within_1nm_tie": sorted(names),
    }


def _summary_classifications(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["classification"] for row in rows).items()))


def _mesh_rows_for_rerun(
    raw_meshes: dict[str, dict[str, Any]],
    parts: dict[str, list[dict[str, Any]]],
    associations: dict[str, list[dict[str, Any]]],
    certified: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_colors = {
        "seed_contact_plane_patch": [0, 205, 255, 225],
        "d350_witness_component_remainder": [155, 155, 155, 45],
        "other_raw_connected_components": [65, 125, 220, 45],
        "inner_contact_patch": [0, 205, 255, 225],
        "outer_negative_patch": [190, 80, 230, 170],
        "structural_remainder": [135, 145, 165, 40],
    }
    rows: list[dict[str, Any]] = []
    expected_entities: list[str] = []
    for body, regions in raw_meshes.items():
        for region, data in regions.items():
            path = f"semantic/source/{body}/{region}"
            rows.append(
                {
                    "entity_path": path,
                    "coordinate_frame": "tf#/",
                    "vertices_m": data["vertices"],
                    "triangles": data["triangles"],
                    "color_rgba": raw_colors[region],
                    "static": True,
                    "semantic_region": region,
                    "authority": "raw Float64 source; Rerun copy is display-only Float32",
                }
            )
            expected_entities.extend([path, f"metadata/meshes/{path.replace('/', '__')}"])
    assoc_maps = {body: {row["name"]: row for row in values} for body, values in associations.items()}
    fixed_carriers = set(certified["link5_fixed"]["part_names"])
    inner_carriers = set(certified["gripper_inner"]["part_names"])
    outer_carriers = set(certified["gripper_outer"]["part_names"])
    for body, body_parts in parts.items():
        for part in body_parts:
            association = assoc_maps[body][part["name"]]
            classification = association["classification"]
            if body == "link5" and part["name"] in fixed_carriers:
                color = [35, 235, 80, 150]
                visual_class = "certified_seed_patch_carrier"
            elif body == "gripper_link" and part["name"] in inner_carriers and part["name"] in outer_carriers:
                color = [245, 210, 35, 175]
                visual_class = "dual_inner_outer_patch_carrier"
            elif body == "gripper_link" and part["name"] in inner_carriers:
                color = [35, 235, 80, 150]
                visual_class = "certified_inner_patch_carrier"
            elif body == "gripper_link" and part["name"] in outer_carriers:
                color = [190, 80, 230, 120]
                visual_class = "outer_negative_patch_carrier"
            elif classification.startswith("mixed_certified:"):
                color = [245, 210, 35, 175]
                visual_class = "mixed_certified_patch_carrier"
            elif classification == "no_certified_contact_face":
                color = [65, 130, 220, 55]
                visual_class = "no_certified_contact_face"
            else:
                color = [65, 130, 220, 65]
                visual_class = "noncritical_or_pure_structural_carrier"
            path = f"semantic/collider/{body}/{visual_class}/{part['name']}"
            rows.append(
                {
                    "entity_path": path,
                    "coordinate_frame": "tf#/",
                    "vertices_m": part["vertices"],
                    "triangles": part["triangles"],
                    "color_rgba": color,
                    "static": True,
                    "semantic_classification": classification,
                    "visual_class": visual_class,
                    "payload_sha256": part["payload_sha256"],
                    "authority": "D348 callback topology; Rerun copy is display-only Float32",
                }
            )
            expected_entities.extend([path, f"metadata/meshes/{path.replace('/', '__')}"])
    return rows, expected_entities


def _render_summary_png(
    raw_meshes: dict[str, dict[str, Any]],
    parts: dict[str, list[dict[str, Any]]],
    associations: dict[str, list[dict[str, Any]]],
    certified: dict[str, dict[str, Any]],
    allocation: dict[str, Any],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    if SUMMARY_PNG_PATH.exists():
        raise FileExistsError(SUMMARY_PNG_PATH)
    fig = plt.figure(figsize=(16, 9), dpi=120, constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.975, top=0.90, bottom=0.155, wspace=0.08, hspace=0.16)
    axes = [fig.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    assoc_maps = {body: {row["name"]: row for row in rows} for body, rows in associations.items()}
    fixed = set(certified["link5_fixed"]["part_names"])
    inner = set(certified["gripper_inner"]["part_names"])
    outer = set(certified["gripper_outer"]["part_names"])

    def add_mesh(ax: Any, vertices: np.ndarray, triangles: np.ndarray, color: str, alpha: float, lw: float = 0.05) -> None:
        poly = Poly3DCollection(vertices[triangles] * 1000.0, facecolor=color, edgecolor=color, alpha=alpha, linewidth=lw)
        ax.add_collection3d(poly)

    def draw_body(ax: Any, body: str, zoom: bool) -> None:
        raw = raw_meshes[body]
        if body == "link5":
            if not zoom:
                add_mesh(ax, raw["d350_witness_component_remainder"]["vertices"], raw["d350_witness_component_remainder"]["triangles"], "#9a9a9a", 0.10)
                add_mesh(ax, raw["other_raw_connected_components"]["vertices"], raw["other_raw_connected_components"]["triangles"], "#3f7fdc", 0.08)
            add_mesh(ax, raw["seed_contact_plane_patch"]["vertices"], raw["seed_contact_plane_patch"]["triangles"], "#00cdef", 0.72, 0.12)
        else:
            if not zoom:
                add_mesh(ax, raw["structural_remainder"]["vertices"], raw["structural_remainder"]["triangles"], "#8c96aa", 0.08)
            add_mesh(ax, raw["outer_negative_patch"]["vertices"], raw["outer_negative_patch"]["triangles"], "#bd55df", 0.32)
            add_mesh(ax, raw["inner_contact_patch"]["vertices"], raw["inner_contact_patch"]["triangles"], "#00cdef", 0.72, 0.12)
        for part in parts[body]:
            cls = assoc_maps[body][part["name"]]["classification"]
            if zoom and body == "link5" and part["name"] not in fixed:
                continue
            if zoom and body == "gripper_link" and part["name"] not in (inner | outer):
                continue
            if body == "link5" and part["name"] in fixed:
                color, alpha = "#19db4f", 0.36
            elif body == "gripper_link" and part["name"] in inner and part["name"] in outer:
                color, alpha = "#e9c91f", 0.42
            elif body == "gripper_link" and part["name"] in inner:
                color, alpha = "#19db4f", 0.36
            elif body == "gripper_link" and part["name"] in outer:
                color, alpha = "#bd55df", 0.26
            elif cls.startswith("mixed_certified:"):
                color, alpha = "#e9c91f", 0.42
            else:
                color, alpha = "#397fd1", 0.11 if not zoom else 0.08
            add_mesh(ax, part["vertices"], part["triangles"], color, alpha, 0.06)
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("y (mm)", fontsize=8)
        ax.set_zlabel("z (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=20, azim=-55)
        if body == "link5" and zoom:
            ax.set_xlim(-14, -6); ax.set_ylim(-19, 18); ax.set_zlim(76, 123)
        elif body == "link5":
            ax.set_xlim(-34, 19); ax.set_ylim(-21, 21); ax.set_zlim(-5, 124)
        elif zoom:
            ax.set_xlim(20, 70); ax.set_ylim(-11, 3); ax.set_zlim(-41, 3)
        else:
            ax.set_xlim(-14, 71); ax.set_ylim(-13, 18); ax.set_zlim(-42, 4)
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        ax.set_box_aspect(
            (abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), abs(zlim[1] - zlim[0]))
        )

    draw_body(axes[0], "link5", False)
    axes[0].set_title("link5 - current 64 hull carriers", fontsize=11)
    draw_body(axes[1], "link5", True)
    axes[1].set_title("link5 - D350 seed-plane patch zoom", fontsize=11)
    draw_body(axes[2], "gripper_link", False)
    axes[2].set_title("moving jaw - current 64 hull carriers", fontsize=11)
    draw_body(axes[3], "gripper_link", True)
    axes[3].set_title("moving jaw - frozen inner patch zoom", fontsize=11)
    fig.suptitle("D368 current 64-cap semantic allocation (OFFLINE, no physics)", fontsize=15, fontweight="bold")
    legend = [
        Line2D([0], [0], color="#00cdef", lw=6, label="raw contact patch"),
        Line2D([0], [0], color="#19db4f", lw=6, label="certified contact carrier"),
        Line2D([0], [0], color="#397fd1", lw=6, label="other / structural carrier"),
        Line2D([0], [0], color="#bd55df", lw=6, label="outer negative patch / carrier"),
        Line2D([0], [0], color="#e9c91f", lw=6, label="dual inner+outer carrier"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.018))
    fig.text(
        0.5,
        0.075,
        (
            f"Certified carriers: link5 {len(fixed)}/64 | moving inner {len(inner)}/64 | "
            f"moving outer {len(outer)}/64.  Hull count optimality = NULL.  "
            "Authority: D348 callback topology; colors are display copies."
        ),
        ha="center",
        fontsize=9,
    )
    fig.savefig(SUMMARY_PNG_PATH, dpi=120, facecolor="white")
    plt.close(fig)
    with Image.open(SUMMARY_PNG_PATH) as image:
        dimensions = [int(image.width), int(image.height)]
        mode = image.mode
    return {
        "path": _rel(SUMMARY_PNG_PATH),
        "bytes": SUMMARY_PNG_PATH.stat().st_size,
        "sha256": _sha(SUMMARY_PNG_PATH),
        "dimensions": dimensions,
        "mode": mode,
        "exact_1920x1080": dimensions == [1920, 1080],
    }


def _write_rerun(
    evidence: dict[str, Any],
    raw_meshes: dict[str, dict[str, Any]],
    parts: dict[str, list[dict[str, Any]]],
    associations: dict[str, list[dict[str, Any]]],
    certified: dict[str, dict[str, Any]],
    allocation: dict[str, Any],
    fixed_patch: dict[str, Any],
    raw: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.viz_debug import log_rerun

    mesh_rows, expected_entities = _mesh_rows_for_rerun(raw_meshes, parts, associations, certified)
    d350_seed = np.asarray(_read_json(D350_BINDING)["seed_local_m"], dtype=np.float64)
    inner_vertices = raw_meshes["gripper_link"]["inner_contact_patch"]["vertices"]
    inner_center = np.mean(inner_vertices, axis=0)
    seed_normal = np.asarray(fixed_patch["seed_normal_local"], dtype=np.float64)
    points = [
        {
            "entity_path": "semantic/anchors/link5/d350_seed",
            "positions_m": [d350_seed],
            "radii": [0.0015],
            "colors": [[255, 230, 20, 255]],
            "labels": ["D350 seed"],
            "coordinate_frame": "tf#/",
            "static": True,
        },
        {
            "entity_path": "semantic/anchors/gripper_link/inner_patch_center",
            "positions_m": [inner_center],
            "radii": [0.0015],
            "colors": [[255, 230, 20, 255]],
            "labels": ["moving inner center"],
            "coordinate_frame": "tf#/",
            "static": True,
        },
    ]
    arrows = [
        {
            "entity_path": "semantic/normals/link5/seed_normal",
            "origins_m": [d350_seed],
            "vectors_m": [seed_normal * 0.015],
            "radii": [0.0005],
            "colors": [[255, 230, 20, 255]],
            "labels": ["+localX seed normal"],
            "coordinate_frame": "tf#/",
            "static": True,
        },
        {
            "entity_path": "semantic/normals/gripper_link/inner_normal",
            "origins_m": [inner_center],
            "vectors_m": [[0.0, -0.015, 0.0]],
            "radii": [0.0005],
            "colors": [[255, 230, 20, 255]],
            "labels": ["-localY closing face"],
            "coordinate_frame": "tf#/",
            "static": True,
        },
    ]
    scalar_values = {
        "link5_certified_carriers": len(certified["link5_fixed"]["part_names"]),
        "link5_recurrent_nearest_sample": len(
            allocation["link5_fixed"]["recurrent_nearest_sample_part_names"]
        ),
        "link5_union_nearest_sample": len(allocation["link5_fixed"]["union_nearest_sample_part_names"]),
        "moving_inner_certified_carriers": len(certified["gripper_inner"]["part_names"]),
        "moving_inner_recurrent_nearest_sample": len(
            allocation["gripper_inner"]["recurrent_nearest_sample_part_names"]
        ),
        "moving_inner_union_nearest_sample": len(
            allocation["gripper_inner"]["union_nearest_sample_part_names"]
        ),
        "moving_outer_certified_carriers": len(certified["gripper_outer"]["part_names"]),
        "negative_controls_passed": evidence["negative_controls"]["passed"],
        "isaac_launch_count": 0,
        "physics_step_count": 0,
    }
    scalars = [
        {"entity_path": f"metrics/d368/{name}", "value": value, "static": True}
        for name, value in scalar_values.items()
    ]
    event_text = (
        "CYAN=raw contact patch | GREEN=certified contact carrier | YELLOW=dual inner+outer carrier | "
        "BLUE/GRAY=no certified contact face/noncritical | PURPLE=outer negative. "
        "OFFLINE only: Isaac=0 PhysX=0 q5=0 steps=0. "
        "Measured allocation does NOT mean 64 is optimal."
    )
    events = [{"entity_path": "events/d368_summary", "text": event_text, "level": "INFO", "static": True}]
    result = log_rerun(
        RRD_PATH,
        meshes=mesh_rows,
        points=points,
        arrows=arrows,
        scalar_trace=scalars,
        events=events,
        recording_metadata={
            "case": "g0a_d368",
            "verdict": evidence["verdict"],
            "evidence_path": _rel(EVIDENCE_PATH),
            "evidence_sha256": _sha(EVIDENCE_PATH),
            "scientific_authority": "D348 original callback polygon/topology arrays plus raw Float64 source",
            "display_geometry_role": "Float32 inspection copy only",
            "current_64cap_optimal": None,
            "physics_execution": 0,
        },
        recording_id="g0a_d368_current_64cap_semantic_allocation",
        blueprint_path=RBL_PATH,
        blueprint_mode="d368_semantic_allocation",
        live_viewer=False,
        app_id="roarm_g0a_d368_semantic_allocation",
    )
    if not result.get("ok", False):
        raise RuntimeError(f"Rerun save-only recording failed: {result}")
    expected_entities.extend([row["entity_path"] for row in points])
    expected_entities.extend([row["entity_path"] for row in arrows])
    expected_entities.extend([row["entity_path"] for row in scalars])
    expected_entities.extend(["events/d368_summary", "metadata/run"])
    component_contract: dict[str, list[str]] = {}
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    for row in mesh_rows:
        component_contract[row["entity_path"]] = mesh_components
        component_contract[f"metadata/meshes/{row['entity_path'].replace('/', '__')}"] = ["TextDocument:text"]
    for row in points:
        component_contract[row["entity_path"]] = ["Points3D:positions"]
    for row in arrows:
        component_contract[row["entity_path"]] = ["Arrows3D:vectors"]
    for row in scalars:
        component_contract[row["entity_path"]] = ["Scalars:scalars"]
    component_contract["events/d368_summary"] = ["TextLog:text", "TextLog:level"]
    component_contract["metadata/run"] = ["TextDocument:text"]
    strict = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected_entities),
        expected_timeline_names=["blueprint", "log_time"],
        exact_entity_paths=sorted(expected_entities),
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=component_contract,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG_PATH,
        screenshot_window_size="2400x1400",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=180.0,
    )
    _write_json_x(RERUN_VALIDATION_PATH, strict)
    return {"log_rerun": result, "strict_validation": strict}


def _public_fixed_patch(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def _public_certified(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def _alarm_handler(signum: int, frame: Any) -> None:
    raise TimeoutError(f"D368 audit exceeded {AUDIT_TIMEOUT_SECONDS}s")


def _audit() -> None:
    if not PREREG_PATH.is_file():
        raise RuntimeError("D368 prepare must pass before audit")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True or not all(prereg.get("checks", {}).values()):
        raise RuntimeError("D368 preregistration is not PASS")
    if {path.name for path in OUT_DIR.iterdir()} != {PREREG_PATH.name}:
        raise RuntimeError("forward-only pre-audit inventory mismatch")
    if not (_git("rev-parse", "HEAD") == _git("rev-parse", "origin/master") == prereg["head"] == EXPECTED_HEAD):
        raise RuntimeError("Git base differs from registered D368 base")
    if _dynamic_hashes() != prereg["dynamic_hashes"]:
        raise RuntimeError("D368 harness/session/rules changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("frozen D368 input changed after preregistration")
    if _sidecar_snapshot() != prereg["d334_sidecar_before"]:
        raise RuntimeError("user-owned D334 sidecar changed before audit")
    environment = _environment_preflight()
    if not environment["pass"]:
        raise RuntimeError(f"offline environment preflight failed: {environment['checks']}")
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(AUDIT_TIMEOUT_SECONDS)
    started = time.monotonic()
    _write_json_x(
        INVOCATION_PATH,
        {
            "artifact": "D368_SINGLE_OFFLINE_AUDIT_INVOCATION_V1",
            "audit_invocation_count": 1,
            "pid": os.getpid(),
            "argv": sys.argv,
            "harness_sha256": _sha(HARNESS),
            "no_retry": True,
            "isaac_physx_q5_physics_count": 0,
        },
    )
    PHASE_PATH.touch(exist_ok=False)
    _phase("audit_started")
    measurement_evidence_committed = False
    try:
        raw = _load_raw_meshes()
        raw_checks = {
            body: {
                "vertex_count": data["stream_summary"]["vertex_count"] == RAW_STREAM_EXPECTED[body]["vertex_count"],
                "triangle_count": data["stream_summary"]["triangle_count"] == RAW_STREAM_EXPECTED[body]["triangle_count"],
                "raw_vertex_hash": data["stream_summary"]["raw_vertex_stream_sha256"]
                == RAW_STREAM_EXPECTED[body]["raw_vertex_stream_sha256"],
                "triangle_hash": data["stream_summary"]["triangle_stream_sha256"]
                == RAW_STREAM_EXPECTED[body]["triangle_stream_sha256"],
            }
            for body, data in raw.items()
        }
        raw_checks["gripper_link"].update(
            {
                "authored_points_hash": raw["gripper_link"]["stream_summary"]["authored_points_f32_mm_sha256"]
                == RAW_STREAM_EXPECTED["gripper_link"]["authored_points_f32_mm_sha256"],
                "face_counts_hash": raw["gripper_link"]["stream_summary"]["face_counts_i64_sha256"]
                == RAW_STREAM_EXPECTED["gripper_link"]["face_counts_i64_sha256"],
            }
        )
        _phase("raw_source_streams_loaded", checks_pass=all(all(row.values()) for row in raw_checks.values()))

        d350 = _read_json(D350_BINDING)
        d350_measurement = _read_json(D350_MEASUREMENT)

        broad_forward = _vertex_connected_component(
            raw["link5"]["vertices_m"], raw["link5"]["triangles"], EXPECTED_D350_COMPONENT["seed_face"], reverse=False
        )
        broad_reverse = _vertex_connected_component(
            raw["link5"]["vertices_m"], raw["link5"]["triangles"], EXPECTED_D350_COMPONENT["seed_face"], reverse=True
        )
        broad_checks = {
            "forward_reverse_face_ids_exact": np.array_equal(broad_forward["face_ids"], broad_reverse["face_ids"]),
            "forward_reverse_digest_exact": broad_forward["digest"] == broad_reverse["digest"],
            "face_count_exact": len(broad_forward["face_ids"]) == EXPECTED_D350_COMPONENT["face_count"],
            "unique_vertex_count_exact": len(broad_forward["vertices"]) == EXPECTED_D350_COMPONENT["unique_vertex_count"],
            "component_count_exact": broad_forward["component_count"] == EXPECTED_D350_COMPONENT["component_count"],
            "digest_exact": broad_forward["digest"] == EXPECTED_D350_COMPONENT["digest"],
        }
        broad_public = {
            "face_count": len(broad_forward["face_ids"]),
            "unique_vertex_count": len(broad_forward["vertices"]),
            "component_count": broad_forward["component_count"],
            "digest": broad_forward["digest"],
            "semantics": "D349/D350 witness-containing broad connected shell; not renamed fixed-jaw pad",
            "checks": broad_checks,
            "pass": all(broad_checks.values()),
        }
        fixed_patch = _fixed_support_patch(
            raw["link5"]["vertices_m"],
            raw["link5"]["triangles"],
            broad_forward["face_ids"],
            d350_oriented_normal_local=np.asarray(
                d350_measurement["actual_surface"]["oriented_surface_normal_local"], dtype=np.float64
            ),
            d350_seed_local_m=np.asarray(d350["seed_local_m"], dtype=np.float64),
        )
        moving_identity = _moving_identity(raw["gripper_link"])
        _phase(
            "semantic_source_authorities_bound",
            broad_pass=broad_public["pass"],
            fixed_patch_pass=fixed_patch["pass"],
            moving_identity_pass=moving_identity["pass"],
        )

        current_parts, callback_inventory = _load_current_parts()
        _phase("d348_current_callback_parts_loaded", total_parts=callback_inventory["total_parts"])

        link5_patch_ids = np.asarray(fixed_patch["_face_ids"], dtype=np.int64)
        broad_set = set(map(int, broad_forward["face_ids"]))
        patch_set = set(map(int, link5_patch_ids))
        link5_regions = {
            "seed_contact_plane_patch": link5_patch_ids,
            "d350_witness_component_remainder": np.asarray(sorted(broad_set - patch_set), dtype=np.int64),
            "other_raw_connected_components": np.asarray(
                sorted(set(range(len(raw["link5"]["triangles"]))) - broad_set), dtype=np.int64
            ),
        }
        gripper_regions = {
            "inner_contact_patch": INNER_FACE_IDS,
            "outer_negative_patch": OUTER_FACE_IDS,
            "structural_remainder": np.asarray(
                sorted(
                    set(range(len(raw["gripper_link"]["triangles"])))
                    - set(map(int, INNER_FACE_IDS))
                    - set(map(int, OUTER_FACE_IDS))
                ),
                dtype=np.int64,
            ),
        }
        link5_region_public, link5_region_meshes = _region_inventory(raw["link5"], link5_regions)
        gripper_region_public, gripper_region_meshes = _region_inventory(raw["gripper_link"], gripper_regions)
        raw_region_meshes = {"link5": link5_region_meshes, "gripper_link": gripper_region_meshes}

        fixed_cert = _certified_faces(
            current_parts["link5"],
            axis=0,
            plane_value_m=float(
                raw["link5"]["vertices_m"][
                    raw["link5"]["triangles"][EXPECTED_D350_COMPONENT["seed_face"], 0], 0
                ]
            ),
            expected_normal=np.asarray(fixed_patch["seed_normal_local"], dtype=np.float64),
            raw_patch_mesh=link5_region_meshes["seed_contact_plane_patch"]["mesh"],
            exact_normal=False,
        )
        inner_cert = _certified_faces(
            current_parts["gripper_link"],
            axis=1,
            plane_value_m=float(raw["gripper_link"]["vertices_m"][raw["gripper_link"]["triangles"][672, 0], 1]),
            expected_normal=np.asarray([0.0, -1.0, 0.0]),
            raw_patch_mesh=gripper_region_meshes["inner_contact_patch"]["mesh"],
            exact_normal=True,
        )
        outer_cert = _certified_faces(
            current_parts["gripper_link"],
            axis=1,
            plane_value_m=float(raw["gripper_link"]["vertices_m"][raw["gripper_link"]["triangles"][13205, 0], 1]),
            expected_normal=np.asarray([0.0, 1.0, 0.0]),
            raw_patch_mesh=gripper_region_meshes["outer_negative_patch"]["mesh"],
            exact_normal=True,
        )
        certified_checks = {
            "link5_fixed_zero_or_nonzero_projection_well_formed": (
                (
                    fixed_cert["face_count"] == 0
                    and fixed_cert["part_count"] == 0
                    and fixed_cert["part_names"] == []
                )
                or (
                    fixed_cert["face_count"] > 0
                    and fixed_cert["source_vertex_residual_within_inherited_face_identity_0p5mm"]
                )
            ),
            "moving_inner_face_count_40": inner_cert["face_count"] == EXPECTED_LIVE_INNER["face_count"],
            "moving_inner_part_count_17": inner_cert["part_count"] == EXPECTED_LIVE_INNER["part_count"],
            "moving_inner_parts_exact": inner_cert["part_names"] == EXPECTED_LIVE_INNER["part_names"],
            "moving_inner_key_hash_exact": inner_cert["face_key_sha256"] == EXPECTED_LIVE_INNER["face_key_sha256"],
            "moving_inner_source_projection_pass": inner_cert["source_vertex_residual_within_inherited_face_identity_0p5mm"],
            "moving_outer_zero_or_nonzero_projection_well_formed": (
                (
                    outer_cert["face_count"] == 0
                    and outer_cert["part_count"] == 0
                    and outer_cert["part_names"] == []
                )
                or (
                    outer_cert["face_count"] > 0
                    and outer_cert["source_vertex_residual_within_inherited_face_identity_0p5mm"]
                )
            ),
        }
        certified = {
            "link5_fixed": fixed_cert,
            "gripper_inner": inner_cert,
            "gripper_outer": outer_cert,
        }
        _phase(
            "certified_callback_patch_faces_partitioned",
            link5_fixed_faces=fixed_cert["face_count"],
            moving_inner_faces=inner_cert["face_count"],
            moving_outer_faces=outer_cert["face_count"],
        )

        allocation = {
            "link5_fixed": _support_allocation(
                link5_region_meshes["seed_contact_plane_patch"]["vertices"],
                link5_region_meshes["seed_contact_plane_patch"]["triangles"],
                current_parts["link5"],
                fixed_cert,
            ),
            "gripper_inner": _support_allocation(
                gripper_region_meshes["inner_contact_patch"]["vertices"],
                gripper_region_meshes["inner_contact_patch"]["triangles"],
                current_parts["gripper_link"],
                inner_cert,
            ),
            "gripper_outer": _support_allocation(
                gripper_region_meshes["outer_negative_patch"]["vertices"],
                gripper_region_meshes["outer_negative_patch"]["triangles"],
                current_parts["gripper_link"],
                outer_cert,
            ),
        }
        certified_region_parts = {
            "link5": {"seed_contact_plane_patch": set(fixed_cert["part_names"])},
            "gripper_link": {
                "inner_contact_patch": set(inner_cert["part_names"]),
                "outer_negative_patch": set(outer_cert["part_names"]),
            },
        }
        associations = {
            "link5": _semantic_association(
                current_parts["link5"],
                {name: data["mesh"] for name, data in link5_region_meshes.items()},
                certified_region_parts["link5"],
            ),
            "gripper_link": _semantic_association(
                current_parts["gripper_link"],
                {name: data["mesh"] for name, data in gripper_region_meshes.items()},
                certified_region_parts["gripper_link"],
            ),
        }
        _phase("part_to_region_association_finished")

        d354 = _read_json(D354_BINDING)
        d359 = _read_json(D359_EVIDENCE)
        d354_partition = _find_artifact(d354, "D351_LIVE_INNER_COMPLEMENT_PARTITION_V1")
        lineage_checks = {
            "d350_binding_original_pass": d350.get("pass") is True,
            "d350_measurement_pass": d350_measurement.get("pass") is True,
            "d350_aligned_null_preserved": d350_measurement.get("aligned_pass") is None,
            "d350_seed_face_1984_exact": d350_measurement["actual_surface"].get("seed_face_index")
            == EXPECTED_D350_COMPONENT["seed_face"],
            "d350_component_digest_exact": d350["first_binding"]["component_digest"] == broad_forward["digest"],
            "d354_overall_false_preserved": d354.get("pass") is False,
            "d354_live_partition_subresult_found": d354_partition is not None,
            "d354_live_partition_subresult_pass": bool(d354_partition and d354_partition.get("pass") is True),
            "d354_live_partition_replayed_exact": bool(
                d354_partition
                and d354_partition.get("inner_face_key_sha256") == inner_cert["face_key_sha256"]
                and d354_partition.get("inner_part_names") == inner_cert["part_names"]
            ),
            "d359_provenance_recovered": d359.get("verdict") == "D359_D351_HASH_PROVENANCE_RECOVERED",
        }
        negative = _negative_controls(raw, current_parts, moving_identity, broad_forward, fixed_patch)
        seed_to_live = _seed_to_live(current_parts["link5"], np.asarray(d350["seed_local_m"], dtype=np.float64))

        association_summary = {
            "link5": _summary_classifications(associations["link5"]),
            "gripper_link": _summary_classifications(associations["gripper_link"]),
        }
        callback_public = {
            body: [
                _part_public(part, {row["name"]: row for row in associations[body]}[part["name"]])
                for part in current_parts[body]
            ]
            for body in ["link5", "gripper_link"]
        }
        scope = prereg["scope_guards"]
        checks = {
            "immutable_inputs_exact": _input_hashes() == prereg["input_hashes"],
            "dynamic_preregistered_files_exact": _dynamic_hashes() == prereg["dynamic_hashes"],
            "d334_sidecar_unchanged": _sidecar_snapshot() == prereg["d334_sidecar_before"],
            "environment_preflight_pass": _environment_preflight()["pass"],
            "nvidia_schema_contract_exact": _schema_facts() == prereg["nvidia_official_source_contract"],
            "raw_streams_exact": all(all(row.values()) for row in raw_checks.values()),
            "d350_broad_component_exact": broad_public["pass"],
            "fixed_seed_contact_plane_patch_exact": fixed_patch["pass"],
            "moving_original_point_id_identity_exact": moving_identity["pass"],
            "d348_callback_inventory_exact": callback_inventory["pass"],
            "certified_face_contracts_pass": all(certified_checks.values()),
            "nearest_attribution_serialization_and_finiteness_invariants": all(
                report["set_relationships"]["recurrent_subset_union"]
                and bool(report["union_nearest_sample_part_names"])
                and all(
                    row["nearest_live_surface_distance"].get("finite") is True
                    for row in report["sampling_methods"].values()
                )
                for report in allocation.values()
            ),
            "certified_classification_serialization_invariant": all(
                (
                    (
                        not row["certified_regions"]
                        and row["classification"] == "no_certified_contact_face"
                    )
                    or (
                        len(row["certified_regions"]) == 1
                        and row["classification"] == "certified:" + row["certified_regions"][0]
                    )
                    or (
                        len(row["certified_regions"]) > 1
                        and row["classification"]
                        == "mixed_certified:" + "+".join(row["certified_regions"])
                    )
                )
                for rows in associations.values()
                for row in rows
            ),
            "lineage_checks_pass": all(lineage_checks.values()),
            "negative_controls_8_of_8": negative["pass"] and negative["passed"] == negative["total"] == 8,
            "forbidden_runtime_modules_absent": _forbidden_modules() == [],
            "scope_counts_exact": scope
            == {
                "simulation_app_or_kit": 0,
                "isaac_or_physx": 0,
                "warp_or_cuda_compute": 0,
                "nvidia_smi": 0,
                "cook_or_decomposition": 0,
                "usd_or_asset_writes": 0,
                "q5_science": 0,
                "physics_steps": 0,
                "contact_queries": 0,
                "target_ik_path_changes": 0,
                "material_mass_actuator_physics_changes": 0,
                "d334_sidecar_writes": 0,
                "rerun_display_render_allowed": 1,
            },
        }
        measurement_pass = all(checks.values())
        verdict = VERDICT_MEASURED if measurement_pass else VERDICT_FAIL
        evidence = {
            "artifact": "D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_EVIDENCE_V1",
            "case": "g0a_d368",
            "elapsed_seconds_before_visualization": time.monotonic() - started,
            "new_variables": NEW_VARIABLES,
            "new_physical_variables": [],
            "environment": environment,
            "nvidia_official_source_contract": _schema_facts(),
            "input_hashes": _input_hashes(),
            "raw_source_streams": {body: data["stream_summary"] for body, data in raw.items()},
            "raw_stream_checks": raw_checks,
            "d350_witness_component": broad_public,
            "d350_seed_contact_plane_patch": _public_fixed_patch(fixed_patch),
            "moving_patch_identity": moving_identity,
            "raw_semantic_regions": {
                "link5": link5_region_public,
                "gripper_link": gripper_region_public,
            },
            "callback_inventory": callback_inventory,
            "certified_callback_surfaces": {
                key: _public_certified(value) for key, value in certified.items()
            },
            "certified_checks": certified_checks,
            "patch_allocation": allocation,
            "part_semantic_associations": callback_public,
            "association_classification_counts": association_summary,
            "d350_seed_to_live": seed_to_live,
            "lineage_checks": lineage_checks,
            "negative_controls": negative,
            "checks": checks,
            "measurement_pass": measurement_pass,
            "verdict": verdict,
            "interpretation_boundary": {
                "pass_means": "current 64-cap allocation inventory/provenance measured",
                "current_64cap_optimal": None,
                "physics_equivalence": None,
                "collider_count_tipping_causality": None,
                "actual_gpu_contact_execution": None,
                "contact_or_grasp_recomputed": False,
                "grasp_feasibility": None,
                "g0a_pass": False,
            },
            "scope_guards": scope,
            "visualization_authority": "display-only; written after this authoritative JSON",
        }
        _write_json_x(EVIDENCE_PATH, evidence)
        measurement_evidence_committed = True
        _phase("authoritative_float64_evidence_written", verdict=verdict)

        summary_png = _render_summary_png(
            raw_region_meshes, current_parts, associations, certified, allocation
        )
        if not summary_png["exact_1920x1080"]:
            raise RuntimeError(f"summary PNG dimensions failed: {summary_png}")
        _phase("professor_summary_png_written", dimensions=summary_png["dimensions"])

        rerun = _write_rerun(
            evidence,
            raw_region_meshes,
            current_parts,
            associations,
            certified,
            allocation,
            fixed_patch,
            raw,
        )
        _phase("rerun_rrd_rbl_and_headless_png_validated", pass_value=rerun["strict_validation"]["pass"])
        post_visualization_forbidden_modules = _forbidden_modules()
        _phase(
            "post_visualization_import_boundary_checked",
            forbidden_modules=post_visualization_forbidden_modules,
        )
        automated = {
            "artifact": "D368_AUTOMATED_SUMMARY_V1",
            "case": "g0a_d368",
            "measurement_pass": measurement_pass,
            "measurement_verdict": verdict,
            "evidence_path": _rel(EVIDENCE_PATH),
            "evidence_sha256": _sha(EVIDENCE_PATH),
            "summary_png": summary_png,
            "rerun": {
                "rrd_path": _rel(RRD_PATH),
                "rrd_sha256": _sha(RRD_PATH),
                "rbl_path": _rel(RBL_PATH),
                "rbl_sha256": _sha(RBL_PATH),
                "headless_png_path": _rel(RERUN_PNG_PATH),
                "headless_png_sha256": _sha(RERUN_PNG_PATH),
                "strict_validation_pass": rerun["strict_validation"]["pass"],
            },
            "key_counts": {
                "link5_certified_carriers": len(fixed_cert["part_names"]),
                "link5_recurrent_nearest_sample": len(
                    allocation["link5_fixed"]["recurrent_nearest_sample_part_names"]
                ),
                "link5_union_nearest_sample": len(
                    allocation["link5_fixed"]["union_nearest_sample_part_names"]
                ),
                "moving_inner_certified_carriers": len(inner_cert["part_names"]),
                "moving_inner_recurrent_nearest_sample": len(
                    allocation["gripper_inner"]["recurrent_nearest_sample_part_names"]
                ),
                "moving_inner_union_nearest_sample": len(
                    allocation["gripper_inner"]["union_nearest_sample_part_names"]
                ),
                "moving_outer_certified_carriers": len(outer_cert["part_names"]),
                "negative_controls": f"{negative['passed']}/{negative['total']}",
            },
            "classification_counts": association_summary,
            "post_visualization_forbidden_modules": post_visualization_forbidden_modules,
            "post_visualization_forbidden_modules_absent": post_visualization_forbidden_modules == [],
            "manual_visual_inspection_pending": True,
            "overall_completion_pass": None,
            "current_64cap_optimal": None,
            "g0a_pass": False,
        }
        _write_json_x(AUTOMATED_PATH, automated)
        report_lines = [
            "# D368 automated offline semantic-allocation report",
            "",
            f"- Measurement verdict: `{verdict}`",
            f"- Measurement checks: `{sum(checks.values())}/{len(checks)}`",
            f"- Negative controls: `{negative['passed']}/{negative['total']}`",
            f"- link5 certified/recurrent-nearest/union-nearest: `{len(fixed_cert['part_names'])}/{len(allocation['link5_fixed']['recurrent_nearest_sample_part_names'])}/{len(allocation['link5_fixed']['union_nearest_sample_part_names'])}` of 64",
            f"- moving inner certified/recurrent-nearest/union-nearest: `{len(inner_cert['part_names'])}/{len(allocation['gripper_inner']['recurrent_nearest_sample_part_names'])}/{len(allocation['gripper_inner']['union_nearest_sample_part_names'])}` of 64",
            f"- moving outer certified carriers: `{len(outer_cert['part_names'])}` of 64",
            f"- link5 semantic classifications: `{association_summary['link5']}`",
            f"- gripper semantic classifications: `{association_summary['gripper_link']}`",
            f"- Rerun strict validation: `{rerun['strict_validation']['pass']}`",
            f"- Exact 1920x1080 summary: `{summary_png['exact_1920x1080']}`",
            "- Manual original-resolution inspection is still required before finalize.",
            "- No Isaac/PhysX/Warp/CUDA/q5/physics/contact run occurred.",
            "- Allocation measurement does not prove that 64 is optimal or physically equivalent.",
        ]
        _write_text_x(REPORT_PATH, "\n".join(report_lines))
        expected_after_audit = set(prereg["expected_success_inventory"]) - {
            MANUAL_JSON_PATH.name,
            MANUAL_MD_PATH.name,
            COMPLETION_PATH.name,
        }
        observed_after_audit = {path.name for path in OUT_DIR.iterdir()}
        if observed_after_audit != expected_after_audit:
            raise RuntimeError(
                f"post-audit inventory mismatch: observed={sorted(observed_after_audit)} expected={sorted(expected_after_audit)}"
            )
        _phase("audit_finished_manual_inspection_pending", elapsed_seconds=time.monotonic() - started)
        signal.alarm(0)
        print(
            json.dumps(
                {
                    "stage": "audit",
                    "measurement_pass": measurement_pass,
                    "verdict": verdict,
                    "rerun_validation_pass": rerun["strict_validation"]["pass"],
                    "manual_inspection_pending": True,
                    "output": _rel(OUT_DIR),
                },
                ensure_ascii=False,
            )
        )
    except Exception as error:
        signal.alarm(0)
        measurement_evidence_written = measurement_evidence_committed and EVIDENCE_PATH.is_file()
        measurement_evidence_valid = False
        preserved_measurement_verdict = None
        if measurement_evidence_written:
            try:
                preserved = _read_json(EVIDENCE_PATH)
                preserved_measurement_verdict = preserved.get("verdict")
                measurement_evidence_valid = (
                    preserved.get("artifact")
                    == "D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_EVIDENCE_V1"
                    and preserved_measurement_verdict in {VERDICT_MEASURED, VERDICT_FAIL}
                    and isinstance(preserved.get("measurement_pass"), bool)
                )
            except Exception:
                preserved_measurement_verdict = None
        exception_verdict = (
            VERDICT_OBSERVABILITY_FAIL
            if measurement_evidence_valid and preserved_measurement_verdict == VERDICT_MEASURED
            else VERDICT_FAIL
        )
        if not EXCEPTION_PATH.exists():
            _write_json_x(
                EXCEPTION_PATH,
                {
                    "artifact": "D368_RUNTIME_EXCEPTION_V1",
                    "case": "g0a_d368",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "audit_invocation_count": 1,
                    "automatic_retry": False,
                    "measurement_evidence_written": measurement_evidence_written,
                    "measurement_evidence_valid": measurement_evidence_valid,
                    "preserved_measurement_verdict": preserved_measurement_verdict,
                    "verdict": exception_verdict,
                    "g0a_pass": False,
                },
            )
        try:
            _phase(
                "audit_exception",
                error=f"{type(error).__name__}: {error}",
                exception_verdict=exception_verdict,
                preserved_measurement_verdict=preserved_measurement_verdict,
            )
        except Exception:
            pass
        raise


def _finalize() -> None:
    prereg = _read_json(PREREG_PATH)
    required = [
        PREREG_PATH,
        INVOCATION_PATH,
        PHASE_PATH,
        EVIDENCE_PATH,
        AUTOMATED_PATH,
        REPORT_PATH,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION_PATH,
        RERUN_PNG_PATH,
        SUMMARY_PNG_PATH,
        MANUAL_JSON_PATH,
        MANUAL_MD_PATH,
    ]
    if COMPLETION_PATH.exists():
        raise FileExistsError("refusing to overwrite D368 completion")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if EXCEPTION_PATH.exists():
        raise RuntimeError("D368 audit exception exists; finalize is forbidden")
    if not (_git("rev-parse", "HEAD") == _git("rev-parse", "origin/master") == prereg["head"] == EXPECTED_HEAD):
        raise RuntimeError("Git base drift before finalize")
    if _dynamic_hashes() != prereg["dynamic_hashes"]:
        raise RuntimeError("preregistered code/session/rules changed before finalize")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("frozen input drift before finalize")
    evidence = _read_json(EVIDENCE_PATH)
    automated = _read_json(AUTOMATED_PATH)
    validation = _read_json(RERUN_VALIDATION_PATH)
    manual = _read_json(MANUAL_JSON_PATH)
    raw_manual_files = manual.get("files", [])
    manual_files_are_well_typed = isinstance(raw_manual_files, list) and all(
        isinstance(row, dict) for row in raw_manual_files
    )
    manual_files = raw_manual_files if manual_files_are_well_typed else []
    manual_file_checks = []
    for row in manual_files:
        try:
            path = REPO / row["path"]
            manual_file_checks.append(
                path.is_file()
                and _sha(path) == row["sha256"]
                and path.stat().st_size == row["bytes"]
                and _png_dimensions(path) == row["dimensions"]
            )
        except (KeyError, TypeError, OSError, ValueError):
            manual_file_checks.append(False)
    inventory_before = {path.name for path in OUT_DIR.iterdir()}
    expected_before = set(prereg["expected_success_inventory"]) - {COMPLETION_PATH.name}
    manual_paths = {str(row.get("path")) for row in manual_files}
    expected_manual_paths = {_rel(RERUN_PNG_PATH), _rel(SUMMARY_PNG_PATH)}
    manual_dimensions = {str(row.get("path")): row.get("dimensions") for row in manual_files}
    manual_observations_nonempty = bool(manual.get("observations")) and all(
        bool(row.get("observations")) for row in manual_files
    )
    manual_checks = manual.get("checks", {})
    manual_checks_are_well_typed = isinstance(manual_checks, dict)
    manual_check_keys_exact = manual_checks_are_well_typed and set(manual_checks) == EXPECTED_MANUAL_CHECKS
    manual_markdown_exact = (
        manual.get("markdown_path") == _rel(MANUAL_MD_PATH)
        and manual.get("markdown_sha256") == _sha(MANUAL_MD_PATH)
        and MANUAL_MD_PATH.stat().st_size > 0
        and "PASS" in MANUAL_MD_PATH.read_text(encoding="utf-8")
        and all(path in MANUAL_MD_PATH.read_text(encoding="utf-8") for path in expected_manual_paths)
    )
    current_summary_dimensions = _png_dimensions(SUMMARY_PNG_PATH)
    current_rerun_dimensions = _png_dimensions(RERUN_PNG_PATH)
    evidence_hash_exact = _sha(EVIDENCE_PATH) == automated.get("evidence_sha256")
    rrd_hash_exact = (
        _sha(RRD_PATH) == automated.get("rerun", {}).get("rrd_sha256") == validation.get("sha256")
    )
    rbl_hash_exact = (
        _sha(RBL_PATH)
        == automated.get("rerun", {}).get("rbl_sha256")
        == validation.get("blueprint_verify", {}).get("sha256")
    )
    rerun_png_hash_exact = (
        _sha(RERUN_PNG_PATH)
        == automated.get("rerun", {}).get("headless_png_sha256")
        == validation.get("headless_render", {}).get("sha256")
    )
    summary_png_exact = (
        _sha(SUMMARY_PNG_PATH) == automated.get("summary_png", {}).get("sha256")
        and SUMMARY_PNG_PATH.stat().st_size == automated.get("summary_png", {}).get("bytes")
        and current_summary_dimensions == automated.get("summary_png", {}).get("dimensions") == [1920, 1080]
    )
    checks = {
        "audit_invocation_exactly_one": _read_json(INVOCATION_PATH).get("audit_invocation_count") == 1,
        "measurement_pass": evidence.get("measurement_pass") is True,
        "measurement_verdict_exact": evidence.get("verdict") == VERDICT_MEASURED,
        "negative_controls_8_of_8": evidence["negative_controls"].get("passed")
        == evidence["negative_controls"].get("total")
        == 8,
        "rerun_strict_validation_pass": validation.get("pass") is True,
        "automated_rerun_pass": automated.get("rerun", {}).get("strict_validation_pass") is True,
        "post_visualization_forbidden_modules_absent": automated.get(
            "post_visualization_forbidden_modules_absent"
        )
        is True
        and automated.get("post_visualization_forbidden_modules") == [],
        "evidence_audit_time_hash_exact": evidence_hash_exact,
        "rrd_audit_time_and_validation_hash_exact": rrd_hash_exact,
        "rbl_audit_time_and_validation_hash_exact": rbl_hash_exact,
        "rerun_png_audit_time_and_validation_hash_exact": rerun_png_hash_exact,
        "summary_png_audit_time_hash_bytes_dimensions_exact": summary_png_exact,
        "rerun_png_actual_dimensions_registered": current_rerun_dimensions
        in ([2400, 1400], [4800, 2800]),
        "summary_png_exact_1920x1080": automated.get("summary_png", {}).get("exact_1920x1080") is True,
        "manual_inspection_performed": manual.get("inspection_performed") is True,
        "manual_inspection_pass": manual.get("pass") is True,
        "manual_files_well_typed": manual_files_are_well_typed,
        "manual_checks_well_typed": manual_checks_are_well_typed,
        "manual_check_keys_exact": manual_check_keys_exact,
        "manual_checks_all_pass": manual_check_keys_exact
        and all(value is True for value in manual_checks.values()),
        "manual_exact_two_png_paths": manual_paths == expected_manual_paths and len(manual_files) == 2,
        "manual_png_dimensions_exact": manual_dimensions.get(_rel(SUMMARY_PNG_PATH)) == [1920, 1080]
        and manual_dimensions.get(_rel(RERUN_PNG_PATH)) in ([2400, 1400], [4800, 2800]),
        "manual_observations_nonempty": manual_observations_nonempty,
        "manual_markdown_exact": manual_markdown_exact,
        "manual_file_hashes_exact": len(manual_file_checks) == 2 and all(manual_file_checks),
        "d334_sidecar_unchanged": _sidecar_snapshot() == prereg["d334_sidecar_before"],
        "forbidden_runtime_modules_absent": _forbidden_modules() == [],
        "precompletion_inventory_exact": inventory_before == expected_before,
    }
    overall_pass = all(checks.values())
    visualization_integrity_keys = {
        "rerun_strict_validation_pass",
        "automated_rerun_pass",
        "post_visualization_forbidden_modules_absent",
        "rrd_audit_time_and_validation_hash_exact",
        "rbl_audit_time_and_validation_hash_exact",
        "rerun_png_audit_time_and_validation_hash_exact",
        "summary_png_audit_time_hash_bytes_dimensions_exact",
        "rerun_png_actual_dimensions_registered",
        "summary_png_exact_1920x1080",
        "manual_inspection_performed",
        "manual_inspection_pass",
        "manual_files_well_typed",
        "manual_checks_well_typed",
        "manual_check_keys_exact",
        "manual_checks_all_pass",
        "manual_exact_two_png_paths",
        "manual_png_dimensions_exact",
        "manual_observations_nonempty",
        "manual_markdown_exact",
        "manual_file_hashes_exact",
    }
    visualization_pass = all(checks[key] for key in visualization_integrity_keys)
    measurement_verdict = evidence["verdict"]
    if measurement_verdict != VERDICT_MEASURED:
        completion_verdict = measurement_verdict
    else:
        completion_verdict = VERDICT_MEASURED if overall_pass else VERDICT_OBSERVABILITY_FAIL
    _phase("finalize_started", checks_pass=overall_pass)
    _phase("completion_ready", measurement_verdict=measurement_verdict, completion_verdict=completion_verdict)
    completion = {
        "artifact": "D368_COMPLETION_SUMMARY_V1",
        "case": "g0a_d368",
        "audit_invocation_count": 1,
        "automatic_retry_count": 0,
        "measurement_pass": evidence["measurement_pass"],
        "visualization_pass": visualization_pass,
        "visualization_integrity_check_names": sorted(visualization_integrity_keys),
        "checks": checks,
        "pass": overall_pass,
        "verdict": measurement_verdict,
        "measurement_verdict": measurement_verdict,
        "completion_verdict": completion_verdict,
        "visualization_failure_does_not_override_measurement_verdict": True,
        "artifacts": {
            _rel(path): {"bytes": path.stat().st_size, "sha256": _sha(path)} for path in required
        },
        "scope_guards": evidence["scope_guards"],
        "interpretation_boundary": evidence["interpretation_boundary"],
        "current_64cap_optimal": None,
        "physics_equivalence": None,
        "collider_count_tipping_causality": None,
        "actual_gpu_contact_execution": None,
        "grasp_feasibility": None,
        "g0a_pass": False,
        "next_runtime_or_candidate_generation_requires_new_approval": True,
    }
    _write_json_x(COMPLETION_PATH, completion)
    print(
        json.dumps(
            {
                "stage": "finalize",
                "pass": overall_pass,
                "measurement_verdict": measurement_verdict,
                "completion_verdict": completion_verdict,
            },
            ensure_ascii=False,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "audit", "finalize"), required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        _prepare()
    elif args.stage == "audit":
        _audit()
    else:
        _finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
