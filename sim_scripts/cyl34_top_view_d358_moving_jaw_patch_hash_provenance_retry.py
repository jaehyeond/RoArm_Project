#!/usr/bin/env python3
"""D358 standalone-core-PXR retry of the frozen moving-jaw hash provenance audit.

This process is CPU-only and file/hash-only.  It never creates SimulationApp,
Kit, PhysX, a GPU context, a q5 evaluator, or a new cap/rim classification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d358"
PREREG_PATH = OUT_DIR / "d358_preregistration.json"
INVOCATION_PATH = OUT_DIR / "d358_audit_invocation.json"
PHASE_PATH = OUT_DIR / "d358_phase_markers.jsonl"
EVIDENCE_PATH = OUT_DIR / "d358_patch_hash_provenance_evidence.json"
REPORT_PATH = OUT_DIR / "d358_automated_report.md"
COMPLETION_PATH = OUT_DIR / "d358_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d358_runtime_exception.json"

HARNESS = Path(__file__).resolve()
SESSION_DOC = REPO / "claudedocs/session_20260716_grasp_g0a_d358_moving_jaw_patch_hash_provenance_retry.md"
D355_HARNESS = REPO / "sim_scripts/cyl34_top_view_d355_moving_jaw_patch_hash_provenance_audit.py"
AUTHORING_USD = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"
D339_MANIFEST = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2/d339_asset_build_manifest.json"
D354_BINDING = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354/d354_moving_jaw_surface_binding.json"
D354_MEASUREMENT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354/d354_zero_step_closure_geometry_measurement.json"
D354_COMPLETION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354/d354_completion_summary.json"
D354_ATTESTATION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354/d354_zero_step_science_attestation.json"
D343_PREREG = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d343/d343_preregistration.json"
D345_PREREG = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d345/d345_preregistration.json"
D345_WORKER = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d345/d345_worker_a.json"
D357_COMPLETION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d357/d357_completion_summary.json"
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"

EXPECTED_HEAD = "161f6d9d185bb41eb29259349ee0fd897a3c6de8"
EXPECTED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
PXR_ROOT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
EXPECTED_PYTHONPATH = str(PXR_ROOT)
EXPECTED_LD_LIBRARY_PATH = (
    "/home/cgxr/miniconda3/envs/isaaclab/lib:" + str(PXR_ROOT / "bin")
)
EXPECTED_USD_VERSION = [0, 24, 5]
AUDIT_TIMEOUT_SECONDS = 300
PXR_SDF_BINARY = PXR_ROOT / "pxr/Sdf/_sdf.so"
PXR_USD_BINARY = PXR_ROOT / "pxr/Usd/_usd.so"
EXPECTED_PXR_BINARY_HASHES = {
    PXR_SDF_BINARY: "b4e3056cf5622e0f3036a74876b180c019e46beddc10beb821987020c0c7bbbc",
    PXR_USD_BINARY: "0071f15c896e2252f647384d2276c9b7c9211c0c08f6553701a2432451d2d3c4",
}

EXPECTED_INPUT_HASHES = {
    AUTHORING_USD: "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
    D339_MANIFEST: "3b46cb39a1f0ff655dcd46172ebaa84f727d833773275b18f944397007ae2589",
    D354_BINDING: "548d45ec4eb1dacbb4cbdefe2b64a3ed99ce72f4f5ffaaa6a9ee1e2b38756b15",
    D354_MEASUREMENT: "fd0d43c2a47abefad939fc0e980456cc396bbf3ba3bb104b28bcd999100f23ed",
    D354_COMPLETION: "5cc70c8aa1e50532fa4ec27756496d6b9f9447156c56ef700084b44c16226f86",
    D354_ATTESTATION: "1975df11b13a774b89f953991d7fdac6e38d795e81a9535850b147d823740d20",
    D355_HARNESS: "b1fe5bf0f42c3d30a2b56d6809e17cfe4785eb7dcb610e2cf6fc05fb57c50d46",
    D343_PREREG: "fb8f9c292042001aeb05d9b693d910797bd4a214d9e01427ccd54b7e2c387ce8",
    D345_PREREG: "9c31b8070d2051c00ebd6789facd6c8a59256cb9beefe8645a63ff41a277b6a3",
    D345_WORKER: "99991b382bf881502dc73009877cd09a5617be8d3a5a5610a0d047f741756974",
    D357_COMPLETION: "89a20139c12d6936ae052d0069829f0381e6935ba5dcb1b3dcbf581fc3581e71",
}

NEW_VARIABLES = [
    "bundled_standalone_core_pxr_execution_contract",
    "derived_moving_jaw_patch_hash_provenance_semantics",
]
FORBIDDEN_MODULE_PREFIXES = ("isaacsim", "omni", "physx", "carb", "warp", "torch")

VERDICT_COHERENT = "D358_HASH_PROVENANCE_LOCALIZED_COHERENT_RECIPE"
VERDICT_COMPOSITE = "D358_HASH_PROVENANCE_LOCALIZED_INCOHERENT_FROZEN_BUNDLE"
VERDICT_UNRESOLVED = "D358_HASH_PROVENANCE_UNRESOLVED_FAIL_STOP"
VERDICT_INPUT_STOP = "D358_OFFLINE_INPUT_OR_RUNTIME_FAIL_STOP"


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
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
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _input_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in EXPECTED_INPUT_HASHES}


def _expected_input_hashes() -> dict[str, str]:
    return {_rel(path): expected for path, expected in EXPECTED_INPUT_HASHES.items()}


def _sidecar_snapshot() -> dict[str, Any]:
    rows = []
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


def _module_under_pxr_root(module: Any) -> bool:
    filename = getattr(module, "__file__", None)
    return bool(filename) and Path(filename).resolve().is_relative_to(PXR_ROOT.resolve())


def _environment_preflight() -> dict[str, Any]:
    before = _forbidden_modules()
    import pxr
    from pxr import Gf, Usd, UsdGeom

    after = _forbidden_modules()
    pxr_paths = [str(Path(value).resolve()) for value in pxr.__path__]
    smoke_stage = Usd.Stage.CreateInMemory()
    report = {
        "python_executable": sys.executable,
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "pxr_package_paths": pxr_paths,
        "module_origins": {
            "Gf": str(Path(Gf.__file__).resolve()),
            "Usd": str(Path(Usd.__file__).resolve()),
            "UsdGeom": str(Path(UsdGeom.__file__).resolve()),
        },
        "usd_version": list(Usd.GetVersion()),
        "numpy_version": np.__version__,
        "psutil_version": importlib.metadata.version("psutil"),
        "forbidden_modules_before": before,
        "forbidden_modules_after": after,
        "pxr_binary_hashes": {str(path): _sha(path) for path in EXPECTED_PXR_BINARY_HASHES},
        "smoke_stage_valid": smoke_stage is not None and smoke_stage.GetPseudoRoot().IsValid(),
    }
    checks = {
        "python_exact_or_same_resolved_binary": report["python_executable"] == EXPECTED_PYTHON
        or Path(report["python_executable_resolved"]) == Path(EXPECTED_PYTHON).resolve(),
        "pythonpath_exact": report["pythonpath"] == EXPECTED_PYTHONPATH,
        "ld_library_path_exact": report["ld_library_path"] == EXPECTED_LD_LIBRARY_PATH,
        "pxr_root_exists": PXR_ROOT.is_dir(),
        "pxr_package_from_registered_root": pxr_paths == [str((PXR_ROOT / "pxr").resolve())],
        "gf_from_registered_root": _module_under_pxr_root(Gf),
        "usd_from_registered_root": _module_under_pxr_root(Usd),
        "usdgeom_from_registered_root": _module_under_pxr_root(UsdGeom),
        "openusd_0_24_5": report["usd_version"] == EXPECTED_USD_VERSION,
        "numpy_1_26_0": report["numpy_version"] == "1.26.0",
        "psutil_5_9_8": report["psutil_version"] == "5.9.8",
        "pxr_binary_hashes_exact": report["pxr_binary_hashes"]
        == {str(path): expected for path, expected in EXPECTED_PXR_BINARY_HASHES.items()},
        "in_memory_smoke_stage_valid": report["smoke_stage_valid"],
        "no_isaac_kit_gpu_modules": before == after == [],
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report


def _load_d355_helpers() -> ModuleType:
    spec = importlib.util.spec_from_file_location("d358_frozen_d355_helpers", D355_HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen D355 helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_count(axes: dict[str, list[str]]) -> int:
    total = 1
    for values in axes.values():
        total *= len(values)
    return total


def _prepare() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"refusing nonempty forward-only output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SESSION_DOC.is_file():
        raise FileNotFoundError(SESSION_DOC)
    environment = _environment_preflight()
    helpers = _load_d355_helpers()
    axes = helpers._candidate_axes()
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/master")
    input_hashes = _input_hashes()
    expected_hashes = _expected_input_hashes()
    checks = {
        "head_expected": head == EXPECTED_HEAD,
        "head_equals_origin_master": head == origin,
        "frozen_input_hashes_exact": input_hashes == expected_hashes,
        "environment_preflight_pass": environment["pass"],
        "candidate_grid_20736": _candidate_count(axes) == 20736,
        "new_variable_count_two": len(NEW_VARIABLES) == 2,
        "d358_output_initially_empty": not any(OUT_DIR.iterdir()),
    }
    prereg = {
        "artifact": "D358_PREREGISTRATION_V1",
        "case": "g0a_d358",
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "head": head,
        "origin_master": origin,
        "git_status_before_prepare": _git("status", "--short").splitlines(),
        "harness": {"path": _rel(HARNESS), "sha256": _sha(HARNESS)},
        "session_doc": {"path": _rel(SESSION_DOC), "sha256": _sha(SESSION_DOC)},
        "frozen_helper": {"path": _rel(D355_HARNESS), "sha256": _sha(D355_HARNESS)},
        "input_hashes": input_hashes,
        "expected_input_hashes": expected_hashes,
        "d334_sidecar_before": _sidecar_snapshot(),
        "standalone_core_pxr_preflight": environment,
        "registered_command": {
            "python": EXPECTED_PYTHON,
            "python_flags": ["-B"],
            "script": _rel(HARNESS),
            "argv": ["--stage", "audit"],
            "environment": {
                "PYTHONPATH": EXPECTED_PYTHONPATH,
                "LD_LIBRARY_PATH": EXPECTED_LD_LIBRARY_PATH,
            },
        },
        "recipe_axes": axes,
        "candidate_count": _candidate_count(axes),
        "independent_recalculation": (
            "NumPy canonicalization versus Python tuple/dict/struct.pack for authored and raw "
            "inner+outer vertex, triangle, patch, paired-XZ, plus the full raw stream"
        ),
        "negative_controls": [
            "wrong_unit_m_instead_of_mm",
            "wrong_dtype_f8_instead_of_f4",
            "big_endian",
            "fortran_order_without_canonical_contiguity",
            "reverse_face_order",
            "flip_winding",
            "digest_order_FTV",
            "each_target_matching_recipe_big_endian_rejected",
            "each_target_matching_recipe_fortran_layout_rejected",
        ],
        "decision_rule": {
            "coherent": "all frozen/current streams and independent controls pass and one registered recipe reproduces all eight frozen expected fields simultaneously",
            "composite": "all frozen/current streams and independent controls pass and every expected field is reproducible, but no one registered recipe reproduces the eight-field bundle",
            "unresolved": "a frozen/current stream, independent implementation, negative control, or one or more expected fields remain unreproduced",
            "science_boundary": "all outcomes preserve D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP and cannot change cap/rim, target/IK/path, or a gate",
        },
        "rerun_omission": (
            "D358 is a pure file/hash/schema provenance audit with no spatial or temporal judgment; "
            "canonical JSON and original byte hashes are sufficient and Rerun is intentionally omitted"
        ),
        "scope_guards": {
            "simulation_app_or_kit": 0,
            "isaac_gui": 0,
            "gpu_or_nvidia_smi": 0,
            "q5_science": 0,
            "physics_steps": 0,
            "distance_contact_queries": 0,
            "new_cap_rim_classifications": 0,
            "asset_writes": 0,
            "target_ik_path_changes": 0,
            "dependency_changes": 0,
            "d334_sidecar_writes": 0,
        },
        "single_run_contract": {
            "prepare_is_not_audit": True,
            "audit_invocation_count": 1,
            "timeout_seconds": AUDIT_TIMEOUT_SECONDS,
            "no_retry_or_overwrite": True,
        },
        "expected_success_inventory": sorted(
            path.name
            for path in [
                PREREG_PATH,
                INVOCATION_PATH,
                PHASE_PATH,
                EVIDENCE_PATH,
                REPORT_PATH,
                COMPLETION_PATH,
            ]
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    if not prereg["pass"]:
        raise RuntimeError(f"D358 prepare failed: {checks}")
    print(json.dumps({"stage": "prepare", "pass": True, "path": _rel(PREREG_PATH)}, ensure_ascii=False))


def _typed_layout(array: np.ndarray, unit: str) -> dict[str, Any]:
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "strides": list(array.strides),
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
        "byteorder": array.dtype.byteorder,
        "unit": unit,
    }


def _independent_patch_checks(
    helpers: ModuleType,
    points: np.ndarray,
    raw: np.ndarray,
    faces: np.ndarray,
    authored_inner: dict[str, Any],
    authored_outer: dict[str, Any],
    raw_inner: dict[str, Any],
    raw_outer: dict[str, Any],
    raw_full_hash: str,
) -> dict[str, bool]:
    py_authored_inner = helpers._pure_python_patch(points, faces, helpers.INNER_FACE_IDS, "f")
    py_authored_outer = helpers._pure_python_patch(points, faces, helpers.OUTER_FACE_IDS, "f")
    py_raw_inner = helpers._pure_python_patch(raw, faces, helpers.INNER_FACE_IDS, "d")
    py_raw_outer = helpers._pure_python_patch(raw, faces, helpers.OUTER_FACE_IDS, "d")
    checks: dict[str, bool] = {}
    for prefix, independent, vectorized in [
        ("authored_inner", py_authored_inner, authored_inner),
        ("authored_outer", py_authored_outer, authored_outer),
        ("raw_inner", py_raw_inner, raw_inner),
        ("raw_outer", py_raw_outer, raw_outer),
    ]:
        for field in ("vertex_sha256", "triangle_sha256", "patch_sha256", "paired_xz_sha256"):
            checks[f"{prefix}_{field}"] = independent[field] == vectorized[field]
    checks["raw_full_stream_sha256"] = helpers._pure_python_full_f64(raw) == raw_full_hash
    return checks


def _first_recipe_by_field(search: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for field, rows in search["matches_expected_by_field_first_50"].items():
        result[field] = rows[0] if rows else None
    return result


def _parse_recipe_id(recipe_id: str) -> tuple[str, dict[str, str]]:
    parts = recipe_id.split("|")
    source_name = parts[0]
    recipe = {}
    for item in parts[1:]:
        key, value = item.split("=", 1)
        recipe[key] = value
    return source_name, recipe


def _python_recipe_patch(
    points: np.ndarray,
    source_faces: np.ndarray,
    face_ids: np.ndarray,
    recipe: dict[str, str],
) -> dict[str, str]:
    """Independent tuple/dict/struct implementation of every registered recipe axis."""
    ids = sorted(int(value) for value in face_ids)
    if recipe["face_order"] == "descending":
        ids.reverse()
    flat: list[tuple[float, float, float]] = []
    triangle_vertices: list[list[tuple[float, float, float]]] = []
    for face_id in ids:
        vertex_ids = [int(value) for value in source_faces[face_id]]
        if recipe["face_winding"] == "flip_1_2":
            vertex_ids = [vertex_ids[0], vertex_ids[2], vertex_ids[1]]
        row = []
        for vertex_id in vertex_ids:
            vertex = tuple(float(value) for value in points[vertex_id])
            if recipe["signed_zero"] == "normalize_positive":
                vertex = tuple(0.0 if value == 0.0 else value for value in vertex)
            row.append(vertex)
            flat.append(vertex)
        triangle_vertices.append(row)
    if recipe["vertex_order"] == "lexicographic_unique":
        vertices = sorted(set(flat))
    else:
        vertices = []
        seen: dict[tuple[float, float, float], int] = {}
        for vertex in flat:
            if vertex not in seen:
                seen[vertex] = len(vertices)
                vertices.append(vertex)
    remap = {vertex: index for index, vertex in enumerate(vertices)}
    triangles = [[remap[vertex] for vertex in row] for row in triangle_vertices]
    if recipe["triangle_mode"] == "cyclic_min":
        triangles = [
            list(min(tuple(row[offset:] + row[:offset]) for offset in range(3)))
            for row in triangles
        ]
    elif recipe["triangle_mode"] == "unoriented_index_sort":
        triangles = [sorted(row) for row in triangles]
    if recipe["triangle_row_order"] == "lexicographic_rows":
        triangles = sorted(triangles)

    float_code = "f" if recipe["vertex_serialization"] == "<f4" else "d"
    triangle_code = "i" if recipe["triangle_serialization"] == "<i4" else "q"
    face_blob = b"".join(struct.pack("<q", value) for value in ids)
    vertex_blob = b"".join(
        struct.pack("<" + float_code * 3, *vertex) for vertex in vertices
    )
    triangle_blob = b"".join(
        struct.pack("<" + triangle_code * 3, *triangle) for triangle in triangles
    )
    blobs = {"F": face_blob, "V": vertex_blob, "T": triangle_blob}
    patch_blob = b"".join(blobs[key] for key in recipe["digest_blob_order"])
    xz = sorted(set((vertex[0], vertex[2]) for vertex in vertices))
    xz_blob = b"".join(struct.pack("<" + float_code * 2, *row) for row in xz)
    return {
        "vertex_sha256": _sha_bytes(vertex_blob),
        "triangle_sha256": _sha_bytes(triangle_blob),
        "patch_sha256": _sha_bytes(patch_blob),
        "paired_xz_sha256": _sha_bytes(xz_blob),
    }


def _discovered_recipe_independent_replay(
    helpers: ModuleType,
    sources: dict[str, np.ndarray],
    faces: np.ndarray,
    search: dict[str, Any],
) -> dict[str, Any]:
    field_to_hash_key = {
        "vertex": "vertex_sha256",
        "triangle": "triangle_sha256",
        "patch": "patch_sha256",
        "paired_xz": "paired_xz_sha256",
    }
    rows = {}
    for field, matches in search["matches_expected_by_field_first_50"].items():
        if not matches:
            rows[field] = {"pass": False, "reason": "no registered recipe matched"}
            continue
        recipe_id = matches[0]["recipe_id"]
        source_name, recipe = _parse_recipe_id(recipe_id)
        face_ids = helpers.INNER_FACE_IDS if field.startswith("inner_") else helpers.OUTER_FACE_IDS
        independent = _python_recipe_patch(sources[source_name], faces, face_ids, recipe)
        suffix = field.removeprefix("inner_").removeprefix("outer_")
        hash_key = field_to_hash_key[suffix]
        expected = helpers.FROZEN_EXPECTED_PATCH_HASHES[field]
        rows[field] = {
            "recipe_id": recipe_id,
            "hash_key": hash_key,
            "independent_sha256": independent[hash_key],
            "expected_sha256": expected,
            "pass": independent[hash_key] == expected,
        }
    coherent_rows = search["exact_frozen_expected_bundle_matches_first_50"]
    coherent_replay: dict[str, Any]
    if coherent_rows:
        recipe_id = coherent_rows[0]["recipe_id"]
        source_name, recipe = _parse_recipe_id(recipe_id)
        inner = _python_recipe_patch(
            sources[source_name], faces, helpers.INNER_FACE_IDS, recipe
        )
        outer = _python_recipe_patch(
            sources[source_name], faces, helpers.OUTER_FACE_IDS, recipe
        )
        bundle = {
            "inner_vertex": inner["vertex_sha256"],
            "outer_vertex": outer["vertex_sha256"],
            "inner_triangle": inner["triangle_sha256"],
            "outer_triangle": outer["triangle_sha256"],
            "inner_patch": inner["patch_sha256"],
            "outer_patch": outer["patch_sha256"],
            "inner_paired_xz": inner["paired_xz_sha256"],
            "outer_paired_xz": outer["paired_xz_sha256"],
        }
        coherent_replay = {
            "required": True,
            "recipe_id": recipe_id,
            "independent_bundle": bundle,
            "expected_bundle": helpers.FROZEN_EXPECTED_PATCH_HASHES,
            "pass": bundle == helpers.FROZEN_EXPECTED_PATCH_HASHES,
        }
    else:
        coherent_replay = {
            "required": False,
            "recipe_id": None,
            "pass": True,
            "reason": "vectorized search found no coherent eight-field recipe to replay",
        }
    field_pass = len(rows) == 8 and all(row["pass"] for row in rows.values())
    return {
        "rows": rows,
        "field_recipes_pass": field_pass,
        "coherent_bundle_replay": coherent_replay,
        "pass": field_pass and coherent_replay["pass"],
    }


def _alternate_serialization_hashes(
    patch: dict[str, Any], recipe: dict[str, str]
) -> dict[str, dict[str, str]]:
    vertex_dtype = recipe["vertex_serialization"]
    triangle_dtype = recipe["triangle_serialization"]
    big_vertex_dtype = ">f4" if vertex_dtype == "<f4" else ">f8"
    big_triangle_dtype = ">i4" if triangle_dtype == "<i4" else ">i8"
    face_le = np.ascontiguousarray(patch["face_ids"], dtype="<i8").tobytes(order="C")
    vertex_le = np.ascontiguousarray(patch["vertices"], dtype=vertex_dtype).tobytes(order="C")
    triangle_le = np.ascontiguousarray(patch["triangles"], dtype=triangle_dtype).tobytes(order="C")
    xz_le = np.ascontiguousarray(patch["xz"], dtype=vertex_dtype).tobytes(order="C")
    big = {
        "F": np.ascontiguousarray(patch["face_ids"], dtype=">i8").tobytes(order="C"),
        "V": np.ascontiguousarray(patch["vertices"], dtype=big_vertex_dtype).tobytes(order="C"),
        "T": np.ascontiguousarray(patch["triangles"], dtype=big_triangle_dtype).tobytes(order="C"),
    }
    fortran = {
        "F": face_le,
        "V": np.asarray(patch["vertices"], dtype=vertex_dtype).tobytes(order="F"),
        "T": np.asarray(patch["triangles"], dtype=triangle_dtype).tobytes(order="F"),
    }
    return {
        "little_endian_c": {
            "vertex_sha256": _sha_bytes(vertex_le),
            "triangle_sha256": _sha_bytes(triangle_le),
            "patch_sha256": _sha_bytes(b"".join({"F": face_le, "V": vertex_le, "T": triangle_le}[key] for key in recipe["digest_blob_order"])),
            "paired_xz_sha256": _sha_bytes(xz_le),
        },
        "big_endian_c": {
            "vertex_sha256": _sha_bytes(big["V"]),
            "triangle_sha256": _sha_bytes(big["T"]),
            "patch_sha256": _sha_bytes(b"".join(big[key] for key in recipe["digest_blob_order"])),
            "paired_xz_sha256": _sha_bytes(np.ascontiguousarray(patch["xz"], dtype=big_vertex_dtype).tobytes(order="C")),
        },
        "little_endian_fortran": {
            "vertex_sha256": _sha_bytes(fortran["V"]),
            "triangle_sha256": _sha_bytes(fortran["T"]),
            "patch_sha256": _sha_bytes(b"".join(fortran[key] for key in recipe["digest_blob_order"])),
            "paired_xz_sha256": _sha_bytes(np.asarray(patch["xz"], dtype=vertex_dtype).tobytes(order="F")),
        },
    }


def _matched_recipe_byte_layout_controls(
    helpers: ModuleType,
    sources: dict[str, np.ndarray],
    faces: np.ndarray,
    search: dict[str, Any],
) -> dict[str, Any]:
    field_to_hash_key = {
        "vertex": "vertex_sha256",
        "triangle": "triangle_sha256",
        "patch": "patch_sha256",
        "paired_xz": "paired_xz_sha256",
    }
    rows = {}
    for field, matches in search["matches_expected_by_field_first_50"].items():
        if not matches:
            rows[field] = {"pass": False, "reason": "no registered recipe matched"}
            continue
        recipe_id = matches[0]["recipe_id"]
        source_name, recipe = _parse_recipe_id(recipe_id)
        face_ids = helpers.INNER_FACE_IDS if field.startswith("inner_") else helpers.OUTER_FACE_IDS
        patch = helpers._canonical_patch(sources[source_name], faces, face_ids, recipe)
        alternatives = _alternate_serialization_hashes(patch, recipe)
        suffix = field.removeprefix("inner_").removeprefix("outer_")
        hash_key = field_to_hash_key[suffix]
        expected = helpers.FROZEN_EXPECTED_PATCH_HASHES[field]
        checks = {
            "little_endian_c_matches_target": alternatives["little_endian_c"][hash_key] == expected,
            "big_endian_c_rejected": alternatives["big_endian_c"][hash_key] != expected,
            "little_endian_fortran_rejected": alternatives["little_endian_fortran"][hash_key] != expected,
        }
        rows[field] = {
            "recipe_id": recipe_id,
            "hash_key": hash_key,
            "expected_sha256": expected,
            "alternative_hashes": {name: values[hash_key] for name, values in alternatives.items()},
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {"rows": rows, "pass": len(rows) == 8 and all(row["pass"] for row in rows.values())}


def _alarm_handler(signum: int, frame: Any) -> None:
    raise TimeoutError(f"D358 audit exceeded {AUDIT_TIMEOUT_SECONDS}s")


def _audit() -> None:
    if not PREREG_PATH.is_file():
        raise RuntimeError("D358 prepare must complete first")
    prereg = _json(PREREG_PATH)
    if prereg.get("pass") is not True or not all(prereg.get("checks", {}).values()):
        raise RuntimeError("D358 preregistration did not pass")
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(AUDIT_TIMEOUT_SECONDS)
    started = time.monotonic()
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/master")
    if not (head == origin == prereg.get("head") == prereg.get("origin_master") == EXPECTED_HEAD):
        raise RuntimeError(
            f"Git base drift before audit: head={head} origin={origin} prereg={prereg.get('head')}"
        )
    if {path.name for path in OUT_DIR.iterdir()} != {PREREG_PATH.name}:
        raise RuntimeError("forward-only pre-audit inventory mismatch")
    if _sha(HARNESS) != prereg["harness"]["sha256"]:
        raise RuntimeError("D358 harness changed after preregistration")
    if _sha(SESSION_DOC) != prereg["session_doc"]["sha256"]:
        raise RuntimeError("D358 session preregistration changed before audit")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("frozen input changed after prepare")
    environment = _environment_preflight()
    if not environment["pass"]:
        raise RuntimeError(f"standalone core-PXR preflight failed: {environment['checks']}")
    if _forbidden_modules():
        raise RuntimeError(f"forbidden runtime modules loaded: {_forbidden_modules()}")

    _write_json_x(
        INVOCATION_PATH,
        {
            "artifact": "D358_SINGLE_AUDIT_INVOCATION_V1",
            "audit_invocation_count": 1,
            "pid": os.getpid(),
            "argv": sys.argv,
            "harness_sha256": _sha(HARNESS),
            "environment_preflight_pass": True,
            "no_retry": True,
        },
    )
    PHASE_PATH.touch(exist_ok=False)
    _phase("audit_started")
    try:
        helpers = _load_d355_helpers()
        arrays = helpers._source_arrays()
        _phase("frozen_usd_streams_loaded")
        points = arrays["points_f32_mm"]
        counts = arrays["counts_i64"]
        indices = arrays["indices_i64"]
        faces = arrays["triangles_i64"]
        raw = arrays["raw_f64_m"]
        streams = {
            "points_f32_mm": _sha_bytes(helpers._blob(points, "<f4")),
            "face_counts_i64": _sha_bytes(helpers._blob(counts, "<i8")),
            "face_indices_i64": _sha_bytes(helpers._blob(indices, "<i8")),
        }

        authored_recipe = helpers._current_recipe("<f4")
        raw_recipe = helpers._current_recipe("<f8")
        authored_inner = helpers._canonical_patch(points, faces, helpers.INNER_FACE_IDS, authored_recipe)
        authored_outer = helpers._canonical_patch(points, faces, helpers.OUTER_FACE_IDS, authored_recipe)
        raw_inner = helpers._canonical_patch(raw, faces, helpers.INNER_FACE_IDS, raw_recipe)
        raw_outer = helpers._canonical_patch(raw, faces, helpers.OUTER_FACE_IDS, raw_recipe)
        authored_bundle = helpers._bundle(authored_inner, authored_outer)
        raw_bundle = helpers._bundle(raw_inner, raw_outer)
        raw_full_hash = _sha_bytes(helpers._blob(raw, "<f8"))

        independent = _independent_patch_checks(
            helpers, points, raw, faces, authored_inner, authored_outer,
            raw_inner, raw_outer, raw_full_hash
        )
        roundtrip = np.asarray(raw * 1000.0, dtype="<f4")
        delta_mm = raw * 1000.0 - points.astype(np.float64)
        mismatch = roundtrip != points
        runtime_roundtrip = {
            "array_equal": bool(np.array_equal(roundtrip, points)),
            "mismatched_component_count": int(np.count_nonzero(mismatch)),
            "mismatched_vertex_count": int(np.count_nonzero(np.any(mismatch, axis=1))),
            "total_component_count": int(points.size),
            "max_abs_delta_mm": float(np.max(np.abs(delta_mm))),
            "mean_abs_delta_mm": float(np.mean(np.abs(delta_mm))),
            "median_abs_delta_mm": float(np.median(np.abs(delta_mm))),
            "per_axis_max_abs_delta_mm": np.max(np.abs(delta_mm), axis=0).tolist(),
            "body_from_mesh_matrix": arrays["body_from_mesh_matrix"],
        }

        sources = helpers._coordinate_sources(points, raw)
        _phase("recipe_grid_started", candidate_count=prereg["candidate_count"])
        search = helpers._search_recipes(sources, faces)
        search["reproduced_expected_field_count"] = sum(
            bool(rows) for rows in search["matches_expected_by_field_first_50"].values()
        )
        _phase(
            "recipe_grid_finished",
            candidate_count=search["candidate_count"],
            reproduced_expected_field_count=search["reproduced_expected_field_count"],
            coherent_bundle_found=search["exact_frozen_expected_bundle_found"],
        )
        discovered_recipe_replay = _discovered_recipe_independent_replay(
            helpers, sources, faces, search
        )
        byte_layout_controls = _matched_recipe_byte_layout_controls(
            helpers, sources, faces, search
        )
        negative = helpers._negative_controls(points, faces)
        manifest = _json(D339_MANIFEST)
        binding = _json(D354_BINDING)
        measurement = _json(D354_MEASUREMENT)
        completion = _json(D354_COMPLETION)
        attestation = _json(D354_ATTESTATION)

        checks = {
            "immutable_inputs_exact": _input_hashes() == prereg["input_hashes"],
            "d334_sidecar_unchanged": _sidecar_snapshot() == prereg["d334_sidecar_before"],
            "standalone_core_pxr_exact": _environment_preflight()["pass"],
            "authored_streams_exact": streams == helpers.EXPECTED_AUTHORED_STREAM_HASHES,
            "all_faces_triangles_13698": bool(np.all(counts == 3) and len(faces) == 13698),
            "raw_full_stream_exact": raw_full_hash
            == manifest["source_meshes"]["gripper_link"]["vertex_stream_sha256"]
            == helpers.EXPECTED_RAW_FULL_VERTEX_STREAM_SHA256,
            "d354_authored_bundle_reproduced": authored_bundle == helpers.OBSERVED_D354_AUTHORED_HASHES,
            "d354_raw_paired_xz_reproduced": raw_inner["paired_xz_sha256"]
            == helpers.OBSERVED_D354_RAW_PAIRED_XZ_SHA256,
            "independent_recalculations_exact": all(independent.values()),
            "discovered_recipe_independent_replay_exact": discovered_recipe_replay["pass"],
            "matched_recipe_byte_layout_controls_exact": byte_layout_controls["pass"],
            "all_expected_fields_reproduced": search["all_expected_fields_have_declared_recipe"],
            "negative_controls_7_of_7": negative["pass"] and negative["passed"] == negative["total"] == 7,
            "d354_binding_original_fail_preserved": binding.get("pass") is False,
            "d354_science_verdict_preserved": measurement["classification"]["scientific_verdict"]
            == "D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP",
            "d354_completion_preserved": completion.get("completion_pass") is True
            and completion.get("g0a_pass") is False,
            "d354_controlled_steps_zero": attestation.get("d354_controlled_physics_steps") == 0,
            "forbidden_runtime_modules_absent": _forbidden_modules() == [],
        }
        localization_prerequisites = all(
            checks[key]
            for key in [
                "immutable_inputs_exact",
                "d334_sidecar_unchanged",
                "standalone_core_pxr_exact",
                "authored_streams_exact",
                "all_faces_triangles_13698",
                "raw_full_stream_exact",
                "d354_authored_bundle_reproduced",
                "d354_raw_paired_xz_reproduced",
                "independent_recalculations_exact",
                "discovered_recipe_independent_replay_exact",
                "matched_recipe_byte_layout_controls_exact",
                "all_expected_fields_reproduced",
                "negative_controls_7_of_7",
                "d354_binding_original_fail_preserved",
                "d354_science_verdict_preserved",
                "d354_completion_preserved",
                "d354_controlled_steps_zero",
                "forbidden_runtime_modules_absent",
            ]
        )
        coherent = bool(search["exact_frozen_expected_bundle_found"])
        if localization_prerequisites and coherent:
            verdict = VERDICT_COHERENT
        elif localization_prerequisites:
            verdict = VERDICT_COMPOSITE
        else:
            verdict = VERDICT_UNRESOLVED
        provenance_localized = verdict in {VERDICT_COHERENT, VERDICT_COMPOSITE}

        evidence = {
            "artifact": "D358_MOVING_JAW_PATCH_HASH_PROVENANCE_EVIDENCE_V1",
            "case": "g0a_d358",
            "elapsed_seconds": time.monotonic() - started,
            "new_variables": NEW_VARIABLES,
            "new_physical_variables": [],
            "environment": environment,
            "input_hashes": _input_hashes(),
            "typed_stream_layout": {
                "authored_points": _typed_layout(points, "millimeters"),
                "face_counts": _typed_layout(counts, "count"),
                "face_indices": _typed_layout(indices, "vertex_index"),
                "raw_body_points": _typed_layout(raw, "meters"),
            },
            "authored_stream_hashes": streams,
            "authored_current_recipe": {
                "recipe": authored_recipe,
                "bundle": authored_bundle,
                "expected_d354_bundle": helpers.OBSERVED_D354_AUTHORED_HASHES,
                "inner_vertex_count": int(len(authored_inner["vertices"])),
                "outer_vertex_count": int(len(authored_outer["vertices"])),
            },
            "raw_current_recipe": {
                "recipe": raw_recipe,
                "full_vertex_stream_sha256": raw_full_hash,
                "bundle": raw_bundle,
                "inner_outer_xz_array_equal": bool(np.array_equal(raw_inner["xz"], raw_outer["xz"])),
                "observed_d354_paired_xz_sha256": helpers.OBSERVED_D354_RAW_PAIRED_XZ_SHA256,
            },
            "runtime_roundtrip": runtime_roundtrip,
            "independent_recalculation": {"checks": independent, "pass": all(independent.values())},
            "discovered_recipe_independent_replay": discovered_recipe_replay,
            "matched_recipe_byte_layout_controls": byte_layout_controls,
            "recipe_search": search,
            "first_matching_recipe_by_expected_field": _first_recipe_by_field(search),
            "negative_controls": negative,
            "checks": checks,
            "localization_prerequisites_pass": localization_prerequisites,
            "coherent_eight_field_recipe_found": coherent,
            "provenance_localized": provenance_localized,
            "verdict": verdict,
            "scope_guards": {
                "simulation_app_or_kit": 0,
                "isaac_gui": 0,
                "gpu_or_nvidia_smi": 0,
                "q5_science": 0,
                "physics_steps": 0,
                "distance_contact_queries": 0,
                "new_cap_rim_classifications": 0,
                "asset_writes": 0,
                "target_ik_path_changes": 0,
                "dependency_changes": 0,
                "d334_sidecar_writes": 0,
            },
            "interpretation_boundary": {
                "d354_verdict_changed": False,
                "barrel_first_decided": False,
                "physical_contact_or_grasp_decided": False,
                "binding_gate_changed": False,
                "target_ik_repair_justified": False,
                "actual_physx_test_requires_separate_approval": True,
            },
            "rerun_omitted": True,
            "rerun_omission_reason": prereg["rerun_omission"],
        }
        _write_json_x(EVIDENCE_PATH, evidence)
        _phase("authoritative_evidence_written", verdict=verdict)
        report = "\n".join(
            [
                "# D358 offline moving-jaw hash provenance audit",
                "",
                f"- Verdict: `{verdict}`",
                f"- Candidate recipes: `{search['candidate_count']}`",
                f"- Frozen expected fields reproduced: `{search['reproduced_expected_field_count']}/8`",
                f"- One coherent eight-field recipe: `{coherent}`",
                f"- D354 authored bundle reproduced: `{checks['d354_authored_bundle_reproduced']}`",
                f"- D354 raw paired-XZ reproduced: `{checks['d354_raw_paired_xz_reproduced']}`",
                f"- Independent checks: `{sum(independent.values())}/{len(independent)}`",
                f"- Discovered target recipes independently replayed: `{discovered_recipe_replay['pass']}`",
                f"- Target-matched big-endian/Fortran controls: `{byte_layout_controls['pass']}`",
                f"- Negative controls: `{negative['passed']}/{negative['total']}`",
                "- No Isaac/Kit/GPU/q5/physics/contact/cap-rim run occurred.",
                "- D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP is preserved.",
            ]
        )
        _write_text_x(REPORT_PATH, report)
        _phase("completion_ready", verdict=verdict, elapsed_seconds=time.monotonic() - started)
        observed_before_completion = sorted(path.name for path in OUT_DIR.iterdir())
        expected_before_completion = sorted(
            set(prereg["expected_success_inventory"]) - {COMPLETION_PATH.name}
        )
        if observed_before_completion != expected_before_completion:
            raise RuntimeError(
                "pre-completion inventory mismatch: "
                f"observed={observed_before_completion}, expected={expected_before_completion}"
            )
        completion_payload = {
            "artifact": "D358_COMPLETION_SUMMARY_V1",
            "case": "g0a_d358",
            "operational_pass": True,
            "provenance_localized": provenance_localized,
            "coherent_eight_field_recipe_found": coherent,
            "verdict": verdict,
            "evidence_path": _rel(EVIDENCE_PATH),
            "evidence_sha256": _sha(EVIDENCE_PATH),
            "report_sha256": _sha(REPORT_PATH),
            "audit_invocation_count": 1,
            "d354_scientific_verdict": "D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP",
            "d354_scientific_verdict_changed": False,
            "g0a_pass": False,
            "scope_guards": evidence["scope_guards"],
            "d334_sidecar_unchanged": checks["d334_sidecar_unchanged"],
            "rerun_omitted_as_pure_hash_audit": True,
            "pass": provenance_localized,
        }
        _write_json_x(COMPLETION_PATH, completion_payload)
        signal.alarm(0)
        print(
            json.dumps(
                {
                    "stage": "audit",
                    "pass": completion_payload["pass"],
                    "verdict": verdict,
                    "coherent_eight_field_recipe_found": coherent,
                    "output": _rel(OUT_DIR),
                },
                ensure_ascii=False,
            )
        )
    except Exception as error:
        signal.alarm(0)
        if not EXCEPTION_PATH.exists():
            _write_json_x(
                EXCEPTION_PATH,
                {
                    "artifact": "D358_RUNTIME_EXCEPTION_V1",
                    "case": "g0a_d358",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "verdict": VERDICT_INPUT_STOP,
                    "audit_invocation_count": 1,
                    "automatic_retry": False,
                    "d354_scientific_verdict_changed": False,
                    "g0a_pass": False,
                },
            )
        try:
            _phase("audit_exception", error=f"{type(error).__name__}: {error}")
        except Exception:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "audit"), required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        _prepare()
    else:
        _audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
