#!/usr/bin/env python3
"""D345 proof-only deterministic USD metadata comparator repair.

This harness reads immutable D339 attempt2 and D344 attempt3 assets.  It never
authors or recooks collision geometry, starts Isaac/Kit, advances physics, or
creates Rerun artifacts.  Two fresh standalone-PXR worker processes encode the
same composed USD scene with a closed, typed serializer.  Default ``repr`` is
used only in a registered negative control that must be rejected as process-
local and nondeterministic.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import re
import secrets
import struct
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d345"
PREREG_PATH = OUT_DIR / "d345_preregistration.json"
PARAMETER_AUDIT_PATH = OUT_DIR / "d345_parameter_freeze_audit.json"
RERUN_OMISSION_PATH = OUT_DIR / "d345_rerun_omission_justification.json"
WORKER_A_PATH = OUT_DIR / "d345_worker_a.json"
WORKER_B_PATH = OUT_DIR / "d345_worker_b.json"
EVIDENCE_PATH = OUT_DIR / "d345_deterministic_usd_metadata_evidence.json"
SUMMARY_PATH = OUT_DIR / "d345_deterministic_usd_metadata_summary.json"
REPORT_PATH = OUT_DIR / "d345_deterministic_usd_metadata_report.md"

D339_ATTEMPT2 = (
    REPO_ROOT
    / "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2"
)
D339_PHYSICS = (
    D339_ATTEMPT2
    / "roarm_m3_fullmesh_convex_parts/configuration/roarm_m3_physics.usd"
)
D344_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d344"
D344_ATTEMPT3 = D344_DIR / "collision_asset/attempt3"
D344_PHYSICS = (
    D344_ATTEMPT3
    / "roarm_m3_fullmesh_fixed_point_parts/configuration/roarm_m3_physics.usd"
)
D344_OUTER_MANIFEST = D344_ATTEMPT3 / "d344_attempt3_asset_manifest.json"
D344_CORE_MANIFEST = D344_ATTEMPT3 / "d340_attempt3_asset_manifest.json"
D344_DIAGNOSIS_A = D344_DIR / "d344_postrun_semantic_diagnosis.json"
D344_DIAGNOSIS_B = D344_DIR / "d344_postrun_semantic_diagnosis_repeat.json"
D344_ROOT_CAUSE = D344_DIR / "d344_postrun_root_cause_audit.json"
D344_BUILD_SUMMARY = D344_DIR / "d344_attempt3_build_summary.json"
D344_HARNESS = (
    REPO_ROOT
    / "sim_scripts/cyl34_top_view_d344_grasp_g0a_attempt3_fixed_point_collision_geometry.py"
)
D340_HARNESS = (
    REPO_ROOT
    / "sim_scripts/cyl34_top_view_d340_grasp_g0a_fixed_point_live_authoring_repair.py"
)
D344_SESSION = (
    REPO_ROOT
    / "claudedocs/session_20260714_grasp_g0a_d344_attempt3_fixed_point_collision_geometry.md"
)
D345_SESSION = (
    REPO_ROOT
    / "claudedocs/session_20260714_grasp_g0a_d345_deterministic_usd_metadata_comparator.md"
)
AGENTS = REPO_ROOT / "AGENTS.md"
START_HERE = REPO_ROOT / "START_HERE.md"
BACKLOG = REPO_ROOT / "claudedocs/BACKLOG.md"

USD_CORE_EXT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
REGISTERED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
REGISTERED_PYTHONPATH = str(USD_CORE_EXT)
REGISTERED_LD_LIBRARY_PATH = ":".join(
    (
        "/home/cgxr/miniconda3/envs/isaaclab/lib",
        str(USD_CORE_EXT / "bin"),
    )
)

NEW_VARIABLES = ["deterministic_usd_metadata_comparator"]
VERDICT_PASS = "D345_DETERMINISTIC_USD_METADATA_COMPARATOR_PASS"
VERDICT_FAIL = "D345_DETERMINISTIC_USD_METADATA_COMPARATOR_FAIL_STOP"
MASK_MARKER = {"$type": "D345RegisteredGeometryValueMask"}
RUNTIME_ADDRESS_RE = re.compile(
    r"<pxr\.[A-Za-z0-9_.]+ object at 0x[0-9a-fA-F]+>"
)

CHANGED_PARTS = {
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
GEOMETRY_ATTRIBUTES = ("points", "faceVertexCounts", "faceVertexIndices")
EXPECTED_ALLOWED_PATHS = tuple(
    sorted(
        f"/colliders/{body}/d338_convex_parts/{part}.{attribute}"
        for body in ("link5", "gripper_link")
        for part in CHANGED_PARTS[body]
        for attribute in GEOMETRY_ATTRIBUTES
    )
)
EXPECTED_ALLOWED_TYPES = {
    "points": "point3f[]",
    "faceVertexCounts": "int[]",
    "faceVertexIndices": "int[]",
}
REPRESENTATIVE_PART_PATH = "/colliders/link5/d338_convex_parts/part_011"
REPRESENTATIVE_AUTHORED_TOKENS = [
    "PhysicsCollisionAPI",
    "PhysicsMeshCollisionAPI",
    "PhysxConvexHullCollisionAPI",
]
REPRESENTATIVE_CORE_APPLIED_TOKENS = [
    "PhysicsCollisionAPI",
    "PhysicsMeshCollisionAPI",
]

EXPECTED_COMPOSED_PRIM_COUNT = 310
EXPECTED_COMPOSED_API_SCHEMA_COUNT = 194
EXPECTED_DIRECT_API_SCHEMA_COUNT = 149
EXPECTED_D339_FILE_COUNT = 18
EXPECTED_D339_DIGEST = (
    "0dae41fd3937a0a8aea18488019c74f097d32f7b8de916943ff31334e30464a1"
)
EXPECTED_D344_ATTEMPT3_FILE_COUNT = 9
EXPECTED_D344_ATTEMPT3_DIGEST = (
    "ea6965199ff1f195a6d19d9c55febfe44cc9838f12651570c80d5bb97fa6caf1"
)
EXPECTED_ALLOWED_PATHS_SHA256 = (
    "8bc84599bdccd45318cb08a27800453eea8b91c041abdfa75201306e968bfe27"
)


class UnsupportedUsdValue(TypeError):
    """Raised instead of silently falling back to repr/str."""


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dumps(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _assert_output_path(path: Path) -> None:
    if not path.resolve().is_relative_to(OUT_DIR.resolve()):
        raise RuntimeError(f"D345 may write only under {OUT_DIR}: {path}")


def _write_json(path: Path, value: Any) -> None:
    _assert_output_path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(_dumps(value, pretty=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    _assert_output_path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(value, encoding="utf-8")


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_status() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        rows.append(
            {
                "path": _relative(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return rows


def _inventory_digest(rows: list[dict[str, Any]]) -> str:
    return _sha256_text(_dumps(rows))


def _allowed_paths_sha256(paths: list[str] | tuple[str, ...]) -> str:
    return _sha256_text(_dumps(list(paths)))


def _unrelated_dirty_snapshot() -> dict[str, Any]:
    owned_prefixes = (
        "START_HERE.md",
        "AGENTS.md",
        "claudedocs/BACKLOG.md",
        _relative(D345_SESSION),
        _relative(Path(__file__).resolve()),
        "claudedocs/runtime_logs/grasp_track/g0a_d345",
    )
    rows = []
    for line in _git_status():
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        if any(path_text == prefix or path_text.startswith(prefix + "/") for prefix in owned_prefixes):
            continue
        path = REPO_ROOT / path_text.rstrip("/")
        inventory = _inventory(path) if path.exists() else []
        rows.append(
            {
                "status": line[:2],
                "path": path_text,
                "inventory": inventory,
                "inventory_digest": _inventory_digest(inventory),
            }
        )
    return {"rows": rows, "digest": _sha256_text(_dumps(rows))}


def _source_hashes() -> dict[str, str]:
    paths = {
        "d345_harness": Path(__file__).resolve(),
        "d345_session": D345_SESSION,
        "agents": AGENTS,
        "start_here": START_HERE,
        "backlog": BACKLOG,
        "d339_physics": D339_PHYSICS,
        "d344_physics": D344_PHYSICS,
        "d344_outer_manifest": D344_OUTER_MANIFEST,
        "d344_core_manifest": D344_CORE_MANIFEST,
        "d344_diagnosis_a": D344_DIAGNOSIS_A,
        "d344_diagnosis_b": D344_DIAGNOSIS_B,
        "d344_root_cause": D344_ROOT_CAUSE,
        "d344_build_summary": D344_BUILD_SUMMARY,
        "d344_harness": D344_HARNESS,
        "d340_harness": D340_HARNESS,
        "d344_session": D344_SESSION,
        "pxr_usd_module": USD_CORE_EXT / "pxr/Usd/_usd.so",
        "pxr_sdf_module": USD_CORE_EXT / "pxr/Sdf/_sdf.so",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _float_bits(value: float, width: int) -> str:
    if width == 16:
        return struct.pack("<e", float(value)).hex()
    if width == 32:
        return struct.pack("<f", float(value)).hex()
    if width == 64:
        return struct.pack("<d", float(value)).hex()
    raise ValueError(f"unsupported float width: {width}")


def _canonical_float(value: float, width: int) -> dict[str, Any]:
    if math.isnan(float(value)):
        classification = "nan"
    elif math.isinf(float(value)):
        classification = "+inf" if value > 0 else "-inf"
    else:
        classification = "finite"
    return {
        "$type": f"float{width}",
        "bits_le": _float_bits(float(value), width),
        "class": classification,
    }


def _pxr_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__name__}"


class CanonicalUsdEncoder:
    """Closed typed encoder. Unknown values are scientific failures."""

    def __init__(self) -> None:
        self.type_counts: Counter[str] = Counter()

    def encode(self, value: Any) -> Any:
        type_name = _pxr_type_name(value)
        self.type_counts[type_name] += 1

        if value is None:
            return {"$type": "none"}
        if isinstance(value, bool):
            return {"$type": "bool", "value": value}
        if isinstance(value, int):
            return {"$type": "int", "value": str(value)}
        if isinstance(value, float):
            return _canonical_float(value, 64)
        if isinstance(value, str):
            return {"$type": "str", "value": value}
        if isinstance(value, bytes):
            return {"$type": "bytes", "hex": value.hex()}

        if isinstance(value, Sdf.Path):
            return {"$type": "pxr.Sdf.Path", "path": value.pathString}
        if isinstance(value, Sdf.AssetPath):
            return {
                "$type": "pxr.Sdf.AssetPath",
                "authored_path": value.path,
                "resolved_path": value.resolvedPath,
            }
        if isinstance(value, Sdf.LayerOffset):
            return {
                "$type": "pxr.Sdf.LayerOffset",
                "offset": _canonical_float(value.offset, 64),
                "scale": _canonical_float(value.scale, 64),
            }
        if isinstance(value, Sdf.Reference):
            return {
                "$type": "pxr.Sdf.Reference",
                "asset_path": value.assetPath,
                "prim_path": value.primPath.pathString,
                "layer_offset": self.encode(value.layerOffset),
                "custom_data": self.encode(value.customData),
            }
        if isinstance(value, Sdf.Payload):
            return {
                "$type": "pxr.Sdf.Payload",
                "asset_path": value.assetPath,
                "prim_path": value.primPath.pathString,
                "layer_offset": self.encode(value.layerOffset),
            }
        if isinstance(value, Sdf.ValueTypeName):
            return {
                "$type": "pxr.Sdf.ValueTypeName",
                "name": str(value),
                "aliases": list(value.aliasesAsStrings),
                "cpp_type_name": value.cppTypeName,
                "is_array": bool(value.isArray),
                "is_scalar": bool(value.isScalar),
                "role": value.role,
            }
        if isinstance(value, (Sdf.Specifier, Sdf.Variability)):
            return {
                "$type": type_name,
                "name": value.name,
                "value": str(value.value),
            }
        if isinstance(value, Sdf.ValueBlock):
            return {"$type": "pxr.Sdf.ValueBlock"}

        if type(value).__module__ == "pxr.Sdf" and type(value).__name__.endswith("ListOp"):
            required = (
                "isExplicit",
                "explicitItems",
                "addedItems",
                "prependedItems",
                "appendedItems",
                "deletedItems",
                "orderedItems",
            )
            if not all(hasattr(value, field) for field in required):
                raise UnsupportedUsdValue(f"incomplete Sdf ListOp binding: {type_name}")
            return {
                "$type": type_name,
                "is_explicit": bool(value.isExplicit),
                "explicit_items": [self.encode(item) for item in value.explicitItems],
                "added_items": [self.encode(item) for item in value.addedItems],
                "prepended_items": [self.encode(item) for item in value.prependedItems],
                "appended_items": [self.encode(item) for item in value.appendedItems],
                "deleted_items": [self.encode(item) for item in value.deletedItems],
                "ordered_items": [self.encode(item) for item in value.orderedItems],
                "applied_items_derived": [
                    self.encode(item) for item in value.GetAppliedItems()
                ],
            }

        module_name = type(value).__module__
        class_name = type(value).__name__
        if module_name == "pxr.Gf" and class_name.startswith("Quat"):
            suffix = class_name[-1]
            width = {"h": 16, "f": 32, "d": 64}.get(suffix)
            if width is None:
                raise UnsupportedUsdValue(f"unsupported quaternion type: {type_name}")
            imaginary = value.GetImaginary()
            return {
                "$type": type_name,
                "real": _canonical_float(value.GetReal(), width),
                "imaginary_xyz": [
                    _canonical_float(component, width) for component in imaginary
                ],
            }
        if module_name == "pxr.Gf" and class_name.startswith("Vec"):
            suffix = class_name[-1]
            if suffix in {"h", "f", "d"}:
                width = {"h": 16, "f": 32, "d": 64}[suffix]
                items = [_canonical_float(component, width) for component in value]
            elif suffix in {"i", "l"}:
                items = [{"$type": "int", "value": str(int(component))} for component in value]
            else:
                raise UnsupportedUsdValue(f"unsupported vector type: {type_name}")
            return {"$type": type_name, "items": items}
        if module_name == "pxr.Gf" and class_name.startswith("Matrix"):
            return {
                "$type": type_name,
                "rows": [self.encode(row) for row in value],
            }

        if module_name == "pxr.Vt" and class_name.endswith("Array"):
            if class_name in {"FloatArray", "DoubleArray", "HalfArray"}:
                width = {"FloatArray": 32, "DoubleArray": 64, "HalfArray": 16}[class_name]
                items = [_canonical_float(item, width) for item in value]
            elif class_name in {
                "IntArray",
                "Int64Array",
                "UIntArray",
                "UInt64Array",
                "ShortArray",
                "UShortArray",
                "CharArray",
                "UCharArray",
            }:
                items = [{"$type": "int", "value": str(int(item))} for item in value]
            elif class_name in {"TokenArray", "StringArray"}:
                items = [{"$type": "str", "value": str(item)} for item in value]
            else:
                items = [self.encode(item) for item in value]
            return {"$type": type_name, "length": len(value), "items": items}

        if isinstance(value, dict):
            encoded_items = []
            for key, item in value.items():
                encoded_key = self.encode(key)
                encoded_items.append((_dumps(encoded_key), encoded_key, self.encode(item)))
            encoded_items.sort(key=lambda row: row[0])
            return {
                "$type": "dict",
                "items": [[row[1], row[2]] for row in encoded_items],
            }
        if isinstance(value, list):
            return {"$type": "list", "items": [self.encode(item) for item in value]}
        if isinstance(value, tuple):
            return {"$type": "tuple", "items": [self.encode(item) for item in value]}

        raise UnsupportedUsdValue(f"unsupported USD/PXR value type: {type_name}")


def _canonical_metadata(encoder: CanonicalUsdEncoder, metadata: dict[str, Any]) -> list[Any]:
    return [[key, encoder.encode(metadata[key])] for key in sorted(metadata)]


def _stage_rows(path: Path, allowed: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open stage: {path}")
    encoder = CanonicalUsdEncoder()
    rows: list[dict[str, Any]] = []
    total_time_sample_count = 0
    prims = sorted(Usd.PrimRange.Stage(stage), key=lambda prim: prim.GetPath().pathString)
    for prim in prims:
        prim_path = prim.GetPath().pathString
        attributes = []
        for attr in sorted(prim.GetAttributes(), key=lambda item: item.GetName()):
            property_path = f"{prim_path}.{attr.GetName()}"
            value = MASK_MARKER if property_path in allowed else encoder.encode(attr.Get())
            time_samples = list(attr.GetTimeSamples())
            total_time_sample_count += len(time_samples)
            attributes.append(
                {
                    "name": attr.GetName(),
                    "type_name": str(attr.GetTypeName()),
                    "value": value,
                    "time_samples": [
                        {
                            "time": _canonical_float(time_code, 64),
                            "value": encoder.encode(attr.Get(time_code)),
                        }
                        for time_code in time_samples
                    ],
                    "connections": [encoder.encode(item) for item in attr.GetConnections()],
                    "metadata": _canonical_metadata(encoder, attr.GetAllMetadata()),
                }
            )
        relationships = []
        for rel in sorted(prim.GetRelationships(), key=lambda item: item.GetName()):
            relationships.append(
                {
                    "name": rel.GetName(),
                    "targets": [encoder.encode(item) for item in rel.GetTargets()],
                    "metadata": _canonical_metadata(encoder, rel.GetAllMetadata()),
                }
            )
        rows.append(
            {
                "path": prim_path,
                "type_name": str(prim.GetTypeName()),
                "active": bool(prim.IsActive()),
                "instanceable": bool(prim.IsInstanceable()),
                "applied_schemas": [str(item) for item in prim.GetAppliedSchemas()],
                "metadata": _canonical_metadata(encoder, prim.GetAllMetadata()),
                "attributes": attributes,
                "relationships": relationships,
            }
        )
    payload = _dumps(rows)
    address_leaks = RUNTIME_ADDRESS_RE.findall(payload)
    return rows, {
        "row_count": len(rows),
        "paths": [row["path"] for row in rows],
        "row_sha256": [_sha256_text(_dumps(row)) for row in rows],
        "canonical_json_bytes": len(payload.encode("utf-8")),
        "canonical_sha256": _sha256_text(payload),
        "runtime_address_pattern_count": len(address_leaks),
        "time_sample_count": total_time_sample_count,
        "type_counts": dict(sorted(encoder.type_counts.items())),
        "unsupported_type_count": 0,
    }


def _api_schema_rows(path: Path) -> dict[str, Any]:
    layer = Sdf.Layer.FindOrOpen(str(path))
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if layer is None or stage is None:
        raise RuntimeError(f"failed to open API schema source: {path}")

    direct_encoder = CanonicalUsdEncoder()
    composed_encoder = CanonicalUsdEncoder()
    prims = sorted(Usd.PrimRange.Stage(stage), key=lambda prim: prim.GetPath().pathString)
    direct_rows = []
    composed_rows = []
    for prim in prims:
        prim_path = prim.GetPath().pathString
        prim_spec = layer.GetPrimAtPath(Sdf.Path(prim_path))
        if prim_spec is not None and prim_spec.HasInfo("apiSchemas"):
            direct_rows.append(
                {
                    "path": prim_path,
                    "authored_list_op": direct_encoder.encode(
                        prim_spec.GetInfo("apiSchemas")
                    ),
                }
            )
        metadata = prim.GetAllMetadata()
        if "apiSchemas" in metadata:
            composed_rows.append(
                {
                    "path": prim_path,
                    "composed_metadata_list_op": composed_encoder.encode(
                        metadata["apiSchemas"]
                    ),
                    "get_applied_schemas": [str(item) for item in prim.GetAppliedSchemas()],
                }
            )
    return {
        "direct_rows": direct_rows,
        "direct_row_count": len(direct_rows),
        "direct_sha256": _sha256_text(_dumps(direct_rows)),
        "direct_type_counts": dict(sorted(direct_encoder.type_counts.items())),
        "composed_rows": composed_rows,
        "composed_row_count": len(composed_rows),
        "composed_sha256": _sha256_text(_dumps(composed_rows)),
        "composed_type_counts": dict(sorted(composed_encoder.type_counts.items())),
    }


def _legacy_api_schema_diagnostic(path: Path) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open legacy diagnostic stage: {path}")
    rows = []
    for prim in sorted(Usd.PrimRange.Stage(stage), key=lambda item: item.GetPath().pathString):
        metadata = prim.GetAllMetadata()
        if "apiSchemas" in metadata:
            rows.append([prim.GetPath().pathString, repr(metadata["apiSchemas"])])
    payload = _dumps(rows)
    return {
        "row_count": len(rows),
        "runtime_address_pattern_count": len(RUNTIME_ADDRESS_RE.findall(payload)),
        "address_bearing_sha256": _sha256_text(payload),
        "scientific_authority": False,
        "rejection_reason": "repr(Sdf.TokenListOp) contains process-local RAM addresses",
    }


def _clone_token_list_op(value: Any) -> Any:
    clone = Sdf.TokenListOp()
    clone.explicitItems = list(value.explicitItems)
    clone.addedItems = list(value.addedItems)
    clone.prependedItems = list(value.prependedItems)
    clone.appendedItems = list(value.appendedItems)
    clone.deletedItems = list(value.deletedItems)
    clone.orderedItems = list(value.orderedItems)
    if value.isExplicit:
        clone = Sdf.TokenListOp.CreateExplicit(list(value.explicitItems))
    return clone


def _token_list_op_controls(path: Path) -> dict[str, Any]:
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise RuntimeError(f"failed to open token control layer: {path}")
    prim_spec = layer.GetPrimAtPath(Sdf.Path(REPRESENTATIVE_PART_PATH))
    if prim_spec is None or not prim_spec.HasInfo("apiSchemas"):
        raise RuntimeError("representative authored apiSchemas missing")
    original = prim_spec.GetInfo("apiSchemas")
    encoder = CanonicalUsdEncoder()
    original_canonical = encoder.encode(original)
    original_hash = _sha256_text(_dumps(original_canonical))
    original_applied = [str(item) for item in original.GetAppliedItems()]

    equivalent_a = _clone_token_list_op(original)
    equivalent_b = _clone_token_list_op(original)
    equivalent_old_repr_different = repr(equivalent_a) != repr(equivalent_b)
    equivalent_canonical_equal = (
        _dumps(CanonicalUsdEncoder().encode(equivalent_a))
        == _dumps(CanonicalUsdEncoder().encode(equivalent_b))
        == _dumps(original_canonical)
    )

    token_removed = _clone_token_list_op(original)
    if token_removed.prependedItems:
        removed_token = str(token_removed.prependedItems[-1])
        token_removed.prependedItems = list(token_removed.prependedItems)[:-1]
    elif token_removed.explicitItems:
        removed_token = str(token_removed.explicitItems[-1])
        token_removed = Sdf.TokenListOp.CreateExplicit(list(token_removed.explicitItems)[:-1])
    else:
        raise RuntimeError("representative list op has no removable token")
    token_removed_canonical = CanonicalUsdEncoder().encode(token_removed)
    token_removed_hash = _sha256_text(_dumps(token_removed_canonical))
    token_removed_applied = [str(item) for item in token_removed.GetAppliedItems()]

    mode_changed = Sdf.TokenListOp.CreateExplicit(original_applied)
    mode_changed_canonical = CanonicalUsdEncoder().encode(mode_changed)
    mode_changed_hash = _sha256_text(_dumps(mode_changed_canonical))
    mode_changed_applied = [str(item) for item in mode_changed.GetAppliedItems()]
    checks = {
        "representative_is_non_explicit": original.isExplicit is False,
        "representative_prepended_tokens_exact": [
            str(item) for item in original.prependedItems
        ]
        == REPRESENTATIVE_AUTHORED_TOKENS,
        "equivalent_old_repr_differs": equivalent_old_repr_different,
        "equivalent_canonical_equal": equivalent_canonical_equal,
        "token_removal_detected": token_removed_hash != original_hash,
        "token_removal_exactly_one": len(token_removed_applied)
        == len(original_applied) - 1,
        "removed_token_absent_after_removal": removed_token not in token_removed_applied,
        "mode_change_effective_tokens_same": mode_changed_applied == original_applied,
        "prepend_to_explicit_mode_change_detected": mode_changed_hash != original_hash,
        "source_file_not_written": True,
    }
    return {
        "representative_path": REPRESENTATIVE_PART_PATH,
        "original": original_canonical,
        "original_sha256": original_hash,
        "original_applied_tokens": original_applied,
        "removed_token": removed_token,
        "token_removed": token_removed_canonical,
        "token_removed_sha256": token_removed_hash,
        "token_removed_applied_tokens": token_removed_applied,
        "mode_changed": mode_changed_canonical,
        "mode_changed_sha256": mode_changed_hash,
        "mode_changed_applied_tokens": mode_changed_applied,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _sanitized_layer_view(path: Path, allowed: list[str]) -> dict[str, Any]:
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise RuntimeError(f"failed to open layer for sanitized view: {path}")
    clone = Sdf.Layer.CreateAnonymous("d345_in_memory_sanitized.usda")
    if not clone.ImportFromString(layer.ExportToString()):
        raise RuntimeError("failed to construct in-memory sanitized layer view")
    edits = Sdf.BatchNamespaceEdit()
    missing = []
    for value in allowed:
        sdf_path = Sdf.Path(value)
        if clone.GetObjectAtPath(sdf_path) is None:
            missing.append(value)
        else:
            edits.Add(Sdf.NamespaceEdit.Remove(sdf_path))
    if missing or not clone.Apply(edits):
        raise RuntimeError(f"failed to mask exact registered properties: {missing}")
    text = clone.ExportToString()
    return {
        "sha256": _sha256_text(text),
        "text_bytes": len(text.encode("utf-8")),
        "removed_property_count": len(allowed),
        "disk_file_written": False,
    }


def _layer_header(path: Path) -> dict[str, Any]:
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise RuntimeError(f"failed to open layer header: {path}")
    encoder = CanonicalUsdEncoder()
    return {
        "default_prim": layer.defaultPrim,
        "sub_layer_paths": list(layer.subLayerPaths),
        "custom_layer_data": encoder.encode(layer.customLayerData),
        "start_time_code": _canonical_float(layer.startTimeCode, 64),
        "end_time_code": _canonical_float(layer.endTimeCode, 64),
        "time_codes_per_second": _canonical_float(layer.timeCodesPerSecond, 64),
        "frames_per_second": _canonical_float(layer.framesPerSecond, 64),
    }


def _allowed_attribute_type_checks(path: Path, allowed: list[str]) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open allowlist type stage: {path}")
    rows = []
    for property_path in allowed:
        sdf_path = Sdf.Path(property_path)
        prim = stage.GetPrimAtPath(sdf_path.GetPrimPath())
        attr = prim.GetAttribute(sdf_path.name)
        rows.append(
            {
                "path": property_path,
                "valid": bool(attr),
                "type_name": str(attr.GetTypeName()) if attr else None,
                "expected_type_name": EXPECTED_ALLOWED_TYPES[sdf_path.name],
            }
        )
    return {
        "rows": rows,
        "pass": all(
            row["valid"] and row["type_name"] == row["expected_type_name"]
            for row in rows
        ),
    }


def _worker_payload(worker_id: str) -> dict[str, Any]:
    core = _json(D344_CORE_MANIFEST)
    allowed = list(core["whole_physics_semantic_allowlist"]["allowed_property_paths"])
    allowed_set = set(allowed)

    source_rows, source_summary = _stage_rows(D339_PHYSICS, allowed_set)
    variant_rows, variant_summary = _stage_rows(D344_PHYSICS, allowed_set)
    source_api = _api_schema_rows(D339_PHYSICS)
    variant_api = _api_schema_rows(D344_PHYSICS)
    source_legacy = _legacy_api_schema_diagnostic(D339_PHYSICS)
    variant_legacy = _legacy_api_schema_diagnostic(D344_PHYSICS)
    token_controls = _token_list_op_controls(D339_PHYSICS)
    source_sanitized = _sanitized_layer_view(D339_PHYSICS, allowed)
    variant_sanitized = _sanitized_layer_view(D344_PHYSICS, allowed)
    source_header = _layer_header(D339_PHYSICS)
    variant_header = _layer_header(D344_PHYSICS)
    source_allowed_types = _allowed_attribute_type_checks(D339_PHYSICS, allowed)
    variant_allowed_types = _allowed_attribute_type_checks(D344_PHYSICS, allowed)

    source_rep_direct = next(
        row for row in source_api["direct_rows"] if row["path"] == REPRESENTATIVE_PART_PATH
    )
    source_rep_composed = next(
        row for row in source_api["composed_rows"] if row["path"] == REPRESENTATIVE_PART_PATH
    )
    checks = {
        "allowed_property_count_exact_39": len(allowed) == 39,
        "allowed_property_paths_literal_exact": allowed == list(EXPECTED_ALLOWED_PATHS),
        "allowed_property_paths_hash_exact": _allowed_paths_sha256(allowed)
        == EXPECTED_ALLOWED_PATHS_SHA256,
        "source_allowed_attribute_types_exact": source_allowed_types["pass"],
        "variant_allowed_attribute_types_exact": variant_allowed_types["pass"],
        "source_row_count_exact_310": source_summary["row_count"]
        == EXPECTED_COMPOSED_PRIM_COUNT,
        "variant_row_count_exact_310": variant_summary["row_count"]
        == EXPECTED_COMPOSED_PRIM_COUNT,
        "source_time_samples_zero": source_summary["time_sample_count"] == 0,
        "variant_time_samples_zero": variant_summary["time_sample_count"] == 0,
        "paths_exact": source_summary["paths"] == variant_summary["paths"],
        "canonical_rows_exact_after_39_mask": source_rows == variant_rows,
        "canonical_hash_exact_after_39_mask": source_summary["canonical_sha256"]
        == variant_summary["canonical_sha256"],
        "canonical_payload_has_no_runtime_address": source_summary[
            "runtime_address_pattern_count"
        ]
        == variant_summary["runtime_address_pattern_count"]
        == 0,
        "unsupported_type_count_zero": source_summary["unsupported_type_count"]
        == variant_summary["unsupported_type_count"]
        == 0,
        "type_distribution_exact": source_summary["type_counts"]
        == variant_summary["type_counts"],
        "direct_api_schema_count_exact_149": source_api["direct_row_count"]
        == variant_api["direct_row_count"]
        == EXPECTED_DIRECT_API_SCHEMA_COUNT,
        "direct_api_schema_rows_exact": source_api["direct_rows"]
        == variant_api["direct_rows"],
        "composed_api_schema_count_exact_194": source_api["composed_row_count"]
        == variant_api["composed_row_count"]
        == EXPECTED_COMPOSED_API_SCHEMA_COUNT,
        "composed_api_schema_rows_exact": source_api["composed_rows"]
        == variant_api["composed_rows"],
        "representative_direct_nonexplicit_prepend_exact": bool(
            source_rep_direct["authored_list_op"]["is_explicit"] is False
            and [
                item["value"]
                for item in source_rep_direct["authored_list_op"]["prepended_items"]
            ]
            == REPRESENTATIVE_AUTHORED_TOKENS
        ),
        "representative_composed_explicit_exact": bool(
            source_rep_composed["composed_metadata_list_op"]["is_explicit"] is True
            and [
                item["value"]
                for item in source_rep_composed["composed_metadata_list_op"][
                    "explicit_items"
                ]
            ]
            == REPRESENTATIVE_AUTHORED_TOKENS
        ),
        "representative_core_applied_schemas_exact": source_rep_composed[
            "get_applied_schemas"
        ]
        == REPRESENTATIVE_CORE_APPLIED_TOKENS,
        "legacy_source_address_rows_exact_194": source_legacy[
            "runtime_address_pattern_count"
        ]
        == source_legacy["row_count"]
        == EXPECTED_COMPOSED_API_SCHEMA_COUNT,
        "legacy_variant_address_rows_exact_194": variant_legacy[
            "runtime_address_pattern_count"
        ]
        == variant_legacy["row_count"]
        == EXPECTED_COMPOSED_API_SCHEMA_COUNT,
        "legacy_repr_marked_non_authoritative": (
            source_legacy["scientific_authority"] is False
            and variant_legacy["scientific_authority"] is False
        ),
        "token_list_op_controls_pass": token_controls["pass"],
        "sanitized_layer_exact": source_sanitized == variant_sanitized,
        "layer_header_exact": source_header == variant_header,
    }
    return {
        "artifact": "D345_STANDALONE_PXR_WORKER_V1",
        "worker_id": worker_id,
        "pid": os.getpid(),
        "nonce": secrets.token_hex(16),
        "python_executable": str(Path(sys.executable).resolve()),
        "pxr_usd_version": list(Usd.GetVersion()),
        "source": _relative(D339_PHYSICS),
        "variant": _relative(D344_PHYSICS),
        "allowed_property_paths": allowed,
        "source_summary": source_summary,
        "variant_summary": variant_summary,
        "source_api_schemas": source_api,
        "variant_api_schemas": variant_api,
        "source_legacy_repr_diagnostic": source_legacy,
        "variant_legacy_repr_diagnostic": variant_legacy,
        "token_list_op_controls": token_controls,
        "source_sanitized_layer": source_sanitized,
        "variant_sanitized_layer": variant_sanitized,
        "source_layer_header": source_header,
        "variant_layer_header": variant_header,
        "source_allowed_attribute_types": source_allowed_types,
        "variant_allowed_attribute_types": variant_allowed_types,
        "checks": checks,
        "pass": all(checks.values()),
        "runtime_environment_created": False,
        "controlled_physics_steps": 0,
        "asset_file_written": False,
        "rerun_artifact_created": False,
    }


def _parameter_audit() -> dict[str, Any]:
    return {
        "artifact": "D345_PARAMETER_FREEZE_AUDIT_V1",
        "pass": True,
        "new_variables": NEW_VARIABLES,
        "variable_count": 1,
        "measurement_only": True,
        "physical_variables_changed": [],
        "existing_parameter_increases": [],
        "existing_parameter_changes": [],
        "decomposition_parameter_changes": [],
        "threshold_relaxations": [],
        "target_control_solver_changes": [],
        "collision_asset_writes": 0,
        "collision_asset_copies": 0,
        "recooks": 0,
        "runtime_environment_created": False,
        "controlled_physics_steps": 0,
        "g0a_pass": False,
        "d344_verdict_reclassified": False,
    }


def _rerun_omission() -> dict[str, Any]:
    return {
        "artifact": "D345_RERUN_OMISSION_JUSTIFICATION_V1",
        "pass": True,
        "pure_file_type_schema_hash_audit": True,
        "spatial_judgment": False,
        "temporal_judgment": False,
        "geometry_distance_judgment": False,
        "pose_contact_trajectory_judgment": False,
        "new_rrd_rbl_png": [0, 0, 0],
        "reason": (
            "Typed metadata and list-edit identity are authoritative in canonical JSON/hash; "
            "a viewer cannot strengthen this non-spatial/non-temporal decision. D346 live "
            "geometry restores the full D341 Rerun lifecycle."
        ),
    }


def _registered_command() -> dict[str, Any]:
    return {
        "python": str(REGISTERED_PYTHON),
        "script": _relative(Path(__file__).resolve()),
        "argv": ["--stage", "run"],
        "environment": {
            "PYTHONPATH": REGISTERED_PYTHONPATH,
            "LD_LIBRARY_PATH": REGISTERED_LD_LIBRARY_PATH,
        },
    }


def _static_mutation_audit(tree: ast.AST) -> dict[str, Any]:
    forbidden_attribute_names = {
        "CopySpec",
        "CreateNew",
        "Export",
        "Save",
        "hardlink_to",
        "rename",
        "replace",
        "symlink_to",
        "touch",
        "unlink",
        "write_bytes",
    }
    forbidden_qualified_names = {
        "os.remove",
        "os.rename",
        "os.replace",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
    }

    def qualified_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = qualified_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    forbidden_calls = []
    unexpected_text_writers = []
    write_mode_opens = []
    subprocess_shell_true = []
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    guarded_write_helpers = set()
    for function in functions:
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            name = qualified_name(node.func)
            attribute = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if function.name in {"_write_json", "_write_text"} and name == "_assert_output_path":
                guarded_write_helpers.add(function.name)
            if attribute in forbidden_attribute_names or name in forbidden_qualified_names:
                forbidden_calls.append(
                    {"function": function.name, "call": name, "line": node.lineno}
                )
            if attribute == "write_text" and function.name not in {
                "_write_json",
                "_write_text",
            }:
                unexpected_text_writers.append(
                    {"function": function.name, "call": name, "line": node.lineno}
                )
            if name == "open" or attribute == "open":
                mode_node = None
                if attribute == "open" and node.args:
                    mode_node = node.args[0]
                elif name == "open" and len(node.args) >= 2:
                    mode_node = node.args[1]
                for keyword in node.keywords:
                    if keyword.arg == "mode":
                        mode_node = keyword.value
                if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
                    if any(marker in mode_node.value for marker in "wax+"):
                        write_mode_opens.append(
                            {
                                "function": function.name,
                                "call": name,
                                "mode": mode_node.value,
                                "line": node.lineno,
                            }
                        )
            if name == "subprocess.run":
                for keyword in node.keywords:
                    if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
                        if keyword.value.value is True:
                            subprocess_shell_true.append(
                                {"function": function.name, "line": node.lineno}
                            )

    checks = {
        "forbidden_asset_mutation_calls_absent": not forbidden_calls,
        "write_text_only_in_guarded_helpers": not unexpected_text_writers,
        "both_write_helpers_call_output_guard": guarded_write_helpers
        == {"_write_json", "_write_text"},
        "direct_write_mode_open_absent": not write_mode_opens,
        "subprocess_shell_true_absent": not subprocess_shell_true,
    }
    return {
        "checks": checks,
        "forbidden_calls": forbidden_calls,
        "unexpected_text_writers": unexpected_text_writers,
        "write_mode_opens": write_mode_opens,
        "subprocess_shell_true": subprocess_shell_true,
        "pass": all(checks.values()),
    }


def _prepare() -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"D345 output already exists: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True)
    parameter_audit = _parameter_audit()
    rerun_omission = _rerun_omission()
    _write_json(PARAMETER_AUDIT_PATH, parameter_audit)
    _write_json(RERUN_OMISSION_PATH, rerun_omission)

    d339_inventory = _inventory(D339_ATTEMPT2)
    d344_attempt3_inventory = _inventory(D344_ATTEMPT3)
    d344_inventory = _inventory(D344_DIR)
    core = _json(D344_CORE_MANIFEST)
    allowed = list(core["whole_physics_semantic_allowlist"]["allowed_property_paths"])
    prereg = {
        "artifact": "D345_PREREGISTRATION_V1",
        "status": "PRE_RUNTIME_LOCKED",
        "new_variables": NEW_VARIABLES,
        "variable_count": 1,
        "git_head": _git_head(),
        "registered_command": _registered_command(),
        "source_hashes": _source_hashes(),
        "parameter_audit_sha256": _sha256(PARAMETER_AUDIT_PATH),
        "rerun_omission_sha256": _sha256(RERUN_OMISSION_PATH),
        "d339_attempt2_file_count": len(d339_inventory),
        "d339_attempt2_inventory_digest": _inventory_digest(d339_inventory),
        "d344_attempt3_file_count": len(d344_attempt3_inventory),
        "d344_attempt3_inventory_digest": _inventory_digest(d344_attempt3_inventory),
        "d344_output_file_count": len(d344_inventory),
        "d344_output_inventory_digest": _inventory_digest(d344_inventory),
        "d344_diagnosis_hashes": {
            "first": _sha256(D344_DIAGNOSIS_A),
            "repeat": _sha256(D344_DIAGNOSIS_B),
            "root_cause": _sha256(D344_ROOT_CAUSE),
        },
        "allowed_property_paths": allowed,
        "allowed_property_count": len(allowed),
        "allowed_property_paths_sha256": _allowed_paths_sha256(allowed),
        "expected_composed_prim_count": EXPECTED_COMPOSED_PRIM_COUNT,
        "expected_composed_api_schema_count": EXPECTED_COMPOSED_API_SCHEMA_COUNT,
        "expected_direct_api_schema_count": EXPECTED_DIRECT_API_SCHEMA_COUNT,
        "representative_part_path": REPRESENTATIVE_PART_PATH,
        "representative_authored_tokens": REPRESENTATIVE_AUTHORED_TOKENS,
        "representative_core_applied_tokens": REPRESENTATIVE_CORE_APPLIED_TOKENS,
        "worker_count": 2,
        "worker_ids": ["a", "b"],
        "old_repr_is_scientific_authority": False,
        "unknown_type_policy": "FAIL_STOP_NO_REPR_OR_STR_FALLBACK",
        "mask_policy": "VALUE_ONLY_EXACT_39_NO_WILDCARD",
        "numpy_version": importlib.metadata.version("numpy"),
        "psutil_version": importlib.metadata.version("psutil"),
        "pxr_usd_version": list(Usd.GetVersion()),
        "unrelated_dirty_snapshot": _unrelated_dirty_snapshot(),
    }
    _write_json(PREREG_PATH, prereg)
    print(_dumps({"prepared": True, "output": _relative(OUT_DIR)}, pretty=True))
    return 0


def _preflight() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    expected_files = {
        PREREG_PATH.name,
        PARAMETER_AUDIT_PATH.name,
        RERUN_OMISSION_PATH.name,
    }
    observed = {path.name for path in OUT_DIR.iterdir()}
    if observed != expected_files:
        raise RuntimeError(f"D345 output folder is not prereg-only: {sorted(observed)}")
    prereg = _json(PREREG_PATH)
    parameter_audit = _json(PARAMETER_AUDIT_PATH)
    omission = _json(RERUN_OMISSION_PATH)
    d339_before = _inventory(D339_ATTEMPT2)
    d344_attempt3_before = _inventory(D344_ATTEMPT3)
    d344_before = _inventory(D344_DIR)
    unrelated_before = _unrelated_dirty_snapshot()
    core = _json(D344_CORE_MANIFEST)
    outer = _json(D344_OUTER_MANIFEST)
    build_summary = _json(D344_BUILD_SUMMARY)
    root_cause = _json(D344_ROOT_CAUSE)
    allowed = list(core["whole_physics_semantic_allowlist"]["allowed_property_paths"])

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    static_mutation_audit = _static_mutation_audit(tree)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    checks = {
        "prereg_artifact": prereg.get("artifact") == "D345_PREREGISTRATION_V1",
        "prereg_status": prereg.get("status") == "PRE_RUNTIME_LOCKED",
        "new_variables": prereg.get("new_variables") == NEW_VARIABLES,
        "variable_count": prereg.get("variable_count") == 1,
        "git_head": prereg.get("git_head") == _git_head(),
        "registered_command": bool(
            prereg.get("registered_command") == _registered_command()
            and Path(sys.executable).resolve() == REGISTERED_PYTHON.resolve()
            and os.environ.get("PYTHONPATH") == REGISTERED_PYTHONPATH
            and os.environ.get("LD_LIBRARY_PATH") == REGISTERED_LD_LIBRARY_PATH
        ),
        "source_hashes": prereg.get("source_hashes") == _source_hashes(),
        "parameter_audit_hash": prereg.get("parameter_audit_sha256")
        == _sha256(PARAMETER_AUDIT_PATH),
        "rerun_omission_hash": prereg.get("rerun_omission_sha256")
        == _sha256(RERUN_OMISSION_PATH),
        "parameter_audit": parameter_audit == _parameter_audit(),
        "rerun_omission": omission == _rerun_omission(),
        "d339_inventory": bool(
            len(d339_before)
            == prereg.get("d339_attempt2_file_count")
            == EXPECTED_D339_FILE_COUNT
            and _inventory_digest(d339_before)
            == prereg.get("d339_attempt2_inventory_digest")
            == EXPECTED_D339_DIGEST
        ),
        "d344_attempt3_inventory": bool(
            len(d344_attempt3_before)
            == prereg.get("d344_attempt3_file_count")
            == EXPECTED_D344_ATTEMPT3_FILE_COUNT
            and _inventory_digest(d344_attempt3_before)
            == prereg.get("d344_attempt3_inventory_digest")
            == EXPECTED_D344_ATTEMPT3_DIGEST
        ),
        "d344_output_inventory": bool(
            len(d344_before) == prereg.get("d344_output_file_count")
            and _inventory_digest(d344_before)
            == prereg.get("d344_output_inventory_digest")
        ),
        "d344_diagnosis_hashes": prereg.get("d344_diagnosis_hashes")
        == {
            "first": _sha256(D344_DIAGNOSIS_A),
            "repeat": _sha256(D344_DIAGNOSIS_B),
            "root_cause": _sha256(D344_ROOT_CAUSE),
        },
        "allowed_count_exact_39": prereg.get("allowed_property_count")
        == len(allowed)
        == len(EXPECTED_ALLOWED_PATHS)
        == 39,
        "allowed_paths_literal_exact": prereg.get("allowed_property_paths")
        == allowed
        == list(EXPECTED_ALLOWED_PATHS),
        "allowed_paths_hash_exact": prereg.get("allowed_property_paths_sha256")
        == _allowed_paths_sha256(allowed)
        == EXPECTED_ALLOWED_PATHS_SHA256,
        "manifest_changed_parts_exact": outer.get("changed_parts")
        == {body: list(CHANGED_PARTS[body]) for body in ("link5", "gripper_link")},
        "manifest_parameters_unchanged": bool(
            outer.get("parameters_increased") == []
            and outer.get("parameters_changed") == []
            and outer.get("thresholds_relaxed") == []
        ),
        "d344_verdict_preserved": build_summary.get("verdict")
        == root_cause.get("registered_verdict_retained")
        == "D344_G0A_ATTEMPT3_AUTHORING_CONTRACT_FAIL_STOP",
        "d344_not_reclassified": root_cause.get("historical_reclassification_allowed")
        is False,
        "d344_no_live_validation": root_cause.get("fresh_validation_run") is False,
        "expected_counts": bool(
            prereg.get("expected_composed_prim_count") == EXPECTED_COMPOSED_PRIM_COUNT
            and prereg.get("expected_composed_api_schema_count")
            == EXPECTED_COMPOSED_API_SCHEMA_COUNT
            and prereg.get("expected_direct_api_schema_count")
            == EXPECTED_DIRECT_API_SCHEMA_COUNT
        ),
        "representative_tokens": bool(
            prereg.get("representative_part_path") == REPRESENTATIVE_PART_PATH
            and prereg.get("representative_authored_tokens")
            == REPRESENTATIVE_AUTHORED_TOKENS
            and prereg.get("representative_core_applied_tokens")
            == REPRESENTATIVE_CORE_APPLIED_TOKENS
        ),
        "worker_contract": prereg.get("worker_count") == 2
        and prereg.get("worker_ids") == ["a", "b"],
        "unknown_type_policy": prereg.get("unknown_type_policy")
        == "FAIL_STOP_NO_REPR_OR_STR_FALLBACK",
        "mask_policy": prereg.get("mask_policy")
        == "VALUE_ONLY_EXACT_39_NO_WILDCARD",
        "package_pins": bool(
            prereg.get("numpy_version") == importlib.metadata.version("numpy") == "1.26.0"
            and prereg.get("psutil_version") == importlib.metadata.version("psutil")
            == "5.9.8"
            and prereg.get("pxr_usd_version") == list(Usd.GetVersion())
        ),
        "forbidden_runtime_imports_absent": imported_roots.isdisjoint(
            {"isaaclab", "isaacsim", "omni", "rerun"}
        ),
        "static_mutation_audit": static_mutation_audit["pass"],
        "unrelated_dirty_baseline_exact": prereg.get("unrelated_dirty_snapshot")
        == unrelated_before,
    }
    if not all(checks.values()):
        raise RuntimeError(f"D345 preregistration gate failed: {checks}")
    return (
        prereg,
        d339_before,
        d344_attempt3_before,
        d344_before,
        unrelated_before,
        {
            "checks": checks,
            "static_mutation_audit": static_mutation_audit,
            "pass": True,
        },
    )


def _run_worker(worker_id: str) -> dict[str, Any]:
    command = [
        str(REGISTERED_PYTHON),
        str(Path(__file__).resolve()),
        "--stage",
        "worker",
        "--worker-id",
        worker_id,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = REGISTERED_PYTHONPATH
    environment["LD_LIBRARY_PATH"] = REGISTERED_LD_LIBRARY_PATH
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker {worker_id} failed rc={completed.returncode}: {completed.stderr[-4000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"worker {worker_id} emitted unexpected stdout lines: {len(lines)}")
    payload = json.loads(lines[0])
    payload["process_stderr"] = {
        "bytes": len(completed.stderr.encode("utf-8")),
        "line_count": len(completed.stderr.splitlines()),
        "sha256": _sha256_text(completed.stderr),
        "empty": completed.stderr == "",
    }
    return payload


def _immutability(
    d339_before: list[dict[str, Any]],
    d344_attempt3_before: list[dict[str, Any]],
    d344_before: list[dict[str, Any]],
    unrelated_before: dict[str, Any],
    expected_source_hashes: dict[str, str],
) -> dict[str, Any]:
    d339_after = _inventory(D339_ATTEMPT2)
    d344_attempt3_after = _inventory(D344_ATTEMPT3)
    d344_after = _inventory(D344_DIR)
    unrelated_after = _unrelated_dirty_snapshot()
    checks = {
        "d339_attempt2_exact": d339_before == d339_after,
        "d344_attempt3_exact": d344_attempt3_before == d344_attempt3_after,
        "d344_output_exact": d344_before == d344_after,
        "unrelated_dirty_snapshot_exact": unrelated_before == unrelated_after,
        "sealed_source_hashes_exact": expected_source_hashes == _source_hashes(),
    }
    return {
        "checks": checks,
        "d339_count_before_after": [len(d339_before), len(d339_after)],
        "d339_digest_before_after": [
            _inventory_digest(d339_before),
            _inventory_digest(d339_after),
        ],
        "d344_attempt3_count_before_after": [
            len(d344_attempt3_before),
            len(d344_attempt3_after),
        ],
        "d344_attempt3_digest_before_after": [
            _inventory_digest(d344_attempt3_before),
            _inventory_digest(d344_attempt3_after),
        ],
        "d344_output_count_before_after": [len(d344_before), len(d344_after)],
        "d344_output_digest_before_after": [
            _inventory_digest(d344_before),
            _inventory_digest(d344_after),
        ],
        "unrelated_dirty_digest_before_after": [
            unrelated_before["digest"],
            unrelated_after["digest"],
        ],
        "pass": all(checks.values()),
    }


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# D345 주소 없는 USD 메타데이터 비교기 결과",
            "",
            f"- 판정: `{summary['verdict']}`",
            f"- 독립 worker: `{summary['worker_count']}`",
            f"- 합성 장면 항목: `{summary['composed_prim_count']}`",
            f"- 직접 apiSchemas 목록 연산: `{summary['direct_api_schema_count']}`",
            f"- 합성 apiSchemas 목록: `{summary['composed_api_schema_count']}`",
            f"- 가린 형상 값: `{summary['masked_geometry_property_count']}`",
            f"- 시간에 따라 변하는 속성값: `{summary['time_sample_count']}`",
            f"- 주소 없는 네 방향 해시 동일: `{summary['four_way_canonical_hash_exact']}`",
            f"- 옛 주소 포함 표현 거부: `{summary['old_repr_nondeterminism_rejected']}`",
            f"- 토큰 삭제/연산 모드 반례 거부: `{summary['token_list_op_negative_controls_pass']}`",
            f"- D339/D344 불변: `{summary['immutability_pass']}`",
            "- 자산 작성, Isaac, Rerun, 물리 진행: 모두 0.",
            "- D344의 과거 FAIL과 g0a_pass=false는 유지한다.",
            "",
        ]
    )


def _run() -> int:
    (
        prereg,
        d339_before,
        d344_attempt3_before,
        d344_before,
        unrelated_before,
        preflight_evidence,
    ) = _preflight()

    worker_a = _run_worker("a")
    _write_json(WORKER_A_PATH, worker_a)
    worker_b = _run_worker("b")
    _write_json(WORKER_B_PATH, worker_b)

    immutability = _immutability(
        d339_before,
        d344_attempt3_before,
        d344_before,
        unrelated_before,
        prereg["source_hashes"],
    )
    checks = {
        "worker_a_pass": worker_a.get("pass") is True,
        "worker_b_pass": worker_b.get("pass") is True,
        "worker_ids_exact": [worker_a.get("worker_id"), worker_b.get("worker_id")]
        == ["a", "b"],
        "worker_pids_distinct": worker_a.get("pid") != worker_b.get("pid"),
        "worker_nonces_distinct": worker_a.get("nonce") != worker_b.get("nonce"),
        "worker_python_exact": worker_a.get("python_executable")
        == worker_b.get("python_executable")
        == str(REGISTERED_PYTHON.resolve()),
        "worker_pxr_version_exact": worker_a.get("pxr_usd_version")
        == worker_b.get("pxr_usd_version")
        == prereg.get("pxr_usd_version"),
        "worker_stderr_empty": worker_a["process_stderr"]["empty"] is True
        and worker_b["process_stderr"]["empty"] is True,
        "source_hash_cross_process_exact": worker_a["source_summary"]["canonical_sha256"]
        == worker_b["source_summary"]["canonical_sha256"],
        "variant_hash_cross_process_exact": worker_a["variant_summary"]["canonical_sha256"]
        == worker_b["variant_summary"]["canonical_sha256"],
        "source_variant_hash_four_way_exact": len(
            {
                worker_a["source_summary"]["canonical_sha256"],
                worker_a["variant_summary"]["canonical_sha256"],
                worker_b["source_summary"]["canonical_sha256"],
                worker_b["variant_summary"]["canonical_sha256"],
            }
        )
        == 1,
        "source_row_hashes_cross_process_exact": worker_a["source_summary"][
            "row_sha256"
        ]
        == worker_b["source_summary"]["row_sha256"],
        "variant_row_hashes_cross_process_exact": worker_a["variant_summary"][
            "row_sha256"
        ]
        == worker_b["variant_summary"]["row_sha256"],
        "direct_api_rows_four_way_exact": worker_a["source_api_schemas"]["direct_rows"]
        == worker_a["variant_api_schemas"]["direct_rows"]
        == worker_b["source_api_schemas"]["direct_rows"]
        == worker_b["variant_api_schemas"]["direct_rows"],
        "composed_api_rows_four_way_exact": worker_a["source_api_schemas"][
            "composed_rows"
        ]
        == worker_a["variant_api_schemas"]["composed_rows"]
        == worker_b["source_api_schemas"]["composed_rows"]
        == worker_b["variant_api_schemas"]["composed_rows"],
        "old_repr_source_hash_changes_between_processes": worker_a[
            "source_legacy_repr_diagnostic"
        ]["address_bearing_sha256"]
        != worker_b["source_legacy_repr_diagnostic"]["address_bearing_sha256"],
        "old_repr_variant_hash_changes_between_processes": worker_a[
            "variant_legacy_repr_diagnostic"
        ]["address_bearing_sha256"]
        != worker_b["variant_legacy_repr_diagnostic"]["address_bearing_sha256"],
        "token_list_op_controls_both_pass": worker_a["token_list_op_controls"]["pass"]
        is True
        and worker_b["token_list_op_controls"]["pass"] is True,
        "unsupported_types_zero": worker_a["source_summary"]["unsupported_type_count"]
        == worker_a["variant_summary"]["unsupported_type_count"]
        == worker_b["source_summary"]["unsupported_type_count"]
        == worker_b["variant_summary"]["unsupported_type_count"]
        == 0,
        "canonical_address_leaks_zero": worker_a["source_summary"][
            "runtime_address_pattern_count"
        ]
        == worker_a["variant_summary"]["runtime_address_pattern_count"]
        == worker_b["source_summary"]["runtime_address_pattern_count"]
        == worker_b["variant_summary"]["runtime_address_pattern_count"]
        == 0,
        "immutability": immutability["pass"],
        "no_rerun_artifacts": (
            not any(OUT_DIR.rglob("*.rrd"))
            and not any(OUT_DIR.rglob("*.rbl"))
            and not any(OUT_DIR.rglob("*.png"))
        ),
        "no_authored_asset_artifacts": not any(
            path.suffix.lower() in {".usd", ".usda", ".usdc", ".stl", ".obj", ".ply"}
            for path in OUT_DIR.rglob("*")
            if path.is_file()
        ),
        "no_collision_asset_output": not (OUT_DIR / "collision_asset").exists(),
    }
    passed = all(checks.values())
    verdict = VERDICT_PASS if passed else VERDICT_FAIL
    evidence = {
        "artifact": "D345_DETERMINISTIC_USD_METADATA_EVIDENCE_V1",
        "verdict": verdict,
        "pass": passed,
        "new_variables": NEW_VARIABLES,
        "worker_files": [_relative(WORKER_A_PATH), _relative(WORKER_B_PATH)],
        "worker_file_sha256": [_sha256(WORKER_A_PATH), _sha256(WORKER_B_PATH)],
        "checks": checks,
        "preflight": preflight_evidence,
        "immutability": immutability,
        "canonical_sha256": worker_a["source_summary"]["canonical_sha256"],
        "canonical_json_bytes": worker_a["source_summary"]["canonical_json_bytes"],
        "composed_prim_count": worker_a["source_summary"]["row_count"],
        "direct_api_schema_count": worker_a["source_api_schemas"]["direct_row_count"],
        "composed_api_schema_count": worker_a["source_api_schemas"][
            "composed_row_count"
        ],
        "masked_geometry_property_count": len(worker_a["allowed_property_paths"]),
        "time_sample_count": worker_a["source_summary"]["time_sample_count"],
        "runtime_environment_created": False,
        "controlled_physics_steps": 0,
        "asset_writes": 0,
        "asset_copies": 0,
        "recooks": 0,
        "rerun_artifacts": 0,
        "g0a_pass": False,
        "d344_verdict_reclassified": False,
    }
    _write_json(EVIDENCE_PATH, evidence)
    summary = {
        "artifact": "D345_DETERMINISTIC_USD_METADATA_SUMMARY_V1",
        "verdict": verdict,
        "pass": passed,
        "new_variables": NEW_VARIABLES,
        "worker_count": 2,
        "composed_prim_count": evidence["composed_prim_count"],
        "direct_api_schema_count": evidence["direct_api_schema_count"],
        "composed_api_schema_count": evidence["composed_api_schema_count"],
        "masked_geometry_property_count": evidence["masked_geometry_property_count"],
        "time_sample_count": evidence["time_sample_count"],
        "canonical_sha256": evidence["canonical_sha256"],
        "canonical_json_bytes": evidence["canonical_json_bytes"],
        "four_way_canonical_hash_exact": checks[
            "source_variant_hash_four_way_exact"
        ],
        "old_repr_nondeterminism_rejected": bool(
            checks["old_repr_source_hash_changes_between_processes"]
            and checks["old_repr_variant_hash_changes_between_processes"]
        ),
        "token_list_op_negative_controls_pass": checks[
            "token_list_op_controls_both_pass"
        ],
        "unsupported_type_count": 0,
        "canonical_runtime_address_count": 0,
        "immutability_pass": immutability["pass"],
        "existing_parameter_increases": 0,
        "existing_parameter_changes": 0,
        "threshold_relaxations": 0,
        "decomposition_changes": 0,
        "runtime_environment_created": False,
        "controlled_physics_steps": 0,
        "rerun_omitted_under_registered_exception": True,
        "g0a_pass": False,
        "d344_verdict_retained": "D344_G0A_ATTEMPT3_AUTHORING_CONTRACT_FAIL_STOP",
        "evidence": _relative(EVIDENCE_PATH),
    }
    _write_json(SUMMARY_PATH, summary)
    _write_text(REPORT_PATH, _report(summary))
    print(_dumps(summary, pretty=True))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "run", "worker"), required=True)
    parser.add_argument("--worker-id", choices=("a", "b"))
    args = parser.parse_args()
    if args.stage == "prepare":
        return _prepare()
    if args.stage == "worker":
        if args.worker_id is None:
            raise RuntimeError("worker stage requires --worker-id")
        print(_dumps(_worker_payload(args.worker_id)))
        return 0
    return _run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        failure = {
            "artifact": "D345_FAILURE_STOP_V1",
            "verdict": VERDICT_FAIL,
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_environment_created": False,
            "controlled_physics_steps": 0,
            "g0a_pass": False,
        }
        if OUT_DIR.exists() and not SUMMARY_PATH.exists():
            try:
                _write_json(SUMMARY_PATH, failure)
            except Exception:
                pass
        print(_dumps(failure, pretty=True), file=sys.stderr)
        raise SystemExit(2)
