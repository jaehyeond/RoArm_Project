#!/usr/bin/env python3
"""Read-only diagnosis of the D344 composed-inventory false/real mismatch.

This script never writes a USD layer.  It reproduces the D340 composed-row
encoding exactly, masks the 39 pre-registered geometry values, and reports the
first remaining semantic difference without printing large mesh arrays.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from pxr import Usd


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d344"
CORE = OUT / "collision_asset/attempt3/d340_attempt3_asset_manifest.json"
SOURCE = (
    ROOT
    / "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2"
    / "roarm_m3_fullmesh_convex_parts/configuration/roarm_m3_physics.usd"
)
VARIANT = (
    OUT
    / "collision_asset/attempt3/roarm_m3_fullmesh_fixed_point_parts"
    / "configuration/roarm_m3_physics.usd"
)
RESULT = OUT / "d344_postrun_semantic_diagnosis.json"
REPEAT_RESULT = OUT / "d344_postrun_semantic_diagnosis_repeat.json"
RUNTIME_OBJECT_RE = re.compile(
    r"^<(?P<type>pxr\.[A-Za-z0-9_.]+) object at 0x[0-9a-fA-F]+>$"
)


def inventory_rows(path: Path, allowed: set[str]) -> list[dict[str, Any]]:
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open {path}")
    rows: list[dict[str, Any]] = []
    for prim in Usd.PrimRange.Stage(stage):
        attributes = []
        for attr in prim.GetAttributes():
            property_path = f"{prim.GetPath().pathString}.{attr.GetName()}"
            value = (
                "<D340_REGISTERED_GEOMETRY_VALUE>"
                if property_path in allowed
                else repr(attr.Get())
            )
            attributes.append(
                (
                    attr.GetName(),
                    str(attr.GetTypeName()),
                    value,
                    tuple(str(item) for item in attr.GetConnections()),
                    tuple(
                        sorted(
                            (key, repr(attr.GetMetadata(key)))
                            for key in attr.GetAllMetadata()
                        )
                    ),
                )
            )
        relationships = [
            (
                rel.GetName(),
                tuple(str(item) for item in rel.GetTargets()),
                tuple(
                    sorted(
                        (key, repr(rel.GetMetadata(key)))
                        for key in rel.GetAllMetadata()
                    )
                ),
            )
            for rel in prim.GetRelationships()
        ]
        rows.append(
            {
                "path": prim.GetPath().pathString,
                "type_name": str(prim.GetTypeName()),
                "active": bool(prim.IsActive()),
                "instanceable": bool(prim.IsInstanceable()),
                "applied_schemas": sorted(str(item) for item in prim.GetAppliedSchemas()),
                "metadata": sorted(
                    (key, repr(value)) for key, value in prim.GetAllMetadata().items()
                ),
                "attributes": sorted(attributes),
                "relationships": sorted(relationships),
            }
        )
    return rows


def compact(value: Any) -> dict[str, Any]:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "json_length": len(encoded),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "preview": encoded[:240],
    }


def normalize_runtime_object_addresses(value: Any) -> Any:
    """Remove only CPython object addresses, never authored scalar values."""
    if isinstance(value, str):
        match = RUNTIME_OBJECT_RE.fullmatch(value)
        if match:
            return f"<{match.group('type')} object at <RUNTIME_ADDRESS>>"
        return value
    if isinstance(value, list):
        return [normalize_runtime_object_addresses(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_runtime_object_addresses(item) for item in value)
    if isinstance(value, dict):
        return {
            key: normalize_runtime_object_addresses(item) for key, item in value.items()
        }
    return value


def main() -> None:
    manifest = json.loads(CORE.read_text())
    allowed = set(
        manifest["whole_physics_semantic_allowlist"]["allowed_property_paths"]
    )
    source_rows = inventory_rows(SOURCE, allowed)
    variant_rows = inventory_rows(VARIANT, allowed)

    raw_different_rows = []
    different_field_counts: dict[str, int] = {}
    metadata_key_counts: dict[str, int] = {}
    address_only_row_count = 0
    non_address_differences = []
    for index, (source, variant) in enumerate(zip(source_rows, variant_rows, strict=True)):
        if source == variant:
            continue
        raw_different_rows.append(index)
        fields = []
        for field in source:
            if source[field] != variant[field]:
                fields.append(field)
                different_field_counts[field] = different_field_counts.get(field, 0) + 1
        source_metadata = dict(source["metadata"])
        variant_metadata = dict(variant["metadata"])
        for key in sorted(set(source_metadata) | set(variant_metadata)):
            if source_metadata.get(key) != variant_metadata.get(key):
                metadata_key_counts[key] = metadata_key_counts.get(key, 0) + 1
        if normalize_runtime_object_addresses(source) == normalize_runtime_object_addresses(variant):
            address_only_row_count += 1
        else:
            non_address_differences.append(
                {
                    "row_index": index,
                    "source_path": source["path"],
                    "variant_path": variant["path"],
                    "different_fields": sorted(fields),
                    "source": compact(source),
                    "variant": compact(variant),
                }
            )

    source_payload = json.dumps(
        source_rows, sort_keys=True, separators=(",", ":"), default=str
    )
    variant_payload = json.dumps(
        variant_rows, sort_keys=True, separators=(",", ":"), default=str
    )
    normalized_source_rows = normalize_runtime_object_addresses(source_rows)
    normalized_variant_rows = normalize_runtime_object_addresses(variant_rows)
    normalized_source_payload = json.dumps(
        normalized_source_rows, sort_keys=True, separators=(",", ":"), default=str
    )
    normalized_variant_payload = json.dumps(
        normalized_variant_rows, sort_keys=True, separators=(",", ":"), default=str
    )
    result = {
        "artifact": "D344_POSTRUN_COMPOSED_SEMANTIC_DIAGNOSIS",
        "read_only": True,
        "source": str(SOURCE.relative_to(ROOT)),
        "variant": str(VARIANT.relative_to(ROOT)),
        "allowed_property_count": len(allowed),
        "source_row_count": len(source_rows),
        "variant_row_count": len(variant_rows),
        "source_sha256": hashlib.sha256(source_payload.encode("utf-8")).hexdigest(),
        "variant_sha256": hashlib.sha256(variant_payload.encode("utf-8")).hexdigest(),
        "raw_different_row_count": len(raw_different_rows),
        "raw_different_field_counts": different_field_counts,
        "raw_metadata_difference_key_counts": metadata_key_counts,
        "runtime_address_only_row_count": address_only_row_count,
        "non_runtime_address_difference_count": len(non_address_differences),
        "non_runtime_address_differences": non_address_differences[:20],
        "rows_equal_after_masking": source_rows == variant_rows,
        "rows_equal_after_masking_and_runtime_address_normalization": (
            normalized_source_rows == normalized_variant_rows
        ),
        "normalized_source_sha256": hashlib.sha256(
            normalized_source_payload.encode("utf-8")
        ).hexdigest(),
        "normalized_variant_sha256": hashlib.sha256(
            normalized_variant_payload.encode("utf-8")
        ).hexdigest(),
        "diagnosis": (
            "COMPARATOR_RUNTIME_OBJECT_ADDRESS_FALSE_DIFFERENCE"
            if normalized_source_rows == normalized_variant_rows
            else "REAL_NON_ALLOWLIST_SEMANTIC_DIFFERENCE_REMAINS"
        ),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if "--write" in sys.argv[1:]:
        RESULT.write_text(payload)
    if "--write-repeat" in sys.argv[1:]:
        REPEAT_RESULT.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
