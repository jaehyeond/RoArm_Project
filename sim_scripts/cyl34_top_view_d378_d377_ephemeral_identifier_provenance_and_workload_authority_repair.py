#!/usr/bin/env python3
"""D378 offline repair of the D377 termination-workload comparator.

This script reads immutable D375/D377 JSON and callback witnesses only.  It
never imports or launches Isaac, Kit, PhysX, Warp, Hydra, Fabric, or pxr.  It
does not create or modify USD, colliders, cylinders, robot poses, q5 commands,
physics steps, contacts, targets, IK, or paths.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

CASE_ROOT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d378"
ATTEMPT1_DIR = (
    CASE_ROOT
    / "attempt1_ephemeral_identifier_provenance_and_workload_authority_repair"
)
OUT_DIR = (
    CASE_ROOT
    / "attempt2_preregistration_status_order_repair"
)
PREREG_PATH = OUT_DIR / "d378_preregistration.json"
PHASE_PATH = OUT_DIR / "d378_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d378_offline_invocation.json"
EVIDENCE_PATH = OUT_DIR / "d378_workload_authority_repair_evidence.json"
BOARD_PATH = OUT_DIR / "d378_corrected_workload_authority_1920x1080.png"
RRD_PATH = OUT_DIR / "d378_workload_authority_repair.rrd"
RBL_PATH = OUT_DIR / "d378_workload_authority_repair.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d378_rerun_validation.json"
RERUN_PNG_PATH = OUT_DIR / "d378_rerun_inspection.png"
AUTOMATED_PATH = OUT_DIR / "d378_automated_summary.json"
MANUAL_TEMPLATE_PATH = OUT_DIR / "d378_manual_visual_inspection_template.json"
MANUAL_PATH = OUT_DIR / "d378_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d378_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d378_runtime_exception.json"

HARNESS = Path(__file__).resolve()
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
START_HERE = REPO / "START_HERE.md"

D375_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d375/"
    "attempt2_external_gpu_attestation_repair"
)
D375_RAW = D375_DIR / "d375_worker_raw_summary.json"
D375_PRECLOSE = D375_DIR / "d375_worker_preclose_sentinel.json"
D375_SUPERVISOR = D375_DIR / "d375_worker_supervisor.json"
D375_PHASES = D375_DIR / "d375_phase_markers.jsonl"

D377_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d377/"
    "attempt1_stagecache_erase_before_close_localization"
)
D377_RAW = D377_DIR / "d377_worker_raw_summary.json"
D377_PRECLOSE = D377_DIR / "d377_worker_preclose_sentinel.json"
D377_SUPERVISOR = D377_DIR / "d377_worker_supervisor.json"
D377_PHASES = D377_DIR / "d377_phase_markers.jsonl"
D377_EVIDENCE = D377_DIR / "d377_stagecache_erase_localization_evidence.json"
D377_COMPLETION = D377_DIR / "d377_completion_summary.json"

D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
EXPECTED_HEAD = "2acb5b99567946d343e95e61087357193da0826c"

NEW_VARIABLES = [
    "ephemeral_identifier_exclusion_and_normalized_witness_authority_v1",
    "ascii_only_corrected_observability_projection_v1",
]
VERDICT_PASS = "D378_EPHEMERAL_IDENTIFIER_PROVENANCE_AND_WORKLOAD_AUTHORITY_REPAIR_PASS"
VERDICT_FAIL = "D378_WORKLOAD_AUTHORITY_REPAIR_OR_OBSERVABILITY_FAIL_STOP"
D377_FROZEN_VERDICT = "D377_STAGECACHE_ERASE_BEFORE_CLOSE_LOCALIZATION_FAIL_STOP"

EXPECTED_RAW_SHA = {
    D375_RAW: "74f959b765860d06ca1d892823d47dc395cad3aea92d0250e21ff706263fc21e",
    D377_RAW: "f14d2cf38cffc03a3121719a4dac0a62d612b46926a5ff6afcc10cd143717fb1",
    D375_SUPERVISOR: "69f5f8ec5760e7804f3d076c377fc0ea597bde902f3d8ec7d941f36208f4f51c",
    D377_SUPERVISOR: "1c0a4754da7fa0bae748e6c1095a1c39982ab50cef6202c6918f742fb635ce49",
    D377_EVIDENCE: "556db509206fd99507b68f0ce6d686ba3dbb15708309e475e4344107da0777b2",
    D377_COMPLETION: "762d13ad790231a9bd810d37f29ee42b4d09b604e5fabe4dbae1b3ebb6f5c5dc",
}
EXPECTED_V1_DIGESTS = {
    "D375": "ec930163ac2a9cdbf7342630dccd34d5467fa3618dfd0d6213066fbaa12b0b7b",
    "D377": "758504733115b8740a972fe99ea63f9303d5759505d03a29e1e9c9570fa13c81",
}
EXPECTED_CORRECTED_WORKLOAD_SHA = (
    "28aadb5ff26270039df58f7cd06080bf7afcdec001402e886a6edf1483fdfe31"
)
EXPECTED_NORMALIZED_WITNESS_SHA = (
    "0a56d7900470f6f75d5f63ac415d7d0f4cca5c5d941951280387ae2378abfe8c"
)
EXPECTED_NORMALIZED_PROPERTY_SHA = (
    "4710c18232e2d2259c569d01b6326bbea20b36507e5aeb9a85fbe15ca94f7c1f"
)
EXPECTED_CALLBACK_COUNTS = {
    "callbacks": 34,
    "vertices": 314,
    "indices": 1016,
    "original_polygons": 262,
}
EXPECTED_EXCLUSION_COUNTS = {
    "selected_callback_witness_sha256": 34,
    "selected_prototype_path_diagnostic": 34,
    "normalized_witness_request_return_repr": 34,
    "property_path_id": 38,
    "property_elapsed_s": 2,
}
EXPECTED_PROPERTY_EXCLUSION_PATHS = sorted(
    [
        "/gripper_link/elapsed_s",
        "/gripper_link/rigid_body/path_id",
        *[
            f"/gripper_link/colliders/{index}/path_id"
            for index in range(19)
        ],
        "/link5/elapsed_s",
        "/link5/rigid_body/path_id",
        *[f"/link5/colliders/{index}/path_id" for index in range(17)],
    ]
)
STRICT_ZERO_SCOPE = [
    "isaac_launches",
    "physx_calls",
    "usd_writes",
    "collider_regenerations",
    "automatic_decomposition_sweeps",
    "physics_steps",
    "public_forwards",
    "q5_commands",
    "q5_samples",
    "contact_queries",
    "cylinder_creates_or_writes",
    "target_ik_path_pose_changes",
    "material_mass_actuator_physics_setting_changes",
]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _phase(name: str, **fields: Any) -> None:
    ordinal = 1
    if PHASE_PATH.is_file():
        ordinal = len(
            [line for line in PHASE_PATH.read_text(encoding="utf-8").splitlines() if line]
        ) + 1
    row = {
        "ordinal": ordinal,
        "phase": name,
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _inventory(root: Path) -> dict[str, Any]:
    rows = [
        {
            "path": _rel(path),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {
        "root": _rel(root),
        "file_count": len(rows),
        "files": rows,
        "inventory_sha256": _canonical_sha(rows),
    }


def _source_hashes() -> dict[str, str]:
    return {
        "harness": _sha(HARNESS),
        "viz_debug": _sha(VIZ_DEBUG),
        "rerun_contract": _sha(RERUN_CONTRACT),
    }


def _input_hashes() -> dict[str, str]:
    paths = [
        D375_RAW,
        D375_PRECLOSE,
        D375_SUPERVISOR,
        D375_PHASES,
        D377_RAW,
        D377_PRECLOSE,
        D377_SUPERVISOR,
        D377_PHASES,
        D377_EVIDENCE,
        D377_COMPLETION,
    ]
    return {_rel(path): _sha(path) for path in paths}


def _forbidden_modules_loaded() -> list[str]:
    roots = ("omni", "isaacsim", "isaaclab", "warp", "pxr")
    return sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in roots)
    )


def _status_paths() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--short", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(record[3:] for record in output.split("\0") if record)


def _recursive_diffs(a: Any, b: Any, path: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if type(a) is not type(b):
        return [{"path": path or "/", "left": a, "right": b}]
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            child = f"{path}/{key}"
            if key not in a or key not in b:
                rows.append({"path": child, "left": a.get(key), "right": b.get(key)})
            else:
                rows.extend(_recursive_diffs(a[key], b[key], child))
    elif isinstance(a, list):
        if len(a) != len(b):
            rows.append(
                {"path": f"{path}/length", "left": len(a), "right": len(b)}
            )
        for index, (left, right) in enumerate(zip(a, b)):
            rows.extend(_recursive_diffs(left, right, f"{path}/{index}"))
    elif a != b:
        rows.append({"path": path or "/", "left": a, "right": b})
    return rows


def _without(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def _v1_workload_payload(raw: dict[str, Any]) -> dict[str, Any]:
    properties = {}
    for body, value in sorted(raw.get("property_queries", {}).items()):
        properties[body] = {
            "body": value.get("body"),
            "body_path": value.get("body_path"),
            "expected_collider_count_including_disabled_legacy": value.get(
                "expected_collider_count_including_disabled_legacy"
            ),
            "finished": value.get("finished"),
            "pass": value.get("pass"),
            "errors": value.get("errors"),
            "simulation_app_update_pumps": value.get("simulation_app_update_pumps"),
            "rigid_body": _without(value.get("rigid_body", {}), "path_id"),
            "colliders": [
                _without(row, "path_id") for row in value.get("colliders", [])
            ],
        }
    callbacks = []
    for row in raw.get("callback_rows", []):
        callback = row.get("callback", {})
        callbacks.append(
            {
                "body": row.get("body"),
                "name": row.get("name"),
                "role": row.get("role"),
                "prim_name": row.get("prim_name"),
                "live_path": row.get("live_path"),
                "authored_f32_topology_payload_sha256": row.get(
                    "authored_f32_topology_payload_sha256"
                ),
                "protocol_pass": row.get("protocol_pass"),
                "callback": _without(callback, "witness_path"),
            }
        )
    counter_keys = (
        "worker_invocations",
        "automatic_retries",
        "simulation_app_launches",
        "derivative_asset_materializations",
        "usd_stage_file_writes",
        "physx_stage_attaches",
        "physx_stage_detaches",
        "physx_property_queries",
        "physx_callback_requests",
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
        "material_mass_actuator_physics_setting_changes",
        "isaac_hydra_renders",
        "simulation_app_update_pumps",
        "d343_contract_retests",
    )
    counters = raw.get("counters", {})
    return {
        "asset_reuse": raw.get("asset_reuse"),
        "authored_readback": raw.get("authored_readback"),
        "callback_rows": callbacks,
        "canonical_outside_collision_subtree_diff": raw.get(
            "canonical_outside_collision_subtree_diff"
        ),
        "common_counters": {key: counters.get(key) for key in counter_keys},
        "live_inventory": raw.get("live_inventory"),
        "mass_api_base_vs_derivative": raw.get("mass_api_base_vs_derivative"),
        "mass_api_inspection_stage": raw.get("mass_api_inspection_stage"),
        "owner_structure": raw.get("owner_structure"),
        "physx_stage_attach_return": raw.get("physx_stage_attach_return"),
        "property_queries": properties,
        "timeline_before": raw.get("timeline_before"),
        "timeline_after": raw.get("timeline_after"),
    }


def _correct_selected_payload(
    raw: dict[str, Any],
    *,
    remove_witness_sha: bool = True,
    remove_prototype_path: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    payload = copy.deepcopy(_v1_workload_payload(raw))
    counts = {
        "selected_callback_witness_sha256": 0,
        "selected_prototype_path_diagnostic": 0,
    }
    if remove_witness_sha:
        for row in payload["callback_rows"]:
            if "witness_sha256" in row["callback"]:
                row["callback"].pop("witness_sha256")
                counts["selected_callback_witness_sha256"] += 1
    if remove_prototype_path:
        for row in payload["live_inventory"]["rows"]:
            if "prototype_path_diagnostic" in row:
                row.pop("prototype_path_diagnostic")
                counts["selected_prototype_path_diagnostic"] += 1
    return payload, counts


def _normalize_witness(value: dict[str, Any]) -> tuple[dict[str, Any], int]:
    normalized = copy.deepcopy(value)
    removed = int("request_return_repr" in normalized)
    normalized.pop("request_return_repr", None)
    return normalized, removed


def _callback_counts(witnesses: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "callbacks": len(witnesses),
        "vertices": 0,
        "indices": 0,
        "original_polygons": 0,
    }
    for witness in witnesses:
        for event in witness.get("events", []):
            for convex in event.get("convexes", []):
                counts["vertices"] += len(convex.get("vertices", []))
                counts["indices"] += len(convex.get("indices", []))
                counts["original_polygons"] += len(convex.get("polygons", []))
    return counts


def _witness_authority(raw: dict[str, Any]) -> dict[str, Any]:
    aggregate_rows = []
    normalized_values = []
    provenance_rows = []
    removed = 0
    for row in raw.get("callback_rows", []):
        witness_path = REPO / row["callback"]["witness_path"]
        witness = _read_json(witness_path)
        normalized, count = _normalize_witness(witness)
        removed += count
        normalized_values.append(normalized)
        aggregate_rows.append(
            {
                "body": row["body"],
                "name": row["name"],
                "normalized_witness_sha256": _canonical_sha(normalized),
            }
        )
        provenance_rows.append(
            {
                "body": row["body"],
                "name": row["name"],
                "path": _rel(witness_path),
                "raw_sha256": _sha(witness_path),
                "raw_sha_matches_summary": _sha(witness_path)
                == row["callback"]["witness_sha256"],
                "normalized_sha256": _canonical_sha(normalized),
            }
        )
    return {
        "aggregate_rows": aggregate_rows,
        "aggregate_sha256": _canonical_sha(aggregate_rows),
        "normalized_values": normalized_values,
        "provenance_rows": provenance_rows,
        "request_return_repr_removed": removed,
        "counts": _callback_counts(normalized_values),
    }


def _normalize_properties(
    value: Any, path: str = ""
) -> tuple[Any, list[dict[str, Any]]]:
    removed: list[dict[str, Any]] = []
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if key in {"path_id", "elapsed_s"}:
                removed.append({"path": child_path, "key": key})
            else:
                normalized, child_removed = _normalize_properties(child, child_path)
                result[key] = normalized
                removed.extend(child_removed)
        return result, removed
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            normalized, child_removed = _normalize_properties(
                child, f"{path}/{index}"
            )
            result.append(normalized)
            removed.extend(child_removed)
        return result, removed
    return value, removed


def _allowed_exclusion_manifest_valid(manifest: dict[str, Any]) -> bool:
    return manifest == {
        "selected_payload": {
            "callback_rows[*].callback.witness_sha256": 34,
            "live_inventory.rows[*].prototype_path_diagnostic": 34,
        },
        "normalized_callback_witness": {
            "request_return_repr": 34,
        },
        "normalized_property_query": {
            path: 1 for path in EXPECTED_PROPERTY_EXCLUSION_PATHS
        },
    }


def _negative_controls(
    d375_raw: dict[str, Any],
    d377_raw: dict[str, Any],
    d375_witness: dict[str, Any],
    d375_property: dict[str, Any],
) -> dict[str, Any]:
    v1_left = _v1_workload_payload(d375_raw)
    v1_right = _v1_workload_payload(d377_raw)
    witness_only_left, _ = _correct_selected_payload(
        d375_raw, remove_witness_sha=True, remove_prototype_path=False
    )
    witness_only_right, _ = _correct_selected_payload(
        d377_raw, remove_witness_sha=True, remove_prototype_path=False
    )
    prototype_only_left, _ = _correct_selected_payload(
        d375_raw, remove_witness_sha=False, remove_prototype_path=True
    )
    prototype_only_right, _ = _correct_selected_payload(
        d377_raw, remove_witness_sha=False, remove_prototype_path=True
    )
    corrected_left, _ = _correct_selected_payload(d375_raw)

    address_mutation = copy.deepcopy(d375_witness["normalized_values"][0])
    address_mutation["request_return_repr"] = "object at 0xDEADBEEF"
    normalized_address_mutation, _ = _normalize_witness(address_mutation)

    prototype_mutation = copy.deepcopy(d375_raw)
    prototype_mutation["live_inventory"]["rows"][0][
        "prototype_path_diagnostic"
    ] = "/__Prototype_999/changed"
    corrected_prototype_mutation, _ = _correct_selected_payload(prototype_mutation)

    path_id_mutation = copy.deepcopy(d375_raw["property_queries"])
    path_id_mutation["link5"]["rigid_body"]["path_id"] = -999
    normalized_path_id, _ = _normalize_properties(path_id_mutation)

    elapsed_mutation = copy.deepcopy(d375_raw["property_queries"])
    elapsed_mutation["link5"]["elapsed_s"] = 999.0
    normalized_elapsed, _ = _normalize_properties(elapsed_mutation)

    vertex_mutation = copy.deepcopy(d375_witness["normalized_values"][0])
    vertex_mutation["events"][0]["convexes"][0]["vertices"][0][0] += 1e-6

    semantic_mutation = copy.deepcopy(d375_raw)
    semantic_mutation["callback_rows"][0]["live_path"] += "_changed"
    corrected_semantic_mutation, _ = _correct_selected_payload(semantic_mutation)

    volume_mutation = copy.deepcopy(d375_raw["property_queries"])
    volume_mutation["link5"]["colliders"][0]["volume_m3"] += 1e-9
    normalized_volume, _ = _normalize_properties(volume_mutation)

    invalid_manifest = {
        "selected_payload": {
            "callback_rows[*].callback.witness_sha256": 34,
            "live_inventory.rows[*].prototype_path_diagnostic": 34,
            "callback_rows[*].live_path": 34,
        },
        "normalized_callback_witness": {"request_return_repr": 34},
        "normalized_property_query": {
            path: 1 for path in EXPECTED_PROPERTY_EXCLUSION_PATHS
        },
    }
    checks = {
        "raw_v1_comparator_must_detect_mismatch": _canonical_sha(v1_left)
        != _canonical_sha(v1_right),
        "witness_sha_exclusion_alone_must_still_fail": _canonical_sha(
            witness_only_left
        )
        != _canonical_sha(witness_only_right),
        "prototype_exclusion_alone_must_still_fail": _canonical_sha(
            prototype_only_left
        )
        != _canonical_sha(prototype_only_right),
        "address_perturbation_must_not_change_normalized_witness": _canonical_sha(
            normalized_address_mutation
        )
        == _canonical_sha(d375_witness["normalized_values"][0]),
        "prototype_ordinal_perturbation_must_not_change_corrected_workload": _canonical_sha(
            corrected_prototype_mutation
        )
        == _canonical_sha(corrected_left),
        "property_path_id_perturbation_must_not_change_normalized_property": _canonical_sha(
            normalized_path_id
        )
        == _canonical_sha(d375_property),
        "property_elapsed_perturbation_must_not_change_normalized_property": _canonical_sha(
            normalized_elapsed
        )
        == _canonical_sha(d375_property),
        "vertex_perturbation_must_change_normalized_witness": _canonical_sha(
            vertex_mutation
        )
        != _canonical_sha(d375_witness["normalized_values"][0]),
        "semantic_path_perturbation_must_change_corrected_workload": _canonical_sha(
            corrected_semantic_mutation
        )
        != _canonical_sha(corrected_left),
        "property_volume_perturbation_must_change_normalized_property": _canonical_sha(
            normalized_volume
        )
        != _canonical_sha(d375_property),
        "overbroad_exclusion_manifest_must_be_rejected": not _allowed_exclusion_manifest_valid(
            invalid_manifest
        ),
    }
    return {
        "checks": checks,
        "passed": sum(value is True for value in checks.values()),
        "total": len(checks),
        "pass": all(checks.values()),
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


def _assert_ascii(value: Any) -> None:
    if isinstance(value, str):
        if any(ord(char) > 127 for char in value):
            raise ValueError(f"non-ASCII display text: {value!r}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_ascii(key)
            _assert_ascii(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_ascii(child)


def _render_board(evidence: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    corrected = evidence["corrected_authority"]
    outcomes = evidence["paired_outcomes"]
    texts = {
        "title": "D378 | Repairing the D377 workload comparator",
        "subtitle": (
            "Offline only | immutable D375 + D377 evidence | "
            "Isaac 0 | PhysX 0 | q5 0 | physics 0 | contact 0"
        ),
        "raw_title": "1. Frozen D377 result",
        "raw_body": (
            "V1 selected digests differed.\n"
            "D375: ec930163...b0b7b\n"
            "D377: 75850473...13c81\n"
            "Formal D377 FAIL_STOP remains frozen."
        ),
        "cause_title": "2. Exact false differences",
        "cause_body": (
            "34 callback witness SHA values:\n"
            "only runtime object addresses differed.\n"
            "34 prototype diagnostics:\n"
            "only generated __Prototype_N differed."
        ),
        "repair_title": "3. Preregistered V2 authority",
        "repair_body": (
            "Keep raw files and provenance.\n"
            "Normalize only exact runtime diagnostics.\n"
            "Keep vertices, indices, polygons, paths,\n"
            "owners, Float32 hashes, mass and volume."
        ),
        "digest_title": "Corrected authoritative equality",
        "digest_body": (
            f"D375 workload: {corrected['D375_corrected_workload_sha256'][:16]}...\n"
            f"D377 workload: {corrected['D377_corrected_workload_sha256'][:16]}...\n"
            f"Normalized witnesses: {corrected['D375_normalized_witness_sha256'][:16]}...\n"
            f"Normalized properties: {corrected['D375_normalized_property_sha256'][:16]}..."
        ),
        "payload_title": "Meaningful payload stayed exact",
        "payload_body": (
            "Callbacks: 34 / 34\n"
            "Vertices: 314 | indices: 1016\n"
            "Original polygons: 262\n"
            f"Negative controls: {evidence['negative_controls']['passed']}/"
            f"{evidence['negative_controls']['total']} PASS"
        ),
        "outcome_title": "Observed terminal outcomes",
        "outcome_body": (
            f"D375: no explicit Erase | timeout | return {outcomes['D375_returncode']}\n"
            f"D377: one Erase | clean exit in {outcomes['D377_elapsed_s']:.3f} s | "
            f"return {outcomes['D377_returncode']}\n"
            "Conditional trigger support for this pair: PASS\n"
            "Universal necessity and exact native root cause: NOT PROVEN"
        ),
        "footer": (
            "D378 verdict: corrected workload authority PASS | "
            "D377 artifact not rewritten | full P34 identity, cylinder physics, "
            "closure and grasp remain NULL | g0a_pass=false"
        ),
    }
    _assert_ascii(texts)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#F7F9FC")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.axis("off")
    fig.text(
        0.5,
        0.955,
        texts["title"],
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#14213D",
    )
    fig.text(
        0.5,
        0.918,
        texts["subtitle"],
        ha="center",
        va="center",
        fontsize=12.5,
        color="#4B5563",
    )

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        fill: str,
        edge: str,
        *,
        body_size: float = 12.0,
    ) -> None:
        canvas.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                linewidth=2,
                edgecolor=edge,
                facecolor=fill,
            )
        )
        fig.text(
            x + 0.018,
            y + h - 0.045,
            title,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=edge,
        )
        fig.text(
            x + 0.018,
            y + h - 0.09,
            body,
            ha="left",
            va="top",
            fontsize=body_size,
            color="#1F2937",
            linespacing=1.45,
        )

    box(
        0.045,
        0.61,
        0.27,
        0.235,
        texts["raw_title"],
        texts["raw_body"],
        "#FDECEC",
        "#9B1C1C",
    )
    box(
        0.365,
        0.61,
        0.27,
        0.235,
        texts["cause_title"],
        texts["cause_body"],
        "#FFF4D8",
        "#9A6700",
    )
    box(
        0.685,
        0.61,
        0.27,
        0.235,
        texts["repair_title"],
        texts["repair_body"],
        "#E8F3FF",
        "#1D4E89",
    )
    box(
        0.045,
        0.31,
        0.43,
        0.22,
        texts["digest_title"],
        texts["digest_body"],
        "#E8F7EE",
        "#176B3A",
    )
    box(
        0.525,
        0.31,
        0.43,
        0.22,
        texts["payload_title"],
        texts["payload_body"],
        "#E8F7EE",
        "#176B3A",
    )
    box(
        0.045,
        0.11,
        0.91,
        0.13,
        texts["outcome_title"],
        texts["outcome_body"],
        "#EEF2FF",
        "#3B4CCA",
        body_size=11.8,
    )
    fig.text(
        0.5,
        0.045,
        texts["footer"],
        ha="center",
        va="center",
        fontsize=12.3,
        fontweight="bold",
        color="#7F1D1D",
    )
    fig.savefig(BOARD_PATH, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    info = _png_info(BOARD_PATH)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"D378 board dimension failure: {info}")
    return info


def _write_rerun(evidence: dict[str, Any]) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.viz_debug import log_rerun

    stages = [
        (0, 0.0, "V1 comparator reproduced: selected workload mismatch."),
        (1, 0.4, "34 witness SHA differences traced to runtime object addresses."),
        (2, 0.6, "34 prototype diagnostics traced to generated ordinals."),
        (3, 0.8, "Property diagnostics: 38 path_id and 2 elapsed values excluded."),
        (4, 1.0, "Corrected workload, witnesses, and properties are exact."),
    ]
    scalars = [
        {
            "entity_path": "metrics/d378/authority_repair_progress",
            "value": value,
            "sequence": {"audit_step": step},
        }
        for step, value, _ in stages
    ]
    events = [
        {
            "entity_path": "events/d378/timeline",
            "text": text,
            "level": "INFO" if step > 0 else "WARN",
            "sequence": {"audit_step": step},
        }
        for step, _, text in stages
    ]
    events.extend(
        [
            {
                "entity_path": "events/d378/verdict",
                "text": (
                    "D378 PASS: corrected termination-workload authority is equal. "
                    "D377 frozen FAIL_STOP is not rewritten."
                ),
                "level": "INFO",
                "static": True,
            },
            {
                "entity_path": "events/d378/boundary",
                "text": (
                    "This pair supports StageCache retention as a conditional trigger. "
                    "Universal necessity and exact native root cause remain unproven."
                ),
                "level": "WARN",
                "static": True,
            },
            {
                "entity_path": "events/d378/scope",
                "text": (
                    "Offline only. Full P34 identity, cylinder physics, q5 closure, "
                    "contact, grasp, target, IK and path remain null."
                ),
                "level": "WARN",
                "static": True,
            },
        ]
    )
    metadata = {
        "case": "g0a_d378",
        "verdict": evidence["verdict"],
        "source": "immutable D375 and D377 JSON plus callback witnesses",
        "display_role": "ASCII-only inspection projection",
        "authority": "canonical JSON and hashes",
        "isaac_launches": 0,
        "physx_calls": 0,
        "physics_steps": 0,
        "q5_samples": 0,
        "contact_queries": 0,
        "g0a_pass": False,
    }
    _assert_ascii({"events": events, "metadata": metadata})
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    try:
        saved = log_rerun(
            RRD_PATH,
            scalar_trace=scalars,
            events=events,
            recording_metadata=metadata,
            recording_id="g0a_d378_workload_authority_repair",
            blueprint_path=RBL_PATH,
            blueprint_mode="d378_workload_authority_repair",
            live_viewer=False,
            app_id="roarm_g0a_d378_workload_authority_repair",
        )
    finally:
        os.environ["PATH"] = old_path
    if not saved.get("ok"):
        raise RuntimeError(f"D378 save-only Rerun failed: {saved}")
    exact_entities = {
        "metadata/run",
        "metrics/d378/authority_repair_progress",
        "events/d378/timeline",
        "events/d378/verdict",
        "events/d378/boundary",
        "events/d378/scope",
    }
    components = {
        "metadata/run": ["TextDocument:text"],
        "metrics/d378/authority_repair_progress": ["Scalars:scalars"],
        "events/d378/timeline": ["TextLog:text", "TextLog:level"],
        "events/d378/verdict": ["TextLog:text", "TextLog:level"],
        "events/d378/boundary": ["TextLog:text", "TextLog:level"],
        "events/d378/scope": ["TextLog:text", "TextLog:level"],
    }
    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(exact_entities),
        exact_entity_paths=sorted(exact_entities),
        expected_timeline_names=["audit_step", "blueprint", "log_time"],
        exact_timeline_names=["audit_step", "blueprint", "log_time"],
        expected_entity_components=components,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG_PATH,
        screenshot_window_size="1920x1080",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=240.0,
    )
    _write_json_x(RERUN_VALIDATION_PATH, validation)
    headless = dict(validation.get("headless_render") or {})
    return {
        "save_only": saved,
        "strict_validation_pass": validation.get("pass") is True,
        "viewer_invocations": 1 if headless.get("attempted") is True else 0,
        "viewer_returncode": headless.get("returncode"),
        "requested_logical_window_size": "1920x1080",
        "physical_raster_note": (
            "HiDPI may produce 3840x2160. Exact 1920x1080 authority belongs "
            "to the Matplotlib board."
        ),
        "rrd": {
            "path": _rel(RRD_PATH),
            "bytes": RRD_PATH.stat().st_size,
            "sha256": _sha(RRD_PATH),
        },
        "rbl": {
            "path": _rel(RBL_PATH),
            "bytes": RBL_PATH.stat().st_size,
            "sha256": _sha(RBL_PATH),
        },
        "screenshot": (
            _png_info(RERUN_PNG_PATH)
            if RERUN_PNG_PATH.is_file()
            else {"path": _rel(RERUN_PNG_PATH), "exists": False}
        ),
        "validation": {
            "path": _rel(RERUN_VALIDATION_PATH),
            "bytes": RERUN_VALIDATION_PATH.stat().st_size,
            "sha256": _sha(RERUN_VALIDATION_PATH),
        },
    }


def _prepare() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"forward-only D378 attempt exists: {_rel(OUT_DIR)}")
    OUT_DIR.mkdir(parents=True)
    _phase("prepare_start")
    allowed_dirty = sorted(
        [
            "START_HERE.md",
            _rel(CASE_ROOT) + "/",
            "roarm_rl/viz_debug.py",
            _rel(HARNESS),
        ]
    )
    input_hashes = _input_hashes()
    checks = {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_master_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "dirty_paths_exactly_approved_D378": _status_paths() == allowed_dirty,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "rerun_sdk_exact": _package_version("rerun-sdk") == "0.34.1",
        "rerun_cli_absolute_exists": RERUN_CLI.is_file(),
        "repo_root_sys_path_zero": sys.path[0] == str(REPO),
        "forbidden_modules_absent": not _forbidden_modules_loaded(),
        "known_raw_hashes_exact": all(
            path.is_file() and _sha(path) == expected
            for path, expected in EXPECTED_RAW_SHA.items()
        ),
        "callback_witness_count_D375_34": len(
            list((D375_DIR / "callback_witnesses").glob("*.json"))
        )
        == 34,
        "callback_witness_count_D377_34": len(
            list((D377_DIR / "callback_witnesses").glob("*.json"))
        )
        == 34,
        "D377_frozen_verdict_exact": _read_json(D377_EVIDENCE).get("verdict")
        == D377_FROZEN_VERDICT,
        "D334_sidecar_exists": D334_SIDECAR.is_dir(),
        "start_here_active_path_registered": _rel(CASE_ROOT)
        in START_HERE.read_text(encoding="utf-8"),
    }
    prereg = {
        "artifact": "D378_PREREGISTRATION_V1",
        "case": "g0a_d378",
        "attempt": OUT_DIR.name,
        "user_authorization": (
            "Proceed step by step with the recommended next minimum D378."
        ),
        "what_and_why": (
            "Repair the D377 termination-workload comparator authority offline "
            "without rewriting D377 or running Isaac/PhysX/science."
        ),
        "new_variables": NEW_VARIABLES,
        "forward_only_output": _rel(OUT_DIR),
        "expected_v1_digests": EXPECTED_V1_DIGESTS,
        "expected_corrected_workload_sha256": EXPECTED_CORRECTED_WORKLOAD_SHA,
        "expected_normalized_witness_sha256": EXPECTED_NORMALIZED_WITNESS_SHA,
        "expected_normalized_property_sha256": EXPECTED_NORMALIZED_PROPERTY_SHA,
        "expected_callback_counts": EXPECTED_CALLBACK_COUNTS,
        "exclusion_manifest": {
            "selected_payload": {
                "callback_rows[*].callback.witness_sha256": 34,
                "live_inventory.rows[*].prototype_path_diagnostic": 34,
            },
            "normalized_callback_witness": {
                "request_return_repr": 34,
            },
            "normalized_property_query": {
                path: 1 for path in EXPECTED_PROPERTY_EXCLUSION_PATHS
            },
        },
        "exclusion_policy": (
            "Raw paths and hashes remain provenance. Only the listed exact "
            "cross-run runtime diagnostics are excluded from identity."
        ),
        "authoritative_fields_retained": [
            "callback vertices, indices, original polygons, planes and result",
            "authored Float32 topology hashes",
            "semantic live path, owner path, part name and role",
            "collision, approximation, hull limit and typed min-thickness bits",
            "mass, COM, inertia, axes, AABB, local pose, volume and VALID result",
            "execution counters, timeline state and frozen no-science scope",
        ],
        "failure_capable_perturbations": [
            "raw V1 mismatch",
            "witness-only exclusion remains mismatch",
            "prototype-only exclusion remains mismatch",
            "runtime address mutation invariant",
            "prototype ordinal mutation invariant",
            "property path_id mutation invariant",
            "property elapsed mutation invariant",
            "vertex mutation detected",
            "semantic live-path mutation detected",
            "property volume mutation detected",
            "overbroad exclusion manifest rejected",
        ],
        "run_contract": {
            "offline_audit_invocations": 1,
            "automatic_retries": 0,
            "rerun_save_only": 1,
            "rerun_viewer_max": 1,
            "D377_frozen_verdict_rewrite": 0,
            "strict_zero_scope": STRICT_ZERO_SCOPE,
        },
        "source_hashes": _source_hashes(),
        "input_hashes": input_hashes,
        "D375_inventory_before": _inventory(D375_DIR),
        "D377_inventory_before": _inventory(D377_DIR),
        "D334_sidecar_before": _inventory(D334_SIDECAR),
        "attempt1_preregistration_fail_stop_before": _inventory(ATTEMPT1_DIR),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "status_short": _git("status", "--short"),
            "allowed_dirty_paths": allowed_dirty,
        },
        "environment": {
            "python": sys.executable,
            "rerun_sdk": _package_version("rerun-sdk"),
            "rerun_cli": str(RERUN_CLI),
            "forbidden_modules_loaded": _forbidden_modules_loaded(),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    _phase(
        "preregistration_frozen",
        preregistration_sha256=_sha(PREREG_PATH),
        passed=prereg["pass"],
    )
    if not prereg["pass"]:
        raise RuntimeError(f"D378 preregistration failed: {checks}")


def _run() -> None:
    if not PREREG_PATH.is_file():
        raise RuntimeError("D378 preregistration missing")
    if EVIDENCE_PATH.exists():
        raise FileExistsError(f"D378 actual audit already ran: {_rel(EVIDENCE_PATH)}")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D378 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D378 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D378 immutable input hash drift")
    if _inventory(D375_DIR) != prereg["D375_inventory_before"]:
        raise RuntimeError("D375 immutable inventory drift")
    if _inventory(D377_DIR) != prereg["D377_inventory_before"]:
        raise RuntimeError("D377 immutable inventory drift")
    if _inventory(D334_SIDECAR) != prereg["D334_sidecar_before"]:
        raise RuntimeError("D334 sidecar drift")
    if _inventory(ATTEMPT1_DIR) != prereg["attempt1_preregistration_fail_stop_before"]:
        raise RuntimeError("D378 attempt1 preregistration FAIL_STOP drift")
    if _forbidden_modules_loaded():
        raise RuntimeError("forbidden NVIDIA runtime module loaded before audit")

    _write_json_x(
        INVOCATION_PATH,
        {
            "artifact": "D378_SINGLE_OFFLINE_AUDIT_INVOCATION_V1",
            "argv": sys.argv,
            "pid": os.getpid(),
            "offline_audit_invocations": 1,
            "automatic_retries": 0,
            "isaac_or_physx_worker_invocations": 0,
            "preregistration_sha256": _sha(PREREG_PATH),
        },
    )
    _phase("offline_audit_start", invocation_sha256=_sha(INVOCATION_PATH))

    d375_raw = _read_json(D375_RAW)
    d377_raw = _read_json(D377_RAW)
    d375_v1 = _v1_workload_payload(d375_raw)
    d377_v1 = _v1_workload_payload(d377_raw)
    d375_corrected, d375_selected_counts = _correct_selected_payload(d375_raw)
    d377_corrected, d377_selected_counts = _correct_selected_payload(d377_raw)
    selected_diffs = _recursive_diffs(d375_v1, d377_v1)
    corrected_diffs = _recursive_diffs(d375_corrected, d377_corrected)

    d375_witness = _witness_authority(d375_raw)
    d377_witness = _witness_authority(d377_raw)
    witness_pair_diffs = [
        {
            "pair_index": index,
            "body": d375_witness["aggregate_rows"][index]["body"],
            "name": d375_witness["aggregate_rows"][index]["name"],
            "diffs": _recursive_diffs(left, right),
        }
        for index, (left, right) in enumerate(
            zip(
                d375_witness["normalized_values"],
                d377_witness["normalized_values"],
            )
        )
    ]

    d375_property, d375_property_removed = _normalize_properties(
        d375_raw["property_queries"]
    )
    d377_property, d377_property_removed = _normalize_properties(
        d377_raw["property_queries"]
    )
    raw_property_diffs = _recursive_diffs(
        d375_raw["property_queries"], d377_raw["property_queries"]
    )
    normalized_property_diffs = _recursive_diffs(d375_property, d377_property)

    exclusion_counts = {
        **d375_selected_counts,
        "normalized_witness_request_return_repr": d375_witness[
            "request_return_repr_removed"
        ],
        "property_path_id": sum(
            row["key"] == "path_id" for row in d375_property_removed
        ),
        "property_elapsed_s": sum(
            row["key"] == "elapsed_s" for row in d375_property_removed
        ),
    }
    exclusions_symmetric = bool(
        d375_selected_counts == d377_selected_counts
        and d375_witness["request_return_repr_removed"]
        == d377_witness["request_return_repr_removed"]
        and [row["key"] for row in d375_property_removed]
        == [row["key"] for row in d377_property_removed]
    )

    negatives = _negative_controls(
        d375_raw, d377_raw, d375_witness, d375_property
    )
    _phase(
        "failure_capable_perturbations_complete",
        passed=negatives["passed"],
        total=negatives["total"],
    )

    d375_supervisor = _read_json(D375_SUPERVISOR)
    d377_supervisor = _read_json(D377_SUPERVISOR)
    d377_evidence = _read_json(D377_EVIDENCE)
    d375_phase_names = [row["phase"] for row in _read_jsonl(D375_PHASES)]
    d377_phase_names = [row["phase"] for row in _read_jsonl(D377_PHASES)]
    corrected_authority = {
        "D375_v1_selected_sha256": _canonical_sha(d375_v1),
        "D377_v1_selected_sha256": _canonical_sha(d377_v1),
        "D375_corrected_workload_sha256": _canonical_sha(d375_corrected),
        "D377_corrected_workload_sha256": _canonical_sha(d377_corrected),
        "D375_normalized_witness_sha256": d375_witness["aggregate_sha256"],
        "D377_normalized_witness_sha256": d377_witness["aggregate_sha256"],
        "D375_normalized_property_sha256": _canonical_sha(d375_property),
        "D377_normalized_property_sha256": _canonical_sha(d377_property),
        "selected_v1_diff_count": len(selected_diffs),
        "selected_v1_diff_classification": {
            "callback_witness_sha256": sum(
                row["path"].endswith("/callback/witness_sha256")
                for row in selected_diffs
            ),
            "prototype_path_diagnostic": sum(
                row["path"].endswith("/prototype_path_diagnostic")
                for row in selected_diffs
            ),
        },
        "selected_corrected_diff_count": len(corrected_diffs),
        "normalized_witness_pair_diff_count": sum(
            len(row["diffs"]) for row in witness_pair_diffs
        ),
        "raw_property_diff_count": len(raw_property_diffs),
        "raw_property_diff_classification": {
            "path_id": sum(
                row["path"].endswith("/path_id") for row in raw_property_diffs
            ),
            "elapsed_s": sum(
                row["path"].endswith("/elapsed_s") for row in raw_property_diffs
            ),
        },
        "normalized_property_diff_count": len(normalized_property_diffs),
        "D375_callback_counts": d375_witness["counts"],
        "D377_callback_counts": d377_witness["counts"],
        "exclusion_counts": exclusion_counts,
        "exclusions_symmetric": exclusions_symmetric,
        "witness_provenance": {
            "D375": d375_witness["provenance_rows"],
            "D377": d377_witness["provenance_rows"],
        },
        "property_removed_paths": {
            "D375": d375_property_removed,
            "D377": d377_property_removed,
        },
    }
    paired_outcomes = {
        "D375_explicit_stagecache_erase_phase_count": sum(
            name.startswith("stagecache_erase") for name in d375_phase_names
        ),
        "D377_explicit_stagecache_erase_phase_count": sum(
            name == "stagecache_erase_call_end" for name in d377_phase_names
        ),
        "D377_erase_contract_pass": d377_evidence["erase_audit"]["pass"],
        "D375_returncode": d375_supervisor["returncode"],
        "D375_timed_out": d375_supervisor["timed_out"],
        "D375_elapsed_s": d375_supervisor["elapsed_s"],
        "D377_returncode": d377_supervisor["returncode"],
        "D377_timed_out": d377_supervisor["timed_out"],
        "D377_elapsed_s": d377_supervisor["elapsed_s"],
        "conditional_trigger_support_this_pair": True,
        "universal_stagecache_erase_necessity": None,
        "exact_native_root_cause": None,
    }
    scope = {
        "offline_audit_invocations": 1,
        "automatic_retries": 0,
        **{key: 0 for key in STRICT_ZERO_SCOPE},
    }
    checks = {
        "v1_digests_exact_and_mismatch_reproduced": {
            "D375": corrected_authority["D375_v1_selected_sha256"],
            "D377": corrected_authority["D377_v1_selected_sha256"],
        }
        == EXPECTED_V1_DIGESTS,
        "selected_v1_exact_68_false_differences": corrected_authority[
            "selected_v1_diff_count"
        ]
        == 68
        and corrected_authority["selected_v1_diff_classification"]
        == {
            "callback_witness_sha256": 34,
            "prototype_path_diagnostic": 34,
        },
        "corrected_workload_digest_exact_both": corrected_authority[
            "D375_corrected_workload_sha256"
        ]
        == corrected_authority["D377_corrected_workload_sha256"]
        == EXPECTED_CORRECTED_WORKLOAD_SHA,
        "normalized_witness_digest_exact_both": corrected_authority[
            "D375_normalized_witness_sha256"
        ]
        == corrected_authority["D377_normalized_witness_sha256"]
        == EXPECTED_NORMALIZED_WITNESS_SHA,
        "normalized_witness_pairs_exact_34": len(witness_pair_diffs) == 34
        and corrected_authority["normalized_witness_pair_diff_count"] == 0,
        "callback_payload_counts_exact_both": d375_witness["counts"]
        == d377_witness["counts"]
        == EXPECTED_CALLBACK_COUNTS,
        "all_raw_witness_hashes_match_summary": all(
            row["raw_sha_matches_summary"]
            for row in (
                d375_witness["provenance_rows"]
                + d377_witness["provenance_rows"]
            )
        ),
        "normalized_property_digest_exact_both": corrected_authority[
            "D375_normalized_property_sha256"
        ]
        == corrected_authority["D377_normalized_property_sha256"]
        == EXPECTED_NORMALIZED_PROPERTY_SHA,
        "raw_property_diff_exact_38_plus_2": corrected_authority[
            "raw_property_diff_count"
        ]
        == 40
        and corrected_authority["raw_property_diff_classification"]
        == {"path_id": 38, "elapsed_s": 2},
        "normalized_property_diff_zero": corrected_authority[
            "normalized_property_diff_count"
        ]
        == 0,
        "property_exclusion_paths_exact_both": sorted(
            row["path"] for row in d375_property_removed
        )
        == sorted(row["path"] for row in d377_property_removed)
        == EXPECTED_PROPERTY_EXCLUSION_PATHS,
        "exclusion_counts_exact_and_symmetric": exclusion_counts
        == EXPECTED_EXCLUSION_COUNTS
        and exclusions_symmetric,
        "exclusion_manifest_exact_allowlist": _allowed_exclusion_manifest_valid(
            prereg["exclusion_manifest"]
        ),
        "negative_controls_all_pass": negatives["pass"]
        and negatives["passed"] == negatives["total"] == 11,
        "D377_frozen_verdict_preserved": d377_evidence["verdict"]
        == D377_FROZEN_VERDICT,
        "paired_external_outcomes_exact": bool(
            paired_outcomes["D375_explicit_stagecache_erase_phase_count"] == 0
            and paired_outcomes["D377_explicit_stagecache_erase_phase_count"] == 1
            and paired_outcomes["D377_erase_contract_pass"] is True
            and paired_outcomes["D375_returncode"] == -9
            and paired_outcomes["D375_timed_out"] is True
            and paired_outcomes["D377_returncode"] == 0
            and paired_outcomes["D377_timed_out"] is False
        ),
        "strict_zero_scope": all(scope[key] == 0 for key in STRICT_ZERO_SCOPE),
        "immutable_inputs_still_exact": _input_hashes() == prereg["input_hashes"],
        "D375_inventory_immutable": _inventory(D375_DIR)
        == prereg["D375_inventory_before"],
        "D377_inventory_immutable": _inventory(D377_DIR)
        == prereg["D377_inventory_before"],
        "D334_sidecar_immutable": _inventory(D334_SIDECAR)
        == prereg["D334_sidecar_before"],
        "attempt1_preregistration_fail_stop_immutable": _inventory(ATTEMPT1_DIR)
        == prereg["attempt1_preregistration_fail_stop_before"],
        "forbidden_modules_absent_before_visualization": not _forbidden_modules_loaded(),
    }
    pass_before_visualization = all(checks.values())
    paired_outcomes["conditional_trigger_support_this_pair"] = bool(
        pass_before_visualization
    )
    evidence = {
        "artifact": "D378_WORKLOAD_AUTHORITY_REPAIR_EVIDENCE_V1",
        "case": "g0a_d378",
        "attempt": OUT_DIR.name,
        "what_and_why": prereg["what_and_why"],
        "new_variables": NEW_VARIABLES,
        "exclusion_manifest": prereg["exclusion_manifest"],
        "corrected_authority": corrected_authority,
        "negative_controls": negatives,
        "paired_outcomes": paired_outcomes,
        "interpretation": {
            "corrected_termination_workload_equivalence": pass_before_visualization,
            "stagecache_retention_conditional_trigger_support_this_pair": (
                pass_before_visualization
            ),
            "D377_frozen_artifact_rewritten": False,
            "D377_formal_verdict_preserved": D377_FROZEN_VERDICT,
            "universal_stagecache_erase_necessity": None,
            "exact_native_root_cause": None,
        },
        "scope_counters": scope,
        "remaining_nulls": {
            "full_p34_live_identity": None,
            "A64_P34_physics_equivalence": None,
            "cylinder_contact_or_tipping": None,
            "q5_closure": None,
            "grasp_feasibility": None,
            "target_IK_path_repair": None,
        },
        "g0a_pass": False,
        "checks": checks,
        "pass": pass_before_visualization,
        "verdict": VERDICT_PASS if pass_before_visualization else VERDICT_FAIL,
        "next_authorization_boundary": (
            "A separately approved full P34 live-identity classifier may be "
            "considered. The 29x50 target rebase and all physics remain separate."
        ),
    }
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase(
        "authoritative_offline_evidence_committed",
        evidence_sha256=_sha(EVIDENCE_PATH),
        verdict=evidence["verdict"],
    )
    if not evidence["pass"]:
        raise RuntimeError(f"D378 authoritative offline audit failed: {checks}")

    board = _render_board(evidence)
    _phase("exact_1920x1080_ascii_board_complete", board_sha256=board["sha256"])
    rerun = _write_rerun(evidence)
    _phase(
        "save_only_ascii_rerun_and_single_headless_capture_complete",
        strict_validation_pass=rerun["strict_validation_pass"],
    )
    after_forbidden = _forbidden_modules_loaded()
    automated_checks = {
        "evidence_pass": evidence["pass"],
        "board_exact_1920x1080": board["exact_1920x1080"],
        "rerun_save_only_ok": rerun["save_only"].get("ok") is True,
        "rerun_strict_validation_pass": rerun["strict_validation_pass"],
        "rerun_viewer_exactly_one": rerun["viewer_invocations"] == 1,
        "rerun_viewer_return_zero": rerun["viewer_returncode"] == 0,
        "rerun_screenshot_exists": RERUN_PNG_PATH.is_file()
        and RERUN_PNG_PATH.stat().st_size > 0,
        "forbidden_NVIDIA_runtime_modules_absent_after_visualization": not after_forbidden,
        "D375_inventory_immutable": _inventory(D375_DIR)
        == prereg["D375_inventory_before"],
        "D377_inventory_immutable": _inventory(D377_DIR)
        == prereg["D377_inventory_before"],
        "D334_sidecar_immutable": _inventory(D334_SIDECAR)
        == prereg["D334_sidecar_before"],
        "attempt1_preregistration_fail_stop_immutable": _inventory(ATTEMPT1_DIR)
        == prereg["attempt1_preregistration_fail_stop_before"],
        "scope_zero_preserved": all(scope[key] == 0 for key in STRICT_ZERO_SCOPE),
        "D377_frozen_verdict_still_preserved": evidence["interpretation"][
            "D377_formal_verdict_preserved"
        ]
        == D377_FROZEN_VERDICT,
    }
    automated = {
        "artifact": "D378_AUTOMATED_SUMMARY_V1",
        "invocation": {
            "path": _rel(INVOCATION_PATH),
            "sha256": _sha(INVOCATION_PATH),
        },
        "evidence": {"path": _rel(EVIDENCE_PATH), "sha256": _sha(EVIDENCE_PATH)},
        "board": board,
        "rerun": rerun,
        "checks": automated_checks,
        "pass": all(automated_checks.values()),
        "manual_visual_inspection": "pending",
        "completion_contract_pass": False,
        "g0a_pass": False,
    }
    _write_json_x(AUTOMATED_PATH, automated)
    _write_json_x(
        MANUAL_TEMPLATE_PATH,
        {
            "artifact": "D378_MANUAL_ORIGINAL_RESOLUTION_VISUAL_INSPECTION_TEMPLATE_V1",
            "expected_sha256": {
                "board": board["sha256"],
                "rerun_inspection": rerun["screenshot"].get("sha256"),
            },
            "required_checks": {
                "board_text_legible_no_overlap_or_clipping": False,
                "raw_fail_and_corrected_pass_visually_separated": False,
                "D377_frozen_verdict_not_rewritten": False,
                "rerun_ascii_text_has_no_missing_glyphs": False,
                "rerun_required_rows_visible": False,
                "scope_and_remaining_nulls_visible": False,
            },
            "pass": False,
        },
    )
    if not automated["pass"]:
        raise RuntimeError(f"D378 automated contract failed: {automated_checks}")
    _phase(
        "run_complete_awaiting_manual_inspection",
        automated_summary_sha256=_sha(AUTOMATED_PATH),
    )


def _finalize() -> None:
    required = [
        PREREG_PATH,
        INVOCATION_PATH,
        EVIDENCE_PATH,
        AUTOMATED_PATH,
        MANUAL_PATH,
        BOARD_PATH,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION_PATH,
        RERUN_PNG_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"D378 finalize prerequisite missing: {_rel(path)}")
    if COMPLETION_PATH.exists():
        raise FileExistsError(f"D378 completion exists: {_rel(COMPLETION_PATH)}")
    _phase("finalize_start")
    prereg = _read_json(PREREG_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    automated = _read_json(AUTOMATED_PATH)
    manual = _read_json(MANUAL_PATH)
    expected_manual_hashes = {
        "board": automated["board"]["sha256"],
        "rerun_inspection": automated["rerun"]["screenshot"]["sha256"],
    }
    required_manual_checks = {
        "board_text_legible_no_overlap_or_clipping",
        "raw_fail_and_corrected_pass_visually_separated",
        "D377_frozen_verdict_not_rewritten",
        "rerun_ascii_text_has_no_missing_glyphs",
        "rerun_required_rows_visible",
        "scope_and_remaining_nulls_visible",
    }
    current_board = _png_info(BOARD_PATH)
    current_rerun_png = _png_info(RERUN_PNG_PATH)
    checks = {
        "preregistration_pass": prereg["pass"] is True,
        "preregistration_current_sha_exact": _sha(PREREG_PATH)
        == _read_json(INVOCATION_PATH)["preregistration_sha256"],
        "invocation_current_sha_exact": _sha(INVOCATION_PATH)
        == automated["invocation"]["sha256"],
        "evidence_pass": evidence["pass"] is True,
        "evidence_current_sha_exact": _sha(EVIDENCE_PATH)
        == automated["evidence"]["sha256"],
        "automated_summary_pass": automated["pass"] is True,
        "manual_original_resolution_inspection_pass": manual.get("pass") is True,
        "manual_hashes_exact": manual.get("inspected_sha256")
        == expected_manual_hashes,
        "manual_artifact_exact": manual.get("artifact")
        == "D378_MANUAL_ORIGINAL_RESOLUTION_VISUAL_INSPECTION_V1",
        "manual_check_keys_exact_and_all_true": set(
            manual.get("checks", {})
        )
        == required_manual_checks
        and all(
            manual["checks"].get(key) is True for key in required_manual_checks
        ),
        "board_current_hash_size_dimensions_exact": current_board
        == automated["board"],
        "rrd_current_hash_size_exact": RRD_PATH.stat().st_size
        == automated["rerun"]["rrd"]["bytes"]
        and _sha(RRD_PATH) == automated["rerun"]["rrd"]["sha256"],
        "rbl_current_hash_size_exact": RBL_PATH.stat().st_size
        == automated["rerun"]["rbl"]["bytes"]
        and _sha(RBL_PATH) == automated["rerun"]["rbl"]["sha256"],
        "rerun_png_current_hash_size_dimensions_exact": current_rerun_png
        == automated["rerun"]["screenshot"],
        "rerun_validation_current_hash_size_exact": RERUN_VALIDATION_PATH.stat().st_size
        == automated["rerun"]["validation"]["bytes"]
        and _sha(RERUN_VALIDATION_PATH)
        == automated["rerun"]["validation"]["sha256"],
        "source_hashes_still_exact": _source_hashes() == prereg["source_hashes"],
        "input_hashes_still_exact": _input_hashes() == prereg["input_hashes"],
        "D375_inventory_immutable": _inventory(D375_DIR)
        == prereg["D375_inventory_before"],
        "D377_inventory_immutable": _inventory(D377_DIR)
        == prereg["D377_inventory_before"],
        "D334_sidecar_immutable": _inventory(D334_SIDECAR)
        == prereg["D334_sidecar_before"],
        "attempt1_preregistration_fail_stop_immutable": _inventory(ATTEMPT1_DIR)
        == prereg["attempt1_preregistration_fail_stop_before"],
        "D377_frozen_verdict_preserved": evidence["interpretation"][
            "D377_formal_verdict_preserved"
        ]
        == D377_FROZEN_VERDICT,
        "remaining_science_nulls_preserved": all(
            value is None for value in evidence["remaining_nulls"].values()
        ),
        "g0a_false": evidence["g0a_pass"] is False
        and automated["g0a_pass"] is False,
        "forbidden_NVIDIA_runtime_modules_absent": not _forbidden_modules_loaded(),
    }
    completion = {
        "artifact": "D378_COMPLETION_SUMMARY_V1",
        "case": "g0a_d378",
        "attempt": OUT_DIR.name,
        "new_variables": NEW_VARIABLES,
        "preregistration": {
            "path": _rel(PREREG_PATH),
            "sha256": _sha(PREREG_PATH),
        },
        "invocation": {
            "path": _rel(INVOCATION_PATH),
            "sha256": _sha(INVOCATION_PATH),
        },
        "evidence": {"path": _rel(EVIDENCE_PATH), "sha256": _sha(EVIDENCE_PATH)},
        "automated_summary": {
            "path": _rel(AUTOMATED_PATH),
            "sha256": _sha(AUTOMATED_PATH),
        },
        "manual_inspection": {
            "path": _rel(MANUAL_PATH),
            "sha256": _sha(MANUAL_PATH),
        },
        "board": automated["board"],
        "rrd": automated["rerun"]["rrd"],
        "rbl": automated["rerun"]["rbl"],
        "rerun_inspection": automated["rerun"]["screenshot"],
        "rerun_validation": automated["rerun"]["validation"],
        "corrected_authority": evidence["corrected_authority"],
        "interpretation": evidence["interpretation"],
        "scope_counters": evidence["scope_counters"],
        "remaining_nulls": evidence["remaining_nulls"],
        "g0a_pass": False,
        "checks": checks,
        "pass": all(checks.values()),
        "verdict": VERDICT_PASS if all(checks.values()) else VERDICT_FAIL,
        "next_authorization_boundary": evidence["next_authorization_boundary"],
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase(
        "finalize_complete",
        completion_sha256=_sha(COMPLETION_PATH),
        verdict=completion["verdict"],
    )
    if not completion["pass"]:
        raise RuntimeError(f"D378 completion failed: {checks}")


def main() -> int:
    parser = argparse.ArgumentParser()
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
    except Exception as exc:
        payload = {
            "artifact": "D378_RUNTIME_EXCEPTION_V1",
            "stage": args.stage,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "verdict": VERDICT_FAIL,
        }
        try:
            if OUT_DIR.exists() and not EXCEPTION_PATH.exists():
                _write_json_x(EXCEPTION_PATH, payload)
            if OUT_DIR.exists():
                _phase(
                    "exception",
                    stage=args.stage,
                    exception_type=type(exc).__name__,
                )
        except Exception:
            pass
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
