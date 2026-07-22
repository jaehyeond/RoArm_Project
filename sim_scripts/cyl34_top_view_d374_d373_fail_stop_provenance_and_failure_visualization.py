#!/usr/bin/env python3
"""D374: offline provenance and visualization of the frozen D373 fail-stop.

This case reads immutable D373 JSON/log/callback witnesses and the frozen D343
typed-Float32 contract.  It never imports or launches Isaac, Kit, PhysX, Warp,
Hydra, or Fabric.  It does not author USD, step physics, command q5, query
contacts, or decide P34 live identity / grasp feasibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASE_ROOT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d374"
OUT_DIR = CASE_ROOT / "attempt1_d373_fail_stop_provenance_and_failure_visualization"
PREREG_PATH = OUT_DIR / "d374_preregistration.json"
PHASE_PATH = OUT_DIR / "d374_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d374_invocation.json"
EVIDENCE_PATH = OUT_DIR / "d374_failure_provenance_evidence.json"
REPAIR_PATH = OUT_DIR / "d374_live_repair_contract.json"
FAILURE_PNG = OUT_DIR / "d374_failure_provenance_1920x1080.png"
OVERVIEW_PNG = OUT_DIR / "d374_p34_assembled_and_exploded_1920x1080.png"
LINK5_PNG = OUT_DIR / "d374_link5_16_colliders_1920x1080.png"
GRIPPER_PNG = OUT_DIR / "d374_gripper_link_18_colliders_1920x1080.png"
RRD_PATH = OUT_DIR / "d374_failure_and_p34_inspection.rrd"
RBL_PATH = OUT_DIR / "d374_failure_and_p34_inspection.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d374_rerun_validation.json"
RERUN_PNG = OUT_DIR / "d374_rerun_inspection.png"
AUTOMATED_PATH = OUT_DIR / "d374_automated_summary.json"
MANUAL_JSON_PATH = OUT_DIR / "d374_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d374_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d374_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d374_runtime_exception.json"

D373_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d373/attempt1_p34_live_asset_identity_preflight"
D373_PREREG = D373_DIR / "d373_preregistration.json"
D373_RAW = D373_DIR / "d373_worker_raw_summary.json"
D373_PRECLOSE = D373_DIR / "d373_worker_preclose_sentinel.json"
D373_SUPERVISOR = D373_DIR / "d373_worker_supervisor.json"
D373_FAIL = D373_DIR / "d373_fail_stop_attestation.json"
D373_STDOUT = D373_DIR / "d373_worker_stdout.log"
D373_STDERR = D373_DIR / "d373_worker_stderr.log"
D373_WITNESSES = D373_DIR / "callback_witnesses"
D343_SUMMARY = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d343/d343_usd_typed_float_readback_summary.json"
D343_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d343/d343_usd_typed_float_readback_evidence.json"
D343_PREREG = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d343/d343_preregistration.json"
D343_RERUN_OMISSION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d343/d343_rerun_omission_justification.json"
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
HARNESS = Path(__file__).resolve()

EXPECTED_HEAD = "548d3517f5a7936529646c5d8b0009427eb936ab"
EXPECTED_D373_HASHES = {
    "preregistration": "f6cc93647c16ad441776846308601d797fcdcdf081ba2e57c0cec4b571b21e2d",
    "raw_summary": "dd57da307acf6134487bcd1dfa4a847fd41f24832177421f6291c45b06091373",
    "preclose": "a32f6f423d0b7620a940d534e9f70bdc873fcdea90852d3731fdf6cc19bfa06a",
    "supervisor": "3891bb51fbab02731edbea43e516048b6b4fac4b005e6bec5f27c5cabcb39643",
    "fail_attestation": "a47ea8600ddc74600644c2d747dd5f95861a2ecbcb2e0667ba0641e17f717206",
    "stdout": "b5766ff871b552118f86049a5b6a38dec609be4e555d3748bc5794bea48ad43a",
    "stderr": "b7da181d227fa9268b1ddfe6dd9a070cbc44a8f0ac0e5fc50f39f2fc207622bb",
}
EXPECTED_D343_HASHES = {
    "summary": "880601aac768df38675603828258850aea796b6436a299c46f8cc489ed8b00da",
    "evidence": "95bb4e3787d300071f1bac22037814b732781cd72a69a0334a34a05a50ac920b",
    "preregistration": "fb8f9c292042001aeb05d9b693d910797bd4a214d9e01427ccd54b7e2c387ce8",
    "rerun_omission": "9c8e69c7f798756eea1a1a34a21bd2e74d7716e53992904b5ee7cc6ca5abd7c5",
}
EXPECTED_SCHEMA_SHA256 = "fe075bce4bde5ba7db69c6ccef0c4c26909336ab34c619129fc276f7cb4d7abc"
EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = [
    "d373_fail_stop_provenance_contract_v1",
    "d373_failure_and_p34_visualization_projection_v1",
]
BODY_COUNTS = {"link5": 16, "gripper_link": 18}
ROLE_COUNTS = {
    "link5": {"structural_body": 1, "connector_support": 3, "fixed_jaw": 10, "fixed_jaw_backbone": 2},
    "gripper_link": {"moving_support": 4, "moving_jaw": 12, "moving_jaw_backbone": 2},
}
ROLE_COLORS_HEX = {
    "structural_body": "#0072B2",
    "connector_support": "#E69F00",
    "fixed_jaw": "#F0E442",
    "fixed_jaw_backbone": "#D97800",
    "moving_support": "#CC79A7",
    "moving_jaw": "#009E73",
    "moving_jaw_backbone": "#00695C",
}
ROLE_COLORS_RGBA = {
    "structural_body": [0, 114, 178, 185],
    "connector_support": [230, 159, 0, 185],
    "fixed_jaw": [240, 228, 66, 210],
    "fixed_jaw_backbone": [217, 120, 0, 205],
    "moving_support": [204, 121, 167, 185],
    "moving_jaw": [0, 158, 115, 210],
    "moving_jaw_backbone": [0, 105, 92, 205],
}
VERDICT_PASS = "D374_D373_FAIL_STOP_PROVENANCE_AND_FAILURE_VISUALIZATION_PASS"
VERDICT_FAIL = "D374_OFFLINE_PROVENANCE_OR_OBSERVABILITY_FAIL_STOP"


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


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


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _phase(name: str, **fields: Any) -> None:
    ordinal = 1
    if PHASE_PATH.is_file():
        ordinal = len([line for line in PHASE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]) + 1
    row = {"ordinal": ordinal, "phase": name, "monotonic_ns": time.monotonic_ns(), **fields}
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _inventory(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": _sha(path)})
    digest = _canonical_sha(rows)
    return {"root": _rel(root), "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows), "inventory_sha256": digest, "files": rows}


def _sidecar_snapshot() -> dict[str, Any]:
    return _inventory(D334_SIDECAR)


def _source_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in (HARNESS, VIZ_DEBUG, RERUN_CONTRACT)}


def _frozen_hash_checks() -> dict[str, bool]:
    return {
        "d373_preregistration": _sha(D373_PREREG) == EXPECTED_D373_HASHES["preregistration"],
        "d373_raw_summary": _sha(D373_RAW) == EXPECTED_D373_HASHES["raw_summary"],
        "d373_preclose": _sha(D373_PRECLOSE) == EXPECTED_D373_HASHES["preclose"],
        "d373_supervisor": _sha(D373_SUPERVISOR) == EXPECTED_D373_HASHES["supervisor"],
        "d373_fail_attestation": _sha(D373_FAIL) == EXPECTED_D373_HASHES["fail_attestation"],
        "d373_stdout": _sha(D373_STDOUT) == EXPECTED_D373_HASHES["stdout"],
        "d373_stderr": _sha(D373_STDERR) == EXPECTED_D373_HASHES["stderr"],
        "d343_summary": _sha(D343_SUMMARY) == EXPECTED_D343_HASHES["summary"],
        "d343_evidence": _sha(D343_EVIDENCE) == EXPECTED_D343_HASHES["evidence"],
        "d343_preregistration": _sha(D343_PREREG) == EXPECTED_D343_HASHES["preregistration"],
        "d343_rerun_omission": _sha(D343_RERUN_OMISSION) == EXPECTED_D343_HASHES["rerun_omission"],
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _prepare_negative_controls() -> dict[str, Any]:
    """Failure-capable offline controls; no typed-Float32 scalar is recomputed."""
    raw = _read_json(D373_RAW)
    supervisor = _read_json(D373_SUPERVISOR)
    preclose = _read_json(D373_PRECLOSE)
    canonical = raw["canonical_outside_collision_subtree_diff"]
    queries = raw["property_queries"]

    returncode_only_would_accept = supervisor["returncode"] == 0
    repaired_supervisor_accepts = bool(
        supervisor["returncode"] == 0
        and supervisor.get("pass") is True
        and raw.get("worker_protocol_pass") is True
        and preclose.get("worker_protocol_pass") is True
        and preclose.get("summary_sha256") == _sha(D373_RAW)
    )
    default_traversal_zero = canonical["variant_p34_mesh_path_count"] == 0
    proxy_aware_inventory_exact = len(raw["live_inventory"]["enabled_collision_paths"]) == 34
    property_sentinel_zero = all(row["rigid_body"]["mass_kg"] == 0.0 for row in queries.values())
    property_results_valid = all(row["rigid_body"]["result_value"] == 0 for row in queries.values())
    callback_protocol_all = all(row["protocol_pass"] for row in raw["callback_rows"])
    full_identity_available = False

    rows = {
        "returncode_only_false_positive_rejected": returncode_only_would_accept and not repaired_supervisor_accepts,
        "default_traversal_zero_not_promoted_to_asset_absence": default_traversal_zero and proxy_aware_inventory_exact,
        "error_sentinel_zero_not_promoted_to_valid_property": property_sentinel_zero and not property_results_valid,
        "callback_protocol_not_promoted_to_full_identity": callback_protocol_all and not full_identity_available,
    }
    return {
        "artifact": "D374_PREPARE_FAILURE_CAPABLE_CONTROLS_V1",
        "typed_float32_retest_count": 0,
        "controls": rows,
        "passed": sum(rows.values()),
        "total": len(rows),
        "pass": all(rows.values()),
    }


def _prepare() -> None:
    if CASE_ROOT.exists():
        raise FileExistsError(f"forward-only D374 path already exists: {CASE_ROOT}")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"wrong Python: {sys.executable}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/master")
    frozen_checks = _frozen_hash_checks()
    negative = _prepare_negative_controls()
    source_hashes = _source_hashes()
    d373_inventory = _inventory(D373_DIR)
    d334 = _sidecar_snapshot()
    forbidden_loaded = sorted(
        name for name in sys.modules
        if name == "omni" or name.startswith(("omni.", "isaacsim", "isaaclab", "warp", "pxr"))
    )
    checks = {
        "head_exact": head == EXPECTED_HEAD,
        "origin_exact": origin == EXPECTED_HEAD,
        "head_equals_origin": head == origin,
        "new_variable_count_1_or_2": 1 <= len(NEW_VARIABLES) <= 2,
        "all_frozen_hashes_exact": all(frozen_checks.values()),
        "d373_witness_file_count_68": len(list(D373_WITNESSES.glob("*.json"))) == 68,
        "negative_controls_4_of_4": negative["pass"],
        "rerun_sdk_0_34_1": _package_version("rerun-sdk") == "0.34.1",
        "numpy_1_26_0": np.__version__ == "1.26.0",
        "psutil_5_9_8": _package_version("psutil") == "5.9.8",
        "rerun_cli_absolute_exists": RERUN_CLI.is_file(),
        "fonts_exist": FONT_REGULAR.is_file() and FONT_BOLD.is_file(),
        "forbidden_runtime_modules_not_loaded": not forbidden_loaded,
    }
    prereg = {
        "artifact": "D374_PREREGISTRATION_V1",
        "case": "g0a_d374",
        "attempt": OUT_DIR.name,
        "status": "PREREGISTERED_NOT_RUN",
        "new_variables": NEW_VARIABLES,
        "scope": {
            "immutable_d373_read_only": True,
            "d343_typed_float32_contract_inherited_without_retest": True,
            "isaac_launches": 0,
            "physx_calls": 0,
            "usd_writes": 0,
            "physics_steps": 0,
            "q5_commands_or_samples": 0,
            "contact_queries": 0,
            "cylinder_writes": 0,
            "target_ik_path_changes": 0,
            "collider_regeneration": 0,
            "automatic_decomposition_sweeps": 0,
            "rerun_viewer_screenshot_invocations_max": 1,
        },
        "registered_questions": [
            "Which frozen D373 observations prove each fail-stop cause?",
            "What must a later live worker change at traversal/property/supervisor boundaries?",
            "What exact 16+18 convex shapes did the D373 instance callback return?",
        ],
        "registered_authority": {
            "failure_provenance": "immutable D373 raw summary, preclose sentinel, supervisor, logs, and fail attestation",
            "typed_float32": "immutable D343 exact typed contract; D374 performs zero scalar readback/packing/ULP tests",
            "display_geometry": "D373 instance callback polygon arrays; prototype arrays are equality cross-check only",
            "display_triangulation": "fan triangulation of callback polygons for rendering only; never a scientific hash authority",
            "coordinate_scope": "each body remains in its own owner-local frame; no fake q5/world assembly",
        },
        "registered_nulls": [
            "full_p34_live_identity",
            "authored_to_callback_surface_bounds_topology_volume",
            "live_property_mass_com_inertia_axes",
            "physics_equivalence_or_runtime_speed",
            "tipping_causality",
            "grasp_feasibility",
        ],
        "registered_negative_controls": negative,
        "expected_part_counts": BODY_COUNTS,
        "expected_role_counts": ROLE_COUNTS,
        "expected_outputs": [
            _rel(EVIDENCE_PATH), _rel(REPAIR_PATH), _rel(FAILURE_PNG), _rel(OVERVIEW_PNG),
            _rel(LINK5_PNG), _rel(GRIPPER_PNG), _rel(RRD_PATH), _rel(RBL_PATH),
            _rel(RERUN_VALIDATION_PATH), _rel(RERUN_PNG), _rel(AUTOMATED_PATH),
            _rel(MANUAL_JSON_PATH), _rel(MANUAL_MD_PATH), _rel(COMPLETION_PATH),
        ],
        "git": {"head": head, "origin_master": origin, "status_short": _git("status", "--short")},
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "psutil": _package_version("psutil"),
            "rerun_sdk": _package_version("rerun-sdk"),
            "rerun_cli": str(RERUN_CLI),
            "forbidden_loaded_modules": forbidden_loaded,
        },
        "frozen_hash_checks": frozen_checks,
        "d373_inventory_before": d373_inventory,
        "d334_sidecar_before": d334,
        "source_hashes": source_hashes,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not prereg["pass"]:
        raise RuntimeError(f"D374 preregistration failed: {checks}")
    _write_json_x(PREREG_PATH, prereg)
    _phase("prepare_complete", preregistration_sha256=_sha(PREREG_PATH), d373_inventory_sha256=d373_inventory["inventory_sha256"])


def _triangulate_callback_convex(convex: dict[str, Any]) -> np.ndarray:
    vertices = np.asarray(convex["vertices"], dtype=np.float64)
    indices = [int(value) for value in convex["indices"]]
    triangles: list[list[int]] = []
    for polygon in convex["polygons"]:
        base = int(polygon["index_base"])
        count = int(polygon["num_vertices"])
        face = indices[base : base + count]
        if len(face) != count or count < 3:
            raise ValueError(f"invalid callback polygon: base={base} count={count}")
        plane_normal = np.asarray(polygon["plane"][:3], dtype=np.float64)
        for index in range(1, count - 1):
            tri = [face[0], face[index], face[index + 1]]
            cross = np.cross(vertices[tri[1]] - vertices[tri[0]], vertices[tri[2]] - vertices[tri[0]])
            if float(np.dot(cross, plane_normal)) < 0.0:
                tri = [tri[0], tri[2], tri[1]]
            triangles.append(tri)
    result = np.asarray(triangles, dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 3 or result.size == 0:
        raise ValueError("callback triangulation produced no triangles")
    if int(result.min()) < 0 or int(result.max()) >= int(vertices.shape[0]):
        raise ValueError("callback triangle index out of bounds")
    return result


def _audit_callback_witnesses(raw: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_rows = {(row["body"], row["prim_name"]): row for row in raw["callback_rows"]}
    witness_groups: dict[tuple[str, str], dict[str, tuple[Path, dict[str, Any]]]] = defaultdict(dict)
    for path in sorted(D373_WITNESSES.glob("*.json")):
        witness = _read_json(path)
        key = (witness["body"], witness["prim_name"])
        channel = witness["channel"]
        if channel in witness_groups[key]:
            raise RuntimeError(f"duplicate callback witness channel: {key} {channel}")
        witness_groups[key][channel] = (path, witness)

    parts: list[dict[str, Any]] = []
    channel_protocol_pass = 0
    witness_sha_match = 0
    channel_payload_match = 0
    total_vertices = 0
    total_polygons = 0
    total_display_triangles = 0
    max_vertices = 0
    max_polygons = 0

    for key in sorted(raw_rows, key=lambda item: (item[0] != "link5", item[1])):
        row = raw_rows[key]
        channels = witness_groups.get(key, {})
        if set(channels) != {"instance", "prototype"}:
            raise RuntimeError(f"callback channel set mismatch for {key}: {sorted(channels)}")
        payloads: dict[str, dict[str, Any]] = {}
        channel_summaries = {}
        for channel in ("instance", "prototype"):
            path, witness = channels[channel]
            event = witness["events"][0] if len(witness.get("events", [])) == 1 else None
            checks = {
                "witness_sha_matches_raw": _sha(path) == row["channels"][channel]["witness_sha256"],
                "path_matches_raw": _rel(path) == row["channels"][channel]["witness_path"],
                "callback_count_one": witness.get("callback_count") == 1,
                "single_event": event is not None,
                "result_valid": event is not None and event.get("result_value") == 0 and event.get("result_name") == "RESULT_VALID",
                "one_convex": event is not None and event.get("convex_count") == 1 and len(event.get("convexes", [])) == 1,
                "serialization_errors_empty": event is not None and event.get("serialization_errors") == [],
                "raw_channel_protocol_pass": row["channels"][channel]["pass"] is True,
            }
            if not all(checks.values()):
                raise RuntimeError(f"callback witness protocol failed for {key} {channel}: {checks}")
            witness_sha_match += int(checks["witness_sha_matches_raw"])
            channel_protocol_pass += 1
            payloads[channel] = event["convexes"][0]
            channel_summaries[channel] = {
                "path": _rel(path),
                "sha256": _sha(path),
                "checks": checks,
            }

        payload_exact = payloads["instance"] == payloads["prototype"]
        channel_payload_match += int(payload_exact)
        if not payload_exact:
            raise RuntimeError(f"instance/prototype callback payload differs for {key}")
        convex = payloads["instance"]
        vertices = np.asarray(convex["vertices"], dtype=np.float64)
        triangles = _triangulate_callback_convex(convex)
        lo = vertices.min(axis=0)
        hi = vertices.max(axis=0)
        total_vertices += int(convex["vertex_count"])
        total_polygons += int(convex["polygon_count"])
        total_display_triangles += int(triangles.shape[0])
        max_vertices = max(max_vertices, int(convex["vertex_count"]))
        max_polygons = max(max_polygons, int(convex["polygon_count"]))
        parts.append(
            {
                "body": row["body"],
                "prim_name": row["prim_name"],
                "part_name": row["name"],
                "role": row["role"],
                "instance_path": row["instance_path"],
                "prototype_path": row["prototype_path"],
                "callback_payload_instance_prototype_exact": payload_exact,
                "callback_payload_sha256": _canonical_sha(convex),
                "vertex_count": int(convex["vertex_count"]),
                "polygon_count": int(convex["polygon_count"]),
                "display_triangle_count": int(triangles.shape[0]),
                "bounds_min_m": lo.tolist(),
                "bounds_max_m": hi.tolist(),
                "bounds_size_mm": ((hi - lo) * 1000.0).tolist(),
                "channels": channel_summaries,
                "vertices_m": vertices,
                "triangles": triangles,
            }
        )

    body_counts = Counter(part["body"] for part in parts)
    role_counts = {body: Counter(part["role"] for part in parts if part["body"] == body) for body in BODY_COUNTS}
    checks = {
        "raw_callback_rows_exact_34": len(raw_rows) == 34,
        "witness_json_exact_68": sum(len(group) for group in witness_groups.values()) == 68,
        "unique_part_keys_exact_34": len(witness_groups) == 34 and set(witness_groups) == set(raw_rows),
        "protocol_channels_68_of_68": channel_protocol_pass == 68,
        "witness_sha_68_of_68": witness_sha_match == 68,
        "instance_prototype_payload_34_of_34": channel_payload_match == 34,
        "link5_16": body_counts["link5"] == 16,
        "gripper_link_18": body_counts["gripper_link"] == 18,
        "role_counts_exact": all(dict(role_counts[body]) == ROLE_COUNTS[body] for body in BODY_COUNTS),
        "raw_protocol_rows_34_of_34": len(raw["callback_rows"]) == 34 and all(row["protocol_pass"] for row in raw["callback_rows"]),
    }
    audit = {
        "artifact": "D374_D373_CALLBACK_VISUAL_SOURCE_AUDIT_V1",
        "display_authority": "D373 frozen instance callback original polygon arrays",
        "prototype_role": "exact-payload cross-check only; not drawn as 34 additional shapes",
        "coordinate_scope": "owner-local per body; link5 and gripper_link are not combined into a world/q5 pose",
        "display_triangulation": "fan triangulation from callback polygon index streams, with plane-normal winding check; display only",
        "part_counts": dict(body_counts),
        "role_counts": {body: dict(role_counts[body]) for body in BODY_COUNTS},
        "channel_count": channel_protocol_pass,
        "instance_prototype_exact_count": channel_payload_match,
        "total_vertex_count": total_vertices,
        "total_original_polygon_count": total_polygons,
        "total_display_triangle_count": total_display_triangles,
        "max_vertices_per_part": max_vertices,
        "max_polygons_per_part": max_polygons,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return audit, parts


def _typed_float32_inheritance(raw: dict[str, Any]) -> dict[str, Any]:
    d343 = _read_json(D343_SUMMARY)
    d343_evidence = _read_json(D343_EVIDENCE)
    d373_prereg = _read_json(D373_PREREG)
    expected_value = d343["expected_float32_value_m"]
    authored_values = [row["min_thickness_m"] for row in raw["authored_readback"]["rows"]]
    live_values = [row["min_thickness_m"] for row in raw["live_inventory"]["rows"]]
    checks = {
        "d343_summary_pass": d343["pass"] is True,
        "d343_expected_bits_exact": d343["expected_float32_bits_hex"] == "0x38d1b717",
        "d343_schema_exact": d343_evidence["contract"]["schema"]["sha256"] == EXPECTED_SCHEMA_SHA256,
        "d373_schema_same": d373_prereg["nvidia_contract"]["schema_sha256"] == EXPECTED_SCHEMA_SHA256,
        "d373_authored_numeric_34_equal_d343_typed_value": len(authored_values) == 34 and all(value == expected_value for value in authored_values),
        "d373_live_numeric_34_equal_d343_typed_value": len(live_values) == 34 and all(value == expected_value for value in live_values),
        "d373_only_row_failure_is_min_thickness": all(
            set(key for key, value in row["checks"].items() if not value) == {"min_thickness_frozen"}
            for row in [*raw["authored_readback"]["rows"], *raw["live_inventory"]["rows"]]
        ),
    }
    return {
        "authority": "D343 frozen USD typed-Float32 contract",
        "d374_typed_scalar_retest_count": 0,
        "d374_float_pack_or_ulp_recompute_count": 0,
        "expected_float32_bits_hex_inherited": d343["expected_float32_bits_hex"],
        "expected_float32_bits_uint32_inherited": d343["expected_float32_bits_uint32"],
        "expected_float32_le_bytes_hex_inherited": d343["expected_float32_le_bytes_hex"],
        "expected_typed_value_m_inherited": expected_value,
        "decimal_requested_value_m": 0.0001,
        "representation_delta_m_inherited": d343["float32_representation_delta_m"],
        "d373_raw_recorded_type_name": None,
        "d373_raw_recorded_bits": None,
        "bit_classification_basis": "D343 exact-bit authority plus identical pinned schema and D373 numeric linkage; D373 did not directly persist typeName/bits",
        "classification": "proven_false_failure_in_D373_decimal_comparator; not geometry corruption",
        "checks": checks,
        "pass": all(checks.values()),
    }


def _failure_provenance(raw: dict[str, Any], typed: dict[str, Any]) -> dict[str, Any]:
    supervisor = _read_json(D373_SUPERVISOR)
    preclose = _read_json(D373_PRECLOSE)
    canonical = raw["canonical_outside_collision_subtree_diff"]
    stdout_lines = D373_STDOUT.read_text(encoding="utf-8", errors="replace").splitlines()
    instance_proxy_lines = [
        {"line": index + 1, "text": line}
        for index, line in enumerate(stdout_lines)
        if "RigidBodyAPI on an instance proxy not supported" in line
    ]
    query_rows = raw["property_queries"]
    property_checks = {
        "two_queries_finished": set(query_rows) == {"link5", "gripper_link"} and all(row["finished"] for row in query_rows.values()),
        "two_queries_error_parsing": all(row["rigid_body"]["result_value"] == 5 and row["rigid_body"]["result_name"] == "ERROR_PARSING" for row in query_rows.values()),
        "error_sentinel_paths_empty": all(row["rigid_body"]["path"] == "" and row["rigid_body"]["path_id"] == 0 for row in query_rows.values()),
        "error_sentinel_mass_zero_not_measurement": all(row["rigid_body"]["mass_kg"] == 0.0 for row in query_rows.values()),
        "runtime_instance_proxy_warning_for_both_subject_bodies": any("/World/Robot/link5" in row["text"] for row in instance_proxy_lines) and any("/World/Robot/gripper_link" in row["text"] for row in instance_proxy_lines),
        "all_live_p34_rows_instance_proxies": len(raw["live_inventory"]["rows"]) == 34 and all(row["is_instance_proxy"] for row in raw["live_inventory"]["rows"]),
    }
    traversal_checks = {
        "default_comparator_reports_zero_p34": canonical["variant_p34_mesh_path_count"] == 0 and canonical["variant_p34_path_count"] == 0,
        "proxy_aware_live_inventory_reports_34": len(raw["live_inventory"]["enabled_collision_paths"]) == 34,
        "callback_rows_report_34": len(raw["callback_rows"]) == 34,
        "outside_subtree_limited_hash_is_exact": canonical["checks"]["outside_registered_subtrees_bit_exact"] is True,
    }
    supervisor_effective = bool(
        supervisor["returncode"] == 0
        and supervisor.get("pass") is True
        and raw.get("worker_protocol_pass") is True
        and preclose.get("worker_protocol_pass") is True
        and preclose.get("summary_sha256") == _sha(D373_RAW)
    )
    supervisor_checks = {
        "process_exit_zero": supervisor["returncode"] == 0,
        "old_supervisor_pass_true": supervisor["pass"] is True,
        "worker_raw_protocol_false": raw["worker_protocol_pass"] is False,
        "preclose_worker_protocol_false": preclose["worker_protocol_pass"] is False,
        "preclose_hash_binds_raw_summary": preclose["summary_sha256"] == _sha(D373_RAW),
        "repaired_effective_pass_false": supervisor_effective is False,
    }
    return {
        "typed_float32_false_failure": typed,
        "whole_robot_instance_proxy_property_query": {
            "classification": "stage structure incompatible with dynamic articulation rigid-body property query",
            "query_results": {
                body: {
                    "result_name": row["rigid_body"]["result_name"],
                    "result_value": row["rigid_body"]["result_value"],
                    "elapsed_s": row["elapsed_s"],
                    "expected_collider_count_including_disabled_legacy": row["expected_collider_count_including_disabled_legacy"],
                    "observed_collider_rows": len(row["colliders"]),
                    "sentinel_mass_kg_not_measurement": row["rigid_body"]["mass_kg"],
                }
                for body, row in query_rows.items()
            },
            "runtime_warning_lines": instance_proxy_lines,
            "checks": property_checks,
            "pass": all(property_checks.values()),
        },
        "instance_proxy_traversal_scope": {
            "classification": "default-stage traversal blind spot, not P34 asset absence",
            "default_visible_rows": canonical["variant_meta"]["row_count"],
            "default_variant_p34_meshes": canonical["variant_p34_mesh_path_count"],
            "proxy_aware_live_p34_paths": len(raw["live_inventory"]["enabled_collision_paths"]),
            "callback_part_rows": len(raw["callback_rows"]),
            "checks": traversal_checks,
            "pass": all(traversal_checks.values()),
        },
        "worker_supervisor_authority": {
            "classification": "returncode-only false positive",
            "old_supervisor_pass": supervisor["pass"],
            "worker_protocol_pass": raw["worker_protocol_pass"],
            "preclose_worker_protocol_pass": preclose["worker_protocol_pass"],
            "hash_bound_effective_pass": supervisor_effective,
            "checks": supervisor_checks,
            "pass": all(supervisor_checks.values()),
        },
    }


def _repair_contract() -> dict[str, Any]:
    return {
        "artifact": "D374_LATER_LIVE_WORKER_REPAIR_CONTRACT_V1",
        "scope": "contract only; D374 does not implement or run the repaired live worker",
        "typed_float32": {
            "rule": "inherit D343 typed Float32 authority; compare stored typed value/bits, never decimal 0.0001 with 1e-12m",
            "D374_retest_count": 0,
        },
        "instance_proxy_traversal": {
            "rule": "separate default Stage traversal from proxy-aware traversal; P34 population requires TraverseInstanceProxies plus direct authored-layer audit",
            "absence_rule": "zero rows from default traversal may not prove collider absence",
        },
        "property_query_stage": {
            "rule": "keep articulation rigid-body owners non-instance; collider geometry leaf/prototype instancing may be separately retained only if installed runtime accepts it",
            "validity_rule": "ERROR_PARSING or nonzero result is an error; zero path/mass/volume in that row is sentinel data, not a measurement",
        },
        "worker_supervisor_authority": {
            "effective_pass_formula": "returncode==0 AND supervisor_operational_pass AND raw.worker_protocol_pass AND preclose.worker_protocol_pass AND preclose.summary_sha256==sha256(raw)",
            "cleanup_rule": "orderly return code proves cleanup only",
        },
        "promotion_boundary": {
            "requires_separate_user_approval": True,
            "next_live_identity_must_pass_before_physics_comparison": True,
            "D374_does_not_authorize_A64_vs_P34_physics": True,
        },
        "pass": True,
    }


def _exploded_vertices(part: dict[str, Any], ordinal: int) -> np.ndarray:
    body = part["body"]
    if body == "link5":
        cols, rows, dx, dz = 4, 4, 0.055, 0.060
    else:
        cols, rows, dx, dz = 6, 3, 0.047, 0.060
    col = ordinal % cols
    row = ordinal // cols
    target = np.asarray(
        [(col - (cols - 1) / 2.0) * dx, 0.0, ((rows - 1) / 2.0 - row) * dz],
        dtype=np.float64,
    )
    vertices = np.asarray(part["vertices_m"], dtype=np.float64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    return vertices - center + target


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


def _render_boards(parts: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch, Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    plt.rcParams["axes.unicode_minus"] = False

    def add_mesh(ax: Any, vertices: np.ndarray, triangles: np.ndarray, role: str, *, alpha: float = 0.72, width: float = 0.18) -> None:
        collection = Poly3DCollection(
            np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)] * 1000.0,
            facecolor=ROLE_COLORS_HEX[role],
            edgecolor="#202020",
            linewidth=width,
            alpha=alpha,
        )
        ax.add_collection3d(collection)

    def frame(ax: Any, vertices: np.ndarray, *, elev: float, azim: float) -> None:
        mm = np.asarray(vertices, dtype=np.float64) * 1000.0
        lo, hi = mm.min(axis=0), mm.max(axis=0)
        span = np.maximum(hi - lo, 1.0)
        center = (lo + hi) * 0.5
        radius = max(float(span.max()) * 0.57, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        ax.set_axis_off()

    # Board 1: explain why D373 stopped without hiding the successes.
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#F6F7F9")
    canvas = fig.add_axes([0, 0, 1, 1]); canvas.axis("off")
    fig.text(0.5, 0.945, "D374 · D373는 왜 ‘34개가 올라갔는데도’ 인증에 실패했나", ha="center", va="center", fontproperties=bold, fontsize=23, color="#17212B")
    fig.text(0.5, 0.900, "성공한 콜백 형상과 실패한 인증 단계를 분리한 오프라인 계보 · Isaac/PhysX 재실행 0회", ha="center", va="center", fontproperties=regular, fontsize=13, color="#4B5563")
    cards = [
        (0.055, 0.505, 0.425, 0.335, "1 · Float32 비교의 거짓 실패", "34개 모두 0.00009999999747378752 m로 읽혔습니다.\nD343이 이미 확정한 Float32 값과 같습니다.\n하지만 D373은 decimal 0.0001과 1e-12 m로 비교해\n2.5262e-12 m 차이를 오류로 만들었습니다.", "#E8F1FB", "#1769AA"),
        (0.520, 0.505, 0.425, 0.335, "2 · 전체 로봇 인스턴스화와 속성 질의", "link5와 gripper_link까지 instance proxy가 됐습니다.\n설치 PhysX는 동적 관절 rigid body의 이 구조를 거부했고\n두 질의가 ERROR_PARSING(5)로 끝났습니다.\n0 kg·0 m³는 측정값이 아니라 오류용 빈 값입니다.", "#FFF0E6", "#C65D00"),
        (0.055, 0.115, 0.425, 0.335, "3 · 기본 순회의 사각지대", "기본 Stage 순회는 instance proxy 아래를 보지 못해\nP34 메시를 0개라고 기록했습니다.\n반면 proxy-aware live 순회와 callback은\n각각 정확히 34개를 관측했습니다.", "#F2ECFA", "#71429B"),
        (0.520, 0.115, 0.425, 0.335, "4 · 종료 성공과 검사 성공의 혼동", "worker는 protocol_pass=false를 저장했습니다.\n그러나 supervisor는 return code 0만 보고 pass=true로 적었습니다.\n정상 종료는 cleanup 성공일 뿐, 형상 인증 PASS가 아닙니다.\n수리 계약은 raw+preclose hash까지 함께 요구합니다.", "#E9F7F0", "#147D5B"),
    ]
    for x, y, w, h, title, body, fill, accent in cards:
        canvas.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=1.3, edgecolor=accent, facecolor=fill, transform=canvas.transAxes))
        fig.text(x + 0.022, y + h - 0.055, title, ha="left", va="center", fontproperties=bold, fontsize=15, color=accent)
        fig.text(x + 0.022, y + h - 0.115, body, ha="left", va="top", fontproperties=regular, fontsize=12.5, color="#1F2937", linespacing=1.55)
    fig.text(0.5, 0.055, "결론: callback protocol 34/34는 보존 · full live identity, 물리 동등성, 전도 원인, 파지 가능성은 모두 미판정(null) · g0a_pass=false", ha="center", va="center", fontproperties=bold, fontsize=12.5, color="#9B1C1C")
    fig.savefig(FAILURE_PNG, dpi=120, facecolor=fig.get_facecolor()); plt.close(fig)

    # Board 2: assembled owner-local shapes and display-only exploded layouts.
    by_body = {body: [part for part in parts if part["body"] == body] for body in BODY_COUNTS}
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="white")
    axes = [fig.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    for col, body in enumerate(("link5", "gripper_link")):
        assembled = axes[col]
        all_vertices = []
        for part in by_body[body]:
            add_mesh(assembled, part["vertices_m"], part["triangles"], part["role"])
            all_vertices.append(part["vertices_m"])
        frame(assembled, np.vstack(all_vertices), elev=18, azim=-58 if body == "link5" else -72)
        assembled.set_title(("link5 · 고정 턱 포함 16개" if body == "link5" else "gripper_link · 움직이는 턱 포함 18개") + "\nD373 instance callback · 소유 링크 로컬 좌표", fontproperties=bold, fontsize=13)

        exploded = axes[2 + col]
        exploded_all = []
        for ordinal, part in enumerate(by_body[body]):
            moved = _exploded_vertices(part, ordinal)
            add_mesh(exploded, moved, part["triangles"], part["role"], alpha=0.78)
            exploded_all.append(moved)
        frame(exploded, np.vstack(exploded_all), elev=20, azim=-65)
        exploded.set_title(("link5 16개 분리 배열" if body == "link5" else "gripper_link 18개 분리 배열") + "\n보기 위해 옮긴 배치 · 실제 물리 자세 아님", fontproperties=bold, fontsize=13)
    legend_roles = ["structural_body", "connector_support", "fixed_jaw", "fixed_jaw_backbone", "moving_support", "moving_jaw", "moving_jaw_backbone"]
    legend_labels = ["몸통", "고정 턱 연결부", "고정 턱 접촉부", "고정 턱 뒷지지", "움직이는 연결부", "움직이는 턱 접촉부", "움직이는 턱 뒷지지"]
    fig.legend([Patch(facecolor=ROLE_COLORS_HEX[role], edgecolor="#202020") for role in legend_roles], legend_labels, loc="lower center", ncol=7, prop=regular, fontsize=9, bbox_to_anchor=(0.5, 0.018))
    fig.suptitle("D374 · PhysX가 D373에서 실제 반환한 P34 충돌체 34개", fontproperties=bold, fontsize=21, y=0.965)
    fig.text(0.5, 0.065, "prototype 34개는 instance와 polygon payload가 34/34 exact라 중복 표시하지 않음 · 삼각형은 화면 표시용 변환", ha="center", fontproperties=regular, fontsize=11.5)
    fig.tight_layout(rect=[0.01, 0.095, 0.99, 0.925]); fig.savefig(OVERVIEW_PNG, dpi=120, facecolor="white"); plt.close(fig)

    def render_sheet(body: str, path: Path, rows: int, cols: int, heading: str) -> None:
        fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="white")
        for ordinal, part in enumerate(by_body[body]):
            ax = fig.add_subplot(rows, cols, ordinal + 1, projection="3d")
            add_mesh(ax, part["vertices_m"], part["triangles"], part["role"], alpha=0.82, width=0.22)
            frame(ax, part["vertices_m"], elev=20, azim=-58 if body == "link5" else -72)
            short_id = part["prim_name"].split("_", 1)[0]
            display_name = part["part_name"]
            if len(display_name) > 29:
                display_name = display_name[:27] + "…"
            size = part["bounds_size_mm"]
            ax.set_title(f"{short_id} · {display_name}\n{size[0]:.1f}×{size[1]:.1f}×{size[2]:.1f} mm", fontproperties=bold, fontsize=7.5, pad=0)
        fig.suptitle(heading, fontproperties=bold, fontsize=20, y=0.975)
        footer = "각 칸은 자기 크기에 맞춰 확대됨 · 칸 사이 크기 비교 금지 · 색은 역할, 검은 선은 callback polygon의 표시용 삼각화"
        fig.text(0.5, 0.025, footer, ha="center", fontproperties=regular, fontsize=11)
        fig.tight_layout(rect=[0.012, 0.055, 0.988, 0.94], h_pad=0.35, w_pad=0.10)
        fig.savefig(path, dpi=120, facecolor="white"); plt.close(fig)

    render_sheet("link5", LINK5_PNG, 4, 4, "D374 · link5 충돌체 16개 — 몸통·연결부·고정 턱을 각각 확인")
    render_sheet("gripper_link", GRIPPER_PNG, 3, 6, "D374 · gripper_link 충돌체 18개 — 움직이는 연결부·접촉 턱·뒷지지를 각각 확인")

    infos = {
        "failure_provenance": _png_info(FAILURE_PNG),
        "assembled_and_exploded": _png_info(OVERVIEW_PNG),
        "link5_16": _png_info(LINK5_PNG),
        "gripper_link_18": _png_info(GRIPPER_PNG),
    }
    if not all(info["exact_1920x1080"] for info in infos.values()):
        raise RuntimeError(f"exact board dimension failure: {infos}")
    return infos


def _write_rerun(parts: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.viz_debug import log_rerun

    meshes = []
    for body in ("link5", "gripper_link"):
        body_parts = [part for part in parts if part["body"] == body]
        for ordinal, part in enumerate(body_parts):
            common = {
                "coordinate_frame": "tf#/",
                "triangles": part["triangles"],
                "color_rgba": ROLE_COLORS_RGBA[part["role"]],
                "static": True,
                "body": body,
                "part_name": part["part_name"],
                "role": part["role"],
                "callback_payload_sha256": part["callback_payload_sha256"],
                "display_role": "Float32 inspection copy only",
            }
            meshes.append({**common, "entity_path": f"d374/assembled/{body}/{part['prim_name']}", "vertices_m": part["vertices_m"]})
            meshes.append({**common, "entity_path": f"d374/exploded/{body}/{part['prim_name']}", "vertices_m": _exploded_vertices(part, ordinal)})
    events = [
        {
            "entity_path": "events/d374/summary",
            "text": "D373 instance callback shapes: link5=16, gripper_link=18. Owner-local display only; no q5/world assembly. Full live identity and all physics/grasp conclusions remain null.",
            "level": "INFO",
            "static": True,
        },
        {
            "entity_path": "events/d374/repair_contract",
            "text": "Later repair contract: inherit D343 typed Float32; traverse instance proxies explicitly; keep dynamic articulation owners non-instance; bind supervisor PASS to raw+preclose verdict/hash. D374 performs no live repair.",
            "level": "WARN",
            "static": True,
        },
    ]
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    try:
        result = log_rerun(
            RRD_PATH,
            meshes=meshes,
            events=events,
            recording_metadata={
                "case": "g0a_d374",
                "attempt": OUT_DIR.name,
                "verdict": evidence["verdict"],
                "evidence_sha256": _sha(EVIDENCE_PATH),
                "source": "immutable D373 callback witnesses",
                "isaac_launches": 0,
                "physx_calls": 0,
                "physics_steps": 0,
                "q5_samples": 0,
                "contact_queries": 0,
                "scientific_nulls_preserved": True,
                "display_role": "inspection only",
            },
            recording_id="g0a_d374_d373_failure_and_p34_callback_inspection",
            blueprint_path=RBL_PATH,
            blueprint_mode="d374_fail_stop_provenance",
            live_viewer=False,
            app_id="roarm_g0a_d374_failure_provenance",
        )
    finally:
        os.environ["PATH"] = old_path
    if not result.get("ok"):
        raise RuntimeError(f"Rerun save-only recording failed: {result}")

    exact_entities = {"metadata/run", "events/d374/summary", "events/d374/repair_contract"}
    component_contract: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "events/d374/summary": ["TextLog:text", "TextLog:level"],
        "events/d374/repair_contract": ["TextLog:text", "TextLog:level"],
    }
    mesh_components = ["CoordinateFrame:frame", "Mesh3D:albedo_factor", "Mesh3D:triangle_indices", "Mesh3D:vertex_positions"]
    for row in meshes:
        path = row["entity_path"]
        metadata_path = f"metadata/meshes/{path.replace('/', '__')}"
        exact_entities.update({path, metadata_path})
        component_contract[path] = mesh_components
        component_contract[metadata_path] = ["TextDocument:text"]
    strict = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(exact_entities),
        exact_entity_paths=sorted(exact_entities),
        expected_timeline_names=["blueprint", "log_time"],
        exact_timeline_names=["blueprint", "log_time"],
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
    screenshot = _png_info(RERUN_PNG) if RERUN_PNG.is_file() else {"path": _rel(RERUN_PNG), "exists": False}
    return {
        "save_only_log": result,
        "strict_validation_pass": strict.get("pass") is True,
        "rrd": {"path": _rel(RRD_PATH), "bytes": RRD_PATH.stat().st_size, "sha256": _sha(RRD_PATH)},
        "rbl": {"path": _rel(RBL_PATH), "bytes": RBL_PATH.stat().st_size, "sha256": _sha(RBL_PATH)},
        "headless_viewer_invocations": 1,
        "screenshot": screenshot,
        "manual_visual_inspection_complete": False,
    }


def _run() -> None:
    if not PREREG_PATH.is_file():
        raise RuntimeError("D374 preregistration is missing")
    for path in (EVIDENCE_PATH, AUTOMATED_PATH, RRD_PATH, RBL_PATH, RERUN_VALIDATION_PATH, RERUN_PNG):
        if path.exists():
            raise FileExistsError(f"forward-only run output already exists: {path}")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D374 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D374 source hash changed after preregistration")
    d373_before = _inventory(D373_DIR)
    if d373_before != prereg["d373_inventory_before"]:
        raise RuntimeError("immutable D373 inventory changed before run")
    if _sidecar_snapshot() != prereg["d334_sidecar_before"]:
        raise RuntimeError("user-owned D334 sidecar changed before run")
    frozen_checks = _frozen_hash_checks()
    if not all(frozen_checks.values()):
        raise RuntimeError(f"frozen input hash mismatch: {frozen_checks}")

    invocation = {
        "artifact": "D374_OFFLINE_INVOCATION_V1",
        "argv": sys.argv,
        "pid": os.getpid(),
        "python": sys.executable,
        "cwd": str(Path.cwd()),
        "preregistration_path": _rel(PREREG_PATH),
        "preregistration_sha256": _sha(PREREG_PATH),
        "offline_process_invocations": 1,
        "automatic_retries": 0,
        "isaac_or_physx_worker_invocations": 0,
        "rerun_viewer_max": 1,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase("offline_audit_start", invocation_sha256=_sha(INVOCATION_PATH))

    raw = _read_json(D373_RAW)
    typed = _typed_float32_inheritance(raw)
    callback_audit, parts = _audit_callback_witnesses(raw)
    provenance = _failure_provenance(raw, typed)
    repair = _repair_contract()
    _write_json_x(REPAIR_PATH, repair)
    _phase("provenance_and_callback_audit_complete", callback_parts=len(parts), callback_channels=callback_audit["channel_count"])

    provenance_pass = all(section["pass"] for section in provenance.values())
    if not typed["pass"] or not callback_audit["pass"] or not provenance_pass:
        raise RuntimeError("D374 offline provenance audit did not satisfy the registered contract")

    boards = _render_boards(parts, provenance)
    _phase("exact_1920x1080_boards_complete", board_count=len(boards))

    d373_after_boards = _inventory(D373_DIR)
    d334_after_boards = _sidecar_snapshot()
    nulls = {
        "full_p34_live_identity": None,
        "authored_to_callback_surface_bounds_topology_volume": None,
        "live_property_mass_com_inertia_axes": None,
        "physics_equivalence_or_runtime_speed": None,
        "tipping_causality": None,
        "grasp_feasibility": None,
    }
    part_summaries = [
        {key: value for key, value in part.items() if key not in {"vertices_m", "triangles"}}
        for part in parts
    ]
    evidence_checks = {
        "D343_typed_contract_inherited_without_retest": typed["pass"] and typed["d374_typed_scalar_retest_count"] == 0,
        "four_failure_causes_proven_from_frozen_sources": provenance_pass,
        "callback_source_identity_pass": callback_audit["pass"],
        "exact_16_plus_18": callback_audit["part_counts"] == BODY_COUNTS,
        "all_four_boards_exact_1920x1080": all(row["exact_1920x1080"] for row in boards.values()),
        "D373_inventory_immutable": d373_after_boards == prereg["d373_inventory_before"],
        "D334_sidecar_immutable": d334_after_boards == prereg["d334_sidecar_before"],
        "scientific_nulls_all_preserved": all(value is None for value in nulls.values()),
        "g0a_not_promoted": True,
    }
    evidence = {
        "artifact": "D374_D373_FAIL_STOP_PROVENANCE_AND_CALLBACK_VISUALIZATION_EVIDENCE_V1",
        "case": "g0a_d374",
        "attempt": OUT_DIR.name,
        "new_variables": NEW_VARIABLES,
        "what_and_why": "Explain the D373 fail-stop from immutable evidence and show the exact 16+18 callback collider shapes before any repaired live worker or physics comparison.",
        "D373_verdict_preserved": "D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP",
        "D373_raw_successes_preserved": _read_json(D373_FAIL)["raw_successes"],
        "failure_provenance": provenance,
        "later_live_repair_contract": {"path": _rel(REPAIR_PATH), "sha256": _sha(REPAIR_PATH), "pass": repair["pass"]},
        "callback_visual_source_audit": callback_audit,
        "callback_part_summaries": part_summaries,
        "visual_boards": boards,
        "immutable_inputs": {
            "frozen_hash_checks": frozen_checks,
            "D373_before": prereg["d373_inventory_before"],
            "D373_after_boards": d373_after_boards,
            "D373_exact": d373_after_boards == prereg["d373_inventory_before"],
            "D334_before": prereg["d334_sidecar_before"],
            "D334_after_boards": d334_after_boards,
            "D334_exact": d334_after_boards == prereg["d334_sidecar_before"],
        },
        "scope_counters": {
            "offline_audit_invocations": 1,
            "automatic_retries": 0,
            "D343_typed_float_retests": 0,
            "isaac_launches": 0,
            "physx_calls": 0,
            "usd_writes": 0,
            "physics_steps": 0,
            "q5_commands": 0,
            "q5_samples": 0,
            "contact_queries": 0,
            "cylinder_creates_or_writes": 0,
            "target_ik_path_changes": 0,
            "collider_regenerations": 0,
            "automatic_decomposition_sweeps": 0,
        },
        "scientific_or_live_identity_nulls": nulls,
        "g0a_pass": False,
        "checks": evidence_checks,
        "pass": all(evidence_checks.values()),
        "verdict": VERDICT_PASS if all(evidence_checks.values()) else VERDICT_FAIL,
        "next_authorization_boundary": "A separately approved repaired live identity worker must pass before any A64/P34 cylinder physics comparison.",
    }
    if not evidence["pass"]:
        raise RuntimeError(f"D374 evidence gate failed: {evidence_checks}")
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase("authoritative_offline_evidence_committed", evidence_sha256=_sha(EVIDENCE_PATH))

    rerun = _write_rerun(parts, evidence)
    _phase("save_only_rerun_and_single_headless_capture_complete", strict_validation_pass=rerun["strict_validation_pass"])
    d373_final = _inventory(D373_DIR)
    d334_final = _sidecar_snapshot()
    automated_checks = {
        "evidence_pass": evidence["pass"],
        "rerun_save_only_log_ok": rerun["save_only_log"]["ok"] is True,
        "rerun_strict_validation_pass": rerun["strict_validation_pass"],
        "rerun_viewer_exactly_one": rerun["headless_viewer_invocations"] == 1,
        "rerun_screenshot_exists": RERUN_PNG.is_file() and RERUN_PNG.stat().st_size > 0,
        "D373_immutable_after_all_outputs": d373_final == prereg["d373_inventory_before"],
        "D334_sidecar_immutable_after_all_outputs": d334_final == prereg["d334_sidecar_before"],
        "no_live_or_physics_scope": all(evidence["scope_counters"][key] == 0 for key in (
            "isaac_launches", "physx_calls", "usd_writes", "physics_steps", "q5_commands", "q5_samples",
            "contact_queries", "cylinder_creates_or_writes", "target_ik_path_changes", "collider_regenerations",
            "automatic_decomposition_sweeps",
        )),
    }
    automated = {
        "artifact": "D374_AUTOMATED_SUMMARY_V1",
        "evidence_path": _rel(EVIDENCE_PATH),
        "evidence_sha256": _sha(EVIDENCE_PATH),
        "repair_contract_path": _rel(REPAIR_PATH),
        "repair_contract_sha256": _sha(REPAIR_PATH),
        "boards": boards,
        "rerun": rerun,
        "D373_inventory_after_all_outputs": d373_final,
        "D334_sidecar_after_all_outputs": d334_final,
        "manual_visual_inspection": "pending",
        "completion_contract_pass": False,
        "checks": automated_checks,
        "pass": all(automated_checks.values()),
        "status": "AWAITING_MANUAL_ORIGINAL_RESOLUTION_INSPECTION",
        "scientific_verdict": None,
        "g0a_pass": False,
    }
    _write_json_x(AUTOMATED_PATH, automated)
    if not automated["pass"]:
        raise RuntimeError(f"D374 automated contract failed: {automated_checks}")
    _phase("run_complete_awaiting_manual_inspection", automated_summary_sha256=_sha(AUTOMATED_PATH))


def _finalize() -> None:
    for path in (PREREG_PATH, EVIDENCE_PATH, REPAIR_PATH, AUTOMATED_PATH, MANUAL_JSON_PATH, MANUAL_MD_PATH):
        if not path.is_file():
            raise RuntimeError(f"finalize prerequisite missing: {path}")
    if COMPLETION_PATH.exists():
        raise FileExistsError(f"forward-only completion already exists: {COMPLETION_PATH}")
    _phase("finalize_start")
    prereg = _read_json(PREREG_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    automated = _read_json(AUTOMATED_PATH)
    manual = _read_json(MANUAL_JSON_PATH)
    d373_after = _inventory(D373_DIR)
    d334_after = _sidecar_snapshot()
    expected_visual_hashes = {
        key: value["sha256"] for key, value in automated["boards"].items()
    }
    expected_visual_hashes["rerun_inspection"] = automated["rerun"]["screenshot"]["sha256"]
    checks = {
        "preregistration_pass": prereg["pass"] is True,
        "offline_evidence_pass": evidence["pass"] is True,
        "automated_summary_pass": automated["pass"] is True,
        "manual_original_resolution_inspection_pass": manual.get("pass") is True,
        "manual_hashes_exact": manual.get("inspected_sha256") == expected_visual_hashes,
        "D373_immutable": d373_after == prereg["d373_inventory_before"],
        "D334_sidecar_immutable": d334_after == prereg["d334_sidecar_before"],
        "scientific_nulls_preserved": all(value is None for value in evidence["scientific_or_live_identity_nulls"].values()),
        "g0a_false": evidence["g0a_pass"] is False and automated["g0a_pass"] is False,
    }
    completion = {
        "artifact": "D374_COMPLETION_SUMMARY_V1",
        "case": "g0a_d374",
        "attempt": OUT_DIR.name,
        "new_variables": NEW_VARIABLES,
        "preregistration": {"path": _rel(PREREG_PATH), "sha256": _sha(PREREG_PATH)},
        "evidence": {"path": _rel(EVIDENCE_PATH), "sha256": _sha(EVIDENCE_PATH)},
        "repair_contract": {"path": _rel(REPAIR_PATH), "sha256": _sha(REPAIR_PATH)},
        "automated_summary": {"path": _rel(AUTOMATED_PATH), "sha256": _sha(AUTOMATED_PATH)},
        "manual_inspection": {"path": _rel(MANUAL_JSON_PATH), "sha256": _sha(MANUAL_JSON_PATH), "report": _rel(MANUAL_MD_PATH)},
        "visual_artifacts": {**automated["boards"], "rerun_inspection": automated["rerun"]["screenshot"]},
        "rrd": automated["rerun"]["rrd"],
        "rbl": automated["rerun"]["rbl"],
        "scope_counters": evidence["scope_counters"],
        "D373_verdict_preserved": evidence["D373_verdict_preserved"],
        "scientific_or_live_identity_nulls": evidence["scientific_or_live_identity_nulls"],
        "g0a_pass": False,
        "checks": checks,
        "pass": all(checks.values()),
        "verdict": VERDICT_PASS if all(checks.values()) else VERDICT_FAIL,
        "next_authorization_boundary": evidence["next_authorization_boundary"],
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase("finalize_complete", completion_sha256=_sha(COMPLETION_PATH), verdict=completion["verdict"])
    if not completion["pass"]:
        raise RuntimeError(f"D374 completion contract failed: {checks}")


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
            "artifact": "D374_RUNTIME_EXCEPTION_V1",
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
                _phase("exception", stage=args.stage, exception_type=type(exc).__name__)
        except Exception:
            pass
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
