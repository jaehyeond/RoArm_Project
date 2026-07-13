#!/usr/bin/env python3
"""D343 proof-only USD typed-float/readback contract repair.

This harness reads only the authored ``minThickness`` scalar attribute on all
128 immutable D339 attempt2 parts.  The 13 parts implicated by D342 are also
cross-joined to the immutable D342 evidence.  Expanding coverage from that
failure subset to the full authored field does not add a variable or change a
parameter; it prevents a future attempt3 from inheriting an unverified scalar
on any of the 115 retained parts.  The harness does not read mesh geometry,
start Isaac Kit, create a SimulationContext, cook collision data, author USD,
advance physics, or create a Rerun recording.

The primary identity gate compares the actual USD ``float`` bit pattern with
``np.float32(requested_value)``.  The already frozen 1e-10 m tolerance is kept
only as a compatibility/root-cause diagnostic because it also admits adjacent
float32 values.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from pxr import Sdf, Usd, UsdGeom


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d343"
PREREG_PATH = OUT_DIR / "d343_preregistration.json"
PARAMETER_AUDIT_PATH = OUT_DIR / "d343_parameter_freeze_audit.json"
RERUN_OMISSION_PATH = OUT_DIR / "d343_rerun_omission_justification.json"
EVIDENCE_PATH = OUT_DIR / "d343_usd_typed_float_readback_evidence.json"
SUMMARY_PATH = OUT_DIR / "d343_usd_typed_float_readback_summary.json"
REPORT_PATH = OUT_DIR / "d343_usd_typed_float_readback_report.md"

D339_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d339"
D339_ATTEMPT2 = D339_DIR / "collision_asset/attempt2"
D339_PHYSICS = (
    D339_ATTEMPT2
    / "roarm_m3_fullmesh_convex_parts/configuration/roarm_m3_physics.usd"
)
D339_HULL_MANIFEST = D339_ATTEMPT2 / "d339_hull_manifest.json"
D339_LIVE_AUDIT = D339_DIR / "d339_live_collider_audit.json"
D339_HARNESS = (
    REPO_ROOT
    / "sim_scripts/cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair.py"
)

D340_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d340"
D340_PARAMETER_AUDIT = D340_DIR / "d340_parameter_freeze_audit.json"
D340_ATTEMPT3 = D340_DIR / "collision_asset/attempt3"
D340_HARNESS = (
    REPO_ROOT
    / "sim_scripts/cyl34_top_view_d340_grasp_g0a_fixed_point_live_authoring_repair.py"
)

D342_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d342"
D342_EVIDENCE = D342_DIR / "d342_authored_coordinate_stream_evidence.json"
D342_COMPLETION = D342_DIR / "d342_completion_summary.json"
D342_ROOT_CAUSE = D342_DIR / "d342_postrun_root_cause_audit.json"
D342_PARAMETER_AUDIT = D342_DIR / "d342_parameter_freeze_audit.json"
D342_RRD = D342_DIR / "d342_authored_coordinate_stream.rrd"
D342_HARNESS = (
    REPO_ROOT
    / "sim_scripts/cyl34_top_view_d342_grasp_g0a_authored_coordinate_stream_repair.py"
)

D343_SESSION = (
    REPO_ROOT
    / "claudedocs/session_20260713_grasp_g0a_d343_usd_typed_float_readback_contract_repair.md"
)
START_HERE = REPO_ROOT / "START_HERE.md"

USD_CORE_EXT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
PHYSX_SCHEMA_FILE = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "plugins/PhysxSchema/resources/schema.usda"
)
REGISTERED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
REGISTERED_PYTHONPATH = str(USD_CORE_EXT)
REGISTERED_LD_LIBRARY_PATH = ":".join(
    (
        "/home/cgxr/miniconda3/envs/isaaclab/lib",
        str(USD_CORE_EXT / "bin"),
    )
)

NEW_VARIABLES = ["usd_float_parameter_readback_contract"]
ATTR_NAME = "physxConvexHullCollision:minThickness"
API_SCHEMA_NAME = "PhysxConvexHullCollisionAPI"
EXPECTED_TYPE_NAME = "float"
EXPECTED_SCHEMA_DEFAULT_M = 0.001

D342_FAILING_PARTS = {
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
ALL_PARTS = {
    body: tuple(f"part_{index:03d}" for index in range(64))
    for body in ("link5", "gripper_link")
}
PART_KEYS = tuple(
    (body, part)
    for body in ("link5", "gripper_link")
    for part in ALL_PARTS[body]
)
PART_COUNT = len(PART_KEYS)
D342_PART_KEYS = tuple(
    (body, part)
    for body in ("link5", "gripper_link")
    for part in D342_FAILING_PARTS[body]
)
D342_PART_SET = set(D342_PART_KEYS)
D342_PART_COUNT = len(D342_PART_KEYS)

VERDICT_PASS = "D343_USD_TYPED_FLOAT_READBACK_CONTRACT_PASS"
VERDICT_FAIL = "D343_USD_TYPED_FLOAT_READBACK_CONTRACT_FAIL_STOP"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(value, encoding="utf-8")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": _relative(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return rows


def _inventory_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _f32_bits(value: float | np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _f32_hex(value: float | np.float32) -> str:
    return f"0x{_f32_bits(value):08x}"


def _f32_le_bytes_hex(value: float | np.float32) -> str:
    return struct.pack("<f", float(value)).hex()


def _source_hashes() -> dict[str, str]:
    paths = {
        "d343_harness": Path(__file__).resolve(),
        "d339_physics_usd": D339_PHYSICS,
        "d339_hull_manifest": D339_HULL_MANIFEST,
        "d339_live_audit": D339_LIVE_AUDIT,
        "d339_harness": D339_HARNESS,
        "d340_parameter_audit": D340_PARAMETER_AUDIT,
        "d340_harness": D340_HARNESS,
        "d342_evidence": D342_EVIDENCE,
        "d342_completion": D342_COMPLETION,
        "d342_root_cause": D342_ROOT_CAUSE,
        "d342_parameter_audit": D342_PARAMETER_AUDIT,
        "d342_harness": D342_HARNESS,
        "d342_rrd_context_only": D342_RRD,
        "pxr_physx_schema": PHYSX_SCHEMA_FILE,
        "pxr_usd_module": USD_CORE_EXT / "pxr/Usd/_usd.so",
        "pxr_sdf_module": USD_CORE_EXT / "pxr/Sdf/_sdf.so",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _requested_value_and_tolerance() -> tuple[float, float]:
    requested = float(_json(D339_HULL_MANIFEST)["decomposition_parameters"]["min_thickness_m"])
    tolerance = float(
        _json(D342_PARAMETER_AUDIT)["frozen_parameters"]["readback_tolerances"][
            "min_thickness_m"
        ]
    )
    return requested, tolerance


def _schema_contract() -> dict[str, Any]:
    text = PHYSX_SCHEMA_FILE.read_text(encoding="utf-8")
    match = re.search(
        r"float\s+physxConvexHullCollision:minThickness\s*=\s*([0-9.eE+-]+)",
        text,
    )
    if match is None:
        raise RuntimeError("cannot locate minThickness declaration in pinned PhysX schema")
    units_match = re.search(
        r"float\s+physxConvexHullCollision:minThickness\s*=.*?Units:\s*([^\"\n]+)",
        text,
        flags=re.DOTALL,
    )
    if units_match is None:
        raise RuntimeError("cannot locate minThickness units in pinned PhysX schema")
    default_requested = float(match.group(1))
    default_typed = np.float32(default_requested)
    return {
        "declared_type": "float",
        "default_requested_m": default_requested,
        "default_typed_m": float(default_typed),
        "default_float32_bits_uint32": _f32_bits(default_typed),
        "default_float32_bits_hex": _f32_hex(default_typed),
        "default_float32_le_bytes_hex": _f32_le_bytes_hex(default_typed),
        "documented_units": units_match.group(1).strip(),
        "path": str(PHYSX_SCHEMA_FILE),
        "sha256": _sha256(PHYSX_SCHEMA_FILE),
    }


def _d342_anchor_checks() -> dict[str, Any]:
    evidence = _json(D342_EVIDENCE)
    completion = _json(D342_COMPLETION)
    part_map = {(row["body"], row["name"]): row for row in evidence["parts"]}
    required_direct_true = (
        "direct_points_f32_array_exact_cold1",
        "direct_points_f32_bytes_exact_preregistered",
        "cold1_cold2_points_f32_exact",
        "direct_triangles_exact_cold1",
        "cold1_cold2_triangles_exact",
        "direct_vertex_stream_hash_matches_manifest",
        "direct_topology_hash_matches_manifest",
        "direct_geometry_hash_matches_manifest",
    )
    per_part = {}
    for key in D342_PART_KEYS:
        row = part_map[key]
        checks = row["direct_checks"]
        false_keys = sorted(name for name, passed in checks.items() if not bool(passed))
        per_part[f"{key[0]}/{key[1]}"] = {
            "registered_part_found": True,
            "required_direct_hash_checks_true": all(checks[name] for name in required_direct_true),
            "sole_false_direct_predicate_is_min_thickness": false_keys
            == ["min_thickness_frozen_1e_4m"],
            "numeric_pass": row["numeric_pass"] is True,
        }
    checks = {
        "completion_verdict_preserved": completion.get("verdict")
        == "D342_AUTHORED_COORDINATE_STREAM_HARNESS_TOLERANCE_DRIFT_FAIL_STOP",
        "d342_completion_pass_false": completion.get("completion_contract_pass") is False,
        "d342_direct_registered_pass_zero": evidence.get("direct_pass_count") == 0,
        "d342_numeric_pass_13": evidence.get("numeric_pass_count") == D342_PART_COUNT,
        "d342_legacy_negative_13": evidence.get("legacy_mixed_stream_rejected_count")
        == D342_PART_COUNT,
        "all_registered_parts_exact_anchor": all(
            all(row.values()) for row in per_part.values()
        ),
    }
    return {"checks": checks, "per_part": per_part, "pass": all(checks.values())}


def _preflight() -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    allowed = {
        PREREG_PATH.name,
        PARAMETER_AUDIT_PATH.name,
        RERUN_OMISSION_PATH.name,
    }
    observed = {path.name for path in OUT_DIR.iterdir()}
    if observed != allowed:
        raise RuntimeError(f"D343 output folder is not pristine: {sorted(observed)}")

    prereg = _json(PREREG_PATH)
    parameter_audit = _json(PARAMETER_AUDIT_PATH)
    omission = _json(RERUN_OMISSION_PATH)
    requested, frozen_tolerance = _requested_value_and_tolerance()
    expected_typed = np.float32(requested)
    d339_before = _inventory(D339_ATTEMPT2)
    d340_before = _inventory(D340_DIR)
    d342_before = _inventory(D342_DIR)
    d342_anchor = _d342_anchor_checks()
    d340_tolerance = float(
        _json(D340_PARAMETER_AUDIT)["frozen_parameters"]["readback_tolerances"][
            "min_thickness_m"
        ]
    )
    schema_contract = _schema_contract()
    manifest_parts = _json(D339_HULL_MANIFEST)["parts"]

    checks = {
        "artifact": prereg.get("artifact") == "D343_USD_TYPED_FLOAT_PREREGISTRATION_V1",
        "status": prereg.get("status") == "PRE_RUNTIME_LOCKED",
        "new_variables": prereg.get("new_variables") == NEW_VARIABLES,
        "variable_count": prereg.get("variable_count") == 1,
        "git_head": prereg.get("git_head") == _git_head(),
        "registered_command": bool(
            prereg.get("registered_command")
            == {
                "python": str(REGISTERED_PYTHON),
                "script": _relative(Path(__file__).resolve()),
                "argv": [],
                "environment": {
                    "PYTHONPATH": REGISTERED_PYTHONPATH,
                    "LD_LIBRARY_PATH": REGISTERED_LD_LIBRARY_PATH,
                },
            }
            and Path(sys.executable).resolve() == REGISTERED_PYTHON.resolve()
            and os.environ.get("PYTHONPATH") == REGISTERED_PYTHONPATH
            and os.environ.get("LD_LIBRARY_PATH") == REGISTERED_LD_LIBRARY_PATH
        ),
        "source_hashes": prereg.get("source_hashes") == _source_hashes(),
        "session_sha256": prereg.get("d343_session_sha256") == _sha256(D343_SESSION),
        "start_here_sha256": prereg.get("start_here_sha256") == _sha256(START_HERE),
        "parameter_audit_sha256": prereg.get("parameter_audit_sha256")
        == _sha256(PARAMETER_AUDIT_PATH),
        "rerun_omission_sha256": prereg.get("rerun_omission_sha256")
        == _sha256(RERUN_OMISSION_PATH),
        "d339_inventory": bool(
            prereg.get("d339_attempt2_file_count") == len(d339_before)
            and prereg.get("d339_attempt2_inventory_digest")
            == _inventory_digest(d339_before)
        ),
        "d340_inventory": bool(
            prereg.get("d340_file_count") == len(d340_before)
            and prereg.get("d340_inventory_digest") == _inventory_digest(d340_before)
        ),
        "d342_inventory": bool(
            prereg.get("d342_file_count") == len(d342_before)
            and prereg.get("d342_inventory_digest") == _inventory_digest(d342_before)
        ),
        "subject_count": prereg.get("subject_count") == PART_COUNT == 128,
        "part_allowlist": prereg.get("part_allowlist")
        == {
            "bodies_in_order": ["link5", "gripper_link"],
            "part_name_format": "part_%03d",
            "index_range_inclusive": [0, 63],
            "per_body_count": 64,
        },
        "manifest_part_allowlist": all(
            [row["name"] for row in manifest_parts[body]] == list(ALL_PARTS[body])
            for body in ("link5", "gripper_link")
        ),
        "d342_failure_subset_allowlist": prereg.get("d342_failure_subset_allowlist")
        == {
            body: list(D342_FAILING_PARTS[body])
            for body in ("link5", "gripper_link")
        },
        "d342_failure_subset_count": prereg.get("d342_failure_subset_count")
        == D342_PART_COUNT
        == 13,
        "requested_value": prereg.get("requested_value_m") == requested == 0.0001,
        "expected_typed_value": prereg.get("expected_float32_value_m")
        == float(expected_typed),
        "expected_typed_uint32": prereg.get("expected_float32_bits_uint32")
        == _f32_bits(expected_typed)
        == 953267991,
        "expected_typed_bits": prereg.get("expected_float32_bits_hex")
        == _f32_hex(expected_typed)
        == "0x38d1b717",
        "expected_typed_le_bytes": prereg.get("expected_float32_le_bytes_hex")
        == _f32_le_bytes_hex(expected_typed)
        == "17b7d138",
        "frozen_compatibility_tolerance": prereg.get("frozen_compatibility_tolerance_m")
        == frozen_tolerance
        == d340_tolerance
        == 1.0e-10,
        "schema_default": prereg.get("schema_default_m")
        == schema_contract["default_requested_m"]
        == EXPECTED_SCHEMA_DEFAULT_M,
        "schema_default_bits": prereg.get("schema_default_float32_bits_hex")
        == schema_contract["default_float32_bits_hex"]
        == "0x3a83126f",
        "schema_units": prereg.get("schema_documented_units")
        == schema_contract["documented_units"]
        == "distance",
        "numpy_pin": str(np.__version__) == prereg.get("numpy_version") == "1.26.0",
        "psutil_pin": str(psutil.__version__) == prereg.get("psutil_version") == "5.9.8",
        "pxr_version": list(Usd.GetVersion()) == prereg.get("pxr_usd_version"),
        "parameter_audit": bool(
            parameter_audit.get("artifact") == "D343_PARAMETER_FREEZE_AUDIT_V1"
            and parameter_audit.get("pass") is True
            and parameter_audit.get("new_variables") == NEW_VARIABLES
            and parameter_audit.get("physical_variables_changed") == []
            and parameter_audit.get("existing_parameter_increases") == []
            and parameter_audit.get("existing_parameter_changes") == []
            and parameter_audit.get("threshold_relaxations") == []
            and parameter_audit.get("subject_count") == PART_COUNT
            and parameter_audit.get("coverage_expansion_is_new_variable") is False
        ),
        "rerun_omission": bool(
            omission.get("artifact") == "D343_RERUN_OMISSION_JUSTIFICATION_V1"
            and omission.get("pass") is True
            and omission.get("spatial_inference") is False
            and omission.get("temporal_inference") is False
            and omission.get("new_rerun_artifacts") == []
            and omission.get("d342_rrd_reused_as_d343_completion_artifact") is False
        ),
        "d342_anchor": d342_anchor["pass"],
        "attempt3_absent": not D340_ATTEMPT3.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D343 preregistration gate failed: {checks}")
    return prereg, checks, d339_before, d340_before, d342_before


def _live_audit_rows() -> dict[tuple[str, str], dict[str, Any]]:
    audit = _json(D339_LIVE_AUDIT)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for body in ("link5", "gripper_link"):
        for row in audit["per_body"][body]["part_checks"]:
            rows[(body, Path(row["path"]).name)] = row
    return rows


def _adjacent_negative(requested: float, tolerance: float) -> dict[str, Any]:
    expected = np.float32(requested)
    lower = np.nextafter(expected, np.float32(-np.inf), dtype=np.float32)
    upper = np.nextafter(expected, np.float32(np.inf), dtype=np.float32)
    expected_bits = _f32_bits(expected)
    rows = []
    for name, value in (("lower_adjacent", lower), ("upper_adjacent", upper)):
        bits = _f32_bits(value)
        exact_bit_validator_accepts = bits == expected_bits
        tolerance_validator_accepts = math.isclose(
            float(value), requested, rel_tol=0.0, abs_tol=tolerance
        )
        rows.append(
            {
                "name": name,
                "value_m": float(value),
                "bits_uint32": bits,
                "bits_hex": f"0x{bits:08x}",
                "little_endian_bytes_hex": _f32_le_bytes_hex(value),
                "signed_ulp_delta_m": float(value - expected),
                "absolute_delta_from_requested_m": abs(float(value) - requested),
                "exact_bit_validator_accepts": exact_bit_validator_accepts,
                "exact_bit_validator_rejected": not exact_bit_validator_accepts,
                "frozen_tolerance_validator_accepts": tolerance_validator_accepts,
            }
        )
    passed = all(
        row["exact_bit_validator_rejected"]
        and row["frozen_tolerance_validator_accepts"]
        for row in rows
    )
    return {
        "kind": "in-memory adjacent-float32 discriminator",
        "source_file_written": False,
        "expected_value_m": float(expected),
        "expected_bits_uint32": expected_bits,
        "expected_bits_hex": _f32_hex(expected),
        "expected_little_endian_bytes_hex": _f32_le_bytes_hex(expected),
        "one_ulp_m": float(upper - expected),
        "frozen_tolerance_m": tolerance,
        "rows": rows,
        "pass": passed,
        "interpretation": (
            "Exact typed bits reject both adjacent floats while the historical 1e-10m "
            "compatibility tolerance accepts them; tolerance cannot prove typed identity."
        ),
    }


def _d342_comparator_reproduction(requested: float, frozen_tolerance: float) -> dict[str, Any]:
    root_cause = _json(D342_ROOT_CAUSE)["min_thickness_contract_audit"]
    executed_tolerance = float(root_cause["d342_harness_applied_tolerance_m"])
    typed = np.float32(requested)
    delta = abs(float(typed) - requested)
    frozen_accepts = math.isclose(
        float(typed), requested, rel_tol=0.0, abs_tol=frozen_tolerance
    )
    executed_accepts = math.isclose(
        float(typed), requested, rel_tol=0.0, abs_tol=executed_tolerance
    )
    checks = {
        "root_cause_readback_matches_typed_value": float(
            root_cause["immutable_d339_float_readback_m"]
        )
        == float(typed),
        "root_cause_delta_matches": float(
            root_cause["absolute_float32_representation_delta_m"]
        )
        == delta,
        "root_cause_frozen_tolerance_matches": float(
            root_cause["registered_d339_d340_readback_tolerance_m"]
        )
        == frozen_tolerance,
        "executed_tolerance_is_1e_12": executed_tolerance == 1.0e-12,
        "correct_typed_value_passes_frozen_1e_10": frozen_accepts,
        "correct_typed_value_fails_executed_1e_12": not executed_accepts,
    }
    return {
        "requested_value_m": requested,
        "correct_typed_value_m": float(typed),
        "correct_typed_bits_hex": _f32_hex(typed),
        "representation_delta_m": delta,
        "frozen_tolerance_m": frozen_tolerance,
        "frozen_tolerance_accepts": frozen_accepts,
        "d342_executed_tolerance_m": executed_tolerance,
        "d342_executed_tolerance_accepts": executed_accepts,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _immutability(
    d339_before: list[dict[str, Any]],
    d340_before: list[dict[str, Any]],
    d342_before: list[dict[str, Any]],
) -> dict[str, Any]:
    d339_after = _inventory(D339_ATTEMPT2)
    d340_after = _inventory(D340_DIR)
    d342_after = _inventory(D342_DIR)
    return {
        "d339_file_count_before_after": [len(d339_before), len(d339_after)],
        "d339_digest_before_after": [
            _inventory_digest(d339_before),
            _inventory_digest(d339_after),
        ],
        "d339_exact": d339_before == d339_after,
        "d340_file_count_before_after": [len(d340_before), len(d340_after)],
        "d340_digest_before_after": [
            _inventory_digest(d340_before),
            _inventory_digest(d340_after),
        ],
        "d340_exact": d340_before == d340_after,
        "d342_file_count_before_after": [len(d342_before), len(d342_after)],
        "d342_digest_before_after": [
            _inventory_digest(d342_before),
            _inventory_digest(d342_after),
        ],
        "d342_exact": d342_before == d342_after,
        "pass": d339_before == d339_after
        and d340_before == d340_after
        and d342_before == d342_after,
    }


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# D343 USD typed-float/readback contract report",
            "",
            f"- Verdict: `{summary['verdict']}`",
            f"- Typed attribute passes: `{summary['typed_attribute_pass_count']}/{summary['part_count']}`",
            f"- D342 failure-subset anchors: `{summary['d342_subset_anchor_pass_count']}/13`",
            f"- Expected float32 bits: `{summary['expected_float32_bits_hex']}`",
            f"- Adjacent-float32 discriminator: `{summary['adjacent_negative_pass']}`",
            f"- D342 comparator reproduction: `{summary['d342_comparator_reproduction_pass']}`",
            f"- D339/D340/D342 immutable: `{summary['immutability']['pass']}`",
            "- New Rerun: omitted by the registered pure scalar/schema/bit exception.",
            "- Attempt3 remains absent; D342 verdict and g0a_pass remain unchanged.",
            "",
        ]
    )


def main() -> int:
    prereg, prereg_checks, d339_before, d340_before, d342_before = _preflight()
    requested = float(prereg["requested_value_m"])
    tolerance = float(prereg["frozen_compatibility_tolerance_m"])
    expected_typed = np.float32(requested)
    expected_bits = _f32_bits(expected_typed)
    schema_contract = _schema_contract()
    schema_default = float(schema_contract["default_requested_m"])
    schema_default_bits = int(schema_contract["default_float32_bits_uint32"])
    live_rows = _live_audit_rows()
    d342_anchor = _d342_anchor_checks()

    direct_layer = Sdf.Layer.FindOrOpen(str(D339_PHYSICS))
    if direct_layer is None:
        raise RuntimeError(f"failed to open immutable D339 layer via Sdf: {D339_PHYSICS}")
    direct_layer_real_path = str(Path(direct_layer.realPath).resolve())
    direct_layer_dirty_before = bool(direct_layer.dirty)
    direct_units_authored = direct_layer.pseudoRoot.HasInfo("metersPerUnit")
    direct_meters_per_unit = (
        float(direct_layer.pseudoRoot.GetInfo("metersPerUnit"))
        if direct_units_authored
        else float("nan")
    )

    stage = Usd.Stage.Open(str(D339_PHYSICS), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open immutable D339 stage: {D339_PHYSICS}")
    stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    stage_root_real_path = str(Path(stage.GetRootLayer().realPath).resolve())

    part_rows: list[dict[str, Any]] = []
    for part_idx, (body, part) in enumerate(PART_KEYS):
        prim_path = f"/colliders/{body}/d338_convex_parts/{part}"
        attribute_path = Sdf.Path(prim_path).AppendProperty(ATTR_NAME)
        direct_prim_spec = direct_layer.GetPrimAtPath(prim_path)
        direct_api_list_op = (
            direct_prim_spec.GetInfo("apiSchemas")
            if direct_prim_spec is not None and direct_prim_spec.HasInfo("apiSchemas")
            else None
        )
        direct_api_schemas = (
            [str(item) for item in direct_api_list_op.GetAppliedItems()]
            if direct_api_list_op is not None
            else []
        )
        direct_spec = direct_layer.GetAttributeAtPath(attribute_path)
        direct_spec_valid = direct_spec is not None
        direct_default_authored = bool(
            direct_spec_valid
            and direct_spec.HasInfo(Sdf.AttributeSpec.DefaultValueKey)
        )
        direct_value = (
            direct_spec.GetInfo(Sdf.AttributeSpec.DefaultValueKey)
            if direct_default_authored
            else None
        )
        direct_float = float(direct_value) if direct_value is not None else float("nan")
        direct_bits = _f32_bits(direct_float) if math.isfinite(direct_float) else -1

        prim = stage.GetPrimAtPath(prim_path)
        attr = prim.GetAttribute(ATTR_NAME) if prim.IsValid() else Usd.Attribute()
        value = attr.Get() if attr else None
        actual_value = float(value) if value is not None else float("nan")
        actual_bits = _f32_bits(actual_value) if math.isfinite(actual_value) else -1
        resolve_info = attr.GetResolveInfo() if attr else None
        resolve_source = resolve_info.GetSource() if resolve_info else None
        resolve_source_text = str(resolve_source) if resolve_source is not None else None
        value_is_blocked = bool(resolve_info.ValueIsBlocked()) if resolve_info else False
        property_stack = []
        if attr:
            for spec in attr.GetPropertyStack():
                layer_real_path = str(Path(spec.layer.realPath).resolve()) if spec.layer.realPath else ""
                spec_default_authored = spec.HasInfo(Sdf.AttributeSpec.DefaultValueKey)
                spec_value = (
                    spec.GetInfo(Sdf.AttributeSpec.DefaultValueKey)
                    if spec_default_authored
                    else None
                )
                spec_float = float(spec_value) if spec_value is not None else float("nan")
                property_stack.append(
                    {
                        "layer_identifier": str(spec.layer.identifier),
                        "layer_real_path": layer_real_path,
                        "spec_path": str(spec.path),
                        "type_name": str(spec.typeName),
                        "default_field_authored": spec_default_authored,
                        "default_value_m": spec_float,
                        "default_float32_bits_hex": (
                            _f32_hex(spec_float) if math.isfinite(spec_float) else None
                        ),
                    }
                )
        live = live_rows[(body, part)]
        live_value = float(live["hull_min_thickness_readback_m"])
        is_d342_subset = (body, part) in D342_PART_SET
        d342_part_anchor_pass = (
            all(d342_anchor["per_part"][f"{body}/{part}"].values())
            if is_d342_subset
            else True
        )
        checks = {
            "prim_valid": prim.IsValid(),
            "direct_sdf_prim_spec_exists": direct_prim_spec is not None,
            "direct_sdf_api_schema_authored": API_SCHEMA_NAME in direct_api_schemas,
            "direct_sdf_attribute_spec_exists": direct_spec_valid,
            "direct_sdf_spec_path_exact": direct_spec_valid
            and str(direct_spec.path) == str(attribute_path),
            "direct_sdf_type_float": direct_spec_valid
            and direct_spec.typeName == Sdf.ValueTypeNames.Float,
            "direct_sdf_default_field_authored": direct_default_authored,
            "direct_sdf_default_bits_exact_expected": direct_bits == expected_bits,
            "attribute_valid": bool(attr),
            "attribute_name_exact": bool(attr) and str(attr.GetName()) == ATTR_NAME,
            "attribute_type_float": bool(attr)
            and str(attr.GetTypeName()) == EXPECTED_TYPE_NAME
            and attr.GetTypeName() == Sdf.ValueTypeNames.Float,
            "authored_value_opinion": bool(attr) and attr.HasAuthoredValueOpinion(),
            "authored_value": bool(attr) and attr.HasAuthoredValue(),
            "resolve_source_authored_default": resolve_source == Usd.ResolveInfoSourceDefault,
            "resolve_source_not_schema_fallback": resolve_source
            != Usd.ResolveInfoSourceFallback,
            "resolved_value_not_blocked": resolve_info is not None and not value_is_blocked,
            "zero_time_samples": bool(attr) and attr.GetNumTimeSamples() == 0,
            "not_time_varying": bool(attr) and not attr.ValueMightBeTimeVarying(),
            "property_stack_exactly_one": len(property_stack) == 1,
            "property_stack_is_direct_immutable_spec": len(property_stack) == 1
            and property_stack[0]["layer_real_path"] == str(D339_PHYSICS.resolve())
            and property_stack[0]["spec_path"] == str(attribute_path)
            and property_stack[0]["type_name"] == EXPECTED_TYPE_NAME
            and property_stack[0]["default_field_authored"] is True
            and property_stack[0]["default_float32_bits_hex"]
            == f"0x{expected_bits:08x}",
            "finite_typed_readback": math.isfinite(actual_value),
            "actual_bits_exact_expected_float32": actual_bits == expected_bits,
            "actual_value_exact_expected_float32": actual_value == float(expected_typed),
            "frozen_tolerance_compatibility": math.isclose(
                actual_value, requested, rel_tol=0.0, abs_tol=tolerance
            ),
            "schema_default_bits_not_used": actual_bits != schema_default_bits,
            "direct_stage_meters_per_unit_authored": direct_units_authored,
            "direct_stage_meters_per_unit_one": math.isclose(
                direct_meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12
            ),
            "composed_stage_meters_per_unit_one": math.isclose(
                stage_meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12
            ),
            "schema_documents_distance_units": schema_contract["documented_units"]
            == "distance",
            "d339_live_readback_check_true": live["checks"][
                "hull_min_thickness_readback"
            ]
            is True,
            "d339_live_value_bits_exact": _f32_bits(live_value) == expected_bits,
            "d342_anchor_if_applicable": d342_part_anchor_pass,
        }
        part_rows.append(
            {
                "part_idx": part_idx,
                "body": body,
                "name": part,
                "prim_path": prim_path,
                "attribute_path": str(attribute_path),
                "is_d342_failure_subset": is_d342_subset,
                "direct_sdf": {
                    "prim_spec_exists": direct_prim_spec is not None,
                    "authored_api_schemas": direct_api_schemas,
                    "physx_api_schema_authored": API_SCHEMA_NAME in direct_api_schemas,
                    "spec_exists": direct_spec_valid,
                    "spec_path": str(direct_spec.path) if direct_spec_valid else None,
                    "type_name": str(direct_spec.typeName) if direct_spec_valid else None,
                    "default_field_authored": direct_default_authored,
                    "default_value_m": direct_float,
                    "default_float32_bits_uint32": direct_bits if direct_bits >= 0 else None,
                    "default_float32_bits_hex": (
                        f"0x{direct_bits:08x}" if direct_bits >= 0 else None
                    ),
                    "default_float32_le_bytes_hex": (
                        _f32_le_bytes_hex(direct_float)
                        if math.isfinite(direct_float)
                        else None
                    ),
                },
                "attribute_name": str(attr.GetName()) if attr else None,
                "core_registered_applied_schemas": (
                    [str(item) for item in prim.GetAppliedSchemas()]
                    if prim.IsValid()
                    else []
                ),
                "core_registered_schema_list_is_identity_authority": False,
                "attribute_type_name": str(attr.GetTypeName()) if attr else None,
                "has_authored_value_opinion": bool(attr) and attr.HasAuthoredValueOpinion(),
                "has_authored_value": bool(attr) and attr.HasAuthoredValue(),
                "resolve_source": resolve_source_text,
                "resolve_source_is_authored_default": resolve_source
                == Usd.ResolveInfoSourceDefault,
                "resolve_source_is_schema_fallback": resolve_source
                == Usd.ResolveInfoSourceFallback,
                "value_is_blocked": value_is_blocked,
                "time_sample_count": int(attr.GetNumTimeSamples()) if attr else None,
                "value_might_be_time_varying": (
                    bool(attr.ValueMightBeTimeVarying()) if attr else None
                ),
                "requested_value_m": requested,
                "expected_typed_value_m": float(expected_typed),
                "expected_float32_bits_uint32": expected_bits,
                "expected_float32_bits_hex": f"0x{expected_bits:08x}",
                "expected_float32_le_bytes_hex": _f32_le_bytes_hex(expected_typed),
                "actual_typed_readback_m": actual_value,
                "actual_float32_bits_uint32": actual_bits if actual_bits >= 0 else None,
                "actual_float32_bits_hex": f"0x{actual_bits:08x}" if actual_bits >= 0 else None,
                "actual_float32_le_bytes_hex": (
                    _f32_le_bytes_hex(actual_value) if math.isfinite(actual_value) else None
                ),
                "decimal_delta_m": abs(actual_value - requested),
                "frozen_compatibility_tolerance_m": tolerance,
                "schema_default_requested_m": schema_default,
                "schema_default_typed_m": schema_contract["default_typed_m"],
                "schema_default_float32_bits_hex": schema_contract[
                    "default_float32_bits_hex"
                ],
                "property_stack": property_stack,
                "d339_live_readback_m": live_value,
                "d339_live_readback_float32_bits_hex": _f32_hex(live_value),
                "d342_failure_subset_anchor_pass": (
                    d342_part_anchor_pass if is_d342_subset else None
                ),
                "checks": checks,
                "pass": all(checks.values()),
            }
        )

    direct_layer_dirty_after = bool(direct_layer.dirty)
    stage = None
    direct_layer = None
    adjacent_negative = _adjacent_negative(requested, tolerance)
    d342_comparator_reproduction = _d342_comparator_reproduction(requested, tolerance)
    immutability = _immutability(d339_before, d340_before, d342_before)
    typed_pass_count = sum(row["pass"] for row in part_rows)
    d342_subset_anchor_pass_count = sum(
        row["d342_failure_subset_anchor_pass"] is True
        for row in part_rows
        if row["is_d342_failure_subset"]
    )
    global_checks = {
        "part_subject_exact_128": len(part_rows) == PART_COUNT == 128,
        "live_audit_rows_exact_128": set(live_rows) == set(PART_KEYS),
        "sdf_layer_is_immutable_d339_physics": direct_layer_real_path
        == str(D339_PHYSICS.resolve()),
        "usd_root_layer_is_immutable_d339_physics": stage_root_real_path
        == str(D339_PHYSICS.resolve()),
        "direct_meters_per_unit_authored": direct_units_authored,
        "direct_meters_per_unit_one": math.isclose(
            direct_meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "composed_meters_per_unit_one": math.isclose(
            stage_meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "schema_declares_float": schema_contract["declared_type"] == "float",
        "schema_documents_distance_units": schema_contract["documented_units"]
        == "distance",
        "schema_fallback_bits_distinct": schema_default_bits != expected_bits,
        "direct_layer_not_dirty_before": not direct_layer_dirty_before,
        "direct_layer_not_dirty_after": not direct_layer_dirty_after,
        "d342_subset_anchor_exact_13": d342_subset_anchor_pass_count
        == D342_PART_COUNT
        == 13,
    }
    scientific_pass = bool(
        typed_pass_count == PART_COUNT
        and all(global_checks.values())
        and adjacent_negative["pass"]
        and d342_comparator_reproduction["pass"]
        and d342_anchor["pass"]
        and immutability["pass"]
        and not D340_ATTEMPT3.exists()
    )
    verdict = VERDICT_PASS if scientific_pass else VERDICT_FAIL

    evidence = {
        "artifact": "D343_USD_TYPED_FLOAT_READBACK_EVIDENCE_V1",
        "case": "g0a_d343",
        "verdict": verdict,
        "pass": scientific_pass,
        "new_variables": NEW_VARIABLES,
        "scientific_subject": {
            "kind": "USD authored scalar attribute type/value/bit identity",
            "attribute": ATTR_NAME,
            "part_count": PART_COUNT,
            "d342_failure_subset_count": D342_PART_COUNT,
            "coverage_note": (
                "The same single authored scalar is checked on all 128 parts; this is "
                "sample coverage, not an added variable or parameter change."
            ),
            "geometry_arrays_read": False,
            "spatial_inference": False,
            "temporal_inference": False,
        },
        "stage": {
            "path": _relative(D339_PHYSICS),
            "sha256": _sha256(D339_PHYSICS),
            "root_real_path": stage_root_real_path,
            "sdf_layer_real_path": direct_layer_real_path,
            "sdf_layer_dirty_before_after": [
                direct_layer_dirty_before,
                direct_layer_dirty_after,
            ],
            "direct_meters_per_unit_authored": direct_units_authored,
            "direct_meters_per_unit": direct_meters_per_unit,
            "composed_meters_per_unit": stage_meters_per_unit,
            "schema_units": schema_contract["documented_units"],
            "interpreted_distance_unit": "metre",
            "checks": global_checks,
        },
        "contract": {
            "requested_source": (
                "D339 hull manifest decomposition_parameters.min_thickness_m"
            ),
            "requested_value_m": requested,
            "usd_storage_type": EXPECTED_TYPE_NAME,
            "expected_typed_value_m": float(expected_typed),
            "expected_float32_bits_uint32": expected_bits,
            "expected_float32_bits_hex": f"0x{expected_bits:08x}",
            "expected_float32_le_bytes_hex": _f32_le_bytes_hex(expected_typed),
            "float32_representation_delta_m": abs(float(expected_typed) - requested),
            "one_ulp_m": adjacent_negative["one_ulp_m"],
            "primary_identity_gate": "exact little-endian float32 bits",
            "frozen_compatibility_tolerance_source": (
                "D342 parameter audit frozen_parameters.readback_tolerances.min_thickness_m"
            ),
            "frozen_compatibility_tolerance_m": tolerance,
            "compatibility_tolerance_is_identity_authority": False,
            "schema": schema_contract,
        },
        "parts": part_rows,
        "typed_attribute_pass_count": typed_pass_count,
        "d342_subset_anchor_pass_count": d342_subset_anchor_pass_count,
        "global_checks": global_checks,
        "adjacent_float32_negative": adjacent_negative,
        "d342_comparator_reproduction": d342_comparator_reproduction,
        "d342_anchor": d342_anchor,
        "immutability": immutability,
        "scope_guards": {
            "standalone_pxr_only": True,
            "isaac_kit_started": False,
            "simulation_context_created": False,
            "geometry_arrays_read": False,
            "collision_asset_writes": [],
            "recook_requests": 0,
            "controlled_physics_steps": 0,
            "physical_variables_changed": [],
            "existing_parameter_increases": [],
            "existing_parameter_changes": [],
            "threshold_relaxations": [],
            "new_rerun_artifacts": [],
            "attempt3_absent": not D340_ATTEMPT3.exists(),
            "d342_verdict_changed": False,
            "g0a_pass": False,
            "ladder_promoted": False,
        },
    }
    _write_json(EVIDENCE_PATH, evidence)

    omission = _json(RERUN_OMISSION_PATH)
    summary = {
        "artifact": "D343_USD_TYPED_FLOAT_READBACK_SUMMARY_V1",
        "case": "g0a_d343",
        "verdict": verdict,
        "pass": scientific_pass,
        "new_variables": NEW_VARIABLES,
        "preregistration_checks": prereg_checks,
        "environment": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "psutil": str(psutil.__version__),
            "pxr_usd_version": list(Usd.GetVersion()),
            "isaac_kit_started": False,
        },
        "evidence_path": _relative(EVIDENCE_PATH),
        "evidence_sha256": _sha256(EVIDENCE_PATH),
        "typed_attribute_pass_count": typed_pass_count,
        "part_count": PART_COUNT,
        "d342_subset_anchor_pass_count": d342_subset_anchor_pass_count,
        "global_checks": global_checks,
        "expected_float32_value_m": float(expected_typed),
        "expected_float32_bits_uint32": expected_bits,
        "expected_float32_bits_hex": f"0x{expected_bits:08x}",
        "expected_float32_le_bytes_hex": _f32_le_bytes_hex(expected_typed),
        "float32_representation_delta_m": abs(float(expected_typed) - requested),
        "one_ulp_m": adjacent_negative["one_ulp_m"],
        "schema_default_typed_m": schema_contract["default_typed_m"],
        "schema_default_float32_bits_hex": schema_contract[
            "default_float32_bits_hex"
        ],
        "actual_unique_float32_bits_hex": sorted(
            {row["actual_float32_bits_hex"] for row in part_rows}
        ),
        "adjacent_negative_pass": adjacent_negative["pass"],
        "d342_comparator_reproduction_pass": d342_comparator_reproduction["pass"],
        "frozen_tolerance_accepts_both_adjacent_values": all(
            row["frozen_tolerance_validator_accepts"]
            for row in adjacent_negative["rows"]
        ),
        "rerun": {
            "new_artifacts": [],
            "omitted": True,
            "omission_justification_path": _relative(RERUN_OMISSION_PATH),
            "omission_justification_sha256": _sha256(RERUN_OMISSION_PATH),
            "omission_contract_pass": omission.get("pass") is True,
            "d342_rrd_context_only_sha256": _sha256(D342_RRD),
            "d342_rrd_reused_as_d343_artifact": False,
        },
        "immutability": immutability,
        "scope_guards": evidence["scope_guards"],
        "d342_registered_verdict_preserved": True,
        "d344_attempt3_eligible_for_separate_approval": scientific_pass,
        "d344_authorized": False,
        "next_gate": (
            "Stop D343. PASS only makes a separately approved D344 attempt3 authoring and "
            "fresh live-validation case eligible; it does not authorize mutation or G0a promotion."
            if scientific_pass
            else "Stop D343 FAIL. Attempt3 remains blocked."
        ),
    }
    _write_json(SUMMARY_PATH, summary)
    _write_text(REPORT_PATH, _report(summary))
    print(
        json.dumps(
            {
                "verdict": verdict,
                "typed_attribute_pass_count": typed_pass_count,
                "part_count": PART_COUNT,
                "adjacent_negative_pass": adjacent_negative["pass"],
                "d342_comparator_reproduction_pass": d342_comparator_reproduction[
                    "pass"
                ],
                "immutable": immutability["pass"],
                "attempt3_absent": not D340_ATTEMPT3.exists(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            json.dumps(
                {
                    "verdict": VERDICT_FAIL,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise
