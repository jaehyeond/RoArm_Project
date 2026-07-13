#!/usr/bin/env python3
"""D340 fixed-point live-authoring repair for cylinder G0a.

This is a two-invocation, pre-physics-only case.  ``--stage capture`` measures
both the live-instance and prototype cook of the 13 D339 failures and records a
single float32 authoring candidate only when the two channels are exactly equal.
``--stage validate`` copies the immutable D339 derivative forward to attempt3,
authors exactly those candidates once, and proves F(F(x)) == F(x) in a fresh
runtime.  There is no iterative retry, parameter fallback, physics step, or
promotion in this case.

The two registered variables are:

* ``failing_part_fixed_point_geometry`` (physical, exactly 13 derivative parts)
* ``enabled_shape_property_binding_contract`` (measurement only)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import secrets
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.viz_debug import draw_frames, log_rerun
from sim_scripts import cyl34_top_view_d332_grasp_g0a_static_collision_discriminator as d332
from sim_scripts import cyl34_top_view_d333_grasp_g0a_sole_support_static_retest as d333
from sim_scripts import cyl34_top_view_d334_grasp_g0a_live_collision_shape_ownership_audit as d334
from sim_scripts import cyl34_top_view_d336_grasp_g0a_finite_grid_caveat_discriminator as d336
from sim_scripts import cyl34_top_view_d337_grasp_g0a_open_jaw_target_gate as d337
from sim_scripts import cyl34_top_view_d338_grasp_g0a_collision_representation_repair as d338
from sim_scripts import cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair as d339


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d340"
PREREG_PATH = DEFAULT_OUT_DIR / "d340_preregistration.json"
PARAMETER_AUDIT_PATH = DEFAULT_OUT_DIR / "d340_parameter_freeze_audit.json"
CAPTURE_SUMMARY_PATH = DEFAULT_OUT_DIR / "d340_capture_summary.json"
CAPTURE_CANDIDATES_PATH = DEFAULT_OUT_DIR / "d340_capture_fixed_point_candidates.json"
CAPTURE_PNG_PATH = DEFAULT_OUT_DIR / "d340_capture_diagnostics.png"
CAPTURE_RRD_PATH = DEFAULT_OUT_DIR / "d340_capture_trace.rrd"
FINAL_SUMMARY_PATH = DEFAULT_OUT_DIR / "g0a_d340_fixed_point_live_authoring_repair_summary.json"

D339_SCRIPT = REPO / "sim_scripts/cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair.py"
D339_SUMMARY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d339/g0a_d339_cook_witness_contract_repair_summary.json"
)
D339_LIVE_AUDIT = (
    REPO / "claudedocs/runtime_logs/grasp_track/g0a_d339/d339_live_collider_audit.json"
)
D339_ATTEMPT2_DIR = (
    REPO / "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2"
)
D339_ASSET_MANIFEST = D339_ATTEMPT2_DIR / "d339_asset_build_manifest.json"
D339_ASSET_DIR = D339_ATTEMPT2_DIR / "roarm_m3_fullmesh_convex_parts"

D332_SCRIPT = REPO / "sim_scripts/cyl34_top_view_d332_grasp_g0a_static_collision_discriminator.py"
D333_SCRIPT = REPO / "sim_scripts/cyl34_top_view_d333_grasp_g0a_sole_support_static_retest.py"
D334_SCRIPT = REPO / "sim_scripts/cyl34_top_view_d334_grasp_g0a_live_collision_shape_ownership_audit.py"
D335_SCRIPT = REPO / "sim_scripts/cyl34_top_view_d335_grasp_g0a_target_family_repair.py"
D336_SCRIPT = REPO / "sim_scripts/cyl34_top_view_d336_grasp_g0a_finite_grid_caveat_discriminator.py"
D337_SCRIPT = REPO / "sim_scripts/cyl34_top_view_d337_grasp_g0a_open_jaw_target_gate.py"
D338_SCRIPT = REPO / "sim_scripts/cyl34_top_view_d338_grasp_g0a_collision_representation_repair.py"
PIN_HELPER_SCRIPT_SHA256 = {
    D332_SCRIPT: "3ab551232b9c3e2a3886578e5f4baa4589d578567758a351203c2260a1428ad4",
    D333_SCRIPT: "e582f274fca44093b0e1367555459f22428c809792b6cfc3a9a336369dac68b7",
    D334_SCRIPT: "19d2f333c2aeec89282d230324b965e6f5af7e6d05648a858c5637fd24adf735",
    D335_SCRIPT: "0eb47ca0224e84820a5074df6359f1026395b24f4c7b77613cd617789e4b234d",
    D336_SCRIPT: "fad16d7029b43a4bea779df9bbe175a0318cb29cac9fcbde701b5456411205ff",
    D337_SCRIPT: "081d0a77c91b27373eadca51fd0d9aa530d14fcdf0f64b24797a9cf8e3109489",
    D338_SCRIPT: "f3d330a9a5ca6f886728d0e5dc8037baa68d83a2b911aa105904d7d369ead426",
    D339_SCRIPT: "fd307cb573699f8a08df1ab580789188774158877b8abf0a05cc4c60ef6562d6",
}
PIN_URDF_SHA256 = "64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2"

PIN_D339_SCRIPT_SHA256 = "fd307cb573699f8a08df1ab580789188774158877b8abf0a05cc4c60ef6562d6"
PIN_D339_SUMMARY_SHA256 = "727fe02f36cf6ae84260985bccf5324b9716e5338623a28003c92589b99f8418"
PIN_D339_LIVE_AUDIT_SHA256 = "6148252b654a6250faf78a1ebcde4caa57870e800fa1d3c45b93c803fdf882cb"
PIN_D339_ASSET_MANIFEST_SHA256 = "3b46cb39a1f0ff655dcd46172ebaa84f727d833773275b18f944397007ae2589"
PIN_D339_ATTEMPT2_SHA256 = {
    "d339_asset_build_manifest.json": "3b46cb39a1f0ff655dcd46172ebaa84f727d833773275b18f944397007ae2589",
    "d339_cook_witness_manifest.json": "7d0a82842af141c1e194ffcb5f9947777b8087c8fd56c72e13f684cf61481e81",
    "d339_gripper_link_cold1_callback_witness.json": "08a500ed54a5c42c02a77b981b96d629a7de9b09df593c9f25dc3698d7220a69",
    "d339_gripper_link_cold1_canonical_geometry.json": "dc258b27cdef5d29e23f1b5ef3041c3afb26f50d8c8ad9b222002532e95f2e5e",
    "d339_gripper_link_cold2_callback_witness.json": "a74fb98c347a8d2989dd16d038c60ab9602b516d04c0b7700846792f54a96dab",
    "d339_gripper_link_cold2_canonical_geometry.json": "24b3a026281beffd18b397762226a2663ec6dd056f03099b03ad350a59edad8c",
    "d339_hull_manifest.json": "d70a13edbb8500cde97ad23779811475e1c8bb2d0f6045b4183e704d2157bedd",
    "d339_link5_cold1_callback_witness.json": "e705e7ed5d3a9b4803eaeeac67e60b6494edba37cfecf9226f86e72d60b73a43",
    "d339_link5_cold1_canonical_geometry.json": "c45bd056b3487f92bc724474dbf850ea6da309fea90c4e0a90879ada7ba2b655",
    "d339_link5_cold2_callback_witness.json": "45741a69927b7df34ebeda421f5e3570f456008da843b2e6ed1e7dd9b7fc300d",
    "d339_link5_cold2_canonical_geometry.json": "d1819851586e7078c5b58fde47bec196e08063c1ed658f79743ec729dda5590e",
    "roarm_m3_fullmesh_convex_parts/.asset_hash": "ae762fcc536a0d02a157b695201e8e64d2f71c07bcc1563981aeda59fa48587f",
    "roarm_m3_fullmesh_convex_parts/configuration/roarm_m3_base.usd": "ea0ee8f258e935799cf927b8c67e871f935c09b3c9be4f971006937334a11841",
    "roarm_m3_fullmesh_convex_parts/configuration/roarm_m3_physics.usd": "9261986d363327e33beb0b555d0ffce320416e827e0b1a8532c8e938d25b8e44",
    "roarm_m3_fullmesh_convex_parts/configuration/roarm_m3_robot.usd": "2227536fcb8c9dae1aa9cc1cf422350fcf85e662eed97fe9ea48535c6b4aa65d",
    "roarm_m3_fullmesh_convex_parts/configuration/roarm_m3_sensor.usd": "3f44081f42b452bc5f9791a8df1c37e00ba5a6dc98a9e49e065c7acacdda0d0f",
    "roarm_m3_fullmesh_convex_parts/config.yaml": "5745bbb8d9e18716e96ffafdf05b01de302e4f86375ac3f155e82ecb94ab2937",
    "roarm_m3_fullmesh_convex_parts/roarm_m3.usd": "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
}

NEW_VARIABLES = [
    "failing_part_fixed_point_geometry",
    "enabled_shape_property_binding_contract",
]
FAILING_PARTS = {
    "gripper_link": ("part_000", "part_035", "part_036", "part_048", "part_057"),
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
}
FAILING_PART_COUNT = sum(len(rows) for rows in FAILING_PARTS.values())
PASSING_PART_COUNT = 128 - FAILING_PART_COUNT

Q5_OPEN_RAD = d339.Q5_OPEN_RAD
OLD_RADIAL_NM = d339.OLD_RADIAL_NM
OLD_TANGENT_NM = d339.OLD_TANGENT_NM
RAW_ANCHOR_TOL_MM = d339.RAW_ANCHOR_TOL_MM
TASK_FIDELITY_TOL_MM = d339.TASK_FIDELITY_TOL_MM
CLEAR_GATE_MM = d339.CLEAR_GATE_MM
PROPERTY_VOLUME_BINDING_REL_TOL = d339.PROPERTY_VOLUME_BINDING_REL_TOL
LIVE_SURFACE_PARITY_TOL_M = d339.LIVE_SURFACE_PARITY_TOL_M
FIXED_POINT_COORD_TOL_M = d339.COLD_COOK_COORD_TOL_M
DECOMPOSITION_PARAMS = copy.deepcopy(d339.DECOMPOSITION_PARAMS)

VERDICT_CAPTURE_PASS = "D340_G0A_FIXED_POINT_CAPTURE_PASS_VALIDATE_PENDING"
VERDICT_CAPTURE_FAIL = "D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP"
VERDICT_BUILD_FAIL = "D340_G0A_ATTEMPT3_BUILD_CONTRACT_FAIL_STOP"
VERDICT_LIVE_FAIL = "D340_G0A_LIVE_SHAPE_CONTRACT_FAIL_STOP"
VERDICT_NOT_CLEAR = "D340_G0A_FIXED_POINT_TARGET_NOT_CLEAR_STOP"
VERDICT_SUPPORTED = "D340_G0A_FIXED_POINT_LIVE_TARGET_SUPPORTED_STOP"
VERDICT_VIZ_FAIL = "D340_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP"


def _rel(path: Path) -> str:
    return d332._rel(path)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    d332._json_dump(path, payload)


def _sha256(path: Path) -> str:
    return d332._sha256(path)


def _public_convex(convex: dict[str, Any], *, include_geometry: bool = False) -> dict[str, Any]:
    public = {
        key: value
        for key, value in convex.items()
        if key not in {"vertices", "triangles"}
    }
    if include_geometry:
        public["vertices_m"] = np.asarray(convex["vertices"], dtype=np.float64).tolist()
        public["triangles"] = np.asarray(convex["triangles"], dtype=np.int64).tolist()
    return public


def _attempt2_integrity() -> dict[str, Any]:
    inventory = (
        sorted(path.relative_to(D339_ATTEMPT2_DIR).as_posix() for path in D339_ATTEMPT2_DIR.rglob("*") if path.is_file())
        if D339_ATTEMPT2_DIR.is_dir()
        else []
    )
    observed = {
        rel: _sha256(D339_ATTEMPT2_DIR / rel)
        for rel in inventory
    }
    checks = {
        "directory_exists": D339_ATTEMPT2_DIR.is_dir(),
        "exact_inventory": inventory == sorted(PIN_D339_ATTEMPT2_SHA256),
        "exact_hashes": observed == PIN_D339_ATTEMPT2_SHA256,
    }
    return {
        "inventory": inventory,
        "sha256": observed,
        "expected_sha256": PIN_D339_ATTEMPT2_SHA256,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _self_hash_from_prereg(prereg: dict[str, Any]) -> tuple[str | None, str | None]:
    harness = prereg.get("harness", {}) if isinstance(prereg.get("harness"), dict) else {}
    expected_hash = harness.get("script_sha256", prereg.get("script_sha256"))
    expected_path = harness.get("path", prereg.get("script_path"))
    return expected_hash, expected_path


def _registered_parameter_hash(prereg: dict[str, Any]) -> str | None:
    return prereg.get(
        "parameter_freeze_audit_sha256",
        prereg.get("parameter_audit_sha256"),
    )


def _preflight_common(args: argparse.Namespace) -> dict[str, Any]:
    import psutil

    out_dir = args.out_dir
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    parameter_audit = json.loads(PARAMETER_AUDIT_PATH.read_text(encoding="utf-8"))
    d334_summary = json.loads(d337.D334_SUMMARY.read_text(encoding="utf-8"))
    d336_summary = json.loads(d337.D336_SUMMARY.read_text(encoding="utf-8"))
    d337_summary = json.loads(d339.D337_SUMMARY.read_text(encoding="utf-8"))
    d338_summary = json.loads(d339.D338_SUMMARY.read_text(encoding="utf-8"))
    d339_summary = json.loads(D339_SUMMARY.read_text(encoding="utf-8"))
    expected_script_hash, expected_script_path = _self_hash_from_prereg(prereg)
    expected_parameter_hash = _registered_parameter_hash(prereg)
    checks = {
        "exact_output_root": out_dir.resolve() == DEFAULT_OUT_DIR.resolve(),
        "seed_frozen_33201": int(args.seed) == 33201,
        "urdf_path_frozen": args.urdf_path.resolve() == d333.DEFAULT_URDF.resolve(),
        "urdf_sha256_frozen": _sha256(args.urdf_path) == PIN_URDF_SHA256,
        "process_nonce_present": bool(getattr(args, "process_nonce", None)),
        "prereg_exists": PREREG_PATH.is_file(),
        "parameter_audit_exists": PARAMETER_AUDIT_PATH.is_file(),
        "script_path_registered": expected_script_path == _rel(Path(__file__).resolve()),
        "script_hash_registered": bool(expected_script_hash),
        "script_hash_exact": expected_script_hash == _sha256(Path(__file__).resolve()),
        "parameter_audit_hash_registered": bool(expected_parameter_hash),
        "parameter_audit_hash_exact": expected_parameter_hash == _sha256(PARAMETER_AUDIT_PATH),
        "new_variables_exact": prereg.get("new_variables") == NEW_VARIABLES,
        "parameter_audit_no_scalar_increase": bool(
            parameter_audit.get("no_existing_parameter_increased")
            or parameter_audit.get("verdict") == "NO_EXISTING_PARAMETER_INCREASE"
        ),
        "d339_script_hash": _sha256(D339_SCRIPT) == PIN_D339_SCRIPT_SHA256,
        "all_helper_script_hashes_pinned": all(
            _sha256(path) == expected
            for path, expected in PIN_HELPER_SCRIPT_SHA256.items()
        ),
        "numpy_pin_1_26_0": str(np.__version__) == "1.26.0",
        "psutil_pin_5_9_8": str(psutil.__version__) == "5.9.8",
        "d334_summary_hash": _sha256(d337.D334_SUMMARY)
        == d337.PIN_D334_SUMMARY_SHA256,
        "d334_verdict": d334_summary.get("verdict")
        == "D334_G0A_ACTUAL_TOOL_OVERLAP_SUPPORTED",
        "d335_summary_hash": _sha256(d337.D335_SUMMARY)
        == d337.PIN_D335_SUMMARY_SHA256,
        "d335_csv_hash": _sha256(d337.D335_CSV) == d337.PIN_D335_CSV_SHA256,
        "d336_summary_hash": _sha256(d337.D336_SUMMARY)
        == d337.PIN_D336_SUMMARY_SHA256,
        "d336_rescore_hash": _sha256(d337.D336_RESCORE_CSV)
        == d337.PIN_D336_RESCORE_CSV_SHA256,
        "d336_verdict": d336_summary.get("verdict")
        == "D336_G0A_FINITE_GRID_CAVEAT_DISCHARGED_NO_CLEAR_STOP",
        "d337_summary_hash": _sha256(d339.D337_SUMMARY)
        == d339.PIN_D337_SUMMARY_SHA256,
        "d337_verdict": d337_summary.get("verdict")
        == "D337_G0A_STATIC_RUNTIME_MIXED_STOP",
        "d338_script_hash": _sha256(d339.D338_SCRIPT)
        == d339.PIN_D338_SCRIPT_SHA256,
        "d338_summary_hash": _sha256(d339.D338_SUMMARY)
        == d339.PIN_D338_SUMMARY_SHA256,
        "d338_verdict": d338_summary.get("verdict")
        == "D338_G0A_ASSET_BUILD_CONTRACT_FAIL_STOP",
        "d339_summary_hash": _sha256(D339_SUMMARY) == PIN_D339_SUMMARY_SHA256,
        "d339_verdict": d339_summary.get("verdict")
        == "D339_G0A_PREPHYSICS_CONTRACT_FAIL_STOP",
        "d339_live_audit_hash": _sha256(D339_LIVE_AUDIT) == PIN_D339_LIVE_AUDIT_SHA256,
        "d339_asset_manifest_hash": _sha256(D339_ASSET_MANIFEST)
        == PIN_D339_ASSET_MANIFEST_SHA256,
        "d339_attempt2_immutable": bool(_attempt2_integrity()["pass"]),
        "d338_attempt1_immutable": bool(d339._d338_attempt1_integrity()["pass"]),
        "decomposition_parameters_frozen": DECOMPOSITION_PARAMS == d338.DECOMPOSITION_PARAMS,
        "target_parameters_frozen": (
            Q5_OPEN_RAD,
            OLD_RADIAL_NM,
            OLD_TANGENT_NM,
            RAW_ANCHOR_TOL_MM,
            TASK_FIDELITY_TOL_MM,
            CLEAR_GATE_MM,
            PROPERTY_VOLUME_BINDING_REL_TOL,
            LIVE_SURFACE_PARITY_TOL_M,
        )
        == (
            d339.Q5_OPEN_RAD,
            d339.OLD_RADIAL_NM,
            d339.OLD_TANGENT_NM,
            d339.RAW_ANCHOR_TOL_MM,
            d339.TASK_FIDELITY_TOL_MM,
            d339.CLEAR_GATE_MM,
            d339.PROPERTY_VOLUME_BINDING_REL_TOL,
            d339.LIVE_SURFACE_PARITY_TOL_M,
        ),
        "exact_two_variables": len(NEW_VARIABLES) == 2,
        "exact_13_failing_parts": FAILING_PART_COUNT == 13,
        "exact_115_preserved_parts": PASSING_PART_COUNT == 115,
    }
    return {
        "preregistration": prereg,
        "parameter_audit": parameter_audit,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _preflight_capture(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    common = _preflight_common(args)
    inventory = sorted(path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*") if path.is_file())
    expected = sorted((PREREG_PATH.name, PARAMETER_AUDIT_PATH.name))
    checks = {
        "common": bool(common["pass"]),
        "exact_initial_output_inventory": inventory == expected,
        "attempt3_absent": not (out_dir / "collision_asset/attempt3").exists(),
        "capture_summary_absent": not CAPTURE_SUMMARY_PATH.exists(),
        "final_summary_absent": not FINAL_SUMMARY_PATH.exists(),
    }
    return {"common": common, "inventory": inventory, "expected_inventory": expected, "checks": checks, "pass": all(checks.values())}


def _load_capture_for_validate(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    capture = json.loads(CAPTURE_SUMMARY_PATH.read_text(encoding="utf-8"))
    candidates = json.loads(CAPTURE_CANDIDATES_PATH.read_text(encoding="utf-8"))
    return capture, candidates


def _preflight_validate(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    common = _preflight_common(args)
    capture, candidates = _load_capture_for_validate(out_dir)
    expected_witnesses = {
        f"d340_capture_cook_witnesses/{body}_{name}_{channel}.json"
        for body, names in FAILING_PARTS.items()
        for name in names
        for channel in ("instance", "prototype")
    }
    expected_files = {
        PREREG_PATH.name,
        PARAMETER_AUDIT_PATH.name,
        CAPTURE_SUMMARY_PATH.name,
        CAPTURE_CANDIDATES_PATH.name,
        CAPTURE_PNG_PATH.name,
        CAPTURE_RRD_PATH.name,
        *expected_witnesses,
    }
    inventory = {path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*") if path.is_file()}
    capture_hashes = capture.get("artifact_sha256", {})
    checks = {
        "common": bool(common["pass"]),
        "capture_process_identity_complete": bool(
            int(capture.get("process_identity", {}).get("pid", -1)) > 0
            and capture.get("process_identity", {}).get("nonce")
        ),
        "capture_validate_pid_distinct": int(
            capture.get("process_identity", {}).get("pid", -1)
        )
        != os.getpid(),
        "capture_validate_nonce_distinct": str(
            capture.get("process_identity", {}).get("nonce", "")
        )
        != str(args.process_nonce),
        "capture_verdict_pass": capture.get("verdict") == VERDICT_CAPTURE_PASS,
        "capture_contract_pass": bool(capture.get("pass")),
        "capture_candidate_contract_pass": bool(candidates.get("pass")),
        "capture_candidate_process_identity_equal": candidates.get("process_identity")
        == capture.get("process_identity"),
        "capture_candidate_count_13": int(candidates.get("candidate_count", -1)) == FAILING_PART_COUNT,
        "capture_artifact_hash_keyset_exact": set(capture_hashes)
        == (expected_files - {CAPTURE_SUMMARY_PATH.name}),
        "capture_artifact_hashes_exact": bool(
            capture_hashes
            and all(
                (out_dir / rel).is_file() and _sha256(out_dir / rel) == digest
                for rel, digest in capture_hashes.items()
                if rel != CAPTURE_SUMMARY_PATH.name
            )
        ),
        "exact_capture_output_inventory": inventory == expected_files,
        "attempt3_absent": not (out_dir / "collision_asset/attempt3").exists(),
        "final_summary_absent": not FINAL_SUMMARY_PATH.exists(),
    }
    return {
        "common": common,
        "capture": capture,
        "candidates": candidates,
        "inventory": sorted(inventory),
        "expected_inventory": sorted(expected_files),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _d339_failure_contract() -> dict[str, Any]:
    audit = json.loads(D339_LIVE_AUDIT.read_text(encoding="utf-8"))
    observed: dict[str, list[str]] = {}
    rows: dict[str, Any] = {}
    for body in d334.BODY_LABELS:
        failures = [
            row
            for row in audit["per_body"][body]["part_checks"]
            if not bool(row["pass"])
        ]
        observed[body] = sorted(Path(row["path"]).name for row in failures)
        rows[body] = {
            Path(row["path"]).name: row
            for row in failures
        }
    expected = {body: sorted(FAILING_PARTS[body]) for body in d334.BODY_LABELS}
    checks = {
        "d339_audit_verdict_context": not bool(audit["pass"]),
        "failure_set_exact": observed == expected,
        "failure_count_13": sum(len(names) for names in observed.values()) == 13,
        "all_failures_include_surface_gate": all(
            not row["checks"]["authored_vs_live_surface_le_0p1mm"]
            for body_rows in rows.values()
            for row in body_rows.values()
        ),
        "only_part045_has_d339_volume_failure": [
            f"{body}/{name}"
            for body, body_rows in rows.items()
            for name, row in body_rows.items()
            if not row["checks"]["property_vs_direct_volume_binding_le_5pct"]
        ]
        == ["link5/part_045"],
    }
    return {
        "observed": observed,
        "expected": expected,
        "d339_rows": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _expected_live_paths(manifest: dict[str, Any], body: str) -> list[str]:
    return sorted(d339._expected_live_part_paths(manifest, body))


def _live_row_by_name(inner: Any, body: str) -> dict[str, dict[str, Any]]:
    enabled = [
        row
        for row in d334._usd_collision_inventory(inner, body)
        if row["collision_enabled"]
    ]
    return {Path(row["path"]).name: row for row in enabled}


def _channel_prim_path(inner: Any, row: dict[str, Any], channel: str) -> str:
    prim = inner.scene.stage.GetPrimAtPath(row["path"])
    if not prim.IsValid():
        raise RuntimeError(f"invalid live part prim {row['path']}")
    if channel == "instance":
        if not prim.IsInstanceProxy():
            raise RuntimeError(f"registered instance channel is not an instance proxy: {row['path']}")
        return prim.GetPath().pathString
    if channel != "prototype":
        raise RuntimeError(f"unknown D340 cook channel {channel}")
    prototype = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else None
    if prototype is None or not prototype.IsValid():
        raise RuntimeError(f"no valid prototype channel for {row['path']}")
    path = prototype.GetPath().pathString
    if path == row["path"]:
        raise RuntimeError(f"prototype channel aliases instance path for {row['path']}")
    return path


def _cook_channel(
    inner: Any,
    row: dict[str, Any],
    body: str,
    *,
    channel: str,
    iteration: int,
    ordinal: int,
    witness_path: Path,
) -> dict[str, Any]:
    """Run and persist one independent channel request before classifying it."""
    from omni.physx import (
        get_physx_cooking_interface,
        get_physx_cooking_private_interface,
    )
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import Gf, PhysicsSchemaTools

    prim_path = _channel_prim_path(inner, row, channel)
    relative = d339._prim_to_body_transform(inner, row["path"], body)
    holder: dict[str, Any] = {"events": []}
    request_active = True

    def _callback(result: Any, convexes: list[Any]) -> None:
        convex_list = list(convexes)
        result_value = getattr(result, "value", None)
        if result_value is None:
            result_value = int(result)
        event: dict[str, Any] = {
            "callback_ordinal": len(holder["events"]) + 1,
            "callback_during_synchronous_request": bool(request_active),
            "result_name": str(getattr(result, "name", "")),
            "result_value": int(result_value),
            "result_repr": repr(result),
            "convex_count": len(convex_list),
            "convexes": [],
            "serialization_errors": [],
            "_raw_convexes": convex_list,
        }
        holder["events"].append(event)
        for index, convex in enumerate(convex_list):
            try:
                event["convexes"].append(d339._callback_convex_payload(convex))
            except Exception as error:  # retain result/count even on malformed payload
                event["serialization_errors"].append(
                    {
                        "convex_index": index,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )

    request_exception = None
    request_return = None
    cache_release = {
        "local_mesh_cache_released_without_exception": False,
        "runtime_mesh_cache_released_without_exception": False,
    }
    settings_evidence: dict[str, Any] = {}
    try:
        with d339._isolated_cooking_settings() as settings_evidence:
            cooking = get_physx_cooking_interface()
            private = get_physx_cooking_private_interface()
            cooking.release_local_mesh_cache()
            cache_release["local_mesh_cache_released_without_exception"] = True
            private.release_runtime_mesh_cache()
            cache_release["runtime_mesh_cache_released_without_exception"] = True
            try:
                request_return = cooking.request_convex_collision_representation(
                    stage_id=d334._stage_id(inner),
                    collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim_path),
                    run_asynchronously=False,
                    on_result=_callback,
                )
            finally:
                request_active = False
    except Exception as error:  # witness the request boundary; classify later
        request_active = False
        request_exception = {
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }

    public_events = []
    for event in holder["events"]:
        public_events.append({key: value for key, value in event.items() if key != "_raw_convexes"})
    witness = {
        "artifact": "D340_SEPARATE_LIVE_COOK_CALLBACK_WITNESS",
        "iteration": int(iteration),
        "request_order_ordinal": int(ordinal),
        "body": body,
        "part_name": Path(row["path"]).name,
        "instance_path": row["path"],
        "channel": channel,
        "cook_prim_path": prim_path,
        "channel_paths_distinct": prim_path != row["path"] if channel == "prototype" else True,
        "events": public_events,
        "callback_count": len(public_events),
        "request_return_type": type(request_return).__name__,
        "request_return_repr": repr(request_return),
        "request_exception": request_exception,
        "cache_release": cache_release,
        "isolated_cooking_settings": settings_evidence,
        "callback_payload_persisted_before_classification": True,
        "classification_performed": False,
    }
    _json_dump(witness_path, witness)

    event = holder["events"][0] if len(holder["events"]) == 1 else None
    payload = event["convexes"][0] if event is not None and len(event["convexes"]) == 1 else None
    payload_checks = d339._callback_payload_checks(payload) if payload is not None else {}
    checks = {
        "request_no_exception": request_exception is None,
        "callback_exactly_once": len(holder["events"]) == 1,
        "callback_inline": bool(event and event["callback_during_synchronous_request"]),
        "result_valid": bool(
            event
            and event["result_name"] == PhysxCollisionRepresentationResult.RESULT_VALID.name
            and int(event["result_value"])
            == int(PhysxCollisionRepresentationResult.RESULT_VALID.value)
        ),
        "one_convex": bool(event and int(event["convex_count"]) == 1),
        "serialization_no_errors": bool(event and not event["serialization_errors"]),
        "payload_structurally_valid": bool(payload_checks and all(payload_checks.values())),
        "cache_release_complete": all(cache_release.values()),
        "settings_isolated_and_restored": d339._isolated_settings_pass(settings_evidence),
        "prim_to_body_identity": bool(relative["identity_pass"]),
        "channel_identity": channel in {"instance", "prototype"},
        "prototype_distinct_from_instance": channel != "prototype" or prim_path != row["path"],
    }
    canonical = None
    canonical_error = None
    if all(checks.values()) and event is not None:
        try:
            convex = event["_raw_convexes"][0]
            vertices_local = np.asarray(
                [[float(v.x), float(v.y), float(v.z)] for v in convex.vertices],
                dtype=np.float64,
            )
            vertices_body = np.asarray(
                [
                    [
                        float(value)
                        for value in relative["body_w2l"].Transform(
                            relative["prim_l2w"].Transform(Gf.Vec3d(*vertex))
                        )
                    ]
                    for vertex in vertices_local
                ],
                dtype=np.float64,
            )
            canonical = d339._canonical_convex(vertices_body)
            canonical["cook_prim_path"] = prim_path
            canonical["coordinate_mapping_prim_path"] = row["path"]
            canonical["live_polygon_count"] = int(len(convex.polygons))
        except Exception as error:  # classification evidence
            canonical_error = {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
    checks["canonical_geometry_valid"] = canonical is not None
    result = {
        "iteration": int(iteration),
        "request_order_ordinal": int(ordinal),
        "body": body,
        "part_name": Path(row["path"]).name,
        "path": row["path"],
        "channel": channel,
        "cook_prim_path": prim_path,
        "witness_path": _rel(witness_path),
        "callback_payload_checks": payload_checks,
        "checks": checks,
        "canonical_error": canonical_error,
        "canonical": None if canonical is None else _public_convex(canonical, include_geometry=True),
        "pass": all(checks.values()),
        "_canonical": canonical,
    }
    return result


def _channel_consensus(instance: dict[str, Any], prototype: dict[str, Any]) -> dict[str, Any]:
    a = instance.get("_canonical")
    b = prototype.get("_canonical")
    coord_delta = math.inf
    exact_arrays = False
    if a is not None and b is not None and a["vertices"].shape == b["vertices"].shape:
        coord_delta = float(np.max(np.abs(a["vertices"] - b["vertices"])))
        exact_arrays = bool(
            np.array_equal(a["vertices"], b["vertices"])
            and np.array_equal(a["triangles"], b["triangles"])
        )
    checks = {
        "instance_pass": bool(instance["pass"]),
        "prototype_pass": bool(prototype["pass"]),
        "paths_distinct": instance["cook_prim_path"] != prototype["cook_prim_path"],
        "vertex_count_equal": bool(a is not None and b is not None and a["vertex_count"] == b["vertex_count"]),
        "triangle_count_equal": bool(a is not None and b is not None and a["triangle_count"] == b["triangle_count"]),
        "geometry_hash_equal": bool(a is not None and b is not None and a["geometry_sha256"] == b["geometry_sha256"]),
        "topology_hash_equal": bool(a is not None and b is not None and a["topology_sha256"] == b["topology_sha256"]),
        "canonical_arrays_bit_exact": exact_arrays,
        "coordinate_delta_le_1e_9m": coord_delta <= FIXED_POINT_COORD_TOL_M,
    }
    consensus = a if all(checks.values()) else None
    return {
        "instance": {key: value for key, value in instance.items() if key != "_canonical"},
        "prototype": {key: value for key, value in prototype.items() if key != "_canonical"},
        "checks": checks,
        "coordinate_max_abs_delta_m": coord_delta,
        "pass": all(checks.values()),
        "consensus": None if consensus is None else _public_convex(consensus, include_geometry=True),
        "_consensus": consensus,
    }


def _float32_roundtrip(convex: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(convex["vertices"], dtype=np.float32).astype(np.float64)
    return d339._canonical_convex(vertices)


def _capture_candidate(authored: dict[str, Any], consensus: dict[str, Any]) -> dict[str, Any]:
    consensus_convex = consensus.get("_consensus")
    candidate = _float32_roundtrip(consensus_convex) if consensus_convex is not None else None
    containment_live_in_authored = math.inf
    authored_in_live = math.inf
    float32_delta = math.inf
    if consensus_convex is not None:
        containment_live_in_authored = d339._directed_convex_solid_distance_m(
            consensus_convex["vertices"], authored
        )
        authored_in_live = d339._directed_convex_solid_distance_m(
            authored["vertices"], consensus_convex
        )
    if candidate is not None:
        fp = d339._convex_solid_hausdorff_m(consensus_convex, candidate)
        float32_delta = float(fp["symmetric_m"])
    checks = {
        "channel_consensus": bool(consensus["pass"]),
        "live_output_contained_in_authored": containment_live_in_authored <= FIXED_POINT_COORD_TOL_M,
        "authored_not_equal_live": bool(
            consensus_convex is not None
            and consensus_convex["geometry_sha256"] != authored["geometry_sha256"]
        ),
        "strict_vertex_decrease": bool(
            candidate is not None and int(candidate["vertex_count"]) < int(authored["vertex_count"])
        ),
        "float32_roundtrip_surface_le_1e_9m": float32_delta <= FIXED_POINT_COORD_TOL_M,
        "float32_roundtrip_no_vertex_growth": bool(
            candidate is not None
            and int(candidate["vertex_count"]) <= int(consensus_convex["vertex_count"])
        ),
        "no_iteration0_cycle": bool(
            candidate is not None and candidate["geometry_sha256"] != authored["geometry_sha256"]
        ),
    }
    return {
        "authored_x0": _public_convex(authored, include_geometry=True),
        "channel_consensus": {key: value for key, value in consensus.items() if key != "_consensus"},
        "live_x1_containment_in_x0_m": containment_live_in_authored,
        "x0_directed_distance_to_live_x1_m": authored_in_live,
        "float32_roundtrip_surface_delta_m": float32_delta,
        "checks": checks,
        "pass": all(checks.values()),
        "candidate_x1": None if candidate is None else _public_convex(candidate, include_geometry=True),
        "_candidate": candidate,
    }


def _safe_cook_channel(
    inner: Any,
    row: dict[str, Any],
    body: str,
    *,
    channel: str,
    iteration: int,
    ordinal: int,
    witness_path: Path,
) -> dict[str, Any]:
    try:
        return _cook_channel(
            inner,
            row,
            body,
            channel=channel,
            iteration=iteration,
            ordinal=ordinal,
            witness_path=witness_path,
        )
    except Exception as error:  # retain a channel-specific STOP witness
        payload = {
            "artifact": "D340_SEPARATE_LIVE_COOK_PRECALL_FAILURE_WITNESS",
            "iteration": int(iteration),
            "request_order_ordinal": int(ordinal),
            "body": body,
            "part_name": Path(row["path"]).name,
            "instance_path": row["path"],
            "channel": channel,
            "callback_count": 0,
            "callback_payload_persisted_before_classification": True,
            "classification_performed": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        _json_dump(witness_path, payload)
        return {
            "iteration": int(iteration),
            "request_order_ordinal": int(ordinal),
            "body": body,
            "part_name": Path(row["path"]).name,
            "path": row["path"],
            "channel": channel,
            "cook_prim_path": None,
            "witness_path": _rel(witness_path),
            "callback_payload_checks": {},
            "checks": {"precall_completed": False},
            "canonical_error": payload,
            "canonical": None,
            "pass": False,
            "_canonical": None,
        }


def _write_capture_figure(path: Path, part_rows: list[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{row['body'][:4]}/{row['name'][-3:]}" for row in part_rows]
    before = [row["fixed_point"]["authored_x0"]["vertex_count"] for row in part_rows]
    after = [
        row["fixed_point"]["candidate_x1"]["vertex_count"]
        if row["fixed_point"]["candidate_x1"] is not None
        else 0
        for row in part_rows
    ]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.0, 7.0), dpi=150)
    x = np.arange(len(labels))
    ax0.bar(x - 0.2, before, width=0.4, label="D339 authored x0")
    ax0.bar(x + 0.2, after, width=0.4, label="float32 candidate x1")
    ax0.set_xticks(x, labels, rotation=45, ha="right")
    ax0.set_ylabel("canonical vertices")
    ax0.set_title("D340 capture: one-authoring fixed-point candidates (instance -> prototype)")
    ax0.legend()
    containment_um = [
        1.0e6 * float(row["fixed_point"]["live_x1_containment_in_x0_m"])
        for row in part_rows
    ]
    roundtrip_um = [
        1.0e6 * float(row["fixed_point"]["float32_roundtrip_surface_delta_m"])
        for row in part_rows
    ]
    ax1.plot(x, containment_um, marker="o", label="x1 outside x0 [um]")
    ax1.plot(x, roundtrip_um, marker="x", label="live -> float32 roundtrip [um]")
    ax1.axhline(FIXED_POINT_COORD_TOL_M * 1.0e6, color="red", linestyle="--", label="1e-9m gate")
    ax1.set_xticks(x, labels, rotation=45, ha="right")
    ax1.set_ylabel("directed/symmetric distance [um]")
    ax1.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return _rel(path)


def _run_capture(args: argparse.Namespace, _simulation_app: Any) -> dict[str, Any]:
    d332._runtime_versions()
    d339_manifest = json.loads(D339_ASSET_MANIFEST.read_text(encoding="utf-8"))
    d334_summary = json.loads(d337.D334_SUMMARY.read_text(encoding="utf-8"))
    failure_contract = _d339_failure_contract()
    attempt2_before = _attempt2_integrity()
    args.robot_usd_path = D339_ASSET_DIR / "roarm_m3.usd"
    inner = d333._make_runtime_env(args)
    part_rows: list[dict[str, Any]] = []
    artifact_errors: list[dict[str, str]] = []
    candidate: dict[str, Any] | None = None
    stage_contract: dict[str, Any] = {}
    sensor_contract: dict[str, Any] = {}
    raw_source_contract: dict[str, Any] = {}
    try:
        inner.reset(seed=int(args.seed))
        counter_start = int(inner._sim_step_counter)
        stage_contract = d333._stage_contract(inner)
        sensor_contract, _filter_map = d333._sensor_contract(inner)
        try:
            raw_shapes, raw_source_contract = d339._build_retained_raw_shapes(
                inner, d334_summary
            )
        except Exception as error:
            raw_shapes = []
            raw_source_contract = {
                "pass": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        inventory_contract: dict[str, Any] = {"per_body": {}}
        all_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for body in d334.BODY_LABELS:
            usd_inventory = d334._usd_collision_inventory(inner, body)
            enabled = sorted(
                [row for row in usd_inventory if row["collision_enabled"]],
                key=lambda row: row["path"],
            )
            expected_paths = _expected_live_paths(d339_manifest, body)
            disabled = [row for row in usd_inventory if not row["collision_enabled"]]
            checks = {
                "usd_inventory_exact_65": len(usd_inventory) == 65,
                "enabled_exact_64": len(enabled) == 64,
                "enabled_paths_exact": [row["path"] for row in enabled] == expected_paths,
                "disabled_exact_known_legacy": len(disabled) == 1
                and disabled[0]["path"] == d339.LIVE_OLD_COLLIDER_PATHS[body],
            }
            inventory_contract["per_body"][body] = {
                "checks": checks,
                "pass": all(checks.values()),
                "usd_inventory": usd_inventory,
            }
            all_rows[body] = {Path(row["path"]).name: row for row in enabled}
        inventory_contract["pass"] = all(
            row["pass"] for row in inventory_contract["per_body"].values()
        )

        witness_dir = args.out_dir / "d340_capture_cook_witnesses"
        witness_dir.mkdir(parents=True, exist_ok=False)
        ordinal = 0
        for body in d334.BODY_LABELS:
            expected_manifest = {row["name"]: row for row in d339_manifest["parts"][body]}
            for name in FAILING_PARTS[body]:
                row = all_rows.get(body, {}).get(name)
                if row is None:
                    # Keep exact witness inventory even when the authored path is missing.
                    row = {
                        "path": f"{d339.LIVE_PART_PARENT_PATHS[body]}/{name}",
                        "mesh_prim_paths": [],
                    }
                try:
                    authored_source = d334._source_mesh_body_local(inner, row, body)
                    authored = d339._canonical_convex(authored_source["_verts_body"])
                    authored_error = None
                except Exception as error:
                    authored = None
                    authored_error = {
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                channels: dict[str, Any] = {}
                # Iteration-0 order is deliberately inverted in validation.
                for channel in ("instance", "prototype"):
                    ordinal += 1
                    channels[channel] = _safe_cook_channel(
                        inner,
                        row,
                        body,
                        channel=channel,
                        iteration=0,
                        ordinal=ordinal,
                        witness_path=witness_dir / f"{body}_{name}_{channel}.json",
                    )
                consensus = _channel_consensus(channels["instance"], channels["prototype"])
                if authored is None:
                    fixed = {
                        "authored_x0": None,
                        "channel_consensus": {
                            key: value for key, value in consensus.items() if key != "_consensus"
                        },
                        "checks": {"authored_geometry_available": False},
                        "pass": False,
                        "candidate_x1": None,
                        "_candidate": None,
                    }
                else:
                    fixed = _capture_candidate(authored, consensus)
                expected = expected_manifest.get(name)
                part_checks = {
                    "authored_available": authored is not None,
                    "manifest_part_found": expected is not None,
                    "authored_hash_matches_d339_manifest": bool(
                        authored is not None
                        and expected is not None
                        and authored["geometry_sha256"] == expected["geometry_sha256"]
                    ),
                    "fixed_point_capture_contract": bool(fixed["pass"]),
                }
                part_rows.append(
                    {
                        "body": body,
                        "name": name,
                        "path": row["path"],
                        "authored_error": authored_error,
                        "checks": part_checks,
                        "pass": all(part_checks.values()),
                        "fixed_point": {
                            key: value for key, value in fixed.items() if key != "_candidate"
                        },
                        "_candidate": fixed.get("_candidate"),
                    }
                )

        scientific_checks = {
            "failure_contract_exact": bool(failure_contract["pass"]),
            "stage_contract": bool(stage_contract["hard_contract_pass"]),
            "sensor_contract": bool(sensor_contract["hard_contract_pass"]),
            "retained_raw_source_contract": bool(raw_source_contract["pass"]),
            "attempt2_integrity_before": bool(attempt2_before["pass"]),
            "usd_inventory_contract": bool(inventory_contract["pass"]),
            "part_count_exact_13": len(part_rows) == FAILING_PART_COUNT,
            "all_13_capture_candidates_pass": bool(part_rows)
            and all(row["pass"] for row in part_rows),
            "all_13_strictly_reduce_vertices": bool(part_rows)
            and all(
                row["fixed_point"]["checks"].get("strict_vertex_decrease", False)
                for row in part_rows
            ),
            "capture_request_order_instance_then_prototype": all(
                row["fixed_point"]["channel_consensus"]["instance"]["request_order_ordinal"]
                < row["fixed_point"]["channel_consensus"]["prototype"]["request_order_ordinal"]
                for row in part_rows
            ),
        }

        if raw_source_contract.get("pass"):
            candidate = d337._evaluate_candidate(
                inner,
                raw_shapes,
                OLD_RADIAL_NM,
                OLD_TANGENT_NM,
                Q5_OPEN_RAD,
                stage="d340_capture_frozen_target_frames",
            )
        else:
            candidate = d339._fallback_candidate_without_raw(
                inner, "D340 capture retained raw-source contract failed"
            )
        counter_end = int(inner._sim_step_counter)
        scientific_checks["global_sim_counter_unchanged"] = counter_end == counter_start
        scientific_pass = all(scientific_checks.values())

        candidates_payload = {
            "artifact": "D340_FLOAT32_FIXED_POINT_AUTHORING_CANDIDATES",
            "process_identity": {
                "pid": os.getpid(),
                "nonce": args.process_nonce,
            },
            "new_variables": NEW_VARIABLES,
            "iteration": 0,
            "request_order": ["instance", "prototype"],
            "one_authoring_application_only": True,
            "no_iterative_retry": True,
            "candidate_count": sum(row["_candidate"] is not None for row in part_rows),
            "parts": [
                {key: value for key, value in row.items() if key != "_candidate"}
                for row in part_rows
            ],
            "checks": scientific_checks,
            "pass": scientific_pass,
        }
        _json_dump(CAPTURE_CANDIDATES_PATH, candidates_payload)

        try:
            _write_capture_figure(CAPTURE_PNG_PATH, part_rows)
        except Exception as error:
            artifact_errors.append(
                {
                    "artifact": "capture_png",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        try:
            marker_status = draw_frames(
                candidate["_frames"], prim_path="/World/D340CaptureFrames"
            )
        except Exception as error:
            marker_status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            artifact_errors.append(
                {
                    "artifact": "frame_markers",
                    "error": marker_status["error"],
                    "traceback": traceback.format_exc(),
                }
            )
        try:
            rrd_status = log_rerun(
                CAPTURE_RRD_PATH,
                frames=candidate["_frames"],
                joint_state={
                    "label": "d340_fixed_point_capture",
                    "object": "cylinder_d34_h90",
                    "iteration": 0,
                    "request_order": "instance_then_prototype",
                    "controlled_physics_steps_total": 0,
                },
                joint_trace=[d339._decision_trace_row(inner, candidate)],
                urdf_path=args.urdf_path,
                live_viewer=False,
                app_id="roarm_g0a_d340_fixed_point_capture",
            )
        except Exception as error:
            rrd_status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            artifact_errors.append(
                {
                    "artifact": "capture_rrd",
                    "error": rrd_status["error"],
                    "traceback": traceback.format_exc(),
                }
            )
        rrd_nonzero = CAPTURE_RRD_PATH.is_file() and CAPTURE_RRD_PATH.stat().st_size > 0
        artifact_checks = {
            "capture_png_nonzero": CAPTURE_PNG_PATH.is_file() and CAPTURE_PNG_PATH.stat().st_size > 0,
            "frame_markers_ok": bool(marker_status.get("ok")),
            "capture_rrd_ok": bool(rrd_status.get("ok")),
            "capture_rrd_nonzero": rrd_nonzero,
            "no_artifact_errors": not artifact_errors,
        }
        artifact_pass = all(artifact_checks.values())
        attempt2_after = _attempt2_integrity()
        immutable_pass = bool(attempt2_after["pass"] and d339._d338_attempt1_integrity()["pass"])
        capture_pass = bool(scientific_pass and artifact_pass and immutable_pass)
        scientific_verdict = VERDICT_CAPTURE_PASS if scientific_pass else VERDICT_CAPTURE_FAIL
        verdict = scientific_verdict if artifact_pass else VERDICT_VIZ_FAIL
        if not immutable_pass:
            verdict = VERDICT_CAPTURE_FAIL
        artifact_sha256 = {
            path.relative_to(args.out_dir).as_posix(): _sha256(path)
            for path in sorted(args.out_dir.rglob("*"))
            if path.is_file() and path != CAPTURE_SUMMARY_PATH
        }
        summary = {
            "verdict": verdict,
            "scientific_verdict_before_artifact_gate": scientific_verdict,
            "pass": capture_pass,
            "active_case": "G0a cylinder D34xH90 fixed-point live-authoring capture",
            "new_variables": NEW_VARIABLES,
            "physical_variables_changed": ["failing_part_fixed_point_geometry"],
            "parameters_increased": [],
            "stage": "capture",
            "process_identity": {
                "pid": os.getpid(),
                "nonce": args.process_nonce,
            },
            "iteration": 0,
            "request_order": ["instance", "prototype"],
            "failure_contract": failure_contract,
            "inventory_contract": inventory_contract,
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "raw_source_contract": raw_source_contract,
            "scientific_checks": scientific_checks,
            "candidate_manifest": _rel(CAPTURE_CANDIDATES_PATH),
            "candidate_manifest_sha256": _sha256(CAPTURE_CANDIDATES_PATH),
            "attempt2_integrity_before": attempt2_before,
            "attempt2_integrity_after": attempt2_after,
            "controlled_physics_steps": 0,
            "simulation_counter": {"start": counter_start, "end": counter_end},
            "visualization": {
                "decision_png": _rel(CAPTURE_PNG_PATH) if CAPTURE_PNG_PATH.is_file() else None,
                "marker_status": marker_status,
                "rrd_status": rrd_status,
                "artifact_errors": artifact_errors,
            },
            "artifact_contract": {"checks": artifact_checks, "pass": artifact_pass},
            "artifact_sha256": artifact_sha256,
            "outcome_guards": {
                "attempt3_created": False,
                "physics_executed": False,
                "g0a_pass": False,
                "ten_trial_run": False,
                "ladder_promoted": False,
            },
        }
        _json_dump(CAPTURE_SUMMARY_PATH, summary)
        return summary
    finally:
        inner.close()


def _part_spec_path(body: str, name: str) -> str:
    return f"{d339.SOURCE_ASSET_PATHS[body]['parts_parent']}/{name}"


def _part_layer_record(physics_path: Path, body: str, name: str) -> dict[str, Any]:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(physics_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open physics layer {physics_path}")
    prim = stage.GetPrimAtPath(_part_spec_path(body, name))
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"missing authored part {_part_spec_path(body, name)}")
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(
        [[float(value) for value in point] for point in list(mesh.GetPointsAttr().Get() or [])],
        dtype=np.float32,
    )
    counts = np.asarray(list(mesh.GetFaceVertexCountsAttr().Get() or []), dtype=np.int64)
    indices = np.asarray(list(mesh.GetFaceVertexIndicesAttr().Get() or []), dtype=np.int64)
    canonical = d339._canonical_convex(points.astype(np.float64))
    mesh_api = UsdPhysics.MeshCollisionAPI(prim)
    hull_api = PhysxSchema.PhysxConvexHullCollisionAPI(prim)
    collision = UsdPhysics.CollisionAPI(prim)
    collision_enabled = collision.GetCollisionEnabledAttr().Get()
    return {
        "path": _part_spec_path(body, name),
        "points_f32_sha256": hashlib.sha256(np.ascontiguousarray(points.astype("<f4")).tobytes()).hexdigest(),
        "points_f32": points.astype(np.float64).tolist(),
        "face_vertex_counts": counts.tolist(),
        "face_vertex_indices": indices.tolist(),
        "face_vertex_counts_sha256": hashlib.sha256(np.ascontiguousarray(counts.astype("<i8")).tobytes()).hexdigest(),
        "face_vertex_indices_sha256": hashlib.sha256(np.ascontiguousarray(indices.astype("<i8")).tobytes()).hexdigest(),
        "canonical": _public_convex(canonical),
        "subdivision_scheme": str(mesh.GetSubdivisionSchemeAttr().Get()),
        "double_sided": bool(mesh.GetDoubleSidedAttr().Get()),
        "collision_enabled": True if collision_enabled is None else bool(collision_enabled),
        "approximation": str(mesh_api.GetApproximationAttr().Get()),
        "hull_vertex_limit": int(hull_api.GetHullVertexLimitAttr().Get()),
        "min_thickness_m": float(hull_api.GetMinThicknessAttr().Get()),
    }


def _record_changed_fields(source: dict[str, Any], variant: dict[str, Any]) -> list[str]:
    return sorted(key for key in source if source[key] != variant.get(key))


def _attempt3_semantic_allowlist_diff(
    source_physics: Path,
    variant_physics: Path,
    changed_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    """Prove the whole physics layer is equal outside 39 geometry properties."""
    from pxr import Sdf, Usd

    source_layer = Sdf.Layer.FindOrOpen(str(source_physics))
    variant_layer = Sdf.Layer.FindOrOpen(str(variant_physics))
    if source_layer is None or variant_layer is None:
        raise RuntimeError("failed to open source/variant physics layer for semantic diff")
    allowed_property_paths = sorted(
        f"{_part_spec_path(body, name)}.{attribute}"
        for body, name in changed_keys
        for attribute in ("points", "faceVertexCounts", "faceVertexIndices")
    )

    def _sanitized(layer: Any, tag: str) -> dict[str, Any]:
        clone = Sdf.Layer.CreateAnonymous(f"d340_{tag}_sanitized_physics.usda")
        if not clone.ImportFromString(layer.ExportToString()):
            raise RuntimeError(f"failed to clone {tag} physics layer")
        missing = []
        edits = Sdf.BatchNamespaceEdit()
        for value in allowed_property_paths:
            path = Sdf.Path(value)
            if clone.GetObjectAtPath(path) is None:
                missing.append(value)
            else:
                edits.Add(Sdf.NamespaceEdit.Remove(path))
        if missing:
            raise RuntimeError(f"missing registered geometry properties in {tag}: {missing}")
        if not clone.Apply(edits):
            raise RuntimeError(f"failed to remove D340 allowlist from {tag} layer")
        text = clone.ExportToString()
        return {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_length": len(text),
            "removed_property_count": len(allowed_property_paths),
        }

    def _composed_inventory(path: Path) -> dict[str, Any]:
        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"failed to open composed physics inventory {path}")
        rows = []
        for prim in Usd.PrimRange.Stage(stage):
            attributes = []
            for attr in prim.GetAttributes():
                value = (
                    "<D340_REGISTERED_GEOMETRY_VALUE>"
                    if f"{prim.GetPath().pathString}.{attr.GetName()}" in allowed_property_paths
                    else repr(attr.Get())
                )
                attributes.append(
                    (
                        attr.GetName(),
                        str(attr.GetTypeName()),
                        value,
                        tuple(str(item) for item in attr.GetConnections()),
                        tuple(sorted((key, repr(attr.GetMetadata(key))) for key in attr.GetAllMetadata())),
                    )
                )
            relationships = [
                (
                    rel.GetName(),
                    tuple(str(item) for item in rel.GetTargets()),
                    tuple(sorted((key, repr(rel.GetMetadata(key))) for key in rel.GetAllMetadata())),
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
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
        return {
            "row_count": len(rows),
            "paths": [row["path"] for row in rows],
            "sanitized_semantic_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }

    source_sanitized = _sanitized(source_layer, "source")
    variant_sanitized = _sanitized(variant_layer, "variant")
    source_inventory = _composed_inventory(source_physics)
    variant_inventory = _composed_inventory(variant_physics)
    header_equal = bool(
        source_layer.defaultPrim == variant_layer.defaultPrim
        and list(source_layer.subLayerPaths) == list(variant_layer.subLayerPaths)
        and source_layer.customLayerData == variant_layer.customLayerData
        and source_layer.startTimeCode == variant_layer.startTimeCode
        and source_layer.endTimeCode == variant_layer.endTimeCode
        and source_layer.timeCodesPerSecond == variant_layer.timeCodesPerSecond
        and source_layer.framesPerSecond == variant_layer.framesPerSecond
    )
    checks = {
        "exact_39_allowed_geometry_properties": len(allowed_property_paths) == 39,
        "physics_layer_header_equal": header_equal,
        "all_prim_paths_exact": source_inventory["paths"] == variant_inventory["paths"],
        "prim_path_count_exact": source_inventory["row_count"] == variant_inventory["row_count"],
        "all_prim_type_schema_metadata_attribute_relationship_semantics_equal_after_allowlist": (
            source_inventory["sanitized_semantic_sha256"]
            == variant_inventory["sanitized_semantic_sha256"]
        ),
        "whole_physics_spec_equal_after_removing_allowlist": source_sanitized
        == variant_sanitized,
    }
    return {
        "allowed_property_paths": allowed_property_paths,
        "source_sanitized_layer": source_sanitized,
        "variant_sanitized_layer": variant_sanitized,
        "source_composed_inventory": source_inventory,
        "variant_composed_inventory": variant_inventory,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _author_attempt3(
    args: argparse.Namespace,
    capture_candidates: dict[str, Any],
) -> dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom

    attempt_dir = args.out_dir / "collision_asset" / "attempt3"
    variant_dir = attempt_dir / "roarm_m3_fullmesh_fixed_point_parts"
    physics_rel = Path("configuration/roarm_m3_physics.usd")
    source_physics = D339_ASSET_DIR / physics_rel
    variant_physics = variant_dir / physics_rel
    if attempt_dir.exists():
        raise RuntimeError("D340 attempt3 already exists; forward-only overwrite refused")
    attempt_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(D339_ASSET_DIR, variant_dir)

    candidate_rows = {
        (row["body"], row["name"]): row
        for row in capture_candidates["parts"]
    }
    expected_keys = {
        (body, name) for body, names in FAILING_PARTS.items() for name in names
    }
    if set(candidate_rows) != expected_keys:
        raise RuntimeError(
            f"capture candidate key mismatch: {sorted(candidate_rows)} != {sorted(expected_keys)}"
        )
    stage = Usd.Stage.Open(str(variant_physics), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open copied attempt3 physics layer {variant_physics}")
    for body, name in sorted(expected_keys):
        row = candidate_rows[(body, name)]
        candidate = row["fixed_point"].get("candidate_x1")
        if not row.get("pass") or candidate is None:
            raise RuntimeError(f"capture candidate is not passing: {body}/{name}")
        prim = stage.GetPrimAtPath(_part_spec_path(body, name))
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"missing copied part for authoring: {body}/{name}")
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(candidate["vertices_m"], dtype=np.float32)
        triangles = np.asarray(candidate["triangles"], dtype=np.int64)
        mesh.GetPointsAttr().Set(
            [Gf.Vec3f(*[float(value) for value in vertex]) for vertex in vertices]
        )
        mesh.GetFaceVertexCountsAttr().Set([3] * int(len(triangles)))
        mesh.GetFaceVertexIndicesAttr().Set([int(value) for value in triangles.reshape(-1)])
    stage.GetRootLayer().Save()

    source_manifest = json.loads(D339_ASSET_MANIFEST.read_text(encoding="utf-8"))
    source_records: dict[tuple[str, str], dict[str, Any]] = {}
    variant_records: dict[tuple[str, str], dict[str, Any]] = {}
    part_audits = []
    preserved_count = 0
    changed_count = 0
    allowed_geometry_fields = {
        "points_f32_sha256",
        "points_f32",
        "face_vertex_counts",
        "face_vertex_indices",
        "face_vertex_counts_sha256",
        "face_vertex_indices_sha256",
        "canonical",
    }
    new_parts = copy.deepcopy(source_manifest["parts"])
    manifest_index = {
        (body, row["name"]): row
        for body, rows in new_parts.items()
        for row in rows
    }
    for body in d334.BODY_LABELS:
        for index in range(64):
            name = f"part_{index:03d}"
            key = (body, name)
            source_record = _part_layer_record(source_physics, body, name)
            variant_record = _part_layer_record(variant_physics, body, name)
            source_records[key] = source_record
            variant_records[key] = variant_record
            changed_fields = _record_changed_fields(source_record, variant_record)
            registered_change = key in expected_keys
            if registered_change:
                changed_count += 1
                candidate = candidate_rows[key]["fixed_point"]["candidate_x1"]
                candidate_points_f32 = np.asarray(
                    candidate["vertices_m"], dtype=np.float32
                )
                candidate_triangles = np.asarray(
                    candidate["triangles"], dtype=np.int64
                )
                candidate_counts = np.full(
                    len(candidate_triangles), 3, dtype=np.int64
                )
                candidate_indices = candidate_triangles.reshape(-1)
                candidate_points_sha256 = hashlib.sha256(
                    np.ascontiguousarray(candidate_points_f32.astype("<f4")).tobytes()
                ).hexdigest()
                checks = {
                    "geometry_changed": bool(changed_fields),
                    "changed_fields_geometry_only": bool(changed_fields)
                    and set(changed_fields) <= allowed_geometry_fields,
                    "variant_geometry_hash_matches_candidate": (
                        variant_record["canonical"]["geometry_sha256"]
                        == candidate["geometry_sha256"]
                    ),
                    "variant_vertex_count_matches_candidate": (
                        variant_record["canonical"]["vertex_count"]
                        == candidate["vertex_count"]
                    ),
                    "variant_topology_matches_candidate": (
                        variant_record["canonical"]["topology_sha256"]
                        == candidate["topology_sha256"]
                    ),
                    "authored_points_f32_stream_bit_exact_candidate": bool(
                        variant_record["points_f32_sha256"]
                        == candidate_points_sha256
                        and np.array_equal(
                            np.asarray(variant_record["points_f32"], dtype=np.float32),
                            candidate_points_f32,
                        )
                    ),
                    "authored_face_counts_exact_candidate": np.array_equal(
                        np.asarray(
                            variant_record["face_vertex_counts"], dtype=np.int64
                        ),
                        candidate_counts,
                    ),
                    "authored_face_indices_exact_candidate": np.array_equal(
                        np.asarray(
                            variant_record["face_vertex_indices"], dtype=np.int64
                        ),
                        candidate_indices,
                    ),
                    "strict_vertex_decrease_preserved": (
                        variant_record["canonical"]["vertex_count"]
                        < source_record["canonical"]["vertex_count"]
                    ),
                }
                manifest_index[key].update(
                    {
                        "vertex_count": variant_record["canonical"]["vertex_count"],
                        "triangle_count": variant_record["canonical"]["triangle_count"],
                        "volume_m3": variant_record["canonical"]["volume_m3"],
                        "bounds_m": variant_record["canonical"]["bounds_m"],
                        "centroid_m": variant_record["canonical"]["centroid_m"],
                        "vertex_stream_sha256": variant_record["canonical"]["vertex_stream_sha256"],
                        "topology_sha256": variant_record["canonical"]["topology_sha256"],
                        "geometry_sha256": variant_record["canonical"]["geometry_sha256"],
                    }
                )
            else:
                preserved_count += 1
                checks = {
                    "bit_exact_full_authored_record": source_record == variant_record,
                    "no_changed_fields": not changed_fields,
                }
            part_audits.append(
                {
                    "body": body,
                    "name": name,
                    "registered_change": registered_change,
                    "changed_fields": changed_fields,
                    "source": source_record,
                    "variant": variant_record,
                    "checks": checks,
                    "pass": all(checks.values()),
                }
            )

    source_hashes = {
        path.relative_to(D339_ASSET_DIR).as_posix(): _sha256(path)
        for path in D339_ASSET_DIR.rglob("*")
        if path.is_file()
    }
    variant_hashes = {
        path.relative_to(variant_dir).as_posix(): _sha256(path)
        for path in variant_dir.rglob("*")
        if path.is_file()
    }
    nonphysics = sorted(set(source_hashes) - {physics_rel.as_posix()})
    source_mass = source_manifest["source_tool_mass_semantics"]
    variant_mass = d339._tool_mass_semantics_from_stage(variant_dir / "roarm_m3.usd")
    semantic_allowlist = _attempt3_semantic_allowlist_diff(
        source_physics, variant_physics, expected_keys
    )
    build_checks = {
        "capture_candidates_pass": bool(capture_candidates["pass"]),
        "exact_13_registered_changes": changed_count == 13,
        "exact_115_preserved_parts": preserved_count == 115,
        "all_part_audits_pass": all(row["pass"] for row in part_audits),
        "nonphysics_file_inventory_exact": sorted(source_hashes) == sorted(variant_hashes),
        "nonphysics_files_bit_exact": all(source_hashes[path] == variant_hashes[path] for path in nonphysics),
        "physics_layer_changed": source_hashes[physics_rel.as_posix()]
        != variant_hashes[physics_rel.as_posix()],
        "tool_mass_semantics_equal": variant_mass == source_mass,
        "whole_physics_semantic_allowlist_exact": bool(semantic_allowlist["pass"]),
        "attempt2_immutable_after_authoring": bool(_attempt2_integrity()["pass"]),
        "d338_attempt1_immutable_after_authoring": bool(d339._d338_attempt1_integrity()["pass"]),
    }
    manifest_path = attempt_dir / "d340_attempt3_asset_manifest.json"
    manifest = {
        "artifact": "D340_ATTEMPT3_FIXED_POINT_ASSET_MANIFEST",
        "asset_attempt": "attempt3",
        "new_variables": NEW_VARIABLES,
        "parameters_increased": [],
        "source_attempt2_dir": _rel(D339_ATTEMPT2_DIR),
        "source_attempt2_integrity": _attempt2_integrity(),
        "source_asset_dir": _rel(D339_ASSET_DIR),
        "variant_asset_dir": _rel(variant_dir),
        "variant_robot_usd": _rel(variant_dir / "roarm_m3.usd"),
        "source_layer_sha256": source_hashes,
        "variant_layer_sha256": variant_hashes,
        "decomposition_parameters": DECOMPOSITION_PARAMS,
        "source_tool_mass_semantics": source_mass,
        "variant_tool_mass_semantics": variant_mass,
        "whole_physics_semantic_allowlist": semantic_allowlist,
        "parts": new_parts,
        "changed_part_keys": [f"{body}/{name}" for body, name in sorted(expected_keys)],
        "preserved_part_count": preserved_count,
        "changed_part_count": changed_count,
        "part_audits": part_audits,
        "checks": build_checks,
        "pass": all(build_checks.values()),
    }
    _json_dump(manifest_path, manifest)
    manifest["manifest_path"] = _rel(manifest_path)
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def _validate_live_attempt3(
    inner: Any,
    simulation_app: Any,
    manifest: dict[str, Any],
    capture_candidates: dict[str, Any],
    witness_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import hppfcl
    import omni.kit.app
    from omni.physxassetvalidator import get_physx_asset_validator_interface
    from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_name = "omni.physx.asset_validator"
    was_enabled = bool(extension_manager.is_extension_enabled(extension_name))
    if not was_enabled:
        extension_manager.set_extension_enabled_immediate(extension_name, True)
    enabled_extension = bool(extension_manager.is_extension_enabled(extension_name))
    if not enabled_extension:
        raise RuntimeError(f"failed to enable {extension_name}")
    validator = get_physx_asset_validator_interface()
    stage_id = d334._stage_id(inner)
    guard_before = d334._snapshot_sim_state(inner)
    capture_by_key = {
        (row["body"], row["name"]): row for row in capture_candidates["parts"]
    }
    changed_keys = {
        (body, name) for body, names in FAILING_PARTS.items() for name in names
    }
    cooked_by_body: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {
        "artifact": "D340_BOTH_CHANNELS_LIVE_COLLIDER_AUDIT",
        "request_order": ["prototype", "instance"],
        "capture_request_order": ["instance", "prototype"],
        "request_order_inverted_across_fresh_processes": True,
        "per_body": {},
        "asset_validator_extension": {
            "name": extension_name,
            "was_enabled": was_enabled,
            "enabled_for_audit": enabled_extension,
        },
    }
    ordinal = 0
    for body in d334.BODY_LABELS:
        usd_inventory = d334._usd_collision_inventory(inner, body)
        enabled = sorted(
            [row for row in usd_inventory if row["collision_enabled"]],
            key=lambda row: row["path"],
        )
        disabled = sorted(
            [row for row in usd_inventory if not row["collision_enabled"]],
            key=lambda row: row["path"],
        )
        expected_paths = _expected_live_paths(manifest, body)
        expected_property_paths = sorted(
            [*expected_paths, d339.LIVE_OLD_COLLIDER_PATHS[body]]
        )
        property_query = d334._property_query_body(inner, simulation_app, body)
        property_rows = sorted(
            property_query["colliders"], key=lambda row: str(row["path"])
        )
        property_paths = [str(row["path"]) for row in property_rows]
        property_by_path = {str(row["path"]): row for row in property_rows}
        manifest_by_name = {row["name"]: row for row in manifest["parts"][body]}
        mass_api = UsdPhysics.MassAPI(
            inner.scene.stage.GetPrimAtPath(d334.BODY_PATHS[body])
        )
        com = mass_api.GetCenterOfMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        axes = mass_api.GetPrincipalAxesAttr().Get()
        live_mass = {
            "mass_kg": float(mass_api.GetMassAttr().Get()),
            "center_of_mass_m": [float(value) for value in com],
            "diagonal_inertia": [float(value) for value in inertia],
            "principal_axes_wxyz": [
                float(axes.GetReal()),
                *[float(value) for value in axes.GetImaginary()],
            ],
        }
        expected_mass = manifest["source_tool_mass_semantics"][body]
        certified_parts: list[dict[str, Any]] = []
        part_rows: list[dict[str, Any]] = []
        surface_pass_count = 0
        volume_pass_count = 0
        fixed_point_pass_count = 0
        for row in enabled:
            name = Path(row["path"]).name
            channels: dict[str, Any] = {}
            # Deliberate inversion relative to capture detects order/cache coupling.
            for channel in ("prototype", "instance"):
                ordinal += 1
                channels[channel] = _safe_cook_channel(
                    inner,
                    row,
                    body,
                    channel=channel,
                    iteration=1,
                    ordinal=ordinal,
                    witness_path=witness_dir / f"{body}_{name}_{channel}.json",
                )
            consensus = _channel_consensus(channels["instance"], channels["prototype"])
            consensus_convex = consensus.get("_consensus")
            expected = manifest_by_name.get(name)
            prop = property_by_path.get(row["path"])
            try:
                authored_source = d334._source_mesh_body_local(inner, row, body)
                authored = d339._canonical_convex(authored_source["_verts_body"])
                authored_error = None
            except Exception as error:
                authored = None
                authored_error = {
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            relative = d339._prim_to_body_transform(inner, row["path"], body)
            live_prim = inner.scene.stage.GetPrimAtPath(row["path"])
            hull_api = PhysxSchema.PhysxConvexHullCollisionAPI(live_prim)
            hull_vertex_limit = hull_api.GetHullVertexLimitAttr().Get()
            hull_min_thickness = hull_api.GetMinThicknessAttr().Get()
            gpu_compatible = bool(
                validator.convex_gpu_compatibility_is_valid(
                    stage_id, PhysicsSchemaTools.sdfPathToInt(row["path"])
                )
            )
            channel_surfaces = {}
            channel_volume_relative = {}
            for channel in ("instance", "prototype"):
                channel_convex = channels[channel].get("_canonical")
                channel_surfaces[channel] = (
                    d339._convex_solid_hausdorff_m(authored, channel_convex)
                    if authored is not None and channel_convex is not None
                    else {
                        "authored_to_live_m": None,
                        "live_to_authored_m": None,
                        "symmetric_m": None,
                        "tolerance_m": LIVE_SURFACE_PARITY_TOL_M,
                        "pass": False,
                    }
                )
                channel_volume_relative[channel] = (
                    abs(float(channel_convex["volume_m3"]) - float(prop["volume_m3"]))
                    / max(abs(float(prop["volume_m3"])), 1.0e-12)
                    if channel_convex is not None and prop is not None
                    else math.inf
                )
            surface_pass = all(
                bool(channel_surfaces[channel]["pass"])
                for channel in ("instance", "prototype")
            )
            volume_pass = all(
                math.isfinite(channel_volume_relative[channel])
                and channel_volume_relative[channel]
                <= PROPERTY_VOLUME_BINDING_REL_TOL
                for channel in ("instance", "prototype")
            )
            key = (body, name)
            fixed_point = {
                "registered_changed_part": key in changed_keys,
                "pass": True,
                "checks": {},
            }
            if key in changed_keys:
                capture_row = capture_by_key.get(key)
                x0 = (
                    capture_row["fixed_point"]["authored_x0"]
                    if capture_row is not None
                    else None
                )
                x1 = (
                    capture_row["fixed_point"]["candidate_x1"]
                    if capture_row is not None
                    else None
                )
                consensus_hash = (
                    consensus_convex["geometry_sha256"]
                    if consensus_convex is not None
                    else None
                )
                authored_hash = authored["geometry_sha256"] if authored is not None else None
                strict_decrease_available = bool(
                    authored is not None
                    and consensus_convex is not None
                    and consensus_hash != authored_hash
                    and d339._directed_convex_solid_distance_m(
                        consensus_convex["vertices"], authored
                    )
                    <= FIXED_POINT_COORD_TOL_M
                    and consensus_convex["vertex_count"] < authored["vertex_count"]
                )
                fixed_checks = {
                    "capture_row_found": capture_row is not None,
                    "attempt3_authored_equals_x1": bool(
                        authored_hash is not None
                        and x1 is not None
                        and authored_hash == x1["geometry_sha256"]
                    ),
                    "instance_Fx1_equals_x1": bool(
                        channels["instance"].get("_canonical") is not None
                        and x1 is not None
                        and channels["instance"]["_canonical"]["geometry_sha256"]
                        == x1["geometry_sha256"]
                    ),
                    "prototype_Fx1_equals_x1": bool(
                        channels["prototype"].get("_canonical") is not None
                        and x1 is not None
                        and channels["prototype"]["_canonical"]["geometry_sha256"]
                        == x1["geometry_sha256"]
                    ),
                    "channel_consensus_equals_x1": bool(
                        consensus_hash is not None
                        and x1 is not None
                        and consensus_hash == x1["geometry_sha256"]
                    ),
                    "no_second_strict_vertex_decrease": not strict_decrease_available,
                    "no_two_cycle_back_to_x0": bool(
                        consensus_hash is not None
                        and x0 is not None
                        and consensus_hash != x0["geometry_sha256"]
                    ),
                }
                fixed_point = {
                    "registered_changed_part": True,
                    "x0_geometry_sha256": None if x0 is None else x0["geometry_sha256"],
                    "x1_geometry_sha256": None if x1 is None else x1["geometry_sha256"],
                    "F_x1_geometry_sha256": consensus_hash,
                    "strict_decrease_available_if_not_fixed": strict_decrease_available,
                    "single_authoring_application": True,
                    "iterative_retry_forbidden": True,
                    "checks": fixed_checks,
                    "pass": all(fixed_checks.values()),
                }
                if fixed_point["pass"]:
                    fixed_point_pass_count += 1
            checks = {
                "manifest_part_found": expected is not None,
                "authored_geometry_available": authored is not None,
                "authored_hash_matches_attempt3_manifest": bool(
                    expected is not None
                    and authored is not None
                    and authored["geometry_sha256"] == expected["geometry_sha256"]
                ),
                "channel_consensus_exact": bool(consensus["pass"]),
                "validate_request_order_prototype_then_instance": (
                    channels["prototype"]["request_order_ordinal"]
                    < channels["instance"]["request_order_ordinal"]
                ),
                "property_query_path_found": prop is not None,
                "property_query_local_position_zero": bool(
                    prop is not None
                    and np.max(np.abs(np.asarray(prop["local_pos_m"], dtype=np.float64)))
                    <= 1.0e-9
                ),
                "property_query_local_rotation_identity": bool(
                    prop is not None
                    and min(
                        np.linalg.norm(
                            np.asarray(prop["local_rot_xyzw"], dtype=np.float64)
                            - np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                        ),
                        np.linalg.norm(
                            np.asarray(prop["local_rot_xyzw"], dtype=np.float64)
                            + np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                        ),
                    )
                    <= 1.0e-9
                ),
                "owner_matches": row["nearest_rigid_body_ancestor"]
                == d334.BODY_PATHS[body],
                "mesh_convex_hull_api": row["type_name"] == "Mesh"
                and row["approximation"] == "convexHull",
                "piece_to_body_transform_identity": bool(relative["identity_pass"]),
                "hull_vertex_limit_frozen_64": int(hull_vertex_limit or -1)
                == DECOMPOSITION_PARAMS["hull_vertex_limit"],
                "min_thickness_frozen_0p0001m": bool(
                    hull_min_thickness is not None
                    and math.isclose(
                        float(hull_min_thickness),
                        DECOMPOSITION_PARAMS["min_thickness_m"],
                        rel_tol=0.0,
                        abs_tol=1.0e-10,
                    )
                ),
                "both_channel_surface_le_0p1mm": surface_pass,
                "property_vs_both_channel_volume_le_5pct": volume_pass,
                "physx_gpu_convex_compatible": gpu_compatible,
                "changed_part_fixed_point_or_preserved": bool(fixed_point["pass"]),
            }
            part_pass = all(checks.values())
            if surface_pass:
                surface_pass_count += 1
            if volume_pass:
                volume_pass_count += 1
            if part_pass and consensus_convex is not None:
                points = d332._fcl_points(hppfcl, consensus_convex["vertices"])
                geometry = hppfcl.Convex.convexHull(points, False, "")
                if geometry is None:
                    raise RuntimeError(f"hppfcl reconstruction failed for {row['path']}")
                certified_parts.append(
                    {
                        "body": body,
                        "path": row["path"],
                        "_vertices": consensus_convex["vertices"],
                        "_triangles": consensus_convex["triangles"],
                        "_geometry": geometry,
                    }
                )
            part_rows.append(
                {
                    "body": body,
                    "name": name,
                    "path": row["path"],
                    "authored_error": authored_error,
                    "channel_consensus": {
                        key: value for key, value in consensus.items() if key != "_consensus"
                    },
                    "channel_surfaces": channel_surfaces,
                    "property_vs_channel_volume_relative_difference": channel_volume_relative,
                    "fixed_point": fixed_point,
                    "hull_vertex_limit_readback": hull_vertex_limit,
                    "hull_min_thickness_readback_m": hull_min_thickness,
                    "checks": checks,
                    "pass": part_pass,
                }
            )
        body_checks = {
            "usd_inventory_exact_65": len(usd_inventory) == 65,
            "usd_enabled_exact_64": len(enabled) == 64,
            "usd_enabled_paths_exact": [row["path"] for row in enabled] == expected_paths,
            "usd_disabled_exact_known_legacy": len(disabled) == 1
            and disabled[0]["path"] == d339.LIVE_OLD_COLLIDER_PATHS[body],
            "property_query_pass": bool(property_query["pass"]),
            "property_query_state_guard": not bool(property_query["state_guard"]["violated"]),
            "property_inventory_exact_65": len(property_rows) == 65,
            "property_paths_exact_64_plus_disabled_legacy": property_paths
            == expected_property_paths,
            "property_paths_unique": len(set(property_paths)) == 65,
            "known_legacy_property_row_present": d339.LIVE_OLD_COLLIDER_PATHS[body]
            in property_paths,
            "legacy_row_not_used_as_active_collision_evidence": True,
            "surface_certified_64_of_64": surface_pass_count == 64,
            "property_volume_bound_64_of_64": volume_pass_count == 64,
            "all_parts_certified_64_of_64": len(certified_parts) == 64,
            "changed_fixed_points_exact": fixed_point_pass_count
            == len(FAILING_PARTS[body]),
            "all_part_checks": bool(part_rows) and all(row["pass"] for row in part_rows),
            "live_mass_semantics_equal": all(
                np.allclose(
                    np.asarray(live_mass[key], dtype=np.float64),
                    np.asarray(expected_mass[key], dtype=np.float64),
                    rtol=0.0,
                    atol=1.0e-12,
                )
                for key in live_mass
            ),
            "property_query_mass_equal": bool(
                property_query.get("rigid_body") is not None
                and math.isclose(
                    float(property_query["rigid_body"]["mass"]),
                    float(expected_mass["mass_kg"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ),
        }
        cooked_by_body[body] = certified_parts
        audit["per_body"][body] = {
            "checks": body_checks,
            "pass": all(body_checks.values()),
            "usd_inventory": usd_inventory,
            "property_query": property_query,
            "expected_property_paths": expected_property_paths,
            "part_checks": part_rows,
            "surface_pass_count": surface_pass_count,
            "volume_pass_count": volume_pass_count,
            "fixed_point_pass_count": fixed_point_pass_count,
            "live_mass_semantics": live_mass,
            "expected_mass_semantics": expected_mass,
        }
    validator_guard = d334._state_guard(
        guard_before, d334._snapshot_sim_state(inner)
    )
    audit["asset_validator_state_guard"] = validator_guard
    audit["checks"] = {
        "asset_validator_extension": enabled_extension,
        "asset_validator_state_guard": not bool(validator_guard["violated"]),
        "request_order_inverted": all(
            row["checks"]["validate_request_order_prototype_then_instance"]
            for body_row in audit["per_body"].values()
            for row in body_row["part_checks"]
        ),
        "surface_certified_128_of_128": sum(
            row["surface_pass_count"] for row in audit["per_body"].values()
        )
        == 128,
        "property_volume_bound_128_of_128": sum(
            row["volume_pass_count"] for row in audit["per_body"].values()
        )
        == 128,
        "live_parts_certified_128_of_128": sum(
            len(cooked_by_body[body]) for body in d334.BODY_LABELS
        )
        == 128,
        "both_bodies_pass": all(
            audit["per_body"][body]["pass"] for body in d334.BODY_LABELS
        ),
    }
    audit["pass"] = all(audit["checks"].values())
    return cooked_by_body, audit


def _write_final_markdown(path: Path, summary: dict[str, Any]) -> None:
    live = summary.get("live_collider_audit", {})
    lines = [
        "# D340 fixed-point live-authoring repair",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        f"- Attempt3 build: `{summary.get('attempt3_build', {}).get('pass')}`",
        f"- Live 128/128 audit: `{live.get('pass')}`",
        f"- Frozen zero-step target: `{summary.get('representation_gate', {}).get('target_clear_and_faithful')}`",
        f"- Controlled physics steps: `{summary.get('controlled_physics_steps')}`",
        f"- Artifact contract: `{summary.get('artifact_contract', {}).get('pass')}`",
        "",
        summary.get("interpretation", "No interpretation recorded") + ".",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_validate(args: argparse.Namespace, simulation_app: Any) -> dict[str, Any]:
    d332._runtime_versions()
    capture_summary, capture_candidates = _load_capture_for_validate(args.out_dir)
    attempt2_before = _attempt2_integrity()
    build = _author_attempt3(args, capture_candidates)
    if not build["pass"]:
        attempt2_after = _attempt2_integrity()
        failure_path = (
            args.out_dir
            / "collision_asset/attempt3/d340_attempt3_build_failure.json"
        )
        failure = {
            "verdict": VERDICT_BUILD_FAIL,
            "artifact": "D340_ATTEMPT3_BUILD_HARD_STOP_BEFORE_RUNTIME_ENV",
            "interpretation": (
                "attempt3 failed the registered one-authoring/preservation contract; "
                "no runtime environment, live cook, target query, or physics was created"
            ),
            "process_identity": {
                "pid": os.getpid(),
                "nonce": args.process_nonce,
            },
            "attempt3_build_manifest": build.get("manifest_path"),
            "attempt3_build_manifest_sha256": build.get("manifest_sha256"),
            "failed_checks": [
                name for name, passed in build.get("checks", {}).items() if not passed
            ],
            "attempt2_integrity_before": attempt2_before,
            "attempt2_integrity_after": attempt2_after,
            "controlled_physics_steps": 0,
            "runtime_env_created": False,
        }
        _json_dump(failure_path, failure)
        artifact_checks = {
            "attempt3_manifest_nonzero": bool(
                build.get("manifest_path")
                and (REPO / build["manifest_path"]).is_file()
                and (REPO / build["manifest_path"]).stat().st_size > 0
            ),
            "structured_build_failure_nonzero": failure_path.is_file()
            and failure_path.stat().st_size > 0,
            "capture_png_retained_nonzero": CAPTURE_PNG_PATH.is_file()
            and CAPTURE_PNG_PATH.stat().st_size > 0,
            "capture_rrd_retained_nonzero": CAPTURE_RRD_PATH.is_file()
            and CAPTURE_RRD_PATH.stat().st_size > 0,
            "attempt2_still_immutable": bool(attempt2_after["pass"]),
            "d338_attempt1_still_immutable": bool(
                d339._d338_attempt1_integrity()["pass"]
            ),
        }
        summary = {
            "verdict": VERDICT_BUILD_FAIL,
            "scientific_verdict_before_artifact_gate": VERDICT_BUILD_FAIL,
            "interpretation": failure["interpretation"],
            "active_case": "G0a cylinder D34xH90 fixed-point live-authoring repair",
            "new_variables": NEW_VARIABLES,
            "physical_variables_changed": ["failing_part_fixed_point_geometry"],
            "parameters_increased": [],
            "stage": "validate_build_stop",
            "process_identity": failure["process_identity"],
            "capture_summary_path": _rel(CAPTURE_SUMMARY_PATH),
            "capture_summary_sha256": _sha256(CAPTURE_SUMMARY_PATH),
            "attempt3_build": build,
            "structured_build_failure": _rel(failure_path),
            "controlled_physics_steps": 0,
            "runtime_env_created": False,
            "live_collider_audit": {"pass": False, "not_run": True},
            "representation_gate": {"not_run": True, "physics_licensed": False},
            "artifact_contract": {
                "checks": artifact_checks,
                "pass": all(artifact_checks.values()),
            },
            "outcome_guards": {
                "g0a_pass": False,
                "runtime_env_created": False,
                "physics_executed": False,
                "ten_trial_run": False,
                "ladder_promoted": False,
            },
        }
        _json_dump(FINAL_SUMMARY_PATH, summary)
        _write_final_markdown(
            args.out_dir / "g0a_d340_fixed_point_live_authoring_repair_summary.md",
            summary,
        )
        return summary
    variant_robot_usd = REPO / build["variant_robot_usd"]
    args.robot_usd_path = variant_robot_usd
    d334_summary = json.loads(d337.D334_SUMMARY.read_text(encoding="utf-8"))
    d336_summary = json.loads(d337.D336_SUMMARY.read_text(encoding="utf-8"))
    inner = d333._make_runtime_env(args)
    artifact_errors: list[dict[str, str]] = []
    try:
        inner.reset(seed=int(args.seed))
        counter_start = int(inner._sim_step_counter)
        stage_contract = d333._stage_contract(inner)
        sensor_contract, _filter_map = d333._sensor_contract(inner)
        try:
            raw_shapes, raw_source_contract = d339._build_retained_raw_shapes(
                inner, d334_summary
            )
        except Exception as error:
            raw_shapes = []
            raw_source_contract = {
                "pass": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        witness_dir = args.out_dir / "d340_validate_cook_witnesses"
        witness_dir.mkdir(parents=True, exist_ok=False)
        try:
            cooked_by_body, live_audit = _validate_live_attempt3(
                inner,
                simulation_app,
                build,
                capture_candidates,
                witness_dir,
            )
        except Exception as error:
            cooked_by_body = {body: [] for body in d334.BODY_LABELS}
            live_audit = {
                "artifact": "D340_BOTH_CHANNELS_LIVE_COLLIDER_AUDIT_EXCEPTION_STOP",
                "pass": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        _json_dump(args.out_dir / "d340_live_collider_audit.json", live_audit)
        scene_checks = {
            "attempt3_build": bool(build["pass"]),
            "stage_contract": bool(stage_contract["hard_contract_pass"]),
            "sensor_contract": bool(sensor_contract["hard_contract_pass"]),
            "retained_raw_source_contract": bool(raw_source_contract["pass"]),
            "live_collider_audit_128_of_128": bool(live_audit["pass"]),
        }
        raw_prerequisites = bool(
            scene_checks["stage_contract"]
            and scene_checks["sensor_contract"]
            and scene_checks["retained_raw_source_contract"]
        )
        if raw_prerequisites:
            d336_rescore = d337._load_d336_rescore()
            cache = d337._Cache(inner, raw_shapes)
            controls = d337._negative_controls(
                inner,
                raw_shapes,
                d334_summary,
                d336_summary,
                d336_rescore,
                cache,
            )
            candidate = d337._evaluate_candidate(
                inner,
                raw_shapes,
                OLD_RADIAL_NM,
                OLD_TANGENT_NM,
                Q5_OPEN_RAD,
                stage="d340_fixed_point_frozen_target",
            )
        else:
            controls = {
                "artifact": "D340_D337_CONTROLS_SKIPPED",
                "pass": False,
                "reason": "stage/sensor/raw prerequisite failed",
            }
            candidate = d339._fallback_candidate_without_raw(
                inner, "D340 retained raw-source contract failed"
            )
        _json_dump(args.out_dir / "d340_d337_controls.json", controls)

        if all(scene_checks.values()):
            representation_gate, decision_raw, decision_cooked = d339._representation_gate(
                inner,
                raw_shapes,
                cooked_by_body,
                candidate,
                controls,
                live_audit,
            )
            representation_gate["artifact"] = "D340_FIXED_POINT_ZERO_STEP_REPRESENTATION_GATE"
            representation_gate["physics_licensed"] = False
            representation_gate["prephysics_support_licensed"] = bool(
                representation_gate["contract_pass"]
                and representation_gate["target_clear_and_faithful"]
            )
            representation_gate["physics_forbidden_in_d340"] = True
        elif raw_prerequisites:
            decision_raw = d336._exact_raw_metrics(
                inner, raw_shapes, "d340_frozen_target_raw_contract_stop"
            )
            decision_cooked = {
                "pose": "d340_cooked_union_not_certified",
                "queries": [],
                "invalid_reason": "128/128 live contract failed; partial union not queried",
            }
            anchor_checks = d339._frozen_anchor_checks(candidate)
            representation_gate = {
                "artifact": "D340_FIXED_POINT_ZERO_STEP_REPRESENTATION_GATE",
                "checks": {
                    "scene_contract": False,
                    "live_collider_audit_128_of_128": bool(live_audit["pass"]),
                    "d337_controls": bool(controls["pass"]),
                    "frozen_anchor": all(anchor_checks.values()),
                },
                "anchor_checks": anchor_checks,
                "per_body": {},
                "contract_pass": False,
                "target_clear_and_faithful": False,
                "prephysics_support_licensed": False,
                "physics_licensed": False,
                "physics_forbidden_in_d340": True,
                "controlled_physics_steps": 0,
                "structured_stop_reason": "128/128 live contract failed; cooked union intentionally not queried",
            }
        else:
            decision_raw = {
                "pose": "d340_raw_not_certified",
                "queries": [],
                "invalid_reason": "raw prerequisite failed",
            }
            decision_cooked = {
                "pose": "d340_cooked_not_queried",
                "queries": [],
                "invalid_reason": "raw prerequisite failed",
            }
            representation_gate = {
                "artifact": "D340_FIXED_POINT_ZERO_STEP_REPRESENTATION_GATE",
                "checks": {"scene_contract": False},
                "anchor_checks": {},
                "per_body": {},
                "contract_pass": False,
                "target_clear_and_faithful": False,
                "prephysics_support_licensed": False,
                "physics_licensed": False,
                "physics_forbidden_in_d340": True,
                "controlled_physics_steps": 0,
                "structured_stop_reason": "raw prerequisite failed; no union query",
            }
        counter_end = int(inner._sim_step_counter)
        counter_unchanged = counter_end == counter_start
        representation_gate["global_sim_counter"] = {
            "start": counter_start,
            "end": counter_end,
            "unchanged": counter_unchanged,
        }
        representation_gate.setdefault("checks", {})["global_sim_counter_unchanged"] = counter_unchanged
        if not counter_unchanged:
            representation_gate["contract_pass"] = False
            representation_gate["target_clear_and_faithful"] = False
            representation_gate["prephysics_support_licensed"] = False
        _json_dump(args.out_dir / "d340_representation_gate.json", representation_gate)

        if not build["pass"]:
            scientific_verdict = VERDICT_BUILD_FAIL
            interpretation = "attempt3 failed the one-authoring build/preservation contract"
        elif not all(scene_checks.values()) or not representation_gate["contract_pass"]:
            scientific_verdict = VERDICT_LIVE_FAIL
            interpretation = "the exact 65-row property or 128/128 two-channel live-shape contract failed"
        elif not representation_gate["target_clear_and_faithful"]:
            scientific_verdict = VERDICT_NOT_CLEAR
            interpretation = "the fully certified union failed the frozen zero-step clearance/task-fidelity gate"
        else:
            scientific_verdict = VERDICT_SUPPORTED
            interpretation = "the one-authoring fixed point and frozen zero-step target are supported; physics remains a later case"

        decision_png = args.out_dir / "d340_fixed_point_representation_decision.png"
        try:
            if decision_raw.get("queries") and decision_cooked.get("queries"):
                d339._write_representation_figure(
                    decision_png,
                    "D340 fixed-point attempt3: frozen zero-step raw vs live convex union",
                    inner,
                    raw_shapes,
                    cooked_by_body,
                    decision_raw,
                    decision_cooked,
                    candidate["_canonical"],
                )
            else:
                d339._write_contract_stop_figure(
                    decision_png,
                    title="D340 fixed-point pre-physics contract STOP",
                    scene_checks=scene_checks,
                    candidate=candidate,
                )
        except Exception as error:
            artifact_errors.append(
                {
                    "artifact": "decision_png",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        try:
            marker_status = draw_frames(
                candidate["_frames"], prim_path="/World/D340FixedPointFrames"
            )
        except Exception as error:
            marker_status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            artifact_errors.append(
                {
                    "artifact": "frame_markers",
                    "error": marker_status["error"],
                    "traceback": traceback.format_exc(),
                }
            )
        rrd_path = args.out_dir / "d340_fixed_point_trace.rrd"
        try:
            rrd_status = log_rerun(
                rrd_path,
                frames=candidate["_frames"],
                joint_state={
                    "label": "d340_fixed_point_live_authoring_validate",
                    "object": "cylinder_d34_h90",
                    "request_order": "prototype_then_instance",
                    "prephysics_support_licensed": representation_gate["prephysics_support_licensed"],
                    "controlled_physics_steps_total": 0,
                },
                joint_trace=[d339._decision_trace_row(inner, candidate)],
                urdf_path=args.urdf_path,
                live_viewer=False,
                app_id="roarm_g0a_d340_fixed_point_validate",
            )
        except Exception as error:
            rrd_status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            artifact_errors.append(
                {
                    "artifact": "rrd",
                    "error": rrd_status["error"],
                    "traceback": traceback.format_exc(),
                }
            )
        rrd_nonzero = rrd_path.is_file() and rrd_path.stat().st_size > 0
        witness_files = sorted(witness_dir.glob("*.json"))
        expected_witness_names = {
            f"{body}_part_{index:03d}_{channel}.json"
            for body in d334.BODY_LABELS
            for index in range(64)
            for channel in ("instance", "prototype")
        }
        observed_witness_names = {path.name for path in witness_files}
        witness_hashes = {
            path.name: _sha256(path) for path in witness_files
        }
        witness_manifest_path = (
            args.out_dir / "d340_validate_cook_witness_manifest.json"
        )
        witness_manifest = {
            "artifact": "D340_VALIDATE_BOTH_CHANNEL_WITNESS_MANIFEST",
            "expected_filename_count": len(expected_witness_names),
            "observed_filename_count": len(observed_witness_names),
            "expected_filenames": sorted(expected_witness_names),
            "observed_filenames": sorted(observed_witness_names),
            "sha256": witness_hashes,
            "exact_filename_set": observed_witness_names == expected_witness_names,
            "pass": bool(
                observed_witness_names == expected_witness_names
                and len(witness_hashes) == 256
            ),
        }
        _json_dump(witness_manifest_path, witness_manifest)
        artifact_checks = {
            "decision_png_nonzero": decision_png.is_file() and decision_png.stat().st_size > 0,
            "frame_markers_ok": bool(marker_status.get("ok")),
            "rrd_ok": bool(rrd_status.get("ok")),
            "rrd_nonzero": rrd_nonzero,
            "both_channel_witness_filename_set_exact_256": bool(
                witness_manifest["pass"]
            ),
            "both_channel_witness_manifest_nonzero": witness_manifest_path.is_file()
            and witness_manifest_path.stat().st_size > 0,
            "no_artifact_errors": not artifact_errors,
        }
        artifact_pass = all(artifact_checks.values())
        final_verdict = scientific_verdict if artifact_pass else VERDICT_VIZ_FAIL
        attempt2_after = _attempt2_integrity()
        if not attempt2_after["pass"] or not d339._d338_attempt1_integrity()["pass"]:
            final_verdict = VERDICT_LIVE_FAIL
            interpretation = "immutable D338/D339 evidence changed during D340"
        summary = {
            "verdict": final_verdict,
            "scientific_verdict_before_artifact_gate": scientific_verdict,
            "interpretation": interpretation,
            "active_case": "G0a cylinder D34xH90 fixed-point live-authoring repair",
            "new_variables": NEW_VARIABLES,
            "physical_variables_changed": ["failing_part_fixed_point_geometry"],
            "parameters_increased": [],
            "stage": "validate",
            "process_identity": {
                "pid": os.getpid(),
                "nonce": args.process_nonce,
                "capture_pid": capture_summary.get("process_identity", {}).get("pid"),
                "capture_nonce": capture_summary.get("process_identity", {}).get("nonce"),
                "fresh_process_pid_distinct": os.getpid()
                != int(capture_summary.get("process_identity", {}).get("pid", -1)),
                "fresh_process_nonce_distinct": args.process_nonce
                != capture_summary.get("process_identity", {}).get("nonce"),
            },
            "capture_summary_path": _rel(CAPTURE_SUMMARY_PATH),
            "capture_summary_sha256": _sha256(CAPTURE_SUMMARY_PATH),
            "capture_candidate_path": _rel(CAPTURE_CANDIDATES_PATH),
            "capture_candidate_sha256": _sha256(CAPTURE_CANDIDATES_PATH),
            "attempt3_build": build,
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "raw_source_contract": raw_source_contract,
            "scene_checks": scene_checks,
            "live_collider_audit": live_audit,
            "d337_controls": controls,
            "frozen_candidate": d337._candidate_public(candidate),
            "frozen_candidate_alignment": candidate["_alignment"],
            "representation_gate": representation_gate,
            "decision_raw": decision_raw,
            "decision_cooked": decision_cooked,
            "controlled_physics_steps": 0,
            "simulation_counter": {
                "start": counter_start,
                "end": counter_end,
                "unchanged": counter_unchanged,
            },
            "attempt2_integrity_before": attempt2_before,
            "attempt2_integrity_after": attempt2_after,
            "visualization": {
                "decision_png": _rel(decision_png) if decision_png.is_file() else None,
                "marker_status": marker_status,
                "rrd_status": rrd_status,
                "artifact_errors": artifact_errors,
            },
            "validate_cook_witness_manifest": _rel(witness_manifest_path),
            "validate_cook_witness_manifest_sha256": _sha256(
                witness_manifest_path
            ),
            "artifact_contract": {"checks": artifact_checks, "pass": artifact_pass},
            "outcome_guards": {
                "g0a_pass": False,
                "physics_executed": False,
                "ten_trial_run": False,
                "ladder_promoted": False,
                "canonical_asset_changed": False,
                "d339_attempt2_changed": False,
                "separate_settle_case_required": final_verdict == VERDICT_SUPPORTED,
            },
            "non_goals_respected": [
                "no raw STL/URDF/canonical USD rewrite",
                "no decomposition/target/q5/IK/solver/threshold increase",
                "no physics/settle/10-trial/G0b/RL/PPO/ladder",
                "no iterative authoring retry",
            ],
        }
        _json_dump(FINAL_SUMMARY_PATH, summary)
        _write_final_markdown(
            args.out_dir / "g0a_d340_fixed_point_live_authoring_repair_summary.md",
            summary,
        )
        return summary
    finally:
        inner.close()


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=("capture", "validate"), required=True)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--urdf_path", type=Path, default=d333.DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=33201)


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    args.process_nonce = secrets.token_hex(16)

    try:
        preflight = (
            _preflight_capture(args)
            if args.stage == "capture"
            else _preflight_validate(args)
        )
    except Exception:
        traceback.print_exc()
        print("D340 pre-AppLauncher preregistration preflight raised; no Isaac app launched.", flush=True)
        return 1
    if not preflight["pass"]:
        print(
            "D340 pre-AppLauncher preregistration/inventory/hash refusal; "
            f"stage={args.stage} checks={preflight['checks']}",
            flush=True,
        )
        return 1

    launcher = AppLauncher(args)
    try:
        try:
            summary = (
                _run_capture(args, launcher.app)
                if args.stage == "capture"
                else _run_validate(args, launcher.app)
            )
            stage_success = (
                bool(summary.get("pass"))
                if args.stage == "capture"
                else summary.get("verdict") == VERDICT_SUPPORTED
            )
            print(
                f"{summary['verdict']}: stage={args.stage} "
                f"stage_success={stage_success} "
                "controlled_steps=0",
                flush=True,
            )
            return 0 if stage_success else 1
        except Exception:
            error = traceback.format_exc()
            traceback.print_exc()
            if args.stage == "capture":
                abort_path = args.out_dir / "d340_capture_abort.json"
            else:
                attempt_dir = args.out_dir / "collision_asset" / "attempt3"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                abort_path = attempt_dir / "d340_validate_abort.json"
            if not abort_path.exists():
                _json_dump(
                    abort_path,
                    {
                        "verdict": VERDICT_CAPTURE_FAIL
                        if args.stage == "capture"
                        else VERDICT_BUILD_FAIL,
                        "stage": args.stage,
                        "interpretation": "D340 invocation aborted before a clean registered outcome",
                        "attempt2_integrity_after_abort": _attempt2_integrity(),
                        "d338_attempt1_integrity_after_abort": d339._d338_attempt1_integrity(),
                        "controlled_physics_steps": 0,
                        "error": error,
                    },
                )
            return 1
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
