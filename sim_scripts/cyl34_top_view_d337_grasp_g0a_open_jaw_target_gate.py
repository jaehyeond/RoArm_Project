#!/usr/bin/env python3
"""D337 open-jaw target gate for cylinder grasp G0a.

Exactly one new physical variable: the commanded gripper joint value at every
exact-state write (``q5 = 1.5413 rad`` URDF = 98.1% of the ``1.571`` open
limit, i.e. approximately ``86.6deg`` real opening under the D322 mapping
``88.3deg real <-> 1.571rad URDF`` — a deliberately sub-maximum opening).
The URDF convention is ``q5=0`` closed / ``1.571`` open; the D325-family text
"open gripper q5=0" was a convention error, so every D330-D336 write placed a
closed moving jaw into the grasp volume.  D337 re-runs the pre-registered
2,629-key r/t scan with the jaw open, using the unchanged D336 exact metric
and clear rule, and licenses one D333-style 200-step sole-support static
settle only if a candidate passes.  No 10-trial run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
from sim_scripts import cyl34_top_view_d335_grasp_g0a_target_family_repair as d335
from sim_scripts import cyl34_top_view_d336_grasp_g0a_finite_grid_caveat_discriminator as d336


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d337"
D334_SUMMARY = d336.D334_SUMMARY
D335_CSV = d336.D335_CSV
D336_SUMMARY = (
    REPO / "claudedocs/runtime_logs/grasp_track/g0a_d336/g0a_d336_finite_grid_caveat_summary.json"
)
D336_RESCORE_CSV = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d336/d336_exact_rescore.csv"
D335_SUMMARY = (
    REPO / "claudedocs/runtime_logs/grasp_track/g0a_d335/g0a_d335_target_family_repair_summary.json"
)

PIN_D334_SUMMARY_SHA256 = d336.PIN_D334_SUMMARY_SHA256
PIN_D335_SUMMARY_SHA256 = d336.PIN_D335_SUMMARY_SHA256
PIN_D335_CSV_SHA256 = d336.PIN_D335_CSV_SHA256
PIN_D336_SUMMARY_SHA256 = "f449801302bd21769aadc43e67fd6bb884071d29d32b9b1e29f0166297220f00"
PIN_D336_RESCORE_CSV_SHA256 = "5f76bde76cd0578883fafa952214a4345c79ba1cca0c5b685da1fd2b352a3853"

# 1.5413rad URDF = 98.1% of the 1.571rad open limit (~86.6deg real opening
# under the D322 mapping "real 88.3deg <-> URDF 1.571rad"); NOT the real max.
Q5_OPEN_RAD = 1.5413
Q5_CLOSED_RAD = 0.0
EXPECTED_GRID_COUNT = 2629
PARITY_TOL_MM = 0.05
SCOPING_OPEN_GRIPPER_MM = 11.175
SCOPING_OPEN_TOL_MM = 0.5
LINK5_INVARIANT_TOL_MM = 0.05
GRID_PARITY_KEYS_UM = ((14_600, 13_900), (0, 9_000))
OLD_RADIAL_NM = 7_000_000
OLD_TANGENT_NM = 11_000_000
TARGET_SETTLE_STEPS = 200

VERDICT_CONTRACT_FAIL = "D337_G0A_PREPHYSICS_CONTRACT_FAIL_STOP"
VERDICT_NO_FEASIBLE = "D337_G0A_OPEN_JAW_NO_FEASIBLE_CLEAR_STOP"
VERDICT_STATIC_PASS = "D337_G0A_OPEN_JAW_STATIC_REPAIR_SUPPORTED_STOP"
VERDICT_COLLIDER_BLOCKED = "D337_G0A_RAW_CLEAR_LIVE_COLLIDER_BLOCKED_STOP"
VERDICT_STATIC_MIXED = "D337_G0A_STATIC_RUNTIME_MIXED_STOP"
VERDICT_VIZ_FAIL = "D337_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP"

D335_TO_D337_VERDICT = {
    d335.VERDICT_STATIC_PASS: VERDICT_STATIC_PASS,
    d335.VERDICT_COLLIDER_BLOCKED: VERDICT_COLLIDER_BLOCKED,
    d335.VERDICT_STATIC_MIXED: VERDICT_STATIC_MIXED,
}

FLAT_KEYS = ("stage", "q5_rad") + d336.FLAT_KEYS[1:]


def _rel(path: Path) -> str:
    return d332._rel(path)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    d332._json_dump(path, payload)


def _canonical_with_q5(radial_um: float, tangent_um: float, q5_rad: float) -> dict[str, Any]:
    canonical = d335._canonical_for_offsets(radial_um, tangent_um)
    q_deg = np.asarray(canonical["commanded_joint_deg"], dtype=np.float64)
    q_deg[5] = math.degrees(q5_rad)
    canonical["commanded_joint_deg"] = q_deg.tolist()
    canonical["commanded_joint_rad"] = np.radians(q_deg).tolist()
    canonical["q5_rad"] = float(q5_rad)
    return canonical


def _evaluate_candidate(
    inner: Any,
    shapes: list[dict[str, Any]],
    radial_nm: int,
    tangent_nm: int,
    q5_rad: float,
    *,
    stage: str,
) -> dict[str, Any]:
    radial_um = radial_nm / 1000.0
    tangent_um = tangent_nm / 1000.0
    canonical = _canonical_with_q5(radial_um, tangent_um, q5_rad)
    counter_before = int(inner._sim_step_counter)
    command = d332._write_exact_state(
        inner,
        np.asarray(canonical["commanded_joint_rad"], dtype=np.float64),
        d332.OBJECT_CENTER_LOCAL_M,
    )
    counter_after = int(inner._sim_step_counter)
    distances = d336._exact_raw_metrics(inner, shapes, f"d337_{stage}")
    queries = {row["body"]: row for row in distances["queries"]}
    alignment, frames = d335._alignment_at_current(inner, canonical)
    canonical["offline_witness_cylinder_local_m"] = queries["gripper_link"][
        "nearest_point_cylinder_m"
    ]
    radial_m = radial_nm * 1.0e-9
    shift_mm = math.hypot(radial_nm - OLD_RADIAL_NM, tangent_nm - OLD_TANGENT_NM) / 1.0e6
    legacy_checks = {
        "ik_converged": bool(canonical["ik"]["converged"]),
        "commanded_tcp_error_le_5mm": float(canonical["commanded_tcp_error_mm"])
        <= d335.TARGET_TCP_GATE_M * 1000.0,
        "live_exact_written_tcp_error_le_5mm": bool(alignment["pass_tcp_pose"]),
        "jaw_tangent_le_15deg": bool(alignment["pass_jaw_tangent"]),
        "fixed_jaw_gap_0_to_5mm": bool(alignment["pass_fixed_jaw_gap"]),
        "no_fixed_jaw_proxy_penetration": bool(alignment["pass_no_penetration"]),
        "contact_at_least_15mm_below_top": bool(alignment["pass_contact_height"]),
        "anti_retreat_tip_past_near_face_nonnegative": bool(
            d332.CYLINDER_RADIUS_M - radial_m >= -1.0e-12
        ),
    }
    raw_checks = {
        body: bool(queries[body]["clear_pass"] and queries[body]["consistency_hard_check_pass"])
        for body in d334.BODY_LABELS
    }
    exact_values = [queries[body]["exact_signed_distance_mm"] for body in d334.BODY_LABELS]
    exact_consistent = bool(
        all(queries[body]["exact_consistent"] for body in d334.BODY_LABELS)
        and all(value is not None for value in exact_values)
    )
    exact_min_clearance_mm = (
        min(float(value) for value in exact_values) if exact_consistent else None
    )
    min_clearance_mm = min(float(queries[body]["signed_distance_mm"]) for body in d334.BODY_LABELS)
    row = {
        "stage": stage,
        "q5_rad": float(q5_rad),
        "radial_offset_nm": int(radial_nm),
        "tangent_offset_nm": int(tangent_nm),
        "radial_offset_mm": radial_nm / 1.0e6,
        "tangent_offset_mm": tangent_nm / 1.0e6,
        "radial_tip_past_near_face_mm": (d332.CYLINDER_RADIUS_M - radial_m) * 1000.0,
        "shift_from_old_mm": shift_mm,
        "ik_converged": bool(canonical["ik"]["converged"]),
        "commanded_tcp_error_mm": float(canonical["commanded_tcp_error_mm"]),
        "jaw_tangent_error_deg": float(alignment["jaw_tangent_error_deg"]),
        "fixed_jaw_face_gap_mm": float(alignment["fixed_jaw_face_gap_mm"]),
        "fixed_jaw_penetration_mm": float(alignment["fixed_jaw_penetration_mm"]),
        "contact_point_below_top_mm": float(alignment["contact_point_below_top_mm"]),
        "link5_raw_signed_distance_mm": float(queries["link5"]["signed_distance_mm"]),
        "link5_raw_state": queries["link5"]["overlap_state"],
        "link5_exact_signed_distance_mm": queries["link5"]["exact_signed_distance_mm"],
        "link5_epa_contact_count": queries["link5"]["epa_contact_count"],
        "link5_epa_cap_saturated": queries["link5"]["epa_cap_saturated"],
        "gripper_link_raw_signed_distance_mm": float(
            queries["gripper_link"]["signed_distance_mm"]
        ),
        "gripper_link_raw_state": queries["gripper_link"]["overlap_state"],
        "gripper_link_exact_signed_distance_mm": queries["gripper_link"][
            "exact_signed_distance_mm"
        ],
        "gripper_link_epa_contact_count": queries["gripper_link"]["epa_contact_count"],
        "gripper_link_epa_cap_saturated": queries["gripper_link"]["epa_cap_saturated"],
        "min_raw_clearance_mm": min_clearance_mm,
        "exact_min_clearance_mm": exact_min_clearance_mm,
        "exact_consistent": exact_consistent,
        "legacy_checks": legacy_checks,
        "raw_checks": raw_checks,
        "legacy_alignment_pass": all(legacy_checks.values()),
        "raw_tool_clear_pass": all(raw_checks.values()),
        "sim_step_counter_before": counter_before,
        "sim_step_counter_after": counter_after,
        "sim_step_counter_unchanged": counter_before == counter_after,
        "pass": bool(
            all(legacy_checks.values())
            and all(raw_checks.values())
            and counter_before == counter_after
        ),
        "_canonical": canonical,
        "_distance_set": distances,
        "_alignment": alignment,
        "_frames": frames,
        "_command": command,
    }
    return row


def _candidate_public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [{key: row.get(key) for key in FLAT_KEYS} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FLAT_KEYS))
        writer.writeheader()
        writer.writerows(flat)


class _Cache:
    def __init__(self, inner: Any, shapes: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._shapes = shapes
        self.by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []

    def evaluate(self, radial_nm: int, tangent_nm: int, q5_rad: float, stage: str) -> dict[str, Any]:
        key = (int(radial_nm), int(tangent_nm), int(round(q5_rad * 1.0e6)))
        cached = self.by_key.get(key)
        if cached is not None:
            return cached
        row = _evaluate_candidate(
            self._inner, self._shapes, key[0], key[1], q5_rad, stage=stage
        )
        self.by_key[key] = row
        self.rows.append(row)
        return row


def _load_d336_rescore() -> dict[tuple[int, int], dict[str, str]]:
    rows: dict[tuple[int, int], dict[str, str]] = {}
    with D336_RESCORE_CSV.open("r", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            key = (
                int(round(float(record["radial_offset_mm"]) * 1000.0)),
                int(round(float(record["tangent_offset_mm"]) * 1000.0)),
            )
            rows[key] = record
    return rows


def _negative_controls(
    inner: Any,
    shapes: list[dict[str, Any]],
    d334_source: dict[str, Any],
    d336_summary: dict[str, Any],
    d336_rescore: dict[tuple[int, int], dict[str, str]],
    cache: _Cache,
) -> dict[str, Any]:
    # Control 1: closed-jaw legacy bit-parity vs D334 (unchanged D335 evaluator).
    legacy_payload, _legacy_candidate = d335._negative_control(inner, shapes, d334_source)

    # Control 2: closed-jaw exact-EPA parity vs the pinned D336 value.
    expected_epa_mm = float(
        d336_summary["negative_controls"]["old_target_exact_layer"][
            "observed_epa_max_abs_depth_mm"
        ]
    )
    closed_old = cache.evaluate(OLD_RADIAL_NM, OLD_TANGENT_NM, Q5_CLOSED_RAD, "closed_old_control")
    observed_epa = closed_old["gripper_link_exact_signed_distance_mm"]
    exact_layer = {
        "expected_d336_epa_depth_mm": expected_epa_mm,
        "observed_exact_signed_distance_mm": observed_epa,
        "tolerance_mm": PARITY_TOL_MM,
        "pass": bool(
            observed_epa is not None
            and abs(abs(float(observed_epa)) - expected_epa_mm) <= PARITY_TOL_MM
            and closed_old["gripper_link_raw_state"] == "overlap"
        ),
    }

    # Control 3: closed-jaw grid parity vs the pinned D336 rescore CSV.
    def _float_or_none(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _abs_delta(a: Any, b: Any) -> float | None:
        fa, fb = _float_or_none(a), _float_or_none(b)
        return None if fa is None or fb is None else abs(fa - fb)

    grid_parity: dict[str, Any] = {"rows": [], "tolerance_mm": PARITY_TOL_MM}
    for r_um, t_um in GRID_PARITY_KEYS_UM:
        reference = d336_rescore[(r_um, t_um)]
        observed = cache.evaluate(r_um * 1000, t_um * 1000, Q5_CLOSED_RAD, "grid_parity_control")
        entry = {
            "radial_offset_um": r_um,
            "tangent_offset_um": t_um,
            "gripper_bvh_delta_mm": _abs_delta(
                observed["gripper_link_raw_signed_distance_mm"],
                reference["gripper_link_raw_signed_distance_mm"],
            ),
            "link5_bvh_delta_mm": _abs_delta(
                observed["link5_raw_signed_distance_mm"],
                reference["link5_raw_signed_distance_mm"],
            ),
            "exact_delta_mm": _abs_delta(
                observed["exact_min_clearance_mm"], reference["exact_min_clearance_mm"]
            ),
            "states_equal": (
                observed["link5_raw_state"] == reference["link5_raw_state"]
                and observed["gripper_link_raw_state"] == reference["gripper_link_raw_state"]
            ),
        }
        entry["pass"] = bool(
            entry["gripper_bvh_delta_mm"] is not None
            and entry["gripper_bvh_delta_mm"] <= PARITY_TOL_MM
            and entry["link5_bvh_delta_mm"] is not None
            and entry["link5_bvh_delta_mm"] <= PARITY_TOL_MM
            and entry["exact_delta_mm"] is not None
            and entry["exact_delta_mm"] <= PARITY_TOL_MM
            and entry["states_equal"]
        )
        grid_parity["rows"].append(entry)
    grid_parity["pass"] = all(entry["pass"] for entry in grid_parity["rows"])

    # Control 4: open-jaw scoping cross-check at the old target.
    open_old = cache.evaluate(OLD_RADIAL_NM, OLD_TANGENT_NM, Q5_OPEN_RAD, "open_old_control")
    link5_delta = abs(
        float(open_old["link5_raw_signed_distance_mm"])
        - float(closed_old["link5_raw_signed_distance_mm"])
    )
    scoping = {
        "expected_open_gripper_exact_mm": SCOPING_OPEN_GRIPPER_MM,
        "tolerance_mm": SCOPING_OPEN_TOL_MM,
        "observed_open_gripper_exact_mm": open_old["gripper_link_exact_signed_distance_mm"],
        "link5_q5_independence_delta_mm": link5_delta,
        "pass": bool(
            open_old["gripper_link_exact_signed_distance_mm"] is not None
            and abs(
                float(open_old["gripper_link_exact_signed_distance_mm"])
                - SCOPING_OPEN_GRIPPER_MM
            )
            <= SCOPING_OPEN_TOL_MM
            and link5_delta <= LINK5_INVARIANT_TOL_MM
        ),
    }

    payload = {
        "artifact": "D337_NEGATIVE_CONTROLS",
        "old_target_legacy_closed": legacy_payload,
        "old_target_exact_layer_closed": exact_layer,
        "grid_parity_closed": grid_parity,
        "open_jaw_scoping_crosscheck": scoping,
        "pass": bool(
            legacy_payload["pass"] and exact_layer["pass"] and grid_parity["pass"]
            and scoping["pass"]
        ),
    }
    return payload


def _decision_trace_row(inner: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    actual = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
    command = candidate["_command"][0].detach().cpu().numpy().astype(np.float64)
    return {
        "step": 0,
        "phase": "d337_decision",
        "actual_joint_rad_by_name": {
            name: float(actual[idx]) for idx, name in enumerate(inner._robot.joint_names)
        },
        "commanded_joint_rad_by_name": {
            name: float(command[idx]) for idx, name in enumerate(inner._robot.joint_names)
        },
        "radial_offset_mm": candidate["radial_offset_mm"],
        "tangent_offset_mm": candidate["tangent_offset_mm"],
        "min_raw_clearance_mm": candidate["min_raw_clearance_mm"],
        "exact_min_clearance_mm": candidate["exact_min_clearance_mm"],
        "frames": candidate["_frames"],
    }


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    decision = summary["search"].get("decision") or {}
    lines = [
        "# D337 open-jaw target gate",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "| Gate / metric | Result |",
        "|---|---:|",
        f"| Frozen contract | `{summary['frozen_contract']['pass']}` |",
        f"| Negative controls (4) | `{summary['negative_controls']['pass']}` |",
        f"| Open-jaw scan points | `{summary['search']['scan_count']}` |",
        f"| link5 q5-invariant max delta | `{summary['search']['link5_invariant_max_delta_mm']} mm` |",
        f"| Passing candidates | `{summary['search']['passing_count']}` |",
        f"| Selected r / t | `{decision.get('radial_offset_mm')} / {decision.get('tangent_offset_mm')} mm` |",
        f"| Selected exact min clearance | `{decision.get('exact_min_clearance_mm')} mm` |",
        f"| Physics executed | `{summary['physics']['executed']}` |",
        f"| Controlled physics steps | `{summary['physics']['controlled_steps_total']}` |",
        f"| Artifact contract | `{summary['artifact_contract']['pass']}` |",
        "",
        summary["classification"]["interpretation"] + ".",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    d332._runtime_versions()  # Pin gate before scene creation.
    source = json.loads(D334_SUMMARY.read_text(encoding="utf-8"))
    d335_summary = json.loads(D335_SUMMARY.read_text(encoding="utf-8"))
    d336_summary = json.loads(D336_SUMMARY.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pin = d334._pin_check()
    grid_keys_um = d336._parse_d335_grid_keys()
    d336_rescore = _load_d336_rescore()
    frozen_checks = {
        "d334_verdict": source["verdict"] == "D334_G0A_ACTUAL_TOOL_OVERLAP_SUPPORTED",
        "d335_verdict": d335_summary["verdict"]
        == "D335_G0A_TARGET_FAMILY_NO_FEASIBLE_CLEAR_STOP",
        "d336_verdict": d336_summary["verdict"]
        == "D336_G0A_FINITE_GRID_CAVEAT_DISCHARGED_NO_CLEAR_STOP",
        "d334_summary_sha256_pinned": d332._sha256(D334_SUMMARY) == PIN_D334_SUMMARY_SHA256,
        "d334_sha256_matches_d335_record": d335_summary["frozen_contract"]["source_sha256"]
        == PIN_D334_SUMMARY_SHA256,
        "d335_summary_sha256_pinned": d332._sha256(D335_SUMMARY) == PIN_D335_SUMMARY_SHA256,
        "d335_csv_sha256_pinned": d332._sha256(D335_CSV) == PIN_D335_CSV_SHA256,
        "d336_summary_sha256_pinned": d332._sha256(D336_SUMMARY) == PIN_D336_SUMMARY_SHA256,
        "d336_rescore_csv_sha256_pinned": d332._sha256(D336_RESCORE_CSV)
        == PIN_D336_RESCORE_CSV_SHA256,
        "seed_33201": int(args.seed) == 33201,
        "robot_usd_hash_matches_d334": d332._sha256(args.robot_usd_path)
        == source["frozen_contract"]["robot_usd_sha256"],
        "urdf_hash_matches_d334": d332._sha256(args.urdf_path)
        == source["frozen_contract"]["urdf_sha256"],
        "numpy_pin": bool(pin["numpy_pin_1_26_0"]),
        "psutil_pin": bool(pin["psutil_pin_5_9_8"]),
        "grid_key_count_2629": len(grid_keys_um) == EXPECTED_GRID_COUNT,
        "rescore_key_count_2629": len(d336_rescore) == EXPECTED_GRID_COUNT,
        "q5_open_within_urdf_limit": 0.0 <= Q5_OPEN_RAD <= 1.571,
        "method_constants_exact": (
            Q5_OPEN_RAD,
            SCOPING_OPEN_GRIPPER_MM,
            SCOPING_OPEN_TOL_MM,
            LINK5_INVARIANT_TOL_MM,
            TARGET_SETTLE_STEPS,
        )
        == (1.5413, 11.175, 0.5, 0.05, 200),
    }
    frozen_contract = {
        "checks": frozen_checks,
        "pass": all(frozen_checks.values()),
        "d334_summary_sha256": d332._sha256(D334_SUMMARY),
        "d335_summary": _rel(D335_SUMMARY),
        "d335_summary_sha256": d332._sha256(D335_SUMMARY),
        "d336_summary_sha256": d332._sha256(D336_SUMMARY),
        "d335_csv_sha256": d332._sha256(D335_CSV),
        "d336_rescore_csv_sha256": d332._sha256(D336_RESCORE_CSV),
        "robot_usd": _rel(args.robot_usd_path),
        "robot_usd_sha256": d332._sha256(args.robot_usd_path),
        "urdf": _rel(args.urdf_path),
        "urdf_sha256": d332._sha256(args.urdf_path),
        "environment": pin,
        "new_variable": {
            "name": "gripper_open_command",
            "q5_open_rad": Q5_OPEN_RAD,
            "q5_open_fraction_of_urdf_open_limit": Q5_OPEN_RAD / 1.571,
            "approx_real_opening_deg_under_d322_mapping": Q5_OPEN_RAD / 1.571 * 88.3,
            "basis": (
                "URDF q5=0 closed / 1.571 open (D322 mapping: real max 88.3deg <-> URDF "
                "1.571rad); commanded 1.5413rad is a deliberately sub-maximum opening "
                "(~98.1% of travel, ~86.6deg real), not the real maximum"
            ),
        },
    }
    _json_dump(args.out_dir / "d337_frozen_contract.json", frozen_contract)
    if not frozen_contract["pass"]:
        raise RuntimeError("D337 frozen contract failed before scene creation")

    inner = d333._make_runtime_env(args)
    controlled_physics_steps = 0
    snapshots: list[str] = []
    snapshot_paths: list[Path] = []
    marker_status: dict[str, Any] = {"ok": False, "error": "not attempted"}
    rrd_status: dict[str, Any] = {"ok": False, "error": "not attempted"}
    try:
        inner.reset(seed=int(args.seed))
        stage_contract = d333._stage_contract(inner)
        sensor_contract, filter_map = d333._sensor_contract(inner)
        shapes, raw_source_contract = d335._build_raw_shapes(inner, source)
        pre_scene_checks = {
            "stage_contract": bool(stage_contract["hard_contract_pass"]),
            "sensor_contract": bool(sensor_contract["hard_contract_pass"]),
            "raw_source_contract": bool(raw_source_contract["pass"]),
        }
        _json_dump(
            args.out_dir / "d337_prephysics_scene_contract.json",
            {
                "artifact": "D337_PREPHYSICS_SCENE_CONTRACT",
                "checks": pre_scene_checks,
                "pass": all(pre_scene_checks.values()),
                "stage_contract": stage_contract,
                "sensor_contract": sensor_contract,
                "raw_source_contract": raw_source_contract,
            },
        )
        if not all(pre_scene_checks.values()):
            raise RuntimeError("D337 stage/sensor/raw-source contract failed")

        cache = _Cache(inner, shapes)
        scan_counter_start = int(inner._sim_step_counter)

        negative = _negative_controls(inner, shapes, source, d336_summary, d336_rescore, cache)
        _json_dump(args.out_dir / "d337_negative_control.json", negative)

        scan_rows: list[dict[str, Any]] = []
        link5_invariant_max_delta = 0.0
        if negative["pass"]:
            for index, (r_um, t_um) in enumerate(grid_keys_um, start=1):
                row = cache.evaluate(r_um * 1000, t_um * 1000, Q5_OPEN_RAD, "open_scan")
                scan_rows.append(row)
                reference = d336_rescore[(r_um, t_um)]
                delta = abs(
                    float(row["link5_raw_signed_distance_mm"])
                    - float(reference["link5_raw_signed_distance_mm"])
                )
                link5_invariant_max_delta = max(link5_invariant_max_delta, delta)
                if index % 200 == 0 or index == len(grid_keys_um):
                    print(f"D337 open-jaw scan {index}/{len(grid_keys_um)}", flush=True)
            _write_rows_csv(args.out_dir / "d337_open_jaw_scan.csv", scan_rows)
        link5_invariant_pass = bool(
            scan_rows and link5_invariant_max_delta <= LINK5_INVARIANT_TOL_MM
        )

        scan_counter_end = int(inner._sim_step_counter)
        counter_unchanged = scan_counter_start == scan_counter_end

        passing = [row for row in scan_rows if row["pass"]]
        passing.sort(key=d336._selection_sort_key)
        selected = passing[0] if passing else None
        eligible = [
            row for row in scan_rows if d336._ranking_metric(row) > d336.METRIC_FAIL_SENTINEL
        ]
        eligible.sort(key=d336._basin_sort_key)
        best = selected if selected is not None else (
            eligible[0] if eligible else (scan_rows[0] if scan_rows else cache.rows[0])
        )

        # Decision re-materialization (registered gate).
        decision_candidate = _evaluate_candidate(
            inner,
            shapes,
            int(best["radial_offset_nm"]),
            int(best["tangent_offset_nm"]),
            Q5_OPEN_RAD,
            stage="decision_snapshot",
        )
        parity_checks = {
            "offsets_exact": (
                int(decision_candidate["radial_offset_nm"]) == int(best["radial_offset_nm"])
                and int(decision_candidate["tangent_offset_nm"]) == int(best["tangent_offset_nm"])
            ),
            "gripper_bvh_delta_le_0p05mm": abs(
                float(decision_candidate["gripper_link_raw_signed_distance_mm"])
                - float(best["gripper_link_raw_signed_distance_mm"])
            )
            <= PARITY_TOL_MM,
            "link5_bvh_delta_le_0p05mm": abs(
                float(decision_candidate["link5_raw_signed_distance_mm"])
                - float(best["link5_raw_signed_distance_mm"])
            )
            <= PARITY_TOL_MM,
            "exact_metric_delta_le_0p05mm": (
                decision_candidate["exact_min_clearance_mm"] is not None
                and best["exact_min_clearance_mm"] is not None
                and abs(
                    float(decision_candidate["exact_min_clearance_mm"])
                    - float(best["exact_min_clearance_mm"])
                )
                <= PARITY_TOL_MM
            ),
            "raw_states_exact": (
                decision_candidate["link5_raw_state"] == best["link5_raw_state"]
                and decision_candidate["gripper_link_raw_state"] == best["gripper_link_raw_state"]
            ),
        }
        decision_parity = {"checks": parity_checks, "pass": all(parity_checks.values())}

        search_payload = {
            "executed": bool(negative["pass"]),
            "q5_open_rad": Q5_OPEN_RAD,
            "domain": {
                "radial_mm": [0.0, 17.0],
                "tangent_mm": [9.0, 14.0],
                "anti_retreat": "17mm-r >= 0",
                "clear_gate_mm": d332.SIGNED_DISTANCE_BORDER_M * 1000.0,
            },
            "scan_count": len(scan_rows),
            "total_unique_count": len(cache.rows),
            "passing_count": len(passing),
            "legacy_alignment_count": sum(1 for row in scan_rows if row["legacy_alignment_pass"]),
            "exact_consistent_count": sum(1 for row in scan_rows if row["exact_consistent"]),
            "link5_invariant_max_delta_mm": link5_invariant_max_delta,
            "link5_invariant_pass": link5_invariant_pass,
            "selected": None if selected is None else _candidate_public(selected),
            "best": _candidate_public(best),
            "decision": _candidate_public(decision_candidate),
            "top_by_exact_clearance": [_candidate_public(row) for row in eligible[:20]],
            "top_passing_by_selection": [_candidate_public(row) for row in passing[:20]],
        }
        _json_dump(args.out_dir / "d337_search.json", {"artifact": "D337_SEARCH", **search_payload})

        gate_checks = {
            "negative_controls": bool(negative["pass"]),
            "probe_controlled_physics_steps_zero": controlled_physics_steps == 0,
            "sim_step_counter_unchanged_during_scan": counter_unchanged,
            "all_rows_counter_unchanged": all(
                row["sim_step_counter_unchanged"] for row in cache.rows
            ),
            "link5_q5_invariant": link5_invariant_pass or not scan_rows,
            "decision_candidate_revalidated": bool(
                selected is None
                or (
                    decision_candidate["pass"]
                    and decision_candidate["sim_step_counter_unchanged"]
                    and decision_parity["pass"]
                )
            ),
            "candidate_selected": selected is not None,
        }
        gate_payload = {
            "artifact": "D337_PREPHYSICS_GATE",
            "checks": gate_checks,
            "contract_pass": bool(
                gate_checks["negative_controls"]
                and gate_checks["probe_controlled_physics_steps_zero"]
                and gate_checks["sim_step_counter_unchanged_during_scan"]
                and gate_checks["all_rows_counter_unchanged"]
                and gate_checks["link5_q5_invariant"]
                and gate_checks["decision_candidate_revalidated"]
            ),
            "candidate_pass": gate_checks["candidate_selected"],
            "controlled_physics_steps": controlled_physics_steps,
            "sim_step_counter_start": scan_counter_start,
            "sim_step_counter_end": scan_counter_end,
            "decision_candidate_parity": decision_parity,
            "selected": search_payload["selected"],
            "decision": search_payload["decision"],
        }
        gate_payload["physics_licensed"] = bool(
            gate_payload["contract_pass"] and gate_payload["candidate_pass"]
        )
        _json_dump(args.out_dir / "d337_prephysics_gate.json", gate_payload)

        # Decision-time snapshot BEFORE any physics: the live stage still holds
        # the decision-candidate exact-written pose at this point.
        decision_png = args.out_dir / "d337_decision.png"
        snapshots.append(
            d335._write_raw_figure(
                decision_png,
                title=(
                    "D337 selected open-jaw raw-clear target"
                    if gate_payload["physics_licensed"]
                    else "D337 best open-jaw point (no full-pass candidate in executed set)"
                ),
                inner=inner,
                shapes=shapes,
                distance_set=decision_candidate["_distance_set"],
                canonical=decision_candidate["_canonical"],
            )
        )
        snapshot_paths.append(decision_png)

        baseline_rows: list[dict[str, Any]] = []
        target_rows: list[dict[str, Any]] = []
        raw_runtime_sets: list[dict[str, Any]] = []
        raw_trace_rows: list[dict[str, Any]] = []
        baseline_stats: dict[str, Any] | None = None
        target_stats: dict[str, Any] | None = None
        final_alignment: dict[str, Any] | None = None
        target_prestep_clear: bool | None = None

        if not gate_payload["contract_pass"]:
            verdict = VERDICT_CONTRACT_FAIL
            classification = {
                "verdict": verdict,
                "interpretation": (
                    "a negative-control/invariant/step-counter/parity contract failed; the "
                    "open-jaw scan conclusions are not licensed"
                ),
            }
        elif not gate_payload["candidate_pass"]:
            verdict = VERDICT_NO_FEASIBLE
            classification = {
                "verdict": verdict,
                "interpretation": (
                    "even with the gripper commanded open (q5=1.5413), no candidate in the "
                    "executed 2,629-key set cleared both audited raw tool shapes by >=+0.1mm "
                    "under the frozen alignment gates"
                ),
            }
        else:
            canonical = decision_candidate["_canonical"]
            q_home = np.radians(np.asarray(d332.HOME_DEG, dtype=np.float64))
            q_home[5] = Q5_OPEN_RAD
            home_target = d332._write_exact_state(inner, q_home, d332.OBJECT_CENTER_LOCAL_M)
            origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
            baseline_start_w = origin + d332.OBJECT_CENTER_LOCAL_M
            baseline_root_pos, baseline_root_quat = d333._root_pose(inner)
            for step in range(d332.BASELINE_PHYSICS_STEPS):
                d332._physics_step(inner)
                controlled_physics_steps += 1
                contact = d332._contact_state(inner._d333_contact_sensor, filter_map)
                baseline_rows.append(
                    d333._state_row(
                        inner,
                        phase="d337_sole_support_baseline",
                        step=step,
                        command_target=home_target,
                        canonical=canonical,
                        object_start_w=baseline_start_w,
                        root_start_pos_w=baseline_root_pos,
                        root_start_quat_wxyz=baseline_root_quat,
                        contact=contact,
                    )
                )
            d333._write_trace_csv(args.out_dir / "d337_baseline_trace.csv", baseline_rows)
            baseline_stats = d333._baseline_statistics(
                baseline_rows, stage_contract=stage_contract, sensor_contract=sensor_contract
            )
            if baseline_stats["hard_gate_pass"]:
                command = d332._write_exact_state(
                    inner,
                    np.asarray(canonical["commanded_joint_rad"], dtype=np.float64),
                    d332.OBJECT_CENTER_LOCAL_M,
                )
                object_start_w = origin + d332.OBJECT_CENTER_LOCAL_M
                target_root_pos, target_root_quat = d333._root_pose(inner)
                pre_target_raw = d335._raw_distance_matrix(inner, shapes, "target_prestep")
                raw_runtime_sets.append(pre_target_raw)
                target_prestep_clear = all(
                    row["clear_pass"] and row["consistency_hard_check_pass"]
                    for row in pre_target_raw["queries"]
                )
                if target_prestep_clear:
                    for step in range(TARGET_SETTLE_STEPS):
                        d332._physics_step(inner)
                        controlled_physics_steps += 1
                        contact = d332._contact_state(inner._d333_contact_sensor, filter_map)
                        target_rows.append(
                            d333._state_row(
                                inner,
                                phase="d337_target_static_settle",
                                step=step,
                                command_target=command,
                                canonical=canonical,
                                object_start_w=object_start_w,
                                root_start_pos_w=target_root_pos,
                                root_start_quat_wxyz=target_root_quat,
                                contact=contact,
                            )
                        )
                        raw_set = d335._raw_distance_matrix(inner, shapes, f"target_poststep_{step}")
                        raw_trace_rows.append(d335._raw_trace_flat(step, raw_set))
                        raw_runtime_sets.append(raw_set)
                    d333._write_trace_csv(
                        args.out_dir / "d337_target_static_trace.csv", target_rows
                    )
                    d335._write_dict_csv(
                        args.out_dir / "d337_target_raw_distance_trace.csv", raw_trace_rows
                    )
                    target_stats = d333._target_statistics(target_rows)
                    target_stats["max_robot_root_position_drift_m"] = max(
                        float(row["robot_root_position_drift_m"]) for row in target_rows
                    )
                    target_stats["max_robot_root_rotation_drift_rad"] = max(
                        float(row["robot_root_rotation_drift_rad"]) for row in target_rows
                    )
                    final_alignment, _final_frames = d335._runtime_alignment(
                        inner, canonical, object_start_w
                    )
            d335_verdict, static_classification = d335._classify_static(
                baseline_stats=baseline_stats,
                target_stats=target_stats,
                final_alignment=final_alignment,
                raw_sets=raw_runtime_sets,
            )
            verdict = D335_TO_D337_VERDICT[d335_verdict]
            classification = {"verdict": verdict, **static_classification}

        marker_frames = target_rows[-1]["frames"] if target_rows else decision_candidate["_frames"]
        marker_status = draw_frames(marker_frames, prim_path="/World/D337OpenJawFrames")
        if target_rows:
            final_png = args.out_dir / "d337_static_final.png"
            snapshots.append(
                d335._write_raw_figure(
                    final_png,
                    title="D337 open-jaw conditional static settle final",
                    inner=inner,
                    shapes=shapes,
                    distance_set=raw_runtime_sets[-1],
                    canonical=decision_candidate["_canonical"],
                )
            )
            snapshot_paths.append(final_png)

        rrd_path = args.out_dir / "d337_open_jaw_target_gate_trace.rrd"
        rrd_trace = target_rows if target_rows else [_decision_trace_row(inner, decision_candidate)]
        rrd_status = log_rerun(
            rrd_path,
            frames=marker_frames,
            joint_state={
                "label": "d337_open_jaw_target_gate",
                "object": "cylinder_d34_h90",
                "q5_open_rad": Q5_OPEN_RAD,
                "controlled_physics_steps_total": controlled_physics_steps,
                "physics_licensed": gate_payload["physics_licensed"],
            },
            joint_trace=rrd_trace,
            urdf_path=args.urdf_path,
            live_viewer=False,
            app_id="roarm_g0a_d337_open_jaw_target_gate",
        )
        if bool(rrd_status.get("ok")):
            rrd_status["nonzero_file"] = bool(rrd_path.is_file() and rrd_path.stat().st_size > 0)
        artifact_checks = {
            "snapshot_count_between_1_and_3": 1 <= len(snapshot_paths) <= 3,
            "snapshots_exist_and_nonzero": all(
                path.is_file() and path.stat().st_size > 0 for path in snapshot_paths
            ),
            "marker_status_ok": bool(marker_status.get("ok")),
            "rrd_status_ok": bool(rrd_status.get("ok")),
            "rrd_nonzero_file": bool(rrd_status.get("nonzero_file")),
        }
        artifact_contract = {"checks": artifact_checks, "pass": all(artifact_checks.values())}
        scientific_verdict = verdict
        final_verdict = scientific_verdict if artifact_contract["pass"] else VERDICT_VIZ_FAIL
        summary = {
            "verdict": final_verdict,
            "scientific_verdict_before_artifact_gate": scientific_verdict,
            "active_case": "G0a cylinder D34xH90 open-jaw target gate (q5 convention repair)",
            "new_variable": ["gripper_open_command"],
            "q5_open_rad": Q5_OPEN_RAD,
            "frozen_contract": frozen_contract,
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "raw_source_contract": raw_source_contract,
            "negative_controls": negative,
            "search": search_payload,
            "prephysics_gate": gate_payload,
            "classification": classification,
            "physics": {
                "executed": bool(gate_payload["physics_licensed"]),
                "controlled_steps_total": controlled_physics_steps,
                "baseline": baseline_stats,
                "target_static": target_stats,
                "final_alignment": final_alignment,
                "target_prestep_raw_clear": target_prestep_clear,
                "raw_distance_sets_count": len(raw_runtime_sets),
            },
            "outcome_guards": {
                "g0a_pass": False,
                "alignment_ladder_promoted": False,
                "ten_trial_run": False,
                "mesh_rewritten": False,
                "usd_or_urdf_changed": False,
                "collision_representation_changed": False,
                "domain_expanded_after_result": False,
                "stop_after_d337": True,
            },
            "visualization": {
                "snapshots": snapshots,
                "snapshot_count": len(snapshots),
                "marker_status": marker_status,
                "rrd_status": rrd_status,
            },
            "artifact_contract": artifact_contract,
            "artifacts": {
                name: (_rel(args.out_dir / filename) if (args.out_dir / filename).is_file() else None)
                for name, filename in (
                    ("frozen_contract", "d337_frozen_contract.json"),
                    ("scene_contract", "d337_prephysics_scene_contract.json"),
                    ("negative_control", "d337_negative_control.json"),
                    ("open_jaw_scan_csv", "d337_open_jaw_scan.csv"),
                    ("search", "d337_search.json"),
                    ("prephysics_gate", "d337_prephysics_gate.json"),
                    ("baseline_trace", "d337_baseline_trace.csv"),
                    ("target_trace", "d337_target_static_trace.csv"),
                    ("raw_distance_trace", "d337_target_raw_distance_trace.csv"),
                    ("decision_png", "d337_decision.png"),
                    ("static_final_png", "d337_static_final.png"),
                    ("rrd", "d337_open_jaw_target_gate_trace.rrd"),
                )
            }
            | {
                "summary_json": _rel(args.out_dir / "g0a_d337_open_jaw_target_gate_summary.json"),
                "summary_markdown": _rel(args.out_dir / "g0a_d337_open_jaw_target_gate_summary.md"),
            },
            "non_goals_respected": [
                "no 10-trial/approach/waypoint run; no close/grasp/lift",
                "no wrist/tool-orientation variable",
                "no URDF/USD/mesh edit or cook compensation",
                "no domain expansion or physics-driven retry",
                "no G0b/RL/PPO/randomization/VLA/RoArm/B200/cube",
            ],
        }
        _json_dump(args.out_dir / "g0a_d337_open_jaw_target_gate_summary.json", summary)
        _write_summary_markdown(
            args.out_dir / "g0a_d337_open_jaw_target_gate_summary.md", summary
        )
        return summary
    finally:
        inner.close()


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=d333.DEFAULT_ROBOT_USD)
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
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    del simulation_app  # App lifetime is owned by the launcher.
    try:
        try:
            summary = _run(args)
            decision = summary["search"].get("decision") or {}
            print(
                f"{summary['verdict']}: scan={summary['search']['scan_count']} "
                f"passes={summary['search']['passing_count']} "
                f"decision_r/t={decision.get('radial_offset_mm')}/{decision.get('tangent_offset_mm')}mm "
                f"exact_clear={decision.get('exact_min_clearance_mm')}mm "
                f"physics={summary['physics']['executed']}",
                flush=True,
            )
            return 0 if bool(summary["artifact_contract"]["pass"]) else 1
        except Exception:
            traceback.print_exc()
            args.out_dir.mkdir(parents=True, exist_ok=True)
            abort_payload = {
                "verdict": VERDICT_CONTRACT_FAIL,
                "interpretation": "the current D337 invocation aborted before clean completion",
                "error": traceback.format_exc(),
            }
            _json_dump(args.out_dir / "d337_abort.json", abort_payload)
            fail_path = args.out_dir / "g0a_d337_open_jaw_target_gate_summary.json"
            if not fail_path.is_file():
                _json_dump(fail_path, abort_payload)
            return 1
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
