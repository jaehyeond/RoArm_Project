#!/usr/bin/env python3
"""D336 finite-grid caveat discriminator for cylinder grasp G0a.

Zero new physical variables. Within the exact D335 frozen family and domain,
re-score the complete 2,629-point D335 grid with an exact contact-level EPA
penetration metric (the BVH ranking scalar is never a judgment quantity),
then run pre-registered Nelder-Mead continuous refinement from the top exact
basins and a micro-grid around the best point.  A candidate is feasible only
if both audited raw tool shapes are clear ``>=+0.1mm`` while every frozen G0a
alignment gate passes.  D336 never calls ``sim.step``; a found candidate is
registered for a separate later physics gate only.
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


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d336"
D334_SUMMARY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d334/g0a_d334_live_collision_audit_summary.json"
)
D335_SUMMARY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d335/g0a_d335_target_family_repair_summary.json"
)
D335_CSV = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d335/d335_candidate_scan.csv"

PIN_D334_SUMMARY_SHA256 = "2ff44744df99c7a99d168cdd62a4f9186a5bbad6d673205282abb62b71097b26"
PIN_D335_SUMMARY_SHA256 = "7ca98f31d6fc23ea0942d4863d2d7dbdce561293e181d5b1bd7a451dd0064d0e"
PIN_D335_CSV_SHA256 = "f7daa545c190416f1117c275c4e8b015bce721507c04500254adc074d25d5f79"

EXPECTED_GRID_COUNT = 2629
EPA_MAX_CONTACTS = 64
NM_SEED_COUNT = 5
NM_MAXFEV = 300
NM_XATOL_MM = 1.0e-4
NM_FATOL = 1.0e-5
MICRO_RADIUS_NM = 50_000
MICRO_STEP_NM = 5_000
RADIAL_MIN_NM = 0
RADIAL_MAX_NM = 17_000_000
TANGENT_MIN_NM = 9_000_000
TANGENT_MAX_NM = 14_000_000
OLD_RADIAL_NM = 7_000_000
OLD_TANGENT_NM = 11_000_000
PARITY_TOL_MM = 0.05
GRID_PARITY_KEYS_UM = ((14_600, 13_900), (0, 9_000))
METRIC_FAIL_SENTINEL = -1.0e9

VERDICT_CONTRACT_FAIL = "D336_G0A_PREPHYSICS_CONTRACT_FAIL_STOP"
VERDICT_DISCHARGED = "D336_G0A_FINITE_GRID_CAVEAT_DISCHARGED_NO_CLEAR_STOP"
VERDICT_FOUND = "D336_G0A_RAW_CLEAR_CANDIDATE_REGISTERED_STOP"
VERDICT_VIZ_FAIL = "D336_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP"


def _rel(path: Path) -> str:
    return d332._rel(path)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    d332._json_dump(path, payload)


def _epa_exact_contacts(
    hppfcl: Any, geometry: Any, transform: Any, cylinder: Any, cylinder_tf: Any
) -> dict[str, Any]:
    request = hppfcl.CollisionRequest()
    request.enable_contact = True
    request.num_max_contacts = EPA_MAX_CONTACTS
    result = hppfcl.CollisionResult()
    hppfcl.collide(geometry, transform, cylinder, cylinder_tf, request, result)
    depths_mm = [
        abs(float(result.getContact(i).penetration_depth)) * 1000.0
        for i in range(result.numContacts())
    ]
    return {
        "is_collision": bool(result.isCollision()),
        "num_contacts": int(result.numContacts()),
        "cap_saturated": bool(int(result.numContacts()) >= EPA_MAX_CONTACTS),
        "max_abs_depth_mm": max(depths_mm) if depths_mm else None,
        "depths_mm_top8_desc": sorted(depths_mm, reverse=True)[:8],
    }


def _exact_raw_metrics(inner: Any, shapes: list[dict[str, Any]], pose_label: str) -> dict[str, Any]:
    import hppfcl

    base = d335._raw_distance_matrix(inner, shapes, pose_label)
    cylinder = hppfcl.Cylinder(d332.CYLINDER_RADIUS_M, d332.CYLINDER_HEIGHT_M)
    obj_pos = np.asarray(base["object_pos_w_m"], dtype=np.float64)
    obj_quat = np.asarray(base["object_quat_wxyz"], dtype=np.float64)
    cyl_tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(obj_quat), obj_pos)
    shape_by_body = {shape["body"]: shape for shape in shapes}
    for row in base["queries"]:
        pose = base["body_poses_w"][row["body"]]
        tf = hppfcl.Transform3f(
            d332._quat_wxyz_to_rot(np.asarray(pose["quat_wxyz"], dtype=np.float64)),
            np.asarray(pose["pos_m"], dtype=np.float64),
        )
        epa = _epa_exact_contacts(
            hppfcl, shape_by_body[row["body"]]["_geom_raw"], tf, cylinder, cyl_tf
        )
        row["epa_contact_count"] = epa["num_contacts"]
        row["epa_cap_saturated"] = epa["cap_saturated"]
        row["epa_max_abs_depth_mm"] = epa["max_abs_depth_mm"]
        row["epa_depths_mm_top8_desc"] = epa["depths_mm_top8_desc"]
        collision_bool_consistent = bool(epa["is_collision"] == bool(row["is_collision"]))
        if bool(row["is_collision"]):
            row["exact_consistent"] = bool(collision_bool_consistent and epa["num_contacts"] > 0)
            row["exact_signed_distance_mm"] = (
                -float(epa["max_abs_depth_mm"]) if epa["num_contacts"] > 0 else None
            )
        else:
            row["exact_consistent"] = collision_bool_consistent
            # Not colliding: the BVH distance IS the exact separation distance.
            row["exact_signed_distance_mm"] = float(row["signed_distance_mm"])
    return base


def _evaluate_candidate_exact(
    inner: Any,
    shapes: list[dict[str, Any]],
    radial_nm: int,
    tangent_nm: int,
    *,
    stage: str,
) -> dict[str, Any]:
    radial_um = radial_nm / 1000.0
    tangent_um = tangent_nm / 1000.0
    canonical = d335._canonical_for_offsets(radial_um, tangent_um)
    counter_before = int(inner._sim_step_counter)
    command = d332._write_exact_state(
        inner,
        np.asarray(canonical["commanded_joint_rad"], dtype=np.float64),
        d332.OBJECT_CENTER_LOCAL_M,
    )
    counter_after = int(inner._sim_step_counter)
    distances = _exact_raw_metrics(inner, shapes, f"d336_{stage}")
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
    # Judgment rule identical to D335: state 'clear' (exact collide + exact
    # separation) with the existing consistency hard check, both bodies.
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


FLAT_KEYS = (
    "stage",
    "radial_offset_nm",
    "tangent_offset_nm",
    "radial_offset_mm",
    "tangent_offset_mm",
    "radial_tip_past_near_face_mm",
    "shift_from_old_mm",
    "ik_converged",
    "commanded_tcp_error_mm",
    "jaw_tangent_error_deg",
    "fixed_jaw_face_gap_mm",
    "fixed_jaw_penetration_mm",
    "contact_point_below_top_mm",
    "link5_raw_signed_distance_mm",
    "link5_raw_state",
    "link5_exact_signed_distance_mm",
    "link5_epa_contact_count",
    "link5_epa_cap_saturated",
    "gripper_link_raw_signed_distance_mm",
    "gripper_link_raw_state",
    "gripper_link_exact_signed_distance_mm",
    "gripper_link_epa_contact_count",
    "gripper_link_epa_cap_saturated",
    "min_raw_clearance_mm",
    "exact_min_clearance_mm",
    "exact_consistent",
    "legacy_alignment_pass",
    "raw_tool_clear_pass",
    "pass",
    "sim_step_counter_unchanged",
)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [{key: row.get(key) for key in FLAT_KEYS} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FLAT_KEYS))
        writer.writeheader()
        writer.writerows(flat)


def _ranking_metric(row: dict[str, Any]) -> float:
    value = row.get("exact_min_clearance_mm")
    if value is None or not row.get("exact_consistent") or not row.get("sim_step_counter_unchanged"):
        return METRIC_FAIL_SENTINEL
    return float(value)


def _basin_sort_key(row: dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        -_ranking_metric(row),
        float(row["shift_from_old_mm"]),
        int(row["radial_offset_nm"]),
        int(row["tangent_offset_nm"]),
    )


def _selection_sort_key(row: dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        float(row["shift_from_old_mm"]),
        -_ranking_metric(row),
        int(row["radial_offset_nm"]),
        int(row["tangent_offset_nm"]),
    )


def _parse_d335_grid_keys() -> list[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    with D335_CSV.open("r", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            r_um = int(round(float(record["radial_offset_mm"]) * 1000.0))
            t_um = int(round(float(record["tangent_offset_mm"]) * 1000.0))
            keys.add((r_um, t_um))
    return sorted(keys)


def _load_d335_rows_by_key() -> dict[tuple[int, int], dict[str, str]]:
    rows: dict[tuple[int, int], dict[str, str]] = {}
    with D335_CSV.open("r", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            key = (
                int(round(float(record["radial_offset_mm"]) * 1000.0)),
                int(round(float(record["tangent_offset_mm"]) * 1000.0)),
            )
            rows[key] = record
    return rows


class _CandidateCache:
    def __init__(self, inner: Any, shapes: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._shapes = shapes
        self.by_key: dict[tuple[int, int], dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []

    def evaluate(self, radial_nm: int, tangent_nm: int, stage: str) -> dict[str, Any]:
        key = (int(radial_nm), int(tangent_nm))
        cached = self.by_key.get(key)
        if cached is not None:
            return cached
        row = _evaluate_candidate_exact(self._inner, self._shapes, key[0], key[1], stage=stage)
        self.by_key[key] = row
        self.rows.append(row)
        return row


def _negative_controls(
    inner: Any,
    shapes: list[dict[str, Any]],
    d334_source: dict[str, Any],
    cache: _CandidateCache,
    d335_rows: dict[tuple[int, int], dict[str, str]],
) -> dict[str, Any]:
    # Control 1: unchanged D335 evaluator at the old target, bit-parity vs D334.
    legacy_payload, _legacy_candidate = d335._negative_control(inner, shapes, d334_source)

    # Control 2: exact-EPA layer at the old target vs the D334 recorded raw
    # gripper EPA depth (pose_a_prestep).
    expected_epa_mm = None
    for distance_set in d334_source["distance_sets"]:
        if distance_set["pose"] != "pose_a_prestep":
            continue
        for row in distance_set["queries"]:
            if row["body"] == "gripper_link" and row["representation"] == "raw":
                depth = row.get("penetration_depth_m")
                expected_epa_mm = None if depth is None else abs(float(depth)) * 1000.0
    exact_old = cache.evaluate(OLD_RADIAL_NM, OLD_TANGENT_NM, "old_target_exact_control")
    gripper_query = {
        query["body"]: query for query in exact_old["_distance_set"]["queries"]
    }["gripper_link"]
    exact_layer = {
        "expected_d334_epa_depth_mm": expected_epa_mm,
        "observed_epa_max_abs_depth_mm": gripper_query["epa_max_abs_depth_mm"],
        "observed_epa_contact_count": gripper_query["epa_contact_count"],
        "observed_is_collision": bool(gripper_query["is_collision"]),
        "tolerance_mm": PARITY_TOL_MM,
    }
    exact_layer["pass"] = bool(
        expected_epa_mm is not None
        and exact_layer["observed_is_collision"]
        and int(exact_layer["observed_epa_contact_count"]) >= 1
        and exact_layer["observed_epa_max_abs_depth_mm"] is not None
        and float(exact_layer["observed_epa_max_abs_depth_mm"]) >= expected_epa_mm - PARITY_TOL_MM
    )

    # Control 3: grid parity on two pinned D335 rows (BVH scalar + states).
    grid_parity: dict[str, Any] = {"rows": [], "tolerance_mm": PARITY_TOL_MM}
    for r_um, t_um in GRID_PARITY_KEYS_UM:
        reference = d335_rows[(r_um, t_um)]
        observed = cache.evaluate(r_um * 1000, t_um * 1000, "grid_parity_control")
        entry = {
            "radial_offset_um": r_um,
            "tangent_offset_um": t_um,
            "expected_gripper_bvh_mm": float(reference["gripper_link_raw_signed_distance_mm"]),
            "observed_gripper_bvh_mm": float(observed["gripper_link_raw_signed_distance_mm"]),
            "expected_link5_bvh_mm": float(reference["link5_raw_signed_distance_mm"]),
            "observed_link5_bvh_mm": float(observed["link5_raw_signed_distance_mm"]),
            "expected_states": (reference["link5_raw_state"], reference["gripper_link_raw_state"]),
            "observed_states": (observed["link5_raw_state"], observed["gripper_link_raw_state"]),
        }
        entry["pass"] = bool(
            abs(entry["observed_gripper_bvh_mm"] - entry["expected_gripper_bvh_mm"]) <= PARITY_TOL_MM
            and abs(entry["observed_link5_bvh_mm"] - entry["expected_link5_bvh_mm"]) <= PARITY_TOL_MM
            and tuple(entry["expected_states"]) == tuple(entry["observed_states"])
        )
        grid_parity["rows"].append(entry)
    grid_parity["pass"] = all(entry["pass"] for entry in grid_parity["rows"])

    payload = {
        "artifact": "D336_NEGATIVE_CONTROLS",
        "old_target_legacy": legacy_payload,
        "old_target_exact_layer": exact_layer,
        "grid_parity": grid_parity,
        "pass": bool(legacy_payload["pass"] and exact_layer["pass"] and grid_parity["pass"]),
    }
    return payload


def _run_nelder_mead(
    cache: _CandidateCache, seeds: list[dict[str, Any]]
) -> dict[str, Any]:
    from scipy.optimize import minimize

    runs = []
    for index, seed in enumerate(seeds):
        stage_label = f"nm_seed_{index}"
        evaluations = {"count": 0}

        def objective(x: np.ndarray, _stage: str = stage_label, _counter: dict = evaluations) -> float:
            r_mm = min(17.0, max(0.0, float(x[0])))
            t_mm = min(14.0, max(9.0, float(x[1])))
            r_nm = int(round(r_mm * 1.0e6))
            t_nm = int(round(t_mm * 1.0e6))
            _counter["count"] += 1
            row = cache.evaluate(r_nm, t_nm, _stage)
            metric = _ranking_metric(row)
            if metric <= METRIC_FAIL_SENTINEL:
                return 1.0e9
            return -metric

        x0 = np.asarray([seed["radial_offset_mm"], seed["tangent_offset_mm"]], dtype=np.float64)
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            bounds=[(0.0, 17.0), (9.0, 14.0)],
            options={
                "xatol": NM_XATOL_MM,
                "fatol": NM_FATOL,
                "maxfev": NM_MAXFEV,
                "adaptive": False,
            },
        )
        runs.append(
            {
                "seed_index": index,
                "seed_r_mm": float(seed["radial_offset_mm"]),
                "seed_t_mm": float(seed["tangent_offset_mm"]),
                "seed_exact_min_clearance_mm": seed.get("exact_min_clearance_mm"),
                "objective_calls": evaluations["count"],
                "nfev": int(result.nfev),
                "nit": int(result.nit),
                "converged": bool(result.success),
                "final_x_mm": [float(result.x[0]), float(result.x[1])],
                "final_objective": float(result.fun),
                "message": str(result.message),
            }
        )
        print(
            f"D336 NM seed {index}: start=({x0[0]:.3f},{x0[1]:.3f})mm "
            f"end=({result.x[0]:.4f},{result.x[1]:.4f})mm nfev={result.nfev}",
            flush=True,
        )
    return {"runs": runs}


def _write_clearance_map(
    path: Path,
    *,
    stage_a_rows: list[dict[str, Any]],
    other_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    selected: dict[str, Any] | None,
    any_clear: bool,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    valid = [row for row in stage_a_rows if _ranking_metric(row) > METRIC_FAIL_SENTINEL]
    grid_r = [row["radial_offset_mm"] for row in valid]
    grid_t = [row["tangent_offset_mm"] for row in valid]
    grid_c = [_ranking_metric(row) for row in valid]
    scatter = ax.scatter(
        grid_r, grid_t, c=grid_c, s=7.0, cmap="viridis",
        vmin=(min(grid_c) if grid_c else -1.0), vmax=0.1, marker="s",
    )
    fig.colorbar(scatter, ax=ax, label="exact min raw clearance [mm] (EPA-based)")
    if other_rows:
        ax.scatter(
            [row["radial_offset_mm"] for row in other_rows],
            [row["tangent_offset_mm"] for row in other_rows],
            s=4.0, color="tab:red", alpha=0.5, marker=".",
            label=f"continuous/micro evals (n={len(other_rows)})",
        )
    ax.scatter(
        [decision["radial_offset_mm"]], [decision["tangent_offset_mm"]],
        s=140, color="black", marker="*",
        label=(
            f"decision point ({decision['radial_offset_mm']:.4f},"
            f"{decision['tangent_offset_mm']:.4f})mm, "
            f"exact={decision.get('exact_min_clearance_mm')}"
        ),
    )
    if selected is not None:
        ax.scatter(
            [selected["radial_offset_mm"]], [selected["tangent_offset_mm"]],
            s=90, facecolors="none", edgecolors="magenta", linewidths=2.0,
            label="selected raw-clear candidate",
        )
    ax.scatter([7.0], [11.0], s=70, color="orange", marker="X", label="old D325 target (7,11)mm")
    ax.set_xlim(-0.4, 17.4)
    ax.set_ylim(8.9, 14.1)
    ax.set_xlabel("radial offset r [mm]")
    ax.set_ylabel("tangent offset t [mm]")
    # Title claims must stay within the executed evaluation set (no continuum
    # overclaim) and must not assert coverage when the search never ran.
    if not stage_a_rows:
        suffix = " (search not executed: contract fail)"
    elif selected is not None:
        suffix = ""
    elif any_clear:
        suffix = " (raw-clear point(s) found but none passed the frozen alignment gates)"
    else:
        suffix = " (no evaluated point >= +0.1mm)"
    ax.set_title("D336 exact raw clearance over the frozen r/t domain" + suffix)
    ax.legend(loc="lower left", fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return _rel(path)


def _decision_trace_row(inner: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    actual = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
    command = candidate["_command"][0].detach().cpu().numpy().astype(np.float64)
    return {
        "step": 0,
        "phase": "d336_decision",
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
        "# D336 finite-grid caveat discriminator",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "| Gate / metric | Result |",
        "|---|---:|",
        f"| Frozen contract | `{summary['frozen_contract']['pass']}` |",
        f"| Negative controls (3) | `{summary['negative_controls']['pass']}` |",
        f"| Stage A rescore points | `{summary['search']['stage_a_count']}` |",
        f"| Stage B continuous evals | `{summary['search']['stage_b_count']}` |",
        f"| Stage C micro evals | `{summary['search']['stage_c_count']}` |",
        f"| Total unique evaluations | `{summary['search']['total_unique_count']}` |",
        f"| Raw-clear passing candidates | `{summary['search']['passing_count']}` |",
        f"| Decision r / t | `{decision.get('radial_offset_mm')} / {decision.get('tangent_offset_mm')} mm` |",
        f"| Decision exact min clearance | `{decision.get('exact_min_clearance_mm')} mm` |",
        f"| Controlled physics steps | `{summary['controlled_physics_steps_total']}` |",
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pin = d334._pin_check()
    grid_keys_um = _parse_d335_grid_keys()
    d335_rows = _load_d335_rows_by_key()
    frozen_checks = {
        "d334_verdict": source["verdict"] == "D334_G0A_ACTUAL_TOOL_OVERLAP_SUPPORTED",
        "d335_verdict": d335_summary["verdict"]
        == "D335_G0A_TARGET_FAMILY_NO_FEASIBLE_CLEAR_STOP",
        "d334_summary_sha256_pinned": d332._sha256(D334_SUMMARY) == PIN_D334_SUMMARY_SHA256,
        "d334_sha256_matches_d335_record": d335_summary["frozen_contract"]["source_sha256"]
        == PIN_D334_SUMMARY_SHA256,
        "d335_summary_sha256_pinned": d332._sha256(D335_SUMMARY) == PIN_D335_SUMMARY_SHA256,
        "d335_csv_sha256_pinned": d332._sha256(D335_CSV) == PIN_D335_CSV_SHA256,
        "seed_33201": int(args.seed) == 33201,
        "robot_usd_hash_matches_d334": d332._sha256(args.robot_usd_path)
        == source["frozen_contract"]["robot_usd_sha256"],
        "urdf_hash_matches_d334": d332._sha256(args.urdf_path)
        == source["frozen_contract"]["urdf_sha256"],
        "numpy_pin": bool(pin["numpy_pin_1_26_0"]),
        "psutil_pin": bool(pin["psutil_pin_5_9_8"]),
        "grid_key_count_2629": len(grid_keys_um) == EXPECTED_GRID_COUNT,
        "method_constants_exact": (
            EPA_MAX_CONTACTS,
            NM_SEED_COUNT,
            NM_MAXFEV,
            NM_XATOL_MM,
            NM_FATOL,
            MICRO_RADIUS_NM,
            MICRO_STEP_NM,
        )
        == (64, 5, 300, 1.0e-4, 1.0e-5, 50_000, 5_000),
    }
    frozen_contract = {
        "checks": frozen_checks,
        "pass": all(frozen_checks.values()),
        "d334_summary": _rel(D334_SUMMARY),
        "d334_summary_sha256": d332._sha256(D334_SUMMARY),
        "d335_summary": _rel(D335_SUMMARY),
        "d335_summary_sha256": d332._sha256(D335_SUMMARY),
        "d335_csv": _rel(D335_CSV),
        "d335_csv_sha256": d332._sha256(D335_CSV),
        "robot_usd": _rel(args.robot_usd_path),
        "robot_usd_sha256": d332._sha256(args.robot_usd_path),
        "urdf": _rel(args.urdf_path),
        "urdf_sha256": d332._sha256(args.urdf_path),
        "environment": pin,
        "method": {
            "epa_max_contacts": EPA_MAX_CONTACTS,
            "nm_seed_count": NM_SEED_COUNT,
            "nm_maxfev": NM_MAXFEV,
            "nm_xatol_mm": NM_XATOL_MM,
            "nm_fatol": NM_FATOL,
            "micro_radius_mm": MICRO_RADIUS_NM / 1.0e6,
            "micro_step_mm": MICRO_STEP_NM / 1.0e6,
            "clear_gate_mm": d332.SIGNED_DISTANCE_BORDER_M * 1000.0,
        },
    }
    _json_dump(args.out_dir / "d336_frozen_contract.json", frozen_contract)
    if not frozen_contract["pass"]:
        raise RuntimeError("D336 frozen contract failed before scene creation")

    inner = d333._make_runtime_env(args)
    controlled_physics_steps = 0
    snapshots: list[str] = []
    snapshot_paths: list[Path] = []
    marker_status: dict[str, Any] = {"ok": False, "error": "not attempted"}
    rrd_status: dict[str, Any] = {"ok": False, "error": "not attempted"}
    try:
        inner.reset(seed=int(args.seed))
        stage_contract = d333._stage_contract(inner)
        sensor_contract, _filter_map = d333._sensor_contract(inner)
        shapes, raw_source_contract = d335._build_raw_shapes(inner, source)
        pre_scene_checks = {
            "stage_contract": bool(stage_contract["hard_contract_pass"]),
            "sensor_contract": bool(sensor_contract["hard_contract_pass"]),
            "raw_source_contract": bool(raw_source_contract["pass"]),
        }
        _json_dump(
            args.out_dir / "d336_prephysics_scene_contract.json",
            {
                "artifact": "D336_PREPHYSICS_SCENE_CONTRACT",
                "checks": pre_scene_checks,
                "pass": all(pre_scene_checks.values()),
                "stage_contract": stage_contract,
                "sensor_contract": sensor_contract,
                "raw_source_contract": raw_source_contract,
            },
        )
        if not all(pre_scene_checks.values()):
            raise RuntimeError("D336 stage/sensor/raw-source contract failed")

        cache = _CandidateCache(inner, shapes)
        scan_counter_start = int(inner._sim_step_counter)

        negative = _negative_controls(inner, shapes, source, cache, d335_rows)
        _json_dump(args.out_dir / "d336_negative_control.json", negative)

        stage_a_rows: list[dict[str, Any]] = []
        nm_report: dict[str, Any] = {"runs": []}
        seeds_public: list[dict[str, Any]] = []
        micro_center_public: dict[str, Any] | None = None
        if negative["pass"]:
            # Stage A: exact re-scoring of the complete D335 grid.
            for index, (r_um, t_um) in enumerate(grid_keys_um, start=1):
                row = cache.evaluate(r_um * 1000, t_um * 1000, "rescore")
                stage_a_rows.append(row)
                if index % 200 == 0 or index == len(grid_keys_um):
                    print(f"D336 stage A rescore {index}/{len(grid_keys_um)}", flush=True)
            _write_rows_csv(args.out_dir / "d336_exact_rescore.csv", stage_a_rows)

            # Stage B: Nelder-Mead from the top exact basins (+ D335 best point).
            ranked = sorted(stage_a_rows, key=_basin_sort_key)
            seeds = ranked[:NM_SEED_COUNT]
            d335_best_key = (14_600_000, 13_900_000)
            if all(
                (row["radial_offset_nm"], row["tangent_offset_nm"]) != d335_best_key
                for row in seeds
            ):
                seeds.append(cache.by_key[d335_best_key])
            seeds_public = [
                {
                    "radial_offset_mm": row["radial_offset_mm"],
                    "tangent_offset_mm": row["tangent_offset_mm"],
                    "exact_min_clearance_mm": row["exact_min_clearance_mm"],
                    "min_raw_clearance_mm": row["min_raw_clearance_mm"],
                }
                for row in seeds
            ]
            nm_report = _run_nelder_mead(cache, seeds)

            # Stage C: micro-grid around the best point so far.
            best_so_far = sorted(cache.rows, key=_basin_sort_key)[0]
            micro_center_public = {
                "radial_offset_mm": best_so_far["radial_offset_mm"],
                "tangent_offset_mm": best_so_far["tangent_offset_mm"],
                "exact_min_clearance_mm": best_so_far["exact_min_clearance_mm"],
            }
            r0_nm = int(best_so_far["radial_offset_nm"])
            t0_nm = int(best_so_far["tangent_offset_nm"])
            micro_keys = set()
            for dr in range(-MICRO_RADIUS_NM, MICRO_RADIUS_NM + 1, MICRO_STEP_NM):
                for dt in range(-MICRO_RADIUS_NM, MICRO_RADIUS_NM + 1, MICRO_STEP_NM):
                    r_nm = min(RADIAL_MAX_NM, max(RADIAL_MIN_NM, r0_nm + dr))
                    t_nm = min(TANGENT_MAX_NM, max(TANGENT_MIN_NM, t0_nm + dt))
                    micro_keys.add((r_nm, t_nm))
            for index, (r_nm, t_nm) in enumerate(sorted(micro_keys), start=1):
                cache.evaluate(r_nm, t_nm, "micro")
                if index % 100 == 0 or index == len(micro_keys):
                    print(f"D336 stage C micro {index}/{len(micro_keys)}", flush=True)

        scan_counter_end = int(inner._sim_step_counter)
        counter_unchanged = scan_counter_start == scan_counter_end

        continuous_rows = [row for row in cache.rows if row["stage"].startswith(("nm_seed", "micro"))]
        if continuous_rows:
            _write_rows_csv(args.out_dir / "d336_continuous_scan.csv", continuous_rows)

        passing = [row for row in cache.rows if row["pass"]]
        passing.sort(key=_selection_sort_key)
        selected = passing[0] if passing else None
        eligible = [row for row in cache.rows if _ranking_metric(row) > METRIC_FAIL_SENTINEL]
        eligible.sort(key=_basin_sort_key)
        best = selected if selected is not None else (eligible[0] if eligible else cache.rows[0])

        search_payload = {
            "executed": bool(negative["pass"]),
            "domain": {
                "radial_mm": [0.0, 17.0],
                "tangent_mm": [9.0, 14.0],
                "anti_retreat": "17mm-r >= 0",
                "clear_gate_mm": d332.SIGNED_DISTANCE_BORDER_M * 1000.0,
            },
            "stage_a_count": len(stage_a_rows),
            "stage_b_count": sum(1 for row in cache.rows if row["stage"].startswith("nm_seed")),
            "stage_c_count": sum(1 for row in cache.rows if row["stage"] == "micro"),
            "total_unique_count": len(cache.rows),
            "passing_count": len(passing),
            "legacy_alignment_count": sum(1 for row in cache.rows if row["legacy_alignment_pass"]),
            "exact_consistent_count": sum(1 for row in cache.rows if row["exact_consistent"]),
            "epa_cap_saturated_count": sum(
                1
                for row in cache.rows
                if row["link5_epa_cap_saturated"] or row["gripper_link_epa_cap_saturated"]
            ),
            "nm_seeds": seeds_public,
            "nm_runs": nm_report["runs"],
            "micro_center": micro_center_public,
            "selected": None if selected is None else _candidate_public(selected),
            "best": _candidate_public(best),
            "top_by_exact_clearance": [_candidate_public(row) for row in eligible[:20]],
        }

        # Re-materialize the decision candidate after the last scan row.
        decision_reference = selected if selected is not None else best
        decision_candidate = _evaluate_candidate_exact(
            inner,
            shapes,
            int(decision_reference["radial_offset_nm"]),
            int(decision_reference["tangent_offset_nm"]),
            stage="decision_snapshot",
        )
        parity_checks = {
            "offsets_exact": (
                int(decision_candidate["radial_offset_nm"])
                == int(decision_reference["radial_offset_nm"])
                and int(decision_candidate["tangent_offset_nm"])
                == int(decision_reference["tangent_offset_nm"])
            ),
            "gripper_bvh_delta_le_0p05mm": abs(
                float(decision_candidate["gripper_link_raw_signed_distance_mm"])
                - float(decision_reference["gripper_link_raw_signed_distance_mm"])
            )
            <= PARITY_TOL_MM,
            "link5_bvh_delta_le_0p05mm": abs(
                float(decision_candidate["link5_raw_signed_distance_mm"])
                - float(decision_reference["link5_raw_signed_distance_mm"])
            )
            <= PARITY_TOL_MM,
            "exact_metric_delta_le_0p05mm": (
                decision_candidate["exact_min_clearance_mm"] is not None
                and decision_reference["exact_min_clearance_mm"] is not None
                and abs(
                    float(decision_candidate["exact_min_clearance_mm"])
                    - float(decision_reference["exact_min_clearance_mm"])
                )
                <= PARITY_TOL_MM
            ),
            "raw_states_exact": (
                decision_candidate["link5_raw_state"] == decision_reference["link5_raw_state"]
                and decision_candidate["gripper_link_raw_state"]
                == decision_reference["gripper_link_raw_state"]
            ),
        }
        decision_parity = {"checks": parity_checks, "pass": all(parity_checks.values())}

        search_payload["decision"] = _candidate_public(decision_candidate)
        _json_dump(
            args.out_dir / "d336_search.json",
            {"artifact": "D336_SEARCH", **search_payload},
        )

        gate_checks = {
            "negative_controls": bool(negative["pass"]),
            "probe_controlled_physics_steps_zero": controlled_physics_steps == 0,
            "sim_step_counter_unchanged_during_scan": counter_unchanged,
            "all_rows_counter_unchanged": all(
                row["sim_step_counter_unchanged"] for row in cache.rows
            ),
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
            "artifact": "D336_PREPHYSICS_GATE",
            "checks": gate_checks,
            "contract_pass": bool(
                gate_checks["negative_controls"]
                and gate_checks["probe_controlled_physics_steps_zero"]
                and gate_checks["sim_step_counter_unchanged_during_scan"]
                and gate_checks["all_rows_counter_unchanged"]
                and gate_checks["decision_candidate_revalidated"]
            ),
            "candidate_pass": gate_checks["candidate_selected"],
            # Physics is never licensed inside D336, found or not.
            "physics_licensed": False,
            "controlled_physics_steps": controlled_physics_steps,
            "sim_step_counter_start": scan_counter_start,
            "sim_step_counter_end": scan_counter_end,
            "decision_candidate_parity": decision_parity,
            "selected": search_payload["selected"],
            "decision": search_payload["decision"],
        }
        _json_dump(args.out_dir / "d336_prephysics_gate.json", gate_payload)

        if not gate_payload["contract_pass"]:
            verdict = VERDICT_CONTRACT_FAIL
            classification = {
                "verdict": verdict,
                "interpretation": (
                    "a negative-control/step-counter/parity contract failed; the D336 search "
                    "conclusions are not licensed"
                ),
            }
        elif selected is not None:
            verdict = VERDICT_FOUND
            classification = {
                "verdict": verdict,
                "interpretation": (
                    "the registered continuous/exact method found at least one raw-tool-clear "
                    "candidate inside the frozen domain; the selected candidate is registered "
                    "for a separate later physics gate only - no physics ran in D336"
                ),
            }
        else:
            verdict = VERDICT_DISCHARGED
            classification = {
                "verdict": verdict,
                "interpretation": (
                    "the registered exact re-scoring, continuous Nelder-Mead, and micro-grid "
                    "method found no candidate clearing both audited raw tool shapes by >=+0.1mm "
                    "under the frozen alignment gates; the D335 finite-grid caveat is discharged "
                    "to this method's coverage, without claiming continuous-space impossibility"
                ),
            }

        marker_status = draw_frames(
            decision_candidate["_frames"], prim_path="/World/D336CaveatDiscriminatorFrames"
        )
        decision_png = args.out_dir / "d336_decision.png"
        snapshots.append(
            d335._write_raw_figure(
                decision_png,
                title=(
                    "D336 selected raw-clear candidate (registered, no physics)"
                    if selected is not None
                    else "D336 best exact-ranked point (no raw-clear candidate)"
                ),
                inner=inner,
                shapes=shapes,
                distance_set=decision_candidate["_distance_set"],
                canonical=decision_candidate["_canonical"],
            )
        )
        snapshot_paths.append(decision_png)
        map_png = args.out_dir / "d336_exact_clearance_map.png"
        any_clear = any(
            _ranking_metric(row) >= d332.SIGNED_DISTANCE_BORDER_M * 1000.0
            for row in cache.rows
        )
        snapshots.append(
            _write_clearance_map(
                map_png,
                stage_a_rows=stage_a_rows,
                other_rows=continuous_rows,
                decision=decision_candidate,
                selected=selected,
                any_clear=any_clear,
            )
        )
        snapshot_paths.append(map_png)

        rrd_path = args.out_dir / "d336_caveat_discriminator_trace.rrd"
        rrd_status = log_rerun(
            rrd_path,
            frames=decision_candidate["_frames"],
            joint_state={
                "label": "d336_finite_grid_caveat_discriminator",
                "object": "cylinder_d34_h90",
                "controlled_physics_steps_total": controlled_physics_steps,
                "physics_licensed": False,
            },
            joint_trace=[_decision_trace_row(inner, decision_candidate)],
            urdf_path=args.urdf_path,
            live_viewer=False,
            app_id="roarm_g0a_d336_finite_grid_caveat_discriminator",
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
            "active_case": "G0a cylinder D34xH90 finite-grid caveat discriminator",
            "new_variable": [],
            "new_physical_variable_count": 0,
            "frozen_contract": frozen_contract,
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "raw_source_contract": raw_source_contract,
            "negative_controls": negative,
            "search": search_payload,
            "prephysics_gate": gate_payload,
            "controlled_physics_steps_total": controlled_physics_steps,
            "classification": classification,
            "physics": {
                "executed": False,
                "licensed": False,
                "note": "physics evaluation of a registered candidate is a separate later gate",
            },
            "physics_eval_candidate_registered": search_payload["selected"],
            "outcome_guards": {
                "g0a_pass": False,
                "alignment_ladder_promoted": False,
                "mesh_rewritten": False,
                "collision_representation_changed": False,
                "domain_expanded_after_result": False,
                "physics_executed": False,
                "stop_after_d336": True,
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
                    ("frozen_contract", "d336_frozen_contract.json"),
                    ("scene_contract", "d336_prephysics_scene_contract.json"),
                    ("negative_control", "d336_negative_control.json"),
                    ("exact_rescore_csv", "d336_exact_rescore.csv"),
                    ("continuous_scan_csv", "d336_continuous_scan.csv"),
                    ("search", "d336_search.json"),
                    ("prephysics_gate", "d336_prephysics_gate.json"),
                    ("decision_png", "d336_decision.png"),
                    ("clearance_map_png", "d336_exact_clearance_map.png"),
                    ("rrd", "d336_caveat_discriminator_trace.rrd"),
                )
            }
            | {
                "summary_json": _rel(args.out_dir / "g0a_d336_finite_grid_caveat_summary.json"),
                "summary_markdown": _rel(args.out_dir / "g0a_d336_finite_grid_caveat_summary.md"),
            },
            "non_goals_respected": [
                "no physics steps of any kind (sim.step never called)",
                "no mesh/collision approximation or cooked-hull target compensation",
                "no target z/wrist/nullspace/gripper-angle change",
                "no domain expansion (r>17mm forbidden) or physics-driven retry",
                "no waypoint/approach/10-trial/close/grasp/lift",
                "no G0b/RL/PPO/randomization/VLA/RoArm/B200/cube",
            ],
        }
        _json_dump(args.out_dir / "g0a_d336_finite_grid_caveat_summary.json", summary)
        _write_summary_markdown(args.out_dir / "g0a_d336_finite_grid_caveat_summary.md", summary)
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
    del simulation_app  # App lifetime is owned by the launcher; no query pump is used in D336.
    try:
        try:
            summary = _run(args)
            decision = summary["search"].get("decision") or {}
            print(
                f"{summary['verdict']}: unique_evals={summary['search']['total_unique_count']} "
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
                "interpretation": "the current D336 invocation aborted before clean completion",
                "error": traceback.format_exc(),
            }
            _json_dump(args.out_dir / "d336_abort.json", abort_payload)
            fail_path = args.out_dir / "g0a_d336_finite_grid_caveat_summary.json"
            if not fail_path.is_file():
                _json_dump(fail_path, abort_payload)
            return 1
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
