#!/usr/bin/env python3
"""D335 audited target-family geometry repair for cylinder grasp G0a.

The only new physical variable is the coupled radial/tangent target geometry.
The probe first reproduces D334's old-target raw-mesh result and searches a
pre-registered grasp-semantic offset domain without calling ``sim.step``.  A
candidate must clear both audited raw tool shapes and retain all frozen G0a
commanded-pose gates before a D333-style baseline/static settle is licensed.
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
from sim_scripts import cyl34_top_view_d330_grasp_g0a_alignment_probe as d330
from sim_scripts import cyl34_top_view_d332_grasp_g0a_static_collision_discriminator as d332
from sim_scripts import cyl34_top_view_d333_grasp_g0a_sole_support_static_retest as d333
from sim_scripts import cyl34_top_view_d334_grasp_g0a_live_collision_shape_ownership_audit as d334


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d335"
D334_SUMMARY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d334/g0a_d334_live_collision_audit_summary.json"
)

OLD_RADIAL_UM = 7_000
OLD_TANGENT_UM = 11_000
RADIAL_MIN_UM = 0
RADIAL_MAX_UM = 17_000
TANGENT_MIN_UM = 9_000
TANGENT_MAX_UM = 14_000
COARSE_STEP_UM = 250
REFINE_RADIUS_UM = 500
REFINE_STEP_UM = 50
REFINE_CENTER_COUNT = 5

NEGATIVE_CONTROL_DISTANCE_TOL_MM = 0.05
RAW_CLEARANCE_M = d332.SIGNED_DISTANCE_BORDER_M
TARGET_TCP_GATE_M = d330.TCP_POS_GATE_M
TARGET_SETTLE_STEPS = 200

VERDICT_CONTRACT_FAIL = "D335_G0A_PREPHYSICS_CONTRACT_FAIL_STOP"
VERDICT_NO_FEASIBLE = "D335_G0A_TARGET_FAMILY_NO_FEASIBLE_CLEAR_STOP"
VERDICT_STATIC_PASS = "D335_G0A_TARGET_FAMILY_STATIC_REPAIR_SUPPORTED_STOP"
VERDICT_COLLIDER_BLOCKED = "D335_G0A_RAW_CLEAR_LIVE_COLLIDER_BLOCKED_STOP"
VERDICT_STATIC_MIXED = "D335_G0A_STATIC_RUNTIME_MIXED_STOP"
VERDICT_VIZ_FAIL = "D335_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP"


def _rel(path: Path) -> str:
    return d332._rel(path)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    d332._json_dump(path, payload)


def _range_inclusive(start: int, stop: int, step: int) -> list[int]:
    values = list(range(start, stop + 1, step))
    if not values or values[-1] != stop:
        values.append(stop)
    return values


def _canonical_for_offsets(radial_um: int, tangent_um: int) -> dict[str, Any]:
    radial_m = float(radial_um) * 1.0e-6
    tangent_m = float(tangent_um) * 1.0e-6
    radial = d332._unit([d332.OBJECT_CENTER_LOCAL_M[0], d332.OBJECT_CENTER_LOCAL_M[1], 0.0])
    tangent = np.asarray([-radial[1], radial[0], 0.0], dtype=np.float64) * d332.ADOPTED_TANGENT_SIGN
    target_tcp = d332.OBJECT_CENTER_LOCAL_M.copy()
    target_tcp -= radial * radial_m
    target_tcp -= tangent * tangent_m
    target_tcp[2] = d332.OBJECT_CENTER_LOCAL_M[2]

    ik = d332._solve_runtime_ik(
        target_tcp,
        d332.HOME_DEG,
        target_x_axis=None,
        target_z_axis=None,
        max_iter=120,
        pos_tol_mm=1.0,
    )
    q_deg = np.asarray(ik["q_deg"], dtype=np.float64)
    q_deg[5] = 0.0
    tcp, link5_pos, link5_rot = d332._fk_runtime_tcp(q_deg)
    return {
        "object_center_local_m": d332.OBJECT_CENTER_LOCAL_M.tolist(),
        "radial_axis": radial.tolist(),
        "tangent_axis": tangent.tolist(),
        "target_tcp_local_m": target_tcp.tolist(),
        "home_seed_deg": d332.HOME_DEG.tolist(),
        "joint_names": list(d332.ALL_JOINT_NAMES),
        "commanded_joint_deg": q_deg.tolist(),
        "commanded_joint_rad": np.radians(q_deg).tolist(),
        "commanded_tcp_local_m": tcp.tolist(),
        "commanded_link5_pos_local_m": link5_pos.tolist(),
        "commanded_link5_rot_local": link5_rot.tolist(),
        "commanded_tcp_error_mm": float(np.linalg.norm(tcp - target_tcp) * 1000.0),
        "ik": ik,
        "radial_center_offset_mm": radial_um / 1000.0,
        "tangent_center_offset_mm": tangent_um / 1000.0,
        "radial_tip_past_near_face_mm": (d332.CYLINDER_RADIUS_M - radial_m) * 1000.0,
        "target_formula": (
            f"TCP=center-radial*{radial_m:.6f}-tangent*{tangent_m:.6f}; z=center_z"
        ),
        # Only a visualization fallback; decision evidence always uses live raw FCL.
        "offline_witness_cylinder_local_m": d332.OBJECT_CENTER_LOCAL_M.tolist(),
    }


def _strip_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in shape.items() if not key.startswith("_")}


def _build_raw_shapes(inner: Any, d334_source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import hppfcl

    shapes: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    for body in d334.BODY_LABELS:
        rows = [row for row in d334._usd_collision_inventory(inner, body) if row["collision_enabled"]]
        expected_paths = d334_source["ownership"]["per_body"][body]["usd_enabled_collision_paths"]
        body_checks = {
            "exactly_one_enabled_collision_prim": len(rows) == 1,
            "path_matches_d334": len(rows) == 1 and [rows[0]["path"]] == expected_paths,
            "nearest_rigid_body_owner_matches": len(rows) == 1
            and rows[0]["nearest_rigid_body_ancestor"] == d334.BODY_PATHS[body],
        }
        checks[body] = {"checks": body_checks, "pass": all(body_checks.values()), "rows": rows}
        if not checks[body]["pass"]:
            continue
        source = d334._source_mesh_body_local(inner, rows[0], body)
        expected_shape = next(row for row in d334_source["shapes"] if row["body"] == body)
        expected_source = expected_shape["source_mesh"]
        source_parity_checks = {
            "mesh_prim_path": source["mesh_prim_path"] == expected_source["mesh_prim_path"],
            "vertex_count": int(source["vertex_count"]) == int(expected_source["vertex_count"]),
            "face_count": int(source["face_count"]) == int(expected_source["face_count"]),
            "triangle_count": int(source["triangle_count"])
            == int(expected_source["triangle_count"]),
            "fan_triangulated": bool(source["fan_triangulated"])
            == bool(expected_source["fan_triangulated"]),
            "body_local_bounds_bit_near": bool(
                np.allclose(
                    np.asarray(source["body_local_bounds_m"], dtype=np.float64),
                    np.asarray(expected_source["body_local_bounds_m"], dtype=np.float64),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ),
        }
        checks[body]["source_mesh_parity"] = source_parity_checks
        checks[body]["checks"]["source_mesh_matches_d334"] = all(source_parity_checks.values())
        checks[body]["pass"] = all(checks[body]["checks"].values())
        if not checks[body]["pass"]:
            continue
        shapes.append(
            {
                "body": body,
                "collider_path": rows[0]["path"],
                "owner_body_path": rows[0]["nearest_rigid_body_ancestor"],
                "source_mesh": {key: value for key, value in source.items() if not key.startswith("_")},
                "_raw_verts": source["_verts_body"],
                "_triangles": source["_triangles"],
                "_geom_raw": d332._build_raw_bvh(
                    hppfcl, source["_verts_body"], source["_triangles"]
                ),
            }
        )
    contract = {
        "per_body": checks,
        "body_set_exact": sorted(shape["body"] for shape in shapes) == sorted(d334.BODY_LABELS),
    }
    contract["pass"] = bool(
        contract["body_set_exact"] and all(checks[body]["pass"] for body in d334.BODY_LABELS)
    )
    return shapes, contract


def _raw_distance_matrix(inner: Any, shapes: list[dict[str, Any]], pose_label: str) -> dict[str, Any]:
    import hppfcl

    cylinder = hppfcl.Cylinder(d332.CYLINDER_RADIUS_M, d332.CYLINDER_HEIGHT_M)
    obj_pos, obj_quat = d334._object_pose_w(inner)
    cyl_tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(obj_quat), obj_pos)
    rows = []
    body_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for shape in shapes:
        body = shape["body"]
        if body not in body_poses:
            body_poses[body] = d334._body_pose_w(inner, body)
        pos, quat = body_poses[body]
        tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(quat), pos)
        query = d332._fcl_query(hppfcl, shape["_geom_raw"], tf, cylinder, cyl_tf)
        depth = None if query["contact"] is None else float(query["contact"]["penetration_depth_m"])
        sign_consistent = bool((not query["is_collision"]) or query["contact"] is not None)
        row = {
            "pose": pose_label,
            "body": body,
            "collider_path": shape["collider_path"],
            "representation": "raw",
            **query,
            "penetration_depth_m": depth,
            "sign_consistent": sign_consistent,
            "gjk_epa_depth_consistent_within_0p1mm": None,
            "consistency_hard_check_pass": sign_consistent,
        }
        row["overlap_state"] = d334._overlap_state(row)
        row["clear_pass"] = bool(row["overlap_state"] == "clear")
        rows.append(row)
    return {
        "pose": pose_label,
        "object_pos_w_m": obj_pos.tolist(),
        "object_quat_wxyz": obj_quat.tolist(),
        "body_poses_w": {
            body: {"pos_m": pose[0].tolist(), "quat_wxyz": pose[1].tolist()}
            for body, pose in body_poses.items()
        },
        "queries": rows,
    }


def _queries_by_body(distance_set: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["body"]: row for row in distance_set["queries"]}


def _alignment_at_current(
    inner: Any,
    canonical: dict[str, Any],
    *,
    object_start_w: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    actual_tcp = inner._tcp_pos_w[0].detach().cpu().numpy().astype(np.float64) - origin
    _link5_pos, link5_quat = d334._body_pose_w(inner, "link5")
    link5_rot = d332._quat_wxyz_to_rot(link5_quat)
    object_now_w = inner._sponge.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    start_w = object_now_w if object_start_w is None else np.asarray(object_start_w, dtype=np.float64)
    actual_all = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
    arm_ids = [int(value) for value in inner._bc_arm_joint_ids]
    target_full = np.asarray(canonical["commanded_joint_rad"], dtype=np.float64)
    actual_arm = actual_all[arm_ids]
    target_arm = target_full[:5]
    return d330._evaluate_alignment(
        trial=1,
        obj_center=d332.OBJECT_CENTER_LOCAL_M,
        obj_radius_m=d332.CYLINDER_RADIUS_M,
        obj_height_m=d332.CYLINDER_HEIGHT_M,
        target_tcp=np.asarray(canonical["target_tcp_local_m"], dtype=np.float64),
        tangent=np.asarray(canonical["tangent_axis"], dtype=np.float64),
        actual_tcp=actual_tcp,
        link5_rot=link5_rot,
        obj_start_w=start_w,
        obj_final_w=object_now_w,
        target_arm=target_arm,
        actual_arm=actual_arm,
        ik_failure_steps=0 if canonical["ik"]["converged"] else 1,
    )


def _candidate_public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _candidate_flat(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "stage",
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
        "gripper_link_raw_signed_distance_mm",
        "gripper_link_raw_state",
        "min_raw_clearance_mm",
        "legacy_alignment_pass",
        "raw_tool_clear_pass",
        "pass",
        "sim_step_counter_unchanged",
    )
    return {key: row.get(key) for key in keys}


def _evaluate_candidate(
    inner: Any,
    shapes: list[dict[str, Any]],
    radial_um: int,
    tangent_um: int,
    *,
    stage: str,
) -> dict[str, Any]:
    canonical = _canonical_for_offsets(radial_um, tangent_um)
    counter_before = int(inner._sim_step_counter)
    command = d332._write_exact_state(
        inner,
        np.asarray(canonical["commanded_joint_rad"], dtype=np.float64),
        d332.OBJECT_CENTER_LOCAL_M,
    )
    counter_after = int(inner._sim_step_counter)
    distances = _raw_distance_matrix(inner, shapes, f"d335_{stage}")
    queries = _queries_by_body(distances)
    alignment, frames = _alignment_at_current(inner, canonical)
    canonical["offline_witness_cylinder_local_m"] = queries["gripper_link"][
        "nearest_point_cylinder_m"
    ]
    radial_m = radial_um * 1.0e-6
    shift_mm = math.hypot(radial_um - OLD_RADIAL_UM, tangent_um - OLD_TANGENT_UM) / 1000.0
    legacy_checks = {
        "ik_converged": bool(canonical["ik"]["converged"]),
        "commanded_tcp_error_le_5mm": float(canonical["commanded_tcp_error_mm"])
        <= TARGET_TCP_GATE_M * 1000.0,
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
    min_clearance_mm = min(float(queries[body]["signed_distance_mm"]) for body in d334.BODY_LABELS)
    row = {
        "stage": stage,
        "radial_offset_um": int(radial_um),
        "tangent_offset_um": int(tangent_um),
        "radial_offset_mm": radial_um / 1000.0,
        "tangent_offset_mm": tangent_um / 1000.0,
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
        "gripper_link_raw_signed_distance_mm": float(
            queries["gripper_link"]["signed_distance_mm"]
        ),
        "gripper_link_raw_state": queries["gripper_link"]["overlap_state"],
        "min_raw_clearance_mm": min_clearance_mm,
        "legacy_checks": legacy_checks,
        "raw_checks": raw_checks,
        "legacy_alignment_pass": all(legacy_checks.values()),
        "raw_tool_clear_pass": all(raw_checks.values()),
        "sim_step_counter_before": counter_before,
        "sim_step_counter_after": counter_after,
        "sim_step_counter_unchanged": counter_before == counter_after,
        "pass": bool(
            all(legacy_checks.values()) and all(raw_checks.values()) and counter_before == counter_after
        ),
        "_canonical": canonical,
        "_distance_set": distances,
        "_alignment": alignment,
        "_frames": frames,
        "_command": command,
    }
    return row


def _negative_control(
    inner: Any,
    shapes: list[dict[str, Any]],
    d334_source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _evaluate_candidate(
        inner, shapes, OLD_RADIAL_UM, OLD_TANGENT_UM, stage="old_target_negative_control"
    )
    expected = {}
    for distance_set in d334_source["distance_sets"]:
        if distance_set["pose"] != "pose_a_prestep":
            continue
        for row in distance_set["queries"]:
            if row["representation"] == "raw":
                expected[row["body"]] = row
    observed = _queries_by_body(candidate["_distance_set"])
    body_rows = {}
    for body in d334.BODY_LABELS:
        exp = expected[body]
        obs = observed[body]
        expected_state = d334._overlap_state(exp)
        body_rows[body] = {
            "expected_signed_distance_mm": float(exp["signed_distance_mm"]),
            "observed_signed_distance_mm": float(obs["signed_distance_mm"]),
            "absolute_delta_mm": abs(
                float(obs["signed_distance_mm"]) - float(exp["signed_distance_mm"])
            ),
            "tolerance_mm": NEGATIVE_CONTROL_DISTANCE_TOL_MM,
            "expected_state": expected_state,
            "observed_state": obs["overlap_state"],
            "consistency_hard_check_pass": bool(obs["consistency_hard_check_pass"]),
        }
        body_rows[body]["pass"] = bool(
            body_rows[body]["absolute_delta_mm"] <= NEGATIVE_CONTROL_DISTANCE_TOL_MM
            and body_rows[body]["expected_state"] == body_rows[body]["observed_state"]
            and body_rows[body]["consistency_hard_check_pass"]
        )
    payload = {
        "artifact": "D335_OLD_TARGET_NEGATIVE_CONTROL",
        "controlled_physics_steps": 0,
        "sim_step_counter_unchanged": candidate["sim_step_counter_unchanged"],
        "per_body": body_rows,
        "pass": bool(
            candidate["sim_step_counter_unchanged"]
            and all(body_rows[body]["pass"] for body in d334.BODY_LABELS)
        ),
    }
    return payload, candidate


def _write_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [_candidate_flat(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def _search_candidates(inner: Any, shapes: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    total_coarse = len(_range_inclusive(RADIAL_MIN_UM, RADIAL_MAX_UM, COARSE_STEP_UM)) * len(
        _range_inclusive(TANGENT_MIN_UM, TANGENT_MAX_UM, COARSE_STEP_UM)
    )
    completed = 0
    for radial_um in _range_inclusive(RADIAL_MIN_UM, RADIAL_MAX_UM, COARSE_STEP_UM):
        for tangent_um in _range_inclusive(TANGENT_MIN_UM, TANGENT_MAX_UM, COARSE_STEP_UM):
            row = _evaluate_candidate(inner, shapes, radial_um, tangent_um, stage="coarse")
            rows.append(row)
            by_key[(radial_um, tangent_um)] = row
            completed += 1
            if completed % 100 == 0 or completed == total_coarse:
                print(f"D335 coarse candidate {completed}/{total_coarse}", flush=True)

    # The preregistered refinement trigger is raw geometry only.  A coarse
    # raw-clear point that later fails an alignment gate does not authorize an
    # adaptive local search around other points.
    coarse_raw_clear = [
        row for row in rows if row["raw_tool_clear_pass"] and row["sim_step_counter_unchanged"]
    ]
    refine_centers: list[dict[str, Any]] = []
    if not coarse_raw_clear:
        eligible = [row for row in rows if row["sim_step_counter_unchanged"]]
        eligible.sort(
            key=lambda row: (
                -float(row["min_raw_clearance_mm"]),
                float(row["shift_from_old_mm"]),
                int(row["radial_offset_um"]),
                int(row["tangent_offset_um"]),
            )
        )
        refine_centers = eligible[:REFINE_CENTER_COUNT]
        refine_keys = set()
        for center in refine_centers:
            r0 = int(center["radial_offset_um"])
            t0 = int(center["tangent_offset_um"])
            for dr in range(-REFINE_RADIUS_UM, REFINE_RADIUS_UM + 1, REFINE_STEP_UM):
                for dt in range(-REFINE_RADIUS_UM, REFINE_RADIUS_UM + 1, REFINE_STEP_UM):
                    r = min(RADIAL_MAX_UM, max(RADIAL_MIN_UM, r0 + dr))
                    t = min(TANGENT_MAX_UM, max(TANGENT_MIN_UM, t0 + dt))
                    refine_keys.add((r, t))
        pending = sorted(key for key in refine_keys if key not in by_key)
        for idx, (radial_um, tangent_um) in enumerate(pending, start=1):
            row = _evaluate_candidate(inner, shapes, radial_um, tangent_um, stage="refine")
            rows.append(row)
            by_key[(radial_um, tangent_um)] = row
            if idx % 100 == 0 or idx == len(pending):
                print(f"D335 refine candidate {idx}/{len(pending)}", flush=True)

    passing = [row for row in rows if row["pass"]]
    passing.sort(
        key=lambda row: (
            float(row["shift_from_old_mm"]),
            -float(row["min_raw_clearance_mm"]),
            int(row["radial_offset_um"]),
            int(row["tangent_offset_um"]),
        )
    )
    eligible_all = [
        row for row in rows if row["legacy_alignment_pass"] and row["sim_step_counter_unchanged"]
    ]
    eligible_all.sort(
        key=lambda row: (
            -float(row["min_raw_clearance_mm"]),
            float(row["shift_from_old_mm"]),
            int(row["radial_offset_um"]),
            int(row["tangent_offset_um"]),
        )
    )
    return {
        "rows": rows,
        "selected": passing[0] if passing else None,
        "best": passing[0] if passing else (eligible_all[0] if eligible_all else rows[0]),
        "coarse_count": total_coarse,
        "refine_count": sum(1 for row in rows if row["stage"] == "refine"),
        "passing_count": len(passing),
        "legacy_alignment_count": sum(1 for row in rows if row["legacy_alignment_pass"]),
        "refine_centers": refine_centers,
        "top_by_clearance": eligible_all[:20],
    }


def _write_raw_figure(
    path: Path,
    *,
    title: str,
    inner: Any,
    shapes: list[dict[str, Any]],
    distance_set: dict[str, Any],
    canonical: dict[str, Any],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.5, 7.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    obj_pos, obj_quat = d334._object_pose_w(inner)
    all_points = [d334._plot_cylinder(ax, obj_pos, obj_quat)]
    colors = {"link5": "tab:blue", "gripper_link": "tab:green"}
    for shape in shapes:
        pos, quat = d334._body_pose_w(inner, shape["body"])
        rot = d332._quat_wxyz_to_rot(quat)
        raw = shape["_raw_verts"]
        stride = max(1, len(raw) // 1500)
        world = (rot @ raw[::stride].T).T + pos
        ax.scatter(
            world[:, 0],
            world[:, 1],
            world[:, 2],
            s=1.1,
            color=colors[shape["body"]],
            alpha=0.55,
            label=f"{shape['body']} raw STL",
        )
        all_points.append(world)
    for query in distance_set["queries"]:
        p_tool = np.asarray(query["nearest_point_geometry_m"], dtype=np.float64)
        p_cyl = np.asarray(query["nearest_point_cylinder_m"], dtype=np.float64)
        segment = np.vstack([p_tool, p_cyl])
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            linewidth=2.0,
            color=colors[query["body"]],
        )
        ax.scatter(
            [p_tool[0], p_cyl[0]], [p_tool[1], p_cyl[1]], [p_tool[2], p_cyl[2]], s=22
        )
        all_points.append(segment)
    origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    target = origin + np.asarray(canonical["target_tcp_local_m"], dtype=np.float64)
    commanded = origin + np.asarray(canonical["commanded_tcp_local_m"], dtype=np.float64)
    actual = inner._tcp_pos_w[0].detach().cpu().numpy().astype(np.float64)
    for point, color, label, marker in (
        (target, "red", "target TCP", "*"),
        (commanded, "purple", "commanded FK TCP", "P"),
        (actual, "black", "actual TCP", "x"),
    ):
        ax.scatter([point[0]], [point[1]], [point[2]], color=color, marker=marker, s=80, label=label)
        all_points.append(point.reshape(1, 3))
    target_rot = np.column_stack(
        [
            np.asarray(canonical["tangent_axis"], dtype=np.float64),
            np.cross(
                [0.0, 0.0, 1.0], np.asarray(canonical["tangent_axis"], dtype=np.float64)
            ),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        ]
    )
    commanded_rot = np.asarray(canonical["commanded_link5_rot_local"], dtype=np.float64)
    _actual_link5_pos, actual_link5_quat = d334._body_pose_w(inner, "link5")
    actual_rot = d332._quat_wxyz_to_rot(actual_link5_quat)
    for point, rot, prefix in (
        (target, target_rot, "target"),
        (commanded, commanded_rot, "commanded"),
        (actual, actual_rot, "actual"),
    ):
        for axis_idx, color in enumerate(("r", "g", "b")):
            vec = rot[:, axis_idx] * 0.018
            ax.quiver(
                point[0], point[1], point[2], vec[0], vec[1], vec[2],
                color=color, linewidth=1.0, arrow_length_ratio=0.18,
            )
    d332._set_axes_equal(ax, np.vstack(all_points))
    query_text = ", ".join(
        f"{row['body']}={row['signed_distance_mm']:+.4f}mm/{row['overlap_state']}"
        for row in distance_set["queries"]
    )
    ax.set_title(title + "\n" + query_text)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper left", fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return _rel(path)


def _joint_trace_row(inner: Any, candidate: dict[str, Any], step: int = 0) -> dict[str, Any]:
    actual = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
    command = candidate["_command"][0].detach().cpu().numpy().astype(np.float64)
    return {
        "step": int(step),
        "phase": "d335_prephysics_decision",
        "actual_joint_rad_by_name": {
            name: float(actual[idx]) for idx, name in enumerate(inner._robot.joint_names)
        },
        "commanded_joint_rad_by_name": {
            name: float(command[idx]) for idx, name in enumerate(inner._robot.joint_names)
        },
        "radial_offset_mm": candidate["radial_offset_mm"],
        "tangent_offset_mm": candidate["tangent_offset_mm"],
        "min_raw_clearance_mm": candidate["min_raw_clearance_mm"],
        "frames": candidate["_frames"],
    }


def _runtime_alignment(
    inner: Any,
    canonical: dict[str, Any],
    object_start_w: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _alignment_at_current(inner, canonical, object_start_w=object_start_w)


def _raw_trace_flat(step: int, distance_set: dict[str, Any]) -> dict[str, Any]:
    queries = _queries_by_body(distance_set)
    return {
        "step": int(step),
        "link5_raw_signed_distance_mm": float(queries["link5"]["signed_distance_mm"]),
        "link5_raw_state": queries["link5"]["overlap_state"],
        "link5_raw_clear_pass": bool(queries["link5"]["clear_pass"]),
        "gripper_link_raw_signed_distance_mm": float(
            queries["gripper_link"]["signed_distance_mm"]
        ),
        "gripper_link_raw_state": queries["gripper_link"]["overlap_state"],
        "gripper_link_raw_clear_pass": bool(queries["gripper_link"]["clear_pass"]),
    }


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _classify_static(
    *,
    baseline_stats: dict[str, Any],
    target_stats: dict[str, Any] | None,
    final_alignment: dict[str, Any] | None,
    raw_sets: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if not baseline_stats["hard_gate_pass"] or target_stats is None or final_alignment is None:
        interpretation = (
            "the conditional sole-support baseline failed; target settle was not licensed"
            if not baseline_stats["hard_gate_pass"]
            else "the re-materialized target failed its pre-settle raw-clear/runtime contract; no target step ran"
        )
        return VERDICT_STATIC_MIXED, {
            "interpretation": interpretation,
            "static_clean_pass": False,
        }
    raw_clear_all = all(
        all(row["clear_pass"] and row["consistency_hard_check_pass"] for row in item["queries"])
        for item in raw_sets
    )
    robot_force_quiet = all(
        float(target_stats["max_force_n_by_link"][body]) < d332.ROBOT_FORCE_EVENT_N
        for body in ("link4", "link5", "gripper_link")
    )
    root_pass = bool(
        float(target_stats["max_robot_root_position_drift_m"]) <= d333.ROOT_POSITION_DRIFT_TOL_M
        and float(target_stats["max_robot_root_rotation_drift_rad"]) <= d333.ROOT_ROTATION_DRIFT_TOL_RAD
    )
    support_pass = bool(
        float(target_stats["support_table_force_z_last50_median_n"])
        > d332.SUPPORT_POSITIVE_CONTROL_N
        and float(target_stats["max_abs_bottom_table_gap_mm"])
        <= d333.TAIL_SUPPORT_GAP_TOL_M * 1000.0
    )
    disturbance_free = bool(
        int(target_stats["object_disturbance_start_step"]) < 0
        and float(target_stats["max_object_disp_xy_mm"]) < d332.DISTURBANCE_XY_M * 1000.0
        and float(target_stats["max_object_tilt_deg"]) < d332.DISTURBANCE_TILT_DEG
    )
    audited_contact_steps = [
        int(target_stats["first_contact_step_by_link"][body])
        for body in ("link5", "gripper_link")
        if int(target_stats["first_contact_step_by_link"][body]) >= 0
    ]
    first_audited_contact = min(audited_contact_steps) if audited_contact_steps else -1
    disturbance_step = int(target_stats["object_disturbance_start_step"])
    link4_contact_step = int(target_stats["first_contact_step_by_link"]["link4"])
    link4_nonconfounding = bool(
        link4_contact_step < 0
        or (
            first_audited_contact >= 0
            and disturbance_step >= 0
            and link4_contact_step > max(first_audited_contact, disturbance_step)
        )
    )
    audited_contact_timing_compatible = bool(
        first_audited_contact >= 0
        and disturbance_step >= 0
        and first_audited_contact <= disturbance_step + 1
        and link4_nonconfounding
    )
    checks = {
        "baseline_hard_gate": bool(baseline_stats["hard_gate_pass"]),
        "raw_tool_clear_all_recorded_readings": raw_clear_all,
        "robot_filters_max_lt_0p1n": robot_force_quiet,
        "root_contract": root_pass,
        "support_contract": support_pass,
        "d333_disturbance_free": disturbance_free,
        "final_g0a_alignment": bool(final_alignment["pass_all"]),
        "final_object_displacement_lt_5mm": float(final_alignment["object_disp_xy_mm"])
        < d330.OBJECT_DISP_GATE_M * 1000.0,
        "audited_tool_contact_timing_compatible": audited_contact_timing_compatible,
    }
    static_clean_checks = {
        key: value for key, value in checks.items() if key != "audited_tool_contact_timing_compatible"
    }
    if all(static_clean_checks.values()):
        verdict = VERDICT_STATIC_PASS
        interpretation = (
            "the selected raw-tool-clear target retained all frozen alignment, support, contact, "
            "and disturbance gates during the one static settle"
        )
    elif raw_clear_all and root_pass and support_pass and audited_contact_timing_compatible:
        verdict = VERDICT_COLLIDER_BLOCKED
        interpretation = (
            "the audited raw tool remained clear while live robot contact or object disturbance "
            "persisted; route to the deferred collision-representation case without target retuning"
        )
    else:
        verdict = VERDICT_STATIC_MIXED
        interpretation = (
            "the conditional static run failed one or more raw-clear, alignment, support, root, or "
            "disturbance gates; no single cause is promoted"
        )
    return verdict, {
        "checks": checks,
        "static_clean_pass": all(static_clean_checks.values()),
        "first_audited_tool_contact_step": first_audited_contact,
        "first_link4_contact_step": link4_contact_step,
        "link4_nonconfounding": link4_nonconfounding,
        "object_disturbance_step": disturbance_step,
        "interpretation": interpretation,
    }


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    best = summary["candidate_search"].get("best") or {}
    lines = [
        "# D335 target-family geometry repair",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "| Gate / metric | Result |",
        "|---|---:|",
        f"| Frozen contract | `{summary['frozen_contract']['pass']}` |",
        f"| Old-target negative control | `{summary['negative_control']['pass']}` |",
        f"| Controlled physics steps before gate | `{summary['controlled_physics_steps_before_gate']}` |",
        f"| Candidates evaluated | `{summary['candidate_search']['evaluated_count']}` |",
        f"| Raw-clear passing candidates | `{summary['candidate_search']['passing_count']}` |",
        f"| Best r / t | `{best.get('radial_offset_mm')} / {best.get('tangent_offset_mm')} mm` |",
        f"| Best minimum raw clearance | `{best.get('min_raw_clearance_mm')} mm` |",
        f"| Physics executed | `{summary['physics']['executed']}` |",
        f"| Artifact contract | `{summary['artifact_contract']['pass']}` |",
        "",
        summary["classification"]["interpretation"] + ".",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    d332._runtime_versions()  # Pin gate before scene creation.
    source = json.loads(D334_SUMMARY.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pin = d334._pin_check()
    frozen_checks = {
        "d334_verdict": source["verdict"] == "D334_G0A_ACTUAL_TOOL_OVERLAP_SUPPORTED",
        "seed_33201": int(args.seed) == 33201,
        "robot_usd_hash_matches_d334": d332._sha256(args.robot_usd_path)
        == source["frozen_contract"]["robot_usd_sha256"],
        "urdf_hash_matches_d334": d332._sha256(args.urdf_path)
        == source["frozen_contract"]["urdf_sha256"],
        "numpy_pin": bool(pin["numpy_pin_1_26_0"]),
        "psutil_pin": bool(pin["psutil_pin_5_9_8"]),
        "search_domain_exact": (
            RADIAL_MIN_UM,
            RADIAL_MAX_UM,
            TANGENT_MIN_UM,
            TANGENT_MAX_UM,
            COARSE_STEP_UM,
            REFINE_RADIUS_UM,
            REFINE_STEP_UM,
        )
        == (0, 17_000, 9_000, 14_000, 250, 500, 50),
    }
    frozen_contract = {
        "checks": frozen_checks,
        "pass": all(frozen_checks.values()),
        "source": _rel(D334_SUMMARY),
        "source_sha256": d332._sha256(D334_SUMMARY),
        "robot_usd": _rel(args.robot_usd_path),
        "robot_usd_sha256": d332._sha256(args.robot_usd_path),
        "urdf": _rel(args.urdf_path),
        "urdf_sha256": d332._sha256(args.urdf_path),
        "environment": pin,
    }
    _json_dump(args.out_dir / "d335_frozen_contract.json", frozen_contract)
    if not frozen_contract["pass"]:
        raise RuntimeError("D335 frozen contract failed before scene creation")

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
        shapes, raw_source_contract = _build_raw_shapes(inner, source)
        pre_scene_checks = {
            "stage_contract": bool(stage_contract["hard_contract_pass"]),
            "sensor_contract": bool(sensor_contract["hard_contract_pass"]),
            "raw_source_contract": bool(raw_source_contract["pass"]),
        }
        _json_dump(
            args.out_dir / "d335_prephysics_scene_contract.json",
            {
                "artifact": "D335_PREPHYSICS_SCENE_CONTRACT",
                "checks": pre_scene_checks,
                "pass": all(pre_scene_checks.values()),
                "stage_contract": stage_contract,
                "sensor_contract": sensor_contract,
                "raw_source_contract": raw_source_contract,
                "raw_shapes": [_strip_shape(shape) for shape in shapes],
            },
        )
        if not all(pre_scene_checks.values()):
            raise RuntimeError("D335 stage/sensor/raw-source contract failed")

        scan_counter_start = int(inner._sim_step_counter)
        negative, negative_candidate = _negative_control(inner, shapes, source)
        _json_dump(args.out_dir / "d335_old_target_negative_control.json", negative)

        search: dict[str, Any] | None = None
        if negative["pass"]:
            search = _search_candidates(inner, shapes)
            _write_candidate_csv(args.out_dir / "d335_candidate_scan.csv", search["rows"])
        scan_counter_end = int(inner._sim_step_counter)
        counter_unchanged = scan_counter_start == scan_counter_end

        if search is None:
            decision_candidate = negative_candidate
            search_payload = {
                "executed": False,
                "evaluated_count": 0,
                "coarse_count": 0,
                "refine_count": 0,
                "legacy_alignment_count": 0,
                "passing_count": 0,
                "selected": None,
                "best": _candidate_public(negative_candidate),
                "top_by_clearance": [],
                "refine_centers": [],
            }
        else:
            decision_candidate = search["selected"] or search["best"]
            search_payload = {
                "executed": True,
                "domain": {
                    "radial_mm": [0.0, 17.0],
                    "tangent_mm": [9.0, 14.0],
                    "coarse_step_mm": 0.25,
                    "refine_radius_mm": 0.50,
                    "refine_step_mm": 0.05,
                    "refine_center_count": 5,
                    "anti_retreat": "17mm-r >= 0",
                },
                "evaluated_count": len(search["rows"]),
                "coarse_count": search["coarse_count"],
                "refine_count": search["refine_count"],
                "legacy_alignment_count": search["legacy_alignment_count"],
                "passing_count": search["passing_count"],
                "selected": None
                if search["selected"] is None
                else _candidate_public(search["selected"]),
                "best": _candidate_public(search["best"]),
                "top_by_clearance": [_candidate_public(row) for row in search["top_by_clearance"]],
                "refine_centers": [_candidate_public(row) for row in search["refine_centers"]],
            }
        _json_dump(
            args.out_dir / "d335_candidate_search.json",
            {"artifact": "D335_CANDIDATE_SEARCH", **search_payload},
        )

        gate_checks = {
            "negative_control": bool(negative["pass"]),
            "probe_controlled_physics_steps_zero": controlled_physics_steps == 0,
            "sim_step_counter_unchanged_during_scan": counter_unchanged,
            "candidate_selected": bool(search is not None and search["selected"] is not None),
        }
        gate_payload = {
            "artifact": "D335_PREPHYSICS_GATE",
            "checks": gate_checks,
            "contract_pass": bool(
                gate_checks["negative_control"]
                and gate_checks["probe_controlled_physics_steps_zero"]
                and gate_checks["sim_step_counter_unchanged_during_scan"]
            ),
            "candidate_pass": gate_checks["candidate_selected"],
            "physics_licensed": all(gate_checks.values()),
            "controlled_physics_steps": controlled_physics_steps,
            "sim_step_counter_start": scan_counter_start,
            "sim_step_counter_end": scan_counter_end,
            "selected": search_payload["selected"],
            "best": search_payload["best"],
        }
        _json_dump(args.out_dir / "d335_prephysics_gate.json", gate_payload)

        # Re-materialize the decision candidate after the last scan row.
        decision_candidate = _evaluate_candidate(
            inner,
            shapes,
            int(decision_candidate["radial_offset_um"]),
            int(decision_candidate["tangent_offset_um"]),
            stage="decision_snapshot",
        )
        selected_reference = None if search is None else search["selected"]
        if selected_reference is None:
            decision_parity = {"required": False, "pass": True}
        else:
            parity_checks = {
                "radial_offset_exact": int(decision_candidate["radial_offset_um"])
                == int(selected_reference["radial_offset_um"]),
                "tangent_offset_exact": int(decision_candidate["tangent_offset_um"])
                == int(selected_reference["tangent_offset_um"]),
                "link5_distance_delta_le_0p05mm": abs(
                    float(decision_candidate["link5_raw_signed_distance_mm"])
                    - float(selected_reference["link5_raw_signed_distance_mm"])
                )
                <= NEGATIVE_CONTROL_DISTANCE_TOL_MM,
                "gripper_distance_delta_le_0p05mm": abs(
                    float(decision_candidate["gripper_link_raw_signed_distance_mm"])
                    - float(selected_reference["gripper_link_raw_signed_distance_mm"])
                )
                <= NEGATIVE_CONTROL_DISTANCE_TOL_MM,
                "raw_states_exact": (
                    decision_candidate["link5_raw_state"] == selected_reference["link5_raw_state"]
                    and decision_candidate["gripper_link_raw_state"]
                    == selected_reference["gripper_link_raw_state"]
                ),
            }
            decision_parity = {
                "required": True,
                "checks": parity_checks,
                "pass": all(parity_checks.values()),
            }
        gate_checks["decision_candidate_revalidated"] = bool(
            not gate_checks["candidate_selected"]
            or (
                decision_candidate["pass"]
                and decision_candidate["sim_step_counter_unchanged"]
                and decision_parity["pass"]
            )
        )
        gate_payload["checks"] = gate_checks
        gate_payload["contract_pass"] = bool(
            gate_checks["negative_control"]
            and gate_checks["probe_controlled_physics_steps_zero"]
            and gate_checks["sim_step_counter_unchanged_during_scan"]
            and gate_checks["decision_candidate_revalidated"]
        )
        gate_payload["candidate_pass"] = gate_checks["candidate_selected"]
        gate_payload["physics_licensed"] = bool(
            gate_payload["contract_pass"] and gate_payload["candidate_pass"]
        )
        gate_payload["decision_candidate_revalidation"] = _candidate_public(decision_candidate)
        gate_payload["decision_candidate_parity"] = decision_parity
        _json_dump(args.out_dir / "d335_prephysics_gate.json", gate_payload)
        pre_png = args.out_dir / "d335_prephysics_decision.png"
        snapshots.append(
            _write_raw_figure(
                pre_png,
                title=(
                    "D335 selected raw-clear target"
                    if gate_payload["physics_licensed"]
                    else "D335 best bounded target (no raw-clear candidate)"
                ),
                inner=inner,
                shapes=shapes,
                distance_set=decision_candidate["_distance_set"],
                canonical=decision_candidate["_canonical"],
            )
        )
        snapshot_paths.append(pre_png)

        baseline_rows: list[dict[str, Any]] = []
        target_rows: list[dict[str, Any]] = []
        raw_runtime_sets: list[dict[str, Any]] = []
        raw_trace_rows: list[dict[str, Any]] = []
        baseline_stats: dict[str, Any] | None = None
        target_stats: dict[str, Any] | None = None
        final_alignment: dict[str, Any] | None = None
        target_prestep_clear: bool | None = None
        classification: dict[str, Any]

        if not gate_payload["contract_pass"]:
            verdict = VERDICT_CONTRACT_FAIL
            classification = {
                "verdict": verdict,
                "interpretation": "the old-target/step-counter pre-physics contract failed; search conclusions are not licensed",
            }
        elif not gate_payload["candidate_pass"]:
            verdict = VERDICT_NO_FEASIBLE
            classification = {
                "verdict": verdict,
                "interpretation": (
                    "no candidate in the executed pre-registered coarse/refinement set within the "
                    "grasp-semantic radial/tangent domain made both audited raw tool shapes clear "
                    "while retaining every frozen alignment gate"
                ),
            }
        else:
            canonical = decision_candidate["_canonical"]
            q_home = np.radians(np.asarray(d332.HOME_DEG, dtype=np.float64))
            q_home[5] = 0.0
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
                        phase="d335_sole_support_baseline",
                        step=step,
                        command_target=home_target,
                        canonical=canonical,
                        object_start_w=baseline_start_w,
                        root_start_pos_w=baseline_root_pos,
                        root_start_quat_wxyz=baseline_root_quat,
                        contact=contact,
                    )
                )
            d333._write_trace_csv(args.out_dir / "d335_baseline_trace.csv", baseline_rows)
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
                pre_target_raw = _raw_distance_matrix(inner, shapes, "target_prestep")
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
                                phase="d335_target_static_settle",
                                step=step,
                                command_target=command,
                                canonical=canonical,
                                object_start_w=object_start_w,
                                root_start_pos_w=target_root_pos,
                                root_start_quat_wxyz=target_root_quat,
                                contact=contact,
                            )
                        )
                        raw_set = _raw_distance_matrix(inner, shapes, f"target_poststep_{step}")
                        raw_trace_rows.append(_raw_trace_flat(step, raw_set))
                        # Keep every reading for classification so a transient raw
                        # overlap cannot be hidden by a clear post-step-0/final pair.
                        raw_runtime_sets.append(raw_set)
                    d333._write_trace_csv(args.out_dir / "d335_target_static_trace.csv", target_rows)
                    _write_dict_csv(args.out_dir / "d335_target_raw_distance_trace.csv", raw_trace_rows)
                    target_stats = d333._target_statistics(target_rows)
                    target_stats["max_robot_root_position_drift_m"] = max(
                        float(row["robot_root_position_drift_m"]) for row in target_rows
                    )
                    target_stats["max_robot_root_rotation_drift_rad"] = max(
                        float(row["robot_root_rotation_drift_rad"]) for row in target_rows
                    )
                    final_alignment, _final_frames = _runtime_alignment(inner, canonical, object_start_w)
                    final_png = args.out_dir / "d335_static_final.png"
                    snapshots.append(
                        _write_raw_figure(
                            final_png,
                            title="D335 conditional static settle final",
                            inner=inner,
                            shapes=shapes,
                            distance_set=raw_runtime_sets[-1],
                            canonical=canonical,
                        )
                    )
                    snapshot_paths.append(final_png)
            verdict, static_classification = _classify_static(
                baseline_stats=baseline_stats,
                target_stats=target_stats,
                final_alignment=final_alignment,
                raw_sets=raw_runtime_sets,
            )
            classification = {"verdict": verdict, **static_classification}

        marker_frames = target_rows[-1]["frames"] if target_rows else decision_candidate["_frames"]
        marker_status = draw_frames(marker_frames, prim_path="/World/D335TargetRepairFrames")
        rrd_path = args.out_dir / "d335_target_family_repair_trace.rrd"
        rrd_trace = target_rows if target_rows else [_joint_trace_row(inner, decision_candidate)]
        rrd_status = log_rerun(
            rrd_path,
            frames=marker_frames,
            joint_state={
                "label": "d335_target_family_geometry_repair",
                "object": "cylinder_d34_h90",
                "controlled_physics_steps_before_gate": 0,
                "controlled_physics_steps_total": controlled_physics_steps,
                "physics_licensed": gate_payload["physics_licensed"],
            },
            joint_trace=rrd_trace,
            urdf_path=args.urdf_path,
            live_viewer=False,
            app_id="roarm_g0a_d335_target_family_repair",
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
            "active_case": "G0a cylinder D34xH90 target-family geometry repair",
            "new_variable": ["target_family_geometry"],
            "frozen_contract": frozen_contract,
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "raw_source_contract": raw_source_contract,
            "negative_control": negative,
            "candidate_search": search_payload,
            "prephysics_gate": gate_payload,
            "controlled_physics_steps_before_gate": 0,
            "classification": classification,
            "physics": {
                "executed": bool(gate_payload["physics_licensed"]),
                "controlled_steps_total": controlled_physics_steps,
                "baseline": baseline_stats,
                "target_static": target_stats,
                "final_alignment": final_alignment,
                "target_prestep_raw_clear": target_prestep_clear,
                "raw_distance_sets": raw_runtime_sets,
            },
            "outcome_guards": {
                "g0a_pass": False,
                "alignment_ladder_promoted": False,
                "mesh_rewritten": False,
                "collision_representation_changed": False,
                "domain_expanded_after_result": False,
                "second_target_retry": False,
                "stop_after_d335": True,
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
                    ("frozen_contract", "d335_frozen_contract.json"),
                    ("scene_contract", "d335_prephysics_scene_contract.json"),
                    ("negative_control", "d335_old_target_negative_control.json"),
                    ("candidate_scan_csv", "d335_candidate_scan.csv"),
                    ("candidate_search", "d335_candidate_search.json"),
                    ("prephysics_gate", "d335_prephysics_gate.json"),
                    ("baseline_trace", "d335_baseline_trace.csv"),
                    ("target_trace", "d335_target_static_trace.csv"),
                    ("raw_distance_trace", "d335_target_raw_distance_trace.csv"),
                    ("rrd", "d335_target_family_repair_trace.rrd"),
                )
            }
            | {
                "summary_json": _rel(args.out_dir / "g0a_d335_target_family_repair_summary.json"),
                "summary_markdown": _rel(args.out_dir / "g0a_d335_target_family_repair_summary.md"),
            },
            "non_goals_respected": [
                "no mesh/collision approximation or cooked-hull target compensation",
                "no target z/wrist/nullspace/gripper-angle change",
                "no domain expansion or physics-driven second target",
                "no waypoint/approach/10-trial/close/grasp/lift",
                "no G0b/RL/PPO/randomization/VLA/RoArm/B200/cube",
            ],
        }
        _json_dump(args.out_dir / "g0a_d335_target_family_repair_summary.json", summary)
        _write_summary_markdown(args.out_dir / "g0a_d335_target_family_repair_summary.md", summary)
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
    del simulation_app  # App lifetime is owned by the launcher; no query pump is used in D335.
    try:
        try:
            summary = _run(args)
            best = summary["candidate_search"].get("best") or {}
            print(
                f"{summary['verdict']}: candidates={summary['candidate_search']['evaluated_count']} "
                f"passes={summary['candidate_search']['passing_count']} "
                f"best_r/t={best.get('radial_offset_mm')}/{best.get('tangent_offset_mm')}mm "
                f"best_clear={best.get('min_raw_clearance_mm')}mm "
                f"physics={summary['physics']['executed']}",
                flush=True,
            )
            return 0 if bool(summary["artifact_contract"]["pass"]) else 1
        except Exception:
            traceback.print_exc()
            args.out_dir.mkdir(parents=True, exist_ok=True)
            abort_payload = {
                "verdict": VERDICT_CONTRACT_FAIL,
                "interpretation": "the current D335 invocation aborted before clean completion",
                "error": traceback.format_exc(),
            }
            _json_dump(args.out_dir / "d335_abort.json", abort_payload)
            fail_path = args.out_dir / "g0a_d335_target_family_repair_summary.json"
            if not fail_path.is_file():
                _json_dump(fail_path, abort_payload)
            return 1
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
