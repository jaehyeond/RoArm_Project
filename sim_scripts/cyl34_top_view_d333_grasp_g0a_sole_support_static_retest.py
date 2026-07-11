#!/usr/bin/env python3
"""D333 sole-support static retest for the cylinder G0a alignment pose.

The only experimental variable relative to D332 is disabling the redundant
global-ground collider. The TapTable remains at the frozen TABLE_Z contract.
The target settle runs only after the support-domain hard gate passes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.viz_debug import draw_frames, log_rerun
from sim_scripts import cyl34_top_view_d332_grasp_g0a_static_collision_discriminator as d332


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d333"
DEFAULT_ROBOT_USD = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"
DEFAULT_URDF = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
D332_SUMMARY = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d332/g0a_d332_static_collision_summary.json"

GROUND_ROOT_PATH = "/World/ground"
GROUND_COLLIDER_PATH = "/World/ground/terrain/GroundPlane/CollisionPlane"
TABLE_BODY_PATH = "/World/envs/env_0/TapTable"
TABLE_COLLIDER_PATH = "/World/envs/env_0/TapTable/geometry/mesh"

FILTER_LABELS = ("support_table", "link4", "link5", "gripper_link")
FILTER_PATHS = (
    "/World/envs/env_.*/TapTable",
    "/World/envs/env_.*/Robot/link4",
    "/World/envs/env_.*/Robot/link5",
    "/World/envs/env_.*/Robot/gripper_link",
)

TABLE_TOP_TOL_M = 1.0e-5
FIRST_STEP_Z_TOL_M = 5.0e-4
TAIL_SUPPORT_GAP_TOL_M = 5.0e-4
ROOT_POSITION_DRIFT_TOL_M = 1.0e-6
ROOT_ROTATION_DRIFT_TOL_RAD = 1.0e-6

# D332 helpers read this module global at call time. Replacing only the labels
# keeps the trace/frame implementation unchanged while making the support body
# explicit in D333 artifacts.
d332.FILTER_LABELS = FILTER_LABELS


def _rel(path: Path) -> str:
    return d332._rel(path)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    d332._json_dump(path, payload)


def _enumerate_collision_prims(stage: Any, root_path: str) -> list[dict[str, Any]]:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    rows: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        api = UsdPhysics.CollisionAPI(prim)
        enabled = api.GetCollisionEnabledAttr().Get()
        rows.append(
            {
                "path": prim.GetPath().pathString,
                "type_name": prim.GetTypeName(),
                "collision_enabled": True if enabled is None else bool(enabled),
            }
        )
    return rows


def _make_runtime_env(args: argparse.Namespace) -> Any:
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import UsdPhysics
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnv

    class D333SoleSupportCylinderEnv(RoArmCubeTap10cmEnv):
        def _setup_scene(self) -> None:
            super()._setup_scene()
            stage = self.scene.stage
            colliders_before = _enumerate_collision_prims(stage, GROUND_ROOT_PATH)
            expected = stage.GetPrimAtPath(GROUND_COLLIDER_PATH)
            errors: list[str] = []
            if [row["path"] for row in colliders_before] != [GROUND_COLLIDER_PATH]:
                errors.append(f"unexpected ground colliders: {colliders_before}")
            if not expected.IsValid() or not expected.HasAPI(UsdPhysics.CollisionAPI):
                errors.append(f"missing expected ground collider: {GROUND_COLLIDER_PATH}")
            else:
                UsdPhysics.CollisionAPI(expected).GetCollisionEnabledAttr().Set(False)
            colliders_after = _enumerate_collision_prims(stage, GROUND_ROOT_PATH)
            if not (
                len(colliders_after) == 1
                and colliders_after[0]["path"] == GROUND_COLLIDER_PATH
                and not colliders_after[0]["collision_enabled"]
            ):
                errors.append(f"ground collider did not disable pre-PLAY: {colliders_after}")
            self._d333_preplay_stage_audit = {
                "expected_ground_collider_path": GROUND_COLLIDER_PATH,
                "ground_colliders_before_disable": colliders_before,
                "ground_colliders_after_disable": colliders_after,
                "errors": errors,
                "pass": not errors,
            }

            sensor_cfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Sponge",
                filter_prim_paths_expr=list(FILTER_PATHS),
                update_period=0.0,
                history_length=1,
                track_pose=True,
                track_contact_points=True,
                max_contact_data_count_per_prim=16,
                force_threshold=0.0,
                debug_vis=False,
            )
            sensor = ContactSensor(sensor_cfg)
            self.scene.sensors["d333_cylinder_contact"] = sensor
            self._d333_contact_sensor = sensor
            # D332 exact-state helper resets this attribute.
            self._d332_contact_sensor = sensor

    env_cfg = d332._configure_runtime_env(args)
    return D333SoleSupportCylinderEnv(cfg=env_cfg)


def _resolved_filter_map(sensor: Any) -> tuple[list[str], dict[str, int]]:
    outer = list(sensor.contact_physx_view.filter_paths)
    if len(outer) == 1 and not isinstance(outer[0], (str, bytes)):
        try:
            raw = [str(item) for item in list(outer[0])]
        except TypeError:
            raw = [str(item) for item in outer]
    else:
        raw = [str(item) for item in outer]
    if len(raw) != len(FILTER_LABELS):
        raise RuntimeError(f"expected four resolved filter paths, got outer={outer!r}, flat={raw!r}")

    suffixes = {
        "support_table": "/TapTable",
        "link4": "/Robot/link4",
        "link5": "/Robot/link5",
        "gripper_link": "/Robot/gripper_link",
    }
    mapping: dict[str, int] = {}
    for idx, path in enumerate(raw):
        for label, suffix in suffixes.items():
            if path.endswith(suffix):
                if label in mapping:
                    raise RuntimeError(f"duplicate resolved filter label {label}: {raw}")
                mapping[label] = idx
    if set(mapping) != set(FILTER_LABELS) or len(set(mapping.values())) != len(FILTER_LABELS):
        raise RuntimeError(f"resolved filter paths are not one-to-one: paths={raw}, mapping={mapping}")
    return raw, mapping


def _sensor_contract(inner: Any) -> tuple[dict[str, Any], dict[str, int]]:
    from pxr import PhysxSchema

    sensor = inner._d333_contact_sensor
    paths, mapping = _resolved_filter_map(sensor)
    data = sensor.data
    expected_shapes = {
        "net_forces_w": [1, 1, 3],
        "net_forces_w_history": [1, 1, 1, 3],
        "force_matrix_w": [1, 1, 4, 3],
        "force_matrix_w_history": [1, 1, 1, 4, 3],
        "contact_pos_w": [1, 1, 4, 3],
        "pos_w": [1, 1, 3],
        "quat_w": [1, 1, 4],
    }
    actual_shapes = {
        name: None if getattr(data, name) is None else list(getattr(data, name).shape)
        for name in expected_shapes
    }
    errors = [
        f"{name}: expected {shape}, got {actual_shapes[name]}"
        for name, shape in expected_shapes.items()
        if actual_shapes[name] != shape
    ]
    stage = inner.scene.stage
    cylinder = stage.GetPrimAtPath("/World/envs/env_0/Sponge")
    reporter = PhysxSchema.PhysxContactReportAPI.Get(stage, cylinder.GetPath())
    reporter_threshold = reporter.GetThresholdAttr().Get()
    rigid = PhysxSchema.PhysxRigidBodyAPI.Get(stage, cylinder.GetPath())
    sleep_threshold = rigid.GetSleepThresholdAttr().Get()
    if reporter_threshold is None or abs(float(reporter_threshold)) > 1.0e-12:
        errors.append(f"reporter_threshold={reporter_threshold}")
    if sleep_threshold is None or abs(float(sleep_threshold)) > 1.0e-12:
        errors.append(f"rigid_body_sleep_threshold={sleep_threshold}")
    if int(sensor.num_instances) != 1 or int(sensor.num_bodies) != 1:
        errors.append(f"instances/bodies={sensor.num_instances}/{sensor.num_bodies}")
    if list(sensor.body_names) != ["Sponge"]:
        errors.append(f"body_names={sensor.body_names}")
    if int(sensor.contact_physx_view.sensor_count) != 1:
        errors.append(f"sensor_count={sensor.contact_physx_view.sensor_count}")
    if int(sensor.contact_physx_view.filter_count) != 4:
        errors.append(f"filter_count={sensor.contact_physx_view.filter_count}")
    return (
        {
            "num_instances": int(sensor.num_instances),
            "num_bodies": int(sensor.num_bodies),
            "body_names": list(sensor.body_names),
            "sensor_count": int(sensor.contact_physx_view.sensor_count),
            "filter_count": int(sensor.contact_physx_view.filter_count),
            "resolved_filter_paths": paths,
            "resolved_filter_index_by_label": mapping,
            "actual_tensor_shapes": actual_shapes,
            "expected_tensor_shapes": expected_shapes,
            "reporter_threshold_n": None if reporter_threshold is None else float(reporter_threshold),
            "rigid_body_sleep_threshold": None if sleep_threshold is None else float(sleep_threshold),
            "instrumentation_side_effect": "activate_contact_sensors authors rigid-body sleep threshold 0",
            "errors": errors,
            "hard_contract_pass": not errors,
        },
        mapping,
    )


def _stage_contract(inner: Any) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = inner.scene.stage
    errors = list(inner._d333_preplay_stage_audit["errors"])
    ground_colliders = _enumerate_collision_prims(stage, GROUND_ROOT_PATH)
    ground_disabled = bool(
        len(ground_colliders) == 1
        and ground_colliders[0]["path"] == GROUND_COLLIDER_PATH
        and not ground_colliders[0]["collision_enabled"]
    )
    if not ground_disabled:
        errors.append(f"ground collider not uniquely disabled after PLAY: {ground_colliders}")

    table_colliders = _enumerate_collision_prims(stage, TABLE_BODY_PATH)
    table_prim = stage.GetPrimAtPath(TABLE_COLLIDER_PATH)
    table_enabled = bool(
        table_prim.IsValid()
        and table_prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(table_prim).GetCollisionEnabledAttr().Get() is not False
    )
    if [row["path"] for row in table_colliders] != [TABLE_COLLIDER_PATH]:
        errors.append(f"unexpected TapTable colliders: {table_colliders}")
    if not table_enabled:
        errors.append(f"TapTable collider is not enabled: {TABLE_COLLIDER_PATH}")

    table_top_z_m = math.nan
    if table_prim.IsValid():
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        aligned = bbox_cache.ComputeWorldBound(table_prim).ComputeAlignedRange()
        table_top_z_m = float(aligned.GetMax()[2])
    table_top_error_m = abs(table_top_z_m - d332.TABLE_Z_M)
    table_top_pass = bool(math.isfinite(table_top_error_m) and table_top_error_m <= TABLE_TOP_TOL_M)
    if not table_top_pass:
        errors.append(
            f"TapTable top mismatch: observed={table_top_z_m}, expected={d332.TABLE_Z_M}, "
            f"tol={TABLE_TOP_TOL_M}"
        )

    fixed_base = bool(inner._robot.is_fixed_base)
    if not fixed_base:
        errors.append("runtime articulation is not fixed-base")
    return {
        "preplay": inner._d333_preplay_stage_audit,
        "postplay_ground_colliders": ground_colliders,
        "ground_collision_disabled_pass": ground_disabled,
        "table_body_path": TABLE_BODY_PATH,
        "table_expected_collider_path": TABLE_COLLIDER_PATH,
        "table_collision_prims": table_colliders,
        "table_collision_enabled_pass": table_enabled,
        "table_top_z_m": table_top_z_m,
        "expected_table_top_z_m": d332.TABLE_Z_M,
        "table_top_error_mm": table_top_error_m * 1000.0,
        "table_top_tolerance_mm": TABLE_TOP_TOL_M * 1000.0,
        "table_top_pass": table_top_pass,
        "robot_is_fixed_base": fixed_base,
        "errors": errors,
        "hard_contract_pass": not errors,
    }


def _root_pose(inner: Any) -> tuple[np.ndarray, np.ndarray]:
    pos = inner._robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    quat = inner._robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
    return pos, quat


def _quat_delta_rad(q0: np.ndarray, q1: np.ndarray) -> float:
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def _state_row(
    inner: Any,
    *,
    phase: str,
    step: int,
    command_target: Any,
    canonical: dict[str, Any],
    object_start_w: np.ndarray,
    root_start_pos_w: np.ndarray,
    root_start_quat_wxyz: np.ndarray,
    contact: dict[str, Any],
) -> dict[str, Any]:
    row = d332._state_row(
        inner,
        phase=phase,
        step=step,
        command_target=command_target,
        canonical=canonical,
        object_start_w=object_start_w,
        contact=contact,
    )
    for frame in row["frames"]:
        frame["name"] = str(frame["name"]).replace("d332_", "d333_")

    root_pos, root_quat = _root_pose(inner)
    root_pos_drift = float(np.linalg.norm(root_pos - root_start_pos_w))
    root_rot_drift = _quat_delta_rad(root_start_quat_wxyz, root_quat)
    object_rot = d332._quat_wxyz_to_rot(np.asarray(row["object_quat_wxyz"], dtype=np.float64))
    axis = object_rot[:, 2]
    vertical_half_extent = (
        0.5 * d332.CYLINDER_HEIGHT_M * abs(float(axis[2]))
        + d332.CYLINDER_RADIUS_M * float(np.linalg.norm(axis[:2]))
    )
    object_bottom_z = float(row["object_pos_local_m"][2]) - vertical_half_extent
    row.update(
        {
            "robot_root_pos_w_m": root_pos.tolist(),
            "robot_root_quat_wxyz": root_quat.tolist(),
            "robot_root_position_drift_m": root_pos_drift,
            "robot_root_rotation_drift_rad": root_rot_drift,
            "object_bottom_z_local_m": object_bottom_z,
            "object_bottom_table_gap_mm": (object_bottom_z - d332.TABLE_Z_M) * 1000.0,
        }
    )
    return row


def _flatten_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    out = d332._flatten_trace_row(row)
    for axis, value in zip("xyz", row["robot_root_pos_w_m"], strict=True):
        out[f"robot_root_pos_w_m_{axis}"] = value
    for axis, value in zip(("w", "x", "y", "z"), row["robot_root_quat_wxyz"], strict=True):
        out[f"robot_root_quat_{axis}"] = value
    out["robot_root_position_drift_m"] = row["robot_root_position_drift_m"]
    out["robot_root_rotation_drift_rad"] = row["robot_root_rotation_drift_rad"]
    out["object_bottom_z_local_m"] = row["object_bottom_z_local_m"]
    out["object_bottom_table_gap_mm"] = row["object_bottom_table_gap_mm"]
    return out


def _write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [_flatten_trace_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def _baseline_statistics(
    rows: list[dict[str, Any]],
    *,
    stage_contract: dict[str, Any],
    sensor_contract: dict[str, Any],
) -> dict[str, Any]:
    tail = rows[-d332.BASELINE_TAIL_STEPS :]
    support_z = [float(row["contact"]["by_filter"]["support_table"]["force_w_n"][2]) for row in tail]
    support_norm = [float(row["contact"]["by_filter"]["support_table"]["force_norm_n"]) for row in tail]
    net_norm = [float(row["contact"]["net_force_norm_n"]) for row in tail]
    robot_max = {
        label: max(float(row["contact"]["by_filter"][label]["force_norm_n"]) for row in rows)
        for label in ("link4", "link5", "gripper_link")
    }
    max_root_pos = max(float(row["robot_root_position_drift_m"]) for row in rows)
    max_root_rot = max(float(row["robot_root_rotation_drift_rad"]) for row in rows)
    max_xy_mm = max(float(row["object_disp_xy_mm"]) for row in rows)
    max_tilt_deg = max(float(row["object_tilt_deg"]) for row in rows)
    max_tail_gap_m = max(abs(float(row["object_bottom_table_gap_mm"])) for row in tail) / 1000.0
    gates = {
        "stage_contract": bool(stage_contract["hard_contract_pass"]),
        "sensor_contract": bool(sensor_contract["hard_contract_pass"]),
        "first_step_abs_z_delta_le_0p5mm": abs(float(rows[0]["object_z_delta_mm"]))
        <= FIRST_STEP_Z_TOL_M * 1000.0,
        "last50_table_fz_median_gt_1n": float(np.median(support_z)) > d332.SUPPORT_POSITIVE_CONTROL_N,
        "last50_bottom_table_gap_abs_max_le_0p5mm": max_tail_gap_m <= TAIL_SUPPORT_GAP_TOL_M,
        "robot_filters_baseline_max_lt_0p1n": max(robot_max.values()) < d332.ROBOT_FORCE_EVENT_N,
        "robot_root_position_drift_le_1e_6m": max_root_pos <= ROOT_POSITION_DRIFT_TOL_M,
        "robot_root_rotation_drift_le_1e_6rad": max_root_rot <= ROOT_ROTATION_DRIFT_TOL_RAD,
        "baseline_object_xy_lt_0p5mm": max_xy_mm < d332.DISTURBANCE_XY_M * 1000.0,
        "baseline_object_tilt_lt_1deg": max_tilt_deg < d332.DISTURBANCE_TILT_DEG,
    }
    return {
        "physics_steps": len(rows),
        "hard_gate": gates,
        "hard_gate_pass": all(gates.values()),
        "first_step_z_delta_mm": float(rows[0]["object_z_delta_mm"]),
        "first_step_z_tolerance_mm": FIRST_STEP_Z_TOL_M * 1000.0,
        "support_table_force_z_last50_median_n": float(np.median(support_z)),
        "support_table_force_norm_last50_median_n": float(np.median(support_norm)),
        "sensor_net_force_norm_last50_median_n": float(np.median(net_norm)),
        "support_positive_threshold_n": d332.SUPPORT_POSITIVE_CONTROL_N,
        "tail_bottom_table_gap_abs_max_mm": max_tail_gap_m * 1000.0,
        "tail_bottom_table_gap_tolerance_mm": TAIL_SUPPORT_GAP_TOL_M * 1000.0,
        "robot_filter_baseline_max_n": robot_max,
        "max_robot_root_position_drift_m": max_root_pos,
        "max_robot_root_rotation_drift_rad": max_root_rot,
        "max_object_disp_xy_mm": max_xy_mm,
        "max_object_tilt_deg": max_tilt_deg,
    }


def _target_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = d332._trace_statistics(rows)
    xy_tilt_step = int(stats["object_disturbance_start_step"])
    center_z_mask = [
        abs(float(row["object_z_delta_mm"])) >= FIRST_STEP_Z_TOL_M * 1000.0 for row in rows
    ]
    support_gap_mask = [
        abs(float(row["object_bottom_table_gap_mm"])) >= TAIL_SUPPORT_GAP_TOL_M * 1000.0
        for row in rows
    ]
    vertical_mask = [z_event or gap_event for z_event, gap_event in zip(center_z_mask, support_gap_mask, strict=True)]
    center_z_step = d332._first_consecutive(center_z_mask)
    support_gap_step = d332._first_consecutive(support_gap_mask)
    vertical_step = d332._first_consecutive(vertical_mask)
    candidates = [value for value in (xy_tilt_step, vertical_step) if value >= 0]
    support_z = [float(row["contact"]["by_filter"]["support_table"]["force_w_n"][2]) for row in rows]
    support_norm = [
        float(row["contact"]["by_filter"]["support_table"]["force_norm_n"]) for row in rows
    ]
    stats.update(
        {
            "xy_tilt_disturbance_start_step": xy_tilt_step,
            "center_z_disturbance_start_step": center_z_step,
            "support_gap_disturbance_start_step": support_gap_step,
            "vertical_support_disturbance_start_step": vertical_step,
            "object_disturbance_start_step": min(candidates) if candidates else -1,
            "max_abs_object_z_delta_mm": max(abs(float(row["object_z_delta_mm"])) for row in rows),
            "final_object_z_delta_mm": float(rows[-1]["object_z_delta_mm"]),
            "max_abs_bottom_table_gap_mm": max(
                abs(float(row["object_bottom_table_gap_mm"])) for row in rows
            ),
            "final_bottom_table_gap_mm": float(rows[-1]["object_bottom_table_gap_mm"]),
            "support_table_force_z_max_n": max(support_z),
            "support_table_force_norm_max_n": max(support_norm),
            "support_table_force_z_last50_median_n": float(
                np.median(support_z[-d332.BASELINE_TAIL_STEPS :])
            ),
        }
    )
    for label in ("link4", "link5", "gripper_link"):
        if float(stats["max_force_n_by_link"][label]) < d332.ROBOT_FORCE_EVENT_N:
            stats["max_force_step_by_link"][label] = -1
    return stats


def _classify(
    support_pass: bool,
    target_runtime_contract_pass: bool,
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    if not support_pass or stats is None:
        return {
            "verdict": "D333_G0A_SCENE_SUPPORT_CONTRACT_FAIL_STOP",
            "interpretation": "sole-support baseline hard gate failed; target settle was not executed",
            "target_settle_executed": False,
        }
    if not target_runtime_contract_pass:
        return {
            "verdict": "D333_G0A_TARGET_RUNTIME_CONTRACT_FAIL_STOP",
            "interpretation": "the target phase violated the fixed-root runtime contract",
            "target_settle_executed": True,
        }
    link5_step = int(stats["first_contact_step_by_link"]["link5"])
    robot_step = int(stats["first_robot_contact_step"])
    disturbance_step = int(stats["object_disturbance_start_step"])
    if robot_step < 0 and disturbance_step < 0:
        verdict = "D333_G0A_D332_STATIC_EVENT_GROUND_ARTIFACT_SUPPORTED"
        interpretation = (
            "the clean final-pose settle has neither a robot-filter event nor object disturbance; "
            "only the D332 step-0 static event is reassigned to the ground confound"
        )
    elif link5_step == 0 and disturbance_step >= 0 and link5_step <= disturbance_step + 1:
        verdict = "D333_G0A_CLEAN_STATIC_LINK5_INTERACTION_SUPPORTED"
        interpretation = (
            "the clean final-pose settle retains an immediate sampled link5 event; the mirror hypothesis is "
            "timing-compatible with object disturbance, but direct live-collider ownership and link5-only "
            "causality remain unresolved"
        )
    else:
        verdict = "D333_G0A_CLEAN_STATIC_BODY_ATTRIBUTION_MIXED_STOP"
        interpretation = (
            "a clean final-pose event or object disturbance remains without an immediate sampled link5 event; "
            "late or absent link5 attribution cannot support body-specific repair"
        )
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "target_settle_executed": True,
        "first_robot_contact_step": robot_step,
        "first_link5_contact_step": link5_step,
        "object_disturbance_start_step": disturbance_step,
        "first_observation_is_post_physics_step": True,
        "contact_onset_left_censored": bool(robot_step == 0),
        "d330_swept_approach_reattributed": False,
    }


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    baseline = summary["runtime"]["baseline"]
    stats = summary["runtime"].get("target_settle")
    lines = [
        "# D333 sole-support static retest",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Stage/support hard contract | `{summary['runtime']['stage_contract']['hard_contract_pass']}` |",
        f"| Sensor structural hard contract | `{summary['runtime']['sensor_contract']['hard_contract_pass']}` |",
        f"| Baseline support hard gate | `{baseline['hard_gate_pass']}` |",
        f"| First-step object z delta | `{baseline['first_step_z_delta_mm']:.6f} mm` |",
        f"| TapTable Fz last-50 median | `{baseline['support_table_force_z_last50_median_n']:.6f} N` |",
        f"| Tail bottom/table max abs gap | `{baseline['tail_bottom_table_gap_abs_max_mm']:.6f} mm` |",
        f"| Baseline max XY / tilt | `{baseline['max_object_disp_xy_mm']:.6f} mm / {baseline['max_object_tilt_deg']:.6f} deg` |",
    ]
    if stats is not None:
        lines.extend(
            [
                f"| First robot contact step | `{stats['first_robot_contact_step']}` |",
                f"| First link5 contact step | `{stats['first_contact_step_by_link']['link5']}` |",
                f"| Object disturbance start step | `{stats['object_disturbance_start_step']}` |",
                f"| Final/max XY displacement | `{stats['final_object_disp_xy_mm']:.6f} / {stats['max_object_disp_xy_mm']:.6f} mm` |",
                f"| Final/max tilt | `{stats['final_object_tilt_deg']:.6f} / {stats['max_object_tilt_deg']:.6f} deg` |",
                f"| Final TCP / commanded TCP error | `{stats['final_tcp_error_mm']:.6f} / {stats['commanded_tcp_error_mm']:.6f} mm` |",
            ]
        )
    lines.extend(["", summary["classification"]["interpretation"] + ".", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_runtime(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(D332_SUMMARY.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    canonical = dict(source["offline"]["canonical"])
    recomputed = d332._canonical_contract()
    if not np.allclose(
        np.asarray(canonical["commanded_joint_rad"]),
        np.asarray(recomputed["commanded_joint_rad"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("D332 frozen canonical joint target changed")
    canonical["offline_witness_cylinder_local_m"] = source["default_physx_mirror_recook"]["query"][
        "nearest_point_cylinder_m"
    ]
    frozen_checks = {
        "seed_same_as_d332": int(args.seed) == int(source["runtime"]["seed"]),
        "robot_usd_sha256_same_as_d332": d332._sha256(args.robot_usd_path)
        == source["offline"]["geometry_contract"]["robot_usd_sha256"],
        "urdf_sha256_same_as_d332": d332._sha256(args.urdf_path)
        == source["offline"]["geometry_contract"]["urdf_sha256"],
        "num_envs_same_as_d332": int(source["runtime"]["num_envs"]) == 1,
        "physics_dt_same_as_d332": math.isclose(
            float(source["runtime"]["physics_dt_s"]), d332.PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "baseline_steps_same_as_d332": int(source["runtime"]["baseline_physics_steps"])
        == d332.BASELINE_PHYSICS_STEPS,
        "target_steps_same_as_d332": int(source["runtime"]["target_settle_physics_steps"])
        == d332.TARGET_SETTLE_PHYSICS_STEPS,
        "object_center_same_as_d332": bool(
            np.allclose(
                np.asarray(source["offline"]["cylinder_contract"]["center_local_m"]),
                d332.OBJECT_CENTER_LOCAL_M,
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "object_radius_same_as_d332": math.isclose(
            float(source["offline"]["cylinder_contract"]["radius_m"]),
            d332.CYLINDER_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "object_height_same_as_d332": math.isclose(
            float(source["offline"]["cylinder_contract"]["height_m"]),
            d332.CYLINDER_HEIGHT_M,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
    }
    frozen_contract = {
        "checks": frozen_checks,
        "pass": all(frozen_checks.values()),
        "seed": int(args.seed),
        "robot_usd": _rel(args.robot_usd_path),
        "robot_usd_sha256": d332._sha256(args.robot_usd_path),
        "urdf": _rel(args.urdf_path),
        "urdf_sha256": d332._sha256(args.urdf_path),
        "mass_kg": d332.OBJECT_MASS_KG,
        "static_friction": d332.STATIC_FRICTION,
        "dynamic_friction": d332.DYNAMIC_FRICTION,
    }
    frozen_contract_path = args.out_dir / "d333_frozen_invariant_contract.json"
    _json_dump(frozen_contract_path, frozen_contract)
    if not frozen_contract["pass"]:
        raise RuntimeError(f"D333 frozen invariant contract failed; see {frozen_contract_path}")

    d332._write_canonical_csv(args.out_dir / "d333_canonical_joint_targets.csv", canonical)
    versions = d332._runtime_versions()
    inner = _make_runtime_env(args)
    try:
        inner.reset(seed=int(args.seed))
        stage_contract = _stage_contract(inner)
        sensor_contract, filter_map = _sensor_contract(inner)
        precheck_path = args.out_dir / "d333_prebaseline_contract.json"
        _json_dump(
            precheck_path,
            {
                "artifact": "D333_PREBASELINE_CONTRACT",
                "stage_contract": stage_contract,
                "sensor_contract": sensor_contract,
            },
        )
        if not stage_contract["hard_contract_pass"] or not sensor_contract["hard_contract_pass"]:
            raise RuntimeError(f"D333 prebaseline contract failed; see {precheck_path}")

        q_home = np.radians(np.asarray(d332.HOME_DEG, dtype=np.float64))
        q_home[5] = 0.0
        home_target = d332._write_exact_state(inner, q_home, d332.OBJECT_CENTER_LOCAL_M)
        origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
        baseline_start_w = origin + d332.OBJECT_CENTER_LOCAL_M
        baseline_root_pos, baseline_root_quat = _root_pose(inner)
        baseline_rows: list[dict[str, Any]] = []
        for step in range(d332.BASELINE_PHYSICS_STEPS):
            d332._physics_step(inner)
            contact = d332._contact_state(inner._d333_contact_sensor, filter_map)
            baseline_rows.append(
                _state_row(
                    inner,
                    phase="sole_support_robot_free_baseline",
                    step=step,
                    command_target=home_target,
                    canonical=canonical,
                    object_start_w=baseline_start_w,
                    root_start_pos_w=baseline_root_pos,
                    root_start_quat_wxyz=baseline_root_quat,
                    contact=contact,
                )
            )

        baseline_stats = _baseline_statistics(
            baseline_rows,
            stage_contract=stage_contract,
            sensor_contract=sensor_contract,
        )
        baseline_csv = args.out_dir / "d333_contact_baseline_trace.csv"
        _write_trace_csv(baseline_csv, baseline_rows)

        target_rows: list[dict[str, Any]] = []
        stats: dict[str, Any] | None = None
        target_csv: Path | None = None
        if baseline_stats["hard_gate_pass"]:
            q_target = np.asarray(canonical["commanded_joint_rad"], dtype=np.float64)
            command_target = d332._write_exact_state(inner, q_target, d332.OBJECT_CENTER_LOCAL_M)
            object_start_w = origin + d332.OBJECT_CENTER_LOCAL_M
            target_root_pos, target_root_quat = _root_pose(inner)
            for step in range(d332.TARGET_SETTLE_PHYSICS_STEPS):
                d332._physics_step(inner)
                contact = d332._contact_state(inner._d333_contact_sensor, filter_map)
                target_rows.append(
                    _state_row(
                        inner,
                        phase="sole_support_canonical_target_settle",
                        step=step,
                        command_target=command_target,
                        canonical=canonical,
                        object_start_w=object_start_w,
                        root_start_pos_w=target_root_pos,
                        root_start_quat_wxyz=target_root_quat,
                        contact=contact,
                    )
                )
            target_csv = args.out_dir / "d333_teleport_settle_trace.csv"
            _write_trace_csv(target_csv, target_rows)
            stats = _target_statistics(target_rows)
            stats["max_robot_root_position_drift_m"] = max(
                float(row["robot_root_position_drift_m"]) for row in target_rows
            )
            stats["max_robot_root_rotation_drift_rad"] = max(
                float(row["robot_root_rotation_drift_rad"]) for row in target_rows
            )

        target_runtime_contract_pass = bool(
            stats is None
            or (
                float(stats["max_robot_root_position_drift_m"]) <= ROOT_POSITION_DRIFT_TOL_M
                and float(stats["max_robot_root_rotation_drift_rad"]) <= ROOT_ROTATION_DRIFT_TOL_RAD
            )
        )
        classification = _classify(
            bool(baseline_stats["hard_gate_pass"]),
            target_runtime_contract_pass,
            stats,
        )
        runtime_core_path = args.out_dir / "d333_runtime_core_provisional.json"
        _json_dump(
            runtime_core_path,
            {
                "artifact": "D333_RUNTIME_CORE_PROVISIONAL",
                "new_variable": ["support_domain_global_ground_collision_disabled"],
                "source_summary": _rel(D332_SUMMARY),
                "source_summary_sha256": d332._sha256(D332_SUMMARY),
                "stage_contract": stage_contract,
                "sensor_contract": sensor_contract,
                "baseline": baseline_stats,
                "target_settle": stats,
                "classification_before_visualization_gate": classification,
                "baseline_trace_csv": _rel(baseline_csv),
                "target_settle_trace_csv": None if target_csv is None else _rel(target_csv),
            },
        )
        snapshots: list[str] = []
        snapshot_paths: list[Path] = []
        baseline_png = args.out_dir / "d333_baseline_final.png"
        snapshot_paths.append(baseline_png)
        snapshots.append(
            d332._write_runtime_snapshot(
                baseline_png,
                baseline_rows[-1],
                "D333 sole-support baseline final",
            )
        )
        trace_for_rrd = baseline_rows
        marker_frames = baseline_rows[-1]["frames"]
        if target_rows:
            onset = [
                value
                for value in (int(stats["first_robot_contact_step"]), int(stats["object_disturbance_start_step"]))
                if value >= 0
            ]
            event_step = min(onset) if onset else 0
            event_png = args.out_dir / "d333_target_first_event.png"
            final_png = args.out_dir / "d333_target_final.png"
            snapshot_paths.extend([event_png, final_png])
            snapshots.append(
                d332._write_runtime_snapshot(
                    event_png,
                    target_rows[event_step],
                    f"D333 clean target first event candidate (step {event_step})",
                )
            )
            snapshots.append(
                d332._write_runtime_snapshot(
                    final_png,
                    target_rows[-1],
                    f"D333 clean target final (step {len(target_rows) - 1})",
                )
            )
            trace_for_rrd = target_rows
            marker_frames = target_rows[-1]["frames"]

        marker_status = draw_frames(marker_frames, prim_path="/World/D333SoleSupportFrames")
        rrd_path = args.out_dir / "d333_sole_support_static_trace_v2.rrd"
        rrd_status = log_rerun(
            rrd_path,
            frames=marker_frames,
            joint_state={
                "label": "d333_sole_support_static_retest",
                "physics_steps": len(trace_for_rrd),
                "physics_dt_s": d332.PHYSICS_DT_S,
                "object": "cylinder_d34_h90",
            },
            joint_trace=trace_for_rrd,
            urdf_path=args.urdf_path,
            live_viewer=False,
            app_id="roarm_g0a_d333_sole_support_static_retest",
        )
        if bool(rrd_status.get("ok")):
            rrd_status["nonzero_file"] = bool(rrd_path.is_file() and rrd_path.stat().st_size > 0)

        artifact_checks = {
            "snapshot_count_between_1_and_3": 1 <= len(snapshot_paths) <= 3,
            "snapshots_exist_and_nonzero": all(path.is_file() and path.stat().st_size > 0 for path in snapshot_paths),
            "marker_status_ok": bool(marker_status.get("ok")),
            "rrd_status_ok": bool(rrd_status.get("ok")),
            "rrd_nonzero_file": bool(rrd_status.get("nonzero_file")),
        }
        artifact_contract = {"checks": artifact_checks, "pass": all(artifact_checks.values())}
        scientific_verdict = classification["verdict"]
        final_verdict = (
            scientific_verdict
            if artifact_contract["pass"]
            else "D333_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP"
        )

        sensor_contract["support_filter_positive_control_valid"] = bool(
            baseline_stats["hard_gate"]["last50_table_fz_median_gt_1n"]
            and baseline_stats["hard_gate"]["sensor_contract"]
        )
        if stats is not None:
            sensor_contract["single_sample_threshold_crossing_by_robot_filter"] = {
                label: float(stats["max_force_n_by_link"][label]) >= d332.ROBOT_FORCE_EVENT_N
                for label in ("link4", "link5", "gripper_link")
            }

        summary = {
            "verdict": final_verdict,
            "active_case": "G0a cylinder D34xH90 alignment-only sole-support static retest",
            "new_variable": ["support_domain_global_ground_collision_disabled"],
            "frozen_d332_contract": {
                "source_summary": _rel(D332_SUMMARY),
                "source_summary_sha256": d332._sha256(D332_SUMMARY),
                "source_verdict": source["verdict"],
                "mirror_signed_distance_mm": source["default_physx_mirror_recook"]["query"][
                    "signed_distance_mm"
                ],
                "canonical": canonical,
                "invariant_hard_contract": frozen_contract,
            },
            "runtime": {
                "seed": int(args.seed),
                "num_envs": 1,
                "physics_dt_s": float(inner.physics_dt),
                "baseline_physics_steps": d332.BASELINE_PHYSICS_STEPS,
                "target_settle_physics_steps_requested": d332.TARGET_SETTLE_PHYSICS_STEPS,
                "target_settle_physics_steps_executed": len(target_rows),
                "target_runtime_contract_pass": target_runtime_contract_pass,
                "stage_contract": stage_contract,
                "sensor_contract": sensor_contract,
                "baseline": baseline_stats,
                "target_settle": stats,
                "diagnostic_thresholds": {
                    "first_step_abs_z_delta_mm": FIRST_STEP_Z_TOL_M * 1000.0,
                    "tail_bottom_table_gap_abs_mm": TAIL_SUPPORT_GAP_TOL_M * 1000.0,
                    "support_table_fz_n": d332.SUPPORT_POSITIVE_CONTROL_N,
                    "robot_force_event_n": d332.ROBOT_FORCE_EVENT_N,
                    "object_xy_disturbance_mm": d332.DISTURBANCE_XY_M * 1000.0,
                    "object_tilt_disturbance_deg": d332.DISTURBANCE_TILT_DEG,
                    "consecutive_steps": d332.CONSECUTIVE_EVENT_STEPS,
                },
            },
            "classification": classification,
            "scientific_verdict_before_artifact_gate": scientific_verdict,
            "artifact_contract": artifact_contract,
            "outcome_guards": {
                "g0a_pass": False,
                "alignment_ladder_promoted": False,
                "collision_repair_authorized": False,
                "d330_swept_approach_reattributed": False,
                "stop_after_d333": True,
            },
            "visualization": {
                "snapshots": snapshots,
                "snapshot_count": len(snapshots),
                "marker_status": marker_status,
                "rrd_status": rrd_status,
            },
            "artifacts": {
                "canonical_joint_csv": _rel(args.out_dir / "d333_canonical_joint_targets.csv"),
                "baseline_trace_csv": _rel(baseline_csv),
                "target_settle_trace_csv": None if target_csv is None else _rel(target_csv),
                "summary_json": _rel(args.out_dir / "g0a_d333_sole_support_static_summary.json"),
                "summary_markdown": _rel(args.out_dir / "g0a_d333_sole_support_static_summary.md"),
                "runtime_core_provisional_json": _rel(runtime_core_path),
                "prebaseline_contract_json": _rel(precheck_path),
                "frozen_invariant_contract_json": _rel(frozen_contract_path),
                "rrd": _rel(rrd_path),
            },
            "environment": versions,
            "non_goals_respected": [
                "no collision mesh rewrite or live-collider ownership scan",
                "no target/gate/offset/standoff tuning",
                "no waypoint, approach path, wrist null-space scan, or 10-trial gate",
                "no gripper close/grasp/lift/G0b",
                "no render beyond at most three diagnostic PNGs; no video",
                "no RL/PPO/randomization/VLA/RoArm/B200/cube",
            ],
        }
        summary_json = args.out_dir / "g0a_d333_sole_support_static_summary.json"
        summary_md = args.out_dir / "g0a_d333_sole_support_static_summary.md"
        _json_dump(summary_json, summary)
        _write_summary_markdown(summary_md, summary)
        return summary
    finally:
        inner.close()


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_ROBOT_USD)
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
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
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        try:
            summary = _run_runtime(args)
            stats = summary["runtime"].get("target_settle")
            if stats is None:
                detail = "target_not_run"
            else:
                detail = (
                    f"contact_step={stats['first_robot_contact_step']} "
                    f"disturbance_step={stats['object_disturbance_start_step']} "
                    f"final_disp={stats['final_object_disp_xy_mm']:.6f}mm"
                )
            print(f"{summary['verdict']}: {detail}", flush=True)
            return 0 if bool(summary["artifact_contract"]["pass"]) else 1
        except Exception:
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
