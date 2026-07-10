#!/usr/bin/env python3
"""D328 G0a runtime-stall discriminator: collision path vs drive semantics.

This stays inside G0a.  It keeps the D325 criterion and D327 2mm alignment
standoff fixed, first removes the cube as a decision experiment, then applies
exactly one branch repair based on that result.
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

import sim_scripts.cube10cm_top_view_d327_grasp_g0a_standoff_execution_probe as d327
from roarm_rl.viz_debug import draw_frames, log_rerun, snapshot_frame_plot
from sim_scripts.cube10cm_top_view_d323_grasp_g0a_frame_repair_probe import (
    _fk_runtime_tcp,
    _quat_wxyz_to_rot,
    _solve_runtime_ik,
)


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d328"
ARM_JOINT_NAMES = d327.ARM_JOINT_NAMES
ALL_JOINT_NAMES = d327.ALL_JOINT_NAMES


def _rel(path: Path) -> str:
    return d327._rel(path)


def _joint_dict(values: np.ndarray) -> dict[str, float]:
    return d327._joint_dict(values)


def _failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tcp_pose": sum(1 for row in rows if not row["pass_tcp_pose"]),
        "jaw_tangent": sum(1 for row in rows if not row["pass_jaw_tangent"]),
        "fixed_jaw_gap": sum(1 for row in rows if not row["pass_fixed_jaw_gap"]),
        "fixed_jaw_penetration": sum(1 for row in rows if not row["pass_no_penetration"]),
        "contact_height": sum(1 for row in rows if not row["pass_contact_height"]),
        "cube_displacement": sum(1 for row in rows if not row["pass_cube_displacement"]),
    }


def _drive_audit(inner: Any, env_id: int = 0) -> dict[str, Any]:
    data = inner._robot.data
    arm_ids = inner._bc_arm_joint_ids
    names = [inner._robot.joint_names[idx] for idx in arm_ids]

    def _take(attr: str) -> list[float] | None:
        val = getattr(data, attr, None)
        if val is None:
            return None
        return [float(x) for x in val[env_id, arm_ids].detach().cpu().numpy().astype(np.float64).tolist()]

    return {
        "joint_names": names,
        "joint_stiffness": _take("joint_stiffness"),
        "joint_damping": _take("joint_damping"),
        "joint_effort_limits": _take("joint_effort_limits"),
        "joint_velocity_limits": _take("joint_velocity_limits"),
        "default_joint_stiffness": _take("default_joint_stiffness"),
        "default_joint_damping": _take("default_joint_damping"),
    }


def _try_make_contact_sensor() -> tuple[Any | None, dict[str, Any]]:
    """Best-effort robot net-force sensor.

    The active env does not configure contact sensors.  D328 attempts a runtime
    ContactSensor for evidence only; the Step 1 branch decision never depends on
    this optional path.
    """
    try:
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=1,
            update_period=0.0,
            force_threshold=0.0,
        )
        sensor = ContactSensor(cfg)
        return sensor, {"ok": True, "mode": "robot_net_forces_w", "prim_path": cfg.prim_path}
    except Exception as exc:  # pragma: no cover - depends on Isaac runtime
        return None, {"ok": False, "mode": "robot_net_forces_w", "error": repr(exc)}


def _contact_force_row(sensor: Any | None) -> dict[str, Any]:
    if sensor is None:
        return {"available": False}
    try:
        sensor.update(0.0)
        forces = sensor.data.net_forces_w
        if forces is None:
            return {"available": False, "reason": "net_forces_w is None"}
        arr = forces[0].detach().cpu().numpy().astype(np.float64)
        norms = np.linalg.norm(arr, axis=-1)
        return {
            "available": True,
            "max_force_n": float(np.max(norms)),
            "argmax_body_index": int(np.argmax(norms)),
            "sum_force_n": float(np.sum(norms)),
        }
    except Exception as exc:  # pragma: no cover - depends on Isaac runtime
        return {"available": False, "error": repr(exc)}


def _move_cube_out(inner: Any, origins: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from roarm_rl.roarm_stack_env import TABLE_Z

    env_ids = torch.arange(inner.num_envs, device=inner.device, dtype=torch.long)
    local = torch.tensor(
        [float(args.cube_removed_x), float(args.cube_removed_y), TABLE_Z + 0.5 * float(args.cube_size_m)],
        device=inner.device,
        dtype=torch.float32,
    )
    pose = torch.zeros((inner.num_envs, 7), device=inner.device, dtype=torch.float32)
    pose[:, 0:3] = origins + local.unsqueeze(0)
    pose[:, 3] = 1.0
    inner._sponge.write_root_pose_to_sim(pose, env_ids=env_ids)
    inner._sponge.write_root_velocity_to_sim(torch.zeros((inner.num_envs, 6), device=inner.device), env_ids=env_ids)
    inner._compute_intermediate_values()
    return {
        "removed_local_xyz_m": [float(args.cube_removed_x), float(args.cube_removed_y), float(local[2].detach().cpu())],
        "note": "target_info remains based on the original cube pose before removal",
    }


def _trace_stats(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace_rows:
        return {"trace_steps": 0}
    tcp = np.asarray([float(r["tcp_pose_error_mm"]) for r in trace_rows], dtype=np.float64)
    joint_err = np.asarray([float(r["arm_joint_err_max_rad"]) for r in trace_rows], dtype=np.float64)
    force = np.asarray([float(r.get("contact_force", {}).get("max_force_n", 0.0)) for r in trace_rows], dtype=np.float64)
    torque_sat = []
    for row in trace_rows:
        torque = row.get("torque", {})
        if isinstance(torque, dict) and "saturation_rate" in torque:
            torque_sat.append(float(torque["saturation_rate"]))
    return {
        "trace_steps": len(trace_rows),
        "first_tcp_error_mm": float(tcp[0]),
        "mid_tcp_error_mm": float(tcp[len(tcp) // 2]),
        "final_tcp_error_mm": float(tcp[-1]),
        "min_tcp_error_mm": float(np.min(tcp)),
        "final_joint_err_rad": float(joint_err[-1]),
        "max_joint_err_rad": float(np.max(joint_err)),
        "max_contact_force_n": float(np.max(force)) if len(force) else 0.0,
        "first_contact_force_step": int(next((i for i, v in enumerate(force) if v > 1.0), -1)),
        "torque_saturation_rate_max": float(max(torque_sat)) if torque_sat else None,
        "torque_saturation_rate_final": float(torque_sat[-1]) if torque_sat else None,
    }


def _waypoint_sequence(info: dict[str, Any], args: argparse.Namespace, mode: str) -> list[np.ndarray]:
    target = np.asarray(info["target_tcp"], dtype=np.float64)
    radial = np.asarray(info["radial_axis"], dtype=np.float64)
    if mode == "d327_radial":
        return [target - radial * float(args.pre_clearance_m), target]
    if mode == "far_side_slide":
        return [target - radial * 0.090, target - radial * 0.025, target]
    if mode == "high_corridor_drop":
        dz = np.asarray([0.0, 0.0, float(args.collision_repair_raise_m)], dtype=np.float64)
        return [target - radial * 0.070 + dz, target + dz, target]
    raise ValueError(f"unsupported waypoint mode {mode!r}")


def _candidate_path_table(cube_start_local: np.ndarray, target_info: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cube = np.asarray(cube_start_local[0], dtype=np.float64)
    half = float(args.cube_size_m) * 0.5
    cube_top = float(cube[2] + half)
    for mode in ("d327_radial", "far_side_slide", "high_corridor_drop"):
        waypoints = _waypoint_sequence(target_info[0], args, mode)
        ik_ok = True
        max_ik_err_mm = 0.0
        q_seed = np.asarray(d327.HOME_DEG, dtype=np.float64)
        min_top_clearance = math.inf
        approach_waypoints = waypoints[:-1] or waypoints
        for wp_index, wp in enumerate(waypoints):
            result = _solve_runtime_ik(wp, q_seed, max_iter=120, pos_tol_mm=1.0)
            ik_ok = ik_ok and bool(result["converged"])
            q_seed = np.asarray(result["q_deg"], dtype=np.float64)
            tcp, _link5, _rot = _fk_runtime_tcp(q_seed)
            max_ik_err_mm = max(max_ik_err_mm, float(np.linalg.norm(tcp - wp) * 1000.0))
            if wp_index >= len(approach_waypoints):
                continue
            inside_xy = abs(float(tcp[0] - cube[0])) <= half and abs(float(tcp[1] - cube[1])) <= half
            if inside_xy:
                min_top_clearance = min(min_top_clearance, float(tcp[2] - cube_top))
        if not math.isfinite(min_top_clearance):
            min_top_clearance = float(args.collision_repair_raise_m)
        rows.append(
            {
                "mode": mode,
                "waypoint_count": len(waypoints),
                "ik_all_converged": bool(ik_ok),
                "max_ik_err_mm": max_ik_err_mm,
                "approach_waypoint_count_for_clearance": len(approach_waypoints),
                "min_approach_tcp_over_cube_top_clearance_mm": float(min_top_clearance * 1000.0),
                "clearance_note": "final waypoint is excluded because D327 teleport-static final pose already passes no-penetration; this table ranks approach corridor collision risk",
            }
        )
    return rows


def _select_collision_repair(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in candidates if bool(r["ik_all_converged"])]
    if not valid:
        valid = candidates
    selected = max(valid, key=lambda r: (float(r["min_approach_tcp_over_cube_top_clearance_mm"]), -float(r["max_ik_err_mm"])))
    return {
        "type": "waypoint_path_repair",
        "selected_mode": selected["mode"],
        "selection_rule": "max top-clearance among IK-feasible candidates",
        "selected_candidate": selected,
        "all_candidates": candidates,
    }


def _run_motion(
    args: argparse.Namespace,
    *,
    num_envs: int,
    label: str,
    trace: bool,
    cube_removed: bool = False,
    direct_write_through: bool = False,
    waypoint_mode: str = "d327_radial",
) -> dict[str, Any]:
    import torch

    env, inner, zero = d327._make_env(args, num_envs)
    origins, cube_start_w, cube_start_local, _cube_start_w_np, _origins_np = d327._reset_env(inner, env, zero)
    target_info = d327._target_info(cube_start_local, args)
    cube_removed_info: dict[str, Any] | None = None
    if cube_removed:
        cube_removed_info = _move_cube_out(inner, origins, args)

    sensor, sensor_status = _try_make_contact_sensor() if trace else (None, {"ok": False, "skipped": True})
    joint_targets_full = inner._robot.data.joint_pos.detach().clone()
    q_targets_kin = np.zeros((inner.num_envs, 6), dtype=np.float64)
    q_targets_kin[:, :5] = joint_targets_full[:, inner._bc_arm_joint_ids].detach().cpu().numpy().astype(np.float64)
    q_targets_kin[:, 5] = joint_targets_full[:, inner.gripper_joint_idx].detach().cpu().numpy().astype(np.float64)
    ik_failure_counts = np.zeros(inner.num_envs, dtype=np.int64)
    final_target_arm = q_targets_kin[:, :5].copy()
    trace_rows: list[dict[str, Any]] = []

    total_steps = int(args.approach_steps) + int(args.hold_steps)
    with torch.inference_mode():
        for step in range(total_steps):
            phase = "approach" if step < int(args.approach_steps) else "hold"
            alpha_global = min(1.0, (step + 1) / float(max(1, int(args.approach_steps))))
            for idx, info in enumerate(target_info):
                waypoints = _waypoint_sequence(info, args, waypoint_mode)
                if len(waypoints) == 2:
                    target_tcp = waypoints[0] + alpha_global * (waypoints[-1] - waypoints[0])
                else:
                    seg_alpha = alpha_global * (len(waypoints) - 1)
                    seg_idx = min(len(waypoints) - 2, int(math.floor(seg_alpha)))
                    local_alpha = seg_alpha - float(seg_idx)
                    target_tcp = waypoints[seg_idx] + local_alpha * (waypoints[seg_idx + 1] - waypoints[seg_idx])
                if phase == "hold":
                    target_tcp = waypoints[-1]
                result = _solve_runtime_ik(
                    target_tcp,
                    np.degrees(q_targets_kin[idx]),
                    target_x_axis=None,
                    target_z_axis=None,
                    max_iter=120,
                    pos_tol_mm=1.0,
                )
                if not bool(result["converged"]):
                    ik_failure_counts[idx] += 1
                q_targets_kin[idx] = np.radians(np.asarray(result["q_deg"], dtype=np.float64))

            targets_t = joint_targets_full.detach().clone()
            targets_t[:, inner._bc_arm_joint_ids] = torch.tensor(q_targets_kin[:, :5], device=inner.device, dtype=torch.float32)
            targets_t[:, inner.gripper_joint_idx] = 0.0
            joint_targets_full = targets_t.detach().clone()
            inner._external_joint_targets_override = targets_t
            if direct_write_through:
                inner.robot_dof_targets[:] = torch.clamp(targets_t, inner.robot_dof_lower_limits, inner.robot_dof_upper_limits)
                inner._robot.set_joint_position_target(inner.robot_dof_targets)
                inner._robot.write_data_to_sim()
            env.step(zero)
            inner._compute_intermediate_values()
            final_target_arm = q_targets_kin[:, :5].copy()

            if trace:
                body_quat_np = inner._robot.data.body_quat_w.detach().cpu().numpy()
                tcp_local = (inner._tcp_pos_w - origins).detach().cpu().numpy()
                actual_all = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
                commanded_all = targets_t[0].detach().cpu().numpy().astype(np.float64)
                actual_arm = actual_all[inner._bc_arm_joint_ids]
                target_arm = commanded_all[inner._bc_arm_joint_ids]
                info = target_info[0]
                target_tcp = np.asarray(info["target_tcp"], dtype=np.float64)
                tangent = d327._unit(np.asarray(info["target_x_axis"], dtype=np.float64))
                link5_rot = _quat_wxyz_to_rot(body_quat_np[0, inner.link5_idx])
                row, _fixed, _contact, frames = d327._evaluate_alignment(
                    trial=1,
                    cube_center=cube_start_local[0],
                    cube_size_m=float(args.cube_size_m),
                    target_tcp=target_tcp,
                    tangent=tangent,
                    actual_tcp=tcp_local[0],
                    link5_rot=link5_rot,
                    cube_start_w=cube_start_w.detach().cpu().numpy()[0],
                    cube_final_w=inner._sponge_pos_w[0].detach().cpu().numpy(),
                    target_arm=target_arm,
                    actual_arm=actual_arm,
                    ik_failure_steps=int(ik_failure_counts[0]),
                )
                trace_rows.append(
                    {
                        "step": int(step),
                        "phase": phase,
                        "alpha": float(alpha_global),
                        "actual_joint_rad_by_name": _joint_dict(actual_all),
                        "commanded_joint_rad_by_name": _joint_dict(commanded_all),
                        "tcp_pose_error_mm": row["tcp_pose_error_mm"],
                        "commanded_tcp_pose_error_mm": row["commanded_tcp_pose_error_mm"],
                        "arm_joint_err_max_rad": row["arm_joint_err_max_rad"],
                        "torque": d327._torque_saturation(inner, 0),
                        "contact_force": _contact_force_row(sensor),
                        "frames": frames,
                    }
                )

    rows, frame_sets = d327._state_eval_rows(
        inner,
        origins,
        cube_start_w,
        cube_start_local,
        target_info,
        final_target_arm,
        ik_failure_counts,
        args,
    )
    marker_status = draw_frames(frame_sets[0], prim_path=f"/World/D328{label}Frames") if frame_sets else {}
    snapshots: list[dict[str, Any]] = []
    trial_indices = [0] if num_envs == 1 else [0, 4, 9]
    for trial_idx in trial_indices:
        if trial_idx < len(rows):
            path = args.out_dir / f"d328_{label}_trial_{trial_idx + 1:02d}_snapshot.png"
            d327._write_snapshot(
                path,
                cube_center=cube_start_local[trial_idx],
                cube_size_m=float(args.cube_size_m),
                frames=frame_sets[trial_idx],
                row=rows[trial_idx],
                title=f"D328 {label} trial {trial_idx + 1}",
            )
            snapshots.append({"trial": int(trial_idx + 1), "path": _rel(path)})

    rrd_path = ""
    rrd_status: dict[str, Any] = {"ok": False, "skipped": not trace}
    if trace and trace_rows:
        rrd_file = args.out_dir / f"d328_{label}_trace_v2.rrd"
        rrd_status = log_rerun(
            rrd_file,
            frames=frame_sets[0],
            joint_state={
                "label": label,
                "cube_removed": bool(cube_removed),
                "direct_write_through": bool(direct_write_through),
                "waypoint_mode": waypoint_mode,
            },
            joint_trace=trace_rows,
            cube={"center": cube_start_local[0].tolist(), "size": float(args.cube_size_m)},
            urdf_path=args.urdf_path,
            live_viewer=bool(args.live_viewer),
            app_id=f"roarm_g0a_d328_{label}",
        )
        if rrd_status.get("ok"):
            rrd_path = _rel(rrd_file)

    result = {
        "label": label,
        "num_envs": int(num_envs),
        "cube_removed": bool(cube_removed),
        "cube_removed_info": cube_removed_info,
        "direct_write_through": bool(direct_write_through),
        "waypoint_mode": waypoint_mode,
        "pass_all_count": int(sum(1 for row in rows if row["pass_all"])),
        "failure_counts": _failure_counts(rows),
        "rows": rows,
        "trace_stats": _trace_stats(trace_rows),
        "contact_sensor_status": sensor_status,
        "drive_audit": _drive_audit(inner, 0),
        "snapshots": snapshots,
        "marker_status": marker_status,
        "rrd_path": rrd_path,
        "rrd_status": rrd_status,
    }
    env.close()
    return result


def _write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "g0a_d328_collision_vs_drive_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    final_rows = summary.get("final_retest", {}).get("rows", [])
    if final_rows:
        csv_path = out_dir / "g0a_d328_final_retest_trials.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
            writer.writeheader()
            writer.writerows(final_rows)
        summary["final_retest"]["trial_csv"] = _rel(csv_path)
        (out_dir / "g0a_d328_collision_vs_drive_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    lines = [
        "# D328 G0a Collision-vs-Drive Probe",
        "",
        "이번 case의 ladder 신규 변수: `[]` — D328 is a G0a runtime-stall diagnosis and one branch repair.",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- branch: `{summary.get('branch', '')}`",
        f"- selected repair: `{summary.get('selected_repair', {}).get('type', '')}`",
        f"- final pass_all: `{summary.get('final_retest', {}).get('pass_all_count', 0)}/10`",
        "",
        "## Step 1 Decision Experiment",
        "",
        f"- prediction A: `{summary['step1_cube_removed']['prediction_a']}`",
        f"- prediction B: `{summary['step1_cube_removed']['prediction_b']}`",
        f"- final TCP error: `{summary['step1_cube_removed']['row']['tcp_pose_error_mm']:.3f}mm`",
        f"- judgement: `{summary['step1_cube_removed']['judgement']}`",
        "",
        "## Evidence Trial",
        "",
        f"- max contact force: `{summary['cube_present_evidence']['trace_stats'].get('max_contact_force_n', 0.0):.3f}N`",
        f"- torque saturation max: `{summary['cube_present_evidence']['trace_stats'].get('torque_saturation_rate_max')}`",
        f"- contact sensor status: `{summary['cube_present_evidence']['contact_sensor_status']}`",
    ]
    if final_rows:
        lines.extend(
            [
                "",
                "## Final 10-Trial Table",
                "",
                "| trial | pos mm | cmd pos mm | tangent deg | gap mm | top clearance mm | cube disp mm | pass |",
                "|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in final_rows:
            lines.append(
                "| {trial} | {tcp_pose_error_mm:.3f} | {commanded_tcp_pose_error_mm:.3f} | "
                "{jaw_tangent_error_deg:.3f} | {fixed_jaw_face_gap_mm:.3f} | "
                "{contact_point_below_top_mm:.3f} | {cube_disp_xy_mm:.3f} | {pass_all} |".format(**row)
            )
    lines.extend(["", "## Artifacts", ""])
    for key in ("step1_cube_removed", "cube_present_evidence", "final_retest"):
        item = summary.get(key, {})
        for snap in item.get("snapshots", []):
            lines.append(f"- {key} trial {snap['trial']} snapshot: `{snap['path']}`")
        if item.get("rrd_path"):
            lines.append(f"- {key} rrd: `{item['rrd_path']}`")
    (out_dir / "g0a_d328_collision_vs_drive_summary.md").write_text("\n".join(lines) + "\n")


def _stage_path(out_dir: Path, stage: str) -> Path:
    return out_dir / f"stage_{stage}.json"


def _write_stage(out_dir: Path, stage: str, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _stage_path(out_dir, stage).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_stage(out_dir: Path, stage: str) -> dict[str, Any]:
    path = _stage_path(out_dir, stage)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _summarize_stages(args: argparse.Namespace) -> dict[str, Any]:
    step1 = _read_stage(args.out_dir, "cube_removed")
    evidence = _read_stage(args.out_dir, "evidence")
    candidates = _read_stage(args.out_dir, "candidate_paths") if _stage_path(args.out_dir, "candidate_paths").exists() else {}
    step1_row = step1["rows"][0]
    step1_tcp_mm = float(step1_row["tcp_pose_error_mm"])
    if step1_tcp_mm <= 5.0:
        branch = "A_collision_path"
        final_stage = "final_collision"
        judgement = "cube_removed_reaches_tcp_under_5mm_collision_path_confirmed"
    else:
        branch = "B_drive_or_override"
        final_stage = "final_direct"
        judgement = "cube_removed_still_stalls_drive_or_override_confirmed"
    final = _read_stage(args.out_dir, final_stage)

    step1["prediction_a"] = "cube removed -> TCP error <5mm if path collision is the blocker"
    step1["prediction_b"] = "cube removed -> ~70mm stall if drive/override semantics are the blocker"
    step1["judgement"] = judgement
    step1["row"] = step1_row

    if branch == "A_collision_path":
        selected_repair = {
            "type": "waypoint_path_repair",
            "selected_mode": final.get("waypoint_mode", ""),
            "source": "stage_final_collision",
            "candidate_table": candidates.get("candidates", []),
            "candidate_selection": candidates.get("selected_repair", {}),
        }
    else:
        selected_repair = {
            "type": "direct_joint_target_write_through",
            "reason": "Step 1 kept the runtime stall without cube; repair targets external override delivery semantics, not effort/stiffness tuning.",
            "implementation": "write robot_dof_targets, set_joint_position_target, and write_data_to_sim before env.step while retaining the external override path.",
        }

    pass_all = int(final["pass_all_count"])
    if pass_all == 10:
        verdict = "D328_G0A_COLLISION_DRIVE_REPAIR_PASS"
    elif any(int(v) >= 3 for v in final["failure_counts"].values()):
        verdict = "D328_G0A_COLLISION_DRIVE_REPAIR_FAIL"
    else:
        verdict = "D328_G0A_COLLISION_DRIVE_REPAIR_PARTIAL"

    summary = {
        "artifact": "d328_g0a_collision_vs_drive_probe",
        "verdict": verdict,
        "active_case": "G0a",
        "new_variable": "none; D328 is diagnosis and one branch repair inside G0a",
        "invariants": {
            "pose_family": "D325 position-only tangent -1",
            "alignment_standoff_m": float(args.alignment_standoff_m),
            "no_gate_or_epsilon_tuning": True,
        },
        "environment": {"summary_stage": "no Isaac app launched"},
        "step1_cube_removed": step1,
        "branch": branch,
        "cube_present_evidence": evidence,
        "selected_repair": selected_repair,
        "candidate_paths": candidates,
        "final_retest": final,
        "non_goals": [
            "no pose-family change",
            "no epsilon/42mm/10mm/15deg/15mm gate tuning",
            "no second repair",
            "no G0b/cylinder",
            "no gripper close/grasp/lift",
            "no RL/PPO/VLA/RoArm/B200",
        ],
    }
    _write_summary(args.out_dir, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=d327.DEFAULT_ROBOT_USD)
    parser.add_argument("--urdf_path", type=Path, default=d327.DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=32801)
    parser.add_argument("--cube_x", type=float, default=0.30)
    parser.add_argument("--cube_y", type=float, default=0.0)
    parser.add_argument("--cube_size_m", type=float, default=0.10)
    parser.add_argument("--cube_mass_kg", type=float, default=0.72)
    parser.add_argument("--static_friction", type=float, default=1.5)
    parser.add_argument("--dynamic_friction", type=float, default=1.2)
    parser.add_argument("--approach_steps", type=int, default=220)
    parser.add_argument("--hold_steps", type=int, default=100)
    parser.add_argument("--pre_clearance_m", type=float, default=0.040)
    parser.add_argument("--radial_tip_past_near_face_m", type=float, default=0.010)
    parser.add_argument("--alignment_standoff_m", type=float, default=d327.ALIGNMENT_STANDOFF_M)
    parser.add_argument("--episode_length_s", type=float, default=16.5)
    parser.add_argument("--cube_removed_x", type=float, default=1.20)
    parser.add_argument("--cube_removed_y", type=float, default=0.55)
    parser.add_argument("--collision_repair_raise_m", type=float, default=0.070)
    parser.add_argument(
        "--stage",
        choices=("all", "cube_removed", "evidence", "candidate_paths", "final_direct", "final_collision", "summarize"),
        default="all",
    )
    parser.add_argument("--final_collision_waypoint_mode", default="high_corridor_drop")
    parser.add_argument("--live_viewer", action="store_true")
    args = parser.parse_args()

    if not args.robot_usd_path.exists():
        raise FileNotFoundError(args.robot_usd_path)
    if not args.urdf_path.exists():
        raise FileNotFoundError(args.urdf_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "summarize":
        summary = _summarize_stages(args)
        print(
            "[d328-g0a-summary] "
            f"verdict={summary['verdict']} branch={summary['branch']} "
            f"final_pass={summary['final_retest']['pass_all_count']}/10 "
            f"out_dir={_rel(args.out_dir)}"
        )
        return 0

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    environment = d327._version_manifest()
    environment["robot_usd_path"] = _rel(args.robot_usd_path)
    environment["urdf_path"] = _rel(args.urdf_path)

    if args.stage == "cube_removed":
        result = _run_motion(args, num_envs=1, label="cube_removed_decision", trace=True, cube_removed=True)
        result["environment"] = environment
        _write_stage(args.out_dir, "cube_removed", result)
        print(
            "[d328-stage-cube-removed] "
            f"tcp_mm={result['rows'][0]['tcp_pose_error_mm']:.3f} "
            f"out={_rel(_stage_path(args.out_dir, 'cube_removed'))}"
        )
        sim_app.close()
        return 0

    if args.stage == "evidence":
        result = _run_motion(args, num_envs=1, label="cube_present_evidence", trace=True, cube_removed=False)
        result["environment"] = environment
        _write_stage(args.out_dir, "evidence", result)
        print(
            "[d328-stage-evidence] "
            f"tcp_mm={result['rows'][0]['tcp_pose_error_mm']:.3f} "
            f"force_max={result['trace_stats'].get('max_contact_force_n', 0.0):.3f} "
            f"out={_rel(_stage_path(args.out_dir, 'evidence'))}"
        )
        sim_app.close()
        return 0

    if args.stage == "candidate_paths":
        env, inner, zero = d327._make_env(args, 1)
        origins, _cube_start_w, cube_start_local, _cube_start_w_np, _origins_np = d327._reset_env(inner, env, zero)
        del origins, zero
        target_info = d327._target_info(cube_start_local, args)
        candidates = _candidate_path_table(cube_start_local, target_info, args)
        selected_repair = _select_collision_repair(candidates)
        result = {
            "label": "candidate_paths",
            "environment": environment,
            "cube_start_local_m": cube_start_local[0].tolist(),
            "candidates": candidates,
            "selected_repair": selected_repair,
        }
        _write_stage(args.out_dir, "candidate_paths", result)
        env.close()
        print(
            "[d328-stage-candidate-paths] "
            f"selected={selected_repair['selected_mode']} "
            f"out={_rel(_stage_path(args.out_dir, 'candidate_paths'))}"
        )
        sim_app.close()
        return 0

    if args.stage == "final_direct":
        result = _run_motion(
            args,
            num_envs=10,
            label="final_direct_write_retest",
            trace=True,
            direct_write_through=True,
        )
        result["environment"] = environment
        _write_stage(args.out_dir, "final_direct", result)
        print(
            "[d328-stage-final-direct] "
            f"pass={result['pass_all_count']}/10 "
            f"out={_rel(_stage_path(args.out_dir, 'final_direct'))}"
        )
        sim_app.close()
        return 0

    if args.stage == "final_collision":
        result = _run_motion(
            args,
            num_envs=10,
            label="final_collision_path_retest",
            trace=True,
            waypoint_mode=str(args.final_collision_waypoint_mode),
        )
        result["environment"] = environment
        _write_stage(args.out_dir, "final_collision", result)
        print(
            "[d328-stage-final-collision] "
            f"mode={args.final_collision_waypoint_mode} pass={result['pass_all_count']}/10 "
            f"out={_rel(_stage_path(args.out_dir, 'final_collision'))}"
        )
        sim_app.close()
        return 0

    step1 = _run_motion(args, num_envs=1, label="cube_removed_decision", trace=True, cube_removed=True)
    step1_row = step1["rows"][0]
    step1_tcp_mm = float(step1_row["tcp_pose_error_mm"])
    if step1_tcp_mm <= 5.0:
        branch = "A_collision_path"
        judgement = "cube_removed_reaches_tcp_under_5mm_collision_path_confirmed"
    else:
        branch = "B_drive_or_override"
        judgement = "cube_removed_still_stalls_drive_or_override_confirmed"
    step1["prediction_a"] = "cube removed -> TCP error <5mm if path collision is the blocker"
    step1["prediction_b"] = "cube removed -> ~70mm stall if drive/override semantics are the blocker"
    step1["judgement"] = judgement
    step1["row"] = step1_row

    evidence = _run_motion(args, num_envs=1, label="cube_present_evidence", trace=True, cube_removed=False)

    selected_repair: dict[str, Any]
    if branch == "A_collision_path":
        # Build the candidate table in a short env solely from the reset geometry.
        env, inner, zero = d327._make_env(args, 1)
        origins, _cube_start_w, cube_start_local, _cube_start_w_np, _origins_np = d327._reset_env(inner, env, zero)
        del origins, zero
        target_info = d327._target_info(cube_start_local, args)
        candidates = _candidate_path_table(cube_start_local, target_info, args)
        selected_repair = _select_collision_repair(candidates)
        env.close()
        final = _run_motion(
            args,
            num_envs=10,
            label="final_collision_path_retest",
            trace=True,
            waypoint_mode=str(selected_repair["selected_mode"]),
        )
    else:
        selected_repair = {
            "type": "direct_joint_target_write_through",
            "reason": "Step 1 kept the ~70mm stall without cube; repair targets external override delivery semantics, not effort/stiffness tuning.",
            "implementation": "write robot_dof_targets, set_joint_position_target, and write_data_to_sim before env.step while retaining the external override path.",
        }
        final = _run_motion(
            args,
            num_envs=10,
            label="final_direct_write_retest",
            trace=True,
            direct_write_through=True,
        )

    pass_all = int(final["pass_all_count"])
    if pass_all == 10:
        verdict = "D328_G0A_COLLISION_DRIVE_REPAIR_PASS"
    elif any(int(v) >= 3 for v in final["failure_counts"].values()):
        verdict = "D328_G0A_COLLISION_DRIVE_REPAIR_FAIL"
    else:
        verdict = "D328_G0A_COLLISION_DRIVE_REPAIR_PARTIAL"

    summary = {
        "artifact": "d328_g0a_collision_vs_drive_probe",
        "verdict": verdict,
        "active_case": "G0a",
        "new_variable": "none; D328 is diagnosis and one branch repair inside G0a",
        "invariants": {
            "pose_family": "D325 position-only tangent -1",
            "alignment_standoff_m": float(args.alignment_standoff_m),
            "no_gate_or_epsilon_tuning": True,
        },
        "environment": environment,
        "step1_cube_removed": step1,
        "branch": branch,
        "cube_present_evidence": evidence,
        "selected_repair": selected_repair,
        "final_retest": final,
        "non_goals": [
            "no pose-family change",
            "no epsilon/42mm/10mm/15deg/15mm gate tuning",
            "no second repair",
            "no G0b/cylinder",
            "no gripper close/grasp/lift",
            "no RL/PPO/VLA/RoArm/B200",
        ],
    }
    _write_summary(args.out_dir, summary)
    print(
        "[d328-g0a] "
        f"verdict={verdict} branch={branch} "
        f"step1_tcp_mm={step1_tcp_mm:.3f} final_pass={pass_all}/10 "
        f"out_dir={_rel(args.out_dir)}"
    )
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
