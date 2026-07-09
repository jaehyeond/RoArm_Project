#!/usr/bin/env python3
"""D324 visual debugging infra demo for G0a frame repair.

This script does not redefine G0a, close the gripper, run RL/PPO, or advance the
variable ladder.  It reuses D323 recorded geometry and produces diagnostic
snapshots that make target-vs-actual frame mismatch visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.viz_debug import draw_frames, frame_from_axes, log_rerun, snapshot_frame_plot
from sim_scripts.cube10cm_top_view_d323_grasp_g0a_frame_repair_probe import (
    _seed_set,
    _solve_runtime_ik,
    _target_geometry,
)


DEFAULT_D323_SUMMARY = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d323/g0a_d323_alignment_summary.json"
DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/viz_infra_d324"
CUBE_CENTER_M = np.asarray([0.30, 0.0, 0.037883], dtype=np.float64)
CUBE_SIZE_M = 0.10
FIXED_JAW_LOCAL_X_M = -0.008


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _as_np3(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(3)


def _frame_from_result(name: str, result: dict[str, Any], *, role: str, label: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "label": label or name,
        "position": result["tcp_local_m"],
        "axes": {
            "x": result["link5_x_axis_world"],
            "y": result["link5_y_axis_world"],
            "z": result["link5_z_axis_world"],
        },
        "role": role,
    }


def _target_frame(name: str, target_tcp: Any, target_x_axis: Any, target_z_axis: Any, *, label: str) -> dict[str, Any]:
    return frame_from_axes(
        name,
        target_tcp,
        x_axis=target_x_axis,
        z_axis=target_z_axis,
        role="target",
        label=label,
    )


def _fixed_jaw_frame(name: str, tcp_frame: dict[str, Any], *, label: str) -> dict[str, Any]:
    rot = np.asarray(tcp_frame["rotation_matrix"], dtype=np.float64).reshape(3, 3) if "rotation_matrix" in tcp_frame else None
    if rot is None:
        axes = tcp_frame["axes"]
        rot = np.column_stack([_as_np3(axes["x"]), _as_np3(axes["y"]), _as_np3(axes["z"])])
    tcp = _as_np3(tcp_frame["position"])
    pos = tcp + rot[:, 0] * FIXED_JAW_LOCAL_X_M
    return frame_from_axes(
        name,
        pos,
        x_axis=rot[:, 0],
        z_axis=rot[:, 2],
        role="fixed_jaw",
        label=label,
    )


def _cube_face_frame(name: str, tangent_axis: np.ndarray, *, label: str) -> dict[str, Any]:
    face = CUBE_CENTER_M - tangent_axis * (CUBE_SIZE_M * 0.5)
    return frame_from_axes(
        name,
        face,
        x_axis=tangent_axis,
        z_axis=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        role="cube_face",
        label=label,
    )


def _object_frame() -> dict[str, Any]:
    return frame_from_axes(
        "cube_object_frame",
        CUBE_CENTER_M,
        x_axis=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        z_axis=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        role="object",
        label="cube",
    )


def _best_attempt_for_weight(weight: float, tangent_sign: float = -1.0) -> dict[str, Any]:
    geom = _target_geometry(
        CUBE_CENTER_M,
        CUBE_SIZE_M,
        tangent_sign=tangent_sign,
        radial_tip_past_near_face_m=0.010,
    )
    best: dict[str, Any] | None = None
    for seed in _seed_set():
        result = _solve_runtime_ik(
            np.asarray(geom["target_tcp"], dtype=np.float64),
            seed,
            target_x_axis=np.asarray(geom["target_x_axis"], dtype=np.float64),
            target_z_axis=np.asarray(geom["target_z_axis"], dtype=np.float64),
            orientation_weight_m=float(weight),
            max_iter=700,
            pos_tol_mm=1.0,
            axis_tol_deg=3.0,
        )
        score = (
            float(result["pos_err_mm"])
            + 10.0 * max(0.0, float(result["x_axis_err_deg"]) - 3.0)
            + 10.0 * max(0.0, float(result["z_axis_err_deg"]) - 3.0)
        )
        result.update(
            {
                "tangent_sign": float(tangent_sign),
                "orientation_weight_m": float(weight),
                "target_tcp_m": [float(v) for v in np.asarray(geom["target_tcp"]).tolist()],
                "target_x_axis_world": [float(v) for v in np.asarray(geom["target_x_axis"]).tolist()],
                "target_z_axis_world": [float(v) for v in np.asarray(geom["target_z_axis"]).tolist()],
                "score": float(score),
            }
        )
        if best is None or score < float(best["score"]):
            best = result
    if best is None:
        raise RuntimeError("no IK candidate produced")
    return best


def _snapshot_case(
    path: Path,
    *,
    title: str,
    target: dict[str, Any],
    actual: dict[str, Any],
    tangent_axis: np.ndarray,
    annotations: list[str],
) -> dict[str, Any]:
    fixed = _fixed_jaw_frame(f"{actual['name']}_fixed_jaw", actual, label="fixed jaw face")
    cube_face = _cube_face_frame("cube_side_face", tangent_axis, label="cube side face")
    pairs = [target, actual, fixed, cube_face, _object_frame()]
    status = snapshot_frame_plot(
        path,
        pairs,
        cube={"center": CUBE_CENTER_M.tolist(), "size": CUBE_SIZE_M},
        title=title,
        annotations=annotations,
    )
    status["frames"] = [item["name"] for item in pairs]
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d323_summary", type=Path, default=DEFAULT_D323_SUMMARY)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip_isaac_markers", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.d323_summary.read_text())
    best_strict = summary["best_strict_attempt"]
    position_only = next(
        item for item in summary["reachable_position_only_family"] if float(item["tangent_sign"]) == -1.0
    )
    target_tcp = best_strict["target_tcp_m"]
    target_x = np.asarray(best_strict["target_x_axis_world"], dtype=np.float64)
    target_z = np.asarray(best_strict["target_z_axis_world"], dtype=np.float64)
    tangent_axis = target_x
    strict_target = _target_frame(
        "strict_target_tcp",
        target_tcp,
        target_x,
        target_z,
        label="target strict TCP",
    )
    strict_actual = _frame_from_result("strict_best_attempt", best_strict, role="actual", label="actual best strict")
    strict_png = args.out_dir / "d324_strict_target_vs_best_attempt.png"
    strict_status = _snapshot_case(
        strict_png,
        title="D324 G0a strict target vs best-attempt actual frame",
        target=strict_target,
        actual=strict_actual,
        tangent_axis=tangent_axis,
        annotations=[
            "D323 strict family: target link5 +z = radial, +x = tangent.",
            f"TCP position error = {float(best_strict['pos_err_mm']):.3f} mm",
            f"tool-axis (+z) radial error = {float(best_strict['z_axis_err_deg']):.3f} deg",
            f"jaw-axis (+x) tangent error = {float(best_strict['x_axis_err_deg']):.3f} deg",
            "Expected visual cue: actual blue z-axis is visibly tilted from target radial z-axis.",
        ],
    )

    pos_target = _target_frame(
        "position_only_target_tcp",
        position_only["target_tcp_m"],
        position_only["target_x_axis_world"],
        position_only["target_z_axis_world"],
        label="target position TCP",
    )
    pos_actual = _frame_from_result("position_only_actual", position_only, role="actual", label="actual position-only")
    pos_png = args.out_dir / "d324_position_only_tangent_minus1.png"
    pos_status = _snapshot_case(
        pos_png,
        title="D324 G0a position-only reachable frame",
        target=pos_target,
        actual=pos_actual,
        tangent_axis=np.asarray(position_only["target_x_axis_world"], dtype=np.float64),
        annotations=[
            "D323 position-only tangent -1: TCP reaches target but wrist orientation does not.",
            f"TCP position error = {float(position_only['pos_err_mm']):.3f} mm",
            f"tool-axis (+z) radial error = {float(position_only['z_axis_err_deg']):.3f} deg",
            f"jaw-axis (+x) tangent error = {float(position_only['x_axis_err_deg']):.3f} deg",
            "Expected visual cue: target/actual origins overlap, but axes diverge strongly.",
        ],
    )

    tilt_reduced = _best_attempt_for_weight(0.02, tangent_sign=-1.0)
    strict_best_recomputed = _best_attempt_for_weight(0.10, tangent_sign=-1.0)
    candidates = [
        ("position_only_tangent_minus1", position_only, "D323 position-only tangent -1"),
        ("tilt_reduced_weight_0p02", tilt_reduced, "orientation-weight 0.02; z tilt reduced about 10-20 deg"),
        ("strict_best_weight_0p10", strict_best_recomputed, "orientation-weight 0.10; strict best trade-off"),
    ]
    candidate_rows: list[dict[str, Any]] = []
    candidate_statuses: dict[str, Any] = {}
    for name, candidate, description in candidates:
        target = _target_frame(
            f"{name}_target",
            candidate["target_tcp_m"],
            candidate["target_x_axis_world"],
            candidate["target_z_axis_world"],
            label=f"{name} target",
        )
        actual = _frame_from_result(f"{name}_actual", candidate, role="candidate", label=f"{name} actual")
        png_path = args.out_dir / f"d324_candidate_{name}.png"
        status = _snapshot_case(
            png_path,
            title=f"D324 G0a candidate sketch: {name}",
            target=target,
            actual=actual,
            tangent_axis=np.asarray(candidate["target_x_axis_world"], dtype=np.float64),
            annotations=[
                description,
                f"TCP position error = {float(candidate['pos_err_mm']):.3f} mm",
                f"jaw-axis (+x) tangent error = {float(candidate['x_axis_err_deg']):.3f} deg",
                f"tool-axis (+z) radial error = {float(candidate['z_axis_err_deg']):.3f} deg",
                "Candidate only: no G0a criterion changed in D324.",
            ],
        )
        candidate_statuses[name] = status
        candidate_rows.append(
            {
                "name": name,
                "description": description,
                "tcp_position_error_mm": float(candidate["pos_err_mm"]),
                "jaw_axis_tangent_error_deg": float(candidate["x_axis_err_deg"]),
                "tool_axis_radial_error_deg": float(candidate["z_axis_err_deg"]),
                "q_deg": candidate["q_deg"],
                "snapshot": _rel(png_path),
            }
        )

    all_marker_frames = [strict_target, strict_actual, pos_target, pos_actual, _object_frame()]
    marker_status = {"ok": False, "backend": "isaac_markers", "error": "skipped"}
    sim_app_to_close = None
    if not args.skip_isaac_markers:
        try:
            from isaaclab.app import AppLauncher

            app_launcher = AppLauncher(headless=True, enable_cameras=False)
            sim_app_to_close = app_launcher.app
            marker_status = draw_frames(all_marker_frames, prim_path="/World/D324VizFrames")
            sim_app_to_close.update()
        except Exception as exc:
            marker_status = {"ok": False, "backend": "isaac_markers", "error": repr(exc)}

    rerun_status = log_rerun(args.out_dir / "d324_g0a_frames.rrd", frames=all_marker_frames)
    visual_gate_pass = bool(strict_status.get("ok")) and bool(pos_status.get("ok"))
    report = {
        "artifact": "d324_g0a_viz_debug_demo",
        "new_experiment_variables": 0,
        "active_case": "G0a",
        "d323_summary": _rel(args.d323_summary),
        "out_dir": _rel(args.out_dir),
        "visual_gate_pass": visual_gate_pass,
        "visual_gate_rule": "strict and position-only D323 case snapshots must exist and carry target-vs-actual frame annotations",
        "snapshot_status": {
            "strict_target_vs_best_attempt": strict_status,
            "position_only_tangent_minus1": pos_status,
            "candidates": candidate_statuses,
        },
        "candidate_rows": candidate_rows,
        "marker_status": marker_status,
        "rerun_status": rerun_status,
        "non_goals": [
            "no G0a criterion change",
            "no 42/10mm tuning",
            "no large render",
            "no new object",
            "no gripper close",
            "no RL/PPO",
            "no ladder advance",
            "no B200/RoArm/VLA",
        ],
        "verdict": "D324_VIZ_DEBUG_SNAPSHOTS_PASS" if visual_gate_pass else "D324_VIZ_DEBUG_SNAPSHOTS_FAIL",
    }
    (args.out_dir / "d324_viz_debug_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# D324 G0a Candidate Pose Visual Sketch",
        "",
        "D324 records candidate visual material only. It does not change G0a criteria.",
        "",
        "| candidate | position err mm | jaw +x err deg | tool +z err deg | snapshot |",
        "|---|---:|---:|---:|---|",
    ]
    for row in candidate_rows:
        lines.append(
            "| {name} | {tcp_position_error_mm:.3f} | {jaw_axis_tangent_error_deg:.3f} | "
            "{tool_axis_radial_error_deg:.3f} | `{snapshot}` |".format(**row)
        )
    (args.out_dir / "d324_candidate_pose_table.md").write_text("\n".join(lines) + "\n")
    print(
        "[d324-viz] "
        f"verdict={report['verdict']} visual_gate={visual_gate_pass} "
        f"marker_ok={marker_status.get('ok')} rerun_ok={rerun_status.get('ok')} "
        f"out_dir={_rel(args.out_dir)}"
    )
    if sim_app_to_close is not None:
        sim_app_to_close.close()
    return 0 if visual_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
