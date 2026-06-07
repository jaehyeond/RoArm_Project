"""Audit seed962 cube10cm contact-frame TCP/target geometry mismatch.

Local trace reader only: no IsaacLab runtime, no GPU, no dataset generation,
no training, no robot control, no SSH. This audit checks whether the visual
sanity blocker is a target-definition problem or a DiffIK tracking/clipping
problem at the first measured contact frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_TRACE = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_trace.csv"
DEFAULT_SUMMARY = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_summary.json"
DEFAULT_VISUAL_AUDIT = LOG_DIR / "cube10cm_visual_sim_sanity_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_contact_frame_geometry_mismatch_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_contact_frame_geometry_mismatch_audit_summary.out"


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "p95": None}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": mean(values),
        "median": median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[p95_idx],
    }


def _read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for source_line, row in enumerate(reader, start=2):
            row["_source_line"] = source_line
            rows.append(row)
    return rows


def _first_contact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_env: dict[int, dict[str, Any]] = {}
    for row in rows:
        env_id = _i(row, "env_id")
        if env_id in by_env:
            continue
        if _i(row, "measured_contact_now") == 1:
            by_env[env_id] = row
    return [by_env[k] for k in sorted(by_env)]


def _mode(values: list[str]) -> tuple[str | None, float | None]:
    if not values:
        return None, None
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value, count / len(values)


def build_audit(trace_csv: Path, summary_json: Path, visual_audit_json: Path) -> dict[str, Any]:
    rows = _read_trace(trace_csv)
    if not rows:
        raise RuntimeError(f"trace has no rows: {trace_csv}")
    summary = json.loads(summary_json.read_text())
    visual = json.loads(visual_audit_json.read_text()) if visual_audit_json.exists() else {}
    contact_rows = _first_contact_rows(rows)
    if not contact_rows:
        raise RuntimeError("no measured_contact_now rows found")

    env_ids = sorted({_i(row, "env_id") for row in rows})
    cube_half_z = _f(contact_rows[0], "cube_size_z_m") / 2.0
    near_threshold = float(summary.get("contact_near_tcp_cube_dist_m", 0.085))

    tcp_minus_target_z: list[float] = []
    link5_minus_link5_target_z: list[float] = []
    tcp_above_live_cube_center_z: list[float] = []
    tcp_below_live_cube_top_z: list[float] = []
    target_minus_live_cube_center_z: list[float] = []
    xy_err: list[float] = []
    z_err_fraction: list[float] = []
    offset_consistency_abs: list[float] = []
    tcp_cube_dist: list[float] = []
    contact_lines: list[int] = []
    clip_names: list[str] = []
    clip_any_count = 0
    top_near_count = 0
    center_near_count = 0
    tcp_dist_gt_near_count = 0

    per_env: list[dict[str, Any]] = []
    for row in contact_rows:
        tcp_x = _f(row, "tcp_x_before_m", _f(row, "tcp_x_m"))
        tcp_y = _f(row, "tcp_y_before_m", _f(row, "tcp_y_m"))
        tcp_z = _f(row, "tcp_z_before_m", _f(row, "tcp_z_m"))
        target_x = _f(row, "target_x_m")
        target_y = _f(row, "target_y_m")
        target_z = _f(row, "target_z_m")
        cube_z = _f(row, "cube_z_m")
        link5_z = _f(row, "link5_z_before_m")
        link5_target_z = _f(row, "link5_target_z_m")
        tcp_err = _f(row, "tcp_target_err_before_m")
        link5_tcp_offset_z = tcp_z - link5_z
        target_link5_offset_z = target_z - link5_target_z
        xy = math.hypot(tcp_x - target_x, tcp_y - target_y)
        tcp_target_z_delta = tcp_z - target_z
        link5_target_z_delta = link5_z - link5_target_z
        tcp_live_center_z_delta = tcp_z - cube_z
        tcp_live_top_delta = cube_z + cube_half_z - tcp_z
        target_live_center_delta = target_z - cube_z
        dist = _f(row, "tcp_cube_dist_m")
        source_line = int(row["_source_line"])
        clip_any = _i(row, "clip_any")
        clip_name = row.get("clip_max_joint_name", "")

        tcp_minus_target_z.append(tcp_target_z_delta)
        link5_minus_link5_target_z.append(link5_target_z_delta)
        tcp_above_live_cube_center_z.append(tcp_live_center_z_delta)
        tcp_below_live_cube_top_z.append(tcp_live_top_delta)
        target_minus_live_cube_center_z.append(target_live_center_delta)
        xy_err.append(xy)
        if tcp_err > 1.0e-9:
            z_err_fraction.append(abs(tcp_target_z_delta) / tcp_err)
        offset_consistency_abs.append(abs(link5_tcp_offset_z - target_link5_offset_z))
        tcp_cube_dist.append(dist)
        contact_lines.append(source_line)
        if clip_any:
            clip_any_count += 1
        if clip_name:
            clip_names.append(clip_name)
        if abs(tcp_live_top_delta) <= 0.010:
            top_near_count += 1
        if abs(tcp_live_center_z_delta) <= 0.010:
            center_near_count += 1
        if dist > near_threshold:
            tcp_dist_gt_near_count += 1

        per_env.append(
            {
                "env_id": _i(row, "env_id"),
                "source_line": source_line,
                "step": _i(row, "step"),
                "frame": _i(row, "frame"),
                "tcp_minus_target_z_m": tcp_target_z_delta,
                "link5_minus_link5_target_z_m": link5_target_z_delta,
                "tcp_above_live_cube_center_z_m": tcp_live_center_z_delta,
                "tcp_below_live_cube_top_z_m": tcp_live_top_delta,
                "target_minus_live_cube_center_z_m": target_live_center_delta,
                "tcp_target_xy_err_m": xy,
                "tcp_target_err_before_m": tcp_err,
                "z_err_fraction_of_tcp_err": abs(tcp_target_z_delta) / tcp_err if tcp_err > 1.0e-9 else None,
                "tcp_cube_dist_m": dist,
                "near_tcp_cube_now": _i(row, "near_tcp_cube_now"),
                "clip_any": clip_any,
                "clip_max_joint_name": clip_name,
            }
        )

    clip_mode, clip_mode_rate = _mode(clip_names)
    n = len(contact_rows)
    offset_consistent = max(offset_consistency_abs) <= 1.0e-5
    vertical_dominant = mean(z_err_fraction) >= 0.90
    tcp_near_top_rate = top_near_count / n
    tcp_near_center_rate = center_near_count / n
    clean_tap_visual_verified = bool(
        visual.get("clean_tap_visual_verified", False)
        and not vertical_dominant
        and (clip_any_count / n) < 0.25
    )

    return {
        "artifact_type": "cube10cm_contact_frame_geometry_mismatch_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_audit_only": True,
        "no_gpu_runtime_dataset_training_robot_ssh": True,
        "source": {
            "trace_csv": str(trace_csv),
            "summary_json": str(summary_json),
            "visual_audit_json": str(visual_audit_json),
            "trace_rows": len(rows),
            "env_ids": env_ids,
            "contact_env_count": n,
            "contact_source_line_min": min(contact_lines),
            "contact_source_line_max": max(contact_lines),
        },
        "runtime_contract": {
            "tcp_height_mode": summary.get("tcp_height_mode"),
            "tcp_center_height_offset_m": summary.get("tcp_center_height_offset_m"),
            "directional_tcp_center_height_offsets_m": summary.get("directional_tcp_center_height_offsets_m"),
            "applied_tcp_center_height_offset_mean_m": summary.get("applied_tcp_center_height_offset_mean_m"),
            "fixed_push_dir": summary.get("fixed_push_dir"),
            "cube_size_m": summary.get("cube_size_m"),
            "cube_start_z_mean_m": summary.get("cube_start_z_mean_m"),
            "diffik_clip_rate_mean": summary.get("diffik_clip_rate_mean"),
            "min_tcp_target_err_mean_m": summary.get("min_tcp_target_err_mean_m"),
            "final_tcp_target_err_mean_m": summary.get("final_tcp_target_err_mean_m"),
        },
        "contact_geometry": {
            "tcp_minus_target_z_m": _stats(tcp_minus_target_z),
            "link5_minus_link5_target_z_m": _stats(link5_minus_link5_target_z),
            "tcp_above_live_cube_center_z_m": _stats(tcp_above_live_cube_center_z),
            "tcp_below_live_cube_top_z_m": _stats(tcp_below_live_cube_top_z),
            "target_minus_live_cube_center_z_m": _stats(target_minus_live_cube_center_z),
            "tcp_target_xy_err_m": _stats(xy_err),
            "z_err_fraction_of_tcp_err": _stats(z_err_fraction),
            "tcp_cube_dist_m": _stats(tcp_cube_dist),
            "tcp_near_live_cube_top_10mm_rate": tcp_near_top_rate,
            "tcp_near_live_cube_center_10mm_rate": tcp_near_center_rate,
            "tcp_cube_dist_gt_near_threshold_rate": tcp_dist_gt_near_count / n,
        },
        "tcp_link5_offset_check": {
            "offset_consistency_abs_m": _stats(offset_consistency_abs),
            "consistent_with_tcp_offset_compensation": offset_consistent,
            "interpretation": (
                "tcp_local_offset_is_accounted_for; mismatch is link5/TCP target tracking"
                if offset_consistent
                else "tcp_local_offset_or_trace_inconsistency_needs_inspection"
            ),
        },
        "clip_at_contact": {
            "clip_any_rate": clip_any_count / n,
            "clip_max_joint_name_mode": clip_mode,
            "clip_max_joint_name_mode_rate": clip_mode_rate,
        },
        "per_env_first_contact": per_env,
        "verdict": {
            "visual_contact_replay_pass": bool(visual.get("visual_contact_evidence", False)),
            "clean_tap_visual_verified": clean_tap_visual_verified,
            "side_center_target_reached": not vertical_dominant,
            "mismatch_class": (
                "SIDE_CENTER_TARGET_NOT_TRACKED_TCP_CONTACTS_NEAR_TOP_UNDER_CLIPPING"
                if offset_consistent and vertical_dominant and tcp_near_top_rate >= 0.75
                else "CONTACT_GEOMETRY_MISMATCH_NEEDS_MORE_LOCAL_INSPECTION"
            ),
            "dataset_rl_roarm_unblocked": False,
            "next": "local_teacher_contact_geometry_or_tracking_fix_before_any_dataset_rl_robot",
        },
    }


def write_summary(audit: dict[str, Any], path: Path) -> None:
    src = audit["source"]
    contract = audit["runtime_contract"]
    geom = audit["contact_geometry"]
    offset = audit["tcp_link5_offset_check"]
    clip = audit["clip_at_contact"]
    verdict = audit["verdict"]
    lines = [
        "line1 artifact=cube10cm_contact_frame_geometry_mismatch_audit_v1 "
        "local_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        "line2 source "
        f"trace_rows={src['trace_rows']} envs={len(src['env_ids'])} first_contact_envs={src['contact_env_count']} "
        f"contact_source_lines={src['contact_source_line_min']}-{src['contact_source_line_max']}",
        "line3 runtime_contract "
        f"tcp_height_mode={contract['tcp_height_mode']} tcp_center_height_offset={contract['tcp_center_height_offset_m']} "
        f"directional_offsets={contract['directional_tcp_center_height_offsets_m']} "
        f"applied_offset_mean={contract['applied_tcp_center_height_offset_mean_m']} "
        f"diffik_clip_rate_mean={contract['diffik_clip_rate_mean']}",
        "line4 first_contact_vertical "
        f"tcp_minus_target_z_mean={geom['tcp_minus_target_z_m']['mean']:.9f} "
        f"link5_minus_target_z_mean={geom['link5_minus_link5_target_z_m']['mean']:.9f} "
        f"z_err_fraction_mean={geom['z_err_fraction_of_tcp_err']['mean']:.9f} "
        f"tcp_target_xy_err_mean={geom['tcp_target_xy_err_m']['mean']:.9f}",
        "line5 contact_surface "
        f"tcp_above_live_cube_center_z_mean={geom['tcp_above_live_cube_center_z_m']['mean']:.9f} "
        f"tcp_below_live_cube_top_z_mean={geom['tcp_below_live_cube_top_z_m']['mean']:.9f} "
        f"target_minus_live_cube_center_z_mean={geom['target_minus_live_cube_center_z_m']['mean']:.9f} "
        f"tcp_near_top_10mm_rate={geom['tcp_near_live_cube_top_10mm_rate']:.9f} "
        f"tcp_near_center_10mm_rate={geom['tcp_near_live_cube_center_10mm_rate']:.9f}",
        "line6 tcp_link5_offset_check "
        f"consistent={offset['consistent_with_tcp_offset_compensation']} "
        f"offset_consistency_abs_max={offset['offset_consistency_abs_m']['max']:.9f} "
        f"interpretation={offset['interpretation']}",
        "line7 clipping "
        f"clip_any_rate_at_first_contact={clip['clip_any_rate']:.9f} "
        f"clip_mode={clip['clip_max_joint_name_mode']} "
        f"clip_mode_rate={clip['clip_max_joint_name_mode_rate']:.9f}",
        "line8 verdict "
        f"visual_contact_replay_pass={verdict['visual_contact_replay_pass']} "
        f"clean_tap_visual_verified={verdict['clean_tap_visual_verified']} "
        f"side_center_target_reached={verdict['side_center_target_reached']} "
        f"mismatch_class={verdict['mismatch_class']}",
        "line9 pipeline "
        f"dataset_rl_roarm_unblocked={verdict['dataset_rl_roarm_unblocked']} "
        f"next={verdict['next']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_csv", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--visual_audit_json", type=Path, default=DEFAULT_VISUAL_AUDIT)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    audit = build_audit(args.trace_csv, args.summary_json, args.visual_audit_json)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_summary(audit, args.out_summary)
    print(args.out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
