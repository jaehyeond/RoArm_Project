"""Posthoc y+ target-path and actuator audit for the professor 10cm branch.

This is a local CSV/trace reader only. It does not run IsaacLab, use a GPU,
train, generate data, touch the robot, or reconnect any remote machine.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ARM_JOINT_COUNT = 5


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value == "" else int(float(value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            row["_line"] = str(line_no)
            rows.append(row)
    return rows


def _target_offsets(row: dict[str, str]) -> dict[str, float]:
    return {
        "target_x_minus_cube_x_m": _float(row, "target_x_m") - _float(row, "cube_x_m"),
        "target_y_minus_cube_y_m": _float(row, "target_y_m") - _float(row, "cube_y_m"),
        "target_z_minus_cube_z_m": _float(row, "target_z_m") - _float(row, "cube_z_m"),
    }


def _target_offsets_from_start(row: dict[str, str], start: dict[str, str]) -> dict[str, float]:
    return {
        "target_x_minus_start_cube_x_m": _float(row, "target_x_m") - _float(start, "cube_x_m"),
        "target_y_minus_start_cube_y_m": _float(row, "target_y_m") - _float(start, "cube_y_m"),
        "target_z_minus_start_cube_z_m": _float(row, "target_z_m") - _float(start, "cube_z_m"),
    }


def _tcp_minus_target(row: dict[str, str]) -> dict[str, float]:
    return {
        "x_m": _float(row, "tcp_x_after_m") - _float(row, "target_x_m"),
        "y_m": _float(row, "tcp_y_after_m") - _float(row, "target_y_m"),
        "z_m": _float(row, "tcp_z_after_m") - _float(row, "target_z_m"),
    }


def _xy_norm(delta: dict[str, float]) -> float:
    return math.hypot(delta["x_m"], delta["y_m"])


def _joint_abs_means(rows: list[dict[str, str]], prefix: str) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for idx in range(ARM_JOINT_COUNT):
        key = f"{prefix}_{idx}_rad"
        vals = [abs(_float(row, key)) for row in rows]
        out.append({"joint": idx, "mean_abs_rad": _mean(vals), "max_abs_rad": max(vals, default=0.0)})
    return out


def _worst_joint(stats: list[dict[str, float | int]]) -> dict[str, float | int]:
    return max(stats, key=lambda row: float(row["mean_abs_rad"])) if stats else {"joint": -1, "mean_abs_rad": 0.0}


def _clip_rates(rows: list[dict[str, str]]) -> list[dict[str, float | int]]:
    return [
        {"joint": idx, "rate": _mean([float(_int(row, f"clip_mask_{idx}")) for row in rows])}
        for idx in range(ARM_JOINT_COUNT)
    ]


def _row_brief(row: dict[str, str]) -> dict[str, Any]:
    delta = _tcp_minus_target(row)
    target_offsets = _target_offsets(row)
    return {
        "line": _int(row, "_line"),
        "step": _int(row, "step"),
        "phase_alpha": _float(row, "phase_alpha"),
        "target_offsets_m": target_offsets,
        "tcp_minus_target_m": delta,
        "tcp_target_err_after_m": _float(row, "tcp_target_err_after_m"),
        "tcp_minus_target_xy_m": _xy_norm(delta),
        "tcp_cube_dist_m": _float(row, "tcp_cube_dist_m"),
        "clip_joint_count": _int(row, "clip_joint_count"),
        "clip_max_joint_name": row.get("clip_max_joint_name", ""),
        "joint_step_scale": _float(row, "joint_step_scale"),
        "measured_contact_seen": bool(_int(row, "measured_contact_seen")),
        "contact_stop_seen": bool(_int(row, "contact_stop_seen")),
    }


def _env_audit(env_id: int, rows: list[dict[str, str]], summary: dict[str, str]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: _int(row, "step"))
    first = ordered[0]
    final = ordered[-1]
    min_tcp_err = min(ordered, key=lambda row: _float(row, "tcp_target_err_after_m"))
    min_tcp_cube = min(ordered, key=lambda row: _float(row, "tcp_cube_dist_m"))
    alpha_zero = [row for row in ordered if _float(row, "phase_alpha") == 0.0]
    alpha_final = [row for row in ordered if _float(row, "phase_alpha") >= 1.0]

    follow_stats = _joint_abs_means(ordered, "joint_follow_err")
    raw_stats = _joint_abs_means(ordered, "raw_delta")
    clipped_stats = _joint_abs_means(ordered, "clipped_delta")

    first_offset = _target_offsets(first)
    final_offset = _target_offsets(final)
    final_delta = _tcp_minus_target(final)
    first_target_from_start = _target_offsets_from_start(first, first)
    final_target_from_start = _target_offsets_from_start(final, first)
    target_world_delta = {
        "x_m": _float(final, "target_x_m") - _float(first, "target_x_m"),
        "y_m": _float(final, "target_y_m") - _float(first, "target_y_m"),
        "z_m": _float(final, "target_z_m") - _float(first, "target_z_m"),
    }

    return {
        "env_id": env_id,
        "summary_line": _int(summary, "_line"),
        "summary_measured_contact_seen": bool(_int(summary, "measured_contact_seen")),
        "summary_contact_stop_seen": bool(_int(summary, "contact_stop_seen")),
        "summary_final_tcp_target_err_m": _float(summary, "final_tcp_target_err_m"),
        "summary_max_disp_along_push_m": _float(summary, "max_disp_along_push_m"),
        "trace_line_span": [_int(first, "_line"), _int(final, "_line")],
        "trace_row_count": len(ordered),
        "alpha_zero_row_count": len(alpha_zero),
        "alpha_final_row_count": len(alpha_final),
        "target_path_delta_y_m": final_offset["target_y_minus_cube_y_m"]
        - first_offset["target_y_minus_cube_y_m"],
        "target_world_delta_m": target_world_delta,
        "first_target_from_start_cube_m": first_target_from_start,
        "final_target_from_start_cube_m": final_target_from_start,
        "first_row": _row_brief(first),
        "final_row": _row_brief(final),
        "min_tcp_target_err_row": _row_brief(min_tcp_err),
        "min_tcp_cube_dist_row": _row_brief(min_tcp_cube),
        "final_abs_z_error_m": abs(final_delta["z_m"]),
        "final_xy_error_m": _xy_norm(final_delta),
        "final_z_error_fraction_of_tcp_error": (
            abs(final_delta["z_m"]) / _float(final, "tcp_target_err_after_m")
            if _float(final, "tcp_target_err_after_m") > 0.0
            else 0.0
        ),
        "clip_any_rate": _mean([float(_int(row, "clip_any")) for row in ordered]),
        "clip_joint_count_mean": _mean([float(_int(row, "clip_joint_count")) for row in ordered]),
        "clip_rates_by_joint": _clip_rates(ordered),
        "joint_follow_err_abs": follow_stats,
        "raw_delta_abs": raw_stats,
        "clipped_delta_abs": clipped_stats,
        "worst_follow_joint": _worst_joint(follow_stats),
        "worst_raw_delta_joint": _worst_joint(raw_stats),
    }


def _split_stats(env_audits: list[dict[str, Any]], contact: bool) -> dict[str, Any]:
    selected = [row for row in env_audits if bool(row["summary_measured_contact_seen"]) == contact]
    return {
        "n": len(selected),
        "env_ids": [int(row["env_id"]) for row in selected],
        "summary_lines": [int(row["summary_line"]) for row in selected],
        "final_abs_z_error_mean_m": _mean([float(row["final_abs_z_error_m"]) for row in selected]),
        "final_xy_error_mean_m": _mean([float(row["final_xy_error_m"]) for row in selected]),
        "final_tcp_target_err_mean_m": _mean(
            [float(row["final_row"]["tcp_target_err_after_m"]) for row in selected]
        ),
        "final_tcp_cube_dist_mean_m": _mean([float(row["final_row"]["tcp_cube_dist_m"]) for row in selected]),
        "final_z_error_fraction_mean": _mean(
            [float(row["final_z_error_fraction_of_tcp_error"]) for row in selected]
        ),
        "target_path_delta_y_mean_m": _mean([float(row["target_path_delta_y_m"]) for row in selected]),
        "target_world_delta_y_mean_m": _mean(
            [float(row["target_world_delta_m"]["y_m"]) for row in selected]
        ),
        "final_target_y_minus_start_cube_y_mean_m": _mean(
            [float(row["final_target_from_start_cube_m"]["target_y_minus_start_cube_y_m"]) for row in selected]
        ),
        "final_target_z_minus_start_cube_z_mean_m": _mean(
            [float(row["final_target_from_start_cube_m"]["target_z_minus_start_cube_z_m"]) for row in selected]
        ),
        "clip_any_rate_mean": _mean([float(row["clip_any_rate"]) for row in selected]),
        "clip_joint_count_mean": _mean([float(row["clip_joint_count_mean"]) for row in selected]),
        "worst_follow_joint_modes": [
            int(row["worst_follow_joint"]["joint"]) for row in selected
        ],
    }


def build_audit(summary_csv: Path, trace_csv: Path) -> dict[str, Any]:
    summary_rows = _load_csv(summary_csv)
    trace_rows = _load_csv(trace_csv)
    if not summary_rows:
        raise RuntimeError(f"empty summary csv: {summary_csv}")
    if not trace_rows:
        raise RuntimeError(f"empty trace csv: {trace_csv}")

    summary_by_env = {_int(row, "env_id"): row for row in summary_rows}
    env_audits: list[dict[str, Any]] = []
    for env_id in sorted({_int(row, "env_id") for row in trace_rows}):
        env_rows = [row for row in trace_rows if _int(row, "env_id") == env_id]
        env_audits.append(_env_audit(env_id, env_rows, summary_by_env[env_id]))

    first_offsets = [audit["first_row"]["target_offsets_m"] for audit in env_audits]
    final_offsets = [audit["final_row"]["target_offsets_m"] for audit in env_audits]
    final_from_start = [audit["final_target_from_start_cube_m"] for audit in env_audits]

    return {
        "scope": "local_posthoc_yplus_trace_path_actuator_no_gpu_no_training_no_dataset",
        "summary_csv": str(summary_csv),
        "trace_csv": str(trace_csv),
        "traced_env_count": len(env_audits),
        "trace_rows": len(trace_rows),
        "target_path_summary": {
            "first_target_y_minus_cube_y_mean_m": _mean(
                [float(row["target_y_minus_cube_y_m"]) for row in first_offsets]
            ),
            "final_target_y_minus_cube_y_mean_m": _mean(
                [float(row["target_y_minus_cube_y_m"]) for row in final_offsets]
            ),
            "target_path_delta_y_mean_m": _mean([float(row["target_path_delta_y_m"]) for row in env_audits]),
            "target_world_delta_y_mean_m": _mean(
                [float(row["target_world_delta_m"]["y_m"]) for row in env_audits]
            ),
            "final_target_x_minus_cube_x_mean_m": _mean(
                [float(row["target_x_minus_cube_x_m"]) for row in final_offsets]
            ),
            "final_target_z_minus_cube_z_mean_m": _mean(
                [float(row["target_z_minus_cube_z_m"]) for row in final_offsets]
            ),
            "final_target_y_minus_start_cube_y_mean_m": _mean(
                [float(row["target_y_minus_start_cube_y_m"]) for row in final_from_start]
            ),
            "final_target_z_minus_start_cube_z_mean_m": _mean(
                [float(row["target_z_minus_start_cube_z_m"]) for row in final_from_start]
            ),
        },
        "contact_split": {
            "contact": _split_stats(env_audits, True),
            "no_contact": _split_stats(env_audits, False),
        },
        "env_audits": env_audits,
        "interpretation": [
            "The y+ near-face target path is short and lateral-neutral; measured-stop freeze can truncate the final target y advance in contact envs.",
            "The side-center target remains near the start cube height, while the final TCP remains several centimeters above the target.",
            "No-contact traced envs retain larger vertical TCP-target error than contact traced envs; the final TCP error is mostly height error.",
            "Clipping and actuator follow lag remain present in both contact and no-contact traced groups, so the next test should change geometry/height/lateral/workspace hypothesis before data or RL.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--trace_csv", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()

    audit = build_audit(args.summary_csv, args.trace_csv)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    contact = audit["contact_split"]["contact"]
    no_contact = audit["contact_split"]["no_contact"]
    target = audit["target_path_summary"]
    print(
        "yplus_trace_path line1 "
        f"traced_env_count={audit['traced_env_count']} trace_rows={audit['trace_rows']} "
        f"target_world_dy_mean={target['target_world_delta_y_mean_m']:.9f} "
        f"final_target_z_minus_start_cube_z_mean={target['final_target_z_minus_start_cube_z_mean_m']:.9f}"
    )
    print(
        "yplus_trace_path line2 "
        f"contact_envs={contact['env_ids']} no_contact_envs={no_contact['env_ids']} "
        f"contact_final_z_err={contact['final_abs_z_error_mean_m']:.9f} "
        f"no_contact_final_z_err={no_contact['final_abs_z_error_mean_m']:.9f}"
    )
    print(
        "yplus_trace_path line3 "
        f"contact_clip_any={contact['clip_any_rate_mean']:.6f} "
        f"no_contact_clip_any={no_contact['clip_any_rate_mean']:.6f} "
        f"contact_clip_joint_count={contact['clip_joint_count_mean']:.6f} "
        f"no_contact_clip_joint_count={no_contact['clip_joint_count_mean']:.6f}"
    )
    print("yplus_trace_path line4 verdict=LOCAL_DIAG_GEOMETRY_HEIGHT_WORKSPACE_BEFORE_GPU_OR_DATA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
