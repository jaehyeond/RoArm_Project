"""Posthoc y+ geometry/reach audit for the professor 10cm cube branch.

This is a local log reader only. It does not run IsaacLab, train, generate a
dataset, touch the robot, or reconnect any remote machine.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable


SUMMARY_METRIC_KEYS = [
    "cube_y0_m",
    "cube_x0_m",
    "min_tcp_cube_dist_m",
    "min_tcp_target_err_m",
    "final_tcp_target_err_m",
    "max_disp_along_push_m",
    "max_cube_z_delta_m",
    "max_tip_angle_deg",
    "max_cube_speed_mps",
    "lateral_abs_m",
]


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


def _group_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "line_numbers": [_int(row, "_line") for row in rows],
        "means": {key: _mean([_float(row, key) for row in rows]) for key in SUMMARY_METRIC_KEYS},
    }


def _rate(rows: list[dict[str, str]], key: str) -> float:
    return _mean([float(_int(row, key)) for row in rows])


def _bin_stats(
    rows: list[dict[str, str]],
    name: str,
    predicate: Callable[[dict[str, str]], bool],
) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    return {
        "name": name,
        "n": len(selected),
        "line_numbers": [_int(row, "_line") for row in selected],
        "contact_evidence_rate": _rate(selected, "measured_contact_seen") if selected else 0.0,
        "reaction_event_rate": _rate(selected, "reaction_event") if selected else 0.0,
    }


def _delta_xyz(row: dict[str, str]) -> dict[str, float]:
    return {
        "x_m": _float(row, "tcp_x_after_m") - _float(row, "target_x_m"),
        "y_m": _float(row, "tcp_y_after_m") - _float(row, "target_y_m"),
        "z_m": _float(row, "tcp_z_after_m") - _float(row, "target_z_m"),
    }


def _trace_env_stats(
    trace_rows: list[dict[str, str]],
    summary_by_env: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    env_ids = sorted({_int(row, "env_id") for row in trace_rows})
    for env_id in env_ids:
        rows = [row for row in trace_rows if _int(row, "env_id") == env_id]
        final = max(rows, key=lambda row: _int(row, "step"))
        min_tcp_cube = min(rows, key=lambda row: _float(row, "tcp_cube_dist_m"))
        min_tcp_err = min(rows, key=lambda row: _float(row, "tcp_target_err_after_m"))
        summary = summary_by_env.get(env_id, {})
        out.append(
            {
                "env_id": env_id,
                "summary_line": _int(summary, "_line", 0),
                "summary_measured_contact_seen": bool(_int(summary, "measured_contact_seen")),
                "summary_contact_stop_seen": bool(_int(summary, "contact_stop_seen")),
                "final_line": _int(final, "_line"),
                "final_step": _int(final, "step"),
                "final_tcp_minus_target_xyz_m": _delta_xyz(final),
                "final_tcp_target_err_after_m": _float(final, "tcp_target_err_after_m"),
                "final_tcp_cube_dist_m": _float(final, "tcp_cube_dist_m"),
                "final_joint_step_scale": _float(final, "joint_step_scale"),
                "min_tcp_cube_line": _int(min_tcp_cube, "_line"),
                "min_tcp_cube_step": _int(min_tcp_cube, "step"),
                "min_tcp_cube_dist_m": _float(min_tcp_cube, "tcp_cube_dist_m"),
                "min_tcp_cube_near_seen": bool(_int(min_tcp_cube, "near_tcp_cube_seen")),
                "min_tcp_cube_measured_seen": bool(_int(min_tcp_cube, "measured_contact_seen")),
                "min_tcp_err_line": _int(min_tcp_err, "_line"),
                "min_tcp_err_step": _int(min_tcp_err, "step"),
                "min_tcp_target_err_after_m": _float(min_tcp_err, "tcp_target_err_after_m"),
                "min_tcp_err_z_delta_m": _delta_xyz(min_tcp_err)["z_m"],
            }
        )
    return out


def _trace_contact_split(trace_stats: list[dict[str, Any]], contact: bool) -> dict[str, float]:
    selected = [row for row in trace_stats if bool(row["summary_measured_contact_seen"]) == contact]
    final_z = [abs(float(row["final_tcp_minus_target_xyz_m"]["z_m"])) for row in selected]
    final_xy = [
        (float(row["final_tcp_minus_target_xyz_m"]["x_m"]) ** 2 + float(row["final_tcp_minus_target_xyz_m"]["y_m"]) ** 2)
        ** 0.5
        for row in selected
    ]
    return {
        "n": len(selected),
        "final_abs_z_error_mean_m": _mean(final_z),
        "final_xy_error_mean_m": _mean(final_xy),
        "final_tcp_cube_dist_mean_m": _mean([float(row["final_tcp_cube_dist_m"]) for row in selected]),
    }


def build_audit(summary_csv: Path, trace_csv: Path) -> dict[str, Any]:
    summary_rows = _load_csv(summary_csv)
    trace_rows = _load_csv(trace_csv)
    if not summary_rows:
        raise RuntimeError(f"empty summary csv: {summary_csv}")
    if not trace_rows:
        raise RuntimeError(f"empty trace csv: {trace_csv}")

    contact_rows = [row for row in summary_rows if _int(row, "measured_contact_seen") == 1]
    no_contact_rows = [row for row in summary_rows if _int(row, "measured_contact_seen") == 0]
    summary_by_env = {_int(row, "env_id"): row for row in summary_rows}
    trace_stats = _trace_env_stats(trace_rows, summary_by_env)

    bins = [
        _bin_stats(summary_rows, "cube_y0_m<=0", lambda row: _float(row, "cube_y0_m") <= 0.0),
        _bin_stats(summary_rows, "cube_y0_m>0", lambda row: _float(row, "cube_y0_m") > 0.0),
        _bin_stats(summary_rows, "cube_x0_m<0.25", lambda row: _float(row, "cube_x0_m") < 0.25),
        _bin_stats(summary_rows, "cube_x0_m>=0.25", lambda row: _float(row, "cube_x0_m") >= 0.25),
    ]

    audit = {
        "scope": "local_posthoc_yplus_geometry_reach_audit_no_gpu_no_training_no_dataset",
        "summary_csv": str(summary_csv),
        "trace_csv": str(trace_csv),
        "trials": len(summary_rows),
        "trace_rows": len(trace_rows),
        "contact_group": _group_stats(contact_rows),
        "no_contact_group": _group_stats(no_contact_rows),
        "position_bins": bins,
        "trace_env_stats": trace_stats,
        "trace_contact_split": {
            "contact": _trace_contact_split(trace_stats, True),
            "no_contact": _trace_contact_split(trace_stats, False),
        },
        "interpretation": [
            "Fixed y+ is not a pure reaction-rate failure; reaction is present but contact evidence is sparse.",
            "Contact is workspace-position dependent in this sample: cube_y0_m<=0 and cube_x0_m>=0.25 bins contact more often.",
            "The traced envs keep large vertical TCP-target error at the final target, so side-center contact height is not reliably reached.",
            "Next work should diagnose y+ target path, lateral/height offsets, reach, and actuator tracking before RL/data scale-up.",
        ],
    }
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--trace_csv", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()

    audit = build_audit(args.summary_csv, args.trace_csv)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    contact = audit["contact_group"]
    no_contact = audit["no_contact_group"]
    bins = {row["name"]: row for row in audit["position_bins"]}
    print(
        "yplus_geometry line1 "
        f"trials={audit['trials']} trace_rows={audit['trace_rows']} "
        f"contact_n={contact['n']} no_contact_n={no_contact['n']}"
    )
    print(
        "yplus_geometry line2 "
        f"contact_max_disp_mean={contact['means']['max_disp_along_push_m']:.9f} "
        f"no_contact_max_disp_mean={no_contact['means']['max_disp_along_push_m']:.9f} "
        f"contact_final_tcp_err={contact['means']['final_tcp_target_err_m']:.9f} "
        f"no_contact_final_tcp_err={no_contact['means']['final_tcp_target_err_m']:.9f}"
    )
    print(
        "yplus_geometry line3 "
        f"y_le_0_contact={bins['cube_y0_m<=0']['contact_evidence_rate']:.6f} "
        f"y_gt_0_contact={bins['cube_y0_m>0']['contact_evidence_rate']:.6f} "
        f"x_lt_025_contact={bins['cube_x0_m<0.25']['contact_evidence_rate']:.6f} "
        f"x_ge_025_contact={bins['cube_x0_m>=0.25']['contact_evidence_rate']:.6f}"
    )
    print("yplus_geometry line4 verdict=LOCAL_DIAG_NEXT_GEOMETRY_REACH_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
