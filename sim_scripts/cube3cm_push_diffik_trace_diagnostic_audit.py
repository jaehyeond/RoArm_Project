"""Summarize the extended DiffIK trace diagnostic CSV.

This is a local posthoc tool. It does not run IsaacLab, train, generate data, or
touch the robot. Feed it a trace produced with --trace_diffik_diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ARM_JOINT_COUNT = 5
REQUIRED_TRACE_FIELDS = {
    "link5_body_idx",
    "jacobi_body_idx",
    "clip_joint_count",
    "clip_any",
    "clip_single_joint",
    "clip_all_joints",
    "clip_max_joint_name",
    "tcp_target_err_before_m",
    "tcp_target_err_after_m",
    "link5_target_err_before_m",
    "link5_target_err_after_m",
    "tcp_x_before_m",
    "tcp_x_after_m",
    "link5_x_before_m",
    "link5_x_after_m",
}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _rate(rows: list[dict[str, str]], key: str) -> float:
    return _mean([_float(row, key) for row in rows])


def _stats(rows: list[dict[str, str]], key: str) -> dict[str, float]:
    values = [_float(row, key) for row in rows]
    return {
        "mean": _mean(values),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "max": max(values, default=0.0),
    }


def _joint_stats(rows: list[dict[str, str]], prefix: str, suffix: str) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for idx in range(ARM_JOINT_COUNT):
        key = f"{prefix}_{idx}_{suffix}"
        vals = [abs(_float(row, key)) for row in rows]
        out.append(
            {
                "joint": idx,
                "mean": _mean(vals),
                "p95": _quantile(vals, 0.95),
                "max": max(vals, default=0.0),
            }
        )
    return out


def _joint_rate(rows: list[dict[str, str]], key_prefix: str) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for idx in range(ARM_JOINT_COUNT):
        key = f"{key_prefix}_{idx}"
        vals = [_float(row, key) for row in rows]
        out.append({"joint": idx, "rate": _mean(vals)})
    return out


def _worst_joint(stats: list[dict[str, float | int]], metric: str) -> dict[str, float | int]:
    if not stats:
        return {"joint": -1, metric: 0.0}
    return max(stats, key=lambda item: float(item.get(metric, 0.0)))


def _phase_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "clip_any_rate": 0.0,
            "clip_joint_count_mean": 0.0,
            "joint_step_scale_mean": 0.0,
            "worst_follow_joint": {"joint": -1, "mean": 0.0, "p95": 0.0, "max": 0.0},
            "worst_raw_delta_joint": {"joint": -1, "mean": 0.0, "p95": 0.0, "max": 0.0},
        }
    follow_stats = _joint_stats(rows, "joint_follow_err", "rad")
    raw_stats = _joint_stats(rows, "raw_delta", "rad")
    return {
        "rows": len(rows),
        "clip_any_rate": _rate(rows, "clip_any"),
        "clip_joint_count_mean": _mean([_float(row, "clip_joint_count") for row in rows]),
        "joint_step_scale_mean": _mean([_float(row, "joint_step_scale") for row in rows]),
        "worst_follow_joint": _worst_joint(follow_stats, "mean"),
        "worst_raw_delta_joint": _worst_joint(raw_stats, "mean"),
    }


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def _format_joint_rates(items: list[dict[str, float | int]]) -> str:
    return ",".join(f"j{int(item['joint'])}:{float(item['rate']):.3f}" for item in items)


def _format_joint_stats(items: list[dict[str, float | int]], metric: str = "mean") -> str:
    return ",".join(f"j{int(item['joint'])}:{float(item[metric]):.6f}" for item in items)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_csv", type=Path, required=True)
    ap.add_argument("--summary_json", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    rows = _load_rows(args.trace_csv)
    summary: dict[str, Any] = json.loads(args.summary_json.read_text())
    if not rows:
        raise RuntimeError(f"empty trace csv: {args.trace_csv}")

    header = set(rows[0].keys())
    missing = sorted(REQUIRED_TRACE_FIELDS - header)
    row_count_match = len(rows) == int(summary.get("trace_frame_count", -1))
    mechanism_ok = (
        summary.get("controller") == "IsaacLab_DifferentialIKController"
        and summary.get("training") is False
        and summary.get("dataset_generation") is False
        and summary.get("grasp_attach") is False
        and summary.get("rollout_object_posewrite") is False
        and int(summary.get("posewrite_calls_during_rollout", -1)) == 0
    )

    if missing:
        print(
            "diffik_trace_diag line1 "
            f"trace_rows={len(rows)} summary_trace_frame_count={summary.get('trace_frame_count')} "
            f"row_count_match={row_count_match} trace_diffik_diagnostics={summary.get('trace_diffik_diagnostics')} "
            f"missing_required={','.join(missing)}"
        )
        print("diffik_trace_diag line2 verdict=FAIL_MISSING_EXTENDED_TRACE_FIELDS")
        return 2

    tcp_before = _stats(rows, "tcp_target_err_before_m")
    tcp_after = _stats(rows, "tcp_target_err_after_m")
    link5_before = _stats(rows, "link5_target_err_before_m")
    link5_after = _stats(rows, "link5_target_err_after_m")
    tcp_improve_rate = _mean(
        [
            1.0 if _float(row, "tcp_target_err_after_m") < _float(row, "tcp_target_err_before_m") else 0.0
            for row in rows
        ]
    )
    link5_improve_rate = _mean(
        [
            1.0 if _float(row, "link5_target_err_after_m") < _float(row, "link5_target_err_before_m") else 0.0
            for row in rows
        ]
    )

    clip_name_counts = Counter(row.get("clip_max_joint_name", "") for row in rows)
    clip_rates = _joint_rate(rows, "clip_mask")
    follow_stats = _joint_stats(rows, "joint_follow_err", "rad")
    raw_stats = _joint_stats(rows, "raw_delta", "rad")
    clipped_stats = _joint_stats(rows, "clipped_delta", "rad")
    worst_follow = _worst_joint(follow_stats, "mean")
    worst_raw = _worst_joint(raw_stats, "mean")
    worst_clipped = _worst_joint(clipped_stats, "mean")
    top_clip_name, top_clip_count = clip_name_counts.most_common(1)[0]
    top_clip_rate = top_clip_count / len(rows)

    link5_body_idx_values = sorted({int(_float(row, "link5_body_idx")) for row in rows})
    jacobi_body_idx_values = sorted({int(_float(row, "jacobi_body_idx")) for row in rows})
    pre_stop_rows = [row for row in rows if int(_float(row, "contact_stop_seen")) == 0]
    post_stop_rows = [row for row in rows if int(_float(row, "contact_stop_seen")) == 1]
    phase_splits = {
        "pre_stop": _phase_summary(pre_stop_rows),
        "post_stop": _phase_summary(post_stop_rows),
    }

    likely_modes: list[str] = []
    if link5_after["mean"] < 0.03 and tcp_after["mean"] > 0.05:
        likely_modes.append("LINK5_TRACKS_BUT_TCP_OFFSET_OR_ORIENTATION_FAILS")
    if link5_after["mean"] > 0.05:
        likely_modes.append("LINK5_BODY_TARGET_NOT_REACHED")
    if _rate(rows, "clip_all_joints") > 0.50:
        likely_modes.append("ALL_JOINTS_CLIPPED_DOMINANT")
    elif _rate(rows, "clip_any") > 0.50:
        likely_modes.append("JOINT_STEP_CLIPPING_DOMINANT")
    if float(worst_follow["mean"]) > 0.02:
        likely_modes.append("ACTUATOR_TARGET_TRACKING_LAG")
    if not likely_modes:
        likely_modes.append("NO_SINGLE_DOMINANT_MODE_FROM_THRESHOLDS")

    audit_summary: dict[str, Any] = {
        "trace_rows": len(rows),
        "summary_trace_frame_count": summary.get("trace_frame_count"),
        "row_count_match": row_count_match,
        "mechanism_ok": mechanism_ok,
        "trace_diffik_diagnostics": summary.get("trace_diffik_diagnostics"),
        "link5_body_idx_values": link5_body_idx_values,
        "jacobi_body_idx_values": jacobi_body_idx_values,
        "tcp_before": tcp_before,
        "tcp_after": tcp_after,
        "link5_before": link5_before,
        "link5_after": link5_after,
        "tcp_improve_rate": tcp_improve_rate,
        "link5_improve_rate": link5_improve_rate,
        "clip_any_rate": _rate(rows, "clip_any"),
        "clip_single_joint_rate": _rate(rows, "clip_single_joint"),
        "clip_all_joints_rate": _rate(rows, "clip_all_joints"),
        "clip_joint_count_mean": _mean([_float(row, "clip_joint_count") for row in rows]),
        "clip_max_joint_name_mode": top_clip_name,
        "clip_max_joint_name_mode_rate": top_clip_rate,
        "clip_rates_by_joint": clip_rates,
        "joint_follow_err_abs": follow_stats,
        "raw_delta_abs": raw_stats,
        "clipped_delta_abs": clipped_stats,
        "worst_follow_joint": worst_follow,
        "worst_raw_delta_joint": worst_raw,
        "worst_clipped_delta_joint": worst_clipped,
        "phase_splits": phase_splits,
        "likely_modes": likely_modes,
    }

    print(
        "diffik_trace_diag line1 "
        f"trace_rows={len(rows)} summary_trace_frame_count={summary.get('trace_frame_count')} "
        f"row_count_match={row_count_match} mechanism_ok={mechanism_ok} "
        f"trace_diffik_diagnostics={summary.get('trace_diffik_diagnostics')} "
        f"link5_body_idx_values={link5_body_idx_values} jacobi_body_idx_values={jacobi_body_idx_values}"
    )
    print(
        "diffik_trace_diag line2 tcp "
        f"before_mean_m={tcp_before['mean']:.9f} after_mean_m={tcp_after['mean']:.9f} "
        f"after_p95_m={tcp_after['p95']:.9f} improve_rate={tcp_improve_rate:.9f}"
    )
    print(
        "diffik_trace_diag line3 link5 "
        f"before_mean_m={link5_before['mean']:.9f} after_mean_m={link5_after['mean']:.9f} "
        f"after_p95_m={link5_after['p95']:.9f} improve_rate={link5_improve_rate:.9f}"
    )
    print(
        "diffik_trace_diag line4 clipping "
        f"clip_any_rate={audit_summary['clip_any_rate']:.9f} "
        f"clip_single_joint_rate={audit_summary['clip_single_joint_rate']:.9f} "
        f"clip_all_joints_rate={audit_summary['clip_all_joints_rate']:.9f} "
        f"clip_joint_count_mean={audit_summary['clip_joint_count_mean']:.9f} "
        f"mode_joint={top_clip_name} mode_rate={top_clip_rate:.9f} "
        f"by_joint={_format_joint_rates(clip_rates)}"
    )
    print(
        "diffik_trace_diag line5 actuator_follow "
        f"worst_joint={int(worst_follow['joint'])} worst_mean_rad={float(worst_follow['mean']):.9f} "
        f"worst_p95_rad={float(worst_follow['p95']):.9f} "
        f"by_joint_mean_rad={_format_joint_stats(follow_stats)}"
    )
    print(
        "diffik_trace_diag line6 deltas "
        f"worst_raw_joint={int(worst_raw['joint'])} worst_raw_mean_rad={float(worst_raw['mean']):.9f} "
        f"worst_clipped_joint={int(worst_clipped['joint'])} "
        f"worst_clipped_mean_rad={float(worst_clipped['mean']):.9f}"
    )
    print(
        "diffik_trace_diag line7 likely_modes="
        f"{'|'.join(likely_modes)} dataset_ready=NO learned_policy=NO track_a=NO"
    )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n")
        pre_stop = phase_splits["pre_stop"]
        post_stop = phase_splits["post_stop"]
        pre_follow = pre_stop["worst_follow_joint"]
        post_follow = post_stop["worst_follow_joint"]
        print(
            "diffik_trace_diag line8 phase_split "
            f"pre_stop_rows={pre_stop['rows']} pre_stop_clip_any={float(pre_stop['clip_any_rate']):.9f} "
            f"pre_stop_worst_follow_joint={int(pre_follow['joint'])} "
            f"pre_stop_worst_follow_mean_rad={float(pre_follow['mean']):.9f} "
            f"post_stop_rows={post_stop['rows']} post_stop_clip_any={float(post_stop['clip_any_rate']):.9f} "
            f"post_stop_worst_follow_joint={int(post_follow['joint'])} "
            f"post_stop_worst_follow_mean_rad={float(post_follow['mean']):.9f}"
        )
        print(f"diffik_trace_diag line9 out_json={args.out_json}")

    return 0 if row_count_match and mechanism_ok and bool(summary.get("trace_diffik_diagnostics")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
