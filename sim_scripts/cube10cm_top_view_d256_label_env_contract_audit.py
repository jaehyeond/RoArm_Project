#!/usr/bin/env python3
"""Audit D256 visual labels against current tap overshoot/env semantics.

This is a data-only audit. It does not launch Isaac Lab and does not run PPO.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    REPO
    / "claudedocs"
    / "runtime_logs"
    / "20260526_cube3cm_push_rollout_probe_20480"
)
D242_ROOT = RUNTIME_ROOT / "cube10cm_top_view_visual_0_999_d242"
DEFAULT_MANIFEST = D242_ROOT / "label_package_d248" / "episode_split_manifest.csv"
DEFAULT_TEACHER_PROBE = (
    RUNTIME_ROOT
    / "d256_reset_bin_teacher_probe_d287_20bins_maxdelta0040_corrected"
    / "tap10cm"
    / "d256_reset_bin_actor_probe_summary_d286.json"
)
DEFAULT_ACTOR_PROBE = (
    RUNTIME_ROOT
    / "d256_reset_bin_actor_probe_d287_maxdelta0040_corrected"
    / "tap10cm"
    / "d256_reset_bin_actor_probe_summary_d286.json"
)
DEFAULT_OUT_DIR = RUNTIME_ROOT / "d256_label_env_contract_audit_d288"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def _quantile(sorted_vals: list[float], frac: float) -> float | None:
    if not sorted_vals:
        return None
    idx = round(frac * (len(sorted_vals) - 1))
    idx = min(len(sorted_vals) - 1, max(0, int(idx)))
    return float(sorted_vals[idx])


def _value_stats(values: list[float]) -> dict[str, float | int | None]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(vals),
        "min": float(vals[0]),
        "p50": _quantile(vals, 0.50),
        "p90": _quantile(vals, 0.90),
        "p95": _quantile(vals, 0.95),
        "p99": _quantile(vals, 0.99),
        "max": float(vals[-1]),
    }


def _aggregate_split(rows: list[dict[str, str]], split: str, overshoot_threshold_m: float) -> dict[str, Any]:
    subset = [row for row in rows if row.get("package_subsplit") == split]
    stats_fields = [
        "max_tap_disp_xy_m",
        "max_tap_disp_along_m",
        "final_tap_disp_xy_m",
        "final_tap_disp_along_m",
        "max_tap_speed_mps",
        "centroid_error_px_p95",
    ]
    result: dict[str, Any] = {
        "package_subsplit": split,
        "episode_count": len(subset),
        "label_status_values": sorted({row.get("label_status", "") for row in subset}),
        "overshoot_seen_any_count": sum(_int(row, "overshoot_seen_any") for row in subset),
        "contact_seen_any_count": sum(_int(row, "contact_seen_any") for row in subset),
        "reaction_seen_any_count": sum(_int(row, "reaction_seen_any") for row in subset),
        "camera_contract_pass_count": sum(_int(row, "camera_contract_pass") for row in subset),
        "max_xy_ge_env_overshoot_threshold_count": sum(
            _float(row, "max_tap_disp_xy_m") >= overshoot_threshold_m for row in subset
        ),
        "max_along_ge_env_overshoot_threshold_count": sum(
            _float(row, "max_tap_disp_along_m") >= overshoot_threshold_m for row in subset
        ),
    }
    for field in stats_fields:
        result[field] = _value_stats([_float(row, field) for row in subset if row.get(field, "") != ""])
    return result


def _probe_ranges(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": _rel(path)}
    data = json.loads(path.read_text())
    bins = data.get("bins", [])
    numeric_fields = [
        "tap_useful_seen_rate_log_max_trace",
        "tap_overshoot_seen_rate_log_max_trace",
        "joint_delta_cap_rate_max_trace",
        "action_abs_max_trace_max",
        "action_abs_mean_trace_mean",
        "tap_max_disp_xy_mean_log_max_trace_m",
        "tap_max_disp_along_mean_log_max_trace_m",
        "tap_contact_vertical_offset_mean_m",
    ]
    ranges: dict[str, Any] = {}
    for field in numeric_fields:
        vals = [float(row[field]) for row in bins if isinstance(row, dict) and row.get(field) is not None]
        ranges[field] = _value_stats(vals)
    compact_bins = []
    for row in bins:
        if not isinstance(row, dict):
            continue
        compact_bins.append(
            {
                "bin_idx": row.get("bin_idx"),
                "episode_min": row.get("episode_min"),
                "episode_max": row.get("episode_max"),
                "d256_frame0_rows": row.get("d256_frame0_rows"),
                "useful_rate_log_max": row.get("tap_useful_seen_rate_log_max_trace"),
                "overshoot_rate_log_max": row.get("tap_overshoot_seen_rate_log_max_trace"),
                "joint_delta_cap_rate_max": row.get("joint_delta_cap_rate_max_trace"),
                "max_disp_xy_mean_log_max_m": row.get("tap_max_disp_xy_mean_log_max_trace_m"),
                "max_disp_along_mean_log_max_m": row.get("tap_max_disp_along_mean_log_max_trace_m"),
                "safe_for_next_smoke_candidate": row.get("safe_for_next_smoke_candidate"),
            }
        )
    return {
        "exists": True,
        "path": _rel(path),
        "verdict": data.get("verdict"),
        "exec_source": data.get("exec_source"),
        "diagnostic_class": data.get("diagnostic_class"),
        "num_envs": data.get("num_envs"),
        "bin_count": data.get("bin_count"),
        "safe_bins": data.get("safe_bins"),
        "issues": data.get("issues"),
        "ranges": ranges,
        "bins": compact_bins,
    }


def _fmt_stat(stats: dict[str, Any], key: str) -> str:
    s = stats[key]
    if s["count"] == 0:
        return "n/a"
    return (
        f"{s['min']:.6f} / {s['p50']:.6f} / {s['p90']:.6f} / "
        f"{s['p95']:.6f} / {s['p99']:.6f} / {s['max']:.6f}"
    )


def _range_minmax(probe: dict[str, Any], field: str) -> str:
    stats = probe.get("ranges", {}).get(field, {})
    if not stats or stats.get("count", 0) == 0:
        return "n/a"
    return f"{stats['min']:.6f}..{stats['max']:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--teacher_probe_json", type=Path, default=DEFAULT_TEACHER_PROBE)
    parser.add_argument("--actor_probe_json", type=Path, default=DEFAULT_ACTOR_PROBE)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overshoot_threshold_m", type=float, default=0.020)
    parser.add_argument("--artifact_tag", type=str, default="d288")
    args = parser.parse_args()

    with args.manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames or [])

    split_names = sorted({row.get("package_subsplit", "") for row in rows})
    split_stats = {
        split: _aggregate_split(rows, split, float(args.overshoot_threshold_m))
        for split in split_names
    }
    train = split_stats.get("train_clean_positive", {})
    eval_over = split_stats.get("eval_overshoot_diagnostic", {})
    teacher_probe = _probe_ranges(args.teacher_probe_json)
    actor_probe = _probe_ranges(args.actor_probe_json)

    train_episode_count = int(train.get("episode_count", 0))
    train_overshoot_count = int(train.get("overshoot_seen_any_count", -1))
    train_over_threshold_count = int(train.get("max_xy_ge_env_overshoot_threshold_count", -1))
    teacher_overshoot_stats = teacher_probe.get("ranges", {}).get("tap_overshoot_seen_rate_log_max_trace", {})
    teacher_overshoot_max = teacher_overshoot_stats.get("max")
    teacher_overshoot_nonzero = teacher_overshoot_max is not None and float(teacher_overshoot_max) > 0.0

    if (
        train_episode_count > 0
        and train_overshoot_count == 0
        and train_over_threshold_count == 0
        and teacher_overshoot_nonzero
    ):
        verdict = "D288_LABEL_CLEAN_TEACHER_ONLINE_CONTRACT_MISMATCH_CONFIRMED"
    elif train_overshoot_count > 0 or train_over_threshold_count > 0:
        verdict = "D288_LABEL_CONTRACT_NOT_STRICTLY_CLEAN_REVIEW_REQUIRED"
    else:
        verdict = "D288_LABEL_AUDIT_INCONCLUSIVE"

    summary = {
        "artifact": f"cube10cm_{args.artifact_tag}_d256_label_env_contract_audit",
        "status": "PASS_AUDIT_EXECUTED",
        "verdict": verdict,
        "no_ppo_learning": True,
        "isaac_lab_launched": False,
        "manifest": _rel(args.manifest),
        "manifest_columns": columns,
        "episode_count_total": len(rows),
        "overshoot_threshold_m": float(args.overshoot_threshold_m),
        "split_stats": split_stats,
        "teacher_probe": teacher_probe,
        "actor_probe": actor_probe,
        "interpretation": (
            "D256 train_clean_positive labels are clean under the same 0.020 m XY overshoot "
            "threshold, while D287 online teacher/actor probes still overshoot. This points "
            "to an online teacher/action execution contract problem, not a permissive label problem."
        ),
        "next_order": [
            "Do not run long PPO.",
            "Run/review D256 recorded-action replay in the live env.",
            "If replay is clean, rebuild or constrain the teacher/action bridge before actor distillation.",
            "If replay also overshoots, fix env physics/action application or label-env semantics first.",
            "Only after teacher-off/bin diagnostics pass, run tiny PPO smoke plus TensorBoard gate.",
        ],
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / f"d256_label_env_contract_audit_{args.artifact_tag}.json"
    summary_md = out_dir / f"d256_label_env_contract_audit_{args.artifact_tag}.md"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    train_stats = split_stats.get("train_clean_positive", {})
    over_stats = split_stats.get("eval_overshoot_diagnostic", {})
    summary_md.write_text(
        "# D288 D256 Label/Env Contract Audit\n\n"
        f"- verdict: `{verdict}`\n"
        f"- no PPO learning: `{summary['no_ppo_learning']}`\n"
        f"- Isaac Lab launched: `{summary['isaac_lab_launched']}`\n"
        f"- overshoot threshold: `{args.overshoot_threshold_m:.3f} m`\n"
        f"- manifest: `{summary['manifest']}`\n\n"
        "## D256 label split\n\n"
        f"- train_clean_positive episodes: `{train_stats.get('episode_count')}`\n"
        f"- train clean overshoot episodes: `{train_stats.get('overshoot_seen_any_count')}`\n"
        f"- train clean max_xy >= threshold: `{train_stats.get('max_xy_ge_env_overshoot_threshold_count')}`\n"
        f"- train clean contact/reaction: `{train_stats.get('contact_seen_any_count')}` / `{train_stats.get('reaction_seen_any_count')}`\n"
        f"- train clean max_tap_disp_xy_m min/p50/p90/p95/p99/max: `{_fmt_stat(train_stats, 'max_tap_disp_xy_m')}`\n"
        f"- train clean max_tap_disp_along_m min/p50/p90/p95/p99/max: `{_fmt_stat(train_stats, 'max_tap_disp_along_m')}`\n"
        f"- eval overshoot episodes: `{over_stats.get('episode_count')}`\n"
        f"- eval overshoot seen episodes: `{over_stats.get('overshoot_seen_any_count')}`\n"
        f"- eval overshoot max_tap_disp_xy_m min/p50/p90/p95/p99/max: `{_fmt_stat(over_stats, 'max_tap_disp_xy_m')}`\n\n"
        "## D287 online probes\n\n"
        f"- teacher probe verdict: `{teacher_probe.get('verdict')}`\n"
        f"- teacher safe bins: `{teacher_probe.get('safe_bins')}`\n"
        f"- teacher overshoot rate range: `{_range_minmax(teacher_probe, 'tap_overshoot_seen_rate_log_max_trace')}`\n"
        f"- teacher useful rate range: `{_range_minmax(teacher_probe, 'tap_useful_seen_rate_log_max_trace')}`\n"
        f"- actor probe verdict: `{actor_probe.get('verdict')}`\n"
        f"- actor safe bins: `{actor_probe.get('safe_bins')}`\n"
        f"- actor overshoot rate range: `{_range_minmax(actor_probe, 'tap_overshoot_seen_rate_log_max_trace')}`\n"
        f"- actor useful rate range: `{_range_minmax(actor_probe, 'tap_useful_seen_rate_log_max_trace')}`\n\n"
        "## Interpretation\n\n"
        f"{summary['interpretation']}\n\n"
        "## Next order\n\n"
        + "\n".join(f"{idx}. {item}" for idx, item in enumerate(summary["next_order"], start=1))
        + "\n"
    )

    print(
        "d256_label_env_contract_audit "
        f"verdict={verdict} "
        f"train_clean_episodes={train_episode_count} "
        f"train_clean_overshoot={train_overshoot_count} "
        f"teacher_overshoot_range={_range_minmax(teacher_probe, 'tap_overshoot_seen_rate_log_max_trace')} "
        f"summary={_rel(summary_json)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
