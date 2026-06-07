"""Local per-window diagnostic for the 10cm y+ Tier C quality issue.

This is a posthoc audit only. It reads existing reaction-window JSON/CSV files
and compares y+ Tier C windows against x-/x+/y- Tier B windows by IK demand,
clipping, actuator follow, contact timing, and target/TCP error.

No IsaacLab app, GPU runtime, training, dataset generation, robot control, SSH,
or trace mutation is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_AUDITS = (
    LOG_DIR / "cube10cm_reaction_window_seed949_audit.json",
    LOG_DIR / "cube10cm_reaction_window_seed950_audit.json",
    LOG_DIR / "cube10cm_reaction_window_seed957_audit.json",
    LOG_DIR / "cube10cm_reaction_window_seed958_audit.json",
    LOG_DIR / "cube10cm_reaction_window_seed959_audit.json",
    LOG_DIR / "cube10cm_reaction_window_seed960_audit.json",
    LOG_DIR / "cube10cm_reaction_window_seed961_audit.json",
)
JOINT_IDS = range(5)


def _float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        raw = row.get(key, "")
        if raw is None or raw == "":
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        raw = row.get(key, "")
        if raw is None or raw == "":
            return default
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _mean(values: list[float]) -> float:
    vals = _finite(values)
    return sum(vals) / len(vals) if vals else 0.0


def _percentile(values: list[float], q: float) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _max_abs(row: dict[str, Any], prefix: str) -> float:
    values = [_float(row, f"{prefix}_{joint}_rad") for joint in JOINT_IDS]
    vals = [abs(v) for v in values if math.isfinite(v)]
    return max(vals) if vals else 0.0


def _joint_abs(row: dict[str, Any], prefix: str, joint: int) -> float:
    value = _float(row, f"{prefix}_{joint}_rad")
    return abs(value) if math.isfinite(value) else 0.0


def _direction(row: dict[str, Any]) -> str:
    dx = _float(row, "push_dx")
    dy = _float(row, "push_dy")
    if abs(dx) >= abs(dy):
        return "x+" if dx >= 0 else "x-"
    return "y+" if dy >= 0 else "y-"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _window_key(row: dict[str, Any], fallback_index: int) -> tuple[int, int]:
    env_id = _int(row, "env_id", fallback_index)
    window_id = _int(row, "reaction_window_id", fallback_index)
    return env_id, window_id


def _per_window_from_rows(
    *,
    audit_name: str,
    rows: list[dict[str, str]],
    joint_step_cap_rad: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_window_key(row, index)].append(row)

    windows: list[dict[str, Any]] = []
    for (env_id, window_id), window_rows in sorted(grouped.items()):
        first = window_rows[0]
        direction = _direction(first)
        tier = str(first.get("reaction_window_quality_tier", ""))
        raw_max = [_max_abs(row, "raw_delta") for row in window_rows]
        clipped_max = [_max_abs(row, "clipped_delta") for row in window_rows]
        follow_max = [_max_abs(row, "joint_follow_err") for row in window_rows]
        clip_any = [_float(row, "clip_any", 0.0) for row in window_rows]
        clip_joint_count = [_float(row, "clip_joint_count", 0.0) for row in window_rows]
        tcp_err_after = [_float(row, "tcp_target_err_after_m") for row in window_rows]
        link5_err_after = [_float(row, "link5_target_err_after_m") for row in window_rows]
        tcp_xy_err = []
        tcp_z_err = []
        link5_xy_err = []
        link5_z_err = []
        for row in window_rows:
            tcp_dx = _float(row, "target_x_m") - _float(row, "tcp_x_after_m")
            tcp_dy = _float(row, "target_y_m") - _float(row, "tcp_y_after_m")
            tcp_dz = _float(row, "target_z_m") - _float(row, "tcp_z_after_m")
            link5_dx = _float(row, "link5_target_x_m") - _float(row, "link5_x_after_m")
            link5_dy = _float(row, "link5_target_y_m") - _float(row, "link5_y_after_m")
            link5_dz = _float(row, "link5_target_z_m") - _float(row, "link5_z_after_m")
            if all(math.isfinite(v) for v in (tcp_dx, tcp_dy)):
                tcp_xy_err.append(math.hypot(tcp_dx, tcp_dy))
            if math.isfinite(tcp_dz):
                tcp_z_err.append(abs(tcp_dz))
            if all(math.isfinite(v) for v in (link5_dx, link5_dy)):
                link5_xy_err.append(math.hypot(link5_dx, link5_dy))
            if math.isfinite(link5_dz):
                link5_z_err.append(abs(link5_dz))

        raw_joint_p95 = {
            f"raw_delta_j{joint}_p95": _percentile(
                [_joint_abs(row, "raw_delta", joint) for row in window_rows],
                0.95,
            )
            for joint in JOINT_IDS
        }
        worst_raw_joint = max(JOINT_IDS, key=lambda joint: raw_joint_p95[f"raw_delta_j{joint}_p95"])
        anchor_step = _int(first, "reaction_window_anchor_step", -1)
        start_step = _int(first, "reaction_window_start_step", -1)
        end_step = _int(first, "reaction_window_end_step", -1)
        first_contact_steps = [
            _int(row, "first_contact_step", -1)
            for row in window_rows
            if _int(row, "first_contact_step", -1) >= 0
        ]
        first_contact_step = min(first_contact_steps) if first_contact_steps else anchor_step
        measured_contact_rows = sum(_int(row, "measured_contact_now", 0) for row in window_rows)
        phase_at_anchor = 0.0
        for row in window_rows:
            if _int(row, "step", -9999) == anchor_step:
                phase_at_anchor = _float(row, "phase_alpha", 0.0)
                break

        follow_p95_rad = _percentile(follow_max, 0.95)
        follow_p95_to_cap = follow_p95_rad / joint_step_cap_rad if joint_step_cap_rad > 0 else 0.0
        window = {
            "audit": audit_name,
            "env_id": env_id,
            "reaction_window_id": window_id,
            "direction": direction,
            "quality_tier": tier,
            "accepted": _int(first, "reaction_window_contract_pass", 0) == 1,
            "rows": len(window_rows),
            "joint_step_cap_rad": joint_step_cap_rad,
            "anchor_step": anchor_step,
            "start_step": start_step,
            "end_step": end_step,
            "first_contact_step": first_contact_step,
            "anchor_minus_start_steps": anchor_step - start_step,
            "end_minus_anchor_steps": end_step - anchor_step,
            "measured_contact_row_rate": measured_contact_rows / len(window_rows) if window_rows else 0.0,
            "phase_alpha_at_anchor": phase_at_anchor,
            "clip_any_rate": _mean(clip_any),
            "clip_joint_count_mean": _mean(clip_joint_count),
            "raw_delta_abs_max_mean": _mean(raw_max),
            "raw_delta_abs_max_p95": _percentile(raw_max, 0.95),
            "raw_delta_abs_max_max": max(_finite(raw_max), default=0.0),
            "clipped_delta_abs_max_mean": _mean(clipped_max),
            "clipped_delta_abs_max_p95": _percentile(clipped_max, 0.95),
            "joint_follow_abs_max_mean": _mean(follow_max),
            "joint_follow_abs_max_p95": follow_p95_rad,
            "joint_follow_p95_to_cap": follow_p95_to_cap,
            "tcp_target_err_after_mean_m": _mean(tcp_err_after),
            "tcp_target_err_after_p95_m": _percentile(tcp_err_after, 0.95),
            "tcp_target_xy_err_p95_m": _percentile(tcp_xy_err, 0.95),
            "tcp_target_z_err_p95_m": _percentile(tcp_z_err, 0.95),
            "link5_target_err_after_mean_m": _mean(link5_err_after),
            "link5_target_err_after_p95_m": _percentile(link5_err_after, 0.95),
            "link5_target_xy_err_p95_m": _percentile(link5_xy_err, 0.95),
            "link5_target_z_err_p95_m": _percentile(link5_z_err, 0.95),
            "disp_xy_max_m": max(_finite([_float(row, "disp_xy_m") for row in window_rows]), default=0.0),
            "disp_along_push_max_m": max(
                _finite([_float(row, "disp_along_push_m") for row in window_rows]),
                default=0.0,
            ),
            "cube_speed_max_mps": max(_finite([_float(row, "cube_speed_mps") for row in window_rows]), default=0.0),
            "tip_angle_max_deg": max(_finite([_float(row, "tip_angle_deg") for row in window_rows]), default=0.0),
            "worst_raw_delta_joint": worst_raw_joint,
            **raw_joint_p95,
        }
        windows.append(window)
    return windows


def _group_name(window: dict[str, Any]) -> str:
    direction = str(window["direction"])
    tier = str(window["quality_tier"])
    if direction == "y+" and tier == "C_REACTION_VALID_FOLLOW_LAG":
        return "target_yplus_tier_c"
    if tier == "B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH":
        return f"baseline_tier_b_{direction}"
    return f"context_{direction}_{tier}"


def _summarize_group(name: str, windows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "rows",
        "anchor_step",
        "first_contact_step",
        "anchor_minus_start_steps",
        "end_minus_anchor_steps",
        "measured_contact_row_rate",
        "phase_alpha_at_anchor",
        "clip_any_rate",
        "clip_joint_count_mean",
        "raw_delta_abs_max_mean",
        "raw_delta_abs_max_p95",
        "raw_delta_abs_max_max",
        "clipped_delta_abs_max_mean",
        "clipped_delta_abs_max_p95",
        "joint_follow_abs_max_mean",
        "joint_follow_abs_max_p95",
        "joint_follow_p95_to_cap",
        "tcp_target_err_after_mean_m",
        "tcp_target_err_after_p95_m",
        "tcp_target_xy_err_p95_m",
        "tcp_target_z_err_p95_m",
        "link5_target_err_after_mean_m",
        "link5_target_err_after_p95_m",
        "link5_target_xy_err_p95_m",
        "link5_target_z_err_p95_m",
        "disp_xy_max_m",
        "disp_along_push_max_m",
        "cube_speed_max_mps",
        "tip_angle_max_deg",
    ]
    out: dict[str, Any] = {
        "group": name,
        "window_count": len(windows),
        "directions": ",".join(sorted({str(w["direction"]) for w in windows})),
        "tiers": ",".join(sorted({str(w["quality_tier"]) for w in windows})),
        "audits": ",".join(sorted({str(w["audit"]) for w in windows})),
    }
    for metric in metrics:
        vals = [float(w[metric]) for w in windows if metric in w and math.isfinite(float(w[metric]))]
        out[f"{metric}_mean"] = _mean(vals)
        out[f"{metric}_p95"] = _percentile(vals, 0.95)
    worst_counts: dict[int, int] = defaultdict(int)
    for window in windows:
        worst_counts[int(window["worst_raw_delta_joint"])] += 1
    out["worst_raw_delta_joint_mode"] = max(worst_counts, key=worst_counts.get) if worst_counts else -1
    out["worst_raw_delta_joint_mode_count"] = max(worst_counts.values()) if worst_counts else 0
    return out


def _compare(target: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def ratio(key: str) -> float:
        base = float(baseline.get(key, 0.0))
        if base == 0.0:
            return 0.0
        return float(target.get(key, 0.0)) / base

    return {
        "baseline_group": baseline["group"],
        "target_group": target["group"],
        "raw_delta_abs_max_p95_ratio": ratio("raw_delta_abs_max_p95_p95"),
        "clipped_delta_abs_max_p95_ratio": ratio("clipped_delta_abs_max_p95_p95"),
        "joint_follow_p95_to_cap_ratio": ratio("joint_follow_p95_to_cap_p95"),
        "tcp_target_err_after_p95_ratio": ratio("tcp_target_err_after_p95_m_p95"),
        "tcp_target_z_err_p95_ratio": ratio("tcp_target_z_err_p95_m_p95"),
        "link5_target_err_after_p95_ratio": ratio("link5_target_err_after_p95_m_p95"),
        "disp_xy_max_ratio": ratio("disp_xy_max_m_mean"),
        "cube_speed_max_ratio": ratio("cube_speed_max_mps_mean"),
    }


def _verdict(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    target = summaries.get("target_yplus_tier_c")
    b_groups = [v for k, v in summaries.items() if k.startswith("baseline_tier_b_")]
    if target is None or not b_groups:
        return {
            "supports_yplus_ik_demand_geometry_hypothesis": False,
            "reasons": ["missing target y+ Tier C group or Tier B baselines"],
        }

    b_all = _summarize_group(
        "baseline_tier_b_all_non_yplus",
        [
            window
            for group in b_groups
            for window in group.get("_windows", [])
            if str(window.get("direction")) != "y+"
        ],
    )
    raw_ratio = _compare(target, b_all)["raw_delta_abs_max_p95_ratio"]
    follow_ratio = _compare(target, b_all)["joint_follow_p95_to_cap_ratio"]
    tcp_ratio = _compare(target, b_all)["tcp_target_err_after_p95_ratio"]
    contact_ratio = _compare(target, b_all)["disp_xy_max_ratio"]
    reasons: list[str] = []
    if raw_ratio > 1.25:
        reasons.append(f"y+ raw IK demand p95 is {raw_ratio:.3f}x Tier B non-y+ baseline")
    if follow_ratio > 1.05:
        reasons.append(f"y+ follow/cap p95 is {follow_ratio:.3f}x Tier B non-y+ baseline")
    if tcp_ratio > 1.25:
        reasons.append(f"y+ target/TCP error p95 is {tcp_ratio:.3f}x Tier B non-y+ baseline")
    if contact_ratio > 1.0:
        reasons.append(f"y+ reaction displacement is not weak: disp max mean is {contact_ratio:.3f}x baseline")
    supports_simple_raw_demand = raw_ratio > 1.25 and follow_ratio > 1.05
    supports_geometry_follow_coupling = raw_ratio <= 1.25 and follow_ratio > 1.05 and contact_ratio > 1.0
    if supports_geometry_follow_coupling:
        reasons.append(
            "y+ follow lag rises despite non-elevated raw IK amplitude, pointing to direction/contact geometry "
            "or load timing rather than a simple bigger-target solve"
        )
    return {
        "supports_simple_raw_ik_demand_hypothesis": supports_simple_raw_demand,
        "supports_yplus_geometry_follow_coupling_hypothesis": supports_geometry_follow_coupling,
        "baseline_group": "baseline_tier_b_all_non_yplus",
        "raw_delta_abs_max_p95_ratio": raw_ratio,
        "joint_follow_p95_to_cap_ratio": follow_ratio,
        "tcp_target_err_after_p95_ratio": tcp_ratio,
        "disp_xy_max_ratio": contact_ratio,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_json", action="append", type=Path, default=None)
    parser.add_argument(
        "--out_json",
        type=Path,
        default=LOG_DIR / "cube10cm_yplus_tierc_failure_diagnostic_existing_seeds.json",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=LOG_DIR / "cube10cm_yplus_tierc_failure_diagnostic_existing_seeds.csv",
    )
    parser.add_argument(
        "--out_window_csv",
        type=Path,
        default=LOG_DIR / "cube10cm_yplus_tierc_failure_diagnostic_existing_seeds_windows.csv",
    )
    parser.add_argument(
        "--out_summary",
        type=Path,
        default=LOG_DIR / "cube10cm_yplus_tierc_failure_diagnostic_existing_seeds_summary.out",
    )
    args = parser.parse_args()

    audit_paths = args.audit_json or list(DEFAULT_AUDITS)
    windows: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []
    for audit_path in audit_paths:
        audit = _load_json(audit_path)
        window_csv = REPO / str(audit["out_window_csv"])
        summary_json = REPO / str(audit["summary_json"])
        summary = _load_json(summary_json)
        joint_step_cap_rad = float(summary.get("max_diffik_joint_step_rad", 0.0))
        rows = _load_rows(window_csv)
        audit_windows = _per_window_from_rows(
            audit_name=audit_path.name,
            rows=rows,
            joint_step_cap_rad=joint_step_cap_rad,
        )
        windows.extend(audit_windows)
        source_files.append(
            {
                "audit_json": str(audit_path),
                "window_csv": str(window_csv),
                "summary_json": str(summary_json),
            }
        )

    grouped_windows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        grouped_windows[_group_name(window)].append(window)

    summaries: dict[str, dict[str, Any]] = {}
    for group, group_windows in sorted(grouped_windows.items()):
        summary = _summarize_group(group, group_windows)
        summary["_windows"] = group_windows
        summaries[group] = summary

    tier_b_non_yplus = [
        window
        for group, group_windows in grouped_windows.items()
        if group.startswith("baseline_tier_b_")
        for window in group_windows
        if str(window.get("direction")) != "y+"
    ]
    if tier_b_non_yplus:
        summary = _summarize_group("baseline_tier_b_all_non_yplus", tier_b_non_yplus)
        summary["_windows"] = tier_b_non_yplus
        summaries["baseline_tier_b_all_non_yplus"] = summary

    comparison_target = summaries.get("target_yplus_tier_c")
    comparisons = []
    if comparison_target is not None:
        for group, summary in sorted(summaries.items()):
            if group == "target_yplus_tier_c" or not group.startswith("baseline_tier_b_"):
                continue
            comparisons.append(_compare(comparison_target, summary))

    csv_rows = []
    for group, summary in sorted(summaries.items()):
        row = {k: v for k, v in summary.items() if k != "_windows"}
        csv_rows.append(row)
    fieldnames = sorted({key for row in csv_rows for key in row})
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    detail_rows = sorted(
        windows,
        key=lambda row: (
            str(row["direction"]),
            str(row["quality_tier"]),
            str(row["audit"]),
            int(row["env_id"]),
        ),
    )
    detail_fields = sorted({key for row in detail_rows for key in row})
    with args.out_window_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detail_rows)

    public_summaries = {
        group: {k: v for k, v in summary.items() if k != "_windows"}
        for group, summary in sorted(summaries.items())
    }
    result = {
        "artifact_type": "cube10cm_yplus_tierc_failure_diagnostic_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_posthoc_only": True,
        "no_gpu_isaaclab_training_dataset_ssh": True,
        "primary_question": "is y+ Tier C driven by target/IK demand geometry rather than actuator strength?",
        "target_group": "target_yplus_tier_c",
        "baseline_groups": sorted(k for k in summaries if k.startswith("baseline_tier_b_")),
        "source_files": source_files,
        "window_count": len(windows),
        "group_summaries": public_summaries,
        "comparisons": comparisons,
        "verdict": _verdict(summaries),
        "out_group_csv": str(args.out_csv),
        "out_window_csv": str(args.out_window_csv),
        "out_summary": str(args.out_summary),
        "do_not_start": ["GPU_runtime", "1024_10240_scaleup", "dataset_generation", "PPO_RL", "VLA", "TrackA"],
    }
    target = public_summaries.get("target_yplus_tier_c", {})
    baseline_all = public_summaries.get("baseline_tier_b_all_non_yplus", {})
    baseline_xpos = public_summaries.get("baseline_tier_b_x+", {})
    baseline_xneg = public_summaries.get("baseline_tier_b_x-", {})
    baseline_yneg = public_summaries.get("baseline_tier_b_y-", {})
    verdict = result["verdict"]
    summary_lines = [
        "line1 artifact=cube10cm_yplus_tierc_failure_diagnostic_v1 local_posthoc_only=YES gpu_runtime=NO dataset_generation=NO",
        (
            "line2 target_yplus_tier_c "
            f"windows={target.get('window_count', 0)} "
            f"follow_p95_to_cap_p95={target.get('joint_follow_p95_to_cap_p95', 0.0):.9f} "
            f"raw_delta_abs_max_p95_p95={target.get('raw_delta_abs_max_p95_p95', 0.0):.9f} "
            f"tcp_target_err_after_p95_p95_m={target.get('tcp_target_err_after_p95_m_p95', 0.0):.9f} "
            f"disp_xy_max_mean_m={target.get('disp_xy_max_m_mean', 0.0):.9f} "
            f"anchor_step_mean={target.get('anchor_step_mean', 0.0):.6f} "
            f"phase_alpha_anchor_mean={target.get('phase_alpha_at_anchor_mean', 0.0):.9f}"
        ),
        (
            "line3 baseline_tier_b_all_non_yplus "
            f"windows={baseline_all.get('window_count', 0)} "
            f"follow_p95_to_cap_p95={baseline_all.get('joint_follow_p95_to_cap_p95', 0.0):.9f} "
            f"raw_delta_abs_max_p95_p95={baseline_all.get('raw_delta_abs_max_p95_p95', 0.0):.9f} "
            f"tcp_target_err_after_p95_p95_m={baseline_all.get('tcp_target_err_after_p95_m_p95', 0.0):.9f} "
            f"disp_xy_max_mean_m={baseline_all.get('disp_xy_max_m_mean', 0.0):.9f} "
            f"anchor_step_mean={baseline_all.get('anchor_step_mean', 0.0):.6f} "
            f"phase_alpha_anchor_mean={baseline_all.get('phase_alpha_at_anchor_mean', 0.0):.9f}"
        ),
        (
            "line4 yplus_vs_tier_b_all_non_yplus "
            f"raw_delta_ratio={verdict.get('raw_delta_abs_max_p95_ratio', 0.0):.9f} "
            f"follow_ratio={verdict.get('joint_follow_p95_to_cap_ratio', 0.0):.9f} "
            f"tcp_err_ratio={verdict.get('tcp_target_err_after_p95_ratio', 0.0):.9f} "
            f"disp_xy_ratio={verdict.get('disp_xy_max_ratio', 0.0):.9f} "
            f"anchor_step_delta={target.get('anchor_step_mean', 0.0) - baseline_all.get('anchor_step_mean', 0.0):.6f} "
            f"phase_alpha_delta={target.get('phase_alpha_at_anchor_mean', 0.0) - baseline_all.get('phase_alpha_at_anchor_mean', 0.0):.9f}"
        ),
        (
            "line5 baseline_tier_b_xplus "
            f"windows={baseline_xpos.get('window_count', 0)} "
            f"follow_p95_to_cap_p95={baseline_xpos.get('joint_follow_p95_to_cap_p95', 0.0):.9f} "
            f"raw_delta_abs_max_p95_p95={baseline_xpos.get('raw_delta_abs_max_p95_p95', 0.0):.9f} "
            f"tcp_target_err_after_p95_p95_m={baseline_xpos.get('tcp_target_err_after_p95_m_p95', 0.0):.9f} "
            f"disp_xy_max_mean_m={baseline_xpos.get('disp_xy_max_m_mean', 0.0):.9f} "
            f"anchor_step_mean={baseline_xpos.get('anchor_step_mean', 0.0):.6f} "
            f"phase_alpha_anchor_mean={baseline_xpos.get('phase_alpha_at_anchor_mean', 0.0):.9f}"
        ),
        (
            "line6 baseline_tier_b_xminus "
            f"windows={baseline_xneg.get('window_count', 0)} "
            f"follow_p95_to_cap_p95={baseline_xneg.get('joint_follow_p95_to_cap_p95', 0.0):.9f} "
            f"raw_delta_abs_max_p95_p95={baseline_xneg.get('raw_delta_abs_max_p95_p95', 0.0):.9f} "
            f"tcp_target_err_after_p95_p95_m={baseline_xneg.get('tcp_target_err_after_p95_m_p95', 0.0):.9f} "
            f"disp_xy_max_mean_m={baseline_xneg.get('disp_xy_max_m_mean', 0.0):.9f} "
            f"anchor_step_mean={baseline_xneg.get('anchor_step_mean', 0.0):.6f} "
            f"phase_alpha_anchor_mean={baseline_xneg.get('phase_alpha_at_anchor_mean', 0.0):.9f}"
        ),
        (
            "line7 baseline_tier_b_yminus "
            f"windows={baseline_yneg.get('window_count', 0)} "
            f"follow_p95_to_cap_p95={baseline_yneg.get('joint_follow_p95_to_cap_p95', 0.0):.9f} "
            f"raw_delta_abs_max_p95_p95={baseline_yneg.get('raw_delta_abs_max_p95_p95', 0.0):.9f} "
            f"tcp_target_err_after_p95_p95_m={baseline_yneg.get('tcp_target_err_after_p95_m_p95', 0.0):.9f} "
            f"disp_xy_max_mean_m={baseline_yneg.get('disp_xy_max_m_mean', 0.0):.9f} "
            f"anchor_step_mean={baseline_yneg.get('anchor_step_mean', 0.0):.6f} "
            f"phase_alpha_anchor_mean={baseline_yneg.get('phase_alpha_at_anchor_mean', 0.0):.9f}"
        ),
        (
            "line8 verdict "
            f"supports_simple_raw_ik_demand={verdict.get('supports_simple_raw_ik_demand_hypothesis')} "
            f"supports_yplus_geometry_follow_coupling={verdict.get('supports_yplus_geometry_follow_coupling_hypothesis')}"
        ),
        (
            "line9 implication=no_bigger_cap_no_stiffness_no_blind_actuator_sweep; "
            "next_local_question=why_yplus_contacts_so_early_and_moves_cube_10x_more_while_follow_lag_exceeds_tier_b"
        ),
    ]
    args.out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["verdict"], indent=2, sort_keys=True))
    print(f"wrote_json={args.out_json}")
    print(f"wrote_csv={args.out_csv}")
    print(f"wrote_window_csv={args.out_window_csv}")
    print(f"wrote_summary={args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
