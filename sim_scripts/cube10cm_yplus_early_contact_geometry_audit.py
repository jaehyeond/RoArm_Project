"""Local early-contact geometry audit for 10cm y+ reaction windows.

This posthoc audit checks whether y+ Tier C windows begin reacting during the
precontact/approach phase before the measured-contact anchor. It compares target
geometry, contact timing, pre-anchor object motion, and target/TCP geometry
against non-y+ Tier B reaction windows.

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
REACTION_DISP_M = 0.001
REACTION_TIP_DEG = 1.0
REACTION_SPEED_MPS = 0.02
REACTION_Z_DELTA_M = 0.002


def _float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        raw = row.get(key, "")
        if raw is None or raw == "":
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, Any], key: str, default: int = -1) -> int:
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _direction_from_row(row: dict[str, Any]) -> str:
    dx = _float(row, "push_dx")
    dy = _float(row, "push_dy")
    if abs(dx) >= abs(dy):
        return "x+" if dx >= 0 else "x-"
    return "y+" if dy >= 0 else "y-"


def _basis(row: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = _float(row, "push_dx", 0.0)
    dy = _float(row, "push_dy", 0.0)
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        return (1.0, 0.0), (0.0, 1.0)
    push = (dx / norm, dy / norm)
    lateral = (-push[1], push[0])
    return push, lateral


def _dot_xy(row: dict[str, Any], x_key: str, y_key: str, base_x_key: str, base_y_key: str, axis: tuple[float, float]) -> float:
    dx = _float(row, x_key) - _float(row, base_x_key)
    dy = _float(row, y_key) - _float(row, base_y_key)
    if not math.isfinite(dx) or not math.isfinite(dy):
        return 0.0
    return dx * axis[0] + dy * axis[1]


def _nearest_row(rows: list[dict[str, str]], step: int) -> dict[str, str] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(_int(row, "step", 0) - step))


def _first_step(rows: list[dict[str, str]], pred) -> int:
    for row in rows:
        if pred(row):
            return _int(row, "step", -1)
    return -1


def _max(rows: list[dict[str, str]], key: str) -> float:
    vals = _finite([_float(row, key) for row in rows])
    return max(vals) if vals else 0.0


def _group_name(window: dict[str, Any]) -> str:
    if window["direction"] == "y+" and window["quality_tier"] == "C_REACTION_VALID_FOLLOW_LAG":
        return "target_yplus_tier_c"
    if window["quality_tier"] == "B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH" and window["direction"] != "y+":
        return f"baseline_tier_b_{window['direction']}"
    return f"context_{window['direction']}_{window['quality_tier']}"


def _window_metrics(
    *,
    audit_path: Path,
    audit: dict[str, Any],
    summary: dict[str, Any],
    trace_rows_by_env: dict[int, list[dict[str, str]]],
    per_window: dict[str, Any],
) -> dict[str, Any]:
    env_id = int(per_window["env_id"])
    rows = sorted(trace_rows_by_env[env_id], key=lambda row: _int(row, "step", 0))
    first = rows[0]
    direction = _direction_from_row(first)
    anchor_step = int(per_window["anchor_step"])
    anchor_row = _nearest_row(rows, anchor_step) or first
    anchor_row_step = _int(anchor_row, "step", anchor_step)
    pre_rows_24 = [row for row in rows if anchor_step - 24 <= _int(row, "step", -1) < anchor_step]
    pre_rows_all = [row for row in rows if _int(row, "step", -1) < anchor_step]
    initial_cube_z = _float(first, "cube_z_m")

    first_push_phase_step = _first_step(rows, lambda row: _float(row, "phase_alpha", 0.0) > 0.0)
    first_near_step = _first_step(rows, lambda row: _int(row, "near_tcp_cube_now", 0) == 1)
    first_measured_contact_step = _first_step(rows, lambda row: _int(row, "measured_contact_now", 0) == 1)
    first_dispxy_step = _first_step(rows, lambda row: _float(row, "disp_xy_m", 0.0) >= REACTION_DISP_M)
    first_lateral_step = _first_step(rows, lambda row: _float(row, "lateral_abs_m", 0.0) >= REACTION_DISP_M)
    first_tip_step = _first_step(rows, lambda row: _float(row, "tip_angle_deg", 0.0) >= REACTION_TIP_DEG)
    first_speed_step = _first_step(rows, lambda row: _float(row, "cube_speed_mps", 0.0) >= REACTION_SPEED_MPS)
    first_z_step = _first_step(
        rows,
        lambda row: _float(row, "cube_z_m", initial_cube_z) - initial_cube_z >= REACTION_Z_DELTA_M,
    )
    reaction_steps = [s for s in (first_dispxy_step, first_lateral_step, first_tip_step, first_speed_step, first_z_step) if s >= 0]
    first_reaction_step = min(reaction_steps) if reaction_steps else -1
    first_reaction_row = _nearest_row(rows, first_reaction_step) if first_reaction_step >= 0 else None

    push_axis, lateral_axis = _basis(first)
    target_offset_along_initial = _dot_xy(first, "target_x_m", "target_y_m", "cube_x_m", "cube_y_m", push_axis)
    target_offset_lateral_initial = _dot_xy(first, "target_x_m", "target_y_m", "cube_x_m", "cube_y_m", lateral_axis)
    tcp_offset_along_anchor = _dot_xy(anchor_row, "tcp_x_after_m", "tcp_y_after_m", "cube_x_m", "cube_y_m", push_axis)
    tcp_offset_lateral_anchor = _dot_xy(anchor_row, "tcp_x_after_m", "tcp_y_after_m", "cube_x_m", "cube_y_m", lateral_axis)
    target_offset_along_anchor = _dot_xy(anchor_row, "target_x_m", "target_y_m", "cube_x_m", "cube_y_m", push_axis)
    target_offset_lateral_anchor = _dot_xy(anchor_row, "target_x_m", "target_y_m", "cube_x_m", "cube_y_m", lateral_axis)

    first_reaction_phase = _float(first_reaction_row, "phase_alpha", 0.0) if first_reaction_row else 0.0
    first_reaction_tcp_dist = _float(first_reaction_row, "tcp_cube_dist_m", 0.0) if first_reaction_row else 0.0
    first_reaction_disp = _float(first_reaction_row, "disp_xy_m", 0.0) if first_reaction_row else 0.0
    first_reaction_tip = _float(first_reaction_row, "tip_angle_deg", 0.0) if first_reaction_row else 0.0

    return {
        "audit": audit_path.name,
        "env_id": env_id,
        "direction": direction,
        "quality_tier": str(per_window["quality_tier"]),
        "anchor_step": anchor_step,
        "anchor_row_step": anchor_row_step,
        "anchor_phase_alpha_nearest": _float(anchor_row, "phase_alpha", 0.0),
        "first_push_phase_step": first_push_phase_step,
        "anchor_minus_first_push_phase_step": anchor_step - first_push_phase_step if first_push_phase_step >= 0 else 0,
        "first_near_step": first_near_step,
        "first_reaction_step": first_reaction_step,
        "first_measured_contact_step": first_measured_contact_step,
        "reaction_leads_measured_contact_steps": (
            first_measured_contact_step - first_reaction_step
            if first_measured_contact_step >= 0 and first_reaction_step >= 0
            else 0
        ),
        "near_leads_measured_contact_steps": (
            first_measured_contact_step - first_near_step
            if first_measured_contact_step >= 0 and first_near_step >= 0
            else 0
        ),
        "first_reaction_phase_alpha": first_reaction_phase,
        "first_reaction_tcp_cube_dist_m": first_reaction_tcp_dist,
        "first_reaction_disp_xy_m": first_reaction_disp,
        "first_reaction_tip_angle_deg": first_reaction_tip,
        "first_dispxy_step": first_dispxy_step,
        "first_lateral_step": first_lateral_step,
        "first_tip_step": first_tip_step,
        "first_speed_step": first_speed_step,
        "first_z_delta_step": first_z_step,
        "pre24_max_disp_xy_m": _max(pre_rows_24, "disp_xy_m"),
        "pre24_max_lateral_abs_m": _max(pre_rows_24, "lateral_abs_m"),
        "pre24_max_tip_angle_deg": _max(pre_rows_24, "tip_angle_deg"),
        "pre24_max_cube_speed_mps": _max(pre_rows_24, "cube_speed_mps"),
        "pre_all_max_disp_xy_m": _max(pre_rows_all, "disp_xy_m"),
        "pre_all_max_lateral_abs_m": _max(pre_rows_all, "lateral_abs_m"),
        "pre_all_max_tip_angle_deg": _max(pre_rows_all, "tip_angle_deg"),
        "pre_all_max_cube_speed_mps": _max(pre_rows_all, "cube_speed_mps"),
        "anchor_disp_xy_m": _float(anchor_row, "disp_xy_m", 0.0),
        "anchor_lateral_abs_m": _float(anchor_row, "lateral_abs_m", 0.0),
        "anchor_tip_angle_deg": _float(anchor_row, "tip_angle_deg", 0.0),
        "anchor_tcp_cube_dist_m": _float(anchor_row, "tcp_cube_dist_m", 0.0),
        "anchor_target_xy_dist_m": _float(anchor_row, "target_xy_dist_m", 0.0),
        "target_offset_along_initial_m": target_offset_along_initial,
        "target_offset_lateral_initial_m": target_offset_lateral_initial,
        "target_z_minus_cube_z_initial_m": _float(first, "target_z_m") - _float(first, "cube_z_m"),
        "tcp_z_minus_cube_z_anchor_m": _float(anchor_row, "tcp_z_after_m") - _float(anchor_row, "cube_z_m"),
        "target_z_minus_cube_z_anchor_m": _float(anchor_row, "target_z_m") - _float(anchor_row, "cube_z_m"),
        "tcp_offset_along_anchor_m": tcp_offset_along_anchor,
        "tcp_offset_lateral_anchor_m": tcp_offset_lateral_anchor,
        "target_offset_along_anchor_m": target_offset_along_anchor,
        "target_offset_lateral_anchor_m": target_offset_lateral_anchor,
        "tcp_center_height_offset_m": float(summary.get("applied_tcp_center_height_offset_mean_m", 0.0)),
        "precontact_clearance_m": float(summary.get("precontact_clearance_m", 0.0)),
        "push_through_m": float(summary.get("push_through_m", 0.0)),
        "base_lateral_offset_m": float(summary.get("base_lateral_offset_m", 0.0)),
        "contact_near_tcp_cube_dist_m": float(summary.get("contact_near_tcp_cube_dist_m", 0.0)),
    }


def _summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "anchor_step",
        "anchor_phase_alpha_nearest",
        "first_push_phase_step",
        "anchor_minus_first_push_phase_step",
        "first_near_step",
        "first_reaction_step",
        "first_measured_contact_step",
        "reaction_leads_measured_contact_steps",
        "near_leads_measured_contact_steps",
        "first_reaction_phase_alpha",
        "first_reaction_tcp_cube_dist_m",
        "first_reaction_disp_xy_m",
        "first_reaction_tip_angle_deg",
        "pre24_max_disp_xy_m",
        "pre24_max_lateral_abs_m",
        "pre24_max_tip_angle_deg",
        "pre24_max_cube_speed_mps",
        "pre_all_max_disp_xy_m",
        "pre_all_max_lateral_abs_m",
        "pre_all_max_tip_angle_deg",
        "pre_all_max_cube_speed_mps",
        "anchor_disp_xy_m",
        "anchor_lateral_abs_m",
        "anchor_tip_angle_deg",
        "anchor_tcp_cube_dist_m",
        "target_offset_along_initial_m",
        "target_offset_lateral_initial_m",
        "target_z_minus_cube_z_initial_m",
        "tcp_z_minus_cube_z_anchor_m",
        "target_z_minus_cube_z_anchor_m",
        "tcp_offset_along_anchor_m",
        "tcp_offset_lateral_anchor_m",
        "target_offset_along_anchor_m",
        "target_offset_lateral_anchor_m",
        "tcp_center_height_offset_m",
        "precontact_clearance_m",
        "push_through_m",
        "base_lateral_offset_m",
    ]
    out: dict[str, Any] = {
        "group": name,
        "window_count": len(rows),
        "directions": ",".join(sorted({str(row["direction"]) for row in rows})),
        "tiers": ",".join(sorted({str(row["quality_tier"]) for row in rows})),
        "audits": ",".join(sorted({str(row["audit"]) for row in rows})),
    }
    for metric in metrics:
        vals = [float(row[metric]) for row in rows if metric in row and math.isfinite(float(row[metric]))]
        out[f"{metric}_mean"] = _mean(vals)
        out[f"{metric}_p95"] = _percentile(vals, 0.95)
    return out


def _ratio(target: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    denom = float(baseline.get(key, 0.0))
    if denom == 0.0:
        return 0.0
    return float(target.get(key, 0.0)) / denom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_json", action="append", type=Path, default=None)
    parser.add_argument(
        "--out_json",
        type=Path,
        default=LOG_DIR / "cube10cm_yplus_early_contact_geometry_audit_existing_seeds.json",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=LOG_DIR / "cube10cm_yplus_early_contact_geometry_audit_existing_seeds.csv",
    )
    parser.add_argument(
        "--out_summary",
        type=Path,
        default=LOG_DIR / "cube10cm_yplus_early_contact_geometry_audit_existing_seeds_summary.out",
    )
    args = parser.parse_args()

    audit_paths = args.audit_json or list(DEFAULT_AUDITS)
    window_rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for audit_path in audit_paths:
        audit = _load_json(audit_path)
        summary_json = REPO / str(audit["summary_json"])
        trace_csv = REPO / str(audit["trace_csv"])
        summary = _load_json(summary_json)
        trace_rows = _load_rows(trace_csv)
        trace_rows_by_env: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in trace_rows:
            trace_rows_by_env[_int(row, "env_id", -1)].append(row)
        for per_window in audit.get("per_window", []):
            if not per_window.get("accepted", False):
                continue
            window_rows.append(
                _window_metrics(
                    audit_path=audit_path,
                    audit=audit,
                    summary=summary,
                    trace_rows_by_env=trace_rows_by_env,
                    per_window=per_window,
                )
            )
        sources.append({"audit_json": str(audit_path), "trace_csv": str(trace_csv), "summary_json": str(summary_json)})

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[_group_name(row)].append(row)

    baseline_non_yplus = [
        row
        for row in window_rows
        if row["quality_tier"] == "B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH" and row["direction"] != "y+"
    ]
    if baseline_non_yplus:
        grouped["baseline_tier_b_all_non_yplus"] = baseline_non_yplus

    summaries = {name: _summarize(name, rows) for name, rows in sorted(grouped.items())}
    target = summaries.get("target_yplus_tier_c", {})
    baseline = summaries.get("baseline_tier_b_all_non_yplus", {})
    pre24_disp_ratio = _ratio(target, baseline, "pre24_max_disp_xy_m_mean")
    pre24_tip_ratio = _ratio(target, baseline, "pre24_max_tip_angle_deg_mean")
    reaction_lead_delta = float(target.get("reaction_leads_measured_contact_steps_mean", 0.0)) - float(
        baseline.get("reaction_leads_measured_contact_steps_mean", 0.0)
    )
    anchor_before_push_delta = float(target.get("anchor_minus_first_push_phase_step_mean", 0.0)) - float(
        baseline.get("anchor_minus_first_push_phase_step_mean", 0.0)
    )
    supports_preanchor_reaction_accumulation = pre24_disp_ratio > 3.0 and pre24_tip_ratio > 3.0
    supports_unique_measured_contact_lead = reaction_lead_delta > 20.0
    supports_approach_phase_geometry = (
        supports_preanchor_reaction_accumulation
        and float(target.get("anchor_minus_first_push_phase_step_mean", 0.0)) < -20.0
        and float(target.get("target_offset_lateral_initial_m_mean", 0.0)) < -0.015
    )
    verdict = {
        "supports_yplus_preanchor_reaction_accumulation": supports_preanchor_reaction_accumulation,
        "supports_unique_measured_contact_lead": supports_unique_measured_contact_lead,
        "supports_yplus_approach_phase_geometry_hypothesis": supports_approach_phase_geometry,
        "pre24_disp_ratio": pre24_disp_ratio,
        "pre24_tip_ratio": pre24_tip_ratio,
        "reaction_lead_delta_steps": reaction_lead_delta,
        "anchor_before_push_delta_steps": anchor_before_push_delta,
        "do_not_start": ["GPU_runtime", "1024_10240_scaleup", "dataset_generation", "PPO_RL", "VLA", "TrackA"],
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in window_rows for key in row})
    with args.out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(window_rows)

    result = {
        "artifact_type": "cube10cm_yplus_early_contact_geometry_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_posthoc_only": True,
        "no_gpu_isaaclab_training_dataset_ssh": True,
        "source_files": sources,
        "window_count": len(window_rows),
        "group_summaries": summaries,
        "verdict": verdict,
        "out_csv": str(args.out_csv),
        "out_summary": str(args.out_summary),
    }
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    baseline_x = summaries.get("baseline_tier_b_x+", {})
    baseline_xneg = summaries.get("baseline_tier_b_x-", {})
    baseline_yneg = summaries.get("baseline_tier_b_y-", {})
    lines = [
        "line1 artifact=cube10cm_yplus_early_contact_geometry_audit_v1 local_posthoc_only=YES gpu_runtime=NO dataset_generation=NO",
        (
            "line2 target_yplus_tier_c "
            f"windows={target.get('window_count', 0)} "
            f"first_reaction_step_mean={target.get('first_reaction_step_mean', 0.0):.6f} "
            f"first_measured_contact_step_mean={target.get('first_measured_contact_step_mean', 0.0):.6f} "
            f"reaction_leads_contact_steps_mean={target.get('reaction_leads_measured_contact_steps_mean', 0.0):.6f} "
            f"anchor_minus_push_start_mean={target.get('anchor_minus_first_push_phase_step_mean', 0.0):.6f} "
            f"first_reaction_phase_alpha_mean={target.get('first_reaction_phase_alpha_mean', 0.0):.9f}"
        ),
        (
            "line3 baseline_tier_b_all_non_yplus "
            f"windows={baseline.get('window_count', 0)} "
            f"first_reaction_step_mean={baseline.get('first_reaction_step_mean', 0.0):.6f} "
            f"first_measured_contact_step_mean={baseline.get('first_measured_contact_step_mean', 0.0):.6f} "
            f"reaction_leads_contact_steps_mean={baseline.get('reaction_leads_measured_contact_steps_mean', 0.0):.6f} "
            f"anchor_minus_push_start_mean={baseline.get('anchor_minus_first_push_phase_step_mean', 0.0):.6f} "
            f"first_reaction_phase_alpha_mean={baseline.get('first_reaction_phase_alpha_mean', 0.0):.9f}"
        ),
        (
            "line4 pre_anchor_reaction "
            f"yplus_pre24_disp_mean_m={target.get('pre24_max_disp_xy_m_mean', 0.0):.9f} "
            f"baseline_pre24_disp_mean_m={baseline.get('pre24_max_disp_xy_m_mean', 0.0):.9f} "
            f"disp_ratio={pre24_disp_ratio:.9f} "
            f"yplus_pre24_tip_mean_deg={target.get('pre24_max_tip_angle_deg_mean', 0.0):.9f} "
            f"baseline_pre24_tip_mean_deg={baseline.get('pre24_max_tip_angle_deg_mean', 0.0):.9f} "
            f"tip_ratio={pre24_tip_ratio:.9f}"
        ),
        (
            "line5 initial_target_geometry "
            f"yplus_target_along_m={target.get('target_offset_along_initial_m_mean', 0.0):.9f} "
            f"yplus_target_lateral_m={target.get('target_offset_lateral_initial_m_mean', 0.0):.9f} "
            f"yplus_target_z_minus_cube_z_m={target.get('target_z_minus_cube_z_initial_m_mean', 0.0):.9f} "
            f"baseline_target_along_m={baseline.get('target_offset_along_initial_m_mean', 0.0):.9f} "
            f"baseline_target_lateral_m={baseline.get('target_offset_lateral_initial_m_mean', 0.0):.9f} "
            f"baseline_target_z_minus_cube_z_m={baseline.get('target_z_minus_cube_z_initial_m_mean', 0.0):.9f}"
        ),
        (
            "line6 anchor_geometry "
            f"yplus_anchor_tcp_cube_dist_m={target.get('anchor_tcp_cube_dist_m_mean', 0.0):.9f} "
            f"baseline_anchor_tcp_cube_dist_m={baseline.get('anchor_tcp_cube_dist_m_mean', 0.0):.9f} "
            f"yplus_tcp_z_minus_cube_z_anchor_m={target.get('tcp_z_minus_cube_z_anchor_m_mean', 0.0):.9f} "
            f"baseline_tcp_z_minus_cube_z_anchor_m={baseline.get('tcp_z_minus_cube_z_anchor_m_mean', 0.0):.9f}"
        ),
        (
            "line7 baselines "
            f"xplus_pre24_disp_mean_m={baseline_x.get('pre24_max_disp_xy_m_mean', 0.0):.9f} "
            f"xminus_pre24_disp_mean_m={baseline_xneg.get('pre24_max_disp_xy_m_mean', 0.0):.9f} "
            f"yminus_pre24_disp_mean_m={baseline_yneg.get('pre24_max_disp_xy_m_mean', 0.0):.9f} "
            f"xplus_anchor_minus_push_start_mean={baseline_x.get('anchor_minus_first_push_phase_step_mean', 0.0):.6f} "
            f"yminus_anchor_minus_push_start_mean={baseline_yneg.get('anchor_minus_first_push_phase_step_mean', 0.0):.6f}"
        ),
        (
            "line8 verdict "
            f"supports_yplus_preanchor_reaction_accumulation={supports_preanchor_reaction_accumulation} "
            f"supports_unique_measured_contact_lead={supports_unique_measured_contact_lead} "
            f"supports_yplus_approach_phase_geometry_hypothesis={supports_approach_phase_geometry}"
        ),
        (
            "line9 implication=yplus_accumulates_large_preanchor_reaction_inside_approach_phase; "
            "next_local_question=adjust_or_audit_yplus_precontact_lateral_height_timing_before_any_gpu_scaleup"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"wrote_json={args.out_json}")
    print(f"wrote_csv={args.out_csv}")
    print(f"wrote_summary={args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
