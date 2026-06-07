"""Local audit for the y+ pre020 seed962 failure-mode shift.

This posthoc audit compares the fixed y+ seed958/960/961/962 reaction-window
artifacts. It asks whether precontact 0.020 reduced early/pre-anchor reaction,
whether it weakened contact/reaction strength, and whether it improves data
readiness. It performs no IsaacLab/GPU runtime, training, dataset generation,
robot control, SSH, or trace mutation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cube10cm_yplus_early_contact_geometry_audit import (
    _float,
    _int,
    _load_json,
    _load_rows,
    _mean,
    _window_metrics,
)


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_AUDITS = (
    LOG_DIR / "cube10cm_reaction_window_seed958_audit.json",
    LOG_DIR / "cube10cm_reaction_window_seed960_audit.json",
    LOG_DIR / "cube10cm_reaction_window_seed961_audit.json",
    LOG_DIR / "cube10cm_reaction_window_seed962_audit.json",
)


def _audit_label(path: Path, summary: dict[str, Any]) -> str:
    name = path.name
    if "seed962" in name:
        return "seed962_pre020"
    if "seed961" in name:
        return "seed961_stiff600"
    if "seed960" in name:
        return "seed960_cap050"
    if "seed958" in name:
        return "seed958_pre010_baseline"
    return f"{path.stem}_pre{int(round(float(summary.get('precontact_clearance_m', 0.0)) * 1000.0)):03d}"


def _summarize(label: str, audit_path: Path, audit: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    tier_counts = Counter(str(row["quality_tier"]) for row in rows)
    vals = lambda key: [float(row[key]) for row in rows if key in row]  # noqa: E731
    return {
        "label": label,
        "audit_json": str(audit_path),
        "summary_json": str(audit.get("summary_json", "")),
        "trace_csv": str(audit.get("trace_csv", "")),
        "precontact_clearance_m": float(summary.get("precontact_clearance_m", 0.0)),
        "max_diffik_joint_step_rad": float(summary.get("max_diffik_joint_step_rad", 0.0)),
        "arm_actuator_stiffness": float(summary.get("arm_actuator_stiffness", 0.0)),
        "controlled_push_rate": float(summary.get("controlled_push_rate", 0.0)),
        "disp_ge_gate_rate": float(summary.get("disp_ge_gate_rate", 0.0)),
        "max_disp_along_push_mean_m": float(summary.get("max_disp_along_push_mean_m", 0.0)),
        "max_cube_z_delta_mean_m": float(summary.get("max_cube_z_delta_mean_m", 0.0)),
        "max_tip_angle_mean_deg": float(summary.get("max_tip_angle_mean_deg", 0.0)),
        "low_motion_rate": float(summary.get("low_motion_rate", 0.0)),
        "diffik_clip_rate_mean": float(summary.get("diffik_clip_rate_mean", 0.0)),
        "final_tcp_target_err_mean_m": float(summary.get("final_tcp_target_err_mean_m", 0.0)),
        "accepted_window_count": int(audit.get("accepted_window_count", 0)),
        "candidate_window_count": int(audit.get("candidate_window_count", 0)),
        "accepted_follow_p95_to_cap_p95": float(audit.get("accepted_window_follow_p95_to_cap_p95", 0.0)),
        "accepted_max_disp_m_mean": float(audit.get("accepted_max_disp_m_mean", 0.0)),
        "quality_tier_counts": dict(sorted(tier_counts.items())),
        "tier_b_count": int(tier_counts.get("B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH", 0)),
        "tier_c_count": int(tier_counts.get("C_REACTION_VALID_FOLLOW_LAG", 0)),
        "anchor_minus_push_start_mean": _mean(vals("anchor_minus_first_push_phase_step")),
        "first_reaction_step_mean": _mean(vals("first_reaction_step")),
        "first_measured_contact_step_mean": _mean(vals("first_measured_contact_step")),
        "first_reaction_phase_alpha_mean": _mean(vals("first_reaction_phase_alpha")),
        "pre24_max_disp_xy_m_mean": _mean(vals("pre24_max_disp_xy_m")),
        "pre24_max_tip_angle_deg_mean": _mean(vals("pre24_max_tip_angle_deg")),
        "pre_all_max_disp_xy_m_mean": _mean(vals("pre_all_max_disp_xy_m")),
        "pre_all_max_tip_angle_deg_mean": _mean(vals("pre_all_max_tip_angle_deg")),
    }


def _load_window_rows(audit_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    audit = _load_json(audit_path)
    summary_json = REPO / str(audit["summary_json"])
    trace_csv = REPO / str(audit["trace_csv"])
    summary = _load_json(summary_json)
    trace_rows = _load_rows(trace_csv)
    trace_rows_by_env: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        trace_rows_by_env[_int(row, "env_id", -1)].append(row)

    rows: list[dict[str, Any]] = []
    for per_window in audit.get("per_window", []):
        if not per_window.get("accepted", False):
            continue
        metric = _window_metrics(
            audit_path=audit_path,
            audit=audit,
            summary=summary,
            trace_rows_by_env=trace_rows_by_env,
            per_window=per_window,
        )
        if metric.get("direction") == "y+":
            rows.append(metric)
    return audit, summary, rows


def _ratio(num: float, denom: float) -> float:
    return 0.0 if denom == 0.0 else num / denom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_json", action="append", type=Path, default=None)
    parser.add_argument(
        "--out_json",
        type=Path,
        default=LOG_DIR / "cube10cm_yplus_pre020_failure_shift_audit.json",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=LOG_DIR / "cube10cm_yplus_pre020_failure_shift_audit.csv",
    )
    parser.add_argument(
        "--out_summary",
        type=Path,
        default=LOG_DIR / "cube10cm_yplus_pre020_failure_shift_audit_summary.out",
    )
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for audit_path in args.audit_json or list(DEFAULT_AUDITS):
        audit, summary, rows = _load_window_rows(audit_path)
        label = _audit_label(audit_path, summary)
        summaries.append(_summarize(label, audit_path, audit, summary, rows))

    by_label = {row["label"]: row for row in summaries}
    previous = [
        row
        for row in summaries
        if row["label"] in {"seed958_pre010_baseline", "seed960_cap050", "seed961_stiff600"}
    ]
    seed962 = by_label.get("seed962_pre020", {})
    prev_pre24_disp = [float(row["pre24_max_disp_xy_m_mean"]) for row in previous]
    prev_pre24_tip = [float(row["pre24_max_tip_angle_deg_mean"]) for row in previous]
    prev_max_disp = [float(row["max_disp_along_push_mean_m"]) for row in previous]
    prev_tip = [float(row["max_tip_angle_mean_deg"]) for row in previous]

    prev_pre24_disp_mean = _mean(prev_pre24_disp)
    prev_pre24_tip_mean = _mean(prev_pre24_tip)
    prev_max_disp_mean = _mean(prev_max_disp)
    prev_tip_mean = _mean(prev_tip)
    seed962_pre24_disp = float(seed962.get("pre24_max_disp_xy_m_mean", 0.0))
    seed962_pre24_tip = float(seed962.get("pre24_max_tip_angle_deg_mean", 0.0))
    seed962_max_disp = float(seed962.get("max_disp_along_push_mean_m", 0.0))
    seed962_tip = float(seed962.get("max_tip_angle_mean_deg", 0.0))
    seed962_tier_c = int(seed962.get("tier_c_count", 0))
    seed962_windows = int(seed962.get("accepted_window_count", 0))
    seed962_follow = float(seed962.get("accepted_follow_p95_to_cap_p95", 0.0))

    pre020_reduces_preanchor = (
        bool(seed962)
        and seed962_pre24_disp < min(prev_pre24_disp)
        and seed962_pre24_tip < min(prev_pre24_tip)
        and float(seed962.get("anchor_minus_push_start_mean", -999.0)) > 0.0
    )
    pre020_weakens_reaction = (
        bool(seed962)
        and seed962_max_disp < min(prev_max_disp)
        and seed962_tip < min(prev_tip)
        and float(seed962.get("controlled_push_rate", 1.0)) < 0.75
    )
    quality_still_blocked = (
        bool(seed962)
        and float(seed962.get("diffik_clip_rate_mean", 0.0)) >= 1.0
        and seed962_follow > 1.0
        and seed962_tier_c > seed962_windows / 2.0
    )

    verdict = {
        "pre020_reduces_preanchor_reaction": pre020_reduces_preanchor,
        "pre020_weakens_reaction_strength": pre020_weakens_reaction,
        "quality_still_blocked": quality_still_blocked,
        "pre24_disp_ratio_seed962_vs_prev_mean": _ratio(seed962_pre24_disp, prev_pre24_disp_mean),
        "pre24_tip_ratio_seed962_vs_prev_mean": _ratio(seed962_pre24_tip, prev_pre24_tip_mean),
        "max_disp_ratio_seed962_vs_prev_mean": _ratio(seed962_max_disp, prev_max_disp_mean),
        "max_tip_ratio_seed962_vs_prev_mean": _ratio(seed962_tip, prev_tip_mean),
        "next_research_step": "local_timing_contact_strength_audit_before_any_gpu",
        "do_not_start": [
            "GPU_runtime_without_explicit_approval",
            "blind_precontact_sweep",
            "lateral_height_actuator_dls_cap_mixing",
            "1024_10240_scaleup",
            "dataset_generation",
            "PPO_RL",
            "VLA",
            "TrackA",
            "B200_SSH",
        ],
    }

    result = {
        "artifact_type": "cube10cm_yplus_pre020_failure_shift_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_posthoc_only": True,
        "no_gpu_isaaclab_training_dataset_ssh": True,
        "seed_summaries": summaries,
        "previous_seed_group": {
            "labels": [row["label"] for row in previous],
            "pre24_disp_mean_m": prev_pre24_disp_mean,
            "pre24_tip_mean_deg": prev_pre24_tip_mean,
            "max_disp_along_push_mean_m": prev_max_disp_mean,
            "max_tip_angle_mean_deg": prev_tip_mean,
        },
        "verdict": verdict,
        "out_csv": str(args.out_csv),
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in summaries for key in row if key != "quality_tier_counts"})
        with args.out_csv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)

    lines = [
        "line1 artifact=cube10cm_yplus_pre020_failure_shift_audit_v1 local_posthoc_only=YES gpu_runtime=NO dataset_generation=NO",
    ]
    for row in summaries:
        lines.append(
            "line2 seed_summary "
            f"label={row['label']} precontact_m={row['precontact_clearance_m']:.6f} "
            f"tiers={row['quality_tier_counts']} anchor_minus_push_mean={row['anchor_minus_push_start_mean']:.6f} "
            f"pre24_disp_mean_m={row['pre24_max_disp_xy_m_mean']:.9f} "
            f"pre24_tip_mean_deg={row['pre24_max_tip_angle_deg_mean']:.9f} "
            f"controlled_push_rate={row['controlled_push_rate']:.9f} "
            f"max_disp_along_push_mean_m={row['max_disp_along_push_mean_m']:.9f} "
            f"max_tip_angle_mean_deg={row['max_tip_angle_mean_deg']:.9f} "
            f"follow_p95_to_cap_p95={row['accepted_follow_p95_to_cap_p95']:.9f}"
        )
    lines.extend(
        [
            "line6 comparison "
            f"seed962_pre24_disp_vs_prev_mean={verdict['pre24_disp_ratio_seed962_vs_prev_mean']:.9f} "
            f"seed962_pre24_tip_vs_prev_mean={verdict['pre24_tip_ratio_seed962_vs_prev_mean']:.9f} "
            f"seed962_max_disp_vs_prev_mean={verdict['max_disp_ratio_seed962_vs_prev_mean']:.9f} "
            f"seed962_max_tip_vs_prev_mean={verdict['max_tip_ratio_seed962_vs_prev_mean']:.9f}",
            "line7 verdict "
            f"pre020_reduces_preanchor_reaction={pre020_reduces_preanchor} "
            f"pre020_weakens_reaction_strength={pre020_weakens_reaction} "
            f"quality_still_blocked={quality_still_blocked}",
            "line8 next=local_timing_contact_strength_audit_before_any_gpu do_not_start=GPU_without_explicit_approval,blind_precontact_sweep,lateral_height_actuator_dls_cap_mixing,1024_10240,dataset,PPO_RL,VLA,TrackA,B200_SSH",
        ]
    )
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"wrote_json={args.out_json}")
    print(f"wrote_csv={args.out_csv}")
    print(f"wrote_summary={args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
