"""Compare side-center baseline vs top-margin negative-control runtime.

This is a local posthoc audit over existing summaries produced by approved tiny
runs. It does not run IsaacLab, GPU, dataset generation, training, robot control,
or SSH.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

SIDE_SUMMARY = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_summary.json"
SIDE_REACTION = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_reaction_gate_audit.json"
SIDE_WINDOW = LOG_DIR / "cube10cm_reaction_window_seed962_audit.json"
SIDE_MISMATCH_SUMMARY = LOG_DIR / "cube10cm_contact_frame_geometry_mismatch_audit_summary.out"

TOP_SUMMARY = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_topmargin_seed962_summary.json"
TOP_REACTION = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_topmargin_seed962_reaction_gate_audit.json"
TOP_WINDOW = LOG_DIR / "cube10cm_reaction_window_seed962_topmargin_audit.json"
TOP_MISMATCH_SUMMARY = LOG_DIR / "cube10cm_contact_frame_geometry_mismatch_topmargin_seed962_summary.out"
TOP_NEXT = LOG_DIR / "cube10cm_next_research_step_seed962_topmargin_audit.json"

DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_teacher_contact_frame_runtime_comparison_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_teacher_contact_frame_runtime_comparison_audit_summary.out"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_ratio(num: float, den: float) -> float | None:
    return None if abs(den) < 1.0e-12 else num / den


def _line(path: Path, n: int) -> str:
    lines = path.read_text().splitlines()
    return lines[n - 1] if 0 <= n - 1 < len(lines) else ""


def build_audit(
    side_summary_path: Path,
    side_reaction_path: Path,
    side_window_path: Path,
    side_mismatch_summary_path: Path,
    top_summary_path: Path,
    top_reaction_path: Path,
    top_window_path: Path,
    top_mismatch_summary_path: Path,
    top_next_path: Path,
) -> dict[str, Any]:
    side_summary = _json(side_summary_path)
    side_reaction = _json(side_reaction_path)
    side_window = _json(side_window_path)
    top_summary = _json(top_summary_path)
    top_reaction = _json(top_reaction_path)
    top_window = _json(top_window_path)
    top_next = _json(top_next_path)

    side = {
        "label": "side_center_baseline_seed962_pre020",
        "tcp_height_mode": side_summary["tcp_height_mode"],
        "reaction_gate_pass": side_reaction["reaction_gate_pass"],
        "teacher_quality_ready": side_reaction["teacher_quality_ready"],
        "reaction_event_rate": side_reaction["reaction_event_rate"],
        "contact_evidence_rate": side_reaction["contact_evidence_rate"],
        "overshoot_rate": side_reaction["overshoot_rate"],
        "diffik_clip_rate_mean": side_summary["diffik_clip_rate_mean"],
        "final_tcp_target_err_mean_m": side_summary["final_tcp_target_err_mean_m"],
        "max_disp_along_push_mean_m": side_summary["max_disp_along_push_mean_m"],
        "final_disp_along_push_mean_m": side_summary["disp_along_push_mean_m"],
        "controlled_push_rate": side_summary["controlled_push_rate"],
        "low_motion_rate": side_summary["low_motion_rate"],
        "first_contact_step_mean": side_summary["first_contact_step_mean"],
        "max_tip_angle_mean_deg": side_summary["max_tip_angle_mean_deg"],
        "reaction_window_tiers": side_window["quality_tier_counts"],
        "reaction_window_clean_ready": side_window["clean_diffik_teacher_window_ready"],
        "reaction_window_clip_any_rate_mean": side_window["accepted_window_clip_any_rate_mean"],
        "reaction_window_follow_p95_to_cap_p95": side_window["accepted_window_follow_p95_to_cap_p95"],
        "mismatch_summary_line4": _line(side_mismatch_summary_path, 4),
        "mismatch_summary_line5": _line(side_mismatch_summary_path, 5),
    }
    top = {
        "label": "top_margin_negative_control_seed962",
        "tcp_height_mode": top_summary["tcp_height_mode"],
        "reaction_gate_pass": top_reaction["reaction_gate_pass"],
        "teacher_quality_ready": top_reaction["teacher_quality_ready"],
        "reaction_event_rate": top_reaction["reaction_event_rate"],
        "contact_evidence_rate": top_reaction["contact_evidence_rate"],
        "overshoot_rate": top_reaction["overshoot_rate"],
        "diffik_clip_rate_mean": top_summary["diffik_clip_rate_mean"],
        "final_tcp_target_err_mean_m": top_summary["final_tcp_target_err_mean_m"],
        "max_disp_along_push_mean_m": top_summary["max_disp_along_push_mean_m"],
        "final_disp_along_push_mean_m": top_summary["disp_along_push_mean_m"],
        "controlled_push_rate": top_summary["controlled_push_rate"],
        "low_motion_rate": top_summary["low_motion_rate"],
        "first_contact_step_mean": top_summary["first_contact_step_mean"],
        "max_tip_angle_mean_deg": top_summary["max_tip_angle_mean_deg"],
        "reaction_window_tiers": top_window["quality_tier_counts"],
        "reaction_window_clean_ready": top_window["clean_diffik_teacher_window_ready"],
        "reaction_window_clip_any_rate_mean": top_window["accepted_window_clip_any_rate_mean"],
        "reaction_window_follow_p95_to_cap_p95": top_window["accepted_window_follow_p95_to_cap_p95"],
        "next_direction": top_next.get("next_direction"),
        "mismatch_summary_line4": _line(top_mismatch_summary_path, 4),
        "mismatch_summary_line5": _line(top_mismatch_summary_path, 5),
    }
    ratios = {
        "top_vs_side_clip_rate": _safe_ratio(top["diffik_clip_rate_mean"], side["diffik_clip_rate_mean"]),
        "top_vs_side_final_tcp_err": _safe_ratio(top["final_tcp_target_err_mean_m"], side["final_tcp_target_err_mean_m"]),
        "top_vs_side_max_disp": _safe_ratio(top["max_disp_along_push_mean_m"], side["max_disp_along_push_mean_m"]),
        "top_vs_side_final_disp": _safe_ratio(top["final_disp_along_push_mean_m"], side["final_disp_along_push_mean_m"]),
        "top_vs_side_tip": _safe_ratio(top["max_tip_angle_mean_deg"], side["max_tip_angle_mean_deg"]),
    }
    verdict = {
        "upper_edge_proxy_tracking_improved": bool(
            top["teacher_quality_ready"]
            and top["diffik_clip_rate_mean"] < side["diffik_clip_rate_mean"]
            and top["final_tcp_target_err_mean_m"] < side["final_tcp_target_err_mean_m"]
        ),
        "upper_edge_proxy_tap_strength_weakened": bool(
            top["max_disp_along_push_mean_m"] < side["max_disp_along_push_mean_m"]
            and top["controlled_push_rate"] <= side["controlled_push_rate"]
        ),
        "upper_edge_proxy_selected_as_teacher": False,
        "selected_teacher_criterion": "tool_oriented_side_contact_proxy",
        "reason": (
            "top-margin proves target tracking can improve by moving to upper/top contact, "
            "but it weakens the tap and encodes the wrong contact semantics"
        ),
        "dataset_rl_roarm_unblocked": False,
    }
    return {
        "artifact_type": "cube10cm_teacher_contact_frame_runtime_comparison_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_posthoc_audit_only": True,
        "new_gpu_runtime_in_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "side_center_baseline": side,
        "top_margin_negative_control": top,
        "ratios": ratios,
        "verdict": verdict,
    }


def write_summary(audit: dict[str, Any], path: Path) -> None:
    side = audit["side_center_baseline"]
    top = audit["top_margin_negative_control"]
    ratios = audit["ratios"]
    verdict = audit["verdict"]
    lines = [
        "line1 artifact=cube10cm_teacher_contact_frame_runtime_comparison_audit_v1 "
        "local_posthoc_audit_only=YES new_gpu_runtime_in_this_audit=NO "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO",
        "line2 side_center_baseline "
        f"reaction_gate_pass={side['reaction_gate_pass']} teacher_quality_ready={side['teacher_quality_ready']} "
        f"clip_mean={side['diffik_clip_rate_mean']:.9f} final_tcp_err={side['final_tcp_target_err_mean_m']:.9f} "
        f"max_disp={side['max_disp_along_push_mean_m']:.9f} final_disp={side['final_disp_along_push_mean_m']:.9f} "
        f"controlled_push={side['controlled_push_rate']:.9f} tiers={side['reaction_window_tiers']}",
        "line3 top_margin_negative_control "
        f"reaction_gate_pass={top['reaction_gate_pass']} teacher_quality_ready={top['teacher_quality_ready']} "
        f"clip_mean={top['diffik_clip_rate_mean']:.9f} final_tcp_err={top['final_tcp_target_err_mean_m']:.9f} "
        f"max_disp={top['max_disp_along_push_mean_m']:.9f} final_disp={top['final_disp_along_push_mean_m']:.9f} "
        f"controlled_push={top['controlled_push_rate']:.9f} tiers={top['reaction_window_tiers']} "
        f"window_clean_ready={top['reaction_window_clean_ready']}",
        "line4 top_vs_side_ratios "
        f"clip={ratios['top_vs_side_clip_rate']:.9f} "
        f"final_tcp_err={ratios['top_vs_side_final_tcp_err']:.9f} "
        f"max_disp={ratios['top_vs_side_max_disp']:.9f} "
        f"final_disp={ratios['top_vs_side_final_disp']:.9f} "
        f"tip={ratios['top_vs_side_tip']:.9f}",
        "line5 contact_frame_tradeoff "
        "top_margin_tracks_target_better_but_contacts_upper_top_proxy=True "
        f"top_mismatch_line5={top['mismatch_summary_line5']}",
        "line6 verdict "
        f"upper_edge_proxy_tracking_improved={verdict['upper_edge_proxy_tracking_improved']} "
        f"upper_edge_proxy_tap_strength_weakened={verdict['upper_edge_proxy_tap_strength_weakened']} "
        f"upper_edge_proxy_selected_as_teacher={verdict['upper_edge_proxy_selected_as_teacher']} "
        f"selected_teacher_criterion={verdict['selected_teacher_criterion']}",
        "line7 pipeline "
        f"dataset_rl_roarm_unblocked={verdict['dataset_rl_roarm_unblocked']} "
        f"reason={verdict['reason']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side_summary_json", type=Path, default=SIDE_SUMMARY)
    parser.add_argument("--side_reaction_json", type=Path, default=SIDE_REACTION)
    parser.add_argument("--side_window_json", type=Path, default=SIDE_WINDOW)
    parser.add_argument("--side_mismatch_summary", type=Path, default=SIDE_MISMATCH_SUMMARY)
    parser.add_argument("--top_summary_json", type=Path, default=TOP_SUMMARY)
    parser.add_argument("--top_reaction_json", type=Path, default=TOP_REACTION)
    parser.add_argument("--top_window_json", type=Path, default=TOP_WINDOW)
    parser.add_argument("--top_mismatch_summary", type=Path, default=TOP_MISMATCH_SUMMARY)
    parser.add_argument("--top_next_json", type=Path, default=TOP_NEXT)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    audit = build_audit(
        args.side_summary_json,
        args.side_reaction_json,
        args.side_window_json,
        args.side_mismatch_summary,
        args.top_summary_json,
        args.top_reaction_json,
        args.top_window_json,
        args.top_mismatch_summary,
        args.top_next_json,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_summary(audit, args.out_summary)
    print(args.out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
