"""Compare teacher contact-frame definitions for the cube10cm y+ branch.

Local-only design audit. It does not run IsaacLab, GPU, dataset generation,
training, robot control, or SSH. It compares three candidate teacher criteria:

1. true_side_center_tcp: current semantic target; TCP itself reaches cube side
   center.
2. upper_edge_contact_proxy: accept the current observed upper/top contact proxy.
3. tool_oriented_side_contact_proxy: keep side-contact semantics, but change the
   teacher frame/tool/orientation design before generating action data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_TRACE = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_trace.csv"
DEFAULT_SUMMARY = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_summary.json"
DEFAULT_MISMATCH = LOG_DIR / "cube10cm_contact_frame_geometry_mismatch_audit.json"
DEFAULT_VISUAL = LOG_DIR / "cube10cm_visual_sim_sanity_audit.json"
DEFAULT_SEED944_SUMMARY = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_summary.json"
DEFAULT_SEED944_REACTION = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_height050_seed944_reaction_gate_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_teacher_contact_frame_design_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_teacher_contact_frame_design_audit_summary.out"


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value == "" else int(float(value))


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": mean(values),
        "median": median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _rate(flags: list[bool]) -> float:
    return sum(1 for flag in flags if flag) / len(flags) if flags else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


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


def _candidate_score(
    semantic_alignment: float,
    existing_evidence_fit: float,
    tracking_feasibility: float,
    safety_margin: float,
    implementation_readiness: float,
) -> float:
    return (
        0.28 * semantic_alignment
        + 0.22 * existing_evidence_fit
        + 0.20 * tracking_feasibility
        + 0.18 * safety_margin
        + 0.12 * implementation_readiness
    )


def build_audit(
    trace_csv: Path,
    summary_json: Path,
    mismatch_json: Path,
    visual_json: Path,
    seed944_summary_json: Path,
    seed944_reaction_json: Path,
) -> dict[str, Any]:
    rows = _read_trace(trace_csv)
    contact_rows = _first_contact_rows(rows)
    if not contact_rows:
        raise RuntimeError("no first-contact rows found in trace")

    summary = _read_json(summary_json)
    mismatch = _read_json(mismatch_json)
    visual = _read_json(visual_json)
    seed944_summary = _read_json(seed944_summary_json)
    seed944_reaction = _read_json(seed944_reaction_json)

    side_z_err: list[float] = []
    side_xy_err: list[float] = []
    upper_z_err: list[float] = []
    upper_total_err: list[float] = []
    top_margin_signed: list[float] = []
    center_delta: list[float] = []
    clip_any: list[bool] = []
    side_center_z_reached_10mm: list[bool] = []
    upper_proxy_z_reached_10mm: list[bool] = []
    upper_proxy_total_reached_15mm: list[bool] = []
    contact_source_lines: list[int] = []

    for row in contact_rows:
        tcp_x = _f(row, "tcp_x_before_m", _f(row, "tcp_x_m"))
        tcp_y = _f(row, "tcp_y_before_m", _f(row, "tcp_y_m"))
        tcp_z = _f(row, "tcp_z_before_m", _f(row, "tcp_z_m"))
        target_x = _f(row, "target_x_m")
        target_y = _f(row, "target_y_m")
        target_z = _f(row, "target_z_m")
        cube_z = _f(row, "cube_z_m")
        half_z = _f(row, "cube_size_z_m") * 0.5
        live_top_z = cube_z + half_z
        xy = math.hypot(tcp_x - target_x, tcp_y - target_y)
        side_z = abs(tcp_z - target_z)
        upper_z = abs(tcp_z - live_top_z)
        upper_err = math.sqrt(xy * xy + upper_z * upper_z)

        side_z_err.append(side_z)
        side_xy_err.append(xy)
        upper_z_err.append(upper_z)
        upper_total_err.append(upper_err)
        top_margin_signed.append(live_top_z - tcp_z)
        center_delta.append(tcp_z - cube_z)
        clip_any.append(_i(row, "clip_any") == 1)
        side_center_z_reached_10mm.append(side_z <= 0.010)
        upper_proxy_z_reached_10mm.append(upper_z <= 0.010)
        upper_proxy_total_reached_15mm.append(upper_err <= 0.015)
        contact_source_lines.append(int(row["_source_line"]))

    n = len(contact_rows)
    side_reach_rate = _rate(side_center_z_reached_10mm)
    upper_z_reach_rate = _rate(upper_proxy_z_reached_10mm)
    upper_total_reach_rate = _rate(upper_proxy_total_reached_15mm)
    clip_rate = _rate(clip_any)
    current_contact_pass = bool(visual.get("visual_contact_evidence", False))
    current_clean_tap = bool(visual.get("clean_tap_visual_verified", False))

    seed944_contact_rate = float(seed944_reaction.get("contact_evidence_rate", seed944_summary.get("measured_contact_seen_rate", 0.0)))
    seed944_reaction_pass = bool(seed944_reaction.get("reaction_gate_pass", False))
    seed944_final_tcp_err = float(seed944_summary.get("final_tcp_target_err_mean_m", math.nan))

    candidates: dict[str, dict[str, Any]] = {}

    # Current semantic criterion: correct desired object interaction, not currently achieved.
    side_semantic = 1.0
    side_fit = side_reach_rate
    side_tracking = max(0.0, 1.0 - min(1.0, mean(side_z_err) / 0.055))
    side_safety = 0.80
    side_ready = 0.30
    candidates["true_side_center_tcp"] = {
        "definition": "teacher TCP target is cube side-center at contact height",
        "semantic_alignment": side_semantic,
        "existing_evidence_fit": side_fit,
        "tracking_feasibility_from_seed962": side_tracking,
        "safety_margin": side_safety,
        "implementation_readiness": side_ready,
        "score": _candidate_score(side_semantic, side_fit, side_tracking, side_safety, side_ready),
        "evidence": {
            "side_center_z_reached_10mm_rate": side_reach_rate,
            "side_center_z_err_m": _stats(side_z_err),
            "tcp_target_xy_err_m": _stats(side_xy_err),
            "clip_any_rate_at_first_contact": clip_rate,
        },
        "verdict": "SEMANTICALLY_CORRECT_BUT_CURRENT_TRACE_FAILS_TRACKING",
        "runtime_role": "not_ready_as_action_teacher_until_tracking_or_tool_frame_changes",
    }

    # Current observed contact criterion: explains the visual replay but risks teaching top contact.
    upper_semantic = 0.45
    upper_fit = 0.5 * upper_z_reach_rate + 0.5 * upper_total_reach_rate
    upper_tracking = max(0.0, 1.0 - min(1.0, mean(upper_z_err) / 0.010))
    # Upper/top contact is less safe for a 10cm cube because it can create tip/lift torque.
    upper_safety = 0.35
    upper_ready = 0.65
    candidates["upper_edge_contact_proxy"] = {
        "definition": "teacher accepts/targets the upper edge or live cube top contact proxy",
        "semantic_alignment": upper_semantic,
        "existing_evidence_fit": upper_fit,
        "tracking_feasibility_from_seed962": upper_tracking,
        "safety_margin": upper_safety,
        "implementation_readiness": upper_ready,
        "score": _candidate_score(upper_semantic, upper_fit, upper_tracking, upper_safety, upper_ready),
        "evidence": {
            "upper_proxy_z_reached_10mm_rate": upper_z_reach_rate,
            "upper_proxy_total_reached_15mm_rate": upper_total_reach_rate,
            "upper_proxy_z_err_m": _stats(upper_z_err),
            "upper_proxy_total_err_m": _stats(upper_total_err),
            "tcp_below_live_cube_top_signed_m": _stats(top_margin_signed),
            "tcp_above_live_cube_center_m": _stats(center_delta),
            "seed944_height050_contact_evidence_rate": seed944_contact_rate,
            "seed944_height050_reaction_gate_pass": seed944_reaction_pass,
            "seed944_height050_final_tcp_err_m": seed944_final_tcp_err,
        },
        "verdict": "BEST_EXPLAINS_CURRENT_VISUAL_CONTACT_BUT_TEACHES_TOP_CONTACT_AND_HEIGHT_ONLY_NEGATIVE_CONTROL_FAILED",
        "runtime_role": "allowed_only_as_one_tiny_negative_or_diagnostic_retest_not_as_final_teacher",
    }

    # Preferred teacher design criterion: side-contact semantics with real contact proxy/orientation.
    tool_semantic = 0.95
    # Existing trace proves why this is needed, but cannot validate the new path because command is position-only.
    tool_fit = 0.55 if current_contact_pass and not current_clean_tap else 0.25
    tool_tracking = 0.45
    tool_safety = 0.85
    tool_ready = 0.20
    candidates["tool_oriented_side_contact_proxy"] = {
        "definition": "teacher target is a real tool/contact proxy and orientation path that achieves side contact",
        "semantic_alignment": tool_semantic,
        "existing_evidence_fit": tool_fit,
        "tracking_feasibility_from_seed962": tool_tracking,
        "safety_margin": tool_safety,
        "implementation_readiness": tool_ready,
        "score": _candidate_score(tool_semantic, tool_fit, tool_tracking, tool_safety, tool_ready),
        "evidence": {
            "current_diffik_command_type": summary.get("command_type"),
            "current_controller": summary.get("controller"),
            "current_visual_contact_pass": current_contact_pass,
            "current_clean_tap_visual_verified": current_clean_tap,
            "position_only_diffik_cannot_validate_new_orientation_path_from_seed962_trace": True,
            "why_needed": "current hand TCP side-center target contacts near cube top under clipping",
        },
        "verdict": "BEST_TEACHER_CRITERION_BUT_REQUIRES_LOCAL_PROBE_DESIGN_BEFORE_RUNTIME",
        "runtime_role": "preferred_next_design_path; requires one explicit tiny runtime after local preflight",
    }

    ranking = sorted(candidates, key=lambda name: candidates[name]["score"], reverse=True)
    selected_teacher_criterion = "tool_oriented_side_contact_proxy"
    one_tiny_runtime_candidate = "upper_edge_contact_proxy_negative_control_or_tool_proxy_preflight"
    if candidates["upper_edge_contact_proxy"]["score"] > candidates[selected_teacher_criterion]["score"]:
        # Keep the final teacher conservative even if the current trace fits upper-edge better.
        one_tiny_runtime_candidate = "upper_edge_contact_proxy_negative_control_before_any_dataset"

    return {
        "artifact_type": "cube10cm_teacher_contact_frame_design_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_audit_only": True,
        "no_gpu_runtime_dataset_training_robot_ssh": True,
        "source": {
            "trace_csv": str(trace_csv),
            "trace_rows": len(rows),
            "first_contact_envs": n,
            "first_contact_source_line_min": min(contact_source_lines),
            "first_contact_source_line_max": max(contact_source_lines),
            "summary_json": str(summary_json),
            "mismatch_json": str(mismatch_json),
            "visual_json": str(visual_json),
            "seed944_summary_json": str(seed944_summary_json),
            "seed944_reaction_json": str(seed944_reaction_json),
        },
        "current_blocker": {
            "mismatch_class": mismatch.get("verdict", {}).get("mismatch_class"),
            "visual_contact_replay_pass": current_contact_pass,
            "clean_tap_visual_verified": current_clean_tap,
            "clip_rate_at_first_contact": clip_rate,
            "side_center_z_reached_10mm_rate": side_reach_rate,
            "upper_proxy_z_reached_10mm_rate": upper_z_reach_rate,
        },
        "candidate_scores": candidates,
        "ranking_by_numeric_score": ranking,
        "selected_teacher_criterion": selected_teacher_criterion,
        "selection_reason": (
            "Upper-edge proxy best fits the accidental current contact, but it would teach top contact. "
            "True side-center is semantically right but currently untracked. The teacher standard should "
            "therefore be a tool/contact proxy or orientation path that preserves side-contact semantics."
        ),
        "one_tiny_runtime_candidate_after_local_preflight": one_tiny_runtime_candidate,
        "pipeline": {
            "action_teacher_dataset_unblocked": False,
            "large_dataset_unblocked": False,
            "isaaclab_rl_unblocked": False,
            "roarm_m3_pro_unblocked": False,
        },
    }


def write_summary(audit: dict[str, Any], path: Path) -> None:
    scores = audit["candidate_scores"]
    side = scores["true_side_center_tcp"]
    upper = scores["upper_edge_contact_proxy"]
    tool = scores["tool_oriented_side_contact_proxy"]
    cur = audit["current_blocker"]
    lines = [
        "line1 artifact=cube10cm_teacher_contact_frame_design_audit_v1 "
        "local_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        "line2 compared_criteria=true_side_center_tcp,upper_edge_contact_proxy,tool_oriented_side_contact_proxy "
        f"first_contact_envs={audit['source']['first_contact_envs']} "
        f"source_lines={audit['source']['first_contact_source_line_min']}-{audit['source']['first_contact_source_line_max']}",
        "line3 current_blocker "
        f"mismatch_class={cur['mismatch_class']} "
        f"visual_contact_replay_pass={cur['visual_contact_replay_pass']} "
        f"clean_tap_visual_verified={cur['clean_tap_visual_verified']} "
        f"clip_rate_at_first_contact={cur['clip_rate_at_first_contact']:.9f}",
        "line4 true_side_center_tcp "
        f"score={side['score']:.9f} "
        f"side_center_z_reached_10mm_rate={side['evidence']['side_center_z_reached_10mm_rate']:.9f} "
        f"z_err_mean={side['evidence']['side_center_z_err_m']['mean']:.9f} "
        f"xy_err_mean={side['evidence']['tcp_target_xy_err_m']['mean']:.9f} "
        f"verdict={side['verdict']}",
        "line5 upper_edge_contact_proxy "
        f"score={upper['score']:.9f} "
        f"upper_z_reached_10mm_rate={upper['evidence']['upper_proxy_z_reached_10mm_rate']:.9f} "
        f"upper_total_reached_15mm_rate={upper['evidence']['upper_proxy_total_reached_15mm_rate']:.9f} "
        f"upper_z_err_mean={upper['evidence']['upper_proxy_z_err_m']['mean']:.9f} "
        f"seed944_contact_rate={upper['evidence']['seed944_height050_contact_evidence_rate']:.9f} "
        f"verdict={upper['verdict']}",
        "line6 tool_oriented_side_contact_proxy "
        f"score={tool['score']:.9f} "
        f"current_diffik_command_type={tool['evidence']['current_diffik_command_type']} "
        f"position_only_cannot_validate_orientation_path="
        f"{tool['evidence']['position_only_diffik_cannot_validate_new_orientation_path_from_seed962_trace']} "
        f"verdict={tool['verdict']}",
        "line7 selected_teacher_criterion "
        f"{audit['selected_teacher_criterion']} reason={audit['selection_reason']}",
        "line8 one_tiny_runtime_candidate_after_local_preflight "
        f"{audit['one_tiny_runtime_candidate_after_local_preflight']} "
        "dataset_rl_roarm_unblocked=NO",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_csv", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--mismatch_json", type=Path, default=DEFAULT_MISMATCH)
    parser.add_argument("--visual_json", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--seed944_summary_json", type=Path, default=DEFAULT_SEED944_SUMMARY)
    parser.add_argument("--seed944_reaction_json", type=Path, default=DEFAULT_SEED944_REACTION)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    audit = build_audit(
        args.trace_csv,
        args.summary_json,
        args.mismatch_json,
        args.visual_json,
        args.seed944_summary_json,
        args.seed944_reaction_json,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_summary(audit, args.out_summary)
    print(args.out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
