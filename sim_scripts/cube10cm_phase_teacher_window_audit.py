"""Phase-window audit for professor cube10cm tap/reaction DiffIK traces.

This is a local posthoc tool. It does not run IsaacLab, train, generate data, or
change the official next-step gate. Its job is to separate the pre-stop motion
window from the post-stop freeze window so we can see whether teacher quality is
blocked only by freeze bookkeeping or also by pre-stop actuator/IK tracking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return bool(value)


def _nested_float(root: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    node: Any = root
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    return _float(node, default)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reaction_audit_json", type=Path, required=True)
    parser.add_argument("--trace_diag_json", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    reaction = _load_json(args.reaction_audit_json)
    trace = _load_json(args.trace_diag_json)
    summary_path = args.summary_json
    if summary_path is None and reaction.get("summary_json"):
        summary_path = Path(str(reaction["summary_json"]))
    summary = _load_json(summary_path) if summary_path is not None and summary_path.exists() else {}

    thresholds = reaction.get("thresholds", {})
    teacher_clip_max = _float(thresholds.get("teacher_max_diffik_clip_rate"), 0.5)
    teacher_tcp_max = _float(thresholds.get("teacher_max_final_tcp_err_m"), 0.03)
    reaction_gate_pass = _bool(reaction.get("reaction_gate_pass"))
    full_clip_rate = _float(reaction.get("summary_diffik_clip_rate_mean"))
    final_tcp_err_m = _float(reaction.get("summary_final_tcp_target_err_mean_m"))
    official_teacher_ready = _bool(reaction.get("teacher_quality_ready"))

    pre_clip_rate = _nested_float(trace, ("phase_splits", "pre_stop", "clip_any_rate"))
    post_clip_rate = _nested_float(trace, ("phase_splits", "post_stop", "clip_any_rate"))
    pre_rows = int(_nested_float(trace, ("phase_splits", "pre_stop", "rows")))
    post_rows = int(_nested_float(trace, ("phase_splits", "post_stop", "rows")))
    pre_follow_mean = _nested_float(trace, ("phase_splits", "pre_stop", "worst_follow_joint", "mean"))
    pre_follow_p95 = _nested_float(trace, ("phase_splits", "pre_stop", "worst_follow_joint", "p95"))
    max_joint_step_rad = _float(summary.get("max_diffik_joint_step_rad"))
    follow_p95_to_cap_ratio = pre_follow_p95 / max_joint_step_rad if max_joint_step_rad > 0.0 else 0.0

    full_clip_pass = full_clip_rate <= teacher_clip_max
    pre_clip_pass = pre_clip_rate <= teacher_clip_max
    tcp_pass = final_tcp_err_m <= teacher_tcp_max
    diagnostic_pre_stop_teacher_ready = reaction_gate_pass and tcp_pass and pre_clip_pass

    if not reaction_gate_pass:
        conclusion = "REACTION_GATE_BLOCKED"
    elif official_teacher_ready:
        conclusion = "OFFICIAL_TEACHER_READY"
    elif not pre_clip_pass:
        conclusion = "PRE_STOP_ACTUATOR_IK_CLIP_STILL_BLOCKS"
    elif not tcp_pass:
        conclusion = "TCP_ERROR_STILL_BLOCKS"
    elif not full_clip_pass and pre_clip_pass:
        conclusion = "OFFICIAL_GATE_BLOCKED_BY_POST_STOP_FREEZE_CLIP"
    else:
        conclusion = "INSPECT_UNCLASSIFIED_TEACHER_FALSE"

    result = {
        "branch": "professor_cube10cm_tap_reaction",
        "diagnostic_only": True,
        "official_next_step_gate_changed": False,
        "primary_objective": "tap_reaction_not_final_1cm",
        "reaction_gate_pass": reaction_gate_pass,
        "official_teacher_quality_ready": official_teacher_ready,
        "diagnostic_pre_stop_teacher_ready": diagnostic_pre_stop_teacher_ready,
        "teacher_clip_max": teacher_clip_max,
        "teacher_tcp_max_m": teacher_tcp_max,
        "full_clip_rate": full_clip_rate,
        "pre_stop_clip_rate": pre_clip_rate,
        "post_stop_clip_rate": post_clip_rate,
        "pre_stop_rows": pre_rows,
        "post_stop_rows": post_rows,
        "final_tcp_err_m": final_tcp_err_m,
        "pre_stop_worst_follow_mean_rad": pre_follow_mean,
        "pre_stop_worst_follow_p95_rad": pre_follow_p95,
        "max_diffik_joint_step_rad": max_joint_step_rad,
        "pre_stop_follow_p95_to_cap_ratio": follow_p95_to_cap_ratio,
        "conclusion": conclusion,
        "do_not_start": ["dataset_generation", "PPO_RL", "VLA", "TrackA", "1024_10k_scaleup"],
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[cube10cm_phase_teacher_window_audit] branch=professor_cube10cm_tap_reaction")
    print("[cube10cm_phase_teacher_window_audit] primary_objective=tap_reaction_not_final_1cm")
    print(f"[cube10cm_phase_teacher_window_audit] reaction_gate_pass={reaction_gate_pass}")
    print(f"[cube10cm_phase_teacher_window_audit] official_teacher_quality_ready={official_teacher_ready}")
    print(
        "[cube10cm_phase_teacher_window_audit] clips "
        f"full={full_clip_rate:.6f} pre_stop={pre_clip_rate:.6f} post_stop={post_clip_rate:.6f} "
        f"teacher_max={teacher_clip_max:.6f}"
    )
    print(
        "[cube10cm_phase_teacher_window_audit] follow "
        f"pre_stop_p95_rad={pre_follow_p95:.6f} max_joint_step_rad={max_joint_step_rad:.6f} "
        f"ratio={follow_p95_to_cap_ratio:.6f}"
    )
    print(f"[cube10cm_phase_teacher_window_audit] diagnostic_pre_stop_teacher_ready={diagnostic_pre_stop_teacher_ready}")
    print(f"[cube10cm_phase_teacher_window_audit] conclusion={conclusion}")
    print("[cube10cm_phase_teacher_window_audit] do_not_start=dataset_generation,PPO_RL,VLA,TrackA,1024_10k_scaleup")
    if args.out_json is not None:
        print(f"[cube10cm_phase_teacher_window_audit] out_json={args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
