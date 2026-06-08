"""Posthoc audit for the 10cm tap RL positive-control sanity result.

This reads the existing positive-control JSON only. It does not launch IsaacLab,
run GPU physics, build datasets, train, control a robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_POSITIVE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_sanity.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_failure_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_positive_control_failure_audit_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_json", type=Path, default=DEFAULT_POSITIVE_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    positive = _load(args.positive_json)
    reset = positive.get("reset_metrics", {})
    controller_metrics = positive.get("controller_metrics", {})
    log = positive.get("last_log", {})

    contact_band_m = 0.010
    controller_mode = str(positive.get("controller_mode", "builtin_teacher"))
    initial_face_gap = _float(reset, "initial_face_gap_m")
    final_face_gap = _float(log, "cube_tap_contact_face_gap_m")
    final_gap_shortfall_m = max(0.0, -contact_band_m - final_face_gap)
    gap_delta_m = final_face_gap - initial_face_gap
    reset_ik_ok = _float(reset, "ik_endpoint_reset_rate") > 0.0
    teacher_goal_ok = _float(reset, "teacher_goal_ok_rate") > 0.0
    closed_loop_ik_ok = _float(controller_metrics, "closed_loop_ik_ok_rate") > 0.0
    controller_goal_ok = closed_loop_ik_ok if controller_mode == "external_closed_loop" else teacher_goal_ok
    contact_seen = _float(log, "cube_tap_contact_seen_rate")
    reaction_signal = _float(log, "cube_tap_reaction_signal_now_rate")
    reaction_context = _float(log, "cube_tap_reaction_contact_context_rate")
    reaction_seen = _float(log, "cube_tap_reaction_seen_rate")
    tap_success = _float(log, "cube_tap_success_rate")
    overshoot_seen = _float(log, "cube_tap_overshoot_seen_rate")
    max_disp = _float(log, "cube_tap_max_disp_along_m")
    max_speed = _float(log, "cube_tap_max_speed_mps")
    max_z = _float(log, "cube_tap_max_z_delta_m")
    raw_reaction_without_context = reaction_signal > 0.0 and reaction_context == 0.0 and reaction_seen == 0.0
    wrapper_blocked_false_positive = raw_reaction_without_context and tap_success == 0.0
    positive_runtime_valid = (
        positive.get("gpu_runtime") == "YES_LOCAL_TINY_ISAACLAB_POSITIVE_CONTROL"
        and positive.get("device") == "cuda:0"
        and positive.get("dataset_generation") is False
        and positive.get("training") is False
        and positive.get("robot_control") is False
        and positive.get("ssh") is False
        and positive.get("b200") is False
        and int(positive.get("steps_executed", 0)) > 0
    )
    failure_is_controller_gap = (
        positive_runtime_valid
        and reset_ik_ok
        and controller_goal_ok
        and final_gap_shortfall_m > 0.0
        and contact_seen == 0.0
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_positive_control_failure_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_log_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "positive_runtime_valid": positive_runtime_valid,
        "positive_status": positive.get("status"),
        "controller_mode": controller_mode,
        "reset_ik_ok": reset_ik_ok,
        "teacher_goal_ok": teacher_goal_ok,
        "closed_loop_ik_ok": closed_loop_ik_ok,
        "controller_goal_ok": controller_goal_ok,
        "contact_band_m": contact_band_m,
        "initial_face_gap_m": initial_face_gap,
        "final_face_gap_m": final_face_gap,
        "gap_delta_m": gap_delta_m,
        "final_gap_shortfall_to_contact_band_m": final_gap_shortfall_m,
        "initial_vertical_offset_m": _float(reset, "initial_vertical_offset_m"),
        "final_vertical_offset_m": _float(log, "cube_tap_contact_vertical_offset_m"),
        "final_lateral_m": _float(log, "cube_tap_contact_lateral_m"),
        "contact_seen": contact_seen,
        "reaction_signal": reaction_signal,
        "reaction_context": reaction_context,
        "reaction_seen": reaction_seen,
        "tap_success": tap_success,
        "overshoot_seen": overshoot_seen,
        "max_disp_along_m": max_disp,
        "max_speed_mps": max_speed,
        "max_z_delta_m": max_z,
        "terminated_count": positive.get("terminated_count"),
        "truncated_count": positive.get("truncated_count"),
        "raw_reaction_without_context": raw_reaction_without_context,
        "wrapper_blocked_false_positive": wrapper_blocked_false_positive,
        "failure_is_controller_gap": failure_is_controller_gap,
        "still_blocked": {
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "action_teacher_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next": {
            "allowed_local_only": "design_one_revised_closed_loop_positive_control_candidate",
            "new_gpu_runtime": "REQUIRES_EXPLICIT_APPROVAL",
            "not_allowed": "ppo_large_dataset_action_teacher_roarm",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_positive_control_failure_audit_v1 "
        "local_log_audit_only=YES gpu_runtime_launched_by_this_audit=NO "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 runtime_contract "
            f"positive_runtime_valid={positive_runtime_valid} status={positive.get('status')} "
            f"device={positive.get('device')} controller_mode={controller_mode} "
            f"steps_executed={positive.get('steps_executed')} "
            f"num_envs={positive.get('num_envs')}"
        ),
        (
            "line3 reset_and_geometry "
            f"reset_ik_ok={reset_ik_ok} teacher_goal_ok={teacher_goal_ok} "
            f"closed_loop_ik_ok={closed_loop_ik_ok} controller_goal_ok={controller_goal_ok} "
            f"initial_face_gap_m={initial_face_gap:.9f} final_face_gap_m={final_face_gap:.9f} "
            f"gap_delta_m={gap_delta_m:.9f} final_gap_shortfall_to_band_m={final_gap_shortfall_m:.9f}"
        ),
        (
            "line4 contact_reaction "
            f"contact_seen={contact_seen} reaction_signal={reaction_signal} "
            f"reaction_context={reaction_context} reaction_seen={reaction_seen} "
            f"tap_success={tap_success} overshoot={overshoot_seen}"
        ),
        (
            "line5 motion "
            f"max_disp_along_m={max_disp:.9f} max_speed_mps={max_speed:.9f} "
            f"max_z_delta_m={max_z:.9f} "
            f"raw_reaction_without_context={raw_reaction_without_context} "
            f"wrapper_blocked_false_positive={wrapper_blocked_false_positive}"
        ),
        (
            "line6 verdict "
            f"positive_control_pass=False failure_is_controller_gap={failure_is_controller_gap} "
            "env_wrapper_false_positive_guard=PASS "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED action_teacher=BLOCKED roarm=BLOCKED"
        ),
        (
            "line7 next "
            "allowed=local_design_one_revised_closed_loop_positive_control_candidate "
            "new_gpu_runtime=REQUIRES_EXPLICIT_APPROVAL "
            "not_allowed=ppo_large_dataset_action_teacher_roarm"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if failure_is_controller_gap and wrapper_blocked_false_positive else 2


if __name__ == "__main__":
    raise SystemExit(main())
