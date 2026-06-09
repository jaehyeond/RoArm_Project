"""Design the next one-knob actuator-follow timing candidate.

This is a local design audit only. It reads the telemetry result audit and does
not launch IsaacLab/GPU runtime, generate datasets, train, control RoArm, SSH,
or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_TELEMETRY_AUDIT = LOG_DIR / "cube10cm_tap_rl_direct_ik_telemetry_result_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_actuator_follow_time_scale_candidate_design.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_actuator_follow_time_scale_candidate_design_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_plan(push_steps: int, max_steps: int = 120) -> dict[str, Any]:
    pre_face_gap = -0.020
    through_face_gap = 0.106
    span = through_face_gap - pre_face_gap
    band = 0.010
    inside = []
    gaps = []
    for step in range(max_steps):
        alpha = min(1.0, max(0.0, float(step + 1) / max(float(push_steps), 1.0)))
        face_gap = pre_face_gap + alpha * span
        gaps.append(face_gap)
        if -band <= face_gap <= band:
            inside.append(step + 1)
    return {
        "closed_loop_push_steps": int(push_steps),
        "max_steps": int(max_steps),
        "target_face_gap_rate_m_per_step": span / max(float(push_steps), 1.0),
        "target_inside_contact_band_step_count": len(inside),
        "target_inside_contact_band_first_step_1based": inside[0] if inside else None,
        "target_inside_contact_band_last_step_1based": inside[-1] if inside else None,
        "target_final_face_gap_m": gaps[-1] if gaps else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry_audit", type=Path, default=DEFAULT_TELEMETRY_AUDIT)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    audit = _load_json(args.telemetry_audit)
    outcome = audit.get("outcome", {})
    follow = audit.get("action_and_follow_split", {})
    target = audit.get("target_and_frame_split", {})

    primary_blocker = str(outcome.get("primary_blocker", "UNKNOWN"))
    basis_ok = (
        primary_blocker == "DIRECT_JOINT_FOLLOW_ACTUATOR_TRACKING_LAG"
        and bool(target.get("target_path_ok")) is True
        and bool(target.get("fk_frame_ok")) is True
        and bool(follow.get("wrapper_path_bypassed")) is True
        and bool(follow.get("final_step_near_velocity_limit")) is True
    )

    baseline = _target_plan(72)
    candidate = _target_plan(240)
    dwell_gain = (
        float(candidate["target_inside_contact_band_step_count"])
        / max(float(baseline["target_inside_contact_band_step_count"]), 1.0)
    )
    rate_ratio = (
        float(candidate["target_face_gap_rate_m_per_step"])
        / max(float(baseline["target_face_gap_rate_m_per_step"]), 1.0e-9)
    )

    next_json = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_slow240_sanity.json"
    next_summary = (
        LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_slow240_sanity_summary.out"
    )
    command = (
        "conda run -n isaaclab --no-capture-output python -u "
        "-m roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 120 --seed 962 --device cuda:0 "
        "--controller_mode external_closed_loop_direct_apply --closed_loop_push_steps 240 "
        f"--out_json {next_json} --out_summary {next_summary}"
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_actuator_follow_time_scale_candidate_design_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_design_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "basis": {
            "basis_ok": basis_ok,
            "primary_blocker": primary_blocker,
            "final_step_near_velocity_limit": follow.get("final_step_near_velocity_limit"),
            "actual_velocity_est_max_final_rad_s": follow.get("actual_velocity_est_max_final_rad_s"),
            "velocity_limit_rad_s": follow.get("velocity_limit_rad_s"),
            "direct_joint_follow_abs_max_final_rad": follow.get("direct_joint_follow_abs_max_final_rad"),
            "target_path_ok": target.get("target_path_ok"),
            "fk_frame_ok": target.get("fk_frame_ok"),
            "wrapper_path_bypassed": follow.get("wrapper_path_bypassed"),
        },
        "baseline_target_plan": baseline,
        "candidate_target_plan": candidate,
        "candidate": {
            "name": "direct_ik_apply_slow240",
            "status": "DESIGNED_NOT_RUN",
            "changed_knobs": 1,
            "changed_knob": "closed_loop_push_steps 72 -> 240",
            "unchanged": {
                "controller_mode": "external_closed_loop_direct_apply",
                "num_envs": 2,
                "max_steps": 120,
                "seed": 962,
                "device": "cuda:0",
                "geometry": "unchanged",
                "precontact_clearance_m": 0.020,
                "tcp_top_margin_m": -0.050,
                "goal_push_m": 0.006,
                "max_joint_delta_per_step_rad": 0.010,
                "joint_target_lead_limit_rad": 0.060,
                "action_scale": "unchanged",
                "action_smoothing_alpha": "unchanged",
            },
            "target_face_gap_rate_ratio_vs_baseline": rate_ratio,
            "contact_band_dwell_gain_vs_baseline": dwell_gain,
            "out_json": str(next_json),
            "out_summary": str(next_summary),
            "command": command,
        },
        "rejected_now": {
            "geometry_change": "telemetry showed target path and frame are OK",
            "lead_cap_action_scale_sweep": "direct apply bypassed wrapper/cap/lead path",
            "dataset_or_rl": "blocked until contact-gated positive-control passes",
            "roarm": "blocked until policy and safety gates pass",
        },
        "verdict": "READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY" if basis_ok else "NOT_READY_RECHECK_TELEMETRY",
        "still_blocked": {
            "contact_gated_positive_control": "RUN_FAILED",
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_actuator_follow_time_scale_candidate_design_v1 "
        "local_design_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 basis "
            f"basis_ok={basis_ok} primary_blocker={primary_blocker} "
            f"final_step_near_velocity_limit={follow.get('final_step_near_velocity_limit')} "
            f"actual_velocity_est_max_final_rad_s={float(follow.get('actual_velocity_est_max_final_rad_s', 0.0)):.9f} "
            f"velocity_limit_rad_s={float(follow.get('velocity_limit_rad_s', 0.0)):.9f}"
        ),
        (
            "line3 selected_candidate "
            "status=DESIGNED_NOT_RUN name=direct_ik_apply_slow240 changed_knobs=1 "
            "closed_loop_push_steps=72->240 controller_mode=external_closed_loop_direct_apply "
            "num_envs=2 max_steps=120 seed=962 device=cuda:0"
        ),
        (
            "line4 target_timing "
            f"baseline_inside_steps={baseline['target_inside_contact_band_step_count']} "
            f"candidate_inside_steps={candidate['target_inside_contact_band_step_count']} "
            f"dwell_gain={dwell_gain:.9f} "
            f"target_face_gap_rate_ratio={rate_ratio:.9f} "
            f"candidate_final_face_gap_m={float(candidate['target_final_face_gap_m']):.9f}"
        ),
        (
            "line5 rejected "
            "geometry_change=NO lead_cap_action_scale_sweep=NO dataset_rl_roarm=NO "
            "reason=target_frame_ok_and_direct_follow_lag_primary"
        ),
        (
            "line6 verdict "
            f"{result['verdict']} contact_gated_positive_control=RUN_FAILED "
            "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
        f"line7 command {command}",
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if result["verdict"] == "READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
