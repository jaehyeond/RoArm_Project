"""Design the next direct-IK telemetry diagnostic after direct apply failed.

This is a local static/design audit only. It does not launch IsaacLab, run GPU
physics, build datasets, train, control a robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DIRECT_AUDIT = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_result_audit.json"
HARNESS = REPO / "roarm_rl/test_positive_control_cube_tap10cm.py"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_direct_ik_telemetry_candidate_design.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_direct_ik_telemetry_candidate_design_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_of(path: Path, needle: str) -> int:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return idx
    return -1


def _f(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_audit", type=Path, default=DIRECT_AUDIT)
    parser.add_argument("--harness", type=Path, default=HARNESS)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    audit = _load(args.direct_audit)
    line_checks = {
        "target_face_gap": _line_of(args.harness, "closed_loop_target_face_gap_m_mean"),
        "target_inside_contact_band": _line_of(args.harness, "closed_loop_target_inside_contact_band_rate"),
        "target_fk_err": _line_of(args.harness, "closed_loop_target_fk_err_mm_mean"),
        "actual_fk_vs_sim_tcp": _line_of(args.harness, "closed_loop_actual_fk_vs_sim_tcp_err_mm_mean"),
        "direct_joint_follow": _line_of(args.harness, "direct_joint_follow_abs_max_rad"),
        "direct_actual_joint_step": _line_of(args.harness, "direct_actual_joint_step_abs_max_rad"),
        "summary_line10": _line_of(args.harness, "line10 controller_telemetry"),
    }
    telemetry_ready = all(value > 0 for value in line_checks.values())
    basis_ok = (
        bool(audit.get("wrapper_only_explanation_falsified"))
        and not bool(audit.get("direct_ik_apply_pass"))
        and bool(audit.get("along_gap_blocker"))
        and bool(audit.get("lateral_ok"))
        and bool(audit.get("vertical_ok"))
    )
    no_control_knob_change = True
    next_json = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity.json"
    next_summary = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity_summary.out"
    command = (
        "conda run -n isaaclab --no-capture-output python -u "
        "-m roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 120 --seed 962 --device cuda:0 "
        "--controller_mode external_closed_loop_direct_apply "
        f"--out_json {next_json} --out_summary {next_summary}"
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_direct_ik_telemetry_candidate_design_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_design_static_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "basis_ok": basis_ok,
        "telemetry_ready": telemetry_ready,
        "line_checks": line_checks,
        "previous_direct_contact_seen": audit.get("contact_seen"),
        "previous_direct_best_shortfall_m": audit.get("best_shortfall_to_contact_band_m"),
        "previous_direct_best_improvement_m": audit.get("best_improvement_from_initial_m"),
        "previous_wrapper_only_explanation_falsified": audit.get("wrapper_only_explanation_falsified"),
        "selected_candidate": {
            "name": "direct_ik_apply_telemetry_repeat",
            "status": "DESIGNED_NOT_RUN",
            "purpose": "separate_target_geometry_fk_frame_and_actuator_follow_before_any_more_control_tuning",
            "control_knobs_changed_vs_direct_apply": 0,
            "controller_mode": "external_closed_loop_direct_apply",
            "num_envs": 2,
            "max_steps": 120,
            "seed": 962,
            "device": "cuda:0",
            "out_json": str(next_json),
            "out_summary": str(next_summary),
            "command": command,
        },
        "rejected": {
            "cap040_lead120": "reserve; would tune target application before knowing direct target/follow cause",
            "joint_delta_reference": "reserve; belongs to RL action wrapper, which direct apply bypasses",
            "action_scale_or_smoothing": "rejected now; wrapper path is not the current immediate question",
            "geometry_change": "rejected until target geometry telemetry says target itself is wrong",
            "dataset_or_rl": "blocked until contact-gated positive control passes",
        },
        "verdict": (
            "READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY"
            if basis_ok and telemetry_ready and no_control_knob_change
            else "NOT_READY_REVIEW_TELEMETRY_PATCH"
        ),
        "still_blocked": {
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
        "line1 artifact=cube10cm_tap_rl_direct_ik_telemetry_candidate_design_v1 "
        "local_design_static_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 basis "
            f"basis_ok={basis_ok} telemetry_ready={telemetry_ready} "
            f"previous_contact_seen={audit.get('contact_seen')} "
            f"previous_best_shortfall_m={_f(audit, 'best_shortfall_to_contact_band_m'):.9f} "
            f"wrapper_only_explanation_falsified={audit.get('wrapper_only_explanation_falsified')}"
        ),
        (
            "line3 line_checks "
            + " ".join(f"{key}={value}" for key, value in line_checks.items())
        ),
        (
            "line4 selected_candidate "
            "status=DESIGNED_NOT_RUN name=direct_ik_apply_telemetry_repeat "
            "control_knobs_changed_vs_direct_apply=0 controller_mode=external_closed_loop_direct_apply "
            "num_envs=2 max_steps=120 seed=962 device=cuda:0"
        ),
        (
            "line5 purpose "
            "separate_target_geometry_fk_frame_and_actuator_follow_before_any_more_control_tuning "
            "cap040_lead120=reserve joint_delta_reference=reserve action_scale_smoothing=not_now geometry_change=not_now"
        ),
        (
            "line6 verdict "
            f"{result['verdict']} diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
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
