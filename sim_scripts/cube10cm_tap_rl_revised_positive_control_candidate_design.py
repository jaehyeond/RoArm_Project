"""Design the single revised positive-control candidate after the first failure.

This is local design/static audit only. It does not launch IsaacLab, run GPU
physics, build datasets, train, control a robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
HARNESS = REPO / "roarm_rl/test_positive_control_cube_tap10cm.py"
FAILURE_AUDIT = LOG_DIR / "cube10cm_tap_rl_positive_control_failure_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_revised_positive_control_candidate_design.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_revised_positive_control_candidate_design_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_of(path: Path, pattern: str) -> int | None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure_audit", type=Path, default=FAILURE_AUDIT)
    parser.add_argument("--harness", type=Path, default=HARNESS)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    failure = _load(args.failure_audit)
    code_lines = {
        "controller_arg": _line_of(args.harness, 'choices=("builtin_teacher", "external_closed_loop")'),
        "closed_loop_fn": _line_of(args.harness, "def _closed_loop_ik_action"),
        "closed_loop_action_call": _line_of(args.harness, 'if args.controller_mode == "external_closed_loop"'),
        "action_smoothing_arg": _line_of(args.harness, 'parser.add_argument("--action_smoothing_alpha"'),
        "contact_delta_scale_arg": _line_of(args.harness, 'parser.add_argument("--contact_joint_delta_scale"'),
    }
    code_ready = all(line is not None for line in code_lines.values())
    failure_mode_ok = (
        failure.get("positive_runtime_valid") is True
        and failure.get("failure_is_controller_gap") is True
        and failure.get("wrapper_blocked_false_positive") is True
    )
    candidate = {
        "status": "DESIGNED_NOT_RUN",
        "requires_explicit_gpu_approval": True,
        "command": (
            "conda run -n isaaclab --no-capture-output python -u "
            "-m roarm_rl.test_positive_control_cube_tap10cm "
            "--num_envs 2 --steps 120 --seed 962 --device cuda:0 "
            "--controller_mode external_closed_loop "
            "--action_smoothing_alpha 1.0 --contact_joint_delta_scale 1.0 "
            "--closed_loop_push_steps 72"
        ),
        "fixed": {
            "env_id": "RoArm-CubeTap10cm-Direct-v0",
            "cube_xy_m": [0.250, 0.000],
            "push_dir": [1.0, 0.0],
            "precontact_clearance_m": 0.020,
            "tcp_top_margin_m": -0.050,
            "goal_push_m": 0.006,
            "num_envs": 2,
            "max_steps": 120,
            "device": "cuda:0",
        },
        "one_variable_intent": "controller_execution_only",
        "does_not_change": [
            "10cm/0.72kg object contract",
            "tap/reaction/contact objective",
            "final 1cm remains default-off",
            "reward/done/pass-fail gate",
            "dataset/RL/RoArm state",
        ],
        "pass_gate": {
            "contact_seen_gt": 0.0,
            "reaction_contact_context_gt": 0.0,
            "reaction_seen_gt": 0.0,
            "tap_success_gt": 0.0,
            "overshoot_eq": 0.0,
            "final_flag_eq": 0.0,
        },
    }
    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_revised_positive_control_candidate_design_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_design_static_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "failure_mode_ok": failure_mode_ok,
        "code_ready": code_ready,
        "code_lines": code_lines,
        "prior_failure": {
            "final_gap_shortfall_to_contact_band_m": failure.get("final_gap_shortfall_to_contact_band_m"),
            "raw_reaction_without_context": failure.get("raw_reaction_without_context"),
            "wrapper_blocked_false_positive": failure.get("wrapper_blocked_false_positive"),
        },
        "candidate": candidate,
        "verdict": "READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY" if code_ready and failure_mode_ok else "BLOCKED",
        "still_blocked": {
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "action_teacher_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_revised_positive_control_candidate_design_v1 "
        "local_design_static_audit_only=YES gpu_runtime=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 failure_basis "
            f"failure_mode_ok={failure_mode_ok} "
            f"gap_shortfall_m={float(failure.get('final_gap_shortfall_to_contact_band_m', 0.0)):.9f} "
            f"raw_reaction_without_context={failure.get('raw_reaction_without_context')} "
            f"wrapper_blocked_false_positive={failure.get('wrapper_blocked_false_positive')}"
        ),
        (
            "line3 candidate "
            f"code_ready={code_ready} controller_mode=external_closed_loop "
            "action_smoothing_alpha=1.0 contact_joint_delta_scale=1.0 "
            "closed_loop_push_steps=72 side_center_tcp=YES status=DESIGNED_NOT_RUN"
        ),
        (
            "line4 pass_gate "
            "contact_seen>0 reaction_context>0 reaction_seen>0 tap_success>0 "
            "overshoot=0 final_flag=0"
        ),
        (
            "line5 verdict "
            f"ready_for_explicit_runtime_approval_only={code_ready and failure_mode_ok} "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED action_teacher=BLOCKED roarm=BLOCKED"
        ),
        (
            "line6 next "
            "new_gpu_runtime=REQUIRES_EXPLICIT_APPROVAL "
            "not_allowed=run_without_approval_or_ppo_large_dataset_action_teacher_roarm"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if code_ready and failure_mode_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
