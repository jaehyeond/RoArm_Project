"""Design the next single action-progress candidate for the 10cm tap wrapper.

This is local design/static audit only. It reads existing logs and source files;
it does not launch IsaacLab, run GPU physics, build datasets, train, control a
robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
RESULT_AUDIT = LOG_DIR / "cube10cm_tap_rl_external_closed_loop_tcp_progress_result_audit.json"
RUNTIME_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_tcp_progress_sanity.json"
HARNESS = REPO / "roarm_rl/test_positive_control_cube_tap10cm.py"
ENV_SOURCE = REPO / "roarm_rl/roarm_cube_push_env.py"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_action_progress_candidate_design.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_action_progress_candidate_design_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_of(path: Path, pattern: str) -> int | None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_audit", type=Path, default=RESULT_AUDIT)
    parser.add_argument("--runtime_json", type=Path, default=RUNTIME_JSON)
    parser.add_argument("--harness", type=Path, default=HARNESS)
    parser.add_argument("--env_source", type=Path, default=ENV_SOURCE)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    audit = _load(args.result_audit)
    runtime = _load(args.runtime_json)

    code_lines = {
        "action_smoothing_default": _line_of(args.env_source, "action_smoothing_alpha: float = 0.25"),
        "action_smoothing_apply": _line_of(args.env_source, "self._smoothed_actions[:] ="),
        "delta_cap_apply": _line_of(args.env_source, "max_joint_delta_per_step_rad"),
        "contact_slowdown_apply": _line_of(args.env_source, "contact_slowdown_tcp_dist_m"),
        "target_reference_apply": _line_of(args.env_source, 'reference == "target"'),
        "target_write": _line_of(args.env_source, "self.robot_dof_targets[:] = targets"),
        "harness_action_smoothing_arg": _line_of(args.harness, 'parser.add_argument("--action_smoothing_alpha"'),
        "harness_action_smoothing_override": _line_of(args.harness, "cfg.action_smoothing_alpha ="),
        "harness_closed_loop_alpha": _line_of(args.harness, "closed_loop_alpha"),
        "harness_trace_summary": _line_of(args.harness, "line8 trace_diagnostics"),
    }
    code_ready = all(value is not None for value in code_lines.values())

    previous = {
        "controller_mode": audit.get("controller_mode"),
        "action_smoothing_alpha": _f(audit.get("action_smoothing_alpha")),
        "contact_joint_delta_scale": _f(audit.get("contact_joint_delta_scale")),
        "closed_loop_push_steps": int(_f(audit.get("closed_loop_push_steps"), 0.0)),
        "external_runtime_valid": bool(audit.get("external_runtime_valid")),
        "strict_knobs_ok": bool(audit.get("strict_knobs_ok")),
        "contact_seen": _f(audit.get("external_contact_seen")),
        "reaction_context": _f(audit.get("external_reaction_context")),
        "tap_success": _f(audit.get("external_tap_success")),
        "face_gap_moved_toward_band": bool(audit.get("face_gap_moved_toward_band")),
        "face_gap_near_band": bool(audit.get("face_gap_near_band")),
        "initial_face_gap_m": _f(audit.get("initial_face_gap_m")),
        "face_gap_best_m": _f(audit.get("face_gap_best_m")),
        "face_gap_best_improvement_from_initial_m": _f(audit.get("face_gap_best_improvement_from_initial_m")),
        "shortfall_best_m": _f(audit.get("shortfall_best_m")),
        "shortfall_final_m": _f(audit.get("shortfall_final_trace_m")),
        "tcp_dist_min_m": _f(audit.get("tcp_dist_min_m")),
        "joint_delta_abs_max": _f(audit.get("joint_delta_abs_max")),
        "contact_slowdown_mean": _f(audit.get("contact_slowdown_mean")),
        "closed_loop_alpha_final": _f(
            runtime.get("controller_trace_stats", {})
            .get("closed_loop_alpha", {})
            .get("final")
        ),
    }

    basis_ok = (
        previous["external_runtime_valid"]
        and previous["strict_knobs_ok"]
        and previous["controller_mode"] == "external_closed_loop"
        and previous["contact_seen"] == 0.0
        and previous["reaction_context"] == 0.0
        and previous["tap_success"] == 0.0
        and previous["face_gap_moved_toward_band"]
        and not previous["face_gap_near_band"]
        and previous["contact_slowdown_mean"] >= 0.999
        and previous["closed_loop_alpha_final"] >= 0.999
    )

    out_json = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_action_smoothing1_sanity.json"
    out_summary = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_action_smoothing1_sanity_summary.out"
    command = (
        "conda run -n isaaclab --no-capture-output python -u "
        "-m roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 120 --seed 962 --device cuda:0 "
        "--controller_mode external_closed_loop "
        "--action_smoothing_alpha 1.0 "
        f"--out_json {out_json} "
        f"--out_summary {out_summary}"
    )

    candidate = {
        "status": "DESIGNED_NOT_RUN",
        "requires_explicit_gpu_approval": True,
        "command": command,
        "selected_runtime_knob_changes_count": 1,
        "selected_runtime_knob_changes": [
            "action_smoothing_alpha: 0.25 -> 1.0",
        ],
        "one_variable_intent": "action_progress_gain_only",
        "fixed": {
            "env_id": "RoArm-CubeTap10cm-Direct-v0",
            "device": "cuda:0",
            "num_envs": 2,
            "max_steps": 120,
            "seed": 962,
            "controller_mode": "external_closed_loop",
            "closed_loop_push_steps": 72,
            "cube_xy_m": [0.250, 0.000],
            "push_dir": [1.0, 0.0],
            "precontact_clearance_m": 0.020,
            "tcp_top_margin_m": -0.050,
            "goal_push_m": 0.006,
            "contact_joint_delta_scale": 0.35,
        },
        "does_not_change": [
            "10cm/0.72kg object contract",
            "tap/reaction/contact objective",
            "final 1cm remains default-off",
            "contact band and reaction thresholds",
            "cube placement and push direction",
            "side-center TCP height",
            "precontact clearance and through distance",
            "controller mode",
            "closed-loop push timing",
            "contact slowdown scale",
            "dataset/RL/RoArm state",
        ],
        "primary_pass_gate": {
            "contact_seen_gt": 0.0,
            "reaction_contact_context_gt": 0.0,
            "reaction_seen_gt": 0.0,
            "tap_success_gt": 0.0,
            "overshoot_eq": 0.0,
            "final_1cm_required_eq": False,
        },
        "diagnostic_readout_if_primary_fails": [
            "face_gap_best_m",
            "shortfall_best_m",
            "joint_delta_abs_max",
            "tcp_dist_min_m",
            "contact_slowdown_mean",
        ],
    }

    not_selected = {
        "contact_joint_delta_scale": {
            "status": "NOT_SELECTED",
            "reason": "latest runtime recorded contact_slowdown_mean=1.0, so the contact slowdown path was inactive",
        },
        "max_joint_delta_per_step_rad": {
            "status": "NOT_SELECTED_FIRST",
            "reason": "latest scalar trace recorded joint_delta_abs_max below the 0.010rad cap; per-joint cap may still be audited later",
        },
        "closed_loop_push_steps": {
            "status": "NOT_SELECTED",
            "reason": "closed_loop_alpha reached 1.0, and the trace never approached the contact band",
        },
        "goal_push_or_contact_band": {
            "status": "NOT_SELECTED",
            "reason": "would change through/contact geometry instead of the action application path",
        },
        "joint_delta_reference": {
            "status": "RESERVE_NOT_SELECTED",
            "reason": "plausible target-application knob, but current logs do not yet show target-vs-joint lead as the dominant cause",
        },
    }

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_action_progress_candidate_design_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_design_static_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "result_audit": str(args.result_audit),
        "runtime_json": str(args.runtime_json),
        "code_ready": code_ready,
        "basis_ok": basis_ok,
        "code_lines": code_lines,
        "previous_runtime": previous,
        "candidate": candidate,
        "not_selected": not_selected,
        "verdict": "READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY" if code_ready and basis_ok else "BLOCKED",
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
        "line1 artifact=cube10cm_tap_rl_action_progress_candidate_design_v1 "
        "local_design_static_audit_only=YES gpu_runtime=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 basis "
            f"code_ready={code_ready} basis_ok={basis_ok} "
            f"contact_seen={previous['contact_seen']:.1f} reaction_context={previous['reaction_context']:.1f} "
            f"face_gap_moved_toward_band={previous['face_gap_moved_toward_band']} "
            f"face_gap_near_band={previous['face_gap_near_band']} "
            f"best_improvement_m={previous['face_gap_best_improvement_from_initial_m']:.9f} "
            f"shortfall_best_m={previous['shortfall_best_m']:.9f}"
        ),
        (
            "line3 action_path_basis "
            f"previous_action_smoothing_alpha={previous['action_smoothing_alpha']:.2f} "
            f"closed_loop_alpha_final={previous['closed_loop_alpha_final']:.1f} "
            f"contact_slowdown_mean={previous['contact_slowdown_mean']:.1f} "
            f"joint_delta_abs_max={previous['joint_delta_abs_max']:.9f} "
            f"tcp_dist_min_m={previous['tcp_dist_min_m']:.9f}"
        ),
        (
            "line4 selected_candidate "
            "status=DESIGNED_NOT_RUN changed_knobs=1 "
            "action_smoothing_alpha=0.25->1.0 "
            "controller_mode=external_closed_loop_fixed contact_joint_delta_scale=0.35_fixed "
            "closed_loop_push_steps=72_fixed geometry=FIXED"
        ),
        (
            "line5 not_selected "
            "contact_joint_delta_scale=inactive_slowdown "
            "closed_loop_push_steps=alpha_already_1.0 "
            "goal_push_or_contact_band=geometry_change "
            "joint_delta_reference=reserve_needs_target_joint_lead_evidence"
        ),
        (
            "line6 pass_gate "
            "contact_seen>0 reaction_context>0 reaction_seen>0 tap_success>0 "
            "overshoot=0 final_1cm_required=False"
        ),
        (
            "line7 next_runtime_command "
            f"{command}"
        ),
        (
            "line8 verdict "
            f"ready_for_explicit_runtime_approval_only={code_ready and basis_ok} "
            "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if code_ready and basis_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
