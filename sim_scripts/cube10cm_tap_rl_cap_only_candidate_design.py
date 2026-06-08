"""Design the next cap-only positive-control candidate for the 10cm tap wrapper.

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
CAP_AUDIT = LOG_DIR / "cube10cm_tap_rl_cap_targetlead_result_audit.json"
ENV_SOURCE = REPO / "roarm_rl/roarm_cube_push_env.py"
HARNESS = REPO / "roarm_rl/test_positive_control_cube_tap10cm.py"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_cap_only_candidate_design.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_cap_only_candidate_design_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_of(path: Path, pattern: str) -> int | None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_audit", type=Path, default=CAP_AUDIT)
    parser.add_argument("--env_source", type=Path, default=ENV_SOURCE)
    parser.add_argument("--harness", type=Path, default=HARNESS)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    audit = _load(args.cap_audit)
    code_lines = {
        "env_action_scale_default": _line_of(args.env_source, "action_scale: float = 0.04"),
        "env_cap_default": _line_of(args.env_source, "max_joint_delta_per_step_rad: float = 0.010"),
        "env_cap_apply": _line_of(args.env_source, "max_delta = float(self.cfg.max_joint_delta_per_step_rad)"),
        "harness_cap_arg": _line_of(args.harness, 'parser.add_argument("--max_joint_delta_per_step_rad"'),
        "harness_cap_override": _line_of(args.harness, "cfg.max_joint_delta_per_step_rad ="),
        "harness_cap_result": _line_of(args.harness, '"max_joint_delta_per_step_rad": float(cfg.max_joint_delta_per_step_rad)'),
    }
    code_ready = all(value is not None for value in code_lines.values())
    basis_ok = (
        audit.get("runtime_valid") is True
        and audit.get("cap_is_primary_current_hypothesis") is True
        and audit.get("action_saturation_observed") is True
        and audit.get("per_joint_cap_observed") is True
        and audit.get("slowdown_observed") is False
    )

    out_json = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap040_sanity.json"
    out_summary = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap040_sanity_summary.out"
    command = (
        "conda run -n isaaclab --no-capture-output python -u "
        "-m roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 120 --seed 962 --device cuda:0 "
        "--controller_mode external_closed_loop "
        "--max_joint_delta_per_step_rad 0.040 "
        f"--out_json {out_json} "
        f"--out_summary {out_summary}"
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_cap_only_candidate_design_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_design_static_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "code_ready": code_ready,
        "basis_ok": basis_ok,
        "code_lines": code_lines,
        "diagnostic_basis": {
            "action_abs_max_trace": audit.get("action_abs_max_trace"),
            "joint_delta_abs_max_trace": audit.get("joint_delta_abs_max_trace"),
            "joint_delta_cap_rate_trace": audit.get("joint_delta_cap_rate_trace"),
            "target_lead_limit_rate_trace": audit.get("target_lead_limit_rate_trace"),
            "cap_is_primary_current_hypothesis": audit.get("cap_is_primary_current_hypothesis"),
        },
        "candidate": {
            "status": "DESIGNED_NOT_RUN",
            "requires_explicit_gpu_approval": True,
            "command": command,
            "selected_runtime_knob_changes_count": 1,
            "selected_runtime_knob_changes": [
                "max_joint_delta_per_step_rad: 0.010 -> 0.040",
            ],
            "one_variable_intent": "remove_env_delta_cap_as_primary_limiter_while_preserving_action_scale",
            "fixed": {
                "env_id": "RoArm-CubeTap10cm-Direct-v0",
                "device": "cuda:0",
                "num_envs": 2,
                "max_steps": 120,
                "seed": 962,
                "controller_mode": "external_closed_loop",
                "action_smoothing_alpha": 0.25,
                "contact_joint_delta_scale": 0.35,
                "closed_loop_push_steps": 72,
                "cube_xy_m": [0.250, 0.000],
                "push_dir": [1.0, 0.0],
                "precontact_clearance_m": 0.020,
                "tcp_top_margin_m": -0.050,
                "goal_push_m": 0.006,
            },
            "does_not_change": [
                "10cm/0.72kg object contract",
                "tap/reaction/contact objective",
                "final 1cm remains default-off",
                "contact band and reaction thresholds",
                "action_scale",
                "action_smoothing_alpha",
                "contact slowdown scale",
                "target reference and lead limit",
                "geometry, side-center height, precontact, and through distance",
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
        },
        "not_selected": {
            "action_scale": "NOT_SELECTED because changing it changes both command normalization and maximum raw delta",
            "joint_delta_reference": "RESERVE because cap/action saturation is primary while lead-limit is secondary in current evidence",
            "joint_target_lead_limit_rad": "RESERVE because lead limit is observed but not primary over the active cap",
            "goal_push_or_contact_band": "NOT_SELECTED because that changes geometry rather than the action bottleneck",
        },
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
        "line1 artifact=cube10cm_tap_rl_cap_only_candidate_design_v1 "
        "local_design_static_audit_only=YES gpu_runtime=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 basis "
            f"code_ready={code_ready} basis_ok={basis_ok} "
            f"action_abs_max_trace={audit.get('action_abs_max_trace')} "
            f"joint_delta_abs_max_trace={audit.get('joint_delta_abs_max_trace')} "
            f"joint_delta_cap_rate_trace={audit.get('joint_delta_cap_rate_trace')} "
            f"cap_is_primary_current_hypothesis={audit.get('cap_is_primary_current_hypothesis')}"
        ),
        (
            "line3 selected_candidate "
            "status=DESIGNED_NOT_RUN changed_knobs=1 "
            "max_joint_delta_per_step_rad=0.010->0.040 "
            "action_scale=0.04_fixed action_smoothing_alpha=0.25_fixed "
            "controller_mode=external_closed_loop_fixed geometry=FIXED"
        ),
        (
            "line4 not_selected "
            "action_scale=changes_command_normalization "
            "joint_delta_reference=reserve "
            "joint_target_lead_limit=reserve "
            "goal_push_or_contact_band=geometry_change"
        ),
        (
            "line5 pass_gate "
            "contact_seen>0 reaction_context>0 reaction_seen>0 tap_success>0 "
            "overshoot=0 final_1cm_required=False"
        ),
        (
            "line6 next_runtime_command "
            f"{command}"
        ),
        (
            "line7 verdict "
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
