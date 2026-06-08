"""Design a direct-IK-apply diagnostic after cap040.

This local audit compares the current RL-action positive-control path with the
installed Isaac Lab task-space action pattern. It does not launch Isaac Lab,
run GPU physics, build datasets, train, control a robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
CAP040_AUDIT = LOG_DIR / "cube10cm_tap_rl_cap040_final_result_audit.json"
CAP040_SUMMARY = LOG_DIR / "cube10cm_tap_rl_cap040_final_result_audit_summary.out"
HARNESS = REPO / "roarm_rl/test_positive_control_cube_tap10cm.py"
ENV_SOURCE = REPO / "roarm_rl/roarm_cube_push_env.py"
ISAAC_TASK_SPACE = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/"
    "isaaclab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py"
)
ISAAC_DIFFIK_TEST = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/"
    "isaaclab/source/isaaclab/test/controllers/test_differential_ik.py"
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_candidate_design.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_candidate_design_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_of(path: Path, pattern: str) -> int | None:
    if not path.exists():
        return None
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap040_audit", type=Path, default=CAP040_AUDIT)
    parser.add_argument("--cap040_summary", type=Path, default=CAP040_SUMMARY)
    parser.add_argument("--harness", type=Path, default=HARNESS)
    parser.add_argument("--env_source", type=Path, default=ENV_SOURCE)
    parser.add_argument("--isaac_task_space", type=Path, default=ISAAC_TASK_SPACE)
    parser.add_argument("--isaac_diffik_test", type=Path, default=ISAAC_DIFFIK_TEST)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    cap040 = _load(args.cap040_audit)
    code_lines = {
        "harness_ik_target_compute": _line_of(args.harness, "tcp_target = pre + alpha * (through - pre)"),
        "harness_action_rewrap": _line_of(args.harness, "action_t = (target_t - target_base)"),
        "harness_direct_mode_choice": _line_of(args.harness, "external_closed_loop_direct_apply"),
        "harness_direct_override_write": _line_of(args.harness, "inner._external_joint_targets_override = joint_target"),
        "env_direct_override_read": _line_of(args.env_source, "_external_joint_targets_override"),
        "env_action_scale": _line_of(args.env_source, "raw_delta = self.cfg.action_scale * self._smoothed_actions"),
        "env_target_lead_clamp": _line_of(args.env_source, "targets = torch.maximum(torch.minimum(targets, joint_pos + lead)"),
        "env_set_joint_position_target": _line_of(args.env_source, "self._robot.set_joint_position_target(self.robot_dof_targets)"),
        "isaac_task_space_compute": _line_of(args.isaac_task_space, "joint_pos_des = self._ik_controller.compute"),
        "isaac_task_space_direct_apply": _line_of(args.isaac_task_space, "self._asset.set_joint_position_target(joint_pos_des"),
        "isaac_test_direct_apply": _line_of(args.isaac_diffik_test, "robot.set_joint_position_target(joint_pos_des, arm_joint_ids)"),
    }
    code_ready = all(value is not None for value in code_lines.values())
    basis_ok = (
        bool(cap040.get("cap_only_falsified_as_primary"))
        and bool(cap040.get("lead_limit_observed"))
        and cap040.get("contact_seen") == 0.0
        and cap040.get("tap_success") == 0.0
    )
    isaac_pattern_supported = (
        code_lines["isaac_task_space_compute"] is not None
        and code_lines["isaac_task_space_direct_apply"] is not None
        and code_lines["isaac_test_direct_apply"] is not None
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_direct_ik_apply_candidate_design_v1",
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
        "isaac_pattern_supported": isaac_pattern_supported,
        "basis": {
            "cap_only_falsified_as_primary": cap040.get("cap_only_falsified_as_primary"),
            "lead_limit_observed": cap040.get("lead_limit_observed"),
            "contact_seen": cap040.get("contact_seen"),
            "tap_success": cap040.get("tap_success"),
            "best_shortfall_m": cap040.get("cap040_best_shortfall_m"),
            "target_lead_limit_rate_trace": cap040.get("target_lead_limit_rate_trace"),
        },
        "diagnosis": {
            "current_path": "IK target -> normalized RL action -> smoothing/action_scale/cap/lead-limit -> robot_dof_targets",
            "isaac_pattern": "task-space action controller compute -> joint_pos_des -> set_joint_position_target",
            "why_next": (
                "separates target_geometry_or_ik_failure from RL_action_target_application_failure "
                "before more lead/cap sweeps or any action dataset"
            ),
        },
        "selected_candidate": {
            "status": "DESIGNED_NOT_RUN",
            "name": "direct_ik_apply_positive_control",
            "changed_semantics": "bypass_RL_action_rewrap_and_apply_IK_joint_target_directly",
            "implementation_status": "HARNESS_AND_ENV_DEFAULT_OFF_MODE_READY",
            "num_envs": 2,
            "max_steps": 120,
            "seed": 962,
            "device": "cuda:0",
            "controller": "external_closed_loop_ik",
            "geometry": "fixed_from_cap040",
            "pass_gate": "contact_seen>0 reaction_context>0 reaction_seen>0 tap_success>0 overshoot=0",
        },
        "not_selected_next": {
            "cap040_lead120": "reserve_after_direct_ik_apply; otherwise it may tune the wrapper before proving the IK target works",
            "joint_delta_reference": "reserve; changes action-base semantics",
            "action_scale": "changes command normalization",
            "geometry_goal_push_or_contact_band": "changes contact geometry",
            "dataset_or_rl": "blocked until contact-gated positive-control passes",
        },
        "ready_for_explicit_runtime_approval_only": code_ready and basis_ok and isaac_pattern_supported,
        "still_blocked": {
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "code_lines": code_lines,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_direct_ik_apply_candidate_design_v1 "
        "local_design_static_audit_only=YES gpu_runtime=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 basis "
            f"code_ready={code_ready} basis_ok={basis_ok} "
            f"isaac_pattern_supported={isaac_pattern_supported} "
            f"cap_only_falsified_as_primary={cap040.get('cap_only_falsified_as_primary')} "
            f"lead_limit_observed={cap040.get('lead_limit_observed')} "
            f"contact_seen={cap040.get('contact_seen')} tap_success={cap040.get('tap_success')}"
        ),
        (
            "line3 path_comparison "
            "current=IK_target_to_normalized_RL_action_to_smoothing_scale_cap_lead_limit_to_robot_dof_targets "
            "isaac_pattern=IK_controller_compute_to_joint_pos_des_to_set_joint_position_target"
        ),
        (
            "line4 selected_candidate "
            "status=DESIGNED_NOT_RUN name=direct_ik_apply_positive_control "
            "implementation_status=HARNESS_AND_ENV_DEFAULT_OFF_MODE_READY "
            "purpose=separate_target_geometry_or_ik_from_RL_action_target_application "
            "num_envs=2 max_steps=120 seed=962 device=cuda:0 geometry=fixed_from_cap040"
        ),
        (
            "line5 not_selected "
            "cap040_lead120=reserve_after_direct_ik_apply "
            "joint_delta_reference=reserve_changes_action_base_semantics "
            "action_scale=changes_command_normalization "
            "goal_push_or_contact_band=geometry_change dataset_or_rl=blocked"
        ),
        (
            "line6 verdict "
            f"ready_for_explicit_runtime_approval_only={result['ready_for_explicit_runtime_approval_only']} "
            "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if result["ready_for_explicit_runtime_approval_only"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
