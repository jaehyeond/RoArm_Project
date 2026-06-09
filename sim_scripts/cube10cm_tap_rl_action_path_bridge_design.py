#!/usr/bin/env python3
"""Static audit for bridging Candidate6 DiffIK into the RL action path.

No Isaac runtime is launched here.  The script records why the current PPO
smoke is only a plumbing check and why the next implementation should be a
default-off Candidate6 residual/DiffIK action mode rather than more raw
joint-delta PPO.  After the patch, it also records the local implementation
lines for that default-off action path.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
SUMMARY = LOG_DIR / "cube10cm_tap_rl_action_path_bridge_design_summary.out"
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_action_path_bridge_design.json"


def _line(path: Path, needle: str) -> int | None:
    for idx, text in enumerate(path.read_text().splitlines(), start=1):
        if needle in text:
            return idx
    return None


def _line_after(path: Path, start_needle: str, needle: str) -> int | None:
    started = False
    for idx, text in enumerate(path.read_text().splitlines(), start=1):
        if start_needle in text:
            started = True
        if started and needle in text:
            return idx
    return None


def _contains(path: Path, needle: str) -> bool:
    return needle in path.read_text()


def main() -> int:
    env_py = REPO / "roarm_rl/roarm_cube_push_env.py"
    pc_py = REPO / "roarm_rl/test_positive_control_cube_tap10cm.py"
    smoke_py = REPO / "roarm_rl/train_cube_tap10cm_ppo_smoke.py"
    promotion_summary = LOG_DIR / "cube10cm_tap_rl_candidate6_promotion_validation_audit_summary.out"
    posthoc_summary = LOG_DIR / "cube10cm_tap_rl_candidate6_pilot_ppo_smoke_posthoc_summary.out"

    evidence = {
        "local_static_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "positive_control": {
            "diffik_init_line": _line(pc_py, "DifferentialIKControllerCfg("),
            "target_path_nearface_line": _line_after(
                pc_py,
                "def _closed_loop_builtin_diffik_joint_target",
                "float(args.goal_push_m) - half_along",
            ),
            "target_base_mode_line": _line(pc_py, "target_base_mode = str(getattr(args, \"builtin_diffik_target_base_mode\""),
            "step_clip_line": _line(pc_py, "clipped_delta_arm = torch_mod.clamp"),
            "direct_apply_line": _line(pc_py, "inner._external_joint_targets_override = joint_target"),
            "promotion_contract_line": 2,
            "promotion_stage0b_line": 7,
        },
        "rl_env_joint_delta_path": {
            "doc_line": _line(env_py, "robot_dof_targets += action_scale * action"),
            "policy_action_clamp_line": _line(env_py, "policy_actions = actions.clone().clamp"),
            "raw_delta_line": _line(env_py, "raw_delta = self.cfg.action_scale * self._smoothed_actions"),
            "target_base_line": _line(env_py, "target_base = self.robot_dof_targets"),
        },
        "existing_teacher_not_candidate6": {
            "scripted_teacher_blend_cfg_line": _line(env_py, "scripted_teacher_blend: float = 0.0"),
            "dls_goal_line": _line(env_py, "tcp_target[:2] += dir_np[idx] * (half_along + float(self.cfg.scripted_teacher_goal_push_m))"),
            "positive_control_built_in_diffik_in_env": _contains(env_py, "DifferentialIKControllerCfg("),
        },
        "bridge_patch": {
            "default_mode_cfg_line": _line(env_py, "rl_action_mode: str = \"joint_delta\""),
            "candidate_mode_cfg_line": _line(env_py, "candidate6_diffik_residual_scale_rad: float = 0.002"),
            "diffik_controller_line": _line(env_py, "def _ensure_candidate6_diffik_controller"),
            "env_nearface_line": _line(env_py, "float(self.cfg.candidate6_diffik_goal_push_m) - half_along"),
            "candidate_branch_line": _line(env_py, "if action_mode == \"candidate6_diffik_residual_joint\""),
            "residual_scale_line": _line(env_py, "policy_actions[:, arm_joint_ids] * float(self.cfg.candidate6_diffik_residual_scale_rad)"),
            "telemetry_log_line": _line(env_py, "\"cube_push_candidate6_diffik_active_rate\""),
            "smoke_arg_line": _line(smoke_py, "\"--rl_action_mode\""),
            "smoke_contract_line": _line(smoke_py, "cfg.rl_action_mode = str(args.rl_action_mode)"),
        },
        "d208_posthoc": {
            "summary_line": 6,
            "policy_task_pass_line": 7,
        },
        "decision": {
            "raw_joint_delta_more_iterations_next": False,
            "existing_scripted_teacher_sufficient": False,
            "next_patch": "tiny_no_training_candidate6_diffik_residual_preflight",
            "default_action_mode_must_remain": "joint_delta",
            "candidate_action_mode": "candidate6_diffik_residual_joint",
            "base_controller": "Candidate6 near-face built-in DiffIK previous-target-base step-clipped joint target",
            "policy_output": "small residual around Candidate6 base target",
            "runtime_after_patch": "tiny no-training preflight first, then tiny PPO smoke only if preflight passes",
        },
    }

    lines = [
        "line1 artifact=cube10cm_tap_rl_action_path_bridge_design_v2 "
        "local_static_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 candidate6_pass_path "
        f"diffik_init_line={evidence['positive_control']['diffik_init_line']} "
        f"nearface_line={evidence['positive_control']['target_path_nearface_line']} "
        f"target_base_line={evidence['positive_control']['target_base_mode_line']} "
        f"step_clip_line={evidence['positive_control']['step_clip_line']} "
        f"direct_apply_line={evidence['positive_control']['direct_apply_line']} "
        "promotion_contract_summary_line=2 promotion_stage0b_summary_line=7",
        "line3 current_ppo_action_path "
        f"doc_line={evidence['rl_env_joint_delta_path']['doc_line']} "
        f"policy_action_clamp_line={evidence['rl_env_joint_delta_path']['policy_action_clamp_line']} "
        f"raw_delta_line={evidence['rl_env_joint_delta_path']['raw_delta_line']} "
        f"target_base_line={evidence['rl_env_joint_delta_path']['target_base_line']} "
        "action_semantics=raw_joint_delta",
        "line4 existing_teacher_gap "
        f"scripted_teacher_blend_cfg_line={evidence['existing_teacher_not_candidate6']['scripted_teacher_blend_cfg_line']} "
        f"dls_far_face_goal_line={evidence['existing_teacher_not_candidate6']['dls_goal_line']} "
        f"env_contains_builtin_diffik={evidence['existing_teacher_not_candidate6']['positive_control_built_in_diffik_in_env']} "
        "existing_teacher_sufficient=NO",
        "line5 d208_caveat "
        "corrected_posthoc_policy_task_pass_line=7 "
        "posthoc_metrics_line=6 "
        "interpretation=tiny_fixed_contract_policy_candidate_not_action_path_transfer",
        "line6 decision "
        "raw_joint_delta_more_iterations_next=NO "
        "next_patch=tiny_no_training_candidate6_diffik_residual_preflight "
        "default_mode_preserve=joint_delta "
        "candidate_mode=candidate6_diffik_residual_joint",
        "line7 bridge_patch_contract "
        "base=Candidate6_nearface_previous_target_base_step_clipped_built_in_DiffIK "
        "policy_output=small_joint_residual "
        "preflight_before_training=YES "
        "large_dataset_rl_roarm_unblocked=NO",
        "line8 bridge_patch_lines "
        f"default_mode_cfg_line={evidence['bridge_patch']['default_mode_cfg_line']} "
        f"candidate_branch_line={evidence['bridge_patch']['candidate_branch_line']} "
        f"env_nearface_line={evidence['bridge_patch']['env_nearface_line']} "
        f"residual_scale_line={evidence['bridge_patch']['residual_scale_line']} "
        f"telemetry_log_line={evidence['bridge_patch']['telemetry_log_line']} "
        f"smoke_arg_line={evidence['bridge_patch']['smoke_arg_line']} "
        f"smoke_contract_line={evidence['bridge_patch']['smoke_contract_line']}",
    ]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
