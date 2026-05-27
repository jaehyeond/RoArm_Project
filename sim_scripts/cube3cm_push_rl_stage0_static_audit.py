#!/usr/bin/env python3
"""Static audit for the no-attach 3cm cube-push RL Stage 0 wiring.

This is a local text/code audit only. It does not import Isaac Lab, create an
environment, run PPO, or generate a dataset.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ENV = REPO / "roarm_rl/roarm_cube_push_env.py"
REG = REPO / "roarm_rl/__init__.py"
TRAIN = REPO / "roarm_rl/train_cube_push_ppo.py"
EVAL = REPO / "roarm_rl/eval_cube_push_policy.py"
FILTER_AUDIT = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/controlled_push_filter_audit.out"
OUT = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube_push_rl_stage0_static_audit.out"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _find(path: Path, needle: str) -> tuple[int, str]:
    for idx, line in enumerate(_read(path), start=1):
        if needle in line:
            return idx, line.strip()
    return 0, ""


def _between(path: Path, start_needle: str, end_needle: str) -> str:
    lines = _read(path)
    start = None
    for idx, line in enumerate(lines):
        if start_needle in line:
            start = idx
            break
    if start is None:
        return ""
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if end_needle in lines[idx]:
            end = idx
            break
    return "\n".join(lines[start:end])


def _check(name: str, ok: bool, detail: str) -> str:
    return f"CHECK name={name} ok={'YES' if ok else 'NO'} detail=\"{detail}\""


def main() -> int:
    lines: list[str] = []
    lines.append(
        "STATIC_AUDIT_INPUT "
        f"env={ENV} env_md5={_md5(ENV)} "
        f"registration={REG} registration_md5={_md5(REG)} "
        f"train={TRAIN} train_md5={_md5(TRAIN)} "
        f"eval={EVAL} eval_md5={_md5(EVAL)}"
    )
    if FILTER_AUDIT.exists():
        lines.append(f"FILTER_AUDIT_SOURCE path={FILTER_AUDIT} md5={_md5(FILTER_AUDIT)}")

    env_text = "\n".join(_read(ENV))
    reg_text = "\n".join(_read(REG))
    train_text = "\n".join(_read(TRAIN))
    eval_text = "\n".join(_read(EVAL))
    apply_action_block = _between(ENV, "def _apply_action", "def _update_grasp_attach")
    attach_block = _between(ENV, "def _update_grasp_attach", "def _compute_intermediate_values")
    reset_block = _between(ENV, "def _reset_idx", "def _push_terms")

    checks = [
        (
            "separate_env_class",
            "class RoArmCubePushEnv" in env_text and "class RoArmCubePushEnvCfg" in env_text,
            "RoArmCubePushEnv/RoArmCubePushEnvCfg exist",
        ),
        (
            "cube_3cm_config",
            "CUBE_SIZE_M = 0.030" in env_text and "mass=0.020" in env_text,
            "3cm cube and 20g mass configured",
        ),
        (
            "registered_env_id",
            "RoArm-CubePush-Direct-v0" in reg_text,
            "gym registration exists",
        ),
        (
            "train_entry_uses_new_env",
            "RoArm-CubePush-Direct-v0" in train_text and "train_cube_push_ppo" in str(TRAIN),
            "separate PPO entry point targets cube-push env",
        ),
        (
            "apply_action_robot_target_only",
            "set_joint_position_target" in apply_action_block and "_update_grasp_attach" not in apply_action_block,
            "_apply_action writes robot joint targets and does not call attach",
        ),
        (
            "attach_noop_override",
            "def _update_grasp_attach" in attach_block and "return" in attach_block,
            "_update_grasp_attach is overridden as no-op",
        ),
        (
            "rollout_posewrite_only_in_reset",
            "_sponge.write_root_pose_to_sim" in reset_block
            and "_sponge.write_root_pose_to_sim" not in apply_action_block,
            "object posewrite appears in reset block, not action block",
        ),
        (
            "grasp_marker_forced_false",
            "self._grasped[:] = False" in env_text and "cube_push_grasped_marker_rate" in env_text,
            "grasp marker forced false and logged",
        ),
        (
            "controlled_filter_thresholds_imported",
            "AUDIT_SPEED_P95_MPS = 1.302103193" in env_text
            and "AUDIT_TIP_P95_DEG = 141.181661216" in env_text
            and "AUDIT_DISP_XY_P99_M = 0.133549188" in env_text,
            "reward uses prior rollout controlled-push thresholds",
        ),
        (
            "ik_endpoint_curriculum_present",
            "ik_endpoint_reset" in env_text
            and "def _ik_precontact_joints" in env_text
            and "ik_dls" in env_text,
            "known endpoint can seed robot at IK pre-contact pose",
        ),
        (
            "low_motion_penalty_present",
            "low_motion_penalty_scale" in env_text and "reverse_push_penalty_scale" in env_text,
            "reward penalizes low-motion and reverse push cases",
        ),
        (
            "clean_success_gate_present",
            "cube_success_target_tol_m" in env_text
            and "terms[\"target_xy_dist\"] <= self.cfg.cube_success_target_tol_m" in env_text
            and "& ~terms[\"impact\"]" in env_text,
            "success requires aligned controlled non-impact push near target",
        ),
        (
            "impact_far_target_termination_present",
            "terminal_impact" in env_text
            and "cube_far_target_terminate_m" in env_text
            and "terminated = success_now | terms[\"terminal_impact\"]" in env_text,
            "impact/far-target episodes terminate instead of accumulating bad rollout",
        ),
        (
            "overshoot_lateral_penalties_present",
            "target_distance_penalty_scale" in env_text
            and "lateral_penalty_scale" in env_text
            and "overshoot_penalty_scale" in env_text,
            "reward penalizes target miss, lateral drift, and oversized pushes",
        ),
        (
            "speed_guard_curriculum_present",
            "action_scale: float = 0.04" in env_text
            and "speed_penalty_scale" in env_text
            and "cube_success_speed_max_mps" in env_text
            and "cube_push_speed_over_0p5_rate" in env_text,
            "speed guard lowers action scale and penalizes high-speed taps",
        ),
        (
            "action_smoothing_velocity_limit_present",
            "def _pre_physics_step" in env_text
            and "action_smoothing_alpha" in env_text
            and "max_joint_delta_per_step_rad" in env_text
            and "joint_target_lead_limit_rad" in env_text
            and "contact_joint_delta_scale" in env_text
            and "gripper_joint_idx] = 0.0" in env_text,
            "v4 action smoothing, per-step joint velocity limit, target lead limit, contact slowdown, and gripper-open hold exist",
        ),
        (
            "scripted_teacher_warmstart_present",
            "scripted_teacher_blend" in env_text
            and "def _ik_teacher_goal_joints" in env_text
            and "cube_push_teacher_blend_mean" in env_text
            and "cube_push_teacher_goal_ok_rate" in env_text,
            "v5 IK/scripted teacher warm-start exists and is logged",
        ),
        (
            "action_semantics_printed",
            "normalized_joint_delta" in train_text
            and "robot_dof_targets += action_scale" in train_text
            and "action_dim=6" in train_text,
            "training entry prints robot action semantics",
        ),
        (
            "training_cli_exposes_curriculum",
            "--ik_endpoint_reset" in train_text
            and "low_motion_penalty_scale" in train_text
            and "target_distance_penalty_scale" in train_text
            and "speed_penalty_scale" in train_text
            and "action_smoothing_alpha" in train_text
            and "max_joint_delta_per_step_rad" in train_text
            and "scripted_teacher_blend" in train_text,
            "training CLI exposes IK endpoint reset, clean reward, speed guard, action smoothing, and teacher warm-start overrides",
        ),
        (
            "eval_cli_exposes_motion_curriculum",
            "--ik_endpoint_reset" in eval_text
            and "--action_scale" in eval_text
            and "--max_joint_delta_per_step_rad" in eval_text
            and "--contact_joint_delta_scale" in eval_text
            and "--fast_cube_joint_delta_scale" in eval_text
            and "--speed_penalty_start_mps" in eval_text
            and "--scripted_teacher_blend" in eval_text,
            "eval CLI can replay frozen checkpoints with the same motion-limit and teacher-off/on parameters",
        ),
        (
            "local_backup_usd_default",
            "DEFAULT_LOCAL_USD" in train_text
            and "b200_backup_20260522_final/tmp_p7" in train_text
            and "env_cfg.robot.spawn.usd_path = args.robot_usd_path" in train_text,
            "training entry defaults to preserved local backup USD, not B200 path",
        ),
        (
            "training_not_dataset",
            "dataset_generation=NO" in train_text,
            "training script does not claim dataset generation",
        ),
    ]

    for name, ok, detail in checks:
        lines.append(_check(name, ok, detail))

    key_needles = [
        (ENV, "class RoArmCubePushEnvCfg"),
        (ENV, "def _ik_precontact_joints"),
        (ENV, "def _ik_teacher_goal_joints"),
        (ENV, "def _pre_physics_step"),
        (ENV, "def _apply_action"),
        (ENV, "def _update_grasp_attach"),
        (ENV, "def _get_rewards"),
        (ENV, "def _get_dones"),
        (REG, "RoArm-CubePush-Direct-v0"),
        (TRAIN, "scope=no_attach_cube_push"),
        (EVAL, "scope=no_attach_cube_push_eval"),
    ]
    for path, needle in key_needles:
        line_no, line = _find(path, needle)
        lines.append(f"LINE_REF file={path.relative_to(REPO)} line={line_no} text=\"{line}\"")

    pass_all = all(ok for _, ok, _ in checks)
    lines.append(
        "NEXT_COMMAND_SHAPE "
        "cmd='python -m roarm_rl.train_cube_push_ppo --num_envs 512 --max_iterations 50 "
        "--ik_endpoint_reset --action_scale 0.025 --max_joint_delta_per_step_rad 0.004 "
        "--contact_joint_delta_scale 0.15 --fast_cube_joint_delta_scale 0.05 "
        "--joint_target_lead_limit_rad 0.030 --ik_precontact_clearance_m 0.020 "
        "--speed_penalty_start_mps 0.200 --speed_penalty_scale 20.0 "
        "--impact_penalty_scale 25.0 --impact_terminal_penalty 60.0 "
        "--experiment_name cube_push_contact_speed_50iter_20260526 "
        "--logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_contact_speed_logs "
        "--robot_usd_path b200_backup_20260522_final/tmp_p7/"
        "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd'"
    )
    lines.append(
        "RESULT cube_push_rl_stage0_static_audit="
        f"{'PASS' if pass_all else 'FAIL'} local_static_only=YES isaac_runtime=NO training_run=NO dataset_generation=NO"
    )

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
