#!/usr/bin/env python3
"""Root-cause audit for the cube10cm applied-target/TCP reach failure.

Local/posthoc only. This formalizes why the command target crosses contact while
the applied joint-target FK and actual TCP do not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_reach_contract_root_cause_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_reach_contract_root_cause_audit_summary.out"

SANITY_X240 = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_sanity.json"
DIAGNOSIS = LOG_DIR / "cube10cm_tap_rl_applied_target_tcp_reach_contract_diagnosis.json"
HARNESS = ROOT / "roarm_rl/test_positive_control_cube_tap10cm.py"
ENV = ROOT / "roarm_rl/roarm_cube_push_env.py"
STACK_ENV = ROOT / "roarm_rl/roarm_stack_env.py"
ISAAC_DIFFIK = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/"
    "source/isaaclab/isaaclab/controllers/differential_ik.py"
)
ISAAC_ACTUATOR_CFG = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/"
    "source/isaaclab/isaaclab/actuators/actuator_cfg.py"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line(path: Path, needle: str) -> int:
    for idx, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in text:
            return idx
    raise ValueError(f"needle not found: {path} {needle!r}")


def _trace_stat(data: dict[str, Any], key: str, stat: str) -> float:
    return float(data["controller_trace_stats"][key][stat])


def main() -> int:
    sanity = _load(SANITY_X240)
    diagnosis = _load(DIAGNOSIS)
    diag_x240 = diagnosis["x240"]
    diag_line = _load(DIAGNOSIS)

    raw_delta_max = _trace_stat(sanity, "builtin_diffik_raw_delta_abs_max_rad", "max")
    clipped_delta_max = _trace_stat(sanity, "builtin_diffik_clipped_delta_abs_max_rad", "max")
    step_clip_rate_max = _trace_stat(sanity, "builtin_diffik_step_clip_rate", "max")
    target_tcp_err_final_m = _trace_stat(sanity, "builtin_diffik_target_tcp_err_before_m_mean", "final")
    target_fk_err_final_mm = _trace_stat(sanity, "closed_loop_target_fk_err_mm_mean", "final")
    target_delta_max = _trace_stat(sanity, "closed_loop_target_delta_from_actual_abs_max_rad_max", "max")
    follow_max = _trace_stat(sanity, "direct_joint_follow_abs_max_rad", "max")
    actual_step_max = _trace_stat(sanity, "direct_actual_joint_step_abs_max_rad", "max")
    actual_to_target_step_ratio = actual_step_max / target_delta_max if target_delta_max > 0 else float("nan")
    control_dt_s = float(2 * (1.0 / 200.0))
    actual_joint_speed_rad_s = actual_step_max / control_dt_s

    first = diag_x240["snapshots"]["first_command_inside"]
    window = diag_x240["windows"]["command_inside_band"]

    lines = {
        "harness_joint_pos_arm": _line(HARNESS, "joint_pos_arm = inner._robot.data.joint_pos"),
        "harness_diffik_compute": _line(HARNESS, "joint_pos_des = diffik.compute"),
        "harness_raw_delta": _line(HARNESS, "raw_delta_arm = joint_pos_des - joint_pos_arm"),
        "harness_step_clip": _line(HARNESS, "clipped_delta_arm = torch_mod.clamp"),
        "harness_arm_joint_target": _line(HARNESS, "arm_joint_target = joint_pos_arm + clipped_delta_arm"),
        "harness_target_full_from_actual": _line(HARNESS, "target_full = inner._robot.data.joint_pos.detach().clone()"),
        "harness_target_full_assign": _line(HARNESS, "target_full[:, arm_joint_ids] = arm_joint_target"),
        "env_override": _line(ENV, "override_targets = getattr"),
        "env_robot_target_write": _line(ENV, "self.robot_dof_targets[:] = targets"),
        "env_set_joint_position_target": _line(ENV, "self._robot.set_joint_position_target(self.robot_dof_targets)"),
        "stack_decimation": _line(STACK_ENV, "decimation = 2"),
        "stack_sim_dt": _line(STACK_ENV, "dt=1 / 200"),
        "stack_arm_stiffness": _line(STACK_ENV, "stiffness=80.0"),
        "stack_arm_damping": _line(STACK_ENV, "damping=4.0"),
        "stack_arm_effort": _line(STACK_ENV, "effort_limit_sim=2.5"),
        "stack_arm_velocity": _line(STACK_ENV, "velocity_limit_sim=3.14"),
        "isaac_diffik_return_current_plus_delta": _line(ISAAC_DIFFIK, "return joint_pos + delta_joint_pos"),
        "isaac_actuator_stiffness": _line(ISAAC_ACTUATOR_CFG, "Stiffness gains"),
        "isaac_implicit_pd": _line(ISAAC_ACTUATOR_CFG, "The PD control is handled implicitly by the simulation."),
    }

    root_cause = {
        "primary_target_generation_cause": {
            "status": "CONFIRMED",
            "reason": (
                "DifferentialIK asks for a large current-pose-to-command correction, but the harness clips each "
                "joint correction to 0.010rad and builds target_full from current actual joint_pos. The FK of this "
                "one-step clipped target never reaches the face band."
            ),
            "raw_delta_abs_max_rad": raw_delta_max,
            "clipped_delta_abs_max_rad": clipped_delta_max,
            "step_clip_rate_max": step_clip_rate_max,
            "target_full_delta_from_actual_abs_max_rad_max": target_delta_max,
            "target_fk_err_final_mm": target_fk_err_final_mm,
            "target_tcp_err_before_final_m": target_tcp_err_final_m,
        },
        "secondary_actuator_follow_cause": {
            "status": "CONFIRMED_AS_LAG_NOT_PARAMETER-SPECIFIC",
            "reason": (
                "The actual joint step is far smaller than the target lead. This is consistent with position-drive "
                "actuator dynamics, but torque telemetry is not present, so stiffness/damping/effort saturation cannot "
                "be split definitively yet."
            ),
            "direct_joint_follow_abs_max_rad": follow_max,
            "direct_actual_joint_step_abs_max_rad": actual_step_max,
            "actual_to_target_step_ratio": actual_to_target_step_ratio,
            "control_dt_s": control_dt_s,
            "actual_joint_speed_rad_s": actual_joint_speed_rad_s,
        },
        "not_primary_causes": {
            "command_target_geometry": "NOT_PRIMARY_CAUSE_COMMAND_TARGET_ENTERS_BAND",
            "fixed_pose_x285": "NOT_SUPPORTED_FOR_PLUSX",
            "contact_gate": "NOT_PRIMARY_CAUSE_FOR_THIS_FAILURE",
            "cube_mass": "NOT_FIRST_CAUSE_CONTACT_NOT_REACHED_BEFORE_PUSH_FORCE_MATTERS",
        },
        "unresolved_without_extra_telemetry": [
            "which joint contributes most to applied FK shortfall in the step-clipped path",
            "whether effort_limit_sim saturates during the actual TCP lag",
            "whether changing controller schedule, clip, target base, or actuator gains is the cleanest fix",
        ],
    }

    result = {
        "artifact_type": "cube10cm_tap_rl_reach_contract_root_cause_audit_v1",
        "local_posthoc_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {
            "sanity_x240": str(SANITY_X240.relative_to(ROOT)),
            "diagnosis": str(DIAGNOSIS.relative_to(ROOT)),
            "harness": str(HARNESS.relative_to(ROOT)),
            "env": str(ENV.relative_to(ROOT)),
            "stack_env": str(STACK_ENV.relative_to(ROOT)),
            "isaac_diffik_source": str(ISAAC_DIFFIK),
            "isaac_actuator_cfg_source": str(ISAAC_ACTUATOR_CFG),
        },
        "line_refs": lines,
        "diagnosis_values": {
            "command_inside_rows": int(diag_x240["command_inside_rows"]),
            "command_inside_unique_steps": int(diag_x240["command_inside_unique_steps"]),
            "applied_inside_rows": int(diag_x240["applied_inside_rows"]),
            "actual_inside_rows": int(diag_x240["actual_inside_rows"]),
            "first_command_minus_applied_m": first["command_minus_applied_m"],
            "first_applied_minus_actual_m": first["applied_minus_actual_m"],
            "inside_window_command_minus_applied_m_mean": window["command_minus_applied_m_mean"],
            "inside_window_applied_minus_actual_m_mean": window["applied_minus_actual_m_mean"],
        },
        "root_cause": root_cause,
        "next_local_step": (
            "Design a default-off code audit/patch candidate that separates target-generation schedule/clip/base "
            "from actuator-follow dynamics before any new tiny runtime."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = [
        "line1 artifact=cube10cm_tap_rl_reach_contract_root_cause_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 primary_cause=STEP_CLIPPED_CURRENT_JOINT_BASED_TARGET_GENERATION "
            f"raw_delta_abs_max_rad={raw_delta_max:.9f} "
            f"clipped_delta_abs_max_rad={clipped_delta_max:.9f} "
            f"step_clip_rate_max={step_clip_rate_max:.9f} "
            f"target_delta_from_actual_max_rad={target_delta_max:.9f} "
            f"target_fk_err_final_mm={target_fk_err_final_mm:.9f} "
            f"target_tcp_err_before_final_m={target_tcp_err_final_m:.9f}"
        ),
        (
            "line3 code_basis "
            f"joint_pos_arm_line={lines['harness_joint_pos_arm']} "
            f"diffik_compute_line={lines['harness_diffik_compute']} "
            f"raw_delta_line={lines['harness_raw_delta']} "
            f"step_clip_line={lines['harness_step_clip']} "
            f"arm_joint_target_line={lines['harness_arm_joint_target']} "
            f"target_full_from_actual_line={lines['harness_target_full_from_actual']} "
            f"target_full_assign_line={lines['harness_target_full_assign']} "
            f"isaac_diffik_return_current_plus_delta_line={lines['isaac_diffik_return_current_plus_delta']}"
        ),
        (
            "line4 effect_on_contact "
            f"command_inside_rows={diag_x240['command_inside_rows']} "
            f"command_inside_unique_steps={diag_x240['command_inside_unique_steps']} "
            f"applied_inside_rows={diag_x240['applied_inside_rows']} "
            f"actual_inside_rows={diag_x240['actual_inside_rows']} "
            f"first_command_minus_applied_m={first['command_minus_applied_m']:.9f} "
            f"inside_window_command_minus_applied_m_mean={window['command_minus_applied_m_mean']:.9f}"
        ),
        (
            "line5 secondary_cause=POSITION_DRIVE_ACTUAL_TCP_LAG "
            f"direct_joint_follow_abs_max_rad={follow_max:.9f} "
            f"direct_actual_joint_step_abs_max_rad={actual_step_max:.9f} "
            f"actual_to_target_step_ratio={actual_to_target_step_ratio:.9f} "
            f"control_dt_s={control_dt_s:.9f} "
            f"actual_joint_speed_rad_s={actual_joint_speed_rad_s:.9f}"
        ),
        (
            "line6 actuator_basis "
            f"env_override_lines={lines['env_override']}-{lines['env_robot_target_write']} "
            f"set_joint_position_target_line={lines['env_set_joint_position_target']} "
            f"decimation_line={lines['stack_decimation']} sim_dt_line={lines['stack_sim_dt']} "
            f"stiffness_line={lines['stack_arm_stiffness']} damping_line={lines['stack_arm_damping']} "
            f"effort_limit_line={lines['stack_arm_effort']} velocity_limit_line={lines['stack_arm_velocity']} "
            f"isaac_implicit_pd_line={lines['isaac_implicit_pd']}"
        ),
        (
            "line7 not_primary_causes "
            "command_target_geometry=NO fixed_pose_x285=NO contact_gate=NO cube_mass=NO_FIRST_CAUSE "
            "reason=command_crosses_but_clipped_target_fk_and_actual_tcp_do_not"
        ),
        (
            "line8 unresolved "
            "exact_effort_stiffness_damping_split=NEEDS_TORQUE_OR_DRIVE_TELEMETRY "
            "worst_joint_contribution=NEEDS_PER_JOINT_TRACE "
            "next=LOCAL_DEFAULT_OFF_TARGET_GENERATION_CONTRACT_DESIGN_BEFORE_TINY_RUNTIME"
        ),
    ]
    OUT_SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
