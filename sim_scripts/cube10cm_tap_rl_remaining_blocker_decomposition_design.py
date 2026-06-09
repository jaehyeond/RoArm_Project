#!/usr/bin/env python3
"""Design audit for the remaining cube10cm near-face contact blocker.

Local-only, no Isaac runtime. This decomposes the remaining failure after
near-face target-path correction into:

1. target-base accumulation / applied target generation
2. precontact reset / initial offset
3. actuator / drive follow telemetry
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

NEAR_SANITY = (
    LOG_DIR
    / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_reachtrace_rerun1_sanity.json"
)
NEAR_TRACE = (
    LOG_DIR
    / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_reachtrace_rerun1_trace.json"
)
NEAR_RESULT_AUDIT = LOG_DIR / "cube10cm_tap_rl_nearface_target_path_result_audit.json"
HARNESS = ROOT / "roarm_rl/test_positive_control_cube_tap10cm.py"
ENV = ROOT / "roarm_rl/roarm_cube_push_env.py"
STACK_ENV = ROOT / "roarm_rl/roarm_stack_env.py"
ISAAC_ARTICULATION_DATA = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/"
    "source/isaaclab/isaaclab/assets/articulation/articulation_data.py"
)

OUT_JSON = LOG_DIR / "cube10cm_tap_rl_remaining_blocker_decomposition_design.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_remaining_blocker_decomposition_design_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line(path: Path, needle: str) -> int:
    for idx, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in text:
            return idx
    raise ValueError(f"needle not found: {path} {needle!r}")


def _avg_step(rows: list[dict[str, Any]], step: int, key: str) -> float:
    vals = [float(row[key]) for row in rows if int(row["step"]) == step]
    if not vals:
        raise ValueError(f"missing step={step} key={key}")
    return mean(vals)


def _best(rows: list[dict[str, Any]], key: str) -> tuple[int, float]:
    by_step: dict[int, list[float]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(float(row[key]))
    return max(((step, mean(vals)) for step, vals in by_step.items()), key=lambda item: item[1])


def _inside_window_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    inside = [row for row in rows if bool(row["command_target_inside_contact_band"])]
    if not inside:
        raise ValueError("command target never enters contact band")
    command_minus_applied = [
        float(row["command_target_face_gap_m"]) - float(row["applied_joint_target_fk_face_gap_m"])
        for row in inside
    ]
    applied_minus_actual = [
        float(row["applied_joint_target_fk_face_gap_m"]) - float(row["actual_tcp_face_gap_m"])
        for row in inside
    ]
    applied_shortfall = [max(0.0, -0.010 - float(row["applied_joint_target_fk_face_gap_m"])) for row in inside]
    actual_shortfall = [max(0.0, -0.010 - float(row["actual_tcp_face_gap_m"])) for row in inside]
    return {
        "command_inside_rows": float(len(inside)),
        "command_inside_first_step": float(min(int(row["step"]) for row in inside)),
        "command_inside_last_step": float(max(int(row["step"]) for row in inside)),
        "command_minus_applied_mean_m": mean(command_minus_applied),
        "applied_minus_actual_mean_m": mean(applied_minus_actual),
        "applied_shortfall_mean_m": mean(applied_shortfall),
        "actual_shortfall_mean_m": mean(actual_shortfall),
        "actual_extra_shortfall_over_applied_mean_m": mean(actual_shortfall) - mean(applied_shortfall),
    }


def _trace_stat(sanity: dict[str, Any], key: str, stat: str) -> float:
    return float(sanity["controller_trace_stats"][key][stat])


def main() -> int:
    sanity = _load(NEAR_SANITY)
    trace = _load(NEAR_TRACE)
    result_audit = _load(NEAR_RESULT_AUDIT)
    rows = trace["rows"]
    face_band_m = 0.010

    command_inside_rows = sum(1 for row in rows if bool(row["command_target_inside_contact_band"]))
    command_inside_steps = sorted({int(row["step"]) for row in rows if bool(row["command_target_inside_contact_band"])})
    applied_inside_rows = sum(1 for row in rows if bool(row["applied_joint_target_fk_inside_contact_band"]))
    actual_inside_rows = sum(1 for row in rows if bool(row["actual_contact_proxy"]))

    final_step = max(int(row["step"]) for row in rows)
    command_final = _avg_step(rows, final_step, "command_target_face_gap_m")
    applied_final = _avg_step(rows, final_step, "applied_joint_target_fk_face_gap_m")
    actual_final = _avg_step(rows, final_step, "actual_tcp_face_gap_m")
    applied_best_step, applied_best = _best(rows, "applied_joint_target_fk_face_gap_m")
    actual_best_step, actual_best = _best(rows, "actual_tcp_face_gap_m")
    applied_best_shortfall = max(0.0, -face_band_m - applied_best)
    actual_best_shortfall = max(0.0, -face_band_m - actual_best)
    actual_extra_shortfall = actual_best_shortfall - applied_best_shortfall
    window = _inside_window_stats(rows)

    reset = sanity["reset_metrics"]
    initial_command_gap = _avg_step(rows, 0, "command_target_face_gap_m")
    initial_actual_gap = _avg_step(rows, 0, "actual_tcp_face_gap_m")
    initial_applied_gap = _avg_step(rows, 0, "applied_joint_target_fk_face_gap_m")
    reset_actual_minus_command_m = initial_actual_gap - initial_command_gap
    reset_applied_minus_command_m = initial_applied_gap - initial_command_gap

    target_delta_final = _trace_stat(sanity, "closed_loop_target_delta_from_actual_abs_max_rad_max", "final")
    raw_delta_final = _trace_stat(sanity, "builtin_diffik_raw_delta_abs_max_rad", "final")
    clipped_delta_final = _trace_stat(sanity, "builtin_diffik_clipped_delta_abs_max_rad", "final")
    follow_max = _trace_stat(sanity, "direct_joint_follow_abs_max_rad", "max")
    actual_step_max = _trace_stat(sanity, "direct_actual_joint_step_abs_max_rad", "max")
    actual_to_follow_step_ratio = actual_step_max / follow_max if follow_max > 0 else float("nan")
    actual_to_target_step_ratio = actual_step_max / target_delta_final if target_delta_final > 0 else float("nan")

    lines = {
        "harness_command_final_near_face_branch": _line(HARNESS, 'if args.target_path_mode == "near_face_goal"'),
        "harness_joint_pos_arm": _line(HARNESS, "joint_pos_arm = inner._robot.data.joint_pos"),
        "harness_raw_delta": _line(HARNESS, "raw_delta_arm = joint_pos_des - joint_pos_arm"),
        "harness_step_clip": _line(HARNESS, "clipped_delta_arm = torch_mod.clamp"),
        "harness_arm_target_actual_base": _line(HARNESS, "arm_joint_target = joint_pos_arm + clipped_delta_arm"),
        "harness_target_full_actual_base": _line(HARNESS, "target_full = inner._robot.data.joint_pos.detach().clone()"),
        "harness_applied_fk_trace": _line(HARNESS, "applied_tcp_local = fk_tcp"),
        "harness_after_step_follow": _line(HARNESS, "follow_abs = torch.abs(joint_target_for_step - joint_pos_after_step)"),
        "env_override_start": _line(ENV, "override_targets = getattr"),
        "env_set_joint_position_target": _line(ENV, "self._robot.set_joint_position_target(self.robot_dof_targets)"),
        "env_contact_proxy": _line(ENV, "contact_proxy = ("),
        "stack_decimation": _line(STACK_ENV, "decimation = 2"),
        "stack_dt": _line(STACK_ENV, "dt=1 / 200"),
        "stack_stiffness": _line(STACK_ENV, "stiffness=80.0"),
        "stack_damping": _line(STACK_ENV, "damping=4.0"),
        "stack_effort": _line(STACK_ENV, "effort_limit_sim=2.5"),
        "stack_velocity": _line(STACK_ENV, "velocity_limit_sim=3.14"),
        "isaac_joint_pos_target": _line(ISAAC_ARTICULATION_DATA, "joint_pos_target: torch.Tensor = None"),
        "isaac_computed_torque": _line(ISAAC_ARTICULATION_DATA, "computed_torque: torch.Tensor = None"),
        "isaac_applied_torque": _line(ISAAC_ARTICULATION_DATA, "applied_torque: torch.Tensor = None"),
        "isaac_joint_vel": _line(ISAAC_ARTICULATION_DATA, "def joint_vel(self):"),
    }

    factor_ranking = [
        {
            "rank": 1,
            "factor": "target_base_accumulation_or_applied_target_generation",
            "status": "PRIMARY_NEXT_DESIGN",
            "why": (
                "Command target is now near-face and inside-band for many rows, but applied joint-target FK "
                "still never enters the strict band. The code builds each target from current actual joint_pos "
                "plus clipped delta, so an actual-base non-accumulating contract remains the first remaining suspect."
            ),
            "evidence": {
                "command_inside_rows": command_inside_rows,
                "applied_inside_rows": applied_inside_rows,
                "applied_best_face_gap_m": applied_best,
                "applied_best_shortfall_m": applied_best_shortfall,
                "target_fk_err_final_mm": _trace_stat(sanity, "closed_loop_target_fk_err_mm_mean", "final"),
                "raw_delta_final_rad": raw_delta_final,
                "clipped_delta_final_rad": clipped_delta_final,
            },
        },
        {
            "rank": 2,
            "factor": "actuator_drive_follow",
            "status": "SECONDARY_BUT_REAL",
            "why": (
                "Actual TCP is behind the already-insufficient applied target. This cannot be the only cause "
                "because applied FK also misses the band, but it likely adds several millimeters of miss."
            ),
            "evidence": {
                "actual_inside_rows": actual_inside_rows,
                "actual_best_face_gap_m": actual_best,
                "actual_best_shortfall_m": actual_best_shortfall,
                "actual_extra_shortfall_over_applied_best_m": actual_extra_shortfall,
                "direct_follow_abs_max_rad": follow_max,
                "actual_joint_step_abs_max_rad": actual_step_max,
                "actual_to_follow_step_ratio": actual_to_follow_step_ratio,
                "actual_to_target_step_ratio": actual_to_target_step_ratio,
            },
        },
        {
            "rank": 3,
            "factor": "precontact_reset_or_initial_offset",
            "status": "LOWER_PRIORITY_NOT_CLEARED",
            "why": (
                "Initial actual TCP is close to the intended precontact command and reset IK error is about 1mm, "
                "so reset offset is not the primary explanation for final failure. It still needs a reset snapshot "
                "because a 1-2mm initial bias matters when the remaining strict-band shortfall is millimeters."
            ),
            "evidence": {
                "initial_command_face_gap_m": initial_command_gap,
                "initial_applied_face_gap_m": initial_applied_gap,
                "initial_actual_face_gap_m": initial_actual_gap,
                "reset_actual_minus_command_m": reset_actual_minus_command_m,
                "reset_applied_minus_command_m": reset_applied_minus_command_m,
                "reset_ik_err_mm": reset["ik_reset_err_mm"],
                "initial_face_gap_m_from_summary": reset["initial_face_gap_m"],
                "initial_vertical_offset_m": reset["initial_vertical_offset_m"],
            },
        },
    ]

    default_off_detail_trace_schema = {
        "argument": "--reach_trace_detail_json",
        "default": None,
        "behavior_when_absent": "NO_OUTPUT_NO_CONTROL_CHANGE",
        "artifact_flags": {
            "action_teacher_dataset": False,
            "dataset_generation": False,
            "training": False,
            "robot_control": False,
        },
        "metadata_fields": [
            "target_path_mode",
            "controller_mode",
            "target_base_mode",
            "step_clip_rad",
            "episode_length_s",
            "fixed_cube_pose",
            "reset_command_face_gap_m",
            "reset_actual_face_gap_m",
            "reset_applied_fk_face_gap_m",
            "reset_tcp_world_xyz",
            "reset_cube_world_xyz",
        ],
        "per_step_scalar_fields": [
            "command_face_gap_m",
            "applied_fk_face_gap_m",
            "actual_tcp_face_gap_m",
            "command_minus_applied_m",
            "applied_minus_actual_m",
            "tap_success_now",
            "terminated",
            "truncated",
        ],
        "per_step_joint_array_fields": [
            "joint_pos_before_arm_rad",
            "previous_joint_target_arm_rad",
            "raw_delta_arm_rad",
            "clipped_delta_arm_rad",
            "arm_joint_target_rad",
            "target_full_arm_rad",
            "joint_pos_after_arm_rad",
            "joint_vel_after_arm_rad",
            "joint_acc_after_arm_rad",
            "follow_after_arm_rad",
            "actual_step_arm_rad",
            "joint_pos_target_arm_rad",
            "computed_torque_arm",
            "applied_torque_arm",
            "joint_effort_limit_arm",
            "joint_vel_limit_arm",
        ],
        "counterfactual_checks": [
            "offline_previous_target_accumulation_delta_available",
            "worst_joint_by_follow",
            "worst_joint_by_clip",
            "worst_joint_by_effort_saturation",
            "whether_applied_fk_would_enter_band_under_previous_target_base",
        ],
    }

    next_decision_tree = [
        {
            "if": "applied_fk_inside_rows_remains_0_and_previous_target_counterfactual_enters_band",
            "then": "design one default-off previous-target-base tiny runtime candidate",
        },
        {
            "if": "applied_fk_enters_band_but_actual_tcp_does_not_or_effort_saturates",
            "then": "design actuator/drive follow candidate or gain/effort telemetry runtime",
        },
        {
            "if": "reset_actual_minus_command_bias_exceeds_2_to_3mm_and_persists",
            "then": "design reset/precontact recalibration candidate before controller tuning",
        },
        {
            "if": "none_of_the_above",
            "then": "do not proceed to dataset/RL; inspect FK/tool frame and joint limits with a visual overlay",
        },
    ]

    design = {
        "artifact_type": "cube10cm_tap_rl_remaining_blocker_decomposition_design_v1",
        "local_design_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {
            "nearface_sanity": str(NEAR_SANITY.relative_to(ROOT)),
            "nearface_trace": str(NEAR_TRACE.relative_to(ROOT)),
            "nearface_result_audit": str(NEAR_RESULT_AUDIT.relative_to(ROOT)),
            "harness": str(HARNESS.relative_to(ROOT)),
            "env": str(ENV.relative_to(ROOT)),
            "stack_env": str(STACK_ENV.relative_to(ROOT)),
            "isaac_articulation_data": str(ISAAC_ARTICULATION_DATA),
        },
        "line_refs": lines,
        "observed_nearface_state": {
            "status": sanity["status"],
            "rl_contact_gated_positive_control": sanity["rl_contact_gated_positive_control"],
            "professor_physical_reaction_evidence": sanity["professor_physical_reaction_evidence"],
            "steps_executed": sanity["steps_executed"],
            "truncated_count": sanity["truncated_count"],
            "terminated_count": sanity["terminated_count"],
            "command_final_face_gap_m": command_final,
            "command_inside_rows": command_inside_rows,
            "command_inside_first_step": command_inside_steps[0],
            "command_inside_last_step": command_inside_steps[-1],
            "applied_inside_rows": applied_inside_rows,
            "actual_inside_rows": actual_inside_rows,
            "applied_best_step": applied_best_step,
            "applied_best_face_gap_m": applied_best,
            "applied_best_shortfall_m": applied_best_shortfall,
            "actual_best_step": actual_best_step,
            "actual_best_face_gap_m": actual_best,
            "actual_best_shortfall_m": actual_best_shortfall,
            "actual_extra_shortfall_over_applied_best_m": actual_extra_shortfall,
            "command_inside_window": window,
            "nearface_result_audit_verdict": result_audit["verdict"],
        },
        "factor_ranking": factor_ranking,
        "default_off_detail_trace_schema": default_off_detail_trace_schema,
        "next_decision_tree": next_decision_tree,
        "verdict": "DESIGN_READY_DEFAULT_OFF_REACH_DETAIL_TRACE_BEFORE_ANY_GATE_RELAXATION_OR_DATASET_RL",
    }
    OUT_JSON.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = [
        "line1 artifact=cube10cm_tap_rl_remaining_blocker_decomposition_design_v1 "
        "local_design_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 current_nearface_state "
            f"status={sanity['status']} rl_contact_gated_positive_control={sanity['rl_contact_gated_positive_control']} "
            f"professor_physical_reaction_evidence={sanity['professor_physical_reaction_evidence']} "
            f"steps_executed={sanity['steps_executed']} truncated_count={sanity['truncated_count']} "
            f"command_final_face_gap_m={command_final:.9f} command_inside_steps={command_inside_steps[0]}..{command_inside_steps[-1]} "
            f"command_inside_rows={command_inside_rows} applied_inside_rows={applied_inside_rows} "
            f"actual_inside_rows={actual_inside_rows}"
        ),
        (
            "line3 factor_ranking "
            "rank1=TARGET_BASE_ACCUMULATION_OR_APPLIED_TARGET_GENERATION "
            "rank2=ACTUATOR_DRIVE_FOLLOW "
            "rank3=PRECONTACT_RESET_INITIAL_OFFSET "
            "contact_gate_relaxation=NOT_NEXT"
        ),
        (
            "line4 target_base_evidence "
            f"code_joint_pos_arm_line={lines['harness_joint_pos_arm']} "
            f"raw_delta_line={lines['harness_raw_delta']} "
            f"step_clip_line={lines['harness_step_clip']} "
            f"actual_base_target_line={lines['harness_arm_target_actual_base']} "
            f"target_full_actual_base_line={lines['harness_target_full_actual_base']} "
            f"applied_best_face_gap_m={applied_best:.9f} "
            f"applied_best_shortfall_m={applied_best_shortfall:.9f} "
            f"target_fk_err_final_mm={_trace_stat(sanity, 'closed_loop_target_fk_err_mm_mean', 'final'):.9f} "
            f"raw_delta_final_rad={raw_delta_final:.9f} clipped_delta_final_rad={clipped_delta_final:.9f}"
        ),
        (
            "line5 reset_offset_evidence "
            f"initial_command_face_gap_m={initial_command_gap:.9f} "
            f"initial_applied_face_gap_m={initial_applied_gap:.9f} "
            f"initial_actual_face_gap_m={initial_actual_gap:.9f} "
            f"reset_actual_minus_command_m={reset_actual_minus_command_m:.9f} "
            f"reset_ik_err_mm={float(reset['ik_reset_err_mm']):.9f} "
            f"initial_vertical_offset_m={float(reset['initial_vertical_offset_m']):.9f} "
            "status=LOWER_PRIORITY_NOT_CLEARED"
        ),
        (
            "line6 actuator_follow_evidence "
            f"env_override_line={lines['env_override_start']} "
            f"set_joint_position_target_line={lines['env_set_joint_position_target']} "
            f"decimation_line={lines['stack_decimation']} sim_dt_line={lines['stack_dt']} "
            f"stiffness_line={lines['stack_stiffness']} damping_line={lines['stack_damping']} "
            f"effort_line={lines['stack_effort']} velocity_line={lines['stack_velocity']} "
            f"direct_follow_abs_max_rad={follow_max:.9f} "
            f"actual_joint_step_abs_max_rad={actual_step_max:.9f} "
            f"actual_to_follow_step_ratio={actual_to_follow_step_ratio:.9f} "
            f"actual_extra_shortfall_over_applied_best_m={actual_extra_shortfall:.9f}"
        ),
        (
            "line7 default_off_trace_patch_design "
            "arg=--reach_trace_detail_json default=None behavior_absent=NO_OUTPUT_NO_CONTROL_CHANGE "
            "fields=reset_snapshot+per_step_command_applied_actual+per_joint_raw_clipped_target_actual_follow+joint_vel_acc_torque_limits "
            f"isaac_joint_pos_target_line={lines['isaac_joint_pos_target']} "
            f"isaac_computed_torque_line={lines['isaac_computed_torque']} "
            f"isaac_applied_torque_line={lines['isaac_applied_torque']} "
            f"isaac_joint_vel_line={lines['isaac_joint_vel']}"
        ),
        (
            "line8 next_decision_tree "
            "if_previous_target_counterfactual_enters_band=DESIGN_PREVIOUS_TARGET_BASE_RUNTIME "
            "if_applied_enters_but_actual_misses_or_torque_saturates=DESIGN_ACTUATOR_FOLLOW_RUNTIME "
            "if_reset_bias_gt_2to3mm=DESIGN_RESET_PRECONTACT_RECALIBRATION "
            "else=FK_TOOL_FRAME_VISUAL_OVERLAY_AUDIT"
        ),
        (
            "line9 verdict DESIGN_READY_DEFAULT_OFF_REACH_DETAIL_TRACE_BEFORE_ANY_GATE_RELAXATION_OR_DATASET_RL "
            "diffik_action_dataset=BLOCKED ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
    ]
    OUT_SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
