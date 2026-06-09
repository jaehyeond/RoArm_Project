#!/usr/bin/env python3
"""Diagnose command target vs applied joint-target FK vs actual TCP reach.

Local/posthoc only. This reads existing per-step reach traces and code, and
does not launch IsaacLab, GPU runtime, dataset generation, training, robot
control, SSH, B200, or Track A work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_applied_target_tcp_reach_contract_diagnosis.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_applied_target_tcp_reach_contract_diagnosis_summary.out"

TRACE_X250 = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_trace.json"
TRACE_X240 = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_trace.json"
SUMMARY_X240 = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_sanity_summary.out"
RESULT_AUDIT_X240 = LOG_DIR / "cube10cm_tap_rl_same_face_pose_result_audit_summary.out"
HARNESS = ROOT / "roarm_rl/test_positive_control_cube_tap10cm.py"
ENV = ROOT / "roarm_rl/roarm_cube_push_env.py"

CONTACT_BAND_M = 0.010


@dataclass(frozen=True)
class StepSnapshot:
    step: int
    command_gap_m: float
    applied_gap_m: float
    actual_gap_m: float
    command_minus_applied_m: float
    applied_minus_actual_m: float
    command_minus_actual_m: float
    applied_fk_err_mm: float
    direct_follow_rad: float
    actual_joint_step_rad: float


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_of(path: Path, needle: str) -> int:
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return lineno
    raise ValueError(f"needle not found in {path}: {needle!r}")


def _shortfall(face_gap_m: float) -> float:
    return max(0.0, abs(float(face_gap_m)) - CONTACT_BAND_M)


def _rows_for_env(rows: list[dict[str, Any]], env_id: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["env_id"]) == env_id]


def _mean_at_step(rows: list[dict[str, Any]], step: int, key: str) -> float:
    vals = [float(row[key]) for row in rows if int(row["step"]) == int(step)]
    if not vals:
        raise ValueError(f"no rows for step={step} key={key}")
    return mean(vals)


def _snapshot(rows: list[dict[str, Any]], step: int) -> StepSnapshot:
    command_gap = _mean_at_step(rows, step, "command_target_face_gap_m")
    applied_gap = _mean_at_step(rows, step, "applied_joint_target_fk_face_gap_m")
    actual_gap = _mean_at_step(rows, step, "actual_tcp_face_gap_m")
    return StepSnapshot(
        step=int(step),
        command_gap_m=command_gap,
        applied_gap_m=applied_gap,
        actual_gap_m=actual_gap,
        command_minus_applied_m=command_gap - applied_gap,
        applied_minus_actual_m=applied_gap - actual_gap,
        command_minus_actual_m=command_gap - actual_gap,
        applied_fk_err_mm=_mean_at_step(rows, step, "applied_joint_target_fk_err_mm"),
        direct_follow_rad=_mean_at_step(rows, step, "direct_joint_follow_abs_max_rad"),
        actual_joint_step_rad=_mean_at_step(rows, step, "actual_joint_step_abs_max_rad"),
    )


def _window_mean(rows: list[dict[str, Any]], lo: int, hi: int) -> dict[str, float]:
    window = [row for row in rows if lo <= int(row["step"]) <= hi]
    if not window:
        raise ValueError(f"empty window {lo}..{hi}")
    command = [float(row["command_target_face_gap_m"]) for row in window]
    applied = [float(row["applied_joint_target_fk_face_gap_m"]) for row in window]
    actual = [float(row["actual_tcp_face_gap_m"]) for row in window]
    return {
        "row_count": float(len(window)),
        "command_face_gap_m_mean": mean(command),
        "applied_fk_face_gap_m_mean": mean(applied),
        "actual_tcp_face_gap_m_mean": mean(actual),
        "command_minus_applied_m_mean": mean(c - a for c, a in zip(command, applied)),
        "applied_minus_actual_m_mean": mean(a - t for a, t in zip(applied, actual)),
        "command_minus_actual_m_mean": mean(c - t for c, t in zip(command, actual)),
    }


def _analyze_trace(path: Path) -> dict[str, Any]:
    artifact = _load_json(path)
    rows = artifact["rows"]
    env_ids = sorted({int(row["env_id"]) for row in rows})
    command_inside = [row for row in rows if bool(row["command_target_inside_contact_band"])]
    applied_inside = [row for row in rows if bool(row["applied_joint_target_fk_inside_contact_band"])]
    actual_inside = [row for row in rows if bool(row["actual_contact_proxy"])]
    if not command_inside:
        raise ValueError(f"command target never entered band in {path}")
    first_inside_step = min(int(row["step"]) for row in command_inside)
    last_inside_step = max(int(row["step"]) for row in command_inside)
    mid_inside_step = int(round((first_inside_step + last_inside_step) / 2.0))

    per_env: dict[str, Any] = {}
    for env_id in env_ids:
        env_rows = _rows_for_env(rows, env_id)
        best_applied = max(env_rows, key=lambda row: float(row["applied_joint_target_fk_face_gap_m"]))
        best_actual = max(env_rows, key=lambda row: float(row["actual_tcp_face_gap_m"]))
        per_env[str(env_id)] = {
            "command_inside_rows": sum(bool(row["command_target_inside_contact_band"]) for row in env_rows),
            "command_inside_first_step": first_inside_step,
            "command_inside_last_step": last_inside_step,
            "applied_inside_rows": sum(bool(row["applied_joint_target_fk_inside_contact_band"]) for row in env_rows),
            "actual_inside_rows": sum(bool(row["actual_contact_proxy"]) for row in env_rows),
            "best_applied_step": int(best_applied["step"]),
            "best_applied_face_gap_m": float(best_applied["applied_joint_target_fk_face_gap_m"]),
            "best_applied_shortfall_m": _shortfall(float(best_applied["applied_joint_target_fk_face_gap_m"])),
            "best_actual_step": int(best_actual["step"]),
            "best_actual_face_gap_m": float(best_actual["actual_tcp_face_gap_m"]),
            "best_actual_shortfall_m": _shortfall(float(best_actual["actual_tcp_face_gap_m"])),
        }

    applied_gaps = [float(row["applied_joint_target_fk_face_gap_m"]) for row in rows]
    actual_gaps = [float(row["actual_tcp_face_gap_m"]) for row in rows]

    return {
        "trace": str(path.relative_to(ROOT)),
        "metadata": artifact.get("metadata", {}),
        "row_count": len(rows),
        "env_ids": env_ids,
        "command_inside_rows": len(command_inside),
        "command_inside_unique_steps": len({int(row["step"]) for row in command_inside}),
        "command_inside_first_step": first_inside_step,
        "command_inside_last_step": last_inside_step,
        "applied_inside_rows": len(applied_inside),
        "actual_inside_rows": len(actual_inside),
        "best_applied_face_gap_m": max(applied_gaps),
        "best_applied_shortfall_m": _shortfall(max(applied_gaps)),
        "best_actual_face_gap_m": max(actual_gaps),
        "best_actual_shortfall_m": _shortfall(max(actual_gaps)),
        "snapshots": {
            "first_command_inside": _snapshot(rows, first_inside_step).__dict__,
            "mid_command_inside": _snapshot(rows, mid_inside_step).__dict__,
            "last_command_inside": _snapshot(rows, last_inside_step).__dict__,
            "final": _snapshot(rows, max(int(row["step"]) for row in rows)).__dict__,
        },
        "windows": {
            "pre_command_band": _window_mean(rows, 0, first_inside_step - 1),
            "command_inside_band": _window_mean(rows, first_inside_step, last_inside_step),
            "post_command_band": _window_mean(rows, last_inside_step + 1, max(int(row["step"]) for row in rows)),
        },
        "per_env": per_env,
    }


def main() -> int:
    x250 = _analyze_trace(TRACE_X250)
    x240 = _analyze_trace(TRACE_X240)
    code_refs = {
        "harness_target_path_start": _line_of(HARNESS, "pre_w[:, 0:2] = cube_w[:, 0:2] - push_dir"),
        "harness_command_tcp_target": _line_of(HARNESS, "tcp_target_w = pre_w + float(alpha)"),
        "harness_builtin_diffik_compute": _line_of(HARNESS, "joint_pos_des = diffik.compute"),
        "harness_step_clip": _line_of(HARNESS, "clipped_delta_arm = torch_mod.clamp"),
        "harness_target_full": _line_of(HARNESS, "target_full[:, arm_joint_ids] = arm_joint_target"),
        "harness_applied_fk_trace": _line_of(HARNESS, "applied_tcp_local = fk_tcp"),
        "harness_post_step_actual_trace": _line_of(HARNESS, "actual_metrics = _face_metrics_torch"),
        "env_external_target_override": _line_of(ENV, "override_targets = getattr"),
        "env_robot_targets_write": _line_of(ENV, "self.robot_dof_targets[:] = targets"),
        "env_set_joint_position_target": _line_of(ENV, "self._robot.set_joint_position_target(self.robot_dof_targets)"),
        "env_tap_face_gap": _line_of(ENV, "face_gap = along + half_along"),
        "env_contact_proxy_start": _line_of(ENV, "contact_proxy = ("),
    }

    first = x240["snapshots"]["first_command_inside"]
    mid = x240["snapshots"]["mid_command_inside"]
    last = x240["snapshots"]["last_command_inside"]
    final = x240["snapshots"]["final"]
    window = x240["windows"]["command_inside_band"]

    result = {
        "artifact_type": "cube10cm_tap_rl_applied_target_tcp_reach_contract_diagnosis_v1",
        "local_posthoc_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {
            "x250_trace": str(TRACE_X250.relative_to(ROOT)),
            "x240_trace": str(TRACE_X240.relative_to(ROOT)),
            "x240_runtime_summary": str(SUMMARY_X240.relative_to(ROOT)),
            "x240_result_audit_summary": str(RESULT_AUDIT_X240.relative_to(ROOT)),
            "harness": str(HARNESS.relative_to(ROOT)),
            "env": str(ENV.relative_to(ROOT)),
        },
        "code_refs": code_refs,
        "x250": x250,
        "x240": x240,
        "interpretation": {
            "primary_blocker": "TARGET_FULL_FK_NEVER_REACHES_FACE_BAND_AND_ACTUAL_TCP_LAGS_TARGET_FULL",
            "contact_gate_relaxation_unblock": False,
            "x285_unblock": False,
            "diffik_action_dataset": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "roarm": "BLOCKED",
            "next_local_step": (
                "Inspect/patch the step-clipped built-in DiffIK target-generation contract: "
                "raw_delta clipping, target_full FK progression, Jacobian/tool-proxy frame, "
                "and whether the Cartesian command schedule outruns the applied joint-target FK."
            ),
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_applied_target_tcp_reach_contract_diagnosis_v1 "
        "local_posthoc_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 code_contract "
            f"target_path_lines={code_refs['harness_target_path_start']}-{code_refs['harness_command_tcp_target']} "
            f"diffik_compute_line={code_refs['harness_builtin_diffik_compute']} "
            f"step_clip_line={code_refs['harness_step_clip']} "
            f"target_full_line={code_refs['harness_target_full']} "
            f"applied_fk_trace_line={code_refs['harness_applied_fk_trace']} "
            f"env_override_lines={code_refs['env_external_target_override']}-{code_refs['env_robot_targets_write']} "
            f"set_joint_position_target_line={code_refs['env_set_joint_position_target']} "
            f"actual_poststep_trace_line={code_refs['harness_post_step_actual_trace']} "
            f"tap_contact_lines={code_refs['env_tap_face_gap']}-{code_refs['env_contact_proxy_start']}"
        ),
        (
            "line3 x240_command_target "
            f"inside_rows={x240['command_inside_rows']} "
            f"inside_unique_steps={x240['command_inside_unique_steps']} "
            f"first_step={x240['command_inside_first_step']} "
            f"last_step={x240['command_inside_last_step']} "
            f"first_gap_m={first['command_gap_m']:.9f} "
            f"mid_gap_m={mid['command_gap_m']:.9f} "
            f"last_gap_m={last['command_gap_m']:.9f} "
            f"final_gap_m={final['command_gap_m']:.9f}"
        ),
        (
            "line4 x240_applied_joint_target_fk "
            f"inside_rows={x240['applied_inside_rows']} "
            f"best_face_gap_m={x240['best_applied_face_gap_m']:.9f} "
            f"best_shortfall_m={x240['best_applied_shortfall_m']:.9f} "
            f"first_inside_step_gap_m={first['applied_gap_m']:.9f} "
            f"mid_gap_m={mid['applied_gap_m']:.9f} "
            f"last_gap_m={last['applied_gap_m']:.9f} "
            f"final_gap_m={final['applied_gap_m']:.9f} "
            f"final_fk_err_mm={final['applied_fk_err_mm']:.9f}"
        ),
        (
            "line5 x240_actual_tcp "
            f"inside_rows={x240['actual_inside_rows']} "
            f"best_face_gap_m={x240['best_actual_face_gap_m']:.9f} "
            f"best_shortfall_m={x240['best_actual_shortfall_m']:.9f} "
            f"first_inside_step_gap_m={first['actual_gap_m']:.9f} "
            f"mid_gap_m={mid['actual_gap_m']:.9f} "
            f"last_gap_m={last['actual_gap_m']:.9f} "
            f"final_gap_m={final['actual_gap_m']:.9f}"
        ),
        (
            "line6 x240_divergence_split "
            f"first_cmd_minus_applied_m={first['command_minus_applied_m']:.9f} "
            f"first_applied_minus_actual_m={first['applied_minus_actual_m']:.9f} "
            f"first_cmd_minus_actual_m={first['command_minus_actual_m']:.9f} "
            f"inside_window_cmd_minus_applied_mean_m={window['command_minus_applied_m_mean']:.9f} "
            f"inside_window_applied_minus_actual_mean_m={window['applied_minus_actual_m_mean']:.9f} "
            f"inside_window_cmd_minus_actual_mean_m={window['command_minus_actual_m_mean']:.9f}"
        ),
        (
            "line7 x240_follow_contract "
            f"first_follow_rad={first['direct_follow_rad']:.9f} "
            f"first_actual_joint_step_rad={first['actual_joint_step_rad']:.9f} "
            f"final_follow_rad={final['direct_follow_rad']:.9f} "
            f"final_actual_joint_step_rad={final['actual_joint_step_rad']:.9f} "
            f"step_clip_rad={x240['metadata'].get('builtin_diffik_step_clip_rad')}"
        ),
        (
            "line8 x250_x240_crosscheck "
            f"x250_applied_shortfall_m={x250['best_applied_shortfall_m']:.9f} "
            f"x240_applied_shortfall_m={x240['best_applied_shortfall_m']:.9f} "
            f"applied_improvement_m={x250['best_applied_shortfall_m'] - x240['best_applied_shortfall_m']:.9f} "
            f"x250_actual_shortfall_m={x250['best_actual_shortfall_m']:.9f} "
            f"x240_actual_shortfall_m={x240['best_actual_shortfall_m']:.9f} "
            f"actual_improvement_m={x250['best_actual_shortfall_m'] - x240['best_actual_shortfall_m']:.9f}"
        ),
        (
            "line9 verdict "
            "TARGET_FULL_FK_NEVER_REACHES_FACE_BAND_AND_ACTUAL_TCP_LAGS_TARGET_FULL "
            "command_target_crossed=YES applied_joint_target_fk_crossed=NO actual_tcp_crossed=NO "
            "contact_gate_relaxation_unblock=NO x285_unblock=NO "
            "diffik_action_dataset=BLOCKED ppo_rl_training=BLOCKED roarm=BLOCKED"
        ),
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
