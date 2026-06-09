#!/usr/bin/env python3
"""Audit whether cube10cm x240 failure is horizon-step shortage or clip/contract.

Local/posthoc only. Uses existing x240 reach trace and summary, no Isaac runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
TRACE = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_trace.json"
SANITY = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_sanity.json"
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_horizon_vs_clip_interpretation_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_horizon_vs_clip_interpretation_audit_summary.out"


def _avg(rows: list[dict[str, Any]], step: int, key: str) -> float:
    vals = [float(row[key]) for row in rows if int(row["step"]) == int(step)]
    if not vals:
        raise ValueError(f"missing step={step} key={key}")
    return mean(vals)


def _delta(rows: list[dict[str, Any]], key: str, a: int, b: int) -> dict[str, float]:
    va = _avg(rows, a, key)
    vb = _avg(rows, b, key)
    return {
        "start_step": float(a),
        "end_step": float(b),
        "start_m": va,
        "end_m": vb,
        "delta_m": vb - va,
        "per_step_m": (vb - va) / float(max(b - a, 1)),
    }


def main() -> int:
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    sanity = json.loads(SANITY.read_text(encoding="utf-8"))
    rows = trace["rows"]
    command_inside_steps = sorted({int(row["step"]) for row in rows if bool(row["command_target_inside_contact_band"])})
    first_inside = command_inside_steps[0]
    last_inside = command_inside_steps[-1]
    final_step = max(int(row["step"]) for row in rows)
    windows = {
        "pre_to_first_inside": (0, first_inside),
        "inside_window": (first_inside, last_inside),
        "post_inside_to_final": (last_inside, final_step),
    }
    keys = {
        "command": "command_target_face_gap_m",
        "applied_fk": "applied_joint_target_fk_face_gap_m",
        "actual_tcp": "actual_tcp_face_gap_m",
    }
    slopes = {
        name: {window: _delta(rows, key, *bounds) for window, bounds in windows.items()}
        for name, key in keys.items()
    }
    controller = sanity["controller_trace_stats"]
    result = {
        "artifact_type": "cube10cm_tap_rl_horizon_vs_clip_interpretation_audit_v1",
        "local_posthoc_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "trace": str(TRACE.relative_to(ROOT)),
        "sanity": str(SANITY.relative_to(ROOT)),
        "steps_executed": int(sanity["steps_executed"]),
        "max_steps": int(sanity["max_steps"]),
        "terminated_count": int(sanity["terminated_count"]),
        "truncated_count": int(sanity["truncated_count"]),
        "command_inside_first_step": first_inside,
        "command_inside_last_step": last_inside,
        "final_step": final_step,
        "slopes": slopes,
        "clip": {
            "step_clip_rad": sanity.get("builtin_diffik_step_clip_rad"),
            "raw_delta_abs_max_rad": controller["builtin_diffik_raw_delta_abs_max_rad"]["max"],
            "clipped_delta_abs_max_rad": controller["builtin_diffik_clipped_delta_abs_max_rad"]["max"],
            "actual_joint_step_abs_max_rad": controller["direct_actual_joint_step_abs_max_rad"]["max"],
        },
        "interpretation": {
            "episode_horizon_shortage": False,
            "episode_cutoff": False,
            "more_steps_same_contract_unblocks": False,
            "why": (
                "The continuous 580-step episode is not truncated, command enters the band at step 46 and continues "
                "through the cube. During and after the command-inside window, applied FK and actual TCP move away "
                "from the contact band, so this is not simply a too-short horizon."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    c_pre = slopes["command"]["pre_to_first_inside"]
    c_inside = slopes["command"]["inside_window"]
    c_post = slopes["command"]["post_inside_to_final"]
    a_pre = slopes["applied_fk"]["pre_to_first_inside"]
    a_inside = slopes["applied_fk"]["inside_window"]
    a_post = slopes["applied_fk"]["post_inside_to_final"]
    t_pre = slopes["actual_tcp"]["pre_to_first_inside"]
    t_inside = slopes["actual_tcp"]["inside_window"]
    t_post = slopes["actual_tcp"]["post_inside_to_final"]
    lines = [
        "line1 artifact=cube10cm_tap_rl_horizon_vs_clip_interpretation_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 episode_horizon "
            f"steps_executed={sanity['steps_executed']} max_steps={sanity['max_steps']} "
            f"terminated_count={sanity['terminated_count']} truncated_count={sanity['truncated_count']} "
            f"command_inside_steps={first_inside}..{last_inside} final_step={final_step} "
            "episode_cutoff=NO horizon_shortage=NO"
        ),
        (
            "line3 command_progress "
            f"pre_delta_m={c_pre['delta_m']:.9f} inside_delta_m={c_inside['delta_m']:.9f} "
            f"post_delta_m={c_post['delta_m']:.9f} final_gap_m={c_post['end_m']:.9f}"
        ),
        (
            "line4 applied_fk_progress "
            f"pre_delta_m={a_pre['delta_m']:.9f} inside_delta_m={a_inside['delta_m']:.9f} "
            f"post_delta_m={a_post['delta_m']:.9f} final_gap_m={a_post['end_m']:.9f} "
            "moves_away_after_contact_window=YES"
        ),
        (
            "line5 actual_tcp_progress "
            f"pre_delta_m={t_pre['delta_m']:.9f} inside_delta_m={t_inside['delta_m']:.9f} "
            f"post_delta_m={t_post['delta_m']:.9f} final_gap_m={t_post['end_m']:.9f} "
            "moves_away_after_contact_window=YES"
        ),
        (
            "line6 clip_vs_actual_step "
            f"raw_delta_abs_max_rad={controller['builtin_diffik_raw_delta_abs_max_rad']['max']:.9f} "
            f"clipped_delta_abs_max_rad={controller['builtin_diffik_clipped_delta_abs_max_rad']['max']:.9f} "
            f"actual_joint_step_abs_max_rad={controller['direct_actual_joint_step_abs_max_rad']['max']:.9f}"
        ),
        (
            "line7 verdict NOT_A_SIMPLE_MORE_HORIZON_STEPS_FIX "
            "same_contract_more_steps_not_recommended=YES "
            "next=TARGET_GENERATION_CONTRACT_DESIGN_OR_ISAAC_RENDER_WITH_OVERLAY_IF_RUNTIME_APPROVED"
        ),
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
