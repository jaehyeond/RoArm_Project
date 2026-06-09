#!/usr/bin/env python3
"""Design audit for cube10cm fixed pose under same-center vs same-face anchors.

This is local-only and does not launch IsaacLab. It uses the current 10cm
reach trace plus repo constants to choose a single fixed pose for the next
IK/TCP diagnostic runtime.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
ENV_PY = ROOT / "roarm_rl/roarm_cube_push_env.py"
HARNESS_PY = ROOT / "roarm_rl/test_positive_control_cube_tap10cm.py"
TRACE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_trace.json"
TRACE_AUDIT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_per_step_reach_trace_result_audit_summary.out"

OUT_JSON = LOG_DIR / "cube10cm_tap_rl_same_center_vs_same_face_pose_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_same_center_vs_same_face_pose_audit_summary.out"


def _line(path: Path, number: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[number - 1] if 1 <= number <= len(lines) else ""


def _float_regex(text: str, pattern: str) -> float:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"missing pattern: {pattern}")
    return float(match.group(1))


def _rows() -> list[dict[str, Any]]:
    trace = json.loads(TRACE_JSON.read_text(encoding="utf-8"))
    return trace["rows"]


def main() -> int:
    env_text = ENV_PY.read_text(encoding="utf-8")
    harness_text = HARNESS_PY.read_text(encoding="utf-8")
    three_size = _float_regex(env_text, r"^CUBE_SIZE_M\s*=\s*([0-9.]+)")
    ten_size = _float_regex(env_text, r"^CUBE10CM_SIZE_M\s*=\s*([0-9.]+)")
    three_x_min = _float_regex(env_text, r"^\s+cube_x_min:\s*float\s*=\s*([0-9.]+)")
    current_center_x = _float_regex(
        harness_text,
        r'parser\.add_argument\("--fixed_cube_x_m",\s*type=float,\s*default=([0-9.]+)\)',
    )
    current_center_y = _float_regex(
        harness_text,
        r'parser\.add_argument\("--fixed_cube_y_m",\s*type=float,\s*default=([0-9.]+)\)',
    )
    contact_band_m = _float_regex(env_text, r"^\s+tap_contact_face_band_m:\s*float\s*=\s*([0-9.]+)")

    half3 = three_size * 0.5
    half10 = ten_size * 0.5
    rows = _rows()
    actual_best_face_gap = max(float(row["actual_tcp_face_gap_m"]) for row in rows)
    current_shortfall_m = max(0.0, -contact_band_m - actual_best_face_gap)
    observed_touch_center_max_x = current_center_x + actual_best_face_gap + contact_band_m

    candidates = {
        "same_center_current": {
            "fixed_cube_x_m": current_center_x,
            "near_face_x_m": current_center_x - half10,
            "interpretation": "current 10cm center is unchanged from the harness default",
        },
        "same_near_face_as_current_3cm_center": {
            "fixed_cube_x_m": current_center_x - half3 + half10,
            "near_face_x_m": current_center_x - half3,
            "interpretation": "keeps the near face that a 3cm cube would have at center x=0.250",
        },
        "same_near_face_as_3cm_xmin": {
            "fixed_cube_x_m": three_x_min - half3 + half10,
            "near_face_x_m": three_x_min - half3,
            "interpretation": "keeps the near face of the 3cm low-x workspace boundary",
        },
        "observed_reach_boundary": {
            "fixed_cube_x_m": observed_touch_center_max_x,
            "near_face_x_m": observed_touch_center_max_x - half10,
            "interpretation": "largest 10cm center whose contact band lower edge matches previous actual best TCP reach",
        },
    }
    selected_x = min(
        candidates["same_near_face_as_3cm_xmin"]["fixed_cube_x_m"],
        candidates["observed_reach_boundary"]["fixed_cube_x_m"],
    )
    selected = {
        "fixed_cube_x_m": round(selected_x, 3),
        "fixed_cube_y_m": current_center_y,
        "fixed_push_dir": [1.0, 0.0],
        "reason": "same-near-face as 3cm low-x boundary and within observed actual TCP reach boundary",
        "changed": "fixed_cube_x_m_only",
        "kept": {
            "controller_mode": "isaac_builtin_diffik_step_clipped_direct_apply",
            "steps": 580,
            "closed_loop_push_steps": 580,
            "builtin_diffik_step_clip_rad": 0.010,
            "episode_length_s": 6.08,
            "fixed_cube_y_m": current_center_y,
            "fixed_push_dir": [1.0, 0.0],
            "contact_gate": "unchanged_strict",
        },
    }

    out_base = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace"
    command = (
        "conda run -n isaaclab --no-capture-output python -u -m "
        "roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 580 --seed 962 --device cuda:0 "
        "--controller_mode isaac_builtin_diffik_step_clipped_direct_apply "
        "--closed_loop_push_steps 580 --builtin_diffik_step_clip_rad 0.010 "
        "--episode_length_s 6.08 "
        f"--fixed_cube_x_m {selected['fixed_cube_x_m']:.3f} "
        f"--fixed_cube_y_m {selected['fixed_cube_y_m']:.3f} "
        f"--reach_trace_json {out_base}_trace.json "
        f"--out_json {out_base}_sanity.json "
        f"--out_summary {out_base}_sanity_summary.out"
    )

    artifact = {
        "artifact_type": "cube10cm_tap_rl_same_center_vs_same_face_pose_audit_v1",
        "local_design_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "sources": {
            "env_py": str(ENV_PY),
            "harness_py": str(HARNESS_PY),
            "trace_json": str(TRACE_JSON),
            "trace_audit_summary": str(TRACE_AUDIT_SUMMARY),
            "trace_audit_line5": _line(TRACE_AUDIT_SUMMARY, 5),
            "trace_audit_line6": _line(TRACE_AUDIT_SUMMARY, 6),
        },
        "constants": {
            "three_cm_size_m": three_size,
            "ten_cm_size_m": ten_size,
            "three_cm_half_m": half3,
            "ten_cm_half_m": half10,
            "three_cm_x_min_m": three_x_min,
            "current_fixed_cube_x_m": current_center_x,
            "current_fixed_cube_y_m": current_center_y,
            "contact_band_m": contact_band_m,
        },
        "previous_reach": {
            "actual_best_face_gap_m": actual_best_face_gap,
            "current_contact_shortfall_m": current_shortfall_m,
            "observed_touch_center_max_x_m": observed_touch_center_max_x,
        },
        "candidates": candidates,
        "selected": selected,
        "candidate_runtime": {
            "status": "DESIGNED_FOR_USER_APPROVED_TINY_TEST",
            "command": command,
        },
        "outcome": {
            "verdict": "SELECT_10CM_X240_SAME_LOWX_NEAR_FACE_POSE",
            "same_3cm_center_near_face_rejected": True,
            "same_3cm_center_rejection_reason": "for +x push it moves the 10cm near face farther from the previous reachable TCP range",
        },
    }

    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_same_center_vs_same_face_pose_audit_v1 "
        "local_design_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 current_same_center "
        f"ten_cm_center_x={current_center_x:.3f} ten_cm_near_face_x={current_center_x - half10:.3f} "
        f"actual_best_face_gap_m={actual_best_face_gap:.9f} current_shortfall_m={current_shortfall_m:.9f}",
        "line3 same_face_candidates "
        f"same_3cm_center_face_center_x={candidates['same_near_face_as_current_3cm_center']['fixed_cube_x_m']:.3f} "
        f"same_3cm_center_face_x={candidates['same_near_face_as_current_3cm_center']['near_face_x_m']:.3f} "
        "verdict=REJECT_FOR_PLUSX_FARTHER_FACE "
        f"same_3cm_xmin_face_center_x={candidates['same_near_face_as_3cm_xmin']['fixed_cube_x_m']:.3f} "
        f"same_3cm_xmin_face_x={candidates['same_near_face_as_3cm_xmin']['near_face_x_m']:.3f}",
        "line4 observed_reach_boundary "
        f"touch_center_max_x={observed_touch_center_max_x:.9f} selected_fixed_cube_x_m={selected['fixed_cube_x_m']:.3f} "
        f"selected_near_face_x_m={selected['fixed_cube_x_m'] - half10:.3f} selected_fixed_cube_y_m={selected['fixed_cube_y_m']:.3f}",
        "line5 selected_runtime_contract "
        "change=fixed_cube_x_m_only keep=controller_stepclip_h580_ep608_y0_pushx_strict_contact_gate "
        "runtime_status=DESIGNED_FOR_USER_APPROVED_TINY_TEST",
        f"line6 command {command}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
