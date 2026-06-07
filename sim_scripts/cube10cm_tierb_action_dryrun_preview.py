"""Build a local tiny dry-run preview for cube10cm Tier-B action rows.

This script uses the teacher-quality revalidation result to select the best
contact-centered Tier-B row policy, then writes a small JSONL preview of action
rows from the existing trace. It is not a large dataset, not LeRobot/RLDS, not
training data, and it performs no IsaacLab/GPU runtime, training, robot control,
SSH, or trace mutation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_REVALIDATION = LOG_DIR / "cube10cm_teacher_quality_revalidation_audit.json"
DEFAULT_TRACE_CSV = (
    LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_trace.csv"
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tierb_action_dryrun_preview.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tierb_action_dryrun_preview_summary.out"
DEFAULT_OUT_ROWS = LOG_DIR / "cube10cm_tierb_action_dryrun_preview_rows.jsonl"

FEATURE_COLUMNS = (
    "push_dx",
    "push_dy",
    "phase_alpha",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "arm_joint_0_rad",
    "arm_joint_1_rad",
    "arm_joint_2_rad",
    "arm_joint_3_rad",
    "arm_joint_4_rad",
    "gripper_joint_rad",
)

ACTION_COLUMNS = (
    "joint_delta_0_rad",
    "joint_delta_1_rad",
    "joint_delta_2_rad",
    "joint_delta_3_rad",
    "joint_delta_4_rad",
)

QUALITY_COLUMNS = (
    "clip_any",
    "clip_joint_count",
    "joint_follow_err_0_rad",
    "joint_follow_err_1_rad",
    "joint_follow_err_2_rad",
    "joint_follow_err_3_rad",
    "joint_follow_err_4_rad",
    "raw_delta_0_rad",
    "raw_delta_1_rad",
    "raw_delta_2_rad",
    "raw_delta_3_rad",
    "raw_delta_4_rad",
)

FORBIDDEN_FIELDS = (
    "final_1cm_relocation",
    "final_1mm_retention",
    "post_push_final_position",
    "success_marker",
    "controlled_push",
    "low_motion",
    "final_disp_m",
    "final_relocation",
    "target_xy_dist_m",
    "cube_success_disp_m",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _flatten_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            key_s = str(key)
            full = f"{prefix}.{key_s}" if prefix else key_s
            keys.add(key_s)
            keys.add(full)
            keys.update(_flatten_keys(child, full))
    elif isinstance(value, list):
        for child in value:
            keys.update(_flatten_keys(child, prefix))
    return keys


def _action_abs_values(rows: list[dict[str, Any]]) -> list[float]:
    values = []
    for row in rows:
        for value in row["action"].values():
            values.append(abs(float(value)))
    return values


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round(0.95 * (len(ordered) - 1)))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revalidation_json", type=Path, default=DEFAULT_REVALIDATION)
    parser.add_argument("--trace_csv", type=Path, default=DEFAULT_TRACE_CSV)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--out_rows_jsonl", type=Path, default=DEFAULT_OUT_ROWS)
    args = parser.parse_args()

    revalidation = _load_json(args.revalidation_json)
    best = revalidation.get("best_policy", {})
    if best.get("policy") != "contact_to_p16":
        raise SystemExit(f"unexpected best policy: {best.get('policy')}")
    if int(best.get("follow_ok_count", 0)) != int(revalidation.get("event_count", 0)):
        raise SystemExit("best policy is not Tier-B-only/follow-ok for all events")
    if int(best.get("strict_clean_count", 0)) != 0:
        raise SystemExit("unexpected strict clean rows; dry-run contract assumes Tier B, not Tier A")

    rel_start, rel_end = [int(x) for x in best.get("relative_window", [0, 16])]
    anchors = {
        int(row["env_id"]): int(row["anchor_step"])
        for row in revalidation.get("policy_results", [])
        if row.get("policy") == "contact_to_p16"
        for row in row.get("per_env", [])
        if bool(row.get("accepted", False))
    }

    with args.trace_csv.open(newline="", encoding="utf-8") as f:
        trace_rows = list(csv.DictReader(f))
    by_env: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        by_env[_int(row.get("env_id"))].append(row)
    for env_rows in by_env.values():
        env_rows.sort(key=lambda row: _int(row.get("step")))

    preview_rows: list[dict[str, Any]] = []
    for env_id, anchor in sorted(anchors.items()):
        start = anchor + rel_start
        end = anchor + rel_end
        for row in by_env[env_id]:
            step = _int(row.get("step"))
            if not (start <= step <= end):
                continue
            preview_rows.append(
                {
                    "record_type": "cube10cm_tierb_action_dryrun_row_v1",
                    "source": {
                        "seed": 962,
                        "env_id": env_id,
                        "frame": _int(row.get("frame")),
                        "step": step,
                        "anchor_step": anchor,
                        "relative_step": step - anchor,
                        "teacher_window_policy": "contact_to_p16",
                    },
                    "observation": {key: _float(row.get(key)) for key in FEATURE_COLUMNS},
                    "action": {key: _float(row.get(key)) for key in ACTION_COLUMNS},
                    "quality_metadata": {key: _float(row.get(key)) for key in QUALITY_COLUMNS if key in row},
                    "labels": {
                        "contact_evidence_window": True,
                        "reaction_signal_window": True,
                        "no_overshoot_window": True,
                        "quality_tier": "B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH",
                        "action_teacher_usable_for_training": False,
                    },
                }
            )

    forbidden_present = sorted(set(FORBIDDEN_FIELDS).intersection(_flatten_keys(preview_rows)))
    action_abs = _action_abs_values(preview_rows)
    clip_any_count = sum(1 for row in preview_rows if _float(row["quality_metadata"].get("clip_any")) >= 0.5)
    rows_per_env = defaultdict(int)
    for row in preview_rows:
        rows_per_env[int(row["source"]["env_id"])] += 1

    rows_per_env_min = min(rows_per_env.values()) if rows_per_env else 0
    rows_per_env_max = max(rows_per_env.values()) if rows_per_env else 0
    dryrun_ready = (
        len(rows_per_env) == len(anchors)
        and rows_per_env_min >= 4
        and len(anchors) == int(revalidation.get("event_count", 0))
        and not forbidden_present
    )
    result = {
        "artifact_type": "cube10cm_tierb_action_dryrun_preview_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_dryrun_preview_only": True,
        "no_gpu_isaaclab_large_dataset_training_robot_ssh": True,
        "not_lerobot_or_rlds_dataset": True,
        "not_training_data": True,
        "input_revalidation": str(args.revalidation_json),
        "input_trace_csv": str(args.trace_csv),
        "selected_policy": {
            "policy": "contact_to_p16",
            "relative_window": [rel_start, rel_end],
            "quality_tier": "B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH",
            "strict_clean_teacher": False,
            "clip_high": True,
        },
        "counts": {
            "events": len(anchors),
            "rows": len(preview_rows),
            "rows_per_env_min": rows_per_env_min,
            "rows_per_env_max": rows_per_env_max,
            "expected_sparse_trace_rows_per_env_min": 4,
            "clip_any_rows": clip_any_count,
            "forbidden_fields_present": forbidden_present,
        },
        "action_abs_stats_rad": {
            "mean": sum(action_abs) / len(action_abs) if action_abs else 0.0,
            "p95": _p95(action_abs),
            "max": max(action_abs) if action_abs else 0.0,
        },
        "statuses": {
            "tierb_action_dryrun_preview": "READY_LOCAL_ONLY" if dryrun_ready else "BLOCKED",
            "actual_action_teacher_dataset": "NOT_BUILT",
            "large_isaaclab_dataset": "BLOCKED",
            "isaaclab_rl": "BLOCKED",
            "roarm_m3_pro": "BLOCKED",
        },
        "out_rows_jsonl": str(args.out_rows_jsonl),
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_rows_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in preview_rows),
        encoding="utf-8",
    )

    lines = [
        "line1 artifact=cube10cm_tierb_action_dryrun_preview_v1 "
        "local_dryrun_preview_only=YES gpu_runtime=NO large_dataset=NO training=NO robot_control=NO",
        (
            "line2 selected_policy "
            f"policy=contact_to_p16 rel=[{rel_start},{rel_end}] tier=B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH "
            "strict_clean_teacher=NO clip_high=YES"
        ),
        (
            "line3 rows "
            f"events={len(anchors)} rows={len(preview_rows)} rows_per_env_min={result['counts']['rows_per_env_min']} "
            f"rows_per_env_max={result['counts']['rows_per_env_max']} forbidden_present={forbidden_present}"
        ),
        (
            "line4 action_stats "
            f"abs_mean_rad={result['action_abs_stats_rad']['mean']:.9f} "
            f"abs_p95_rad={result['action_abs_stats_rad']['p95']:.9f} "
            f"abs_max_rad={result['action_abs_stats_rad']['max']:.9f} clip_any_rows={clip_any_count}"
        ),
        (
            "line5 status "
            f"tierb_action_dryrun_preview={result['statuses']['tierb_action_dryrun_preview']} "
            "actual_action_teacher_dataset=NOT_BUILT large_isaaclab_dataset=BLOCKED "
            "isaaclab_rl=BLOCKED roarm_m3_pro=BLOCKED"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
