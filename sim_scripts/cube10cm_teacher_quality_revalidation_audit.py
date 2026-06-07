"""Local revalidation audit for cube10cm DiffIK teacher quality.

This audit checks whether seed962 teacher quality is blocked because the current
reaction-window/action-row definition is too wide, or because the underlying
DiffIK/actuator tracking is poor across any reasonable contact-centered slice.
It reads existing trace/window artifacts only and performs no IsaacLab/GPU
runtime, dataset generation, training, robot control, SSH, or trace mutation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_WINDOW_AUDIT = LOG_DIR / "cube10cm_reaction_window_seed962_audit.json"
DEFAULT_TRACE_CSV = (
    LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_trace.csv"
)
DEFAULT_SUMMARY_JSON = (
    LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_summary.json"
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_teacher_quality_revalidation_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_teacher_quality_revalidation_audit_summary.out"


WINDOW_POLICIES = (
    ("official_reaction_window_m24_p48", -24, 48),
    ("pre_anchor_m24_p0", -24, 0),
    ("contact_micro_m8_p8", -8, 8),
    ("contact_to_p16", 0, 16),
    ("contact_to_p24", 0, 24),
    ("pre8_to_p16", -8, 16),
    ("pre4_to_p12", -4, 12),
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


def _flag(row: dict[str, str], key: str) -> bool:
    return _float(row.get(key)) >= 0.5


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round(0.95 * (len(ordered) - 1)))
    return ordered[idx]


def _joint_follow_values(rows: list[dict[str, str]]) -> list[float]:
    if not rows:
        return []
    keys = [key for key in rows[0] if key.startswith("joint_follow_err_") and key.endswith("_rad")]
    values: list[float] = []
    for row in rows:
        values.extend(abs(_float(row.get(key))) for key in keys)
    return values


def _joint_raw_values(rows: list[dict[str, str]]) -> list[float]:
    if not rows:
        return []
    keys = [key for key in rows[0] if key.startswith("raw_delta_") and key.endswith("_rad")]
    values: list[float] = []
    for row in rows:
        values.extend(abs(_float(row.get(key))) for key in keys)
    return values


def _tier(accepted: bool, clip_any_rate: float, follow_p95_to_cap: float) -> str:
    if not accepted:
        return "REJECTED"
    if clip_any_rate <= 0.5 and follow_p95_to_cap <= 1.0:
        return "A_CLEAN_DIFFIK_TEACHER"
    if follow_p95_to_cap <= 1.0:
        return "B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH"
    return "C_REACTION_VALID_FOLLOW_LAG"


def _slice_summary(
    *,
    env_id: int,
    anchor: int,
    rows: list[dict[str, str]],
    start_delta: int,
    end_delta: int,
    max_joint_step_rad: float,
    reaction_disp_m: float,
    reaction_z_delta_m: float,
    reaction_speed_mps: float,
    reaction_tip_angle_deg: float,
    overshoot_disp_m: float,
) -> dict[str, Any]:
    start = anchor + start_delta
    end = anchor + end_delta
    window_rows = [row for row in rows if start <= _int(row.get("step")) <= end]
    if not window_rows:
        return {
            "env_id": env_id,
            "anchor_step": anchor,
            "start_step": start,
            "end_step": end,
            "rows": 0,
            "accepted": False,
            "quality_tier": "REJECTED",
            "reject_reasons": ["empty_slice"],
        }

    z0 = _float(rows[0].get("cube_z_m"))
    max_disp = max(_float(row.get("disp_along_push_m")) for row in window_rows)
    max_z_delta = max(_float(row.get("cube_z_m")) - z0 for row in window_rows)
    max_speed = max(_float(row.get("cube_speed_mps")) for row in window_rows)
    max_tip = max(_float(row.get("tip_angle_deg")) for row in window_rows)
    contact_evidence = any(
        _flag(row, "measured_contact_now") or _flag(row, "measured_contact_seen") or _flag(row, "contact_stop_seen")
        for row in window_rows
    )
    reaction_signal = (
        max_disp >= reaction_disp_m
        or max_z_delta >= reaction_z_delta_m
        or max_speed >= reaction_speed_mps
        or (contact_evidence and max_tip >= reaction_tip_angle_deg)
    )
    overshoot = any(_flag(row, "contact_overshoot_seen") for row in window_rows) or max_disp >= overshoot_disp_m
    clip_values = [_flag(row, "clip_any") for row in window_rows if "clip_any" in row]
    clip_any_rate = sum(1 for value in clip_values if value) / len(clip_values) if clip_values else 0.0
    follow_p95_rad = _p95(_joint_follow_values(window_rows))
    raw_p95_rad = _p95(_joint_raw_values(window_rows))
    follow_p95_to_cap = follow_p95_rad / max_joint_step_rad if max_joint_step_rad > 0.0 else 0.0

    reject_reasons: list[str] = []
    if not contact_evidence:
        reject_reasons.append("no_contact_evidence_in_slice")
    if not reaction_signal:
        reject_reasons.append("no_reaction_signal_in_slice")
    if overshoot:
        reject_reasons.append("overshoot")
    accepted = not reject_reasons
    quality_tier = _tier(accepted, clip_any_rate, follow_p95_to_cap)
    return {
        "env_id": env_id,
        "anchor_step": anchor,
        "start_step": start,
        "end_step": end,
        "rows": len(window_rows),
        "accepted": accepted,
        "quality_tier": quality_tier,
        "reject_reasons": reject_reasons,
        "contact_evidence": contact_evidence,
        "reaction_signal": reaction_signal,
        "overshoot": overshoot,
        "max_disp_m": max_disp,
        "max_z_delta_m": max_z_delta,
        "max_speed_mps": max_speed,
        "max_tip_angle_deg": max_tip,
        "clip_any_rate": clip_any_rate,
        "joint_follow_p95_rad": follow_p95_rad,
        "joint_follow_p95_to_cap": follow_p95_to_cap,
        "raw_delta_p95_rad": raw_p95_rad,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reaction_window_json", type=Path, default=DEFAULT_WINDOW_AUDIT)
    parser.add_argument("--trace_csv", type=Path, default=DEFAULT_TRACE_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    window = _load_json(args.reaction_window_json)
    summary = _load_json(args.summary_json)
    thresholds = window.get("thresholds", {})
    max_joint_step_rad = _float(summary.get("max_diffik_joint_step_rad"), 0.035)

    with args.trace_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_env: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_env[_int(row.get("env_id"))].append(row)
    for env_rows in by_env.values():
        env_rows.sort(key=lambda row: _int(row.get("step")))

    accepted_windows = [
        row for row in window.get("per_window", []) if isinstance(row, dict) and bool(row.get("accepted", False))
    ]
    anchors = {int(row["env_id"]): int(row["anchor_step"]) for row in accepted_windows}

    policy_results = []
    best_policy: dict[str, Any] | None = None
    for policy_name, start_delta, end_delta in WINDOW_POLICIES:
        per_env = []
        for env_id, anchor in sorted(anchors.items()):
            per_env.append(
                _slice_summary(
                    env_id=env_id,
                    anchor=anchor,
                    rows=by_env[env_id],
                    start_delta=start_delta,
                    end_delta=end_delta,
                    max_joint_step_rad=max_joint_step_rad,
                    reaction_disp_m=_float(thresholds.get("reaction_disp_m"), 0.001),
                    reaction_z_delta_m=_float(thresholds.get("reaction_z_delta_m"), 0.002),
                    reaction_speed_mps=_float(thresholds.get("reaction_speed_mps"), 0.02),
                    reaction_tip_angle_deg=_float(thresholds.get("reaction_tip_angle_deg"), 1.0),
                    overshoot_disp_m=_float(thresholds.get("overshoot_disp_m"), 0.02),
                )
            )

        accepted = [row for row in per_env if row["accepted"]]
        tier_counts = Counter(str(row["quality_tier"]) for row in per_env)
        clip_rates = [_float(row.get("clip_any_rate")) for row in accepted]
        follow_ratios = [_float(row.get("joint_follow_p95_to_cap")) for row in accepted]
        raw_p95s = [_float(row.get("raw_delta_p95_rad")) for row in accepted]
        candidate = {
            "policy": policy_name,
            "relative_window": [start_delta, end_delta],
            "event_count": len(per_env),
            "accepted_count": len(accepted),
            "accepted_rate": len(accepted) / len(per_env) if per_env else 0.0,
            "row_count_total": sum(int(row["rows"]) for row in per_env),
            "row_count_mean": _mean([float(row["rows"]) for row in per_env]),
            "quality_tier_counts": dict(sorted(tier_counts.items())),
            "accepted_clip_any_rate_mean": _mean(clip_rates),
            "accepted_clip_any_rate_p95": _p95(clip_rates),
            "accepted_follow_p95_to_cap_mean": _mean(follow_ratios),
            "accepted_follow_p95_to_cap_p95": _p95(follow_ratios),
            "accepted_raw_delta_p95_mean_rad": _mean(raw_p95s),
            "strict_clean_count": int(tier_counts.get("A_CLEAN_DIFFIK_TEACHER", 0)),
            "follow_ok_count": int(
                tier_counts.get("A_CLEAN_DIFFIK_TEACHER", 0)
                + tier_counts.get("B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH", 0)
            ),
            "per_env": per_env,
        }
        policy_results.append(candidate)
        if best_policy is None:
            best_policy = candidate
        else:
            best_key = (
                int(best_policy["strict_clean_count"]),
                int(best_policy["follow_ok_count"]),
                float(best_policy["accepted_rate"]),
                -float(best_policy["accepted_clip_any_rate_mean"]),
            )
            cand_key = (
                int(candidate["strict_clean_count"]),
                int(candidate["follow_ok_count"]),
                float(candidate["accepted_rate"]),
                -float(candidate["accepted_clip_any_rate_mean"]),
            )
            if cand_key > best_key:
                best_policy = candidate

    assert best_policy is not None
    official = next(row for row in policy_results if row["policy"] == "official_reaction_window_m24_p48")
    revalidation_improves_action_teacher = (
        best_policy["strict_clean_count"] > official["strict_clean_count"]
        or best_policy["follow_ok_count"] > official["follow_ok_count"]
    )
    enough_clean_for_action_dataset = (
        best_policy["accepted_count"] == len(anchors)
        and best_policy["strict_clean_count"] == len(anchors)
    )
    enough_follow_ok_without_tier_c = (
        best_policy["accepted_count"] == len(anchors)
        and best_policy["follow_ok_count"] == len(anchors)
    )

    verdict = {
        "revalidation_improves_teacher_quality": revalidation_improves_action_teacher,
        "enough_clean_for_action_dataset": enough_clean_for_action_dataset,
        "enough_follow_ok_without_tier_c": enough_follow_ok_without_tier_c,
        "quality_blocker_is_window_definition_only": enough_clean_for_action_dataset,
        "quality_blocker_likely_control_tracking": not enough_clean_for_action_dataset,
        "next_default": "do_not_build_action_dataset_improve_or_retest_controller_quality",
    }

    result = {
        "artifact_type": "cube10cm_teacher_quality_revalidation_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_revalidation_only": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "inputs": {
            "reaction_window_json": str(args.reaction_window_json),
            "trace_csv": str(args.trace_csv),
            "summary_json": str(args.summary_json),
        },
        "max_joint_step_rad": max_joint_step_rad,
        "event_count": len(anchors),
        "official_policy": {
            key: value for key, value in official.items() if key != "per_env"
        },
        "best_policy": {
            key: value for key, value in best_policy.items() if key != "per_env"
        },
        "policy_results": policy_results,
        "verdict": verdict,
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_teacher_quality_revalidation_audit_v1 "
        "local_revalidation_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO",
        (
            "line2 official_window "
            f"policy={official['policy']} accepted={official['accepted_count']}/{official['event_count']} "
            f"tiers={official['quality_tier_counts']} clip_mean={official['accepted_clip_any_rate_mean']:.9f} "
            f"follow_p95_to_cap={official['accepted_follow_p95_to_cap_p95']:.9f}"
        ),
        (
            "line3 best_trimmed_window "
            f"policy={best_policy['policy']} rel={best_policy['relative_window']} "
            f"accepted={best_policy['accepted_count']}/{best_policy['event_count']} "
            f"tiers={best_policy['quality_tier_counts']} clip_mean={best_policy['accepted_clip_any_rate_mean']:.9f} "
            f"follow_p95_to_cap={best_policy['accepted_follow_p95_to_cap_p95']:.9f}"
        ),
        (
            "line4 improvement "
            f"revalidation_improves_teacher_quality={revalidation_improves_action_teacher} "
            f"strict_clean_count={best_policy['strict_clean_count']} "
            f"follow_ok_count={best_policy['follow_ok_count']} "
            f"enough_clean_for_action_dataset={enough_clean_for_action_dataset}"
        ),
        (
            "line5 verdict "
            f"quality_blocker_is_window_definition_only={verdict['quality_blocker_is_window_definition_only']} "
            f"quality_blocker_likely_control_tracking={verdict['quality_blocker_likely_control_tracking']}"
        ),
        (
            "line6 next_default "
            "do_not_build_action_dataset_improve_or_retest_controller_quality "
            "no_gpu_without_explicit_single_candidate_approval"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
