"""Build and audit reaction windows for professor cube10cm tap/reaction traces.

This is a local posthoc tool. It reads existing trace/summary logs only. It may
write a small window CSV audit artifact, but it does not run IsaacLab, train,
generate new rollouts, or create a final training dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def _md5(path: Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.md5()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    return int(round(_float(value, float(default))))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _row_flag(row: dict[str, str], key: str) -> bool:
    return _boolish(row.get(key, "0"))


def _rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(math.ceil(0.95 * len(sorted_values))) - 1))
    return sorted_values[index]


def _resolve_trace_csv(summary_json: Path | None, summary: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    raw_path = summary.get("trace_csv")
    if not raw_path:
        raise ValueError("--trace_csv is required because summary_json does not contain trace_csv")
    candidate = Path(str(raw_path))
    if candidate.exists():
        return candidate
    if summary_json is not None:
        relative = summary_json.parent / candidate.name
        if relative.exists():
            return relative
    return candidate


def _group_by_env(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by_env: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_env[_int(row.get("env_id"))].append(row)
    for env_rows in by_env.values():
        env_rows.sort(key=lambda row: _int(row.get("step")))
    return dict(sorted(by_env.items()))


def _first_nonnegative_step(env_rows: list[dict[str, str]], key: str) -> int | None:
    values = sorted({_int(row.get(key), -1) for row in env_rows if _int(row.get(key), -1) >= 0})
    return values[0] if values else None


def _anchor_step(env_rows: list[dict[str, str]]) -> tuple[int | None, str]:
    first_contact = _first_nonnegative_step(env_rows, "first_contact_step")
    if first_contact is not None:
        return first_contact, "first_contact_step"
    first_stop = _first_nonnegative_step(env_rows, "first_stop_step")
    if first_stop is not None:
        return first_stop, "first_stop_step"
    for row in env_rows:
        if _row_flag(row, "measured_contact_now"):
            return _int(row.get("step")), "measured_contact_now"
    for row in env_rows:
        if _row_flag(row, "measured_contact_seen") or _row_flag(row, "contact_stop_seen"):
            return _int(row.get("step")), "contact_seen_row"
    return None, "missing_contact_anchor"


def _summary_no_posewrite(summary: dict[str, Any], max_posewrite_calls: int) -> bool:
    return (
        int(summary.get("posewrite_calls_during_rollout", -1)) <= max_posewrite_calls
        and not _boolish(summary.get("rollout_object_posewrite"))
        and not _boolish(summary.get("training"))
        and not _boolish(summary.get("dataset_generation"))
        and not _boolish(summary.get("grasp_attach"))
    )


def _required_columns(rows: list[dict[str, str]]) -> list[str]:
    required = [
        "step",
        "env_id",
        "push_dx",
        "push_dy",
        "cube_x_m",
        "cube_y_m",
        "cube_z_m",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_z_m",
        "target_x_m",
        "target_y_m",
        "target_z_m",
        "measured_contact_seen",
        "contact_stop_seen",
        "first_contact_step",
        "disp_along_push_m",
        "cube_speed_mps",
        "tip_angle_deg",
    ]
    return [key for key in required if key not in rows[0]]


def _joint_follow_values(rows: list[dict[str, str]]) -> list[float]:
    if not rows:
        return []
    keys = [key for key in rows[0] if key.startswith("joint_follow_err_") and key.endswith("_rad")]
    values: list[float] = []
    for row in rows:
        values.extend(abs(_float(row.get(key))) for key in keys)
    return values


def _quality_tier(
    *,
    accepted: bool,
    clip_any_rate: float,
    follow_p95_to_cap: float,
    teacher_max_window_clip_rate: float,
    teacher_max_follow_p95_to_cap: float,
) -> str:
    if not accepted:
        return "REJECTED"
    clip_clean = clip_any_rate <= teacher_max_window_clip_rate
    follow_clean = follow_p95_to_cap <= teacher_max_follow_p95_to_cap
    if clip_clean and follow_clean:
        return "A_CLEAN_DIFFIK_TEACHER"
    if follow_clean:
        return "B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH"
    return "C_REACTION_VALID_FOLLOW_LAG"


def _window_summary(
    env_id: int,
    env_rows: list[dict[str, str]],
    *,
    controller_ok: bool,
    no_posewrite: bool,
    pre_steps: int,
    post_steps: int,
    reaction_disp_m: float,
    reaction_z_delta_m: float,
    reaction_speed_mps: float,
    reaction_tip_angle_deg: float,
    overshoot_disp_m: float,
    max_joint_step_rad: float,
    teacher_max_window_clip_rate: float,
    teacher_max_follow_p95_to_cap: float,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    anchor, anchor_source = _anchor_step(env_rows)
    if anchor is None:
        return {
            "env_id": env_id,
            "anchor_step": None,
            "anchor_source": anchor_source,
            "accepted": False,
            "quality_tier": "REJECTED",
            "reject_reasons": ["missing_contact_anchor"],
        }, []

    start_step = anchor - int(pre_steps)
    end_step = anchor + int(post_steps)
    window_rows = [
        row for row in env_rows if start_step <= _int(row.get("step")) <= end_step
    ]
    if not window_rows:
        return {
            "env_id": env_id,
            "anchor_step": anchor,
            "anchor_source": anchor_source,
            "window_start_step": start_step,
            "window_end_step": end_step,
            "accepted": False,
            "quality_tier": "REJECTED",
            "reject_reasons": ["empty_window"],
        }, []

    z0 = _float(env_rows[0].get("cube_z_m"))
    max_disp = max(_float(row.get("disp_along_push_m")) for row in window_rows)
    max_z_delta = max(_float(row.get("cube_z_m")) - z0 for row in window_rows)
    max_speed = max(_float(row.get("cube_speed_mps")) for row in window_rows)
    max_tip = max(_float(row.get("tip_angle_deg")) for row in window_rows)
    contact_evidence = any(
        _row_flag(row, "measured_contact_now")
        or _row_flag(row, "measured_contact_seen")
        or _row_flag(row, "contact_stop_seen")
        for row in window_rows
    )
    disp_reaction = max_disp >= reaction_disp_m
    z_reaction = max_z_delta >= reaction_z_delta_m
    speed_reaction = max_speed >= reaction_speed_mps
    tip_reaction = contact_evidence and max_tip >= reaction_tip_angle_deg
    reaction_signal = disp_reaction or z_reaction or speed_reaction or tip_reaction
    overshoot = any(_row_flag(row, "contact_overshoot_seen") for row in window_rows) or max_disp >= overshoot_disp_m
    clip_any_rate = _rate([_row_flag(row, "clip_any") for row in window_rows if "clip_any" in row])
    follow_values = _joint_follow_values(window_rows)
    follow_p95_rad = _p95(follow_values)
    follow_p95_to_cap = follow_p95_rad / max_joint_step_rad if max_joint_step_rad > 0.0 else 0.0

    reject_reasons: list[str] = []
    if not controller_ok:
        reject_reasons.append("controller_not_diffik")
    if not no_posewrite:
        reject_reasons.append("posewrite_training_or_attach")
    if not contact_evidence:
        reject_reasons.append("no_contact_evidence")
    if not reaction_signal:
        reject_reasons.append("no_reaction_signal")
    if overshoot:
        reject_reasons.append("overshoot")

    accepted = not reject_reasons
    quality_tier = _quality_tier(
        accepted=accepted,
        clip_any_rate=clip_any_rate,
        follow_p95_to_cap=follow_p95_to_cap,
        teacher_max_window_clip_rate=teacher_max_window_clip_rate,
        teacher_max_follow_p95_to_cap=teacher_max_follow_p95_to_cap,
    )
    summary: dict[str, Any] = {
        "env_id": env_id,
        "anchor_step": anchor,
        "anchor_source": anchor_source,
        "window_start_step": start_step,
        "window_end_step": end_step,
        "rows": len(window_rows),
        "accepted": accepted,
        "quality_tier": quality_tier,
        "reject_reasons": reject_reasons,
        "contact_evidence": contact_evidence,
        "reaction_signal": reaction_signal,
        "disp_reaction": disp_reaction,
        "z_reaction": z_reaction,
        "speed_reaction": speed_reaction,
        "contact_gated_tip_reaction": tip_reaction,
        "overshoot": overshoot,
        "max_disp_m": max_disp,
        "max_z_delta_m": max_z_delta,
        "max_speed_mps": max_speed,
        "max_tip_angle_deg": max_tip,
        "clip_any_rate": clip_any_rate,
        "joint_follow_p95_rad": follow_p95_rad,
        "joint_follow_p95_to_cap": follow_p95_to_cap,
    }
    return summary, window_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--trace_csv", type=Path, default=None)
    parser.add_argument("--out_window_csv", type=Path, default=None)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--pre_contact_steps", type=int, default=24)
    parser.add_argument("--post_contact_steps", type=int, default=48)
    parser.add_argument("--reaction_disp_m", type=float, default=0.001)
    parser.add_argument("--reaction_z_delta_m", type=float, default=0.002)
    parser.add_argument("--reaction_speed_mps", type=float, default=0.020)
    parser.add_argument("--reaction_tip_angle_deg", type=float, default=1.0)
    parser.add_argument("--overshoot_disp_m", type=float, default=0.020)
    parser.add_argument("--min_window_acceptance_rate", type=float, default=1.0)
    parser.add_argument("--teacher_max_window_clip_rate", type=float, default=0.50)
    parser.add_argument("--teacher_max_follow_p95_to_cap", type=float, default=1.0)
    parser.add_argument("--max_posewrite_calls", type=int, default=0)
    parser.add_argument("--allow_contract_fail", action="store_true")
    args = parser.parse_args()

    if args.pre_contact_steps < 0 or args.post_contact_steps < 0:
        raise ValueError("pre/post contact steps must be non-negative")
    if not (0.0 <= args.min_window_acceptance_rate <= 1.0):
        raise ValueError("--min_window_acceptance_rate must be in [0, 1]")
    for name in [
        "reaction_disp_m",
        "reaction_z_delta_m",
        "reaction_speed_mps",
        "reaction_tip_angle_deg",
    ]:
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    if args.overshoot_disp_m <= 0.0:
        raise ValueError("--overshoot_disp_m must be positive")

    summary = _load_json(args.summary_json)
    trace_csv = _resolve_trace_csv(args.summary_json, summary, args.trace_csv)
    rows = _load_csv(trace_csv)
    missing = _required_columns(rows)
    if missing:
        raise ValueError(f"trace missing required columns: {missing}")

    controller_ok = summary.get("controller") == "IsaacLab_DifferentialIKController"
    no_posewrite = _summary_no_posewrite(summary, int(args.max_posewrite_calls))
    max_joint_step_rad = _float(summary.get("max_diffik_joint_step_rad"))
    by_env = _group_by_env(rows)

    window_summaries: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, str | int | float]] = []
    for env_id, env_rows in by_env.items():
        window, window_rows = _window_summary(
            env_id,
            env_rows,
            controller_ok=controller_ok,
            no_posewrite=no_posewrite,
            pre_steps=int(args.pre_contact_steps),
            post_steps=int(args.post_contact_steps),
            reaction_disp_m=float(args.reaction_disp_m),
            reaction_z_delta_m=float(args.reaction_z_delta_m),
            reaction_speed_mps=float(args.reaction_speed_mps),
            reaction_tip_angle_deg=float(args.reaction_tip_angle_deg),
            overshoot_disp_m=float(args.overshoot_disp_m),
            max_joint_step_rad=max_joint_step_rad,
            teacher_max_window_clip_rate=float(args.teacher_max_window_clip_rate),
            teacher_max_follow_p95_to_cap=float(args.teacher_max_follow_p95_to_cap),
        )
        window_summaries.append(window)
        if window.get("accepted"):
            for local_index, row in enumerate(window_rows):
                annotated: dict[str, str | int | float] = dict(row)
                annotated.update(
                    {
                        "reaction_window_id": int(env_id),
                        "reaction_window_local_index": local_index,
                        "reaction_window_anchor_step": int(window["anchor_step"]),
                        "reaction_window_start_step": int(window["window_start_step"]),
                        "reaction_window_end_step": int(window["window_end_step"]),
                        "reaction_window_contract_pass": 1,
                        "reaction_window_contact_evidence": int(bool(window["contact_evidence"])),
                        "reaction_window_reaction_signal": int(bool(window["reaction_signal"])),
                        "reaction_window_overshoot": int(bool(window["overshoot"])),
                        "reaction_window_quality_tier": str(window["quality_tier"]),
                    }
                )
                accepted_rows.append(annotated)

    accepted_windows = [window for window in window_summaries if bool(window.get("accepted"))]
    rejected_windows = [window for window in window_summaries if not bool(window.get("accepted"))]
    acceptance_rate = len(accepted_windows) / len(window_summaries) if window_summaries else 0.0
    reject_counter: Counter[str] = Counter()
    for window in rejected_windows:
        reject_counter.update(str(reason) for reason in window.get("reject_reasons", []))
    quality_tier_counts = Counter(str(window.get("quality_tier", "UNKNOWN")) for window in window_summaries)

    all_window_clip_rates = [_float(window.get("clip_any_rate")) for window in accepted_windows]
    all_follow_ratios = [_float(window.get("joint_follow_p95_to_cap")) for window in accepted_windows]
    reaction_window_contract_pass = (
        controller_ok
        and no_posewrite
        and acceptance_rate >= float(args.min_window_acceptance_rate)
        and len(accepted_rows) > 0
    )
    clean_diffik_teacher_window_ready = (
        reaction_window_contract_pass
        and _mean(all_window_clip_rates) <= float(args.teacher_max_window_clip_rate)
        and _p95(all_follow_ratios) <= float(args.teacher_max_follow_p95_to_cap)
    )

    audit: dict[str, Any] = {
        "artifact_type": "cube10cm_reaction_window_contract_audit_v2",
        "branch": "professor_cube10cm_tap_reaction",
        "primary_objective": "contact_reaction_not_final_1cm",
        "summary_json": str(args.summary_json) if args.summary_json is not None else None,
        "summary_json_md5": _md5(args.summary_json),
        "trace_csv": str(trace_csv),
        "trace_csv_md5": _md5(trace_csv),
        "out_window_csv": str(args.out_window_csv) if args.out_window_csv is not None else None,
        "controller_ok": controller_ok,
        "no_posewrite": no_posewrite,
        "source_trace_rows": len(rows),
        "source_env_count": len(by_env),
        "candidate_window_count": len(window_summaries),
        "accepted_window_count": len(accepted_windows),
        "rejected_window_count": len(rejected_windows),
        "accepted_window_row_count": len(accepted_rows),
        "window_acceptance_rate": acceptance_rate,
        "reaction_window_contract_pass": reaction_window_contract_pass,
        "clean_diffik_teacher_window_ready": clean_diffik_teacher_window_ready,
        "accepted_window_clip_any_rate_mean": _mean(all_window_clip_rates),
        "accepted_window_clip_any_rate_p95": _p95(all_window_clip_rates),
        "accepted_window_follow_p95_to_cap_mean": _mean(all_follow_ratios),
        "accepted_window_follow_p95_to_cap_p95": _p95(all_follow_ratios),
        "quality_tier_counts": dict(sorted(quality_tier_counts.items())),
        "rejected_window_reasons": dict(sorted(reject_counter.items())),
        "contract": {
            "window_anchor_priority": [
                "first_contact_step",
                "first_stop_step",
                "measured_contact_now",
                "contact_seen_row",
            ],
            "pre_contact_steps": int(args.pre_contact_steps),
            "post_contact_steps": int(args.post_contact_steps),
            "requires": [
                "IsaacLab_DifferentialIKController",
                "no_posewrite_training_dataset_generation_or_attach",
                "contact_evidence",
                "reaction_signal",
                "no_overshoot",
            ],
            "reaction_signal_any_of": [
                "max_disp_m >= reaction_disp_m",
                "max_z_delta_m >= reaction_z_delta_m",
                "max_speed_mps >= reaction_speed_mps",
                "contact_evidence AND max_tip_angle_deg >= reaction_tip_angle_deg",
            ],
            "final_1cm_relocation_required": False,
            "quality_metadata_not_default_reject": [
                "clip_any_rate",
                "joint_follow_p95_to_cap",
            ],
            "quality_tiers": {
                "A_CLEAN_DIFFIK_TEACHER": "accepted and clip/follow pass clean teacher thresholds",
                "B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH": "accepted with follow under threshold but clip above threshold",
                "C_REACTION_VALID_FOLLOW_LAG": "accepted with follow p95/cap above threshold",
                "REJECTED": "not a valid reaction window",
            },
        },
        "thresholds": {
            "reaction_disp_m": float(args.reaction_disp_m),
            "reaction_z_delta_m": float(args.reaction_z_delta_m),
            "reaction_speed_mps": float(args.reaction_speed_mps),
            "reaction_tip_angle_deg": float(args.reaction_tip_angle_deg),
            "overshoot_disp_m": float(args.overshoot_disp_m),
            "min_window_acceptance_rate": float(args.min_window_acceptance_rate),
            "teacher_max_window_clip_rate": float(args.teacher_max_window_clip_rate),
            "teacher_max_follow_p95_to_cap": float(args.teacher_max_follow_p95_to_cap),
        },
        "per_window": window_summaries,
    }

    if args.out_window_csv is not None:
        args.out_window_csv.parent.mkdir(parents=True, exist_ok=True)
        annotation_fields = [
            "reaction_window_id",
            "reaction_window_local_index",
            "reaction_window_anchor_step",
            "reaction_window_start_step",
            "reaction_window_end_step",
            "reaction_window_contract_pass",
            "reaction_window_contact_evidence",
            "reaction_window_reaction_signal",
            "reaction_window_overshoot",
            "reaction_window_quality_tier",
        ]
        if accepted_rows:
            fieldnames = list(accepted_rows[0].keys())
        else:
            fieldnames = list(rows[0].keys()) + annotation_fields
        with args.out_window_csv.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(accepted_rows)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verdict = "PASS" if reaction_window_contract_pass else "FAIL"
    teacher_verdict = "READY" if clean_diffik_teacher_window_ready else "NOT_READY"
    print(
        "reaction_window_contract line1 "
        f"verdict={verdict} envs={len(by_env)} accepted_windows={len(accepted_windows)} "
        f"acceptance_rate={acceptance_rate:.9f} rows={len(accepted_rows)}"
    )
    print(
        "reaction_window_contract line2 "
        f"controller_ok={controller_ok} no_posewrite={no_posewrite} "
        f"reject_reasons={dict(sorted(reject_counter.items()))}"
    )
    print(
        "reaction_window_contract line3 "
        f"clean_diffik_teacher_window={teacher_verdict} "
        f"clip_mean={_mean(all_window_clip_rates):.9f} "
        f"follow_p95_to_cap_p95={_p95(all_follow_ratios):.9f}"
    )
    print(f"reaction_window_contract line4 quality_tier_counts={dict(sorted(quality_tier_counts.items()))}")
    if args.out_window_csv is not None:
        print(f"reaction_window_contract line5 out_window_csv={args.out_window_csv}")
    if args.out_json is not None:
        print(f"reaction_window_contract line6 out_json={args.out_json}")
    if reaction_window_contract_pass or args.allow_contract_fail:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
