#!/usr/bin/env python3
"""Posthoc TensorBoard scalar gate for cube10cm PPO smoke/ladder runs.

This script does not run PPO. It reads an existing TensorBoard event log,
summarizes reward/loss/policy/task scalars, and writes a promotion-gate verdict.
Use it together with the live TensorBoard dashboard before considering any
longer PPO run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]

CORE_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Policy/mean_noise_std",
]

TASK_TAGS = [
    "Episode/cube_push_disp_along_m",
    "Episode/cube_push_disp_xy_m",
    "Episode/cube_push_target_xy_dist_m",
    "Episode/cube_push_tcp_cube_dist_m",
    "Episode/cube_push_controlled_rate",
    "Episode/cube_push_impact_rate",
    "Episode/cube_push_low_motion_rate",
    "Episode/cube_push_success_rate",
    "Episode/cube_push_joint_delta_abs_mean",
    "Episode/cube_push_joint_delta_abs_max",
    "Episode/cube_push_joint_delta_cap_rate",
    "Episode/cube_push_action_abs_mean",
    "Episode/cube_push_action_abs_max",
    "Episode/cube_push_target_lead_limit_rate",
    "Episode/cube_push_bc_teacher_blend_mean",
    "Episode/cube_push_bc_teacher_imitation_mse",
    "Episode/cube_push_bc_teacher_action_abs_mean",
    "Episode/cube_push_d256_reset_active_rate",
    "Episode/cube_push_d256_reset_episode_index_mean",
    "Episode/cube_tap_bc_teacher_blend_mean",
    "Episode/cube_tap_bc_teacher_imitation_mse",
    "Episode/cube_tap_bc_teacher_action_abs_mean",
    "Episode/cube_tap_d256_reset_active_rate",
    "Episode/cube_tap_d256_reset_episode_index_mean",
    "Episode/bc_teacher_imitation_penalty",
    "Episode/cube_tap_contact_seen_rate",
    "Episode/cube_tap_contact_proxy_rate",
    "Episode/cube_tap_reaction_seen_rate",
    "Episode/cube_tap_reaction_signal_now_rate",
    "Episode/cube_tap_reaction_contact_context_rate",
    "Episode/cube_tap_reaction_now_rate",
    "Episode/cube_tap_contact_reaction_seen_rate",
    "Episode/cube_tap_useful_now_rate",
    "Episode/cube_tap_useful_seen_rate",
    "Episode/cube_tap_success_rate",
    "Episode/cube_tap_no_overshoot_seen_rate",
    "Episode/cube_tap_overshoot_now_rate",
    "Episode/cube_tap_overshoot_seen_rate",
    "Episode/cube_tap_max_disp_along_m",
    "Episode/cube_tap_max_disp_xy_m",
    "Episode/cube_tap_max_disp_along_ge_1mm_rate",
    "Episode/cube_tap_max_disp_xy_ge_1mm_rate",
    "Episode/cube_tap_max_disp_along_ge_3mm_rate",
    "Episode/cube_tap_max_disp_xy_ge_3mm_rate",
    "Episode/cube_tap_contact_face_gap_m",
    "Episode/cube_tap_contact_lateral_m",
    "Episode/cube_tap_contact_vertical_offset_m",
    "Episode/cube_tap_min_contact_vertical_offset_m",
    "Episode/cube_tap_min_contact_vertical_finite_rate",
    "Episode/cube_tap_stop_after_useful_hold_rate",
    "Episode/cube_tap_stop_after_disp_hold_rate",
    "Episode/cube_tap_stop_after_disp_m",
    "CollectionFinal/cube_tap_contact_seen_rate",
    "CollectionFinal/cube_tap_reaction_seen_rate",
    "CollectionFinal/cube_tap_contact_reaction_seen_rate",
    "CollectionFinal/cube_tap_useful_seen_rate",
    "CollectionFinal/cube_tap_success_rate",
    "CollectionFinal/cube_tap_overshoot_seen_rate",
    "CollectionFinal/cube_tap_max_disp_along_m",
    "CollectionFinal/cube_tap_max_disp_xy_m",
    "CollectionFinal/cube_tap_max_disp_along_max_m",
    "CollectionFinal/cube_tap_max_disp_xy_max_m",
    "CollectionFinal/cube_tap_max_disp_along_ge_1mm_rate",
    "CollectionFinal/cube_tap_max_disp_xy_ge_1mm_rate",
    "CollectionFinal/cube_tap_max_disp_along_ge_3mm_rate",
    "CollectionFinal/cube_tap_max_disp_xy_ge_3mm_rate",
    "CollectionFinal/cube_tap_d256_reset_active_rate",
    "CollectionFinal/cube_push_joint_delta_cap_rate",
]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _stat(values: list[Any]) -> dict[str, Any]:
    floats = [float(v.value) for v in values]
    steps = [int(v.step) for v in values]
    last_window = floats[-min(3, len(floats)) :]
    return {
        "n": len(floats),
        "first_step": steps[0],
        "last_step": steps[-1],
        "first": floats[0],
        "last": floats[-1],
        "min": min(floats),
        "max": max(floats),
        "delta": floats[-1] - floats[0],
        "last3_mean": sum(last_window) / len(last_window),
    }


def _load_scalars(log_dir: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    from tensorboard.backend.event_processing import event_accumulator

    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event file under {log_dir}")

    scalars: dict[str, dict[str, Any]] = {}
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            values = ea.Scalars(tag)
            if values:
                scalars[tag] = _stat(values)
    return [str(p) for p in event_files], scalars


def _get(scalars: dict[str, dict[str, Any]], tag: str, key: str = "last") -> float | None:
    item = scalars.get(tag)
    if not item:
        return None
    value = item.get(key)
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _max_existing(scalars: dict[str, dict[str, Any]], tags: list[str], key: str = "max") -> float | None:
    values = [_get(scalars, tag, key) for tag in tags]
    finite_values = [float(v) for v in values if _finite(v)]
    if not finite_values:
        return None
    return max(finite_values)


def _has_any(scalars: dict[str, dict[str, Any]], tags: list[str]) -> bool:
    return any(tag in scalars for tag in tags)


def _detect_env_kind(args: argparse.Namespace, scalars: dict[str, dict[str, Any]]) -> str:
    if args.env_kind != "auto":
        return str(args.env_kind)
    if any(tag.startswith("Episode/cube_tap_") for tag in scalars):
        return "tap10cm"
    return "push3cm"


def _metric_line(tag: str, scalars: dict[str, dict[str, Any]]) -> str:
    item = scalars.get(tag)
    if not item:
        return f"- `{tag}`: missing"
    return (
        f"- `{tag}`: n `{item['n']}`, first `{item['first']}`, "
        f"last `{item['last']}`, min `{item['min']}`, max `{item['max']}`"
    )


def _gate(args: argparse.Namespace, scalars: dict[str, dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    env_kind = _detect_env_kind(args, scalars)

    train_episode_tags = {"Train/mean_reward", "Train/mean_episode_length"}
    missing_core = [tag for tag in CORE_TAGS if tag not in scalars]
    if bool(args.allow_missing_train_episode_scalars):
        missing_train = [tag for tag in missing_core if tag in train_episode_tags]
        if missing_train:
            warnings.append(
                "missing Train episode scalars allowed for no-termination gate: "
                f"{missing_train}"
            )
        missing_core = [tag for tag in missing_core if tag not in train_episode_tags]
    if missing_core:
        issues.append(f"missing core TensorBoard scalars: {missing_core}")

    train_iters = int(scalars.get("Train/mean_reward", {}).get("n", 0))
    if train_iters < int(args.min_iterations_for_promotion):
        warnings.append(
            f"short run: Train/mean_reward has {train_iters} points, "
            f"promotion gate expects at least {args.min_iterations_for_promotion}"
        )

    push_success = _get(scalars, "Episode/cube_push_success_rate", "max")
    tap_contact_tags = [
        "Episode/cube_tap_contact_proxy_rate",
        "Episode/cube_tap_contact_seen_rate",
        "Episode/cube_tap_reaction_seen_rate",
        "Episode/cube_tap_contact_reaction_seen_rate",
        "Episode/cube_tap_useful_seen_rate",
        "Episode/cube_tap_success_rate",
    ]
    tap_success = _get(scalars, "Episode/cube_tap_success_rate", "max")
    tap_contact = _max_existing(scalars, tap_contact_tags, "max")
    tap_useful = _max_existing(
        scalars,
        ["Episode/cube_tap_useful_seen_rate", "Episode/cube_tap_success_rate"],
        "max",
    )
    controlled = _get(scalars, "Episode/cube_push_controlled_rate", "last")
    low_motion = _get(scalars, "Episode/cube_push_low_motion_rate", "last")
    tcp_dist = _get(scalars, "Episode/cube_push_tcp_cube_dist_m", "last")
    disp_along = _get(scalars, "Episode/cube_push_disp_along_m", "last")

    if env_kind == "tap10cm":
        if not _has_any(scalars, tap_contact_tags):
            issues.append("missing tap task scalars for tap10cm TensorBoard gate")
        elif _finite(tap_contact) and float(tap_contact) < float(args.min_success_or_contact_rate):
            issues.append(
                "tap contact/reaction signal below threshold in TensorBoard "
                f"(max={tap_contact}, threshold={args.min_success_or_contact_rate})"
            )
        if _finite(tap_useful) and float(tap_useful) < float(args.min_tap_useful_seen_rate):
            issues.append(
                "tap useful/success signal below threshold in TensorBoard "
                f"(max={tap_useful}, threshold={args.min_tap_useful_seen_rate})"
            )
    else:
        success_like = max([v for v in (push_success, tap_success, tap_contact) if _finite(v)], default=None)
        if _finite(success_like) and float(success_like) < float(args.min_success_or_contact_rate):
            issues.append(
                "task success/contact signal below threshold in TensorBoard "
                f"(max={success_like}, threshold={args.min_success_or_contact_rate})"
            )

    if _finite(low_motion) and float(low_motion) > float(args.max_low_motion_rate):
        issues.append(f"low-motion rate too high: last={low_motion}")
    if _finite(tcp_dist) and float(tcp_dist) > float(args.max_tcp_cube_dist_m):
        if env_kind == "tap10cm":
            warnings.append(f"raw TCP-cube distance is high for tap/AABB diagnostic: last={tcp_dist}")
        else:
            warnings.append(f"TCP-cube distance remains high: last={tcp_dist}")
    if _finite(disp_along) and float(disp_along) < float(args.min_disp_along_m):
        warnings.append(f"disp_along remains too small: last={disp_along}")
    if _finite(controlled) and float(controlled) < float(args.min_controlled_rate):
        warnings.append(f"controlled rate remains low: last={controlled}")
    if env_kind == "tap10cm":
        tap_disp = _get(scalars, "Episode/cube_tap_max_disp_along_m", "max")
        tap_vertical_last = _get(scalars, "Episode/cube_tap_contact_vertical_offset_m", "last")
        tap_vertical_min_contact = _get(scalars, "Episode/cube_tap_min_contact_vertical_offset_m", "last")
        if args.tap_vertical_gate_mode == "min_contact" and _finite(tap_vertical_min_contact):
            tap_vertical = tap_vertical_min_contact
        else:
            tap_vertical = tap_vertical_last
            if args.tap_vertical_gate_mode == "min_contact" and not _finite(tap_vertical_min_contact):
                warnings.append("min-contact vertical scalar missing; fell back to last-frame vertical")
        tap_overshoot = _get(scalars, "Episode/cube_tap_overshoot_seen_rate", "max")
        if _finite(tap_disp) and float(tap_disp) < float(args.min_tap_max_disp_along_m):
            msg = f"tap max displacement remains small: max={tap_disp}"
            if bool(args.require_tap_displacement_gate):
                issues.append(msg)
            else:
                warnings.append(msg)
        elif bool(args.require_tap_displacement_gate) and not _finite(tap_disp):
            issues.append("tap max displacement scalar is missing for required displacement gate")
        tap_disp_along_1mm_rate = _get(scalars, "Episode/cube_tap_max_disp_along_ge_1mm_rate", "max")
        tap_disp_xy_1mm_rate = _get(scalars, "Episode/cube_tap_max_disp_xy_ge_1mm_rate", "max")
        if float(args.min_tap_disp_along_ge_1mm_rate) > 0.0:
            if _finite(tap_disp_along_1mm_rate):
                if float(tap_disp_along_1mm_rate) < float(args.min_tap_disp_along_ge_1mm_rate):
                    issues.append(
                        "tap along displacement 1mm rate below threshold: "
                        f"max={tap_disp_along_1mm_rate}"
                    )
            else:
                issues.append("tap along displacement 1mm rate scalar is missing")
        if float(args.min_tap_disp_xy_ge_1mm_rate) > 0.0:
            if _finite(tap_disp_xy_1mm_rate):
                if float(tap_disp_xy_1mm_rate) < float(args.min_tap_disp_xy_ge_1mm_rate):
                    issues.append(
                        "tap XY displacement 1mm rate below threshold: "
                        f"max={tap_disp_xy_1mm_rate}"
                    )
            else:
                issues.append("tap XY displacement 1mm rate scalar is missing")
        if _finite(tap_vertical) and float(tap_vertical) > float(args.max_tap_contact_vertical_offset_m):
            warnings.append(
                f"tap contact vertical offset remains high: mode={args.tap_vertical_gate_mode} value={tap_vertical}"
            )
        if _finite(tap_overshoot) and float(tap_overshoot) > float(args.max_tap_overshoot_seen_rate):
            issues.append(f"tap overshoot seen rate too high: max={tap_overshoot}")

        if bool(args.require_collection_final_tap_gate):
            final_contact_tags = [
                "CollectionFinal/cube_tap_contact_seen_rate",
                "CollectionFinal/cube_tap_reaction_seen_rate",
                "CollectionFinal/cube_tap_contact_reaction_seen_rate",
            ]
            final_useful_tags = [
                "CollectionFinal/cube_tap_useful_seen_rate",
                "CollectionFinal/cube_tap_success_rate",
            ]
            if not _has_any(scalars, final_contact_tags + final_useful_tags):
                issues.append("collection-final tap scalars are missing")
            final_contact = _max_existing(scalars, final_contact_tags, "last")
            final_useful = _get(scalars, "CollectionFinal/cube_tap_useful_seen_rate", "last")
            final_success = _get(scalars, "CollectionFinal/cube_tap_success_rate", "last")
            final_overshoot = _get(scalars, "CollectionFinal/cube_tap_overshoot_seen_rate", "last")
            final_xy_1mm = _get(scalars, "CollectionFinal/cube_tap_max_disp_xy_ge_1mm_rate", "last")
            final_d256_reset = _get(scalars, "CollectionFinal/cube_tap_d256_reset_active_rate", "last")
            final_joint_cap = _get(scalars, "CollectionFinal/cube_push_joint_delta_cap_rate", "last")
            if _finite(final_contact) and float(final_contact) < float(args.min_collection_final_success_or_contact_rate):
                issues.append(
                    "collection-final contact/reaction below threshold: "
                    f"last={final_contact}, threshold={args.min_collection_final_success_or_contact_rate}"
                )
            if _finite(final_useful) and float(final_useful) < float(args.min_collection_final_tap_useful_seen_rate):
                issues.append(
                    "collection-final useful below threshold: "
                    f"last={final_useful}, threshold={args.min_collection_final_tap_useful_seen_rate}"
                )
            if _finite(final_success) and float(args.min_collection_final_tap_success_rate) > 0.0:
                if float(final_success) < float(args.min_collection_final_tap_success_rate):
                    issues.append(
                        "collection-final success below threshold: "
                        f"last={final_success}, threshold={args.min_collection_final_tap_success_rate}"
                    )
            if _finite(final_overshoot) and float(final_overshoot) > float(args.max_collection_final_tap_overshoot_seen_rate):
                issues.append(
                    "collection-final overshoot above threshold: "
                    f"last={final_overshoot}, threshold={args.max_collection_final_tap_overshoot_seen_rate}"
                )
            if float(args.min_collection_final_tap_disp_xy_ge_1mm_rate) > 0.0:
                if _finite(final_xy_1mm):
                    if float(final_xy_1mm) < float(args.min_collection_final_tap_disp_xy_ge_1mm_rate):
                        issues.append(
                            "collection-final XY displacement 1mm rate below threshold: "
                            f"last={final_xy_1mm}, threshold={args.min_collection_final_tap_disp_xy_ge_1mm_rate}"
                        )
                else:
                    issues.append("collection-final XY displacement 1mm scalar is missing")
            if args.expect_d256_reset and _finite(final_d256_reset) and float(final_d256_reset) < 0.99:
                issues.append(f"collection-final D256 reset hook is not active for all envs: last={final_d256_reset}")
            if _finite(final_joint_cap) and float(final_joint_cap) > float(args.max_joint_delta_cap_rate):
                issues.append(f"collection-final joint-delta cap rate too high: last={final_joint_cap}")

    joint_cap = _get(scalars, "Episode/cube_push_joint_delta_cap_rate", "max")
    target_lead_cap = _get(scalars, "Episode/cube_push_target_lead_limit_rate", "max")
    if _finite(joint_cap) and float(joint_cap) > float(args.max_joint_delta_cap_rate):
        issues.append(f"joint-delta cap rate too high: max={joint_cap}")
    if _finite(target_lead_cap) and float(target_lead_cap) > float(args.max_target_lead_limit_rate):
        issues.append(f"target-lead limit rate too high: max={target_lead_cap}")

    bc_blend_tags = [
        "Episode/cube_push_bc_teacher_blend_mean",
        "Episode/cube_tap_bc_teacher_blend_mean",
    ]
    bc_mse_tags = [
        "Episode/cube_push_bc_teacher_imitation_mse",
        "Episode/cube_tap_bc_teacher_imitation_mse",
    ]
    bc_action_tags = [
        "Episode/cube_push_bc_teacher_action_abs_mean",
        "Episode/cube_tap_bc_teacher_action_abs_mean",
    ]
    bc_blend = _max_existing(scalars, bc_blend_tags, "max")
    if args.expect_bc_teacher:
        if not _has_any(scalars, bc_blend_tags):
            issues.append("BC teacher blend scalar is missing")
        elif not _finite(bc_blend) or float(bc_blend) <= 0.0:
            issues.append(f"BC teacher blend is not active: max={bc_blend}")
        if not _has_any(scalars, bc_mse_tags):
            issues.append("BC teacher imitation MSE scalar is missing")
        if not _has_any(scalars, bc_action_tags):
            warnings.append("BC teacher action magnitude scalar is missing")
    elif _finite(bc_blend) and float(bc_blend) > 0.0:
        if not _has_any(scalars, bc_mse_tags):
            issues.append("BC teacher blend is nonzero but imitation MSE is missing")

    d256_reset_tags = [
        "Episode/cube_push_d256_reset_active_rate",
        "Episode/cube_tap_d256_reset_active_rate",
    ]
    d256_reset = _max_existing(scalars, d256_reset_tags, "max")
    if args.expect_d256_reset:
        if not _has_any(scalars, d256_reset_tags):
            issues.append("D256 reset active scalar is missing")
        elif not _finite(d256_reset) or float(d256_reset) < 0.99:
            issues.append(f"D256 reset hook is not active for all envs: max={d256_reset}")

    reward = scalars.get("Train/mean_reward")
    if reward and float(reward["last"]) < float(reward["first"]):
        warnings.append(
            f"mean reward decreased: first={reward['first']} last={reward['last']}"
        )

    noise = scalars.get("Policy/mean_noise_std")
    if noise:
        if float(noise["last"]) <= 0.0:
            issues.append(f"policy noise std invalid: last={noise['last']}")
        elif float(noise["last"]) > 2.0 * max(float(noise["first"]), 1.0e-6):
            warnings.append(
                f"policy noise std expanded strongly: first={noise['first']} last={noise['last']}"
            )

    if issues:
        verdict = "TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION"
    elif warnings:
        verdict = "TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW"
    else:
        verdict = "TENSORBOARD_GATE_PASS_FOR_NEXT_SHORT_GATE_NOT_LONG_PPO"
    return verdict, issues, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    parser.add_argument("--artifact_tag", type=str, default="d260")
    parser.add_argument("--env_kind", choices=("auto", "push3cm", "tap10cm"), default="auto")
    parser.add_argument("--expect_bc_teacher", action="store_true")
    parser.add_argument("--expect_d256_reset", action="store_true")
    parser.add_argument("--allow_missing_train_episode_scalars", action="store_true")
    parser.add_argument("--min_iterations_for_promotion", type=int, default=10)
    parser.add_argument("--min_success_or_contact_rate", type=float, default=0.01)
    parser.add_argument("--min_tap_useful_seen_rate", type=float, default=0.01)
    parser.add_argument("--min_disp_along_m", type=float, default=0.001)
    parser.add_argument("--min_tap_max_disp_along_m", type=float, default=0.001)
    parser.add_argument("--min_tap_disp_along_ge_1mm_rate", type=float, default=0.0)
    parser.add_argument("--min_tap_disp_xy_ge_1mm_rate", type=float, default=0.0)
    parser.add_argument("--require_tap_displacement_gate", action="store_true")
    parser.add_argument("--require_collection_final_tap_gate", action="store_true")
    parser.add_argument("--min_collection_final_success_or_contact_rate", type=float, default=0.90)
    parser.add_argument("--min_collection_final_tap_useful_seen_rate", type=float, default=0.90)
    parser.add_argument("--min_collection_final_tap_success_rate", type=float, default=0.0)
    parser.add_argument("--max_collection_final_tap_overshoot_seen_rate", type=float, default=0.05)
    parser.add_argument("--min_collection_final_tap_disp_xy_ge_1mm_rate", type=float, default=0.25)
    parser.add_argument("--max_tcp_cube_dist_m", type=float, default=0.08)
    parser.add_argument("--max_tap_contact_vertical_offset_m", type=float, default=0.08)
    parser.add_argument("--tap_vertical_gate_mode", choices=("last", "min_contact"), default="last")
    parser.add_argument("--max_tap_overshoot_seen_rate", type=float, default=0.05)
    parser.add_argument("--min_controlled_rate", type=float, default=0.10)
    parser.add_argument("--max_low_motion_rate", type=float, default=0.80)
    parser.add_argument("--max_joint_delta_cap_rate", type=float, default=0.25)
    parser.add_argument("--max_target_lead_limit_rate", type=float, default=0.25)
    args = parser.parse_args()

    for name in (
        "min_disp_along_m",
        "min_tap_max_disp_along_m",
        "min_tap_disp_along_ge_1mm_rate",
        "min_tap_disp_xy_ge_1mm_rate",
        "min_collection_final_success_or_contact_rate",
        "min_collection_final_tap_useful_seen_rate",
        "min_collection_final_tap_success_rate",
        "max_collection_final_tap_overshoot_seen_rate",
        "min_collection_final_tap_disp_xy_ge_1mm_rate",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    for name in (
        "min_tap_disp_along_ge_1mm_rate",
        "min_tap_disp_xy_ge_1mm_rate",
        "min_collection_final_success_or_contact_rate",
        "min_collection_final_tap_useful_seen_rate",
        "min_collection_final_tap_success_rate",
        "max_collection_final_tap_overshoot_seen_rate",
        "min_collection_final_tap_disp_xy_ge_1mm_rate",
    ):
        if float(getattr(args, name)) > 1.0:
            raise ValueError(f"--{name} must be <= 1.0")

    log_dir = args.log_dir
    if args.out_json is None:
        args.out_json = log_dir / "tensorboard_scalar_gate_d260.json"
    if args.out_md is None:
        args.out_md = log_dir / "tensorboard_scalar_gate_d260.md"

    event_files, scalars = _load_scalars(log_dir)
    env_kind = _detect_env_kind(args, scalars)
    verdict, issues, warnings = _gate(args, scalars)

    selected_tags = [tag for tag in CORE_TAGS + TASK_TAGS if tag in scalars]
    summary = {
        "artifact": f"cube10cm_top_view_tensorboard_scalar_gate_{args.artifact_tag}",
        "status": "PASS_EVENT_READ",
        "verdict": verdict,
        "env_kind": env_kind,
        "log_dir": _rel(log_dir),
        "event_files": [_rel(Path(p)) for p in event_files],
        "dashboard_command": (
            f"conda run -n isaaclab tensorboard --logdir {_rel(log_dir)} "
            "--host 127.0.0.1 --port 6006"
        ),
        "issues": issues,
        "warnings": warnings,
        "selected_tags": selected_tags,
        "scalars": scalars,
        "gate_args": {
            "env_kind": args.env_kind,
            "detected_env_kind": env_kind,
            "expect_bc_teacher": bool(args.expect_bc_teacher),
            "expect_d256_reset": bool(args.expect_d256_reset),
            "allow_missing_train_episode_scalars": bool(args.allow_missing_train_episode_scalars),
            "min_iterations_for_promotion": args.min_iterations_for_promotion,
            "min_success_or_contact_rate": args.min_success_or_contact_rate,
            "min_tap_useful_seen_rate": args.min_tap_useful_seen_rate,
            "min_disp_along_m": args.min_disp_along_m,
            "min_tap_max_disp_along_m": args.min_tap_max_disp_along_m,
            "min_tap_disp_along_ge_1mm_rate": args.min_tap_disp_along_ge_1mm_rate,
            "min_tap_disp_xy_ge_1mm_rate": args.min_tap_disp_xy_ge_1mm_rate,
            "require_tap_displacement_gate": bool(args.require_tap_displacement_gate),
            "require_collection_final_tap_gate": bool(args.require_collection_final_tap_gate),
            "min_collection_final_success_or_contact_rate": args.min_collection_final_success_or_contact_rate,
            "min_collection_final_tap_useful_seen_rate": args.min_collection_final_tap_useful_seen_rate,
            "min_collection_final_tap_success_rate": args.min_collection_final_tap_success_rate,
            "max_collection_final_tap_overshoot_seen_rate": args.max_collection_final_tap_overshoot_seen_rate,
            "min_collection_final_tap_disp_xy_ge_1mm_rate": args.min_collection_final_tap_disp_xy_ge_1mm_rate,
            "max_tcp_cube_dist_m": args.max_tcp_cube_dist_m,
            "max_tap_contact_vertical_offset_m": args.max_tap_contact_vertical_offset_m,
            "tap_vertical_gate_mode": args.tap_vertical_gate_mode,
            "max_tap_overshoot_seen_rate": args.max_tap_overshoot_seen_rate,
            "min_controlled_rate": args.min_controlled_rate,
            "max_low_motion_rate": args.max_low_motion_rate,
            "max_joint_delta_cap_rate": args.max_joint_delta_cap_rate,
            "max_target_lead_limit_rate": args.max_target_lead_limit_rate,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    metric_lines = [_metric_line(tag, scalars) for tag in selected_tags]
    args.out_md.write_text(
        f"# {args.artifact_tag.upper()} TensorBoard Scalar Gate\n\n"
        f"- verdict: `{verdict}`\n"
        f"- env kind: `{env_kind}`\n"
        f"- log dir: `{_rel(log_dir)}`\n"
        f"- event files: `{len(event_files)}`\n"
        f"- dashboard command: `{summary['dashboard_command']}`\n\n"
        "## Issues\n\n"
        + ("\n".join(f"- {item}" for item in issues) if issues else "- none")
        + "\n\n## Warnings\n\n"
        + ("\n".join(f"- {item}" for item in warnings) if warnings else "- none")
        + "\n\n## Selected Scalars\n\n"
        + "\n".join(metric_lines)
        + "\n"
    )

    print(
        "tensorboard_scalar_gate "
        f"verdict={verdict} tags={len(scalars)} issues={len(issues)} "
        f"warnings={len(warnings)} json={_rel(args.out_json)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
