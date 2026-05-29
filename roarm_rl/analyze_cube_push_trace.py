"""Summarize cube-push actor trace CSV/JSON outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _rate(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.trace_dir / "actor_trace_summary.json"
    steps_path = args.trace_dir / "actor_trace_steps.csv"
    envs_path = args.trace_dir / "actor_trace_envs.csv"
    summary = json.loads(summary_path.read_text())
    with steps_path.open() as f:
        steps = list(csv.DictReader(f))
    with envs_path.open() as f:
        envs = list(csv.DictReader(f))

    lines: list[str] = []
    lines.append(
        "trace_analysis line1 "
        f"source_summary={summary_path} traced_envs={summary['traced_envs']} "
        f"step_rows={summary['step_rows']} teacher_off=YES teacher_sidecar=COMPARE_ONLY"
    )
    lines.append(
        "trace_analysis line2 action_error "
        f"mean_action_mse={summary['mean_action_mse']:.9f} "
        f"mean_arm_mse={summary['mean_arm_mse']:.9f} "
        f"mean_action_mae={summary['mean_action_mae']:.9f} "
        f"mean_action_cos={summary['mean_action_cos']:.9f}"
    )
    lines.append(
        "trace_analysis line3 scale "
        f"actor_abs={summary['mean_actor_abs']:.9f} "
        f"teacher_abs={summary['mean_teacher_abs']:.9f} "
        f"effective_abs={summary['mean_effective_abs']:.9f} "
        f"effective_vs_actor_mse={summary['mean_effective_vs_actor_mse']:.9f} "
        f"joint_move_abs={summary['mean_joint_move_abs']:.9f}"
    )
    lines.append(
        "trace_analysis line4 rollout "
        f"controlled={summary['controlled_push_rate']:.9f} "
        f"impact={summary['impact_outlier_rate']:.9f} "
        f"low_motion={summary['low_motion_rate']:.9f} "
        f"success={summary['success_marker_rate']:.9f} "
        f"disp_along_mean_m={summary['disp_along_push_mean_m']:.9f} "
        f"contact_reached={summary['contact_reached_rate']:.9f}"
    )

    for name in ("approach_or_alpha0", "push_alpha01", "post_alpha1"):
        phase = summary["phase_summary"][name]
        lines.append(
            f"trace_analysis phase {name} samples={phase['samples']} "
            f"mean_action_mse={phase['mean_action_mse']:.9f} "
            f"mean_arm_mse={phase['mean_arm_mse']:.9f}"
        )

    for name in ("low_x", "mid_x", "high_x", "not_posx", "dir_-1_0", "dir_0_-1", "dir_0_1", "dir_1_0"):
        group = summary["grouped"].get(name)
        if not group:
            continue
        lines.append(
            f"trace_analysis group {name} n={group['n']} "
            f"controlled={group['controlled']:.9f} "
            f"low_motion={group['low_motion']:.9f} "
            f"success={group['success']:.9f} "
            f"action_mse={group['mean_action_mse']:.9f} "
            f"actor_abs={group['mean_actor_abs']:.9f} "
            f"teacher_abs={group['mean_teacher_abs']:.9f} "
            f"disp_along_m={group['disp_along_push_m']:.9f}"
        )

    for lo, hi in ((0, 99), (100, 219), (220, 349), (350, 599)):
        rows = [row for row in steps if lo <= int(row["step"]) <= hi]
        lines.append(
            f"trace_analysis stepbin {lo}_{hi} rows={len(rows)} "
            f"action_mse={_mean(_float(row, 'action_mse') for row in rows):.9f} "
            f"arm_mse={_mean(_float(row, 'arm_mse') for row in rows):.9f} "
            f"actor_abs={_mean(_float(row, 'actor_abs') for row in rows):.9f} "
            f"teacher_abs={_mean(_float(row, 'teacher_abs') for row in rows):.9f} "
            f"cos={_mean(_float(row, 'action_cos') for row in rows):.9f} "
            f"eff_vs_actor={_mean(_float(row, 'effective_vs_actor_mse') for row in rows):.9f} "
            f"disp_along={_mean(_float(row, 'disp_along_m') for row in rows):.9f}"
        )

    contact_groups = (
        ("contact_yes", [row for row in envs if int(row["first_contact_step"]) >= 0]),
        ("contact_no", [row for row in envs if int(row["first_contact_step"]) < 0]),
    )
    for label, rows in contact_groups:
        lines.append(
            f"trace_analysis contact {label} n={len(rows)} "
            f"controlled={_rate(int(row['controlled_push']) for row in rows):.9f} "
            f"low_motion={_rate(int(row['low_motion']) for row in rows):.9f} "
            f"success={_rate(int(row['success_marker']) for row in rows):.9f} "
            f"disp_along={_mean(_float(row, 'disp_along_push_m') for row in rows):.9f} "
            f"action_mse={_mean(_float(row, 'mean_action_mse') for row in rows):.9f}"
        )

    for prefix in ("actor_a", "teacher_a", "effective_a"):
        vals = [_mean(abs(_float(row, f"{prefix}{idx}")) for row in steps) for idx in range(6)]
        lines.append("trace_analysis joint_abs_mean " + prefix + "=" + ",".join(f"{value:.6f}" for value in vals))

    for idx in range(6):
        lines.append(
            f"trace_analysis joint{idx} signed "
            f"actor={_mean(_float(row, 'actor_a' + str(idx)) for row in steps):.9f} "
            f"teacher={_mean(_float(row, 'teacher_a' + str(idx)) for row in steps):.9f} "
            f"effective={_mean(_float(row, 'effective_a' + str(idx)) for row in steps):.9f} "
            f"mse={_mean((_float(row, 'actor_a' + str(idx)) - _float(row, 'teacher_a' + str(idx))) ** 2 for row in steps):.9f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
