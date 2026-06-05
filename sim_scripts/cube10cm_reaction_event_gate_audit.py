"""Audit professor 10cm/0.72kg cube push/tap reaction-event logs.

This is a local posthoc tool. It reads existing summary/per-env CSV logs only.
It does not run IsaacLab, train, generate data, or touch the robot.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(values: list[bool]) -> float:
    return _mean([1.0 if value else 0.0 for value in values])


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def _resolve_csv_path(summary_path: Path, summary: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    raw_path = summary.get("out_csv")
    if not raw_path:
        raise ValueError("--csv is required because summary_json does not contain out_csv")
    candidate = Path(str(raw_path))
    if candidate.exists():
        return candidate
    relative = summary_path.parent / candidate.name
    if relative.exists():
        return relative
    return candidate


def _row_flag(row: dict[str, str], key: str) -> bool:
    return _boolish(row.get(key, "0"))


def _summary_flag(summary: dict[str, Any], key: str) -> bool:
    return _boolish(summary.get(key))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--trace_diag_json", type=Path, default=None)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--reaction_disp_m", type=float, default=0.001)
    parser.add_argument("--reaction_z_delta_m", type=float, default=0.002)
    parser.add_argument("--reaction_speed_mps", type=float, default=0.020)
    parser.add_argument("--tap_gate_disp_m", type=float, default=0.001)
    parser.add_argument("--final_relocation_disp_m", type=float, default=None)
    parser.add_argument("--overshoot_disp_m", type=float, default=0.020)
    parser.add_argument("--min_reaction_event_rate", type=float, default=1.0)
    parser.add_argument("--min_contact_evidence_rate", type=float, default=1.0)
    parser.add_argument("--max_overshoot_rate", type=float, default=0.0)
    parser.add_argument("--max_posewrite_calls", type=int, default=0)
    parser.add_argument("--teacher_max_final_tcp_err_m", type=float, default=0.030)
    parser.add_argument("--teacher_max_diffik_clip_rate", type=float, default=0.50)
    args = parser.parse_args()

    if args.reaction_disp_m < 0.0:
        raise ValueError("--reaction_disp_m must be non-negative")
    if args.reaction_z_delta_m < 0.0:
        raise ValueError("--reaction_z_delta_m must be non-negative")
    if args.reaction_speed_mps < 0.0:
        raise ValueError("--reaction_speed_mps must be non-negative")
    if args.tap_gate_disp_m < 0.0:
        raise ValueError("--tap_gate_disp_m must be non-negative")
    if args.final_relocation_disp_m is not None and args.final_relocation_disp_m <= 0.0:
        raise ValueError("--final_relocation_disp_m must be positive when provided")
    if args.overshoot_disp_m <= 0.0:
        raise ValueError("--overshoot_disp_m must be positive")
    if not (0.0 <= args.min_reaction_event_rate <= 1.0):
        raise ValueError("--min_reaction_event_rate must be in [0, 1]")
    if not (0.0 <= args.min_contact_evidence_rate <= 1.0):
        raise ValueError("--min_contact_evidence_rate must be in [0, 1]")
    if not (0.0 <= args.max_overshoot_rate <= 1.0):
        raise ValueError("--max_overshoot_rate must be in [0, 1]")

    summary: dict[str, Any] = json.loads(args.summary_json.read_text())
    csv_path = _resolve_csv_path(args.summary_json, summary, args.csv)
    rows = _load_csv(csv_path)
    if not rows:
        raise RuntimeError(f"empty CSV: {csv_path}")

    trace_diag: dict[str, Any] | None = None
    if args.trace_diag_json is not None:
        trace_diag = json.loads(args.trace_diag_json.read_text())

    final_disp = [_float(row.get("disp_along_push_m")) for row in rows]
    max_disp = [_float(row.get("max_disp_along_push_m"), _float(row.get("disp_along_push_m"))) for row in rows]
    max_z_delta = [_float(row.get("max_cube_z_delta_m")) for row in rows]
    max_speed = [_float(row.get("max_cube_speed_mps")) for row in rows]

    computed_reaction = [
        (
            max_disp[idx] >= args.reaction_disp_m
            or max_z_delta[idx] >= args.reaction_z_delta_m
            or max_speed[idx] >= args.reaction_speed_mps
        )
        for idx in range(len(rows))
    ]
    row_reaction = [
        _row_flag(row, "reaction_event") if "reaction_event" in row else computed_reaction[idx]
        for idx, row in enumerate(rows)
    ]
    contact_evidence = [
        _row_flag(row, "measured_contact_seen") or _row_flag(row, "contact_stop_seen")
        for row in rows
    ]
    overshoot = [
        _row_flag(row, "contact_overshoot_seen") or max_disp[idx] >= args.overshoot_disp_m
        for idx, row in enumerate(rows)
    ]
    tap_gate = [value >= args.tap_gate_disp_m for value in max_disp]
    final_relocation_gate_rate = None
    final_relocation_pass = None
    if args.final_relocation_disp_m is not None:
        final_relocation_gate_rate = _mean(
            [1.0 if value >= args.final_relocation_disp_m else 0.0 for value in final_disp]
        )
        final_relocation_pass = final_relocation_gate_rate >= args.min_reaction_event_rate

    no_posewrite = (
        int(summary.get("posewrite_calls_during_rollout", -1)) <= args.max_posewrite_calls
        and not _summary_flag(summary, "rollout_object_posewrite")
        and not _summary_flag(summary, "training")
        and not _summary_flag(summary, "dataset_generation")
        and not _summary_flag(summary, "grasp_attach")
    )
    controller_ok = summary.get("controller") == "IsaacLab_DifferentialIKController"
    reaction_event_rate = _rate(row_reaction)
    computed_reaction_event_rate = _rate(computed_reaction)
    contact_evidence_rate = _rate(contact_evidence)
    overshoot_rate = _rate(overshoot)
    tap_gate_rate = _rate(tap_gate)

    reaction_gate_pass = (
        controller_ok
        and no_posewrite
        and reaction_event_rate >= args.min_reaction_event_rate
        and contact_evidence_rate >= args.min_contact_evidence_rate
        and overshoot_rate <= args.max_overshoot_rate
    )
    teacher_quality_ready = (
        reaction_gate_pass
        and _float(summary.get("final_tcp_target_err_mean_m")) <= args.teacher_max_final_tcp_err_m
        and _float(summary.get("diffik_clip_rate_mean")) <= args.teacher_max_diffik_clip_rate
    )

    likely_modes = []
    if trace_diag is not None:
        likely_modes = list(trace_diag.get("likely_modes", []))

    audit_summary: dict[str, Any] = {
        "summary_json": str(args.summary_json),
        "csv": str(csv_path),
        "trace_diag_json": str(args.trace_diag_json) if args.trace_diag_json is not None else None,
        "trials": len(rows),
        "controller_ok": controller_ok,
        "no_posewrite": no_posewrite,
        "reaction_gate_pass": reaction_gate_pass,
        "final_relocation_pass": final_relocation_pass,
        "final_relocation_gate_rate": final_relocation_gate_rate,
        "teacher_quality_ready": teacher_quality_ready,
        "reaction_event_rate": reaction_event_rate,
        "computed_reaction_event_rate": computed_reaction_event_rate,
        "contact_evidence_rate": contact_evidence_rate,
        "overshoot_rate": overshoot_rate,
        "tap_gate_rate": tap_gate_rate,
        "final_disp_mean_m": _mean(final_disp),
        "max_disp_mean_m": _mean(max_disp),
        "max_z_delta_mean_m": _mean(max_z_delta),
        "max_speed_mean_mps": _mean(max_speed),
        "summary_measured_contact_seen_rate": _float(summary.get("measured_contact_seen_rate")),
        "summary_contact_stop_seen_rate": _float(summary.get("contact_stop_seen_rate")),
        "summary_max_disp_ge_gate_rate": _float(summary.get("max_disp_ge_gate_rate")),
        "summary_disp_ge_gate_rate": _float(summary.get("disp_ge_gate_rate")),
        "summary_final_tcp_target_err_mean_m": _float(summary.get("final_tcp_target_err_mean_m")),
        "summary_diffik_clip_rate_mean": _float(summary.get("diffik_clip_rate_mean")),
        "likely_modes": likely_modes,
        "thresholds": {
            "reaction_disp_m": args.reaction_disp_m,
            "reaction_z_delta_m": args.reaction_z_delta_m,
            "reaction_speed_mps": args.reaction_speed_mps,
            "tap_gate_disp_m": args.tap_gate_disp_m,
            "final_relocation_disp_m": args.final_relocation_disp_m,
            "overshoot_disp_m": args.overshoot_disp_m,
            "min_reaction_event_rate": args.min_reaction_event_rate,
            "min_contact_evidence_rate": args.min_contact_evidence_rate,
            "max_overshoot_rate": args.max_overshoot_rate,
            "teacher_max_final_tcp_err_m": args.teacher_max_final_tcp_err_m,
            "teacher_max_diffik_clip_rate": args.teacher_max_diffik_clip_rate,
        },
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n")

    verdict = "PASS" if reaction_gate_pass else "FAIL"
    teacher_verdict = "READY" if teacher_quality_ready else "NOT_READY"
    print(
        "reaction_gate line1 "
        f"verdict={verdict} trials={len(rows)} controller_ok={controller_ok} no_posewrite={no_posewrite} "
        f"reaction_event_rate={reaction_event_rate:.9f} contact_evidence_rate={contact_evidence_rate:.9f} "
        f"overshoot_rate={overshoot_rate:.9f}"
    )
    print(
        "reaction_gate line2 "
        f"tap_gate_rate={tap_gate_rate:.9f} "
        f"final_relocation_gate_rate={final_relocation_gate_rate} "
        f"max_disp_mean_m={_mean(max_disp):.9f} final_disp_mean_m={_mean(final_disp):.9f} "
        f"max_speed_mean_mps={_mean(max_speed):.9f}"
    )
    print(
        "reaction_gate line3 "
        f"teacher_quality={teacher_verdict} final_tcp_err_mean_m={_float(summary.get('final_tcp_target_err_mean_m')):.9f} "
        f"diffik_clip_rate_mean={_float(summary.get('diffik_clip_rate_mean')):.9f} "
        f"likely_modes={','.join(likely_modes) if likely_modes else 'NA'}"
    )
    return 0 if reaction_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
