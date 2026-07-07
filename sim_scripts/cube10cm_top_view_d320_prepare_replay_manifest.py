#!/usr/bin/env python3
"""Prepare the D320 D319-replay smoke manifest and upper-bin audit.

This is offline bookkeeping only. It selects a tiny render smoke set from the
D319 env-level conveyor rows and records the pre-registered upper-bin physicality
audit that decides whether the upper bin is a real dynamics target or solver
contamination.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
D319_AUDIT = RUNTIME_ROOT / "data_conveyor_d319" / "audit"
DEFAULT_ROWS = D319_AUDIT / "d319_all_env_filter_rows.csv"
DEFAULT_OUT = RUNTIME_ROOT / "data_conveyor_d320" / "replay_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upper-failure-count", type=int, default=3)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty D319 row csv: {path}")
    return rows


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def take_rows(
    rows: list[dict[str, str]],
    *,
    bin_name: str,
    accepted: int,
    overshoot: int,
    count: int,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["bin"] == bin_name
        and int(row["accepted"]) == int(accepted)
        and int(row["overshoot"]) == int(overshoot)
    ]
    if len(selected) < count:
        raise RuntimeError(
            f"not enough rows for bin={bin_name} accepted={accepted} "
            f"overshoot={overshoot}: have {len(selected)} need {count}"
        )
    return selected[:count]


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    idx = int(round(float(q) * (len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def histogram(values: list[float]) -> list[dict[str, Any]]:
    bins_m = [0.0, 0.02, 0.03, 0.05, 0.10, 0.30, 1.00, math.inf]
    labels = ["0-20mm", "20-30mm", "30-50mm", "50-100mm", "100-300mm", "300-1000mm", ">=1000mm"]
    out: list[dict[str, Any]] = []
    for lo, hi, label in zip(bins_m[:-1], bins_m[1:], labels, strict=True):
        count = sum((value >= lo) and (value < hi) for value in values)
        out.append(
            {
                "bin": label,
                "lo_m": None if math.isinf(lo) else float(lo),
                "hi_m": None if math.isinf(hi) else float(hi),
                "count": int(count),
                "rate": float(count / max(1, len(values))),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("cannot write empty manifest")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.rows_csv)

    selections: list[tuple[str, list[dict[str, str]]]] = [
        ("low_accepted", take_rows(rows, bin_name="bin_low_0p7_0p9", accepted=1, overshoot=0, count=2)),
        ("mid_accepted", take_rows(rows, bin_name="bin_mid_0p9_1p2", accepted=1, overshoot=0, count=2)),
        ("upper_accepted", take_rows(rows, bin_name="bin_upper_1p2_1p6", accepted=1, overshoot=0, count=2)),
        (
            "upper_overshoot_failure",
            take_rows(
                rows,
                bin_name="bin_upper_1p2_1p6",
                accepted=0,
                overshoot=1,
                count=int(args.upper_failure_count),
            ),
        ),
    ]

    manifest_rows: list[dict[str, Any]] = []
    for role, role_rows in selections:
        for row in role_rows:
            out_row: dict[str, Any] = {
                "d320_episode_id": len(manifest_rows),
                "source_role": role,
                "selection_reason": "accepted_smoke" if "accepted" in role else "upper_overshoot_visual_audit",
            }
            out_row.update(row)
            manifest_rows.append(out_row)

    manifest_path = args.out_dir / "d320_replay_smoke_manifest.csv"
    write_csv(manifest_path, manifest_rows)

    upper_overshoot = [
        as_float(row, "max_disp_xy_m")
        for row in rows
        if row["bin"] == "bin_upper_1p2_1p6" and int(row["overshoot"]) == 1
    ]
    upper_overshoot.sort()
    quantiles = {str(q): percentile(upper_overshoot, q) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)}
    below_300mm = sum(value < 0.300 for value in upper_overshoot)
    meter_scale = sum(value >= 1.000 for value in upper_overshoot)
    majority_below_300mm = below_300mm > len(upper_overshoot) * 0.5
    verdict = (
        "MOSTLY_PHYSICAL_FAILURE_RL_CONTRIBUTION_TARGET"
        if majority_below_300mm and meter_scale == 0
        else "MIXED_PHYSICAL_FAILURE_WITH_SOLVER_OUTLIERS"
        if majority_below_300mm
        else "SOLVER_CONTAMINATION_REGENERATE_WITH_LOWER_FRICTION_CAP"
    )
    audit = {
        "artifact": "d320_upper_bin_physicality_audit",
        "source_csv": rel(args.rows_csv),
        "pre_registered_decision_rule": {
            "mostly_under_300mm": "physical failure; RL contribution target",
            "meter_scale_present": "solver contamination present; isolate outliers before using upper bin",
        },
        "upper_overshoot_count": len(upper_overshoot),
        "max_disp_xy_m_quantiles": quantiles,
        "max_disp_xy_mm_quantiles": {key: value * 1000.0 for key, value in quantiles.items()},
        "histogram": histogram(upper_overshoot),
        "below_300mm_count": int(below_300mm),
        "below_300mm_rate": float(below_300mm / max(1, len(upper_overshoot))),
        "meter_scale_count": int(meter_scale),
        "meter_scale_rate": float(meter_scale / max(1, len(upper_overshoot))),
        "max_disp_xy_m_max": max(upper_overshoot) if upper_overshoot else math.nan,
        "decision": verdict,
    }
    audit_json = args.out_dir / "d320_upper_bin_physicality_audit.json"
    audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    audit_md = args.out_dir / "d320_upper_bin_physicality_audit.md"
    lines = [
        "# D320 upper-bin physicality audit",
        "",
        f"- Source: `{rel(args.rows_csv)}`",
        f"- Upper overshoot rows: {len(upper_overshoot)}",
        f"- Below 300mm: {below_300mm}/{len(upper_overshoot)} ({audit['below_300mm_rate']:.3f})",
        f"- Meter-scale rows: {meter_scale}/{len(upper_overshoot)} ({audit['meter_scale_rate']:.3f})",
        f"- Decision: `{verdict}`",
        "",
        "| quantile | max XY (mm) |",
        "|---:|---:|",
    ]
    for key, value in audit["max_disp_xy_mm_quantiles"].items():
        lines.append(f"| {key} | {value:.3f} |")
    lines.extend(["", "| bin | count | rate |", "|---|---:|---:|"])
    for item in audit["histogram"]:
        lines.append(f"| {item['bin']} | {item['count']} | {item['rate']:.3f} |")
    audit_md.write_text("\n".join(lines) + "\n")

    summary = {
        "artifact": "d320_replay_smoke_manifest",
        "manifest": rel(manifest_path),
        "manifest_rows": len(manifest_rows),
        "upper_audit_json": rel(audit_json),
        "upper_audit_md": rel(audit_md),
        "roles": {role: sum(row["source_role"] == role for row in manifest_rows) for role, _ in selections},
    }
    summary_path = args.out_dir / "d320_prepare_replay_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "[d320-prepare-replay] "
        f"manifest={manifest_path} rows={len(manifest_rows)} "
        f"upper_decision={verdict} audit={audit_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
