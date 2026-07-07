#!/usr/bin/env python3
"""Aggregate D319 data-conveyor pilot CSVs.

This is an offline audit: it reads already generated D290 env-level CSVs and
the existing 0-999 script-render labels. It does not run Isaac, PPO, or render.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pvariance
from typing import Any


ROOT = Path("claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480")
D319_ROOT = ROOT / "data_conveyor_d319"
ENVCV_ROOT = D319_ROOT / "tap10cm_envcsv"
SCRIPT_LABELS = ROOT / "cube10cm_top_view_visual_0_999_d242/postrender_label_validation_d246/episode_labels.csv"
OUT_DIR = D319_ROOT / "audit"


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "variance": None}
    return {
        "count": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "min": min(clean),
        "max": max(clean),
        "variance": pvariance(clean) if len(clean) > 1 else 0.0,
    }


def _disp_bin_m(value: float) -> str:
    if value < 0.001:
        return "<1mm"
    if value < 0.003:
        return "1-3mm"
    if value < 0.007:
        return "3-7mm"
    if value < 0.020:
        return "7-20mm"
    return ">=20mm"


def _angle_bin(dx: float, dy: float) -> str:
    if math.hypot(dx, dy) < 1e-9:
        return "zero"
    deg = math.degrees(math.atan2(dy, dx))
    if -22.5 <= deg < 22.5:
        return "+x"
    if 22.5 <= deg < 67.5:
        return "+x/+y"
    if 67.5 <= deg < 112.5:
        return "+y"
    if 112.5 <= deg < 157.5:
        return "-x/+y"
    if deg >= 157.5 or deg < -157.5:
        return "-x"
    if -157.5 <= deg < -112.5:
        return "-x/-y"
    if -112.5 <= deg < -67.5:
        return "-y"
    return "+x/-y"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_d319() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary_by_tag: dict[str, dict[str, Any]] = {}
    for csv_path in sorted(ENVCV_ROOT.glob("bin_*/chunk_*/closed_loop_recovery_envs_*.csv")):
        summary_path = next(csv_path.parent.glob("closed_loop_recovery_summary_*.json"))
        summary = _read_json(summary_path)
        tag = str(summary["artifact_tag"])
        summary_by_tag[tag] = summary
        bin_name = csv_path.parts[-3]
        chunk = csv_path.parts[-2]
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                accepted = (
                    _i(row, "tap_contact_seen") == 1
                    and _i(row, "tap_reaction_seen") == 1
                    and _i(row, "tap_useful_seen") == 1
                    and _i(row, "tap_overshoot_seen") == 0
                    and _f(row, "max_disp_xy_m") >= 0.001
                )
                rows.append(
                    {
                        "bin": bin_name,
                        "chunk": chunk,
                        "artifact_tag": tag,
                        "env_id": _i(row, "env_id"),
                        "episode_index": _i(row, "episode_index"),
                        "static_friction": float(summary["cube_static_friction"]),
                        "dynamic_friction": float(summary["cube_dynamic_friction"]),
                        "contact": _i(row, "tap_contact_seen"),
                        "reaction": _i(row, "tap_reaction_seen"),
                        "useful": _i(row, "tap_useful_seen"),
                        "overshoot": _i(row, "tap_overshoot_seen"),
                        "max_disp_xy_m": _f(row, "max_disp_xy_m"),
                        "max_disp_along_m": _f(row, "max_disp_along_m"),
                        "max_lateral_disp_m": _f(row, "max_lateral_disp_m"),
                        "final_proxy": _i(row, "final_tap_contact_proxy_now"),
                        "hybrid_latched": _i(row, "candidate8_hybrid_stop_latched"),
                        "hybrid_stop_step": _i(row, "candidate8_hybrid_stop_step", -1),
                        "accepted": int(accepted),
                    }
                )
    return rows, summary_by_tag


def collect_script_labels() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with SCRIPT_LABELS.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            camera_pass = _i(row, "camera_contract_pass")
            useful_clean = _i(row, "label_useful_clean_numeric")
            overshoot = _i(row, "label_overshoot_numeric")
            contact = _i(row, "contact_seen_any")
            reaction = _i(row, "reaction_seen_any")
            max_xy = _f(row, "max_tap_disp_xy_m")
            dx = _f(row, "final_dx_m")
            dy = _f(row, "final_dy_m")
            accepted = camera_pass == 1 and useful_clean == 1 and max_xy >= 0.001
            rows.append(
                {
                    "episode_index": _i(row, "episode_index"),
                    "split_candidate": row.get("split_candidate", ""),
                    "label_status": row.get("label_status", ""),
                    "camera_pass": camera_pass,
                    "contact": contact,
                    "reaction": reaction,
                    "overshoot": overshoot,
                    "useful_clean": useful_clean,
                    "max_disp_xy_m": max_xy,
                    "max_disp_along_m": _f(row, "max_tap_disp_along_m"),
                    "final_dx_m": dx,
                    "final_dy_m": dy,
                    "direction_bin": _angle_bin(dx, dy),
                    "accepted": int(accepted),
                }
            )
    return rows


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    accepted = [row for row in rows if int(row["accepted"]) == 1]
    return {
        "generated": n,
        "accepted": len(accepted),
        "pass_rate": len(accepted) / n if n else 0.0,
        "contact": sum(int(row["contact"]) for row in rows),
        "reaction": sum(int(row["reaction"]) for row in rows),
        "useful": sum(int(row.get("useful", row.get("useful_clean", 0))) for row in rows),
        "overshoot": sum(int(row["overshoot"]) for row in rows),
        "low_motion_lt_1mm": sum(1 for row in rows if float(row["max_disp_xy_m"]) < 0.001),
        "max_disp_xy_m": _stats([float(row["max_disp_xy_m"]) for row in rows]),
        "accepted_max_disp_xy_m": _stats([float(row["max_disp_xy_m"]) for row in accepted]),
        "disp_bins_all": dict(Counter(_disp_bin_m(float(row["max_disp_xy_m"])) for row in rows)),
        "disp_bins_accepted": dict(Counter(_disp_bin_m(float(row["max_disp_xy_m"])) for row in accepted)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def main() -> None:
    d319_rows, summaries = collect_d319()
    script_rows = collect_script_labels()

    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in d319_rows:
        by_bin[str(row["bin"])].append(row)

    d319_summary = {bin_name: summarize_group(rows) for bin_name, rows in sorted(by_bin.items())}
    d319_all = summarize_group(d319_rows)
    script_all = summarize_group(script_rows)
    script_accepted = [row for row in script_rows if int(row["accepted"]) == 1]
    d319_accepted = [row for row in d319_rows if int(row["accepted"]) == 1]

    script_direction = dict(Counter(row["direction_bin"] for row in script_accepted))
    d319_commanded_direction = {"+x_object_frame_commanded": len(d319_accepted)}

    selected = []
    for row in d319_accepted:
        if str(row["bin"]) == "bin_upper_1p2_1p6":
            continue
        selected.append(row)
        if len(selected) >= 200:
            break

    quota_eval = {}
    for bin_name, summary in d319_summary.items():
        quota_eval[bin_name] = {
            "pass_rate": summary["pass_rate"],
            "meets_generator_gate_ge_30pct": bool(summary["pass_rate"] >= 0.30),
            "interpretation": "producer_bin" if summary["pass_rate"] >= 0.30 else "rl_contribution_candidate_freeze",
        }

    payload = {
        "artifact": "d319_data_conveyor_audit",
        "d319_envcsv_root": str(ENVCV_ROOT),
        "script_label_source": str(SCRIPT_LABELS),
        "filter_rule": "contact=1 AND reaction=1 AND tap_useful_seen=1 AND overshoot=0 AND max_disp_xy>=0.001m",
        "d319_total": d319_all,
        "d319_by_bin": d319_summary,
        "d319_quota_eval": quota_eval,
        "script_0_999_total": script_all,
        "script_accepted_direction_histogram": script_direction,
        "d319_commanded_direction_histogram": d319_commanded_direction,
        "d319_accepted_lateral_m": _stats([float(row["max_lateral_disp_m"]) for row in d319_accepted]),
        "script_accepted_count": len(script_accepted),
        "d319_selected_for_replay_count": len(selected),
        "d319_summary_tags": summaries,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "d319_data_conveyor_audit_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_csv(OUT_DIR / "d319_all_env_filter_rows.csv", d319_rows)
    write_csv(OUT_DIR / "d319_accepted_env_rows.csv", d319_accepted)
    write_csv(OUT_DIR / "d319_selected_200_for_replay_manifest.csv", selected)

    bin_rows = []
    for bin_name, summary in d319_summary.items():
        bin_rows.append(
            [
                bin_name,
                str(summary["generated"]),
                f"{summary['accepted']} ({summary['pass_rate'] * 100:.1f}%)",
                str(summary["contact"]),
                str(summary["reaction"]),
                str(summary["useful"]),
                str(summary["overshoot"]),
                f"{summary['max_disp_xy_m']['mean'] * 1000:.2f}mm",
                f"{summary['max_disp_xy_m']['max'] * 1000:.2f}mm",
                quota_eval[bin_name]["interpretation"],
            ]
        )

    diversity_rows = [
        [
            "script_0_999 accepted",
            str(script_all["accepted"]),
            f"{script_all['accepted_max_disp_xy_m']['mean'] * 1000:.2f}mm",
            f"{script_all['accepted_max_disp_xy_m']['variance'] * 1_000_000:.2f}mm^2",
            json.dumps(script_direction, sort_keys=True),
        ],
        [
            "d319 accepted",
            str(d319_all["accepted"]),
            f"{d319_all['accepted_max_disp_xy_m']['mean'] * 1000:.2f}mm",
            f"{d319_all['accepted_max_disp_xy_m']['variance'] * 1_000_000:.2f}mm^2",
            json.dumps(d319_commanded_direction, sort_keys=True),
        ],
    ]

    md = "\n".join(
        [
            "# D319 data conveyor audit",
            "",
            "Offline audit only: no Isaac runtime, no PPO, no render.",
            "",
            "Filter rule: contact=1, reaction=1, useful=1, overshoot=0, max XY >= 1mm.",
            "",
            "## Bin pass rates",
            "",
            md_table(
                [
                    "bin",
                    "generated",
                    "accepted",
                    "contact",
                    "reaction",
                    "useful",
                    "overshoot",
                    "mean XY",
                    "max XY",
                    "interpretation",
                ],
                bin_rows,
            ),
            "",
            "## Script-only vs D319 diversity",
            "",
            md_table(
                ["corpus", "accepted", "mean accepted XY", "accepted XY variance", "direction histogram"],
                diversity_rows,
            ),
            "",
            "## Critical findings",
            "",
            "- `bin_low_0p7_0p9` and `bin_mid_0p9_1p2` clear the >=30% generator gate.",
            "- `bin_upper_1p2_1p6` is below the generator gate and should be frozen as an RL contribution candidate instead of hand-patching the controller.",
            "- D319 accepted trajectories remain directionally narrow: the commanded primitive direction is fixed +x in object frame. This is acceptable for a fixture pilot but not sufficient for POSCO-style generalization.",
            "- The 200-row replay manifest is a selection manifest only. Existing render tooling does not yet replay D319 D290 env rows into LeRobot v3 without an additional replay renderer.",
            "",
            f"JSON: `{OUT_DIR / 'd319_data_conveyor_audit_summary.json'}`",
            f"Accepted rows: `{OUT_DIR / 'd319_accepted_env_rows.csv'}`",
            f"Replay selection: `{OUT_DIR / 'd319_selected_200_for_replay_manifest.csv'}`",
        ]
    )
    (OUT_DIR / "d319_data_conveyor_audit_summary.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
