#!/usr/bin/env python3
"""Audit D321 low/mid data-conveyor rows with a physicality gate.

This is offline bookkeeping. It reads D290 env-level CSVs produced by the
D321 conveyor runs, applies the reward-independent label filter, and excludes
meter-scale solver artifacts via the D320-derived 300mm physicality gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pvariance
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
D242_ROOT = RUNTIME_ROOT / "cube10cm_top_view_visual_0_999_d242"
DEFAULT_ENVCV_ROOT = RUNTIME_ROOT / "data_conveyor_d321" / "tap10cm_envcsv"
DEFAULT_OUT = RUNTIME_ROOT / "data_conveyor_d321" / "audit"
DEFAULT_D256_CSV = D242_ROOT / "rl_transition_preflight_d256" / "ppo_actor_prior_teacher_rows_d256.csv"
DEFAULT_RUN_SCRIPT = RUNTIME_ROOT / "data_conveyor_d321" / "run_d321_conveyor_chunks_envcsv.sh"
SCRIPT_LABELS = D242_ROOT / "postrender_label_validation_d246" / "episode_labels.csv"


D319_REFERENCE = {
    "bin_low_0p7_0p9": 0.9633333333333334,
    "bin_mid_0p9_1p2": 0.965,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envcsv-root", type=Path, default=DEFAULT_ENVCV_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--d256-csv", type=Path, default=DEFAULT_D256_CSV)
    parser.add_argument("--run-script", type=Path, default=DEFAULT_RUN_SCRIPT)
    parser.add_argument("--script-labels", type=Path, default=SCRIPT_LABELS)
    parser.add_argument("--physicality-max-disp-xy-m", type=float, default=0.300)
    return parser.parse_args()


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def i(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


def stats(values: list[float]) -> dict[str, float | int | None]:
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


def disp_bin(value: float) -> str:
    if value < 0.001:
        return "<1mm"
    if value < 0.003:
        return "1-3mm"
    if value < 0.007:
        return "3-7mm"
    if value < 0.020:
        return "7-20mm"
    if value < 0.300:
        return "20-300mm"
    return ">=300mm_solver_outlier"


def angle_bin(dx: float, dy: float) -> str:
    norm = math.hypot(dx, dy)
    if norm < 1.0e-9:
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


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_d256_reset_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError(f"empty D256 CSV: {path}")
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            if i(row, "frame_index_t", -1) != 0:
                continue
            ep = i(row, "episode_index", -1)
            if ep >= 0 and ep not in rows:
                rows[ep] = row
    if not rows:
        raise RuntimeError(f"no D256 frame_index_t=0 rows in {path}")
    return rows


def load_seed_map(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    seed_by_tag: dict[str, int] = {}
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text.startswith("run_chunk "):
            continue
        parts = shlex.split(text)
        if len(parts) < 7:
            continue
        seed = int(parts[3])
        tag = parts[6]
        seed_by_tag[tag] = seed
    return seed_by_tag


def load_script_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            camera_pass = i(row, "camera_contract_pass")
            useful_clean = i(row, "label_useful_clean_numeric")
            max_xy = f(row, "max_tap_disp_xy_m")
            accepted = camera_pass == 1 and useful_clean == 1 and max_xy >= 0.001
            rows.append(
                {
                    "episode_index": i(row, "episode_index"),
                    "accepted": int(accepted),
                    "max_disp_xy_m": max_xy,
                    "direction_bin": angle_bin(f(row, "final_dx_m"), f(row, "final_dy_m")),
                }
            )
    return rows


def rejection_reason(row: dict[str, str], physicality_max_m: float) -> str:
    if f(row, "max_disp_xy_m") >= float(physicality_max_m):
        return "solver_outlier"
    if i(row, "tap_contact_seen") != 1:
        return "no_contact"
    if i(row, "tap_reaction_seen") != 1:
        return "no_reaction"
    if i(row, "tap_useful_seen") != 1:
        return "not_useful"
    if i(row, "tap_overshoot_seen") != 0:
        return "overshoot"
    if f(row, "max_disp_xy_m") < 0.001:
        return "low_motion_lt_1mm"
    return "accepted"


def collect_rows(
    args: argparse.Namespace,
    d256_rows: dict[int, dict[str, str]],
    seed_by_tag: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    csv_paths = sorted(args.envcsv_root.glob("bin_*/chunk_*/closed_loop_recovery_envs_*.csv"))
    if not csv_paths:
        raise RuntimeError(f"no env CSVs under {args.envcsv_root}")

    for csv_path in csv_paths:
        summary_path = next(csv_path.parent.glob("closed_loop_recovery_summary_*.json"))
        summary = load_json(summary_path)
        tag = str(summary["artifact_tag"])
        summaries[tag] = summary
        bin_name = csv_path.parts[-3]
        chunk = csv_path.parts[-2]
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ep = i(row, "episode_index")
                d256 = d256_rows.get(ep)
                if d256 is None:
                    raise RuntimeError(f"D256 reset row missing for episode_index={ep}")
                reason = rejection_reason(row, float(args.physicality_max_disp_xy_m))
                accepted = reason == "accepted"
                out = {
                    "source": "script_v2",
                    "bin": bin_name,
                    "chunk": chunk,
                    "artifact_tag": tag,
                    "seed": int(seed_by_tag.get(tag, -1)),
                    "env_id": i(row, "env_id"),
                    "episode_index": ep,
                    "static_friction": float(summary["cube_static_friction"]),
                    "dynamic_friction": float(summary["cube_dynamic_friction"]),
                    "reset_frame_index": 0,
                    "reset_cube_local_x_m": f(d256, "cube_local_x_m"),
                    "reset_cube_local_y_m": f(d256, "cube_local_y_m"),
                    "reset_cube_local_z_m": f(d256, "cube_local_z_m"),
                    "reset_target_local_x_m": f(d256, "target_local_x_m"),
                    "reset_target_local_y_m": f(d256, "target_local_y_m"),
                    "reset_target_local_z_m": f(d256, "target_local_z_m"),
                    "reset_push_dx": f(d256, "push_dx"),
                    "reset_push_dy": f(d256, "push_dy"),
                    "contact": i(row, "tap_contact_seen"),
                    "reaction": i(row, "tap_reaction_seen"),
                    "useful": i(row, "tap_useful_seen"),
                    "overshoot": i(row, "tap_overshoot_seen"),
                    "max_disp_xy_m": f(row, "max_disp_xy_m"),
                    "max_disp_along_m": f(row, "max_disp_along_m"),
                    "max_lateral_disp_m": f(row, "max_lateral_disp_m"),
                    "final_proxy": i(row, "final_tap_contact_proxy_now"),
                    "hybrid_latched": i(row, "candidate8_hybrid_stop_latched"),
                    "hybrid_stop_step": i(row, "candidate8_hybrid_stop_step", -1),
                    "solver_outlier": int(reason == "solver_outlier"),
                    "accepted": int(accepted),
                    "reject_reason": reason,
                }
                rows.append(out)
    return rows, summaries


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    accepted = [row for row in rows if int(row["accepted"]) == 1]
    reasons = Counter(str(row["reject_reason"]) for row in rows)
    return {
        "generated": n,
        "accepted": len(accepted),
        "pass_rate": len(accepted) / n if n else 0.0,
        "reject_reasons": dict(sorted(reasons.items())),
        "contact": sum(int(row["contact"]) for row in rows),
        "reaction": sum(int(row["reaction"]) for row in rows),
        "useful": sum(int(row["useful"]) for row in rows),
        "overshoot": sum(int(row["overshoot"]) for row in rows),
        "solver_outlier": sum(int(row["solver_outlier"]) for row in rows),
        "low_motion_lt_1mm": sum(1 for row in rows if float(row["max_disp_xy_m"]) < 0.001),
        "max_disp_xy_m": stats([float(row["max_disp_xy_m"]) for row in rows]),
        "accepted_max_disp_xy_m": stats([float(row["max_disp_xy_m"]) for row in accepted]),
        "accepted_lateral_disp_m": stats([float(row["max_lateral_disp_m"]) for row in accepted]),
        "disp_bins_all": dict(Counter(disp_bin(float(row["max_disp_xy_m"])) for row in rows)),
        "disp_bins_accepted": dict(Counter(disp_bin(float(row["max_disp_xy_m"])) for row in accepted)),
        "static_friction": stats([float(row["static_friction"]) for row in rows]),
        "dynamic_friction": stats([float(row["dynamic_friction"]) for row in rows]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
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
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if float(args.physicality_max_disp_xy_m) <= 0.0:
        raise ValueError("--physicality-max-disp-xy-m must be positive")
    d256_rows = load_d256_reset_rows(args.d256_csv)
    seed_by_tag = load_seed_map(args.run_script)
    rows, summaries = collect_rows(args, d256_rows, seed_by_tag)
    script_rows = load_script_rows(args.script_labels)

    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bin[str(row["bin"])].append(row)

    d321_by_bin = {bin_name: summarize_group(bin_rows) for bin_name, bin_rows in sorted(by_bin.items())}
    d321_total = summarize_group(rows)
    accepted_rows = [row for row in rows if int(row["accepted"]) == 1]
    script_accepted = [row for row in script_rows if int(row["accepted"]) == 1]
    script_summary = summarize_group(
        [
            {
                "accepted": row["accepted"],
                "contact": 0,
                "reaction": 0,
                "useful": row["accepted"],
                "overshoot": 0,
                "solver_outlier": 0,
                "reject_reason": "accepted" if row["accepted"] else "not_useful",
                "max_disp_xy_m": row["max_disp_xy_m"],
                "max_lateral_disp_m": 0.0,
                "static_friction": 0.0,
                "dynamic_friction": 0.0,
            }
            for row in script_rows
        ]
    )

    pass_regression_flags: dict[str, Any] = {}
    for bin_name, summary in d321_by_bin.items():
        ref = D319_REFERENCE.get(bin_name)
        pass_regression_flags[bin_name] = {
            "d321_pass_rate": summary["pass_rate"],
            "d319_reference_pass_rate": ref,
            "delta_vs_d319": None if ref is None else summary["pass_rate"] - ref,
            "below_90pct_failure_condition": bool(summary["pass_rate"] < 0.90),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_csv = args.out_dir / "d321_all_env_filter_rows.csv"
    accepted_csv = args.out_dir / "d321_accepted_env_rows.csv"
    write_csv(all_csv, rows)
    write_csv(accepted_csv, accepted_rows)

    direction_hist = {
        "+x_object_frame_commanded": len(accepted_rows),
    }
    script_direction_hist = dict(Counter(str(row["direction_bin"]) for row in script_accepted))

    payload = {
        "artifact": "d321_data_conveyor_audit",
        "runtime": "OFFLINE_FILTER_WITH_D320_PHYSICALITY_GATE",
        "envcsv_root": rel(args.envcsv_root),
        "d256_csv": rel(args.d256_csv),
        "run_script": rel(args.run_script),
        "seed_map_count": len(seed_by_tag),
        "physicality_gate": {
            "max_disp_xy_m_gte": float(args.physicality_max_disp_xy_m),
            "reject_reason": "solver_outlier",
            "default_active": True,
            "basis": "D320 upper-bin audit: 6 meter-scale solver outliers, max 11.14m; 300mm is 3x cube width.",
        },
        "filter_rule": (
            "contact=1 AND reaction=1 AND useful=1 AND overshoot=0 "
            "AND max XY >=1mm AND max XY < physicality_max"
        ),
        "d321_total": d321_total,
        "d321_by_bin": d321_by_bin,
        "pass_regression_flags": pass_regression_flags,
        "script_0_999_total": script_summary,
        "script_accepted_direction_histogram": script_direction_hist,
        "d321_commanded_direction_histogram": direction_hist,
        "accepted_rows_csv": rel(accepted_csv),
        "all_rows_csv": rel(all_csv),
        "summaries": summaries,
    }
    summary_json = args.out_dir / "d321_data_conveyor_audit_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    bin_rows = []
    for bin_name, summary in d321_by_bin.items():
        ref = D319_REFERENCE.get(bin_name)
        delta = "" if ref is None else f"{(summary['pass_rate'] - ref) * 100:+.2f}pp"
        bin_rows.append(
            [
                bin_name,
                str(summary["generated"]),
                f"{summary['accepted']} ({summary['pass_rate'] * 100:.1f}%)",
                str(summary["contact"]),
                str(summary["reaction"]),
                str(summary["useful"]),
                str(summary["overshoot"]),
                str(summary["solver_outlier"]),
                json.dumps(summary["reject_reasons"], sort_keys=True),
                delta,
            ]
        )

    diversity_rows = [
        [
            "script_0_999 accepted",
            str(len(script_accepted)),
            f"{script_summary['accepted_max_disp_xy_m']['mean'] * 1000.0:.2f}mm",
            f"{script_summary['accepted_max_disp_xy_m']['variance'] * 1_000_000.0:.2f}mm^2",
            json.dumps(script_direction_hist, sort_keys=True),
        ],
        [
            "d321 accepted",
            str(len(accepted_rows)),
            f"{d321_total['accepted_max_disp_xy_m']['mean'] * 1000.0:.2f}mm",
            f"{d321_total['accepted_max_disp_xy_m']['variance'] * 1_000_000.0:.2f}mm^2",
            json.dumps(direction_hist, sort_keys=True),
        ],
    ]

    md = "\n".join(
        [
            "# D321 data conveyor audit",
            "",
            "Offline audit only: no Isaac runtime, no PPO, no render.",
            "",
            "Filter rule: contact=1, reaction=1, useful=1, overshoot=0, max XY >= 1mm, max XY < 300mm.",
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
                    "solver_outlier",
                    "reject_reasons",
                    "delta vs D319",
                ],
                bin_rows,
            ),
            "",
            "## Script-only vs D321 diversity",
            "",
            md_table(
                ["corpus", "accepted", "mean accepted XY", "accepted XY variance", "direction histogram"],
                diversity_rows,
            ),
            "",
            "## Gate interpretation",
            "",
            "- Any bin below 90% pass rate triggers the D321 failable-experiment failure condition.",
            "- `solver_outlier` is a physicality gate, not controller tuning.",
            "- D321 production remains +x only; direction diversification is reserved for D322+ goal-conditioned learning.",
            "",
            f"JSON: `{rel(summary_json)}`",
            f"Accepted rows: `{rel(accepted_csv)}`",
            f"All rows: `{rel(all_csv)}`",
        ]
    )
    (args.out_dir / "d321_data_conveyor_audit_summary.md").write_text(md + "\n")
    print(
        "[d321-audit] done "
        f"rows={len(rows)} accepted={len(accepted_rows)} summary={summary_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
