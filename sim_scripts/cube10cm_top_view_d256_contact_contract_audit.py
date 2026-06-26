#!/usr/bin/env python3
"""Audit D256 visual-label contact contract against TCP-point contact gates."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    REPO
    / "claudedocs"
    / "runtime_logs"
    / "20260526_cube3cm_push_rollout_probe_20480"
)
D242_ROOT = RUNTIME_ROOT / "cube10cm_top_view_visual_0_999_d242"
DEFAULT_FRAMES_JSONL = D242_ROOT / "frames.jsonl"
DEFAULT_TEACHER_CSV = D242_ROOT / "rl_transition_preflight_d256" / "ppo_actor_prior_teacher_rows_d256.csv"
DEFAULT_OUT_DIR = RUNTIME_ROOT / "d256_contact_contract_audit_d270"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p01": quantile(values, 0.01),
        "p50": quantile(values, 0.50),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def load_teacher_keys(csv_path: Path) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty csv: {csv_path}")
        required = {"episode_index", "frame_index_t"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"missing required columns in {csv_path}: {missing}")
        for row in reader:
            keys.add((int(float(row["episode_index"])), int(float(row["frame_index_t"]))))
    return keys


def audit(frames_jsonl: Path, teacher_csv: Path) -> dict[str, Any]:
    teacher_keys = load_teacher_keys(teacher_csv)
    counts = {
        "rows": 0,
        "tap_contact_proxy": 0,
        "tap_contact_seen": 0,
        "tap_reaction_seen": 0,
        "tap_overshoot_seen": 0,
        "tcp_sphere_055": 0,
        "tcp_point_face_band": 0,
        "aabb_contact_tcp_sphere_false": 0,
        "aabb_contact_tcp_point_false": 0,
        "clean_useful_like": 0,
    }
    tcp_dists: list[float] = []
    face_gaps: list[float] = []
    disp_alongs: list[float] = []
    disp_xys: list[float] = []

    with frames_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["episode_id"]), int(row["frame_id"]))
            if key not in teacher_keys:
                continue
            counts["rows"] += 1

            cube = row["cube_position_world_m"]
            tcp = row["tcp_position_world_m"]
            push = row["push_dir_xy"]
            dx = float(tcp[0]) - float(cube[0])
            dy = float(tcp[1]) - float(cube[1])
            dz = float(tcp[2]) - float(cube[2])
            tcp_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            along = dx * float(push[0]) + dy * float(push[1])
            half_along = 0.05 * (abs(float(push[0])) + abs(float(push[1])))
            face_gap = along + half_along
            lateral_x = dx - along * float(push[0])
            lateral_y = dy - along * float(push[1])
            lateral = math.sqrt(lateral_x * lateral_x + lateral_y * lateral_y)
            vertical = abs(dz)

            tcp_sphere = tcp_dist < 0.055
            tcp_point = (-0.010 <= face_gap <= 0.010) and (lateral <= 0.065) and (vertical <= 0.070)
            aabb = float(row.get("tap_contact_proxy", 0.0)) >= 0.5
            seen = float(row.get("tap_contact_seen", 0.0)) >= 0.5
            reaction = float(row.get("tap_reaction_seen", 0.0)) >= 0.5
            overshoot = float(row.get("tap_overshoot_seen", 0.0)) >= 0.5

            counts["tap_contact_proxy"] += int(aabb)
            counts["tap_contact_seen"] += int(seen)
            counts["tap_reaction_seen"] += int(reaction)
            counts["tap_overshoot_seen"] += int(overshoot)
            counts["tcp_sphere_055"] += int(tcp_sphere)
            counts["tcp_point_face_band"] += int(tcp_point)
            counts["aabb_contact_tcp_sphere_false"] += int(aabb and not tcp_sphere)
            counts["aabb_contact_tcp_point_false"] += int(aabb and not tcp_point)
            counts["clean_useful_like"] += int(seen and reaction and not overshoot)

            tcp_dists.append(tcp_dist)
            face_gaps.append(face_gap)
            disp_alongs.append(float(row.get("tap_disp_along_m", 0.0)))
            disp_xys.append(float(row.get("tap_disp_xy_m", 0.0)))

    total = max(1, counts["rows"])
    return {
        "artifact": "d270_d256_contact_contract_audit",
        "source_frames": _rel(frames_jsonl),
        "teacher_rows": _rel(teacher_csv),
        "counts": counts,
        "rates": {key: float(value / total) for key, value in counts.items() if key != "rows"},
        "quantiles": {
            "tcp_cube_dist_m": stats(tcp_dists),
            "tcp_point_face_gap_m": stats(face_gaps),
            "tap_disp_along_m": stats(disp_alongs),
            "tap_disp_xy_m": stats(disp_xys),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_jsonl", type=Path, default=DEFAULT_FRAMES_JSONL)
    parser.add_argument("--teacher_csv", type=Path, default=DEFAULT_TEACHER_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifact_tag", default="d270")
    args = parser.parse_args()

    summary = audit(args.frames_jsonl, args.teacher_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_dir / f"d256_contact_contract_audit_{args.artifact_tag}.json"
    summary_md = args.out_dir / f"d256_contact_contract_audit_{args.artifact_tag}.md"
    summary["summary_json"] = _rel(summary_json)
    summary["summary_md"] = _rel(summary_md)

    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary_md.write_text(
        "# D256 Contact Contract Audit\n\n"
        f"- rows: `{summary['counts']['rows']}`\n"
        f"- tap_contact_proxy rate: `{summary['rates']['tap_contact_proxy']}`\n"
        f"- tap_contact_seen rate: `{summary['rates']['tap_contact_seen']}`\n"
        f"- tap_reaction_seen rate: `{summary['rates']['tap_reaction_seen']}`\n"
        f"- tcp_sphere_055 rate: `{summary['rates']['tcp_sphere_055']}`\n"
        f"- tcp_point_face_band rate: `{summary['rates']['tcp_point_face_band']}`\n"
        f"- tcp_cube_dist min/p50/max: "
        f"`{summary['quantiles']['tcp_cube_dist_m']['min']}` / "
        f"`{summary['quantiles']['tcp_cube_dist_m']['p50']}` / "
        f"`{summary['quantiles']['tcp_cube_dist_m']['max']}`\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
