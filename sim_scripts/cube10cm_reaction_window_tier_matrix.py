"""Report cube10cm reaction-window tier distribution by direction/workspace.

This is a local posthoc tool. It reads existing reaction-window audit JSONs and
their trace CSVs only. It does not run IsaacLab, train, generate new rollouts,
or create a final training dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_reaction_window_tier_matrix_existing_seeds.json"
DEFAULT_OUT_CSV = LOG_DIR / "cube10cm_reaction_window_tier_matrix_existing_seeds.csv"
TIER_NAMES = (
    "A_CLEAN_DIFFIK_TEACHER",
    "B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH",
    "C_REACTION_VALID_FOLLOW_LAG",
    "REJECTED",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def _resolve_path(raw: Any, *, anchor: Path) -> Path:
    if raw is None or str(raw) == "":
        raise ValueError(f"missing path in {anchor}")
    path = Path(str(raw))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([path, REPO / path, anchor.parent / path.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1))
    return ordered[idx]


def _direction_label(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy) and abs(dx) > 0.5:
        return "x+" if dx > 0.0 else "x-"
    if abs(dy) > 0.5:
        return "y+" if dy > 0.0 else "y-"
    return "unknown"


def _workspace_label(local_x: float, local_y: float, *, x_split_m: float, y_split_m: float) -> str:
    x_part = f"x>={x_split_m:.3f}" if local_x >= x_split_m else f"x<{x_split_m:.3f}"
    y_part = f"y>={y_split_m:.3f}" if local_y >= y_split_m else f"y<{y_split_m:.3f}"
    return f"{x_part},{y_part}"


def _trace_env_metadata(trace_csv: Path, *, x_split_m: float, y_split_m: float) -> dict[int, dict[str, Any]]:
    rows = _load_csv(trace_csv)
    if not rows:
        raise ValueError(f"empty trace CSV: {trace_csv}")
    required = {"env_id", "push_dx", "push_dy", "cube_x_m", "cube_y_m", "env_origin_x_m", "env_origin_y_m"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{trace_csv} missing required columns: {missing}")

    by_env: dict[int, dict[str, Any]] = {}
    for row in rows:
        env_id = _int(row.get("env_id"), -1)
        if env_id in by_env:
            continue
        local_x = _float(row.get("cube_x_m")) - _float(row.get("env_origin_x_m"))
        local_y = _float(row.get("cube_y_m")) - _float(row.get("env_origin_y_m"))
        dx = _float(row.get("push_dx"))
        dy = _float(row.get("push_dy"))
        by_env[env_id] = {
            "env_id": env_id,
            "push_dx": dx,
            "push_dy": dy,
            "direction": _direction_label(dx, dy),
            "cube_x0_local_m": local_x,
            "cube_y0_local_m": local_y,
            "workspace_bin": _workspace_label(local_x, local_y, x_split_m=x_split_m, y_split_m=y_split_m),
        }
    return by_env


def _records_for_audit(path: Path, *, x_split_m: float, y_split_m: float) -> list[dict[str, Any]]:
    audit = _load_json(path)
    if audit.get("branch") != "professor_cube10cm_tap_reaction":
        raise ValueError(f"{path} is not a professor cube10cm tap/reaction audit")
    if audit.get("artifact_type") != "cube10cm_reaction_window_contract_audit_v2":
        raise ValueError(f"{path} is not reaction-window contract audit v2")
    trace_csv = _resolve_path(audit.get("trace_csv"), anchor=path)
    env_meta = _trace_env_metadata(trace_csv, x_split_m=x_split_m, y_split_m=y_split_m)

    records: list[dict[str, Any]] = []
    for window in audit.get("per_window", []):
        if not isinstance(window, dict):
            continue
        env_id = _int(window.get("env_id"), -1)
        meta = env_meta.get(env_id, {})
        tier = str(window.get("quality_tier", "REJECTED"))
        accepted = bool(window.get("accepted"))
        records.append(
            {
                "audit_json": str(path),
                "audit_name": path.name,
                "summary_json": audit.get("summary_json"),
                "trace_csv": str(trace_csv),
                "env_id": env_id,
                "direction": meta.get("direction", "unknown"),
                "workspace_bin": meta.get("workspace_bin", "unknown"),
                "audit_direction": f"{path.name}|{meta.get('direction', 'unknown')}",
                "direction_workspace": f"{meta.get('direction', 'unknown')}|{meta.get('workspace_bin', 'unknown')}",
                "audit_direction_workspace": (
                    f"{path.name}|{meta.get('direction', 'unknown')}|{meta.get('workspace_bin', 'unknown')}"
                ),
                "cube_x0_local_m": meta.get("cube_x0_local_m"),
                "cube_y0_local_m": meta.get("cube_y0_local_m"),
                "push_dx": meta.get("push_dx"),
                "push_dy": meta.get("push_dy"),
                "accepted": accepted,
                "quality_tier": tier,
                "reject_reasons": window.get("reject_reasons", []),
                "anchor_source": window.get("anchor_source"),
                "max_disp_m": _float(window.get("max_disp_m")),
                "max_z_delta_m": _float(window.get("max_z_delta_m")),
                "max_speed_mps": _float(window.get("max_speed_mps")),
                "max_tip_angle_deg": _float(window.get("max_tip_angle_deg")),
                "clip_any_rate": _float(window.get("clip_any_rate")),
                "joint_follow_p95_to_cap": _float(window.get("joint_follow_p95_to_cap")),
            }
        )
    return records


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    accepted = [record for record in records if bool(record.get("accepted"))]
    rejected = [record for record in records if not bool(record.get("accepted"))]
    tier_counts = Counter(str(record.get("quality_tier", "UNKNOWN")) for record in records)
    reject_counts: Counter[str] = Counter()
    for record in rejected:
        reject_counts.update(str(reason) for reason in record.get("reject_reasons", []))

    clip_values = [_float(record.get("clip_any_rate")) for record in accepted]
    follow_values = [_float(record.get("joint_follow_p95_to_cap")) for record in accepted]
    disp_values = [_float(record.get("max_disp_m")) for record in accepted]
    return {
        "candidate_window_count": total,
        "accepted_window_count": len(accepted),
        "rejected_window_count": len(rejected),
        "acceptance_rate": len(accepted) / total if total else 0.0,
        "quality_tier_counts": {tier: tier_counts.get(tier, 0) for tier in TIER_NAMES if tier_counts.get(tier, 0) > 0},
        "rejected_window_reasons": dict(sorted(reject_counts.items())),
        "accepted_clip_any_rate_mean": sum(clip_values) / len(clip_values) if clip_values else 0.0,
        "accepted_clip_any_rate_p95": _p95(clip_values),
        "accepted_follow_p95_to_cap_mean": sum(follow_values) / len(follow_values) if follow_values else 0.0,
        "accepted_follow_p95_to_cap_p95": _p95(follow_values),
        "accepted_max_disp_m_mean": sum(disp_values) / len(disp_values) if disp_values else 0.0,
    }


def _group(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(key, "unknown"))].append(record)
    return {name: _summarize(items) for name, items in sorted(grouped.items())}


def _direction_acceptance_is_config_mixed(records: list[dict[str, Any]], direction: str) -> bool:
    audit_direction = _group(records, "audit_direction")
    rates = {
        round(float(summary.get("acceptance_rate", 0.0)), 6)
        for name, summary in audit_direction.items()
        if name.endswith(f"|{direction}")
    }
    return len(rates) > 1


def _readiness_reasons(records: list[dict[str, Any]], min_windows_per_direction: int) -> list[str]:
    reasons: list[str] = ["local_posthoc_matrix_only_not_a_runtime_or_dataset"]
    overall = _summarize(records)
    if overall["quality_tier_counts"].get("A_CLEAN_DIFFIK_TEACHER", 0) == 0:
        reasons.append("no_tier_a_clean_teacher_windows_in_existing_matrix")
    by_direction = _group(records, "direction")
    canonical_dirs = {"x+", "x-", "y+", "y-"}
    missing_dirs = sorted(canonical_dirs - set(by_direction))
    if missing_dirs:
        reasons.append(f"missing_direction_coverage={','.join(missing_dirs)}")
    for direction in sorted(canonical_dirs & set(by_direction)):
        summary = by_direction[direction]
        if summary["candidate_window_count"] < min_windows_per_direction:
            reasons.append(
                f"direction_{direction}_windows={summary['candidate_window_count']}_lt_{min_windows_per_direction}"
            )
        if summary["acceptance_rate"] < 1.0:
            if _direction_acceptance_is_config_mixed(records, direction):
                reasons.append(
                    f"direction_{direction}_config_mixed_acceptance_rate="
                    f"{summary['acceptance_rate']:.6f}_inspect_audit_direction"
                )
            else:
                reasons.append(f"direction_{direction}_acceptance_rate={summary['acceptance_rate']:.6f}_lt_1")
    return reasons


def _csv_rows(grouped: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_level, groups in grouped.items():
        for group_name, summary in groups.items():
            tier_counts = summary.get("quality_tier_counts", {})
            rows.append(
                {
                    "group_level": group_level,
                    "group": group_name,
                    "candidate_window_count": summary["candidate_window_count"],
                    "accepted_window_count": summary["accepted_window_count"],
                    "rejected_window_count": summary["rejected_window_count"],
                    "acceptance_rate": f"{summary['acceptance_rate']:.9f}",
                    "tier_a": tier_counts.get("A_CLEAN_DIFFIK_TEACHER", 0),
                    "tier_b": tier_counts.get("B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH", 0),
                    "tier_c": tier_counts.get("C_REACTION_VALID_FOLLOW_LAG", 0),
                    "rejected": tier_counts.get("REJECTED", 0),
                    "accepted_clip_any_rate_mean": f"{summary['accepted_clip_any_rate_mean']:.9f}",
                    "accepted_follow_p95_to_cap_p95": f"{summary['accepted_follow_p95_to_cap_p95']:.9f}",
                    "accepted_max_disp_m_mean": f"{summary['accepted_max_disp_m_mean']:.9f}",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_json", type=Path, nargs="*", default=None)
    parser.add_argument("--log_dir", type=Path, default=LOG_DIR)
    parser.add_argument("--audit_glob", default="cube10cm_reaction_window_seed*_audit.json")
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--x_workspace_split_m", type=float, default=0.250)
    parser.add_argument("--y_workspace_split_m", type=float, default=0.000)
    parser.add_argument("--min_windows_per_direction", type=int, default=16)
    args = parser.parse_args()

    audit_paths = args.audit_json
    if not audit_paths:
        audit_paths = sorted(args.log_dir.glob(args.audit_glob))
    if not audit_paths:
        raise ValueError("no reaction-window audit JSONs found")

    records: list[dict[str, Any]] = []
    for path in audit_paths:
        records.extend(
            _records_for_audit(
                path,
                x_split_m=float(args.x_workspace_split_m),
                y_split_m=float(args.y_workspace_split_m),
            )
        )
    if not records:
        raise ValueError("no per-window records found")

    grouped = {
        "audit": _group(records, "audit_name"),
        "audit_direction": _group(records, "audit_direction"),
        "direction": _group(records, "direction"),
        "workspace": _group(records, "workspace_bin"),
        "direction_workspace": _group(records, "direction_workspace"),
        "audit_direction_workspace": _group(records, "audit_direction_workspace"),
    }
    readiness_reasons = _readiness_reasons(records, int(args.min_windows_per_direction))
    report = {
        "artifact_type": "cube10cm_reaction_window_tier_matrix_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "primary_objective": "reaction_window_validity_with_quality_tier_not_final_1cm",
        "inputs": [str(path) for path in audit_paths],
        "contract": {
            "local_posthoc_only": True,
            "runs_isaaclab": False,
            "generates_rollouts_or_dataset": False,
            "final_1cm_relocation_required": False,
            "quality_tiers_are_metadata": True,
            "x_workspace_split_m": float(args.x_workspace_split_m),
            "y_workspace_split_m": float(args.y_workspace_split_m),
        },
        "overall": _summarize(records),
        "by_audit": grouped["audit"],
        "by_audit_direction": grouped["audit_direction"],
        "by_direction": grouped["direction"],
        "by_workspace": grouped["workspace"],
        "by_direction_workspace": grouped["direction_workspace"],
        "by_audit_direction_workspace": grouped["audit_direction_workspace"],
        "dataset_scaleup_readiness": {
            "ready_for_1024_or_data": False,
            "reasons": readiness_reasons,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = _csv_rows(grouped)
        with args.out_csv.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    overall = report["overall"]
    print(
        "tier_matrix line1 "
        f"audits={len(audit_paths)} windows={overall['candidate_window_count']} "
        f"accepted={overall['accepted_window_count']} acceptance_rate={overall['acceptance_rate']:.9f} "
        "ready_for_1024_or_data=NO"
    )
    print(f"tier_matrix line2 quality_tier_counts={overall['quality_tier_counts']}")
    for direction, summary in grouped["direction"].items():
        print(
            "tier_matrix line3 "
            f"direction={direction} windows={summary['candidate_window_count']} "
            f"accepted={summary['accepted_window_count']} acceptance_rate={summary['acceptance_rate']:.9f} "
            f"tiers={summary['quality_tier_counts']}"
        )
    print(f"tier_matrix line4 readiness_reasons={readiness_reasons}")
    print(f"tier_matrix line5 out_json={args.out_json}")
    if args.out_csv is not None:
        print(f"tier_matrix line6 out_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
