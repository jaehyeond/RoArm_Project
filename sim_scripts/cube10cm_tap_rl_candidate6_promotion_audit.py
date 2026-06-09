#!/usr/bin/env python3
"""Posthoc audit for Candidate6 promotion validation.

This script reads only local runtime artifacts. It does not run Isaac Lab,
generate datasets, train policies, control robots, use SSH/B200, or touch
Track A.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_candidate6_promotion_validation_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_candidate6_promotion_validation_audit_summary.out"

BASE_CANDIDATE6 = (
    "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_"
    "h580_ep608_x240_nearface_prevtargetbase_link5aabb_pre040_candidate6"
)

PROMOTION_PREFIX = "cube10cm_tap_rl_candidate6_promotion"

EXPECTED_RUNS = {
    "candidate6_seed962_n2_existing": {
        "stage": "baseline",
        "seed": 962,
        "num_envs": 2,
        "sanity": LOG_DIR / f"{BASE_CANDIDATE6}_sanity.json",
        "detail": LOG_DIR / f"{BASE_CANDIDATE6}_detail_trace.json",
    },
    "stage0a_seed963_n2": {
        "stage": "stage0a_multiseed_fixed_geometry",
        "seed": 963,
        "num_envs": 2,
        "sanity": LOG_DIR / f"{PROMOTION_PREFIX}_stage0a_seed963_n2_sanity.json",
        "detail": LOG_DIR / f"{PROMOTION_PREFIX}_stage0a_seed963_n2_detail_trace.json",
    },
    "stage0a_seed964_n2": {
        "stage": "stage0a_multiseed_fixed_geometry",
        "seed": 964,
        "num_envs": 2,
        "sanity": LOG_DIR / f"{PROMOTION_PREFIX}_stage0a_seed964_n2_sanity.json",
        "detail": LOG_DIR / f"{PROMOTION_PREFIX}_stage0a_seed964_n2_detail_trace.json",
    },
    "stage0a_seed965_n2": {
        "stage": "stage0a_multiseed_fixed_geometry",
        "seed": 965,
        "num_envs": 2,
        "sanity": LOG_DIR / f"{PROMOTION_PREFIX}_stage0a_seed965_n2_sanity.json",
        "detail": LOG_DIR / f"{PROMOTION_PREFIX}_stage0a_seed965_n2_detail_trace.json",
    },
    "stage0b_seed962_n8": {
        "stage": "stage0b_small_env_scale",
        "seed": 962,
        "num_envs": 8,
        "sanity": LOG_DIR / f"{PROMOTION_PREFIX}_stage0b_seed962_n8_sanity.json",
        "detail": LOG_DIR / f"{PROMOTION_PREFIX}_stage0b_seed962_n8_detail_trace.json",
    },
}

ACTION_FIELD_NAMES = {
    "action",
    "actions",
    "teacher_action",
    "policy_action",
    "action_teacher",
    "action_delta",
}

CONTRACT = {
    "cube_size_m": 0.1,
    "cube_mass_kg": 0.72,
    "fixed_cube_x_m": 0.24,
    "fixed_cube_y_m": 0.0,
    "fixed_push_dir_x": 1.0,
    "fixed_push_dir_y": 0.0,
    "controller_mode": "isaac_builtin_diffik_step_clipped_direct_apply",
    "target_path_mode": "near_face_goal",
    "tap_contact_proxy_mode": "link5_collision_aabb",
    "tool_contact_proxy_mode": "hand_tcp",
    "builtin_diffik_target_base_mode": "previous_joint_target",
    "precontact_clearance_m": 0.04,
    "tcp_top_margin_m": -0.05,
    "goal_push_m": 0.006,
    "builtin_diffik_step_clip_rad": 0.01,
    "joint_target_lead_limit_rad": 0.06,
    "arm_stiffness": 80.0,
    "arm_damping": 4.0,
    "arm_effort_limit": 2.5,
    "arm_velocity_limit": 3.14,
    "episode_length_s": 6.08,
    "max_steps": 580,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _close(a: Any, b: float, tol: float = 1.0e-9) -> bool:
    return abs(_num(a, float("nan")) - b) <= tol


def _first_step(rows: list[dict[str, Any]], key: str) -> int | None:
    steps = [int(row["step"]) for row in rows if bool(row.get(key))]
    return min(steps) if steps else None


def _row_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _action_fields(schema: list[str]) -> list[str]:
    fields: list[str] = []
    for field in schema:
        if field in ACTION_FIELD_NAMES:
            fields.append(field)
        elif field.startswith("action_") or field.startswith("actions_"):
            fields.append(field)
        elif field.startswith("teacher_action_") or field.startswith("policy_action_"):
            fields.append(field)
        elif field.endswith("_action") or field.endswith("_actions"):
            fields.append(field)
    return sorted(fields)


def _contract_violations(result: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for key, expected_value in CONTRACT.items():
        actual_value = result.get(key)
        if isinstance(expected_value, float):
            if not _close(actual_value, expected_value):
                violations.append(f"{key}={actual_value!r} expected {expected_value!r}")
        elif actual_value != expected_value:
            violations.append(f"{key}={actual_value!r} expected {expected_value!r}")
    for key in ("seed", "num_envs"):
        if int(result.get(key, -1)) != int(expected[key]):
            violations.append(f"{key}={result.get(key)!r} expected {expected[key]!r}")
    return violations


def _summarize_run(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    missing = [str(spec[key]) for key in ("sanity", "detail") if not spec[key].exists()]
    if missing:
        return {
            "name": name,
            "stage": spec["stage"],
            "seed": spec["seed"],
            "num_envs": spec["num_envs"],
            "available": False,
            "missing": missing,
            "promotion_pass": False,
        }

    result = _load(spec["sanity"])
    detail = _load(spec["detail"])
    rows: list[dict[str, Any]] = detail.get("rows", [])
    schema: list[str] = detail.get("schema", [])
    action_fields = _action_fields(schema)

    reset = result.get("reset_metrics", {})
    last_log = result.get("last_log", {})
    first_contact_step = _first_step(rows, "actual_contact_proxy")
    first_success_step = _first_step(rows, "tap_success_seen")
    actual_inside_rows = _row_count(rows, "actual_contact_proxy")

    checks = {
        "runtime_pass": result.get("status") == "PASS",
        "rl_contact_gated_pass": result.get("rl_contact_gated_positive_control") == "PASS",
        "initial_noncontact": _num(reset.get("initial_contact_proxy_rate")) == 0.0,
        "first_contact_after_reset": first_contact_step is not None and first_contact_step > 0,
        "actual_contact_rows_positive": actual_inside_rows > 0,
        "tap_success_positive": _num(last_log.get("cube_tap_success_rate")) > 0.0,
        "contact_seen_positive": _num(last_log.get("cube_tap_contact_seen_rate")) > 0.0,
        "reaction_context_positive": _num(last_log.get("cube_tap_reaction_contact_context_rate")) > 0.0,
        "professor_reaction_positive": _num(last_log.get("cube_tap_professor_physical_reaction_seen_rate")) > 0.0,
        "no_overshoot": _num(last_log.get("cube_tap_overshoot_seen_rate")) == 0.0,
        "no_termination": int(result.get("terminated_count", 0)) == 0,
        "no_truncation": int(result.get("truncated_count", 0)) == 0,
        "detail_rows_match": int(result.get("reach_trace_detail_row_count", -1)) == len(rows),
        "no_action_fields": not action_fields,
    }
    contract_violations = _contract_violations(result, spec)
    promotion_pass = all(checks.values()) and not contract_violations

    return {
        "name": name,
        "stage": spec["stage"],
        "seed": spec["seed"],
        "num_envs": spec["num_envs"],
        "available": True,
        "sanity": str(spec["sanity"].relative_to(ROOT)),
        "detail": str(spec["detail"].relative_to(ROOT)),
        "status": result.get("status"),
        "rl_contact_gated_positive_control": result.get("rl_contact_gated_positive_control"),
        "checks": checks,
        "contract_violations": contract_violations,
        "promotion_pass": promotion_pass,
        "metrics": {
            "steps_executed": int(result.get("steps_executed", 0)),
            "initial_contact": _num(reset.get("initial_contact_proxy_rate")),
            "initial_face_gap_m": _num(reset.get("initial_face_gap_m")),
            "first_contact_step": first_contact_step,
            "first_success_step": first_success_step,
            "actual_inside_rows": actual_inside_rows,
            "tap_success": _num(last_log.get("cube_tap_success_rate")),
            "contact_seen": _num(last_log.get("cube_tap_contact_seen_rate")),
            "reaction_seen": _num(last_log.get("cube_tap_reaction_seen_rate")),
            "reaction_contact_context": _num(last_log.get("cube_tap_reaction_contact_context_rate")),
            "professor_physical_reaction_seen": _num(
                last_log.get("cube_tap_professor_physical_reaction_seen_rate")
            ),
            "overshoot_seen": _num(last_log.get("cube_tap_overshoot_seen_rate")),
            "max_disp_along_m": _num(last_log.get("cube_tap_max_disp_along_m")),
            "max_speed_mps": _num(last_log.get("cube_tap_max_speed_mps")),
            "terminated_count": int(result.get("terminated_count", 0)),
            "truncated_count": int(result.get("truncated_count", 0)),
            "detail_rows": len(rows),
            "schema_len": len(schema),
            "contains_action_fields": bool(action_fields),
        },
    }


def main() -> int:
    runs = {name: _summarize_run(name, spec) for name, spec in EXPECTED_RUNS.items()}

    stage0a_names = [name for name, run in runs.items() if run["stage"] == "stage0a_multiseed_fixed_geometry"]
    stage0b_names = [name for name, run in runs.items() if run["stage"] == "stage0b_small_env_scale"]
    baseline_pass = runs["candidate6_seed962_n2_existing"].get("promotion_pass", False)
    stage0a_complete = all(runs[name].get("available", False) for name in stage0a_names)
    stage0a_pass = stage0a_complete and all(runs[name].get("promotion_pass", False) for name in stage0a_names)
    stage0b_complete = all(runs[name].get("available", False) for name in stage0b_names)
    stage0b_pass = stage0b_complete and all(runs[name].get("promotion_pass", False) for name in stage0b_names)
    promotion_pass = bool(baseline_pass and stage0a_pass and stage0b_pass)

    artifact = {
        "artifact_type": "cube10cm_tap_rl_candidate6_promotion_validation_audit_v1",
        "local_posthoc_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "contract": CONTRACT,
        "runs": runs,
        "verdict": {
            "baseline_pass": baseline_pass,
            "stage0a_complete": stage0a_complete,
            "stage0a_pass": stage0a_pass,
            "stage0b_complete": stage0b_complete,
            "stage0b_pass": stage0b_pass,
            "candidate6_promotion_validation_pass": promotion_pass,
            "pilot_rl_smoke_design_unblocked": promotion_pass,
            "large_dataset_rl_roarm_unblocked": False,
        },
    }
    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_candidate6_promotion_validation_audit_v1 "
        "local_posthoc_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 contract fixed_candidate6=YES controller=isaac_builtin_diffik_step_clipped_direct_apply "
        "target_base=previous_joint_target tap_contact_proxy_mode=link5_collision_aabb "
        "precontact_clearance_m=0.040 fixed_cube=(0.240,0.000) push_dir=(1.0,0.0)",
    ]
    for idx, (name, run) in enumerate(runs.items(), start=3):
        if not run.get("available", False):
            lines.append(
                f"line{idx} run={name} stage={run['stage']} available=NO "
                f"promotion_pass=NO missing_count={len(run.get('missing', []))}"
            )
            continue
        m = run["metrics"]
        lines.append(
            f"line{idx} run={name} stage={run['stage']} status={run['status']} "
            f"promotion_pass={run['promotion_pass']} seed={run['seed']} num_envs={run['num_envs']} "
            f"initial_contact={m['initial_contact']:.9f} first_contact_step={m['first_contact_step']} "
            f"first_success_step={m['first_success_step']} actual_inside_rows={m['actual_inside_rows']} "
            f"tap_success={m['tap_success']:.9f} contact_seen={m['contact_seen']:.9f} "
            f"overshoot_seen={m['overshoot_seen']:.9f} terminated={m['terminated_count']} "
            f"truncated={m['truncated_count']} contains_action_fields={str(m['contains_action_fields']).lower()} "
            f"contract_violations={len(run['contract_violations'])}"
        )
    line_base = 3 + len(runs)
    lines.extend(
        [
            f"line{line_base} verdict baseline_pass={baseline_pass} "
            f"stage0a_complete={stage0a_complete} stage0a_pass={stage0a_pass} "
            f"stage0b_complete={stage0b_complete} stage0b_pass={stage0b_pass}",
            f"line{line_base + 1} next candidate6_promotion_validation_pass={promotion_pass} "
            f"pilot_rl_smoke_design_unblocked={promotion_pass} "
            "large_dataset_rl_roarm_unblocked=NO action_teacher_dataset=NO",
            f"line{line_base + 2} outputs json={OUT_JSON}",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
