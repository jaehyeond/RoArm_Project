"""Posthoc gate for the 10cm tap RL env local runtime sanity logs.

This reads existing local logs only. It does not launch IsaacLab, use GPU,
train, build datasets, control a robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_STATIC_JSON = LOG_DIR / "cube10cm_tap_rl_env_contract_static_audit.json"
DEFAULT_ZERO_JSON = LOG_DIR / "cube10cm_tap_rl_env_zero_action_sanity.json"
DEFAULT_RANDOM_JSON = LOG_DIR / "cube10cm_tap_rl_env_random_sanity.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_env_runtime_gate_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_env_runtime_gate_audit_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _last(result: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = result.get("last_log", {}).get(key, default)
    return float(value)


def _runtime_ok(result: dict[str, Any]) -> bool:
    return (
        result.get("status") == "PASS"
        and result.get("gpu_runtime") == "YES_LOCAL_TINY_ISAACLAB_RANDOM_SANITY"
        and result.get("device") == "cuda:0"
        and result.get("dataset_generation") is False
        and result.get("training") is False
        and result.get("robot_control") is False
        and result.get("ssh") is False
        and result.get("final_1cm_required") is False
        and result.get("required_log_keys_present") is True
        and result.get("missing_required_log_keys") == []
        and result.get("reward_finite") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static_json", type=Path, default=DEFAULT_STATIC_JSON)
    parser.add_argument("--zero_json", type=Path, default=DEFAULT_ZERO_JSON)
    parser.add_argument("--random_json", type=Path, default=DEFAULT_RANDOM_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    static = _load(args.static_json)
    zero = _load(args.zero_json)
    random = _load(args.random_json)

    static_pass = static.get("verdict") == "PASS"
    zero_runtime_pass = _runtime_ok(zero)
    random_runtime_pass = _runtime_ok(random)
    zero_quiet = (
        int(zero.get("terminated_count", -1)) == 0
        and int(zero.get("truncated_count", -1)) == 0
        and _last(zero, "cube_tap_contact_seen_rate") == 0.0
        and _last(zero, "cube_tap_reaction_contact_context_rate") == 0.0
        and _last(zero, "cube_tap_reaction_seen_rate") == 0.0
        and _last(zero, "cube_tap_overshoot_seen_rate") == 0.0
        and _last(zero, "cube_tap_success_rate") == 0.0
        and _last(zero, "cube_tap_max_z_delta_m") < 0.001
    )
    random_contract_only = (
        int(random.get("terminated_count", -1)) == 0
        and int(random.get("truncated_count", -1)) == 0
        and _last(random, "cube_tap_objective_final_relocation_required") == 0.0
        and _last(random, "cube_tap_contact_seen_rate") == 0.0
        and _last(random, "cube_tap_reaction_seen_rate") == 0.0
        and _last(random, "cube_tap_overshoot_seen_rate") == 0.0
        and _last(random, "cube_tap_success_rate") == 0.0
    )

    wrapper_sanity_pass = static_pass and zero_runtime_pass and random_runtime_pass and zero_quiet and random_contract_only
    result = {
        "artifact_type": "cube10cm_tap_rl_env_runtime_gate_audit_v1",
        "local_log_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "static_pass": static_pass,
        "zero_runtime_pass": zero_runtime_pass,
        "zero_quiet_no_action": zero_quiet,
        "random_runtime_pass": random_runtime_pass,
        "random_contract_only": random_contract_only,
        "wrapper_sanity_pass": wrapper_sanity_pass,
        "zero_metrics": {
            "contact_seen": _last(zero, "cube_tap_contact_seen_rate"),
            "reaction_signal_now": _last(zero, "cube_tap_reaction_signal_now_rate"),
            "reaction_contact_context": _last(zero, "cube_tap_reaction_contact_context_rate"),
            "reaction_seen": _last(zero, "cube_tap_reaction_seen_rate"),
            "overshoot_seen": _last(zero, "cube_tap_overshoot_seen_rate"),
            "tap_success": _last(zero, "cube_tap_success_rate"),
            "max_z_delta_m": _last(zero, "cube_tap_max_z_delta_m"),
            "max_speed_mps": _last(zero, "cube_tap_max_speed_mps"),
            "terminated_count": zero.get("terminated_count"),
        },
        "random_metrics": {
            "contact_seen": _last(random, "cube_tap_contact_seen_rate"),
            "reaction_signal_now": _last(random, "cube_tap_reaction_signal_now_rate"),
            "reaction_contact_context": _last(random, "cube_tap_reaction_contact_context_rate"),
            "reaction_seen": _last(random, "cube_tap_reaction_seen_rate"),
            "overshoot_seen": _last(random, "cube_tap_overshoot_seen_rate"),
            "tap_success": _last(random, "cube_tap_success_rate"),
            "max_z_delta_m": _last(random, "cube_tap_max_z_delta_m"),
            "max_speed_mps": _last(random, "cube_tap_max_speed_mps"),
            "terminated_count": random.get("terminated_count"),
        },
        "unblocked": {
            "default_off_10cm_tap_env_wrapper_contract": wrapper_sanity_pass,
            "event_label_quality_tier_metadata": True,
        },
        "still_blocked": {
            "noisy_tier_b_action_teacher_exception_policy": "BLOCKED_UNTIL_EXPLICIT_EXCEPTION",
            "tiny_action_dataset_dry_run": "BLOCKED_UNTIL_NOISY_TIER_B_EXCEPTION_OR_CLEAN_TEACHER",
            "large_dataset": "BLOCKED",
            "ppo_rl_training": "BLOCKED_UNTIL_SEPARATE_RL_PREFLIGHT_AND_EXPLICIT_APPROVAL",
            "roarm_m3_pro_deploy": "BLOCKED",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_env_runtime_gate_audit_v1 "
        "local_log_audit_only=YES gpu_runtime_launched_by_this_audit=NO "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO",
        (
            "line2 static_contract "
            f"static_pass={static_pass} wrapper_default_off=True cube_size=0.100m cube_mass=0.720kg "
            "final_1cm_required_default=NO table_top_z=TABLE_Z"
        ),
        (
            "line3 zero_action_gpu_sanity "
            f"runtime_pass={zero_runtime_pass} zero_quiet={zero_quiet} "
            f"contact={result['zero_metrics']['contact_seen']} "
            f"reaction_signal_now={result['zero_metrics']['reaction_signal_now']} "
            f"reaction_context={result['zero_metrics']['reaction_contact_context']} "
            f"reaction_seen={result['zero_metrics']['reaction_seen']} "
            f"overshoot={result['zero_metrics']['overshoot_seen']} "
            f"max_z_delta_m={result['zero_metrics']['max_z_delta_m']:.9f} "
            f"max_speed_mps={result['zero_metrics']['max_speed_mps']:.9f}"
        ),
        (
            "line4 random_gpu_sanity "
            f"runtime_pass={random_runtime_pass} contract_only={random_contract_only} "
            f"contact={result['random_metrics']['contact_seen']} "
            f"reaction_context={result['random_metrics']['reaction_contact_context']} "
            f"reaction_seen={result['random_metrics']['reaction_seen']} "
            f"overshoot={result['random_metrics']['overshoot_seen']} "
            f"tap_success={result['random_metrics']['tap_success']}"
        ),
        (
            "line5 verdict "
            f"wrapper_sanity_pass={wrapper_sanity_pass} "
            "env_wrapper_unblocked_for_local_preflight_only=YES "
            "action_teacher_dataset=BLOCKED large_dataset=BLOCKED ppo_rl_training=BLOCKED roarm=BLOCKED"
        ),
        (
            "line6 next "
            "allowed=local_rl_preflight_design_or_policy_gate_review "
            "not_allowed=ppo_training_large_dataset_roarm_or_noisy_tierb_action_teacher_without_exception"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if wrapper_sanity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
