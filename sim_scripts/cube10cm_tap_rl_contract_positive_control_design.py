"""Freeze the 10cm tap RL contract and design the next positive-control sanity.

This is local design/static audit only. It does not launch IsaacLab, run GPU
physics, build datasets, train, control a robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
ENV_PY = REPO / "roarm_rl/roarm_cube_push_env.py"
SANITY_PY = REPO / "roarm_rl/test_sanity_cube_tap10cm.py"
PREFLIGHT_JSON = LOG_DIR / "cube10cm_tap_rl_preflight_policy_gate.json"
RUNTIME_GATE_JSON = LOG_DIR / "cube10cm_tap_rl_env_runtime_gate_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_contract_positive_control_design.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_contract_positive_control_design_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_of(path: Path, pattern: str) -> int | None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def _class_block(text: str, class_name: str) -> str:
    marker = f"class {class_name}("
    start = text.index(marker)
    next_class = text.find("\nclass ", start + len(marker))
    if next_class < 0:
        return text[start:]
    return text[start:next_class]


def _patterns_present(lines: dict[str, int | None]) -> bool:
    return all(line is not None for line in lines.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_py", type=Path, default=ENV_PY)
    parser.add_argument("--sanity_py", type=Path, default=SANITY_PY)
    parser.add_argument("--preflight_json", type=Path, default=PREFLIGHT_JSON)
    parser.add_argument("--runtime_gate_json", type=Path, default=RUNTIME_GATE_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    env_text = args.env_py.read_text(encoding="utf-8")
    tap_env_block = _class_block(env_text, "RoArmCubeTap10cmEnv")
    preflight = _load(args.preflight_json)
    runtime_gate = _load(args.runtime_gate_json)

    objective_lines = {
        "objective_name": _line_of(args.env_py, 'tap_objective_name: str = "tap_reaction_contact_not_final_relocation"'),
        "final_relocation_default_off": _line_of(args.env_py, "tap_final_relocation_required: bool = False"),
        "tap_target_1mm": _line_of(args.env_py, "tap_reaction_disp_m: float = 0.001"),
        "overshoot_2cm": _line_of(args.env_py, "tap_overshoot_disp_m: float = 0.020"),
        "success_terminate_default_off": _line_of(args.env_py, "tap_success_terminate: bool = False"),
    }
    reaction_lines = {
        "raw_reaction_signal": _line_of(args.env_py, "reaction_signal_now = disp_reaction | z_reaction | speed_reaction | tip_reaction"),
        "contact_context": _line_of(args.env_py, "contact_context = contact_proxy | self._tap_contact_seen"),
        "contact_gated_reaction": _line_of(args.env_py, "reaction_now = contact_context & reaction_signal_now"),
        "overshoot": _line_of(args.env_py, 'overshoot_now = terms["disp_xy"] >= float(self.cfg.tap_overshoot_disp_m)'),
        "success": _line_of(args.env_py, "success_now = (contact_proxy | self._tap_contact_seen) & reaction_now & ~overshoot_now"),
    }
    reward_lines = {
        "progress": _line_of(args.env_py, "self.cfg.push_progress_reward_scale * progress"),
        "contact": _line_of(args.env_py, 'self.cfg.tap_contact_reward_scale * terms["tap_contact_proxy"].float()'),
        "contact_proximity": _line_of(args.env_py, 'self.cfg.tap_contact_proximity_reward_scale * terms["tap_contact_proximity"]'),
        "reaction": _line_of(args.env_py, "self.cfg.tap_reaction_reward_scale * just_succeeded.float()"),
        "transient_disp": _line_of(args.env_py, "self.cfg.tap_transient_disp_reward_scale * transient_disp"),
        "overshoot_penalty": _line_of(args.env_py, "self.cfg.tap_overshoot_penalty_scale * self._tap_overshoot_seen.float()"),
        "tip_penalty": _line_of(args.env_py, 'self.cfg.tap_tip_penalty_scale * terms["tip_angle_deg"]'),
    }
    done_lines = {
        "terminated_on_overshoot": _line_of(args.env_py, "terminated = self._tap_overshoot_seen"),
        "optional_success_terminate": _line_of(args.env_py, "terminated = terminated | self._tap_success_flag"),
        "time_truncation": _line_of(args.env_py, "truncated = self.episode_length_buf >= self.max_episode_length - 1"),
    }
    log_lines = {
        "final_required_flag": _line_of(args.env_py, '"cube_tap_objective_final_relocation_required"'),
        "size": _line_of(args.env_py, '"cube_tap_object_size_m"'),
        "mass": _line_of(args.env_py, '"cube_tap_object_mass_kg"'),
        "contact_seen": _line_of(args.env_py, '"cube_tap_contact_seen_rate"'),
        "reaction_signal": _line_of(args.env_py, '"cube_tap_reaction_signal_now_rate"'),
        "reaction_context": _line_of(args.env_py, '"cube_tap_reaction_contact_context_rate"'),
        "reaction_seen": _line_of(args.env_py, '"cube_tap_reaction_seen_rate"'),
        "overshoot_seen": _line_of(args.env_py, '"cube_tap_overshoot_seen_rate"'),
        "tap_success": _line_of(args.env_py, '"cube_tap_success_rate"'),
        "max_disp": _line_of(args.env_py, '"cube_tap_max_disp_along_m"'),
        "max_z": _line_of(args.env_py, '"cube_tap_max_z_delta_m"'),
        "max_speed": _line_of(args.env_py, '"cube_tap_max_speed_mps"'),
    }
    sanity_lines = {
        "default_cuda": _line_of(args.sanity_py, 'parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")'),
        "local_usd": _line_of(args.sanity_py, "DEFAULT_LOCAL_USD ="),
        "required_logs": _line_of(args.sanity_py, "required_log_keys = {"),
        "final_flag_assert": _line_of(args.sanity_py, "final relocation flag must be 0"),
    }

    final_success_leaks = [
        pattern
        for pattern in (
            'terms["target_xy_dist"] <= self.cfg.cube_success_target_tol_m',
            "cube_push_success_rate",
            "success_bonus *",
        )
        if pattern in tap_env_block
    ]

    reward_contract_frozen = _patterns_present(reward_lines) and not final_success_leaks
    done_contract_frozen = _patterns_present(done_lines)
    log_contract_frozen = _patterns_present(log_lines)
    reaction_contract_frozen = _patterns_present(reaction_lines)
    objective_contract_frozen = _patterns_present(objective_lines)
    wrapper_ready = bool(preflight.get("high_level_verdict", {}).get("may_move_to_local_preflight_design")) is True
    runtime_random_is_not_tap_success = (
        runtime_gate.get("random_metrics", {}).get("contact_seen") == 0.0
        and runtime_gate.get("random_metrics", {}).get("tap_success") == 0.0
    )

    positive_control_design = {
        "status": "DESIGNED_NOT_RUN",
        "runtime_permission": "REQUIRES_EXPLICIT_APPROVAL",
        "env_id": "RoArm-CubeTap10cm-Direct-v0",
        "device": "cuda:0",
        "num_envs": 2,
        "max_steps": 120,
        "local_asset_required": True,
        "controller": "scripted_tcp_differential_ik_to_joint_delta_actions",
        "action_semantics": "normalized_joint_delta_command",
        "trajectory": [
            "reset with cube at 10cm/0.72kg contract pose",
            "move TCP to a conservative precontact point in front of live side face",
            "advance through side-center contact using small joint-delta steps",
            "stop when contact-gated reaction is observed or overshoot guard fires",
        ],
        "pass_criteria": {
            "final_1cm_required_flag": 0.0,
            "contact_seen_rate_gt": 0.0,
            "reaction_contact_context_rate_gt": 0.0,
            "reaction_seen_rate_gt": 0.0,
            "tap_success_rate_gt": 0.0,
            "overshoot_seen_rate_eq": 0.0,
            "terminated_count_eq": 0,
        },
        "fail_criteria": [
            "contact remains 0",
            "reaction signal appears without contact context",
            "overshoot seen",
            "final relocation flag is nonzero",
            "non-finite reward",
            "missing required tap logs",
        ],
    }

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_contract_positive_control_design_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_design_static_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "objective_contract": {
            "frozen": objective_contract_frozen,
            "lines": objective_lines,
        },
        "reaction_contract": {
            "frozen": reaction_contract_frozen,
            "lines": reaction_lines,
        },
        "reward_contract": {
            "frozen": reward_contract_frozen,
            "lines": reward_lines,
            "final_success_leaks": final_success_leaks,
        },
        "done_contract": {
            "frozen": done_contract_frozen,
            "lines": done_lines,
            "success_terminate_default": False,
        },
        "log_contract": {
            "frozen": log_contract_frozen,
            "lines": log_lines,
        },
        "sanity_harness_contract": {
            "frozen_for_design": _patterns_present(sanity_lines),
            "lines": sanity_lines,
        },
        "positive_control_design": positive_control_design,
        "gate_interpretation": {
            "wrapper_ready_for_local_preflight_design": wrapper_ready,
            "random_sanity_is_not_tap_success": runtime_random_is_not_tap_success,
            "after_design_can_request_runtime": False,
            "ppo_unblocked_by_this": False,
            "large_dataset_unblocked_by_this": False,
            "roarm_unblocked_by_this": False,
        },
    }

    verdict_pass = (
        objective_contract_frozen
        and reaction_contract_frozen
        and reward_contract_frozen
        and done_contract_frozen
        and log_contract_frozen
        and wrapper_ready
        and runtime_random_is_not_tap_success
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_contract_positive_control_design_v1 "
        "local_design_static_audit_only=YES gpu_runtime=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 objective_reaction_contract "
            f"objective_frozen={objective_contract_frozen} reaction_contact_gated={reaction_contract_frozen} "
            "final_1cm_required_default=NO tap_target=0.001m overshoot=0.020m"
        ),
        (
            "line3 reward_done_contract "
            f"reward_frozen={reward_contract_frozen} done_frozen={done_contract_frozen} "
            f"final_success_leak_count={len(final_success_leaks)} "
            "terminate_on_overshoot=YES success_terminate_default=NO"
        ),
        (
            "line4 log_contract "
            f"log_frozen={log_contract_frozen} "
            "separate_raw_reaction_contact_context_seen_overshoot_success=YES"
        ),
        (
            "line5 positive_control_design "
            "status=DESIGNED_NOT_RUN requires_explicit_gpu_approval=YES "
            "controller=scripted_tcp_differential_ik_to_joint_delta_actions "
            "env_id=RoArm-CubeTap10cm-Direct-v0 device=cuda:0 num_envs=2 max_steps=120"
        ),
        (
            "line6 pass_fail_gate "
            "pass=contact_seen>0,reaction_context>0,reaction_seen>0,tap_success>0,overshoot=0,final_flag=0 "
            "fail=contact0_or_reaction_without_context_or_overshoot_or_nonfinite_reward"
        ),
        (
            "line7 verdict "
            f"contract_design_ready={verdict_pass} "
            "unblocks=local_positive_control_runtime_request_consideration_only "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED action_teacher=BLOCKED"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)

    return 0 if verdict_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
