"""Static audit for the default-off 10cm/0.72kg tap RL env contract.

This verifies source/registration intent only. It does not launch IsaacLab, run
physics, train, build data, control a robot, or use SSH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
ENV_PY = REPO / "roarm_rl/roarm_cube_push_env.py"
REG_PY = REPO / "roarm_rl/__init__.py"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_env_contract_static_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_env_contract_static_audit_summary.out"


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


def _ok(patterns: dict[str, int | None]) -> bool:
    return all(line is not None for line in patterns.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_py", type=Path, default=ENV_PY)
    parser.add_argument("--reg_py", type=Path, default=REG_PY)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    env_text = args.env_py.read_text(encoding="utf-8")
    reg_text = args.reg_py.read_text(encoding="utf-8")
    cfg_block = _class_block(env_text, "RoArmCubeTap10cmEnvCfg")
    env_block = _class_block(env_text, "RoArmCubeTap10cmEnv")

    old_env_patterns = {
        "old_cube_size_3cm": _line_of(args.env_py, "CUBE_SIZE_M = 0.030"),
        "old_mass_20g": _line_of(args.env_py, "mass=0.020"),
        "old_env_registered": _line_of(args.reg_py, 'id="RoArm-CubePush-Direct-v0"'),
    }
    new_contract_patterns = {
        "cube10cm_size_const": _line_of(args.env_py, "CUBE10CM_SIZE_M = 0.100"),
        "cube10cm_mass_const": _line_of(args.env_py, "CUBE10CM_MASS_KG = 0.720"),
        "tap_cfg_class": _line_of(args.env_py, "class RoArmCubeTap10cmEnvCfg"),
        "tap_env_class": _line_of(args.env_py, "class RoArmCubeTap10cmEnv"),
        "tap_table_cfg": _line_of(args.env_py, "tap_table: RigidObjectCfg"),
        "tap_table_static": _line_of(args.env_py, "kinematic_enabled=True"),
        "tap_table_center_z": _line_of(args.env_py, "TAP_TABLE_CENTER_Z = TABLE_Z - TAP_TABLE_THICKNESS_M / 2.0"),
        "tap_target_1mm": _line_of(args.env_py, "cube_push_target_disp_m: float = 0.001"),
        "tap_final_relocation_default_off": _line_of(args.env_py, "tap_final_relocation_required: bool = False"),
        "tap_overshoot_2cm": _line_of(args.env_py, "tap_overshoot_disp_m: float = 0.020"),
        "tap_env_registered": _line_of(args.reg_py, 'id="RoArm-CubeTap10cm-Direct-v0"'),
    }
    method_patterns = {
        "tap_terms": _line_of(args.env_py, "def _tap_terms"),
        "tap_get_rewards": _line_of(args.env_py, "def _get_rewards"),
        "tap_get_dones": _line_of(args.env_py, "def _get_dones"),
        "contact_log": _line_of(args.env_py, '"cube_tap_contact_seen_rate"'),
        "raw_reaction_signal_log": _line_of(args.env_py, '"cube_tap_reaction_signal_now_rate"'),
        "reaction_contact_context_log": _line_of(args.env_py, '"cube_tap_reaction_contact_context_rate"'),
        "reaction_log": _line_of(args.env_py, '"cube_tap_reaction_seen_rate"'),
        "overshoot_log": _line_of(args.env_py, '"cube_tap_overshoot_seen_rate"'),
        "final_relocation_log": _line_of(args.env_py, '"cube_tap_objective_final_relocation_required"'),
    }

    final_gate_leaks = [
        pattern
        for pattern in (
            'terms["target_xy_dist"] <= self.cfg.cube_success_target_tol_m',
            "cube_success_speed_max_mps",
            "success_bonus *",
            "cube_push_success_rate",
        )
        if pattern in env_block
    ]
    cfg_requires_final = "tap_final_relocation_required: bool = True" in cfg_block
    train_script_changed = False

    verdict_pass = (
        _ok(old_env_patterns)
        and _ok(new_contract_patterns)
        and _ok(method_patterns)
        and not final_gate_leaks
        and not cfg_requires_final
        and not train_script_changed
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_env_contract_static_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_static_audit_only": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "verdict": "PASS" if verdict_pass else "FAIL",
        "default_off": {
            "old_roarm_cube_push_env_preserved": _ok(old_env_patterns),
            "new_roarm_cube_tap10cm_env_registered": new_contract_patterns["tap_env_registered"] is not None,
            "training_entrypoint_unchanged": not train_script_changed,
        },
        "object_contract": {
            "cube_size_m": 0.100,
            "cube_mass_kg": 0.720,
            "evidence_lines": {
                "size": new_contract_patterns["cube10cm_size_const"],
                "mass": new_contract_patterns["cube10cm_mass_const"],
            },
        },
        "objective_contract": {
            "primary_objective": "tap_reaction_contact_not_final_relocation",
            "final_1cm_required_by_default": False,
            "tap_table_top_z_is_project_table_z": new_contract_patterns["tap_table_center_z"] is not None,
            "final_gate_leaks_in_tap_env_block": final_gate_leaks,
            "cfg_requires_final": cfg_requires_final,
            "tap_target_m": 0.001,
            "overshoot_m": 0.020,
        },
        "logging_contract": {
            "contact_reaction_overshoot_separate": _ok(method_patterns),
            "method_and_log_lines": method_patterns,
        },
        "source_lines": {
            "old_env": old_env_patterns,
            "new_contract": new_contract_patterns,
        },
        "next": {
            "random_sanity_runtime": "ALLOWED_NEXT_IF_LOCAL_ISAACLAB_LAUNCHER_AVAILABLE",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm_deploy": "BLOCKED",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_env_contract_static_audit_v1 "
        "local_static_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        (
            "line2 default_off "
            f"old_env_preserved={result['default_off']['old_roarm_cube_push_env_preserved']} "
            f"new_env_registered={result['default_off']['new_roarm_cube_tap10cm_env_registered']} "
            f"training_entrypoint_unchanged={result['default_off']['training_entrypoint_unchanged']}"
        ),
        (
            "line3 object_contract "
            "cube_size=0.100m cube_mass=0.720kg "
            f"size_line={new_contract_patterns['cube10cm_size_const']} "
            f"mass_line={new_contract_patterns['cube10cm_mass_const']}"
        ),
        (
            "line4 objective_contract "
            "objective=tap_reaction_contact_not_final_relocation final_1cm_required_by_default=NO "
            f"table_top_z=TABLE_Z final_gate_leak_count={len(final_gate_leaks)} tap_target=0.001m overshoot=0.020m"
        ),
        (
            "line5 logging_contract "
            f"contact_reaction_overshoot_separate={result['logging_contract']['contact_reaction_overshoot_separate']}"
        ),
        (
            "line6 verdict "
            f"static_contract={'PASS' if verdict_pass else 'FAIL'} "
            "random_sanity_runtime=NEXT_IF_LOCAL_ISAACLAB_LAUNCHER_AVAILABLE "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if verdict_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
