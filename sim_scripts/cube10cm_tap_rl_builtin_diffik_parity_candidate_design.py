#!/usr/bin/env python3
"""Design audit for the 10cm tap built-in DifferentialIKController parity runtime.

This is local-only. It reads repo docs, harness code, IsaacLab source, and prior
local logs. It does not launch IsaacLab/GPU runtime, generate datasets, train,
control RoArm, SSH, or touch B200.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
MEMORY = Path("/home/cgxr/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/MEMORY.md")

PATHS = {
    "claude": ROOT / "CLAUDE.md",
    "start_here": ROOT / "START_HERE.md",
    "decisions": ROOT / "claudedocs/DECISIONS.md",
    "ledger": ROOT / "claudedocs/EXPERIMENT_LEDGER.md",
    "session": ROOT / "claudedocs/session_20260608_cube10cm_tap_rl_preflight_policy_gate.md",
    "memory": MEMORY,
    "tap_harness": ROOT / "roarm_rl/test_positive_control_cube_tap10cm.py",
    "tap_env": ROOT / "roarm_rl/roarm_cube_push_env.py",
    "cube3cm_probe": ROOT / "sim_scripts/cube3cm_push_diffik_probe.py",
    "cube10cm_probe": ROOT / "sim_scripts/cube10cm_push_diffik_probe.py",
    "isaac_task_space": Path(
        "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/"
        "source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py"
    ),
    "direct_telemetry_audit": LOG_DIR / "cube10cm_tap_rl_direct_ik_telemetry_result_audit_summary.out",
    "slow240_audit": LOG_DIR / "cube10cm_tap_rl_slow240_result_audit_summary.out",
    "contract_audit": LOG_DIR / "cube10cm_vs_cube3cm_controller_contract_audit_summary.out",
}

OUT_JSON = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_parity_candidate_design.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_parity_candidate_design_summary.out"


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _line(path: Path, one_based: int) -> str:
    rows = _lines(path)
    return rows[one_based - 1] if 0 < one_based <= len(rows) else ""


def _find(path: Path, needle: str) -> dict[str, Any]:
    for idx, text in enumerate(_lines(path), start=1):
        if needle in text:
            return {"line": idx, "text": text.strip()}
    return {"line": None, "text": None}


def _has(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def _summary_lines(path: Path, start: int, end: int) -> list[str]:
    return [_line(path, idx) for idx in range(start, end + 1)]


def main() -> int:
    for key, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")

    harness = PATHS["tap_harness"]
    env = PATHS["tap_env"]
    isaac_task_space = PATHS["isaac_task_space"]
    direct_json = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_direct_apply_sanity.json"
    direct_summary = (
        LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_direct_apply_sanity_summary.out"
    )
    command = (
        "conda run -n isaaclab --no-capture-output python -u "
        "-m roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 120 --seed 962 --device cuda:0 "
        "--controller_mode isaac_builtin_diffik_direct_apply "
        f"--out_json {direct_json} --out_summary {direct_summary}"
    )

    code_checks = {
        "controller_mode_choice": _has(harness, '"isaac_builtin_diffik_direct_apply"'),
        "builtin_diffik_import": _has(harness, "DifferentialIKController, DifferentialIKControllerCfg"),
        "builtin_diffik_cfg_position_abs_dls": all(
            _has(harness, needle)
            for needle in (
                'command_type="position"',
                "use_relative_mode=False",
                'ik_method="dls"',
                'ik_params={"lambda_val": float(args.builtin_diffik_lambda)}',
            )
        ),
        "live_jacobian": _has(harness, "root_physx_view.get_jacobians()"),
        "base_frame_transform": all(
            _has(harness, needle)
            for needle in (
                "subtract_frame_transforms(root_pos_w, root_quat_w",
                "base_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))",
                "torch_mod.bmm(base_rot_matrix",
            )
        ),
        "tcp_tool_proxy_offset": _has(harness, '"tool_proxy_local": inner._tcp_local.unsqueeze(0).repeat'),
        "diffik_compute": all(
            _has(harness, needle) for needle in ("diffik.set_command(", "diffik.compute(")
        ),
        "direct_joint_target_override": _has(harness, "inner._external_joint_targets_override = joint_target"),
        "metadata": all(
            _has(harness, needle)
            for needle in (
                '"isaac_builtin_diffik_controller_apply": args.controller_mode',
                '"builtin_diffik_lambda": float(args.builtin_diffik_lambda)',
                '"direct_ik_joint_target_apply": args.controller_mode',
            )
        ),
        "contact_gate_unchanged": all(
            _has(env, needle)
            for needle in (
                "face_gap >= -float(self.cfg.tap_contact_face_band_m)",
                "success_now = (contact_proxy | self._tap_contact_seen)",
            )
        ),
    }
    ready = all(code_checks.values())

    artifact: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_builtin_diffik_parity_candidate_design_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_design_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {key: str(path) for key, path in PATHS.items()},
        "memory_checked_but_repo_docs_primary": {
            "memory_line_74": _line(PATHS["memory"], 74),
            "claude_current_state_lines": [
                _line(PATHS["claude"], idx) for idx in (5, 14, 18, 23, 31, 53, 55)
            ],
        },
        "basis": {
            "original_10cm_transition_preserved_builtin_diffik": _has(
                PATHS["decisions"], "Use the IsaacLab built-in `DifferentialIKController` probe path"
            ),
            "cube10cm_probe_wraps_cube3cm_diffik_engine": _has(
                PATHS["cube10cm_probe"], "from sim_scripts import cube3cm_push_diffik_probe as shared_probe"
            ),
            "cube3cm_probe_uses_builtin_diffik": all(
                _has(PATHS["cube3cm_probe"], needle)
                for needle in (
                    "from isaaclab.controllers import DifferentialIKController",
                    "diffik.compute",
                    "inner.robot_dof_targets[:] = target_full",
                    '"env_joint_delta_action_loop_bypassed": True',
                )
            ),
            "isaac_task_space_reference": [
                _find(isaac_task_space, "self._ik_controller = DifferentialIKController"),
                _find(isaac_task_space, "joint_pos_des = self._ik_controller.compute"),
                _find(isaac_task_space, "self._asset.set_joint_position_target(joint_pos_des"),
            ],
            "prior_direct_apply_failed": _summary_lines(PATHS["direct_telemetry_audit"], 2, 6),
            "slow240_failed": _summary_lines(PATHS["slow240_audit"], 2, 7),
            "controller_contract_gap": _summary_lines(PATHS["contract_audit"], 5, 9),
        },
        "code_checks": {
            "ready": ready,
            "checks": code_checks,
            "evidence_lines": {
                "mode_choice": _find(harness, '"isaac_builtin_diffik_direct_apply"'),
                "helper": _find(harness, "def _init_builtin_diffik_state"),
                "controller_cfg": _find(harness, "DifferentialIKControllerCfg("),
                "live_jacobian": _find(harness, "root_physx_view.get_jacobians()"),
                "set_command": _find(harness, "diffik.set_command("),
                "compute": _find(harness, "joint_pos_des = diffik.compute("),
                "override": _find(harness, "inner._external_joint_targets_override = joint_target"),
                "metadata_apply": _find(harness, '"isaac_builtin_diffik_controller_apply": args.controller_mode'),
                "contact_gate": _find(env, "success_now = (contact_proxy | self._tap_contact_seen)"),
            },
        },
        "candidate": {
            "name": "isaac_builtin_diffik_direct_apply_positive_control",
            "status": "DESIGNED_NOT_RUN",
            "runtime_requires_explicit_approval": True,
            "changed_controller": "local_ik_dls_direct_apply -> isaac_builtin_diffik_direct_apply",
            "changed_knobs": ["controller implementation only"],
            "unchanged": {
                "cube_size_m": 0.100,
                "cube_mass_kg": 0.720,
                "cube_xy_m": [0.250, 0.000],
                "push_dir_xy": [1.0, 0.0],
                "num_envs": 2,
                "steps": 120,
                "seed": 962,
                "contact_gate": "unchanged_strict",
                "geometry": "unchanged",
                "action_wrapper_knobs": "unchanged_bypassed_as_existing_direct_apply",
                "dataset_generation": False,
                "training": False,
                "robot_control": False,
            },
            "builtin_diffik_lambda": 0.010,
            "out_json": str(direct_json),
            "out_summary": str(direct_summary),
            "command": command,
        },
        "rejected_before_this_runtime": {
            "contact_gate_relaxation_or_tier_b_exception": "blocked_until_builtin_diffik_parity_runtime_result",
            "slow360_or_timing_sweep": "blocked_until_controller_mismatch_checked",
            "lead_cap_action_scale_sweep": "blocked_until_controller_mismatch_checked",
            "dataset_or_rl": "blocked_until_contact_gated_positive_control_passes_or_explicit_noisy_teacher_policy_gate",
            "roarm": "blocked_until_policy_and_safety_gates",
        },
        "outcome": {
            "parity_candidate": "READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY" if ready else "NOT_READY_CODE_CHECK_FAIL",
            "contact_gated_positive_control": "RUN_FAILED",
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "outputs": {"json": str(OUT_JSON), "summary": str(OUT_SUMMARY)},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_builtin_diffik_parity_candidate_design_v1 "
        "local_design_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 basis "
        f"original_10cm_transition_builtin_diffik={artifact['basis']['original_10cm_transition_preserved_builtin_diffik']} "
        f"cube10cm_wraps_cube3cm_diffik={artifact['basis']['cube10cm_probe_wraps_cube3cm_diffik_engine']} "
        f"cube3cm_builtin_diffik={artifact['basis']['cube3cm_probe_uses_builtin_diffik']}",
        "line3 code_ready "
        f"ready={ready} "
        f"mode_choice={code_checks['controller_mode_choice']} "
        f"builtin_import={code_checks['builtin_diffik_import']} "
        f"position_abs_dls={code_checks['builtin_diffik_cfg_position_abs_dls']} "
        f"live_jacobian={code_checks['live_jacobian']} "
        f"base_frame_transform={code_checks['base_frame_transform']} "
        f"tcp_tool_proxy_offset={code_checks['tcp_tool_proxy_offset']} "
        f"direct_override={code_checks['direct_joint_target_override']} "
        f"metadata={code_checks['metadata']}",
        "line4 selected_candidate "
        "status=DESIGNED_NOT_RUN name=isaac_builtin_diffik_direct_apply_positive_control "
        "changed_controller=local_ik_dls_direct_apply->isaac_builtin_diffik_direct_apply "
        "changed_knobs=controller_implementation_only "
        "geometry=UNCHANGED contact_gate=UNCHANGED action_wrapper_knobs=UNCHANGED_BYPASSED",
        "line5 rejected "
        "contact_gate_relaxation_or_tier_b=NO slow360=NO lead_cap_action_scale_sweep=NO "
        "dataset_rl_roarm=NO reason=must_test_builtin_diffik_parity_before_hiding_controller_mismatch",
        "line6 verdict "
        f"{artifact['outcome']['parity_candidate']} "
        "contact_gated_positive_control=RUN_FAILED diffik_action_dataset=BLOCKED "
        "tiny_action_dataset_dry_run=BLOCKED ppo_rl_training=BLOCKED "
        "large_dataset=BLOCKED roarm=BLOCKED runtime_requires_explicit_approval=YES",
        f"line7 command {command}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
