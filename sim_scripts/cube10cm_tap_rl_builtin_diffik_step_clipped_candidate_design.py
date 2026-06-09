#!/usr/bin/env python3
"""Design audit for 10cm tap built-in DiffIK with 3cm-style step-clipped target application.

This is local-only. It reads repo code and prior local logs. It does not launch
IsaacLab/GPU runtime, generate datasets, train, control RoArm, SSH, or touch B200.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
PATHS = {
    "tap_harness": ROOT / "roarm_rl/test_positive_control_cube_tap10cm.py",
    "tap_env": ROOT / "roarm_rl/roarm_cube_push_env.py",
    "cube3cm_probe": ROOT / "sim_scripts/cube3cm_push_diffik_probe.py",
    "builtin_result_audit": LOG_DIR / "cube10cm_tap_rl_builtin_diffik_parity_result_audit_summary.out",
    "builtin_result_json": LOG_DIR / "cube10cm_tap_rl_builtin_diffik_parity_result_audit.json",
}
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_step_clipped_candidate_design.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_step_clipped_candidate_design_summary.out"


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _line(path: Path, one_based: int) -> str:
    rows = _lines(path)
    return rows[one_based - 1] if 0 < one_based <= len(rows) else ""


def _has(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def _find(path: Path, needle: str) -> dict[str, Any]:
    for idx, text in enumerate(_lines(path), start=1):
        if needle in text:
            return {"line": idx, "text": text.strip()}
    return {"line": None, "text": None}


def main() -> int:
    for key, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")

    harness = PATHS["tap_harness"]
    cube3 = PATHS["cube3cm_probe"]
    env = PATHS["tap_env"]
    previous_audit = json.loads(PATHS["builtin_result_json"].read_text(encoding="utf-8"))

    code_checks = {
        "controller_mode_choice": _has(harness, '"isaac_builtin_diffik_step_clipped_direct_apply"'),
        "builtin_diffik_compute_kept": _has(harness, "joint_pos_des = diffik.compute("),
        "raw_delta": _has(harness, "raw_delta_arm = joint_pos_des - joint_pos_arm"),
        "step_clip": _has(harness, "clipped_delta_arm = torch_mod.clamp(raw_delta_arm"),
        "target_from_clipped_delta": _has(harness, "arm_joint_target = joint_pos_arm + clipped_delta_arm"),
        "direct_override_kept": _has(harness, "inner._external_joint_targets_override = joint_target"),
        "telemetry": all(
            _has(harness, needle)
            for needle in (
                '"builtin_diffik_step_clipped_target_apply"',
                '"builtin_diffik_step_clip_rate"',
                '"builtin_diffik_raw_delta_abs_max_rad"',
                '"builtin_diffik_clipped_delta_abs_max_rad"',
            )
        ),
        "metadata": all(
            _has(harness, needle)
            for needle in (
                '"builtin_diffik_step_clipped_target_apply": args.controller_mode',
                '"builtin_diffik_step_clip_rad": float(args.builtin_diffik_step_clip_rad)',
            )
        ),
        "contact_gate_unchanged": all(
            _has(env, needle)
            for needle in (
                "face_gap >= -float(self.cfg.tap_contact_face_band_m)",
                "success_now = (contact_proxy | self._tap_contact_seen)",
            )
        ),
        "cube3cm_reference_step_clipped": all(
            _has(cube3, needle)
            for needle in (
                "raw_delta = joint_pos_des - joint_pos_arm",
                "clipped_delta = torch.maximum(torch.minimum(raw_delta, max_step), -max_step)",
                "target_full[:, arm_joint_ids] = joint_pos_arm + clipped_delta",
                "inner.robot_dof_targets[:] = target_full",
            )
        ),
    }
    ready = all(code_checks.values())

    out_json = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_sanity.json"
    out_summary = (
        LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_sanity_summary.out"
    )
    command = (
        "conda run -n isaaclab --no-capture-output python -u "
        "-m roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 120 --seed 962 --device cuda:0 "
        "--controller_mode isaac_builtin_diffik_step_clipped_direct_apply "
        "--builtin_diffik_step_clip_rad 0.010 "
        f"--out_json {out_json} --out_summary {out_summary}"
    )

    artifact: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_builtin_diffik_step_clipped_candidate_design_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_design_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "basis": {
            "previous_builtin_direct_apply_primary_blocker": previous_audit["outcome"]["primary_blocker"],
            "previous_builtin_direct_apply_summary_lines": [
                _line(PATHS["builtin_result_audit"], idx) for idx in range(2, 9)
            ],
            "cube3cm_reference_lines": {
                "raw_delta": _find(cube3, "raw_delta = joint_pos_des - joint_pos_arm"),
                "max_step": _find(cube3, 'max_step = traj["max_joint_step"].unsqueeze(-1)'),
                "clipped_delta": _find(cube3, "clipped_delta = torch.maximum"),
                "target_full": _find(cube3, "target_full[:, arm_joint_ids] = joint_pos_arm + clipped_delta"),
                "target_write": _find(cube3, "inner.robot_dof_targets[:] = target_full"),
            },
        },
        "code_checks": {"ready": ready, "checks": code_checks},
        "candidate": {
            "name": "isaac_builtin_diffik_step_clipped_direct_apply_positive_control",
            "status": "DESIGNED_NOT_RUN",
            "changed_from_previous_builtin_direct": "full_joint_pos_des_direct_apply -> step_clipped_joint_target_apply",
            "step_clip_rad": 0.010,
            "unchanged": {
                "controller": "IsaacLab DifferentialIKController",
                "diffik_lambda": 0.010,
                "cube_size_m": 0.100,
                "cube_mass_kg": 0.720,
                "cube_xy_m": [0.250, 0.000],
                "push_dir_xy": [1.0, 0.0],
                "steps": 120,
                "num_envs": 2,
                "seed": 962,
                "geometry": "unchanged",
                "contact_gate": "unchanged_strict",
                "dataset_generation": False,
                "training": False,
                "robot_control": False,
            },
            "out_json": str(out_json),
            "out_summary": str(out_summary),
            "command": command,
        },
        "rejected_before_this_runtime": {
            "contact_gate_relaxation_or_tier_b_exception": "blocked_until_step_clipped_parity_result",
            "dataset_or_rl": "blocked_until_contact_gated_positive_control_passes_or_explicit_noisy_teacher_policy_gate",
            "roarm": "blocked_until_policy_and_safety_gates",
        },
        "outcome": {
            "candidate": "READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY" if ready else "NOT_READY_CODE_CHECK_FAIL",
            "contact_gated_positive_control": "RUN_FAILED",
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "outputs": {"json": str(OUT_JSON), "summary": str(OUT_SUMMARY)},
    }

    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_builtin_diffik_step_clipped_candidate_design_v1 "
        "local_design_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 basis "
        f"previous_blocker={previous_audit['outcome']['primary_blocker']} "
        "three_cm_reference=raw_delta_to_clipped_delta_to_robot_dof_targets",
        "line3 code_ready "
        f"ready={ready} mode_choice={code_checks['controller_mode_choice']} "
        f"builtin_compute={code_checks['builtin_diffik_compute_kept']} raw_delta={code_checks['raw_delta']} "
        f"step_clip={code_checks['step_clip']} target_from_clipped_delta={code_checks['target_from_clipped_delta']} "
        f"direct_override={code_checks['direct_override_kept']} telemetry={code_checks['telemetry']} "
        f"metadata={code_checks['metadata']} contact_gate_unchanged={code_checks['contact_gate_unchanged']}",
        "line4 selected_candidate "
        "status=DESIGNED_NOT_RUN name=isaac_builtin_diffik_step_clipped_direct_apply_positive_control "
        "changed=full_joint_pos_des_direct_apply->step_clipped_joint_target_apply "
        "step_clip_rad=0.010 geometry=UNCHANGED contact_gate=UNCHANGED dataset_rl_roarm=NO",
        "line5 rejected "
        "contact_gate_relaxation_or_tier_b=NO dataset_rl_roarm=NO "
        "reason=must_test_3cm_target_application_parity_before_gate_exception",
        "line6 verdict "
        f"{artifact['outcome']['candidate']} contact_gated_positive_control=RUN_FAILED "
        "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
        "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
        f"line7 command {command}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
