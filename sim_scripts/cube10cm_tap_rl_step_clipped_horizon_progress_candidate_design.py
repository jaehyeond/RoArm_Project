#!/usr/bin/env python3
"""Design audit for the 10cm step-clipped DiffIK horizon/progress candidate.

This is local-only. It reads existing code and logs, then selects a single
3cm-parity horizon/progress runtime candidate. It does not launch IsaacLab/GPU
runtime, generate datasets, train, control RoArm, SSH, or touch B200.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
PATHS = {
    "contract_summary": LOG_DIR / "cube10cm_vs_cube3cm_controller_contract_audit_summary.out",
    "step_clipped_audit": LOG_DIR / "cube10cm_tap_rl_builtin_diffik_step_clipped_result_audit_summary.out",
    "step_clipped_json": LOG_DIR / "cube10cm_tap_rl_builtin_diffik_step_clipped_result_audit.json",
    "cube3cm_probe": ROOT / "sim_scripts/cube3cm_push_diffik_probe.py",
    "tap_harness": ROOT / "roarm_rl/test_positive_control_cube_tap10cm.py",
}
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_step_clipped_horizon_progress_candidate_design.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_step_clipped_horizon_progress_candidate_design_summary.out"


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


def _target_plan(push_steps: int, max_steps: int) -> dict[str, Any]:
    pre_face_gap = -0.020
    through_face_gap = 0.106
    span = through_face_gap - pre_face_gap
    band = 0.010
    inside_steps: list[int] = []
    for step in range(max_steps):
        alpha = min(1.0, max(0.0, float(step + 1) / max(float(push_steps), 1.0)))
        face_gap = pre_face_gap + alpha * span
        if -band <= face_gap <= band:
            inside_steps.append(step + 1)
    final_alpha = min(1.0, max(0.0, float(max_steps) / max(float(push_steps), 1.0)))
    return {
        "max_steps": max_steps,
        "closed_loop_push_steps": push_steps,
        "runtime_s_at_dt_0p01": max_steps * 0.01,
        "target_final_alpha": final_alpha,
        "target_final_face_gap_m": pre_face_gap + final_alpha * span,
        "target_face_gap_rate_m_per_step": span / max(float(push_steps), 1.0),
        "target_inside_band_step_count": len(inside_steps),
        "target_inside_band_first_step_1based": inside_steps[0] if inside_steps else None,
        "target_inside_band_last_step_1based": inside_steps[-1] if inside_steps else None,
    }


def main() -> int:
    for key, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")

    step_audit = json.loads(PATHS["step_clipped_json"].read_text(encoding="utf-8"))
    current = _target_plan(push_steps=72, max_steps=120)
    candidate = _target_plan(push_steps=580, max_steps=580)
    slow240 = _target_plan(push_steps=240, max_steps=120)

    code_ready = all(
        _has(PATHS["tap_harness"], needle)
        for needle in (
            '"isaac_builtin_diffik_step_clipped_direct_apply"',
            "--builtin_diffik_step_clip_rad",
            "raw_delta_arm = joint_pos_des - joint_pos_arm",
            "clipped_delta_arm = torch_mod.clamp(raw_delta_arm",
            "step_clip_rad=float(args.builtin_diffik_step_clip_rad)",
        )
    )
    three_cm_reference_ok = all(
        _has(PATHS["cube3cm_probe"], needle)
        for needle in (
            "DifferentialIKControllerCfg(",
            "joint_pos_des = diffik.compute(",
            "clipped_delta = torch.maximum(torch.minimum(raw_delta, max_step), -max_step)",
            "inner.robot_dof_targets[:] = target_full",
        )
    )
    basis_ok = (
        step_audit["outcome"]["primary_blocker"]
        == "STEP_CLIPPED_DIFFIK_TARGET_APPLICATION_HORIZON_OR_PROGRESS_TOO_SHORT"
        and code_ready
        and three_cm_reference_ok
    )

    out_json = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_sanity.json"
    out_summary = (
        LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_sanity_summary.out"
    )
    command = (
        "conda run -n isaaclab --no-capture-output python -u "
        "-m roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 580 --seed 962 --device cuda:0 "
        "--controller_mode isaac_builtin_diffik_step_clipped_direct_apply "
        "--closed_loop_push_steps 580 --builtin_diffik_step_clip_rad 0.010 "
        f"--out_json {out_json} --out_summary {out_summary}"
    )

    artifact: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_step_clipped_horizon_progress_candidate_design_v1",
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
            "basis_ok": basis_ok,
            "contract_summary_lines_2_9": [_line(PATHS["contract_summary"], idx) for idx in range(2, 10)],
            "step_clipped_audit_lines_2_7": [_line(PATHS["step_clipped_audit"], idx) for idx in range(2, 8)],
            "cube3cm_reference_lines": {
                "diffik_cfg": _find(PATHS["cube3cm_probe"], "DifferentialIKControllerCfg("),
                "compute": _find(PATHS["cube3cm_probe"], "joint_pos_des = diffik.compute("),
                "clipped_delta": _find(PATHS["cube3cm_probe"], "clipped_delta = torch.maximum"),
                "target_write": _find(PATHS["cube3cm_probe"], "inner.robot_dof_targets[:] = target_full"),
            },
        },
        "plans": {
            "current_step_clipped_10cm": current,
            "slow240_direct_apply_reference": slow240,
            "selected_h580_candidate": candidate,
        },
        "candidate": {
            "name": "isaac_builtin_diffik_step_clipped_h580_positive_control",
            "status": "DESIGNED_NOT_RUN",
            "changed_as_one_contract": {
                "steps": "120 -> 580",
                "closed_loop_push_steps": "72 -> 580",
            },
            "unchanged": {
                "controller_mode": "isaac_builtin_diffik_step_clipped_direct_apply",
                "builtin_diffik_step_clip_rad": 0.010,
                "builtin_diffik_lambda": 0.010,
                "cube_size_m": 0.100,
                "cube_mass_kg": 0.720,
                "cube_xy_m": [0.250, 0.000],
                "push_dir_xy": [1.0, 0.0],
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
        "rejected": {
            "contact_gate_relaxation": "blocked_until_horizon_progress_result",
            "geometry_change": "not selected before horizon parity is tested",
            "step_clip_change": "not selected; user asked steps/closed_loop_push_steps",
            "dataset_or_rl": "blocked",
            "roarm": "blocked",
        },
        "outcome": {
            "candidate": "READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY" if basis_ok else "NOT_READY_RECHECK_BASIS",
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
        "line1 artifact=cube10cm_tap_rl_step_clipped_horizon_progress_candidate_design_v1 "
        "local_design_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 basis "
        f"basis_ok={basis_ok} previous_blocker={step_audit['outcome']['primary_blocker']} "
        "three_cm_steps=580 three_cm_episode_s=6.080 ten_cm_current_steps=120 ten_cm_current_s=1.200",
        "line3 current_vs_candidate "
        f"current_inside_steps={current['target_inside_band_step_count']} "
        f"current_inside_first_last={current['target_inside_band_first_step_1based']}:"
        f"{current['target_inside_band_last_step_1based']} "
        f"candidate_inside_steps={candidate['target_inside_band_step_count']} "
        f"candidate_inside_first_last={candidate['target_inside_band_first_step_1based']}:"
        f"{candidate['target_inside_band_last_step_1based']} "
        f"candidate_runtime_s={candidate['runtime_s_at_dt_0p01']:.3f}",
        "line4 selected_candidate "
        "status=DESIGNED_NOT_RUN name=isaac_builtin_diffik_step_clipped_h580_positive_control "
        "changed_as_one_contract=steps_120_to_580_and_closed_loop_push_steps_72_to_580 "
        "controller=isaac_builtin_diffik_step_clipped_direct_apply step_clip_rad=0.010 "
        "geometry=UNCHANGED contact_gate=UNCHANGED",
        "line5 rejected "
        "contact_gate_relaxation=NO geometry_change=NO step_clip_change=NO dataset_rl_roarm=NO",
        "line6 verdict "
        f"{artifact['outcome']['candidate']} contact_gated_positive_control=RUN_FAILED "
        "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
        "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
        f"line7 command {command}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if basis_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
