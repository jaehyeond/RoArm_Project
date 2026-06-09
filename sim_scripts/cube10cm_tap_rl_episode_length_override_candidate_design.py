#!/usr/bin/env python3
"""Design audit for the default-off episode-length override h580 repeat.

This is local-only. It checks that the harness can override episode_length_s,
uses the failed h580 truncation audit as basis, and selects one repeat runtime
that keeps the step-clipped h580 contract while changing only episode length
relative to the failed h580 diagnostic. It does not launch IsaacLab/GPU runtime,
generate datasets, train, control RoArm, SSH, or touch B200.
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
    "contract_summary": LOG_DIR / "cube10cm_vs_cube3cm_controller_contract_audit_summary.out",
    "h580_audit": LOG_DIR / "cube10cm_tap_rl_step_clipped_h580_result_audit_summary.out",
    "h580_audit_json": LOG_DIR / "cube10cm_tap_rl_step_clipped_h580_result_audit.json",
}
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_episode_length_override_candidate_design.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_episode_length_override_candidate_design_summary.out"


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

    h580_audit = json.loads(PATHS["h580_audit_json"].read_text(encoding="utf-8"))
    harness = PATHS["tap_harness"]
    env = PATHS["tap_env"]

    code_checks = {
        "episode_arg": _has(harness, 'parser.add_argument("--episode_length_s"'),
        "cfg_override": _has(harness, "cfg.episode_length_s = float(args.episode_length_s)"),
        "summary_metadata": _has(harness, "env_max_episode_length")
        and _has(harness, "episode_length_s"),
        "env_default_1p2": _has(env, "episode_length_s = 1.2"),
        "env_truncation_contract": _has(env, "truncated = self.episode_length_buf >= self.max_episode_length - 1"),
    }
    basis_ok = (
        h580_audit["outcome"]["primary_blocker"]
        == "ENV_EPISODE_LENGTH_1P2S_TRUNCATES_H580_HORIZON_TEST"
        and all(code_checks.values())
    )

    out_json = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_sanity.json"
    out_summary = (
        LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_sanity_summary.out"
    )
    command = (
        "conda run -n isaaclab --no-capture-output python -u "
        "-m roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 580 --seed 962 --device cuda:0 "
        "--controller_mode isaac_builtin_diffik_step_clipped_direct_apply "
        "--closed_loop_push_steps 580 --builtin_diffik_step_clip_rad 0.010 "
        "--episode_length_s 6.08 "
        f"--out_json {out_json} --out_summary {out_summary}"
    )

    artifact: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_episode_length_override_candidate_design_v1",
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
            "previous_primary_blocker": h580_audit["outcome"]["primary_blocker"],
            "contract_summary_lines_2_5": [_line(PATHS["contract_summary"], idx) for idx in range(2, 6)],
            "h580_audit_lines_2_7": [_line(PATHS["h580_audit"], idx) for idx in range(2, 8)],
            "code_evidence": {
                "episode_arg": _find(harness, 'parser.add_argument("--episode_length_s"'),
                "cfg_override": _find(harness, "cfg.episode_length_s = float(args.episode_length_s)"),
                "env_default": _find(env, "episode_length_s = 1.2"),
                "env_truncation": _find(
                    env, "truncated = self.episode_length_buf >= self.max_episode_length - 1"
                ),
            },
        },
        "candidate": {
            "name": "isaac_builtin_diffik_step_clipped_h580_ep608_positive_control",
            "status": "DESIGNED_NOT_RUN",
            "changed_relative_to_failed_h580": {"episode_length_s": "1.2 -> 6.08"},
            "kept_from_h580": {
                "steps": 580,
                "closed_loop_push_steps": 580,
                "controller_mode": "isaac_builtin_diffik_step_clipped_direct_apply",
                "builtin_diffik_step_clip_rad": 0.010,
                "cube_size_m": 0.100,
                "cube_mass_kg": 0.720,
                "cube_xy_m": [0.250, 0.000],
                "push_dir_xy": [1.0, 0.0],
                "geometry": "unchanged",
                "contact_gate": "unchanged_strict",
            },
            "expected_env_max_episode_length": 608,
            "out_json": str(out_json),
            "out_summary": str(out_summary),
            "command": command,
        },
        "rejected": {
            "contact_gate_relaxation": "blocked_until_valid_continuous_horizon_result",
            "geometry_change": "not selected",
            "step_clip_change": "not selected",
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
        "line1 artifact=cube10cm_tap_rl_episode_length_override_candidate_design_v1 "
        "local_design_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 basis "
        f"basis_ok={basis_ok} previous_blocker={h580_audit['outcome']['primary_blocker']} "
        "env_default_episode_length_s=1.2 three_cm_episode_s=6.080",
        "line3 code_ready "
        f"episode_arg={code_checks['episode_arg']} cfg_override={code_checks['cfg_override']} "
        f"summary_metadata={code_checks['summary_metadata']} "
        f"env_default_1p2={code_checks['env_default_1p2']} "
        f"env_truncation_contract={code_checks['env_truncation_contract']}",
        "line4 selected_candidate "
        "status=DESIGNED_NOT_RUN name=isaac_builtin_diffik_step_clipped_h580_ep608_positive_control "
        "changed_relative_to_failed_h580=episode_length_s_1p2_to_6p08 "
        "kept=steps_580_closed_loop_push_steps_580_step_clip_0p010_geometry_unchanged_contact_gate_unchanged",
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
