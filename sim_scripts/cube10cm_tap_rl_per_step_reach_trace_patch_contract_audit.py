#!/usr/bin/env python3
"""Static contract audit for the default-off per-step reach trace patch.

This is local-only. It checks code and existing audit basis, then records the one
allowed tiny repeat command for explicit approval. It does not launch IsaacLab,
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
    "reach_contract_audit_json": (
        LOG_DIR / "cube10cm_tap_rl_target_actual_contact_trajectory_reach_contract_audit.json"
    ),
    "reach_contract_audit_summary": (
        LOG_DIR / "cube10cm_tap_rl_target_actual_contact_trajectory_reach_contract_audit_summary.out"
    ),
}
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_per_step_reach_trace_patch_contract_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_per_step_reach_trace_patch_contract_audit_summary.out"


REQUIRED_SNIPPETS = {
    "reach_trace_arg": 'parser.add_argument("--reach_trace_json", type=Path, default=None)',
    "trace_writer": "def _write_reach_trace(",
    "trace_artifact_type": "cube10cm_tap_rl_per_step_reach_trace_v1",
    "command_target_gap_field": '"command_target_face_gap_m"',
    "applied_target_fk_gap_field": '"applied_joint_target_fk_face_gap_m"',
    "actual_tcp_gap_field": '"actual_tcp_face_gap_m"',
    "joint_follow_field": '"direct_joint_follow_abs_max_rad"',
    "cube_reaction_field": '"professor_physical_reaction_now"',
    "done_flags": '"terminated"',
    "truncated_flag": '"truncated"',
    "action_teacher_false": '"action_teacher_dataset": False',
    "result_trace_metadata": '"reach_trace_row_count": len(reach_trace_rows)',
    "applied_fk_metric_enabled": '"closed_loop_target_fk_err_mm_mean": applied_target_fk_err_mm_mean',
}


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


def main() -> int:
    for key, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")

    harness = PATHS["tap_harness"]
    reach_audit = json.loads(PATHS["reach_contract_audit_json"].read_text(encoding="utf-8"))
    snippet_hits = {name: _find(harness, snippet) for name, snippet in REQUIRED_SNIPPETS.items()}
    code_ready = all(item["line"] is not None for item in snippet_hits.values())
    basis_ok = reach_audit["outcome"]["verdict"] == "REACH_TRACE_CONTRACT_GAP_IDENTIFIED"

    out_json = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_sanity.json"
    out_summary = (
        LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_sanity_summary.out"
    )
    reach_trace_json = (
        LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_trace.json"
    )
    command = (
        "conda run -n isaaclab --no-capture-output python -u "
        "-m roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 580 --seed 962 --device cuda:0 "
        "--controller_mode isaac_builtin_diffik_step_clipped_direct_apply "
        "--closed_loop_push_steps 580 --builtin_diffik_step_clip_rad 0.010 "
        "--episode_length_s 6.08 "
        f"--reach_trace_json {reach_trace_json} "
        f"--out_json {out_json} --out_summary {out_summary}"
    )

    artifact: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_per_step_reach_trace_patch_contract_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_static_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "basis": {
            "basis_ok": basis_ok,
            "reach_contract_verdict": reach_audit["outcome"]["verdict"],
            "reach_contract_summary_lines_6_9": [
                _line(PATHS["reach_contract_audit_summary"], idx) for idx in range(6, 10)
            ],
        },
        "code_checks": {
            "code_ready": code_ready,
            "snippet_hits": snippet_hits,
        },
        "trace_contract": {
            "default_off": True,
            "enabled_only_by": "--reach_trace_json",
            "separate_json": str(reach_trace_json),
            "not_action_teacher_dataset": True,
            "no_action_fields_for_dataset_training": True,
            "required_fields_present": sorted(REQUIRED_SNIPPETS),
        },
        "candidate_runtime": {
            "status": "DESIGNED_NOT_RUN_REQUIRES_EXPLICIT_APPROVAL",
            "changed_relative_to_ep608": "adds_reach_trace_json_only",
            "kept": {
                "num_envs": 2,
                "steps": 580,
                "seed": 962,
                "device": "cuda:0",
                "controller_mode": "isaac_builtin_diffik_step_clipped_direct_apply",
                "closed_loop_push_steps": 580,
                "builtin_diffik_step_clip_rad": 0.010,
                "episode_length_s": 6.08,
                "geometry": "unchanged",
                "contact_gate": "unchanged_strict",
            },
            "out_json": str(out_json),
            "out_summary": str(out_summary),
            "reach_trace_json": str(reach_trace_json),
            "command": command,
        },
        "outcome": {
            "patch_contract": "READY_LOCAL_ONLY" if code_ready and basis_ok else "NOT_READY_RECHECK_CODE",
            "runtime_approval_required": True,
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
        "line1 artifact=cube10cm_tap_rl_per_step_reach_trace_patch_contract_audit_v1 "
        "local_static_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 basis "
        f"basis_ok={basis_ok} reach_contract_verdict={reach_audit['outcome']['verdict']} "
        "next_from_prior=patch_default_off_per_step_reach_trace",
        "line3 code_ready "
        f"code_ready={code_ready} reach_trace_arg={snippet_hits['reach_trace_arg']['line']} "
        f"trace_writer={snippet_hits['trace_writer']['line']} "
        f"applied_fk_metric={snippet_hits['applied_fk_metric_enabled']['line']} "
        f"row_count_metadata={snippet_hits['result_trace_metadata']['line']}",
        "line4 schema "
        "fields=command_target_gap,applied_joint_target_fk_gap,actual_tcp_gap,joint_follow,"
        "cube_reaction,done_flags action_teacher_dataset=False default_off=True separate_json=True",
        "line5 candidate_runtime "
        "status=DESIGNED_NOT_RUN_REQUIRES_EXPLICIT_APPROVAL "
        "changed=reach_trace_json_only kept=h580_ep608_step_clipped_geometry_contact_gate",
        "line6 verdict "
        f"patch_contract={artifact['outcome']['patch_contract']} runtime_approval_required=True "
        "contact_gated_positive_control=RUN_FAILED diffik_action_dataset=BLOCKED "
        "tiny_action_dataset_dry_run=BLOCKED ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
        f"line7 command {command}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if artifact["outcome"]["patch_contract"] == "READY_LOCAL_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
