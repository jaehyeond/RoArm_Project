"""Design the next target-application positive-control candidate after cap040.

This is local design/static audit only. It reads existing logs and source files;
it does not launch IsaacLab, run GPU physics, build datasets, train, control a
robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
CAP040_AUDIT = LOG_DIR / "cube10cm_tap_rl_cap040_final_result_audit.json"
HARNESS = REPO / "roarm_rl/test_positive_control_cube_tap10cm.py"
ENV_SOURCE = REPO / "roarm_rl/roarm_cube_push_env.py"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_target_application_candidate_design.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_target_application_candidate_design_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_of(path: Path, pattern: str) -> int | None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap040_audit", type=Path, default=CAP040_AUDIT)
    parser.add_argument("--harness", type=Path, default=HARNESS)
    parser.add_argument("--env_source", type=Path, default=ENV_SOURCE)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    cap040 = _load(args.cap040_audit)
    code_lines = {
        "env_joint_target_lead_default": _line_of(args.env_source, "joint_target_lead_limit_rad: float = 0.060"),
        "env_joint_delta_reference_default": _line_of(args.env_source, 'joint_delta_reference: str = "target"'),
        "env_target_application_start": _line_of(args.env_source, "reference = str(getattr(self.cfg, \"joint_delta_reference\""),
        "harness_lead_override_arg": _line_of(args.harness, "--joint_target_lead_limit_rad"),
        "harness_lead_override_apply": _line_of(args.harness, "cfg.joint_target_lead_limit_rad ="),
        "harness_summary_lead": _line_of(args.harness, "joint_target_lead_limit_rad="),
    }
    code_ready = all(value is not None for value in code_lines.values())
    basis_ok = (
        bool(cap040.get("cap_only_falsified_as_primary"))
        and bool(cap040.get("lead_limit_observed"))
        and cap040.get("contact_seen") == 0.0
        and cap040.get("tap_success") == 0.0
        and bool(cap040.get("cap_no_longer_active"))
    )

    next_json = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap040_lead120_sanity.json"
    next_summary = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap040_lead120_sanity_summary.out"
    command = (
        "conda run -n isaaclab --no-capture-output python -u -m "
        "roarm_rl.test_positive_control_cube_tap10cm "
        "--num_envs 2 --steps 120 --seed 962 --device cuda:0 "
        "--controller_mode external_closed_loop "
        "--max_joint_delta_per_step_rad 0.040 "
        "--joint_target_lead_limit_rad 0.120 "
        f"--out_json {next_json} --out_summary {next_summary}"
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_target_application_candidate_design_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_design_static_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "code_ready": code_ready,
        "basis_ok": basis_ok,
        "basis": {
            "cap_only_falsified_as_primary": cap040.get("cap_only_falsified_as_primary"),
            "cap_no_longer_active": cap040.get("cap_no_longer_active"),
            "lead_limit_observed": cap040.get("lead_limit_observed"),
            "contact_seen": cap040.get("contact_seen"),
            "tap_success": cap040.get("tap_success"),
            "best_shortfall_m": cap040.get("cap040_best_shortfall_m"),
            "target_lead_abs_max_trace": cap040.get("target_lead_abs_max_trace"),
            "target_lead_limit_rate_trace": cap040.get("target_lead_limit_rate_trace"),
        },
        "selected_candidate": {
            "status": "DESIGNED_NOT_RUN",
            "baseline": "cap040_positive_control",
            "changed_knobs_vs_cap040": 1,
            "joint_target_lead_limit_rad": "0.060->0.120",
            "max_joint_delta_per_step_rad": "0.040_fixed",
            "joint_delta_reference": "target_fixed",
            "action_scale": "0.04_fixed",
            "action_smoothing_alpha": "0.25_fixed",
            "controller_mode": "external_closed_loop_fixed",
            "geometry": "FIXED",
        },
        "not_selected": {
            "joint_delta_reference": "would change target-base semantics and may require matching harness action-base logic",
            "action_scale": "changes command normalization",
            "geometry_goal_push_or_contact_band": "changes contact geometry",
            "smoothing": "already falsified as root cause",
            "cap_only": "falsified as primary by cap040",
        },
        "pass_gate": {
            "contact_seen": ">0",
            "reaction_context": ">0",
            "reaction_seen": ">0",
            "tap_success": ">0",
            "overshoot": "0",
            "final_1cm_required": False,
        },
        "next_runtime_command": command,
        "ready_for_explicit_runtime_approval_only": code_ready and basis_ok,
        "still_blocked": {
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "code_lines": code_lines,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_target_application_candidate_design_v1 "
        "local_design_static_audit_only=YES gpu_runtime=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 basis "
            f"code_ready={code_ready} basis_ok={basis_ok} "
            f"cap_only_falsified_as_primary={cap040.get('cap_only_falsified_as_primary')} "
            f"cap_no_longer_active={cap040.get('cap_no_longer_active')} "
            f"lead_limit_observed={cap040.get('lead_limit_observed')} "
            f"target_lead_abs_max_trace={float(cap040.get('target_lead_abs_max_trace', 0.0)):.9f} "
            f"target_lead_limit_rate_trace={float(cap040.get('target_lead_limit_rate_trace', 0.0)):.9f}"
        ),
        (
            "line3 selected_candidate "
            "status=DESIGNED_NOT_RUN baseline=cap040 changed_knobs_vs_cap040=1 "
            "joint_target_lead_limit_rad=0.060->0.120 "
            "max_joint_delta_per_step_rad=0.040_fixed joint_delta_reference=target_fixed "
            "action_scale=0.04_fixed action_smoothing_alpha=0.25_fixed "
            "controller_mode=external_closed_loop_fixed geometry=FIXED"
        ),
        (
            "line4 not_selected "
            "joint_delta_reference=requires_matching_harness_action_base_review "
            "action_scale=changes_command_normalization "
            "goal_push_or_contact_band=geometry_change smoothing=already_tested "
            "cap_only=falsified_as_primary"
        ),
        (
            "line5 pass_gate "
            "contact_seen>0 reaction_context>0 reaction_seen>0 tap_success>0 "
            "overshoot=0 final_1cm_required=False"
        ),
        f"line6 next_runtime_command {command}",
        (
            "line7 verdict "
            f"ready_for_explicit_runtime_approval_only={code_ready and basis_ok} "
            "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if code_ready and basis_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
