"""Local contract audit for link5-corner proxy and pose/trace support.

No IsaacLab runtime, no GPU, no dataset generation, no training, no robot
control, no SSH. This checks that the DiffIK probe can express the selected
`link5_collision:corner_011` proxy while preserving the old hand-TCP defaults,
and that a future tiny runtime would emit enough trace columns to judge the
proxy/contact geometry directly.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_PROBE = REPO / "sim_scripts/cube3cm_push_diffik_probe.py"
DEFAULT_PREV_SUMMARY = LOG_DIR / "cube10cm_tool_contact_proxy_orientation_preflight_summary.out"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_link5_proxy_pose_trace_contract_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_link5_proxy_pose_trace_contract_audit_summary.out"

REQUIRED_TRACE_FIELDS = (
    "diffik_command_type",
    "diffik_pose_quat_mode",
    "tool_contact_proxy_mode",
    "tool_proxy_label",
    "tool_contact_target_x_m",
    "tool_contact_target_y_m",
    "tool_contact_target_z_m",
    "tool_proxy_target_err_before_m",
    "tool_proxy_target_err_after_m",
    "tool_proxy_target_z_err_before_m",
    "tool_proxy_target_z_err_after_m",
    "link5_qw_before",
    "link5_qx_before",
    "link5_qy_before",
    "link5_qz_before",
    "link5_qw_target",
    "link5_qx_target",
    "link5_qy_target",
    "link5_qz_target",
    "link5_qw_after",
    "link5_qx_after",
    "link5_qy_after",
    "link5_qz_after",
    "tool_proxy_x_before_m",
    "tool_proxy_y_before_m",
    "tool_proxy_z_before_m",
    "tool_proxy_x_after_m",
    "tool_proxy_y_after_m",
    "tool_proxy_z_after_m",
)

REQUIRED_SUMMARY_KEYS = (
    "command_type",
    "diffik_pose_quat_mode",
    "tool_contact_proxy_mode",
    "tool_proxy_label",
    "tool_proxy_local_m",
    "min_tool_proxy_target_err_mean_m",
    "final_tool_proxy_target_err_mean_m",
)


def _literal_from_node(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _arg_defaults(tree: ast.AST) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    choices: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        attr = node.func
        if not isinstance(attr, ast.Attribute) or attr.attr != "add_argument":
            continue
        if not node.args:
            continue
        option = _literal_from_node(node.args[0])
        if not isinstance(option, str) or not option.startswith("--"):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                defaults[option] = _literal_from_node(kw.value)
            elif kw.arg == "choices":
                choices[option] = _literal_from_node(kw.value)
    return {"defaults": defaults, "choices": choices}


def _read_previous_summary(path: Path) -> dict[str, str]:
    lines: dict[str, str] = {}
    if not path.exists():
        return lines
    for raw in path.read_text().splitlines():
        key = raw.split(" ", 1)[0]
        lines[key] = raw
    return lines


def build_audit(probe_path: Path, previous_summary: Path) -> dict[str, Any]:
    source = probe_path.read_text()
    tree = ast.parse(source)
    parsed = _arg_defaults(tree)
    defaults = parsed["defaults"]
    choices = parsed["choices"]
    prev = _read_previous_summary(previous_summary)

    required_snippets = {
        "diffik_cfg_uses_arg_command_type": "command_type=args.diffik_command_type" in source,
        "link5_corner_constant_present": "LINK5_COLLISION_CORNER_011_LOCAL_M" in source,
        "link5_proxy_branch_present": 'args.tool_contact_proxy_mode == "link5_collision_corner_011"' in source,
        "pose_command_uses_7d_command": "torch.cat((link5_target_b, link5_quat_target_b), dim=-1)" in source,
        "position_default_uses_3d_link5_target": "diffik_command = link5_target_b" in source,
        "initial_link5_quat_mode_present": 'args.diffik_pose_quat_mode == "initial_link5"' in source,
        "current_link5_quat_mode_present": 'args.diffik_pose_quat_mode == "current_link5"' in source,
        "proxy_offset_uses_target_quat": "tool_proxy_target_offset_w = quat_rotate(link5_quat_target_w, tool_proxy_local)" in source,
    }
    trace_fields = {field: f'"{field}"' in source for field in REQUIRED_TRACE_FIELDS}
    summary_keys = {field: f'"{field}"' in source for field in REQUIRED_SUMMARY_KEYS}

    cli_contract_ok = (
        defaults.get("--diffik_command_type") == "position"
        and tuple(choices.get("--diffik_command_type", ())) == ("position", "pose")
        and defaults.get("--tool_contact_proxy_mode") == "hand_tcp"
        and tuple(choices.get("--tool_contact_proxy_mode", ())) == ("hand_tcp", "link5_collision_corner_011")
        and defaults.get("--diffik_pose_quat_mode") == "current_link5"
        and tuple(choices.get("--diffik_pose_quat_mode", ())) == ("current_link5", "initial_link5")
    )
    runtime_mapping_ok = all(required_snippets.values())
    trace_contract_ok = all(trace_fields.values())
    summary_contract_ok = all(summary_keys.values())

    pose_support_available = bool(
        cli_contract_ok
        and runtime_mapping_ok
        and defaults.get("--diffik_command_type") == "position"
        and "pose" in choices.get("--diffik_command_type", ())
    )
    # The first runtime should isolate the physical proxy retargeting. Pose is
    # implemented and traceable, but making it the first GPU change would mix
    # proxy retargeting with a 6D pose constraint on a 5-joint arm.
    pose_first_runtime_recommended = False
    first_tiny_candidate = {
        "name": "seed962_yplus_pre020_link5corner_position_trace_only",
        "requires_explicit_approval": True,
        "gpu_runtime": "ONE_TINY_LOCAL_ONLY",
        "change_from_seed962_pre020": {
            "tool_contact_proxy_mode": "hand_tcp -> link5_collision_corner_011",
            "diffik_command_type": "position",
            "diffik_pose_quat_mode": "current_link5",
            "geometry_lateral_height_actuator_dls_cap": "UNCHANGED",
        },
        "judge_order": [
            "reaction/contact/no-posewrite/no-overshoot",
            "tool_proxy_target_err and link5 quaternion trace",
            "quality tier",
            "final displacement secondary only",
        ],
    }

    contract_ready = bool(cli_contract_ok and runtime_mapping_ok and trace_contract_ok and summary_contract_ok)
    return {
        "artifact_type": "cube10cm_link5_proxy_pose_trace_contract_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_audit_only": True,
        "no_gpu_runtime_dataset_training_robot_ssh": True,
        "source": {
            "probe_path": str(probe_path),
            "previous_preflight_summary": str(previous_summary),
            "previous_line9": prev.get("line9"),
            "previous_line10": prev.get("line10"),
            "previous_line11": prev.get("line11"),
            "previous_line12": prev.get("line12"),
        },
        "cli_contract": {
            "defaults": {
                "diffik_command_type": defaults.get("--diffik_command_type"),
                "tool_contact_proxy_mode": defaults.get("--tool_contact_proxy_mode"),
                "diffik_pose_quat_mode": defaults.get("--diffik_pose_quat_mode"),
            },
            "choices": {
                "diffik_command_type": choices.get("--diffik_command_type"),
                "tool_contact_proxy_mode": choices.get("--tool_contact_proxy_mode"),
                "diffik_pose_quat_mode": choices.get("--diffik_pose_quat_mode"),
            },
            "default_preserves_existing_hand_tcp_position_path": cli_contract_ok,
        },
        "runtime_mapping": {
            **required_snippets,
            "runtime_mapping_ok": runtime_mapping_ok,
        },
        "trace_contract": {
            "required_trace_fields_present": trace_fields,
            "trace_contract_ok": trace_contract_ok,
        },
        "summary_contract": {
            "required_summary_keys_present": summary_keys,
            "summary_contract_ok": summary_contract_ok,
        },
        "pose_support": {
            "pose_support_available": pose_support_available,
            "pose_first_runtime_recommended": pose_first_runtime_recommended,
            "reason": "pose command is implemented, but first runtime should isolate link5 proxy retargeting before adding a 6D pose constraint",
        },
        "first_tiny_runtime_candidate": first_tiny_candidate,
        "verdict": {
            "code_contract_ready_for_one_tiny_runtime_consideration": contract_ready,
            "dataset_rl_roarm_unblocked": False,
            "next": "consider_exactly_one_local_tiny_link5_proxy_position_trace_runtime_only_after_explicit_approval",
        },
    }


def write_summary(audit: dict[str, Any], out_summary: Path) -> None:
    cli = audit["cli_contract"]
    mapping = audit["runtime_mapping"]
    trace = audit["trace_contract"]
    summary = audit["summary_contract"]
    pose = audit["pose_support"]
    verdict = audit["verdict"]
    candidate = audit["first_tiny_runtime_candidate"]
    source = audit["source"]
    lines = [
        "line1 artifact=cube10cm_link5_proxy_pose_trace_contract_audit_v1 "
        "local_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        "line2 prior_evidence "
        f"previous_line9_present={source['previous_line9'] is not None} "
        f"previous_line10_present={source['previous_line10'] is not None} "
        f"previous_line11_present={source['previous_line11'] is not None} "
        f"previous_line12_present={source['previous_line12'] is not None}",
        "line3 cli_contract "
        f"default_command_type={cli['defaults']['diffik_command_type']} "
        f"default_proxy_mode={cli['defaults']['tool_contact_proxy_mode']} "
        f"default_pose_quat_mode={cli['defaults']['diffik_pose_quat_mode']} "
        f"default_preserves_existing={cli['default_preserves_existing_hand_tcp_position_path']}",
        "line4 runtime_mapping "
        f"diffik_cfg_uses_arg_command_type={mapping['diffik_cfg_uses_arg_command_type']} "
        f"link5_proxy_branch_present={mapping['link5_proxy_branch_present']} "
        f"pose_7d_command_present={mapping['pose_command_uses_7d_command']} "
        f"proxy_offset_uses_target_quat={mapping['proxy_offset_uses_target_quat']} "
        f"runtime_mapping_ok={mapping['runtime_mapping_ok']}",
        "line5 trace_contract "
        f"required_trace_fields={len(trace['required_trace_fields_present'])} "
        f"trace_contract_ok={trace['trace_contract_ok']}",
        "line6 summary_contract "
        f"required_summary_keys={len(summary['required_summary_keys_present'])} "
        f"summary_contract_ok={summary['summary_contract_ok']}",
        "line7 pose_support "
        f"pose_support_available={pose['pose_support_available']} "
        f"pose_first_runtime_recommended={pose['pose_first_runtime_recommended']} "
        "reason=first_runtime_should_isolate_proxy_retargeting_before_6d_pose_constraint",
        "line8 tiny_runtime_candidate "
        f"name={candidate['name']} requires_explicit_approval={candidate['requires_explicit_approval']} "
        "change=tool_contact_proxy_mode_hand_tcp_to_link5_collision_corner_011 "
        "command_type=position pose_quat_mode=current_link5 "
        "geometry_lateral_height_actuator_dls_cap=UNCHANGED",
        "line9 verdict "
        f"code_contract_ready_for_one_tiny_runtime_consideration={verdict['code_contract_ready_for_one_tiny_runtime_consideration']} "
        f"dataset_rl_roarm_unblocked={verdict['dataset_rl_roarm_unblocked']} "
        f"next={verdict['next']}",
    ]
    out_summary.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_path", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--previous_summary", type=Path, default=DEFAULT_PREV_SUMMARY)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    audit = build_audit(args.probe_path, args.previous_summary)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_summary(audit, args.out_summary)
    print(args.out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
