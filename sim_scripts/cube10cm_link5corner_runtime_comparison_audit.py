"""Compare seed962 hand-TCP baseline with link5-corner proxy runtime.

Local-only post-runtime audit. It reads existing summaries/audits and writes a
short, line-oriented verdict. No IsaacLab runtime, no GPU, no dataset generation,
no training, no robot control, no SSH.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
BASE_SUMMARY = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_summary.json"
BASE_GATE = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_reaction_gate_audit.json"
BASE_WINDOW = LOG_DIR / "cube10cm_reaction_window_seed962_audit.json"
NEW_SUMMARY = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_summary.json"
NEW_GATE = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_reaction_gate_audit.json"
NEW_WINDOW = LOG_DIR / "cube10cm_reaction_window_link5corner_position_seed962_audit.json"
NEW_TRACE_DIAG = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_trace_diagnostic_summary.json"
OUT_JSON = LOG_DIR / "cube10cm_link5corner_position_seed962_comparison_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_link5corner_position_seed962_comparison_audit_summary.out"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _f(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    return default if value is None else float(value)


def _ratio(new: float, base: float) -> float | None:
    if abs(base) < 1.0e-12:
        return None
    return new / base


def _tier_counts(window: dict[str, Any]) -> dict[str, int]:
    counts = window.get("quality_tier_counts")
    if isinstance(counts, dict):
        return {str(k): int(v) for k, v in counts.items()}
    out: dict[str, int] = {}
    for row in window.get("per_window", []):
        tier = str(row.get("quality_tier", "UNKNOWN"))
        out[tier] = out.get(tier, 0) + 1
    return out


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    base_summary = _read(args.base_summary)
    base_gate = _read(args.base_gate)
    base_window = _read(args.base_window)
    new_summary = _read(args.new_summary)
    new_gate = _read(args.new_gate)
    new_window = _read(args.new_window)
    new_trace_diag = _read(args.new_trace_diag)

    metrics = {
        "reaction_event_rate": (
            _f(base_summary, "reaction_event_rate"),
            _f(new_summary, "reaction_event_rate"),
        ),
        "contact_evidence_rate": (
            _f(base_gate, "contact_evidence_rate"),
            _f(new_gate, "contact_evidence_rate"),
        ),
        "overshoot_rate": (
            _f(base_gate, "overshoot_rate"),
            _f(new_gate, "overshoot_rate"),
        ),
        "posewrite_calls": (
            _f(base_summary, "posewrite_calls_during_rollout"),
            _f(new_summary, "posewrite_calls_during_rollout"),
        ),
        "controlled_push_rate": (
            _f(base_summary, "controlled_push_rate"),
            _f(new_summary, "controlled_push_rate"),
        ),
        "low_motion_rate": (
            _f(base_summary, "low_motion_rate"),
            _f(new_summary, "low_motion_rate"),
        ),
        "max_disp_along_push_mean_m": (
            _f(base_summary, "max_disp_along_push_mean_m"),
            _f(new_summary, "max_disp_along_push_mean_m"),
        ),
        "final_disp_along_push_mean_m": (
            _f(base_summary, "disp_along_push_mean_m"),
            _f(new_summary, "disp_along_push_mean_m"),
        ),
        "max_cube_speed_mean_mps": (
            _f(base_summary, "max_cube_speed_mean_mps"),
            _f(new_summary, "max_cube_speed_mean_mps"),
        ),
        "final_tcp_target_err_mean_m": (
            _f(base_summary, "final_tcp_target_err_mean_m"),
            _f(new_summary, "final_tcp_target_err_mean_m"),
        ),
        "diffik_clip_rate_mean": (
            _f(base_summary, "diffik_clip_rate_mean"),
            _f(new_summary, "diffik_clip_rate_mean"),
        ),
    }
    ratios = {key: _ratio(new, base) for key, (base, new) in metrics.items()}
    proxy_metrics = {
        "min_tool_proxy_target_err_mean_m": _f(new_summary, "min_tool_proxy_target_err_mean_m"),
        "final_tool_proxy_target_err_mean_m": _f(new_summary, "final_tool_proxy_target_err_mean_m"),
        "trace_clip_any_rate": _f(new_trace_diag, "clip_any_rate"),
        "trace_pre_stop_clip_any_rate": _f(new_trace_diag.get("phase_splits", {}).get("pre_stop", {}), "clip_any_rate"),
        "trace_post_stop_clip_any_rate": _f(new_trace_diag.get("phase_splits", {}).get("post_stop", {}), "clip_any_rate"),
    }
    verdict = {
        "reaction_gate_pass": bool(new_gate.get("reaction_gate_pass")),
        "reaction_contact_no_posewrite_no_overshoot_pass": bool(
            new_gate.get("reaction_gate_pass")
            and new_gate.get("contact_evidence_rate") == 1.0
            and new_gate.get("overshoot_rate") == 0.0
            and new_gate.get("no_posewrite")
        ),
        "proxy_tracking_improved": bool(
            _f(new_summary, "diffik_clip_rate_mean") < _f(base_summary, "diffik_clip_rate_mean")
            and _f(new_summary, "final_tcp_target_err_mean_m") < _f(base_summary, "final_tcp_target_err_mean_m")
            and _f(new_summary, "final_tool_proxy_target_err_mean_m") <= 0.005
        ),
        "tap_strength_weakened_vs_baseline": bool(
            ratios["max_disp_along_push_mean_m"] is not None
            and ratios["max_disp_along_push_mean_m"] < 0.60
            and ratios["max_cube_speed_mean_mps"] is not None
            and ratios["max_cube_speed_mean_mps"] < 0.25
        ),
        "clean_diffik_teacher_ready": bool(new_window.get("clean_diffik_teacher_window_ready")),
        "dataset_rl_roarm_unblocked": False,
        "next": "do_not_scale_dataset_or_rl; inspect_visual_proxy_contact_or_design_one_strength_preserving_proxy_variant_only_if_needed",
    }
    return {
        "artifact_type": "cube10cm_link5corner_runtime_comparison_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_audit_only": True,
        "no_gpu_runtime_dataset_training_robot_ssh": True,
        "source": {
            "base_summary": str(args.base_summary),
            "base_gate": str(args.base_gate),
            "base_window": str(args.base_window),
            "new_summary": str(args.new_summary),
            "new_gate": str(args.new_gate),
            "new_window": str(args.new_window),
            "new_trace_diag": str(args.new_trace_diag),
        },
        "contract": {
            "new_tool_contact_proxy_mode": new_summary.get("tool_contact_proxy_mode"),
            "new_diffik_command_type": new_summary.get("command_type"),
            "new_tool_proxy_label": new_summary.get("tool_proxy_label"),
            "new_tool_proxy_local_m": new_summary.get("tool_proxy_local_m"),
        },
        "metrics": {
            key: {"baseline": base, "new": new, "new_vs_baseline_ratio": ratios[key]}
            for key, (base, new) in metrics.items()
        },
        "proxy_metrics": proxy_metrics,
        "window": {
            "baseline_accepted_window_count": int(base_window.get("accepted_window_count", 0)),
            "new_accepted_window_count": int(new_window.get("accepted_window_count", 0)),
            "baseline_quality_tier_counts": _tier_counts(base_window),
            "new_quality_tier_counts": _tier_counts(new_window),
            "baseline_follow_p95_to_cap_p95": _f(base_window, "accepted_window_follow_p95_to_cap_p95"),
            "new_follow_p95_to_cap_p95": _f(new_window, "accepted_window_follow_p95_to_cap_p95"),
            "baseline_clip_any_rate_mean": _f(base_window, "accepted_window_clip_any_rate_mean"),
            "new_clip_any_rate_mean": _f(new_window, "accepted_window_clip_any_rate_mean"),
        },
        "verdict": verdict,
    }


def write_summary(audit: dict[str, Any], out_summary: Path) -> None:
    m = audit["metrics"]
    p = audit["proxy_metrics"]
    w = audit["window"]
    v = audit["verdict"]
    c = audit["contract"]
    lines = [
        "line1 artifact=cube10cm_link5corner_runtime_comparison_audit_v1 "
        "local_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        "line2 contract "
        f"tool_contact_proxy_mode={c['new_tool_contact_proxy_mode']} command_type={c['new_diffik_command_type']} "
        f"tool_proxy_label={c['new_tool_proxy_label']} tool_proxy_local_m={c['new_tool_proxy_local_m']}",
        "line3 reaction_gate "
        f"baseline_reaction={m['reaction_event_rate']['baseline']:.9f} new_reaction={m['reaction_event_rate']['new']:.9f} "
        f"baseline_contact={m['contact_evidence_rate']['baseline']:.9f} new_contact={m['contact_evidence_rate']['new']:.9f} "
        f"baseline_overshoot={m['overshoot_rate']['baseline']:.9f} new_overshoot={m['overshoot_rate']['new']:.9f} "
        f"baseline_posewrite={m['posewrite_calls']['baseline']:.0f} new_posewrite={m['posewrite_calls']['new']:.0f}",
        "line4 tracking "
        f"baseline_clip={m['diffik_clip_rate_mean']['baseline']:.9f} new_clip={m['diffik_clip_rate_mean']['new']:.9f} "
        f"clip_ratio={m['diffik_clip_rate_mean']['new_vs_baseline_ratio']:.9f} "
        f"baseline_final_tcp_err={m['final_tcp_target_err_mean_m']['baseline']:.9f} "
        f"new_final_tcp_err={m['final_tcp_target_err_mean_m']['new']:.9f} "
        f"tcp_err_ratio={m['final_tcp_target_err_mean_m']['new_vs_baseline_ratio']:.9f}",
        "line5 proxy_tracking "
        f"min_tool_proxy_target_err_mean={p['min_tool_proxy_target_err_mean_m']:.9f} "
        f"final_tool_proxy_target_err_mean={p['final_tool_proxy_target_err_mean_m']:.9f} "
        f"trace_clip_any={p['trace_clip_any_rate']:.9f} "
        f"pre_stop_clip_any={p['trace_pre_stop_clip_any_rate']:.9f} "
        f"post_stop_clip_any={p['trace_post_stop_clip_any_rate']:.9f}",
        "line6 tap_strength "
        f"baseline_max_disp={m['max_disp_along_push_mean_m']['baseline']:.9f} "
        f"new_max_disp={m['max_disp_along_push_mean_m']['new']:.9f} "
        f"max_disp_ratio={m['max_disp_along_push_mean_m']['new_vs_baseline_ratio']:.9f} "
        f"baseline_speed={m['max_cube_speed_mean_mps']['baseline']:.9f} "
        f"new_speed={m['max_cube_speed_mean_mps']['new']:.9f} "
        f"speed_ratio={m['max_cube_speed_mean_mps']['new_vs_baseline_ratio']:.9f} "
        f"baseline_controlled={m['controlled_push_rate']['baseline']:.9f} "
        f"new_controlled={m['controlled_push_rate']['new']:.9f} "
        f"new_low_motion={m['low_motion_rate']['new']:.9f}",
        "line7 reaction_window "
        f"baseline_windows={w['baseline_accepted_window_count']} new_windows={w['new_accepted_window_count']} "
        f"baseline_tiers={w['baseline_quality_tier_counts']} new_tiers={w['new_quality_tier_counts']} "
        f"baseline_follow_p95_cap={w['baseline_follow_p95_to_cap_p95']:.9f} "
        f"new_follow_p95_cap={w['new_follow_p95_to_cap_p95']:.9f} "
        f"baseline_window_clip={w['baseline_clip_any_rate_mean']:.9f} "
        f"new_window_clip={w['new_clip_any_rate_mean']:.9f}",
        "line8 verdict "
        f"reaction_contact_no_posewrite_no_overshoot_pass={v['reaction_contact_no_posewrite_no_overshoot_pass']} "
        f"proxy_tracking_improved={v['proxy_tracking_improved']} "
        f"tap_strength_weakened_vs_baseline={v['tap_strength_weakened_vs_baseline']} "
        f"clean_diffik_teacher_ready={v['clean_diffik_teacher_ready']} "
        f"dataset_rl_roarm_unblocked={v['dataset_rl_roarm_unblocked']} "
        f"next={v['next']}",
    ]
    out_summary.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_summary", type=Path, default=BASE_SUMMARY)
    parser.add_argument("--base_gate", type=Path, default=BASE_GATE)
    parser.add_argument("--base_window", type=Path, default=BASE_WINDOW)
    parser.add_argument("--new_summary", type=Path, default=NEW_SUMMARY)
    parser.add_argument("--new_gate", type=Path, default=NEW_GATE)
    parser.add_argument("--new_window", type=Path, default=NEW_WINDOW)
    parser.add_argument("--new_trace_diag", type=Path, default=NEW_TRACE_DIAG)
    parser.add_argument("--out_json", type=Path, default=OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=OUT_SUMMARY)
    args = parser.parse_args()

    audit = build_audit(args)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_summary(audit, args.out_summary)
    print(args.out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
