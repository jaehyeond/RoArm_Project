"""Local next-step audit for the professor cube10cm tap/reaction branch.

Reads existing reaction/trace diagnostic JSONs and prints the narrow next
research direction. This keeps the branch from drifting into 1cm relocation,
dataset generation, PPO/RL scale-up, or broad random searches before teacher
quality and robustness evidence justify them.

No IsaacLab app, GPU runtime, training, dataset generation, robot control, or log
mutation is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_REACTION_AUDIT = (
    LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_reaction_gate_audit.json"
)
DEFAULT_TRACE_DIAG = (
    LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_seed946_trace_diagnostic_summary.json"
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_next_research_step_seed946_audit.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return bool(value)


def _next_direction(reaction: dict[str, Any], trace: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    likely_modes = {str(x) for x in reaction.get("likely_modes", [])}
    likely_modes.update(str(x) for x in trace.get("likely_modes", []))

    if not _bool(reaction.get("controller_ok")) or not _bool(reaction.get("no_posewrite")):
        reasons.append("controller integrity failed")
        return "STOP_FIX_CONTROLLER_OR_POSEWRITE_BEFORE_ANY_RUNTIME", reasons

    if _float(reaction.get("overshoot_rate")) > _float(reaction.get("thresholds", {}).get("max_overshoot_rate"), 0.0):
        reasons.append("overshoot exceeds reaction gate")
        return "TUNE_CONTACT_STOP_OVERSHOOT_BEFORE_DATA_OR_RL", reasons

    if _float(reaction.get("contact_evidence_rate")) < _float(
        reaction.get("thresholds", {}).get("min_contact_evidence_rate"), 1.0
    ):
        reasons.append("reaction exists but contact evidence is incomplete")
        return "FIX_CONTACT_GEOMETRY_OR_WORKSPACE_BUCKET_FIRST", reasons

    if not _bool(reaction.get("reaction_gate_pass")):
        reasons.append("reaction gate is false despite controller/contact checks")
        return "FIX_REACTION_GATE_INPUTS_BEFORE_TEACHER_OR_RL", reasons

    teacher_ready = _bool(reaction.get("teacher_quality_ready"))
    if teacher_ready:
        reasons.append("teacher quality gate is true on the supplied audit")
        return "RUN_TINY_HELDOUT_ROBUSTNESS_CHECK_BEFORE_DATASET_OR_RL", reasons

    clip_rate = _float(reaction.get("summary_diffik_clip_rate_mean"))
    tcp_err = _float(reaction.get("summary_final_tcp_target_err_mean_m"))
    teacher_clip_max = _float(reaction.get("thresholds", {}).get("teacher_max_diffik_clip_rate"), 0.5)
    teacher_tcp_max = _float(reaction.get("thresholds", {}).get("teacher_max_final_tcp_err_m"), 0.03)
    if clip_rate > teacher_clip_max or "JOINT_STEP_CLIPPING_DOMINANT" in likely_modes:
        reasons.append(f"diffik_clip_rate={clip_rate:.6f} > teacher_max={teacher_clip_max:.6f}")
    if tcp_err > teacher_tcp_max or "LINK5_BODY_TARGET_NOT_REACHED" in likely_modes:
        reasons.append(f"final_tcp_err_m={tcp_err:.6f} > teacher_max={teacher_tcp_max:.6f}")
    if "ACTUATOR_TARGET_TRACKING_LAG" in likely_modes:
        reasons.append("trace reports actuator target tracking lag")
    if reasons:
        return "NARROW_ACTUATOR_IK_TRACKING_CLEANUP_INSIDE_WORKING_TAP_GEOMETRY", reasons

    reasons.append("teacher quality false without a classified trace mode")
    return "INSPECT_TRACE_DIAGNOSTICS_BEFORE_ANY_GPU_SCALEUP", reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reaction_audit_json", type=Path, default=DEFAULT_REACTION_AUDIT)
    parser.add_argument("--trace_diag_json", type=Path, default=DEFAULT_TRACE_DIAG)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    reaction = _load_json(args.reaction_audit_json)
    trace = _load_json(args.trace_diag_json)
    direction, reasons = _next_direction(reaction, trace)
    result = {
        "branch": "professor_cube10cm_tap_reaction",
        "primary_objective": "tap_reaction_not_final_1cm",
        "reaction_audit_json": str(args.reaction_audit_json),
        "trace_diag_json": str(args.trace_diag_json),
        "reaction_gate_pass": reaction.get("reaction_gate_pass"),
        "final_relocation_pass_secondary": reaction.get("final_relocation_pass"),
        "teacher_quality_ready": reaction.get("teacher_quality_ready"),
        "contact_evidence_rate": _float(reaction.get("contact_evidence_rate")),
        "overshoot_rate": _float(reaction.get("overshoot_rate")),
        "diffik_clip_rate": _float(reaction.get("summary_diffik_clip_rate_mean")),
        "final_tcp_err_m": _float(reaction.get("summary_final_tcp_target_err_mean_m")),
        "next_direction": direction,
        "reasons": reasons,
        "do_not_start": ["dataset_generation", "PPO_RL", "VLA", "TrackA", "1024_10k_scaleup"],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[cube10cm_next_research_step_audit] branch=professor_cube10cm_tap_reaction")
    print("[cube10cm_next_research_step_audit] primary_objective=tap_reaction_not_final_1cm")
    print(f"[cube10cm_next_research_step_audit] reaction_gate_pass={reaction.get('reaction_gate_pass')}")
    print(f"[cube10cm_next_research_step_audit] final_relocation_pass_secondary={reaction.get('final_relocation_pass')}")
    print(f"[cube10cm_next_research_step_audit] teacher_quality_ready={reaction.get('teacher_quality_ready')}")
    print(f"[cube10cm_next_research_step_audit] contact_evidence_rate={_float(reaction.get('contact_evidence_rate')):.6f}")
    print(f"[cube10cm_next_research_step_audit] overshoot_rate={_float(reaction.get('overshoot_rate')):.6f}")
    print(f"[cube10cm_next_research_step_audit] diffik_clip_rate={_float(reaction.get('summary_diffik_clip_rate_mean')):.6f}")
    print(f"[cube10cm_next_research_step_audit] final_tcp_err_m={_float(reaction.get('summary_final_tcp_target_err_mean_m')):.6f}")
    print(f"[cube10cm_next_research_step_audit] next_direction={direction}")
    for reason in reasons:
        print(f"[cube10cm_next_research_step_audit] reason={reason}")
    print("[cube10cm_next_research_step_audit] do_not_start=dataset_generation,PPO_RL,VLA,TrackA,1024_10k_scaleup")
    print(f"[cube10cm_next_research_step_audit] out_json={args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
