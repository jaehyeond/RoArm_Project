"""Local audit for y+ contact-strength evidence after seed962.

This posthoc/config audit uses existing seed958/960/961/962 y+ artifacts to
decide whether the next single-variable candidate should target path timing,
path-through depth, or contact-stop strength. It keeps final displacement as a
secondary diagnostic only; it performs no IsaacLab/GPU runtime, training,
dataset generation, robot control, SSH, or trace mutation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_FAILURE_SHIFT_JSON = LOG_DIR / "cube10cm_yplus_pre020_failure_shift_audit.json"
DEFAULT_REACTION_GATE_JSON = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_reaction_gate_audit.json"
)
DEFAULT_TRACE_DIAG_JSON = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_trace_diagnostic_summary.json"
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_yplus_contact_strength_candidate_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_yplus_contact_strength_candidate_audit_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(num: float, denom: float) -> float:
    return 0.0 if denom == 0.0 else num / denom


def _csv_retention_metrics(csv_path: Path, gate_disp_m: float) -> dict[str, float]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    finals = [_float(row.get("disp_along_push_m")) for row in rows]
    maxes = [_float(row.get("max_disp_along_push_m")) for row in rows]
    stop_rates = [_float(row.get("contact_stop_step_rate")) for row in rows]
    controlled = [_float(row.get("controlled_push")) for row in rows]
    retentions = [
        final / max_disp
        for final, max_disp in zip(finals, maxes)
        if max_disp > 0.0 and math.isfinite(final) and math.isfinite(max_disp)
    ]
    n = len(rows)
    return {
        "trials": float(n),
        "final_disp_mean_m": _mean(finals),
        "max_disp_mean_m": _mean(maxes),
        "final_ge_gate_rate": _mean([1.0 if value >= gate_disp_m else 0.0 for value in finals]),
        "max_ge_gate_rate": _mean([1.0 if value >= gate_disp_m else 0.0 for value in maxes]),
        "retention_mean": _mean(retentions),
        "retention_min": min(retentions) if retentions else 0.0,
        "contact_stop_step_rate_mean": _mean(stop_rates),
        "controlled_push_rate_from_csv": _mean(controlled),
    }


def _seed_rows(failure_shift: dict[str, Any]) -> list[dict[str, Any]]:
    rows = failure_shift.get("seed_summaries", [])
    if not isinstance(rows, list):
        raise ValueError("failure shift audit must contain seed_summaries list")
    return [row for row in rows if isinstance(row, dict)]


def _command_lines(*, candidate_stop_disp_m: float, out_stem: str) -> list[str]:
    base = "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
    return [
        "OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output "
        "python -u sim_scripts/cube10cm_push_diffik_probe.py \\",
        "  --num_envs 16 \\",
        "  --episodes 1 \\",
        "  --seed 962 \\",
        "  --fixed_cube_x_m 0.295 \\",
        "  --fixed_cube_y_m -0.044 \\",
        "  --fixed_push_dir 0 1 \\",
        "  --base_lateral_offset_m -0.020 \\",
        "  --xneg_tcp_center_height_offset_m 0.050 \\",
        "  --precontact_clearance_m 0.020 \\",
        "  --push_through_m 0.010 \\",
        f"  --contact_stop_disp_m {candidate_stop_disp_m:.3f} \\",
        "  --trace_diffik_diagnostics \\",
        "  --trace_all_envs \\",
        "  --trace_stride 4 \\",
        f"  --out_csv {base}/{out_stem}.csv \\",
        f"  --summary_json {base}/{out_stem}_summary.json \\",
        f"  --trace_csv {base}/{out_stem}_trace.csv",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure_shift_json", type=Path, default=DEFAULT_FAILURE_SHIFT_JSON)
    parser.add_argument("--reaction_gate_json", type=Path, default=DEFAULT_REACTION_GATE_JSON)
    parser.add_argument("--trace_diag_json", type=Path, default=DEFAULT_TRACE_DIAG_JSON)
    parser.add_argument("--candidate_contact_stop_disp_m", type=float, default=0.002)
    parser.add_argument("--gate_disp_m", type=float, default=0.001)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    if args.candidate_contact_stop_disp_m <= args.gate_disp_m:
        raise ValueError("--candidate_contact_stop_disp_m must be above --gate_disp_m for this diagnostic")
    if args.candidate_contact_stop_disp_m >= 0.010:
        raise ValueError("candidate contact stop must remain far below final 1cm relocation")

    failure_shift = _load_json(args.failure_shift_json)
    reaction_gate = _load_json(args.reaction_gate_json)
    trace_diag = _load_json(args.trace_diag_json)

    seed_summaries: list[dict[str, Any]] = []
    for row in _seed_rows(failure_shift):
        csv_path = REPO / str(row["summary_json"]).replace("_summary.json", ".csv")
        metrics = _csv_retention_metrics(csv_path, float(args.gate_disp_m))
        seed_summaries.append({**row, "csv": str(csv_path), "retention_metrics": metrics})

    seed962 = next(row for row in seed_summaries if row.get("label") == "seed962_pre020")
    previous = [
        row
        for row in seed_summaries
        if row.get("label") in {"seed958_pre010_baseline", "seed960_cap050", "seed961_stiff600"}
    ]
    prev_retention_mean = _mean([row["retention_metrics"]["retention_mean"] for row in previous])
    prev_final_gate_mean = _mean([row["retention_metrics"]["final_ge_gate_rate"] for row in previous])
    prev_stop_rate_mean = _mean([row["retention_metrics"]["contact_stop_step_rate_mean"] for row in previous])
    prev_max_disp_mean = _mean([row["retention_metrics"]["max_disp_mean_m"] for row in previous])
    seed962_ret = seed962["retention_metrics"]

    stop_retention_failure = (
        seed962_ret["max_ge_gate_rate"] >= 1.0
        and seed962_ret["final_ge_gate_rate"] < 0.75
        and seed962_ret["retention_mean"] < prev_retention_mean
        and _float(reaction_gate.get("overshoot_rate")) == 0.0
    )
    quality_still_blocked = (
        _float(reaction_gate.get("summary_diffik_clip_rate_mean")) >= 1.0
        or "JOINT_STEP_CLIPPING_DOMINANT" in {str(x) for x in reaction_gate.get("likely_modes", [])}
        or _float(reaction_gate.get("summary_final_tcp_target_err_mean_m")) > 0.03
    )

    diagnostic_candidate = {
        "name": "seed962_pre020_contact_stop002",
        "changed_variable": "contact_stop_disp_m",
        "baseline_contact_stop_disp_m": float(args.gate_disp_m),
        "candidate_contact_stop_disp_m": float(args.candidate_contact_stop_disp_m),
        "unchanged_tap_gate_disp_m": float(args.gate_disp_m),
        "unchanged_reaction_disp_m": float(args.gate_disp_m),
        "unchanged_precontact_clearance_m": 0.020,
        "unchanged_push_through_m": 0.010,
        "unchanged_approach_push_post_steps": [220, 90, 80],
        "unchanged_lateral_offset_m": -0.020,
        "unchanged_tcp_height_offset_m": 0.050,
        "unchanged_max_diffik_joint_step_rad": 0.035,
        "unchanged_dls_lambda": 0.010,
        "unchanged_arm_actuator_stiffness": 400.0,
        "hypothesis": "optional stronger transient 2mm contact-stop diagnostic, not a final-position objective",
        "expected_good_outcome": (
            "reaction/contact/no-posewrite/no-overshoot still pass and max transient displacement/tip increase without overshoot; "
            "final 1mm retention is not a required success gate"
        ),
        "known_risk": (
            "more contact strength can increase tip/follow lag; this is not a data-readiness or final-1cm candidate"
        ),
    }
    out_stem = "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_stop002_seed962"

    rejected_options = [
        {
            "candidate": "approach_steps 220 -> 200",
            "reason": "mostly shifts the same phase earlier while total episode length also shrinks; weak test for stop-retention failure",
        },
        {
            "candidate": "push_steps 90 -> 70",
            "reason": "increases per-step path demand and can worsen existing clip/follow lag before isolating stop retention",
        },
        {
            "candidate": "push_through_m 0.010 -> 0.020",
            "reason": "path-through increase is plausible but changes target speed/depth; stop-retention evidence is more direct and smaller",
        },
        {
            "candidate": "contact_stop_joint_step_scale 0.2 -> larger",
            "reason": "step-scale/actuator-follow mixing is explicitly out of scope for this candidate",
        },
        {
            "candidate": "precontact/lateral/height/DLS/cap changes",
            "reason": "would break the current one-variable separation requested for y+",
        },
    ]

    result = {
        "artifact_type": "cube10cm_yplus_contact_strength_candidate_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_posthoc_config_only": True,
        "no_gpu_isaaclab_training_dataset_ssh": True,
        "source_files": {
            "failure_shift_json": str(args.failure_shift_json),
            "reaction_gate_json": str(args.reaction_gate_json),
            "trace_diag_json": str(args.trace_diag_json),
        },
        "seed_summaries": seed_summaries,
        "previous_seed_group": {
            "labels": [str(row.get("label")) for row in previous],
            "retention_mean": prev_retention_mean,
            "final_ge_gate_rate_mean": prev_final_gate_mean,
            "contact_stop_step_rate_mean": prev_stop_rate_mean,
            "max_disp_mean_m": prev_max_disp_mean,
        },
        "seed962_ratios": {
            "retention_vs_previous_mean": _ratio(seed962_ret["retention_mean"], prev_retention_mean),
            "final_gate_vs_previous_mean": _ratio(seed962_ret["final_ge_gate_rate"], prev_final_gate_mean),
            "contact_stop_step_rate_vs_previous_mean": _ratio(
                seed962_ret["contact_stop_step_rate_mean"], prev_stop_rate_mean
            ),
            "max_disp_vs_previous_mean": _ratio(seed962_ret["max_disp_mean_m"], prev_max_disp_mean),
        },
        "decision": {
            "stop_retention_failure_supported": stop_retention_failure,
            "quality_still_blocked": quality_still_blocked,
            "final_retention_is_primary_objective": False,
            "selected_next_candidate": None,
            "diagnostic_candidate_if_stronger_transient_push_is_explicitly_requested": diagnostic_candidate,
            "rejected_options": rejected_options,
            "go_no_go_order": [
                "reaction_event",
                "contact_evidence",
                "no_posewrite",
                "no_overshoot",
                "max_transient_1mm_2mm_3mm_contact_strength_if_explicitly_requested",
                "quality_tier_A_B_C_metadata",
                "final_1cm_secondary_only_if_explicitly_requested",
            ],
            "do_not_start": [
                "GPU_runtime_without_explicit_approval",
                "blind_precontact_sweep",
                "lateral_height_actuator_dls_cap_mixing",
                "1024_10240_scaleup",
                "dataset_generation",
                "PPO_RL",
                "VLA",
                "TrackA",
                "B200_SSH",
            ],
        },
        "proposed_tiny_gpu_command_not_run": _command_lines(
            candidate_stop_disp_m=float(args.candidate_contact_stop_disp_m),
            out_stem=out_stem,
        ),
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_yplus_contact_strength_candidate_audit_v1 "
        "local_posthoc_config_only=YES gpu_runtime=NO dataset_generation=NO",
        (
            "line2 seed962_stop_retention "
            f"max_ge_gate_rate={seed962_ret['max_ge_gate_rate']:.9f} "
            f"final_ge_gate_rate={seed962_ret['final_ge_gate_rate']:.9f} "
            f"retention_mean={seed962_ret['retention_mean']:.9f} "
            f"previous_retention_mean={prev_retention_mean:.9f} "
            f"retention_ratio={result['seed962_ratios']['retention_vs_previous_mean']:.9f}"
        ),
        (
            "line3 seed962_strength "
            f"max_disp_mean_m={seed962_ret['max_disp_mean_m']:.9f} "
            f"previous_max_disp_mean_m={prev_max_disp_mean:.9f} "
            f"max_disp_ratio={result['seed962_ratios']['max_disp_vs_previous_mean']:.9f} "
            f"contact_stop_step_rate={seed962_ret['contact_stop_step_rate_mean']:.9f} "
            f"previous_contact_stop_step_rate={prev_stop_rate_mean:.9f}"
        ),
        (
            "line4 diagnostic_candidate_not_next_gpu "
            "changed_variable=contact_stop_disp_m "
            f"baseline_m={args.gate_disp_m:.6f} "
            f"candidate_m={args.candidate_contact_stop_disp_m:.6f} "
            "precontact_m=0.020000 push_through_m=0.010000 "
            "tap_gate_m=0.001000 final_1cm=NO final_retention_primary=NO"
        ),
        (
            "line5 rejected "
            "approach_steps_220_to_200=weak_for_retention "
            "push_steps_90_to_70=raises_per_step_demand "
            "push_through_010_to_020=less_direct_than_stop_retention "
            "contact_stop_joint_step_scale=actuator_mixing"
        ),
        (
            "line6 verdict "
            f"stop_retention_drop_observed={stop_retention_failure} "
            "final_retention_primary_objective=False "
            f"quality_still_blocked={quality_still_blocked} "
            "selected_next_candidate=NONE_FROM_FINAL_RETENTION_ALONE "
            "optional_diagnostic_if_stronger_transient_push_requested=seed962_pre020_contact_stop002"
        ),
        (
            "line7 next=do_not_request_gpu_from_final_retention_alone "
            "judge_order=reaction_contact_no_posewrite_no_overshoot_then_optional_max_1mm_2mm_3mm_then_quality_tier "
            "do_not_start=GPU_without_explicit_approval,blind_precontact_sweep,"
            "lateral_height_actuator_dls_cap_mixing,1024_10240,dataset,PPO_RL,VLA,TrackA,B200_SSH"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
