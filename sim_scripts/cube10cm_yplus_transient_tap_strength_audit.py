"""Local audit for y+ transient tap strength without final-position gates.

This posthoc audit compares fixed y+ seed958/960/961/962 artifacts using
transient max displacement thresholds (1/2/3/5mm), contact evidence, overshoot,
tip/z/speed, and quality-tier metadata. It intentionally does not use final
cube displacement as a success gate. No IsaacLab/GPU runtime, training, dataset
generation, robot control, SSH, or trace mutation is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_RUNS = (
    (
        "seed958_pre010",
        LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_seed958.csv",
        LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_seed958_summary.json",
        LOG_DIR / "cube10cm_reaction_window_seed958_audit.json",
    ),
    (
        "seed960_cap050",
        LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_cap050_seed960.csv",
        LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_cap050_seed960_summary.json",
        LOG_DIR / "cube10cm_reaction_window_seed960_audit.json",
    ),
    (
        "seed961_stiff600",
        LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_stiff600_seed961.csv",
        LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_stiff600_seed961_summary.json",
        LOG_DIR / "cube10cm_reaction_window_seed961_audit.json",
    ),
    (
        "seed962_pre020",
        LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962.csv",
        LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_summary.json",
        LOG_DIR / "cube10cm_reaction_window_seed962_audit.json",
    ),
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_yplus_transient_tap_strength_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_yplus_transient_tap_strength_audit_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_flag(row: dict[str, str], key: str) -> int:
    return 1 if int(_float(row.get(key), 0.0)) != 0 else 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(flags: list[bool]) -> float:
    return _mean([1.0 if flag else 0.0 for flag in flags])


def _threshold_rates(values: list[float]) -> dict[str, float]:
    thresholds = {
        "max_ge_1mm_rate": 0.001,
        "max_ge_2mm_rate": 0.002,
        "max_ge_3mm_rate": 0.003,
        "max_ge_5mm_rate": 0.005,
        "max_ge_10mm_rate": 0.010,
    }
    return {name: _rate([value >= threshold for value in values]) for name, threshold in thresholds.items()}


def _quality_counts(audit: dict[str, Any]) -> dict[str, int]:
    rows = audit.get("per_window", [])
    counter = Counter(
        str(row.get("quality_tier", "UNKNOWN"))
        for row in rows
        if isinstance(row, dict) and bool(row.get("accepted", False))
    )
    return dict(sorted(counter.items()))


def _summarize_run(label: str, csv_path: Path, summary_path: Path, audit_path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    summary = _load_json(summary_path)
    audit = _load_json(audit_path)
    max_disp = [_float(row.get("max_disp_along_push_m")) for row in rows]
    max_tip = [_float(row.get("max_tip_angle_deg")) for row in rows]
    max_z = [_float(row.get("max_cube_z_delta_m")) for row in rows]
    max_speed = [_float(row.get("max_cube_speed_mps")) for row in rows]
    contact = [
        bool(_int_flag(row, "measured_contact_seen") or _int_flag(row, "contact_stop_seen"))
        for row in rows
    ]
    overshoot = [bool(_int_flag(row, "contact_overshoot_seen")) for row in rows]
    reaction_event = [bool(_int_flag(row, "reaction_event")) for row in rows]

    return {
        "label": label,
        "csv": str(csv_path),
        "summary_json": str(summary_path),
        "reaction_window_audit_json": str(audit_path),
        "trials": len(rows),
        "final_position_used_as_success_gate": False,
        "precontact_clearance_m": _float(summary.get("precontact_clearance_m")),
        "push_through_m": _float(summary.get("push_through_m")),
        "max_diffik_joint_step_rad": _float(summary.get("max_diffik_joint_step_rad")),
        "arm_actuator_stiffness": _float(summary.get("arm_actuator_stiffness")),
        "contact_evidence_rate": _rate(contact),
        "reaction_event_rate": _rate(reaction_event),
        "overshoot_rate": _rate(overshoot),
        "max_disp_along_push_mean_m": _mean(max_disp),
        "max_tip_angle_mean_deg": _mean(max_tip),
        "max_cube_z_delta_mean_m": _mean(max_z),
        "max_cube_speed_mean_mps": _mean(max_speed),
        "transient_strength_rates": _threshold_rates(max_disp),
        "reaction_window_contract_pass": bool(audit.get("reaction_window_contract_pass", False)),
        "quality_tier_counts": _quality_counts(audit),
        "accepted_window_follow_p95_to_cap_p95": _float(audit.get("accepted_window_follow_p95_to_cap_p95")),
        "clean_diffik_teacher_window_ready": bool(audit.get("clean_diffik_teacher_window_ready", False)),
    }


def _ratio(num: float, denom: float) -> float:
    return 0.0 if denom == 0.0 else num / denom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    run_summaries = [_summarize_run(label, csv_path, summary_path, audit_path) for label, csv_path, summary_path, audit_path in DEFAULT_RUNS]
    by_label = {str(row["label"]): row for row in run_summaries}
    seed962 = by_label["seed962_pre020"]
    previous = [row for row in run_summaries if row["label"] != "seed962_pre020"]

    def prev_mean(key: str) -> float:
        return _mean([float(row[key]) for row in previous])

    def prev_rate(rate_key: str) -> float:
        return _mean([float(row["transient_strength_rates"][rate_key]) for row in previous])

    seed962_rates = seed962["transient_strength_rates"]
    comparisons = {
        "seed962_max_disp_vs_previous_mean": _ratio(
            float(seed962["max_disp_along_push_mean_m"]), prev_mean("max_disp_along_push_mean_m")
        ),
        "seed962_tip_vs_previous_mean": _ratio(
            float(seed962["max_tip_angle_mean_deg"]), prev_mean("max_tip_angle_mean_deg")
        ),
        "seed962_z_vs_previous_mean": _ratio(
            float(seed962["max_cube_z_delta_mean_m"]), prev_mean("max_cube_z_delta_mean_m")
        ),
        "seed962_1mm_rate_vs_previous_mean": _ratio(seed962_rates["max_ge_1mm_rate"], prev_rate("max_ge_1mm_rate")),
        "seed962_2mm_rate_vs_previous_mean": _ratio(seed962_rates["max_ge_2mm_rate"], prev_rate("max_ge_2mm_rate")),
        "seed962_3mm_rate_vs_previous_mean": _ratio(seed962_rates["max_ge_3mm_rate"], prev_rate("max_ge_3mm_rate")),
    }

    verdict = {
        "final_position_used_as_success_gate": False,
        "seed962_primary_tap_event_pass": (
            float(seed962["contact_evidence_rate"]) >= 1.0
            and float(seed962["reaction_event_rate"]) >= 1.0
            and float(seed962["overshoot_rate"]) == 0.0
            and seed962_rates["max_ge_1mm_rate"] >= 1.0
        ),
        "seed962_two_mm_transient_majority": seed962_rates["max_ge_2mm_rate"] >= 0.75,
        "seed962_three_mm_transient_not_reliable": seed962_rates["max_ge_3mm_rate"] < 0.75,
        "seed962_less_aggressive_than_pre020_predecessors": (
            comparisons["seed962_max_disp_vs_previous_mean"] < 0.75
            and comparisons["seed962_tip_vs_previous_mean"] < 0.50
            and comparisons["seed962_z_vs_previous_mean"] < 0.60
        ),
        "quality_still_blocks_data_readiness": (
            not bool(seed962["clean_diffik_teacher_window_ready"])
            and int(seed962["quality_tier_counts"].get("C_REACTION_VALID_FOLLOW_LAG", 0)) > 0
        ),
        "next_step_order": [
            "do_not_use_final_1cm_or_final_1mm_retention_as_primary_gate",
            "report_seed962_as_primary_1mm_tap_event_pass_if_minimal_tap_is_enough",
            "ask_or_define_explicit_transient_strength_target_before_any_stronger_tap_runtime",
            "if_target_is_3mm_then_design_one_optional_local_candidate_without_mixing_knobs",
            "if_target_is_1_to_2mm_then_stop_yplus_contact_geometry_tuning_and focus_quality_tier_metadata",
        ],
        "do_not_start": [
            "GPU_runtime_without_explicit_approval",
            "final_1cm_relocation_chasing",
            "blind_precontact_sweep",
            "lateral_height_actuator_dls_cap_mixing",
            "1024_10240_scaleup",
            "dataset_generation",
            "PPO_RL",
            "VLA",
            "TrackA",
            "B200_SSH",
        ],
    }

    result = {
        "artifact_type": "cube10cm_yplus_transient_tap_strength_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_posthoc_only": True,
        "no_gpu_isaaclab_training_dataset_ssh": True,
        "objective_contract": {
            "primary": "reaction_contact_no_posewrite_no_overshoot",
            "final_position_success_gate": False,
            "transient_thresholds_m": [0.001, 0.002, 0.003, 0.005, 0.010],
        },
        "run_summaries": run_summaries,
        "previous_seed_group": {
            "labels": [str(row["label"]) for row in previous],
            "max_disp_along_push_mean_m": prev_mean("max_disp_along_push_mean_m"),
            "max_tip_angle_mean_deg": prev_mean("max_tip_angle_mean_deg"),
            "max_cube_z_delta_mean_m": prev_mean("max_cube_z_delta_mean_m"),
            "max_ge_1mm_rate_mean": prev_rate("max_ge_1mm_rate"),
            "max_ge_2mm_rate_mean": prev_rate("max_ge_2mm_rate"),
            "max_ge_3mm_rate_mean": prev_rate("max_ge_3mm_rate"),
        },
        "comparisons": comparisons,
        "verdict": verdict,
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_yplus_transient_tap_strength_audit_v1 local_posthoc_only=YES "
        "gpu_runtime=NO dataset_generation=NO final_position_gate=NO",
        (
            "line2 seed962_primary_event "
            f"contact_rate={seed962['contact_evidence_rate']:.9f} "
            f"reaction_rate={seed962['reaction_event_rate']:.9f} "
            f"overshoot_rate={seed962['overshoot_rate']:.9f} "
            f"max_ge_1mm_rate={seed962_rates['max_ge_1mm_rate']:.9f}"
        ),
        (
            "line3 seed962_transient_strength "
            f"max_ge_2mm_rate={seed962_rates['max_ge_2mm_rate']:.9f} "
            f"max_ge_3mm_rate={seed962_rates['max_ge_3mm_rate']:.9f} "
            f"max_ge_5mm_rate={seed962_rates['max_ge_5mm_rate']:.9f} "
            f"max_disp_mean_m={seed962['max_disp_along_push_mean_m']:.9f}"
        ),
        (
            "line4 previous_yplus_strength "
            f"max_ge_2mm_rate_mean={prev_rate('max_ge_2mm_rate'):.9f} "
            f"max_ge_3mm_rate_mean={prev_rate('max_ge_3mm_rate'):.9f} "
            f"max_disp_mean_m={prev_mean('max_disp_along_push_mean_m'):.9f}"
        ),
        (
            "line5 seed962_aggression_reduction "
            f"max_disp_ratio={comparisons['seed962_max_disp_vs_previous_mean']:.9f} "
            f"tip_ratio={comparisons['seed962_tip_vs_previous_mean']:.9f} "
            f"z_ratio={comparisons['seed962_z_vs_previous_mean']:.9f}"
        ),
        (
            "line6 verdict "
            f"primary_1mm_tap_event_pass={verdict['seed962_primary_tap_event_pass']} "
            f"two_mm_transient_majority={verdict['seed962_two_mm_transient_majority']} "
            f"three_mm_transient_not_reliable={verdict['seed962_three_mm_transient_not_reliable']} "
            f"quality_still_blocks_data_readiness={verdict['quality_still_blocks_data_readiness']}"
        ),
        (
            "line7 next_order "
            "if_minimal_1_to_2mm_tap_enough=stop_yplus_contact_geometry_tuning "
            "if_3mm_required=define_explicit_transient_target_before_one_candidate "
            "do_not_use_final_1cm_or_final_retention"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
