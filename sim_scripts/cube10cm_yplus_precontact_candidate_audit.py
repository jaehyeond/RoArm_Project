"""Local candidate audit for the next 10cm y+ precontact screen.

This posthoc/config audit checks the narrowest proposed next GPU variable:
increase only y+ `precontact_clearance_m` before another tiny runtime. It reads
existing y+ diagnostics and computes the nominal target-geometry change.

No IsaacLab app, GPU runtime, training, dataset generation, robot control, SSH,
or trace mutation is performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_BASELINE_SUMMARY = (
    LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_seed958_summary.json"
)
DEFAULT_EARLY_GEOMETRY = LOG_DIR / "cube10cm_yplus_early_contact_geometry_audit_existing_seeds.json"
DEFAULT_TIERC_DIAGNOSTIC = LOG_DIR / "cube10cm_yplus_tierc_failure_diagnostic_existing_seeds.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_yplus_precontact_candidate_audit_existing_seeds.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_yplus_precontact_candidate_audit_existing_seeds_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(data: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        raw = data.get(key, default)
        if raw is None or raw == "":
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _nested_float(data: dict[str, Any], path: tuple[str, ...], default: float = 0.0) -> float:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    try:
        if cur is None or cur == "":
            return default
        return float(cur)
    except (TypeError, ValueError):
        return default


def _target_geometry(*, half_along_m: float, precontact_m: float, push_through_m: float) -> dict[str, float]:
    pre_target_along_m = -(half_along_m + precontact_m)
    near_face_through_along_m = -(half_along_m - push_through_m)
    return {
        "precontact_clearance_m": precontact_m,
        "pre_target_along_m": pre_target_along_m,
        "near_face_through_target_along_m": near_face_through_along_m,
        "push_phase_target_path_length_m": abs(near_face_through_along_m - pre_target_along_m),
    }


def _command_lines(
    *,
    candidate_precontact_m: float,
    out_stem: str,
    seed: int,
) -> list[str]:
    base = "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
    return [
        "OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output "
        "python -u sim_scripts/cube10cm_push_diffik_probe.py \\",
        "  --num_envs 16 \\",
        "  --episodes 1 \\",
        f"  --seed {seed} \\",
        "  --fixed_cube_x_m 0.295 \\",
        "  --fixed_cube_y_m -0.044 \\",
        "  --fixed_push_dir 0 1 \\",
        "  --base_lateral_offset_m -0.020 \\",
        "  --xneg_tcp_center_height_offset_m 0.050 \\",
        f"  --precontact_clearance_m {candidate_precontact_m:.3f} \\",
        "  --trace_diffik_diagnostics \\",
        "  --trace_all_envs \\",
        "  --trace_stride 4 \\",
        f"  --out_csv {base}/{out_stem}.csv \\",
        f"  --summary_json {base}/{out_stem}_summary.json \\",
        f"  --trace_csv {base}/{out_stem}_trace.csv",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--early_geometry_json", type=Path, default=DEFAULT_EARLY_GEOMETRY)
    parser.add_argument("--tierc_diagnostic_json", type=Path, default=DEFAULT_TIERC_DIAGNOSTIC)
    parser.add_argument("--candidate_precontact_m", type=float, default=0.020)
    parser.add_argument("--candidate_seed", type=int, default=962)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    baseline_summary = _load_json(args.baseline_summary)
    early_geometry = _load_json(args.early_geometry_json)
    tierc_diagnostic = _load_json(args.tierc_diagnostic_json)

    cube_size = baseline_summary.get("cube_size_m", [0.100, 0.100, 0.100])
    cube_size_y = float(cube_size[1])
    half_along_m = cube_size_y / 2.0
    baseline_precontact_m = _float(baseline_summary, "precontact_clearance_m", 0.010)
    push_through_m = _float(baseline_summary, "push_through_m", 0.010)
    candidate_precontact_m = float(args.candidate_precontact_m)

    baseline_geometry = _target_geometry(
        half_along_m=half_along_m,
        precontact_m=baseline_precontact_m,
        push_through_m=push_through_m,
    )
    candidate_geometry = _target_geometry(
        half_along_m=half_along_m,
        precontact_m=candidate_precontact_m,
        push_through_m=push_through_m,
    )
    delta_precontact_m = candidate_precontact_m - baseline_precontact_m
    delta_push_path_m = (
        candidate_geometry["push_phase_target_path_length_m"]
        - baseline_geometry["push_phase_target_path_length_m"]
    )

    target = early_geometry.get("group_summaries", {}).get("target_yplus_tier_c", {})
    baseline = early_geometry.get("group_summaries", {}).get("baseline_tier_b_all_non_yplus", {})
    early_verdict = early_geometry.get("verdict", {})
    tierc_verdict = tierc_diagnostic.get("verdict", {})

    pre24_disp_ratio = _nested_float(early_geometry, ("verdict", "pre24_disp_ratio"))
    pre24_tip_ratio = _nested_float(early_geometry, ("verdict", "pre24_tip_ratio"))
    anchor_minus_push_start = _float(target, "anchor_minus_first_push_phase_step_mean", 0.0)
    baseline_anchor_minus_push_start = _float(baseline, "anchor_minus_first_push_phase_step_mean", 0.0)
    raw_delta_ratio = _nested_float(tierc_diagnostic, ("verdict", "raw_delta_abs_max_p95_ratio"), 0.0)
    follow_ratio = _nested_float(tierc_diagnostic, ("verdict", "joint_follow_p95_to_cap_ratio"), 0.0)
    supports_simple_raw_ik = bool(
        tierc_verdict.get(
            "supports_simple_raw_ik_demand_hypothesis",
            tierc_verdict.get("supports_simple_raw_ik_demand", False),
        )
    )

    supports_precontact_first = (
        bool(early_verdict.get("supports_yplus_preanchor_reaction_accumulation"))
        and bool(early_verdict.get("supports_yplus_approach_phase_geometry_hypothesis"))
        and not supports_simple_raw_ik
        and delta_precontact_m > 0.0
        and math.isclose(push_through_m, 0.010, rel_tol=0.0, abs_tol=1.0e-12)
    )
    candidate_is_tiny = 0.0 < delta_precontact_m <= 0.010000001

    pre_mm = int(round(candidate_precontact_m * 1000.0))
    out_stem = (
        "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_"
        f"pre{pre_mm:03d}_seed{int(args.candidate_seed)}"
    )
    command_lines = _command_lines(
        candidate_precontact_m=candidate_precontact_m,
        out_stem=out_stem,
        seed=int(args.candidate_seed),
    )

    result = {
        "artifact_type": "cube10cm_yplus_precontact_candidate_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_posthoc_config_only": True,
        "no_gpu_isaaclab_training_dataset_ssh": True,
        "source_files": {
            "baseline_summary": str(args.baseline_summary),
            "early_geometry_json": str(args.early_geometry_json),
            "tierc_diagnostic_json": str(args.tierc_diagnostic_json),
        },
        "objective_contract": {
            "primary": "reaction_contact_no_posewrite_no_overshoot",
            "final_1cm_relocation": "secondary_not_used_for_this_candidate",
            "tap_threshold_m": 0.001,
        },
        "baseline_geometry": baseline_geometry,
        "candidate_geometry": candidate_geometry,
        "candidate_change": {
            "changed_variable": "precontact_clearance_m",
            "baseline_precontact_m": baseline_precontact_m,
            "candidate_precontact_m": candidate_precontact_m,
            "delta_precontact_m": delta_precontact_m,
            "delta_push_phase_target_path_length_m": delta_push_path_m,
            "unchanged_push_through_m": push_through_m,
            "unchanged_lateral_offset_m": _float(baseline_summary, "base_lateral_offset_m", -0.020),
            "unchanged_tcp_center_height_offset_m": _float(baseline_summary, "tcp_center_height_offset_m", 0.0),
            "unchanged_fixed_push_dir": baseline_summary.get("fixed_push_dir", [0.0, 1.0]),
            "unchanged_fixed_cube_xy_m": [
                _float(baseline_summary, "fixed_cube_x_m", 0.295),
                _float(baseline_summary, "fixed_cube_y_m", -0.044),
            ],
        },
        "evidence": {
            "yplus_windows": int(target.get("window_count", 0)),
            "yplus_pre24_disp_mean_m": _float(target, "pre24_max_disp_xy_m_mean", 0.0),
            "baseline_pre24_disp_mean_m": _float(baseline, "pre24_max_disp_xy_m_mean", 0.0),
            "pre24_disp_ratio": pre24_disp_ratio,
            "pre24_tip_ratio": pre24_tip_ratio,
            "yplus_anchor_minus_push_start_steps": anchor_minus_push_start,
            "baseline_anchor_minus_push_start_steps": baseline_anchor_minus_push_start,
            "raw_delta_ratio": raw_delta_ratio,
            "follow_ratio": follow_ratio,
        },
        "screening_decision": {
            "supports_precontact_first": supports_precontact_first,
            "candidate_is_tiny_one_variable_change": candidate_is_tiny,
            "reject_height_first_reason": "seed944 height050 improved TCP error but killed contact; y- low side-center still has low pre-anchor motion",
            "reject_lateral_first_reason": "lateral changes side asymmetry; precontact directly targets approach-phase early contact first",
            "known_risk": "push phase target path length increases from baseline by delta_precontact_m; the runtime may reduce early contact but still stay Tier C or lose contact",
            "go_no_go_order": [
                "reaction_evidence",
                "contact_evidence",
                "no_posewrite",
                "no_overshoot",
                "quality_tier_A_B_C_metadata",
                "final_1cm_secondary_only_if_explicitly_requested",
            ],
            "do_not_start": ["1024_10240_scaleup", "dataset_generation", "PPO_RL", "VLA", "TrackA"],
        },
        "proposed_tiny_gpu_command_not_run": command_lines,
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_yplus_precontact_candidate_audit_v1 local_posthoc_config_only=YES gpu_runtime=NO dataset_generation=NO",
        (
            "line2 evidence "
            f"yplus_windows={target.get('window_count', 0)} "
            f"pre24_disp_ratio={pre24_disp_ratio:.9f} "
            f"pre24_tip_ratio={pre24_tip_ratio:.9f} "
            f"anchor_minus_push_start_steps={anchor_minus_push_start:.6f} "
            f"raw_delta_ratio={raw_delta_ratio:.9f} "
            f"follow_ratio={follow_ratio:.9f}"
        ),
        (
            "line3 candidate_change "
            f"changed_variable=precontact_clearance_m "
            f"baseline_m={baseline_precontact_m:.6f} "
            f"candidate_m={candidate_precontact_m:.6f} "
            f"delta_m={delta_precontact_m:.6f} "
            f"pre_target_along_baseline_m={baseline_geometry['pre_target_along_m']:.6f} "
            f"pre_target_along_candidate_m={candidate_geometry['pre_target_along_m']:.6f}"
        ),
        (
            "line4 geometry_invariant "
            f"through_target_along_m={candidate_geometry['near_face_through_target_along_m']:.6f} "
            f"baseline_push_path_m={baseline_geometry['push_phase_target_path_length_m']:.6f} "
            f"candidate_push_path_m={candidate_geometry['push_phase_target_path_length_m']:.6f} "
            f"delta_push_path_m={delta_push_path_m:.6f}"
        ),
        (
            "line5 decision "
            f"supports_precontact_first={supports_precontact_first} "
            f"candidate_is_tiny_one_variable_change={candidate_is_tiny} "
            "height_first=REJECTED_FOR_NOW "
            "lateral_first=REJECTED_FOR_NOW"
        ),
        "line6 runtime_status=NOT_RUN requires_explicit_GPU_APPROVAL next_seed=962 command_in_json=proposed_tiny_gpu_command_not_run",
        "line7 reporting_order=reaction/contact/no-posewrite/no-overshoot first; quality tier second; final_1cm secondary only if explicit",
        "line8 do_not_start=1024_10240_scaleup,dataset_generation,PPO_RL,VLA,TrackA",
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
