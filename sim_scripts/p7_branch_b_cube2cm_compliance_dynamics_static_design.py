#!/usr/bin/env python3
"""Static dynamics design audit for the next P7 Branch B compliance proxy.

This is a design calculator, not a simulator. It reads the already-encoded v7
close_26 samples from the static compliance audit and quantifies what the next
runtime candidate would have to change in close-step telemetry.

No Isaac launch, runtime telemetry, training, dataset generation, default edits,
constraint insertion, SurfaceGripper, transport, release, gate tuning, or success
claim happens here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from p7_branch_b_cube2cm_compliance_proxy_static_analysis import V7_CLOSE_SAMPLES


@dataclass(frozen=True)
class Candidate:
    name: str
    assumption: str
    gate: str
    static_verdict: str
    reason: str
    required_telemetry_change: str
    falsifier: str


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _ratio(required: float, observed: float) -> float:
    if observed <= 0.0:
        return 0.0
    return required / observed


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push_speed_gate_mps", type=float, default=0.005)
    ap.add_argument("--push_drift_gate_m", type=float, default=0.00020)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--current_object_mass_kg", type=float, default=0.020)
    ap.add_argument("--max_plausible_diagnostic_mass_kg", type=float, default=0.050)
    ap.add_argument("--support_budget_m", type=float, default=0.002)
    args = ap.parse_args()

    if args.push_speed_gate_mps <= 0.0:
        raise ValueError("push_speed_gate_mps must be positive")
    if args.current_object_mass_kg <= 0.0:
        raise ValueError("current_object_mass_kg must be positive")
    if args.max_plausible_diagnostic_mass_kg <= 0.0:
        raise ValueError("max_plausible_diagnostic_mass_kg must be positive")
    if args.support_budget_m < 0.0:
        raise ValueError("support_budget_m must be non-negative")

    samples_by_step = {sample.step: sample for sample in V7_CLOSE_SAMPLES}
    step3 = samples_by_step[3]
    step4 = samples_by_step[4]
    step5 = samples_by_step[5]
    final = samples_by_step[45]

    step3_residual_ratio = _ratio(args.push_speed_gate_mps, step3.object_speed_mps)
    step4_residual_ratio = _ratio(args.push_speed_gate_mps, step4.object_speed_mps)
    step5_residual_ratio = _ratio(args.push_speed_gate_mps, step5.object_speed_mps)
    worst_residual_ratio_3_to_5 = min(step3_residual_ratio, step4_residual_ratio, step5_residual_ratio)
    required_speed_suppression = max(0.0, 1.0 - worst_residual_ratio_3_to_5)

    mass_needed_step3 = args.current_object_mass_kg * step3.object_speed_mps / args.push_speed_gate_mps
    mass_needed_step4 = args.current_object_mass_kg * step4.object_speed_mps / args.push_speed_gate_mps
    mass_needed_step5 = args.current_object_mass_kg * step5.object_speed_mps / args.push_speed_gate_mps
    mass_needed_worst = max(mass_needed_step3, mass_needed_step4, mass_needed_step5)
    mass_only_plausible = mass_needed_worst <= args.max_plausible_diagnostic_mass_kg

    support_step4_ok = max(step4.counter_gap_m) <= args.support_budget_m
    support_step5_ok = max(step5.counter_gap_m) <= args.support_budget_m
    final_support_ok = max(final.counter_gap_m) <= args.support_budget_m

    target_step4_ok_if_unchanged = step4.target_error_m <= args.target_error_gate_m
    final_target_ok_if_unchanged = final.target_error_m <= args.target_error_gate_m

    candidates = (
        Candidate(
            name="label_only_contact_patch",
            assumption="Foam compliance is represented only by expanding the contact/support envelope.",
            gate="static/prep only",
            static_verdict="REJECT",
            reason=(
                "A 2mm support envelope can cover step 4, but the observed speed push remains over the "
                "existing gate."
            ),
            required_telemetry_change=(
                "Would still need step3/4 speed below push gate; support relabeling alone does not provide it."
            ),
            falsifier="Any future result that changes only contact labels while step3 speed remains above gate.",
        ),
        Candidate(
            name="mass_only_inertia",
            assumption="The same asymmetric impulse is tolerated by making the cube heavier.",
            gate="static design; runtime close contact only if mass remains plausible",
            static_verdict="REJECT" if not mass_only_plausible else "WEAK_CANDIDATE",
            reason=(
                f"Constant-impulse estimate needs object mass {mass_needed_worst:.3f}kg to hold steps 3-5 below "
                f"{args.push_speed_gate_mps:.3f}m/s from current {args.current_object_mass_kg:.3f}kg."
            ),
            required_telemetry_change="Object speed below push gate without increasing counter gap or target error.",
            falsifier="Required mass exceeds plausible diagnostic mass or runtime still shows one-sided push.",
        ),
        Candidate(
            name="soft_contact_material_diagnostic",
            assumption=(
                "Softer contact response / higher effective damping absorbs the initial moving-jaw impulse before "
                "the cube exits the counter-support basin."
            ),
            gate="future runtime close contact, after separate approval",
            static_verdict="SELECT_AS_MINIMAL_RUNTIME_MECHANISM",
            reason=(
                f"Needs at least {_pct(required_speed_suppression)} speed suppression across steps 3-5 while "
                f"keeping counter gap <= {args.support_budget_m:.3f}m through step 4."
            ),
            required_telemetry_change=(
                "step3 object_speed_mps <= push_speed_gate, one_sided_push=NO, counter support still true at step4, "
                "close_reached=YES or at least target_error not worse than the 3mm gate."
            ),
            falsifier="step3 speed remains above gate, counter support disappears by step4, or target_error grows.",
        ),
        Candidate(
            name="virtual_compression_plus_damping",
            assumption=(
                "Foam is modeled as bounded compression with explicit impulse damping; support exists within the "
                "compression budget and asymmetric velocity is suppressed."
            ),
            gate="static/prep model first; future runtime close contact only after approval",
            static_verdict="RESERVE_IF_MATERIAL_DIAGNOSTIC_FAILS",
            reason=(
                "It directly represents the needed mechanism but is more artificial than a material/contact-parameter "
                "diagnostic."
            ),
            required_telemetry_change=(
                "same as soft-contact diagnostic, plus logged compression budget must stay bounded and below the "
                "declared support budget."
            ),
            falsifier="Needs more than the declared compression budget or behaves like posewrite/attachment.",
        ),
    )

    print("[cube2cm_compliance_dynamics_static] local_static_only=YES isaac_run=NO training=NO dataset_generation=NO")
    print(
        "[cube2cm_compliance_dynamics_static] diagnostic_only=YES env_default_edits=NO chain_defaults_edits=NO "
        "constraint_prim_insertion=NO surface_gripper=NO attached_transport=NO transport_target=NO "
        "release_marker=NO scalar_or_gate_tuning=NO success_claim=NO",
        flush=True,
    )
    print(
        f"[cube2cm_compliance_dynamics_static] source_samples="
        "/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:377-379,419 "
        f"push_speed_gate_mps={args.push_speed_gate_mps:.6f} push_drift_gate_m={args.push_drift_gate_m:.6f} "
        f"target_error_gate_m={args.target_error_gate_m:.6f} support_budget_m={args.support_budget_m:.6f}",
        flush=True,
    )
    print(
        f"[cube2cm_compliance_dynamics_static] speed_requirement "
        f"step3_speed_mps={step3.object_speed_mps:.6f} step3_allowed_residual_ratio={step3_residual_ratio:.6f} "
        f"step4_speed_mps={step4.object_speed_mps:.6f} step4_allowed_residual_ratio={step4_residual_ratio:.6f} "
        f"step5_speed_mps={step5.object_speed_mps:.6f} step5_allowed_residual_ratio={step5_residual_ratio:.6f} "
        f"required_speed_suppression_3_to_5={required_speed_suppression:.6f}",
        flush=True,
    )
    print(
        f"[cube2cm_compliance_dynamics_static] mass_only_check current_object_mass_kg={args.current_object_mass_kg:.6f} "
        f"mass_needed_step3_kg={mass_needed_step3:.6f} mass_needed_step4_kg={mass_needed_step4:.6f} "
        f"mass_needed_step5_kg={mass_needed_step5:.6f} mass_needed_worst_kg={mass_needed_worst:.6f} "
        f"max_plausible_diagnostic_mass_kg={args.max_plausible_diagnostic_mass_kg:.6f} "
        f"mass_only_plausible={_yes(mass_only_plausible)}",
        flush=True,
    )
    print(
        f"[cube2cm_compliance_dynamics_static] support_and_target_check "
        f"step4_counter_gap_m={max(step4.counter_gap_m):.6f} support_step4_ok={_yes(support_step4_ok)} "
        f"step5_counter_gap_m={max(step5.counter_gap_m):.6f} support_step5_ok={_yes(support_step5_ok)} "
        f"final_counter_gap_m={max(final.counter_gap_m):.6f} final_support_ok={_yes(final_support_ok)} "
        f"step4_target_error_m={step4.target_error_m:.6f} target_step4_ok_if_unchanged={_yes(target_step4_ok_if_unchanged)} "
        f"final_target_error_m={final.target_error_m:.6f} final_target_ok_if_unchanged={_yes(final_target_ok_if_unchanged)}",
        flush=True,
    )

    for candidate in candidates:
        print(
            f"[cube2cm_compliance_dynamics_static] candidate name={candidate.name} "
            f"gate='{candidate.gate}' static_verdict={candidate.static_verdict} "
            f"assumption='{candidate.assumption}' reason='{candidate.reason}' "
            f"required_telemetry_change='{candidate.required_telemetry_change}' "
            f"falsifier='{candidate.falsifier}'",
            flush=True,
        )

    print(
        "[cube2cm_compliance_dynamics_static] selected_next_static_design="
        "soft_contact_material_diagnostic_first",
        flush=True,
    )
    print(
        "[cube2cm_compliance_dynamics_static] future_close26_pass_criteria="
        "approach_ok=YES descend_ok=YES close_reached=YES step3_speed_below_gate=YES "
        "one_sided_push_steps_2_to_4=NO counter_support_step4=YES attach_calls=0 posewrite_calls=0 success_claim=NO",
        flush=True,
    )
    print("[cube2cm_compliance_dynamics_static] CUBE2CM_COMPLIANCE_DYNAMICS_STATIC_DESIGN_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
