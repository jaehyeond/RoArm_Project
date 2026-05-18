#!/usr/bin/env python3
"""Diagnostic-only guard probe for the pre-close candidate selector.

This wrapper keeps the authoritative selector implementation unchanged, but
drives it with an adversarial near-top invalid candidate: a final target just
below the sponge top and inside the nominal footprint. The point is to verify
that the selector rejects below-top/inside-footprint targets even when the 3mm
exact gate could make the motion look acceptable.

Still pre-integration only: no training, constraints, SurfaceGripper, transport,
release, scripted release variant, or env/train/chain default edits.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--invalid_top_margin_m", type=float, default=-0.0015)
    ap.add_argument("--above_margin_m", type=float, default=0.0010)
    ap.add_argument("--clearance_margin_m", type=float, default=0.0240)
    ap.add_argument("--side_margin_m", type=float, default=0.0020)
    ap.add_argument("--side_top_margin_m", type=float, default=0.0005)
    ap.add_argument(
        "selector_args",
        nargs=argparse.REMAINDER,
        help="Optional extra args passed through to the selector after '--'.",
    )
    args = ap.parse_args()

    selector = Path(__file__).with_name("p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py")
    passthrough = list(args.selector_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    print("[roarm_chain_preclose_selector_guard] preclose_selector_guard_probe", flush=True)
    print(
        "[roarm_chain_preclose_selector_guard] "
        "diagnostic_preclose_only=YES constraint_prim_insertion=NO fixed_dynamic_constraint_integration=NO "
        "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO "
        "scripted_release_variant=NO p7_training=NO p7_tuning=NO diagnostic_gate_tuning=NO "
        "env_default_edits=NO chain_defaults_edits=NO attach_physics_validated=NO "
        "release_physics_validated=NO claim_attach_success=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_preclose_selector_guard] guard_case invalid_top_margin_m={args.invalid_top_margin_m:.6f} "
        "expected_selector_behavior=REJECT reason=below_top_inside_footprint_invalid "
        "interpretation=below_top_inside_invalid_even_if_exact_gate_passes",
        flush=True,
    )

    sys.argv = [
        str(selector),
        "--top_margin_m",
        str(args.invalid_top_margin_m),
        "--above_margin_m",
        str(args.above_margin_m),
        "--clearance_margin_m",
        str(args.clearance_margin_m),
        "--side_margin_m",
        str(args.side_margin_m),
        "--side_top_margin_m",
        str(args.side_top_margin_m),
        *passthrough,
    ]
    try:
        runpy.run_path(str(selector), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
