#!/usr/bin/env python3
"""USD-selectable wrapper for the 2cm cube local grasp/close sweep.

This wrapper preserves the completed v4 diagnostic script and only sets
ROARM_M3_USD_PATH before delegating to it. It is diagnostic-only: no training,
no env/chain default edit, no constraints, no SurfaceGripper, no transport, and
no release. Use it to compare the current RoArm USD against a separately
authored diagnostic collision-geometry USD under the same grasp/lift gates.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim_scripts"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Set ROARM_M3_USD_PATH and delegate remaining arguments to "
            "p7_branch_b_cube2cm_local_grasp_close_sweep_probe.py."
        )
    )
    parser.add_argument("--robot_usd_path", required=True)
    parser.add_argument("--print_wrapper_scope", action="store_true")
    known, remaining = parser.parse_known_args()

    usd_path = Path(known.robot_usd_path).expanduser()
    if not usd_path.exists():
        raise FileNotFoundError(f"robot_usd_path does not exist: {usd_path}")
    if known.print_wrapper_scope:
        print(
            "[cube2cm_local_grasp_usd] wrapper_scope diagnostic_only=YES "
            "env_default_edits=NO chain_defaults_edits=NO p7_training=NO "
            "constraint_prim_insertion=NO surface_gripper=NO attached_transport=NO "
            "transport_target=NO release_marker=NO scripted_release_variant=NO "
            f"robot_usd_path={usd_path}",
            flush=True,
        )

    os.environ["ROARM_M3_USD_PATH"] = str(usd_path)
    sys.argv = [sys.argv[0], *remaining]

    from p7_branch_b_cube2cm_local_grasp_close_sweep_probe import main as delegated_main

    return int(delegated_main())


if __name__ == "__main__":
    raise SystemExit(main())
