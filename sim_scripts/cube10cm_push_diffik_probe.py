"""10cm/0.72kg professor cube push/tap DiffIK entrypoint.

This is a thin wrapper around the shared DiffIK probe engine. The legacy engine
file still carries the original 3cm name because it also owns the earlier 3cm
logs/dataset tooling, but the runtime object parameters are configurable. This
entrypoint injects professor-branch 10cm defaults unless the caller explicitly
overrides them.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim_scripts import cube3cm_push_diffik_probe as shared_probe


LOG_DIR = shared_probe.LOG_DIR
PROFESSOR_10CM_DEFAULTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("--cube_size_m", ("0.100", "0.100", "0.100")),
    ("--cube_mass_kg", ("0.720",)),
    ("--cube_push_target_disp_m", ("0.001",)),
    ("--cube_success_disp_m", ("0.001",)),
    ("--gate_disp_m", ("0.001",)),
    ("--tcp_height_mode", ("side_center",)),
    ("--through_target_mode", ("near_face",)),
    ("--contact_controller_mode", ("measured_stop",)),
    ("--contact_stop_target_mode", ("freeze",)),
    ("--contact_detect_disp_m", ("0.001",)),
    ("--contact_stop_disp_m", ("0.001",)),
    ("--contact_overshoot_disp_m", ("0.020",)),
    ("--contact_near_joint_step_scale", ("1.0",)),
    ("--contact_stop_joint_step_scale", ("0.2",)),
    ("--precontact_clearance_m", ("0.010",)),
    ("--push_through_m", ("0.010",)),
    ("--approach_steps", ("220",)),
    ("--push_steps", ("90",)),
    ("--post_steps", ("80",)),
    ("--max_diffik_joint_step_rad", ("0.035",)),
    ("--arm_stiffness_override", ("400",)),
    ("--arm_damping_override", ("20",)),
    ("--arm_effort_limit_sim_override", ("25",)),
    ("--arm_velocity_limit_sim_override", ("12",)),
    ("--out_csv", (str(LOG_DIR / "diffik_probe_cube10cm_m072_probe.csv"),)),
    ("--summary_json", (str(LOG_DIR / "diffik_probe_cube10cm_m072_probe_summary.json"),)),
)


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _with_professor_10cm_defaults(argv: Sequence[str]) -> list[str]:
    args: list[str] = []
    for option, values in PROFESSOR_10CM_DEFAULTS:
        if not _has_option(argv, option):
            args.append(option)
            args.extend(values)
    args.extend(argv)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    return shared_probe.main(_with_professor_10cm_defaults(user_args))


if __name__ == "__main__":
    raise SystemExit(main())
