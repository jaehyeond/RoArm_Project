"""Local contract audit for the professor 10cm tap/reaction objective.

This script prevents a repeat of the 1cm-relocation/default confusion. It imports
the 10cm wrapper only, checks the injected defaults, and exits nonzero if the
wrapper encodes final 1cm relocation as the default objective.

No IsaacLab app, GPU runtime, training, dataset generation, robot control, or log
mutation is performed.
"""
from __future__ import annotations

import math
import sys
import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim_scripts import cube10cm_push_diffik_probe as wrapper


DEFAULT_OUT_JSON = wrapper.LOG_DIR / "cube10cm_tap_objective_contract_audit.json"
TAP_DISP_DEFAULTS = {
    "--cube_push_target_disp_m": 0.001,
    "--cube_success_disp_m": 0.001,
    "--gate_disp_m": 0.001,
    "--contact_stop_disp_m": 0.001,
}
OBJECT_DEFAULTS = {
    "--cube_mass_kg": 0.720,
}
OBJECT_SIZE = ("0.100", "0.100", "0.100")


def _value(args: list[str], option: str) -> str:
    try:
        return args[args.index(option) + 1]
    except (ValueError, IndexError) as exc:
        raise AssertionError(f"missing value for {option}") from exc


def _float_value(args: list[str], option: str) -> float:
    return float(_value(args, option))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    args_ns = parser.parse_args()

    args = wrapper._with_professor_10cm_defaults([])
    failures: list[str] = []

    size_idx = args.index("--cube_size_m") if "--cube_size_m" in args else -1
    size = tuple(args[size_idx + 1 : size_idx + 4]) if size_idx >= 0 else ()
    if size != OBJECT_SIZE:
        failures.append(f"cube_size_m expected={OBJECT_SIZE} actual={size}")

    for option, expected in OBJECT_DEFAULTS.items():
        actual = _float_value(args, option)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            failures.append(f"{option} expected={expected} actual={actual}")

    for option, expected in TAP_DISP_DEFAULTS.items():
        actual = _float_value(args, option)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            failures.append(f"{option} expected_tap_default={expected} actual={actual}")
        if math.isclose(actual, 0.010, rel_tol=0.0, abs_tol=1.0e-12):
            failures.append(f"{option} incorrectly defaults to final 1cm relocation")

    override_args = wrapper._with_professor_10cm_defaults(["--gate_disp_m", "0.010"])
    override_count = override_args.count("--gate_disp_m")
    override_value = _float_value(override_args, "--gate_disp_m")
    if override_count != 1 or not math.isclose(override_value, 0.010, rel_tol=0.0, abs_tol=1.0e-12):
        failures.append(
            f"explicit relocation override not preserved: count={override_count} value={override_value}"
        )

    result = {
        "contract": "professor_cube10cm_tap_reaction",
        "primary_objective": "reaction_contact_no_posewrite_no_overshoot",
        "final_1cm_relocation_default": False,
        "cube_size_m": [float(x) for x in size],
        "cube_mass_kg": _float_value(args, "--cube_mass_kg"),
        "tap_defaults": {option: _float_value(args, option) for option in TAP_DISP_DEFAULTS},
        "explicit_1cm_override_allowed": True,
        "explicit_gate_disp_override_m": override_value,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args_ns.out_json.parent.mkdir(parents=True, exist_ok=True)
    args_ns.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[cube10cm_tap_objective_contract_audit] contract=professor_cube10cm_tap_reaction")
    print("[cube10cm_tap_objective_contract_audit] primary_objective=reaction_contact_no_posewrite_no_overshoot")
    print("[cube10cm_tap_objective_contract_audit] final_1cm_relocation_default=NO")
    print(f"[cube10cm_tap_objective_contract_audit] cube_size_m={' '.join(size)}")
    print(f"[cube10cm_tap_objective_contract_audit] cube_mass_kg={_float_value(args, '--cube_mass_kg'):.6f}")
    for option in TAP_DISP_DEFAULTS:
        print(f"[cube10cm_tap_objective_contract_audit] {option}={_float_value(args, option):.6f}")
    print(
        "[cube10cm_tap_objective_contract_audit] explicit_1cm_override_allowed=YES "
        f"gate_disp_override={override_value:.6f}"
    )

    print(f"[cube10cm_tap_objective_contract_audit] out_json={args_ns.out_json}")
    if failures:
        for failure in failures:
            print(f"[cube10cm_tap_objective_contract_audit] FAIL {failure}")
        return 2

    print("[cube10cm_tap_objective_contract_audit] verdict=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
