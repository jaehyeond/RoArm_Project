#!/usr/bin/env python3
"""p13 / t3x_bite81 — IK-conditioned bite-window preflight (case g0b_d420).

This is a NEW, read-only derivative.  It never edits an asset and never launches
Isaac.  It measures the frozen attempt3 collision hulls, separates one-jaw
(`unilateral`) admission from true two-jaw (`bilateral`) admission, and then
places selected measurements at actual p10 IK/FK poses to check the finite
D29xH50 cylinder and the support plane.

Preregistration:
  claudedocs/runtime_logs/grasp_track/g0b_d420/t3x_bite81_prereg.md

The scientific result is a preflight for PhysX, not a grasp-success claim.
PhysX must run regardless of whether a bilateral static window is found.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
USD_LIBS = (
    Path("/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages")
    / "isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
_REEXEC_FLAG = "G0B_JAW_AUDIT_REEXEC"
LOG = "p13_t3x_bite81"


def _bootstrap_pxr_env() -> None:
    """Plain-python pxr bootstrap; Kit/Isaac is deliberately not launched."""
    if os.environ.get(_REEXEC_FLAG) == "1":
        return
    if not USD_LIBS.is_dir():
        print(f"[{LOG}] ABORT missing_usd_libs={USD_LIBS}", flush=True)
        raise SystemExit(3)
    conda_lib = str(Path(sys.executable).resolve().parents[1] / "lib")
    env = dict(os.environ)
    env[_REEXEC_FLAG] = "1"
    env["PYTHONPATH"] = str(USD_LIBS) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    extra = f"{USD_LIBS / 'bin'}:{conda_lib}"
    env["LD_LIBRARY_PATH"] = extra + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


_bootstrap_pxr_env()

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "sim_scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "sim_scripts"))

CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"
TAG = "t3x_bite81"
PREREG = CASE_DIR / f"{TAG}_prereg.md"

P10_PATH = REPO / "sim_scripts/p10_g0b_t3t_cyld29h50_tilted_close_sweep_grasp_probe.py"
P12_PATH = REPO / "sim_scripts/p12_g0b_t3w_reach_boundary_radius_azimuth_sweep.py"
N8_PATH = REPO / "sim_scripts/g0b_t3r_n8_tilt_admission_readonly_audit.py"
JAW_PATH = REPO / "sim_scripts/g0b_t3_attempt3_jaw_throat_occlusion_readonly_vertex_audit.py"
N10_RESULTS = CASE_DIR / "t3r_n10_ctq5_results.json"
T3W_RESULTS = CASE_DIR / "t3w_reach1_results.json"
T3W_GRID = CASE_DIR / "t3w_reach1_grid.npz"
URDF_PATH = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"

EXPECTED_SHA256 = {
    P10_PATH: "63c6b2127d969e3291da6943eab6da1037034c154a8f21fe447519cbcb2f6cff",
    P12_PATH: "f8703fa2ee1db16c74445291a5fba3a6d4330b2e7f35c8936842afce6ebe4aca",
    N8_PATH: "84ab44dc9d9d87afa060280f967762a7e4298190ed6458f60418223f9801d5e7",
    JAW_PATH: "bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3",
    N10_RESULTS: "236243d4cfaa58aea76345662db42876154d5f55b885b960378f92d4f51d4c43",
    T3W_RESULTS: "a6186811537007957fb0b93c342fd908208041e5f4713533d49231dd22aeffe0",
    T3W_GRID: "b2a41d0aed4deba7544654d4fded44a2820ce90310cc623343e466ae9144e57a",
    URDF_PATH: "64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2",
}

NUMPY_PIN = "1.26.0"
PSUTIL_PIN = "5.9.8"
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"

OBJ_DIAM_M = 0.029
OBJ_RADIUS_M = OBJ_DIAM_M / 2.0
OBJ_HEIGHT_M = 0.050
SUPPORT_Z_M = 0.0
WALL_SPAN_M = 0.006
HULL_SAMPLE_UNCERTAINTY_M = 0.0005
NUMERIC_PEN_EPS_M = 1.0e-6
WRIST_R_V6_DEG = (-90.0, 90.0)
TARGET_ERROR_GATE_M = 0.003
TARGET_TILT_GATE_DEG = 5.0
DEFAULT_MARGIN_M = -0.0010997957078144082
Q5_OPEN_DEG = 88.30998496351378
CONTROL_THETA = (6.0, 15.0, 24.0, 29.0, 35.0)
PHYSICS_CONTROL_THETA = (6.0, 15.0, 24.0, 35.0, 60.0, 69.0)
REGRESSION_THETA = (29.0, 35.0)


class _PlanArgs:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # p10 contains dataclasses; dataclasses resolves postponed annotations via
    # sys.modules[cls.__module__] while the module is being executed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _finite_bite(
    prep: tuple[np.ndarray, np.ndarray, np.ndarray],
    chat_z: float,
    delta_m: float,
) -> float | None:
    """Signed deepest material beside the wall, capped at the finite bottom.

    Negative ``u`` is retained: it means the closest jaw material is still above
    the top face and is required to reproduce n10's signed-bite controls.
    """
    u0, rho0_sq, b = prep
    a = 1.0 - chat_z * chat_z
    rho_sq = rho0_sq + 2.0 * delta_m * b + delta_m * delta_m * a
    u = u0 - delta_m * chat_z
    beside = (
        (rho_sq >= OBJ_RADIUS_M * OBJ_RADIUS_M)
        & (rho_sq <= (OBJ_RADIUS_M + WALL_SPAN_M) ** 2)
        & (u <= OBJ_HEIGHT_M)
    )
    return float(u[beside].max() * 1000.0) if beside.any() else None


def _finite_inside_count(
    prep: tuple[np.ndarray, np.ndarray, np.ndarray],
    chat_z: float,
    delta_m: float,
) -> int:
    u0, rho0_sq, b = prep
    a = 1.0 - chat_z * chat_z
    rho_sq = rho0_sq + 2.0 * delta_m * b + delta_m * delta_m * a
    u = u0 - delta_m * chat_z
    return int(
        (
            (rho_sq < (OBJ_RADIUS_M - NUMERIC_PEN_EPS_M) ** 2)
            & (u > NUMERIC_PEN_EPS_M)
            & (u < OBJ_HEIGHT_M - NUMERIC_PEN_EPS_M)
        ).sum()
    )


def _metric(row: dict[str, Any], name: str) -> float:
    value = row.get(f"{name}_bite_mm")
    return -math.inf if value is None else float(value)


def _positive_windows(rows: list[dict[str, Any]], name: str, fine_step: float) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_q: float | None = None
    for row in rows:
        positive = _metric(row, name) > 0.0
        contiguous = previous_q is not None and row["q5_deg"] - previous_q <= 0.55
        if positive:
            if current and not contiguous:
                groups.append(current)
                current = []
            current.append(row)
        elif current:
            groups.append(current)
            current = []
        previous_q = float(row["q5_deg"])
    if current:
        groups.append(current)
    return [
        {
            "q5_lo_deg": float(group[0]["q5_deg"]),
            "q5_hi_deg": float(group[-1]["q5_deg"]),
            "edge_uncertainty_deg": float(fine_step),
        }
        for group in groups
    ]


def _window_targets(window: list[float] | None) -> list[float]:
    if window is None:
        return []
    lo, hi = window
    return [round(lo + frac * (hi - lo), 1) for frac in (0.25, 0.50, 0.75)]


def _scan_orientation(
    n8: Any,
    fixed_pts: np.ndarray,
    moving_fn: Any,
    chat: np.ndarray,
    q5_max_deg: float,
    coarse_step_deg: float,
    fine_step_deg: float,
) -> dict[str, Any]:
    """Adaptive q5 scan.  Coarse scan is global; 0.1 deg refinement covers every
    detected window, every sign transition, and the best point when no window is
    detected.  The detection resolution is disclosed in the output."""
    c0 = np.array([0.0, 0.0, float(n8.TCP_Z_MM) / 1000.0], dtype=np.float64)
    fixed_prep = n8.prep(fixed_pts, chat, c0)
    d_fixed, _ = n8.deepest_delta(*fixed_prep, float(chat[2]), OBJ_RADIUS_M)

    cache: dict[float, dict[str, Any]] = {}

    def evaluate(q5_deg: float) -> dict[str, Any]:
        q = round(float(q5_deg), 6)
        if q in cache:
            return cache[q]
        moving_pts = moving_fn(q)
        moving_prep = n8.prep(moving_pts, chat, c0)
        d_moving, _ = n8.deepest_delta(*moving_prep, float(chat[2]), OBJ_RADIUS_M)
        delta = max(d_fixed, d_moving)
        fixed_bite = moving_bite = None
        pen_count = None
        if math.isfinite(delta):
            fixed_bite = _finite_bite(fixed_prep, float(chat[2]), delta)
            moving_bite = _finite_bite(moving_prep, float(chat[2]), delta)
            pen_count = (
                _finite_inside_count(fixed_prep, float(chat[2]), delta + 1.0e-9)
                + _finite_inside_count(moving_prep, float(chat[2]), delta + 1.0e-9)
            )
        finite_bites = [v for v in (fixed_bite, moving_bite) if v is not None]
        unilateral = max(finite_bites) if finite_bites else None
        bilateral = min(fixed_bite, moving_bite) if fixed_bite is not None and moving_bite is not None else None
        row = {
            "q5_deg": q,
            "depth_top_min_mm": float(delta * 1000.0) if math.isfinite(delta) else None,
            "delta_m": float(delta) if math.isfinite(delta) else None,
            "bite_fixed_mm": fixed_bite,
            "bite_moving_mm": moving_bite,
            "unilateral_bite_mm": unilateral,
            "bilateral_bite_mm": bilateral,
            "finite_inside_count_at_delta_plus_1e-9m": pen_count,
            "blocker": "fixed" if d_fixed >= d_moving else "moving",
        }
        cache[q] = row
        return row

    coarse_q = np.unique(
        np.concatenate(
            ([np.arange(0.0, q5_max_deg + 1.0e-9, coarse_step_deg),
              [q5_max_deg, Q5_OPEN_DEG, 24.0]])
        )
    )
    coarse = [evaluate(float(q)) for q in coarse_q]

    refine_q: set[float] = set()
    for name in ("unilateral", "bilateral"):
        vals = [_metric(row, name) for row in coarse]
        flags = [value > 0.0 for value in vals]
        positive_idx = [i for i, flag in enumerate(flags) if flag]
        if positive_idx:
            lo = max(0.0, coarse[positive_idx[0]]["q5_deg"] - coarse_step_deg)
            hi = min(q5_max_deg, coarse[positive_idx[-1]]["q5_deg"] + coarse_step_deg)
        else:
            best_i = int(np.argmax(np.asarray(vals, dtype=np.float64)))
            lo = max(0.0, coarse[best_i]["q5_deg"] - coarse_step_deg)
            hi = min(q5_max_deg, coarse[best_i]["q5_deg"] + coarse_step_deg)
        refine_q.update(round(float(q), 6) for q in np.arange(lo, hi + 1.0e-9, fine_step_deg))
        for left, right, fl, fr in zip(coarse[:-1], coarse[1:], flags[:-1], flags[1:]):
            if fl != fr:
                refine_q.update(
                    round(float(q), 6)
                    for q in np.arange(left["q5_deg"], right["q5_deg"] + 1.0e-9, fine_step_deg)
                )

    for q in sorted(refine_q):
        evaluate(q)
    rows = [cache[q] for q in sorted(cache)]

    out: dict[str, Any] = {
        "chat": [float(v) for v in chat],
        "q5_sampling": {
            "coarse_step_deg": float(coarse_step_deg),
            "fine_step_deg": float(fine_step_deg),
            "n_coarse": len(coarse),
            "n_total": len(rows),
            "miss_bound": (
                "a positive island narrower than the coarse step and not containing the sampled global "
                "maximum could be missed; negative verdict is grid-bounded, not an analytic impossibility"
            ),
        },
        "curve": rows,
    }
    for name in ("unilateral", "bilateral"):
        best = max(rows, key=lambda row: _metric(row, name))
        out[f"{name}_positive_windows_deg"] = _positive_windows(rows, name, fine_step_deg)
        out[f"max_{name}_bite_mm"] = None if _metric(best, name) == -math.inf else _metric(best, name)
        out[f"q5_star_{name}_deg"] = float(best["q5_deg"])
        out[f"{name}_at_star"] = {
            key: best[key]
            for key in (
                "depth_top_min_mm",
                "delta_m",
                "bite_fixed_mm",
                "bite_moving_mm",
                "unilateral_bite_mm",
                "bilateral_bite_mm",
                "blocker",
            )
        }
    return out


def _plan_kwargs(margin_m: float, close_deg: float) -> dict[str, Any]:
    return {
        "object_size_m": np.array([OBJ_DIAM_M, OBJ_DIAM_M, OBJ_HEIGHT_M], dtype=np.float64),
        "grasp_surface_margin_m": float(margin_m),
        "approach_clearance_m": 0.040,
        "lift_delta_m": 0.025,
        "descend_open_deg": Q5_OPEN_DEG,
        "close_deg": [Q5_OPEN_DEG, float(close_deg)],
        "target_error_gate_m": TARGET_ERROR_GATE_M,
        "plan_tilt_gate_deg": TARGET_TILT_GATE_DEG,
    }


def _actual_frame(jaw: Any, q_descend_deg: np.ndarray) -> dict[str, Any]:
    T5 = jaw.fk_T_link5(np.asarray(q_descend_deg, dtype=np.float64))
    axis = T5[:3, 2].copy()
    axis /= float(np.linalg.norm(axis))
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    theta = math.degrees(math.acos(max(-1.0, min(1.0, float(axis @ down)))))
    chat = T5[:3, :3].T @ down
    chat /= float(np.linalg.norm(chat))
    phi = math.degrees(math.atan2(float(chat[1]), float(chat[0]))) % 360.0
    tcp = jaw.transform_pts(T5, np.asarray([jaw.TCP_LOCAL], dtype=np.float64))[0]
    return {
        "T5": T5,
        "axis_world": axis,
        "tcp_world": tcp,
        "chat": chat,
        "theta_actual_deg": float(theta),
        "phi_actual_deg": float(phi),
    }


def _world_collision_metrics(points: np.ndarray, center: np.ndarray) -> dict[str, Any]:
    radial = np.hypot(points[:, 0] - center[0], points[:, 1] - center[1])
    z = points[:, 2]
    inside = (
        (radial < OBJ_RADIUS_M - NUMERIC_PEN_EPS_M)
        & (z > SUPPORT_Z_M + NUMERIC_PEN_EPS_M)
        & (z < SUPPORT_Z_M + OBJ_HEIGHT_M - NUMERIC_PEN_EPS_M)
    )
    below = z < SUPPORT_Z_M - NUMERIC_PEN_EPS_M
    return {
        "finite_object_penetration_count": int(inside.sum()),
        "table_penetration_count": int(below.sum()),
        "min_table_clearance_mm": float((z.min() - SUPPORT_Z_M) * 1000.0),
    }


def _phase_wrist_r(plan: Any) -> tuple[list[float], list[bool]]:
    """Wrist-roll values/gates for the complete approach-descend-lift plan."""
    values = [
        float(plan.q_approach_deg[4]),
        float(plan.q_descend_deg[4]),
        float(plan.q_lift_deg[4]),
    ]
    checks = [
        WRIST_R_V6_DEG[0] - 1.0e-9 <= value <= WRIST_R_V6_DEG[1] + 1.0e-9
        for value in values
    ]
    return values, checks


def _candidate_specs(p10: Any, t3w: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    specs: list[dict[str, Any]] = []
    for label in ("seed0_S1", "seed0_S2", "seed0_S3", "seed0_S4"):
        x, y = p10._workspace_xy_from_label(label)
        specs.append({"pose_key": label, "xy": [float(x), float(y)], "negative_control": False})
    psi = math.radians(float(t3w["grid"]["psi_pos_deg"]))
    for radius, key, negative in ((0.45, "r045_dpsi0", False), (0.525, "r0525_negative", True)):
        specs.append(
            {
                "pose_key": key,
                "xy": [radius * math.cos(psi), radius * math.sin(psi)],
                "negative_control": negative,
            }
        )
    source_max_r = max(math.hypot(*spec["xy"]) for spec in specs[:4])
    return specs, float(source_max_r)


def _evaluate_candidates(
    args: argparse.Namespace,
    p10: Any,
    jaw: Any,
    n8: Any,
    fixed_pts: np.ndarray,
    moving_fn: Any,
    t3w: dict[str, Any],
    theta_values: list[float],
    q5_max_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]], float]:
    specs, source_max_r = _candidate_specs(p10, t3w)
    candidates: list[dict[str, Any]] = []
    views: dict[str, dict[str, np.ndarray]] = {}
    original_limits = dict(p10.V6_LIMITS_DEG)

    for spec in specs:
        x, y = spec["xy"]
        center = np.array([x, y, SUPPORT_Z_M + OBJ_HEIGHT_M / 2.0], dtype=np.float64)
        psi_axis = math.degrees(math.atan2(y, x)) % 360.0
        plan_rows: list[dict[str, Any]] = []
        for theta in theta_values:
            p10.V6_LIMITS_DEG = dict(original_limits)
            p10.set_target_axis(theta, psi_axis)
            p10.PHI_STAR_DEG = 0.0
            try:
                plan = p10._build_plan_from_center(
                    _PlanArgs(**_plan_kwargs(DEFAULT_MARGIN_M, 24.0)), center, spec["pose_key"]
                )
                phase_ok = bool(plan.approach_ik_ok and plan.descend_ik_ok and plan.lift_ik_ok)
                phase_wrist, phase_wrist_ok = _phase_wrist_r(plan)
                wrist_ok = all(phase_wrist_ok)
                plan_rows.append(
                    {
                        "theta_target_deg": float(theta),
                        "phase_ik_ok": phase_ok,
                        "wrist_r_ok": bool(wrist_ok),
                        "q_descend_deg": [float(v) for v in plan.q_descend_deg],
                        "phase_position_error_mm": [
                            float(plan.approach_ik_err_mm),
                            float(plan.descend_ik_err_mm),
                            float(plan.lift_ik_err_mm),
                        ],
                        "phase_axis_residual_deg": [
                            float(plan.approach_tilt_deg),
                            float(plan.descend_tilt_deg),
                            float(plan.lift_tilt_deg),
                        ],
                        "phase_wrist_r_deg": phase_wrist,
                        "phase_wrist_r_ok": phase_wrist_ok,
                        "wrist_r_fail_phases": [
                            phase
                            for phase, passed in zip(("approach", "descend", "lift"), phase_wrist_ok)
                            if not passed
                        ],
                        "_plan": plan,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                plan_rows.append(
                    {
                        "theta_target_deg": float(theta),
                        "phase_ik_ok": False,
                        "wrist_r_ok": False,
                        "phase_wrist_r_deg": None,
                        "phase_wrist_r_ok": [False, False, False],
                        "wrist_r_fail_phases": ["approach", "descend", "lift"],
                        "error": repr(exc),
                    }
                )
        feasible = [row for row in plan_rows if row["phase_ik_ok"] and row["wrist_r_ok"]]
        selected = max(feasible, key=lambda row: row["theta_target_deg"]) if feasible else None
        row_out: dict[str, Any] = {
            "candidate_id": f"{spec['pose_key']}_best",
            "pose_key": spec["pose_key"],
            "center_m": [float(v) for v in center],
            "r_m": float(math.hypot(x, y)),
            "psi_pos_deg": float(psi_axis),
            "psi_axis_deg": float(psi_axis),
            "negative_control": bool(spec["negative_control"]),
            "source_envelope_max_r_m": source_max_r,
            "n_theta_evaluated": len(plan_rows),
            "n_phase_and_wrist_feasible": len(feasible),
            "theta_plan_audit": [
                {key: value for key, value in plan_row.items() if key != "_plan"}
                for plan_row in plan_rows
            ],
            "theta_target_deg": None,
            "theta_actual_deg": None,
            "phi_actual_deg": None,
            "q_descend_deg": None,
            "q5_window_deg": None,
            "q5_close_targets_deg": [],
            "grasp_surface_margin_m": None,
            "window_kind": "no_window",
            "finite_object_penetration_count": None,
            "min_table_clearance_mm": None,
            "eligible_for_physics": False,
            "exclusion_reasons": [],
        }
        if selected is None:
            row_out["exclusion_reasons"].append("no_phase_and_wrist_feasible_theta")
            candidates.append(row_out)
            continue

        plan = selected.pop("_plan")
        frame = _actual_frame(jaw, plan.q_descend_deg)
        scan = _scan_orientation(
            n8,
            fixed_pts,
            moving_fn,
            frame["chat"],
            q5_max_deg,
            args.q5_coarse_step_deg,
            args.q5_fine_step_deg,
        )
        bilateral_windows = scan["bilateral_positive_windows_deg"]
        unilateral_windows = scan["unilateral_positive_windows_deg"]
        if bilateral_windows:
            window_kind = "bilateral"
            selected_window = [
                bilateral_windows[0]["q5_lo_deg"], bilateral_windows[0]["q5_hi_deg"]
            ]
        elif unilateral_windows:
            window_kind = "unilateral_negative_control"
            selected_window = [
                unilateral_windows[0]["q5_lo_deg"], unilateral_windows[0]["q5_hi_deg"]
            ]
        else:
            window_kind = "no_window"
            selected_window = None
        targets = _window_targets(selected_window)
        q_selected = targets[1] if targets else 24.0
        curve_row = min(scan["curve"], key=lambda item: abs(item["q5_deg"] - q_selected))
        margin = curve_row["delta_m"] if curve_row["delta_m"] is not None else DEFAULT_MARGIN_M

        p10.set_target_axis(selected["theta_target_deg"], psi_axis)
        p10.PHI_STAR_DEG = 0.0
        final_plan = p10._build_plan_from_center(
            _PlanArgs(**_plan_kwargs(float(margin), q_selected)), center, spec["pose_key"] + "_measured"
        )
        final_phase_ok = bool(
            final_plan.approach_ik_ok and final_plan.descend_ik_ok and final_plan.lift_ik_ok
        )
        final_phase_wrist, final_phase_wrist_ok = _phase_wrist_r(final_plan)
        final_wrist_ok = all(final_phase_wrist_ok)
        final_frame = _actual_frame(jaw, final_plan.q_descend_deg)
        fixed_world = jaw.transform_pts(final_frame["T5"], fixed_pts)
        moving_local = moving_fn(q_selected)
        moving_world = jaw.transform_pts(final_frame["T5"], moving_local)
        fixed_world_metrics = _world_collision_metrics(fixed_world, center)
        moving_world_metrics = _world_collision_metrics(moving_world, center)
        penetration_count = (
            fixed_world_metrics["finite_object_penetration_count"]
            + moving_world_metrics["finite_object_penetration_count"]
        )
        table_count = (
            fixed_world_metrics["table_penetration_count"]
            + moving_world_metrics["table_penetration_count"]
        )
        min_table = min(
            fixed_world_metrics["min_table_clearance_mm"],
            moving_world_metrics["min_table_clearance_mm"],
        )

        exclusion: list[str] = []
        if spec["negative_control"] or row_out["r_m"] > source_max_r + 1.0e-9:
            exclusion.append("outside_source_envelope_or_declared_negative_control")
        if window_kind != "bilateral":
            exclusion.append("no_bilateral_window")
        if not final_phase_ok:
            exclusion.append("final_margin_phase_ik_fail")
        if not final_wrist_ok:
            failed = [
                phase
                for phase, passed in zip(("approach", "descend", "lift"), final_phase_wrist_ok)
                if not passed
            ]
            exclusion.append("final_margin_wrist_r_outside_v6:" + ",".join(failed))
        if penetration_count != 0:
            exclusion.append("finite_object_penetration")
        if table_count != 0 or min_table < -NUMERIC_PEN_EPS_M * 1000.0:
            exclusion.append("support_plane_penetration")

        row_out.update(
            {
                "theta_target_deg": float(selected["theta_target_deg"]),
                "theta_actual_deg": float(final_frame["theta_actual_deg"]),
                "phi_actual_deg": float(final_frame["phi_actual_deg"]),
                "q_descend_deg": [float(v) for v in final_plan.q_descend_deg],
                "phase_position_error_mm": [
                    float(final_plan.approach_ik_err_mm),
                    float(final_plan.descend_ik_err_mm),
                    float(final_plan.lift_ik_err_mm),
                ],
                "phase_axis_residual_deg": [
                    float(final_plan.approach_tilt_deg),
                    float(final_plan.descend_tilt_deg),
                    float(final_plan.lift_tilt_deg),
                ],
                "phase_ik_ok": final_phase_ok,
                "selected_default_margin_phase_wrist_r_deg": selected["phase_wrist_r_deg"],
                "selected_default_margin_phase_wrist_r_ok": selected["phase_wrist_r_ok"],
                "phase_wrist_r_deg": final_phase_wrist,
                "phase_wrist_r_ok": final_phase_wrist_ok,
                "wrist_r_fail_phases": [
                    phase
                    for phase, passed in zip(("approach", "descend", "lift"), final_phase_wrist_ok)
                    if not passed
                ],
                "wrist_r_ok": bool(final_wrist_ok),
                "q5_window_deg": selected_window,
                "q5_close_targets_deg": targets,
                "q5_selected_for_world_check_deg": float(q_selected),
                "grasp_surface_margin_m": float(margin),
                "window_kind": window_kind,
                "actual_collision": {k: v for k, v in scan.items() if k != "curve"},
                "finite_object_penetration_count": int(penetration_count),
                "table_penetration_count": int(table_count),
                "min_table_clearance_mm": float(min_table),
                "eligible_for_physics": not exclusion,
                "exclusion_reasons": exclusion,
            }
        )
        candidates.append(row_out)
        views[row_out["candidate_id"]] = {
            "fixed_world": fixed_world[::32].astype(np.float32),
            "moving_world": moving_world[::16].astype(np.float32),
            "center": center.astype(np.float32),
            "tcp": final_frame["tcp_world"].astype(np.float32),
            "axis": final_frame["axis_world"].astype(np.float32),
        }

    p10.V6_LIMITS_DEG = original_limits
    return candidates, views, source_max_r


def _control_handoff(per_theta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_theta = {row["theta_deg"]: row for row in per_theta}
    controls: list[dict[str, Any]] = []
    for theta in PHYSICS_CONTROL_THETA:
        row = by_theta[theta]
        collision = row["collision"]
        bilateral = collision["bilateral_positive_windows_deg"]
        unilateral = collision["unilateral_positive_windows_deg"]
        if bilateral:
            kind, raw = "bilateral", bilateral[0]
        elif unilateral:
            kind, raw = "unilateral_negative_control", unilateral[0]
        else:
            kind, raw = "no_window", None
        window = None if raw is None else [raw["q5_lo_deg"], raw["q5_hi_deg"]]
        # PhysX still runs when static admission finds no window.  These three
        # broad q5 controls keep that negative-control stratum executable while
        # `window_kind=no_window` prevents them being mistaken for measured bite.
        targets = _window_targets(window) if window is not None else [16.0, 24.0, 32.0]
        mid = targets[1] if targets else 24.0
        curve = collision["curve"]
        at_mid = min(curve, key=lambda item: abs(item["q5_deg"] - mid))
        measured_margin = at_mid["delta_m"]
        margin = float(measured_margin) if measured_margin is not None else DEFAULT_MARGIN_M
        controls.append(
            {
                "theta_deg": float(theta),
                "phi_deg": 0.0,
                "window_kind": kind,
                "q5_window_deg": window,
                "q5_close_targets_deg": targets,
                "grasp_surface_margin_m": margin,
                "margin_source": (
                    "measured_delta" if measured_margin is not None
                    else "default_no_finite_intersection"
                ),
                "q_descend_deg": None,
                "requires_pose_specific_ik": True,
            }
        )
    return controls


def _cylinder_lines(center: np.ndarray) -> list[np.ndarray]:
    ang = np.linspace(0.0, 2.0 * math.pi, 65)
    rings = []
    for z in (SUPPORT_Z_M, SUPPORT_Z_M + OBJ_HEIGHT_M):
        rings.append(
            np.column_stack(
                [center[0] + OBJ_RADIUS_M * np.cos(ang),
                 center[1] + OBJ_RADIUS_M * np.sin(ang),
                 np.full_like(ang, z)]
            ).astype(np.float32)
        )
    for a in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
        rings.append(
            np.asarray(
                [[center[0] + OBJ_RADIUS_M * math.cos(a), center[1] + OBJ_RADIUS_M * math.sin(a), SUPPORT_Z_M],
                 [center[0] + OBJ_RADIUS_M * math.cos(a), center[1] + OBJ_RADIUS_M * math.sin(a), SUPPORT_Z_M + OBJ_HEIGHT_M]],
                dtype=np.float32,
            )
        )
    return rings


def _emit_rerun(
    paths: dict[str, Path], out: dict[str, Any], views: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    import rerun as rr
    import rerun.blueprint as rrb

    from roarm_rl.rerun_contract import validate_rerun_artifact

    if rr.__version__ != RERUN_VERSION:
        raise RuntimeError(f"rerun pin mismatch {rr.__version__} != {RERUN_VERSION}")
    entity_names = [
        "metadata/run",
        "events/gates",
        "events/verdict",
        "world/support_plane",
        "world/object",
        "world/link5_collision",
        "world/gripper_collision",
        "world/tcp",
        "world/tool_axis",
        "plots/theta_actual_deg",
        "plots/bilateral_window",
        "plots/unilateral_window",
        "plots/table_clearance_mm",
    ]
    components = {
        "metadata/run": ["TextDocument:text"],
        "events/gates": ["TextLog:text", "TextLog:level"],
        "events/verdict": ["TextLog:text", "TextLog:level"],
        "world/support_plane": ["LineStrips3D:strips"],
        "world/object": ["LineStrips3D:strips"],
        "world/link5_collision": ["Points3D:positions"],
        "world/gripper_collision": ["Points3D:positions"],
        "world/tcp": ["Points3D:positions"],
        "world/tool_axis": ["Arrows3D:origins", "Arrows3D:vectors"],
        "plots/theta_actual_deg": ["Scalars:scalars"],
        "plots/bilateral_window": ["Scalars:scalars"],
        "plots/unilateral_window": ["Scalars:scalars"],
        "plots/table_clearance_mm": ["Scalars:scalars"],
    }
    reported_verdict = out.get("verdict", out["scientific_verdict"])
    summary = (
        f"# p13 / t3x IK-conditioned bite window\n\n"
        f"**reported verdict**: `{reported_verdict}`  \n"
        f"**scientific result before validity gates**: `{out['scientific_verdict']}`  \n"
        f"collision asset: frozen attempt3, 64+64 convexHull parts  \n"
        f"candidate count: {len(out['pose_candidates'])}; eligible bilateral: "
        f"{sum(1 for row in out['pose_candidates'] if row['eligible_for_physics'])}  \n\n"
        "Float64 JSON/NPZ is authoritative. Rerun spatial values are Float32 inspection copies."
    )
    plane = np.asarray(
        [[[-0.05, -0.60, SUPPORT_Z_M], [0.60, -0.60, SUPPORT_Z_M],
          [0.60, 0.60, SUPPORT_Z_M], [-0.05, 0.60, SUPPORT_Z_M], [-0.05, -0.60, SUPPORT_Z_M]]],
        dtype=np.float32,
    )
    app_id = "roarm_g0b_t3x_bite81"
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/world", contents="/world/**", name="1 | actual FK + convex jaws + finite cylinder"),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run", name="2 | run summary"),
                    rrb.TextLogView(origin="/events", contents="/events/**", name="3 | gates and verdict"),
                ),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/plots", contents=["/plots/theta_actual_deg/**", "/plots/table_clearance_mm/**"], name="4 | actual theta + table clearance"),
                rrb.TimeSeriesView(origin="/plots", contents=["/plots/bilateral_window/**", "/plots/unilateral_window/**"], name="5 | window present [0/1]"),
            ),
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )
    by_id = {row["candidate_id"]: row for row in out["pose_candidates"]}
    with rr.RecordingStream(app_id, recording_id="g0b_d420_t3x_bite81", make_default=False, send_properties=True) as rec:
        rec.save(str(paths["timeline.rrd"]), write_footer=True)
        rec.log("metadata/run", rr.TextDocument(summary, media_type=rr.MediaType.MARKDOWN), static=True)
        for index, (candidate_id, view) in enumerate(views.items()):
            row = by_id[candidate_id]
            rec.reset_time()
            rec.set_time("candidate_index", sequence=index)
            rec.log("events/gates", rr.TextLog(
                f"candidate={candidate_id} phase_ik={row.get('phase_ik_ok')} "
                f"phase_wrist_r_deg={row.get('phase_wrist_r_deg')} "
                f"phase_wrist_r_ok={row.get('phase_wrist_r_ok')} "
                f"wrist_fail_phases={row.get('wrist_r_fail_phases')} "
                f"object_pen={row['finite_object_penetration_count']} table_mm={row['min_table_clearance_mm']}",
                level=rr.TextLogLevel.INFO if row["eligible_for_physics"] else rr.TextLogLevel.WARN,
            ))
            rec.log("events/verdict", rr.TextLog(
                f"window_kind={row['window_kind']} eligible={row['eligible_for_physics']} reasons={row['exclusion_reasons']}",
                level=rr.TextLogLevel.INFO if row["eligible_for_physics"] else rr.TextLogLevel.WARN,
            ))
            rec.log("world/support_plane", rr.LineStrips3D(plane, colors=[110, 110, 110], radii=0.001))
            rec.log("world/object", rr.LineStrips3D(_cylinder_lines(view["center"]), colors=[210, 145, 70], radii=0.0015))
            rec.log("world/link5_collision", rr.Points3D(view["fixed_world"], colors=[70, 140, 245], radii=0.0006))
            rec.log("world/gripper_collision", rr.Points3D(view["moving_world"], colors=[230, 70, 110], radii=0.0006))
            rec.log("world/tcp", rr.Points3D(np.asarray([view["tcp"]]), colors=[255, 230, 40], radii=0.004))
            rec.log("world/tool_axis", rr.Arrows3D(
                origins=np.asarray([view["tcp"]]), vectors=np.asarray([0.06 * view["axis"]]),
                colors=np.asarray([[255, 230, 40]], dtype=np.uint8), radii=np.asarray([0.002], dtype=np.float32),
            ))
            rec.log("plots/theta_actual_deg", rr.Scalars(float(row["theta_actual_deg"])))
            rec.log("plots/bilateral_window", rr.Scalars(float(row["window_kind"] == "bilateral")))
            rec.log("plots/unilateral_window", rr.Scalars(float(row["window_kind"] != "no_window")))
            rec.log("plots/table_clearance_mm", rr.Scalars(float(row["min_table_clearance_mm"])))
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=180.0)
    blueprint.save(app_id, str(paths["timeline.rbl"]))

    validation = validate_rerun_artifact(
        paths["timeline.rrd"],
        expected_entity_paths=entity_names,
        exact_entity_paths=entity_names,
        exact_timeline_names=["blueprint", "candidate_index", "log_time"],
        expected_entity_components=components,
        blueprint_path=paths["timeline.rbl"],
        screenshot_path=paths["inspection.png"],
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=600.0,
    )
    paths["rerun_validation.json"].write_text(json.dumps(validation, indent=2, default=str) + "\n")
    return {"pass": bool(validation.get("pass")), "errors": validation.get("errors", [])}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_label", default="bite81", choices=["bite81"])
    parser.add_argument("--theta_hi_deg", type=float, default=81.0)
    parser.add_argument("--theta_step_deg", type=float, default=1.0)
    parser.add_argument("--q5_coarse_step_deg", type=float, default=0.5)
    parser.add_argument("--q5_fine_step_deg", type=float, default=0.1)
    return parser


def main() -> int:
    started = time.time()
    source_path = Path(__file__).resolve()
    source_start_bytes = source_path.read_bytes()
    source_start_sha = hashlib.sha256(source_start_bytes).hexdigest()
    args = build_argparser().parse_args()
    prefix = f"t3x_{args.run_label}"
    paths = {name: CASE_DIR / f"{prefix}_{name}" for name in (
        "results.json", "grid.npz", "curves.csv", "timeline.rrd", "timeline.rbl",
        "rerun_validation.json", "inspection.png", "script.py.txt", "argv.txt",
    )}
    existing = [path.name for path in paths.values() if path.exists()]
    if existing:
        print(f"[{LOG}] G0B_T3X_BITE_VERDICT=ARTIFACT_EXISTS_ABORT existing={existing}", flush=True)
        return 3
    if not PREREG.exists():
        print(f"[{LOG}] ABORT missing_prereg={PREREG}", flush=True)
        return 3

    observed_sha: dict[str, str] = {}
    sha_checks: dict[str, bool] = {}
    for path, expected in EXPECTED_SHA256.items():
        observed = sha256_file(path) if path.exists() else "MISSING"
        observed_sha[str(path.relative_to(REPO))] = observed
        sha_checks[str(path.relative_to(REPO))] = observed == expected
    pin_checks = {
        "numpy": np.__version__ == NUMPY_PIN,
        "psutil": psutil.__version__ == PSUTIL_PIN,
    }
    if not all(sha_checks.values()) or not all(pin_checks.values()):
        print(f"[{LOG}] G0B_T3X_BITE_VERDICT=INPUT_GATE_FAIL sha={sha_checks} pins={pin_checks}", flush=True)
        return 3

    n8 = load_module("p13_n8_core", N8_PATH)
    jaw = load_module("p13_jaw_core", JAW_PATH)
    p10 = load_module("p13_p10_core", P10_PATH)
    n10 = json.loads(N10_RESULTS.read_text())
    t3w = json.loads(T3W_RESULTS.read_text())

    asset_sha = {
        str(jaw.ATTEMPT3_USD.relative_to(REPO)): sha256_file(jaw.ATTEMPT3_USD),
        str(jaw.ATTEMPT3_PHYSICS_LAYER.relative_to(REPO)): sha256_file(jaw.ATTEMPT3_PHYSICS_LAYER),
    }
    asset_sha_ok = bool(
        asset_sha[str(jaw.ATTEMPT3_USD.relative_to(REPO))] == jaw.ATTEMPT3_ROOT_SHA256
        and asset_sha[str(jaw.ATTEMPT3_PHYSICS_LAYER.relative_to(REPO))] == jaw.ATTEMPT3_PHYSICS_SHA256
    )
    asset = jaw.extract_asset()
    bodies = asset["bodies"]
    counts = {name: len(bodies[name]["parts"]) for name in ("link5", "gripper_link")}
    legacy_disabled = all(
        not enabled for name in ("link5", "gripper_link") for _path, enabled in bodies[name]["legacy"]
    )
    no_bad_approx = not any(bodies[name]["approx_bad"] for name in ("link5", "gripper_link"))
    asset_identity_ok = bool(
        asset_sha_ok
        and counts == {"link5": 64, "gripper_link": 64}
        and legacy_disabled
        and no_bad_approx
    )
    if not asset_identity_ok:
        print(f"[{LOG}] G0B_T3X_BITE_VERDICT=ASSET_IDENTITY_FAIL counts={counts}", flush=True)
        return 3

    fixed_pts = np.vstack([part["samples"] for part in bodies["link5"]["parts"]])
    moving_base = np.vstack([part["samples"] for part in bodies["gripper_link"]["parts"]])
    joint = asset["joint"]

    def moving_fn(q5_deg: float) -> np.ndarray:
        transform = jaw.gripper_T_l5(joint, float(q5_deg))
        return jaw.transform_pts(transform, moving_base)

    q5_max_deg = float(n10["q5_grids"]["q5_max_deg"])
    extension = np.arange(36.0, args.theta_hi_deg + 1.0e-9, args.theta_step_deg)
    theta_values = sorted(
        {*CONTROL_THETA, *PHYSICS_CONTROL_THETA, *(round(float(v), 6) for v in extension)}
    )
    per_theta: list[dict[str, Any]] = []
    for index, theta in enumerate(theta_values):
        chat = n8.axis_dir(math.radians(theta), 0.0)
        collision = _scan_orientation(
            n8, fixed_pts, moving_fn, chat, q5_max_deg,
            args.q5_coarse_step_deg, args.q5_fine_step_deg,
        )
        per_theta.append({"theta_deg": float(theta), "phi_deg": 0.0, "collision": collision})
        print(
            f"[{LOG}] geometry {index + 1}/{len(theta_values)} theta={theta:.1f} "
            f"uni={collision['max_unilateral_bite_mm']} bi={collision['max_bilateral_bite_mm']} "
            f"uni_win={collision['unilateral_positive_windows_deg']} "
            f"bi_win={collision['bilateral_positive_windows_deg']}",
            flush=True,
        )

    by_theta = {row["theta_deg"]: row for row in per_theta}
    n10_by_theta = {float(row["theta_deg"]): row for row in n10["per_theta"]}
    regression_rows: list[dict[str, Any]] = []
    regression_ok = True
    for theta in REGRESSION_THETA:
        got = by_theta[theta]["collision"]
        ref = n10_by_theta[theta]["collision"]
        got_star = got["unilateral_at_star"]
        checks = {
            "max_unilateral_bite_mm": abs(got["max_unilateral_bite_mm"] - ref["max_bite_mm"]) < 1.0e-6,
            "q5_star_unilateral_deg": abs(got["q5_star_unilateral_deg"] - ref["q5_star_deg"]) < 1.0e-9,
            "bite_fixed_mm_at_star": abs(got_star["bite_fixed_mm"] - ref["bite_fixed_mm_at_star"]) < 1.0e-6,
            "bite_moving_mm_at_star": abs(got_star["bite_moving_mm"] - ref["bite_moving_mm_at_star"]) < 1.0e-6,
            "unilateral_window": got["unilateral_positive_windows_deg"] == [
                {"q5_lo_deg": w["q5_lo_deg"], "q5_hi_deg": w["q5_hi_deg"], "edge_uncertainty_deg": 0.1}
                for w in ref["positive_windows_deg"]
            ],
        }
        passed = all(checks.values())
        regression_ok = regression_ok and passed
        regression_rows.append({"theta_deg": theta, "pass": passed, "checks": checks})
    if not regression_ok:
        print(f"[{LOG}] G0B_T3X_BITE_VERDICT=N10_REGRESSION_FAIL rows={regression_rows}", flush=True)
        return 3

    candidates, views, source_max_r = _evaluate_candidates(
        args, p10, jaw, n8, fixed_pts, moving_fn, t3w, theta_values, q5_max_deg
    )
    eligible = [row for row in candidates if row["eligible_for_physics"]]
    scientific_verdict = (
        "BILATERAL_WINDOW_EXISTS_IN_SPAWN_ENVELOPE"
        if eligible
        else "NO_BILATERAL_WINDOW_IN_SPAWN_ENVELOPE"
    )

    flat_rows = [
        {
            "theta_deg": row["theta_deg"],
            "phi_deg": row["phi_deg"],
            **curve,
        }
        for row in per_theta
        for curve in row["collision"]["curve"]
    ]
    with paths["curves.csv"].open("w", newline="") as f:
        fields = list(flat_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    np.savez_compressed(
        paths["grid.npz"],
        theta_deg=np.asarray([row["theta_deg"] for row in flat_rows], dtype=np.float64),
        phi_deg=np.asarray([row["phi_deg"] for row in flat_rows], dtype=np.float64),
        q5_deg=np.asarray([row["q5_deg"] for row in flat_rows], dtype=np.float64),
        depth_top_min_mm=np.asarray([np.nan if row["depth_top_min_mm"] is None else row["depth_top_min_mm"] for row in flat_rows], dtype=np.float64),
        bite_fixed_mm=np.asarray([np.nan if row["bite_fixed_mm"] is None else row["bite_fixed_mm"] for row in flat_rows], dtype=np.float64),
        bite_moving_mm=np.asarray([np.nan if row["bite_moving_mm"] is None else row["bite_moving_mm"] for row in flat_rows], dtype=np.float64),
        unilateral_bite_mm=np.asarray([np.nan if row["unilateral_bite_mm"] is None else row["unilateral_bite_mm"] for row in flat_rows], dtype=np.float64),
        bilateral_bite_mm=np.asarray([np.nan if row["bilateral_bite_mm"] is None else row["bilateral_bite_mm"] for row in flat_rows], dtype=np.float64),
    )
    paths["argv.txt"].write_text(" ".join(sys.argv) + "\n")

    out: dict[str, Any] = {
        "schema_version": 1,
        "tool": "p13_g0b_t3x_cyld29h50_ik_conditioned_bite_window_audit",
        "tag": TAG,
        "case": "g0b_d420",
        "prereg": str(PREREG.relative_to(REPO)),
        "argv": list(sys.argv),
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "psutil": psutil.__version__,
            "rerun_sdk_expected": RERUN_VERSION,
            "isaac_launched": False,
            "physics_used": False,
            "robot_used": False,
        },
        "input_sha256": {**observed_sha, **asset_sha},
        "asset_identity": {
            "pass": asset_identity_ok,
            "enabled_convexhull_parts": counts,
            "legacy_all_disabled": legacy_disabled,
            "non_convexhull_approximation": {
                name: bodies[name]["approx_bad"] for name in ("link5", "gripper_link")
            },
            "hull_sample_spacing_m": float(jaw.SAMPLE_SPACING_M),
            "collision_asset_decisive": True,
        },
        "object": {
            "shape": "finite_cylinder",
            "diameter_m": OBJ_DIAM_M,
            "height_m": OBJ_HEIGHT_M,
            "support_z_m": SUPPORT_Z_M,
            "grasp_point": "top_center_D419_unchanged",
        },
        "sampling": {
            "theta_deg": theta_values,
            "phi_deg": 0.0,
            "q5_coarse_step_deg": args.q5_coarse_step_deg,
            "q5_fine_step_deg": args.q5_fine_step_deg,
        },
        "gates": {
            "X1_input_sha": all(sha_checks.values()),
            "X1_env_pins": all(pin_checks.values()),
            "X2_asset_identity_64_plus_64": asset_identity_ok,
            "X3_n10_theta29_35_regression": regression_ok,
            "X4_pose_phase_and_wrist_recorded": all(
                isinstance(row.get("theta_plan_audit"), list)
                and row["theta_plan_audit"]
                and all(
                    "phase_ik_ok" in plan_row
                    and "phase_wrist_r_deg" in plan_row
                    and "phase_wrist_r_ok" in plan_row
                    and "wrist_r_fail_phases" in plan_row
                    for plan_row in row["theta_plan_audit"]
                )
                and (
                    row["theta_target_deg"] is None
                    or all(
                        key in row
                        for key in (
                            "phase_ik_ok", "phase_wrist_r_deg", "phase_wrist_r_ok",
                            "wrist_r_fail_phases", "wrist_r_ok",
                        )
                    )
                )
                for row in candidates
            ),
            "X5_finite_object_and_table_recorded": all(row["finite_object_penetration_count"] is not None for row in candidates if row["theta_target_deg"] is not None),
        },
        "n10_regression": {"pass": regression_ok, "rows": regression_rows},
        "per_theta": per_theta,
        "source_envelope_max_r_m": source_max_r,
        "pose_candidates": candidates,
        "physics_handoff": {
            "schema_version": 1,
            "source_tag": TAG,
            "run_physx_regardless_of_bilateral": True,
            "window_kind_enum": ["bilateral", "unilateral_negative_control", "no_window"],
            "controls": _control_handoff(per_theta),
            "candidates": candidates,
            "warning": "every row still requires the consuming physics script's own pose-specific IK gate",
        },
        "scientific_verdict": scientific_verdict,
        "claims_not_made": [
            "force_closure", "lift_success", "side_midpoint_grasp", "g0a_pass_true", "Arm-F_authoring",
        ],
        "g0a_pass": False,
    }

    source_pre_rerun_bytes = source_path.read_bytes()
    source_pre_rerun_sha = hashlib.sha256(source_pre_rerun_bytes).hexdigest()
    stable_before_rerun = source_pre_rerun_bytes == source_start_bytes
    out["source_freeze"] = {
        "path": str(source_path.relative_to(REPO)),
        "start_sha256": source_start_sha,
        "start_bytes": len(source_start_bytes),
        "pre_rerun_sha256": source_pre_rerun_sha,
        "stable_before_rerun": stable_before_rerun,
        "end_sha256": None,
        "stable_at_end": None,
        "frozen_copy_uses_start_bytes": True,
    }
    out["gates"]["X6_source_stable"] = stable_before_rerun
    out["run_valid"] = stable_before_rerun
    out["verdict"] = scientific_verdict if stable_before_rerun else "SOURCE_DRIFT_INVALID"

    rerun = _emit_rerun(paths, out, views)
    out["rerun"] = rerun
    source_end_bytes = source_path.read_bytes()
    source_end_sha = hashlib.sha256(source_end_bytes).hexdigest()
    source_stable = stable_before_rerun and source_end_bytes == source_start_bytes
    out["source_freeze"]["end_sha256"] = source_end_sha
    out["source_freeze"]["stable_at_end"] = source_stable
    out["gates"]["X6_source_stable"] = source_stable
    out["run_valid"] = source_stable
    out["verdict"] = scientific_verdict if source_stable else "SOURCE_DRIFT_INVALID"
    paths["script.py.txt"].write_bytes(source_start_bytes)
    out["artifacts"] = {
        name: {"name": path.name, "sha256_16": sha256_file(path)[:16], "bytes": path.stat().st_size}
        for name, path in paths.items()
        if name != "results.json" and path.exists()
    }
    out["wall_seconds"] = round(time.time() - started, 1)
    paths["results.json"].write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(
        f"[{LOG}] G0B_T3X_BITE_VERDICT={out['verdict']} eligible={len(eligible)} "
        f"rerun_pass={rerun['pass']} results_sha={sha256_file(paths['results.json'])[:16]}",
        flush=True,
    )
    return 0 if rerun["pass"] and source_stable else 3


if __name__ == "__main__":
    raise SystemExit(main())
