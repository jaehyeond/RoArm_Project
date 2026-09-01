#!/usr/bin/env python3
"""Summarise / plot the PBD probe cells written by `sim_pbd_pellet_probe.py`.

Read-only over `claudedocs/runtime_logs/pbd_probe/cell_*.json|npz`.  Kept in a
separate file so the probe module never needs matplotlib inside the Kit process.

Usage:
  python sim_pbd_pellet_report.py --table
  python sim_pbd_pellet_report.py --diagnose <label>      # depth vs analytic PNG
  python sim_pbd_pellet_report.py --gates                 # G1..G5 verdicts
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import sim_pellet_model as p3

OUT_DIR = Path("claudedocs/runtime_logs/pbd_probe")
DEME_SPHERE_RANGE_DEG = (8.90, 25.64)
DEME_CLUMP2_RANGE_DEG = (22.45, 37.56)
DEME_CLUMP3_RANGE_DEG = (20.03, 41.77)
TARGET_DEG = 30.0
TARGET_TOL_DEG = 3.0
G1_MIN_REACHABLE_DEG = 27.0


def _backfill_conical_check(d: dict[str, Any]) -> None:
    """Apply `heap_is_conical` to cells written before that check existed.

    The check (plateau radius <= 0.5 * toe radius) rejects a flat cake with a
    steep rim, whose 80%-20% band lies entirely on the rim and returns a steep,
    well-fitted, meaningless angle.  It is computed here from fields the older
    cells already store, so no run has to be repeated and no earlier number is
    silently rewritten - only `measurement_pass` tightens.
    """
    for key in ("measurement_depth", "measurement_analytic_readback"):
        m = d.get(key)
        if not isinstance(m, dict) or "checks" not in m:
            continue
        if "heap_is_conical" in m["checks"]:
            continue
        toe = m.get("toe_radius_m", float("nan"))
        plateau = m.get("plateau_radius_m", float("nan"))
        ratio = plateau / toe if isinstance(toe, (int, float)) and toe else float("nan")
        m["plateau_over_toe_ratio"] = ratio
        m["checks"]["heap_is_conical"] = bool(math.isfinite(ratio) and ratio <= 0.5)
        m["measurement_pass"] = bool(all(m["checks"].values()))
        m["conical_check_backfilled"] = True


def load_cells(out_dir: Path = OUT_DIR) -> dict[str, dict[str, Any]]:
    cells = {}
    for path in sorted(out_dir.glob("cell_*.json")):
        d = json.loads(path.read_text())
        _backfill_conical_check(d)
        cells[d.get("label", path.stem)] = d
    return cells


def _g(d: dict, *keys, default=float("nan")):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def table(cells: dict[str, dict[str, Any]]) -> None:
    hdr = (
        f"{'label':28s} {'fric':>5s} {'pfs':>4s} {'damp':>5s} {'adh':>5s} {'it':>3s} {'Hz':>4s} "
        f"{'N':>6s} {'ang_dep':>8s} {'ang_ana':>8s} {'r2':>6s} {'pass':>5s} {'settl':>6s} "
        f"{'apex_mm':>8s} {'toe_mm':>7s} {'pl/toe':>7s} {'cone':>5s} {'creep_deg':>9s} {'rms_mm_s':>9s} {'wall_s':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, d in cells.items():
        if d.get("failed"):
            print(f"{label:28s} CELL_FAILED {d.get('error','')[:80]}")
            continue
        c = d["config"]
        hist = _g(d, "settle", "history", default=[])
        rms = hist[-1]["speed_rms_m_s"] * 1e3 if hist else float("nan")
        print(
            f"{label:28s} {c['friction']:5.2f} {c['particle_friction_scale']:4.1f} "
            f"{c['damping']:5.2f} {c['adhesion']:5.2f} {c['solver_position_iterations']:3d} "
            f"{c['time_steps_per_second']:4d} {c['n_particles']:6d} "
            f"{_g(d,'measurement_depth','repose_angle_deg'):8.3f} "
            f"{_g(d,'measurement_analytic_readback','repose_angle_deg'):8.3f} "
            f"{_g(d,'measurement_depth','fit_r_squared'):6.3f} "
            f"{str(_g(d,'measurement_depth','measurement_pass',default=False)):>5s} "
            f"{str(_g(d,'settle','settled',default=False)):>6s} "
            f"{_g(d,'measurement_depth','apex_height_m')*1e3:8.2f} "
            f"{_g(d,'measurement_depth','toe_radius_m')*1e3:7.1f} "
            f"{_g(d,'measurement_depth','plateau_over_toe_ratio'):7.3f} "
            f"{str(_g(d,'measurement_depth','checks','heap_is_conical',default=False)):>5s} "
            f"{_g(d,'creep','angle_drift_deg'):9.3f} "
            f"{rms:9.2f} {_g(d,'timing','wall_s'):7.1f}"
        )


def diagnose(label: str, cells: dict[str, dict[str, Any]]) -> int:
    """Depth vs analytic height field + radial profiles + the fitted band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = cells[label]
    z = np.load(OUT_DIR / f"cell_{label}.npz", allow_pickle=True)
    hd = z["height_depth_m"].astype(np.float64)
    ha = z["height_analytic_m"].astype(np.float64)
    axis = z["grid_axis_m"]
    cell = float(axis[1] - axis[0])
    md = d["measurement_depth"]
    ma = d["measurement_analytic_readback"]

    prof_d = p3.radial_profile(hd, axis, tuple(md["heap_axis_xy_m"]), cell)
    prof_a = p3.radial_profile(ha, axis, tuple(ma.get("heap_axis_xy_m", [0.0, 0.0])), cell)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    vmax = max(hd.max(), ha.max()) * 1e3
    ext = [axis[0] * 1e3, axis[-1] * 1e3, axis[0] * 1e3, axis[-1] * 1e3]
    im0 = ax[0, 0].imshow(hd * 1e3, origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="viridis")
    ax[0, 0].set_title(f"{label}\ndepth-render height field [mm]")
    plt.colorbar(im0, ax=ax[0, 0])
    im1 = ax[0, 1].imshow(ha * 1e3, origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="viridis")
    ax[0, 1].set_title("analytic sphere-surface height field (readback) [mm]")
    plt.colorbar(im1, ax=ax[0, 1])
    diff = (hd - ha) * 1e3
    lim = float(np.abs(diff).max())
    im2 = ax[1, 0].imshow(diff, origin="lower", extent=ext, vmin=-lim, vmax=lim, cmap="coolwarm")
    ax[1, 0].set_title(f"depth - analytic [mm]  rms={diff.std():.2f} max|.|={lim:.2f}")
    plt.colorbar(im2, ax=ax[1, 0])

    ax[1, 1].plot(prof_d[:, 0] * 1e3, prof_d[:, 1] * 1e3, label="depth annulus mean")
    ax[1, 1].plot(prof_a[:, 0] * 1e3, prof_a[:, 1] * 1e3, label="analytic annulus mean")
    band = md["fit_band_m"]
    ax[1, 1].axvspan(band[0] * 1e3, band[1] * 1e3, alpha=0.15, color="tab:red",
                     label="depth fit band (80%-20% apex)")
    r = np.linspace(band[0], band[1], 10)
    slope = -math.tan(math.radians(md["repose_angle_deg"]))
    h0 = md["apex_height_m"]
    ax[1, 1].plot(
        r * 1e3,
        (h0 + slope * (r - prof_d[int(np.argmax(prof_d[:, 1])), 0])) * 1e3,
        "k--",
        label=f"depth fit {md['repose_angle_deg']:.2f} deg",
    )
    ax[1, 1].set_xlabel("radius from heap axis [mm]")
    ax[1, 1].set_ylabel("annulus mean height [mm]")
    ax[1, 1].set_xlim(0, max(md["toe_radius_m"], 0.02) * 1.3e3)
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].set_title(
        f"depth {md['repose_angle_deg']:.2f} deg (r2={md['fit_r_squared']:.3f})  "
        f"analytic {ma.get('repose_angle_deg', float('nan')):.2f} deg"
    )
    fig.tight_layout()
    png = OUT_DIR / f"diagnose_{label}.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"wrote {png}")
    return 0


def figures(cells: dict[str, dict[str, Any]]) -> int:
    """Two decision figures: the pile never stops spreading, and where PBD lands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) settle histories -------------------------------------------------
    show = [k for k in cells if k.startswith(("f1_", "f2_", "ctrl_solid_ref", "ctrl_fluid_true"))]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for k in sorted(show):
        d = cells[k]
        if d.get("failed"):
            continue
        h = d["settle"]["history"]
        t = [r["sim_time_s"] for r in h]
        ax[0].plot(t, [r["apex_m"] * 1e3 for r in h], label=k, lw=1.2)
        ax[1].plot(t, [r["toe_radius_m"] * 1e3 for r in h], label=k, lw=1.2)
        ax[2].semilogy(t, [max(r["speed_rms_m_s"], 1e-6) * 1e3 for r in h], label=k, lw=1.2)
    ax[0].set_ylabel("apex height [mm]")
    ax[1].set_ylabel("toe radius [mm]")
    ax[2].set_ylabel("particle speed rms [mm/s]")
    for a, ttl in zip(ax, ("heap apex collapses", "heap keeps spreading", "motion never stops")):
        a.set_xlabel("sim time [s]")
        a.grid(alpha=0.3)
        a.set_title(ttl)
    ax[2].axhline(1.5, color="k", ls="--", lw=1, label="P3 settle gate 1.5 mm/s rms")
    ax[0].legend(fontsize=6, ncol=2)
    ax[2].legend(fontsize=6)
    fig.suptitle("Isaac PBD solid particles, 10k pellets, depth-observed (P4 probe)")
    fig.tight_layout()
    p1 = OUT_DIR / "fig_settle_history.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # (2) time-step dependence - the headline result ------------------------
    dt_cells = {
        d["config"]["time_steps_per_second"]: d
        for k, d in cells.items()
        if not d.get("failed")
        and (k.startswith("dt_") or k == "ctrl_solid_ref")
    }
    if len(dt_cells) >= 3:
        hz = sorted(dt_cells)
        ang = [_g(dt_cells[h], "measurement_depth", "repose_angle_deg") for h in hz]
        rms = [dt_cells[h]["settle"]["history"][-1]["speed_rms_m_s"] * 1e3 for h in hz]
        apex = [_g(dt_cells[h], "measurement_depth", "apex_height_m") * 1e3 for h in hz]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ax[0].semilogx(hz, ang, "o-", color="tab:red", lw=2)
        ax[0].axhspan(*DEME_SPHERE_RANGE_DEG, alpha=0.15, color="tab:blue",
                      label="DEME sphere 8.9-25.6 deg (dt = 4e-5 s, one step for all 27 cells)")
        ax[0].axhline(TARGET_DEG, color="k", ls="--", lw=1.4,
                      label="target 30 deg (placeholder, not measured)")
        for h, y in zip(hz, ang):
            ax[0].annotate(f"{y:.1f}", (h, y), textcoords="offset points", xytext=(0, 8),
                           ha="center", fontsize=8)
        ax[0].set_xlabel("substeps per second [Hz]")
        ax[0].set_ylabel("sidewall_regression angle [deg]")
        ax[0].set_title("SAME material (friction 0.60, damping 0, adhesion 0)\n"
                        "the angle is a function of the TIME STEP, and does not converge")
        ax[0].legend(fontsize=8, loc="upper left")
        ax[0].grid(alpha=0.3, which="both")
        ax[1].loglog(hz, rms, "o-", label="residual particle speed rms [mm/s]")
        ax[1].loglog(hz, apex, "s-", label="heap apex [mm]")
        ax[1].axhline(1.5, color="k", ls="--", lw=1, label="P3 settle gate 1.5 mm/s")
        ax[1].set_xlabel("substeps per second [Hz]")
        ax[1].set_title("residual motion falls exactly in proportion to dt\n"
                        "= numerical energy injected per step, not physics")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        p3f = OUT_DIR / "fig_timestep_dependence.png"
        fig.savefig(p3f, dpi=110)
        plt.close(fig)
        print(f"wrote {p3f}")

    # (3) angle landscape --------------------------------------------------
    fig, a = plt.subplots(figsize=(11, 5.2))
    labels, vals, valid, settled = [], [], [], []
    for k, d in sorted(cells.items()):
        if d.get("failed") or k.startswith(("g3_", "g5_")):
            continue
        v = _g(d, "measurement_depth", "repose_angle_deg")
        if not math.isfinite(v):
            v = 0.0
        labels.append(k)
        vals.append(v)
        valid.append(bool(_g(d, "measurement_depth", "measurement_pass", default=False)))
        settled.append(bool(_g(d, "settle", "settled", default=False)))
    xs = np.arange(len(labels))
    a.bar(xs, vals, color=["tab:green" if v else "tab:red" for v in valid])
    for x, y, s in zip(xs, vals, settled):
        if s:
            a.annotate("settled", (x, y), textcoords="offset points", xytext=(0, 4),
                       ha="center", fontsize=6, rotation=90)
    for lo, hi, name, c in (
        (*DEME_SPHERE_RANGE_DEG, "DEME sphere", "tab:blue"),
        (*DEME_CLUMP3_RANGE_DEG, "DEME clump3", "tab:purple"),
    ):
        a.axhspan(lo, hi, alpha=0.13, color=c, label=f"{name} {lo:.1f}-{hi:.1f} deg")
    a.axhline(TARGET_DEG, color="k", ls="--", lw=1.5, label=f"target {TARGET_DEG:.0f} deg (placeholder)")
    a.set_xticks(xs)
    a.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    a.set_ylabel("sidewall_regression angle [deg]")
    a.set_title(
        "PBD cells. Red = the measurement itself failed its validity checks (e.g. the heap is a "
        "pancake, so the fit lands on the rim).\nNOT ONE cell that settled also produced a valid "
        "angle."
    )
    a.legend(fontsize=8)
    a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p2 = OUT_DIR / "fig_angle_landscape.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print(f"wrote {p1}\nwrote {p2}")
    return 0


def _settled_and_measured(d: dict) -> bool:
    return bool(
        not d.get("failed")
        and _g(d, "settle", "settled", default=False)
        and _g(d, "measurement_depth", "measurement_pass", default=False)
    )


def gates(cells: dict[str, dict[str, Any]]) -> int:
    """Evaluate the pre-registered G1..G5 of GATES_pbd_probe.md."""
    verdicts: dict[str, dict[str, Any]] = {}
    usable = {k: v for k, v in cells.items() if not v.get("failed")}

    # ---- G1 ---------------------------------------------------------------
    angles = {
        k: _g(v, "measurement_depth", "repose_angle_deg")
        for k, v in usable.items()
        if not k.startswith(("g5_", "dt_"))
    }
    valid = {k: a for k, a in angles.items() if _settled_and_measured(usable[k])}
    on_target = {
        k: a for k, a in valid.items() if abs(a - TARGET_DEG) <= TARGET_TOL_DEG
    }
    seeds = {}
    for k in on_target:
        seeds.setdefault(
            tuple(
                usable[k]["config"][f]
                for f in ("friction", "particle_friction_scale", "damping", "adhesion",
                          "particle_adhesion_scale", "adhesion_offset_scale",
                          "solver_position_iterations", "time_steps_per_second")
            ),
            [],
        ).append(usable[k]["config"]["seed"])
    best_family = max(seeds.items(), key=lambda kv: len(set(kv[1])), default=(None, []))
    reproduced = len(set(best_family[1])) >= 3
    max_any = max(angles.values()) if angles else float("nan")
    max_valid = max(valid.values()) if valid else float("nan")
    verdicts["G1"] = {
        "name": "angle of repose reaches the 30 +/- 3 deg target",
        "pass": bool(on_target and reproduced),
        "cells_evaluated": len(angles),
        "cells_settled_and_valid": len(valid),
        "max_angle_any_cell_deg": max_any,
        "max_angle_settled_valid_deg": max_valid,
        "on_target_cells": sorted(on_target),
        "distinct_seeds_for_best_family": len(set(best_family[1])),
        "reachable_27deg": bool(math.isfinite(max_valid) and max_valid >= G1_MIN_REACHABLE_DEG),
        "deme_sphere_range_deg": list(DEME_SPHERE_RANGE_DEG),
        "excluded_from_this_gate": {
            "labels": sorted(k for k in usable if k.startswith(("g5_", "dt_"))),
            "why": (
                "g5_* are scoop-repeat trials of an already-evaluated parameter set, and dt_* are "
                "the G4 time-step convergence study - a NUMERICAL axis, not a material one. They "
                "are excluded so G1 cannot be passed by choosing a time step. Nothing is hidden: "
                "the best dt_* angle is reported below and is discussed in the report."
            ),
            "best_dt_cell_angle_deg": max(
                [
                    _g(v, "measurement_depth", "repose_angle_deg")
                    for k, v in usable.items()
                    if k.startswith("dt_")
                    and math.isfinite(_g(v, "measurement_depth", "repose_angle_deg"))
                ],
                default=float("nan"),
            ),
            "best_dt_cell_settled": any(
                _g(v, "settle", "settled", default=False)
                for k, v in usable.items()
                if k.startswith("dt_")
            ),
        },
        "blind_spot": (
            "30 deg is a placeholder, so PASS would mean 'PBD can be dialled near 30 deg', not "
            "'PBD matches polypropylene'. One scalar also fixes no flow rule: two materials with "
            "the same repose angle can respond completely differently to a scoop, and a poured "
            "free surface is a different boundary condition from a cut scoop face."
        ),
    }

    # ---- G2 ---------------------------------------------------------------
    settled = {k: v for k, v in usable.items() if _g(v, "settle", "settled", default=False)}
    creeps = {
        k: {
            "angle_drift_deg": _g(v, "creep", "angle_drift_deg"),
            "toe_drift_mm": _g(v, "creep", "toe_drift_mm"),
            "apex_drift_mm": _g(v, "creep", "apex_drift_mm"),
        }
        for k, v in settled.items()
    }
    ok_creep = {
        k: c
        for k, c in creeps.items()
        if abs(c["angle_drift_deg"]) <= 1.0 and abs(c["toe_drift_mm"]) <= 1.048
        and abs(c["apex_drift_mm"]) <= 0.419
    }
    verdicts["G2"] = {
        "name": "the heap settles and then stops moving",
        "pass": bool(ok_creep),
        "cells_total": len(usable),
        "cells_that_settled": len(settled),
        "cells_that_settled_labels": sorted(settled),
        "cells_that_also_survived_the_creep_watch": sorted(ok_creep),
        "creep_detail": creeps,
        "blind_spot": (
            "10 s of simulated watching. A creep slower than ~0.1 mm/s is invisible here and would "
            "still ruin a minutes-long scooping episode. The pile is also never perturbed, so a "
            "metastable heap passes."
        ),
    }

    # ---- G3 ---------------------------------------------------------------
    scooped = {k: v for k, v in usable.items() if "scoop" in v}
    g3_rows = {}
    for k, v in scooped.items():
        s = v["scoop"]
        cfgd = v["config"]
        tool_speed = abs(cfgd["scoop_start_r_m"]) / cfgd["scoop_travel_s"]
        diameter = 2.0 * _g(v, "derived", "sphere_radius_m")
        apex = _g(v, "measurement_depth", "apex_height_m")
        g3_rows[k] = {
            "tool_speed_m_s": tool_speed,
            "max_particle_speed_m_s": s["max_particle_speed_during_entry_m_s"],
            "speed_ratio": s["max_particle_speed_during_entry_m_s"] / tool_speed,
            "speed_ratio_limit": 10.0,
            "max_height_during_entry_m": s["max_particle_height_during_entry_m"],
            "height_limit_m": apex + 2.0 * diameter,
            "radial_growth_mm": (
                s["max_radial_extent_during_entry_m"] - s["max_radial_extent_before_m"]
            ) * 1e3,
            "radial_growth_limit_mm": 2.0 * diameter * 1e3,
        }
        g3_rows[k]["pass"] = bool(
            g3_rows[k]["speed_ratio"] <= 10.0
            and g3_rows[k]["max_height_during_entry_m"] <= g3_rows[k]["height_limit_m"]
            and g3_rows[k]["radial_growth_mm"] <= g3_rows[k]["radial_growth_limit_mm"]
        )
    verdicts["G3"] = {
        "name": "scoop entry does not blow the pile up",
        "pass": bool(g3_rows) and all(r["pass"] for r in g3_rows.values()),
        "cells": g3_rows,
        "blind_spot": (
            "one scoop geometry, one entry point, one speed, one pile. A slow plunge is the easy "
            "case; a fast or rotating grab is not covered."
        ),
    }

    # ---- G4 ---------------------------------------------------------------
    dt_cells = {k: v for k, v in usable.items() if k.startswith("dt_")}
    by_hz = {v["config"]["time_steps_per_second"]: v for v in dt_cells.values()}
    g4 = {
        "rates_run": sorted(by_hz),
        "angles_deg": {
            hz: _g(v, "measurement_depth", "repose_angle_deg") for hz, v in sorted(by_hz.items())
        },
        "measurement_pass": {
            hz: _g(v, "measurement_depth", "measurement_pass", default=False)
            for hz, v in sorted(by_hz.items())
        },
        "settled": {
            hz: _g(v, "settle", "settled", default=False) for hz, v in sorted(by_hz.items())
        },
    }
    # the 60 Hz point of the convergence study is the base friction=0.60 cell
    if 60 not in by_hz and "ctrl_solid_ref" in usable:
        by_hz[60] = usable["ctrl_solid_ref"]
        for k in ("angles_deg", "measurement_pass", "settled"):
            pass
        g4["angles_deg"][60] = _g(usable["ctrl_solid_ref"], "measurement_depth", "repose_angle_deg")
        g4["measurement_pass"][60] = _g(
            usable["ctrl_solid_ref"], "measurement_depth", "measurement_pass", default=False
        )
        g4["settled"][60] = _g(usable["ctrl_solid_ref"], "settle", "settled", default=False)
        g4["rates_run"] = sorted(by_hz)
        g4["note_60hz"] = "60 Hz point taken from ctrl_solid_ref (same config, base rate)"
    if 60 in by_hz and 240 in by_hz:
        g4["delta_60_vs_240_deg"] = abs(g4["angles_deg"][60] - g4["angles_deg"][240])
        g4["pass"] = bool(
            g4["delta_60_vs_240_deg"] <= 2.0 and all(g4["measurement_pass"].values())
        )
    else:
        g4["pass"] = False
        g4["note"] = "60 Hz and 240 Hz cells are required and were not both produced"
    # A monotone angle that is still climbing at the smallest step run is the
    # informative statement, not just "the two ends differ".
    ordered = [g4["angles_deg"][hz] for hz in sorted(g4["angles_deg"]) if math.isfinite(g4["angles_deg"][hz])]
    hz_sorted = [hz for hz in sorted(g4["angles_deg"]) if math.isfinite(g4["angles_deg"][hz])]
    if len(ordered) >= 3:
        g4["angle_change_over_last_halving_deg"] = abs(ordered[-1] - ordered[-2])
        g4["converged"] = bool(g4["angle_change_over_last_halving_deg"] <= 1.0)
        g4["smallest_step_run_s"] = 1.0 / hz_sorted[-1]
        g4["trend"] = (
            "angle still increasing with smaller step; NOT converged"
            if ordered[-1] > ordered[-2] + 1.0
            else "angle change under 1 deg over the last halving"
        )
    g4["name"] = "no absurdly small time step is needed"
    g4["blind_spot"] = (
        "convergence of a STATIC settled heap says nothing about the step a moving scoop needs."
    )
    verdicts["G4"] = g4

    # ---- G5 ---------------------------------------------------------------
    g5_cells = {k: v for k, v in usable.items() if k.startswith("g5_") and "scoop" in v}
    vols = np.array([v["scoop"]["removed_volume_m3"] for v in g5_cells.values()])
    nets = np.array([v["scoop"]["net_removed_volume_m3"] for v in g5_cells.values()])
    counts = np.array([v["scoop"]["particles_in_bucket"] for v in g5_cells.values()])
    if vols.size >= 3:
        cov = float(vols.std(ddof=1) / vols.mean()) if vols.mean() > 0 else float("nan")
        cov_n = float(counts.std(ddof=1) / counts.mean()) if counts.mean() > 0 else float("nan")
    else:
        cov = cov_n = float("nan")
    # PRECONDITION, not a moved threshold: a repeatability number is only about a
    # scooped amount if a scooped amount exists.  With an empty bucket and a
    # NEGATIVE net removed volume, the CoV describes how much the (still moving)
    # pile drifted between the two frames, so reporting it as a PASS would be a
    # measurement of noise dressed as a result.
    actually_removed = bool(counts.size and counts.mean() > 0 and nets.mean() > 0)
    verdicts["G5"] = {
        "name": "scooped amount repeats across pours",
        "pass": bool(actually_removed and math.isfinite(cov) and cov <= 0.10 and vols.size >= 5),
        "not_evaluable": not actually_removed,
        "not_evaluable_reason": (
            None
            if actually_removed
            else (
                f"the tool caught {counts.tolist()} particles and the net removed volume is "
                f"{nets.mean()*1e6:.2f} cm^3 (negative = the pile GREW). Nothing was scooped, so "
                "the coefficient of variation below is the scatter of an unsettled pile drifting "
                "between the before and after frames, not the scatter of a scooped amount. "
                "Reported, not passed."
            )
        ),
        "marginal": bool(actually_removed and math.isfinite(cov) and 0.10 < cov <= 0.15),
        "n_trials": int(vols.size),
        "removed_volume_m3": vols.tolist(),
        "removed_volume_cov": cov,
        "net_removed_volume_m3": nets.tolist(),
        "particles_in_bucket": counts.tolist(),
        "particles_in_bucket_cov": cov_n,
        "deme_reference": "lip-equivalent reaction CoV 1.012; captured particles 191-204",
        "blind_spot": (
            "fixed trajectory, piles differ only through the pour seed. This is pour-to-pour "
            "scatter, not sensitivity to scoop placement, which is the variable the proposal "
            "actually wants to learn."
        ),
    }

    all_pass = all(v.get("pass") for v in verdicts.values())
    out = {
        "artifact": "PBD_PELLET_PROBE_GATES",
        "verdicts": verdicts,
        "verdict": "ISAAC_PBD_ADOPT" if all_pass else "KEEP_DEME",
        "verdict_rule": "Isaac PBD replaces DEME only if G1..G5 all pass; any FAIL keeps DEME",
    }
    path = OUT_DIR / "gates_pbd_probe.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    for name, v in verdicts.items():
        print(f"{name}: {'PASS' if v.get('pass') else 'FAIL'}  {v['name']}")
    print(f"VERDICT: {out['verdict']}")
    print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--diagnose", type=str)
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args(argv)
    OUT_DIR = Path(args.out_dir)
    cells = load_cells(OUT_DIR)
    if args.table:
        table(cells)
    if args.diagnose:
        diagnose(args.diagnose, cells)
    if args.figures:
        figures(cells)
    if args.gates:
        return gates(cells)
    if not (args.table or args.diagnose or args.gates or args.figures):
        ap.error("pick --table / --diagnose / --figures / --gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
