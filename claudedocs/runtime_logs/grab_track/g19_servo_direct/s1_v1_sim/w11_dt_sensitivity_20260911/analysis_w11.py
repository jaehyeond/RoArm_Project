"""W10/W11 dt comparison from saved raw data; no DEME or robot initialization."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from w11_workflow import OUT, REPO, BASE, CELL, PARAMS, read, save, sha
sys.path.insert(0, str(REPO))
from roarm_rl.heightmap import GridSpec, heightmap_from_particles


def cell_metrics(cell):
    result_path = cell / "scoop_s1_seed460.json"
    res = read(result_path)
    z = np.load(cell / "scoop_s1_seed460.npz")
    timeline = read(cell / "timeline_seed460.json")
    rows = timeline["rows"]
    rt = np.load(res["render_timeline"]["path"])
    P = res["params"]
    pile_path = next(k for k in res["inputs_sha16"] if k.endswith(".npz"))
    pile = np.load(pile_path, allow_pickle=True)
    tpl = json.loads(str(pile["clump_template_json"]))
    checks = {}
    checks["all_input_hashes_match"] = all(sha(p)[:16] == h for p, h in res["inputs_sha16"].items())
    checks["seed_460_full_run"] = res["seed"] == 460 and not res["smoke"]
    t = np.asarray(z["frame_t_s"])
    checks["timeline_times_match_arrays"] = np.array_equal(t, np.array([r["sim_t"] for r in rows]))
    phase_counts = Counter(r["phase"] for r in rows)
    checks["steps_match"] = len(t) == res["steps_completed"] and {k: phase_counts[k] for k in res["trajectory"]["steps_actual"]} == res["trajectory"]["steps_actual"]
    phase_ids = {"settle": 0, "descend": 1, "close": 2, "lift": 3, "reclose": 4}
    checks["timeline_phases_match_arrays"] = np.array_equal(z["frame_phase"], [phase_ids[r["phase"]] for r in rows])
    pp = np.asarray(z["positions_m"], float)
    # Reconstruct the fixed mesh translation from its saved world nodes and OBJ.
    fixed = trimesh.load(cell / "_obj/fixed_seed460.obj", process=False)
    lip = np.asarray(z["nodes_F_m"][-1], float).mean(0) - np.asarray(fixed.vertices, float).mean(0)
    rw = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
    origin = lip - rw @ (np.asarray(P["lip_l5_mm"]) / 1000)
    local = (pp - origin) @ rw
    center = np.asarray(P["bowl_center_l5_mm"]) / 1000
    radial = np.hypot(local[:, 0] - center[0], local[:, 2] - center[2])
    margin_r = P["bowl_r_in_mm"] / 1000 - radial
    margin_y = P["cheek_half_y_mm"] / 1000 - np.abs(local[:, 1])
    capture = (margin_r > 0) & (margin_y > 0)
    checks["cavity_geometry_matches_mask"] = np.array_equal(capture, z["in_cavity"])
    checks["capture_ids_match_render_timeline"] = np.array_equal(np.where(capture)[0], rt["captured_ids"])
    count = int(capture.sum())
    mass = count * tpl["mass_kg"] * 1000
    checks["count_matches_json"] = count == res["capture"]["n_in_cavity"]
    checks["mass_matches_json_rounding"] = abs(mass - res["capture"]["mass_g"]) <= 0.000050001
    checks["particle_mass_matches_template"] = res["particle"]["mass_kg"] == tpl["mass_kg"]
    # Input identity is checked by hash. Initialize/readback may quantize positions;
    # measure that discrepancy, and compare the two actual initialized arrays later.
    initial_position_error = float(np.abs(rt["clump_pos_m"][0].astype(float) - pile["clump_positions_m"]).max())
    initial_positions_exact = np.array_equal(rt["clump_pos_m"][0], pile["clump_positions_m"].astype(np.float32))
    checks["initial_orientations_match_same_pile"] = np.array_equal(rt["clump_quat_xyzw"][0], pile["clump_quaternions_xyzw"].astype(np.float32))
    checks["last_particle_positions_match"] = np.array_equal(rt["clump_pos_m"][-1], pp.astype(np.float32))
    offsets = np.asarray(tpl["offsets_m"], float)
    sp = (pp[:, None, :] + np.einsum("nij,kj->nki", Rotation.from_quat(z["clump_quaternions_xyzw"]).as_matrix(), offsets)).reshape(-1, 3)
    sr = np.tile(tpl["sphere_radii_m"], len(pp))
    checks["sphere_expansion_matches_final_raw"] = np.allclose(sp, z["sphere_positions_m"], rtol=0, atol=1e-14)
    box = np.asarray(z["box_bounds_m"], float)
    spec = GridSpec(origin_xy_m=(box[0, 0], box[1, 0]), cell_m=res["heightmap"]["cell_m"], shape=tuple(z["heightmap_m"].shape), frame="deme_box_floor_center", z_datum_m=0.0)
    rest_spheres = np.repeat(~z["carried"], len(offsets))
    recomputed_hm = heightmap_from_particles(sp[rest_spheres], sr[rest_spheres], spec).height
    checks["post_heightmap_recomputed"] = np.array_equal(recomputed_hm, z["heightmap_m"])
    checks["initial_heightmap_recomputed"] = np.array_equal(heightmap_from_particles(pile["positions_m"], pile["radii_m"], spec).height, z["heightmap_npz_m"])
    vmax = max(r["v_particle_max"] for r in rows)
    over5 = sum(r["v_particle_max"] > P["pop_speed_m_s"] for r in rows)
    checks["max_speed_matches_json_rounding"] = abs(vmax - res["pops"]["v_particle_max_m_s"]) <= 0.000550001
    checks["speed_warning_count_matches"] = over5 == res["pops"]["steps_over_pop_speed"]
    checks["finite_raw_arrays"] = all(np.isfinite(z[k]).all() for k in ("positions_m", "nodes_F_m", "nodes_D_m", "contact_force_N", "heightmap_m"))
    stops = []
    q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
    door_mesh = trimesh.load(cell / f"_obj/door_open{q_open:.1f}_seed460.obj", process=False)
    door_base = np.asarray(door_mesh.vertices, float)
    door_nodes = np.asarray(z["nodes_D_m"], float)

    def angle_from_mesh(nodes):
        u, _, vt = np.linalg.svd((door_base - door_base.mean(0)).T @ (nodes - nodes.mean(0)))
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(vt.T @ u.T)
        rot = vt.T @ correction @ u.T
        residual = (door_base - door_base.mean(0)) @ rot.T - (nodes - nodes.mean(0))
        return float(q_open - np.degrees(Rotation.from_matrix(rot).as_rotvec()[0])), float(np.linalg.norm(residual, axis=1).max() * 1e6)

    for stop in res["door"]["stops"]:
        ix = int(np.argmin(abs(t - stop.get("sim_t", t[-1]))))
        row = rows[ix]
        checks[f"{stop['phase']}_stop_matches_timeline"] = row["phase"] == stop["phase"] and row["q_deg"] == stop["q_deg"] and abs(row["M_hinge_res_Nm"] - stop.get("M_hinge_res_Nm", row["M_hinge_res_Nm"])) < 1e-9
        mesh_angle, fit_residual = angle_from_mesh(door_nodes[ix])
        checks[f"{stop['phase']}_actual_angle_matches_mesh"] = abs(mesh_angle - stop["q_from_quat_deg"]) < 0.001
        stops.append(dict(stop, nominal_servo_deg=stop["q_deg"] + P["servo_zero_offset_deg"], actual_servo_deg=stop["q_from_quat_deg"] + P["servo_zero_offset_deg"], timeline_index=ix,
                          actual_angle_from_mesh_deg=mesh_angle, mesh_rigid_fit_max_residual_um=fit_residual))
    stderr = (cell / "stderr.txt").read_text(errors="replace")
    warnings = [line for line in stderr.splitlines() if re.search(r"warning|anomal|exceed|error", line, re.I)]
    complete = not res["diverged"] and timeline["state"] == "complete" and all(res["trajectory"]["steps_actual"].get(k, 0) > 0 for k in phase_ids) and len(stops) == 2
    metrics = {"cell": str(cell), "result_sha256": sha(result_path), "dt_s": P["timestep_s"], "completed": complete,
               "diverged": res["diverged"], "timeline_state": timeline["state"], "syncs": len(t), "particle_frames": len(rt["t_s"]),
               "physical_time_s": float(t[-1]), "physics_loop_wall_s": res["wall_seconds"], "steps_actual": res["trajectory"]["steps_actual"],
               "stops": stops, "door": res["door"], "capture_count": count, "capture_mass_g": mass, "n_carried_z": int(z["carried"].sum()),
               "capture_boundary_min_margin_um": float(np.minimum(abs(margin_r), abs(margin_y)).min() * 1e6),
               "initial_readback_equals_pile_at_float32": bool(initial_positions_exact), "initial_readback_vs_pile_max_abs_m": initial_position_error,
               "max_speed_m_s_saved_syncs": vmax, "over_5m_s_syncs": over5, "maximum_speed_row": max(rows, key=lambda r: r["v_particle_max"]),
               "stderr_warning_lines": warnings, "events": [read(p)["trigger"] for p in sorted(cell.glob("diverge_event*_seed460.json"))],
               "forces": res["forces"], "scoop_site": res["scoop_site"], "trajectory": res["trajectory"], "heightmap": res["heightmap"],
               "raw_crater_angles_deg": {k: v.get("angle_deg") for k, v in res["crater"]["azimuths"].items()},
               "checks": {k: bool(v) for k, v in checks.items()}, "evidence_pass": all(checks.values())}
    return metrics


def spatial_stats(diff_mm, mask):
    a = diff_mm[mask]
    return {"cells": int(len(a)), "mae_mm": float(abs(a).mean()), "rmse_mm": float(np.sqrt((a * a).mean())),
            "max_abs_mm": float(abs(a).max()), "mean_signed_mm": float(a.mean()), "cells_over_5mm": int((abs(a) > 5).sum())}


def compare():
    metrics = {"w10": cell_metrics(BASE), "w11": cell_metrics(CELL)}
    a, b = metrics["w10"], metrics["w11"]
    za, zb = np.load(BASE / "scoop_s1_seed460.npz"), np.load(CELL / "scoop_s1_seed460.npz")
    pa, pb = read(BASE / "scoop_s1_seed460.json")["params"], read(CELL / "scoop_s1_seed460.json")["params"]
    pd = {k: [pa.get(k), pb.get(k)] for k in set(pa) | set(pb) if pa.get(k) != pb.get(k)}
    assert set(pd) == {"timestep_s", "render_timeline_path"}, pd
    box = za["box_bounds_m"]
    cell_m = a["heightmap"]["cell_m"]
    shape = za["heightmap_m"].shape
    assert np.array_equal(box, zb["box_bounds_m"]) and shape == zb["heightmap_m"].shape
    x = box[0, 0] + (np.arange(shape[1]) + 0.5) * cell_m
    y = box[1, 0] + (np.arange(shape[0]) + 0.5) * cell_m
    X, Y = np.meshgrid(x, y)
    region = np.hypot(X - a["scoop_site"]["x_mm"] / 1000, Y - a["scoop_site"]["y_mm"] / 1000) <= 0.08
    full = np.ones(shape, bool)
    maps = {}
    for name, arr in {"initial": zb["heightmap_npz_m"].astype(float) - za["heightmap_npz_m"],
                      "pre": zb["heightmap_pre_m"].astype(float) - za["heightmap_pre_m"],
                      "post": zb["heightmap_m"].astype(float) - za["heightmap_m"],
                      "removed_depth": (zb["heightmap_pre_m"].astype(float) - zb["heightmap_m"]) - (za["heightmap_pre_m"].astype(float) - za["heightmap_m"])}.items():
        maps[name] = {"whole_grid": spatial_stats(arr * 1000, full), "site_radius_80mm": spatial_stats(arr * 1000, region)}
    delta = {"capture_count": b["capture_count"] - a["capture_count"], "capture_mass_g": b["capture_mass_g"] - a["capture_mass_g"],
             "capture_percent": 100 * (b["capture_count"] / a["capture_count"] - 1),
             "max_speed_m_s": b["max_speed_m_s_saved_syncs"] - a["max_speed_m_s_saved_syncs"],
             "physics_loop_wall_ratio": b["physics_loop_wall_s"] / a["physics_loop_wall_s"]}
    for i, name in enumerate(("close", "reclose")):
        if len(a["stops"]) > i and len(b["stops"]) > i:
            delta[f"{name}_nominal_deg"] = b["stops"][i]["q_deg"] - a["stops"][i]["q_deg"]
            delta[f"{name}_actual_deg"] = b["stops"][i]["q_from_quat_deg"] - a["stops"][i]["q_from_quat_deg"]
    improved = b["completed"] and b["max_speed_m_s_saved_syncs"] < 5 and b["over_5m_s_syncs"] == 0
    rta = np.load(read(BASE / "scoop_s1_seed460.json")["render_timeline"]["path"])
    rtb = np.load(read(CELL / "scoop_s1_seed460.json")["render_timeline"]["path"])
    initial_checks = {k: bool(np.array_equal(rta[k][0], rtb[k][0])) for k in ("clump_pos_m", "clump_quat_xyzw", "tool_pos_m", "tool_quat_xyzw", "door_pos_m", "door_quat_xyzw")}
    result = {"artifact": "W11_DT_COMPARISON", "baseline": a, "new": b, "effective_params_diff": pd, "delta": delta,
              "actual_initialized_state_exact_checks": initial_checks,
              "heightmap_differences_w11_minus_w10": maps,
              "scientific_observations": {"completed_both": a["completed"] and b["completed"],
                 "two_torque_stops_retained": len(b["stops"]) == 2 and all(s["reason"] == "servo_stall" for s in b["stops"]),
                 "warning_improvement_criterion": improved, "strict_convergence_claimed": False, "sim_real_match_claimed": False},
              "evidence_pass": a["evidence_pass"] and b["evidence_pass"] and all(initial_checks.values()),
              "limitations": ["One run per dt; within-dt run variation is not measured.", "Maximum speed is sampled at saved syncs, not every internal DEME step.",
                              "Post heightmap excludes height-classified carried particles, not velocity-classified moving particles.",
                              "Capture mass is inside the scoop, not delivered mass.", "Initial pile is identical, but settle outcomes and event-dependent stop times can differ."]}
    return result


def plots(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    za, zb = np.load(BASE / "scoop_s1_seed460.npz"), np.load(CELL / "scoop_s1_seed460.npz")
    box = za["box_bounds_m"]
    extent = [*list(box[0] * 1000), *list(box[1] * 1000)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    pairs = [("pre", "heightmap_pre_m"), ("post", "heightmap_m")]
    for ri, (name, key) in enumerate(pairs):
        arrays = [za[key].astype(float) * 1000, zb[key].astype(float) * 1000]
        top = max(x.max() for x in arrays)
        diff = arrays[1] - arrays[0]
        for ci, label in enumerate(("W10 2 us", "W11 1 us")):
            im = axes[ri, ci].imshow(arrays[ci], origin="lower", extent=extent, vmin=0, vmax=top, cmap="viridis")
            axes[ri, ci].set_title(f"{label}: {name} height (mm)")
            fig.colorbar(im, ax=axes[ri, ci], shrink=0.8)
        dlim = max(abs(diff).max(), 0.001)
        im = axes[ri, 2].imshow(diff, origin="lower", extent=extent, cmap="RdBu_r", vmin=-dlim, vmax=dlim)
        stat = result["heightmap_differences_w11_minus_w10"][name]["site_radius_80mm"]
        axes[ri, 2].set_title(f"W11 - W10: {name}\nROI MAE {stat['mae_mm']:.3f} mm, max {stat['max_abs_mm']:.3f} mm")
        fig.colorbar(im, ax=axes[ri, 2], shrink=0.8)
    for ax in axes.flat:
        ax.add_patch(plt.Circle((0, 0), 80, fill=False, linestyle="--", linewidth=0.8, color="white"))
        ax.plot(0, 0, "+", color="black")
        ax.set_xlabel("world x (mm)"); ax.set_ylabel("world y (mm)")
    fig.suptitle("Same initial pile; re-settle pre and post compared separately. Dashed circle: 80 mm ROI.")
    fig.savefig(OUT / "heightmap_comparison.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for cell, label in [(BASE, "W10 2 us"), (CELL, "W11 1 us")]:
        rows = read(cell / "timeline_seed460.json")["rows"]
        t = [r["sim_t"] for r in rows]
        for ax, key in zip(axes, ("q_deg", "M_hinge_res_Nm", "v_particle_max")):
            ax.plot(t, [r[key] for r in rows], label=label, linewidth=0.8)
    axes[0].set_ylabel("nominal door (deg)")
    axes[1].set_ylabel("hinge resistance (N m)"); axes[1].axhline(1.96 * 0.9, color="black", ls="--", lw=0.7)
    axes[2].set_ylabel("max particle speed (m/s)"); axes[2].axhline(5, color="red", ls="--", lw=0.7)
    axes[2].set_xlabel("physical time (s)")
    for ax in axes:
        ax.grid(alpha=0.2); ax.legend()
    fig.suptitle("W11 dt comparison: all saved syncs; event-dependent stop times may differ")
    fig.savefig(OUT / "timeline_comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    result = compare()
    save(OUT / "comparison.json", result)
    plots(result)
    print(json.dumps({"evidence_pass": result["evidence_pass"], "delta": result["delta"], "observations": result["scientific_observations"]}, indent=2))
    raise SystemExit(0 if result["evidence_pass"] else 1)
