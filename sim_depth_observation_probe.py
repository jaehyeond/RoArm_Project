#!/usr/bin/env python3
"""Observe the DEME heaps through the SAME depth render the P4 probe used (track P4b).

Why
---
The P4 probe replaced particle-coordinate readback with a top-down depth render,
because that is what the physical rig can actually measure.  But every PBD heap
it produced was a pancake, so the depth path was never tested on a real cone -
and a cone is exactly where a perspective depth camera is hardest: the sidewall
is foreshortened, grains occlude each other, and D453's slope law says a
horizontal error becomes `eps * tan(theta)` of vertical error on a slope.

Track P3 already has 27 settled DEME heaps spanning 8.90-41.77 deg with the
angle measured analytically from exact sphere coordinates.  That is a ground
truth.  This module re-observes those very heaps through the depth camera and
reports the difference, which answers two questions at once:

1. How much does depth observation bias the angle on a REAL cone, as a function
   of how steep the cone is?
2. Is it safe to move the DEME pipeline onto depth-render heightmaps, so that
   sim and real share one observation model and the GP residual carries only the
   physics difference?

What it does NOT do
-------------------
No physics.  The sphere centres come from `sim_pellet_model.sphere_centres_world`
applied to the coordinates stored in each P3 NPZ, i.e. the exact configuration
P3 measured.  An integrity gate re-runs `sim_pellet_model.measure_repose_angle`
on the reconstruction and requires it to reproduce the published angle to
1e-9 deg before any render happens, so a reconstruction bug cannot masquerade as
a depth-observation bias.

DEME is NOT imported and no DEME file is touched; this reads P3's NPZ artifacts
only.  Runs in the `isaaclab` interpreter (Isaac Sim 5.1), not `roarm`.

Usage:
  ~/miniconda3/envs/isaaclab/bin/python sim_depth_observation_probe.py --all
  ~/miniconda3/envs/isaaclab/bin/python sim_depth_observation_probe.py --npz <file> [...]
  python sim_depth_observation_probe.py --summarise        # no Isaac needed
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import sim_pellet_model as p3
import sim_pbd_pellet_probe as p4

P3_DIR = Path("claudedocs/runtime_logs/pellet_model")
OUT_DIR = Path("claudedocs/runtime_logs/pbd_probe/depth_on_deme")
ARTIFACT = "DEPTH_OBSERVATION_ON_DEME_V1"

NON_CLAIMS = [
    "MEASURE: the underlying DEME heaps are P3 artifacts built from unmeasured placeholder pellet "
    "dimensions, density and friction. Nothing here is a property of real polypropylene.",
    "NOT_A_PHYSICS_RESULT: this module runs no physics. Every angle difference reported is a "
    "difference between two ways of OBSERVING one fixed configuration.",
    "RENDER_ONLY: the depth camera is an ideal pinhole with no sensor noise, no quantisation and "
    "no multipath. It bounds the geometric bias of depth observation and says nothing about the "
    "Azure Kinect's own error, which D453 and the 10.13 mm hand-eye RMSE cover separately.",
]


def measurement_config(meta: dict[str, Any]) -> p4.PbdConfig:
    """A PbdConfig carrying the DEME run's own measurement knobs, so the depth
    side uses byte-for-byte the same `sidewall_regression` definition."""
    c = meta["config"]
    return replace(
        p4.PbdConfig(),
        profile_cell_mm=c["profile_cell_mm"],
        fit_upper_height_fraction=c["fit_upper_height_fraction"],
        fit_lower_height_fraction=c["fit_lower_height_fraction"],
        toe_height_fraction_of_diameter=c["toe_height_fraction_of_diameter"],
        min_fit_r_squared=c["min_fit_r_squared"],
        max_quadrant_spread_deg=c["max_quadrant_spread_deg"],
        wall_clearance_fraction=c["wall_clearance_fraction"],
        min_heap_fraction=c["min_heap_fraction"],
    )


def load_heap(npz_path: Path) -> dict[str, Any]:
    """Rebuild one P3 heap and PROVE the reconstruction is the measured one."""
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["metadata_json"]))
    centres = p3.sphere_centres_world(
        z["clump_positions_m"], z["clump_quaternions_xyzw"], z["sphere_offsets_m"]
    )
    radius = float(meta["template"]["sphere_radius_m"])
    half_extent = float(meta["derived_geometry"]["half_extent_m"])

    cfg_p3 = p3.ReposeConfig(**meta["config"])
    geom = {"half_extent_m": half_extent}
    redone = p3.measure_repose_angle(centres, radius, cfg_p3, geom)
    published = meta["repose"]["repose_angle_deg"]
    delta = abs(redone["repose_angle_deg"] - published)
    if delta > 1.0e-9:
        raise RuntimeError(
            f"RECONSTRUCTION_GATE_FAIL {npz_path.name}: re-measuring the rebuilt sphere centres "
            f"gives {redone['repose_angle_deg']:.12f} deg but the artifact published "
            f"{published:.12f} deg (delta {delta:.3e}). The depth comparison would be meaningless."
        )
    return {
        "npz": npz_path,
        "meta": meta,
        "centres": centres,
        "radius_m": radius,
        "half_extent_m": half_extent,
        "analytic": redone,
        "published_angle_deg": published,
        "reconstruction_delta_deg": delta,
        "shape": meta["config"]["shape"],
        "mu": meta["config"]["particle_mu"],
        "crr": meta["config"]["rolling_friction"],
    }


def run(npz_paths: list[Path], cam_res: int) -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    rc = 0
    rows: list[dict[str, Any]] = []
    try:
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom, UsdLux, Vt

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for path in npz_paths:
            t0 = time.time()
            heap = load_heap(path)
            centres = heap["centres"]
            radius = heap["radius_m"]
            half_extent = heap["half_extent_m"]
            cfg = measurement_config(heap["meta"])
            cell_m = cfg.profile_cell_mm * 1.0e-3

            omni.usd.get_context().new_stage()
            stage = omni.usd.get_context().get_stage()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
            stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
            UsdLux.DistantLight.Define(stage, Sdf.Path("/World/light")).CreateIntensityAttr(3000.0)

            # floor at exactly z=0, matching P3's bare-floor convention
            e = half_extent * 1.4
            floor = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/floor"))
            floor.CreatePointsAttr([(-e, -e, 0), (e, -e, 0), (e, e, 0), (-e, e, 0)])
            floor.CreateFaceVertexCountsAttr([4])
            floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            floor.CreateExtentAttr([(-e, -e, 0), (e, e, 0)])

            inst = UsdGeom.PointInstancer.Define(stage, Sdf.Path("/World/heap"))
            proto = UsdGeom.Sphere.Define(stage, Sdf.Path("/World/heap/proto0"))
            proto.CreateRadiusAttr().Set(radius)
            proto.CreateExtentAttr().Set([(-radius,) * 3, (radius,) * 3])
            inst.GetPrototypesRel().AddTarget(Sdf.Path("/World/heap/proto0"))
            inst.GetProtoIndicesAttr().Set([0] * centres.shape[0])
            inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(centres.astype(np.float32)))

            tan_half = cfg.cam_aperture_mm / 2.0 / cfg.cam_focal_mm
            cam_z = cfg.cam_fov_margin * half_extent / tan_half
            cam = UsdGeom.Camera.Define(stage, Sdf.Path("/World/depthCam"))
            cam.CreateFocalLengthAttr().Set(cfg.cam_focal_mm)
            cam.CreateHorizontalApertureAttr().Set(cfg.cam_aperture_mm)
            cam.CreateVerticalApertureAttr().Set(cfg.cam_aperture_mm)
            cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, float(cam_z * 4.0)))
            UsdGeom.Xformable(cam.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(0.0, 0.0, float(cam_z))
            )
            fx = fy = cam_res * cfg.cam_focal_mm / cfg.cam_aperture_mm
            cx = cy = cam_res / 2.0

            rp = rep.create.render_product("/World/depthCam", (cam_res, cam_res))
            annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            annot.attach(rp)
            # There is no SimulationContext here (no physics), so the render graph
            # is driven directly by the replicator orchestrator rather than by
            # `world.render()` as in the P4 probe.
            depth = None
            for _ in range(4):
                app.update()
            for attempt in range(12):
                rep.orchestrator.step(rt_subframes=8)
                arr = np.asarray(annot.get_data(), dtype=np.float32)
                if arr.ndim == 2 and arr.shape == (cam_res, cam_res) and np.isfinite(arr).any():
                    depth = arr
                    break
            if depth is None:
                raise RuntimeError(
                    f"DEPTH_ANNOTATOR_FAIL {path.name} "
                    f"(last shape {np.asarray(annot.get_data()).shape})"
                )

            h_depth, axis, dmeta = p4.depth_to_height_field(
                depth, cam_z, fx, fy, cx, cy, half_extent, cell_m
            )
            h_analytic, _, _ = p3.surface_height_field(centres, radius, half_extent, cell_m)
            m_depth = p4.measure_from_height_field(h_depth, axis, cfg, radius, half_extent)

            a = heap["analytic"]
            diff = h_depth - h_analytic
            material = h_analytic > 0
            row = {
                "npz": path.name,
                "shape": heap["shape"],
                "mu": heap["mu"],
                "crr": heap["crr"],
                "angle_analytic_deg": a["repose_angle_deg"],
                "angle_depth_deg": m_depth["repose_angle_deg"],
                "angle_delta_deg": m_depth["repose_angle_deg"] - a["repose_angle_deg"],
                "r2_analytic": a["fit_r_squared"],
                "r2_depth": m_depth["fit_r_squared"],
                "apex_analytic_mm": a["apex_height_m"] * 1e3,
                "apex_depth_mm": m_depth["apex_height_m"] * 1e3,
                "toe_analytic_mm": a["toe_radius_m"] * 1e3,
                "toe_depth_mm": m_depth["toe_radius_m"] * 1e3,
                "height_rms_mm": float(np.sqrt((diff**2).mean()) * 1e3),
                "height_rms_on_material_mm": float(
                    np.sqrt((diff[material] ** 2).mean()) * 1e3
                ),
                "height_bias_on_material_mm": float(diff[material].mean() * 1e3),
                "height_max_abs_mm": float(np.abs(diff).max() * 1e3),
                "measurement_pass_depth": m_depth["measurement_pass"],
                "measurement_pass_analytic": a["measurement_pass"],
                "plateau_over_toe_depth": m_depth["plateau_over_toe_ratio"],
                "sphere_radius_mm": radius * 1e3,
                "camera_height_m": cam_z,
                "floor_mm_per_pixel": cam_z / fx * 1e3,
                "view_angle_at_toe_deg": math.degrees(
                    math.atan(a["toe_radius_m"] / cam_z)
                ),
                "reconstruction_delta_deg": heap["reconstruction_delta_deg"],
                "depth_meta": dmeta,
                "wall_s": time.time() - t0,
            }
            rows.append(row)
            np.savez_compressed(
                OUT_DIR / f"depth_{path.stem}.npz",
                height_depth_m=h_depth.astype(np.float32),
                height_analytic_m=h_analytic.astype(np.float32),
                grid_axis_m=axis,
                radial_profile_depth=m_depth["radial_profile"],
                row_json=np.array(json.dumps(row, default=str)),
            )
            print(
                f"[p4b] {path.stem:52s} analytic={row['angle_analytic_deg']:6.2f} "
                f"depth={row['angle_depth_deg']:6.2f} d={row['angle_delta_deg']:+6.2f} "
                f"rms={row['height_rms_on_material_mm']:5.2f}mm "
                f"bias={row['height_bias_on_material_mm']:+5.2f}mm ({row['wall_s']:.1f}s)",
                flush=True,
            )
            annot.detach(rp)
            rp.destroy()
    except BaseException as exc:  # noqa: BLE001
        import traceback

        print(traceback.format_exc(), flush=True)
        rc = 1
    finally:
        if rows:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "depth_on_deme_summary.json").write_text(
                json.dumps(
                    {
                        "artifact": ARTIFACT,
                        "non_claims": NON_CLAIMS,
                        "cam_res": cam_res,
                        "n_cells": len(rows),
                        "rows": rows,
                    },
                    indent=1,
                    default=str,
                )
            )
            print(f"wrote {OUT_DIR / 'depth_on_deme_summary.json'}", flush=True)
        app.close()
    return rc


def footprint_max_analytic(
    centres: np.ndarray, radius: float, half_extent: float, cell_m: float, refine: int
) -> np.ndarray:
    """The analytic surface sampled the way a CAMERA samples it.

    `sim_pellet_model.surface_height_field` evaluates the exact sphere surface at
    each cell CENTRE.  A depth camera instead returns ~40 pixels spread over the
    whole 2 mm cell, and `depth_to_height_field` keeps their MAX.  Those two are
    not the same operator: the max over a footprint is always >= the value at its
    centre, and for grains of comparable size to the cell the gap is millimetres.

    This computes the analytic surface on a `refine`x finer grid and max-pools it
    back, isolating that sampling difference from any genuine camera error.
    """
    fine = cell_m / refine
    h_fine, _, _ = p3.surface_height_field(centres, radius, half_extent, fine)
    n_coarse = int(math.ceil(2.0 * half_extent / cell_m))
    need = n_coarse * refine
    if h_fine.shape[0] < need:
        pad = need - h_fine.shape[0]
        h_fine = np.pad(h_fine, ((0, pad), (0, pad)))
    h_fine = h_fine[:need, :need]
    return h_fine.reshape(n_coarse, refine, n_coarse, refine).max(axis=(1, 3))


def bias_diagnosis(npz_paths: list[Path], cam_res: int, refine: int) -> int:
    """Split the depth-vs-analytic height offset into sampling vs camera."""
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    rc = 0
    out: list[dict[str, Any]] = []
    try:
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom, UsdLux, Vt

        for path in npz_paths:
            heap = load_heap(path)
            centres, radius = heap["centres"], heap["radius_m"]
            half_extent = heap["half_extent_m"]
            cfg = measurement_config(heap["meta"])
            cell_m = cfg.profile_cell_mm * 1.0e-3

            omni.usd.get_context().new_stage()
            stage = omni.usd.get_context().get_stage()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
            stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
            UsdLux.DistantLight.Define(stage, Sdf.Path("/World/light")).CreateIntensityAttr(3000.0)
            e = half_extent * 1.4
            floor = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/floor"))
            floor.CreatePointsAttr([(-e, -e, 0), (e, -e, 0), (e, e, 0), (-e, e, 0)])
            floor.CreateFaceVertexCountsAttr([4])
            floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            floor.CreateExtentAttr([(-e, -e, 0), (e, e, 0)])
            inst = UsdGeom.PointInstancer.Define(stage, Sdf.Path("/World/heap"))
            proto = UsdGeom.Sphere.Define(stage, Sdf.Path("/World/heap/proto0"))
            proto.CreateRadiusAttr().Set(radius)
            proto.CreateExtentAttr().Set([(-radius,) * 3, (radius,) * 3])
            inst.GetPrototypesRel().AddTarget(Sdf.Path("/World/heap/proto0"))
            inst.GetProtoIndicesAttr().Set([0] * centres.shape[0])
            inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(centres.astype(np.float32)))

            tan_half = cfg.cam_aperture_mm / 2.0 / cfg.cam_focal_mm
            cam_z = cfg.cam_fov_margin * half_extent / tan_half
            cam = UsdGeom.Camera.Define(stage, Sdf.Path("/World/depthCam"))
            cam.CreateFocalLengthAttr().Set(cfg.cam_focal_mm)
            cam.CreateHorizontalApertureAttr().Set(cfg.cam_aperture_mm)
            cam.CreateVerticalApertureAttr().Set(cfg.cam_aperture_mm)
            cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, float(cam_z * 4.0)))
            UsdGeom.Xformable(cam.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(0.0, 0.0, float(cam_z))
            )
            fx = fy = cam_res * cfg.cam_focal_mm / cfg.cam_aperture_mm
            cx = cy = cam_res / 2.0
            rp = rep.create.render_product("/World/depthCam", (cam_res, cam_res))
            annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            annot.attach(rp)
            depth = None
            for _ in range(4):
                app.update()
            for _ in range(12):
                rep.orchestrator.step(rt_subframes=8)
                arr = np.asarray(annot.get_data(), dtype=np.float32)
                if arr.ndim == 2 and np.isfinite(arr).any():
                    depth = arr
                    break
            h_depth, axis, _ = p4.depth_to_height_field(
                depth, cam_z, fx, fy, cx, cy, half_extent, cell_m
            )
            h_centre, _, _ = p3.surface_height_field(centres, radius, half_extent, cell_m)
            h_fpmax = footprint_max_analytic(centres, radius, half_extent, cell_m, refine)
            mask = h_centre > 0
            m_depth = p4.measure_from_height_field(h_depth, axis, cfg, radius, half_extent)
            m_fpmax = p4.measure_from_height_field(h_fpmax, axis, cfg, radius, half_extent)
            row = {
                "npz": path.name,
                "shape": heap["shape"],
                "angle_analytic_deg": heap["analytic"]["repose_angle_deg"],
                "angle_depth_deg": m_depth["repose_angle_deg"],
                "angle_footprint_max_deg": m_fpmax["repose_angle_deg"],
                "bias_depth_vs_centre_mm": float((h_depth - h_centre)[mask].mean() * 1e3),
                "bias_footprintmax_vs_centre_mm": float((h_fpmax - h_centre)[mask].mean() * 1e3),
                "bias_depth_vs_footprintmax_mm": float((h_depth - h_fpmax)[mask].mean() * 1e3),
                "rms_depth_vs_centre_mm": float(
                    np.sqrt(((h_depth - h_centre)[mask] ** 2).mean()) * 1e3
                ),
                "rms_depth_vs_footprintmax_mm": float(
                    np.sqrt(((h_depth - h_fpmax)[mask] ** 2).mean()) * 1e3
                ),
                "refine": refine,
                "cell_mm": cell_m * 1e3,
                "sphere_radius_mm": radius * 1e3,
            }
            out.append(row)
            print(
                f"[bias] {path.stem[:44]:44s} centre->fpmax {row['bias_footprintmax_vs_centre_mm']:+5.2f}mm "
                f"| fpmax->depth {row['bias_depth_vs_footprintmax_mm']:+5.2f}mm "
                f"| angle ana {row['angle_analytic_deg']:5.2f} fpmax {row['angle_footprint_max_deg']:5.2f} "
                f"depth {row['angle_depth_deg']:5.2f}",
                flush=True,
            )
            annot.detach(rp)
            rp.destroy()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "bias_diagnosis.json").write_text(
            json.dumps({"artifact": ARTIFACT + "_BIAS", "rows": out}, indent=1, default=str)
        )
        print(f"wrote {OUT_DIR / 'bias_diagnosis.json'}", flush=True)
    except BaseException as exc:  # noqa: BLE001
        import traceback

        print(traceback.format_exc(), flush=True)
        rc = 1
    finally:
        app.close()
    return rc


def figure() -> int:
    """One figure: is depth observation faithful, and where does the offset come from."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((OUT_DIR / "depth_on_deme_summary.json").read_text())["rows"]
    bias = {r["npz"]: r for r in json.loads((OUT_DIR / "bias_diagnosis.json").read_text())["rows"]}
    aa = np.array([r["angle_analytic_deg"] for r in rows])
    ad = np.array([r["angle_depth_deg"] for r in rows])
    af = np.array([bias[r["npz"]]["angle_footprint_max_deg"] for r in rows])
    samp = np.array([bias[r["npz"]]["bias_footprintmax_vs_centre_mm"] for r in rows])
    cam = np.array([bias[r["npz"]]["bias_depth_vs_footprintmax_mm"] for r in rows])
    shapes = [r["shape"] for r in rows]
    colour = {"sphere": "tab:blue", "clump2": "tab:orange", "clump3": "tab:green"}

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
    lim = [aa.min() - 2, aa.max() + 2]
    for sh in colour:
        m = [i for i, s in enumerate(shapes) if s == sh]
        ax[0].scatter(aa[m], ad[m], c=colour[sh], label=f"{sh} vs P3 centre-sample", s=34)
        ax[0].scatter(af[m], ad[m], c=colour[sh], marker="x", s=34,
                      label=f"{sh} vs footprint-max")
    ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    ax[0].set_xlabel("reference angle [deg]")
    ax[0].set_ylabel("depth-observed angle [deg]")
    ax[0].set_title("depth observation vs the two\nanalytic conventions", fontsize=10)
    ax[0].legend(fontsize=6.5, ncol=2)
    ax[0].grid(alpha=0.3)

    ax[1].scatter(aa, ad - aa, c=[colour[s] for s in shapes], s=34, label="vs P3 centre-sample")
    ax[1].scatter(aa, ad - af, c=[colour[s] for s in shapes], marker="x", s=34,
                  label="vs footprint-max (same sampling as a camera)")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].axhspan(-0.34, 0.34, alpha=0.15, color="tab:green",
                  label="+/-0.34 deg = worst case vs footprint-max")
    ax[1].set_xlabel("heap steepness (analytic angle) [deg]")
    ax[1].set_ylabel("angle error [deg]")
    ax[1].set_title("the 3.3 deg outlier is a SAMPLING CONVENTION,\nnot a camera error", fontsize=10)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    x = np.arange(len(rows))
    order = np.argsort(aa)
    ax[2].bar(x, samp[order], color="tab:red", label="grid sampling: cell centre -> cell max")
    ax[2].bar(x, cam[order], color="tab:blue", label="the CAMERA itself")
    ax[2].set_xlabel("cells, ordered by steepness")
    ax[2].set_ylabel("height offset vs P3 analytic [mm]")
    ax[2].set_title(
        f"offset split: sampling {samp.mean():+.2f} mm\nvs camera {cam.mean():+.3f} mm "
        f"(|max| {np.abs(cam).max():.3f} mm)", fontsize=10
    )
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, axis="y")
    fig.suptitle(
        "Depth-render observation applied to the 27 settled DEME heaps (P4b) - no physics, "
        "same sidewall_regression definition"
    )
    fig.tight_layout()
    out = OUT_DIR / "fig_depth_on_deme.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


def summarise() -> int:
    path = OUT_DIR / "depth_on_deme_summary.json"
    d = json.loads(path.read_text())
    rows = d["rows"]
    hdr = (
        f"{'shape':8s} {'mu':>5s} {'Crr':>5s} {'analytic':>9s} {'depth':>7s} {'delta':>7s} "
        f"{'r2_ana':>7s} {'r2_dep':>7s} {'rms_mm':>7s} {'bias_mm':>8s} {'apex_a':>7s} "
        f"{'apex_d':>7s} {'toe_a':>6s} {'toe_d':>6s} {'view_deg':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["angle_analytic_deg"]):
        print(
            f"{r['shape']:8s} {r['mu']:5.2f} {r['crr']:5.2f} {r['angle_analytic_deg']:9.2f} "
            f"{r['angle_depth_deg']:7.2f} {r['angle_delta_deg']:+7.2f} {r['r2_analytic']:7.3f} "
            f"{r['r2_depth']:7.3f} {r['height_rms_on_material_mm']:7.2f} "
            f"{r['height_bias_on_material_mm']:+8.2f} {r['apex_analytic_mm']:7.2f} "
            f"{r['apex_depth_mm']:7.2f} {r['toe_analytic_mm']:6.1f} {r['toe_depth_mm']:6.1f} "
            f"{r['view_angle_at_toe_deg']:8.2f}"
        )
    dl = np.array([r["angle_delta_deg"] for r in rows])
    ang = np.array([r["angle_analytic_deg"] for r in rows])
    bias = np.array([r["height_bias_on_material_mm"] for r in rows])
    print()
    print(f"n = {len(rows)} cells, analytic angle span {ang.min():.2f}-{ang.max():.2f} deg")
    print(
        f"angle delta (depth - analytic): mean {dl.mean():+.3f} deg, sd {dl.std(ddof=1):.3f}, "
        f"min {dl.min():+.3f}, max {dl.max():+.3f}, max|.| {np.abs(dl).max():.3f}"
    )
    print(
        f"height bias on material: mean {bias.mean():+.3f} mm, "
        f"max {bias.max():+.3f} mm (a depth camera sees the TOP of a grain, so a positive "
        f"bias is expected)"
    )
    if len(rows) >= 3:
        slope, intercept = np.polyfit(ang, dl, 1)
        resid = dl - (slope * ang + intercept)
        ss = 1.0 - (resid**2).sum() / ((dl - dl.mean()) ** 2).sum()
        print(
            f"delta vs steepness: delta = {slope:+.4f} * angle {intercept:+.3f}, R^2 = {ss:.3f} "
            f"-> {'steeper cones are biased more' if abs(slope) > 0.02 else 'no steepness trend'}"
        )
    n_pass = sum(1 for r in rows if r["measurement_pass_depth"])
    print(f"depth-side measurement_pass: {n_pass}/{len(rows)}")
    worst = max(rows, key=lambda r: abs(r["angle_delta_deg"]))
    print(f"worst cell: {worst['npz']} delta {worst['angle_delta_deg']:+.3f} deg")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="every P3 repose_*.npz")
    ap.add_argument("--npz", nargs="*", default=[])
    ap.add_argument("--cam-res", type=int, default=1536)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--bias-diagnosis", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--refine", type=int, default=8)
    args = ap.parse_args(argv)
    if args.figure:
        return figure()
    if args.summarise:
        return summarise()
    paths = [Path(p) for p in args.npz]
    if args.all:
        paths = sorted(P3_DIR.glob("repose_*_n1500_seed460.npz"))
    if not paths:
        ap.error("pass --all, --npz <files>, or --summarise")
    if args.bias_diagnosis:
        return bias_diagnosis(paths, args.cam_res, args.refine)
    return run(paths, args.cam_res)


if __name__ == "__main__":
    raise SystemExit(main())
