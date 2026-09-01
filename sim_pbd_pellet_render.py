#!/usr/bin/env python3
"""Render the SETTLED PILES that the P4 probe measured, so they can be LOOKED AT.

The probe itself only ever consumed `distance_to_image_plane` depth arrays.  That
is a real RTX render, but a depth array is not something a human has inspected.
This script closes that gap (D324 visualization definition of done): it rebuilds
each pile as static geometry from the particle coordinates stored in the cell
NPZ and renders RGB from a side-on and an oblique view.

It runs NO physics.  The positions come verbatim from `positions_settled_m`, so
the pictures show the exact configuration the reported angle was measured on -
they cannot drift from it.

Usage:
  ~/miniconda3/envs/isaaclab/bin/python sim_pbd_pellet_render.py <label> [<label> ...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("claudedocs/runtime_logs/pbd_probe")
RES_W, RES_H = 1600, 900
FOCAL_MM = 35.0
H_APER_MM = 20.955


def main(labels: list[str]) -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    rc = 0
    try:
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom, UsdLux, Vt
        from PIL import Image

        for label in labels:
            npz = np.load(OUT_DIR / f"cell_{label}.npz", allow_pickle=True)
            meta = json.loads(str(npz["metadata_json"]))
            pos = npz["positions_settled_m"].astype(np.float64)
            released = npz["released_mask"]
            radius = float(meta["derived"]["sphere_radius_m"])
            half = float(meta["derived"]["half_extent_m"])
            angle = meta["measurement_depth"]["repose_angle_deg"]
            apex = meta["measurement_depth"]["apex_height_m"]
            hz = meta["config"]["time_steps_per_second"]

            # only the measured heap: released, inside the profile grid
            keep = released & (np.abs(pos[:, 0]) <= half) & (np.abs(pos[:, 1]) <= half)
            p = pos[keep]

            omni.usd.get_context().new_stage()
            stage = omni.usd.get_context().get_stage()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
            stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

            key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/key"))
            key.CreateIntensityAttr(2600.0)
            key.CreateAngleAttr(2.0)
            UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 25.0))
            dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/dome"))
            dome.CreateIntensityAttr(500.0)

            floor = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/floor"))
            e = half * 1.6
            floor.CreatePointsAttr([(-e, -e, 0), (e, -e, 0), (e, e, 0), (-e, e, 0)])
            floor.CreateFaceVertexCountsAttr([4])
            floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            floor.CreateExtentAttr([(-e, -e, 0), (e, e, 0)])
            floor.CreateDisplayColorAttr([(0.22, 0.24, 0.27)])

            inst = UsdGeom.PointInstancer.Define(stage, Sdf.Path("/World/pile"))
            proto = UsdGeom.Sphere.Define(stage, Sdf.Path("/World/pile/proto0"))
            proto.CreateRadiusAttr().Set(radius)
            proto.CreateExtentAttr().Set([(-radius,) * 3, (radius,) * 3])
            proto.CreateDisplayColorAttr([(0.85, 0.72, 0.35)])
            inst.GetPrototypesRel().AddTarget(Sdf.Path("/World/pile/proto0"))
            inst.GetProtoIndicesAttr().Set([0] * p.shape[0])
            inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))

            cam = UsdGeom.Camera.Define(stage, Sdf.Path("/World/cam"))
            cam.CreateFocalLengthAttr().Set(FOCAL_MM)
            cam.CreateHorizontalApertureAttr().Set(H_APER_MM)
            cam.CreateVerticalApertureAttr().Set(H_APER_MM * RES_H / RES_W)
            cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 20.0))
            cam_op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()

            rp = rep.create.render_product("/World/cam", (RES_W, RES_H))
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach(rp)

            def look(eye: np.ndarray, target: np.ndarray) -> None:
                d = eye - target
                z = d / np.linalg.norm(d)
                up = np.array([0.0, 0.0, 1.0])
                x = np.cross(up, z)
                x = x / np.linalg.norm(x)
                y = np.cross(z, x)
                cam_op.Set(
                    Gf.Matrix4d(
                        x[0], x[1], x[2], 0.0,
                        y[0], y[1], y[2], 0.0,
                        z[0], z[1], z[2], 0.0,
                        eye[0], eye[1], eye[2], 1.0,
                    )
                )

            centre = np.array([0.0, 0.0, apex * 0.45])
            views = {
                # side-on at pile height: this is the view the angle lives in
                "side": centre + np.array([0.0, -half * 3.1, apex * 0.9]),
                "oblique": centre + np.array([half * 2.0, -half * 2.0, half * 1.15]),
            }
            for name, eye in views.items():
                look(eye, centre)
                for _ in range(4):
                    app.update()
                rep.orchestrator.step(rt_subframes=24)
                img = np.asarray(annot.get_data())
                if img.ndim != 3:
                    raise RuntimeError(f"RGB_SHAPE {img.shape}")
                rgb = img[:, :, :3].astype(np.uint8)
                if rgb.std() < 3.0:
                    raise RuntimeError(f"IMAGE_FLAT {label}/{name} std={rgb.std():.2f}")
                out = OUT_DIR / f"render_{label}_{name}.png"
                Image.fromarray(rgb).save(out)
                print(
                    f"wrote {out}  n={p.shape[0]} particles  {hz} Hz  "
                    f"angle={angle:.2f} deg  apex={apex*1e3:.1f} mm  "
                    f"std={rgb.std():.1f}",
                    flush=True,
                )
            annot.detach(rp)
            rp.destroy()
    except BaseException as exc:  # noqa: BLE001
        import traceback

        print(traceback.format_exc(), flush=True)
        rc = 1
    finally:
        app.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["ctrl_solid_ref"]))
