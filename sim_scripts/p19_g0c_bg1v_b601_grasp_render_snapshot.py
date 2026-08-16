#!/usr/bin/env python3
"""p19 / bg1v — g0c_d446 B601 grasp illustration renders (state restore, no physics).

User-approved visualization-only pass (60th session): render two RGB snapshots of
the bg1 variant-B SUCCESS states by restoring the exact post-CLOSE poses recorded
in bg1_results.json pose_snaps (rows 13 side_phi000 and 21 top_theta00).

This pass steps ZERO physics, re-adjudicates nothing, and edits no bg1 artifact.
Authority remains bg1_results.json + bg1_trace.npz (D446).  Single-frame
diagnostic snapshots per D324; no RRD (no new spatial/temporal judgment — the
renders re-project pinned recorded numbers verbatim).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0c_d446"
TAG = "bg1v"
LOG = "g0c_bg1v"

RESULTS = CASE_DIR / "bg1_results.json"
RESULTS_SHA16_PIN = "cb88c549dc459272"
RESULTS_BYTES_PIN = 88723
VARIANT_B_USD = CASE_DIR / "bg1_gripper_split2.usd"
SPLIT2_AUDIT = CASE_DIR / "bg1_split2_audit.json"
NUMPY_PIN = "1.26.0"
PSUTIL_PIN = "5.9.8"

ROOT = "/World/bg1_gripper"
GLF = ROOT + "/gripper_link"
BODY_L = ROOT + "/gripper_left"
BODY_R = ROOT + "/gripper_right"
OBJECT_PRIM = "/World/object"
SUPPORT_PRIM = "/World/support"
CAMERA_PRIM = "/World/camera"

OBJ_RADIUS_M = 0.0145
OBJ_HEIGHT_M = 0.050
OBJ_CENTER = np.array([0.4235072423787768, 0.17237803311822986, 0.025])

RES_W, RES_H = 1920, 1080
FOCAL_MM = 18.0
H_APERTURE_MM = 20.955
V_APERTURE_MM = H_APERTURE_MM * RES_H / RES_W
WARMUP_UPDATES = 80

# (row index in bg1_results.json, output stem, camera direction from subject)
SNAPSHOTS = [
    (13, "bg1v_side_phi000_postclose", np.array([0.35, -0.85, 0.40])),
    (21, "bg1v_top_theta00_postclose", np.array([0.75, -0.75, 0.45])),
]

OUT_PNG = {row: CASE_DIR / f"{stem}.png" for row, stem, _ in SNAPSHOTS}
OUT_META = CASE_DIR / f"{TAG}_render_meta.json"
OUT_SCRIPT = CASE_DIR / f"{TAG}_script.py.txt"
OUT_ARGV = CASE_DIR / f"{TAG}_argv.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fsync_write(path: Path, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def preflight() -> dict:
    import numpy as _np
    import psutil
    if _np.__version__ != NUMPY_PIN or psutil.__version__ != PSUTIL_PIN:
        raise RuntimeError(f"ENV_PIN numpy={_np.__version__} psutil={psutil.__version__}")

    got = sha256(RESULTS)
    size = RESULTS.stat().st_size
    if got[:16] != RESULTS_SHA16_PIN or size != RESULTS_BYTES_PIN:
        raise RuntimeError(f"RESULTS_PIN_DRIFT sha16={got[:16]} bytes={size}")

    audit_b = json.loads(SPLIT2_AUDIT.read_text())
    usd_sha = sha256(VARIANT_B_USD)
    if usd_sha != audit_b["out"]["sha256"]:
        raise RuntimeError(f"VARIANT_B_SHA_DRIFT {usd_sha}")

    existing = [p.name for p in (*OUT_PNG.values(), OUT_META) if p.exists()]
    if existing:
        raise RuntimeError(f"WRITE_GUARD existing={existing}")

    results = json.loads(RESULTS.read_text())
    snaps = {}
    for row, stem, _ in SNAPSHOTS:
        s = results["pose_snaps"][row]
        r = results["rows"][row]
        if not (s["variant"] == "B" == r["variant"] and r["success"] is True
                and r["index"] == row):
            raise RuntimeError(f"ROW_SELECT_FAIL row={row} {r['variant']} {r['success']}")
        snaps[row] = {"snap": s, "row": r}
    return {"results_sha256": got, "usd_sha256": usd_sha, "snaps": snaps}


def run(pre: dict) -> int:
    t_start = time.time()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    rc = 1
    try:
        rc = run_inner(pre, app, t_start)
    except BaseException as exc:  # noqa: BLE001 - loud failure marker is the contract
        import traceback
        print(f"[{LOG}] FAILURE {exc!r}\n{traceback.format_exc()}", flush=True)
        rc = 3
    finally:
        print(f"[{LOG}] rc={rc} wall={time.time() - t_start:.1f}s", flush=True)
        sys.stdout.flush()
        app.close()
    return rc


def run_inner(pre: dict, app, t_start: float) -> int:
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux
    from isaacsim.core.utils.extensions import enable_extension
    print(f"[{LOG}] imports base OK", flush=True)
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep
    from PIL import Image
    print(f"[{LOG}] replicator + PIL OK", flush=True)

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

    sup = UsdGeom.Cube.Define(stage, SUPPORT_PRIM)
    sup.CreateSizeAttr().Set(1.0)
    sup.AddTranslateOp().Set(Gf.Vec3d(OBJ_CENTER[0], OBJ_CENTER[1], -0.5))
    sup.AddScaleOp().Set(Gf.Vec3f(0.1, 0.1, 1.0))
    sup.CreateDisplayColorAttr([(0.42, 0.43, 0.46)])

    cyl = UsdGeom.Cylinder.Define(stage, OBJECT_PRIM)
    cyl.CreateRadiusAttr().Set(OBJ_RADIUS_M)
    cyl.CreateHeightAttr().Set(OBJ_HEIGHT_M)
    cyl.CreateAxisAttr().Set("Z")
    cyl.CreateExtentAttr().Set([(-OBJ_RADIUS_M, -OBJ_RADIUS_M, -OBJ_HEIGHT_M / 2),
                                (OBJ_RADIUS_M, OBJ_RADIUS_M, OBJ_HEIGHT_M / 2)])
    obj_translate = cyl.AddTranslateOp()
    obj_orient = cyl.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    cyl.CreateDisplayColorAttr([(0.85, 0.30, 0.25)])

    grip = stage.DefinePrim(ROOT)
    grip.GetReferences().AddReference(str(VARIANT_B_USD), "/bg1_gripper")
    print(f"[{LOG}] stage authored + gripper referenced", flush=True)

    # hide collision/split2 meshes so only visual geometry renders
    hidden = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT),
                              Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        name = prim.GetName().lower()
        if "collision" in name or "split2" in name:
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(prim.GetPath().pathString)

    dome = UsdLux.DomeLight.Define(stage, "/World/lights/dome")
    dome.CreateIntensityAttr().Set(700.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/lights/sun")
    sun.CreateIntensityAttr().Set(2500.0)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 20.0, 0.0))

    cam = UsdGeom.Camera.Define(stage, CAMERA_PRIM)
    cam.CreateFocalLengthAttr().Set(FOCAL_MM)
    cam.CreateHorizontalApertureAttr().Set(H_APERTURE_MM)
    cam.CreateVerticalApertureAttr().Set(V_APERTURE_MM)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.005, 10.0))
    cam_op = cam.AddTransformOp()

    def world_matrix(pos, quat_wxyz) -> "Gf.Matrix4d":
        w, x, y, z = quat_wxyz
        m = Gf.Matrix4d(1.0)
        m.SetRotate(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
        m.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in pos]))
        return m

    # world-absolute overrides for the three recorded bodies: op order must be
    # exactly [my op] with resetXformStack, or the referenced rig's own xform
    # ops keep composing on top (observed dev ~1.0 without this)
    body_ops = {}
    for path in (GLF, BODY_L, BODY_R):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "bg1v")
        if not xf.SetXformOpOrder([op], resetXformStack=True):
            raise RuntimeError(f"XFORM_ORDER_FAIL {path}")
        body_ops[path] = op

    def pose_bodies(pc: dict) -> float:
        body_ops[GLF].Set(world_matrix(pc["palm_pos"], pc["palm_quat_wxyz"]))
        body_ops[BODY_L].Set(world_matrix(pc["left_pos"], pc["left_quat_wxyz"]))
        body_ops[BODY_R].Set(world_matrix(pc["right_pos"], pc["right_quat_wxyz"]))
        obj_translate.Set(Gf.Vec3d(*pc["obj_pos"]))
        q = pc["obj_quat_wxyz"]
        obj_orient.Set(Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3])))
        # verification gate: composed world transform == recorded, per body
        dev = 0.0
        for path, key in ((GLF, "palm"), (BODY_L, "left"), (BODY_R, "right")):
            want = world_matrix(pc[f"{key}_pos"], pc[f"{key}_quat_wxyz"])
            got = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
            delta = np.array([[got[i][j] - want[i][j] for j in range(4)] for i in range(4)])
            dev = max(dev, float(np.abs(delta).max()))
        return dev

    def frame_camera(direction: np.ndarray) -> dict:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeWorldBound(stage.GetPrimAtPath(ROOT)).ComputeAlignedRange()
        rng.UnionWith(cache.ComputeWorldBound(cyl.GetPrim()).ComputeAlignedRange())
        lo = np.array(rng.GetMin())
        hi = np.array(rng.GetMax())
        center = (lo + hi) / 2.0
        radius = float(np.linalg.norm(hi - lo)) / 2.0
        if not (0.02 < radius < 1.0):
            raise RuntimeError(f"BBOX_RADIUS_IMPLAUSIBLE {radius}")
        fovx = 2.0 * math.atan(H_APERTURE_MM / 2.0 / FOCAL_MM)
        fovy = 2.0 * math.atan(V_APERTURE_MM / 2.0 / FOCAL_MM)
        dist = max(radius / math.tan(fovx / 2.0), radius / math.tan(fovy / 2.0)) * 1.30
        d = direction / np.linalg.norm(direction)
        eye = center + d * dist
        z_cam = d
        x_cam = np.cross(np.array([0.0, 0.0, 1.0]), z_cam)
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        m = Gf.Matrix4d(
            x_cam[0], x_cam[1], x_cam[2], 0.0,
            y_cam[0], y_cam[1], y_cam[2], 0.0,
            z_cam[0], z_cam[1], z_cam[2], 0.0,
            eye[0], eye[1], eye[2], 1.0)
        cam_op.Set(m)
        return {"eye": eye.tolist(), "look_center": center.tolist(),
                "bbox_radius_m": radius, "dist_m": dist}

    print(f"[{LOG}] hidden={len(hidden)} lights+camera authored", flush=True)
    rp = rep.create.render_product(CAMERA_PRIM, (RES_W, RES_H))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach(rp)
    print(f"[{LOG}] render product + rgb annotator attached", flush=True)

    meta = {"tool": TAG, "case": "g0c_d446",
            "purpose": "illustration renders of bg1 variant-B SUCCESS states "
                       "(user-approved 60th session); authority remains "
                       "bg1_results.json + bg1_trace.npz (D446)",
            "physics_steps": 0,
            "source_pins": {"bg1_results.json": pre["results_sha256"],
                            "bg1_gripper_split2.usd": pre["usd_sha256"]},
            "hidden_prims_n": len(hidden),
            "camera": {"resolution": [RES_W, RES_H], "focal_mm": FOCAL_MM,
                       "h_aperture_mm": H_APERTURE_MM, "v_aperture_mm": V_APERTURE_MM},
            "snapshots": []}

    for row, stem, direction in SNAPSHOTS:
        snap = pre["snaps"][row]["snap"]
        rrow = pre["snaps"][row]["row"]
        pc = snap["post_close"]
        dev = pose_bodies(pc)
        if dev > 1e-9:
            raise RuntimeError(f"POSE_RESTORE_GATE row={row} dev={dev}")
        framing = frame_camera(direction)
        for _ in range(WARMUP_UPDATES):
            app.update()
        rep.orchestrator.step(rt_subframes=32)
        data = annot.get_data()
        img = np.asarray(data)
        if img.ndim != 3 or img.shape[0] != RES_H or img.shape[1] != RES_W:
            raise RuntimeError(f"IMAGE_SHAPE {img.shape}")
        rgb = img[:, :, :3].astype(np.uint8)
        std = float(rgb.std())
        mean = float(rgb.mean())
        if std < 5.0 or not (5.0 < mean < 250.0):
            raise RuntimeError(f"IMAGE_FLAT row={row} std={std} mean={mean}")
        out_path = OUT_PNG[row]
        Image.fromarray(rgb).save(out_path)
        entry = {"row": row, "label": rrow["label"], "file": out_path.name,
                 "restored_state": "post_close (verbatim from pose_snaps)",
                 "pose_restore_max_dev": dev, "framing": framing,
                 "image_std": std, "image_mean": mean,
                 "recorded": {"post_close": pc, "hang": snap["hang"],
                              "close_bilateral_peak_n": rrow["close_bilateral_peak_n"],
                              "hang_drop_mm": rrow["hang_drop_m"] * 1000.0},
                 "png_sha256_16": sha256(out_path)[:16],
                 "png_bytes": out_path.stat().st_size}
        meta["snapshots"].append(entry)
        print(f"[{LOG}] row {row} {rrow['label']}: saved {out_path.name} "
              f"std={std:.1f} mean={mean:.1f} dev={dev:.2e}", flush=True)

    # end-of-run pin recheck: this pass must not have touched its sources
    if sha256(RESULTS)[:16] != RESULTS_SHA16_PIN:
        raise RuntimeError("END_PIN_DRIFT results.json")
    if sha256(VARIANT_B_USD) != pre["usd_sha256"]:
        raise RuntimeError("END_PIN_DRIFT variant B usd")
    meta["end_pin_recheck"] = True
    meta["wall_seconds"] = round(time.time() - t_start, 1)
    fsync_write(OUT_META, json.dumps(meta, indent=2) + "\n")
    print(f"[{LOG}] meta={sha256(OUT_META)[:16]} wall={meta['wall_seconds']}s", flush=True)
    return 0


def main() -> int:
    pre = preflight()
    fsync_write(OUT_ARGV, " ".join([sys.executable, *sys.argv]) + "\n")
    shutil.copyfile(__file__, OUT_SCRIPT)
    print(f"[{LOG}] preflight OK: rows 13/21 variant B SUCCESS, pins verified", flush=True)
    return run(pre)


if __name__ == "__main__":
    sys.exit(main())
