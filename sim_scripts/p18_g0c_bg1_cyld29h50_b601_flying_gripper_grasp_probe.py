#!/usr/bin/env python3
"""p18 / bg1 — g0c_d446 reBot B601 flying-gripper grasp probe (2 collision variants).

Contract: claudedocs/runtime_logs/grasp_track/g0c_d446/bg1_prereg.md (59th session,
user-approved case).  New variables (exactly two): gripper model = B601 parallel
gripper (arm removed, fixed-root rig), and collision representation
A (official USD single convex hull per finger, verbatim) vs
B (blade/mount 2-piece split of the same official collision points).

13 analytic poses (8 side azimuths + 5 top tilts 0/6/15/24/35 deg) x 2 variants.
Phases per pose: PREGRASP (open 0.0715 m both, 60 steps) -> CLOSE (0.0 m both,
120 steps) via GraspingManager.evaluate_grasp_poses (default scene), then a HANG
test outside the manager (pedestal collider disabled, 240 steps in 30-step
chunks).  SUCCESS = same-step bilateral close force > 0.01 N AND hang drop < 6 mm.

Authoring revisions R1-R4 and harness settings (sleepThreshold 0) are documented
in the prereg; asset gates re-verify the extracted variants against the pinned
official source on the composed run stage at run time.
"""
from __future__ import annotations

import asyncio
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
sys.path.insert(0, str(REPO))

CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0c_d446"
ASSET_DIR = CASE_DIR / "b601_asset"
TAG = "bg1"
LOG = "g0c_bg1"

SRC_PINS = {
    ASSET_DIR / "reBot_B601_DM.usda":
        "6b9d39de1200732c581c91e895bee412844e101006fb0c3df54259d81ee28e84",
    ASSET_DIR / "payloads/base.usda":
        "e8a217cb3cfe56129b25e00c8f2171e9ba0f5c6145651f4340dba5707999bdc9",
    ASSET_DIR / "payloads/geometries.usd":
        "4ead3b7d29627101085634014893b680ad148c7e87b33325bc8a525ba836ace6",
    ASSET_DIR / "payloads/instances.usda":
        "b2209f637eec1831a59e95e862768313aaeae77ba460eabf2ec9ddb3143833d6",
    ASSET_DIR / "payloads/materials.usda":
        "0f5b1bb484ce696b4dc987321bf267c8e5b484417ba53344df912f55fe537b05",
    ASSET_DIR / "payloads/robot.usda":
        "497e66972d6f4bdfca9dc3592601d9843a125de8d654f646c97c38b0f298102b",
    ASSET_DIR / "payloads/Physics/mujoco.usda":
        "ecff1ef3aa0e7daa7e8402bfecec7bd21fd7601b53351eee971e64a57ebaf134",
    ASSET_DIR / "payloads/Physics/physics.usda":
        "131e9e667403adf10bfdd641ebbb66ab49af55befa0d598f6efeefcfec8af4a2",
    ASSET_DIR / "payloads/Physics/physx.usda":
        "58038daab9219f0a7809a868fe3ce3f491f0387f9e41272ab6fa6f8211e4048f",
}
EXT_MANIFEST_SHA256 = "5e599aafec0d1c66776c70318535faeffc539e66070d64bf5ca15f6c5e21393a"
NUMPY_PIN = "1.26.0"
PSUTIL_PIN = "5.9.8"
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"

OBJ_RADIUS_M = 0.0145
OBJ_HEIGHT_M = 0.050
OBJ_MASS_KG = 0.02483
OBJ_CENTER = np.array([0.4235072423787768, 0.17237803311822986, 0.025])
OBJ_TOP_Z = 0.05
STATIC_FRICTION = 0.40
DYNAMIC_FRICTION = 0.30
RESTITUTION = 0.0
SUPPORT_FRICTION = 1.0
GRAVITY = 9.81
DT = 1.0 / 60.0
PREGRASP_STEPS = 60
CLOSE_STEPS = 120
HANG_STEPS = 240
HANG_CHUNK = 30
OPEN_M = 0.0715
CLOSE_M = 0.0
BILATERAL_GATE_N = 0.01
HOLD_DROP_GATE_M = 0.006
X_TCP = -0.02448          # blade-span midpoint in glf frame (prereg SS4)
BITE_M = 0.012
TOP_THETAS = (0.0, 6.0, 15.0, 24.0, 35.0)
Q_LIMITS_M = (0.0, 0.0715)

VARIANTS = ("A", "B")
VARIANT_USD = {"A": "bg1_gripper_only.usd", "B": "bg1_gripper_split2.usd"}
VARIANT_CENSUS = {"A": {"gripper_link": (1, 0), "gripper_left": (1, 0),
                        "gripper_right": (1, 0)},
                  "B": {"gripper_link": (1, 0), "gripper_left": (2, 1),
                        "gripper_right": (2, 1)}}

ROOT = "/World/bg1_gripper"
GLF = ROOT + "/gripper_link"
BODY_L = ROOT + "/gripper_left"
BODY_R = ROOT + "/gripper_right"
J1 = ROOT + "/joints/gripper_joint1"
J2 = ROOT + "/joints/gripper_joint2"
OBJECT_PRIM = "/World/object"
SUPPORT_PRIM = "/World/support"

SRC_ROBOT = "/reBot_B601_DM"
SRC_GLF = SRC_ROBOT + "/Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_link"
SRC_L = SRC_GLF + "/gripper_left"
SRC_R = SRC_GLF + "/gripper_right"
SRC_J1 = SRC_ROBOT + "/Physics/gripper_joint1"
SRC_J2 = SRC_ROBOT + "/Physics/gripper_joint2"

JOINT_ATTRS = [
    "physics:axis", "physics:lowerLimit", "physics:upperLimit",
    "physics:localPos0", "physics:localRot0", "physics:localPos1",
    "physics:localRot1", "drive:linear:physics:maxForce",
    "physxJoint:maxJointVelocity", "urdf:limit:effort", "urdf:limit:velocity",
]
MASS_ATTRS = ["physics:mass", "physics:centerOfMass", "physics:diagonalInertia",
              "physics:principalAxes"]

OUT = {k: CASE_DIR / f"{TAG}_{k}" for k in (
    "results.json", "trace.npz", "timeline.rrd", "timeline.rbl", "rerun_validation.json",
    "inspection.png", "stdout.log", "exit_status.txt", "script.py.txt", "argv.txt",
    "failure.json")}

N_POSES = 13
N_ROWS = N_POSES * len(VARIANTS)


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


def quat_from_R(R: np.ndarray) -> np.ndarray:
    """Shepperd branch method — robust for 180-deg rotations (side phi=0)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] >= R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    # round-trip gate: rebuilt R must match
    w, x, y, z = q
    Rr = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    if float(np.abs(Rr - R).max()) > 1e-9:
        raise RuntimeError("QUAT_ROUNDTRIP_FAIL")
    return q


def build_poses() -> dict:
    """Analytic 13-pose construction with fatal verification gates.  No Isaac."""
    plan = {"poses": [], "gates": {}}
    gate = 0.0
    for k in range(8):
        phi = math.radians(45.0 * k)
        x = np.array([-math.cos(phi), -math.sin(phi), 0.0])
        y = np.array([math.sin(phi), -math.cos(phi), 0.0])
        z = np.cross(x, y)
        R = np.column_stack([x, y, z])
        t = OBJ_CENTER - R @ np.array([X_TCP, 0.0, 0.0])
        gate = max(gate,
                   float(np.abs(R @ R.T - np.eye(3)).max()),
                   abs(np.linalg.det(R) - 1.0),
                   float(np.abs(R @ np.array([X_TCP, 0, 0]) + t - OBJ_CENTER).max()),
                   float(np.abs(z - np.array([0.0, 0.0, 1.0])).max()))
        plan["poses"].append({
            "kind": "side", "label": f"side_phi{45 * k:03d}",
            "pos": t.tolist(), "quat_wxyz": quat_from_R(R).tolist(),
            "phi_deg": 45.0 * k,
            "source": "analytic: approach horizontal inward, aperture horizontal, "
                      "TCP at cylinder mid-height (prereg SS4)"})
    for th_deg in TOP_THETAS:
        th = math.radians(th_deg)
        x = np.array([math.sin(th), 0.0, -math.cos(th)])
        y = np.array([0.0, 1.0, 0.0])
        z = np.cross(x, y)
        R = np.column_stack([x, y, z])
        t = np.array([OBJ_CENTER[0], OBJ_CENTER[1], OBJ_TOP_Z - BITE_M])
        gate = max(gate,
                   float(np.abs(R @ R.T - np.eye(3)).max()),
                   abs(np.linalg.det(R) - 1.0),
                   float(np.abs(R @ np.array([1.0, 0, 0]) - x).max()))
        plan["poses"].append({
            "kind": "top", "label": f"top_theta{int(th_deg):02d}",
            "pos": t.tolist(), "quat_wxyz": quat_from_R(R).tolist(),
            "theta_deg": th_deg,
            "source": "analytic: fingertip plane at z_top - BITE, tilt about +x, "
                      "aperture along world y (prereg SS4)"})
    if gate > 1e-12:
        raise RuntimeError(f"POSE_CONSTRUCTION_GATE {gate}")
    plan["gates"]["pose_construction_max_err"] = gate
    if len(plan["poses"]) != N_POSES:
        raise RuntimeError("POSE_COUNT != 13")
    for p in plan["poses"]:
        for v in (OPEN_M, CLOSE_M):
            if not (Q_LIMITS_M[0] <= v <= Q_LIMITS_M[1]):
                raise RuntimeError(f"TARGET_OUT_OF_LIMITS {p['label']} {v}")
    return plan


def preflight() -> dict:
    import numpy
    import psutil
    if numpy.__version__ != NUMPY_PIN or psutil.__version__ != PSUTIL_PIN:
        raise RuntimeError(f"ENV_PIN numpy={numpy.__version__} psutil={psutil.__version__}")
    pins = {}
    for path, want in SRC_PINS.items():
        got = sha256(path)
        pins[str(path.relative_to(CASE_DIR))] = {"sha256": got, "match": got == want}
        if got != want:
            raise RuntimeError(f"SHA_DRIFT {path}")
    ext = Path("/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
               "exts/isaacsim.replicator.grasping/config/extension.toml")
    if sha256(ext) != EXT_MANIFEST_SHA256:
        raise RuntimeError("EXT_MANIFEST_DRIFT")
    pins["extension.toml"] = {"sha256": EXT_MANIFEST_SHA256, "match": True}

    audit_a = json.loads((CASE_DIR / "bg1_asset_audit.json").read_text())
    audit_b = json.loads((CASE_DIR / "bg1_split2_audit.json").read_text())
    variant_sha = {"A": audit_a["extraction"]["sha256"], "B": audit_b["out"]["sha256"]}
    for v in VARIANTS:
        p = CASE_DIR / VARIANT_USD[v]
        got = sha256(p)
        pins[VARIANT_USD[v]] = {"sha256": got, "match": got == variant_sha[v]}
        if got != variant_sha[v]:
            raise RuntimeError(f"VARIANT_SHA_DRIFT {v} {got}")
    blade_extreme = {
        "gripper_left": audit_b["gates"]["gripper_left"]["blade_inner_extreme_glf_m"],
        "gripper_right": audit_b["gates"]["gripper_right"]["blade_inner_extreme_glf_m"]}

    existing = [p.name for k, p in OUT.items()
                if p.exists() and k not in ("script.py.txt", "argv.txt", "stdout.log")]
    if existing:
        raise RuntimeError(f"WRITE_GUARD existing={existing}")

    plan = build_poses()
    plan["pins"] = pins
    plan["blade_extreme_pins"] = blade_extreme
    plan["env"] = {"numpy": numpy.__version__, "psutil": psutil.__version__,
                   "python": sys.version.split()[0]}
    return plan


# --------------------------------------------------------------------------- #
def run(plan: dict) -> int:
    t_start = time.time()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    rc = 1
    try:
        rc = run_inner(plan, t_start)
    except BaseException as exc:  # noqa: BLE001 - failure marker is the contract
        import traceback
        fsync_write(OUT["failure.json"], json.dumps(
            {"tag": TAG, "error": repr(exc), "traceback": traceback.format_exc(),
             "wall_seconds": round(time.time() - t_start, 1)}, indent=1))
        rc = 3
    finally:
        sentinel = f"PRE_CLOSE_SENTINEL rc={rc} tag={TAG} wall={time.time() - t_start:.1f}s\n"
        fsync_write(OUT["exit_status.txt"], sentinel)
        print(f"[{LOG}] {sentinel.strip()}", flush=True)
        sys.stdout.flush()
        app.close()
    return rc


def run_inner(plan: dict, t_start: float) -> int:
    import omni.kit.app
    import omni.physx
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.replicator.grasping")
    from isaacsim.replicator.grasping.grasping_manager import GraspPhase, GraspingManager
    import isaacsim.replicator.grasping.grasping_utils as grasping_utils

    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        raise RuntimeError(f"RERUN_VERSION {rr.__version__} != {RERUN_VERSION}")

    out: dict = {"tool": TAG, "case": "g0c_d446", "prereg": "bg1_prereg.md",
                 "plan": {k: plan[k] for k in ("gates", "pins", "blade_extreme_pins", "env")},
                 "poses_in": plan["poses"], "variants": list(VARIANTS)}

    pred_prox = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)

    # source stage (shared by both variants' gates)
    src = Usd.Stage.Open(str(ASSET_DIR / "reBot_B601_DM.usda"))
    SRC_BODIES = {"gripper_link": SRC_GLF, "gripper_left": SRC_L, "gripper_right": SRC_R}
    SRC_CHILD = {SRC_GLF: (SRC_L, SRC_R), SRC_L: (), SRC_R: ()}
    RUN_BODIES = {"gripper_link": GLF, "gripper_left": BODY_L, "gripper_right": BODY_R}

    def body_prims(st, body_path, child_map, pred):
        excl = child_map.get(body_path, ())
        outp = []
        for prim in Usd.PrimRange(st.GetPrimAtPath(body_path), pred):
            ps = prim.GetPath().pathString
            if any(ps == e or ps.startswith(e + "/") for e in excl):
                continue
            outp.append(prim)
        return outp

    def census(st, body_path, child_map, pred):
        en = dis = 0
        for prim in body_prims(st, body_path, child_map, pred):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                a = prim.GetAttribute("physics:collisionEnabled")
                if a.Get() if (a and a.Get() is not None) else True:
                    en += 1
                else:
                    dis += 1
        return en, dis

    def mesh_hashes(st, body_path, child_map, pred, skip_substr=None):
        rows = {}
        for prim in body_prims(st, body_path, child_map, pred):
            ps = prim.GetPath().pathString
            if skip_substr and skip_substr in ps:
                continue
            if prim.IsA(UsdGeom.Mesh):
                m = UsdGeom.Mesh(prim)
                h = hashlib.sha256()
                for attr, dtp in ((m.GetPointsAttr(), np.float64),
                                  (m.GetFaceVertexCountsAttr(), np.int64),
                                  (m.GetFaceVertexIndicesAttr(), np.int64)):
                    v = attr.Get()
                    h.update((np.array(v, dtype=dtp) if v is not None
                              else np.zeros(0, dtype=dtp)).tobytes())
                rows[ps[len(body_path):]] = h.hexdigest()
        return rows

    def attr_snap(st, path, names):
        prim = st.GetPrimAtPath(path)
        return {n: repr(prim.GetAttribute(n).Get()) if prim.GetAttribute(n)
                else "<none>" for n in names}

    src_census = {k: census(src, p, SRC_CHILD, pred_prox) for k, p in SRC_BODIES.items()}
    if any(v != (1, 0) for v in src_census.values()):
        raise RuntimeError(f"SRC_CENSUS_UNEXPECTED {src_census}")
    src_joints = {"j1": attr_snap(src, SRC_J1, JOINT_ATTRS),
                  "j2": attr_snap(src, SRC_J2, JOINT_ATTRS)}
    src_mesh = {k: mesh_hashes(src, p, SRC_CHILD, pred_prox)
                for k, p in SRC_BODIES.items()}
    src_mass = {k: attr_snap(src, p, MASS_ATTRS) for k, p in SRC_BODIES.items()}

    ctx = omni.usd.get_context()
    physx_iface = omni.physx.get_physx_interface()
    physx_sim = omni.physx.get_physx_simulation_interface()

    # instrumentation over all rows
    pre_F = np.zeros((N_ROWS, PREGRASP_STEPS, 2, 3))
    close_F = np.zeros((N_ROWS, CLOSE_STEPS, 2, 3))
    palm_F_close = np.zeros((N_ROWS, CLOSE_STEPS, 3))
    support_F_close = np.zeros((N_ROWS, CLOSE_STEPS, 3))
    hang_z = np.full((N_ROWS, HANG_STEPS // HANG_CHUNK + 1), np.nan)
    hang_q = np.full((N_ROWS, HANG_STEPS // HANG_CHUNK + 1, 2), np.nan)
    spawn_check = np.full((N_ROWS, 3), np.nan)
    raw_events = {"n_total": 0, "n_unknown_actor": 0, "pairs_seen": set()}
    pose_snaps: list = [None] * N_ROWS
    asset_gate_rows = {}
    jaw_pts_cache = {}

    verdict_rows: list = []

    for vi, variant in enumerate(VARIANTS):
        ctx.new_stage()
        stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
        scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr().Set(GRAVITY)

        sup = UsdGeom.Cube.Define(stage, SUPPORT_PRIM)
        sup.CreateSizeAttr().Set(1.0)
        sup.AddTranslateOp().Set(Gf.Vec3d(OBJ_CENTER[0], OBJ_CENTER[1], -0.5))
        sup.AddScaleOp().Set(Gf.Vec3f(0.1, 0.1, 1.0))
        UsdPhysics.CollisionAPI.Apply(sup.GetPrim())

        def bind_material(prim, path, sf, df, rest):
            mat = UsdShade.Material.Define(stage, path)
            pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            pm.CreateStaticFrictionAttr().Set(sf)
            pm.CreateDynamicFrictionAttr().Set(df)
            pm.CreateRestitutionAttr().Set(rest)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics")

        bind_material(sup.GetPrim(), "/World/materials/support_mat",
                      SUPPORT_FRICTION, SUPPORT_FRICTION, RESTITUTION)

        cyl = UsdGeom.Cylinder.Define(stage, OBJECT_PRIM)
        cyl.CreateRadiusAttr().Set(OBJ_RADIUS_M)
        cyl.CreateHeightAttr().Set(OBJ_HEIGHT_M)
        cyl.CreateAxisAttr().Set("Z")
        cyl.CreateExtentAttr().Set([(-OBJ_RADIUS_M, -OBJ_RADIUS_M, -OBJ_HEIGHT_M / 2),
                                    (OBJ_RADIUS_M, OBJ_RADIUS_M, OBJ_HEIGHT_M / 2)])
        cyl.AddTranslateOp().Set(Gf.Vec3d(*OBJ_CENTER))
        cyl.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(1.0, Gf.Vec3d(0, 0, 0)))
        obj_prim = cyl.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(obj_prim)
        UsdPhysics.MassAPI.Apply(obj_prim).CreateMassAttr().Set(OBJ_MASS_KG)
        UsdPhysics.CollisionAPI.Apply(obj_prim)
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(obj_prim)
        prb.CreateSolverPositionIterationCountAttr().Set(8)
        prb.CreateSolverVelocityIterationCountAttr().Set(1)
        prb.CreateMaxAngularVelocityAttr().Set(10.0)
        prb.CreateMaxLinearVelocityAttr().Set(10.0)
        prb.CreateMaxDepenetrationVelocityAttr().Set(5.0)
        prb.CreateSleepThresholdAttr().Set(0.0)
        PhysxSchema.PhysxContactReportAPI.Apply(obj_prim).CreateThresholdAttr().Set(0.0)
        bind_material(obj_prim, "/World/materials/object_mat",
                      STATIC_FRICTION, DYNAMIC_FRICTION, RESTITUTION)

        grip = stage.DefinePrim(ROOT)
        grip.GetReferences().AddReference(str(CASE_DIR / VARIANT_USD[variant]),
                                          "/bg1_gripper")
        for jaw in (BODY_L, BODY_R, GLF):
            PhysxSchema.PhysxContactReportAPI.Apply(
                stage.GetPrimAtPath(jaw)).CreateThresholdAttr().Set(0.0)

        # ---- asset gates on the composed run stage ------------------------- #
        RUN_CHILD = {GLF: (), BODY_L: (), BODY_R: ()}
        g_census = {k: census(stage, p, RUN_CHILD, pred_prox)
                    for k, p in RUN_BODIES.items()}
        ok_census = all(g_census[k] == VARIANT_CENSUS[variant][k] for k in RUN_BODIES)

        run_joints = {"j1": attr_snap(stage, J1, JOINT_ATTRS),
                      "j2": attr_snap(stage, J2, JOINT_ATTRS)}
        jdiff = {f"{jn}.{k}": (src_joints[jn][k], run_joints[jn][k])
                 for jn in ("j1", "j2") for k in JOINT_ATTRS
                 if src_joints[jn][k] != run_joints[jn][k]}
        ok_joints = not jdiff

        skip = "split2" if variant == "B" else None
        n_mesh_diff = 0
        mesh_counts = {}
        for k in RUN_BODIES:
            hs = src_mesh[k]
            hd = mesh_hashes(stage, RUN_BODIES[k], RUN_CHILD, pred_prox,
                             skip_substr=skip)
            mesh_counts[k] = (len(hs), len(hd))
            n_mesh_diff += sum(1 for kk in set(hs) | set(hd) if hs.get(kk) != hd.get(kk))
        ok_mesh = n_mesh_diff == 0 and all(a == b and a > 0
                                           for a, b in mesh_counts.values())

        run_mass = {k: attr_snap(stage, p, MASS_ATTRS) for k, p in RUN_BODIES.items()}
        mdiff = {k: {n: (src_mass[k][n], run_mass[k][n]) for n in MASS_ATTRS
                     if src_mass[k][n] != run_mass[k][n]} for k in RUN_BODIES}
        mdiff = {k: v for k, v in mdiff.items() if v}
        ok_mass = not mdiff

        gate_row = {"census": {k: list(v) for k, v in g_census.items()},
                    "census_pass": ok_census, "joint_diffs": jdiff,
                    "joints_pass": ok_joints,
                    "mesh_counts": {k: list(v) for k, v in mesh_counts.items()},
                    "mesh_n_diffs": n_mesh_diff, "mesh_pass": ok_mesh,
                    "mass_diffs": mdiff, "mass_pass": ok_mass}

        if variant == "B":
            xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
            Mg = xfc.GetLocalToWorldTransform(stage.GetPrimAtPath(GLF)).GetInverse()
            bdev = 0.0
            for body, key, sign in ((BODY_L, "gripper_left", +1),
                                    (BODY_R, "gripper_right", -1)):
                bp = stage.GetPrimAtPath(body + f"/collision_blade_split2")
                pts = np.array(UsdGeom.Mesh(bp).GetPointsAttr().Get(), dtype=np.float64)
                M = xfc.GetLocalToWorldTransform(bp.GetParent()) * Mg
                Mn = np.array([[M[i][j] for j in range(4)] for i in range(4)])
                g = pts @ Mn[:3, :3] + Mn[3, :3]
                extreme = float(g[:, 1].max() if sign > 0 else g[:, 1].min())
                bdev = max(bdev, abs(extreme - plan["blade_extreme_pins"][key]))
            gate_row["blade_extreme_dev_m"] = bdev
            gate_row["blade_pass"] = bdev < 1e-12
        else:
            gate_row["blade_pass"] = True

        asset_gate_rows[variant] = gate_row
        if not (ok_census and ok_joints and ok_mesh and ok_mass
                and gate_row["blade_pass"]):
            raise RuntimeError(f"ASSET_GATE_FAIL variant={variant} see results")
        print(f"[{LOG}] variant {variant}: asset gates PASS", flush=True)

        if variant not in jaw_pts_cache:
            def disp_points(body):
                acc = []
                for prim in body_prims(stage, body, RUN_CHILD, pred_prox):
                    ps = prim.GetPath().pathString
                    if "split2" in ps:
                        continue
                    if prim.IsA(UsdGeom.Mesh):
                        pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(),
                                       dtype=np.float64)
                        acc.append(pts[::60])
                return np.vstack(acc)
            jaw_pts_cache[variant] = {"L": disp_points(BODY_L),
                                     "R": disp_points(BODY_R),
                                     "palm": disp_points(GLF)}

        # ---- instrumentation subscriptions --------------------------------- #
        ctxst = {"mode": "idle", "pose": -1, "step": -1, "micro": 0}

        def world_pose(path):
            prim = stage.GetPrimAtPath(path)
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
            t = xf.ExtractTranslation()
            q = xf.ExtractRotationQuat()
            return (np.array([t[0], t[1], t[2]]),
                    np.array([q.GetReal(), *q.GetImaginary()]))

        def on_step(dt):
            ctxst["step"] += 1
            if ctxst["mode"] == "micro":
                ctxst["micro"] += 1
            elif ctxst["mode"] == "phases" and ctxst["step"] == 0:
                spawn_check[ctxst["pose"]] = world_pose(OBJECT_PRIM)[0]

        def on_contact(headers, data):
            pose_i, step_i, mode = ctxst["pose"], ctxst["step"], ctxst["mode"]
            for h in headers:
                raw_events["n_total"] += 1
                a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
                a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                pair = frozenset((a0, a1))
                raw_events["pairs_seen"].add((variant, a0, a1))
                if not a0 or not a1:
                    raw_events["n_unknown_actor"] += 1
                    continue
                if OBJECT_PRIM not in pair:
                    continue
                other = next(iter(pair - {OBJECT_PRIM}), None)
                imp = np.zeros(3)
                for i in range(h.contact_data_offset,
                               h.contact_data_offset + h.num_contact_data):
                    d = data[i]
                    imp += np.array([d.impulse.x, d.impulse.y, d.impulse.z])
                if mode != "phases" or pose_i < 0:
                    continue
                jaw_idx = {BODY_L: 0, BODY_R: 1}.get(other)
                if step_i < PREGRASP_STEPS:
                    if jaw_idx is not None:
                        pre_F[pose_i, step_i, jaw_idx] += imp / DT
                elif step_i < PREGRASP_STEPS + CLOSE_STEPS:
                    cs = step_i - PREGRASP_STEPS
                    if jaw_idx is not None:
                        close_F[pose_i, cs, jaw_idx] += imp / DT
                    elif other == GLF:
                        palm_F_close[pose_i, cs] += imp / DT
                    elif other == SUPPORT_PRIM:
                        support_F_close[pose_i, cs] += imp / DT

        step_sub = physx_iface.subscribe_physics_step_events(on_step)
        contact_sub = physx_sim.subscribe_contact_report_events(on_contact)

        mgr = GraspingManager()
        if not mgr.set_gripper(ROOT):
            raise RuntimeError(f"SET_GRIPPER_FAIL variant={variant}")
        mgr.set_object_prim_path(OBJECT_PRIM)
        grasping_utils.apply_joint_pregrasp_states({J1: OPEN_M, J2: OPEN_M})
        park = Gf.Vec3d(OBJ_CENTER[0], OBJ_CENTER[1], 0.6)
        idq = Gf.Quatd(1.0, Gf.Vec3d(0, 0, 0))
        mgr.set_gripper_pose(park, idq)
        mgr.store_initial_gripper_pose(park, idq)

        sup_enabled = UsdPhysics.CollisionAPI(sup.GetPrim()).GetCollisionEnabledAttr()
        if not sup_enabled:
            sup_enabled = UsdPhysics.CollisionAPI(
                sup.GetPrim()).CreateCollisionEnabledAttr(True)

        async def micro_check():
            ctxst.update(mode="micro", micro=0, step=-1)
            await grasping_utils.simulate_physics_async(5, DT, None, render=False)
            t_obj = world_pose(OBJECT_PRIM)[0]
            t_root = world_pose(ROOT)[0]
            ok = (ctxst["micro"] == 5
                  and float(np.linalg.norm(t_obj - OBJ_CENTER)) < 2e-3
                  and float(np.linalg.norm(t_root - np.array([park[0], park[1], park[2]]))) < 1e-6)
            out.setdefault("harness_selfcheck", {})[variant] = {
                "step_events_counted": ctxst["micro"], "expected": 5,
                "object_drift_m": float(np.linalg.norm(t_obj - OBJ_CENTER)),
                "root_drift_m": float(np.linalg.norm(
                    t_root - np.array([park[0], park[1], park[2]]))), "pass": ok}
            if not ok:
                raise RuntimeError(
                    f"HARNESS_SELFCHECK_FAIL {out['harness_selfcheck'][variant]}")
            ctxst.update(mode="idle")

        async def hang_test(gi):
            snap = pose_snaps[gi]
            sup_enabled.Set(False)
            z_series, q_series = [], []
            z_series.append(world_pose(OBJECT_PRIM)[0][2])
            q_series.append([grasping_utils.get_joint_state(stage.GetPrimAtPath(J1))[0],
                             grasping_utils.get_joint_state(stage.GetPrimAtPath(J2))[0]])
            for _ in range(HANG_STEPS // HANG_CHUNK):
                await grasping_utils.simulate_physics_async(HANG_CHUNK, DT, None,
                                                            render=False)
                z_series.append(world_pose(OBJECT_PRIM)[0][2])
                q_series.append(
                    [grasping_utils.get_joint_state(stage.GetPrimAtPath(J1))[0],
                     grasping_utils.get_joint_state(stage.GetPrimAtPath(J2))[0]])
            sup_enabled.Set(True)
            hang_z[gi, :len(z_series)] = z_series
            hang_q[gi, :len(q_series)] = q_series
            t_root_end = world_pose(ROOT)[0]
            snap["hang"] = {
                "z_start_m": float(z_series[0]), "z_end_m": float(z_series[-1]),
                "drop_m": float(z_series[0] - z_series[-1]),
                "q_end_m": [float(q_series[-1][0]), float(q_series[-1][1])],
                "root_drift_end_m": float(np.linalg.norm(
                    t_root_end - np.array(snap["target_pos"]))),
            }

        def make_progress(offset):
            async def progress(idx):
                gi = offset + idx - 1
                ctxst.update(mode="hang")
                pose = plan["poses"][gi - offset]
                t_root, q_root = world_pose(ROOT)
                t_l, q_l = world_pose(BODY_L)
                t_r, q_r = world_pose(BODY_R)
                t_g, q_g = world_pose(GLF)
                t_obj, q_obj = world_pose(OBJECT_PRIM)
                qs = [grasping_utils.get_joint_state(stage.GetPrimAtPath(J1))[0],
                      grasping_utils.get_joint_state(stage.GetPrimAtPath(J2))[0]]
                pose_snaps[gi] = {
                    "variant": variant,
                    "target_pos": pose["pos"], "target_quat_wxyz": pose["quat_wxyz"],
                    "post_close": {
                        "root_pos": t_root.tolist(), "root_quat_wxyz": q_root.tolist(),
                        "left_pos": t_l.tolist(), "left_quat_wxyz": q_l.tolist(),
                        "right_pos": t_r.tolist(), "right_quat_wxyz": q_r.tolist(),
                        "palm_pos": t_g.tolist(), "palm_quat_wxyz": q_g.tolist(),
                        "obj_pos": t_obj.tolist(), "obj_quat_wxyz": q_obj.tolist(),
                        "q_m": [float(qs[0]), float(qs[1])],
                        "root_target_err_m": float(np.linalg.norm(
                            t_root - np.array(pose["pos"]))),
                        "glf_root_err_m": float(np.linalg.norm(t_g - t_root)),
                    }}
                await hang_test(gi)
                ctxst.update(mode="idle", pose=-1, step=-1)
            return progress

        async def run_variant():
            await micro_check()
            t_obj = world_pose(OBJECT_PRIM)[0]
            if float(np.linalg.norm(t_obj - OBJ_CENTER)) > 1e-4:
                raise RuntimeError(f"OBJECT_NOT_AT_SPAWN_BEFORE_GROUP {variant} {t_obj}")
            mgr.grasp_phases = [
                GraspPhase("PREGRASP", {J1: OPEN_M, J2: OPEN_M}, PREGRASP_STEPS, DT),
                GraspPhase("CLOSE", {J1: CLOSE_M, J2: CLOSE_M}, CLOSE_STEPS, DT),
            ]
            poses = []
            for p in plan["poses"]:
                q = p["quat_wxyz"]
                poses.append((Gf.Vec3d(*p["pos"]),
                              Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3]))))

            orig_eval = mgr.evaluate_grasp_pose

            async def eval_with_ctx(loc, quat, **kw):
                qi = np.array([quat.GetReal(), *quat.GetImaginary()])
                li = np.array([loc[0], loc[1], loc[2]])
                pi = next(k for k, p in enumerate(plan["poses"])
                          if np.allclose(p["pos"], li, atol=1e-12)
                          and np.allclose(p["quat_wxyz"], qi, atol=1e-12))
                ctxst.update(mode="phases", pose=vi * N_POSES + pi, step=-1)
                await orig_eval(loc, quat, **kw)
            mgr.evaluate_grasp_pose = eval_with_ctx
            try:
                await mgr.evaluate_grasp_poses(
                    poses, render=False, physics_scene_path=None,
                    isolate_simulation=False,
                    progress_callback=make_progress(vi * N_POSES))
            finally:
                mgr.evaluate_grasp_pose = orig_eval

        task = asyncio.ensure_future(run_variant())
        kit = omni.kit.app.get_app()
        guard = 0
        while not task.done():
            kit.update()
            guard += 1
            if guard > 800000:
                raise RuntimeError("EVENT_LOOP_GUARD")
        if task.exception() is not None:
            raise task.exception()
        step_sub = None
        contact_sub = None
        print(f"[{LOG}] variant {variant}: 13 poses complete", flush=True)

    out["asset_gates"] = asset_gate_rows

    # ---- gates + taxonomy over 26 rows ------------------------------------- #
    preF_mag = np.linalg.norm(pre_F, axis=3)
    closeF_mag = np.linalg.norm(close_F, axis=3)
    bilateral_min = closeF_mag.min(axis=2)
    n_success = {"A": 0, "B": 0}
    for gi in range(N_ROWS):
        variant = VARIANTS[gi // N_POSES]
        pose = plan["poses"][gi % N_POSES]
        snap = pose_snaps[gi]
        close_bilateral = bool((bilateral_min[gi] > BILATERAL_GATE_N).any())
        drop = snap["hang"]["drop_m"]
        hold = bool(drop < HOLD_DROP_GATE_M and np.isfinite(drop))
        success = close_bilateral and hold
        preclose = bool((preF_mag[gi] > BILATERAL_GATE_N).any())
        left_contact = bool((closeF_mag[gi, :, 0] > BILATERAL_GATE_N).any())
        right_contact = bool((closeF_mag[gi, :, 1] > BILATERAL_GATE_N).any())
        if success:
            taxonomy = "SUCCESS"
        elif close_bilateral:
            taxonomy = "BILATERAL_NO_HOLD"
        elif preclose:
            taxonomy = "PRECLOSE_COLLISION"
        elif left_contact and not right_contact:
            taxonomy = "ONE_JAW_ONLY_LEFT"
        elif right_contact and not left_contact:
            taxonomy = "ONE_JAW_ONLY_RIGHT"
        else:
            taxonomy = "NO_JAW_CONTACT"
        measurement_valid = bool(
            snap["post_close"]["root_target_err_m"] < 1e-6
            and snap["post_close"]["glf_root_err_m"] < 1e-6
            and snap["hang"]["root_drift_end_m"] < 1e-6
            and (np.isnan(spawn_check[gi]).any()
                 or float(np.linalg.norm(spawn_check[gi] - OBJ_CENTER)) < 2e-3))
        n_success[variant] += int(success)
        row = {
            "index": gi, "variant": variant, "label": pose["label"],
            "kind": pose["kind"],
            "close_bilateral": close_bilateral,
            "close_bilateral_peak_n": float(bilateral_min[gi].max()),
            "close_left_peak_n": float(closeF_mag[gi, :, 0].max()),
            "close_right_peak_n": float(closeF_mag[gi, :, 1].max()),
            "close_palm_peak_n": float(np.linalg.norm(palm_F_close[gi], axis=1).max()),
            "preclose_jaw_contact": preclose,
            "preclose_peak_n": float(preF_mag[gi].max()),
            "hang_drop_m": drop, "hold": hold, "success": success,
            "taxonomy": taxonomy, "measurement_valid": measurement_valid,
            "post_close_q_m": snap["post_close"]["q_m"],
            "post_close_obj_xy_dev_mm": float(np.linalg.norm(
                np.array(snap["post_close"]["obj_pos"][:2]) - OBJ_CENTER[:2]) * 1000),
            "hang_end_q_m": snap["hang"]["q_end_m"],
            "root_target_err_m": snap["post_close"]["root_target_err_m"],
            "spawn_check_drift_m": (None if np.isnan(spawn_check[gi]).any() else
                                    float(np.linalg.norm(spawn_check[gi] - OBJ_CENTER))),
        }
        verdict_rows.append(row)
        print(f"[{LOG}] row {gi:02d} {variant}:{row['label']}: {taxonomy} "
              f"bilat={row['close_bilateral_peak_n']:.3f}N "
              f"drop={drop * 1000:.2f}mm valid={measurement_valid}", flush=True)

    all_valid = all(r["measurement_valid"] for r in verdict_rows)
    nA, nB = n_success["A"], n_success["B"]
    if nB >= 1 and nA == 0:
        code = "BG1_REAL_GEOM_HOLDS_USD_COLLISION_BLOCKS"
    elif nA >= 1 and nB >= 1:
        code = "BG1_HOLDS_EVEN_ASWRITTEN"
    elif nA == 0 and nB == 0:
        code = "BG1_B601_FAILS_UNDER_PROTOCOL"
    else:
        code = "BG1_ANOMALY_A_HOLDS_B_FAILS"
    out["rows"] = verdict_rows
    out["verdict"] = {
        "code": code, "n_success_A": nA, "n_success_B": nB, "n_poses_per_variant": N_POSES,
        "all_measurements_valid": all_valid,
        "gates": {"bilateral_n": BILATERAL_GATE_N, "hold_drop_m": HOLD_DROP_GATE_M},
        "branch_semantics": {
            "B>=1 & A==0": "real blade geometry holds; official USD 1-hull collision "
                           "rep is the blocker (prereg SS1 branch i)",
            "A>=1 & B>=1": "holds even as-written (branch ii)",
            "A==0 & B==0": "B601 geometry also fails under this protocol (branch iii)",
            "A>=1 & B==0": "anomaly - representation fidelity inverted (branch iv)"},
        "non_claims": "no claim about real-robot B601 grasping, arm kinematics/IK, "
                      "friction realism, or D419/fg1 re-judgment (prereg SS1)"}
    out["raw_contact_stats"] = {
        "n_total_events": raw_events["n_total"],
        "n_unknown_actor": raw_events["n_unknown_actor"],
        "pairs_seen": sorted(str(p) for p in raw_events["pairs_seen"])}
    out["pose_snaps"] = pose_snaps

    # ---- trace ------------------------------------------------------------- #
    np.savez(
        OUT["trace.npz"],
        pre_F=pre_F, close_F=close_F, palm_F_close=palm_F_close,
        support_F_close=support_F_close, hang_z=hang_z, hang_q=hang_q,
        spawn_check=spawn_check, bilateral_min=bilateral_min,
        row_variants=np.array([VARIANTS[gi // N_POSES] for gi in range(N_ROWS)]),
        pose_labels=np.array([plan["poses"][gi % N_POSES]["label"]
                              for gi in range(N_ROWS)]),
        target_pos=np.array([plan["poses"][gi % N_POSES]["pos"]
                             for gi in range(N_ROWS)]),
        target_quat=np.array([plan["poses"][gi % N_POSES]["quat_wxyz"]
                              for gi in range(N_ROWS)]))
    with open(OUT["trace.npz"], "rb") as f:
        os.fsync(f.fileno())

    # ---- rerun (D341, save-only) ------------------------------------------- #
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    app_id = f"roarm_g0c_d446_{TAG}"

    def box_wire(center, half):
        c, h = np.asarray(center), np.asarray(half)
        s = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
             [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
        v = [c + h * np.array(x) for x in s]
        e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
        return [[v[a].tolist(), v[b].tolist()] for a, b in e]

    def cyl_wire(center_xyz, quat_wxyz=None):
        c = np.asarray(center_xyz)
        ang = np.linspace(0, 2 * math.pi, 49)
        rings = []
        for zk in (-OBJ_HEIGHT_M / 2, 0.0, OBJ_HEIGHT_M / 2):
            ring = np.column_stack([OBJ_RADIUS_M * np.cos(ang),
                                    OBJ_RADIUS_M * np.sin(ang),
                                    np.full_like(ang, zk)])
            if quat_wxyz is not None:
                w, x, y, z = quat_wxyz
                R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                              [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                              [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
                ring = ring @ R.T
            rings.append((ring + c).tolist())
        return rings

    def axes_strips(pos, quat_wxyz, scale=0.04):
        w, x, y, z = quat_wxyz
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        p = np.asarray(pos)
        return [[p.tolist(), (p + R[:, k] * scale).tolist()] for k in range(3)]

    def to_world(pts, pos, quat_wxyz):
        w, x, y, z = quat_wxyz
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return pts @ R.T + np.asarray(pos)

    with rr.RecordingStream(app_id, recording_id=f"g0c_d446_{TAG}", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(OUT["timeline.rrd"]), write_footer=True)
        rec.log("scene/support", rr.LineStrips3D(
            box_wire([OBJ_CENTER[0], OBJ_CENTER[1], -0.5], [0.05, 0.05, 0.5]),
            colors=[[120, 120, 120]], radii=0.0008), static=True)
        rec.log("scene/object_spawn", rr.LineStrips3D(
            cyl_wire(OBJ_CENTER), colors=[[80, 200, 120]], radii=0.0006), static=True)

        gstep = 0
        for gi in range(N_ROWS):
            variant = VARIANTS[gi // N_POSES]
            jp = jaw_pts_cache[variant]
            snap = pose_snaps[gi]
            row = verdict_rows[gi]
            rec.reset_time()
            rec.set_time("row_index", sequence=gi)
            pc = snap["post_close"]
            rec.log("gripper/left_points", rr.Points3D(
                to_world(jp["L"], pc["left_pos"], pc["left_quat_wxyz"]),
                colors=[70, 130, 230], radii=0.0006))
            rec.log("gripper/right_points", rr.Points3D(
                to_world(jp["R"], pc["right_pos"], pc["right_quat_wxyz"]),
                colors=[230, 130, 70], radii=0.0006))
            rec.log("gripper/palm_points", rr.Points3D(
                to_world(jp["palm"], pc["palm_pos"], pc["palm_quat_wxyz"]),
                colors=[150, 150, 160], radii=0.0006))
            rec.log("gripper/root_target_axes", rr.LineStrips3D(
                axes_strips(snap["target_pos"], snap["target_quat_wxyz"]),
                colors=[[255, 60, 60], [60, 255, 60], [60, 60, 255]], radii=0.0009))
            rec.log("gripper/root_actual_axes", rr.LineStrips3D(
                axes_strips(pc["root_pos"], pc["root_quat_wxyz"]),
                colors=[[255, 160, 160], [160, 255, 160], [160, 160, 255]], radii=0.0005))
            rec.log("object/post_close", rr.LineStrips3D(
                cyl_wire(pc["obj_pos"], pc["obj_quat_wxyz"]),
                colors=[[225, 60, 60]], radii=0.0006))
            rec.log("plots/hang_drop_mm",
                    rr.Scalars(float(row["hang_drop_m"] * 1000.0)))
            rec.log("events/verdict", rr.TextLog(
                f"row {gi:02d} {variant}:{row['label']}: {row['taxonomy']} "
                f"bilat={row['close_bilateral_peak_n']:.3f}N "
                f"drop={row['hang_drop_m'] * 1000:.2f}mm",
                level=rr.TextLogLevel.INFO if row["success"] else rr.TextLogLevel.WARN))

            for phase, arr, nst in (("PREGRASP", preF_mag[gi], PREGRASP_STEPS),
                                    ("CLOSE", closeF_mag[gi], CLOSE_STEPS)):
                rec.reset_time()
                rec.set_time("row_index", sequence=gi)
                rec.set_time("global_step", sequence=gstep)
                rec.log("events/phase", rr.TextLog(
                    f"row {gi:02d} {variant} {phase} begin",
                    level=rr.TextLogLevel.INFO))
                for s in range(nst):
                    rec.reset_time()
                    rec.set_time("row_index", sequence=gi)
                    rec.set_time("global_step", sequence=gstep)
                    rec.log("forces/left_n", rr.Scalars(float(arr[s, 0])))
                    rec.log("forces/right_n", rr.Scalars(float(arr[s, 1])))
                    rec.log("forces/bilateral_min_n", rr.Scalars(float(arr[s].min())))
                    gstep += 1
            rec.reset_time()
            rec.set_time("row_index", sequence=gi)
            rec.set_time("global_step", sequence=gstep)
            rec.log("events/phase", rr.TextLog(f"row {gi:02d} {variant} HANG begin",
                                               level=rr.TextLogLevel.INFO))
            zrow = hang_z[gi]
            for k in range(zrow.shape[0]):
                if np.isnan(zrow[k]):
                    continue
                rec.reset_time()
                rec.set_time("row_index", sequence=gi)
                rec.set_time("global_step", sequence=gstep + k * HANG_CHUNK)
                rec.log("plots/hang_z_m", rr.Scalars(float(zrow[k])))
            gstep += HANG_STEPS

        rec.reset_time()
        rec.set_time("row_index", sequence=0)
        summary_md = (
            f"# g0c_d446 {TAG} — B601 flying-gripper probe (2 collision variants)\n\n"
            f"**VERDICT: {code}** (A {nA}/13, B {nB}/13)\n\n"
            f"13 analytic poses (8 side + 5 top-tilt) x variants A (official USD "
            f"1-hull/finger) and B (blade/mount split of the same official points).  "
            f"SUCCESS = same-step bilateral close force > {BILATERAL_GATE_N} N AND "
            f"hang drop < {HOLD_DROP_GATE_M * 1000:.0f} mm ({HANG_STEPS} steps, "
            f"pedestal collider off).\n\n"
            f"Blue/orange points = left/right finger, grey = palm at post-CLOSE; "
            f"red = object post-CLOSE; green = spawn; bright/pale axes = root "
            f"target/actual.\n\n"
            f"Authority = bg1_results.json + bg1_trace.npz; Rerun is inspection "
            f"evidence only (D341).")
        rec.log("metadata/run", rr.TextDocument(summary_md,
                                                media_type=rr.MediaType.MARKDOWN),
                static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run",
                                         name="1 | verdict"),
                    rrb.Spatial3DView(origin="/", contents=["/scene/**", "/gripper/**",
                                                            "/object/**"],
                                      name="2 | post-close geometry per row"),
                    rrb.TextLogView(origin="/events", contents="/events/**",
                                    name="3 | phases + verdicts"),
                    column_shares=[0.26, 0.48, 0.26],
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/forces", contents="/forces/**",
                                       name="4 | jaw-object contact force [N]"),
                    rrb.TimeSeriesView(origin="/plots", contents="/plots/**",
                                       name="5 | hang drop [mm] / hang z [m]"),
                ),
                row_shares=[0.58, 0.42],
            ),
            auto_layout=False, auto_views=False, collapse_panels=True,
        )
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=30.0)
    blueprint.save(app_id, str(OUT["timeline.rbl"]))

    expected_entities = [
        "metadata/run", "scene/support", "scene/object_spawn",
        "gripper/left_points", "gripper/right_points", "gripper/palm_points",
        "gripper/root_target_axes", "gripper/root_actual_axes", "object/post_close",
        "plots/hang_drop_mm", "plots/hang_z_m",
        "forces/left_n", "forces/right_n", "forces/bilateral_min_n",
        "events/phase", "events/verdict"]
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    lin3 = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    components = {
        "metadata/run": ["TextDocument:text"],
        "scene/support": lin3, "scene/object_spawn": lin3,
        "gripper/left_points": pts3, "gripper/right_points": pts3,
        "gripper/palm_points": pts3,
        "gripper/root_target_axes": lin3, "gripper/root_actual_axes": lin3,
        "object/post_close": lin3,
        "plots/hang_drop_mm": ["Scalars:scalars"], "plots/hang_z_m": ["Scalars:scalars"],
        "forces/left_n": ["Scalars:scalars"], "forces/right_n": ["Scalars:scalars"],
        "forces/bilateral_min_n": ["Scalars:scalars"],
        "events/phase": ["TextLog:text", "TextLog:level"],
        "events/verdict": ["TextLog:text", "TextLog:level"],
    }
    validation = validate_rerun_artifact(
        OUT["timeline.rrd"],
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time", "row_index", "global_step"],
        expected_entity_components=components,
        blueprint_path=OUT["timeline.rbl"],
        screenshot_path=OUT["inspection.png"],
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=240.0,
    )
    fsync_write(OUT["rerun_validation.json"],
                json.dumps(validation, indent=2, default=str) + "\n")
    out["rerun_validation_pass"] = bool(validation.get("pass"))
    print(f"[{LOG}] rerun_validation pass={validation.get('pass')} "
          f"errors={validation.get('errors')}", flush=True)

    # ---- end-of-run pin recheck -------------------------------------------- #
    end_pins = {}
    for path, want in SRC_PINS.items():
        end_pins[path.name] = sha256(path) == want
    for v in VARIANTS:
        p = CASE_DIR / VARIANT_USD[v]
        end_pins[p.name] = sha256(p) == plan["pins"][VARIANT_USD[v]]["sha256"]
    out["end_pin_recheck"] = end_pins
    if not all(end_pins.values()):
        raise RuntimeError(f"END_PIN_DRIFT {end_pins}")

    out["artifacts"] = {p.name: {"sha256_16": sha256(p)[:16], "bytes": p.stat().st_size}
                        for k, p in OUT.items()
                        if p.exists() and k not in ("results.json", "failure.json")}
    out["artifacts_note"] = ("results.json deliberately absent (D429-R1: hash it from "
                             "disk, a self-manifest cannot carry its own final hash)")
    out["wall_seconds"] = round(time.time() - t_start, 1)
    out["physics_backend_note"] = ("default physics scene, direct simulate/fetch_results "
                                   "stepping (CPU PhysX path), no GPU dynamics authored")
    fsync_write(OUT["results.json"], json.dumps(out, indent=2, default=str) + "\n")
    print(f"[{LOG}] results.json={sha256(OUT['results.json'])[:16]} "
          f"bytes={OUT['results.json'].stat().st_size}", flush=True)
    print(f"[{LOG}] G0C_BG1_VERDICT={code} A={nA}/13 B={nB}/13 all_valid={all_valid}",
          flush=True)
    return 0


def main() -> int:
    plan = preflight()
    fsync_write(OUT["argv.txt"], " ".join([sys.executable, *sys.argv]) + "\n")
    shutil.copyfile(__file__, OUT["script.py.txt"])
    print(f"[{LOG}] preflight OK: 13 poses x 2 variants, pins verified, tag locked",
          flush=True)
    return run(plan)


if __name__ == "__main__":
    sys.exit(main())
