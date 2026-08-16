#!/usr/bin/env python3
"""p17 / fg1 — g0b_d444 flying-gripper bottleneck isolation probe.

Contract: claudedocs/runtime_logs/grasp_track/g0b_d444/fg1_prereg.md (57th session,
user-approved case, D444).  New variable (exactly one): the arm is removed — the frozen
attempt3 gripper subtree runs as a standalone fixed-root articulation
(fg1_gripper_only.usd, reference + enumerable overrides, zero geometry/joint edits).

13 probe poses: 8 sdg2 side candidates (verbatim link5-origin targets) + 5 rim-tilt
argmax rows (theta in {6,15,24,29,35} deg).  Phases per pose: PREGRASP (open, 60 steps)
-> CLOSE (120 steps) via GraspingManager.evaluate_grasp_poses, then a HANG test outside
the manager (support collider disabled, 240 steps).  SUCCESS = same-step bilateral
close force > 0.01 N AND hang drop < 6 mm.

Preregistration deviations carried by this runner (documented, reactive):
  DEV-1  rim-row source: prereg SS4(b) cites t3r_n8_tilt_results.json, but that file's
         per-theta argmax rows at theta=6/15 have NEGATIVE bite (-0.1258/-0.0055 mm)
         and cannot be the D431(2) rows the prereg describes; the semantically
         identified rows (argmax bite over the full 34-q5 sweep per theta) live in
         t3r_n8b_tiltmin_results.json `theta_ladder_full_q5`, whose values D431(2)
         quotes verbatim.  fg1 consumes the n8b rows; both files SHA-pinned; the
         n8_tilt counter-rows are recorded in the results for audit.
  DEV-2  isolation off: prereg SS2 mirrors the manager's isolate_simulation=True temp
         scene.  The 57th smoke harness proved that mode (a) drops the STATIC support
         collider from the temp scene (object free-fell 4.9 m in 60 settle steps) and
         (b) scene migration corrupts reset-restore state across poses (object
         restored to a mid-fall pose, not spawn).  fg1 runs the manager's official
         default-scene mode (physics_scene_path=None); the purpose-built stage has
         exactly 4 actors so isolation is vacuous.  Smoke evidence in session doc.
  DEV-3  side close target: sdg2 q5_control is null; prereg SS5(a) names the D431(6)
         band 14-22 deg, deepest value -> 14.0 deg exactly (closing decreases q5).
  DEV-4  hang stepping: 240 steps executed as 8 x 30-step calls of the audited
         grasping_utils.simulate_physics_async (default scene) to obtain a drop
         curve; total steps and dt unchanged.

Support-plane material (not pinned by prereg): static/dynamic friction 1.0/1.0,
restitution 0.0 — the Isaac Lab ground default of the g0b_d420 runs.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
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

CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d444"
D420_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"
ATTEMPT3_DIR = (REPO / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/"
                "attempt3/roarm_m3_fullmesh_fixed_point_parts")
TAG = "fg1"
LOG = "g0b_fg1"

PINS = {
    ATTEMPT3_DIR / "roarm_m3.usd":
        "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
    ATTEMPT3_DIR / "configuration/roarm_m3_base.usd":
        "ea0ee8f258e935799cf927b8c67e871f935c09b3c9be4f971006937334a11841",
    ATTEMPT3_DIR / "configuration/roarm_m3_physics.usd":
        "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503",
    ATTEMPT3_DIR / "configuration/roarm_m3_robot.usd":
        "2227536fcb8c9dae1aa9cc1cf422350fcf85e662eed97fe9ea48535c6b4aa65d",
    ATTEMPT3_DIR / "configuration/roarm_m3_sensor.usd":
        "3f44081f42b452bc5f9791a8df1c37e00ba5a6dc98a9e49e065c7acacdda0d0f",
    CASE_DIR / "fg1_gripper_only.usd":
        "0e9fc601df9379fabc118eb2495ac0100350ef9931662413c5a2c0f00690dd76",
    D420_DIR / "t3s_side_sdg2_candidates.json":
        "67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384",
    D420_DIR / "t3r_n8b_tiltmin_results.json":
        "180e03734544c89470b69320402f5e26b3a451a540f8f2ba3e80fb5407727362",
    D420_DIR / "t3r_n8_tilt_script.py.txt":
        "84ab44dc9d9d87afa060280f967762a7e4298190ed6458f60418223f9801d5e7",
    D420_DIR / "t3r_n8_tilt_results.json":
        "4b2a342880ac6a5dbabad4e1cc11c3f17b0f7055a84bc19288d9e2f0e36a4da1",
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
OBJ_TOP = OBJ_CENTER + np.array([0.0, 0.0, OBJ_HEIGHT_M / 2.0])
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
Q5_OPEN_DEG = 88.30998496351378
SIDE_CLOSE_DEG = 14.0
RIM_CLOSE_MARGIN_DEG = 2.0
BILATERAL_GATE_N = 0.01
HOLD_DROP_GATE_M = 0.006
TCP_Z_M = 0.115428
RIM_THETAS = (6.0, 15.0, 24.0, 29.0, 35.0)
# D431 (2) pinned n8b ladder values, verified against the loaded file at 1e-9.
N8B_EXPECT = {
    6.0: (0.0, 19.690052728292695, 0.1619832103215593, 3.4038608891713236),
    15.0: (0.0, 19.690052728292695, 4.403417249408376, 1.7712018824697409),
    24.0: (0.0, 22.50291740376308, 9.035777501177574, -0.00617560460674142),
    29.0: (0.0, 25.315782079233465, 12.108689869177224, -1.0997989217767794),
    35.0: (0.0, 25.315782079233465, 15.542998402354836, -2.562555169278158),
}
PARK_POS = np.array([0.4235072423787768, 0.17237803311822986, 0.5])
Q5_JOINT = "/World/fg1_gripper/joints/link5_to_gripper_link"
JAW_FIXED = "/World/fg1_gripper/link5"
JAW_MOVING = "/World/fg1_gripper/gripper_link"
OBJECT_PRIM = "/World/object"
SUPPORT_PRIM = "/World/support"
GRIPPER_ROOT = "/World/fg1_gripper"
Q5_LIMITS_DEG = (0.0, 90.01166534423828)

OUT = {k: CASE_DIR / f"{TAG}_{k}" for k in (
    "results.json", "trace.npz", "timeline.rrd", "timeline.rbl", "rerun_validation.json",
    "inspection.png", "stdout.log", "exit_status.txt", "script.py.txt", "argv.txt",
    "failure.json")}


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
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-9:
        raise RuntimeError("degenerate quaternion")
    q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                  (R[1, 0] - R[0, 1]) / (4 * w)])
    return q / np.linalg.norm(q)


def load_n8_module():
    src = (D420_DIR / "t3r_n8_tilt_script.py.txt").read_text()
    spec = importlib.util.spec_from_loader("frozen_n8_tilt", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(D420_DIR / "t3r_n8_tilt_script.py.txt")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def build_poses() -> dict:
    """Preflight pose construction with fatal verification gates.  No Isaac."""
    plan = {"poses": [], "gates": {}, "deviations": {}}

    cand = json.loads((D420_DIR / "t3s_side_sdg2_candidates.json").read_text())
    rows = sorted(cand["candidates"], key=lambda r: r["candidate_rank"])
    if len(rows) != 8:
        raise RuntimeError(f"SIDE_ROW_COUNT {len(rows)} != 8")
    oc = cand["object_contract"]
    if (abs(np.array(oc["center_base_m"]) - OBJ_CENTER).max() > 1e-12
            or oc["radius_m"] != OBJ_RADIUS_M or oc["height_m"] != OBJ_HEIGHT_M
            or oc["mass_kg"] != OBJ_MASS_KG):
        raise RuntimeError("OBJECT_CONTRACT_MISMATCH")
    quat_dev = 0.0
    for r in rows:
        R = np.array(r["R_base_link5_proposal"], dtype=np.float64)
        if abs(np.linalg.det(R) - 1.0) > 1e-9 or np.abs(R @ R.T - np.eye(3)).max() > 1e-9:
            raise RuntimeError(f"SIDE_ROTATION_INVALID {r['candidate_id']}")
        q = quat_from_R(R)
        q_ref = np.array(r["orientation_quaternion_wxyz_base"], dtype=np.float64)
        dev = min(np.abs(q - q_ref).max(), np.abs(q + q_ref).max())
        quat_dev = max(quat_dev, dev)
        pos = np.array(r["geometry_mapped_roarm_targets"]["link5_origin_target_base_m"],
                       dtype=np.float64)
        plan["poses"].append({
            "kind": "side", "label": r["candidate_id"], "pos": pos.tolist(),
            "quat_wxyz": q.tolist(), "close_deg": SIDE_CLOSE_DEG,
            "source": "t3s_side_sdg2_candidates.json geometry_mapped_roarm_targets "
                      "+ R_base_link5_proposal (verbatim)"})
    if quat_dev > 1e-9:
        raise RuntimeError(f"SIDE_QUAT_CROSSCHECK_DEV {quat_dev}")
    plan["gates"]["side_quat_crosscheck_max_dev"] = quat_dev

    n8b = json.loads((D420_DIR / "t3r_n8b_tiltmin_results.json").read_text())
    ladder = {row["theta_deg"]: row for row in n8b["theta_ladder_full_q5"]}
    n8 = load_n8_module()
    if abs(n8.TCP_Z_MM - 115.428) > 1e-12:
        raise RuntimeError("N8_TCP_CONSTANT_MISMATCH")
    down = np.array([0.0, 0.0, -1.0])
    rim_gate = 0.0
    for th in RIM_THETAS:
        row = ladder[th]
        exp = N8B_EXPECT[th]
        got = (row["phi_deg"], row["q5_deg"], row["bite_mm"], row["depth_top_min_mm"])
        if any(abs(a - b) > 1e-9 for a, b in zip(got, exp)):
            raise RuntimeError(f"N8B_ROW_MISMATCH theta={th} got={got} expect={exp}")
        chat = n8.axis_dir(math.radians(th), math.radians(row["phi_deg"]))
        delta_m = row["depth_top_min_mm"] / 1000.0
        t_top_l5 = np.array([0.0, 0.0, TCP_Z_M + delta_m])
        a = np.cross(chat, down)
        s, c = np.linalg.norm(a), float(chat @ down)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (s * s))
        pos = OBJ_TOP - R @ t_top_l5
        rim_gate = max(rim_gate,
                       float(np.abs(R @ chat - down).max()),
                       float(np.abs(R @ R.T - np.eye(3)).max()),
                       abs(np.linalg.det(R) - 1.0),
                       float(np.abs(R @ t_top_l5 + pos - OBJ_TOP).max()),
                       float(np.abs(R.T @ down - chat).max()))
        plan["poses"].append({
            "kind": "rim", "label": f"rim_theta{int(th):02d}", "pos": pos.tolist(),
            "quat_wxyz": quat_from_R(R).tolist(),
            "close_deg": row["q5_deg"] - RIM_CLOSE_MARGIN_DEG,
            "theta_deg": th, "phi_deg": row["phi_deg"], "q5_row_deg": row["q5_deg"],
            "bite_mm": row["bite_mm"], "depth_top_min_mm": row["depth_top_min_mm"],
            "source": "t3r_n8b_tiltmin_results.json theta_ladder_full_q5 (DEV-1) + "
                      "frozen n8 axis_dir verbatim; world map: R@chat=-z minimal rotation, "
                      "top_center_link5=[0,0,TCP+delta]"})
    if rim_gate > 1e-12:
        raise RuntimeError(f"RIM_POSE_GATE {rim_gate}")
    plan["gates"]["rim_pose_construction_max_err"] = rim_gate

    n8t = json.loads((D420_DIR / "t3r_n8_tilt_results.json").read_text())
    counter = {}
    pool = list(n8t.get("grid_theta_phi_q5", [])) + list(n8t.get("grid_refined", []))
    pool += list(n8t.get("full_q5_sweep_at_best_tilt", []))
    for th in RIM_THETAS:
        c = [r for r in pool if isinstance(r, dict)
             and abs(r.get("theta_deg", -1) - th) < 1e-9 and r.get("bite_mm") is not None]
        best = max(c, key=lambda r: r["bite_mm"]) if c else None
        counter[str(th)] = ({k: best[k] for k in ("phi_deg", "q5_deg", "bite_mm",
                                                  "depth_top_min_mm")} if best else None)
    plan["deviations"]["DEV1_rim_source_erratum"] = {
        "prereg_cited_file": "t3r_n8_tilt_results.json",
        "consumed_file": "t3r_n8b_tiltmin_results.json theta_ladder_full_q5",
        "reason": "prereg-cited file's per-theta argmax rows contradict the D431(2) "
                  "values the prereg describes (negative bite at theta=6/15)",
        "cited_file_argmax_counter_rows": counter}
    plan["deviations"]["DEV2_isolation_off"] = {
        "prereg_language": "isolate_simulation=True temp scene (SS2)",
        "run_mode": "physics_scene_path=None default scene",
        "evidence": "57th fg1 smoke: temp-scene mode dropped the static support collider "
                    "(object free fall in settle) and scene migration corrupted "
                    "reset-restore across poses"}
    plan["deviations"]["DEV3_side_close_target"] = {
        "value_deg": SIDE_CLOSE_DEG,
        "rationale": "sdg2 q5_control null; D431(6) band 14-22 deg, deepest (closing "
                     "decreases q5)"}
    plan["deviations"]["DEV4_hang_chunks"] = {
        "steps": HANG_STEPS, "chunk": HANG_CHUNK,
        "rationale": "drop curve via audited simulate_physics_async; steps/dt unchanged"}

    for p in plan["poses"]:
        for v in (p["close_deg"], Q5_OPEN_DEG):
            if not (Q5_LIMITS_DEG[0] <= v <= Q5_LIMITS_DEG[1]):
                raise RuntimeError(f"TARGET_OUT_OF_LIMITS {p['label']} {v}")
    if len(plan["poses"]) != 13:
        raise RuntimeError("POSE_COUNT != 13")
    return plan


def preflight() -> dict:
    import numpy
    import psutil
    if numpy.__version__ != NUMPY_PIN or psutil.__version__ != PSUTIL_PIN:
        raise RuntimeError(f"ENV_PIN numpy={numpy.__version__} psutil={psutil.__version__}")
    pins = {}
    for path, want in PINS.items():
        got = sha256(path)
        pins[path.name] = {"sha256": got, "match": got == want}
        if got != want:
            raise RuntimeError(f"SHA_DRIFT {path}")
    ext = Path("/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
               "exts/isaacsim.replicator.grasping/config/extension.toml")
    if sha256(ext) != EXT_MANIFEST_SHA256:
        raise RuntimeError("EXT_MANIFEST_DRIFT")
    pins["extension.toml"] = {"sha256": EXT_MANIFEST_SHA256, "match": True}

    existing = [p.name for k, p in OUT.items()
                if p.exists() and k not in ("script.py.txt", "argv.txt", "stdout.log")]
    if existing:
        raise RuntimeError(f"WRITE_GUARD existing={existing}")

    plan = build_poses()
    plan["pins"] = pins
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

    out: dict = {"tool": TAG, "case": "g0b_d444", "prereg": "fg1_prereg.md",
                 "plan": {k: plan[k] for k in ("gates", "deviations", "pins", "env")},
                 "poses_in": plan["poses"]}

    # ---- stage ------------------------------------------------------------ #
    ctx = omni.usd.get_context()
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
    sup.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.17, -0.5))
    sup.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 1.0))
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
    cyl.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, Gf.Vec3d(0, 0, 0)))
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
    PhysxSchema.PhysxContactReportAPI.Apply(obj_prim).CreateThresholdAttr().Set(0.0)
    bind_material(obj_prim, "/World/materials/object_mat",
                  STATIC_FRICTION, DYNAMIC_FRICTION, RESTITUTION)

    grip = stage.DefinePrim(GRIPPER_ROOT)
    grip.GetReferences().AddReference(str(CASE_DIR / "fg1_gripper_only.usd"), "/fg1_gripper")
    for jaw in (JAW_FIXED, JAW_MOVING):
        PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(jaw)).CreateThresholdAttr().Set(0.0)

    # ---- asset gates (a)(b)(c) on the composed run stage ------------------ #
    pred = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    src = Usd.Stage.Open(str(ATTEMPT3_DIR / "roarm_m3.usd"))

    def hull_census(st, root_path):
        en = dis = 0
        for prim in Usd.PrimRange(st.GetPrimAtPath(root_path), pred):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                a = prim.GetAttribute("physics:collisionEnabled")
                if a.Get() if (a and a.Get() is not None) else True:
                    en += 1
                else:
                    dis += 1
        return en, dis

    gate_a = {p: hull_census(stage, p) for p in (JAW_FIXED, JAW_MOVING)}
    ok_a = all(v == (64, 1) for v in gate_a.values())

    JOINT_ATTRS = ["physics:axis", "physics:lowerLimit", "physics:upperLimit",
                   "physics:localPos0", "physics:localRot0", "physics:localPos1",
                   "physics:localRot1", "drive:angular:physics:targetVelocity",
                   "drive:angular:physics:stiffness", "drive:angular:physics:damping",
                   "drive:angular:physics:maxForce", "drive:angular:physics:type",
                   "physxJoint:maxJointVelocity", "physxJoint:armature",
                   "physxJoint:jointFriction"]

    def joint_snap(st, path):
        prim = st.GetPrimAtPath(path)
        return {n: repr(prim.GetAttribute(n).Get()) if prim.GetAttribute(n) else "<none>"
                for n in JOINT_ATTRS}

    js, jd = (joint_snap(src, "/roarm_m3/joints/link5_to_gripper_link"),
              joint_snap(stage, Q5_JOINT))
    gate_b_diffs = {k: (js[k], jd[k]) for k in js if js[k] != jd[k]}
    ok_b = not gate_b_diffs

    def mesh_hashes(st, root_path, strip):
        rows = {}
        for prim in Usd.PrimRange(st.GetPrimAtPath(root_path), pred):
            if prim.IsA(UsdGeom.Mesh):
                m = UsdGeom.Mesh(prim)
                h = hashlib.sha256()
                for attr, dtp in ((m.GetPointsAttr(), np.float64),
                                  (m.GetFaceVertexCountsAttr(), np.int64),
                                  (m.GetFaceVertexIndicesAttr(), np.int64)):
                    v = attr.Get()
                    h.update((np.array(v, dtype=dtp) if v is not None
                              else np.zeros(0, dtype=dtp)).tobytes())
                rows[prim.GetPath().pathString.replace(strip, "", 1)] = h.hexdigest()
        return rows

    n_mesh_diff = 0
    mesh_counts = {}
    for sp, dp in (("/roarm_m3/link5", JAW_FIXED), ("/roarm_m3/gripper_link", JAW_MOVING)):
        hs = mesh_hashes(src, sp, "/roarm_m3")
        hd = mesh_hashes(stage, dp, "/World/fg1_gripper")
        mesh_counts[sp] = (len(hs), len(hd))
        n_mesh_diff += sum(1 for k in set(hs) | set(hd) if hs.get(k) != hd.get(k))
    ok_c = n_mesh_diff == 0 and all(a == b and a >= 65 for a, b in mesh_counts.values())
    out["asset_gates"] = {
        "a_hull_census_enabled_disabled": {k: list(v) for k, v in gate_a.items()},
        "a_pass": ok_a,
        "b_q5_joint_snapshot": js, "b_diffs": gate_b_diffs, "b_pass": ok_b,
        "c_mesh_counts": {k: list(v) for k, v in mesh_counts.items()},
        "c_n_hash_diffs": n_mesh_diff, "c_pass": ok_c,
        "note": "attempt3 meshes are inline (internal references only); gate (c) is "
                "applied to inline points/faceVertexCounts/faceVertexIndices SHA-256",
    }
    if not (ok_a and ok_b and ok_c):
        raise RuntimeError("ASSET_GATE_FAIL see results asset_gates")
    print(f"[{LOG}] asset gates a/b/c PASS", flush=True)

    # ---- instrumentation --------------------------------------------------- #
    physx_iface = omni.physx.get_physx_interface()
    physx_sim = omni.physx.get_physx_simulation_interface()

    N = len(plan["poses"])
    pre_F = np.zeros((N, PREGRASP_STEPS, 2, 3))
    close_F = np.zeros((N, CLOSE_STEPS, 2, 3))
    support_F_close = np.zeros((N, CLOSE_STEPS, 3))
    hang_z = np.full((N, HANG_STEPS // HANG_CHUNK + 1), np.nan)
    hang_q5 = np.full((N, HANG_STEPS // HANG_CHUNK + 1), np.nan)
    spawn_check = np.full((N, 3), np.nan)
    raw_events = {"n_total": 0, "n_unknown_actor": 0, "pairs_seen": set()}

    ctxst = {"mode": "idle", "pose": -1, "step": -1, "micro": 0}

    def world_pose(path):
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
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

    step_sub = physx_iface.subscribe_physics_step_events(on_step)

    def on_contact(headers, data):
        pose_i, step_i, mode = ctxst["pose"], ctxst["step"], ctxst["mode"]
        for h in headers:
            raw_events["n_total"] += 1
            a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            pair = frozenset((a0, a1))
            raw_events["pairs_seen"].add((a0, a1))
            if not a0 or not a1:
                raw_events["n_unknown_actor"] += 1
                continue
            if OBJECT_PRIM not in pair:
                continue
            other = next(iter(pair - {OBJECT_PRIM}), None)
            imp = np.zeros(3)
            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                d = data[i]
                imp += np.array([d.impulse.x, d.impulse.y, d.impulse.z])
            if mode != "phases" or pose_i < 0:
                continue
            jaw_idx = {JAW_FIXED: 0, JAW_MOVING: 1}.get(other)
            if step_i < PREGRASP_STEPS:
                if jaw_idx is not None:
                    pre_F[pose_i, step_i, jaw_idx] += imp / DT
            elif step_i < PREGRASP_STEPS + CLOSE_STEPS:
                cs = step_i - PREGRASP_STEPS
                if jaw_idx is not None:
                    close_F[pose_i, cs, jaw_idx] += imp / DT
                elif other == SUPPORT_PRIM:
                    support_F_close[pose_i, cs] += imp / DT

    contact_sub = physx_sim.subscribe_contact_report_events(on_contact)

    mgr = GraspingManager()
    if not mgr.set_gripper(GRIPPER_ROOT):
        raise RuntimeError("SET_GRIPPER_FAIL")
    mgr.set_object_prim_path(OBJECT_PRIM)
    grasping_utils.apply_joint_pregrasp_states({Q5_JOINT: Q5_OPEN_DEG})
    park = Gf.Vec3d(*PARK_POS)
    idq = Gf.Quatd(1.0, Gf.Vec3d(0, 0, 0))
    mgr.set_gripper_pose(park, idq)
    mgr.store_initial_gripper_pose(park, idq)

    sup_enabled = UsdPhysics.CollisionAPI(sup.GetPrim()).GetCollisionEnabledAttr()
    if not sup_enabled:
        sup_enabled = UsdPhysics.CollisionAPI(sup.GetPrim()).CreateCollisionEnabledAttr(True)

    pose_snaps = [None] * N

    async def micro_check():
        ctxst.update(mode="micro", micro=0, step=-1)
        await grasping_utils.simulate_physics_async(5, DT, None, render=False)
        t_obj = world_pose(OBJECT_PRIM)[0]
        t_root = world_pose(GRIPPER_ROOT)[0]
        ok = (ctxst["micro"] == 5
              and float(np.linalg.norm(t_obj - OBJ_CENTER)) < 2e-3
              and float(np.linalg.norm(t_root - PARK_POS)) < 1e-6)
        out["harness_selfcheck"] = {
            "step_events_counted": ctxst["micro"], "expected": 5,
            "object_drift_m": float(np.linalg.norm(t_obj - OBJ_CENTER)),
            "root_drift_m": float(np.linalg.norm(t_root - PARK_POS)), "pass": ok}
        if not ok:
            raise RuntimeError(f"HARNESS_SELFCHECK_FAIL {out['harness_selfcheck']}")
        ctxst.update(mode="idle")

    async def hang_test(gi):
        snap = pose_snaps[gi]
        sup_enabled.Set(False)
        z_series, q5_series = [], []
        z0 = world_pose(OBJECT_PRIM)[0][2]
        z_series.append(z0)
        q5_prim = stage.GetPrimAtPath(Q5_JOINT)
        q5_series.append(grasping_utils.get_joint_state(q5_prim)[0])
        for _ in range(HANG_STEPS // HANG_CHUNK):
            await grasping_utils.simulate_physics_async(HANG_CHUNK, DT, None, render=False)
            z_series.append(world_pose(OBJECT_PRIM)[0][2])
            q5_series.append(grasping_utils.get_joint_state(q5_prim)[0])
        sup_enabled.Set(True)
        hang_z[gi, :len(z_series)] = z_series
        hang_q5[gi, :len(q5_series)] = q5_series
        t_root_end = world_pose(GRIPPER_ROOT)[0]
        snap["hang"] = {
            "z_start_m": float(z_series[0]), "z_end_m": float(z_series[-1]),
            "drop_m": float(z_series[0] - z_series[-1]),
            "q5_end_deg": float(q5_series[-1]),
            "root_drift_end_m": float(np.linalg.norm(t_root_end - np.array(snap["target_pos"]))),
        }

    def make_progress(offset):
        async def progress(idx):
            gi = offset + idx - 1
            ctxst.update(mode="hang")
            pose = plan["poses"][gi]
            t_root, q_root = world_pose(GRIPPER_ROOT)
            t_l5, _ = world_pose(JAW_FIXED)
            t_gl, q_gl = world_pose(JAW_MOVING)
            t_obj, q_obj = world_pose(OBJECT_PRIM)
            q5_pos = grasping_utils.get_joint_state(stage.GetPrimAtPath(Q5_JOINT))[0]
            pose_snaps[gi] = {
                "target_pos": pose["pos"], "target_quat_wxyz": pose["quat_wxyz"],
                "post_close": {
                    "root_pos": t_root.tolist(), "root_quat_wxyz": q_root.tolist(),
                    "link5_pos": t_l5.tolist(), "moving_pos": t_gl.tolist(),
                    "moving_quat_wxyz": q_gl.tolist(),
                    "obj_pos": t_obj.tolist(), "obj_quat_wxyz": q_obj.tolist(),
                    "q5_deg": float(q5_pos),
                    "root_target_err_m": float(np.linalg.norm(t_root - np.array(pose["pos"]))),
                    "link5_root_err_m": float(np.linalg.norm(t_l5 - t_root)),
                }}
            await hang_test(gi)
            ctxst.update(mode="idle", pose=-1, step=-1)
        return progress

    async def run_groups():
        await micro_check()
        groups = []
        i = 0
        while i < N:
            j = i
            while j < N and plan["poses"][j]["close_deg"] == plan["poses"][i]["close_deg"]:
                j += 1
            groups.append((i, j))
            i = j
        out["groups"] = [{"range": [a, b], "close_deg": plan["poses"][a]["close_deg"]}
                         for a, b in groups]
        for a, b in groups:
            t_obj = world_pose(OBJECT_PRIM)[0]
            if float(np.linalg.norm(t_obj - OBJ_CENTER)) > 1e-4:
                raise RuntimeError(f"OBJECT_NOT_AT_SPAWN_BEFORE_GROUP {a} {t_obj}")
            close_deg = plan["poses"][a]["close_deg"]
            mgr.grasp_phases = [
                GraspPhase("PREGRASP", {Q5_JOINT: Q5_OPEN_DEG}, PREGRASP_STEPS, DT),
                GraspPhase("CLOSE", {Q5_JOINT: close_deg}, CLOSE_STEPS, DT),
            ]
            poses = []
            for k in range(a, b):
                p = plan["poses"][k]
                q = p["quat_wxyz"]
                poses.append((Gf.Vec3d(*p["pos"]),
                              Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3]))))

            orig_eval = mgr.evaluate_grasp_pose

            async def eval_with_ctx(loc, quat, **kw):
                gi = next(k for k in range(a, b)
                          if np.allclose(plan["poses"][k]["pos"],
                                         [loc[0], loc[1], loc[2]], atol=1e-12))
                ctxst.update(mode="phases", pose=gi, step=-1)
                await orig_eval(loc, quat, **kw)
            mgr.evaluate_grasp_pose = eval_with_ctx
            try:
                await mgr.evaluate_grasp_poses(
                    poses, render=False, physics_scene_path=None,
                    isolate_simulation=False, progress_callback=make_progress(a))
            finally:
                mgr.evaluate_grasp_pose = orig_eval

    task = asyncio.ensure_future(run_groups())
    app = omni.kit.app.get_app()
    guard = 0
    while not task.done():
        app.update()
        guard += 1
        if guard > 400000:
            raise RuntimeError("EVENT_LOOP_GUARD")
    if task.exception() is not None:
        raise task.exception()

    # ---- gates + taxonomy -------------------------------------------------- #
    preF_mag = np.linalg.norm(pre_F, axis=3)
    closeF_mag = np.linalg.norm(close_F, axis=3)
    bilateral_min = closeF_mag.min(axis=2)
    verdicts = []
    n_success = 0
    for gi in range(N):
        snap = pose_snaps[gi]
        close_bilateral = bool((bilateral_min[gi] > BILATERAL_GATE_N).any())
        drop = snap["hang"]["drop_m"]
        hold = bool(drop < HOLD_DROP_GATE_M and np.isfinite(drop))
        success = close_bilateral and hold
        preclose = bool((preF_mag[gi] > BILATERAL_GATE_N).any())
        fixed_contact = bool((closeF_mag[gi, :, 0] > BILATERAL_GATE_N).any())
        moving_contact = bool((closeF_mag[gi, :, 1] > BILATERAL_GATE_N).any())
        if success:
            taxonomy = "SUCCESS"
        elif close_bilateral:
            taxonomy = "BILATERAL_NO_HOLD"
        elif preclose:
            taxonomy = "PRECLOSE_COLLISION"
        elif fixed_contact and not moving_contact:
            taxonomy = "ONE_JAW_ONLY_FIXED"
        elif moving_contact and not fixed_contact:
            taxonomy = "ONE_JAW_ONLY_MOVING"
        else:
            taxonomy = "NO_JAW_CONTACT"
        measurement_valid = bool(
            snap["post_close"]["root_target_err_m"] < 1e-6
            and snap["post_close"]["link5_root_err_m"] < 1e-6
            and snap["hang"]["root_drift_end_m"] < 1e-6
            and (np.isnan(spawn_check[gi]).any()
                 or float(np.linalg.norm(spawn_check[gi] - OBJ_CENTER)) < 2e-3))
        n_success += int(success)
        row = {
            "index": gi, "label": plan["poses"][gi]["label"],
            "kind": plan["poses"][gi]["kind"],
            "close_target_deg": plan["poses"][gi]["close_deg"],
            "close_bilateral": close_bilateral,
            "close_bilateral_peak_n": float(bilateral_min[gi].max()),
            "close_fixed_peak_n": float(closeF_mag[gi, :, 0].max()),
            "close_moving_peak_n": float(closeF_mag[gi, :, 1].max()),
            "preclose_jaw_contact": preclose,
            "preclose_peak_n": float(preF_mag[gi].max()),
            "hang_drop_m": drop, "hold": hold, "success": success,
            "taxonomy": taxonomy, "measurement_valid": measurement_valid,
            "post_close_q5_deg": snap["post_close"]["q5_deg"],
            "hang_end_q5_deg": snap["hang"]["q5_end_deg"],
            "root_target_err_m": snap["post_close"]["root_target_err_m"],
            "spawn_check_drift_m": (None if np.isnan(spawn_check[gi]).any() else
                                    float(np.linalg.norm(spawn_check[gi] - OBJ_CENTER))),
        }
        verdicts.append(row)
        print(f"[{LOG}] pose {gi:02d} {row['label']}: {taxonomy} "
              f"bilateral_peak={row['close_bilateral_peak_n']:.4f}N "
              f"drop={drop * 1000:.3f}mm valid={measurement_valid}", flush=True)

    all_valid = all(r["measurement_valid"] for r in verdicts)
    if n_success == 0:
        code = "FG1_ALL_13_FAIL_GRIPPER_GEOMETRY_BOTTLENECK_SUPPORTED"
    else:
        code = "FG1_SOME_SUCCESS_ARM_POSE_TRAJECTORY_BOTTLENECK_SUPPORTED"
    out["rows"] = verdicts
    out["verdict"] = {
        "code": code, "n_success": n_success, "n_poses": N,
        "all_measurements_valid": all_valid,
        "gates": {"bilateral_n": BILATERAL_GATE_N, "hold_drop_m": HOLD_DROP_GATE_M},
        "branch_semantics": {
            "all_fail": "bottleneck = gripper geometry (prereg SS1 branch i)",
            "any_success": "bottleneck = arm/pose/trajectory (prereg SS1 branch ii)"},
        "non_claims": "no claim about real-robot grasping, IK reachability, friction "
                      "realism, or D419 protocol judgment (prereg SS1)"}
    out["raw_contact_stats"] = {
        "n_total_events": raw_events["n_total"],
        "n_unknown_actor": raw_events["n_unknown_actor"],
        "pairs_seen": sorted(str(p) for p in raw_events["pairs_seen"])}
    out["pose_snaps"] = pose_snaps

    # ---- trace ------------------------------------------------------------- #
    np.savez(
        OUT["trace.npz"],
        pre_F=pre_F, close_F=close_F, support_F_close=support_F_close,
        hang_z=hang_z, hang_q5=hang_q5, spawn_check=spawn_check,
        bilateral_min=bilateral_min,
        pose_labels=np.array([p["label"] for p in plan["poses"]]),
        close_targets=np.array([p["close_deg"] for p in plan["poses"]]),
        target_pos=np.array([p["pos"] for p in plan["poses"]]),
        target_quat=np.array([p["quat_wxyz"] for p in plan["poses"]]))
    with open(OUT["trace.npz"], "rb") as f:
        os.fsync(f.fileno())

    # ---- rerun (D341, save-only) ------------------------------------------- #
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    app_id = f"roarm_g0b_d444_{TAG}"

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

    # subsampled jaw visual point clouds in body frame (from the run stage)
    def jaw_points(body_path):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(body_path + "/visuals"), pred):
            if prim.IsA(UsdGeom.Mesh):
                pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                return pts[::40]
        raise RuntimeError(f"NO_VISUAL_MESH {body_path}")

    l5_pts = jaw_points(JAW_FIXED)
    gl_pts = jaw_points(JAW_MOVING)

    def to_world(pts, pos, quat_wxyz):
        w, x, y, z = quat_wxyz
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return pts @ R.T + np.asarray(pos)

    with rr.RecordingStream(app_id, recording_id=f"g0b_d444_{TAG}", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(OUT["timeline.rrd"]), write_footer=True)
        rec.log("scene/support", rr.LineStrips3D(
            box_wire([0.4, 0.17, -0.5], [1.0, 1.0, 0.5]),
            colors=[[120, 120, 120]], radii=0.0008), static=True)
        rec.log("scene/object_spawn", rr.LineStrips3D(
            cyl_wire(OBJ_CENTER), colors=[[80, 200, 120]], radii=0.0006), static=True)

        gstep = 0
        for gi in range(N):
            snap = pose_snaps[gi]
            rec.reset_time()
            rec.set_time("pose_index", sequence=gi)
            pc = snap["post_close"]
            rec.log("gripper/link5_points", rr.Points3D(
                to_world(l5_pts, pc["link5_pos"], pc["root_quat_wxyz"]),
                colors=[150, 150, 160], radii=0.0006))
            rec.log("gripper/gripper_link_points", rr.Points3D(
                to_world(gl_pts, pc["moving_pos"], pc["moving_quat_wxyz"]),
                colors=[70, 130, 230], radii=0.0006))
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
                    rr.Scalars(float(verdicts[gi]["hang_drop_m"] * 1000.0)))
            row = verdicts[gi]
            rec.log("events/verdict", rr.TextLog(
                f"pose {gi:02d} {row['label']}: {row['taxonomy']} "
                f"bilateral_peak={row['close_bilateral_peak_n']:.4f}N "
                f"drop={row['hang_drop_m'] * 1000:.2f}mm",
                level=rr.TextLogLevel.INFO if row["success"] else rr.TextLogLevel.WARN))

            for phase, arr, nst in (("PREGRASP", preF_mag[gi], PREGRASP_STEPS),
                                    ("CLOSE", closeF_mag[gi], CLOSE_STEPS)):
                rec.reset_time()
                rec.set_time("pose_index", sequence=gi)
                rec.set_time("global_step", sequence=gstep)
                rec.log("events/phase", rr.TextLog(f"pose {gi:02d} {phase} begin",
                                                   level=rr.TextLogLevel.INFO))
                for s in range(nst):
                    rec.reset_time()
                    rec.set_time("pose_index", sequence=gi)
                    rec.set_time("global_step", sequence=gstep)
                    rec.log("forces/fixed_n", rr.Scalars(float(arr[s, 0])))
                    rec.log("forces/moving_n", rr.Scalars(float(arr[s, 1])))
                    rec.log("forces/bilateral_min_n", rr.Scalars(float(arr[s].min())))
                    gstep += 1
            rec.reset_time()
            rec.set_time("pose_index", sequence=gi)
            rec.set_time("global_step", sequence=gstep)
            rec.log("events/phase", rr.TextLog(f"pose {gi:02d} HANG begin",
                                               level=rr.TextLogLevel.INFO))
            zrow = hang_z[gi]
            for k in range(zrow.shape[0]):
                if np.isnan(zrow[k]):
                    continue
                rec.reset_time()
                rec.set_time("pose_index", sequence=gi)
                rec.set_time("global_step", sequence=gstep + k * HANG_CHUNK)
                rec.log("plots/hang_z_m", rr.Scalars(float(zrow[k])))
            gstep += HANG_STEPS

        rec.reset_time()
        rec.set_time("pose_index", sequence=0)
        summary_md = (
            f"# g0b_d444 {TAG} — flying-gripper bottleneck isolation\n\n"
            f"**VERDICT: {code}** ({n_success}/{N} SUCCESS)\n\n"
            f"13 poses (8 sdg2 side + 5 n8b rim-tilt), frozen attempt3 gripper as a "
            f"fixed-root articulation, no arm.  SUCCESS = same-step bilateral close "
            f"force > {BILATERAL_GATE_N} N AND hang drop < {HOLD_DROP_GATE_M * 1000:.0f} mm "
            f"(support collider disabled, {HANG_STEPS} steps).\n\n"
            f"Grey/blue points = fixed/moving jaw at post-CLOSE; red = object at "
            f"post-CLOSE; green = spawn; bright/pale axes = root target/actual.\n\n"
            f"Authority = fg1_results.json + fg1_trace.npz; Rerun is inspection "
            f"evidence only (D341).")
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN),
                static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run",
                                         name="1 | verdict"),
                    rrb.Spatial3DView(origin="/", contents=["/scene/**", "/gripper/**",
                                                            "/object/**"],
                                      name="2 | post-close geometry per pose"),
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
        "gripper/link5_points", "gripper/gripper_link_points",
        "gripper/root_target_axes", "gripper/root_actual_axes", "object/post_close",
        "plots/hang_drop_mm", "plots/hang_z_m",
        "forces/fixed_n", "forces/moving_n", "forces/bilateral_min_n",
        "events/phase", "events/verdict"]
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    lin3 = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    components = {
        "metadata/run": ["TextDocument:text"],
        "scene/support": lin3, "scene/object_spawn": lin3,
        "gripper/link5_points": pts3, "gripper/gripper_link_points": pts3,
        "gripper/root_target_axes": lin3, "gripper/root_actual_axes": lin3,
        "object/post_close": lin3,
        "plots/hang_drop_mm": ["Scalars:scalars"], "plots/hang_z_m": ["Scalars:scalars"],
        "forces/fixed_n": ["Scalars:scalars"], "forces/moving_n": ["Scalars:scalars"],
        "forces/bilateral_min_n": ["Scalars:scalars"],
        "events/phase": ["TextLog:text", "TextLog:level"],
        "events/verdict": ["TextLog:text", "TextLog:level"],
    }
    validation = validate_rerun_artifact(
        OUT["timeline.rrd"],
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time", "pose_index", "global_step"],
        expected_entity_components=components,
        blueprint_path=OUT["timeline.rbl"],
        screenshot_path=OUT["inspection.png"],
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=240.0,
    )
    fsync_write(OUT["rerun_validation.json"], json.dumps(validation, indent=2, default=str) + "\n")
    out["rerun_validation_pass"] = bool(validation.get("pass"))
    print(f"[{LOG}] rerun_validation pass={validation.get('pass')} "
          f"errors={validation.get('errors')}", flush=True)

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
    print(f"[{LOG}] G0B_FG1_VERDICT={code} n_success={n_success}/13 "
          f"all_valid={all_valid}", flush=True)
    return 0


def main() -> int:
    plan = preflight()
    fsync_write(OUT["argv.txt"], " ".join([sys.executable, *sys.argv]) + "\n")
    shutil.copyfile(__file__, OUT["script.py.txt"])
    print(f"[{LOG}] preflight OK: 13 poses, pins verified, tag locked", flush=True)
    return run(plan)


if __name__ == "__main__":
    sys.exit(main())
