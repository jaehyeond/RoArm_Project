#!/usr/bin/env python3
"""p20 / ba1 — g0d_d448 B601 full-arm side grasp probe + RTX mp4.

Contract: claudedocs/runtime_logs/grasp_track/g0d_d448/ba1_prereg.md (61st session,
user-approved case; the user's request explicitly lifts the D324 trajectory-video
ban for this one mp4 + keyframes).

New variables (exactly two): full arm attached to a world-fixed base (flying
gripper removed), and arm trajectory control (offline numeric IK waypoints +
min-jerk joint interpolation + PD drives with official maxForce).  Everything
else inherited from bg1: same D29xH50 cylinder, same split2 finger collision
(ported as run-stage overrides from bg1_gripper_split2.usd), same bilateral
contact gate constants.

Single side-grasp pose (phi=135 deg), phases SETTLE 30 / APPROACH 120 /
SETTLE2 30 / CLOSE 120 / LIFT 120 / HOLD 120 at dt 1/60 (540 steps, 9 s).
SUCCESS = G1 close bilateral > 0.01 N AND G2 object follows glf rise within
6 mm (glf rise >= 60 mm) AND G3 relative slip < 6 mm.

Authority = ba1_results.json + ba1_trace.npz; the mp4/keyframes are the
user-facing visual evidence layer and must always be captioned with G1-G3
numbers (bg1v discipline).  RRD is mandatory (trajectory verdict, D341).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0d_d448"
G0C_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0c_d446"
ASSET_DIR = G0C_DIR / "b601_asset"
TAG = "ba1"
LOG = "g0d_ba1"

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
SPLIT2_USD = G0C_DIR / "bg1_gripper_split2.usd"
SPLIT2_AUDIT = G0C_DIR / "bg1_split2_audit.json"
NUMPY_PIN = "1.26.0"
PSUTIL_PIN = "5.9.8"
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"
FFMPEG = "/home/cgxr/.local/bin/ffmpeg"
FRAMES_DIR = Path("/tmp/claude-1000/-home-cgxr-Documents-Robotics-RoArm-Project/"
                  "cee00159-bcaf-4005-98c9-8f519c5473ce/scratchpad/ba1_frames")

# ---- physical contract (bg1 verbatim where shared) -------------------------- #
OBJ_RADIUS_M = 0.0145
OBJ_HEIGHT_M = 0.050
OBJ_MASS_KG = 0.02483
STATIC_FRICTION = 0.40
DYNAMIC_FRICTION = 0.30
RESTITUTION = 0.0
SUPPORT_FRICTION = 1.0
GRAVITY = 9.81
DT = 1.0 / 60.0

OBJ_CENTER = np.array([0.34, 0.0, 0.12])
PED_TOP = 0.095
PED_HALF_XY = 0.025
GROUND_TOP = 0.0

PHI_DEG = 225.0
X_TCP = -0.02448
STANDOFF_M = 0.05
LIFT_H = 0.08
OPEN_M = 0.0715
CLOSE_M = 0.0

ARM_STIFF = 1e5
ARM_DAMP = 500.0
FING_STIFF = 5e3
FING_DAMP = 2e2

BILATERAL_GATE_N = 0.01
FOLLOW_GATE_M = 0.006
SLIP_GATE_M = 0.006
TRACK_GATE_M = 0.010
GLF_LIFT_MIN_M = 0.06

PHASES = [("SETTLE", 30), ("APPROACH", 120), ("SETTLE2", 30),
          ("CLOSE", 120), ("LIFT", 120), ("HOLD", 120)]
N_STEPS = sum(n for _, n in PHASES)  # 540
CAPTURE_EVERY = 2
FPS = 30
RES_W, RES_H = 1920, 1080
RT_SUB = 8
RT_SUB_KEY = 32
FOCAL_MM = 18.0
H_APERTURE_MM = 20.955
V_APERTURE_MM = H_APERTURE_MM * RES_H / RES_W
CAM_EYE = np.array([1.00, -0.80, 0.52])
CAM_TGT = np.array([0.22, -0.02, 0.12])
WARMUP_UPDATES = 40

# design-time IK solutions (prereg SS4 REV-1; preflight recomputation must match)
Q_DESIGN = {
    "standoff": [-37.52463916441598, -119.2375801126596, -20.19479253171161,
                 -99.04604768892128, -82.52463915549, 0.0028670807161452153],
    "grasp": [-26.26634238106853, -119.59484634978143, -22.855328532667208,
              -96.74077941377394, -71.26634237621056, 0.0009111873687887888],
    "lift": [-26.266409776048516, -92.18325927288676, -25.461539058766494,
             -66.72298182564933, -71.2664097735052, 0.0009112015934386221],
}

# kinematic model pin (payloads/Physics/physics.usda verbatim; runtime-gated)
JOINTS_TBL = [
    ("joint1", (-0.00008416, 0, 0.08465), (1, 0, 0, 0), (1, 0, 0, 0),
     -160.42818, 160.42818, 27.0),
    ("joint2", (0.020084, 0.031625, 0.05555), (0.7071081, 0.70710546, 0, 0),
     (0, -1, 0, 0), -179.90875, 0.0, 27.0),
    ("joint3", (-0.264, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0),
     -179.90875, 0.0, 27.0),
    ("joint4", (0.2426, -0.054, -0.001625), (1, 0, 0, 0), (1, 0, 0, 0),
     -107.143105, 89.95438, 7.0),
    ("joint5", (0.078308, -0.0375, -0.03), (0.70710546, -0.7071081, 0, 0),
     (1, 0, 0, 0), -89.95438, 89.95438, 7.0),
    ("joint6", (0.023692, 0, 0.04),
     (0.70710546, -3.3148634e-17, 0.7071081, 3.7116984e-17), (1, 0, 0, 0),
     -179.90875, 179.90875, 7.0),
]
GLF_FIX = ((2.7755576e-17, 0, 0.15971),
           (0.0000025977158, -0.70710546, -0.0000025977254, -0.7071081))
LIMITS = np.array([[j[4], j[5]] for j in JOINTS_TBL])
BLADE_Z_HALF = 0.0196
PALM_X_NEAR = 0.073

ROBOT = "/World/ba1_robot"
GEOM = ROBOT + "/Geometry"
LINK_REL = ["base_link", "base_link/link1", "base_link/link1/link2",
            "base_link/link1/link2/link3", "base_link/link1/link2/link3/link4",
            "base_link/link1/link2/link3/link4/link5",
            "base_link/link1/link2/link3/link4/link5/link6",
            "base_link/link1/link2/link3/link4/link5/link6/gripper_link",
            "base_link/link1/link2/link3/link4/link5/link6/gripper_link/gripper_left",
            "base_link/link1/link2/link3/link4/link5/link6/gripper_link/gripper_right"]
BODY = {p.split("/")[-1]: GEOM + "/" + p for p in LINK_REL}
JROOT = ROBOT + "/Physics/"
OBJECT_PRIM = "/World/object"
PED_PRIM = "/World/pedestal"
GROUND_PRIM = "/World/ground"
CAMERA_PRIM = "/World/camera"

SPLIT_SRC = {("gripper_left", "blade"): "/bg1_gripper/gripper_left/collision_blade_split2",
             ("gripper_left", "mount"): "/bg1_gripper/gripper_left/collision_mount_split2",
             ("gripper_right", "blade"): "/bg1_gripper/gripper_right/collision_blade_split2",
             ("gripper_right", "mount"): "/bg1_gripper/gripper_right/collision_mount_split2"}

KEYFRAMES = {29: "settle", 149: "approach", 179: "pregrasp",
             299: "close", 419: "lift", 539: "hold"}

OUT = {k: CASE_DIR / f"{TAG}_{k}" for k in (
    "results.json", "trace.npz", "timeline.rrd", "timeline.rbl",
    "rerun_validation.json", "inspection.png", "side_grasp.mp4",
    "exit_status.txt", "script.py.txt", "argv.txt", "failure.json")}
OUT_KEY = {name: CASE_DIR / f"{TAG}_key_{name}.png" for name in KEYFRAMES.values()}


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


# ---- kinematics (design-scan lineage, quaternion error, Shepperd) ----------- #
def quat_to_R(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
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
    return q / np.linalg.norm(q)


def Tm(pos, quat):
    M = np.eye(4)
    M[:3, :3] = quat_to_R(quat)
    M[:3, 3] = pos
    return M


def Rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    M = np.eye(4)
    M[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return M


PRE = [Tm(j[1], j[2]) for j in JOINTS_TBL]
POST = [np.linalg.inv(Tm((0, 0, 0), j[3])) for j in JOINTS_TBL]
GLF_T = Tm(*GLF_FIX)


def fk_glf(q_deg):
    M = np.eye(4)
    for i in range(6):
        M = M @ PRE[i] @ Rz(q_deg[i]) @ POST[i]
    return M @ GLF_T


def pose_err(M, target_pos, target_R):
    ep = M[:3, 3] - target_pos
    q = R_to_quat(target_R.T @ M[:3, :3])
    sgn = 1.0 if q[0] >= 0.0 else -1.0
    return np.concatenate([ep, -2.0 * sgn * q[1:]])


def ik(target_pos, target_R, seed, iters=150, lam=1e-6):
    q = np.array(seed, dtype=float)
    try:
        for _ in range(iters):
            e = pose_err(fk_glf(q), target_pos, target_R)
            if np.linalg.norm(e[:3]) < 1e-9 and np.linalg.norm(e[3:]) < 1e-9:
                break
            J = np.zeros((6, 6))
            h = 1e-4
            for j in range(6):
                qp = q.copy()
                qp[j] += h
                J[:, j] = (pose_err(fk_glf(qp), target_pos, target_R) - e) / h
            dq = np.linalg.solve(J.T @ J + lam * np.eye(6), -J.T @ e)
            step = min(1.0, 30.0 / max(1e-9, float(np.abs(dq).max())))
            q = np.clip(q + step * dq, LIMITS[:, 0], LIMITS[:, 1])
    except np.linalg.LinAlgError:
        return q, 1e9, 1e9
    e = pose_err(fk_glf(q), target_pos, target_R)
    return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))


def side_R(phi_deg):
    phi = math.radians(phi_deg)
    x = np.array([-math.cos(phi), -math.sin(phi), 0.0])
    y = np.array([math.sin(phi), -math.cos(phi), 0.0])
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def min_jerk(tau):
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def build_plan() -> dict:
    R_t = side_R(PHI_DEG)
    ortho = float(np.abs(R_t @ R_t.T - np.eye(3)).max())
    det = abs(float(np.linalg.det(R_t)) - 1.0)
    if max(ortho, det) > 1e-12:
        raise RuntimeError(f"POSE_CONSTRUCTION_GATE {ortho} {det}")
    grasp_pos = OBJ_CENTER - R_t @ np.array([X_TCP, 0.0, 0.0])
    # bg1 3-2: approach axis (wrist->tips) = +x_glf; standoff retreats toward
    # the base along -approach (REV-1 - the try-3 abort came from the sign
    # error that placed the standoff beyond the object, palm over the pedestal)
    approach = R_t[:, 0]
    targets = {"standoff": grasp_pos - approach * STANDOFF_M,
               "grasp": grasp_pos,
               "lift": grasp_pos + np.array([0.0, 0.0, LIFT_H])}

    # clearance gates (analytic, prereg SS4 REV-1)
    blade_lo = grasp_pos[2] - BLADE_Z_HALF
    if not blade_lo > PED_TOP + 0.003:
        raise RuntimeError(f"CLEARANCE_BLADE_PED {blade_lo}")
    # palm band starts PALM_X_NEAR behind glf origin = 0.0485 m behind C
    palm_near_from_c = PALM_X_NEAR - abs(X_TCP)
    palm_clear = palm_near_from_c - PED_HALF_XY * math.sqrt(2.0)
    if not palm_clear > 0.008:
        raise RuntimeError(f"CLEARANCE_PALM_PED {palm_clear}")

    qs = {}
    ik_rows = {}
    for name in ("standoff", "grasp", "lift"):
        q, ep, eo = ik(targets[name], R_t, Q_DESIGN[name])
        margin = float(np.minimum(q - LIMITS[:, 0], LIMITS[:, 1] - q).min())
        dq = float(np.abs(q - np.array(Q_DESIGN[name])).max())
        ik_rows[name] = {"q_deg": q.tolist(), "pos_err_m": ep, "ori_err": eo,
                         "limit_margin_deg": margin, "dev_from_design_deg": dq}
        if not (ep < 1e-6 and eo < 1e-6 and margin > 5.0 and dq < 0.1):
            raise RuntimeError(f"IK_GATE {name} {ik_rows[name]}")
        qs[name] = q

    # per-step command table (540 x 8): arm min-jerk between waypoints
    q_cmd = np.zeros((N_STEPS, 8))
    seg = {"SETTLE": (qs["standoff"], qs["standoff"]),
           "APPROACH": (qs["standoff"], qs["grasp"]),
           "SETTLE2": (qs["grasp"], qs["grasp"]),
           "CLOSE": (qs["grasp"], qs["grasp"]),
           "LIFT": (qs["grasp"], qs["lift"]),
           "HOLD": (qs["lift"], qs["lift"])}
    fing = {"SETTLE": OPEN_M, "APPROACH": OPEN_M, "SETTLE2": OPEN_M,
            "CLOSE": CLOSE_M, "LIFT": CLOSE_M, "HOLD": CLOSE_M}
    phase_idx = np.zeros(N_STEPS, dtype=np.int64)
    t = 0
    for pi, (pname, n) in enumerate(PHASES):
        a, b = seg[pname]
        for k in range(n):
            s = min_jerk((k + 1) / n)
            q_cmd[t, :6] = a + s * (b - a)
            q_cmd[t, 6] = fing[pname]
            q_cmd[t, 7] = fing[pname]
            phase_idx[t] = pi
            t += 1
    if t != N_STEPS:
        raise RuntimeError("PHASE_TABLE_LEN")

    return {"R_target": R_t.tolist(), "targets": {k: v.tolist() for k, v in targets.items()},
            "approach_dir": approach.tolist(), "ik": ik_rows,
            "q_cmd": q_cmd, "phase_idx": phase_idx,
            "clearance": {"blade_lo_z_m": blade_lo, "palm_clear_m": palm_clear}}


def preflight() -> dict:
    import numpy
    import psutil
    if numpy.__version__ != NUMPY_PIN or psutil.__version__ != PSUTIL_PIN:
        raise RuntimeError(f"ENV_PIN numpy={numpy.__version__} psutil={psutil.__version__}")
    pins = {}
    for path, want in SRC_PINS.items():
        got = sha256(path)
        pins[str(path.relative_to(G0C_DIR))] = {"sha256": got, "match": got == want}
        if got != want:
            raise RuntimeError(f"SHA_DRIFT {path}")
    audit = json.loads(SPLIT2_AUDIT.read_text())
    split2_sha = sha256(SPLIT2_USD)
    if split2_sha != audit["out"]["sha256"]:
        raise RuntimeError(f"SPLIT2_SHA_DRIFT {split2_sha}")
    pins["bg1_gripper_split2.usd"] = {"sha256": split2_sha, "match": True}
    blade_pins = {
        "gripper_left": audit["gates"]["gripper_left"]["blade_inner_extreme_glf_m"],
        "gripper_right": audit["gates"]["gripper_right"]["blade_inner_extreme_glf_m"]}

    if not Path(FFMPEG).exists():
        raise RuntimeError("FFMPEG_MISSING")

    existing = [p.name for k, p in OUT.items()
                if p.exists() and k not in ("script.py.txt", "argv.txt")]
    existing += [p.name for p in OUT_KEY.values() if p.exists()]
    if existing:
        raise RuntimeError(f"WRITE_GUARD existing={existing}")

    plan = build_plan()
    plan["pins"] = pins
    plan["blade_extreme_pins"] = blade_pins
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
        rc = run_inner(plan, app, t_start)
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


def run_inner(plan: dict, app, t_start: float) -> int:
    import faulthandler
    faulthandler.dump_traceback_later(1800, exit=True)
    import omni.physx
    import omni.usd
    from pxr import (Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom,
                     UsdLux, UsdPhysics, UsdShade)
    from isaacsim.core.utils.extensions import enable_extension
    print(f"[{LOG}] imports base OK", flush=True)
    # smoke-proven order: replicator.core first, app.update() pumps between
    # enables (double-enable without pumps deadlocked the 61st smoke probe)
    enable_extension("omni.replicator.core")
    for _ in range(3):
        app.update()
    import omni.replicator.core as rep
    from PIL import Image
    print(f"[{LOG}] replicator + PIL OK", flush=True)
    enable_extension("isaacsim.replicator.grasping")
    for _ in range(3):
        app.update()
    from isaacsim.replicator.grasping import grasping_utils
    print(f"[{LOG}] grasping utils OK", flush=True)
    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        raise RuntimeError(f"RERUN_VERSION {rr.__version__} != {RERUN_VERSION}")

    out: dict = {"tool": TAG, "case": "g0d_d448", "prereg": "ba1_prereg.md",
                 "plan": {k: plan[k] for k in ("R_target", "targets", "approach_dir",
                                               "ik", "clearance", "pins",
                                               "blade_extreme_pins", "env")}}
    pred_prox = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)

    # ---- split2 source extraction (bit-copy payload) ----------------------- #
    split_stage = Usd.Stage.Open(str(SPLIT2_USD))
    split_data = {}
    for (finger, part), src_path in SPLIT_SRC.items():
        prim = split_stage.GetPrimAtPath(src_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"SPLIT_SRC_MISSING {src_path}")
        if UsdGeom.Xformable(prim).GetOrderedXformOps():
            raise RuntimeError(f"SPLIT_SRC_HAS_XFORM {src_path}")
        m = UsdGeom.Mesh(prim)
        pts = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
        fvc = np.array(m.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        fvi = np.array(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        h = hashlib.sha256()
        for arr in (pts, fvc, fvi):
            h.update(arr.tobytes())
        split_data[(finger, part)] = {"pts": pts, "fvc": fvc, "fvi": fvi,
                                      "sha": h.hexdigest()}
    out["split_port_source"] = {f"{f}/{p}": d["sha"] for (f, p), d in split_data.items()}

    # ---- stage ------------------------------------------------------------- #
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(GRAVITY)
    ps_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    ps_api.CreateUpdateTypeAttr().Set(PhysxSchema.Tokens.Disabled)
    scene_int = PhysicsSchemaTools.sdfPathToInt("/World/physicsScene")

    def bind_material(prim, path, sf, df, rest):
        mat = UsdShade.Material.Define(stage, path)
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr().Set(sf)
        pm.CreateDynamicFrictionAttr().Set(df)
        pm.CreateRestitutionAttr().Set(rest)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    gnd = UsdGeom.Cube.Define(stage, GROUND_PRIM)
    gnd.CreateSizeAttr().Set(1.0)
    gnd.AddTranslateOp().Set(Gf.Vec3d(0.3, 0.0, GROUND_TOP - 0.01))
    gnd.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.02))
    UsdPhysics.CollisionAPI.Apply(gnd.GetPrim())
    PhysxSchema.PhysxContactReportAPI.Apply(gnd.GetPrim()).CreateThresholdAttr().Set(0.0)
    gnd.CreateDisplayColorAttr([(0.32, 0.33, 0.36)])
    bind_material(gnd.GetPrim(), "/World/materials/ground_mat",
                  SUPPORT_FRICTION, SUPPORT_FRICTION, RESTITUTION)

    ped = UsdGeom.Cube.Define(stage, PED_PRIM)
    ped.CreateSizeAttr().Set(1.0)
    ped.AddTranslateOp().Set(Gf.Vec3d(OBJ_CENTER[0], OBJ_CENTER[1], PED_TOP / 2))
    ped.AddScaleOp().Set(Gf.Vec3f(PED_HALF_XY * 2, PED_HALF_XY * 2, PED_TOP))
    UsdPhysics.CollisionAPI.Apply(ped.GetPrim())
    PhysxSchema.PhysxContactReportAPI.Apply(ped.GetPrim()).CreateThresholdAttr().Set(0.0)
    ped.CreateDisplayColorAttr([(0.42, 0.43, 0.46)])
    bind_material(ped.GetPrim(), "/World/materials/ped_mat",
                  SUPPORT_FRICTION, SUPPORT_FRICTION, RESTITUTION)

    cyl = UsdGeom.Cylinder.Define(stage, OBJECT_PRIM)
    cyl.CreateRadiusAttr().Set(OBJ_RADIUS_M)
    cyl.CreateHeightAttr().Set(OBJ_HEIGHT_M)
    cyl.CreateAxisAttr().Set("Z")
    cyl.CreateExtentAttr().Set([(-OBJ_RADIUS_M, -OBJ_RADIUS_M, -OBJ_HEIGHT_M / 2),
                                (OBJ_RADIUS_M, OBJ_RADIUS_M, OBJ_HEIGHT_M / 2)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(*OBJ_CENTER))
    cyl.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, Gf.Vec3d(0, 0, 0)))
    cyl.CreateDisplayColorAttr([(0.85, 0.30, 0.25)])
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

    robot = stage.DefinePrim(ROBOT)
    robot.GetReferences().AddReference(str(ASSET_DIR / "reBot_B601_DM.usda"),
                                       "/reBot_B601_DM")
    for short, p in BODY.items():
        if not stage.GetPrimAtPath(p).IsValid():
            raise RuntimeError(f"MISSING_LINK {p}")

    # ---- nested-rigid-body hierarchy repair (prereg ADD-2, smoke-proven) ---- #
    # The vendor asset ships 9 nested RigidBodyAPI links without xformstack
    # reset -> omni.physicsschema rejects every body ("missing xformstack
    # reset") and all joints get "no bodies defined"; nothing simulates.
    # Repair = bake the composed world matrix into a single transform op with
    # resetXformStack (the fix the parser error itself prescribes; D447 (2)).
    xfc0 = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_mats = {p: xfc0.GetLocalToWorldTransform(stage.GetPrimAtPath(p))
                  for short, p in BODY.items() if short != "base_link"}
    for p, M in world_mats.items():
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(p))
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ba1world")
        if not xf.SetXformOpOrder([op], resetXformStack=True):
            raise RuntimeError(f"XFORM_ORDER_FAIL {p}")
        op.Set(M)
    xfc1 = UsdGeom.XformCache(Usd.TimeCode.Default())
    reset_dev = 0.0
    for p, M in world_mats.items():
        got = xfc1.GetLocalToWorldTransform(stage.GetPrimAtPath(p))
        delta = np.array([[got[i][j] - M[i][j] for j in range(4)] for i in range(4)])
        reset_dev = max(reset_dev, float(np.abs(delta).max()))
    if reset_dev > 1e-9:
        raise RuntimeError(f"RESET_BAKE_DEV {reset_dev}")
    out["hierarchy_repair"] = {"n_links_baked": len(world_mats),
                               "reset_dev": reset_dev}
    print(f"[{LOG}] hierarchy repair: {len(world_mats)} links world-baked, "
          f"dev={reset_dev:.2e}", flush=True)

    # ---- joint model runtime gate ------------------------------------------ #
    def vec_close(a, b, tol=1e-6):
        return float(np.abs(np.array(a, dtype=float) - np.array(b, dtype=float)).max()) <= tol

    jmodel_rows = {}
    for name, lp0, lr0, lr1, lo, hi, mf in JOINTS_TBL:
        prim = stage.GetPrimAtPath(JROOT + name)
        g = {
            "axis": prim.GetAttribute("physics:axis").Get(),
            "localPos0": tuple(prim.GetAttribute("physics:localPos0").Get()),
            "localRot0": prim.GetAttribute("physics:localRot0").Get(),
            "localPos1": tuple(prim.GetAttribute("physics:localPos1").Get()),
            "localRot1": prim.GetAttribute("physics:localRot1").Get(),
            "lower": prim.GetAttribute("physics:lowerLimit").Get(),
            "upper": prim.GetAttribute("physics:upperLimit").Get(),
            "maxForce": prim.GetAttribute("drive:angular:physics:maxForce").Get(),
        }
        q0 = g["localRot0"]
        q1 = g["localRot1"]
        okj = (g["axis"] == "Z"
               and vec_close(g["localPos0"], lp0)
               and vec_close([q0.GetReal(), *q0.GetImaginary()], lr0)
               and vec_close(g["localPos1"], (0, 0, 0))
               and vec_close([q1.GetReal(), *q1.GetImaginary()], lr1)
               and abs(g["lower"] - lo) < 1e-4 and abs(g["upper"] - hi) < 1e-4
               and abs(g["maxForce"] - mf) < 1e-6)
        jmodel_rows[name] = {"pass": bool(okj)}
        if not okj:
            raise RuntimeError(f"JOINT_MODEL_GATE {name} {g}")
    for gk, body1 in (("gripper_joint1", "gripper_left"), ("gripper_joint2", "gripper_right")):
        prim = stage.GetPrimAtPath(JROOT + gk)
        lo = prim.GetAttribute("physics:lowerLimit").Get()
        hi = prim.GetAttribute("physics:upperLimit").Get()
        mf = prim.GetAttribute("drive:linear:physics:maxForce").Get()
        # float32-authored attrs: tolerance at float32 quantization scale
        if not (abs(lo - 0.0) < 1e-6 and abs(hi - OPEN_M) < 1e-6 and abs(mf - 100.0) < 1e-6):
            raise RuntimeError(f"GRIPPER_JOINT_GATE {gk} {lo} {hi} {mf}")
    out["joint_model_gate"] = jmodel_rows

    # ---- split2 collision port --------------------------------------------- #
    xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
    Mg_inv = xfc.GetLocalToWorldTransform(
        stage.GetPrimAtPath(BODY["gripper_link"])).GetInverse()
    def ensure_authorable(path):
        """De-instance enclosing instance roots so the prim accepts overrides.

        The full asset's geometry references are instanceable; collision prims
        under them are instance proxies (authoring forbidden).  Composition is
        unchanged by SetInstanceable(False) - scenegraph de-dup only."""
        prim = stage.GetPrimAtPath(path)
        roots = []
        guard = 0
        while prim.IsInstanceProxy():
            anc = prim.GetParent()
            while anc.IsInstanceProxy():
                anc = anc.GetParent()
            anc.SetInstanceable(False)
            roots.append(anc.GetPath().pathString)
            prim = stage.GetPrimAtPath(path)
            guard += 1
            if guard > 4:
                raise RuntimeError(f"DEINSTANCE_LOOP {path}")
        return prim, roots

    port_rows = {}
    deinstanced = []
    for finger in ("gripper_left", "gripper_right"):
        fprim = stage.GetPrimAtPath(BODY[finger])
        disabled = []
        coll_paths = [prim.GetPath().pathString
                      for prim in Usd.PrimRange(fprim, pred_prox)
                      if prim.HasAPI(UsdPhysics.CollisionAPI)
                      and "split2" not in prim.GetName()]
        for cp in coll_paths:
            prim, roots = ensure_authorable(cp)
            deinstanced += roots
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True).Set(False)
            disabled.append(cp)
        if len(disabled) != 1:
            raise RuntimeError(f"PORT_ORIG_COUNT {finger} {disabled}")
        for part in ("blade", "mount"):
            d = split_data[(finger, part)]
            mp = BODY[finger] + f"/collision_{part}_split2"
            mesh = UsdGeom.Mesh.Define(stage, mp)
            mesh.GetPointsAttr().Set([Gf.Vec3f(*v) for v in d["pts"]])
            mesh.GetFaceVertexCountsAttr().Set([int(v) for v in d["fvc"]])
            mesh.GetFaceVertexIndicesAttr().Set([int(v) for v in d["fvi"]])
            mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            mapi = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            mapi.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
            back = np.array(UsdGeom.Mesh(stage.GetPrimAtPath(mp)).GetPointsAttr().Get(),
                            dtype=np.float64)
            if back.shape != d["pts"].shape or float(
                    np.abs(back - d["pts"].astype(np.float32).astype(np.float64)).max()) > 0:
                raise RuntimeError(f"PORT_PTS_ROUNDTRIP {mp}")
        # census on this finger: 2 enabled + 1 disabled
        en = dis = 0
        for prim in Usd.PrimRange(fprim, pred_prox):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                a = prim.GetAttribute("physics:collisionEnabled")
                if a.Get() if (a and a.Get() is not None) else True:
                    en += 1
                else:
                    dis += 1
        if (en, dis) != (2, 1):
            raise RuntimeError(f"PORT_CENSUS {finger} {(en, dis)}")
        # blade inner extreme in glf frame at asset default state (q=0)
        bp = stage.GetPrimAtPath(BODY[finger] + "/collision_blade_split2")
        pts = np.array(UsdGeom.Mesh(bp).GetPointsAttr().Get(), dtype=np.float64)
        M = xfc.GetLocalToWorldTransform(bp.GetParent()) * Mg_inv
        Mn = np.array([[M[i][j] for j in range(4)] for i in range(4)])
        g = pts @ Mn[:3, :3] + Mn[3, :3]
        extreme = float(g[:, 1].max() if finger == "gripper_left" else g[:, 1].min())
        dev = abs(extreme - plan["blade_extreme_pins"][finger])
        port_rows[finger] = {"disabled": disabled, "census": [en, dis],
                             "blade_extreme_glf_m": extreme, "extreme_dev_m": dev}
        if dev > 1e-9:
            raise RuntimeError(f"PORT_BLADE_EXTREME {finger} dev={dev}")
    out["split_port_gate"] = port_rows
    out["deinstanced_roots"] = deinstanced
    print(f"[{LOG}] split2 port gates PASS (deinstanced={len(deinstanced)})", flush=True)

    # ---- articulation/harness settings ------------------------------------- #
    for p in BODY.values():
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            stage.GetPrimAtPath(p)).CreateSleepThresholdAttr().Set(0.0)
    for short in ("gripper_left", "gripper_right", "gripper_link"):
        PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(BODY[short])).CreateThresholdAttr().Set(0.0)
    art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(BODY["base_link"]))
    art.CreateEnabledSelfCollisionsAttr().Set(False)
    art.CreateSolverPositionIterationCountAttr().Set(32)
    art.CreateSolverVelocityIterationCountAttr().Set(1)
    fpair = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(BODY["gripper_left"]))
    fpair.CreateFilteredPairsRel().AddTarget(BODY["gripper_right"])
    gpair = UsdPhysics.FilteredPairsAPI.Apply(gnd.GetPrim())
    gpair.CreateFilteredPairsRel().AddTarget(BODY["base_link"])

    # ---- drives + initial states ------------------------------------------- #
    jpaths = {j[0]: JROOT + j[0] for j in JOINTS_TBL}
    jpaths["g1"] = JROOT + "gripper_joint1"
    jpaths["g2"] = JROOT + "gripper_joint2"
    q_standoff = np.array(plan["ik"]["standoff"]["q_deg"])
    drive_attrs = {}
    for i, (name, *_rest) in enumerate(JOINTS_TBL):
        prim = stage.GetPrimAtPath(jpaths[name])
        if not grasping_utils.set_joint_drive_parameters(
                prim, float(q_standoff[i]), "position",
                stiffness=ARM_STIFF, damping=ARM_DAMP, max_force=None):
            raise RuntimeError(f"DRIVE_SET_FAIL {name}")
        st = PhysxSchema.JointStateAPI.Apply(prim, "angular")
        st.CreatePositionAttr().Set(float(q_standoff[i]))
        st.CreateVelocityAttr().Set(0.0)
        drive_attrs[i] = UsdPhysics.DriveAPI.Get(prim, "angular").GetTargetPositionAttr()
    for k, gk in ((6, "g1"), (7, "g2")):
        prim = stage.GetPrimAtPath(jpaths[gk])
        if not grasping_utils.set_joint_drive_parameters(
                prim, OPEN_M, "position",
                stiffness=FING_STIFF, damping=FING_DAMP, max_force=None):
            raise RuntimeError(f"DRIVE_SET_FAIL {gk}")
        st = PhysxSchema.JointStateAPI.Apply(prim, "linear")
        st.CreatePositionAttr().Set(OPEN_M)
        st.CreateVelocityAttr().Set(0.0)
        drive_attrs[k] = UsdPhysics.DriveAPI.Get(prim, "linear").GetTargetPositionAttr()

    # ---- lights / camera / capture ------------------------------------------ #
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
    zc = (CAM_EYE - CAM_TGT) / np.linalg.norm(CAM_EYE - CAM_TGT)
    xc = np.cross(np.array([0.0, 0.0, 1.0]), zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    cam.AddTransformOp().Set(Gf.Matrix4d(
        xc[0], xc[1], xc[2], 0.0, yc[0], yc[1], yc[2], 0.0,
        zc[0], zc[1], zc[2], 0.0, CAM_EYE[0], CAM_EYE[1], CAM_EYE[2], 1.0))

    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True)
    rp = rep.create.render_product(CAMERA_PRIM, (RES_W, RES_H))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach(rp)

    def capture(rt):
        rep.orchestrator.step(rt_subframes=rt)
        img = np.asarray(annot.get_data())
        if img.ndim != 3 or img.shape[0] != RES_H or img.shape[1] != RES_W:
            raise RuntimeError(f"IMAGE_SHAPE {img.shape}")
        rgb = img[:, :, :3].astype(np.uint8)
        if float(rgb.std()) < 5.0:
            raise RuntimeError("IMAGE_FLAT")
        return rgb

    # ---- instrumentation ---------------------------------------------------- #
    counters = {"steps": 0, "contacts": 0, "unknown": 0}
    F_L = np.zeros((N_STEPS, 3))
    F_R = np.zeros((N_STEPS, 3))
    palm_F = np.zeros((N_STEPS, 3))
    support_F = np.zeros((N_STEPS, 3))
    arm_other_F = np.zeros((N_STEPS, 3))
    pairs_seen = set()
    cur = {"t": -1}
    L_PATH, R_PATH, GLF_PATH = BODY["gripper_left"], BODY["gripper_right"], BODY["gripper_link"]
    FINGERY = {L_PATH, R_PATH, GLF_PATH}
    ROBOT_BODIES = set(BODY.values())

    def on_step(dt):
        counters["steps"] += 1

    def on_contact(headers, data):
        t = cur["t"]
        for h in headers:
            counters["contacts"] += 1
            a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            pair = {a0, a1}
            pairs_seen.add((a0, a1))
            if not a0 or not a1:
                counters["unknown"] += 1
                continue
            imp = np.zeros(3)
            for i in range(h.contact_data_offset,
                           h.contact_data_offset + h.num_contact_data):
                d = data[i]
                imp += np.array([d.impulse.x, d.impulse.y, d.impulse.z])
            if t < 0 or t >= N_STEPS:
                continue
            if OBJECT_PRIM in pair:
                other = next(iter(pair - {OBJECT_PRIM}), None)
                if other == L_PATH:
                    F_L[t] += imp / DT
                elif other == R_PATH:
                    F_R[t] += imp / DT
                elif other == GLF_PATH:
                    palm_F[t] += imp / DT
                elif other == PED_PRIM:
                    support_F[t] += imp / DT
                elif other in ROBOT_BODIES:
                    arm_other_F[t] += imp / DT
            elif pair & ROBOT_BODIES and (pair & {PED_PRIM, GROUND_PRIM}):
                if not pair & FINGERY:
                    arm_other_F[t] += imp / DT

    physx_iface = omni.physx.get_physx_interface()
    physx_sim = omni.physx.get_physx_simulation_interface()
    sub_step = physx_iface.subscribe_physics_step_events(on_step)
    sub_contact = physx_sim.subscribe_contact_report_events(on_contact)

    def world_pose(path):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        t = xf.ExtractTranslation()
        q = xf.ExtractRotationQuat()
        return (np.array([t[0], t[1], t[2]]),
                np.array([q.GetReal(), *q.GetImaginary()]))

    def jstate(key):
        return grasping_utils.get_joint_state(stage.GetPrimAtPath(jpaths[key]))[0]

    JKEYS = [j[0] for j in JOINTS_TBL] + ["g1", "g2"]

    # visual point cache for RRD keyframe geometry (visual meshes, subsampled)
    def disp_points(body, exclude_substr=("collision", "split2")):
        acc = []
        base = stage.GetPrimAtPath(body)
        Mb_inv = xfc.GetLocalToWorldTransform(base).GetInverse()
        child_excl = [BODY["gripper_left"], BODY["gripper_right"]] \
            if body == GLF_PATH else []
        for prim in Usd.PrimRange(base, pred_prox):
            ps = prim.GetPath().pathString
            if any(ps == e or ps.startswith(e + "/") for e in child_excl):
                continue
            if any(s in prim.GetName().lower() for s in exclude_substr):
                continue
            if prim.IsA(UsdGeom.Mesh):
                pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(),
                               dtype=np.float64)[::40]
                M = xfc.GetLocalToWorldTransform(prim) * Mb_inv
                Mn = np.array([[M[i][j] for j in range(4)] for i in range(4)])
                acc.append(pts @ Mn[:3, :3] + Mn[3, :3])
        return np.vstack(acc)

    jaw_pts = {"L": disp_points(L_PATH), "R": disp_points(R_PATH),
               "palm": disp_points(GLF_PATH)}

    # ---- warmup (annotator/material readiness; capture starts post-step 0
    # so the video never shows the pre-physics default q=0 pose) -------------- #
    for _ in range(WARMUP_UPDATES):
        app.update()
    if counters["steps"] != 0:
        raise RuntimeError("WARMUP_STEPPED_PHYSICS")
    frame_i = 0

    # ---- main loop ----------------------------------------------------------- #
    q_cmd = plan["q_cmd"]
    phase_idx = plan["phase_idx"]
    q_meas = np.zeros((N_STEPS, 8))
    obj_pos = np.zeros((N_STEPS, 3))
    obj_quat = np.zeros((N_STEPS, 4))
    glf_pos = np.zeros((N_STEPS, 3))
    glf_quat = np.zeros((N_STEPS, 4))
    keyframe_snaps = {}

    for t in range(N_STEPS):
        for k in range(8):
            drive_attrs[k].Set(float(q_cmd[t, k]))
        cur["t"] = t
        physx_sim.simulate_scene(scene_int, DT, 0)
        physx_sim.fetch_results_scene(scene_int)
        xfc.Clear()
        q_meas[t] = [jstate(k) for k in JKEYS]
        obj_pos[t], obj_quat[t] = world_pose(OBJECT_PRIM)
        glf_pos[t], glf_quat[t] = world_pose(GLF_PATH)
        if counters["steps"] != t + 1:
            raise RuntimeError(f"STEP_COUNT {counters['steps']} != {t + 1}")
        if not np.isfinite(q_meas[t]).all():
            raise RuntimeError(f"JOINT_STATE_NAN t={t}")
        if t % CAPTURE_EVERY == 0:
            Image.fromarray(capture(RT_SUB)).save(FRAMES_DIR / f"{frame_i:04d}.png")
            frame_i += 1
            if counters["steps"] != t + 1:
                raise RuntimeError("CAPTURE_STEPPED_PHYSICS_LOOP")
        if t in KEYFRAMES:
            name = KEYFRAMES[t]
            Image.fromarray(capture(RT_SUB_KEY)).save(OUT_KEY[name])
            keyframe_snaps[name] = {
                "t": t, "q_deg": q_meas[t].tolist(),
                "obj_pos": obj_pos[t].tolist(), "obj_quat_wxyz": obj_quat[t].tolist(),
                "glf_pos": glf_pos[t].tolist(), "glf_quat_wxyz": glf_quat[t].tolist(),
                "left_pose": [world_pose(L_PATH)[0].tolist(),
                              world_pose(L_PATH)[1].tolist()],
                "right_pose": [world_pose(R_PATH)[0].tolist(),
                               world_pose(R_PATH)[1].tolist()]}
        # phase-boundary gates
        if t == 29:  # SETTLE end
            drift = float(np.linalg.norm(obj_pos[t] - OBJ_CENTER))
            track = float(np.abs(q_meas[t, :6] - q_cmd[t, :6]).max())
            M = fk_glf(q_meas[t, :6])
            fk_dev = float(np.linalg.norm(M[:3, 3] - glf_pos[t]))
            sup_mag = np.linalg.norm(support_F[:30], axis=1)
            calib = float(np.median(sup_mag[sup_mag > 0])) if (sup_mag > 0).any() else float("nan")
            out["settle_gates"] = {
                "obj_drift_m": drift, "arm_track_err_deg": track,
                "fk_vs_usd_m": fk_dev,
                "support_calib_n": calib, "mg_n": OBJ_MASS_KG * GRAVITY}
            print(f"[{LOG}] SETTLE gates: drift={drift * 1000:.3f}mm "
                  f"track={track:.4f}deg fk_dev={fk_dev * 1000:.3f}mm "
                  f"calib={calib:.8f}N vs mg={OBJ_MASS_KG * GRAVITY:.8f}N", flush=True)
            if drift > 2e-3:
                raise RuntimeError(f"SETTLE_OBJ_DRIFT {drift}")
            if fk_dev > 1e-3:
                raise RuntimeError(f"SETTLE_FK_DEV {fk_dev}")
        if t == 179:  # SETTLE2 end - TCP track measurement
            tgt = np.array(plan["targets"]["grasp"])
            tcp_err = float(np.linalg.norm(glf_pos[t] - tgt))
            out["track_gate"] = {"tcp_err_m": tcp_err,
                                 "arm_q_err_deg": float(
                                     np.abs(q_meas[t, :6] - q_cmd[t, :6]).max())}
            print(f"[{LOG}] SETTLE2 end: tcp_err={tcp_err * 1000:.2f}mm", flush=True)

    print(f"[{LOG}] 540 steps + {frame_i} frames complete", flush=True)
    sub_step = None
    sub_contact = None

    # ---- gates + verdict ----------------------------------------------------- #
    FL_mag = np.linalg.norm(F_L, axis=1)
    FR_mag = np.linalg.norm(F_R, axis=1)
    bilateral_min = np.minimum(FL_mag, FR_mag)
    close_lo, close_hi = 180, 300
    g1 = bool((bilateral_min[close_lo:close_hi] > BILATERAL_GATE_N).any())
    z_obj_close, z_obj_hold = obj_pos[299, 2], obj_pos[539, 2]
    z_glf_close, z_glf_hold = glf_pos[299, 2], glf_pos[539, 2]
    glf_rise = float(z_glf_hold - z_glf_close)
    obj_rise = float(z_obj_hold - z_obj_close)
    slip = float(abs((z_glf_hold - z_obj_hold) - (z_glf_close - z_obj_close)))
    g2 = bool(obj_rise >= glf_rise - FOLLOW_GATE_M and glf_rise >= GLF_LIFT_MIN_M)
    g3 = bool(slip < SLIP_GATE_M)
    tcp_err = out["track_gate"]["tcp_err_m"]
    track_ok = bool(tcp_err <= TRACK_GATE_M)
    success = bool(g1 and g2 and g3 and track_ok)
    if success:
        code = "BA1_FULL_ARM_SIDE_GRASP_LIFT_SUCCESS"
    elif not track_ok:
        code = "BA1_TCP_TRACK_FAIL"
    elif not g1:
        code = "BA1_NO_BILATERAL"
    else:
        code = "BA1_SLIP_DURING_LIFT"
    measurement_valid = bool(
        out["settle_gates"]["obj_drift_m"] < 2e-3
        and out["settle_gates"]["fk_vs_usd_m"] < 1e-3
        and counters["steps"] == N_STEPS
        and np.isfinite(q_meas).all())
    arm_touch_peak = float(np.linalg.norm(arm_other_F, axis=1).max())
    out["verdict"] = {
        "code": code, "success": success,
        "G1_close_bilateral": g1,
        "G1_peak_bilateral_n": float(bilateral_min[close_lo:close_hi].max()),
        "close_left_peak_n": float(FL_mag[close_lo:close_hi].max()),
        "close_right_peak_n": float(FR_mag[close_lo:close_hi].max()),
        "G2_lift_follow": g2, "glf_rise_m": glf_rise, "obj_rise_m": obj_rise,
        "G3_hold_slip": g3, "slip_m": slip,
        "G_track_tcp_err_m": tcp_err, "track_ok": track_ok,
        "hold_end_bilateral_n": float(bilateral_min[539]),
        "arm_unintended_contact_peak_n": arm_touch_peak,
        "measurement_valid": measurement_valid,
        "gates": {"bilateral_n": BILATERAL_GATE_N, "follow_m": FOLLOW_GATE_M,
                  "slip_m": SLIP_GATE_M, "track_m": TRACK_GATE_M,
                  "glf_lift_min_m": GLF_LIFT_MIN_M},
        "non_claims": "no claim about real-robot B601 grasping/control realism, "
                      "top-down full-arm reachability, other azimuths, or any "
                      "re-judgment of D446/D445 (prereg SS1)"}
    out["keyframes"] = keyframe_snaps
    out["raw_contact_stats"] = {"n_total": counters["contacts"],
                                "n_unknown": counters["unknown"],
                                "pairs_seen": sorted(str(p) for p in pairs_seen)}
    print(f"[{LOG}] verdict={code} G1={g1} G2={g2} G3={g3} track={track_ok} "
          f"slip={slip * 1000:.2f}mm obj_rise={obj_rise * 1000:.1f}mm", flush=True)

    # ---- trace --------------------------------------------------------------- #
    np.savez(OUT["trace.npz"],
             q_cmd=q_cmd, q_meas=q_meas, F_L=F_L, F_R=F_R, palm_F=palm_F,
             support_F=support_F, arm_other_F=arm_other_F,
             obj_pos=obj_pos, obj_quat=obj_quat, glf_pos=glf_pos,
             glf_quat=glf_quat, phase_idx=phase_idx,
             bilateral_min=bilateral_min)
    with open(OUT["trace.npz"], "rb") as f:
        os.fsync(f.fileno())

    # ---- mp4 ------------------------------------------------------------------ #
    cmd = [FFMPEG, "-n", "-framerate", str(FPS), "-i",
           str(FRAMES_DIR / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-preset", "medium", str(OUT["side_grasp.mp4"])]
    pr = subprocess.run(cmd, capture_output=True, text=True)
    if pr.returncode != 0 or not OUT["side_grasp.mp4"].exists():
        raise RuntimeError(f"FFMPEG_FAIL rc={pr.returncode} {pr.stderr[-800:]}")
    out["video"] = {"file": OUT["side_grasp.mp4"].name, "frames": frame_i,
                    "fps": FPS, "duration_s": frame_i / FPS,
                    "ffmpeg_cmd": " ".join(cmd),
                    "caption_obligation": "always present with G1-G3 numbers; "
                                          "authority is results/trace (prereg SS7)"}
    print(f"[{LOG}] mp4 written: {frame_i} frames {frame_i / FPS:.2f}s", flush=True)

    # ---- rerun (D341, save-only) ---------------------------------------------- #
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    app_id = f"roarm_g0d_d448_{TAG}"

    def cyl_wire(center_xyz, quat_wxyz=None):
        c = np.asarray(center_xyz)
        ang = np.linspace(0, 2 * math.pi, 49)
        rings = []
        for zk in (-OBJ_HEIGHT_M / 2, 0.0, OBJ_HEIGHT_M / 2):
            ring = np.column_stack([OBJ_RADIUS_M * np.cos(ang),
                                    OBJ_RADIUS_M * np.sin(ang),
                                    np.full_like(ang, zk)])
            if quat_wxyz is not None:
                ring = ring @ quat_to_R(quat_wxyz).T
            rings.append((ring + c).tolist())
        return rings

    def box_wire(center, half):
        c, h = np.asarray(center), np.asarray(half)
        s = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
             [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
        v = [c + h * np.array(x) for x in s]
        e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
        return [[v[a].tolist(), v[b].tolist()] for a, b in e]

    def axes_strips(pos, R, scale=0.04):
        p = np.asarray(pos)
        return [[p.tolist(), (p + np.asarray(R)[:, k] * scale).tolist()]
                for k in range(3)]

    def to_world(pts, pos, quat_wxyz):
        return pts @ quat_to_R(quat_wxyz).T + np.asarray(pos)

    with rr.RecordingStream(app_id, recording_id=f"g0d_d448_{TAG}",
                            make_default=False, send_properties=True) as rec:
        rec.save(str(OUT["timeline.rrd"]), write_footer=True)
        rec.log("scene/pedestal", rr.LineStrips3D(
            box_wire([OBJ_CENTER[0], OBJ_CENTER[1], PED_TOP / 2],
                     [PED_HALF_XY, PED_HALF_XY, PED_TOP / 2]),
            colors=[[120, 120, 120]], radii=0.0008), static=True)
        rec.log("scene/object_spawn", rr.LineStrips3D(
            cyl_wire(OBJ_CENTER), colors=[[80, 200, 120]], radii=0.0006), static=True)
        rec.log("scene/grasp_target_axes", rr.LineStrips3D(
            axes_strips(plan["targets"]["grasp"], plan["R_target"]),
            colors=[[255, 60, 60], [60, 255, 60], [60, 60, 255]], radii=0.0009),
            static=True)

        for t in range(N_STEPS):
            rec.reset_time()
            rec.set_time("step", sequence=t)
            rec.log("forces/left_n", rr.Scalars(float(FL_mag[t])))
            rec.log("forces/right_n", rr.Scalars(float(FR_mag[t])))
            rec.log("forces/bilateral_min_n", rr.Scalars(float(bilateral_min[t])))
            rec.log("plots/obj_z_m", rr.Scalars(float(obj_pos[t, 2])))
            rec.log("plots/glf_z_m", rr.Scalars(float(glf_pos[t, 2])))
            rec.log("joints/arm_track_err_deg", rr.Scalars(
                float(np.abs(q_meas[t, :6] - q_cmd[t, :6]).max())))
            rec.log("joints/grip_q_sum_m", rr.Scalars(float(q_meas[t, 6] + q_meas[t, 7])))
            if t in KEYFRAMES:
                ks = keyframe_snaps[KEYFRAMES[t]]
                rec.log("gripper/left_points", rr.Points3D(
                    to_world(jaw_pts["L"], ks["left_pose"][0], ks["left_pose"][1]),
                    colors=[70, 130, 230], radii=0.0006))
                rec.log("gripper/right_points", rr.Points3D(
                    to_world(jaw_pts["R"], ks["right_pose"][0], ks["right_pose"][1]),
                    colors=[230, 130, 70], radii=0.0006))
                rec.log("gripper/palm_points", rr.Points3D(
                    to_world(jaw_pts["palm"], ks["glf_pos"], ks["glf_quat_wxyz"]),
                    colors=[150, 150, 160], radii=0.0006))
                rec.log("object/wire", rr.LineStrips3D(
                    cyl_wire(ks["obj_pos"], ks["obj_quat_wxyz"]),
                    colors=[[225, 60, 60]], radii=0.0006))
                rec.log("events/phase", rr.TextLog(
                    f"keyframe {KEYFRAMES[t]} t={t}", level=rr.TextLogLevel.INFO))
        rec.reset_time()
        rec.set_time("step", sequence=N_STEPS - 1)
        rec.log("events/verdict", rr.TextLog(
            f"{code} G1={g1}({out['verdict']['G1_peak_bilateral_n']:.3f}N) "
            f"G2={g2}(obj_rise={obj_rise * 1000:.1f}mm/glf {glf_rise * 1000:.1f}mm) "
            f"G3={g3}(slip={slip * 1000:.2f}mm) tcp_err={tcp_err * 1000:.2f}mm",
            level=rr.TextLogLevel.INFO if success else rr.TextLogLevel.WARN))
        summary_md = (
            f"# g0d_d448 {TAG} — B601 full-arm side grasp + lift (phi=135)\n\n"
            f"**VERDICT: {code}**\n\n"
            f"SETTLE30/APPROACH120/SETTLE2 30/CLOSE120/LIFT120/HOLD120 @dt 1/60.\n"
            f"G1 bilateral>{BILATERAL_GATE_N}N in CLOSE: {g1} "
            f"(peak {out['verdict']['G1_peak_bilateral_n']:.3f} N)\n"
            f"G2 obj follows glf rise within {FOLLOW_GATE_M * 1000:.0f}mm: {g2} "
            f"(obj {obj_rise * 1000:.1f} / glf {glf_rise * 1000:.1f} mm)\n"
            f"G3 slip<{SLIP_GATE_M * 1000:.0f}mm: {g3} ({slip * 1000:.2f} mm)\n"
            f"TCP err at grasp settle: {tcp_err * 1000:.2f} mm\n\n"
            f"Authority = ba1_results.json + ba1_trace.npz; mp4/keyframes are "
            f"visual evidence only (D341/prereg SS7).")
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
                                      name="2 | keyframe geometry"),
                    rrb.TextLogView(origin="/events", contents="/events/**",
                                    name="3 | phases + verdict"),
                    column_shares=[0.26, 0.48, 0.26],
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/forces", contents="/forces/**",
                                       name="4 | jaw-object contact force [N]"),
                    rrb.TimeSeriesView(origin="/plots", contents="/plots/**",
                                       name="5 | obj/glf z [m]"),
                    rrb.TimeSeriesView(origin="/joints", contents="/joints/**",
                                       name="6 | tracking err / grip aperture"),
                ),
                row_shares=[0.55, 0.45],
            ),
            auto_layout=False, auto_views=False, collapse_panels=True,
        )
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=30.0)
    blueprint.save(app_id, str(OUT["timeline.rbl"]))

    expected_entities = [
        "metadata/run", "scene/pedestal", "scene/object_spawn",
        "scene/grasp_target_axes",
        "gripper/left_points", "gripper/right_points", "gripper/palm_points",
        "object/wire", "plots/obj_z_m", "plots/glf_z_m",
        "forces/left_n", "forces/right_n", "forces/bilateral_min_n",
        "joints/arm_track_err_deg", "joints/grip_q_sum_m",
        "events/phase", "events/verdict"]
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    lin3 = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    components = {
        "metadata/run": ["TextDocument:text"],
        "scene/pedestal": lin3, "scene/object_spawn": lin3,
        "scene/grasp_target_axes": lin3,
        "gripper/left_points": pts3, "gripper/right_points": pts3,
        "gripper/palm_points": pts3, "object/wire": lin3,
        "plots/obj_z_m": ["Scalars:scalars"], "plots/glf_z_m": ["Scalars:scalars"],
        "forces/left_n": ["Scalars:scalars"], "forces/right_n": ["Scalars:scalars"],
        "forces/bilateral_min_n": ["Scalars:scalars"],
        "joints/arm_track_err_deg": ["Scalars:scalars"],
        "joints/grip_q_sum_m": ["Scalars:scalars"],
        "events/phase": ["TextLog:text", "TextLog:level"],
        "events/verdict": ["TextLog:text", "TextLog:level"],
    }
    validation = validate_rerun_artifact(
        OUT["timeline.rrd"],
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time", "step"],
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

    # ---- end-of-run pin recheck ---------------------------------------------- #
    end_pins = {}
    for path, want in SRC_PINS.items():
        end_pins[path.name] = sha256(path) == want
    end_pins["bg1_gripper_split2.usd"] = (
        sha256(SPLIT2_USD) == plan["pins"]["bg1_gripper_split2.usd"]["sha256"])
    out["end_pin_recheck"] = end_pins
    if not all(end_pins.values()):
        raise RuntimeError(f"END_PIN_DRIFT {end_pins}")

    out["plan"]["q_cmd_note"] = "per-step command table in trace.npz q_cmd"
    out["wall_seconds"] = round(time.time() - t_start, 1)
    out["physics_backend_note"] = ("explicit scene-int stepping "
                                   "(PhysxSceneAPI update Disabled, "
                                   "simulate_scene/fetch_results_scene), CPU PhysX")
    out["artifacts"] = {p.name: {"sha256_16": sha256(p)[:16], "bytes": p.stat().st_size}
                        for k, p in {**OUT, **{f"key_{n}": p for n, p in OUT_KEY.items()}}.items()
                        if p.exists() and k not in ("results.json", "failure.json")}
    fsync_write(OUT["results.json"], json.dumps(out, indent=2, default=str) + "\n")
    print(f"[{LOG}] results.json={sha256(OUT['results.json'])[:16]} "
          f"bytes={OUT['results.json'].stat().st_size}", flush=True)
    print(f"[{LOG}] G0D_BA1_VERDICT={code} success={success} valid={measurement_valid}",
          flush=True)
    return 0


def main() -> int:
    plan = preflight()
    fsync_write(OUT["argv.txt"], " ".join([sys.executable, *sys.argv]) + "\n")
    shutil.copyfile(__file__, OUT["script.py.txt"])
    print(f"[{LOG}] preflight OK: IK recompute matched design, pins verified",
          flush=True)
    return run(plan)


if __name__ == "__main__":
    sys.exit(main())
