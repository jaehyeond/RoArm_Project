"""G0b T3 — attempt3 collision-asset jaw/throat occlusion READ-ONLY vertex audit.

Case g0b_d420, session 23rd. Diagnosis layer ONLY:
  - This is NOT a re-decomposition (D415-③ untouched). The frozen attempt3 USD
    is opened read-only via pxr Usd.Stage.Open; no Save()/Export()/layer-edit
    API exists anywhere in this file.
  - Verdict layer: pure geometry cross-check against the four Isaac-measured
    T3 outcomes. It makes no physics claim and no real-gripper claim.

Question (failable, session-progress rule):
  Does the frozen attempt3 collision geometry (64+64 parts), placed by the
  same FK chain / TCP offset / USD joint frame the probes used, REPRODUCE:
    (CA1) attempt1 descend stop  = floor contact at cylinder top (88.31 deg),
    (CA4) attempt4 descend stop  = same floor, 45 deg (angle invariance),
    (CB)  attempt2 close sweep   = contact-free at every band angle?
  All pass -> G0B_T3_JAW_AUDIT_VERDICT=JAW_AUDIT_CONSISTENT
  Any fail -> GEOMETRY_MISMATCH (frame/joint/FK assumption wrong — investigate
  before the user escalation). Identity/frame guard failure -> AUDIT_ABORT.

Authority: stdout verdict line + t3_jaw_audit_results.json (exit code is NOT a
verdict channel, D424-③). Float32 Rerun copies are inspection evidence only.
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


def _bootstrap_pxr_env() -> None:
    """Re-exec once with pxr module + shared-library paths (LD_LIBRARY_PATH is
    read by the dynamic linker at process start, so in-process assignment is
    not reliable). Kit is NOT launched — this stays a plain-python process."""
    if os.environ.get(_REEXEC_FLAG) == "1":
        return
    if not USD_LIBS.is_dir():
        print(f"[g0b_t3_jaw_audit] G0B_T3_JAW_AUDIT_VERDICT=AUDIT_ABORT missing usd libs {USD_LIBS}", flush=True)
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
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim_scripts"))

from roarm_kinematics import _CHAIN, Tmat, Trot_z, fk_full  # noqa: E402

LOG = "g0b_t3_jaw_audit"
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"

# --- frozen attempt3 asset identity (same pins as p9 D-3) ---------------------
ATTEMPT3_USD = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3"
    / "roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd"
)
ATTEMPT3_ROOT_SHA256 = "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff"
ATTEMPT3_PHYSICS_LAYER = ATTEMPT3_USD.parent / "configuration/roarm_m3_physics.usd"
ATTEMPT3_PHYSICS_SHA256 = "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503"

BODY_PRIMS = {"link5": "/roarm_m3/link5", "gripper_link": "/roarm_m3/gripper_link"}
JOINT_PRIM = "/roarm_m3/joints/link5_to_gripper_link"
EXPECTED_PART_COUNT = 64
LEGACY_COLLIDER_FRAGMENT = "node_STL_BINARY_"

TCP_LOCAL = np.array([0.0, 0.0, 0.115428], dtype=np.float64)  # link5 frame (== env TCP_LOCAL_OFFSET_M)
Q5_OPEN_DEG = math.degrees(1.5413)  # 88.3096 (D-1 frozen)

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"
A1_RESULTS = OUT_DIR / "t3_grasp_results.json"
A1_CSV = OUT_DIR / "t3_grasp_steps.csv"
A2_RESULTS = OUT_DIR / "t3_grasp2_results.json"
A2_CSV = OUT_DIR / "t3_grasp2_steps.csv"
A4_RESULTS = OUT_DIR / "t3_grasp4_results.json"
A4_CSV = OUT_DIR / "t3_grasp4_steps.csv"

SAMPLE_SPACING_M = 0.0005     # hull-face surface sampling pitch (stated error bound)
VIEW_STRIDE = 16              # Rerun view decimation (inspection-only layer; audit2
                              # lesson: 28MB RRD raced the headless screenshot loader)
DEPTH_BAND_MM = 0.5           # throat-profile depth binning
FLOOR_GATE_MM = 1.0           # |floor clearance| gate for CA1/CA4
ANGLE_INV_GATE_MM = 0.5       # |d_floor(88.31) - d_floor(45)| gate
RIM_BAND_MM = (5.0, 15.0)     # T1 real rim-pinch depth region of interest


# ---------------------------------------------------------------------------
# small math helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _gf_to_np(gf_mat) -> np.ndarray:
    """Gf.Matrix4d (row-vector convention) -> column-vector 4x4."""
    return np.array(gf_mat, dtype=np.float64).T


def _quat_to_R(q) -> np.ndarray:
    w = float(q.GetReal())
    x, y, z = (float(v) for v in q.GetImaginary())
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _T_from(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(quat)
    T[:3, 3] = np.array(pos, dtype=np.float64)
    return T


def _Rz(theta_rad: float) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def fk_T_link5(q6_deg: np.ndarray) -> np.ndarray:
    """World 4x4 of the link5 frame (chain stops after the link4_to_link5 joint)."""
    q = np.radians(np.asarray(q6_deg, dtype=np.float64))
    T = np.eye(4)
    for name, xyz, rpy, qi in _CHAIN:
        T = T @ Tmat(xyz, rpy)
        if qi is not None:
            T = T @ Trot_z(q[qi])
        if name == "link4_to_link5":
            return T
    raise RuntimeError("link4_to_link5 not found in _CHAIN")


def hull_samples(pts: np.ndarray, spacing_m: float) -> tuple[np.ndarray, bool]:
    """Vertices + dense face samples of the part's convex hull. PhysX cooks a
    convexHull per part; the scipy hull APPROXIMATES that cooked surface — here
    cook-faithful because every part has <=13 authored vertices, far under the
    authored physxConvexHullCollision:hullVertexLimit=64 (no simplification).
    Falls back to raw points on degenerate input."""
    pts = np.asarray(pts, dtype=np.float64)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts.copy(), False
    out = [pts[hull.vertices]]
    for tri in pts[hull.simplices]:
        a, b, c = tri
        n = max(
            1,
            int(math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / spacing_m)),
        )
        ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        mask = (ii + jj) <= n
        u = (ii[mask] / n)[:, None]
        v = (jj[mask] / n)[:, None]
        out.append(a[None, :] + u * (b - a)[None, :] + v * (c - a)[None, :])
    return np.vstack(out), True


def transform_pts(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------
# measured-run input parsing (authoritative artifacts of a1/a2/a4)
# ---------------------------------------------------------------------------
def load_attempt(results_path: Path, csv_path: Path, pose_phase: str, reduce: str) -> dict:
    doc = json.loads(results_path.read_text())
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["phase"] == pose_phase:
                rows.append((float(row["tcp_x"]), float(row["tcp_y"]), float(row["tcp_z"]), float(row["q5_deg"])))
    if not rows:
        raise RuntimeError(f"{csv_path.name}: no rows with phase={pose_phase}")
    arr = np.array(rows, dtype=np.float64)
    picked = arr[-1] if reduce == "last" else arr.mean(axis=0)
    return {
        "tag": doc["tag"],
        "verdict": doc["verdict"],
        "q_descend_deg": np.array(doc["plan"]["q_descend_deg"], dtype=np.float64),
        "cyl_center": np.array(doc["plan"]["center"], dtype=np.float64),
        "cyl_top_z": float(doc["plan"]["world_grasp"][2]),
        "cyl_r": float(doc["object"]["size_m"][0]) / 2.0,
        "cyl_h": float(doc["object"]["size_m"][2]),
        "close_deg": [float(v) for v in doc["gates"]["close_deg"]],
        "tcp_meas": picked[:3].copy(),
        "q5_meas_deg": float(picked[3]),
        "pose_source": f"{csv_path.name}:{pose_phase}:{reduce}(n={len(rows)})",
    }


# ---------------------------------------------------------------------------
# USD extraction (READ-ONLY)
# ---------------------------------------------------------------------------
def extract_asset() -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(ATTEMPT3_USD))
    xf = UsdGeom.XformCache()
    bodies: dict[str, dict] = {}
    for label, body_path in BODY_PRIMS.items():
        L = _gf_to_np(xf.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path)))
        Linv = np.linalg.inv(L)
        parts, legacy, approx_bad = [], [], []
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            path = prim.GetPath().pathString
            if path != body_path and not path.startswith(body_path + "/"):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            enabled = True if enabled is None else bool(enabled)
            name = path.rsplit("/", 1)[-1]
            if LEGACY_COLLIDER_FRAGMENT in path:
                legacy.append((path, enabled))
                continue
            if "part_" not in name or not enabled:
                continue
            approx = None
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if str(approx) != "convexHull":
                approx_bad.append((path, str(approx)))
            mesh = UsdGeom.Mesh(prim)
            raw = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
            X_rel = Linv @ _gf_to_np(xf.GetLocalToWorldTransform(prim))
            pts_link = transform_pts(X_rel, raw)
            samples, hull_ok = hull_samples(pts_link, SAMPLE_SPACING_M)
            parts.append({"name": name, "path": path, "n_raw": int(raw.shape[0]), "hull_ok": hull_ok, "samples": samples})
        bodies[label] = {
            "world_T": L,
            "parts": parts,
            "legacy": legacy,
            "approx_bad": approx_bad,
        }

    j = stage.GetPrimAtPath(JOINT_PRIM)
    rj = UsdPhysics.RevoluteJoint(j)
    joint = {
        "axis": str(rj.GetAxisAttr().Get()),
        "body0": [str(t) for t in rj.GetBody0Rel().GetTargets()],
        "body1": [str(t) for t in rj.GetBody1Rel().GetTargets()],
        "T0": _T_from(rj.GetLocalPos0Attr().Get(), rj.GetLocalRot0Attr().Get()),
        "T1": _T_from(rj.GetLocalPos1Attr().Get(), rj.GetLocalRot1Attr().Get()),
        "lower_deg": float(rj.GetLowerLimitAttr().Get()),
        "upper_deg": float(rj.GetUpperLimitAttr().Get()),
    }
    # authored-pose witness: authored rel transform must equal joint frame at theta=0
    rel_authored = np.linalg.inv(bodies["link5"]["world_T"]) @ bodies["gripper_link"]["world_T"]
    joint["authored_theta0_residual"] = float(
        np.abs(rel_authored - joint["T0"] @ _Rz(0.0) @ np.linalg.inv(joint["T1"])).max()
    )
    return {"bodies": bodies, "joint": joint}


def gripper_T_l5(joint: dict, q5_deg: float) -> np.ndarray:
    """gripper_link pose in link5 frame at joint coordinate q5 (deg).
    Sign basis: the UsdPhysics revolute convention (body1 rotates +theta
    right-handed about the joint frame +Z) — the limits [0, 90] pin the range,
    NOT the direction. Corroboration (adversarial review, 23rd): under the
    opposite sign the moving jaw interpenetrates link5's own hulls at legal
    angles (unbuildable mechanism); under +theta, zero self-overlap."""
    return joint["T0"] @ _Rz(math.radians(q5_deg)) @ np.linalg.inv(joint["T1"])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def concat_parts(parts: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.vstack([p["samples"] for p in parts])
    pid = np.concatenate([np.full(p["samples"].shape[0], i, dtype=np.int32) for i, p in enumerate(parts)])
    return pts, pid


def floor_scan(world_pts: np.ndarray, pid: np.ndarray, parts: list[dict], body: str,
               cyl_center: np.ndarray, cyl_r: float) -> tuple[float, list[dict]]:
    r = np.hypot(world_pts[:, 0] - cyl_center[0], world_pts[:, 1] - cyl_center[1])
    mask = r <= cyl_r
    if not mask.any():
        return math.inf, []
    z = world_pts[:, 2]
    floor_z = float(z[mask].min())
    rows = []
    for i, p in enumerate(parts):
        m = mask & (pid == i)
        if not m.any():
            continue
        zmin = float(z[m].min())
        if zmin <= floor_z + 0.0003:
            k = np.flatnonzero(m)[np.argmin(z[m])]
            rows.append({"body": body, "part": p["name"], "min_z_m": zmin, "r_at_min_mm": float(r[k] * 1000.0)})
    return floor_z, rows


def contact_scan(world_pts: np.ndarray, cyl_center: np.ndarray, cyl_top: float, cyl_bot: float,
                 cyl_r: float) -> dict:
    r = np.hypot(world_pts[:, 0] - cyl_center[0], world_pts[:, 1] - cyl_center[1])
    z = world_pts[:, 2]
    side = z < cyl_top
    in_fp = r <= cyl_r
    inside = in_fp & (z <= cyl_top) & (z >= cyl_bot)
    return {
        "contact": bool(inside.any()),
        "min_horiz_clearance_mm": float((r[side] - cyl_r).min() * 1000.0) if side.any() else None,
        "vertical_gap_mm": float((z[in_fp].min() - cyl_top) * 1000.0) if in_fp.any() else None,
        "min_struct_z_m": float(z.min()),
    }


def profile_tool_frame(pts_tool: np.ndarray, depth_lo_mm: float, depth_hi_mm: float) -> dict:
    """pts_tool: link5-frame points. depth = z - |TCP| (positive = below TCP,
    toward the object along the tool axis); r = distance to the tool axis."""
    depth_mm = (pts_tool[:, 2] - TCP_LOCAL[2]) * 1000.0
    r_mm = np.hypot(pts_tool[:, 0], pts_tool[:, 1]) * 1000.0
    edges = np.arange(depth_lo_mm, depth_hi_mm + DEPTH_BAND_MM, DEPTH_BAND_MM)
    bands = []
    for lo in edges[:-1]:
        m = (depth_mm >= lo) & (depth_mm < lo + DEPTH_BAND_MM)
        bands.append(float(r_mm[m].min()) if m.any() else None)
    in_fp = r_mm <= 14.5
    rim = (depth_mm >= RIM_BAND_MM[0]) & (depth_mm <= RIM_BAND_MM[1])
    return {
        "band_edges_mm": [float(v) for v in edges[:-1]],
        "min_r_mm_per_band": bands,
        "max_depth_in_footprint_mm": float(depth_mm[in_fp].max()) if in_fp.any() else None,
        "assembly_max_depth_mm": float(depth_mm.max()),
        "min_r_mm_in_rim_band": float(r_mm[rim].min()) if rim.any() else None,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, default="t3_jaw_audit")
    ap.add_argument("--extra_close_deg", type=float, nargs="*", default=[16.0, 8.0, 0.0])
    args = ap.parse_args()
    if args.tag.startswith("t3_grasp"):
        raise ValueError("tag must not collide with the frozen t3_grasp* artifact namespace")

    results_path = OUT_DIR / f"{args.tag}_results.json"
    parts_csv_path = OUT_DIR / f"{args.tag}_parts.csv"
    rrd_path = OUT_DIR / f"{args.tag}_timeline.rrd"
    rbl_path = OUT_DIR / f"{args.tag}_timeline.rbl"
    png_path = OUT_DIR / f"{args.tag}_inspection.png"
    validation_path = OUT_DIR / f"{args.tag}_rerun_validation.json"
    existing = [p.name for p in (results_path, parts_csv_path, rrd_path, rbl_path, png_path, validation_path) if p.exists()]
    if existing:
        print(f"[{LOG}] G0B_T3_JAW_AUDIT_VERDICT=AUDIT_ABORT preexisting_artifacts={existing}", flush=True)
        return 3

    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        print(f"[{LOG}] G0B_T3_JAW_AUDIT_VERDICT=AUDIT_ABORT rerun_version={rr.__version__}!={RERUN_VERSION}", flush=True)
        return 3
    import scipy

    print(f"[{LOG}] env python={sys.version.split()[0]} numpy={np.__version__} scipy={scipy.__version__} "
          f"rerun_sdk={rr.__version__} sample_spacing_mm={SAMPLE_SPACING_M*1000:.2f}", flush=True)
    print(f"[{LOG}] readonly_declaration=UsdStageOpen_only no_save_no_export (diagnosis, not re-decomposition; D415-3 untouched)", flush=True)

    # ---- G1: asset identity --------------------------------------------------
    root_sha = _sha256_file(ATTEMPT3_USD)
    physics_sha = _sha256_file(ATTEMPT3_PHYSICS_LAYER)
    g1 = root_sha == ATTEMPT3_ROOT_SHA256 and physics_sha == ATTEMPT3_PHYSICS_SHA256
    print(f"[{LOG}] G1_sha_pin pass={g1} root={root_sha[:16]} physics={physics_sha[:16]}", flush=True)
    if not g1:
        print(f"[{LOG}] G0B_T3_JAW_AUDIT_VERDICT=AUDIT_ABORT sha_mismatch", flush=True)
        return 3

    # ---- extract + G2: structure --------------------------------------------
    asset = extract_asset()
    g2 = True
    for label, b in asset["bodies"].items():
        n = len(b["parts"])
        legacy_ok = len(b["legacy"]) == 1 and b["legacy"][0][1] is False
        ok = n == EXPECTED_PART_COUNT and legacy_ok and not b["approx_bad"]
        g2 = g2 and ok
        n_samples = sum(p["samples"].shape[0] for p in b["parts"])
        print(f"[{LOG}] G2_structure body={label} parts={n}/{EXPECTED_PART_COUNT} "
              f"legacy_disabled_exact_one={legacy_ok} approx_bad={len(b['approx_bad'])} "
              f"hull_fallback={sum(0 if p['hull_ok'] else 1 for p in b['parts'])} samples={n_samples} pass={ok}", flush=True)
    if not g2:
        print(f"[{LOG}] G0B_T3_JAW_AUDIT_VERDICT=AUDIT_ABORT structure_mismatch", flush=True)
        return 3

    # ---- G3: joint frame witness --------------------------------------------
    joint = asset["joint"]
    g3 = (
        joint["axis"] == "Z"
        and joint["body0"] == [BODY_PRIMS["link5"]]
        and joint["body1"] == [BODY_PRIMS["gripper_link"]]
        and joint["authored_theta0_residual"] < 1e-5
        and abs(joint["lower_deg"]) < 0.1
        and 89.0 < joint["upper_deg"] < 91.0
    )
    print(f"[{LOG}] G3_joint_frame pass={g3} axis={joint['axis']} limits=[{joint['lower_deg']:.3f},{joint['upper_deg']:.3f}]deg "
          f"authored_theta0_residual={joint['authored_theta0_residual']:.2e} "
          f"sign_basis=usd_limits_0_to_90_with_authored_closed", flush=True)
    if not g3:
        print(f"[{LOG}] G0B_T3_JAW_AUDIT_VERDICT=AUDIT_ABORT joint_frame_mismatch", flush=True)
        return 3

    # ---- G4: FK chain-split self check --------------------------------------
    q_test = np.array([-42.465, 40.638, 75.116, 64.963, 0.0, 88.31])
    tcp_full, _ = fk_full(q_test)
    tcp_split = transform_pts(fk_T_link5(q_test), TCP_LOCAL[None, :])[0]
    g4_res = float(np.linalg.norm(tcp_full - tcp_split))
    g4 = g4_res < 1e-9
    print(f"[{LOG}] G4_fk_split pass={g4} residual_m={g4_res:.3e}", flush=True)

    # ---- G5: env TCP constant cross-check (text parse — no isaac import) -----
    env_src = (REPO / "roarm_rl/roarm_stack_env.py").read_text()
    m = re.search(r"TCP_LOCAL_OFFSET_M\s*=\s*\(([^)]*)\)", env_src)
    env_tcp = np.array([float(v) for v in m.group(1).split(",")]) if m else None
    g5 = env_tcp is not None and np.allclose(env_tcp, TCP_LOCAL, atol=1e-9)
    print(f"[{LOG}] G5_env_tcp pass={g5} env_value={None if env_tcp is None else env_tcp.tolist()}", flush=True)
    if not (g4 and g5):
        print(f"[{LOG}] G0B_T3_JAW_AUDIT_VERDICT=AUDIT_ABORT frame_check_failed", flush=True)
        return 3

    # ---- measured inputs -----------------------------------------------------
    a1 = load_attempt(A1_RESULTS, A1_CSV, "descend", "last")
    a4 = load_attempt(A4_RESULTS, A4_CSV, "descend", "last")
    a2 = load_attempt(A2_RESULTS, A2_CSV, "close", "mean")
    for label, at in (("a1", a1), ("a4", a4), ("a2", a2)):
        print(f"[{LOG}] input {label} tag={at['tag']} verdict={at['verdict']} q5_pose={at['q5_meas_deg']:.2f}deg "
              f"tcp_meas=({at['tcp_meas'][0]:+.6f},{at['tcp_meas'][1]:+.6f},{at['tcp_meas'][2]:+.6f}) "
              f"source={at['pose_source']}", flush=True)

    link5_pts_l, link5_pid = concat_parts(asset["bodies"]["link5"]["parts"])
    grip_pts_g, grip_pid = concat_parts(asset["bodies"]["gripper_link"]["parts"])

    def pose_T(at: dict) -> np.ndarray:
        R = fk_T_link5(at["q_descend_deg"])[:3, :3]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = at["tcp_meas"] - R @ TCP_LOCAL
        return T

    # ---- CA1/CA4: floor replication -----------------------------------------
    floor_results = {}
    for label, at in (("a1", a1), ("a4", a4)):
        T = pose_T(at)
        Xg = T @ gripper_T_l5(joint, at["q5_meas_deg"])
        w_l5 = transform_pts(T, link5_pts_l)
        w_g = transform_pts(Xg, grip_pts_g)
        fz5, rows5 = floor_scan(w_l5, link5_pid, asset["bodies"]["link5"]["parts"], "link5", at["cyl_center"], at["cyl_r"])
        fzg, rowsg = floor_scan(w_g, grip_pid, asset["bodies"]["gripper_link"]["parts"], "gripper_link", at["cyl_center"], at["cyl_r"])
        floor_z = min(fz5, fzg)
        rows = [r for r in rows5 + rowsg if r["min_z_m"] <= floor_z + 0.0003]
        clearance_mm = (floor_z - at["cyl_top_z"]) * 1000.0
        d_floor_mm = (at["tcp_meas"][2] - floor_z) * 1000.0
        floor_results[label] = {
            "floor_z_m": float(floor_z),
            "clearance_mm": float(clearance_mm),
            "d_floor_mm": float(d_floor_mm),
            "floor_parts": rows,
            "gate_pass": bool(abs(clearance_mm) <= FLOOR_GATE_MM),
        }
        print(f"[{LOG}] floor {label} q5={at['q5_meas_deg']:.2f}deg floor_z={floor_z:+.6f} cyl_top={at['cyl_top_z']:+.6f} "
              f"clearance_mm={clearance_mm:+.3f} d_floor_mm={d_floor_mm:.3f} gate(|c|<= {FLOOR_GATE_MM})={abs(clearance_mm) <= FLOOR_GATE_MM}", flush=True)
        for r in rows:
            print(f"[{LOG}] floor_part {label} body={r['body']} part={r['part']} min_z={r['min_z_m']:+.6f} r_mm={r['r_at_min_mm']:.2f}", flush=True)
    ca1 = floor_results["a1"]["gate_pass"]
    ca4 = floor_results["a4"]["gate_pass"]
    # native python types only — np.bool_/np.float64 through json default=str
    # serialize as STRINGS ("True" is truthy even when "False") — audit3 lesson.
    angle_inv_mm = float(abs(floor_results["a1"]["d_floor_mm"] - floor_results["a4"]["d_floor_mm"]))
    angle_inv = bool(angle_inv_mm <= ANGLE_INV_GATE_MM)
    print(f"[{LOG}] angle_invariance |d_floor(a1)-d_floor(a4)|={angle_inv_mm:.3f}mm gate(<= {ANGLE_INV_GATE_MM})={angle_inv}", flush=True)

    # ---- CB: attempt2 close sweep must be contact-free ----------------------
    T2 = pose_T(a2)
    w_l5_a2 = transform_pts(T2, link5_pts_l)
    cyl_bot = a2["cyl_top_z"] - a2["cyl_h"]
    sweep_rows = []
    cb = True
    for q5 in a2["close_deg"]:
        w_g = transform_pts(T2 @ gripper_T_l5(joint, q5), grip_pts_g)
        both = np.vstack([w_l5_a2, w_g])
        scan = contact_scan(both, a2["cyl_center"], a2["cyl_top_z"], cyl_bot, a2["cyl_r"])
        scan["q5_deg"] = float(q5)
        sweep_rows.append(scan)
        cb = cb and not scan["contact"]
        print(f"[{LOG}] sweep q5={q5:7.3f}deg contact={scan['contact']} "
              f"min_horiz_clearance_mm={'None' if scan['min_horiz_clearance_mm'] is None else format(scan['min_horiz_clearance_mm'], '+.3f')} "
              f"vertical_gap_mm={'None' if scan['vertical_gap_mm'] is None else format(scan['vertical_gap_mm'], '+.3f')} "
              f"min_struct_z={scan['min_struct_z_m']:+.6f}", flush=True)
    print(f"[{LOG}] CB_no_contact_across_band pass={cb}", flush=True)

    # ---- throat profile (nominal tool frame, pose-free) ----------------------
    profile_band = [float(v) for v in a2["close_deg"]] + [float(v) for v in args.extra_close_deg]
    profiles = []
    for q5 in profile_band:
        pts_tool = np.vstack([link5_pts_l, transform_pts(gripper_T_l5(joint, q5), grip_pts_g)])
        prof = profile_tool_frame(pts_tool, -2.0, 30.0)
        prof["q5_deg"] = float(q5)
        prof["in_measured_band"] = q5 in a2["close_deg"]
        profiles.append(prof)
        band_marker = "" if prof["in_measured_band"] else " (beyond-band diagnostic)"
        print(f"[{LOG}] profile{band_marker} q5={q5:7.3f}deg max_depth_in_footprint_mm="
              f"{'None' if prof['max_depth_in_footprint_mm'] is None else format(prof['max_depth_in_footprint_mm'], '.3f')} "
              f"assembly_max_depth_mm={prof['assembly_max_depth_mm']:.3f} "
              f"min_r_mm_in_rim_band(5-15mm)={'None' if prof['min_r_mm_in_rim_band'] is None else format(prof['min_r_mm_in_rim_band'], '.2f')}", flush=True)

    # ---- per-part table ------------------------------------------------------
    part_rows = []
    for q5 in (Q5_OPEN_DEG, 45.0, 24.0, 0.0):
        Xg = gripper_T_l5(joint, q5)
        for body, pts_l, pid, parts in (
            ("link5", link5_pts_l, link5_pid, asset["bodies"]["link5"]["parts"]),
            ("gripper_link", transform_pts(Xg, grip_pts_g), grip_pid, asset["bodies"]["gripper_link"]["parts"]),
        ):
            depth_mm = (pts_l[:, 2] - TCP_LOCAL[2]) * 1000.0
            r_mm = np.hypot(pts_l[:, 0], pts_l[:, 1]) * 1000.0
            for i, p in enumerate(parts):
                m = pid == i
                d = depth_mm[m]
                r = r_mm[m]
                fp = m & (r_mm <= 14.5)
                rim = m & (depth_mm >= RIM_BAND_MM[0]) & (depth_mm <= RIM_BAND_MM[1])
                part_rows.append([
                    f"{q5:.4f}", body, p["name"], p["n_raw"], int(p["hull_ok"]),
                    f"{d.max():.3f}", f"{r[np.argmax(d)]:.2f}",
                    f"{depth_mm[fp].max():.3f}" if fp.any() else "",
                    f"{r_mm[rim].min():.2f}" if rim.any() else "",
                ])
    with parts_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q5_deg", "body", "part", "n_raw_points", "hull_ok",
                    "max_depth_mm", "r_at_max_depth_mm", "max_depth_in_footprint_mm", "min_r_mm_in_rim_band"])
        w.writerows(part_rows)

    consistent = ca1 and ca4 and angle_inv and cb
    verdict = "JAW_AUDIT_CONSISTENT" if consistent else "GEOMETRY_MISMATCH"

    # ---- results JSON --------------------------------------------------------
    results_doc = {
        "artifact": "G0B_T3_ATTEMPT3_JAW_THROAT_OCCLUSION_VERTEX_AUDIT_V1",
        "case": "g0b_d420",
        "tag": args.tag,
        "verdict": verdict,
        "layer": "read-only geometry diagnosis (not re-decomposition; not a physics or real-gripper claim)",
        "usd": {
            "root_path": str(ATTEMPT3_USD),
            "root_sha256": root_sha,
            "physics_sha256": physics_sha,
            "part_counts": {k: len(v["parts"]) for k, v in asset["bodies"].items()},
            "hull_fallback_parts": {k: [p["name"] for p in v["parts"] if not p["hull_ok"]] for k, v in asset["bodies"].items()},
        },
        "frames": {
            "tcp_local_m": TCP_LOCAL.tolist(),
            "env_tcp_match": bool(g5),
            "fk_split_residual_m": g4_res,
            "joint_axis": joint["axis"],
            "joint_limits_deg": [joint["lower_deg"], joint["upper_deg"]],
            "authored_theta0_residual": joint["authored_theta0_residual"],
            "sign_convention": "q5_rad positive right-handed about joint frame +Z; authored=0=closed (USD limits)",
        },
        "gates": {
            "G1_sha_pin": g1, "G2_structure": g2, "G3_joint_frame": g3, "G4_fk_split": g4, "G5_env_tcp": g5,
            "CA1_a1_floor": ca1, "CA4_a4_floor": ca4,
            "ANGLE_INV": angle_inv, "angle_inv_delta_mm": angle_inv_mm,
            "CB_no_contact": cb,
            "floor_gate_mm": FLOOR_GATE_MM, "angle_inv_gate_mm": ANGLE_INV_GATE_MM,
            "sample_spacing_mm": SAMPLE_SPACING_M * 1000.0,
        },
        "inputs": {
            k: {
                "tag": at["tag"], "verdict": at["verdict"], "pose_source": at["pose_source"],
                "tcp_meas_m": at["tcp_meas"].tolist(), "q5_pose_deg": at["q5_meas_deg"],
                "q_descend_deg": at["q_descend_deg"].tolist(),
                "cyl_center_m": at["cyl_center"].tolist(), "cyl_top_z_m": at["cyl_top_z"],
                "cyl_r_m": at["cyl_r"],
            }
            for k, at in (("a1", a1), ("a2", a2), ("a4", a4))
        },
        "floor": floor_results,
        "sweep_a2": sweep_rows,
        "profile": profiles,
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "rerun_sdk": RERUN_VERSION},
    }
    results_path.write_text(json.dumps(results_doc, indent=2, default=str) + "\n")

    # ---- D341 Rerun artifact (save-only) ------------------------------------
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    app_id = f"roarm_g0b_{args.tag}"

    def view_pts(pts_l5: np.ndarray) -> np.ndarray:
        # tool-frame view: origin=TCP, +z up; depth below TCP is negative z.
        out = pts_l5 - TCP_LOCAL[None, :]
        return np.column_stack([out[:, 0], out[:, 1], -out[:, 2]])

    def cyl_strips(at: dict) -> list[list[list[float]]]:
        # cylinder drawn in the a-pose tool frame (exact inverse of pose_T)
        T = pose_T(at)
        Tinv = np.linalg.inv(T)
        strips = []
        for z in (at["cyl_top_z"], at["cyl_top_z"] - at["cyl_h"]):
            ring = []
            for k in range(49):
                ang = 2.0 * math.pi * k / 48
                pw = np.array([at["cyl_center"][0] + at["cyl_r"] * math.cos(ang),
                               at["cyl_center"][1] + at["cyl_r"] * math.sin(ang), z])
                ring.append(view_pts(transform_pts(Tinv, pw[None, :]))[0].tolist())
            strips.append(ring)
        return strips

    floor_part_names = {(r["body"], r["part"]) for r in floor_results["a1"]["floor_parts"]}
    floor_mask_l5 = np.zeros(link5_pts_l.shape[0], dtype=bool)
    for i, p in enumerate(asset["bodies"]["link5"]["parts"]):
        if ("link5", p["name"]) in floor_part_names:
            floor_mask_l5 |= link5_pid == i
    floor_mask_g = np.zeros(grip_pts_g.shape[0], dtype=bool)
    for i, p in enumerate(asset["bodies"]["gripper_link"]["parts"]):
        if ("gripper_link", p["name"]) in floor_part_names:
            floor_mask_g |= grip_pid == i

    with rr.RecordingStream(app_id, recording_id=f"g0b_d420_{args.tag}", make_default=False, send_properties=True) as rec:
        rec.save(str(rrd_path), write_footer=True)
        rec.log("audit/link5_parts", rr.Points3D(view_pts(link5_pts_l[~floor_mask_l5][::VIEW_STRIDE]), colors=[150, 150, 160], radii=0.0004), static=True)
        rec.log("audit/cylinder_a1", rr.LineStrips3D(cyl_strips(a1), colors=[[230, 120, 40]] * 2, radii=0.0004), static=True)
        rec.log("audit/cylinder_a2", rr.LineStrips3D(cyl_strips(a2), colors=[[210, 170, 110]] * 2, radii=0.0004), static=True)
        rec.log("audit/tcp", rr.Points3D([[0.0, 0.0, 0.0]], colors=[[40, 200, 80]], radii=0.002), static=True)
        rec.log("audit/tool_axis", rr.LineStrips3D([[[0.0, 0.0, 0.06], [0.0, 0.0, -0.06]]], colors=[[40, 200, 80]], radii=0.0003), static=True)
        for i, q5 in enumerate(profile_band):
            rec.reset_time()
            rec.set_time("band_index", sequence=i)
            g_now = transform_pts(gripper_T_l5(joint, q5), grip_pts_g)
            rec.log("audit/gripper_parts", rr.Points3D(view_pts(g_now[~floor_mask_g][::VIEW_STRIDE]), colors=[70, 130, 230], radii=0.0004))
            fl = np.vstack([link5_pts_l[floor_mask_l5], g_now[floor_mask_g]]) if (floor_mask_l5.any() or floor_mask_g.any()) else np.zeros((0, 3))
            rec.log("audit/floor_parts", rr.Points3D(view_pts(fl), colors=[235, 60, 60], radii=0.0006))
            rec.log("plots/q5_deg", rr.Scalars(float(q5)))
            prof = profiles[i]
            rec.log("plots/assembly_max_depth_mm", rr.Scalars(float(prof["assembly_max_depth_mm"])))
            rec.log("plots/footprint_floor_depth_mm", rr.Scalars(
                float("nan") if prof["max_depth_in_footprint_mm"] is None else float(prof["max_depth_in_footprint_mm"])))
            match = [srow for srow in sweep_rows if abs(srow["q5_deg"] - q5) < 1e-9]
            if match:
                # contract repair (audit1 lesson): always log the declared plot
                # entities — NaN marks "undefined at this angle", never absence.
                mh = match[0]["min_horiz_clearance_mm"]
                vg = match[0]["vertical_gap_mm"]
                rec.log("plots/min_horiz_clearance_mm", rr.Scalars(float("nan") if mh is None else float(mh)))
                rec.log("plots/vertical_gap_mm", rr.Scalars(float("nan") if vg is None else float(vg)))
        rec.reset_time()
        # audit3 lesson: without a band_index stamp the gates land on log_time
        # only and the TextLog view is empty under the band_index timeline.
        rec.set_time("band_index", sequence=0)
        for name, ok in (("G1_sha_pin", g1), ("G2_structure", g2), ("G3_joint_frame", g3), ("G4_fk_split", g4),
                         ("G5_env_tcp", g5), ("CA1_a1_floor", ca1), ("CA4_a4_floor", ca4),
                         ("ANGLE_INV", angle_inv), ("CB_no_contact", cb)):
            rec.log("events/gates", rr.TextLog(f"{name} pass={ok}", level=rr.TextLogLevel.INFO if ok else rr.TextLogLevel.ERROR))
        summary_md = (
            f"# G0b T3 jaw/throat occlusion vertex audit (case g0b_d420)\n\n"
            f"- verdict: **{verdict}** (read-only geometry diagnosis — not re-decomposition, not a physics/real claim)\n"
            f"- asset: attempt3 64+64 frozen, root {ATTEMPT3_ROOT_SHA256[:16]}, physics {ATTEMPT3_PHYSICS_SHA256[:16]}\n"
            f"- floor a1 (q5={a1['q5_meas_deg']:.2f}deg): clearance {floor_results['a1']['clearance_mm']:+.3f} mm, "
            f"d_floor {floor_results['a1']['d_floor_mm']:.3f} mm\n"
            f"- floor a4 (q5={a4['q5_meas_deg']:.2f}deg): clearance {floor_results['a4']['clearance_mm']:+.3f} mm, "
            f"d_floor {floor_results['a4']['d_floor_mm']:.3f} mm (angle-invariance delta {angle_inv_mm:.3f} mm)\n"
            f"- floor parts: {sorted({(r['body'] + '/' + r['part']) for r in floor_results['a1']['floor_parts']})}\n"
            f"- a2 sweep: contact-free={cb} across {len(a2['close_deg'])} band angles\n"
            f"- sampling: hull-face {SAMPLE_SPACING_M * 1000:.2f} mm pitch (stated error bound); "
            f"view decimated 1/{VIEW_STRIDE} (inspection-only); authority = stdout + results JSON\n"
        )
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN), static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run", name="1 | audit verdict + contract"),
                    rrb.Spatial3DView(origin="/", contents=["/audit/**"], name="2 | tool-frame jaw geometry vs cylinder"),
                    rrb.TextLogView(origin="/events/gates", contents="/events/gates/**", name="3 | gates"),
                    column_shares=[0.28, 0.47, 0.25],
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/plots", contents=["/plots/q5_deg/**", "/plots/footprint_floor_depth_mm/**"], name="4 | q5 + footprint floor depth"),
                    rrb.TimeSeriesView(origin="/plots", contents=["/plots/min_horiz_clearance_mm/**", "/plots/assembly_max_depth_mm/**"], name="5 | clearance + assembly depth"),
                ),
                row_shares=[0.58, 0.42],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=True,
        )
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=30.0)
    blueprint.save(app_id, str(rbl_path))

    expected_entities = [
        "metadata/run", "audit/link5_parts", "audit/gripper_parts", "audit/floor_parts",
        "audit/cylinder_a1", "audit/cylinder_a2", "audit/tcp", "audit/tool_axis",
        "plots/q5_deg", "plots/footprint_floor_depth_mm", "plots/min_horiz_clearance_mm",
        "plots/vertical_gap_mm", "plots/assembly_max_depth_mm", "events/gates",
    ]
    components = {
        "metadata/run": ["TextDocument:text"],
        "audit/link5_parts": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "audit/gripper_parts": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "audit/floor_parts": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "audit/cylinder_a1": ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"],
        "audit/cylinder_a2": ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"],
        "audit/tcp": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "audit/tool_axis": ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"],
        "plots/q5_deg": ["Scalars:scalars"],
        "plots/footprint_floor_depth_mm": ["Scalars:scalars"],
        "plots/min_horiz_clearance_mm": ["Scalars:scalars"],
        "plots/vertical_gap_mm": ["Scalars:scalars"],
        "plots/assembly_max_depth_mm": ["Scalars:scalars"],
        "events/gates": ["TextLog:text", "TextLog:level"],
    }
    validation = validate_rerun_artifact(
        rrd_path,
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["band_index", "blueprint", "log_time"],
        expected_entity_components=components,
        blueprint_path=rbl_path,
        screenshot_path=png_path,
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=180.0,
    )
    validation_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(f"[{LOG}] rerun_validation pass={validation.get('pass')} errors={validation.get('errors')}", flush=True)
    print(f"[{LOG}] artifacts rrd={rrd_path.name} sha={_sha256_file(rrd_path)[:16]} results={results_path.name} "
          f"parts_csv={parts_csv_path.name}", flush=True)
    print(f"[{LOG}] G0B_T3_JAW_AUDIT_VERDICT={verdict}", flush=True)
    return 0 if consistent and validation.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
