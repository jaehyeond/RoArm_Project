"""G0b T3R Gate-0 — visual-mesh distal-finger depth READ-ONLY audit (D426-⑤ step 1).

Case g0b_d420, repair track (t3r_*). Authoring-source layer ONLY:
  - READ-ONLY: binary STL reads + pxr Usd.Stage.Open (joint frame witness only).
    No Save()/Export()/layer-edit API exists anywhere in this file. Kit is NOT
    launched. No re-decomposition (D415-③ untouched; B/F/D are incremental
    authoring per D426-④, and this gate runs BEFORE any authoring).
  - Verdict layer: presence/depth of distal finger geometry in the URDF VISUAL
    meshes, as the authoring source for Arm-F/D incremental collision parts.
    It makes NO physics claim, NO real-gripper claim, and NO claim that the
    visual mesh matches the physical gripper (that comparison is T4 layer).

Question (failable, session-progress rule): D368 circumstantially suggested the
visual meshes may LACK distal finger geometry. Do the pinned visual sources
  gripper_link.stl (sha 7946a374…)  — moving jaw body
  link5.stl        (sha 1d63f374…)  — fixed jaw body
contain geometry reaching depth >= L_MIN below TCP (toward the object, tool
frame) within the finger search window, for BOTH bodies (24th doc §4-2: both
bodies need distal authoring)? gripper_left_link.stl is a dead asset (URDF
non-referenced) and MUST NOT be used (D426-⑤).

Preregistered gate constants (fixed BEFORE any mesh was profiled — no peeking):
  L_MIN_MM        = 5.5   minimum viable finger length from x = L - m with
                          margin m = 5.5 mm (attempt2 precedent) and T1-real
                          grip depth x in (0, 12] mm  =>  L in (5.5, 17.5] mm
                          (24th doc §4-1). Source geometry shallower than this
                          cannot yield an in-range L.
  INDET_HALF_MM   = 0.5   |L_vis - L_MIN| <= 0.5 mm  => indeterminate
                          (24th tolerance rule: |margin| < 0.5 mm predictions
                          are indeterminate, applied symmetrically here).
  FINGER_WINDOW_R_MM = 30.0   tool-axis radial window that bounds "finger/jaw
                          structure near the grasp" vs arm body further out.
  WALL_ANNULUS_MM = (12.5, 20.0)  side-wall-contactable radial band around the
                          cylinder wall r=14.5 mm (report-only, no gate).
Per-body tri-state: PASS if L_vis > L_MIN + 0.5, FAIL if L_vis < L_MIN - 0.5,
else INDETERMINATE. Overall verdict:
  both PASS                        -> GATE0_SOURCE_PRESENT
  any FAIL                         -> GATE0_SOURCE_ABSENT
  otherwise                        -> GATE0_INDETERMINATE
Identity/frame/contract failure    -> GATE0_ABORT
On SOURCE_ABSENT the only branches are: user-approved hand-authoring, or stop
and re-consult (Arm-C fallback is extinct per D426-①) — user re-query is
mandatory; this script only reports.

Panel-repair lineage (wf_f6c65ef4, 29th doc §4-1 — pre-run preregistration
edits, applied BEFORE any execution): authored rev sha256 81259edc… -> this rev.
  M1 coarse-peak pose logged to RRD (anchor_ids | {coarse argmax});
  M2 peak_metrics at the refined peak q5 + bands CSV peak_<q5> rows + peak-point
     azimuth/body-local coordinates for both bodies;
  M3 report-only secondary metrics l_vis_wall_mm (12.5<=r<=20) and
     l_vis_grasp_range_mm (moving jaw, q5 in [0,45]) + interpretation rules in
     the results JSON (gate unchanged — one-sided false-PRESENT localization).
  Minor set: empty-finger-window -> GATE0_ABORT guard (write-free); GV4 URDF
  <axis> element gate + URDF sha256 record; strict-JSON non-finite -> null
  sanitize; early authority-JSON verdict print; sweep/refine depth error bounds
  preregistered in gates; Q5_OPEN comment fix. L_MIN anchors to m=5.5 (attempt2
  precedent), NOT the D426 G-e floor 4.5, because grip depths x <= 1.0 mm are
  not operationally usable — L_vis in [4.5,5.0) still verdicts FAIL.
  Gate constants, tri-state thresholds, verdict mapping, sweep/refine grids,
  and sampling pitch are unchanged from the authored rev.

Authority: stdout verdict line + t3r_gate0_vismesh_results.json (exit code is
NOT a verdict channel, D424-③). Float32 Rerun copies are inspection-only.
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
_REEXEC_FLAG = "G0B_GATE0_REEXEC"


def _bootstrap_pxr_env() -> None:
    """Re-exec once with pxr module + shared-library paths (same proven pattern
    as the 23rd jaw audit). Kit is NOT launched — plain-python process."""
    if os.environ.get(_REEXEC_FLAG) == "1":
        return
    if not USD_LIBS.is_dir():
        print(f"[g0b_t3r_gate0] G0B_T3R_GATE0_VERDICT=GATE0_ABORT missing usd libs {USD_LIBS}", flush=True)
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
import struct  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import numpy as np  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim_scripts"))

from roarm_kinematics import Tmat  # noqa: E402  (URDF rpy convention, G4-proven in 23rd)

LOG = "g0b_t3r_gate0"
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"

# --- pinned visual sources (D426-⑤ / 24th §4-4) -------------------------------
URDF_PATH = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
MESH_DIR = URDF_PATH.parent / "meshes"
VIS_PINS = {
    "gripper_link": ("gripper_link.stl", "7946a374e24a2f467a0581b4946e0ec41b1b86a92f070bc00aa9bced1bf65a56"),
    "link5": ("link5.stl", "1d63f374a78c1419b21eec63fa8efeef40d0d42ca89c5de3ceb0d86476d9c7eb"),
}
DEAD_ASSET = "gripper_left_link.stl"  # URDF non-referenced; forbidden as source
STL_SCALE = 0.001  # URDF <mesh scale="0.001 …"> — STL units are mm

# --- frozen attempt3 asset identity (joint-frame witness source; same pins as
#     the 23rd audit / p9 D-3). Opened READ-ONLY for the revolute joint only. --
ATTEMPT3_USD = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3"
    / "roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd"
)
ATTEMPT3_ROOT_SHA256 = "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff"
ATTEMPT3_PHYSICS_LAYER = ATTEMPT3_USD.parent / "configuration/roarm_m3_physics.usd"
ATTEMPT3_PHYSICS_SHA256 = "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503"
JOINT_PRIM = "/roarm_m3/joints/link5_to_gripper_link"
BODY_PRIMS = {"link5": "/roarm_m3/link5", "gripper_link": "/roarm_m3/gripper_link"}

AUDIT3_RESULTS = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420/t3_jaw_audit3_results.json"

TCP_LOCAL = np.array([0.0, 0.0, 0.115428], dtype=np.float64)  # link5 frame (URDF link5_to_hand_tcp)
Q5_OPEN_DEG = math.degrees(1.5413)  # 88.3100 (D-1 frozen; anchor keys read "88.3100")

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"

# --- preregistered measurement constants (header rationale) -------------------
L_MIN_MM = 5.5
INDET_HALF_MM = 0.5
FINGER_WINDOW_R_MM = 30.0
WALL_ANNULUS_MM = (12.5, 20.0)
FOOTPRINT_R_MM = 14.5           # cylinder D29 radius (matches 23rd audit)
RIM_BAND_MM = (5.0, 15.0)       # T1 real rim-pinch depth region (23rd constant)
SAMPLE_SPACING_M = 0.0005       # triangle-surface sampling pitch (error bound)
DEPTH_BAND_MM = 0.5
DEPTH_LO_MM, DEPTH_HI_MM = -5.0, 30.0   # band range (expanded vs 23rd -2..30)
SWEEP_STEP_DEG = 0.25
REFINE_STEP_DEG = 0.02
REFINE_HALF_DEG = 1.0
ANCHOR_Q5_DEG = [Q5_OPEN_DEG, 60.0, 45.0, 24.0, 5.0, 0.0]  # 3D-view poses
VIEW_STRIDE = 16                # Rerun view decimation (D425-②/28th accel-⑤ default)
JOINT_WITNESS_TOL = 1e-5
HOVER_MARGIN_REF_MM = 4.5       # Arm-F hover reference (viz cylinder placement only)
CYL_R_MM, CYL_H_MM = 14.5, 50.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def transform_pts(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def load_binary_stl(path: Path) -> np.ndarray:
    """Strict binary-STL triangle loader -> (n_tri, 3, 3) float64 in FILE units.
    The size formula (84 + 50*n bytes) is the decisive binary check; an ASCII
    STL cannot satisfy it except by pathological coincidence, which the
    tri-count sanity bound also guards."""
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path.name}: too small for binary STL ({len(raw)} B)")
    (n_tri,) = struct.unpack_from("<I", raw, 80)
    if len(raw) != 84 + 50 * n_tri:
        raise ValueError(f"{path.name}: size {len(raw)} != 84+50*{n_tri} (not binary STL)")
    if not (1 <= n_tri <= 2_000_000):
        raise ValueError(f"{path.name}: implausible triangle count {n_tri}")
    arr = np.frombuffer(raw, dtype=np.uint8, offset=84).reshape(n_tri, 50)
    tris = arr[:, :48].copy().view("<f4").reshape(n_tri, 4, 3)[:, 1:, :]  # drop normal row
    return tris.astype(np.float64)


def sample_triangles(tris_m: np.ndarray, spacing_m: float) -> np.ndarray:
    """Vertices + dense surface samples of a RAW triangle soup (visual meshes
    are non-convex — no hulling here, unlike the collision-part audit)."""
    out = [tris_m.reshape(-1, 3)]
    e1 = np.linalg.norm(tris_m[:, 1] - tris_m[:, 0], axis=1)
    e2 = np.linalg.norm(tris_m[:, 2] - tris_m[:, 0], axis=1)
    e3 = np.linalg.norm(tris_m[:, 2] - tris_m[:, 1], axis=1)
    n_per = np.maximum(1, np.ceil(np.maximum(e1, np.maximum(e2, e3)) / spacing_m).astype(np.int64))
    for n in np.unique(n_per):
        idx = np.flatnonzero(n_per == n)
        ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        mask = (ii + jj) <= n
        u = (ii[mask] / n)[None, :, None]
        v = (jj[mask] / n)[None, :, None]
        a = tris_m[idx, 0][:, None, :]
        b = tris_m[idx, 1][:, None, :]
        c = tris_m[idx, 2][:, None, :]
        out.append((a + u * (b - a) + v * (c - a)).reshape(-1, 3))
    return np.vstack(out)


def depth_r_mm(pts_l5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tool-frame depth/radius (same convention as the 23rd audit):
    depth = (z - TCP_z) * 1000, positive = below TCP toward the object;
    r = distance to the tool axis, mm."""
    depth = (pts_l5[:, 2] - TCP_LOCAL[2]) * 1000.0
    r = np.hypot(pts_l5[:, 0], pts_l5[:, 1]) * 1000.0
    return depth, r


def l_vis_mm(depth: np.ndarray, r: np.ndarray) -> float:
    m = r <= FINGER_WINDOW_R_MM
    return float(depth[m].max()) if m.any() else -math.inf


def body_metrics(depth: np.ndarray, r: np.ndarray) -> dict:
    win = r <= FINGER_WINDOW_R_MM
    fp = r <= FOOTPRINT_R_MM
    wall = (r >= WALL_ANNULUS_MM[0]) & (r <= WALL_ANNULUS_MM[1])
    rim = (depth >= RIM_BAND_MM[0]) & (depth <= RIM_BAND_MM[1])
    k = int(np.argmax(np.where(win, depth, -np.inf))) if win.any() else -1
    return {
        "l_vis_mm": l_vis_mm(depth, r),
        "r_at_l_vis_mm": float(r[k]) if k >= 0 else None,
        "max_depth_in_footprint_mm": float(depth[fp].max()) if fp.any() else None,
        "max_depth_in_wall_annulus_mm": float(depth[wall].max()) if wall.any() else None,
        "min_r_mm_in_rim_band": float(r[rim].min()) if rim.any() else None,
        "n_pts_in_rim_band": int(rim.sum()),
        "n_pts": int(depth.shape[0]),
    }


def _finite_or_none(v: float | None) -> float | None:
    """Strict-JSON sanitize: -inf/inf/nan -> None (null = no sampled point inside
    the referenced window at that pose). tri_state keeps raw -inf semantics."""
    return None if v is None or not math.isfinite(v) else float(v)


def _san_metrics(m: dict) -> dict:
    return {key: (_finite_or_none(val) if isinstance(val, float) else val) for key, val in m.items()}


def peak_point(depth: np.ndarray, r: np.ndarray, pts_l5: np.ndarray, pts_body: np.ndarray) -> dict | None:
    """Verdict-driving deepest in-window point, localized (panel M2 — report-only):
    tool-frame azimuth atan2(y,x) + tool-frame and body-local coordinates."""
    win = r <= FINGER_WINDOW_R_MM
    if not win.any():
        return None
    j = int(np.argmax(np.where(win, depth, -np.inf)))
    return {
        "depth_mm": float(depth[j]),
        "r_mm": float(r[j]),
        "azimuth_deg": float(math.degrees(math.atan2(pts_l5[j, 1], pts_l5[j, 0]))),
        "tool_frame_m": [float(v) for v in pts_l5[j]],
        "body_frame_m": [float(v) for v in pts_body[j]],
    }


def band_rows(body: str, q5_label: str, depth: np.ndarray, r: np.ndarray) -> list[list]:
    rows = []
    edges = np.arange(DEPTH_LO_MM, DEPTH_HI_MM + DEPTH_BAND_MM, DEPTH_BAND_MM)
    for lo in edges[:-1]:
        m = (depth >= lo) & (depth < lo + DEPTH_BAND_MM) & (r <= FINGER_WINDOW_R_MM)
        rows.append([
            body, q5_label, f"{lo:.1f}", f"{lo + DEPTH_BAND_MM:.1f}", int(m.sum()),
            f"{r[m].min():.2f}" if m.any() else "", f"{r[m].max():.2f}" if m.any() else "",
        ])
    return rows


def tri_state(l_vis: float) -> str:
    if l_vis > L_MIN_MM + INDET_HALF_MM:
        return "PASS"
    if l_vis < L_MIN_MM - INDET_HALF_MM:
        return "FAIL"
    return "INDETERMINATE"


# ---------------------------------------------------------------------------
def extract_usd_joint() -> dict:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(ATTEMPT3_USD))
    rj = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(JOINT_PRIM))
    return {
        "axis": str(rj.GetAxisAttr().Get()),
        "body0": [str(t) for t in rj.GetBody0Rel().GetTargets()],
        "body1": [str(t) for t in rj.GetBody1Rel().GetTargets()],
        "T0": _T_from(rj.GetLocalPos0Attr().Get(), rj.GetLocalRot0Attr().Get()),
        "T1": _T_from(rj.GetLocalPos1Attr().Get(), rj.GetLocalRot1Attr().Get()),
        "lower_deg": float(rj.GetLowerLimitAttr().Get()),
        "upper_deg": float(rj.GetUpperLimitAttr().Get()),
    }


def gripper_T_l5(joint: dict, q5_deg: float) -> np.ndarray:
    """gripper_link pose in link5 frame at q5 (deg). Same sign basis as the
    23rd audit: +theta right-handed about joint +Z, authored=0=closed."""
    return joint["T0"] @ _Rz(math.radians(q5_deg)) @ np.linalg.inv(joint["T1"])


def parse_urdf() -> dict:
    root = ET.parse(str(URDF_PATH)).getroot()
    out = {"visual": {}, "joint_origin": None, "tcp_origin": None, "dead_asset_referenced": None}
    for link in root.findall("link"):
        name = link.get("name")
        if name not in VIS_PINS:
            continue
        vis = link.find("visual")
        origin = vis.find("origin")
        mesh = vis.find("geometry/mesh")
        out["visual"][name] = {
            "filename": mesh.get("filename"),
            "scale": mesh.get("scale"),
            "origin_xyz": origin.get("xyz"),
            "origin_rpy": origin.get("rpy"),
        }
    for joint in root.findall("joint"):
        if joint.get("name") == "link5_to_gripper_link":
            o = joint.find("origin")
            out["joint_origin"] = {
                "xyz": [float(v) for v in o.get("xyz").split()],
                "rpy": [float(v) for v in o.get("rpy").split()],
                "axis": joint.find("axis").get("xyz"),
                "lower": float(joint.find("limit").get("lower")),
                "upper": float(joint.find("limit").get("upper")),
            }
        if joint.get("name") == "link5_to_hand_tcp":
            out["tcp_origin"] = [float(v) for v in joint.find("origin").get("xyz").split()]
    out["dead_asset_referenced"] = DEAD_ASSET in URDF_PATH.read_text()
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, default="t3r_gate0_vismesh")
    args = ap.parse_args()
    if not args.tag.startswith("t3r_gate0"):
        raise ValueError("tag must stay inside the t3r_gate0* namespace (frozen t3_*/t3r_* protection)")

    results_path = OUT_DIR / f"{args.tag}_results.json"
    bands_csv_path = OUT_DIR / f"{args.tag}_bands.csv"
    rrd_path = OUT_DIR / f"{args.tag}_timeline.rrd"
    rbl_path = OUT_DIR / f"{args.tag}_timeline.rbl"
    png_path = OUT_DIR / f"{args.tag}_inspection.png"
    validation_path = OUT_DIR / f"{args.tag}_rerun_validation.json"
    existing = [p.name for p in (results_path, bands_csv_path, rrd_path, rbl_path, png_path, validation_path) if p.exists()]
    if existing:
        print(f"[{LOG}] G0B_T3R_GATE0_VERDICT=GATE0_ABORT preexisting_artifacts={existing}", flush=True)
        return 3

    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        print(f"[{LOG}] G0B_T3R_GATE0_VERDICT=GATE0_ABORT rerun_version={rr.__version__}!={RERUN_VERSION}", flush=True)
        return 3

    print(f"[{LOG}] env python={sys.version.split()[0]} numpy={np.__version__} rerun_sdk={rr.__version__} "
          f"sample_spacing_mm={SAMPLE_SPACING_M*1000:.2f} l_min_mm={L_MIN_MM} indet_half_mm={INDET_HALF_MM} "
          f"finger_window_r_mm={FINGER_WINDOW_R_MM}", flush=True)
    print(f"[{LOG}] readonly_declaration=stl_read+UsdStageOpen_only no_save_no_export no_kit "
          f"(authoring-source audit; runs BEFORE any authoring; D415-3 untouched)", flush=True)

    # ---- GV1: source identity pins ------------------------------------------
    shas = {}
    gv1 = True
    for body, (fname, pin) in VIS_PINS.items():
        p = MESH_DIR / fname
        shas[fname] = _sha256_file(p)
        ok = shas[fname] == pin
        gv1 = gv1 and ok
        print(f"[{LOG}] GV1_sha body={body} file={fname} sha={shas[fname][:16]} pin_match={ok}", flush=True)
    dead_path = MESH_DIR / DEAD_ASSET
    shas[DEAD_ASSET] = _sha256_file(dead_path) if dead_path.exists() else None
    usd_root_sha = _sha256_file(ATTEMPT3_USD)
    usd_phys_sha = _sha256_file(ATTEMPT3_PHYSICS_LAYER)
    urdf_sha = _sha256_file(URDF_PATH)  # recorded for the evidence chain (not a pin — URDF content gated at GV2/GV4)
    gv1 = gv1 and usd_root_sha == ATTEMPT3_ROOT_SHA256 and usd_phys_sha == ATTEMPT3_PHYSICS_SHA256
    print(f"[{LOG}] GV1_sha usd_root={usd_root_sha[:16]} usd_physics={usd_phys_sha[:16]} "
          f"urdf_sha={urdf_sha[:16]} (recorded) "
          f"dead_asset_sha={None if shas[DEAD_ASSET] is None else shas[DEAD_ASSET][:16]} (recorded, NOT used) pass={gv1}", flush=True)
    if not gv1:
        print(f"[{LOG}] G0B_T3R_GATE0_VERDICT=GATE0_ABORT sha_mismatch", flush=True)
        return 3

    # ---- GV2: URDF visual wiring --------------------------------------------
    urdf = parse_urdf()
    gv2 = True
    for body, (fname, _) in VIS_PINS.items():
        v = urdf["visual"].get(body)
        ok = (
            v is not None
            and v["filename"] == f"meshes/{fname}"
            and v["scale"] == "0.001 0.001 0.001"
            and [float(x) for x in v["origin_xyz"].split()] == [0.0, 0.0, 0.0]
            and [float(x) for x in v["origin_rpy"].split()] == [0.0, 0.0, 0.0]
        )
        gv2 = gv2 and ok
        print(f"[{LOG}] GV2_urdf body={body} visual={None if v is None else v['filename']} "
              f"origin_identity_and_scale_ok={ok}", flush=True)
    tcp_ok = urdf["tcp_origin"] is not None and np.allclose(urdf["tcp_origin"], TCP_LOCAL, atol=1e-9)
    dead_ok = not urdf["dead_asset_referenced"]
    gv2 = gv2 and tcp_ok and dead_ok
    print(f"[{LOG}] GV2_urdf tcp_origin={urdf['tcp_origin']} tcp_match={tcp_ok} "
          f"dead_asset_referenced={urdf['dead_asset_referenced']} (must be False) pass={gv2}", flush=True)
    if not gv2:
        print(f"[{LOG}] G0B_T3R_GATE0_VERDICT=GATE0_ABORT urdf_wiring_mismatch", flush=True)
        return 3

    # ---- load + sample meshes -----------------------------------------------
    meshes = {}
    gv3 = True
    for body, (fname, _) in VIS_PINS.items():
        tris = load_binary_stl(MESH_DIR / fname) * STL_SCALE
        pts = sample_triangles(tris, SAMPLE_SPACING_M)
        bbox = pts.max(axis=0) - pts.min(axis=0)
        ok = bool(np.all(bbox > 0.005) and np.all(bbox < 0.4))
        gv3 = gv3 and ok
        meshes[body] = {"tris": tris, "pts": pts, "bbox_m": bbox}
        print(f"[{LOG}] GV3_units body={body} n_tri={tris.shape[0]} n_samples={pts.shape[0]} "
              f"bbox_mm=({bbox[0]*1000:.1f},{bbox[1]*1000:.1f},{bbox[2]*1000:.1f}) plausible={ok}", flush=True)
    if not gv3:
        print(f"[{LOG}] G0B_T3R_GATE0_VERDICT=GATE0_ABORT units_sanity_failed", flush=True)
        return 3

    # ---- GV4: URDF<->USD joint-frame witness --------------------------------
    joint = extract_usd_joint()
    T_o = Tmat(urdf["joint_origin"]["xyz"], urdf["joint_origin"]["rpy"])
    witness = 0.0
    for th in (0.0, 45.0, Q5_OPEN_DEG):
        witness = max(witness, float(np.abs(T_o @ _Rz(math.radians(th)) - gripper_T_l5(joint, th)).max()))
    urdf_axis = [float(v) for v in urdf["joint_origin"]["axis"].split()]
    urdf_axis_ok = bool(np.allclose(urdf_axis, [0.0, 0.0, 1.0], atol=1e-12))  # witness assumes +z (T_o @ Rz)
    gv4 = (
        joint["axis"] == "Z"
        and urdf_axis_ok
        and joint["body0"] == [BODY_PRIMS["link5"]]
        and joint["body1"] == [BODY_PRIMS["gripper_link"]]
        and witness < JOINT_WITNESS_TOL
        and abs(urdf["joint_origin"]["lower"]) < 1e-9
        and abs(math.degrees(urdf["joint_origin"]["upper"]) - 90.0) < 0.1
        and abs(joint["lower_deg"]) < 0.1
        and 89.0 < joint["upper_deg"] < 91.0
    )
    print(f"[{LOG}] GV4_joint_witness pass={gv4} urdf_vs_usd_max_abs_diff={witness:.3e} axis={joint['axis']} "
          f"urdf_axis={urdf_axis} urdf_axis_gate={urdf_axis_ok} "
          f"usd_limits=[{joint['lower_deg']:.3f},{joint['upper_deg']:.3f}]deg "
          f"placement_frame=USD (audit3-anchored); urdf agreement gated", flush=True)
    if not gv4:
        print(f"[{LOG}] G0B_T3R_GATE0_VERDICT=GATE0_ABORT joint_frame_mismatch", flush=True)
        return 3

    # ---- fixed jaw (link5): pose-free ---------------------------------------
    d5, r5 = depth_r_mm(meshes["link5"]["pts"])
    fixed = body_metrics(d5, r5)
    if not math.isfinite(fixed["l_vis_mm"]):  # window anomaly is ABORT, not a scientific FAIL (write-free)
        print(f"[{LOG}] G0B_T3R_GATE0_VERDICT=GATE0_ABORT empty_finger_window body=link5", flush=True)
        return 3
    fixed_peak_pt = peak_point(d5, r5, meshes["link5"]["pts"], meshes["link5"]["pts"])
    fixed_secondary = {"l_vis_wall_mm": _finite_or_none(fixed["max_depth_in_wall_annulus_mm"])}
    fixed_state = tri_state(fixed["l_vis_mm"])
    print(f"[{LOG}] fixed_jaw link5 l_vis_mm={fixed['l_vis_mm']:.3f} r_at_l_vis_mm={fixed['r_at_l_vis_mm']:.2f} "
          f"fp_max_depth_mm={fixed['max_depth_in_footprint_mm']} wall_max_depth_mm={fixed['max_depth_in_wall_annulus_mm']} "
          f"rim_min_r_mm={fixed['min_r_mm_in_rim_band']} state={fixed_state}", flush=True)
    print(f"[{LOG}] fixed_peak_point r_mm={fixed_peak_pt['r_mm']:.2f} azimuth_deg={fixed_peak_pt['azimuth_deg']:.1f} "
          f"body_frame_m={[round(v, 6) for v in fixed_peak_pt['body_frame_m']]} "
          f"l_vis_wall_mm={fixed_secondary['l_vis_wall_mm']}", flush=True)

    # ---- moving jaw (gripper_link): q5 sweep --------------------------------
    grip_pts_g = meshes["gripper_link"]["pts"]
    grid = [Q5_OPEN_DEG] + [round(v, 4) for v in np.arange(88.0, -0.001, -SWEEP_STEP_DEG)]
    sweep_l, sweep_wall = [], []
    for q5 in grid:
        d, r = depth_r_mm(transform_pts(gripper_T_l5(joint, q5), grip_pts_g))
        sweep_l.append(l_vis_mm(d, r))
        wall_m = (r >= WALL_ANNULUS_MM[0]) & (r <= WALL_ANNULUS_MM[1])
        sweep_wall.append(float(d[wall_m].max()) if wall_m.any() else None)
    sweep_l = np.array(sweep_l)
    k = int(np.argmax(sweep_l))
    lo = max(0.0, grid[k] - REFINE_HALF_DEG)
    hi = min(Q5_OPEN_DEG, grid[k] + REFINE_HALF_DEG)
    refine_grid = np.arange(lo, hi + REFINE_STEP_DEG / 2, REFINE_STEP_DEG)
    refine_l = []
    for q5 in refine_grid:
        d, r = depth_r_mm(transform_pts(gripper_T_l5(joint, float(q5)), grip_pts_g))
        refine_l.append(l_vis_mm(d, r))
    refine_l = np.array(refine_l)
    kr = int(np.argmax(refine_l))
    moving_peak = {"q5_deg": float(refine_grid[kr]), "l_vis_mm": float(refine_l[kr])}
    if not math.isfinite(moving_peak["l_vis_mm"]):  # window anomaly is ABORT, not a scientific FAIL (write-free)
        print(f"[{LOG}] G0B_T3R_GATE0_VERDICT=GATE0_ABORT empty_finger_window body=gripper_link", flush=True)
        return 3
    pts_pk = transform_pts(gripper_T_l5(joint, moving_peak["q5_deg"]), grip_pts_g)
    d_pk, r_pk = depth_r_mm(pts_pk)
    peak_metrics = body_metrics(d_pk, r_pk)  # discriminator metrics at the verdict-driving pose (panel M2)
    moving_peak_pt = peak_point(d_pk, r_pk, pts_pk, grip_pts_g)
    grid_arr = np.asarray(grid)
    moving_secondary = {
        "l_vis_wall_mm": max((v for v in sweep_wall if v is not None), default=None),
        "l_vis_grasp_range_mm": _finite_or_none(float(np.max(sweep_l[grid_arr <= 45.0 + 1e-9]))),
        "note": ("report-only, gate unchanged (panel M3): wall = max depth over 12.5<=r<=20 mm across the coarse "
                 "sweep; grasp_range = window l_vis max over grasp-relevant q5<=45 deg on the coarse 0.25 deg grid"),
    }
    moving_state = tri_state(moving_peak["l_vis_mm"])
    print(f"[{LOG}] moving_jaw gripper_link coarse_peak q5={grid[k]:.2f}deg l_vis={sweep_l[k]:.3f}mm | "
          f"refined_peak q5={moving_peak['q5_deg']:.2f}deg l_vis={moving_peak['l_vis_mm']:.3f}mm state={moving_state}", flush=True)
    print(f"[{LOG}] moving_peak_metrics q5={moving_peak['q5_deg']:.2f}deg l_vis_mm={peak_metrics['l_vis_mm']:.3f} "
          f"r_at_l_vis_mm={peak_metrics['r_at_l_vis_mm']:.2f} azimuth_deg={moving_peak_pt['azimuth_deg']:.1f} "
          f"fp_max_depth_mm={peak_metrics['max_depth_in_footprint_mm']} "
          f"wall_max_depth_mm={peak_metrics['max_depth_in_wall_annulus_mm']} "
          f"rim_min_r_mm={peak_metrics['min_r_mm_in_rim_band']}", flush=True)
    print(f"[{LOG}] moving_secondary l_vis_wall_mm={moving_secondary['l_vis_wall_mm']} "
          f"l_vis_grasp_range_mm={moving_secondary['l_vis_grasp_range_mm']}", flush=True)

    anchors = {}
    for q5 in ANCHOR_Q5_DEG:
        d, r = depth_r_mm(transform_pts(gripper_T_l5(joint, q5), grip_pts_g))
        anchors[f"{q5:.4f}"] = body_metrics(d, r)
        a = anchors[f"{q5:.4f}"]
        print(f"[{LOG}] moving_anchor q5={q5:7.3f}deg l_vis_mm={a['l_vis_mm']:.3f} "
              f"fp_max_depth_mm={a['max_depth_in_footprint_mm']} wall_max_depth_mm={a['max_depth_in_wall_annulus_mm']} "
              f"rim_min_r_mm={a['min_r_mm_in_rim_band']}", flush=True)

    # ---- bands CSV ----------------------------------------------------------
    rows = band_rows("link5", "static", d5, r5)
    for q5 in ANCHOR_Q5_DEG:
        d, r = depth_r_mm(transform_pts(gripper_T_l5(joint, q5), grip_pts_g))
        rows += band_rows("gripper_link", f"{q5:.4f}", d, r)
    rows += band_rows("gripper_link", f"peak_{moving_peak['q5_deg']:.4f}", d_pk, r_pk)  # panel M2
    with bands_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["body", "q5_deg", "band_lo_mm", "band_hi_mm", "n_pts_in_window", "min_r_mm", "max_r_mm"])
        w.writerows(rows)

    # ---- collision-layer cross-reference (report-only; numeric fields only,
    #      D425-④ string-boolean trap avoidance) ------------------------------
    audit3 = json.loads(AUDIT3_RESULTS.read_text())
    coll_assembly_max_depth = max(float(p["assembly_max_depth_mm"]) for p in audit3["profile"])
    coll_rim_none = sum(1 for p in audit3["profile"] if p["min_r_mm_in_rim_band"] is None)
    cross = {
        "audit3_file": str(AUDIT3_RESULTS),
        "collision_assembly_max_depth_mm": coll_assembly_max_depth,
        "collision_rim_band_empty_angles": f"{coll_rim_none}/{len(audit3['profile'])}",
        "visual_minus_collision_max_depth_mm": {
            "fixed_link5": float(fixed["l_vis_mm"] - coll_assembly_max_depth),
            "moving_peak": float(moving_peak["l_vis_mm"] - coll_assembly_max_depth),
        },
        "note": (f"collision assembly_max_depth is plug-dominated per D425 (part_029/030); rim band empty at "
                 f"{coll_rim_none}/{len(audit3['profile'])} angles in the frozen audit3 JSON"),
    }
    print(f"[{LOG}] cross_ref collision_assembly_max_depth_mm={coll_assembly_max_depth:.3f} "
          f"collision_rim_band_empty={cross['collision_rim_band_empty_angles']} "
          f"visual_minus_collision fixed={cross['visual_minus_collision_max_depth_mm']['fixed_link5']:+.3f}mm "
          f"moving={cross['visual_minus_collision_max_depth_mm']['moving_peak']:+.3f}mm", flush=True)

    # ---- verdict ------------------------------------------------------------
    states = {"fixed_link5": fixed_state, "moving_gripper_link": moving_state}
    if any(s == "FAIL" for s in states.values()):
        verdict = "GATE0_SOURCE_ABSENT"
    elif all(s == "PASS" for s in states.values()):
        verdict = "GATE0_SOURCE_PRESENT"
    else:
        verdict = "GATE0_INDETERMINATE"

    bbox_diag_mm = float(np.linalg.norm(meshes["gripper_link"]["bbox_m"])) * 1000.0
    results_doc = {
        "artifact": "G0B_T3R_GATE0_VISUAL_MESH_DISTAL_DEPTH_AUDIT_V1",
        "case": "g0b_d420",
        "tag": args.tag,
        "verdict": verdict,
        "layer": ("read-only visual-mesh authoring-source audit (not re-decomposition; not a physics, "
                  "real-gripper, or visual-matches-real claim — that is T4 layer)"),
        "gate_contract": {
            "l_min_mm": float(L_MIN_MM),
            "indet_half_mm": float(INDET_HALF_MM),
            "finger_window_r_mm": float(FINGER_WINDOW_R_MM),
            "wall_annulus_mm": [float(v) for v in WALL_ANNULUS_MM],
            "derivation": ("x = L - m; m = 5.5 mm (attempt2 precedent); T1-real x in (0,12] mm => L in (5.5,17.5] mm "
                           "(24th §4-1). L_MIN anchors to the attempt2 margin precedent m=5.5, NOT the D426 G-e floor "
                           "4.5, because grip depths x <= 1.0 mm are not operationally usable; a measured L_vis in "
                           "[4.5,5.0) therefore still verdicts FAIL by preregistration."),
            "recommended_L_band_mm": [9.5, 13.5],
            "recommended_L_band_source": "24th doc §4-1 (report-only; a PASS below this band is authorable but sub-recommended)",
            "per_body_states": states,
        },
        "sources": {
            "urdf": str(URDF_PATH),
            "urdf_sha256": urdf_sha,
            "visual_pins": {b: {"file": f, "sha256": shas[f]} for b, (f, _) in VIS_PINS.items()},
            "dead_asset": {"file": DEAD_ASSET, "sha256": shas[DEAD_ASSET], "urdf_referenced": bool(urdf["dead_asset_referenced"]), "used": False},
            "stl_scale": STL_SCALE,
            "usd_joint_source": {"path": str(ATTEMPT3_USD), "root_sha256": usd_root_sha, "physics_sha256": usd_phys_sha},
        },
        "frames": {
            "tcp_local_m": [float(v) for v in TCP_LOCAL],
            "urdf_vs_usd_joint_max_abs_diff": float(witness),
            "joint_limits_deg": [float(joint["lower_deg"]), float(joint["upper_deg"])],
            "sign_convention": "q5_rad positive right-handed about joint frame +Z; authored=0=closed (USD limits); depth +below TCP",
            "placement_frame": "USD joint (audit3-anchored); URDF agreement gated at GV4",
        },
        "gates": {
            "GV1_sha_pins": bool(gv1), "GV2_urdf_wiring": bool(gv2), "GV3_units": bool(gv3),
            "GV4_joint_witness": bool(gv4), "joint_witness_max_abs_diff": float(witness),
            "sample_spacing_mm": float(SAMPLE_SPACING_M * 1000.0),
            "sweep_step_deg": float(SWEEP_STEP_DEG), "refine_step_deg": float(REFINE_STEP_DEG),
            "sweep_depth_error_bound_mm": float(bbox_diag_mm * math.sin(math.radians(SWEEP_STEP_DEG / 2.0))),
            "refine_depth_error_bound_mm": float(bbox_diag_mm * math.sin(math.radians(REFINE_STEP_DEG / 2.0))),
            "depth_error_bound_derivation": ("one-sided understate-only: point radius about the joint axis <= gripper "
                                            f"bbox diagonal {bbox_diag_mm:.1f} mm x sin(step/2); the sweep bound applies "
                                            "only to a secondary local maximum >1.0 deg from the refined argmax "
                                            "(single-peak refine retained as authored; boundary-adjacent tri-states "
                                            "must be interpreted with this bound)"),
        },
        "fixed_link5": {**_san_metrics(fixed), "state": fixed_state,
                        "peak_point": fixed_peak_pt, "secondary": fixed_secondary},
        "moving_gripper_link": {
            "peak": moving_peak, "state": moving_state,
            "peak_metrics": _san_metrics(peak_metrics),
            "peak_point": moving_peak_pt,
            "secondary": moving_secondary,
            "coarse_grid_deg": [float(g) for g in grid],
            "coarse_l_vis_mm": [_finite_or_none(float(v)) for v in sweep_l],
            "coarse_l_vis_wall_mm": sweep_wall,
            "anchors": {kk: _san_metrics(vv) for kk, vv in anchors.items()},
        },
        "interpretation_rules": {
            "gate_unchanged": ("verdict gate = l_vis (max depth in the r<=30 mm window; max over the q5 sweep for the "
                               "moving jaw) vs L_MIN — all secondary/peak metrics are report-only and never enter the verdict"),
            "null_convention": "null in depth/l_vis fields = no sampled point inside the referenced window at that pose",
            "manual_review_rule": ("a PRESENT verdict whose r_at_l_vis_mm < 12.5 mm (near-axis structure, plug-analog) "
                                   "or whose peak is reached only at q5 > 60 deg requires explicit manual localization "
                                   "review during the D341 inspection before Arm-F/D authoring proceeds"),
        },
        "collision_cross_reference": cross,
        "mesh_stats": {b: {"n_tri": int(m["tris"].shape[0]), "n_samples": int(m["pts"].shape[0]),
                           "bbox_mm": [float(v * 1000) for v in m["bbox_m"]]} for b, m in meshes.items()},
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "rerun_sdk": RERUN_VERSION},
    }
    results_path.write_text(json.dumps(results_doc, indent=2) + "\n")
    print(f"[{LOG}] G0B_T3R_GATE0_VERDICT={verdict} authority_json={results_path.name} "
          f"(early print — final verdict line repeats after the observability layer)", flush=True)

    # ---- D341 Rerun artifact (save-only) ------------------------------------
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    app_id = f"roarm_g0b_{args.tag}"

    def view_pts(pts_l5: np.ndarray) -> np.ndarray:
        out = pts_l5 - TCP_LOCAL[None, :]
        return np.column_stack([out[:, 0], out[:, 1], -out[:, 2]])

    def ring(r_mm: float, depth_mm: float, n: int = 48) -> list[list[float]]:
        return [[r_mm / 1000.0 * math.cos(2 * math.pi * i / n), r_mm / 1000.0 * math.sin(2 * math.pi * i / n),
                 -depth_mm / 1000.0] for i in range(n + 1)]

    with rr.RecordingStream(app_id, recording_id=f"g0b_d420_{args.tag}", make_default=False, send_properties=True) as rec:
        rec.save(str(rrd_path), write_footer=True)
        rec.log("gate0/link5_vis", rr.Points3D(view_pts(meshes["link5"]["pts"][::VIEW_STRIDE]), colors=[150, 150, 160], radii=0.0004), static=True)
        rec.log("gate0/tcp", rr.Points3D([[0.0, 0.0, 0.0]], colors=[[40, 200, 80]], radii=0.002), static=True)
        rec.log("gate0/tool_axis", rr.LineStrips3D([[[0.0, 0.0, 0.06], [0.0, 0.0, -0.06]]], colors=[[40, 200, 80]], radii=0.0003), static=True)
        rec.log("gate0/cylinder_ref", rr.LineStrips3D(
            [ring(CYL_R_MM, HOVER_MARGIN_REF_MM), ring(CYL_R_MM, HOVER_MARGIN_REF_MM + CYL_H_MM)],
            colors=[[230, 120, 40]] * 2, radii=0.0004), static=True)
        rec.log("gate0/l_min_plane", rr.LineStrips3D(
            [ring(WALL_ANNULUS_MM[0], L_MIN_MM), ring(WALL_ANNULUS_MM[1], L_MIN_MM)],
            colors=[[235, 60, 60]] * 2, radii=0.0004), static=True)
        anchor_idx = {q5: min(range(len(grid)), key=lambda i: abs(grid[i] - q5)) for q5 in ANCHOR_Q5_DEG}
        anchor_ids = set(anchor_idx.values()) | {k}  # panel M1: coarse-peak pose = decision subject, must be in the RRD
        for i, q5 in enumerate(grid):
            rec.reset_time()
            rec.set_time("sweep_index", sequence=i)
            rec.log("plots/q5_deg", rr.Scalars(float(q5)))
            rec.log("plots/l_vis_moving_mm", rr.Scalars(float(sweep_l[i])))
            rec.log("plots/l_vis_fixed_mm", rr.Scalars(float(fixed["l_vis_mm"])))
            rec.log("plots/l_min_mm", rr.Scalars(float(L_MIN_MM)))
            if i in anchor_ids:
                g_now = transform_pts(gripper_T_l5(joint, q5), grip_pts_g)
                rec.log("gate0/gripper_vis", rr.Points3D(view_pts(g_now[::VIEW_STRIDE]), colors=[70, 130, 230], radii=0.0004))
        rec.reset_time()
        rec.set_time("sweep_index", sequence=0)
        for name, ok in (("GV1_sha_pins", gv1), ("GV2_urdf_wiring", gv2), ("GV3_units", gv3), ("GV4_joint_witness", gv4),
                         ("G0_fixed_" + fixed_state, fixed_state == "PASS"), ("G0_moving_" + moving_state, moving_state == "PASS")):
            rec.log("events/gates", rr.TextLog(f"{name}", level=rr.TextLogLevel.INFO if ok else rr.TextLogLevel.ERROR))
        summary_md = (
            f"# G0b T3R Gate-0 visual-mesh distal depth audit (case g0b_d420)\n\n"
            f"- verdict: **{verdict}** (authoring-source layer — not physics, not real-gripper, not visual-matches-real)\n"
            f"- fixed jaw link5.stl ({shas['link5.stl'][:12]}): L_vis={fixed['l_vis_mm']:.3f} mm -> {fixed_state}\n"
            f"- moving jaw gripper_link.stl ({shas['gripper_link.stl'][:12]}): L_vis peak={moving_peak['l_vis_mm']:.3f} mm "
            f"@ q5={moving_peak['q5_deg']:.2f} deg -> {moving_state}\n"
            f"- gate: L_MIN={L_MIN_MM} mm (from x=L-m, 24th §4-1), indet half-width {INDET_HALF_MM} mm, "
            f"finger window r<={FINGER_WINDOW_R_MM:.0f} mm\n"
            f"- collision cross-ref: assembly max depth {coll_assembly_max_depth:.3f} mm (plug-dominated), "
            f"rim band empty {cross['collision_rim_band_empty_angles']} angles; "
            f"visual-minus-collision: fixed {cross['visual_minus_collision_max_depth_mm']['fixed_link5']:+.2f} mm, "
            f"moving {cross['visual_minus_collision_max_depth_mm']['moving_peak']:+.2f} mm\n"
            f"- peak localization (panel M2/M3, report-only): fixed r_at_l_vis={fixed['r_at_l_vis_mm']:.2f} mm "
            f"az={fixed_peak_pt['azimuth_deg']:.0f} deg; moving(peak) r_at_l_vis={peak_metrics['r_at_l_vis_mm']:.2f} mm "
            f"az={moving_peak_pt['azimuth_deg']:.0f} deg; l_vis_wall fixed={fixed_secondary['l_vis_wall_mm']} / "
            f"moving={moving_secondary['l_vis_wall_mm']} mm, moving grasp-range(q5<=45)={moving_secondary['l_vis_grasp_range_mm']} mm "
            f"(r<12.5 mm = near-axis non-wall structure -> manual review rule)\n"
            f"- scene: grey=link5 visual, blue=gripper visual (anchor + coarse-peak poses), green=TCP+axis, "
            f"orange=cylinder D29 at hover margin {HOVER_MARGIN_REF_MM} mm (reference only), "
            f"red rings=L_MIN plane at wall annulus radii {WALL_ANNULUS_MM[0]:.1f}/{WALL_ANNULUS_MM[1]:.1f} mm\n"
            f"- sampling: raw-triangle surface {SAMPLE_SPACING_M*1000:.2f} mm pitch (no hulling — visual mesh is non-convex); "
            f"view decimated 1/{VIEW_STRIDE} (inspection-only); authority = stdout + results JSON\n"
        )
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN), static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run", name="1 | gate-0 verdict + contract"),
                    rrb.Spatial3DView(origin="/", contents=["/gate0/**"], name="2 | tool-frame visual meshes vs L_MIN plane"),
                    rrb.TextLogView(origin="/events/gates", contents="/events/gates/**", name="3 | gates"),
                    column_shares=[0.28, 0.47, 0.25],
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/plots", contents=["/plots/l_vis_moving_mm/**", "/plots/l_vis_fixed_mm/**", "/plots/l_min_mm/**"], name="4 | L_vis vs L_MIN over close sweep"),
                    rrb.TimeSeriesView(origin="/plots", contents=["/plots/q5_deg/**"], name="5 | q5 sweep angle"),
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
        "metadata/run", "gate0/link5_vis", "gate0/gripper_vis", "gate0/tcp", "gate0/tool_axis",
        "gate0/cylinder_ref", "gate0/l_min_plane",
        "plots/q5_deg", "plots/l_vis_moving_mm", "plots/l_vis_fixed_mm", "plots/l_min_mm", "events/gates",
    ]
    components = {
        "metadata/run": ["TextDocument:text"],
        "gate0/link5_vis": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "gate0/gripper_vis": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "gate0/tcp": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "gate0/tool_axis": ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"],
        "gate0/cylinder_ref": ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"],
        "gate0/l_min_plane": ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"],
        "plots/q5_deg": ["Scalars:scalars"],
        "plots/l_vis_moving_mm": ["Scalars:scalars"],
        "plots/l_vis_fixed_mm": ["Scalars:scalars"],
        "plots/l_min_mm": ["Scalars:scalars"],
        "events/gates": ["TextLog:text", "TextLog:level"],
    }
    validation = validate_rerun_artifact(
        rrd_path,
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time", "sweep_index"],
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
          f"bands_csv={bands_csv_path.name}", flush=True)
    print(f"[{LOG}] G0B_T3R_GATE0_VERDICT={verdict}", flush=True)
    return 0 if verdict == "GATE0_SOURCE_PRESENT" and validation.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
