#!/usr/bin/env python3
"""
g0b_t3r_n8_tilt_admission_readonly_audit.py
    (READ-ONLY on every asset; writes only NEW t3r_n8_tilt_* artifacts)

41st session - the TILT test the user authorised.
Permitted by START_HERE.md:171-172, which froze t3r_n7_assembly_* but carved out
exactly this derivation: "§5 기움 파라미터 검정은 ... 신규 태그 파생만 허용".

WHY THIS RUN EXISTS
  40th proved the URDF assembly reproduces the sim physics runs to 0.058 mm
  (deepest reachable top face = TCP+4.4576 mm vs attempt1's measured stop at
  top+4.4 mm) and that the achievable bite never crosses 0 over the whole q5
  sweep (max -2.1137 mm).  It could NOT reproduce the one remaining real
  observation: T1's photo-verified rim pinch of 0..12 mm below the cylinder's
  top face.  40th §5 named the suspect and left it untested:

      T1's tool-axis tilt was 0..35 deg, varying frame to frame (D420 Impl.(2),
      "10~20 deg single value does not hold"), whereas T2 and T3 PRE-REGISTERED
      a fully vertical tool axis (D421 descend tilt 0.1989 deg).

  So the model has been asked to reproduce a pose the real arm never held.
  This run puts the tilt back in and measures what it does.  Nothing else
  changes: same two visual meshes, same sha pins, same sampler, same numeric
  path as D427 and 40th.

CLAIM UNDER TEST (pre-registered, falsifiable both ways)
  "Within T1's measured tool-axis tilt range (0 < theta <= 35 deg) there exists
   an approach direction and a jaw opening for which the assembled gripper
   admits the D29 cylinder to a rim-pinch bite > 0 mm below its top face."

  PASS -> the fully-vertical constraint (not the asset) is what makes the task
          impossible; the T1 contradiction is explained without authoring any
          geometry, and "allow an approach tilt" becomes a quantified request
          to put to the professor (D419 is his instruction - HARD RULE #18).
  FAIL -> tilt is NOT the explanation.  The T1 contradiction survives with the
          asset layer already exhausted (D427, D429), which is a strictly
          worse position and must be reported as such, not softened.

GEOMETRY (no physics, no Isaac, no robot, pure rigid-body non-penetration)
  Tool axis = link5 +z ; TCP at link5 z = 115.428 mm (roarm_m3.urdf:234-239).
  The cylinder STANDS UPRIGHT on the table (erect, HARD RULE #18); it is the
  APPROACH that tilts.  In the tool frame that is one and the same thing: the
  cylinder axis direction becomes
      chat = (sin th cos ph, sin th sin ph, cos th)
  with th = tool-axis tilt and ph = the azimuth of that tilt in the tool frame.
  ph must be swept because the gripper is NOT rotationally symmetric (the
  blocking feature sits at r = 10.1244 mm, azimuth 172 deg - D427).

  The top-face CENTRE stays on the tool axis (D419 "수직 상부 상면 중심" keeps its
  centre-of-top-face meaning; only the axis tilts), at descent delta measured
  along the tool axis:  c(delta) = (0, 0, TCP_Z + delta).
  Object = { p : rho(p) <= R  and  u(p) >= 0 } where, with w = p - c0,
      u    = w.chat - delta*cos th                        (axial, + = into body)
      rho^2 = rho0^2 + 2*delta*b + delta^2*sin^2 th,
      rho0^2 = |w|^2 - u0^2,   b = u0*cos th - w_z,   u0 = w.chat
  Semi-infinite body (u >= 0, no lower cap) - identical to 40th's convention, so
  th = 0 reproduces 40th exactly.  H = 50 mm is reported as a sensitivity only.

  Per point the forbidden delta set is a SINGLE interval [d_lo, d_hi]:
      rho^2 <= R^2  ->  delta in [d_minus, d_plus]   (a = sin^2 th > 0)
                        or all delta                 (a = 0, i.e. th = 0)
      u >= 0        ->  delta <= u0 / cos th
  so the deepest admissible descent is the closed form
      delta_min = max{ d_hi(p) : p forbidden for some delta }
  and above it no point is inside the object.  At th = 0 this collapses to
      delta_min = max{ z_i - TCP_Z : r_i <= R }
  which is 40th's footprint blocker verbatim -> gate N8e.

  Bite is measured ON THE OBJECT, along the cylinder axis, exactly as T1's photo
  measured it:
      bite = max{ u_i : rho_i in [R, R + 6] }   evaluated at delta = delta_min
  Positive = jaw material lies alongside the cylinder wall below its top face.

PRE-REGISTERED GATES
  N8a  sha pins of link5.stl and gripper_link.stl                (40th verbatim)
  N8b  D427 l_vis reproduced: 4.457620117187505, n_pts 2266503    (40th verbatim)
  N8c  joint5 rotation matrix within the URDF's own rpy truncation (40th verbatim)
  N8d  moving-jaw z range matches the 37th report                 (40th verbatim)
  N8e  NEW - at th = 0 the tilted formulation must reproduce 40th's per-q5
       combined depth_top_min and bite for all 34 q5 values, |delta| < 1e-6 mm.
       Read from t3r_n7_assembly_results.json (READ-ONLY).  This is the gate that
       proves the new mathematics IS the old mathematics when the tilt is removed.
  N8f  NEW - at th = 0 the result must be invariant under ph (ph is meaningless
       when there is no tilt): spread over all 24 ph values < 1e-9 mm.
  N8g  NEW - brute-force check of the closed-form solve at the decisive config:
       no point inside at delta_min + 1e-9 m, at least one inside at
       delta_min - 1e-6 m.
  N8h  THE CLAIM: exists th in (0, 35] with bite > 0 for D29.
  N8i  THE CLAIM's strong form: exists th in (0, 35] with bite >= 12 mm (T1 upper).

NOT IN SCOPE (explicit)
  - No physics, no contact solver, no Isaac, no robot, no Gate-0 re-run or
    re-judgement (D427/D429 unchanged), no re-derivation of the 81.4065 hull,
    no 86.4 hypothesis, no touching of any t3r_n7_* / t3_* artifact.
  - gripper_left_link.stl excluded: not referenced by any URDF <mesh> and D429
    showed it is a re-tessellation of link5.stl geometry.
  - The TABLE is not modelled.  A tilted approach could put the lower jaw into
    the table; that constraint needs the arm configuration (IK) and is out of
    scope here.  T1 itself tilted 0..35 deg on a real table, so the table does
    not preclude the range - but this run does not prove reachability.
  - Static admission only.  bite > 0 is a NECESSARY condition for closing
    contact, never a sufficient one, and never a grasp-success prediction.
"""
import hashlib
import json
import math
import shutil
import struct
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # for roarm_rl.rerun_contract (D341 validator)
URDF = ROOT / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
MESH_DIR = ROOT / "local_assets/roarm_m3/urdf/meshes"
OUT_DIR = ROOT / "claudedocs/runtime_logs/grasp_track/g0b_d420"
TAG = "t3r_n8_tilt"
LOG = "g0b_t3r_n8"
N7_RESULTS = OUT_DIR / "t3r_n7_assembly_results.json"      # READ-ONLY reference

L5 = MESH_DIR / "link5.stl"
GJ = MESH_DIR / "gripper_link.stl"

TCP_Z_MM = 115.428
TCP_LOCAL = np.array([0.0, 0.0, 0.115428])
SAMPLE_SPACING_M = 0.0005
STL_SCALE = 0.001
CYL_R_MM = 14.5                 # D29 cylinder, HARD RULE #18 object
CYL_H_MM = 50.0
WALL_SPAN_MM = 6.0              # annulus [R, R+6] = "beside the cylinder wall" (40th convention)
T1_BITE_MM = (0.0, 12.0)        # real photo-verified rim pinch, D420 Impl.(2)
T1_TILT_DEG = (0.0, 35.0)       # real measured tool-axis tilt, D420 Impl.(2)
Q5_REAL_MAX_RAD = math.radians(88.31)
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"
VIEW_STRIDE = 40

# swept grids (declared, no silent caps - see the scoping note printed at run time)
THETA_DEG = np.arange(0.0, 35.0 + 1e-9, 1.0)          # 36 values, T1's measured range
PHI_DEG = np.arange(0.0, 360.0, 15.0)                 # 24 values, gripper is not axisymmetric
Q5_ANCHOR_DEG = (0.0, 16.877188052822312, 88.31)      # closed jaw / 40th argmax bite / real max opening

SHA = {"link5.stl": "1d63f374a78c1419b21eec63fa8efeef40d0d42ca89c5de3ceb0d86476d9c7eb",
       "gripper_link.stl": "7946a374e24a2f467a0581b4946e0ec41b1b86a92f070bc00aa9bced1bf65a56"}
D427_L_VIS_MM = 4.457620117187505
D427_N_PTS = 2266503
J5_R_EXPECT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
MOVING_Z_RANGE_37TH_MM = (41.2676, 119.1176)
N8E_TOL_MM = 1e-6
N8F_TOL_MM = 1e-9


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_binary_stl(path):
    """VERBATIM from the Gate-0 audit - raw triangle soup, (n,3,3), file units."""
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    body = np.frombuffer(data[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
    return body[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)


def sample_triangles(tris_m, spacing_m):
    """VERBATIM from the Gate-0 audit."""
    out_ = [tris_m.reshape(-1, 3)]
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
        out_.append((a + u * (b - a) + v * (c - a)).reshape(-1, 3))
    return np.vstack(out_)


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def parse_urdf(path):
    """Read the joint/visual facts from the URDF itself - never from memory."""
    root = ET.parse(path).getroot()
    out = {"meshes": {}, "joints": {}}
    for link in root.findall("link"):
        name = link.get("name")
        for tag in ("visual", "collision"):
            el = link.find(tag)
            if el is None:
                continue
            mesh = el.find("geometry/mesh")
            org = el.find("origin")
            out["meshes"][f"{name}/{tag}"] = {
                "filename": None if mesh is None else mesh.get("filename"),
                "scale": None if mesh is None else mesh.get("scale"),
                "origin_xyz": "0 0 0" if org is None else org.get("xyz", "0 0 0"),
                "origin_rpy": "0 0 0" if org is None else org.get("rpy", "0 0 0"),
            }
    for j in root.findall("joint"):
        org, ax, lim = j.find("origin"), j.find("axis"), j.find("limit")
        out["joints"][j.get("name")] = {
            "type": j.get("type"),
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": [float(v) for v in (org.get("xyz", "0 0 0")).split()] if org is not None else [0, 0, 0],
            "rpy": [float(v) for v in (org.get("rpy", "0 0 0")).split()] if org is not None else [0, 0, 0],
            "axis": [float(v) for v in ax.get("xyz").split()] if ax is not None else None,
            "limit": None if lim is None else [float(lim.get("lower")), float(lim.get("upper"))],
        }
    return out


# --------------------------------------------------------------------------- #
# the tilted-admission core.  Three functions, all closed form (no bisection).
# --------------------------------------------------------------------------- #
def axis_dir(theta_rad, phi_rad):
    st, ct = math.sin(theta_rad), math.cos(theta_rad)
    return np.array([st * math.cos(phi_rad), st * math.sin(phi_rad), ct])


def prep(P_m, chat, c0_m):
    """Per-point quantities that depend on (theta, phi) but NOT on the descent delta.

    Returns u0, rho0_sq, b with
        u(delta)    = u0 - delta*chat_z
        rho(delta)^2 = rho0_sq + 2*delta*b + delta^2*(1 - chat_z^2)
    """
    w = P_m - c0_m
    u0 = w @ chat
    w2 = np.einsum("ij,ij->i", w, w)
    rho0_sq = w2 - u0 * u0
    np.maximum(rho0_sq, 0.0, out=rho0_sq)          # kill -1e-19 round-off
    b = u0 * chat[2] - w[:, 2]
    return u0, rho0_sq, b


def deepest_delta(u0, rho0_sq, b, chat_z, R_m):
    """max over points of the upper end of that point's forbidden-delta interval.

    Above this delta no sampled gripper point lies inside the semi-infinite
    cylinder, so it IS the deepest admissible descent.  -inf if nothing blocks.
    """
    a = 1.0 - chat_z * chat_z                      # sin^2(theta)
    cq = rho0_sq - R_m * R_m
    d_u = u0 / chat_z                              # from u >= 0
    if a <= 0.0:                                   # theta == 0: rho is delta-independent
        blocked = cq <= 0.0
        return (float(d_u[blocked].max()), int(blocked.sum())) if blocked.any() else (-math.inf, 0)
    disc = b * b - a * cq
    ok = disc >= 0.0
    if not ok.any():
        return -math.inf, 0
    s = np.sqrt(disc[ok])
    bo, cqo, duo = b[ok], cq[ok], d_u[ok]
    # numerically stable max/min roots of a*d^2 + 2*b*d + cq = 0
    neg = bo <= 0.0
    num_hi = -bo + s
    num_lo = -bo - s
    with np.errstate(divide="ignore", invalid="ignore"):
        d_plus = np.where(neg, num_hi / a, np.where(num_lo != 0.0, cqo / num_lo, num_hi / a))
        d_minus = np.where(neg, np.where(num_hi != 0.0, cqo / num_hi, num_lo / a), num_lo / a)
    d_hi = np.minimum(d_plus, duo)
    live = np.isfinite(d_hi) & (d_minus <= duo)     # forbidden interval non-empty
    if not live.any():
        return -math.inf, 0
    return float(d_hi[live].max()), int(live.sum())


def bite_at(u0, rho0_sq, b, chat_z, R_m, span_m, delta):
    """max axial depth below the top face reached by material BESIDE the wall."""
    a = 1.0 - chat_z * chat_z
    rho_sq = rho0_sq + 2.0 * delta * b + delta * delta * a
    lo, hi = R_m * R_m, (R_m + span_m) ** 2
    beside = (rho_sq >= lo) & (rho_sq <= hi)
    if not beside.any():
        return -math.inf, 0
    u = u0[beside] - delta * chat_z
    return float(u.max()), int(beside.sum())


def inside_count(u0, rho0_sq, b, chat_z, R_m, delta):
    """Brute-force membership count - used only by gate N8g."""
    a = 1.0 - chat_z * chat_z
    rho_sq = rho0_sq + 2.0 * delta * b + delta * delta * a
    return int(((rho_sq <= R_m * R_m) & ((u0 - delta * chat_z) >= 0.0)).sum())


# =========================================================================== #
def main() -> int:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {k: OUT_DIR / f"{TAG}_{k}" for k in
             ("results.json", "timeline.rrd", "timeline.rbl", "rerun_validation.json",
              "inspection.png", "diagnostic.png", "script.py.txt")}
    existing = [p.name for p in paths.values() if p.exists()]
    if existing:
        print(f"[{LOG}] ABORT write_guard existing={existing}", flush=True)
        return 3
    if not N7_RESULTS.exists():
        print(f"[{LOG}] ABORT missing_n7_reference={N7_RESULTS}", flush=True)
        return 3

    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        print(f"[{LOG}] ABORT rerun_version={rr.__version__}!={RERUN_VERSION}", flush=True)
        return 3

    out = {"tool": TAG, "read_only_assets": True,
           "claim_under_test": ("within T1's measured tilt 0<theta<=35deg some approach direction and "
                               "jaw opening admits the D29 cylinder to a rim-pinch bite > 0mm"),
           "permitted_by": "START_HERE.md:171-172 (t3r_n7 frozen except this tilt derivation, new tag)",
           "env": {"numpy": np.__version__, "rerun_sdk": rr.__version__, "python": sys.version.split()[0]},
           "scoping_declared": {
               "theta_deg": [float(THETA_DEG[0]), float(THETA_DEG[-1]), float(THETA_DEG[1] - THETA_DEG[0])],
               "n_theta": int(len(THETA_DEG)), "n_phi": int(len(PHI_DEG)),
               "phi_step_deg": float(PHI_DEG[1] - PHI_DEG[0]),
               "q5_anchors_deg": list(Q5_ANCHOR_DEG),
               "note": ("the (theta,phi) grid is exhaustive at 3 q5 anchors (closed / 40th argmax bite / "
                        "real max opening); the FULL 34-point q5 sweep is then run at theta=0 (gate N8e) "
                        "and at the best (theta,phi). The full 3-way product is NOT swept - declared, "
                        "not silently capped."),
               "body_model": "semi-infinite (u>=0), identical to 40th; H=50mm reported as sensitivity only",
           }}

    # ---- 1. asset identity ------------------------------------------------
    out["sha256"] = {p.name: sha256(p) for p in (L5, GJ)}
    out["sha256_matches_record"] = {k: (out["sha256"][k] == v) for k, v in SHA.items()}
    print(f"[{LOG}] sha_ok={out['sha256_matches_record']}", flush=True)

    # ---- 2. URDF facts, read from the file --------------------------------
    u = parse_urdf(URDF)
    j5 = u["joints"]["link5_to_gripper_link"]
    R_j5 = rpy_matrix(*j5["rpy"])
    t_j5 = np.array(j5["xyz"])
    out["urdf"] = {
        "joint5": j5, "tcp_joint": u["joints"]["link5_to_hand_tcp"],
        "joint5_R": R_j5.tolist(),
        "joint5_R_max_abs_diff_vs_expected": float(np.abs(R_j5 - J5_R_EXPECT).max()),
        "link5_visual": u["meshes"]["link5/visual"],
        "gripper_visual": u["meshes"]["gripper_link/visual"],
        "gripper_left_link_referenced": any(
            "gripper_left_link" in (m["filename"] or "") for m in u["meshes"].values()),
    }

    # ---- 3. sample both visual meshes on the Gate-0 numeric path ----------
    t5 = load_binary_stl(L5) * STL_SCALE
    tg = load_binary_stl(GJ) * STL_SCALE
    S5 = sample_triangles(t5, SAMPLE_SPACING_M)
    SGj = sample_triangles(tg, SAMPLE_SPACING_M)
    out["assets"] = {"link5_tris": int(len(t5)), "link5_samples": int(len(S5)),
                     "gripper_link_tris": int(len(tg)), "gripper_link_samples": int(len(SGj))}
    print(f"[{LOG}] samples link5={len(S5)} gripper={len(SGj)}", flush=True)

    # ---- 4. gates N8a..N8d (40th verbatim) --------------------------------
    z5 = S5[:, 2] * 1000.0
    r5 = np.hypot(S5[:, 0], S5[:, 1]) * 1000.0
    l_vis = float((z5 - TCP_Z_MM)[r5 <= 30.0].max())
    g_a = all(out["sha256_matches_record"].values())
    g_b = (abs(l_vis - D427_L_VIS_MM) < 1e-9) and (len(S5) == D427_N_PTS)
    rpy_trunc_rad = float(np.abs(np.array(j5["rpy"]) - np.round(np.array(j5["rpy"]) /
                                                               (math.pi / 2)) * (math.pi / 2)).max())
    r_far_m = float(np.linalg.norm(SGj, axis=1).max())
    out["urdf"]["joint5_rpy_truncation_rad"] = rpy_trunc_rad
    out["urdf"]["joint5_rpy_truncation_max_vertex_shift_mm"] = rpy_trunc_rad * r_far_m * 1000.0
    g_c = out["urdf"]["joint5_R_max_abs_diff_vs_expected"] < 1e-5

    def to_link5(P_child_m, q5):
        c, s = math.cos(q5), math.sin(q5)
        Rq = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return P_child_m @ (R_j5 @ Rq).T + t_j5

    zq0 = to_link5(SGj, 0.0)[:, 2] * 1000.0
    g_d = (abs(zq0.min() - MOVING_Z_RANGE_37TH_MM[0]) < 0.02 and
           abs(zq0.max() - MOVING_Z_RANGE_37TH_MM[1]) < 0.02)
    print(f"[{LOG}] gates a={g_a} b={g_b}(l_vis={l_vis:.12f}) c={g_c} "
          f"d={g_d}(z=[{zq0.min():.4f},{zq0.max():.4f}])", flush=True)
    if not (g_a and g_b and g_c and g_d):
        print(f"[{LOG}] ABORT gate_failure_N8a_to_N8d", flush=True)
        return 3

    # ---- 5. the sweep -----------------------------------------------------
    c0 = np.array([0.0, 0.0, TCP_Z_MM / 1000.0])
    R_m = CYL_R_MM / 1000.0
    SPAN_m = WALL_SPAN_MM / 1000.0
    q5_full_rad = np.unique(np.concatenate([np.linspace(0.0, j5["limit"][1], 33), [Q5_REAL_MAX_RAD]]))
    anchors_rad = [math.radians(d) for d in Q5_ANCHOR_DEG]
    mov_cache = {d: to_link5(SGj, math.radians(d)) for d in Q5_ANCHOR_DEG}

    def evaluate(theta_deg, phi_deg, P_mov, prep_fixed=None):
        """One (theta, phi, q5) configuration -> depth / bite / provenance."""
        th, ph = math.radians(theta_deg), math.radians(phi_deg)
        chat = axis_dir(th, ph)
        pf = prep_fixed if prep_fixed is not None else prep(S5, chat, c0)
        pm = prep(P_mov, chat, c0)
        d_fix, n_fix = deepest_delta(*pf, chat[2], R_m)
        d_mov, n_mov = deepest_delta(*pm, chat[2], R_m)
        delta = max(d_fix, d_mov)
        if not math.isfinite(delta):
            return {"theta_deg": float(theta_deg), "phi_deg": float(phi_deg),
                    "depth_top_min_mm": None, "bite_mm": None, "blocker": "none",
                    "n_blocking_fixed": n_fix, "n_blocking_moving": n_mov}
        b_fix, nb_fix = bite_at(*pf, chat[2], R_m, SPAN_m, delta)
        b_mov, nb_mov = bite_at(*pm, chat[2], R_m, SPAN_m, delta)
        bite = max(b_fix, b_mov)
        if not math.isfinite(bite):        # no material beside the wall at all in either body
            return {"theta_deg": float(theta_deg), "phi_deg": float(phi_deg),
                    "depth_top_min_mm": delta * 1000.0, "bite_mm": None,
                    "blocker": "fixed" if d_fix >= d_mov else "moving",
                    "no_material_beside_wall": True, "delta_m": delta, "chat": chat.tolist()}
        return {"theta_deg": float(theta_deg), "phi_deg": float(phi_deg),
                "depth_top_min_mm": delta * 1000.0,
                "bite_mm": bite * 1000.0,
                "bite_fixed_mm": b_fix * 1000.0 if math.isfinite(b_fix) else None,
                "bite_moving_mm": b_mov * 1000.0 if math.isfinite(b_mov) else None,
                "blocker": "fixed" if d_fix >= d_mov else "moving",
                "n_blocking_fixed": n_fix, "n_blocking_moving": n_mov,
                "n_beside_fixed": nb_fix, "n_beside_moving": nb_mov,
                "delta_m": delta, "chat": chat.tolist()}

    def run_pairs(pairs, label):
        """(theta,phi) list x the q5 anchors. The fixed-jaw prep is shared per pair."""
        rows, t0 = [], time.time()
        for k, (th_d, ph_d) in enumerate(pairs):
            chat = axis_dir(math.radians(th_d), math.radians(ph_d))
            pf = prep(S5, chat, c0)                       # q5-independent, computed once
            for q_d in Q5_ANCHOR_DEG:
                row = evaluate(th_d, ph_d, mov_cache[q_d], prep_fixed=pf)
                row["q5_deg"] = float(q_d)
                rows.append(row)
            if k % 150 == 0 or k == len(pairs) - 1:
                print(f"[{LOG}] {label} {k + 1}/{len(pairs)} pairs t={time.time() - t0:.0f}s",
                      flush=True)
        return rows

    coarse_pairs = [(float(t), float(p)) for t in THETA_DEG for p in PHI_DEG]
    grid = run_pairs(coarse_pairs, "coarse_grid")
    out["grid_theta_phi_q5"] = grid

    # ---- 6. N8e / N8f : reproduce 40th at theta = 0 ------------------------
    n7 = json.loads(N7_RESULTS.read_text())
    n7_by_q5 = {round(s["q5_deg"], 9): s["combined"] for s in n7["q5_sweep"]}
    chat0 = axis_dir(0.0, 0.0)
    pf0 = prep(S5, chat0, c0)
    repro, d_dev, b_dev = [], 0.0, 0.0
    for q5 in q5_full_rad:
        pmv = to_link5(SGj, float(q5))
        r = evaluate(0.0, 0.0, pmv, prep_fixed=pf0)
        ref = n7_by_q5[round(math.degrees(q5), 9)]
        dd = abs(r["depth_top_min_mm"] - ref["depth_top_min_mm"])
        db = abs(r["bite_mm"] - ref["bite_mm"])
        d_dev, b_dev = max(d_dev, dd), max(b_dev, db)
        repro.append({"q5_deg": math.degrees(q5), "depth_mm": r["depth_top_min_mm"],
                      "bite_mm": r["bite_mm"], "n7_depth_mm": ref["depth_top_min_mm"],
                      "n7_bite_mm": ref["bite_mm"], "d_depth_mm": dd, "d_bite_mm": db})
    g_e = (d_dev < N8E_TOL_MM) and (b_dev < N8E_TOL_MM)
    out["N8e_theta0_reproduces_40th"] = {
        "pass": bool(g_e), "n_q5": len(repro),
        "max_abs_depth_delta_mm": d_dev, "max_abs_bite_delta_mm": b_dev,
        "tol_mm": N8E_TOL_MM, "per_q5": repro,
        "reference": {"file": N7_RESULTS.name, "sha256_16": sha256(N7_RESULTS)[:16]}}
    print(f"[{LOG}] N8e theta0_vs_40th pass={g_e} max|d_depth|={d_dev:.3e}mm "
          f"max|d_bite|={b_dev:.3e}mm", flush=True)

    th0 = [g for g in grid if g["theta_deg"] == 0.0 and g["q5_deg"] == Q5_ANCHOR_DEG[-1]]
    spread = float(max(x["bite_mm"] for x in th0) - min(x["bite_mm"] for x in th0))
    g_f = spread < N8F_TOL_MM
    out["N8f_theta0_phi_invariant"] = {"pass": bool(g_f), "bite_spread_over_phi_mm": spread,
                                       "n_phi": len(th0), "tol_mm": N8F_TOL_MM}
    print(f"[{LOG}] N8f phi_invariance_at_theta0 pass={g_f} spread={spread:.3e}mm", flush=True)
    if not (g_e and g_f):
        print(f"[{LOG}] ABORT reproduction_gate_failure_N8e_or_N8f", flush=True)
        return 3

    # ---- 7. local refinement, then the FULL q5 sweep at the best ----------
    def tilted_of(rows):
        return [g for g in rows if 0.0 < g["theta_deg"] <= T1_TILT_DEG[1] and g["bite_mm"] is not None]

    coarse_tilted = tilted_of(grid)
    seeds, seen = [], set()
    for g in sorted(coarse_tilted, key=lambda g: g["bite_mm"], reverse=True):
        key = (g["theta_deg"], g["phi_deg"])
        if key in seen:
            continue
        seen.add(key)
        seeds.append(g)
        if len(seeds) == 2:
            break
    ref_pairs, seen_pairs = [], set()
    for s in seeds:
        for th in np.clip(np.arange(s["theta_deg"] - 2.0, s["theta_deg"] + 2.0 + 1e-9, 0.5),
                          0.05, T1_TILT_DEG[1]):
            for ph in np.arange(s["phi_deg"] - 8.0, s["phi_deg"] + 8.0 + 1e-9, 1.0) % 360.0:
                key = (round(float(th), 6), round(float(ph), 6))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    ref_pairs.append(key)
    out["refinement_seeds"] = [{k: s[k] for k in ("theta_deg", "phi_deg", "q5_deg", "bite_mm")}
                               for s in seeds]
    out["refinement_resolution_deg"] = {"theta": 0.5, "phi": 1.0, "window_theta": 2.0,
                                        "window_phi": 8.0, "n_pairs": len(ref_pairs)}
    refined = run_pairs(ref_pairs, "refine_grid")
    out["grid_refined"] = refined

    all_tilted = coarse_tilted + tilted_of(refined)
    best_grid = max(all_tilted, key=lambda g: g["bite_mm"])
    out["best_in_grid_tilted"] = best_grid
    print(f"[{LOG}] best tilted (coarse+refined): theta={best_grid['theta_deg']:.2f} "
          f"phi={best_grid['phi_deg']:.1f} q5={best_grid['q5_deg']:.2f} "
          f"bite={best_grid['bite_mm']:+.4f}mm depth={best_grid['depth_top_min_mm']:+.4f}mm", flush=True)

    chat_b = axis_dir(math.radians(best_grid["theta_deg"]), math.radians(best_grid["phi_deg"]))
    pf_b = prep(S5, chat_b, c0)
    full_sweep = []
    for q5 in q5_full_rad:
        r = evaluate(best_grid["theta_deg"], best_grid["phi_deg"], to_link5(SGj, float(q5)),
                     prep_fixed=pf_b)
        r["q5_deg"] = math.degrees(q5)
        full_sweep.append(r)
    out["full_q5_sweep_at_best_tilt"] = full_sweep
    fs_ok = [g for g in full_sweep if g["bite_mm"] is not None]
    best_full = max(fs_ok, key=lambda g: g["bite_mm"]) if fs_ok else best_grid
    best = best_full if best_full["bite_mm"] > best_grid["bite_mm"] else best_grid
    out["best_overall_tilted"] = best
    print(f"[{LOG}] full q5 sweep at best tilt: max bite={best_full['bite_mm']:+.4f}mm "
          f"at q5={best_full['q5_deg']:.2f}deg", flush=True)

    # ---- 8. N8g : brute-force check of the closed-form solve --------------
    chat_v = np.array(best["chat"])
    pmv_b = to_link5(SGj, math.radians(best["q5_deg"]))
    pf_v, pm_v = prep(S5, chat_v, c0), prep(pmv_b, chat_v, c0)
    d_v = best["delta_m"]
    n_above = (inside_count(*pf_v, chat_v[2], R_m, d_v + 1e-9) +
               inside_count(*pm_v, chat_v[2], R_m, d_v + 1e-9))
    n_below = (inside_count(*pf_v, chat_v[2], R_m, d_v - 1e-6) +
               inside_count(*pm_v, chat_v[2], R_m, d_v - 1e-6))
    g_g = (n_above == 0) and (n_below > 0)
    out["N8g_closed_form_bruteforce_check"] = {
        "pass": bool(g_g), "delta_m": d_v, "n_inside_at_delta_plus_1e-9m": n_above,
        "n_inside_at_delta_minus_1e-6m": n_below}
    print(f"[{LOG}] N8g bruteforce pass={g_g} inside(+1e-9)={n_above} inside(-1e-6)={n_below}",
          flush=True)

    # ---- 9. object-diameter x tilt interaction ----------------------------
    dscan = []
    for R in np.arange(1.0, 30.0 + 0.5, 0.5):
        rm = R / 1000.0
        d_f, _ = deepest_delta(*pf_v, chat_v[2], rm)
        d_m, _ = deepest_delta(*pm_v, chat_v[2], rm)
        dl = max(d_f, d_m)
        if not math.isfinite(dl):
            continue
        bf, _ = bite_at(*pf_v, chat_v[2], rm, SPAN_m, dl)
        bm, _ = bite_at(*pm_v, chat_v[2], rm, SPAN_m, dl)
        bb = max(bf, bm)
        dscan.append({"R_mm": float(R), "D_mm": float(2 * R),
                      "bite_mm": bb * 1000.0 if math.isfinite(bb) else None,
                      "depth_top_min_mm": dl * 1000.0})
    out["radius_scan_at_best_tilt"] = dscan
    feas = [s for s in dscan if s["bite_mm"] is not None and s["bite_mm"] > 0.0]
    out["feasible_radii_at_best_tilt"] = {
        "any": bool(feas),
        "max_D_mm_with_positive_bite": max((s["D_mm"] for s in feas), default=None),
        "best": max(feas, key=lambda s: s["bite_mm"]) if feas else None}

    # ---- 9b. EXPLORATORY: how much tilt WOULD be needed? ------------------
    # Strictly outside T1's measured 0..35 deg range and therefore NOT part of the
    # pre-registered claim. Reported because "how far away is it?" changes the
    # decision the user has to take to the professor, and a bound is honest either way.
    ext_pairs = [(float(t), best_grid["phi_deg"]) for t in np.arange(36.0, 75.0 + 1e-9, 1.0)]
    ext = run_pairs(ext_pairs, "extended_scan_OUTSIDE_T1_range")
    ext_ok = [g for g in ext if g["bite_mm"] is not None and g["bite_mm"] > 0.0]
    out["extended_theta_scan_outside_T1_range"] = {
        "note": ("theta > 35 deg is OUTSIDE T1's measured tilt range (D420 Impl.(2)) and is NOT "
                 "part of the pre-registered claim. Exploratory bound only."),
        "phi_deg": best_grid["phi_deg"], "rows": ext,
        "min_theta_deg_with_positive_bite": min((g["theta_deg"] for g in ext_ok), default=None),
        "max_bite_mm": max((g["bite_mm"] for g in ext if g["bite_mm"] is not None), default=None)}
    print(f"[{LOG}] extended scan (OUTSIDE T1 range, phi={best_grid['phi_deg']:.1f}): "
          f"first positive bite at theta="
          f"{out['extended_theta_scan_outside_T1_range']['min_theta_deg_with_positive_bite']}", flush=True)

    # ---- 10. verdict ------------------------------------------------------
    max_bite_tilted = max([g["bite_mm"] for g in all_tilted] +
                          [g["bite_mm"] for g in full_sweep if g["bite_mm"] is not None])
    admits = bool(max_bite_tilted > 0.0)
    reaches_T1 = bool(max_bite_tilted >= T1_BITE_MM[1])
    g_h, g_i = admits, reaches_T1
    out["verdict"] = {
        "code": "TILT_EXPLAINS_T1_CONTRADICTION" if admits else "TILT_DOES_NOT_EXPLAIN_T1_CONTRADICTION",
        "tilt_admits_any_bite": admits,
        "tilt_reaches_T1_upper_bite_12mm": reaches_T1,
        "max_bite_mm_over_tilted_grid": float(max_bite_tilted),
        "max_bite_mm_at_theta0_from_40th": float(n7["best_over_sweep"]["max_bite_mm"]),
        "improvement_vs_vertical_mm": float(max_bite_tilted - n7["best_over_sweep"]["max_bite_mm"]),
        "best_config": {k: best[k] for k in ("theta_deg", "phi_deg", "q5_deg", "bite_mm",
                                             "depth_top_min_mm", "blocker")},
        "real_photo_verified_bite_mm": list(T1_BITE_MM),
        "real_measured_tilt_deg": list(T1_TILT_DEG),
        "D427_D429_status": "UNCHANGED - this run neither re-runs nor re-judges Gate-0",
        "g0a_pass": False,
    }
    out["gates"] = {"N8a_sha_pins": bool(g_a), "N8b_D427_l_vis_reproduced": bool(g_b),
                    "N8c_joint5_rotation_matches": bool(g_c),
                    "N8d_moving_jaw_z_range_matches_37th": bool(g_d),
                    "N8e_theta0_reproduces_40th": bool(g_e),
                    "N8f_theta0_phi_invariant": bool(g_f),
                    "N8g_closed_form_bruteforce": bool(g_g),
                    "N8h_tilt_admits_rim_pinch": bool(g_h),
                    "N8i_tilt_reaches_T1_12mm": bool(g_i)}
    print(f"[{LOG}] G0B_T3R_N8_VERDICT={out['verdict']['code']} "
          f"max_bite_tilted={max_bite_tilted:+.4f}mm "
          f"(vertical was {n7['best_over_sweep']['max_bite_mm']:+.4f}mm)", flush=True)

    # ---- 11. D324 decision diagnostic -------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(17.0, 11.0))
    qsel = Q5_ANCHOR_DEG[-1]

    per_theta = {}
    for g in grid + refined:
        if g["bite_mm"] is None:
            continue
        k = g["theta_deg"]
        if k not in per_theta or g["bite_mm"] > per_theta[k]["bite_mm"]:
            per_theta[k] = g
    ths = sorted(per_theta)
    ax[0][0].plot(ths, [per_theta[t]["bite_mm"] for t in ths], "-o", ms=3.5, color="#c02020",
                  label="best bite over phi and q5 anchors")
    ax[0][0].axhspan(*T1_BITE_MM, color="#2a9d3a", alpha=0.16, label="T1 real photo bite 0-12mm")
    ax[0][0].axhline(0.0, color="k", lw=1.0)
    ax[0][0].axhline(n7["best_over_sweep"]["max_bite_mm"], color="#1f6fb4", ls=":", lw=1.4,
                     label=f"40th vertical best {n7['best_over_sweep']['max_bite_mm']:+.2f}mm")
    ax[0][0].set_xlabel("tool-axis tilt theta [deg]   (T1 measured 0-35, D420 (2))")
    ax[0][0].set_ylabel("bite below cylinder top face [mm]")
    ax[0][0].set_title("A | does an approach TILT let the jaws get beside D29?")
    ax[0][0].legend(fontsize=8)
    ax[0][0].grid(alpha=0.3)

    Z = np.full((len(THETA_DEG), len(PHI_DEG)), np.nan)
    ti = {float(t): i for i, t in enumerate(THETA_DEG)}
    pi = {float(p): i for i, p in enumerate(PHI_DEG)}
    for g in grid:
        if g["q5_deg"] == qsel and g["bite_mm"] is not None:
            Z[ti[g["theta_deg"]], pi[g["phi_deg"]]] = g["bite_mm"]
    im = ax[0][1].imshow(Z, aspect="auto", origin="lower", cmap="RdYlGn",
                         extent=[PHI_DEG[0], PHI_DEG[-1], THETA_DEG[0], THETA_DEG[-1]],
                         vmin=float(np.nanmin(Z)), vmax=max(0.5, float(np.nanmax(Z))))
    fig.colorbar(im, ax=ax[0][1], label="bite [mm]")
    ax[0][1].set_xlabel("tilt azimuth phi [deg] (tool frame; D427 blocker sits at az 172)")
    ax[0][1].set_ylabel("tilt theta [deg]")
    ax[0][1].set_title(f"B | bite over (theta, phi) at q5 = {qsel:.2f} deg")

    ax[1][0].plot(ths, [per_theta[t]["depth_top_min_mm"] for t in ths], "-o", ms=3.5,
                  color="#1f6fb4", label="deepest reachable top-face plane")
    ax[1][0].axhline(4.4, color="#e08a1e", ls="--", lw=1.4,
                     label="attempt1 physical stop top+4.4mm (D424)")
    ax[1][0].set_xlabel("tool-axis tilt theta [deg]")
    ax[1][0].set_ylabel("depth = z - TCP [mm]")
    ax[1][0].set_title("C | descent limit vs tilt (theta=0 must land on 4.4576)")
    ax[1][0].legend(fontsize=8)
    ax[1][0].grid(alpha=0.3)

    ax[1][1].plot([s["D_mm"] for s in n7["radius_scan_at_88_31deg"]],
                  [s["bite_mm"] for s in n7["radius_scan_at_88_31deg"]], "-o", ms=3,
                  color="#8a8a8a", label="40th: vertical, q5=88.31")
    dsf = [s for s in dscan if s["bite_mm"] is not None]
    ax[1][1].plot([s["D_mm"] for s in dsf], [s["bite_mm"] for s in dsf], "-o", ms=3,
                  color="#7a3fb5",
                  label=f"best tilt theta={best['theta_deg']:.0f} phi={best['phi_deg']:.0f} "
                        f"q5={best['q5_deg']:.1f}")
    ax[1][1].axhline(0.0, color="k", lw=1.0)
    ax[1][1].axvline(29.0, color="#c02020", ls="--", lw=1.4, label="current object D29")
    ax[1][1].axvline(20.2487442193214, color="#2a9d3a", ls=":", lw=1.4,
                     label="vertical admission edge D20.2487 (41st F2)")
    ax[1][1].set_xlabel("object diameter [mm]")
    ax[1][1].set_ylabel("bite [mm]  (>0 = admitted)")
    ax[1][1].set_title("D | admissible diameter band: vertical vs best tilt")
    ax[1][1].legend(fontsize=8)
    ax[1][1].grid(alpha=0.3)

    fig.suptitle(f"g0b_d420 {TAG} - approach TILT cross-check (read-only).  "
                 f"VERDICT: {out['verdict']['code']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(paths["diagnostic.png"], dpi=125)
    plt.close(fig)

    # ---- 12. D341 Rerun artifact (save-only) ------------------------------
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    app_id = f"roarm_g0b_{TAG}"

    def view_pts(pts_m):
        o = np.asarray(pts_m) - TCP_LOCAL[None, :]
        return np.column_stack([o[:, 0], o[:, 1], -o[:, 2]])

    def perp_basis(c):
        a = np.array([1.0, 0.0, 0.0]) if abs(c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(c, a)
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(c, e1)

    def cyl_wire(chat, delta_m, R_mm, H_mm, ring_ks_mm=(0.0, 5.0, 12.0, 25.0, 50.0), n_ring=64,
                 n_wall=24):
        """Wireframe of the tilted cylinder, in Rerun view coordinates."""
        cc = np.array([0.0, 0.0, TCP_Z_MM / 1000.0 + delta_m])
        e1, e2 = perp_basis(chat)
        rr_m = R_mm / 1000.0
        rings = []
        for k in ring_ks_mm:
            base = cc + chat * (k / 1000.0)
            ang = np.linspace(0.0, 2 * math.pi, n_ring + 1)
            rings.append(view_pts(base[None, :] + rr_m * (np.cos(ang)[:, None] * e1[None, :] +
                                                          np.sin(ang)[:, None] * e2[None, :])).tolist())
        walls = []
        for a in np.linspace(0.0, 2 * math.pi, n_wall, endpoint=False):
            off = rr_m * (math.cos(a) * e1 + math.sin(a) * e2)
            walls.append(view_pts(np.array([cc + off, cc + chat * (H_mm / 1000.0) + off])).tolist())
        return rings, walls

    def topface_ring(chat, delta_m, R_mm, n=96):
        cc = np.array([0.0, 0.0, TCP_Z_MM / 1000.0 + delta_m])
        e1, e2 = perp_basis(chat)
        ang = np.linspace(0.0, 2 * math.pi, n + 1)
        rr_m = R_mm / 1000.0
        return view_pts(cc[None, :] + rr_m * (np.cos(ang)[:, None] * e1[None, :] +
                                              np.sin(ang)[:, None] * e2[None, :])).tolist()

    theta_rows = [per_theta[t] for t in ths]
    P_mov_best = to_link5(SGj, math.radians(best["q5_deg"]))

    with rr.RecordingStream(app_id, recording_id=f"g0b_d420_{TAG}", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(paths["timeline.rrd"]), write_footer=True)
        rec.log("assembly/link5_fixed_jaw", rr.Points3D(view_pts(S5[::VIEW_STRIDE]),
                colors=[150, 150, 160], radii=0.0004), static=True)
        rec.log("assembly/gripper_link_moving_jaw", rr.Points3D(view_pts(P_mov_best[::VIEW_STRIDE]),
                colors=[70, 130, 230], radii=0.0004), static=True)
        rec.log("assembly/tcp", rr.Points3D([[0.0, 0.0, 0.0]], colors=[[40, 200, 80]],
                radii=0.002), static=True)
        rec.log("assembly/tool_axis", rr.LineStrips3D([[[0.0, 0.0, 0.07], [0.0, 0.0, -0.07]]],
                colors=[[40, 200, 80]], radii=0.0003), static=True)
        rec.log("reference/d427_blocker_peak", rr.Points3D(
                [[-0.010025849, 0.001408991, -(119.885620 - TCP_Z_MM) / 1000.0]],
                colors=[[255, 210, 40]], radii=0.0016), static=True)

        for i, row in enumerate(theta_rows):
            chat = np.array(row["chat"]) if row.get("chat") else axis_dir(
                math.radians(row["theta_deg"]), math.radians(row["phi_deg"]))
            rec.reset_time()
            rec.set_time("theta_index", sequence=i)
            rings, walls = cyl_wire(chat, row["delta_m"], CYL_R_MM, CYL_H_MM)
            rec.log("object/cylinder_tilted", rr.LineStrips3D(
                rings + walls, colors=[[225, 60, 60]] * (len(rings) + len(walls)), radii=0.00035))
            rec.log("object/topface_plane_tilted", rr.LineStrips3D(
                [topface_ring(chat, row["delta_m"], 30.0)], colors=[[235, 140, 30]], radii=0.0005))
            rec.log("object/cylinder_axis", rr.LineStrips3D(
                [view_pts(np.array([[0.0, 0.0, TCP_Z_MM / 1000.0 + row["delta_m"]],
                                    [chat[0] * 0.06, chat[1] * 0.06,
                                     TCP_Z_MM / 1000.0 + row["delta_m"] + chat[2] * 0.06]])).tolist()],
                colors=[[255, 120, 200]], radii=0.0004))
            rec.log("plots/theta_deg", rr.Scalars(float(row["theta_deg"])))
            rec.log("plots/best_phi_deg", rr.Scalars(float(row["phi_deg"])))
            rec.log("plots/bite_mm", rr.Scalars(float(row["bite_mm"])))
            rec.log("plots/depth_top_min_mm", rr.Scalars(float(row["depth_top_min_mm"])))

        rec.reset_time()
        rec.set_time("theta_index", sequence=0)
        for name, ok in (("N8a_sha_pins", g_a), ("N8b_D427_l_vis_reproduced", g_b),
                         ("N8c_joint5_rotation_matches", g_c),
                         ("N8d_moving_jaw_z_range_matches_37th", g_d),
                         ("N8e_theta0_reproduces_40th", g_e),
                         ("N8f_theta0_phi_invariant", g_f),
                         ("N8g_closed_form_bruteforce", g_g),
                         ("N8h_tilt_admits_rim_pinch", g_h),
                         ("N8i_tilt_reaches_T1_12mm", g_i)):
            rec.log("events/gates", rr.TextLog(name, level=rr.TextLogLevel.INFO if ok
                                               else rr.TextLogLevel.ERROR))
        summary_md = (
            f"# g0b_d420 {TAG} - approach TILT cross-check (read-only)\n\n"
            f"**VERDICT: {out['verdict']['code']}**\n\n"
            f"## Question\n40th showed the URDF assembly reproduces the sim physics runs to "
            f"0.058 mm but could not reproduce T1's photo-verified rim pinch of 0-12 mm. "
            f"40th §5 named the suspect and left it untested: T1's tool axis was tilted "
            f"**0-35 deg, frame to frame** (D420 Impl.(2)) while T2/T3 pre-registered a "
            f"**fully vertical** axis (D421 tilt 0.1989 deg). The model was being asked to "
            f"reproduce a pose the real arm never held.\n\n"
            f"## Result\n"
            f"- best bite over the tilted grid: **{max_bite_tilted:+.4f} mm** "
            f"(vertical best was {n7['best_over_sweep']['max_bite_mm']:+.4f} mm -> change "
            f"**{max_bite_tilted - n7['best_over_sweep']['max_bite_mm']:+.4f} mm**);\n"
            f"- best config: theta = **{best['theta_deg']:.0f} deg**, phi = "
            f"**{best['phi_deg']:.0f} deg**, q5 = **{best['q5_deg']:.2f} deg**, "
            f"deepest top face = TCP**{best['depth_top_min_mm']:+.4f} mm**;\n"
            f"- T1's real bite band is **0-12 mm**; reached = "
            f"**{out['verdict']['tilt_reaches_T1_upper_bite_12mm']}**;\n"
            f"- largest admitted diameter at the best tilt: "
            f"**{out['feasible_radii_at_best_tilt']['max_D_mm_with_positive_bite']} mm** "
            f"(vertical was 20 mm; current object is D29).\n\n"
            f"## Reproduction gates\n"
            f"- N8e: at theta = 0 this formulation reproduces 40th's per-q5 numbers over all "
            f"34 q5 values, max|delta| depth **{d_dev:.3e} mm** / bite **{b_dev:.3e} mm**;\n"
            f"- N8f: at theta = 0 the result is phi-invariant, spread **{spread:.3e} mm**;\n"
            f"- N8g: brute force at the decisive config - inside count "
            f"**{n_above}** at delta+1e-9 m, **{n_below}** at delta-1e-6 m.\n\n"
            f"## Scene\ngrey = link5 (fixed jaw), blue = gripper_link (moving jaw at the best "
            f"opening), red = the D29 cylinder at its deepest admissible height for the tilt of "
            f"the current timeline step, pink = the cylinder axis, orange = its top-face plane, "
            f"yellow = the D427 blocking peak (r 10.1244 mm, z 119.8856 mm), green = TCP and tool "
            f"axis. View convention: z_view = -(z - z_TCP), so DOWN = distal.\n\n"
            f"Authority = stdout + `{paths['results.json'].name}`. Rerun is inspection evidence "
            f"only (D341). Static admission test only - bite > 0 is necessary, never sufficient.\n"
        )
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN),
                static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run",
                                         name="1 | tilt verdict"),
                    rrb.Spatial3DView(origin="/", contents=["/assembly/**", "/object/**",
                                                            "/reference/**"],
                                      name="2 | tilted cylinder vs assembled gripper"),
                    rrb.TextLogView(origin="/events/gates", contents="/events/gates/**",
                                    name="3 | gates"),
                    column_shares=[0.30, 0.46, 0.24],
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/bite_mm/**", "/plots/depth_top_min_mm/**"],
                                       name="4 | bite [mm] and descent limit vs tilt"),
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/theta_deg/**", "/plots/best_phi_deg/**"],
                                       name="5 | swept tilt theta / best phi [deg]"),
                ),
                row_shares=[0.58, 0.42],
            ),
            auto_layout=False, auto_views=False, collapse_panels=True,
        )
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=30.0)
    blueprint.save(app_id, str(paths["timeline.rbl"]))

    expected_entities = ["metadata/run", "assembly/link5_fixed_jaw",
                         "assembly/gripper_link_moving_jaw", "assembly/tcp", "assembly/tool_axis",
                         "reference/d427_blocker_peak", "object/cylinder_tilted",
                         "object/topface_plane_tilted", "object/cylinder_axis",
                         "plots/theta_deg", "plots/best_phi_deg", "plots/bite_mm",
                         "plots/depth_top_min_mm", "events/gates"]
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    lin3 = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    components = {
        "metadata/run": ["TextDocument:text"],
        "assembly/link5_fixed_jaw": pts3, "assembly/gripper_link_moving_jaw": pts3,
        "assembly/tcp": pts3, "reference/d427_blocker_peak": pts3,
        "assembly/tool_axis": lin3, "object/cylinder_tilted": lin3,
        "object/topface_plane_tilted": lin3, "object/cylinder_axis": lin3,
        "plots/theta_deg": ["Scalars:scalars"], "plots/best_phi_deg": ["Scalars:scalars"],
        "plots/bite_mm": ["Scalars:scalars"], "plots/depth_top_min_mm": ["Scalars:scalars"],
        "events/gates": ["TextLog:text", "TextLog:level"],
    }
    validation = validate_rerun_artifact(
        paths["timeline.rrd"],
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time", "theta_index"],
        expected_entity_components=components,
        blueprint_path=paths["timeline.rbl"],
        screenshot_path=paths["inspection.png"],
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=240.0,
    )
    paths["rerun_validation.json"].write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(f"[{LOG}] rerun_validation pass={validation.get('pass')} errors={validation.get('errors')}",
          flush=True)

    shutil.copyfile(__file__, paths["script.py.txt"])
    # D429-R1 (2)(3): the manifest must NOT contain its own file - hash results.json from disk.
    out["artifacts"] = {k: {"name": v.name, "sha256": sha256(v)[:16], "bytes": v.stat().st_size}
                        for k, v in paths.items() if v.exists() and k != "results.json"}
    out["artifacts_note"] = ("results.json is deliberately absent from this manifest - D429-R1: a "
                             "self-referential manifest can never carry the file's final hash. "
                             "Hash it from disk.")
    out["rerun_validation_pass"] = bool(validation.get("pass"))
    out["wall_seconds"] = round(time.time() - t_start, 1)
    paths["results.json"].write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{LOG}] artifacts " + " ".join(f"{v['name']}={v['sha256']}"
                                           for v in out["artifacts"].values()), flush=True)
    print(f"[{LOG}] results.json={sha256(paths['results.json'])[:16]} "
          f"bytes={paths['results.json'].stat().st_size}", flush=True)
    print(f"[{LOG}] G0B_T3R_N8_VERDICT={out['verdict']['code']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
