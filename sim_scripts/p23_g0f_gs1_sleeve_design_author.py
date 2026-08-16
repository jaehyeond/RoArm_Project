#!/usr/bin/env python3
"""p23 / gs1 — g0f_d452 jaw-sleeve design + authoring (G-step, no physics).

Contract: claudedocs/runtime_logs/grasp_track/g0f_d452/gs1_prereg.md SS2.
Derives every design number reproducibly from two pinned sources (frozen fg1
gripper USD + fg2 results stop21 rows), applies gates G-kin / G-contact /
G-design / G-interference, then authors:
  gs1_gripper_sleeved.usd  (reference fg1_gripper_only.usd + 2 convex pad
                            meshes per jaw, decomposed collision - D446)
  gs1_sleeve_link5.stl / gs1_sleeve_gripper.stl  (mm, print candidates)
  gs1_design.json          (all numbers + gate results)

Free constants fixed by the prereg: t=3.5 mm, alpha=10 deg, design width 28 mm,
u-extent +/-15 mm, w-extent -14/+12 mm, back embed 0.7 mm, 2 convex pieces/jaw.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0f_d452"
FG1_USD = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d444/fg1_gripper_only.usd"
FG2_RESULTS = REPO / "claudedocs/runtime_logs/grasp_track/g0e_d451/fg2_results.json"
PINS = {
    FG1_USD: "0e9f c601 df93 79fa bc11 8eb2 495a c010 0350 ef99 3166 2413 c5a2 c0f0 0690 dd76".replace(" ", ""),
    FG2_RESULTS: "3b591352555bdaf02a3ca5259b52275263eac4db86482472225dc21ca7cd4a36",
}

T_SLEEVE = 0.0035
ALPHA_DEG = 10.0
DESIGN_WIDTH = 0.028
U_EXT = 0.015
W_NEG, W_POS = 0.014, 0.012
BACK_EMBED = 0.0007
OBJ_RADIUS = 0.0145
OBJ_D = 0.029
HALF_H = 0.025
FULL_CLOSE_DEG = 14.0
CONTACT_BRACKET = (21.07, 23.0)
PLA_G_CM3 = 1.24

OUT_USD = CASE_DIR / "gs1_gripper_sleeved.usd"
OUT_STL5 = CASE_DIR / "gs1_sleeve_link5.stl"
OUT_STLG = CASE_DIR / "gs1_sleeve_gripper.stl"
OUT_JSON = CASE_DIR / "gs1_design.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    for p, want in PINS.items():
        got = sha256(p)
        if got != want:
            raise RuntimeError(f"SHA_DRIFT {p} {got}")
    for p in (OUT_USD, OUT_STL5, OUT_STLG, OUT_JSON):
        if p.exists():
            raise RuntimeError(f"WRITE_GUARD {p}")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    rc = 1
    try:
        rc = inner(t0)
    except BaseException:  # noqa: BLE001 - D447: close() swallows exceptions
        import traceback
        tb = traceback.format_exc()
        print(f"[g0f_gs1] FAILURE\n{tb}", flush=True)
        (CASE_DIR / "gs1_p23_failure.txt").write_text(tb)
        rc = 3
    finally:
        print(f"[g0f_gs1] PRE_CLOSE_SENTINEL rc={rc}", flush=True)
        sys.stdout.flush()
        app.close()
    return rc


def inner(t0) -> int:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

    D: dict = {"tool": "gs1_p23", "case": "g0f_d452", "prereg": "gs1_prereg.md",
               "pins": {p.name: v for p, v in PINS.items()},
               "constants": {"t_m": T_SLEEVE, "alpha_deg": ALPHA_DEG,
                             "design_width_m": DESIGN_WIDTH, "u_ext_m": U_EXT,
                             "w_neg_m": W_NEG, "w_pos_m": W_POS,
                             "back_embed_m": BACK_EMBED},
               "rev1": {"what": "valley planes anchored to own stock ridge + t "
                                "along the pinch axis; pad back depth derived "
                                "from strip burial analysis (not the prereg "
                                "fixed 0.7 mm embed)",
                        "why": "moving stock face tilted ~23.5 deg: gap-model "
                               "valley placement failed its offset sanity "
                               "(1.03 mm) and a fixed-depth flat back cannot "
                               "bury the tilted stock face across the "
                               "footprint",
                        "when": "reactive, after the first p23 run aborted on "
                                "MOVING_VALLEY_OFFSET_SANITY; no physics was "
                                "consumed"}}

    st = Usd.Stage.Open(str(FG1_USD))
    pred = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    j = st.GetPrimAtPath("/fg1_gripper/joints/link5_to_gripper_link")
    lp0 = np.array(j.GetAttribute("physics:localPos0").Get(), dtype=np.float64)
    lr0 = j.GetAttribute("physics:localRot0").Get()
    lp1 = np.array(j.GetAttribute("physics:localPos1").Get(), dtype=np.float64)
    lr1 = j.GetAttribute("physics:localRot1").Get()
    if np.abs(lp1).max() > 1e-12:
        raise RuntimeError("JOINT_LP1_NONZERO")

    def quat_to_R(q):
        w = q.GetReal(); x, y, z = q.GetImaginary()
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    R0 = quat_to_R(lr0)
    R1 = quat_to_R(lr1)
    if np.abs(R1 - np.eye(3)).max() > 1e-9:
        raise RuntimeError("JOINT_LR1_NOT_IDENTITY")

    def rotz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def X(theta_deg):
        return R0 @ rotz(np.radians(theta_deg)), lp0

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())

    def local_points(link_path, sub):
        link = st.GetPrimAtPath(link_path)
        L = np.array(xc.GetLocalToWorldTransform(link), dtype=np.float64).T
        Linv = np.linalg.inv(L)
        pts = []
        root = st.GetPrimAtPath(link_path + "/" + sub)
        for prim in Usd.PrimRange(root, pred):
            if prim.IsA(UsdGeom.Mesh):
                M = np.array(xc.GetLocalToWorldTransform(prim), dtype=np.float64).T
                rel = Linv @ M
                p = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                ph = np.hstack([p, np.ones((len(p), 1))])
                pts.append((rel @ ph.T).T[:, :3])
        return np.vstack(pts)

    l5_col = local_points("/fg1_gripper/link5", "collisions")
    gl_col = local_points("/fg1_gripper/gripper_link", "collisions")
    D["point_counts"] = {"l5_col": int(len(l5_col)), "gl_col": int(len(gl_col))}

    # ---- G-kin: reproduce fg2 stop21 moving-jaw poses ---------------------- #
    fg2 = json.loads(FG2_RESULTS.read_text())
    snaps = fg2["pose_snaps"]

    def qnp_to_R(q):
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    kin = []
    for gi in range(8, 16):
        pc = snaps[gi]["post_close"]
        Rl5, tl5 = qnp_to_R(pc["root_quat_wxyz"]), np.array(pc["link5_pos"])
        Rrel, trel = X(pc["q5_deg"])
        tp = Rl5 @ trel + tl5
        Rp = Rl5 @ Rrel
        kin.append({"gi": gi,
                    "t_err_m": float(np.linalg.norm(tp - np.array(pc["moving_pos"]))),
                    "R_err": float(np.abs(Rp - qnp_to_R(pc["moving_quat_wxyz"])).max())})
    g_kin = all(r["t_err_m"] < 1e-6 and r["R_err"] < 1e-5 for r in kin)
    D["gate_kin"] = {"rows": kin, "pass": bool(g_kin)}
    if not g_kin:
        raise RuntimeError("GATE_KIN_FAIL")

    # object anchor in l5 (mean of stop21 rows)
    o_rows, a_rows = [], []
    for gi in range(8, 16):
        pc = snaps[gi]["post_close"]
        Rl5, tl5 = qnp_to_R(pc["root_quat_wxyz"]), np.array(pc["link5_pos"])
        o_rows.append(Rl5.T @ (np.array(pc["obj_pos"]) - tl5))
        a_rows.append((Rl5.T @ qnp_to_R(pc["obj_quat_wxyz"]))[:, 2])
    o_mean = np.mean(o_rows, axis=0)
    a_mean = np.mean(a_rows, axis=0)
    a_mean /= np.linalg.norm(a_mean)
    D["object_anchor_l5"] = {"center": o_mean.tolist(), "cyl_axis": a_mean.tolist()}
    if abs(a_mean[1]) < 0.999:
        raise RuntimeError("CYL_AXIS_NOT_Y")

    # ---- G-contact: stock clearance curve + contact angle ------------------ #
    def band_min_radius(pts):
        d = pts - o_mean
        axc = d @ a_mean
        m = np.abs(axc) < HALF_H
        rad = np.linalg.norm(d[m] - np.outer(axc[m], a_mean), axis=1)
        return float(rad.min())

    r5_min = band_min_radius(l5_col)

    def gl_min_radius(theta_deg):
        Rrel, trel = X(theta_deg)
        return band_min_radius(gl_col @ Rrel.T + trel)

    lo, hi = 14.0, 40.0
    for _ in range(60):  # bisect: clearance(theta)=r5+rg-OBJ_D crossing
        mid = 0.5 * (lo + hi)
        if r5_min + gl_min_radius(mid) - OBJ_D > 0:
            hi = mid
        else:
            lo = mid
    stock_contact_deg = 0.5 * (lo + hi)
    g_contact = CONTACT_BRACKET[0] < stock_contact_deg < CONTACT_BRACKET[1]
    slope = ((r5_min + gl_min_radius(stock_contact_deg + 2.0))
             - (r5_min + gl_min_radius(stock_contact_deg))) / 2.0 * 1000.0
    D["gate_contact"] = {"stock_contact_deg": stock_contact_deg,
                        "fg2_bracket_deg": CONTACT_BRACKET,
                        "gap_slope_mm_per_deg": slope, "pass": bool(g_contact)}
    if not g_contact:
        raise RuntimeError(f"GATE_CONTACT_FAIL {stock_contact_deg}")

    # stock wedge angle at hold (recorded, not gated)
    Rrel, trel = X(21.06)
    gl_at = gl_col @ Rrel.T + trel
    d = gl_at - o_mean
    axc = d @ a_mean
    m = np.abs(axc) < HALF_H
    rv = d[m] - np.outer(axc[m], a_mean)
    i = np.linalg.norm(rv, axis=1).argmin()
    dirn = rv[i] / np.linalg.norm(rv[i])
    D["stock_wedge_deg_at_hold"] = float(np.degrees(np.arccos(abs(dirn[0]))))

    # ---- design frame numbers (REV-1: ridge-anchored valley placement) ----- #
    # REV-1 rationale: the moving stock face is tilted ~23.5 deg w.r.t. the pinch
    # axis, so gap-model algebra misplaces the moving valley (initial run failed
    # its own offset sanity at 1.03 mm) and a flat back cannot bury the tilted
    # stock face.  Fix: (a) place each valley plane at (own stock ridge + t)
    # along the pinch axis +x, (b) derive each pad's back depth from a strip
    # burial analysis over the pad footprint, gated so the stock surface never
    # protrudes past the pad inner face.
    pts_band = l5_col[np.abs((l5_col - o_mean) @ a_mean) < HALF_H]
    radf = np.linalg.norm((pts_band - o_mean)
                          - np.outer((pts_band - o_mean) @ a_mean, a_mean), axis=1)
    x_f0 = float(pts_band[radf.argmin()][0])
    x_v = x_f0 + T_SLEEVE
    o_y, o_z = float(o_mean[1]), float(o_mean[2])

    def gl_face_region(theta_deg):
        Rr, tr = X(theta_deg)
        p = gl_col @ Rr.T + tr
        m = (np.abs(p[:, 1] - o_y) < HALF_H) & (np.abs(p[:, 2] - o_z) < 0.02)
        return p[m]

    def x_ridge_min(theta_deg):
        return float(gl_face_region(theta_deg)[:, 0].min())

    def solve_theta(target_gap):
        lo, hi = 14.0, 45.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if x_ridge_min(mid) - T_SLEEVE - x_v > target_gap:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    theta_design = solve_theta(DESIGN_WIDTH)
    sleeved_contact_pred = solve_theta(OBJ_D)
    g_design = (theta_design < sleeved_contact_pred < theta_design + 3.0
                and 0.0 < theta_design < 88.0)
    D["gate_design"] = {"x_f0_m": x_f0, "x_valley_m": x_v,
                        "theta_design_deg": theta_design,
                        "x_ridge_min_at_design_m": x_ridge_min(theta_design),
                        "sleeved_contact_pred_deg_D29": sleeved_contact_pred,
                        "note": "valley planes = own stock ridge + t along the "
                                "pinch axis; prediction ignores the V recession "
                                "(approximation, ~+0.5 deg)",
                        "pass": bool(g_design)}
    if not g_design:
        raise RuntimeError("GATE_DESIGN_FAIL")

    # ---- pad piece construction ------------------------------------------- #
    tan_a = np.tan(np.radians(ALPHA_DEG))

    def hexa(frame_R, frame_t, n_back, n_valley, w_lo, w_hi, taper_sign):
        """8-vertex oblique prism in a (u,w,n) pad frame -> world verts.
        frame_R columns = [u, w, n]; n_back/n_valley = n-coords of back/valley;
        w in [w_lo, w_hi]; inner face recedes from valley by tan_a*|w| on the
        taper side (taper_sign = -1 piece covers w<0, +1 piece covers w>0)."""
        verts = []
        for u in (-U_EXT, U_EXT):
            for w in (w_lo, w_hi):
                inner_n = n_valley - tan_a * abs(w) * (1 if np.sign(w) == taper_sign or w == 0 else 1)
                verts.append([u, w, n_back])
                verts.append([u, w, inner_n])
        V = np.array(verts, dtype=np.float64)
        return V @ frame_R.T + frame_t

    def piece_pair(frame_R, frame_t, n_back, n_valley):
        a = hexa(frame_R, frame_t, n_back, n_valley, -W_NEG, 0.0, -1)
        b = hexa(frame_R, frame_t, n_back, n_valley, 0.0, W_POS, +1)
        return a, b

    def burial_analysis(stock_pts_padframe, n_valley):
        """Strip analysis over the pad footprint: per 1 mm w-strip take the
        outermost stock n (the surface facing the object).  Returns the back
        depth needed to span the stock surface and the worst protrusion of the
        stock past the pad inner face (must stay negative)."""
        q = stock_pts_padframe
        m = (np.abs(q[:, 0]) < U_EXT) & (q[:, 1] > -W_NEG) & (q[:, 1] < W_POS) \
            & (q[:, 2] > n_valley - 0.03) & (q[:, 2] < n_valley + 0.01)
        q = q[m]
        strips = []
        p_max = -np.inf
        n_face_min = np.inf
        for wc in np.arange(-W_NEG + 0.0005, W_POS, 0.001):
            s = q[np.abs(q[:, 1] - wc) < 0.0005]
            if len(s) == 0:
                continue
            n_face = float(s[:, 2].max())
            inner_n = n_valley - tan_a * abs(wc)
            strips.append({"w_m": float(wc), "n_face_m": n_face,
                           "protrusion_m": n_face - inner_n})
            p_max = max(p_max, n_face - inner_n)
            n_face_min = min(n_face_min, n_face)
        return {"n_strips": len(strips), "p_max_m": float(p_max),
                "n_face_min_m": float(n_face_min),
                "back_n_m": float(n_face_min - BACK_EMBED)}

    # fixed jaw pad frame: u=+y, w=+z, n=+x  (l5 frame), origin under the valley
    Rf = np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float64)
    tf = np.array([0.0, o_y, o_z])
    bur_f = burial_analysis((l5_col - tf) @ Rf, x_v)

    # moving jaw pad frame at theta_design: in l5 frame u=+y, w=+z, n=-x;
    # valley plane at x = x_ridge_min(theta_design) - t; transform into gl frame.
    Rrel_d, trel_d = X(theta_design)
    u_gl = Rrel_d.T @ np.array([0, 1, 0.0])
    w_gl = Rrel_d.T @ np.array([0, 0, 1.0])
    n_gl = Rrel_d.T @ np.array([-1, 0, 0.0])
    Rm = np.column_stack([u_gl, w_gl, n_gl])
    x_mv = x_ridge_min(theta_design) - T_SLEEVE
    p_valley_l5 = np.array([x_mv, o_y, o_z])
    tm = Rrel_d.T @ (p_valley_l5 - trel_d)
    bur_m = burial_analysis((gl_col - tm) @ Rm, 0.0)

    g_burial = bur_f["p_max_m"] <= -0.0003 and bur_m["p_max_m"] <= -0.0003
    D["gate_burial"] = {"fixed": bur_f, "moving": bur_m,
                        "x_moving_valley_l5_at_design_m": x_mv,
                        "sleeved_gap_at_design_m": x_mv - x_v,
                        "pass": bool(g_burial)}
    if not g_burial:
        raise RuntimeError(f"GATE_BURIAL_FAIL fixed={bur_f['p_max_m']} "
                           f"moving={bur_m['p_max_m']}")

    fixA, fixB = piece_pair(Rf, tf, bur_f["back_n_m"], x_v)
    movA, movB = piece_pair(Rm, tm, bur_m["back_n_m"], 0.0)
    # moving pad frame origin sits on the valley line (valley n-coord 0).

    pieces = {"link5": {"padA": fixA, "padB": fixB},
              "gripper_link": {"padA": movA, "padB": movB}}

    # convexity check (each piece: every vertex behind every triangulated face)
    TRI = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
           (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
           (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6)]

    def convexity_ok(V):
        c = V.mean(0)
        for (i, jx, k) in TRI:
            n = np.cross(V[jx] - V[i], V[k] - V[i])
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue
            n = n / ln
            if (V[i] - c) @ n < 0:
                n = -n
            if ((V - V[i]) @ n).max() > 1e-9:
                return False
        return True

    conv = {f"{lk}/{nm}": bool(convexity_ok(V))
            for lk, d2 in pieces.items() for nm, V in d2.items()}
    D["convexity"] = conv
    if not all(conv.values()):
        raise RuntimeError("CONVEXITY_FAIL")

    # ---- G-interference ---------------------------------------------------- #
    def sample_surface(V, n=6):
        # barycentric samples over the 12 triangles
        pts = []
        for (i, jx, k) in TRI:
            for a in np.linspace(0, 1, n):
                for b in np.linspace(0, 1 - a, max(1, int(n * (1 - a)))):
                    pts.append(V[i] + a * (V[jx] - V[i]) + b * (V[k] - V[i]))
        return np.array(pts)

    fix_s = np.vstack([sample_surface(fixA), sample_surface(fixB)])
    mov_s = np.vstack([sample_surface(movA), sample_surface(movB)])

    def min_dist(A, B):
        best = np.inf
        for chunk in np.array_split(A, max(1, len(A) // 512)):
            dd = np.linalg.norm(chunk[:, None, :] - B[None, :, :], axis=2)
            best = min(best, float(dd.min()))
        return best

    interf = []
    for th in np.arange(88.0, 13.9, -2.0).tolist() + [FULL_CLOSE_DEG]:
        Rr, tr = X(th)
        mov_l5 = mov_s @ Rr.T + tr
        interf.append({"theta_deg": float(th),
                       "sleeve_sleeve_min_m": min_dist(fix_s, mov_l5)})
    d_at_close = [r for r in interf if abs(r["theta_deg"] - FULL_CLOSE_DEG) < 1e-9][0]
    # sleeve vs opposite stock jaw at full close (vertex-cloud proxy)
    Rr, tr = X(FULL_CLOSE_DEG)
    mov_l5s = mov_s @ Rr.T + tr
    d_fix_vs_glstock = min_dist(fix_s, gl_col @ Rr.T + tr)
    d_mov_vs_l5stock = min_dist(mov_l5s, l5_col)
    g_interf = (d_at_close["sleeve_sleeve_min_m"] > 0.0005
                and d_fix_vs_glstock > 0.0003 and d_mov_vs_l5stock > 0.0003)
    D["gate_interference"] = {
        "curve": interf, "at_full_close": d_at_close,
        "fix_sleeve_vs_moving_stock_m": d_fix_vs_glstock,
        "moving_sleeve_vs_fixed_stock_m": d_mov_vs_l5stock, "pass": bool(g_interf)}
    if not g_interf:
        raise RuntimeError("GATE_INTERFERENCE_FAIL")

    # mass estimate (per piece volume via divergence theorem on tris)
    def volume(V):
        v = 0.0
        for (i, jx, k) in TRI:
            n = np.cross(V[jx] - V[i], V[k] - V[i])
            c = V.mean(0)
            nn = n if (V[i] - c) @ n >= 0 else -n
            v += float(V[i] @ np.cross(V[jx] - V[i], V[k] - V[i])) / 6.0 * (1 if (nn == n).all() else -1)
        return abs(v)

    vols = {f"{lk}/{nm}": volume(V) for lk, d2 in pieces.items() for nm, V in d2.items()}
    D["mass_estimate_g"] = {k: v * 1e6 * PLA_G_CM3 for k, v in vols.items()}
    D["mass_estimate_g"]["total"] = sum(v * 1e6 * PLA_G_CM3 for v in vols.values())

    # ---- author USD -------------------------------------------------------- #
    layer = Usd.Stage.CreateNew(str(OUT_USD))
    UsdGeom.SetStageUpAxis(layer, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(layer, 1.0)
    root = layer.DefinePrim("/gs1_gripper", "Xform")
    layer.SetDefaultPrim(root)
    root.GetReferences().AddReference(str(FG1_USD), "/fg1_gripper")
    for lk, d2 in pieces.items():
        for nm, V in d2.items():
            mpath = f"/gs1_gripper/{lk}/gs1_sleeve_{nm}"
            mesh = UsdGeom.Mesh.Define(layer, mpath)
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in V])
            mesh.CreateFaceVertexCountsAttr([3] * len(TRI))
            idx = []
            c = V.mean(0)
            for (i, jx, k) in TRI:
                n = np.cross(V[jx] - V[i], V[k] - V[i])
                if (V[i] - c) @ n < 0:
                    i, jx, k = i, k, jx
                idx += [i, jx, k]
            mesh.CreateFaceVertexIndicesAttr(idx)
            mesh.CreateExtentAttr([Gf.Vec3f(*V.min(0)), Gf.Vec3f(*V.max(0))])
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.55, 0.1)])
            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            mapi = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mapi.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
    layer.GetRootLayer().Save()

    # ---- STL export (mm) --------------------------------------------------- #
    def write_stl(path, piece_list):
        tris = []
        for V in piece_list:
            c = V.mean(0)
            for (i, jx, k) in TRI:
                n = np.cross(V[jx] - V[i], V[k] - V[i])
                if (V[i] - c) @ n < 0:
                    i, jx, k = i, k, jx
                    n = -n
                ln = np.linalg.norm(n)
                tris.append((n / ln if ln > 0 else n, V[i] * 1000, V[jx] * 1000, V[k] * 1000))
        with open(path, "wb") as f:
            f.write(b"gs1 sleeve (mm)".ljust(80, b"\0"))
            f.write(struct.pack("<I", len(tris)))
            for n, a, b, c3 in tris:
                f.write(struct.pack("<3f", *n))
                for v in (a, b, c3):
                    f.write(struct.pack("<3f", *v))
                f.write(struct.pack("<H", 0))

    write_stl(OUT_STL5, [fixA, fixB])
    write_stl(OUT_STLG, [movA, movB])

    # composed-stage audit: open the authored USD and census the sleeves
    st2 = Usd.Stage.Open(str(OUT_USD))
    pred2 = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    census = {}
    for lk in ("link5", "gripper_link"):
        en = 0
        sleeves = []
        for prim in Usd.PrimRange(st2.GetPrimAtPath(f"/gs1_gripper/{lk}"), pred2):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                a = prim.GetAttribute("physics:collisionEnabled")
                if a.Get() if (a and a.Get() is not None) else True:
                    en += 1
                if "gs1_sleeve" in prim.GetName():
                    sleeves.append(prim.GetPath().pathString)
        census[lk] = {"enabled": en, "sleeve_prims": sleeves}
    D["authored_census"] = census
    if not all(v["enabled"] == 66 and len(v["sleeve_prims"]) == 2 for v in census.values()):
        raise RuntimeError(f"AUTHOR_CENSUS_FAIL {census}")

    D["pieces_vertices"] = {f"{lk}/{nm}": V.tolist()
                            for lk, d2 in pieces.items() for nm, V in d2.items()}
    D["artifacts"] = {p.name: {"sha256_16": sha256(p)[:16], "bytes": p.stat().st_size}
                      for p in (OUT_USD, OUT_STL5, OUT_STLG)}
    D["wall_seconds"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(D, indent=1) + "\n")
    print("[g0f_gs1] GATES kin/contact/design/interference PASS")
    print(f"[g0f_gs1] stock_contact={stock_contact_deg:.3f} deg  "
          f"theta_design={theta_design:.3f} deg  "
          f"sleeved_D29_contact_pred={sleeved_contact_pred:.3f} deg  "
          f"wedge_stock={D['stock_wedge_deg_at_hold']:.2f} deg")
    print(f"[g0f_gs1] mass_total={D['mass_estimate_g']['total']:.2f} g  "
          f"interference_at_close={d_at_close['sleeve_sleeve_min_m']*1000:.2f} mm")
    print(f"[g0f_gs1] artifacts={D['artifacts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
