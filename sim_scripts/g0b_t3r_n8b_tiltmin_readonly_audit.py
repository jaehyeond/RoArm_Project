#!/usr/bin/env python3
"""
g0b_t3r_n8b_tiltmin_readonly_audit.py
    (READ-ONLY on every asset; writes only NEW t3r_n8b_tiltmin_* artifacts)

41st session, second read-only derivation.  Answers ONE decision-relevant number
that t3r_n8_tilt left open.

WHY THIS RUN EXISTS
  t3r_n8_tilt swept (theta, phi) exhaustively but only at THREE q5 anchors, then
  ran the full 34-point q5 sweep at the single best (theta, phi).  That full sweep
  raised the bite at theta=35 from +3.8570 mm (anchors) to +14.9746 mm (q5=25.32),
  which means the anchors UNDER-report every theta.  The crossing theta read off
  the anchor grid (~17 deg) is therefore an over-estimate of the tilt actually
  required, and "how much tilt must we ask the professor for?" is exactly the
  number that decides how large that request is.

CLAIM UNDER TEST (pre-registered)
  "theta_min = the smallest tool-axis tilt for which the assembled gripper admits
   the D29 cylinder to a bite > 0 mm, when the jaw opening q5 is free."
  This is reported as an UPPER BOUND on the true minimum: phi is searched only in
  the +-15 deg neighbourhood of the phi that maximised bite at that theta in the
  n8 anchor grid, not over all azimuths.  A smaller theta_min may exist.

METHOD
  Geometry, frames, sampler and the closed-form admission solve are IMPORTED
  VERBATIM from g0b_t3r_n8_tilt_admission_readonly_audit.py, whose own copy is
  frozen as t3r_n8_tilt_script.py.txt (sha 84ab44dc9d9d87af).  Nothing is
  re-derived here, so every number sits on the same numeric path as D427 / 40th.
  phi candidates per theta come from t3r_n8_tilt_results.json (READ-ONLY).

GATES
  N8Ba  sha pins of link5.stl / gripper_link.stl
  N8Bb  D427 l_vis reproduced (4.457620117187505, n_pts 2266503)
  N8Bc  theta = 0 reproduces 40th: best bite over q5 = -2.1136968498010162 and
        deepest top face = +4.457620117187517, |delta| < 1e-6 mm
  N8Bd  the n8 headline is reproduced: theta=35, phi=347, full q5 -> max bite
        +14.974644662792878, |delta| < 1e-6 mm

NOT IN SCOPE
  No physics, no Isaac, no robot, no Gate-0 re-run or re-judgement (D427/D429
  unchanged), no overwrite of any t3r_n7_* / t3r_n8_tilt_* / t3_* artifact.
  Static admission only: bite > 0 is NECESSARY for closing contact, never
  sufficient, and never a grasp-success prediction.  Tilted-pose IK reachability
  is NOT established here - T2/T2b only ever tested the fully vertical axis.
"""
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
N8_SRC = ROOT / "sim_scripts/g0b_t3r_n8_tilt_admission_readonly_audit.py"
OUT_DIR = ROOT / "claudedocs/runtime_logs/grasp_track/g0b_d420"
N8_RESULTS = OUT_DIR / "t3r_n8_tilt_results.json"
TAG = "t3r_n8b_tiltmin"
LOG = "g0b_t3r_n8b"

RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"
VIEW_STRIDE = 40
THETA_DEG = np.arange(0.0, 35.0 + 1e-9, 1.0)
PHI_NEIGHBOURS_DEG = (-15.0, 0.0, 15.0)
N8_BEST = {"theta_deg": 35.0, "phi_deg": 347.0, "max_bite_mm": 14.974644662792878}
N40_BEST_BITE_MM = -2.1136968498010162
N40_DEPTH_MM = 4.457620117187517
TOL_MM = 1e-6


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_n8_module():
    spec = importlib.util.spec_from_file_location("n8_core", N8_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    if not N8_RESULTS.exists():
        print(f"[{LOG}] ABORT missing_n8_reference={N8_RESULTS}", flush=True)
        return 3

    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        print(f"[{LOG}] ABORT rerun_version={rr.__version__}!={RERUN_VERSION}", flush=True)
        return 3

    m = load_n8_module()
    out = {"tool": TAG, "read_only_assets": True,
           "claim_under_test": ("theta_min = smallest tool-axis tilt admitting the D29 cylinder to "
                                "bite > 0 with the jaw opening q5 free (reported as an UPPER BOUND)"),
           "geometry_imported_verbatim_from": {
               "path": str(N8_SRC.relative_to(ROOT)), "sha256_16": sha256(N8_SRC)[:16],
               "frozen_copy": "t3r_n8_tilt_script.py.txt"},
           "env": {"numpy": np.__version__, "rerun_sdk": rr.__version__,
                   "python": sys.version.split()[0]}}

    # ---- assets + gates N8Ba / N8Bb ---------------------------------------
    out["sha256"] = {p.name: sha256(p) for p in (m.L5, m.GJ)}
    out["sha256_matches_record"] = {k: (out["sha256"][k] == v) for k, v in m.SHA.items()}
    g_a = all(out["sha256_matches_record"].values())

    S5 = m.sample_triangles(m.load_binary_stl(m.L5) * m.STL_SCALE, m.SAMPLE_SPACING_M)
    SG = m.sample_triangles(m.load_binary_stl(m.GJ) * m.STL_SCALE, m.SAMPLE_SPACING_M)
    z5 = S5[:, 2] * 1000.0
    r5 = np.hypot(S5[:, 0], S5[:, 1]) * 1000.0
    l_vis = float((z5 - m.TCP_Z_MM)[r5 <= 30.0].max())
    g_b = (abs(l_vis - m.D427_L_VIS_MM) < 1e-9) and (len(S5) == m.D427_N_PTS)
    out["assets"] = {"link5_samples": int(len(S5)), "gripper_link_samples": int(len(SG)),
                     "l_vis_mm": l_vis, "l_vis_delta_vs_D427": l_vis - m.D427_L_VIS_MM}
    print(f"[{LOG}] gates a={g_a} b={g_b}(l_vis={l_vis:.12f} n={len(S5)})", flush=True)
    if not (g_a and g_b):
        print(f"[{LOG}] ABORT gate_failure_N8Ba_or_N8Bb", flush=True)
        return 3

    u = m.parse_urdf(m.URDF)
    j5 = u["joints"]["link5_to_gripper_link"]
    R_j5 = m.rpy_matrix(*j5["rpy"])
    t_j5 = np.array(j5["xyz"])

    def to_link5(P, q5):
        c, s = math.cos(q5), math.sin(q5)
        Rq = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return P @ (R_j5 @ Rq).T + t_j5

    c0 = np.array([0.0, 0.0, m.TCP_Z_MM / 1000.0])
    R_m = m.CYL_R_MM / 1000.0
    SPAN = m.WALL_SPAN_MM / 1000.0
    q5_full = np.unique(np.concatenate([np.linspace(0.0, j5["limit"][1], 33), [m.Q5_REAL_MAX_RAD]]))
    mov = {float(q): to_link5(SG, float(q)) for q in q5_full}
    print(f"[{LOG}] q5 grid n={len(q5_full)} moving-jaw poses cached", flush=True)

    def scan_pair(th_deg, ph_deg):
        """full q5 sweep at one (theta, phi) -> best row over q5."""
        chat = m.axis_dir(math.radians(th_deg), math.radians(ph_deg))
        pf = m.prep(S5, chat, c0)
        d_f, _ = m.deepest_delta(*pf, chat[2], R_m)
        rows = []
        for q, P in mov.items():
            pm = m.prep(P, chat, c0)
            d_m, _ = m.deepest_delta(*pm, chat[2], R_m)
            delta = max(d_f, d_m)
            if not math.isfinite(delta):
                continue
            bf, _ = m.bite_at(*pf, chat[2], R_m, SPAN, delta)
            bm, _ = m.bite_at(*pm, chat[2], R_m, SPAN, delta)
            bb = max(bf, bm)
            if not math.isfinite(bb):
                continue
            rows.append({"q5_deg": math.degrees(q), "bite_mm": bb * 1000.0,
                         "bite_fixed_mm": bf * 1000.0 if math.isfinite(bf) else None,
                         "bite_moving_mm": bm * 1000.0 if math.isfinite(bm) else None,
                         "depth_top_min_mm": delta * 1000.0, "delta_m": delta,
                         "blocker": "fixed" if d_f >= d_m else "moving"})
        if not rows:
            return None
        best = max(rows, key=lambda r: r["bite_mm"])
        best.update({"theta_deg": float(th_deg), "phi_deg": float(ph_deg),
                     "chat": chat.tolist(),
                     "q5_window_positive_deg": [r["q5_deg"] for r in rows if r["bite_mm"] > 0.0]})
        return best

    # phi candidates per theta: the anchor-grid argmax phi (n8) +- 15 deg
    n8 = json.loads(N8_RESULTS.read_text())
    anchor_best_phi, anchor_bite = {}, {}
    for g in n8["grid_theta_phi_q5"]:
        if g["bite_mm"] is None:
            continue
        t = g["theta_deg"]
        if t not in anchor_bite or g["bite_mm"] > anchor_bite[t]:
            anchor_bite[t], anchor_best_phi[t] = g["bite_mm"], g["phi_deg"]

    ladder, t0 = [], time.time()
    for i, th in enumerate(THETA_DEG):
        ph0 = anchor_best_phi.get(float(th), 0.0)
        cands = sorted({round((ph0 + d) % 360.0, 6) for d in PHI_NEIGHBOURS_DEG})
        rows = [r for r in (scan_pair(float(th), p) for p in cands) if r is not None]
        best = max(rows, key=lambda r: r["bite_mm"])
        best["phi_candidates_deg"] = cands
        best["anchor_only_bite_mm"] = anchor_bite.get(float(th))
        ladder.append(best)
        print(f"[{LOG}] theta={th:5.1f} phi*={best['phi_deg']:6.1f} q5*={best['q5_deg']:6.2f} "
              f"bite={best['bite_mm']:+9.4f}mm (anchors said {best['anchor_only_bite_mm']:+8.4f}) "
              f"depth={best['depth_top_min_mm']:+8.4f} blocker={best['blocker']} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    out["theta_ladder_full_q5"] = ladder

    # ---- gates N8Bc / N8Bd ------------------------------------------------
    th0 = next(r for r in ladder if r["theta_deg"] == 0.0)
    g_c = (abs(th0["bite_mm"] - N40_BEST_BITE_MM) < TOL_MM and
           abs(th0["depth_top_min_mm"] - N40_DEPTH_MM) < TOL_MM)
    out["N8Bc_theta0_reproduces_40th"] = {
        "pass": bool(g_c), "bite_mm": th0["bite_mm"], "ref_bite_mm": N40_BEST_BITE_MM,
        "d_bite_mm": th0["bite_mm"] - N40_BEST_BITE_MM,
        "depth_top_min_mm": th0["depth_top_min_mm"], "ref_depth_mm": N40_DEPTH_MM,
        "d_depth_mm": th0["depth_top_min_mm"] - N40_DEPTH_MM, "tol_mm": TOL_MM}
    chk = scan_pair(N8_BEST["theta_deg"], N8_BEST["phi_deg"])
    g_d = abs(chk["bite_mm"] - N8_BEST["max_bite_mm"]) < TOL_MM
    out["N8Bd_reproduces_n8_headline"] = {
        "pass": bool(g_d), "bite_mm": chk["bite_mm"], "ref_bite_mm": N8_BEST["max_bite_mm"],
        "d_bite_mm": chk["bite_mm"] - N8_BEST["max_bite_mm"], "q5_deg": chk["q5_deg"],
        "tol_mm": TOL_MM}
    print(f"[{LOG}] N8Bc theta0_vs_40th pass={g_c} d_bite={out['N8Bc_theta0_reproduces_40th']['d_bite_mm']:.3e} "
          f"d_depth={out['N8Bc_theta0_reproduces_40th']['d_depth_mm']:.3e}", flush=True)
    print(f"[{LOG}] N8Bd n8_headline pass={g_d} "
          f"d_bite={out['N8Bd_reproduces_n8_headline']['d_bite_mm']:.3e}", flush=True)
    if not (g_c and g_d):
        print(f"[{LOG}] ABORT reproduction_gate_failure_N8Bc_or_N8Bd", flush=True)
        return 3

    # ---- the answer -------------------------------------------------------
    pos = [r for r in ladder if r["theta_deg"] > 0.0 and r["bite_mm"] > 0.0]
    theta_min = min((r["theta_deg"] for r in pos), default=None)
    row_min = next((r for r in ladder if r["theta_deg"] == theta_min), None) if pos else None
    anchor_pos = [t for t, b in anchor_bite.items() if t > 0.0 and b > 0.0]
    out["verdict"] = {
        "code": "TILT_MIN_BOUNDED" if theta_min is not None else "NO_POSITIVE_BITE_WITHIN_T1_RANGE",
        "theta_min_deg_upper_bound": theta_min,
        "theta_min_is_upper_bound_because": ("phi searched only within +-15 deg of the anchor-grid "
                                            "argmax; a smaller theta_min may exist at another azimuth"),
        "at_theta_min": ({k: row_min[k] for k in ("theta_deg", "phi_deg", "q5_deg", "bite_mm",
                                                  "depth_top_min_mm", "blocker")}
                         if row_min else None),
        "theta_min_deg_from_n8_anchor_grid_only": min(anchor_pos) if anchor_pos else None,
        "max_bite_mm_over_ladder": max(r["bite_mm"] for r in ladder),
        "T1_measured_tilt_deg": list(m.T1_TILT_DEG),
        "T1_photo_bite_mm": list(m.T1_BITE_MM),
        "D427_D429_status": "UNCHANGED - this run neither re-runs nor re-judges Gate-0",
        "tilted_pose_ik_reachability": "NOT ESTABLISHED - T2/T2b tested the vertical axis only",
        "g0a_pass": False,
    }
    out["gates"] = {"N8Ba_sha_pins": bool(g_a), "N8Bb_D427_l_vis_reproduced": bool(g_b),
                    "N8Bc_theta0_reproduces_40th": bool(g_c),
                    "N8Bd_reproduces_n8_headline": bool(g_d),
                    "N8Be_positive_bite_within_T1_range": bool(theta_min is not None)}
    print(f"[{LOG}] G0B_T3R_N8B_VERDICT={out['verdict']['code']} "
          f"theta_min<={theta_min} deg (anchor grid said "
          f"{out['verdict']['theta_min_deg_from_n8_anchor_grid_only']})", flush=True)

    # ---- D324 diagnostic --------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(19.5, 6.0))
    ths = [r["theta_deg"] for r in ladder]

    ax[0].plot(ths, [r["bite_mm"] for r in ladder], "-o", ms=3.5, color="#c02020",
               label="bite with q5 FREE (this run)")
    ax[0].plot(ths, [r["anchor_only_bite_mm"] for r in ladder], "-s", ms=3, color="#8a8a8a",
               label="bite from n8's 3 q5 anchors only")
    ax[0].axhspan(*m.T1_BITE_MM, color="#2a9d3a", alpha=0.16, label="T1 real photo bite 0-12mm")
    ax[0].axhline(0.0, color="k", lw=1.0)
    if theta_min is not None:
        ax[0].axvline(theta_min, color="#7a3fb5", ls="--", lw=1.6,
                      label=f"theta_min <= {theta_min:.0f} deg")
    ax[0].set_xlabel("tool-axis tilt theta [deg]   (T1 measured 0-35, D420 (2))")
    ax[0].set_ylabel("bite below cylinder top face [mm]")
    ax[0].set_title("A | freeing q5 lowers the tilt actually required")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for r in ladder:
        w = r["q5_window_positive_deg"]
        if w:
            ax[1].plot([r["theta_deg"]] * 2, [min(w), max(w)], "-", lw=3.0, color="#2a9d3a",
                       solid_capstyle="butt")
    ax[1].plot(ths, [r["q5_deg"] for r in ladder], "o", ms=3.5, color="#c02020",
               label="q5 at max bite")
    ax[1].axhline(24.0, color="#1f6fb4", ls="--", lw=1.2,
                  label="T3 attempt2/3 closed down to 24 deg (D424)")
    ax[1].axhline(88.31, color="#e08a1e", ls=":", lw=1.2, label="real max opening 88.31 deg")
    ax[1].set_xlabel("tool-axis tilt theta [deg]")
    ax[1].set_ylabel("jaw opening q5 [deg]")
    ax[1].set_title("B | green = q5 band with POSITIVE bite at each tilt")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].plot(ths, [r["depth_top_min_mm"] for r in ladder], "-o", ms=3.5, color="#1f6fb4",
               label="deepest reachable top-face plane at the bite-max config")
    ax[2].axhline(4.4, color="#e08a1e", ls="--", lw=1.4,
                  label="attempt1 physical stop top+4.4mm (D424)")
    ax[2].axhline(0.0, color="k", lw=1.0)
    ax[2].set_xlabel("tool-axis tilt theta [deg]")
    ax[2].set_ylabel("depth = z - TCP [mm]   (<0 = object enters the throat)")
    ax[2].set_title("C | descent limit along the same ladder")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.suptitle(f"g0b_d420 {TAG} - minimum required approach tilt (read-only).  "
                 f"VERDICT: {out['verdict']['code']}  theta_min <= {theta_min} deg", fontsize=12)
    fig.tight_layout()
    fig.savefig(paths["diagnostic.png"], dpi=125)
    plt.close(fig)

    # ---- D341 Rerun -------------------------------------------------------
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact
    app_id = f"roarm_g0b_{TAG}"

    def view_pts(pts_m):
        o = np.asarray(pts_m) - m.TCP_LOCAL[None, :]
        return np.column_stack([o[:, 0], o[:, 1], -o[:, 2]])

    def perp_basis(c):
        a = np.array([1.0, 0.0, 0.0]) if abs(c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(c, a)
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(c, e1)

    def cyl_wire(chat, delta_m, R_mm, H_mm):
        cc = np.array([0.0, 0.0, m.TCP_Z_MM / 1000.0 + delta_m])
        e1, e2 = perp_basis(chat)
        rr_m = R_mm / 1000.0
        rings, walls = [], []
        for k in (0.0, 5.0, 12.0, 25.0, H_mm):
            base = cc + chat * (k / 1000.0)
            ang = np.linspace(0.0, 2 * math.pi, 65)
            rings.append(view_pts(base[None, :] + rr_m * (np.cos(ang)[:, None] * e1[None, :] +
                                                          np.sin(ang)[:, None] * e2[None, :])).tolist())
        for a in np.linspace(0.0, 2 * math.pi, 24, endpoint=False):
            off = rr_m * (math.cos(a) * e1 + math.sin(a) * e2)
            walls.append(view_pts(np.array([cc + off, cc + chat * (H_mm / 1000.0) + off])).tolist())
        return rings, walls

    def topface_ring(chat, delta_m, R_mm):
        cc = np.array([0.0, 0.0, m.TCP_Z_MM / 1000.0 + delta_m])
        e1, e2 = perp_basis(chat)
        ang = np.linspace(0.0, 2 * math.pi, 97)
        rr_m = R_mm / 1000.0
        return view_pts(cc[None, :] + rr_m * (np.cos(ang)[:, None] * e1[None, :] +
                                              np.sin(ang)[:, None] * e2[None, :])).tolist()

    with rr.RecordingStream(app_id, recording_id=f"g0b_d420_{TAG}", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(paths["timeline.rrd"]), write_footer=True)
        rec.log("assembly/link5_fixed_jaw", rr.Points3D(view_pts(S5[::VIEW_STRIDE]),
                colors=[150, 150, 160], radii=0.0004), static=True)
        rec.log("assembly/tcp", rr.Points3D([[0.0, 0.0, 0.0]], colors=[[40, 200, 80]],
                radii=0.002), static=True)
        rec.log("assembly/tool_axis", rr.LineStrips3D([[[0.0, 0.0, 0.07], [0.0, 0.0, -0.07]]],
                colors=[[40, 200, 80]], radii=0.0003), static=True)
        rec.log("reference/d427_blocker_peak", rr.Points3D(
                [[-0.010025849, 0.001408991, -(119.885620 - m.TCP_Z_MM) / 1000.0]],
                colors=[[255, 210, 40]], radii=0.0016), static=True)
        for i, r in enumerate(ladder):
            chat = np.array(r["chat"])
            rec.reset_time()
            rec.set_time("theta_index", sequence=i)
            # the moving jaw is logged AT THIS THETA'S OWN argmax q5, so every timeline
            # step is a self-consistent (jaw pose, cylinder pose) pair.
            rec.log("assembly/gripper_link_moving_jaw", rr.Points3D(
                view_pts(to_link5(SG, math.radians(r["q5_deg"]))[::VIEW_STRIDE]),
                colors=[70, 130, 230], radii=0.0004))
            rings, walls = cyl_wire(chat, r["delta_m"], m.CYL_R_MM, m.CYL_H_MM)
            rec.log("object/cylinder_tilted", rr.LineStrips3D(
                rings + walls, colors=[[225, 60, 60]] * (len(rings) + len(walls)), radii=0.00035))
            rec.log("object/topface_plane_tilted", rr.LineStrips3D(
                [topface_ring(chat, r["delta_m"], 30.0)], colors=[[235, 140, 30]], radii=0.0005))
            rec.log("plots/theta_deg", rr.Scalars(float(r["theta_deg"])))
            rec.log("plots/bite_mm", rr.Scalars(float(r["bite_mm"])))
            rec.log("plots/bite_anchors_only_mm", rr.Scalars(float(r["anchor_only_bite_mm"])))
            rec.log("plots/q5_at_max_bite_deg", rr.Scalars(float(r["q5_deg"])))
            rec.log("plots/depth_top_min_mm", rr.Scalars(float(r["depth_top_min_mm"])))
        rec.reset_time()
        rec.set_time("theta_index", sequence=0)
        for name, ok in (("N8Ba_sha_pins", g_a), ("N8Bb_D427_l_vis_reproduced", g_b),
                         ("N8Bc_theta0_reproduces_40th", g_c),
                         ("N8Bd_reproduces_n8_headline", g_d),
                         ("N8Be_positive_bite_within_T1_range", theta_min is not None)):
            rec.log("events/gates", rr.TextLog(name, level=rr.TextLogLevel.INFO if ok
                                               else rr.TextLogLevel.ERROR))
        summary_md = (
            f"# g0b_d420 {TAG} - minimum required approach tilt (read-only)\n\n"
            f"**VERDICT: {out['verdict']['code']} - theta_min <= {theta_min} deg**\n\n"
            f"## Question\n`t3r_n8_tilt` swept (theta, phi) exhaustively but only at 3 q5 anchors, "
            f"and freeing q5 at the best (theta, phi) raised the bite from +3.8570 to "
            f"**+14.9746 mm**. So the anchors under-report every theta and the ~17 deg crossing "
            f"read off that grid over-states the tilt actually required. How much tilt must we "
            f"ask for?\n\n"
            f"## Answer\n"
            f"- **theta_min <= {theta_min} deg** (upper bound: phi searched only within +-15 deg "
            f"of the anchor argmax);\n"
            f"- the n8 anchor grid said "
            f"{out['verdict']['theta_min_deg_from_n8_anchor_grid_only']} deg;\n"
            f"- at theta_min: phi = **{row_min['phi_deg']:.0f} deg**, q5 = "
            f"**{row_min['q5_deg']:.2f} deg**, bite = **{row_min['bite_mm']:+.4f} mm**, deepest "
            f"top face = TCP**{row_min['depth_top_min_mm']:+.4f} mm**, blocker = "
            f"**{row_min['blocker']}**;\n"
            f"- max bite over the whole ladder: **{out['verdict']['max_bite_mm_over_ladder']:+.4f} "
            f"mm** vs T1's photo band **0-12 mm**;\n"
            f"- T1's own measured tilt range is **0-35 deg** (D420 Impl.(2)).\n\n"
            f"## Reproduction gates\n"
            f"- N8Bc: theta = 0 reproduces 40th - bite delta "
            f"**{out['N8Bc_theta0_reproduces_40th']['d_bite_mm']:.3e} mm**, depth delta "
            f"**{out['N8Bc_theta0_reproduces_40th']['d_depth_mm']:.3e} mm**;\n"
            f"- N8Bd: the n8 headline (+14.974644662792878 mm) reproduces to "
            f"**{out['N8Bd_reproduces_n8_headline']['d_bite_mm']:.3e} mm**;\n"
            f"- geometry imported verbatim from `{N8_SRC.name}` "
            f"(sha `{sha256(N8_SRC)[:16]}`), frozen copy `t3r_n8_tilt_script.py.txt`.\n\n"
            f"## Limits\nStatic admission only - bite > 0 is NECESSARY for closing contact, never "
            f"sufficient. **Tilted-pose IK reachability is NOT established** (T2/T2b tested the "
            f"vertical axis only). The table is not modelled. D427/D429 unchanged, "
            f"`g0a_pass=false`.\n\n"
            f"## Scene\ngrey = link5 (fixed jaw), blue = moving jaw at THIS step's own argmax q5, "
            f"red = the D29 cylinder at its deepest admissible height for this step's tilt, orange "
            f"= its top-face plane, yellow = the D427 blocking peak, green = TCP and tool axis. "
            f"z_view = -(z - z_TCP), so DOWN = distal.\n\n"
            f"Authority = stdout + `{paths['results.json'].name}`. Rerun is inspection evidence "
            f"only (D341).\n")
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN),
                static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run",
                                         name="1 | minimum tilt verdict"),
                    rrb.Spatial3DView(origin="/", contents=["/assembly/**", "/object/**",
                                                            "/reference/**"],
                                      name="2 | tilt ladder: jaw + cylinder per step"),
                    rrb.TextLogView(origin="/events/gates", contents="/events/gates/**",
                                    name="3 | gates"),
                    column_shares=[0.30, 0.46, 0.24],
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/bite_mm/**",
                                                 "/plots/bite_anchors_only_mm/**",
                                                 "/plots/depth_top_min_mm/**"],
                                       name="4 | bite: q5 free vs q5 anchors"),
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/theta_deg/**",
                                                 "/plots/q5_at_max_bite_deg/**"],
                                       name="5 | tilt theta and the q5 that achieves it"),
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
                         "object/topface_plane_tilted", "plots/theta_deg", "plots/bite_mm",
                         "plots/bite_anchors_only_mm", "plots/q5_at_max_bite_deg",
                         "plots/depth_top_min_mm", "events/gates"]
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    lin3 = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    components = {
        "metadata/run": ["TextDocument:text"],
        "assembly/link5_fixed_jaw": pts3, "assembly/gripper_link_moving_jaw": pts3,
        "assembly/tcp": pts3, "reference/d427_blocker_peak": pts3,
        "assembly/tool_axis": lin3, "object/cylinder_tilted": lin3,
        "object/topface_plane_tilted": lin3,
        "plots/theta_deg": ["Scalars:scalars"], "plots/bite_mm": ["Scalars:scalars"],
        "plots/bite_anchors_only_mm": ["Scalars:scalars"],
        "plots/q5_at_max_bite_deg": ["Scalars:scalars"],
        "plots/depth_top_min_mm": ["Scalars:scalars"],
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
    out["artifacts"] = {k: {"name": v.name, "sha256": sha256(v)[:16], "bytes": v.stat().st_size}
                        for k, v in paths.items() if v.exists() and k != "results.json"}
    out["artifacts_note"] = ("results.json is deliberately absent from this manifest - D429-R1. "
                             "Hash it from disk.")
    out["rerun_validation_pass"] = bool(validation.get("pass"))
    out["wall_seconds"] = round(time.time() - t_start, 1)
    paths["results.json"].write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{LOG}] artifacts " + " ".join(f"{v['name']}={v['sha256']}"
                                           for v in out["artifacts"].values()), flush=True)
    print(f"[{LOG}] results.json={sha256(paths['results.json'])[:16]} "
          f"bytes={paths['results.json'].stat().st_size}", flush=True)
    print(f"[{LOG}] G0B_T3R_N8B_VERDICT={out['verdict']['code']} theta_min<={theta_min}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
