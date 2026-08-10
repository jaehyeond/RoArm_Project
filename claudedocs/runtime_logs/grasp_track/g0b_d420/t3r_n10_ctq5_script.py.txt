#!/usr/bin/env python3
"""
g0b_t3r_n10_collision_asset_tilt_and_q5_refine_readonly_audit.py
    (READ-ONLY on every asset; writes only NEW t3r_n10_ctq5_* artifacts)

43rd session, single read-only derivation.  The professor's approval of an
approach TILT (relayed by the user) closed the last open decision on this track,
so the next step is T3 physics.  This run is the PRE-FLIGHT that T3 physics
needs, and it answers the two questions that are still open in writing.

WHY THIS RUN EXISTS  (both halves can fail, and either failure changes the plan)

  A) 41st 5-2 left the closing target UNDECIDED.  The positive-bite q5 window
     was measured on a 2.81 deg grid whose top rung is 22.50 deg, the next rung
     is 25.32 deg, and T3 attempt2/3 closed to exactly 24 deg (D424 (2)) - i.e.
     inside the undetermined gap.  A 0.1 deg sweep decides whether T3 missed the
     window, and hands T3 an exact closing target instead of a range.

  B) EVERY tilt number on this track (D431, D432) was measured on the VISUAL
     mesh.  T3 physics does not consume the visual mesh: it consumes the frozen
     attempt3 collision USD (prereg supersession S-2).  D425 found that asset's
     jaws occluded, and D427 found the visual SOURCE itself lacks distal jaw
     geometry.  So "tilt produces a positive bite" has never been checked on the
     object the pipeline actually consumes.  This is D428 #29 exactly.  If the
     collision asset does not admit a positive bite under tilt, running T3
     physics would burn an Isaac run and return a misleading "tilt fails".

CLAIMS UNDER TEST (pre-registered, both falsifiable)
  A) "On the visual mesh, at 0.1 deg q5 resolution, the positive-bite window has
      a definite upper edge, and q5 = 24.00 deg either is or is not inside it."
  B) "The frozen attempt3 collision asset - the geometry T3 physics actually
      loads - admits the D29 cylinder to a bite > 0 mm at some tilt in the
      D432 approach band theta in [15, 29] deg."
  B fails  -> COLLISION_ASSET_BLOCKS_TILTED_BITE -> do NOT start T3 physics;
              the blocker is the asset, and that is a D426 (1) matter again.
  B passes -> COLLISION_ASSET_ADMITS_TILTED_BITE -> T3 physics is grounded and
              gets an exact (theta, q5) target measured on its own geometry.

METHOD
  The admission engine (axis_dir / prep / deepest_delta / bite_at), the mesh
  loader, the sampler, the URDF parser and every constant are IMPORTED VERBATIM
  from g0b_t3r_n8_tilt_admission_readonly_audit.py (frozen copy
  t3r_n8_tilt_script.py.txt, sha 84ab44dc9d9d87af), so every number here sits on
  the same numeric path as D427 / 40th / D431.  The collision-asset reader
  (extract_asset / gripper_T_l5 / hull_samples) is IMPORTED VERBATIM from the
  frozen g0b_t3_attempt3_jaw_throat_occlusion_readonly_vertex_audit.py that
  produced D425.  Nothing is re-derived, nothing is re-decomposed, no USD layer
  is edited - Usd.Stage.Open only.  phi* per theta comes from
  t3r_n8b_tiltmin_results.json (READ-ONLY).

GATES
  N10a  sha pins: link5.stl, gripper_link.stl, attempt3 root USD, physics layer
  N10b  D427 l_vis reproduced (4.457620117187505 mm, n_pts 2266503)
  N10c  theta = 0, coarse 34-point q5 grid, visual -> 40th's -2.1136968498010162
        mm bite and +4.457620117187517 mm deepest top face
  N10d  theta = 35, phi = 347, coarse grid -> n8 headline +14.974644662792878 mm
  N10e  the whole 41st theta ladder reproduces on the coarse grid at its own phi*
  N10f  URDF joint frame == USD joint frame for the moving jaw (sign + origin)
  N10g  collision fixed-jaw distal peak reproduces the visual l_vis (frame check)
  N10h  asset identity: 64 + 64 enabled convexHull parts, every legacy collider
        on these two bodies disabled, no non-convexHull approximation

NOT IN SCOPE
  No physics, no Isaac launch, no robot, no Gate-0 re-run or re-judgement
  (D427 / D429 / D430 / D431 / D432 unchanged), no overwrite of any t3_* or
  t3r_* artifact, no modification of any frozen source, no re-decomposition
  (D415 (3) / D426 untouched).  Static admission only: bite > 0 is NECESSARY
  for closing contact, never sufficient, and never a grasp-success prediction.
  Force closure remains unproven.  g0a_pass stays false.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
USD_LIBS = (
    Path("/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages")
    / "isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
# Deliberately the SAME flag name the frozen jaw audit uses, so importing that
# module below is a no-op instead of a second re-exec.
_REEXEC_FLAG = "G0B_JAW_AUDIT_REEXEC"
LOG = "g0b_t3r_n10"


def _bootstrap_pxr_env() -> None:
    """Re-exec once with pxr module + shared-library paths (LD_LIBRARY_PATH is
    read by the dynamic linker at process start, so in-process assignment is not
    reliable).  Kit is NOT launched - this stays a plain-python process."""
    if os.environ.get(_REEXEC_FLAG) == "1":
        return
    if not USD_LIBS.is_dir():
        print(f"[{LOG}] ABORT missing_usd_libs {USD_LIBS}", flush=True)
        raise SystemExit(3)
    conda_lib = str(Path(sys.executable).resolve().parents[1] / "lib")
    env = dict(os.environ)
    env[_REEXEC_FLAG] = "1"
    env["PYTHONPATH"] = str(USD_LIBS) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    extra = f"{USD_LIBS / 'bin'}:{conda_lib}"
    env["LD_LIBRARY_PATH"] = extra + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


_bootstrap_pxr_env()

import hashlib          # noqa: E402
import importlib.util   # noqa: E402
import json             # noqa: E402
import math             # noqa: E402
import shutil           # noqa: E402
import time             # noqa: E402

import numpy as np      # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim_scripts"))

N8_SRC = ROOT / "sim_scripts/g0b_t3r_n8_tilt_admission_readonly_audit.py"
JA_SRC = ROOT / "sim_scripts/g0b_t3_attempt3_jaw_throat_occlusion_readonly_vertex_audit.py"
OUT_DIR = ROOT / "claudedocs/runtime_logs/grasp_track/g0b_d420"
N8B_RESULTS = OUT_DIR / "t3r_n8b_tiltmin_results.json"
TAG = "t3r_n10_ctq5"

RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"
VIEW_STRIDE_VIS = 40
VIEW_STRIDE_COL = 16

# theta rungs: the 41st decision ladder intersected with the D432 IK ladder
THETA_SET = (0.0, 6.0, 10.0, 15.0, 17.0, 20.0, 24.0, 29.0, 35.0)
T3_APPROACH_BAND_DEG = (15.0, 29.0)     # D432 Implication (3)
Q5_FINE_STEP_DEG = 0.1                  # 28x finer than the 2.81 deg grid of 41st
T3_ATTEMPTED_CLOSE_DEG = 24.0           # what attempt2/3 actually commanded (D424 (2))

# frozen reference numbers (all re-derived nowhere - quoted for gates only)
N40_BEST_BITE_MM = -2.1136968498010162
N40_DEPTH_MM = 4.457620117187517
N8_BEST = {"theta_deg": 35.0, "phi_deg": 347.0, "max_bite_mm": 14.974644662792878}
TOL_MM = 1e-6
TOL_JOINT = 1e-3          # URDF rpy is truncated to 4 decimals (40th: 3.673e-06 rad)
TOL_PEAK_MM = 0.05        # hull-sampled collision vs triangle-sampled visual


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def contiguous_windows(vals, flags, step):
    """[(lo, hi)] of maximal runs where flags is True, on a uniform grid."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((vals[start], vals[i - 1]))
            start = None
    if start is not None:
        out.append((vals[start], vals[-1]))
    return [{"q5_lo_deg": float(a), "q5_hi_deg": float(b),
             "grid_step_deg": step, "edge_uncertainty_deg": step} for a, b in out]


# =========================================================================== #
def main() -> int:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {k: OUT_DIR / f"{TAG}_{k}" for k in
             ("results.json", "timeline.rrd", "timeline.rbl", "rerun_validation.json",
              "inspection.png", "diagnostic.png", "script.py.txt", "curves.csv")}
    existing = [p.name for p in paths.values() if p.exists()]
    if existing:
        print(f"[{LOG}] ABORT write_guard existing={existing}", flush=True)
        return 3
    if not N8B_RESULTS.exists():
        print(f"[{LOG}] ABORT missing_n8b_reference={N8B_RESULTS}", flush=True)
        return 3

    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        print(f"[{LOG}] ABORT rerun_version={rr.__version__}!={RERUN_VERSION}", flush=True)
        return 3

    m = load_module("n8_core", N8_SRC)
    ja = load_module("jaw_audit_core", JA_SRC)

    out = {
        "tool": TAG,
        "session": "43rd",
        "read_only_assets": True,
        "isaac_launched": False,
        "claims_under_test": {
            "A_q5_refine": ("at 0.1 deg q5 resolution the positive-bite window has a definite "
                            "upper edge, and q5 = 24.00 deg (what T3 attempt2/3 commanded) either "
                            "is or is not inside it"),
            "B_collision_asset": ("the frozen attempt3 collision asset - the geometry T3 physics "
                                  "actually loads - admits the D29 cylinder to bite > 0 mm at some "
                                  "tilt in the D432 band theta in [15, 29] deg")},
        "imported_verbatim_from": {
            "admission_engine": {"path": str(N8_SRC.relative_to(ROOT)),
                                 "sha256_16": sha256(N8_SRC)[:16],
                                 "frozen_copy": "t3r_n8_tilt_script.py.txt"},
            "collision_asset_reader": {"path": str(JA_SRC.relative_to(ROOT)),
                                       "sha256_16": sha256(JA_SRC)[:16],
                                       "produced": "D425"}},
        "env": {"numpy": np.__version__, "rerun_sdk": rr.__version__,
                "python": sys.version.split()[0]},
    }

    # ---- gate N10a: sha pins ----------------------------------------------
    out["sha256"] = {p.name: sha256(p) for p in (m.L5, m.GJ)}
    out["sha256"][ja.ATTEMPT3_USD.name + "(root)"] = sha256(ja.ATTEMPT3_USD)
    out["sha256"][ja.ATTEMPT3_PHYSICS_LAYER.name] = sha256(ja.ATTEMPT3_PHYSICS_LAYER)
    mesh_ok = all(out["sha256"][k] == v for k, v in m.SHA.items())
    usd_ok = (out["sha256"][ja.ATTEMPT3_USD.name + "(root)"] == ja.ATTEMPT3_ROOT_SHA256 and
              out["sha256"][ja.ATTEMPT3_PHYSICS_LAYER.name] == ja.ATTEMPT3_PHYSICS_SHA256)
    g_a = bool(mesh_ok and usd_ok)
    print(f"[{LOG}] N10a sha mesh={mesh_ok} usd={usd_ok}", flush=True)

    # ---- visual clouds + gate N10b ----------------------------------------
    S5 = m.sample_triangles(m.load_binary_stl(m.L5) * m.STL_SCALE, m.SAMPLE_SPACING_M)
    SG = m.sample_triangles(m.load_binary_stl(m.GJ) * m.STL_SCALE, m.SAMPLE_SPACING_M)
    z5 = S5[:, 2] * 1000.0
    r5 = np.hypot(S5[:, 0], S5[:, 1]) * 1000.0
    l_vis = float((z5 - m.TCP_Z_MM)[r5 <= 30.0].max())
    g_b = bool(abs(l_vis - m.D427_L_VIS_MM) < 1e-9 and len(S5) == m.D427_N_PTS)
    print(f"[{LOG}] N10b l_vis={l_vis:.12f} n={len(S5)} pass={g_b}", flush=True)
    if not (g_a and g_b):
        print(f"[{LOG}] ABORT gate_failure_N10a_or_N10b", flush=True)
        return 3

    # ---- URDF joint (visual path, verbatim from 41st) ---------------------
    u = m.parse_urdf(m.URDF)
    j5 = u["joints"]["link5_to_gripper_link"]
    R_j5 = m.rpy_matrix(*j5["rpy"])
    t_j5 = np.array(j5["xyz"])

    def T_urdf(q5_deg):
        c, s = math.cos(math.radians(q5_deg)), math.sin(math.radians(q5_deg))
        Rq = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = R_j5 @ Rq
        T[:3, 3] = t_j5
        return T

    def mov_visual(q5_deg):
        T = T_urdf(q5_deg)
        return SG @ T[:3, :3].T + T[:3, 3]

    # ---- collision asset (READ-ONLY Usd.Stage.Open) + gates N10f/N10g/N10h -
    asset = ja.extract_asset()
    bodies = asset["bodies"]
    joint = asset["joint"]
    C5 = np.vstack([p["samples"] for p in bodies["link5"]["parts"]])
    CGb = np.vstack([p["samples"] for p in bodies["gripper_link"]["parts"]])

    def mov_collision(q5_deg):
        T = ja.gripper_T_l5(joint, q5_deg)
        return CGb @ T[:3, :3].T + T[:3, 3]

    joint_dev = {}
    for q in (0.0, T3_ATTEMPTED_CLOSE_DEG, 45.0, math.degrees(m.Q5_REAL_MAX_RAD)):
        d = float(np.abs(T_urdf(q) - ja.gripper_T_l5(joint, q)).max())
        joint_dev[f"q5={q:.2f}"] = d
    g_f = bool(max(joint_dev.values()) < TOL_JOINT)
    out["N10f_urdf_joint_equals_usd_joint"] = {
        "pass": g_f, "max_abs_elementwise": joint_dev, "tol": TOL_JOINT,
        "meaning": ("same origin AND same rotation sign, so the moving-jaw pose used on the "
                    "visual mesh and on the collision asset is the same mechanism"),
        "authored_theta0_residual": joint["authored_theta0_residual"],
        "usd_limits_deg": [joint["lower_deg"], joint["upper_deg"]],
        "urdf_limit_upper_deg": math.degrees(j5["limit"][1])}

    zc = C5[:, 2] * 1000.0
    rc = np.hypot(C5[:, 0], C5[:, 1]) * 1000.0
    l_col = float((zc - m.TCP_Z_MM)[rc <= 30.0].max())
    g_g = bool(abs(l_col - l_vis) < TOL_PEAK_MM)
    out["N10g_collision_fixed_peak_vs_visual"] = {
        "pass": g_g, "collision_l_mm": l_col, "visual_l_vis_mm": l_vis,
        "delta_mm": l_col - l_vis, "tol_mm": TOL_PEAK_MM,
        "meaning": "frame check: the two assets put the fixed-jaw distal peak in the same place"}

    legacy = {k: bodies[k]["legacy"] for k in ("link5", "gripper_link")}
    legacy_all_disabled = all(not en for v in legacy.values() for _, en in v)
    approx_bad = {k: bodies[k]["approx_bad"] for k in ("link5", "gripper_link")}
    counts = {k: len(bodies[k]["parts"]) for k in ("link5", "gripper_link")}
    g_h = bool(counts["link5"] == ja.EXPECTED_PART_COUNT and
               counts["gripper_link"] == ja.EXPECTED_PART_COUNT and
               legacy_all_disabled and not any(approx_bad.values()))
    out["N10h_asset_identity"] = {
        "pass": g_h, "enabled_convexhull_parts": counts,
        "expected_each": ja.EXPECTED_PART_COUNT,
        "legacy_colliders": {k: [[p, bool(e)] for p, e in v] for k, v in legacy.items()},
        "legacy_all_disabled": bool(legacy_all_disabled),
        "non_convexhull_approximation": approx_bad,
        "note": ("legacy node_STL_BINARY_ colliders exist on both jaw bodies but are "
                 "collisionEnabled=False, so excluding them is what physics does too "
                 "(D425 (3) concerned world/link1..4)")}
    out["assets"] = {
        "visual_link5_samples": int(len(S5)), "visual_gripper_samples": int(len(SG)),
        "collision_link5_samples": int(len(C5)), "collision_gripper_samples": int(len(CGb)),
        "l_vis_mm": l_vis, "l_vis_delta_vs_D427": l_vis - m.D427_L_VIS_MM,
        "collision_link5_bbox_mm": [C5.min(0).tolist(), C5.max(0).tolist()],
        "hull_sample_spacing_m": ja.SAMPLE_SPACING_M}
    print(f"[{LOG}] N10f joint pass={g_f} max={max(joint_dev.values()):.3e} | "
          f"N10g peak pass={g_g} col={l_col:.6f} vis={l_vis:.6f} d={l_col - l_vis:+.3e} | "
          f"N10h asset pass={g_h} parts={counts} legacy_disabled={legacy_all_disabled}", flush=True)

    # ---- the shared admission scan (engine imported verbatim) -------------
    c0 = np.array([0.0, 0.0, m.TCP_Z_MM / 1000.0])
    R_m = m.CYL_R_MM / 1000.0
    SPAN = m.WALL_SPAN_MM / 1000.0

    def scan(fixed_pts, mov_fn, th_deg, ph_deg, q5_list):
        chat = m.axis_dir(math.radians(th_deg), math.radians(ph_deg))
        pf = m.prep(fixed_pts, chat, c0)
        d_f, _ = m.deepest_delta(*pf, chat[2], R_m)
        rows = []
        for q in q5_list:
            pm = m.prep(mov_fn(float(q)), chat, c0)
            d_m, _ = m.deepest_delta(*pm, chat[2], R_m)
            delta = max(d_f, d_m)
            if not math.isfinite(delta):
                continue
            bf, _ = m.bite_at(*pf, chat[2], R_m, SPAN, delta)
            bm, _ = m.bite_at(*pm, chat[2], R_m, SPAN, delta)
            bb = max(bf, bm)
            if not math.isfinite(bb):
                continue
            rows.append({"q5_deg": float(q), "bite_mm": bb * 1000.0,
                         "bite_fixed_mm": bf * 1000.0 if math.isfinite(bf) else None,
                         "bite_moving_mm": bm * 1000.0 if math.isfinite(bm) else None,
                         "depth_top_min_mm": delta * 1000.0, "delta_m": delta,
                         "blocker": "fixed" if d_f >= d_m else "moving"})
        return rows, chat

    # coarse grid = EXACTLY the 34 points 40th / 41st used
    q5_coarse = np.degrees(np.unique(np.concatenate(
        [np.linspace(0.0, j5["limit"][1], 33), [m.Q5_REAL_MAX_RAD]])))
    q5_max_deg = float(math.degrees(j5["limit"][1]))
    q5_fine = np.unique(np.concatenate([
        np.round(np.arange(0.0, q5_max_deg + 1e-9, Q5_FINE_STEP_DEG), 6),
        [q5_max_deg, T3_ATTEMPTED_CLOSE_DEG, math.degrees(m.Q5_REAL_MAX_RAD)]]))
    out["q5_grids"] = {"coarse_n": int(len(q5_coarse)),
                       "coarse_step_deg": float(np.diff(q5_coarse)[:31].mean()),
                       "fine_n": int(len(q5_fine)), "fine_step_deg": Q5_FINE_STEP_DEG,
                       "q5_max_deg": q5_max_deg,
                       "explicitly_included": [T3_ATTEMPTED_CLOSE_DEG,
                                               math.degrees(m.Q5_REAL_MAX_RAD)]}

    # phi* per theta from the frozen 41st ladder (READ-ONLY)
    n8b = json.loads(N8B_RESULTS.read_text())
    ladder41 = {r["theta_deg"]: r for r in n8b["theta_ladder_full_q5"]}
    missing = [t for t in THETA_SET if t not in ladder41]
    if missing:
        print(f"[{LOG}] ABORT n8b_ladder_missing_theta={missing}", flush=True)
        return 3

    # ---- gates N10c / N10d / N10e on the coarse grid ----------------------
    rows0, _ = scan(S5, mov_visual, 0.0, ladder41[0.0]["phi_deg"], q5_coarse)
    best0 = max(rows0, key=lambda r: r["bite_mm"])
    g_c = bool(abs(best0["bite_mm"] - N40_BEST_BITE_MM) < TOL_MM and
               abs(best0["depth_top_min_mm"] - N40_DEPTH_MM) < TOL_MM)
    out["N10c_theta0_reproduces_40th"] = {
        "pass": g_c, "bite_mm": best0["bite_mm"], "ref_bite_mm": N40_BEST_BITE_MM,
        "d_bite_mm": best0["bite_mm"] - N40_BEST_BITE_MM,
        "depth_top_min_mm": best0["depth_top_min_mm"], "ref_depth_mm": N40_DEPTH_MM,
        "d_depth_mm": best0["depth_top_min_mm"] - N40_DEPTH_MM, "tol_mm": TOL_MM}

    rowsH, _ = scan(S5, mov_visual, N8_BEST["theta_deg"], N8_BEST["phi_deg"], q5_coarse)
    bestH = max(rowsH, key=lambda r: r["bite_mm"])
    g_d = bool(abs(bestH["bite_mm"] - N8_BEST["max_bite_mm"]) < TOL_MM)
    out["N10d_reproduces_n8_headline"] = {
        "pass": g_d, "bite_mm": bestH["bite_mm"], "ref_bite_mm": N8_BEST["max_bite_mm"],
        "d_bite_mm": bestH["bite_mm"] - N8_BEST["max_bite_mm"],
        "q5_deg": bestH["q5_deg"], "tol_mm": TOL_MM}
    print(f"[{LOG}] N10c pass={g_c} d_bite={out['N10c_theta0_reproduces_40th']['d_bite_mm']:.3e} | "
          f"N10d pass={g_d} d_bite={out['N10d_reproduces_n8_headline']['d_bite_mm']:.3e}", flush=True)

    ladder_repro, g_e = [], True
    for th in THETA_SET:
        ref = ladder41[th]
        rows, _ = scan(S5, mov_visual, th, ref["phi_deg"], q5_coarse)
        b = max(rows, key=lambda r: r["bite_mm"])
        d = b["bite_mm"] - ref["bite_mm"]
        ok = abs(d) < TOL_MM
        g_e = g_e and ok
        ladder_repro.append({"theta_deg": th, "phi_deg": ref["phi_deg"],
                             "bite_mm": b["bite_mm"], "ref_41st_bite_mm": ref["bite_mm"],
                             "d_bite_mm": d, "pass": bool(ok)})
    out["N10e_reproduces_41st_ladder"] = {"pass": bool(g_e), "tol_mm": TOL_MM,
                                          "rows": ladder_repro}
    print(f"[{LOG}] N10e ladder pass={g_e} max|d|="
          f"{max(abs(r['d_bite_mm']) for r in ladder_repro):.3e}", flush=True)

    if not (g_c and g_d and g_e and g_f and g_g and g_h):
        print(f"[{LOG}] ABORT reproduction_or_identity_gate_failure "
              f"c={g_c} d={g_d} e={g_e} f={g_f} g={g_g} h={g_h}", flush=True)
        return 3

    # ---- Part A + Part B: fine q5 sweep on BOTH assets --------------------
    per_theta, t0 = [], time.time()
    for th in THETA_SET:
        ph = ladder41[th]["phi_deg"]
        rv, chat = scan(S5, mov_visual, th, ph, q5_fine)
        rc_, _ = scan(C5, mov_collision, th, ph, q5_fine)
        entry = {"theta_deg": th, "phi_deg": ph, "chat": chat.tolist()}
        for key, rows in (("visual", rv), ("collision", rc_)):
            qs = [r["q5_deg"] for r in rows]
            bites = [r["bite_mm"] for r in rows]
            best = max(rows, key=lambda r: r["bite_mm"])
            at24 = min(rows, key=lambda r: abs(r["q5_deg"] - T3_ATTEMPTED_CLOSE_DEG))
            entry[key] = {
                "max_bite_mm": best["bite_mm"], "q5_star_deg": best["q5_deg"],
                "depth_top_min_mm_at_star": best["depth_top_min_mm"],
                "delta_m_at_star": best["delta_m"], "blocker_at_star": best["blocker"],
                "bite_fixed_mm_at_star": best["bite_fixed_mm"],
                "bite_moving_mm_at_star": best["bite_moving_mm"],
                "positive_windows_deg": contiguous_windows(qs, [b > 0.0 for b in bites],
                                                           Q5_FINE_STEP_DEG),
                "at_T3_commanded_close_24deg": {
                    "q5_deg": at24["q5_deg"], "bite_mm": at24["bite_mm"],
                    "inside_positive_window": bool(at24["bite_mm"] > 0.0),
                    "depth_top_min_mm": at24["depth_top_min_mm"]},
                "curve": [[round(r["q5_deg"], 4), round(r["bite_mm"], 6),
                           round(r["depth_top_min_mm"], 6)] for r in rows]}
        per_theta.append(entry)
        print(f"[{LOG}] theta={th:5.1f} phi={ph:6.1f} | visual q5*={entry['visual']['q5_star_deg']:6.2f} "
              f"bite={entry['visual']['max_bite_mm']:+9.4f} | collision q5*="
              f"{entry['collision']['q5_star_deg']:6.2f} bite={entry['collision']['max_bite_mm']:+9.4f} "
              f"depth={entry['collision']['depth_top_min_mm_at_star']:+8.4f} "
              f"blk={entry['collision']['blocker_at_star']} [{time.time() - t0:.0f}s]", flush=True)
    out["per_theta"] = per_theta

    # ---- answers ----------------------------------------------------------
    lo_b, hi_b = T3_APPROACH_BAND_DEG
    in_band = [e for e in per_theta if lo_b <= e["theta_deg"] <= hi_b]
    best_col_band = max(in_band, key=lambda e: e["collision"]["max_bite_mm"])
    best_col_all = max(per_theta, key=lambda e: e["collision"]["max_bite_mm"])
    admits = bool(best_col_band["collision"]["max_bite_mm"] > 0.0)

    # A: does q5 = 24.00 sit inside the visual positive window?
    a_rows = []
    for e in per_theta:
        w = e["visual"]["positive_windows_deg"]
        a_rows.append({
            "theta_deg": e["theta_deg"],
            "visual_window_deg": w,
            "visual_window_upper_edge_deg": (max(x["q5_hi_deg"] for x in w) if w else None),
            "coarse_grid_said_upper_22_50": True,
            "bite_at_24deg_mm": e["visual"]["at_T3_commanded_close_24deg"]["bite_mm"],
            "24deg_inside_window": e["visual"]["at_T3_commanded_close_24deg"]["inside_positive_window"]})
    resolved = [r for r in a_rows if r["visual_window_upper_edge_deg"] is not None]
    upper_edges = [r["visual_window_upper_edge_deg"] for r in resolved]
    t3_missed = [r["theta_deg"] for r in resolved if not r["24deg_inside_window"]]

    out["answer_A_q5_refinement"] = {
        "question": ("41st 5-2 left (22.50, 25.32] undetermined at 2.81 deg spacing and T3 "
                     "commanded exactly 24.00 deg - inside that gap"),
        "resolution_deg": Q5_FINE_STEP_DEG,
        "rows": a_rows,
        "upper_edge_range_deg": [min(upper_edges), max(upper_edges)] if upper_edges else None,
        "theta_where_24deg_is_OUTSIDE_the_window": t3_missed,
        "verdict": ("T3_CLOSE_TARGET_RESOLVED" if resolved else "NO_POSITIVE_WINDOW_ON_VISUAL")}

    out["answer_B_collision_asset"] = {
        "question": ("does the geometry T3 physics actually loads (frozen attempt3 collision USD) "
                     "admit a positive bite under the approved tilt?"),
        "band_deg": list(T3_APPROACH_BAND_DEG),
        "best_in_band": {"theta_deg": best_col_band["theta_deg"],
                         "phi_deg": best_col_band["phi_deg"],
                         **{k: best_col_band["collision"][k] for k in
                            ("max_bite_mm", "q5_star_deg", "depth_top_min_mm_at_star",
                             "blocker_at_star", "positive_windows_deg")}},
        "best_over_all_theta": {"theta_deg": best_col_all["theta_deg"],
                                "max_bite_mm": best_col_all["collision"]["max_bite_mm"],
                                "q5_star_deg": best_col_all["collision"]["q5_star_deg"]},
        "visual_vs_collision_at_best_band_theta": {
            "visual_max_bite_mm": best_col_band["visual"]["max_bite_mm"],
            "collision_max_bite_mm": best_col_band["collision"]["max_bite_mm"],
            "delta_mm": (best_col_band["collision"]["max_bite_mm"]
                         - best_col_band["visual"]["max_bite_mm"])}}

    out["verdict"] = {
        "code": "COLLISION_ASSET_ADMITS_TILTED_BITE" if admits
                else "COLLISION_ASSET_BLOCKS_TILTED_BITE",
        "t3_physics_grounded": admits,
        "t3_target_measured_on_the_consumed_geometry": {
            "theta_deg": best_col_band["theta_deg"],
            "phi_deg_tool_frame": best_col_band["phi_deg"],
            "q5_close_target_deg": best_col_band["collision"]["q5_star_deg"],
            "expected_bite_mm": best_col_band["collision"]["max_bite_mm"],
            "descent_delta_m": best_col_band["collision"]["delta_m_at_star"],
            "psi_world": "cell azimuth, radially outward (D432 - not a free choice)"} if admits else None,
        "q5_close_target_deg_visual": best_col_band["visual"]["q5_star_deg"],
        "T3_commanded_close_deg": T3_ATTEMPTED_CLOSE_DEG,
        "D427_D429_D430_D431_D432_status": "UNCHANGED - this run neither re-runs nor re-judges any",
        "tilt_approved_by": "professor, relayed by user 2026-08-10 (43rd) - D419 scope change",
        "force_closure": "STILL UNPROVEN - static admission only",
        "g0a_pass": False}

    gates = {"N10a_sha_pins": g_a, "N10b_D427_l_vis": g_b, "N10c_theta0_vs_40th": g_c,
             "N10d_n8_headline": g_d, "N10e_41st_ladder": g_e, "N10f_joint_frames": g_f,
             "N10g_collision_peak": g_g, "N10h_asset_identity": g_h,
             "N10i_collision_admits_in_band": admits}
    out["gates"] = {k: bool(v) for k, v in gates.items()}

    with paths["curves.csv"].open("w") as f:
        f.write("theta_deg,phi_deg,asset,q5_deg,bite_mm,depth_top_min_mm\n")
        for e in per_theta:
            for key in ("visual", "collision"):
                for q, b, d in e[key]["curve"]:
                    f.write(f"{e['theta_deg']},{e['phi_deg']},{key},{q},{b},{d}\n")

    # ---- D324 diagnostic figure ------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(19, 11))
    cmap = plt.get_cmap("viridis")
    cols = {t: cmap(i / max(1, len(THETA_SET) - 1)) for i, t in enumerate(THETA_SET)}

    for e in per_theta:
        c = cols[e["theta_deg"]]
        qv = [r[0] for r in e["visual"]["curve"]]
        ax[0][0].plot(qv, [r[1] for r in e["visual"]["curve"]], "-", lw=1.5, color=c,
                      label=f"visual th={e['theta_deg']:.0f}")
        ax[0][0].plot([r[0] for r in e["collision"]["curve"]],
                      [r[1] for r in e["collision"]["curve"]], "--", lw=1.5, color=c)
    ax[0][0].axhline(0.0, color="k", lw=1.2)
    ax[0][0].axvline(T3_ATTEMPTED_CLOSE_DEG, color="#d02020", ls=":", lw=1.8,
                     label="T3 attempt2/3 commanded close = 24 deg")
    ax[0][0].axvspan(22.50, 25.32, color="#d02020", alpha=0.08,
                     label="41st undetermined gap (2.81 deg grid)")
    ax[0][0].set_xlim(0, q5_max_deg)
    ax[0][0].set_xlabel("jaw opening q5 [deg]  (larger = more OPEN, D420 (4))")
    ax[0][0].set_ylabel("bite beside the wall [mm]")
    ax[0][0].set_title("A | bite vs q5 at 0.1 deg.  solid = visual mesh, dashed = attempt3 "
                       "COLLISION asset (what T3 loads)")
    ax[0][0].legend(fontsize=6.5, ncol=2)
    ax[0][0].grid(alpha=0.3)

    ths = [e["theta_deg"] for e in per_theta]
    ax[0][1].plot(ths, [e["visual"]["max_bite_mm"] for e in per_theta], "-o", ms=5,
                  color="#1f6fb4", label="visual mesh (D431 basis)")
    ax[0][1].plot(ths, [e["collision"]["max_bite_mm"] for e in per_theta], "-s", ms=5,
                  color="#c02020", label="attempt3 collision asset (T3 basis)")
    ax[0][1].plot(ths, [ladder41[t]["bite_mm"] for t in ths], "x", ms=8, color="#888888",
                  label="41st coarse grid (reference)")
    ax[0][1].axhline(0.0, color="k", lw=1.2)
    ax[0][1].axvspan(lo_b, hi_b, color="#20a020", alpha=0.10, label="D432 approach band [15, 29]")
    ax[0][1].set_xlabel("tool-axis tilt theta [deg]")
    ax[0][1].set_ylabel("max bite over q5 [mm]")
    ax[0][1].set_title(f"B | does the CONSUMED geometry bite?  verdict: {out['verdict']['code']}")
    ax[0][1].legend(fontsize=8)
    ax[0][1].grid(alpha=0.3)

    for i, e in enumerate(per_theta):
        for key, off, col in (("visual", -0.18, "#1f6fb4"), ("collision", +0.18, "#c02020")):
            for w in e[key]["positive_windows_deg"]:
                ax[1][0].plot([w["q5_lo_deg"], w["q5_hi_deg"]], [i + off, i + off],
                              lw=7, solid_capstyle="butt", color=col)
    ax[1][0].axvline(T3_ATTEMPTED_CLOSE_DEG, color="#d02020", ls=":", lw=1.8)
    ax[1][0].set_yticks(range(len(per_theta)))
    ax[1][0].set_yticklabels([f"{e['theta_deg']:.0f}" for e in per_theta])
    ax[1][0].set_xlim(0, q5_max_deg)
    ax[1][0].set_xlabel("q5 [deg]")
    ax[1][0].set_ylabel("theta [deg]")
    ax[1][0].set_title("C | positive-bite q5 window.  blue = visual (upper), red = collision (lower).  "
                       "dotted = the 24 deg T3 actually commanded")
    ax[1][0].grid(alpha=0.3, axis="x")

    eb = best_col_band
    for key, sty, col in (("visual", "-", "#1f6fb4"), ("collision", "--", "#c02020")):
        ax[1][1].plot([r[0] for r in eb[key]["curve"]], [r[2] for r in eb[key]["curve"]],
                      sty, lw=1.8, color=col, label=f"{key} descent limit")
    ax[1][1].axhline(0.0, color="k", lw=1.0)
    ax[1][1].axhline(4.4, color="#e08a1e", ls="--", lw=1.4,
                     label="attempt1 physical stop top+4.4 mm (D424)")
    ax[1][1].axvline(T3_ATTEMPTED_CLOSE_DEG, color="#d02020", ls=":", lw=1.8)
    ax[1][1].set_xlim(0, q5_max_deg)
    ax[1][1].set_xlabel("q5 [deg]")
    ax[1][1].set_ylabel("depth = z - TCP [mm]   (<0 = object enters the throat)")
    ax[1][1].set_title(f"D | descent limit at the winning tilt theta = {eb['theta_deg']:.0f} deg")
    ax[1][1].legend(fontsize=8)
    ax[1][1].grid(alpha=0.3)

    fig.suptitle(f"g0b_d420 {TAG} - T3 pre-flight on the CONSUMED geometry + 0.1 deg q5 refinement "
                 f"(read-only).  VERDICT: {out['verdict']['code']}", fontsize=13)
    fig.tight_layout()
    fig.savefig(paths["diagnostic.png"], dpi=118)
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

    start_idx = next(i for i, e in enumerate(per_theta) if e["theta_deg"] == eb["theta_deg"])
    with rr.RecordingStream(app_id, recording_id=f"g0b_d420_{TAG}", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(paths["timeline.rrd"]), write_footer=True)
        rec.log("assembly/link5_visual", rr.Points3D(view_pts(S5[::VIEW_STRIDE_VIS]),
                colors=[150, 150, 160], radii=0.0004), static=True)
        rec.log("assembly/link5_collision", rr.Points3D(view_pts(C5[::VIEW_STRIDE_COL]),
                colors=[70, 200, 200], radii=0.0004), static=True)
        rec.log("assembly/tcp", rr.Points3D([[0.0, 0.0, 0.0]], colors=[[40, 200, 80]],
                radii=0.002), static=True)
        rec.log("assembly/tool_axis", rr.LineStrips3D([[[0.0, 0.0, 0.07], [0.0, 0.0, -0.07]]],
                colors=[[40, 200, 80]], radii=0.0003), static=True)
        rec.log("reference/d427_blocker_peak", rr.Points3D(
                [[-0.010025849, 0.001408991, -(119.885620 - m.TCP_Z_MM) / 1000.0]],
                colors=[[255, 210, 40]], radii=0.0016), static=True)
        for i, e in enumerate(per_theta):
            chat = np.array(e["chat"])
            rec.reset_time()
            rec.set_time("theta_index", sequence=i)
            rec.log("assembly/gripper_visual", rr.Points3D(
                view_pts(mov_visual(e["visual"]["q5_star_deg"])[::VIEW_STRIDE_VIS]),
                colors=[70, 130, 230], radii=0.0004))
            rec.log("assembly/gripper_collision", rr.Points3D(
                view_pts(mov_collision(e["collision"]["q5_star_deg"])[::VIEW_STRIDE_COL]),
                colors=[225, 90, 200], radii=0.0004))
            rings, walls = cyl_wire(chat, e["collision"]["delta_m_at_star"], m.CYL_R_MM, m.CYL_H_MM)
            rec.log("object/cylinder_tilted", rr.LineStrips3D(
                rings + walls, colors=[[225, 60, 60]] * (len(rings) + len(walls)), radii=0.00035))
            rec.log("object/topface_plane_tilted", rr.LineStrips3D(
                [topface_ring(chat, e["collision"]["delta_m_at_star"], 30.0)],
                colors=[[235, 140, 30]], radii=0.0005))
            rec.log("plots/theta_deg", rr.Scalars(float(e["theta_deg"])))
            rec.log("plots/bite_visual_mm", rr.Scalars(float(e["visual"]["max_bite_mm"])))
            rec.log("plots/bite_collision_mm", rr.Scalars(float(e["collision"]["max_bite_mm"])))
            rec.log("plots/q5_star_visual_deg", rr.Scalars(float(e["visual"]["q5_star_deg"])))
            rec.log("plots/q5_star_collision_deg", rr.Scalars(float(e["collision"]["q5_star_deg"])))
            rec.log("plots/depth_collision_mm",
                    rr.Scalars(float(e["collision"]["depth_top_min_mm_at_star"])))
        rec.reset_time()
        rec.set_time("theta_index", sequence=0)
        for name, ok in gates.items():
            rec.log("events/gates", rr.TextLog(name, level=rr.TextLogLevel.INFO if ok
                                               else rr.TextLogLevel.ERROR))
        wins = out["answer_B_collision_asset"]["best_in_band"]["positive_windows_deg"]
        win_txt = ", ".join(f"[{w['q5_lo_deg']:.1f}, {w['q5_hi_deg']:.1f}]" for w in wins) or "none"
        summary_md = (
            f"# g0b_d420 {TAG} - T3 pre-flight on the CONSUMED geometry (read-only)\n\n"
            f"**VERDICT: {out['verdict']['code']}**\n\n"
            f"## Why\nThe professor approved an approach tilt, so T3 physics is next. But every "
            f"tilt number so far (D431, D432) was measured on the VISUAL mesh, and T3 physics "
            f"loads the frozen **attempt3 collision USD** instead (prereg supersession S-2). "
            f"D425 found that asset's jaws occluded. So this run re-measures the tilt admission on "
            f"the geometry the pipeline actually consumes - D428 #29 - and at the same time closes "
            f"41st 5-2, which left the closing target undetermined between 22.50 and 25.32 deg "
            f"while T3 commanded exactly 24.00 deg.\n\n"
            f"## Answer A - closing target (visual mesh, 0.1 deg)\n"
            f"- positive-bite window upper edge across theta: "
            f"**{out['answer_A_q5_refinement']['upper_edge_range_deg']} deg**;\n"
            f"- theta where the commanded 24.00 deg falls OUTSIDE the window: "
            f"**{out['answer_A_q5_refinement']['theta_where_24deg_is_OUTSIDE_the_window']}**.\n\n"
            f"## Answer B - the consumed geometry\n"
            f"- best in the D432 band [15, 29] deg: theta = **{eb['theta_deg']:.0f} deg**, "
            f"q5 = **{eb['collision']['q5_star_deg']:.2f} deg**, bite = "
            f"**{eb['collision']['max_bite_mm']:+.4f} mm**, deepest top face = TCP"
            f"**{eb['collision']['depth_top_min_mm_at_star']:+.4f} mm**, blocker = "
            f"**{eb['collision']['blocker_at_star']}**;\n"
            f"- its positive q5 window: **{win_txt} deg**;\n"
            f"- same theta on the visual mesh: **{eb['visual']['max_bite_mm']:+.4f} mm** "
            f"(delta {out['answer_B_collision_asset']['visual_vs_collision_at_best_band_theta']['delta_mm']:+.4f} mm).\n\n"
            f"## Reproduction + identity gates\n"
            f"- N10c theta=0 vs 40th: d_bite **{out['N10c_theta0_reproduces_40th']['d_bite_mm']:.3e} mm**;\n"
            f"- N10d n8 headline: d_bite **{out['N10d_reproduces_n8_headline']['d_bite_mm']:.3e} mm**;\n"
            f"- N10e whole 41st ladder: max|d| "
            f"**{max(abs(r['d_bite_mm']) for r in ladder_repro):.3e} mm**;\n"
            f"- N10f URDF joint == USD joint: max **{max(joint_dev.values()):.3e}**;\n"
            f"- N10g collision fixed peak vs visual l_vis: **{l_col - l_vis:+.3e} mm**;\n"
            f"- N10h 64+64 enabled convexHull parts, both legacy colliders **disabled**.\n\n"
            f"## Limits\nStatic admission only - `bite > 0` is NECESSARY for closing contact, never "
            f"sufficient, and this is **not** a grasp-success prediction. Force closure remains "
            f"unproven (D431's positive bite is one jaw only). No physics, no friction, no torque. "
            f"Convex-hull parts are what PhysX cooks, but the scipy hull only APPROXIMATES that "
            f"cooked surface. Gate-0 was neither re-run nor re-judged; "
            f"D427/D429/D430/D431/D432 unchanged; `g0a_pass=false`.\n\n"
            f"## Scene\ngrey = link5 visual mesh, cyan = link5 attempt3 COLLISION hulls, "
            f"blue = moving jaw (visual) at its own argmax q5, magenta = moving jaw (collision) at "
            f"ITS argmax q5, red = the D29 cylinder at the collision asset's deepest admissible "
            f"height for this step's tilt, orange = its top-face plane, yellow = the D427 blocking "
            f"peak, green = TCP and tool axis. z_view = -(z - z_TCP), so DOWN = distal.\n\n"
            f"The 3D view is PINNED by blueprint to theta = **{eb['theta_deg']:.0f} deg**, the "
            f"winning tilt, because 42nd 6-1 recorded that its screenshot landed on theta = 0 "
            f"where the tilt under test is invisible. All {len(per_theta)} steps are in the RRD; "
            f"removing that `time_ranges` override restores free scrubbing.\n\n"
            f"Authority = stdout + `{paths['results.json'].name}`. Rerun is inspection evidence "
            f"only (D341).\n")
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN),
                static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run",
                                         name="1 | pre-flight verdict"),
                    # 42nd 6-1 limit 2: its screenshot landed on theta=0, where the tilt
                    # under test is invisible. Pin this view to the decision step by
                    # absolute sequence so the headless screenshot always shows the
                    # winning tilt. Every step is still in the RRD - removing this
                    # time_ranges override restores free scrubbing.
                    rrb.Spatial3DView(origin="/", contents=["/assembly/**", "/object/**",
                                                            "/reference/**"],
                                      name=f"2 | PINNED to the decision tilt theta="
                                           f"{eb['theta_deg']:.0f} deg (visual vs collision)",
                                      time_ranges=rrb.VisibleTimeRange(
                                          "theta_index",
                                          start=rrb.TimeRangeBoundary.absolute(seq=start_idx),
                                          end=rrb.TimeRangeBoundary.absolute(seq=start_idx))),
                    rrb.TextLogView(origin="/events/gates", contents="/events/gates/**",
                                    name="3 | gates"),
                    column_shares=[0.30, 0.46, 0.24],
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/bite_visual_mm/**",
                                                 "/plots/bite_collision_mm/**",
                                                 "/plots/depth_collision_mm/**"],
                                       name="4 | bite: visual mesh vs consumed collision asset"),
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/theta_deg/**",
                                                 "/plots/q5_star_visual_deg/**",
                                                 "/plots/q5_star_collision_deg/**"],
                                       name="5 | tilt and the closing target that achieves it"),
                ),
                row_shares=[0.58, 0.42],
            ),
            auto_layout=False, auto_views=False, collapse_panels=True,
        )
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=30.0)
    blueprint.save(app_id, str(paths["timeline.rbl"]))

    expected_entities = ["metadata/run", "assembly/link5_visual", "assembly/link5_collision",
                         "assembly/gripper_visual", "assembly/gripper_collision", "assembly/tcp",
                         "assembly/tool_axis", "reference/d427_blocker_peak",
                         "object/cylinder_tilted", "object/topface_plane_tilted",
                         "plots/theta_deg", "plots/bite_visual_mm", "plots/bite_collision_mm",
                         "plots/q5_star_visual_deg", "plots/q5_star_collision_deg",
                         "plots/depth_collision_mm", "events/gates"]
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    lin3 = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    sca = ["Scalars:scalars"]
    components = {
        "metadata/run": ["TextDocument:text"],
        "assembly/link5_visual": pts3, "assembly/link5_collision": pts3,
        "assembly/gripper_visual": pts3, "assembly/gripper_collision": pts3,
        "assembly/tcp": pts3, "reference/d427_blocker_peak": pts3,
        "assembly/tool_axis": lin3, "object/cylinder_tilted": lin3,
        "object/topface_plane_tilted": lin3,
        "plots/theta_deg": sca, "plots/bite_visual_mm": sca, "plots/bite_collision_mm": sca,
        "plots/q5_star_visual_deg": sca, "plots/q5_star_collision_deg": sca,
        "plots/depth_collision_mm": sca,
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
    print(f"[{LOG}] G0B_T3R_N10_VERDICT={out['verdict']['code']} "
          f"theta={eb['theta_deg']:.0f} q5={eb['collision']['q5_star_deg']:.2f} "
          f"bite={eb['collision']['max_bite_mm']:+.4f}mm", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
