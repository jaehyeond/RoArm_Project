"""g0b_t3d_perstep_jaw_clearance_readonly_audit.py — 46th session, 2026-08-10

PREREGISTERED: claudedocs/runtime_logs/grasp_track/g0b_d420/t3d_prereg.md

WHAT THIS IS
  D434 / D434-R1 attributed the T3t contact events (which jaw touched the D29
  cylinder, and when) by watching ONE proxy variable: delta, the TCP depth along
  the tool axis.  Nothing was measured against geometry.  D435 reproduced those
  numbers and then blocked the follow-up because the preregistration carried
  three defects (no leg on bracket (b); a delta gate finer than the bookkeeping
  that produced it; a step range that truncates leg 3).

  This run does the measurement the attribution was standing in for:

      for every logged physics step of every leg, the SIGNED CLEARANCE between
      each jaw and the cylinder, computed on the geometry the physics actually
      loaded (frozen attempt3 collision USD) at the pose the physics actually
      reached (logged TCP + logged object position/quaternion + logged q5).

  Positive = nearest-approach distance.  Negative = deepest penetration depth.

NEW VARIABLES (two, declared - Variable Ladder Protocol)
  1. consumed collision asset  (every per-step clearance so far used the visual mesh)
  2. measured per-step pose    (the static engine assumes the cylinder axis passes
                                exactly through TCP with zero lateral offset)

DECLARED ASSUMPTION A-4 (prereg 3-2)
  steps.csv logs TCP POSITION only; arm joint angles are not logged.  The tool
  ORIENTATION is therefore modelled, not measured:
    M-A (primary)     R_tool = FK(q_descend_deg), constant through close
    M-B (sensitivity) R_tool(s) from a minimum-norm Jacobian reconstruction
  Verdicts are judged on M-A.  If M-B moves a crossing step, that prediction is
  reported ORIENTATION_SENSITIVE and does not get a STRICT PASS.

NOT IN SCOPE
  No Isaac, no physics, no robot, no contact forces, no friction, no Gate-0
  re-run or re-judgement (D427/D429/D430/D431/D432/D433/D434/D434-R1 unchanged),
  no overwrite of any t3_*, t3r_*, t3t_* or existing t3d_* artifact, no frozen
  source modified, no re-decomposition.  The USD is opened READ-ONLY through the
  frozen reader.  Clearance < 0 is a NECESSARY condition for closing contact,
  never sufficient, and never a grasp-success prediction.  g0a_pass stays false.
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
# Same flag name the frozen jaw audit uses, so importing that module is a no-op
# instead of a second re-exec (verbatim from n10's bootstrap).
_REEXEC_FLAG = "G0B_JAW_AUDIT_REEXEC"
LOG = "g0b_t3d_perstep"


def _bootstrap_pxr_env() -> None:
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

import csv              # noqa: E402
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
KIN_SRC = ROOT / "sim_scripts/roarm_kinematics.py"
OUT_DIR = ROOT / "claudedocs/runtime_logs/grasp_track/g0b_d420"
PREREG = OUT_DIR / "t3d_prereg.md"
TAG = "t3d_perstep"

RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"
NUMPY_PIN = "1.26.0"

LEGS = (
    (1, "t3t_grasp_results.json", "t3t_grasp_steps.csv"),
    (2, "t3t_grasp2_results.json", "t3t_grasp2_steps.csv"),
    (3, "t3t_grasp3_results.json", "t3t_grasp3_steps.csv"),
)
STEP_LO = 386                       # prereg A-3: 386 .. per-leg max_step (no truncation)

# prereg 4: (name, leg, step_before, step_after, quantity, direction)
#   direction "enter" = positive -> non-positive ; "exit" = non-positive -> positive
PREREG_PREDICTIONS = (
    ("P-a", 3, 388, 389, "fixed", "enter"),
    ("P-b", 1, 490, 491, "fixed", "exit"),
    ("P-c", 3, 500, 501, "moving", "enter"),
)
STEP_TOL = 3                        # prereg 4, fixed BEFORE the run, not tuned after

# reproduction pins quoted for gates only (never re-derived from these)
D427_L_VIS_MM = 4.457620117187505
D427_N_PTS = 2266503
N10G_TOL_MM = 0.05
G4_TOL = 1e-9
G5_TOL_MM = 1e-9
# frozen exploratory reference values reproduced by G5 (44th, t3d_explore_clearance_stdout.log)
G5_REF = {"fixed_cmd_mm": 0.0000, "fixed_act_mm": 1.5639, "moving_act_q5_19p50_mm": -0.4059}
G5_REF_TOL_MM = 5e-4                # the log prints 4 decimals

VIEW_STRIDE_FIXED = 40
VIEW_STRIDE_MOVING = 128


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


# --------------------------------------------------------------------------- #
# the one new piece of maths: signed distance to a finite solid cylinder.
# Expressed in whatever frame the caller hands it; gate G5 proves it agrees
# with the frozen n8 `prep` path on the same nominal pose.
# --------------------------------------------------------------------------- #
def signed_clearance(P, c, ahat, R_m, h_m):
    """min signed distance (mm) from points P to the solid cylinder
    (centre c, unit axis ahat, radius R_m, half-height h_m).
    + = nearest-approach distance outside, - = deepest penetration depth.
    Returns (value_mm, argmin_index, n_inside)."""
    v = P - c[None, :]
    u = v @ ahat
    rho = np.sqrt(np.maximum(np.einsum("ij,ij->i", v, v) - u * u, 0.0))
    dr = rho - R_m                       # + outside the wall
    dz = np.abs(u) - h_m                 # + beyond an end cap
    inside = (dr <= 0.0) & (dz <= 0.0)
    d = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))
    if inside.any():
        d[inside] = np.maximum(dr[inside], dz[inside])   # negative
    k = int(np.argmin(d))
    return float(d[k]) * 1000.0, k, int(inside.sum())


def quat_to_R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def crossing_step(steps, vals, direction):
    """First index i with a sign change of the prereg kind. Returns (s_before,
    s_after) or None.  'enter' = >0 -> <=0 ; 'exit' = <=0 -> >0."""
    for i in range(1, len(vals)):
        a, b = vals[i - 1], vals[i]
        if direction == "enter" and a > 0.0 >= b:
            return int(steps[i - 1]), int(steps[i])
        if direction == "exit" and a <= 0.0 < b:
            return int(steps[i - 1]), int(steps[i])
    return None


# =========================================================================== #
def main() -> int:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {k: OUT_DIR / f"{TAG}_{k}" for k in
             ("results.json", "curves.csv", "timeline.rrd", "timeline.rbl",
              "rerun_validation.json", "inspection.png", "diagnostic.png", "script.py.txt")}

    # ---- G0 write guard ---------------------------------------------------
    existing = [p.name for p in paths.values() if p.exists()]
    if existing:
        print(f"[{LOG}] ABORT G0_write_guard existing={existing}", flush=True)
        return 3
    if not PREREG.is_file():
        print(f"[{LOG}] ABORT missing_prereg={PREREG}", flush=True)
        return 3

    import rerun as rr
    if rr.__version__ != RERUN_VERSION or np.__version__ != NUMPY_PIN:
        print(f"[{LOG}] ABORT G6_env rerun={rr.__version__} numpy={np.__version__}", flush=True)
        return 3

    m = load_module("n8_core", N8_SRC)
    ja = load_module("jaw_audit_core", JA_SRC)
    from roarm_kinematics import _CHAIN, Tmat, Trot_z          # noqa: N811

    gates: dict[str, bool] = {}
    out = {
        "tool": TAG, "session": "46th", "prereg": PREREG.name,
        "prereg_sha256_16": sha256(PREREG)[:16],
        "read_only_assets": True, "isaac_launched": False, "physics_rerun": False,
        "question": ("at every logged physics step, what is the signed clearance between each jaw "
                     "and the D29 cylinder, on the CONSUMED collision asset at the MEASURED pose"),
        "imported_verbatim_from": {
            "admission_engine": {"path": str(N8_SRC.relative_to(ROOT)), "sha256_16": sha256(N8_SRC)[:16]},
            "collision_asset_reader": {"path": str(JA_SRC.relative_to(ROOT)), "sha256_16": sha256(JA_SRC)[:16]},
            "kinematics": {"path": str(KIN_SRC.relative_to(ROOT)), "sha256_16": sha256(KIN_SRC)[:16]}},
        "env": {"numpy": np.__version__, "rerun_sdk": rr.__version__, "python": sys.version.split()[0]},
    }

    # ---- G1 visual reproduction (D427) ------------------------------------
    S5 = m.sample_triangles(m.load_binary_stl(m.L5) * m.STL_SCALE, m.SAMPLE_SPACING_M)
    SG = m.sample_triangles(m.load_binary_stl(m.GJ) * m.STL_SCALE, m.SAMPLE_SPACING_M)
    z5 = S5[:, 2] * 1000.0
    r5 = np.hypot(S5[:, 0], S5[:, 1]) * 1000.0
    l_vis = float((z5 - m.TCP_Z_MM)[r5 <= 30.0].max())
    gates["G1_D427_visual"] = bool(abs(l_vis - D427_L_VIS_MM) < 1e-9 and len(S5) == D427_N_PTS)
    print(f"[{LOG}] G1 l_vis={l_vis:.12f} n_pts={len(S5)} pass={gates['G1_D427_visual']}", flush=True)

    # ---- collision asset, READ-ONLY Usd.Stage.Open (frozen reader) --------
    asset = ja.extract_asset()
    bodies, joint = asset["bodies"], asset["joint"]
    C5, PID5 = ja.concat_parts(bodies["link5"]["parts"])
    CG, PIDG = ja.concat_parts(bodies["gripper_link"]["parts"])
    NAME5 = [p["name"] for p in bodies["link5"]["parts"]]
    NAMEG = [p["name"] for p in bodies["gripper_link"]["parts"]]

    zc = C5[:, 2] * 1000.0
    rc = np.hypot(C5[:, 0], C5[:, 1]) * 1000.0
    l_col = float((zc - m.TCP_Z_MM)[rc <= 30.0].max())
    gates["G2_N10g_peak"] = bool(abs(l_col - l_vis) < N10G_TOL_MM)
    counts = {k: len(bodies[k]["parts"]) for k in ("link5", "gripper_link")}
    legacy_all_disabled = all(not en for v in (bodies[k]["legacy"] for k in bodies) for _, en in v)
    approx_bad = {k: bodies[k]["approx_bad"] for k in ("link5", "gripper_link")}
    gates["G3_N10h_identity"] = bool(counts["link5"] == ja.EXPECTED_PART_COUNT and
                                     counts["gripper_link"] == ja.EXPECTED_PART_COUNT and
                                     legacy_all_disabled and not any(approx_bad.values()))
    out["asset"] = {
        "usd": str(ja.ATTEMPT3_USD), "usd_root_sha256": sha256(ja.ATTEMPT3_USD),
        "usd_physics_sha256": sha256(ja.ATTEMPT3_PHYSICS_LAYER),
        "enabled_convexhull_parts": counts, "legacy_all_disabled": bool(legacy_all_disabled),
        "non_convexhull_approximation": approx_bad,
        "collision_link5_samples": int(len(C5)), "collision_gripper_samples": int(len(CG)),
        "visual_link5_samples": int(len(S5)), "visual_gripper_samples": int(len(SG)),
        "hull_sample_spacing_m": ja.SAMPLE_SPACING_M,
        "l_vis_mm": l_vis, "l_col_mm": l_col, "l_col_minus_l_vis_mm": l_col - l_vis}
    print(f"[{LOG}] G2 l_col={l_col:.6f} d={l_col - l_vis:+.4e} pass={gates['G2_N10g_peak']} | "
          f"G3 parts={counts} legacy_disabled={legacy_all_disabled} pass={gates['G3_N10h_identity']}",
          flush=True)

    # ---- read the three legs (no transcription: cap / d / q come from JSON) -
    legs = []
    for leg_id, rjson, rcsv in LEGS:
        doc = json.loads((OUT_DIR / rjson).read_text())
        rows = list(csv.DictReader((OUT_DIR / rcsv).open()))
        legs.append({
            "leg": leg_id, "results": rjson, "steps_csv": rcsv,
            "results_sha256_16": sha256(OUT_DIR / rjson)[:16],
            "steps_sha256_16": sha256(OUT_DIR / rcsv)[:16],
            "doc": doc, "rows": rows,
            "max_step": max(int(r["physics_step"]) for r in rows)})
    d3 = legs[2]["doc"]
    cap = np.array(d3["plan"]["world_grasp"], dtype=np.float64)
    dvec = np.array(d3["tilt_preflight"]["tilt"]["target_axis_world"], dtype=np.float64)
    q_descend = np.array(d3["plan"]["q_descend_deg"], dtype=np.float64)
    size_m = d3["object"]["size_m"]
    CYL_R = float(size_m[0]) / 2.0
    CYL_H = float(size_m[2])
    out["inputs_bitpinned"] = {
        "cap_world_grasp_m": cap.tolist(), "cap_hex": [float(v).hex() for v in cap],
        "target_axis_world": dvec.tolist(), "d_hex": [float(v).hex() for v in dvec],
        "d_norm_minus_1": float(np.linalg.norm(dvec) - 1.0),
        "q_descend_deg": q_descend.tolist(),
        "cylinder_radius_m": CYL_R, "cylinder_height_m": CYL_H,
        "note": ("read from t3t_grasp3_results.json, never transcribed. D435 (4)'s +0.0002 mm was "
                 "cap rounding (0.050 vs 0.04999978616833687), not d normalisation.")}

    # ---- G4 FK chain -------------------------------------------------------
    def fk_T_link5(q6_deg):
        q = np.radians(np.asarray(q6_deg, dtype=np.float64))
        T = np.eye(4)
        for name, xyz, rpy, qi in _CHAIN:
            T = T @ Tmat(xyz, rpy)
            if qi is not None:
                T = T @ Trot_z(q[qi])
            if name == "link4_to_link5":
                return T
        raise RuntimeError("link4_to_link5 not in _CHAIN")

    def tcp_of(q6_deg):
        T = fk_T_link5(q6_deg)
        return T[:3, :3] @ ja.TCP_LOCAL + T[:3, 3], T[:3, :3]

    g4_rows = []
    for key, qk, i in (("approach_tcp", "q_approach_deg", 0), ("descend_tcp", "q_descend_deg", 1),
                       ("lift_tcp", "q_lift_deg", 2)):
        p, R = tcp_of(d3["plan"][qk])
        err_mm = float(np.linalg.norm(p - np.array(d3["plan"][key])) * 1000.0)
        tilt = math.degrees(math.acos(float(np.clip(R[:, 2] @ dvec, -1.0, 1.0))))
        g4_rows.append({"key": key, "fk_err_mm": err_mm, "artifact_ik_err_mm": d3["plan"]["ik_err_mm"][i],
                        "d_err": abs(err_mm - d3["plan"]["ik_err_mm"][i]),
                        "fk_tilt_deg": tilt, "artifact_ik_tilt_deg": d3["plan"]["ik_tilt_deg"][i],
                        "d_tilt": abs(tilt - d3["plan"]["ik_tilt_deg"][i])})
    gates["G4_fk_chain"] = bool(max(max(r["d_err"], r["d_tilt"]) for r in g4_rows) < G4_TOL)
    out["G4_fk_chain"] = {"pass": gates["G4_fk_chain"], "rows": g4_rows, "tol": G4_TOL,
                          "meaning": "sim world == URDF world and the transform chain is the probe's"}
    print(f"[{LOG}] G4 fk max_dev={max(max(r['d_err'], r['d_tilt']) for r in g4_rows):.3e} "
          f"pass={gates['G4_fk_chain']}", flush=True)

    R_TOOL_MA = fk_T_link5(q_descend)[:3, :3]
    TCP_FK_DESCEND = R_TOOL_MA @ ja.TCP_LOCAL + fk_T_link5(q_descend)[:3, 3]

    # ---- G5 clearance cross-check against the frozen n8 path ---------------
    chat = m.axis_dir(math.radians(29.0), 0.0)
    delta_cmd = float(d3["tilt_preflight"]["T3T_b_matches_n10_collision_measurement"]
                      ["descend_delta_m_commanded"])
    delta_act = -float((np.array([0.213115, -0.194919, 0.050178]) - cap) @ dvec)   # step 385 TCP

    def nominal_pose(delta_m):
        """cylinder in the link5 frame exactly as the frozen engine places it:
        top-face centre on the tool axis at TCP + delta*z_hat, body along +chat."""
        c_top = ja.TCP_LOCAL + np.array([0.0, 0.0, delta_m])
        return c_top + chat * (CYL_H / 2.0), chat

    u_urdf = m.parse_urdf(m.URDF)["joints"]["link5_to_gripper_link"]
    R_j5 = m.rpy_matrix(*u_urdf["rpy"])
    t_j5 = np.array(u_urdf["xyz"], dtype=np.float64)

    def mov_visual(q5_deg):
        """VERBATIM shape of the frozen explore-script `mov()` (44th)."""
        c, s_ = math.cos(math.radians(q5_deg)), math.sin(math.radians(q5_deg))
        Rq = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        return SG @ (R_j5 @ Rq).T + t_j5

    g5_rows = []
    for label, pts, delta_m, ref in (
            ("fixed_cmd", S5, delta_cmd, G5_REF["fixed_cmd_mm"]),
            ("fixed_act", S5, delta_act, G5_REF["fixed_act_mm"]),
            ("moving_act_q5_19.50", mov_visual(19.5), delta_act, G5_REF["moving_act_q5_19p50_mm"])):
        c, a = nominal_pose(delta_m)
        mine, _, n_in = signed_clearance(pts, c, a, CYL_R, CYL_H / 2.0)
        # the frozen path, called directly
        u0, rho0_sq, b = m.prep(pts, a, np.array([0.0, 0.0, m.TCP_Z_MM / 1000.0]))
        cz = a[2]
        aa = 1.0 - cz * cz
        u = u0 - delta_m * cz
        rho = np.sqrt(np.maximum(rho0_sq + 2.0 * delta_m * b + delta_m * delta_m * aa, 0.0))
        dr = rho - CYL_R
        du_top, du_bot = -u, u - CYL_H
        ins = (dr <= 0.0) & (du_top <= 0.0) & (du_bot <= 0.0)
        dd = np.hypot(np.maximum(dr, 0.0), np.maximum(np.maximum(du_top, du_bot), 0.0))
        if ins.any():
            dd[ins] = -np.minimum(np.minimum(-dr[ins], -du_top[ins]), -du_bot[ins])
        frozen = float(dd.min()) * 1000.0
        g5_rows.append({"case": label, "world_form_mm": mine, "frozen_prep_form_mm": frozen,
                        "delta_mm": abs(mine - frozen), "frozen_44th_log_mm": ref,
                        "delta_vs_44th_log_mm": abs(mine - ref), "n_inside": n_in})
    gates["G5_clearance_crosscheck"] = bool(
        max(r["delta_mm"] for r in g5_rows) < G5_TOL_MM and
        max(r["delta_vs_44th_log_mm"] for r in g5_rows) < G5_REF_TOL_MM)
    out["G5_clearance_crosscheck"] = {"pass": gates["G5_clearance_crosscheck"], "rows": g5_rows,
                                      "tol_mm": G5_TOL_MM, "tol_vs_log_mm": G5_REF_TOL_MM,
                                      "delta_act_mm_step385": delta_act * 1000.0}
    for r in g5_rows:
        print(f"[{LOG}] G5 {r['case']:20s} world={r['world_form_mm']:+.6f} "
              f"frozen={r['frozen_prep_form_mm']:+.6f} d={r['delta_mm']:.2e} "
              f"vs44th={r['frozen_44th_log_mm']:+.4f} d={r['delta_vs_44th_log_mm']:.2e}", flush=True)
    gates["G6_env_pins"] = True

    if not all(gates.values()):
        print(f"[{LOG}] ABORT gate_failure {gates}", flush=True)
        return 3

    # ---- the measurement ---------------------------------------------------
    J = None
    try:
        from roarm_kinematics import jacobian_numerical
        J = jacobian_numerical(q_descend[:5])
    except Exception as exc:                                  # pragma: no cover
        print(f"[{LOG}] WARN jacobian unavailable ({exc!r}) - M-B skipped", flush=True)
    Jpinv = np.linalg.pinv(J) if J is not None else None

    def R_tool_MB(tcp_meas):
        if Jpinv is None:
            return R_TOOL_MA
        dq = Jpinv @ (tcp_meas - TCP_FK_DESCEND)
        q = q_descend.copy()
        q[:5] += dq
        return fk_T_link5(q)[:3, :3]

    Tg_cache: dict[float, np.ndarray] = {}

    def gripper_T(q5_deg):
        key = round(float(q5_deg), 6)
        T = Tg_cache.get(key)
        if T is None:
            T = ja.gripper_T_l5(joint, key)
            Tg_cache[key] = T
        return T

    curves = []
    per_leg = {}
    for L in legs:
        rows = [r for r in L["rows"] if STEP_LO <= int(r["physics_step"]) <= L["max_step"]]
        rec = {"leg": L["leg"], "n_steps": len(rows), "step_lo": STEP_LO, "step_hi": L["max_step"],
               "steps": [], "fixed_mm": [], "moving_mm": [], "fixed_mm_MB": [], "moving_mm_MB": [],
               "fixed_part": [], "moving_part": [], "fixed_inside": [], "moving_inside": [],
               "fixed_vis_mm": [], "moving_vis_mm": [],
               "delta_mm": [], "q5_deg": [], "tilt_deg": [], "phase": []}
        t_leg = time.time()
        for r in rows:
            s = int(r["physics_step"])
            tcp = np.array([float(r["tcp_x"]), float(r["tcp_y"]), float(r["tcp_z"])])
            obj = np.array([float(r["obj_x"]), float(r["obj_y"]), float(r["obj_z"])])
            R_obj = quat_to_R(float(r["quat_w"]), float(r["quat_x"]), float(r["quat_y"]), float(r["quat_z"]))
            a_w = R_obj[:, 2]
            q5 = float(r["q5_deg"])
            for model, R_tool in (("MA", R_TOOL_MA), ("MB", R_tool_MB(tcp))):
                t_w = tcp - R_tool @ ja.TCP_LOCAL
                c_l5 = R_tool.T @ (obj - t_w)               # cylinder in the link5 frame
                a_l5 = R_tool.T @ a_w
                cf, kf, nf = signed_clearance(C5, c_l5, a_l5, CYL_R, CYL_H / 2.0)
                Tg = gripper_T(q5)
                c_g = Tg[:3, :3].T @ (c_l5 - Tg[:3, 3])     # cylinder in the gripper_link frame
                a_g = Tg[:3, :3].T @ a_l5
                cm, km, nm = signed_clearance(CG, c_g, a_g, CYL_R, CYL_H / 2.0)
                if model == "MA":
                    cfv, _, _ = signed_clearance(S5, c_l5, a_l5, CYL_R, CYL_H / 2.0)
                    cmv, _, _ = signed_clearance(SG, c_g, a_g, CYL_R, CYL_H / 2.0)
                    rec["steps"].append(s)
                    rec["fixed_mm"].append(cf); rec["moving_mm"].append(cm)
                    rec["fixed_part"].append(NAME5[int(PID5[kf])])
                    rec["moving_part"].append(NAMEG[int(PIDG[km])])
                    rec["fixed_inside"].append(nf); rec["moving_inside"].append(nm)
                    rec["fixed_vis_mm"].append(cfv); rec["moving_vis_mm"].append(cmv)
                    rec["delta_mm"].append(-float((tcp - cap) @ dvec) * 1000.0)
                    rec["q5_deg"].append(q5); rec["tilt_deg"].append(float(r["tilt_deg"]))
                    rec["phase"].append(r["phase"])
                else:
                    rec["fixed_mm_MB"].append(cf); rec["moving_mm_MB"].append(cm)
        per_leg[L["leg"]] = rec
        print(f"[{LOG}] leg {L['leg']}: {len(rows)} steps [{STEP_LO}..{L['max_step']}] "
              f"fixed min {min(rec['fixed_mm']):+.4f} max {max(rec['fixed_mm']):+.4f} | "
              f"moving min {min(rec['moving_mm']):+.4f} | {time.time() - t_leg:.0f}s", flush=True)
        for i, s in enumerate(rec["steps"]):
            curves.append([L["leg"], s, rec["phase"][i], f"{rec['q5_deg'][i]:.4f}",
                           f"{rec['delta_mm'][i]:.6f}", f"{rec['tilt_deg'][i]:.4f}",
                           f"{rec['fixed_mm'][i]:.6f}", f"{rec['moving_mm'][i]:.6f}",
                           f"{rec['fixed_mm_MB'][i]:.6f}", f"{rec['moving_mm_MB'][i]:.6f}",
                           f"{rec['fixed_vis_mm'][i]:.6f}", f"{rec['moving_vis_mm'][i]:.6f}",
                           rec["fixed_part"][i], rec["moving_part"][i],
                           rec["fixed_inside"][i], rec["moving_inside"][i]])

    with paths["curves.csv"].open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["leg", "physics_step", "phase", "q5_deg", "delta_mm", "tilt_deg",
                    "clr_fixed_mm_MA", "clr_moving_mm_MA", "clr_fixed_mm_MB", "clr_moving_mm_MB",
                    "clr_fixed_mm_visual", "clr_moving_mm_visual",
                    "fixed_argmin_part", "moving_argmin_part", "fixed_n_inside", "moving_n_inside"])
        w.writerows(curves)

    # ---- preregistered verdicts -------------------------------------------
    preds = []
    for name, leg_id, s_before, s_after, quantity, direction in PREREG_PREDICTIONS:
        rec = per_leg[leg_id]
        key = "fixed_mm" if quantity == "fixed" else "moving_mm"
        got = crossing_step(rec["steps"], rec[key], direction)
        got_mb = crossing_step(rec["steps"], rec[key + "_MB"], direction)
        if got is None:
            code, off = "FAIL_NO_CROSSING", None
        else:
            off = got[0] - s_before
            if got == (s_before, s_after):
                code = "STRICT_PASS"
            elif abs(off) <= STEP_TOL:
                code = "WEAK_PASS"
            else:
                code = "FAIL_OUT_OF_TOLERANCE"
        sensitive = bool(got_mb != got)
        if sensitive and code == "STRICT_PASS":
            code = "STRICT_PASS_ORIENTATION_SENSITIVE"
        preds.append({"name": name, "leg": leg_id, "quantity": quantity, "direction": direction,
                      "predicted_pair": [s_before, s_after], "observed_pair_MA": list(got) if got else None,
                      "observed_pair_MB": list(got_mb) if got_mb else None,
                      "offset_steps": off, "orientation_sensitive": sensitive,
                      "step_tolerance": STEP_TOL, "code": code})
        print(f"[{LOG}] {name} leg{leg_id} {quantity}/{direction} predicted={s_before}->{s_after} "
              f"observed_MA={got} observed_MB={got_mb} => {code}", flush=True)

    strict = all(p["code"] == "STRICT_PASS" for p in preds)
    any_fail = any(p["code"].startswith("FAIL") for p in preds)
    verdict = ("PERSTEP_CLEARANCE_CONFIRMS_ATTRIBUTION" if strict else
               "PERSTEP_CLEARANCE_REFUTES_ATTRIBUTION" if any_fail else
               "PERSTEP_CLEARANCE_PARTIALLY_CONFIRMS_ATTRIBUTION")

    # ---- exploratory (prereg 5, NOT gates) --------------------------------
    def summarise(rec):
        f = np.array(rec["fixed_mm"]); mv = np.array(rec["moving_mm"])
        st = np.array(rec["steps"])
        parts_f = sorted({p for p, v in zip(rec["fixed_part"], rec["fixed_mm"]) if v <= 0.0})
        parts_m = sorted({p for p, v in zip(rec["moving_part"], rec["moving_mm"]) if v <= 0.0})
        return {
            "n_steps": len(st), "step_range": [int(st[0]), int(st[-1])],
            "fixed_min_mm": float(f.min()), "fixed_min_at_step": int(st[int(f.argmin())]),
            "fixed_argmin_part_at_min": rec["fixed_part"][int(f.argmin())],
            "fixed_steps_nonpositive": int((f <= 0.0).sum()),
            "fixed_contact_parts": parts_f,
            "moving_min_mm": float(mv.min()), "moving_min_at_step": int(st[int(mv.argmin())]),
            "moving_argmin_part_at_min": rec["moving_part"][int(mv.argmin())],
            "moving_steps_nonpositive": int((mv <= 0.0).sum()),
            "moving_contact_parts": parts_m,
            "max_abs_MA_minus_MB_fixed_mm": float(np.abs(f - np.array(rec["fixed_mm_MB"])).max()),
            "max_abs_MA_minus_MB_moving_mm": float(np.abs(mv - np.array(rec["moving_mm_MB"])).max()),
            "max_abs_collision_minus_visual_fixed_mm":
                float(np.abs(f - np.array(rec["fixed_vis_mm"])).max()),
            "max_abs_collision_minus_visual_moving_mm":
                float(np.abs(mv - np.array(rec["moving_vis_mm"])).max()),
        }

    out["gates"] = gates
    out["per_leg"] = {str(k): summarise(v) for k, v in per_leg.items()}
    out["preregistered_predictions"] = preds
    out["verdict"] = {"code": verdict,
                      "strict_pass_count": sum(1 for p in preds if p["code"] == "STRICT_PASS"),
                      "meaning": ("clearance < 0 is a NECESSARY condition for closing contact, never "
                                  "sufficient. No force closure, no friction, no contact force. "
                                  "g0a_pass unchanged (false).")}
    out["tail_after_517"] = {
        "why": "prereg A-3: the 44th range 386..517 truncated leg 3, whose last 17 steps carry D435 (6)",
        "leg3_steps_518_534": [
            {"step": s, "fixed_mm": round(per_leg[3]["fixed_mm"][i], 6),
             "moving_mm": round(per_leg[3]["moving_mm"][i], 6),
             "tilt_deg": per_leg[3]["tilt_deg"][i], "phase": per_leg[3]["phase"][i]}
            for i, s in enumerate(per_leg[3]["steps"]) if s >= 518]}

    # ---- D324 diagnostic ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(19.5, 11.0))
    gs = fig.add_gridspec(3, 3, hspace=0.34, wspace=0.22)
    colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
    axf = fig.add_subplot(gs[0, :2])
    axm = fig.add_subplot(gs[1, :2])
    for L, ax, key, ttl in ((None, axf, "fixed_mm", "FIXED jaw (link5) signed clearance"),
                            (None, axm, "moving_mm", "MOVING jaw (gripper_link) signed clearance")):
        for leg_id, rec in per_leg.items():
            ax.plot(rec["steps"], rec[key], "-", lw=1.4, color=colors[leg_id], label=f"leg {leg_id} (M-A)")
            ax.plot(rec["steps"], rec[key + "_MB"], ":", lw=1.0, color=colors[leg_id], alpha=0.8)
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_ylabel("clearance [mm]  (- = penetration)")
        ax.set_title(ttl + "   solid = M-A (planned tool orientation), dotted = M-B (Jacobian recon.)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
    for name, leg_id, s0, s1, q, _dir in PREREG_PREDICTIONS:
        ax = axf if q == "fixed" else axm
        ax.axvline(s0, color=colors[leg_id], ls="--", lw=1.0, alpha=0.7)
        ax.annotate(f"{name}\nleg{leg_id} {s0}->{s1}", xy=(s0, 0.0), xytext=(s0 + 3, 0.0),
                    fontsize=8, color=colors[leg_id])

    axd = fig.add_subplot(gs[2, 0])
    for leg_id, rec in per_leg.items():
        axd.plot(rec["steps"], rec["delta_mm"], color=colors[leg_id], lw=1.2, label=f"leg {leg_id}")
    axd.axhline(0.0, color="k", lw=0.8)
    axd.axhline(delta_cmd * 1000.0, color="grey", ls="--", lw=1.0)
    axd.set_title(f"context: delta [mm] (commanded {delta_cmd * 1000.0:+.4f})")
    axd.set_xlabel("physics step"); axd.grid(alpha=0.3); axd.legend(fontsize=8)

    axt = fig.add_subplot(gs[2, 1])
    for leg_id, rec in per_leg.items():
        axt.plot(rec["steps"], rec["tilt_deg"], color=colors[leg_id], lw=1.2)
    axt.set_title("context: logged object tilt [deg]"); axt.set_xlabel("physics step"); axt.grid(alpha=0.3)

    # D324: target-vs-actual frame markers at the decision step
    axr = fig.add_subplot(gs[:, 2], projection="3d")
    dec_leg, dec_step = PREREG_PREDICTIONS[0][1], PREREG_PREDICTIONS[0][3]
    rec = per_leg[dec_leg]
    di = rec["steps"].index(dec_step)
    rowd = next(r for r in legs[dec_leg - 1]["rows"] if int(r["physics_step"]) == dec_step)
    tcp_a = np.array([float(rowd["tcp_x"]), float(rowd["tcp_y"]), float(rowd["tcp_z"])])
    obj_a = np.array([float(rowd["obj_x"]), float(rowd["obj_y"]), float(rowd["obj_z"])])
    R_obj_a = quat_to_R(float(rowd["quat_w"]), float(rowd["quat_x"]),
                        float(rowd["quat_y"]), float(rowd["quat_z"]))
    tgt = np.array(d3["plan"]["descend_tcp"])
    axr.scatter(*tgt, c="#2ca02c", s=70, marker="^", label="TARGET descend_tcp (plan)")
    axr.scatter(*tcp_a, c="#d62728", s=70, marker="o", label=f"ACTUAL TCP @leg{dec_leg} step {dec_step}")
    axr.scatter(*cap, c="#ff7f0e", s=70, marker="*", label="cap = spawn top-face centre")
    for vec, col, lab in ((dvec, "#2ca02c", "target tool axis d"),
                          (R_TOOL_MA[:, 2], "#9467bd", "M-A tool axis (FK)"),
                          (R_obj_a[:, 2], "#d62728", "ACTUAL object axis")):
        base = tcp_a if lab != "ACTUAL object axis" else obj_a
        axr.quiver(*base, *(vec * 0.02), color=col, lw=2.0, label=lab)
    ang = np.linspace(0, 2 * math.pi, 65)
    e1 = np.cross(R_obj_a[:, 2], [1.0, 0.0, 0.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(R_obj_a[:, 2], e1)
    for k in (-CYL_H / 2.0, CYL_H / 2.0):
        ring = obj_a + R_obj_a[:, 2] * k + CYL_R * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
        axr.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#d62728", lw=1.0)
    axr.set_title(f"D324 target-vs-actual frames\ndecision step: leg {dec_leg} step {dec_step}\n"
                  f"clr_fixed={rec['fixed_mm'][di]:+.4f} mm  clr_moving={rec['moving_mm'][di]:+.4f} mm",
                  fontsize=10)
    axr.legend(fontsize=7, loc="upper left")
    axr.set_xlabel("x [m]"); axr.set_ylabel("y [m]"); axr.set_zlabel("z [m]")
    fig.suptitle(f"g0b_d420 {TAG} — consumed collision asset x measured per-step pose   "
                 f"VERDICT {verdict}", fontsize=13)
    fig.savefig(paths["diagnostic.png"], dpi=112, bbox_inches="tight")
    plt.close(fig)

    # ---- D341 Rerun --------------------------------------------------------
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact
    app_id = f"roarm_g0b_{TAG}"

    def view(pts):
        o = np.asarray(pts) - ja.TCP_LOCAL[None, :]
        return np.column_stack([o[:, 0], o[:, 1], -o[:, 2]])

    frames = [(leg_id, i, s) for leg_id in (1, 2, 3)
              for i, s in enumerate(per_leg[leg_id]["steps"])]
    pin_frame = next(k for k, (lg, i, s) in enumerate(frames)
                     if lg == PREREG_PREDICTIONS[0][1] and s == PREREG_PREDICTIONS[0][3])

    with rr.RecordingStream(app_id, recording_id=f"g0b_d420_{TAG}", make_default=False,
                            send_properties=True) as rec_s:
        rec_s.save(str(paths["timeline.rrd"]), write_footer=True)
        rec_s.log("assembly/link5_collision", rr.Points3D(
            view(C5[::VIEW_STRIDE_FIXED]), colors=[70, 200, 200], radii=0.0004), static=True)
        rec_s.log("assembly/tcp", rr.Points3D([[0.0, 0.0, 0.0]], colors=[[40, 200, 80]],
                                              radii=0.002), static=True)
        rec_s.log("assembly/tool_axis", rr.LineStrips3D(
            [[[0.0, 0.0, 0.06], [0.0, 0.0, -0.06]]], colors=[[40, 200, 80]], radii=0.0003), static=True)
        for k, (leg_id, i, s) in enumerate(frames):
            rc = per_leg[leg_id]
            rec_s.reset_time()
            rec_s.set_time("frame", sequence=k)
            row = next(r for r in legs[leg_id - 1]["rows"] if int(r["physics_step"]) == s)
            tcp = np.array([float(row["tcp_x"]), float(row["tcp_y"]), float(row["tcp_z"])])
            obj = np.array([float(row["obj_x"]), float(row["obj_y"]), float(row["obj_z"])])
            R_obj = quat_to_R(float(row["quat_w"]), float(row["quat_x"]),
                              float(row["quat_y"]), float(row["quat_z"]))
            t_w = tcp - R_TOOL_MA @ ja.TCP_LOCAL
            c_l5 = R_TOOL_MA.T @ (obj - t_w)
            a_l5 = R_TOOL_MA.T @ R_obj[:, 2]
            Tg = gripper_T(rc["q5_deg"][i])
            rec_s.log("assembly/gripper_collision", rr.Points3D(
                view(CG[::VIEW_STRIDE_MOVING] @ Tg[:3, :3].T + Tg[:3, 3]),
                colors=[225, 90, 200], radii=0.0004))
            e1 = np.cross(a_l5, np.array([1.0, 0.0, 0.0]))
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(a_l5, e1)
            strips = []
            for kk in (-CYL_H / 2.0, 0.0, CYL_H / 2.0):
                base = c_l5 + a_l5 * kk
                strips.append(view(base[None, :] + CYL_R * (np.cos(ang)[:, None] * e1[None, :]
                                                            + np.sin(ang)[:, None] * e2[None, :])).tolist())
            for aa2 in np.linspace(0.0, 2 * math.pi, 8, endpoint=False):
                off = CYL_R * (math.cos(aa2) * e1 + math.sin(aa2) * e2)
                strips.append(view(np.array([c_l5 - a_l5 * (CYL_H / 2.0) + off,
                                             c_l5 + a_l5 * (CYL_H / 2.0) + off])).tolist())
            rec_s.log("object/cylinder_measured", rr.LineStrips3D(
                strips, colors=[[225, 60, 60]] * len(strips), radii=0.00035))
            rec_s.log("plots/clr_fixed_mm", rr.Scalars(rc["fixed_mm"][i]))
            rec_s.log("plots/clr_moving_mm", rr.Scalars(rc["moving_mm"][i]))
            rec_s.log("plots/delta_mm", rr.Scalars(rc["delta_mm"][i]))
            rec_s.log("plots/tilt_deg", rr.Scalars(rc["tilt_deg"][i]))
            rec_s.log("plots/q5_deg", rr.Scalars(rc["q5_deg"][i]))
            rec_s.log("plots/leg", rr.Scalars(float(leg_id)))
        rec_s.reset_time()
        rec_s.set_time("frame", sequence=0)
        for name, ok in gates.items():
            rec_s.log("events/gates", rr.TextLog(name, level=rr.TextLogLevel.INFO if ok
                                                 else rr.TextLogLevel.ERROR))
        for p in preds:
            rec_s.log("events/prereg", rr.TextLog(
                f"{p['name']} leg{p['leg']} {p['quantity']}/{p['direction']} "
                f"predicted {p['predicted_pair']} observed {p['observed_pair_MA']} => {p['code']}",
                level=rr.TextLogLevel.INFO if "PASS" in p["code"] else rr.TextLogLevel.ERROR))
        s3 = out["per_leg"]["3"]
        summary_md = (
            f"# g0b_d420 {TAG} — per-step jaw clearance on the CONSUMED asset at the MEASURED pose\n\n"
            f"**VERDICT: {verdict}**\n\n"
            f"## Why this run exists\nD434 / D434-R1 decided *which jaw touched the cylinder and "
            f"when* from one proxy — the TCP depth `delta`. Nothing was measured against geometry. "
            f"D435 reproduced those numbers and then **blocked** the follow-up because the "
            f"preregistration had three defects. `t3d_prereg.md` fixes them (A-1 leg named, A-2 "
            f"judgement by step index, A-3 range 386..max_step per leg) and this run executes it.\n\n"
            f"## Preregistered predictions (judged on M-A)\n"
            + "".join(f"- **{p['name']}** leg {p['leg']} `{p['quantity']}` {p['direction']}: "
                      f"predicted {p['predicted_pair']}, observed **{p['observed_pair_MA']}** "
                      f"=> **{p['code']}**\n" for p in preds)
            + f"\n## Leg 3 (the only leg whose MOVING jaw ever bit)\n"
              f"- fixed jaw minimum **{s3['fixed_min_mm']:+.4f} mm** at step "
              f"**{s3['fixed_min_at_step']}**, contact part `{s3['fixed_argmin_part_at_min']}`;\n"
              f"- moving jaw minimum **{s3['moving_min_mm']:+.4f} mm** at step "
              f"**{s3['moving_min_at_step']}**, contact part `{s3['moving_argmin_part_at_min']}`;\n"
              f"- steps with non-positive clearance: fixed **{s3['fixed_steps_nonpositive']}**, "
              f"moving **{s3['moving_steps_nonpositive']}** of {s3['n_steps']}.\n\n"
              f"## Gates\n"
              + "".join(f"- {k}: **{'PASS' if v else 'FAIL'}**\n" for k, v in gates.items())
            + f"\n## Scene\ncyan = link5 (fixed jaw) attempt3 COLLISION hulls, magenta = "
              f"gripper_link (moving jaw) collision hulls at the **logged** q5, red = the D29 "
              f"cylinder at the **logged** position and quaternion, green = TCP and tool axis. "
              f"Everything is drawn in the link5 frame with `z_view = -(z - z_TCP)`, so DOWN is "
              f"distal. The 3D view is PINNED by blueprint to the P-a decision frame; all "
              f"{len(frames)} frames (leg 1, then leg 2, then leg 3) are in the RRD.\n\n"
              f"## Limits\nA-4: the tool ORIENTATION is not logged — M-A holds it at the planned "
              f"descend orientation, M-B reconstructs it with a minimum-norm Jacobian step. "
              f"Clearance < 0 is NECESSARY for closing contact, never sufficient; this is not a "
              f"grasp-success prediction and carries no contact force or friction. Convex hulls "
              f"are sampled at {ja.SAMPLE_SPACING_M * 1000:.1f} mm, which bounds the resolution. "
              f"Gate-0 was neither re-run nor re-judged; D427/D429/D430/D431/D432/D433/D434/"
              f"D434-R1 unchanged; `g0a_pass=false`.\n\n"
              f"Authority = stdout + `{paths['results.json'].name}`. Rerun is inspection evidence "
              f"only (D341).\n")
        rec_s.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN),
                  static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run",
                                         name="1 | verdict"),
                    rrb.Spatial3DView(
                        origin="/", contents=["/assembly/**", "/object/**"],
                        name=f"2 | PINNED to P-a decision frame (leg "
                             f"{PREREG_PREDICTIONS[0][1]} step {PREREG_PREDICTIONS[0][3]})",
                        time_ranges=rrb.VisibleTimeRange(
                            "frame",
                            start=rrb.TimeRangeBoundary.absolute(seq=pin_frame),
                            end=rrb.TimeRangeBoundary.absolute(seq=pin_frame))),
                    rrb.TextLogView(origin="/events/prereg", contents="/events/prereg/**",
                                    name="3 | preregistered verdicts"),
                    column_shares=[0.30, 0.46, 0.24]),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/clr_fixed_mm/**", "/plots/clr_moving_mm/**",
                                                 "/plots/delta_mm/**"],
                                       name="4 | signed clearance of both jaws vs the delta proxy"),
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/q5_deg/**", "/plots/tilt_deg/**",
                                                 "/plots/leg/**"],
                                       name="5 | closing angle, object tilt, leg index")),
                row_shares=[0.58, 0.42]),
            auto_layout=False, auto_views=False, collapse_panels=True)
        rec_s.send_blueprint(blueprint, make_active=True, make_default=True)
        rec_s.flush(timeout_sec=60.0)
    blueprint.save(app_id, str(paths["timeline.rbl"]))

    expected_entities = ["metadata/run", "assembly/link5_collision", "assembly/gripper_collision",
                         "assembly/tcp", "assembly/tool_axis", "object/cylinder_measured",
                         "plots/clr_fixed_mm", "plots/clr_moving_mm", "plots/delta_mm",
                         "plots/tilt_deg", "plots/q5_deg", "plots/leg",
                         "events/gates", "events/prereg"]
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    lin3 = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    sca = ["Scalars:scalars"]
    tlog = ["TextLog:text", "TextLog:level"]
    components = {"metadata/run": ["TextDocument:text"],
                  "assembly/link5_collision": pts3, "assembly/gripper_collision": pts3,
                  "assembly/tcp": pts3, "assembly/tool_axis": lin3,
                  "object/cylinder_measured": lin3,
                  "plots/clr_fixed_mm": sca, "plots/clr_moving_mm": sca, "plots/delta_mm": sca,
                  "plots/tilt_deg": sca, "plots/q5_deg": sca, "plots/leg": sca,
                  "events/gates": tlog, "events/prereg": tlog}
    validation = validate_rerun_artifact(
        paths["timeline.rrd"],
        expected_entity_paths=expected_entities, exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time", "frame"],
        expected_entity_components=components,
        blueprint_path=paths["timeline.rbl"], screenshot_path=paths["inspection.png"],
        screenshot_window_size="2400x1400", expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI, timeout_s=300.0)
    paths["rerun_validation.json"].write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(f"[{LOG}] rerun_validation pass={validation.get('pass')} "
          f"errors={validation.get('errors')}", flush=True)

    shutil.copyfile(__file__, paths["script.py.txt"])
    out["rerun_validation_pass"] = bool(validation.get("pass"))
    out["artifacts"] = {k: {"name": v.name, "sha256_16": sha256(v)[:16], "bytes": v.stat().st_size}
                        for k, v in paths.items() if v.exists() and k != "results.json"}
    out["artifacts_note"] = "results.json is deliberately absent from this manifest (D429-R1)."
    out["wall_seconds"] = round(time.time() - t_start, 1)
    paths["results.json"].write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{LOG}] artifacts " + " ".join(f"{v['name']}={v['sha256_16']}"
                                           for v in out["artifacts"].values()), flush=True)
    print(f"[{LOG}] results.json={sha256(paths['results.json'])[:16]} "
          f"bytes={paths['results.json'].stat().st_size}", flush=True)
    print(f"[{LOG}] G0B_T3D_PERSTEP_VERDICT={verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
