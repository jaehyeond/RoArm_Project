#!/usr/bin/env python3
"""G0b T3R N9 - tilted tool-axis IK reachability + table clearance (read-only).

Case g0b_d420.  Numpy-only kinematics + rigid-body geometry.  NO Isaac launch, NO robot,
NO training, NO package install, NO Gate-0 re-run, NO overwrite of any frozen artifact.
New outputs only, tag `t3r_n9_tiltik`, under claudedocs/runtime_logs/grasp_track/g0b_d420/.

WHY THIS RUN EXISTS
-------------------
41st (D431) showed the D29 cylinder is admitted (bite > 0) once the tool axis is tilted:
theta_min <= 6 deg, +12.11 mm at 29 deg.  But that run measured geometry in the TOOL frame
and said so explicitly - `t3r_n8b_tiltmin_results.json` /verdict/tilted_pose_ik_reachability
= "NOT ESTABLISHED - T2/T2b tested the vertical axis only".  START_HERE.md marks the tilted
IK grid as the REQUIRED PRECURSOR (Next-session item 3-(1)) before any T3 physics restart.

Two questions, both able to fail and either answer changes the decision:
  Q1  Can this 5-DOF arm put the TCP at the pinned grasp target with the tool axis tilted by
      theta, and can the wrist roll (q4) deliver the tool-frame azimuth phi that D431's bite
      numbers were computed at?   If NO -> the tilt request to the professor is moot and the
      fallback is D <= 16 mm (D430 F2).
  Q2  Does the tilted jaw clear the support surface?  41st listed "table not modelled" as an
      open limit; a pose that reaches but ploughs the fixed jaw into the ground is useless.

WORLD CONVENTION (read from the frozen record, not from memory)
--------------------------------------------------------------
T3 spawns on the ground plane: t3_prereg.md:176-179 "wp006(target z=0.0505 = top+0.5mm)",
"원통 상면(z=0.050)", "수직 압입은 지면이 지지".  That is exactly the T2b annex convention
(z_offset +0.012117 on top of T2's TABLE_Z).  So the DECISION sweep runs at support z = 0,
cap centre z = 0.050, while the reproduction gate N9c runs each reference in its OWN frame.

REDUCTION TO T2 (the honesty gate)
----------------------------------
The tilted task error uses an orthonormal pair (u, v) perpendicular to the target axis d:
    d = ( sin th cos psi,  sin th sin psi, -cos th )
    u = ( cos th cos psi,  cos th sin psi,  sin th )
    v = (       -sin psi,        cos psi,      0   )
At th = 0, psi = 0 this is d=(0,0,-1), u=(1,0,0), v=(0,1,0) and the error vector becomes
T2's own (-w*axis_x, -w*axis_y) term for term.  N9c therefore demands EXACT equality with the
frozen t2_ik_results.json / t2b_ik_results.json named-pose numbers (tol 0.0).

PREREGISTERED GATES (abort on failure, no verdict)
  N9a  asset + reference sha pins (link5.stl, gripper_link.stl, t3r_n8b_tiltmin_results.json)
  N9b  full-rotation FK agrees with the frozen T2 fk_points to 0.0 (TCP and link5 origin)
  N9c  theta=0 reproduces BOTH t2 and t2b named-pose pos_err/tilt exactly (tol 0.0)
  N9d  moving-jaw z range in link5 frame = [41.2676, 119.1176] mm (40th N8d, tol 0.02)
  N9e  psi-invariance at theta=0: solving with any psi gives the same result (tol 1e-9)
  N9f  wrist-roll closed form verifies: re-running FK at the solved q4 hits phi* (tol 1e-6)
CLAIMS UNDER TEST (may fail without aborting - that IS the result)
  N9g  theta = 6 deg (D431 theta_min) reachable at the pinned spawn, both z, URDF limits
  N9h  theta = 17 deg (T1 band centre) reachable at the pinned spawn, both z, URDF limits
  N9i  the reachable poses clear the support plane

NOT CLAIMED: force closure, grasp success, contact physics, collision-USD fidelity (this is
the visual mesh, as in 40th/41st), servo torque, real-robot reproduction.  g0a_pass stays
false.  D427 / D429 / D430 / D431 are neither re-run nor re-judged.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "sim_scripts"))

OUT_DIR = ROOT / "claudedocs/runtime_logs/grasp_track/g0b_d420"
TAG = "t3r_n9_tiltik"
LOG = "g0b_t3r_n9"

T2_SRC = ROOT / "sim_scripts/p8_g0b_t2_cyld29h50_vertical_tool_axis_ik_reachability_probe.py"
N8_SRC = ROOT / "sim_scripts/g0b_t3r_n8_tilt_admission_readonly_audit.py"
T2_REF = OUT_DIR / "t2_ik_results.json"
T2B_REF = OUT_DIR / "t2b_ik_results.json"
N8B_REF = OUT_DIR / "t3r_n8b_tiltmin_results.json"
N8B_REF_SHA16 = "180e03734544c894"          # START_HERE.md pin, re-hashed from disk here

RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"

# --- declared scoping (no silent caps) -------------------------------------------------
LADDER_THETA_DEG = (0.0, 3.0, 6.0, 10.0, 15.0, 17.0, 20.0, 24.0, 29.0, 35.0)
PSI_DEG = tuple(float(v) for v in range(0, 360, 15))        # 24 world azimuths of the tilt
GRID_THETA_DEG = 17.0                                      # T1 band centre, bite +5.39 mm
GRID_PSI_OFFSETS_MAX = 2                                   # best-2 in-plane offsets from the pin
PIN_POSE = "seed0_S1"
AXIS_GATE_DEG = 5.0                                        # same primary gate as T2
SUPPORT_Z_T3 = 0.0                                         # t3_prereg.md:176-179 (ground plane)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def axis_frame(theta_deg: float, psi_deg: float):
    """Target tool axis d (link5 +z direction in world) and an orthonormal pair (u, v)."""
    th, ps = math.radians(theta_deg), math.radians(psi_deg)
    st, ct, cp, sp = math.sin(th), math.cos(th), math.cos(ps), math.sin(ps)
    d = np.array([st * cp, st * sp, -ct])
    u = np.array([ct * cp, ct * sp, st])
    v = np.array([-sp, cp, 0.0])
    return d, u, v


def main() -> int:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {k: OUT_DIR / f"{TAG}_{k}" for k in
             ("results.json", "timeline.rrd", "timeline.rbl", "rerun_validation.json",
              "inspection.png", "diagnostic.png", "script.py.txt", "grid.csv")}
    existing = [p.name for p in paths.values() if p.exists()]
    if existing:
        print(f"[{LOG}] ABORT write_guard existing={existing}", flush=True)
        return 3
    for req in (T2_SRC, N8_SRC, T2_REF, T2B_REF, N8B_REF):
        if not req.exists():
            print(f"[{LOG}] ABORT missing_reference={req}", flush=True)
            return 3

    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        print(f"[{LOG}] ABORT rerun_version={rr.__version__}!={RERUN_VERSION}", flush=True)
        return 3

    t2 = load_module(T2_SRC, "g0b_t2_frozen")
    n8 = load_module(N8_SRC, "g0b_n8_frozen")
    from roarm_kinematics import _CHAIN, Tmat, Trot_z          # noqa: E402

    out = {
        "tool": TAG,
        "read_only_assets": True,
        "claim_under_test": ("the tilted approach that D431 showed admits the D29 cylinder is "
                             "actually reachable by this 5-DOF arm at the pinned T3 spawn, with "
                             "the wrist roll able to deliver D431's tool-frame azimuth, and "
                             "without the jaw entering the support plane"),
        "permitted_by": ("START_HERE.md next-session item 3-(1): tilted-target T2-class IK grid "
                         "is the required precursor, new tag, frozen assets untouched"),
        "env": {"numpy": np.__version__, "rerun_sdk": rr.__version__,
                "python": sys.version.split()[0]},
        "imported_verbatim_from": {
            "t2": {"path": str(T2_SRC.relative_to(ROOT)), "sha256_16": sha256(T2_SRC)[:16]},
            "n8": {"path": str(N8_SRC.relative_to(ROOT)), "sha256_16": sha256(N8_SRC)[:16]},
        },
        "scoping_declared": {
            "theta_ladder_deg": list(LADDER_THETA_DEG),
            "psi_deg_count": len(PSI_DEG), "psi_step_deg": 15.0,
            "pin_pose": PIN_POSE,
            "grid_theta_deg": GRID_THETA_DEG,
            "grid_psi_offsets_max": GRID_PSI_OFFSETS_MAX,
            "note": ("the (theta, psi) sweep is exhaustive at the pinned spawn only; the (x, y) "
                     "table grid is run at ONE theta (T1 band centre) with the best-2 in-plane "
                     "psi offsets carried over from the pin. The full theta x psi x cell product "
                     "is NOT swept - declared, not silently capped."),
            "world": ("decision sweep uses the T3/T2b convention (support z = 0, cap z = 0.050) "
                      "read from t3_prereg.md:176-179; gate N9c reproduces each frozen reference "
                      "in its own frame"),
        },
    }

    # ---- N9a: asset + reference identity ------------------------------------------
    out["sha256"] = {p.name: sha256(p) for p in (n8.L5, n8.GJ)}
    out["sha256"][N8B_REF.name] = sha256(N8B_REF)
    sha_ok = all(out["sha256"][k] == v for k, v in n8.SHA.items())
    n8b_ok = out["sha256"][N8B_REF.name][:16] == N8B_REF_SHA16
    g_a = bool(sha_ok and n8b_ok)
    out["N9a_sha_pins"] = {"pass": g_a, "meshes_match_record": sha_ok,
                           "n8b_results_matches_start_here_pin": n8b_ok,
                           "n8b_sha256_16": out["sha256"][N8B_REF.name][:16]}
    print(f"[{LOG}] N9a sha_ok={sha_ok} n8b_pin_ok={n8b_ok}", flush=True)

    ref_t2 = json.loads(T2_REF.read_text())
    ref_t2b = json.loads(T2B_REF.read_text())
    ref_n8b = json.loads(N8B_REF.read_text())
    ladder_ref = {round(float(r["theta_deg"]), 6): r for r in ref_n8b["theta_ladder_full_q5"]}

    # ---- full-rotation FK (extends the frozen fk_points with the link5 rotation) ----
    def fk_full(q5v_deg):
        """q = [q0..q4] deg -> (T_tcp 4x4, T_link5 4x4). q4 = wrist roll about the tool axis."""
        q = np.radians(np.array([q5v_deg[0], q5v_deg[1], q5v_deg[2], q5v_deg[3],
                                 q5v_deg[4], 0.0], dtype=np.float64))
        T = np.eye(4)
        T5 = None
        for name, xyz, rpy, qi in _CHAIN:
            T = T @ Tmat(xyz, rpy)
            if qi is not None:
                T = T @ Trot_z(q[qi])
            if name == "link4_to_link5":
                T5 = T.copy()
        return T, T5

    # ---- N9b: FK agreement with the frozen probe -----------------------------------
    probe_q = [np.array([0.0, 0.0, 90.0, 0.0]), t2.SELF_CHECK_Q_DEG.copy(),
               np.array([-42.4858, 41.4063, 78.6095, 59.7854]), np.array([30.0, -20.0, 120.0, -25.0])]
    d_tcp = d_l5 = 0.0
    for q in probe_q:
        tcp_ref, l5_ref, _ = t2.fk_points(q)
        T, T5 = fk_full(np.concatenate([q, [0.0]]))
        d_tcp = max(d_tcp, float(np.abs(T[:3, 3] - tcp_ref).max()))
        d_l5 = max(d_l5, float(np.abs(T5[:3, 3] - l5_ref).max()))
    g_b = (d_tcp == 0.0 and d_l5 == 0.0)
    out["N9b_fk_matches_frozen"] = {"pass": g_b, "max_abs_d_tcp_m": d_tcp,
                                    "max_abs_d_link5_m": d_l5, "n_probe_configs": len(probe_q)}
    print(f"[{LOG}] N9b fk d_tcp={d_tcp:.3e} d_link5={d_l5:.3e}", flush=True)

    # ---- tilted DLS: T2's solver, line for line, with a general target axis ---------
    def task_error_tilt(q4_deg, target_p, u, v, w_axis):
        tcp, _tilt, axis = t2.axis_tilt(q4_deg)
        return np.array([target_p[0] - tcp[0], target_p[1] - tcp[1], target_p[2] - tcp[2],
                         -w_axis * float(axis @ u), -w_axis * float(axis @ v)], dtype=np.float64)

    def axis_err_deg(q4_deg, d):
        _tcp, _t, axis = t2.axis_tilt(q4_deg)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(axis @ d)))))

    def dls_tilt(target_p, d, u, v, seed4_deg, limits, max_iter=160, w_axis=0.03,
                 damping=0.002, step_clip_deg=4.0, eps_deg=0.05):
        q = t2.clip4(np.asarray(seed4_deg, dtype=np.float64).copy(), limits)
        best = None
        for it in range(max_iter):
            e = task_error_tilt(q, target_p, u, v, w_axis)
            tcp, _tilt, _axis = t2.axis_tilt(q)
            ae = axis_err_deg(q, d)
            pos_err_mm = float(np.linalg.norm(target_p - tcp)) * 1000.0
            key = (pos_err_mm > t2.POS_GATE_MM,
                   ae if pos_err_mm <= t2.POS_GATE_MM else 1.0e9, pos_err_mm)
            if best is None or key < best[0]:
                best = (key, q.copy(), pos_err_mm, ae, it)
            if pos_err_mm < 0.2 and ae < 0.2:
                break
            J = np.zeros((5, 4), dtype=np.float64)
            for i in range(4):
                qp = q.copy(); qp[i] += eps_deg
                qm = q.copy(); qm[i] -= eps_deg
                J[:, i] = (task_error_tilt(qp, target_p, u, v, w_axis)
                           - task_error_tilt(qm, target_p, u, v, w_axis)) / (2.0 * eps_deg)
            M = J @ J.T + (damping ** 2) * np.eye(5)
            try:
                dq = -J.T @ np.linalg.solve(M, e)
            except np.linalg.LinAlgError:
                break
            mx = float(np.max(np.abs(dq)))
            if mx > step_clip_deg:
                dq = dq * (step_clip_deg / mx)
            q = t2.clip4(q + dq, limits)
        _key, qb, pe, ae, it_used = best
        return qb, pe, ae, it_used

    def solve_tilt(x, y, target_p, theta_deg, psi_deg, limits):
        d, u, v = axis_frame(theta_deg, psi_deg)
        best = None
        for si, seed in enumerate(t2.seeds_for(float(x), float(y))):
            q, pe, ae, it_used = dls_tilt(np.asarray(target_p, dtype=np.float64), d, u, v,
                                          seed, limits)
            key = (pe > t2.POS_GATE_MM, ae if pe <= t2.POS_GATE_MM else 1.0e9, pe)
            if best is None or key < best["_key"]:
                best = {"_key": key, "q_deg": [round(float(w), 4) for w in q],
                        "pos_err_mm": round(pe, 4), "axis_err_deg": round(ae, 4),
                        "seed_idx": si, "iters": it_used}
            if pe < 0.3 and ae < 0.5:
                break
        best.pop("_key")
        best["pos_ok"] = best["pos_err_mm"] <= t2.POS_GATE_MM
        best["axis_ok"] = bool(best["pos_ok"] and best["axis_err_deg"] <= AXIS_GATE_DEG)
        return best

    # ---- N9c: theta = 0 reproduces the two frozen references EXACTLY ---------------
    def repro_against(ref, z_off):
        worst_pos = worst_axis = 0.0
        for name, (nx, ny) in t2.NAMED_POSES.items():
            for zn, z in (("descend", t2.DESCEND_Z + z_off), ("approach", t2.APPROACH_Z + z_off)):
                mine = solve_tilt(nx, ny, [nx, ny, z], 0.0, 0.0, t2.URDF_LIMITS_DEG)
                r = ref["named"][name][zn]["urdf"]
                worst_pos = max(worst_pos, abs(mine["pos_err_mm"] - r["pos_err_mm"]))
                worst_axis = max(worst_axis, abs(mine["axis_err_deg"] - r["tilt_deg"]))
        return worst_pos, worst_axis

    wp2, wa2 = repro_against(ref_t2, 0.0)
    wp2b, wa2b = repro_against(ref_t2b, 0.012117)
    g_c = (wp2 == 0.0 and wa2 == 0.0 and wp2b == 0.0 and wa2b == 0.0)
    out["N9c_theta0_reproduces_T2_and_T2b"] = {
        "pass": g_c, "tol_exact": 0.0,
        "t2": {"max_abs_d_pos_mm": wp2, "max_abs_d_axis_deg": wa2, "verdict_ref": ref_t2["verdict"]},
        "t2b": {"max_abs_d_pos_mm": wp2b, "max_abs_d_axis_deg": wa2b,
                "verdict_ref": ref_t2b["verdict"]},
        "n_poses": len(t2.NAMED_POSES) * 2}
    print(f"[{LOG}] N9c repro t2 dpos={wp2:.3e} daxis={wa2:.3e} | "
          f"t2b dpos={wp2b:.3e} daxis={wa2b:.3e}", flush=True)

    # ---- N9e: psi-invariance at theta = 0 ------------------------------------------
    nx, ny = t2.NAMED_POSES[PIN_POSE]
    z_pin = SUPPORT_Z_T3 + n8.CYL_H_MM / 1000.0
    base0 = solve_tilt(nx, ny, [nx, ny, z_pin], 0.0, 0.0, t2.URDF_LIMITS_DEG)
    spread_pos = spread_axis = 0.0
    for ps in (37.0, 123.0, 271.0):
        r = solve_tilt(nx, ny, [nx, ny, z_pin], 0.0, ps, t2.URDF_LIMITS_DEG)
        spread_pos = max(spread_pos, abs(r["pos_err_mm"] - base0["pos_err_mm"]))
        spread_axis = max(spread_axis, abs(r["axis_err_deg"] - base0["axis_err_deg"]))
    g_e = (spread_pos <= 1e-9 and spread_axis <= 1e-9)
    out["N9e_psi_invariant_at_theta0"] = {"pass": g_e, "spread_pos_mm": spread_pos,
                                          "spread_axis_deg": spread_axis, "tol": 1e-9}
    print(f"[{LOG}] N9e psi-invariance spread pos={spread_pos:.3e} axis={spread_axis:.3e}",
          flush=True)

    # ---- jaw geometry in the link5 frame (40th / 41st numeric path) ----------------
    u_urdf = n8.parse_urdf(n8.URDF)
    j5 = u_urdf["joints"]["link5_to_gripper_link"]
    R_j5 = n8.rpy_matrix(*j5["rpy"])
    t_j5 = np.array(j5["xyz"])
    V5 = n8.load_binary_stl(n8.L5).reshape(-1, 3) * n8.STL_SCALE
    VG = n8.load_binary_stl(n8.GJ).reshape(-1, 3) * n8.STL_SCALE

    def to_link5(P, q5_rad):
        c, s = math.cos(q5_rad), math.sin(q5_rad)
        Rq = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return P @ (R_j5 @ Rq).T + t_j5

    zq0 = to_link5(VG, 0.0)[:, 2] * 1000.0
    g_d = (abs(float(zq0.min()) - n8.MOVING_Z_RANGE_37TH_MM[0]) < 0.02
           and abs(float(zq0.max()) - n8.MOVING_Z_RANGE_37TH_MM[1]) < 0.02)
    out["N9d_moving_jaw_z_range"] = {"pass": g_d, "z_min_mm": round(float(zq0.min()), 6),
                                     "z_max_mm": round(float(zq0.max()), 6),
                                     "reference_mm": list(n8.MOVING_Z_RANGE_37TH_MM),
                                     "note": ("min/max of a linear functional is attained at a "
                                              "vertex, so raw vertices are exact here")}
    print(f"[{LOG}] N9d moving-jaw z=[{zq0.min():.4f},{zq0.max():.4f}] pass={g_d}", flush=True)

    if not (g_a and g_b and g_c and g_d and g_e):
        out["verdict"] = {"code": "GATE_FAILURE_N9a_TO_N9e"}
        paths["results.json"].write_text(json.dumps(out, indent=2, default=str) + "\n")
        print(f"[{LOG}] ABORT gate_failure a={g_a} b={g_b} c={g_c} d={g_d} e={g_e}", flush=True)
        return 3

    # ---- wrist roll: closed form for the tool-frame azimuth ------------------------
    CHAT_WORLD = np.array([0.0, 0.0, -1.0])      # cylinder body extends DOWN from its top cap

    def phi_at(q5v_deg):
        _T, T5 = fk_full(q5v_deg)
        R5 = T5[:3, :3]
        c = R5.T @ CHAT_WORLD
        return math.degrees(math.atan2(float(c[1]), float(c[0]))) % 360.0, R5, T5[:3, 3]

    def solve_q4(q4_deg, phi_target_deg):
        """phi(alpha) = phi0 - alpha  ->  alpha = phi0 - phi*."""
        phi0, _R0, _o = phi_at(np.concatenate([q4_deg, [0.0]]))
        alpha = wrap180(phi0 - phi_target_deg)
        phi_chk, R5, o5 = phi_at(np.concatenate([q4_deg, [alpha]]))
        err = abs(wrap180(phi_chk - phi_target_deg))
        return alpha, err, R5, o5

    def jaw_min_world_z(R5, o5, q5_rad):
        pts = np.vstack([V5, to_link5(VG, q5_rad)])
        return float((pts @ R5.T + o5)[:, 2].min())

    # ---- Sweep A: the pinned spawn, exhaustive over (theta, psi) -------------------
    print(f"[{LOG}] sweep A pin={PIN_POSE} xy=({nx:+.6f},{ny:+.6f}) cap_z={z_pin:.6f} "
          f"theta={list(LADDER_THETA_DEG)} psi_n={len(PSI_DEG)}", flush=True)
    pin_rows = []
    n9f_worst = 0.0
    for th in LADDER_THETA_DEG:
        lref = ladder_ref[round(float(th), 6)]
        delta_m = float(lref["delta_m"])
        phi_star = float(lref["phi_deg"])
        q5_star_rad = math.radians(float(lref["q5_deg"]))
        cap = np.array([nx, ny, z_pin])
        for ps in PSI_DEG:
            d, _u, _v = axis_frame(th, ps)
            tgt = {"descend": cap - delta_m * d,
                   "approach": cap - (delta_m + t2.APPROACH_CLEARANCE_M) * d}
            rec = {"theta_deg": float(th), "psi_deg": float(ps), "phi_star_deg": phi_star,
                   "q5_star_deg": float(lref["q5_deg"]), "delta_mm": delta_m * 1000.0,
                   "bite_mm_D431": float(lref["bite_mm"])}
            for zn in ("descend", "approach"):
                r_u = solve_tilt(nx, ny, tgt[zn], th, ps, t2.URDF_LIMITS_DEG)
                r_v = (solve_tilt(nx, ny, tgt[zn], th, ps, t2.V6_LIMITS_DEG)
                       if r_u["pos_ok"] else
                       {"pos_err_mm": None, "axis_err_deg": None, "pos_ok": False,
                        "axis_ok": False, "q_deg": None, "seed_idx": None, "iters": 0})
                rec[zn] = {"urdf": r_u, "v6clip": r_v, "target_m": [float(w) for w in tgt[zn]]}
                if r_u["axis_ok"]:
                    a, err, R5, o5 = solve_q4(np.array(r_u["q_deg"]), phi_star)
                    n9f_worst = max(n9f_worst, err)
                    mz = jaw_min_world_z(R5, o5, q5_star_rad if zn == "descend" else 0.0)
                    rec[zn]["q4_required_deg"] = round(float(a), 4)
                    rec[zn]["q4_within_urdf"] = bool(abs(a) <= 180.0)
                    rec[zn]["q4_within_v6clip"] = bool(abs(a) <= 90.0)
                    rec[zn]["jaw_min_world_z_m"] = round(mz, 6)
                    rec[zn]["support_clearance_mm"] = round((mz - SUPPORT_Z_T3) * 1000.0, 4)
                    rec[zn]["ok_full"] = bool(rec[zn]["q4_within_urdf"] and mz > SUPPORT_Z_T3)
                else:
                    rec[zn]["ok_full"] = False
            rec["pass_urdf_both_z"] = bool(rec["descend"]["ok_full"] and rec["approach"]["ok_full"])
            pin_rows.append(rec)
        best = [r for r in pin_rows if r["theta_deg"] == float(th) and r["pass_urdf_both_z"]]
        print(f"[{LOG}] pin theta={th:5.1f} phi*={phi_star:5.1f} pass_psi={len(best)}/{len(PSI_DEG)}"
              + (f" best_psi={min(b['psi_deg'] for b in best):.0f}" if best else ""), flush=True)

    g_f = n9f_worst <= 1e-6
    out["N9f_wrist_roll_closed_form"] = {"pass": g_f, "max_abs_phi_residual_deg": n9f_worst,
                                         "tol_deg": 1e-6}

    def pin_pass(th):
        return [r for r in pin_rows if r["theta_deg"] == float(th) and r["pass_urdf_both_z"]]

    theta_pass = [float(th) for th in LADDER_THETA_DEG if pin_pass(th)]
    g_g = 6.0 in theta_pass
    g_h = 17.0 in theta_pass
    clearances = [r[zn]["support_clearance_mm"] for r in pin_rows for zn in ("descend", "approach")
                  if r[zn].get("support_clearance_mm") is not None]
    g_i = bool(clearances) and min(clearances) > 0.0

    out["N9g_theta6_reachable_at_pin"] = {"pass": bool(g_g)}
    out["N9h_theta17_reachable_at_pin"] = {"pass": bool(g_h)}
    out["N9i_support_clearance_positive"] = {
        "pass": bool(g_i),
        "min_clearance_mm": round(min(clearances), 4) if clearances else None,
        "n_evaluated": len(clearances)}

    # ---- Sweep C: table grid at one theta -----------------------------------------
    grid_rows = []
    psi_offsets = []
    if pin_pass(GRID_THETA_DEG):
        az_pin = math.degrees(math.atan2(ny, nx))
        cand = sorted(pin_pass(GRID_THETA_DEG),
                      key=lambda r: r["descend"]["urdf"]["axis_err_deg"])
        for r in cand:
            off = round(wrap180(r["psi_deg"] - az_pin), 4)
            if all(abs(wrap180(off - o)) > 1e-6 for o in psi_offsets):
                psi_offsets.append(off)
            if len(psi_offsets) >= GRID_PSI_OFFSETS_MAX:
                break
        lref = ladder_ref[round(float(GRID_THETA_DEG), 6)]
        delta_m = float(lref["delta_m"])
        print(f"[{LOG}] sweep C grid theta={GRID_THETA_DEG} psi_offsets={psi_offsets} "
              f"cells={len(t2.GRID_X) * len(t2.GRID_Y)}", flush=True)
        n_done = 0
        for gx in t2.GRID_X:
            for gy in t2.GRID_Y:
                az = math.degrees(math.atan2(float(gy), float(gx)))
                cap = np.array([float(gx), float(gy), z_pin])
                best_cell = None
                for off in psi_offsets:
                    ps = (az + off) % 360.0
                    d, _u, _v = axis_frame(GRID_THETA_DEG, ps)
                    ok_both, worst, qs = True, 0.0, {}
                    for zn, dd in (("descend", delta_m),
                                   ("approach", delta_m + t2.APPROACH_CLEARANCE_M)):
                        r_u = solve_tilt(gx, gy, cap - dd * d, GRID_THETA_DEG, ps,
                                         t2.URDF_LIMITS_DEG)
                        qs[zn] = r_u
                        ok_both = ok_both and r_u["axis_ok"]
                        worst = max(worst, r_u["axis_err_deg"])
                    cand_cell = {"psi_deg": round(ps, 3), "psi_offset_deg": off,
                                 "axis_ok_both": bool(ok_both), "worst_axis_err_deg": round(worst, 4),
                                 "descend": qs["descend"], "approach": qs["approach"]}
                    if best_cell is None or (cand_cell["axis_ok_both"], -worst) > \
                            (best_cell["axis_ok_both"], -best_cell["worst_axis_err_deg"]):
                        best_cell = cand_cell
                    if ok_both:
                        break
                if best_cell["axis_ok_both"]:
                    a, err, R5, o5 = solve_q4(np.array(best_cell["descend"]["q_deg"]),
                                              float(lref["phi_deg"]))
                    mz = jaw_min_world_z(R5, o5, math.radians(float(lref["q5_deg"])))
                    best_cell["q4_required_deg"] = round(float(a), 4)
                    best_cell["support_clearance_mm"] = round((mz - SUPPORT_Z_T3) * 1000.0, 4)
                    best_cell["cell_pass"] = bool(abs(a) <= 180.0 and mz > SUPPORT_Z_T3)
                else:
                    best_cell["cell_pass"] = False
                grid_rows.append({"x": float(gx), "y": float(gy), **best_cell})
                n_done += 1
                if n_done % 60 == 0:
                    print(f"[{LOG}] sweep C progress {n_done}/{len(t2.GRID_X) * len(t2.GRID_Y)}",
                          flush=True)
    else:
        print(f"[{LOG}] sweep C SKIPPED - theta={GRID_THETA_DEG} did not pass at the pin", flush=True)

    grid_pass = [g for g in grid_rows if g["cell_pass"]]
    t2b_vertical_cells = {(round(a, 3), round(b, 3))
                          for a, b in ref_t2b["grid"]["pass_urdf_cells"]}
    tilt_cells = {(round(g["x"], 3), round(g["y"], 3)) for g in grid_pass}

    # ---- verdict ------------------------------------------------------------------
    if g_g and g_h:
        code = "TILTED_IK_REACHABLE"
    elif theta_pass:
        code = "TILTED_IK_PARTIAL"
    else:
        code = "TILTED_IK_UNREACHABLE"

    theta_min_pass = min(theta_pass) if theta_pass else None
    theta_max_pass = max(theta_pass) if theta_pass else None
    out["pin_sweep"] = pin_rows
    out["grid_sweep"] = {
        "theta_deg": GRID_THETA_DEG, "psi_offsets_deg": psi_offsets,
        "n_cells": len(grid_rows), "n_pass": len(grid_pass),
        "n_pass_t2b_vertical_reference": len(t2b_vertical_cells),
        "n_pass_in_both": len(tilt_cells & t2b_vertical_cells),
        "n_pass_tilt_only": len(tilt_cells - t2b_vertical_cells),
        "n_vertical_only": len(t2b_vertical_cells - tilt_cells),
        "cells": grid_rows,
    }
    out["verdict"] = {
        "code": code,
        "theta_pass_at_pin_deg": theta_pass,
        "theta_min_pass_deg": theta_min_pass,
        "theta_max_pass_deg": theta_max_pass,
        "pin_pose": PIN_POSE, "pin_xy": [nx, ny],
        "D431_theta_min_upper_bound_deg": ref_n8b["verdict"]["theta_min_deg_upper_bound"],
        "min_support_clearance_mm": out["N9i_support_clearance_positive"]["min_clearance_mm"],
        "grid_pass_cells_at_theta17": len(grid_pass),
        "D427_D429_D430_D431_status": "UNCHANGED - none re-run, none re-judged",
        "force_closure": "NOT ESTABLISHED - static admission + kinematics only",
        "collision_usd_fidelity": "NOT ESTABLISHED - visual mesh, as in 40th/41st",
        "g0a_pass": False,
    }
    out["gates"] = {"N9a": g_a, "N9b": g_b, "N9c": g_c, "N9d": g_d, "N9e": g_e,
                    "N9f": g_f, "N9g": bool(g_g), "N9h": bool(g_h), "N9i": bool(g_i)}
    print(f"[{LOG}] N9f q4 residual={n9f_worst:.3e} N9g={g_g} N9h={g_h} N9i={g_i} "
          f"min_clear={out['N9i_support_clearance_positive']['min_clearance_mm']}", flush=True)
    print(f"[{LOG}] G0B_T3R_N9_VERDICT={code} theta_pass={theta_pass}", flush=True)

    # ---- CSV ----------------------------------------------------------------------
    import csv
    with paths["grid.csv"].open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "theta_deg", "psi_deg", "x", "y", "z_name", "pos_err_mm",
                    "axis_err_deg", "pos_ok", "axis_ok", "q4_required_deg",
                    "support_clearance_mm", "pass"])
        for r in pin_rows:
            for zn in ("descend", "approach"):
                z = r[zn]
                w.writerow(["pin", r["theta_deg"], r["psi_deg"], nx, ny, zn,
                            z["urdf"]["pos_err_mm"], z["urdf"]["axis_err_deg"],
                            z["urdf"]["pos_ok"], z["urdf"]["axis_ok"],
                            z.get("q4_required_deg"), z.get("support_clearance_mm"),
                            z["ok_full"]])
        for g in grid_rows:
            for zn in ("descend", "approach"):
                w.writerow(["grid", GRID_THETA_DEG, g["psi_deg"], g["x"], g["y"], zn,
                            g[zn]["pos_err_mm"], g[zn]["axis_err_deg"], g[zn]["pos_ok"],
                            g[zn]["axis_ok"], g.get("q4_required_deg"),
                            g.get("support_clearance_mm"), g["cell_pass"]])

    # ---- D324 diagnostic figure ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(17, 11))
    thetas = [float(t) for t in LADDER_THETA_DEG]

    a0 = ax[0][0]
    best_err_d = [min(r["descend"]["urdf"]["axis_err_deg"] for r in pin_rows
                      if r["theta_deg"] == t) for t in thetas]
    best_err_a = [min(r["approach"]["urdf"]["axis_err_deg"] for r in pin_rows
                      if r["theta_deg"] == t) for t in thetas]
    a0.plot(thetas, best_err_d, "o-", color="crimson", label="descend (best psi)")
    a0.plot(thetas, best_err_a, "s--", color="steelblue", label="approach (best psi)")
    a0.axhline(AXIS_GATE_DEG, color="green", ls=":", label=f"gate {AXIS_GATE_DEG} deg")
    for t in theta_pass:
        a0.axvspan(t - 0.4, t + 0.4, color="green", alpha=0.12)
    a0.set_xlabel("commanded tool-axis tilt theta [deg]")
    a0.set_ylabel("achieved axis error vs target axis [deg]")
    a0.set_title(f"(A) pin {PIN_POSE}: can the arm hold the tilted axis?  "
                 f"green band = full pass")
    a0.legend(); a0.grid(alpha=0.3)

    a1 = ax[0][1]
    M = np.zeros((len(thetas), len(PSI_DEG)))
    for i, t in enumerate(thetas):
        for j, ps in enumerate(PSI_DEG):
            r = next(r for r in pin_rows if r["theta_deg"] == t and r["psi_deg"] == ps)
            M[i, j] = min(r["descend"]["urdf"]["axis_err_deg"], 90.0)
    im = a1.imshow(M, aspect="auto", origin="lower", cmap="viridis_r",
                   extent=[PSI_DEG[0], PSI_DEG[-1], thetas[0], thetas[-1]])
    az_pin_deg = math.degrees(math.atan2(ny, nx)) % 360.0
    a1.axvline(az_pin_deg, color="white", ls="--", lw=1.2, label=f"pin azimuth {az_pin_deg:.0f}")
    a1.axvline((az_pin_deg + 180.0) % 360.0, color="orange", ls="--", lw=1.2, label="+180")
    plt.colorbar(im, ax=a1, label="axis error [deg], descend")
    a1.set_xlabel("psi = world azimuth of the tilt [deg]")
    a1.set_ylabel("theta [deg]")
    a1.set_title("(B) which tilt DIRECTIONS are kinematically available")
    a1.legend(loc="upper right", fontsize=8)

    a2 = ax[1][0]
    q4s, clr, tt = [], [], []
    for t in thetas:
        ps = [r for r in pin_rows if r["theta_deg"] == t and r["pass_urdf_both_z"]]
        if ps:
            b = min(ps, key=lambda r: r["descend"]["urdf"]["axis_err_deg"])
            tt.append(t); q4s.append(b["descend"]["q4_required_deg"])
            clr.append(b["descend"]["support_clearance_mm"])
    if tt:
        a2.plot(tt, q4s, "o-", color="purple", label="q4 (wrist roll) required [deg]")
        a2.axhline(180.0, color="purple", ls=":", lw=1)
        a2.axhline(-180.0, color="purple", ls=":", lw=1, label="URDF wrist-roll limit")
        a2.axhline(90.0, color="brown", ls="-.", lw=1)
        a2.axhline(-90.0, color="brown", ls="-.", lw=1, label="v6-clip limit")
        a2b = a2.twinx()
        a2b.plot(tt, clr, "s--", color="darkgreen", label="jaw clearance over support [mm]")
        a2b.axhline(0.0, color="red", ls="-", lw=1)
        a2b.set_ylabel("support clearance [mm]", color="darkgreen")
        a2b.legend(loc="lower right", fontsize=8)
    a2.set_xlabel("theta [deg]"); a2.set_ylabel("q4 [deg]", color="purple")
    a2.set_title("(C) wrist roll needed for D431's phi*, and the ground clearance")
    a2.legend(loc="upper left", fontsize=8); a2.grid(alpha=0.3)

    a3 = ax[1][1]
    if grid_rows:
        gx = [g["x"] for g in grid_rows if not g["cell_pass"]]
        gy = [g["y"] for g in grid_rows if not g["cell_pass"]]
        a3.scatter(gx, gy, s=16, c="lightgrey", label="tilted: fail")
        px = [g["x"] for g in grid_pass]; py = [g["y"] for g in grid_pass]
        a3.scatter(px, py, s=22, c="seagreen", label=f"tilted {GRID_THETA_DEG:.0f} deg: pass "
                                                     f"({len(grid_pass)})")
        vx = [a for a, b in t2b_vertical_cells]; vy = [b for a, b in t2b_vertical_cells]
        a3.scatter(vx, vy, s=70, facecolors="none", edgecolors="steelblue", linewidths=0.8,
                   label=f"T2b vertical pass ({len(t2b_vertical_cells)})")
    a3.scatter([nx], [ny], s=140, marker="*", c="crimson", zorder=5, label=f"pin {PIN_POSE}")
    a3.set_xlabel("x [m]"); a3.set_ylabel("y [m]"); a3.set_aspect("equal")
    a3.set_title(f"(D) where on the table the tilted grasp is reachable")
    a3.legend(fontsize=8); a3.grid(alpha=0.3)

    fig.suptitle(f"g0b_d420 {TAG} - tilted tool-axis IK reachability + support clearance "
                 f"(read-only)   VERDICT: {code}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(paths["diagnostic.png"], dpi=130)
    plt.close(fig)

    # ---- D341 Rerun ---------------------------------------------------------------
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact
    app_id = f"roarm_g0b_{TAG}"
    STRIDE = 60

    def cyl_wire(cx, cy, top_z, R_m, H_m):
        rings, walls = [], []
        ang = np.linspace(0.0, 2 * math.pi, 65)
        for k in (0.0, H_m * 0.5, H_m):
            rings.append(np.column_stack([cx + R_m * np.cos(ang), cy + R_m * np.sin(ang),
                                          np.full_like(ang, top_z - k)]).tolist())
        for a in np.linspace(0.0, 2 * math.pi, 16, endpoint=False):
            walls.append([[cx + R_m * math.cos(a), cy + R_m * math.sin(a), top_z],
                          [cx + R_m * math.cos(a), cy + R_m * math.sin(a), top_z - H_m]])
        return rings, walls

    R_cyl = n8.CYL_R_MM / 1000.0
    H_cyl = n8.CYL_H_MM / 1000.0
    rings, walls = cyl_wire(nx, ny, z_pin, R_cyl, H_cyl)
    table = [[[-0.05, -0.35, SUPPORT_Z_T3], [0.55, -0.35, SUPPORT_Z_T3],
              [0.55, 0.35, SUPPORT_Z_T3], [-0.05, 0.35, SUPPORT_Z_T3],
              [-0.05, -0.35, SUPPORT_Z_T3]]]

    with rr.RecordingStream(app_id, recording_id=f"g0b_d420_{TAG}", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(paths["timeline.rrd"]), write_footer=True)
        rec.log("scene/support_plane", rr.LineStrips3D(table, colors=[[110, 110, 120]],
                radii=0.0015), static=True)
        rec.log("object/cylinder", rr.LineStrips3D(
            rings + walls, colors=[[225, 60, 60]] * (len(rings) + len(walls)), radii=0.0008),
            static=True)
        for i, th in enumerate(LADDER_THETA_DEG):
            cand = pin_pass(th)
            row = (min(cand, key=lambda r: r["descend"]["urdf"]["axis_err_deg"]) if cand
                   else min((r for r in pin_rows if r["theta_deg"] == float(th)),
                            key=lambda r: r["descend"]["urdf"]["axis_err_deg"]))
            q4v = np.array(row["descend"]["urdf"]["q_deg"] + [row["descend"].get("q4_required_deg", 0.0)])
            T, T5 = fk_full(q4v)
            _tcp, _l5, origins = t2.fk_points(np.array(row["descend"]["urdf"]["q_deg"]))
            q5r = math.radians(row["q5_star_deg"])
            R5, o5 = T5[:3, :3], T5[:3, 3]
            jaw_f = (V5[::STRIDE] @ R5.T + o5)
            jaw_m = (to_link5(VG, q5r)[::STRIDE] @ R5.T + o5)
            d_axis, _u, _v = axis_frame(row["theta_deg"], row["psi_deg"])
            rec.reset_time()
            rec.set_time("theta_index", sequence=i)
            rec.log("arm/skeleton", rr.LineStrips3D([[[float(w) for w in p] for p in origins]],
                    colors=[[245, 245, 245]], radii=0.003))
            rec.log("arm/tool_axis", rr.Arrows3D(origins=[[float(w) for w in T[:3, 3]]],
                    vectors=[[float(w) * 0.05 for w in d_axis]], colors=[[40, 200, 80]],
                    radii=0.0018))
            rec.log("jaw/fixed_link5", rr.Points3D(jaw_f.tolist(), colors=[150, 150, 160],
                    radii=0.0006))
            rec.log("jaw/moving_gripper_link", rr.Points3D(jaw_m.tolist(), colors=[70, 130, 230],
                    radii=0.0006))
            rec.log("target/tcp_descend", rr.Points3D([row["descend"]["target_m"]],
                    colors=[[255, 210, 40]], radii=0.0035))
            rec.log("plots/theta_deg", rr.Scalars(float(row["theta_deg"])))
            rec.log("plots/axis_err_deg", rr.Scalars(float(row["descend"]["urdf"]["axis_err_deg"])))
            rec.log("plots/pos_err_mm", rr.Scalars(float(row["descend"]["urdf"]["pos_err_mm"])))
            rec.log("plots/q4_required_deg",
                    rr.Scalars(float(row["descend"].get("q4_required_deg") or 0.0)))
            rec.log("plots/support_clearance_mm",
                    rr.Scalars(float(row["descend"].get("support_clearance_mm") or 0.0)))
            rec.log("plots/bite_mm_D431", rr.Scalars(float(row["bite_mm_D431"])))
        rec.reset_time()
        rec.set_time("theta_index", sequence=0)
        for name, ok in (("N9a_sha_pins", g_a), ("N9b_fk_matches_frozen", g_b),
                         ("N9c_theta0_reproduces_T2_and_T2b", g_c),
                         ("N9d_moving_jaw_z_range", g_d),
                         ("N9e_psi_invariant_at_theta0", g_e),
                         ("N9f_wrist_roll_closed_form", g_f),
                         ("N9g_theta6_reachable_at_pin", bool(g_g)),
                         ("N9h_theta17_reachable_at_pin", bool(g_h)),
                         ("N9i_support_clearance_positive", bool(g_i))):
            rec.log("events/gates", rr.TextLog(name, level=rr.TextLogLevel.INFO if ok
                                               else rr.TextLogLevel.ERROR))
        summary_md = (
            f"# g0b_d420 {TAG} - tilted tool-axis IK reachability (read-only)\n\n"
            f"**VERDICT: {code}**\n\n"
            f"## Question\nD431 proved the D29 cylinder is admitted once the tool axis tilts "
            f"(theta_min <= 6 deg), but measured it in the TOOL frame and stated "
            f"*'tilted-pose IK reachability NOT established'*. Can the arm actually hold that "
            f"pose at the pinned T3 spawn `{PIN_POSE}`, can the wrist roll deliver D431's phi*, "
            f"and does the jaw clear the ground?\n\n"
            f"## Answer\n"
            f"- theta values that fully pass at the pin: **{theta_pass} deg**\n"
            f"- minimum passing tilt **{theta_min_pass} deg**, maximum **{theta_max_pass} deg**\n"
            f"- smallest ground clearance over all evaluated poses: "
            f"**{out['N9i_support_clearance_positive']['min_clearance_mm']} mm**\n"
            f"- table cells reachable tilted at {GRID_THETA_DEG:.0f} deg: **{len(grid_pass)}** "
            f"(T2b vertical reference: {len(t2b_vertical_cells)})\n\n"
            f"## Reproduction gates\n"
            f"- N9b: full-rotation FK equals the frozen T2 `fk_points` to "
            f"**{d_tcp:.1e} m**\n"
            f"- N9c: theta = 0 reproduces frozen **t2** and **t2b** named poses exactly "
            f"(max |d| = {max(wp2, wp2b, wa2, wa2b):.1e})\n"
            f"- N9d: moving-jaw z = [{zq0.min():.4f}, {zq0.max():.4f}] mm vs 40th "
            f"[{n8.MOVING_Z_RANGE_37TH_MM[0]}, {n8.MOVING_Z_RANGE_37TH_MM[1]}]\n"
            f"- N9f: wrist-roll closed form residual **{n9f_worst:.1e} deg**\n\n"
            f"## Limits\nKinematics + rigid-body clearance only. **Force closure is NOT "
            f"established**; D431's positive bite is one-sided (moving jaw). Visual mesh, not the "
            f"attempt3 collision USD. No contact physics, no servo torque, no real-robot claim. "
            f"D427 / D429 / D430 / D431 unchanged, `g0a_pass=false`.\n\n"
            f"## Scene\nwhite = arm skeleton, green arrow = commanded tool axis, grey = fixed jaw "
            f"(link5), blue = moving jaw at this theta's D431 argmax q5, red = the standing D29 "
            f"cylinder, yellow = the TCP target, grey plane = the support surface (z = 0).\n\n"
            f"Authority = stdout + `{paths['results.json'].name}`. Rerun is inspection evidence "
            f"only (D341).\n")
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN),
                static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run",
                                         name="1 | tilted IK verdict"),
                    rrb.Spatial3DView(origin="/", contents=["/arm/**", "/jaw/**", "/object/**",
                                                            "/scene/**", "/target/**"],
                                      name="2 | arm + jaw + cylinder per tilt step"),
                    rrb.TextLogView(origin="/events/gates", contents="/events/gates/**",
                                    name="3 | gates"),
                    column_shares=[0.30, 0.46, 0.24],
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/axis_err_deg/**", "/plots/pos_err_mm/**",
                                                 "/plots/support_clearance_mm/**"],
                                       name="4 | axis error, position error, ground clearance"),
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/theta_deg/**",
                                                 "/plots/q4_required_deg/**",
                                                 "/plots/bite_mm_D431/**"],
                                       name="5 | tilt, wrist roll, and D431's bite"),
                ),
                row_shares=[0.58, 0.42],
            ),
            auto_layout=False, auto_views=False, collapse_panels=True,
        )
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=30.0)
    blueprint.save(app_id, str(paths["timeline.rbl"]))

    expected_entities = ["metadata/run", "scene/support_plane", "object/cylinder", "arm/skeleton",
                         "arm/tool_axis", "jaw/fixed_link5", "jaw/moving_gripper_link",
                         "target/tcp_descend", "plots/theta_deg", "plots/axis_err_deg",
                         "plots/pos_err_mm", "plots/q4_required_deg",
                         "plots/support_clearance_mm", "plots/bite_mm_D431", "events/gates"]
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    lin3 = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    components = {
        "metadata/run": ["TextDocument:text"],
        "scene/support_plane": lin3, "object/cylinder": lin3, "arm/skeleton": lin3,
        "arm/tool_axis": ["Arrows3D:origins", "Arrows3D:vectors", "Arrows3D:colors",
                          "Arrows3D:radii"],
        "jaw/fixed_link5": pts3, "jaw/moving_gripper_link": pts3, "target/tcp_descend": pts3,
        "plots/theta_deg": ["Scalars:scalars"], "plots/axis_err_deg": ["Scalars:scalars"],
        "plots/pos_err_mm": ["Scalars:scalars"], "plots/q4_required_deg": ["Scalars:scalars"],
        "plots/support_clearance_mm": ["Scalars:scalars"], "plots/bite_mm_D431": ["Scalars:scalars"],
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
    print(f"[{LOG}] rerun_validation pass={validation.get('pass')} "
          f"errors={validation.get('errors')}", flush=True)

    shutil.copyfile(__file__, paths["script.py.txt"])
    out["artifacts"] = {k: {"name": v.name, "sha256": sha256(v)[:16], "bytes": v.stat().st_size}
                        for k, v in paths.items() if v.exists() and k != "results.json"}
    out["artifacts_note"] = ("results.json is deliberately absent from this manifest - D429-R1. "
                             "Hash it from disk.")
    out["rerun_validation_pass"] = bool(validation.get("pass"))
    out["wall_seconds"] = round(time.time() - t_start, 1)
    paths["results.json"].write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"[{LOG}] artifacts " + " ".join(f"{v['name']}={v['sha256']}"
                                           for v in out["artifacts"].values()), flush=True)
    print(f"[{LOG}] results.json={sha256(paths['results.json'])[:16]} "
          f"bytes={paths['results.json'].stat().st_size}", flush=True)
    print(f"[{LOG}] G0B_T3R_N9_VERDICT={code}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
