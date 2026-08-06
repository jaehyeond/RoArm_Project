#!/usr/bin/env python3
"""G0b T2 — vertical tool-axis IK reachability probe for the standing D29xH50 cylinder.

Case g0b_d420 (T-ladder T2 + T2b annex, D419). Numpy-only kinematics probe: NO Isaac launch,
NO robot hardware, NO training, NO frozen-file modification. New outputs only under
claudedocs/runtime_logs/grasp_track/g0b_d420/.

Question: can this 5-DOF arm place the TCP at the top-center grasp target of a
standing D29xH50 cylinder (p7 conventions: world_grasp = top cap center,
descend = +0.5 mm margin, approach = +40 mm) with the tool axis vertical
(straight down), and where on the table does that hold? If exact vertical fails,
what is the minimum achievable tilt per cell (the T1-observed "slightly tilted
vertical" fallback)?

Tool-axis definition: unit vector from link5 origin to TCP origin (equals the TCP
frame x-axis). Preregistered self-check (abort on failure, no cell results):
at q = [1.7730, 35.6563, 111.8334, 9.4908, 0] deg (D418 frozen band midpoint) the
axis must tilt 21.7..24.4 deg from vertical-down (D410/D413 band) AND fk tcp z
must lie in 0.013486..0.013628 m (D418 frozen band).

Solver: 5-task DLS (position 3 + axis x/y 2) over q0..q3 with q4 = 0 (wrist roll
spins about the tool axis; affects neither position nor axis). Two joint-limit
sets are evaluated: URDF literals (sim authority) and the v6-clip limits used by
the p7 pipeline (v6 box is a strict subset of the URDF box).

T2b annex (t3_conversion_design.md D-5): pass --z_offset_m 0.012117 --tag t2b to
re-run the identical sweep at the real settled height (cylinder standing on the
ground plane z=0 instead of TABLE_Z). Defaults reproduce T2 exactly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim_scripts"))

from roarm_kinematics import _CHAIN, JOINT_LIMITS_DEG, Tmat, Trot_z  # noqa: E402

RERUN_VERSION = "0.34.1"
TABLE_Z = -0.012117
CYL_RADIUS_M = 0.0145
CYL_HEIGHT_M = 0.050
GRASP_SURFACE_MARGIN_M = 0.0005
APPROACH_CLEARANCE_M = 0.040
DESCEND_Z = TABLE_Z + CYL_HEIGHT_M + GRASP_SURFACE_MARGIN_M
APPROACH_Z = DESCEND_Z + APPROACH_CLEARANCE_M

POS_GATE_MM = 3.0
TILT_GATE_PRIMARY_DEG = 5.0
TILT_GATE_FALLBACK_DEG = 10.0

SELF_CHECK_Q_DEG = np.array([1.7730, 35.6563, 111.8334, 9.4908], dtype=np.float64)
SELF_CHECK_TILT_BAND_DEG = (21.7, 24.4)
SELF_CHECK_TCPZ_BAND_M = (0.013486, 0.013628)

# URDF literals (local_assets/roarm_m3/urdf/roarm_m3.urdf:185,194,203,212) — sim authority.
URDF_LIMITS_DEG = {
    "base": (math.degrees(-3.1416), math.degrees(3.1416)),
    "shoulder": (math.degrees(-1.5708), math.degrees(1.5708)),
    "elbow": (math.degrees(-1.0), math.degrees(2.95)),
    "wrist_p": (math.degrees(-1.92), math.degrees(1.92)),
}
V6_LIMITS_DEG = {k: JOINT_LIMITS_DEG[k] for k in ("base", "shoulder", "elbow", "wrist_p")}

GRID_X = np.round(np.arange(0.10, 0.46 + 1e-9, 0.02), 3)
GRID_Y = np.round(np.arange(-0.26, 0.26 + 1e-9, 0.02), 3)

# p7 candidate spawn poses (p7_branch_b probe constants).
NAMED_POSES = {
    "seed0_S1": (+0.21369616873214542, -0.19571919576125169),
    "seed0_S2": (+0.15165276355285290, +0.17572513109603544),
    "seed0_S3": (+0.39066357757671800, -0.13246041268192021),
    "seed0_S4": (+0.42350724237877680, +0.17237803311822986),
    "R1_center": (0.200, -0.175),
    "R2_center": (0.200, +0.135),
    "R3_center": (0.380, -0.160),
    "R4_center": (0.380, +0.125),
}

OUT_DIR = REPO / "claudedocs" / "runtime_logs" / "grasp_track" / "g0b_d420"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"


def fk_points(q4_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """FK for [q0..q3, q4=0]; returns (tcp, link5_origin, all frame origins)."""
    q = np.radians(np.array([q4_deg[0], q4_deg[1], q4_deg[2], q4_deg[3], 0.0, 0.0]))
    T = np.eye(4)
    origins: list[np.ndarray] = []
    link5 = None
    for name, xyz, rpy, qi in _CHAIN:
        T = T @ Tmat(xyz, rpy)
        if qi is not None:
            T = T @ Trot_z(q[qi])
        origins.append(T[:3, 3].copy())
        if name == "link4_to_link5":
            link5 = T[:3, 3].copy()
    tcp = T[:3, 3].copy()
    return tcp, link5, origins


def axis_tilt(q4_deg: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    tcp, link5, _ = fk_points(q4_deg)
    axis = tcp - link5
    axis = axis / np.linalg.norm(axis)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(axis, [0.0, 0.0, -1.0]))))))
    return tcp, tilt, axis


def clip4(q4_deg: np.ndarray, limits: dict[str, tuple[float, float]]) -> np.ndarray:
    out = q4_deg.copy()
    for i, name in enumerate(("base", "shoulder", "elbow", "wrist_p")):
        lo, hi = limits[name]
        out[i] = max(lo, min(hi, out[i]))
    return out


def task_error(q4_deg: np.ndarray, target_p: np.ndarray, w_axis: float) -> np.ndarray:
    tcp, _tilt, axis = axis_tilt(q4_deg)
    return np.array(
        [
            target_p[0] - tcp[0],
            target_p[1] - tcp[1],
            target_p[2] - tcp[2],
            -w_axis * axis[0],
            -w_axis * axis[1],
        ],
        dtype=np.float64,
    )


def dls_vertical(
    target_p: np.ndarray,
    seed4_deg: np.ndarray,
    limits: dict[str, tuple[float, float]],
    max_iter: int = 160,
    w_axis: float = 0.03,
    damping: float = 0.002,
    step_clip_deg: float = 4.0,
    eps_deg: float = 0.05,
) -> tuple[np.ndarray, float, float, int]:
    """Returns (q4_deg, pos_err_mm, tilt_deg, iters)."""
    q = clip4(np.asarray(seed4_deg, dtype=np.float64).copy(), limits)
    best = None
    for it in range(max_iter):
        e = task_error(q, target_p, w_axis)
        tcp, tilt, _axis = axis_tilt(q)
        pos_err_mm = float(np.linalg.norm(target_p - tcp)) * 1000.0
        key = (pos_err_mm > POS_GATE_MM, tilt if pos_err_mm <= POS_GATE_MM else 1.0e9, pos_err_mm)
        if best is None or key < best[0]:
            best = (key, q.copy(), pos_err_mm, tilt, it)
        if pos_err_mm < 0.2 and tilt < 0.2:
            break
        J = np.zeros((5, 4), dtype=np.float64)
        for i in range(4):
            qp = q.copy()
            qp[i] += eps_deg
            qm = q.copy()
            qm[i] -= eps_deg
            # central difference of the error: J = de/dq
            J[:, i] = (task_error(qp, target_p, w_axis) - task_error(qm, target_p, w_axis)) / (2.0 * eps_deg)
        M = J @ J.T + (damping**2) * np.eye(5)
        try:
            # e(q+dq) ~ e + J dq -> 0  =>  dq = -J^+ e (DLS)
            dq = -J.T @ np.linalg.solve(M, e)
        except np.linalg.LinAlgError:
            break
        m = float(np.max(np.abs(dq)))
        if m > step_clip_deg:
            dq = dq * (step_clip_deg / m)
        q = clip4(q + dq, limits)
    _key, qb, pe, tl, it_used = best
    return qb, pe, tl, it_used


def seeds_for(x: float, y: float) -> list[np.ndarray]:
    az = math.degrees(math.atan2(y, x))
    return [
        np.array([az, 0.0, 90.0, 90.0]),
        np.array([az, 45.0, 100.0, 35.0]),
        np.array([az, 20.0, 130.0, 30.0]),
        np.array([az, 60.0, 85.0, 35.0]),
        SELF_CHECK_Q_DEG.copy(),
    ]


def solve_cell(x: float, y: float, z: float, limits: dict[str, tuple[float, float]]) -> dict:
    target = np.array([x, y, z], dtype=np.float64)
    best = None
    for si, seed in enumerate(seeds_for(x, y)):
        q, pe, tl, it_used = dls_vertical(target, seed, limits)
        key = (pe > POS_GATE_MM, tl if pe <= POS_GATE_MM else 1.0e9, pe)
        if best is None or key < best["_key"]:
            best = {
                "_key": key,
                "q_deg": [round(float(v), 4) for v in q],
                "pos_err_mm": round(pe, 4),
                "tilt_deg": round(tl, 4),
                "seed_idx": si,
                "iters": it_used,
            }
        if pe < 0.3 and tl < 0.5:
            break
    best.pop("_key")
    best["pos_ok"] = best["pos_err_mm"] <= POS_GATE_MM
    best["vertical_ok"] = bool(best["pos_ok"] and best["tilt_deg"] <= TILT_GATE_PRIMARY_DEG)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="G0b T2/T2b vertical tool-axis IK reachability probe")
    parser.add_argument(
        "--z_offset_m", type=float, default=0.0,
        help="T2b annex: shift both target heights by this amount "
             "(+0.012117 = cylinder standing on the ground plane z=0 instead of TABLE_Z)",
    )
    parser.add_argument("--tag", default="t2", help="artifact prefix / log tag (T2b annex uses t2b)")
    args = parser.parse_args()
    if args.z_offset_m != 0.0 and args.tag == "t2":
        print("[g0b_t2_ik] ABORT nonzero --z_offset_m requires non-default --tag (protects T2 artifacts)", flush=True)
        return 3
    tag = args.tag
    log = f"g0b_{tag}_ik"
    vtag = tag.upper()
    descend_z = DESCEND_Z + args.z_offset_m
    approach_z = APPROACH_Z + args.z_offset_m
    table_z_eff = TABLE_Z + args.z_offset_m
    rrd_path = OUT_DIR / f"{tag}_ik_reachability.rrd"
    rbl_path = OUT_DIR / f"{tag}_ik_reachability.rbl"
    png_path = OUT_DIR / f"{tag}_ik_reachability_inspection.png"
    validation_path = OUT_DIR / f"{tag}_ik_rerun_validation.json"
    results_path = OUT_DIR / f"{tag}_ik_results.json"
    csv_path = OUT_DIR / f"{tag}_ik_grid.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import rerun as rr

    if str(rr.__version__) != RERUN_VERSION:
        print(f"[{log}] RERUN_VERSION_MISMATCH have={rr.__version__} want={RERUN_VERSION}", flush=True)
        return 3

    print(f"[{log}] G0b {vtag} vertical tool-axis IK reachability probe (numpy-only, no Isaac)", flush=True)
    print(
        f"[{log}] gates pos_gate_mm={POS_GATE_MM} tilt_primary_deg={TILT_GATE_PRIMARY_DEG} "
        f"tilt_fallback_deg={TILT_GATE_FALLBACK_DEG} z_offset_m={args.z_offset_m:+.6f} "
        f"descend_z={descend_z:.6f} approach_z={approach_z:.6f} "
        f"grid_x=[{GRID_X[0]},{GRID_X[-1]}]x{len(GRID_X)} grid_y=[{GRID_Y[0]},{GRID_Y[-1]}]x{len(GRID_Y)}",
        flush=True,
    )

    # ---- Preregistered self-check --------------------------------------------
    sc_tcp, sc_tilt, _sc_axis = axis_tilt(SELF_CHECK_Q_DEG)
    sc_tilt_ok = SELF_CHECK_TILT_BAND_DEG[0] <= sc_tilt <= SELF_CHECK_TILT_BAND_DEG[1]
    sc_tcpz_ok = SELF_CHECK_TCPZ_BAND_M[0] <= float(sc_tcp[2]) <= SELF_CHECK_TCPZ_BAND_M[1]
    print(
        f"[{log}] self_check q={list(SELF_CHECK_Q_DEG)} tilt_deg={sc_tilt:.4f} "
        f"band={SELF_CHECK_TILT_BAND_DEG} tilt_ok={sc_tilt_ok} tcp_z={float(sc_tcp[2]):.6f} "
        f"band={SELF_CHECK_TCPZ_BAND_M} tcpz_ok={sc_tcpz_ok}",
        flush=True,
    )
    if not (sc_tilt_ok and sc_tcpz_ok):
        print(f"[{log}] {vtag}_VERTICAL_IK_VERDICT=SELF_CHECK_FAIL", flush=True)
        return 3

    # ---- Grid sweep ----------------------------------------------------------
    rows: list[dict] = []
    n_done = 0
    for x in GRID_X:
        for y in GRID_Y:
            for z_name, z in (("descend", descend_z), ("approach", approach_z)):
                r_urdf = solve_cell(float(x), float(y), z, URDF_LIMITS_DEG)
                if r_urdf["pos_ok"]:
                    r_v6 = solve_cell(float(x), float(y), z, V6_LIMITS_DEG)
                else:
                    r_v6 = {"pos_err_mm": None, "tilt_deg": None, "pos_ok": False, "vertical_ok": False,
                            "q_deg": None, "seed_idx": None, "iters": 0}
                rows.append({"x": float(x), "y": float(y), "z_name": z_name, "z": z,
                             "urdf": r_urdf, "v6clip": r_v6})
            n_done += 1
            if n_done % 60 == 0:
                print(f"[{log}] progress cells={n_done}/{len(GRID_X) * len(GRID_Y)}", flush=True)

    def cell_pass(x: float, y: float, key: str) -> bool:
        rs = [r for r in rows if r["x"] == x and r["y"] == y]
        return all(r[key]["vertical_ok"] for r in rs)

    grid_pass_urdf = [(x, y) for x in GRID_X for y in GRID_Y if cell_pass(float(x), float(y), "urdf")]
    grid_pass_v6 = [(x, y) for x in GRID_X for y in GRID_Y if cell_pass(float(x), float(y), "v6clip")]

    # ---- Named p7 candidate poses -------------------------------------------
    named_results = {}
    for name, (nx, ny) in NAMED_POSES.items():
        rec = {}
        for z_name, z in (("descend", descend_z), ("approach", approach_z)):
            rec[z_name] = {
                "urdf": solve_cell(nx, ny, z, URDF_LIMITS_DEG),
                "v6clip": solve_cell(nx, ny, z, V6_LIMITS_DEG),
            }
        rec["pass_urdf"] = all(rec[zn]["urdf"]["vertical_ok"] for zn in ("descend", "approach"))
        rec["pass_v6clip"] = all(rec[zn]["v6clip"]["vertical_ok"] for zn in ("descend", "approach"))
        rec["pass_both"] = rec["pass_urdf"] and rec["pass_v6clip"]
        named_results[name] = rec
        print(
            f"[{log}] named pose={name} xy=({nx:+.4f},{ny:+.4f}) "
            f"descend urdf pos_mm={rec['descend']['urdf']['pos_err_mm']} tilt={rec['descend']['urdf']['tilt_deg']} "
            f"approach urdf pos_mm={rec['approach']['urdf']['pos_err_mm']} tilt={rec['approach']['urdf']['tilt_deg']} "
            f"pass_urdf={rec['pass_urdf']} pass_v6clip={rec['pass_v6clip']}",
            flush=True,
        )

    named_pass_both = [n for n, r in named_results.items() if r["pass_both"]]

    fallback_ok = any(
        row["urdf"]["pos_ok"]
        and row["urdf"]["tilt_deg"] is not None
        and row["urdf"]["tilt_deg"] <= TILT_GATE_FALLBACK_DEG
        for row in rows
        if row["z_name"] == "descend"
    )

    if named_pass_both:
        verdict = f"{vtag}_PASS"
    elif grid_pass_urdf:
        verdict = f"{vtag}_PARTIAL"
    elif fallback_ok:
        verdict = f"{vtag}_PARTIAL"
    else:
        verdict = f"{vtag}_FAIL"

    # best named pose for T3 recommendation: prefer pass_both, min descend tilt
    def named_key(item):
        _name, r = item
        tilt = r["descend"]["urdf"]["tilt_deg"]
        return (not r["pass_both"], not r["pass_urdf"], tilt if tilt is not None else 1.0e9)

    best_named_name, best_named = sorted(named_results.items(), key=named_key)[0]
    q_best = best_named["descend"]["urdf"]["q_deg"]

    print(
        f"[{log}] aggregate grid_cells={len(GRID_X) * len(GRID_Y)} "
        f"pass_urdf_cells={len(grid_pass_urdf)} pass_v6clip_cells={len(grid_pass_v6)} "
        f"named_pass_both={named_pass_both} best_named={best_named_name}",
        flush=True,
    )
    print(f"[{log}] {vtag}_VERTICAL_IK_VERDICT={verdict}", flush=True)

    # ---- Artifacts: JSON + CSV ----------------------------------------------
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z_name", "limits", "pos_err_mm", "tilt_deg", "pos_ok", "vertical_ok",
                    "q0", "q1", "q2", "q3", "seed_idx", "iters"])
        for row in rows:
            for lim in ("urdf", "v6clip"):
                r = row[lim]
                q = r["q_deg"] or [None] * 4
                w.writerow([row["x"], row["y"], row["z_name"], lim, r["pos_err_mm"], r["tilt_deg"],
                            r["pos_ok"], r["vertical_ok"], q[0], q[1], q[2], q[3], r["seed_idx"], r["iters"]])

    results = {
        "artifact": f"G0B_{vtag}_VERTICAL_TOOL_AXIS_IK_REACHABILITY_V1",
        "case": "g0b_d420",
        "verdict": verdict,
        "self_check": {
            "q_deg": [float(v) for v in SELF_CHECK_Q_DEG],
            "tilt_deg": round(sc_tilt, 4),
            "tilt_band_deg": SELF_CHECK_TILT_BAND_DEG,
            "tcp_z_m": round(float(sc_tcp[2]), 6),
            "tcpz_band_m": SELF_CHECK_TCPZ_BAND_M,
            "pass": True,
        },
        "gates": {
            "pos_gate_mm": POS_GATE_MM,
            "tilt_primary_deg": TILT_GATE_PRIMARY_DEG,
            "tilt_fallback_deg": TILT_GATE_FALLBACK_DEG,
            "z_offset_m": args.z_offset_m,
            "descend_z_m": descend_z,
            "approach_z_m": approach_z,
        },
        "limits": {"urdf_deg": URDF_LIMITS_DEG, "v6clip_deg": V6_LIMITS_DEG},
        "grid": {
            "x": [float(v) for v in GRID_X],
            "y": [float(v) for v in GRID_Y],
            "pass_urdf_cells": [[float(a), float(b)] for a, b in grid_pass_urdf],
            "pass_v6clip_cells": [[float(a), float(b)] for a, b in grid_pass_v6],
        },
        "named": named_results,
        "best_named": best_named_name,
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "rerun_sdk": rr.__version__},
    }
    results_path.write_text(json.dumps(results, indent=2, default=str) + "\n")

    # ---- D341 Rerun artifact -------------------------------------------------
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    def tilt_color(r: dict) -> list[int]:
        if not r["pos_ok"]:
            return [120, 120, 120]
        t = r["tilt_deg"]
        if t <= TILT_GATE_PRIMARY_DEG:
            return [40, 200, 80]
        if t <= TILT_GATE_FALLBACK_DEG:
            return [230, 210, 40]
        if t <= 20.0:
            return [240, 140, 40]
        return [230, 60, 60]

    grid_pts = {"descend": [], "approach": []}
    grid_cols = {"descend": [], "approach": []}
    for row in rows:
        grid_pts[row["z_name"]].append([row["x"], row["y"], row["z"]])
        grid_cols[row["z_name"]].append(tilt_color(row["urdf"]))

    named_pts, named_cols, named_labels = [], [], []
    for name, (nx, ny) in NAMED_POSES.items():
        r = named_results[name]
        named_pts.append([nx, ny, descend_z])
        named_cols.append([60, 170, 255] if r["pass_both"] else [200, 60, 200])
        named_labels.append(
            f"{name}: both={'PASS' if r['pass_both'] else 'FAIL'} "
            f"tilt={r['descend']['urdf']['tilt_deg']}"
        )

    bx, by = NAMED_POSES[best_named_name]
    _tcp_b, _l5_b, origins = fk_points(np.array(q_best[:4] if q_best else [0, 0, 90, 0]))
    skeleton = [[float(a) for a in p] for p in origins]
    tcp_b, _tilt_b, axis_b = axis_tilt(np.array(q_best[:4] if q_best else [0, 0, 90, 0]))

    circle = []
    for k in range(33):
        a = 2.0 * math.pi * k / 32
        circle.append([bx + CYL_RADIUS_M * math.cos(a), by + CYL_RADIUS_M * math.sin(a), table_z_eff + CYL_HEIGHT_M])
    circle_bot = [[p[0], p[1], table_z_eff] for p in circle]

    summary_md = (
        f"# G0b {vtag} vertical tool-axis IK reachability (case g0b_d420)\n\n"
        f"- verdict: **{verdict}**\n"
        f"- z_offset_m: {args.z_offset_m:+.6f} (descend {descend_z:.6f}, approach {approach_z:.6f})\n"
        f"- self-check: tilt {sc_tilt:.3f} deg in {SELF_CHECK_TILT_BAND_DEG}, tcp_z {float(sc_tcp[2]):.6f} in {SELF_CHECK_TCPZ_BAND_M}\n"
        f"- gates: pos<= {POS_GATE_MM} mm, vertical tilt <= {TILT_GATE_PRIMARY_DEG} deg (fallback report {TILT_GATE_FALLBACK_DEG} deg)\n"
        f"- grid pass (URDF limits): {len(grid_pass_urdf)} / {len(GRID_X) * len(GRID_Y)} cells; "
        f"v6clip: {len(grid_pass_v6)}\n"
        f"- named p7 poses pass(both limits): {named_pass_both}\n"
        f"- best named: {best_named_name} q_descend={q_best}\n"
        f"- colors: green<=5deg, yellow<=10, orange<=20, red>20, grey pos-fail\n"
    )

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run", name="1 | T2 verdict + gates"),
            rrb.Spatial3DView(
                origin="/",
                contents=["/grid/**", "/named/**", "/arm/**", "/target/**"],
                name="2 | reachability grid + best solution",
            ),
            rrb.TextLogView(origin="/events/summary", contents="/events/summary/**", name="3 | summary events"),
            column_shares=[0.28, 0.50, 0.22],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )

    app_id = f"roarm_g0b_{tag}_vertical_ik"
    with rr.RecordingStream(app_id, recording_id=f"g0b_d420_{tag}_ik", make_default=False, send_properties=True) as rec:
        rec.save(str(rrd_path), write_footer=True)
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN), static=True)
        rec.log(
            "grid/descend",
            rr.Points3D(grid_pts["descend"], colors=grid_cols["descend"], radii=0.004),
            static=True,
        )
        rec.log(
            "grid/approach",
            rr.Points3D(grid_pts["approach"], colors=grid_cols["approach"], radii=0.002),
            static=True,
        )
        rec.log(
            "named/candidates",
            rr.Points3D(named_pts, colors=named_cols, radii=0.007, labels=named_labels),
            static=True,
        )
        rec.log(
            "arm/best",
            rr.LineStrips3D([skeleton], colors=[[255, 255, 255]], radii=0.003),
            static=True,
        )
        rec.log(
            "arm/best_axis",
            rr.Arrows3D(
                origins=[[float(v) for v in tcp_b]],
                vectors=[[float(v) * 0.06 for v in axis_b]],
                colors=[[60, 170, 255]],
                radii=0.002,
            ),
            static=True,
        )
        rec.log(
            "target/cylinder",
            rr.LineStrips3D([circle, circle_bot], colors=[[210, 170, 110], [210, 170, 110]], radii=0.001),
            static=True,
        )
        rec.log(
            "events/summary",
            rr.TextLog(
                f"verdict={verdict} grid_pass_urdf={len(grid_pass_urdf)} grid_pass_v6={len(grid_pass_v6)} "
                f"named_pass_both={named_pass_both} best={best_named_name}",
                level=rr.TextLogLevel.INFO,
            ),
            static=True,
        )
        rec.flush(timeout_sec=30.0)
    blueprint.save(app_id, str(rbl_path))

    expected_entities = [
        "metadata/run", "grid/descend", "grid/approach", "named/candidates",
        "arm/best", "arm/best_axis", "target/cylinder", "events/summary",
    ]
    components = {
        "metadata/run": ["TextDocument:text"],
        "grid/descend": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "grid/approach": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "named/candidates": ["Points3D:positions", "Points3D:colors", "Points3D:radii", "Points3D:labels"],
        "arm/best": ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"],
        "arm/best_axis": ["Arrows3D:origins", "Arrows3D:vectors", "Arrows3D:colors", "Arrows3D:radii"],
        "target/cylinder": ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"],
        "events/summary": ["TextLog:level", "TextLog:text"],
    }
    validation = validate_rerun_artifact(
        rrd_path,
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=components,
        blueprint_path=rbl_path,
        screenshot_path=png_path,
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=180.0,
    )
    validation_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(
        f"[{log}] rerun_validation pass={validation.get('pass')} errors={validation.get('errors')}",
        flush=True,
    )
    print(
        f"[{log}] artifacts rrd={rrd_path.name} sha={sha256_file(rrd_path)[:16]} "
        f"results={results_path.name} csv={csv_path.name}",
        flush=True,
    )
    return 0 if verdict in (f"{vtag}_PASS", f"{vtag}_PARTIAL") and validation.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
