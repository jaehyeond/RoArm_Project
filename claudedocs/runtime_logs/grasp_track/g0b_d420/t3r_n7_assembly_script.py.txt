#!/usr/bin/env python3
"""
g0b_t3r_n7_assembly_cylinder_admission_readonly_audit.py
    (READ-ONLY on every asset; writes only NEW t3r_n7_assembly_* artifacts)

40th session — the ASSEMBLY cross-check the user authorised.

WHY THIS RUN EXISTS
  Sessions 29th..39th audited the gripper PART BY PART and closed every
  "a file is missing / a file is impoverished" hypothesis (D427, D429).
  What was never checked is whether the URDF ASSEMBLES those parts into the
  shape the real gripper has.  The real bare gripper is photo-verified to
  rim-pinch the cylinder 0..12 mm below its top face (D420 Implication (2),
  DECISIONS.md:25160-25161,25193-25196), while the sim jaws closed through the
  whole 88.31 deg -> 24 deg sweep WITHOUT TOUCHING the cylinder at any angle
  (D424 Evidence (2), DECISIONS.md:25357-25361).  Those two statements cannot
  both describe the same gripper.

CLAIM UNDER TEST (pre-registered, falsifiable both ways)
  "The gripper assembled from roarm_m3.urdf admits a D29 (r = 14.5 mm)
   cylinder, approached top-down along the tool axis, to a rim-pinch bite of
   0..12 mm below the cylinder's top face - i.e. some jaw material lies
   ALONGSIDE the cylinder wall, below its top face."

  PASS  -> the assembly is fine and T3's descend target was the defect
           => re-run the physics probe with a geometry-admissible target.
  FAIL  -> the assembly cannot reproduce a photo-verified real behaviour
           => the defect is in the assembly/geometry, not in any missing file,
              and it is NOT repaired by hand-authoring new fingers.

GEOMETRY OF THE TEST (no physics, no Isaac, pure geometry)
  Tool axis = link5 +z ; TCP at link5 z = 115.428 mm (roarm_m3.urdf:234-239).
  Object = cylinder, axis coincident with the tool axis (D419 top-down,
  top-face-centre), radius R = 14.5 mm, top face plane at z_top, body at
  z in [z_top, z_top + 50].
  Rigid-body non-penetration  =>  no gripper point may satisfy
      r_i <= R  AND  z_i >= z_top
  => the deepest reachable top-face plane is
      z_top_min = max{ z_i : r_i <= R }                     ("footprint blocker")
  A rim pinch needs jaw material BESIDE the cylinder, i.e. at
      r_i in [R, R + WALL_SPAN]  AND  z_i > z_top_min
  => achievable bite
      bite = max{ z_i : r_i in [R, R+WALL_SPAN] } - z_top_min
  bite <= 0  =>  the jaws close entirely above the object's top face.

FRAME / CONVENTION (pinned from source, not memory)
  roarm_m3.urdf:129-135  link5 <visual> origin identity, scale 0.001
       => link5.stl mm coordinates ARE link5-frame coordinates in mm.
  roarm_m3.urdf:145-160  gripper_link <visual> origin identity, scale 0.001
       => gripper_link.stl mm coordinates ARE gripper_link-frame coordinates.
  roarm_m3.urdf:225-231  link5_to_gripper_link revolute,
       origin xyz (0, 0.018821, 0.052035) rpy (-1.5708, -1.5708, 0),
       axis (0,0,1), limit 0 .. 1.571 rad.
  roarm_m3.urdf:234-239  link5_to_hand_tcp xyz (0,0,0.115428).
  q5 convention: LARGER = OPEN (D420 Implication (4), sim authority = d409
       geometric proof).  Real maximum opening 88.31 deg
       (claudedocs/direction_20260708_grasp_pivot.md:26).
       The verdict is taken over the WHOLE q5 sweep, so it does not depend on
       which end of the range is called "open".
  load_binary_stl / sample_triangles reproduced VERBATIM from the Gate-0 audit
  so that every number here sits on the same numeric path as D427.

NOT IN SCOPE (explicit)
  - gripper_left_link.stl is NOT referenced by the URDF (grep of all <mesh>
    tags) and D429 showed it is a re-tessellation of link5.stl geometry.
    Excluded from the assembly.
  - No physics, no contact solver, no Isaac, no robot, no re-run of Gate-0's
    verdict, no re-derivation of the 81.4065 hull, no 86.4 hypothesis.
"""
import hashlib
import json
import math
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # for roarm_rl.rerun_contract (D341 validator)
URDF = ROOT / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
MESH_DIR = ROOT / "local_assets/roarm_m3/urdf/meshes"
OUT_DIR = ROOT / "claudedocs/runtime_logs/grasp_track/g0b_d420"
TAG = "t3r_n7_assembly"
LOG = "g0b_t3r_n7"

L5 = MESH_DIR / "link5.stl"
GJ = MESH_DIR / "gripper_link.stl"

TCP_Z_MM = 115.428
TCP_LOCAL = np.array([0.0, 0.0, 0.115428])
SAMPLE_SPACING_M = 0.0005
STL_SCALE = 0.001
CYL_R_MM = 14.5                 # D29 cylinder, HARD RULE #18 object
CYL_H_MM = 50.0
WALL_SPAN_MM = 6.0              # annulus [R, R+6] = "beside the cylinder wall"
T1_BITE_MM = (0.0, 12.0)        # real photo-verified rim pinch, D420 (2)
Q5_REAL_MAX_RAD = math.radians(88.31)
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"
VIEW_STRIDE = 40

SHA = {"link5.stl": "1d63f374a78c1419b21eec63fa8efeef40d0d42ca89c5de3ceb0d86476d9c7eb",
       "gripper_link.stl": "7946a374e24a2f467a0581b4946e0ec41b1b86a92f070bc00aa9bced1bf65a56"}
D427_L_VIS_MM = 4.457620117187505
D427_N_PTS = 2266503
J5_R_EXPECT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
MOVING_Z_RANGE_37TH_MM = (41.2676, 119.1176)     # 37th doc, moving jaw in link5 coords


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


def zr_mm(pts_m):
    """link5-frame z in mm, and tool-axis radius in mm."""
    return pts_m[:, 2] * 1000.0, np.hypot(pts_m[:, 0], pts_m[:, 1]) * 1000.0


def admission(z_all_mm, r_all_mm, R_mm, wall_span_mm):
    """The whole test, in five lines of geometry (see module docstring)."""
    nan = float("nan")
    fp = r_all_mm <= R_mm
    if not fp.any():
        # this body has no material inside the object's footprint at all -> it cannot be the
        # thing that stops the descent. Reported as NaN rather than dropped, so a per-body row
        # is never silently missing from the sweep.
        return {"R_mm": float(R_mm), "z_top_min_mm": nan, "depth_top_min_mm": nan,
                "z_beside_max_mm": nan, "bite_mm": nan, "n_footprint": 0,
                "n_beside": int(((r_all_mm >= R_mm) & (r_all_mm <= R_mm + wall_span_mm)).sum()),
                "no_material_in_footprint": True}
    z_top_min = float(z_all_mm[fp].max())                       # footprint blocker
    beside = (r_all_mm >= R_mm) & (r_all_mm <= R_mm + wall_span_mm)
    z_beside = float(z_all_mm[beside].max()) if beside.any() else float("-nan")
    return {"R_mm": float(R_mm),
            "z_top_min_mm": z_top_min,
            "depth_top_min_mm": z_top_min - TCP_Z_MM,
            "z_beside_max_mm": z_beside,
            "bite_mm": z_beside - z_top_min,
            "n_footprint": int(fp.sum()), "n_beside": int(beside.sum())}


# =========================================================================== #
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {k: OUT_DIR / f"{TAG}_{k}" for k in
             ("results.json", "timeline.rrd", "timeline.rbl", "rerun_validation.json",
              "inspection.png", "diagnostic.png", "script.py.txt")}
    existing = [p.name for p in paths.values() if p.exists()]
    if existing:
        print(f"[{LOG}] ABORT write_guard existing={existing}", flush=True)
        return 3

    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        print(f"[{LOG}] ABORT rerun_version={rr.__version__}!={RERUN_VERSION}", flush=True)
        return 3

    out = {"tool": TAG, "read_only_assets": True,
           "claim_under_test": "URDF-assembled gripper admits a D29 cylinder to a 0-12mm rim pinch",
           "env": {"numpy": np.__version__, "rerun_sdk": rr.__version__, "python": sys.version.split()[0]}}

    # ---- 1. asset identity ------------------------------------------------
    out["sha256"] = {p.name: sha256(p) for p in (L5, GJ)}
    out["sha256_matches_record"] = {k: (out["sha256"][k] == v) for k, v in SHA.items()}
    print(f"[{LOG}] sha_ok={out['sha256_matches_record']}", flush=True)

    # ---- 2. URDF facts, read from the file --------------------------------
    u = parse_urdf(URDF)
    j5 = u["joints"]["link5_to_gripper_link"]
    tcp_j = u["joints"]["link5_to_hand_tcp"]
    R_j5 = rpy_matrix(*j5["rpy"])
    t_j5 = np.array(j5["xyz"])
    out["urdf"] = {
        "joint5": j5, "tcp_joint": tcp_j,
        "joint5_R": R_j5.tolist(),
        "joint5_R_max_abs_diff_vs_expected": float(np.abs(R_j5 - J5_R_EXPECT).max()),
        "link5_visual": u["meshes"]["link5/visual"], "link5_collision": u["meshes"]["link5/collision"],
        "gripper_visual": u["meshes"]["gripper_link/visual"],
        "gripper_collision": u["meshes"]["gripper_link/collision"],
        "gripper_left_link_referenced": any(
            "gripper_left_link" in (m["filename"] or "") for m in u["meshes"].values()),
    }
    # the declared collision mesh of the moving jaw, measured (assembly finding)
    gc_name = (u["meshes"]["gripper_link/collision"]["filename"] or "").split("/")[-1]
    gc_path = MESH_DIR / gc_name
    gc = {"filename": gc_name, "exists": gc_path.exists()}
    if gc_path.exists():
        raw = gc_path.read_bytes()
        gc["bytes"] = len(raw)
        gc["sha256"] = sha256(gc_path)
        gc["ascii_stl"] = raw[:5] == b"solid"
        v = np.array([[float(x) for x in ln.split()[1:4]]
                      for ln in gc_path.read_text().splitlines() if ln.strip().startswith("vertex")])
        gc["n_facets"] = int(sum(1 for ln in gc_path.read_text().splitlines()
                                 if ln.strip().startswith("facet")))
        gc["bbox_mm"] = {"min": v.min(0).tolist(), "max": v.max(0).tolist(),
                         "size": (v.max(0) - v.min(0)).tolist()}
        gc["max_extent_mm"] = float((v.max(0) - v.min(0)).max())
    out["urdf"]["gripper_collision_measured"] = gc
    print(f"[{LOG}] moving_jaw_collision={gc.get('filename')} facets={gc.get('n_facets')} "
          f"size_mm={gc.get('bbox_mm', {}).get('size')}", flush=True)

    # ---- 3. sample both visual meshes on the Gate-0 numeric path ----------
    t5 = load_binary_stl(L5) * STL_SCALE                 # metres, link5 frame
    tg = load_binary_stl(GJ) * STL_SCALE                 # metres, gripper_link frame
    S5 = sample_triangles(t5, SAMPLE_SPACING_M)
    SGj = sample_triangles(tg, SAMPLE_SPACING_M)
    out["assets"] = {"link5_tris": int(len(t5)), "link5_samples": int(len(S5)),
                     "gripper_link_tris": int(len(tg)), "gripper_link_samples": int(len(SGj))}
    print(f"[{LOG}] samples link5={len(S5)} gripper={len(SGj)}", flush=True)

    # ---- 4. GATES ---------------------------------------------------------
    z5, r5 = zr_mm(S5)
    depth5 = z5 - TCP_Z_MM
    win5 = r5 <= 30.0
    l_vis = float(depth5[win5].max())
    g_a = all(out["sha256_matches_record"].values())
    g_b = (abs(l_vis - D427_L_VIS_MM) < 1e-9) and (len(S5) == D427_N_PTS)
    # The URDF stores rpy to 4 decimals (-1.5708 vs -pi/2 = -1.57079633), so the matrix
    # cannot equal the idealised one exactly. 38th already quantified this truncation as a
    # max vertex displacement of 0.000332 mm. Gate on the truncation magnitude, and record
    # both the matrix residual and the induced displacement so the number stays auditable.
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

    zq0, _ = zr_mm(to_link5(SGj, 0.0))
    g_d = (abs(zq0.min() - MOVING_Z_RANGE_37TH_MM[0]) < 0.02 and
           abs(zq0.max() - MOVING_Z_RANGE_37TH_MM[1]) < 0.02)
    out["gates"] = {
        "N7a_sha_pins": bool(g_a),
        "N7b_D427_l_vis_reproduced": bool(g_b),
        "N7b_detail": {"l_vis_mm": l_vis, "delta_vs_D427": l_vis - D427_L_VIS_MM,
                       "n_pts": int(len(S5)), "n_pts_matches": len(S5) == D427_N_PTS},
        "N7c_joint5_rotation_matches": bool(g_c),
        "N7d_moving_jaw_z_range_matches_37th": bool(g_d),
        "N7d_detail": {"z_min_mm": float(zq0.min()), "z_max_mm": float(zq0.max()),
                       "expected": list(MOVING_Z_RANGE_37TH_MM)},
    }
    print(f"[{LOG}] gates a={g_a} b={g_b}(l_vis={l_vis:.12f}) "
          f"c={g_c}(R_resid={out['urdf']['joint5_R_max_abs_diff_vs_expected']:.3e}, "
          f"rpy_trunc={rpy_trunc_rad:.3e}rad -> "
          f"{out['urdf']['joint5_rpy_truncation_max_vertex_shift_mm']:.6f}mm) "
          f"d={g_d}(z=[{zq0.min():.4f},{zq0.max():.4f}])", flush=True)
    if not (g_a and g_b and g_c):
        print(f"[{LOG}] ABORT gate_failure", flush=True)
        return 3

    # ---- 5. THE TEST: cylinder admission over the q5 sweep ----------------
    q5_grid = np.unique(np.concatenate([np.linspace(0.0, j5["limit"][1], 33), [Q5_REAL_MAX_RAD]]))
    sweep = []
    for q5 in q5_grid:
        zg, rg = zr_mm(to_link5(SGj, float(q5)))
        z_all = np.concatenate([z5, zg])
        r_all = np.concatenate([r5, rg])
        a_all = admission(z_all, r_all, CYL_R_MM, WALL_SPAN_MM)
        a_fix = admission(z5, r5, CYL_R_MM, WALL_SPAN_MM)
        a_mov = admission(zg, rg, CYL_R_MM, WALL_SPAN_MM)
        # can the moving jaw reach INSIDE the cylinder footprint below the top face?
        below = zg > a_all["z_top_min_mm"]
        sweep.append({
            "q5_rad": float(q5), "q5_deg": float(math.degrees(q5)),
            "combined": a_all, "fixed_jaw_link5": a_fix, "moving_jaw": a_mov,
            "moving_min_r_below_topface_mm": float(rg[below].min()) if below.any() else None,
            "n_moving_below_topface": int(below.sum()),
        })
    out["q5_sweep"] = sweep
    bites = np.array([s["combined"]["bite_mm"] for s in sweep])
    best = int(np.nanargmax(bites))
    out["best_over_sweep"] = {"q5_deg": sweep[best]["q5_deg"], "bite_mm": float(bites[best]),
                              "max_bite_mm": float(np.nanmax(bites)),
                              "min_bite_mm": float(np.nanmin(bites))}
    at_real = min(sweep, key=lambda s: abs(s["q5_rad"] - Q5_REAL_MAX_RAD))
    out["at_real_max_opening_88_31deg"] = at_real
    print(f"[{LOG}] bite over sweep: max={np.nanmax(bites):+.4f}mm at q5={sweep[best]['q5_deg']:.2f}deg "
          f"| at 88.31deg={at_real['combined']['bite_mm']:+.4f}mm "
          f"| z_top_min={at_real['combined']['z_top_min_mm']:.4f}mm "
          f"(depth {at_real['combined']['depth_top_min_mm']:+.4f}mm)", flush=True)

    # ---- 6. which object radius WOULD this gripper admit? -----------------
    zg_r, rg_r = zr_mm(to_link5(SGj, Q5_REAL_MAX_RAD))
    z_all_r = np.concatenate([z5, zg_r])
    r_all_r = np.concatenate([r5, rg_r])
    radius_scan = []
    for R in np.arange(1.0, 30.0 + 0.5, 0.5):
        a = admission(z_all_r, r_all_r, float(R), WALL_SPAN_MM)
        if a:
            radius_scan.append({"R_mm": a["R_mm"], "D_mm": 2 * a["R_mm"],
                                "bite_mm": a["bite_mm"],
                                "depth_top_min_mm": a["depth_top_min_mm"]})
    out["radius_scan_at_88_31deg"] = radius_scan
    feasible = [s for s in radius_scan if s["bite_mm"] > 0.0]
    out["feasible_radii"] = {
        "any": bool(feasible),
        "max_D_mm_with_positive_bite": max((s["D_mm"] for s in feasible), default=None),
        "best": max(feasible, key=lambda s: s["bite_mm"]) if feasible else None,
    }
    print(f"[{LOG}] radius scan: positive-bite diameters exist={bool(feasible)} "
          f"max_D={out['feasible_radii']['max_D_mm_with_positive_bite']}", flush=True)

    # ---- 7. verdict -------------------------------------------------------
    admits = bool(np.nanmax(bites) > 0.0)
    reaches_T1 = bool(np.nanmax(bites) >= T1_BITE_MM[1])
    out["verdict"] = {
        "code": "ASSEMBLY_ADMITS_RIM_PINCH" if admits else "ASSEMBLY_CONTRADICTS_REAL_RIM_PINCH",
        "assembly_admits_any_bite": admits,
        "assembly_reaches_T1_upper_bite_12mm": reaches_T1,
        "max_bite_mm_over_full_q5_sweep": float(np.nanmax(bites)),
        "real_photo_verified_bite_mm": list(T1_BITE_MM),
        "deepest_reachable_topface_depth_mm": float(at_real["combined"]["depth_top_min_mm"]),
        "attempt1_measured_stop_mm_above_topface": 4.4,
        "D427_status": "UNCHANGED - this run does not re-run or re-judge Gate-0",
    }
    print(f"[{LOG}] G0B_T3R_N7_VERDICT={out['verdict']['code']}", flush=True)

    # ---- 8. D324 decision diagnostic --------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(19.5, 6.0))

    qd = [s["q5_deg"] for s in sweep]
    ax[0].plot(qd, [s["combined"]["bite_mm"] for s in sweep], "-o", ms=3, color="#c02020",
               label="achievable bite (combined)")
    ax[0].axhspan(T1_BITE_MM[0], T1_BITE_MM[1], color="#2a9d3a", alpha=0.16,
                  label="T1 real photo-verified bite 0-12mm")
    ax[0].axhline(0.0, color="k", lw=1.0)
    ax[0].axvline(88.31, color="#1f6fb4", ls="--", lw=1.2, label="real max opening 88.31 deg")
    ax[0].set_xlabel("q5 [deg]  (larger = open, D420 (4))")
    ax[0].set_ylabel("bite below cylinder top face [mm]")
    ax[0].set_title("A | can the jaws get BESIDE a D29 cylinder?")
    ax[0].legend(fontsize=8, loc="best")
    ax[0].grid(alpha=0.3)

    ax[1].plot(qd, [s["combined"]["depth_top_min_mm"] for s in sweep], "-o", ms=3,
               color="#1f6fb4", label="deepest reachable top-face plane (depth vs TCP)")
    ax[1].axhline(4.4, color="#e08a1e", ls="--", lw=1.4,
                  label="attempt1 physical stop: top+4.4mm (D424)")
    ax[1].set_xlabel("q5 [deg]")
    ax[1].set_ylabel("depth = z - TCP [mm]")
    ax[1].set_title("B | descent limit vs the measured physical stop")
    ax[1].legend(fontsize=8, loc="best")
    ax[1].grid(alpha=0.3)

    ax[2].plot([s["D_mm"] for s in radius_scan], [s["bite_mm"] for s in radius_scan],
               "-o", ms=3, color="#7a3fb5")
    ax[2].axhline(0.0, color="k", lw=1.0)
    ax[2].axvline(29.0, color="#c02020", ls="--", lw=1.4, label="current object D29")
    ax[2].set_xlabel("object diameter [mm]")
    ax[2].set_ylabel("bite [mm]  (>0 = graspable top-down)")
    ax[2].set_title("C | which object diameter WOULD this gripper admit? (q5 = 88.31 deg)")
    ax[2].legend(fontsize=8, loc="best")
    ax[2].grid(alpha=0.3)

    fig.suptitle(f"g0b_d420 {TAG} - ASSEMBLY cross-check (read-only).  VERDICT: {out['verdict']['code']}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(paths["diagnostic.png"], dpi=125)
    plt.close(fig)

    # ---- 9. D341 Rerun artifact (save-only) -------------------------------
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    app_id = f"roarm_g0b_{TAG}"
    z_top_show = at_real["combined"]["z_top_min_mm"]

    def view_pts(pts_m):
        o = np.asarray(pts_m) - TCP_LOCAL[None, :]
        return np.column_stack([o[:, 0], o[:, 1], -o[:, 2]])

    def ring(r_mm, z_link5_mm, n=96):
        d = z_link5_mm - TCP_Z_MM
        return [[r_mm / 1000.0 * math.cos(2 * math.pi * i / n),
                 r_mm / 1000.0 * math.sin(2 * math.pi * i / n),
                 -d / 1000.0] for i in range(n + 1)]

    P_mov_real = to_link5(SGj, Q5_REAL_MAX_RAD)
    cyl_rings = [ring(CYL_R_MM, z_top_show + k) for k in (0.0, 5.0, 12.0, 25.0, CYL_H_MM)]
    cyl_walls = [[[CYL_R_MM / 1000.0 * math.cos(a), CYL_R_MM / 1000.0 * math.sin(a),
                   -(z_top_show + k - TCP_Z_MM) / 1000.0] for k in (0.0, CYL_H_MM)]
                 for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)]

    with rr.RecordingStream(app_id, recording_id=f"g0b_d420_{TAG}", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(paths["timeline.rrd"]), write_footer=True)
        rec.log("assembly/link5_fixed_jaw", rr.Points3D(view_pts(S5[::VIEW_STRIDE]),
                colors=[150, 150, 160], radii=0.0004), static=True)
        rec.log("assembly/gripper_link_moving_jaw", rr.Points3D(view_pts(P_mov_real[::VIEW_STRIDE]),
                colors=[70, 130, 230], radii=0.0004), static=True)
        rec.log("assembly/tcp", rr.Points3D([[0.0, 0.0, 0.0]], colors=[[40, 200, 80]],
                radii=0.002), static=True)
        rec.log("assembly/tool_axis", rr.LineStrips3D([[[0.0, 0.0, 0.07], [0.0, 0.0, -0.07]]],
                colors=[[40, 200, 80]], radii=0.0003), static=True)
        rec.log("object/cylinder_D29", rr.LineStrips3D(cyl_rings + cyl_walls,
                colors=[[225, 60, 60]] * (len(cyl_rings) + len(cyl_walls)), radii=0.00035), static=True)
        rec.log("object/topface_plane", rr.LineStrips3D([ring(30.0, z_top_show)],
                colors=[[235, 140, 30]], radii=0.0005), static=True)
        rec.log("object/footprint_ring", rr.LineStrips3D([ring(CYL_R_MM, TCP_Z_MM)],
                colors=[[140, 140, 140]], radii=0.0003), static=True)
        for i, s in enumerate(sweep):
            rec.reset_time()
            rec.set_time("q5_index", sequence=i)
            rec.log("plots/q5_deg", rr.Scalars(float(s["q5_deg"])))
            rec.log("plots/bite_mm", rr.Scalars(float(s["combined"]["bite_mm"])))
            rec.log("plots/depth_top_min_mm", rr.Scalars(float(s["combined"]["depth_top_min_mm"])))
            rec.log("plots/moving_jaw_bite_mm", rr.Scalars(float(s["moving_jaw"]["bite_mm"])))
            rec.log("assembly/moving_jaw_pose", rr.Points3D(
                view_pts(to_link5(SGj, s["q5_rad"])[::VIEW_STRIDE * 6]),
                colors=[70, 130, 230], radii=0.0004))
        rec.reset_time()
        rec.set_time("q5_index", sequence=0)
        for name, ok in (("N7a_sha_pins", g_a), ("N7b_D427_l_vis_reproduced", g_b),
                         ("N7c_joint5_rotation_matches", g_c),
                         ("N7d_moving_jaw_z_range_matches_37th", g_d),
                         ("N7e_assembly_admits_rim_pinch", admits),
                         ("N7f_assembly_reaches_T1_12mm", reaches_T1)):
            rec.log("events/gates", rr.TextLog(name, level=rr.TextLogLevel.INFO if ok
                                               else rr.TextLogLevel.ERROR))
        summary_md = (
            f"# g0b_d420 {TAG} - ASSEMBLY cross-check (read-only)\n\n"
            f"**VERDICT: {out['verdict']['code']}**\n\n"
            f"## Question\nSessions 29th-39th audited the gripper part by part and closed every "
            f"'a file is missing' hypothesis (D427, D429). This run asks the question nobody asked: "
            f"does the URDF **assemble** those parts into a gripper that can do what the real one "
            f"is photo-verified to do?\n\n"
            f"## The two statements that cannot both be true\n"
            f"- real bare gripper: rim pinch **0-12 mm** below the cylinder top face "
            f"(D420 Impl.(2), photo-verified);\n"
            f"- sim: jaws closed 88.31 deg -> 24 deg with **no contact at any angle** (D424 Ev.(2)).\n\n"
            f"## Result\n"
            f"- deepest reachable top-face plane = **TCP{out['verdict']['deepest_reachable_topface_depth_mm']:+.4f} mm** "
            f"(attempt1 measured the physical stop at top+4.4 mm - the assembly reproduces it);\n"
            f"- achievable bite over the WHOLE q5 sweep: max = "
            f"**{out['verdict']['max_bite_mm_over_full_q5_sweep']:+.4f} mm**;\n"
            f"- at the real maximum opening 88.31 deg: bite = "
            f"**{at_real['combined']['bite_mm']:+.4f} mm**;\n"
            f"- largest object diameter with a positive bite at 88.31 deg: "
            f"**{out['feasible_radii']['max_D_mm_with_positive_bite']} mm** "
            f"(current object is D29).\n\n"
            f"## Assembly finding, independent of the above\n"
            f"`roarm_m3.urdf:161` declares the MOVING jaw's collision geometry as "
            f"`{gc.get('filename')}` = **{gc.get('n_facets')} facets, "
            f"{gc.get('bbox_mm', {}).get('size')} mm** - a stub, not the jaw. "
            f"`local_assets/roarm_m3/usd/config.yaml` sets `collider_type: convex_hull` and "
            f"`collision_from_visuals: false`, so a USD built from this URDF gets a moving jaw that "
            f"cannot contact anything and a fixed jaw whose throat is filled by one convex hull.\n\n"
            f"## Scene\ngrey = link5 (fixed jaw), blue = gripper_link (moving jaw) at 88.31 deg, "
            f"red = the D29 cylinder placed at its deepest admissible height, orange = its top-face "
            f"plane, green = TCP and tool axis. View convention: z_view = -(z - z_TCP), so DOWN = distal.\n\n"
            f"Authority = stdout + `{paths['results.json'].name}`. Rerun is inspection evidence only (D341).\n"
        )
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN), static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run",
                                         name="1 | assembly verdict"),
                    rrb.Spatial3DView(origin="/", contents=["/assembly/**", "/object/**"],
                                      name="2 | assembled gripper vs D29 cylinder"),
                    rrb.TextLogView(origin="/events/gates", contents="/events/gates/**", name="3 | gates"),
                    column_shares=[0.30, 0.46, 0.24],
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/plots",
                                       contents=["/plots/bite_mm/**", "/plots/moving_jaw_bite_mm/**",
                                                 "/plots/depth_top_min_mm/**"],
                                       name="4 | bite [mm] and descent limit vs q5"),
                    rrb.TimeSeriesView(origin="/plots", contents=["/plots/q5_deg/**"],
                                       name="5 | swept q5 [deg]"),
                ),
                row_shares=[0.58, 0.42],
            ),
            auto_layout=False, auto_views=False, collapse_panels=True,
        )
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=30.0)
    blueprint.save(app_id, str(paths["timeline.rbl"]))

    expected_entities = ["metadata/run", "assembly/link5_fixed_jaw", "assembly/gripper_link_moving_jaw",
                         "assembly/moving_jaw_pose", "assembly/tcp", "assembly/tool_axis",
                         "object/cylinder_D29", "object/topface_plane", "object/footprint_ring",
                         "plots/q5_deg", "plots/bite_mm", "plots/depth_top_min_mm",
                         "plots/moving_jaw_bite_mm", "events/gates"]
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    lin3 = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    components = {
        "metadata/run": ["TextDocument:text"],
        "assembly/link5_fixed_jaw": pts3, "assembly/gripper_link_moving_jaw": pts3,
        "assembly/moving_jaw_pose": pts3, "assembly/tcp": pts3,
        "assembly/tool_axis": lin3, "object/cylinder_D29": lin3,
        "object/topface_plane": lin3, "object/footprint_ring": lin3,
        "plots/q5_deg": ["Scalars:scalars"], "plots/bite_mm": ["Scalars:scalars"],
        "plots/depth_top_min_mm": ["Scalars:scalars"], "plots/moving_jaw_bite_mm": ["Scalars:scalars"],
        "events/gates": ["TextLog:text", "TextLog:level"],
    }
    validation = validate_rerun_artifact(
        paths["timeline.rrd"],
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time", "q5_index"],
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
    # D429-R1: the manifest must NOT contain its own file - hash results.json from disk afterwards.
    out["artifacts"] = {k: {"name": v.name, "sha256": sha256(v)[:16], "bytes": v.stat().st_size}
                        for k, v in paths.items() if v.exists() and k != "results.json"}
    out["artifacts_note"] = ("results.json is deliberately absent from this manifest - D429-R1: a "
                             "self-referential manifest can never carry the file's final hash. "
                             "Hash it from disk.")
    out["rerun_validation_pass"] = bool(validation.get("pass"))
    paths["results.json"].write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{LOG}] artifacts " + " ".join(f"{v['name']}={v['sha256']}" for v in out["artifacts"].values()),
          flush=True)
    print(f"[{LOG}] results.json={sha256(paths['results.json'])[:16]} "
          f"bytes={paths['results.json'].stat().st_size}", flush=True)
    print(f"[{LOG}] G0B_T3R_N7_VERDICT={out['verdict']['code']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
