#!/usr/bin/env python3
"""o1 — POSCO-yard pivot O-step: procedural irregular convex rock generator.

63rd doc SS7 + 66th session contract.  Generates 52 deterministic convex
polyhedra ("rock analogues") in 4 grasp-width classes 22/26/30/34 mm x 13 each.

Design rules
  - grasp width = MIN support width of the hull, scaled exactly to the class
    value (the jaw grasps across the narrowest orientation; the 40-45 mm jaw
    opening constrains this width, not the longest axis).
  - max extent capped at 1.5 x class (pile behavior; absolute cap 52 mm).
  - sim <-> real identical mesh: the canonical unit is the vertex/face arrays
    recorded in manifest.json (meters); the STL is the same mesh in mm.
  - determinism: per-object seed = class_mm * 1000 + index; a regeneration
    gate proves bit-identical vertices for the first object of every class.
  - material contract (print step): PLA, light matte grey - NO black / NO
    gloss (Azure Kinect ToF readability, 63rd SS7).  Mass estimates recorded
    at solid and ~15 % infill; real printed mass must be weighed and written
    back before any sim mass claim (D446-style sim<->real fidelity).

Outputs (sim_assets/posco_rocks_o1/):
  stl/rock_<class>_<idx>.stl   (binary STL, mm)  x 52
  preview/class_<class>.png    (matplotlib tri preview, first 3 per class)
  manifest.json                (per-object geometry + gates + SHA-256)
  README.md                    (print + material instructions)
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "sim_assets/posco_rocks_o1"
STL_DIR = OUT_DIR / "stl"
PREV_DIR = OUT_DIR / "preview"

CLASSES_MM = (22, 26, 30, 34)
PER_CLASS = 13
MAX_EXTENT_FACTOR = 1.5
MAX_EXTENT_ABS_MM = 52.0
N_DIRS = 4096
PLA_G_CM3 = 1.24
INFILL_FACTOR = 0.40  # ~2 walls + 15 % infill rough estimate (record-only)
MAX_RESAMPLE = 20


def fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    phi = (1 + 5 ** 0.5) / 2
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    th = 2 * np.pi * i / phi
    return np.column_stack([r * np.cos(th), r * np.sin(th), z])


DIRS = fib_sphere(N_DIRS)


def widths(verts: np.ndarray) -> tuple[float, float]:
    proj = verts @ DIRS.T
    w = proj.max(axis=0) - proj.min(axis=0)
    return float(w.min()), float(w.max())


def gen_object(class_mm: int, idx: int) -> dict:
    seed = class_mm * 1000 + idx
    rng = np.random.default_rng(seed)
    for attempt in range(MAX_RESAMPLE):
        n_pts = int(rng.integers(24, 37))
        d = rng.normal(size=(n_pts, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        axes = np.array([1.0, rng.uniform(0.78, 0.95), rng.uniform(0.68, 0.88)])
        pts = d * axes * (1.0 + rng.uniform(-0.12, 0.12, size=(n_pts, 1)))
        hull = ConvexHull(pts)
        v = pts[hull.vertices]
        # reindex faces to the reduced vertex set
        remap = {int(o): k for k, o in enumerate(hull.vertices)}
        faces = np.array([[remap[int(a)] for a in tri] for tri in hull.simplices],
                         dtype=np.int64)
        wmin, wmax = widths(v)
        scale = (class_mm / 1000.0) / wmin
        v = v * scale
        wmin2, wmax2 = widths(v)
        max_cap = min(MAX_EXTENT_FACTOR * class_mm, MAX_EXTENT_ABS_MM) / 1000.0
        if wmax2 <= max_cap:
            hull2 = ConvexHull(v)
            vol = float(hull2.volume)
            # outward-orient faces
            c = v.mean(axis=0)
            oriented = []
            for tri in faces:
                a, b, cc = v[tri[0]], v[tri[1]], v[tri[2]]
                n = np.cross(b - a, cc - a)
                if np.dot(n, a - c) < 0:
                    tri = tri[[0, 2, 1]]
                oriented.append(tri.tolist())
            return {
                "name": f"rock_{class_mm}_{idx:02d}", "class_mm": class_mm,
                "index": idx, "seed": seed, "resample_attempts": attempt,
                "n_vertices": int(len(v)), "n_faces": int(len(oriented)),
                "grasp_min_width_mm": wmin2 * 1000.0,
                "max_extent_mm": wmax2 * 1000.0,
                "volume_cm3": vol * 1e6,
                "mass_solid_g": vol * 1e6 * PLA_G_CM3,
                "mass_est_15infill_g": vol * 1e6 * PLA_G_CM3 * INFILL_FACTOR,
                "vertices_m": v.tolist(), "faces": oriented,
            }
    raise RuntimeError(f"RESAMPLE_EXHAUSTED class={class_mm} idx={idx}")


def write_stl(path: Path, verts_m: np.ndarray, faces) -> str:
    v = verts_m * 1000.0  # mm
    with open(path, "wb") as f:
        f.write(b"o1 posco rock (mm)".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        for tri in faces:
            a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
            n = np.cross(b - a, c - a)
            ln = np.linalg.norm(n)
            n = n / ln if ln > 0 else n
            f.write(struct.pack("<3f", *n))
            for p in (a, b, c):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def preview(rows, class_mm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sel = [r for r in rows if r["class_mm"] == class_mm][:3]
    fig = plt.figure(figsize=(12, 4))
    for k, r in enumerate(sel):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        v = np.array(r["vertices_m"]) * 1000.0
        polys = [[v[i] for i in tri] for tri in r["faces"]]
        ax.add_collection3d(Poly3DCollection(polys, facecolor="#b9b9b9",
                                             edgecolor="#404040", linewidths=0.6))
        m = np.abs(v).max()
        ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
        ax.set_title(f"{r['name']}  w={r['grasp_min_width_mm']:.1f} mm  "
                     f"L={r['max_extent_mm']:.1f} mm", fontsize=8)
        ax.set_box_aspect((1, 1, 1))
    fig.suptitle(f"class {class_mm} mm — first 3 of {PER_CLASS}")
    fig.tight_layout()
    p = PREV_DIR / f"class_{class_mm}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def main() -> int:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise RuntimeError(f"WRITE_GUARD {OUT_DIR} not empty")
    STL_DIR.mkdir(parents=True, exist_ok=True)
    PREV_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for cm in CLASSES_MM:
        for i in range(PER_CLASS):
            rows.append(gen_object(cm, i))

    # gates -------------------------------------------------------------- #
    g_width = all(abs(r["grasp_min_width_mm"] - r["class_mm"]) < 0.01 for r in rows)
    g_extent = all(r["max_extent_mm"] <= min(MAX_EXTENT_FACTOR * r["class_mm"],
                                             MAX_EXTENT_ABS_MM) + 1e-6 for r in rows)
    # determinism: regenerate idx 0 of every class, compare bit-exact
    g_det = True
    for cm in CLASSES_MM:
        again = gen_object(cm, 0)
        ref = next(r for r in rows if r["class_mm"] == cm and r["index"] == 0)
        if (np.array(again["vertices_m"]) != np.array(ref["vertices_m"])).any():
            g_det = False
    # convexity/watertight: hull by construction; verify Euler V-E+F=2
    g_euler = True
    for r in rows:
        E = len({tuple(sorted((tri[a], tri[(a + 1) % 3]))) for tri in r["faces"]
                 for a in range(3)})
        if r["n_vertices"] - E + r["n_faces"] != 2:
            g_euler = False
    if not (g_width and g_extent and g_det and g_euler):
        raise RuntimeError(f"GATE_FAIL width={g_width} extent={g_extent} "
                           f"det={g_det} euler={g_euler}")

    for r in rows:
        sha = write_stl(STL_DIR / f"{r['name']}.stl",
                        np.array(r["vertices_m"]), r["faces"])
        r["stl_sha256_16"] = sha[:16]

    previews = [preview(rows, cm) for cm in CLASSES_MM]

    stats = {}
    for cm in CLASSES_MM:
        rs = [r for r in rows if r["class_mm"] == cm]
        stats[str(cm)] = {
            "n": len(rs),
            "max_extent_mm_range": [round(min(r["max_extent_mm"] for r in rs), 1),
                                    round(max(r["max_extent_mm"] for r in rs), 1)],
            "mass_solid_g_range": [round(min(r["mass_solid_g"] for r in rs), 1),
                                   round(max(r["mass_solid_g"] for r in rs), 1)],
            "mass_15infill_g_range": [round(min(r["mass_est_15infill_g"] for r in rs), 1),
                                      round(max(r["mass_est_15infill_g"] for r in rs), 1)],
            "n_vertices_range": [min(r["n_vertices"] for r in rs),
                                 max(r["n_vertices"] for r in rs)]}

    manifest = {
        "tool": "o1_posco_rock_generator", "step": "O-step (63rd SS7, 66th)",
        "classes_mm": list(CLASSES_MM), "per_class": PER_CLASS,
        "total": len(rows),
        "gates": {"grasp_width_exact": g_width, "max_extent_cap": g_extent,
                  "determinism_bitexact": g_det, "euler_watertight": g_euler},
        "design_rules": {
            "grasp_width_def": "min support width over 4096 fibonacci directions, "
                               "scaled exactly to class",
            "max_extent_cap": f"{MAX_EXTENT_FACTOR}x class, abs {MAX_EXTENT_ABS_MM} mm",
            "material": "PLA light matte grey; NO black, NO gloss (Kinect ToF)",
            "mass_note": "solid + ~15% infill estimates only; weigh real prints "
                         "and write masses back before any sim mass claim"},
        "class_stats": stats,
        "previews": previews,
        "objects": rows,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    readme = f"""# posco_rocks_o1 — 비정형 convex 암석 유사체 (O-step)

- 52개 = 클래스 22/26/30/34 mm x 13. 파지 폭(min width) = 클래스값 정확 스케일.
- 프린트: PLA **무광 밝은 회색** (흑색/광택 절대 금지 — Azure Kinect ToF 판독성).
  권장 15% infill + 2 walls. STL 단위 = mm.
- 프린트 후 **개당 실측 질량을 manifest에 기록**한 뒤에만 sim 질량 사용
  (sim<->real 동일 메쉬 원칙: manifest.json vertices_m/faces가 정본).
- 게이트: 폭 정확 / 최장축 <=1.5x클래스 / 시드 결정론 bit-동일 / Euler 폐합.
"""
    (OUT_DIR / "README.md").write_text(readme)

    man_sha = hashlib.sha256((OUT_DIR / "manifest.json").read_bytes()).hexdigest()
    print(f"[o1] GATES width={g_width} extent={g_extent} det={g_det} euler={g_euler}")
    for cm in CLASSES_MM:
        print(f"[o1] class {cm}: {stats[str(cm)]}")
    print(f"[o1] total {len(rows)} objects, manifest sha256_16={man_sha[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
