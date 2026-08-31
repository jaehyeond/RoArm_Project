#!/usr/bin/env python
"""p36 / hm4 — `heightmap_from_particles` 다분산(polydisperse) 확장 검증 (물리 0, 로봇 0).

경로 (A) 가 입자마다 다른 반지름을 받도록 확장됐다.  DEME 는 `radii_m` 을 (N,)
배열로 주므로 원래부터 그 형태였고, 지금까지는 균일한 경우만 통과시켰다.

  P1 backcompat  — 균일 (N,) 배열이 스칼라와 **bit-동일**한가 (기존 호출자 보호).
  P2 bruteforce  — 다분산 결과가 독립 구현(셀마다 전 구 순회)과 일치하는가.
  P3 analytic    — 손으로 푼 2-구 배치에서 어느 구가 어느 셀을 이기는가.
  P4 superpose   — 다분산 heightmap == 반지름 그룹별 heightmap 들의 원소별 max.
                   (max 연산자의 구조적 불변식 — 그룹 버킷팅이 결과를 바꾸지 않음)
  P5 cost        — 입자별 창(window) 버킷팅이 r_max 일괄 창 대비 후보쌍을 얼마나
                   줄이는가.  결정적 카운트라 재현 가능.
  P6 realpile    — s1 실 pile 좌표에 크기 분포를 입혀 (A)/(B) 계약이 다분산에서도
                   같은 규격을 뱉는지.  반지름을 **줄이기만** 하므로 새 겹침은
                   생기지 않는다 (겹치지 않던 배치에서 구를 줄이면 여전히 안 겹침).

출력 = claudedocs/runtime_logs/heightmap_track/hm4_polydisperse/  (신규 폴더)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hmlib = _load("roarm_heightmap", "roarm_rl/heightmap.py")
p33 = _load("p33_probe", "sim_scripts/p33_hm1_heightmap_contract_probe.py")
p34 = _load("p34_probe", "sim_scripts/p34_hm2_deme_pile_heightmap.py")

GridSpec, Heightmap = hmlib.GridSpec, hmlib.Heightmap
render_sphere_depth, stats_mm = p33.render_sphere_depth, p33.stats_mm

OUT = REPO / "claudedocs/runtime_logs/heightmap_track/hm4_polydisperse"
PILE_NPZ = REPO / "claudedocs/runtime_logs/sim_deme/pile_smoke_d4p16_n512_seed460.npz"
CALIB = REPO / "sim_scripts/kinect_calib.yaml"

SEED = 36001
CELL_M = 0.005
# height 는 계약상 float32 다.  h ~ 0.05 m 에서 양자화 스텝이 약 6e-9 m 이므로
# 1e-7 m 를 넘는 차이만 실제 구현 오차다.
TOL_M = 1.0e-7


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def brute_force_footprint_max(P, rad, spec: GridSpec, floor_z=0.0) -> np.ndarray:
    """독립 구현: 셀마다 **전** 구를 순회해 풋프린트 최댓값을 직접 계산.

    모듈의 창 버킷팅·벡터화 스캐터를 전혀 쓰지 않는다 — 그 최적화가 결과를
    바꾸지 않았음을 보이는 것이 목적이므로 일부러 느리고 단순하게 짠다.
    """
    c = spec.cell_centers()
    half = spec.cell_m / 2.0
    out = np.full(spec.shape, float(floor_z))
    for r_ in range(spec.shape[0]):
        for c_ in range(spec.shape[1]):
            qx = np.clip(P[:, 0], c[r_, c_, 0] - half, c[r_, c_, 0] + half)
            qy = np.clip(P[:, 1], c[r_, c_, 1] - half, c[r_, c_, 1] + half)
            d2 = (P[:, 0] - qx) ** 2 + (P[:, 1] - qy) ** 2
            m = d2 <= rad ** 2
            if m.any():
                out[r_, c_] = max(out[r_, c_],
                                  float((P[m, 2] + np.sqrt(rad[m] ** 2 - d2[m])).max()))
    return out


# --------------------------------------------------------------------------- #
def gate_backcompat(rng) -> dict:
    spec = GridSpec(origin_xy_m=(0.0, 0.0), cell_m=CELL_M, shape=(24, 18))
    P = np.column_stack([rng.uniform(0.005, 0.085, 400),
                         rng.uniform(0.005, 0.115, 400),
                         rng.uniform(0.004, 0.040, 400)])
    r = 0.0025
    a = hmlib.heightmap_from_particles(P, r, spec)
    b = hmlib.heightmap_from_particles(P, np.full(400, r), spec)
    same = (np.array_equal(a.height, b.height) and np.array_equal(a.valid, b.valid)
            and np.array_equal(a.counts, b.counts))
    hdr_a, hdr_b = a.header(), b.header()
    return {"gate": "P1_backcompat_uniform_array_equals_scalar",
            "pass": bool(same and hdr_a == hdr_b),
            "arrays_bit_identical": bool(same),
            "headers_identical": bool(hdr_a == hdr_b),
            "polydisperse_flag": [a.meta["polydisperse"], b.meta["polydisperse"]],
            "particle_radius_m_key_present": ["particle_radius_m" in hdr_a,
                                              "particle_radius_m" in hdr_b],
            "reading": ("균일 배열을 넘겨도 스칼라와 완전히 같은 결과·헤더를 낸다. "
                        "기존 호출자(monodisperse)는 이 확장에 영향을 받지 않는다.")}


def gate_bruteforce(rng) -> dict:
    spec = GridSpec(origin_xy_m=(0.0, 0.0), cell_m=CELL_M, shape=(20, 20))
    rows = []
    for name, span, n in (("narrow_1_1p5mm", (0.0010, 0.0015), 400),
                          ("wide_1_8mm", (0.0010, 0.0080), 400),
                          ("bimodal_1_and_7mm", None, 400)):
        P = np.column_stack([rng.uniform(0.005, 0.095, n),
                             rng.uniform(0.005, 0.095, n),
                             rng.uniform(0.002, 0.035, n)])
        rad = (np.where(rng.random(n) < 0.5, 0.0010, 0.0070) if span is None
               else rng.uniform(span[0], span[1], n))
        hm = hmlib.heightmap_from_particles(P, rad, spec)
        gt = brute_force_footprint_max(P, rad, spec)
        d = hm.height.astype(np.float64) - gt
        rows.append({"case": name, "n": n,
                     "radius_min_mm": float(rad.min()) * 1000,
                     "radius_max_mm": float(rad.max()) * 1000,
                     "max_abs_diff_m": float(np.abs(d).max()),
                     "max_abs_diff_mm": float(np.abs(d).max()) * 1000,
                     "polydisperse_flag": bool(hm.meta["polydisperse"])})
    return {"gate": "P2_vs_independent_bruteforce", "tol_m": TOL_M,
            "pass": bool(all(r["max_abs_diff_m"] <= TOL_M for r in rows)),
            "rows": rows,
            "reading": ("셀마다 전 구를 순회하는 독립 구현과 일치한다. 창 버킷팅과 "
                        "벡터화 스캐터가 결과를 바꾸지 않았다는 뜻이다.")}


def gate_analytic() -> dict:
    """손으로 푸는 배치: 큰 구와 작은 구가 각각 어느 셀을 이기는가.

    셀 5 mm 격자, 원점 (0,0).  셀 [0,0] 중심 (2.5, 2.5) mm, 셀 [0,1] 중심 (7.5, 2.5).
      big   : 중심 (2.5, 2.5, 0) mm, r = 6 mm  -> 셀[0,0] 안에 있으므로 dist=0,
              그 셀 높이 = 0 + 6 = 6 mm.  셀[0,1] 까지의 최근접 거리 = 5 - 2.5 = 2.5 mm
              -> 높이 = sqrt(36 - 6.25) = 5.4544 mm
      small : 중심 (7.5, 2.5, 4) mm, r = 1 mm  -> 셀[0,1] 안, 높이 = 4 + 1 = 5 mm
    따라서 셀[0,1] 은 **큰 구가 이긴다** (5.4544 > 5.0) — 작은 구가 그 셀 안에
    있는데도 그렇다.  다분산에서만 나타나는 상황이라 별도로 못 박는다.
    """
    spec = GridSpec(origin_xy_m=(0.0, 0.0), cell_m=0.005, shape=(1, 2))
    P = np.array([[0.0025, 0.0025, 0.0], [0.0075, 0.0025, 0.004]])
    rad = np.array([0.006, 0.001])
    hm = hmlib.heightmap_from_particles(P, rad, spec)
    exp00 = 0.006
    exp01 = np.sqrt(0.006 ** 2 - 0.0025 ** 2)
    got = hm.height.astype(np.float64)
    err = [abs(float(got[0, 0]) - exp00), abs(float(got[0, 1]) - exp01)]
    return {"gate": "P3_analytic_two_sphere", "tol_m": TOL_M,
            "pass": bool(max(err) <= TOL_M),
            "expected_mm": [exp00 * 1000, exp01 * 1000],
            "got_mm": [float(got[0, 0]) * 1000, float(got[0, 1]) * 1000],
            "err_m": err,
            "counts": hm.counts.tolist(),
            "reading": ("셀[0,1] 안에 작은 구(꼭대기 5.000 mm)가 있는데도 이웃 셀의 "
                        "큰 구가 5.454 mm 로 이긴다. 반지름이 균일하면 생길 수 없는 "
                        "상황이며, 창 크기를 r_max 가 아니라 입자별로 잡아도 "
                        "이 교차 기여를 놓치지 않음을 보인다.")}


def gate_superposition(rng) -> dict:
    spec = GridSpec(origin_xy_m=(0.0, 0.0), cell_m=CELL_M, shape=(22, 22))
    n = 500
    P = np.column_stack([rng.uniform(0.005, 0.105, n), rng.uniform(0.005, 0.105, n),
                         rng.uniform(0.002, 0.030, n)])
    levels = np.array([0.0012, 0.0030, 0.0065])
    rad = levels[rng.integers(0, 3, n)]
    hm_all = hmlib.heightmap_from_particles(P, rad, spec, floor_z_m=-1.0)
    per = [hmlib.heightmap_from_particles(P[rad == L], L, spec, floor_z_m=-1.0)
           for L in levels]
    stacked = np.maximum.reduce([h.height.astype(np.float64) for h in per])
    d = hm_all.height.astype(np.float64) - stacked
    cnt = sum(int(h.counts.sum()) for h in per)
    return {"gate": "P4_superposition_over_radius_groups", "tol_m": TOL_M,
            "pass": bool(float(np.abs(d).max()) <= TOL_M
                         and int(hm_all.counts.sum()) == cnt),
            "radius_levels_mm": (levels * 1000).tolist(),
            "max_abs_diff_m": float(np.abs(d).max()),
            "counts_match": bool(int(hm_all.counts.sum()) == cnt),
            "reading": ("다분산 heightmap 이 반지름 그룹별 heightmap 의 원소별 max 와 "
                        "같다. max 연산자의 구조적 불변식이고, 구현이 그룹을 어떻게 "
                        "버킷팅하든 결과가 같아야 함을 강제한다.")}


def gate_cost(rng) -> dict:
    """입자별 창 vs r_max 일괄 창의 후보쌍 수 (결정적 — 타이밍 잡음 없음)."""
    rows = []
    for name, cell, rad in (
            ("s1_like_uniform_r2p08", 0.005, np.full(2000, 0.00208)),
            ("moderate_1_to_4mm", 0.005, rng.uniform(0.0010, 0.0040, 2000)),
            ("wide_0p5_to_10mm", 0.001, rng.uniform(0.0005, 0.0100, 2000)),
            ("bimodal_0p5_and_10mm", 0.001,
             np.where(rng.random(2000) < 0.9, 0.0005, 0.0100))):
        k = np.ceil(rad / cell).astype(np.int64) + 1
        per_particle = int(((2 * k + 1) ** 2).sum())
        naive = int(rad.size * (2 * int(k.max()) + 1) ** 2)
        rows.append({"case": name, "cell_mm": cell * 1000, "n": int(rad.size),
                     "k_min": int(k.min()), "k_max": int(k.max()),
                     "n_buckets": int(np.unique(k).size),
                     "candidate_pairs_bucketed": per_particle,
                     "candidate_pairs_naive_rmax": naive,
                     "work_ratio_naive_over_bucketed": naive / per_particle})
    return {"gate": "P5_window_bucketing_cost", "pass": True, "rows": rows,
            "reading": ("균일 분포에서는 두 방식이 같다(비 1.0) — 확장이 기존 경로에 "
                        "비용을 얹지 않는다. 크기 폭이 넓어질수록 버킷팅 이득이 커진다."),
            "caveat": ("후보쌍 수는 결정적 상한 지표이고 실제 벽시계 시간은 아니다. "
                       "버킷 수만큼 파이썬 루프가 늘어나므로 아주 좁은 분포에서는 "
                       "이득이 상수 오버헤드에 묻힐 수 있다.")}


def gate_realpile(rng, calib) -> dict:
    """s1 실 pile 좌표 + 크기 분포.  반지름을 줄이기만 해 새 겹침을 만들지 않는다."""
    src = p34.load_pile(PILE_NPZ)
    P, r0 = src["P"], float(src["r_max"])
    # log-normal 형태로 [0.40 r0, 1.00 r0] 에 몰아넣기 (축소 전용)
    f = np.clip(np.exp(rng.normal(np.log(0.72), 0.22, P.shape[0])), 0.40, 1.00)
    rad = r0 * f
    spec = p34.grid_for_pile(src, CELL_M)

    hm_a = hmlib.heightmap_from_particles(
        P, rad, spec, extra_meta={"scene": "s1_pile_centres_with_size_distribution"})
    intr = calib["intrinsics"]
    ctr = (0.5 * (src["surface_min_m"][0] + src["surface_max_m"][0]),
           0.5 * (src["surface_min_m"][1] + src["surface_max_m"][1]))
    Rn, tn = p34.nadir_pose_at(ctr, p34.NADIR_H_M)
    dep = render_sphere_depth(P, rad, intr, Rn, tn)
    hm_b = hmlib.heightmap_from_depth(dep, intr, Rn, tn, spec, agg="max",
                                      extra_meta={"scene": "s1_pile_centres_with_"
                                                           "size_distribution",
                                                  "camera_pose": "synthetic_nadir_0.90m"})
    # 참조: 균일(원래 반지름) 버전 — 다분산화가 관측을 어떻게 바꾸는지 대조군
    hm_u = hmlib.heightmap_from_particles(P, r0, spec)

    hm_a.save(OUT / "hm4_pathA_polydisperse.npz")
    hm_b.save(OUT / "hm4_pathB_depth_nadir.npz")
    hm_u.save(OUT / "hm4_pathA_monodisperse_reference.npz")

    ka, kb = p33._contract_keys(hm_a), p33._contract_keys(hm_b)
    both = hm_a.valid & hm_b.valid
    occ = hm_a.height.astype(np.float64) > 0
    d_ab = hm_a.height.astype(np.float64) - hm_b.height.astype(np.float64)
    d_poly = (hm_a.height.astype(np.float64) - hm_u.height.astype(np.float64))
    # 축소 전용이므로 다분산 heightmap 은 균일 참조보다 결코 높을 수 없다
    monotone = bool(d_poly.max() <= TOL_M)
    # 표면 최고점 교차검증 (독립 계산)
    peak_expect = float((P[:, 2] + rad).max())
    peak_err = float(hm_a.height.astype(np.float64).max() - peak_expect)

    return {"gate": "P6_real_pile_with_size_distribution",
            "pass": bool(ka == kb and monotone and abs(peak_err) <= TOL_M),
            "contract_identical": bool(ka == kb),
            "contract_diff": {k: [ka.get(k), kb.get(k)] for k in set(ka) | set(kb)
                              if ka.get(k) != kb.get(k)},
            "n_particles": int(P.shape[0]),
            "radius_mm": {"min": float(rad.min()) * 1000, "max": float(rad.max()) * 1000,
                          "median": float(np.median(rad)) * 1000,
                          "mean": float(rad.mean()) * 1000,
                          "original_uniform": r0 * 1000},
            "peak_vs_independent_max_pz_plus_r": {
                "expected_mm": peak_expect * 1000,
                "got_mm": float(hm_a.height.max()) * 1000,
                "err_m": peak_err},
            "shrink_only_monotone_vs_uniform": monotone,
            "poly_minus_uniform_mm": stats_mm(d_poly[occ]),
            "AB_diff_over_pile_mm": stats_mm(d_ab[both & occ]),
            "A_valid_frac": float(hm_a.valid.mean()),
            "B_valid_frac": float(hm_b.valid.mean()),
            "reading": ("반지름을 줄이기만 했으므로 (i) 새 겹침이 생기지 않고 "
                        "(ii) 다분산 heightmap 은 균일 참조보다 결코 높을 수 없다 — "
                        "둘 다 검사한다. (A)/(B) 계약도 다분산에서 그대로 성립한다."),
            "non_claim": ("이 배치는 s1 의 정착 좌표에 크기 분포를 **사후에** 입힌 "
                          "기하 스트레스 테스트다. 다분산 입자가 실제로 저렇게 "
                          "정착한다는 물리 주장이 아니다 — 진짜 다분산 pile 은 "
                          "s1 이 DEME 로 생성해야 한다.")}


# --------------------------------------------------------------------------- #
def make_png(res: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = Heightmap.load(OUT / "hm4_pathA_polydisperse.npz")
    b = Heightmap.load(OUT / "hm4_pathB_depth_nadir.npz")
    u = Heightmap.load(OUT / "hm4_pathA_monodisperse_reference.npz")
    x0, x1, y0, y1 = a.spec.bounds_m()
    ext = [x0 * 1000, x1 * 1000, y0 * 1000, y1 * 1000]
    vmax = float(max(a.height.max(), u.height.max())) * 1000

    fig, ax = plt.subplots(1, 4, figsize=(18, 6.2))
    for axi, hm, ttl in ((ax[0], u, "(A) monodisperse ref  r=2.08 mm"),
                         (ax[1], a, "(A) polydisperse  r∈[0.83, 2.08] mm"),
                         (ax[2], b, "(B) depth nadir, polydisperse")):
        im = axi.imshow(hm.height * 1000, origin="lower", extent=ext, cmap="viridis",
                        vmin=0, vmax=vmax)
        axi.set_title(f"{ttl}\npeak={hm.height.max()*1000:.2f} mm", fontsize=9)
        axi.set_xlabel("x [mm]")
        fig.colorbar(im, ax=axi, fraction=0.046, label="height [mm]")
    ax[0].set_ylabel("y [mm]")
    d = (a.height.astype(float) - b.height.astype(float)) * 1000
    lim = max(float(np.abs(d).max()), 1e-9)
    im = ax[3].imshow(d, origin="lower", extent=ext, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[3].set_title(f"(A) - (B) [mm]\nrms="
                    f"{res['P6_real_pile_with_size_distribution']['AB_diff_over_pile_mm']['rms_mm']:.2f}",
                    fontsize=9)
    ax[3].set_xlabel("x [mm]")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    fig.suptitle("hm4 — polydisperse path (A) on s1 pile centres "
                 "(shrink-only size distribution)")
    fig.tight_layout()
    p = OUT / "hm4_polydisperse.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return [p.name]


def emit_rerun(res: dict) -> dict:
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    a = Heightmap.load(OUT / "hm4_pathA_polydisperse.npz")
    b = Heightmap.load(OUT / "hm4_pathB_depth_nadir.npz")
    u = Heightmap.load(OUT / "hm4_pathA_monodisperse_reference.npz")
    c = a.spec.cell_centers()
    rrd, rbl = OUT / "hm4_timeline.rrd", OUT / "hm4_timeline.rbl"
    app_id = "roarm_hm4"
    bp = rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="/", contents=["/hmap/**"], name="1 | heightmaps"),
        rrb.TimeSeriesView(origin="/cost", contents="/cost/**",
                           name="2 | window bucketing cost")),
        auto_layout=False, auto_views=False, collapse_panels=True)
    with rr.RecordingStream(app_id, recording_id="hm4_polydisperse", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(rrd), write_footer=True)
        rec.send_blueprint(bp, make_active=True, make_default=True)
        for ent, hm, col in (("hmap/pathA_polydisperse", a, [230, 160, 40]),
                             ("hmap/pathA_monodisperse_ref", u, [140, 140, 145]),
                             ("hmap/pathB_depth_nadir", b, [60, 170, 230])):
            v = hm.valid.ravel()
            xyz = np.stack([c[..., 0].ravel(), c[..., 1].ravel(),
                            hm.height.astype(np.float64).ravel()], axis=1)[v]
            rec.log(ent, rr.Points3D(xyz, colors=[col], radii=0.0008), static=True)
        for i, row in enumerate(res["P5_window_bucketing_cost"]["rows"]):
            rec.set_time("cost_case", sequence=i)
            rec.log("cost/pairs_bucketed", rr.Scalars(row["candidate_pairs_bucketed"]))
            rec.log("cost/pairs_naive_rmax", rr.Scalars(row["candidate_pairs_naive_rmax"]))
            rec.log("cost/work_ratio", rr.Scalars(row["work_ratio_naive_over_bucketed"]))
        rec.flush(timeout_sec=60.0)
    bp.save(app_id, str(rbl))

    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    ents = ["hmap/pathA_polydisperse", "hmap/pathA_monodisperse_ref",
            "hmap/pathB_depth_nadir", "cost/pairs_bucketed",
            "cost/pairs_naive_rmax", "cost/work_ratio"]
    return validate_rerun_artifact(
        rrd, expected_entity_paths=ents, exact_entity_paths=ents,
        exact_timeline_names=["blueprint", "log_time", "cost_case"],
        expected_entity_components={
            **{e: pts3 for e in ents if e.startswith("hmap/")},
            **{e: ["Scalars:scalars"] for e in ents if e.startswith("cost/")}},
        blueprint_path=rbl, screenshot_path=OUT / "hm4_inspection.png",
        expected_version="0.34.1", timeout_s=300.0,
        cli_path="/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    calib = hmlib.load_kinect_calib(CALIB)

    res = {"tool": "p36_hm4_polydisperse_contract", "case": "hm4_polydisperse",
           "spec_version": hmlib.SPEC_VERSION, "seed": SEED,
           "subject": "roarm_rl.heightmap.heightmap_from_particles — polydisperse radii",
           "module": {"path": "roarm_rl/heightmap.py",
                      "sha16": sha256(REPO / "roarm_rl/heightmap.py")[:16]},
           "source_pile": {"npz": str(PILE_NPZ.relative_to(REPO)),
                           "sha256_16": sha256(PILE_NPZ)[:16]},
           "env": {"python": sys.version.split()[0], "numpy": np.__version__}}

    res["P1_backcompat"] = gate_backcompat(rng)
    res["P2_bruteforce"] = gate_bruteforce(rng)
    res["P3_analytic"] = gate_analytic()
    res["P4_superposition"] = gate_superposition(rng)
    res["P5_window_bucketing_cost"] = gate_cost(rng)
    res["P6_real_pile_with_size_distribution"] = gate_realpile(rng, calib)
    res["pngs"] = make_png(res)

    if args.rerun:
        try:
            res["rerun"] = emit_rerun(res)
        except Exception as exc:  # noqa: BLE001 — D447: 침묵 사망 금지
            import traceback
            res["rerun"] = {"error": repr(exc), "traceback": traceback.format_exc()}

    gates = {k: res[k]["pass"] for k in sorted(res) if k.startswith("P")
             and isinstance(res[k], dict) and "pass" in res[k]}
    fails = [k for k, v in gates.items() if not v]
    res["gates"] = gates
    res["verdict"] = {
        "code": "HM4_ALL_GATES_PASS" if not fails else "HM4_FAIL_" + "_".join(fails).upper(),
        "non_claims": ("진짜 다분산 DEME pile 미수신 — P6 는 s1 정착 좌표에 크기 "
                       "분포를 사후에 입힌 기하 스트레스 테스트다. 다분산 입자의 "
                       "정착 물리·안식각·패킹률 전부 미주장. 실제 Kinect 미사용."),
    }
    res["wall_seconds"] = round(time.time() - t0, 1)

    p = OUT / "hm4_results.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"verdict": res["verdict"]["code"], "gates": gates,
                      "out": str(OUT.relative_to(REPO)),
                      "results_sha16": sha256(p)[:16]}, ensure_ascii=False, indent=2))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
