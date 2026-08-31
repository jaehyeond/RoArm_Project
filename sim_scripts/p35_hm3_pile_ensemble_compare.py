#!/usr/bin/env python
"""p35 / hm3 — s1 이 낸 정착 pile **전부**를 공통 격자에서 경로 (A) 로 비교 (물리 0).

배경: s1 이 같은 seed(460)·같은 입자수로 여러 pile 을 냈는데 표면 최고점이 서로
다르다.  왜 다른지는 s1 의 판정 영역이고 여기서 주장하지 않는다.  이 probe 가
답하는 것은 **관측층 질문 하나**다:

    "그 차이가 heightmap 에서 몇 mm 인가?"

이 값이 곧 s3 예측 모델의 **바닥 잡음**이다.  모델이 예측해야 할 '퍼낸 뒤 남을
형상'의 오차 목표가 이 값보다 작으면 의미가 없다.  대조군: hm2 에서 잰 경로
(A)/(B) 규격 차이는 rms 0.90 mm 였다.

출력 = claudedocs/runtime_logs/heightmap_track/hm3_pile_ensemble/  (신규 폴더)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
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
p34 = _load("p34_probe", "sim_scripts/p34_hm2_deme_pile_heightmap.py")

GridSpec, Heightmap = hmlib.GridSpec, hmlib.Heightmap
load_pile, volume_m3 = p34.load_pile, p34.volume_m3
stats_mm = p34.p33.stats_mm

OUT = REPO / "claudedocs/runtime_logs/heightmap_track/hm3_pile_ensemble"
SRC_DIR = REPO / "claudedocs/runtime_logs/sim_deme"
CELL_M = 0.005
MARGIN_M = 0.010


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def common_grid(srcs: list[dict], cell_m: float) -> GridSpec:
    """모든 pile 의 표면 경계를 덮는 하나의 공통 격자 (비교 가능성의 전제)."""
    lo = np.min([s["surface_min_m"][:2] for s in srcs], axis=0)
    hi = np.max([s["surface_max_m"][:2] for s in srcs], axis=0)
    ctr = 0.5 * (lo + hi)
    n = np.ceil(((hi - lo) + 2 * MARGIN_M) / cell_m - 1e-9).astype(int)
    origin = ctr - n * cell_m / 2.0
    return GridSpec(origin_xy_m=(float(origin[0]), float(origin[1])),
                    cell_m=float(cell_m), shape=(int(n[1]), int(n[0])),
                    frame="deme_container_floor_center", z_datum_m=0.0)


def variant_of(name: str) -> str:
    """s1 변형 라벨 (rep 접미사 제거) — same-variant 짝짓기용."""
    stem = name.replace("pile_smoke_", "").replace(".npz", "")
    return stem.split("d4p16")[0].strip("_") or "base"


def label_of(name: str) -> str:
    """표시용 라벨 — rep 반복까지 구분."""
    v = variant_of(name)
    return f"{v}_rep2" if name.replace(".npz", "").endswith("_rep2") else v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="pile_*.npz")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(SRC_DIR.glob(args.glob))
    if len(paths) < 2:
        raise SystemExit(f"NEED_2_OR_MORE_PILES found={len(paths)} in {SRC_DIR}")

    srcs, names = [], []
    for p in paths:
        s = load_pile(p)          # 미정착·미봉쇄·다분산 pile 은 여기서 거부된다
        s["_name"] = p.name
        s["_sha256"] = sha256(p)
        srcs.append(s)
        names.append(p.name)

    # 비교 전제: 입자수·반지름이 같아야 부피 비교가 의미를 가진다
    ns = {s["n"] for s in srcs}
    rs = {(round(s["r_min"], 12), round(s["r_max"], 12)) for s in srcs}
    comparable = (len(ns) == 1 and len(rs) == 1)

    spec = common_grid(srcs, CELL_M)
    maps, per_pile = {}, []
    for s in srcs:
        hm = hmlib.heightmap_from_particles(
            s["P"], s["r"], spec,
            extra_meta={"scene": "s1_deme_settled_pile", "source_npz": s["_name"]})
        hm.save(OUT / f"hm3_{Path(s['_name']).stem}.npz")
        maps[s["_name"]] = hm
        h = hm.height.astype(np.float64)
        v = volume_m3(hm)
        per_pile.append({
            "npz": s["_name"], "sha256_16": s["_sha256"][:16],
            "variant": variant_of(s["_name"]),
            "label": label_of(s["_name"]),
            "n_particles": s["n"],
            "diameter_median_mm": s["r_med"] * 2000.0,
            "polydisperse": not s["r_uniform"],
            "seed": s["meta"]["config"]["seed"],
            "settled_sim_time_s": s["meta"]["settling_gate"]["settled_sim_time_s"],
            "s1_surface_max_mm": s["surface_max_z_m"] * 1000.0,
            "heightmap_peak_mm": float(h.max()) * 1000.0,
            "peak_err_vs_s1_mm": float(h.max() - s["surface_max_z_m"]) * 1000.0,
            "occupied_cells": int((h > 0).sum()),
            "volume_cm3": v * 1e6,
            "phi_eff_at_5mm": s["v_solid_m3"] / v if v > 0 else None,
            "centroid_xy_mm": [float(np.average(s["P"][:, 0])) * 1000.0,
                               float(np.average(s["P"][:, 1])) * 1000.0],
        })

    # 모든 pile 에서 peak 복원이 여전히 정확한가 (H1 을 전 pile 로 확장)
    peak_ok = all(abs(r["peak_err_vs_s1_mm"]) <= 1.0e-4 for r in per_pile)

    pairs = []
    for a, b in itertools.combinations(names, 2):
        ha = maps[a].height.astype(np.float64)
        hb = maps[b].height.astype(np.float64)
        occ = (ha > 0) | (hb > 0)
        pairs.append({
            "a": a, "b": b,
            "same_variant": variant_of(a) == variant_of(b),
            "diff_over_union_footprint": stats_mm((ha - hb)[occ]),
            "volume_diff_cm3": (volume_m3(maps[a]) - volume_m3(maps[b])) * 1e6,
        })

    same = [p for p in pairs if p["same_variant"]]
    diff = [p for p in pairs if not p["same_variant"]]

    def agg(rows, key="rms_mm"):
        if not rows:
            return None
        v = [r["diff_over_union_footprint"][key] for r in rows]
        return {"n_pairs": len(v), "min": float(min(v)), "mean": float(np.mean(v)),
                "max": float(max(v))}

    vols = [r["volume_cm3"] for r in per_pile]
    peaks = [r["heightmap_peak_mm"] for r in per_pile]

    res = {
        "tool": "p35_hm3_pile_ensemble_compare", "case": "hm3_pile_ensemble",
        "question": ("s1 이 낸 정착 pile 들 사이의 차이가 heightmap 에서 몇 mm 인가 "
                     "— s3 예측 모델의 바닥 잡음"),
        "spec_version": hmlib.SPEC_VERSION,
        "module": {"path": "roarm_rl/heightmap.py",
                   "sha16": sha256(REPO / "roarm_rl/heightmap.py")[:16]},
        "env": {"python": sys.version.split()[0], "numpy": np.__version__},
        "common_grid": spec.to_header(),
        "n_piles": len(srcs),
        "comparable_population": bool(comparable),
        "peak_recovery_all_piles_exact": bool(peak_ok),
        "per_pile": per_pile,
        "spread": {
            "peak_mm": {"min": min(peaks), "max": max(peaks),
                        "range": max(peaks) - min(peaks),
                        "std": float(np.std(peaks, ddof=1))},
            "volume_cm3": {"min": min(vols), "max": max(vols),
                           "range": max(vols) - min(vols),
                           "std": float(np.std(vols, ddof=1)),
                           "range_pct_of_mean": (max(vols) - min(vols))
                           / float(np.mean(vols)) * 100.0},
        },
        "pairwise": pairs,
        "pairwise_summary": {
            "same_variant_rms_mm": agg(same),
            "cross_variant_rms_mm": agg(diff),
            "same_variant_max_mm": agg(same, "max_abs_mm"),
            "cross_variant_max_mm": agg(diff, "max_abs_mm"),
        },
        "reference_scales_mm": {
            "hm2_pathA_vs_pathB_rms": 0.904,
            "hm2_pathA_vs_pathB_max": 3.929,
            "pellet_radius": 2.08,
            "pellet_diameter": 4.16,
            "cell_m_mm": CELL_M * 1000.0,
            "note": ("pile 간 차이를 이 값들과 비교하라. 관측 경로 차이(0.9 mm)나 "
                     "펠릿 반지름(2.08 mm)보다 크면, 관측 파이프라인이 아니라 "
                     "pile 생성이 s3 오차 예산을 지배한다는 뜻이다."),
        },
        "interpretation_boundary": (
            "이 probe 는 pile 들이 **왜** 다른지 주장하지 않는다 (DEME 결정성·정렬·"
            "동기화는 s1 의 판정 영역). 여기서 재는 것은 오직 그 차이의 관측층 크기다. "
            "또한 이 pile 들이 '같아야 하는지'도 주장하지 않는다 — 변형 라벨(sorted/"
            "sync/rep2)의 의미는 s1 이 정의한다."),
        "non_claims": ("s1 재료값 provisional 미측정. DEME 결정성 판정 없음. "
                       "실제 Kinect 미사용. RoArm base 프레임 배치 미확정."),
    }

    # ---- D324 진단 그림 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x0, x1, y0, y1 = spec.bounds_m()
    ext = [x0 * 1000, x1 * 1000, y0 * 1000, y1 * 1000]
    n = len(names)
    fig, ax = plt.subplots(2, n, figsize=(3.4 * n, 9.0), squeeze=False)
    vmax = max(float(maps[k].height.max()) for k in names) * 1000
    ref = names[0]
    for i, k in enumerate(names):
        im = ax[0][i].imshow(maps[k].height * 1000, origin="lower", extent=ext,
                             cmap="viridis", vmin=0, vmax=vmax)
        ax[0][i].set_title(f"{label_of(k)}\npeak={maps[k].height.max()*1000:.2f} mm",
                           fontsize=8)
        fig.colorbar(im, ax=ax[0][i], fraction=0.046)
        hk = maps[k].height.astype(float)
        hr = maps[ref].height.astype(float)
        d = (hk - hr) * 1000
        # JSON 과 동일한 마스크(합집합 발자국)로 통계를 내야 숫자가 일치한다
        u = (hk > 0) | (hr > 0)
        dm = d[u] if u.any() else d.ravel()
        lim = max(float(np.abs(d).max()), 1e-9)
        im2 = ax[1][i].imshow(np.where(u, d, np.nan), origin="lower", extent=ext,
                              cmap="coolwarm", vmin=-lim, vmax=lim)
        ax[1][i].set_title(f"minus {label_of(ref)} (union footprint)\nrms="
                           f"{np.sqrt((dm**2).mean()):.2f} max={np.abs(dm).max():.2f} mm",
                           fontsize=8)
        fig.colorbar(im2, ax=ax[1][i], fraction=0.046)
    fig.suptitle(f"hm3 — s1 pile ensemble on a common {spec.shape[0]}x{spec.shape[1]} "
                 f"grid @ {CELL_M*1000:.0f} mm  (seed 460, n=512, d=4.16 mm)")
    fig.tight_layout()
    png = OUT / "hm3_ensemble.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    res["pngs"] = [png.name]
    res["wall_seconds"] = round(time.time() - t0, 1)

    gates = {"G_peak_recovery_all_piles": peak_ok,
             "G_comparable_population": comparable}
    fails = [k for k, v in gates.items() if not v]
    res["gates"] = gates
    res["verdict"] = {"code": "HM3_OK" if not fails else "HM3_FAIL_" + "_".join(fails).upper()}

    p = OUT / "hm3_results.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({
        "verdict": res["verdict"]["code"], "n_piles": len(srcs), "gates": gates,
        "peak_range_mm": round(res["spread"]["peak_mm"]["range"], 3),
        "volume_range_pct": round(res["spread"]["volume_cm3"]["range_pct_of_mean"], 2),
        "same_variant_rms_mm": res["pairwise_summary"]["same_variant_rms_mm"],
        "cross_variant_rms_mm": res["pairwise_summary"]["cross_variant_rms_mm"],
        "out": str(OUT.relative_to(REPO)),
    }, ensure_ascii=False, indent=2))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
