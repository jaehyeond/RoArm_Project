#!/usr/bin/env python
"""p34 / hm2 — s1 DEME 정착 pile 을 heightmap 경로 (A) 에 실제로 투입 (물리 0, 로봇 0).

p33/hm1 은 합성 원뿔·경사면으로 계약을 검증했다.  여기서는 s1 이 실제로 뱉은
정착 결과(`DEME_PELLET_PILE_V1`)를 그대로 넣어, 가정이 아니라 실측 배치에서
계약이 성립하는지 본다.

  H0 source   — s1 npz 계약/게이트 확인.  정착 실패·침투·미봉쇄 pile 은 거부한다.
  H1 pathA    — 실 pile -> heightmap.  풋프린트-max 연산자가 진짜 표면 최고점을
                복원하는가(s1 이 독립 계산한 surface_max_m[2] 와 대조), 발자국이
                s1 이 보고한 경계 안에 들어오는가.
  H2 cell     — 셀 크기 스윕.  5 mm 선택 근거를 가정이 아니라 실 pile 로 재검.
  H3 volume   — **퍼낸 양의 편향.**  heightmap 적분 부피가 집계 선택에 따라
                얼마나 달라지는가.  s3 의 "퍼낸 양" 라벨이 여기에 직접 걸린다.
  H4 slope    — 실제 pile 사면 기울기 분포 -> D453 증폭 법칙을 가상의 79.5°가
                아니라 **이 pile 의 실제 각도**에 적용한 실물 오차 예산.
  H5 parity   — 실 pile 에서 (A) vs (B).  nadir 합성 깊이 + 실제 Kinect 포즈 그림자.

출력 = claudedocs/runtime_logs/heightmap_track/hm2_s1pile/  (신규 폴더)

사용:
  python3 sim_scripts/p34_hm2_deme_pile_heightmap.py
  /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p34_hm2_deme_pile_heightmap.py --rerun
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


# roarm_rl/__init__.py 는 gymnasium/Isaac 을 끌어온다 -> 파일 경로로 직접 로드
hmlib = _load("roarm_heightmap", "roarm_rl/heightmap.py")
# 합성 깊이 렌더러는 p33 의 테스트 픽스처다 — 복제하지 않고 재사용
p33 = _load("p33_probe", "sim_scripts/p33_hm1_heightmap_contract_probe.py")

GridSpec, Heightmap = hmlib.GridSpec, hmlib.Heightmap
render_sphere_depth, stats_mm = p33.render_sphere_depth, p33.stats_mm

OUT = REPO / "claudedocs/runtime_logs/heightmap_track/hm2_s1pile"
DEFAULT_NPZ = REPO / "claudedocs/runtime_logs/sim_deme/pile_smoke_d4p16_n512_seed460.npz"
CALIB = REPO / "sim_scripts/kinect_calib.yaml"

CELL_M = 0.005                 # hm1 에서 고정한 계약 셀 크기
CELL_SWEEP_M = (0.0005, 0.001, 0.002, 0.003, 0.00416, 0.005, 0.006, 0.008,
                0.010, 0.015)
# 참조 포락면 = 가장 미세한 셀의 raw 적분.  풋프린트-max 는 cell -> 0 에서 구 합집합의
# 진짜 상단 포락면으로 수렴하므로, 5 mm 계약의 편향은 이 값 대비로 읽는다.
CELL_REF_M = 0.0005
MARGIN_M = 0.010               # pile 표면 경계 바깥 여유
NADIR_H_M = 0.900              # Kinect 표준 standoff
SEED = 34001
# s1 이 float64 로 독립 계산한 표면 최고점과의 허용차.  height 는 계약상 float32 이고
# h ~ 0.012 m 에서 양자화 스텝이 약 1e-9 m 이므로 1e-7 m 는 순수 구현오차만 잡는다.
TOL_PEAK_M = 1.0e-7


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# H0 — source contract
# --------------------------------------------------------------------------- #
def load_pile(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=False)
    m = json.loads(str(d["metadata_json"]))
    if m.get("artifact") != "DEME_PELLET_PILE_V1" or m.get("schema_version") != 1:
        raise RuntimeError(f"UNKNOWN_SOURCE_CONTRACT {m.get('artifact')} v{m.get('schema_version')}")

    P = np.asarray(d["positions_m"], dtype=np.float64)
    R = np.asarray(d["radii_m"], dtype=np.float64)
    if P.shape[0] != R.shape[0] or P.ndim != 2 or P.shape[1] != 3:
        raise RuntimeError(f"SHAPE_MISMATCH positions{P.shape} radii{R.shape}")
    cf, sg, pg = m["coordinate_frame"], m["settling_gate"], m["post_settle_gates"]
    if not sg.get("settled"):
        raise RuntimeError("SOURCE_NOT_SETTLED — 정착 실패 pile 은 관측 대상이 아니다")
    if not (pg.get("containment_pass") and pg.get("penetration_pass")):
        raise RuntimeError(f"SOURCE_GATE_FAIL containment={pg.get('containment_pass')} "
                           f"penetration={pg.get('penetration_pass')}")
    if abs(float(cf.get("floor_z_m", 0.0))) > 1e-12:
        raise RuntimeError(f"UNEXPECTED_FLOOR_Z {cf.get('floor_z_m')} — z_datum 매핑 재확인 필요")

    # 경로 (A) 는 다분산을 지원한다 — radii 배열을 그대로 넘긴다.
    # 파생 스칼라(대표 지름 등)는 분포에서 뽑고, 어느 통계인지 명시해 기록한다.
    return {"P": P, "r": R, "meta": m,
            "r_uniform": bool(R.min() == R.max()),
            "r_min": float(R.min()), "r_max": float(R.max()),
            "r_mean": float(R.mean()), "r_med": float(np.median(R)),
            "frame": ("deme_container_floor_center  "
                      "(origin=" + str(cf["origin"]) + ", z up, floor z=0)"),
            # s1 이 float64 로 독립 계산한 표면 최고점 = max_i(pz_i + r_i).
            # 내 산술을 끼우지 않아야 H1 의 교차검증이 독립적이다.
            "surface_max_z_m": float(pg["surface_max_m"][2]),
            "surface_min_m": [float(v) for v in pg["surface_min_m"]],
            "surface_max_m": [float(v) for v in pg["surface_max_m"]],
            "n": int(P.shape[0]),
            "v_solid_m3": float((4.0 / 3.0) * np.pi * (R ** 3).sum())}


def grid_for_pile(src: dict, cell_m: float, margin_m: float = MARGIN_M) -> GridSpec:
    """pile 표면 경계 + margin 을 덮는 최소 정수 셀 격자, pile 중심에 정렬."""
    lo, hi = np.array(src["surface_min_m"][:2]), np.array(src["surface_max_m"][:2])
    ctr = 0.5 * (lo + hi)
    need = (hi - lo) + 2.0 * margin_m
    n = np.ceil(need / cell_m - 1e-9).astype(int)
    origin = ctr - n * cell_m / 2.0
    return GridSpec(origin_xy_m=(float(origin[0]), float(origin[1])),
                    cell_m=float(cell_m), shape=(int(n[1]), int(n[0])),
                    frame="deme_container_floor_center", z_datum_m=0.0)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def volume_m3(hm: Heightmap) -> float:
    h = hm.height.astype(np.float64)
    return float(np.where(hm.valid, h, 0.0).sum() * hm.spec.cell_m ** 2)


def debiased_volume_m3(hm: Heightmap) -> tuple[float, float]:
    """풋프린트-max 편향을 셀별 실측 기울기로 되빼고 적분.

    반환 = (보정 부피, 제거된 평균 편향 [m]).  D453 증폭 법칙의 (a) 항을
    downstream 부피에 그대로 적용한 것 — 편향은 항상 부피를 과대추정한다.
    """
    slope = hmlib.cell_slope_deg(hm)
    bias = hmlib.slope_bias_m(hm.spec.cell_m, slope, agg="max")
    h = np.maximum(hm.height.astype(np.float64) - bias, 0.0)
    keep = hm.valid & (hm.height.astype(np.float64) > 0)
    return (float(np.where(hm.valid, h, 0.0).sum() * hm.spec.cell_m ** 2),
            float(bias[keep].mean()) if keep.any() else 0.0)


def nadir_pose_at(ctr_xy, height_m: float):
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return R, np.array([float(ctr_xy[0]), float(ctr_xy[1]), float(height_m)])


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def gate_pathA(src: dict, spec: GridSpec) -> tuple[Heightmap, dict]:
    hm = hmlib.heightmap_from_particles(
        src["P"], src["r"], spec,
        extra_meta={"scene": "s1_deme_settled_pile",
                    "source_npz": DEFAULT_NPZ.name,
                    "source_artifact": "DEME_PELLET_PILE_V1"})
    h = hm.height.astype(np.float64)

    peak_err = float(h.max() - src["surface_max_z_m"])
    occ = h > 0.0
    rr, cc = np.nonzero(occ)
    c = spec.cell_centers()
    half = spec.cell_m / 2.0
    fp = {"x_min_m": float(c[rr, cc, 0].min() - half), "x_max_m": float(c[rr, cc, 0].max() + half),
          "y_min_m": float(c[rr, cc, 1].min() - half), "y_max_m": float(c[rr, cc, 1].max() + half)}
    sx0, sy0 = src["surface_min_m"][0], src["surface_min_m"][1]
    sx1, sy1 = src["surface_max_m"][0], src["surface_max_m"][1]
    # 발자국은 s1 표면 경계를 덮되, 한 셀 이상 삐져나가면 안 된다
    covers = (fp["x_min_m"] <= sx0 and fp["x_max_m"] >= sx1
              and fp["y_min_m"] <= sy0 and fp["y_max_m"] >= sy1)
    tight = (max(sx0 - fp["x_min_m"], fp["x_max_m"] - sx1,
                 sy0 - fp["y_min_m"], fp["y_max_m"] - sy1) <= spec.cell_m)

    v = volume_m3(hm)
    info = {
        "gate": "H1_pathA_on_real_pile",
        "pass": bool(abs(peak_err) <= TOL_PEAK_M and covers and tight and hm.valid.all()),
        "grid": spec.to_header(),
        "n_particles": src["n"],
        "pellet_radius_m": {"uniform": src["r_uniform"], "min": src["r_min"],
                            "median": src["r_med"], "max": src["r_max"],
                            "mean": src["r_mean"]},
        "pellet_diameter_median_mm": src["r_med"] * 2000.0,
        "pellet_diameter_max_mm": src["r_max"] * 2000.0,
        "cell_per_pellet_diameter_median": spec.cell_m / (2.0 * src["r_med"]),
        "cell_per_pellet_diameter_max": spec.cell_m / (2.0 * src["r_max"]),
        "peak": {"heightmap_max_mm": float(h.max()) * 1000.0,
                 "s1_surface_max_z_mm": src["surface_max_z_m"] * 1000.0,
                 "err_mm": peak_err * 1000.0, "tol_mm": TOL_PEAK_M * 1000.0,
                 "reading": ("풋프린트-max 연산자는 셀 중심이 아니라 셀 안 최고점을 재므로 "
                             "pile 전체 최고점을 정확히 복원해야 한다 — s1 이 독립 "
                             "계산한 값과 대조")},
        "footprint": {**{k: v_ * 1000.0 for k, v_ in fp.items()},
                      "covers_s1_surface_bounds": bool(covers),
                      "within_one_cell": bool(tight)},
        "occupied_cells": int(occ.sum()), "total_cells": spec.n_cells,
        "all_valid": bool(hm.valid.all()),
        "counts": {"max": int(hm.counts.max()), "mean_over_occupied": float(
            hm.counts[occ].mean())},
        "volume_raw_m3": v,
        "packing_fraction_raw": src["v_solid_m3"] / v if v > 0 else None,
    }
    return hm, info


def gate_cell_sweep(src: dict) -> dict:
    rows = []
    for cm in CELL_SWEEP_M:
        sp = grid_for_pile(src, cm)
        hm = hmlib.heightmap_from_particles(src["P"], src["r"], sp)
        v = volume_m3(hm)
        vd, bias = debiased_volume_m3(hm)
        occ = hm.height.astype(np.float64) > 0
        rows.append({
            "cell_mm": cm * 1000.0,
            "cell_per_pellet_diameter": cm / (2.0 * src["r_med"]),
            "shape": list(sp.shape), "n_cells": sp.n_cells,
            "cells_across_pile_width": (src["surface_max_m"][0]
                                        - src["surface_min_m"][0]) / cm,
            "peak_mm": float(hm.height.max()) * 1000.0,
            "occupied_cells": int(occ.sum()),
            "volume_raw_cm3": v * 1e6,
            "volume_debiased_cm3": vd * 1e6,
            "volume_bias_pct": (v - vd) / vd * 100.0 if vd > 0 else None,
            "mean_removed_bias_mm": bias * 1000.0,
            "packing_fraction_raw": src["v_solid_m3"] / v if v > 0 else None,
            "packing_fraction_debiased": src["v_solid_m3"] / vd if vd > 0 else None,
        })
    ref = [r for r in rows if abs(r["cell_mm"] - CELL_REF_M * 1000) < 1e-9][0]
    v_ref = ref["volume_raw_cm3"]
    for r in rows:
        r["volume_vs_ref_pct"] = (r["volume_raw_cm3"] - v_ref) / v_ref * 100.0
        # 기전 가설: 셀이 펠릿 지름 이상이면 셀마다 crown 을 잡으므로 heightmap 이
        # 평균 표면보다 대략 펠릿 반지름만큼 위에 뜬다 -> 초과부피 ~ 점유면적 x r
        excess = r["volume_raw_cm3"] - v_ref
        # r-오프셋 기전: heightmap 이 평균 표면보다 약 반지름만큼 뜬다.
        # 다분산이면 평균 반지름이 그 대표값이다.
        pred = (r["occupied_cells"] * (r["cell_mm"] * 1e-1) ** 2 * (src["r_mean"] * 1e2))
        r["excess_cm3"] = excess
        r["excess_pred_area_times_radius_cm3"] = pred
        r["excess_pred_ratio"] = pred / excess if excess > 1e-12 else None
    return {"gate": "H2_cell_sweep", "pass": True, "rows": rows,
            "v_solid_cm3": src["v_solid_m3"] * 1e6,
            "envelope_reference": {
                "cell_mm": CELL_REF_M * 1000, "volume_cm3": v_ref,
                "packing_fraction": ref["packing_fraction_raw"],
                "meaning": ("풋프린트-max 는 cell->0 에서 구 합집합의 진짜 상단 "
                            "포락면으로 수렴한다. 이 값이 다른 셀 크기의 기준선이다.")},
            "excess_mechanism": (
                "셀 >= 펠릿 지름이면 셀마다 pellet crown 을 잡으므로 heightmap 이 평균 "
                "표면보다 약 펠릿 반지름(2.08 mm)만큼 떠오른다. 초과부피 예측 "
                "= 점유면적 x r 이며, excess_pred_ratio 가 1 에 가까울수록 그 기전이 "
                "지배적이라는 뜻이다. 계약 셀(5 mm) 근처에서 가장 잘 맞고, 셀이 "
                "펠릿보다 작아지면(모든 셀이 crown 을 잡지는 않음) 과대예측, 훨씬 "
                "커지면(pile 사면 tan(theta) 항이 커짐) 과소예측한다."),
            "note": ("① peak 는 전 셀 크기에서 동일하다 — max 연산자가 셀 안 최고점을 "
                     "재기 때문에 구조적으로 그렇게 되며, **셀 크기 선택의 근거가 아니다**. "
                     "② 진짜 판별자는 부피 편향과 pile 폭을 가로지르는 셀 수(형상 해상도). "
                     "③ debiased 열은 pile 사면 (c/2)tan(theta) 항만 제거하고 펠릿 crown "
                     "r-오프셋은 못 없앤다 — 이 얕은 pile 에서는 후자가 지배적이라 "
                     "63% 중 20%p 만 줄어든다. 미세 셀에서는 측정 기울기가 pile 사면이 "
                     "아니라 펠릿 곡률이라 과보정된다.")}


def gate_volume(src: dict, hm_a: Heightmap, hm_b_max: Heightmap,
                hm_b_med: Heightmap, v_ref_m3: float) -> dict:
    v_a = volume_m3(hm_a)
    v_ad, bias_a = debiased_volume_m3(hm_a)
    v_bmax, v_bmed = volume_m3(hm_b_max), volume_m3(hm_b_med)
    vs = src["v_solid_m3"]

    # median 집계가 성긴 펠릿 표면에서 포락면 추정자가 되지 못하는 직접 증거:
    # 셀 안 ~11 개 깊이 픽셀 중 절반 이상이 펠릿 사이 바닥을 보면 median 이 0 이 된다.
    ha = hm_a.height.astype(np.float64)
    hm_ = hm_b_med.height.astype(np.float64)
    occ = ha > 0.0
    med_floor = int(((hm_ <= 1e-9) & occ).sum())

    def pct(v):
        return (v - v_ref_m3) / v_ref_m3 * 100.0 if v_ref_m3 > 0 else None

    return {
        "gate": "H3_volume_bias", "pass": True,
        "envelope_reference_cm3": v_ref_m3 * 1e6,
        "envelope_reference_note": (f"path A at cell={CELL_REF_M*1000:g} mm — "
                                    "풋프린트-max 의 cell->0 극한 = 구 합집합의 상단 포락면"),
        "v_solid_cm3": vs * 1e6,
        "estimators_cm3": {
            "pathA_footprint_max_5mm": v_a * 1e6,
            "pathA_slope_debiased_5mm": v_ad * 1e6,
            "pathB_nadir_max_5mm": v_bmax * 1e6,
            "pathB_nadir_median_5mm": v_bmed * 1e6},
        "vs_envelope_reference_pct": {
            "pathA_footprint_max_5mm": pct(v_a),
            "pathA_slope_debiased_5mm": pct(v_ad),
            "pathB_nadir_max_5mm": pct(v_bmax),
            "pathB_nadir_median_5mm": pct(v_bmed)},
        "pathA_mean_removed_bias_mm": bias_a * 1000.0,
        "median_is_not_an_envelope_estimator": {
            "pile_cells_A_gt_0": int(occ.sum()),
            "of_those_median_reads_exactly_zero": med_floor,
            "frac": med_floor / max(int(occ.sum()), 1),
            "mean_depth_samples_per_cell": float(hm_b_med.counts[occ].mean()),
            "reading": ("성긴 펠릿 표면에서는 셀 안 깊이 픽셀의 절반 이상이 펠릿 사이 "
                        "**바닥**을 본다. 그러면 median 이 0 이 된다 — hm1 의 연속 평면 "
                        "테스트에서 median 이 무편향이었던 것은 표면이 연속이었기 때문이고, "
                        "이산 입자 표면에는 그 결론이 전이되지 않는다. "
                        "**부피 라벨에 median 집계를 쓰면 안 된다.**")},
        "packing_fraction": {
            "envelope_reference": vs / v_ref_m3 if v_ref_m3 > 0 else None,
            "pathA_footprint_max_5mm": vs / v_a if v_a > 0 else None,
            "bulk_random_packing_range": [0.55, 0.64],
            "note": ("phi 는 판정 게이트가 아니라 타당성 지표다. 이 pile 은 최고 12.4 mm "
                     "= 펠릿 지름 3개 높이의 얕은 ridge 라, 상단 구들의 crown 이 포락면에는 "
                     "전부 들어가고 부피는 절반만 기여한다. 벌크 random packing 값보다 "
                     "낮게 나오는 것이 정상이며, 깊은 더미에서는 올라갈 것이다.")},
        "reading": ("s3 의 '퍼낸 양' 라벨은 heightmap 부피 차분이다. 여기 숫자가 그 라벨의 "
                    "연산자 의존성이다. 실용 결론: **같은 연산자를 퍼내기 전후에 일관되게** "
                    "쓰고, 고체부피 환산 계수 phi 는 그 연산자에 맞춰 보정할 것. "
                    "연산자를 바꾸면 phi 도 다시 보정해야 한다."),
    }


def gate_slope(src: dict, hm: Heightmap, calib: dict) -> dict:
    slope = hmlib.cell_slope_deg(hm)
    occ = (hm.height.astype(np.float64) > 0) & hm.valid
    # 사면 = pile 안쪽이면서 실제로 기울어진 셀
    s = slope[occ]
    rmse_m = calib["rmse_mm"] / 1000.0
    qs = [50, 75, 90, 95, 99, 100]
    pct = {f"p{q}": float(np.percentile(s, q)) for q in qs}
    bands = {}
    for name, (lo, hi) in {"0-10deg": (0, 10), "10-20deg": (10, 20),
                           "20-30deg": (20, 30), "30-45deg": (30, 45),
                           "45-90deg": (45, 90)}.items():
        sel = (s >= lo) & (s < hi)
        bands[name] = {"cell_frac": float(sel.mean()),
                       "calib_rmse_amp_mm": float(hmlib.lateral_to_vertical_m(
                           rmse_m, 0.5 * (lo + min(hi, 89.0)))) * 1000.0}
    return {
        "gate": "H4_real_slope_error_budget", "pass": True,
        "target_repose_angle_deg": src["meta"]["material_assumptions"][
            "target_repose_angle_deg"],
        "target_role": src["meta"]["material_assumptions"]["target_repose_angle_role"],
        "measured_slope_deg": pct,
        "mean_slope_deg": float(s.mean()),
        "bands": bands,
        "error_budget_at_measured_slopes_mm": {
            f"p{q}": {
                "slope_deg": pct[f"p{q}"],
                "cell_footprint_bias": float(hmlib.slope_bias_m(
                    hm.spec.cell_m, pct[f"p{q}"])) * 1000.0,
                "calib_rmse_10p13mm": float(hmlib.lateral_to_vertical_m(
                    rmse_m, pct[f"p{q}"])) * 1000.0,
                "slope_aware_tol": float(hmlib.slope_aware_tol_m(
                    hm.spec.cell_m, rmse_m, pct[f"p{q}"])) * 1000.0}
            for q in qs},
        "reading": ("hm1 은 D453 을 따라 79.5° 같은 극단각까지 표를 만들었지만, 이 "
                    "실제 pile 의 사면은 훨씬 완만하다. 실물 오차 예산은 이 측정된 "
                    "각도에서 읽어야 한다."),
        "caveat": ("셀 기울기는 heightmap 자체의 중앙차분이라 셀 크기에 의존하고, "
                   "펠릿 crown 요철이 사면각을 부풀린다. 안식각 물리 주장이 아니라 "
                   "관측면 기울기다."),
    }


def gate_parity(src: dict, spec: GridSpec, hm_a: Heightmap, calib: dict) -> dict:
    intr = calib["intrinsics"]
    ctr = (0.5 * (src["surface_min_m"][0] + src["surface_max_m"][0]),
           0.5 * (src["surface_min_m"][1] + src["surface_max_m"][1]))
    Rn, tn = nadir_pose_at(ctr, NADIR_H_M)
    dep_n = render_sphere_depth(src["P"], src["r"], intr, Rn, tn)
    hm_b = hmlib.heightmap_from_depth(dep_n, intr, Rn, tn, spec, agg="max",
                                      extra_meta={"scene": "s1_deme_settled_pile",
                                                  "camera_pose": "synthetic_nadir_0.90m"})
    hm_bm = hmlib.heightmap_from_depth(dep_n, intr, Rn, tn, spec, agg="median",
                                       extra_meta={"scene": "s1_deme_settled_pile",
                                                   "camera_pose": "synthetic_nadir_0.90m"})
    # 실제 캘리브 Kinect 포즈: pile 을 그 포즈 아래에 두려면 pile 을 base 프레임의
    # source 영역으로 옮겨야 한다.  배치가 아직 확정 전이므로, 카메라를 pile 프레임
    # 원점 위로 rigid 이동시켜 "같은 사각 시점"만 재현한다 (회전 그대로).
    Rk = calib["R"]
    tk = calib["t"] - np.array([0.200, 0.000, 0.000]) + np.array([ctr[0], ctr[1], 0.0])
    dep_k = render_sphere_depth(src["P"], src["r"], intr, Rk, tk)
    hm_k = hmlib.heightmap_from_depth(dep_k, intr, Rk, tk, spec, agg="max",
                                      extra_meta={"scene": "s1_deme_settled_pile",
                                                  "camera_pose": "kinect_calib_rotation_"
                                                                 "recentred_on_pile"})

    ka, kb = p33._contract_keys(hm_a), p33._contract_keys(hm_b)
    both = hm_a.valid & hm_b.valid
    diff = hm_a.height.astype(np.float64) - hm_b.height.astype(np.float64)
    slope = hmlib.cell_slope_deg(hm_a)
    occ = hm_a.height.astype(np.float64) > 0

    for h_, name in ((hm_a, "hm2_pathA_particles"), (hm_b, "hm2_pathB_depth_nadir"),
                     (hm_bm, "hm2_pathB_depth_nadir_median"),
                     (hm_k, "hm2_pathB_depth_kinectpose")):
        h_.save(OUT / f"{name}.npz")

    return {"gate": "H5_parity_on_real_pile", "pass": bool(ka == kb),
            "contract_identical": bool(ka == kb),
            "contract_diff": {k: [ka.get(k), kb.get(k)] for k in set(ka) | set(kb)
                              if ka.get(k) != kb.get(k)},
            "A_valid_frac": float(hm_a.valid.mean()),
            "B_nadir_valid_frac": float(hm_b.valid.mean()),
            "B_kinectpose_valid_frac": float(hm_k.valid.mean()),
            "B_kinectpose_invalid_cells": int((~hm_k.valid).sum()),
            "B_kinectpose_invalid_cells_over_pile": int(
                (~hm_k.valid & occ).sum()),
            "kinect_pose_elevation_deg": float(np.degrees(np.arctan2(
                -Rk[2, 2], np.hypot(Rk[0, 2], Rk[1, 2])))),
            "AB_diff_all": stats_mm(diff[both]),
            "AB_diff_over_pile": stats_mm(diff[both & occ]),
            "AB_diff_by_slope_band": {
                b: (stats_mm(diff[both & occ & (slope >= lo) & (slope < hi)])
                    if (both & occ & (slope >= lo) & (slope < hi)).any() else None)
                for b, (lo, hi) in {"0-10deg": (0, 10), "10-20deg": (10, 20),
                                    "20-30deg": (20, 30), "30-45deg": (30, 45),
                                    "45-90deg": (45, 90)}.items()},
            "hm_b_median": "hm2_pathB_depth_nadir_median.npz",
            "note": ("Kinect 포즈 렌더는 kinect_calib.yaml 의 회전을 그대로 쓰고 "
                     "카메라를 pile 위로 rigid 평행이동한 것이다. pile 의 base 프레임 "
                     "배치가 확정되면 그 배치로 다시 재야 한다 — 현재 값은 '같은 "
                     "앙각에서의 그림자 비용' 추정치다.")}


# --------------------------------------------------------------------------- #
def make_pngs(src: dict, spec: GridSpec, res: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hm_a = Heightmap.load(OUT / "hm2_pathA_particles.npz")
    hm_b = Heightmap.load(OUT / "hm2_pathB_depth_nadir.npz")
    hm_k = Heightmap.load(OUT / "hm2_pathB_depth_kinectpose.npz")
    x0, x1, y0, y1 = spec.bounds_m()
    ext = [x0 * 1000, x1 * 1000, y0 * 1000, y1 * 1000]

    fig, ax = plt.subplots(1, 4, figsize=(18, 6.2))
    for a, hm, ttl in ((ax[0], hm_a, "(A) particles, footprint max"),
                       (ax[1], hm_b, "(B) depth, nadir 0.90 m"),
                       (ax[2], hm_k, "(B) depth, kinect_calib elevation")):
        im = a.imshow(hm.height * 1000, origin="lower", extent=ext, cmap="viridis")
        a.set_title(f"{ttl}\nvalid={hm.valid.mean()*100:.1f}%  "
                    f"peak={hm.height.max()*1000:.2f} mm", fontsize=9)
        a.set_xlabel("x [mm]")
        fig.colorbar(im, ax=a, label="height [mm]")
    ax[0].set_ylabel("y [mm]")
    d = (hm_a.height.astype(float) - hm_b.height.astype(float)) * 1000
    lim = max(float(np.abs(d).max()), 1e-9)
    im = ax[3].imshow(d, origin="lower", extent=ext, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[3].set_title("(A) - (B nadir) [mm]", fontsize=9)
    ax[3].set_xlabel("x [mm]")
    fig.colorbar(im, ax=ax[3])
    fig.suptitle(f"hm2 — s1 DEME settled pile (n={src['n']}, "
                 f"d_med={src['r_med']*2000:.2f} mm), "
                 f"cell={spec.cell_m*1000:.0f} mm, {spec.shape[0]}x{spec.shape[1]}")
    fig.tight_layout()
    p1 = OUT / "hm2_pile_maps.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close(fig)

    sw = res["H2_cell_sweep"]["rows"]
    cm = [r["cell_mm"] for r in sw]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ref = res["H2_cell_sweep"]["envelope_reference"]
    ax[0].plot(cm, [r["volume_raw_cm3"] for r in sw], "o-", label="footprint max")
    ax[0].plot(cm, [r["volume_debiased_cm3"] for r in sw], "s-",
               label="slope-debiased (unreliable at small cell)")
    ax[0].axhline(ref["volume_cm3"], color="k", ls="-", lw=1.2,
                  label=f"envelope ref (cell={ref['cell_mm']:g} mm)")
    ax[0].axhline(res["H2_cell_sweep"]["v_solid_cm3"], color="k", ls=":", lw=1,
                  label="solid volume (512 spheres)")
    ax[0].axvline(CELL_M * 1000, color="r", ls="--", lw=1, label="contract 5 mm")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("cell [mm]  (log)")
    ax[0].set_ylabel("integrated volume [cm3]")
    ax[0].set_title("volume vs cell size — the 'scooped amount' label")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax[1].plot(cm, [r["volume_vs_ref_pct"] for r in sw], "o-", color="tab:red")
    ax[1].axvline(CELL_M * 1000, color="r", ls="--", lw=1)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("cell [mm]  (log)")
    ax[1].set_ylabel("volume vs envelope reference [%]")
    ax[1].set_title("footprint-max volume overestimate")
    ax[1].grid(alpha=0.3)

    ax[2].plot(cm, [r["peak_mm"] for r in sw], "o-",
               label="heightmap peak (invariant by construction)")
    ax[2].axhline(src["surface_max_z_m"] * 1000, color="k", ls=":", lw=1,
                  label="s1 true surface peak")
    ax[2].plot(cm, [r["cells_across_pile_width"] for r in sw], "s-",
               color="tab:green", label="cells across pile width")
    ax[2].axvline(CELL_M * 1000, color="r", ls="--", lw=1)
    ax[2].axvline(src["r_med"] * 2000, color="tab:purple", ls="-.", lw=1,
                  label=f"pellet diameter (median) {src['r_med']*2000:.2f} mm")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("cell [mm]  (log)")
    ax[2].set_title("peak is NOT a discriminator; shape resolution is")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    p2 = OUT / "hm2_cell_sweep.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return [p1.name, p2.name]


def emit_rerun(src: dict, spec: GridSpec, res: dict) -> dict:
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    hm_a = Heightmap.load(OUT / "hm2_pathA_particles.npz")
    hm_b = Heightmap.load(OUT / "hm2_pathB_depth_nadir.npz")
    hm_k = Heightmap.load(OUT / "hm2_pathB_depth_kinectpose.npz")
    c = spec.cell_centers()
    rrd, rbl = OUT / "hm2_timeline.rrd", OUT / "hm2_timeline.rbl"
    app_id = "roarm_hm2"
    bp = rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="/", contents=["/pile/**", "/hmap/**"],
                          name="1 | pile + heightmaps"),
        rrb.TimeSeriesView(origin="/sweep", contents="/sweep/**",
                           name="2 | cell-size sweep")),
        auto_layout=False, auto_views=False, collapse_panels=True)

    with rr.RecordingStream(app_id, recording_id="hm2_s1pile", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(rrd), write_footer=True)
        rec.send_blueprint(bp, make_active=True, make_default=True)
        rec.log("pile/pellets", rr.Points3D(src["P"], colors=[[200, 200, 205]],
                                            radii=src["r"]), static=True)
        for ent, hm, col in (("hmap/pathA_particles", hm_a, [230, 160, 40]),
                             ("hmap/pathB_nadir", hm_b, [60, 170, 230]),
                             ("hmap/pathB_kinectpose", hm_k, [180, 90, 200])):
            v = hm.valid.ravel()
            xyz = np.stack([c[..., 0].ravel(), c[..., 1].ravel(),
                            hm.height.astype(np.float64).ravel()], axis=1)[v]
            rec.log(ent, rr.Points3D(xyz, colors=[col], radii=0.0008), static=True)
        for r in res["H2_cell_sweep"]["rows"]:
            rec.set_time("cell_mm", duration=float(r["cell_mm"]))
            rec.log("sweep/volume_raw_cm3", rr.Scalars(r["volume_raw_cm3"]))
            rec.log("sweep/volume_debiased_cm3", rr.Scalars(r["volume_debiased_cm3"]))
            rec.log("sweep/volume_bias_pct", rr.Scalars(r["volume_bias_pct"]))
            rec.log("sweep/peak_mm", rr.Scalars(r["peak_mm"]))
        rec.flush(timeout_sec=60.0)
    bp.save(app_id, str(rbl))

    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    ents = ["pile/pellets", "hmap/pathA_particles", "hmap/pathB_nadir",
            "hmap/pathB_kinectpose", "sweep/volume_raw_cm3",
            "sweep/volume_debiased_cm3", "sweep/volume_bias_pct", "sweep/peak_mm"]
    return validate_rerun_artifact(
        rrd, expected_entity_paths=ents, exact_entity_paths=ents,
        exact_timeline_names=["blueprint", "log_time", "cell_mm"],
        expected_entity_components={
            **{e: pts3 for e in ents if not e.startswith("sweep/")},
            **{e: ["Scalars:scalars"] for e in ents if e.startswith("sweep/")}},
        blueprint_path=rbl, screenshot_path=OUT / "hm2_inspection.png",
        expected_version="0.34.1", timeout_s=300.0,
        cli_path="/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(DEFAULT_NPZ),
                    help="s1 DEME_PELLET_PILE_V1 npz")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    npz = Path(args.npz)
    calib = hmlib.load_kinect_calib(CALIB)
    src = load_pile(npz)
    spec = grid_for_pile(src, CELL_M)

    res = {"tool": "p34_hm2_deme_pile_heightmap", "case": "hm2_s1pile",
           "spec_version": hmlib.SPEC_VERSION,
           "module": {"path": "roarm_rl/heightmap.py",
                      "sha16": sha256(REPO / "roarm_rl/heightmap.py")[:16]},
           "source": {"npz": str(npz.relative_to(REPO)), "sha256": sha256(npz),
                      "artifact": src["meta"]["artifact"],
                      "schema_version": src["meta"]["schema_version"],
                      "DEME": src["meta"]["software"]["DEME"],
                      "config_name": src["meta"]["config"]["config_name"],
                      "seed": src["meta"]["config"]["seed"],
                      "n_particles": src["n"],
                      "diameter_median_mm": src["r_med"] * 2000.0,
                      "diameter_min_max_mm": [src["r_min"] * 2000.0,
                                              src["r_max"] * 2000.0],
                      "polydisperse": not src["r_uniform"],
                      "settled_sim_time_s": src["meta"]["settling_gate"][
                          "settled_sim_time_s"],
                      "frame": src["frame"],
                      "material_status": src["meta"]["material_assumptions"]["status"]},
           "calib": {"path": "sim_scripts/kinect_calib.yaml",
                     "sha16": sha256(CALIB)[:16], "rmse_mm": calib["rmse_mm"]},
           "env": {"python": sys.version.split()[0], "numpy": np.__version__}}

    hm_a, res["H1_pathA_on_real_pile"] = gate_pathA(src, spec)
    res["H2_cell_sweep"] = gate_cell_sweep(src)
    res["H5_parity_on_real_pile"] = gate_parity(src, spec, hm_a, calib)
    hm_bmax = Heightmap.load(OUT / "hm2_pathB_depth_nadir.npz")
    hm_bmed = Heightmap.load(OUT / "hm2_pathB_depth_nadir_median.npz")
    v_ref = res["H2_cell_sweep"]["envelope_reference"]["volume_cm3"] * 1e-6
    res["H3_volume_bias"] = gate_volume(src, hm_a, hm_bmax, hm_bmed, v_ref)
    res["H4_real_slope_error_budget"] = gate_slope(src, hm_a, calib)
    res["pngs"] = make_pngs(src, spec, res)

    if args.rerun:
        try:
            res["rerun"] = emit_rerun(src, spec, res)
        except Exception as exc:  # noqa: BLE001 — D447: 침묵 사망 금지
            import traceback
            res["rerun"] = {"error": repr(exc), "traceback": traceback.format_exc()}

    gates = {k: res[k]["pass"] for k in sorted(res) if k.startswith("H")}
    fails = [k for k, v in gates.items() if not v]
    res["gates"] = gates
    res["verdict"] = {
        "code": "HM2_ALL_GATES_PASS" if not fails else "HM2_FAIL_" + "_".join(fails).upper(),
        "non_claims": ("s1 재료값은 provisional 미측정(펠릿 미구매) — 안식각/밀도/마찰 "
                       "현실성 미주장. 실제 Kinect 프레임 미사용(합성 깊이 렌더). "
                       "pile 의 RoArm base 프레임 배치 미확정 — 좌표계는 s1 의 "
                       "container-floor-center 그대로다. 스쿱 물리·퍼내기 동작 미주장."),
    }
    res["wall_seconds"] = round(time.time() - t0, 1)

    p = OUT / "hm2_results.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"verdict": res["verdict"]["code"], "gates": gates,
                      "out": str(OUT.relative_to(REPO)),
                      "results_sha16": sha256(p)[:16]}, ensure_ascii=False, indent=2))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
