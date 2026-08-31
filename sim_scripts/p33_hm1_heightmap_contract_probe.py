#!/usr/bin/env python
"""p33 / hm1 — heightmap 공통 모듈 계약 검증 (물리 0, Isaac 0, 로봇 0).

검증 대상 = ``roarm_rl.heightmap`` (s2 신규 모듈).  세 가지를 묻는다:

  G1 analytic  — 알려진 합성 형상(평면 / 경사면 0..80° / 원뿔 더미)에서 heightmap이
                 기하학적으로 맞는가.  셀 집계 편향의 해석해와 실측이 일치하는가.
  G2 steep     — D453이 경고한 지점.  가파른 면에서 몇 도에 몇 mm가 나는가.
                 (a) 셀 풋프린트 편향  (b) 수평 표현오차 tan(θ) 증폭
                 (c) Kinect 캘리브 RMSE 전파  (d) 입자 경로 자체 양자화
  G3 parity    — 경로 (A) 입자배열 / (B) 깊이영상이 같은 규격을 뱉는가.
                 헤더·dtype·shape·빈셀 규약 완전 일치 + 동일 장면 값 비교.

출력 = claudedocs/runtime_logs/heightmap_track/hm1_s2/  (신규 폴더, 기존 것 재사용 0)

사용:
  python3 sim_scripts/p33_hm1_heightmap_contract_probe.py
  /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p33_hm1_heightmap_contract_probe.py --rerun
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# roarm_rl/__init__.py pulls in gymnasium/Isaac; heightmap.py is numpy-only, so
# load it by path to stay runnable in a bare python as well as in isaaclab.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "roarm_heightmap", REPO / "roarm_rl/heightmap.py")
hmlib = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hmlib          # dataclasses needs the module registered
_spec.loader.exec_module(hmlib)

SPEC_VERSION = hmlib.SPEC_VERSION
GridSpec, Heightmap = hmlib.GridSpec, hmlib.Heightmap
cell_slope_deg = hmlib.cell_slope_deg
heightmap_from_depth = hmlib.heightmap_from_depth
heightmap_from_particles = hmlib.heightmap_from_particles
heightmap_from_points = hmlib.heightmap_from_points
lateral_to_vertical_m = hmlib.lateral_to_vertical_m
load_kinect_calib = hmlib.load_kinect_calib
slope_aware_tol_m = hmlib.slope_aware_tol_m
slope_bias_m = hmlib.slope_bias_m

OUT = REPO / "claudedocs/runtime_logs/heightmap_track/hm1_s2"
CALIB = REPO / "sim_scripts/kinect_calib.yaml"

SEED = 33001
CELL_M = 0.005                     # 5 mm — 근거는 roarm_rl/heightmap.py 모듈 docstring
GRID_CENTER = (0.200, 0.000)       # RoArm base frame, source 영역 대표점
GRID_EXTENT = 0.150                # 150 mm -> 30 x 30 cells
# 해석 표면 서브샘플: 셀당 SUB x SUB 점.  SUB=50 -> 서브피치 p = 0.1 mm 이고
# D453 수평갭 1.2 mm = 12p 로 정확히 나누어떨어진다.  주입 시프트가 p 의 정수배가
# 아니면 셀 안의 최댓값 샘플이 격자에 걸려 tan(θ) 법칙이 5/6 로 축소 측정된다
# (SUB=10, eps=1.2mm 에서 실제로 관측된 인공물).
SUB = 50
SLOPES_DEG = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 79.5, 80.0)
EPS_COOKED_M = 0.0012              # D453 실측 cooked convex 수평 갭 ~1.2 mm
PELLET_R_M = 0.0025                # s1 DEME smoke 값 (radius_m 2.5 mm = 지름 5 mm)
CONE_H_M, CONE_R_M = 0.060, 0.060  # 원뿔 더미: 높이 60 mm, 바닥반경 60 mm (45° 면)

# 해석해 대비 잔차 허용치.  계약이 height 를 float32 로 저장하므로 h ~ 0.05 m 에서
# 양자화 스텝이 약 6e-9 m (6e-6 mm) 다.  1e-7 m 는 그 위, 물리적 효과(최소 0.1 mm)
# 아래로 잡은 값 — 순수 구현 오차만 잡아낸다.
TOL_ANALYTIC_M = 1.0e-7


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def sample_surface(spec: GridSpec, fn, sub: int = SUB) -> np.ndarray:
    """해석 표면 fn(x, y) -> z 를 셀당 sub x sub 정규 격자로 샘플 -> (N,3) 점군.

    셀 [0, c) 안의 샘플 x-오프셋은 (i + 0.5) * c/sub 이므로 셀 중심 기준 최대
    오프셋은 정확히 (c - c/sub)/2 다.  max 집계 편향의 해석 예측이 이 값에
    의존하므로 무작위가 아니라 결정적 격자를 쓴다 (bit-재현).
    """
    x_min, x_max, y_min, y_max = spec.bounds_m()
    p = spec.cell_m / sub
    xs = np.arange(x_min + p / 2, x_max, p)
    ys = np.arange(y_min + p / 2, y_max, p)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    gz = fn(gx, gy)
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def plane_fn(slope_deg: float, z0: float = 0.010, along="x"):
    """중심을 지나는 경사면. along='x' 단일축, 'xy' 대각(코너 케이스)."""
    cx, cy = GRID_CENTER
    t = np.tan(np.radians(slope_deg))
    if along == "x":
        return lambda x, y: z0 + t * (x - cx)
    k = t / np.sqrt(2.0)            # |grad| = t 를 유지한 대각 경사
    return lambda x, y: z0 + k * (x - cx) + k * (y - cy)


def cone_fn(x, y):
    cx, cy = GRID_CENTER
    r = np.hypot(x - cx, y - cy)
    return np.maximum(0.0, CONE_H_M * (1.0 - r / CONE_R_M))


def cone_pellets(rng: np.random.Generator) -> np.ndarray:
    """원뿔 표면에 접하도록 놓인 펠릿 중심 (N,3).

    삼각격자 x-y 배치 후 각 지점의 원뿔 법선 방향으로 반지름만큼 밀어낸다.
    (표면 접선 배치 -> 상단 포락면 = 원뿔면을 법선으로 r 만큼 오프셋)
    """
    cx, cy = GRID_CENTER
    d = 2.0 * PELLET_R_M
    xs = np.arange(cx - CONE_R_M, cx + CONE_R_M + 1e-9, d)
    ys = np.arange(cy - CONE_R_M, cy + CONE_R_M + 1e-9, d * np.sqrt(3) / 2)
    pts = []
    for j, y in enumerate(ys):
        row_x = xs + (d / 2 if j % 2 else 0.0)
        for x in row_x:
            r = np.hypot(x - cx, y - cy)
            if r > CONE_R_M:
                continue
            # 원뿔면 z = H (1 - r/R): 경사 tan = H/R, 법선 (반경방향 -grad, 1)/|..|
            g = CONE_H_M / CONE_R_M
            n = np.array([0.0, 0.0, 1.0]) if r < 1e-9 else np.array(
                [g * (x - cx) / r, g * (y - cy) / r, 1.0])
            n = n / np.linalg.norm(n)
            base = np.array([x, y, CONE_H_M * (1.0 - r / CONE_R_M)])
            pts.append(base + PELLET_R_M * n)
    P = np.asarray(pts, dtype=np.float64)
    rng.shuffle(P)                 # 순서 무관성 확인용 (max 집계는 순서 불변)
    return P


def render_sphere_depth(centers, radius, intr, R, t, hw=(720, 1280), *,
                        floor_z=0.0, noise_sigma_m=0.0,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """구 집합 + 바닥평면을 카메라(R,t: cam->base)에서 본 depth 영상 [m].

    depth[v,u] = 히트점의 카메라 z (roarm_rl.heightmap.deproject_depth 규약).
    픽셀 광선 cam 방향 m = [(u-cx)/fx, (v-cy)/fy, 1] 이므로 파라미터 s 가 곧 z_cam.

    radius : 스칼라 또는 (N,) 배열 (다분산).  heightmap_from_particles 와 같은 규약.
    """
    H, W = hw
    v, u = np.mgrid[0:H, 0:W]
    mx = (u - intr["cx"]) / intr["fx"]
    my = (v - intr["cy"]) / intr["fy"]
    m = np.stack([mx, my, np.ones_like(mx)], axis=-1)          # (H,W,3) cam
    a = m @ R.T                                                # (H,W,3) base dir
    depth = np.full((H, W), np.inf)

    # 바닥평면 z = floor_z
    with np.errstate(divide="ignore", invalid="ignore"):
        s_floor = (floor_z - t[2]) / a[..., 2]
    ok = np.isfinite(s_floor) & (s_floor > 0)
    depth[ok] = s_floor[ok]

    C = np.asarray(centers, dtype=np.float64)
    rad = np.asarray(radius, dtype=np.float64).ravel()
    if rad.size == 1:
        rad = np.full(C.shape[0], float(rad[0]))
    elif rad.size != C.shape[0]:
        raise ValueError(f"radius must be scalar or ({C.shape[0]},), got {rad.shape}")
    Rt = R.T
    for i, c in enumerate(C):
        radius = float(rad[i])
        pc = Rt @ (c - t)
        if pc[2] <= 1e-6:
            continue
        uc = intr["cx"] + intr["fx"] * pc[0] / pc[2]
        vc = intr["cy"] + intr["fy"] * pc[1] / pc[2]
        pr = intr["fx"] * radius / pc[2] + 2.0
        u0, u1 = int(max(0, np.floor(uc - pr))), int(min(W, np.ceil(uc + pr) + 1))
        v0, v1 = int(max(0, np.floor(vc - pr))), int(min(H, np.ceil(vc + pr) + 1))
        if u0 >= u1 or v0 >= v1:
            continue
        aw = a[v0:v1, u0:u1, :]
        b = t - c
        qa = np.einsum("ijk,ijk->ij", aw, aw)
        qb = 2.0 * (aw @ b)
        qc = float(b @ b - radius * radius)
        disc = qb * qb - 4.0 * qa * qc
        hit = disc >= 0.0
        if not hit.any():
            continue
        s = (-qb[hit] - np.sqrt(disc[hit])) / (2.0 * qa[hit])
        win = depth[v0:v1, u0:u1]
        cur = win[hit]
        win[hit] = np.where((s > 0) & (s < cur), s, cur)
        depth[v0:v1, u0:u1] = win

    depth[~np.isfinite(depth)] = 0.0                            # Kinect "no return"
    if noise_sigma_m > 0.0:
        assert rng is not None
        nz = depth > 0
        depth[nz] += rng.normal(0.0, noise_sigma_m, size=int(nz.sum()))
    return depth


def nadir_pose(height_m: float) -> tuple[np.ndarray, np.ndarray]:
    """격자 바로 위 nadir 카메라 (cam->base).  x_cam=+x, y_cam=-y, z_cam=-z."""
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    t = np.array([GRID_CENTER[0], GRID_CENTER[1], height_m])
    return R, t


def stats_mm(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=np.float64).ravel() * 1000.0
    return {"max_abs_mm": float(np.abs(d).max()), "rms_mm": float(np.sqrt((d ** 2).mean())),
            "mean_mm": float(d.mean()), "p95_abs_mm": float(np.percentile(np.abs(d), 95))}


# --------------------------------------------------------------------------- #
# G1 — analytic shapes
# --------------------------------------------------------------------------- #
def gate_analytic(spec: GridSpec) -> dict:
    centers = spec.cell_centers()
    cx_, cy_ = centers[..., 0], centers[..., 1]
    lattice_half = (spec.cell_m - spec.cell_m / SUB) / 2.0   # 결정적 격자의 실효 반폭
    rows = []

    for name, fn, along in (("flat_z40mm", lambda x, y: np.full_like(x, 0.040), None),
                            ("cone_H60_R60", cone_fn, None)) + tuple(
            (f"slope_{s:g}deg_x", plane_fn(s), "x") for s in SLOPES_DEG) + tuple(
            (f"slope_{s:g}deg_xy", plane_fn(s, along="xy"), "xy") for s in (45.0, 70.0)):
        pts = sample_surface(spec, fn)
        gt = fn(cx_, cy_)
        for agg in ("max", "median"):
            hm = heightmap_from_points(pts, spec, agg=agg)
            err = hm.height.astype(np.float64) - gt
            p_sub = spec.cell_m / SUB
            if name.startswith("slope_") and along == "x":
                s_deg = float(name.split("_")[1].replace("deg", ""))
                t = np.tan(np.radians(s_deg))
                # max  = 셀 안 최상단 샘플 = 중심에서 lattice_half 만큼 위쪽
                # median = 짝수 샘플수 nearest-rank(round-half-up) -> 서브피치 절반
                pred = (lattice_half * t) if agg == "max" else (p_sub / 2.0) * t
            elif name.startswith("slope_") and along == "xy" and agg == "max":
                s_deg = float(name.split("_")[1].replace("deg", ""))
                pred = lattice_half * np.tan(np.radians(s_deg)) * np.sqrt(2.0)
            else:
                pred = None   # 원뿔(셀 내 비선형) / 대각 median(보고 전용)
            row = {"shape": name, "agg": agg, "n_pts": int(pts.shape[0]),
                   "all_valid": bool(hm.valid.all()),
                   "err_vs_cellcenter": stats_mm(err)}
            if pred is not None:
                row["predicted_bias_mm"] = float(pred * 1000.0)
                row["residual_after_pred"] = stats_mm(err - pred)
            rows.append(row)

    # 게이트: 평면류에서 (실측 - 해석예측) 잔차가 float64 수치오차 수준인가
    plane_rows = [r for r in rows if r["shape"].startswith(("flat", "slope"))
                  and "residual_after_pred" in r]
    worst = max(r["residual_after_pred"]["max_abs_mm"] for r in plane_rows)
    # flat 은 예측식 없이 직접 검사
    flat = [r for r in rows if r["shape"] == "flat_z40mm"]
    worst_flat = max(r["err_vs_cellcenter"]["max_abs_mm"] for r in flat)
    passed = (worst <= TOL_ANALYTIC_M * 1000.0) and (worst_flat <= TOL_ANALYTIC_M * 1000.0)
    return {"gate": "G1_analytic", "pass": bool(passed),
            "tol_mm": TOL_ANALYTIC_M * 1000.0,
            "worst_plane_residual_mm": worst, "worst_flat_err_mm": worst_flat,
            "lattice_effective_half_width_m": lattice_half,
            "note": ("경사면에서 max 집계의 편향은 결함이 아니라 결정적 풋프린트 "
                     "효과다.  해석예측 (c - c/SUB)/2 * tan(t) 를 빼면 잔차가 "
                     "float64 수치오차로 떨어져야 정상.  median 집계는 편향 0."),
            "rows": rows}


# --------------------------------------------------------------------------- #
# G2 — steep-face error (D453)
# --------------------------------------------------------------------------- #
def gate_steep(spec: GridSpec, rng: np.random.Generator, calib: dict) -> dict:
    centers = spec.cell_centers()
    cx_, cy_ = centers[..., 0], centers[..., 1]
    lattice_half = (spec.cell_m - spec.cell_m / SUB) / 2.0
    rmse_m = calib["rmse_mm"] / 1000.0
    rows = []
    for s in SLOPES_DEG:
        fn = plane_fn(s)
        t = np.tan(np.radians(min(s, 89.9)))
        pts = sample_surface(spec, fn)

        # (a) 셀 풋프린트 편향 — 실측
        hm_max = heightmap_from_points(pts, spec, agg="max")
        hm_med = heightmap_from_points(pts, spec, agg="median")
        gt = fn(cx_, cy_)
        bias_max = float((hm_max.height.astype(np.float64) - gt).mean())
        bias_med = float((hm_med.height.astype(np.float64) - gt).mean())

        # (b) 수평 표현오차 eps_h 의 tan(t) 증폭 — 실측 (D453 cooked gap 1.2 mm 주입)
        shifted = pts.copy()
        shifted[:, 0] += EPS_COOKED_M
        hm_shift = heightmap_from_points(shifted, spec, agg="max")
        dz_cooked = (hm_shift.height.astype(np.float64)
                     - hm_max.height.astype(np.float64))
        # 격자 경계 효과를 피해 내부 셀만
        core_b = dz_cooked[2:-2, 2:-2]

        # (d) 입자 경로 자체 양자화 — 경사면에 접한 펠릿 층의 footprint-max 편차
        d = 2.0 * PELLET_R_M
        x_min, x_max, y_min, y_max = spec.bounds_m()
        gxp, gyp = np.meshgrid(np.arange(x_min - d, x_max + d, d),
                               np.arange(y_min - d, y_max + d, d * np.sqrt(3) / 2),
                               indexing="xy")
        gxp = gxp + (d / 2) * (np.arange(gyp.shape[0])[:, None] % 2)
        n = np.array([-t, 0.0, 1.0]) / np.sqrt(1 + t * t)      # 경사면 법선
        base = np.stack([gxp.ravel(), gyp.ravel(), fn(gxp, gyp).ravel()], axis=1)
        # 이 경사면은 격자 좌반부에서 z<0 로 내려간다 (기하 테스트용 무한 램프).
        # 지지면을 -1 m 로 내려 floor 가 max 를 이기지 못하게 하고, 실제로 구가
        # 닿은 셀(counts>0)만 평가한다.
        hm_par = heightmap_from_particles(base + PELLET_R_M * n, PELLET_R_M, spec,
                                          floor_z_m=-1.0)
        # 접구 상단 포락면 = 경사면을 법선으로 r 오프셋 = 수직으로 r/cos(t)
        env = gt + PELLET_R_M / np.cos(np.radians(min(s, 89.0)))
        env_fp = env + float(slope_bias_m(spec.cell_m, s))   # 포락면의 footprint-max
        sup = hm_par.counts > 0
        core = np.zeros_like(sup)
        core[3:-3, 3:-3] = True
        m_ = sup & core
        par_dev = (hm_par.height.astype(np.float64) - env_fp)[m_]

        rows.append({
            "slope_deg": s, "tan": float(t),
            "a_footprint_bias_max_mm": bias_max * 1000.0,
            "a_footprint_bias_max_pred_mm": lattice_half * t * 1000.0,
            "a_footprint_bias_median_mm": bias_med * 1000.0,
            "a_footprint_bound_half_cell_mm": 0.5 * spec.cell_m * t * 1000.0,
            "b_cooked_eps1p2mm_measured_mm": float(np.abs(core_b).max()) * 1000.0,
            "b_cooked_eps1p2mm_pred_mm": float(lateral_to_vertical_m(EPS_COOKED_M, s)) * 1000.0,
            "c_calib_rmse10p13mm_pred_mm": float(lateral_to_vertical_m(rmse_m, s)) * 1000.0,
            "d_particle_dev_vs_envelope_mm": stats_mm(par_dev),
            "d_particle_support_frac": float(m_.sum() / max(core.sum(), 1)),
            "d_particle_quantization_bound_mm": (PELLET_R_M / np.cos(
                np.radians(min(s, 89.0)))) * 1000.0,
            "suggested_slope_aware_tol_mm": float(
                slope_aware_tol_m(spec.cell_m, rmse_m, s)) * 1000.0,
        })

    tol_mm = TOL_ANALYTIC_M * 1000.0
    d453 = [r for r in rows if r["slope_deg"] == 79.5][0]
    ok_law = all(abs(r["b_cooked_eps1p2mm_measured_mm"]
                     - r["b_cooked_eps1p2mm_pred_mm"]) <= tol_mm for r in rows)
    ok_bias = all(abs(r["a_footprint_bias_max_mm"]
                      - r["a_footprint_bias_max_pred_mm"]) <= tol_mm for r in rows)
    return {"gate": "G2_steep_face", "pass": bool(ok_law and ok_bias),
            "law_matches_measurement": bool(ok_law),
            "footprint_bias_matches_prediction": bool(ok_bias),
            "d453_crosscheck_79p5deg": {
                "d453_reported_mm": 6.4,
                "this_probe_cooked_eps1p2mm_mm": d453["b_cooked_eps1p2mm_measured_mm"],
                "reading": ("D453의 6.4 mm 는 1.2 mm 수평갭 x tan(79.5°) 로 "
                            "재현된다 -> 증폭 법칙 확인.  단 D453 은 Isaac cooked "
                            "hull 이 원인이었고, 여기서는 같은 법칙이 캘리브 "
                            "오차/셀 풋프린트에도 그대로 적용됨을 보인다.")},
            "eps_cooked_m": EPS_COOKED_M, "calib_rmse_mm": calib["rmse_mm"],
            "rows": rows}


# --------------------------------------------------------------------------- #
# G3 — path (A) / (B) parity
# --------------------------------------------------------------------------- #
def _contract_keys(hm: Heightmap) -> dict:
    h = hm.header()
    return {k: h[k] for k in ("spec_version", "frame", "cell_m", "cell_mm",
                              "origin_xy_m", "origin_is", "shape", "indexing",
                              "cell_center_formula", "z_datum_m", "height_unit",
                              "height_dtype", "valid_dtype", "counts_dtype",
                              "empty_cell")}


def gate_parity(spec: GridSpec, rng: np.random.Generator, calib: dict) -> dict:
    pellets = cone_pellets(rng)
    intr = calib["intrinsics"]

    # (A) 입자 -> heightmap
    hm_a = heightmap_from_particles(pellets, PELLET_R_M, spec,
                                    extra_meta={"scene": "cone_pellets"})

    # (B) 같은 펠릿을 nadir 깊이영상으로 렌더 -> heightmap
    Rn, tn = nadir_pose(0.900)
    depth_n = render_sphere_depth(pellets, PELLET_R_M, intr, Rn, tn)
    hm_b = heightmap_from_depth(depth_n, intr, Rn, tn, spec, agg="max",
                                depth_valid_range_m=(0.30, 2.00),
                                extra_meta={"scene": "cone_pellets",
                                            "camera_pose": "synthetic_nadir_0.90m"})

    # (B') 실제 캘리브 Kinect 포즈(사각) -> 그림자/무효 셀 정량화
    depth_k = render_sphere_depth(pellets, PELLET_R_M, intr, calib["R"], calib["t"])
    hm_k = heightmap_from_depth(depth_k, intr, calib["R"], calib["t"], spec,
                                agg="max",
                                extra_meta={"scene": "cone_pellets",
                                            "camera_pose": "kinect_calib_yaml"})

    # (B'') 깊이 잡음 하에서 집계 선택 비교.
    #  sigma 17 mm = Azure Kinect NFOV unbinned 무작위오차 스펙 상한 (전 사거리),
    #  sigma 3 mm  = 0.9 m 근거리에서 흔히 보고되는 실측 수준 (참고치, 실측 아님)
    noise_rows = []
    for sig in (0.003, 0.017):
        depth_nz = render_sphere_depth(pellets, PELLET_R_M, intr, Rn, tn,
                                       noise_sigma_m=sig, rng=rng)
        for agg in ("max", "p95", "p90", "median"):
            hm_n = heightmap_from_depth(depth_nz, intr, Rn, tn, spec, agg=agg)
            d = (hm_n.height.astype(np.float64)
                 - hm_b.height.astype(np.float64))[hm_b.valid]
            noise_rows.append({"sigma_mm": sig * 1000.0, "agg": agg,
                               "vs_noiseless_max": stats_mm(d)})

    ka, kb = _contract_keys(hm_a), _contract_keys(hm_b)
    same_contract = ka == kb
    diff_ab = (hm_a.height.astype(np.float64) - hm_b.height.astype(np.float64))
    both = hm_a.valid & hm_b.valid
    slope = cell_slope_deg(hm_a)

    hm_a.save(OUT / "hm1_pathA_particles.npz")
    hm_b.save(OUT / "hm1_pathB_depth_nadir.npz")
    hm_k.save(OUT / "hm1_pathB_depth_kinectpose.npz")

    return {"gate": "G3_parity", "pass": bool(same_contract),
            "contract_identical": bool(same_contract),
            "contract_keys_A": ka,
            "contract_diff": {k: [ka.get(k), kb.get(k)] for k in set(ka) | set(kb)
                              if ka.get(k) != kb.get(k)},
            "dtypes": {"A": [str(hm_a.height.dtype), str(hm_a.valid.dtype),
                             str(hm_a.counts.dtype)],
                       "B": [str(hm_b.height.dtype), str(hm_b.valid.dtype),
                             str(hm_b.counts.dtype)]},
            "n_pellets": int(pellets.shape[0]),
            "A_valid_frac": float(hm_a.valid.mean()),
            "B_nadir_valid_frac": float(hm_b.valid.mean()),
            "B_kinectpose_valid_frac": float(hm_k.valid.mean()),
            "B_kinectpose_shadow_cells": int((~hm_k.valid).sum()),
            "AB_value_diff_nadir": stats_mm(diff_ab[both]),
            "AB_value_diff_by_slope_band_mm": {
                band: stats_mm(diff_ab[both & (slope >= lo) & (slope < hi)])
                if (both & (slope >= lo) & (slope < hi)).any() else None
                for band, (lo, hi) in {"0-15deg": (0, 15), "15-35deg": (15, 35),
                                       "35-60deg": (35, 60), "60-90deg": (60, 90)}.items()},
            "kinect_noise_sigma17mm_agg_compare": noise_rows,
            "note": ("(A)는 수직 ray-drop, (B)는 카메라 광선이라 시점이 다르다. "
                     "nadir 렌더는 시점 차이를 제거한 규격/값 비교이고, "
                     "kinect_calib.yaml 포즈 렌더는 실제 사각 시점의 그림자 비용이다.")}


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def make_pngs(spec: GridSpec, res: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    hm_a = Heightmap.load(OUT / "hm1_pathA_particles.npz")
    hm_b = Heightmap.load(OUT / "hm1_pathB_depth_nadir.npz")
    hm_k = Heightmap.load(OUT / "hm1_pathB_depth_kinectpose.npz")
    ext = [spec.bounds_m()[0], spec.bounds_m()[1], spec.bounds_m()[2], spec.bounds_m()[3]]

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    for a, hm, ttl in ((ax[0], hm_a, "(A) particles ray-drop"),
                       (ax[1], hm_b, "(B) depth, nadir 0.90 m"),
                       (ax[2], hm_k, "(B) depth, kinect_calib pose")):
        im = a.imshow(hm.height * 1000, origin="lower", extent=ext, cmap="viridis")
        a.set_title(f"{ttl}\nvalid={hm.valid.mean()*100:.1f}%", fontsize=9)
        fig.colorbar(im, ax=a, label="height [mm]")
    d = (hm_a.height.astype(float) - hm_b.height.astype(float)) * 1000
    im = ax[3].imshow(d, origin="lower", extent=ext, cmap="coolwarm",
                      vmin=-np.abs(d).max(), vmax=np.abs(d).max())
    ax[3].set_title("(A) - (B nadir) [mm]", fontsize=9)
    fig.colorbar(im, ax=ax[3])
    fig.suptitle(f"hm1 — cone pellet pile, cell={spec.cell_m*1000:.0f} mm, "
                 f"{spec.shape[0]}x{spec.shape[1]}, spec={SPEC_VERSION}")
    fig.tight_layout()
    p = OUT / "hm1_parity.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(p.name)

    rows = res["G2_steep_face"]["rows"]
    s = [r["slope_deg"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(s, [r["a_footprint_bound_half_cell_mm"] for r in rows], "o-",
               label="(a) cell footprint, agg=max  (c/2)·tanθ")
    ax[0].plot(s, [r["b_cooked_eps1p2mm_measured_mm"] for r in rows], "s-",
               label="(b) cooked hull ε=1.2 mm (D453) measured")
    ax[0].plot(s, [r["c_calib_rmse10p13mm_pred_mm"] for r in rows], "^-",
               label="(c) Kinect calib RMSE 10.13 mm")
    ax[0].plot(s, [r["d_particle_quantization_bound_mm"] for r in rows], "d-",
               label="(d) pellet r/cosθ (path A quantisation)")
    ax[0].axvline(79.5, color="k", ls=":", lw=1)
    ax[0].annotate("D453: 6.4 mm @79.5°", (79.5, 6.4), fontsize=8,
                   xytext=(45, 12), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[0].set_yscale("log")
    ax[0].set_xlabel("face slope θ [deg]")
    ax[0].set_ylabel("vertical error [mm]  (log)")
    ax[0].set_title("steep-face vertical error sources — tan(θ) amplification")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(s, [r["a_footprint_bias_max_mm"] for r in rows], "o-", label="max agg (measured)")
    ax[1].plot(s, [r["a_footprint_bias_max_pred_mm"] for r in rows], "k--", lw=1,
               label="max agg (predicted)")
    ax[1].plot(s, [r["a_footprint_bias_median_mm"] for r in rows], "s-", label="median agg")
    ax[1].plot(s, [r["suggested_slope_aware_tol_mm"] for r in rows], "r:",
               label="suggested slope-aware tol")
    ax[1].set_xlabel("face slope θ [deg]")
    ax[1].set_ylabel("mean cell bias [mm]")
    ax[1].set_title("aggregation bias on a plane (cell = 5 mm)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "hm1_slope_error.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(p.name)
    return paths


def emit_rerun(spec: GridSpec, res: dict) -> dict:
    """선택적 D341 관측층 — isaaclab env(rerun 0.34.1)에서만 동작."""
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    hm_a = Heightmap.load(OUT / "hm1_pathA_particles.npz")
    hm_b = Heightmap.load(OUT / "hm1_pathB_depth_nadir.npz")
    hm_k = Heightmap.load(OUT / "hm1_pathB_depth_kinectpose.npz")
    c = spec.cell_centers()
    rrd = OUT / "hm1_timeline.rrd"
    rbl = OUT / "hm1_timeline.rbl"
    app_id = "roarm_hm1"
    bp = rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="/", contents=["/hmap/**"], name="1 | heightmaps"),
        rrb.TimeSeriesView(origin="/slope", contents="/slope/**",
                           name="2 | steep-face error")),
        auto_layout=False, auto_views=False, collapse_panels=True)

    with rr.RecordingStream(app_id, recording_id="hm1_s2", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(rrd), write_footer=True)
        rec.send_blueprint(bp, make_active=True, make_default=True)
        for ent, hm, col in (("hmap/pathA_particles", hm_a, [230, 160, 40]),
                             ("hmap/pathB_nadir", hm_b, [60, 170, 230]),
                             ("hmap/pathB_kinectpose", hm_k, [180, 90, 200])):
            v = hm.valid.ravel()
            xyz = np.stack([c[..., 0].ravel(), c[..., 1].ravel(),
                            hm.height.astype(np.float64).ravel()], axis=1)[v]
            rec.log(ent, rr.Points3D(xyz, colors=[col], radii=0.0012), static=True)
        for r in res["G2_steep_face"]["rows"]:
            rec.set_time("slope_deg", duration=float(r["slope_deg"]))
            rec.log("slope/footprint_max_mm", rr.Scalars(r["a_footprint_bound_half_cell_mm"]))
            rec.log("slope/cooked_1p2mm_mm", rr.Scalars(r["b_cooked_eps1p2mm_measured_mm"]))
            rec.log("slope/calib_rmse_mm", rr.Scalars(r["c_calib_rmse10p13mm_pred_mm"]))
            rec.log("slope/pellet_quant_mm", rr.Scalars(r["d_particle_quantization_bound_mm"]))
        rec.flush(timeout_sec=60.0)
    bp.save(app_id, str(rbl))
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    ents = ["hmap/pathA_particles", "hmap/pathB_nadir", "hmap/pathB_kinectpose",
            "slope/footprint_max_mm", "slope/cooked_1p2mm_mm",
            "slope/calib_rmse_mm", "slope/pellet_quant_mm"]
    return validate_rerun_artifact(
        rrd,
        expected_entity_paths=ents, exact_entity_paths=ents,
        exact_timeline_names=["blueprint", "log_time", "slope_deg"],
        expected_entity_components={
            "hmap/pathA_particles": pts3, "hmap/pathB_nadir": pts3,
            "hmap/pathB_kinectpose": pts3,
            **{e: ["Scalars:scalars"] for e in ents if e.startswith("slope/")}},
        blueprint_path=rbl, screenshot_path=OUT / "hm1_inspection.png",
        expected_version="0.34.1", timeout_s=300.0,
        cli_path="/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true",
                    help="D341 관측층 RRD 생성 (isaaclab env 필요)")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    calib = load_kinect_calib(CALIB)
    spec = GridSpec.centered(GRID_CENTER, GRID_EXTENT, CELL_M)

    res = {"tool": "p33_hm1_heightmap_contract_probe", "case": "hm1_s2",
           "spec_version": SPEC_VERSION, "seed": SEED,
           "module": {"path": "roarm_rl/heightmap.py",
                      "sha16": sha16(REPO / "roarm_rl/heightmap.py")},
           "calib": {"path": "sim_scripts/kinect_calib.yaml", "sha16": sha16(CALIB),
                     "rmse_mm": calib["rmse_mm"]},
           "env": {"python": sys.version.split()[0], "numpy": np.__version__},
           "grid": spec.to_header(),
           "cell_size_rationale": {
               "chosen_mm": CELL_M * 1000.0,
               "pellet_diameter_mm": PELLET_R_M * 2000.0,
               "lower_bound": ("< 1 pellet diameter -> 셀당 펠릿 1개 미만 -> 크라운/골 "
                               "교대로 +-반지름(2.5 mm) 앨리어싱만 늘어남"),
               "upper_bound": ("10 mm(yard_track, D453/p26)는 '최소 물체 폭 22 mm의 "
                               "절반' 논거였고 연속 입자엔 미적용; 펠릿 지름 2배라 "
                               "스쿱 트렌치 벽이 뭉개짐 — 잔여 형상 예측이 연구 목표"),
               "sensor": ("Kinect 표준거리 0.9 m 에서 색정렬 depth 픽셀 풋프린트 "
                          "z/fx = 0.9/608.33 = 1.48 mm -> 5 mm 셀당 약 11 샘플, "
                          "3 mm 셀은 약 4 샘플로 NFOV 무작위오차 평균화 부족"),
               "model_input": "150x150 mm 영역 -> 30x30 (5 mm) / 50x50 (3 mm) / 15x15 (10 mm)",
               "d454_note": ("yard_track 착지분산 p95 43~63 mm 는 이산 물체 place "
                             "결정층 수치 — 배출 위치가 고정된 현 방향에서는 "
                             "관측 셀 크기의 근거가 되지 않음")}}

    res["G1_analytic"] = gate_analytic(spec)
    res["G2_steep_face"] = gate_steep(spec, rng, calib)
    res["G3_parity"] = gate_parity(spec, rng, calib)
    res["pngs"] = make_pngs(spec, res)

    if args.rerun:
        try:
            res["rerun"] = emit_rerun(spec, res)
        except Exception as exc:  # noqa: BLE001 — D447: 침묵 사망 금지
            import traceback
            res["rerun"] = {"error": repr(exc), "traceback": traceback.format_exc()}

    gates = {k: res[k]["pass"] for k in ("G1_analytic", "G2_steep_face", "G3_parity")}
    fails = [k for k, v in gates.items() if not v]
    res["gates"] = gates
    res["verdict"] = {
        "code": "HM1_ALL_GATES_PASS" if not fails else "HM1_FAIL_" + "_".join(fails).upper(),
        "non_claims": ("실제 Kinect 프레임 미사용(합성 깊이 렌더) — 실 depth 잡음 "
                       "모델·다중경로·표면 반사율 충실도 미주장.  DEME 정착 결과 "
                       "미수신(s1 smoke 는 타이밍만) — 실제 입자 배치 통계 미주장. "
                       "안식각/스쿱 물리 미주장.")}
    res["wall_seconds"] = round(time.time() - t0, 1)

    p = OUT / "hm1_results.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"verdict": res["verdict"]["code"], "gates": gates,
                      "out": str(OUT.relative_to(REPO)),
                      "results_sha16": sha16(p)}, ensure_ascii=False, indent=2))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
