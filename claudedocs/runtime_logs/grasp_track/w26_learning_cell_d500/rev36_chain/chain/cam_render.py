"""rev36-chain 카메라 시점 높이지도(CPU). 알 구 + 트레이 벽 + 바닥을 깊이 카메라로 그려(z-버퍼) 깊이 영상을 만들고,
실물 파이프라인과 **같은 연산자**(`roarm_rl.heightmap.heightmap_from_kinect_depth`, W22 real_pipeline
`box_heightmap.py:97-101` 의 상수: 유효 깊이 0.30~2.00 m · 높이 −0.02~0.20 m · 모서리 30 mm · 이상치 15 mm ·
agg=max · 채움 0)로 상자 좌표 높이지도로 바꾼다. 가려진 칸은 valid=False 로 남는다(메우지 않음).

카메라 자세 출처
    W24 카메라 배치 권고(H 0.80 m, 로봇 쪽 6° 기울임, NFOV unbinned 640×576) —
    `orca_worktree_archive/RoArm_Project/w24-camera-placement/claudedocs/research/camera_placement_20260929/
    sim_camera_definition.json` (sha256 85e028b0…) 의 `recommended.T_box_depthcam_k4a_axes`.
    ⚠ 그 파일의 상자 축은 로봇이 상자 **+y** 쪽인 규약(상자 x → 로봇 +y, = rev34 규약 "B")이다.
    rev34 셀 시뮬은 규약 **"A"**(상자 x → 로봇 −y, 로봇이 상자 −y 쪽)라 두 좌표는 z 축 둘레 180° 다르다.
    물리 배치는 같고 이름표만 다르므로 A 로 쓸 때는 T_A = diag(−1,−1,1)·T_B 로 옮긴다.
    내부 파라미터는 **사양서 FoI 역산 공칭값**(W24 가 "dry run 전용"이라 표시) — 실기 공장 캘리브레이션 값이 아니다.
    팔 가림은 넣지 않는다(촬영 때 팔은 P1 자세라는 W24 권고 전제; P1 가림 검사는 W24 몫).
"""
import math

import numpy as np

from roarm_rl.heightmap import GridSpec, heightmap_from_kinect_depth

W24_SOURCE = {"path": "/media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/w24-camera-placement/claudedocs/"
                      "research/camera_placement_20260929/sim_camera_definition.json",
              "sha256": "85e028b0d4b18f905bf27fce466e6c4cbb26e18023ebc2ea2acb4e751698d913",
              "key": "recommended.T_box_depthcam_k4a_axes", "frame": "B (robot at box +y)"}
T_BOX_B_DEPTHCAM = np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.994522, 0.104528, -0.058858],
                             [0.0, 0.104528, -0.994522, 0.6], [0.0, 0.0, 0.0, 1.0]])
INTR_NOMINAL = {"fx": 417.0321193091858, "fy": 452.0694462098372, "cx": 320.0, "cy": 288.0, "W": 640, "H": 576,
                "source": "W24 intrinsics_nominal_from_spec_ONLY_for_dry_run (FoI 75°×65° 핀홀 역산)"}
REAL_OP = {"depth_valid_range_m": (0.30, 2.00), "z_range_m": (-0.020, 0.200), "edge_jump_m": 0.030,
           "outlier_delta_m": 0.015, "fill_m": 0.0,
           "source": "W22 real_pipeline rp_common.py:50-55 · box_heightmap.py:97-101 (P38_* 상수)"}
FLIP_B_TO_A = np.diag([-1.0, -1.0, 1.0])


def camera_T(convention):
    if convention == "B":
        return T_BOX_B_DEPTHCAM.copy()
    if convention == "A":
        T = np.eye(4)
        T[:3, :3] = FLIP_B_TO_A @ T_BOX_B_DEPTHCAM[:3, :3]
        T[:3, 3] = FLIP_B_TO_A @ T_BOX_B_DEPTHCAM[:3, 3]
        return T
    raise SystemExit(f"카메라는 rev34 상자 규약 A/B 에서만 정의된다: {convention!r}")


def _rays(intr):
    v, u = np.mgrid[0:intr["H"], 0:intr["W"]].astype(np.float64)
    return np.stack([(u - intr["cx"]) / intr["fx"], (v - intr["cy"]) / intr["fy"], np.ones_like(u)], -1)  # d_z = 1


def render_depth(sph_c, sph_r, tray_tris_box, T_box_cam, intr=INTR_NOMINAL, floor_z=0.0, chunk=100_000):
    """깊이 영상(m, H×W, 없는 곳 NaN). 깊이 = 카메라 z(Kinect 규약). 구는 정확한 광선-구 교차, 벽은 광선-삼각형,
    바닥은 z=floor_z 평면."""
    R, t = T_box_cam[:3, :3], T_box_cam[:3, 3]
    H, W = intr["H"], intr["W"]
    D = np.full(H * W, np.inf)
    rays = _rays(intr).reshape(-1, 3)
    # 바닥 평면(상자 좌표 z = floor_z): n·(R p_cam + t) = floor_z → (Rᵀn)·p_cam = floor_z − t_z
    n_c = R.T @ np.array([0.0, 0.0, 1.0]); c_c = floor_z - t[2]
    den = rays @ n_c
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = c_c / den
    ok = np.isfinite(tf) & (tf > 0)
    D[ok] = np.minimum(D[ok], tf[ok])
    # 트레이 벽 삼각형
    for tri in tray_tris_box:
        v0, v1, v2 = [(R.T @ (np.asarray(p, float) - t)) for p in tri]
        e1, e2 = v1 - v0, v2 - v0
        pv = np.cross(rays, e2); det = pv @ e1
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            tv = -v0
            uu = (pv @ tv) * inv
            qv = np.cross(tv, e1)
            vv = (rays @ qv) * inv
            tt = (qv @ e2) * inv
            hit = (np.abs(det) > 1e-15) & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt > 0)
        D[hit] = np.minimum(D[hit], tt[hit])
    # 구: 투영 중심 주변 창의 픽셀마다 광선-구 교차
    C = (np.asarray(sph_c, float) - t) @ R          # 카메라 좌표 = R^T (p - t)
    r_all = np.asarray(sph_r, float)
    for s in range(0, len(C), chunk):
        c, r = C[s:s + chunk], r_all[s:s + chunk]
        front = c[:, 2] > 1e-3
        c, r = c[front], r[front]
        u = intr["fx"] * c[:, 0] / c[:, 2] + intr["cx"]; v = intr["fy"] * c[:, 1] / c[:, 2] + intr["cy"]
        k = int(math.ceil(max(intr["fx"], intr["fy"]) * float(r.max()) / float(c[:, 2].min()))) + 1
        u0, v0 = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
        for du in range(-k, k + 1):
            for dv in range(-k, k + 1):
                uu, vv = u0 + du, v0 + dv
                inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                if not inside.any():
                    continue
                ui, vi, ci, ri = uu[inside], vv[inside], c[inside], r[inside]
                d = np.stack([(ui - intr["cx"]) / intr["fx"], (vi - intr["cy"]) / intr["fy"], np.ones(len(ui))], 1)
                a = (d * d).sum(1); b = (d * ci).sum(1); cc = (ci * ci).sum(1) - ri * ri
                disc = b * b - a * cc
                hit = disc >= 0
                tt = (b[hit] - np.sqrt(disc[hit])) / a[hit]
                idx = vi[hit] * W + ui[hit]
                np.minimum.at(D, idx, tt)
    D[~np.isfinite(D)] = np.nan
    return D.reshape(H, W)


def heightmap_from_render(depth_m, T_box_cam, spec: GridSpec, intr=INTR_NOMINAL):
    op = REAL_OP
    hm, filt = heightmap_from_kinect_depth(
        depth_m, {k: intr[k] for k in ("fx", "fy", "cx", "cy")}, T_box_cam[:3, :3], T_box_cam[:3, 3], spec,
        unit="m", depth_valid_range_m=op["depth_valid_range_m"], z_range_m=op["z_range_m"],
        edge_jump_m=op["edge_jump_m"], outlier_delta_m=op["outlier_delta_m"], fill_m=op["fill_m"],
        extra_meta={"sim_render": True, "camera_source": W24_SOURCE, "intrinsics_source": intr.get("source")})
    return hm, filt
