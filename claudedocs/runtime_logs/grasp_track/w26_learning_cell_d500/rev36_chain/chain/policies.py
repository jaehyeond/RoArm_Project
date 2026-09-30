"""rev36-chain 2단계 결정 실험용 **학습 없는 규칙 정책** 3개(CPU). 오케스트레이터가 셀마다 "다음 명령점(+롤)"을 고를 때 쓴다.

정의(사전 등록 — 2단계 결과를 본 뒤 바꾸지 않는다)
    후보 = 가능 위치 지도(site_map.py, 롤 탐색 포함)의 feasible 행. 각 후보는 명령점(x, y)·best_roll·립 xy·툴 yaw 를 가진다.
    발자국(footprint) = 셀 시뮬과 같은 툴 충돌 셸(고정부 + 문 열림 27.5°)의 정점을 툴 yaw 로 돌려 립 xy 에 놓고,
        xy 평면에 투영한 볼록 껍질 안에 중심이 드는 5 mm 격자 칸. (툴이 실제로 덮는 자리 — 구덩이 관측 범위 x −10~+65 mm 와 일치)
    ① random           : 후보 중 균등 무작위(시드 고정).
    ② highest          : 발자국 안 높이지도(참값, m) **평균**이 가장 큰 후보. 동점은 무작위.
    ③ footprint_volume : 발자국 안 Σ max(h − h_ref, 0)·칸면적 이 가장 큰 후보. h_ref = 더미 전체 칸 높이의 10 백분위(바닥 근처 기준면).
        ②는 "가장 높은 곳", ③은 "퍼 올릴 재료가 가장 많이 모인 곳"이며 평평한 더미에서는 거의 같고 원뿔·경사 더미에서 갈린다.
    높이지도 = 더미 NPZ 의 구를 `heightmap_from_particles` 로 정확히 올린 것(셀 행의 hm_pre_truth 와 같은 정의).
"""
import json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import site_map as SM                                            # noqa: E402  (경로·파라미터 병합·툴 셸)
from roarm_rl.heightmap import GridSpec, heightmap_from_particles  # noqa: E402

POLICIES = ("random", "highest", "footprint_volume")


def grid_from_box(box):
    c = 0.005
    return GridSpec(origin_xy_m=(float(box[0, 0]), float(box[1, 0])), cell_m=c,
                    shape=(int(np.ceil((box[1, 1] - box[1, 0]) / c)), int(np.ceil((box[0, 1] - box[0, 0]) / c))),
                    frame="deme_box_floor_center", z_datum_m=0.0)


def heightmap_from_pile(pile_npz, spec):
    z = np.load(pile_npz, allow_pickle=True)
    hm = heightmap_from_particles(np.asarray(z["positions_m"], float), np.asarray(z["radii_m"], float), spec)
    return hm.height.astype(np.float64)


def _convex_hull(pts):
    """2D 볼록 껍질(Andrew monotone chain), 반시계."""
    P = sorted(set(map(tuple, np.asarray(pts, float))))
    if len(P) < 3:
        return np.asarray(P)
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return np.asarray(lo[:-1] + up[:-1])


class Footprint:
    """툴 셸 → 툴 프레임 xy 볼록 껍질(고정부 + 문 열림). 후보마다 yaw 회전·립 평행이동 → 격자 마스크."""

    def __init__(self, P, T):
        # 문 OBJ 는 q_open(열림) 자세로 구워져 있어 door_v + hinge_off 가 곧 열린 문의 툴 프레임 좌표다.
        pts = np.vstack([T["fixed_v"][:, :2], (T["door_v"] + T["hinge_off"])[:, :2]])
        self.hull_tool = _convex_hull(pts)                                    # 툴 프레임(립 원점, m)
        self.area_m2 = 0.5 * abs(np.dot(self.hull_tool[:, 0], np.roll(self.hull_tool[:, 1], -1)) - np.dot(self.hull_tool[:, 1], np.roll(self.hull_tool[:, 0], -1)))

    def hull_world(self, lip_xy, yaw_deg):
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        R = np.array([[c, -s], [s, c]])
        return (R @ self.hull_tool.T).T + np.asarray(lip_xy, float)

    def mask(self, spec, lip_xy, yaw_deg):
        hull = self.hull_world(lip_xy, yaw_deg)
        rows, cols = spec.shape
        yc = spec.origin_xy_m[1] + (np.arange(rows) + 0.5) * spec.cell_m
        xc = spec.origin_xy_m[0] + (np.arange(cols) + 0.5) * spec.cell_m
        X, Y = np.meshgrid(xc, yc)
        inside = np.ones(X.shape, bool)
        n = len(hull)
        for i in range(n):                                                   # 반시계 껍질: 모든 변의 왼쪽이면 안
            ax, ay = hull[i]; bx, by = hull[(i + 1) % n]
            inside &= ((bx - ax) * (Y - ay) - (by - ay) * (X - ax)) >= 0
        return inside


def tool_yaw_of(row, P):
    """지도 행의 툴 yaw(상자 좌표, °). 롤 탐색 지도(V2)는 tool_yaw_box_deg 가 있고, 옛 지도(V1)는 R_box_robot 의 yaw + base."""
    if row.get("tool_yaw_box_deg") is not None:
        return float(row["tool_yaw_box_deg"])
    _, R, _ = SM.FK.w25_frame(P)
    y0 = math.degrees(math.atan2(R.T[1, 0], R.T[0, 0]))
    return y0 + float(row.get("base_deg") or 0.0)


def score_candidates(policy, H, spec, cands, P, fp):
    if policy == "random":
        return np.zeros(len(cands))
    h_ref = float(np.percentile(H, 10))
    sc = np.empty(len(cands))
    for i, r in enumerate(cands):
        m = fp.mask(spec, r["lip_box_xy_m"], tool_yaw_of(r, P))
        if not m.any():
            sc[i] = -np.inf; continue
        if policy == "highest":
            sc[i] = float(H[m].mean())
        elif policy == "footprint_volume":
            sc[i] = float(np.clip(H[m] - h_ref, 0, None).sum() * spec.cell_m ** 2)
        else:
            raise SystemExit(f"알 수 없는 정책 {policy}")
    return sc


def choose(policy, H, spec, cands, P, fp, rng):
    """반환 (선택 행, 점수 배열, 선택 인덱스). 동점(또는 random)은 rng 로."""
    sc = score_candidates(policy, H, spec, cands, P, fp)
    if policy == "random":
        i = int(rng.integers(len(cands)))
    else:
        best = np.flatnonzero(sc >= sc.max() - 1e-12)
        i = int(best[rng.integers(len(best))])
    return cands[i], sc, i


def load_site_map(path):
    d = json.load(open(path))
    return [r for r in d["rows"] if r["feasible"]], d
