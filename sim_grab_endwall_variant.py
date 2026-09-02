#!/usr/bin/env python3
"""그랩 v1 에 끝벽(end wall)을 붙인 변형본 생성 + 밀폐 공동 검사 (track P4g).

왜
--
`claudedocs/runtime_logs/newton_mpm/b1_scoop/grab_cavity_scan.json` 이
그랩 v1 셸 2매가 **개폐 전 구간(0~44.5도)에서 밀폐 공동 0.28~0.36 cm³** 밖에 만들지
않으며 그마저도 상단 플랜지 안이라는 것을 보였다. 보울 영역은 어느 각도에서도 아무것도
가두지 않는다. `design.json` 의 `inner_volume_cm3 = 70.34` 는 단면 1,407 mm² × 폭 50 mm
의 **각기둥 추정**이고, 실제 형상은 y 로 뚫린 채널이다. bracket(y 0~46.7)·drive(y 2~37)는
+y 쪽에만 있어 −y 끝을 막지 않는다.

그래서 세 엔진(DEME·PBD·Newton MPM)의 포획 0 은 전부 맞는 답이었다.

이 파일이 하는 것
-----------------
각 셸의 y 양끝에 **끝벽 판**을 붙인 STL 을 만들고, **시뮬을 돌리기 전에 형상만으로**
밀폐 공동이 실제로 생기는지 검사한다. 공동이 안 생기면 시뮬을 돌릴 이유가 없다.

끝벽은 셸의 X–Z 투영 **볼록 껍질**을 y 방향으로 `wall_mm` 두께만큼 밀어낸 판이다.
셸과 함께 힌지 둘레로 돌므로 두 셸이 닫히면 양끝이 막힌 보울이 된다.

⚠️ 이것은 **타당성 시험용 변형본**이지 승인된 설계가 아니다. 실제 채택은 그랩 트랙
(D462/D463 계열)에서 판단할 일이며, 특히 +y 끝은 bracket/drive 와 간섭할 수 있다.
여기서는 −y 만 막는 변형과 양끝을 막는 변형을 둘 다 만들어 비교한다.

원본 `scoop_grab_v1_design.py` 와 `claudedocs/runtime_logs/scoop_grab_v1/*.stl` 은
읽기만 하고 수정하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from collections import deque
from pathlib import Path

import numpy as np

import sim_newton_scoop_probe as b1

OUT_DIR = Path("claudedocs/runtime_logs/newton_mpm/b1_scoop/endwall")
LIP_PIVOT_RADIUS_M = 38.332e-3
PIVOT_GAP_M = 26.0e-3
TRAVEL_DEG = 44.5


def convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Andrew monotone chain. 반시계 방향 껍질 정점을 돌려준다."""
    p = np.unique(np.round(pts, 9), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    if len(p) < 3:
        return p

    def half(points):
        out = []
        for q in points:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    lower = half(p)
    upper = half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def prism_z(hull_xy: np.ndarray, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """X–Y 볼록 다각형을 z0..z1 로 밀어낸 닫힌 삼각형 메시 (바닥판)."""
    n = len(hull_xy)
    v = np.zeros((2 * n, 3))
    v[:n, 0], v[:n, 1], v[:n, 2] = hull_xy[:, 0], hull_xy[:, 1], z0
    v[n:, 0], v[n:, 1], v[n:, 2] = hull_xy[:, 0], hull_xy[:, 1], z1
    f = []
    for i in range(1, n - 1):
        f += [[0, i + 1, i], [n, n + i, n + i + 1]]
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i]]
    return v, np.array(f, dtype=np.int32).reshape(-1)


def prism(hull_xz: np.ndarray, y0: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
    """X–Z 볼록 다각형을 y0..y1 로 밀어낸 닫힌 삼각형 메시."""
    n = len(hull_xz)
    v = np.zeros((2 * n, 3))
    v[:n, 0], v[:n, 2], v[:n, 1] = hull_xz[:, 0], hull_xz[:, 1], y0
    v[n:, 0], v[n:, 2], v[n:, 1] = hull_xz[:, 0], hull_xz[:, 1], y1
    f = []
    for i in range(1, n - 1):                      # 양 끝면 (볼록이므로 팬 삼각분할)
        f += [[0, i + 1, i], [n, n + i, n + i + 1]]
    for i in range(n):                             # 옆면
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i]]
    return v, np.array(f, dtype=np.int32).reshape(-1)


def build_variant(kind: str, wall_m: float) -> dict:
    """kind: none | minus_y | both. 끝벽을 붙인 셸 2매를 만든다."""
    out = {}
    for side, path in ((-1, b1.SHELL_L), (+1, b1.SHELL_R)):
        v, idx = b1.load_stl_mm(path)
        parts_v, parts_f, off = [v], [idx.copy()], len(v)
        if kind != "none":
            # 호가 X–Y 에 있으므로 관의 축은 Z. 막아야 할 곳은 **바닥(z_min)** 이다.
            hull = convex_hull_2d(np.stack([v[:, 0], v[:, 1]], axis=1))
            ends = [(v[:, 2].min() - wall_m, v[:, 2].min())]
            if kind == "both":
                ends.append((v[:, 2].max(), v[:, 2].max() + wall_m))
            for y0, y1 in ends:
                pv, pf = prism_z(hull, y0, y1)
                parts_v.append(pv)
                parts_f.append(pf + off)
                off += len(pv)
        out[side] = (np.vstack(parts_v), np.concatenate(parts_f).astype(np.int32))
    return out


def write_stl(path: Path, v: np.ndarray, idx: np.ndarray) -> None:
    F = idx.reshape(-1, 3)
    tri = v[F] * 1000.0                                     # m -> mm
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(F)))
        for k in range(len(F)):
            fh.write(struct.pack("<12f", *n[k], *tri[k, 0], *tri[k, 1], *tri[k, 2]))
            fh.write(b"\0\0")


def tri_samples(v: np.ndarray, idx: np.ndarray, n: int = 28) -> np.ndarray:
    F = idx.reshape(-1, 3)
    A, B, C = v[F[:, 0]], v[F[:, 1]], v[F[:, 2]]
    W = np.array([(i / n, j / n, 1 - i / n - j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    return (A[:, None, :] * W[None, :, 0, None]
            + B[:, None, :] * W[None, :, 1, None]
            + C[:, None, :] * W[None, :, 2, None]).reshape(-1, 3)


def cavity_cm3(shells: dict, frac: float, pitch: float = 0.0005) -> tuple[float, float, float]:
    """닫힌(또는 임의 개폐각) 형상이 가두는 부피. 바깥에서 flood fill 해 도달 못한 빈 복셀."""
    pts = []
    for side, (v, idx) in shells.items():
        piv = np.array([side * PIVOT_GAP_M / 2, 0.0, 0.0])   # 피벗은 X–Y 평면상 (±13, 0)
        a = side * math.radians(TRAVEL_DEG) * frac
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])   # 힌지축 Z (설계 D461 §3)
        pts.append(tri_samples(((v - piv) @ R.T) + piv, idx))
    P = np.vstack(pts)
    lo = P.min(0) - 6 * pitch
    hi = P.max(0) + 6 * pitch
    dims = np.ceil((hi - lo) / pitch).astype(int) + 1
    occ = np.zeros(dims, bool)
    ijk = np.floor((P - lo) / pitch).astype(int)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    free = ~occ
    vis = np.zeros_like(free)
    dq = deque()
    for face in (np.s_[0, :, :], np.s_[-1, :, :], np.s_[:, 0, :],
                 np.s_[:, -1, :], np.s_[:, :, 0], np.s_[:, :, -1]):
        m = np.zeros_like(free)
        m[face] = True
        for cell in np.argwhere(m & free):
            t = tuple(cell)
            if not vis[t]:
                vis[t] = True
                dq.append(t)
    nbrs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while dq:
        i, j, k = dq.popleft()
        for d in nbrs:
            a2, b2, c2 = i + d[0], j + d[1], k + d[2]
            if (0 <= a2 < dims[0] and 0 <= b2 < dims[1] and 0 <= c2 < dims[2]
                    and free[a2, b2, c2] and not vis[a2, b2, c2]):
                vis[a2, b2, c2] = True
                dq.append((a2, b2, c2))
    enc = free & ~vis
    if not enc.any():
        return 0.0, float("nan"), float("nan")
    w = np.argwhere(enc) * pitch + lo
    return enc.sum() * pitch**3 * 1e6, w[:, 2].min() * 1e3, w[:, 2].max() * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wall-mm", type=float, default=2.0, help="끝벽 두께 (설계 wall_mm=2.0)")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    rows = []
    print(f"{'변형':10s} {'각도':>7s} {'밀폐 공동':>12s} {'공동 z 범위 mm':>22s}")
    for kind in ("none", "floor", "both"):
        shells = build_variant(kind, a.wall_mm * 1e-3)
        if kind != "none":
            for side, tag in ((-1, "L"), (+1, "R")):
                write_stl(out_dir / f"shell_{tag}_{kind}.stl", *shells[side])
        for frac in (0.0, 0.5, 1.0):
            vol, zlo, zhi = cavity_cm3(shells, frac)
            rows.append({"variant": kind, "shell_angle_deg": TRAVEL_DEG * frac,
                         "cavity_cm3": vol, "cavity_z_lo_mm": zlo, "cavity_z_hi_mm": zhi,
                         "endwall_mm": a.wall_mm})
            print(f"{kind:10s} {TRAVEL_DEG*frac:6.1f}° {vol:10.2f} cm³   z[{zlo:7.1f},{zhi:7.1f}]")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "endwall_cavity_scan.json").write_text(json.dumps({
        "artifact": "GRAB_V1_ENDWALL_VARIANT_CAVITY",
        "method": ("셸 X–Z 투영 볼록껍질을 y 로 wall_mm 밀어낸 끝벽을 붙이고, 삼각형당 435점 "
                   "0.5mm 격자 래스터화 + 바깥 flood fill 로 밀폐 공동을 센다"),
        "design_claim_inner_volume_cm3": 70.34,
        "design_claim_is": "단면 1407mm^2 x 폭 50mm 각기둥 추정",
        "rows": rows,
        "non_claims": [
            "타당성 시험용 변형본이지 승인된 설계가 아니다. 채택 판단은 그랩 트랙 몫.",
            "+y 끝벽은 bracket(y 0~46.7)·drive(y 2~37)와 간섭할 수 있다. 여기서는 간섭을 보지 않았다.",
            "0.5mm flood fill 은 미세 틈에서 공동을 과소평가할 수 있다(과대는 아님).",
        ],
    }, indent=1, ensure_ascii=False))
    print(f"\nwrote {out_dir/'endwall_cavity_scan.json'}")
    print("설계 design.json inner_volume_cm3 = 70.34 (각기둥 추정)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
