#!/usr/bin/env python3
"""B1 재시험: 그랩 v1 이 Newton MPM 에서 실제로 재료를 담는가 (track P4f).

앞선 시도가 실패한 이유는 엔진이 아니라 내 하네스였다:
  * 채움 영역·보울 높이를 **하드코딩**해서 메시가 바뀌면 어긋났다
  * 그랩 STL 이 **watertight 가 아니다** (에지 91 개가 3면 이상 공유).
    설계가 D446 준수로 볼록 조각을 겹쳐 합집합한 결과다.

이 파일은 두 가지를 고친다:
  1. 채움/측정 좌표를 **메시 실측 extent 에서 유도**한다. 하드코딩 없음.
  2. `--mesh watertight` 로 복셀 리메시본을 쓸 수 있다 (원본과 bbox 0.8 mm 이내 일치,
     경계 에지 0, 3면 이상 공유 에지 0).

기준선은 `sim_newton_collider_cup_test.py` 가 낸 최소 재현 결과다: 복셀 5 mm 에서
벽 1.5 mm 컵이 98.8 %, 2.0 mm 가 99.0 % 를 담는다. 설계 `wall_mm = 2.0` 이므로
그랩도 담겨야 한다. 안 담기면 남는 용의자는 형상(watertight 여부)이다.

NOT MEASURED: 물성 미실측. 담긴 개수는 시뮬 재료의 값이지 실물 예측이 아니다.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import sim_newton_scoop_probe as b1
import sim_newton_mpm_probe as n1

OUT_DIR = Path("claudedocs/runtime_logs/newton_mpm/b1_scoop")
LIP_PIVOT_RADIUS_MM = 38.332      # design.json derived
PIVOT_GAP_MM = 26.0
TRAVEL_DEG = 44.5


def build_shells(mesh_kind: str, wt_dir: Path) -> dict:
    """셸 2매를 피벗 원점 로컬 좌표로 싣는다. 좌표는 메시에서 직접 읽는다."""
    if mesh_kind == "watertight":
        paths = {-1: wt_dir / "shell_L_wt.stl", +1: wt_dir / "shell_R_wt.stl"}
    else:
        paths = {-1: b1.SHELL_L, +1: b1.SHELL_R}
    gap = PIVOT_GAP_MM * 1e-3
    shells = {}
    for side, path in paths.items():
        v, idx = b1.load_stl_mm(path)
        z_pivot = math.sqrt((LIP_PIVOT_RADIUS_MM * 1e-3) ** 2 - (gap / 2) ** 2) + float(v[:, 2].min())
        pivot = np.array([side * gap / 2.0, 0.0, z_pivot])
        F = idx.reshape(-1, 3)
        E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        _, cnt = np.unique(E, axis=0, return_counts=True)
        shells[side] = {
            "v": v - pivot, "i": idx, "pivot": pivot,
            "bbox_lo": v.min(0), "bbox_hi": v.max(0),
            "n_tri": len(F), "boundary_edges": int((cnt == 1).sum()),
            "nonmanifold_edges": int((cnt > 2).sum()),
        }
    return shells


def angle(side: int, frac: float) -> float:
    """frac 1 = 열림 44.5도(립이 바깥으로), 0 = 닫힘."""
    return -side * math.radians(TRAVEL_DEG) * frac


def posed_extents(shells: dict, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """주어진 개폐각에서 두 셸을 합친 bbox (몸체 원점 기준)."""
    lo = np.full(3, 1e9)
    hi = np.full(3, -1e9)
    for side, sh in shells.items():
        a = angle(side, frac)
        c, s = math.cos(a), math.sin(a)
        vx = sh["v"][:, 0] * c + sh["v"][:, 2] * s
        vz = -sh["v"][:, 0] * s + sh["v"][:, 2] * c
        p = np.stack([vx + sh["pivot"][0], sh["v"][:, 1], vz], axis=1)
        lo = np.minimum(lo, p.min(0))
        hi = np.maximum(hi, p.max(0))
    return lo, hi


def run(mesh_kind: str, voxel_m: float, bowl_h: float, wt_dir: Path) -> dict:
    import warp as wp
    import newton
    from newton.solvers import SolverImplicitMPM

    t0 = time.time()
    shells = build_shells(mesh_kind, wt_dir)
    lo_open, hi_open = posed_extents(shells, 1.0)
    z_body = bowl_h - lo_open[2]              # 열린 상태 최하단을 bowl_h 로

    sp = voxel_m / 3.0
    rad = sp * 0.5
    # 채움: 그랩 실측 footprint 전체를 덮는 넓은 기둥을 위에서 떨어뜨린다.
    # 좌표를 메시에서 유도하므로 메시가 바뀌어도 어긋나지 않는다.
    fx = np.arange(lo_open[0], hi_open[0], sp)
    fy = np.arange(lo_open[1], hi_open[1], sp)
    z_fill0 = z_body + hi_open[2] + 0.005
    fz = np.arange(z_fill0, z_fill0 + 0.030, sp)
    P = np.array([[x, y, z] for x in fx for y in fy for z in fz])

    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    mass = sp**3 * 950.0 * n1.PACKING_FRACTION
    for p in P:
        builder.add_particle(wp.vec3(*p), wp.vec3(0.0, 0.0, 0.0), mass, radius=rad)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    bodies = {}
    for side, sh in shells.items():
        b = builder.add_body(
            xform=wp.transform(wp.vec3(float(sh["pivot"][0]), 0.0, float(z_body)), wp.quat_identity()),
            mass=0.0, is_kinematic=True)
        builder.add_shape_mesh(
            body=b, mesh=newton.Mesh(sh["v"], sh["i"], compute_inertia=False, is_solid=True),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0))
        bodies[side] = b

    model = builder.finalize()
    model.set_gravity(wp.vec3(0.0, 0.0, -9.81))
    opt = SolverImplicitMPM.Config()
    opt.voxel_size = voxel_m
    opt.collider_velocity_mode = "backward"
    model.mpm.friction.fill_(0.68)
    s0, s1 = model.state(), model.state()
    solver = SolverImplicitMPM(model, config=opt)
    solver.setup_collider(body_mass=wp.zeros_like(model.body_mass), body_q=s0.body_q)
    nb = model.body_count

    def pose(zb: float, frac: float) -> None:
        q = np.zeros((nb, 7), dtype=np.float32)
        for side, sh in shells.items():
            a = angle(side, frac)
            q[bodies[side], 0:3] = (sh["pivot"][0], sh["pivot"][1], zb)
            q[bodies[side], 3:7] = (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
        s0.body_q.assign(q)
        s1.body_q.assign(q)

    def step(n: int) -> None:
        nonlocal s0, s1
        for _ in range(n):
            solver.step(s0, s1, None, None, 1 / 60.0)
            solver.project_outside(s1, s1, 1 / 60.0)
            s0, s1 = s1, s0

    def in_footprint(q: np.ndarray) -> np.ndarray:
        return ((q[:, 0] > lo_open[0]) & (q[:, 0] < hi_open[0])
                & (q[:, 1] > lo_open[1]) & (q[:, 1] < hi_open[1]))

    pose(z_body, 1.0)
    step(120)                                   # 2.0 s 담기
    q = s0.particle_q.numpy().astype(np.float64)
    in_bowl = int(((q[:, 2] > bowl_h) & in_footprint(q)).sum())

    for k in range(42):                          # 0.7 s 폐합
        pose(z_body, 1.0 - (k + 1) / 42)
        step(1)
    for k in range(36):                          # 0.6 s 리프트 +80 mm
        pose(z_body + 0.080 * (k + 1) / 36, 0.0)
        step(1)
    pose(z_body + 0.080, 0.0)
    step(36)

    q = s0.particle_q.numpy().astype(np.float64)
    carried = int((q[:, 2] > bowl_h + 0.040).sum())
    pellet_bulk = n1.PELLET_VOLUME_M3 / n1.PACKING_FRACTION
    return {
        "mesh": mesh_kind,
        "voxel_mm": voxel_m * 1e3,
        "bowl_height_mm": bowl_h * 1e3,
        "n_particles": int(P.shape[0]),
        "shell_tri": shells[-1]["n_tri"],
        "shell_nonmanifold_edges": shells[-1]["nonmanifold_edges"],
        "shell_boundary_edges": shells[-1]["boundary_edges"],
        "in_bowl_before_close": in_bowl,
        "carried_after_lift": carried,
        "retention_pct": 100.0 * carried / max(in_bowl, 1),
        "captured_pellet_equivalents": carried * sp**3 / pellet_bulk,
        "wall_s": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", choices=["original", "watertight"], default="original")
    ap.add_argument("--voxel-mm", type=float, default=5.0)
    ap.add_argument("--bowl-mm", type=float, default=40.0)
    ap.add_argument("--wt-dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    r = run(a.mesh, a.voxel_mm * 1e-3, a.bowl_mm * 1e-3, Path(a.wt_dir))
    print(
        f"[grab] mesh={r['mesh']:10s} vox={r['voxel_mm']:4.1f}mm tri={r['shell_tri']:6d} "
        f"nonmanifold={r['shell_nonmanifold_edges']:4d} | 담김 {r['in_bowl_before_close']:6d} "
        f"-> 운반 {r['carried_after_lift']:6d} ({r['retention_pct']:5.1f}% 유지) "
        f"= 펠릿 {r['captured_pellet_equivalents']:6.0f}개 | {r['wall_s']:.1f}s",
        flush=True)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
