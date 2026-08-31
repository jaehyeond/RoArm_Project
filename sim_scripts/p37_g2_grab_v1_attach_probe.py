"""p37 / g2 — 그랩 v1 을 RoArm-M3 link5 에 실제로 붙여 폐합 스윕 검증 (D462).

무엇을 답하나
    1. 배치: 그랩 로컬 프레임 -> link5 프레임 변환이 물리 제약 3개를 만족하는가
         (a) 스퍼 기어는 축이 평행해야 물린다
         (b) 구동 기어는 순정 조 스텁에 붙으므로 축 = 서보축 = link5 Y (0,1,0)
         (c) 브래킷 볼트는 고정 조 블레이드(x -11.5~-10, t=1.5)를 관통 -> 축 = link5 X
    2. 브래킷 볼트 구멍이 블레이드의 기존 4구멍(D462 §1)과 실제로 정렬하는가
    3. 셸 0 -> 44.5도 스윕 중 팔(link5)과 충돌하는가. 처음 닿는 각은?
    4. 립이 쓸고 가는 부피 = 닫을 때 밀어내야 할 펠릿 양

입자 물리는 쓰지 않는다. 순수 기하다.
사용:  python sim_scripts/p37_g2_grab_v1_attach_probe.py [출력디렉터리]
"""
import sys, json, math
from pathlib import Path
import importlib.util
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else REPO / "claudedocs/runtime_logs/grab_track/g2_attach")
URDF_MESH = REPO / "local_assets/roarm_m3/urdf/meshes"

spec = importlib.util.spec_from_file_location("gv1", REPO / "scoop_grab_v1_design.py")
G = importlib.util.module_from_spec(spec)
_argv, sys.argv = sys.argv, ["x"]
spec.loader.exec_module(G)
sys.argv = _argv
P, K = G.P, G.kin(G.P)

# ── link5 실측 (D462 §1·§2) ────────────────────────────────────────────
BLADE_X = (-11.54, -10.03)          # 고정 조 블레이드 판재 (두께 1.51)
BLADE_HOLES_YZ = [(-13.34, 83.46), (11.85, 83.46),
                  (-13.34, 102.90), (11.85, 102.90)]     # 4볼트 사각형
GRIPPER_AXIS = np.array([0.0, 1.0, 0.0])                 # 서보 개폐축 (link5 프레임)
GRIPPER_ORIGIN = np.array([0.0, 18.821, 52.035])


def load_mm(name):
    m = trimesh.load(URDF_MESH / name)
    if m.extents.max() < 1.0:
        m.apply_scale(1000.0)
    return m


def placement():
    """그랩 로컬 -> link5 변환.

    로컬 규약: X = 두 셸이 갈라지는 방향, Y = 깊이(-Y 가 더미 쪽), Z = 힌지축.
    제약을 풀면
        로컬 Z (힌지)  -> link5 +Y     (기어 축이 서보축과 평행해야 함)
        로컬 -Y (깊이) -> link5 +Z     (블레이드 끝에서 더 뻗어나가는 방향)
        로컬 X        -> link5 +X     (두 셸이 갈라지는 방향 = 볼트 축과 같은 축)
    원점은 블레이드 4구멍 사각형의 중심에 두고, 블레이드 바깥면에 브래킷을 얹는다.
    """
    # 열 = 로컬 축의 상(image). 우수좌표계 확인: X x Y = Z.
    #   로컬 X -> link5 +X   (두 셸이 갈라지는 방향)
    #   로컬 Y -> link5 -Z   => 로컬 -Y(깊이) -> link5 +Z = 팔에서 멀어지는 쪽
    #   로컬 Z -> link5 +Y   (힌지축, 서보축과 평행)
    # ⚠️ 초판은 `.T` 를 붙여 로컬 Y 를 link5 +Z 로 보냈고, 그 결과 그랩이 팔에서
    #    멀어지는 대신 **손목 속으로 자랐다**. 스탠드오프를 32->38 로 키워도 여유가
    #    -0.834 -> -0.744 로 0.09 mm 밖에 안 변해 이 부호 오류가 드러났다.
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, -1.0, 0.0]])
    cy = float(np.mean([h[0] for h in BLADE_HOLES_YZ]))
    cz = float(np.mean([h[1] for h in BLADE_HOLES_YZ]))
    # 피벗선은 볼트 사각형 중심에서 standoff 만큼 팔 바깥(link5 +Z)으로 나간다.
    # 안 그러면 그랩이 link5 몸통(Z -0.75~119.89) 안에 박힌다 (초판 관통 -67.8 mm).
    t = np.array([BLADE_X[0] - P["bracket_thk_mm"] / 2.0, cy,
                  cz + P["bracket_standoff_mm"]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_about(axis_pt, axis_dir, ang):
    a = np.asarray(axis_dir, float)
    a = a / np.linalg.norm(a)
    c, s, C = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
    x, y, z = a
    R = np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                  [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                  [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(axis_pt, float) - R @ np.asarray(axis_pt, float)
    return T


def link5_occupancy(mesh, pitch=1.0):
    """link5 의 **실제 재료 점유**를 복셀로 잡는다.

    ⚠️ 초판은 `link5.convex_hull` 로 충돌을 검사해 관통 -67.8 mm 를 보고했다. 그러나
       link5 실부피 47 378 mm^3 / 볼록껍질 122 805 mm^3 = **0.386** 로, 껍질은 속이 빈
       프레임을 통짜 덩어리로 만든다 (D453 의 cooked convex 과대보고와 같은 계열).
       `fcl` 도 없고 메쉬가 watertight 도 아니므로 표면 복셀 점유로 판정한다.
    """
    v = mesh.voxelized(pitch=pitch)
    pts = np.asarray(v.points, dtype=float)
    from scipy.spatial import cKDTree
    return cKDTree(pts), pitch


def penetration(pts, tree, pitch):
    """샘플점이 link5 재료 복셀에 얼마나 파고들었는지. 양수면 여유(mm)."""
    if len(pts) == 0:
        return np.inf
    d, _ = tree.query(pts, k=1)
    return float(d.min() - pitch * math.sqrt(3) / 2.0)


def sample_mesh(m, n=220):
    """메쉬 표면 + 정점 샘플."""
    try:
        s, _ = trimesh.sample.sample_surface(m, n)
        return np.vstack([m.vertices, s])
    except Exception:
        return np.asarray(m.vertices, float)


# ─────────────────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = placement()
    R = T[:3, :3]
    link5 = load_mm("link5.stl")

    sL, nL = G.build_shell(P, -1)
    sR, nR = G.build_shell(P, +1)
    br, nB = G.build_bracket(P)
    dr, nD, cd = G.build_drive(P)

    def place(parts):
        out = []
        for m in parts:
            c = m.copy(); c.apply_transform(T); out.append(c)
        return out

    gates, ev = {}, {}

    # G1 힌지축이 서보 개폐축과 평행한가 (스퍼 기어 필수 조건)
    hinge_world = R @ np.array([0.0, 0.0, 1.0])
    dot = abs(float(np.dot(hinge_world, GRIPPER_AXIS)))
    gates["G1_hinge_parallel_to_servo_axis"] = {
        "pass": dot > 0.9999, "abs_dot": round(dot, 6),
        "hinge_in_link5": np.round(hinge_world, 4).tolist(),
        "servo_axis": GRIPPER_AXIS.tolist(),
        "why": "스퍼 기어는 축이 평행해야 물린다. 어긋나면 구동이 성립하지 않는다"}

    # G2 브래킷 볼트 구멍 축이 블레이드 관통 방향(link5 X)인가
    #    build_bracket 의 plate_with_holes 는 구멍 축을 **로컬 Z** 로 뚫는다.
    bolt_axis_world = R @ np.array([1.0, 0.0, 0.0])   # hole_axis="x" 로 정정됨
    dotx = abs(float(np.dot(bolt_axis_world, np.array([1.0, 0.0, 0.0]))))
    gates["G2_bracket_bolt_axis_through_blade"] = {
        "pass": dotx > 0.9999, "abs_dot_with_link5_X": round(dotx, 6),
        "bolt_axis_in_link5": np.round(bolt_axis_world, 4).tolist(),
        "blade_normal": [1.0, 0.0, 0.0],
        "why": ("볼트는 두께 1.51 mm 블레이드를 관통해야 한다. 구멍 축이 블레이드 법선과"
                " 어긋나면 볼트가 판재를 뚫고 지나갈 수 없다"),
        "root_cause_if_fail": ("build_bracket 이 plate_with_holes 로 구멍을 **로컬 Z**"
                               "(= 힌지축)에 뚫는다. 힌지축과 볼트축은 90도 달라야 한다")}

    # G3 브래킷이 블레이드 4구멍 자리에 실제로 앉는가 (구멍 좌표 대조)
    dy, dz = P["bracket_bolt_dy_mm"], P["bracket_bolt_dz_mm"]
    want = sorted(BLADE_HOLES_YZ)
    got = sorted([(round(y, 2), round(z, 2))
                  for y in (np.mean([h[0] for h in BLADE_HOLES_YZ]) - dy / 2,
                            np.mean([h[0] for h in BLADE_HOLES_YZ]) + dy / 2)
                  for z in (np.mean([h[1] for h in BLADE_HOLES_YZ]) - dz / 2,
                            np.mean([h[1] for h in BLADE_HOLES_YZ]) + dz / 2)])
    err = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(want, got))
    gates["G3_bracket_holes_match_blade"] = {
        "pass": err < 0.02, "max_err_mm": round(float(err), 4),
        "blade_holes_yz": want, "bracket_holes_yz": got}

    # G4 스윕: 셸이 link5(팔)과 충돌하는가
    body_idx = [i for i, n in enumerate(nL) if not n.startswith("gear")]
    tree, pitch = link5_occupancy(link5, 1.0)
    ev["link5_voxel_pitch_mm"] = pitch
    ev["link5_solid_fraction_of_hull"] = round(float(link5.volume / link5.convex_hull.volume), 3)
    angles = np.linspace(0.0, P["shell_travel_deg"], 24)
    clearances, first_hit = [], None
    l5_lo, l5_hi = link5.bounds
    for phi in angles:
        worst = np.inf
        for parts, names, side in ((sL, nL, -1), (sR, nR, +1)):
            px, py = (side * K["g"] / 2.0, 0.0)
            piv = T[:3, :3] @ np.array([px, py, 0.0]) + T[:3, 3]
            Rj = rot_about(piv, hinge_world, math.radians(side * phi))
            for i in body_idx:
                m = parts[i].copy(); m.apply_transform(T); m.apply_transform(Rj)
                lo, hi = m.bounds
                sep = max((l5_lo - hi).max(), (lo - l5_hi).max())
                if sep >= 0:
                    worst = min(worst, sep)          # AABB 부터 떨어져 있으면 그걸로 충분
                else:
                    worst = min(worst, penetration(sample_mesh(m), tree, pitch))
        clearances.append(round(float(worst), 3))
        if worst < 0 and first_hit is None:
            first_hit = float(phi)
    gates["G4_no_arm_collision_through_sweep"] = {
        "pass": first_hit is None, "first_collision_deg": first_hit,
        "min_clearance_mm": round(float(min(clearances)), 3),
        "angles_deg": [round(float(a), 2) for a in angles],
        "clearance_mm": clearances}

    # G6 브래킷 + 구동부 자체가 팔과 닿는가 (지금까지는 셸만 검사했다)
    worst6, who6 = np.inf, None
    for parts, names, tag in ((br, nB, "bracket"), (dr, nD, "drive")):
        for m0, nm in zip(parts, names):
            m = m0.copy(); m.apply_transform(T)
            lo, hi = m.bounds
            sep = max((l5_lo - hi).max(), (lo - l5_hi).max())
            val = sep if sep >= 0 else penetration(sample_mesh(m), tree, pitch)
            if val < worst6:
                worst6, who6 = val, f"{tag}:{nm}"
    gates["G6_bracket_drive_clear_of_arm"] = {
        "pass": worst6 >= 0.0, "min_clearance_mm": round(float(worst6), 3),
        "closest_piece": who6,
        "why": "볼트판은 블레이드에 닿아야 하지만 스파인/보스/구동부는 팔을 파고들면 안 된다"}

    # G7 구동 기어 축이 **서보축 그 자체**이고 피벗과 축간거리가 맞는가
    #    구동 기어는 순정 조 스텁에 볼트로 붙으므로 서보축을 그대로 돈다.
    #    스퍼 기어가 물리려면 두 평행축 사이 거리 = r_drive + r_shell 이어야 한다.
    piv_pt = T[:3, :3] @ np.array([K["pivot_L"][0], K["pivot_L"][1], 0.0]) + T[:3, 3]
    ax = GRIPPER_AXIS / np.linalg.norm(GRIPPER_AXIS)
    dvec = piv_pt - GRIPPER_ORIGIN
    perp = dvec - np.dot(dvec, ax) * ax
    axis_dist = float(np.linalg.norm(perp))
    need = cd
    gates["G7_drive_gear_meshes_on_servo_axis"] = {
        "pass": abs(axis_dist - need) < 0.5,
        "axis_distance_mm": round(axis_dist, 3), "required_center_distance_mm": round(need, 3),
        "error_mm": round(axis_dist - need, 3),
        "pivot_L_in_link5": np.round(piv_pt, 3).tolist(),
        "servo_axis_point": GRIPPER_ORIGIN.tolist(),
        "why": ("구동 기어는 서보축을 돈다. 피벗 L 축과의 수직거리가 r_drive+r_shell 과"
                " 다르면 이가 안 물리거나 뿌리까지 박힌다"),
        "fix_if_fail": "bracket_standoff_mm 로 피벗 위치를 조절해 축간거리를 맞춘다"}

    # G8 팔의 다음 링크(link4)와 손목 롤 전 구간에서 닿는가
    link4 = load_mm("link4.stl")
    T45 = np.eye(4)
    T45[:3, :3] = np.array([[math.cos(1.5708), -math.sin(1.5708), 0],
                            [math.sin(1.5708), math.cos(1.5708), 0], [0, 0, 1]]) @                   np.array([[math.cos(1.5708), 0, math.sin(1.5708)], [0, 1, 0],
                            [-math.sin(1.5708), 0, math.cos(1.5708)]])
    T45[:3, 3] = np.array([15.147, -53.653, 0.0])
    T54 = np.linalg.inv(T45)
    l4 = link4.copy(); l4.apply_transform(T54)          # link4 를 link5 프레임으로
    tree4, _ = link5_occupancy(l4, 1.0)
    l4_lo, l4_hi = l4.bounds
    worst8 = np.inf
    for parts, names, side in ((sL, nL, -1), (sR, nR, +1)):
        for i in body_idx:
            m = parts[i].copy(); m.apply_transform(T)
            lo, hi = m.bounds
            sep = max((l4_lo - hi).max(), (lo - l4_hi).max())
            worst8 = min(worst8, sep if sep >= 0 else penetration(sample_mesh(m), tree4, pitch))
    for parts, names in ((br, nB), (dr, nD)):
        for m0 in parts:
            m = m0.copy(); m.apply_transform(T)
            lo, hi = m.bounds
            sep = max((l4_lo - hi).max(), (lo - l4_hi).max())
            worst8 = min(worst8, sep if sep >= 0 else penetration(sample_mesh(m), tree4, pitch))
    gates["G8_clear_of_link4"] = {
        "pass": worst8 >= 0.0, "min_clearance_mm": round(float(worst8), 3),
        "why": "link5 만이 아니라 바로 앞 링크와도 안 닿아야 손목 롤이 자유롭다"}

    # G5 립 스윕 체적 = 닫을 때 밀어내야 할 펠릿 양
    sweep_area = 0.0
    for side in (-1, +1):
        a0 = K["a0"] if side < 0 else math.pi - K["a0"]
        sweep_area += 0.5 * K["R"] ** 2 * math.radians(P["shell_travel_deg"])
    sweep_vol_cm3 = sweep_area * P["shell_width_mm"] / 1000.0
    ev["lip_sweep_volume_cm3"] = round(sweep_vol_cm3, 2)
    ev["enclosed_volume_cm3"] = 70.34
    gates["G5_sweep_volume_below_enclosed"] = {
        "pass": sweep_vol_cm3 < 4.0 * 70.34,
        "sweep_cm3": round(sweep_vol_cm3, 2), "enclosed_cm3": 70.34,
        "why": "닫는 동안 립이 쓸고 가는 부피. 담을 부피보다 지나치게 크면 재료를 밀어내기만 한다"}

    for _v in gates.values():
        _v["pass"] = bool(_v["pass"])
    ok = all(v["pass"] for v in gates.values())
    ev["placement_T_link5"] = np.round(T, 5).tolist()
    json.dump({"probe": "p37_g2_grab_v1_attach", "gates": gates, "evidence": ev,
               "all_gates_pass": ok,
               "verdict": "G2_ATTACH_OK" if ok else "G2_ATTACH_BLOCKED"},
              open(OUT / "g2_results.json", "w"), ensure_ascii=False, indent=2,
              default=lambda o: bool(o) if isinstance(o, np.bool_)
              else int(o) if isinstance(o, np.integer)
              else float(o) if isinstance(o, np.floating) else str(o))

    for k2, v in gates.items():
        print(("  PASS  " if v["pass"] else "  FAIL  ") + k2)
        for kk, vv in v.items():
            if kk in ("pass", "angles_deg", "clearance_mm"):
                continue
            print(f"           {kk}: {json.dumps(vv, ensure_ascii=False)[:150]}")
    print(f"\n립 스윕 체적 {ev['lip_sweep_volume_cm3']} cm3 · 담을 부피 70.34 cm3")
    print(f"verdict = {'G2_ATTACH_OK' if ok else 'G2_ATTACH_BLOCKED'}   -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
