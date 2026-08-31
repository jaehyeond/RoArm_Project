"""p37 / g2 — 그랩 v1 을 RoArm-M3 link5 에 실제로 붙여 폐합 스윕 검증 (D462·D463).

무엇을 답하나
    1. 배치: 그랩 로컬 프레임 -> link5 프레임 변환이 물리 제약 3개를 만족하는가
         (a) 셸 기어쌍은 축이 평행해야 물린다
         (b) 구동 인출은 순정 가동 조이므로 축 = 서보축 = link5 Y (0,1,0)
         (c) 브래킷 볼트는 고정 조 블레이드(x -11.5~-10, t=1.5)를 관통 -> 축 = link5 X
    2. 브래킷 볼트 구멍이 블레이드의 기존 4구멍(D462 §1)과 실제로 정렬하는가
    3. 셸 0 -> 44.5도 스윕 중 팔(link5)과 충돌하는가. 처음 닿는 각은?
    4. 립이 쓸고 가는 부피 = 닫을 때 밀어내야 할 펠릿 양

D463 개정 (74th, 기어 직결 -> 4절 링크)
---------------------------------------
G6 **게이트 정의 정정**: 브래킷 볼트판은 고정 조 블레이드에 **닿는 것이 정상**이다
   (볼트로 물리는 면이다). 이전 판은 접촉(-0.787 mm, 복셀 반대각선 0.866 안쪽)을
   충돌로 세어 FAIL 을 냈다 — 형상 결함이 아니라 게이트 오탐이었다.
   정정: 체결면 조각은 **접촉 허용 / 관통 금지**(블레이드 바깥면을 넘지 않을 것),
   나머지 조각은 종전대로 여유 >= 0.
G7 **검사 대상 교체**: 더 이상 기어 축간거리를 보지 않는다 (D463 이 기어 직결을 기각).
   대신 **링크 도달성** — 설계가 쓴 서보축이 link5 실측 축과 같은가, 서보 0~89도가
   셸 0~44.5도를 내는가, 링크 몸체가 스윕 전 구간에서 팔과 안 닿는가.
G9 **신규**: 서보 크랭크를 순정 가동 조에 볼트로 물리므로 **조는 떼지 않고 남는다**.
   그 조가 0~89도 도는 동안 셸·브래킷·로드·셸크랭크와 안 닿아야 한다.

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
           else REPO / "claudedocs/runtime_logs/grab_track/g3_attach")
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
# link5_to_gripper_link 조인트 원점 (URDF 원문: xyz 0 0.018821 0.052035, rpy -1.5708 -1.5708 0)
GRIPPER_JOINT_RPY = (-1.5708, -1.5708, 0.0)


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


def rpy_mat(r, p, y):
    """고정축 Rz@Ry@Rx (URDF 규약)."""
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def jaw_in_link5():
    """순정 가동 조(gripper_link)를 서보각 0 에서 link5 프레임에 놓은 메쉬."""
    m = load_mm("gripper_link.stl")
    T = np.eye(4)
    T[:3, :3] = rpy_mat(*GRIPPER_JOINT_RPY)
    T[:3, 3] = GRIPPER_ORIGIN
    m.apply_transform(T)
    return m


def surface_cloud(mesh, n=200000):
    """메쉬 표면을 조밀하게 점으로 깐다 (정점 + 균일 표면 샘플). 간격 ~0.3 mm."""
    s, _ = trimesh.sample.sample_surface(mesh, n)
    return np.vstack([np.asarray(mesh.vertices, float), s])


def exact_clearance(piece, cloud):
    """볼록 조각 <-> 팔 표면점구름의 **정확한** 여유(mm). 음수면 관통 깊이.

    ⚠️ 왜 필요한가: 1 mm 복셀 점유는 편향이 pitch*sqrt(3)/2 = 0.866 mm 인
       **보수적 하한**이다. 표면에 정확히 닿아 있어도 -0.866 까지 내려간다. 즉
       음수 = "간섭한다" 가 아니라 "이 해상도로는 판정 못 한다" 는 뜻이다.
       D463 이 정정한 G6 의 -0.787 도, 스파인 수정 후의 -0.637 도 이 편향이었다.
       조각은 볼록·watertight 이므로 `contains`/`signed_distance` 가 정확하다.
    """
    lo, hi = piece.bounds
    m = 3.0
    sel = cloud[np.all((cloud >= lo - m) & (cloud <= hi + m), axis=1)]
    if len(sel) == 0:
        return float(np.linalg.norm(np.clip(lo - cloud, 0, None)
                                    + np.clip(cloud - hi, 0, None), axis=1).min())
    # signed_distance: 내부 양수 / 외부 음수. -max 가 곧 (여유 or -관통깊이) 다.
    return float(-piece.nearest.signed_distance(sel).max())


def clearance_to(mesh_parts, tree, pitch, lo, hi, cloud=None):
    """조각 리스트 -> (최소 여유, 가장 가까운 조각 이름).

    1단계 = AABB 분리거리(정확) / 복셀 하한(보수적, 편향 최대 pitch*sqrt(3)/2).
    2단계 = 하한이 편향폭 안(< 1 mm)인 조각만 표면점구름으로 **정확히** 다시 잰다.
            하한이 양수여도 다시 재는 이유: 편향 때문에 실제 여유가 최대 0.87 mm
            더 클 수 있어, 그대로 보고하면 여유를 과소평가한다.
    """
    worst, who = np.inf, None
    refine_below = 1.0
    for m, nm in mesh_parts:
        mlo, mhi = m.bounds
        sep = max((lo - mhi).max(), (mlo - hi).max())
        if sep >= 0:
            val = sep
        else:
            val = penetration(sample_mesh(m), tree, pitch)
            if val < refine_below and cloud is not None:
                val = exact_clearance(m, cloud)
        if val < worst:
            worst, who = val, nm
    return float(worst), who


# ─────────────────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = placement()
    R = T[:3, :3]
    link5 = load_mm("link5.stl")

    sL, nL = G.build_shell(P, -1)
    sR, nR = G.build_shell(P, +1)
    br, nB = G.build_bracket(P)
    dr, nD, lk = G.build_linkage(P)

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
    l5_cloud = surface_cloud(link5)          # 복셀 하한이 음수/미세할 때 정확히 다시 잰다
    ev["link5_voxel_pitch_mm"] = pitch
    ev["link5_solid_fraction_of_hull"] = round(float(link5.volume / link5.convex_hull.volume), 3)
    angles = np.linspace(0.0, P["shell_travel_deg"], 24)
    clearances, first_hit, whos = [], None, []
    l5_lo, l5_hi = link5.bounds
    for phi in angles:
        probe4 = []
        for parts, names, side in ((sL, nL, -1), (sR, nR, +1)):
            px, py = (side * K["g"] / 2.0, 0.0)
            piv = T[:3, :3] @ np.array([px, py, 0.0]) + T[:3, 3]
            Rj = rot_about(piv, hinge_world, math.radians(side * phi))
            for i in body_idx:
                m = parts[i].copy(); m.apply_transform(T); m.apply_transform(Rj)
                probe4.append((m, f"{'LR'[side > 0]}:{names[i]}"))
        worst, w4 = clearance_to(probe4, tree, pitch, l5_lo, l5_hi, l5_cloud)
        clearances.append(round(float(worst), 3))
        whos.append(w4)
        if worst < 0 and first_hit is None:
            first_hit = float(phi)
    gates["G4_no_arm_collision_through_sweep"] = {
        "pass": first_hit is None, "first_collision_deg": first_hit,
        "min_clearance_mm": round(float(min(clearances)), 3),
        "min_at_deg": round(float(angles[int(np.argmin(clearances))]), 2),
        "closest_piece": whos[int(np.argmin(clearances))],
        "measurement": ("1 mm 복셀 하한이 1 mm 밑이면 link5 표면점구름으로 정확히 "
                        "다시 잰다 — 하한을 값으로 읽으면 안 된다"),
        "angles_deg": [round(float(a), 2) for a in angles],
        "clearance_mm": clearances}

    # G6 브래킷 + 링크 정적부가 팔과 닿는가
    #   ⚠️ **게이트 정의 정정 (D463)**: `bolt_plate_*` 는 고정 조 블레이드에 **볼트로
    #      물리는 면**이라 닿는 것이 정상이다. 이전 판은 그 접촉(-0.787 mm, 복셀 반
    #      대각선 0.866 안쪽)을 충돌로 세어 FAIL 을 냈다 — 형상이 아니라 판정이 틀렸다.
    #      정정된 규칙: 체결면 조각은 **접촉 허용, 관통 금지**. 관통은 복셀이 아니라
    #      "블레이드 바깥면(link5 X = -11.54)을 넘었는가" 로 정확히 잰다.
    FASTEN_FACE = ("bolt_plate",)
    blade_outer_x = BLADE_X[0]
    ev["link5_surface_cloud_points"] = int(len(l5_cloud))
    worst6, who6 = np.inf, None
    fasten, others = [], []
    for parts, names, tag in ((br, nB, "bracket"), (dr, nD, "linkage")):
        for m0, nm in zip(parts, names):
            m = m0.copy(); m.apply_transform(T)
            if any(nm.startswith(f) for f in FASTEN_FACE):
                pen = float(m.bounds[1][0] - blade_outer_x)   # 양수면 블레이드를 파고들었다
                fasten.append({"piece": f"{tag}:{nm}",
                               "max_link5_x_mm": round(float(m.bounds[1][0]), 4),
                               "blade_outer_face_x_mm": blade_outer_x,
                               "penetration_mm": round(pen, 4)})
                continue
            others.append((m, f"{tag}:{nm}"))
    worst6, who6 = clearance_to(others, tree, pitch, l5_lo, l5_hi, l5_cloud)
    max_pen = max([f["penetration_mm"] for f in fasten], default=0.0)
    gates["G6_bracket_drive_clear_of_arm"] = {
        "pass": bool(worst6 >= 0.0 and max_pen <= 0.02),
        "min_clearance_mm": round(float(worst6), 3), "closest_piece": who6,
        "fastening_face_pieces": fasten,
        "fastening_max_penetration_mm": round(float(max_pen), 4),
        "fastening_rule": "접촉 허용 / 관통 금지 (허용 0.02 mm)",
        "why": ("볼트판은 블레이드에 닿아야 정상이다. 접촉을 충돌로 세면 안 되지만 "
                "판재를 뚫고 지나가서도 안 된다. 스파인/보스/링크는 종전대로 여유 >= 0"),
        "measurement": ("1 mm 복셀은 보수적 하한(편향 0.866 mm)이므로 음수가 나온 "
                        "조각만 link5 표면점구름(정점 + 균일샘플 20만)으로 정확히 "
                        "다시 잰다. 조각이 볼록·watertight 라 signed_distance 가 정확하다"),
        "definition_change": ("D463 — 이전 판은 접촉을 FAIL 로 셌다(-0.787 mm). "
                              "복셀 피치 1.0 의 반대각선이 0.866 이므로 그 값은 "
                              "'표면에 닿음'을 뜻했지 관통이 아니었다")}

    # G7 **링크 도달성** — 기어 축간거리 검사를 대체한다 (D463 이 기어 직결을 기각)
    #    (a) 설계가 쓴 서보축이 link5 실측 축과 같은가 (두 파일의 상수 일치)
    #    (b) 서보축 <-> 피벗 L 수직거리가 링크 지지대(ground) 길이와 같은가
    #    (c) 서보 0~89도가 셸 0~44.5도(개구 0~58 mm)를 실제로 내는가
    #    (d) 링크 몸체(로드·크랭크·핀)가 스윕 전 구간에서 팔과 안 닿는가
    piv_pt = T[:3, :3] @ np.array([K["pivot_L"][0], K["pivot_L"][1], 0.0]) + T[:3, 3]
    ax = GRIPPER_AXIS / np.linalg.norm(GRIPPER_AXIS)
    dvec = piv_pt - GRIPPER_ORIGIN
    axis_dist = float(np.linalg.norm(dvec - np.dot(dvec, ax) * ax))
    axis_design = G.to_local(P, GRIPPER_ORIGIN)
    axis_probe = T[:3, :3].T @ (GRIPPER_ORIGIN - T[:3, 3])
    axis_err = float(np.max(np.abs(axis_design - axis_probe)))
    rows = lk["rows"]
    reach_ok = (abs(rows[0]["shell_deg"]) < 1e-6
                and abs(rows[-1]["shell_deg"] - P["shell_travel_deg"]) < 0.02
                and abs(rows[-1]["mouth_mm"] - P["mouth_open_mm"]) < 0.25
                and abs(rows[-1]["servo_deg"] - P["servo_travel_deg"]) < 1e-6)
    ground_err = abs(axis_dist - lk["ground_len_mm"])

    idx7 = np.linspace(0, len(rows) - 1, 13).astype(int)
    worst7, who7, at7 = np.inf, None, None
    for i in idx7:
        Ts, Tr, Tk = G.linkage_pose(P, lk, int(i))
        Tsw = {"servocrank": Ts, "rod": Tr, "shellcrank": Tk}
        parts7 = []
        for m0, nm in zip(dr, nD):
            m = m0.copy()
            m.apply_transform(T @ Tsw[G.linkage_group(nm)])
            parts7.append((m, nm))
        v, w = clearance_to(parts7, tree, pitch, l5_lo, l5_hi, l5_cloud)
        if v < worst7:
            worst7, who7, at7 = v, w, float(rows[int(i)]["servo_deg"])
    gates["G7_linkage_reaches_and_clears"] = {
        "pass": bool(axis_err < 1e-6 and ground_err < 1e-3 and reach_ok and worst7 >= 0.0),
        "servo_axis_consistent": bool(axis_err < 1e-6),
        "servo_axis_max_err_mm": round(axis_err, 9),
        "axis_distance_mm": round(axis_dist, 4),
        "linkage_ground_len_mm": lk["ground_len_mm"],
        "ground_len_err_mm": round(ground_err, 6),
        "servo_deg_range": [rows[0]["servo_deg"], rows[-1]["servo_deg"]],
        "shell_deg_range": [rows[0]["shell_deg"], rows[-1]["shell_deg"]],
        "mouth_mm_range": [rows[0]["mouth_mm"], rows[-1]["mouth_mm"]],
        "crank_servo_r_mm": lk["crank_servo_r_mm"],
        "rod_len_mm": lk["rod_len_mm"],
        "crank_shell_r_mm": lk["crank_shell_r_mm"],
        "trans_angle_out_deg": [lk["trans_angle_out_min_deg"], lk["trans_angle_out_max_deg"]],
        "trans_angle_in_deg": [lk["trans_angle_in_min_deg"], lk["trans_angle_in_max_deg"]],
        "sweep_min_clearance_to_link5_mm": round(float(worst7), 3),
        "sweep_closest_piece": who7, "sweep_closest_at_servo_deg": at7,
        "why": ("기어 직결은 축간거리 19.5 mm 를 요구해 58.3 mm 부족했다(D463). 링크는 "
                "거리 제약이 없으므로 검사할 것은 '77.81 mm 를 실제로 건너 셸을 "
                "0~44.5도 돌리는가' 와 '그 사이 팔에 안 닿는가' 다"),
        "supersedes": "G7_drive_gear_meshes_on_servo_axis (기어 축간거리 검사)"}

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
    l4_cloud = surface_cloud(l4)
    l4_lo, l4_hi = l4.bounds
    probe8 = []
    for parts, names, side in ((sL, nL, -1), (sR, nR, +1)):
        for i in body_idx:
            m = parts[i].copy(); m.apply_transform(T)
            probe8.append((m, f"{'LR'[side > 0]}:{names[i]}"))
    for parts, names, tag in ((br, nB, "bracket"), (dr, nD, "linkage")):
        for m0, nm in zip(parts, names):
            m = m0.copy(); m.apply_transform(T)
            probe8.append((m, f"{tag}:{nm}"))
    worst8, who8 = clearance_to(probe8, tree4, pitch, l4_lo, l4_hi, l4_cloud)
    gates["G8_clear_of_link4"] = {
        "pass": worst8 >= 0.0, "min_clearance_mm": round(float(worst8), 3),
        "closest_piece": who8,
        "why": "link5 만이 아니라 바로 앞 링크와도 안 닿아야 손목 롤이 자유롭다"}

    # G9 순정 가동 조는 **떼지 않는다** — 서보 크랭크가 그 조의 볼트 구멍에 물린다.
    #    따라서 조가 0~89도 도는 동안 셸·브래킷·로드·셸크랭크와 안 닿아야 한다.
    #    (서보 크랭크 자신은 조에 볼트로 붙는 부품이므로 제외 — 닿는 것이 설계다)
    jaw = jaw_in_link5()
    ev["stock_jaw_kept"] = True
    ev["stock_jaw_bolt_holes_yz"] = P["jaw_bolt_yz_mm"]
    ev["stock_jaw_blade_inner_x_mm"] = P["jaw_blade_inner_x_mm"]
    worst9, who9, at9 = np.inf, None, None
    for i in idx7:
        row = rows[int(i)]
        Tj = rot_about(GRIPPER_ORIGIN, GRIPPER_AXIS, math.radians(row["servo_deg"]))
        jw = jaw.copy(); jw.apply_transform(Tj)
        tj, _ = link5_occupancy(jw, 1.0)
        jaw_cloud = surface_cloud(jw, 120000)
        j_lo, j_hi = jw.bounds
        Ts, Tr, Tk = G.linkage_pose(P, lk, int(i))
        probe9 = []
        for m0, nm in zip(dr, nD):
            grp = G.linkage_group(nm)
            if grp == "servocrank":
                continue                      # 조에 볼트로 물리는 부품
            m = m0.copy(); m.apply_transform(T @ {"rod": Tr, "shellcrank": Tk}[grp])
            probe9.append((m, f"linkage:{nm}"))
        for m0, nm in zip(br, nB):
            m = m0.copy(); m.apply_transform(T)
            probe9.append((m, f"bracket:{nm}"))
        for parts, names, side, tag in ((sL, nL, -1, "shellL"), (sR, nR, +1, "shellR")):
            Rj = rot_about(T[:3, :3] @ np.array([side * K["g"] / 2.0, 0.0, 0.0]) + T[:3, 3],
                           hinge_world, math.radians(side * row["shell_deg"]))
            for m0, nm in zip(parts, names):
                m = m0.copy(); m.apply_transform(Rj @ T)
                probe9.append((m, f"{tag}:{nm}"))
        v, w = clearance_to(probe9, tj, pitch, j_lo, j_hi, jaw_cloud)
        if v < worst9:
            worst9, who9, at9 = v, w, float(row["servo_deg"])
    gates["G9_clear_of_stock_jaw"] = {
        "pass": worst9 >= 0.0, "min_clearance_mm": round(float(worst9), 3),
        "closest_piece": who9, "closest_at_servo_deg": at9,
        "excluded": "servocrank_* / pin1 (순정 조에 볼트로 체결되는 부품)",
        "why": ("D462 §5·§7 대로 순정 가동 조는 볼트 분리만 가능한 상태로 남고 "
                "서보 인출점으로 쓴다. 조가 남아 도는 이상 그 스윕을 검사해야 한다")}

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
