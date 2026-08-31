"""조개형 그랩 v1 — 대칭 2피벗 + 1:1 셸 기어쌍 + **4절 링크 구동** 생성기 (D462·D463).

v0(`scoop_shell_design.py`)와의 차이
------------------------------------
v0 = 순정 2-jaw 의 **각 조에 셸을 하나씩 볼트로 덧댐**. 한쪽 조만 회전하므로
     대칭 클램셸이 아니었고, 폐합 여유가 순정 판재 간격 4.05 mm 에 갇혀 0.05 mm 였다.
v1 = **앞부분 교체**. 고정 조 블레이드의 기존 4볼트 사각형
     (25.19 x 19.44 mm, D462 §1)에 브래킷을 물려 **피벗 2개를 자체 설계**한다.
     두 셸 기어가 직접 맞물려 **대칭 폐합**한다 (1:1, 피벗 간격 26 mm 에서 접함).

구동 전달: 기어 직결 -> **4절 링크** (D463 이 D462 §7 을 scoped-supersede)
-------------------------------------------------------------------------
g2 프로브(p37)가 **서보축 ↔ 셸 피벗 L 축 수직거리 77.811 mm** 를 실측했다.
스퍼 기어 직결은 이 거리가 r_drive + r_shell = 19.5 mm 여야 하므로 **58.311 mm 부족**이고,
그랩을 팔에서 멀리 빼야 충돌이 없고(G4) 가까이 붙여야 기어가 물리는(G7) **구조적 모순**이라
스탠드오프로 풀 수 없다. 그래서 서보 -> 셸 L 은 **크랭크 2 + 로드 1 + 핀 2** 의 4절로 잇는다.
셸 R 은 기존 1:1 셸 기어쌍이 그대로 대칭 구동한다 (기어쌍은 살아 있다).

    서보 크랭크 : 순정 가동 조가 쓰던 볼트 구멍(스팬 25 mm, link5 X 관통)에 체결.
                  서보축 둘레로 0~89도 회전한다.
    로드        : 핀 2개로 두 크랭크를 잇는다. 힌지축 방향으로 셸 바깥에 층을 이룬다.
    셸 크랭크   : 셸 L 과 한 몸. 피벗 L 에서 뻗어 로드를 받는다. 0~44.5도.

⚠️ 대가: **개구가 서보각에 비례하지 않는다**(기어 직결의 직선성이 깨진다).
   `design.json` 의 `linkage.table` 에 servo_deg -> shell_deg -> mouth_mm 전 구간을 싣는다.
   사점(dead center)·자기잠김은 `linkage_no_dead_center` 게이트가 전달각으로 막는다.

왜 2:1 인가 (D462 g1 스윕)
--------------------------
서보 89도를 셸 44.5도로 줄이면, 같은 개구 58 mm 를 내기 위해 립이 더 멀리 있어야 한다
(16 -> 36 mm). 립이 길어지면 보울이 깊어져 적재가 늘고, 감속만큼 폐합 토크도 2배가 된다.
링크에서는 이 2:1 이 **행정 끝점 사이의 총비**이고, 순간비는 각도마다 달라진다.
D446 준수: 보울·기어·판 전부 **볼록 조각으로 분해**한다. 1-hull 금지.

사용:  python scoop_grab_v1_design.py [출력디렉터리]
"""
import sys, json, math
from pathlib import Path
import numpy as np
import trimesh

OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else "claudedocs/runtime_logs/grab_track/g3_linkage")

# ─────────────────────────────────────────────────────────────────────────
# 1. 파라미터
# ─────────────────────────────────────────────────────────────────────────
P = {
    # ── 기구 (D462 g1 스윕에서 선정) ─────────────────────────────────────
    "pivot_gap_mm":       26.0,   # 두 피벗 간격. 셸 기어 피치원이 여기서 접한다
    "lip_depth_mm":       36.06,  # 닫힘 시 립이 피벗선 아래로 내려간 깊이
    "mouth_open_mm":      58.0,   # 완전개방 립-투-립 (D457 §12 유지)
    "shell_travel_deg":   44.5,   # 셸 편측 회전 = 서보 89도 / 2
    "servo_travel_deg":   89.0,   # URDF link5_to_gripper_link limit 0~1.571 rad
    "gear_ratio":          2.0,   # 서보 -> 셸 **총행정** 비 (링크에서 순간비는 비선형)
    "bulge_frac":         0.45,   # 보울 바깥 부풀림 / 립깊이

    # ── 셸 ───────────────────────────────────────────────────────────────
    "shell_width_mm":     50.0,   # 힌지축(Z) 방향 너비. 40 -> 50 으로 넓혀 자중/적재를
                              # 개선했다: 적재는 너비에 비례하나 기어·브래킷·허브·구동암은
                              # 너비와 무관해 분모만 커진다. link5 본체 35.5 mm + 편측 7.25 오버행
    "wall_mm":             2.0,
    "side_plate_thk_mm":   1.5,
    "lip_land_mm":         0.8,   # 립 평탄 랜드 (나이프에지 금지 — 칩핑 방지)
    "seg_n":              20,     # 보울 각도 분해 수 (D461 §5: 10은 각져서 펠릿이 낀다)

    # ── 기어 (모듈 1.0. 피치 = m*z) ──────────────────────────────────────
    "gear_module_mm":      1.0,
    "shell_gear_teeth":   26,     # 피치 ⌀26 -> 두 개가 g=26 에서 접함 (1:1 대칭)
    "gear_width_mm":       6.0,
    "gear_backlash_mm":    0.15,  # FDM 인쇄 여유. 0 이면 물려서 안 돈다

    # ── 브래킷 (D462 §1 실측 — 고정 조 블레이드의 기존 구멍) ─────────────
    "bracket_bolt_dy_mm": 25.19,  # Y span  (−13.34 ↔ +11.85)
    "bracket_bolt_dz_mm": 19.44,  # Z span  ( 83.46 ↔ 102.90)
    "bolt_clear_d_mm":     3.4,   # ISO 273 M2.5 normal. D461 §1: 2.8은 FDM 수축으로 안 들어갔다
    "bracket_thk_mm":      4.0,
    # ⚠️ g2 프로브(p37) 가 잡은 결함: 피벗을 볼트 사각형 중심에 두면 그랩이
    #    link5 몸통(Z -0.75~119.89) 안에 박힌다(관통 -67.8 mm). 블레이드 끝 너머로
    #    피벗을 빼내는 스탠드오프가 필요하다. 볼트 중심 Z=93.18, 블레이드 끝 Z=119.89
    #    -> 최소 26.7 mm. 여유 5 mm 를 더해 32 로 잡는다.
    "bracket_standoff_mm": 32.0,
    "pivot_shaft_d_mm":    3.0,   # M3 볼트 축
    "pivot_boss_d_mm":     6.0,   # 브래킷 피벗 부시 바깥지름. 셸/셸크랭크가 이 위를 돈다.
                                  # 8.0 -> 6.0 (2026-09-01): 퇴화 정정으로 허브가 실체를 갖자
                                  # 팔 여유가 G4 0.024 / G9 0.081 mm 로 소진됐다. 허브 바깥반지름은
                                  # boss_d/2 + bore_clear + hub_wall 이라 boss_d 를 줄이면 그대로 따라 준다.
                                  # 실측: G4 0.024->0.194, G9 0.081->0.325, 자중 52.67->48.27 g (상한 50 통과).
                                  # 5.0 이면 여유 0.694 로 더 좋으나 M3(3mm) 위 벽이 1.0mm 뿐이라 채택 안 함.
    "pivot_bore_clear_mm": 0.6,   # 보스 OD <-> 회전체 보어 틈 (FDM 러닝핏. 0.25 는
                                  # 인쇄 공차로 뻑뻑하고 게이트 0.5 mm 하한도 못 넘는다)
    "hub_wall_mm":         1.5,   # 셸 보어 랜드 벽두께

    # ── link5 실측 상수 (배치를 푸는 데 필요. p37 placement() 와 같은 출처) ──
    #    이 값들이 있어야 서보축을 그랩 로컬 좌표로 옮길 수 있다. 링크는 서보축과
    #    피벗 L 을 잇는 기구이므로 두 축의 상대 위치 없이는 설계가 성립하지 않는다.
    "link5_blade_x_mm":      [-11.54, -10.03],   # 고정 조 블레이드 판재 (D462 §2)
    "link5_blade_tip_z_mm":  119.886,            # link5 최대 Z = 블레이드 끝 (D462 §2)
    "link5_blade_hole_c_yz": [-0.745, 93.18],    # 4볼트 사각형 중심 (D462 §1)
    "servo_axis_link5":      [0.0, 18.821, 52.035],   # 그리퍼 개폐축 통과점 (D462 §5)
    "jaw_blade_inner_x_mm":  -6.00,   # 순정 가동 조 블레이드 안쪽면 (74th 실측)
    "jaw_bolt_yz_mm":        [[-12.66, 82.98], [12.45, 82.98]],   # 스팬 25.11
    "jaw_mount_thk_mm":      3.0,     # 조 사이 간극 4.03 mm 안에 들어가는 판 두께
    "jaw_mount_len_y_mm":   11.0,     # 크랭크판 깊이(로컬 Y). ⌀3.4 구멍 둘레 랜드 3.8 mm
    "jaw_mount_z_mm":       [-18.5, 16.5],   # 크랭크판의 힌지축 방향 범위

    # ── 4절 링크 구동 (D463 — 기어 직결 기각) ────────────────────────────
    #    자유 파라미터는 3개(크랭크 반경 1 + 초기 방위 2)뿐이다. 로드 길이와 셸
    #    크랭크 반경은 "서보 0/89도 <-> 셸 0/44.5도" 두 끝점 폐합 조건에서
    #    **닫힌 형태로 유일하게 결정**된다 (linkage_solve). 튜닝 여지가 없다.
    #    선정 근거: 3파라미터 전수 탐색(r2 1 mm · 방위 2도 격자, 후보 444개 생존)에서
    #    **입력·출력 전달각이 둘 다 40~140 대역 안에 머무는 여유(margin)** 를 최대화하고
    #    핀 하중(1/r2)·셸 크랭크 길이·포락선으로 균형을 잡은 값. 여유 8.67도.
    "crank_servo_r_mm":     21.0,     # 서보축 -> 핀1
    "crank_servo_a0_deg": -106.0,     # 닫힘(서보 0도)에서 서보축->핀1 방위 (로컬 XY)
    "crank_shell_a0_deg":  128.0,     # 닫힘(셸 0도)에서 피벗L->핀2 방위 (로컬 XY)
    "trans_angle_band_deg": [40.0, 140.0],   # 전달각 허용대역 (사점·자기잠김 금지)
    # 힌지축(로컬 Z) 방향 층 구성. 셸 바깥면 -26.5 -> 크랭크 -> 로드 순으로 쌓는다
    "crank_plane_z_mm":   -29.0,      # 두 크랭크 판 중심
    "crank_thk_mm":         4.0,
    "crank_servo_w_mm":    12.0,      # 서보 크랭크 팔 폭 (면내)
    "crank_shell_w_mm":    14.0,      # 셸 크랭크 팔 폭 (면내). 뿌리 굽힘이 지배한다
    "rod_plane_z_mm":     -33.5,      # 로드 판 중심
    "rod_thk_mm":           4.0,
    "rod_width_mm":         9.0,      # 쌍레일 바깥 폭 (레일 중심간이 아니라 전폭)
    "rod_rail_w_mm":        2.5,      # 레일 1개 폭. 두 개 = 유효 단면 5.0 x 4.0
    "crank_tip_w_mm":       9.0,      # 크랭크 팔 끝 폭 (핀 보스 지름 8 + 여유)
    "pin_d_mm":             3.0,      # M3 핀
    "pin_eye_d_mm":         9.0,      # 로드 아이 바깥지름
    "pin_boss_d_mm":        8.0,      # 크랭크 쪽 핀 보스 바깥지름

    # ── 재료 · 검증 기준 ────────────────────────────────────────────────
    "density_g_cm3":       1.24,  # ⚠️ 프린터에 물린 것은 PLA(1.24)다. PETG면 1.27
    "material":           "PLA",  # 파이프라인 실측(filament_full.json = ['PLA'])
    "tool_mass_max_g":    50.0,
    "fill_factor":         0.70,
    "bulk_density_g_cm3":  0.55,  # ⚠️MEASURE 펠릿 도착 후 250 ml 계량컵 칭량
    "bulk_density_src":   "ASSUMED",   # 실측하면 "MEASURED_YYYY-MM-DD" 로 교체.
                                       # ASSUMED 인 한 self_load_ratio 게이트는 FAIL 유지
    "pellet_d_max_mm":     5.0,
    "pile_width_mm":     377.5,   # s1 DEME 더미 (D462 격자)
    "servo_torque_nm":     1.91,  # ST3215 @7.4V (공식). ⚠️ 실측 조 힘은 1.8~6.3 N 이었다
    "self_load_ratio_band": [0.72, 0.88],   # 실물 그랩 (D457 §12)
}

# ─────────────────────────────────────────────────────────────────────────
# 2. 볼록 조각 생성기 (D446)
# ─────────────────────────────────────────────────────────────────────────
def box(lx, ly, lz, center=(0, 0, 0)):
    m = trimesh.creation.box(extents=(lx, ly, lz))
    m.apply_translation(center)
    return m


def arc_segment(cx, cy, r_in, r_out, a0, a1, width):
    """중심 (cx,cy) 기준 호 세그먼트. 8정점 -> 볼록.

    축 규약(D461 §3): 호는 회전 평면 X-Y 에, 너비는 힌지축 Z 를 따라.
    """
    pts = []
    for a in (a0, a1):
        c, s = math.cos(a), math.sin(a)
        pts += [(cx + r_in * c, cy + r_in * s), (cx + r_out * c, cy + r_out * s)]
    v = [(x, y, z) for z in (-width / 2, width / 2) for (x, y) in pts]
    return trimesh.convex.convex_hull(trimesh.Trimesh(vertices=np.array(v, float)))


def plate_with_holes(lx, ly, lz, hole_ys, hd, center=(0, 0, 0), hole_axis="z"):
    """볼트 구멍 있는 평판을 **볼록 박스로 분해**. 구멍은 사각 (D446·D460 §6).

    hole_axis: 구멍이 뚫리는 축. "z" 면 Z 를 따라, "x" 면 X 를 따라 관통한다.
    ⚠️ g2 프로브(p37)가 잡은 결함: 브래킷 볼트는 블레이드(법선 = link5 X)를 관통해야
       하는데 초판은 힌지축과 같은 방향으로 뚫려 있었다. 축이 90도 어긋나면 볼트가
       판재를 지나갈 수 없다.
    """
    out, edges = [], []
    for hy in sorted(hole_ys):
        edges += [hy - hd / 2, hy + hd / 2]
    bounds = [-ly / 2] + edges + [ly / 2]
    cx, cy, cz = center
    for i in range(len(bounds) - 1):
        y0, y1 = bounds[i], bounds[i + 1]
        if y1 - y0 < 1e-6:
            continue
        ymid = (y0 + y1) / 2
        if any(abs(ymid - hy) < hd / 2 for hy in hole_ys):
            if hole_axis == "x":            # X 를 따라 관통 -> Z 방향으로 갈라 띄운다
                side = (lz - hd) / 2
                for sgn in (+1, -1):
                    out.append(box(lx, y1 - y0, side,
                                   center=(cx, cy + ymid, cz + sgn * (hd + side) / 2)))
            else:                           # Z 를 따라 관통 -> X 방향으로 갈라 띄운다
                side = (lx - hd) / 2
                for sgn in (+1, -1):
                    out.append(box(side, y1 - y0, lz,
                                   center=(cx + sgn * (hd + side) / 2, cy + ymid, cz)))
        else:
            out.append(box(lx, y1 - y0, lz, center=(cx, cy + ymid, cz)))
    return out


RING_SEG_N = 16      # 전체원·환형을 쪼갤 쐐기 수 (D463 후속: 퇴화 조각 정정)


def ring(cx, cy, r_in, r_out, z0, z1, n_seg=RING_SEG_N, bore_clear=True, outer_full=False):
    """전체원·환형·원반을 **볼록 쐐기 n개로 분해**한다. 보울의 seg_n 분해와 같은 방식.

    🔴 왜 필요한가 (D463 후속 정정): `arc_segment(cx,cy,r_in,r_out,a0,a1,w)` 는 a0·a1
       두 각의 점만으로 볼록 껍질을 만든다. 그래서
         - a0=0, a1=2*pi  -> 두 각의 좌표가 같아 정점 4개  -> **납작, 부피 0**
         - a0=0, a1=pi    -> 정점 4개가 한 평면          -> **납작, 부피 0**
       였고, 무엇보다 **환형(annulus)은 정의상 볼록이 아니라 한 조각으로 표현할 수
       없다**. 기존 `hub` / `pivot_boss_L,R` / 기어 뿌리원반 8개가 이 상태로
       "인쇄하면 아무것도 안 나오는" 조각이었다. 볼록 게이트는 부피 0 을 건너뛰어
       이를 못 봤다 (D461 §6 "게이트가 무엇을 못 보는지 물어라" 의 재판).

    bore_clear : 다각형 현이 보어를 잠식하지 않도록 r_in 을 1/cos(pi/n) 로 키운다
                 -> 실제 보어가 요구 지름 이상으로 유지된다 (M3 축이 들어가야 한다).
    outer_full : 바깥지름을 다각형이 **감싸도록** r_out 을 1/cos(pi/n) 로 키운다.
                 기어 뿌리원반처럼 이빨 뿌리(r_f)까지 반드시 닿아야 하는 경우에 쓴다
                 (안 쓰면 다각형 변 중앙에서 r_f*(1-cos(pi/n)) 만큼 이빨이 뜬다).
    """
    k = 1.0 / math.cos(math.pi / n_seg)
    ri = r_in * k if (bore_clear and r_in > 0.0) else r_in
    ro = r_out * k if outer_full else r_out
    out = []
    for i in range(n_seg):
        a0 = 2 * math.pi * i / n_seg
        a1 = 2 * math.pi * (i + 1) / n_seg
        m = arc_segment(cx, cy, ri, ro, a0, a1, z1 - z0)
        m.apply_translation((0.0, 0.0, (z0 + z1) / 2.0))
        out.append(m)
    return out


def bar_xy(p0, p1, width, z0, z1, ext0=0.0, ext1=0.0):
    """XY 평면 p0->p1 막대(볼록 박스 1개). Z 로 두께를 준다.

    ext0/ext1 은 각 끝을 축방향으로 늘린다 (뿌리 겹침·아이 보강용).
    """
    p0 = np.asarray(p0, float)[:2]
    p1 = np.asarray(p1, float)[:2]
    d = p1 - p0
    L = float(np.linalg.norm(d))
    e = d / L
    q0, q1 = p0 - ext0 * e, p1 + ext1 * e
    m = box(float(np.linalg.norm(q1 - q0)), width, z1 - z0)
    c, s = float(e[0]), float(e[1])
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = np.array([(q0[0] + q1[0]) / 2.0, (q0[1] + q1[1]) / 2.0, (z0 + z1) / 2.0])
    m.apply_transform(T)
    return m


def taper_xy(p0, p1, w0, w1, z0, z1, ext0=0.0, ext1=0.0):
    """뿌리 w0 -> 끝 w1 로 좁아지는 사다리꼴 기둥 (볼록 1조각).

    크랭크 팔의 굽힘 모멘트는 뿌리에서 끝으로 **선형으로 줄어든다**. 폭을 같은
    비율로 줄이면 뿌리 단면(=지배 단면)은 그대로 두고 질량만 던다 — 강성을 근거
    없이 깎는 것이 아니다. 뿌리 응력은 변하지 않는다.
    """
    p0 = np.asarray(p0, float)[:2]
    p1 = np.asarray(p1, float)[:2]
    d = p1 - p0
    L = float(np.linalg.norm(d))
    e = d / L
    n = np.array([-e[1], e[0]])
    q0, q1 = p0 - ext0 * e, p1 + ext1 * e
    v = []
    for z in (z0, z1):
        v += [(*(q0 + w0 / 2 * n), z), (*(q0 - w0 / 2 * n), z),
              (*(q1 + w1 / 2 * n), z), (*(q1 - w1 / 2 * n), z)]
    return trimesh.convex.convex_hull(trimesh.Trimesh(vertices=np.array(v, float)))


def plate_holes_along_z(thk, cx, cy, y_len, z_lo, z_hi, hole_zs, hd):
    """볼트 구멍이 **로컬 Z 로 떨어져** 있고 구멍 축은 로컬 X 인 판.

    `plate_with_holes` 는 구멍이 로컬 Y 로 떨어진 경우만 다룬다. 순정 가동 조의
    체결 구멍 2개는 힌지축(로컬 Z)을 따라 25.11 mm 떨어져 있으므로 축이 다르다.
    구멍은 D446·D460 §6 규약대로 **사각**으로 근사한다.
    """
    out, edges = [], []
    for hz in sorted(hole_zs):
        edges += [hz - hd / 2.0, hz + hd / 2.0]
    bounds = [z_lo] + edges + [z_hi]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 1e-6:
            continue
        zm = (a + b) / 2.0
        if any(abs(zm - hz) < hd / 2.0 for hz in hole_zs):
            side = (y_len - hd) / 2.0          # 구멍 줄은 로컬 Y 로 갈라 띄운다
            for sgn in (+1, -1):
                out.append(box(thk, side, b - a,
                               center=(cx, cy + sgn * (hd + side) / 2.0, zm)))
        else:
            out.append(box(thk, y_len, b - a, center=(cx, cy, zm)))
    return out


def gear_teeth(cx, cy, n_teeth, module, width, backlash=0.0, a_start=0.0, a_span=None):
    """스퍼 기어 이빨을 **볼록 쐐기**로 생성. 인볼류트가 아니라 사다리꼴 근사다.

    저속·저부하·왕복 44.5도 용도라 사다리꼴로 충분하며, 인볼류트를 쓰면 조각이
    비볼록이 되어 D446 등식이 깨진다. 백래시는 이 두께를 깎아 만든다.
    """
    r_p = module * n_teeth / 2.0          # 피치원
    r_a = r_p + module                    # 이끝원
    r_f = r_p - 1.25 * module             # 이뿌리원
    # 뿌리 원반: 반원 2개(arc_segment(0,pi)/(pi,2pi))는 정점 4개가 한 평면에 놓여
    # **부피 0 으로 퇴화**했다. 쐐기로 분해하고, 이빨 뿌리에 반드시 닿도록 외접시킨다.
    pieces = list(ring(cx, cy, 0.0, r_f, -width / 2.0, width / 2.0,
                       n_seg=RING_SEG_N, outer_full=True))
    span = a_span if a_span is not None else 2 * math.pi
    half = (math.pi / n_teeth) / 2.0 - backlash / (2.0 * r_p)
    k = 0
    while k * (2 * math.pi / n_teeth) < span - 1e-9:
        a = a_start + k * (2 * math.pi / n_teeth)
        pieces.append(arc_segment(cx, cy, r_f, r_a, a - half, a + half, width))
        k += 1
    return pieces


# ─────────────────────────────────────────────────────────────────────────
# 3. 기구학
# ─────────────────────────────────────────────────────────────────────────
def kin(P):
    """피벗/립/호 중심을 푼다. 좌표계: X=분리 방향, Y=깊이(-가 아래), Z=힌지축."""
    g, d = P["pivot_gap_mm"], P["lip_depth_mm"]
    R = math.hypot(g / 2.0, d)                       # 립의 피벗 반경
    a0 = math.atan2(-d, g / 2.0)                     # 닫힘 시 립 방위 (좌측 셸 기준)
    return {"g": g, "d": d, "R": R, "a0": a0,
            "pivot_L": (-g / 2.0, 0.0), "pivot_R": (+g / 2.0, 0.0),
            "lip_closed": (0.0, -d)}


def mouth_at(P, phi_deg):
    k = kin(P)
    return k["g"] - 2.0 * k["R"] * math.cos(k["a0"] - math.radians(phi_deg))


def to_local(P, p_link5):
    """link5 좌표 -> 그랩 로컬 좌표.

    p37 `placement()` 의 역변환이며 **같은 상수**를 쓴다 (두 파일이 어긋나면
    링크가 서보축을 못 맞춘다). 로컬 X = link5 +X, 로컬 Y = link5 -Z(깊이),
    로컬 Z = link5 +Y(힌지축).
    """
    t = np.array([P["link5_blade_x_mm"][0] - P["bracket_thk_mm"] / 2.0,
                  P["link5_blade_hole_c_yz"][0],
                  P["link5_blade_hole_c_yz"][1] + P["bracket_standoff_mm"]])
    d = np.asarray(p_link5, float) - t
    return np.array([d[0], -d[2], d[1]])


def servo_axis_local(P):
    """서보 개폐축 통과점을 그랩 로컬 좌표로. 축 방향은 로컬 +Z (= link5 +Y)."""
    return to_local(P, P["servo_axis_link5"])


def jaw_mount_local(P):
    """서보 크랭크판이 순정 가동 조에 물리는 볼트선 중심 (그랩 로컬)."""
    ys = [h[0] for h in P["jaw_bolt_yz_mm"]]
    zs = [h[1] for h in P["jaw_bolt_yz_mm"]]
    x = P["jaw_blade_inner_x_mm"] - P["jaw_mount_thk_mm"] / 2.0
    return to_local(P, [x, float(np.mean(ys)), float(np.mean(zs))])


def _u(a):
    return np.array([math.cos(a), math.sin(a)])


def linkage_solve(P, n=91):
    """서보 -> 셸 L 4절 링크를 푼다.

    자유 파라미터는 (crank_servo_r, crank_servo_a0, crank_shell_a0) 3개뿐이다.
    로드 길이 r_rod 와 셸 크랭크 반경 r_shell 은 두 끝점 폐합 조건
        |P1(0)   - P2(0)|      = r_rod
        |P1(89도) - P2(-44.5도)| = r_rod
    에서 **1차 방정식으로 유일하게** 결정된다 (r_rod^2 항이 소거된다).
    -> 행정 끝점은 정의상 정확히 맞고, 사이 구간만 검사 대상이 된다.

    반환 표의 servo_deg -> shell_deg -> mouth_mm 가 **비선형 개구 곡선**이다.
    """
    A = servo_axis_local(P)[:2]
    B = np.array(kin(P)["pivot_L"], float)
    th_max = math.radians(P["servo_travel_deg"])
    ph_max = math.radians(P["shell_travel_deg"])
    b0 = math.radians(P["crank_servo_a0_deg"])
    g0 = math.radians(P["crank_shell_a0_deg"])
    r2 = float(P["crank_servo_r_mm"])

    d0 = A + r2 * _u(b0) - B
    d1 = A + r2 * _u(b0 + th_max) - B
    u0, u1 = _u(g0), _u(g0 - ph_max)
    den = 2.0 * (d0 @ u0 - d1 @ u1)
    if abs(den) < 1e-9:
        raise ValueError("링크 폐합 방정식이 퇴화했다 (크랭크 방위를 바꿀 것)")
    r4 = float((d0 @ d0 - d1 @ d1) / den)
    r3 = float(np.linalg.norm(d0 - r4 * u0))

    rows, gprev = [], g0
    for i in range(n):
        th = th_max * i / (n - 1)
        P1 = A + r2 * _u(b0 + th)
        d = P1 - B
        Ld = float(np.linalg.norm(d))
        if Ld > r4 + r3 or Ld < abs(r4 - r3):
            raise ValueError(f"서보 {math.degrees(th):.1f}도에서 링크가 닫히지 않는다")
        a_ = (Ld ** 2 + r4 ** 2 - r3 ** 2) / (2.0 * Ld)
        h = math.sqrt(max(0.0, r4 ** 2 - a_ ** 2))
        e = d / Ld
        nn = np.array([-e[1], e[0]])
        best, bestg = None, None
        for c in (B + a_ * e + h * nn, B + a_ * e - h * nn):
            g = math.atan2(*(c - B)[::-1])
            while g - gprev > math.pi:
                g -= 2 * math.pi
            while g - gprev < -math.pi:
                g += 2 * math.pi
            if best is None or abs(g - gprev) < abs(bestg - gprev):
                best, bestg = c, g
        P2, gprev = best, bestg
        phi = math.degrees(g0 - bestg)

        def _ang(a, b):
            return math.degrees(math.acos(float(np.clip(
                (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))

        # 출력측 전달각: 로드 <-> 셸 크랭크 (P2 에서)
        mu_out = _ang(P1 - P2, B - P2)
        # 입력측 전달각: 서보 크랭크 <-> 로드 (P1 에서).
        #   0/180 도가 되면 크랭크와 로드가 일직선 = **입력 사점(토글)** 이고
        #   그 점에서 셸은 서보 회전에 거의 반응하지 않는다(자기잠김).
        #   출력측 전달각만으로는 이 상태를 못 잡는다 — 둘 다 봐야 한다.
        mu_in = _ang(A - P1, P2 - P1)
        rows.append({"servo_deg": round(math.degrees(th), 4),
                     "shell_deg": round(phi, 4),
                     "mouth_mm": round(mouth_at(P, phi), 4),
                     "trans_angle_deg": round(mu_out, 3),
                     "trans_angle_in_deg": round(mu_in, 3),
                     "pin1_xy": [round(float(P1[0]), 4), round(float(P1[1]), 4)],
                     "pin2_xy": [round(float(P2[0]), 4), round(float(P2[1]), 4)]})
    # 순간비 dshell/dservo (중앙차분). 1 보다 작으면 그 구간에서 토크가 증폭된다
    sd = np.array([r["servo_deg"] for r in rows])
    ph = np.array([r["shell_deg"] for r in rows])
    grad = np.gradient(ph, sd)
    for r, gv in zip(rows, grad):
        r["dshell_dservo"] = round(float(gv), 4)
        r["torque_gain"] = round(float(1.0 / gv) if abs(gv) > 1e-9 else float("inf"), 3)
    return {"servo_axis_xy": [round(float(A[0]), 4), round(float(A[1]), 4)],
            "pivot_L_xy": [round(float(B[0]), 4), round(float(B[1]), 4)],
            "jaw_mount_xy": [round(float(v), 4) for v in jaw_mount_local(P)[:2]],
            "jaw_mount_z_mm": round(float(jaw_mount_local(P)[2]), 4),
            "ground_len_mm": round(float(np.linalg.norm(A - B)), 4),
            "crank_servo_r_mm": round(r2, 4),
            "crank_shell_r_mm": round(r4, 4),
            "rod_len_mm": round(r3, 4),
            "crank_servo_a0_deg": P["crank_servo_a0_deg"],
            "crank_shell_a0_deg": P["crank_shell_a0_deg"],
            "trans_angle_out_min_deg": round(min(r["trans_angle_deg"] for r in rows), 3),
            "trans_angle_out_max_deg": round(max(r["trans_angle_deg"] for r in rows), 3),
            "trans_angle_in_min_deg": round(min(r["trans_angle_in_deg"] for r in rows), 3),
            "trans_angle_in_max_deg": round(max(r["trans_angle_in_deg"] for r in rows), 3),
            "rows": rows}


def bowl_arc(P, side):
    """셸 안쪽 보울 호의 (중심, 반지름, 시작각, 끝각). side=-1 좌, +1 우."""
    k = kin(P)
    px, py = (side * k["g"] / 2.0, 0.0)
    lx, ly = k["lip_closed"]
    bulge = P["bulge_frac"] * k["d"]
    chord = math.hypot(lx - px, ly - py)
    r = (chord ** 2 / 4.0 + bulge ** 2) / (2.0 * bulge)
    mx, my = (px + lx) / 2.0, (py + ly) / 2.0
    nx, ny = -(ly - py) / chord, (lx - px) / chord
    nx, ny = side * nx, side * ny
    cx, cy = mx - nx * (r - bulge), my - ny * (r - bulge)
    a1 = math.atan2(py - cy, px - cx)
    a2 = math.atan2(ly - cy, lx - cx)
    if side < 0 and a2 < a1:
        a2 += 2 * math.pi
    if side > 0 and a2 > a1:
        a2 -= 2 * math.pi
    # 립 끝단 보정: 안쪽면(r)이 아니라 **바깥면(r+wall)** 이 중심선 x=0 에 닿게 끝각을 당긴다.
    # 안 하면 립에서 바깥 법선이 반대편으로 기울어 벽 두께만큼 중심선을 넘어간다
    # (실측 0.36 mm/셸 -> 두 셸 0.717 mm 간섭. shells_meet_not_overlap 게이트가 잡았다).
    ro = r + P["wall_mm"]
    if abs(cx) < ro:
        t_out = math.acos(max(-1.0, min(1.0, -cx / ro)))      # 바깥면이 x=0 이 되는 각
        for cand in (t_out, -t_out, 2 * math.pi - t_out, t_out - 2 * math.pi):
            if min(a1, a2) - 1e-6 <= cand <= max(a1, a2) + 1e-6:
                a2 = cand
                break
    return (cx, cy), r, a1, a2, bulge


# ─────────────────────────────────────────────────────────────────────────
# 4. 부품 조립 (전부 볼록 조각 리스트로 반환)
# ─────────────────────────────────────────────────────────────────────────
def build_shell(P, side):
    """셸 1매 = 보울 세그먼트 + 측판 2 + 립 + 허브 + 기어섹터. 전부 볼록."""
    k = kin(P)
    (cx, cy), r, a1, a2, _ = bowl_arc(P, side)
    w, wall = P["shell_width_mm"], P["wall_mm"]
    parts, names = [], []

    n = P["seg_n"]
    for i in range(n):
        t0 = a1 + (a2 - a1) * i / n
        t1 = a1 + (a2 - a1) * (i + 1) / n
        parts.append(arc_segment(cx, cy, r, r + wall, min(t0, t1), max(t0, t1), w))
        names.append(f"bowl_{i:02d}")

    # 측판: 보울 호를 덮는 부채꼴 판을 Z 양끝에
    st = P["side_plate_thk_mm"]
    for sgn, tag in ((+1, "a"), (-1, "b")):
        z = sgn * (w / 2 + st / 2)
        sp = arc_segment(cx, cy, 0.0, r + wall, min(a1, a2), max(a1, a2), st)
        sp.apply_translation((0, 0, z))
        parts.append(sp); names.append(f"side_{tag}")

    # 립: 닫힘 시 맞닿는 평탄 랜드
    lx, ly = k["lip_closed"]
    land = P["lip_land_mm"]
    parts.append(box(land, wall, w, center=(lx + side * land / 2, ly + wall / 2, 0)))
    names.append("lip")

    # 허브 + 기어 섹터 (피벗 둘레)
    px, py = (side * k["g"] / 2.0, 0.0)
    gw = P["gear_width_mm"]
    # 허브(환형): 이전에는 arc_segment(0, 2*pi) 한 조각이라 **부피 0** 이었다.
    # 🔴 실체화하면서 드러난 두 번째 결함: 반지름이 브래킷 피벗 보스와 **완전히 같아**
    #    (둘 다 r 1.5~4, 같은 Z 구간) 두 부품이 서로 파고든다. 부피 0 일 때는 안 보였다.
    #    보스가 고정 축(부시)이고 셸이 그 위를 도는 것이 맞으므로, 허브를 보스 바깥의
    #    **보어 랜드**로 옮긴다. 셸 크랭크 슬리브가 쓰는 반지름 규약과 같다.
    r_bore_in = P["pivot_boss_d_mm"] / 2.0 + P["pivot_bore_clear_mm"]
    for j, hp in enumerate(ring(px, py, r_bore_in, r_bore_in + P["hub_wall_mm"],
                                -w / 2.0, w / 2.0)):
        parts.append(hp); names.append(f"hub_{j:02d}")
    for j, gp in enumerate(gear_teeth(px, py, P["shell_gear_teeth"], P["gear_module_mm"],
                                      gw, P["gear_backlash_mm"])):
        gp.apply_translation((0, 0, w / 2 + st + gw / 2))
        parts.append(gp); names.append(f"gear_{j:02d}")

    # 보울과 허브를 잇는 웹
    parts.append(box(abs(px - lx) * 0.4, wall, w,
                     center=((px + cx) / 2, (py + cy) / 2, 0)))
    names.append("web")
    return parts, names


def build_bracket(P):
    """고정 조 블레이드의 기존 4볼트 사각형에 물리는 브래킷 + 피벗 보스 2개."""
    k = kin(P)
    t, hd = P["bracket_thk_mm"], P["bolt_clear_d_mm"]
    dy, dz = P["bracket_bolt_dy_mm"], P["bracket_bolt_dz_mm"]
    parts, names = [], []
    so = P["bracket_standoff_mm"]
    # 볼트판: 블레이드에 닿는 면. 구멍은 **로컬 X** 를 따라 관통(hole_axis="x").
    # 사각형은 로컬 Y(=25.19 스팬) x 로컬 Z(=19.44 스팬) 이며 로컬 +Y 쪽, 즉 팔 쪽에 있다.
    for i, pc in enumerate(plate_with_holes(t, dz + 10.0, dy + 10.0,
                                            [-dz / 2, +dz / 2], hd,
                                            center=(0, so, 0), hole_axis="x")):
        parts.append(pc); names.append(f"bolt_plate_{i}")
    # 피벗 보스 2개 (셸이 도는 축)
    # 피벗 보스(환형): 이전에는 arc_segment(0, 2*pi) 한 조각이라 **부피 0** 이었다.
    bw = P["shell_width_mm"] + 2 * P["side_plate_thk_mm"] + 6
    for tag, (px, py) in (("L", k["pivot_L"]), ("R", k["pivot_R"])):
        for j, bp in enumerate(ring(px, py, P["pivot_shaft_d_mm"] / 2,
                                    P["pivot_boss_d_mm"] / 2, -bw / 2.0, bw / 2.0)):
            parts.append(bp); names.append(f"pivot_boss_{tag}_{j:02d}")
    # 스파인: 볼트판(로컬 +Y=standoff)에서 피벗선(로컬 Y=0)까지 뻗는 연장 팔.
    # 🔴 D463 정정: **통짜 판이면 고정 조 블레이드와 순정 가동 조를 뚫고 지나간다.**
    #    p37 G6 이 -0.383 mm(블레이드), G9 가 -0.597 mm(순정 조)로 잡았다. 이전 판정에서는
    #    같은 게이트가 볼트판 접촉(-0.787 mm)을 최소값으로 보고해 **이 관통이 가려져 있었다**.
    #    블레이드는 로컬 X [thk/2, thk/2 + 판두께], 로컬 Y >= (끝단 z 까지의 거리) 를 차지하고
    #    순정 조는 그보다 더 +X 쪽에 있다. 그래서 스파인을 둘로 나눈다:
    #      rail  = 블레이드보다 -X 쪽에서만 볼트판까지 올라가는 레일
    #      cross = 블레이드 끝보다 **더 나간 곳**(팔 바깥)에서 두 피벗을 잇는 가로대
    bx_lo = to_local(P, [P["link5_blade_x_mm"][0], 0.0, 0.0])[0]
    tip_y = to_local(P, [0.0, 0.0, P["link5_blade_tip_z_mm"]])[1]
    clr = 0.5
    rail_hi = bx_lo - clr
    rail_lo = -(k["g"] / 2.0 + P["pivot_boss_d_mm"] / 2.0)
    parts.append(box(rail_hi - rail_lo, so, t,
                     center=((rail_lo + rail_hi) / 2.0, so / 2.0, 0)))
    names.append("spine_rail")
    cross_y = tip_y - clr
    parts.append(box(k["g"] + P["pivot_boss_d_mm"], cross_y, t,
                     center=(0, cross_y / 2.0, 0)))
    names.append("spine_cross")
    return parts, names


def build_linkage(P):
    """서보 -> 셸 L 4절 링크 (D463). 크랭크 2 + 로드 1 + 핀 2.

    부품 이름 접두사가 **어느 몸체를 따라 도는지**를 정한다 (p37 스윕이 이걸 읽는다):
        servocrank_* / pin1_*   서보축 둘레로 +theta
        shellcrank_* / pin2_*   피벗 L 둘레로 -phi (셸 L 과 한 몸)
        rod_*                   두 핀을 따라가는 자유 강체

    힌지축(로컬 Z) 층 구성 — 셸 바깥면 -26.5 보다 바깥에 쌓아 셸·브래킷·팔을
    Z 로 비켜간다. 브래킷 피벗 보스는 Z +-29.5 까지 오므로 크랭크 팔은 피벗에서
    반경 (보스반경 + 0.25) 밖에서 시작한다.
    """
    k = kin(P)
    lk = linkage_solve(P)
    B = np.array(lk["pivot_L_xy"], float)
    Dm = np.array(lk["jaw_mount_xy"], float)
    P1 = np.array(lk["rows"][0]["pin1_xy"], float)
    P2 = np.array(lk["rows"][0]["pin2_xy"], float)

    ct = P["crank_thk_mm"]
    cz0, cz1 = P["crank_plane_z_mm"] - ct / 2.0, P["crank_plane_z_mm"] + ct / 2.0
    rt = P["rod_thk_mm"]
    rz0, rz1 = P["rod_plane_z_mm"] - rt / 2.0, P["rod_plane_z_mm"] + rt / 2.0
    r_pin = P["pin_d_mm"] / 2.0
    r_boss = P["pin_boss_d_mm"] / 2.0
    r_eye = P["pin_eye_d_mm"] / 2.0
    r_sleeve_in = P["pivot_boss_d_mm"] / 2.0 + P["pivot_bore_clear_mm"]   # 보스 위를 도는 슬리브
    r_sleeve_out = r_sleeve_in + 2.25
    shell_face_z = -(P["shell_width_mm"] / 2.0 + P["side_plate_thk_mm"])   # -26.5
    boss_face_z = -(P["shell_width_mm"] / 2.0 + P["side_plate_thk_mm"] + 3.0)  # -29.5

    parts, names = [], []

    def add(ms, tag):
        if not isinstance(ms, (list, tuple)):
            ms = [ms]
        for j, m in enumerate(ms):
            parts.append(m)
            names.append(tag if len(ms) == 1 else f"{tag}_{j:02d}")

    # ── 1. 서보 크랭크 ────────────────────────────────────────────────────
    #    순정 가동 조 블레이드 안쪽면(조 사이 간극 4.03 mm)에 판을 대고 기존
    #    볼트 구멍 2개(스팬 25.11 mm, 축 = link5 X)로 물린다. 서보 내부 형상을
    #    알 필요가 없다 (D462 §5).
    zs = [to_local(P, [0.0, h[0], h[1]])[2] for h in P["jaw_bolt_yz_mm"]]
    add(plate_holes_along_z(P["jaw_mount_thk_mm"], Dm[0], Dm[1],
                            P["jaw_mount_len_y_mm"],
                            P["jaw_mount_z_mm"][0], P["jaw_mount_z_mm"][1],
                            zs, P["bolt_clear_d_mm"]), "servocrank_plate")
    # 판(힌지축 안쪽) -> 링크 평면(바깥)으로 건너가는 웨브. Z 만 바뀌므로 서보가
    # 돌아도 Z 는 불변 -> link5(로컬 Z >= -17.0) 밖에 계속 머문다.
    web_z1 = -18.0                          # 판 쪽 끝 (link5 로컬 Z >= -17.0 바깥)
    add(box(P["jaw_mount_thk_mm"], P["jaw_mount_len_y_mm"], abs(web_z1 - cz0),
            center=(Dm[0], Dm[1], (cz0 + web_z1) / 2.0)), "servocrank_web")
    add(taper_xy(Dm, P1, P["crank_servo_w_mm"], P["crank_tip_w_mm"],
                 cz0, cz1, ext0=6.0, ext1=r_boss), "servocrank_arm")
    add(ring(P1[0], P1[1], r_pin, r_boss, cz0, cz1), "servocrank_eye")
    add(ring(P1[0], P1[1], 0.0, r_pin, rz0 - 0.5, cz1), "pin1")

    # ── 2. 로드 (자유 강체) ───────────────────────────────────────────────
    # 로드는 중실 막대가 아니라 **쌍레일**(평면상 H형)이다. 좌굴 약축은 힌지축
    # 방향(두께 4 mm)이므로 두께는 그대로 두고 폭 가운데를 비운다.
    #   I_약축 = 2*(rail_w * t^3/12), 오일러 임계하중 Pcr = pi^2*E*I/L^2
    #   PLA E=3.5 GPa, L=r_rod -> 아래 rod_buckling_* 에 실수치로 남긴다.
    er = (P2 - P1) / np.linalg.norm(P2 - P1)
    nr = np.array([-er[1], er[0]])
    off = (P["rod_width_mm"] - P["rod_rail_w_mm"]) / 2.0
    for sgn in (+1, -1):
        add(bar_xy(P1 + sgn * off * nr, P2 + sgn * off * nr,
                   P["rod_rail_w_mm"], rz0, rz1), f"rod_bar_{'ab'[sgn > 0]}")
    add(ring(P1[0], P1[1], r_pin, r_eye, rz0, rz1), "rod_eye1")
    add(ring(P2[0], P2[1], r_pin, r_eye, rz0, rz1), "rod_eye2")

    # ── 3. 셸 L 크랭크 (셸과 한 몸) ───────────────────────────────────────
    e = (P2 - B) / np.linalg.norm(P2 - B)
    # 슬리브는 보스 OD 위를 돈다. 허브(축까지 내려가는 부분)는 보스 **끝면**과
    # 0.5 mm 스러스트 여유를 둔다 — 안 두면 면끼리 딱 붙어 여유 0.0 이 된다.
    thrust = boss_face_z - 0.5
    add(ring(B[0], B[1], r_sleeve_in, r_sleeve_out, thrust, shell_face_z),
        "shellcrank_sleeve")
    add(ring(B[0], B[1], P["pivot_shaft_d_mm"] / 2.0, r_sleeve_out, cz0, thrust),
        "shellcrank_hub")
    add(taper_xy(B + r_sleeve_in * e, P2, P["crank_shell_w_mm"], P["crank_tip_w_mm"],
                 cz0, cz1, ext1=r_boss), "shellcrank_arm")
    add(ring(P2[0], P2[1], r_pin, r_boss, cz0, cz1), "shellcrank_eye")
    add(ring(P2[0], P2[1], 0.0, r_pin, rz0 - 0.5, cz1), "pin2")
    return parts, names, lk


# ─────────────────────────────────────────────────────────────────────────
# 5. 게이트 (D461 §6: 스칼라만으로는 공간 배치 오류를 못 잡는다 → 형상 게이트 포함)
# ─────────────────────────────────────────────────────────────────────────
def _rotz(pt_xy, ang):
    """로컬 Z(힌지축)에 평행하고 pt_xy 를 지나는 축 둘레 회전 4x4."""
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p = np.array([pt_xy[0], pt_xy[1], 0.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p - R @ p
    return T


def linkage_pose(P, lk, i):
    """표 i 행에서 (서보크랭크, 로드, 셸크랭크) 각각의 변환. 닫힘 형상에 곱한다."""
    rows = lk["rows"]
    A = np.array(lk["servo_axis_xy"], float)
    B = np.array(lk["pivot_L_xy"], float)
    Ts = _rotz(A, math.radians(rows[i]["servo_deg"]))
    Tk = _rotz(B, -math.radians(rows[i]["shell_deg"]))
    p10 = np.array(rows[0]["pin1_xy"], float)
    p20 = np.array(rows[0]["pin2_xy"], float)
    p1i = np.array(rows[i]["pin1_xy"], float)
    p2i = np.array(rows[i]["pin2_xy"], float)
    Tr = _rotz(p10, math.atan2(*(p2i - p1i)[::-1]) - math.atan2(*(p20 - p10)[::-1]))
    Tr[0, 3] += p1i[0] - p10[0]
    Tr[1, 3] += p1i[1] - p10[1]
    return Ts, Tr, Tk


def linkage_group(name):
    """부품 이름 -> 따라 도는 몸체."""
    if name.startswith("servocrank") or name.startswith("pin1"):
        return "servocrank"
    if name.startswith("shellcrank") or name.startswith("pin2"):
        return "shellcrank"
    return "rod"


def _aabb_gap(a, b):
    """두 볼록 조각의 AABB 분리거리. 양수면 확실히 떨어져 있다(정확한 하한)."""
    alo, ahi = a.bounds
    blo, bhi = b.bounds
    return float(max((blo - ahi).max(), (alo - bhi).max()))


def _cloud_of(pieces, names, tag, T=None, n=160):
    """조각 집합의 표면을 점으로 깔고 (점, 조각이름) 을 함께 돌려준다."""
    pts, own = [], []
    for m0, nm in zip(pieces, names):
        m = m0 if T is None else m0.copy()
        if T is not None:
            m.apply_transform(T)
        s_, _ = trimesh.sample.sample_surface(m, n)
        pts.append(np.vstack([np.asarray(m.vertices, float), s_]))
        own += [f"{tag}:{nm}"] * len(pts[-1])
    return np.vstack(pts), np.array(own, dtype=object)


def _clearance_to_cloud(piece, cloud, owner, margin=2.5):
    """볼록·watertight 조각 <-> 점구름의 여유(mm). 음수면 관통 깊이.

    🔴 왜 점구름인가: 초판은 조각쌍 AABB 분리거리를 **그대로 값으로 보고**했다.
       AABB 는 축정렬 하한이라 비스듬한 두 조각에서 극단적으로 작아진다 — 실제로
       `shellcrank_arm` <-> `pivot_boss_L_08` 이 참값 2.181 mm 인데 0.002 mm 로
       보고돼 게이트가 헛되이 FAIL 했다 (p37 의 복셀 하한과 같은 함정: **하한을
       값으로 읽지 말 것**). 조각이 볼록·watertight 이므로 상대 표면점의 부호거리가
       정확하다. 조각쌍 전수 대신 점구름 1회 질의라 비용도 O(조각수) 로 떨어진다.
    """
    lo, hi = piece.bounds
    sel = np.all((cloud >= lo - margin) & (cloud <= hi + margin), axis=1)
    if not sel.any():
        d = np.maximum(lo - cloud, 0.0) + np.maximum(cloud - hi, 0.0)
        i = int(np.linalg.norm(d, axis=1).argmin())
        return float(np.linalg.norm(d[i])), owner[i]
    sd = piece.nearest.signed_distance(cloud[sel])
    i = int(sd.argmax())
    return float(-sd[i]), owner[sel][i]


def _linkage_sweep_clearance(P, lk, link, nmK, shellL, nmL, shellR, nmR,
                             bracket, nmB, n_samp=9):
    """서보 0~89도 스윕 전 구간에서 링크 <-> 셸/브래킷 최소 여유.

    제외 규칙: `shellcrank_*`/`pin2` vs 셸 L 은 **같은 강체**다 (셸 L 과 한 몸으로
    인쇄되고 함께 돈다). 나머지는 전부 잰다.
    """
    k = kin(P)
    rows = lk["rows"]
    idx = np.linspace(0, len(rows) - 1, n_samp).astype(int)
    br_cloud, br_own = _cloud_of(bracket, nmB, "bracket")
    worst, who, at = float("inf"), None, None
    for i in idx:
        Ts, Tr, Tk = linkage_pose(P, lk, int(i))
        phi = math.radians(rows[int(i)]["shell_deg"])
        cL, oL = _cloud_of(shellL, nmL, "shellL", _rotz(k["pivot_L"], -phi))
        cR, oR = _cloud_of(shellR, nmR, "shellR", _rotz(k["pivot_R"], +phi))
        with_L = (np.vstack([br_cloud, cL, cR]), np.concatenate([br_own, oL, oR]))
        no_L = (np.vstack([br_cloud, cR]), np.concatenate([br_own, oR]))
        for m0, nm in zip(link, nmK):
            grp = linkage_group(nm)
            m = m0.copy()
            m.apply_transform({"servocrank": Ts, "rod": Tr, "shellcrank": Tk}[grp])
            cloud, owner = no_L if grp == "shellcrank" else with_L
            v, w = _clearance_to_cloud(m, cloud, owner)
            if v < worst:
                worst, who, at = v, f"{nm} <-> {w}", float(rows[int(i)]["servo_deg"])
    return worst, who, at


def _linkage_structure(P, lk):
    """링크 경량화의 근거 수치. 모두 **서보 실속 토크**(최악) 기준이다.

    ⚠️ D462 §6: 실측 조 힘은 1.8~6.3 N 이었다. 실속은 실제로 도달하지 않는 상한이므로
       여기 안전율은 보수적이다. 그래도 근거 없이 단면을 깎지 않기 위해 남긴다.
    """
    E_PLA = 3500.0                      # MPa, FDM PLA 굽힘탄성률 보수값
    T = P["servo_torque_nm"] * 1000.0   # N*mm
    r2 = lk["crank_servo_r_mm"]
    mu_in = math.radians(lk["trans_angle_in_min_deg"])
    f_rod = T / (r2 * math.sin(mu_in))          # 입력 전달각 최악에서 로드 축력
    t, rw = P["rod_thk_mm"], P["rod_rail_w_mm"]
    I_weak = 2.0 * rw * t ** 3 / 12.0           # 약축 = 힌지축 방향
    p_cr = math.pi ** 2 * E_PLA * I_weak / lk["rod_len_mm"] ** 2
    gain = max(r["torque_gain"] for r in lk["rows"])
    z_shell = P["crank_thk_mm"] * P["crank_shell_w_mm"] ** 2 / 6.0
    z_servo = P["crank_thk_mm"] * P["crank_servo_w_mm"] ** 2 / 6.0
    return {
        "servo_stall_torque_Nmm": round(T, 1),
        "rod_axial_force_N_at_stall": round(f_rod, 1),
        "rod_section_mm": [rw * 2, t], "rod_I_weak_mm4": round(I_weak, 2),
        "rod_euler_Pcr_N": round(p_cr, 1),
        "rod_buckling_SF": round(p_cr / f_rod, 2),
        "pin_bearing_MPa": round(f_rod / (P["pin_d_mm"] * P["crank_thk_mm"]), 1),
        "pin_double_shear_MPa": round(f_rod / (2 * math.pi * (P["pin_d_mm"] / 2) ** 2), 1),
        "shell_crank_root_bending_MPa": round(T * gain / z_shell, 1),
        "servo_crank_root_bending_MPa": round(f_rod * 10.9 / z_servo, 1),
        "PLA_yield_MPa_ref": 50.0,
        "note": ("로드는 압축·인장을 받는다. 약축 좌굴이 지배하므로 두께(힌지축 방향)는 "
                 "유지하고 폭 가운데만 비웠다(쌍레일). 크랭크는 뿌리에서 끝으로 모멘트가 "
                 "선형 감소하므로 폭을 테이퍼했고 **뿌리 단면은 그대로**다")}


def run_gates(P, shellL, shellR, bracket, link, lk, nmL, nmR, nmB, nmK):
    k = kin(P)
    g = {}
    allp = shellL + shellR + bracket + link
    allnm = nmL + nmR + nmB + nmK
    # 🔴 정정: 초판은 `m.volume > 1e-9` 로 **부피 0 조각을 검사에서 건너뛰었다**.
    #    그래서 인쇄하면 아무것도 안 나오는 납작한 조각 8개(hub, pivot_boss_L/R,
    #    기어 뿌리원반 x4)가 그대로 PASS 했다. 볼록성 게이트가 퇴화 기하를 못 본 것이다
    #    (D461 §6 계열 — 게이트가 무엇을 못 보는지 먼저 물어야 한다).
    #    이제 부피 0 은 위반으로 센다.
    zero = [n for m, n in zip(allp, allnm) if m.volume <= 1e-9]
    nonconvex = [n for m, n in zip(allp, allnm)
                 if m.volume > 1e-9
                 and abs(m.volume - m.convex_hull.volume) / m.convex_hull.volume > 1e-6]
    g["all_pieces_convex"] = {
        "pass": not nonconvex and not zero,
        "violations_nonconvex": nonconvex[:8], "violations_zero_volume": zero[:8],
        "n_pieces": len(allp), "n_zero_volume": len(zero),
        "note": ("부피 0 조각도 위반이다. 볼록 판정식 |V - V_hull|/V_hull 은 V=0 에서 "
                 "0/0 이라 무의미하므로 예전엔 건너뛰었고, 그 결과 퇴화 조각을 놓쳤다")}
    # 별도 게이트로도 분리해 둔다 — 원인이 다르므로 실패 시 진단이 갈린다
    g["pieces_have_positive_volume"] = {
        "pass": not zero, "n_zero_volume": len(zero), "zero_volume_pieces": zero[:16],
        "n_pieces": len(allp),
        "min_piece_volume_mm3": round(float(min(m.volume for m in allp)), 6),
        "why": ("부피 0 = 인쇄하면 아무것도 안 나오는 조각. arc_segment 에 a0=0/a1=2pi "
                "(또는 0/pi)를 주면 정점이 4개뿐이라 납작해지고, 환형은 애초에 볼록이 "
                "아니라 한 조각으로 표현할 수 없다. 전체원·환형·원반은 ring() 으로 "
                "쐐기 분해할 것")}

    # 형상 ①: 닫힘 시 두 셸이 서로 파고들지 않고 립에서 만난다
    # ⚠️ 기어는 제외한다. **맞물리는 기어는 이끝원이 반드시 겹친다**
    #    (2*addendum - center_distance = 2.0 mm). 이빨이 교대로 지나가는 공간이며
    #    충돌이 아니다. 초판 게이트가 이걸 FAIL 로 셌다 = 게이트 오탐(D459 §8 계열).
    bodyL = [p for p, n in zip(shellL, nmL) if not n.startswith("gear")]
    bodyR = [p for p, n in zip(shellR, nmR) if not n.startswith("gear")]
    L = trimesh.util.concatenate(bodyL); R = trimesh.util.concatenate(bodyR)
    ov = min(L.bounds[1][0], R.bounds[1][0]) - max(L.bounds[0][0], R.bounds[0][0])
    g["shells_meet_not_overlap"] = {
        "pass": -1.5 <= ov <= 0.30, "x_overlap_mm": round(float(ov), 3),
        "band_mm": [-1.5, 0.30],
        "note": "0 근처 = 립이 맞닿음. 크게 양수면 서로 파고든 것"}

    # 형상 ②: 좌우 셸이 X 거울 대칭인가 (대칭 폐합의 정의)
    Lm = L.copy(); Lm.apply_scale((-1, 1, 1))
    sym = float(np.max(np.abs(np.sort(Lm.bounds, axis=0) - np.sort(R.bounds, axis=0))))
    g["shells_are_mirror_symmetric"] = {"pass": sym < 0.05, "max_bound_diff_mm": round(sym, 4)}

    # 형상 ③: 셸 기어 피치원이 접한다 (맞물려야 반대로 돈다)
    r_s = P["gear_module_mm"] * P["shell_gear_teeth"] / 2.0
    g["shell_gears_mesh"] = {"pass": abs(2 * r_s - k["g"]) < 1e-6,
                             "sum_pitch_r_mm": 2 * r_s, "pivot_gap_mm": k["g"]}
    # 형상 ③-b: 기어 이가 실제로 물리되 뿌리까지 박히지는 않는가
    mod = P["gear_module_mm"]
    r_a = r_s + mod                      # 이끝원
    r_f = r_s - 1.25 * mod               # 이뿌리원
    tip_root_clear = k["g"] - (r_a + r_f)   # 상대 이끝 <-> 내 이뿌리 여유
    g["gear_tip_root_clearance"] = {
        "pass": 0.05 <= tip_root_clear <= 1.0,
        "value_mm": round(tip_root_clear, 3), "band_mm": [0.05, 1.0],
        "tip_overlap_mm": round(2 * r_a - k["g"], 3),
        "note": "이끝원 겹침 2.0 mm 는 맞물림의 정의다. 여기서 보는 것은 이뿌리 여유"}

    # 형상 ④: 구동 감속비가 목표와 같다
    #   ⚠️ D463 로 구동이 기어 직결 -> 4절 링크가 되면서 **이 게이트의 계산식이
    #      바뀌었다**. 예전엔 잇수비(26/13)라는 파라미터 항등식이었고, 지금은
    #      실제로 푼 링크 궤적의 **끝점 총행정비**다. 이름·목표값(2.0)·판정
    #      엄격도는 그대로이고, 파라미터가 아니라 기구학 해를 보므로 오히려 강해졌다.
    servo_end = lk["rows"][-1]["servo_deg"]
    shell_end = lk["rows"][-1]["shell_deg"]
    ratio = servo_end / shell_end if shell_end else float("inf")
    g["drive_ratio_matches"] = {"pass": abs(ratio - P["gear_ratio"]) < 1e-3,
                                "ratio": round(ratio, 6), "target": P["gear_ratio"],
                                "basis": "linkage_end_to_end_travel",
                                "servo_travel_deg": servo_end,
                                "shell_travel_deg": round(shell_end, 4),
                                "ground_len_mm": lk["ground_len_mm"],
                                "note": ("기어 직결 시절엔 26/13 잇수비였다. 링크에서는 "
                                         "순간비가 각도마다 다르므로 총행정비로 잰다 "
                                         "(순간비 표 = linkage.table 의 dshell_dservo)")}

    # 기구학 ⑤: 셸 회전각에서 개구가 목표와 같다
    m = mouth_at(P, P["shell_travel_deg"])
    g["mouth_open_at_travel"] = {"pass": abs(m - P["mouth_open_mm"]) < 0.25,
                                 "value_mm": round(m, 3), "target_mm": P["mouth_open_mm"],
                                 "closed_mm": round(mouth_at(P, 0.0), 4)}

    # ── 링크 ⑫~⑭ (D463 신규) ───────────────────────────────────────────
    rows = lk["rows"]
    lo_mu, hi_mu = P["trans_angle_band_deg"]
    mus = [r["trans_angle_deg"] for r in rows]          # 출력측: 로드 <-> 셸 크랭크
    mis = [r["trans_angle_in_deg"] for r in rows]       # 입력측: 서보 크랭크 <-> 로드
    i_lo, i_hi = int(np.argmin(mus)), int(np.argmax(mus))
    j_lo, j_hi = int(np.argmin(mis)), int(np.argmax(mis))
    margin = min(min(mus) - lo_mu, hi_mu - max(mus),
                 min(mis) - lo_mu, hi_mu - max(mis))
    g["linkage_no_dead_center"] = {
        "pass": bool(margin >= 0.0),
        "band_deg": [lo_mu, hi_mu], "worst_margin_deg": round(float(margin), 3),
        "out_min_deg": round(min(mus), 3), "out_max_deg": round(max(mus), 3),
        "out_min_at_servo_deg": rows[i_lo]["servo_deg"],
        "out_max_at_servo_deg": rows[i_hi]["servo_deg"],
        "in_min_deg": round(min(mis), 3), "in_max_deg": round(max(mis), 3),
        "in_min_at_servo_deg": rows[j_lo]["servo_deg"],
        "in_max_at_servo_deg": rows[j_hi]["servo_deg"],
        "why": ("**두 각을 다 본다.** 출력측 전달각(로드 <-> 셸 크랭크)이 0/180도면 "
                "셸이 로드를 못 밀고, 입력측 각(서보 크랭크 <-> 로드)이 0/180도면 "
                "입력 사점(토글)이라 서보가 아무리 돌아도 셸이 안 움직인다(자기잠김). "
                "출력측만 보면 입력 사점을 놓친다 — 첫 후보가 실제로 그랬다"
                "(servo 0도에서 크랭크와 로드가 0.1도 차이로 일직선, "
                "dshell/dservo = 0.007)")}

    servo_travel = float(P["servo_travel_deg"])
    shell_travel = float(P["shell_travel_deg"])
    dphi = np.diff([r["shell_deg"] for r in rows])
    mouth_end = rows[-1]["mouth_mm"]
    ok13 = (abs(rows[0]["shell_deg"]) < 1e-6
            and abs(rows[-1]["shell_deg"] - shell_travel) < 0.02
            and abs(rows[-1]["servo_deg"] - servo_travel) < 1e-6
            and bool(np.all(dphi > 0.0))
            and abs(mouth_end - P["mouth_open_mm"]) < 0.25)
    g["linkage_reaches_full_travel"] = {
        "pass": ok13,
        "servo_deg_range": [rows[0]["servo_deg"], rows[-1]["servo_deg"]],
        "shell_deg_range": [rows[0]["shell_deg"], rows[-1]["shell_deg"]],
        "shell_target_deg": shell_travel,
        "mouth_mm_range": [rows[0]["mouth_mm"], mouth_end],
        "monotonic": bool(np.all(dphi > 0.0)),
        "min_step_deg": round(float(dphi.min()), 5),
        "dshell_dservo_range": [round(float(min(r["dshell_dservo"] for r in rows)), 4),
                                round(float(max(r["dshell_dservo"] for r in rows)), 4)],
        "why": ("서보 0~89도가 셸 0~44.5도(개구 0~58 mm)를 실제로 내는가. 끝점은 "
                "폐합 방정식이 보장하지만 사이 구간의 단조성은 보장하지 않는다")}

    # ⑭ 로드가 셸·브래킷과 안 닿는가. 서보 스윕 전 구간에서 링크를 실제로 옮겨 잰다
    #    (팔 = link5 와의 여유는 link5 메쉬가 필요하므로 p37 G6/G7 이 맡는다)
    clr, who, at_deg = _linkage_sweep_clearance(P, lk, link, nmK, shellL, nmL,
                                                shellR, nmR, bracket, nmB)
    g["linkage_rod_clears_shells"] = {
        "pass": clr >= 0.5, "min_clearance_mm": round(clr, 3), "limit_mm": 0.5,
        "closest_pair": who, "closest_at_servo_deg": at_deg,
        "why": ("로드·크랭크는 힌지축 방향으로 셸 바깥면(-26.5)보다 밖에 층을 이루지만 "
                "브래킷 피벗 보스는 -29.5 까지 나오므로 실제로 스윕해서 재야 한다")}

    # 수치 ⑥~⑨
    vol_mm3 = sum(p.volume for p in allp)
    mass = vol_mm3 * P["density_g_cm3"] / 1000.0
    g["tool_mass_under_max"] = {"pass": mass <= P["tool_mass_max_g"],
                                "value_g": round(mass, 2), "limit_g": P["tool_mass_max_g"],
                                "material": P["material"]}
    # 닫힘 내부 체적 = 두 보울 안쪽 폴리곤 면적 x 너비
    from math import cos, sin
    polys = []
    for side in (-1, +1):
        (cx, cy), r, a1, a2, _ = bowl_arc(P, side)
        t = np.linspace(a1, a2, 80)
        polys.append(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1))
    poly = np.vstack([polys[0], polys[1][::-1]])
    A = abs(0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                         - np.roll(poly[:, 0], -1) * poly[:, 1]))
    inner_cm3 = A * P["shell_width_mm"] / 1000.0
    load_g = inner_cm3 * P["fill_factor"] * P["bulk_density_g_cm3"]
    ratio_sl = mass / load_g if load_g > 0 else float("inf")
    lo, hi = P["self_load_ratio_band"]
    # ⚠️ 이 게이트의 분모(적재량)는 **아직 측정되지 않은** bulk_density 로 계산된다.
    #    형상을 더 깎아 이 숫자를 맞추는 것은 추측값에 실물을 맞추는 것이므로 하지 않는다.
    #    대신 실측 여부 자체를 막아 세운다 (h1 의 closed_gap_measured_on_hardware 와 같은 방식).
    rho_needed_hi = ratio_sl * P["bulk_density_g_cm3"] / hi   # 대역 상한을 만족시킬 밀도
    rho_needed_lo = ratio_sl * P["bulk_density_g_cm3"] / lo
    measured = not str(P.get("bulk_density_src", "ASSUMED")).startswith("ASSUMED")
    g["self_load_ratio_in_real_band"] = {
        "pass": bool(measured and lo <= ratio_sl <= hi),
        "value_with_assumed_density": round(ratio_sl, 3), "band": [lo, hi],
        "v0_was": 3.3,
        "blocked_on": "bulk_density_g_cm3 실측 (펠릿 미조달)",
        "bulk_density_used": P["bulk_density_g_cm3"],
        "bulk_density_src": P.get("bulk_density_src", "ASSUMED"),
        "density_range_that_would_pass_g_cm3": [round(rho_needed_hi, 3), round(rho_needed_lo, 3)],
        "note": ("실물 그랩은 자기 무게만으로 관입한다(D457 §12). v0는 팔이 눌러야 했다. "
                 "분모가 추측이므로 이 게이트는 실측 전까지 FAIL 을 유지한다 — "
                 "형상 튜닝으로 통과시키지 말 것")}
    g["particle_ratio_min10"] = {"pass": P["mouth_open_mm"] / P["pellet_d_max_mm"] >= 10.0,
                                 "value": round(P["mouth_open_mm"] / P["pellet_d_max_mm"], 2)}
    outer_w = k["g"] + 2 * P["bulge_frac"] * k["d"]
    g["grab_to_pile_width"] = {"pass": 0.03 <= outer_w / P["pile_width_mm"] <= 0.20,
                               "value": round(outer_w / P["pile_width_mm"], 4),
                               "industrial_band": [0.05, 0.10], "v0_was": 0.53}
    derived = {"inner_volume_cm3": round(inner_cm3, 2), "load_per_scoop_g": round(load_g, 2),
               "tool_mass_g": round(mass, 2), "self_load_ratio": round(ratio_sl, 3),
               "lip_pivot_radius_mm": round(k["R"], 3),
               "shell_gear_center_distance_mm": round(k["g"], 3),
               "outer_width_mm": round(outer_w, 2),
               "linkage_mass_g": round(sum(p.volume for p in link)
                                       * P["density_g_cm3"] / 1000.0, 2),
               # 퇴화 정정으로 **새로 실체를 갖게 된** 조각들의 질량 (전에는 0 이었다)
               "degenerate_repair_mass_g": round(
                   sum(m.volume for m, n in zip(allp, allnm)
                       if n.startswith("hub_") or n.startswith("pivot_boss_")
                       or (n.startswith("gear_") and int(n.split("_")[1]) < RING_SEG_N))
                   * P["density_g_cm3"] / 1000.0, 2),
               "tool_z_extent_mm": [round(float(min(p.bounds[0][2] for p in allp)), 2),
                                    round(float(max(p.bounds[1][2] for p in allp)), 2)],
               "servo_axis_local_mm": [round(float(v), 4)
                                       for v in servo_axis_local(P)],
               "servo_to_pivotL_perp_mm": lk["ground_len_mm"],
               "linkage_structure": _linkage_structure(P, lk)}
    return g, derived



def _jsafe(o):
    """numpy 스칼라를 JSON 가능 타입으로."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


# ─────────────────────────────────────────────────────────────────────────
# 6. main
# ─────────────────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ⚠️ 생성기는 출력 폴더를 비우지 않는다 (D463 §5-4: 구버전 조각 14개 잔류 사례).
    #    같은 폴더를 재사용하면 이전 이름의 STL 이 남으므로 먼저 지운다.
    for old in OUT.glob("*.stl"):
        old.unlink()
    sL, nL = build_shell(P, -1)
    sR, nR = build_shell(P, +1)
    br, nB = build_bracket(P)
    lkp, nK, lk = build_linkage(P)
    gates, derived = run_gates(P, sL, sR, br, lkp, lk, nL, nR, nB, nK)

    for parts, names, tag in ((sL, nL, "shell_L"), (sR, nR, "shell_R"),
                              (br, nB, "bracket"), (lkp, nK, "linkage")):
        for m, n in zip(parts, names):
            m.export(OUT / f"{tag}_{n}.stl")
        trimesh.util.concatenate(parts).export(OUT / f"{tag}_ALL.stl")

    for _v in gates.values():
        _v["pass"] = bool(_v["pass"])
    ok = all(v["pass"] for v in gates.values())
    cols = ("servo_deg", "shell_deg", "mouth_mm", "trans_angle_deg",
            "trans_angle_in_deg", "dshell_dservo", "torque_gain")
    table = [{kk: r[kk] for kk in cols} for r in lk["rows"]]
    json.dump({"params": P, "gates": gates, "derived": derived,
               "all_gates_pass": ok,
               "linkage": {kk: vv for kk, vv in lk.items() if kk != "rows"} |
                          {"table_columns": list(cols),
                           "table": table,
                           "note": ("개구는 서보각에 비례하지 않는다 (D463). "
                                    "torque_gain = 1/dshell_dservo — 1 보다 크면 "
                                    "그 구간에서 서보 토크가 증폭된다")},
               "piece_counts": {"shell_L": len(sL), "shell_R": len(sR),
                                "bracket": len(br), "linkage": len(lkp)},
               "measure_pending": ["bulk_density_g_cm3"],
               "degenerate_geometry_repair": {
                   "what": "부피 0 으로 퇴화했던 조각 8개를 쐐기 분해로 실체화",
                   "was": ["hub", "gear_00", "gear_01", "(셸 L·R 각각)",
                           "pivot_boss_L", "pivot_boss_R"],
                   "cause": ("arc_segment(...,0,2*pi,...) 는 두 끝각의 좌표가 같아 정점이 "
                             "4개뿐이고, (0,pi)/(pi,2*pi) 는 4점이 한 평면에 놓인다. "
                             "게다가 환형은 정의상 볼록이 아니라 한 조각으로 표현 불가"),
                   "fix": f"ring() 으로 쐐기 {RING_SEG_N} 개 분해 (보울 seg_n 분해와 같은 방식)",
                   "gate": ("all_pieces_convex 가 부피 0 을 위반으로 세도록 정정 + "
                            "pieces_have_positive_volume 게이트 신설"),
                   "mass_effect_g": derived["degenerate_repair_mass_g"]},
               "supersedes": ("scoop_shell_design.py (v0 오버레이, D462) / "
                              "기어 직결 구동 (D462 §7 -> D463 이 링크로 대체)")},
              open(OUT / "design.json", "w"), ensure_ascii=False, indent=2,
              default=_jsafe)

    print(f"조각 수  셸L {len(sL)} · 셸R {len(sR)} · 브래킷 {len(br)} · 링크 {len(lkp)}")
    print(f"링크  크랭크(서보) {lk['crank_servo_r_mm']} · 로드 {lk['rod_len_mm']}"
          f" · 크랭크(셸) {lk['crank_shell_r_mm']} mm")
    print(f"전달각  출력 {lk['trans_angle_out_min_deg']}~{lk['trans_angle_out_max_deg']}"
          f" · 입력 {lk['trans_angle_in_min_deg']}~{lk['trans_angle_in_max_deg']} 도")
    print(f"툴 자중 {derived['tool_mass_g']} g · 적재 {derived['load_per_scoop_g']} g"
          f" · 자중/적재 {derived['self_load_ratio']}")
    print()
    for kk, v in gates.items():
        print(("  PASS  " if v["pass"] else "  FAIL  ") + kk + "  " +
              json.dumps({a: b for a, b in v.items() if a != "pass"}, ensure_ascii=False, default=_jsafe)[:120])
    print(f"\nall_gates_pass = {ok}   -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
