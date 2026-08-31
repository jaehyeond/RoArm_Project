"""조개형 그랩 v1 — 대칭 2피벗 + 1:1 기어쌍 파라메트릭 생성기 (D462).

v0(`scoop_shell_design.py`)와의 차이
------------------------------------
v0 = 순정 2-jaw 의 **각 조에 셸을 하나씩 볼트로 덧댐**. 한쪽 조만 회전하므로
     대칭 클램셸이 아니었고, 폐합 여유가 순정 판재 간격 4.05 mm 에 갇혀 0.05 mm 였다.
v1 = **앞부분 교체**. 순정 가동 조를 떼고, 고정 조 블레이드의 기존 4볼트 사각형
     (25.19 x 19.44 mm, D462 §1)에 브래킷을 물려 **피벗 2개를 자체 설계**한다.
     두 셸 기어가 직접 맞물려 **대칭 폐합**하고, 구동 기어가 서보에서 **2:1 감속**한다.

왜 2:1 인가 (D462 g1 스윕)
--------------------------
서보 89도를 셸 44.5도로 줄이면, 같은 개구 58 mm 를 내기 위해 립이 더 멀리 있어야 한다
(16 -> 36 mm). 립이 길어지면 보울이 깊어져 적재가 늘고, 감속만큼 폐합 토크도 2배가 된다.
    자중/적재  3.3 (v0)  ->  0.76 (v1)     실물 그랩 0.72~0.88 (D457 §12) 대역
D446 준수: 보울·기어·판 전부 **볼록 조각으로 분해**한다. 1-hull 금지.

사용:  python scoop_grab_v1_design.py [출력디렉터리]
"""
import sys, json, math
from pathlib import Path
import numpy as np
import trimesh

OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else "claudedocs/runtime_logs/scoop_grab_v1")

# ─────────────────────────────────────────────────────────────────────────
# 1. 파라미터
# ─────────────────────────────────────────────────────────────────────────
P = {
    # ── 기구 (D462 g1 스윕에서 선정) ─────────────────────────────────────
    "pivot_gap_mm":       26.0,   # 두 피벗 간격. 셸 기어 피치원이 여기서 접한다
    "lip_depth_mm":       36.06,  # 닫힘 시 립이 피벗선 아래로 내려간 깊이
    "mouth_open_mm":      58.0,   # 완전개방 립-투-립 (D457 §12 유지)
    "shell_travel_deg":   44.5,   # 셸 편측 회전 = 서보 89도 / 2
    "gear_ratio":          2.0,   # 서보 -> 셸 감속비
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
    "drive_gear_teeth":   13,     # 피치 ⌀13 -> 26/13 = 2:1 감속
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
    "pivot_boss_d_mm":     8.0,

    # ── 구동 암 (순정 가동 조가 쓰던 볼트 구멍에서 인출) ─────────────────
    "drive_bolt_span_mm": 25.0,   # D460 §4 실측
    "drive_arm_thk_mm":    3.0,

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


def gear_teeth(cx, cy, n_teeth, module, width, backlash=0.0, a_start=0.0, a_span=None):
    """스퍼 기어 이빨을 **볼록 쐐기**로 생성. 인볼류트가 아니라 사다리꼴 근사다.

    저속·저부하·왕복 44.5도 용도라 사다리꼴로 충분하며, 인볼류트를 쓰면 조각이
    비볼록이 되어 D446 등식이 깨진다. 백래시는 이 두께를 깎아 만든다.
    """
    r_p = module * n_teeth / 2.0          # 피치원
    r_a = r_p + module                    # 이끝원
    r_f = r_p - 1.25 * module             # 이뿌리원
    pieces = [arc_segment(cx, cy, 0.0, r_f, 0.0, math.pi, width),
              arc_segment(cx, cy, 0.0, r_f, math.pi, 2 * math.pi, width)]
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
    parts.append(arc_segment(px, py, P["pivot_shaft_d_mm"] / 2,
                             P["pivot_boss_d_mm"] / 2, 0, 2 * math.pi, w))
    names.append("hub")
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
    for tag, (px, py) in (("L", k["pivot_L"]), ("R", k["pivot_R"])):
        b = arc_segment(px, py, P["pivot_shaft_d_mm"] / 2, P["pivot_boss_d_mm"] / 2,
                        0, 2 * math.pi, P["shell_width_mm"] + 2 * P["side_plate_thk_mm"] + 6)
        parts.append(b); names.append(f"pivot_boss_{tag}")
    # 스파인: 볼트판(로컬 +Y=standoff)에서 피벗선(로컬 Y=0)까지 뻗는 연장 팔
    parts.append(box(k["g"] + P["pivot_boss_d_mm"], so, t,
                     center=(0, so / 2, 0)))
    names.append("spine")
    return parts, names


def build_drive(P):
    """구동 기어 + 순정 가동 조 볼트 구멍(스팬 25 mm)에서 힘을 받는 암."""
    k = kin(P)
    m, gw = P["gear_module_mm"], P["gear_width_mm"]
    r_drive = m * P["drive_gear_teeth"] / 2.0
    r_shell = m * P["shell_gear_teeth"] / 2.0
    cd = r_drive + r_shell                      # 축간거리
    dx, dy = k["pivot_L"][0], k["pivot_L"][1] + cd
    parts, names = [], []
    for j, gp in enumerate(gear_teeth(dx, dy, P["drive_gear_teeth"], m, gw,
                                      P["gear_backlash_mm"])):
        parts.append(gp); names.append(f"drive_gear_{j:02d}")
    span = P["drive_bolt_span_mm"]
    for i, pc in enumerate(plate_with_holes(10.0, span + 10.0, P["drive_arm_thk_mm"],
                                            [-span / 2, +span / 2], P["bolt_clear_d_mm"],
                                            center=(dx, dy, gw / 2 + P["drive_arm_thk_mm"] / 2))):
        parts.append(pc); names.append(f"drive_arm_{i}")
    return parts, names, cd


# ─────────────────────────────────────────────────────────────────────────
# 5. 게이트 (D461 §6: 스칼라만으로는 공간 배치 오류를 못 잡는다 → 형상 게이트 포함)
# ─────────────────────────────────────────────────────────────────────────
def run_gates(P, shellL, shellR, bracket, drive, cd, nmL, nmR):
    k = kin(P)
    g = {}
    allp = shellL + shellR + bracket + drive
    bad = [i for i, m in enumerate(allp)
           if m.volume > 1e-9 and abs(m.volume - m.convex_hull.volume) / m.convex_hull.volume > 1e-6]
    g["all_pieces_convex"] = {"pass": not bad, "violations": bad[:8], "n_pieces": len(allp)}

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
    ratio = P["shell_gear_teeth"] / P["drive_gear_teeth"]
    g["drive_ratio_matches"] = {"pass": abs(ratio - P["gear_ratio"]) < 1e-9,
                                "ratio": ratio, "target": P["gear_ratio"],
                                "center_distance_mm": round(cd, 3)}

    # 기구학 ⑤: 셸 회전각에서 개구가 목표와 같다
    m = mouth_at(P, P["shell_travel_deg"])
    g["mouth_open_at_travel"] = {"pass": abs(m - P["mouth_open_mm"]) < 0.25,
                                 "value_mm": round(m, 3), "target_mm": P["mouth_open_mm"],
                                 "closed_mm": round(mouth_at(P, 0.0), 4)}

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
               "gear_center_distance_mm": round(cd, 3),
               "outer_width_mm": round(outer_w, 2)}
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
    sL, nL = build_shell(P, -1)
    sR, nR = build_shell(P, +1)
    br, nB = build_bracket(P)
    dr, nD, cd = build_drive(P)
    gates, derived = run_gates(P, sL, sR, br, dr, cd, nL, nR)

    for parts, names, tag in ((sL, nL, "shell_L"), (sR, nR, "shell_R"),
                              (br, nB, "bracket"), (dr, nD, "drive")):
        for m, n in zip(parts, names):
            m.export(OUT / f"{tag}_{n}.stl")
        trimesh.util.concatenate(parts).export(OUT / f"{tag}_ALL.stl")

    for _v in gates.values():
        _v["pass"] = bool(_v["pass"])
    ok = all(v["pass"] for v in gates.values())
    json.dump({"params": P, "gates": gates, "derived": derived,
               "all_gates_pass": ok,
               "piece_counts": {"shell_L": len(sL), "shell_R": len(sR),
                                "bracket": len(br), "drive": len(dr)},
               "measure_pending": ["bulk_density_g_cm3"],
               "supersedes": "scoop_shell_design.py (v0 오버레이 방식, D462로 전환)"},
              open(OUT / "design.json", "w"), ensure_ascii=False, indent=2,
              default=_jsafe)

    print(f"조각 수  셸L {len(sL)} · 셸R {len(sR)} · 브래킷 {len(br)} · 구동 {len(dr)}")
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
