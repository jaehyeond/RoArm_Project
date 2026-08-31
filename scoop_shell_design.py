"""조개형 그랩 셸 v0 — 파라메트릭 생성기 (D457 §12 치수 + D458 실측 반영).

RoArm-M3-Pro 순정 2-jaw 클램프의 **얇은 블레이드 2매**에 오버레이 칼라로 덧대는
조개껍데기 셸. 63rd `:120`(비가역 개조 0) 준수 — 순정 조에 구멍을 뚫지 않고
C-채널 칼라로 감싸 조인다.

D446 준수: 보울을 각도 세그먼트로 분해해 **모든 조각이 convex**. 1-hull 금지.
같은 스크립트가 STL(프린트) + design.json(파라미터·유도량)을 함께 낸다.

사용:  python scoop_shell_design.py [출력디렉터리]

⚠️ MEASURE 표시 파라미터는 **실측 대기값**이다. 기본값은 사진 판독 기반 추정이며
   출력물을 프린트하기 전에 실측으로 대체할 것 (§검증 대기 참조).
"""
import sys, json, math
from pathlib import Path
import numpy as np
import trimesh

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "claudedocs/runtime_logs/scoop_shell_v0")

# ─────────────────────────────────────────────────────────────────────────
# 1. 파라미터
# ─────────────────────────────────────────────────────────────────────────
P = {
    # ── 그랩 개구 (D457 §12 역산, D458이 성립 확인) ──────────────────────
    "mouth_open_mm":      58.0,   # 립-투-립 완전개방. 입자비 하한 W>=10*d_max(=50) 통과
    "mouth_closed_mm":    46.0,   # 폐합 외폭. 58/46 = 1.26 = 실물 그랩 8모델(1.22~1.31) 일치
    "bowl_inner_r_mm":    21.0,   # 반원 단면 내반경
    "shell_width_mm":     40.0,   # 힌지축 방향 셸 너비 (link5 본체 35.5 + 편측 2.25 오버행)
    "wall_mm":             2.0,   # 셸 벽 두께
    "lip_land_mm":         0.8,   # 립 평탄 랜드. 나이프 에지 금지(칩핑) — 실물 그랩 커팅립 등가
    "side_overlap_mm":     1.2,   # 두 셸 측판 인터리브 겹침 (측면 누출·잼 동시 방지)
    "side_plate_thk_mm":   1.5,   # 측판 두께. shell_axis_convention 게이트가 Z 전폭 기대값에 쓴다

    # ── 오버레이 칼라 (순정 블레이드에 무개조 장착) ──────────────────────
    # ── 마운트 v1: 클램프 폐기, 기존 M2.5 구멍에 볼트 체결 ────────────────
    # v0 실패(2026-08-27 실물 시험): "들어가는데 헐겁고 미끄러짐".
    #   원인 ① blade_width 34.5mm를 통판으로 가정했으나 **두 조 모두 프레임 구조**다.
    #          URDF 정점밀도 스캔: 가동 조는 위/아래 레일만 재료, 가운데 개방.
    #          실제 물림면 = 높이 5.54mm 레일 하나 → 가정의 1/6 접촉.
    #   원인 ② 쿠폰에 볼트 구멍을 안 넣어 **순수 마찰만** 시험했다. 매끈한 1.5mm
    #          판재에 마찰만으로는 당연히 미끄러진다.
    # → v1은 마찰에 의존하지 않는다. **기존 구멍에 볼트 관통** = 형상 고정.
    #   구멍을 새로 뚫지 않으므로 63rd `:120` 비가역 개조 0도 유지된다.
    "blade_thk_mm":        1.5,   # ✅ 실물 자 측정으로 CAD값 확인 (URDF↔실물 일치)
    # 레일 실측 (가동 조, 칼라 구간 X38~52)
    "rail_bottom_z":     [-38.62, -33.08],   # 높이 5.54, 두께 1.51
    "rail_top_z":        [-7.88, -4.79],     # 높이 3.10, 두께 1.50
    # 기존 M2.5 구멍 (메쉬 래스터 검출, 등가지름)
    "moving_holes_xz":   [[50.0, -6.6], [50.2, -27.2], [50.0, -31.6]],
    "moving_hole_d":     [2.56, 3.75, 2.68],
    "fixed_holes_zy":    [[102.9, -13.2], [102.9, 12.0], [116.1, -0.6]],
    "fixed_hole_d":      [2.71, 2.71, 2.68],
    "bolt_span_mm":       25.0,   # 두 조 모두 약 25mm 간격 쌍을 갖는다
    # M2.5 관통 구멍. 2026-08-27 실물: **2.8mm는 볼트가 안 들어갔다.**
    # FDM은 구멍이 0.1~0.3mm 작게 출력되어 2.8 설계 → 실효 2.5~2.6 → M2.5(외경 2.5)와 간섭.
    # ISO 273 M2.5 = close 2.9 / normal 3.4. 관통이라 커도 무방하고, 3.4면 FDM 수축을 흡수하며
    # 스팬 오차(가동 25.0 vs 고정 25.2 + 래스터 검출 오차)까지 함께 흡수한다.
    "bolt_clear_d_mm":     3.4,
    "plate_thk_mm":        3.0,   # 백킹 플레이트 두께
    "plate_len_mm":       14.0,   # X방향. 가동 조 전폭 구간 14.6mm 이내

    # ── 마운트 각 (D430: top-down 완전 수직 불가, 6~24° 기울임 필수) ────
    "wedge_deg":          15.0,   # ⚠️MEASURE r=260mm에서 실제 도달 최대 수직도로 확정할 것

    # ── 재료 ────────────────────────────────────────────────────────────
    "density_g_cm3":       1.27,  # PETG. PA-CF면 1.15로 교체
    "seg_n":              20,     # 보울 각도 분해 수 (convex 조각 수).
                                  # 10은 반원이 각져 펠릿이 모서리에 끼고 유동이 실물과 달라진다.
                                  # 20 = 9도/세그먼트. D446 분해 convex를 지키면서 매끄럽게 하는 대가로
                                  # 셸당 조각 수가 21->31로 는다.

    # ── 검증 기준 (D457 §12 / D458 §7) ──────────────────────────────────
    "tool_mass_target_g": 35.0,
    "tool_mass_max_g":    50.0,
    "fill_factor":         0.70,  # 실물 그랩 0.6~0.9
    "bulk_density_g_cm3":  0.55,  # 플라스틱 펠릿 ⚠️MEASURE (250ml 계량컵 칭량)
    "pellet_d_max_mm":     5.0,
    "jaw_max_open_mm":    80.0,   # D458 실측

    # ── 조 운동학 (셸 배치의 근거. 2026-08-27 URDF 조인트 변환으로 확정) ──
    # 한쪽 조만 회전하므로 립-투-립 = 2*R*sin(theta/2). 개구 58mm를 theta_max에서
    # 얻으려면 립이 조인트 축에서 반경 R에 있어야 한다. 이 값을 쓰지 않으면 셸이
    # 조인트와 무관한 자리에 놓여 두 셸이 겹친다 (v1 회전 뷰에서 실제 발생).
    "jaw_max_angle_deg":  89.0,   # D458 실측 도달각 (state 88.77)
    # 닫힘 시 두 판재 사이 실제 빈 간격. link5 좌표계 변환 실측:
    #   고정 판 X -11.54~-10.03 / 가동 판 X -5.98~-4.47  ->  안쪽 면 간격 4.05mm
    # ⚠️MEASURE **이 값은 아직 CAD값이다** (URDF 조인트 변환에서 나온 값이지 실물 자로 잰 값이 아니다).
    #   D461 §4가 🔴로 남긴 유일한 미해소 항목. 다른 MEASURE 항목과 성격이 다르다:
    #   wedge_deg·bulk_density는 틀려도 **성능 추정치**만 흔들리지만, 이 값은 립 위치를 직접
    #   결정하므로 틀리면 **두 립이 안 만나거나 서로 민다** = 툴이 닫히지 않는다.
    #   그래서 아래 `closed_gap_measured_on_hardware` 게이트로 실측 여부 자체를 막아 세운다.
    "closed_plate_gap_mm": 4.05,
    # 위 값의 출처. 실측하면 "MEASURED_YYYY-MM-DD" 로 교체하고 값도 실측치로 바꿀 것.
    # 이 문자열이 CAD_ 로 시작하는 한 게이트는 FAIL을 유지한다.
    "closed_plate_gap_src": "CAD_URDF",
}

# ─────────────────────────────────────────────────────────────────────────
# 2. convex 조각 생성기
# ─────────────────────────────────────────────────────────────────────────
def box(lx, ly, lz, center=(0, 0, 0)):
    """축정렬 직육면체 (convex)."""
    m = trimesh.creation.box(extents=(lx, ly, lz))
    m.apply_translation(center)
    return m

def arc_segment(r_in, r_out, a0, a1, width):
    """반원 보울의 각도 세그먼트 하나. 8정점 육면체 → convex.

    ⚠️ 축 규약은 URDF 조 프레임을 따른다 (2026-08-27 정정):
        Z = 조인트 축 = 셸 너비 방향  (joint axis xyz="0 0 1")
        X = 블레이드가 뻗는 반경 방향
        Y = 두 판재가 갈라지는 방향 (닫힘 간격 4.05mm)
    따라서 보울 반원은 **회전 평면인 X-Y에** 그리고 **Z로 밀어낸다**.
    초판은 X-Z에 호를 그리고 Y로 밀어내어 두 셸이 Y로 40mm 겹쳤다 (회전 뷰에서 발견).
    """
    pts2 = []
    for a in (a0, a1):
        c, s = math.cos(a), math.sin(a)
        pts2 += [(r_in * c, r_in * s), (r_out * c, r_out * s)]
    v = []
    for z in (-width / 2, width / 2):
        for (x, y) in pts2:
            v.append((x, y, z))
    v = np.array(v, dtype=float)
    return trimesh.convex.convex_hull(trimesh.Trimesh(vertices=v))

def plate_with_holes(lx, ly, lz, hole_ys, hd, cz):
    """볼트 구멍이 있는 평판을 **볼록 박스들로 분해**해 반환. (D446: 분해 convex, 1-hull 금지)

    불리언으로 원형 구멍을 뚫으면 조각이 비볼록이 되어 게이트에 걸린다. 구멍을 **사각**으로
    두고 그 주변을 띠로 자르면 전 조각이 박스 = 볼록이며, 프린트 파일과 충돌 기하가 동일해진다.
    M2.5 관통에는 사각 구멍이 기능상 동일하고, 출력 시 원형보다 치수가 정확하다.
    """
    out, edges = [], []
    for hy in sorted(hole_ys):
        edges += [hy - hd / 2, hy + hd / 2]
    bounds = [-ly / 2] + edges + [ly / 2]
    for i in range(len(bounds) - 1):
        y0, y1 = bounds[i], bounds[i + 1]
        if y1 - y0 < 1e-6:
            continue
        is_hole_band = any(abs((y0 + y1) / 2 - hy) < hd / 2 for hy in hole_ys)
        if is_hole_band:                      # 구멍 띠: 구멍 좌우만 남긴다
            side = (lx - hd) / 2
            for sgn in (+1, -1):
                out.append(box(side, y1 - y0, lz,
                               center=(sgn * (hd + side) / 2, (y0 + y1) / 2, cz + lz / 2)))
        else:
            out.append(box(lx, y1 - y0, lz, center=(0, (y0 + y1) / 2, cz + lz / 2)))
    return out

def lip_radius(P):
    """개구 요구치를 만족하는 립의 조인트축 반경. lip_gap = 2*R*sin(theta/2)."""
    return P["mouth_open_mm"] / (2.0 * math.sin(math.radians(P["jaw_max_angle_deg"]) / 2.0))

def build_shell(P, moving: bool):
    """셸 1매 = 보울 세그먼트 N개 + 측판 2매 + 마운트 플레이트. 전부 convex.

    좌표계 = **해당 조의 로컬 프레임**. 원점 = 조인트 축, +X = 블레이드가 뻗는 방향.
    립을 X = lip_radius(P) 에 놓고, 보울은 거기서 조 판면 바깥(-Z)으로 자란다.
    두 셸은 닫힘 시 판재 간격(closed_plate_gap_mm)의 중앙에서 립이 만난다.
    """
    r_in, w = P["bowl_inner_r_mm"], P["wall_mm"]
    r_out = r_in + w
    width = P["shell_width_mm"]
    R_lip = lip_radius(P)
    # 셸 두께 중심을 판재 안쪽 면에서 간격 절반만큼 안으로 (닫히면 두 립이 맞닿음)
    y_off = (P["closed_plate_gap_mm"] / 2.0 - w / 2.0) * (1 if moving else -1)
    pieces = []

    # 보울: 0~180°를 seg_n 등분 (반원 단면)
    # 보울은 반원(180도)이며, 두 셸이 서로 마주 보도록 **고정 셸만 반대편 반원**을 쓴다.
    # (0~pi 는 +Y, pi~2pi 는 -Y. 같은 범위를 쓰면 두 보울이 같은 쪽으로 자라 겹친다.)
    n = P["seg_n"]
    base = 0.0 if moving else math.pi
    for i in range(n):
        a0 = base + math.pi * i / n
        a1 = base + math.pi * (i + 1) / n
        seg = arc_segment(r_in, r_out, a0, a1, width)
        seg.apply_translation((R_lip, y_off, 0.0))
        pieces.append(("bowl_%02d" % i, seg))

    # 립 랜드: 양 끝단에 평탄면 부여 (칩핑 방지 + 펠릿 쐐기물림 방지 챔퍼 대용)
    land = P["lip_land_mm"]
    for sgn, name in ((+1, "lip_a"), (-1, "lip_b")):
        pieces.append((name, box(land, w, width,
                                 center=(R_lip + sgn * (r_in + w / 2), y_off, 0.0))))

    # 측판 2매: 인터리브 겹침만큼 어긋나게 (실물 클램셸 방식)
    sp_t = P["side_plate_thk_mm"]
    off = P["side_overlap_mm"] / 2 * (1 if moving else -1)
    for sgn, name in ((+1, "side_a"), (-1, "side_b")):
        pieces.append((name, box(2 * r_out, r_out, sp_t,
                                 center=(R_lip, y_off + r_out / 2 * (1 if moving else -1),
                                         sgn * (width / 2 + sp_t / 2) + off))))

    # 마운트 v1: 백킹 플레이트 + M2.5 관통 구멍 2개 (기존 조 구멍에 체결).
    # 클램프 슬롯 없음 — 마찰이 아니라 볼트가 하중을 받는다.
    pt, pl, span, bd = (P["plate_thk_mm"], P["plate_len_mm"],
                        P["bolt_span_mm"], P["bolt_clear_d_mm"])
    # 마운트 플레이트는 조 판재 **바깥면**에 앉는다 (안쪽은 보울 공간이라 점유 금지).
    y_plate = (P["closed_plate_gap_mm"] / 2.0 + P["blade_thk_mm"] + pt / 2.0) * (1 if moving else -1)
    for i, m in enumerate(plate_with_holes(pl, span + 10.0, pt,
                                           [span / 2, -span / 2], bd, -pt / 2)):
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0)))
        m.apply_translation((R_lip, y_plate, 0.0))
        pieces.append((f"mount_{i:02d}", m))
    return pieces

# ─────────────────────────────────────────────────────────────────────────
# 3. 생성 · 게이트 · 기록
# ─────────────────────────────────────────────────────────────────────────
OUT.mkdir(parents=True, exist_ok=True)
report = {"params": P, "shells": {}, "gates": {}}
total_vol_mm3 = 0.0
nonconvex = []
shell_meshes = {}                      # 형상 게이트·회전 뷰가 실제 생성 기하를 재는 대상

for moving in (False, True):
    tag = "moving" if moving else "fixed"
    pieces = build_shell(P, moving)
    vol = 0.0
    for name, m in pieces:
        if not m.is_convex:
            nonconvex.append(f"{tag}/{name}")
        vol += m.volume
        m.export(OUT / f"shell_{tag}_{name}.stl")
    combined = trimesh.util.concatenate([m for _, m in pieces])
    combined.export(OUT / f"shell_{tag}_ALL.stl")
    shell_meshes[tag] = combined
    total_vol_mm3 += vol
    report["shells"][tag] = {
        "pieces": len(pieces),
        "volume_mm3": round(vol, 1),
        "mass_g": round(vol / 1000.0 * P["density_g_cm3"], 2),
        "bbox_mm": [round(x, 2) for x in combined.extents],
    }

# ── 유도량 ────────────────────────────────────────────────────────────────
r_in, width = P["bowl_inner_r_mm"], P["shell_width_mm"]
v_geo_cm3 = (math.pi * r_in ** 2 / 2) * width / 1000.0          # 폐합 시 반원 보울 용적
v_load_cm3 = v_geo_cm3 * P["fill_factor"]
m_load_g = v_load_cm3 * P["bulk_density_g_cm3"]
tool_g = total_vol_mm3 / 1000.0 * P["density_g_cm3"]

report["derived"] = {
    "bowl_geometric_cm3": round(v_geo_cm3, 2),
    "load_per_scoop_cm3": round(v_load_cm3, 2),
    "load_per_scoop_g":   round(m_load_g, 2),
    "tool_mass_g":        round(tool_g, 2),
    "open_close_ratio":   round(P["mouth_open_mm"] / P["mouth_closed_mm"], 3),
    "particle_ratio_W_over_d": round(P["mouth_open_mm"] / P["pellet_d_max_mm"], 2),
    "jaw_headroom_mm":    round(P["jaw_max_open_mm"] - P["mouth_open_mm"], 1),
}

# ── 게이트 ────────────────────────────────────────────────────────────────
g = report["gates"]
g["all_pieces_convex"]   = {"pass": bool(not nonconvex), "violations": nonconvex}
g["tool_mass_under_max"] = {"pass": bool(tool_g <= P["tool_mass_max_g"]),
                            "value_g": round(tool_g, 2), "limit_g": P["tool_mass_max_g"]}
g["open_close_ratio_in_real_band"] = {
    "pass": bool(1.22 <= report["derived"]["open_close_ratio"] <= 1.31),
    "value": report["derived"]["open_close_ratio"], "band": [1.22, 1.31]}
g["particle_ratio_min10"] = {
    "pass": bool(report["derived"]["particle_ratio_W_over_d"] >= 10.0),
    "value": report["derived"]["particle_ratio_W_over_d"]}
g["fits_in_measured_jaw_stroke"] = {
    "pass": bool(P["mouth_open_mm"] <= P["jaw_max_open_mm"]),
    "headroom_mm": report["derived"]["jaw_headroom_mm"]}

# ── 형상 게이트 (D461 §6) ─────────────────────────────────────────────────
# 위 5종은 **셸이 공간 어디에 놓이든 PASS한다** — 자중·개구비·입자비는 배치와 무관하다.
# 실제로 v1->v2에서 배치 오류 3건(볼트 구멍 과소 / 두 셸이 같은 로컬 원점 / 축 규약 반전)이
# 나는 내내 5종 전부 PASS였고, 발견은 사람이 3D 회전 뷰를 눈으로 보고서야 됐다.
# 아래 4종은 그 3건을 각각 FAIL로 잡도록 **생성된 메쉬를 직접 재서** 판정한다.
mv, fx = shell_meshes["moving"], shell_meshes["fixed"]

def _urdf_hinge_axis(path="local_assets/roarm_m3/urdf/roarm_m3.urdf",
                     joint="link5_to_gripper_link"):
    """조 힌지축을 URDF에서 직접 읽는다 (게이트 근거를 코드 주석이 아니라 원본에 둔다)."""
    import xml.etree.ElementTree as ET
    for j in ET.parse(path).getroot().iter("joint"):
        if j.get("name") == joint:
            return [float(v) for v in j.find("axis").get("xyz").split()]
    return None

# (A) 닫힘 상태에서 두 셸이 서로를 파고들지 않는가 — 오류 ②·③을 잡는 게이트.
#     Y = 두 조가 갈라지는 방향이므로, 닫힘 시 두 셸의 Y 구간은 맞닿기만 하고 겹치면 안 된다.
#     설계 의도값 = -(closed_plate_gap - 2*wall): 두 립 안쪽 면이 판재 간격 중앙에서 만난다.
#     허용 대역 [-pellet_d_max, +0.05] mm 근거:
#       상한 +0.05 = 양수는 곧 두 셸이 같은 공간을 점유(=닫히지 않음)라 0이 진짜 한계다.
#                    0.05는 mm 단위 float 왕복 여유일 뿐 설계 여유가 아니다.
#                    v1 축 반전 시 +40 mm, 두 보울을 같은 반원 범위로 두면 +21 mm -> 즉시 FAIL.
#       하한 -5.0 = 립 사이가 최대 펠릿 지름(pellet_d_max)보다 벌어지면 폐합해도 입자가 샌다.
#                   셸이 조인트와 무관한 자리로 밀려나는 반대 방향 오류도 여기서 걸린다.
y_overlap = float(min(mv.bounds[1][1], fx.bounds[1][1]) - max(mv.bounds[0][1], fx.bounds[0][1]))
y_expect = -(P["closed_plate_gap_mm"] - 2 * P["wall_mm"])
y_band = [-P["pellet_d_max_mm"], 0.05]
g["shells_do_not_interpenetrate"] = {
    "pass": bool(y_band[0] <= y_overlap <= y_band[1]),
    "y_overlap_mm": round(y_overlap, 3), "design_intent_mm": round(y_expect, 3),
    "band_mm": y_band,
    "moving_y_mm": [round(mv.bounds[0][1], 2), round(mv.bounds[1][1], 2)],
    "fixed_y_mm":  [round(fx.bounds[0][1], 2), round(fx.bounds[1][1], 2)]}

# (B) 축 규약이 URDF와 같은가 — 오류 ③을 잡는 게이트.
#     힌지축 Z(link5_to_gripper_link axis xyz="0 0 1") -> 보울 호는 회전 평면 X-Y에,
#     셸 폭은 Z를 따라 있어야 한다. 초판은 호를 X-Z에 그리고 Y로 밀어내어
#     4.05 mm만 갈라져야 할 Y에 셸 폭 40 mm를 넣었다.
axis = _urdf_hinge_axis()
r_out_g = P["bowl_inner_r_mm"] + P["wall_mm"]
x_expect = 2 * r_out_g                                        # 보울 지름이 X에 놓인다
z_expect = P["shell_width_mm"] + 2 * P["side_plate_thk_mm"]   # 셸 폭 + 측판 2매가 Z에 놓인다
ext = [float(e) for e in mv.extents]
# 허용 0.5 mm: 이 셋은 파라미터로 결정되는 정확값이라 여유가 필요 없다.
# 0.5는 노즐 지름(0.4)보다 작아 프린트로 구별조차 안 되는 폭이며, 축이 뒤바뀌면 오차는 mm가 아니라
# 수십 mm 단위로 벌어지므로 이 정도 여유로도 반전은 확실히 걸린다.
ax_viol = []
if axis != [0.0, 0.0, 1.0]:
    ax_viol.append(f"URDF hinge axis {axis} != [0,0,1]")
if abs(ext[0] - x_expect) > 0.5:
    ax_viol.append(f"X extent {ext[0]:.2f} != bowl dia {x_expect:.2f} (보울 호가 X-Y 평면에 없다)")
if abs(ext[2] - z_expect) > 0.5:
    ax_viol.append(f"Z extent {ext[2]:.2f} != shell width {z_expect:.2f} (셸 폭이 힌지축 Z에 없다)")
if ext[1] >= P["shell_width_mm"]:
    ax_viol.append(f"Y extent {ext[1]:.2f} >= shell width {P['shell_width_mm']} (조 간격 방향에 셸 폭이 들어갔다)")
g["shell_axis_convention"] = {
    "pass": bool(not ax_viol), "violations": ax_viol,
    "urdf_hinge_axis": axis, "extents_xyz_mm": [round(e, 2) for e in ext],
    "expected_x_mm": round(x_expect, 2), "expected_z_mm": round(z_expect, 2)}

# (C) 립 반경이 조 운동학에서 나왔는가 — 오류 ②를 잡는 게이트.
#     한쪽 조만 회전하므로 립-투-립 = 2*R*sin(theta/2). 개구 58 mm를 theta_max=89°에서
#     얻으려면 R = 58/(2*sin(44.5°)) = 41.37 mm. 이 값을 안 쓰면 셸이 조인트와 무관한
#     자리에 놓여 두 셸이 겹친다(v1에서 실제 발생).
#     여기서는 공식을 다시 계산하는 데 그치지 않고 **메쉬에서 잰 보울 축 반경**과 대조한다
#     (X 범위의 중점 = 보울 축). 그래야 파라미터가 아니라 생성 결과가 판정된다.
R_kin = P["mouth_open_mm"] / (2.0 * math.sin(math.radians(P["jaw_max_angle_deg"]) / 2.0))
R_meas = {t: float((m.bounds[0][0] + m.bounds[1][0]) / 2) for t, m in shell_meshes.items()}
chord = {t: 2.0 * r * math.sin(math.radians(P["jaw_max_angle_deg"]) / 2.0) for t, r in R_meas.items()}
# 허용 0.1 mm — 노즐 0.4 mm의 1/4로, 프린트로는 구현조차 안 되는 정밀도다.
# 즉 "계산이 일치한다"만 통과시키고 임의 배치는 통과시키지 않는다.
lip_viol = [f"{t}: R={R_meas[t]:.3f} (kin {R_kin:.3f}), lip gap@{P['jaw_max_angle_deg']}deg={chord[t]:.3f} "
            f"(target {P['mouth_open_mm']})"
            for t in R_meas
            if abs(R_meas[t] - R_kin) > 0.1 or abs(chord[t] - P["mouth_open_mm"]) > 0.1]
g["lip_radius_matches_kinematics"] = {
    "pass": bool(not lip_viol), "violations": lip_viol,
    "R_kinematic_mm": round(R_kin, 3),
    "R_measured_mm": {t: round(v, 3) for t, v in R_meas.items()},
    "lip_gap_at_theta_max_mm": {t: round(v, 3) for t, v in chord.items()},
    "formula": "R = mouth_open / (2*sin(theta_max/2))"}

# (D) 볼트 관통 구멍이 FDM 수축을 흡수하는가 — 오류 ①을 잡는 게이트.
#     2026-08-27 실물: 2.8 mm 설계 -> M2.5 볼트가 안 들어갔다. FDM은 구멍이 0.1~0.3 mm
#     작게 출력되어 실효 2.5~2.6 mm가 되고 볼트 외경 2.5 mm와 간섭한다.
#     하한 3.4 = ISO 273 M2.5 normal이자 **2026-08-31 실물 쿠폰으로 통과 확인된 값**.
#                검증된 값 아래로는 회귀 금지 (2.8 -> 즉시 FAIL).
#     상한 4.0 = ISO 273 M2.5 coarse. 그 이상이면 M2.5 헤드(약 4.5 mm)의 지지 면적이 사라진다.
bolt_band = [3.4, 4.0]
g["bolt_hole_absorbs_fdm_shrink"] = {
    "pass": bool(bolt_band[0] <= P["bolt_clear_d_mm"] <= bolt_band[1]),
    "value_mm": P["bolt_clear_d_mm"], "band_mm": bolt_band,
    "basis": "ISO 273 M2.5 close 2.9 / normal 3.4 / coarse 4.0 + FDM 수축 0.1~0.3 mm"}

# (E) 닫힘 간격이 **실물 측정값**인가, 그리고 립이 만나는 범위 안인가 — D461 §4의 유일한 미해소 🔴.
#     이 게이트가 필요한 이유는 (A)와 다르다. (A)는 "지금 파라미터로 만든 셸이 서로 안 겹치나"를
#     보는데, closed_plate_gap_mm 에서 y_off 가 유도되므로 **CAD값이 틀려도 (A)는 항상 PASS한다**.
#     즉 (A)는 자기 자신과의 일관성만 볼 뿐 실물과의 일치는 못 본다. D461 §6이 말한
#     "계산이 성립하니까 통과한다"의 또 다른 얼굴이다. 그래서 값이 아니라 **출처**를 판정한다.
#
#     measure_pending 전체를 게이트로 묶지 않는 이유: wedge_deg 는 도달 자세, bulk_density 는
#     펠릿 조달에 걸려 있고 둘 다 틀려도 툴은 닫힌다. 닫힘 간격만이 **닫히느냐 마느냐**를 정한다.
#
#     허용 범위는 립 기하에서 그대로 유도된다. 립 안쪽 면은 중심선에서 gap/2 - wall 에 있고
#     Y 중첩 = -(gap - 2*wall) 이므로:
#       gap < 2*wall               -> 중첩 양수 = 두 립이 서로를 민다 (닫히지 않음)
#       gap > 2*wall + pellet_d_max -> 립 사이가 최대 펠릿보다 벌어짐 = 폐합해도 샌다
gap, wall = P["closed_plate_gap_mm"], P["wall_mm"]
gap_band = [2 * wall, 2 * wall + P["pellet_d_max_mm"]]
gap_measured = not P["closed_plate_gap_src"].startswith("CAD")
gap_viol = []
if not gap_measured:
    gap_viol.append(f"closed_plate_gap_mm={gap} 출처가 '{P['closed_plate_gap_src']}' = 실측이 아니다. "
                    "닫힘 상태에서 두 조 판재 안쪽 면 간격을 실측하고 "
                    "closed_plate_gap_src 를 'MEASURED_YYYY-MM-DD' 로 교체할 것")
if not (gap_band[0] <= gap <= gap_band[1]):
    gap_viol.append(f"gap {gap} 이 립 폐합 범위 {gap_band} 밖 "
                    f"({'두 립이 서로를 민다' if gap < gap_band[0] else '립 사이로 펠릿이 샌다'})")
g["closed_gap_measured_on_hardware"] = {
    "pass": bool(not gap_viol), "violations": gap_viol,
    "value_mm": gap, "source": P["closed_plate_gap_src"], "measured": bool(gap_measured),
    "lip_closure_band_mm": [round(b, 2) for b in gap_band],
    # 실측치가 이보다 조금만 작아도 두 립이 서로를 민다. FDM 치수 오차(±0.1~0.2mm)와 직접 비교할 것.
    "margin_to_collision_mm": round(gap - gap_band[0], 3),
}

report["measure_pending"] = ["closed_plate_gap_mm", "wedge_deg", "bulk_density_g_cm3"]
report["measure_confirmed"] = {
    "blade_thk_mm": "2026-08-27 실물 자 측정 = CAD 1.5 일치",
    "bolt_span_mm": "2026-08-31 v1 쿠폰 실물 대조 = 조의 기존 M2.5 구멍과 25 mm 스팬 일치 "
                    "(D460 §4 '구멍 좌표는 래스터 근사' 유보 해소)",
    "bolt_clear_d_mm": "2026-08-31 v1 쿠폰 실물 = M2.5 볼트가 3.4 mm 구멍 통과",
}

# ── 칼라 적합성 시험쿠폰 ───────────────────────────────────────────────────
# 셸 본체 없이 칼라만 뽑아 1.5mm 블레이드 물림을 먼저 검증한다.
# 설계 최대 미지수 = "8mm × 34.5mm TPU 심 마찰로 굴착 반력(gs2 실측 3.5~6.3N)을 견디나".
# 손잡이 탭을 붙여 손으로 밀어넣고 빼면서 물림·슬립을 확인.
# v1 쿠폰: 볼트 패턴 적합성만 시험한다. 평판 1장 + M2.5 관통 2개.
# 확인 항목 = (a) 25mm 스팬이 조의 기존 구멍과 맞나 (b) 볼트가 1.5mm 판재를
# 물었을 때 판이 휘지 않나 (c) 조인 뒤 회전 유격이 남나.
# 출력 방향: 평판이 베드에 평평 → 오버행 0, 서포트 불필요, 구멍은 수직 관통.
pt, pl, span, bd = (P["plate_thk_mm"], P["plate_len_mm"],
                    P["bolt_span_mm"], P["bolt_clear_d_mm"])
coupon = plate_with_holes(pl, span + 12.0, pt, [span / 2, -span / 2], bd, 0.0)
cp = trimesh.util.concatenate(coupon)
cp.export(OUT / "collar_test_coupon.stl")
report["collar_coupon"] = {
    "rev": "v1 (볼트 체결)",
    "bolt_span_mm": span, "bolt_clear_d_mm": bd, "plate_thk_mm": pt, "plate_len_mm": pl,
    "mass_g": round(sum(m.volume for m in coupon) / 1000.0 * P["density_g_cm3"], 2),
    "bbox_mm": [round(x, 2) for x in cp.extents],
    "purpose": "기존 M2.5 구멍 25mm 스팬 적합성 + 1.5mm 판재 볼트 체결 시 변형·유격 검증",
    "v0_result": "들어가나 헐겁고 미끄러짐 — 원인: 조가 통판이 아니라 프레임(실물 물림면 "
                 "높이 5.54mm 레일 1개 = 가정의 1/6), 그리고 쿠폰에 볼트 구멍 부재로 마찰만 시험됨",
    # 2026-08-31 사용자 실물 시험 보고. 시험 항목 (a)(b)(c) 전부 PASS.
    "v1_result": {
        "date": "2026-08-31",
        "source": "사용자 실물 시험 보고",
        "a_bolt_through_hole": "PASS — M2.5 볼트가 3.4mm 구멍을 통과 (2.8mm는 안 들어갔었다)",
        "b_span_matches_jaw_holes": "PASS — 스팬 25mm가 조의 기존 M2.5 구멍과 일치",
        "c_plate_no_deflection": "PASS — 조여도 1.5mm 판재가 휘지 않음",
        "verdict": "3/3 PASS",
        "resolves": "D460 §4 유보 해소 — 구멍 좌표는 메쉬 래스터(0.35mm 격자) 근사였고 "
                    "실물 대조가 필요했다. (b)가 그 대조이며 일치했다.",
    },
}

(OUT / "design.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

# ── 콘솔 요약 ─────────────────────────────────────────────────────────────
print(f"출력: {OUT}")
for tag, s in report["shells"].items():
    print(f"  {tag:7s} pieces={s['pieces']:3d}  {s['mass_g']:6.2f} g  bbox {s['bbox_mm']}")
print(f"\n  툴 자중 합계   {tool_g:6.2f} g  (목표 {P['tool_mass_target_g']}, 상한 {P['tool_mass_max_g']})")
print(f"  1회 실적재     {v_load_cm3:6.2f} cm3 = {m_load_g:5.2f} g")
print("\n게이트:")
for k, v in g.items():
    print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}")
    if not v["pass"]:                       # FAIL은 이유가 보여야 게이트다 (D461 §6)
        for w in v.get("violations", []):
            print(f"         ! {w}")
print(f"  형상: Y 중첩 {y_overlap:+.3f} mm (의도 {y_expect:+.3f}, 대역 {y_band}) · "
      f"립 반경 {R_meas['moving']:.2f} mm (운동학 {R_kin:.2f}) · "
      f"bbox XYZ {[round(e, 2) for e in ext]}")
print(f"  닫힘 간격: {gap} mm [{P['closed_plate_gap_src']}] · 립 폐합 범위 {gap_band} · "
      f"충돌까지 여유 {gap - gap_band[0]:+.2f} mm")
if gap - gap_band[0] < 0.2:   # FDM 치수 오차 ±0.1~0.2mm. 여유가 그보다 작으면 출력물은 실제로 닫히지 않는다.
    print(f"  ⚠️ 충돌 여유 {gap - gap_band[0]:.2f} mm < FDM 오차 0.2 mm — 실측이 조금만 작아도 두 립이 서로 민다. "
          f"벽 두께({wall})나 립 위치 조정은 설계 결정 사항.")
print("\n실측 대기:", ", ".join(report["measure_pending"]))

# ─────────────────────────────────────────────────────────────────────────
# 4. 조립도 — 부품만 주고 어디 붙는지 안 알려주면 쓸 수 없다 (v0 교훈)
# ─────────────────────────────────────────────────────────────────────────
def render_assembly(P, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.collections as mc
    from matplotlib.patches import Rectangle

    jaw = trimesh.load("local_assets/roarm_m3/urdf/meshes/gripper_link.stl")
    sc = 1000.0 if jaw.extents.max() < 1 else 1.0
    V, F = jaw.vertices * sc, jaw.faces

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.add_collection(mc.PolyCollection(V[F][:, :, [0, 2]], facecolors="0.32", edgecolors="none"))

    rb, rt = P["rail_bottom_z"], P["rail_top_z"]
    ax.add_patch(Rectangle((38, rb[0]), 14, rb[1] - rb[0], fc="red", alpha=.28, ec="red", lw=2, zorder=6))
    ax.add_patch(Rectangle((38, rt[0]), 14, rt[1] - rt[0], fc="dodgerblue", alpha=.28, ec="dodgerblue", lw=2, zorder=6))
    ax.text(53, (rb[0]+rb[1])/2, f"BOTTOM RAIL  h={rb[1]-rb[0]:.2f}  t={P['blade_thk_mm']}mm",
            color="red", fontsize=9.5, weight="bold", va="center")
    ax.text(53, (rt[0]+rt[1])/2, f"TOP RAIL  h={rt[1]-rt[0]:.2f}  t={P['blade_thk_mm']}mm",
            color="dodgerblue", fontsize=9.5, weight="bold", va="center")

    # 기존 M2.5 구멍 = 체결점
    for (hx, hz), hd in zip(P["moving_holes_xz"], P["moving_hole_d"]):
        ax.add_patch(plt.Circle((hx, hz), hd/2, fc="lime", ec="darkgreen", lw=2, zorder=8))
        ax.text(hx+2.5, hz, f"O{hd:.2f}", color="darkgreen", fontsize=8.5, weight="bold", va="center")

    # 체결에 쓰는 두 구멍 (스팬)
    span = P["bolt_span_mm"]
    ax.annotate("", xy=(50, -6.6), xytext=(50, -31.6),
                arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=2.2))
    ax.text(44.5, -19, f"BOLT SPAN\n{span:.1f} mm", color="darkgreen", fontsize=10,
            weight="bold", ha="right", va="center")

    # 마운트 플레이트가 앉는 자리
    ax.add_patch(Rectangle((38, -34), P["plate_len_mm"], 30, fill=False,
                           ec="purple", ls="--", lw=2.2, zorder=7))
    ax.text(45, -0.5, "MOUNT PLATE sits here\n(bolts through existing holes - no drilling)",
            color="purple", fontsize=10, weight="bold", ha="center", va="bottom")

    ax.add_patch(Rectangle((52.5, -42), 15, 45, fill=False, ec="gray", ls=":", lw=1.5))
    ax.text(60, -44, "tapered tip\nDO NOT mount", ha="center", va="top", color="gray", fontsize=9)
    ax.text(10, -20, "HUB\n(servo side)", ha="center", color="0.15", fontsize=10)

    ax.set_xlim(-14, 92); ax.set_ylim(-49, 8); ax.set_aspect("equal"); ax.grid(alpha=.25)
    ax.set_xlabel("X (mm) - from joint origin"); ax.set_ylabel("Z (mm) - blade width direction")
    ax.set_title("ASSEMBLY - gripper_link (moving jaw)\nmount plate bolts through the jaw's EXISTING M2.5 holes",
                 fontsize=13, weight="bold")
    plt.tight_layout(); plt.savefig(out_png, dpi=95); plt.close()
    return out_png

_asm = render_assembly(P, OUT / "ASSEMBLY.png")
print(f"조립도: {_asm}")

# ─────────────────────────────────────────────────────────────────────────
# 5. 회전 뷰 — 위 형상 게이트가 놓치는 것을 사람 눈이 잡는 마지막 경로 (D461 §6·§7)
#    게이트는 "알고 있는 오류 3종"만 잡는다. v1의 겹침·축 반전을 실제로 드러낸 것은
#    사용자의 "3D로 돌려볼 수 없냐"였다. 그 뷰를 수동 렌더가 아니라 생성기에 내장한다.
# ─────────────────────────────────────────────────────────────────────────
def render_turntable(meshes, out_png, y_overlap_mm, elev_deg=22.0,
                     azims=(0, 60, 120, 180, 240, 300)):
    """닫힘 상태 두 셸을 공통 원점 기준으로 6각 회전 뷰. 파랑=가동, 빨강=고정.

    두 셸이 포개져 보이면 배치 오류, 한쪽으로만 자라 보이면 반원 범위 오류다.
    시점은 -Y에서 +Y를 본다(깊이=Y). 화면 좌표 = (X, Z).
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.collections as mc

    COL = {"moving": (0.20, 0.42, 0.88), "fixed": (0.88, 0.26, 0.20)}
    allv = np.vstack([m.vertices for m in meshes.values()])
    ctr = (allv.min(0) + allv.max(0)) / 2

    def view_mat(az):
        a, e = math.radians(az), math.radians(elev_deg)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
        return Rx @ Rz

    # 전 패널 공통 축척 = 어느 각도에서도 안 잘리는 최소 반폭 (bbox 8정점만 투영해 구한다).
    # 패널마다 축척이 다르면 두 셸의 상대 크기·간격을 눈으로 비교할 수 없다.
    lo, hi = allv.min(0) - ctr, allv.max(0) - ctr
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    span = max(float(np.abs((corners @ view_mat(az).T)[:, [0, 2]]).max()) for az in azims) * 1.06

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, az in zip(axes.ravel(), azims):
        M = view_mat(az)
        tris, depth, cols = [], [], []
        for tag, m in meshes.items():
            T = ((m.vertices - ctr) @ M.T)[m.faces]                     # (F,3,3)
            n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            ln = np.linalg.norm(n, axis=1); ln[ln == 0] = 1.0
            shade = 0.38 + 0.62 * np.abs(n[:, 1] / ln)                  # 광원 = 시선 방향
            tris.append(T[:, :, [0, 2]])
            depth.append(T[:, :, 1].mean(1))
            cols.append(np.clip(np.array(COL[tag])[None, :] * shade[:, None], 0, 1))
        tris, depth, cols = np.vstack(tris), np.concatenate(depth), np.vstack(cols)
        o = np.argsort(depth)[::-1]                                     # 먼 것(=Y 큰 것)부터
        ax.add_collection(mc.PolyCollection(tris[o], facecolors=cols[o],
                                            edgecolors=(0, 0, 0, 0.20), linewidths=0.2))
        ax.set_xlim(-span, span); ax.set_ylim(-span, span)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"azim {az}deg", fontsize=10)

    fig.suptitle("TURNTABLE - closed state, both shells in the shared jaw frame\n"
                 f"blue = moving jaw   red = fixed jaw   |   Y overlap {y_overlap_mm:+.3f} mm "
                 "(0 = lips just touch; positive = shells collide)",
                 fontsize=12.5, weight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig(out_png, dpi=95); plt.close()
    return out_png

_tt = render_turntable(shell_meshes, OUT / "TURNTABLE.png", y_overlap)
print(f"회전 뷰: {_tt}")
