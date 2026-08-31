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
    "closed_plate_gap_mm": 4.05,
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
    sp_t = 1.5
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

report["measure_pending"] = ["wedge_deg", "bulk_density_g_cm3"]
report["measure_confirmed"] = {"blade_thk_mm": "2026-08-27 실물 자 측정 = CAD 1.5 일치"}

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
