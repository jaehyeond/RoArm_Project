#!/usr/bin/env python3
"""S1 그랩 생성기 — 순정 가동 조 자리에 "포크 + 암 + 반쪽 보울"(가동부), 순정 고정 조 바깥면에 "판 + 스파인 + 반쪽 보울"(고정부).
D480 (2026-09-03, 사용자 승인): 양쪽 가동(g18, D462~D479 기구) → 한쪽 가동, 서보축 직결. 링크·기어·요크·크랭크 전부 삭제.

좌표 = link5 프레임(mm). 힌지축 = link5 Y, 통과점 (X 0, Z 52.035). 팁 방향 +Z. 고정 조 = −X 쪽, 가동부는 +X 로 열림.
형상 근거 = 벤더 STEP (g19_servo_direct/vendor_step_parts/*, stock_jaw_interface.json) + 실물 대조(real_arm_confirmation_20260903.json).
D446 준수: collision 용 조각은 전부 볼록. 시각 메시는 정확한 윤곽(창·구멍 포함) 압출.

사용: python scoop_grab_s1_design.py [출력 디렉터리]
"""
import os, sys, json, math, hashlib
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point, box as sbox
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parent
VS = REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/vendor_step_parts"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1")   # forward-only: v0 는 09-04 출력본, 덮어쓰지 않음

P = {
    # ── 힌지·순정 인터페이스 (STEP) ──
    "hinge_xz": (0.0, 52.035), "hinge_y": 18.821,
    "cheek_inner_y": (-18.30, 18.07),          # 순정 뺨 안쪽면 (STEP). 판은 여기서 바깥으로 자란다
    "plate_t": 2.0,                              # PLA 판 두께. v0 3.0 은 순정 M3×4 가 안 물림(09-04 조립 실측) → 2.0: 순정 M3×4 로 디스크 물림 2.0(순정 1.5 뺨은 2.5)
    "hub_pcd": 14.0, "hub_hole_d": 3.4, "hub_window_d": 11.0, "pin_hole_d": 1.2,
    "pin_r": 6.0, "pin_ang_deg": -28.6,          # jaw 프레임(x=Z, y=X) 각도
    "cheek_cut_z": 100.0,                        # 순정 뺨 윤곽을 여기까지 복사 (STEP 블레이드 뿌리 이전)
    # ── 가동부 암·다리 ──
    "arm_x": (0.0, 22.0), "arm_z0": 66.0, "arm_step_z": 118.0, "arm_window": (4.0, 82.0, 18.0, 110.0),   # 암 = L자: X0~22 (Z66~120) + X8.1~22 (Z118~zc). 고정 캡(X≤8.1, Z≥123.4) 회피
    "bridge_x": (8.0, 18.0), "bridge_z": (68.0, 76.0), "tie_bolt_xz": (13.0, 72.0), "tie_bolt_sq": 3.5,   # 다리 관통 타이볼트 M3×45 (출력 분할 접합, 09-04)
    # ── 보울 (원통, 축 ∥ Y) ──
    "bowl_xc": 8.1, "bowl_zc": 145.0, "bowl_r_in": 20.0, "wall": 1.6, "seg_n": 12,
    # ── 고정부 ──
    "fixed_plate_x": (-15.54, -11.54),           # 고정 조 블레이드 바깥면 (blade X −11.54)
    "fixed_plate_y": (-15.95, 14.92),            # 플랜지 안쪽 −0.3 (D476)
    "fixed_plate_z": (95.0, 121.0),
    "fixed_holes_yz": [(-13.08, 103.11), (12.04, 103.11), (-0.51, 116.20)],   # STEP 관통 ⌀3.2 (M3 확인, 09-03)
    "fixed_hole_d": 3.4, "spine_y": (-15.0, 15.0), "spine_z1": 150.0, "spine_flare_z": (107.5, 112.5),   # 플랜지 끝 106.4 + 1.1 여유에서 45° 로 캡 폭까지
    # ── 물성·예산 ──
    "density_g_cm3": 1.24, "E_pla_MPa": 3000.0, "tool_mass_max_g": 65.0, "stock_jaw_removed_g": 9.49,
    "screws": [["M3x4_disc_stock", 8, 0.40], ["M3x8_fixed_plate", 3, 0.60], ["M3_nut", 3, 0.40]],   # 디스크 나사 = 순정 M3×4 재사용(+중앙 M3 유지)
    "open_deg_max": 30.0, "mouth_target_mm": 58.0, "B1_required_cm3": 6.41,
    "bulk_density_g_cm3": 0.55, "fill_factor": 0.70, "lip_force_test_N": 20.0,
    "servo_torque_Nm": 1.96,                     # ST3215-HS 20 kg·cm @12 V (실물 라벨, 09-03)
}
H = np.array(P["hinge_xz"])

# ────────────────────────────── 기하 도구 ──────────────────────────────
def extrude_xz(poly, y0, y1):
    """XZ 평면 shapely 다각형(구멍 포함)을 Y 방향 [y0,y1] 로 압출 → link5 좌표 trimesh."""
    poly = poly.simplify(0.02, preserve_topology=True)     # 공선 점·부스러기 제거 (퇴화 삼각형 → NaN 법선 → RTX 검정 렌더 방지)
    m = trimesh.creation.extrude_polygon(poly, y1 - y0)   # 다각형 (u,v) → 정점 (u,v,w), w∈[0,h]
    v = m.vertices.copy()
    # (u,v,w) = (X, Z, Y-y0)  →  link5 (X, Y, Z)
    m.vertices = np.stack([v[:, 0], v[:, 2] + y0, v[:, 1]], 1)   # 축 교환 = 반사 → 면 방향 뒤집힘
    m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces[:, ::-1] if m.volume < 0 else m.faces, process=True)
    m.update_faces(m.nondegenerate_faces(height=1e-4)); m.remove_unreferenced_vertices(); m.merge_vertices()
    m.fix_normals()
    assert m.volume > 0, "압출 판 부피 음수 — 면 방향 확인"
    assert np.isfinite(m.face_normals).all() and m.area_faces.min() > 1e-4, "퇴화 삼각형 잔류"
    return m

def extrude_xz_prisms(poly, y0, y1):
    """XZ 다각형(구멍 포함)을 삼각형 프리즘(볼록 조각) 묶음으로 압출 → 한 메시로 연결. RTX 가 압출 판의 큰 평면을 검게 그리던 문제 우회(09-03)."""
    poly = poly.simplify(0.02, preserve_topology=True)
    v2, tri = trimesh.creation.triangulate_polygon(poly)
    pieces = []
    for t in tri:
        p = v2[t]                                             # (3,2) = (X, Z)
        a = np.abs((p[1,0]-p[0,0])*(p[2,1]-p[0,1]) - (p[2,0]-p[0,0])*(p[1,1]-p[0,1]))/2
        if a < 1e-3: continue
        pts = [(x, y, z) for y in (y0, y1) for (x, z) in p]
        pieces.append(trimesh.convex.convex_hull(trimesh.Trimesh(vertices=np.array(pts, float))))
    m = trimesh.util.concatenate(pieces); m.fix_normals(); return m

def box_l5(x, y, z):
    m = trimesh.creation.box(extents=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
    m.apply_translation(((x[0]+x[1])/2, (y[0]+y[1])/2, (z[0]+z[1])/2)); return m

def wedge_xz(cx, cz, r_in, r_out, a0, a1, y0, y1):
    """XZ 평면 호 쐐기(볼록). 각도 = +Z(바닥, 팁 방향)에서 +X 로. sweep < 90°."""
    assert abs(a1 - a0) < math.pi/2
    pts = []
    for a in (a0, a1):
        s, c = math.sin(a), math.cos(a)
        pts += [(cx + r_in*s, cz + r_in*c), (cx + r_out*s, cz + r_out*c)]
    v = [(x, y, z) for y in (y0, y1) for (x, z) in pts]
    return trimesh.convex.convex_hull(trimesh.Trimesh(vertices=np.array(v, float)))

def half_cyl(cx, cz, r_in, r_out, side, y0, y1, n):
    """side +1: +X 반원(0..180°), −1: −X 반원(180..360°). 쐐기 n개."""
    out = []; a_start = 0.0 if side > 0 else math.pi
    for i in range(n):
        a0 = a_start + math.pi*i/n; a1 = a_start + math.pi*(i+1)/n
        out.append(wedge_xz(cx, cz, r_in, r_out, a0, a1, y0, y1))
    return out

def half_disc_poly(cx, cz, r, side, n=48):
    a = np.linspace(0, math.pi, n+1) if side > 0 else np.linspace(math.pi, 2*math.pi, n+1)
    pts = [(cx + r*math.sin(t), cz + r*math.cos(t)) for t in a]
    return Polygon(pts)

def circle(cx, cz, d, n=32): return Point(cx, cz).buffer(d/2, resolution=n)

def rot_about_hinge(m, deg):
    """가동부를 힌지(Y축, 통과점 H) 둘레로 회전. +deg = 열림(+X 로)."""
    T = trimesh.transformations.rotation_matrix(math.radians(deg), [0, -1, 0], [H[0], 0.0, H[1]])
    m2 = m.copy(); m2.apply_transform(T); return m2

def rot_pts(pts, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    p = np.asarray(pts, float) - np.array([H[0], 0, H[1]])
    x = p[:, 0]*c + p[:, 2]*s; z = -p[:, 0]*s + p[:, 2]*c
    return np.stack([x + H[0], p[:, 1], z + H[1]], 1)

# ────────────────────────────── 순정 뺨 윤곽 (STEP) ──────────────────────────────
def stock_cheek_outline():
    """STEP movable_jaw.stl 구동측 뺨 단면 → link5 XZ 다각형 (외곽만)."""
    mj = trimesh.load(VS / "movable_jaw.stl", force="mesh")
    sec = mj.section(plane_origin=[0, 0, 327.25], plane_normal=[0, 0, 1])
    rings = []
    for d in sec.discrete:                                          # 닫힌 경로 여러 개(외곽 + 창들)
        if len(d) < 4: continue
        X = -(d[:, 1] + 0.88); Z = d[:, 0] - 236.967                # STEP (x,y) → link5 (X,Z)
        pg = Polygon(np.stack([X, Z], 1)).buffer(0)
        if not pg.is_empty: rings.append(pg)
    ext = max(rings, key=lambda g: g.area)
    holes = [r for r in rings if r is not ext and ext.contains(r.representative_point())]
    out = ext
    for h in holes: out = out.difference(h)                         # 순정 창 그대로(경량, 순정 부피 안)
    return out

# ────────────────────────────── 가동부 ──────────────────────────────
def build_door(P):
    """반환: pieces(볼록, collision), names, visual_meshes(정확 윤곽), poly_side"""
    hx, hz = P["hinge_xz"]; t = P["plate_t"]; xc, zc = P["bowl_xc"], P["bowl_zc"]
    r_in = P["bowl_r_in"]; r_out = r_in + P["wall"]
    stock = stock_cheek_outline()
    cheek = stock.intersection(sbox(-50, -50, 50, P["cheek_cut_z"]))
    arm = unary_union([sbox(P["arm_x"][0], P["arm_z0"], P["arm_x"][1], P["arm_step_z"] + 2.0),
                       sbox(xc, P["arm_step_z"], P["arm_x"][1], zc)])          # L자 암: 고정 캡 영역(X≤xc, Z≥123.4) 회피
    cap = half_disc_poly(xc, zc, r_out, +1)
    side = unary_union([cheek, arm, cap]).buffer(0)
    # 구멍: PCD14 ×4, 중앙 창 ⌀11, 핀 ⌀1.2
    holes = [circle(hx + 7*math.sin(math.radians(a)), hz + 7*math.cos(math.radians(a)), P["hub_hole_d"]) for a in (45, 135, 225, 315)]
    holes.append(circle(hx, hz, P["hub_window_d"]))
    wx0, wz0, wx1, wz1 = P["arm_window"]; holes.append(sbox(wx0, wz0, wx1, wz1))   # 암 경량 창 (레일 4 mm ×2)
    holes.append(circle(P["tie_bolt_xz"][0], P["tie_bolt_xz"][1], 3.4))             # 타이볼트 관통 (측판)
    pa = math.radians(P["pin_ang_deg"]); holes.append(circle(hx + P["pin_r"]*math.sin(pa), hz + P["pin_r"]*math.cos(pa), P["pin_hole_d"]))
    # 보울 안쪽 (캡은 r_out 반원판이지만 원통 안쪽 r_in 은 비워야 공동이 된다 → 캡은 r_in 안을 남긴다? 아니다: 캡은 막힌 벽(측판). 그대로 둔다.)
    side_h = side
    for h in holes: side_h = side_h.difference(h)
    print("[door] side polygon:", side.geom_type, "→ with holes:", side_h.geom_type,
          [round(g.area, 1) for g in (side_h.geoms if side_h.geom_type == "MultiPolygon" else [side_h])])
    if side_h.geom_type == "MultiPolygon":
        parts = sorted(side_h.geoms, key=lambda g: -g.area)
        print("  ⚠ 조각 분리 — 최대 면적 조각만 사용, 나머지:", [(round(g.area, 1), [round(v, 1) for v in g.bounds]) for g in parts[1:]])
        side_h = parts[0]
    yL = (P["cheek_inner_y"][0] - t, P["cheek_inner_y"][0]); yR = (P["cheek_inner_y"][1], P["cheek_inner_y"][1] + t)
    visual = [extrude_xz_prisms(side_h, *yL), extrude_xz_prisms(side_h, *yR)]
    pieces, names = [], []
    # collision: 뺨 볼록껍질 + 암 상자 + 캡 쐐기 (양쪽)
    for tag, yy in (("L", yL), ("R", yR)):
        ch = extrude_xz(cheek, *yy); pieces.append(trimesh.convex.convex_hull(ch)); names.append(f"cheek_hull_{tag}")
        pieces.append(box_l5(P["arm_x"], yy, (P["arm_z0"], P["arm_step_z"] + 2.0))); names.append(f"arm_{tag}_upper")
        pieces.append(box_l5((xc, P["arm_x"][1]), yy, (P["arm_step_z"], zc))); names.append(f"arm_{tag}_lower")
        for i, w in enumerate(half_cyl(xc, zc, 0.0, r_out, +1, yy[0], yy[1], 6)): pieces.append(w); names.append(f"cap_{tag}_{i:02d}")
    # 원통 벽 (측판 안쪽면 사이)
    y0, y1 = P["cheek_inner_y"]
    for i, w in enumerate(half_cyl(xc, zc, r_in, r_out, +1, y0, y1, P["seg_n"])): pieces.append(w); names.append(f"wall_{i:02d}")
    visual = visual[:2] + [p for p, n in zip(pieces, names) if n.startswith("wall_")]
    # 다리
    # 다리 = 각구멍(타이볼트) 둘레 상자 4개 (볼록 유지, 불리언 없음)
    bx0, bx1 = P["bridge_x"]; bz0, bz1 = P["bridge_z"]; tx, tz = P["tie_bolt_xz"]; h = P["tie_bolt_sq"] / 2.0
    for i, (xx, zz) in enumerate((((bx0, tx - h), (bz0, bz1)), ((tx + h, bx1), (bz0, bz1)), ((tx - h, tx + h), (bz0, tz - h)), ((tx - h, tx + h), (tz + h, bz1)))):
        br = box_l5(xx, (y0, y1), zz); pieces.append(br); names.append(f"bridge_{i}"); visual.append(br)
    return pieces, names, visual, side_h

# ────────────────────────────── 고정부 ──────────────────────────────
def build_fixed(P):
    xc, zc = P["bowl_xc"], P["bowl_zc"]; r_in = P["bowl_r_in"]; r_out = r_in + P["wall"]; t = P["plate_t"]
    pieces, names, visual = [], [], []
    # 판 (구멍 = 시각용 압출, collision = 상자)
    px, py, pz = P["fixed_plate_x"], P["fixed_plate_y"], P["fixed_plate_z"]
    # v1(09-04): v0 는 판에만 구멍을 뚫고 스파인 상자(Z 112.5~150)·플레어를 같은 X 슬래브에 따로 얹어 팁 구멍(Z 116.2)이 메워졌다(실물 2구멍).
    #            → 판 ∪ 플레어 ∪ 스파인을 (Y,Z) 한 다각형으로 합친 뒤 구멍 3개를 빼고 한 번에 압출한다. collision 조각(상자)은 그대로.
    y0c, y1c = P["cheek_inner_y"]; ywc = (y0c - t, y1c + t); fz0c, fz1c = P["spine_flare_z"]
    slab_poly = unary_union([sbox(py[0], pz[0], py[1], pz[1]),
                             Polygon([(py[0], fz0c), (py[1], fz0c), (ywc[1], fz1c), (ywc[0], fz1c)]),
                             sbox(ywc[0], fz1c, ywc[1], P["spine_z1"])]).buffer(0)
    plate_poly = slab_poly
    for (Y, Z) in P["fixed_holes_yz"]: plate_poly = plate_poly.difference(Point(Y, Z).buffer(P["fixed_hole_d"]/2, resolution=16))
    assert len(plate_poly.interiors) == len(P["fixed_holes_yz"]), f"고정부 슬래브 관통 구멍 {len(plate_poly.interiors)} ≠ {len(P['fixed_holes_yz'])}"
    m = trimesh.creation.extrude_polygon(plate_poly, px[1]-px[0]); v = m.vertices.copy()
    m.vertices = np.stack([v[:, 2] + px[0], v[:, 0], v[:, 1]], 1)
    if m.volume < 0: m.invert()
    m.update_faces(m.nondegenerate_faces(height=1e-4)); m.remove_unreferenced_vertices(); m.merge_vertices(); m.fix_normals(); visual.append(m)
    pieces.append(box_l5(px, py, pz)); names.append("plate")
    # 스파인 (판 아래로 → 보울 등과 융합)
    y0, y1 = P["cheek_inner_y"]; yw = (y0 - t, y1 + t)                  # 캡 바깥면 폭 (−21.30, 21.07)
    fz0, fz1 = P["spine_flare_z"]
    sp = box_l5(px, yw, (fz1, P["spine_z1"])); pieces.append(sp); names.append("spine")                             # 넓은 스파인 (Z 112.5~150) — visual 은 슬래브 압출에 포함
    # 45° 플레어: 판 폭(±15) → 캡 폭(±21), Z 107.5~112.5 (볼록 쐐기 1개: 6 점 → hull)
    fl = trimesh.convex.convex_hull(trimesh.Trimesh(vertices=np.array([(x, y, z) for x in px for (y, z) in ((py[0], fz0), (py[1], fz0), (yw[0], fz1), (yw[1], fz1), (py[0], fz1), (py[1], fz1))], float)))
    pieces.append(fl); names.append("spine_flare")                                                                   # visual 은 슬래브 압출에 포함
    # 반원통 벽 (−X)
    y0, y1 = P["cheek_inner_y"]
    for i, w in enumerate(half_cyl(xc, zc, r_in, r_out, -1, y0, y1, P["seg_n"])): pieces.append(w); names.append(f"wall_{i:02d}"); visual.append(w)
    # 캡 (측판, 가동부 측판과 같은 Y 평면)
    yL = (y0 - t, y0); yR = (y1, y1 + t)
    for tag, yy in (("L", yL), ("R", yR)):
        cap = extrude_xz_prisms(half_disc_poly(xc, zc, r_out, -1), *yy); visual.append(cap)
        for i, w in enumerate(half_cyl(xc, zc, 0.0, r_out, -1, yy[0], yy[1], 6)): pieces.append(w); names.append(f"cap_{tag}_{i:02d}")
    return pieces, names, visual

# ────────────────────────────── 게이트 ──────────────────────────────
def surface_pts(meshes, n_per=400):
    pts = []
    for m in meshes:
        k = max(60, int(n_per * m.area / 2000.0)); pts.append(m.sample(k))
    return np.vstack(pts)

def run_gates(P, door_p, door_n, fixed_p, fixed_n, door_vis, fixed_vis):
    G = {}; der = {}
    # G1 볼록·양부피
    bad = [n for m, n in zip(door_p + fixed_p, door_n + fixed_n) if (not m.is_volume) or m.volume < 1.0 or abs(m.convex_hull.volume - m.volume) > 0.02*m.volume + 1e-6]
    G["all_pieces_convex_positive"] = {"pass": len(bad) == 0, "bad": bad, "n": len(door_p) + len(fixed_p)}
    # 질량
    rho = P["density_g_cm3"]/1000.0
    v_door = sum(m.volume for m in door_vis); v_fixed = sum(m.volume for m in fixed_vis)
    hw = sum(n*g for _, n, g in P["screws"])
    tool = (v_door + v_fixed)*rho + hw
    der.update(door_g=round(v_door*rho, 2), fixed_g=round(v_fixed*rho, 2), hardware_g=round(hw, 2), tool_mass_g=round(tool, 2), net_added_g=round(tool - P["stock_jaw_removed_g"], 2))
    G["tool_mass_under_max"] = {"pass": tool <= P["tool_mass_max_g"], "value_g": round(tool, 2), "limit_g": P["tool_mass_max_g"], "note": f"순정 조 {P['stock_jaw_removed_g']} g 제거분은 예산에 안 넣음(보수적)"}
    # 공동 체적 (닫힘 원통)
    w_in = P["cheek_inner_y"][1] - P["cheek_inner_y"][0]; cav = math.pi*P["bowl_r_in"]**2*w_in/1000.0
    der.update(cavity_cm3=round(cav, 2), load_per_scoop_g=round(cav*P["fill_factor"]*P["bulk_density_g_cm3"], 2), inner_width_mm=round(w_in, 2))
    G["cavity_volume_ge_B1"] = {"pass": cav >= P["B1_required_cm3"], "cavity_cm3": round(cav, 2), "B1_cm3": P["B1_required_cm3"]}
    # 립 정합·입 개구
    xc, zc = P["bowl_xc"], P["bowl_zc"]; r_out = P["bowl_r_in"] + P["wall"]
    lip = np.array([xc, 0.0, zc + r_out]); r_lip = np.linalg.norm(lip[[0, 2]] - H)
    def mouth(deg):
        q = rot_pts([lip], deg)[0]; return float(np.linalg.norm(q - lip))
    th = 0.0
    while mouth(th) < P["mouth_target_mm"] and th < 89: th += 0.1
    der.update(lip_radius_from_hinge_mm=round(float(r_lip), 2), open_deg_for_mouth=round(th, 1), top_edge_gap_at_open_mm=round(float(np.linalg.norm(rot_pts([[xc, 0, zc - r_out]], th)[0] - np.array([xc, 0, zc - r_out]))), 1))
    G["mouth_reaches_target_within_servo"] = {"pass": th <= 89.0 and th <= P["open_deg_max"] + 1e-6, "open_deg": round(th, 1), "mouth_mm": P["mouth_target_mm"], "limit_deg": P["open_deg_max"]}
    # 립 힘·자중 모멘트
    der.update(lip_force_max_N=round(P["servo_torque_Nm"]/(r_lip/1000.0), 1))
    # 스윕 간섭: 가동부 표면점 vs 고정부 조각(볼록 포함 검사) + link5 메쉬·서보·베이스 (거리)
    l5 = trimesh.load(REPO / "local_assets/roarm_m3/urdf/meshes/link5.stl", force="mesh")
    fixed_env = [trimesh.load(VS / f, force="mesh") for f in ("gripper_servo_case_SG.stl", "gripper_servo_case_ZK.stl", "gripper_servo_case_XG.stl", "gripper_base.stl", "fixed_jaw.stl")]
    def step_to_l5(m):   # STEP → link5: X=−(y+0.88), Y=346.07−z, Z=x−236.967
        v = m.vertices; m2 = m.copy(); m2.vertices = np.stack([-(v[:, 1] + 0.88), 346.07 - v[:, 2], v[:, 0] - 236.967], 1); return m2
    env = [l5] + [step_to_l5(m) for m in fixed_env]
    # 🔴 거리 = 표면 표본(간격 ≈0.5 mm) KD-트리. 삼각형 근접 질의는 6천 점 × 15 스텝에서 메모리 폭주(OOM, 09-03).
    env_pts = np.vstack([m.sample(min(int(m.area / 0.25) + 1, 150000)) for m in env] + [m.vertices for m in env])
    env_tree = cKDTree(env_pts)
    class _Env:  # nearest.on_surface 호환 어댑터
        def on_surface(self, q): d, _ = env_tree.query(q, workers=4); return None, d
    env_all = type("E", (), {"nearest": _Env()})()
    pts0 = surface_pts(door_vis + [p for p, n in zip(door_p, door_n) if n.startswith(("arm", "bridge"))])
    r_hub = np.linalg.norm(pts0[:, [0, 2]] - H, axis=1)
    mask = r_hub > 12.5                                        # 디스크 접촉 영역 제외
    sweep = []
    for deg in np.arange(1.0, P["open_deg_max"] + 0.01, 2.0):
        q = rot_pts(pts0[mask], deg)
        d_env = env_all.nearest.on_surface(q)[1].min()
        pen = 0
        for fp in fixed_p:
            pen += int(fp.contains(q).sum())
        sweep.append({"deg": float(deg), "min_dist_env_mm": round(float(d_env), 2), "points_inside_fixed": pen})
    worst_env = min(s["min_dist_env_mm"] for s in sweep); pen_open = max(s["points_inside_fixed"] for s in sweep)
    pen0 = int(sum(fp.contains(rot_pts(pts0[mask], 0.0)).sum() for fp in fixed_p))
    G["door_sweep_clears_robot"] = {"pass": worst_env >= 1.0, "min_dist_mm": worst_env, "table": sweep, "excluded": "힌지 반경 12.5 이내(디스크 접촉면)", "method": "표면 표본 KD-트리 거리(±0.5 mm)"}
    G["door_never_enters_fixed_part"] = {"pass": pen_open == 0, "points_inside_when_open": pen_open, "points_inside_closed": pen0, "note": "닫힘(0°) 은 파팅면 접촉이라 참고값만(게이트는 1°~)"}
    # 고정부 vs 로봇 간섭 (블레이드 팁·LED)
    fpts = surface_pts(fixed_vis)
    inside_plate = (fpts[:, 0] > P["fixed_plate_x"][0] - 0.01) & (fpts[:, 0] < P["fixed_plate_x"][1] + 0.01) & (fpts[:, 2] < P["fixed_plate_z"][1] + 0.01) & (fpts[:, 2] > P["fixed_plate_z"][0] - 0.01)
    d_fixed = env_all.nearest.on_surface(fpts[~inside_plate])[1].min()
    G["fixed_part_clears_robot"] = {"pass": d_fixed >= 1.0, "min_dist_mm": round(float(d_fixed), 2), "excluded": "판 자체(블레이드 바깥면 밀착)"}
    # 립 정합: 가동 벽 첫 쐐기(a=0) 와 고정 벽 마지막 쐐기(a=2π) 의 립 모서리 일치
    dw = [p for p, n in zip(door_p, door_n) if n == "wall_00"][0]; fw = [p for p, n in zip(fixed_p, fixed_n) if n == f"wall_{P['seg_n']-1:02d}"][0]
    dl = dw.vertices[np.argsort(dw.vertices[:, 2])[-4:]]; fl = fw.vertices[np.argsort(fw.vertices[:, 2])[-4:]]
    gap = float(np.abs(dl[:, 2].max() - fl[:, 2].max()) + np.abs(dl[:, 0].min() - fl[:, 0].max()))
    G["lips_meet_when_closed"] = {"pass": gap < 0.05, "mismatch_mm": round(gap, 3)}
    # 외팔보 처짐 (측판 2장, 암 띠 구간 L = 보울 중심 − (힌지+7), 립까지 레버)
    b = P["plate_t"]; h = P["arm_x"][1] - P["arm_x"][0]; wx0, _, wx1, _ = P["arm_window"]
    I = 2*(b*h**3/12.0 - b*(wx1 - wx0)**3/12.0); E = P["E_pla_MPa"]; F = P["lip_force_test_N"]   # 창 구간 단면(레일 2) 보수적으로 전 구간 적용
    L = (P["bowl_zc"] - P["bowl_r_in"] - P["wall"]) - (P["hinge_xz"][1] + 7.0); lever = 2*(P["bowl_r_in"] + P["wall"])
    M = F*lever; d_end = F*L**3/(3*E*I) + M*L**2/(2*E*I); slope = F*L**2/(2*E*I) + M*L/(E*I); d_lip = d_end + slope*lever
    G["cantilever_deflection_at_lip"] = {"pass": d_lip <= 2.0, "delta_mm_at_20N": round(d_lip, 2), "L_mm": round(L, 1), "I_mm4": round(I, 1), "model": "측판 2장, 창 구간 레일 단면을 전 구간에 적용(보수), 원통부 강체, E 3 GPa"}
    # 스윕 최저점 (바닥 여유 기준, D478 교훈)
    lows = []
    for deg in np.arange(0.0, P["open_deg_max"] + 0.01, 2.0):
        q = rot_pts(pts0, deg); lows.append((float(deg), round(float(q[:, 2].max()), 2)))
    der["door_lowest_z_sweep"] = lows; der["lip_z_closed"] = round(float(lip[2]), 2)
    # 판 플랜지 한계 (Z ≤ 106.4 구간 Y 범위)
    G["fixed_plate_inside_flange"] = {"pass": P["fixed_plate_y"][0] >= -16.25 + 0.29 and P["fixed_plate_y"][1] <= 15.22 - 0.29, "plate_y": P["fixed_plate_y"], "flange_edges": [-16.25, 15.22]}
    return G, der

def _jsafe(o):
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    raise TypeError(str(type(o)))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.stl"): old.unlink()
    door_p, door_n, door_vis, side_poly = build_door(P)
    fixed_p, fixed_n, fixed_vis = build_fixed(P)
    G, der = run_gates(P, door_p, door_n, fixed_p, fixed_n, door_vis, fixed_vis)
    for m, n in zip(door_p, door_n): m.export(OUT / f"door_{n}.stl")
    for m, n in zip(fixed_p, fixed_n): m.export(OUT / f"fixed_{n}.stl")
    trimesh.util.concatenate(door_vis).export(OUT / "door_ALL.stl"); trimesh.util.concatenate(fixed_vis).export(OUT / "fixed_ALL.stl")
    for i, m in enumerate(door_vis): m.export(OUT / f"doorvis_{i:02d}.stl")      # 관성 계산용 시각 조각(합성 스크립트가 읽음)
    for i, m in enumerate(fixed_vis): m.export(OUT / f"fixedvis_{i:02d}.stl")
    # jaw 프레임(gripper_link) 가동부 시각 메시: (x,y,z)_jaw = (Z−52.035, X, Y−18.821)
    dj = trimesh.util.concatenate(door_vis).copy(); v = dj.vertices; dj.vertices = np.stack([v[:, 2] - 52.035, v[:, 0], v[:, 1] - 18.821], 1); dj.export(OUT / "door_ALL_jawframe.stl")
    src = {f: hashlib.sha256(open(VS / f, "rb").read()).hexdigest()[:16] for f in ("movable_jaw.stl", "fixed_jaw.stl", "gripper_base.stl")}
    ok = all(v["pass"] for v in G.values())
    json.dump({"design": "S1 (D480)", "params": P, "gates": G, "derived": der, "all_gates_pass": ok,
               "piece_counts": {"door": len(door_p), "fixed": len(fixed_p)}, "source_step_parts_sha16": src,
               "frames": {"link5": "모든 STL(door_*/fixed_*) 은 link5 mm", "door_ALL_jawframe.stl": "gripper_link 프레임 mm (URDF 용)"},
               "supersedes": "s1_v0 (09-04 출력·조립 피드백: 뺨 3→2 mm 로 순정 M3×4 사용, 고정부 팁 구멍이 스파인 조각에 메워진 결함 수정) ← g18_nut_trap (D462~D479) ← D480"},
              open(OUT / "design.json", "w"), ensure_ascii=False, indent=1, default=_jsafe)
    print(f"조각  door {len(door_p)} · fixed {len(fixed_p)}   자중 {der['tool_mass_g']} g (door {der['door_g']} + fixed {der['fixed_g']} + hw {der['hardware_g']})")
    print(f"공동 {der['cavity_cm3']} cm³ · 적재 {der['load_per_scoop_g']} g · 립 반경 {der['lip_radius_from_hinge_mm']} · 입 58 에 {der['open_deg_for_mouth']}° · 립 힘 최대 {der['lip_force_max_N']} N")
    for k, v in G.items(): print(("  PASS  " if v["pass"] else "  FAIL  ") + k + "  " + json.dumps({a: b for a, b in v.items() if a not in ("pass", "table")}, ensure_ascii=False, default=_jsafe)[:150])
    print(f"all_gates_pass = {ok} → {OUT}"); return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
