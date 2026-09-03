"""p38 / g2 — 구동 인출부(순정 가동 조 볼트 → 서보 크랭크판) + 브래킷 체결구 자리 3D 검증 (D475 → D476).

무엇을 답하나 (link5 프레임, mm. 정적 검사는 서보 0° = 닫힘, 머리 스윕은 0~89°)
    G10 크랭크판 구멍이 순정 가동 조의 실제 구멍(메쉬 래스터 검출)과 겹치는가
    G11 크랭크판이 가동 조 블레이드 안쪽면에 **닿되 관통하지 않고**, 웨브·암·아이가 조 몸체를 안 뚫는가
    G12 크랭크 체결: 조 바깥면 머리 자리(자유) · 판 안 **너트 포켓**(빈 공간 + 바닥) · 볼트 스택 산술 · 간극 돌출 0
    G13 브래킷 체결 3점: 안쪽(간극) 버튼머리 자리 · 바깥 **너트 트랩**(터널/레일 슬롯: 빈 공간·측벽·캡·꼬리)
        + 머리가 순정 가동 조·크랭크와 서보 0~89° 스윕 중 안 닿는가

역사 (D475, 09-03 1차): 4볼트 사각형 + M2.5x10 + 안쪽 너트 설계에서 G12·G13 FAIL — 두 조 사이 간극 4.05 에
크랭크판 3.0 이 들어가 잔여 1.03 뿐인데 고정 조 Z 83.46 쌍 ↔ 가동 조 Z 82.98 쌍이 같은 자리였다.
→ D476: Z 83.46 쌍 포기(3점) + 너트 트랩 종단. 이 판은 그 설계를 검사한다. 옛 쌍은 "dropped" 로 기록.

체결구 치수는 ISO 공칭치(P["m25_nut"], P["m25_button_head"]). **실물 대조 전 근사값.** 순수 기하.
사용:  python sim_scripts/p38_g2_drive_extraction_probe.py [출력디렉터리]   (기본 = g18_nut_trap/p38_drive)
"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else REPO / "claudedocs/runtime_logs/grab_track/g18_nut_trap/p38_drive")
sys.path.insert(0, str(REPO / "sim_scripts"))
_argv, sys.argv = sys.argv, ["x"]
import p37_g2_grab_v1_attach_probe as p37          # placement/jaw_in_link5/상수 재사용
sys.argv = _argv
G, P = p37.G, p37.P

BLADE_X = p37.BLADE_X
MOUNT_HOLES_YZ = list(p37.MOUNT_HOLES_YZ)
DROPPED_YZ = [h for h in p37.BLADE_HOLES_ALL_YZ if h not in p37.MOUNT_HOLES_YZ]
NUT, HEAD = P["m25_nut"], P["m25_button_head"]
CLEAR = P["fastener_clear_mm"]


def free_depth(occupiers, yz, x0, sgn, r, max_depth=12.0, step=0.05):
    """(yz) 중심, 반경 r 원통을 x0 에서 sgn 방향으로 밀며 처음 재료를 만나는 깊이. 없으면 max_depth."""
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([np.zeros_like(ang), np.cos(ang), np.sin(ang)], 1)
    disc = np.vstack([np.zeros((1, 3))] + [ring * rr for rr in (r * 0.5, r)])
    for d in np.arange(0.0, max_depth + 1e-9, step):
        pts = disc.copy()
        pts[:, 0] = x0 + sgn * d; pts[:, 1] += yz[0]; pts[:, 2] += yz[1]
        for occ in occupiers:
            if (occ(pts) if callable(occ) else occ.contains(pts)).any():
                return float(d)
    return float(max_depth)


def any_contains(meshes, pts):
    hit = np.zeros(len(pts), bool)
    for m in meshes:
        hit |= m.contains(pts)
    return hit


def box_pts(cx, cy, cz, lx, ly, lz, n=5):
    xs = np.linspace(cx - lx / 2, cx + lx / 2, n); ys = np.linspace(cy - ly / 2, cy + ly / 2, n)
    zs = np.linspace(cz - lz / 2, cz + lz / 2, n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def raster_holes(mesh_or_list, x_plane, y_rng, z_rng, pitch=0.25):
    ys = np.arange(*y_rng, pitch); zs = np.arange(*z_rng, pitch)
    YY, ZZ = np.meshgrid(ys, zs, indexing="ij")
    pts = np.stack([np.full(YY.size, x_plane), YY.ravel(), ZZ.ravel()], 1)
    ms = mesh_or_list if isinstance(mesh_or_list, list) else [mesh_or_list]
    inside = any_contains(ms, pts).reshape(YY.shape)
    holes = ndimage.binary_fill_holes(inside) & ~inside
    lab, n = ndimage.label(holes)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        a = sel.sum() * pitch * pitch
        if a < 2.0:
            continue
        out.append({"y": round(float(YY[sel].mean()), 2), "z": round(float(ZZ[sel].mean()), 2),
                    "eq_d_mm": round(2 * math.sqrt(a / math.pi), 2), "area_mm2": round(float(a), 2)})
    return out


def section_segments(mesh, origin, normal):
    s = mesh.section(plane_origin=origin, plane_normal=normal)
    if s is None:
        return []
    v = s.vertices
    return [(v[e.points][i], v[e.points][i + 1]) for e in s.entities for i in range(len(e.points) - 1)]


def head_mesh(yz):
    """안쪽(간극)에 앉는 버튼머리: 고정 조 안쪽면 x=BLADE_X[1] 에서 +x 로 k."""
    c = trimesh.creation.cylinder(radius=HEAD["dk_mm"] / 2.0, height=HEAD["k_mm"], sections=32)
    c.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    c.apply_translation([BLADE_X[1] + HEAD["k_mm"] / 2.0, yz[0], yz[1]])
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = p37.placement()
    jaw = p37.jaw_in_link5()
    link5 = p37.load_mm("link5.stl")
    br, nB = G.build_bracket(P)
    dr, nD, lk = G.build_linkage(P)

    def placed(parts, names, pred):
        out = []
        for m0, nm in zip(parts, names):
            if pred(nm):
                m = m0.copy(); m.apply_transform(T); out.append((m, nm))
        return out
    crank = placed(dr, nD, lambda n: G.linkage_group(n) == "servocrank")
    plate = [(m, n) for m, n in crank if n.startswith("servocrank_plate") or n.startswith("servocrank_pocket")]
    floor_pcs = [(m, n) for m, n in crank if n.startswith("servocrank_plate")]
    bolt_plates = placed(br, nB, lambda n: n.startswith("bolt_plate"))
    bracket_all = placed(br, nB, lambda n: True)
    bracket_rest = placed(br, nB, lambda n: not n.startswith("bolt_plate"))

    pl_lo = np.min([m.bounds[0] for m, _ in plate], 0); pl_hi = np.max([m.bounds[1] for m, _ in plate], 0)
    bp_lo = np.min([m.bounds[0] for m, _ in bolt_plates], 0); bp_hi = np.max([m.bounds[1] for m, _ in bolt_plates], 0)
    jaw_inner_x = float(jaw.vertices[:, 0][(jaw.vertices[:, 2] > 78) & (jaw.vertices[:, 2] < 88)].min())

    def blade_outer_at(yz, r=2.3):
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        for x in np.arange(jaw_inner_x + 0.02, jaw_inner_x + 6.0, 0.02):
            q = np.stack([np.full_like(ang, x), yz[0] + r * np.cos(ang), yz[1] + r * np.sin(ang)], 1)
            if not jaw.contains(q).any():
                return float(round(x - 0.01, 2))
        return float("nan")
    jaw_outer_per_hole = [blade_outer_at(tuple(h)) for h in P["jaw_bolt_yz_mm"]]
    jaw_outer_x = float(max(jaw_outer_per_hole))
    gap_blades = jaw_inner_x - BLADE_X[1]
    gap_residual = float(pl_lo[0] - BLADE_X[1])
    plate_thk = float(pl_hi[0] - pl_lo[0])
    gates, ev = {}, {}
    ev["frames"] = "link5 프레임, mm, 서보 0°(닫힘). 체결구 = ISO 공칭(P m25_nut/m25_button_head), 실물 대조 필요"
    ev["fixed_blade_x"] = list(BLADE_X)
    ev["movable_blade_x"] = [round(jaw_inner_x, 2), round(jaw_outer_x, 2)]
    ev["movable_blade_outer_x_per_hole"] = jaw_outer_per_hole
    ev["gap_between_blades_mm"] = round(gap_blades, 3)
    ev["crank_plate_bounds"] = [np.round(pl_lo, 2).tolist(), np.round(pl_hi, 2).tolist()]
    ev["bracket_bolt_plate_bounds"] = [np.round(bp_lo, 2).tolist(), np.round(bp_hi, 2).tolist()]
    ev["gap_residual_fixed_blade_to_crank_plate_mm"] = round(gap_residual, 3)
    ev["mount_holes_yz"] = MOUNT_HOLES_YZ
    ev["dropped_holes_yz_D475"] = DROPPED_YZ
    ev["free_depth_step_mm"] = 0.05
    ev["fasteners_nominal"] = {"nut": NUT, "button_head": HEAD, "clear_mm": CLEAR}

    # ── G10 크랭크판 구멍 ≡ 순정 가동 조 구멍 ──────────────────────────────
    x_mid = (jaw_inner_x + jaw_outer_x) / 2.0
    holes = raster_holes(jaw, x_mid, (-21, 21), (60, 121), 0.25)
    design = [tuple(h) for h in P["jaw_bolt_yz_mm"]]
    match, errs = [], []
    for (hy, hz) in design:
        best = min(holes, key=lambda h: math.hypot(h["y"] - hy, h["z"] - hz))
        e = math.hypot(best["y"] - hy, best["z"] - hz)
        errs.append(e)
        cxf = float(np.mean([(m.bounds[0][0] + m.bounds[1][0]) / 2 for m, _ in floor_pcs]))
        best["floor_leaves_hole"] = not any_contains([m for m, _ in floor_pcs], np.array([[cxf, hy, hz]]))[0]
        match.append({"design_yz": [hy, hz], "mesh_hole": best, "err_mm": round(e, 3)})
    gates["G10_crank_holes_match_stock_jaw"] = {
        "pass": max(errs) <= 0.3 and all(m["mesh_hole"]["floor_leaves_hole"] for m in match),
        "max_err_mm": round(max(errs), 3), "matches": match, "all_mesh_holes_in_blade": holes,
        "why": "크랭크판은 순정 가동 조의 기존 구멍(스팬 25.11)으로 물린다(D462 §5). 구멍이 안 맞으면 인출 자체가 없다",
        "blind_spot": "래스터 0.25 mm 근사 — 지름은 ±0.3 오차. 나사산 유무는 메쉬로 못 본다(실물 확인 항목)"}

    # ── G11 크랭크판 좌면 접촉 / 조 몸체 무관통 ─────────────────────────────
    jaw_cloud = p37.surface_cloud(jaw, 200000)
    seat_gap = float(jaw_inner_x - pl_hi[0])
    pen = [{"piece": nm, "clearance_mm": round(p37.exact_clearance(m, jaw_cloud), 3)} for m, nm in crank]
    worst = min(pen, key=lambda r: r["clearance_mm"])
    gates["G11_crank_plate_seats_on_jaw"] = {
        "pass": (-0.05 <= seat_gap <= 0.1) and worst["clearance_mm"] >= -0.05,
        "seat_gap_mm": round(seat_gap, 3), "seat_rule": "판은 블레이드에 닿아야 한다: -0.05(관통 허용치) ~ +0.1",
        "worst_piece": worst, "per_piece": pen,
        "why": "판이 블레이드 면에 밀착해야 볼트 체결이 성립한다. 웨브·암은 조 몸체(측벽·칼라)를 뚫으면 안 된다"}

    # ── G12 크랭크 체결: 바깥 머리 자리 + 판 안 너트 포켓 + 스택 산술 ─────────
    blade_t = jaw_outer_x - jaw_inner_x
    floor_t = P["jaw_mount_thk_mm"] - P["crank_pocket_depth_mm"]
    engage = P["crank_bolt_len_mm"] - blade_t - floor_t
    protrude = max(P["crank_bolt_len_mm"] - blade_t - plate_thk, 0.0)
    per12 = []
    crank_meshes = [m for m, _ in crank]
    for (hy, hz) in design:
        out_free = free_depth([jaw], (hy, hz), jaw_outer_x, +1, r=HEAD["dk_mm"] / 2.0 + 0.3)
        # 포켓 빈 공간: 너트 박스(x 두께 h, Y 대변 af, Z 대각 corner) — 판 -x 면부터 안쪽으로
        nb = box_pts(pl_lo[0] + NUT["h_mm"] / 2.0, hy, hz, NUT["h_mm"], NUT["af_mm"], NUT["corner_mm"], n=7)
        pocket_free = not any_contains(crank_meshes, nb).any()
        # 포켓 측벽(회전 구속): Y 대변 밖 0.5 에 재료
        ring = np.array([[pl_lo[0] + 1.0, hy + s * (NUT["af_mm"] / 2 + P["nut_trap_clear_mm"] / 2 + 0.4), hz]
                         for s in (+1, -1)])
        walls = bool(any_contains(crank_meshes, ring).all())
        # 바닥: 포켓 밑(x = 판 -x 면 + pocket_depth + floor/2) 구멍 둘레에 재료
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        fl = np.stack([np.full_like(ang, pl_lo[0] + P["crank_pocket_depth_mm"] + floor_t / 2),
                       hy + 2.3 * np.cos(ang), hz + 2.3 * np.sin(ang)], 1)
        floor_ok = bool(any_contains(crank_meshes, fl).all())
        per12.append({"hole_yz": [hy, hz], "outboard_free_depth_mm": out_free,
                      "head_needs_mm": round(HEAD["k_mm"] + CLEAR, 2),
                      "pocket_void_free": pocket_free, "pocket_walls_present": walls, "floor_present": floor_ok})
    g12_ok = all(r["outboard_free_depth_mm"] >= HEAD["k_mm"] + CLEAR and r["pocket_void_free"]
                 and r["pocket_walls_present"] and r["floor_present"] for r in per12) \
        and engage >= 1.4 and protrude <= gap_residual - CLEAR and floor_t >= 0.8
    gates["G12_crank_fastener_envelope"] = {
        "pass": bool(g12_ok), "per_hole": per12, "plate_thk_mm": round(plate_thk, 2),
        "stack": {"bolt": f"M2.5x{P['crank_bolt_len_mm']:g} button, 조 바깥면에서", "blade_thk_mm": round(blade_t, 2),
                  "floor_thk_mm": round(floor_t, 2), "nut_engagement_mm": round(engage, 2),
                  "protrusion_into_gap_mm": round(protrude, 2), "residual_gap_mm": round(gap_residual, 3),
                  "residual_needed_mm": round(protrude + CLEAR, 2)},
        "why": ("D475: 판 -x 면 밖으로 나올 수 있는 높이가 잔여 1.03-0.3 = 0.73 < 표준 최소 1.35 라, 볼트는 조 바깥면에서 "
                "넣고 판 안 포켓 너트로 끝내야 한다. 포켓의 빈 공간·측벽·바닥이 실제 조각으로 있는지 본다"),
        "blind_spot": "너트 물림 1.49 mm(3.3 산)·PLA 바닥 1.0 압축 강도는 실물 시험. 렌치(육각 1.5) 접근은 조를 연 상태 전제"}

    # ── G13 브래킷 3점 (D476 v2): 쌍 구멍 = 바깥 볼트(머리 판 위) + 안쪽 너트 · 팁 = 안쪽 머리 + 레일 슬롯 너트 ──
    #    link5 플랜지(|Y|>15.2/16.25, Z<=106.4)가 쌍 구멍 바깥을 막아 트랩을 못 붙인다 → 방향을 뒤집었다.
    occ_in = [jaw] + crank_meshes
    bracket_meshes = [m for m, _ in bracket_all]
    l5_cloud = p37.surface_cloud(link5, 300000)
    x_face = float(bp_lo[0])                      # 판 바깥면
    td = P["rail_slot_depth_mm"]; slot_w = NUT["af_mm"] + P["nut_trap_clear_mm"]
    mount_tail = P["mount_bolt_len_mm"] - blade_t - P["bracket_thk_mm"] - NUT["h_mm"]
    rows13 = []
    for (hy, hz) in MOUNT_HOLES_YZ:
        is_tip = abs(hz - 115.91) < 0.5
        plate_hole = not any_contains([m for m, _ in bolt_plates], np.array([[(bp_lo[0] + bp_hi[0]) / 2, hy, hz]])).any()
        if is_tip:
            in_free = free_depth(occ_in, (hy, hz), BLADE_X[1], +1, r=HEAD["dk_mm"] / 2.0 + 0.3)
            need_in = HEAD["k_mm"] + CLEAR
            nb = box_pts(x_face - NUT["h_mm"] / 2.0, hy, hz, NUT["h_mm"], NUT["corner_mm"], NUT["af_mm"], n=7)
            void_ok = not any_contains(bracket_meshes, nb).any()
            tail_ok = not any_contains(bracket_meshes, np.array([[x_face - NUT["h_mm"] - mount_tail - 0.1, hy, hz]])).any()
            walls_ok = bool(any_contains(bracket_meshes, np.array([[x_face - 1.0, hy, hz + sg * (slot_w / 2 + 0.5)] for sg in (+1, -1)])).all())
            cap_ok = bool(any_contains(bracket_meshes, np.array([[x_face - td - 0.5, hy, hz]])).all())
            # 슬롯 안 너트 vs link5 (플랜지 밖이어야 함)
            nutm = trimesh.creation.box(extents=[NUT["h_mm"], NUT["corner_mm"], NUT["af_mm"]])
            nutm.apply_translation([x_face - NUT["h_mm"] / 2.0, hy, hz])
            out_vs_link5 = p37.exact_clearance(nutm, l5_cloud)
            rows13.append({"hole_yz": [hy, hz], "scheme": "inside_out (머리 간극 / 너트 레일 슬롯)",
                           "inboard_free_depth_mm": in_free, "inboard_needed_mm": round(need_in, 2),
                           "plate_has_hole": plate_hole, "slot_void_free": void_ok, "tail_space_free": tail_ok,
                           "slot_walls_present": walls_ok, "slot_cap_present": cap_ok,
                           "outboard_element_vs_link5_mm": round(out_vs_link5, 3),
                           "ok": bool(in_free >= need_in and plate_hole and void_ok and tail_ok and walls_ok and cap_ok and out_vs_link5 >= CLEAR)})
        else:
            in_free = free_depth(occ_in, (hy, hz), BLADE_X[1], +1, r=NUT["corner_mm"] / 2.0 + 0.3)
            need_in = NUT["h_mm"] + mount_tail + CLEAR
            hm = trimesh.creation.cylinder(radius=HEAD["dk_mm"] / 2.0, height=HEAD["k_mm"], sections=32)
            hm.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
            hm.apply_translation([x_face - HEAD["k_mm"] / 2.0, hy, hz])
            head_void = not any_contains(bracket_meshes, box_pts(x_face - HEAD["k_mm"] / 2.0, hy, hz, HEAD["k_mm"], HEAD["dk_mm"], HEAD["dk_mm"], n=7)).any()
            out_vs_link5 = p37.exact_clearance(hm, l5_cloud)
            # 육각 렌치 경로: 판 바깥면에서 -x 로 25 mm, 반경 2.5 → link5·브래킷 무점유
            key = box_pts(x_face - 12.5, hy, hz, 25.0, 5.0, 5.0, n=9)
            key_ok = (not any_contains(bracket_meshes, key).any()) and \
                     (p37.exact_clearance(trimesh.creation.box(extents=[25.0, 5.0, 5.0]).apply_translation([x_face - 12.5, hy, hz]), l5_cloud) >= 0.0)
            rows13.append({"hole_yz": [hy, hz], "scheme": "outside_in (머리 판 바깥면 / 너트 간극)",
                           "inboard_free_depth_mm": in_free, "inboard_needed_mm": round(need_in, 2),
                           "plate_has_hole": plate_hole, "head_seat_void_free": head_void,
                           "outboard_element_vs_link5_mm": round(out_vs_link5, 3), "hex_key_path_free": bool(key_ok),
                           "ok": bool(in_free >= need_in and plate_hole and head_void and out_vs_link5 >= CLEAR and key_ok)})
    # 간극 안 요소(쌍: 너트+꼬리 / 팁: 머리) vs 순정 가동 조 + 크랭크 (서보 0~89°)
    inner_elems = []
    for (hy, hz) in MOUNT_HOLES_YZ:
        if abs(hz - 115.91) < 0.5:
            inner_elems.append((head_mesh((hy, hz)), f"tip_head@{hz}"))
        else:
            c = trimesh.creation.cylinder(radius=NUT["corner_mm"] / 2.0, height=NUT["h_mm"] + mount_tail, sections=6)
            c.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
            c.apply_translation([BLADE_X[1] + (NUT["h_mm"] + mount_tail) / 2.0, hy, hz])
            inner_elems.append((c, f"pair_nut@{hy}"))
    rows = lk["rows"]
    sweep = []
    for i in np.linspace(0, len(rows) - 1, 9).astype(int):
        row = rows[int(i)]
        Tj = p37.rot_about(p37.GRIPPER_ORIGIN, p37.GRIPPER_AXIS, math.radians(row["servo_deg"]))
        jw = jaw.copy(); jw.apply_transform(Tj)
        cloud = p37.surface_cloud(jw, 60000)
        Ts, Tr, Tk = G.linkage_pose(P, lk, int(i))
        cr = []
        for m0, nm in zip(dr, nD):
            if G.linkage_group(nm) == "servocrank":
                m = m0.copy(); m.apply_transform(T @ Ts); cr.append(m)
        worst_h = np.inf; who = None
        for h, tag in inner_elems:
            c = p37.exact_clearance(h, cloud)
            if c < worst_h:
                worst_h, who = c, f"jaw:{tag}"
            hc = p37.surface_cloud(h, 4000)
            for m in cr:
                c2 = p37.exact_clearance(m, hc)
                if c2 < worst_h:
                    worst_h, who = c2, f"crank:{tag}"
        sweep.append({"servo_deg": round(row["servo_deg"], 1), "min_clear_mm": round(float(worst_h), 3), "closest": who})
    sweep_min = min(r["min_clear_mm"] for r in sweep)
    g13_ok = all(r["ok"] for r in rows13) and sweep_min >= CLEAR and len(rows13) == 3
    gates["G13_bracket_fastener_envelope"] = {
        "pass": bool(g13_ok), "per_hole": rows13,
        "stack": {"bolt": f"M2.5x{P['mount_bolt_len_mm']:g} button x3", "scheme": P["mount_bolt_scheme"],
                  "head_k_mm": HEAD["k_mm"], "nut_h_mm": NUT["h_mm"], "tail_beyond_nut_mm": round(mount_tail, 2),
                  "rail_slot_depth_mm": td, "flange_edges_y": P["link5_flange_y_edges"]},
        "inner_elements_sweep_vs_jaw_and_crank": sweep, "inner_elements_sweep_min_clear_mm": sweep_min,
        "dropped_holes_yz": DROPPED_YZ,
        "why": ("D475/D476: Z 83.46 쌍은 크랭크판 자리(잔여 1.03)라 버리고 Z 102.9 쌍 + 팁 구멍 3점. link5 플랜지가 쌍 구멍 "
                "바깥을 막아 쌍은 바깥 볼트 + 안쪽 너트(간극 4.05 에 2.49), 팁은 안쪽 머리 + 레일 슬롯 너트. 간극 안 요소는 "
                "조가 닫히고 열리는 동안 순정 가동 조·크랭크판과 안 닿아야 한다"),
        "blind_spot": "정적 자세 + 9점 스윕. 너트를 간극에서 잡는 손/공구 여유·팁 블레이드 강도는 안 본다. 조를 연 상태 전제"}

    for v in gates.values():
        v["pass"] = bool(v["pass"])
    ok = all(v["pass"] for v in gates.values())
    dj = OUT.parent / "design.json"
    res = {"probe": "p38_g2_drive_extraction", "source_dir": str(OUT.parent.relative_to(REPO)) if dj.exists() else None,
           "design_json_sha256_16": hashlib.sha256(dj.read_bytes()).hexdigest()[:16] if dj.exists() else None,
           "gates": gates, "evidence": ev, "all_gates_pass": ok,
           "verdict": "G2_DRIVE_EXTRACTION_OK" if ok else "G2_DRIVE_EXTRACTION_BLOCKED"}
    json.dump(res, open(OUT / "p38_results.json", "w"), ensure_ascii=False, indent=2,
              default=lambda o: bool(o) if isinstance(o, np.bool_) else int(o) if isinstance(o, np.integer)
              else float(o) if isinstance(o, np.floating) else str(o))

    # ── 단면 그림 (판정 시점 스냅샷, D324) ───────────────────────────────────
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.5))

    def draw(ax, lines, fills, origin, normal, ix, iy):
        for m, col, lab in lines:
            first = True
            for a, b in section_segments(m, origin, normal):
                ax.plot([a[ix], b[ix]], [a[iy], b[iy]], color=col, lw=0.9, label=lab if first else None); first = False
        for m, col, lab in fills:
            if m.section(plane_origin=origin, plane_normal=normal) is None:
                continue
            lo, hi = m.bounds
            ax.add_patch(Rectangle((lo[ix], lo[iy]), hi[ix] - lo[ix], hi[iy] - lo[iy], fc=col, ec="none", alpha=0.55, label=lab))
    fills_b = [(m, "tab:green", "브래킷(판+트랩+레일)" if i == 0 else None) for i, (m, _) in enumerate(bracket_all)]
    fills_c = [(m, "tab:orange", "크랭크판(바닥+포켓)" if i == 0 else None) for i, (m, _) in enumerate(plate)]
    lines = [(link5, "k", "link5(고정 조)"), (jaw, "tab:blue", "순정 가동 조")]

    ax = axs[0]  # XZ @ Y=-13.34 : 쌍 구멍 + 크랭크 포켓 + 버려진 Z83
    yc = MOUNT_HOLES_YZ[0][0]
    draw(ax, lines, fills_c + fills_b, [0, yc, 0], [0, 1, 0], 0, 2)
    for (hy, hz) in MOUNT_HOLES_YZ[:2]:
        ax.add_patch(Rectangle((BLADE_X[1], hz - NUT["corner_mm"] / 2), NUT["h_mm"] + mount_tail, NUT["corner_mm"], fc="red", alpha=0.5, label="너트+꼬리 2.49(안쪽 간극)" if hy < 0 else None))
        ax.add_patch(Rectangle((x_face - HEAD["k_mm"], hz - HEAD["dk_mm"] / 2), HEAD["k_mm"], HEAD["dk_mm"], fc="purple", alpha=0.5, label="버튼머리 1.35(판 바깥)" if hy < 0 else None))
    for (hy, hz) in design:
        if hy < 0:
            ax.add_patch(Rectangle((pl_lo[0], hz - NUT["corner_mm"] / 2), NUT["h_mm"], NUT["corner_mm"], fc="red", alpha=0.5))
            ax.text(pl_lo[0] - 0.3, hz + 3.6, "크랭크 포켓 너트", fontsize=8, ha="right")
    for (hy, hz) in DROPPED_YZ:
        if hy < 0:
            ax.annotate("Z83.46 쌍 포기(D475)", xy=(BLADE_X[1], hz), xytext=(-17.5, hz - 9), fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-22, 4); ax.set_ylim(70, 124); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("link5 X (mm)"); ax.set_ylabel("link5 Z (mm)"); ax.set_title(f"XZ 단면 @ Y={yc}: 쌍 구멍(바깥 머리·안쪽 너트) + 크랭크 포켓")
    ax.legend(fontsize=7, loc="lower left")

    ax = axs[1]  # XY @ Z=102.9
    zc = MOUNT_HOLES_YZ[0][1]
    draw(ax, lines, fills_c + fills_b, [0, 0, zc], [0, 0, 1], 0, 1)
    for (hy, hz) in MOUNT_HOLES_YZ[:2]:
        ax.add_patch(Rectangle((BLADE_X[1], hy - NUT["corner_mm"] / 2), NUT["h_mm"] + mount_tail, NUT["corner_mm"], fc="red", alpha=0.5))
        ax.add_patch(Rectangle((x_face - HEAD["k_mm"], hy - HEAD["dk_mm"] / 2), HEAD["k_mm"], HEAD["dk_mm"], fc="purple", alpha=0.5))
    ax.set_xlim(-22, 4); ax.set_ylim(-22, 22); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("link5 X (mm)"); ax.set_ylabel("link5 Y (mm)"); ax.set_title(f"XY 단면 @ Z={zc}: 너트(간극 {gap_blades:.2f}) · 머리(판 바깥) · link5 플랜지")

    ax = axs[2]  # XY @ Z=115.91 : 팁 슬롯
    zc = MOUNT_HOLES_YZ[2][1]
    draw(ax, lines, fills_c + fills_b, [0, 0, zc], [0, 0, 1], 0, 1)
    hy = MOUNT_HOLES_YZ[2][0]
    ax.add_patch(Rectangle((BLADE_X[1], hy - HEAD["dk_mm"] / 2), HEAD["k_mm"], HEAD["dk_mm"], fc="purple", alpha=0.5))
    ax.add_patch(Rectangle((x_face - NUT["h_mm"], hy - NUT["corner_mm"] / 2), NUT["h_mm"], NUT["corner_mm"], fc="red", alpha=0.5))
    ax.set_xlim(-32, 4); ax.set_ylim(-22, 22); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("link5 X (mm)"); ax.set_ylabel("link5 Y (mm)"); ax.set_title(f"XY 단면 @ Z={zc}: 팁 구멍 — 레일 뿌리 너트 슬롯")
    fig.suptitle(f"p38 체결구 자리 (D476 v2: 쌍=바깥 볼트·안쪽 너트 / 팁=레일 슬롯 / 크랭크=포켓) — verdict {res['verdict']}", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "p38_fastener_sections.png", dpi=150)

    for k2, v in gates.items():
        print(("  PASS  " if v["pass"] else "  FAIL  ") + k2)
        for kk, vv in v.items():
            if kk in ("pass", "per_piece", "all_mesh_holes_in_blade", "per_hole", "matches", "head_sweep_vs_jaw_and_crank"):
                continue
            print(f"           {kk}: {json.dumps(vv, ensure_ascii=False, default=str)[:170]}")
    print(f"verdict = {res['verdict']}   -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
