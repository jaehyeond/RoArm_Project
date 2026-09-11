"""W3b ① — S1 고정 셸이 더미에 들어갈 때 터지는 DEME 발산의 최소 재현·귀속 하네스.

W3 보고(plunge10 타임라인 1513→1514: 립 12 µm 이동 사이에 단일 접촉 0.054→68 N, 입자 0.95→41 m/s)를
입자 1개 / 27개 격자 / 실더미 절편 + 고정 셸 1매로 재현하고, 매 sync 마다 최대 힘 접촉의
구 id·삼각형 id·면 그룹·DEME 법선 vs 내 면 법선·중심-면 부호거리·관입(r−거리)·구 속도·구 중심의 셸 내부 여부를 기록한다.
후보를 토글로 끄고 켠다: (a) --wall-extra 벽 두께 (b) --vz --dt 스텝당 관입 (c) --subdiv 삼각형 크기
(d) --groups 면 그룹(inner,outer,cap_in,cap_out,part) (e) --flip 법선 뒤집기.
셸 기하·프레임은 sim_deme_scoop_s1.py 와 같다(공동 r 20 · 폭 ±18.2 · 립 = 최하점 = owner 원점, 툴 수직, 입 −Y 쪽).
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent
PILE = REPO / "claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz"
R_W = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)      # link5 → 세계
C = np.array([8.1, 0.0, 145.0]); R_IN, WALL, CAP, H_IN = 20.0, 1.6, 2.0, 18.2
GROUPS = ("inner", "outer", "cap_in", "cap_out", "part")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def half_bowl(extra, groups, n_th=24, subdiv=0.0, flip=False, side=-1, flip_part=False):
    """면 그룹 태그가 붙은 반쪽 보울 셸(link5 mm). 반환: (Trimesh, group-per-face)."""
    r_out = R_IN + WALL + extra; H = H_IN + CAP + extra; h = H_IN
    th = np.linspace(0.0, side * math.pi, n_th + 1)
    pt = lambda t, r, y: np.array([C[0] + r * math.sin(t), y, C[2] + r * math.cos(t)])
    V, F, G = [], [], []

    def add(poly, outward, g):
        i = len(V); V.extend(poly); n = np.cross(np.asarray(poly[1]) - poly[0], np.asarray(poly[2]) - poly[0])
        tris = [[i, i + 1, i + 2]] + ([[i, i + 2, i + 3]] if len(poly) == 4 else [])
        F.extend(tris if np.dot(n, outward) >= 0 else [[a, c, b] for a, b, c in tris]); G.extend([g] * len(tris))
    for k in range(n_th):
        t0, t1 = th[k], th[k + 1]; rd = pt((t0 + t1) / 2, 1, 0) - np.array([C[0], 0, C[2]])
        add([pt(t0, R_IN, -h), pt(t1, R_IN, -h), pt(t1, R_IN, h), pt(t0, R_IN, h)], -rd, "inner")
        add([pt(t0, r_out, -H), pt(t1, r_out, -H), pt(t1, r_out, H), pt(t0, r_out, H)], rd, "outer")
        for yi, yo in ((h, H), (-h, -H)):
            ny = np.array([0, np.sign(yi), 0])
            add([pt(t0, 0, yi), pt(t0, R_IN, yi), pt(t1, R_IN, yi)], -ny, "cap_in")
            add([pt(t0, 0, yo), pt(t0, r_out, yo), pt(t1, r_out, yo)], ny, "cap_out")
    # 파팅면(x = 8.1 평면)의 바깥 = 반대쪽 반쪽 방향: 고정(side −1) → +X, 문(side +1) → −X.
    # 🔴 W3 는 여기에 (0,0,cos t) 를 줘서(면 법선 ±X 와 직교) 방향이 임의였고 일부가 뒤집혔다.
    #    DEME triangle_sphere_CD 는 양면이라 뒤집힌 면의 발자국 안·뒤쪽 구에 관입 r+|h| (수 mm, ~60 N) 를 준다.
    nx = np.array([-side, 0, 0], float) if not flip_part else np.array([side, 0, 0], float)
    for t in (th[0], th[-1]):
        add([pt(t, R_IN, -h), pt(t, r_out, -h), pt(t, r_out, h), pt(t, R_IN, h)], nx, "part")
        for yi, yo in ((h, H), (-h, -H)):
            add([pt(t, 0, yi), pt(t, r_out, yi), pt(t, r_out, yo), pt(t, 0, yo)], nx, "part")
    V, F, G = np.asarray(V, float), np.asarray(F), np.asarray(G)
    keep = np.isin(G, list(groups)); F, G = F[keep], G[keep]
    if subdiv > 0:
        V, F, idx = trimesh.remesh.subdivide_to_size(V, F, max_edge=subdiv, return_index=True); G = G[idx]
    m = trimesh.Trimesh(V, F, process=False); m.merge_vertices()
    if flip:
        m.invert()
    return m, G, r_out, H


def classify(p5, r_out, H):
    """구 중심(link5 mm) 이 셸 재료 안인가: wall / cap / cavity / outside (고정 반쪽 = x ≤ 8.1 쪽)."""
    rad = np.hypot(p5[0] - C[0], p5[2] - C[2]); y = abs(p5[1]); on_side = p5[0] <= C[0] + 1e-9
    if not on_side:
        return "outside"
    if R_IN <= rad <= r_out and y <= H_IN:
        return "wall"
    if rad <= r_out and H_IN <= y <= H:
        return "cap"
    if rad < R_IN and y < H_IN:
        return "cavity"
    return "outside"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["one", "lattice", "cut"], default="one")
    ap.add_argument("--cell", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--wall-extra", type=float, default=3.0); ap.add_argument("--subdiv", type=float, default=0.0)
    ap.add_argument("--flip", action="store_true"); ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--flip-part", action="store_true", help="(e) 파팅면 법선만 일부러 안쪽으로(W3 결함 재현)")
    ap.add_argument("--vz", type=float, default=25.0, help="하강 mm/s"); ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--sync", type=float, default=2.5e-4); ap.add_argument("--E", type=float, default=5e6)
    ap.add_argument("--Emesh", type=float, default=3e9); ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--cor", type=float, default=0.3); ap.add_argument("--crr", type=float, default=0.05)
    ap.add_argument("--yoff", type=float, default=0.0, help="one: 구 중심의 립선 기준 세계 y 오프셋 mm (+ = 바깥 호 아래)")
    ap.add_argument("--xoff", type=float, default=0.0, help="one: 세계 x (립선 방향) 오프셋 mm, ±20.7 = 캡 아래")
    ap.add_argument("--plunge", type=float, default=25.0); ap.add_argument("--cd", type=int, default=20)
    ap.add_argument("--max-wall-s", type=float, default=280.0); ap.add_argument("--vstop", type=float, default=20.0)
    ap.add_argument("--cut-half", type=float, default=45.0, help="cut: 절편 반폭 mm")
    ap.add_argument("--errvel", type=float, default=50.0); ap.add_argument("--lat-spacing", type=float, default=1.02)
    ap.add_argument("--detail-force", type=float, default=1.0); ap.add_argument("--detail-speed", type=float, default=2.0)
    ap.add_argument("--detail-every", type=int, default=50); ap.add_argument("--approach", type=float, default=3.0)
    ap.add_argument("--watch-sphere", type=int, default=-1, help="이 구의 모든 셸 접촉을 매 sync 기록")
    ap.add_argument("--fine-from", type=float, default=-1.0, help="이 sim 시각부터 --fine-sync 로 동기화")
    ap.add_argument("--fine-sync", type=float, default=1e-4)
    a = ap.parse_args()
    import DEME
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    groups = a.groups.split(",")
    shell, G, r_out, H = half_bowl(a.wall_extra, groups, subdiv=a.subdiv, flip=a.flip, flip_part=a.flip_part)
    fv = np.asarray(shell.vertices); L5 = np.array([C[0], 0.0, fv[:, 2].max()])           # 립 = 최하점
    local = trimesh.Trimesh((R_W @ (fv - L5).T).T / 1000.0, shell.faces, process=False)  # 세계 방향, owner 원점 = 립
    face_n = np.asarray(local.face_normals)
    objp = out / f"shell_{a.cell}.obj"; local.export(objp)
    r = 0.00208; rho = 950.0; m_p = rho * 4 / 3 * math.pi * r ** 3
    y_s = -0.0241                                                                   # W3 seed460 과 같은 자리
    # ── 입자 ───────────────────────────────────────────────────────────────
    if a.scenario == "one":
        pos = np.array([[a.xoff / 1000.0, y_s + a.yoff / 1000.0, r]])
        box = np.array([[-0.05, 0.05], [y_s - 0.06, y_s + 0.06], [0.0, 0.25]]); z_surf = 2 * r
    elif a.scenario == "lattice":
        g = np.arange(3) - 1; d = 2 * r * a.lat_spacing
        pos = np.array([[a.xoff / 1000.0 + i * d, y_s + a.yoff / 1000.0 + j * d, r + k * d] for i in g for j in g for k in (0, 1, 2)])   # k ≥ 0: 바닥 아래 층 금지(초기 관입 = 폭발)
        box = np.array([[-0.05, 0.05], [y_s - 0.06, y_s + 0.06], [0.0, 0.25]]); z_surf = 2 * r + 2 * d
    else:
        z = np.load(PILE, allow_pickle=True); P0 = np.asarray(z["positions_m"], float)
        hw = a.cut_half / 1000.0
        sel = (np.abs(P0[:, 0]) < hw) & (np.abs(P0[:, 1] - y_s) < hw + 0.005)
        pos = P0[sel]; wm = r + 0.001                                       # 벽은 모든 중심에서 r 이상 밖에
        box = np.array([[-hw - wm, hw + wm], [y_s - hw - 0.005 - wm, y_s + hw + 0.005 + wm], [0.0, 0.30]])
        near = np.hypot(pos[:, 0], pos[:, 1] - y_s) < 0.022; z_surf = float(pos[near, 2].max()) + r
    n_p = len(pos)
    s = DEME.DEMSolver(); s.SetVerbosity("ERROR")
    mp = {"E": a.E, "nu": 0.3, "CoR": a.cor, "mu": a.mu, "Crr": a.crr}
    mat_p, mat_w, mat_m = s.LoadMaterial(mp), s.LoadMaterial(mp), s.LoadMaterial(dict(mp, E=a.Emesh))
    s.UseFrictionalHertzianModel()
    s.AddClumps(s.LoadSphereType(m_p, r, mat_p), pos.tolist())
    s.InstructBoxDomainDimension(tuple(box[0]), tuple(box[1]), tuple(box[2])); s.InstructBoxDomainBoundingBC("all", mat_w)
    s.SetContactOutputContent(["OWNER", "GEO_ID", "FORCE", "POINT", "NORMAL"])
    z_lip0 = z_surf + a.approach / 1000.0
    z_end = max(z_surf - a.plunge / 1000.0, 0.0015)
    me = s.AddWavefrontMeshObject(str(objp), mat_m, True, False)
    me.SetMass(0.02); me.SetMOI([1e-5] * 3); me.SetFamily(10); me.SetInitPos([0.0, y_s, z_lip0])
    vz = -a.vz / 1000.0
    T_settle = 0.15 if a.scenario == "cut" else 0.02
    s.SetFamilyPrescribedLinVel(10, "0", "0", f"(t < {T_settle}) ? 0.0 : {vz:.9f}", True)
    s.SetFamilyPrescribedAngVel(10, "0", "0", "0", True)
    s.SetInitTimeStep(a.dt); s.SetGravitationalAcceleration([0, 0, -9.81]); s.SetCDUpdateFreq(a.cd); s.SetErrorOutVelocity(a.errvel)
    trk = s.Track(me); s.Initialize(); owner_m = int(trk.GetOwnerID())
    print(f"[{a.cell}] {a.scenario} n={n_p} tri={len(local.faces)} groups={groups} wall={WALL + a.wall_extra:.1f} "
          f"cap={CAP + a.wall_extra:.1f} subdiv={a.subdiv} flip={a.flip} vz={a.vz} dt={a.dt} sync={a.sync} "
          f"z_surf={z_surf * 1000:.2f} lip0={z_lip0 * 1000:.2f} -> {z_end * 1000:.2f} mm", flush=True)

    rows, event, worst = [], None, {"single_N": 0.0, "v_max": 0.0}
    prox = trimesh.proximity.ProximityQuery(local)

    def detail(step_t, lip):
        """이 sync 의 셸 접촉 상세: 최대 힘 3건 + 구 중심 분류."""
        c = s.GetContactDetailedInfo()
        typ = np.asarray(c.GetContactType()); sm = np.where(typ == "SM")[0]
        if not len(sm):
            return [], 0
        A, B = np.asarray(c.GetAOwner()), np.asarray(c.GetBOwner()); geoA, geoB = np.asarray(c.GetAGeo()), np.asarray(c.GetBGeo())
        F, N, Pt = np.asarray(c.GetForce(), float), np.asarray(c.GetNormal(), float), np.asarray(c.GetPoint(), float)
        pp = np.asarray(s.GetOwnerPosition(0, n_p), float); vv = np.asarray(s.GetOwnerVelocity(0, n_p), float)
        items = []
        order = list(sm[np.argsort(-np.linalg.norm(F[sm], axis=1))][:3])   # 최대 힘 3건만 상세 계산
        if a.watch_sphere >= 0:                                              # + 감시 구의 접촉 전부
            order += [k for k in sm if (A[k] == a.watch_sphere or B[k] == a.watch_sphere) and k not in order]
        for k in order:
            sph, tri = (A[k], geoB[k]) if B[k] == owner_m else (B[k], geoA[k])
            if sph >= n_p or tri >= len(face_n):
                continue
            f = np.asarray(F[k]); fm = float(np.linalg.norm(f)); n_deme = np.asarray(N[k]); nf = face_n[tri]
            cen = pp[sph]; rel = cen - lip                                   # 세계, 립 원점
            tri_pts = np.asarray(local.vertices)[local.faces[tri]] + lip
            d_plane = float(np.dot(cen - tri_pts[0], nf))                     # 면 법선 방향 부호거리(+ = 바깥)
            d_close = float(np.linalg.norm(cen - trimesh.triangles.closest_point(tri_pts[None], cen[None])[0]))
            p5 = (R_W.T @ rel) * 1000.0 + L5
            items.append({"sphere": int(sph), "tri": int(tri), "group": str(G[tri]), "F_N": round(fm, 5),
                          "n_deme": [round(float(x), 4) for x in n_deme], "n_face": [round(float(x), 4) for x in nf],
                          "n_dot": round(float(np.dot(n_deme, nf)), 4), "d_plane_mm": round(d_plane * 1000, 4),
                          "d_closest_mm": round(d_close * 1000, 4), "pen_mm": round((r - d_close) * 1000, 4),
                          "v_sphere_m_s": round(float(np.linalg.norm(vv[sph])), 4),
                          "center_l5_mm": [round(float(x), 3) for x in p5], "center_class": classify(p5, r_out, H),
                          "point_rel_lip_mm": [round(float(x) * 1000, 3) for x in (np.asarray(Pt[k]) - lip)]})
        return items, int(len(sm))

    n_max = int(min(60000, (z_lip0 - z_end) / (a.vz / 1000.0) / a.sync + T_settle / a.sync + 200))
    hist = []
    watch = []
    for i in range(n_max):
        fine = a.fine_from >= 0 and s.GetSimTime() >= a.fine_from
        s.DoDynamicsThenSync(a.fine_sync if fine else a.sync)
        lip = np.asarray(trk.Pos(), float)
        vv = np.asarray(s.GetOwnerVelocity(0, n_p), float); sp = np.linalg.norm(vv, axis=1); j = int(sp.argmax())
        pts, frcs = trk.GetContactForces()                                  # 셸 접촉(싼 경로): 매 sync
        fm = np.linalg.norm(np.asarray(frcs, float), axis=1) if len(pts) else np.zeros(0)
        single = float(fm.max()) if len(fm) else 0.0
        # 상세(GetContactDetailedInfo, 비쌈·대규모에서 불안정) 는 트리거 시에만: 힘 스파이크·속도 점프·주기
        trig = single > a.detail_force or sp[j] > a.detail_speed or i % a.detail_every == 0 or fine
        top, n_sm = detail(s.GetSimTime(), lip) if trig else ([], int(len(pts)))
        if fine and a.watch_sphere >= 0:
            w = a.watch_sphere; pw = np.asarray(s.GetOwnerPosition(w, 1), float)[0]
            watch.append({"i": i, "t": round(float(s.GetSimTime()), 6), "pos_rel_lip_mm": [round(float(x) * 1000, 4) for x in (pw - lip)],
                          "v_m_s": round(float(sp[w]), 4), "contacts": [t for t in top if t["sphere"] == w]})
        row = {"i": i, "t": round(float(s.GetSimTime()), 6), "z_lip_mm": round(float(lip[2]) * 1000, 4),
               "insert_mm": round((z_surf - lip[2]) * 1000, 3), "n_SM": n_sm, "single_N": round(single, 5),
               "single_point_rel_lip_mm": ([round(float(x) * 1000, 3) for x in (np.asarray(pts[int(fm.argmax())]) - lip)] if len(fm) else None),
               "v_max": round(float(sp[j]), 4), "v_max_sphere": j, "v_max_pos_rel_lip_mm": [round(float(x) * 1000, 3) for x in (np.asarray(s.GetOwnerPosition(j, 1))[0] - lip)],
               "top": top}
        hist.append(row); hist = hist[-8:]
        worst["single_N"] = max(worst["single_N"], single); worst["v_max"] = max(worst["v_max"], row["v_max"])
        rows.append({k: row[k] for k in ("i", "t", "z_lip_mm", "insert_mm", "n_SM", "single_N", "v_max")}
                    | ({"top1": top[0]} if top else {}))
        if i % 200 == 0 and top:
            print(f"  {i:5d} t={row['t']:.4f} ins={row['insert_mm']:7.3f} SM={n_sm:3d} single={single:8.4f} N "
                  f"v_max={row['v_max']:.3f} top:{top[0]['group']} pen={top[0]['pen_mm']:.3f} ndot={top[0]['n_dot']:.2f} "
                  f"cls={top[0]['center_class']}", flush=True)
        if i % 100 == 0:                                                    # 중간 저장(외부 timeout 에 죽어도 남긴다)
            json.dump({"cell": a.cell, "partial": True, "rows": rows, "history_last8": hist, "watch": watch}, open(out / f"cell_{a.cell}.partial.json", "w"))
        if row["v_max"] > a.vstop:
            event = {"at_row": i, "history_last8": hist}; print(f"  🔴 v_max {row['v_max']:.1f} m/s > {a.vstop} at insert {row['insert_mm']:.2f} mm", flush=True)
            break
        if lip[2] <= z_end + 1e-9 or time.time() - t0 > a.max_wall_s:
            break
    res = {"cell": a.cell, "args": vars(a), "scenario": a.scenario, "n_particles": n_p, "n_tri": int(len(local.faces)),
           "groups": groups, "wall_mm": WALL + a.wall_extra, "cap_mm": CAP + a.wall_extra, "lip_l5_z_mm": float(L5[2]),
           "inputs_sha16": {str(PILE): sha16(PILE), str(REPO / "sim_deme_s1_diverge_min.py"): sha16(__file__)},
           "z_surf_mm": z_surf * 1000, "insert_target_mm": (z_surf - z_end) * 1000,
           "insert_reached_mm": rows[-1]["insert_mm"] if rows else None, "diverged": event is not None,
           "worst": worst, "wall_s": round(time.time() - t0, 1), "event": event, "watch": watch, "rows": rows}
    json.dump(res, open(out / f"cell_{a.cell}.json", "w"), ensure_ascii=False, indent=1)
    print(f"[{a.cell}] diverged={event is not None} insert={res['insert_reached_mm']} / {res['insert_target_mm']:.1f} mm "
          f"single_max={worst['single_N']:.3f} N v_max={worst['v_max']:.2f} m/s wall={res['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
