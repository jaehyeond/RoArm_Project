"""DEME 스쿱 시뮬 — 셸 2매 대칭 **회전** 폐합 + 폐합 반력 측정 (트랙 P1).

무엇을 재는가
    1. 폐합 반력 — 두 형태로 낸다.
       (a) 셸에 작용하는 접촉 합력 |F| (N)
       (b) 🔴 **피벗 축 둘레 모멘트 M_y (N·m) 를 립 반경으로 나눈 등가 립 힘 (N)**
           이쪽이 실측 조 힘 1.8~6.3 N (D451/D452) 과 같은 차원이다. 합력 |F| 는
           관입 항력까지 섞여 있어 조 힘과 직접 비교할 수 없다.
    2. 1회 스쿱에 담긴 입자 수·질량
    3. 퍼낸 뒤 heightmap (roarm_rl.heightmap, 계약 roarm-heightmap-v1)
    4. 1회 스쿱 벽시계 -> 3,000 시행 예산

🔴 D464 §4 의 "조합 제약" 은 오진이었다 (본 파일이 반증한다)
    `s.Track(mesh)` 를 **`Initialize()` 뒤에** 부르면 트래커가 owner 에 묶이지 않고
    `GetOwnerID()` 가 `4294967295`(UINT_MAX = 미할당) 를 돌려주며, 다음 스텝에서
    세그폴트한다. 메시 매수나 구동 방식과는 **무관하다** — 메시 1매도 똑같이 죽는다.
    `Track` 을 `Initialize` **앞으로** 옮기면
        메시 2매 + Track            OK
        규정 선속도 + Track          OK
        규정 각속도(회전) + Track     OK
    가 전부 성립한다. 작동하던 `sim_deme_mesh_min_example.py` 가 우연히
    `Track` 을 Initialize 앞에서 부르고 있었을 뿐이다.

폐합은 근사가 아니라 진짜 회전이다
    셸 메시를 **피벗이 로컬 원점에 오도록** 옮겨 굽고, `SetFamilyPrescribedAngVel`
    로 피벗 둘레를 돌린다. 평행이동 근사(구버전)는 립 궤적이 틀렸다.

좌표 프레임 (여기서 한 번만 변환한다)
    설계 프레임(`scoop_grab_v1_design.py`) = x,y 평면에서 `rotz`, z = 셸 너비,
    깊이 방향은 **-y**. DEME 월드는 중력이 -z 다. 그래서
        world = (lx, -lz, ly)          # 설계 -y(깊이) -> 월드 -z(아래)
    를 적용하면 설계의 `rotz` 는 **월드 Y축 회전**이 된다.
    좌셸은 +phi, 우셸은 -phi 로 벌어진다 (phi=44.5° 에서 입 58.005 mm — 설계 게이트값).

⚠️ 물성은 실측 전이다. E 는 수치 안정용 임시값이며 펠릿 실측이 아니다.
   실측하면 반력이 바뀐다. 이 파일의 반력을 최종값으로 인용하지 마라.
"""
import sys, os, json, math, time, hashlib
from pathlib import Path
import numpy as np
import trimesh
import DEME

REPO = Path(__file__).resolve().parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else REPO / "claudedocs/runtime_logs/scoop_track/s2_closure_rot")

PILE = REPO / "claudedocs/runtime_logs/sim_deme/pile_practical_targetridge_d4p16_n18796_seed460.npz"
SHELL = {-1: REPO / "claudedocs/runtime_logs/scoop_grab_v1/shell_L_ALL.stl",
         +1: REPO / "claudedocs/runtime_logs/scoop_grab_v1/shell_R_ALL.stl"}

P = {
    # 기구 (트랙 A 확정: D462/D463/D464)
    "pivot_gap_mm":      26.0,
    "shell_travel_deg":  44.5,     # 링크가 서보 89도에서 내는 셸 편측 회전
    "lip_depth_mm":      36.06,
    "lip_pivot_radius_mm": 38.332,  # design.json derived. 모멘트 -> 등가 립 힘 환산에 쓴다
    "shell_mass_kg":     0.0263,
    # 궤적
    "approach_gap_mm":   10.0,     # 하강 시작 시 립이 더미 표면 위로 띄우는 높이
    "insert_depth_mm":   18.0,     # 더미 표면 아래로 넣는 깊이
    "close_end_deg":      0.0,     # 폐합 종료각. 0 = 립이 완전히 맞닿을 때까지
    "descend_steps":     70,
    "close_steps":      100,
    "lift_steps":       100,
    "lift_height_mm":    40.0,
    "settle_steps":       5,
    "dt_sync_s":        0.004,     # 스텝당 물리 시간 (0.02 는 죽는다 — D464 §4)
    # 물리 (⚠️ 임시값)
    "timestep_s":       1.0e-5,
    "E_pa":             5.0e6,     # 수치 안정용. 펠릿 실측 아님
    "nu":               0.30,
    "CoR":              0.30,
    "mu":               0.50,
    "Crr":              0.05,
    "particle_density_kg_m3": 950.0,
    "error_out_vel":    20.0,
    "cd_update_freq":   20,
    "domain_top_m":     0.20,      # 더미 상자(0.08)보다 높여 그랩이 들어갈 자리를 만든다
    # 판정 기준
    "jaw_force_band_N": [1.8, 6.3],   # D451/D452 실측
    "seed":             460,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def roty(deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def bake_shell(side, phi_deg, tmpdir):
    """셸 STL -> 월드 프레임 · 피벗이 로컬 원점 · phi 만큼 벌린 자세의 OBJ.

    반환 OBJ 의 로컬 원점 = 피벗축 위의 점이므로, 솔버가 owner 를 회전시키면
    그대로 **피벗 둘레 회전**이 된다.
    """
    m = trimesh.load(SHELL[side])
    V = np.asarray(m.vertices, float)                        # mm, 설계 프레임
    V = V - np.array([side * P["pivot_gap_mm"] / 2.0, 0.0, 0.0])
    W = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)            # -> 월드 프레임
    W = W @ roty(-side * phi_deg).T                          # 좌 +phi / 우 -phi 로 벌림
    mm = trimesh.Trimesh(vertices=W / 1000.0, faces=m.faces, process=False)
    p = Path(tmpdir) / f"shell_{'L' if side < 0 else 'R'}_open{phi_deg:.2f}.obj"
    mm.export(p)
    return str(p), mm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_obj"
    tmp.mkdir(exist_ok=True)
    smoke = int(os.environ.get("SCOOP_SMOKE", "0"))
    rep_tag = os.environ.get("SCOOP_REP", "rep1")
    # 조정 가능한 두 축. 기본값은 P 안에 있고, 스윕할 때만 환경변수로 덮는다.
    P["close_end_deg"] = float(os.environ.get("SCOOP_CLOSE_END_DEG", P["close_end_deg"]))
    P["insert_depth_mm"] = float(os.environ.get("SCOOP_INSERT_MM", P["insert_depth_mm"]))
    P["error_out_vel"] = float(os.environ.get("SCOOP_ERRVEL", P["error_out_vel"]))
    t_start = time.time()

    z = np.load(PILE, allow_pickle=True)
    pos = np.asarray(z["positions_m"], float)
    rad = float(np.asarray(z["radii_m"], float)[0])
    box = np.asarray(z["box_bounds_m"], float)
    z_top = float(pos[:, 2].max())
    print(f"더미 {len(pos)} 입자 · r={rad*1000:.2f} mm · 최상단 z={z_top*1000:.1f} mm",
          flush=True)

    s = DEME.DEMSolver()
    s.SetVerbosity("ERROR")
    mp = {"E": P["E_pa"], "nu": P["nu"], "CoR": P["CoR"], "mu": P["mu"], "Crr": P["Crr"]}
    mat_p = s.LoadMaterial(mp)
    mat_w = s.LoadMaterial(mp)
    mat_m = s.LoadMaterial(mp)
    s.UseFrictionalHertzianModel()

    mass_p = P["particle_density_kg_m3"] * 4.0 / 3.0 * math.pi * rad ** 3
    s.AddClumps(s.LoadSphereType(mass_p, rad, mat_p), pos.tolist())

    # 벽은 더미를 정착시킬 때와 **같은 x/y** 로 둔다 (위치가 달라지면 더미가 튄다).
    # 천장만 그랩이 들어올 만큼 올린다.
    s.InstructBoxDomainDimension((float(box[0, 0]), float(box[0, 1])),
                                 (float(box[1, 0]), float(box[1, 1])),
                                 (float(box[2, 0]), P["domain_top_m"]))
    s.InstructBoxDomainBoundingBC("top_open", mat_w)

    # ── 셸 2매를 벌린 자세로 굽고 피벗 높이를 정한다 ───────────────────────────
    phi = P["shell_travel_deg"]
    meshes, low = {}, {}
    for side in (-1, +1):
        objp, mm = bake_shell(side, phi, tmp)
        low[side] = float(np.asarray(mm.vertices)[:, 2].min())   # 벌린 자세 최하점(로컬)
        me = s.AddWavefrontMeshObject(objp, mat_m, True, False)
        me.SetMass(P["shell_mass_kg"])
        me.SetMOI([2e-5, 2e-5, 2e-5])
        me.SetFamily(10 if side < 0 else 11)
        meshes[side] = me
        print(f"  셸 {'L' if side<0 else 'R'} 삼각형 {me.GetNumTriangles()} "
              f"· 벌린 자세 최하점(로컬) {low[side]*1000:.2f} mm", flush=True)

    lo = min(low.values())
    z_piv0 = z_top + P["approach_gap_mm"] / 1000.0 - lo
    z_piv1 = z_top - P["insert_depth_mm"] / 1000.0 - lo
    for side in (-1, +1):
        # ⚠️ SetInitPos 는 Initialize **전에** 필수 (D464 §4). 안 주면 원점에서 시작한다.
        meshes[side].SetInitPos([side * P["pivot_gap_mm"] / 2000.0, 0.0, z_piv0])

    # ── 규정 궤적을 시간 수식으로 (Initialize 전에 확정해야 한다) ──────────────
    dt = P["dt_sync_s"]
    phi_end = P["close_end_deg"]
    n_desc = P["descend_steps"] if not smoke else 20
    n_close = P["close_steps"] if not smoke else 20
    n_lift = P["lift_steps"] if not smoke else 10
    T0 = P["settle_steps"] * dt
    T1 = T0 + n_desc * dt
    T2 = T1 + n_close * dt
    T3 = T2 + n_lift * dt
    vz_desc = (z_piv1 - z_piv0) / (n_desc * dt)
    w_close = math.radians(phi - phi_end) / (n_close * dt)
    vz_lift = (P["lift_height_mm"] / 1000.0) / (n_lift * dt)
    TRAJ = {"T_settle_s": T0, "T_descend_s": T1, "T_close_s": T2, "T_lift_s": T3,
            "z_pivot_start_m": z_piv0, "z_pivot_insert_m": z_piv1,
            "vz_descend_m_s": vz_desc, "omega_close_rad_s": w_close,
            "vz_lift_m_s": vz_lift, "phi_open_deg": phi}

    vz_expr = (f"(t < {T0:.6f}) ? 0.0 : ((t < {T1:.6f}) ? {vz_desc:.9f} : "
               f"((t < {T2:.6f}) ? 0.0 : ((t < {T3:.6f}) ? {vz_lift:.9f} : 0.0)))")
    for side in (-1, +1):
        fam = 10 if side < 0 else 11
        # 좌셸은 +phi 에서 0 으로 (wy 음수), 우셸은 -phi 에서 0 으로 (wy 양수)
        wy = side * w_close
        s.SetFamilyPrescribedLinVel(fam, "0", "0", vz_expr, True)
        s.SetFamilyPrescribedAngVel(
            fam, "0",
            f"((t >= {T1:.6f}) && (t < {T2:.6f})) ? {wy:.9f} : 0.0", "0", True)

    s.SetInitTimeStep(P["timestep_s"])
    s.SetGravitationalAcceleration([0, 0, -9.81])
    s.SetCDUpdateFreq(P["cd_update_freq"])
    s.SetErrorOutVelocity(P["error_out_vel"])

    # 🔴 Track 은 반드시 Initialize **전에**. 뒤에서 부르면 owner 미할당 -> 세그폴트.
    trks = {side: s.Track(meshes[side]) for side in (-1, +1)}
    print("  Initialize...", flush=True)
    s.Initialize()
    owner_ids = {side: int(trks[side].GetOwnerID()) for side in (-1, +1)}
    print(f"  Initialize OK · owner IDs {owner_ids}", flush=True)
    if any(v >= 2 ** 32 - 1 for v in owner_ids.values()):
        raise RuntimeError("트래커가 owner 에 묶이지 않았다 — Track 이 Initialize 뒤로 갔는지 확인")

    n_p = len(pos)
    log = []
    R_lip = P["lip_pivot_radius_mm"] / 1000.0
    # 재생용 기록 (D341 RRD 는 rerun 0.34.1 이 있는 다른 env 에서 후처리로 굽는다.
    # 여기서는 그 후처리가 쓸 원자료 — 실제 판정 대상인 셸 노드와 접촉점 — 만 모은다.)
    PHASE_CODE = {"descend": 0, "close": 1, "lift": 2}
    rec = {"nodes_L": [], "nodes_R": [], "t": [], "phase": [],
           "cp": [], "cf": [], "cside": [], "cframe": []}

    def sample(phase, i, extra=None):
        """셸별 접촉 수 · 합력 · 피벗 둘레 모멘트를 한 줄로 기록한다."""
        row = {"phase": phase, "i": i, "sim_t": round(float(s.GetSimTime()), 6)}
        nc_tot, F_tot, M_tot, f1_max, f1_at = 0, 0.0, 0.0, 0.0, None
        for side in (-1, +1):
            tag = "L" if side < 0 else "R"
            piv = np.asarray(trks[side].Pos(), float)          # owner 원점 = 피벗축 위의 점
            pts, frcs = trks[side].GetContactForces()
            if len(pts):
                Pp = np.asarray(pts, float)
                Ff = np.asarray(frcs, float)
                Fsum = Ff.sum(0)
                # 피벗 둘레 모멘트. 폐합 저항은 회전축(월드 Y) 성분이다.
                M = np.cross(Pp - piv, Ff).sum(0)
                My = float(M[1])
                Fmag = float(np.linalg.norm(Fsum))
                # 단일 접촉 최대 — 이게 합력을 지배하면 "핀치 1개(수치)",
                # 넓게 퍼져 있으면 "다짐(물리)" 이다. 발산 원인 판별용.
                mags = np.linalg.norm(Ff, axis=1)
                j = int(mags.argmax())
                if float(mags[j]) > f1_max:
                    f1_max = float(mags[j])
                    f1_at = [round(float(v) * 1000, 2) for v in Pp[j]]
            else:
                My, Fmag = 0.0, 0.0
            row[f"n_{tag}"] = len(pts)
            row[f"F_{tag}_N"] = round(Fmag, 4)
            row[f"My_{tag}_Nm"] = round(My, 6)
            row[f"lipF_{tag}_N"] = round(abs(My) / R_lip, 4)
            nc_tot += len(pts)
            F_tot += Fmag
            M_tot += abs(My)
        row["n_total"] = nc_tot
        row["F_total_N"] = round(F_tot, 4)
        row["lipF_total_N"] = round(M_tot / R_lip, 4)
        row["max_single_contact_N"] = round(f1_max, 4)
        row["max_single_contact_at_mm"] = f1_at
        fi = len(rec["t"])
        rec["t"].append(row["sim_t"])
        rec["phase"].append(PHASE_CODE[phase])
        for side, key in ((-1, "nodes_L"), (+1, "nodes_R")):
            rec[key].append(np.asarray(trks[side].GetMeshNodesGlobal(), np.float32))
            pts, frcs = trks[side].GetContactForces()
            if len(pts):
                rec["cp"].append(np.asarray(pts, np.float32))
                rec["cf"].append(np.asarray(frcs, np.float32))
                rec["cside"].append(np.full(len(pts), side, np.int8))
                rec["cframe"].append(np.full(len(pts), fi, np.int32))
        if extra:
            row.update(extra)
        log.append(row)
        return row

    for _ in range(P["settle_steps"]):
        s.DoDynamicsThenSync(dt)
    print(f"피벗 z {z_piv0*1000:.1f} -> {z_piv1*1000:.1f} mm "
          f"(vz={vz_desc:.3f} m/s, w={w_close:.3f} rad/s)", flush=True)

    # ⚠️ 발산해도 여기까지 모은 타임라인은 반드시 남긴다 (매 스텝 저장하면 느리므로
    #    파이썬 예외로 잡히는 경우만 처리한다. DEME 의 발산은 C++ abort 라 파이썬으로
    #    올라오지 않으므로, 타임라인을 20 스텝마다 디스크에 흘려 둔다.)
    tl_path = OUT / f"scoop_timeline_{rep_tag}.json"

    def flush_timeline(state):
        json.dump({"state": state, "rows": log}, open(tl_path, "w"), ensure_ascii=False)

    close0 = lift0 = None
    diverged = False
    try:
        for i in range(n_desc):
            s.DoDynamicsThenSync(dt)
            r = sample("descend", i,
                       {"z_pivot_mm": round(float(trks[-1].Pos()[2]) * 1000, 2)})
            if i % 20 == 0:
                flush_timeline("descend")
            if i % 10 == 0:
                print(f"  하강 {i:3d}  z={r['z_pivot_mm']:7.2f}  접촉={r['n_total']:4d}  "
                      f"F={r['F_total_N']:8.3f} N  립등가={r['lipF_total_N']:7.3f} N  "
                      f"단일최대={r['max_single_contact_N']:8.3f} N", flush=True)

        close0 = len(log)
        for i in range(n_close):
            s.DoDynamicsThenSync(dt)
            ph = phi + (phi_end - phi) * (i + 1) / n_close
            r = sample("close", i, {"phi_deg": round(ph, 2)})
            flush_timeline("close")
            if i % 10 == 0:
                print(f"  폐합 {i:3d}  phi={ph:5.1f}  접촉={r['n_total']:4d}  "
                      f"F={r['F_total_N']:8.3f} N  립등가={r['lipF_total_N']:7.3f} N  "
                      f"단일최대={r['max_single_contact_N']:8.3f} N", flush=True)

        lift0 = len(log)
        for i in range(n_lift):
            s.DoDynamicsThenSync(dt)
            sample("lift", i, {"z_pivot_mm": round(float(trks[-1].Pos()[2]) * 1000, 2)})
            if i % 20 == 0:
                flush_timeline("lift")
    except Exception as exc:                       # noqa: BLE001 — 발산도 산출은 남긴다
        diverged = True
        print(f"🔴 발산/중단: {exc}", flush=True)
        flush_timeline("diverged")

    # 🔴 두 번째 인자는 **개수**지 끝 인덱스가 아니다. `n_p - 1` 로 부르면 마지막 입자
    #    하나가 조용히 빠진다 (sim_deme_mesh_min_example.py 와 구 sim_deme_scoop.py 가
    #    둘 다 이 실수를 하고 있었다). 메시 owner 는 18797~ 이라 n_p 까지가 입자다.
    pp = np.asarray(s.GetOwnerPosition(0, n_p), float)
    if len(pp) != n_p:
        raise RuntimeError(f"입자 위치 개수 불일치: {len(pp)} != {n_p}")
    captured = int((pp[:, 2] > z_top + 0.005).sum())
    wall = time.time() - t_start
    if close0 is None:
        close0 = len(log)
    if lift0 is None:
        lift0 = len(log)

    # ── heightmap (계약 roarm-heightmap-v1) ──────────────────────────────────
    from roarm_rl.heightmap import GridSpec, heightmap_from_particles
    cell = 0.005
    spec = GridSpec(origin_xy_m=(float(box[0, 0]), float(box[1, 0])), cell_m=cell,
                    shape=(int(math.ceil((box[1, 1] - box[1, 0]) / cell)),
                           int(math.ceil((box[0, 1] - box[0, 0]) / cell))),
                    frame="deme_box_floor_center", z_datum_m=0.0)
    hm = heightmap_from_particles(pp, np.full(n_p, rad), spec).height

    desc = [r for r in log if r["phase"] == "descend"] or [{"F_total_N": 0.0}]
    clos = log[close0:lift0] or [{"F_total_N": 0.0, "lipF_total_N": 0.0, "n_total": 0,
                                  "max_single_contact_N": 0.0}]
    band = P["jaw_force_band_N"]
    F_desc = max(r["F_total_N"] for r in desc)
    F_close = max(r["F_total_N"] for r in clos)
    lip_close = max(r["lipF_total_N"] for r in clos)
    n_close_max = max(r["n_total"] for r in clos)
    n_close_first = clos[0]["n_total"]
    single_max = max(r.get("max_single_contact_N", 0.0) for r in clos)

    res = {
        "artifact": "DEME_SCOOP_CLOSURE_V2_ROTATION",
        "rep": rep_tag,
        "smoke": bool(smoke),
        "params": P,
        "trajectory": TRAJ,
        "steps": {"descend": n_desc, "close": n_close, "lift": n_lift},
        "engine": {"DEME": "2.4.0", "force_model": "UseFrictionalHertzianModel"},
        "pile": {"n": n_p, "radius_m": rad, "top_z_m": z_top,
                 "source": str(PILE), "sha256": sha256(PILE)},
        "shells": {"L_sha256": sha256(SHELL[-1]), "R_sha256": sha256(SHELL[+1]),
                   "owner_ids": owner_ids},
        "contacts": {"close_first": n_close_first, "close_max": n_close_max,
                     "monotone_increase_first_half": bool(
                         max(r["n_total"] for r in clos[:len(clos)//2])
                         <= max(r["n_total"] for r in clos[len(clos)//2:]))},
        "diverged": diverged,
        "steps_completed": {"descend": len(desc) if log else 0,
                            "close": lift0 - close0, "lift": len(log) - lift0},
        "forces_N": {"descend_peak_total": round(F_desc, 3),
                     "close_peak_total": round(F_close, 3),
                     "close_peak_lip_equiv": round(lip_close, 3),
                     "close_peak_single_contact": round(single_max, 3),
                     "jaw_band_measured": band},
        "verdict_closure": ("WITHIN_MEASURED_JAW_FORCE" if lip_close <= band[1]
                            else "EXCEEDS_MEASURED_JAW_FORCE"),
        "captured_particles": captured,
        "captured_mass_g": round(captured * mass_p * 1000, 3),
        "wall_seconds": round(wall, 2),
        "budget_3000_hours": round(wall * 3000 / 3600.0, 2),
        "non_claims": [
            "강성 E=5e6 Pa 는 수치 안정용 임시값이며 펠릿 실측이 아니다. "
            "밀도 950 kg/m3 · mu 0.50 · CoR 0.30 · Crr 0.05 도 전부 미실측 임시값이다. "
            "**따라서 이 파일의 반력 수치는 어떤 판정에도 인용할 수 없다.** "
            "PP 펠릿 실측(밀도·안식각)이 들어오면 숫자가 통째로 바뀐다.",
            "이 실행이 주장하는 것은 오직 **경로가 성립한다**는 것이다: "
            "셸 2매가 진짜로 피벗 둘레를 회전해 닫히고, 그 동안 접촉 수와 반력이 "
            "0 이 아닌 값으로 읽힌다.",
            "더미 npz 는 E=1.0e7 Pa · dt=2.0e-5 로 정착시킨 것이고 본 실행은 "
            "E=5.0e6 Pa · dt=1.0e-5 다. 강성이 다르므로 t=0 직후 더미가 미세하게 "
            "다시 앉는다. 실측 물성을 넣을 때 더미도 같은 물성으로 재정착시켜야 한다.",
            "실측 조 힘 1.8~6.3 N (D451/D452) 은 **슬리브 그리퍼(D452)** 값이다. "
            "그랩 v1 은 다른 기구이므로 이 대역은 참고선이지 합격선이 아니다.",
            "D341 Rerun 완결 계약(rrd verify · 블루프린트 .rbl · 헤드리스 스크린샷 · "
            "육안 검수)은 본 실행에서 **미이행**이다. RRD 는 기록하되 검증 게이트는 "
            "돌리지 않았다.",
        ],
    }
    def cat(key, dtype):
        return (np.concatenate(rec[key]) if rec[key]
                else np.zeros((0, 3) if dtype == np.float32 else (0,), dtype))

    np.savez_compressed(
        OUT / f"scoop_{rep_tag}.npz", positions_m=pp,
        radii_m=np.full(n_p, rad), heightmap_m=hm, box_bounds_m=box,
        # 재생용 원자료 (판정 대상 = 셸 노드 궤적 + 접촉점/접촉력)
        frame_t_s=np.asarray(rec["t"], np.float64),
        frame_phase=np.asarray(rec["phase"], np.int8),
        nodes_L_m=np.asarray(rec["nodes_L"], np.float32),
        nodes_R_m=np.asarray(rec["nodes_R"], np.float32),
        contact_point_m=cat("cp", np.float32), contact_force_N=cat("cf", np.float32),
        contact_side=cat("cside", np.int8), contact_frame=cat("cframe", np.int32))
    json.dump(res, open(OUT / f"scoop_closure_{rep_tag}.json", "w"),
              ensure_ascii=False, indent=2)
    flush_timeline("diverged" if diverged else "complete")

    print()
    print(f"하강 최대 합력    {F_desc:8.3f} N")
    print(f"폐합 최대 합력    {F_close:8.3f} N")
    print(f"폐합 최대 립등가  {lip_close:8.3f} N   <- 조 힘과 같은 차원")
    print(f"폐합 단일접촉최대 {single_max:8.3f} N   <- 합력을 지배하면 핀치 1개")
    print(f"폐합 접촉 수      {n_close_first} -> 최대 {n_close_max}")
    print(f"실측 조 힘        {band[0]}~{band[1]} N  ->  {res['verdict_closure']}")
    print(f"담긴 입자         {captured} 개 = {res['captured_mass_g']} g")
    print(f"벽시계            {wall:.1f} s  ->  3,000 시행 {res['budget_3000_hours']} 시간")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
