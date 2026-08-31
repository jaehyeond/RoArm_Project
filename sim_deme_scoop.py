"""DEME 스쿱 시뮬 — 셸 2매 대칭 폐합 규정 궤적 (트랙 B, 코디네이터 직접 수행).

무엇을 재는가
    1. 폐합 반력 (N)  <- 이 트랙의 핵심. 실측 조 힘 1.8~6.3 N (D451/D452) 과 비교한다.
       이보다 크면 실물에서 닫히지 않는다는 뜻이다.
    2. 1회 스쿱에 담긴 입자 수·질량
    3. 퍼낸 뒤 heightmap (roarm_rl.heightmap, 계약 roarm-heightmap-v1)
    4. 1회 스쿱 벽시계 -> 3,000 시행 예산

왜 코디네이터가 직접 하는가
    트랙 B 워커가 네트워크 차단 환경에서 외부 문서를 3시간 넘게 조회하다 산출 0 으로 종료됐다.
    코디네이터가 로컬 바인딩에서 API 를 직접 확정했다 (sim_deme_mesh_min_example.py 참조).

규정 궤적 (scoop_v0_fixed_path_v1). 학습 대상이 아니다.
    (a) 개방 하강   셸 2매를 44.5도 벌린 채 더미 위에서 관입 깊이까지 내린다
    (b) 대칭 폐합   두 셸을 44.5 -> 0 도로 서로를 향해 회전 (트랙 A 링크가 이 각을 낸다)
    (c) 리프트      닫힌 채로 들어올린다

⚠️ 물성은 실측 전이다. 강성 E 는 수치 안정용 임시값이며 펠릿 실측이 아니다.
   실측하면 반력이 바뀐다. 이 파일의 반력을 최종값으로 인용하지 마라.
"""
import sys, json, math, time
from pathlib import Path
import numpy as np
import trimesh
import DEME

REPO = Path(__file__).resolve().parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else REPO / "claudedocs/runtime_logs/scoop_track/s1_closure")

PILE = REPO / "claudedocs/runtime_logs/sim_deme/pile_practical_targetridge_d4p16_n18796_seed460.npz"
SHELL_L = REPO / "claudedocs/runtime_logs/scoop_grab_v1/shell_L_ALL.stl"
SHELL_R = REPO / "claudedocs/runtime_logs/scoop_grab_v1/shell_R_ALL.stl"

P = {
    # 기구 (트랙 A 확정: D462/D463)
    "pivot_gap_mm":      26.0,
    "shell_travel_deg":  44.5,     # 링크가 서보 89도에서 내는 셸 편측 회전
    "lip_depth_mm":      36.06,
    # 궤적
    "insert_depth_mm":   18.0,     # 더미 표면 아래로 넣는 깊이
    "descend_steps":     90,
    "close_steps":      120,
    "lift_steps":        40,
    "dt_sync_s":        0.004,     # 스텝당 물리 시간 (예제에서 안정 확인)
    # 물리 (⚠️ 임시값)
    "timestep_s":       1.0e-5,
    "E_pa":             5.0e6,     # 수치 안정용. 펠릿 실측 아님
    "nu":               0.30,
    "CoR":              0.30,
    "mu":               0.50,
    "Crr":              0.05,
    "error_out_vel":    20.0,
    "cd_update_freq":   20,
    # 판정 기준
    "jaw_force_band_N": [1.8, 6.3],   # D451/D452 실측
    "seed":             460,
}



def load_mesh_m(path):
    m = trimesh.load(path)
    if m.extents.max() > 1.0:
        m.apply_scale(0.001)
    return m


def rotz(deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def shell_obj(src, side, angle_deg, pivot_xy_m, tmpdir):
    """셸을 자기 피벗 기준으로 angle 만큼 회전시켜 OBJ 로 굽는다.

    DEME 메시는 SetPos 로 평행이동만 시킨다(회전 API 는 프레임 단위라 다루기 번거롭다).
    대신 **각 폐합 스텝마다 회전된 형상을 새로 굽지 않고**, 회전은 궤적 생성 시
    피벗 둘레 원운동 + 사전 회전으로 표현한다. 여기서는 사전 회전본을 만든다.
    """
    m = load_mesh_m(src)
    px, py = pivot_xy_m
    V = np.asarray(m.vertices, float)
    V[:, :2] -= [px, py]
    V = V @ rotz(side * angle_deg).T
    V[:, :2] += [px, py]
    m2 = trimesh.Trimesh(vertices=V, faces=m.faces, process=False)
    p = Path(tmpdir) / f"shell_{'L' if side < 0 else 'R'}_{angle_deg:+07.2f}.obj"
    m2.export(p)
    return str(p), m2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_obj"
    tmp.mkdir(exist_ok=True)
    t_start = time.time()

    z = np.load(PILE, allow_pickle=True)
    pos = np.asarray(z["positions_m"], float)
    rad = float(np.asarray(z["radii_m"], float)[0])
    box = np.asarray(z["box_bounds_m"], float)
    print(f"더미 {len(pos)} 입자 · r={rad*1000:.2f} mm · 최상단 z={pos[:,2].max()*1000:.1f} mm",
          flush=True)

    s = DEME.DEMSolver()
    s.SetVerbosity("ERROR")
    mat_p = s.LoadMaterial({"E": P["E_pa"], "nu": P["nu"], "CoR": P["CoR"],
                            "mu": P["mu"], "Crr": P["Crr"]})
    mat_m = s.LoadMaterial({"E": P["E_pa"], "nu": P["nu"], "CoR": P["CoR"],
                            "mu": P["mu"], "Crr": P["Crr"]})

    rho = 950.0
    mass = rho * 4.0 / 3.0 * math.pi * rad ** 3
    tmpl = s.LoadSphereType(mass, rad, mat_p)
    sub = int(__import__("os").environ.get("SCOOP_NSUB", "0"))
    if sub and sub < len(pos):          # 디버그용 부분집합. 그랩 주변만 남긴다
        keep = np.argsort(np.abs(pos[:, 1]))[:sub]
        pos = pos[keep]
        print(f"  [디버그] 입자 {sub} 개로 축소", flush=True)
    s.AddClumps(tmpl, pos.tolist())

    s.AddBCPlane([0, 0, box[2, 0]], [0, 0, 1], mat_p)
    for ax, sgn in ((0, -1), (0, +1), (1, -1), (1, +1)):
        n = [0, 0, 0]; n[ax] = -sgn
        p0 = [0, 0, 0]; p0[ax] = box[ax, 0 if sgn < 0 else 1]
        s.AddBCPlane(p0, n, mat_p)

    # 셸 2매를 **열린 상태**로 굽는다. 폐합은 형상을 다시 구워 교체하지 않고
    # 피벗 둘레 원운동으로 근사한다 (각 스텝의 평행이동량 = 피벗 기준 호의 현).
    g = P["pivot_gap_mm"] / 1000.0
    piv = {-1: (-g / 2, 0.0), +1: (+g / 2, 0.0)}
    z_top = float(pos[:, 2].max())
    # ⚠️ SetInitPos 를 반드시 Initialize **전에** 줘야 한다. 안 주면 메시가 원점(0,0,0)에
    #    놓인 채 초기화되어 더미 속에 박히고, 첫 스텝에서 코어 덤프한다 (실측).
    _tmpm = load_mesh_m(SHELL_L)
    zmin_local = float(np.asarray(_tmpm.vertices)[:, 2].min())
    z0 = z_top + 0.010 - zmin_local
    z1 = z_top - P["insert_depth_mm"] / 1000.0 - zmin_local
    meshes, trks = {}, {}
    for side in (-1, +1):
        src = SHELL_L if side < 0 else SHELL_R
        objp, m2 = shell_obj(src, side, P["shell_travel_deg"], piv[side], tmp)
        mesh = s.AddWavefrontMeshObject(objp, mat_m, True, False)
        mesh.SetInitPos([0, 0, z0])
        mesh.SetMass(0.0263)
        mesh.SetMOI([2e-5, 2e-5, 2e-5])
        fam = 10 if side < 0 else 11
        mesh.SetFamily(fam)
        # ⚠️ 규정 속도는 **Initialize 전에만** 설정 가능하다 (실측: assertSysNotInit).
        #    그래서 궤적 전체를 시뮬 시간 t 의 수식으로 한 번에 준다. 아래 _vel_expr 참조.
        #    (SetPos 로 옮기면 메시 2매일 때 코어 덤프한다 — iso.py STAGE 5 실측)
        pass  # 속도는 아래에서 일괄 설정
        meshes[side] = (mesh, m2)
        print(f"  셸 {'L' if side<0 else 'R'} 삼각형 {mesh.GetNumTriangles()}", flush=True)

    # ── 규정 궤적을 시간 수식으로 (Initialize 전에 확정해야 한다) ──────────
    dt = P["dt_sync_s"]
    T_settle = 5 * dt
    T_desc = T_settle + P["descend_steps"] * dt
    T_close = T_desc + P["close_steps"] * dt
    T_lift = T_close + P["lift_steps"] * dt
    vz_desc = (z1 - z0) / (P["descend_steps"] * dt)
    R_lip = math.hypot(P["pivot_gap_mm"] / 2, P["lip_depth_mm"]) / 1000.0
    a0 = math.atan2(-P["lip_depth_mm"], P["pivot_gap_mm"] / 2)
    dx_close = abs(R_lip * (math.cos(a0 - math.radians(P["shell_travel_deg"]))
                            - math.cos(a0)))
    vx_close = dx_close / (P["close_steps"] * dt)
    vz_lift = 0.040 / (P["lift_steps"] * dt)
    TRAJ = {"T_settle": T_settle, "T_desc": T_desc, "T_close": T_close,
            "T_lift": T_lift, "vz_desc": vz_desc, "dx_close_m": dx_close,
            "vx_close": vx_close, "vz_lift": vz_lift}

    def _vz():
        return (f"(t < {T_settle:.6f}) ? 0.0 : "
                f"((t < {T_desc:.6f}) ? {vz_desc:.9f} : "
                f"((t < {T_close:.6f}) ? 0.0 : "
                f"((t < {T_lift:.6f}) ? {vz_lift:.9f} : 0.0)))")

    def _vx(side):
        v = -side * vx_close        # 좌(-1)는 +x, 우(+1)는 -x 로 모인다
        return (f"((t >= {T_desc:.6f}) && (t < {T_close:.6f})) ? {v:.9f} : 0.0")

    for side in (-1, +1):
        s.SetFamilyPrescribedLinVel(10 if side < 0 else 11, _vx(side), "0", _vz(), True)

    s.SetInitTimeStep(P["timestep_s"])
    s.SetGravitationalAcceleration([0, 0, -9.81])
    s.SetCDUpdateFreq(P["cd_update_freq"])
    s.SetErrorOutVelocity(P["error_out_vel"])
    print("  Initialize...", flush=True)
    s.Initialize()
    print("  Initialize OK", flush=True)
    for side in (-1, +1):
        trks[side] = s.Track(meshes[side][0])

    # ⚠️ 한 번에 0.02 s 를 돌리면 코어 덤프한다 (실측). 0.004 s 씩 쪼갠다.
    for _ in range(5):
        s.DoDynamicsThenSync(0.004)
    print(f"시작 z={z0*1000:.1f} mm -> 관입 목표 z={z1*1000:.1f} mm", flush=True)

    log = []

    def sample(phase, i, extra=None):
        row = {"phase": phase, "i": i, "sim_t": float(s.GetSimTime())}
        tot = 0.0
        for side in (-1, +1):
            pts, frcs = trks[side].GetContactForces()
            F = np.asarray(frcs, float) if len(frcs) else np.zeros((0, 3))
            f = float(np.linalg.norm(F.sum(0))) if len(F) else 0.0
            row[f"n_{'L' if side<0 else 'R'}"] = len(pts)
            row[f"F_{'L' if side<0 else 'R'}_N"] = round(f, 4)
            tot += f
        row["F_total_N"] = round(tot, 4)
        if extra:
            row.update(extra)
        log.append(row)
        return tot

    # (a) 개방 하강 — 속도는 이미 수식으로 설정됨
    for i in range(P["descend_steps"]):
        zz = z0 + (z1 - z0) * (i + 1) / P["descend_steps"]
        s.DoDynamicsThenSync(dt)
        f = sample("descend", i, {"z_mm": round(zz * 1000, 2)})
        if i % 15 == 0:
            print(f"  하강 {i:3d}  z={zz*1000:7.2f}  F={f:8.3f} N", flush=True)
    F_descend = max(r["F_total_N"] for r in log if r["phase"] == "descend")

    # (b) 대칭 폐합 — 립이 피벗 둘레로 도는 것을 평행이동으로 근사
    close_start = len(log)
    for i in range(P["close_steps"]):
        phi = math.radians(P["shell_travel_deg"]) * (1 - (i + 1) / P["close_steps"])
        s.DoDynamicsThenSync(dt)
        f = sample("close", i, {"phi_deg": round(math.degrees(phi), 2)})
        if i % 20 == 0:
            print(f"  폐합 {i:3d}  phi={math.degrees(phi):5.1f}  F={f:8.3f} N", flush=True)
    F_close = max(r["F_total_N"] for r in log[close_start:])

    # (c) 리프트
    lift_start = len(log)
    for i in range(P["lift_steps"]):
        zz = z1 + 0.040 * (i + 1) / P["lift_steps"]
        s.DoDynamicsThenSync(dt)
        sample("lift", i, {"z_mm": round(zz * 1000, 2)})

    # 담긴 입자 = 리프트 후 z 가 원래 더미 최상단보다 높은 것
    pp = np.asarray(s.GetOwnerPosition(0, len(pos) - 1), float)
    captured = int((pp[:, 2] > z_top + 0.005).sum())
    wall = time.time() - t_start

    band = P["jaw_force_band_N"]
    res = {
        "artifact": "DEME_SCOOP_CLOSURE_V1",
        "params": P,
        "pile": {"n": int(len(pos)), "radius_m": rad, "top_z_m": z_top,
                 "source": str(PILE)},
        "forces_N": {"descend_peak": round(F_descend, 3),
                     "close_peak": round(F_close, 3),
                     "jaw_band_measured": band},
        "verdict_closure": ("WITHIN_MEASURED_JAW_FORCE" if F_close <= band[1]
                            else "EXCEEDS_MEASURED_JAW_FORCE"),
        "captured_particles": captured,
        "captured_mass_g": round(captured * mass * 1000, 3),
        "wall_seconds": round(wall, 2),
        "budget_3000_hours": round(wall * 3000 / 3600.0, 2),
        "trajectory": TRAJ,
        "non_claims": (
            "강성 E=5e6 Pa 는 수치 안정용 임시값이며 펠릿 실측이 아니다. "
            "폐합 회전을 피벗 둘레 평행이동으로 근사했다(메시 자체는 회전시키지 않았다). "
            "물성 실측 후 반력은 바뀐다. 벽시계는 이 근사 기준이다."),
    }
    json.dump(res, open(OUT / "scoop_closure.json", "w"), ensure_ascii=False, indent=2)
    json.dump(log, open(OUT / "scoop_timeline.json", "w"), ensure_ascii=False)

    print()
    print(f"하강 최대 반력  {F_descend:8.3f} N")
    print(f"폐합 최대 반력  {F_close:8.3f} N   <- 핵심")
    print(f"실측 조 힘      {band[0]}~{band[1]} N  ->  {res['verdict_closure']}")
    print(f"담긴 입자       {captured} 개 = {res['captured_mass_g']} g")
    print(f"벽시계          {wall:.1f} s  ->  3,000 시행 {res['budget_3000_hours']} 시간")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
