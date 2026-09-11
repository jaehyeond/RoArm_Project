"""DEME S1 그랩(고정 반쪽 보울 + 서보 직결 문) 퍼내기 — 실물 절차(D481) 를 시뮬로 재현해 회당 포획 질량을 낸다.

무엇을 재는가 (실물 `hw_s1_manual.py scoop` 과 같은 순서)
    펠릿면에 립 → 문 열림(서보 30°) → 잠김 2.5 cm 하강 → 문 닫기 → +8 cm 상승 → 재닫기.
    출력 = 포획 입자 수·질량 g · 문 정지각 · 립 등가 힘 피크 · 퍼낸 뒤 heightmap · 타임라인.

기구 (s1_v1/design.json · s1_meta.json · roarm_m3_s1.urdf 원문으로 확인)
    고정부 STL = link5 프레임 mm. 보울 중심 link5 (8.1, 0, 145), 닫힘 립 = (8.1, 0, 166.6).
    문 STL = gripper_link 프레임 mm. gripper_link → link5: x_g=link5 Z, y_g=link5 X, z_g=link5 Y,
        원점 = link5 (0, 18.821, 52.035) (URDF joint link5_to_gripper_link, rpy −90,−90,0).
    문 관절축 = link5 +Y, 힌지점 (0, *, 52.035). +q 가 열림 — STL 로 검증: q=+29.3° 에서
        문 립 (63.1, ·, 148.0), 고정 립과 58.09 mm (design 게이트 58.0 과 일치).
    서보각 → 관절각: 빈 상태 닫힘이 서보 2.5° (기계 정지 = 립 맞닿음, D481) 이므로
        q_joint = servo_deg − 2.5. 문 30° 명령 = q 27.5°.

좌표 (실물 09-07 FK, hw_s1_scoop_probe.chain 으로 확인): 툴 수직.
    link5 X = 세계 −Y, link5 Y = 세계 −X, link5 Z = 세계 −Z  →  R_W 열벡터.
    DEME 세계 = 더미 npz 프레임(상자 바닥 중심, z 위). 문은 세계 −Y 쪽에서 +Y 로 닫힌다
    (= 능선 방향을 따라 쓸어 담는다).

구동 (sim_deme_scoop.py 규약 그대로: Track 은 Initialize 전에, 규정 속도, dt_sync 0.004)
    고정부(family 10)·문(11 닫는 중 / 12 정지) 모두 규정 선속도 vz(t). 문은 힌지(=owner 원점)
    둘레 규정 각속도. 문 정지 = 트래커 `SetFamily(12)` 로 실행 중 전환.
    서보 정지 모델: 힌지축 둘레 저항 모멘트 ≥ 서보 토크(1.96 N·m × 토크 설정 0.9) 이면 그 각에서 멈춘다.
    실물처럼 상승 뒤 재닫기 1회.

W8 (09-10): `particle_shape: "npz_template"` = W7 렌즈 클럼프 더미(npz schema 2) 를 clump_template_json 으로 되살려 읽는다.
    구 전개(GetOwnerOriQ) 로 펠릿면·heightmap(전/후/차분) 을 내고 crater_angles() 로 구덩이 옆면 각을 보고한다.
    구 더미 경로(sphere) 는 그대로다.

⚠️ 물성은 실측 전(JSON 으로 교체). E 5e6 은 수치 안정값(더미는 1e7 로 정착 → t=0 미세 재안착).
   반력·정지각은 물성에 종속이므로 실측 후 다시 돌려야 한다. 이 파일은 경로·절차의 성립과
   상대 비교(설계 17.6 g 대비)만 주장한다.
"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent
S1 = REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1"
FIXED_STL, DOOR_STL, DESIGN = S1 / "fixed_ALL.stl", S1 / "door_ALL_jawframe.stl", S1 / "design.json"

DEFAULT = {
    # 기구
    "bowl_center_l5_mm": [8.1, 0.0, 145.0], "lip_l5_mm": [8.1, 0.0, 166.6],
    "hinge_l5_mm": [0.0, 18.821, 52.035], "bowl_r_in_mm": 20.0, "cheek_half_y_mm": 18.2,
    "servo_zero_offset_deg": 2.5,        # 서보 2.5° = 립 맞닿음(기계 정지, D481)
    "door_open_servo_deg": 30.0, "close_end_joint_deg": 0.0,
    "servo_torque_Nm": 1.96, "servo_torque_fraction": 0.9, "servo_stall_model": True,
    "door_mass_kg": 0.02416, "fixed_mass_kg": 0.01827,
    # 궤적 (실물: 하강 ~10 mm/s · 문 ~12°/s · 상승 ~30 mm/s. 시뮬은 예산 때문에 빠르게)
    "approach_gap_mm": 10.0, "plunge_mm": 25.0, "lift_mm": 80.0,
    "descend_mm_s": 25.0, "close_deg_s": 45.0, "lift_mm_s": 150.0,
    "reclose_max_steps": 30, "settle_steps": 25, "dt_sync_s": 0.004,
    # 팔 힘 제한 하강: 고정부에 걸리는 위쪽 반력 > 이 값이면 그 스텝은 정지(실물 어깨 1.96 N·m / 리치 0.34 m ≈ 5.7 N).
    # 규정 위치 하강(무한 힘)은 E 5e6 구를 바닥 힘사슬에 눌러 4 mm 까지 찌그러뜨린 뒤 20~50 m/s 로 튕긴다(진단 09-09).
    "arm_force_max_N": 2.0, "descend_max_steps_factor": 3.0, "pop_speed_m_s": 5.0,
    "dt_sync_descend_s": 0.0005,         # 하강만 잘게 동기화 — 립 아래 펠릿이 4 ms 안에 눌렸다 튀므로 힘 제한이 그 안에 봐야 한다
    "footprint_r_mm": 22.0,              # 펠릿면 = 이 반경 안 입자 최상단
    # 충돌 셸 = 보울 벽+캡만(힌지·암·판·뺨은 펠릿과 닿을 수 없다. DEME 는 삼각형 수·크기에 스텝 비용이
    # 크게 민감: 원본 STL 6.5k 삼각형 = 12 s/스텝). 벽(1.6)·캡(2.0) 이 입자 반지름(2.08) 보다 얇으면
    # E 5e6 입자가 벽을 관통해 튀므로 바깥쪽으로만 두껍게 한다(공동·입 불변, 립 기준은 최하점을 따라간다).
    "collision_wall_extra_mm": 3.0, "wall_t_mm": 1.6, "cap_t_mm": 2.0,
    "E_mesh_pa": 3.0e9,                  # 툴 재질 = PLA(design.json E_pla_MPa 3000). None 이면 입자와 동일
    "scoop_x_mm": 0.0, "scoop_y_range_mm": [-25.0, 25.0],   # seed → 능선 위 y 위치
    # 물성 (⚠️ 임시값 — --params JSON 으로 교체)
    "particle_shape": "sphere", "timestep_s": 1.0e-5, "E_pa": 5.0e6, "nu": 0.30,
    "CoR": 0.30, "mu": 0.50, "Crr": 0.05, "particle_density_kg_m3": 950.0,
    "pellet_dia_mm": 4.16, "pellet_len_mm": 4.16, "clump_aspect": None,
    "error_out_vel": 60.0, "cd_update_freq": 20, "domain_top_m": 0.50, "domain_bc": "all",   # 천장으로 pop 가둠
    "bulk_density_g_cm3": 0.55,
    # W8: 구덩이 옆면 각(절단면 각) 정의 파라미터 — crater_angles() 도크스트링이 정의의 정본
    "crater_r_max_mm": 80.0, "crater_wedge_half_deg": 22.5, "crater_min_depth_mm": 2.0, "crater_fit_band": [0.2, 0.8],
    # W8 렌더 타임라인(코디네이터 요청, Isaac 재생용 — 물리 판정과 무관): 물리 시간 dt 마다 클럼프 pos/quat·툴 포즈·문 각 저장. None = 끔
    "render_timeline_dt_s": None, "render_timeline_path": None,
    # W8 옵션 A: 폐합·재폐합만 잘게 동기화(서보 정지 판정 간격). None = dt_sync_s(4 ms) 그대로. 회전 속도(close_deg_s)는 불변.
    "dt_sync_close_s": None,
    # W8 옵션 E: 물림 가드 — 폐합·재폐합 중 툴(고정부·문) 단일 접촉력 ≥ 이 값이면 서보 정지로 취급(reason "pinch_guard"). None = 끔.
    # 근거: 임시 강성 E 5e6 에서는 렌즈 구(r 1.16 mm)가 립 면을 F = 4/3·E*·r² ≈ 4.9 N 에 통과하고(정지 립 힘 20.5 N 보다 훨씬 작음),
    # 통과 즉시 셸 바깥 면이 r+벽두께 관입으로 ~50 N 에 튕겨낸다(셀 c·xp50·A_c 발산 81/203/226 m/s). 실물 PP 알은 립을 통과하지 못하므로
    # "단단한 물림 = 정지" 로 대신한다. 사전 등록 외 수치 가드이며 문 정지각은 토크 정지보다 이를 수 있다(보고서 비주장).
    "door_pinch_guard_N": None,
    # W8 옵션 F: 문 하한각(관절 °) — q 가 이 값에 닿으면 정지(reason "door_floor"). 렌즈 클럼프 발산 구간(q 2.3~3.6°, 립 틈 3.4~5.4 mm) 위에서 멈춘다. None = 끔.
    "door_min_q_deg": None,
    # W8 진단: q 가 이 값 아래(폐합·재폐합)면 매 sync 타임라인 flush + 최대속도 owner 위치·최대 접촉점 기록. None = 끔.
    "diag_flush_below_q_deg": None,
    # W10 진단 v2 (전부 None 이면 W8 경로 그대로). diag_flush_below_q_deg 구간(폐합·재폐합)에서:
    #   diag_fine_sync_s   = sync 를 이 길이로(예 1e-4 = 10 스텝). 문 각속도·물성 불변, q 감소량은 sync 길이에 비례.
    #   diag_event_v_m_s / diag_event_force_N = pop 이벤트 트리거(입자 최대속도 / 툴 단일 접촉) → 링버퍼(diag_ring_syncs) + culprit 직전 sync 표 +
    #                        GetContactDetailedInfo(try) 를 diverge_event_<tag>.{json,npz} 로 덤프(1회).
    #   diag_stop_v_m_s    = 파이썬 pop-stop(전 phase): 입자 최대속도가 넘으면 RuntimeError → 기존 발산 경로(산출은 남긴다).
    #                        DEME 의 C++ terminate(error_out_vel) 대신 상태를 남기려면 params 로 error_out_vel 을 올려 쓴다.
    #   diag_near_lip_mm   = 링버퍼에 담는 클럼프 = 두 립선에서 이 거리 안(+외접 반지름).
    #   max_velocity_m_s   = DEME SetMaxVelocity(CD 마진 상한; 넘으면 anomaly 로그만, 물리 불변). pop 뒤 마진 폭주 방지용. None = 미설정(auto).
    "diag_fine_sync_s": None, "diag_event_v_m_s": None, "diag_event_force_N": None, "diag_stop_v_m_s": None,
    "diag_ring_syncs": 12, "diag_near_lip_mm": 12.0, "max_velocity_m_s": None,
}
R_G2L5 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)     # gripper → link5
R_W = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)      # link5 → 세계 (열 = link5 축)


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def roty(deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def half_bowl(P, side, n_th=24, groups=None):
    """반쪽 보울 충돌 셸(link5 mm) 을 설계값으로 해석적으로 만든다. side −1 = 고정(x ≤ 8.1), +1 = 문.

    🔴 STL(fixed_ALL/door_ALL) 을 그대로 쓰면 안 된다: 조각 볼록체 합집합이라 내부 중복면이 많고,
    입자가 공동 안쪽 면에 닿는 순간 DEME 가 터진다(진단: 잠김 6 mm 에서 141 m/s, 단일 표면 셸은 정상).
    내경·폭은 설계와 같고(공동 45.7 cm³ 불변), 벽·캡 두께만 바깥으로 collision_wall_extra_mm 더한다.
    p(θ, r, y) = (8.1 + r sinθ, y, 145 + r cosθ); θ=0 이 립(z 최대), 문 쪽은 sinθ ≥ 0.
    """
    C = P["bowl_center_l5_mm"]; r_in = P["bowl_r_in_mm"]; ex = P["collision_wall_extra_mm"]
    r_out = r_in + P["wall_t_mm"] + ex; h = P["cheek_half_y_mm"]; H = h + P["cap_t_mm"] + ex
    cap = P.get("cap_mode", "solid")
    if cap != "solid":
        H = h                                                       # 바깥 면 폭도 벽 폭으로
    th = np.linspace(0.0, side * math.pi, n_th + 1)
    pt = lambda t, r, y: np.array([C[0] + r * math.sin(t), y, C[2] + r * math.cos(t)])
    V, F = [], []

    def quad(a, b, c, d, outward, g=None):    # 네 점 + 바깥 방향 → 삼각형 2, 법선을 바깥으로. g = 삼각형 그룹 태그(W10 진단, groups 가 주어질 때만)
        i = len(V); V.extend([a, b, c, d])
        n = np.cross(np.asarray(b) - a, np.asarray(c) - a)
        F.extend([[i, i + 1, i + 2], [i, i + 2, i + 3]] if np.dot(n, outward) >= 0 else [[i, i + 2, i + 1], [i, i + 3, i + 2]])
        if groups is not None:
            groups.extend([g, g])

    def tri(a, b, c, outward, g=None):
        i = len(V); V.extend([a, b, c]); n = np.cross(np.asarray(b) - a, np.asarray(c) - a)
        F.append([i, i + 1, i + 2] if np.dot(n, outward) >= 0 else [i, i + 2, i + 1])
        if groups is not None:
            groups.append(g)
    for k in range(n_th):
        t0, t1 = th[k], th[k + 1]; rad_dir = pt((t0 + t1) / 2, 1, 0) - np.array([C[0], 0, C[2]])
        quad(pt(t0, r_in, -h), pt(t1, r_in, -h), pt(t1, r_in, h), pt(t0, r_in, h), -rad_dir, "inner")      # 안쪽 면
        quad(pt(t0, r_out, -H), pt(t1, r_out, -H), pt(t1, r_out, H), pt(t0, r_out, H), rad_dir, "outer")   # 바깥 면
        for y_in, y_out in ((h, H), (-h, -H)):                                                   # 캡 안/바깥 면
            ny = np.array([0, np.sign(y_in), 0])
            if cap == "solid":
                tri(pt(t0, 0, y_in), pt(t0, r_in, y_in), pt(t1, r_in, y_in), -ny, "cap_in")
                tri(pt(t0, 0, y_out), pt(t0, r_out, y_out), pt(t1, r_out, y_out), ny, "cap_out")
            elif cap == "annular":                                                               # 진단용: 벽 끝만 막음
                quad(pt(t0, r_in, y_in), pt(t1, r_in, y_in), pt(t1, r_out, y_in), pt(t0, r_out, y_in), ny, "cap_in")
    # 파팅면(x = 8.1 평면: 립·윗단·캡 단면) 의 바깥 = 반대쪽 반쪽 방향: 고정(side −1) → +X, 문(side +1) → −X.
    # 🔴 W3(09-09) 는 여기에 (0,0,cos t) 를 줘서 면 법선(±X)과 직교 → 방향이 임의였고 20면 중 8면이 뒤집혔다.
    #    DEME triangle_sphere_CD 는 양면이라 뒤집힌 면의 발자국 안·뒤쪽에 있는 정상 입자에 관입 r+|h|(수 mm, ~60 N)
    #    를 준다 = W3 발산의 원인(W3b b_diverge/: cut_base 21 m/s → cut_fixnorm 0.37 m/s).
    nx = np.array([-side, 0, 0], float)
    for t, lab in ((th[0], "lip"), (th[-1], "bottom")):     # θ=0 = 립 띠, θ=π = 바닥 띠 (그룹 태그 part_lip / part_bottom / part_cap_*)
        quad(pt(t, r_in, -h), pt(t, r_out, -h), pt(t, r_out, h), pt(t, r_in, h), nx, f"part_{lab}")
        if cap == "solid":
            for y_in, y_out in ((h, H), (-h, -H)):
                quad(pt(t, 0, y_in), pt(t, r_out, y_in), pt(t, r_out, y_out), pt(t, 0, y_out), nx, f"part_cap_{lab}")
    m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F), process=False); m.merge_vertices()
    return m


def load_tool(P, q_open_deg):
    """고정부·문 충돌 셸을 세계 프레임(m)·owner 원점 기준으로 굽는다. 문은 q_open 자세.

    고정부 owner 원점 = 립점(8.1, 0, z_lip) → Pos() 가 곧 립 세계 위치. 문 owner 원점 = 힌지점.
    STL 은 출처 해시와 기하 일치 검사(보울 구간 정점의 반경·폭 범위)에만 쓴다.
    """
    H5 = np.array(P["hinge_l5_mm"]); C = P["bowl_center_l5_mm"]
    Fm, Dm = half_bowl(P, -1), half_bowl(P, +1)
    fv, dv = np.asarray(Fm.vertices, float), np.asarray(Dm.vertices, float)
    L5 = np.array([C[0], 0.0, fv[:, 2].max()])                      # 두꺼워진 립(최하점)
    lip_idx_f = np.where(fv[:, 2] > L5[2] - 0.5)[0]
    lip_idx_d = np.where(dv[:, 2] > L5[2] - 0.5)[0]
    # STL 기하 일치 검사: 보울 구간(z_l5 > 124) 정점의 반경이 [r_in, r_in+wall] 안, |y| ≤ 캡 바깥
    stl = trimesh.load(FIXED_STL); sv = np.asarray(stl.vertices, float); sv = sv[sv[:, 2] > 124]
    r = np.hypot(sv[:, 0] - C[0], sv[:, 2] - C[2]); wall = P["bowl_r_in_mm"] + P["wall_t_mm"]
    check = {"stl_bowl_vertices": int(len(sv)), "r_min_mm": round(float(r.min()), 2), "r_max_mm": round(float(r.max()), 2),
             "abs_y_max_mm": round(float(np.abs(sv[:, 1]).max()), 2),
             "r_in_ok": bool(abs(r[r < wall - 0.8].min() - P["bowl_r_in_mm"]) < 0.3) if (r < wall - 0.8).any() else False,
             "r_out_design_mm": wall, "note_r_max": "STL 보울 구간에는 스파인·뺨도 포함되어 r_max 가 벽 바깥값보다 크다", "shell_watertight": bool(Fm.is_watertight and Dm.is_watertight),
             "shell_tri": [int(len(Fm.faces)), int(len(Dm.faces))]}
    dv_open = (roty(q_open_deg) @ (dv - H5).T).T + H5
    fixed = trimesh.Trimesh((R_W @ (fv - L5).T).T / 1000.0, Fm.faces, process=False)
    door = trimesh.Trimesh((R_W @ (dv_open - H5).T).T / 1000.0, Dm.faces, process=False)
    return fixed, door, lip_idx_f, lip_idx_d, R_W @ (H5 - L5) / 1000.0, L5, check


def add_particles(s, z, P, mat):
    """더미 npz → DEME 입자. sphere = positions_m/radii_m, clump2 = clump_positions_m/quaternions,
    npz_template(W8) = npz 의 clump_template_json 으로 LoadClumpType 재구성(W7 렌즈 더미).
    반환 5번째 = 템플릿 dict(구 전개용; sphere/clump2 는 None)."""
    if P["particle_shape"] == "sphere":
        if "clump_template_json" in z:
            raise SystemExit("이 더미는 클럼프 npz(schema 2, clump_template_json) — particle_shape 'npz_template' 로 읽어야 한다")
        pos = np.asarray(z["positions_m"], float); rad = float(np.asarray(z["radii_m"], float)[0])
        m_p = P["particle_density_kg_m3"] * 4.0 / 3.0 * math.pi * rad ** 3
        s.AddClumps(s.LoadSphereType(m_p, rad, mat), pos.tolist())
        return pos, rad, m_p, {"shape": "sphere", "radius_m": rad}, None
    if P["particle_shape"] == "npz_template":
        return add_template_clumps(s, z, mat)
    if "clump_positions_m" not in z:
        raise SystemExit("clump2 는 클럼프로 정착한 더미 npz(clump_positions_m/clump_quaternions_xyzw)가 필요하다")
    import sim_pellet_model as PM
    tpl = PM.build_template(PM.PelletSpec(P["pellet_dia_mm"], P["pellet_len_mm"],
                                          P["particle_density_kg_m3"]), P["particle_shape"],
                            aspect=P["clump_aspect"])
    ct = s.LoadClumpType(tpl.mass_kg, list(tpl.moi_kg_m2), [tpl.sphere_radius_m] * tpl.n_spheres,
                         [list(o) for o in tpl.offsets_m], mat)
    ct.SetVolume(tpl.union_volume_m3)
    pos = np.asarray(z["clump_positions_m"], float)
    s.AddClumps(ct, pos.tolist()).SetOriQ(np.asarray(z["clump_quaternions_xyzw"], float).tolist())
    return pos, tpl.bounding_diameter_m / 2, tpl.mass_kg, {"shape": tpl.name, "aspect": tpl.aspect,
                                                            "sphere_radius_m": tpl.sphere_radius_m}, None


def expand_spheres(pos, quat_xyzw, tpl):
    """클럼프 중심·자세(xyzw) → 구 중심·반경(클럼프-major). W7 npz 행 규약과 같은 식 p_sphere = p_clump + R(q)·offset."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(np.asarray(quat_xyzw, float)).as_matrix()          # scipy 도 xyzw
    offs = np.asarray(tpl["offsets_m"], float)
    sp = np.asarray(pos, float)[:, None, :] + np.einsum("nij,kj->nki", R, offs)
    return sp.reshape(-1, 3), np.tile(np.asarray(tpl["sphere_radii_m"], float), len(pos))


def add_template_clumps(s, z, mat):
    """W7 렌즈 더미(schema 2): npz 의 clump_template_json 만으로 LoadClumpType 을 되살려 정착 위치·자세 그대로 놓는다.
    질량(알 20.26 mg)·MOI·구별 반경·상대위치 = 템플릿 값. 물성만 mat(파라미터). sim_pellet_model 은 import 하지 않는다
    (메인 워크트리 판은 lens 필드가 없다)."""
    for k in ("clump_template_json", "clump_positions_m", "clump_quaternions_xyzw"):
        if k not in z:
            raise SystemExit(f"npz_template 는 npz 키 {k} 가 필요하다")
    tpl = json.loads(str(z["clump_template_json"]))
    radii = [float(r) for r in tpl["sphere_radii_m"]]; offs = [[float(v) for v in o] for o in tpl["offsets_m"]]
    ct = s.LoadClumpType(float(tpl["mass_kg"]), [float(v) for v in tpl["moi_kg_m2"]], radii, offs, mat)
    ct.SetVolume(float(tpl["union_volume_m3"]))
    pos = np.asarray(z["clump_positions_m"], float); quat = np.asarray(z["clump_quaternions_xyzw"], float)
    s.AddClumps(ct, pos.tolist()).SetOriQ(quat.tolist())
    sp, _ = expand_spheres(pos, quat, tpl)                                       # 자체 검사: 전개식이 npz 구 행을 재현하는가
    err = float(np.abs(sp - np.asarray(z["positions_m"], float)).max()) if "positions_m" in z else None
    ids_ok = bool(np.array_equal(np.asarray(z["clump_ids"]), np.repeat(np.arange(len(pos)), len(radii)))) if "clump_ids" in z else None
    info = {"shape": f"npz_template:{tpl.get('name')}", "n_spheres": len(radii), "sphere_radii_m": radii, "offsets_m": offs,
            "mass_kg": float(tpl["mass_kg"]), "moi_kg_m2": tpl["moi_kg_m2"], "union_volume_m3": tpl["union_volume_m3"],
            "bounding_diameter_m": tpl["bounding_diameter_m"], "lens_axes_mm": tpl.get("lens", {}).get("axes_mm_input"),
            "expand_vs_npz_max_err_m": err, "clump_ids_clump_major": ids_ok}
    print(f"템플릿 클럼프 {len(pos)} × {len(radii)}구 · 알 {tpl['mass_kg']*1e6:.3f} mg · 전개 검사 max|Δ| {err} m · clump_ids {ids_ok}", flush=True)
    return pos, float(tpl["bounding_diameter_m"]) / 2, float(tpl["mass_kg"]), info, tpl


def surface_z(sp, sr, x_s, y_s, P):
    """펠릿면 = 발자국 반경 안 구의 윗면 최대(구별 반경)."""
    m = np.hypot(sp[:, 0] - x_s, sp[:, 1] - y_s) < P["footprint_r_mm"] / 1000.0
    return float((sp[m, 2] + sr[m]).max())


def crater_angles(hm_pre, hm_post, spec, site_xy, P):
    """퍼낸 뒤 구덩이 옆면 각 — R1 §5·§7 "한 입 뒤 절단면 각" 의 시뮬판 정의. 이 함수가 정의의 정본이다(GATES_w8.md 요약).

    dh = hm_pre − hm_post (m, > 0 = 깎여 나간 깊이). hm 은 5 mm 셀 heightmap(경로 A, 셀 발자국 최고점).
    구덩이 중심 = 퍼내기 위치 반경 crater_r_max 안에서 dh ≥ crater_min_depth 인 셀의 dh 가중 무게중심(없으면 퍼내기 위치).
    4 방위(+x, +y, −x, −y 세계축) 마다: 중심에서 그 방위 ±crater_wedge_half 쐐기 안·r ≤ crater_r_max 셀을 셀 폭 고리로 묶어
      고리 평균 프로파일 dh(r), h_post(r) 를 만든다. r_peak = dh(r) 최대 고리. 그 바깥쪽으로 dh 가 band[0]·d_max 아래로
      처음 떨어지기 전까지의 고리 중 band[0]·d_max ≤ dh ≤ band[1]·d_max 인 고리에 최소제곱 직선 dh = a·r + b 를 맞추고
      옆면 각 = atan(|a|) (수평 기준, °). 같은 고리에 h_post = a'·r + b' 를 맞춘 atan(|a'|) 이 절대 표면 경사각(보조;
      실물 Kinect 는 절대 표면을 본다). 고리 2개 미만이면 None + 사유. 판정 아님, 보고값.
    """
    c = spec.cell_m; rows, cols = hm_pre.shape
    xs = spec.origin_xy_m[0] + (np.arange(cols) + 0.5) * c; ys = spec.origin_xy_m[1] + (np.arange(rows) + 0.5) * c
    X, Y = np.meshgrid(xs, ys)
    dh = np.asarray(hm_pre, float) - np.asarray(hm_post, float)
    r_max = P["crater_r_max_mm"] / 1000.0; d_min = P["crater_min_depth_mm"] / 1000.0; lo, hi = P["crater_fit_band"]
    site = np.hypot(X - site_xy[0], Y - site_xy[1]) <= r_max
    deep = site & (dh >= d_min)
    if deep.any():
        w = dh[deep]; cx, cy = float((X[deep] * w).sum() / w.sum()), float((Y[deep] * w).sum() / w.sum())
    else:
        cx, cy = float(site_xy[0]), float(site_xy[1])
    R = np.hypot(X - cx, Y - cy); A = np.degrees(np.arctan2(Y - cy, X - cx))
    out = {"definition": "crater_angles() docstring (sim_deme_scoop_s1.py): dh=h_pre-h_post, wedge ring-averaged radial profile, "
                         "least-squares line on rings with band[0]*d_max<=dh<=band[1]*d_max outward of r_peak, angle=atan|slope|",
           "params": {"r_max_mm": P["crater_r_max_mm"], "wedge_half_deg": P["crater_wedge_half_deg"],
                      "min_depth_mm": P["crater_min_depth_mm"], "fit_band": [lo, hi], "cell_mm": c * 1000},
           "site_xy_mm": [site_xy[0] * 1000, site_xy[1] * 1000], "center_xy_mm": [cx * 1000, cy * 1000],
           "center_rule": "dh-weighted centroid of cells with dh>=min_depth within r_max of site" if deep.any() else "site (no cell deeper than min_depth)",
           "n_cells_deep": int(deep.sum()), "removed_volume_cm3": float(np.clip(dh[site], 0, None).sum() * c * c * 1e6),
           "dh_max_mm": float(dh[site].max() * 1000) if site.any() else None, "azimuths": {}}
    for name, a0 in (("+x", 0.0), ("+y", 90.0), ("-x", 180.0), ("-y", -90.0)):
        dang = np.abs((A - a0 + 180.0) % 360.0 - 180.0)
        m = (dang <= P["crater_wedge_half_deg"]) & (R <= r_max)
        ib = np.floor(R[m] / c).astype(int); nb = int(ib.max()) + 1 if m.any() else 0
        cnt = np.bincount(ib, minlength=nb).astype(float)
        prof_dh = np.bincount(ib, weights=dh[m], minlength=nb) / np.maximum(cnt, 1)
        prof_h = np.bincount(ib, weights=np.asarray(hm_post, float)[m], minlength=nb) / np.maximum(cnt, 1)
        r_c = (np.arange(nb) + 0.5) * c
        valid = cnt > 0
        az = {"r_mm": (r_c * 1000).round(2).tolist(), "n_cells": cnt.astype(int).tolist(),
              "dh_ring_mm": [None if not v else round(float(d * 1000), 3) for v, d in zip(valid, prof_dh)],
              "h_post_ring_mm": [None if not v else round(float(h * 1000), 3) for v, h in zip(valid, prof_h)]}
        if not valid.any() or prof_dh[valid].max() <= 0:
            az.update(angle_deg=None, angle_surface_deg=None, reason="no positive depth in wedge"); out["azimuths"][name] = az; continue
        ipk = int(np.argmax(np.where(valid, prof_dh, -np.inf))); d_max = float(prof_dh[ipk])
        sel = []
        for i in range(ipk, nb):
            if not valid[i] or prof_dh[i] < lo * d_max:
                break
            if prof_dh[i] <= hi * d_max:
                sel.append(i)
        az.update(r_peak_mm=float(r_c[ipk] * 1000), d_max_mm=d_max * 1000, fit_bins=[int(i) for i in sel])
        if len(sel) < 2:
            az.update(angle_deg=None, angle_surface_deg=None, reason=f"only {len(sel)} ring(s) in band outward of r_peak"); out["azimuths"][name] = az; continue
        rr_ = r_c[sel]; a, b = np.polyfit(rr_, prof_dh[sel], 1); a2, b2 = np.polyfit(rr_, prof_h[sel], 1)
        az.update(angle_deg=float(np.degrees(np.arctan(abs(a)))), slope_dh=float(a), intercept_dh_m=float(b),
                  angle_surface_deg=float(np.degrees(np.arctan(abs(a2)))), slope_h_post=float(a2), intercept_h_post_m=float(b2),
                  r_fit_mm=[float(rr_.min() * 1000), float(rr_.max() * 1000)], n_rings=len(sel))
        out["azimuths"][name] = az
    return out


def plot_heightmaps(out, tag, spec, hm_pre, hm_post, crater, site_xy):
    """D324 결정 시점 진단: heightmap 전/후/차분 + 구덩이 중심·4방위, 그리고 방위별 고리 프로파일과 직선 적합."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    c = spec.cell_m; ext = [spec.origin_xy_m[0] * 1000, (spec.origin_xy_m[0] + spec.shape[1] * c) * 1000,
                            spec.origin_xy_m[1] * 1000, (spec.origin_xy_m[1] + spec.shape[0] * c) * 1000]
    dh = (np.asarray(hm_pre, float) - np.asarray(hm_post, float)) * 1000
    vmax = float(max(np.nanmax(hm_pre), np.nanmax(hm_post)) * 1000); dmax = float(max(np.abs(dh).max(), 1e-6))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for k, (arr, ttl, cmap, vm) in enumerate(((np.asarray(hm_pre) * 1000, "pre (after settle, before descend) mm", "viridis", (0, vmax)),
                                             (np.asarray(hm_post) * 1000, "post (final, resting particles) mm", "viridis", (0, vmax)),
                                             (dh, "removed depth = pre - post mm", "RdBu_r", (-dmax, dmax)))):
        im = ax[k].imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=vm[0], vmax=vm[1], aspect="equal")
        fig.colorbar(im, ax=ax[k], fraction=0.03)
        ax[k].set_title(ttl, fontsize=9); ax[k].set_xlabel("world x mm"); ax[k].set_ylabel("world y mm")
        ax[k].plot(site_xy[0] * 1000, site_xy[1] * 1000, "k+", ms=10, label="scoop site (lip)")
        cx, cy = crater["center_xy_mm"]; ax[k].plot(cx, cy, "wx", ms=8, label="crater centre")
        r = crater["params"]["r_max_mm"]
        for name, a0 in (("+x", 0), ("+y", 90), ("-x", 180), ("-y", -90)):
            ax[k].plot([cx, cx + r * math.cos(math.radians(a0))], [cy, cy + r * math.sin(math.radians(a0))], "w--", lw=0.6)
            az = crater["azimuths"].get(name, {})
            lab = f"{name} {az['angle_deg']:.1f}" if az.get("angle_deg") is not None else f"{name} n/a"
            ax[k].text(cx + (r + 8) * math.cos(math.radians(a0)), cy + (r + 8) * math.sin(math.radians(a0)), lab, color="k", fontsize=7, ha="center")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"S1 scoop {tag}: heightmap pre / post / removed depth (5 mm cells); labels = crater wall angle deg"); fig.tight_layout()
    fig.savefig(out / f"heightmap_{tag}.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for k, name in enumerate(("+x", "+y", "-x", "-y")):
        az = crater["azimuths"][name]; r = np.asarray(az["r_mm"], float)
        d = np.asarray([np.nan if v is None else v for v in az["dh_ring_mm"]], float)
        h = np.asarray([np.nan if v is None else v for v in az["h_post_ring_mm"]], float)
        ax[k].plot(r, d, "o-", ms=3, label="depth dh(r) mm"); ax[k].plot(r, h, "s--", ms=3, c="0.5", label="h_post(r) mm")
        if az.get("angle_deg") is not None:
            rf = np.linspace(az["r_fit_mm"][0] - 5, az["r_fit_mm"][1] + 5, 10)
            ax[k].plot(rf, (az["slope_dh"] * rf / 1000 + az["intercept_dh_m"]) * 1000, "r-", lw=1.5,
                       label=f"fit {az['angle_deg']:.1f} deg (surface {az['angle_surface_deg']:.1f})")
            ax[k].axhline(az["d_max_mm"] * 0.8, c="r", lw=0.4, ls=":"); ax[k].axhline(az["d_max_mm"] * 0.2, c="r", lw=0.4, ls=":")
        ttl = f"azimuth {name}: " + (f"{az['angle_deg']:.1f} deg" if az.get("angle_deg") is not None else f"n/a ({az.get('reason', '')})")
        ax[k].set_title(ttl, fontsize=8); ax[k].set_xlabel("r from crater centre mm"); ax[k].legend(fontsize=6); ax[k].grid(alpha=0.3)
    ax[0].set_ylabel("mm")
    fig.suptitle(f"{tag}: ring-averaged radial profiles; line fit on 20-80 % depth band outward of r_peak"); fig.tight_layout()
    fig.savefig(out / f"crater_profiles_{tag}.png", dpi=110); plt.close(fig)


def run(a):
    import DEME
    P = dict(DEFAULT)
    if a.params:
        P.update(json.load(open(a.params)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{a.seed}" + ("_smoke" if a.smoke else "")
    t_start = time.time()
    rng = np.random.default_rng(a.seed)
    y_s = float(rng.uniform(*P["scoop_y_range_mm"])) / 1000.0
    x_s = P["scoop_x_mm"] / 1000.0
    dt = P["dt_sync_s"]
    q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
    q_end = P["close_end_joint_deg"]
    design = json.load(open(DESIGN))
    R_lip = design["derived"]["lip_radius_from_hinge_mm"] / 1000.0
    M_stall = P["servo_torque_Nm"] * P["servo_torque_fraction"]

    z = np.load(a.pile, allow_pickle=True)
    box = np.asarray(z["box_bounds_m"], float)
    from roarm_rl.heightmap import GridSpec, heightmap_from_particles
    cell = 0.005
    spec = GridSpec(origin_xy_m=(float(box[0, 0]), float(box[1, 0])), cell_m=cell,
                    shape=(int(math.ceil((box[1, 1] - box[1, 0]) / cell)),
                           int(math.ceil((box[0, 1] - box[0, 0]) / cell))),
                    frame="deme_box_floor_center", z_datum_m=0.0)
    s = DEME.DEMSolver(); s.SetVerbosity("ERROR")
    mp = {"E": P["E_pa"], "nu": P["nu"], "CoR": P["CoR"], "mu": P["mu"], "Crr": P["Crr"]}
    mat_p, mat_w = s.LoadMaterial(mp), s.LoadMaterial(mp)
    mat_m = s.LoadMaterial(dict(mp, E=P["E_mesh_pa"] or P["E_pa"]))
    s.UseFrictionalHertzianModel()
    pos, rad, m_p, shape_info, tpl = add_particles(s, z, P, mat_p)
    n_p = len(pos)
    k_sph = 1 if tpl is None else len(tpl["sphere_radii_m"])
    spheres = lambda pp, oq: (pp, np.full(len(pp), rad)) if tpl is None else expand_spheres(pp, oq, tpl)   # 구 전개(heightmap·펠릿면)
    near = np.hypot(pos[:, 0] - x_s, pos[:, 1] - y_s) < P["footprint_r_mm"] / 1000.0
    if tpl is None:
        z_surf = float(pos[near, 2].max()) + rad            # 펠릿면 = 국소 최상단 입자의 윗면
    else:
        z_surf = surface_z(*spheres(pos, np.asarray(z["clump_quaternions_xyzw"], float)), x_s, y_s, P)
    print(f"더미 {n_p} · r {rad*1000:.2f} mm · 퍼내기 위치 ({x_s*1000:.0f},{y_s*1000:.1f}) mm "
          f"· 펠릿면 z {z_surf*1000:.1f} mm", flush=True)
    s.InstructBoxDomainDimension((float(box[0, 0]), float(box[0, 1])),
                                 (float(box[1, 0]), float(box[1, 1])),
                                 (float(box[2, 0]), P["domain_top_m"]))
    s.InstructBoxDomainBoundingBC(P["domain_bc"], mat_w)

    # ── 툴 ──────────────────────────────────────────────────────────────────
    fixed_m, door_m, lip_f, lip_d, hinge_off, L5mm, mesh_check = load_tool(P, q_open)
    P["lip_l5_mm"] = [float(v) for v in L5mm]                    # 두꺼워진 립 기준(결과 JSON 에 남는다)
    print(f"충돌 셸 {mesh_check}", flush=True)
    objs = out / "_obj"; objs.mkdir(exist_ok=True)
    fpath, dpath = objs / f"fixed_{tag}.obj", objs / f"door_open{q_open:.1f}_{tag}.obj"
    fixed_m.export(fpath); door_m.export(dpath)
    z_lip0 = z_surf + P["approach_gap_mm"] / 1000.0
    z_lip1 = z_surf - P["plunge_mm"] / 1000.0
    lip0 = np.array([x_s, y_s, z_lip0])
    mf = s.AddWavefrontMeshObject(str(fpath), mat_m, True, False)
    mf.SetMass(P["fixed_mass_kg"]); mf.SetMOI([1e-5, 1e-5, 1e-5]); mf.SetFamily(10)
    mf.SetInitPos(lip0.tolist())
    md = s.AddWavefrontMeshObject(str(dpath), mat_m, True, False)
    md.SetMass(P["door_mass_kg"]); md.SetMOI([4e-5, 4e-5, 4e-5]); md.SetFamily(12)
    md.SetInitPos((lip0 + hinge_off).tolist())
    axis_w = R_W @ np.array([0.0, 1.0, 0.0])            # 관절축(link5 +Y) 의 세계 방향
    # 문 립 시작 자세 자체 검사: 입(문 립 ↔ 고정 립 최소거리)
    from scipy.spatial import cKDTree
    mouth0 = float(cKDTree(np.asarray(fixed_m.vertices)[lip_f] + lip0).query(
        np.asarray(door_m.vertices)[lip_d] + lip0 + hinge_off)[0].min()) * 1000

    # ── 구동 = family 별 상수 속도. 팔: 10 하강 / 13 정지 / 14 상승. 문: 20/21/22 = 팔과 동행, 23 = 정지+닫힘 회전.
    #    시간 수식 대신 실행 중 트래커 SetFamily 로 전환한다(팔 힘 제한·문 정지 모두 상태 의존이라 t 로 못 쓴다).
    dt_d = P["dt_sync_descend_s"]
    n_desc = max(1, int(math.ceil((z_lip0 - z_lip1) / (P["descend_mm_s"] / 1000.0) / dt_d)))
    n_close = max(1, int(math.ceil((q_open - q_end) / P["close_deg_s"] / dt)))
    n_lift = max(1, int(math.ceil((P["lift_mm"] / 1000.0) / (P["lift_mm_s"] / 1000.0) / dt)))
    if a.smoke:                     # 속도는 그대로, 상승 거리만 짧게 (속도를 올리면 발산한다)
        P["lift_mm"], P["reclose_max_steps"] = 20.0, 10
        n_lift = max(1, int(math.ceil((P["lift_mm"] / 1000.0) / (P["lift_mm_s"] / 1000.0) / dt)))
    vz_desc = -P["descend_mm_s"] / 1000.0
    w_close = math.radians(q_open - q_end) / (n_close * dt)     # n_close 스텝 회전이면 정확히 q_end
    dt_c = P.get("dt_sync_close_s") or dt                       # 폐합 sync (옵션 A: 1 ms → 정지 판정 4배 촘촘)
    n_close_sync = max(1, int(round(n_close * dt / dt_c))); n_reclose_max = max(1, int(round(P["reclose_max_steps"] * dt / dt_c)))
    vz_lift = P["lift_mm_s"] / 1000.0
    w_vec = -w_close * axis_w                                   # 닫힘 = q 감소
    FAM_F = {"desc": 10, "hold": 13, "lift": 14}; FAM_D = {"desc": 20, "hold": 21, "lift": 22, "close": 23}
    VZ = {"desc": vz_desc, "hold": 0.0, "lift": vz_lift, "close": 0.0}
    for k, fam in list(FAM_F.items()) + list(FAM_D.items()):
        s.SetFamilyPrescribedLinVel(fam, "0", "0", f"{VZ[k]:.9f}", True)
        s.SetFamilyPrescribedAngVel(fam, *([f"{v:.9f}" for v in w_vec] if k == "close" else ["0"] * 3), True)
    mf.SetFamily(FAM_F["hold"]); md.SetFamily(FAM_D["hold"])
    s.SetInitTimeStep(P["timestep_s"]); s.SetGravitationalAcceleration([0, 0, -9.81])
    s.SetCDUpdateFreq(P["cd_update_freq"]); s.SetErrorOutVelocity(P["error_out_vel"])
    if P.get("max_velocity_m_s"):                               # W10: CD 마진 상한(anomaly 로그만; 물리 불변)
        s.SetMaxVelocity(float(P["max_velocity_m_s"]))
    if P.get("diag_event_v_m_s") or P.get("diag_event_force_N"):   # W10: GetContactDetailedInfo 에 법선·geo id·접촉형을 싣는다(출력 내용만, 물리 불변)
        s.SetContactOutputContent(["OWNER", "GEO_ID", "POINT", "FORCE", "NORMAL", "CNT_TYPE"])
    trk_f, trk_d = s.Track(mf), s.Track(md)                     # 🔴 Initialize 전에
    s.Initialize()
    owners = {"fixed": int(trk_f.GetOwnerID()), "door": int(trk_d.GetOwnerID())}
    if any(v >= 2 ** 32 - 1 for v in owners.values()):
        raise RuntimeError("트래커 owner 미할당")
    print(f"Initialize OK · owner {owners} · 스텝(명목) 하강 {n_desc} 폐합 {n_close} 상승 {n_lift} "
          f"· 시작 입 {mouth0:.1f} mm · 팔 힘 상한 {P['arm_force_max_N']} N", flush=True)

    # ── 기록 ───────────────────────────────────────────────────────────────
    hinge_w = lambda: np.asarray(trk_d.Pos(), float)
    log, rec = [], {"t": [], "phase": [], "nodes_F": [], "nodes_D": [], "cp": [], "cf": [], "cframe": []}
    state = {"q": q_open, "arm": "hold", "door": "hold", "stop": None, "holds": 0, "pop_steps": 0, "v_max": 0.0}
    rt = None
    if P.get("render_timeline_dt_s"):
        rt = {k: [] for k in ("t", "phase", "frame", "pos", "quat", "tool_pos", "tool_quat", "door_pos", "door_quat", "door_deg")}; rt["next"] = 0.0

    def rt_record(sim_t, phase_id, frame_idx):
        """렌더 타임라인 1 프레임(float32). 물리 판정에는 쓰지 않는다."""
        rt["t"].append(float(sim_t)); rt["phase"].append(phase_id); rt["frame"].append(frame_idx)
        rt["pos"].append(np.asarray(s.GetOwnerPosition(0, n_p), np.float32)); rt["quat"].append(np.asarray(s.GetOwnerOriQ(0, n_p), np.float32))
        rt["tool_pos"].append(np.asarray(trk_f.Pos(), np.float32)); rt["tool_quat"].append(np.asarray(trk_f.OriQ(), np.float32))
        rt["door_pos"].append(np.asarray(trk_d.Pos(), np.float32)); rt["door_quat"].append(np.asarray(trk_d.OriQ(), np.float32))
        rt["door_deg"].append(float(state["q"]))
        rt["next"] = (math.floor(sim_t / P["render_timeline_dt_s"] + 1e-9) + 1) * P["render_timeline_dt_s"]
    if rt is not None:
        rt_record(0.0, 0, -1)                                    # t=0 = Initialize 직후(더미 npz 그대로)

    def set_fams(arm, door):
        if arm != state["arm"]:
            trk_f.SetFamily(FAM_F[arm]); state["arm"] = arm
        if door != state["door"]:
            trk_d.SetFamily(FAM_D[door]); state["door"] = door
    PH = {"settle": 0, "descend": 1, "close": 2, "lift": 3, "reclose": 4}

    def door_angle_from_quat():
        """솔버가 실제로 돌린 각 → q (독립 검산). OriQ 는 (x,y,z,w), 회전축 ∥ w_vec."""
        qx, qy, qz, qw = trk_d.OriQ()
        ang = 2.0 * math.degrees(math.atan2(math.hypot(qx, qy, qz), qw))
        sgn = 1.0 if np.dot([qx, qy, qz], w_vec) >= 0 else -1.0
        return q_open - sgn * ang

    # ── W10 진단 v2 (params 게이트; 전부 None 이면 아래 블록은 쓰이지 않는다) ─────────────────────────
    diag2 = bool(P.get("diag_fine_sync_s") or P.get("diag_event_v_m_s") or P.get("diag_event_force_N") or P.get("diag_stop_v_m_s"))
    q_diag = float(P.get("diag_flush_below_q_deg") or 0.0)
    dt_fine = float(P["diag_fine_sync_s"]) if P.get("diag_fine_sync_s") else None
    grp_f = grp_d = None; ring = None; ev = {"v": False, "force": False}
    if diag2:
        from collections import deque
        from scipy.spatial.transform import Rotation as _Rot
        gf, gd = [], []; half_bowl(P, -1, groups=gf); half_bowl(P, +1, groups=gd)
        grp_f, grp_d = np.asarray(gf), np.asarray(gd)
        if len(grp_f) != len(fixed_m.faces) or len(grp_d) != len(door_m.faces):
            raise RuntimeError(f"삼각형 그룹 수 불일치 {len(grp_f)}/{len(fixed_m.faces)} {len(grp_d)}/{len(door_m.faces)}")
        ring = deque(maxlen=int(P.get("diag_ring_syncs") or 12))
        E_star = 1.0 / ((1 - P["nu"] ** 2) / P["E_pa"] + (1 - P["nu"] ** 2) / (P["E_mesh_pa"] or P["E_pa"]))
        print(f"진단 v2: fine sync {dt_fine} s (q<{q_diag}°) · 이벤트 v>{P.get('diag_event_v_m_s')} m/s / F≥{P.get('diag_event_force_N')} N · pop-stop {P.get('diag_stop_v_m_s')} m/s "
              f"· 그룹 {dict(zip(*np.unique(grp_f, return_counts=True)))} · E* {E_star:.3e}", flush=True)

    def tool_world(trk, base):
        """툴 셸을 트래커 자세로 세계에 놓는다(OBJ 정점 = owner 원점 기준 로컬 → pos + R·v; 노드 순서에 의존하지 않음)."""
        R = _Rot.from_quat(np.asarray(trk.OriQ(), float)).as_matrix(); p = np.asarray(trk.Pos(), float)
        return trimesh.Trimesh((R @ np.asarray(base.vertices, float).T).T + p, base.faces, process=False)

    def sphere_vs_mesh(c, r, mesh, grp):
        """구(중심 c, 반경 r) ↔ 셸 최근접 삼각형: 그룹·면 부호거리 h(+ = 바깥)·투영-안 여부·기하 관입(r − 최근접거리)·유령(h<0 ∧ 안 ∧ |h|<r).
        DEME triangle_sphere_CD 규칙(W3b §1): 투영이 삼각형 안이면 관입 = r − h (h<0 이면 r+|h|), 아니면 r − 최근접점 거리."""
        cp, dist, tid = trimesh.proximity.closest_point(mesh, np.asarray(c, float)[None])
        tid = int(tid[0]); n = mesh.face_normals[tid]; v0 = mesh.vertices[mesh.faces[tid][0]]
        h = float(np.dot(c - v0, n)); inside = bool(np.linalg.norm((c - h * n) - cp[0]) < 1e-7); d = float(dist[0])
        return {"tri": tid, "group": str(grp[tid]), "h_mm": round(h * 1000, 4), "inside": inside, "d_closest_mm": round(d * 1000, 4),
                "pen_geo_mm": round((r - d) * 1000, 4), "deme_pen_mm": round(((r - h) if inside else (r - d)) * 1000, 4),
                "ghost": bool(inside and h < 0 and abs(h) < r), "normal": [round(float(x), 4) for x in n]}

    def diag_analyse(phase, row, vv, vn, con):
        """한 sync: 립 근방 클럼프 전개 → 메시별 최대 접촉의 최근접 구·삼각형 기하 → 행에 기록, 링버퍼 push, pop 이벤트 트리거."""
        pp = np.asarray(s.GetOwnerPosition(0, n_p), float); oq = None if tpl is None else np.asarray(s.GetOwnerOriQ(0, n_p), float)
        mf_w, md_w = tool_world(trk_f, fixed_m), tool_world(trk_d, door_m)
        lips = np.vstack([mf_w.vertices[lip_f], md_w.vertices[lip_d]])
        near = np.where(cKDTree(lips).query(pp)[0] < P["diag_near_lip_mm"] / 1000.0 + rad)[0]
        sp_n, sr_n = spheres(pp[near], None if oq is None else oq[near])
        owner_of = np.repeat(near, k_sph); k_of = np.tile(np.arange(k_sph), len(near))
        tree = cKDTree(sp_n) if len(sp_n) else None
        dg = {"n_near_clumps": int(len(near)), "v_max_owner": int(np.argmax(vn)), "v_max_owner_v_m_s": round(float(vn.max()), 4)}
        for key, mesh, grp in (("fixed", mf_w, grp_f), ("door", md_w, grp_d)):
            Pp, Ff = con.get(key, (None, None))
            if Pp is None or not len(Pp) or tree is None:
                continue
            j = int(np.argmax(np.linalg.norm(Ff, axis=1))); p, f = Pp[j], Ff[j]; fm = float(np.linalg.norm(f))
            d, si = tree.query(p); si = int(si)
            g = sphere_vs_mesh(sp_n[si], float(sr_n[si]), mesh, grp)
            g.update(pt_mm=(p * 1000).round(3).tolist(), F_N=round(fm, 4), F_vec_N=f.round(4).tolist(), owner=int(owner_of[si]), k=int(k_of[si]),
                     sphere_r_mm=round(float(sr_n[si]) * 1000, 4), pt_to_sphere_centre_mm=round(float(d) * 1000, 4),
                     owner_v_m_s=round(float(np.linalg.norm(vv[owner_of[si]])), 4),
                     delta_hertz_mm=round((3 * fm / (4 * E_star * math.sqrt(float(sr_n[si])))) ** (2 / 3) * 1000, 4))
            dg[f"top_{key}"] = g
        row["diag"] = dg
        ring.append({"t": row["sim_t"], "q": float(state["q"]), "phase": phase, "i": row["i"], "lip": np.asarray(trk_f.Pos(), float),
                     "near": near, "pos": pp[near], "quat": (np.zeros((len(near), 4)) if oq is None else oq[near]), "vel": vv[near],
                     "nodes_F": np.asarray(mf_w.vertices, np.float32), "nodes_D": np.asarray(md_w.vertices, np.float32),
                     "con": {k: (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32)) for k, v in con.items() if v[0] is not None and len(v[0])},
                     "v_max_owner": int(np.argmax(vn)), "v_max": float(vn.max()), "f1": row["max_single_contact_N"]})
        if P.get("diag_event_v_m_s") and vn.max() > P["diag_event_v_m_s"] and not ev["v"]:
            ev["v"] = True; diag_event(phase, row, int(np.argmax(vn)), pp, oq, vv, "v")            # pop: culprit = 최대속도 owner
        elif P.get("diag_event_force_N") and row["max_single_contact_N"] >= P["diag_event_force_N"] and not ev["force"]:
            tops = [dg[k] for k in ("top_fixed", "top_door") if k in dg]
            if tops:
                ev["force"] = True; diag_event(phase, row, int(max(tops, key=lambda g: g["F_N"])["owner"]), pp, oq, vv, "force")   # 힘: culprit = 최대 접촉의 구 owner

    def diag_event(phase, row, culprit, pp, oq, vv, kind):
        """pop 이벤트 덤프: 링버퍼(마지막 4 sync = 직전 3 + 트리거) 에서 culprit 의 위치·속도·구별 최근접 삼각형 기하·툴 접촉력, 기작 분류(GATES_w10 G1), GetContactDetailedInfo(try)."""
        t0 = time.time(); R = list(ring); last = R[-4:]; table = []; suffix = "" if kind == "v" else f"_{kind}"
        for e in last:
            k = np.where(e["near"] == culprit)[0]
            entry = {"sim_t": e["t"], "q_deg": round(e["q"], 4), "phase": e["phase"], "i": e["i"], "pop_sync": e is R[-1], "in_ring_near_set": bool(len(k))}
            if len(k):
                k = int(k[0]); c = e["pos"][k]; sp_c, sr_c = spheres(e["pos"][k:k + 1], None if tpl is None else e["quat"][k:k + 1])
                mf_w = trimesh.Trimesh(e["nodes_F"].astype(float), fixed_m.faces, process=False); md_w = trimesh.Trimesh(e["nodes_D"].astype(float), door_m.faces, process=False)
                v = e["vel"][k]
                entry.update(centre_mm=(c * 1000).round(3).tolist(), centre_rel_lip_mm=((c - e["lip"]) * 1000).round(3).tolist(),
                             v_m_s=round(float(np.linalg.norm(v)), 4), vel_m_s=v.round(4).tolist(), vs_mesh={})
                for key, mesh, grp in (("fixed", mf_w, grp_f), ("door", md_w, grp_d)):
                    per = [dict(sphere_vs_mesh(sp_c[kk], float(sr_c[kk]), mesh, grp), k=kk, r_mm=round(float(sr_c[kk]) * 1000, 4)) for kk in range(k_sph)]
                    best = max(per, key=lambda g: (g["ghost"], g["pen_geo_mm"]))
                    Pp, Ff = e["con"].get(key, (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)))
                    if len(Pp):
                        m = cKDTree(sp_c).query(Pp.astype(float))[0] <= float(sr_c.max()) + 0.0005
                        fmag = np.linalg.norm(Ff[m].astype(float), axis=1) if m.any() else np.zeros(0)
                        best["contacts_on_culprit"] = {"n": int(m.sum()), "F_sum_N": round(float(np.linalg.norm(Ff[m].astype(float).sum(0))), 4) if m.any() else 0.0,
                                                       "F_max_N": round(float(fmag.max()), 4) if m.any() else 0.0,
                                                       "points_mm": (Pp[m].astype(float) * 1000).round(3).tolist()[:8]}
                    else:
                        best["contacts_on_culprit"] = {"n": 0, "F_sum_N": 0.0, "F_max_N": 0.0, "points_mm": []}
                    best["all_spheres"] = [{kk_: g[kk_] for kk_ in ("k", "group", "h_mm", "inside", "pen_geo_mm", "ghost")} for g in per]
                    entry["vs_mesh"][key] = best
            table.append(entry)
        pre = [e for e in table if not e["pop_sync"]]
        ghost = any(e.get("vs_mesh", {}).get(m, {}).get("ghost") for e in pre for m in ("fixed", "door"))
        f_seq = [max(e.get("vs_mesh", {}).get(m, {}).get("contacts_on_culprit", {}).get("F_max_N", 0.0) for m in ("fixed", "door")) for e in pre]
        squeeze = len(f_seq) >= 3 and all(a < b for a, b in zip(f_seq[-3:], f_seq[-2:])) and f_seq[-1] >= 2.0
        detail = None
        try:
            t1 = time.time(); ci = s.GetContactDetailedInfo()
            A, B = np.asarray(ci.GetAOwner()), np.asarray(ci.GetBOwner()); m = (A == culprit) | (B == culprit)
            def _get(fn, dflt):
                try:
                    return np.asarray(fn(), float if dflt is not None else None)
                except Exception as exc_:  # noqa: BLE001 — 필드 미설정이면 기본값
                    return None
            typ = _get(ci.GetContactType, None); F = np.asarray(ci.GetForce(), float); Pt = _get(ci.GetPoint, 0.0); N = _get(ci.GetNormal, 0.0)
            gA, gB = _get(ci.GetAGeo, 0), _get(ci.GetBGeo, 0)
            if typ is None: typ = np.full(len(A), "?")
            if Pt is None: Pt = np.zeros((len(A), 3))
            if N is None: N = np.zeros((len(A), 3))
            if gA is None: gA = np.full(len(A), -1)
            if gB is None: gB = np.full(len(A), -1)
            idx = np.where(m)[0]; idx = idx[np.argsort(-np.linalg.norm(F[idx], axis=1))] if len(idx) else idx
            def grp_of(o, gi):
                if o == owners["fixed"]:
                    return str(grp_f[gi]) if 0 <= gi < len(grp_f) else f"out_of_range({gi})"
                if o == owners["door"]:
                    return (str(grp_d[gi]) if 0 <= gi < len(grp_d) else (str(grp_d[gi - len(grp_f)]) if 0 <= gi - len(grp_f) < len(grp_d) else f"out_of_range({gi})"))
                return None
            detail = {"wall_s": round(time.time() - t1, 2), "n_pairs_total": int(len(A)), "n_pairs_culprit": int(len(idx)),
                      "pairs": [{"type": str(typ[j]), "A": int(A[j]), "B": int(B[j]), "AGeo": int(gA[j]), "BGeo": int(gB[j]),
                                 "B_group_by_geo": grp_of(int(B[j]), int(gB[j])), "A_group_by_geo": grp_of(int(A[j]), int(gA[j])),
                                 "F_N": round(float(np.linalg.norm(F[j])), 4), "force": F[j].round(4).tolist(),
                                 "point_mm": (Pt[j] * 1000).round(3).tolist(), "normal": N[j].round(4).tolist()} for j in idx[:16]],
                      "note": "GetContact-like 는 잠재 쌍(potential pairs) 도 포함 — 힘 0 인 쌍은 접촉 아님"}
            detail["top_is_sphere_sphere"] = bool(detail["pairs"] and detail["pairs"][0]["type"] == "SS" and detail["pairs"][0]["F_N"] > 0)
        except Exception as exc:  # noqa: BLE001 — 대규모에서 불안정할 수 있음(diverge_min 주석)
            detail = {"error": repr(exc)}
        mech = "ghost" if ghost else ("sphere_sphere" if (detail or {}).get("top_is_sphere_sphere") else ("squeeze" if squeeze else "unresolved"))
        now = {"pos_mm": (pp[culprit] * 1000).round(3).tolist(), "vel_m_s": vv[culprit].round(4).tolist(), "v_m_s": round(float(np.linalg.norm(vv[culprit])), 4),
               "quat_xyzw": (None if oq is None else oq[culprit].round(6).tolist()),
               "dist_to_lip_lines_mm": round(float(cKDTree(np.vstack([R[-1]["nodes_F"][lip_f], R[-1]["nodes_D"][lip_d]]).astype(float)).query(pp[culprit])[0]) * 1000, 3)}
        try:                                                     # 지금 자세의 culprit 구별 최근접 삼각형(근방 집합 밖이어도 남긴다)
            sp_c, sr_c = spheres(pp[culprit:culprit + 1], None if oq is None else oq[culprit:culprit + 1])
            mf_w, md_w = trimesh.Trimesh(R[-1]["nodes_F"].astype(float), fixed_m.faces, process=False), trimesh.Trimesh(R[-1]["nodes_D"].astype(float), door_m.faces, process=False)
            now["vs_mesh"] = {key: max([dict(sphere_vs_mesh(sp_c[kk], float(sr_c[kk]), mesh, grp), k=kk) for kk in range(k_sph)], key=lambda g: (g["ghost"], g["pen_geo_mm"]))
                              for key, mesh, grp in (("fixed", mf_w, grp_f), ("door", md_w, grp_d))}
        except Exception as exc:  # noqa: BLE001
            now["vs_mesh_error"] = repr(exc)
        evj = {"artifact": "W10_DIVERGE_EVENT_V1", "tag": tag, "kind": kind,
               "trigger": {"phase": phase, "i": row["i"], "sim_t": row["sim_t"], "q_deg": round(float(state["q"]), 4), "v_max": row["v_particle_max"],
                           "max_single_contact_N": row["max_single_contact_N"], "rule": {"v_m_s": P.get("diag_event_v_m_s"), "force_N": P.get("diag_event_force_N")}},
               "culprit": int(culprit), "culprit_now": now, "culprit_last4": table,
               "mechanism": mech, "mechanism_rule": "GATES_w10.md G1: pre-pop sync 중 culprit 구가 툴 삼각형에 h<0∧inside∧|h|<r → ghost; 상세 최대힘 pair 가 SS → sphere_sphere; 3 sync 연속 상승·≥2 N → squeeze; else unresolved",
               "ghost_any_prepop": ghost, "squeeze_seq_F_max_N": f_seq, "contact_detail": detail, "ring_syncs": len(R), "dt_fine_s": dt_fine,
               "owner_ids": owners, "n_tri": {"fixed": int(len(grp_f)), "door": int(len(grp_d))}, "wall_s": round(time.time() - t0, 2)}
        json.dump(evj, open(out / f"diverge_event{suffix}_{tag}.json", "w"), ensure_ascii=False, indent=1)
        cps = [(kk, e["con"][kk][0], e["con"][kk][1], i_) for i_, e in enumerate(R) for kk in e["con"]]
        np.savez_compressed(out / f"diverge_event{suffix}_{tag}.npz", t_s=np.array([e["t"] for e in R]), q_deg=np.array([e["q"] for e in R]),
                            lip_m=np.stack([e["lip"] for e in R]), nodes_F_m=np.stack([e["nodes_F"] for e in R]), nodes_D_m=np.stack([e["nodes_D"] for e in R]),
                            faces_F=np.asarray(fixed_m.faces), faces_D=np.asarray(door_m.faces), groups_F=grp_f.astype(str), groups_D=grp_d.astype(str),
                            near_ids=np.concatenate([e["near"] for e in R]), near_frame=np.concatenate([np.full(len(e["near"]), i_, np.int32) for i_, e in enumerate(R)]),
                            near_pos_m=np.concatenate([e["pos"] for e in R]), near_quat_xyzw=np.concatenate([e["quat"] for e in R]), near_vel_m_s=np.concatenate([e["vel"] for e in R]),
                            contact_point_m=(np.concatenate([c[1] for c in cps]) if cps else np.zeros((0, 3), np.float32)),
                            contact_force_N=(np.concatenate([c[2] for c in cps]) if cps else np.zeros((0, 3), np.float32)),
                            contact_frame=(np.concatenate([np.full(len(c[1]), c[3], np.int32) for c in cps]) if cps else np.zeros(0, np.int32)),
                            contact_mesh=(np.concatenate([np.full(len(c[1]), 0 if c[0] == "fixed" else 1, np.int8) for c in cps]) if cps else np.zeros(0, np.int8)),
                            culprit=np.int64(culprit), template_json=json.dumps(tpl) if tpl is not None else "", metadata_json=json.dumps(
                                {"artifact": "W10_DIVERGE_EVENT_RING_V1", "frames": "링버퍼 sync 순(마지막 = 트리거 sync)", "near_*": "립선 diag_near_lip_mm 안 클럼프(near_frame 로 프레임 구분)",
                                 "contact_mesh": "0 = fixed, 1 = door", "groups": "셸 삼각형 그룹 태그(half_bowl 생성 순)"}, ensure_ascii=False))
        print(f"  🔴 이벤트({kind}) [{phase}] i={row['i']} q={state['q']:.3f}° v_max={row['v_particle_max']:.2f} m/s 단일={row['max_single_contact_N']:.2f} N "
              f"culprit={culprit} 기작={mech} ghost_prepop={ghost} → diverge_event{suffix}_{tag}.* ({time.time() - t0:.1f}s)", flush=True)

    def sample(phase, i):
        row = {"phase": phase, "i": i, "sim_t": round(float(s.GetSimTime()), 6),
               "z_lip_mm": round(float(trk_f.Pos()[2]) * 1000, 3), "q_deg": round(state["q"], 3),
               "arm": state["arm"], "door": state["door"]}
        vv = np.asarray(s.GetOwnerVelocity(0, n_p), float); vn = np.linalg.norm(vv, axis=1); vmax = float(vn.max())
        row["v_particle_max"] = round(vmax, 4); state["v_max"] = max(state["v_max"], vmax)
        diag = bool(P.get("diag_flush_below_q_deg")) and state["q"] < P["diag_flush_below_q_deg"] and phase in ("close", "reclose")
        if diag:
            vi = int(np.argmax(vn)); row["v_max_owner"] = vi
            row["v_max_owner_pos_mm"] = (np.asarray(s.GetOwnerPosition(vi, 1), float)[0] * 1000).round(2).tolist()
        state["pop_steps"] += int(vmax > P["pop_speed_m_s"])
        hw = hinge_w(); M_res, f1 = 0.0, 0.0; con = {}
        for key, trk in (("fixed", trk_f), ("door", trk_d)):
            pts, frcs = trk.GetContactForces()
            row[f"n_{key}"] = len(pts)
            if len(pts):
                Pp, Ff = np.asarray(pts, float), np.asarray(frcs, float); con[key] = (Pp, Ff)
                row[f"F_{key}_N"] = round(float(np.linalg.norm(Ff.sum(0))), 4)
                if key == "fixed":
                    row["Fz_fixed_up_N"] = round(float(Ff[:, 2].sum()), 4)     # 더미가 팔을 미는 위쪽 힘
                f1 = max(f1, float(np.linalg.norm(Ff, axis=1).max()))
                if key == "door":
                    M_res = float(np.dot(np.cross(Pp - hw, Ff).sum(0), axis_w))  # >0 = 닫힘 저항
                if diag:
                    j = int(np.argmax(np.linalg.norm(Ff, axis=1)))
                    row[f"maxF_{key}_pt_mm"] = (Pp[j] * 1000).round(2).tolist(); row[f"maxF_{key}_vec_N"] = Ff[j].round(3).tolist()
                rec["cp"].append(Pp.astype(np.float32)); rec["cf"].append(Ff.astype(np.float32))
                rec["cframe"].append(np.full(len(pts), len(rec["t"]), np.int32))
            else:
                row[f"F_{key}_N"] = 0.0
        row.setdefault("Fz_fixed_up_N", 0.0)
        row["M_hinge_res_Nm"] = round(M_res, 6); row["lipF_N"] = round(abs(M_res) / R_lip, 4)
        row["max_single_contact_N"] = round(f1, 4)
        rec["t"].append(row["sim_t"]); rec["phase"].append(PH[phase])
        rec["nodes_F"].append(np.asarray(trk_f.GetMeshNodesGlobal(), np.float32))
        rec["nodes_D"].append(np.asarray(trk_d.GetMeshNodesGlobal(), np.float32))
        log.append(row)
        if diag and diag2:
            diag_analyse(phase, row, vv, vn, con)                 # W10: 최근접 구·삼각형 기하 + 링버퍼 + pop 이벤트
        if diag:
            flush(phase)                                          # 발산 직전 행까지 남기기 위해 매 sync 저장
        if diag2 and P.get("diag_stop_v_m_s") and vmax > P["diag_stop_v_m_s"]:
            flush(phase); raise RuntimeError(f"pop-stop: 입자 최대속도 {vmax:.1f} m/s > {P['diag_stop_v_m_s']} (owner {int(np.argmax(vn))}, {phase} i={i})")
        if rt is not None and row["sim_t"] >= rt["next"] - 1e-9:
            rt_record(row["sim_t"], PH[phase], len(log) - 1)
        return row

    tl_path = out / f"timeline_{tag}.json"
    flush = lambda st: json.dump({"state": st, "rows": log}, open(tl_path, "w"), ensure_ascii=False)

    def door_step(phase, i, dts=None):
        """문이 family 11 이면 q 를 한 스텝 진행시키고 정지 조건(도달/서보 실속)을 판정한다. dts = 이 sync 의 길이(None = dt_c)."""
        r = sample(phase, i)
        if state["door"] != "close":
            return r
        state["q"] = max(q_end, state["q"] - math.degrees(w_close) * (dt_c if dts is None else dts))
        r["q_deg"] = round(state["q"], 3)
        reason = None
        if state["q"] <= q_end + 1e-9:
            reason = "reached_close_end"
        elif P.get("door_min_q_deg") is not None and state["q"] <= P["door_min_q_deg"] + 1e-9:
            reason = "door_floor"
        elif P["servo_stall_model"] and r["M_hinge_res_Nm"] >= M_stall:
            reason = "servo_stall"
        elif P.get("door_pinch_guard_N") and r["max_single_contact_N"] >= P["door_pinch_guard_N"]:
            reason = "pinch_guard"
        if reason:
            set_fams("hold", "hold")
            state["stop"] = {"phase": phase, "q_deg": round(state["q"], 3), "reason": reason,
                             "M_hinge_res_Nm": r["M_hinge_res_Nm"], "sim_t": r["sim_t"], "max_single_contact_N": r["max_single_contact_N"]}
            print(f"  문 정지 [{phase}] q={state['q']:.2f}° ({reason}, M={r['M_hinge_res_Nm']:.3f} N·m)",
                  flush=True)
        return r

    diverged, stops, hm_pre = False, [], None
    z_target = None
    try:
        for i in range(P["settle_steps"]):
            s.DoDynamicsThenSync(dt); sample("settle", i)
        pp0 = np.asarray(s.GetOwnerPosition(0, n_p), float)     # 재안착 뒤 펠릿면으로 잠김 목표를 잡는다
        oq0 = None if tpl is None else np.asarray(s.GetOwnerOriQ(0, n_p), float)
        if tpl is None:
            z_surf = float(pp0[near, 2].max()) + rad
        else:
            z_surf = surface_z(*spheres(pp0, oq0), x_s, y_s, P)
        hm_pre = heightmap_from_particles(*spheres(pp0, oq0), spec).height     # 퍼내기 전 heightmap(재안착 뒤·하강 전)
        z_target = z_surf - P["plunge_mm"] / 1000.0
        print(f"  재안착 뒤 펠릿면 {z_surf*1000:.1f} mm → 잠김 목표 립 z {z_target*1000:.1f} mm · settle 벽시계 {time.time()-t_start:.0f} s", flush=True)
        set_fams("desc", "desc")
        for i in range(int(P["descend_max_steps_factor"] * n_desc)):
            s.DoDynamicsThenSync(dt_d); r = sample("descend", i)
            if i % 40 == 0:
                flush("descend")
            if float(trk_f.Pos()[2]) <= z_target + 1e-6:
                break
            hold = r["Fz_fixed_up_N"] > P["arm_force_max_N"]
            state["holds"] += int(hold); set_fams(*(("hold", "hold") if hold else ("desc", "desc")))
            if i % 80 == 0:
                print(f"  하강 {i:3d} z_lip={r['z_lip_mm']:7.2f} {state['arm']:4s} 접촉 F/D={r['n_fixed']}/{r['n_door']} "
                      f"Fz={r['Fz_fixed_up_N']:.3f} N 단일최대={r['max_single_contact_N']:.3f} v_max={r['v_particle_max']:.2f} w={time.time()-t_start:.0f}s", flush=True)
        z_reached = float(trk_f.Pos()[2])
        set_fams("hold", "close")
        fine_on = lambda: dt_fine is not None and state["q"] < q_diag          # W10: q<q_diag 구간은 잘게 sync(문 각속도 불변)
        n_fine_extra = 0 if dt_fine is None else int(math.ceil(q_diag / P["close_deg_s"] / dt_fine)) + 50
        for i in range(n_close_sync + n_fine_extra):
            dts = dt_fine if fine_on() else dt_c
            s.DoDynamicsThenSync(dts); r = door_step("close", i, dts)
            if i % 10 == 0:
                flush("close")
                print(f"  폐합 {i:3d} q={r['q_deg']:6.2f}° 접촉 D={r['n_door']:4d} M_res={r['M_hinge_res_Nm']:8.4f} "
                      f"립등가={r['lipF_N']:7.3f} N 단일최대={r['max_single_contact_N']:.3f} v_max={r['v_particle_max']:.2f} w={time.time()-t_start:.0f}s", flush=True)
            if state["door"] != "close":
                break
        if state["door"] == "close":                 # 스텝 수 소진(도달 판정 전) — 안전망
            set_fams("hold", "hold")
            state["stop"] = {"phase": "close", "q_deg": round(state["q"], 3), "reason": "steps_exhausted"}
        stops.append(dict(state["stop"], q_from_quat_deg=round(door_angle_from_quat(), 3)))
        q_after_close = state["q"]
        set_fams("lift", "lift")
        for i in range(n_lift):
            s.DoDynamicsThenSync(dt); r = sample("lift", i); flush("lift")
            if i % 40 == 0:
                print(f"  상승 {i:3d} z_lip={r['z_lip_mm']:7.2f} 접촉 F/D={r['n_fixed']}/{r['n_door']} v_max={r['v_particle_max']:.2f} w={time.time()-t_start:.0f}s", flush=True)
        set_fams("hold", "hold")
        n_reclose = 0
        if state["q"] > q_end + 1e-9:
            set_fams("hold", "close"); state["stop"] = None
            t_rc0, t_rc_budget = float(s.GetSimTime()), n_reclose_max * dt_c      # W10 fine sync 시 재폐합 예산은 시간으로(원래 = 스텝 수 × dt_c)
            for i in range(n_reclose_max + n_fine_extra):
                dts = dt_fine if fine_on() else dt_c
                s.DoDynamicsThenSync(dts); door_step("reclose", i, dts); n_reclose += 1
                if state["door"] != "close":
                    break
                if dt_fine is not None and float(s.GetSimTime()) - t_rc0 >= t_rc_budget - 1e-9:
                    break
            if state["door"] == "close":
                set_fams("hold", "hold")
                state["stop"] = {"phase": "reclose", "q_deg": round(state["q"], 3), "reason": "steps_exhausted"}
            stops.append(dict(state["stop"], q_from_quat_deg=round(door_angle_from_quat(), 3)))
        for i in range(3):                           # 정착 3 스텝 뒤 최종 판독
            s.DoDynamicsThenSync(dt); sample("lift", n_lift + i)
    except Exception as exc:                          # noqa: BLE001 — 발산도 산출은 남긴다
        diverged = True; print(f"🔴 발산/중단: {exc}", flush=True); flush("diverged")
        q_after_close, n_reclose, z_reached = state["q"], 0, float(trk_f.Pos()[2])
    wall = time.time() - t_start
    if rt is not None and float(s.GetSimTime()) > rt["t"][-1] + 1e-9:
        rt_record(float(s.GetSimTime()), PH["lift"], len(log) - 1)      # 최종 상태 프레임(포획 판정 시점) 을 반드시 남긴다

    # ── 최종 판독: 포획(툴 프레임 보울 원통 안) · 물림 · 입 ──────────────────
    pp = np.asarray(s.GetOwnerPosition(0, n_p), float)
    if len(pp) != n_p:
        raise RuntimeError(f"입자 수 불일치 {len(pp)} != {n_p}")
    oq = None if tpl is None else np.asarray(s.GetOwnerOriQ(0, n_p), float)
    lip_w = np.asarray(trk_f.Pos(), float)
    L5 = np.array(P["lip_l5_mm"]) / 1000.0; C5 = np.array(P["bowl_center_l5_mm"]) / 1000.0
    p5 = (R_W.T @ (pp - (lip_w - R_W @ L5)).T).T          # 세계 → link5 (m)
    in_cav = (np.hypot(p5[:, 0] - C5[0], p5[:, 2] - C5[2]) < P["bowl_r_in_mm"] / 1000.0) & \
             (np.abs(p5[:, 1]) < P["cheek_half_y_mm"] / 1000.0)
    carried = pp[:, 2] > z_surf + P["lift_mm"] / 2000.0
    touch_both = np.array(sorted(set(trk_f.GetContactClumps()) & set(trk_d.GetContactClumps())), int)
    touch_both = touch_both[touch_both < n_p] if len(touch_both) else touch_both
    pinched = int(((p5[touch_both, 2] > L5[2] - 1.5 * 2 * rad).sum()) if len(touch_both) else 0)
    nodes_F, nodes_D = rec["nodes_F"][-1].astype(float), rec["nodes_D"][-1].astype(float)
    mouth_end = float(cKDTree(nodes_F[lip_f]).query(nodes_D[lip_d])[0].min()) * 1000
    q_final = state["q"]; gap_mm = R_lip * 1000 * math.radians(q_final)

    rest = ~carried
    if tpl is None:
        hm = heightmap_from_particles(pp[rest], np.full(int(rest.sum()), rad), spec).height
    else:
        hm = heightmap_from_particles(*spheres(pp[rest], oq[rest]), spec).height
    hm_npz = heightmap_from_particles(np.asarray(z["positions_m"], float), np.asarray(z["radii_m"], float), spec).height
    if hm_pre is None:                                   # settle 중 발산 시 안전망
        hm_pre = hm_npz
    sp_final, sr_final = spheres(pp, oq)
    crater = crater_angles(hm_pre, hm, spec, (x_s, y_s), P)
    print("  구덩이 옆면 각(°) " + " ".join(f"{k}:{v['angle_deg']:.1f}" if v.get("angle_deg") is not None else f"{k}:n/a"
                                        for k, v in crater["azimuths"].items()), flush=True)

    clos = [r for r in log if r["phase"] in ("close", "reclose")] or [{"lipF_N": 0.0, "M_hinge_res_Nm": 0.0,
                                                                        "n_door": 0, "max_single_contact_N": 0.0}]
    cavity_cm3 = design["derived"]["cavity_cm3"]
    mass_g = int(in_cav.sum()) * m_p * 1000
    res = {
        "artifact": "DEME_SCOOP_S1_V1", "tag": tag, "seed": a.seed, "smoke": bool(a.smoke),
        "inputs_sha16": {str(k): v for k, v in {
            a.pile: sha16(a.pile), str(FIXED_STL): sha16(FIXED_STL), str(DOOR_STL): sha16(DOOR_STL),
            str(DESIGN): sha16(DESIGN), str(Path(__file__)): sha16(__file__),
            **({a.params: sha16(a.params)} if a.params else {})}.items()},
        "params": P, "particle": dict(shape_info, mass_kg=m_p, n=n_p), "mesh_check": mesh_check,
        "engine": {"DEME": "2.4.0", "force_model": "UseFrictionalHertzianModel", "owner_ids": owners},
        "scoop_site": {"x_mm": x_s * 1000, "y_mm": round(y_s * 1000, 2), "surface_z_mm": round(z_surf * 1000, 2),
                       "lip_z_start_mm": round(z_lip0 * 1000, 2), "lip_z_plunge_nominal_mm": round(z_lip1 * 1000, 2),
                       "lip_z_end_mm": round(float(lip_w[2]) * 1000, 2)},
        "trajectory": {"q_open_joint_deg": q_open, "q_close_end_deg": q_end, "steps_nominal": {
            "settle": P["settle_steps"], "descend": n_desc, "close": n_close, "lift": n_lift},
            "steps_actual": {k: sum(1 for r in log if r["phase"] == k) for k in ("settle", "descend", "close", "lift", "reclose")},
            "dt_sync_close_s": dt_c, "steps_nominal_close_sync": n_close_sync,
            "descend_hold_steps": state["holds"], "descend_reached_lip_z_mm": round(z_reached * 1000, 2),
            "descend_target_lip_z_mm": None if z_target is None else round(z_target * 1000, 2),
            "plunge_reached_mm": None if z_target is None else round((z_surf - z_reached) * 1000, 2),
            "vz_descend_m_s": vz_desc, "omega_close_rad_s": w_close, "vz_lift_m_s": vz_lift,
            "mouth_start_mm": round(mouth0, 2)},
        "pops": {"v_particle_max_m_s": round(state["v_max"], 3), "steps_over_pop_speed": state["pop_steps"],
                 "pop_speed_m_s": P["pop_speed_m_s"], "note": "무른 구(E 5e6)의 탄성 방출 인공물. 5 m/s 초과 스텝 수가 0 이 아니면 정량값에 인공물이 섞여 있다"},
        "servo": {"M_stall_Nm": M_stall, "lip_radius_m": R_lip, "lipF_stall_N": round(M_stall / R_lip, 3)},
        "diverged": diverged, "steps_completed": len(log),
        "door": {"stops": stops, "q_after_close_deg": round(q_after_close, 3), "q_final_deg": round(q_final, 3),
                 "servo_deg_final": round(q_final + P["servo_zero_offset_deg"], 3),
                 "lip_gap_final_mm": round(gap_mm, 3), "mouth_end_mm": round(mouth_end, 3),
                 "n_touch_both_meshes": int(len(touch_both)), "n_pinched_at_lip": pinched},
        "forces": {"close_peak_lipF_N": round(max(r["lipF_N"] for r in clos), 4),
                   "close_peak_M_res_Nm": round(max(r["M_hinge_res_Nm"] for r in clos), 6),
                   "close_peak_single_contact_N": round(max(r["max_single_contact_N"] for r in clos), 4),
                   "close_contacts_door_max": max(r["n_door"] for r in clos),
                   "descend_peak_F_fixed_N": round(max([r["F_fixed_N"] for r in log if r["phase"] == "descend"] or [0.0]), 4)},
        "capture": {"n_in_cavity": int(in_cav.sum()), "n_carried_z": int(carried.sum()),
                    "mass_g": round(mass_g, 4), "cavity_cm3": cavity_cm3,
                    "fill_vs_bulk": round(mass_g / (cavity_cm3 * P["bulk_density_g_cm3"]), 4),
                    "design_load_g": design["derived"]["load_per_scoop_g"], "design_fill_factor": design["params"]["fill_factor"]},
        "heightmap": {"spec_version": "roarm-heightmap-v1", "cell_m": cell, "shape": list(hm.shape),
                      "max_m": float(np.nanmax(hm)), "n_particles_used": int(rest.sum()),
                      "pre_max_m": float(np.nanmax(hm_pre)), "npz_max_m": float(np.nanmax(hm_npz)),
                      "pre_vs_npz_max_abs_diff_m": float(np.abs(hm_pre - hm_npz).max()),
                      "removed_volume_cm3": crater["removed_volume_cm3"], "spheres_per_particle": k_sph},
        "crater": crater,
        "wall_seconds": round(wall, 2),
        "non_claims": ["물성(E 5e6·mu 0.5·Crr 0.05·CoR 0.3·밀도 950) 전부 임시값 — 반력·정지각·포획량 절대값 인용 금지",
                       "더미는 E 1e7 로 정착, 본 실행 E 5e6 → t=0 미세 재안착",
                       "문 개방을 펠릿면에서 하지 않고 이미 열린 채 접근(실물은 펠릿면에서 열림) — 문 립은 열릴 때 위로만 움직여 더미와 접촉 없음",
                       "D341 Rerun 완결 계약 미이행(원자료 npz 만 저장)"],
    }
    cat = lambda k, shp: (np.concatenate(rec[k]) if rec[k] else np.zeros(shp, np.float32))
    np.savez_compressed(out / f"scoop_s1_{tag}.npz", positions_m=pp, radii_m=np.full(n_p, rad),
                        in_cavity=in_cav, carried=carried, heightmap_m=hm, box_bounds_m=box,
                        heightmap_pre_m=hm_pre, heightmap_npz_m=hm_npz, sphere_positions_m=sp_final, sphere_radii_m=sr_final,
                        clump_quaternions_xyzw=(np.zeros((0, 4)) if oq is None else oq),
                        frame_t_s=np.asarray(rec["t"]), frame_phase=np.asarray(rec["phase"], np.int8),
                        nodes_F_m=np.asarray(rec["nodes_F"], np.float32), nodes_D_m=np.asarray(rec["nodes_D"], np.float32),
                        contact_point_m=cat("cp", (0, 3)), contact_force_N=cat("cf", (0, 3)),
                        contact_frame=(np.concatenate(rec["cframe"]) if rec["cframe"] else np.zeros(0, np.int32)),
                        lip_idx_f=lip_f, lip_idx_d=lip_d)
    if rt is not None:
        rp = (REPO / P["render_timeline_path"]) if P.get("render_timeline_path") else out / f"render_timeline_{tag}.npz"
        rt_meta = {"artifact": "DEME_SCOOP_S1_W8_RENDER_TIMELINE_V1", "purpose": "Isaac Sim 재생·렌더용 — 물리 판정과 무관(정본은 scoop_s1_*.json/npz)",
                   "cell_tag": tag, "params_file": a.params, "pile_npz": a.pile, "pile_sha16": sha16(a.pile), "dt_s": P["render_timeline_dt_s"],
                   "world_frame": "DEME 세계 = 더미 npz 프레임: 상자 바닥 중심 원점, z 위, 단위 m",
                   "clump_pos_m": "[T,N,3] float32 클럼프(알) 중심 = DEME owner 위치. 구 전개 p_sphere = p + R(q)·offset, offset·반경 = 더미 npz clump_template_json(offsets_m, sphere_radii_m); 구 더미면 N = 구",
                   "clump_quat_xyzw": "[T,N,4] float32 DEME OriQ, xyzw",
                   "tool_pos_m": "[T,3] 고정부(보울) owner 원점 = 두꺼워진 립점(lip_l5_mm 의 세계 위치). 고정부 셸 OBJ(_obj/fixed_*.obj) 정점은 이 원점 기준 세계 프레임(m) — 규정 병진만 하므로 tool_quat 는 단위",
                   "tool_quat_xyzw": "[T,4] 고정부 owner OriQ xyzw",
                   "door_pos_m": "[T,3] 문 owner 원점 = 힌지점 세계 위치", "door_quat_xyzw": "[T,4] 문 owner OriQ xyzw — 문 OBJ(_obj/door_open*.obj, q_open 자세로 구운 것) 기준 추가 회전",
                   "door_deg": "[T] 문 관절각 q(°, + = 열림). 서보각 = q + servo_zero_offset_deg. door_step 은 sample 뒤에 q 를 갱신하므로 한 sync(4 ms) 지연",
                   "link5_to_world_R_W_columns_are_link5_axes": R_W.tolist(), "lip_l5_mm": P["lip_l5_mm"], "hinge_l5_mm": P["hinge_l5_mm"], "bowl_center_l5_mm": P["bowl_center_l5_mm"],
                   "door_axis_world": axis_w.tolist(), "q_open_joint_deg": q_open, "servo_zero_offset_deg": P["servo_zero_offset_deg"], "phase_ids": PH,
                   "captured_ids": "최종 프레임에서 보울 공동 안(포획 판정) 클럼프 id", "carried_ids": "최종 z 기준 딸려 올라간 클럼프 id",
                   "timeline_frame": "scoop_s1 timeline/npz 프레임 번호(-1 = Initialize 직후)"}
        np.savez_compressed(rp, t_s=np.asarray(rt["t"], np.float64), clump_pos_m=np.stack(rt["pos"]), clump_quat_xyzw=np.stack(rt["quat"]),
                            tool_pos_m=np.stack(rt["tool_pos"]), tool_quat_xyzw=np.stack(rt["tool_quat"]), door_pos_m=np.stack(rt["door_pos"]),
                            door_quat_xyzw=np.stack(rt["door_quat"]), door_deg=np.asarray(rt["door_deg"], np.float32), phase=np.asarray(rt["phase"], np.int8),
                            timeline_frame=np.asarray(rt["frame"], np.int32), captured_ids=np.where(in_cav)[0].astype(np.int32),
                            carried_ids=np.where(carried)[0].astype(np.int32), metadata_json=json.dumps(rt_meta, ensure_ascii=False))
        res["render_timeline"] = {"path": str(rp), "n_frames": len(rt["t"]), "dt_s": P["render_timeline_dt_s"], "bytes": int(rp.stat().st_size), "n_captured_ids": int(in_cav.sum())}
        print(f"  렌더 타임라인 {len(rt['t'])} 프레임 → {rp} ({rp.stat().st_size/1e6:.1f} MB)", flush=True)
    json.dump(res, open(out / f"scoop_s1_{tag}.json", "w"), ensure_ascii=False, indent=2)
    flush("diverged" if diverged else "complete")
    plot_heightmaps(out, tag, spec, hm_pre, hm, crater, (x_s, y_s))
    snapshot(out, tag, res, rec, sp_final, rad, np.repeat(in_cav, k_sph), x_s)
    print(f"\n포획 {res['capture']['n_in_cavity']} 개 = {mass_g:.2f} g (충전율/벌크 {res['capture']['fill_vs_bulk']:.3f}) "
          f"· 딸려 올라감(z) {int(carried.sum())} · 문 최종 q {q_final:.2f}° (서보 {q_final + P['servo_zero_offset_deg']:.2f}°) "
          f"· 립 틈 {gap_mm:.2f} mm · 입 {mouth_end:.2f} mm · 물림 {pinched} · 립등가 피크 {res['forces']['close_peak_lipF_N']:.3f} N "
          f"· 벽시계 {wall:.1f} s · 발산 {diverged}\n-> {out}")


def snapshot(out, tag, res, rec, pp, rad, in_cav, x_s):
    """결정 시점 진단(D324): 폐합 종료·최종 프레임의 YZ 단면(툴 중심 |x−x_s|<20 mm)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ph = np.asarray(rec["phase"]); fr = [int(np.where(ph == 2)[0][-1]) if (ph == 2).any() else 0, len(ph) - 1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    sl = np.abs(pp[:, 0] - x_s) < 0.02
    for k, f in enumerate(fr):
        nf, nd = rec["nodes_F"][f], rec["nodes_D"][f]
        if k == 1:
            ax[k].scatter(pp[sl & ~in_cav, 1] * 1000, pp[sl & ~in_cav, 2] * 1000, s=4, c="0.6", label="입자")
            ax[k].scatter(pp[sl & in_cav, 1] * 1000, pp[sl & in_cav, 2] * 1000, s=6, c="tab:orange", label="보울 안")
        ax[k].scatter(nf[:, 1] * 1000, nf[:, 2] * 1000, s=1, c="tab:green", label="고정부")
        ax[k].scatter(nd[:, 1] * 1000, nd[:, 2] * 1000, s=1, c="tab:blue", label="문")
        cp = rec["cp"]; cfr = rec["cframe"]
        if cp:
            P_, F_ = np.concatenate(cp), np.concatenate(cfr)
            m = F_ == f; ax[k].scatter(P_[m, 1] * 1000, P_[m, 2] * 1000, s=8, c="red", label=f"접촉 {m.sum()}")
        ax[k].set_title(f"frame {f} t={rec['t'][f]:.3f}s phase={['settle','descend','close','lift','reclose'][ph[f]]}"
                        + (f"  q={res['door']['q_after_close_deg']}°" if k == 0 else f"  q_final={res['door']['q_final_deg']}° 포획 {res['capture']['n_in_cavity']}"))
        ax[k].set_xlabel("world y mm"); ax[k].set_ylabel("z mm"); ax[k].set_aspect("equal"); ax[k].legend(fontsize=7)
        ax[k].axhline(res["scoop_site"]["surface_z_mm"], ls="--", c="k", lw=0.5)
    fig.suptitle(f"S1 scoop {tag}: 폐합 종료 vs 최종 (YZ 단면)")
    fig.tight_layout(); fig.savefig(out / f"snapshot_{tag}.png", dpi=110); plt.close(fig)


def gates(a):
    """폴더의 scoop_s1_seed*.json 을 읽어 G1~G5 판정. (스모크 제외)"""
    out = Path(a.out)
    runs = [json.load(open(p)) for p in sorted(out.glob("scoop_s1_seed*.json")) if "smoke" not in p.name]
    if not runs:
        raise SystemExit("판정할 실행 결과가 없다")
    mass = np.array([r["capture"]["mass_g"] for r in runs]); n_in = [r["capture"]["n_in_cavity"] for r in runs]
    walls = [r["wall_seconds"] for r in runs]
    cov = float(mass.std(ddof=1) / mass.mean()) if len(mass) > 1 and mass.mean() > 0 else None
    g = {
        "artifact": "GATES_W3_DEME_SCOOP_S1", "n_runs": len(runs), "run_files": [f"scoop_s1_{r['tag']}.json" for r in runs],
        "inputs_sha16": runs[0]["inputs_sha16"],
        "G1_no_divergence": {"pass": all(not r["diverged"] for r in runs), "diverged": [r["diverged"] for r in runs],
                             "steps_completed": [r["steps_completed"] for r in runs],
                             "plunge_mm_param": [r["params"]["plunge_mm"] for r in runs],
                             "plunge_reached_mm": [r["trajectory"]["plunge_reached_mm"] for r in runs],
                             "descend_hold_steps": [r["trajectory"]["descend_hold_steps"] for r in runs],
                             "pops": [r["pops"] for r in runs],
                             "real_procedure_plunge_mm": 25.0,
                             "note": "실물 잠김 25 mm 는 이 더미(능선 37 mm·강체 바닥)에서 발산(attempt_plunge25_*). 잠김 10 mm 로 절차 완주"},
        "G2_door_closure_report": {"pass": all(r["door"]["stops"] for r in runs), "per_run": [
            {"tag": r["tag"], "stops": r["door"]["stops"], "q_final_deg": r["door"]["q_final_deg"],
             "servo_deg_final": r["door"]["servo_deg_final"], "lip_gap_final_mm": r["door"]["lip_gap_final_mm"],
             "mouth_end_mm": r["door"]["mouth_end_mm"], "n_pinched_at_lip": r["door"]["n_pinched_at_lip"],
             "n_touch_both": r["door"]["n_touch_both_meshes"], "close_peak_lipF_N": r["forces"]["close_peak_lipF_N"],
             "close_peak_M_res_Nm": r["forces"]["close_peak_M_res_Nm"], "M_stall_Nm": r["servo"]["M_stall_Nm"]}
            for r in runs], "real_reference_servo_deg": "D481 cycle5 닫힘 2.8~3.5 / 재닫힘 2.3~3.4 (기계 정지 2.5)"},
        "G3_capture_mass": {"pass": bool((mass > 0).all()), "mass_g": mass.round(4).tolist(), "n_in_cavity": n_in,
                            "n_carried_z": [r["capture"]["n_carried_z"] for r in runs],
                            "fill_vs_bulk": [r["capture"]["fill_vs_bulk"] for r in runs],
                            "mean_mass_g": round(float(mass.mean()), 4), "cavity_cm3": runs[0]["capture"]["cavity_cm3"],
                            "design_load_g": runs[0]["capture"]["design_load_g"], "design_fill_factor": runs[0]["capture"]["design_fill_factor"],
                            "note": "충전율은 보고값(판정 아님)"},
        "G4_seed_cov": {"pass": cov is not None and len(runs) >= 3, "n": len(runs), "cov": None if cov is None else round(cov, 4),
                        "seeds": [r["seed"] for r in runs], "scoop_y_mm": [r["scoop_site"]["y_mm"] for r in runs],
                        "note": "n=3 COV 는 보고값. seed 는 능선 위 y 위치를 바꾼다"},
        "G5_wall_le_5min": {"pass": all(w <= 300 for w in walls), "wall_seconds": walls},
    }
    g["all_pass"] = all(g[k]["pass"] for k in ("G1_no_divergence", "G2_door_closure_report", "G3_capture_mass",
                                                 "G4_seed_cov", "G5_wall_le_5min"))
    json.dump(g, open(out / "gates_w3_deme_scoop.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v) for k, v in g.items()
                      if k.startswith("G") or k == "all_pass"}, ensure_ascii=False))
    print(f"질량 g {mass.round(3).tolist()} 평균 {mass.mean():.3f} COV {cov}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params")
    ap.add_argument("--pile", default=str(REPO / "claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=460); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gates", action="store_true", help="--out 폴더의 결과로 G1~G5 판정만")
    args = ap.parse_args()
    gates(args) if args.gates else run(args)
