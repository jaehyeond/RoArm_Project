"""W13 — HOME→퍼내기→운반→고정 용기 배출→HOME 연속 한 사이클 (DEME, 단일 solver 인스턴스).

W11 과 같은 것 (변경 0)
    더미 npz(seed460, 렌즈 클럼프 20,000알) · S1 충돌 셸 생성 · Hertz 마찰 접촉법 · E/nu/CoR/mu/Crr/밀도 ·
    메시 E · dt 1e-6 s · 폐합 22.5 °/s · sync(하강 4 ms / 폐합 1 ms / q<6° 0.1 ms) · 서보 정지선 1.764 N·m ·
    물림 가드 3 N · 하강 팔힘 6 N · 잠김 25 mm · 상승 80 mm · 접근 간격 10 mm · 취점 자리(상자 중심).
    툴 기하·물성·클럼프 템플릿·heightmap 정의는 **동결 메인 `sim_deme_scoop_s1.py` 를 import** 해서 만든다.

rev11 (W13 resume 2026-09-13) 가 rev10 에 더한 것 — 물리/경로/fixture 변경 0
    ⓐ 접촉 의존 resume bridge 의 **실행 중 fail-closed 사전 인증**(`bridge_inputs.gated_bridge`).
       `resume_fk_here()` 직후 · bridge 의 첫 물리 step 전. 실패 = 계획된 `CLEARANCE_UNCERTIFIED` 중단이며
       발산이 아니다. 원시/RRD 는 정상 finalize 한다. timestep/state reset/extra hold/retry 없음.
    ⓑ 결정 태그 `bridge_clearance_decision` 1 개 추가(관측층). phase 열·물리·물성은 불변.
    ⓒ `--max-wall-s` 소프트 벽시계 상한 + SIGTERM/SIGINT 우아한 중단. 둘 다 step 경계에서만 본다.
       트리거되지 않으면 물리는 rev10 과 같다. 재시도·연장 아님.

W13 이 새로 하는 것 (작업 통합에 필요한 것만)
    ① 전 사이클 궤적을 **실물 관절 웨이포인트의 오프라인 FK/IK** 로 만든다(`w13_fk.py`).
       HOME = 저장된 실물 [0,0,90,0,0] 이며 시작·끝이 같은 동결 포즈다.
    ② 툴은 6자유도로 움직인다. family 9 의 "속도 유지" 규정 + 매 sync 트래커 속도 서보(순간이동 없음).
       힌지축·문 상대회전·공동 분류를 **실제 자세로 같이 회전**시킨다.
    ③ 더미 상자를 무한 BC 평면 → 유한 높이 메시 트레이로 바꾸고 도메인을 용기까지 넓힌다(DEVIATION-1).
    ④ 수신 용기(선언 픽스처) 메시 1개 고정 추가. 배출 = 실물 `place()` 규약(문 30° → 1.5 s → 닫기).
    ⑤ 조밀 제어 sync 축과 희소 입자 프레임 축을 분리 저장하고 정확한 index 로 잇는다(감사 스키마).
    ⑥ 모든 초기 입자 ID 를 source/receiving_bin/tool_residual/spill/in_flight/ambiguous 로 배타 분류한다.

주장하지 않는 것
    규정 툴 운동이다. IK 해가 있음을 보였을 뿐 실제 서보/관절 토크 가능성의 증거가 아니다.
    dt 1e-6 s 는 W13 잠정 설정이며 수렴·기본값 승격이 아니다.
    도메인/경계 변경 때문에 퍼내기 구간 수치가 W11 의 517개/10.4731 g 와 같아야 할 이유가 없다.
"""
import argparse
import hashlib
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN_REPO = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(MAIN_REPO))
sys.path.insert(0, str(HERE))

import inventory_geometry as IG                                        # noqa: E402
import sim_deme_scoop_s1 as W11SRC                                     # noqa: E402  동결 메인 소스(수정 금지)
import w13_kinematics as K                                             # noqa: E402
import w13_fk as FK                                                    # noqa: E402
import bridge_inputs as BR                                             # noqa: E402  rev11 fail-closed bridge gate

# 감사 사전등록이 동결한 phase 순서
PHASES = ["initial_home", "settle", "approach", "descend", "close", "lift", "reclose",
          "transport", "discharge", "discharge_wait", "close_after_discharge", "return_home"]
PHASE_CODE = {p: i for i, p in enumerate(PHASES)}
INV_NAMES = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]
INV = {n: i for i, n in enumerate(INV_NAMES)}
DECISIONS = ["initial_home_end", "settle_end", "approach_end", "descend_end", "close_stop", "lift_end",
             "reclose_end", "bridge_clearance_decision", "transport_end", "release_before", "release_after",
             "wait_end", "close_after_discharge_end", "return_home_end"]


def sha256_full(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha16(p):
    return sha256_full(p)[:16]


class _StopCycle(Exception):
    """--stop-after-phase 로 계획된 조기 종료(스모크용). 발산/실패와 구분한다."""


class _SignalStop(Exception):
    """SIGTERM/SIGINT 를 step 경계에서 받아 우아하게 멈춘다. 발산이 아니다."""


class _WallCapStop(Exception):
    """--max-wall-s 소프트 벽시계 상한. 발산이 아니며 연장·재시도를 하지 않는다."""


def run(a):
    import DEME

    P = dict(W11SRC.DEFAULT)
    P.update(K.W13_DEFAULT)
    if a.params:
        P.update(json.load(open(a.params)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{a.seed}" + ("_smoke" if a.smoke else "")
    t_start = time.time()

    # ── 우아한 중단: 핸들러는 **플래그만** 세우고 step 경계에서 예외로 바꾼다.
    #    (C++ solver 호출 중 예외를 던지면 버퍼가 일관되지 않아 finalize 가 깨진다.)
    sig_state = {"hit": None, "count": 0}

    def _on_signal(signum, _frame):
        sig_state["count"] += 1
        if sig_state["hit"] is None:
            sig_state["hit"] = int(signum)
        print(f"\n■ 신호 {signum} 수신 — 다음 step 경계에서 우아하게 종료한다"
              f"(누적 {sig_state['count']}회). 물리 재시작·연장 없음.", flush=True)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _on_signal)
    wall_cap = float(a.max_wall_s) if a.max_wall_s else None

    rng = np.random.default_rng(a.seed)
    y_s = float(rng.uniform(*P["scoop_y_range_mm"])) / 1000.0
    x_s = P["scoop_x_mm"] / 1000.0
    dt = P["dt_sync_s"]
    dt_d = P["dt_sync_descend_s"]
    dt_c = P.get("dt_sync_close_s") or dt
    dt_fine = float(P["diag_fine_sync_s"]) if P.get("diag_fine_sync_s") else None
    q_diag = float(P.get("diag_flush_below_q_deg") or 0.0)
    q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
    q_end = P["close_end_joint_deg"]
    design = json.load(open(W11SRC.DESIGN))
    R_lip = design["derived"]["lip_radius_from_hinge_mm"] / 1000.0
    M_stall = P["servo_torque_Nm"] * P["servo_torque_fraction"]
    v_t = P["transport_speed_mm_s"] / 1000.0
    pf_dt = float(P["particle_frame_dt_s"])
    v_settle = 9.81 * dt                      # 한 제어 sync 동안 중력이 주는 속도 — 정착 판정의 독립 근거

    # ── 입력 ────────────────────────────────────────────────────────────────
    z_full = np.load(a.pile, allow_pickle=True)
    box = np.asarray(z_full["box_bounds_m"], float)
    if a.max_particles:                       # 스모크 전용 축소 — 본 실행 금지
        k_keep = int(a.max_particles)
        z = {k: np.asarray(z_full[k]) for k in z_full.files}
        tpl_n = len(json.loads(str(z_full["clump_template_json"]))["sphere_radii_m"])
        cp_all = np.asarray(z_full["clump_positions_m"], float)
        sel = np.sort(np.argsort(np.hypot(cp_all[:, 0] - x_s, cp_all[:, 1]))[:k_keep])
        z["clump_positions_m"] = cp_all[sel]
        z["clump_quaternions_xyzw"] = np.asarray(z_full["clump_quaternions_xyzw"])[sel]
        rows = (sel[:, None] * tpl_n + np.arange(tpl_n)[None, :]).ravel()
        for key in ("positions_m", "radii_m", "velocities_m_s", "initial_positions_m", "particle_ids"):
            if key in z:
                z[key] = np.asarray(z_full[key])[rows]
        z["clump_ids"] = np.repeat(np.arange(len(sel)), tpl_n)
        print(f"⚠ 스모크 축소: 클럼프 {k_keep} 개만 사용(본 실행 금지)", flush=True)
    else:
        z = z_full
    # rev34 (W25-A): 트레이 안쪽 경계 선언. 키 없음/npz_box_bounds = rev32 그대로(npz box_bounds_m).
    box, w25_tray = FK.w25_tray_bounds(
        box, P, np.asarray(z_full["positions_m"], float) if "positions_m" in z_full.files else None,
        np.asarray(z_full["radii_m"], float) if "radii_m" in z_full.files else None)

    from roarm_rl.heightmap import GridSpec, heightmap_from_particles
    cell = 0.005
    spec = GridSpec(origin_xy_m=(float(box[0, 0]), float(box[1, 0])), cell_m=cell,
                    shape=(int(math.ceil((box[1, 1] - box[1, 0]) / cell)),
                           int(math.ceil((box[0, 1] - box[0, 0]) / cell))),
                    frame="deme_box_floor_center", z_datum_m=0.0)

    # ── 툴 셸(동결 소스가 굽는 그대로) ─────────────────────────────────────
    fixed_m, door_m, lip_f, lip_d, hinge_off, L5mm, mesh_check = W11SRC.load_tool(P, q_open)
    P["lip_l5_mm"] = [float(v) for v in L5mm]                 # 두꺼워진 충돌 owner 립 (z = 169.6 mm)
    # rev34: 취점 자리. rev32 경로면 w11 포즈는 (0,0) 그대로이고 x_s/y_s 도 건드리지 않는다.
    site_xy, w25_site = FK.w25_scoop_site(P, P["lip_l5_mm"])
    site_w11 = (0.0, 0.0) if w25_site.get("rev32_path") else site_xy
    if not w25_site.get("rev32_path"):
        x_s, y_s = site_xy
    # rev36-chain (W26): 셀 위치 입력. 키 없음(기본) = rev35 그대로(W11 취점 자리·rev34 회전).
    W26SITE = None
    if P.get("w26_cell_start_at_scoop_pose") and P.get("w26_cell_cmd_box_xy_m") is not None:
        import w26_cell_site as CS
        site_xy, R_cell_site, W26SITE = CS.cell_site(P, P["lip_l5_mm"], P["w26_cell_cmd_box_xy_m"],
                                                     float(P.get("w26_cell_tool_roll_deg", 0.0)))
        site_w11 = site_xy
        x_s, y_s = site_xy
    axis_w = W11SRC.R_W @ np.array([0.0, 1.0, 0.0])           # 굽힌 프레임의 문 힌지축
    fixed_v = np.asarray(fixed_m.vertices, float)
    door_v = np.asarray(door_m.vertices, float)
    L5_m = np.array(P["lip_l5_mm"], float) / 1000.0
    C5_m = np.array(P["bowl_center_l5_mm"], float) / 1000.0

    # ── 입자 등록 ──────────────────────────────────────────────────────────
    s = DEME.DEMSolver()
    s.SetVerbosity("ERROR")
    mp = {"E": P["E_pa"], "nu": P["nu"], "CoR": P["CoR"], "mu": P["mu"], "Crr": P["Crr"]}
    mat_p, mat_w = s.LoadMaterial(mp), s.LoadMaterial(mp)
    mat_m = s.LoadMaterial(dict(mp, E=P["E_mesh_pa"] or P["E_pa"]))
    s.UseFrictionalHertzianModel()
    pos, rad, m_p, shape_info, tpl = W11SRC.add_particles(s, z, P, mat_p)
    n_p = len(pos)
    k_sph = 1 if tpl is None else len(tpl["sphere_radii_m"])

    def spheres(pp, oq):
        return (pp, np.full(len(pp), rad)) if tpl is None else W11SRC.expand_spheres(pp, oq, tpl)

    near0 = np.hypot(pos[:, 0] - x_s, pos[:, 1] - y_s) < P["footprint_r_mm"] / 1000.0
    z_surf_pre = (float(pos[near0, 2].max()) + rad) if tpl is None else \
        W11SRC.surface_z(*spheres(pos, np.asarray(z["clump_quaternions_xyzw"], float)), x_s, y_s, P)
    # rev34 절차 (a): 닫힌 채 접근해 **표면에서** 문을 연다(실물 hw_s1_manual.py:159-162). OFF = rev32 approach_gap_mm.
    gap_mm = P["w25_door_open_gap_mm"] if P.get("w25_proc_open_at_surface") else P["approach_gap_mm"]
    z_lip0 = z_surf_pre + gap_mm / 1000.0

    # ── 프레임 어댑터 + 웨이포인트(오프라인 FK/IK) ─────────────────────────
    ad, ad_info, w25_frame_info = FK.build_adapter_w25(box, z_surf_pre, P, P["lip_l5_mm"])
    R_scoop = None if w25_frame_info["rev32_path"] else np.asarray(w25_frame_info["R_scoop_owner_box"], float)
    if W26SITE is not None:                                   # rev36-chain: 베이스 회전을 반영한 툴 회전 + 벽 여유(fail-closed)
        R_scoop = R_cell_site
        z_chk = [z_lip0 + float(P.get("w26_cell_start_clearance_mm", 10.0)) / 1000.0,
                 z_lip0 - P["plunge_mm"] / 1000.0, z_lip0 + P["lift_mm"] / 1000.0]
        w_min, w_each = CS.wall_clearance(fixed_v, door_v, hinge_off, axis_w, q_open,
                                          [q_end, q_open], site_xy, z_chk, R_scoop, box)
        W26SITE.update({"wall_clearance_min_m": w_min, "wall_clearance_m": w_each, "wall_check_z_m": z_chk,
                        "wall_margin_m": float(P.get("w26_cell_wall_margin_mm", 2.0)) / 1000.0})
        if w_min < W26SITE["wall_margin_m"]:
            raise SystemExit(f"셀 위치 벽 여유 부족: 최소 {w_min*1000:.2f} mm < {W26SITE['wall_margin_m']*1000:.1f} mm "
                             f"(명령 상자 xy {W26SITE['cmd_box_xy_m']})")
        print(f"셀 위치: 명령 상자 xy {np.round(np.array(W26SITE['cmd_box_xy_m'])*1000, 2).tolist()} mm → 립 "
              f"{np.round(np.array(site_xy)*1000, 3).tolist()} mm · 베이스 {W26SITE['base_deg']:.3f}° · 롤 {W26SITE['roll_deg']:.1f}° · "
              f"툴 yaw {W26SITE['tool_yaw_box_deg']:.3f}° · 벽 여유 최소 {w_min*1000:.2f} mm", flush=True)
    r0 = float(np.hypot(*FK.lip_pose(ad_info["reference_pose_q5"], P["lip_l5_mm"])[0][:2]))
    z_travel = float(ad_info["floor_robot_z_m"] + P["travel_cm"] / 100.0 - ad_info["t_robot_m"][2])
    z_surf5 = float(ad_info["pellet_robot_z_m"] + 0.05 - ad_info["t_robot_m"][2])
    box_top = float(box[2, 1])
    tray = K.tray_mesh(box, P["tray_wall_t_mm"] / 1000.0)
    wps_probe, _ = FK.build_waypoints(ad, r0, z_travel, z_lip0, z_surf5, 0.0, P["place_base_deg"])
    bin_center = np.asarray([w for w in wps_probe if w["name"] == "place_target"][0]["lip_world_m"])[:2]
    binm, bin_info = K.bin_mesh(P, bin_center)
    z_release = bin_info["rim_z_m"] + P["release_clearance_mm"] / 1000.0
    wps, wp_extra = FK.build_waypoints(ad, r0, z_travel, z_lip0, z_surf5, z_release, P["place_base_deg"])
    W = {w["name"]: w for w in wps}
    ik_warn = [{"name": w["name"], "branch": w["ik"].get("branch"), "err_mm": None if w["ik"]["err_m"] is None
                else round(w["ik"]["err_m"] * 1000, 4), "tilt_deg": w["ik"].get("tilt_deg"),
                "limits": w["ik"]["limits_violations"]}
               for w in wps if w["q5"] is None or w["ik"]["limits_violations"] or (w["ik"]["err_m"] or 0) > 0.001]
    print(f"FK 어댑터 t_robot {np.round(ad_info['t_robot_m'], 6).tolist()} · r0 {r0:.6f} m · "
          f"운반 z {z_travel*1000:.2f} · 배출 z {z_release*1000:.2f} · 용기 중심 "
          f"{np.round(bin_center*1000, 2).tolist()} mm", flush=True)
    print(f"IK 경고 웨이포인트: {json.dumps(ik_warn, ensure_ascii=False) if ik_warn else '없음'}", flush=True)
    if any(w["q5"] is None for w in wps):
        raise SystemExit("웨이포인트 IK 실패 — preflight 에서 해결해야 한다")

    bridge_log = []

    # ── 도메인: 전 경로 툴 스윕 + 더미 + 용기 ─────────────────────────────
    swept = []
    for w in wps:
        p, R = ad.owner_pose(w["q5"])
        swept.append((R @ fixed_v.T).T + p)
        swept.append((R @ (door_v + hinge_off).T).T + p)
    swept = np.vstack(swept + [np.asarray(tray.vertices), np.asarray(binm.vertices)])
    pad = float(P["domain_pad_m"])
    dom_x = (float(min(swept[:, 0].min(), box[0, 0])) - pad, float(max(swept[:, 0].max(), box[0, 1])) + pad)
    dom_y = (float(min(swept[:, 1].min(), box[1, 0])) - pad, float(max(swept[:, 1].max(), box[1, 1])) + pad)
    dom_z = (float(box[2, 0]), float(max(swept[:, 2].max() + pad, P["domain_top_m"])))
    s.InstructBoxDomainDimension(dom_x, dom_y, dom_z)
    s.InstructBoxDomainBoundingBC(P["domain_bc"], mat_w)

    # ── rev11 bridge 인증 정적 입력(순수 기하 · 물리 전에 동결) ────────────
    #    door_vertices_local_m = **힌지 기준 문 정점**(door_v). 문은 자기 owner 원점을 가진 별도
    #    추적 강체이므로 국소 반경도 그 원점 기준이다(강체 힌지 가정 없음).
    #    length_unit_l_m/voxel_size_m 은 설치본 초기화값이며 params 로만 주입한다. 없으면 위치 격자
    #    항이 증명 불가라 인증이 fail-closed 된다(SOURCE_EVIDENCE §1). **추정하지 않는다.**
    # 수치 증거는 **물리 파라미터와 분리된 별도 config** 다(CONTRACT_ADDENDUM_01 §4).
    # params_w13.json 은 건드리지 않는다. 없으면 위치 격자 항이 증명 불가라 인증이 fail-closed 된다.
    NE = json.load(open(a.numeric_evidence)) if a.numeric_evidence else None
    if NE is not None:
        if not NE.get("REUSABLE_FOR_REV11"):
            raise SystemExit(f"수치 증거가 이 revision 에 재사용 가능하다고 표시돼 있지 않다: "
                             f"{a.numeric_evidence}")
        nev = NE.get("values_if_usable") or {}
        # 재확인: 이 run 의 실제 도메인이 증거의 고정 도메인과 binary64 로 같아야 한다.
        ev_dom = (NE.get("rev11_recomputed_domain_m") or {})
        got = {"x": list(dom_x), "y": list(dom_y), "z": list(dom_z)}
        if any(list(ev_dom.get(k, [])) != got[k] for k in ("x", "y", "z")):
            raise SystemExit(f"도메인이 수치 증거의 고정 입력과 다르다 — lattice 재사용 금지. "
                             f"evidence={ev_dom} actual={got}")
        print(f"수치 증거 적용: l={nev.get('l_m')} voxel={nev.get('voxel_size_m')} "
              f"coord_bound={nev.get('absolute_coordinate_bound_m')} (정적 재구성, 런타임 관측 아님)",
              flush=True)
    else:
        nev = {}
    bridge_static = BR.build_static_inputs(
        box_bounds_m=box, P=P, bin_center_xy=bin_center, fixed_v_m=fixed_v, door_v_m=door_v,
        hinge_off_m=hinge_off, lip_l5_owner_mm=P["lip_l5_mm"], door_axis_owner=axis_w,
        door_q_open_deg=q_open, timestep_s=P["timestep_s"], transport_speed_m_s=v_t,
        close_deg_s=P["close_deg_s"], dts_s=dt, domain_x=dom_x, domain_y=dom_y, domain_z=dom_z,
        length_unit_l_m=nev.get("l_m"), voxel_size_m=nev.get("voxel_size_m"),
        domain_max_coord_m=nev.get("absolute_coordinate_bound_m"),
        numeric_evidence={"path": a.numeric_evidence,
                          "sha256": sha256_full(a.numeric_evidence) if a.numeric_evidence else None,
                          "source_document": NE.get("source_document") if NE else None,
                          "kind": "installed_binary_static_reconstruction_not_runtime_observation"})

    objs = out / "_obj"
    objs.mkdir(exist_ok=True)
    fpath, dpath = objs / f"fixed_{tag}.obj", objs / f"door_open{q_open:.1f}_{tag}.obj"
    tpath, bpath = objs / f"tray_{tag}.obj", objs / f"bin_{tag}.obj"
    for m_, p_ in ((fixed_m, fpath), (door_m, dpath), (tray, tpath), (binm, bpath)):
        m_.export(p_)

    p_home, R_home = ad.owner_pose(FK.HOME_Q5)
    # rev35-cell (W26): 연쇄 퍼내기 셀. 키가 없으면(기본) rev34 와 동일하게 HOME 에서 시작한다.
    #   켜면 툴(문 닫힘)을 취점 립 자리 표면 위 clearance 에서 시작하고, 접근 관절 이동을 건너뛴다.
    CELL = bool(P.get("w26_cell_start_at_scoop_pose"))
    cell_clear = float(P.get("w26_cell_start_clearance_mm", 10.0)) / 1000.0 if CELL else 0.0
    if CELL:
        p_home = np.array([site_w11[0], site_w11[1], z_lip0 + cell_clear], float)
        R_home = np.eye(3) if R_scoop is None else R_scoop.copy()
    w26_cell_info = {"enabled": CELL, "start_clearance_m": cell_clear,
                     "start_owner_pos_m": [float(v) for v in p_home] if CELL else None,
                     "rule": "rev35-cell: 취점 립 자리 표면+clearance 에서 문 닫힘으로 시작, 접근 관절 이동 생략, "
                             "표면까지 하강 후 rev34 절차(표면 문 열기·플런지·닫기·채터링·상승·재닫기) 그대로"}
    q_home_xyzw = K.mat_to_quat_xyzw(R_home)
    # HOME 은 문 닫힘(사용자 규약 [0,0,90,0,0,0] 의 6번째 0). 문 OBJ 는 q_open 자세로 구웠으므로
    # 초기 자세에 상대회전 (q_end - q_open) 을 곱한다.
    R_door_home = R_home @ K.axis_angle(axis_w, q_end - q_open)
    q_door_home_xyzw = K.mat_to_quat_xyzw(R_door_home)
    mf = s.AddWavefrontMeshObject(str(fpath), mat_m, True, False)
    mf.SetMass(P["fixed_mass_kg"]); mf.SetMOI([1e-5, 1e-5, 1e-5]); mf.SetFamily(9)
    mf.SetInitPos(p_home.tolist()); mf.SetInitQuat(q_home_xyzw.tolist())      # xyzw (스모크 S3 실측)
    md = s.AddWavefrontMeshObject(str(dpath), mat_m, True, False)
    md.SetMass(P["door_mass_kg"]); md.SetMOI([4e-5, 4e-5, 4e-5]); md.SetFamily(9)
    md.SetInitPos((p_home + R_home @ hinge_off).tolist()); md.SetInitQuat(q_door_home_xyzw.tolist())
    mt = s.AddWavefrontMeshObject(str(tpath), mat_w, True, False)
    mt.SetMass(1.0); mt.SetMOI([1.0, 1.0, 1.0]); mt.SetFamily(1)
    mb = s.AddWavefrontMeshObject(str(bpath), mat_w, True, False)
    mb.SetMass(1.0); mb.SetMOI([1.0, 1.0, 1.0]); mb.SetFamily(1)
    s.SetFamilyFixed(1)
    s.SetFamilyPrescribedLinVel(9)
    s.SetFamilyPrescribedAngVel(9)

    s.SetInitTimeStep(P["timestep_s"])
    s.SetGravitationalAcceleration([0, 0, -9.81])
    s.SetCDUpdateFreq(P["cd_update_freq"])
    s.SetErrorOutVelocity(P["error_out_vel"])
    if P.get("max_velocity_m_s"):
        s.SetMaxVelocity(float(P["max_velocity_m_s"]))
    # rev11: 수신 용기(bin) **관찰 전용** 트래커 추가(코디네이터 msg_cd2ae8394287 승인).
    #   · 이미 존재하는 `mb` 메시에 read-only Track 만 붙인다. 새 물체·물성·시간·경로·제어 변경 0.
    #   · family 1 고정 메시라 Track 은 접촉 조회 핸들일 뿐 물리를 바꾸지 않는다.
    #   · 옛 자료에는 이 관찰이 없다 → 그쪽은 계속 "bin not-observed" 다(0 접촉이 아니다).
    trk_f, trk_d, trk_t, trk_b = s.Track(mf), s.Track(md), s.Track(mt), s.Track(mb)
    s.Initialize()
    # 실제 owner ID (의미상 mesh role id 와 별개다 — 혼동 금지)
    owners = {"fixed": int(trk_f.GetOwnerID()), "door": int(trk_d.GetOwnerID()),
              "tray": int(trk_t.GetOwnerID()), "bin": int(trk_b.GetOwnerID())}
    if any(v >= 2 ** 32 - 1 for v in owners.values()):
        raise RuntimeError("트래커 owner 미할당")
    print(f"Initialize OK · owner {owners} · 도메인 x{np.round(dom_x,3).tolist()} y{np.round(dom_y,3).tolist()} "
          f"z{np.round(dom_z,3).tolist()} · 트레이 tri {len(tray.faces)} · 용기 tri {len(binm.faces)} "
          f"· 셸 {mesh_check['shell_tri']}", flush=True)

    # ── 명령 상태 / 포즈 ───────────────────────────────────────────────────
    # mode: "fk" = 관절 웨이포인트 FK / "w11" = 동결 W11 직교 자세(취점 구간) / "abs" = 정렬 보간
    cmd = {"q5": list(FK.HOME_Q5), "q": float(q_end), "dz": 0.0, "mode": "fk",
           "w11_z": float(z_lip0), "p": None, "R": None}
    if CELL:                                        # rev35-cell: 시작부터 W11 취점 자세(표면 + clearance)
        cmd["mode"], cmd["dz"] = "w11", float(cell_clear)

    def pose_of(c):
        if c["mode"] == "w11":                       # 동결 W11: 립 (0,0,z), 자세 항등 — IK 에 맞춰 바꾸지 않는다
            p_f = np.array([site_w11[0], site_w11[1], c["w11_z"] + c["dz"]], float)
            R_f = np.eye(3) if R_scoop is None else R_scoop.copy()
        elif c["mode"] == "abs":
            p_f, R_f = np.asarray(c["p"], float), np.asarray(c["R"], float)
        else:
            p_f, R_f = ad.owner_pose(c["q5"])
            p_f = p_f + np.array([0.0, 0.0, c["dz"]])
        R_rel = K.axis_angle(axis_w, c["q"] - q_open)
        return p_f, R_f, p_f + R_f @ hinge_off, R_f @ R_rel

    def actual_pose():
        return (np.asarray(trk_f.Pos(), float), K.quat_xyzw_to_mat(trk_f.OriQ()),
                np.asarray(trk_d.Pos(), float), K.quat_xyzw_to_mat(trk_d.OriQ()))

    def servo(dts):
        """다음 sync 끝에 목표 포즈에 닿는 유한 속도. SetPos/SetOriQ 순간이동 없음.

        ERRATUM_04 E8: `SetVel(float3)`/`SetAngVel(float3)` 은 성분을 **binary32** 로 저장한다
        (AuxClasses.h:242-250). 그래서 파이썬에서 **미리 float32 로 만들어** 넘기고, 인증기도
        같은 벡터로 상한을 계산한다. 바인딩이 어차피 하던 변환을 명시화한 것이라
        **실효 명령은 바뀌지 않는다**(round-to-nearest 1회는 동일).
        """
        p_ft, R_ft, p_dt, R_dt = pose_of(cmd)
        p_fa, R_fa, p_da, R_da = actual_pose()
        v_f32, w_f32 = BR.BN.command_float32_vectors(p_ft - p_fa, K.rotvec_of(R_fa.T @ R_ft), dts)
        v_d32, w_d32 = BR.BN.command_float32_vectors(p_dt - p_da, K.rotvec_of(R_da.T @ R_dt), dts)
        trk_f.SetVel(v_f32.tolist())
        trk_f.SetAngVel(w_f32.tolist())
        trk_d.SetVel(v_d32.tolist())
        trk_d.SetAngVel(w_d32.tolist())
        return p_ft, R_ft, p_dt, R_dt

    def q_from_pose(R_fa, R_da):
        rv = K.rotvec_of(R_fa.T @ R_da)
        ang = float(np.linalg.norm(rv))
        if ang < 1e-12:
            return q_open
        return q_open + (1.0 if float(np.dot(rv, axis_w)) >= 0 else -1.0) * math.degrees(ang)

    # ── 재고 분류 (면분할 용기 · 경계 밴드는 ambiguous) ────────────────────
    margin = float(P["classify_margin_mm"]) / 1000.0
    n_th = int(P["bin_n_theta"])
    th = np.linspace(0, 2 * math.pi, n_th, endpoint=False) + math.pi / n_th     # 면 법선 각(정다각형)
    bin_nrm = np.stack([np.cos(th), np.sin(th)], 1)
    bin_apothem = bin_info["inner_r_m"] * math.cos(math.pi / n_th)              # 내접 반평면 거리
    bin_c = np.asarray(bin_info["center_xy_m"], float)

    # 분류 기하 설정 — 전부 **동결 파라미터에서 파생**. 판정식 자체는 IG 한 곳에만 있다.
    inv_cfg = {"R_W": W11SRC.R_W, "lip_l5_m": L5_m, "bowl_center_l5_m": C5_m,
               "bowl_r_in_m": P["bowl_r_in_mm"] / 1000.0,
               "cheek_half_y_m": P["cheek_half_y_mm"] / 1000.0,
               "bin_center_xy_m": bin_c, "bin_normals": bin_nrm, "bin_apothem_m": bin_apothem,
               "bin_floor_inner_z_m": bin_info["floor_inner_z_m"], "bin_rim_z_m": bin_info["rim_z_m"],
               "box_bounds_m": box, "box_top_m": box_top, "margin_m": margin,
               "v_settle_m_s": v_settle, "spill_rest_z_m": P["spill_rest_z_m"]}

    def classify(pp, vv, oq, p_fa, R_fa):
        """배타·전수 6분류 — **회전된 알 전체 형상**(모든 구체 + 반지름)으로 판정한다.

        정정 근거: 감사 `RAW_CALLSITE_REVIEW_01.md` + 코디네이터 `msg_fb93d27a72b9`.
        예전 판은 owner 중심만 써서 공통 raw 계약(`inventory_code` 는 전체 oriented clump 구체에서
        재계산되는 주장)과 어긋났다. 렌즈 템플릿의 owner→표면 지지는 최대 약 2.25035 mm 로
        동결 margin 2.5 mm 와 같은 자릿수여서, 중심만 보면 벽 판정이 실제로 달라진다.

        의미(동결 독립 oracle 과 동일):
          · `in_*`  = **모든 구체가 통째로** margin 만큼 안쪽 (중심 + 반지름까지 고려)
          · `near_*`= **어느 한 구체라도** 경계 밴드에 걸침 → ambiguous
        동결값(margin 2.5 mm · 우선순위 · 속도창 v_settle)은 **바꾸지 않는다**.
        이 분류는 기록/정착 수지용이며 제어·추가 대기·경로에 연결하지 않는다.
        """
        sp, sr = spheres(pp, oq)                      # 정준 템플릿 전개: **실제** owner 쿼터니언
        n = len(pp)
        return IG.classify_spheres(np.asarray(sp, float).reshape(n, k_sph, 3),
                                   np.asarray(sr, float).reshape(n, k_sph),
                                   np.linalg.norm(np.asarray(vv, float), axis=1),
                                   p_fa, R_fa, inv_cfg)

    def inv_counts(code):
        c = np.bincount(np.asarray(code).astype(int), minlength=len(INV_NAMES))
        return {INV_NAMES[i]: int(c[i]) for i in range(len(INV_NAMES))}

    # ── 기록 버퍼 ──────────────────────────────────────────────────────────
    D = {k: [] for k in ("t", "wall", "phase", "dts", "tool_pos", "tool_quat", "tool_tgt_pos",
                         "tool_tgt_quat", "door_pos", "door_quat", "door_deg", "door_tgt_deg", "q5",
                         "nodes_F", "nodes_D", "cmd_mode", "t_legacy9",
                         "door_tgt_pos", "door_tgt_quat", "int_steps")}
    SC = {k: [] for k in ("v_particle_max", "n_fixed", "n_door", "F_fixed_N", "F_door_N", "Fz_fixed_up_N",
                          "M_hinge_rel_Nm", "lipF_N", "max_single_contact_N", "lip_track_err_mm",
                          "hinge_track_err_mm", "door_rigid_vel_resid_mm_s", "door_rel_omega_deg_s",
                          "n_tray_contacts", "F_tray_N", "n_bin_contacts", "F_bin_N",
                          "n_sphere_near_wall", "n_sphere_overlap_wall",
                          "min_sphere_wall_gap_mm", "max_abs_pos_nonfinite",
                          "n_sphere_center_outside_wall", "n_clump_center_outside_box",
                          # rev32 프로파일링 관측층(읽기 전용 엔진 질의). 물리·제어·저장프레임 규약 불변.
                          "engine_num_contacts", "engine_cd_update_freq", "engine_query_wall_s")}
    CT = {"idx": [], "pt": [], "f": [], "mesh": []}
    PF = {k: [] for k in ("t", "sync", "pos", "quat", "vel", "omega", "inv")}
    _int_steps_cache = {}

    def _int_steps(dts):
        """설치본 누산 규칙 재현값(파생). dts 종류가 몇 개뿐이라 캐시한다."""
        if dts <= 0:
            return 0
        k = float(dts)
        if k not in _int_steps_cache:
            _int_steps_cache[k] = int(BR.BN.internal_steps(k, P["timestep_s"])[0])
        return _int_steps_cache[k]

    subphases, trans_idx, decisions, stops = [], [], [], []
    state = {"holds": 0, "pop_steps": 0, "v_max": 0.0, "phase": "initial_home", "sub": "hold", "next_pf": 0.0}
    tl_path = out / f"timeline_{tag}.json"

    def flush(tagstr):
        n = len(D["t"])
        lo = max(0, n - 40)
        json.dump({"state": tagstr, "n_sync": n, "stops": stops,
                   "last_rows": [dict({k: SC[k][i] for k in SC}, t=D["t"][i], phase=PHASES[D["phase"][i]],
                                      subphase=subphases[i], door_deg=D["door_deg"][i]) for i in range(lo, n)]},
                  open(tl_path, "w"), ensure_ascii=False)

    def record_particle_frame(force=False):
        t_now = float(s.GetSimTime())
        if not force and t_now < state["next_pf"] - 1e-12:
            return len(PF["t"]) - 1
        p_fa, R_fa, _, _ = actual_pose()
        pp = np.asarray(s.GetOwnerPosition(0, n_p), float)
        vv = np.asarray(s.GetOwnerVelocity(0, n_p), float)
        PF["t"].append(t_now)
        PF["sync"].append(len(D["t"]) - 1)
        PF["pos"].append(pp.astype(np.float32))
        PF["quat"].append(np.asarray(s.GetOwnerOriQ(0, n_p), np.float32))
        PF["vel"].append(vv.astype(np.float32))
        PF["omega"].append(np.asarray(s.GetOwnerAngVel(0, n_p), np.float32))
        # 실제 owner 쿼터니언을 분류에 **전달**한다(예전엔 저장만 하고 쓰지 않았다).
        oq_now = None if tpl is None else np.asarray(s.GetOwnerOriQ(0, n_p), float)
        PF["inv"].append(classify(pp, vv, oq_now, p_fa, R_fa))
        state["next_pf"] = (math.floor(t_now / pf_dt + 1e-9) + 1) * pf_dt
        return len(PF["t"]) - 1

    def decision(name):
        fi = record_particle_frame(force=True)
        p_fa, R_fa, p_da, R_da = actual_pose()
        d = {"tag": name, "phase": state["phase"], "subphase": state["sub"],
             "sync_index": len(D["t"]) - 1, "particle_frame_index": fi,
             "sim_t": float(PF["t"][fi]), "wall_s": round(time.time() - t_start, 2),
             "counts": inv_counts(PF["inv"][fi]), "n_total": int(n_p),
             "q_cmd_deg": round(cmd["q"], 4), "q_actual_deg": round(q_from_pose(R_fa, R_da), 4),
             "q5_cmd": [round(v, 4) for v in cmd["q5"]],
             "lip_actual_mm": (p_fa * 1000).round(4).tolist(),
             "v_max_m_s": round(float(np.linalg.norm(PF["vel"][fi], axis=1).max()), 6)}
        decisions.append(d)
        print(f"  ▣ 결정 {name} t={d['sim_t']:.4f}s {d['counts']} q={d['q_actual_deg']:.3f}° "
              f"w={d['wall_s']:.0f}s", flush=True)
        return d

    def sample(dts, tgt):
        p_ft, R_ft, p_dt, R_dt = tgt
        p_fa, R_fa, p_da, R_da = actual_pose()
        # ERRATUM_03: 정본은 **반올림하지 않은 binary64** 엔진 시각이다. rev10 의 round(...,9) 는
        # 차이에 최대 1 ns 십진 양자화를 넣어 내부 duration 의 정확한 증거가 될 수 없다.
        # 옛 9자리 값은 display/legacy 이름으로 따로 남겨 비교 가능성만 유지한다.
        _t_raw = float(s.GetSimTime())
        D["t"].append(_t_raw)
        D["t_legacy9"].append(round(_t_raw, 9))
        D["wall"].append(round(time.time() - t_start, 4))
        D["phase"].append(PHASE_CODE[state["phase"]]); D["dts"].append(dts); subphases.append(state["sub"])
        D["tool_pos"].append(p_fa.astype(np.float32)); D["tool_quat"].append(np.asarray(trk_f.OriQ(), np.float32))
        D["tool_tgt_pos"].append(p_ft.astype(np.float32))
        D["tool_tgt_quat"].append(K.mat_to_quat_xyzw(R_ft).astype(np.float32))
        D["door_pos"].append(p_da.astype(np.float32)); D["door_quat"].append(np.asarray(trk_d.OriQ(), np.float32))
        # 문 **목표 포즈**를 고정부 목표와 따로 남긴다(감사 RAW_SCHEMA_REQUIRED).
        D["door_tgt_pos"].append(np.asarray(p_dt, np.float32))
        D["door_tgt_quat"].append(K.mat_to_quat_xyzw(R_dt).astype(np.float32))
        D["door_deg"].append(q_from_pose(R_fa, R_da)); D["door_tgt_deg"].append(cmd["q"])
        # 내부 step 수는 설치본 누산 규칙을 **재현한 파생값**이다 — 엔진이 노출하는 관측 카운터가 아니다.
        D["int_steps"].append(_int_steps(dts))
        # rev32: 엔진 관측 카운터 읽기 전용 질의. 상태를 **읽기만** 한다(Set* 호출 0, 물리 불변).
        # 예외를 낼 수 있는 코드보다 **앞**에 둔다 — pop-stop 이 떠도 SC 배열 길이가 어긋나지 않는다.
        _q0 = time.perf_counter()
        SC["engine_num_contacts"].append(float(s.GetNumContacts()))
        SC["engine_cd_update_freq"].append(float(s.GetUpdateFreq()))
        SC["engine_query_wall_s"].append(time.perf_counter() - _q0)
        D["q5"].append(np.asarray(cmd["q5"], np.float32))
        # rev11 관측층: 이 sync 의 명령 모드. "fk" 면 q5 가 실제 관절 상태이고,
        # "w11"/"abs" 면 직교 포즈 명령이라 q5 는 그 시점의 관절 상태가 아니다.
        # Isaac 표시 렌더러가 관절을 쓸지 IK 를 풀지 여기서 정확히 고른다(추측 금지).
        D["cmd_mode"].append(str(cmd["mode"]))
        D["nodes_F"].append(np.asarray(trk_f.GetMeshNodesGlobal(), np.float32))
        D["nodes_D"].append(np.asarray(trk_d.GetMeshNodesGlobal(), np.float32))
        vv = np.asarray(s.GetOwnerVelocity(0, n_p), float)
        vmax = float(np.linalg.norm(vv, axis=1).max())
        state["v_max"] = max(state["v_max"], vmax)
        state["pop_steps"] += int(vmax > P["pop_speed_m_s"])
        axis_now = R_fa @ axis_w
        w_tool = R_fa @ np.asarray(trk_f.AngVelLocal(), float)
        w_door = R_da @ np.asarray(trk_d.AngVelLocal(), float)
        v_door_exp = np.asarray(trk_f.Vel(), float) + np.cross(w_tool, p_da - p_fa)
        M_rel = Fz_up = F_f = F_d = f1 = 0.0
        n_f = n_d = 0
        for key, trk, mid in (("fixed", trk_f, 0), ("door", trk_d, 1)):
            pts, frcs = trk.GetContactForces()
            if key == "fixed":
                n_f = len(pts)
            else:
                n_d = len(pts)
            if len(pts):
                Pp, Ff = np.asarray(pts, float), np.asarray(frcs, float)
                tot = float(np.linalg.norm(Ff.sum(0)))
                if key == "fixed":
                    F_f, Fz_up = tot, float(Ff[:, 2].sum())
                else:
                    F_d = tot
                    # 문 액추에이터가 이기는 저항 = 접촉력의 힌지축 모멘트(상대 힌지 저항).
                    # 베이스 회전의 강체 운동은 접촉력이 아니므로 여기 들어가지 않는다.
                    M_rel = float(np.dot(np.cross(Pp - p_da, Ff).sum(0), axis_now))
                f1 = max(f1, float(np.linalg.norm(Ff, axis=1).max()))
                CT["idx"].append(np.full(len(pts), len(D["t"]) - 1, np.int32))
                CT["pt"].append(Pp.astype(np.float32)); CT["f"].append(Ff.astype(np.float32))
                CT["mesh"].append(np.full(len(pts), mid, np.int8))
        SC["v_particle_max"].append(vmax); SC["n_fixed"].append(n_f); SC["n_door"].append(n_d)
        SC["F_fixed_N"].append(F_f); SC["F_door_N"].append(F_d); SC["Fz_fixed_up_N"].append(Fz_up)
        SC["M_hinge_rel_Nm"].append(M_rel); SC["lipF_N"].append(abs(M_rel) / R_lip)
        SC["max_single_contact_N"].append(f1)
        SC["lip_track_err_mm"].append(float(np.linalg.norm(p_fa - p_ft)) * 1000)
        SC["hinge_track_err_mm"].append(float(np.linalg.norm(p_da - p_dt)) * 1000)
        SC["door_rigid_vel_resid_mm_s"].append(float(np.linalg.norm(np.asarray(trk_d.Vel(), float) - v_door_exp)) * 1000)
        SC["door_rel_omega_deg_s"].append(math.degrees(float(np.dot(w_door - w_tool, axis_now))))
        # 벽(트레이) 접촉·근접·관통 — 세 수치를 따로 보고한다(근접이 있다고 접촉을 요구하지 않는다)
        tpts, tfrc = trk_t.GetContactForces()
        SC["n_tray_contacts"].append(len(tpts))
        SC["F_tray_N"].append(float(np.linalg.norm(np.asarray(tfrc, float).sum(0))) if len(tpts) else 0.0)
        if len(tpts):                      # 벽 접촉 점·힘은 사후 복원이 불가능하므로 원자료에 남긴다(mesh_id 2 = tray)
            CT["idx"].append(np.full(len(tpts), len(D["t"]) - 1, np.int32))
            CT["pt"].append(np.asarray(tpts, np.float32)); CT["f"].append(np.asarray(tfrc, np.float32))
            CT["mesh"].append(np.full(len(tpts), 2, np.int8))
        # rev11 관찰 추가: 수신 용기 접촉(의미상 mesh role id 3). 스칼라 축약은 트레이와 **같은 규약**
        # (norm_of_vector_sum)을 쓴다. 점·힘 원시는 사후 복원이 불가능하므로 그대로 남긴다.
        bpts, bfrc = trk_b.GetContactForces()
        SC["n_bin_contacts"].append(len(bpts))
        SC["F_bin_N"].append(float(np.linalg.norm(np.asarray(bfrc, float).sum(0))) if len(bpts) else 0.0)
        if len(bpts):
            CT["idx"].append(np.full(len(bpts), len(D["t"]) - 1, np.int32))
            CT["pt"].append(np.asarray(bpts, np.float32)); CT["f"].append(np.asarray(bfrc, np.float32))
            CT["mesh"].append(np.full(len(bpts), 3, np.int8))
        pp_now = np.asarray(s.GetOwnerPosition(0, n_p), float)
        oq_now = None if tpl is None else np.asarray(s.GetOwnerOriQ(0, n_p), float)
        sp_now, sr_now = spheres(pp_now, oq_now)
        gap = np.minimum.reduce([sp_now[:, 0] - box[0, 0], box[0, 1] - sp_now[:, 0],
                                 sp_now[:, 1] - box[1, 0], box[1, 1] - sp_now[:, 1]]) - sr_now
        SC["n_sphere_near_wall"].append(int((gap < float(P["wall_near_tol_mm"]) / 1000.0).sum()))
        SC["n_sphere_overlap_wall"].append(int((gap < 0.0).sum()))
        SC["min_sphere_wall_gap_mm"].append(float(gap.min()) * 1000.0)
        # 봉쇄 판정은 **중심 기준**이다(동결 더미가 t=0 부터 얕은 구 겹침을 갖기 때문).
        cen = np.minimum.reduce([sp_now[:, 0] - box[0, 0], box[0, 1] - sp_now[:, 0],
                                 sp_now[:, 1] - box[1, 0], box[1, 1] - sp_now[:, 1]])
        SC["n_sphere_center_outside_wall"].append(int((cen < 0.0).sum()))
        SC["n_clump_center_outside_box"].append(int(((pp_now[:, 0] < box[0, 0]) | (pp_now[:, 0] > box[0, 1]) |
                                                     (pp_now[:, 1] < box[1, 0]) | (pp_now[:, 1] > box[1, 1]) |
                                                     (pp_now[:, 2] < box[2, 0]) | (pp_now[:, 2] > box_top)).sum()))
        SC["max_abs_pos_nonfinite"].append(float((~np.isfinite(pp_now)).sum() + (~np.isfinite(vv)).sum()))
        if P.get("diag_stop_v_m_s") and vmax > P["diag_stop_v_m_s"]:
            flush("pop_stop")
            np.savez_compressed(out / f"pop_event_{tag}.npz",
                                positions_m=np.asarray(s.GetOwnerPosition(0, n_p), float), velocities_m_s=vv,
                                sync_index=np.int64(len(D["t"]) - 1), sim_t=np.float64(D["t"][-1]),
                                phase=state["phase"], subphase=state["sub"])
            raise RuntimeError(f"pop-stop: 입자 최대속도 {vmax:.1f} m/s > {P['diag_stop_v_m_s']} "
                               f"({state['phase']}/{state['sub']})")
        record_particle_frame()
        return len(D["t"]) - 1

    step_guard = {"fn": None}           # rev11: bridge 구간에만 꽂히는 per-sync fail-before-physics 검사기

    def step(dts):
        tgt = servo(dts)
        g = step_guard["fn"]
        if g is not None:                # ← DoDynamicsThenSync **전에** 본다(실패 시 물리 0 step)
            # 고정부와 문의 **실측 포즈·명령 목표를 각각 따로** 넘긴다(강체 힌지 가정 없음).
            p_fa_g, R_fa_g, p_da_g, R_da_g = actual_pose()
            row = g(dts, (p_fa_g, R_fa_g), (p_da_g, R_da_g), (tgt[0], tgt[1]), (tgt[2], tgt[3]))
            if not row.get("ok"):
                raise BR.PrecheckFailed(row)
        s.DoDynamicsThenSync(dts)
        r = sample(dts, tgt)
        # step 경계 — 여기서만 우아한 중단을 일으킨다(솔버 호출 중 예외 금지).
        if sig_state["hit"] is not None:
            raise _SignalStop(f"signal {sig_state['hit']} at sync {r} t={D['t'][r]:.6f}s")
        if wall_cap is not None and (time.time() - t_start) > wall_cap:
            raise _WallCapStop(f"wall {time.time()-t_start:.1f}s > --max-wall-s {wall_cap}s "
                               f"at sync {r} t={D['t'][r]:.6f}s")
        return r

    def record_t0():
        """t=0 의 진짜 HOME 상태를 dts=0 행으로 남긴다. 경과시간을 늘리지 않는다(감사 요구)."""
        state["phase"], state["sub"] = "initial_home", "t0_state"
        # rev29: 행 0 은 phase 전환이 아니다 → transition_sync_index 에 넣지 않는다(RAW_SCHEMA_REQUIRED.md:52).
        return sample(0.0, pose_of(cmd))

    def enter(phase, sub):
        # rev29: transition_sync_index 는 **phase 가 바뀌는 행**만 담는다(subphase 전환·행 0 제외,
        # RAW_SCHEMA_REQUIRED.md:52 "exactly equals dense phase-change indices"). subphase 전환의
        # state 갱신과 강제 입자 프레임 저장은 그대로다(저장 프레임 규약·ERRATUM_01/03 불변).
        if phase != state["phase"] or sub != state["sub"]:
            phase_changed = phase != state["phase"]
            state["phase"], state["sub"] = phase, sub
            if phase_changed:
                trans_idx.append(len(D["t"]))
            if D["t"]:
                record_particle_frame(force=True)

    # ── 이동 프리미티브 ────────────────────────────────────────────────────
    def joint_move(phase, sub, q5_goal, dts=None):
        """관절 공간 직선 보간. 립 선속도가 transport_speed 를 넘지 않도록 단계를 나눈다."""
        dts = dts or dt
        enter(phase, sub)
        q0 = np.asarray(cmd["q5"], float)
        q1 = np.asarray(q5_goal, float)
        p0, _ = ad.owner_pose(q0)
        p1, _ = ad.owner_pose(q1)
        n = max(1, int(math.ceil(float(np.linalg.norm(p1 - p0)) / (v_t * dts))))
        for i in range(1, n + 1):
            cmd["q5"] = (q0 + (q1 - q0) * (i / n)).tolist()
            r = step(dts)
            if i % 200 == 0:
                flush(sub)
                print(f"  {phase}/{sub} {i}/{n} lip={np.round(np.asarray(D['tool_pos'][r], float)*1000,2).tolist()} "
                      f"v_max={SC['v_particle_max'][r]:.2f} 오차={SC['lip_track_err_mm'][r]:.6f} mm "
                      f"w={time.time()-t_start:.0f}s", flush=True)
        flush(sub)

    def z_move(phase, sub, dz_goal, speed, dts, hold_fn=None, max_factor=3.0):
        """직교 수직 이동(W11 하강·상승 보존). 자세는 그대로 두고 owner z 만 바꾼다."""
        enter(phase, sub)
        n = max(1, int(math.ceil(abs(dz_goal - cmd["dz"]) / speed / dts)))
        for i in range(int(max_factor * n) if hold_fn else n + 2):
            hold = bool(hold_fn() if hold_fn else False)
            if not hold:
                d = dz_goal - cmd["dz"]
                cmd["dz"] += math.copysign(min(abs(d), speed * dts), d)
            state["holds"] += int(hold)
            r = step(dts)
            if i % 60 == 0:
                flush(sub)
                print(f"  {phase}/{sub} {i} z_lip={float(D['tool_pos'][r][2])*1000:8.3f} "
                      f"{'HOLD' if hold else '    '} F/D={SC['n_fixed'][r]}/{SC['n_door'][r]} "
                      f"Fz={SC['Fz_fixed_up_N'][r]:.3f} 단일={SC['max_single_contact_N'][r]:.3f} "
                      f"v_max={SC['v_particle_max'][r]:.2f} w={time.time()-t_start:.0f}s", flush=True)
            if abs(cmd["dz"] - dz_goal) < 1e-12 and not hold:
                break
        flush(sub)

    def fine_on():
        return dt_fine is not None and cmd["q"] < q_diag

    def door_move(phase, sub, opening, budget_syncs, time_budget_s=None, q_goal=None):
        enter(phase, sub)
        q_hi = q_open if q_goal is None else float(q_goal)      # rev34: 채터링 개방(서보 8°) 목표. None = rev32
        stop = None
        t0 = float(s.GetSimTime())
        n_extra = 0 if dt_fine is None else int(math.ceil(q_diag / P["close_deg_s"] / dt_fine)) + 50
        i = 0
        while i < budget_syncs + n_extra:
            dts = dt_fine if fine_on() else dt_c
            d = P["close_deg_s"] * dts
            cmd["q"] = min(q_hi, cmd["q"] + d) if opening else max(q_end, cmd["q"] - d)
            r = step(dts)
            reason = None
            if opening:
                if cmd["q"] >= q_hi - 1e-12:
                    reason = "reached_open_end"
                elif P["servo_stall_model"] and -SC["M_hinge_rel_Nm"][r] >= M_stall:
                    reason = "servo_stall_open"
                elif P.get("door_pinch_guard_N") and SC["max_single_contact_N"][r] >= P["door_pinch_guard_N"]:
                    reason = "pinch_guard_open"
            else:
                if cmd["q"] <= q_end + 1e-12:
                    reason = "reached_close_end"
                elif P.get("door_min_q_deg") is not None and cmd["q"] <= P["door_min_q_deg"] + 1e-9:
                    reason = "door_floor"
                elif P["servo_stall_model"] and SC["M_hinge_rel_Nm"][r] >= M_stall:
                    reason = "servo_stall"
                elif P.get("door_pinch_guard_N") and SC["max_single_contact_N"][r] >= P["door_pinch_guard_N"]:
                    reason = "pinch_guard"
            if reason:
                _, R_fa, _, R_da = actual_pose()
                stop = {"phase": phase, "subphase": sub, "reason": reason, "sync_index": r,
                        "q_cmd_deg": round(cmd["q"], 4), "q_actual_deg": round(q_from_pose(R_fa, R_da), 4),
                        "servo_deg": round(cmd["q"] + P["servo_zero_offset_deg"], 4),
                        "sim_t": D["t"][r], "M_hinge_rel_Nm": SC["M_hinge_rel_Nm"][r],
                        "max_single_contact_N": SC["max_single_contact_N"][r]}
                print(f"  문 정지 [{sub}] q={cmd['q']:.3f}° ({reason}, M={SC['M_hinge_rel_Nm'][r]:.4f} N·m, "
                      f"단일 {SC['max_single_contact_N'][r]:.3f} N)", flush=True)
                break
            if i % 100 == 0:
                flush(sub)
                print(f"  {sub} {i} q={cmd['q']:6.3f}° D={SC['n_door'][r]:4d} M={SC['M_hinge_rel_Nm'][r]:8.4f} "
                      f"단일={SC['max_single_contact_N'][r]:.3f} v_max={SC['v_particle_max'][r]:.2f} "
                      f"w={time.time()-t_start:.0f}s", flush=True)
            if time_budget_s is not None and float(s.GetSimTime()) - t0 >= time_budget_s - 1e-12:
                break
            i += 1
        if stop is None:
            _, R_fa, _, R_da = actual_pose()
            stop = {"phase": phase, "subphase": sub, "reason": "steps_exhausted",
                    "sync_index": len(D["t"]) - 1, "q_cmd_deg": round(cmd["q"], 4),
                    "q_actual_deg": round(q_from_pose(R_fa, R_da), 4), "sim_t": D["t"][-1]}
        stops.append(stop)
        flush(sub)
        return stop

    def dwell(phase, sub, seconds, dts=None):
        dts = dts or dt
        enter(phase, sub)
        for i in range(max(1, int(round(seconds / dts)))):
            r = step(dts)
            if i % 100 == 0:
                flush(sub)
                print(f"  {phase}/{sub} {i} t={D['t'][r]:.4f} F/D={SC['n_fixed'][r]}/{SC['n_door'][r]} "
                      f"v_max={SC['v_particle_max'][r]:.2f} w={time.time()-t_start:.0f}s", flush=True)
        flush(sub)

    def align_move(phase, sub, p_to, R_to, dts=None):
        """현재 명령 포즈 → 목표 포즈로 연속 보간(순간이동 금지). 보정량을 반환해 보고한다."""
        dts = dts or dt
        enter(phase, sub)
        p_fr, R_fr, _, _ = pose_of(cmd)
        rv = K.rotvec_of(R_fr.T @ np.asarray(R_to, float))
        ang = float(np.degrees(np.linalg.norm(rv)))
        dist = float(np.linalg.norm(np.asarray(p_to, float) - p_fr))
        n = max(1, int(math.ceil(max(dist / (v_t * dts), ang / (P["close_deg_s"] * dts)))))
        cmd["mode"] = "abs"
        for i in range(1, n + 1):
            f = i / n
            cmd["p"] = (p_fr + (np.asarray(p_to, float) - p_fr) * f).tolist()
            cmd["R"] = (R_fr @ K.axis_angle(rv / max(np.linalg.norm(rv), 1e-12), ang * f)
                        if np.linalg.norm(rv) > 1e-12 else R_fr).tolist()
            step(dts)
        flush(sub)
        return {"subphase": sub, "distance_mm": round(dist * 1000, 6), "rotation_deg": round(ang, 6),
                "syncs": n, "from_mm": (p_fr * 1000).round(4).tolist(),
                "to_mm": (np.asarray(p_to, float) * 1000).round(4).tolist()}

    def resume_fk_here():
        """현재 owner z 에서 IK 를 풀어 FK 모드로 복귀할 관절해를 준다. 실패는 보고한다."""
        z_now = float(np.asarray(trk_f.Pos(), float)[2])
        zr = float(np.asarray(ad.to_robot([0.0, 0.0, z_now]))[2])
        sol = FK.solve_fast(r0, zr, P["lip_l5_mm"])
        if sol is None:
            return None, {"z_world_mm": round(z_now * 1000, 3), "solved": False}
        q = list(sol["q5"]); q[0] = 0.0
        return q, {"z_world_mm": round(z_now * 1000, 3), "solved": True, "branch": sol["branch"],
                   "err_mm": round(sol["err_m"] * 1000, 4), "tilt_deg": round(sol["tilt_deg"], 4),
                   "limits": FK.in_limits(q)}

    def maybe_stop(name):
        if a.stop_after_phase and name == a.stop_after_phase:
            raise _StopCycle(name)

    # ── 사이클 ──────────────────────────────────────────────────────────────
    diverged, fail_reason, stopped_early, abort_class = False, None, None, None
    hm_pre, z_target, z_reached = None, None, None
    align_log, resume_log = [], []
    chatter_log, reclose_read = [], {}
    try:
        record_t0()                       # 진짜 HOME t=0 상태(추가 hold 없음)
        decision("initial_home_end")
        maybe_stop("initial_home")

        # W11 과 **같은 누적 경과시간**: initial_home 은 dts=0 기록행이므로 settle 25 sync 뒤
        #   t = 0.100025 s 로 W11 settle 종료 시각과 정확히 같은 격자에 선다.
        dwell("settle", "settle", P["settle_steps"] * dt)
        pp0 = np.asarray(s.GetOwnerPosition(0, n_p), float)
        oq0 = None if tpl is None else np.asarray(s.GetOwnerOriQ(0, n_p), float)
        z_surf = (float(pp0[near0, 2].max()) + rad) if tpl is None else W11SRC.surface_z(*spheres(pp0, oq0), x_s, y_s, P)
        hm_pre = heightmap_from_particles(*spheres(pp0, oq0), spec).height
        z_target = z_surf - P["plunge_mm"] / 1000.0
        decision("settle_end")
        print(f"  재안착 뒤 펠릿면 {z_surf*1000:.3f} mm → 잠김 목표 립 z {z_target*1000:.3f} mm", flush=True)
        maybe_stop("settle")

        if CELL:
            # rev35-cell: 접근 관절 이동·정렬 생략 — 표면 + clearance 에서 표면(립 z_lip0)까지 하강 속도로 내린다.
            z_move("approach", "cell_lower_to_surface", 0.0, P["descend_mm_s"] / 1000.0, dt_d)
            cmd["mode"], cmd["w11_z"], cmd["dz"] = "w11", float(z_lip0), 0.0
        else:
            for nm in ("p1_tool_vertical", "above_pile_travel", "surface_plus_50mm", "approach_gap"):
                joint_move("approach", nm, W[nm]["q5"])
            # 동결 W11 취점 포즈(립 (0,0,z_lip0), 자세 항등)로 정확히 정렬한다. IK 에 맞춰 W11 을 바꾸지 않는다.
            align_log.append(align_move("approach", "align_to_w11_scoop_pose",
                                        np.array([site_w11[0], site_w11[1], z_lip0]),
                                        np.eye(3) if R_scoop is None else R_scoop))
            cmd["mode"], cmd["w11_z"], cmd["dz"] = "w11", float(z_lip0), 0.0
        # HOME 은 문 닫힘이었다 → 여기서 실물 규약대로 연속 개방한다(별도 subphase, 저장 프레임 포함).
        door_move("approach", "door_open_at_approach", True,
                  max(1, int(round((q_open - q_end) / P["close_deg_s"] / dt_c))))
        decision("approach_end")
        maybe_stop("approach")

        z_move("descend", "plunge", z_target - z_lip0,
               P["descend_mm_s"] / 1000.0, dt_d,
               hold_fn=lambda: bool(SC["Fz_fixed_up_N"] and SC["Fz_fixed_up_N"][-1] > P["arm_force_max_N"]),
               max_factor=P["descend_max_steps_factor"])
        z_reached = float(np.asarray(trk_f.Pos(), float)[2])
        decision("descend_end")
        maybe_stop("descend")

        door_move("close", "close", False, max(1, int(round((q_open - q_end) / P["close_deg_s"] / dt_c))))
        decision("close_stop")
        # rev34 절차 (b): 실물 채터링(hw_s1_manual.py:163-166) — 닫힘 서보 읽기 > 문턱이면 서보 8° 로 열었다
        # 다시 닫기를 최대 N 회. 서보 읽기 = 실제 관절각(추적 포즈) + servo_zero_offset_deg. phase 는 "close" 유지.
        if P.get("w25_proc_chatter"):
            off = P["servo_zero_offset_deg"]
            q_ch = P["w25_chatter_open_servo_deg"] - off
            nb_ch = max(1, int(round((q_ch - q_end) / P["close_deg_s"] / dt_c)))
            for k in range(int(P["w25_chatter_max_retries"]) + 1):
                _, R_fa_c, _, R_da_c = actual_pose()
                q_rd = float(q_from_pose(R_fa_c, R_da_c))
                row = {"k": k, "sim_t": float(s.GetSimTime()), "sync_index": len(D["t"]) - 1,
                       "read_joint_deg": round(q_rd, 6), "read_servo_deg": round(q_rd + off, 6),
                       "cmd_joint_deg": round(cmd["q"], 6), "cmd_servo_deg": round(cmd["q"] + off, 6),
                       "threshold_servo_deg": P["w25_chatter_threshold_servo_deg"]}
                if q_rd + off <= P["w25_chatter_threshold_servo_deg"] + 1e-12:
                    row["action"] = "stop_at_or_below_threshold"
                elif k >= int(P["w25_chatter_max_retries"]):
                    row["action"] = "retries_exhausted"
                else:
                    row["action"] = "chatter"
                chatter_log.append(row)
                if row["action"] != "chatter":
                    break
                row["open_stop"] = door_move("close", f"chatter_open_{k+1}", True, nb_ch, q_goal=q_ch)
                row["close_stop"] = door_move("close", f"chatter_close_{k+1}", False,
                                              max(1, int(round((q_ch - q_end) / P["close_deg_s"] / dt_c))))
        maybe_stop("close")

        # rev34 절차 (c): 실물은 펠릿면 +8 cm 로 올린 뒤(hw_s1_manual.py:167) 항상 door(0) 재닫기.
        # OFF = rev32(도달 깊이에서 lift_mm 상승, 명령이 닫힘 끝이 아닐 때만 재닫기).
        if P.get("w25_proc_lift_to_surface_mm") is not None:
            lift_goal = (z_surf + P["w25_proc_lift_to_surface_mm"] / 1000.0) - z_lip0
        else:
            lift_goal = cmd["dz"] + P["lift_mm"] / 1000.0
        z_move("lift", "lift", lift_goal, P["lift_mm_s"] / 1000.0, dt)
        decision("lift_end")

        if cmd["q"] > q_end + 1e-9 or P.get("w25_proc_always_reclose"):
            nb = max(1, int(round(P["reclose_max_steps"] * dt / dt_c)))
            door_move("reclose", "reclose", False, nb, time_budget_s=nb * dt_c)
        decision("reclose_end")
        if P.get("w25_proc_chatter") or P.get("w25_proc_always_reclose"):
            _, R_fa_c, _, R_da_c = actual_pose()
            q_rd = float(q_from_pose(R_fa_c, R_da_c))
            reclose_read.update(read_joint_deg=round(q_rd, 6), read_servo_deg=round(q_rd + P["servo_zero_offset_deg"], 6),
                                cmd_joint_deg=round(cmd["q"], 6), sim_t=float(s.GetSimTime()))
        maybe_stop("reclose")

        # 취점 구간 끝 — 현재 높이에서 IK 를 풀어 FK 모드로 복귀한다(정렬량·IK 잔차를 기록).
        q_res, ik_res = resume_fk_here()
        resume_log.append(ik_res)
        if q_res is None:
            raise RuntimeError(f"취점 구간 뒤 FK 복귀 IK 실패: {ik_res}")
        p_res, R_res = ad.owner_pose(q_res)
        # ── rev11: bridge 사전 인증 결정 지점 ────────────────────────────────
        #    여기서 원시 결정 스냅샷을 먼저 남긴다(인증이 실패해도 보존된다).
        bridge_decision = decision("bridge_clearance_decision")
        p_fa_b, R_fa_b, p_da_b, R_da_b = actual_pose()          # 실측 포즈
        p_fc_b, R_fc_b, p_dc_b, R_dc_b = pose_of(cmd)           # align_move 가 실제로 보간하는 시작점

        def _bridge_align():
            align_log.append(align_move("transport", "align_from_w11_scoop_pose", p_res, R_res))

        def _bridge_joint():
            cmd["mode"], cmd["q5"], cmd["dz"] = "fk", q_res, 0.0
            joint_move("transport", "post_lift_travel", W["post_lift_travel"]["q5"])

        # 인증 대상은 **이 두 구간만**이다. 뒤따르는 place_* 관절 이동은 감사의 3,584 구간
        # bounded 증명이 이미 덮는다(그 증명이 빼 둔 것이 바로 이 두 bridge 다).
        def _install_step_guard(fn):
            step_guard["fn"] = fn

        BR.gated_bridge(
            static=bridge_static,
            cert_kwargs=dict(
                actual_fixed=(p_fa_b, R_fa_b), actual_door=(p_da_b, R_da_b),
                commanded_fixed=(p_fc_b, R_fc_b),
                align_target_pose=(p_res, R_res),
                q_res_deg=q_res, q_post_lift_deg=W["post_lift_travel"]["q5"],
                owner_pose_fn=ad.owner_pose,
                door_q_deg_actual=float(q_from_pose(R_fa_b, R_da_b)),
                door_q_deg_commanded=float(cmd["q"]),
                z_reached_m=z_reached, ik_resume=ik_res,
                sim_t_s=float(s.GetSimTime()), sync_index=len(D["t"]) - 1,
                wall_s=round(time.time() - t_start, 3)),
            physics_step_counter=lambda: len(D["t"]),
            install_step_guard=_install_step_guard,
            bridge_moves=[_bridge_align, _bridge_joint],
            record=lambda c: (c.update(decision_tag=bridge_decision["tag"],
                                       decision_particle_frame_index=bridge_decision["particle_frame_index"]),
                              bridge_log.append(c))[-1])
        for nm in ("place_retract_base0", "place_retract_base90",
                   "place_extend_travel", "place_target"):
            joint_move("transport", nm, W[nm]["q5"])
        decision("transport_end")
        decision("release_before")
        maybe_stop("transport")

        door_move("discharge", "door_open", True, max(1, int(round((q_open - q_end) / P["close_deg_s"] / dt_c))))
        decision("release_after")
        dwell("discharge_wait", "wait", P["discharge_hold_s"])
        decision("wait_end")
        door_move("close_after_discharge", "close_after_discharge", False,
                  max(1, int(round((q_open - q_end) / P["close_deg_s"] / dt_c))))
        decision("close_after_discharge_end")
        maybe_stop("close_after_discharge")

        for nm in ("place_up_travel", "return_retract_base90", "return_retract_base0", "return_home"):
            joint_move("return_home", nm, W[nm]["q5"])
        dwell("return_home", "home_hold", P["final_home_hold_s"])
    except _StopCycle as exc:
        stopped_early = str(exc)
        print(f"■ 계획된 조기 종료(스모크): {exc}", flush=True)
        flush("stopped_early")
    except (BR.ClearanceUncertified, BR.PrecheckFailed) as exc:   # ← Exception 보다 **먼저** 와야 한다
        abort_class, fail_reason = "CLEARANCE_UNCERTIFIED", repr(exc)
        print(f"■ 계획된 중단 CLEARANCE_UNCERTIFIED(발산 아님): {exc}", flush=True)
        flush("clearance_uncertified")
    except _SignalStop as exc:
        abort_class, fail_reason = "SIGNAL_STOP", repr(exc)
        print(f"■ 신호 종료 — 부분 원시를 finalize 한다: {exc}", flush=True)
        flush("signal_stop")
    except _WallCapStop as exc:
        abort_class, fail_reason = "WALL_CAP_STOP", repr(exc)
        print(f"■ 벽시계 상한 종료 — 부분 원시를 finalize 한다: {exc}", flush=True)
        flush("wall_cap_stop")
    except Exception as exc:                                  # noqa: BLE001
        diverged, fail_reason = True, repr(exc)
        print(f"🔴 중단: {exc}", flush=True)
        flush("diverged")
    wall = time.time() - t_start
    try:
        decision("return_home_end")
    except Exception as exc:                                  # noqa: BLE001
        print(f"⚠ 최종 결정 기록 실패: {exc}", flush=True)
        if not decisions:
            raise

    # ── 최종 판독 ───────────────────────────────────────────────────────────
    pp = np.asarray(s.GetOwnerPosition(0, n_p), float)
    oq = None if tpl is None else np.asarray(s.GetOwnerOriQ(0, n_p), float)
    vv = np.asarray(s.GetOwnerVelocity(0, n_p), float)
    p_fa, R_fa, p_da, R_da = actual_pose()
    code_final = np.asarray(PF["inv"][-1])
    if hm_pre is None:
        hm_pre = heightmap_from_particles(np.asarray(z["positions_m"], float),
                                          np.asarray(z["radii_m"], float), spec).height
    src_mask = code_final == INV["source"]
    hm = heightmap_from_particles(*spheres(pp[src_mask], None if oq is None else oq[src_mask]), spec).height \
        if src_mask.any() else np.zeros_like(hm_pre)
    hm_npz = heightmap_from_particles(np.asarray(z["positions_m"], float),
                                      np.asarray(z["radii_m"], float), spec).height
    crater = W11SRC.crater_angles(hm_pre, hm, spec, (x_s, y_s), P)

    # 배출 질량: 단일값이 아니라 하한(확정)~상한(가능) 구간으로 보고한다(감사 3층 판정)
    over_bin_xy = ((pp[:, :2] - bin_c) @ bin_nrm.T).max(1) < bin_apothem + margin
    in_bin_z_band = (pp[:, 2] > bin_info["floor_inner_z_m"] - margin) & (pp[:, 2] < bin_info["rim_z_m"] + margin)
    definite = int((code_final == INV["receiving_bin"]).sum())
    amb_or_flight = (code_final == INV["ambiguous"]) | (code_final == INV["in_flight"])
    possible = definite + int((amb_or_flight & over_bin_xy & in_bin_z_band).sum()) + \
        int((amb_or_flight & over_bin_xy & (pp[:, 2] >= bin_info["rim_z_m"] - margin)).sum())
    # 정착 관측창(사전 고정 operational criterion — 결과를 보고 고르지 않는다)
    win = float(P["settlement_window_s"])
    t_end = float(PF["t"][-1])
    widx = [i for i, t_ in enumerate(PF["t"]) if t_ >= t_end - win - 1e-9]
    settle_report = {"window_s": win, "n_frames": len(widx),
                     "frame_times_s": [round(float(PF["t"][i]), 9) for i in widx],
                     "max_frame_gap_s": round(max([PF["t"][widx[k + 1]] - PF["t"][widx[k]]
                                                   for k in range(len(widx) - 1)] or [0.0]), 9),
                     "criterion": {"speed_max_m_s": P["settle_speed_max_m_s"],
                                   "center_move_max_m": P["settle_move_max_m"],
                                   "basis": "감사 사전 고정 operational criterion. 물리적 rest truth 가 아니다."},
                     "cadence_ok": bool(len(widx) >= 6 and max([PF["t"][widx[k + 1]] - PF["t"][widx[k]]
                                                                for k in range(len(widx) - 1)] or [1e9]) <= 0.05 + 1e-9)}
    if len(widx) >= 2:
        P_w = np.stack([PF["pos"][i].astype(float) for i in widx])
        V_w = np.stack([PF["vel"][i].astype(float) for i in widx])
        I_w = np.stack([np.asarray(PF["inv"][i]) for i in widx])
        move = np.linalg.norm(P_w - P_w[0], axis=2).max(0)
        spd = np.linalg.norm(V_w, axis=2).max(0)
        stable_bin = (I_w == INV["receiving_bin"]).all(0)
        settled = stable_bin & (spd <= P["settle_speed_max_m_s"]) & (move <= P["settle_move_max_m"])
        settle_report.update(n_stable_bin_all_frames=int(stable_bin.sum()), n_settled=int(settled.sum()),
                             max_speed_of_stable_m_s=round(float(spd[stable_bin].max()), 8) if stable_bin.any() else None,
                             max_move_of_stable_mm=round(float(move[stable_bin].max()) * 1000, 6) if stable_bin.any() else None)
        definite = int(settled.sum())
    else:
        settle_report["note"] = "관측창 프레임이 부족하다 — exact settled mass 를 주장하지 않는다."
        definite = 0
    disp = None

    trans = []
    for k in range(1, len(decisions)):
        a0 = np.asarray(PF["inv"][decisions[k - 1]["particle_frame_index"]]).astype(int)
        a1 = np.asarray(PF["inv"][decisions[k]["particle_frame_index"]]).astype(int)
        M = np.zeros((6, 6), int)
        np.add.at(M, (a0, a1), 1)
        trans.append({"from": decisions[k - 1]["tag"], "to": decisions[k]["tag"], "labels": INV_NAMES,
                      "matrix": M.tolist(), "moved": int(M.sum() - np.trace(M))})

    res = {
        "artifact": "DEME_W13_FULL_CYCLE_V2", "tag": tag, "seed": a.seed, "smoke": bool(a.smoke),
        "inputs_sha256": {str(k): v for k, v in {
            a.pile: sha256_full(a.pile), str(W11SRC.FIXED_STL): sha256_full(W11SRC.FIXED_STL),
            str(W11SRC.DOOR_STL): sha256_full(W11SRC.DOOR_STL), str(W11SRC.DESIGN): sha256_full(W11SRC.DESIGN),
            str(MAIN_REPO / "sim_deme_scoop_s1.py"): sha256_full(MAIN_REPO / "sim_deme_scoop_s1.py"),
            str(MAIN_REPO / "sim_scripts/roarm_kinematics.py"): sha256_full(MAIN_REPO / "sim_scripts/roarm_kinematics.py"),
            str(MAIN_REPO / "hw_s1_scoop_probe.py"): sha256_full(MAIN_REPO / "hw_s1_scoop_probe.py"),
            str(HERE / "sim_w13_full_cycle.py"): sha256_full(HERE / "sim_w13_full_cycle.py"),
            str(HERE / "w13_kinematics.py"): sha256_full(HERE / "w13_kinematics.py"),
            str(HERE / "w13_fk.py"): sha256_full(HERE / "w13_fk.py"),
            str(tpath): sha256_full(tpath), str(bpath): sha256_full(bpath),
            str(fpath): sha256_full(fpath), str(dpath): sha256_full(dpath),
            **({a.params: sha256_full(a.params)} if a.params else {})}.items()},
        "params": P, "particle": dict(shape_info, mass_kg=m_p, n=n_p), "mesh_check": mesh_check,
        "engine": {"DEME": getattr(__import__("deme"), "__version__", "?"),
                   "force_model": "UseFrictionalHertzianModel", "owner_ids": owners,
                   "domain_x_m": list(dom_x), "domain_y_m": list(dom_y), "domain_z_m": list(dom_z),
                   "prescription": "family 9 keep-as-is LinVel/AngVel + per-sync tracker velocity servo "
                                   "(no SetPos/SetOriQ teleport)"},
        "frames": {"R_W_frozen_columns_are_link5_axes": W11SRC.R_W.tolist(),
                   "fk_confirms_world_axes_parallel_to_robot": True,
                   "lip_physical_l5_mm": FK.LIP_L5_PHYSICAL_MM,
                   "lip_collision_owner_l5_mm": P["lip_l5_mm"],
                   "lip_adapter_mm": round(P["lip_l5_mm"][2] - FK.LIP_L5_PHYSICAL_MM[2], 6),
                   "cavity_center_owner_m": (W11SRC.R_W @ (np.array(P["bowl_center_l5_mm"], float) -
                                                           np.array(P["lip_l5_mm"], float)) / 1000.0).tolist(),
                   "door_hinge_offset_tool_owner_m": hinge_off.tolist(),
                   "door_axis_tool_owner": axis_w.tolist(),
                   "adapter": ad_info, "r0_robot_m": r0, "waypoints": FK.waypoint_report(wps),
                   "waypoint_extra": wp_extra, "source_constants_check": FK.verify_source_constants(),
                   "ik_warning_waypoints": ik_warn},
        "fixtures": {"tray": {"path": str(tpath), "n_tri": int(len(tray.faces)), "top_z_m": box_top,
                              "wall_t_m": P["tray_wall_t_mm"] / 1000.0, "watertight": bool(tray.is_watertight)},
                     # RAW_SCHEMA_REQUIRED_ERRATUM_02 ①: 다각 프리즘의 반경 의미를 **명시**한다.
                     # trimesh.creation.cylinder/annulus(sections=n) 는 정점을 주어진 반경의 원 위에
                     # 놓으므로 inner_r_m 는 **외접(circumradius)** 이고, 분류용 apothem 은
                     # inner_r_m * cos(pi/n) 로 유도한 값이다(같은 수식이 verify 에서도 쓰인다).
                     "bin": dict(bin_info, path=str(bpath), apothem_m=bin_apothem,
                                 circumradius_m=bin_info["inner_r_m"],
                                 radius_semantics="circumradius",
                                 radius_semantics_note=("inner_r_m/outer_r_m 은 외접 반경이다. "
                                                        "apothem_m = circumradius_m * cos(pi/n_theta)."),
                                 n_theta=int(P["bin_n_theta"]),
                                 pos_m=[float(bin_c[0]), float(bin_c[1]), float(P["bin_floor_z_m"])],
                                 quat_xyzw=[0.0, 0.0, 0.0, 1.0]),
                     "declared_not_measured": True},
        "scoop_site": {"x_mm": x_s * 1000, "y_mm": round(y_s * 1000, 2),
                       "surface_z_pre_settle_mm": round(z_surf_pre * 1000, 3),
                       "lip_z_start_mm": round(z_lip0 * 1000, 3)},
        "trajectory": {"q_open_joint_deg": q_open, "q_close_end_deg": q_end,
                       "z_travel_m": z_travel, "z_release_m": z_release, "z_lip0_m": z_lip0,
                       "place_base_deg": P["place_base_deg"], "transport_speed_m_s": v_t,
                       "sim_time_s": round(float(s.GetSimTime()), 9), "n_sync": len(D["t"]),
                       "n_particle_frames": len(PF["t"]), "n_transitions": len(trans_idx),
                       "syncs_per_phase": {p: int(sum(1 for c in D["phase"] if c == PHASE_CODE[p])) for p in PHASES},
                       "descend_hold_steps": state["holds"],
                       "descend_reached_lip_z_mm": None if z_reached is None else round(z_reached * 1000, 3),
                       "descend_target_lip_z_mm": None if z_target is None else round(z_target * 1000, 3),
                       "max_lip_track_err_mm": round(float(max(SC["lip_track_err_mm"])), 8),
                       "max_hinge_track_err_mm": round(float(max(SC["hinge_track_err_mm"])), 8),
                       "max_door_rigid_vel_resid_mm_s": round(float(max(SC["door_rigid_vel_resid_mm_s"])), 8),
                       "home_start_lip_mm": (np.asarray(D["tool_pos"][0], float) * 1000).round(4).tolist(),
                       "home_end_lip_mm": (np.asarray(D["tool_pos"][-1], float) * 1000).round(4).tolist(),
                       "align_moves": align_log, "fk_resume_ik": resume_log,
                       "home_pose_return_err_mm": round(float(np.linalg.norm(
                           np.asarray(D["tool_pos"][-1], float) - np.asarray(D["tool_pos"][0], float))) * 1000, 6)},
        "pops": {"v_particle_max_m_s": round(state["v_max"], 4), "syncs_over_pop_speed": state["pop_steps"],
                 "pop_speed_m_s": P["pop_speed_m_s"]},
        "servo": {"M_stall_Nm": M_stall, "lip_radius_m": R_lip, "pinch_guard_N": P.get("door_pinch_guard_N"),
                  "torque_note": "문 저항 모멘트는 접촉력만으로 계산한 상대 힌지 저항이다. 베이스 회전의 강체 운동은 포함하지 않는다."},
        "diverged": diverged, "fail_reason": fail_reason, "stopped_early_after_phase": stopped_early,
        "abort_class": abort_class,
        "abort_class_semantics": {
            "None": "계획된 전 사이클을 끝까지 돌았다(과학 판정은 별도).",
            "CLEARANCE_UNCERTIFIED": "bridge 사전 인증 실패로 bridge 첫 물리 step 전에 멈췄다. 발산 아님.",
            "SIGNAL_STOP": "외부 신호를 step 경계에서 받아 멈췄다. 부분 원시.",
            "WALL_CAP_STOP": "--max-wall-s 소프트 상한에서 멈췄다. 부분 원시. 연장·재시도 없음."},
        "max_wall_s": None if not a.max_wall_s else float(a.max_wall_s),
        "signal_received": sig_state["hit"], "signal_count": sig_state["count"],
        "bridge_clearance": [BR.dump_json_safe(c) for c in bridge_log],
        "smoke_max_particles": a.max_particles,
        "door": {"stops": stops, "q_final_cmd_deg": round(cmd["q"], 4),
                 "q_final_actual_deg": round(q_from_pose(R_fa, R_da), 4),
                 "servo_deg_final": round(cmd["q"] + P["servo_zero_offset_deg"], 4)},
        "delivery": {"layer1_accounting_integrity": bool(all(sum(d["counts"].values()) == n_p for d in decisions)),
                     "definite_delivered_n": definite, "definite_delivered_g": round(definite * m_p * 1000, 4),
                     "possible_delivered_n": possible, "possible_delivered_g": round(possible * m_p * 1000, 4),
                     "exact_single_value_allowed": bool(definite == possible),
                     "inventory_moving_threshold_m_s": v_settle,
                     "inventory_moving_threshold_basis": "프레임별 재고 분류에서 in_flight 를 가르는 값(g*dt_sync). "
                                                         "**정착 판정 기준이 아니다** — exact settled 판정은 settlement_window 가 한다.",
                     "settlement_window": settle_report,
                     "max_bin_particle_displacement_last_frame_mm": None if disp is None else round(disp * 1000, 6),
                     "particle_mass_g": round(m_p * 1000, 8),
                     "inventory_final": inv_counts(code_final),
                     "cycle_sim_time_s": round(float(s.GetSimTime()), 9)},
        "decisions": decisions, "transitions": trans,
        "heightmap": {"spec_version": "roarm-heightmap-v1", "cell_m": cell, "shape": list(hm.shape),
                      "max_m": float(np.nanmax(hm)), "n_particles_used": int(src_mask.sum()),
                      "pre_max_m": float(np.nanmax(hm_pre)), "npz_max_m": float(np.nanmax(hm_npz)),
                      "pre_vs_npz_max_abs_diff_m": float(np.abs(hm_pre - hm_npz).max()),
                      "removed_volume_cm3": crater["removed_volume_cm3"], "spheres_per_particle": k_sph,
                      "post_filter": "최종 재고 source 인 입자만"},
        "crater": crater, "wall_seconds": round(wall, 2),
        "w26_cell": dict(w26_cell_info, site=W26SITE),
        "w25": {"revision": "rev34", "frame": w25_frame_info, "scoop_site": w25_site,
                "scoop_site_w11_xy_m": list(site_w11), "tray": w25_tray,
                "procedure": {"open_at_surface": bool(P.get("w25_proc_open_at_surface")),
                              "door_open_gap_mm": gap_mm,
                              "chatter": bool(P.get("w25_proc_chatter")),
                              "chatter_log": chatter_log,
                              "lift_to_surface_mm": P.get("w25_proc_lift_to_surface_mm"),
                              "always_reclose": bool(P.get("w25_proc_always_reclose")),
                              "reclose_read": reclose_read,
                              "units": "joint_deg = 시뮬 문 관절각(0 = 립 맞닿음), servo_deg = joint_deg + servo_zero_offset_deg"},
                "non_claims": ["열기 토크는 실물 200(tor)이 아니라 rev32 정지 모델(0.9×1.96 N·m) 그대로다.",
                               "배치 치수(33.2/24/25 cm·NTC106)는 사용자 선언·실측 혼합이며 펠릿면 24 cm 는 미실측이다."]},
        "non_claims": [
            "규정 툴 운동이다. IK 해가 존재함을 보였을 뿐 실제 서보·관절 토크 가능성의 증거가 아니다.",
            "수신 용기 치수·자리는 선언된 시뮬 픽스처이며 실측이 아니다.",
            "물성(E 5e6·mu 0.45·Crr 0.06·CoR 0.3·밀도 905)은 실측 전 임시값 — 절대값 인용 금지.",
            "도메인 확대와 트레이 메시 벽(DEVIATION-1) 때문에 퍼내기 구간이 W11 수치와 같아야 할 이유가 없다.",
            "dt 1e-6 s 는 W13 잠정 설정이며 수렴·기본값 승격이 아니다.",
            "한 사이클이며 반복·정상상태·자율 시스템·석사 결과가 아니다.",
            "사이클 g/s 는 시뮬 물리시간 기준이며 실물 서보 처리량이 아니다.",
        ],
    }

    arr = lambda k, dtype=np.float32: np.asarray(D[k], dtype)
    cat = lambda k, shp, dtype: (np.concatenate(CT[k]) if CT[k] else np.zeros(shp, dtype))
    np.savez_compressed(
        out / f"w13_cycle_{tag}.npz",
        # ERRATUM_03: sync_t_s = 반올림하지 않은 binary64 엔진 시각(정본).
        sync_t_s=np.asarray(D["t"], np.float64),
        sync_t_s_legacy_round9_display=np.asarray(D["t_legacy9"], np.float64),
        sync_wall_elapsed_s=np.asarray(D["wall"], np.float64),
        sync_dts_s=np.asarray(D["dts"], np.float64),
        sync_requested_duration_s=np.asarray(D["dts"], np.float64),
        sync_derived_internal_steps=np.asarray(D["int_steps"], np.int64),
        sync_phase_code=np.asarray(D["phase"], np.int8),
        sync_subphase=np.asarray(subphases), sync_joint_deg=arr("q5"),
        sync_cmd_mode=np.asarray(D["cmd_mode"]),
        tool_pos_m=arr("tool_pos"), tool_quat_xyzw=arr("tool_quat"),
        # 감사 RAW_SCHEMA_REQUIRED: fixed_* 는 tool_* 와 같은 추적 강체의 별칭이다(역할 명시용).
        fixed_pos_m=arr("tool_pos"), fixed_quat_xyzw=arr("tool_quat"),
        fixed_target_pos_m=arr("tool_tgt_pos"), fixed_target_quat_xyzw=arr("tool_tgt_quat"),
        tool_target_pos_m=arr("tool_tgt_pos"), tool_target_quat_xyzw=arr("tool_tgt_quat"),
        door_pos_m=arr("door_pos"), door_quat_xyzw=arr("door_quat"),
        door_target_pos_m=arr("door_tgt_pos"), door_target_quat_xyzw=arr("door_tgt_quat"),
        door_actual_deg=np.asarray(D["door_deg"], np.float64),
        door_target_deg=np.asarray(D["door_tgt_deg"], np.float64),
        bin_pos_m=np.asarray(res["fixtures"]["bin"]["pos_m"], np.float64),
        bin_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], np.float64),
        nodes_F_m=arr("nodes_F"), nodes_D_m=arr("nodes_D"),
        **{f"scalar_{k}": np.asarray(v, np.float64) for k, v in SC.items()},
        particle_frame_t_s=np.asarray(PF["t"], np.float64),
        particle_frame_sync_index=np.asarray(PF["sync"], np.int64),
        # ERRATUM_03: 저장 행의 **정본 정체성**을 명시 배열로도 남긴다. 암묵 인덱스와 같아야 하며
        # `sync_index` 는 비감소이지 유일하지 않다(같은 sync 결정 행은 정상, 합치지 않는다).
        particle_frame_row=np.arange(len(PF["t"]), dtype=np.int64),
        particle_ids=np.arange(n_p, dtype=np.int64),
        particle_pos_m=np.stack(PF["pos"]), particle_quat_xyzw=np.stack(PF["quat"]),
        particle_vel_m_s=np.stack(PF["vel"]), particle_omega_rad_s=np.stack(PF["omega"]),
        inventory_code=np.stack(PF["inv"]), inventory_labels=np.asarray(INV_NAMES),
        contact_sync_index=cat("idx", (0,), np.int32), contact_point_m=cat("pt", (0, 3), np.float32),
        contact_force_N=cat("f", (0, 3), np.float32), contact_mesh=cat("mesh", (0,), np.int8),
        transition_sync_index=np.asarray(trans_idx, np.int64),
        decision_tags=np.asarray([d["tag"] for d in decisions]),
        decision_sync_index=np.asarray([d["sync_index"] for d in decisions], np.int64),
        decision_particle_frame_index=np.asarray([d["particle_frame_index"] for d in decisions], np.int64),
        bridge_interval_rows=np.concatenate([BR.intervals_to_arrays(c)[0] for c in bridge_log])
        if bridge_log else np.zeros((0, len(BR.INTERVAL_COLUMNS)), np.float64),
        bridge_interval_names=np.concatenate([BR.intervals_to_arrays(c)[1] for c in bridge_log])
        if bridge_log else np.zeros((0,), dtype="<U1"),
        bridge_interval_columns=np.asarray(BR.INTERVAL_COLUMNS),
        bridge_precheck_rows=np.concatenate([BR.precheck_to_arrays(c) for c in bridge_log])
        if bridge_log else np.zeros((0, len(BR.PRECHECK_COLUMNS)), np.float64),
        bridge_precheck_columns=np.asarray(BR.PRECHECK_COLUMNS),
        bridge_planned_fixed_pos_m=np.concatenate([BR.planned_targets_to_arrays(c)[0] for c in bridge_log])
        if bridge_log else np.zeros((0, 3), np.float64),
        bridge_planned_fixed_quat_xyzw=np.concatenate([BR.planned_targets_to_arrays(c)[1]
                                                       for c in bridge_log])
        if bridge_log else np.zeros((0, 4), np.float64),
        bridge_planned_door_pos_m=np.concatenate([BR.planned_targets_to_arrays(c)[2] for c in bridge_log])
        if bridge_log else np.zeros((0, 3), np.float64),
        bridge_planned_door_quat_xyzw=np.concatenate([BR.planned_targets_to_arrays(c)[3]
                                                      for c in bridge_log])
        if bridge_log else np.zeros((0, 4), np.float64),
        bridge_planned_target_segment=np.concatenate([BR.planned_targets_to_arrays(c)[4] for c in bridge_log])
        if bridge_log else np.zeros((0,), dtype="<U1"),
        bridge_clearance_json=np.asarray(json.dumps([BR.dump_json_safe(c) for c in bridge_log],
                                                    ensure_ascii=False)),
        heightmap_m=hm, heightmap_pre_m=hm_pre, heightmap_npz_m=hm_npz, box_bounds_m=box,
        tray_vertices_m=np.asarray(tray.vertices, np.float32), tray_faces=np.asarray(tray.faces),
        bin_vertices_m=np.asarray(binm.vertices, np.float32), bin_faces=np.asarray(binm.faces),
        lip_idx_f=lip_f, lip_idx_d=lip_d,
        final_positions_m=pp, final_velocities_m_s=vv,
        final_quat_xyzw=(np.zeros((0, 4)) if oq is None else oq),
        particle_mass_kg=np.float64(m_p),
        metadata_json=json.dumps({
            "artifact": "W13_RAW_V3", "phase_order": PHASES, "phase_code": PHASE_CODE,
            "inventory_labels": INV_NAMES, "particle_frame_dt_s": pf_dt,
            "quaternion_order": "xyzw",
            "contact_mesh_id": {"0": "fixed(tool bowl)", "1": "door", "2": "tray(source wall)",
                                "3": "bin(receiving cup)"},
            # RAW_SCHEMA_REQUIRED_ERRATUM_02 ②: 이름에서 유추하지 못하도록 명시 선언한다.
            "contact_semantics": {
                "force_frame": "world",
                # ⚠️ 이 표는 **의미상 role id** 이며 DEME 의 실제 owner ID 와 별개다(msg_cd2ae8394287).
                "mesh_role_to_id": {"fixed": 0, "door": 1, "tray": 2, "bin": 3},
                "role_id_is_semantic_not_owner_id": True,
                "actual_owner_ids": {"fixed": int(trk_f.GetOwnerID()), "door": int(trk_d.GetOwnerID()),
                                     "tray": int(trk_t.GetOwnerID()), "bin": int(trk_b.GetOwnerID())},
                "scalar_force_reduction": "norm_of_vector_sum",
                "scalar_force_reduction_source": ("sample(): F = float(np.linalg.norm(Ff.sum(0))) — "
                                                  "벡터합의 노름이지 노름의 합이 아니다. 네 role 모두 동일 규약."),
                "contact_point_frame": "world", "units": {"force": "N", "point": "m"},
                "sign_convention": "DEME GetContactForces 원본 그대로(부호 변환 없음)",
                "bin_observation": {
                    "observed_in_this_run": True,
                    "how": ("이미 존재하는 bin 메시(mb)에 **read-only** Track 만 붙여 "
                            "GetContactForces 를 읽고 원시로 남긴다. 새 물체·물성·시간·경로·제어 변경 0."),
                    "scalars": ["scalar_n_bin_contacts", "scalar_F_bin_N"],
                    "older_runs": ("이 관찰이 없던 자료는 계속 **bin not-observed** 다. 0 접촉이 아니며 "
                                   "소급해 채우지 않는다.")}},
            "time_authority": {
                "sync_t_s": "unrounded binary64 float(s.GetSimTime()) — 정본",
                "sync_t_s_legacy_round9_display": "rev10 호환 표시용 round(...,9). duration 근거 금지(ERRATUM_03)",
                "sync_requested_duration_s": "그 sync 에 요청한 duration",
                "sync_derived_internal_steps": "DERIVED — 설치본 누산 규칙 재현값. 엔진 관측 카운터가 아니다"},
            "body_roles": {"fixed": "tool bowl fixed half (tool_*/fixed_* 동일 owner)",
                           "door": "door half, own owner origin = hinge",
                           "independent_motion_bound": "실행 운동 상한에서 문을 고정부에서 유도하지 않는다"},
            "tool_cavity": {"tool_to_cavity_R": W11SRC.R_W.T.tolist(),
                            "cavity_origin_l5_m": (np.asarray(P["lip_l5_mm"], float) / 1000.0).tolist(),
                            "cavity_center_xz_l5_m": [P["bowl_center_l5_mm"][0] / 1000.0,
                                                      P["bowl_center_l5_mm"][2] / 1000.0],
                            "radius_m": P["bowl_r_in_mm"] / 1000.0,
                            "half_y_m": P["cheek_half_y_mm"] / 1000.0,
                            "equation": "p_cavity = R_tool_to_cavity @ R(q_tool).T @ (p_world - p_tool) + cavity_origin"},
            "source_bounds_m": box.tolist(),
            "moving_threshold_m_s": v_settle,
            "moving_threshold_operator": ">= (in_flight if speed >= threshold)",
            "spill_rest_z_m": P["spill_rest_z_m"],
            # ERRATUM_03 저장 행 규약 선언 — 소비자(Isaac/Rerun)가 추측하지 않게 한다.
            "particle_frame_row_is_authoritative_identity": True,
            "particle_frame_row_equals_arange": True,
            "particle_frame_sync_index_nondecreasing_not_unique": True,
            "particle_frame_rows_deduplicated": False,
            "same_sync_decision_rows_are_intentional": True,
            "classify_margin_m": P["classify_margin_mm"] / 1000.0,
            # 분류 기하 규약 — **중심점이 아니라 회전된 구체 전체**. 감사 RAW_CALLSITE_REVIEW_01.
            "classify_geometry_basis": "oriented_clump_all_spheres",
            "classify_spheres_per_clump": int(k_sph),
            "classify_sphere_expansion": "p_sphere = p_owner + R(owner_quat_xyzw) @ offset_m (clump-major)",
            "classify_owner_quat_source": "GetOwnerOriQ(0, n_p) at the same particle frame",
            "classify_inside_rule": "all spheres strictly inside by margin, center distance +/- sphere radius",
            "classify_boundary_rule": "any single sphere overlaps the margin band -> ambiguous",
            "classify_spill_basis": "max(sphere_center_z + sphere_radius) <= spill_rest_z_m",
            "overlap_priority": "tool > near_tool > bin > near_bin > source > near_source > in_flight > spill",
            "velocity_severity": {"warning_m_s": P["pop_speed_m_s"], "warning_operator": ">",
                                  "hard_stop_m_s": P.get("diag_stop_v_m_s"), "hard_stop_operator": ">"},
            # rev32 = rev31 + per-sync 엔진 진단(관측층). 물리 파라미터·제어·가드·저장프레임 규약 불변.
            "engine_diagnostics_rev32": {
                "purpose": "W16 비용 프로파일링 — sync 벽시계를 엔진 상태와 함께 읽는다. 과학 판정 정본 아님.",
                "read_only_queries_physics_unchanged": True,
                "queries": {
                    "scalar_engine_num_contacts": {
                        "api": "DEME.DEMSolver.GetNumContacts()",
                        "api_h": "DEM/API.h:101 (size_t; dT->getNumContacts())",
                        "meaning": "kT 가 보고한 **잠재 접촉쌍** 수. 실제 힘 발생 접촉수와 같지 않다.",
                        "dtype_note": "size_t -> float64 저장(2^53 까지 무손실)"},
                    "scalar_engine_cd_update_freq": {
                        "api": "DEME.DEMSolver.GetUpdateFreq()",
                        "api_h": "DEM/API.h:315 (float; 'Get the current update frequency')",
                        "meaning": "dT 가 kT 접촉쌍 갱신을 기다리기까지의 현재 step 수(적응형이면 변한다)."},
                    "scalar_engine_query_wall_s": {
                        "api": "time.perf_counter() delta around the two queries above",
                        "meaning": "관측층 자체의 호출 비용. sync 벽시계 해석에서 관측 오버헤드를 분리하려고 남긴다."}},
                "unavailable_queries": {
                    "expand_factor_cd_margin": {
                        "requested_api": "DEME.DEMSolver.GetExpandFactor()",
                        "api_h": "DEM/API.h:105 (C++ 에는 선언되어 있다)",
                        "status": "NOT_BOUND_IN_PYTHON_MODULE",
                        "evidence": "dir(DEME.DEMSolver) 에 'SetExpandFactor' 만 있고 'GetExpandFactor' 는 없다 (DEME 2.4.0 설치본)",
                        "action": "기록하고 건너뛴다. 엔진·바인딩 패치 0."}},
                "wall_attribution": ("sync_wall_elapsed_s 는 sample() **진입 직후** 찍힌다. 따라서 이 질의 비용은 "
                                     "같은 행이 아니라 **다음 sync 의 벽시계 차분**에 들어간다. 크기는 "
                                     "scalar_engine_query_wall_s 로 직접 잰다."),
                "not_a_scientific_result": "이 배열들은 비용 관측이다. 포획량·배출량 판정에 쓰지 않는다."},
            "numeric_receipt": BR.dump_json_safe(bridge_static["numeric_receipt"]),
            # rev34: 상자(DEME) ↔ 로봇 좌표 변환. p_robot = R_robot_box @ p_box + t_robot (소비자는 추측 금지).
            "w25_frame": {"R_robot_box": w25_frame_info["R_robot_box"], "t_robot_m": ad_info["t_robot_m"],
                          "box_frame_convention": w25_frame_info["box_frame_convention"],
                          "box_anchor": w25_frame_info["box_anchor"]},
            "note": "원시 배열이 과학 정본. RRD/영상은 관측층."}, ensure_ascii=False))
    json.dump(res, open(out / f"w13_cycle_{tag}.json", "w"), ensure_ascii=False, indent=2)
    flush("diverged" if diverged else (abort_class.lower() if abort_class
                                      else ("stopped_early" if stopped_early else "complete")))
    W11SRC.plot_heightmaps(out, tag, spec, hm_pre, hm, crater, (x_s, y_s))
    print(f"\n확정 배출 {definite} 개 = {definite*m_p*1000:.4f} g · 가능 상한 {possible} 개 "
          f"· 최종 재고 {inv_counts(code_final)} · 물리시간 {float(s.GetSimTime()):.4f} s "
          f"· sync {len(D['t'])} · 입자프레임 {len(PF['t'])} · 벽시계 {wall:.1f} s · 발산 {diverged} "
          f"· abort {abort_class} · bridge {[c['verdict'] for c in bridge_log] or 'none'}\n-> {out}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params")
    ap.add_argument("--pile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=460)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stop-after-phase", help="스모크 전용: 이 phase 뒤 계획 종료(발산 아님)")
    ap.add_argument("--max-particles", type=int, help="스모크 전용: 취점 최근접 N 클럼프만. 본 실행 금지")
    ap.add_argument("--numeric-evidence",
                    help="설치본 정적 재구성 수치 증거 JSON(l/voxelSize/좌표상한). **물리 파라미터 아님**. "
                         "없으면 위치 격자 항이 증명 불가라 bridge 인증이 fail-closed 된다")
    ap.add_argument("--max-wall-s", type=float,
                    help="소프트 벽시계 상한(초). 넘으면 step 경계에서 우아하게 멈추고 원시를 finalize 한다. 연장·재시도 없음")
    run(ap.parse_args())
