"""W13 사전검토 ① — DEME 없이 도는 기하·단위·회전·FK/IK·간섭 검사 (rev3).

C1  R_W 무결성 + **오프라인 FK 가 세계축 = 로봇축임을 확인**(이전 '선언'을 실제 FK 로 확정)
C2  mat↔quat↔rotvec 왕복
C3  원문 텍스트 재파싱으로 FK 상수 표류 검사
C4  W11 초기 포즈 재현 — approach 웨이포인트에서 owner 위치/자세가 W11 이 세운 값과 같은가
C5  입(문 립↔고정 립 최소거리)이 베이스각 β 에 불변인가
C6  q_from_pose 역산
C7  공동 분류: W11 최종 입자 517개 재현 + β 회전 후 불변
C8  용기 분류를 **실제 면분할 메시**로 — 이상 원식과의 차이를 수치로 보이고 메시식을 쓴다
C9  고정물 메시 무결성: watertight · winding 일관 · 성분 비겹침 · 부피
C10 **양방향·삼각형 수준 + 보수적 스윕 분리 증명** (툴 정점↔장애물, 장애물 정점↔툴, 이동량 상계)
C11 전 궤적 조밀 샘플의 IK 해와 JOINT_LIMITS
C12 일정·프레임 수·저장량·벽시계(외삽임을 명시)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
MAIN_REPO = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(MAIN_REPO))
sys.path.insert(0, str(HERE))
import sim_deme_scoop_s1 as W11SRC                                       # noqa: E402
import w13_kinematics as K                                               # noqa: E402
import w13_fk as FK                                                      # noqa: E402

W11_DIR = MAIN_REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911"
PILE = Path("/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/"
            "pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz")


def main(out_dir, params_path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P = dict(W11SRC.DEFAULT)
    P.update(K.W13_DEFAULT)
    P.update(json.load(open(params_path)))
    R = {}
    RW = W11SRC.R_W

    # ── 툴 기하 ────────────────────────────────────────────────────────────
    q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
    fixed_m, door_m, lip_f, lip_d, hinge_off, L5mm, mesh_check = W11SRC.load_tool(P, q_open)
    P["lip_l5_mm"] = [float(v) for v in L5mm]
    axis_w = RW @ np.array([0.0, 1.0, 0.0])
    fixed_v = np.asarray(fixed_m.vertices, float)
    door_v = np.asarray(door_m.vertices, float)
    L5_m = np.array(P["lip_l5_mm"], float) / 1000.0
    C5_m = np.array(P["bowl_center_l5_mm"], float) / 1000.0

    # ── C1 ────────────────────────────────────────────────────────────────
    _, R_scoop = FK.lip_pose([0.0, 72.3, 43.19, 64.53, 0.0], P["lip_l5_mm"])
    ref = FK.solve_vertical(0.35, -(0.38 + FK.SHOULDER_ABOVE_PLATE) + 0.26, P["lip_l5_mm"])
    _, R_ref = FK.lip_pose(ref["q5"], P["lip_l5_mm"])
    R["C1_frames"] = {"R_W": RW.tolist(), "det": round(float(np.linalg.det(RW)), 12),
                      "orthonormal_max_err": float(np.abs(RW @ RW.T - np.eye(3)).max()),
                      "fk_link5_axes_at_vertical_scoop_pose": R_ref.tolist(),
                      "fk_vs_R_W_max_abs_diff": float(np.abs(R_ref - RW).max()),
                      "conclusion": "FK 가 준 link5 축(로봇 프레임)이 동결 R_W 와 0.02°(IK 격자 잔차) 안에서 같다. "
                                    "따라서 **yaw = 0 은 동결 R_W 와 자기일관된 선언 장면 규약**이다. "
                                    "이것은 실물 등록(registration) 측정이 아니며 절대 위치를 증명하지 않는다. "
                                    "다른 yaw(예: 90°)를 쓰면 R_W 와 90.000002° 어긋나 자기모순이 된다.",
                      "angle_fk_to_R_W_deg": round(float(np.degrees(np.linalg.norm(K.rotvec_of(RW.T @ R_ref)))), 8),
                      "angle_source": "IK 격자 잔차(물리 기울임 아님). 이 각이 작다는 것이 yaw=0 을 고정한다.",
                      "angle_if_yaw_were_90deg": round(float(np.degrees(np.linalg.norm(
                          K.rotvec_of((K.rot_z(90.0) @ RW).T @ R_ref)))), 6),
                      "pass": bool(abs(np.linalg.det(RW) - 1) < 1e-12
                                   and np.degrees(np.linalg.norm(K.rotvec_of(RW.T @ R_ref))) < 0.1)}

    # ── C2 ────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(13)
    eq, ev = [], []
    for _ in range(2000):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        ang = float(rng.uniform(-179.0, 179.0))
        M = K.axis_angle(ax, ang)
        eq.append(float(np.abs(K.quat_xyzw_to_mat(K.mat_to_quat_xyzw(M)) - M).max()))
        ev.append(abs(float(np.linalg.norm(K.rotvec_of(M))) - abs(math.radians(ang))))
    R["C2_quat_roundtrip"] = {"max_mat_err": max(eq), "max_rotvec_angle_err_rad": max(ev),
                              "pass": bool(max(eq) < 1e-9 and max(ev) < 1e-9)}

    # ── C3 ────────────────────────────────────────────────────────────────
    R["C3_source_constants"] = FK.verify_source_constants()

    # ── 어댑터·웨이포인트 ──────────────────────────────────────────────────
    pile = np.load(PILE, allow_pickle=True)
    box = np.asarray(pile["box_bounds_m"], float)
    box_top = float(box[2, 1])
    z_surf_pre = 0.0411807760
    z_lip0 = z_surf_pre + P["approach_gap_mm"] / 1000.0
    ad, ad_info = FK.build_adapter(box, z_surf_pre, P["arm_radius_m"], P["lip_l5_mm"],
                                 P["declared_base_cm"], P["declared_pellet_cm"])
    r0 = float(np.hypot(*FK.lip_pose(ad_info["reference_pose_q5"], P["lip_l5_mm"])[0][:2]))
    z_travel = float(ad_info["floor_robot_z_m"] + P["travel_cm"] / 100.0 - ad_info["t_robot_m"][2])
    z_surf5 = float(ad_info["pellet_robot_z_m"] + 0.05 - ad_info["t_robot_m"][2])
    tray = K.tray_mesh(box, P["tray_wall_t_mm"] / 1000.0)
    wps0, _ = FK.build_waypoints(ad, r0, z_travel, z_lip0, z_surf5, 0.0, P["place_base_deg"])
    bin_center = np.asarray([w for w in wps0 if w["name"] == "place_target"][0]["lip_world_m"])[:2]
    binm, bin_info = K.bin_mesh(P, bin_center)
    z_release = bin_info["rim_z_m"] + P["release_clearance_mm"] / 1000.0
    wps, wp_extra = FK.build_waypoints(ad, r0, z_travel, z_lip0, z_surf5, z_release, P["place_base_deg"])
    Wp = {w["name"]: w for w in wps}
    R["adapter"] = dict(ad_info, **{k: v for k, v in wp_extra.items() if k != "r0_robot_m"},
                        r0_robot_m=r0, z_travel_m=z_travel, z_release_m=z_release,
                        z_lip0_m=z_lip0, box_top_m=box_top, bin_center_m=bin_center.tolist())
    R["waypoints"] = FK.waypoint_report(wps)

    # ── C4 ────────────────────────────────────────────────────────────────
    p_ap = np.asarray(Wp["approach_gap"]["lip_world_m"], float)
    R_ap = Wp["approach_gap"]["R_owner"]
    R["C4_w11_initial_pose"] = {
        "w11_lip_world_m": [0.0, 0.0, 0.051180776],
        "w13_approach_lip_world_m": p_ap.tolist(),
        "lip_xy_err_mm": round(float(np.linalg.norm(p_ap[:2])) * 1000, 6),
        "lip_z_err_mm": round(abs(p_ap[2] - 0.051180776) * 1000, 6),
        "R_owner_vs_identity_max": float(np.abs(R_ap - np.eye(3)).max()),
        "hinge_offset_m": hinge_off.tolist(),
        "w11_hinge_offset_m": [-0.018821, 0.0081, 0.117565],
        "hinge_offset_err_m": float(np.abs(hinge_off - np.array([-0.018821, 0.0081, 0.117565])).max()),
        "pass": bool(np.linalg.norm(p_ap[:2]) < 1e-4 and abs(p_ap[2] - 0.051180776) < 1e-4
                     and np.abs(R_ap - np.eye(3)).max() < 1e-3)}

    # ── C5 / C6 ───────────────────────────────────────────────────────────
    from scipy.spatial import cKDTree

    def pose(beta, q, z):
        Rt = K.rot_z(beta)
        p_f = np.array([0.0, 0.0, z], float)
        return p_f, Rt, p_f + Rt @ hinge_off, Rt @ K.axis_angle(axis_w, q - q_open)

    rows = []
    for q in (q_open, 12.0, 3.0, 0.0):
        vals = []
        for b in (0.0, 45.0, 90.0):
            pf, Rf, pd, Rd = pose(b, q, 0.15)
            F = (Rf @ fixed_v[lip_f].T).T + pf
            Dd = (Rd @ door_v[lip_d].T).T + pd
            vals.append(float(cKDTree(F).query(Dd)[0].min()) * 1000)
        rows.append({"q_deg": q, "mouth_mm": [round(v, 9) for v in vals],
                     "max_abs_diff_mm": round(max(abs(v - vals[0]) for v in vals), 12)})
    R["C5_rotation_invariant_mouth"] = {"rows": rows, "w11_mouth_start_mm": 56.02,
                                        "pass": bool(max(r["max_abs_diff_mm"] for r in rows) < 1e-9)}

    def q_from_pose(R_fa, R_da):
        rv = K.rotvec_of(R_fa.T @ R_da)
        ang = float(np.linalg.norm(rv))
        if ang < 1e-12:
            return q_open
        return q_open + (1.0 if float(np.dot(rv, axis_w)) >= 0 else -1.0) * math.degrees(ang)

    e = [abs(q_from_pose(pose(b, q, 0.15)[1], pose(b, q, 0.15)[3]) - q)
         for b in (0.0, 17.3, 45.0, 90.0) for q in (0.0, 1.0, 3.076, 12.0, q_open)]
    R["C6_q_inverse"] = {"max_err_deg": max(e), "pass": bool(max(e) < 1e-9)}

    # ── C7 ────────────────────────────────────────────────────────────────
    w11 = np.load(W11_DIR / "cell_dt1e6_seed460/scoop_s1_seed460.npz", allow_pickle=True)
    w11j = json.load(open(W11_DIR / "cell_dt1e6_seed460/scoop_s1_seed460.json"))
    pp = np.asarray(w11["positions_m"], float)
    in_cav_w11 = np.asarray(w11["in_cavity"], bool)
    lip_w11 = np.array([0.0, 0.0, w11j["scoop_site"]["lip_z_end_mm"] / 1000.0])
    L5_w = np.array(w11j["params"]["lip_l5_mm"], float) / 1000.0
    C5_w = np.array(w11j["params"]["bowl_center_l5_mm"], float) / 1000.0

    def cav(points, p_fa, R_fa):
        p5 = (RW.T @ (R_fa.T @ (points - p_fa).T)).T + L5_w
        return (np.hypot(p5[:, 0] - C5_w[0], p5[:, 2] - C5_w[2]) < P["bowl_r_in_mm"] / 1000.0) & \
               (np.abs(p5[:, 1]) < P["cheek_half_y_mm"] / 1000.0)

    m_id = cav(pp, lip_w11, np.eye(3))
    Rt = K.rot_z(90.0)
    piv = np.array([0.05, -0.02, 0.0])
    m_rot = cav((Rt @ (pp - piv).T).T + piv, Rt @ (lip_w11 - piv) + piv, Rt)
    R["C7_cavity_classification"] = {"n_w11_json": int(w11j["capture"]["n_in_cavity"]),
                                     "n_w11_npz": int(in_cav_w11.sum()), "n_w13_identity": int(m_id.sum()),
                                     "diff_vs_w11_npz": int((m_id != in_cav_w11).sum()),
                                     "n_w13_rotated": int(m_rot.sum()),
                                     "diff_identity_vs_rotated": int((m_id != m_rot).sum()),
                                     "pass": bool((m_id != in_cav_w11).sum() == 0 and (m_id != m_rot).sum() == 0)}

    # ── C8 면분할 용기 분류 ────────────────────────────────────────────────
    n_th = int(P["bin_n_theta"])
    th = np.linspace(0, 2 * math.pi, n_th, endpoint=False) + math.pi / n_th
    nrm = np.stack([np.cos(th), np.sin(th)], 1)
    apo = bin_info["inner_r_m"] * math.cos(math.pi / n_th)
    probe = bin_center + np.stack([np.cos(np.linspace(0, 2 * math.pi, 720)),
                                   np.sin(np.linspace(0, 2 * math.pi, 720))], 1) * bin_info["inner_r_m"] * 0.999
    ideal_in = np.hypot(probe[:, 0] - bin_center[0], probe[:, 1] - bin_center[1]) < bin_info["inner_r_m"]
    facet_in = ((probe - bin_center) @ nrm.T).max(1) < apo
    R["C8_faceted_bin_classifier"] = {
        "n_theta": n_th, "circumradius_m": bin_info["inner_r_m"], "apothem_m": apo,
        "max_radial_gap_mm": round((bin_info["inner_r_m"] - apo) * 1000, 5),
        "probe_points": int(len(probe)), "ideal_circle_says_inside": int(ideal_in.sum()),
        "faceted_mesh_says_inside": int(facet_in.sum()),
        "disagreement": int((ideal_in != facet_in).sum()),
        "decision": "분류는 이상 원식이 아니라 메시와 같은 면분할 내접 반평면(apothem)을 쓴다. "
                    "경계 ±classify_margin_mm 밴드는 ambiguous 로 남긴다.",
        "classify_margin_mm": P["classify_margin_mm"], "particle_bounding_radius_mm": 2.2504,
        "pass": bool(apo < bin_info["inner_r_m"] and P["classify_margin_mm"] > 2.2504)}

    # ── C9 메시 무결성 ────────────────────────────────────────────────────
    def mesh_report(m, name):
        comps = m.split(only_watertight=False)
        return {"name": name, "n_tri": int(len(m.faces)), "n_components": len(comps),
                "watertight": bool(m.is_watertight), "winding_consistent": bool(m.is_winding_consistent),
                "volume_cm3": round(float(m.volume) * 1e6, 4) if m.is_watertight else None,
                "euler_number": int(m.euler_number),
                "components_watertight": [bool(c.is_watertight) for c in comps],
                "components_winding_consistent": [bool(c.is_winding_consistent) for c in comps],
                "bounds_m": m.bounds.tolist()}

    tr_r, bn_r = mesh_report(tray, "tray"), mesh_report(binm, "bin")
    # 성분 겹침(트레이 4벽 · 용기 2성분) — AABB 교차 부피로 본다
    def overlap_volume(m):
        comps = m.split(only_watertight=False)
        tot = 0.0
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = comps[i].bounds, comps[j].bounds
                lo = np.maximum(a[0], b[0]); hi = np.minimum(a[1], b[1])
                d = np.clip(hi - lo, 0, None)
                tot += float(np.prod(d))
        return tot * 1e6
    tr_r["component_aabb_overlap_cm3"] = round(overlap_volume(tray), 6)
    bn_r["component_aabb_overlap_cm3"] = round(overlap_volume(binm), 6)
    R["C9_fixture_mesh_integrity"] = {"tray": tr_r, "bin": bn_r,
                                      "pass": bool(tr_r["watertight"] and tr_r["winding_consistent"]
                                                   and bn_r["watertight"] and bn_r["winding_consistent"]
                                                   and tr_r["component_aabb_overlap_cm3"] < 1e-9
                                                   and bn_r["component_aabb_overlap_cm3"] < 1e-9)}

    # ── C10 양방향·삼각형 수준 + 보수적 스윕 분리 ─────────────────────────
    order = ["initial_home", "p1_tool_vertical", "above_pile_travel", "surface_plus_50mm", "approach_gap",
             "post_lift_travel", "place_retract_base0", "place_retract_base90", "place_extend_travel",
             "place_target", "place_up_travel", "return_retract_base90", "return_retract_base0", "return_home"]
    N_SEG = int(P.get("preflight_segment_samples", 60))
    poses = []
    for k in range(len(order) - 1):
        q0 = np.asarray(Wp[order[k]]["q5"], float)
        q1 = np.asarray(Wp[order[k + 1]]["q5"], float)
        for t in np.linspace(0, 1, N_SEG, endpoint=(k == len(order) - 2)):
            poses.append((f"{order[k]}->{order[k+1]}", (q0 + (q1 - q0) * t).tolist()))
    # 하강/상승 구간(직교 수직)도 추가
    for t in np.linspace(0, 1, N_SEG):
        poses.append(("descend_lift_column", Wp["approach_gap"]["q5"]))
    dz_extra = np.linspace(0.0, -(P["plunge_mm"] / 1000.0), N_SEG).tolist() + \
        np.linspace(-(P["plunge_mm"] / 1000.0), -(P["plunge_mm"] / 1000.0) + P["lift_mm"] / 1000.0, N_SEG).tolist()

    obst = {"tray": tray, "bin": binm}
    Vt = np.vstack([fixed_v, door_v + hinge_off])
    frames = []
    for nm, q5 in poses:
        p, Rm = ad.owner_pose(q5)
        frames.append((nm, (Rm @ Vt.T).T + p))
    # 취점 구간은 동결 W11 포즈(립 (0,0,z), 자세 항등)로 따로 샘플한다
    for dz in dz_extra:
        frames.append(("w11_scoop_column", Vt + np.array([0.0, 0.0, z_lip0 + dz])))

    # 감사 권고: 껍데기 bbox 가 아니라 **실제 고체 성분별로** 따로 검사한다
    obst_split = {}
    for nm_, mesh_ in obst.items():
        comps = mesh_.split(only_watertight=False)
        for ci_, cm_ in enumerate(comps):
            obst_split[f"{nm_}_component{ci_}"] = cm_
    obst.update(obst_split)
    c10 = {"n_sampled_poses": len(frames), "component_wise": True, "obstacles": {}}
    for name, mesh in obst.items():
        pq = trimesh.proximity.ProximityQuery(mesh)
        mins, inside, at = [], 0, None
        for i, (nm, V) in enumerate(frames):
            d = pq.signed_distance(V)                     # >0 = 메시 내부
            dmin = float(np.abs(d).min())
            n_in = int((d > 0).sum())
            inside += n_in
            mins.append(dmin)
            if at is None or dmin < at[1]:
                at = (nm, dmin, i)
        # 역방향: 장애물 정점 ↔ 툴(최근접 포즈에서)
        Vw = frames[at[2]][1]
        back = float(trimesh.proximity.ProximityQuery(
            trimesh.Trimesh(Vw, np.vstack([np.asarray(fixed_m.faces),
                                           np.asarray(door_m.faces) + len(fixed_v)]), process=False)
        ).signed_distance(np.asarray(mesh.vertices)).max())
        # 스텝 변위는 **같은 세그먼트 안에서만** 잰다(세그먼트 사이 건너뛰기는 실제 경로가 아니다)
        steps = [float(np.linalg.norm(frames[i + 1][1] - frames[i][1], axis=1).max())
                 for i in range(len(frames) - 1) if frames[i + 1][0] == frames[i][0]]
        step_disp = max(steps) if steps else 0.0
        c10["obstacles"][name] = {
            "min_abs_distance_mm": round(min(mins) * 1000, 5), "at_segment": at[0],
            "tool_vertices_inside_obstacle": inside,
            "obstacle_vertices_max_signed_distance_into_tool_mm": round(back * 1000, 5),
            "max_pose_to_pose_vertex_move_mm": round(step_disp * 1000, 5),
            "conservative_swept_separation_ok": bool(min(mins) > step_disp),
            "rule": "연속 두 샘플 사이 어떤 정점도 step_disp 이상 움직이지 않으므로, 두 끝점의 최소거리가 "
                    "step_disp 보다 크면 그 사이 구간에도 교차가 없다(보수적 상계)."}
    c10["pass"] = all(v["tool_vertices_inside_obstacle"] == 0 and v["obstacle_vertices_max_signed_distance_into_tool_mm"] < 0
                      and v["conservative_swept_separation_ok"] for v in c10["obstacles"].values())
    R["C10_clearance"] = c10

    # ── C11a 한계 출처 명시(활성 S1v1 URDF) ───────────────────────────────
    import re as _re
    lim = {}
    for nm, path in (("s1_v1_active", MAIN_REPO / "local_assets/roarm_m3/urdf/roarm_m3_s1_v1.urdf"),
                     ("generic_m3", MAIN_REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf")):
        txt = path.read_text()
        got = {}
        for jn, key in (("base_link_to_link1", "base"), ("link1_to_link2", "shoulder"), ("link2_to_link3", "elbow"),
                        ("link3_to_link4", "wrist_p"), ("link4_to_link5", "wrist_r"),
                        ("link5_to_gripper_link", "gripper")):
            i = txt.index(f'<joint name="{jn}"')
            m = _re.search(r'<limit lower="([-\d.]+)" upper="([-\d.]+)" effort="([\d.]+)"', txt[i:i + 900])
            got[key] = {"lower_deg": round(math.degrees(float(m.group(1))), 4),
                        "upper_deg": round(math.degrees(float(m.group(2))), 4), "effort_Nm": float(m.group(3))}
        lim[nm] = {"path": str(path), "limits": got}
    same = all(lim["s1_v1_active"]["limits"][k] == lim["generic_m3"]["limits"][k] for k in lim["generic_m3"]["limits"])
    R["C11a_limit_sources"] = {
        "A_active_urdf": lim["s1_v1_active"], "A_generic_urdf_for_comparison": lim["generic_m3"],
        "active_equals_generic": bool(same),
        "version_note": "활성 도구 URDF 는 roarm_m3_s1_v1.urdf 다. 관절 한계는 일반 roarm_m3.urdf 와 **동일**하므로 버전 불일치가 없다.",
        "B_v6_distribution_clips_deg": {k: list(v) for k, v in FK.JOINT_LIMITS_DEG.items()},
        "B_note": "sim_scripts/roarm_kinematics.py:29-36 — v6 학습 분포 유지용 클립. 물리 한계가 아니다.",
        "C_operational_guard": {"wrist_p_symmetric_guard_deg": [-FK.WRIST_MAX, FK.WRIST_MAX],
                                "measured_firmware_clamp": "+90 (09-07 실측, hw_s1_scoop_probe.py:21 주석)",
                                "qualification": "코드가 쓰는 |wrist_p| <= 90 은 **대칭 운용 가드**다. 실측된 것은 +90 쪽 "
                                                 "펌웨어 클램프이며 −90 쪽 대칭성은 측정으로 확인된 바 없다.",
                                "door_max_servo_deg": P["door_open_servo_deg"]},
        "pass": bool(same)}

    # ── C11 IK / JOINT_LIMITS ─────────────────────────────────────────────
    viol, worst = [], 0.0
    for nm, q5 in poses:
        bad = FK.in_limits(q5)
        if bad:
            viol.append({"segment": nm, "q5": [round(v, 3) for v in q5], "violations": bad})
    col_ik = []
    for dz in dz_extra:
        zr = float(np.asarray(ad.to_robot([0, 0, float(Wp["approach_gap"]["lip_world_m"][2]) + dz]))[2])
        sol = FK.solve_fast(r0, zr, P["lip_l5_mm"])
        col_ik.append({"z_world_mm": round((float(Wp["approach_gap"]["lip_world_m"][2]) + dz) * 1000, 3),
                       "err_mm": None if sol is None else round(sol["err_m"] * 1000, 4),
                       "branch": None if sol is None else sol["branch"],
                       "limits": None if sol is None else sol["limits_violations"]})
        worst = max(worst, 1e9 if sol is None else sol["err_m"] * 1000)
    R["C11_ik_limits"] = {"n_interpolated_poses": len(poses), "limit_violations": viol,
                          "descend_lift_column_ik": col_ik, "worst_column_ik_err_mm": round(worst, 4),
                          "waypoint_ik": [{"name": w["name"], "branch": w["ik"].get("branch"),
                                           "err_mm": None if w["ik"]["err_m"] is None else round(w["ik"]["err_m"] * 1000, 4),
                                           "tilt_deg": None if w["ik"].get("tilt_deg") is None else round(w["ik"]["tilt_deg"], 3),
                                           "limits": w["ik"]["limits_violations"]} for w in wps],
                          "pass": bool(not viol and worst < 1.0)}

    # ── C12 일정·저장량 ───────────────────────────────────────────────────
    v_t = P["transport_speed_mm_s"] / 1000.0
    seg_s, total_transit = {}, 0.0
    for k in range(len(order) - 1):
        p0, _ = ad.owner_pose(Wp[order[k]]["q5"])
        p1, _ = ad.owner_pose(Wp[order[k + 1]]["q5"])
        d = float(np.linalg.norm(p1 - p0))
        seg_s[f"{order[k]}->{order[k+1]}"] = round(d / v_t, 6)
        total_transit += d / v_t
    nominal = dict(seg_s)
    nominal["initial_home_hold"] = P["home_hold_s"]
    nominal["settle"] = P["settle_steps"] * P["dt_sync_s"]
    nominal["descend_w11_actual"] = 1.400350
    nominal["close_w11_actual"] = 1.069571
    nominal["lift"] = P["lift_mm"] / 1000.0 / (P["lift_mm_s"] / 1000.0)
    nominal["reclose_w11_actual"] = 0.018685
    nominal["discharge_open"] = (q_open - 3.0) / P["close_deg_s"]
    nominal["discharge_wait"] = P["discharge_hold_s"]
    nominal["close_after_discharge"] = (q_open - 3.0) / P["close_deg_s"]
    nominal["return_home_hold"] = P["home_hold_s"]
    total = float(sum(nominal.values()))
    w11_rate = 2730.0589 / 3.141871
    n_pf = int(math.ceil(total / P["particle_frame_dt_s"])) + 40
    n_part = int(np.asarray(pile["clump_positions_m"]).shape[0])
    R["C12_schedule"] = {
        "nominal_s": {k: round(v, 6) for k, v in nominal.items()}, "total_sim_s": round(total, 6),
        "transit_total_s": round(total_transit, 6),
        "w11_measured_wall_per_sim_s": round(w11_rate, 3),
        "EXTRAPOLATED_wall_estimate_h": round(total * w11_rate / 3600.0, 2),
        "estimate_basis": "⚠ W11 실측 비율의 선형 외삽이다. 승인된 dt 1e-6 회귀 스모크 실측으로 갱신해야 한다.",
        "particle_frames_estimate": n_pf, "n_particles": n_part,
        "particle_bytes_estimate_MB": round(n_pf * n_part * (3 * 4 + 4 * 4 + 3 * 4 + 3 * 4 + 1) / 1e6, 1),
        "dense_sync_estimate": int(total / P["dt_sync_s"]) + 6000,
        "pass": True}

    R["fixtures"] = K.dump_fixture_json(out / "fixture_01.json", P, box, bin_info, tray,
                                        {"z_travel_m": z_travel, "z_release_m": z_release, "z_lip0_m": z_lip0,
                                         "transport_speed_m_s": v_t, "place_base_deg": P["place_base_deg"]},
                                        type("F", (), {"R": P["arm_radius_m"],
                                                       "base_xy": np.asarray(wp_extra["base_axis_world_xy_m"])})(),
                                        extra={"mesh_check_from_frozen_source": mesh_check,
                                               "adapter": ad_info, "waypoints": FK.waypoint_report(wps)})
    R["all_pass"] = all(v.get("pass", True) for v in R.values() if isinstance(v, dict))
    json.dump(R, open(out / "preflight_geometry.json", "w"), ensure_ascii=False, indent=2, default=float)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in R.items()},
                     ensure_ascii=False, indent=1))
    print(f"총 명목 물리시간 {total:.4f} s · 선형외삽 벽시계 {total*w11_rate/3600:.2f} h(실측 아님) · "
          f"입자 프레임 {n_pf} (~{R['C12_schedule']['particle_bytes_estimate_MB']} MB)")
    print(f"-> {out/'preflight_geometry.json'}")
    return R


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
