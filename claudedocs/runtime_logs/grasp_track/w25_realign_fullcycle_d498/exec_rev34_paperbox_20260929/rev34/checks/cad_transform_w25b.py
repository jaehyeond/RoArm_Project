"""W25-A §2-(5) — S1 v1 CAD(link5 mm) ↔ DEME owner 강체 변환 문서화 + W19 A sync 0 셸 노드 재현 (CPU, 읽기 전용).

변환 (유도는 REPORT.md §5; 여기서는 수치로 검산만 한다)
  고정 owner 원점 L5 = (8.1, 0, z_lip_thick) mm (link5), 문 owner 원점 H5 = hinge_l5_mm.
  고정부:  v_w = R_f · R_W · (v_l5 − L5)/1000 + p_f
  문(관절각 q, 0 = 립 맞닿음):  v_w = R_f · R_W · [roty(q)·(v_l5 − H5) + (H5 − L5)]/1000 + p_f
           = R_d · R_W · roty(q_open)·(v_l5 − H5)/1000 + p_d    (R_d = R_f·axis_angle(axis_w, q−q_open), p_d = p_f + R_f·hinge_off)
  로봇 좌표: v_r = R_robot_box · v_w + t_robot  = T_link5(q5) · v_l5   (FK owner 포즈를 쓰면 정확히 FK link5 로 환원)

usage: python cad_transform_w25b.py <W19 A run dir> <out json>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
MAIN = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(SRC))
import sim_deme_scoop_s1 as W11SRC                                       # noqa: E402
import w13_kinematics as K                                               # noqa: E402
import w13_fk as FK                                                      # noqa: E402


def main(run, out_path):
    run = Path(run)
    res = json.load(open(run / "w13_cycle_seed460.json"))
    z = np.load(run / "w13_cycle_seed460.npz", allow_pickle=True)
    P = dict(W11SRC.DEFAULT)
    P.update(res["params"])
    q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
    RW = W11SRC.R_W
    Fm, Dm = W11SRC.half_bowl(P, -1), W11SRC.half_bowl(P, +1)
    fv, dv = np.asarray(Fm.vertices, float), np.asarray(Dm.vertices, float)
    L5 = np.array([P["bowl_center_l5_mm"][0], 0.0, fv[:, 2].max()])
    H5 = np.array(P["hinge_l5_mm"], float)
    fixed_m, door_m, _, _, hinge_off, L5_src, _ = W11SRC.load_tool(P, q_open)
    axis_w = RW @ np.array([0.0, 1.0, 0.0])

    def fixed_world(v_l5, p_f, R_f):
        return (R_f @ RW @ ((v_l5 - L5) / 1000.0).T).T + p_f

    def door_world(v_l5, p_f, R_f, q):
        return (R_f @ RW @ ((W11SRC.roty(q) @ (v_l5 - H5).T).T + (H5 - L5)).T / 1000.0).T + p_f

    i = 0
    nF = np.asarray(z["nodes_F_m"][i], float)
    nD = np.asarray(z["nodes_D_m"][i], float)
    p_f = np.asarray(z["tool_pos_m"][i], float)
    R_f = K.quat_xyzw_to_mat(np.asarray(z["tool_quat_xyzw"][i], float))
    q_i = float(z["door_actual_deg"][i])
    p_d = np.asarray(z["door_pos_m"][i], float)
    R_d = K.quat_xyzw_to_mat(np.asarray(z["door_quat_xyzw"][i], float))

    def err(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape == b.shape:
            same = float(np.linalg.norm(a - b, axis=1).max()) * 1000
        else:
            same = None
        from scipy.spatial import cKDTree
        nn = float(cKDTree(b).query(a)[0].max()) * 1000
        return {"same_order_max_mm": same, "nearest_max_mm": nn, "n_model": int(len(a)), "n_npz": int(len(b))}

    # ① 추적 포즈(원시 tool_* · door_actual_deg) 로 재현
    rec_tracker = {"fixed": err(fixed_world(fv, p_f, R_f), nF), "door": err(door_world(dv, p_f, R_f, q_i), nD),
                   "door_via_door_owner": err((R_d @ RW @ ((W11SRC.roty(q_open) @ (dv - H5).T).T).T / 1000.0).T + p_d, nD)}
    # ② 실물 FK(HOME, 결과 JSON 어댑터) 로 재현 — 추적기와 독립
    ad = FK.Adapter(res["frames"]["adapter"]["t_robot_m"], P["lip_l5_mm"])
    p_h, R_h = ad.owner_pose(FK.HOME_Q5)
    rec_fk = {"fixed": err(fixed_world(fv, p_h, R_h), nF), "door": err(door_world(dv, p_h, R_h, 0.0), nD)}
    # ③ 로봇 좌표 환원 검산: v_r = T_link5·v_l5 (FK) vs 어댑터 경유
    T = FK.link5_T(FK.HOME_Q5)
    v_r_fk = (T[:3, :3] @ (fv / 1000.0).T).T + T[:3, 3] - np.array([0, 0, FK.SHOULDER_ABOVE_PLATE])
    v_r_ad = np.array([ad.to_robot(v) for v in fixed_world(fv, p_h, R_h)])
    red = float(np.linalg.norm(v_r_fk - v_r_ad, axis=1).max()) * 1000

    # ④ STL 원본(link5 mm) — 충돌 셸과 같은 공동에 있는가, 문 STL 은 어느 각으로 모델링됐나
    S1 = W11SRC.S1
    stl = {}
    for nm in ("fixed_ALL.stl", "door_ALL.stl", "door_ALL_jawframe.stl"):
        m = trimesh.load(S1 / nm)
        v = np.asarray(m.vertices, float)
        C = P["bowl_center_l5_mm"]
        bowl = v[(v[:, 2] > 124) & (np.abs(v[:, 1]) < P["cheek_half_y_mm"])]
        row = {"n_tri": int(len(m.faces)), "n_vert": int(len(v)), "bounds_l5_mm": m.bounds.round(3).tolist(),
               "sha256_16": W11SRC.sha16(S1 / nm)}
        if nm.startswith("door"):
            best = None
            for th in np.arange(-5.0, 35.01, 0.25):
                vv = (W11SRC.roty(-th) @ (bowl - H5).T).T + H5           # th 로 열린 모델이면 되돌리면 공동 반경이 r_in
                r = np.hypot(vv[:, 0] - C[0], vv[:, 2] - C[2])
                inner = r[r < P["bowl_r_in_mm"] + 0.8]
                sc = float(np.abs(inner.min() - P["bowl_r_in_mm"])) if inner.size else 1e9
                n_on = int((np.abs(r - P["bowl_r_in_mm"]) < 0.3).sum())
                if best is None or n_on > best["n_vertices_within_0p3mm_of_r_in"]:
                    best = {"theta_deg": round(float(th), 3), "n_vertices_within_0p3mm_of_r_in": n_on,
                            "r_min_minus_r_in_mm": round(sc, 4)}
            row["modeled_open_angle_search"] = best
        r0 = np.hypot(bowl[:, 0] - C[0], bowl[:, 2] - C[2])
        row["bowl_region_r_min_mm"] = round(float(r0.min()), 4) if bowl.size else None
        row["n_bowl_vertices_within_0p3mm_of_r_in"] = int((np.abs(r0 - P["bowl_r_in_mm"]) < 0.3).sum())
        stl[nm] = row

    out = {"artifact": "W25A_CAD_PLACEMENT_TRANSFORM",
           "run": str(run), "sync_index": i, "door_actual_deg_sync0": q_i,
           "constants": {"L5_fixed_owner_origin_l5_mm": L5.tolist(), "L5_from_load_tool_mm": np.asarray(L5_src).tolist(),
                         "H5_door_owner_origin_l5_mm": H5.tolist(), "R_W": RW.tolist(),
                         "hinge_off_owner_m": np.asarray(hinge_off).tolist(), "axis_w_owner": axis_w.tolist(),
                         "q_open_joint_deg": q_open, "shell_tri": [int(len(Fm.faces)), int(len(Dm.faces))]},
           "formulas": {
               "fixed": "v_w = R_f · R_W · (v_l5 − L5)/1000 + p_f",
               "door": "v_w = R_f · R_W · [roty(q)·(v_l5 − H5) + (H5 − L5)]/1000 + p_f",
               "door_owner_form": "v_w = R_d · R_W · roty(q_open)·(v_l5 − H5)/1000 + p_d",
               "link5_in_deme_4x4": "T_w_l5 = [[R_f·R_W, p_f − R_f·R_W·L5/1000],[0,0,0,1]]  (mm→m 는 v_l5/1000)",
               "robot": "v_r = R_robot_box · v_w + t_robot = T_link5(q5)·v_l5 − (0,0,SHOULDER_ABOVE_PLATE)",
               "rev34_note": "rev34 규약 A 에서는 R_robot_box = Rz(−90°) 이고 FK owner 회전이 Rᵀ·R_l5·R_Wᵀ 라 R_f·R_W = Rᵀ·R_l5."},
           "reproduce_sync0_from_tracker_pose_mm": rec_tracker,
           "reproduce_sync0_from_fk_home_mm": rec_fk,
           "robot_frame_reduction_max_mm": red,
           "stl_sources": stl,
           "non_claims": ["충돌 셸(half_bowl)은 STL 을 단순화·두껍게 한 것이다 — STL 을 이 변환에 넣으면 **표시 위치**가 맞을 뿐 충돌면은 아니다.",
                          "nodes_*_m 은 float32 저장이라 재현 오차 하한은 ~1e-5 mm 급이다."]}
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in ("reproduce_sync0_from_tracker_pose_mm", "reproduce_sync0_from_fk_home_mm",
                                          "robot_frame_reduction_max_mm")}, ensure_ascii=False, indent=1))
    print(json.dumps({k: {kk: v.get(kk) for kk in ("n_tri", "bowl_region_r_min_mm", "modeled_open_angle_search",
                                                  "n_bowl_vertices_within_0p3mm_of_r_in")} for k, v in stl.items()},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
