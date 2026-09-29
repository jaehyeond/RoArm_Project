"""W25-A 후속 — 높은 트레이 벽(종이 상자 230 mm)에 대한 **팔 링크·S1 전체 STL** 간섭 사전검토 (CPU, 읽기 전용).

preflight C10 은 충돌 셸(half_bowl)만 본다. 벽이 바닥판보다 높아지면 팔뚝·손목·S1 몸체(스파인·뺨·서보)가
벽 윗단에 걸릴 수 있어 따로 본다. 판정 정본이 아니라 **보고용 사전검토**다.

몸체
  · 팔 링크 base_link, link1..link5 = 설치 URDF(roarm_m3_s1_v1) visual STL 의 local AABB(`arm_link_bounds`) 를
    FK 로 옮긴 OBB — 메시보다 크거나 같다(보수적). 겹침 = "가능성", 여유 = 하한 추정.
  · S1 v1 전체 STL(link5 mm): fixed_ALL.stl + door_ALL.stl(문 q 를 roty 로 회전). 정점 기준.
장애물 = 트레이 벽 4장(`bridge_inputs.tray_cells` 와 같은 식, 상자 좌표 → 로봇 좌표).
자세 = preflight C10/C11 과 같은 웨이포인트 관절 보간(구간당 60) + 취점 기둥(실물 IK, 문 닫힘/열림 둘 다).
usage: python arm_wall_check_w25.py <params> <pile npz> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, "/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(SRC))
import sim_deme_scoop_s1 as W11SRC                                       # noqa: E402
import w13_fk as FK                                                      # noqa: E402
import w13_kinematics as K                                               # noqa: E402
import arm_link_bounds as AB                                             # noqa: E402
import bridge_inputs as BR                                               # noqa: E402

ORDER = ["initial_home", "p1_tool_vertical", "above_pile_travel", "surface_plus_50mm", "approach_gap",
         "post_lift_travel", "place_retract_base0", "place_retract_base90", "place_extend_travel",
         "place_target", "place_up_travel", "return_retract_base90", "return_retract_base0", "return_home"]


def aabb_dist(pts, lo, hi):
    d = np.maximum(np.maximum(lo - pts, 0.0), pts - hi)
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    return np.linalg.norm(d, axis=1), inside


def obb_vs_aabb(corners, lo, hi):
    """SAT: OBB(8 모서리) vs AABB 겹침. 축 = AABB 3 + OBB 3 + 교차 9."""
    c = np.asarray(corners, float)
    ctr = c.mean(0)
    e = [c[1] - c[0], c[2] - c[0], c[4] - c[0]]          # load_link_local_bounds 의 i&1,i&2,i&4 순서
    ax_o = [v / np.linalg.norm(v) for v in e if np.linalg.norm(v) > 1e-12]
    axes = [np.eye(3)[i] for i in range(3)] + ax_o + [np.cross(a, b) for a in np.eye(3) for b in ax_o]
    bc = np.array([[lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1], lo[2] if i & 4 else hi[2]]
                   for i in range(8)])
    for a in axes:
        n = np.linalg.norm(a)
        if n < 1e-12:
            continue
        a = a / n
        p1, p2 = c @ a, bc @ a
        if p1.max() < p2.min() or p2.max() < p1.min():
            return False
    del ctr
    return True


def obb_samples(corners, k=6):
    c = np.asarray(corners, float)
    o, e1, e2, e3 = c[0], c[1] - c[0], c[2] - c[0], c[4] - c[0]
    t = np.linspace(0, 1, k)
    g = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
    return o + g[:, :1] * e1 + g[:, 1:2] * e2 + g[:, 2:] * e3


def main(params_path, pile_path, out_path):
    P = dict(W11SRC.DEFAULT)
    P.update(K.W13_DEFAULT)
    P.update(json.load(open(params_path)))
    z = np.load(pile_path, allow_pickle=True)
    box, tray_info = FK.w25_tray_bounds(np.asarray(z["box_bounds_m"], float), P, fail_closed=False)
    q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
    L5mm = W11SRC.load_tool(P, q_open)[5]
    P["lip_l5_mm"] = [float(v) for v in L5mm]
    site_xy, site = FK.w25_scoop_site(P, P["lip_l5_mm"])
    s_xy = (P["scoop_x_mm"] / 1000.0, 0.0) if site.get("rev32_path") else site_xy
    z_surf = W11SRC.surface_z(np.asarray(z["positions_m"], float), np.asarray(z["radii_m"], float), *s_xy, P)
    gap = P["w25_door_open_gap_mm"] if P.get("w25_proc_open_at_surface") else P["approach_gap_mm"]
    z_lip0 = z_surf + gap / 1000.0
    ad, info, fr = FK.build_adapter_w25(box, z_surf, P, P["lip_l5_mm"])
    r0 = float(np.hypot(*FK.lip_pose(info["reference_pose_q5"], P["lip_l5_mm"])[0][:2]))
    z_travel = float(info["floor_robot_z_m"] + P["travel_cm"] / 100.0 - info["t_robot_m"][2])
    z_surf5 = float(info["pellet_robot_z_m"] + 0.05 - info["t_robot_m"][2])
    wps0, _ = FK.build_waypoints(ad, r0, z_travel, z_lip0, z_surf5, 0.0, P["place_base_deg"])
    bc = np.asarray([w for w in wps0 if w["name"] == "place_target"][0]["lip_world_m"])[:2]
    _, bin_info = K.bin_mesh(P, bc)
    z_rel = bin_info["rim_z_m"] + P["release_clearance_mm"] / 1000.0
    wps, _ = FK.build_waypoints(ad, r0, z_travel, z_lip0, z_surf5, z_rel, P["place_base_deg"])
    W = {w["name"]: w for w in wps}

    poses = []                                          # (segment, q5, door_q_list)
    for k in range(len(ORDER) - 1):
        q0, q1 = np.asarray(W[ORDER[k]]["q5"], float), np.asarray(W[ORDER[k + 1]]["q5"], float)
        for t in np.linspace(0, 1, 60, endpoint=(k == len(ORDER) - 2)):
            poses.append((f"{ORDER[k]}->{ORDER[k+1]}", (q0 + (q1 - q0) * t).tolist(), [0.0]))
    dz_bot = (z_surf - P["plunge_mm"] / 1000.0) - z_lip0
    dz_top = ((z_surf + P["w25_proc_lift_to_surface_mm"] / 1000.0) - z_lip0
              if P.get("w25_proc_lift_to_surface_mm") is not None else dz_bot + P["lift_mm"] / 1000.0)
    for dz in np.linspace(0.0, dz_bot, 30).tolist() + np.linspace(dz_bot, dz_top, 30).tolist():
        zr = float(ad.to_robot([0.0, 0.0, z_lip0 + dz])[2])
        sol = FK.solve_fast(r0, zr, P["lip_l5_mm"])
        if sol is not None:
            poses.append(("scoop_column_ik", list(sol["q5"]), [0.0, q_open]))

    # 장애물: 트레이 벽 4장(상자 좌표 AABB) → 로봇 좌표 AABB (규약 A/B/rev32 모두 90° 배수 회전이라 정확)
    walls = []
    for c in BR.tray_cells(box, P["tray_wall_t_mm"]):
        cs = np.array([[c["min_m"][0] if i & 1 else c["max_m"][0], c["min_m"][1] if i & 2 else c["max_m"][1],
                        c["min_m"][2] if i & 4 else c["max_m"][2]] for i in range(8)])
        r = np.array([ad.to_robot(v) for v in cs])
        walls.append((c["name"], r.min(0), r.max(0)))

    lb = AB.load_link_local_bounds()
    S1 = W11SRC.S1
    Fv = np.asarray(trimesh.load(S1 / "fixed_ALL.stl").vertices, float)
    Dv = np.asarray(trimesh.load(S1 / "door_ALL.stl").vertices, float)
    H5 = np.array(P["hinge_l5_mm"], float)

    res = {"links": {}, "s1_full_stl": {}}
    for seg, q5, dqs in poses:
        _, per = AB.link_world_corners(FK.CHAIN, q5, lb, FK._T, FK._Tz, FK.SHOULDER_ABOVE_PLATE)
        for link, cr in per.items():
            smp = obb_samples(cr)
            for wn, lo, hi in walls:
                hit = obb_vs_aabb(cr, lo, hi)
                d = 0.0 if hit else float(aabb_dist(smp, lo, hi)[0].min())
                cur = res["links"].setdefault(link, {"min_gap_mm": None, "n_pose_overlap": 0, "overlap_at": []})
                if hit:
                    cur["n_pose_overlap"] += 1
                    if len(cur["overlap_at"]) < 8:
                        cur["overlap_at"].append({"segment": seg, "wall": wn, "q5": [round(v, 3) for v in q5]})
                if cur["min_gap_mm"] is None or d * 1000 < cur["min_gap_mm"]:
                    cur.update(min_gap_mm=round(d * 1000, 3), at={"segment": seg, "wall": wn,
                                                                  "q5": [round(v, 3) for v in q5]})
        T = FK.link5_T(q5)
        for dq in dqs:
            Dq = (W11SRC.roty(dq) @ (Dv - H5).T).T + H5
            for nm, V in (("fixed_ALL", Fv), (f"door_ALL@q{dq:.1f}", Dq)):
                Vr = (T[:3, :3] @ (V / 1000.0).T).T + T[:3, 3] - np.array([0, 0, FK.SHOULDER_ABOVE_PLATE])
                key = nm.split("@")[0]
                cur = res["s1_full_stl"].setdefault(key, {"min_gap_mm": None, "n_vertices_inside": 0})
                for wn, lo, hi in walls:
                    d, ins = aabb_dist(Vr, lo, hi)
                    cur["n_vertices_inside"] += int(ins.sum())
                    if cur["min_gap_mm"] is None or d.min() * 1000 < cur["min_gap_mm"]:
                        cur.update(min_gap_mm=round(float(d.min()) * 1000, 3), at={"segment": seg, "wall": wn,
                                                                                    "door_q": dq})
    out = {"artifact": "W25A_ARM_WALL_CHECK", "params": str(params_path), "pile": str(pile_path),
           "tray_box_bounds_m": box.tolist(), "tray_wall_t_mm": P["tray_wall_t_mm"], "tray": tray_info,
           "walls_robot_frame_m": [{"name": n, "lo": lo.round(6).tolist(), "hi": hi.round(6).tolist()}
                                   for n, lo, hi in walls],
           "wall_top_robot_z_m": float(max(h[2] for _, _, h in walls)),
           "wall_top_above_room_floor_cm": round((float(max(h[2] for _, _, h in walls)) - info["floor_robot_z_m"]) * 100, 4),
           "n_poses": len(poses), "result": res,
           "any_link_overlap": any(v["n_pose_overlap"] for v in res["links"].values()),
           "any_s1_vertex_inside": any(v["n_vertices_inside"] for v in res["s1_full_stl"].values()),
           "non_claims": ["링크는 local AABB 를 옮긴 OBB — 실제 메시보다 크다(겹침 = 가능성, 증명 아님).",
                          "취점 기둥 자세는 실물 IK 해(러너는 이 구간을 직교 명령으로 움직인다, 차 < 0.05 mm).",
                          "벽은 선언 두께·높이의 직육면체이며 종이 상자의 휨·접힌 윗단은 모델에 없다."]}
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in ("wall_top_above_room_floor_cm", "n_poses", "any_link_overlap",
                                          "any_s1_vertex_inside")}, ensure_ascii=False))
    for k, v in res["links"].items():
        print(" link", k, v["min_gap_mm"], v["n_pose_overlap"], v["at"]["segment"], v["at"]["wall"])
    for k, v in res["s1_full_stl"].items():
        print(" s1", k, v["min_gap_mm"], v["n_vertices_inside"], v["at"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
