"""러너가 DEME 에 넘길 도메인(x/y/z, 상자 좌표 m)을 **러너와 같은 식·같은 순서**로 계산한다 (CPU, DEME 0).

`sim_w13_full_cycle.py`(rev34) :116-118(자리 rng) · :135-171(더미·트레이·자리) · :192-211(펠릿면·어댑터·웨이포인트)
· :227-236(스윕 도메인) 을 옮겨 적었다. 입자 등록·솔버는 없다. 양성 대조 = CPU 스텁 결과 JSON 의
`engine.domain_*` 과 binary64 비교(`--expect <stub result json>`).
usage: python domain_w25.py <params> <pile npz> <out json> [--expect stub.json] [--no-fail-closed]
출력 JSON 은 `engine.domain_{x,y,z}_m` 형식이라 make_numeric_evidence_w25.py 입력으로 바로 쓸 수 있다.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, "/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(SRC))
import sim_deme_scoop_s1 as W11SRC                                       # noqa: E402
import w13_fk as FK                                                      # noqa: E402
import w13_kinematics as K                                               # noqa: E402


def domain(params_path, pile_path, fail_closed=True, seed=460):
    P = dict(W11SRC.DEFAULT)
    P.update(K.W13_DEFAULT)
    P.update(json.load(open(params_path)))
    rng = np.random.default_rng(seed)
    y_s = float(rng.uniform(*P["scoop_y_range_mm"])) / 1000.0
    x_s = P["scoop_x_mm"] / 1000.0
    z = np.load(pile_path, allow_pickle=True)
    box = np.asarray(z["box_bounds_m"], float)
    box, tray_info = FK.w25_tray_bounds(box, P, np.asarray(z["positions_m"], float), np.asarray(z["radii_m"], float),
                                        fail_closed=fail_closed)
    q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
    fixed_m, door_m, lip_f, lip_d, hinge_off, L5mm, _ = W11SRC.load_tool(P, q_open)
    P["lip_l5_mm"] = [float(v) for v in L5mm]
    site_xy, site = FK.w25_scoop_site(P, P["lip_l5_mm"])
    if not site.get("rev32_path"):
        x_s, y_s = site_xy
    fixed_v, door_v = np.asarray(fixed_m.vertices, float), np.asarray(door_m.vertices, float)
    tpl = json.loads(str(z["clump_template_json"]))
    sp, sr = W11SRC.expand_spheres(np.asarray(z["clump_positions_m"], float),
                                   np.asarray(z["clump_quaternions_xyzw"], float), tpl)
    z_surf_pre = W11SRC.surface_z(sp, sr, x_s, y_s, P)
    gap_mm = P["w25_door_open_gap_mm"] if P.get("w25_proc_open_at_surface") else P["approach_gap_mm"]
    z_lip0 = z_surf_pre + gap_mm / 1000.0
    ad, ad_info, fr = FK.build_adapter_w25(box, z_surf_pre, P, P["lip_l5_mm"])
    r0 = float(np.hypot(*FK.lip_pose(ad_info["reference_pose_q5"], P["lip_l5_mm"])[0][:2]))
    z_travel = float(ad_info["floor_robot_z_m"] + P["travel_cm"] / 100.0 - ad_info["t_robot_m"][2])
    z_surf5 = float(ad_info["pellet_robot_z_m"] + 0.05 - ad_info["t_robot_m"][2])
    tray = K.tray_mesh(box, P["tray_wall_t_mm"] / 1000.0)
    wps_probe, _ = FK.build_waypoints(ad, r0, z_travel, z_lip0, z_surf5, 0.0, P["place_base_deg"])
    bin_center = np.asarray([w for w in wps_probe if w["name"] == "place_target"][0]["lip_world_m"])[:2]
    binm, bin_info = K.bin_mesh(P, bin_center)
    z_release = bin_info["rim_z_m"] + P["release_clearance_mm"] / 1000.0
    wps, _ = FK.build_waypoints(ad, r0, z_travel, z_lip0, z_surf5, z_release, P["place_base_deg"])
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
    z_src = "swept_max+pad" if swept[:, 2].max() + pad > P["domain_top_m"] else "domain_top_m"
    return {"artifact": "W25A_DOMAIN_CPU", "params": str(params_path), "pile": str(pile_path),
            "engine": {"domain_x_m": list(dom_x), "domain_y_m": list(dom_y), "domain_z_m": list(dom_z)},
            "domain_z_top_source": z_src, "box_bounds_m": box.tolist(), "tray": tray_info,
            "surface_z_pre_m": z_surf_pre, "t_robot_m": ad_info["t_robot_m"], "bin_center_m": bin_center.tolist(),
            "extent_m": [dom_x[1] - dom_x[0], dom_y[1] - dom_y[0], dom_z[1] - dom_z[0]],
            "rule": "sim_w13_full_cycle.py:227-236 과 같은 식(웨이포인트 툴 셸 + 트레이 + 용기 정점, pad)"}


if __name__ == "__main__":
    a = sys.argv[1:]
    fc = "--no-fail-closed" not in a
    out = domain(a[0], a[1], fail_closed=fc)
    if "--expect" in a:
        exp = json.load(open(a[a.index("--expect") + 1]))["engine"]
        out["positive_control"] = {"expect_file": a[a.index("--expect") + 1],
                                   "binary64_equal": all(exp[f"domain_{k}_m"] == out["engine"][f"domain_{k}_m"]
                                                         for k in "xyz")}
    json.dump(out, open(a[2], "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in ("engine", "domain_z_top_source", "extent_m")}, indent=1),
          out.get("positive_control"))
