#!/usr/bin/env python3
"""W2b (09-10) v0(가짜 hand_tcp 1 kg, w2_env_replay/) vs v1(질량 수정 usd_s1_v1, w2_env_replay/v1/) 비교표 → compare_v0_v1.json.
항목: 립 오차(sim−jsonl) 중앙값/최대·sim−FK(cmd)·상자 최소거리·어깨 정역학 모멘트(URDF 질량, 부록 M_model / M_model_tcp)·어깨 처짐·정착 규약. 입력 sha16 기록(D470).
사용: python3 compare_v0_v1.py
"""
import json, hashlib, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); V0 = os.path.dirname(HERE)
def sha16(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
def load(d):
    g = json.load(open(os.path.join(d, "gates_w2_env_replay.json"))); s = json.load(open(os.path.join(d, "settle_records.json"))); r = json.load(open(os.path.join(d, "replay_result.json"))); return g, s, r
out = {"inputs_sha256_16": {}}
rows = {}
for tag, d in (("v0", V0), ("v1", HERE)):
    g, S, R = load(d)
    for f in ("gates_w2_env_replay.json", "settle_records.json", "replay_result.json"): out["inputs_sha256_16"][os.path.join(os.path.relpath(d, os.getcwd()), f)] = sha16(os.path.join(d, f))
    G = [x for x in S if x["kind"] == "goto"]; dq = np.array([np.subtract(x["q_sim"], x["q_cmd"]) for x in G]); ap = g.get("appendix_loads_vs_torque", {})
    ext = [x for x in G if x["name"] in ("above_move", "place_target", "place_up", "scoop_lift8", "scoop_plunge", "scoop_surface", "scoop_travel")]   # 뻗은 자세군
    rows[tag] = {"usd": R["usd"], "hand_tcp_mass_kg": (R.get("body_mass_kg") or {}).get("hand_tcp"), "physx_mass_total_kg": round(sum((R.get("body_mass_kg") or {}).values()), 5),
        "gates": {k: g[k]["pass"] for k in ("G1_no_interference", "G2_lip_sim_vs_real", "G3_completion", "G4_media")}, "all_pass": g["all_pass"],
        "lip_sim_real_mm": g["G2_lip_sim_vs_real"]["d_sim_real_mm"], "lip_sim_fkcmd_mm": g["G2_lip_sim_vs_real"]["d_sim_fkcmd_mm"], "lip_real_fkcmd_mm": g["G2_lip_sim_vs_real"]["d_real_fkcmd_mm"],
        "lip_axis_bias_mm": g["G2_lip_sim_vs_real"]["axis_bias_mm_sim_minus_real_mean"], "lip_worst": {k: g["G2_lip_sim_vs_real"]["worst"].get(k) for k in ("cycle", "name", "d_sim_real_mm")},
        "box_dmin_mm": {b: g["G1_no_interference"]["dmin_mm"][b][0] for b in ("gripper_link", "grab_fixed", "link5")}, "box_worst": g["G1_no_interference"]["worst"], "contact_max_N": {b: (v[0] if v else None) for b, v in g["G1_no_interference"]["contact_max_N"].items()},
        "shoulder_sag_deg": {"median_abs": round(float(np.median(np.abs(dq[:, 1]))), 3), "max_abs": round(float(np.abs(dq[:, 1]).max()), 3), "extended_poses_median": round(float(np.median([abs(x["q_sim"][1] - x["q_cmd"][1]) for x in ext])), 3) if ext else None},
        "elbow_dev_deg": {"median_abs": round(float(np.median(np.abs(dq[:, 2]))), 3), "max_abs": round(float(np.abs(dq[:, 2]).max()), 3)},
        "joint_dev_deg_median_abs": g["G3_completion"]["joint_dev_deg_sim_minus_cmd"]["per_joint_median_abs"], "joint_dev_deg_max_abs": g["G3_completion"]["joint_dev_deg_sim_minus_cmd"]["per_joint_max_abs"],
        "shoulder_static_Nm": {"M_model_urdf_range": ap.get("M_model_range_Nm"), "M_model_tcp_range(sim 실제)": ap.get("M_model_tcp_range_Nm"), "hand_tcp_used_kg": ap.get("hand_tcp_mass_used_kg"), "K_eff": ap.get("K_eff_Nm_per_rad(M_model_tcp/sag)"),
                              "median_by_name": {k: v for k, v in (ap.get("median_by_name") or {}).items()}},
        "settle": g["G3_completion"]["settle"], "settle_rule": R.get("settle_rule"), "sim_seconds": R["sim_seconds"], "frames": R["frames"], "mesh_tag": R.get("mesh_tag", "s1(v0)"), "urdf": R.get("urdf")}
def delta(a, b):
    try: return round(b - a, 3)
    except Exception: return None
out["v0"] = rows["v0"]; out["v1"] = rows["v1"]
out["delta_v1_minus_v0"] = {"lip_median_mm": delta(rows["v0"]["lip_sim_real_mm"]["median"], rows["v1"]["lip_sim_real_mm"]["median"]), "lip_max_mm": delta(rows["v0"]["lip_sim_real_mm"]["max"], rows["v1"]["lip_sim_real_mm"]["max"]),
    "lip_sim_fkcmd_median_mm": delta(rows["v0"]["lip_sim_fkcmd_mm"]["median"], rows["v1"]["lip_sim_fkcmd_mm"]["median"]), "lip_sim_fkcmd_max_mm": delta(rows["v0"]["lip_sim_fkcmd_mm"]["max"], rows["v1"]["lip_sim_fkcmd_mm"]["max"]),
    "box_dmin_gripper_mm": delta(rows["v0"]["box_dmin_mm"]["gripper_link"], rows["v1"]["box_dmin_mm"]["gripper_link"]), "shoulder_sag_median_deg": delta(rows["v0"]["shoulder_sag_deg"]["median_abs"], rows["v1"]["shoulder_sag_deg"]["median_abs"]),
    "shoulder_sag_max_deg": delta(rows["v0"]["shoulder_sag_deg"]["max_abs"], rows["v1"]["shoulder_sag_deg"]["max_abs"]), "elbow_dev_max_deg": delta(rows["v0"]["elbow_dev_deg"]["max_abs"], rows["v1"]["elbow_dev_deg"]["max_abs"])}
json.dump(out, open(os.path.join(HERE, "compare_v0_v1.json"), "w"), indent=1, ensure_ascii=False)
for tag in ("v0", "v1"):
    r = rows[tag]; print(tag, "tcp", r["hand_tcp_mass_kg"], "gates", r["gates"], "| lip med/max", r["lip_sim_real_mm"]["median"], r["lip_sim_real_mm"]["max"], "| sim-fkcmd med/max", r["lip_sim_fkcmd_mm"]["median"], r["lip_sim_fkcmd_mm"]["max"],
          "| dmin", r["box_dmin_mm"], "| sag med/max", r["shoulder_sag_deg"], "| elbow", r["elbow_dev_deg"], "| M_tcp", r["shoulder_static_Nm"]["M_model_tcp_range(sim 실제)"], "| settle", r["settle"] if isinstance(r["settle"], str) else {k: r["settle"][k] for k in ("settle_s_median", "settle_s_max", "n_hit_max")}, "| sim s", r["sim_seconds"])
print("delta", out["delta_v1_minus_v0"])
