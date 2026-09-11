#!/usr/bin/env python3
"""W1 게이트 (09-09): s1_v1 실물 형상으로 만든 roarm_m3_s1_v1.urdf / usd_s1_v1 검증. 결과 = gates_w1_usd_v1.json.
G1 meta source=s1_v1 + sha16 일치 / G2 URDF 링크·조인트 이름 집합 = v0 (+USD 관절·바디 이름) / G3 문+고정 질량 vs design.json derived
G4 Isaac 문 닫힘 정지각(door_close_selfcol_on.json) + 그 각에서 기하 간격·관통 깊이(trimesh) / G5 손목 피치 ±1.92 유지 + meta firmware_clamp_deg 90 / USD 실물 존재·collider.
D470: 읽은 입력 전부 path+sha16. D476: 의도가 아니라 산출물을 읽어 판정.
사용: python gates_w1_usd_v1.py [결과 폴더]   (isaaclab env python; numpy 1.26 / trimesh 4.5.1). 결과 폴더를 주면 그 폴더의 door_close_*.json 을 읽고 gates JSON 도 거기에 쓴다(W1b fix_mass 재실행용).
"""
import json, hashlib, math, re
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ET

import sys
REPO = Path(__file__).resolve().parents[6]; HERE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent   # 인자 = 결과 폴더(문 프로브 JSON·gates 출력). 기본 = 이 폴더. W1b 는 fix_mass/
SRC = REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1"; RDIR = REPO / "local_assets/roarm_m3/urdf"; UDIR = REPO / "local_assets/roarm_m3/usd_s1_v1"
EXPECT = {"door_ALL_sha16": "32521db98db7e421", "fixed_ALL_sha16": "2fea1c37b63ff36e"}
inputs = {}
def sha16(p):
    p = Path(p); h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]; inputs[str(p.relative_to(REPO))] = h; return h
def names(urdf):
    r = ET.parse(urdf).getroot(); return sorted(l.get("name") for l in r.findall("link")), sorted((j.get("name"), j.get("type")) for j in r.findall("joint")), r
G = {}
# ---- G1
meta = json.loads((RDIR / "s1_v1_meta.json").read_text()); sha16(RDIR / "s1_v1_meta.json")
G["G1_meta_source_and_sha"] = {"source_dir_tail": Path(meta["source_dir"]).name, "meta_door_sha16": meta["door_ALL_sha16"], "meta_fixed_sha16": meta["fixed_ALL_sha16"],
    "file_door_sha16": sha16(SRC / "door_ALL.stl"), "file_fixed_sha16": sha16(SRC / "fixed_ALL.stl"), "expected": EXPECT}
g = G["G1_meta_source_and_sha"]; g["pass"] = bool(g["source_dir_tail"] == "s1_v1" and g["meta_door_sha16"] == g["file_door_sha16"] == EXPECT["door_ALL_sha16"] and g["meta_fixed_sha16"] == g["file_fixed_sha16"] == EXPECT["fixed_ALL_sha16"])
# ---- G2
L0, J0, _ = names(RDIR / "roarm_m3_s1.urdf"); L1, J1, r1 = names(RDIR / "roarm_m3_s1_v1.urdf"); sha16(RDIR / "roarm_m3_s1.urdf"); sha16(RDIR / "roarm_m3_s1_v1.urdf")
door_j = next(j for j in r1.findall("joint") if j.get("name") == "link5_to_gripper_link")
usd_names = None
pr = HERE / "door_close_selfcol_on.json"
if pr.exists():
    d = json.loads(pr.read_text()); sha16(pr); usd_names = {"joint_names": d.get("joint_names"), "body_names": d.get("body_names")}
G["G2_names_equal_v0"] = {"links_v1": L1, "joints_v1": J1, "links_equal": L0 == L1, "joints_equal": J0 == J1, "door_joint": {"name": door_j.get("name"), "type": door_j.get("type"), "child": door_j.find("child").get("link"), "axis": door_j.find("axis").get("xyz"), "limit": door_j.find("limit").attrib},
    "usd": usd_names, "usd_has_door_joint": (usd_names is not None and "link5_to_gripper_link" in usd_names["joint_names"]), "usd_has_grab_fixed_body": (usd_names is not None and "grab_fixed" in usd_names["body_names"])}
g = G["G2_names_equal_v0"]; g["pass"] = bool(g["links_equal"] and g["joints_equal"] and door_j.get("type") == "revolute" and g["usd_has_door_joint"] and g["usd_has_grab_fixed_body"])
# ---- G3
design = json.loads((SRC / "design.json").read_text()); sha16(SRC / "design.json"); dv = design["derived"]
m_urdf = 1000 * (meta["door"]["mass_kg"] + meta["fixed"]["mass_kg"]); ref_parts = dv["door_g"] + dv["fixed_g"]
G["G3_mass"] = {"urdf_door_g": round(1000 * meta["door"]["mass_kg"], 2), "urdf_fixed_g": round(1000 * meta["fixed"]["mass_kg"], 2), "urdf_sum_g": round(m_urdf, 2),
    "design_door_g": dv["door_g"], "design_fixed_g": dv["fixed_g"], "design_parts_sum_g(나사 제외)": round(ref_parts, 2), "design_hardware_g(나사)": dv["hardware_g"], "design_tool_mass_g(나사 포함)": dv["tool_mass_g"],
    "dev_vs_parts_pct": round(100 * (m_urdf / ref_parts - 1), 2), "dev_vs_tool_mass_pct": round(100 * (m_urdf / dv["tool_mass_g"] - 1), 2),
    "note": "URDF 는 인쇄 부품(문+고정부)만 — 나사 6.2 g 는 모델에 없다. 지시문의 '42.93(나사 제외)' 은 design.json 에서 나사 포함값이며, 나사 제외값은 door_g+fixed_g=36.73."}
G["G3_mass"]["pass"] = bool(abs(G["G3_mass"]["dev_vs_parts_pct"]) <= 10.0); G["G3_mass"]["pass_vs_tool_mass_42p93"] = bool(abs(G["G3_mass"]["dev_vs_tool_mass_pct"]) <= 10.0)
# ---- G5
wp = next(j for j in r1.findall("joint") if j.get("name") == "link3_to_link4"); lim = wp.find("limit").attrib
G["G5_wrist_pitch"] = {"urdf_limit": lim, "meta_wrist_pitch": meta.get("wrist_pitch"), "pass": bool(float(lim["lower"]) == -1.92 and float(lim["upper"]) == 1.92 and meta.get("wrist_pitch", {}).get("firmware_clamp_deg") == 90)}
# ---- G4 기하: 문을 힌지(link5 X=0, Z=52.035, 축 Y, +각 = +X 로 열림) 둘레로 돌려 고정부와의 최소 간격·관통 깊이
HZ = 52.035
door = trimesh.load(SRC / "door_ALL.stl", force="mesh"); fixed_all = trimesh.load(SRC / "fixed_ALL.stl", force="mesh")
fixed_pieces = [trimesh.load(p, force="mesh") for p in sorted(SRC.glob("fixed_*.stl")) if not p.name.startswith("fixed_ALL")]
for p in sorted(SRC.glob("fixed_*.stl")): sha16(p)
rng = np.random.default_rng(0)
dpts = np.vstack([door.vertices, trimesh.sample.sample_surface(door, 300000, seed=0)[0]]); fpts = np.vstack([fixed_all.vertices, trimesh.sample.sample_surface(fixed_all, 300000, seed=0)[0]]); ftree = cKDTree(fpts)
planes = []
for m in fixed_pieces:
    n = m.face_normals; d = np.einsum("ij,ij->i", n, m.triangles[:, 0]); planes.append((n, d))
def rot(pts, deg):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th); x = pts[:, 0]; z = pts[:, 2] - HZ
    return np.stack([x * c + z * s, pts[:, 1], -x * s + z * c + HZ], 1)
def analyze(deg):
    p = rot(dpts, deg); dist, idx = ftree.query(p); i = int(np.argmin(dist)); depth = 0.0; n_in = 0
    for n, d in planes:
        sd = p @ n.T - d; inside = np.all(sd <= 0.0, axis=1)
        if inside.any(): n_in += int(inside.sum()); depth = max(depth, float((-sd[inside].max(axis=1)).max()))
    return {"deg": round(deg, 4), "min_gap_mm": round(float(dist[i]), 3), "closest_door_pt_link5_mm": np.round(p[i], 2).tolist(), "closest_fixed_pt_link5_mm": np.round(fpts[idx[i]], 2).tolist(), "door_pts_inside_fixed": n_in, "max_penetration_mm": round(depth, 3)}
sweep = [analyze(a) for a in (0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 5.0)]
sim = None
if pr.exists():
    d = json.loads(pr.read_text()); sim = {k: d.get(k) for k in ("stop_deg_mean_last0p5s", "stop_deg_min", "stop_deg_max", "settled", "F_door_fixed_N_last0p5s_mean", "F_door_fixed_N_max", "first_contact", "tau_Nm_last0p5s_mean", "collision_offsets_m", "door_joint_limits_rad", "ok", "error")}
    sim["geometry_at_stop"] = analyze(d["stop_deg_mean_last0p5s"]) if d.get("stop_deg_mean_last0p5s") is not None else None
ctrl = HERE / "door_close_selfcol_off.json"; ctrl_d = None
if ctrl.exists():
    c = json.loads(ctrl.read_text()); sha16(ctrl); ctrl_d = {k: c.get(k) for k in ("stop_deg_mean_last0p5s", "F_door_fixed_N_max", "ok", "error")}
G["G4_door_close"] = {"real_stop_deg": [2.5, 3.5], "real_source": "D481 §2/§3 (servo_deg 0 닫힘·+ 열림; 부팅 π 에서 2.55° 읽힘, 사이클 닫힘 2.8~3.5°)", "sim_selfcol_on": sim, "sim_selfcol_off_control": ctrl_d,
    "geometry_sweep": sweep, "sampling_resolution_mm": "표면 표본 30만+정점, KD-트리 최근접 (±0.5 mm 급)", "hinge": "link5 X=0, Z=52.035, 축 Y, +deg = +X 열림"}
g = G["G4_door_close"]
if sim and sim.get("stop_deg_mean_last0p5s") is not None:
    sd = sim["stop_deg_mean_last0p5s"]; g["sim_stop_deg"] = sd; g["sim_minus_real_deg"] = [round(sd - 2.5, 3), round(sd - 3.5, 3)]
    g["contact_evidence"] = bool((sim.get("F_door_fixed_N_last0p5s_mean") or 0) > 1e-3); g["within_real_band"] = bool(2.5 <= sd <= 3.5)
    g["pass"] = bool(sim.get("ok") and sim["geometry_at_stop"]["max_penetration_mm"] <= 0.5)   # 측정 성공 + 정지각에서 관통 ≤ 0.5 mm 면 PASS (실물 대역 일치 여부는 별도 보고)
else: g["pass"] = False; g["reason"] = "door_close_selfcol_on.json 없음/실패"
# ---- USD 실물
usd = UDIR / "roarm_m3_s1_v1.usd"; cfg = UDIR / "config.yaml"; U = {"usd_exists": usd.exists(), "usd_bytes": (usd.stat().st_size if usd.exists() else 0)}
if usd.exists(): U["usd_sha16"] = sha16(usd)
if cfg.exists():
    t = cfg.read_text(); sha16(cfg); U["collider_type"] = re.search(r"collider_type:\s*(\S+)", t).group(1); U["mimic_flag"] = re.search(r"convert_mimic_joints_to_normal_joints:\s*(\S+)", t).group(1); U["asset_path_tail"] = Path(re.search(r"asset_path:\s*(\S+)", t).group(1)).name
    base = UDIR / "configuration" / "roarm_m3_s1_v1_base.usd"; U["base_usd_bytes"] = base.stat().st_size if base.exists() else 0
U["pass"] = bool(U["usd_exists"] and U["usd_bytes"] > 0 and U.get("asset_path_tail") == "roarm_m3_s1_v1.urdf" and U.get("mimic_flag") == "false" and U.get("base_usd_bytes", 0) > 100000)
G["USD_artifact"] = U
sha16(RDIR / "roarm_m3.urdf")
out = {"worker": "W1 w1_usd_v1", "date": "2026-09-09", "all_pass": all(v.get("pass") for v in G.values()), "gates": G, "inputs_sha16": inputs}
json.dump(out, open(HERE / "gates_w1_usd_v1.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v.get("pass") for k, v in G.items()}, ensure_ascii=False), "all_pass =", out["all_pass"])
for k, v in G.items():
    if k == "G4_door_close": print("G4 sim:", v.get("sim_selfcol_on") and {kk: v["sim_selfcol_on"].get(kk) for kk in ("stop_deg_mean_last0p5s", "F_door_fixed_N_last0p5s_mean", "first_contact")}, "| geom@stop:", v.get("sim_selfcol_on", {}) and v["sim_selfcol_on"].get("geometry_at_stop")); print("G4 sweep:", [(s["deg"], s["min_gap_mm"], s["max_penetration_mm"]) for s in v["geometry_sweep"]])
    elif k == "G3_mass": print("G3:", {kk: v[kk] for kk in ("urdf_sum_g", "design_parts_sum_g(나사 제외)", "dev_vs_parts_pct", "dev_vs_tool_mass_pct")})
