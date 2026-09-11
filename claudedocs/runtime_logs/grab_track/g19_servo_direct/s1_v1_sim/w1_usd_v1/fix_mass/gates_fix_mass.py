#!/usr/bin/env python3
"""W1b 게이트 (09-09): 가짜 링크 질량 제거 후 v1 URDF/USD 재검증. 결과 = gates_fix_mass.json.
G1~G6 = 상위 gates_w1_usd_v1.py 를 그대로 재실행한 결과(gates_w1_usd_v1.json, 재생성본 기준) 를 읽어 옮김.
G7 (신규) = ① PhysX hand_tcp 질량 ≤ 1e-3 kg ② HOME·스쿱 자세 어깨(link1_to_link2) 중력 모멘트: URDF 질량·COM 합(순수 FK, 이 파일) vs Isaac PhysX 질량·COM 정역학 재계산(mass_static_post.json) ±5 %.
   비교는 같은 관절각(Isaac 측정각) 에서 한다. applied_torque(PD 계산값) 는 참고로만 적는다(W2 ③: 물리 토크 아님).
D470: 읽은 입력 path+sha16.  사용: python gates_fix_mass.py (isaaclab env)
"""
import json, hashlib, math, re
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[6]; RDIR = REPO / "local_assets/roarm_m3/urdf"
inputs = {}
def sha16(p):
    p = Path(p); h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]; inputs[str(p.relative_to(REPO))] = h; return h
def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]); Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); return Rz @ Ry @ Rx
def axis_rot(a, q):
    a = np.asarray(a, float); a /= np.linalg.norm(a); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]); return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K
def T(R, t): M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M

def urdf_static(urdf, qmap, g=9.81):
    """링크 월드 프레임(FK) → 각 revolute 관절에 대해 원위 링크의 중력 모멘트 Σ[(r_com−r_j)×m g]·axis_w. 반환 {joint: (moment, per_body)}"""
    r = ET.parse(urdf).getroot(); links = {l.get("name"): l for l in r.findall("link")}; joints = r.findall("joint")
    children = {}; parent_of = {}
    for j in joints:
        parent_of[j.find("child").get("link")] = j; children.setdefault(j.find("parent").get("link"), []).append(j)
    root = next(n for n in links if n not in parent_of); W = {root: np.eye(4)}; Rj = {}
    def rec(name):
        for j in children.get(name, []):
            o = j.find("origin"); xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]; rp = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            M = W[name] @ T(rpy(*rp), xyz)
            if j.get("type") in ("revolute", "continuous"):
                ax = [float(v) for v in j.find("axis").get("xyz").split()]; q = qmap.get(j.get("name"), 0.0); M = M @ T(axis_rot(ax, q), [0, 0, 0]); Rj[j.get("name")] = (M[:3, 3].copy(), M[:3, :3] @ (np.asarray(ax) / np.linalg.norm(ax)))
            W[j.find("child").get("link")] = M; rec(j.find("child").get("link"))
    rec(root)
    com_w, mass = {}, {}
    for n, l in links.items():
        i = l.find("inertial")
        if i is None: continue
        o = i.find("origin"); xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        com_w[n] = (W[n] @ np.array(xyz + [1.0]))[:3]; mass[n] = float(i.find("mass").get("value"))
    def distal(jname):
        j = next(x for x in joints if x.get("name") == jname); out = []; stack = [j.find("child").get("link")]
        while stack:
            n = stack.pop(); out.append(n); stack += [c.find("child").get("link") for c in children.get(n, [])]
        return out
    res = {}
    for jname, (rj, ax) in Rj.items():
        per = {}; M = 0.0
        for n in distal(jname):
            if n in mass: mi = float(np.dot(np.cross(com_w[n] - rj, mass[n] * np.array([0, 0, -g])), ax)); per[n] = round(mi, 5); M += mi
        res[jname] = {"gravity_moment_Nm": round(M, 5), "per_body_Nm": per}
    return res, mass

G = {}
# ---- G1~G6 (상위 게이트 재실행 결과를 읽음)
up = json.loads((HERE / "gates_w1_usd_v1.json").read_text()); sha16(HERE / "gates_w1_usd_v1.json")   # 재생성본 기준 재실행 결과 (python ../gates_w1_usd_v1.py fix_mass)
for k, v in up["gates"].items(): G[k] = {"pass": v.get("pass"), "from": "gates_w1_usd_v1.json (재생성본 기준 재실행)", "key_values": {kk: v.get(kk) for kk in ("sim_stop_deg", "sim_minus_real_deg", "urdf_sum_g", "dev_vs_parts_pct", "usd_sha16", "links_equal", "joints_equal") if kk in v}}
# ---- G7
post = json.loads((HERE / "mass_static_post.json").read_text()); sha16(HERE / "mass_static_post.json")
pre = json.loads((HERE / "mass_static_pre.json").read_text()); sha16(HERE / "mass_static_pre.json")
urdf = RDIR / "roarm_m3_s1_v1.urdf"; sha16(urdf); sha16(RDIR / "s1_v1_meta.json")
for p in ("usd_s1_v1/configuration/roarm_m3_s1_v1_base.usd", "usd_s1_v1/configuration/roarm_m3_s1_v1_physics.usd", "usd_s1_v1.bak_pre_massfix/configuration/roarm_m3_s1_v1_physics.usd", "usd_s1_v1/roarm_m3_s1_v1.usd", "usd_s1_v1.bak_pre_massfix/configuration/roarm_m3_s1_v1_base.usd", "urdf/roarm_m3_s1_v1.urdf.bak_pre_massfix", "urdf/roarm_m3.urdf"): sha16(REPO / "local_assets/roarm_m3" / p)
sha16(REPO / "compose_roarm_s1_urdf.py")
g7 = {"hand_tcp_mass_kg": {"pre": pre["physx_bodies"]["hand_tcp"]["mass_kg"], "post": post["physx_bodies"]["hand_tcp"]["mass_kg"], "limit": 1e-3},
      "world_mass_kg": {"pre": pre["physx_bodies"]["world"]["mass_kg"], "post": post["physx_bodies"]["world"]["mass_kg"], "note": "루트 world 는 fix_root_link=True 로 고정 → 관절 하중 무관, 미수정"},
      "physx_mass_total_kg": {"pre": pre["physx_mass_total_kg"], "post": post["physx_mass_total_kg"]}, "poses": {}}
urdf_mass_sum = None
for pname in ("home", "scoop"):
    rec = post["poses"][pname]; q_meas = rec["measured"]; q_tgt = rec["target"]
    u_meas, mass = urdf_static(urdf, q_meas); u_tgt, _ = urdf_static(urdf, q_tgt); urdf_mass_sum = sum(mass.values())
    row = {}
    for jn in ("link1_to_link2", "link2_to_link3"):
        isaac = rec["static"][jn]["gravity_moment_Nm"]; um = u_meas[jn]["gravity_moment_Nm"]; ut = u_tgt[jn]["gravity_moment_Nm"]
        dev = (isaac / um - 1) * 100 if abs(um) > 1e-9 else None
        row[jn] = {"urdf_static_at_measured_q_Nm": um, "urdf_static_at_target_q_Nm": ut, "isaac_physx_static_at_measured_q_Nm": isaac, "dev_pct": (round(dev, 3) if dev is not None else None),
                   "isaac_applied_torque_Nm(참고,PD계산값)": rec["applied_torque_Nm"][jn], "pre_fix_isaac_static_Nm": pre["poses"][pname]["static"][jn]["gravity_moment_Nm"],
                   "deflection_deg_post": rec["deflection_deg"][jn], "deflection_deg_pre": pre["poses"][pname]["deflection_deg"][jn], "urdf_per_body_Nm": u_meas[jn]["per_body_Nm"], "isaac_per_body_Nm": rec["static"][jn]["per_body_Nm"]}
    g7["poses"][pname] = row
g7["urdf_mass_sum_kg(world 제외)"] = round(urdf_mass_sum, 6)
g7["physx_minus_urdf_mass_kg(post, world 1.0 제외)"] = round(post["physx_mass_total_kg"] - post["physx_bodies"]["world"]["mass_kg"] - urdf_mass_sum, 6)
sh_ok = all(abs(g7["poses"][p]["link1_to_link2"]["dev_pct"]) <= 5.0 for p in ("home", "scoop"))
g7["pass"] = bool(g7["hand_tcp_mass_kg"]["post"] <= 1e-3 and sh_ok); g7["criteria"] = "hand_tcp ≤ 1e-3 kg AND 어깨 정역학(URDF vs PhysX 재계산, 측정각 기준) |dev| ≤ 5 % (HOME·스쿱)"
G["G7_fake_mass_and_static"] = g7
out = {"worker": "W1b fix_mass", "date": "2026-09-09", "all_pass": all(v.get("pass") for v in G.values()), "gates": G, "inputs_sha16": inputs}
json.dump(out, open(HERE / "gates_fix_mass.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v.get("pass") for k, v in G.items()}, ensure_ascii=False), "all_pass =", out["all_pass"])
for p in ("home", "scoop"):
    for jn in ("link1_to_link2", "link2_to_link3"):
        r = g7["poses"][p][jn]; print(p, jn, "urdf", r["urdf_static_at_measured_q_Nm"], "isaac", r["isaac_physx_static_at_measured_q_Nm"], "dev%", r["dev_pct"], "| pre", r["pre_fix_isaac_static_Nm"], "| tau", r["isaac_applied_torque_Nm(참고,PD계산값)"], "| defl post/pre", r["deflection_deg_post"], r["deflection_deg_pre"])
print("hand_tcp", g7["hand_tcp_mass_kg"], "total", g7["physx_mass_total_kg"])
