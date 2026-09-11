#!/usr/bin/env python3
"""[W2b 09-10 사본: 정착 규약 문자열을 replay_result.json settle_rule 에서 읽고 settle_dq_last 최대를 기록 — 그 외 w2_gates.py 와 동일] W2 게이트 후처리 — `sim_isaaclab_s1_env_replay.py` 산출(replay_result.json · settle_records.json · replay_log.json · frames/)을 **읽어서** 판정한다(D476).
G1 간섭 / G2 립 sim-vs-real / G3 완주·NaN·관절 편차 / G4 mp4·strip·놓기 프레임·빈 프레임 검사(D474)
+ 부록: 실물 tS(어깨 부하) vs URDF 질량 정역학 어깨 중력 토크 모델 · sim 처짐으로 본 유효 강성. Isaac 불필요.
사용: python3 w2_gates.py --out <replay 산출 폴더> [--fps 10] [--urdf local_assets/roarm_m3/urdf/roarm_m3_s1.urdf]
"""
import argparse, hashlib, json, math, os, subprocess
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GATE_BODIES = ("gripper_link", "grab_fixed", "link5")
STRIP_SEQ = ["p1", "above_move", "scoop_5cm_above", "scoop_surface", "door30", "scoop_plunge", "door0", "scoop_lift8", "scoop_travel", "place_rot90", "place_extend", "place_target", "door30", "place_retract2"]
PI = math.pi
CH = [("base_link", (0, 0, 0.0701), (0, 0, 0), None), ("link1", (0, 0, 0), (0, 0, 0), 0), ("link2", (0, 0, 0.051959), (-PI / 2, -PI / 2, 0), 1),
      ("link3", (0.236815, 0.030002, 0), (0, 0, PI / 2), 2), ("link4", (0, -0.144586, 0), (0, 0, 0), 3), ("link5", (0.015147, -0.053653, 0), (PI / 2, PI / 2, 0), 4)]   # = roarm_kinematics._CHAIN


def sha16(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
def stat(x): x = np.asarray(x, float); return {"median": round(float(np.median(x)), 2), "mean": round(float(np.mean(x)), 2), "p90": round(float(np.percentile(x, 90)), 2), "max": round(float(np.max(x)), 2)}
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
def _T(xyz, rpy): T = np.eye(4); T[:3, :3] = _rpy(*rpy); T[:3, 3] = xyz; return T
def _Rz(q): T = np.eye(4); c, s = math.cos(q), math.sin(q); T[:2, :2] = [[c, -s], [s, c]]; return T


def urdf_inertials(path):
    out = {}
    for l in ET.parse(path).getroot().findall("link"):
        i = l.find("inertial")
        if i is None: continue
        o = i.find("origin"); out[l.get("name")] = (float(i.find("mass").get("value")), np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]))
    return out


def shoulder_gravity_moment(q5_deg, door_deg, iner):
    """어깨축(link2 z) 둘레 중력 모멘트 [N·m] (link2~문·고정보울, URDF 질량·COM). 유지 토크 = −M."""
    T = np.eye(4); Ts = {}
    for name, xyz, rpy, qi in CH:
        T = T @ _T(xyz, rpy)
        if qi is not None: T = T @ _Rz(math.radians(q5_deg[qi]))
        Ts[name] = T.copy()
    Ts["grab_fixed"] = Ts["link5"]; Ts["gripper_link"] = Ts["link5"] @ _T((0, 0.018821, 0.052035), (-PI / 2, -PI / 2, 0)) @ _Rz(math.radians(door_deg))
    p_ax, ax = Ts["link2"][:3, 3], Ts["link2"][:3, 2]; M = 0.0
    for n in ("link2", "link3", "link4", "link5", "grab_fixed", "gripper_link"):
        m, c = iner[n]; pw = (Ts[n] @ np.r_[c, 1.0])[:3]; M += float(np.dot(np.cross(pw - p_ax, [0, 0, -m * 9.81]), ax))
    return M


ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--fps", type=int, default=10); ap.add_argument("--urdf", default="local_assets/roarm_m3/urdf/roarm_m3_s1.urdf"); a = ap.parse_args()
O = os.path.abspath(a.out); FR = os.path.join(O, "frames")
R = json.load(open(os.path.join(O, "replay_result.json"))); S = json.load(open(os.path.join(O, "settle_records.json"))); L = json.load(open(os.path.join(O, "replay_log.json")))
frames = sorted(f for f in os.listdir(FR) if f.endswith(".jpg")); nF = len(frames)
fidx = lambda t: min(nF - 1, int(t * a.fps))
G = [r for r in S if r["kind"] == "goto"]
gates = {"inputs_sha256_16": {**R["inputs_sha256_16"], **{f: sha16(os.path.join(O, f)) for f in ("replay_result.json", "settle_records.json", "replay_log.json")}, a.urdf: sha16(a.urdf)},
         "usd": R["usd"], "log": R["log"], "env": R["env"], "actuators": R["actuators"], "blend_speed_dps": R["blend_speed_dps"], "fk_selfcheck_max_mm": R["fk_selfcheck_max_mm"], "dist_points": R.get("dist_points"),
         "sim_mass_kg": R.get("body_mass_kg"), "sim_dof_gains": R.get("physx_dof_gains")}

# ── G1 간섭: 표면 표본점 부호거리 최소(매 물리 스텝, 음수 = 관통 깊이) > 0 그리고 PhysX 접촉력 0 ──
dmin = {b: R["dmin_overall_mm"][b] for b in GATE_BODIES}; cmax = {b: R["contact_max_N"].get(b) for b in GATE_BODIES}; wb = min(dmin, key=lambda b: dmin[b][0])
seg_min = {}
for r in L:
    if "dmin_mm" not in r: continue
    k = (r["cycle"], r["name"]); v = min(r["dmin_mm"][b] for b in GATE_BODIES)
    if k not in seg_min or v < seg_min[k][0]: seg_min[k] = (v, r["t"], fidx(r["t"]))
closest = sorted(seg_min.items(), key=lambda kv: kv[1][0])[:8]
gates["G1_no_interference"] = {"pass": bool(all(dmin[b][0] > 0 for b in GATE_BODIES) and all(cmax[b] is None or cmax[b][0] < 1e-6 for b in GATE_BODIES)),
    "criterion": "door(gripper_link)·grab_fixed·link5 표면 표본점 vs 상자(벽4+바닥+받침 AABB) 부호거리 최소 > 0 [mm] 그리고 PhysX net contact force = 0 [N], 전 구간 매 물리 스텝. 'bound' = 몸체 원점-상자 AABB 거리 − 몸체 반경(하한, 50 mm 초과 시 정밀 계산 생략)",
    "dmin_mm": dmin, "contact_max_N": cmax, "worst": {"body": wb, "dmin_mm": dmin[wb][0], **(dmin[wb][1] or {})},
    "closest_segments": [{"cycle": k[0], "name": k[1], "dmin_mm": v[0], "t": v[1], "frame": v[2]} for k, v in closest],
    "other_bodies_dmin_mm": {b: v for b, v in R["dmin_overall_mm"].items() if b not in GATE_BODIES}}

# ── G2 립: sim(link5 자세 + LIP_L5) vs jsonl lip(FK(read)) ──
dsr = np.array([r["d_sim_real_mm"] for r in G]); dsc = np.array([r["d_sim_fkcmd_mm"] for r in G]); drc = np.array([r["d_real_fkcmd_mm"] for r in G])
err = np.array([np.subtract(r["lip_sim"], r["lip_real"]) for r in G]) * 1000.0; iw = int(np.argmax(dsr))
by_name = {}
for r in G: by_name.setdefault(r["name"], []).append(r["d_sim_real_mm"])
gates["G2_lip_sim_vs_real"] = {"pass": bool(np.median(dsr) <= 10.0), "criterion": "goto 정착 시 |sim 립 − jsonl lip| 중앙값 ≤ 10 mm (최대값 보고)", "n_goto": len(G),
    "d_sim_real_mm": stat(dsr), "d_sim_fkcmd_mm": stat(dsc), "d_real_fkcmd_mm": stat(drc),
    "note": "d_sim_fkcmd = sim 립 vs FK(cmd): sim 추종(PD 처짐)+기하 오차. d_real_fkcmd = FK(read) vs FK(cmd): 실물 서보 편차(처짐)의 립 환산, 시뮬과 무관. sim 은 cmd 를 추종하므로 d_sim_real 은 두 항의 합성.",
    "axis_bias_mm_sim_minus_real_mean": [round(float(v), 2) for v in err.mean(0)], "axis_abs_max_mm": [round(float(v), 2) for v in np.abs(err).max(0)],
    "worst": {k: G[iw].get(k) for k in ("cycle", "name", "d_sim_real_mm", "d_sim_fkcmd_mm", "d_real_fkcmd_mm", "dev_real_deg", "loads_real", "frame", "q_cmd", "q_read", "q_sim")},
    "median_by_name": {k: round(float(np.median(v)), 2) for k, v in sorted(by_name.items())}}

# ── G3 완주·NaN·관절 편차 ──
vals = [v for r in L for k in ("q_sim", "door_sim") if k in r for v in np.atleast_1d(r[k])] + [v for r in S for v in r["lip_sim"] + r["q_sim"]]
nan = int(np.sum(~np.isfinite(np.array(vals, float))))
dq = np.array([np.subtract(r["q_sim"], r["q_cmd"]) for r in G]); dd = np.array([r["door_sim"] - r["door_tgt"] for r in S]); jw = np.unravel_index(int(np.argmax(np.abs(dq))), dq.shape)
gates["G3_completion"] = {"pass": bool(R["n_scoop_done"] == 5 and R["n_place_done"] == 5 and nan == 0 and R["finite"]), "criterion": "scoop_done 5 · place_done 5 · 비유한값 0 (관절 목표 대비 sim 읽기 편차는 보고)",
    "n_scoop_done": R["n_scoop_done"], "n_place_done": R["n_place_done"], "n_events": R["n_events"], "n_segments": R["n_segments"], "n_goto": R["n_goto"], "n_door": R["n_door"], "sim_seconds": R["sim_seconds"], "frames": R["frames"], "nan_count": nan,
    "joint_dev_deg_sim_minus_cmd": {"per_joint_median_abs": [round(float(v), 3) for v in np.median(np.abs(dq), 0)], "per_joint_max_abs": [round(float(v), 3) for v in np.abs(dq).max(0)], "per_joint_mean_signed": [round(float(v), 3) for v in dq.mean(0)],
                                    "worst": {"joint": int(jw[1]), "dev_deg": round(float(dq[jw]), 3), **{k: G[jw[0]].get(k) for k in ("cycle", "name", "frame")}}},
    "door_dev_deg_sim_minus_tgt": {"median_abs": round(float(np.median(np.abs(dd))), 3), "max_abs": round(float(np.abs(dd).max()), 3)},
    "settle": ({"rule": R.get("settle_rule", "블렌드 후 최소 0.5 s, 전 관절 |속도| < 0.5°/s 면 정착, 상한 2.0 s"), "settle_dq_last_deg_max": (round(float(max(r["settle_dq_last_deg"] for r in S if r.get("settle_dq_last_deg") is not None)), 5) if any(r.get("settle_dq_last_deg") is not None for r in S) else None), "settle_s_median": round(float(np.median([r["settle_s"] for r in S])), 3), "settle_s_max": round(float(max(r["settle_s"] for r in S)), 3),
                "n_hit_max": int(sum(r["settle_s"] >= 1.99 for r in S)), "vel_max_dps_at_settle_max": round(float(max(r["vel_max_dps"] for r in S)), 3)} if S and "settle_s" in S[0] else "고정 0.5 s (run1)")}

# ── G4 mp4 · 키프레임 strip · 놓기 프레임 · 빈 프레임(D474) ──
mp4 = os.path.join(O, "replay.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(FR, "f_%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4], check=True)
c1 = [r for r in S if r["cycle"] == 1]; picks, j = [], 0
for r in c1:
    if j < len(STRIP_SEQ) and r["name"] == STRIP_SEQ[j]: picks.append(r); j += 1
try: FONT = ImageFont.load_default(size=18)
except TypeError: FONT = ImageFont.load_default()
sc, per_row = 0.4, 7
if picks:
    im0 = Image.open(os.path.join(FR, frames[picks[0]["frame"]])); w, h = int(im0.width * sc), int(im0.height * sc); rows = -(-len(picks) // per_row)
    strip = Image.new("RGB", (per_row * w, rows * (h + 22)), (20, 20, 20)); d = ImageDraw.Draw(strip)
    for n, r in enumerate(picks):
        x, y = (n % per_row) * w, (n // per_row) * (h + 22); strip.paste(Image.open(os.path.join(FR, frames[r["frame"]])).resize((w, h)), (x, y + 22))
        d.text((x + 4, y + 3), f"{n+1}. c{r['cycle']} {r['name']} t={r['t']:.1f}s door={r['door_sim']:.1f} dmin={min(r['dmin_mm'][b] for b in GATE_BODIES):.0f}mm", fill=(255, 255, 0), font=FONT)
    strip.save(os.path.join(O, "keyframe_strip_cycle1.png"))
rel = [r for r in c1 if r["name"] == "door30"]; tgt = [r for r in c1 if r["name"] == "place_target"]; place = {}
if len(rel) >= 2: Image.open(os.path.join(FR, frames[rel[1]["frame"]])).save(os.path.join(O, "place_frame.png")); place = {"file": "place_frame.png", "event": "cycle1 place 문 30° 개방(놓기)", "frame": rel[1]["frame"], "t": rel[1]["t"], "lip_sim": rel[1]["lip_sim"]}
if tgt: Image.open(os.path.join(FR, frames[tgt[0]["frame"]])).save(os.path.join(O, "place_target_frame.png"))
stds = [[float(np.asarray(Image.open(os.path.join(FR, frames[i])).crop(c), np.float32).std()) for c in ((0, 0, 640, 400), (640, 0, 1280, 400))] for i in range(0, nF, max(1, nF // 60))]
blank = {"n_checked": len(stds), "min_std_side": round(min(v[0] for v in stds), 1), "min_std_top": round(min(v[1] for v in stds), 1), "criterion": "측면·위 각각 화소 표준편차 > 5 (빈 프레임 = 단색)"}
gates["G4_media"] = {"pass": bool(os.path.exists(mp4) and os.path.getsize(mp4) > 0 and picks and place and blank["min_std_side"] > 5 and blank["min_std_top"] > 5), "mp4": mp4, "mp4_bytes": os.path.getsize(mp4) if os.path.exists(mp4) else 0, "frames": nF, "fps": a.fps, "blank_check": blank,
                     "strip": {"file": "keyframe_strip_cycle1.png", "n": len(picks), "names": [r["name"] for r in picks]}, "place_frame": place, "place_target_frame": "place_target_frame.png" if tgt else None,
                     "place_target_lip_sim_vs_real_mm": [round(float(v), 2) for v in (np.subtract(tgt[0]["lip_sim"], tgt[0]["lip_real"]) * 1000)] if tgt else None}

# ── 부록: 실물 tS vs 어깨 중력 토크 모델(URDF 질량) · sim 유효 강성 ──
INER = urdf_inertials(a.urdf); rows = []
TCP_M = (R.get("tcp_mass_override_kg") if R.get("tcp_mass_override_kg") is not None else (R.get("body_mass_kg") or {}).get("hand_tcp"))   # PhysX 가 실제로 쓰는 hand_tcp 질량(URDF 무질량 → 임포터 기본 1.0 kg)
INER_TCP = dict(INER)
if TCP_M: m5, c5 = INER["link5"]; INER_TCP["link5"] = (m5 + TCP_M, (m5 * c5 + TCP_M * np.array([0, 0, 0.115428])) / (m5 + TCP_M))
for r in G:
    if not r.get("loads_real") or r["loads_real"][1] is None: continue
    M = shoulder_gravity_moment(r["q_cmd"], r["door_tgt"], INER); Mt = shoulder_gravity_moment(r["q_cmd"], r["door_tgt"], INER_TCP); sag = math.radians(r["q_sim"][1] - r["q_cmd"][1])
    rows.append(dict(cycle=r["cycle"], name=r["name"], tS=r["loads_real"][1], M_model_Nm=round(M, 4), M_model_tcp_Nm=round(Mt, 4), tau_sim_reported=r["tau"][1], sag_sim_deg=round(math.degrees(sag), 3), K_eff=round(abs(Mt) / abs(sag), 2) if abs(sag) > 1e-4 else None))
if len(rows) > 3:
    ts, Mm = np.array([x["tS"] for x in rows], float), np.array([x["M_model_Nm"] for x in rows]); cc = float(np.corrcoef(ts, Mm)[0, 1]); byn = {}
    for x in rows: byn.setdefault(x["name"], []).append(x)
    Ks = [x["K_eff"] for x in rows if x["K_eff"] is not None]
    gates["appendix_loads_vs_torque"] = {"n": len(rows), "pearson_r(tS, M_model)": round(cc, 3), "tS_range": [float(ts.min()), float(ts.max())], "M_model_range_Nm": [round(float(Mm.min()), 3), round(float(Mm.max()), 3)],
        "sim_tau_reported_saturated_frac": round(float(np.mean([abs(x["tau_sim_reported"]) >= 7.99 for x in rows])), 3),
        "hand_tcp_mass_used_kg": TCP_M, "M_model_tcp_range_Nm": [round(float(min(x["M_model_tcp_Nm"] for x in rows)), 3), round(float(max(x["M_model_tcp_Nm"] for x in rows)), 3)],
        "K_eff_Nm_per_rad(M_model_tcp/sag)": {"median": round(float(np.median(Ks)), 2), "min": round(float(np.min(Ks)), 2), "max": round(float(np.max(Ks)), 2), "nominal_cfg": 800.0, "physx_readback": (R.get("physx_dof_gains") or {}).get("link1_to_link2")} if Ks else None,
        "median_by_name": {k: {"tS": float(np.median([x["tS"] for x in v])), "M_model_Nm": round(float(np.median([x["M_model_Nm"] for x in v])), 3), "sag_sim_deg": round(float(np.median([x["sag_sim_deg"] for x in v])), 3)} for k, v in sorted(byn.items())},
        "rows": rows, "urdf_masses_kg": {k: v[0] for k, v in INER.items()},
        "note": "tS = 펌웨어 부하값(단위 미상·부호 포함, 실물은 펠릿 적재 포함). M_model = URDF 질량·COM 으로 계산한 어깨축 중력 모멘트(펠릿 없음, 정역학). M_model_tcp = 여기에 PhysX 가 hand_tcp 에 부여한 질량(임포터 기본 1.0 kg, link5 +Z 115.4 mm)을 더한 것 = sim 이 실제로 받는 중력 모멘트. sim 이 보고한 applied_torque 는 Isaac Lab 의 PD 계산값(effort_limit_sim 8.0 에 포화) — 물리 토크가 아니므로 비교에 쓰지 않는다. K_eff = |M_model_tcp| / sim 처짐각 = sim 이 실제로 보인 유효 강성(명목 800 과 비교)."}
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].scatter(Mm, ts, s=14); ax[0].set_xlabel("shoulder gravity moment model [N·m] (URDF masses, no pellets)"); ax[0].set_ylabel("real tS load [fw units]"); ax[0].set_title(f"goto settle, n={len(rows)}, r={cc:.2f}"); ax[0].grid(alpha=0.3)
        tt = [r["t"] for r in L if "dmin_mm" in r]
        for b in GATE_BODIES: ax[1].plot(tt, [r["dmin_mm"][b] for r in L if "dmin_mm" in r], lw=0.8, label=b)
        ax[1].axhline(0, color="r", lw=0.8); ax[1].set_ylim(-5, 120); ax[1].set_xlabel("sim t [s]"); ax[1].set_ylabel("min signed dist to box [mm]"); ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_title("G1 timeline (clipped 120 mm)")
        fig.tight_layout(); fig.savefig(os.path.join(O, "appendix_loads_dmin.png"), dpi=110); gates["appendix_loads_vs_torque"]["png"] = "appendix_loads_dmin.png"
    except Exception as e: gates["appendix_loads_vs_torque"]["png_error"] = str(e)

gates["all_pass"] = bool(all(gates[k]["pass"] for k in ("G1_no_interference", "G2_lip_sim_vs_real", "G3_completion", "G4_media")))
json.dump(gates, open(os.path.join(O, "gates_w2_env_replay.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v) for k, v in gates.items() if k.startswith("G") or k == "all_pass"}, ensure_ascii=False))
