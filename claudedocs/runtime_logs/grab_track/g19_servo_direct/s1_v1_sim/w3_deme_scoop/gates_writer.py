"""G1~G5 판정 JSON — 완주한 실행이 없으므로 시도별 타임라인·stderr 를 직접 읽어 증거와 함께 FAIL/판정불가를 기록한다 (D476)."""
import json, hashlib, re, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent
REPO = OUT.parents[5]
def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
attempts = {
    "plunge25_provisional": (OUT / "attempt_plunge25_seed460_timeline_diverged.json", OUT / "attempt_plunge25_seed460_stderr.txt", "E 5e6 · dt 1e-5 · 잠김 25 · 하강 25 mm/s · sync 0.5 ms · 팔 힘 상한 2 N"),
    "plunge10_provisional_seed460": (OUT / "plunge10_failed/timeline_seed460.json", OUT / "plunge10_failed/stderr_seed460.txt", "E 5e6 · dt 1e-5 · 잠김 10"),
    "plunge25_stiff_E1e8": (OUT / "stiff_E1e8/timeline_seed460.json", OUT / "stiff_E1e8/stderr_seed460.txt", "E 1e8 · dt 2e-6 · 잠김 25 · 팔 힘 상한 6 N"),
}
ev = {}
for k, (tl, err, desc) in attempts.items():
    d = json.load(open(tl)); rows = d["rows"]; last = rows[-1]
    m = re.search(r"max velocity is ([0-9.]+)", open(err).read())
    desc_rows = [r for r in rows if r["phase"] == "descend"]; clos = [r for r in rows if r["phase"] == "close"]
    ev[k] = {"setting": desc, "timeline": str(tl.relative_to(OUT)), "timeline_sha16": sha16(tl), "state": d["state"],
             "rows": len(rows), "last_phase": last["phase"], "last_sim_t_s": last["sim_t"], "last_z_lip_mm": last["z_lip_mm"],
             "surface_z_mm": None, "insertion_reached_mm": None,
             "door_q_deg_at_abort": last["q_deg"], "abort_max_velocity_m_s": float(m.group(1)) if m else None,
             "v_particle_max_seen_m_s": max(r.get("v_particle_max", 0) for r in rows),
             "max_single_contact_N": max(r["max_single_contact_N"] for r in rows),
             "max_Fz_fixed_up_N": max(r.get("Fz_fixed_up_N", 0) for r in rows),
             "close_max_M_res_Nm": max((r["M_hinge_res_Nm"] for r in clos), default=None)}
    z0 = desc_rows[0]["z_lip_mm"] if desc_rows else None
    if z0 is not None:            # 하강 시작 립 = 펠릿면 + 10 mm (approach_gap)
        ev[k]["surface_z_mm"] = round(z0 - 10.0 + 0.125, 2); ev[k]["insertion_reached_mm"] = round(ev[k]["surface_z_mm"] - last["z_lip_mm"], 2)
inputs = {str(p): sha16(p) for p in [REPO / "claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz",
          REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/fixed_ALL.stl",
          REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/door_ALL_jawframe.stl",
          REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/design.json",
          REPO / "local_assets/roarm_m3/urdf/s1_meta.json", REPO / "sim_deme_scoop_s1.py",
          OUT / "params_provisional_plunge10.json", OUT / "params_stiff_E1e8_plunge25.json"]}
g = {"artifact": "GATES_W3_DEME_SCOOP_S1", "date": "2026-09-09", "completed_runs": 0, "inputs_sha16": inputs, "attempt_evidence": ev,
     "G1_no_divergence": {"pass": False, "verdict": "FAIL", "reason": "3 설정 모두 DEME 'System max velocity exceeded' 로 C++ abort (규정 툴이 더미에 들어간 뒤 단일 접촉이 0.5 ms 안에 0.05→68 N 으로 뛰며 입자가 40~140 m/s 로 튐). 잠김 25 는 20 mm, 잠김 10 은 문 폐합 q 5.9° 에서, E 1e8 은 잠김 2 mm 에서 중단"},
     "G2_door_closure": {"pass": False, "verdict": "판정 불가(부분 증거)", "partial": "잠김 10 mm 시도에서 문이 27.3°→5.9°(립 틈 ≈12 mm) 까지 닫히는 동안 힌지 저항 모멘트 최대 0.013 N·m(립 등가 0.11 N, 서보 정지 1.76 N·m 의 1 %) — 그 각까지는 서보 정지 없음. 립 맞닿음·물림은 미도달"},
     "G3_capture_mass": {"pass": False, "verdict": "판정 불가", "reason": "상승까지 간 실행 없음 → 포획 질량 산출 0건"},
     "G4_seed_cov": {"pass": False, "verdict": "판정 불가", "reason": "완주 실행 0"},
     "G5_wall_le_5min": {"pass": None, "verdict": "참고값", "note": "해석적 셸(204+204 삼각형) 기준 0.22~0.3 s/스텝(4 ms sync). 명목 경로(하강 175·폐합 153·상승 134 + 재닫기 ≤30, sync 4 ms)면 ≈ 2~2.5 min 으로 G5 안. 하강 sync 0.5 ms 옵션이면 ≈ 4 min. 원본 STL 6.5k 삼각형이면 12 s/스텝(불가)"},
     "all_pass": False}
json.dump(g, open(OUT / "gates_w3_deme_scoop.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps({k: v.get("verdict") for k, v in g.items() if k.startswith("G")}, ensure_ascii=False))
for k, v in ev.items():
    print(k, "| state", v["state"], "| last", v["last_phase"], "z_lip", v["last_z_lip_mm"], "| insert", v["insertion_reached_mm"], "mm | q", v["door_q_deg_at_abort"], "| abort v", v["abort_max_velocity_m_s"], "| single max", v["max_single_contact_N"], "| Fz max", v["max_Fz_fixed_up_N"])
