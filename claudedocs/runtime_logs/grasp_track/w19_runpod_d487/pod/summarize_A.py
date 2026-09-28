"""A(전체 사이클) 결과 요약 — 생산 JSON/NPZ/러너 영수증에서 사실만 뽑는다. 판정 승격 없음(독립 감사 전 '생산 기록값').
사용: python summarize_A.py <run_dir> [out.json]"""
import json, sys
from pathlib import Path
import numpy as np
run = Path(sys.argv[1])
d = json.load(open(run / "w13_cycle_seed460.json"))
rec = json.load(open(run / "EXECUTION_RECEIPT.json")) if (run / "EXECUTION_RECEIPT.json").exists() else {}
st = json.load(open(run / "RUN_STATUS.json")) if (run / "RUN_STATUS.json").exists() else {}
z = np.load(run / "w13_cycle_seed460.npz", allow_pickle=False)
m = json.loads(str(z["metadata_json"])); inv = {v: k for k, v in m["phase_code"].items()}
ph = np.array([inv[int(c)] for c in z["sync_phase_code"]]); w = np.asarray(z["sync_wall_elapsed_s"], float); dw = np.diff(w, prepend=0.0)
t = np.asarray(z["sync_t_s"], float); dt = np.diff(t, prepend=0.0)
phases = {}
for p in dict.fromkeys(ph.tolist()):
    s = ph == p; phases[p] = {"n_sync": int(s.sum()), "wall_s": round(float(dw[s].sum()), 1), "sim_s": round(float(dt[s].sum()), 4),
                              "wall_per_sim_s": round(float(dw[s].sum() / max(dt[s].sum(), 1e-12)), 1)}
dl = d.get("delivery", {})
out = {
 "artifact": "W19_A_SUMMARY_PRODUCTION_VALUES", "run_dir": str(run),
 "runner": {"state": st.get("state"), "rc": rec.get("rc"), "wall_s": rec.get("wall_s"), "timed_out": rec.get("timed_out"), "killed": rec.get("killed"),
            "started_local": rec.get("started_local"), "ended_local": rec.get("ended_local")},
 "sim": {"wall_seconds": d.get("wall_seconds"), "cycle_sim_time_s": dl.get("cycle_sim_time_s"), "n_sync": int(w.size), "n_particle_frames": int(np.asarray(z["particle_frame_sync_index"]).size) if "particle_frame_sync_index" in z.files else None,
         "diverged": d.get("diverged"), "abort_class": d.get("abort_class"), "fail_reason": str(d.get("fail_reason")), "stopped_early_after_phase": d.get("stopped_early_after_phase"),
         "signal_received": d.get("signal_received"), "max_wall_s": d.get("max_wall_s")},
 "pops": d.get("pops"), "door_stops": [(s.get("phase"), s.get("reason"), s.get("sim_t"), s.get("max_single_contact_N")) for s in d.get("door", {}).get("stops", [])],
 "door_final": {k: d.get("door", {}).get(k) for k in ("q_final_cmd_deg", "q_final_actual_deg", "servo_deg_final")},
 "decisions": [(x["tag"], round(x["sim_t"], 6), x["counts"]) for x in d.get("decisions", [])],
 "delivery": {k: dl.get(k) for k in ("definite_delivered_n", "definite_delivered_g", "possible_delivered_n", "possible_delivered_g", "inventory_final", "layer1_accounting_integrity")},
 "settlement_window": dl.get("settlement_window"),
 "home": {k: d.get("trajectory", {}).get(k) for k in ("home_start_lip_mm", "home_end_lip_mm", "home_pose_return_err_mm", "syncs_per_phase")},
 "phases_wall": phases, "total_wall_s": round(float(w[-1]), 1), "total_sim_s": round(float(t[-1]), 6),
 "note": "생산 기록값(rev32, 규약 v2 분류). 독립 감사·rev31 파생 회계 전. 판정 승격 아님."}
print(json.dumps({k: out[k] for k in ("runner", "sim", "pops", "door_stops", "door_final", "delivery", "total_wall_s", "total_sim_s")}, ensure_ascii=False, indent=1))
print("decisions:"); [print(" ", x) for x in out["decisions"]]
print("phases_wall:"); [print(" ", p, v) for p, v in phases.items()]
print("settlement:", json.dumps(out["settlement_window"], ensure_ascii=False)[:400])
if len(sys.argv) > 2: open(sys.argv[2], "w").write(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
