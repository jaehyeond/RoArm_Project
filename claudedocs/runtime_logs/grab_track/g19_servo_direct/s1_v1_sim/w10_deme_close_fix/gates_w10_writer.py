"""W10 게이트 판정 — GATES_w10.md(사전 등록) 를 셀·회귀 결과 파일로 판정해 gates_w10.json 에 쓴다.
usage: python gates_w10_writer.py   (어디서든; 경로는 이 파일 기준)
판정은 전부 파일에서 읽는다(손으로 적는 값 없음). G0~G3 = PASS/FAIL, G4·R·S = 보고/완결 항목."""
import hashlib, json, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]; OUT = Path(__file__).resolve().parent
W8 = OUT.parent / "w8_deme_scoop_lens"; W3B = OUT.parent / "w3_deme_scoop/b_diverge"
LENS = Path("/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
J = lambda p: json.load(open(p)) if Path(p).exists() else None
runlog = (OUT / "run.log").read_text() if (OUT / "run.log").exists() else ""
stage = {}
for m in re.finditer(r"stage (\S+)(?: attempt (\d))? rc=(\d+) wall=(\d+)s(?: stalled=(\d))?", runlog):
    stage.setdefault(m.group(1), []).append({"attempt": m.group(2), "rc": int(m.group(3)), "wall_s": int(m.group(4)), "stalled": m.group(5)})

# G0 회귀
base = J(W3B / "scoop_fixnorm/scoop_s1_seed460.json")
reg_path = OUT / "regression_sphere_resume_20260911/scoop_s1_seed460.json"
if not reg_path.exists():
    reg_path = OUT / "regression_sphere/scoop_s1_seed460.json"
reg = J(reg_path)
reg_script_current = bool(reg and reg["inputs_sha16"].get(str(REPO / "sim_deme_scoop_s1.py")) == sha(REPO / "sim_deme_scoop_s1.py")[:16])
lo, hi = round(base["capture"]["n_in_cavity"] * 0.85), round(base["capture"]["n_in_cavity"] * 1.15)
G0 = {"pass": bool(reg_script_current and reg and not reg["diverged"] and lo <= reg["capture"]["n_in_cavity"] <= hi and reg["door"]["stops"] and all(s["reason"] == "servo_stall" for s in reg["door"]["stops"])),
      "regression_file": str(reg_path), "regression_script_matches_current": reg_script_current,
      "rule": f"diverged false ∧ n_in_cavity ∈ [{lo}, {hi}] (W3b {base['capture']['n_in_cavity']} ±15 %) ∧ stops 전부 servo_stall",
      "baseline": {"file": str(W3B / "scoop_fixnorm/scoop_s1_seed460.json"), "n_in_cavity": base["capture"]["n_in_cavity"], "mass_g": base["capture"]["mass_g"], "stops": base["door"]["stops"]},
      "regression": None if reg is None else {"n_in_cavity": reg["capture"]["n_in_cavity"], "mass_g": reg["capture"]["mass_g"], "stops": reg["door"]["stops"], "diverged": reg["diverged"],
                                              "wall_s": reg["wall_seconds"], "pops": reg["pops"], "script_sha16": reg["inputs_sha16"].get(str(REPO / "sim_deme_scoop_s1.py"))},
      "w8_history_n_in_cavity": {"W3b": 315, "W8_patched": 273, "W8_prew8_backup": 301, "W8_final_script": 300}, "stage": stage.get(reg_path.parent.name)}

# G1 진단
ev = J(OUT / "cell_diag_c/diverge_event_seed460.json"); dres = J(OUT / "cell_diag_c/scoop_s1_seed460.json")
tl = J(OUT / "cell_diag_c/timeline_seed460.json")
diag_rows = [r for r in (tl or {}).get("rows", []) if "diag" in r]
def last3(ev):
    if not ev:
        return None
    rows = []
    for e in ev["culprit_last4"]:
        row = {"sim_t": e["sim_t"], "q_deg": e["q_deg"], "pop_sync": e["pop_sync"], "in_near_set": e["in_ring_near_set"], "v_m_s": e.get("v_m_s"), "centre_rel_lip_mm": e.get("centre_rel_lip_mm")}
        for mkey in ("fixed", "door"):
            g = (e.get("vs_mesh") or {}).get(mkey)
            row[mkey] = None if g is None else {k: g.get(k) for k in ("group", "tri", "k", "h_mm", "inside", "pen_geo_mm", "deme_pen_mm", "ghost")} | {"F_max_on_culprit_N": g["contacts_on_culprit"]["F_max_N"], "n_contacts_on_culprit": g["contacts_on_culprit"]["n"]}
        rows.append(row)
    return rows
G1 = {"pass": bool(ev and ev.get("culprit_last4") and ev.get("mechanism") in ("ghost", "squeeze", "sphere_sphere", "unresolved")),
      "event_file": str(OUT / "cell_diag_c/diverge_event_seed460.json") if ev else None, "mechanism": None if not ev else ev["mechanism"],
      "ghost_any_prepop": None if not ev else ev["ghost_any_prepop"], "squeeze_seq_F_max_N": None if not ev else ev["squeeze_seq_F_max_N"],
      "trigger": None if not ev else ev["trigger"], "culprit": None if not ev else ev["culprit"], "culprit_now": None if not ev else ev["culprit_now"],
      "culprit_last4_table": last3(ev),
      "contact_detail_summary": None if not ev else ({"error": ev["contact_detail"].get("error")} if "error" in ev["contact_detail"] else
                                                     {"wall_s": ev["contact_detail"]["wall_s"], "n_pairs_total": ev["contact_detail"]["n_pairs_total"], "n_pairs_culprit": ev["contact_detail"]["n_pairs_culprit"],
                                                      "top_pairs": ev["contact_detail"]["pairs"][:6], "top_is_sphere_sphere": ev["contact_detail"].get("top_is_sphere_sphere")}),
      "diag_rows_n": len(diag_rows), "diag_first_q_deg": diag_rows[0]["q_deg"] if diag_rows else None, "diag_last_q_deg": diag_rows[-1]["q_deg"] if diag_rows else None,
      "result": None if not dres else {"diverged": dres["diverged"], "stops": dres["door"]["stops"], "v_particle_max": dres["pops"]["v_particle_max_m_s"], "wall_s": dres["wall_seconds"], "steps_actual": dres["trajectory"]["steps_actual"]},
      "stage": stage.get("diag_c")}

# G2/G3 완주
def cell_summary(r):
    if r is None:
        return None
    az = r["crater"]["azimuths"]
    return {"diverged": r["diverged"], "stops": r["door"]["stops"], "q_final_deg": r["door"]["q_final_deg"], "servo_deg_final": r["door"]["servo_deg_final"],
            "lip_gap_final_mm": r["door"]["lip_gap_final_mm"], "mouth_end_mm": r["door"]["mouth_end_mm"], "n_pinched_at_lip": r["door"]["n_pinched_at_lip"], "n_touch_both": r["door"]["n_touch_both_meshes"],
            "n_in_cavity": r["capture"]["n_in_cavity"], "n_carried_z": r["capture"]["n_carried_z"], "mass_g": r["capture"]["mass_g"], "fill_vs_bulk": r["capture"]["fill_vs_bulk"],
            "close_peak_lipF_N": r["forces"]["close_peak_lipF_N"], "close_peak_M_res_Nm": r["forces"]["close_peak_M_res_Nm"], "close_peak_single_contact_N": r["forces"]["close_peak_single_contact_N"],
            "descend_peak_F_fixed_N": r["forces"]["descend_peak_F_fixed_N"], "plunge_reached_mm": r["trajectory"]["plunge_reached_mm"], "descend_hold_steps": r["trajectory"]["descend_hold_steps"],
            "steps_actual": r["trajectory"]["steps_actual"], "pops": r["pops"], "wall_s": r["wall_seconds"], "removed_volume_cm3": r["crater"]["removed_volume_cm3"], "dh_max_mm": r["crater"]["dh_max_mm"],
            "crater_angle_deg": {k: v.get("angle_deg") for k, v in az.items()}, "crater_reason": {k: v.get("reason") for k, v in az.items()},
            "params_key": {k: r["params"].get(k) for k in ("E_pa", "timestep_s", "close_deg_s", "dt_sync_close_s", "door_pinch_guard_N", "door_min_q_deg", "error_out_vel", "diag_fine_sync_s", "max_velocity_m_s")}}
def rest_angles(cell):
    c = J(cell / "crater_rest_seed460.json")
    if not c:
        return None
    return {k: {"angle_deg": v.get("angle_deg"), "angle_surface_deg": v.get("angle_surface_deg"), "secant_deg": v.get("secant_angle_deg"), "reason": v.get("reason")} for k, v in c["crater"]["azimuths"].items()} | {"removed_volume_cm3": c["crater"]["removed_volume_cm3"], "dh_max_mm": c["crater"]["dh_max_mm"]}
STAGES = ("DE_c", "DE_dt2e6_c", "DE_E1e8_c")  # dt-only 우선: RESUME_W10_20260911.md §2–3
LIMITS_S = {"DE_c": 5400, "DE_dt2e6_c": 14400, "DE_E1e8_c": 14400}
cells = {k: J(OUT / f"cell_{k}/scoop_s1_seed460.json") for k in STAGES}
events = {k: J(OUT / f"cell_{k}/diverge_event_seed460.json") for k in STAGES}
ok_reason = lambda r: r is not None and not r["diverged"] and bool(r["door"]["stops"]) and r["params"].get("door_min_q_deg") is None and all(s["reason"] in ("servo_stall", "reached_close_end") for s in r["door"]["stops"])
complete = {k: (r is not None and not r["diverged"]) for k, r in cells.items()}
torque_stop = {k: ok_reason(r) for k, r in cells.items()}
def process_ok(k):
    last = stage.get(k, [{}])[-1]
    return last.get("rc") == 0 and 0 < last.get("wall_s", float("inf")) <= LIMITS_S[k] and last.get("stalled") in (None, "0")
winner = next((k for k in STAGES if torque_stop[k] and process_ok(k)), None)
G2 = {"pass": winner is not None, "winner": winner, "completed": complete, "torque_stop": torque_stop,
      "cells": {k: cell_summary(r) for k, r in cells.items()}, "events": {k: (None if e is None else {"mechanism": e["mechanism"], "trigger": e["trigger"], "ghost_any_prepop": e["ghost_any_prepop"], "culprit": e["culprit"]}) for k, e in events.items()},
      "stage": {k: stage.get(k) for k in STAGES}, "wall_limits_s": LIMITS_S,
      "rule": "diverged false ∧ rc 0 ∧ wall 상한 내 ∧ door_floor 없음 ∧ 정지 기록 존재 ∧ stops reason ∈ {servo_stall, reached_close_end} (pinch_guard 는 완주지만 토크 정지 아님)"}
G3 = {"pass": bool(winner and cells[winner]["capture"]["mass_g"] > 0), "mass_g": None if not winner else cells[winner]["capture"]["mass_g"]}
# G4 비교표
F = J(W8 / "cell_F_c/scoop_s1_seed460.json")
G4 = {"reported": winner is not None, "w8_option_F_c": cell_summary(F) | {"crater_rest": rest_angles(W8 / "cell_F_c")},
      "w10_cell": None if not winner else cell_summary(cells[winner]) | {"crater_rest": rest_angles(OUT / f"cell_{winner}")},
      "w3b_sphere_reference": {"n_in_cavity": base["capture"]["n_in_cavity"], "mass_g": base["capture"]["mass_g"], "stops": base["door"]["stops"], "lip_gap_final_mm": base["door"]["lip_gap_final_mm"]},
      "render_timeline": {k: (str(OUT / f"render_timeline_w10_{t}.npz") if (OUT / f"render_timeline_w10_{t}.npz").exists() else None) for k, t in (("diag_c", "diag"), ("DE_c", "DE"), ("DE_dt2e6_c", "DE_dt2e6"), ("DE_E1e8_c", "DE_E1e8"))}}
# R Rerun
rcell = winner or ("diag_c" if ev else None)
R = {"cell": rcell, "validation": None, "inspection": None, "pass": False}
if rcell:
    v = J(OUT / f"cell_{rcell}/scoop_s1_seed460_w10_rerun_validation.json"); ins = J(OUT / f"cell_{rcell}/scoop_s1_seed460_w10_inspection.json")
    R.update(validation=None if not v else {k: v.get(k) for k in ("pass", "rerun_sdk_version", "cli_version", "rrd_verify", "entity_paths_ok", "timelines_ok", "components_ok", "blueprint_ok", "screenshot_ok") if k in v} | {"pass": v.get("pass")},
             inspection=None if not ins else {"observations_n": len(ins.get("observations", [])), "limitations_n": len(ins.get("limitations", [])), "png": ins.get("png")})
    R["pass"] = bool(v and v.get("pass") and ins and ins.get("observations"))
# S sha
files = {"sim_deme_scoop_s1.py": REPO / "sim_deme_scoop_s1.py", "sim_deme_scoop_s1.py.bak_20260910_pre_w10": REPO / "sim_deme_scoop_s1.py.bak_20260910_pre_w10",
         "sim_deme_scoop_s1.py.bak_20260910_pre_w8": REPO / "sim_deme_scoop_s1.py.bak_20260910_pre_w8", "sim_deme_scoop.py(protected)": REPO / "sim_deme_scoop.py",
         "roarm_rl/heightmap.py": REPO / "roarm_rl/heightmap.py", "lens_pile.npz": LENS, "GATES_w10.md": OUT / "GATES_w10.md"}
files.update({p.name: p for p in OUT.glob("params_w10_*.json")})
S = {"sha256": {k: (sha(v) if Path(v).exists() else None) for k, v in files.items()},
     "inputs_sha16_per_run": {k: (r["inputs_sha16"] if r else None) for k, r in {"regression_sphere": reg, "diag_c": dres, **cells}.items()},
     "script_diff": str(OUT / "sim_deme_scoop_s1_w10.diff")}
g = {"artifact": "GATES_W10_DEME_CLOSE_FIX", "pre_registration": str(OUT / "GATES_w10.md"), "G0_sphere_regression": G0, "G1_mechanism_identified": G1, "G2_torque_stop_completion": G2,
     "G3_capture_mass_positive": G3, "G4_comparison_table": G4, "R_rerun_d341": R, "S_source_sha256": S,
     "all_pass": bool(G0["pass"] and G1["pass"] and G2["pass"] and G3["pass"])}
json.dump(g, open(OUT / "gates_w10.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v) for k, v in g.items() if k.startswith(("G", "R", "all"))}, ensure_ascii=False))
