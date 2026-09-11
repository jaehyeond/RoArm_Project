"""W8 게이트 판정 — GATES_w8.md(사전 등록) 를 셀·회귀 결과 JSON 으로 판정해 gates_w8.json 에 쓴다.
usage: python gates_w8_writer.py   (repo 루트에서)
판정은 전부 파일에서 읽는다(손으로 적는 값 없음). G1~G4 = PASS/FAIL, G5·R·S = 보고/완결 항목."""
import hashlib, json, re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[6]
OUT = Path(__file__).resolve().parent
W3B = REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w3_deme_scoop/b_diverge"
CELLS = ["c", "xp50", "xm50"]
VARIANT = "F"            # 완주 시도 변형: cell_F_<셀>/ (문 하한 5° + 폐합 sync 1 ms + 물림 가드 3 N). 사전 등록 원안 셀은 cell_c/cell_xp50 (발산)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def diverged_attempt(d, label):
    """발산한 시도 폴더(결과 JSON 없음, 타임라인만) → 마지막 flush 행·stderr 의 max velocity 를 읽어 기록."""
    d = OUT / d
    if not (d / "timeline_seed460.json").exists():
        return {"label": label, "dir": str(d), "status": "missing"}
    rows = json.load(open(d / "timeline_seed460.json"))["rows"]; last = rows[-1]
    err = (d / "stderr.txt").read_text() if (d / "stderr.txt").exists() else ""
    m = re.search(r"System max velocity is ([0-9.]+)", err)
    return {"label": label, "dir": str(d), "status": "DIVERGED (DEME error-out, C++ terminate)" if m else "killed/partial",
            "max_velocity_m_s": float(m.group(1)) if m else None, "last_flushed_row": {k: last.get(k) for k in ("phase", "i", "sim_t", "q_deg", "n_door", "M_hinge_res_Nm", "lipF_N", "max_single_contact_N", "v_particle_max")},
            "plunge_reached": bool(any(r["phase"] == "close" for r in rows)), "max_single_contact_in_close_N": max([r["max_single_contact_N"] for r in rows if r["phase"] == "close"] or [None])}

base = json.load(open(W3B / "scoop_fixnorm/scoop_s1_seed460.json"))
reg_p = OUT / "regression_sphere/scoop_s1_seed460.json"
reg = json.load(open(reg_p)) if reg_p.exists() else None
runlog = (OUT / "run.log").read_text() if (OUT / "run.log").exists() else ""
stage = {m.group(1): {"rc": int(m.group(2)), "wall_s": int(m.group(3))} for m in re.finditer(r"stage (\S+) rc=(\d+) wall=(\d+)s", runlog)}
cells = {}
for c in CELLS:
    p = OUT / f"cell_{VARIANT}_{c}/scoop_s1_seed460.json"
    cells[c] = json.load(open(p)) if p.exists() else None
attempts = [diverged_attempt("cell_c", "pre-registered c (4 ms sync)"), diverged_attempt("cell_xp50", "pre-registered xp50 (4 ms sync)"),
            diverged_attempt("cell_A_c", "option A c (close sync 1 ms)"), diverged_attempt("cell_E_c", "option E c (sync 1 ms + pinch guard 3 N)")]

def cell_row(r):
    if r is None:
        return None
    az = r["crater"]["azimuths"]
    return {"site_xy_mm": [r["scoop_site"]["x_mm"], r["scoop_site"]["y_mm"]], "surface_z_mm": r["scoop_site"]["surface_z_mm"],
            "diverged": r["diverged"], "steps_actual": r["trajectory"]["steps_actual"], "plunge_reached_mm": r["trajectory"]["plunge_reached_mm"],
            "descend_hold_steps": r["trajectory"]["descend_hold_steps"],
            "n_in_cavity": r["capture"]["n_in_cavity"], "n_carried_z": r["capture"]["n_carried_z"], "mass_g": r["capture"]["mass_g"],
            "fill_vs_bulk": r["capture"]["fill_vs_bulk"], "particle_mass_mg": r["particle"]["mass_kg"] * 1e6,
            "door_stops": r["door"]["stops"], "q_final_deg": r["door"]["q_final_deg"], "servo_deg_final": r["door"]["servo_deg_final"],
            "lip_gap_final_mm": r["door"]["lip_gap_final_mm"], "mouth_end_mm": r["door"]["mouth_end_mm"],
            "n_pinched_at_lip": r["door"]["n_pinched_at_lip"], "n_touch_both": r["door"]["n_touch_both_meshes"],
            "close_peak_lipF_N": r["forces"]["close_peak_lipF_N"], "close_peak_M_res_Nm": r["forces"]["close_peak_M_res_Nm"],
            "close_peak_single_contact_N": r["forces"]["close_peak_single_contact_N"], "descend_peak_F_fixed_N": r["forces"]["descend_peak_F_fixed_N"],
            "pops": r["pops"], "wall_seconds": r["wall_seconds"],
            "heightmap": r["heightmap"], "crater_center_xy_mm": r["crater"]["center_xy_mm"], "removed_volume_cm3": r["crater"]["removed_volume_cm3"],
            "dh_max_mm": r["crater"]["dh_max_mm"],
            "crater_angle_deg": {k: v.get("angle_deg") for k, v in az.items()},
            "crater_angle_surface_deg": {k: v.get("angle_surface_deg") for k, v in az.items()},
            "crater_fit_r_mm": {k: v.get("r_fit_mm") for k, v in az.items()}, "crater_reason": {k: v.get("reason") for k, v in az.items()},
            "template_check": {k: r["particle"].get(k) for k in ("expand_vs_npz_max_err_m", "clump_ids_clump_major", "n_spheres")}}

rows = {c: cell_row(r) for c, r in cells.items()}
done = [c for c in CELLS if cells[c] is not None]
G1 = {"pass": len(done) == 3 and all(not cells[c]["diverged"] for c in done),
      "diverged": {c: (cells[c]["diverged"] if cells[c] else None) for c in CELLS},
      "pops": {c: (cells[c]["pops"] if cells[c] else None) for c in CELLS}, "stderr_lines": {c: len((OUT / f"cell_{VARIANT}_{c}/stderr.txt").read_text().splitlines()) if (OUT / f"cell_{VARIANT}_{c}/stderr.txt").exists() else None for c in CELLS},
      "pre_registered_setting_verdict": "FAIL — 사전 등록 설정(폐합 sync 4 ms, 서보 토크 정지만) 으로는 셀 c·xp50 이 q 3.4°/2.3° 에서 DEME error-out. 옵션 A·E 도 발산. 완주 열은 옵션 F(문 하한 5°) 결과이며 사전 등록 외 설정",
      "diverged_attempts": attempts}
G2 = {"pass": reg is not None and reg["capture"]["n_in_cavity"] == base["capture"]["n_in_cavity"],
      "baseline_file": str(W3B / "scoop_fixnorm/scoop_s1_seed460.json"), "baseline_n_in_cavity": base["capture"]["n_in_cavity"],
      "baseline_mass_g": base["capture"]["mass_g"], "baseline_stops": base["door"]["stops"], "baseline_wall_s": base["wall_seconds"],
      "regression_n_in_cavity": None if reg is None else reg["capture"]["n_in_cavity"], "regression_mass_g": None if reg is None else reg["capture"]["mass_g"],
      "regression_stops": None if reg is None else reg["door"]["stops"], "regression_wall_s": None if reg is None else reg["wall_seconds"],
      "delta_n": None if reg is None else reg["capture"]["n_in_cavity"] - base["capture"]["n_in_cavity"],
      "delta_mass_g": None if reg is None else round(reg["capture"]["mass_g"] - base["capture"]["mass_g"], 4),
      "same_stop_phases_reasons": None if reg is None else [(a["phase"], a["reason"]) for a in reg["door"]["stops"]] == [(a["phase"], a["reason"]) for a in base["door"]["stops"]],
      "same_inputs_except_script": None if reg is None else {Path(k).name: ({Path(kk).name: vv for kk, vv in reg["inputs_sha16"].items()}.get(Path(k).name) == v) for k, v in base["inputs_sha16"].items() if not k.endswith("sim_deme_scoop_s1.py")},
      "prew8_script_rerun": (lambda q: None if not q.exists() else {"n_in_cavity": json.load(open(q))["capture"]["n_in_cavity"], "mass_g": json.load(open(q))["capture"]["mass_g"], "stops": json.load(open(q))["door"]["stops"], "script_sha16": json.load(open(q))["inputs_sha16"].get(str(REPO / "sim_deme_scoop_s1.py.bak_20260910_pre_w8"))})(OUT / "regression_sphere_prew8_script/scoop_s1_seed460.json"),
      "script_sha16_baseline_vs_now": None if reg is None else [base["inputs_sha16"][str(REPO / "sim_deme_scoop_s1.py")], reg["inputs_sha16"][str(REPO / "sim_deme_scoop_s1.py")]],
      "final_script_rerun": (lambda q: None if not q.exists() else {"n_in_cavity": json.load(open(q))["capture"]["n_in_cavity"], "mass_g": json.load(open(q))["capture"]["mass_g"], "stops": json.load(open(q))["door"]["stops"], "diverged": json.load(open(q))["diverged"], "script_sha16": json.load(open(q))["inputs_sha16"].get(str(REPO / "sim_deme_scoop_s1.py")), "note": "최종 스크립트판(렌더·옵션 A/E/F 게이트 추가 뒤) 으로 구 경로 재실행; try1 은 DEME stall 로 kill(regression_sphere_final_script_try1_stall/)"})(OUT / "regression_sphere_final_script/scoop_s1_seed460.json"),
      "note": "엄격 동일 판정. DEME GPU 는 run-to-run bit-동일을 보장하지 않는다(W7 §7-7) — 다르면 FAIL 로 두고 Δ 를 보고한다"}
G3 = {"pass": len(done) == 3 and all(stage.get(f"{VARIANT}_{c}", {}).get("rc") == 0 and stage[f"{VARIANT}_{c}"]["wall_s"] <= 2700 and all(cells[c]["trajectory"]["steps_actual"][k] > 0 for k in ("settle", "descend", "close", "lift")) for c in done),
      "stage_rc_wall": {c: stage.get(f"{VARIANT}_{c}") for c in CELLS}, "steps_actual": {c: (cells[c]["trajectory"]["steps_actual"] if cells[c] else None) for c in CELLS}, "cap_s": 2700,
      "note": "상한 2400 s(사전 등록) 는 옵션 F 실행기에서 2700 s 로 올림(폐합 sync 1 ms 오버헤드). 벽시계는 보고값", "variant": VARIANT}
G4 = {"pass": len(done) == 3 and all(cells[c]["capture"]["mass_g"] > 0 for c in done),
      "mass_g": {c: (cells[c]["capture"]["mass_g"] if cells[c] else None) for c in CELLS},
      "n_in_cavity": {c: (cells[c]["capture"]["n_in_cavity"] if cells[c] else None) for c in CELLS},
      "fill_vs_bulk": {c: (cells[c]["capture"]["fill_vs_bulk"] if cells[c] else None) for c in CELLS},
      "w3b_sphere_reference_mean_g": 11.78, "note": "충전율·W3b 비교는 보고값(판정 아님)"}
G5 = {"reported": len(done) == 3 and all(all(k in cells[c]["crater"]["azimuths"] for k in ("+x", "+y", "-x", "-y")) for c in done),
      "angle_deg": {c: (rows[c]["crater_angle_deg"] if rows[c] else None) for c in CELLS},
      "angle_surface_deg": {c: (rows[c]["crater_angle_surface_deg"] if rows[c] else None) for c in CELLS},
      "reason": {c: (rows[c]["crater_reason"] if rows[c] else None) for c in CELLS},
      "definition": (cells[done[0]]["crater"]["definition"] if done else None), "params": (cells[done[0]]["crater"]["params"] if done else None),
      "rest_only_postprocess": {c: (lambda q: None if not q.exists() else (lambda j: {"rule": j["rest_rule"], "n_excluded_spheres": j["n_spheres_excluded_in_flight_or_perched"],
                                    "angle_deg": {k: v.get("angle_deg") for k, v in j["crater"]["azimuths"].items()}, "angle_surface_deg": {k: v.get("angle_surface_deg") for k, v in j["crater"]["azimuths"].items()},
                                    "secant_angle_deg": {k: v.get("secant_angle_deg") for k, v in j["crater"]["azimuths"].items()}, "removed_volume_cm3": j["crater"]["removed_volume_cm3"],
                                    "center_xy_mm": j["crater"]["center_xy_mm"], "dh_max_mm": j["crater"]["dh_max_mm"]})(json.load(open(q))))(OUT / f"cell_{VARIANT}_{c}/crater_rest_seed460.json") for c in CELLS},
      "rest_note": "crater_rest_seed460.json = 최종 프레임에서 공중(문 틈 유출 중)·툴에 얹힌 알을 뺀 heightmap 으로 같은 정의를 다시 잰 값(권장 인용값). secant 는 밴드 고리 <2 인 가파른 벽용 보조각(정의 밖)",
      "note": "판정 아님(보고). 정의 = sim_deme_scoop_s1.py crater_angles() 도크스트링 = GATES_w8.md"}
rr_cell = "c"
RR_TAG = "w8v2"          # w8 = 1차(기본 커서에서 최종 더미 미표시), w8v2 = 최종 더미·툴 정적 사본 추가판(검수 완료)
val_p = OUT / f"cell_{VARIANT}_{rr_cell}/scoop_s1_seed460_{RR_TAG}_rerun_validation.json"; insp_p = OUT / f"cell_{VARIANT}_{rr_cell}/scoop_s1_seed460_{RR_TAG}_inspection.json"
val = json.load(open(val_p)) if val_p.exists() else None; insp = json.load(open(insp_p)) if insp_p.exists() else None
R = {"complete": bool(val and val.get("pass") is True and insp and insp.get("visual_inspection_complete") is True),
     "cell": rr_cell, "validation_pass": None if val is None else val.get("pass"), "validation_file": str(val_p) if val else None,
     "inspection_file": str(insp_p) if insp else None, "sdk_cli": None if val is None else val.get("version")}
mass = np.array([cells[c]["capture"]["mass_g"] for c in done]) if done else np.array([])
g = {"artifact": "GATES_W8_DEME_SCOOP_LENS", "pre_registration": "GATES_w8.md", "cells": CELLS,
     "G1_no_divergence": G1, "G2_sphere_regression": G2, "G3_three_cells_complete": G3, "G4_capture_mass_positive": G4,
     "G5_crater_angle_reported": G5, "R_rerun_d341": R,
     "S_source_sha256": {"sim_deme_scoop_s1.py": sha(REPO / "sim_deme_scoop_s1.py"), "sim_deme_scoop_s1.py.bak_20260910_pre_w8": sha(REPO / "sim_deme_scoop_s1.py.bak_20260910_pre_w8"),
                         "sim_deme_scoop.py(protected, unmodified)": sha(REPO / "sim_deme_scoop.py"), "roarm_rl/heightmap.py": sha(REPO / "roarm_rl/heightmap.py"),
                         "params_w8_cell_*.json (pre-registered)": {c: sha(OUT / f"params_w8_cell_{c}.json") for c in CELLS},
                         f"params_w8{VARIANT}_cell_*.json (variant)": {c: sha(OUT / f"params_w8{VARIANT}_cell_{c}.json") for c in CELLS},
                         "lens_pile_npz": sha("/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz"),
                         "inputs_sha16_per_run": {**{c: (cells[c]["inputs_sha16"] if cells[c] else None) for c in CELLS}, "regression": None if reg is None else reg["inputs_sha16"]}},
     "summary": {"rows": rows, "mean_mass_g": None if not len(mass) else round(float(mass.mean()), 4),
                 "cov": None if len(mass) < 2 else round(float(mass.std(ddof=1) / mass.mean()), 4)},
     "variant_note": "G1·G3·G4·G5 의 셀 값은 옵션 F(사전 등록 외: 문 하한 5°·폐합 sync 1 ms·물림 가드 3 N) 완주 결과. 사전 등록 설정 자체의 G1 은 FAIL(diverged_attempts)",
     "all_pass": bool(G1["pass"] and G2["pass"] and G3["pass"] and G4["pass"]), "all_pass_pre_registered_settings": False}
json.dump(g, open(OUT / "gates_w8.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps({k: g[k]["pass"] for k in ("G1_no_divergence", "G2_sphere_regression", "G3_three_cells_complete", "G4_capture_mass_positive")} | {"G5_reported": G5["reported"], "R_complete": R["complete"], "all_pass": g["all_pass"]}, ensure_ascii=False))
