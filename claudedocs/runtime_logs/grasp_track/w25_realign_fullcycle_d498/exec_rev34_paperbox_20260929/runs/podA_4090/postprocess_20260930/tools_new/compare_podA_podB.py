#!/usr/bin/env python3
"""P8 — 같은 입력·다른 장비 짝(podA RTX 4090 vs podB RTX PRO 6000 x2) 수치 병기. **관측만**.

podB 후처리의 `tools/compare_w19_w25.py` 는 W25 vs W19 전용이라 이 짝을 만들지 못한다
(그 파일 `:3-4` 가 "podA 가 아직 실행 중이라 이 시점에 비교 불가"라고 적고 있다).
그래서 이 파일만 새로 쓴다 — **동결 식(분류/전개/전환) 은 한 글자도 건드리지 않는다**.
여기서 하는 일은 이미 기록된 배열/JSON 값을 읽어 나란히 적고 차이를 빼는 것뿐이다.

판정 문구·인과 해석은 만들지 않는다(D490: n>=3 전 인과 주장 금지).
"""
import argparse, json, time
from pathlib import Path

import numpy as np

LABELS = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]
FOCUS_TAGS = ["close_stop", "lift_end", "reclose_end", "transport_end",
              "release_before", "release_after", "wait_end", "return_home_end"]


def counts(code):
    return {n: int(v) for n, v in zip(LABELS, np.bincount(np.asarray(code).astype(int), minlength=6))}


def phase_table(z, meta):
    """단계별 실제 경과 시간(wall-clock) · 물리 시간. sync 배열에서 직접 읽는다."""
    pcode = np.asarray(z["sync_phase_code"]).astype(int)
    st = np.asarray(z["sync_t_s"], float)
    sw = np.asarray(z["sync_wall_elapsed_s"], float)
    phases = list(meta["phase_order"])
    rows = []
    for pi, name in enumerate(phases):
        idx = np.flatnonzero(pcode == pi)
        if idx.size == 0:
            rows.append({"phase": name, "n_sync_rows": 0})
            continue
        i0, i1 = int(idx[0]), int(idx[-1])
        contiguous = bool(idx.size == (i1 - i0 + 1))
        prev_wall = float(sw[i0 - 1]) if i0 > 0 else 0.0
        prev_t = float(st[i0 - 1]) if i0 > 0 else 0.0
        rows.append({
            "phase": name, "phase_code": pi, "n_sync_rows": int(idx.size), "contiguous": contiguous,
            "sync_first": i0, "sync_last": i1,
            "sim_t_first_s": round(float(st[i0]), 9), "sim_t_last_s": round(float(st[i1]), 9),
            "sim_duration_from_prev_boundary_s": round(float(st[i1]) - prev_t, 9),
            "wall_elapsed_first_s": round(float(sw[i0]), 3), "wall_elapsed_last_s": round(float(sw[i1]), 3),
            "wall_duration_from_prev_boundary_s": round(float(sw[i1]) - prev_wall, 3),
        })
    return rows


def door_rows(res):
    stops = res["door"]["stops"]
    return {
        "n_stops": len(stops),
        "q_final_actual_deg": res["door"]["q_final_actual_deg"],
        "close_stops": [{"phase": s["phase"], "subphase": s["subphase"], "sync_index": s.get("sync_index"),
                         "q_actual_deg": s["q_actual_deg"], "servo_deg": s["servo_deg"],
                         "reason": s["reason"]} for s in stops if s["subphase"] == "close"],
        "reclose_stops": [{"phase": s["phase"], "subphase": s["subphase"], "sync_index": s.get("sync_index"),
                           "q_actual_deg": s["q_actual_deg"], "servo_deg": s["servo_deg"],
                           "reason": s["reason"]} for s in stops if s["phase"] == "reclose"],
        "all_stop_reasons": [s["reason"] for s in stops],
    }


def chatter_rows(res):
    proc = res.get("w25", {}).get("procedure", {})
    log = proc.get("chatter_log", []) or []
    return {"chatter_enabled": proc.get("chatter", False),
            "n_chatter_events": sum(1 for e in log if e.get("action") == "chatter"),
            "terminal_action": (log[-1].get("action") if log else None),
            "log": log}


def pod_row(tag, run_dir, derived_npz, receipt_json):
    run = Path(run_dir)
    res = json.load(open(run / "w13_cycle_seed460.json"))
    rec = json.load(open(receipt_json))
    z = np.load(run / "w13_cycle_seed460.npz", allow_pickle=False)
    d = np.load(derived_npz, allow_pickle=False)
    meta = json.loads(str(z["metadata_json"]))
    code = np.asarray(z["inventory_code"])
    c34 = np.asarray(d["inventory_code_rev34_support"])
    tags = [str(t) for t in z["decision_tags"]]
    dpf = np.asarray(z["decision_particle_frame_index"]).astype(int)
    pft = np.asarray(z["particle_frame_t_s"], float)
    sw = np.asarray(z["sync_wall_elapsed_s"], float)
    dec = {}
    for t in FOCUS_TAGS:
        if t not in tags:
            continue
        i = tags.index(t)
        r = int(dpf[i])
        dec[t] = {"particle_frame_row": r, "sync_index": int(z["decision_sync_index"][i]),
                  "sim_t_s": round(float(pft[r]), 9),
                  "wall_elapsed_s": round(float(sw[int(z["decision_sync_index"][i])]), 3),
                  "recorded_counts": counts(code[r]), "rev34_recomputed_counts": counts(c34[r]),
                  "recorded_equals_rev34": counts(code[r]) == counts(c34[r])}
    dl = res["delivery"]
    sw_win = dl["settlement_window"]
    sp = LABELS.index("spill")
    return {
        "tag": tag, "run_dir": str(run),
        "revision": res.get("w25", {}).get("revision"),
        "pod_tag": json.load(open(run / "EXECUTION_RECEIPT.json")).get("pod_tag"),
        "hostname": json.load(open(run / "EXECUTION_RECEIPT.json")).get("hostname"),
        "scale": {"n_particles": int(res["particle"]["n"]),
                  "n_sync": int(res["trajectory"]["n_sync"]),
                  "n_particle_frames": int(res["trajectory"]["n_particle_frames"]),
                  "sim_time_s": res["trajectory"]["sim_time_s"],
                  "sim_wall_seconds": res["wall_seconds"],
                  "runner_wall_s": rec.get("wall_s"), "runner_cap_s": rec.get("cap_s"),
                  "runner_rc": rec.get("rc"), "runner_timed_out": rec.get("timed_out"),
                  "sync_wall_elapsed_last_s": round(float(sw[-1]), 3)},
        "delivery": {"definite_delivered_n": dl["definite_delivered_n"],
                     "definite_delivered_g": dl["definite_delivered_g"],
                     "possible_delivered_n": dl["possible_delivered_n"],
                     "possible_delivered_g": dl["possible_delivered_g"],
                     "exact_single_value_allowed": dl["exact_single_value_allowed"],
                     "inventory_final": dl["inventory_final"],
                     "inventory_final_raw_npz_last_frame": counts(code[-1]),
                     "inventory_final_rev34_recomputed": counts(c34[-1])},
        "settlement_window": {k: sw_win.get(k) for k in
                              ("window_s", "n_frames", "frame_times_s", "max_frame_gap_s", "cadence_ok",
                               "n_stable_bin_all_frames", "n_settled", "max_speed_of_stable_m_s",
                               "max_move_of_stable_mm")},
        "door": door_rows(res),
        "chatter": chatter_rows(res),
        "decision_inventory": dec,
        "spill": {"n_ever_spill_recorded": int((code == sp).any(0).sum()),
                  "n_final_spill_recorded": int((code[-1] == sp).sum()),
                  "n_final_spill_rev34": int((c34[-1] == sp).sum())},
        "bridge": {"verdict": res["bridge_clearance"][0]["verdict"],
                   "worst_slack_m": res["bridge_clearance"][0].get("worst_slack_m")},
        "crater": {"removed_volume_cm3": res["crater"]["removed_volume_cm3"],
                   "dh_max_mm": res["crater"]["dh_max_mm"]},
        "phase_wall_clock": phase_table(z, meta),
    }


def num_pairs(a, b, prefix=""):
    out = {}
    for k, v in a.items():
        w = b.get(k)
        key = prefix + k
        if isinstance(v, dict) and isinstance(w, dict):
            out.update(num_pairs(v, w, key + "."))
        elif isinstance(v, (int, float)) and isinstance(w, (int, float)) \
                and not isinstance(v, bool) and not isinstance(w, bool):
            out[key] = {"podA": v, "podB": w, "podA_minus_podB": round(v - w, 9),
                        "ratio_podA_over_podB": round(v / w, 9) if w else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("a-run", "b-run", "a-derived", "b-derived", "a-receipt", "b-receipt", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    A = pod_row("W25-A podA_4090 (rev34, paper box, n=67,737)", a.a_run, a.a_derived, a.a_receipt)
    B = pod_row("W25-A podB_pro6000x2 (rev34, paper box, n=67,737)", a.b_run, a.b_derived, a.b_receipt)

    phase_pairs = []
    bmap = {r["phase"]: r for r in B["phase_wall_clock"]}
    for r in A["phase_wall_clock"]:
        s = bmap.get(r["phase"], {})
        phase_pairs.append({
            "phase": r["phase"],
            "n_sync_rows": {"podA": r.get("n_sync_rows"), "podB": s.get("n_sync_rows")},
            "sim_duration_from_prev_boundary_s": {"podA": r.get("sim_duration_from_prev_boundary_s"),
                                                  "podB": s.get("sim_duration_from_prev_boundary_s")},
            "wall_duration_from_prev_boundary_s": {"podA": r.get("wall_duration_from_prev_boundary_s"),
                                                   "podB": s.get("wall_duration_from_prev_boundary_s")},
            "wall_ratio_podA_over_podB": (round(r["wall_duration_from_prev_boundary_s"]
                                                / s["wall_duration_from_prev_boundary_s"], 6)
                                          if s.get("wall_duration_from_prev_boundary_s") else None),
        })

    same_input = {}
    ra = json.load(open(Path(a.a_run) / "EXECUTION_RECEIPT.json"))
    rb = json.load(open(Path(a.b_run) / "EXECUTION_RECEIPT.json"))

    def argv_val(argv, flag):
        return argv[argv.index(flag) + 1] if flag in argv else None
    for flag, key in (("--params", "params_path"), ("--pile", "pile_path"), ("--seed", "seed"),
                      ("--max-wall-s", "max_wall_s"), ("--numeric-evidence", "numeric_evidence_path")):
        same_input[key] = {"podA": argv_val(ra["argv"], flag), "podB": argv_val(rb["argv"], flag),
                           "identical": argv_val(ra["argv"], flag) == argv_val(rb["argv"], flag)}
    same_input["sim_script"] = {"podA": ra["argv"][2], "podB": rb["argv"][2],
                                "identical": ra["argv"][2] == rb["argv"][2]}
    same_input["hostname"] = {"podA": ra.get("hostname"), "podB": rb.get("hostname")}

    out = {"artifact": "W25_PODA_VS_PODB_NUMERIC_SIDE_BY_SIDE_V1",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "observation_only": True,
           "question": "같은 입력·다른 장비(podA RTX 4090 1장 vs podB RTX PRO 6000 2장)",
           "same_input_evidence": same_input,
           "rows": [A, B],
           "numeric_pairs": num_pairs({k: A[k] for k in ("scale", "delivery", "settlement_window",
                                                         "spill", "bridge", "crater")},
                                      {k: B[k] for k in ("scale", "delivery", "settlement_window",
                                                         "spill", "bridge", "crater")}),
           "phase_wall_clock_pairs": phase_pairs,
           "non_claims": [
               "이 파일은 관측이다 — 어떤 차이도 원인으로 읽지 않는다(n>=3 전 인과 주장 금지, D490).",
               "두 실행은 같은 동결 스크립트·params·pile·seed 를 썼지만 GPU 장비와 GPU 개수가 다르다. "
               "DEM 접촉 해는 비결정적이라 수치 차이를 장비 탓으로 귀속할 수 없다.",
               "두 실행 모두 정착 cadence 계약 미충족이므로 배출 수치는 구간으로만 읽는다.",
               "'전체 사이클 성공' 선언이 아니다. 판정 승격은 재생·독립 감사 뒤 사용자 몫이다."]}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({
        "scale": {k: out["numeric_pairs"]["scale." + k] for k in
                  ("n_sync", "n_particle_frames", "sim_time_s", "runner_wall_s")},
        "delivery": {k: out["numeric_pairs"]["delivery." + k] for k in
                     ("definite_delivered_n", "possible_delivered_n")},
        "lift_end_tool_residual": {"podA": A["decision_inventory"]["lift_end"]["recorded_counts"],
                                   "podB": B["decision_inventory"]["lift_end"]["recorded_counts"]},
        "reclose": {"podA": A["door"]["reclose_stops"], "podB": B["door"]["reclose_stops"]},
        "settlement": {"podA": A["settlement_window"], "podB": B["settlement_window"]},
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
