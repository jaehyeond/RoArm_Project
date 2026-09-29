#!/usr/bin/env python3
"""P4 — 정착 창(settlement window)·배출량 독립 재계산. NumPy 만. 생산 모듈 import 0.

규약 문구(동결 params_w25_paperbox.json + rev34 src/sim_w13_full_cycle.py:995-1017)를 읽어
같은 정의를 여기서 다시 구현한다:
  · 창 = t >= t_end - settlement_window_s - 1e-9 인 저장 입자 프레임
  · cadence 계약 = (창 프레임 수 >= 6) AND (창 안 최대 프레임 간격 <= 0.05 + 1e-9 s)
  · stable_bin = 창의 **모든** 프레임에서 receiving_bin
  · settled  = stable_bin AND max|v| <= settle_speed_max_m_s AND max|x-x0| <= settle_move_max_m
  · definite = settled 수 ;  possible = definite(라벨 receiving_bin 수) + 용기 기하 밴드 안의 ambiguous/in_flight
사후 완화 0 — 규약 미충족은 FAIL 로 그대로 적는다.
"""
import argparse, json, time
from pathlib import Path

import numpy as np

LABELS = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]
BIN = LABELS.index("receiving_bin")
AMB = LABELS.index("ambiguous")
FLY = LABELS.index("in_flight")
CADENCE_MIN_FRAMES = 6            # 규약 계약 (rev34 src/sim_w13_full_cycle.py:1005 `len(widx) >= 6`)
CADENCE_MAX_GAP_S = 0.05          # 규약 계약 (같은 줄 `<= 0.05 + 1e-9`)


def window_stats(t, win):
    t_end = float(t[-1])
    idx = [i for i, v in enumerate(t) if float(v) >= t_end - win - 1e-9]
    gaps = [float(t[idx[k + 1]] - t[idx[k]]) for k in range(len(idx) - 1)]
    return idx, gaps


def recompute(t, pos, vel, code, P, bin_fix, margin, pp_final):
    win = float(P["settlement_window_s"])
    idx, gaps = window_stats(t, win)
    max_gap = max(gaps) if gaps else 0.0
    cadence_ok = bool(len(idx) >= CADENCE_MIN_FRAMES and (max(gaps) if gaps else 1e9) <= CADENCE_MAX_GAP_S + 1e-9)
    out = {"window_s": win, "n_frames": len(idx),
           "frame_rows": [int(i) for i in idx],
           "frame_times_s": [round(float(t[i]), 9) for i in idx],
           "frame_gaps_s": [round(g, 9) for g in gaps],
           "max_frame_gap_s": round(max_gap, 9),
           "settlement_frame_dt_s_param": float(P["settlement_frame_dt_s"]),
           "particle_frame_dt_s_param": float(P["particle_frame_dt_s"]),
           "cadence_contract": {"min_frames": CADENCE_MIN_FRAMES, "max_gap_s": CADENCE_MAX_GAP_S,
                                "n_frames_ok": bool(len(idx) >= CADENCE_MIN_FRAMES),
                                "max_gap_ok": bool(max_gap <= CADENCE_MAX_GAP_S + 1e-9)},
           "cadence_ok": cadence_ok,
           "criterion": {"speed_max_m_s": float(P["settle_speed_max_m_s"]),
                         "center_move_max_m": float(P["settle_move_max_m"])}}
    if len(idx) >= 2:
        Pw = pos[idx].astype(float); Vw = vel[idx].astype(float); Iw = code[idx]
        move = np.linalg.norm(Pw - Pw[0], axis=2).max(0)
        spd = np.linalg.norm(Vw, axis=2).max(0)
        stable = (Iw == BIN).all(0)
        settled = stable & (spd <= float(P["settle_speed_max_m_s"])) & (move <= float(P["settle_move_max_m"]))
        out.update(n_stable_bin_all_frames=int(stable.sum()), n_settled=int(settled.sum()),
                   max_speed_of_stable_m_s=round(float(spd[stable].max()), 8) if stable.any() else None,
                   max_move_of_stable_mm=round(float(move[stable].max()) * 1000, 6) if stable.any() else None,
                   n_stable_but_moving=int((stable & ~settled).sum()))
        definite = int(settled.sum())
    else:
        out["note"] = "관측창 프레임이 2 미만 — exact settled 를 주장하지 않는다."
        definite = 0
    # possible = 라벨 receiving_bin + 용기 기하 밴드 안의 ambiguous/in_flight (최종 프레임 owner 중심)
    n_th = int(P["bin_n_theta"])
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False) + np.pi / n_th
    nrm = np.stack([np.cos(th), np.sin(th)], 1)
    apo = float(bin_fix["inner_r_m"]) * np.cos(np.pi / n_th)
    bc = np.asarray(bin_fix["center_xy_m"], float)
    zf, zr = float(bin_fix["floor_inner_z_m"]), float(bin_fix["rim_z_m"])
    code_f = code[-1]
    over_xy = ((pp_final[:, :2] - bc) @ nrm.T).max(1) < apo + margin
    z_band = (pp_final[:, 2] > zf - margin) & (pp_final[:, 2] < zr + margin)
    amb_fly = (code_f == AMB) | (code_f == FLY)
    label_bin = int((code_f == BIN).sum())
    add_band = int((amb_fly & over_xy & z_band).sum())
    add_rim = int((amb_fly & over_xy & (pp_final[:, 2] >= zr - margin)).sum())
    out["possible_terms"] = {"label_receiving_bin_n": label_bin, "amb_or_flight_in_band_n": add_band,
                             "amb_or_flight_at_or_above_rim_n": add_rim,
                             "bin_apothem_m": float(apo), "bin_floor_inner_z_m": zf, "bin_rim_z_m": zr,
                             "margin_m": float(margin)}
    out["definite_delivered_n"] = definite
    out["possible_delivered_n"] = label_bin + add_band + add_rim
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("raw", "meta", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--derived", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = json.load(open(a.meta))
    z = np.load(a.raw, allow_pickle=False)
    d = np.load(a.derived, allow_pickle=False)
    P = res["params"]; bin_fix = res["fixtures"]["bin"]
    margin = float(P["classify_margin_mm"]) / 1000.0
    t = np.asarray(z["particle_frame_t_s"], float)
    pos = np.asarray(z["particle_pos_m"]); vel = np.asarray(z["particle_vel_m_s"])
    pp_final = np.asarray(z["final_positions_m"], float)
    rec = np.asarray(z["inventory_code"])
    c34 = np.asarray(d["inventory_code_rev34_support"])
    got_rec = recompute(t, pos, vel, rec, P, bin_fix, margin, pp_final)
    got_34 = recompute(t, pos, vel, c34, P, bin_fix, margin, pp_final)
    prod = res["delivery"]["settlement_window"]
    cmp_keys = ["window_s", "n_frames", "frame_times_s", "max_frame_gap_s", "cadence_ok",
                "n_stable_bin_all_frames", "n_settled", "max_speed_of_stable_m_s", "max_move_of_stable_mm"]
    diff = {k: {"production": prod.get(k), "recomputed_recorded_labels": got_rec.get(k)}
            for k in cmp_keys if prod.get(k) != got_rec.get(k)}
    verdicts = [
        {"name": "settlement_window_recompute_matches_production",
         "verdict": "PASS" if not diff else "FAIL",
         "measured": {"n_differing_keys": len(diff), "diff": diff}},
        {"name": "settlement_cadence_contract_min6frames_and_max_gap_0p05s",
         "verdict": "PASS" if got_rec["cadence_ok"] else "FAIL",
         "measured": {"n_frames": got_rec["n_frames"], "min_required": CADENCE_MIN_FRAMES,
                      "max_frame_gap_s": got_rec["max_frame_gap_s"], "max_allowed_s": CADENCE_MAX_GAP_S,
                      "settlement_frame_dt_s_param": got_rec["settlement_frame_dt_s_param"],
                      "particle_frame_dt_s_param": got_rec["particle_frame_dt_s_param"],
                      "production_cadence_ok": prod.get("cadence_ok")},
         "note": "사후 완화 0. 저장 간격이 0.1 s 라 0.05 s 계약을 못 채운다는 사실을 그대로 적는다."},
        {"name": "definite_delivered_recompute_equals_production",
         "verdict": "PASS" if got_rec["definite_delivered_n"] == res["delivery"]["definite_delivered_n"] else "FAIL",
         "measured": {"recomputed": got_rec["definite_delivered_n"],
                      "production": res["delivery"]["definite_delivered_n"]}},
        {"name": "possible_delivered_recompute_equals_production",
         "verdict": "PASS" if got_rec["possible_delivered_n"] == res["delivery"]["possible_delivered_n"] else "FAIL",
         "measured": {"recomputed": got_rec["possible_delivered_n"],
                      "production": res["delivery"]["possible_delivered_n"],
                      "terms": got_rec["possible_terms"]}},
        {"name": "exact_single_value_allowed_flag_matches_definite_equals_possible",
         "verdict": "PASS" if bool(res["delivery"]["exact_single_value_allowed"]) ==
                    (res["delivery"]["definite_delivered_n"] == res["delivery"]["possible_delivered_n"]) else "FAIL",
         "measured": {"flag": res["delivery"]["exact_single_value_allowed"],
                      "definite": res["delivery"]["definite_delivered_n"],
                      "possible": res["delivery"]["possible_delivered_n"]}},
        {"name": "settlement_under_rev34_recomputed_labels_equals_under_recorded_labels",
         "verdict": "PASS" if (got_34["n_settled"] == got_rec["n_settled"] and
                               got_34["n_stable_bin_all_frames"] == got_rec["n_stable_bin_all_frames"]) else "FAIL",
         "measured": {"rev34": {k: got_34.get(k) for k in ("n_stable_bin_all_frames", "n_settled")},
                      "recorded": {k: got_rec.get(k) for k in ("n_stable_bin_all_frames", "n_settled")}}},
    ]
    m_p = float(z["particle_mass_kg"]) * 1000.0
    out = {"artifact": "W25_SETTLEMENT_WINDOW_RECOMPUTE_V1",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "raw": a.raw, "derived": a.derived, "cpu_only": True, "new_physics_runs": 0,
           "recomputed_recorded_labels": got_rec, "recomputed_rev34_labels": got_34,
           "production_settlement_window": prod,
           "production_delivery": {k: res["delivery"][k] for k in
                                   ("definite_delivered_n", "definite_delivered_g", "possible_delivered_n",
                                    "possible_delivered_g", "exact_single_value_allowed", "particle_mass_g")},
           "mass_g": {"definite_g": round(got_rec["definite_delivered_n"] * m_p, 4),
                      "possible_g": round(got_rec["possible_delivered_n"] * m_p, 4),
                      "particle_mass_g": m_p},
           "items": verdicts,
           "n_fail": sum(v["verdict"] == "FAIL" for v in verdicts), "n_items": len(verdicts),
           "non_claims": ["정착 창 결과는 기록 계약의 결과이며 물리적 rest 의 증명이 아니다.",
                          "cadence FAIL 을 사후 허용값으로 완화하지 않았다.",
                          "definite/possible 은 구간이다 — 단일값 인용 금지(exact_single_value_allowed=false)."],
           "wall_s": round(time.time() - t0, 3)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"items": [{k: v[k] for k in ("name", "verdict")} for v in verdicts],
                      "n_fail": out["n_fail"],
                      "cadence": got_rec["cadence_contract"],
                      "n_frames": got_rec["n_frames"], "max_gap_s": got_rec["max_frame_gap_s"],
                      "definite": got_rec["definite_delivered_n"], "possible": got_rec["possible_delivered_n"],
                      "wall_s": out["wall_s"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
