#!/usr/bin/env python3
"""P7 — W25 전용 관측(원시 사실만). 원인 주장 0, 판정 승격 0. NumPy 만.

읽는 것: 실물 정렬 절차 로그(채터링), 결정 시점 문 관절각, 흘림 입자의 이탈 단계,
bridge_clearance 결과, w25.frame 값. 모든 값은 원자료 NPZ/생산 JSON 에서 직접 읽거나
그 배열로 다시 센 것이다.
"""
import argparse, json, time
from pathlib import Path

import numpy as np

LABELS = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]
SPILL = LABELS.index("spill")


def jdef(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(type(o)))


def main():
    ap = argparse.ArgumentParser()
    for k in ("raw", "meta", "derived", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = json.load(open(a.meta))
    z = np.load(a.raw, allow_pickle=False)
    d = np.load(a.derived, allow_pickle=False)
    meta = json.loads(str(z["metadata_json"]))
    P = res["params"]
    phases = meta["phase_order"]
    pcode = np.asarray(z["sync_phase_code"]).astype(int)
    sub = np.asarray(z["sync_subphase"]).astype(str)
    st = np.asarray(z["sync_t_s"], float)
    pfs = np.asarray(z["particle_frame_sync_index"]).astype(int)
    pft = np.asarray(z["particle_frame_t_s"], float)
    door_deg = np.asarray(z["door_actual_deg"], float)
    door_tgt = np.asarray(z["door_target_deg"], float)
    off = float(P["servo_zero_offset_deg"])
    rec = np.asarray(z["inventory_code"])
    tags = [str(t) for t in z["decision_tags"]]
    dsi = np.asarray(z["decision_sync_index"]).astype(int)
    dpf = np.asarray(z["decision_particle_frame_index"]).astype(int)

    # ── 1. 채터링(실물 절차) ────────────────────────────────────────────────
    proc = res["w25"]["procedure"]
    log = proc.get("chatter_log", [])
    n_chatter = sum(1 for e in log if e.get("action") == "chatter")
    chatter_rows = []
    for e in log:
        row = {"k": e["k"], "action": e["action"], "sim_t_s": e["sim_t"], "sync_index": e["sync_index"],
               "read_joint_deg": e["read_joint_deg"], "read_servo_deg": e["read_servo_deg"],
               "cmd_joint_deg": e.get("cmd_joint_deg"), "cmd_servo_deg": e.get("cmd_servo_deg"),
               "threshold_servo_deg": e["threshold_servo_deg"],
               "raw_door_actual_deg_at_sync": float(door_deg[e["sync_index"]]),
               "raw_servo_deg_at_sync": float(door_deg[e["sync_index"]]) + off,
               "raw_subphase_at_sync": str(sub[e["sync_index"]]),
               "raw_phase_at_sync": phases[int(pcode[e["sync_index"]])]}
        for key in ("open_stop", "close_stop"):
            if key in e:
                s = e[key]
                row[key] = {"subphase": s["subphase"], "reason": s["reason"], "sync_index": s["sync_index"],
                            "q_cmd_deg": s["q_cmd_deg"], "q_actual_deg": s["q_actual_deg"],
                            "servo_deg": s["servo_deg"], "sim_t_s": s["sim_t"],
                            "M_hinge_rel_Nm": s["M_hinge_rel_Nm"],
                            "max_single_contact_N": s["max_single_contact_N"],
                            "raw_door_actual_deg_at_sync": float(door_deg[s["sync_index"]])}
        chatter_rows.append(row)

    # ── 2. 결정 시점 문 관절각 ──────────────────────────────────────────────
    want = ["close_stop", "lift_end", "reclose_end", "release_before"]
    prod_dec = {x["tag"]: x for x in res["decisions"]}
    dec_rows = {}
    for t in tags:
        i = tags.index(t)
        si = int(dsi[i])
        dec_rows[t] = {"sync_index": si, "particle_frame_row": int(dpf[i]),
                       "sim_t_s": float(st[si]), "phase": phases[int(pcode[si])],
                       "subphase": str(sub[si]),
                       "json_q_cmd_deg": prod_dec[t].get("q_cmd_deg"),
                       "json_q_actual_deg": prod_dec[t].get("q_actual_deg"),
                       "raw_door_actual_deg": float(door_deg[si]),
                       "raw_door_target_deg": float(door_tgt[si]),
                       "raw_servo_deg": float(door_deg[si]) + off,
                       "json_matches_raw_within_1e-4_deg":
                           bool(abs(float(prod_dec[t].get("q_actual_deg", np.nan)) - float(door_deg[si])) <= 1e-4)
                           if prod_dec[t].get("q_actual_deg") is not None else None,
                       "inventory_counts_recorded": {n: int(v) for n, v in
                                                     zip(LABELS, np.bincount(rec[int(dpf[i])].astype(int),
                                                                             minlength=6))}}
    focus = {t: dec_rows[t] for t in want if t in dec_rows}

    # ── 3. 흘림(spill) 입자의 이탈 단계 ─────────────────────────────────────
    ever = np.flatnonzero((rec == SPILL).any(0))
    first_row = {int(i): int(np.argmax(rec[:, i] == SPILL)) for i in ever}
    by_phase, by_frame = {}, {}
    for pid, fr in first_row.items():
        si = int(pfs[fr])
        ph = phases[int(pcode[si])]
        by_phase.setdefault(ph, []).append(int(pid))
        by_frame.setdefault(fr, []).append(int(pid))
    spill_rows = [{"particle_id": int(pid), "first_spill_particle_frame_row": fr,
                   "sync_index": int(pfs[fr]), "t_s": float(pft[fr]),
                   "phase": phases[int(pcode[int(pfs[fr])])], "subphase": str(sub[int(pfs[fr])]),
                   "label_before": LABELS[int(rec[fr - 1, pid])] if fr > 0 else None,
                   "final_label": LABELS[int(rec[-1, pid])],
                   "final_z_m": float(np.asarray(z["final_positions_m"], float)[pid, 2])}
                  for pid, fr in sorted(first_row.items(), key=lambda kv: (kv[1], kv[0]))]
    spill = {"n_ever_spill": int(len(ever)), "n_final_spill": int((rec[-1] == SPILL).sum()),
             "count_by_departure_phase": {k: len(v) for k, v in sorted(by_phase.items())},
             "count_by_first_spill_frame": {str(k): len(v) for k, v in sorted(by_frame.items())},
             "spill_rest_z_m": float(meta["spill_rest_z_m"]),
             "rows": spill_rows,
             "n_spill_ids_equal_between_recorded_and_rev34":
                 int(len(np.intersect1d(ever, np.asarray(d["ever_spill_ids_rev34"])))),
             "n_ever_spill_rev34": int(len(np.asarray(d["ever_spill_ids_rev34"])))}

    # ── 4. bridge_clearance 결과 ────────────────────────────────────────────
    bc = res["bridge_clearance"][0]
    bridge = {"verdict": bc["verdict"], "pass": bc["pass"], "decision_point": bc["decision_point"],
              "sim_t_s": bc["sim_t_s"], "sync_index": bc["sync_index"], "wall_s": bc["wall_s"],
              "z_reached_m": bc["z_reached_m"],
              "door_q_deg_actual": bc["door_q_deg_actual"], "door_q_deg_commanded": bc["door_q_deg_commanded"],
              "raw_door_actual_deg_at_sync": float(door_deg[int(bc["sync_index"])]),
              "n_align_sync_targets": bc["n_align_sync_targets"],
              "n_joint_sync_targets": bc["n_joint_sync_targets"],
              "physics_steps_before_certify": bc["physics_steps_before_certify"],
              "physics_steps_after_certify": bc["physics_steps_after_certify"],
              "certify_consumed_zero_physics": bc["certify_consumed_zero_physics"],
              "precheck_summary": bc["precheck_summary"],
              "global_worst": bc["planned_certificate"]["global_worst"],
              "limit_failures": bc["planned_certificate"]["limit_failures"],
              "n_separation_failures_total": bc["planned_certificate"]["n_separation_failures_total"],
              "door_vs_fixed_residual_at_decision": bc["door_vs_fixed_residual_at_decision"],
              "npz_bridge_precheck_rows_shape": list(np.asarray(z["bridge_precheck_rows"]).shape),
              "npz_bridge_interval_rows_shape": list(np.asarray(z["bridge_interval_rows"]).shape),
              "npz_bridge_precheck_columns": [str(c) for c in z["bridge_precheck_columns"]],
              "non_claims": bc["non_claims"]}

    # ── 5. w25.frame ────────────────────────────────────────────────────────
    frame = {"metadata_w25_frame": meta.get("w25_frame"),
             "json_w25_frame": res["w25"]["frame"],
             "json_w25_scoop_site": res["w25"]["scoop_site"],
             "json_w25_tray": res["w25"]["tray"],
             "metadata_equals_json_R_robot_box":
                 bool(np.array_equal(np.asarray(meta["w25_frame"]["R_robot_box"], float),
                                     np.asarray(res["w25"]["frame"]["R_robot_box"], float))),
             "metadata_equals_json_t_robot":
                 bool(np.allclose(np.asarray(meta["w25_frame"]["t_robot_m"], float),
                                  np.asarray(res["w25"]["frame"].get("box_center_robot_xy_m", [0, 0]) +
                                             [res["w25"]["frame"]["box_floor_robot_z_m"]], float), atol=0.0)),
             "t_robot_m_metadata": meta["w25_frame"]["t_robot_m"],
             "box_floor_robot_z_m": res["w25"]["frame"]["box_floor_robot_z_m"],
             "box_floor_above_floor_cm": res["w25"]["frame"]["box_floor_above_floor_cm"],
             "declared_not_measured": res["fixtures"]["declared_not_measured"]}

    # ── 6. 문 정지 기록 전체 ────────────────────────────────────────────────
    stops = [{**s, "raw_door_actual_deg_at_sync": float(door_deg[int(s["sync_index"])]),
              "raw_phase": phases[int(pcode[int(s["sync_index"])])]} for s in res["door"]["stops"]]

    out = {"artifact": "W25_RAW_OBSERVATIONS_V1",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "raw": a.raw, "cpu_only": True, "new_physics_runs": 0, "gpu_used": False,
           "chatter": {"enabled": proc["chatter"], "threshold_servo_deg": P["w25_chatter_threshold_servo_deg"],
                       "open_servo_deg": P["w25_chatter_open_servo_deg"],
                       "max_retries": P["w25_chatter_max_retries"],
                       "n_chatter_events": n_chatter, "n_log_rows": len(log),
                       "terminal_action": log[-1]["action"] if log else None,
                       "units": proc["units"], "log": chatter_rows,
                       "procedure_provenance": P["w25_procedure_provenance"]},
           "decision_door_angles": {"all": dec_rows, "focus": focus,
                                    "servo_zero_offset_deg": off},
           "door_stops": stops,
           "door_final": res["door"],
           "reclose_read": proc["reclose_read"],
           "spill": spill, "bridge_clearance": bridge, "w25_frame": frame,
           "non_claims": ["이 표는 관측만이다 — 차이를 원인으로 읽지 않는다(n>=3 전 인과 주장 금지).",
                          "실물 절차 재현 로그는 시뮬 값이며 실물 측정이 아니다.",
                          "배치 치수는 사용자 선언·실측 혼합이다(declared_not_measured=true)."],
           "wall_s": round(time.time() - t0, 3)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=jdef))
    print(json.dumps({"n_chatter_events": n_chatter, "terminal_action": out["chatter"]["terminal_action"],
                      "focus_door_deg": {k: {"raw_door_actual_deg": v["raw_door_actual_deg"],
                                             "raw_servo_deg": v["raw_servo_deg"]} for k, v in focus.items()},
                      "spill_by_phase": spill["count_by_departure_phase"],
                      "bridge_verdict": bridge["verdict"], "bridge_worst_slack_m":
                          bridge["global_worst"]["slack_m"],
                      "wall_s": out["wall_s"]}, ensure_ascii=False, indent=1, default=jdef))


if __name__ == "__main__":
    main()
