#!/usr/bin/env python3
"""P8 — W19 A(rev32, 알 20,000) 와 W25-A podB(rev34, 알 67,737) 수치 병기. **관측만**.
두 실행은 알 개수·더미 형상·상자 선언·절차(채터링)·장비가 모두 다르다 → 인과 비교 금지(n>=3 전).
'같은 입력·다른 장비' 짝은 podA 가 아직 실행 중이라 이 시점에 비교 불가."""
import argparse, json, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    for k in ("w25-json", "w19-json", "w25-receipt", "w19-receipt", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--poda-dir", required=True)
    a = ap.parse_args()
    A = json.load(open(a.w25_json)); B = json.load(open(a.w19_json))
    ra = json.load(open(a.w25_receipt)); rb = json.load(open(a.w19_receipt))

    def row(j, r, tag):
        d = j["delivery"]; t = j["trajectory"]
        rec = [s for s in j["door"]["stops"] if s["phase"] == "reclose"]
        cls = [s for s in j["door"]["stops"] if s["subphase"] == "close"]
        w25 = j.get("w25", {})
        proc = w25.get("procedure", {})
        return {
            "tag": tag, "revision": w25.get("revision", "rev32 (W19 A)"),
            "n_particles": j["particle"]["n"], "n_sync": t["n_sync"],
            "n_particle_frames": t["n_particle_frames"], "sim_time_s": t["sim_time_s"],
            "sim_wall_seconds": j["wall_seconds"],
            "runner_wall_s": r.get("wall_s"), "runner_cap_s": r.get("cap_s"),
            "runner_rc": r.get("rc"), "runner_timed_out": r.get("timed_out"),
            "definite_delivered_n": d["definite_delivered_n"], "definite_delivered_g": d["definite_delivered_g"],
            "possible_delivered_n": d["possible_delivered_n"], "possible_delivered_g": d["possible_delivered_g"],
            "exact_single_value_allowed": d["exact_single_value_allowed"],
            "inventory_final": d["inventory_final"],
            "settlement_n_frames": d["settlement_window"]["n_frames"],
            "settlement_max_frame_gap_s": d["settlement_window"]["max_frame_gap_s"],
            "settlement_cadence_ok": d["settlement_window"]["cadence_ok"],
            "settlement_n_settled": d["settlement_window"]["n_settled"],
            "close_stop_joint_deg": cls[0]["q_actual_deg"] if cls else None,
            "close_stop_reason": cls[0]["reason"] if cls else None,
            "reclose_joint_deg": rec[0]["q_actual_deg"] if rec else None,
            "reclose_servo_deg": rec[0]["servo_deg"] if rec else None,
            "reclose_reason": rec[0]["reason"] if rec else None,
            "door_final_actual_deg": j["door"]["q_final_actual_deg"],
            "n_door_stops": len(j["door"]["stops"]),
            "chatter_enabled": proc.get("chatter", False),
            "n_chatter_events": sum(1 for e in proc.get("chatter_log", []) if e.get("action") == "chatter"),
            "chatter_terminal_action": (proc.get("chatter_log") or [{}])[-1].get("action"),
            "box_bounds_m": j.get("w25", {}).get("tray", {}).get("box_bounds_m"),
            "bridge_verdict": j["bridge_clearance"][0]["verdict"],
            "v_particle_max_m_s": j["pops"]["v_particle_max_m_s"],
            "crater_removed_volume_cm3": j["crater"]["removed_volume_cm3"],
            "crater_dh_max_mm": j["crater"]["dh_max_mm"],
            "heightmap_max_m": j["heightmap"]["max_m"],
        }

    w25 = row(A, ra, "W25-A podB_pro6000x2 (rev34, paper box, n=67,737)")
    w19 = row(B, rb, "W19-A (rev32, n=20,000)")
    diff = {}
    for k, v in w25.items():
        if isinstance(v, (int, float)) and isinstance(w19.get(k), (int, float)) and not isinstance(v, bool):
            diff[k] = {"w25": v, "w19": w19[k],
                       "w25_minus_w19": round(v - w19[k], 6),
                       "ratio_w25_over_w19": round(v / w19[k], 6) if w19[k] else None}
    podA = Path(a.poda_dir)
    same_input_pair = {
        "question": "같은 입력·다른 장비(podA 4090 vs podB RTX PRO 6000 x2)",
        "podA_run_01_present": (podA / "run_01").exists(),
        "podA_dirs": sorted(p.name for p in podA.iterdir()) if podA.exists() else [],
        "verdict": "COMPARISON_NOT_POSSIBLE_YET",
        "reason": "podA 는 이 시점에 run_01 원자료가 없다(스모크 3종만 회수). 장비 짝 비교는 podA 완주 후에만 가능하다."}
    out = {"artifact": "W25_VS_W19_NUMERIC_SIDE_BY_SIDE_V1",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "observation_only": True,
           "rows": [w25, w19], "numeric_diff": diff, "same_input_different_hardware": same_input_pair,
           "what_differs_between_the_two_runs": [
               "알 개수 20,000 -> 67,737 (더미: 슬랩 -> 310x220 평평한 40 mm 층)",
               "상자 선언: npz box_bounds -> declared 310x220x230 (종이 상자)",
               "절차: rev34 가 표면에서 문 열기 + 채터링(임계 서보 3.6 deg, 최대 3회) + 항상 재닫기를 추가",
               "좌표계: rev34 는 상자 좌표계 규약 A(R_robot_box, t_robot) 사용",
               "장비: RunPod RTX 4090(W19) -> RTX PRO 6000 x2(W25 podB)"],
           "non_claims": [
               "이 표는 관측이다 — 어떤 차이도 원인으로 읽지 않는다(n>=3 전 인과 주장 금지, D490).",
               "배출량 증가는 알 개수·더미 형상·절차가 동시에 바뀐 결과이며 단일 원인으로 귀속할 수 없다.",
               "두 실행 모두 정착 cadence 계약 미충족 상태이므로 배출 수치는 구간으로만 읽는다.",
               "'전체 사이클 성공' 선언이 아니다."]}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"w25": {k: w25[k] for k in ("n_particles", "definite_delivered_n", "possible_delivered_n",
                                                  "inventory_final", "reclose_joint_deg", "n_chatter_events",
                                                  "settlement_cadence_ok", "runner_wall_s")},
                      "w19": {k: w19[k] for k in ("n_particles", "definite_delivered_n", "possible_delivered_n",
                                                  "inventory_final", "reclose_joint_deg", "n_chatter_events",
                                                  "settlement_cadence_ok", "runner_wall_s")},
                      "same_input_pair": same_input_pair["verdict"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
