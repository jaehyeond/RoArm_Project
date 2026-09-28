"""W13 벽 회귀 스모크 판정기 — **실행 전에 공개**한다. 읽기 전용, 사후에 임계값을 바꾸지 않는다.

usage: python compare_wall_regression.py <attempt_dir>
gates (사전 고정):
  S0 프로세스   EXECUTION_RECEIPT.json 의 **실제 step1 rc == 0** · timeout 아님 · auto_retry false
  S1 완주      diverged=false · stopped_early_after_phase="settle"
  S2 무접촉    25 sync 전부 scalar_n_fixed = scalar_n_door = 0
  S3 벽 영향   t=0.100025 s 클럼프 중심 vs W11 render_timeline frame(t=0.100025) **행 순서(row-wise) 대조**.
               W11 원자료에는 명시 ID 배열이 없고 두 실행이 같은 동결 npz 순서를 그대로 쓰므로 보존된 입력 순서로 맞춘다.
               PASS = max|dp| <= 0.5 mm AND p99 <= 0.1 mm — **서술적 비인과 게이트**다.
               실패해도 production 을 자동으로 막지 않으며, 실패하면 "W13 퍼내기를 W11 과 비교 가능"이라고 쓰지 않는다.
  S4 봉쇄      **중심 기준**. 동결 더미는 t=0 부터 구성 구 89개·최대 0.001220 mm 의 얕은 겹침을 이미 갖고 있으므로
               "구 겹침 0" 은 t=0 에도 성립하지 않는 잘못된 게이트였다. 대신
               S4a 클럼프 중심이 동결 안쪽 상자를 벗어나지 않음(관통-통과 없음)
               S4b 구성 구 **중심**이 안쪽 옆면을 넘지 않음(through-wall 없음)
               S4c 구성 구 soft overlap 개수/최대깊이와 바닥/윗단 여유는 **보고값**(t=0 기준선과 나란히)
  S5 무결성    NaN/Inf 0 · 입자/ID 수 불변 · pop-stop 0 · max v 보고
  S6 벽시계    측정 wall_s/sim_s 보고(외삽 대체)
비교 불가 필드(W11 원자료에 속도/접촉 동시각 배열 없음)는 "unavailable" 로 보고하며 만들어내지 않는다.
"""
import json, sys
from pathlib import Path
import numpy as np

W11_RT = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grab_track/g19_servo_direct/"
              "s1_v1_sim/w11_dt_sensitivity_20260911/cell_dt1e6_seed460/render_timeline_seed460.npz")
TARGET_T = 0.100025
G = {"S3_max_mm": 0.5, "S3_p99_mm": 0.1}

def main(att):
    att = Path(att)
    rcp = att / "EXECUTION_RECEIPT.json"
    if not rcp.exists():
        print("  [FAIL] execution_receipt_exists: " + str(rcp))
        print("W13_WALL_SMOKE_FAIL")
        raise SystemExit(1)
    receipt = json.loads(rcp.read_text())
    sim_step = next((s_ for s_ in receipt.get("steps", []) if s_["step"] == "step1_simulation"), None)
    res = json.load(open(att / "w13_cycle_seed460.json"))
    z = np.load(att / "w13_cycle_seed460.npz", allow_pickle=True)
    R = {"artifact": "W13_WALL_REGRESSION_VERDICT", "attempt": str(att), "prefrozen_gates": G}
    hvp = att / "HASH_VERIFICATION_RECEIPT.json"
    hv = json.loads(hvp.read_text()) if hvp.exists() else {}
    R["S0_hash_verification"] = {"receipt": str(hvp), "n_checked": hv.get("n_checked"),
                                 "mismatches": hv.get("mismatches"),
                                 "pre_existing_attempt_entries": hv.get("pre_existing_attempt_entries_other_than_manifest"),
                                 "pass": bool(hvp.exists() and not hv.get("mismatches")
                                              and not hv.get("pre_existing_attempt_entries_other_than_manifest"))}
    R["S0_process_rc"] = {"receipt": str(rcp), "step1_returncode": None if sim_step is None else sim_step["returncode"],
                          "timed_out": None if sim_step is None else sim_step["timed_out"],
                          "wall_s": None if sim_step is None else sim_step["wall_s"],
                          "auto_retry": receipt.get("auto_retry"),
                          "commands_json_sha256": receipt.get("commands_json_sha256"),
                          "pass": bool(sim_step is not None and sim_step["returncode"] == 0
                                       and not sim_step["timed_out"] and receipt.get("auto_retry") is False)}
    R["S1_completed"] = {"diverged": res["diverged"], "stopped_early": res.get("stopped_early_after_phase"),
                         "fail_reason": res.get("fail_reason"),
                         "pass": bool(not res["diverged"] and res.get("stopped_early_after_phase") == "settle")}
    nf, nd = np.asarray(z["scalar_n_fixed"]), np.asarray(z["scalar_n_door"])
    R["S2_tool_no_contact"] = {"max_n_fixed": int(nf.max()), "max_n_door": int(nd.max()), "n_sync": int(len(nf)),
                               "pass": bool(nf.max() == 0 and nd.max() == 0)}
    t = np.asarray(z["particle_frame_t_s"], float)
    k = int(np.argmin(np.abs(t - TARGET_T)))
    w11 = np.load(W11_RT, allow_pickle=True)
    tw = np.asarray(w11["t_s"], float)
    kw = int(np.argmin(np.abs(tw - TARGET_T)))
    a = np.asarray(z["particle_pos_m"][k], float)
    b = np.asarray(w11["clump_pos_m"][kw], float)
    ok_t = bool(abs(t[k] - TARGET_T) < 1e-9 and abs(tw[kw] - TARGET_T) < 1e-9)
    d = np.linalg.norm(a - b, axis=1) * 1000 if a.shape == b.shape else None
    R["S3_wall_change_effect"] = {
        "w13_frame_t_s": float(t[k]), "w11_frame_t_s": float(tw[kw]), "exact_same_source_time": ok_t,
        "n_w13": int(a.shape[0]), "n_w11": int(b.shape[0]),
        "pairing": "row-wise under preserved frozen-npz input order (W11 raw has no explicit ID array)",
        "max_mm": None if d is None else round(float(d.max()), 6),
        "p99_mm": None if d is None else round(float(np.percentile(d, 99)), 6),
        "median_mm": None if d is None else round(float(np.median(d)), 6),
        "velocity_comparison": "unavailable — W11 render_timeline 에 속도 배열이 없다. 두 번째 원본 실행을 만들지 않는다.",
        "pass": bool(ok_t and d is not None and d.max() <= G["S3_max_mm"] and np.percentile(d, 99) <= G["S3_p99_mm"])}
    box = np.asarray(z["box_bounds_m"], float)
    T0 = {"constituent_spheres_overlapping_inner_side_planes": 89, "max_overlap_depth_mm": 0.001220,
          "clump_centers_outside_inner_box": 0, "min_sphere_bottom_z_mm": -0.010539,
          "max_sphere_top_z_mm": 41.180772,
          "source": "동결 더미 npz 에서 직접 계산한 immutable t=0 기준선(감사 보고값과 일치)"}
    ov = np.asarray(z["scalar_n_sphere_overlap_wall"])
    gapmin = np.asarray(z["scalar_min_sphere_wall_gap_mm"])
    cen_s = np.asarray(z["scalar_n_sphere_center_outside_wall"])
    cen_c = np.asarray(z["scalar_n_clump_center_outside_box"])
    pp = np.asarray(z["particle_pos_m"][-1], float)
    R["S4_containment"] = {
        "t0_immutable_baseline": T0,
        "S4a_clump_center_containment": {"max_outside_count": int(cen_c.max()),
                                         "pass": bool(cen_c.max() == 0)},
        "S4b_no_through_wall_sphere_centers": {"max_outside_count": int(cen_s.max()),
                                               "pass": bool(cen_s.max() == 0)},
        "S4c_reported_only": {"max_sphere_soft_overlap_count": int(ov.max()),
                              "min_sphere_wall_gap_mm": round(float(gapmin.min()), 6),
                              "max_near_count": int(np.asarray(z["scalar_n_sphere_near_wall"]).max()),
                              "max_tray_contacts": int(np.asarray(z["scalar_n_tray_contacts"]).max()),
                              "final_clump_z_min_m": round(float(pp[:, 2].min()), 6),
                              "final_clumps_above_box_top": int((pp[:, 2] > float(box[2, 1])).sum()),
                              "note": "얕은 soft overlap 은 동결 더미가 t=0 부터 갖는 성질이다. near/overlap/contact 를 "
                                      "각각 보고하며 근접만으로 접촉을 요구하지 않는다. 이 항목은 게이트가 아니다."},
        "pass": bool(cen_c.max() == 0 and cen_s.max() == 0)}
    nonf = np.asarray(z["scalar_max_abs_pos_nonfinite"])
    pid = np.asarray(z["particle_ids"], int)
    id_ok = bool(pid.shape[0] == 20000 and np.array_equal(pid, np.arange(20000)))
    R["S5_integrity"] = {"nonfinite_total": float(nonf.sum()), "n_particles": int(res["particle"]["n"]),
                         "particle_ids_len": int(pid.shape[0]),
                         "particle_ids_exact_0_to_19999_unique_ordered": id_ok,
                         "syncs_over_pop_speed": res["pops"]["syncs_over_pop_speed"],
                         "v_particle_max_m_s": res["pops"]["v_particle_max_m_s"],
                         "pass": bool(nonf.sum() == 0 and id_ok and int(pid.shape[0]) == int(res["particle"]["n"])
                                      and res["pops"]["v_particle_max_m_s"] < 20.0)}
    sim_s = float(res["trajectory"]["sim_time_s"])
    wall_s = float(sim_step["wall_s"]) if sim_step else float(res["wall_seconds"])
    R["S6_wall_rate"] = {"sim_s": sim_s, "wall_s_process": wall_s,
                         "wall_s_in_result_json": float(res["wall_seconds"]),
                         "measured_wall_per_sim_s": round(wall_s / max(sim_s, 1e-12), 2),
                         "w11_reference_wall_per_sim_s": 868.93,
                         "note": "이 측정값이 선형 외삽을 대체한다. jitify 초기화가 포함된 값임을 함께 본다.",
                         "pass": True}
    R["all_pass"] = all(v.get("pass", True) for v in R.values() if isinstance(v, dict))
    json.dump(R, open(att / "wall_regression_verdict.json", "w"), ensure_ascii=False, indent=2, default=float)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in R.items()}, ensure_ascii=False, indent=1))
    print("S3", json.dumps(R["S3_wall_change_effect"], ensure_ascii=False))
    print("S6", json.dumps(R["S6_wall_rate"], ensure_ascii=False))
    print("W13_WALL_SMOKE_PASS" if R["all_pass"] else "W13_WALL_SMOKE_FAIL")

if __name__ == "__main__":
    main(sys.argv[1])
