"""W10 G0 구 회귀 판정을 pod 결과에 그대로 적용: diverged=false ∧ n_in_cavity∈[268,362] ∧ stops 전부 servo_stall. 사후 완화 없음."""
import json, sys
d = json.load(open(sys.argv[1]))
n = d["capture"]["n_in_cavity"]; stops = d["door"]["stops"]
checks = {"diverged_false": d["diverged"] is False,
          "n_in_cavity_in_268_362": 268 <= n <= 362,
          "stops_all_servo_stall": bool(stops) and all(s["reason"] == "servo_stall" for s in stops),
          "pop_steps_over_5ms_zero": d["pops"]["steps_over_pop_speed"] == 0}
rec = {"artifact": "W19_SPHERE_REGRESSION_CHECK", "result_json": sys.argv[1], "n_in_cavity": n, "mass_g": d["capture"]["mass_g"],
       "stops": [(s["phase"], s["reason"], s["q_deg"]) for s in stops], "wall_seconds": d.get("wall_seconds"),
       "v_particle_max_m_s": d["pops"]["v_particle_max_m_s"], "checks": checks, "pass": all(checks.values()),
       "local_reference_w10_G0": {"n_in_cavity": 287, "mass_g": 10.2774, "wall_seconds": 103.51, "stops": ["servo_stall", "servo_stall"]}}
print(json.dumps(rec, ensure_ascii=False, indent=1))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
sys.exit(0 if rec["pass"] else 3)
