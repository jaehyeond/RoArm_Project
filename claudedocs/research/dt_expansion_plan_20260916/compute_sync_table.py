"""설치 DEME 2.4.0 누산 규칙(감사 DEME_SOURCE_BOUND.md: float32 h 를 float64 로 반복 더해 요청 D 이상이 될 때까지)을
CPU 로 재현해 dt 별 실제 sync 지속시간·내부 스텝·문 이동량(22.5°/s)·CD 간격을 표로 만든다. 새 물리 0."""
import json, sys
import numpy as np

def steps(D, h):
    h32 = float(np.float32(h)); acc = 0.0; n = 0
    while D > acc:
        acc += h32; n += 1
    return n, acc

CD_FREQ = 20          # 동결 params cd_update_freq (dynamics steps between contact detections, 프로젝트 전달값)
VMAX = 5.0            # 동결 params max_velocity_m_s (SetMaxVelocity)
rows = []
for D in (0.004, 0.001, 0.0001):
    for h in (1e-6, 2e-6, 1e-5, 1e-4, 1e-3):
        n, acc = steps(D, h)
        rows.append({"requested_D_s": D, "nominal_dt_s": h, "float32_dt_s": float(np.float32(h)), "steps": n,
                     "actual_s": acc, "overshoot_s": acc - D, "door_motion_deg_at_22p5deg_s": 22.5 * acc,
                     "cd_interval_s_if_freq20": CD_FREQ * float(np.float32(h)),
                     "cd_margin_m_if_vmax5": VMAX * CD_FREQ * float(np.float32(h))})
out = sys.argv[1] if len(sys.argv) > 1 else "sync_duration_table.json"
json.dump({"artifact": "W15_DT_PLAN_SYNC_TABLE", "rule": "A_final = first repeated-binary64 sum of float32(h) with A >= D (DEME_SOURCE_BOUND.md)",
           "cd_update_freq_frozen": CD_FREQ, "max_velocity_frozen_m_s": VMAX, "rows": rows}, open(out, "w"), indent=1)
for r in rows:
    print(f"D={r['requested_D_s']:.4f} dt={r['nominal_dt_s']:.0e} steps={r['steps']:5d} actual={r['actual_s']*1e3:.7f} ms "
          f"door={r['door_motion_deg_at_22p5deg_s']:.5f} deg  CD_int={r['cd_interval_s_if_freq20']*1e3:.4f} ms  CD_margin={r['cd_margin_m_if_vmax5']*1e3:.3f} mm")
