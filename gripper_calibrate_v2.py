"""Calibrate v2: DIAGNOSTIC probe (no caliper).

Goal: disambiguate v1 anomaly cause among:
  H_caliper:   user caliper exerted external force on jaw
  H_settle:    SETTLE_S=3s insufficient for large delta cmd
  H_first_cmd: SDK first-command lag
  H_direction: closing direction has lower torque than opening

Method:
  Sweep cmd 0 → 30 → 60 → 80 → 89 (ascending OPEN, matches H1).
  Sample state @ {1, 3, 6, 9}s after each cmd issue.
  Compare with H1 (open direction, no caliper) and v1 (close direction, with caliper).
End: rest at cmd=30°.
"""
import time
import logging
logging.getLogger().setLevel(logging.CRITICAL)

from roarm_sdk.roarm import roarm
from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback


def _silent_process(self, data, genre):
    if not data:
        return None
    res = []
    valid_data = []
    if genre == JsonCmd.FEEDBACK_GET:
        valid_data.append(data['x'])
        valid_data.append(data['y'])
        valid_data.append(data['z'])
        if self.type == "roarm_m3":
            valid_data = handle_m3_feedback(valid_data, data)
    else:
        valid_data = data
    res.append(valid_data)
    return res

DataProcessor._process_received = _silent_process


arm = roarm(roarm_type="roarm_m3", port="/dev/ttyUSB1", baudrate=115200)
time.sleep(0.5)
arm.torque_set(cmd=1)
time.sleep(0.3)


def safe_get(arm, retries=8):
    for i in range(retries):
        try:
            v = arm.joints_angle_get()
            if v is not None and len(v) == 6 and all(x is not None for x in v):
                return v
        except Exception:
            pass
        time.sleep(0.15)
    raise RuntimeError("joints_angle_get failed after retries")


cur = safe_get(arm)
print(f"INITIAL: {[round(a, 2) for a in cur]}")
base, shoulder, elbow, wp, wr, _ = cur

SWEEP = [0, 30, 60, 80, 89]
SAMPLES_S = [1.0, 3.0, 6.0, 9.0]
END_CMD = 30

print("=" * 72)
print("CALIBRATE v2 (DIAGNOSTIC, no caliper)")
print(f"  Sweep cmd {SWEEP}, sample state @ {SAMPLES_S}s each.")
print("=" * 72)

prev_state = cur[5]
for g in SWEEP:
    delta_cmd = g - prev_state
    print(f"\n>>> CMD {g:3d} deg  (delta={delta_cmd:+.1f} deg from prev state {prev_state:.2f})")
    arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, g], speed=200, acc=100)
    t_prev = 0.0
    states = []
    for t in SAMPLES_S:
        time.sleep(t - t_prev)
        cur_s = safe_get(arm)
        states.append(cur_s[5])
        t_prev = t
    final = states[-1]
    print(f"  state @1s={states[0]:6.2f}  @3s={states[1]:6.2f}  @6s={states[2]:6.2f}  @9s={states[3]:6.2f}")
    print(f"  final gap (state - cmd) = {final - g:+.2f} deg")
    prev_state = final

print(f"\nResting at safe cmd={END_CMD} deg")
arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, END_CMD], speed=200, acc=100)
time.sleep(6.0)
arm.disconnect()
print("Done.")
