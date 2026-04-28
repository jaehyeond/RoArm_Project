"""Probe follower gripper stroke: command deg → read back deg.

Safe: slow speed, small gripper-only motion, keeps other joints at current pose.
"""
import time
import logging
logging.getLogger().setLevel(logging.CRITICAL)

from roarm_sdk.roarm import roarm
from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback

# Suppress SDK print(data) spam — preserve original processing logic (collect_data_manual.py pattern)
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

arm = roarm(roarm_type="roarm_m3", port="/dev/ttyUSB1", baudrate=115200)  # Follower (USB1). Leader=USB0.
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
print(f"INITIAL: {[round(a,2) for a in cur]}")

base, shoulder, elbow, wp, wr, _ = cur

probe = [0, 30, 60, 90]
SETTLE_S = 3.0  # 4/27 verified: 1.5s reads mid-motion (~17 deg/s effective). 3s = full arrival.
print(f"\nCommanding gripper through {probe} deg (settle {SETTLE_S}s, other joints fixed):")
for g in probe:
    arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, g], speed=200, acc=100)
    time.sleep(SETTLE_S)
    cur = safe_get(arm)
    print(f"  cmd gripper={g:3d} deg  →  state gripper={cur[5]:6.2f} deg  (full: {[round(a,2) for a in cur]})")

print(f"\nRestoring original gripper angle {probe[0]}")
arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, probe[0]], speed=200, acc=100)
time.sleep(SETTLE_S)
arm.disconnect()
print("Done.")
