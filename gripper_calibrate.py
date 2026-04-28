"""Gripper deg→mm calibration: hold each cmd 15s for caliper measurement.

Convention (verified 4/27):
  cmd 0°  = jaw CLOSED (narrowest)
  cmd 89° = jaw OPEN   (widest, mechanical max)

Sequence: descends 89→0 so user sees jaw progressively close.
End state: jaw rests at cmd=30° (partially open) for follow-up offline measurements.

Usage:
  python gripper_calibrate.py
  → Run with caliper in hand. Each 15s hold = measurement window.
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


arm = roarm(roarm_type="roarm_m3", port="/dev/ttyUSB1", baudrate=115200)  # Follower (USB1)
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

POINTS = [89, 80, 60, 30, 0]
SETTLE_S = 3.0
HOLD_S = 15.0
END_CMD = 30  # safe mid position

print()
print("=" * 72)
print("GRIPPER deg→mm CALIBRATION")
print("  cmd 89° = OPEN (widest)   |   cmd 0° = CLOSED (narrowest)")
print()
print("  At each position, robot holds 15s. Insert caliper between jaws,")
print("  measure INNER WIDTH (mm) at jaw mid-depth.")
print()
print("  Recording template:")
for g in POINTS:
    print(f"    cmd {g:2d}°  →  state ___° (printed below)  →  inner width = ____ mm")
print()
print(f"  After end (jaw rests at cmd={END_CMD}°), measure offline:")
print("    1. Finger length:    link5 housing edge → fingertip (jaw closed)")
print("    2. Finger thickness: jaw outer profile")
print("    3. Sponge thickness: re-confirm 20 mm")
print("=" * 72)
print()

for g in POINTS:
    arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, g], speed=200, acc=100)
    time.sleep(SETTLE_S)
    cur = safe_get(arm)
    print(f"\n>>> HOLDING cmd={g:3d}°  state={cur[5]:6.2f}°  for {HOLD_S}s — MEASURE NOW")
    time.sleep(HOLD_S)

print(f"\nResting at safe cmd={END_CMD}° (jaw partially open)")
arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, END_CMD], speed=200, acc=100)
time.sleep(SETTLE_S)
arm.disconnect()
print(f"\nDone. Robot stopped at cmd={END_CMD}°. Proceed with offline measurements.")
