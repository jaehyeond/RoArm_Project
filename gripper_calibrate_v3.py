"""Gripper deg→mm calibration v3 (caliper measurement, fail-safe).

V3 fixes from v1 (4/27 first attempt failed):
  - Direction: ascending 0→89 (matches H1/v2, caliper insertion is passive)
  - SETTLE_S: 3s → 6s (verified by v2: all cmd reach in ≤6s)
  - Warmup cmd 0° first (absorbs SDK init lag)
  - Hold-start + Hold-end state measurement → detects external force
  - Auto-flags |drift| > 1° as INVALID measurement

Convention:
  cmd 0°  = jaw CLOSED (narrowest)
  cmd 89° = jaw OPEN   (widest, mechanical max state ~88.3°)

Usage (run in user terminal, NOT via claude — direct stdout visibility):
  python gripper_calibrate_v3.py
  → Caliper rules below. Total runtime ~2 min.
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

POINTS = [0, 30, 60, 80, 89]
SETTLE_S = 6.0
HOLD_S = 15.0
DRIFT_THRESHOLD = 1.0  # deg
END_CMD = 30

print()
print("=" * 72)
print("GRIPPER deg->mm CALIBRATION v3")
print()
print("CALIPER RULES (CRITICAL — v1 failed due to violation):")
print("  1. Insert caliper between jaws GENTLY. Do not push jaws apart.")
print("  2. Read caliper value, then WITHDRAW IMMEDIATELY (do not hold).")
print("  3. If jaws feel resistance against caliper, MEASUREMENT IS INVALID.")
print("  4. Total contact time per measurement: < 5 seconds.")
print()
print("MEASUREMENT TEMPLATE (record these):")
for g in POINTS:
    print(f"    cmd {g:2d} deg  ->  state ___ deg (printed)  ->  inner width ___ mm")
print()
print("OFFLINE MEASUREMENTS (after script ends, robot at cmd=30):")
print("  A. Finger length:    link5 housing edge -> fingertip (jaw closed)")
print("  B. Finger thickness: jaw outer profile (single jaw)")
print("  C. Sponge thickness: re-confirm 22-23 mm")
print("=" * 72)
print()

records = []
prev_state = cur[5]
for g in POINTS:
    delta = g - prev_state
    print(f"\n>>> CMD {g:3d} deg  (delta={delta:+.1f} deg from prev state {prev_state:.2f})")
    arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, g], speed=200, acc=100)
    time.sleep(SETTLE_S)
    state_settled = safe_get(arm)[5]
    print(f"    SETTLED  state = {state_settled:6.2f} deg  gap={state_settled - g:+.2f} deg")
    print(f"    --> MEASURE NOW (15s window). Caliper IN gently, READ, OUT immediately.")
    time.sleep(HOLD_S)
    state_after = safe_get(arm)[5]
    drift = state_after - state_settled
    flag = "OK" if abs(drift) <= DRIFT_THRESHOLD else "INVALID (external force detected)"
    print(f"    AFTER 15s state = {state_after:6.2f} deg  drift={drift:+.2f} deg  [{flag}]")
    records.append((g, state_settled, state_after, drift, flag))
    prev_state = state_after

print(f"\nResting at safe cmd={END_CMD} deg")
arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, END_CMD], speed=200, acc=100)
time.sleep(SETTLE_S)
arm.disconnect()

print()
print("=" * 72)
print("SUMMARY (use SETTLED state for jaw width matching):")
print(f"  {'cmd':>4} {'settled':>9} {'after15s':>9} {'drift':>7}  flag")
for cmd, s0, s1, d, f in records:
    print(f"  {cmd:>4} {s0:>9.2f} {s1:>9.2f} {d:>+7.2f}  {f}")
print("=" * 72)
print("\nDone. Robot stopped at cmd=30 deg. Now offline-measure A/B/C.")
