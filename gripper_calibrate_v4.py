"""Gripper deg->mm calibration v4: poll-until-settled.

V4 fixes from V3:
  V3 single-read settle exposed observer effect: servo motion speed depends on
  read frequency.  V3 (1 read/cmd, sleep 6s) -> ~0.87 deg/s, never reached cmd.
  V2 (4 reads at 1/3/6/9s)                  -> ~4.8 deg/s,  reached cmd by 6s.
  V4 polls every 1s (V2 pattern) until state is stable AND close to cmd.

Drift detection (V3 carry-over):
  After SETTLED, hold HOLD_S seconds for caliper measurement.
  AFTER state vs SETTLED state -> drift > 1 deg = external force INVALID.

Convention: cmd 0 = jaw CLOSED, cmd 89 = jaw OPEN (mech max state ~88.3).

Usage (run in user terminal for direct stdout visibility):
  python gripper_calibrate_v4.py
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
    raise RuntimeError("safe_get failed")


def poll_until_settled(arm, target_g, max_s=12.0, poll=1.0, settle_tol=0.3, reach_tol=1.5):
    """Read state every `poll` s. Settled when |state[t]-state[t-1]|<settle_tol AND |state-cmd|<reach_tol."""
    states = []
    t0 = time.time()
    while time.time() - t0 < max_s:
        time.sleep(poll)
        s = safe_get(arm)[5]
        states.append(s)
        if len(states) >= 2:
            stable = abs(states[-1] - states[-2]) < settle_tol
            reached = abs(states[-1] - target_g) < reach_tol
            if stable and reached:
                return states[-1], len(states), time.time() - t0, "SETTLED"
    return states[-1] if states else float('nan'), len(states), time.time() - t0, "TIMEOUT"


cur = safe_get(arm)
print(f"INITIAL: {[round(a, 2) for a in cur]}")
base, shoulder, elbow, wp, wr, _ = cur

POINTS = [0, 30, 60, 80, 89]
HOLD_S = 12.0
DRIFT_THRESHOLD = 1.0
END_CMD = 30

print("=" * 72)
print("GRIPPER deg->mm CALIBRATION v4 (poll-until-settled)")
print()
print("CALIPER RULES:")
print("  1. Insert GENTLY -- do not push jaws apart.")
print("  2. READ then WITHDRAW within 5s. Do not hold caliper in jaws.")
print("  3. If jaw resistance felt against caliper -> measurement INVALID.")
print()
print("MEASUREMENT TEMPLATE (record):")
for g in POINTS:
    print(f"    cmd {g:2d}  ->  settled ___ deg  ->  inner width ___ mm")
print()
print("OFFLINE (after script ends, robot rests at cmd=30):")
print("  A. Finger length    (link5 housing edge -> fingertip, jaw closed)")
print("  B. Finger thickness (single jaw outer profile)")
print("  C. Sponge thickness (re-confirm 22-23 mm)")
print("=" * 72)

records = []
prev_state = cur[5]
for g in POINTS:
    delta = g - prev_state
    print(f"\n>>> CMD {g:3d} deg  (delta={delta:+.1f} from prev state {prev_state:.2f})")
    arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, g], speed=200, acc=100)
    state_settled, n_reads, t_elapsed, status = poll_until_settled(arm, g)
    gap = state_settled - g
    print(f"    {status} after {t_elapsed:4.1f}s ({n_reads} reads)  state={state_settled:6.2f}  gap={gap:+.2f}")
    print(f"    --> MEASURE NOW ({HOLD_S:.0f}s window). Caliper IN gently, READ, OUT immediately.")
    time.sleep(HOLD_S)
    state_after = safe_get(arm)[5]
    drift = state_after - state_settled
    flag = "OK" if abs(drift) <= DRIFT_THRESHOLD else "INVALID (external force)"
    print(f"    AFTER {HOLD_S:.0f}s state={state_after:6.2f}  drift={drift:+.2f}  [{flag}]")
    records.append((g, state_settled, state_after, drift, n_reads, t_elapsed, status, flag))
    prev_state = state_after

print(f"\nResting at safe cmd={END_CMD} deg")
arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, END_CMD], speed=200, acc=100)
poll_until_settled(arm, END_CMD)
arm.disconnect()

print()
print("=" * 72)
print("SUMMARY (match jaw width to SETTLED state, not cmd):")
print(f"  {'cmd':>4} {'settled':>9} {'after':>9} {'drift':>7} {'reads':>6} {'time':>6}  {'status':>8}  flag")
for g, s0, s1, d, n, t, st, f in records:
    print(f"  {g:>4} {s0:>9.2f} {s1:>9.2f} {d:>+7.2f} {n:>6} {t:>5.1f}s  {st:>8}  {f}")
print("=" * 72)
print("Done. Robot at cmd=30 deg. Proceed with offline A/B/C measurements.")
