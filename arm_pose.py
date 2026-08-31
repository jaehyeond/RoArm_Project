"""관절 6개를 명시 각도로 보내고 도달까지 폴링. 사용:
    python arm_pose.py <base> <shoulder> <elbow> <wrist_pitch> <wrist_roll> <gripper>
    python arm_pose.py --read                       # 읽기만

jaw_step.py의 문제 2건을 고친 판:
  1) 현재 관절을 읽어서 되먹임하면 중력 처짐이 목표로 굳는다 (2026-08-27 실측:
     shoulder 2.81→3.52→4.13, 호출당 +0.7°). → 자세를 인자로 받아 고정한다.
  2) blind sleep 하면 서보가 기어간다 (tech_servo_observer_effect.md).
     → 자지 않고 joints_angle_get()을 계속 호출한다. 읽기가 곧 구동이다.
"""
import sys, time, json, logging
logging.getLogger().setLevel(logging.CRITICAL)

from roarm_sdk.roarm import roarm
from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback

def _silent_process(self, data, genre):
    if not data:
        return None
    res, valid = [], []
    if genre == JsonCmd.FEEDBACK_GET:
        valid = [data['x'], data['y'], data['z']]
        if self.type == "roarm_m3":
            valid = handle_m3_feedback(valid, data)
    else:
        valid = data
    res.append(valid)
    return res
DataProcessor._process_received = _silent_process

PORT = "/dev/ttyUSB0"
STABLE_N, TOL, TIMEOUT_S = 6, 0.15, 25.0

arm = roarm(roarm_type="roarm_m3", port=PORT, baudrate=115200)
time.sleep(0.5)
arm.torque_set(cmd=1)
time.sleep(0.3)

def safe_get(retries=8):
    for _ in range(retries):
        try:
            v = arm.joints_angle_get()
            if v is not None and len(v) == 6 and all(x is not None for x in v):
                return v
        except Exception:
            pass
        time.sleep(0.15)
    raise RuntimeError("joints_angle_get failed after retries")

before = safe_get()

if len(sys.argv) == 2 and sys.argv[1] == "--read":
    arm.disconnect()
    print(json.dumps({"joints": [round(a, 2) for a in before]}, ensure_ascii=False))
    sys.exit(0)

target = [float(x) for x in sys.argv[1:7]]
arm.joints_angle_ctrl(angles=target, speed=200, acc=100)

t0, hist, after = time.time(), [], before
while time.time() - t0 < TIMEOUT_S:
    after = safe_get()
    hist.append(tuple(round(a, 2) for a in after))
    if len(hist) >= STABLE_N:
        win = hist[-STABLE_N:]
        if all(max(w[i] for w in win) - min(w[i] for w in win) < TOL for i in range(6)):
            break
    time.sleep(0.05)

arm.disconnect()
print(json.dumps({
    "target":  [round(a, 2) for a in target],
    "before":  [round(a, 2) for a in before],
    "after":   [round(a, 2) for a in after],
    "err":     [round(after[i] - target[i], 2) for i in range(6)],
    "settle_s": round(time.time() - t0, 2),
    "polls": len(hist),
}, ensure_ascii=False))
