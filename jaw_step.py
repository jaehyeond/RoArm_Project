"""그리퍼를 지정 각도 하나로 보내고 state°를 읽어 출력한다. 사용: python jaw_step.py <cmd_deg>

에이전트가 구동하고 사람이 캘리퍼로 재는 분업용. gripper_stroke_probe.py(4/27 검증)에서
연결·억압·safe_get·SETTLE_S를 그대로 가져왔고, 각도 1개만 처리하도록 줄였다.
그리퍼 외 관절은 현재 자세 고정 — 팔은 움직이지 않는다.
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

PORT = "/dev/ttyUSB0"   # 2026-08-27 실측: 한 팔만 연결 시 어느 팔이든 USB0
SETTLE_S = 3.0          # 4/27 실증: 1.5s는 이동 중 read(~17 deg/s). 3s = 완전 도달.

target = float(sys.argv[1])

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

cur = safe_get()
base, shoulder, elbow, wp, wr, g_before = cur

arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, target], speed=200, acc=100)

# ⚠️ tech_servo_observer_effect.md: 서보 속도가 joints_angle_get() 호출 빈도에 의존한다.
# blind sleep 하면 서보가 기어간다 (2026-08-27 실측: cmd=60에 3s sleep → 5.63°에서 정지).
# 그래서 자지 않고 **계속 읽으면서** 도달을 감지한다 — 읽기가 곧 구동이다.
STABLE_N, TOL, TIMEOUT_S = 6, 0.15, 20.0
t0, hist, after = time.time(), [], None
while time.time() - t0 < TIMEOUT_S:
    after = safe_get()
    hist.append(after[5])
    if len(hist) >= STABLE_N and (max(hist[-STABLE_N:]) - min(hist[-STABLE_N:])) < TOL:
        break
    time.sleep(0.05)

arm.disconnect()

print(json.dumps({
    "cmd_deg": target,
    "state_before": round(g_before, 2),
    "state_after": round(after[5], 2),
    "settle_s": round(time.time() - t0, 2),
    "polls": len(hist),
    "other_joints": [round(a, 2) for a in after[:5]],
}, ensure_ascii=False))
