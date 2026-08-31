"""Jaw stroke 캘리퍼 실측 — cmd° / state° / 개구 mm 동시 기록 (63rd :131 처방).

gripper_stroke_probe.py(4/27)의 확장. 바뀐 것 두 가지뿐:
  1) 각 지점에서 멈추고 캘리퍼 측정값을 입력받는다 (원본은 논스톱이라 측정 불가)
  2) 열림/닫힘 양방향 왕복 → 백래시 측정

목적 (둘 다 필요):
  (A) cmd 30° 초과의 실제 개구 → 그랩 개구부 58mm 설계 확정
  (B) 서보 state° ↔ URDF θ 매핑 → 30mm 물체가 물리 37.9° / sim 22.8°에서 정지하는 불일치 해소

안전:
  - 그리퍼 외 관절은 현재 자세 고정. 팔은 안 움직인다.
  - 그리퍼에 아무것도 물리지 않은 상태에서 실행할 것.
  - 저항이 느껴지면 Enter 대신 q 입력 → 그 지점까지 저장하고 종료.
"""
import time
import json
import logging
logging.getLogger().setLevel(logging.CRITICAL)

from roarm_sdk.roarm import roarm
from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback

# SDK print(data) 스팸 억제 — 원본 처리 로직 보존 (collect_data_manual.py 패턴)
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

# ⚠️ USB0=Leader / USB1=Follower 매핑은 **두 팔이 모두 연결됐을 때의 열거 순서**다.
# 한 팔만 꽂으면 어느 팔이든 ttyUSB0으로 잡힌다. 2026-08-27 실측: ttyUSB0만 존재.
# 두 팔을 다 연결해서 돌릴 때는 "/dev/ttyUSB1"로 되돌릴 것.
PORT = "/dev/ttyUSB0"
SETTLE_S = 3.0                 # 4/27 실증: 1.5s는 이동 중 read(~17 deg/s). 3s = 완전 도달.
OUT = "claudedocs/runtime_logs/jaw_stroke_caliper_20260827.json"

# 0~38°는 기존 피팅(jaw_mm ≈ 0.75 × state°)과 겹쳐야 두 데이터셋을 이을 수 있다.
UP   = [0, 15, 30, 45, 60, 75, 89]   # 열면서
DOWN = [75, 60, 45, 30, 15, 0]       # 닫으면서 (백래시)

arm = roarm(roarm_type="roarm_m3", port=PORT, baudrate=115200)
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
base, shoulder, elbow, wp, wr, g0 = cur

print("\n" + "=" * 62)
print("  측정 기준점을 먼저 정하고 사진으로 남길 것 — 조 끝(tip) vs 패드 면.")
print("  기존 0.75×state 피팅과 이으려면 **패드 면** 권장.")
print("  전 지점에서 같은 점을 재야 한다.")
print("=" * 62)

records = []

def sweep(seq, direction):
    for g in seq:
        arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, g],
                              speed=200, acc=100)
        time.sleep(SETTLE_S)                       # 멈춘 뒤에 읽는다 (observer effect)
        st = safe_get(arm)[5]
        print(f"\n[{direction}] cmd={g:3d}°  →  state={st:6.2f}°")
        raw = input("    캘리퍼 개구 mm (Enter=건너뜀, q=중단): ").strip()
        if raw.lower() == "q":
            return False
        jaw = None
        if raw:
            try:
                jaw = float(raw)
            except ValueError:
                print("    숫자가 아님 — 건너뜀")
        records.append({"dir": direction, "cmd_deg": g,
                        "state_deg": round(st, 2), "jaw_mm": jaw})
    return True

ok = sweep(UP, "open")
if ok:
    ok = sweep(DOWN, "close")

# 원래 각도 복귀
arm.joints_angle_ctrl(angles=[base, shoulder, elbow, wp, wr, g0], speed=200, acc=100)
time.sleep(SETTLE_S)
arm.disconnect()

with open(OUT, "w") as f:
    json.dump({"port": PORT, "settle_s": SETTLE_S, "initial": [round(a, 2) for a in cur],
               "records": records}, f, indent=2, ensure_ascii=False)

print(f"\n저장: {OUT}  ({len(records)} 지점)")
print("\ncmd  state   open_mm  close_mm  backlash")
by = {}
for r in records:
    by.setdefault(r["cmd_deg"], {})[r["dir"]] = r
for c in sorted(by):
    o = by[c].get("open", {}).get("jaw_mm")
    d = by[c].get("close", {}).get("jaw_mm")
    st = by[c].get("open", by[c].get("close"))["state_deg"]
    bl = f"{abs(o - d):6.2f}" if (o is not None and d is not None) else "     -"
    print(f"{c:3d}  {st:6.2f}  {o if o is not None else '   -':>8}  "
          f"{d if d is not None else '   -':>8}  {bl}")
