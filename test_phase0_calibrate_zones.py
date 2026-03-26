#!/usr/bin/env python3
"""Phase 0-5b: Zone calibration — torque OFF에서 손으로 이동하며 FK 범위 실측

사용법:
  python test_phase0_calibrate_zones.py

1. 토크 OFF 상태로 시작
2. 각 zone 설명에 따라 팔을 해당 위치로 이동
3. Enter를 누르면 현재 FK 기록
4. 5개 zone × 3회 = 15 측정
5. 결과로 classify_zone() 임계값 제안
"""
import time
import math
import logging
import sys

logging.getLogger('BaseController').setLevel(logging.CRITICAL)

from roarm_sdk.roarm import roarm
from roarm_sdk.common import handle_m3_feedback, JsonCmd

# SDK print 억제
_orig = roarm._process_received
def _silent(self, data, genre):
    if not data:
        return None
    res, valid_data = [], []
    if genre == JsonCmd.FEEDBACK_GET:
        valid_data.extend([data['x'], data['y'], data['z']])
        if self.type == "roarm_m3":
            valid_data = handle_m3_feedback(valid_data, data)
    else:
        valid_data = data
    res.append(valid_data)
    return res
roarm._process_received = _silent

arm = roarm(roarm_type='roarm_m3', port='/dev/ttyUSB0', baudrate=115200)
time.sleep(0.5)

# 토크 OFF
arm.torque_set(cmd=0)
time.sleep(0.3)
print("=== Zone Calibration (토크 OFF) ===")
print("손으로 로봇을 각 위치로 이동 후 Enter\n")

def read_state():
    """현재 관절 각도 + FK 읽기"""
    angles = None
    for _ in range(5):
        angles = arm.joints_angle_get()
        if angles and len(angles) == 6 and angles[0] != 180:
            break
        time.sleep(0.1)

    pose = None
    for _ in range(5):
        try:
            pose = arm.pose_get()
            if pose and len(pose) >= 3:
                break
        except Exception:
            time.sleep(0.1)

    return angles, pose

ZONES = [
    ("NEAR", "가까운 중앙 — 로봇 바로 앞, 물체를 가까이 놓는 위치 (3회)"),
    ("MID_LEFT", "왼쪽 중간 — 로봇 왼쪽 작업 영역 (3회)"),
    ("MID_RIGHT", "오른쪽 중간 — 로봇 오른쪽 작업 영역 (3회)"),
    ("FAR_CENTER", "먼 중앙 — 팔을 앞으로 쭉 뻗어 닿는 먼 위치 (3회)"),
    ("OVERHEAD", "높은 위치 — 물체를 들어올린 상태 (3회)"),
]

all_data = {}

for zone_name, desc in ZONES:
    print(f"\n{'='*50}")
    print(f"ZONE: {zone_name}")
    print(f"설명: {desc}")
    print(f"{'='*50}")

    zone_readings = []
    for trial in range(3):
        input(f"\n  [{zone_name} {trial+1}/3] 해당 위치로 이동 후 Enter... ")
        angles, pose = read_state()

        if angles and pose:
            base_angle = angles[0]
            fk_x, fk_y, fk_z = pose[0], pose[1], pose[2]
            fk_dist = math.sqrt(fk_x**2 + fk_y**2)

            reading = {
                'base_angle': base_angle,
                'fk_x': fk_x, 'fk_y': fk_y, 'fk_z': fk_z,
                'fk_dist': fk_dist,
                'joints': angles,
            }
            zone_readings.append(reading)

            print(f"  Joints: [{', '.join(f'{a:.1f}' for a in angles)}]")
            print(f"  FK: x={fk_x:.1f} y={fk_y:.1f} z={fk_z:.1f} dist={fk_dist:.1f}mm")
            print(f"  Base angle: {base_angle:.1f}°")
        else:
            print(f"  ERROR: 읽기 실패!")

    all_data[zone_name] = zone_readings

# 토크 ON + 초기 위치
arm.torque_set(cmd=1)
time.sleep(0.3)
arm.move_init()
time.sleep(1)
arm.disconnect()

# 결과 분석
print("\n\n" + "="*60)
print("=== CALIBRATION RESULTS ===")
print("="*60)

for zone_name in ["NEAR", "MID_LEFT", "MID_RIGHT", "FAR_CENTER", "OVERHEAD"]:
    readings = all_data[zone_name]
    if not readings:
        print(f"\n{zone_name}: NO DATA")
        continue

    dists = [r['fk_dist'] for r in readings]
    zs = [r['fk_z'] for r in readings]
    bases = [r['base_angle'] for r in readings]

    print(f"\n{zone_name}:")
    print(f"  FK dist: {min(dists):.0f} ~ {max(dists):.0f}mm (mean={sum(dists)/len(dists):.0f})")
    print(f"  FK z:    {min(zs):.0f} ~ {max(zs):.0f}mm (mean={sum(zs)/len(zs):.0f})")
    print(f"  Base:    {min(bases):.0f} ~ {max(bases):.0f}° (mean={sum(bases)/len(bases):.0f})")

print("\n\n=== SUGGESTED THRESHOLDS ===")
print("위 데이터를 기반으로 classify_zone() 임계값을 수정하세요.")
print("주의: zone 간 겹침이 있으면 base_angle로 먼저 분류 후 거리로 세분화")
