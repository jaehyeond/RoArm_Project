#!/usr/bin/env python3
"""Phase 0-5c: Safe auto-calibration with Z_FLOOR protection
Z_FLOOR(-70mm) 이상에서만 움직임. 책상 충돌 방지."""
import time
import math
import logging

logging.getLogger('BaseController').setLevel(logging.CRITICAL)

from roarm_sdk.roarm import roarm
from roarm_sdk.common import handle_m3_feedback, JsonCmd

# 안전 한계: FK z가 이 값 아래면 해당 위치 건너뜀
Z_SAFE_LIMIT = -70  # 실측 책상 z=-120, -70이면 50mm 여유

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
time.sleep(1)
arm.torque_set(cmd=1)
time.sleep(0.5)
arm.move_init()
time.sleep(2)

# 관절 범위 (안전한 범위만)
# shoulder: 15~75 (너무 작으면 팔이 위로만, 너무 크면 책상 충돌)
# elbow: 20~100 (120 이상은 책상 충돌 위험)
BASE_ANGLES = [-60, -30, 0, 30, 60]
SHOULDER_ANGLES = [15, 30, 45, 60, 75]
ELBOW_ANGLES = [20, 40, 60, 80]

results = []
skipped = 0
total = len(BASE_ANGLES) * len(SHOULDER_ANGLES) * len(ELBOW_ANGLES)
print(f"=== Safe Auto Calibration: {total} positions (Z_SAFE={Z_SAFE_LIMIT}mm) ===\n")

count = 0
for base in BASE_ANGLES:
    for shoulder in SHOULDER_ANGLES:
        for elbow in ELBOW_ANGLES:
            count += 1
            angles_cmd = [base, shoulder, elbow, 0, 0, 50]
            arm.joints_angle_ctrl(angles=angles_cmd, speed=500, acc=200)
            time.sleep(1.5)

            # FK z 안전 체크
            pose = None
            for _ in range(5):
                try:
                    pose = arm.pose_get()
                    if pose and len(pose) >= 3:
                        break
                except Exception:
                    time.sleep(0.1)

            if pose:
                fk_x, fk_y, fk_z = pose[0], pose[1], pose[2]
                fk_dist = math.sqrt(fk_x**2 + fk_y**2)

                if fk_z < Z_SAFE_LIMIT:
                    skipped += 1
                    # 즉시 안전 위치로 복귀
                    arm.joints_angle_ctrl(angles=[base, 30, 30, 0, 0, 50], speed=500, acc=200)
                    time.sleep(1)
                    if count % 10 == 0:
                        print(f"  [{count}/{total}] base={base:+3d} sh={shoulder:2d} el={elbow:3d} "
                              f"→ SKIPPED (z={fk_z:.0f} < {Z_SAFE_LIMIT})")
                    continue

                results.append({
                    'base': base, 'shoulder': shoulder, 'elbow': elbow,
                    'fk_x': fk_x, 'fk_y': fk_y, 'fk_z': fk_z,
                    'fk_dist': fk_dist,
                })
                if count % 10 == 0 or count == total:
                    print(f"  [{count}/{total}] base={base:+3d} sh={shoulder:2d} el={elbow:3d} "
                          f"→ dist={fk_dist:.0f} z={fk_z:.0f}")

# 초기 위치 복귀
arm.move_init()
time.sleep(1)
arm.disconnect()

# 분석
print(f"\n=== {len(results)} positions OK, {skipped} skipped (too low) ===\n")

if not results:
    print("No valid positions!")
    exit()

dists = [r['fk_dist'] for r in results]
zs = [r['fk_z'] for r in results]
print(f"FK dist range: {min(dists):.0f} ~ {max(dists):.0f}mm")
print(f"FK z range:    {min(zs):.0f} ~ {max(zs):.0f}mm")

# base별 분류
print("\n=== Base angle별 FK dist ===")
for base in BASE_ANGLES:
    subset = [r for r in results if r['base'] == base]
    if subset:
        d = [r['fk_dist'] for r in subset]
        z = [r['fk_z'] for r in subset]
        print(f"  base={base:+3d}°: dist={min(d):.0f}~{max(d):.0f} z={min(z):.0f}~{max(z):.0f}")

# Shoulder+Elbow 조합별 (base=0)
print("\n=== Shoulder+Elbow 조합별 dist (base=0) ===")
center = [r for r in results if r['base'] == 0]
for sh in SHOULDER_ANGLES:
    for el in ELBOW_ANGLES:
        matches = [r for r in center if r['shoulder'] == sh and r['elbow'] == el]
        if matches:
            r = matches[0]
            mark = " ★NEAR" if r['fk_dist'] < 320 else " ★FAR" if r['fk_dist'] > 380 else ""
            print(f"  sh={sh:2d} el={el:3d} → dist={r['fk_dist']:.0f}mm z={r['fk_z']:.0f}mm{mark}")

# classify_zone 검증
from collect_data_manual import classify_zone
print("\n=== classify_zone() 검증 ===")
zone_counts = {}
for r in results:
    zone = classify_zone(r['base'], r['fk_dist'], r['fk_z'])
    zone_counts[zone] = zone_counts.get(zone, 0) + 1
    # 대표 샘플 출력
for zone, cnt in sorted(zone_counts.items()):
    samples = [r for r in results if classify_zone(r['base'], r['fk_dist'], r['fk_z']) == zone][:2]
    sample_str = ", ".join(f"b={s['base']:+d} d={s['fk_dist']:.0f}" for s in samples)
    print(f"  {zone:12s}: {cnt:3d} positions (ex: {sample_str})")
