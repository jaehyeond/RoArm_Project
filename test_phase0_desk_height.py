#!/usr/bin/env python3
"""Phase 0: 책상 표면 높이 실측 — 토크 OFF, 30초간 FK 실시간 출력

사용법:
  1. 스크립트 실행
  2. 토크 OFF 상태에서 손으로 그리퍼를 책상 표면에 대기
  3. 30초간 1초마다 FK 출력 → z값 중 가장 낮은 것 = 책상 높이
  4. 여러 위치에서 테스트 (앞, 좌, 우)
"""
import time
import math
import logging

logging.getLogger('BaseController').setLevel(logging.CRITICAL)

from roarm_sdk.roarm import roarm
from roarm_sdk.common import handle_m3_feedback, JsonCmd

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
time.sleep(1.5)  # 시리얼 초기화 대기 (0.5초는 부족!)

# 먼저 토크 ON → 초기 위치 → 토크 OFF (안전한 시작)
arm.torque_set(cmd=1)
time.sleep(0.5)
arm.move_init()
time.sleep(2)
print("초기 위치로 이동 완료. 이제 토크 OFF합니다.")
arm.torque_set(cmd=0)
time.sleep(1)

print("=" * 60)
print("책상 표면 높이 실측 (45초)")
print("=" * 60)
print()
print("★★★ 토크 OFF! 손으로 팔을 자유롭게 움직일 수 있습니다 ★★★")
print()
print("그리퍼 끝을 책상 표면에 대세요.")
print("여러 위치 (앞/좌/우)에서 테스트하세요.")
print("z값이 변하면 정상입니다!")
print()
print("45초간 1초마다 FK z를 출력합니다.")
print()
print(f"{'초':>3s}  {'FK_x':>8s} {'FK_y':>8s} {'FK_z':>8s} {'dist':>6s}  {'base°':>6s} {'sh°':>6s} {'el°':>6s}")
print("-" * 70)

all_z = []
all_readings = []

for sec in range(45):
    # FK 읽기
    pose = None
    for _ in range(5):
        try:
            pose = arm.pose_get()
            if pose and len(pose) >= 3:
                break
        except Exception:
            time.sleep(0.05)

    angles = None
    for _ in range(5):
        angles = arm.joints_angle_get()
        if angles and len(angles) == 6 and angles[0] != 180:
            break
        time.sleep(0.05)

    if pose and angles:
        fk_x, fk_y, fk_z = pose[0], pose[1], pose[2]
        fk_dist = math.sqrt(fk_x**2 + fk_y**2)
        all_z.append(fk_z)
        all_readings.append({
            'sec': sec, 'fk_x': fk_x, 'fk_y': fk_y, 'fk_z': fk_z,
            'fk_dist': fk_dist, 'base': angles[0], 'sh': angles[1], 'el': angles[2],
        })
        print(f"{sec:3d}  {fk_x:+8.1f} {fk_y:+8.1f} {fk_z:+8.1f} {fk_dist:6.0f}  "
              f"{angles[0]:+6.1f} {angles[1]:+6.1f} {angles[2]:+6.1f}")
    else:
        print(f"{sec:3d}  READ FAILED")

    time.sleep(1)

# 토크 ON + 복귀
arm.torque_set(cmd=1)
time.sleep(0.3)
arm.move_init()
time.sleep(1)
arm.disconnect()

# 분석
print()
print("=" * 60)
print("=== RESULTS ===")
print("=" * 60)

if all_z:
    z_min = min(all_z)
    z_max = max(all_z)
    z_mean = sum(all_z) / len(all_z)

    # 책상에 닿은 시점의 z값들 (가장 낮은 z = 책상 표면)
    # 가장 낮은 5개 z값의 평균
    sorted_z = sorted(all_z)
    lowest_5 = sorted_z[:min(5, len(sorted_z))]
    z_desk = sum(lowest_5) / len(lowest_5)

    print(f"FK z 범위: {z_min:.1f} ~ {z_max:.1f}mm")
    print(f"FK z 평균: {z_mean:.1f}mm")
    print(f"가장 낮은 5개 z 평균: {z_desk:.1f}mm  ← 책상 표면 추정")
    print()

    # 안전 마진
    z_floor_10 = z_desk + 10
    z_floor_20 = z_desk + 20
    z_floor_30 = z_desk + 30
    print(f"Z_FLOOR 후보:")
    print(f"  보수적 (+30mm): {z_floor_30:.0f}mm  ← 스펀지 두께 여유")
    print(f"  표준   (+20mm): {z_floor_20:.0f}mm  ← 권장")
    print(f"  공격적 (+10mm): {z_floor_10:.0f}mm  ← 최소 마진")
    print()
    print("이 Z_FLOOR를 deploy_smolvla.py와 collect_data_manual.py에 적용합니다.")
    print()

    # 실제 물체 잡기에 필요한 z 범위 추정
    # 스펀지 높이 ~30mm → 그리퍼가 스펀지 위에서 z_desk + 30~50mm
    print("=== 물체 잡기 z 범위 추정 ===")
    print(f"책상 표면: z ≈ {z_desk:.0f}mm")
    print(f"스펀지 상면 (두께 ~30mm): z ≈ {z_desk + 30:.0f}mm")
    print(f"접근 높이: z ≈ {z_desk + 80:.0f}mm")
    print(f"들어올린 높이: z ≈ {z_desk + 150:.0f}mm")
