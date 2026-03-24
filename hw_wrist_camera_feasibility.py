"""
hw_wrist_camera_feasibility.py
RoArm-M3-Pro 손목 카메라 장착 타당성 분석 도구

용도:
  1. 손목 서보 토크 마진 계산
  2. 스테레오 깊이 유효 범위 계산 (ZED Mini 등)
  3. Joint 4 케이블 꼬임 한계 계산
  4. 카메라 후보 비교 리포트 출력

실행:
  python hw_wrist_camera_feasibility.py
  python hw_wrist_camera_feasibility.py --camera zed_mini
  python hw_wrist_camera_feasibility.py --camera c270
  python hw_wrist_camera_feasibility.py --joint4-limit 90  # deg
"""

import argparse
import math

# ──────────────────────────────────────────────
# 하드웨어 상수
# ──────────────────────────────────────────────

# ST3215 서보 (RoArm-M3-Pro 사용)
SERVO_STALL_TORQUE_KG_CM = 15.0   # kg·cm (ST3215 스팩시트)
SERVO_RATED_TORQUE_KG_CM = 10.0   # kg·cm (연속 운용 권장)
SERVO_SAFETY_MARGIN = 0.5          # 50% 이하 연속 운용 권장

# RoArm-M3-Pro 물리 파라미터
WRIST_PLATE_DIAMETER_MM = 50       # 손목 엔드이펙터 플레이트 직경 (추정)
CAMERA_MOUNT_OFFSET_MM = 65        # 손목 중심에서 카메라 중심까지 (마운트 포함)

# Joint 4 (Wrist Roll) 범위
JOINT4_RANGE_DEG = 380             # -190 ~ +190

# 중력
G = 9.81  # m/s^2


# ──────────────────────────────────────────────
# 카메라 스펙 데이터베이스
# ──────────────────────────────────────────────

CAMERAS = {
    "zed_mini": {
        "name": "ZED Mini",
        "mass_g": 62.9,
        "dims_mm": (124.6, 30.6, 26.0),  # L x H x D
        "baseline_mm": 63.0,
        "depth_type": "passive_stereo",
        "min_depth_cm": 10.0,   # 실용적 최소 깊이 (제조사: 10cm)
        "cable": "USB 3.0 (두꺼움, 직경 ~5mm)",
        "bandwidth_mbps": 500,
        "note": "기준선 63mm → 근거리(<10cm) 깊이 무효",
    },
    "c270": {
        "name": "Logitech C270",
        "mass_g": 68.0,
        "dims_mm": (64.0, 40.0, 46.0),
        "baseline_mm": 0,
        "depth_type": "none",
        "min_depth_cm": None,
        "cable": "USB 2.0 (얇음, 직경 ~3.5mm)",
        "bandwidth_mbps": 60,
        "note": "RGB only, 깊이 없음",
    },
    "oak_d_lite": {
        "name": "OAK-D Lite",
        "mass_g": 61.0,
        "dims_mm": (91.0, 28.0, 17.5),
        "baseline_mm": 75.0,
        "depth_type": "passive_stereo",
        "min_depth_cm": 19.6,
        "cable": "USB 3.0",
        "bandwidth_mbps": 500,
        "note": "기준선 75mm, 근거리 깊이 ZED Mini보다 나쁨",
    },
    "realsense_d405": {
        "name": "Intel RealSense D405",
        "mass_g": 61.0,
        "dims_mm": (42.0, 42.0, 23.0),
        "baseline_mm": 18.0,
        "depth_type": "active_stereo_ir",
        "min_depth_cm": 7.0,    # eye-in-hand 특화 모델
        "cable": "USB 3.0",
        "bandwidth_mbps": 500,
        "note": "eye-in-hand 특화. 근거리 깊이 가장 우수. 보유 없음.",
    },
    "esp32_cam": {
        "name": "ESP32-CAM",
        "mass_g": 7.0,
        "dims_mm": (40.0, 27.0, 10.0),
        "baseline_mm": 0,
        "depth_type": "none",
        "min_depth_cm": None,
        "cable": "WiFi 전송 (레이턴시 20~100ms)",
        "bandwidth_mbps": 20,
        "note": "가장 가벼움. WiFi 레이턴시로 closed-loop 제어 불가.",
    },
}


# ──────────────────────────────────────────────
# 분석 함수
# ──────────────────────────────────────────────

def calc_torque(mass_g: float, lever_mm: float) -> dict:
    """손목 서보에 가해지는 정적 토크 계산"""
    mass_kg = mass_g / 1000.0
    lever_m = lever_mm / 1000.0
    torque_nm = mass_kg * G * lever_m
    torque_kg_cm = torque_nm * 100.0 / G  # N·m → kg·cm
    utilization = torque_kg_cm / SERVO_STALL_TORQUE_KG_CM
    return {
        "torque_nm": torque_nm,
        "torque_kg_cm": torque_kg_cm,
        "stall_utilization_pct": utilization * 100,
        "rated_utilization_pct": (torque_kg_cm / SERVO_RATED_TORQUE_KG_CM) * 100,
        "safe": utilization < SERVO_SAFETY_MARGIN,
    }


def calc_stereo_depth_error(baseline_mm: float, focal_px: float = 700,
                             distance_cm: float = 10.0) -> float:
    """스테레오 삼각측량 깊이 오차 (1픽셀 시차 오차 기준)

    σ_depth ≈ Z² / (f × b)
    Z: 거리 (mm), f: focal (px), b: baseline (mm)
    반환값: mm 단위 깊이 오차
    """
    if baseline_mm <= 0:
        return float("inf")
    Z_mm = distance_cm * 10
    # 1픽셀 시차 오차에 대한 깊이 오차
    depth_error_mm = (Z_mm ** 2) / (focal_px * baseline_mm)
    return depth_error_mm


def calc_cable_twist(joint4_range_deg: float) -> dict:
    """Joint 4 회전에 의한 케이블 꼬임 분석"""
    full_rotations = joint4_range_deg / 360.0
    return {
        "joint4_range_deg": joint4_range_deg,
        "full_rotations": full_rotations,
        "cable_twists": full_rotations,
        "verdict": "파단 위험" if full_rotations > 0.75 else "허용 가능",
        "note": f"±{joint4_range_deg/2:.0f}° 범위에서 USB 케이블 {full_rotations:.2f}회 꼬임 발생",
    }


def grasp_clearance_check(camera_dims_mm: tuple) -> dict:
    """손목 카메라가 작업 공간에서 충돌 위험 여부"""
    length, height, depth = camera_dims_mm
    plate_dia = WRIST_PLATE_DIAMETER_MM
    overhang_mm = max(length - plate_dia, 0)
    return {
        "camera_length_mm": length,
        "wrist_plate_mm": plate_dia,
        "overhang_mm": overhang_mm,
        "collision_risk": overhang_mm > 30,
        "verdict": "충돌 위험" if overhang_mm > 30 else "허용 가능",
    }


def analyze_camera(cam_key: str, mount_mass_g: float = 30.0) -> None:
    cam = CAMERAS[cam_key]
    total_mass_g = cam["mass_g"] + mount_mass_g

    print(f"\n{'='*60}")
    print(f"카메라: {cam['name']}")
    print(f"{'='*60}")
    print(f"  무게: {cam['mass_g']}g (마운트 {mount_mass_g}g 포함 → {total_mass_g}g)")
    print(f"  크기: {cam['dims_mm'][0]}×{cam['dims_mm'][1]}×{cam['dims_mm'][2]} mm")
    print(f"  케이블: {cam['cable']}")
    print(f"  깊이 방식: {cam['depth_type']}")
    print(f"  메모: {cam['note']}")

    # 1. 토크 분석
    torque = calc_torque(total_mass_g, CAMERA_MOUNT_OFFSET_MM)
    print(f"\n  [토크 분석]")
    print(f"    정적 토크: {torque['torque_kg_cm']:.2f} kg·cm")
    print(f"    스톨 토크 대비: {torque['stall_utilization_pct']:.1f}%")
    print(f"    정격 토크 대비: {torque['rated_utilization_pct']:.1f}%")
    verdict = "OK (정적)" if torque["safe"] else "WARN: 안전 마진 미달"
    print(f"    판정: {verdict}")
    print(f"    주의: 가속/감속 시 2~3x 동적 하중 → 실질 부하 증가")

    # 2. 케이블 꼬임
    cable_twist = calc_cable_twist(JOINT4_RANGE_DEG)
    print(f"\n  [케이블 꼬임 분석 - Joint 4 전체 범위]")
    print(f"    {cable_twist['note']}")
    print(f"    판정: {cable_twist['verdict']}")

    # 3. 물리적 간섭
    clearance = grasp_clearance_check(cam["dims_mm"])
    print(f"\n  [공간 간섭 분석]")
    print(f"    카메라 돌출: {clearance['overhang_mm']:.0f}mm")
    print(f"    판정: {clearance['verdict']}")

    # 4. 깊이 유효 범위 (스테레오 카메라인 경우)
    if cam["baseline_mm"] > 0:
        print(f"\n  [깊이 유효 범위 분석]")
        print(f"    기준선: {cam['baseline_mm']}mm")
        for dist_cm in [5, 8, 10, 15, 20]:
            err = calc_stereo_depth_error(cam["baseline_mm"], distance_cm=dist_cm)
            flag = " << GRASP ZONE (신뢰 불가)" if dist_cm <= 8 else ""
            print(f"    거리 {dist_cm:3d}cm → 깊이 오차 {err:.1f}mm{flag}")
        print(f"    최소 유효 깊이 (제조사): {cam['min_depth_cm']}cm")
        print(f"    grasp 직전 물체 거리: ~3~8cm → 유효 깊이 범위 밖")

    # 5. 종합 판정
    print(f"\n  [종합 판정]")
    issues = []
    if cable_twist["verdict"] == "파단 위험":
        issues.append("케이블 꼬임 (Joint 4 전체 범위 사용 시)")
    if clearance["collision_risk"]:
        issues.append(f"작업공간 충돌 위험 ({clearance['overhang_mm']:.0f}mm 돌출)")
    if cam["baseline_mm"] > 0 and cam["min_depth_cm"] and cam["min_depth_cm"] > 8:
        issues.append(f"grasp 거리에서 깊이 무효 (최소 {cam['min_depth_cm']}cm)")
    if "WiFi" in cam["cable"]:
        issues.append("WiFi 레이턴시 (closed-loop 불가)")

    if not issues:
        print("    권장 가능 (조건부)")
    else:
        print(f"    권장하지 않음 — 문제 {len(issues)}개:")
        for i, issue in enumerate(issues, 1):
            print(f"      {i}. {issue}")


def joint4_limited_analysis(limit_deg: float) -> None:
    """Joint 4 범위를 제한한 경우의 케이블 꼬임 재분석"""
    print(f"\n{'='*60}")
    print(f"Joint 4 소프트웨어 제한 분석: ±{limit_deg}°")
    print(f"{'='*60}")
    limited_range = limit_deg * 2
    twist = calc_cable_twist(limited_range)
    print(f"  제한 범위: {limited_range}° (원래 {JOINT4_RANGE_DEG}°의 {limited_range/JOINT4_RANGE_DEG*100:.0f}%)")
    print(f"  케이블 꼬임: {twist['full_rotations']:.2f}회")
    print(f"  판정: {twist['verdict']}")
    if twist["verdict"] == "허용 가능":
        lost_pct = (1 - limited_range / JOINT4_RANGE_DEG) * 100
        print(f"  주의: 작업 공간 {lost_pct:.0f}% 축소 → VLA 학습 다양성 감소")


def summary_comparison() -> None:
    print(f"\n{'='*60}")
    print("카메라 후보 종합 비교")
    print(f"{'='*60}")
    print(f"{'카메라':<20} {'무게':>6} {'최소깊이':>8} {'돌출':>6} {'케이블':>12} {'eye-in-hand 적합도'}")
    print("-" * 80)

    ratings = {
        "zed_mini": "낮음",
        "c270": "조건부",
        "oak_d_lite": "낮음",
        "realsense_d405": "높음 (미보유)",
        "esp32_cam": "낮음 (레이턴시)",
    }

    for key, cam in CAMERAS.items():
        min_d = f"{cam['min_depth_cm']}cm" if cam["min_depth_cm"] else "없음"
        overhang = max(cam["dims_mm"][0] - WRIST_PLATE_DIAMETER_MM, 0)
        print(f"  {cam['name']:<18} {cam['mass_g']:>5}g {min_d:>8} {overhang:>5}mm {cam['cable'][:12]:>12}  {ratings[key]}")

    print(f"\n현재 보유 장비 기준 권장: Azure Kinect 2대 멀티뷰 (외부 시점)")
    print("  이유: 케이블 없음, 깊이 유효, 기존 74ep 데이터 부분 재활용 가능")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RoArm-M3 손목 카메라 장착 타당성 분석")
    parser.add_argument("--camera", choices=list(CAMERAS.keys()), default=None,
                        help="특정 카메라 분석 (기본: 전체)")
    parser.add_argument("--mount-mass", type=float, default=30.0,
                        help="카메라 마운트 질량 (g, 기본: 30g)")
    parser.add_argument("--joint4-limit", type=float, default=None,
                        help="Joint 4 소프트웨어 제한 각도 (deg, 예: 90)")
    args = parser.parse_args()

    print("RoArm-M3-Pro 손목 카메라 장착 타당성 분석")
    print(f"  손목 서보 스톨 토크: {SERVO_STALL_TORQUE_KG_CM} kg·cm (ST3215)")
    print(f"  손목 플레이트 직경: {WRIST_PLATE_DIAMETER_MM}mm")
    print(f"  카메라 레버 암: {CAMERA_MOUNT_OFFSET_MM}mm")
    print(f"  Joint 4 전체 범위: {JOINT4_RANGE_DEG}°")

    if args.camera:
        analyze_camera(args.camera, mount_mass_g=args.mount_mass)
    else:
        for key in CAMERAS:
            analyze_camera(key, mount_mass_g=args.mount_mass)

    if args.joint4_limit is not None:
        joint4_limited_analysis(args.joint4_limit)

    summary_comparison()


if __name__ == "__main__":
    main()
