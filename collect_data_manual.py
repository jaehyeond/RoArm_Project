"""
RoArm M3 + Azure Kinect 수동 데이터 수집 스크립트
토크 OFF 상태에서 손으로 로봇을 직접 움직여서 데이터 수집

듀얼 카메라 지원:
  --second-camera zed_wrist      ZED Mini wrist 카메라 (pyzed, RGB only)
  --second-camera kinect_external Azure Kinect 2대째 외부 시점 (pyk4a)

사용법:
1. 스크립트 실행하면 로봇 토크가 꺼짐 (손으로 자유롭게 이동 가능)
2. 손으로 로봇을 움직여서 물체 집기 동작 수행
3. Space 누르면 녹화 시작/중지
4. Enter 누르면 에피소드 저장

조작법:
  Space: 녹화 시작/중지 (토글)
  Enter: 에피소드 저장
  Backspace: 현재 에피소드 취소
  T: 토크 ON/OFF 토글
  I: 초기 위치로 이동 (토크 ON 필요)
  ESC: 종료
"""

import os
import json
import time
import datetime
import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A
from pynput import keyboard
from roarm_sdk.roarm import roarm
import logging

# SDK 로그 억제
logging.getLogger('BaseController').setLevel(logging.CRITICAL)


# 책상 표면 z 하한 (2026-03-26 실측: 책상 z ≈ -95 ~ -121mm)
Z_DESK_SURFACE = -120   # 책상 표면 FK z (mm)
Z_FLOOR_DEPLOY = -90    # 배포 시 안전 하한 (스펀지 상면)
Z_FLOOR_WARNING = -110  # 수집 시 경고 (책상 접근)


def classify_zone(base_angle, fk_dist, fk_z):
    """FK 기반 5-zone 분류 (2026-03-26 실측 기반)

    실측 FK 범위:
      dist: 220~470mm,  base: -75~+69°,  z: -121~+500mm
      책상 표면 z ≈ -120mm

    Args:
        base_angle: base joint angle (degrees)
        fk_dist: XY 평면 거리 sqrt(x^2+y^2) (mm)
        fk_z: end-effector Z height (mm)

    Returns:
        zone name: NEAR, MID_LEFT, MID_RIGHT, FAR_CENTER, OVERHEAD, UNKNOWN
    """
    if fk_z is not None and fk_z > 0:
        return "OVERHEAD"
    if fk_dist is None:
        return "UNKNOWN"
    # 1차 분류: base angle로 좌/우
    if base_angle < -30:
        return "MID_LEFT"
    if base_angle > 30:
        return "MID_RIGHT"
    # 2차 분류: dist로 가까이/멀리 (base ±30° 이내)
    if fk_dist < 320:
        return "NEAR"
    if fk_dist > 380:
        return "FAR_CENTER"
    # 경계 영역 (320~380mm, base ±30°)
    if base_angle < -15:
        return "MID_LEFT"
    elif base_angle > 15:
        return "MID_RIGHT"
    else:
        return "FAR_CENTER"


ZONE_TARGETS = {
    "NEAR": 30, "MID_LEFT": 25, "MID_RIGHT": 25,
    "FAR_CENTER": 35, "OVERHEAD": 15,
}
ZONE_COLORS = {
    "NEAR": (255, 200, 0),      # cyan-ish
    "MID_LEFT": (255, 100, 100), # blue-ish
    "MID_RIGHT": (100, 100, 255), # red-ish
    "FAR_CENTER": (0, 255, 0),   # green
    "OVERHEAD": (0, 200, 255),   # yellow-orange
    "UNKNOWN": (128, 128, 128),
}


class DatasetStats:
    """기존 데이터셋 통계 분석 클래스 (zone + depth 기반)"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.deep_count = 0
        self.approach_count = 0
        self.shallow_count = 0
        self.total_count = 0
        # Zone 카운터
        self.zone_counts = {z: 0 for z in ZONE_TARGETS}
        self.analyze_existing_episodes()

    def analyze_existing_episodes(self):
        """기존 에피소드 분석 (metadata.json 기반)"""
        if not os.path.exists(self.save_dir):
            return

        for episode_dir in sorted(os.listdir(self.save_dir)):
            if not episode_dir.startswith("episode_"):
                continue

            metadata_path = os.path.join(self.save_dir, episode_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                continue

            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                    # 분류 우선순위: shoulder_at_grip_close > min_z > max_shoulder
                    sh_at_close = metadata.get('shoulder_at_grip_close', None)
                    min_z = metadata.get('min_z', None)
                    max_sh = metadata.get('max_shoulder', None)

                    if sh_at_close is not None:
                        # 가장 정확: 그리퍼 닫기 시점의 shoulder
                        if sh_at_close > 60:
                            self.deep_count += 1
                        elif sh_at_close > 40:
                            self.approach_count += 1
                        else:
                            self.shallow_count += 1
                    elif min_z is not None and min_z < 9999:
                        # Z-height 기반 (calibrated: Z=30mm=table, Z=80mm=grasp, Z=160mm=approach)
                        if min_z < 80:
                            self.deep_count += 1
                        elif min_z < 160:
                            self.approach_count += 1
                        else:
                            self.shallow_count += 1
                    elif max_sh is not None:
                        # Shoulder 최대값 폴백
                        if max_sh > 60:
                            self.deep_count += 1
                        elif max_sh > 40:
                            self.approach_count += 1
                        else:
                            self.shallow_count += 1
                    else:
                        # 구 에피소드: 데이터 없으면 APPROACH로 추정
                        self.approach_count += 1

                    self.total_count += 1

                    # Zone 카운팅 (metadata에 zone 필드가 있으면 사용)
                    zone = metadata.get("zone", None)
                    if zone and zone in self.zone_counts:
                        self.zone_counts[zone] += 1
            except Exception:
                pass

    def get_recommendation(self):
        """다음 추천 수집 zone 반환 (가장 부족한 zone)"""
        # Zone 기반 추천 (quota 대비 가장 부족한 zone)
        min_ratio = 999
        rec_zone = "NEAR"
        for zone, target in ZONE_TARGETS.items():
            ratio = self.zone_counts.get(zone, 0) / max(1, target)
            if ratio < min_ratio:
                min_ratio = ratio
                rec_zone = zone
        return f"{rec_zone} ({self.zone_counts.get(rec_zone, 0)}/{ZONE_TARGETS[rec_zone]})", ZONE_COLORS.get(rec_zone, (255, 255, 255))

    def get_progress_str(self):
        """진행률 문자열 생성 (5-zone 기반, 목표: 150 에피소드)"""
        target_total = sum(ZONE_TARGETS.values())
        lines = []
        for zone, target in ZONE_TARGETS.items():
            count = self.zone_counts.get(zone, 0)
            pct = count / max(1, target) * 100
            bar = "█" * min(int(pct / 10), 10)
            lines.append(f"{zone:12s}: {count:2d}/{target} {bar}")
        lines.append(f"{'TOTAL':12s}: {self.total_count}/{target_total}")
        return lines


class ManualDataCollector:
    def __init__(self, robot_port="/dev/ttyUSB0", save_dir="collected_data", object_name="sponge",
                 second_camera="none"):
        self.save_dir = save_dir
        self.object_name = object_name
        self.second_cam_type = second_camera  # none / zed_wrist / kinect_external
        os.makedirs(save_dir, exist_ok=True)

        # 데이터셋 통계 초기화
        self.stats = DatasetStats(save_dir)

        # 로봇 연결
        print(f"로봇 연결 중... ({robot_port})")
        self.robot = roarm(roarm_type="roarm_m3", port=robot_port, baudrate=115200)
        time.sleep(0.5)
        print("로봇 연결됨!")

        # Azure Kinect 초기화 (primary)
        print("Azure Kinect 초기화 중...")
        self.k4a = PyK4A(Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ))
        self.k4a.start()
        time.sleep(1)
        print("Azure Kinect 시작됨!")

        # 두 번째 카메라 초기화 (optional)
        self.zed = None
        self.k4a2 = None
        if second_camera == "zed_wrist":
            self._init_zed_mini()
        elif second_camera == "kinect_external":
            self._init_kinect_second()

        # 데이터 수집 상태
        self.current_episode = []
        self.episode_count = len([d for d in os.listdir(save_dir) if d.startswith("episode_")])
        self.is_recording = False
        self.torque_on = True

        # 실행 상태
        self.running = True

        # 녹화 FPS 설정
        self.record_fps = 30
        self.last_record_time = 0

        # Elbow + Gripper 추적 (에피소드별)
        self.min_elbow = 999
        self.max_elbow = -999
        self.min_gripper = 999
        self.max_gripper = -999

        # Z-height 추적 (에피소드별, FK 기반)
        self.min_z = 9999
        self.max_z = -9999
        self.current_pose = None  # 캐시 for display

        # Shoulder + Gripper 타이밍 추적 (에피소드별)
        self.max_shoulder = -999
        self.shoulder_at_grip_close = None  # 그리퍼 닫히는 시점의 shoulder
        self.z_at_grip_close = None         # 그리퍼 닫히는 시점의 Z
        self.grip_was_open = False          # 에피소드 중 그리퍼가 충분히 열렸는지
        self.grip_open_frame = None         # 그리퍼 처음 열린 프레임
        self.grip_close_frame = None        # 그리퍼 닫힌 프레임
        self.prev_gripper = None            # 이전 프레임 그리퍼 값

    def _init_zed_mini(self):
        """ZED Mini wrist 카메라 초기화 (RGB only, depth 비활성, MCU 재시도)"""
        print("ZED Mini 초기화 중...")
        import pyzed.sl as sl
        self._sl = sl  # 모듈 참조 보관
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.sensors_required = False  # MCU 에러 방지

        # MCU 간헐 에러 대응: 최대 3회 재시도
        status = None
        for attempt in range(3):
            status = self.zed.open(init_params)
            if status == sl.ERROR_CODE.SUCCESS:
                break
            print(f"ZED Mini open 시도 {attempt + 1}/3: {status}")
            self.zed.close()
            time.sleep(2)

        if status != sl.ERROR_CODE.SUCCESS:
            print(f"ZED Mini open 실패: {status}")
            print("듀얼 카메라 없이 단일 카메라로 계속합니다.")
            self.zed = None
            self.second_cam_type = "none"
            return

        # 수동 노출 설정 (검은 arm이 auto-exposure를 교란하므로)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)

        self.zed_image = sl.Mat()  # 재사용 버퍼
        # 노출 안정화 대기
        for _ in range(15):
            self.zed.grab()
        print("ZED Mini 시작됨! (depth=NONE, 수동 노출=50)")

    def _init_kinect_second(self):
        """Azure Kinect 2대째 초기화 (외부 시점)"""
        print("Azure Kinect 2대째 초기화 중...")
        self.k4a2 = PyK4A(Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ), device_id=1)
        try:
            self.k4a2.start()
            time.sleep(1)
            print("Azure Kinect 2대째 시작됨!")
        except Exception as e:
            print(f"Azure Kinect 2대째 open 실패: {e}")
            print("듀얼 카메라 없이 단일 카메라로 계속합니다.")
            self.k4a2 = None
            self.second_cam_type = "none"

    def get_camera_frame(self):
        """카메라 프레임 가져오기 (primary + optional second)

        동기화: SW polling — 동일 루프 내 순차 캡처.
        30fps에서 양쪽 합산 ~15ms, <30ms tolerance (DROID/pi0/Octo 표준).
        """
        # Primary: Azure Kinect
        capture = self.k4a.get_capture()
        rgb = np.ascontiguousarray(capture.color[:, :, :3])  # BGRA -> BGR
        depth = capture.transformed_depth  # RGB에 정렬된 깊이

        # Second camera (순차 캡처)
        second_rgb = None
        if self.second_cam_type == "zed_wrist" and self.zed is not None:
            sl = self._sl
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.zed_image, sl.VIEW.LEFT)
                # ZED returns BGRA
                second_rgb = np.ascontiguousarray(self.zed_image.get_data()[:, :, :3])
        elif self.second_cam_type == "kinect_external" and self.k4a2 is not None:
            try:
                capture2 = self.k4a2.get_capture()
                second_rgb = np.ascontiguousarray(capture2.color[:, :, :3])
            except Exception:
                pass  # 프레임 드롭 허용, None 유지

        return rgb, depth, second_rgb

    def get_robot_angles(self):
        """로봇 관절 각도 읽기 (재시도 로직 포함)"""
        for _ in range(5):
            try:
                angles = self.robot.joints_angle_get()
                if angles is not None and len(angles) >= 6:
                    return list(angles)
            except Exception:
                time.sleep(0.05)
        return [0, 0, 0, 0, 0, 0]  # 실패시 기본값

    def get_robot_pose(self):
        """로봇 엔드이펙터 위치 읽기 (FK, 재시도 로직 포함)"""
        for _ in range(5):
            try:
                pose = self.robot.pose_get()
                if pose is not None and len(pose) >= 3:
                    return pose  # [x, y, z, tilt_deg, roll_deg, gripper_deg]
            except Exception:
                time.sleep(0.05)
        return None

    def set_torque(self, on: bool):
        """토크 ON/OFF 설정"""
        self.robot.torque_set(cmd=1 if on else 0)
        self.torque_on = on
        time.sleep(0.3)
        status = "ON" if on else "OFF"
        print(f"\n토크 {status}!")
        if not on:
            print("→ 이제 손으로 로봇을 자유롭게 움직일 수 있습니다.")

    def save_frame(self, rgb, depth, angles, pose, second_rgb=None):
        """현재 프레임을 에피소드에 추가"""
        frame_data = {
            "timestamp": time.time(),
            "angles": angles.copy(),
            "pose": pose[:3] if pose else None,  # [x_mm, y_mm, z_mm]
            "frame_idx": len(self.current_episode)
        }

        entry = {
            "data": frame_data,
            "rgb": rgb.copy(),
            "depth": depth.copy()
        }
        if second_rgb is not None:
            entry["second_rgb"] = second_rgb.copy()
        self.current_episode.append(entry)

    def validate_episode(self):
        """에피소드 품질 검증 (shoulder + Z + gripper timing)

        Z Calibration (confirmed by user, 2026-02-23):
          Z=30mm  = arm fully extended to table surface (DEEP limit)
          Z=80mm  = typical gripper-close height on object (~30mm tall box)
          Z=160mm = approach height (arm moving toward object)
          Z=230mm+ = home/neutral height (arm at rest)

        Episode pattern: orange → yellow → green (grasp moment) → yellow → orange
        Only the GRASPING MOMENT needs to be in the green zone.
        """
        issues = []
        warnings = []

        num_frames = len(self.current_episode)

        # 1. 그리퍼가 충분히 열렸는가? (40° 이상)
        if not self.grip_was_open:
            issues.append(f"그리퍼 미개방 (max={self.max_gripper:.0f}° < 40°)")
        else:
            # 2. 그리퍼 열림 크기 체크 (열린 경우에만 range 평가)
            gripper_range = self.max_gripper - self.min_gripper
            if gripper_range < 15:
                issues.append(f"그리퍼 개폐 부족 (range={gripper_range:.1f}° < 15°)")
            elif self.max_gripper < 50:
                warnings.append(f"그리퍼 열림 부족 (max={self.max_gripper:.0f}°, 60°+ 권장)")

        # 3. 그리퍼 닫기 타이밍 — 핵심 체크!
        if self.shoulder_at_grip_close is not None:
            # Shoulder 기준: 닫힐 때 shoulder > 50° = 팔이 충분히 내려간 상태
            if self.shoulder_at_grip_close < 40:
                issues.append(
                    f"그리퍼 닫기 시 팔 높음! (shoulder={self.shoulder_at_grip_close:.0f}°, 50°+ 필요)")
            elif self.shoulder_at_grip_close < 50:
                warnings.append(
                    f"그리퍼 닫기 시 팔 약간 높음 (shoulder={self.shoulder_at_grip_close:.0f}°)")

            # Z 기준: 닫힐 때 Z < 130mm = 물체 근처에서 잡기
            # 근거: Z=30mm=테이블, Z=80mm=일반 물체 위, Z=130mm=안전 상한
            # 에피소드 전체가 green일 필요 없음 — 잡기 순간만 green 필요!
            if self.z_at_grip_close is not None and self.z_at_grip_close > 130:
                issues.append(
                    f"그리퍼 닫기 시 높이 높음! (Z={self.z_at_grip_close:.0f}mm, 130mm 이하 필요)")
        else:
            # shoulder_at_grip_close가 None인 경우:
            # - 열렸는데 닫기를 감지 못함 (열린 채 종료, 또는 임계값 미달)
            # - 아예 안 열린 경우는 check 1에서 이미 처리됨
            if self.grip_was_open:
                warnings.append("그리퍼가 열렸지만 닫기 감지 안됨 (열린 상태로 끝남?)")

        # 4. 프레임 수 체크 (SmolVLA 공식: 평균 393프레임/13초, 최소 150프레임/5초)
        if num_frames < 90:
            issues.append(f"프레임 수 부족 ({num_frames} < 90, 최소 3초)")
        elif num_frames < 150:
            warnings.append(f"에피소드 짧음 ({num_frames}프레임 = {num_frames/30:.1f}초, 10초+ 권장)")
        elif num_frames > 600:
            warnings.append(f"에피소드 너무 김 ({num_frames}프레임 = {num_frames/30:.1f}초, 15초 이하 권장)")

        # 5. Z-height 체크 (DEEP grasp 기준)
        # min_z는 에피소드 전체에서 가장 낮은 값 — 160mm 이하면 충분히 내려간 것
        if self.min_z > 160:
            warnings.append(f"얕은 그리핑 (min_z={self.min_z:.0f}mm, 160mm 이하 권장)")

        return issues, warnings

    def save_episode(self):
        """현재 에피소드를 디스크에 저장"""
        if len(self.current_episode) == 0:
            print("저장할 프레임이 없습니다!")
            return

        # 품질 검증
        issues, warnings = self.validate_episode()
        if issues or warnings:
            print(f"\n{'='*50}")
            if issues:
                print("FAIL - 에피소드 품질 문제:")
                for i, issue in enumerate(issues, 1):
                    print(f"  [{i}] {issue}")
            if warnings:
                print("WARN - 개선 권장:")
                for i, warn in enumerate(warnings, 1):
                    print(f"  ({i}) {warn}")

            # 그리퍼 타이밍 정보 출력
            if self.shoulder_at_grip_close is not None:
                z_part = f", Z={self.z_at_grip_close:.0f}mm" if self.z_at_grip_close is not None else ""
                print(f"\n  그리퍼 닫기 시점: shoulder={self.shoulder_at_grip_close:.0f}°{z_part}")
                if self.grip_open_frame is not None and self.grip_close_frame is not None:
                    open_pct = self.grip_open_frame / max(1, len(self.current_episode)) * 100
                    close_pct = self.grip_close_frame / max(1, len(self.current_episode)) * 100
                    print(f"  그리퍼 열림: {open_pct:.0f}% 지점, 닫힘: {close_pct:.0f}% 지점")
            print(f"{'='*50}")

            if issues:
                # issues가 있으면 기본적으로 재녹화 권유
                choice = input("저장하시겠습니까? (y=강제저장, n=취소, r=재녹화): ").strip().lower()
                if choice == 'n':
                    print("에피소드 저장 취소됨")
                    return
                elif choice != 'y':
                    print("에피소드 취소 후 재녹화하세요 (Backspace)")
                    return
            # warnings만 있으면 자동 저장 진행

        episode_dir = os.path.join(self.save_dir, f"episode_{self.episode_count:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        # Zone 판정 (그리퍼 닫기 시점 또는 min_z 시점의 위치 기반)
        ep_zone = "UNKNOWN"
        if self.current_episode:
            # 그리퍼 닫기 시점 프레임의 base angle + FK distance로 판정
            grasp_frame_idx = self.grip_close_frame if self.grip_close_frame is not None else len(self.current_episode) // 2
            grasp_frame_idx = min(grasp_frame_idx, len(self.current_episode) - 1)
            gf = self.current_episode[grasp_frame_idx]
            gf_base = gf["data"].get("angles", [0])[0] if "angles" in gf["data"] else 0
            gf_pose = gf["data"].get("pose", None)
            gf_dist = None
            gf_z = None
            if gf_pose and len(gf_pose) >= 3:
                gf_dist = (gf_pose[0]**2 + gf_pose[1]**2)**0.5
                gf_z = gf_pose[2]
            ep_zone = classify_zone(gf_base, gf_dist, gf_z)

        # 메타데이터 저장
        gripper_range = self.max_gripper - self.min_gripper
        z_range = self.max_z - self.min_z
        metadata = {
            "episode_id": self.episode_count,
            "object": self.object_name,
            "zone": ep_zone,
            "second_camera": self.second_cam_type,
            "num_frames": len(self.current_episode),
            "timestamp": datetime.datetime.now().isoformat(),
            "fps": self.record_fps,
            "min_elbow": round(self.min_elbow, 2),
            "max_elbow": round(self.max_elbow, 2),
            "elbow_range": round(self.max_elbow - self.min_elbow, 2),
            "max_shoulder": round(self.max_shoulder, 2),
            "shoulder_at_grip_close": round(self.shoulder_at_grip_close, 2) if self.shoulder_at_grip_close is not None else None,
            "z_at_grip_close": round(self.z_at_grip_close, 2) if self.z_at_grip_close is not None else None,
            "min_z": round(self.min_z, 2),
            "max_z": round(self.max_z, 2),
            "z_range": round(z_range, 2),
            "gripper_min": round(self.min_gripper, 2),
            "gripper_max": round(self.max_gripper, 2),
            "gripper_range": round(gripper_range, 2),
            "grip_open_frame": self.grip_open_frame,
            "grip_close_frame": self.grip_close_frame,
            "frames": []
        }

        # 각 프레임 저장
        for i, frame in enumerate(self.current_episode):
            rgb_path = os.path.join(episode_dir, f"rgb_{i:04d}.jpg")
            depth_path = os.path.join(episode_dir, f"depth_{i:04d}.npy")

            cv2.imwrite(rgb_path, frame["rgb"])
            np.save(depth_path, frame["depth"])

            frame_info = frame["data"].copy()
            frame_info["rgb_path"] = f"rgb_{i:04d}.jpg"
            frame_info["depth_path"] = f"depth_{i:04d}.npy"

            if "second_rgb" in frame:
                second_path = os.path.join(episode_dir, f"second_{i:04d}.jpg")
                cv2.imwrite(second_path, frame["second_rgb"])
                frame_info["second_path"] = f"second_{i:04d}.jpg"
            metadata["frames"].append(frame_info)

        with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Z-height 품질 판정 (min_z = 에피소드에서 가장 낮은 값)
        # Calibrated: Z=30mm=table, Z=80mm=object grasp, Z=160mm=approach, Z=230mm+=home
        if self.min_z < 80:
            quality = "DEEP GRASP"
            quality_color = "DEEP"
        elif self.min_z < 160:
            quality = "APPROACH"
            quality_color = "APPROACH"
        else:
            quality = "SHALLOW"
            quality_color = "SHALLOW"

        print(f"\n{'='*50}")
        print(f"에피소드 {self.episode_count} 저장 완료! [Zone: {ep_zone}]")
        print(f"  프레임: {len(self.current_episode)} ({len(self.current_episode)/30:.1f}초)")
        print(f"  Zone: {ep_zone} | Min Z: {self.min_z:.0f}mm → [{quality}]")
        print(f"  Max Shoulder: {self.max_shoulder:.0f}°")
        print(f"  Gripper: {self.min_gripper:.0f}° ~ {self.max_gripper:.0f}° (range={gripper_range:.0f}°)")
        if self.shoulder_at_grip_close is not None:
            z_str = f", Z={self.z_at_grip_close:.0f}mm" if self.z_at_grip_close else ""
            timing_ok = "OK" if self.shoulder_at_grip_close >= 50 else "LOW"
            print(f"  Grasp: shoulder={self.shoulder_at_grip_close:.0f}°{z_str} [{timing_ok}]")
        else:
            print(f"  Grasp: 닫기 감지 안됨")
        print(f"  저장: {episode_dir}")
        print(f"{'='*50}\n")

        # 통계 업데이트
        self.stats.analyze_existing_episodes()

        self.episode_count += 1
        self._reset_episode_tracking()

    def _reset_episode_tracking(self):
        """에피소드 추적 변수 전체 초기화"""
        self.current_episode = []
        self.is_recording = False
        self.min_elbow = 999
        self.max_elbow = -999
        self.min_gripper = 999
        self.max_gripper = -999
        self.min_z = 9999
        self.max_z = -9999
        self.max_shoulder = -999
        self.shoulder_at_grip_close = None
        self.z_at_grip_close = None
        self.grip_was_open = False
        self.grip_open_frame = None
        self.grip_close_frame = None
        self.prev_gripper = None

    def cancel_episode(self):
        """현재 에피소드 취소"""
        if len(self.current_episode) > 0:
            print(f"\n에피소드 취소됨 ({len(self.current_episode)} 프레임 삭제)")
            self._reset_episode_tracking()
        else:
            print("취소할 에피소드가 없습니다.")

    def on_key_press(self, key):
        """키 눌림 이벤트"""
        # ESC로 종료
        if key == keyboard.Key.esc:
            self.running = False
            return False

        # Space로 녹화 시작/중지 토글
        if key == keyboard.Key.space:
            self.is_recording = not self.is_recording
            if self.is_recording:
                print("\n[REC] 녹화 시작! 물체를 집어보세요...")
            else:
                print(f"\n[STOP] 녹화 중지 ({len(self.current_episode)} 프레임)")

        # Enter로 에피소드 저장
        if key == keyboard.Key.enter:
            if self.is_recording:
                self.is_recording = False
            self.save_episode()

        # Backspace로 에피소드 취소
        if key == keyboard.Key.backspace:
            self.cancel_episode()

        # 문자 키 처리
        try:
            k = key.char.lower() if hasattr(key, 'char') and key.char else None
        except Exception:
            k = None

        if k == 't':  # 토크 토글
            self.set_torque(not self.torque_on)

        if k == 'i':  # 초기 위치로 이동
            if not self.torque_on:
                print("\n초기 위치 이동을 위해 토크를 켭니다...")
                self.set_torque(True)
            print("초기 위치로 이동 중...")
            self.robot.move_init()
            time.sleep(2)
            print("초기 위치 도착!")

    def run(self):
        """메인 루프"""
        print("\n" + "="*60)
        print("RoArm M3 수동 데이터 수집 (토크 OFF 모드)")
        print("="*60)
        print("\n조작법:")
        print("  Space: 녹화 시작/중지 (토글)")
        print("  Enter: 에피소드 저장")
        print("  Backspace: 에피소드 취소")
        print("  T: 토크 ON/OFF 토글")
        print("  I: 초기 위치로 이동")
        print("  ESC: 종료")
        print("="*60)

        # 초기 위치로 이동
        print("\n초기 위치로 이동 중...")
        self.robot.move_init()
        time.sleep(2)

        # 토크 OFF로 시작
        print("\n토크를 끕니다...")
        self.set_torque(False)

        # 키보드 리스너 시작
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.start()

        # OpenCV 창 생성
        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera View", 960, 540)
        if self.second_cam_type != "none":
            cv2.namedWindow("Second Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Second Camera", 480, 270)

        print("\n준비 완료! Space를 눌러 녹화를 시작하세요.\n")

        try:
            while self.running:
                current_time = time.time()

                # 카메라 프레임 가져오기
                rgb, depth, second_rgb = self.get_camera_frame()
                angles = self.get_robot_angles()
                pose = self.get_robot_pose()
                self.current_pose = pose  # 캐시 for display

                # 관절 + Z 추적 (save_frame 호출 전에 수행해야 frame_idx가 정확함)
                shoulder = angles[1]
                elbow = angles[2]
                gripper = angles[5]
                z_height = pose[2] if pose else 9999  # Z in mm

                # 녹화 중이면 프레임 저장 (30 FPS) + 통계/타이밍 추적
                if self.is_recording:
                    if current_time - self.last_record_time >= 1.0 / self.record_fps:
                        # frame_idx는 저장되는 프레임에서만 기록 (FPS 제한 블록 안)
                        frame_idx = len(self.current_episode)
                        self.save_frame(rgb, depth, angles, pose, second_rgb)
                        self.last_record_time = current_time

                        # 통계 추적 (저장된 프레임에서만)
                        self.min_elbow = min(self.min_elbow, elbow)
                        self.max_elbow = max(self.max_elbow, elbow)
                        self.min_gripper = min(self.min_gripper, gripper)
                        self.max_gripper = max(self.max_gripper, gripper)
                        self.max_shoulder = max(self.max_shoulder, shoulder)
                        if pose:
                            self.min_z = min(self.min_z, z_height)
                            self.max_z = max(self.max_z, z_height)

                        # 그리퍼 열림/닫힘 감지 (저장된 프레임 인덱스 사용)
                        if gripper > 40 and not self.grip_was_open:
                            self.grip_was_open = True
                            self.grip_open_frame = frame_idx
                        if (self.grip_was_open and
                                gripper < self.max_gripper * 0.5 and
                                self.shoulder_at_grip_close is None):
                            self.shoulder_at_grip_close = shoulder
                            self.z_at_grip_close = z_height if pose else None
                            self.grip_close_frame = frame_idx
                        self.prev_gripper = gripper

                # Z-height 존 판정 + 컬러
                # Calibrated: Z=30mm=table touch, Z=80mm=object grasp, Z=160mm=approach, Z=230mm+=home
                # NOTE: entire episode does NOT need to be green — only the grasp-close moment needs green
                # Z 깊이 판별 (2026-03-26 실측 기반)
                # 책상 z≈-120, 스펀지 상면≈-90, 접근≈-40, 높이≈+50
                if z_height < -80:
                    z_zone = "DEEP"
                    z_color = (0, 255, 0)      # 초록 (잡기 위치, 책상 근처)
                elif z_height < -20:
                    z_zone = "APPROACH"
                    z_color = (0, 255, 255)    # 노랑 (접근 중)
                else:
                    z_zone = "SHALLOW"
                    z_color = (0, 100, 255)    # 주황 (홈/높은 위치)

                # Spatial zone 판정 (5-zone)
                base_angle = angles[0]
                fk_dist = (pose[0]**2 + pose[1]**2)**0.5 if pose else None
                spatial_zone = classify_zone(base_angle, fk_dist, z_height if z_height < 9999 else None)
                spatial_color = ZONE_COLORS.get(spatial_zone, (128, 128, 128))

                # 화면에 정보 표시
                display = rgb.copy()
                torque_status = "ON" if self.torque_on else "OFF"
                rec_status = "RECORDING" if self.is_recording else "STANDBY"
                status_color = (0, 0, 255) if self.is_recording else (0, 255, 0)

                # 상단: 물체명 + 에피소드 번호 + 프레임 수
                y = 30
                cv2.putText(display, f"[{self.object_name.upper()}] Ep {self.episode_count} | Frames {len(self.current_episode)}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                y += 30

                cv2.putText(display, f"Torque {torque_status} | {rec_status}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                y += 35

                # Spatial Zone 크게 표시 (우상단)
                cv2.putText(display, f"ZONE: {spatial_zone}", (display.shape[1] - 280, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, spatial_color, 2)
                if fk_dist is not None:
                    cv2.putText(display, f"Dist:{fk_dist:.0f}mm Base:{base_angle:+.0f}", (display.shape[1] - 280, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, spatial_color, 1)

                # Z-Height + Shoulder 크게 표시
                cv2.putText(display, f"Z: {z_height:.0f}mm", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, z_color, 3)
                cv2.putText(display, f"[{z_zone}]", (220, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, z_color, 2)
                # Shoulder 표시 (그리핑 깊이의 실제 지표)
                sh_color = (0, 255, 0) if shoulder > 60 else (0, 255, 255) if shoulder > 40 else (0, 100, 255)
                cv2.putText(display, f"Sh: {shoulder:.0f}deg", (380, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, sh_color, 2)
                y += 35

                # 그리퍼 상태 크게 표시
                grip_color = (0, 255, 0) if gripper > 40 else (0, 200, 255) if gripper > 15 else (100, 100, 255)
                grip_label = "OPEN" if gripper > 40 else "PARTIAL" if gripper > 15 else "CLOSED"
                cv2.putText(display, f"Grip: {gripper:.0f}deg [{grip_label}]", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, grip_color, 2)
                y += 30

                # Wrist Pitch 표시 (top-down 접근 가이드)
                wp = angles[3]
                if z_zone == "DEEP" and abs(wp) < 20:
                    wp_color = (0, 0, 255)  # 빨강: 책상 근처인데 wrist 안 꺾임
                    wp_warn = " !FLAT"
                else:
                    wp_color = (180, 180, 180)
                    wp_warn = ""
                cv2.putText(display, f"WristP:{wp:+.0f}{wp_warn}", (380, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, wp_color, 2)

                # Z 경고 (책상 접근)
                if z_height < Z_FLOOR_WARNING:
                    cv2.putText(display, "!! DESK CLOSE !!", (display.shape[1]//2 - 100, display.shape[0] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # 위치 정보 (작게): X, Y 좌표
                if pose:
                    cv2.putText(display, f"X:{pose[0]:.0f}  Y:{pose[1]:.0f}  Elbow:{elbow:+.0f}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                    y += 22

                # 녹화 중이면 에피소드 통계 + 타이밍 표시
                if self.is_recording:
                    ep_secs = len(self.current_episode) / 30.0
                    dur_color = (0, 255, 0) if ep_secs < 8 else (0, 255, 255) if ep_secs < 10 else (0, 0, 255)
                    cv2.putText(display, f"Time: {ep_secs:.1f}s | Frames: {len(self.current_episode)}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dur_color, 2)
                    y += 25

                    if self.min_z < 9999:
                        gripper_range_now = self.max_gripper - self.min_gripper
                        cv2.putText(display, f"MinZ:{self.min_z:.0f}mm MaxSh:{self.max_shoulder:.0f}deg GripR:{gripper_range_now:.0f}deg",
                                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, z_color, 1)
                        y += 22

                    # 그리퍼 타이밍 피드백
                    if self.grip_was_open and self.shoulder_at_grip_close is None:
                        # 그리퍼 열린 상태 — 좋음!
                        cv2.putText(display, "Gripper OPEN - move down then close!", (10, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                        y += 22
                    elif self.shoulder_at_grip_close is not None:
                        timing_ok = self.shoulder_at_grip_close >= 50
                        t_color = (0, 255, 0) if timing_ok else (0, 0, 255)
                        t_label = "GOOD" if timing_ok else "TOO HIGH"
                        cv2.putText(display, f"Grasp @Sh={self.shoulder_at_grip_close:.0f}deg [{t_label}] - now lift!",
                                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, t_color, 2)
                        y += 22

                # 데이터셋 진행률 (왼쪽 하단)
                progress_lines = self.stats.get_progress_str()
                y_progress = display.shape[0] - 140
                cv2.putText(display, "=== Dataset Progress ===", (10, y_progress),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_progress += 20
                for line in progress_lines:
                    cv2.putText(display, line, (10, y_progress),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                    y_progress += 20

                # 추천 수집 타입 (강조)
                recommendation, rec_color = self.stats.get_recommendation()
                cv2.putText(display, f"Next: {recommendation}", (10, y_progress + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)

                # 나머지 관절 (작게, 상단)
                joint_names = ["Base", "Shldr", "Elbow", "Wrist", "Roll", "Grip"]
                joint_str = " | ".join(f"{n}:{a:+.0f}" for n, a in zip(joint_names, angles))
                cv2.putText(display, joint_str, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y += 25

                cv2.putText(display, "Space:Rec | Enter:Save | Bksp:Cancel | T:Torque | ESC:Quit",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

                # 녹화 중 표시 (우상단)
                if self.is_recording:
                    cv2.circle(display, (display.shape[1] - 30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(display, "REC", (display.shape[1] - 70, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(display, f"Frame: {len(self.current_episode)}",
                               (display.shape[1] - 140, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow("Camera View", display)

                # 두 번째 카메라 표시
                if self.second_cam_type != "none" and second_rgb is not None:
                    second_display = cv2.resize(second_rgb, (480, 270))
                    cam_label = "Wrist (ZED)" if self.second_cam_type == "zed_wrist" else "External (Kinect2)"
                    cv2.putText(second_display, cam_label, (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    if self.is_recording:
                        cv2.circle(second_display, (second_display.shape[1] - 20, 20), 8, (0, 0, 255), -1)
                    cv2.imshow("Second Camera", second_display)

                # OpenCV 키 처리
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.running = False

                time.sleep(0.01)  # CPU 부하 감소

        except KeyboardInterrupt:
            print("\n중단됨")
        finally:
            # 정리
            listener.stop()
            cv2.destroyAllWindows()

            # 녹화 중이면 저장 여부 확인
            if len(self.current_episode) > 0:
                print(f"\n저장되지 않은 {len(self.current_episode)} 프레임이 있습니다.")
                save = input("저장하시겠습니까? (y/n): ").strip().lower()
                if save == 'y':
                    self.save_episode()

            # 토크 켜고 종료
            print("\n토크를 켜고 종료합니다...")
            self.set_torque(True)

            self.k4a.stop()
            if self.zed is not None:
                self.zed.close()
            if self.k4a2 is not None:
                self.k4a2.stop()
            self.robot.disconnect()
            print("\n정리 완료!")
            print(f"총 {self.episode_count} 에피소드 수집됨")
            print(f"저장 위치: {os.path.abspath(self.save_dir)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoArm M3 수동 데이터 수집")
    parser.add_argument("--object", default="sponge",
                        help="수집할 물체 이름 (sponge/cup/box/tool)")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="로봇 시리얼 포트")
    parser.add_argument("--save-dir", default=None,
                        help="저장 디렉토리 (기본: collected_data_{object})")
    parser.add_argument("--second-camera", default="none",
                        choices=["none", "zed_wrist", "kinect_external"],
                        help="두 번째 카메라 (none/zed_wrist/kinect_external)")
    args = parser.parse_args()

    save_dir = args.save_dir or f"collected_data_{args.object}"

    collector = ManualDataCollector(
        robot_port=args.port,
        save_dir=save_dir,
        object_name=args.object,
        second_camera=args.second_camera,
    )
    collector.run()
