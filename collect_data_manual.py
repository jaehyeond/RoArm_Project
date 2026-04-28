"""
RoArm M3 + Azure Kinect 데이터 수집 스크립트

모드:
  1) 단일팔 토크OFF (기본): 손으로 로봇을 직접 움직여서 데이터 수집
  2) Leader-Follower: Leader 팔을 손으로 움직이면 Follower가 미러링, 카메라는 Follower 촬영

듀얼 카메라 지원:
  --second-camera zed_wrist      ZED Mini wrist 카메라 (pyzed, RGB only)
  --second-camera kinect_external Azure Kinect 2대째 외부 시점 (pyk4a)

Leader-Follower 모드:
  --leader-port /dev/ttyUSB0     Leader 팔 포트 (유저가 손으로 조작)
  --follower-port /dev/ttyUSB1   Follower 팔 포트 (미러링, 카메라가 촬영)
  Leader = action (유저 의도), Follower = state (실제 로봇 위치)

조작법:
  Space: 녹화 시작/중지 (토글)
  Enter: 에피소드 저장
  Backspace: 현재 에피소드 취소
  T: 토크 ON/OFF 토글 (단일팔) / Leader 토크 토글 (L-F)
  I: 초기 위치로 이동
  ESC: 종료
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A
from pynput import keyboard
from roarm_sdk.roarm import roarm
from roarm_sdk.common import handle_m3_feedback, JsonCmd
import logging

# SDK 로그 억제
logging.getLogger('BaseController').setLevel(logging.CRITICAL)

# SDK print(data) 스팸 억제 — _process_received 몽키패치
_orig_process = roarm._process_received
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
roarm._process_received = _silent_process


# RoArm M3 관절 제한 (L-F 미러링 clamp용)
JOINT_LIMITS = [(-190, 190), (-110, 110), (-70, 190), (-110, 110), (-190, 190), (-10, 100)]

# 책상 표면 z 하한 (2026-03-26 실측: 책상 z ≈ -95 ~ -121mm)
Z_DESK_SURFACE = -120   # 책상 표면 FK z (mm)
Z_FLOOR_DEPLOY = -90    # 배포 시 안전 하한 (스펀지 상면)
Z_FLOOR_WARNING = -110  # 수집 시 경고 (책상 접근)


def classify_zone(base_angle, fk_dist=None, fk_z=None):
    """Base 각도 중심 5-zone 분류 (2026-03-31 재설계)

    v5 visual grounding 실패 원인: 이전 zone 시스템이 거리+높이 기반이라
    5개 zone 중 3개(NEAR/FAR_CENTER/OVERHEAD)가 base≈10°로 수렴.
    80.1%의 데이터가 |base|<30°에 집중 → 모델이 base 각도 다양성 학습 불가.

    재설계: base 각도가 유일한 분류 축. 거리/높이는 zone 내 자연 변동으로 처리.
    이렇게 해야 "스펀지가 어디에 있든 vision으로 찾아서 잡기" 학습 가능.

    Args:
        base_angle: base joint angle (degrees)
        fk_dist: (unused, backward compat)
        fk_z: (unused, backward compat)

    Returns:
        zone name: FAR_LEFT, LEFT, CENTER, RIGHT, FAR_RIGHT
    """
    if base_angle < -40:
        return "FAR_LEFT"
    elif base_angle < -10:
        return "LEFT"
    elif base_angle <= 10:
        return "CENTER"
    elif base_angle <= 40:
        return "RIGHT"
    else:
        return "FAR_RIGHT"


# Zone 이름 목록 (OSD 참고 표시용 — quota 강제 없음)
# 바닐라 SmolVLA 공식 레시피: zone 시스템 없음, 88% center로도 성공
# quota는 v5 실패의 잘못된 진단에서 나온 과잉 대응이었음 → 제거 (2026-04-01)
ZONE_NAMES = ["FAR_LEFT", "LEFT", "CENTER", "RIGHT", "FAR_RIGHT"]

ZONE_COLORS = {
    "FAR_LEFT": (255, 50, 50),    # 진한 파랑 (BGR)
    "LEFT": (255, 150, 100),      # 밝은 파랑
    "CENTER": (0, 255, 0),        # 초록
    "RIGHT": (100, 150, 255),     # 밝은 빨강
    "FAR_RIGHT": (50, 50, 255),   # 진한 빨강
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
        # Zone 카운터 (참고 표시용, quota 강제 없음)
        self.zone_counts = {z: 0 for z in ZONE_NAMES}
        self.analyze_existing_episodes()

    def analyze_existing_episodes(self):
        """기존 에피소드 분석 (metadata.json 기반)"""
        # 카운트 리셋 후 재스캔 (누적 합산 버그 방지)
        self.deep_count = 0
        self.approach_count = 0
        self.shallow_count = 0
        self.total_count = 0
        self.zone_counts = {z: 0 for z in ZONE_NAMES}

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

    def get_zone_summary(self):
        """Zone 분포 요약 문자열 (참고 표시용, quota 없음)"""
        parts = []
        for zone in ZONE_NAMES:
            count = self.zone_counts.get(zone, 0)
            if count > 0:
                parts.append(f"{zone}:{count}")
        return " | ".join(parts) if parts else "No episodes yet"


class ManualDataCollector:
    def __init__(self, robot_port="/dev/ttyUSB1", save_dir="collected_data", object_name="sponge",
                 second_camera="none", leader_port=None, follower_port=None):
        self.save_dir = save_dir
        self.object_name = object_name
        self.second_cam_type = second_camera  # none / zed_wrist / kinect_external
        os.makedirs(save_dir, exist_ok=True)

        # L-F 모드 판별
        self.lf_mode = leader_port is not None and follower_port is not None

        # 데이터셋 통계 초기화
        self.stats = DatasetStats(save_dir)

        if self.lf_mode:
            # Leader-Follower 모드: 두 팔 연결
            print(f"[L-F MODE] Leader 연결 중... ({leader_port})")
            self.leader = roarm(roarm_type="roarm_m3", port=leader_port, baudrate=115200)
            time.sleep(0.5)
            print(f"[L-F MODE] Follower 연결 중... ({follower_port})")
            self.robot = roarm(roarm_type="roarm_m3", port=follower_port, baudrate=115200)
            time.sleep(0.5)
            print("Leader + Follower 연결됨!")
        else:
            # 단일팔 모드 (기존)
            self.leader = None
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
        self.pending_confirmation = False  # FAIL 에피소드 강제저장 대기
        self.pending_fail_reasons = []    # FAIL 이유 목록 (OSD 표시용)

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

    def _safe_angle_read(self, arm):
        """팔에서 관절 각도 읽기 (재시도 로직 포함)"""
        for _ in range(5):
            try:
                angles = arm.joints_angle_get()
                if angles is not None and len(angles) >= 6:
                    return list(angles)
            except Exception:
                time.sleep(0.05)
        return None  # 실패시 None (호출부에서 처리)

    def get_robot_angles(self):
        """Follower(또는 단일팔) 관절 각도 = observation state"""
        return self._safe_angle_read(self.robot)

    def get_leader_angles(self):
        """Leader 관절 각도 = action (L-F 모드 전용)"""
        if self.leader is None:
            return None
        return self._safe_angle_read(self.leader)

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
        """토크 ON/OFF 설정

        L-F 모드: Leader 토크만 토글. Follower는 항상 ON (미러링 수행).
        단일팔 모드: 팔 토크 토글.
        """
        if self.lf_mode:
            self.leader.torque_set(cmd=1 if on else 0)
            # Follower는 항상 토크 ON
            self.robot.torque_set(cmd=1)
        else:
            self.robot.torque_set(cmd=1 if on else 0)
        self.torque_on = on
        time.sleep(0.3)
        status = "ON" if on else "OFF"
        if self.lf_mode:
            print(f"\nLeader 토크 {status}! (Follower는 항상 ON)")
        else:
            print(f"\n토크 {status}!")
        if not on:
            print("→ 이제 손으로 팔을 자유롭게 움직일 수 있습니다.")

    def save_frame(self, rgb, depth, angles, pose, second_rgb=None, leader_angles=None):
        """현재 프레임을 에피소드에 추가

        Args:
            angles: Follower(또는 단일팔) 각도 = observation state
            leader_angles: Leader 각도 = action (L-F 모드, 공식 LeRobot 방식)
        """
        frame_data = {
            "timestamp": time.time(),
            "angles": angles.copy(),
            "pose": pose[:3] if pose else None,  # [x_mm, y_mm, z_mm]
            "frame_idx": len(self.current_episode)
        }
        # L-F 모드: Leader 각도를 action으로 별도 저장
        if leader_angles is not None:
            frame_data["leader_angles"] = leader_angles.copy()

        entry = {
            "data": frame_data,
            "rgb": rgb.copy(),
            "depth": depth.copy()
        }
        if second_rgb is not None:
            entry["second_rgb"] = second_rgb.copy()
        self.current_episode.append(entry)

    def validate_episode(self):
        """에피소드 품질 검증 (공식 lerobot-record 기준: 최소한의 검증만)

        공식 lerobot-record: 에피소드 검증 = 제로. re-record 옵션만 존재.
        우리 추가 검증: HOME 시작(C0a)과 Z 안전(C5)만 HARD BLOCK.
        나머지는 WARNING으로 운영자 판단에 위임.

        Z Calibration (confirmed by user, 2026-02-23):
          Z=30mm  = table surface, Z=80mm  = object grasp
          Z=160mm = approach, Z=230mm+ = home
        """
        issues = []
        warnings = []

        num_frames = len(self.current_episode)

        # C0a. HOME 시작 검증 — HARD BLOCK (v5 실패 근본 원인, 유일하게 정당한 FAIL)
        if num_frames >= 10:
            start_state = [self.current_episode[0]["data"]["angles"][i] for i in range(6)]
            home = [0, 0, 90, 0, 0, 0]
            home_dist = sum((s - h) ** 2 for s, h in zip(start_state, home)) ** 0.5

            if home_dist > 30:
                issues.append(
                    f"NOT started from HOME! dist={home_dist:.0f}deg "
                    f"(start=[{start_state[0]:+.0f},{start_state[1]:+.0f},{start_state[2]:+.0f},"
                    f"{start_state[3]:+.0f},{start_state[4]:+.0f},{start_state[5]:+.0f}]). "
                    f"Press I to go HOME, THEN start recording!")

        # C1. Gripper must open (WARNING only — 30° threshold, relaxed from 40°)
        if not self.grip_was_open and self.max_gripper < 30:
            warnings.append(f"Gripper barely opened (max={self.max_gripper:.0f}deg)")

        # C3. Grasp depth (WARNING only — shoulder < 30° means arm didn't reach down)
        if self.shoulder_at_grip_close is not None and self.shoulder_at_grip_close < 30:
            warnings.append(f"Arm may be too high at grasp (Sh={self.shoulder_at_grip_close:.0f}deg)")

        # C5. Z safety — HARD BLOCK (physical safety)
        if self.z_at_grip_close is not None and self.z_at_grip_close > 130:
            issues.append(f"Z too high at grasp (Z={self.z_at_grip_close:.0f}mm > 130)")

        # C4. Frame count (WARNING at <90, no FAIL — operator decides)
        if num_frames < 90:
            warnings.append(f"Short episode ({num_frames}fr = {num_frames/30:.1f}s)")
        elif num_frames > 600:
            warnings.append(f"Long episode ({num_frames}fr = {num_frames/30:.1f}s)")

        # 5. Z-height check (DEEP grasp)
        if self.min_z > 160:
            warnings.append(f"Shallow grasp (min_z={self.min_z:.0f}mm, need <160)")

        return issues, warnings

    def save_episode(self, force=False):
        """현재 에피소드를 디스크에 저장 (force=True: 품질 검증 건너뜀)"""
        if len(self.current_episode) == 0:
            print("저장할 프레임이 없습니다!")
            return

        # 품질 검증 (force=True면 건너뜀)
        if not force:
            issues, warnings = self.validate_episode()
            if issues or warnings:
                print(f"\n{'='*50}", flush=True)
                if issues:
                    print("FAIL - 에피소드 품질 문제:", flush=True)
                    for i, issue in enumerate(issues, 1):
                        print(f"  [{i}] {issue}", flush=True)
                if warnings:
                    print("WARN - 개선 권장:", flush=True)
                    for i, warn in enumerate(warnings, 1):
                        print(f"  ({i}) {warn}", flush=True)

                # 그리퍼 타이밍 정보 출력
                if self.shoulder_at_grip_close is not None:
                    z_part = f", Z={self.z_at_grip_close:.0f}mm" if self.z_at_grip_close is not None else ""
                    print(f"\n  그리퍼 닫기 시점: shoulder={self.shoulder_at_grip_close:.0f}°{z_part}", flush=True)
                    if self.grip_open_frame is not None and self.grip_close_frame is not None:
                        open_pct = self.grip_open_frame / max(1, len(self.current_episode)) * 100
                        close_pct = self.grip_close_frame / max(1, len(self.current_episode)) * 100
                        print(f"  그리퍼 열림: {open_pct:.0f}% 지점, 닫힘: {close_pct:.0f}% 지점", flush=True)
                print(f"{'='*50}", flush=True)

                if issues:
                    # FAIL: 키보드로 확인 (input() 대신 — conda run 호환)
                    print("Enter=강제저장, Backspace=취소", flush=True)
                    sys.stdout.flush()  # pynput 스레드에서 호출 — 버퍼 명시적 플러시
                    self.pending_confirmation = True
                    self.pending_fail_reasons = issues  # OSD에 이유 표시
                    return
            # warnings만 있으면 자동 저장 진행

        episode_dir = os.path.join(self.save_dir, f"episode_{self.episode_count:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        # Zone 판정 (그리퍼 닫기 시점의 base angle 기반 — 2026-03-31 재설계)
        ep_zone = "UNKNOWN"
        if self.current_episode:
            grasp_frame_idx = self.grip_close_frame if self.grip_close_frame is not None else len(self.current_episode) // 2
            grasp_frame_idx = min(grasp_frame_idx, len(self.current_episode) - 1)
            gf = self.current_episode[grasp_frame_idx]
            gf_base = gf["data"].get("angles", [0])[0] if "angles" in gf["data"] else 0
            ep_zone = classify_zone(gf_base)

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

        print(f"\n{'='*50}", flush=True)
        print(f"에피소드 {self.episode_count} 저장 완료! [Zone: {ep_zone}]", flush=True)
        print(f"  프레임: {len(self.current_episode)} ({len(self.current_episode)/30:.1f}초)", flush=True)
        print(f"  Zone: {ep_zone} | Min Z: {self.min_z:.0f}mm → [{quality}]", flush=True)
        print(f"  Max Shoulder: {self.max_shoulder:.0f}°", flush=True)
        print(f"  Gripper: {self.min_gripper:.0f}° ~ {self.max_gripper:.0f}° (range={gripper_range:.0f}°)", flush=True)
        if self.shoulder_at_grip_close is not None:
            z_str = f", Z={self.z_at_grip_close:.0f}mm" if self.z_at_grip_close else ""
            timing_ok = "OK" if self.shoulder_at_grip_close >= 50 else "LOW"
            print(f"  Grasp: shoulder={self.shoulder_at_grip_close:.0f}°{z_str} [{timing_ok}]", flush=True)
        else:
            print(f"  Grasp: 닫기 감지 안됨", flush=True)
        print(f"  저장: {episode_dir}", flush=True)
        print(f"{'='*50}\n", flush=True)

        # 통계 업데이트
        self.stats.analyze_existing_episodes()

        self.episode_count += 1
        self._reset_episode_tracking()

    def _reset_episode_tracking(self):
        """에피소드 추적 변수 전체 초기화"""
        self.current_episode = []
        self.is_recording = False
        self.pending_confirmation = False
        self.pending_fail_reasons = []
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

        # Pending confirmation 처리 (FAIL 에피소드)
        if self.pending_confirmation:
            if key == keyboard.Key.enter:
                print("강제 저장...", flush=True)
                self.pending_confirmation = False
                self.save_episode(force=True)
            elif key == keyboard.Key.backspace:
                self.pending_confirmation = False
                self.cancel_episode()
            # pending 상태에서 다른 키는 무시
            return

        # Space로 녹화 시작/중지 토글
        if key == keyboard.Key.space:
            if not self.is_recording:
                # 녹화 시작 전 HOME 위치 확인 — HARD BLOCK
                # v5 실패 원인: 타겟 근처에서 시작 → approach phase 없음 → echo → 배포 실패
                cur_angles = self.get_robot_angles()
                if cur_angles and len(self.current_episode) == 0:
                    home = [0, 0, 90, 0, 0, 0]
                    home_dist = sum((a - h) ** 2 for a, h in zip(cur_angles, home)) ** 0.5
                    if home_dist > 30:
                        print(f"\n[BLOCKED] HOME에서 시작하세요! (현재 dist={home_dist:.0f}°)")
                        print(f"  현재: [{cur_angles[0]:+.0f},{cur_angles[1]:+.0f},{cur_angles[2]:+.0f},"
                              f"{cur_angles[3]:+.0f},{cur_angles[4]:+.0f},{cur_angles[5]:+.0f}]")
                        print(f"  HOME:  [  0,  0, 90,  0,  0,  0]")
                        print(f"  I키를 눌러 HOME으로 이동한 뒤 다시 Space")
                        print(f"  (v5 136ep 실패 방지: HOME→스펀지 approach가 visual grounding 핵심)")
                        return  # 녹화 시작 차단
                    else:
                        print(f"\n[REC] 녹화 시작! (HOME 확인 OK, dist={home_dist:.0f}°)")
                else:
                    # 이어서 녹화 (이미 프레임 있음) 또는 각도 읽기 실패
                    print(f"\n[REC] 녹화 재개 ({len(self.current_episode)} 프레임 기존)")
                self.is_recording = True
            else:
                self.is_recording = False
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
            HOME = [0, 0, 90, 0, 0, 0]
            if self.lf_mode:
                print("\n[L-F] 양쪽 팔을 HOME으로 이동 중...")
                self.leader.torque_set(cmd=1)
                self.robot.torque_set(cmd=1)
                self.torque_on = True
                time.sleep(0.3)
                self.robot.joints_angle_ctrl(angles=HOME, speed=500, acc=200)
                self.leader.joints_angle_ctrl(angles=HOME, speed=500, acc=200)
                time.sleep(1)
                self.robot.joints_angle_ctrl(angles=HOME, speed=500, acc=200)
                self.leader.joints_angle_ctrl(angles=HOME, speed=500, acc=200)
                time.sleep(3)
                print("HOME 도착! T키로 Leader 토크 OFF하세요.")
            else:
                if not self.torque_on:
                    print("\n초기 위치 이동을 위해 토크를 켭니다...")
                    self.set_torque(True)
                print("초기 위치로 이동 중...")
                self.robot.move_init()
                time.sleep(2)
                print("초기 위치 도착!")

    def run(self):
        """메인 루프"""
        mode_str = "Leader-Follower" if self.lf_mode else "토크 OFF 단일팔"
        print("\n" + "="*60)
        print(f"RoArm M3 데이터 수집 ({mode_str})")
        print("="*60)
        if self.lf_mode:
            print("\n[L-F] 워크플로우:")
            print("  1. 양쪽 팔이 HOME으로 이동")
            print("  2. Leader 토크 OFF (손으로 자유 조작)")
            print("  3. Space → 녹화 시작 (HOME에서!)")
            print("  4. Leader를 움직여서 Follower가 스펀지 잡기")
            print("  5. Enter → 에피소드 저장")
            print("  ※ 카메라는 Follower만 촬영. Leader+손은 화각 밖!")
        else:
            print("\n[중요] Visual Grounding 학습을 위한 수집 워크플로우:")
            print("  1. I키 → 홈 위치로 이동")
            print("  2. 스펀지를 원하는 위치에 배치")
            print("  3. T키 → 토크 OFF")
            print("  4. Space → 녹화 시작 (홈 위치에서!)")
            print("  5. 손으로 로봇을 스펀지 방향으로 이동 → 잡기 → 들기")
            print("  6. Enter → 에피소드 저장")
            print("  ※ 반드시 홈 위치에서 녹화 시작! 이미 타겟 근처면 경고 표시")
        print("\n조작법:")
        print("  Space: 녹화 시작/중지 (토글)")
        print("  Enter: 에피소드 저장")
        print("  Backspace: 에피소드 취소")
        print(f"  T: {'Leader' if self.lf_mode else ''} 토크 ON/OFF 토글")
        print("  I: 초기 위치로 이동")
        print("  ESC: 종료")
        print("="*60)

        HOME = [0, 0, 90, 0, 0, 0]

        if self.lf_mode:
            # L-F: 양쪽 HOME → Leader 토크 OFF
            print("\n양쪽 팔을 HOME으로 이동 중...")
            self.robot.torque_set(cmd=1)
            self.leader.torque_set(cmd=1)
            time.sleep(0.5)
            self.robot.joints_angle_ctrl(angles=HOME, speed=500, acc=200)
            self.leader.joints_angle_ctrl(angles=HOME, speed=500, acc=200)
            time.sleep(1)
            # 첫 명령 드랍 대비 재전송
            self.robot.joints_angle_ctrl(angles=HOME, speed=500, acc=200)
            self.leader.joints_angle_ctrl(angles=HOME, speed=500, acc=200)
            time.sleep(3)
            print("*** Leader 팔을 손으로 잡으세요! 토크를 끕니다. ***")
            time.sleep(1)
            self.set_torque(False)
        else:
            # 단일팔: 초기 위치 → 토크 OFF
            print("\n초기 위치로 이동 중...")
            self.robot.move_init()
            time.sleep(2)
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
                angles = self.get_robot_angles()  # Follower(또는 단일팔) = state
                if angles is None:
                    continue  # Follower 읽기 실패 — 프레임 스킵
                pose = self.get_robot_pose()
                self.current_pose = pose  # 캐시 for display

                # L-F 모드: Leader 각도 읽기 + Follower 미러링
                leader_angles = None
                if self.lf_mode:
                    leader_angles = self.get_leader_angles()
                    if leader_angles is None:
                        continue  # Leader 읽기 실패 — 프레임 스킵 (B3 fix: [0]*6 대신 스킵)
                    clamped = [max(lo, min(hi, a))
                               for a, (lo, hi) in zip(leader_angles, JOINT_LIMITS)]
                    self.robot.joints_angle_ctrl(angles=clamped, speed=0, acc=0)

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
                        self.save_frame(rgb, depth, angles, pose, second_rgb, leader_angles)
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
                # L-F 모드: Leader base가 유저 의도(=물체 위치)를 더 정확히 반영
                base_angle = leader_angles[0] if leader_angles is not None else angles[0]
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

                mode_label = "[L-F]" if self.lf_mode else ""
                cv2.putText(display, f"{mode_label} Torque {torque_status} | {rec_status}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                # HOME 거리 표시 — 녹화 전 항상 확인 (STANDBY에서 더 강조)
                home_ref = [0, 0, 90, 0, 0, 0]
                home_d = sum((a - h) ** 2 for a, h in zip(angles, home_ref)) ** 0.5
                if not self.is_recording:
                    if home_d <= 30:
                        home_label = f"HOME OK (dist={home_d:.0f})"
                        home_color = (0, 255, 0)
                    else:
                        home_label = f"NOT HOME! (dist={home_d:.0f}) Press I"
                        home_color = (0, 0, 255)
                    cv2.putText(display, home_label,
                               (300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, home_color, 2)
                y += 35

                # Spatial Zone 표시 (우상단) — 참고용, quota 없음
                cv2.putText(display, f"ZONE: {spatial_zone}", (display.shape[1] - 300, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, spatial_color, 2)
                zone_count_now = self.stats.zone_counts.get(spatial_zone, 0)
                cv2.putText(display, f"Base:{base_angle:+.0f} | count:{zone_count_now}",
                           (display.shape[1] - 300, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, spatial_color, 1)
                # 전체 zone 분포 (참고)
                cv2.putText(display, f"Total: {self.stats.total_count}ep",
                           (display.shape[1] - 300, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

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

                # 데이터셋 zone 분포 (왼쪽 하단, 참고용)
                y_progress = display.shape[0] - 60
                zone_summary = self.stats.get_zone_summary()
                cv2.putText(display, f"Zones: {zone_summary}", (10, y_progress),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                cv2.putText(display, f"Total: {self.stats.total_count} episodes", (10, y_progress + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                # Pending confirmation 표시 (FAIL 에피소드)
                if self.pending_confirmation:
                    mid_y = display.shape[0] // 2
                    cv2.putText(display, "FAIL! Enter=Force Save | Bksp=Cancel",
                               (display.shape[1] // 2 - 250, mid_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    for r_idx, reason in enumerate(self.pending_fail_reasons):
                        cv2.putText(display, f"  [{r_idx+1}] {reason}",
                                   (display.shape[1] // 2 - 250, mid_y + 30 + r_idx * 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

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

            # 녹화 중이면 자동 저장 (input() 대신 — conda run 호환)
            if len(self.current_episode) > 0:
                print(f"\n저장되지 않은 {len(self.current_episode)} 프레임 자동 저장...")
                self.save_episode(force=True)

            # 토크 켜고 종료
            print("\n토크를 켜고 종료합니다...")
            self.set_torque(True)

            self.k4a.stop()
            if self.zed is not None:
                self.zed.close()
            if self.k4a2 is not None:
                self.k4a2.stop()
            self.robot.disconnect()
            if self.leader is not None:
                self.leader.torque_set(cmd=1)
                time.sleep(0.3)
                self.leader.disconnect()
            print("\n정리 완료!")
            print(f"총 {self.episode_count} 에피소드 수집됨")
            print(f"저장 위치: {os.path.abspath(self.save_dir)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoArm M3 데이터 수집")
    parser.add_argument("--object", default="sponge",
                        help="수집할 물체 이름 (sponge/cup/box/tool)")
    parser.add_argument("--port", default="/dev/ttyUSB1", help="로봇 시리얼 포트 (단일팔 모드)")
    parser.add_argument("--save-dir", default=None,
                        help="저장 디렉토리 (기본: collected_data_{object})")
    parser.add_argument("--second-camera", default="none",
                        choices=["none", "zed_wrist", "kinect_external"],
                        help="두 번째 카메라 (none/zed_wrist/kinect_external)")
    # Leader-Follower 모드
    parser.add_argument("--leader-port", default=None,
                        help="Leader 팔 포트 (L-F 모드 활성화, 예: /dev/ttyUSB0)")
    parser.add_argument("--follower-port", default=None,
                        help="Follower 팔 포트 (L-F 모드, 예: /dev/ttyUSB1)")
    args = parser.parse_args()

    save_dir = args.save_dir or f"collected_data_{args.object}"

    collector = ManualDataCollector(
        robot_port=args.port,
        save_dir=save_dir,
        object_name=args.object,
        second_camera=args.second_camera,
        leader_port=args.leader_port,
        follower_port=args.follower_port,
    )
    collector.run()
