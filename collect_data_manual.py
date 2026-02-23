"""
RoArm M3 + Azure Kinect 수동 데이터 수집 스크립트
토크 OFF 상태에서 손으로 로봇을 직접 움직여서 데이터 수집

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


class DatasetStats:
    """기존 데이터셋 통계 분석 클래스"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.deep_count = 0
        self.approach_count = 0
        self.shallow_count = 0
        self.total_count = 0
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
            except:
                pass

    def get_recommendation(self):
        """다음 추천 수집 타입 반환"""
        # 목표: DEEP 50%, APPROACH 30%, SHALLOW 20%
        deep_ratio = self.deep_count / max(1, self.total_count)
        approach_ratio = self.approach_count / max(1, self.total_count)
        shallow_ratio = self.shallow_count / max(1, self.total_count)

        if deep_ratio < 0.50:
            return "DEEP GRASP", (0, 255, 0)
        elif approach_ratio < 0.30:
            return "APPROACH", (0, 255, 255)
        elif shallow_ratio < 0.20:
            return "SHALLOW", (0, 100, 255)
        else:
            return "DEEP GRASP", (0, 255, 0)  # 기본값

    def get_progress_str(self):
        """진행률 문자열 생성 (목표: 120 에피소드)"""
        target_total = 120
        target_deep = int(target_total * 0.5)
        target_approach = int(target_total * 0.3)
        target_shallow = int(target_total * 0.2)

        lines = []
        lines.append(f"DEEP: {self.deep_count}/{target_deep} ({self.deep_count/max(1,target_deep)*100:.0f}%)")
        lines.append(f"APPROACH: {self.approach_count}/{target_approach} ({self.approach_count/max(1,target_approach)*100:.0f}%)")
        lines.append(f"SHALLOW: {self.shallow_count}/{target_shallow} ({self.shallow_count/max(1,target_shallow)*100:.0f}%)")
        lines.append(f"Total: {self.total_count}/{target_total} ({self.total_count/max(1,target_total)*100:.0f}%)")
        return lines


class ManualDataCollector:
    def __init__(self, robot_port="/dev/ttyUSB0", save_dir="collected_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 데이터셋 통계 초기화
        self.stats = DatasetStats(save_dir)

        # 로봇 연결
        print(f"로봇 연결 중... ({robot_port})")
        self.robot = roarm(roarm_type="roarm_m3", port=robot_port, baudrate=115200)
        time.sleep(0.5)
        print("로봇 연결됨!")

        # Azure Kinect 초기화
        print("Azure Kinect 초기화 중...")
        self.k4a = PyK4A(Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ))
        self.k4a.start()
        time.sleep(1)
        print("Azure Kinect 시작됨!")

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

    def get_camera_frame(self):
        """Azure Kinect에서 RGB + Depth 프레임 가져오기"""
        capture = self.k4a.get_capture()
        rgb = np.ascontiguousarray(capture.color[:, :, :3])  # BGRA -> BGR
        depth = capture.transformed_depth  # RGB에 정렬된 깊이
        return rgb, depth

    def get_robot_angles(self):
        """로봇 관절 각도 읽기 (재시도 로직 포함)"""
        for _ in range(5):
            try:
                angles = self.robot.joints_angle_get()
                if angles is not None and len(angles) >= 6:
                    return list(angles)
            except:
                time.sleep(0.05)
        return [0, 0, 0, 0, 0, 0]  # 실패시 기본값

    def get_robot_pose(self):
        """로봇 엔드이펙터 위치 읽기 (FK, 재시도 로직 포함)"""
        for _ in range(5):
            try:
                pose = self.robot.pose_get()
                if pose is not None and len(pose) >= 3:
                    return pose  # [x, y, z, tilt_deg, roll_deg, gripper_deg]
            except:
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

    def save_frame(self, rgb, depth, angles, pose):
        """현재 프레임을 에피소드에 추가"""
        frame_data = {
            "timestamp": time.time(),
            "angles": angles.copy(),
            "pose": pose[:3] if pose else None,  # [x_mm, y_mm, z_mm]
            "frame_idx": len(self.current_episode)
        }

        self.current_episode.append({
            "data": frame_data,
            "rgb": rgb.copy(),
            "depth": depth.copy()
        })

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

        # 4. 프레임 수 체크
        if num_frames < 50:
            issues.append(f"프레임 수 부족 ({num_frames} < 50)")
        elif num_frames > 300:
            warnings.append(f"에피소드 너무 김 ({num_frames}프레임 = {num_frames/30:.1f}초, 5-6초 권장)")

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

        # 메타데이터 저장
        gripper_range = self.max_gripper - self.min_gripper
        z_range = self.max_z - self.min_z
        metadata = {
            "episode_id": self.episode_count,
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
        print(f"에피소드 {self.episode_count} 저장 완료!")
        print(f"  프레임: {len(self.current_episode)} ({len(self.current_episode)/30:.1f}초)")
        print(f"  Min Z: {self.min_z:.0f}mm → [{quality}] {quality_color}")
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
        except:
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

        print("\n준비 완료! Space를 눌러 녹화를 시작하세요.\n")

        try:
            while self.running:
                current_time = time.time()

                # 카메라 프레임 가져오기
                rgb, depth = self.get_camera_frame()
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
                    # frame_idx를 save_frame 전에 기록 (append 전 길이 = 현재 프레임 인덱스)
                    frame_idx = len(self.current_episode)

                    if current_time - self.last_record_time >= 1.0 / self.record_fps:
                        self.save_frame(rgb, depth, angles, pose)
                        self.last_record_time = current_time

                    self.min_elbow = min(self.min_elbow, elbow)
                    self.max_elbow = max(self.max_elbow, elbow)
                    self.min_gripper = min(self.min_gripper, gripper)
                    self.max_gripper = max(self.max_gripper, gripper)
                    self.max_shoulder = max(self.max_shoulder, shoulder)
                    if pose:
                        self.min_z = min(self.min_z, z_height)
                        self.max_z = max(self.max_z, z_height)

                    # 그리퍼 열림/닫힘 감지
                    if gripper > 40 and not self.grip_was_open:
                        self.grip_was_open = True
                        self.grip_open_frame = frame_idx
                    if (self.prev_gripper is not None and
                            self.grip_was_open and
                            self.prev_gripper > 30 and gripper < 15 and
                            self.shoulder_at_grip_close is None):
                        # 그리퍼가 열렸다가 닫히는 순간 (첫 번째만 기록)
                        self.shoulder_at_grip_close = shoulder
                        self.z_at_grip_close = z_height if pose else None  # None when pose_get() failed
                        self.grip_close_frame = frame_idx
                    self.prev_gripper = gripper

                # Z-height 존 판정 + 컬러
                # Calibrated: Z=30mm=table touch, Z=80mm=object grasp, Z=160mm=approach, Z=230mm+=home
                # NOTE: entire episode does NOT need to be green — only the grasp-close moment needs green
                if z_height < 80:
                    z_zone = "DEEP"
                    z_color = (0, 255, 0)      # 초록 (잡기 위치, 물체 높이)
                elif z_height < 160:
                    z_zone = "APPROACH"
                    z_color = (0, 255, 255)    # 노랑 (접근 중)
                else:
                    z_zone = "SHALLOW"
                    z_color = (0, 100, 255)    # 주황 (홈 위치)

                # 화면에 정보 표시
                display = rgb.copy()
                torque_status = "ON" if self.torque_on else "OFF"
                rec_status = "RECORDING" if self.is_recording else "STANDBY"
                status_color = (0, 0, 255) if self.is_recording else (0, 255, 0)

                # 상단: 에피소드 번호 + 프레임 수
                y = 30
                cv2.putText(display, f"Episode {self.episode_count} | Frames {len(self.current_episode)}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                y += 30

                cv2.putText(display, f"Torque {torque_status} | {rec_status}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                y += 35

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
            self.robot.disconnect()
            print("\n정리 완료!")
            print(f"총 {self.episode_count} 에피소드 수집됨")
            print(f"저장 위치: {os.path.abspath(self.save_dir)}")


if __name__ == "__main__":
    collector = ManualDataCollector(
        robot_port="/dev/ttyUSB0",
        save_dir="collected_data"
    )
    collector.run()
