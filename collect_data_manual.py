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

                    # Z-height 우선 사용 (새 에피소드), 없으면 elbow로 폴백 (구 에피소드)
                    min_z = metadata.get('min_z', None)
                    if min_z is not None:
                        # Z-based classification: DEEP (< 100mm), APPROACH (100-200mm), SHALLOW (> 200mm)
                        if min_z < 100:
                            self.deep_count += 1
                        elif min_z < 200:
                            self.approach_count += 1
                        else:
                            self.shallow_count += 1
                    else:
                        # Fallback to elbow-based classification (backward compat)
                        min_elbow = metadata.get('min_elbow', 999)
                        if min_elbow < -30:
                            self.deep_count += 1
                        elif min_elbow < -10:
                            self.approach_count += 1
                        else:
                            self.shallow_count += 1

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
        """에피소드 품질 검증"""
        issues = []

        # Z-height 체크 (DEEP grasp 기준: < 100mm)
        if self.min_z > 200:
            issues.append(f"얕은 그리핑 (min_z={self.min_z:.0f}mm > 200mm)")

        # Gripper 체크
        gripper_range = self.max_gripper - self.min_gripper
        if gripper_range < 15:
            issues.append(f"그리퍼 개폐 부족 (range={gripper_range:.1f}° < 15°)")

        # 프레임 수 체크
        if len(self.current_episode) < 50:
            issues.append(f"프레임 수 부족 ({len(self.current_episode)} < 50)")

        return issues

    def save_episode(self):
        """현재 에피소드를 디스크에 저장"""
        if len(self.current_episode) == 0:
            print("저장할 프레임이 없습니다!")
            return

        # 품질 검증
        issues = self.validate_episode()
        if issues:
            print(f"\n{'='*50}")
            print("⚠ 에피소드 품질 경고:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            print(f"{'='*50}")

            # 사용자 선택
            choice = input("저장하시겠습니까? (y=저장, n=취소, r=재녹화): ").strip().lower()
            if choice == 'n':
                print("에피소드 저장 취소됨")
                return
            elif choice == 'r':
                print("에피소드 취소 후 재녹화하세요 (Backspace)")
                return
            # y 또는 기타: 강제 저장 진행

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
            "min_z": round(self.min_z, 2),
            "max_z": round(self.max_z, 2),
            "z_range": round(z_range, 2),
            "gripper_min": round(self.min_gripper, 2),
            "gripper_max": round(self.max_gripper, 2),
            "gripper_range": round(gripper_range, 2),
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

        # Z-height 품질 판정
        if self.min_z < 100:
            quality = "DEEP GRASP"
            quality_color = "🟢"
        elif self.min_z < 200:
            quality = "APPROACH"
            quality_color = "🟡"
        else:
            quality = "SHALLOW"
            quality_color = "🟠"

        print(f"\n{'='*50}")
        print(f"에피소드 {self.episode_count} 저장 완료!")
        print(f"  프레임 수: {len(self.current_episode)}")
        print(f"  Min Z-Height: {self.min_z:.0f}mm → [{quality}] {quality_color}")
        print(f"  Min Elbow: {self.min_elbow:.1f}° (legacy)")
        print(f"  Gripper Range: {gripper_range:.1f}° (min={self.min_gripper:.1f}, max={self.max_gripper:.1f})")
        if self.min_z > 200:
            print(f"  ⚠ WARNING: 얕은 그리핑! Deep Grasp(< 100mm) 에피소드 더 필요")
        if gripper_range < 15:
            print(f"  ⚠ WARNING: 그리퍼 개폐 부족! 완전히 열고 닫으세요")
        print(f"  저장 위치: {episode_dir}")
        print(f"{'='*50}\n")

        # 통계 업데이트
        self.stats.analyze_existing_episodes()

        self.episode_count += 1
        self.current_episode = []
        self.is_recording = False
        self.min_elbow = 999
        self.max_elbow = -999
        self.min_gripper = 999
        self.max_gripper = -999
        self.min_z = 9999
        self.max_z = -9999

    def cancel_episode(self):
        """현재 에피소드 취소"""
        if len(self.current_episode) > 0:
            print(f"\n에피소드 취소됨 ({len(self.current_episode)} 프레임 삭제)")
            self.current_episode = []
            self.is_recording = False
            self.min_elbow = 999
            self.max_elbow = -999
            self.min_gripper = 999
            self.max_gripper = -999
            self.min_z = 9999
            self.max_z = -9999
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

                # 녹화 중이면 프레임 저장 (30 FPS)
                if self.is_recording:
                    if current_time - self.last_record_time >= 1.0 / self.record_fps:
                        self.save_frame(rgb, depth, angles, pose)
                        self.last_record_time = current_time

                # Elbow + Gripper + Z 추적
                elbow = angles[2]
                gripper = angles[5]
                z_height = pose[2] if pose else 9999  # Z in mm
                if self.is_recording:
                    self.min_elbow = min(self.min_elbow, elbow)
                    self.max_elbow = max(self.max_elbow, elbow)
                    self.min_gripper = min(self.min_gripper, gripper)
                    self.max_gripper = max(self.max_gripper, gripper)
                    if pose:
                        self.min_z = min(self.min_z, z_height)
                        self.max_z = max(self.max_z, z_height)

                # Z-height 존 판정 + 컬러
                if z_height < 100:
                    z_zone = "DEEP"
                    z_color = (0, 255, 0)      # 초록 (좋음)
                elif z_height < 200:
                    z_zone = "APPROACH"
                    z_color = (0, 255, 255)    # 노랑
                else:
                    z_zone = "SHALLOW"
                    z_color = (0, 100, 255)    # 주황

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

                # Z-Height 크게 표시 (현재값 + 존)
                cv2.putText(display, f"Height: {z_height:.0f}mm", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, z_color, 3)
                cv2.putText(display, f"[{z_zone}]", (280, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, z_color, 2)
                y += 40

                # 위치 정보 (작게): X, Y 좌표
                if pose:
                    cv2.putText(display, f"X:{pose[0]:.0f}mm  Y:{pose[1]:.0f}mm  Elbow:{elbow:+.1f}deg",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    y += 25

                # 녹화 중이면 에피소드 통계 표시
                if self.is_recording and self.min_z < 9999:
                    gripper_range = self.max_gripper - self.min_gripper
                    cv2.putText(display, f"Min Z: {self.min_z:.0f}mm | Gripper: {gripper_range:.1f}deg",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, z_color, 2)
                    y += 30

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
