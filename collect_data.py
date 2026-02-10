"""
RoArm M3 + Azure Kinect 데이터 수집 스크립트
키보드로 로봇을 조작하면서 카메라 이미지와 관절 각도를 저장

조작법:
  Q/A: Joint 1 (베이스 회전) +/-
  W/S: Joint 2 (어깨) +/-
  E/D: Joint 3 (팔꿈치) +/-
  R/F: Joint 4 (손목 피치) +/-
  T/G: Joint 5 (손목 롤) +/-
  Y/H: Joint 6 (그리퍼) +/-

  Space: 현재 프레임 저장 (에피소드에 추가)
  Enter: 에피소드 종료 및 저장
  Backspace: 현재 에피소드 취소

  -/=: 속도 감소/증가
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


class DataCollector:
    def __init__(self, robot_port="/dev/ttyUSB0", save_dir="collected_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

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

        # 조인트 리미트
        self.joint_limits = {
            0: (-190, 190),   # base
            1: (-110, 110),   # shoulder
            2: (-70, 190),    # elbow
            3: (-110, 110),   # wrist pitch
            4: (-190, 190),   # wrist roll
            5: (-10, 100),    # gripper
        }

        # 텔레옵 상태
        self.current_angles = [0, 0, 0, 0, 0, 0]
        self.step_size = 5  # degrees per keypress
        self.step_sizes = [2, 5, 10]
        self.step_index = 1

        # 키 상태
        self.pressed_keys = set()

        # 데이터 수집 상태
        self.current_episode = []
        self.episode_count = len([d for d in os.listdir(save_dir) if d.startswith("episode_")])
        self.frame_count = 0
        self.is_recording = False

        # 실행 상태
        self.running = True

    def get_camera_frame(self):
        """Azure Kinect에서 RGB + Depth 프레임 가져오기"""
        capture = self.k4a.get_capture()
        rgb = capture.color[:, :, :3]  # BGRA -> BGR
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
        return self.current_angles  # 실패시 현재 값 반환

    def clamp_angle(self, joint_idx, angle):
        """관절 각도를 리미트 내로 클램핑"""
        low, high = self.joint_limits[joint_idx]
        return max(low, min(high, angle))

    def move_robot(self, angles):
        """로봇을 지정된 각도로 이동"""
        self.robot.joints_angle_ctrl(
            angles=angles,
            speed=500,
            acc=200
        )

    def save_frame(self, rgb, depth, angles):
        """현재 프레임을 에피소드에 추가"""
        frame_data = {
            "timestamp": time.time(),
            "angles": angles.copy(),
            "frame_idx": len(self.current_episode)
        }

        # 이미지는 메모리에 저장
        self.current_episode.append({
            "data": frame_data,
            "rgb": rgb.copy(),
            "depth": depth.copy()
        })

        self.frame_count += 1
        print(f"  프레임 {len(self.current_episode)} 저장됨 (angles: {[f'{a:.1f}' for a in angles]})")

    def save_episode(self):
        """현재 에피소드를 디스크에 저장"""
        if len(self.current_episode) == 0:
            print("저장할 프레임이 없습니다!")
            return

        episode_dir = os.path.join(self.save_dir, f"episode_{self.episode_count:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        # 메타데이터 저장
        metadata = {
            "episode_id": self.episode_count,
            "num_frames": len(self.current_episode),
            "timestamp": datetime.datetime.now().isoformat(),
            "frames": []
        }

        # 각 프레임 저장
        for i, frame in enumerate(self.current_episode):
            # 이미지 저장
            rgb_path = os.path.join(episode_dir, f"rgb_{i:04d}.jpg")
            depth_path = os.path.join(episode_dir, f"depth_{i:04d}.npy")

            cv2.imwrite(rgb_path, frame["rgb"])
            np.save(depth_path, frame["depth"])

            # 메타데이터에 추가
            frame_info = frame["data"].copy()
            frame_info["rgb_path"] = f"rgb_{i:04d}.jpg"
            frame_info["depth_path"] = f"depth_{i:04d}.npy"
            metadata["frames"].append(frame_info)

        # 메타데이터 저장
        with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n=== 에피소드 {self.episode_count} 저장 완료! ({len(self.current_episode)} 프레임) ===")
        print(f"    저장 위치: {episode_dir}")

        self.episode_count += 1
        self.current_episode = []

    def cancel_episode(self):
        """현재 에피소드 취소"""
        if len(self.current_episode) > 0:
            print(f"\n에피소드 취소됨 ({len(self.current_episode)} 프레임 삭제)")
            self.current_episode = []
        else:
            print("취소할 에피소드가 없습니다.")

    def on_key_press(self, key):
        """키 눌림 이벤트"""
        try:
            k = key.char.lower() if hasattr(key, 'char') and key.char else None
        except:
            k = None

        if k:
            self.pressed_keys.add(k)

        # ESC로 종료
        if key == keyboard.Key.esc:
            self.running = False
            return False

        # Space로 프레임 저장
        if key == keyboard.Key.space:
            self.is_recording = True

        # Enter로 에피소드 저장
        if key == keyboard.Key.enter:
            self.save_episode()

        # Backspace로 에피소드 취소
        if key == keyboard.Key.backspace:
            self.cancel_episode()

    def on_key_release(self, key):
        """키 뗌 이벤트"""
        try:
            k = key.char.lower() if hasattr(key, 'char') and key.char else None
        except:
            k = None

        if k and k in self.pressed_keys:
            self.pressed_keys.discard(k)

    def process_teleop(self):
        """키보드 입력에 따라 로봇 이동"""
        # 현재 각도 읽기
        self.current_angles = self.get_robot_angles()

        new_angles = self.current_angles.copy()
        moved = False

        # Joint 1 (base): Q/A
        if 'q' in self.pressed_keys:
            new_angles[0] = self.clamp_angle(0, new_angles[0] + self.step_size)
            moved = True
        if 'a' in self.pressed_keys:
            new_angles[0] = self.clamp_angle(0, new_angles[0] - self.step_size)
            moved = True

        # Joint 2 (shoulder): W/S
        if 'w' in self.pressed_keys:
            new_angles[1] = self.clamp_angle(1, new_angles[1] + self.step_size)
            moved = True
        if 's' in self.pressed_keys:
            new_angles[1] = self.clamp_angle(1, new_angles[1] - self.step_size)
            moved = True

        # Joint 3 (elbow): E/D
        if 'e' in self.pressed_keys:
            new_angles[2] = self.clamp_angle(2, new_angles[2] + self.step_size)
            moved = True
        if 'd' in self.pressed_keys:
            new_angles[2] = self.clamp_angle(2, new_angles[2] - self.step_size)
            moved = True

        # Joint 4 (wrist pitch): R/F
        if 'r' in self.pressed_keys:
            new_angles[3] = self.clamp_angle(3, new_angles[3] + self.step_size)
            moved = True
        if 'f' in self.pressed_keys:
            new_angles[3] = self.clamp_angle(3, new_angles[3] - self.step_size)
            moved = True

        # Joint 5 (wrist roll): T/G
        if 't' in self.pressed_keys:
            new_angles[4] = self.clamp_angle(4, new_angles[4] + self.step_size)
            moved = True
        if 'g' in self.pressed_keys:
            new_angles[4] = self.clamp_angle(4, new_angles[4] - self.step_size)
            moved = True

        # Joint 6 (gripper): Y/H
        if 'y' in self.pressed_keys:
            new_angles[5] = self.clamp_angle(5, new_angles[5] + self.step_size)
            moved = True
        if 'h' in self.pressed_keys:
            new_angles[5] = self.clamp_angle(5, new_angles[5] - self.step_size)
            moved = True

        # 속도 조절: -/=
        if '-' in self.pressed_keys:
            self.step_index = max(0, self.step_index - 1)
            self.step_size = self.step_sizes[self.step_index]
            self.pressed_keys.discard('-')
            print(f"속도: {self.step_size}°/step")
        if '=' in self.pressed_keys:
            self.step_index = min(len(self.step_sizes) - 1, self.step_index + 1)
            self.step_size = self.step_sizes[self.step_index]
            self.pressed_keys.discard('=')
            print(f"속도: {self.step_size}°/step")

        if moved:
            self.move_robot(new_angles)
            self.current_angles = new_angles

        return moved

    def run(self):
        """메인 루프"""
        print("\n" + "="*60)
        print("RoArm M3 + Azure Kinect 데이터 수집")
        print("="*60)
        print("\n조작법:")
        print("  Q/A: Joint 1 (베이스)    W/S: Joint 2 (어깨)")
        print("  E/D: Joint 3 (팔꿈치)    R/F: Joint 4 (손목 피치)")
        print("  T/G: Joint 5 (손목 롤)   Y/H: Joint 6 (그리퍼)")
        print("\n데이터 수집:")
        print("  Space: 프레임 저장")
        print("  Enter: 에피소드 저장")
        print("  Backspace: 에피소드 취소")
        print("\n  -/=: 속도 감소/증가")
        print("  ESC: 종료")
        print("="*60 + "\n")

        # 초기 위치로 이동
        print("초기 위치로 이동 중...")
        self.robot.move_init()
        time.sleep(2)
        self.current_angles = self.get_robot_angles()
        print(f"현재 각도: {[f'{a:.1f}' for a in self.current_angles]}")

        # 키보드 리스너 시작
        listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        listener.start()

        # OpenCV 창 생성
        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera View", 960, 540)

        print("\n준비 완료! 키보드로 로봇을 조작하세요.\n")

        try:
            while self.running:
                # 텔레옵 처리
                self.process_teleop()

                # 카메라 프레임 가져오기
                rgb, depth = self.get_camera_frame()

                # Space가 눌리면 프레임 저장
                if self.is_recording:
                    self.save_frame(rgb, depth, self.current_angles)
                    self.is_recording = False

                # 화면에 정보 표시
                display = rgb.copy()

                # 상태 정보 표시
                info_lines = [
                    f"Episode: {self.episode_count} | Frames: {len(self.current_episode)}",
                    f"Speed: {self.step_size} deg/step",
                    f"Angles: {[f'{a:.0f}' for a in self.current_angles]}",
                    "",
                    "Space: Save frame | Enter: Save episode",
                    "Backspace: Cancel | ESC: Quit"
                ]

                y = 30
                for line in info_lines:
                    cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 2)
                    y += 25

                # 녹화 중 표시
                if len(self.current_episode) > 0:
                    cv2.circle(display, (display.shape[1] - 30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(display, "REC", (display.shape[1] - 70, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow("Camera View", display)

                # OpenCV 키 처리 (창 업데이트용)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.running = False

                time.sleep(0.02)  # ~50Hz

        except KeyboardInterrupt:
            print("\n중단됨")
        finally:
            # 정리
            listener.stop()
            cv2.destroyAllWindows()

            # 저장되지 않은 에피소드 확인
            if len(self.current_episode) > 0:
                print(f"\n저장되지 않은 {len(self.current_episode)} 프레임이 있습니다.")
                save = input("저장하시겠습니까? (y/n): ").strip().lower()
                if save == 'y':
                    self.save_episode()

            # 종료
            self.k4a.stop()
            self.robot.disconnect()
            print("\n정리 완료!")
            print(f"총 {self.episode_count} 에피소드 수집됨")
            print(f"저장 위치: {os.path.abspath(self.save_dir)}")


if __name__ == "__main__":
    collector = DataCollector(
        robot_port="/dev/ttyUSB0",
        save_dir="collected_data"
    )
    collector.run()
