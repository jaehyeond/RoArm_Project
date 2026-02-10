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


class ManualDataCollector:
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
        return [0, 0, 0, 0, 0, 0]  # 실패시 기본값

    def set_torque(self, on: bool):
        """토크 ON/OFF 설정"""
        self.robot.torque_set(cmd=1 if on else 0)
        self.torque_on = on
        time.sleep(0.3)
        status = "ON" if on else "OFF"
        print(f"\n토크 {status}!")
        if not on:
            print("→ 이제 손으로 로봇을 자유롭게 움직일 수 있습니다.")

    def save_frame(self, rgb, depth, angles):
        """현재 프레임을 에피소드에 추가"""
        frame_data = {
            "timestamp": time.time(),
            "angles": angles.copy(),
            "frame_idx": len(self.current_episode)
        }

        self.current_episode.append({
            "data": frame_data,
            "rgb": rgb.copy(),
            "depth": depth.copy()
        })

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
            "fps": self.record_fps,
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

        print(f"\n{'='*50}")
        print(f"에피소드 {self.episode_count} 저장 완료!")
        print(f"  프레임 수: {len(self.current_episode)}")
        print(f"  저장 위치: {episode_dir}")
        print(f"{'='*50}\n")

        self.episode_count += 1
        self.current_episode = []
        self.is_recording = False

    def cancel_episode(self):
        """현재 에피소드 취소"""
        if len(self.current_episode) > 0:
            print(f"\n에피소드 취소됨 ({len(self.current_episode)} 프레임 삭제)")
            self.current_episode = []
            self.is_recording = False
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

                # 녹화 중이면 프레임 저장 (30 FPS)
                if self.is_recording:
                    if current_time - self.last_record_time >= 1.0 / self.record_fps:
                        self.save_frame(rgb, depth, angles)
                        self.last_record_time = current_time

                # 화면에 정보 표시
                display = rgb.copy()

                # 상태 정보
                torque_status = "ON" if self.torque_on else "OFF (Free Move)"
                rec_status = "RECORDING" if self.is_recording else "STANDBY"

                info_lines = [
                    f"Episode: {self.episode_count} | Frames: {len(self.current_episode)}",
                    f"Torque: {torque_status}",
                    f"Status: {rec_status}",
                    f"Angles: {[f'{a:.0f}' for a in angles]}",
                    "",
                    "Space: Start/Stop | Enter: Save | T: Torque"
                ]

                y = 30
                for line in info_lines:
                    color = (0, 0, 255) if self.is_recording else (0, 255, 0)
                    cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, color, 2)
                    y += 25

                # 녹화 중 표시
                if self.is_recording:
                    cv2.circle(display, (display.shape[1] - 30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(display, "REC", (display.shape[1] - 70, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # 프레임 카운터
                    cv2.putText(display, f"Frame: {len(self.current_episode)}",
                               (display.shape[1] - 120, 90),
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
