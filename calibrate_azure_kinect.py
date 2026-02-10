"""
Azure Kinect - RoArm 캘리브레이션 스크립트
카메라 좌표계 → 로봇 좌표계 변환 행렬 계산

사용법:
1. 로봇 그리퍼에 빨간색 마커(펜) 물리기
2. 스크립트 실행
3. 로봇을 4개 위치로 이동하며 'S' 키로 데이터 수집
4. 'C' 키로 변환 행렬 자동 계산 및 저장
"""

import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A
import time

# RoArm SDK
from roarm_sdk.roarm import roarm as RoArm


class AzureKinectCalibrator:
    def __init__(self, robot_port="/dev/ttyUSB0"):
        # Azure Kinect 초기화
        self.k4a = PyK4A(Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        ))

        # 로봇 연결
        print(f"로봇 연결 중... ({robot_port})")
        self.robot = RoArm(roarm_type="roarm_m3", port=robot_port)
        print("로봇 연결됨!")

        # 수집된 점들
        self.camera_points = []
        self.robot_points = []

        # 변환 행렬
        self.R = None
        self.t = None

        # 카메라 intrinsic (720p 기준 근사값)
        self.fx = 607.0
        self.fy = 607.0
        self.ppx = 638.0
        self.ppy = 367.0

    def start_camera(self):
        """카메라 시작"""
        self.k4a.start()
        print("Azure Kinect 시작됨")
        # 카메라 안정화 대기
        time.sleep(1)

    def stop_camera(self):
        """카메라 정지"""
        self.k4a.stop()
        print("Azure Kinect 정지됨")

    def get_frame(self):
        """RGB + Depth 프레임 획득"""
        capture = self.k4a.get_capture()
        rgb = capture.color[:, :, :3]  # BGRA -> BGR
        depth = capture.transformed_depth  # RGB에 정렬된 깊이
        return rgb, depth

    def get_robot_position(self):
        """
        로봇의 현재 End-Effector 위치 읽기 (pose_get)
        반환: [x, y, z] 미터 단위
        """
        pose = self.robot.pose_get()
        # pose = [x, y, z, wrist_tilt, wrist_roll, gripper]
        # x, y, z는 mm 단위로 반환됨 → 미터로 변환
        x = pose[0] / 1000.0
        y = pose[1] / 1000.0
        z = pose[2] / 1000.0
        return [x, y, z]

    def detect_marker(self, rgb, depth):
        """
        빨간색 마커 감지 및 3D 좌표 계산
        반환: (pixel_x, pixel_y, cam_x, cam_y, cam_z) 또는 None
        """
        # BGR -> HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        # 빨간색 범위 (HSV)
        # 빨간색은 H값이 0 근처와 180 근처 두 군데
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        mask = mask1 | mask2

        # 깊이 필터: 로봇 작업 영역만 (0.2m ~ 0.8m)
        depth_mask = (depth > 200) & (depth < 800)  # mm 단위
        mask = mask & depth_mask.astype(np.uint8) * 255

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 가장 큰 컨투어 선택
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        # 마커 크기 필터 (너무 작거나 너무 크면 무시)
        if area < 200 or area > 10000:
            return None

        # 중심점 계산
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # 깊이값 읽기 (주변 픽셀 평균)
        depth_roi = depth[max(0, cy-2):cy+3, max(0, cx-2):cx+3]
        valid_depths = depth_roi[depth_roi > 0]

        if len(valid_depths) == 0:
            return None

        z_mm = np.median(valid_depths)
        z = z_mm / 1000.0  # mm -> m

        # 깊이 범위 체크 (0.2m ~ 0.8m)
        if z < 0.2 or z > 0.8:
            return None

        # 카메라 좌표 계산 (pinhole model)
        x = (cx - self.ppx) * z / self.fx
        y = (cy - self.ppy) * z / self.fy

        return (cx, cy, x, y, z)

    def collect_point(self):
        """
        현재 프레임에서 마커 위치 감지하고 로봇 좌표와 함께 저장
        로봇 좌표는 pose_get()으로 자동 읽기
        """
        rgb, depth = self.get_frame()
        result = self.detect_marker(rgb, depth)

        if result is None:
            print("마커를 찾을 수 없습니다!")
            return False

        cx, cy, cam_x, cam_y, cam_z = result

        # 로봇 좌표 자동 읽기
        robot_pos = self.get_robot_position()

        self.camera_points.append([cam_x, cam_y, cam_z])
        self.robot_points.append(robot_pos)

        print(f"\n=== Point {len(self.camera_points)} 수집됨 ===")
        print(f"  픽셀: ({cx}, {cy})")
        print(f"  카메라 좌표: ({cam_x:.4f}, {cam_y:.4f}, {cam_z:.4f}) m")
        print(f"  로봇 좌표:   ({robot_pos[0]:.4f}, {robot_pos[1]:.4f}, {robot_pos[2]:.4f}) m")

        return True

    def delete_last_point(self):
        """마지막으로 수집한 점 삭제"""
        if len(self.camera_points) == 0:
            print("삭제할 점이 없습니다!")
            return False

        self.camera_points.pop()
        self.robot_points.pop()
        print(f"\n마지막 점 삭제됨. 현재 {len(self.camera_points)}개 점 보유")
        return True

    def clear_all_points(self):
        """모든 점 삭제"""
        self.camera_points = []
        self.robot_points = []
        print("\n모든 점 삭제됨!")

    def calculate_transform(self):
        """
        수집된 점들로 변환 행렬 계산 (SVD 기반)
        robot = R @ camera + t
        """
        if len(self.camera_points) < 3:
            print("최소 3개 점이 필요합니다!")
            return False

        cam = np.array(self.camera_points)
        rob = np.array(self.robot_points)

        # 중심점 계산
        cam_center = cam.mean(axis=0)
        rob_center = rob.mean(axis=0)

        # 중심 이동
        cam_centered = cam - cam_center
        rob_centered = rob - rob_center

        # SVD로 회전 행렬 계산
        H = cam_centered.T @ rob_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 반사 보정 (det(R) = -1인 경우)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 이동 벡터 계산
        t = rob_center - R @ cam_center

        self.R = R
        self.t = t

        # 오차 계산
        errors = []
        for i in range(len(cam)):
            predicted = R @ cam[i] + t
            error = np.linalg.norm(predicted - rob[i])
            errors.append(error)

        print("\n" + "=" * 50)
        print("=== 캘리브레이션 결과 ===")
        print("=" * 50)
        print(f"\n회전 행렬 R:\n{R}")
        print(f"\n이동 벡터 t: {t}")
        print(f"\n평균 오차: {np.mean(errors)*100:.2f} cm")
        print(f"최대 오차: {np.max(errors)*100:.2f} cm")

        if np.mean(errors)*100 < 2.0:
            print("\n✅ 캘리브레이션 성공! (오차 < 2cm)")
        else:
            print("\n⚠️ 오차가 큽니다. 더 많은 점을 수집하거나 다시 시도하세요.")

        return True

    def save_calibration(self, filepath="calibration.npz"):
        """캘리브레이션 결과 저장"""
        if self.R is None or self.t is None:
            print("캘리브레이션이 완료되지 않았습니다!")
            return False

        np.savez(filepath,
                 R=self.R,
                 t=self.t,
                 camera_points=np.array(self.camera_points),
                 robot_points=np.array(self.robot_points),
                 intrinsics=np.array([self.fx, self.fy, self.ppx, self.ppy]))

        print(f"\n캘리브레이션 저장됨: {filepath}")
        return True

    def camera_to_robot(self, cam_pos):
        """카메라 좌표 → 로봇 좌표 변환"""
        if self.R is None or self.t is None:
            raise ValueError("캘리브레이션이 완료되지 않았습니다!")
        return self.R @ np.array(cam_pos) + self.t


def interactive_calibration():
    """대화형 캘리브레이션 실행"""

    print("=" * 50)
    print("  Azure Kinect - RoArm 캘리브레이션")
    print("=" * 50)
    print("\n준비사항:")
    print("1. 로봇 그리퍼에 빨간색 펜을 물린 상태")
    print("2. Azure Kinect가 로봇 작업 영역을 볼 수 있게 배치")
    print("3. 최소 4개 점에서 데이터 수집 필요")
    print("\n조작법:")
    print("  S: 현재 위치 저장 (로봇 좌표 자동 읽기)")
    print("  D: 마지막 점 삭제 (잘못 찍었을 때)")
    print("  R: 모든 점 삭제 (처음부터 다시)")
    print("  C: 캘리브레이션 계산 및 저장")
    print("  Q/ESC: 종료\n")

    input("준비되면 Enter를 누르세요...")

    calibrator = AzureKinectCalibrator(robot_port="/dev/ttyUSB0")
    calibrator.start_camera()

    print("\n카메라 미리보기 시작...")
    print("빨간 마커가 초록색 원으로 표시되면 'S' 키로 저장")

    collecting = True
    while collecting:
        rgb, depth = calibrator.get_frame()
        display = rgb.copy()

        # 마커 감지
        result = calibrator.detect_marker(rgb, depth)
        if result:
            cx, cy, cam_x, cam_y, cam_z = result
            # 마커 위치 표시 (초록색 원)
            cv2.circle(display, (cx, cy), 15, (0, 255, 0), 3)
            # 카메라 좌표 표시
            cv2.putText(display, f"Cam: ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})m",
                       (cx + 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 마커 없음 경고
            cv2.putText(display, "No marker detected!",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 상태 정보 표시
        cv2.putText(display, f"Points: {len(calibrator.camera_points)}/4+",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, "S:Save D:Delete R:Reset C:Calc Q:Quit",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') or key == ord('S'):
            # 자동으로 로봇 좌표 읽기 + 카메라 좌표 저장
            calibrator.collect_point()

        elif key == ord('d') or key == ord('D'):
            # 마지막 점 삭제
            calibrator.delete_last_point()

        elif key == ord('r') or key == ord('R'):
            # 모든 점 삭제
            calibrator.clear_all_points()

        elif key == ord('c') or key == ord('C'):
            if calibrator.calculate_transform():
                calibrator.save_calibration("E:/RoArm_Project/calibration.npz")

        elif key == ord('q') or key == ord('Q') or key == 27:  # ESC
            collecting = False

    cv2.destroyAllWindows()
    calibrator.stop_camera()

    print("\n캘리브레이션 종료")


def load_calibration(filepath="E:/RoArm_Project/calibration.npz"):
    """저장된 캘리브레이션 로드"""
    data = np.load(filepath)
    R = data['R']
    t = data['t']

    def camera_to_robot(cam_pos):
        return R @ np.array(cam_pos) + t

    return R, t, camera_to_robot


if __name__ == "__main__":
    interactive_calibration()
