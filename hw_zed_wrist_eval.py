"""
ZED Mini wrist 카메라 위치 적합성 평가
- Azure Kinect (외부) + ZED Mini (wrist) 동시 캡처
- 두 카메라 뷰를 나란히 표시하여 시각적 비교
- 이미지 저장하여 SigLIP 테스트용

사용법: python hw_zed_wrist_eval.py
  Space: 스냅샷 저장
  ESC: 종료
"""

import os
import time
import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A

SAVE_DIR = "zed_eval_snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)


def init_azure_kinect():
    """Azure Kinect 초기화"""
    print("Azure Kinect 초기화...")
    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    # 첫 몇 프레임 버리기 (자동노출 안정화)
    for _ in range(10):
        k4a.get_capture()
    print("Azure Kinect OK")
    return k4a


def init_zed_mini():
    """ZED Mini 초기화 (depth=NONE, sensors 불필요, 수동 노출)"""
    print("ZED Mini 초기화...")
    import pyzed.sl as sl

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.sensors_required = False
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ZED Mini open 실패: {status}")
        return None, None

    # 수동 노출 설정 (auto-exposure가 검은 팔 때문에 실패하므로)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 60)

    zed_image = sl.Mat()

    # 첫 몇 프레임 버리기 (노출 안정화)
    for _ in range(30):
        zed.grab()

    print("ZED Mini OK (manual exposure=50, gain=60)")
    return zed, zed_image


def capture_kinect(k4a):
    """Azure Kinect RGB 캡처"""
    capture = k4a.get_capture()
    if capture.color is None:
        return None
    rgb = np.ascontiguousarray(capture.color[:, :, :3])  # BGRA -> BGR
    return rgb


def capture_zed(zed, zed_image):
    """ZED Mini RGB 캡처"""
    import pyzed.sl as sl
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(zed_image, sl.VIEW.LEFT)
        rgb = np.ascontiguousarray(zed_image.get_data()[:, :, :3])  # BGRA -> BGR
        return rgb
    return None


def compute_image_stats(img):
    """이미지 기본 통계"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def draw_info(img, label, stats, position="top"):
    """이미지에 정보 오버레이"""
    h, w = img.shape[:2]
    overlay = img.copy()

    y_start = 10 if position == "top" else h - 100

    cv2.putText(overlay, label, (10, y_start + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(overlay, f"Bright: {stats['brightness']:.0f}  Contrast: {stats['contrast']:.0f}",
                (10, y_start + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"Sharpness: {stats['sharpness']:.0f}",
                (10, y_start + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return overlay


def main():
    k4a = init_azure_kinect()
    zed, zed_image = init_zed_mini()

    if zed is None:
        print("ZED Mini 연결 실패! USB-C 방향 확인 (한쪽만 USB 3.0)")
        k4a.stop()
        return

    snapshot_count = 0
    print("\n=== 듀얼 카메라 실시간 뷰 ===")
    print("Space: 스냅샷 저장 | ESC: 종료")
    print("물체(스펀지)를 작업대 위에 놓고 다양한 위치로 옮겨보세요.\n")

    while True:
        kinect_img = capture_kinect(k4a)
        zed_img = capture_zed(zed, zed_image)

        if kinect_img is None or zed_img is None:
            continue

        # 통계 계산
        k_stats = compute_image_stats(kinect_img)
        z_stats = compute_image_stats(zed_img)

        # 두 이미지를 같은 크기로 (720p)
        kinect_disp = cv2.resize(kinect_img, (640, 360))
        zed_disp = cv2.resize(zed_img, (640, 360))

        # 정보 오버레이
        kinect_disp = draw_info(kinect_disp, "Azure Kinect (External)", k_stats)
        zed_disp = draw_info(zed_disp, "ZED Mini (Wrist)", z_stats)

        # 나란히 표시
        combined = np.hstack([kinect_disp, zed_disp])

        # 하단 안내
        cv2.putText(combined, "SPACE=snapshot  ESC=quit  |  Evaluate: gripper visible? workspace covered? occlusion?",
                    (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        cv2.imshow("Dual Camera Eval", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            snapshot_count += 1
            prefix = f"{SAVE_DIR}/snap_{snapshot_count:03d}"
            cv2.imwrite(f"{prefix}_kinect.jpg", kinect_img)
            cv2.imwrite(f"{prefix}_zed.jpg", zed_img)
            # 원본 해상도 합성도 저장
            kinect_full = draw_info(kinect_img.copy(), "Kinect External", k_stats)
            zed_full = draw_info(zed_img.copy(), "ZED Wrist", z_stats)
            combined_full = np.hstack([kinect_full, zed_full])
            cv2.imwrite(f"{prefix}_combined.jpg", combined_full)
            print(f"[Snapshot {snapshot_count}] saved to {prefix}_*.jpg")
            print(f"  Kinect: bright={k_stats['brightness']:.0f} contrast={k_stats['contrast']:.0f} sharp={k_stats['sharpness']:.0f}")
            print(f"  ZED:    bright={z_stats['brightness']:.0f} contrast={z_stats['contrast']:.0f} sharp={z_stats['sharpness']:.0f}")

    # 정리
    k4a.stop()
    zed.close()
    cv2.destroyAllWindows()

    print(f"\n총 {snapshot_count}개 스냅샷 저장됨 → {SAVE_DIR}/")
    if snapshot_count > 0:
        print("\n다음 단계: python hw_zed_wrist_eval.py 로 SigLIP 임베딩 비교")
        print("  - 스냅샷을 확인하고 다음을 평가하세요:")
        print("  1. 그리퍼가 ZED 뷰에서 보이는가?")
        print("  2. 작업대 + 물체가 ZED 뷰에 포함되는가?")
        print("  3. 손으로 로봇 움직일 때 ZED 뷰가 심하게 흔들리는가?")
        print("  4. 조명 조건에서 ZED 이미지 품질이 충분한가?")


if __name__ == "__main__":
    main()
