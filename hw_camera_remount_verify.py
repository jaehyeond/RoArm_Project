"""
hw_camera_remount_verify.py
===========================
카메라 재장착 후 위치 변화 검증 스크립트

A3 Hardware & Sensing Specialist

용도:
  카메라를 탈거/재장착 후 학습 데이터와 동일한 시점인지 확인.
  SmolVLA는 픽셀을 raw하게 학습하므로 카메라 포즈 변화 = OOD.

사용법:
  python hw_camera_remount_verify.py --mode check   # 빠른 시각 확인
  python hw_camera_remount_verify.py --mode aruco   # ArUco 마커로 정량적 측정
  python hw_camera_remount_verify.py --mode save_ref  # 현재를 기준 이미지로 저장

의존성:
  pip install opencv-contrib-python  (ArUco 모드 필요)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A

# ───────────────────────────────────────────────────────────────────────────────
# 상수
# ───────────────────────────────────────────────────────────────────────────────
REFERENCE_IMAGE_PATH = "hw_camera_reference.png"
DATASET_DIR = "collected_data"
PIXEL_SHIFT_WARNING_THRESHOLD = 10  # 224px 기준 — 이 이상이면 경고
PIXEL_SHIFT_CRITICAL_THRESHOLD = 20  # 224px 기준 — 이 이상이면 재수집 권장

# Azure Kinect 720P intrinsics (근사값, 실제는 pyk4a에서 읽기)
FX_APPROX = 607.0
FY_APPROX = 607.0
CX_APPROX = 638.0
CY_APPROX = 367.0


# ───────────────────────────────────────────────────────────────────────────────
# 카메라 유틸
# ───────────────────────────────────────────────────────────────────────────────
def start_kinect():
    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    time.sleep(1.0)  # 노출 안정화
    return k4a


def grab_frame(k4a) -> np.ndarray:
    """BGR 720P 프레임 반환 (np.ascontiguousarray 필수 — pyk4a BGRA 메모리 레이아웃)"""
    for _ in range(5):
        capture = k4a.get_capture()
        if capture.color is not None:
            bgra = capture.color
            bgr = np.ascontiguousarray(bgra[:, :, :3])
            return bgr
    raise RuntimeError("카메라 프레임을 가져올 수 없습니다.")


def to_model_size(img: np.ndarray) -> np.ndarray:
    """SmolVLA 입력 크기(224x224)로 리사이즈"""
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)


# ───────────────────────────────────────────────────────────────────────────────
# 모드 1: save_ref — 현재 프레임을 기준으로 저장
# ───────────────────────────────────────────────────────────────────────────────
def mode_save_ref():
    print("=== 기준 이미지 저장 모드 ===")
    print(f"저장 위치: {REFERENCE_IMAGE_PATH}")
    print("카메라가 올바른 위치에 있을 때 실행하세요.")
    print()

    k4a = start_kinect()
    print("라이브 피드 — 위치 확인 후 'S'로 저장, 'Q'로 종료")

    while True:
        frame = grab_frame(k4a)
        display = frame.copy()
        cv2.putText(display, "S: Save reference  Q: Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Save Reference", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('s'), ord('S')):
            cv2.imwrite(REFERENCE_IMAGE_PATH, frame)
            print(f"기준 이미지 저장됨: {REFERENCE_IMAGE_PATH}")
            break
        elif key in (ord('q'), ord('Q'), 27):
            print("저장 취소")
            break

    cv2.destroyAllWindows()
    k4a.stop()


# ───────────────────────────────────────────────────────────────────────────────
# 모드 2: check — 기준 이미지와 현재 이미지 시각 비교
# ───────────────────────────────────────────────────────────────────────────────
def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """간단한 SSIM 근사 (opencv 없이)"""
    i1 = img1.astype(float)
    i2 = img2.astype(float)
    mu1, mu2 = i1.mean(), i2.mean()
    sigma1 = i1.std()
    sigma2 = i2.std()
    sigma12 = ((i1 - mu1) * (i2 - mu2)).mean()
    c1, c2 = 6.5025, 58.5225
    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2)
    return float(num / den)


def estimate_shift_orb(ref: np.ndarray, cur: np.ndarray):
    """
    ORB 특징점으로 두 이미지 간 픽셀 변위 추정.
    반환: (dx_px, dy_px) — 224px 기준
    """
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(cur, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:50]  # 상위 50개

    if len(good) < 6:
        return None, None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # 중앙값 shift (회전/스케일 무시, 단순 병진 추정)
    shifts = pts2 - pts1
    dx = float(np.median(shifts[:, 0]))
    dy = float(np.median(shifts[:, 1]))
    return dx, dy, len(good)


def mode_check():
    print("=== 카메라 재장착 검증 모드 ===")

    # 기준 이미지 로드
    if not os.path.exists(REFERENCE_IMAGE_PATH):
        print(f"기준 이미지 없음: {REFERENCE_IMAGE_PATH}")
        print("먼저 --mode save_ref 로 기준 이미지를 저장하세요.")
        print()
        print("대안: 학습 데이터에서 첫 프레임 사용")

        ref_from_dataset = _load_ref_from_dataset()
        if ref_from_dataset is None:
            print(f"데이터셋도 없음 ({DATASET_DIR}/). 검증 불가.")
            return
        ref_full = ref_from_dataset
        print("학습 데이터 첫 프레임을 기준으로 사용합니다.")
    else:
        ref_full = cv2.imread(REFERENCE_IMAGE_PATH)
        print(f"기준 이미지 로드: {REFERENCE_IMAGE_PATH}")

    ref_224 = to_model_size(ref_full)
    ref_gray = cv2.cvtColor(ref_224, cv2.COLOR_BGR2GRAY)

    k4a = start_kinect()
    print()
    print("라이브 비교 중 — 'Q' 또는 ESC로 종료")
    print()
    print("해석 기준 (224px 기준):")
    print(f"  < {PIXEL_SHIFT_WARNING_THRESHOLD}px : OK (모델 허용 범위)")
    print(f"  {PIXEL_SHIFT_WARNING_THRESHOLD}~{PIXEL_SHIFT_CRITICAL_THRESHOLD}px : 주의 (성능 저하 가능)")
    print(f"  > {PIXEL_SHIFT_CRITICAL_THRESHOLD}px : 위험 (재수집 권장)")
    print()

    while True:
        cur_full = grab_frame(k4a)
        cur_224 = to_model_size(cur_full)
        cur_gray = cv2.cvtColor(cur_224, cv2.COLOR_BGR2GRAY)

        # SSIM (224px 기준)
        ssim_val = compute_ssim_simple(ref_gray.astype(float), cur_gray.astype(float))

        # 픽셀 변위 (ORB)
        dx, dy, n_matches = estimate_shift_orb(ref_gray, cur_gray)

        # 차이 이미지
        diff = cv2.absdiff(ref_224, cur_224)
        diff_enhanced = cv2.convertScaleAbs(diff, alpha=3.0)

        # Alpha blend 오버레이 (현재=파랑, 기준=빨강)
        overlay = np.zeros_like(ref_224)
        overlay[:, :, 2] = ref_gray  # 빨강 채널 = 기준
        overlay[:, :, 0] = cur_gray  # 파랑 채널 = 현재
        # 완벽 정렬 시 보라색, 차이 있으면 색분리

        # 4분할 디스플레이 구성
        top_left = ref_224.copy()
        cv2.putText(top_left, "REFERENCE", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        top_right = cur_224.copy()
        cv2.putText(top_right, "CURRENT", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        bottom_left = diff_enhanced
        cv2.putText(bottom_left, "DIFF (x3)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        bottom_right = overlay
        cv2.putText(bottom_right, "OVERLAY (R=ref, B=cur)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 수치 표시
        if dx is not None:
            shift_mag = (dx**2 + dy**2) ** 0.5
            if shift_mag < PIXEL_SHIFT_WARNING_THRESHOLD:
                color = (0, 255, 0)
                verdict = "OK"
            elif shift_mag < PIXEL_SHIFT_CRITICAL_THRESHOLD:
                color = (0, 200, 255)
                verdict = "CAUTION"
            else:
                color = (0, 0, 255)
                verdict = "DANGER — recollect"

            text = f"Shift: {shift_mag:.1f}px ({verdict})  dx={dx:.1f} dy={dy:.1f}  matches={n_matches}"
            cv2.putText(bottom_right, text, (5, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        else:
            cv2.putText(bottom_right, "ORB matching failed (low texture?)",
                        (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        ssim_text = f"SSIM: {ssim_val:.3f}  (>0.95=OK, >0.90=caution, <0.90=danger)"
        ssim_color = (0, 255, 0) if ssim_val > 0.95 else (0, 200, 255) if ssim_val > 0.90 else (0, 0, 255)
        cv2.putText(bottom_right, ssim_text, (5, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, ssim_color, 1)

        top = np.hstack([top_left, top_right])
        bottom = np.hstack([bottom_left, bottom_right])
        display = np.vstack([top, bottom])
        display = cv2.resize(display, (896, 448))  # 2x 확대

        cv2.imshow("Camera Remount Verification", display)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()
    k4a.stop()


def _load_ref_from_dataset() -> np.ndarray | None:
    """학습 데이터 첫 에피소드의 첫 프레임 로드"""
    if not os.path.exists(DATASET_DIR):
        return None
    episodes = sorted([
        d for d in os.listdir(DATASET_DIR)
        if d.startswith("episode_") and os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    if not episodes:
        return None

    ep_dir = os.path.join(DATASET_DIR, episodes[0])
    frames = sorted([f for f in os.listdir(ep_dir) if f.endswith(".jpg") or f.endswith(".png")])
    if not frames:
        return None

    img_path = os.path.join(ep_dir, frames[0])
    img = cv2.imread(img_path)
    if img is None:
        return None
    print(f"데이터셋 기준 프레임: {img_path}")
    return img


# ───────────────────────────────────────────────────────────────────────────────
# 모드 3: aruco — ArUco 마커로 정량적 pose 측정
# ───────────────────────────────────────────────────────────────────────────────
def mode_aruco():
    """
    ArUco 마커를 이용한 카메라 pose 측정.
    마커를 작업 공간에 고정 배치하고 재장착 전후 pose를 비교.

    필요: pip install opencv-contrib-python
    마커 생성: python -c "
        import cv2
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        img = cv2.aruco.generateImageMarker(d, 0, 300)
        cv2.imwrite('aruco_marker_0.png', img)
    "
    """
    print("=== ArUco 마커 Pose 측정 모드 ===")
    print()
    print("준비:")
    print("  1. aruco_marker_0.png 프린트 (A4, 10x10cm 목표)")
    print("  2. 작업 공간에 평평하게 고정 (테이프)")
    print("  3. 마커 실제 크기를 MARKER_SIZE_M 에 입력")
    print()

    MARKER_SIZE_M = 0.10  # 마커 실제 크기 (m) — 직접 측정해서 수정

    # opencv-contrib 체크
    try:
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
    except AttributeError:
        print("cv2.aruco 모듈 없음 — pip install opencv-contrib-python")
        return

    # 카메라 intrinsics (근사값 사용 — 실제로는 pyk4a로 읽어야 더 정확)
    camera_matrix = np.array([
        [FX_APPROX, 0, CX_APPROX],
        [0, FY_APPROX, CY_APPROX],
        [0, 0, 1]
    ], dtype=float)
    dist_coeffs = np.zeros((5, 1))

    k4a = start_kinect()
    print("마커를 향해 카메라를 배치 후 'S'로 현재 pose 저장, 'Q'로 종료")

    saved_poses = []
    labels = ["before", "after"]

    while True:
        frame = grab_frame(k4a)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            for i in range(len(ids)):
                marker_corners = corners[i].reshape(4, 2)
                obj_pts = np.array([
                    [-MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
                    [ MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
                    [ MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
                    [-MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
                ], dtype=float)

                _, rvec, tvec = cv2.solvePnP(obj_pts, marker_corners,
                                             camera_matrix, dist_coeffs)
                cv2.drawFrameAxes(display, camera_matrix, dist_coeffs,
                                  rvec, tvec, MARKER_SIZE_M * 0.5)

                # 위치/회전 표시
                t = tvec.flatten()
                r = rvec.flatten()
                r_deg = np.degrees(r)
                text = (f"ID={ids[i][0]}  "
                        f"t=({t[0]*1000:.1f},{t[1]*1000:.1f},{t[2]*1000:.1f})mm  "
                        f"r=({r_deg[0]:.1f},{r_deg[1]:.1f},{r_deg[2]:.1f})deg")
                cv2.putText(display, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if len(saved_poses) > 0:
                    prev_t = saved_poses[-1]["t"]
                    dt = np.linalg.norm(t - prev_t) * 1000
                    prev_r = saved_poses[-1]["r"]
                    dr = np.degrees(np.linalg.norm(r - prev_r))
                    delta_text = f"Delta from saved: {dt:.1f}mm  {dr:.2f}deg"
                    color = (0, 255, 0) if dt < 5 else (0, 200, 255) if dt < 20 else (0, 0, 255)
                    cv2.putText(display, delta_text, (10, 60 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(display, "마커 미감지 — 프린트 크기/조명 확인",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        label = labels[min(len(saved_poses), len(labels) - 1)]
        cv2.putText(display, f"S: Save '{label}' pose  Q: Quit  Saved: {len(saved_poses)}",
                    (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("ArUco Camera Pose", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('s'), ord('S')) and ids is not None and len(ids) > 0:
            i = 0
            marker_corners = corners[i].reshape(4, 2)
            obj_pts = np.array([
                [-MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
                [ MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
                [ MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
                [-MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
            ], dtype=float)
            _, rvec, tvec = cv2.solvePnP(obj_pts, marker_corners,
                                         camera_matrix, dist_coeffs)
            saved_poses.append({
                "label": label,
                "t": tvec.flatten().copy(),
                "r": rvec.flatten().copy(),
            })
            print(f"\n[{label}] pose 저장:")
            t = tvec.flatten()
            r = np.degrees(rvec.flatten())
            print(f"  위치: ({t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}) mm")
            print(f"  회전: ({r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}) deg")

        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()
    k4a.stop()

    # 결과 분석
    if len(saved_poses) >= 2:
        print("\n=== 재장착 오차 분석 ===")
        p0 = saved_poses[0]
        p1 = saved_poses[1]
        dt_mm = np.linalg.norm(p1["t"] - p0["t"]) * 1000
        dr_deg = np.degrees(np.linalg.norm(p1["r"] - p0["r"]))

        print(f"  병진 오차: {dt_mm:.1f} mm")
        print(f"  회전 오차: {dr_deg:.2f} deg")

        # 픽셀 영향 추정 (500mm 작업거리 기준)
        working_dist_mm = 500.0
        pixel_shift_full = (dt_mm / working_dist_mm) * FX_APPROX
        pixel_shift_224 = pixel_shift_full * (224.0 / 1280.0)

        print(f"\n  추정 픽셀 변위 (1280px 기준): {pixel_shift_full:.1f} px")
        print(f"  추정 픽셀 변위 (224px 기준):  {pixel_shift_224:.1f} px")

        if pixel_shift_224 < PIXEL_SHIFT_WARNING_THRESHOLD:
            print(f"\n  판정: OK — 재수집 불필요")
        elif pixel_shift_224 < PIXEL_SHIFT_CRITICAL_THRESHOLD:
            print(f"\n  판정: CAUTION — 소수의 추가 에피소드 수집 권장")
            print(f"         기존 74ep + 신규 20~30ep 혼합 테스트")
        else:
            print(f"\n  판정: DANGER — 카메라 재조정 또는 전체 재수집 필요")


# ───────────────────────────────────────────────────────────────────────────────
# 보조: 像질 분석 (노이즈, 흔들림 없는지)
# ───────────────────────────────────────────────────────────────────────────────
def mode_stability():
    """
    카메라 흔들림/진동 정도 측정.
    삼각대/클램프 고정이 충분한지 확인.
    10초간 프레임 간 차이를 측정.
    """
    print("=== 카메라 안정성 측정 ===")
    print("10초간 카메라를 고정 상태로 유지하세요.")
    print()

    k4a = start_kinect()
    diffs = []
    prev_gray = None
    n_frames = 0
    start_time = time.time()

    while time.time() - start_time < 10.0:
        frame = grab_frame(k4a)
        gray = cv2.cvtColor(to_model_size(frame), cv2.COLOR_BGR2GRAY).astype(float)

        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray - prev_gray)))
            diffs.append(diff)

        prev_gray = gray
        n_frames += 1

        remaining = 10.0 - (time.time() - start_time)
        cv2.putText(frame, f"Measuring stability... {remaining:.1f}s remaining",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Stability", frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    k4a.stop()

    if diffs:
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        print(f"\n결과 ({n_frames} frames):")
        print(f"  평균 프레임간 픽셀 차이: {mean_diff:.3f}")
        print(f"  최대 프레임간 픽셀 차이: {max_diff:.3f}")

        if mean_diff < 0.5:
            print("\n  판정: 안정적 (진동/흔들림 없음)")
        elif mean_diff < 2.0:
            print("\n  판정: 약간의 흔들림 (조임 상태 확인)")
        else:
            print("\n  판정: 심한 진동 — 클램프/삼각대 재조임 필요")
            print("         학습 중 카메라 이동 가능성 있음")


# ───────────────────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Azure Kinect 재장착 검증 스크립트 (A3 Hardware)"
    )
    parser.add_argument(
        "--mode",
        choices=["save_ref", "check", "aruco", "stability"],
        default="check",
        help=(
            "save_ref: 현재를 기준으로 저장 | "
            "check: 기준과 현재 비교 | "
            "aruco: ArUco 정량 측정 | "
            "stability: 카메라 고정 안정성 측정"
        ),
    )
    args = parser.parse_args()

    mode_map = {
        "save_ref": mode_save_ref,
        "check": mode_check,
        "aruco": mode_aruco,
        "stability": mode_stability,
    }
    mode_map[args.mode]()


if __name__ == "__main__":
    main()
