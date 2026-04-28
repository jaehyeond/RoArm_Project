#!/usr/bin/env python3
"""Phase 0-4: Dual camera + robot simultaneous test"""
import time
import numpy as np

# 1. Robot (SDK print spam 억제 — roarm 클래스의 _process_received 패치)
import logging
logging.getLogger('BaseController').setLevel(logging.CRITICAL)

from roarm_sdk.roarm import roarm
from roarm_sdk.common import handle_m3_feedback, JsonCmd

_orig_process = roarm._process_received
def _silent_process(self, data, genre):
    if not data:
        return None
    # print(data) 제거 — 나머지 로직 동일
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

# LEGACY pre-L-F (2026-04-01): USB0=Leader (gripper clamp). Edit to /dev/ttyUSB1 for Follower.
arm = roarm(roarm_type='roarm_m3', port='/dev/ttyUSB0', baudrate=115200)
time.sleep(0.5)
print("[1/3] Robot connected")

# 2. Azure Kinect
import pyk4a
from pyk4a import Config, PyK4A
k4a = PyK4A(Config(
    color_resolution=pyk4a.ColorResolution.RES_720P,
    depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
    synchronized_images_only=True,
))
k4a.start()
for _ in range(3):
    k4a.get_capture()
print("[2/3] Azure Kinect started")

# 3. ZED Mini
import pyzed.sl as sl
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NONE
init_params.sensors_required = False
status = None
for attempt in range(3):
    status = zed.open(init_params)
    if status == sl.ERROR_CODE.SUCCESS:
        break
    print(f"ZED open attempt {attempt+1}/3: {status}")
    zed.close()
    time.sleep(2)
    zed = sl.Camera()  # 재생성

if status != sl.ERROR_CODE.SUCCESS:
    print(f"ZED FAILED after 3 attempts: {status}")
    print("Continuing with single camera only")
    zed = None
else:
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)
    zed_image = sl.Mat()
    for _ in range(15):
        zed.grab()
    print("[3/3] ZED Mini started")

# Simultaneous capture test
print("\n=== Simultaneous capture test (5 rounds) ===")
for i in range(5):
    t0 = time.time()

    # Robot
    angles = arm.joints_angle_get()

    # Kinect
    cap = k4a.get_capture()
    kinect_rgb = np.ascontiguousarray(cap.color[:, :, :3])

    # ZED
    if zed is not None:
        zed.grab()
        zed.retrieve_image(zed_image, sl.VIEW.LEFT)
        zed_rgb = np.ascontiguousarray(zed_image.get_data()[:, :, :3])
        zed_mean = zed_rgb.mean()
    else:
        zed_mean = -1

    dt = (time.time() - t0) * 1000
    angles_valid = angles is not None and len(angles) == 6 and angles[0] != 180

    print(f"Round {i+1}: {dt:.0f}ms | Joints={angles_valid} "
          f"| Kinect={kinect_rgb.mean():.0f} | ZED={zed_mean:.0f}")

# Cleanup
k4a.stop()
if zed is not None:
    zed.close()
arm.disconnect()
print("\n=== ALL DEVICES OK ===")
