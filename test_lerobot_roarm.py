"""
Test LeRobot RoArm M3 connection
"""
from lerobot.common.robot_devices.robots.configs import RoarmRobotConfig
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

# Linux: /dev/ttyUSB0 (check with: ls /dev/ttyUSB*)
ROBOT_PORT = "/dev/ttyUSB0"  # RoArm M3 via Silicon Labs CP210x
CAMERA_INDEX = 0     # IMX335 camera

def test_config():
    """Test RoArm config creation"""
    print("=== Testing RoarmRobotConfig ===")

    # Create config with Windows port
    config = RoarmRobotConfig(
        leader_arms={},  # No leader arm for keyboard teleop
        follower_arms={
            "main": ROBOT_PORT
        },
        cameras={
            "main": OpenCVCameraConfig(
                camera_index=CAMERA_INDEX,
                fps=30,
                width=640,
                height=480,
            )
        }
    )

    print(f"Robot type: {config.type}")
    print(f"Follower arms: {config.follower_arms}")
    print(f"Cameras: {config.cameras}")
    print("Config created successfully!")
    return config

def test_robot_connection(config):
    """Test actual robot connection"""
    print("\n=== Testing Robot Connection ===")

    from lerobot.common.robot_devices.robots.roarm_m3 import RoarmRobot

    robot = RoarmRobot(config)
    print(f"Features: {robot.features}")

    try:
        print("Connecting to robot...")
        robot.connect()
        print("Connected!")

        # Read current position
        obs = robot.capture_observation()
        print(f"Current state: {obs['observation.state']}")

        # Disconnect
        robot.disconnect()
        print("Disconnected successfully!")
        return True

    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def test_camera_only():
    """Test camera without robot"""
    print("\n=== Testing Camera Only ===")
    import cv2

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera working! Frame shape: {frame.shape}")
        cap.release()
        return True
    else:
        print("Camera not available")
        return False

if __name__ == "__main__":
    print("LeRobot RoArm M3 Test")
    print("=" * 50)

    # Test 1: Camera
    cam_ok = test_camera_only()

    # Test 2: Config
    config = test_config()

    # Test 3: Robot connection (only if robot is connected)
    print("\n" + "=" * 50)
    user_input = input("Is the robot connected and powered on? (y/n): ")
    if user_input.lower() == 'y':
        test_robot_connection(config)
    else:
        print("Skipping robot connection test")

    print("\n=== Test Complete ===")
