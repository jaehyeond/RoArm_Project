"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import abc
import logging
import time
import threading

import numpy as np
import torch
from pynput import keyboard
from roarm_sdk.roarm import roarm
import roarm_sdk.common as sdk_common

# Suppress SDK background thread decode errors (SDK bug: readline returns None on timeout)
logging.getLogger('BaseController').setLevel(logging.CRITICAL)

# Monkey-patch SDK to suppress debug print statements (SDK line 320: print(data))
_original_process_received = sdk_common.DataProcessor._process_received

def _patched_process_received(self, data, genre):
    """Patched version that suppresses SDK's internal print(data) call."""
    if not data:
        return None
    # Replicate SDK logic without the print(data) statement
    switch_dict = {
        "roarm_m2": sdk_common.handle_m2_feedback,
        "roarm_m3": sdk_common.handle_m3_feedback
    }
    res = []
    valid_data = []
    if genre == sdk_common.JsonCmd.FEEDBACK_GET:
        valid_data.append(data['x'])
        valid_data.append(data['y'])
        valid_data.append(data['z'])
        if self.type in switch_dict:
            valid_data = switch_dict[self.type](valid_data, data)
    else:
        valid_data = data
    res.append(valid_data)
    return res

sdk_common.DataProcessor._process_received = _patched_process_received

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import RoarmRobotConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

# RoArm M3 joint limits (degrees) - from roarm_sdk/utils.py
JOINT_LIMITS = {
    0: (-190, 190),   # Joint 1 (base)
    1: (-110, 110),   # Joint 2 (shoulder)
    2: (-70, 190),    # Joint 3 (elbow) - NOTE: asymmetric!
    3: (-110, 110),   # Joint 4 (wrist pitch)
    4: (-190, 190),   # Joint 5 (wrist roll)
    5: (-10, 100),    # Joint 6 (gripper)
}

# Default keyboard teleop keys for 6-DOF arm
DEFAULT_TELEOP_KEYS = {
    # Joint 1 (base rotation)
    "joint1_pos": "q",
    "joint1_neg": "a",
    # Joint 2 (shoulder)
    "joint2_pos": "w",
    "joint2_neg": "s",
    # Joint 3 (elbow)
    "joint3_pos": "e",
    "joint3_neg": "d",
    # Joint 4 (wrist pitch)
    "joint4_pos": "r",
    "joint4_neg": "f",
    # Joint 5 (wrist roll)
    "joint5_pos": "t",
    "joint5_neg": "g",
    # Joint 6 (gripper)
    "joint6_pos": "y",
    "joint6_neg": "h",
    # Speed control
    "speed_up": "=",
    "speed_down": "-",
    # Quit
    "quit": "p",
}


def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    if not torch.allclose(goal_pos, safe_goal_pos):
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"  requested relative goal position target: {diff}\n"
            f"    clamped relative goal position target: {safe_diff}"
        )

    return safe_goal_pos


def safe_joints_angle_get(arm, max_retries=5, delay=0.1):
    """Safely get joint angles with retry logic.

    The SDK background thread can have intermittent decode errors,
    so we retry a few times before giving up.

    Returns: list of 6 joint angles in degrees, or raises RuntimeError
    """
    for attempt in range(max_retries):
        try:
            angles = arm.joints_angle_get()
            if angles is not None and len(angles) >= 6:
                return list(angles)
        except (KeyError, TypeError, AttributeError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed to read joints after {max_retries} attempts: {e}")
    raise RuntimeError(f"Failed to read joints: got None or invalid data")


def make_roarm_from_configs(configs: dict[str, str]) -> dict[str, roarm]:
    roarms = {}

    for key, port in configs.items():
        roarms[key] = roarm(roarm_type="roarm_m3", port=port, baudrate=115200)

    return roarms


# ============================================================================
# Teleop Strategy Pattern
# ============================================================================

class TeleopStrategy(abc.ABC):
    """Abstract base for teleop mode strategies.

    Each strategy encapsulates mode-specific logic for initialization,
    goal position generation, and cleanup. The robot instance is passed
    as a parameter to access shared hardware (follower_arms, leader_arms, etc.).
    """

    @abc.abstractmethod
    def initialize(self, robot: "RoarmRobot") -> None:
        """Mode-specific setup called at the end of robot.connect()."""

    @abc.abstractmethod
    def generate_goal_positions(self, robot: "RoarmRobot") -> dict[str, torch.Tensor]:
        """Generate goal positions and send motor commands for all follower arms.
        Returns {arm_name: goal_position_tensor} for data recording."""

    @abc.abstractmethod
    def cleanup(self, robot: "RoarmRobot") -> None:
        """Mode-specific teardown called at the start of robot.disconnect()."""

    @property
    @abc.abstractmethod
    def is_active(self) -> bool:
        """Whether the teleop session should continue (False = quit requested)."""


class KeyboardTeleopStrategy(TeleopStrategy):
    """Keyboard-based teleoperation: QWERTY keys control individual joints."""

    def __init__(self):
        self.teleop_keys = DEFAULT_TELEOP_KEYS
        self.pressed_keys = {f"joint{i}_pos": False for i in range(1, 7)}
        self.pressed_keys.update({f"joint{i}_neg": False for i in range(1, 7)})
        self.keyboard_listener = None
        self.step_size = 5  # degrees per step
        self.speed_levels = [2, 5, 10]
        self.speed_index = 1  # start at medium speed
        self.running = True
        self._motor_speed = 500  # moderate speed for teleop
        self._motor_acc = 200    # moderate acceleration

    @property
    def is_active(self) -> bool:
        return self.running

    def initialize(self, robot: "RoarmRobot") -> None:
        print("\n=== Keyboard Teleop Mode ===")
        print("Controls:")
        print("  Q/A: Joint 1 (base)")
        print("  W/S: Joint 2 (shoulder)")
        print("  E/D: Joint 3 (elbow)")
        print("  R/F: Joint 4 (wrist pitch)")
        print("  T/G: Joint 5 (wrist roll)")
        print("  Y/H: Joint 6 (gripper)")
        print("  -/=: Speed down/up")
        print("  P or ESC: Quit")
        print("============================\n")
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.keyboard_listener.start()

    def generate_goal_positions(self, robot: "RoarmRobot") -> dict[str, torch.Tensor]:
        follower_goal_pos = {}
        for name in robot.follower_arms:
            before_fread_t = time.perf_counter()
            present_pos = np.array(
                safe_joints_angle_get(robot.follower_arms[name]), dtype=np.float32
            )
            delta = self._get_keyboard_delta()
            goal_pos = torch.from_numpy(present_pos + delta)

            # Cap goal position for safety
            if robot.config.max_relative_target is not None:
                present_tensor = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(
                    goal_pos, present_tensor, robot.config.max_relative_target
                )

            # Clamp to joint limits
            goal_pos_np = goal_pos.numpy()
            for j in range(6):
                min_val, max_val = JOINT_LIMITS[j]
                goal_pos_np[j] = np.clip(goal_pos_np[j], min_val, max_val)
            goal_pos = torch.from_numpy(goal_pos_np)

            follower_goal_pos[name] = goal_pos

            goal_pos_list = goal_pos.numpy().astype(np.int32).tolist()
            robot.follower_arms[name].joints_angle_ctrl(
                angles=goal_pos_list, speed=self._motor_speed, acc=self._motor_acc
            )
            robot.logs[f"write_follower_{name}_goal_pos_dt_s"] = (
                time.perf_counter() - before_fread_t
            )
        return follower_goal_pos

    def cleanup(self, robot: "RoarmRobot") -> None:
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

    # --- Private keyboard event handlers ---

    def _on_press(self, key):
        try:
            for joint_key, char in self.teleop_keys.items():
                if hasattr(key, 'char') and key.char == char:
                    if joint_key in self.pressed_keys:
                        self.pressed_keys[joint_key] = True
                    elif joint_key == "speed_up":
                        self.speed_index = min(self.speed_index + 1, len(self.speed_levels) - 1)
                        self.step_size = self.speed_levels[self.speed_index]
                        print(f"Speed: {self.step_size} deg/step")
                    elif joint_key == "speed_down":
                        self.speed_index = max(self.speed_index - 1, 0)
                        self.step_size = self.speed_levels[self.speed_index]
                        print(f"Speed: {self.step_size} deg/step")
                    elif joint_key == "quit":
                        self.running = False
                    break
        except AttributeError:
            if key == keyboard.Key.esc:
                self.running = False

    def _on_release(self, key):
        try:
            for joint_key, char in self.teleop_keys.items():
                if hasattr(key, 'char') and key.char == char:
                    if joint_key in self.pressed_keys:
                        self.pressed_keys[joint_key] = False
                    break
        except AttributeError:
            pass

    def _get_keyboard_delta(self):
        """Get joint delta from pressed keys."""
        delta = np.zeros(6, dtype=np.float32)
        for i in range(1, 7):
            if self.pressed_keys.get(f"joint{i}_pos", False):
                delta[i - 1] += self.step_size
            if self.pressed_keys.get(f"joint{i}_neg", False):
                delta[i - 1] -= self.step_size
        return delta


class LeaderFollowerTeleopStrategy(TeleopStrategy):
    """Leader-follower teleoperation: physical leader arm mirrors to follower."""

    @property
    def is_active(self) -> bool:
        return True  # Runs until Ctrl+C

    def __init__(self):
        self._first_step = True  # Flag for gradual initial sync

    def initialize(self, robot: "RoarmRobot") -> None:
        # Step 1: Move BOTH arms to HOME position first (with torque ON)
        print("\n=== Leader-Follower Mode Setup ===")
        print("Step 1: Moving both arms to HOME position...")

        for name in robot.leader_arms:
            print(f"  Moving leader '{name}' to HOME...")
            robot.leader_arms[name].joints_angle_ctrl(
                angles=[0, 0, 0, 0, 0, 0], speed=0, acc=0
            )

        # Follower already moved to HOME in connect(), but ensure sync
        for name in robot.follower_arms:
            print(f"  Ensuring follower '{name}' at HOME...")
            result = robot.follower_arms[name].joints_angle_ctrl(
                angles=[0, 0, 0, 0, 0, 0], speed=0, acc=0
            )
            print(f"  Follower joints_angle_ctrl result: {result}")

        time.sleep(3)  # Wait for both arms to reach HOME

        # Verify positions
        for name in robot.leader_arms:
            try:
                angles = safe_joints_angle_get(robot.leader_arms[name])
                print(f"  Leader '{name}' at: {[round(a, 1) for a in angles]}")
            except RuntimeError:
                pass

        for name in robot.follower_arms:
            try:
                angles = safe_joints_angle_get(robot.follower_arms[name])
                print(f"  Follower '{name}' at: {[round(a, 1) for a in angles]}")
            except RuntimeError:
                pass

        # Step 2: User confirmation before disabling torque
        print("\n" + "=" * 50)
        print("*** IMPORTANT: Hold the LEADER arm with your hand! ***")
        print("*** Torque will be disabled - the arm will go limp! ***")
        print("=" * 50)
        input("Press Enter when you are holding the leader arm...")

        # Step 3: Disable leader torque
        for name in robot.leader_arms:
            print(f"  Disabling torque on leader '{name}'...")
            robot.leader_arms[name].torque_set(cmd=0)
            time.sleep(0.3)

        self._first_step = False  # Both arms already at HOME, no initial sync needed

        print("\n=== Leader-Follower Mode Ready ===")
        print("Move the LEADER arm by hand.")
        print("The FOLLOWER arm will mirror your movements.")
        print("Press Ctrl+C to stop.")
        print("==================================\n")

    def generate_goal_positions(self, robot: "RoarmRobot") -> dict[str, torch.Tensor]:
        # Read leader positions
        leader_pos = {}
        for name in robot.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = torch.from_numpy(
                np.array(safe_joints_angle_get(robot.leader_arms[name]), dtype=np.float32)
            )
            robot.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Mirror leader to follower
        follower_goal_pos = {}
        for name in robot.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name].clone()

            # CRITICAL: Clamp leader angles to follower joint limits
            # Leader arm may reach angles outside follower's valid range (e.g., elbow -80° vs limit -70°)
            # SDK will reject out-of-range angles with "Has invalid angles value" error
            goal_pos_np = goal_pos.numpy()
            for j in range(6):
                min_val, max_val = JOINT_LIMITS[j]
                goal_pos_np[j] = np.clip(goal_pos_np[j], min_val, max_val)
            goal_pos = torch.from_numpy(goal_pos_np)

            # Read current follower position for safety checks
            present_pos = torch.from_numpy(
                np.array(safe_joints_angle_get(robot.follower_arms[name]), dtype=np.float32)
            )

            # Cap goal position when too far away from present position
            if robot.config.max_relative_target is not None:
                goal_pos = ensure_safe_goal_position(
                    goal_pos, present_pos, robot.config.max_relative_target
                )

            follower_goal_pos[name] = goal_pos

            goal_pos_send = goal_pos.numpy().astype(np.int32).tolist()

            # SAFETY: First step or large delta → move slowly to prevent violent jumps
            max_delta = float(torch.max(torch.abs(goal_pos - present_pos)).item())
            if self._first_step or max_delta > 30.0:
                if self._first_step:
                    print(f"  Initial sync: moving follower to leader position (slow)...")
                robot.follower_arms[name].joints_angle_ctrl(
                    angles=goal_pos_send, speed=300, acc=100
                )
                if self._first_step:
                    time.sleep(1.5)  # Wait for initial sync to complete
                    self._first_step = False
            else:
                # Normal tracking — instant max speed
                robot.follower_arms[name].joints_angle_ctrl(
                    angles=goal_pos_send, speed=0, acc=0
                )

            robot.logs[f"write_follower_{name}_goal_pos_dt_s"] = (
                time.perf_counter() - before_fwrite_t
            )
        return follower_goal_pos

    def cleanup(self, robot: "RoarmRobot") -> None:
        # Re-enable torque on leader arms before disconnecting (safety)
        for name in robot.leader_arms:
            try:
                robot.leader_arms[name].torque_set(cmd=1)
                time.sleep(0.2)
            except Exception:
                pass  # Best effort - arm may already be disconnected
            robot.leader_arms[name].disconnect()


# ============================================================================
# Main Robot Class
# ============================================================================

class RoarmRobot:
    def __init__(
        self,
        config: RoarmRobotConfig,
    ):
        self.config = config
        self.robot_type = self.config.type
        self.leader_arms = make_roarm_from_configs(self.config.leader_arms)
        self.follower_arms = make_roarm_from_configs(self.config.follower_arms)
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}

        # Motor control parameters for send_action() (policy inference)
        self.motor_speed = 500
        self.motor_acc = 200

        # Select teleop strategy based on configuration
        self.teleop_strategy: TeleopStrategy = self._create_teleop_strategy()

    def _create_teleop_strategy(self) -> TeleopStrategy:
        """Factory: select teleop strategy based on whether leader arms are configured."""
        if len(self.leader_arms) == 0:
            return KeyboardTeleopStrategy()
        else:
            return LeaderFollowerTeleopStrategy()

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        return {
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["1", "2", "3", "4", "5", "6"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["1", "2", "3", "4", "5", "6"],
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "RoarmRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "RoarmRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Initialize follower arms - shared by both modes
        for name in self.follower_arms:
            print(f"Initializing {name}...")
            time.sleep(0.5)

            try:
                current_angles = safe_joints_angle_get(self.follower_arms[name])
                print(f"  Current position: {current_angles}")
            except RuntimeError as e:
                print(f"  Warning: Could not read initial position: {e}")
                current_angles = None

            print(f"  Moving to home position...")
            # NOTE: speed=0, acc=0 means "max speed" in Waveshare SDK
            result = self.follower_arms[name].joints_angle_ctrl(angles=[0, 0, 0, 0, 0, 0], speed=0, acc=0)
            print(f"  joints_angle_ctrl result: {result}")
            time.sleep(3.0)

            try:
                home_angles = safe_joints_angle_get(self.follower_arms[name])
                print(f"  At home position: {home_angles}")
            except RuntimeError as e:
                print(f"  Warning: Could not verify home position: {e}")

        # Connect cameras - shared by both modes
        for name in self.cameras:
            self.cameras[name].connect()

        # Delegate mode-specific initialization to strategy
        self.teleop_strategy.initialize(self)

        self.is_connected = True

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "RoarmRobot is not connected. You need to run `robot.connect()`."
            )

        # Check if quit was requested
        if not self.teleop_strategy.is_active:
            raise KeyboardInterrupt("Teleop quit requested")

        # Delegate goal position generation to strategy
        follower_goal_pos = self.teleop_strategy.generate_goal_positions(self)

        # Early exit when recording data is not requested
        if not record_data:
            return

        # TODO(rcadene): Add velocity and other info
        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = np.array(safe_joints_angle_get(self.follower_arms[name]), dtype=np.float32)
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "RoarmRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = np.array(safe_joints_angle_get(self.follower_arms[name]), dtype=np.float32)
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "RoarmRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 6
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = np.array(safe_joints_angle_get(self.follower_arms[name]), dtype=np.float32)
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.int32)
            goal_pos = goal_pos.tolist() if isinstance(goal_pos, (np.ndarray, torch.Tensor)) else goal_pos
            self.follower_arms[name].joints_angle_ctrl(angles=goal_pos, speed=self.motor_speed, acc=self.motor_acc)

        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "RoarmRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        # Delegate mode-specific cleanup to strategy
        self.teleop_strategy.cleanup(self)

        # Shared: disconnect followers
        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        # Shared: disconnect cameras
        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
