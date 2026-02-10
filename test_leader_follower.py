"""
RoArm M3 Leader-Follower Test Script

Tests:
1. Both arms connect via USB serial
2. Leader arm torque is disabled (free movement)
3. Follower mirrors leader position in real-time

Usage:
    python test_leader_follower.py

Before running:
    - Connect both RoArm M3 arms via USB
    - Check ports: ls /dev/ttyUSB*
    - Update LEADER_PORT and FOLLOWER_PORT below
"""
import sys
import time

# Suppress SDK noise
import logging
logging.getLogger('BaseController').setLevel(logging.CRITICAL)
import roarm_sdk.common as sdk_common

_original_process_received = sdk_common.DataProcessor._process_received
def _patched_process_received(self, data, genre):
    if not data:
        return None
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

from roarm_sdk.roarm import roarm

# ============================================
# CONFIGURE THESE PORTS
# ============================================
LEADER_PORT = "/dev/ttyUSB1"    # Leader arm - 그리퍼 클램프 있는 팔 (손으로 움직임)
FOLLOWER_PORT = "/dev/ttyUSB0"  # Follower arm - 그리퍼 클램프 없는 팔 (미러링)
# ============================================

# Joint limits for RoArm M3 (degrees)
JOINT_LIMITS = [
    (-190, 190),   # Joint 0: Base rotation
    (-110, 110),   # Joint 1: Shoulder
    (-70, 190),    # Joint 2: Elbow (asymmetric!)
    (-110, 110),   # Joint 3: Wrist pitch
    (-190, 190),   # Joint 4: Wrist roll
    (-10, 100),    # Joint 5: Gripper
]


def clamp_angles(angles):
    """Clamp angles to joint limits to prevent SDK validation errors."""
    clamped = []
    for i, angle in enumerate(angles):
        min_val, max_val = JOINT_LIMITS[i]
        clamped.append(max(min_val, min(max_val, angle)))
    return clamped


def safe_get_angles(arm, retries=5):
    for attempt in range(retries):
        try:
            angles = arm.joints_angle_get()
            if angles is not None and len(angles) >= 6:
                return list(angles)
        except Exception:
            time.sleep(0.1)
    return None


def test_connection():
    """Step 1: Test that both arms connect."""
    print("=" * 50)
    print("STEP 1: Testing connections")
    print("=" * 50)

    print(f"\nConnecting Leader arm on {LEADER_PORT}...")
    try:
        leader = roarm(roarm_type="roarm_m3", port=LEADER_PORT, baudrate=115200)
        time.sleep(0.5)
        angles = safe_get_angles(leader)
        print(f"  Leader connected! Position: {angles}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Check that Leader is connected to {LEADER_PORT}")
        return None, None

    print(f"\nConnecting Follower arm on {FOLLOWER_PORT}...")
    try:
        follower = roarm(roarm_type="roarm_m3", port=FOLLOWER_PORT, baudrate=115200)
        time.sleep(0.5)
        angles = safe_get_angles(follower)
        print(f"  Follower connected! Position: {angles}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Check that Follower is connected to {FOLLOWER_PORT}")
        leader.disconnect()
        return None, None

    print("\nBoth arms connected successfully!")
    return leader, follower


def test_torque_off(leader, follower):
    """Step 2: Move BOTH arms to home, then disable leader torque."""
    print("\n" + "=" * 50)
    print("STEP 2: Safe torque disable sequence")
    print("=" * 50)

    # SAFETY: Move BOTH arms to home position BEFORE disabling torque
    print("Moving BOTH arms to HOME position...")
    leader.joints_angle_ctrl(angles=[0, 0, 0, 0, 0, 0], speed=500, acc=200)
    follower.joints_angle_ctrl(angles=[0, 0, 0, 0, 0, 0], speed=500, acc=200)
    time.sleep(3)
    print("  Both arms at HOME position.")

    print("\n*** WARNING: HOLD the leader arm with your hand before proceeding! ***")
    print("*** Torque will be disabled — the arm will go limp! ***")
    input("Hold the arm firmly, then press Enter...")

    print("Disabling leader arm torque...")
    leader.torque_set(cmd=0)
    time.sleep(0.3)
    print("Leader arm torque is OFF.")
    print("You can now move the leader arm freely by hand.")
    print("Keep holding it — do NOT let go at an extended position!")
    input("Press Enter when ready for mirroring (keep holding the arm)...")


def test_mirroring(leader, follower):
    """Step 3: Real-time mirroring test with safe startup."""
    print("\n" + "=" * 50)
    print("STEP 3: Leader-Follower mirroring test")
    print("=" * 50)

    # SAFETY: Gradual initial sync — move follower to leader's current position SLOWLY
    print("Syncing follower to leader position (slow)...")
    leader_angles = safe_get_angles(leader)
    if leader_angles is None:
        print("  ERROR: Cannot read leader position!")
        return
    print(f"  Leader at: {[round(a, 1) for a in leader_angles]}")
    clamped_leader = clamp_angles(leader_angles)
    follower.joints_angle_ctrl(angles=clamped_leader, speed=300, acc=100)
    time.sleep(2)
    print("  Follower synced to leader position.")

    print("\nMove the LEADER arm by hand.")
    print("The FOLLOWER arm should mirror your movements.")
    print("Press Ctrl+C to stop.\n")

    try:
        step = 0
        prev_angles = leader_angles
        while True:
            # Read leader position
            cur_angles = safe_get_angles(leader)
            if cur_angles is None:
                continue

            # CRITICAL: Clamp leader angles to joint limits before sending to follower
            clamped_angles = clamp_angles(cur_angles)

            # SAFETY: Check max delta per joint — use slower speed for large jumps
            max_delta = max(abs(clamped_angles[i] - prev_angles[i]) for i in range(6))
            if max_delta > 30:
                # Large jump detected — move slowly to prevent violent motion
                follower.joints_angle_ctrl(angles=clamped_angles, speed=300, acc=100)
            else:
                # Normal tracking — instant max speed
                follower.joints_angle_ctrl(angles=clamped_angles, speed=0, acc=0)

            prev_angles = cur_angles
            step += 1
            if step % 10 == 0:
                print(f"  Step {step}: Leader={[round(a, 1) for a in cur_angles]}")

            time.sleep(0.02)  # ~50Hz control loop

    except KeyboardInterrupt:
        print("\n\nMirroring stopped.")


def cleanup(leader, follower):
    """Re-enable torque and disconnect."""
    print("\nCleaning up...")
    try:
        # Move follower to home
        print("Moving follower to home position...")
        follower.joints_angle_ctrl(angles=[0, 0, 0, 0, 0, 0], speed=500, acc=200)
        time.sleep(2)

        # Re-enable leader torque
        print("Re-enabling leader torque...")
        leader.torque_set(cmd=1)
        time.sleep(0.3)

        # Disconnect
        leader.disconnect()
        follower.disconnect()
        print("Disconnected both arms.")
    except Exception as e:
        print(f"Cleanup error: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("RoArm M3 Leader-Follower Test")
    print("=" * 50)
    print(f"Leader port:   {LEADER_PORT}")
    print(f"Follower port: {FOLLOWER_PORT}")
    print()

    leader, follower = test_connection()
    if leader is None:
        print("\nConnection failed. Exiting.")
        sys.exit(1)

    try:
        test_torque_off(leader, follower)
        test_mirroring(leader, follower)
    finally:
        cleanup(leader, follower)

    print("\nTest complete!")
