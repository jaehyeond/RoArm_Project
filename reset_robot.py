#!/usr/bin/env python3
"""Send reset commands to robot via serial."""

import serial
import time
import sys

def reset_robot(port: str):
    print(f"Connecting to {port}...")
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(0.5)

    # Command T:604 - Reset/clear settings
    print("[1] Sending T:604 (reset settings)...")
    ser.write(b'{"T":604}\n')
    time.sleep(0.5)
    response = ser.read_all()
    print(f"    Response: {response}")

    # Command T:600 - Get device info
    print("[2] Sending T:600 (device info)...")
    ser.write(b'{"T":600}\n')
    time.sleep(0.5)
    response = ser.read_all()
    print(f"    Response: {response}")

    # Command T:210 - Torque ON
    print("[3] Sending T:210 cmd:1 (torque ON)...")
    ser.write(b'{"T":210,"cmd":1}\n')
    time.sleep(0.5)
    response = ser.read_all()
    print(f"    Response: {response}")

    # Command T:603 - Move to init position
    print("[4] Sending T:603 (move to init)...")
    ser.write(b'{"T":603}\n')
    time.sleep(0.5)
    response = ser.read_all()
    print(f"    Response: {response}")

    ser.close()
    print("\nDone! If no movement, power cycle the robot and try again.")

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    reset_robot(port)
