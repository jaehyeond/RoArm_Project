#!/usr/bin/env python3
"""Scan for connected servos."""

import serial
import time
import sys

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"

print(f"Scanning servos on {port}...")
ser = serial.Serial(port, 115200, timeout=2)
time.sleep(1)

# Try different scan/info commands
commands = [
    ('T:106', b'{"T":106}\n'),      # Servo scan
    ('T:600', b'{"T":600}\n'),      # Device info
    ('T:105', b'{"T":105}\n'),      # Read all servo positions
]

for name, cmd in commands:
    print(f"\n[{name}] Sending...")
    ser.write(cmd)
    time.sleep(0.8)
    response = ser.read_all()
    if response:
        print(f"  Response: {response.decode(errors='ignore')}")
    else:
        print(f"  NO RESPONSE")

ser.close()
print("\nDone.")
