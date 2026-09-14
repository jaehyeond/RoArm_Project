# Gate output capture control

- [x] C1: Gate runner must capture child stdout before it can certify task checks.
  CHECK: /bin/echo W12_CAPTURE_CONTROL
  EXPECT: W12_CAPTURE_CONTROL
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=0295630a4757/19 entries; EXPECT=matched; output-sha256=94d89a9101abe655342df5d2ebc9027c525379d4711843749335d374ee7cdaa7; output-bytes=20
