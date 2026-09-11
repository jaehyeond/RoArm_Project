# Gates: S1 real robot boot and measurement

Scope: User readiness authorizes the previously explained sequence: raw feedback, staged controller checks, and five measured scoop transitions. Each physical stage waits for its factual setup prerequisites, not repeat permission.

- [x] G1: Raw feedback from the attached robot is captured with explicit transmitted commands and SDK-independent gripper field inspection.
  CHECK: python3 claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/verify_feedback.py
  EXPECT: RAW_FEEDBACK_AUDIT_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=d8c1fc7bfaf5/19 entries; EXPECT=matched; output-sha256=04965a4f54ff3105cd7993901a5cb34327b299aa6b6713cddd5db1f1ef8083a1; output-bytes=22
- [x] G2: User confirms current geometry, clear motion workspace, and measurement readiness before dependent motion.
  EVIDENCE: User replied “그대로야” to the factual setup check; base38cm, pellet26cm, box top38.5cm, box x35cm retained. User later corrected that the first scale was not set up; first mass recorded as missing. User requested this repeat for weighing. Azure Kinect remains unobserved on USB.
- [x] G3: The approved shoulder P comparison has a measured outcome, bounded motion record, and a recorded default P16/I0 restoration command (register readback unavailable).
  CHECK: python3 claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/verify_pid.py
  EXPECT: PID_RAW_CSV_METRICS_RERUN_INSPECTION_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=d8c1fc7bfaf5/19 entries; EXPECT=matched; output-sha256=48e79edfba382427473b50715d3b2f85afb386c9d0d334b9a744e4b12d5f3d53; output-bytes=40
- [ ] G4: The approved closing-torque comparison has a measured reopen outcome and preserved command/feedback timeline.
  EVIDENCE: Fixed900_03 hardware18/18 explicit target checks and full3739-row Rerun verified/inspected. Matching790 trial remains pending factual post-photo robot/cup setup and residue-reset confirmation; the current mass reading has been recorded.
- [ ] G5: Five scoop masses are paired with actual initial conditions and angle records, with current-run-only statistics and required visualization evidence.
  EVIDENCE: Legacy900 mass not measured (scale not set up); fixed-jaw few-pellet residue reported. Fixed900_03 delivered mass19.95 g (gross29.60 minus empty cup9.65), residual approximately10–15 pellets per user. operator_measurement_01.json/CSV and original photos saved; residue mass and initial residue reset unknown. This is one exploratory measurement; five fixed-condition mass trials not yet complete.


- [ ] G6: The user-authorized single release tilt has a verified joint path and an actual paired before/after delivered-mass and residue observation, or an observed failure with no unvalidated follow-up motion.
  EVIDENCE: User “진행해” authorizes the proposed790→release-tilt→fixed-condition5 sequence. Post-photo physical setup answer pending. Conceptual outlet5deg is not an executable path.
- [ ] G7: The requested project commit is pushed to the verified origin/master and the remote commit hash matches the local commit, with required evidence accessible and local originals preserved.
  EVIDENCE: Repository /home/cgxr/Documents/Robotics/RoArm_Project; origin git@github.com:jaehyeond/RoArm_Project.git; master tracks origin/master; remote and local pre-work commit d41c258782da49492f0474b76d110454224597f3. Public repository, ADMIN permission. Oversize W10 RRD needs Git LFS; no push yet.

These are staged hardware/manual gates. No CHECK may initiate or repeat robot motion. Artifact checks will be defined after the observation schema is concrete.
