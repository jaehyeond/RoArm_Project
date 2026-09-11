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
- [x] G4: The approved closing-torque comparison has a measured reopen outcome and preserved command/feedback timeline.
  EVIDENCE: torque790_01 completed71.812934841s with3905 feedback rows,4.218749987→4.130859350deg, peak additional opening0. Corrected900 peak additional opening0. Actual door-target audit PASS and3905-row Rerun readback/visual inspection recorded. torque_comparison_01.json records sequential-pile limitations; no causal torque or mass superiority claim. Current gross22.28g minus estimated cup0.05g gives estimated delivered22.23g.
- [ ] G5: Five scoop masses are paired with actual initial conditions and angle records, with current-run-only statistics and required visualization evidence.
  EVIDENCE: Legacy900 mass not measured (scale not set up); fixed-jaw few-pellet residue reported. Fixed900_03 delivered mass19.95 g (gross29.60 minus empty cup9.65), residual approximately10–15 pellets per user. operator_measurement_01.json/CSV and original photos saved; residue mass and initial residue reset unknown. This is one exploratory measurement; five fixed-condition mass trials not yet complete.


- [ ] G6: The user-authorized single release tilt has a verified joint path and an actual paired before/after delivered-mass and residue observation, or an observed failure with no unvalidated follow-up motion.
  EVIDENCE: outlet_tilt_01 completed the physical perturbation attempt:64/64 planned T122 issued,2935 rows,58.143003s, actual outlet slope7.762060→22.001692deg. Final shoulder tracking error5.014578deg exceeded unchanged5deg gate; result.completed=false and no later actuation. Path704 samples and actual2935-row RRD/inspection PASS. This is a documented controller failure with a useful directional result; the overall discharge criterion remains unchecked while post-mass/residue is missing. Do not mark a final discharge method validated.
- [x] G7: The requested project commit is pushed to the verified origin/master and the remote commit hash matches the local commit, with required evidence accessible and local originals preserved.
  LATEST: outlet_tilt artifact d497fbc61dc659402f84f7e1d14cd722bb76ddc3 pushed to verified origin/master; local/remote hash equality,41 committed original hashes and LFS fsck PASS. See outlet_tilt_publish_01.json.
  EVIDENCE: Artifact commit52cc20cd7af500e92fbfb4692301839cf4d51527 pushed rc0 to git@github.com:jaehyeond/RoArm_Project.git master; independent git ls-remote and local HEAD matched. LFS14/14 originals uploaded, post-commit git lfs fsck PASS. Original1008 staged byte checks and canonical manifest88 entries PASS; final artifact commit1013 files includes audit reports. Details git_publish_01/push_result_01.json. External pile/W9 MP4 exclusions are explicit in README. This verification is published in a following ordinary documentation commit; no force push.

These are staged hardware/manual gates. No CHECK may initiate or repeat robot motion. Artifact checks will be defined after the observation schema is concrete.
