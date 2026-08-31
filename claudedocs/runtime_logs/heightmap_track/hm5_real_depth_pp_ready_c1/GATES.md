# Gates: Kinect real-depth to roarm-heightmap-v1 PP readiness

OWNS: roarm_rl/heightmap.py, sim_scripts/p38_hm5_kinect_depth_pipeline.py, claudedocs/runtime_logs/heightmap_track/hm5_real_depth_pp_ready/**, claudedocs/runtime_logs/heightmap_track/hm5_real_depth_pp_ready_c1/**

Scope: Accept an aligned Azure Kinect depth frame, reject invalid and edge-flight pixels, preserve unseen cells as invalid, and emit a fixed-contract heightmap plus complete D341 evidence.

- [x] G0: this ledger states outcome checks that can fail
  CHECK: node /home/cgxr/.codex/skills/unlazy/scripts/gate-lint.mjs claudedocs/runtime_logs/heightmap_track/hm5_real_depth_pp_ready_c1/GATES.md
  EXPECT: LINT OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=4a679307fbd5/37 entries; EXPECT=matched; output-sha256=48630b7361dd44ee870917b12c3d19b9d7bdea738aaca16bb04d4cab83b772d2; output-bytes=8

- [x] G1: the host audit records pyk4a and libk4a availability separately from physical device presence
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_scripts/p38_hm5_kinect_depth_pipeline.py --validate-runtime-audit claudedocs/runtime_logs/heightmap_track/hm5_real_depth_pp_ready_c1/kinect_runtime_audit.json
  EXPECT: KINECT_RUNTIME_AUDIT_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=4a679307fbd5/37 entries; EXPECT=matched; output-sha256=f01732d47062d41ee5fd4079b8bcaa04df46e9cdbc2945b1edd5622e86244f15; output-bytes=24

- [x] G2: a depth frame produces the frozen 76-by-38 max-aggregation roarm-heightmap-v1 contract in roarm_base
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_scripts/p38_hm5_kinect_depth_pipeline.py --validate-bundle claudedocs/runtime_logs/heightmap_track/hm5_real_depth_pp_ready_c1
  EXPECT: HM5_BUNDLE_CONTRACT_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=4a679307fbd5/37 entries; EXPECT=matched; output-sha256=7569f31884b8bfaac6206c5a7c0b36daef3c6c9de6cbfd49939f03576172e864; output-bytes=23

- [x] G3: the positive-control frame proves zero and NaN pixels, edge-flight pixels, and unseen grid cells remain distinguishable from measured zero height
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_scripts/p38_hm5_kinect_depth_pipeline.py --validate-masks claudedocs/runtime_logs/heightmap_track/hm5_real_depth_pp_ready_c1
  EXPECT: HM5_MASKS_AND_OCCLUSION_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=4a679307fbd5/37 entries; EXPECT=matched; output-sha256=57c71bdd80e586556ddb15f931f07c68ac1b7153517031885d6c2bc3ac54b6cb; output-bytes=27

- [x] G4: the implementation compiles while P1 DEME and protected state-ledger files remain untouched
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python -m py_compile roarm_rl/heightmap.py sim_scripts/p38_hm5_kinect_depth_pipeline.py && git diff --exit-code -- sim_deme_pile.py sim_deme_scoop.py START_HERE.md claudedocs/DECISIONS.md claudedocs/DECISIONS_ACTIVE.md claudedocs/EXPERIMENT_LEDGER.md claudedocs/LEDGER_RECENT.md claudedocs/relay/from_claude.md && printf 'HM5_PROTECTED_FILES_OK\n'
  EXPECT: HM5_PROTECTED_FILES_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=4a679307fbd5/37 entries; EXPECT=matched; output-sha256=7c0875c7ff0122546145cc8af9eb4d9aad11ec442889a6fc3555a55c141d3054; output-bytes=23

- [x] G5: the decision bundle has rerun 0.34.1 footer verification, exact entities and timelines, a fixed RBL, a headless screenshot, and recorded visual inspection
  CHECK: /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p38_hm5_kinect_depth_pipeline.py --validate-rerun-bundle claudedocs/runtime_logs/heightmap_track/hm5_real_depth_pp_ready_c1
  EXPECT: HM5_RERUN_OBSERVABILITY_OK visual_inspection=complete
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=4a679307fbd5/37 entries; EXPECT=matched; output-sha256=8f255ab955f2c9f117ecc7a7bfb9e56ff15a2ff15f6911e85845fcf1575f5966; output-bytes=54

- [x] G6: the PP-arrival workflow requires only a live capture or saved transformed-depth frame and no recalibration or robot control
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_scripts/p38_hm5_kinect_depth_pipeline.py --validate-pp-readiness claudedocs/runtime_logs/heightmap_track/hm5_real_depth_pp_ready_c1
  EXPECT: HM5_PP_FRAME_SWAP_READY
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=4a679307fbd5/37 entries; EXPECT=matched; output-sha256=447ecd8f4d3cac0fd37cabb8973f97f043925909276f6fb70d6e75db3a4ec331; output-bytes=24
