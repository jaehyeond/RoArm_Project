# Gates: P2b sequential scoop decision loop

OWNS: sim_scripts/p39_decision_loop.py, claudedocs/runtime_logs/decision_track/p2b_decision_loop/**

Scope: Connect a contract-frozen heightmap to replaceable policies and executors, run four deterministic analytic-scoop episodes to a bounded terminal condition, and preserve replayable decision evidence.

- [x] G0: this ledger states outcome checks that can fail
  CHECK: node /home/cgxr/.codex/skills/unlazy/scripts/gate-lint.mjs claudedocs/runtime_logs/decision_track/p2b_decision_loop/GATES.md
  EXPECT: LINT OK
  CWD: .
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=d7d07d7cb56d/37 entries; EXPECT=matched; output-sha256=48630b7361dd44ee870917b12c3d19b9d7bdea738aaca16bb04d4cab83b772d2; output-bytes=8

- [x] G1: the decision-loop module compiles in the pinned Rerun environment
  CHECK: /home/cgxr/miniconda3/envs/isaaclab/bin/python -m py_compile sim_scripts/p39_decision_loop.py && printf 'DECISION_LOOP_COMPILE_OK\n'
  EXPECT: DECISION_LOOP_COMPILE_OK
  CWD: .
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=d7d07d7cb56d/37 entries; EXPECT=matched; output-sha256=f941eb1fd1e74065ff5ca3dfe2b5a0b66eb11b5c909ab94e4f5211ed3848f69e; output-bytes=25

- [x] G2: contracts, bounded failure paths, executor substitution, policy set, and uncertainty delivery pass executable self-checks
  CHECK: /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p39_decision_loop.py --self-check
  EXPECT: DECISION_LOOP_SELF_CHECK_OK
  CWD: .
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=d7d07d7cb56d/37 entries; EXPECT=matched; output-sha256=8885982414f611f08699fd73a067b08a618b35425747f2d02e926d8bf1217c7c; output-bytes=28

- [x] G3: four policies reach a bounded terminal condition and produce at least two distinct scoop counts with exact primary metrics
  CHECK: /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p39_decision_loop.py --validate-output claudedocs/runtime_logs/decision_track/p2b_decision_loop/run_primary/summary.json --require-distinct-scoop-counts
  EXPECT: DECISION_LOOP_OUTPUT_OK policies=4 distinct_scoop_counts=true
  CWD: .
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=d7d07d7cb56d/37 entries; EXPECT=matched; output-sha256=8634b19f4dd0a87cc42654ed90a5c7cde84353f61d2e43bc9a947e5a1c178ffc; output-bytes=86

- [x] G4: a same-seed independent rerun is canonically identical to the primary four-policy result
  CHECK: /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p39_decision_loop.py --compare-results claudedocs/runtime_logs/decision_track/p2b_decision_loop/run_primary/summary.json claudedocs/runtime_logs/decision_track/p2b_decision_loop/run_repeat/summary.json
  EXPECT: DECISION_LOOP_REPRODUCIBILITY_OK canonical_equal=true
  CWD: .
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=d7d07d7cb56d/37 entries; EXPECT=matched; output-sha256=b0a00d35b1e9e1b3170eb90f61c163ef866e4fc566bc4c44a0e0eeadc0b69f90; output-bytes=54

- [x] G5: the protected producer, predictor, DEME, pellet, state-ledger, and relay files remain untouched while forbidden positioning prose is caught by a positive-control scanner and absent from the new script
  CHECK: git diff --exit-code -- roarm_rl/heightmap.py model_scoop_predictor.py sim_deme_pile.py sim_deme_scoop.py sim_pellet_model.py START_HERE.md claudedocs/DECISIONS.md claudedocs/DECISIONS_ACTIVE.md claudedocs/EXPERIMENT_LEDGER.md claudedocs/LEDGER_RECENT.md claudedocs/relay/from_claude.md && /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p39_decision_loop.py --check-positioning-language
  EXPECT: PROTECTED_FILES_AND_POSITIONING_LANGUAGE_OK
  CWD: .
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=d7d07d7cb56d/37 entries; EXPECT=matched; output-sha256=42955c2222aedfe517c201a3cb75926df3ceaa64285d287d72933a2e908137b9; output-bytes=44

- [x] G6: the combined policy replay has a footer-verified RRD, exact decision entities and timelines, fixed RBL, and a headless screenshot
  CHECK: /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p39_decision_loop.py --validate-rerun-contract claudedocs/runtime_logs/decision_track/p2b_decision_loop/run_primary/rerun_validation.json
  EXPECT: DECISION_LOOP_RERUN_OK footer=true exact_contract=true screenshot=true
  CWD: .
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=d7d07d7cb56d/37 entries; EXPECT=matched; output-sha256=e4c85ce678125eb679e30c1d5301d648e75307ef25631655d01b71bca5f8178a; output-bytes=71

- [x] G7: visual inspection is tied to the exact headless screenshot and records observations of pile evolution, selected actions, and cumulative metrics
  CHECK: /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p39_decision_loop.py --validate-inspection claudedocs/runtime_logs/decision_track/p2b_decision_loop/run_primary/visual_inspection.json
  EXPECT: DECISION_LOOP_VISUAL_INSPECTION_OK observations=3
  CWD: .
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/orca/workspaces/RoArm_Project/kinect-hm; path=d7d07d7cb56d/37 entries; EXPECT=matched; output-sha256=9e09ebf5aba1ccc9608e2293bf60a279f66874e74f5e65a8d14e6d12107b0e4d; output-bytes=50
