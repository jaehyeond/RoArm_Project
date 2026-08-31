# Gates: DEME pellet pile settling environment

OWNS: sim_deme_pile.py, GATES.md, claudedocs/runtime_logs/sim_deme/pile_*

Scope: Build and verify a seeded, parameterized DEME pellet-pile settling environment with exact inputs, pellet-scale repeat agreement, and reusable SI-unit output.

- [x] G0: this ledger states outcome checks that can fail
  CHECK: node /home/cgxr/.codex/skills/unlazy/scripts/gate-lint.mjs GATES.md
  EXPECT: LINT OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=7a43578904ba/37 entries; EXPECT=matched; output-sha256=48630b7361dd44ee870917b12c3d19b9d7bdea738aaca16bb04d4cab83b772d2; output-bytes=8

- [x] G1: the command-line format contract documents array shapes, SI units, and coordinates
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_deme_pile.py --describe-format
  EXPECT: FORMAT_CONTRACT_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=7a43578904ba/37 entries; EXPECT=matched; output-sha256=2eff7c50b56974b61e5ad2bb2037b8287dd6ac17d8f31cceb986423efa7284f9; output-bytes=2174

- [x] G2: smoke and practical target-ridge artifacts both pass settlement, containment, non-penetration, and finite-value validation
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_deme_pile.py --validate-output claudedocs/runtime_logs/sim_deme/pile_targetridge_smoke_d4p16_n512_seed460.npz claudedocs/runtime_logs/sim_deme/pile_practical_targetridge_d4p16_n18796_seed460.npz
  EXPECT: OUTPUT_VALIDATION_OK count=2
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=7a43578904ba/37 entries; EXPECT=matched; output-sha256=230a63286eefe6b9e4192b0e52df80d6cce3320e8a957788a001e0cd30f0c60b; output-bytes=487

- [x] G3: two same-seed practical runs preserve bit-exact inputs and agree at pellet-scale in the shared 5 mm heightmap contract while recording raw GPU bit-exact failure
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_deme_pile.py --compare-reproducibility claudedocs/runtime_logs/sim_deme/pile_practical_targetridge_d4p16_n18796_seed460.npz claudedocs/runtime_logs/sim_deme/pile_practical_targetridge_d4p16_n18796_seed460_rep2.npz claudedocs/runtime_logs/sim_deme/pile_practical_targetridge_d4p16_n18796_seed460_reproducibility.json
  EXPECT: REPRODUCIBILITY_OK initial=bit_exact final=heightmap_tolerance raw_final_bit_exact=false
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=7a43578904ba/37 entries; EXPECT=matched; output-sha256=2305ad5462ec92f973334638d48cf170b37eb42f65cf796bd769f278984d85b2; output-bytes=230

- [x] G4: the practical artifact records a completed 18,796-particle timed settling run
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_deme_pile.py --validate-output claudedocs/runtime_logs/sim_deme/pile_practical_targetridge_d4p16_n18796_seed460.npz --expected-n 18796 --require-timing
  EXPECT: PRACTICAL_COST_OK n=18796
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=7a43578904ba/37 entries; EXPECT=matched; output-sha256=148a09301e87d88d95d15def8a6fabac58defcad48a99325e685abe17e28445e; output-bytes=339

- [x] G5: the new implementation compiles without modifying prior DEME smoke evidence
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python -m py_compile sim_deme_pile.py && git diff --exit-code -- sim_deme_smoke.py claudedocs/runtime_logs/sim_deme/smoke_N500.json claudedocs/runtime_logs/sim_deme/smoke_N2000.json claudedocs/runtime_logs/sim_deme/smoke_N8000.json claudedocs/runtime_logs/sim_deme/smoke_N20000.json claudedocs/runtime_logs/sim_deme/smoke_N50000.json claudedocs/runtime_logs/sim_deme/smoke_N100000.json && printf 'PROTECTED_SMOKE_EVIDENCE_OK\n'
  EXPECT: PROTECTED_SMOKE_EVIDENCE_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=7a43578904ba/37 entries; EXPECT=matched; output-sha256=bc1a80c01c94792fb71f1ab02df3417531e084a1cef7bb64e31c25d7cd4fe226; output-bytes=28

- [x] G6: the practical pile has a footer-verified RRD, fixed RBL, exact contracts, headless screenshot, and recorded visual inspection
  CHECK: /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_deme_pile.py --validate-rerun-contract claudedocs/runtime_logs/sim_deme/pile_practical_targetridge_d4p16_n18796_seed460_rerun_validation.json claudedocs/runtime_logs/sim_deme/pile_practical_targetridge_d4p16_n18796_seed460_inspection.json
  EXPECT: RERUN_OBSERVABILITY_OK visual_inspection=complete
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=7a43578904ba/37 entries; EXPECT=matched; output-sha256=3763b87826ebb82c08f0805f15e87aabdc5f0dad5b6cb1cf6621fc323841d2f6; output-bytes=50
