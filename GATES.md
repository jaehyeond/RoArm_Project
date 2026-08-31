# Gates: DEME pellet pile settling environment

OWNS: sim_deme_pile.py, GATES.md, claudedocs/runtime_logs/sim_deme/pile_*

Scope: Build and verify a deterministic, parameterized DEME pellet-pile settling environment with reusable SI-unit output.

- [ ] G0: this ledger states outcome checks that can fail
  CHECK: node /home/cgxr/.codex/skills/unlazy/scripts/gate-lint.mjs GATES.md
  EXPECT: LINT OK
  EVIDENCE: pending

- [ ] G1: the command-line format contract documents array shapes, SI units, and coordinates
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_deme_pile.py --describe-format
  EXPECT: FORMAT_CONTRACT_OK
  EVIDENCE: pending

- [ ] G2: smoke and practical artifacts both pass settlement, containment, shape, and finite-value validation
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_deme_pile.py --validate-output claudedocs/runtime_logs/sim_deme/pile_smoke_d4p16_n512_seed460.npz claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz
  EXPECT: OUTPUT_VALIDATION_OK count=2
  EVIDENCE: pending

- [ ] G3: two same-seed practical runs preserve bit-exact inputs and agree at pellet-scale in the shared 5 mm heightmap contract while recording raw GPU bit-exact failure
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_deme_pile.py --compare-reproducibility claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460_rep2.npz claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460_reproducibility.json
  EXPECT: REPRODUCIBILITY_OK initial=bit_exact final=heightmap_tolerance raw_final_bit_exact=false
  EVIDENCE: pending

- [ ] G4: the practical artifact records a completed 18,796-particle timed settling run
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_deme_pile.py --validate-output claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz --expected-n 18796 --require-timing
  EXPECT: PRACTICAL_COST_OK n=18796
  EVIDENCE: pending

- [ ] G5: the new implementation compiles without modifying prior DEME smoke evidence
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python -m py_compile sim_deme_pile.py && git diff --exit-code -- sim_deme_smoke.py claudedocs/runtime_logs/sim_deme/smoke_N500.json claudedocs/runtime_logs/sim_deme/smoke_N2000.json claudedocs/runtime_logs/sim_deme/smoke_N8000.json claudedocs/runtime_logs/sim_deme/smoke_N20000.json claudedocs/runtime_logs/sim_deme/smoke_N50000.json claudedocs/runtime_logs/sim_deme/smoke_N100000.json && printf 'PROTECTED_SMOKE_EVIDENCE_OK\n'
  EXPECT: PROTECTED_SMOKE_EVIDENCE_OK
  EVIDENCE: pending

- [ ] G6: the practical pile has a footer-verified RRD, fixed RBL, exact contracts, headless screenshot, and recorded visual inspection
  CHECK: /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_deme_pile.py --validate-rerun-contract claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460_v2_rerun_validation.json claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460_v2_inspection.json
  EXPECT: RERUN_OBSERVABILITY_OK visual_inspection=complete
  EVIDENCE: pending
