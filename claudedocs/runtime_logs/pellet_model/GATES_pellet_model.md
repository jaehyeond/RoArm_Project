# Gates: pellet shape model + angle-of-repose calibration harness (track P3)

OWNS: sim_pellet_model.py, claudedocs/runtime_logs/pellet_model/**

Scope: replace the perfect-sphere pellet of `sim_deme_pile.py:639` with multi-sphere
clump templates whose mass and inertia come from the actual solid, measure the angle of
repose of a poured heap under an explicitly stated definition, sweep `(shape, mu, Crr)`,
and expose the inverse lookup `fit_to_measured_repose` so a measured PP angle can be
turned into simulation parameters immediately.

This ledger is separate from the repo-root `GATES.md`, which belongs to the P1
(`sim_deme_pile.py` / `sim_deme_scoop.py`) track and is not written by P3.

NOT MEASURED: no pellet has been procured or measured. Every physical value in this
track is a placeholder tagged `MEASURE` in each artifact's `non_claims` block. No number
below may be cited as a property of real polypropylene.

Interpreters: `~/miniconda3/envs/roarm/bin/python` has DEME 2.4.0 (no rerun 0.34.1);
`~/miniconda3/envs/isaaclab/bin/python` has rerun 0.34.1 (no DEME). Same split as
`sim_deme_pile.py`.

- [x] P0: this ledger states outcome checks that can fail
  CHECK: node /home/cgxr/.codex/skills/unlazy/scripts/gate-lint.mjs claudedocs/runtime_logs/pellet_model/GATES_pellet_model.md
  EXPECT: LINT OK
  EVIDENCE: exit=0; shell=/bin/bash; cwd=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model; EXPECT=matched; output-sha256=7eb72bb02be561859c17d243091b7406f4c60ee9c0452c39eef11893326377d1; output-bytes=7

- [x] P1: the format contract documents the arrays, the repose definition, and the unmeasured placeholders
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_pellet_model.py --describe-format
  EXPECT: FORMAT_CONTRACT_OK
  EVIDENCE: exit=0; shell=/bin/bash; cwd=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model; EXPECT=matched; output-sha256=583c640e4f2850fb1c7cf7d1500fde193a652d38c01c528d2ea4cd17c381fdb3; output-bytes=4105

- [x] P2: every template's mass and inertia are derived from its true solid and survive independent checks
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_pellet_model.py --describe-templates
  EXPECT: TEMPLATE_CONTRACT_OK
  EVIDENCE: exit=0; shell=/bin/bash; cwd=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model; EXPECT=matched; output-sha256=6927b0caea30d080b245dad44979b4fe881428bdd08199b54d57c6c9029e3e9c; output-bytes=955

- [x] P3: a calipered pellet can be turned into a template without editing code
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_pellet_model.py --measured-pellet 3.2 5.1 905
  EXPECT: MEASURED_TEMPLATE_OK
  EVIDENCE: exit=0; shell=/bin/bash; cwd=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model; EXPECT=matched; output-sha256=c2ab7ca78d49a0d56b5fbd755bdd10d9a1b472ff94343edf5e150a0659ba2394; output-bytes=763

- [x] P4: all three shapes settle with no inter-clump penetration and yield a gated repose measurement
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_pellet_model.py --validate-runs claudedocs/runtime_logs/pellet_model/repose_sphere_mu0p45_crr0p06_n1500_seed460.npz claudedocs/runtime_logs/pellet_model/repose_clump2_mu0p45_crr0p06_n1500_seed460.npz claudedocs/runtime_logs/pellet_model/repose_clump3_mu0p45_crr0p06_n1500_seed460.npz
  EXPECT: RUN_VALIDATION_OK count=3
  EVIDENCE: exit=0; shell=/bin/bash; cwd=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model; EXPECT=matched; output-sha256=eed55055e1d034cc87459fe967961ece5da695ec73c8fe18d008e9fa9347f5bc; output-bytes=412

- [x] P5: the sweep completes and the repose angle actually responds to particle shape
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_pellet_model.py --validate-sweep claudedocs/runtime_logs/pellet_model/repose_sweep.json --min-cells 27
  EXPECT: SWEEP_VALIDATION_OK
  EVIDENCE: exit=0; shell=/bin/bash; cwd=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model; EXPECT=matched; output-sha256=22a0f1f84e27d3eee1366e6381696565311bbccf059da01079202408c6a3dd22; output-bytes=234

- [x] P6: a measured angle can be inverted into candidate parameter sets
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python sim_pellet_model.py --fit-repose 30.0 --tol-deg 2.0
  EXPECT: FIT_OK measured=30.00deg
  EVIDENCE: exit=0; shell=/bin/bash; cwd=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model; EXPECT=matched; output-sha256=a44e6d4a265968eff8a08e78d7b625688c8df7c955fbc82e9ef7cbd9cd190154; output-bytes=609

- [x] P7: the module compiles and no other track's files were modified
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python -m py_compile sim_pellet_model.py && git diff --exit-code -- sim_deme_pile.py sim_deme_scoop.py sim_deme_mesh_min_example.py roarm_rl/heightmap.py GATES.md START_HERE.md claudedocs/DECISIONS.md claudedocs/DECISIONS_ACTIVE.md claudedocs/EXPERIMENT_LEDGER.md claudedocs/LEDGER_RECENT.md claudedocs/relay && printf 'P3_PROTECTED_FILES_OK\n'
  EXPECT: P3_PROTECTED_FILES_OK
  EVIDENCE: exit=0; shell=/bin/bash; cwd=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model; EXPECT=matched; output-sha256=2a7fb00469726e19d607bfd4e1f56ed51a5aac0dbc485f04a6158decc24acb5e; output-bytes=21

- [x] P8: the decision run has a footer-verified full-timeline RRD, fixed RBL, exact contracts, headless screenshot, and a recorded visual inspection
  CHECK: /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_pellet_model.py --validate-rerun-contract claudedocs/runtime_logs/pellet_model/repose_clump3_mu0p45_crr0p06_n1500_seed460_rerun_validation.json claudedocs/runtime_logs/pellet_model/repose_clump3_mu0p45_crr0p06_n1500_seed460_inspection.json
  EXPECT: RERUN_OBSERVABILITY_OK visual_inspection=complete
  EVIDENCE: exit=0; shell=/bin/bash; cwd=/home/cgxr/orca/workspaces/RoArm_Project/pellet-model; EXPECT=matched; output-sha256=5c37dea5e3db8f2d9b67af45006e24ce38d955310aacb5c541f14d7d10c9d9fd; output-bytes=49
