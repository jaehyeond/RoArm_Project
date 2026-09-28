Follow-up to your rev30 v2 audit (same repo, same rules). The coordinator created rev31 = rev30 with ONLY the two metadata declaration strings changed to the exact ERRATUM_04 wording you registered; the label predicates are untouched. Re-verify quickly.

READ-ONLY inputs:
- rev31: /home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w14_w13_raw_repair_d484/repair_20260916_01/rev31/ (REVISION_PIN.json, DIFF_rev30_to_rev31_src.patch, src/inventory_geometry.py, src/derive_v2.py)
- rev31 derived: .../repair_20260916_01/derived_v2_rev31/w13_cycle_seed460_rev31_derived_v2.npz + DERIVED_V2_MANIFEST.json (the label array is still named inventory_code_rev30_support inside the NPZ; the manifest "semantics" carries the declarations).
- Registered erratum: /home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/RAW_SCHEMA_REQUIRED_ERRATUM_04.md (sha256 4290a9cf...e8e1a6).

DO, writing only under /home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w14_w13_raw_repair_d484/repair_20260916_01/audit/ :
1. Item E1: exact string check — manifest semantics classify_floor_rule == "support_surface_v2: min(center_z - r) > floor - margin" and classify_contract_version == "RAW_SCHEMA_REQUIRED + ERRATUM_04"; rev31 inventory_geometry.py constants equal the same strings.
2. Item E2: rev30->rev31 diff scope — only inventory_geometry.py (REVISION, CONTRACT_VERSION, FLOOR_RULE constants + docstring line) and derive_v2.py (assert/artifact/name strings); no predicate change. Configs byte-identical.
3. Item E3: label identity — rev31 derived label array bit-identical to rev30 derived array (and therefore to your own v2 recomputation: 0 mismatches; you may re-run your checker against the rev31 NPZ, ~40 s).
4. Item E4: preservation unchanged (raw/pile/rev28/rev29/rev30 pins).
5. Write W14_REV31_V2_INDEPENDENT_AUDIT_02.json (findings E1-E4 with pass/evidence, overall verdict) and append a short section to REPORT_rev30_v2.md (or write REPORT_rev31_v2.md) in Korean. Update nothing else. Then worker_done (--outcome succeeded when the re-verification ran; report FAIL honestly if any item fails) with --report-path to the new JSON.
Constraints unchanged: CPU, numpy only, forbidden imports as before, no edits outside the audit folder, no git, ~15 minutes.
