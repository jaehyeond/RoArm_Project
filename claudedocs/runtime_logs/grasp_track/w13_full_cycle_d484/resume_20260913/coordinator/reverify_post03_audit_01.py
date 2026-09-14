#!/usr/bin/env python3
"""Re-run the fully reviewed CPU auditor into a new root-owned receipt only.

Manual image observations are preserved prior inspection evidence, not an
automated or repeated visual inspection. The expected result is rc1/three FAILs.
"""
import hashlib
import runpy
from pathlib import Path

AUDITOR = Path("/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/test_post03_partial_results.py")
TARGET = Path(__file__).with_name("ROOT_POST03_AUDIT_REPRO_01.json")

if TARGET.exists():
    raise SystemExit("Refuse to overwrite root evidence: " + str(TARGET))
source_sha = hashlib.sha256(AUDITOR.read_bytes()).hexdigest()
print("Reviewed auditor source SHA256:", source_sha, flush=True)
namespace = runpy.run_path(str(AUDITOR), run_name="w13_root_reviewed_auditor")
namespace["main"].__globals__["OUT"] = TARGET
raise SystemExit(namespace["main"]())
