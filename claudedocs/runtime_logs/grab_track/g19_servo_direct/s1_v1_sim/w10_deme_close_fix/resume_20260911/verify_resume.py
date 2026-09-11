"""Recompute W10 continuation completion from original files; run after cells finish."""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
REPO = OUT.parents[5]
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
launch = read(HERE / "launch_evidence.json")
log = (OUT / "run.log").read_text()
stages = {}
for m in re.finditer(r"stage (\S+)(?: attempt (\d))? rc=(\d+) wall=(\d+)s(?: stalled=(\d))?", log):
    stages[m[1]] = {"rc": int(m[3]), "wall_s": int(m[4]), "stalled": m[5]}

physics_files = ["sim_deme_scoop_s1.py", "sim_deme_scoop.py", str(OUT.relative_to(REPO) / "GATES_w10.md"),
                 str(OUT.relative_to(REPO) / "params_w10_DE_dt2e6_c.json"), str(OUT.relative_to(REPO) / "params_w10_DE_E1e8_c.json")]
unchanged = {p: sha(REPO / p) == launch["source_sha256"][p] for p in physics_files}
cell_checks = {}
for stage in ("DE_dt2e6_c", "DE_E1e8_c"):
    cell = OUT / f"cell_{stage}"
    p = cell / "scoop_s1_seed460.json"
    terminal = stages.get(stage)
    result = read(p) if p.exists() else None
    details = {"terminal": terminal, "has_result": result is not None, "diverged": None if result is None else result["diverged"]}
    if result:
        z = np.load(cell / "scoop_s1_seed460.npz")
        timeline = read(cell / "timeline_seed460.json")
        params = read(OUT / f"params_w10_{stage}.json")
        params_ok = all(result["params"].get(k) == v for k,v in params.items())
        pile_path = next(k for k in result["inputs_sha16"] if k.endswith(".npz"))
        pile = np.load(pile_path, allow_pickle=True)
        tpl = json.loads(str(pile["clump_template_json"].item()))
        count = int(z["in_cavity"].sum())
        mass = count * tpl["mass_kg"] * 1000
        count_ok = count == result["capture"]["n_in_cavity"]
        mass_ok = abs(mass-result["capture"]["mass_g"]) <= 0.00005001
        source_ok = all(sha(Path(k) if Path(k).is_absolute() else REPO / k)[:16] == v for k,v in result["inputs_sha16"].items())
        timeline_ok = len(timeline["rows"]) == len(z["frame_t_s"]) == result["steps_completed"] and timeline["state"] == ("diverged" if result["diverged"] else "complete")
        details.update(params_match=params_ok, inputs_sha_match=source_ok, capture_count=count, capture_mass_g_recomputed=mass,
                       capture_count_matches=count_ok, capture_mass_matches=mass_ok, timeline_matches=timeline_ok,
                       stops=result["door"]["stops"], source_evidence_ok=bool(params_ok and source_ok and count_ok and mass_ok and timeline_ok))
    cell_checks[stage] = details

dt = cell_checks["DE_dt2e6_c"]
dt_done = bool(dt["terminal"] and dt["has_result"] and dt.get("source_evidence_ok"))
diverged = dt["diverged"] is True or (dt["terminal"] or {}).get("rc") == 134
ladder = cell_checks["DE_E1e8_c"]
ladder_done = bool(ladder["terminal"] and ((ladder["has_result"] and ladder.get("source_evidence_ok")) or ladder["terminal"]["rc"] == 134))
disposition = bool(all(unchanged.values()) and (dt_done or (dt["terminal"] or {}).get("rc") == 134) and (not diverged or ladder_done))
gates = read(OUT / "gates_w10.json")
rcell = gates["R_rerun_d341"]["cell"]
validation = None if not rcell else OUT / f"cell_{rcell}/scoop_s1_seed460_w10_rerun_validation.json"
v = read(validation) if validation and validation.exists() else {}
coverage = read(v["coverage_readback"]["path"]) if v.get("coverage_readback", {}).get("path") else {}
artifact_hashes_match = bool(v and Path(v["path"]).is_file() and sha(v["path"]) == v["sha256"])
machine = bool(v.get("pass") and coverage.get("pass") and artifact_hashes_match and v.get("log_status_summary", {}).get("sink_finalized"))
output = {"artifact": "W10_RESUME_VERIFICATION", "physics_files_unchanged": unchanged, "cells": cell_checks,
          "scientific_disposition_verified": disposition, "scientific_gates_all_pass": gates["all_pass"],
          "d341_machine_contract_pass": machine, "d341_validation": str(validation),
          "limitation": "A completed experiment may fail the scientific gates. Completion and success are reported separately."}
(HERE / "verification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
print(json.dumps({k:output[k] for k in ("scientific_disposition_verified", "scientific_gates_all_pass", "d341_machine_contract_pass")}, ensure_ascii=False))
sys.exit(0 if disposition and machine else 1)
