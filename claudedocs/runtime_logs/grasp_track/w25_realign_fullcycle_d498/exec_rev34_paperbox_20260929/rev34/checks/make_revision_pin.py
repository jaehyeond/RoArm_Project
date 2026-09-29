"""rev34 REVISION_PIN.json 작성 (sha256 전수). usage: python make_revision_pin.py  (rev34 디렉터리 기준)"""
import hashlib
import json
import time
from pathlib import Path

R34 = Path(__file__).resolve().parent.parent
R32 = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/"
           "w19_runpod_d487/rev32_frozen_copy")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


pinned = sorted([p for p in R34.rglob("*") if p.is_file()
                 and p.parts[len(R34.parts)] in ("src", "checks")
                 and "out" not in p.relative_to(R34).parts[:2]] +
                [R34 / n for n in ("params_w25.json", "params_w25_OFF_rev32equiv.json", "params_w25_TEMP_oldpile.json",
                                   "params_w13.json", "criteria.json", "numeric_inputs.json",
                                   "numeric_inputs_w25_TEMP_oldpile.json", "COMMANDS.json", "params_w25_paperbox.json", "params_w25_paperbox_inner306.json",
                                   "domain/numeric_inputs_w25_paperbox_310x220_TEMP.json", "domain/numeric_inputs_w25_paperbox_inner306_TEMP.json",
                                   "COMMANDS_w25_template.json", "DIFF_rev32_to_rev34_src.patch")])
changed = {}
for f in sorted((R32 / "src").glob("*.py")):
    a, b = sha(f), sha(R34 / "src" / f.name)
    if a != b:
        changed[f"src/{f.name}"] = {"rev32": a, "rev34": b}
pin = {
    "artifact": "W25A_REV34_REVISION_PIN_V1", "revision": "rev34",
    "derived_from": "rev32 frozen copy (byte copy, 34/34 sha256 = rev32 REVISION_PIN) + W25-A 실물 정렬 패치",
    "rev32_dir": str(R32), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "changed_vs_rev32": changed,
    "unchanged_src_files": sorted(f.name for f in (R32 / "src").glob("*.py") if sha(f) == sha(R34 / "src" / f.name)),
    "criteria_json_identical_to_rev32": sha(R32 / "criteria.json") == sha(R34 / "criteria.json"),
    "physics_params_changed": False,
    "placement_params_changed": ["declared_base_cm 38→33.2", "declared_pellet_cm 26→24", "arm_radius_m 0.35→0.25",
                                 "+ box_frame_convention/w25_box_anchor/tray_* (params_w25.json)"],
    "control_procedure_changed": "params 스위치(w25_proc_*) — OFF 이면 rev32 와 원시 동일(checks/out/equality_*.json)",
    "saved_frame_contract_changed": False,
    "new_npz_arrays": [], "new_result_json_keys": ["w25"], "new_metadata_json_keys": ["w25_frame"],
    "physics_executed_with_rev34": False, "deme_solver_constructed": False, "gpu_used": False,
    "frozen_copies_sha256": {str(p.relative_to(R34)): sha(p) for p in pinned if p.exists()},
}
json.dump(pin, open(R34 / "REVISION_PIN.json", "w"), ensure_ascii=False, indent=1)
print(len(pin["frozen_copies_sha256"]), "files pinned;", len(changed), "src changed")
