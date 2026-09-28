"""W14 raw repair — 테스트 공통 경로/로더. 생산 분류 함수는 여기서 호출하지 않는다."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
MAIN = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
OUT = MAIN / "claudedocs/runtime_logs/grasp_track/w14_w13_raw_repair_d484/repair_20260916_01"
REV29_SRC = OUT / "rev29/src"
DERIVED = OUT / "derived"
IMPL28 = Path("/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/"
              "w13_full_cycle_d484/resume_20260913/implementation")
REV28_SRC = IMPL28 / "rev28/src"
RAW = IMPL28 / "run_01/w13_cycle_seed460.npz"
META = IMPL28 / "run_01/w13_cycle_seed460.json"
PILE = Path("/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/"
            "pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz")
AUDIT = Path("/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/"
             "w13_full_cycle_d484/resume_20260913/audit")
EXPECTED = {"raw": "529f422e962730b17b520c3c9af2261aae7c431ec5e078cd2c368dfab0c46b0f",
            "meta": "e482b939a4cf42e5284b9b15e36a9f3613e989161ec538475e2c05f4260d9340",
            "pile": "659d6b0bc771678a0c7209d91f550edc933d03e41922245ea0adb64eeb818812"}
# 독립 감사(REV28_PRODUCTION_PARTIAL_RAW_AUDIT_01.json) 가 사전 기록한 기대값 — 이 테스트가 다시 유도해 대조한다
AUDIT_EXPECTED_PHASE_ONLY = [1, 26, 4810, 5161, 7166, 7300, 7336, 8835, 10881, 11256, 14875]
AUDIT_RECORDED_25 = [0, 1, 26, 556, 885, 1119, 1186, 1187, 4810, 5161, 7166, 7300, 7336, 7337, 7563, 7892,
                     8275, 8604, 8835, 10881, 11256, 14875, 15106, 15435, 15818]
AUDIT_ALL_FRAME_MISMATCH = 1507161
AUDIT_STRICT_FINAL = {"source": 14350, "receiving_bin": 0, "tool_residual": 0, "spill": 73, "in_flight": 0, "ambiguous": 5577}
AUDIT_RECORDED_FINAL = {"source": 19712, "receiving_bin": 0, "tool_residual": 0, "spill": 73, "in_flight": 0, "ambiguous": 215}
AUDIT_COHORT_STRICT_FINAL = {"source": 132, "receiving_bin": 0, "tool_residual": 0, "spill": 7, "in_flight": 0, "ambiguous": 5}
LABELS = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def import_from(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ig_rev28():
    return import_from(REV28_SRC / "inventory_geometry.py", "ig_rev28_frozen")


def ig_rev29():
    return import_from(REV29_SRC / "inventory_geometry.py", "ig_rev29")


def derive_module():
    if str(REV29_SRC) not in sys.path:
        sys.path.insert(0, str(REV29_SRC))
    return import_from(REV29_SRC / "derive_repaired_raw.py", "derive_repaired_raw")


def template():
    with np.load(PILE, allow_pickle=False) as pile:
        return json.loads(str(pile["clump_template_json"]))


def counts(code):
    return {n: int(v) for n, v in zip(LABELS, np.bincount(np.asarray(code).astype(int), minlength=6))}

# ── v2 (rev30, ERRATUM_04 support-floor) 추가 경로 ─────────────────────────
REV30_SRC = OUT / "rev30/src"
DERIVED_V2 = OUT / "derived_v2"


def ig_rev30():
    return import_from(REV30_SRC / "inventory_geometry.py", "ig_rev30")
