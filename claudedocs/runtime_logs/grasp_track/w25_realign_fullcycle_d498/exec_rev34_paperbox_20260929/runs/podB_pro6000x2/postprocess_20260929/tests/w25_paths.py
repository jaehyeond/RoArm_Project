"""W25 후처리 경로·헬퍼. **기대값 하드코딩 0** — 모든 생산 수치는 실행 시 생산 JSON/원자료에서 읽는다."""
import hashlib, importlib.util, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POST = HERE.parent
RUN = POST.parent / "run_01"
RAW = RUN / "w13_cycle_seed460.npz"
META = RUN / "w13_cycle_seed460.json"
TIMELINE = RUN / "timeline_seed460.json"
RECEIPT = POST.parent / "RETRIEVAL_RECEIPT.json"
COPY = POST / "rev34_copy"
REV29_SRC = COPY / "rev29/src"
REV34_SRC = COPY / "rev34/src"
DERIVED = POST / "derived_v2_w25"
DERIVED_NPZ = DERIVED / "w25_podB_seed460_rev34_derived.npz"
PILE = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/"
            "w25_realign_fullcycle_d498/pile_flat40_20260929/"
            "pile_lens6_a4p5_b3p8_c2p5_slab40_outer_310x220_n67737_rho0p503_seed460.npz")
LABELS = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def derive_module():
    sys.path.insert(0, str(REV29_SRC))
    return _imp(REV29_SRC / "derive_repaired_raw.py", "derive_repaired_raw_rev29")


def ig_rev34():
    return _imp(REV34_SRC / "inventory_geometry.py", "inventory_geometry_rev34")


def template():
    with np.load(PILE, allow_pickle=False) as p:
        return json.loads(str(p["clump_template_json"]))


def receipt_sha():
    return json.load(open(RECEIPT))["files"]


def counts(code):
    return {n: int(v) for n, v in zip(LABELS, np.bincount(np.asarray(code).astype(int), minlength=6))}


class FrameView:
    def __init__(self, z, keys):
        self._c = {k: np.asarray(z[k]) for k in keys}
        self._z = z

    def __getitem__(self, k):
        return self._c[k] if k in self._c else self._z[k]
