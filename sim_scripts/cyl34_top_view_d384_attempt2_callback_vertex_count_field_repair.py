#!/usr/bin/env python3
"""Forward-only D384 attempt2 field-name preflight repair.

Attempt1 stopped before canonical computation because it requested the absent
key ``live_callback_vertex_count``.  Immutable D379 stores the same count as
``vertex_count``.  This wrapper:

* freezes attempt1 and never writes to its directory;
* verifies all 34 D379 callback rows have ``vertex_count`` equal to the length
  of ``live_callback_vertices_m``;
* supplies only an in-memory compatibility alias to the unchanged D384 design
  implementation;
* keeps the two D384 scientific/design variables unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BASE_SCRIPT = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d384_p34_failed_part_representation_repair_design.py"
)
SPEC = importlib.util.spec_from_file_location("d384_attempt1_frozen", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load D384 base design: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

ATTEMPT = "attempt2_callback_vertex_count_field_preflight_repair"
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d384"
    / ATTEMPT
)
SCRIPT_PATH = Path(__file__).resolve()
PREFLIGHT_PATH = OUT_DIR / "d384_attempt2_callback_field_preflight.json"

base.ATTEMPT = ATTEMPT
base.OUT_DIR = OUT_DIR
base.SCRIPT_PATH = SCRIPT_PATH

_PATH_NAMES = {
    "PREREG_PATH": "d384_preregistration.json",
    "PHASE_PATH": "d384_phase_markers.jsonl",
    "INVOCATION_PATH": "d384_offline_design_invocation.json",
    "WORKER_STDOUT": "d384_offline_worker_stdout.log",
    "WORKER_STDERR": "d384_offline_worker_stderr.log",
    "WORKER_CLAIM": "d384_offline_worker_claim.json",
    "SUPERVISOR_PATH": "d384_offline_worker_supervisor.json",
    "EVIDENCE_PATH": "d384_p34_representation_repair_design_evidence.json",
    "METRICS_CSV": "d384_repair_design_part_metrics.csv",
    "BOARD_PATH": "d384_p34_representation_repair_design_1920x1080.png",
    "RRD_PATH": "d384_p34_representation_repair_design.rrd",
    "RBL_PATH": "d384_p34_representation_repair_design.rbl",
    "RERUN_VALIDATION": "d384_rerun_validation.json",
    "RERUN_SCREENSHOT": "d384_rerun_inspection.png",
    "MANUAL_TEMPLATE": "d384_manual_visual_inspection_template.json",
    "MANUAL_INSPECTION": "d384_manual_visual_inspection.json",
    "COMPLETION_PATH": "d384_completion_summary.json",
}
for name, filename in _PATH_NAMES.items():
    setattr(base, name, OUT_DIR / filename)


def _source_hashes() -> dict[str, str]:
    return {
        "attempt2_wrapper": base._sha(SCRIPT_PATH),
        "attempt1_design_module": base._sha(BASE_SCRIPT),
    }


base._source_hashes = _source_hashes
_original_read_json = base._read_json


def _callback_preflight() -> dict[str, Any]:
    with base.D379_EVIDENCE.open("r", encoding="utf-8") as stream:
        d379 = json.load(stream)
    rows = d379.get("callback_rows")
    if not isinstance(rows, list):
        raise TypeError("D379 callback_rows is not a list")
    checks = {
        "callback_rows_exact_34": len(rows) == 34,
        "vertex_count_present_34": sum(
            isinstance(row, dict) and "vertex_count" in row for row in rows
        )
        == 34,
        "obsolete_live_callback_vertex_count_absent_34": sum(
            isinstance(row, dict)
            and "live_callback_vertex_count" not in row
            for row in rows
        )
        == 34,
        "vertex_count_matches_live_array_length_34": sum(
            isinstance(row, dict)
            and row.get("vertex_count")
            == len(row.get("live_callback_vertices_m") or [])
            for row in rows
        )
        == 34,
    }
    return {
        "artifact": "D384_ATTEMPT2_CALLBACK_FIELD_PREFLIGHT_V1",
        "attempt1_failure": {
            "exception": "KeyError: live_callback_vertex_count",
            "classification": "HARNESS_FIELD_NAME_ERROR_BEFORE_CANONICAL_COMPUTE",
            "attempt1_rerun_or_overwrite": False,
        },
        "immutable_d379": {
            "path": base._rel(base.D379_EVIDENCE),
            "sha256": base._sha(base.D379_EVIDENCE),
        },
        "registered_field": "vertex_count",
        "derived_alias": "live_callback_vertex_count",
        "alias_scope": "in-memory compatibility only; no input mutation",
        "checks": checks,
        "pass": all(checks.values()),
    }


def _read_json_with_alias(path: Path) -> dict[str, Any]:
    value = _original_read_json(path)
    if path.resolve() == base.D379_EVIDENCE.resolve():
        for row in value["callback_rows"]:
            if row["vertex_count"] != len(row["live_callback_vertices_m"]):
                raise RuntimeError(
                    "D379 vertex_count/live array mismatch after preflight"
                )
            row["live_callback_vertex_count"] = row["vertex_count"]
    return value


base._read_json = _read_json_with_alias


def prepare() -> int:
    preflight = _callback_preflight()
    if preflight["pass"] is not True:
        raise RuntimeError(f"attempt2 callback preflight failed: {preflight}")
    result = base.prepare()
    base._write_json_x(PREFLIGHT_PATH, preflight)
    base._phase(
        "attempt2_callback_field_preflight_frozen",
        preflight_sha256=base._sha(PREFLIGHT_PATH),
        passed=True,
    )
    return result


def worker() -> int:
    expected = _callback_preflight()
    frozen = base._read_json(PREFLIGHT_PATH)
    if frozen != expected or frozen.get("pass") is not True:
        raise RuntimeError("attempt2 callback preflight drifted")
    return base.worker()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "run", "worker", "finalize"),
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        return prepare()
    if args.stage == "run":
        return base.run_supervisor()
    if args.stage == "worker":
        return worker()
    return base.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
