#!/usr/bin/env python3
"""D355 failure-observability attempt3: actual prior inventory repair only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim_scripts import cyl34_top_view_d355_attempt1_failure_observability as base


OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d355/attempt3_failure_observability"
PREREG_PATH = OUT_DIR / "d355_attempt3_failure_observability_preregistration.json"
ATTEMPT1_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d355/attempt1_failure_observability"
ATTEMPT2_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d355/attempt2_failure_observability"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(path: Path) -> dict[str, str]:
    return {item.name: _sha256(item) for item in sorted(path.iterdir()) if item.is_file()}


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _bind_paths() -> None:
    base.OUT_DIR = OUT_DIR
    base.PREREG_PATH = PREREG_PATH
    base.INVOCATION_PATH = OUT_DIR / "d355_attempt3_failure_observability_invocation.json"
    base.RRD_PATH = OUT_DIR / "d355_attempt1_failure_explained.rrd"
    base.RBL_PATH = OUT_DIR / "d355_attempt1_failure_explained.rbl"
    base.PNG_PATH = OUT_DIR / "d355_attempt1_failure_explained_rerun.png"
    base.VALIDATION_PATH = OUT_DIR / "d355_attempt3_failure_observability_validation.json"
    base.SUMMARY_PATH = OUT_DIR / "d355_attempt3_failure_observability_summary.json"
    base.MANUAL_PATH = OUT_DIR / "d355_attempt3_failure_manual_visual_inspection.json"
    base.COMPLETION_PATH = OUT_DIR / "d355_attempt3_failure_observability_completion.json"
    base.__file__ = str(Path(__file__).resolve())


def _prepare() -> None:
    _bind_paths()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attempt1_inventory = _inventory(ATTEMPT1_DIR)
    attempt2_inventory = _inventory(ATTEMPT2_DIR)
    source_hashes = base._source_hashes()
    checks = {
        "immutable_failure_sources_exact": source_hashes == base.EXPECTED_SOURCE_SHA256,
        "d354_measurement_exact": _sha256(base.D354_MEASUREMENT)
        == base.EXPECTED_D354_MEASUREMENT_SHA256,
        "original_audit_invocation_exactly_one": base._json(base.SOURCE_PATHS["invocation"])[
            "audit_invocation_count"
        ]
        == 1,
        "attempt1_actual_inventory_exact": set(attempt1_inventory)
        == {"d355_attempt1_failure_observability_preregistration.json"},
        "attempt2_actual_inventory_exact": set(attempt2_inventory)
        == {"d355_attempt2_failure_observability_preregistration.json"},
        "attempt2_prepare_recorded_fail": base._json(
            ATTEMPT2_DIR / "d355_attempt2_failure_observability_preregistration.json"
        ).get("pass")
        is False,
        "repo_root_inserted": str(REPO) == sys.path[0],
    }
    payload = {
        "artifact": "D355_ATTEMPT3_FAILURE_OBSERVABILITY_INVENTORY_REPAIR_PREREGISTRATION_V1",
        "role": "failure render only; never a provenance audit retry",
        "new_operational_variable": ["actual_prior_observability_inventory"],
        "source_hashes": source_hashes,
        "expected_source_hashes": base.EXPECTED_SOURCE_SHA256,
        "prior_observability_inventories": {
            "attempt1": attempt1_inventory,
            "attempt2": attempt2_inventory,
        },
        "harness": {"sha256": _sha256(Path(__file__).resolve())},
        "wrapper_harness": {
            "path": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "shared_renderer": {
            "path": "sim_scripts/cyl34_top_view_d355_attempt1_failure_observability.py",
            "sha256": _sha256(
                REPO / "sim_scripts/cyl34_top_view_d355_attempt1_failure_observability.py"
            ),
        },
        "scope_guards": {
            "second_audit_count": 0,
            "pxr_import_count": 0,
            "isaac_launch_count": 0,
            "patch_hash_computation_count": 0,
            "q5_evaluation_count": 0,
            "controlled_physics_steps": 0,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, payload)
    if not payload["pass"]:
        raise RuntimeError(f"attempt3 prepare failed: {checks}")
    print(json.dumps({"prepared": True, "path": str(PREREG_PATH.relative_to(REPO))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "render", "finalize"], required=True)
    parser.add_argument("--confirm-visual-inspection", action="store_true")
    args = parser.parse_args()
    _bind_paths()
    if args.stage == "prepare":
        _prepare()
    elif args.stage == "render":
        base._render()
    else:
        base._finalize(args.confirm_visual_inspection)


if __name__ == "__main__":
    main()
