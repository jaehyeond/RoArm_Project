#!/usr/bin/env python3
"""D355 failure-observability attempt2: repo-root import path repair only."""

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


OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d355/attempt2_failure_observability"
PREREG_PATH = OUT_DIR / "d355_attempt2_failure_observability_preregistration.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _bind_attempt2_paths() -> None:
    base.OUT_DIR = OUT_DIR
    base.PREREG_PATH = PREREG_PATH
    base.INVOCATION_PATH = OUT_DIR / "d355_attempt2_failure_observability_invocation.json"
    base.RRD_PATH = OUT_DIR / "d355_attempt1_failure_explained.rrd"
    base.RBL_PATH = OUT_DIR / "d355_attempt1_failure_explained.rbl"
    base.PNG_PATH = OUT_DIR / "d355_attempt1_failure_explained_rerun.png"
    base.VALIDATION_PATH = OUT_DIR / "d355_attempt2_failure_observability_validation.json"
    base.SUMMARY_PATH = OUT_DIR / "d355_attempt2_failure_observability_summary.json"
    base.MANUAL_PATH = OUT_DIR / "d355_attempt2_failure_manual_visual_inspection.json"
    base.COMPLETION_PATH = OUT_DIR / "d355_attempt2_failure_observability_completion.json"
    # The shared implementation hashes its own module-level __file__.  Point it
    # at this forward-only wrapper so the attempt2 preregistration pins the code
    # that actually establishes the repaired repo-root import path.
    base.__file__ = str(Path(__file__).resolve())


def _prepare() -> None:
    _bind_attempt2_paths()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes = base._source_hashes()
    attempt1_obs_dir = (
        REPO
        / "claudedocs/runtime_logs/grasp_track/g0a_d355/attempt1_failure_observability"
    )
    attempt1_obs_inventory = {
        path.name: _sha256(path) for path in sorted(attempt1_obs_dir.iterdir()) if path.is_file()
    }
    checks = {
        "immutable_failure_sources_exact": source_hashes == base.EXPECTED_SOURCE_SHA256,
        "d354_measurement_exact": _sha256(base.D354_MEASUREMENT)
        == base.EXPECTED_D354_MEASUREMENT_SHA256,
        "attempt1_audit_invocation_exactly_one": base._json(base.SOURCE_PATHS["invocation"])[
            "audit_invocation_count"
        ]
        == 1,
        "attempt1_observability_preserved": sorted(attempt1_obs_inventory)
        == [
            "d355_attempt1_failure_observability_invocation.json",
            "d355_attempt1_failure_observability_preregistration.json",
        ],
        "repo_root_inserted_before_shared_import": str(REPO) == sys.path[0],
    }
    payload = {
        "artifact": "D355_ATTEMPT2_FAILURE_OBSERVABILITY_PATH_REPAIR_PREREGISTRATION_V1",
        "role": "failure render only; not a second provenance audit",
        "new_operational_variable": ["repo_root_python_import_path"],
        "source_hashes": source_hashes,
        "expected_source_hashes": base.EXPECTED_SOURCE_SHA256,
        "attempt1_observability_inventory_sha256": attempt1_obs_inventory,
        "wrapper_harness": {
            "path": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "shared_renderer": {
            "path": str(
                (
                    REPO
                    / "sim_scripts/cyl34_top_view_d355_attempt1_failure_observability.py"
                ).relative_to(REPO)
            ),
            "sha256": _sha256(
                REPO / "sim_scripts/cyl34_top_view_d355_attempt1_failure_observability.py"
            ),
        },
        # The shared render guard reads this field after base.__file__ is rebound.
        "harness": {"sha256": _sha256(Path(__file__).resolve())},
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
        raise RuntimeError(f"attempt2 failure observability prepare failed: {checks}")
    print(json.dumps({"prepared": True, "path": str(PREREG_PATH.relative_to(REPO))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "render", "finalize"], required=True)
    parser.add_argument("--confirm-visual-inspection", action="store_true")
    args = parser.parse_args()
    _bind_attempt2_paths()
    if args.stage == "prepare":
        _prepare()
    elif args.stage == "render":
        base._render()
    else:
        base._finalize(args.confirm_visual_inspection)


if __name__ == "__main__":
    main()
