#!/usr/bin/env python3
"""D377 one-shot worker: D375 workload plus one StageCache erase.

The frozen D375 worker source is transformed in memory with four registered,
exact text edits.  The only lifecycle mutation is one
``UsdUtils.StageCache.Get().Erase(stage)`` immediately after the inherited
successful PhysX detach.  The other edits add durable lifecycle observations
and make the inherited worker use forward-only D377 artifact names.

No SimulationContext, reset, timeline play/commit, physics step, public
forward, q5 sample, contact query, cylinder operation, asset write, collider
regeneration, or target/IK/path/physics-setting change is introduced here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
D375_WORKER = (
    REPO / "sim_scripts/cyl34_top_view_d375_p34_live_asset_identity_contract_repair_worker.py"
)
EXPECTED_D375_WORKER_SHA256 = (
    "4b2bbef3cf445ef4c9c9a8de2bac8a01087180502ef134129f9bcfb444020fa4"
)
EXPECTED_HEAD = "e30f7f99d44252f509e383627738f3ad7967ea93"


OLD_DETACH = '''                get_physx_simulation_interface().detach_stage()
                counters["physx_stage_detaches"] = 1
'''

NEW_DETACH = '''                _phase(out_dir, "physx_stage_detach_start", stage_id=stage_id)
                get_physx_simulation_interface().detach_stage()
                counters["physx_stage_detaches"] = 1
                _phase(out_dir, "physx_stage_detach_end", stage_id=stage_id)
                from pxr import UsdUtils
                cache = UsdUtils.StageCache.Get()
                before_id = cache.GetId(stage)
                found_before = cache.Find(before_id) if before_id.IsValid() else None
                before = {
                    "contains_stage": bool(cache.Contains(stage)),
                    "id_valid": bool(before_id.IsValid()),
                    "id_int": int(before_id.ToLongInt()) if before_id.IsValid() else None,
                    "id_matches_registered_stage_id": bool(
                        before_id.IsValid() and int(before_id.ToLongInt()) == int(stage_id)
                    ),
                    "find_old_id_present": found_before is not None,
                    "find_old_id_matches_stage": bool(found_before == stage),
                }
                _phase(out_dir, "stagecache_erase_before", **before)
                counters["stagecache_erase_calls"] = counters.get("stagecache_erase_calls", 0) + 1
                _phase(
                    out_dir,
                    "stagecache_erase_call_start",
                    erase_call_ordinal=counters["stagecache_erase_calls"],
                )
                erase_return = bool(cache.Erase(stage))
                _phase(out_dir, "stagecache_erase_call_end", erase_return=erase_return)
                after_id = cache.GetId(stage)
                found_after = cache.Find(before_id) if before_id.IsValid() else None
                after = {
                    "contains_stage": bool(cache.Contains(stage)),
                    "id_valid": bool(after_id.IsValid()),
                    "id_int": int(after_id.ToLongInt()) if after_id.IsValid() else None,
                    "find_old_id_present": found_after is not None,
                }
                erase_checks = {
                    "before_contains_true": before["contains_stage"] is True,
                    "before_id_valid": before["id_valid"] is True,
                    "before_id_matches_registered_stage_id": before[
                        "id_matches_registered_stage_id"
                    ] is True,
                    "before_find_matches_stage": before["find_old_id_matches_stage"] is True,
                    "erase_return_true": erase_return is True,
                    "erase_call_count_exactly_one": counters["stagecache_erase_calls"] == 1,
                    "after_contains_false": after["contains_stage"] is False,
                    "after_id_invalid": after["id_valid"] is False,
                    "after_find_old_id_absent": after["find_old_id_present"] is False,
                    "python_stage_reference_retained": stage is not None,
                }
                result["stagecache_erase"] = {
                    "api": "UsdUtils.StageCache.Get().Erase(stage)",
                    "placement": "immediately_after_successful_physx_detach",
                    "before": before,
                    "erase_return": erase_return,
                    "after": after,
                    "python_stage_reference_retained": True,
                    "checks": erase_checks,
                    "pass": all(erase_checks.values()),
                }
                _phase(
                    out_dir,
                    "stagecache_erase_after",
                    **after,
                    erase_contract_pass=result["stagecache_erase"]["pass"],
                )
'''

OLD_PROTOCOL = '''            and counters["physx_stage_detaches"] == 1
            and counters["physx_property_queries"] == 2
'''

NEW_PROTOCOL = '''            and counters["physx_stage_detaches"] == 1
            and counters.get("stagecache_erase_calls") == 1
            and result.get("stagecache_erase", {}).get("pass") is True
            and counters["physx_property_queries"] == 2
'''

OLD_PRECLOSE = '''                "counters": counters,
                "timeline_after": result.get("timeline_after"),
                "worker_protocol_pass": result["worker_protocol_pass"],
'''

NEW_PRECLOSE = '''                "counters": counters,
                "timeline_after": result.get("timeline_after"),
                "stagecache_erase": result.get("stagecache_erase"),
                "phase_prefix_bytes": (out_dir / PHASE_NAME).stat().st_size,
                "phase_prefix_sha256": _sha(out_dir / PHASE_NAME),
                "worker_protocol_pass": result["worker_protocol_pass"],
'''

OLD_CLOSE = '''        if launcher is not None:
            launcher.app.close()
'''

NEW_CLOSE = '''        if launcher is not None:
            _phase(out_dir, "simulation_app_close_start")
            launcher.app.close()
            _phase(out_dir, "simulation_app_close_returned_optional")
'''


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source() -> str:
    payload = D375_WORKER.read_bytes()
    observed = _sha_bytes(payload)
    if observed != EXPECTED_D375_WORKER_SHA256:
        raise RuntimeError(
            f"frozen D375 worker hash drift: {observed} != {EXPECTED_D375_WORKER_SHA256}"
        )
    return payload.decode("utf-8")


def transformed_source() -> str:
    source = _source()
    replacements = (
        ("detach_then_erase", OLD_DETACH, NEW_DETACH),
        ("erase_protocol_gate", OLD_PROTOCOL, NEW_PROTOCOL),
        ("preclose_erase_binding", OLD_PRECLOSE, NEW_PRECLOSE),
        ("close_phase_markers", OLD_CLOSE, NEW_CLOSE),
    )
    for name, old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(f"D377 exact transform {name} expected one match, observed {count}")
        source = source.replace(old, new, 1)
    ast.parse(source, filename=str(D375_WORKER))
    return source


def _call_counts(source: str) -> dict[str, int]:
    tree = ast.parse(source)
    names = {
        "detach_stage",
        "Erase",
        "close",
        "update",
        "step",
        "forward",
        "play",
        "commit",
        "reset",
        "collect",
    }
    counts = {name: 0 for name in sorted(names)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else (
            function.id if isinstance(function, ast.Name) else None
        )
        if name in counts:
            counts[name] += 1
    return counts


def source_attestation() -> dict[str, Any]:
    source = transformed_source()
    calls = _call_counts(source)
    checks = {
        "frozen_d375_worker_hash_exact": _sha_bytes(D375_WORKER.read_bytes())
        == EXPECTED_D375_WORKER_SHA256,
        "explicit_detach_call_exactly_one": calls["detach_stage"] == 1,
        "explicit_stagecache_erase_call_exactly_one": calls["Erase"] == 1,
        "explicit_simulation_app_close_call_exactly_one": calls["close"] == 1,
        "no_app_update_call": "launcher.app.update(" not in source
        and "simulation_app.update(" not in source,
        "no_step_call": calls["step"] == 0,
        "no_public_forward_call": calls["forward"] == 0,
        "no_timeline_play_call": calls["play"] == 0,
        "no_timeline_commit_call": calls["commit"] == 0,
        "no_reset_call": calls["reset"] == 0,
        "no_gc_collect_call": calls["collect"] == 0,
        "no_simulation_context": "SimulationContext(" not in source,
        "no_skip_cleanup_change": "skip_cleanup" not in source,
        "no_postuse_stage_reference_release": source.count("stage = None") == 1,
    }
    return {
        "artifact": "D377_D375_EXACT_SOURCE_TRANSFORM_ATTESTATION_V1",
        "frozen_source_path": str(D375_WORKER.relative_to(REPO)),
        "frozen_source_sha256": EXPECTED_D375_WORKER_SHA256,
        "transformed_source_sha256": _sha_bytes(source.encode("utf-8")),
        "registered_replacements": [
            "detach_then_one_stagecache_erase",
            "erase_protocol_gate",
            "preclose_erase_hash_binding",
            "close_start_and_optional_return_markers",
        ],
        "semantic_lifecycle_variable": "explicit_stagecache_erase_after_physx_detach_v1",
        "call_counts": calls,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _load_derived_namespace() -> dict[str, Any]:
    source = transformed_source()
    namespace: dict[str, Any] = {
        "__name__": "d377_exact_derivative_of_frozen_d375_worker",
        "__file__": str(D375_WORKER),
        "__package__": None,
    }
    exec(compile(source, str(D375_WORKER), "exec"), namespace)
    namespace["EXPECTED_HEAD"] = EXPECTED_HEAD
    namespace["CLAIM_NAME"] = "d377_worker_claim.json"
    namespace["SUMMARY_NAME"] = "d377_worker_raw_summary.json"
    namespace["PRECLOSE_NAME"] = "d377_worker_preclose_sentinel.json"
    namespace["EXCEPTION_NAME"] = "d377_worker_exception.json"
    namespace["WITNESS_DIR_NAME"] = "callback_witnesses"
    namespace["PHASE_NAME"] = "d377_phase_markers.jsonl"
    return namespace


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--source-attestation":
        print(json.dumps(source_attestation(), indent=2, sort_keys=True))
        return 0
    attestation = source_attestation()
    if not attestation["pass"]:
        raise RuntimeError(f"D377 source attestation failed: {attestation}")
    namespace = _load_derived_namespace()
    return int(namespace["main"]())


if __name__ == "__main__":
    raise SystemExit(main())
