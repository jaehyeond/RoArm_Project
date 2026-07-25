#!/usr/bin/env python3
"""Forward-only D384 attempt6 Rerun glyph-compatibility repair.

Attempt5 completed its worker and placed notifications in the empty buffer,
but Rerun's bundled font rendered Korean view titles and decision text as
square glyphs.  This wrapper freezes that evidence and changes only Rerun
presentation strings to ASCII English.  Canonical evidence, geometry, board,
camera-layout contract, and all runtime boundaries remain unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ATTEMPT5_SCRIPT = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d384_attempt5_project_root_import_bootstrap_repair.py"
)
SPEC = importlib.util.spec_from_file_location(
    "d384_attempt5_frozen",
    ATTEMPT5_SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load frozen attempt5: {ATTEMPT5_SCRIPT}")
frozen5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen5)
base = frozen5.base

ATTEMPT = "attempt6_rerun_ascii_glyph_compatibility_repair"
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d384"
    / ATTEMPT
)
SCRIPT_PATH = Path(__file__).resolve()
ATTEMPT5_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d384"
    / "attempt5_project_root_import_bootstrap_repair"
)
ATTEMPT5_SUPERVISOR = (
    ATTEMPT5_DIR / "d384_attempt5_worker_supervisor.json"
)
ATTEMPT5_MANUAL = (
    ATTEMPT5_DIR / "d384_attempt5_manual_visual_inspection.json"
)
ATTEMPT5_COMPLETION = (
    ATTEMPT5_DIR / "d384_attempt5_completion_summary.json"
)
ATTEMPT5_SCREENSHOT = (
    ATTEMPT5_DIR / "d384_attempt5_rerun_inspection.png"
)
PREFLIGHT_PATH = OUT_DIR / "d384_attempt6_ascii_glyph_preflight.json"
BOOTSTRAP_RECEIPT = OUT_DIR / "d384_attempt6_import_bootstrap_receipt.json"

EXPECTED_ATTEMPT5_HASHES = {
    "script": (
        "5c77876f33dd14dcba5e626197ed17084e8c765863f8aeadc9a39b30781d09cd"
    ),
    "supervisor": (
        "12e4fa0ca68e509daf54c813294747025c784cd19b0ef24a2b24231d054e7624"
    ),
    "manual": (
        "a7fc0a681573031e659365f8b9f03727b6a3e4300ee21e4faa05c7d3de42e067"
    ),
    "completion": (
        "c7303764ba7823026ecc1487756642d59a978443b8999d4c4cd83917d3b96f70"
    ),
    "screenshot": (
        "dcbaa645ea2223fd787d210c05a4fd868f433f422577a9be7b5eac258e9fa714"
    ),
}

base.ATTEMPT = ATTEMPT
base.OUT_DIR = OUT_DIR
base.SCRIPT_PATH = SCRIPT_PATH

_PATH_NAMES = {
    "PREREG_PATH": "d384_attempt6_preregistration.json",
    "PHASE_PATH": "d384_attempt6_phase_markers.jsonl",
    "INVOCATION_PATH": "d384_attempt6_worker_invocation.json",
    "WORKER_STDOUT": "d384_attempt6_worker_stdout.log",
    "WORKER_STDERR": "d384_attempt6_worker_stderr.log",
    "WORKER_CLAIM": "d384_attempt6_worker_claim.json",
    "SUPERVISOR_PATH": "d384_attempt6_worker_supervisor.json",
    "BOARD_PATH": "d384_attempt6_presentation_board_1920x1080.png",
    "BOARD_LAYOUT": "d384_attempt6_board_layout_validation.json",
    "RECORDING_ONLY": "d384_attempt6_recording_only.rrd",
    "OVERLAY_RRD": "d384_attempt6_summary_overlay.rrd",
    "RBL_PATH": "d384_attempt6_presentation.rbl",
    "PRESENTATION_RRD": "d384_attempt6_presentation.rrd",
    "RECORDING_EQUIVALENCE": "d384_attempt6_recording_equivalence.json",
    "RERUN_VALIDATION": "d384_attempt6_rerun_validation.json",
    "VIEWER_RECEIPT": "d384_attempt6_viewer_receipt.json",
    "RERUN_SCREENSHOT": "d384_attempt6_rerun_inspection.png",
    "MANUAL_TEMPLATE": "d384_attempt6_manual_visual_inspection_template.json",
    "MANUAL_INSPECTION": "d384_attempt6_manual_visual_inspection.json",
    "COMPLETION_PATH": "d384_attempt6_completion_summary.json",
}
for name, filename in _PATH_NAMES.items():
    setattr(base, name, OUT_DIR / filename)


def _source_hashes() -> dict[str, str]:
    return {
        "attempt6_wrapper": base._sha(SCRIPT_PATH),
        "frozen_attempt5_wrapper": base._sha(ATTEMPT5_SCRIPT),
        "frozen_attempt4_wrapper": base._sha(frozen5.ATTEMPT4_SCRIPT),
        "frozen_attempt3_module": base._sha(
            frozen5.frozen4.BASE_SCRIPT
        ),
    }


base._source_hashes = _source_hashes


ASCII_VIEW_NAMES = [
    "Profile prism | authored parent and exact children",
    "Source hull | recursive-partition witness",
    "D384 frozen result and next boundary",
    "Notification buffer | no decision content",
]


def _build_ascii_blueprint() -> Any:
    import rerun.blueprint as rrb

    def spatial(
        contents: str,
        name: str,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> Any:
        return rrb.Spatial3DView(
            origin="/",
            contents=contents,
            name=name,
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=position,
                look_target=target,
                eye_up=(0.0, 0.0, 1.0),
            ),
            spatial_information=rrb.SpatialInformation(
                target_frame="tf#/",
                show_axes=False,
                show_bounding_box=False,
            ),
        )

    geometry = rrb.Horizontal(
        spatial(
            "/d384/profile/**",
            ASCII_VIEW_NAMES[0],
            (0.068, -0.052, 0.004),
            (0.0328, -0.0052, -0.0319),
        ),
        spatial(
            "/d384/source/**",
            ASCII_VIEW_NAMES[1],
            (0.085, -0.105, 0.135),
            (-0.0174, 0.0160, 0.0618),
        ),
        column_shares=[0.5, 0.5],
    )
    decision_area = rrb.Vertical(
        geometry,
        rrb.TextDocumentView(
            origin="/presentation/d384_attempt3/summary",
            contents="/presentation/d384_attempt3/summary",
            name=ASCII_VIEW_NAMES[2],
        ),
        row_shares=[0.72, 0.28],
    )
    notification_buffer = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d384_attempt3/notification_buffer/**",
        name=ASCII_VIEW_NAMES[3],
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=False,
            show_bounding_box=False,
        ),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            decision_area,
            notification_buffer,
            column_shares=[0.74, 0.26],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _ascii_overlay_markdown(evidence: dict[str, Any]) -> str:
    candidates = evidence["repair_candidates"]
    return "\n".join(
        [
            "## D384 frozen design result",
            "",
            "- Failed parts: **17/34** (profile prisms 9, source hulls 8)",
            (
                "- Exact authored partition: **"
                f"{candidates['registered_recursive_partition']['total_collider_parts']}"
                " parts** -> fails the below-128 project budget"
            ),
            (
                "- Exact tetra upper bound: **"
                f"{candidates['exact_tetra_upper_bound']['total_collider_parts']}"
                " parts** -> fails the below-128 project budget"
            ),
            (
                "- Direct points+polygons: theoretical **34 parts**; no "
                "public USD selector; runtime capability is **NULL**"
            ),
            "- Asset / Isaac / PhysX / cylinder / physics / q5 / contact: **0**",
            "- P34 authored-to-cooked identity: **FAIL**",
            "- g0a_pass=false",
            "",
            (
                "**Next separate approval:** semantic low-count redesign of "
                "the eight source hulls. Asset materialization remains blocked."
            ),
        ]
    )


base._build_blueprint = _build_ascii_blueprint
base._overlay_markdown = _ascii_overlay_markdown


def _attempt6_preflight() -> dict[str, Any]:
    supervisor = base._read_json(ATTEMPT5_SUPERVISOR)
    manual = base._read_json(ATTEMPT5_MANUAL)
    completion = base._read_json(ATTEMPT5_COMPLETION)
    observed_hashes = {
        "script": base._sha(ATTEMPT5_SCRIPT),
        "supervisor": base._sha(ATTEMPT5_SUPERVISOR),
        "manual": base._sha(ATTEMPT5_MANUAL),
        "completion": base._sha(ATTEMPT5_COMPLETION),
        "screenshot": base._sha(ATTEMPT5_SCREENSHOT),
    }
    sample_overlay = _ascii_overlay_markdown(
        base._read_json(base.SOURCE_EVIDENCE)
    )
    ascii_payloads = [*ASCII_VIEW_NAMES, sample_overlay]
    checks = {
        "attempt5_hashes_exact": (
            observed_hashes == EXPECTED_ATTEMPT5_HASHES
        ),
        "attempt5_worker_and_viewer_passed": (
            supervisor.get("pass") is True
            and supervisor.get("actual_offline_worker_invocations") == 1
            and supervisor.get("automatic_retries") == 0
        ),
        "attempt5_no_timeout_signal_or_residue": (
            supervisor.get("timed_out") is False
            and supervisor.get("sigterm_sent") is False
            and supervisor.get("sigkill_sent") is False
            and supervisor.get("process_group_alive_after_wait") is False
        ),
        "attempt5_only_decision_summary_check_failed": (
            manual.get("pass") is False
            and manual["inspection_checks"]["decision_summary_visible"]
            is False
            and sum(
                value is False
                for value in manual["inspection_checks"].values()
            )
            == 1
        ),
        "attempt5_completion_visual_fail_preserved": (
            completion.get("presentation_verdict")
            == "D384_ATTEMPT3_PRESENTATION_INTEGRITY_FAIL_STOP"
            and completion.get("canonical_design_verdict")
            == "D384_REPRESENTATION_REPAIR_DESIGN_NO_ADMISSIBLE_LOW_COUNT_CANDIDATE_FAIL_STOP"
        ),
        "replacement_strings_ascii_only": all(
            text.isascii() for text in ascii_payloads
        ),
        "repair_is_rerun_text_compatibility_only": True,
        "attempt5_not_rerun_or_overwritten": True,
    }
    return {
        "artifact": "D384_ATTEMPT6_RERUN_ASCII_GLYPH_PREFLIGHT_V1",
        "attempt5_failure": {
            "classification": (
                "RERUN_KOREAN_GLYPHS_RENDERED_AS_UNREADABLE_SQUARES"
            ),
            "observed_hashes": observed_hashes,
            "attempt5_rerun_or_overwrite": False,
        },
        "repair": {
            "scope": (
                "replace Rerun-only titles and TextDocument with ASCII "
                "English while preserving values, layout, geometry, and board"
            ),
            "new_scientific_or_design_variables": 0,
            "presentation_variables_inherited": list(base.NEW_VARIABLES),
            "ascii_view_names": ASCII_VIEW_NAMES,
            "ascii_overlay_sha256": base._canonical_sha(sample_overlay),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def prepare() -> int:
    preflight = _attempt6_preflight()
    if preflight["pass"] is not True:
        raise RuntimeError(f"attempt6 preflight failed: {preflight}")
    result = base.prepare()
    base._write_json_x(PREFLIGHT_PATH, preflight)
    base._phase(
        "attempt6_rerun_ascii_glyph_preflight_frozen",
        preflight_sha256=base._sha(PREFLIGHT_PATH),
        passed=True,
    )
    return result


def worker() -> int:
    expected = _attempt6_preflight()
    frozen = base._read_json(PREFLIGHT_PATH)
    if frozen != expected or frozen.get("pass") is not True:
        raise RuntimeError("attempt6 glyph preflight drifted")
    receipt = frozen5._bootstrap_project_root()
    if receipt["pass"] is not True:
        raise RuntimeError(f"attempt6 bootstrap failed: {receipt}")
    receipt["artifact"] = "D384_ATTEMPT6_PROJECT_ROOT_IMPORT_RECEIPT_V1"
    base._write_json_x(BOOTSTRAP_RECEIPT, receipt)
    base._phase(
        "attempt6_project_root_inserted_once",
        receipt_sha256=base._sha(BOOTSTRAP_RECEIPT),
        count_after=receipt["count_after"],
        index_after=receipt["index_after"],
    )
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
