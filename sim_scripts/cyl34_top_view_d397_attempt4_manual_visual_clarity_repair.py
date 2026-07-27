#!/usr/bin/env python3
"""D397 attempt4: manual visual-clarity repair only.

This forward-only wrapper reads the frozen attempt2 scientific evidence and
the frozen attempt3 presentation failure.  It removes the redundant per-panel
failure annotation that overlapped axis text and tightens only the embedded
Rerun cameras.  It does not rerun construction or invoke Isaac/PhysX.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
ATTEMPT3_SCRIPT = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d397_attempt3_failure_visualization_repair.py"
)


def _load_attempt3() -> Any:
    spec = importlib.util.spec_from_file_location("d397_a4_base", ATTEMPT3_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {ATTEMPT3_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A3 = _load_attempt3()
ATTEMPT = "attempt4_manual_visual_clarity_repair"
NEW_VARIABLES = [
    "annotation_safe_board_and_tighter_exploded_camera_v1"
]
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d397/"
    "attempt4_manual_visual_clarity_repair"
)
ATTEMPT3_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d397/"
    "attempt3_failure_visualization_repair"
)
ATTEMPT3_COMPLETION = ATTEMPT3_DIR / "d397_attempt3_completion_summary.json"
ATTEMPT3_MANUAL = (
    ATTEMPT3_DIR / "d397_attempt3_manual_visual_inspection.json"
)

PREREG = OUT_DIR / "d397_attempt4_preregistration.json"
PHASES = OUT_DIR / "d397_attempt4_phase_markers.jsonl"
OBSERVE_INVOCATION = OUT_DIR / "d397_attempt4_observe_invocation.json"
EVIDENCE = OUT_DIR / "d397_attempt4_presentation_evidence.json"
GEOMETRY = OUT_DIR / "d397_attempt4_display_geometry.json"
BOARD = OUT_DIR / "d397_failure_by_parent_clean_1920x1080.png"
LAYOUT = OUT_DIR / "d397_attempt4_layout_validation.json"
RRD = OUT_DIR / "d397_failure_exploded_tight.rerun.rrd"
RBL = OUT_DIR / "d397_failure_exploded_tight.rerun.rbl"
RERUN_VALIDATION = OUT_DIR / "d397_attempt4_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d397_attempt4_rerun_inspection.png"
OBSERVABILITY = OUT_DIR / "d397_attempt4_observability_claim.json"
MANUAL_TEMPLATE = OUT_DIR / "d397_attempt4_manual_template.json"
MANUAL = OUT_DIR / "d397_attempt4_manual_visual_inspection.json"
FINALIZE_INVOCATION = OUT_DIR / "d397_attempt4_finalize_invocation.json"
COMPLETION = OUT_DIR / "d397_attempt4_completion_summary.json"
FAILURE = OUT_DIR / "d397_attempt4_runtime_failure.json"

A3.SCRIPT = SCRIPT
A3.ATTEMPT = ATTEMPT
A3.NEW_VARIABLES = NEW_VARIABLES
A3.OUT_DIR = OUT_DIR
A3.PREREG = PREREG
A3.PHASES = PHASES
A3.OBSERVE_INVOCATION = OBSERVE_INVOCATION
A3.EVIDENCE = EVIDENCE
A3.GEOMETRY = GEOMETRY
A3.BOARD = BOARD
A3.LAYOUT = LAYOUT
A3.RRD = RRD
A3.RBL = RBL
A3.RERUN_VALIDATION = RERUN_VALIDATION
A3.RERUN_SCREENSHOT = RERUN_SCREENSHOT
A3.OBSERVABILITY = OBSERVABILITY
A3.MANUAL_TEMPLATE = MANUAL_TEMPLATE
A3.MANUAL = MANUAL
A3.FINALIZE_INVOCATION = FINALIZE_INVOCATION
A3.COMPLETION = COMPLETION
A3.FAILURE = FAILURE
A3.SOURCE_COMPLETION = ATTEMPT3_COMPLETION
A3.SOURCE_MANUAL = ATTEMPT3_MANUAL
A3.EXPECTED_INPUT_SHA256 = {
    ATTEMPT3_SCRIPT: (
        "cb257264da8608dffd22f5f20bc8685d4da71dcfb0c354d7700989f4ba686212"
    ),
    A3.BASE_SCRIPT: (
        "52745beab46bc695467dd8d676a06b30fa3ea873c7dcad685861e65cfecf4b36"
    ),
    A3.SOURCE_EVIDENCE: (
        "ea7fd61c38f12b9e03f4e7154536579b831c6f85703bfd4d14e34807cdf327b6"
    ),
    A3.SOURCE_GEOMETRY: (
        "b9a44d430f647e45292fe71804bd17e6f53bf37eea28913389316beac60fa623"
    ),
    A3.SOURCE_WORKER: (
        "2bac06043e35e095660ed3a0562930f98425a9eba436dc5b72e58f313ed1ed79"
    ),
    ATTEMPT3_COMPLETION: (
        "cc3a6338e35311636d34f45e282397f2191a9d066706cf1987cfd207163ff232"
    ),
    ATTEMPT3_MANUAL: (
        "55074ae1a4a1c07fe6e39103a90931054a75e4f87544e6829bea55f431b3b1ca"
    ),
    A3.VIZ_DEBUG: (
        "4b5f821ad43652f529dfaa2f92b2826d9cd4973635e34521cc2b3a93ab0193d0"
    ),
    A3.RERUN_CONTRACT: (
        "aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e"
    ),
}
A3._STAGE_WRITE_STARTED = False

A3.BASE.ATTEMPT = ATTEMPT
A3.BASE.OUT_DIR = OUT_DIR
A3.BASE.EVIDENCE = EVIDENCE
A3.BASE.GEOMETRY = GEOMETRY
A3.BASE.BOARD = BOARD
A3.BASE.LAYOUT = LAYOUT
A3.BASE.RRD = RRD
A3.BASE.RBL = RBL
A3.BASE.RERUN_VALIDATION = RERUN_VALIDATION
A3.BASE.RERUN_SCREENSHOT = RERUN_SCREENSHOT

_ORIGINAL_WRITE = A3._write
_ORIGINAL_RENDER_BOARD = A3._render_board


def _write(path: Path, value: Any) -> None:
    if isinstance(value, dict):
        value = dict(value)
        artifact = value.get("artifact")
        if isinstance(artifact, str):
            value["artifact"] = artifact.replace("ATTEMPT3", "ATTEMPT4")
        if path == PREREG:
            value["purpose"] = (
                "repair only the frozen attempt3 manual presentation failure "
                "by removing its duplicate per-panel failure annotation and "
                "tightening the two embedded Rerun cameras"
            )
            value["immutable_attempt3_failure_provenance"] = {
                "completion": A3._rel(ATTEMPT3_COMPLETION),
                "manual": A3._rel(ATTEMPT3_MANUAL),
            }
    _ORIGINAL_WRITE(path, value)


A3._write = _write


def _render_board(
    source_evidence: dict[str, Any], source_geometry: dict[str, Any]
) -> dict[str, Any]:
    os.environ.setdefault(
        "MPLCONFIGDIR", "/tmp/roarm_d397_attempt4_matplotlib"
    )
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    original_text2d = Axes3D.text2D

    def annotation_safe_text2d(
        self: Any, x: float, y: float, text: str, *args: Any, **kwargs: Any
    ) -> Any:
        if text == "no admissible next shared plane":
            return None
        return original_text2d(self, x, y, text, *args, **kwargs)

    Axes3D.text2D = annotation_safe_text2d
    try:
        result = _ORIGINAL_RENDER_BOARD(source_evidence, source_geometry)
    finally:
        Axes3D.text2D = original_text2d
    result["artifact"] = "D397_ATTEMPT4_LAYOUT_VALIDATION_V1"
    return result


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    def view(
        body: str,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> Any:
        return rrb.Spatial3DView(
            origin="/",
            contents=f"/d397/{body}/**",
            name=f"D397 {body} exploded failure geometry",
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=position,
                look_target=target,
                eye_up=(0.0, 0.0, 1.0),
            ),
            spatial_information=rrb.SpatialInformation(
                target_frame="tf#/",
                show_axes=True,
                show_bounding_box=False,
            ),
        )

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                view(
                    "link5",
                    (0.160, -0.120, 0.145),
                    (0.035, -0.001, 0.062),
                ),
                view(
                    "gripper_link",
                    (0.300, -0.120, 0.075),
                    (0.126, 0.060, -0.019),
                ),
                column_shares=[0.43, 0.57],
            ),
            rrb.TextDocumentView(
                origin="/metadata/run",
                contents="/metadata/run",
                name="D397 immutable failure authority",
            ),
            row_shares=[0.90, 0.10],
        ),
        rrb.TimePanel(state=rrb.PanelState.Hidden),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


A3._render_board = _render_board
A3._build_blueprint = _build_blueprint


if __name__ == "__main__":
    raise SystemExit(A3.main())
