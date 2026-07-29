#!/usr/bin/env python3
"""D405 thin controller wrapper: observability first-live-render repair.

D404 actual attempt1 passed the ENTIRE technical chain for the first time
(``technical_pass=true``: repairs 1-4 live-passed, SDF cook drained, PhysX
property queries VALID, mass gate passed) and then fail-stopped in the
observability branch, which had never executed live during D400-D404.  The
D405 static preparation replayed that frozen branch offline against the
frozen D404 evidence and surfaced two further latent defects of the same
never-executed code path.  This wrapper carries ONE bundled new variable
(``observability_first_live_render_repair_v1``, the same bundling the D404
case used for its four gate-contract repairs) consisting of three minimal
reactive repairs, all confined to the controller process:

1. **Import path** (observed live in D404): a script-path launch puts
   ``sim_scripts/`` at ``sys.path[0]`` and never adds the repo root, so
   ``from roarm_rl.rerun_contract import validate_rerun_artifact``
   (preflight.py:2779) raised ``ModuleNotFoundError``.  The repo root is
   appended to the END of ``sys.path`` (no stdlib/site-packages shadowing;
   the worker subprocess never imports ``roarm_rl``).
2. **Headless logical window size** (observed in the D405 replay): the
   frozen ``_write_rerun`` requests ``screenshot_window_size='1920x1080'``
   intending a 1920x1080 PHYSICAL screenshot, but the installed rerun
   0.34.1 CLI documents ``--window-size`` in LOGICAL points and its
   headless harness renders at a fixed pixels-per-point of 2.0 (empirically
   environment-independent), yielding 3840x2160.  The module-global
   function object ``roarm_rl.rerun_contract.validate_rerun_artifact`` is
   wrapped so exactly that frozen literal is translated to the logical
   request ``'960x540'``; the frozen physical gates then evaluate the real
   1920x1080 PNG.
3. **Blueprint text-view multiplicity** (observed in the D405 replay): the
   frozen ``_build_blueprint`` groups 4 and 3 TextDocument entities into
   single ``TextDocumentView``s, which rerun 0.34.1 renders as "Can only
   show one text document at a time; was given 3" instead of the content,
   and viewer toasts anchor over the frozen right text column.  Via the
   same loader-seam pattern the frozen D402 controller already uses on
   ``d401._load_frozen_d400_controller``, the loaded frozen preflight
   module's ``_build_blueprint`` function object is replaced with a layout
   of the SAME nine logged entities: the frozen Spatial3DView verbatim on
   top (full width, so toasts overlap only its empty sky), then one
   TextDocumentView per text entity.  No entity, timeline, component, or
   logging change of any kind.

A fail-closed pre-delegation probe verifies the observability
prerequisites (contract resolution, Rerun SDK ``0.34.1`` pin, numpy D326
pin ``1.26.0``, frozen ``RERUN_CLI`` presence) BEFORE any frozen module
load or forward-only write, so a bad configuration consumes nothing.
Everything else is a pure path/provenance rebind onto the hash-pinned
frozen D404 controller (D405 -> D404 -> D403 -> D402 -> D401 -> D400, all
byte-for-byte unchanged).  Importing this module does not import or launch
Isaac, Kit, PhysX, Warp, CUDA, Rerun, or the Worker.  The single runtime
invocation is authorized by the user's 2026-07-28 sequential directive for
the next minimal rung and is consumed by this attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = Path(__file__).resolve()
WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d405_d404_observability_import_path_repair_worker.py"
)
D404_CONTROLLER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d404_d403_authored_derivative_gate_contract_repair_controller.py"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d405/"
    "attempt1_d404_observability_import_path_repair"
)
PREREG_PATH = OUT_DIR / "d405_preregistration.json"
ATTESTATION_PATH = OUT_DIR / "d405_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d405_proposed_runtime_hash_tuple.json"
RUNTIME_MANIFEST_PATH = OUT_DIR / "d400_runtime_freeze_manifest.json"
PHASE_PATH = OUT_DIR / "d400_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d400_worker_invocation.json"
CLAIM_PATH = OUT_DIR / "d400_worker_claim.json"
KIT_LOG_PATH = OUT_DIR / "d400_kit_log.txt"
RAW_PATH = OUT_DIR / "d400_worker_raw_summary.json"
PRECLOSE_PATH = OUT_DIR / "d400_worker_preclose_sentinel.json"
OWNER_EVIDENCE_PATH = OUT_DIR / "d400_live_configuration_owner_evidence.json"
SUPERVISOR_PATH = OUT_DIR / "d400_worker_supervisor.json"
COMPLETION_PATH = OUT_DIR / "d400_completion_summary.json"
RRD_PATH = OUT_DIR / "d400_sdf_preflight.rrd"
RBL_PATH = OUT_DIR / "d400_sdf_preflight.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d400_rerun_validation.json"
BOARD_PATH = OUT_DIR / "d400_decision_board_1920x1080.png"
RERUN_SCREENSHOT_PATH = OUT_DIR / "d400_rerun_viewer_1920x1080.png"
RERUN_RECEIPT_PATH = OUT_DIR / "d400_rerun_render_receipt.json"
MANUAL_INSPECTION_PATH = OUT_DIR / "d400_manual_visual_inspection.json"
COLLISION_ASSET_ROOT = OUT_DIR / "collision_asset"

EXPECTED_PREREG_SHA256 = (
    "f63e6c69953926697cbb87202fbbb24bd751c897d2dca370373157dd1f4195b2"
)
EXPECTED_D404_CONTROLLER_SHA256 = (
    "75070713db433ade735b2b227a1c642c6355fef352d82d12a3069c69b7642cef"
)

# Derived from the frozen preflight.py:101 RERUN_VERSION and verified equal
# to roarm_rl.rerun_contract.RERUN_CONTRACT_VERSION by the static fixtures.
EXPECTED_RERUN_SDK_VERSION = "0.34.1"
# Derived from the D326 IsaacLab environment package rule.
EXPECTED_NUMPY_VERSION = "1.26.0"
# Mirror of the frozen preflight.py:72 exact-path pin (verified equal by the
# static fixtures; the frozen environment gate re-checks it live).
EXPECTED_RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")

ROARM_RL_CONTRACT_FILE = REPO / "roarm_rl" / "rerun_contract.py"

# The frozen physical screenshot contract (preflight.py: _png_info
# exact_1920x1080, manual original_resolution [1920,1080]) divided by the
# installed rerun 0.34.1 headless harness's fixed pixels-per-point of 2.0
# (empirical: identical under DISPLAY=:1, no DISPLAY, and Xvfb).
FROZEN_PHYSICAL_WINDOW_SIZE = "1920x1080"
HEADLESS_LOGICAL_WINDOW_SIZE = "960x540"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repair_observability_import_path() -> None:
    """Repair 1: append the repo root to the END of sys.path (idempotent)."""

    root = str(REPO)
    if root not in sys.path:
        sys.path.append(root)


def _observability_preflight_probe() -> dict[str, object]:
    """Fail closed before any frozen module load or forward-only write."""

    contract = importlib.import_module("roarm_rl.rerun_contract")
    resolved = Path(contract.__file__).resolve()
    if resolved != ROARM_RL_CONTRACT_FILE.resolve():
        raise RuntimeError(
            "D405 probe: roarm_rl.rerun_contract resolved outside the repo: "
            f"{resolved}"
        )
    if not callable(getattr(contract, "validate_rerun_artifact", None)):
        raise RuntimeError(
            "D405 probe: validate_rerun_artifact missing or not callable"
        )
    if str(contract.RERUN_CONTRACT_VERSION) != EXPECTED_RERUN_SDK_VERSION:
        raise RuntimeError(
            "D405 probe: RERUN_CONTRACT_VERSION drift: "
            f"{contract.RERUN_CONTRACT_VERSION}"
        )
    rerun = importlib.import_module("rerun")
    if str(rerun.__version__) != EXPECTED_RERUN_SDK_VERSION:
        raise RuntimeError(
            f"D405 probe: Rerun SDK version drift: {rerun.__version__}"
        )
    numpy = importlib.import_module("numpy")
    if str(numpy.__version__) != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(
            f"D405 probe: numpy version drift (D326): {numpy.__version__}"
        )
    if not EXPECTED_RERUN_CLI.is_file():
        raise RuntimeError(
            f"D405 probe: frozen RERUN_CLI missing: {EXPECTED_RERUN_CLI}"
        )
    return {
        "roarm_rl_rerun_contract_file": str(resolved),
        "rerun_sdk_version": str(rerun.__version__),
        "numpy_version": str(numpy.__version__),
        "rerun_cli_exists": True,
    }


def _install_screenshot_logical_size_repair() -> None:
    """Repair 2: translate the frozen physical size literal to logical."""

    contract = importlib.import_module("roarm_rl.rerun_contract")
    if getattr(contract, "_d405_screenshot_repair_installed", False):
        return
    frozen_validate = contract.validate_rerun_artifact

    def validate_with_headless_logical_size(path, **kwargs) -> dict[str, Any]:
        if kwargs.get("screenshot_window_size") == FROZEN_PHYSICAL_WINDOW_SIZE:
            kwargs["screenshot_window_size"] = HEADLESS_LOGICAL_WINDOW_SIZE
        return frozen_validate(path, **kwargs)

    contract._d405_frozen_validate = frozen_validate
    contract.validate_rerun_artifact = validate_with_headless_logical_size
    contract._d405_screenshot_repair_installed = True


def _build_single_document_blueprint() -> Any:
    """Repair 3: same nine entities, one TextDocumentView per text entity.

    The Spatial3DView parameters are byte-for-byte the frozen
    preflight.py:2441-2456 values.  It spans the full window width on top
    (share 0.55) so viewer toasts, which anchor at the window top-right,
    overlap only its empty sky and never a text panel.
    """

    import rerun.blueprint as rrb

    spatial = rrb.Spatial3DView(
        origin="/",
        contents="/d400/inspection/**",
        name="D400 source, live SDF input, and frozen link5 A64",
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=(0.28, -0.34, 0.22),
            look_target=(0.03, 0.0, -0.01),
            eye_up=(0.0, 0.0, 1.0),
        ),
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=True,
            show_bounding_box=False,
        ),
    )

    def document_view(path: str, name: str) -> Any:
        return rrb.TextDocumentView(origin=path, name=name)

    status_row = rrb.Horizontal(
        document_view(
            "/d400/status/api_token_attributes", "API token attributes"
        ),
        document_view(
            "/d400/status/inventory_owner_query", "Owner inventory query"
        ),
        document_view("/d400/status/cook_queue", "Cook queue"),
        document_view(
            "/d400/status/mass_counters_instance_state",
            "Mass, counters, instance state",
        ),
    )
    phase_row = rrb.Horizontal(
        document_view(
            "/d400/phase/source_baseline", "Phase 0 source baseline"
        ),
        document_view(
            "/d400/phase/live_configuration", "Phase 1 live configuration"
        ),
        document_view(
            "/d400/phase/post_query_decision", "Phase 2 post-query decision"
        ),
    )
    return rrb.Blueprint(
        rrb.Vertical(
            spatial, status_row, phase_row, row_shares=[0.55, 0.27, 0.18]
        ),
        rrb.TimePanel(state=rrb.PanelState.Hidden),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _install_observability_render_repairs(base: ModuleType) -> None:
    """Install repairs 2-3 for the loaded frozen D400 preflight module."""

    _install_screenshot_logical_size_repair()
    base._build_blueprint = _build_single_document_blueprint


def _install_chain_render_repair(d404: ModuleType) -> None:
    """Hook the loader seams so the loaded frozen preflight carries repair 3.

    Same pattern as the frozen D402 controller's own wrap of
    ``d401._load_frozen_d400_controller`` (D402 controller lines 222-233)
    and the D404 worker's triple seam; both compose safely because each
    wrapper replaces a different attribute of the loaded base module.
    """

    frozen_load_d403 = d404._load_frozen_d403_controller

    def load_d403_with_render_repair() -> ModuleType:
        d403 = frozen_load_d403()
        frozen_load_d402 = d403._load_frozen_d402_controller

        def load_d402_with_render_repair() -> ModuleType:
            d402 = frozen_load_d402()
            frozen_load_d401 = d402._load_frozen_d401_controller

            def load_d401_with_render_repair() -> ModuleType:
                d401 = frozen_load_d401()
                frozen_load_base = d401._load_frozen_d400_controller

                def load_base_with_render_repair() -> ModuleType:
                    base = frozen_load_base()
                    _install_observability_render_repairs(base)
                    return base

                d401._load_frozen_d400_controller = (
                    load_base_with_render_repair
                )
                return d401

            d402._load_frozen_d401_controller = load_d401_with_render_repair
            return d402

        d403._load_frozen_d402_controller = load_d402_with_render_repair
        return d403

    d404._load_frozen_d403_controller = load_d403_with_render_repair


def _load_frozen_d404_controller() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D405 controller requires python -B before loading the frozen "
            "D404 controller"
        )
    observed = _sha(D404_CONTROLLER_PATH)
    if observed != EXPECTED_D404_CONTROLLER_SHA256:
        raise RuntimeError(
            "frozen D404 controller hash drift: "
            f"{observed} != {EXPECTED_D404_CONTROLLER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d405_frozen_d404_controller",
        D404_CONTROLLER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D404 controller import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_d404_paths(d404: ModuleType) -> None:
    bindings = {
        "CONTROLLER_PATH": CONTROLLER_PATH,
        "WORKER_PATH": WORKER_PATH,
        "OUT_DIR": OUT_DIR,
        "PREREG_PATH": PREREG_PATH,
        "ATTESTATION_PATH": ATTESTATION_PATH,
        "TUPLE_PATH": TUPLE_PATH,
        "RUNTIME_MANIFEST_PATH": RUNTIME_MANIFEST_PATH,
        "PHASE_PATH": PHASE_PATH,
        "INVOCATION_PATH": INVOCATION_PATH,
        "CLAIM_PATH": CLAIM_PATH,
        "KIT_LOG_PATH": KIT_LOG_PATH,
        "RAW_PATH": RAW_PATH,
        "PRECLOSE_PATH": PRECLOSE_PATH,
        "OWNER_EVIDENCE_PATH": OWNER_EVIDENCE_PATH,
        "SUPERVISOR_PATH": SUPERVISOR_PATH,
        "COMPLETION_PATH": COMPLETION_PATH,
        "RRD_PATH": RRD_PATH,
        "RBL_PATH": RBL_PATH,
        "RERUN_VALIDATION_PATH": RERUN_VALIDATION_PATH,
        "BOARD_PATH": BOARD_PATH,
        "RERUN_SCREENSHOT_PATH": RERUN_SCREENSHOT_PATH,
        "RERUN_RECEIPT_PATH": RERUN_RECEIPT_PATH,
        "MANUAL_INSPECTION_PATH": MANUAL_INSPECTION_PATH,
        "COLLISION_ASSET_ROOT": COLLISION_ASSET_ROOT,
        "EXPECTED_PREREG_SHA256": EXPECTED_PREREG_SHA256,
    }
    for name, value in bindings.items():
        setattr(d404, name, value)


def run_runtime(approved_tuple_sha256: str) -> int:
    """Repair the render path, probe fail-closed, delegate to frozen D404."""

    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D405 controller must be launched with python -B; runtime "
            "refused before frozen D404 module load or any forward-only "
            "write"
        )
    _repair_observability_import_path()
    _observability_preflight_probe()
    _install_screenshot_logical_size_repair()
    d404 = _load_frozen_d404_controller()
    _configure_d404_paths(d404)
    _install_chain_render_repair(d404)
    return int(d404.run_runtime(approved_tuple_sha256))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved-tuple-sha256",
        required=True,
        help=(
            "Exact SHA-256 of d405_proposed_runtime_hash_tuple.json under "
            "the user's 2026-07-28 sequential next-minimal-rung approval."
        ),
    )
    args = parser.parse_args()
    try:
        return run_runtime(args.approved_tuple_sha256)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
