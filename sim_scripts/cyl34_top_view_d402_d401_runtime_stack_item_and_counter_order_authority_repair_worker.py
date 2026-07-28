#!/usr/bin/env python3
"""D402 thin Worker wrapper for Kit Item-compatible version access.

The frozen D401 wrapper and D400 Worker remain byte-for-byte unchanged.  This
module overrides only the D400 runtime stack probe so that the active
``omni.physx`` extension's ``package.version`` is read through the nested
indexing protocol supported by ``carb.dictionary.Item`` and built-in ``dict``.

Importing this module does not import or launch Isaac, Kit, PhysX, Warp, CUDA,
Rerun, or the actual Worker.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WORKER_PATH = Path(__file__).resolve()
CONTROLLER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d402_d401_runtime_stack_item_and_counter_order_authority_repair_controller.py"
)
D401_WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d401_d400_runtime_freeze_snapshot_order_repair_worker.py"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d402/"
    "attempt1_d401_runtime_stack_item_and_counter_order_authority_repair"
)
PREREG_PATH = OUT_DIR / "d402_preregistration.json"
ATTESTATION_PATH = OUT_DIR / "d402_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d402_proposed_runtime_hash_tuple.json"
RUNTIME_MANIFEST_PATH = OUT_DIR / "d400_runtime_freeze_manifest.json"
INVOCATION_PATH = OUT_DIR / "d400_worker_invocation.json"

EXPECTED_PREREG_SHA256 = (
    "9868b1f60035682295610ce9e38e23d8fa1df37804a69386b00aaf3cf1fdfc4e"
)
EXPECTED_D401_WORKER_SHA256 = (
    "fc019d0d74bc868a5f2cac928824f5de875e05783472f288873f01342775673d"
)
EXPECTED_OMNI_PHYSX_EXTENSION_VERSION = "107.3.26"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extension_package_version(extension_config: Any) -> str | None:
    """Read a strict string version through the Item/dict indexing protocol."""

    try:
        package = extension_config["package"]
        version = package["version"]
    except (KeyError, TypeError, IndexError):
        return None
    return version if type(version) is str else None


def _make_item_compatible_runtime_stack_probe(base: ModuleType):
    frozen_probe = base._runtime_stack_probe

    def repaired_runtime_stack_probe() -> dict[str, Any]:
        result = frozen_probe()
        version = None
        extension_config_type = None
        if result["checks"]["omni_physx_extension_id_resolved"] is True:
            import omni.kit.app

            manager = omni.kit.app.get_app().get_extension_manager()
            extension_id = result["supported_runtime_probe"]["extension_id"]
            extension_config = manager.get_extension_dict(extension_id)
            extension_config_type = (
                f"{type(extension_config).__module__}."
                f"{type(extension_config).__qualname__}"
            )
            version = _extension_package_version(extension_config)

        supported = result["supported_runtime_probe"]
        supported["method"] = (
            "frozen D400 dict-only probe first, followed by one "
            "Item-compatible ExtensionManager.get_extension_dict(active "
            "omni.physx id)['package']['version'] nested-indexing re-read "
            "when the frozen probe resolved that id"
        )
        supported["frozen_probe_executed_first"] = True
        supported["item_compatible_requery_used_frozen_extension_id"] = (
            result["checks"]["omni_physx_extension_id_resolved"] is True
        )
        supported["item_compatible_requery_count"] = (
            1
            if result["checks"]["omni_physx_extension_id_resolved"] is True
            else 0
        )
        supported["total_extension_config_reads"] = (
            2
            if result["checks"]["omni_physx_extension_id_resolved"] is True
            else 0
        )
        supported["extension_config_runtime_type"] = extension_config_type
        supported["version_value_exact_builtin_str"] = type(version) is str
        supported["omni_physx_extension_version"] = version
        result["checks"]["omni_physx_extension_version_exact"] = bool(
            type(version) is str
            and version == EXPECTED_OMNI_PHYSX_EXTENSION_VERSION
        )
        result["pass"] = all(result["checks"].values())
        return result

    return repaired_runtime_stack_probe


def _load_frozen_d401_worker() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D402 worker requires python -B before loading the frozen D401 "
            "worker"
        )
    observed = _sha(D401_WORKER_PATH)
    if observed != EXPECTED_D401_WORKER_SHA256:
        raise RuntimeError(
            "frozen D401 worker hash drift: "
            f"{observed} != {EXPECTED_D401_WORKER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d402_frozen_d401_worker",
        D401_WORKER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D401 worker import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_d401_paths(d401: ModuleType) -> None:
    bindings = {
        "WORKER_PATH": WORKER_PATH,
        "CONTROLLER_PATH": CONTROLLER_PATH,
        "OUT_DIR": OUT_DIR,
        "PREREG_PATH": PREREG_PATH,
        "ATTESTATION_PATH": ATTESTATION_PATH,
        "TUPLE_PATH": TUPLE_PATH,
        "RUNTIME_MANIFEST_PATH": RUNTIME_MANIFEST_PATH,
        "INVOCATION_PATH": INVOCATION_PATH,
        "EXPECTED_PREREG_SHA256": EXPECTED_PREREG_SHA256,
    }
    for name, value in bindings.items():
        setattr(d401, name, value)


def _install_item_accessor_repair(d401: ModuleType) -> None:
    frozen_loader = d401._load_frozen_d400_worker

    def load_d400_with_repaired_runtime_probe() -> ModuleType:
        base = frozen_loader()
        base._runtime_stack_probe = _make_item_compatible_runtime_stack_probe(
            base
        )
        return base

    d401._load_frozen_d400_worker = load_d400_with_repaired_runtime_probe


def main() -> int:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D402 worker must be launched with python -B before frozen D401 "
            "module load"
        )
    d401 = _load_frozen_d401_worker()
    _configure_d401_paths(d401)
    _install_item_accessor_repair(d401)
    return int(d401.main())


if __name__ == "__main__":
    raise SystemExit(main())
