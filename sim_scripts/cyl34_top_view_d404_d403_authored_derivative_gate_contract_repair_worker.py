#!/usr/bin/env python3
"""D404 thin Worker wrapper: authored-derivative gate contract repair.

The frozen D403, D402, D401, and D400 Workers remain byte-for-byte unchanged.
This module rebinds D404 case paths and provenance onto the hash-pinned frozen
D403 Worker wrapper, then installs exactly two function-object replacements on
the eventually loaded frozen D400 Worker module (the same pattern as the D402
``_install_item_accessor_repair``):

* ``_sdf_prim_readback`` — the frozen record is recomputed so that (1) the
  ``physics:collisionEnabled`` shape expectation is varying, as declared by the
  installed usdPhysics ``schema.usda:285`` (no ``uniform`` keyword), and (2)
  float-typed registered SDF attributes with an expected bit pattern use
  ``float32_bits_hex`` as the value authority instead of an impossible double
  equality (``Set(0.01)`` reads back ``0.009999999776482582``).  Check key
  names and the record key set are unchanged.
* ``_normalize_allowlisted_semantics`` — the frozen normalization is extended
  so that (3) the ``default`` metadata entry of ``physics:collisionEnabled``
  on the 64 allowlisted gripper A64 part paths is masked with the same
  allowlist marker as the attribute value, and (4) the builtin relationship
  ``physics:simulationOwner`` (installed usdPhysics ``schema.usda:293``) with
  an empty authored target list is filtered on the registered SDF mesh path
  only.  Any other semantic difference still fails closed.

All four expectations are derived from the installed schema declarations and
float32 storage semantics (DECISIONS D403).  Importing this module does not
import or launch Isaac, Kit, PhysX, Warp, CUDA, Rerun, or the actual Worker.
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
    "cyl34_top_view_d404_d403_authored_derivative_gate_contract_repair_controller.py"
)
D403_WORKER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d403_d402_host_boundary_git_repin_rerun_worker.py"
)
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d404/"
    "attempt1_d403_authored_derivative_gate_contract_repair"
)
PREREG_PATH = OUT_DIR / "d404_preregistration.json"
ATTESTATION_PATH = OUT_DIR / "d404_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d404_proposed_runtime_hash_tuple.json"
RUNTIME_MANIFEST_PATH = OUT_DIR / "d400_runtime_freeze_manifest.json"
INVOCATION_PATH = OUT_DIR / "d400_worker_invocation.json"

EXPECTED_PREREG_SHA256 = (
    "4514e824a93902e1b69715df923d43a6c8b86790777b913f3e8c72434b254db0"
)
EXPECTED_D403_WORKER_SHA256 = (
    "f594eb36940d25e48985b1ea5cdb1d8e19796353bd1103d61b9ea156b2277f05"
)

# Installed usdPhysics schema.usda:285 declares
# ``bool physics:collisionEnabled = true`` without the ``uniform`` keyword,
# so a schema-conforming authored attribute reads back as varying.
EXPECTED_COLLISION_ENABLED_UNIFORM = False
ALLOWLIST_COLLISION_ENABLED_MARKER = "$ALLOWLIST_COLLISION_ENABLED_VALUE"
# The three applied APIs contribute exactly one builtin relationship to the
# composed mesh prim definition (usdPhysics schema.usda:293).
SEMANTIC_ALLOWED_MESH_BUILTIN_RELATIONSHIPS = ("physics:simulationOwner",)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repaired_readback_contract(
    record: dict[str, Any], sdf_attribute_specs: dict[str, tuple]
) -> dict[str, Any]:
    """Recompute the two defective checks of a frozen readback record.

    Pure function: raw observation fields produced by the frozen
    ``_sdf_prim_readback`` are left untouched; only the boolean semantics of
    ``collision_enabled_authored_uniform_noncustom_default_only`` and the
    per-attribute value authority inside ``all_seven_attrs_exact`` change.
    """

    checks = record.get("checks")
    if not isinstance(checks, dict):
        return record

    shape = record["collision_enabled_shape"]
    checks["collision_enabled_authored_uniform_noncustom_default_only"] = bool(
        shape["valid"]
        and shape["authored"]
        and shape["custom"] is False
        and shape["uniform"] is EXPECTED_COLLISION_ENABLED_UNIFORM
        and shape["time_samples"] == []
        and shape["connections"] == []
    )

    attrs = record["attributes"]
    attr_checks = record["attribute_checks"]
    for name, (
        expected_type,
        expected_value,
        expected_bits,
    ) in sdf_attribute_specs.items():
        row = attrs.get(name)
        if not isinstance(row, dict):
            attr_checks[name] = False
            continue
        if expected_type == "float" and expected_bits is not None:
            value_authority = row["float32_bits_hex"] == expected_bits
        else:
            value_authority = row["value"] == expected_value
        attr_checks[name] = bool(
            row["valid"]
            and row["authored"]
            and row["custom"] is False
            and row["uniform"] is True
            and row["time_samples"] == []
            and row["connections"] == []
            and row["usd_type"] == expected_type
            and value_authority
            and (
                expected_bits is None
                or row["float32_bits_hex"] == expected_bits
            )
        )
    checks["all_seven_attrs_exact"] = (
        all(attr_checks.values()) and len(attrs) == 7
    )
    record["pass"] = all(checks.values())
    return record


def _extended_allowlist_normalization(
    normalized: list[dict[str, Any]],
    gripper_a64_paths: frozenset[str],
    mesh_path: str,
) -> list[dict[str, Any]]:
    """Extend the frozen allowlist normalization with repairs 3 and 4.

    Pure function over rows already normalized by the frozen
    ``_normalize_allowlisted_semantics``.  Masking replaces only the value of
    an existing ``default`` metadata entry, so a presence difference still
    fails closed; the relationship filter drops only an exactly named builtin
    relationship with an empty authored target list on the mesh path.
    """

    for row in normalized:
        path = row["path"]
        if path in gripper_a64_paths:
            for attr in row["attributes"]:
                if attr["name"] == "physics:collisionEnabled":
                    attr["metadata"] = [
                        [
                            entry_key,
                            ALLOWLIST_COLLISION_ENABLED_MARKER
                            if entry_key == "default"
                            else entry_value,
                        ]
                        for entry_key, entry_value in attr["metadata"]
                    ]
        if path == mesh_path:
            row["relationships"] = [
                relationship
                for relationship in row["relationships"]
                if not (
                    relationship["name"]
                    in SEMANTIC_ALLOWED_MESH_BUILTIN_RELATIONSHIPS
                    and relationship["targets"] == []
                )
            ]
    return normalized


def _install_repaired_gate_functions(base: ModuleType) -> None:
    """Replace exactly two function objects on the frozen D400 Worker module."""

    frozen_readback = base._sdf_prim_readback
    frozen_normalizer = base._normalize_allowlisted_semantics

    def repaired_sdf_prim_readback(
        stage: Any, path: str, *, expected_live: bool
    ) -> dict[str, Any]:
        record = frozen_readback(stage, path, expected_live=expected_live)
        return _repaired_readback_contract(record, base.SDF_ATTRIBUTE_SPECS)

    def repaired_normalize_allowlisted_semantics(
        rows: list[dict[str, Any]], asset_root: Path
    ) -> list[dict[str, Any]]:
        normalized = frozen_normalizer(rows, asset_root)
        return _extended_allowlist_normalization(
            normalized,
            frozenset(base.SOURCE_GRIPPER_A64_PATHS),
            base.SOURCE_MESH_PATH,
        )

    base._sdf_prim_readback = repaired_sdf_prim_readback
    base._normalize_allowlisted_semantics = (
        repaired_normalize_allowlisted_semantics
    )


def _install_gate_contract_repair(d403: ModuleType) -> None:
    """Hook the frozen chain so the loaded D400 module carries both repairs.

    Composition order at runtime: d401._load_frozen_d400_worker is first
    wrapped by the frozen D402 ``_install_item_accessor_repair`` (runtime
    stack probe repair), then wrapped again here so the D404 gate-contract
    repairs are installed on the same loaded module.
    """

    frozen_load_d402 = d403._load_frozen_d402_worker

    def load_d402_with_gate_contract_repair() -> ModuleType:
        d402 = frozen_load_d402()
        frozen_install_item = d402._install_item_accessor_repair

        def install_item_then_gate_contract(d401: ModuleType) -> None:
            frozen_install_item(d401)
            item_repaired_loader = d401._load_frozen_d400_worker

            def load_d400_with_gate_contract_repair() -> ModuleType:
                base = item_repaired_loader()
                _install_repaired_gate_functions(base)
                return base

            d401._load_frozen_d400_worker = load_d400_with_gate_contract_repair

        d402._install_item_accessor_repair = install_item_then_gate_contract
        return d402

    d403._load_frozen_d402_worker = load_d402_with_gate_contract_repair


def _load_frozen_d403_worker() -> ModuleType:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D404 worker requires python -B before loading the frozen D403 "
            "worker"
        )
    observed = _sha(D403_WORKER_PATH)
    if observed != EXPECTED_D403_WORKER_SHA256:
        raise RuntimeError(
            "frozen D403 worker hash drift: "
            f"{observed} != {EXPECTED_D403_WORKER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "_d404_frozen_d403_worker",
        D403_WORKER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create frozen D403 worker import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_d403_paths(d403: ModuleType) -> None:
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
        setattr(d403, name, value)


def main() -> int:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "D404 worker must be launched with python -B before frozen D403 "
            "module load"
        )
    d403 = _load_frozen_d403_worker()
    _configure_d403_paths(d403)
    _install_gate_contract_repair(d403)
    return int(d403.main())


if __name__ == "__main__":
    raise SystemExit(main())
