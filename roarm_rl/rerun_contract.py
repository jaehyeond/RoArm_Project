"""Post-finalization validation for RoArm Rerun artifacts.

Rerun is an observability layer, not the authoritative numerical gate.  This
module checks that an observability artifact is complete and contains the
registered entities/timelines before a session may call it inspectable.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


RERUN_CONTRACT_VERSION = "0.34.1"


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 for an existing file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, timeout_s: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "ok": completed.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
            "ok": False,
        }


def _normalize_entity_path(path: str) -> str:
    normalized = str(path).strip().strip("/")
    if not normalized:
        raise ValueError("empty Rerun entity path")
    return f"/{normalized}"


def _index_names(stats_text: str) -> set[str]:
    match = re.search(
        r"Num chunks per index\n-+\n(?P<body>.*?)(?:\n\n|\Z)",
        stats_text,
        flags=re.DOTALL,
    )
    if match is None:
        return set()
    names: set[str] = set()
    for line in match.group("body").splitlines():
        if ":" in line:
            names.add(line.split(":", 1)[0].strip())
    return names


def _entity_paths(stats_text: str) -> set[str]:
    match = re.search(
        r"Num chunks per entity\n-+\n(?P<body>.*?)(?:\n\n|\Z)",
        stats_text,
        flags=re.DOTALL,
    )
    if match is None:
        return set()
    paths: set[str] = set()
    for line in match.group("body").splitlines():
        if ":" in line:
            paths.add(line.rsplit(":", 1)[0].strip())
    return paths


def _entity_components(print_text: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    pattern = re.compile(r" - (?P<path>/\S+) - data columns: \[(?P<columns>.*)\]$")
    for line in print_text.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        path = match.group("path")
        components = set(re.findall(r"@([^\s\]]+)", match.group("columns")))
        result.setdefault(path, set()).update(components)
    return result


def _is_system_entity(path: str) -> bool:
    return bool(
        path == "/__properties"
        or path in {"/viewport", "/blueprint_panel", "/selection_panel", "/time_panel"}
        or path.startswith("/container/")
        or path.startswith("/view/")
    )


def validate_rerun_artifact(
    path: str | Path,
    *,
    expected_entity_paths: Iterable[str] = (),
    expected_timeline_names: Iterable[str] = (),
    exact_entity_paths: Iterable[str] | None = None,
    exact_timeline_names: Iterable[str] | None = None,
    expected_entity_components: dict[str, Iterable[str]] | None = None,
    forbidden_entity_fragments: Iterable[str] = ("/\\/",),
    blueprint_path: str | Path | None = None,
    screenshot_path: str | Path | None = None,
    screenshot_window_size: str = "2400x1400",
    screenshot_port: str | None = None,
    cli_path: str | Path | None = None,
    expected_version: str = RERUN_CONTRACT_VERSION,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Validate a finalized RRD and optionally render a headless screenshot.

    The screenshot check proves that the viewer can render the registered
    recording.  It does *not* constitute human/agent visual inspection; that
    separate review must be documented by the session that consumes it.
    """
    artifact = Path(path)
    blueprint = Path(blueprint_path) if blueprint_path is not None else None
    screenshot = Path(screenshot_path) if screenshot_path is not None else None
    cli = str(cli_path) if cli_path is not None else shutil.which("rerun")
    expected_entities = [_normalize_entity_path(item) for item in expected_entity_paths]
    expected_timelines = [str(item) for item in expected_timeline_names]
    exact_entities = (
        {_normalize_entity_path(item) for item in exact_entity_paths}
        if exact_entity_paths is not None
        else None
    )
    exact_timelines = (
        {str(item) for item in exact_timeline_names}
        if exact_timeline_names is not None
        else None
    )
    component_contract = {
        _normalize_entity_path(path): {str(component) for component in components}
        for path, components in dict(expected_entity_components or {}).items()
    }
    forbidden_fragments = [str(item) for item in forbidden_entity_fragments]

    report: dict[str, Any] = {
        "artifact": "ROARM_RERUN_ARTIFACT_VALIDATION_V1",
        "path": str(artifact),
        "expected_rerun_version": str(expected_version),
        "expected_entity_paths": expected_entities,
        "expected_timeline_names": expected_timelines,
        "exact_entity_paths": sorted(exact_entities) if exact_entities is not None else None,
        "exact_timeline_names": sorted(exact_timelines) if exact_timelines is not None else None,
        "expected_entity_components": {
            path: sorted(components) for path, components in component_contract.items()
        },
        "forbidden_entity_fragments": forbidden_fragments,
        "blueprint_path": str(blueprint) if blueprint is not None else None,
        "rendered_blueprint_source": "active blueprint embedded in the RRD",
        "external_rbl_role": (
            "verified fixed-layout export; the 0.34.1 CLI does not guarantee that an external RBL "
            "overrides an already-active embedded blueprint"
        ),
        "screenshot_path": str(screenshot) if screenshot is not None else None,
        "screenshot_window_size": str(screenshot_window_size),
        "screenshot_port": screenshot_port,
        "human_visual_inspection_required": screenshot is not None,
    }
    if not artifact.is_file():
        report.update({"pass": False, "errors": ["RRD artifact does not exist"]})
        return report
    if cli is None:
        report.update({"pass": False, "errors": ["rerun CLI not found on PATH"]})
        return report

    report["bytes"] = artifact.stat().st_size
    report["sha256"] = sha256_file(artifact)
    version = _run([cli, "--version"], timeout_s=timeout_s)
    version_text = f"{version['stdout']}\n{version['stderr']}"
    version_match = bool(re.search(rf"\b{re.escape(str(expected_version))}\b", version_text))
    version["expected_version_match"] = version_match
    report["version"] = version

    verify = _run(
        [cli, "rrd", "verify", "--check-footers", "true", str(artifact)],
        timeout_s=timeout_s,
    )
    report["verify"] = verify
    stats = _run([cli, "rrd", "stats", str(artifact)], timeout_s=timeout_s)
    report["stats"] = stats
    printed = _run([cli, "rrd", "print", "-v", str(artifact)], timeout_s=timeout_s)
    report["print_verbose"] = printed

    stats_text = f"{stats['stdout']}\n{stats['stderr']}"
    print_text = f"{printed['stdout']}\n{printed['stderr']}"
    observed_entities = _entity_paths(stats_text)
    observed_user_entities = {path for path in observed_entities if not _is_system_entity(path)}
    observed_components = _entity_components(print_text)
    entity_checks = {
        entity: f"{entity}:" in stats_text
        for entity in expected_entities
    }
    forbidden_checks = {
        fragment: fragment not in stats_text
        for fragment in forbidden_fragments
    }
    observed_indexes = sorted(_index_names(stats_text))
    timeline_checks = {
        timeline: timeline in observed_indexes
        for timeline in expected_timelines
    }
    footer_manifest_present = (
        "Missing RRD footer" not in f"{verify['stdout']}\n{verify['stderr']}"
        and "(none — no RRD footer was found)" not in stats_text
    )
    report["entity_path_contract"] = {
        "observed_all": sorted(observed_entities),
        "observed_non_system": sorted(observed_user_entities),
        "checks": entity_checks,
        "forbidden_fragment_absent": forbidden_checks,
        "exact_non_system_match": (
            observed_user_entities == exact_entities if exact_entities is not None else None
        ),
        "missing_exact": (
            sorted(exact_entities - observed_user_entities) if exact_entities is not None else []
        ),
        "unexpected_non_system": (
            sorted(observed_user_entities - exact_entities) if exact_entities is not None else []
        ),
    }
    report["entity_path_contract"]["pass"] = bool(
        all(entity_checks.values())
        and all(forbidden_checks.values())
        and (exact_entities is None or observed_user_entities == exact_entities)
    )
    component_checks = {
        path: {
            "required": sorted(required),
            "observed": sorted(observed_components.get(path, set())),
            "pass": required.issubset(observed_components.get(path, set())),
        }
        for path, required in component_contract.items()
    }
    report["component_contract"] = {
        "checks": component_checks,
        "pass": printed["ok"] and all(row["pass"] for row in component_checks.values()),
    }
    report["timeline_contract"] = {
        "observed": observed_indexes,
        "checks": timeline_checks,
        "exact_match": set(observed_indexes) == exact_timelines if exact_timelines is not None else None,
        "pass": all(timeline_checks.values())
        and (exact_timelines is None or set(observed_indexes) == exact_timelines),
    }
    report["footer_manifest_present"] = footer_manifest_present

    blueprint_verify: dict[str, Any] = {"attempted": blueprint is not None}
    if blueprint is not None:
        if blueprint.is_file():
            blueprint_verify = _run(
                [cli, "rrd", "verify", "--check-footers", "true", str(blueprint)],
                timeout_s=timeout_s,
            )
            blueprint_verify.update(
                {
                    "attempted": True,
                    "path": str(blueprint),
                    "bytes": blueprint.stat().st_size,
                    "sha256": sha256_file(blueprint),
                }
            )
        else:
            blueprint_verify = {
                "attempted": True,
                "ok": False,
                "path": str(blueprint),
                "error": "blueprint does not exist",
            }
    report["blueprint_verify"] = blueprint_verify

    render: dict[str, Any] = {"attempted": screenshot is not None}
    if screenshot is not None:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        if screenshot.exists():
            render = {
                "attempted": True,
                "ok": False,
                "error": "refusing to overwrite an existing inspection screenshot",
                "path": str(screenshot),
            }
        else:
            command = [cli, "--headless"]
            if screenshot_port is not None:
                command.extend(["--port", str(screenshot_port)])
            command.extend(
                [
                    "--window-size",
                    str(screenshot_window_size),
                    "--screenshot-to",
                    str(screenshot),
                    str(artifact),
                ]
            )
            render = _run(command, timeout_s=timeout_s)
            render.update(
                {
                    "attempted": True,
                    "path": str(screenshot),
                    "file_nonzero": screenshot.is_file() and screenshot.stat().st_size > 0,
                }
            )
            render["ok"] = bool(render["ok"] and render["file_nonzero"])
            if render["file_nonzero"]:
                render["bytes"] = screenshot.stat().st_size
                render["sha256"] = sha256_file(screenshot)
    report["headless_render"] = render

    errors: list[str] = []
    if not version["ok"] or not version_match:
        errors.append("Rerun CLI version contract failed")
    if not verify["ok"] or not footer_manifest_present:
        errors.append("RRD footer verification failed")
    if not stats["ok"]:
        errors.append("RRD stats failed")
    if not printed["ok"]:
        errors.append("RRD verbose print failed")
    if not report["entity_path_contract"]["pass"]:
        errors.append("RRD entity-path contract failed")
    if not report["timeline_contract"]["pass"]:
        errors.append("RRD timeline contract failed")
    if not report["component_contract"]["pass"]:
        errors.append("RRD component contract failed")
    if blueprint is not None and not blueprint_verify.get("ok", False):
        errors.append("RBL blueprint verification failed")
    if screenshot is not None and not render.get("ok", False):
        errors.append("headless Rerun render failed")
    report["errors"] = errors
    report["pass"] = not errors
    return report
