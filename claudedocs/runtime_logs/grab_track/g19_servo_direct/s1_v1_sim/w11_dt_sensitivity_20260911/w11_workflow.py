"""W11 isolated setup, execution and evidence checks; never accesses robot devices."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime

OUT = Path(__file__).resolve().parent
REPO = next(p for p in OUT.parents if (p / "sim_deme_scoop_s1.py").is_file())
W10 = OUT.parent / "w10_deme_close_fix"
BASE = W10 / "cell_DE_dt2e6_c"
CELL = OUT / "cell_dt1e6_seed460"
PARAMS = OUT / "params_w11_dt1e6.json"
AUDIT = REPO / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/scoop_tilt_cycle_01/closeout_01/simulation_readiness_audit.json"
ISAAC_PY = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args):
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def prepare():
    assert not PARAMS.exists() and not CELL.exists(), "Use a fresh attempt directory"
    checks = []
    for item in read(AUDIT)["w10_input_checks"]:
        p = Path(item["path"])
        if not p.is_absolute():
            p = REPO / p
        digest = sha(p)
        checks.append({"path": str(p), "sha256": digest, "expected_sha16": item["expected"], "pass": digest[:16] == item["expected"]})
    assert len(checks) == 6 and all(x["pass"] for x in checks), checks
    baseline = read(W10 / "params_w10_DE_dt2e6_c.json")
    new = dict(baseline, timestep_s=1e-6, render_timeline_path=str(CELL / "render_timeline_seed460.npz"))
    assert baseline["timestep_s"] == 2e-6
    diff = {k: [baseline.get(k), new.get(k)] for k in set(baseline) | set(new) if baseline.get(k) != new.get(k)}
    assert set(diff) == {"timestep_s", "render_timeline_path"}
    import torch
    cuda_value = (torch.tensor([2.0], device="cuda") ** 2).item()
    assert cuda_value == 4.0
    versions = {x: metadata.version(x) for x in ("deme", "torch", "numpy", "psutil", "trimesh", "scipy")}
    assert versions["deme"] == "2.4.0"
    isaac_versions = json.loads(command([ISAAC_PY, "-c", 'import importlib.metadata as m,json; print(json.dumps({x:m.version(x) for x in ["numpy","psutil","rerun-sdk"]}))']))
    assert isaac_versions == {"numpy": "1.26.0", "psutil": "5.9.8", "rerun-sdk": "0.34.1"}
    gpu = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.free", "--format=csv,noheader,nounits"])
    assert int(gpu.split(",")[-1].strip()) >= 3072, gpu
    protected = [p for p in W10.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    protected += [REPO / "sim_deme_scoop.py", REPO / "roarm_rl/heightmap.py"]
    protected += [Path(c["path"]) for c in checks]
    manifest = {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(set(protected))}
    prefixes = {}
    for name in ["claudedocs/DECISIONS.md", "claudedocs/EXPERIMENT_LEDGER.md"]:
        p = REPO / name
        prefixes[name] = {"bytes": p.stat().st_size, "sha256": sha(p), "lines": len(p.read_text().splitlines())}
    CELL.mkdir()
    save(PARAMS, new)
    report = {"created_local": datetime.now().astimezone().isoformat(), "inputs": checks, "params_diff": diff,
              "params_sha256": sha(PARAMS), "versions_roarm": versions, "versions_isaaclab": isaac_versions,
              "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda, "cuda_check": cuda_value,
              "gpu": gpu, "git_head": command(["git", "rev-parse", "HEAD"]),
              "git_status": command(["git", "status", "--short"]), "protected_files": manifest, "ledger_prefixes": prefixes,
              "new_output_directory": str(CELL), "all_pass": True}
    save(OUT / "preflight.json", report)
    print("W11 preparation complete", json.dumps({"inputs": len(checks), "protected_files": len(manifest), "gpu": gpu, "params_diff": diff}))


def execute():
    verify("preflight")
    assert not (CELL / "stdout.txt").exists(), "Refusing to overwrite an attempt"
    pile = read(OUT / "preflight.json")["inputs"][0]["path"]
    args = [sys.executable, "-u", str(REPO / "sim_deme_scoop_s1.py"), "--params", str(PARAMS), "--pile", pile, "--out", str(CELL), "--seed", "460"]
    started = time.monotonic()
    start_stamp = datetime.now().astimezone().isoformat()
    with (CELL / "stdout.txt").open("x") as stdout, (CELL / "stderr.txt").open("x") as stderr:
        child = subprocess.Popen(args, cwd=REPO, stdout=stdout, stderr=stderr, start_new_session=True)
        save(OUT / "launch.json", {"command": args, "cwd": str(REPO), "pid": child.pid, "started_local": start_stamp,
                                   "max_wall_s": 14400, "no_progress_s": 300, "automatic_retry": False})
        reason = None
        while child.poll() is None:
            time.sleep(10)
            elapsed = time.monotonic() - started
            files = [CELL / "stdout.txt", CELL / "timeline_seed460.json"]
            last = max(p.stat().st_mtime for p in files if p.exists())
            age = time.time() - last
            if elapsed >= 14400 or age >= 300:
                reason = "wall_timeout" if elapsed >= 14400 else "no_progress"
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
        rc = child.wait()
    terminal = {"started_local": start_stamp, "ended_local": datetime.now().astimezone().isoformat(),
                "returncode": rc, "wall_s": time.monotonic() - started, "guard_stop": reason,
                "result_exists": (CELL / "scoop_s1_seed460.json").exists()}
    save(OUT / "terminal.json", terminal)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)
    return rc


def verify(stage):
    pre = read(OUT / "preflight.json")
    assert pre["all_pass"] and len(pre["inputs"]) == 6
    for item in pre["inputs"]:
        assert sha(item["path"]) == item["sha256"], item["path"]
    assert sha(PARAMS) == pre["params_sha256"]
    old, new = read(W10 / "params_w10_DE_dt2e6_c.json"), read(PARAMS)
    changed = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
    assert changed == {"timestep_s", "render_timeline_path"} and new["timestep_s"] == 1e-6
    assert Path(new["render_timeline_path"]).parent == CELL
    assert pre["versions_roarm"]["deme"] == "2.4.0"
    assert pre["versions_isaaclab"] == {"numpy": "1.26.0", "psutil": "5.9.8", "rerun-sdk": "0.34.1"}
    if stage in {"result", "observability", "closeout"}:
        terminal = read(OUT / "terminal.json")
        assert terminal["result_exists"]
        from analysis_w11 import compare
        actual = compare()
        stored = read(OUT / "comparison.json")
        assert actual == stored, "Stored comparison does not match a fresh raw-data calculation"
        assert actual["evidence_pass"], actual
        assert terminal["returncode"] == 0 and terminal["guard_stop"] is None, terminal
        for png in ["heightmap_comparison.png", "timeline_comparison.png"]:
            assert (OUT / png).is_file() and (OUT / png).stat().st_size > 0
    if stage in {"observability", "closeout"}:
        stem = CELL / "scoop_s1_seed460_w11"
        validation = read(stem.with_name(stem.name + "_rerun_validation.json"))
        assert validation["pass"] and validation["footer_manifest_present"]
        assert validation["version"]["expected_version_match"]
        assert validation["entity_path_contract"]["exact_non_system_match"]
        assert validation["timeline_contract"]["exact_match"]
        assert validation["component_contract"]["pass"]
        assert validation["blueprint_verify"]["ok"] and validation["headless_render"]["ok"]
        for k in ("sink_attached_before_logging", "sink_finalized", "flush_ok"):
            assert validation["log_status_summary"][k]
        assert sha(stem.with_suffix(".rrd")) == validation["sha256"]
        assert sha(stem.with_suffix(".rbl")) == validation["blueprint_verify"]["sha256"]
        assert sha(validation["screenshot_path"]) == validation["headless_render"]["sha256"]
        coverage = read(stem.with_name(stem.name + "_coverage.json"))
        assert coverage["pass"] and all(c["values_exact_at_storage_precision"] for c in coverage["checks"].values())
        assert coverage["source_syncs"] == stored["new"]["syncs"]
        assert coverage["source_particle_frames"] == stored["new"]["particle_frames"]
        controls = read(OUT / "rrd_controls.json")
        assert controls["positive_pass"] and controls["negative_rejected"]
    if stage == "closeout":
        inspection = read(OUT / "inspection.json")
        assert inspection["actually_viewed"] and len(inspection["images"]) >= 4
        for entry in inspection["images"]:
            assert entry["observations"] and sha(entry["path"]) == entry["sha256"]
        for p, expected in pre["protected_files"].items():
            assert Path(p).stat().st_size == expected["bytes"] and sha(p) == expected["sha256"], p
        for name, expected in pre["ledger_prefixes"].items():
            with (REPO / name).open("rb") as f:
                prefix = f.read(expected["bytes"])
            assert hashlib.sha256(prefix).hexdigest() == expected["sha256"], name
        state = (REPO / "START_HERE.md").read_text()
        session = (REPO / "claudedocs/session_20260912_w11_dt_sensitivity.md").read_text()
        ledger = (REPO / "claudedocs/EXPERIMENT_LEDGER.md").read_text()
        recent = (REPO / "claudedocs/LEDGER_RECENT.md").read_text()
        relay = (REPO / "claudedocs/relay/from_codex.md").read_text()
        for text in (state, session, ledger, recent, relay):
            assert "W11" in text
        for text in (state, session, ledger, recent):
            assert str(stored["new"]["capture_count"]) in text
            assert f'{stored["new"]["capture_mass_g"]:.4f}' in text
        assert "session_20260912_w11_dt_sensitivity.md" in state
        assert pre["git_head"] == command(["git", "rev-parse", "HEAD"]), "HEAD changed during this session"
        command(["git", "diff", "--check"])
        current_pins = json.loads(command([ISAAC_PY, "-c", 'import importlib.metadata as m,json; print(json.dumps({x:m.version(x) for x in ["numpy","psutil","rerun-sdk"]}))']))
        assert current_pins == pre["versions_isaaclab"]
    print(f"W11_{stage.upper()}_VERIFIED")


def manifest():
    """Fingerprint final artifacts after the completion ledger has been updated."""
    gates = (OUT / "GATES.md").read_text()
    met = sum(line.startswith("- [x] G") for line in gates.splitlines())
    assert met == 5 and "- [ ]" not in gates
    files = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "manifest.json"]
    files += [REPO / p for p in ("START_HERE.md", "claudedocs/DECISIONS.md", "claudedocs/DECISIONS_ACTIVE.md", "claudedocs/EXPERIMENT_LEDGER.md", "claudedocs/LEDGER_RECENT.md", "claudedocs/relay/from_codex.md", "claudedocs/session_20260912_w11_dt_sensitivity.md")]
    entries = {str(p): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(set(files))}
    save(OUT / "manifest.json", {"artifact": "W11_FINAL_MANIFEST", "created_local": datetime.now().astimezone().isoformat(),
                               "gates_met": met, "gates_unmet": 0, "gates_abandoned": 0, "files": entries,
                               "exclusions": ["manifest.json (self)", "__pycache__"], "git_head": command(["git", "rev-parse", "HEAD"])})
    print("W11_FINAL_MANIFEST_SAVED", len(entries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "verify", "manifest"])
    parser.add_argument("stage", nargs="?", choices=["preflight", "result", "observability", "closeout"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        raise SystemExit(execute())
    elif args.action == "manifest":
        manifest()
    else:
        if not args.stage:
            parser.error("verify requires a stage")
        verify(args.stage)
