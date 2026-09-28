"""W19 pod 러너 — COMMANDS_w19.json 의 argv 를 **그대로** 실행하고 rc/시각/신호를 영수증에 남긴다.
W16 run_w16.py 의 계약을 그대로 옮기고 두 가지만 바꿨다: (1) 해시 대조 대상 = BUNDLE_MANIFEST.json 전체,
(2) step 이름은 COMMANDS 의 키. 재시도 0 · 기존 attempt 덮어쓰기 거부 · 자식은 자기 세션(setsid) · 타임아웃≠성공.
사용: setsid nohup python run_w19.py --step <name> > <attempt>/runner.log 2> <attempt>/runner.err &
"""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
loc = lambda: time.strftime("%Y-%m-%d %H:%M:%S %Z")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("--step", required=True)
ap.add_argument("--allow-go-step", action="store_true", help="본 실행(go_required_steps) 은 이 플래그 없이는 거부한다")
A = ap.parse_args()

C = json.loads((HERE / "COMMANDS_w19.json").read_text())
if A.step not in C["steps"]:
    raise SystemExit(f"unknown step {A.step}; steps={list(C['steps'])}")
if A.step in C.get("go_required_steps", []) and not A.allow_go_step:
    raise SystemExit(f"{A.step} 는 사용자 GO 뒤 --allow-go-step 으로만 실행한다")
S = C["steps"][A.step]
att = Path(S["attempt_dir"])
cap, grace = float(S["cap_s"]), float(S["grace_s"])
receipt, hash_receipt, status_path = att / "EXECUTION_RECEIPT.json", att / "HASH_VERIFICATION_RECEIPT.json", att / "RUN_STATUS.json"

rec = {"artifact": "W19_EXECUTION_RECEIPT", "step": A.step, "attempt_dir": str(att),
       "commands_json_sha256": sha(HERE / "COMMANDS_w19.json"), "runner_self_sha256": sha(Path(__file__).resolve()),
       "auto_retry": False, "cap_s": cap, "grace_s": grace, "started_utc": utc(), "started_local": loc(),
       "runner_pid": os.getpid(), "argv": S["argv"], "cwd": S.get("cwd"), "env_overrides": C.get("env") or {},
       "hostname": os.uname().nodename, "runpod_pod_id": os.environ.get("RUNPOD_POD_ID")}
status = {"artifact": "W19_RUN_STATUS", "step": A.step, "state": "starting", "started_utc": utc(),
          "started_local": loc(), "attempt_dir": str(att), "signals_received": []}

def flush_all(state, **kw):
    status["state"] = state; status["updated_utc"] = utc(); status["updated_local"] = loc(); status.update(kw)
    att.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    receipt.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")

sig_box, proc_box = {"hit": None}, {"p": None, "pgid": None}

def _on_signal(sig, _frm):
    sig_box["hit"] = int(sig); status["signals_received"].append({"sig": int(sig), "utc": utc()})
    if proc_box["pgid"]:
        try: os.killpg(proc_box["pgid"], signal.SIGTERM)
        except ProcessLookupError: pass

rc_final = 1
try:
    for p in (receipt, hash_receipt, status_path):
        if p.exists():
            raise SystemExit(f"이미 실행된 attempt 다(덮어쓰지 않는다): {p}")
    att.mkdir(parents=True, exist_ok=True)
    stray = sorted(q.name for q in att.iterdir())
    man = json.loads(Path(C["manifest"]).read_text())
    checks, bad = [], []
    for path_, want in sorted(man["files"].items()):
        q = Path(path_); got = sha(q) if q.exists() else None
        ok = got == want["sha256"]
        checks.append({"path": path_, "expected": want["sha256"], "actual": got, "match": ok})
        if not ok: bad.append(path_)
    hash_receipt.write_text(json.dumps({"artifact": "W19_HASH_VERIFICATION_RECEIPT", "verified_utc": utc(), "step": A.step,
        "manifest": C["manifest"], "n_checked": len(checks), "mismatches": bad, "pre_existing_attempt_entries": stray,
        "checks": checks, "rule": "실행 전 한 번. 불일치나 기존 항목이 있으면 실행하지 않는다."}, ensure_ascii=False, indent=2) + "\n")
    rec["n_hash_checks"] = len(checks)
    flush_all("hash_verified", n_hash_checks=len(checks), hash_mismatches=bad, stray_entries=stray)
    if bad or stray:
        flush_all("aborted_hash_or_stray", exit_code=3)
        print(f"ABORT: 해시 불일치 {len(bad)} · 기존 항목 {stray}", flush=True); sys.exit(3)
    print(f"hash verification OK: {len(checks)} files", flush=True)

    env = dict(os.environ); env.update({str(k): str(v) for k, v in (C.get("env") or {}).items()})
    for s_ in (signal.SIGTERM, signal.SIGINT): signal.signal(s_, _on_signal)
    so, se = att / f"{A.step}.stdout.txt", att / f"{A.step}.stderr.txt"
    t0 = time.monotonic(); rec["stage_t0_utc"] = utc(); rec["stage_t0_local"] = loc(); rec["stdout"], rec["stderr"] = str(so), str(se)
    with open(so, "w") as fo, open(se, "w") as fe:
        p = subprocess.Popen(S["argv"], stdout=fo, stderr=fe, env=env, cwd=S.get("cwd"), start_new_session=True)
    proc_box["p"] = p; proc_box["pgid"] = os.getpgid(p.pid); rec["child_pid"], rec["child_pgid"] = p.pid, proc_box["pgid"]
    flush_all("running", child_pid=p.pid, child_pgid=proc_box["pgid"])
    print(f"[{loc()}] started pid={p.pid} pgid={proc_box['pgid']} cap={cap}s grace={grace}s", flush=True)
    term_at, kill_at = t0 + cap - grace, t0 + cap - 0.05
    timed_out = killed = False; rc = None
    while True:
        rc = p.poll()
        if rc is not None: break
        now = time.monotonic()
        if not timed_out and now >= term_at:
            timed_out = True; rec["term_sent_utc"], rec["term_sent_local"] = utc(), loc(); rec["elapsed_at_term_s"] = round(now - t0, 3)
            try: os.killpg(proc_box["pgid"], signal.SIGTERM)
            except ProcessLookupError: pass
            flush_all("graceful_term_sent", elapsed_at_term_s=rec["elapsed_at_term_s"])
            print(f"[{loc()}] SIGTERM -> pgid {proc_box['pgid']} at {rec['elapsed_at_term_s']}s", flush=True)
        if timed_out and now >= kill_at:
            killed = True; rec["kill_sent_utc"] = utc(); rec["elapsed_at_kill_s"] = round(now - t0, 3)
            try: os.killpg(proc_box["pgid"], signal.SIGKILL)
            except ProcessLookupError: pass
            try: rc = p.wait(timeout=30)
            except subprocess.TimeoutExpired: rc = None
            break
        if int(now - t0) % 600 == 0:
            flush_all("running", elapsed_s=round(now - t0, 1))
        time.sleep(1.0)
    wall = time.monotonic() - t0
    try: alive = os.killpg(proc_box["pgid"], 0) is None
    except ProcessLookupError: alive = False
    except PermissionError: alive = True
    rec.update({"rc": rc, "wall_s": round(wall, 3), "timed_out": timed_out, "killed": killed, "group_alive_after": alive,
                "ended_utc": utc(), "ended_local": loc(), "runner_signal_received": sig_box["hit"]})
    state = ("killed_after_grace" if killed else "halted_timeout_not_success" if timed_out else "completed_rc0" if rc == 0 else "failed_nonzero_rc")
    flush_all(state, rc=rc, wall_s=rec["wall_s"], timed_out=timed_out, killed=killed, group_alive_after=alive, ended_local=loc())
    print(f"[{loc()}] rc={rc} wall={wall:.1f}s timed_out={timed_out} killed={killed} state={state}", flush=True)
    rc_final = 0 if (rc == 0 and not timed_out and not killed) else (124 if timed_out else 1)
except SystemExit as e:
    rc_final = int(e.code or 0); raise
except BaseException as e:  # noqa: BLE001 — 영수증은 항상 남긴다
    rec["exception"] = f"{type(e).__name__}: {e}"; flush_all("runner_exception", exception=rec["exception"]); raise
finally:
    if status.get("state") in (None, "starting"): flush_all("aborted_before_launch")
sys.exit(rc_final)
