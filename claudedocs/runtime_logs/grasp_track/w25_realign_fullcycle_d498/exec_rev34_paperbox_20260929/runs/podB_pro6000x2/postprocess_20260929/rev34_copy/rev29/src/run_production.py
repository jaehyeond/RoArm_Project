"""W13 resume **본 실행 동결 러너** — 승인된 argv 만 실행하고 실제 rc/시간/신호를 영수증에 남긴다.

usage: python run_production.py <frozen_revision_dir> [--step <key>]

규약 (계약 §C)
  · 명령은 같은 폴더 `COMMANDS.json` 배열 **그대로**. 문자열 조립·인자 변경 없음.
  · 실행 **직전에** 전 입력/소스/criteria 의 full SHA256 을 재계산해 동결 기록과 대조한다(불일치 → 실행 0).
  · 기존 산출/로그가 있으면 거부한다(덮어쓰기 금지). attempt 폴더는 prospective manifest 하나만 허용.
  · 단계별 전용 stdout/stderr 분리 기록. 합치지 않는다.
  · 물리 단계 벽시계 상한 = `COMMANDS.json.step_caps_s.step1_simulation`(본 실행 32400 s).
    ⚠️ **32400 s 는 종료 처리(유예 flush + SIGKILL)를 포함한 총 한도다**(CONTRACT_ADDENDUM_01 §6).
    32400 + grace = 33600 이 아니다. 그래서 SIGTERM 은 `cap - graceful_grace_s` 에 보내고
    나머지 유예가 끝나면 SIGKILL 해서 **총합이 cap 을 넘지 않게** 한다.
  · 러너 자신이 받은 SIGTERM/SIGINT 도 같은 경로로 자식 **그룹**에 전달한다. orphan 을 남기지 않는다.
  · **타임아웃은 성공이 아니다**(ADDENDUM §6). 자식이 신호를 받고 rc 0 으로 끝나도
    timed_out 이면 **다음 단계로 진행하지 않고** 중단한다.
  · **재시도 0.** rc != 0 이면 산출을 그대로 보존하고 멈춘다.
  · `RUN_STATUS.json` 은 **항상** 쓴다(계획된 guard abort·신호·예외 포함). 단, **기존 파일을 덮지 않는다**
    — 거부 경로에서도 새 이름(`RUN_STATUS_refused_*.json`)으로 남긴다.
  · 시간 예산은 **단조 시계**(`time.monotonic`)로 잰다. 벽시계 보고용 UTC 는 따로 적는다.
  · `--step` 은 중복·미선언 키를 **fail-closed** 로 거절한다.
  · heartbeat/로그 진행은 solver 진행의 증거가 아니다 — 영수증은 rc·시각·신호만 주장한다.

주장하지 않는 것
    rc 0 이나 스크립트 self-PASS 는 배출·과학 성공의 증거가 아니다(criteria.json process.rc0_is_not_delivery).
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


ap = argparse.ArgumentParser()
ap.add_argument("revision_dir")
ap.add_argument("--step", action="append",
                help="이 키만 실행(기본 = COMMANDS.json.step_order 전체). GPU 순차 실행용")
A = ap.parse_args()

rev = Path(A.revision_dir).resolve()
C = json.loads((rev / "COMMANDS.json").read_text())
declared = list(C["step_order"])
order = list(A.step) if A.step else list(declared)
# D-runner-6: 중복·미선언 --step 은 fail-closed (임의 단계 실행 금지)
if len(set(order)) != len(order):
    raise SystemExit(f"--step 에 중복이 있다(거부): {order}")
unknown = [k for k in order if k not in declared]
if unknown:
    raise SystemExit(f"--step 에 동결 step_order 밖의 키가 있다(거부): {unknown} / 허용 {declared}")
if order != [k for k in declared if k in set(order)]:
    raise SystemExit(f"--step 순서가 동결 step_order 순서와 다르다(거부): {order} vs {declared}")
caps = {k: float(v) for k, v in C["step_caps_s"].items()}
grace = float(C["graceful_grace_s"])
att = Path(C["attempt_dir"]).resolve()
receipt = att / "EXECUTION_RECEIPT.json"
hash_receipt = att / "HASH_VERIFICATION_RECEIPT.json"
status_path = att / "RUN_STATUS.json"

rec = {"artifact": "W13R_EXECUTION_RECEIPT", "frozen_revision_dir": str(rev), "attempt_dir": str(att),
       "commands_json_sha256": sha(rev / "COMMANDS.json"),
       "runner_self_sha256": sha(Path(__file__).resolve()),
       "step_order_requested": order, "step_caps_s": caps, "graceful_grace_s": grace,
       "auto_retry": False, "started_utc": utc(), "runner_pid": os.getpid(), "steps": []}
status = {"artifact": "W13R_RUN_STATUS", "state": "starting", "started_utc": utc(),
          "attempt_dir": str(att), "steps_completed": [], "signals_received": [],
          "note": "이 파일은 계획된 guard abort·신호·예외에서도 항상 쓰인다."}


def write_status(state, **kw):
    """RUN_STATUS 를 쓴다. **기존 파일을 절대 덮지 않는다**(거부 경로 포함).

    D-runner-1: 예전 구현은 거부(SystemExit) 경로에서도 status_path 에 그대로 써서
    이미 있던 RUN_STATUS 를 덮을 수 있었다. 이제 대상이 이미 있으면 새 이름으로 남긴다.
    """
    status["state"] = state
    status["updated_utc"] = utc()
    status.update(kw)
    p = status_path
    if p.exists() and not status.get("_owns_status_file"):
        p = status_path.with_name(f"RUN_STATUS_refused_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        status["wrote_to_alternate_path_because_existing"] = str(status_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    status["_owns_status_file"] = (p == status_path)
    status["status_path_written"] = str(p)


def write_receipt():
    receipt.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")


# 바깥 인터프리터 기동 시간은 **단계 예산과 분리해** 보고한다(`msg_93994d60e7e1`).
# import 이전의 파이썬 기동 자체는 이 프로세스 안에서 측정할 수 없다 — 그 사실을 그대로 적는다.
T_RUNNER_IMPORT = time.monotonic()
sig_box = {"hit": None}
# `owned_groups` = **불변 이력**(pgid_history). 한 번 들어온 기록은 지우지 않는다.
# 각 기록이 **자기 자신의 절대 마감**을 들고 있어서 except/finally 처럼 단계 지역변수를
# 볼 수 없는 경로도 **같은 예산**으로 정리할 수 있다(`msg_50cada7a0f94`).
# `active` 는 신호 대상 여부다. **그룹 전체 부재를 실제 조회로 확인한 뒤에만** 내린다
# (`msg_20a4aef77c86`). 리더 exit 만으로 내리던 옛 결함은 재도입하지 않는다.
owned_groups = []
REAP_OBSERVE_S = 0.5               # KILL 뒤 부재 확인에 쓰는 최대 관측 시간(마감 안에서만)
# TERM 유예를 마감까지 꽉 쓰면 **KILL 발행 자체가 마감을 넘는다**(실측: 마감 +0.2 ms ~ 수 ms).
# 그래서 같은 예산의 뒷자리에서 이만큼을 KILL 발행용으로 뗀다. **새 시간은 만들지 않는다** —
# TERM 유예가 그만큼 짧아질 뿐이고, 모든 신호는 자기 마감 안에서 발행된다.
KILL_ISSUE_MARGIN_S = 0.05


def pgid_alive(pgid):
    """그룹에 살아 있는 구성원이 있는가. 리더가 끝나도 손자가 남을 수 있다(감사 재현 결함 ③)."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def register_owned_group(pgid, step, pid, total_deadline, kill_deadline, cap_s, reserve_s):
    """새 소유 그룹을 이력에 넣는다. 같은 pgid 가 다시 나오면 **새 기록**으로 append 한다.

    `msg_46d498bae806` + `msg_93994d60e7e1`: 기록이 **두 마감을 함께** 들고 간다.
      · `total_deadline_monotonic` = 단계 전체(정리 포함) 바깥 경계 = t0 + cap
      · `kill_deadline_monotonic`  = **조기 차단선** = total − reserve
    모든 정리 경로가 **같은 조기 차단선 전에 신호를 발행**하고, 부재 확인은 남은 total 안에서 한다.
    예전 결함: 본 루프만 reserve 를 쓰고 cleanup 은 total 까지 기다렸다 KILL 해서,
    호출 지연이 조금이라도 있으면 **발행이 마감 뒤**가 됐다.
    """
    owned_groups.append({"pgid": int(pgid), "step": step, "pid": int(pid),
                         "total_deadline_monotonic": float(total_deadline),
                         "kill_deadline_monotonic": float(kill_deadline),
                         "hard_deadline_s_from_start": float(cap_s),
                         "cleanup_reserve_s": float(reserve_s),
                         "active": True, "spawned_utc": utc(),
                         "retired_utc": None, "retired_reason": None,
                         "signals": []})
    return owned_groups[-1]


def active_owned_groups():
    """신호를 보낼 수 있는 기록만. 이력에서 빼는 것이 아니라 **active 플래그**로 가린다."""
    return [g for g in owned_groups if g["active"]]


def active_owned_pgids():
    return [g["pgid"] for g in active_owned_groups()]


def all_owned_pgids_history():
    """이력 전체(은퇴분 포함). 영수증에 보존한다."""
    return [g["pgid"] for g in owned_groups]


def _retire(g, reason):
    """**그룹 전체 부재 확인** 뒤에만 호출한다. 리더 exit 만으로는 절대 호출하지 않는다."""
    g["active"] = False
    g["retired_reason"] = reason
    g["retired_utc"] = utc()


def _wait_absent(pgid, d_end, poll):
    """`d_end`(절대 monotonic) 를 **넘기지 않고** 부재를 기다린다. sleep 도 남은 예산으로 자른다."""
    while pgid_alive(pgid):
        rem = float(d_end) - time.monotonic()
        if rem <= 0.0:
            return False
        time.sleep(min(float(poll), rem))
    return True


def _bounded_leader_wait(proc, entry, total_deadline, why, poll=0.02):
    """리더 회수를 **남은 total 예산 안에서만** 기다린다. 확인 못 하면 unverified 로 보고한다.

    `msg_93994d60e7e1`: 예전 판은 KILL 뒤와 루프 탈출 fallback 에서 `p.wait()` 를 **무경계**로 불렀다.
    SIGKILL 이면 보통 즉시 회수되지만 "보통"은 계약이 아니다. 여기서 기다리다 마감을 넘기면
    단계 전체가 cap 을 넘는다. 확인 못 한 사실을 **영수증에 적고** 돌려준다(None).
    반환 None = 리더 rc 미확인. 호출자는 그 상태로 timeout 분류를 유지한다.
    """
    rc = proc.poll()
    while rc is None:
        rem = float(total_deadline) - time.monotonic()
        if rem <= 0.0:
            entry.setdefault("leader_reap_unverified", []).append(
                {"why": why, "utc": utc(),
                 "note": ("리더 종료를 남은 예산 안에 확인하지 못했다. 무한 대기하지 않고 "
                          "unverified 로 남긴다 — 보통 즉시 회수되지만 그것은 계약이 아니다.")})
            return None
        try:
            rc = proc.wait(timeout=min(poll, rem))
        except subprocess.TimeoutExpired:
            rc = proc.poll()
    return rc


def cleanup_owned_groups(why, deadline=None, poll=0.05):
    """어떤 종료 경로에서도 **내가 띄운 active 그룹만**, **같은 예산 안에서** 정리한다.

    감사 재현 결함 ③: 리더 exit 0 뒤에도 같은 PGID 손자가 남을 수 있으므로 기록을 계속 들고
    있다가 **실제 조회로** 확인한 뒤 정리하고, 재조회 후에만 "남은 것 없음"을 주장한다.

    감사 결합 반례(`msg_76ac30d8b716`): 리더가 TERM 에 0 으로 죽고 같은 PGID 후손이 TERM 을
    무시하면, 예전 구현은 여기서 **새 5 초 grace** 를 시작해 cap 1.2 s 실행이 5.917 s 가 됐다.

    `msg_50cada7a0f94` 가 지적한 **남은 구멍 3개**를 여기서 닫는다:
      (a) `fallback_grace_s` 자체 — 이제 없다. 마감은 인자 `deadline` 과 **각 기록이 들고 있는
          자기 마감** 중 **더 이른 쪽**이다. except/finally 도 그래서 새 예산을 못 만든다.
      (b) KILL 뒤 회수 대기가 `max(deadline, now) + 0.5` 라 마감을 넘었다 → 이제
          `min(now + REAP_OBSERVE_S, d_eff)` 로 **마감 안에서만** 관측한다.
      (c) `sleep(poll)` 가 마감을 최대 poll 만큼 넘겼다 → `_wait_absent` 가 남은 예산으로 자른다.

    마감이 이미 지났으면 TERM 대기를 **건너뛰고 곧바로 KILL** 한다(예산을 새로 만들지 않는다).
    신호 발행 시각과 부재 확인 시각을 **따로** 남긴다 — OS 신호 발행은 마감 안이어도
    회수 관측은 늦을 수 있고, 그것을 확인하지 못하면 `unverified` 로 정직하게 남긴다.
    """
    t_call = time.monotonic()
    acted, still, retired, notes = [], [], [], []
    for g in list(active_owned_groups()):
        pgid = g["pgid"]
        # 신호는 **조기 차단선**까지, 부재 확인은 **total** 까지. 둘 다 기록이 들고 온 값이다.
        d_kill = g["kill_deadline_monotonic"]
        d_total = g["total_deadline_monotonic"]
        if deadline is not None:                         # 바깥에서 더 이른 마감을 주면 그쪽이 이긴다
            d_total = min(d_total, float(deadline))
            d_kill = min(d_kill, d_total - g["cleanup_reserve_s"])
        d_eff = d_total
        if not pgid_alive(pgid):
            _retire(g, f"absent_before_signal:{why}")    # 전체 부재 확인 → active 에서 제외
            retired.append(pgid)
            continue
        # TERM 유예는 **조기 차단선**까지만 쓴다 → KILL 발행도 그 선 안에서 끝난다.
        d_term = d_kill - KILL_ISSUE_MARGIN_S
        ev = {"why": why, "kill_deadline_monotonic": d_kill, "total_deadline_monotonic": d_total,
              "term_wait_until": d_term,
              "term_issued_s": None, "kill_issued_s": None, "absence_confirmed": False,
              "absence_confirmed_s": None, "past_deadline_on_entry": t_call >= d_kill}
        if time.monotonic() < d_term:
            try:
                os.killpg(pgid, signal.SIGTERM)
                ev["term_issued_s"] = time.monotonic()
            except ProcessLookupError:
                pass
            except PermissionError:
                notes.append(f"{pgid}: TERM permission denied — 부재 확인 불가")
            gone = _wait_absent(pgid, d_term, poll)
        else:
            notes.append(f"{pgid}: TERM 유예 예산 없음(조기 차단선 {d_kill:.6f}) — 즉시 KILL")
            gone = not pgid_alive(pgid)
        if not gone:
            try:
                os.killpg(pgid, signal.SIGKILL)
                ev["kill_issued_s"] = time.monotonic()
            except ProcessLookupError:
                pass
            except PermissionError:
                notes.append(f"{pgid}: KILL permission denied — 부재 확인 불가")
            # 부재 확인은 **남은 total** 안에서만. 예산이 없으면 관측 없이 unverified 로 남긴다.
            gone = _wait_absent(pgid, min(time.monotonic() + REAP_OBSERVE_S, d_total), poll)
        ev["absence_confirmed"] = bool(gone)
        if gone:
            ev["absence_confirmed_s"] = time.monotonic()
        g["signals"].append(ev)
        acted.append(pgid)
        if gone:
            _retire(g, f"absence_confirmed:{why}")       # **전체 부재 확인 뒤에만** 제외
            retired.append(pgid)
        else:
            still.append(pgid)
            notes.append(f"{pgid}: 부재 미확인 — unverified 로 남긴다(active 유지)")
    t_done = time.monotonic()
    status.setdefault("owned_group_cleanup", []).append(
        {"why": why, "pgids_acted": acted, "pgids_still_alive_after_cleanup": still,
         "pgids_retired_from_active": retired,
         "active_owned_pgids_after": active_owned_pgids(),
         "all_owned_pgids_history": all_owned_pgids_history(),
         "used_absolute_deadline": True,           # 언제나 마감이 있다(기록 자신의 마감)
         "explicit_deadline_passed": deadline is not None,
         "elapsed_in_cleanup_s": t_done - t_call,
         "signal_events": [dict(e, pgid=g["pgid"]) for g in owned_groups for e in g["signals"]
                           if e["why"] == why],
         "notes": notes, "utc": utc()})
    return acted, still


def owned_groups_still_alive():
    """정리 뒤 **실제 조회**로 확인한다. 빈 리스트를 근거 없이 주장하지 않는다.

    은퇴한 기록은 조회하지 않는다 — 번호 재사용된 남의 그룹을 "내 잔존물"로 오인하지 않기 위해서다.
    """
    return [p for p in active_owned_pgids() if pgid_alive(p)]


def _runner_signal(signum, _frame):
    sig_box["hit"] = int(signum)
    status["signals_received"].append({"signum": int(signum), "utc": utc()})
    print(f"\n[runner] 신호 {signum} 수신 — 현재 자식에게 전달하고 우아하게 종료한다", flush=True)


try:
    # ── 1. 기존 산출 거부 ────────────────────────────────────────────────────
    if receipt.exists():
        raise SystemExit(f"이미 실행된 attempt 다(덮어쓰지 않는다): {receipt}")
    if hash_receipt.exists():
        raise SystemExit(f"이미 해시 검증이 돌았다(덮어쓰지 않는다): {hash_receipt}")
    if status_path.exists():
        raise SystemExit(f"이미 run status 가 있다(덮어쓰지 않는다): {status_path}")
    att.mkdir(parents=True, exist_ok=True)
    # 예상 매니페스트는 **attempt 폴더 밖**(구현 루트)에 둔다 → attempt 폴더는 비어 있어야 한다.
    allowed = set(C.get("preexisting_attempt_entries_allowed") or [])
    stray = sorted(p.name for p in att.iterdir() if p.name not in allowed)

    # ── 2. 실행 직전 full SHA256 검증 ────────────────────────────────────────
    pin = json.loads((rev / "REVISION_PIN.json").read_text())
    manifest = json.loads(Path(C["prospective_manifest"]).read_text())
    checks, bad = [], []

    def chk(scope, path, want):
        p = Path(path)
        got = sha(p) if p.exists() else None
        ok = (got == want) and want is not None
        checks.append({"scope": scope, "path": str(p), "expected": want, "actual": got, "match": ok})
        if not ok:
            bad.append(str(p))

    for rel, want in sorted(pin["frozen_copies_sha256"].items()):
        chk("revision_frozen_copy", rev / rel, want)
    for path_, want in sorted(manifest["external_frozen_inputs_sha256"].items()):
        chk("external_frozen_input", path_, want)
    for path_, want in sorted(manifest.get("criteria_sha256", {}).items()):
        chk("criteria", path_, want)
    self_path = Path(__file__).resolve()
    self_rel = str(self_path.relative_to(rev)) if str(self_path).startswith(str(rev)) else None
    chk("runner_self", self_path, pin["frozen_copies_sha256"].get(self_rel) if self_rel else None)

    hv = {"artifact": "W13R_HASH_VERIFICATION_RECEIPT", "verified_utc": utc(),
          "revision_dir": str(rev), "attempt_dir": str(att), "n_checked": len(checks),
          "mismatches": bad, "pre_existing_attempt_entries_not_allowed": stray,
          "allowed_preexisting": sorted(allowed), "checks": checks,
          "rule": "물리 실행 전에 한 번만 기록한다. 불일치나 허용 외 기존 항목이 있으면 실행하지 않는다."}
    hash_receipt.write_text(json.dumps(hv, ensure_ascii=False, indent=2) + "\n")
    write_status("hash_verified", n_hash_checks=len(checks), hash_mismatches=bad, stray_entries=stray)
    if bad or stray:
        print(f"ABORT: 해시 불일치 {len(bad)} · 허용 외 기존 항목 {stray} → {hash_receipt}", flush=True)
        write_status("aborted_hash_or_stray", exit_code=3)
        write_receipt()
        sys.exit(3)
    print(f"hash verification OK: {len(checks)} files → {hash_receipt.name}", flush=True)

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (C.get("env") or {}).items()})
    for s_ in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s_, _runner_signal)
    write_receipt()

    # ── 3. 단계 실행 ────────────────────────────────────────────────────────
    for key in order:
        argv = C[key]
        cap = caps[key]
        so, se = att / f"{key}.stdout.txt", att / f"{key}.stderr.txt"
        for f in (so, se):
            if f.exists():
                raise SystemExit(f"기존 로그가 있다(덮어쓰지 않는다): {f}")
        entry = {"step": key, "argv": argv, "stdout": str(so), "stderr": str(se),
                 "wall_cap_s": cap, "started_utc": utc(), "graceful": None}
        rec["steps"].append(entry)
        write_receipt()
        write_status(f"running:{key}", current_step=key, current_step_started_utc=entry["started_utc"])
        # D-runner-3: 예산은 **단조 시계**로 잰다(벽시계 점프에 영향받지 않게).
        t0 = time.monotonic()
        # D-runner-2: cap 은 종료 처리를 **포함한 총 한도**다. SIGTERM 은 cap-grace 에 보낸다.
        term_at = max(0.0, cap - grace)
        # `msg_50cada7a0f94`: 후처리 정리에 시간이 필요하면 **기존 cap 안에 사전예약**한다
        # (새 cap/grace 연장 금지). grace 안에서 뒷자리를 떼어 루프 KILL 을 그만큼 앞당긴다.
        cleanup_reserve = min(REAP_OBSERVE_S, grace / 2.0)
        kill_at = max(term_at, cap - cleanup_reserve)
        entry["stage_t0_monotonic"] = t0
        entry["stage_t0_utc"] = utc()
        entry["runner_pre_stage_s"] = round(t0 - T_RUNNER_IMPORT, 6)
        entry["runner_pre_stage_note"] = ("러너 import 이후부터 이 단계 시작까지. 파이썬 인터프리터 "
                                          "자체의 기동 시간은 이 프로세스 안에서 측정할 수 없다 — "
                                          "바깥 런처가 따로 잰다. 단계 예산에 포함되지 않는다.")
        entry["term_deadline_s_from_start"] = term_at
        entry["loop_kill_deadline_s_from_start"] = kill_at
        entry["cleanup_reserved_s_inside_cap"] = cleanup_reserve
        entry["hard_deadline_s_from_start"] = cap
        with open(so, "w") as fo, open(se, "w") as fe:
            # 새 프로세스 그룹 → 그룹 전체에 신호를 보낼 수 있고 orphan 을 남기지 않는다
            p = subprocess.Popen(argv, stdout=fo, stderr=fe, env=env, start_new_session=True)
            entry["child_pid"] = p.pid
            entry["child_pgid"] = os.getpgid(p.pid)
            # 기록이 **자기 마감**을 들고 간다 → except/finally 도 같은 예산을 쓴다.
            register_owned_group(entry["child_pgid"], key, p.pid,
                                 total_deadline=t0 + cap, kill_deadline=t0 + kill_at,
                                 cap_s=cap, reserve_s=cleanup_reserve)
            write_receipt()
            print(f"[runner] {key} pid={p.pid} pgid={entry['child_pgid']} cap={cap}s "
                  f"(TERM at {term_at}s, loop KILL at {kill_at}s, hard {cap}s "
                  f"— cap INCLUDES shutdown + {cleanup_reserve}s reserved cleanup)", flush=True)
            rc, timed_out, killed = None, False, False
            sent_term_at = None
            # 감사 재현 결함 ①: 고정 1 s 폴링이 cap 을 넘겼다(1.2 s cap 에 2.013 s).
            # → **절대 monotonic 마감**과 남은 예산만큼만 기다리는 폴링으로 바꾼다.
            term_deadline = t0 + term_at
            loop_kill_deadline = t0 + kill_at
            hard_deadline = t0 + cap
            POLL_MAX = 0.05
            while True:
                rc = p.poll()
                if rc is not None:
                    break
                now = time.monotonic()
                need_term = (now >= term_deadline) or (sig_box["hit"] is not None)
                if need_term and sent_term_at is None:
                    timed_out = now >= term_deadline
                    reason = "wall_cap_minus_grace" if timed_out else f"runner_signal_{sig_box['hit']}"
                    print(f"[runner] {key}: {reason} at {now - t0:.3f}s → 그룹 SIGTERM, "
                          f"남은 유예 안에서 부분 원시/RRD flush 대기", flush=True)
                    entry["graceful"] = {"reason": reason, "term_sent_utc": utc(),
                                         "elapsed_at_term_s": round(now - t0, 3)}
                    write_receipt()
                    write_status(f"graceful_stop:{key}", graceful_reason=reason)
                    try:
                        os.killpg(entry["child_pgid"], signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                    sent_term_at = now
                if now >= loop_kill_deadline:
                    print(f"[runner] {key}: 루프 한도 {kill_at}s 도달 → 그룹 SIGKILL(경계 하드 정지, "
                          f"cap {cap}s 안에 정리 {cleanup_reserve}s 예약분 남김)", flush=True)
                    timed_out = True          # 감사 재현 결함 ②: KILL 도 timeout 분류가 먼저다
                    try:
                        os.killpg(entry["child_pgid"], signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    killed = True
                    # 감사 요구: **두 신호 시각을 모두** 기록한다(TERM 은 위에서, KILL 은 여기서).
                    entry["kill_sent_utc"] = utc()
                    entry["kill_elapsed_s_from_start"] = round(now - t0, 6)
                    # `msg_93994d60e7e1`: KILL 뒤 `p.wait()` 가 **무경계**였다 → 남은 total 안에서만
                    # 리더 회수를 기다리고, 확인 못 하면 **unverified 로 보고**하고 멈춘다(무한 대기 금지).
                    rc = _bounded_leader_wait(p, entry, hard_deadline, "after_kill")
                    break
                # 다음 마감까지 남은 만큼만 잔다(절대 마감을 넘기지 않는다)
                nxt = term_deadline if sent_term_at is None else loop_kill_deadline
                time.sleep(max(0.0, min(POLL_MAX, nxt - time.monotonic())))
            if rc is None:
                rc = _bounded_leader_wait(p, entry, hard_deadline, "loop_exit_fallback")
        # 리더가 끝나도 같은 PGID 손자가 남을 수 있다 → **조회로** 확인하고 정리한다.
        # `msg_20a4aef77c86`: 정상 종료 경로에서도 **반드시** 여기를 지나야 한다. 그래야 부재를
        # 확인한 그룹이 active 에서 은퇴하고, 뒤에 번호가 재사용돼도 신호 대상이 되지 않는다.
        # 리더 rc 만으로 은퇴시키는 옛 결함은 재도입하지 않는다 — 은퇴는 그룹 조회 결과로만 한다.
        entry["descendants_alive_after_leader_exit"] = list(owned_groups_still_alive())
        cleanup_owned_groups(f"after_step:{key}", deadline=hard_deadline)
        entry["descendants_alive_after_cleanup"] = owned_groups_still_alive()
        entry["cleanup_bounded_by_hard_deadline_s_from_start"] = cap
        entry["active_owned_pgids_after_step"] = active_owned_pgids()
        # 단계 끝 시각과 **전체 부재 확인** 결과를 같이 남긴다(신호 발행만으로 끝났다고 하지 않는다).
        entry["stage_end_monotonic"] = time.monotonic()
        entry["stage_total_s_including_cleanup"] = round(time.monotonic() - t0, 6)
        entry["stage_within_cap_including_cleanup"] = bool(
            entry["stage_total_s_including_cleanup"] <= cap)
        entry["whole_group_absence_confirmed"] = bool(not owned_groups_still_alive())
        entry["signal_events_this_step"] = [dict(e, pgid=g["pgid"]) for g in owned_groups
                                            if g["step"] == key for e in g["signals"]]
        entry["contract_note"] = ("계약이 묶는 것은 **정리까지 포함한 단계 전체**다. 신호 발행만으로 "
                                  "재정의하지 않는다. 일반 OS 에서 하드 실시간은 보장되지 않으며, "
                                  "실제 측정값과 미확인 항목을 그대로 보고한다.")
        # 리더 rc 를 남은 예산 안에 확인하지 못하면 `rc is None` 이다. **0 으로 지어내지 않는다** —
        # timeout 분류를 유지하고 미확인 사실을 그대로 기록한다(`msg_93994d60e7e1`).
        rc_unverified = rc is None
        if rc_unverified:
            timed_out = True
            entry["returncode_unverified"] = True
        entry.update(returncode=(None if rc_unverified else int(rc)),
                     returncode_verified=(not rc_unverified),
                     timed_out=timed_out, killed_after_grace=killed,
                     gracefully_stopped=bool(sent_term_at is not None and not killed),
                     wall_s=round(time.monotonic() - t0, 3), ended_utc=utc(),
                     stdout_sha256=sha(so) if so.exists() else None,
                     stderr_sha256=sha(se) if se.exists() else None,
                     stdout_bytes=so.stat().st_size if so.exists() else None,
                     stderr_bytes=se.stat().st_size if se.exists() else None,
                     # 감사 재현 결함 ②: **timeout 분류가 rc 보다 먼저**다. SIGKILL 로 죽어 rc 가 -9/247 이
                     # 되더라도 원인은 timeout 이며 성공이 아니다.
                     outcome=("timeout" if (timed_out or killed or rc_unverified)
                              else ("ok" if int(rc) == 0 else "nonzero_rc")),
                     exit_class_exit_code=(124 if (timed_out or killed or rc_unverified)
                                           else (0 if int(rc) == 0 else int(rc))),
                     success=bool((not rc_unverified) and int(rc) == 0
                                  and not timed_out and not killed))
        write_receipt()
        status["steps_completed"].append({"step": key, "rc": (None if rc_unverified else int(rc)),
                                          "rc_verified": not rc_unverified, "wall_s": entry["wall_s"],
                                          "timed_out": timed_out, "killed_after_grace": killed})
        write_status(f"finished:{key}")
        print(f"{key}: rc={rc} timed_out={timed_out} killed={killed} wall={entry['wall_s']}s "
              f"-> {so.name}/{se.name}", flush=True)
        # 감사 재현 결함 ②: **timeout 분류를 rc 보다 먼저** 본다. SIGKILL 로 rc 가 0 이 아니게 돼도
        # 원인은 timeout 이며 halted_timeout_not_success 로 분류해야 한다.
        if timed_out or killed:
            rec["halted_after"] = key
            rec["note"] = ("시간 상한에서 중단됐다. 자식 rc 가 0 이어도 성공으로 취급하지 않으며 "
                           "다음 단계를 자동으로 시작하지 않는다.")
            write_receipt()
            cleanup_owned_groups(f"halt_timeout:{key}", deadline=hard_deadline)
            write_status("halted_timeout_not_success", halted_after=key,
                         child_returncode=(None if rc_unverified else int(rc)),
                         child_returncode_verified=not rc_unverified, timed_out=timed_out,
                         killed_after_grace=killed, exit_code=124,
                         all_owned_pgids=all_owned_pgids_history(),
                         active_owned_pgids=active_owned_pgids(),
                         owned_groups_left_alive=owned_groups_still_alive())
            print("HALTED on timeout (child rc is not success). 산출 보존됨.", flush=True)
            sys.exit(124)
        # rc 미확인은 위 timeout 분기에서 이미 124 로 멈췄으므로 여기서 rc 는 항상 정수다.
        if rc != 0:
            rec["halted_after"] = key
            rec["note"] = "rc != 0 — 재시도하지 않는다. 산출을 그대로 보존하고 보고한다."
            write_receipt()
            cleanup_owned_groups(f"halt_rc:{key}", deadline=hard_deadline)
            write_status("halted_nonzero_rc", halted_after=key, exit_code=int(rc),
                         all_owned_pgids=all_owned_pgids_history(),
                         active_owned_pgids=active_owned_pgids(),
                         owned_groups_left_alive=owned_groups_still_alive())
            print("HALTED (no retry). 산출 보존됨.", flush=True)
            sys.exit(int(rc))
        if sig_box["hit"] is not None:
            rec["halted_after"] = key
            rec["note"] = f"러너가 신호 {sig_box['hit']} 를 받아 남은 단계를 시작하지 않았다."
            write_receipt()
            write_status("halted_on_signal", halted_after=key, signal=sig_box["hit"], exit_code=130)
            sys.exit(130)
    rec["all_steps_rc0"] = True
    rec["all_steps_success"] = all(s.get("success") for s in rec["steps"])
    rec["process_accounting"] = {
        "runner_processes": 1, "child_processes_spawned": len(rec["steps"]),
        "concurrent_children_max": 1,
        "all_owned_pgids_history": all_owned_pgids_history(),
        "active_owned_pgids": active_owned_pgids(),
        "owned_groups_left_alive": owned_groups_still_alive(),   # 조회 결과이지 가정이 아니다
        "rule": "단계당 자식 1개, 동시 실행 0. GPU 는 순차. 남은 소유 그룹이 있으면 여기 보인다."}
    write_receipt()
    write_status("complete", exit_code=0, all_steps_success=rec["all_steps_success"],
                 note=("finalization/coverage 와 과학적 배출 판정은 별개다. 이 상태는 프로세스 완료만 말한다."))
    print("W13R_PRODUCTION_RUNNER_DONE", flush=True)
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else 1
    # 이미 확정 상태를 쓴 경로는 **덮지 않는다**(terminal state 를 가리면 판정이 흐려진다).
    TERMINAL = {"halted_nonzero_rc", "halted_on_signal", "halted_timeout_not_success",
                "aborted_hash_or_stray", "complete"}
    if status.get("state") not in TERMINAL:
        write_status("systemexit", exit_code=code, detail=str(exc))
    raise
except BaseException as exc:                                      # noqa: BLE001
    rec["runner_exception"] = f"{type(exc).__name__}: {exc}"
    try:
        write_receipt()
    except Exception:                                             # noqa: BLE001
        pass
    try:
        cleanup_owned_groups(f"runner_exception:{type(exc).__name__}")   # D-runner-5
    except Exception:                                             # noqa: BLE001
        pass
    try:
        write_status("runner_exception", exit_code=1, detail=rec["runner_exception"])
    except Exception:                                             # noqa: BLE001
        pass
    raise
finally:
    # D-runner-5: 어떤 경로로 끝나도 내 소유 그룹을 남기지 않는다.
    try:
        _acted, left = cleanup_owned_groups("finally")
    except Exception:                                             # noqa: BLE001
        left = ["cleanup_failed"]
    # 어떤 경로로 끝나도 상태 파일은 남는다(계약 §C "final JSON/run status even on planned guard abort")
    try:
        if not status.get("status_path_written"):
            write_status("unknown_terminated", owned_groups_cleaned_in_finally=left)
    except Exception:                                             # noqa: BLE001
        pass
