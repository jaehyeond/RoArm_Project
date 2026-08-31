"""print_job.json을 읽어 검증 후 프린터로 전송한다. 매니페스트에 없는 것은 보내지 않는다.

전송 전 5중 확인 — 하나라도 실패하면 아무것도 보내지 않고 종료한다:
  1) 매니페스트의 all_gates_pass
  2) 3mf 해시가 검증 시점과 동일 (검증 후 파일이 바뀌지 않았나)
  3) 프린터 MQTT 도달
  4) 프린터가 IDLE (진행 중 작업을 덮어쓰지 않는다)
  5) --yes 명시 (docs/print.md: 연속 출력 금지, 매 출력 사용자 시작 신호)

사용:  python send_print_job.py <print_job.json> --yes
"""
import sys, json, hashlib, argparse
from pathlib import Path

sys.path.insert(0, "/home/cgxr/Documents/DK/DTR/bamboo-3dprinter")
import printer as bp

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("--yes", action="store_true", help="실제 전송. 없으면 검증만 (dry-run)")
a = ap.parse_args()

job = json.loads(Path(a.manifest).read_text())
print(f"작업 : {job['name']}  —  {job['purpose'][:60]}...")
print(f"대상 : {job['printer']['model']} @ {job['printer']['ip']}\n")

fail = []

# 1) 게이트
if not job.get("all_gates_pass"):
    fail.append("매니페스트 all_gates_pass=false — 슬라이스 검증을 통과하지 못함")
print(f"  [{'OK ' if job.get('all_gates_pass') else 'X  '}] 1. 슬라이스 게이트 "
      f"{sum(g['pass'] for g in job['gates'].values())}/{len(job['gates'])}")

# 2) 아티팩트 무결성
art = Path(job["artifact"]["path"])
if not art.exists():
    fail.append(f"3mf 없음: {art}")
    ok2 = False
else:
    h = hashlib.sha256(art.read_bytes()).hexdigest()[:16]
    ok2 = (h == job["artifact"]["sha256_16"])
    if not ok2:
        fail.append(f"3mf 해시 불일치 — 검증 후 파일이 바뀜 (기대 {job['artifact']['sha256_16']}, 실제 {h})")
print(f"  [{'OK ' if ok2 else 'X  '}] 2. 아티팩트 무결성  {art.name} ({job['artifact']['bytes']} B)")

# 3~4) 프린터 도달 + IDLE
cfg = bp.load_config(); cfg["printer"]["ip"] = job["printer"]["ip"]
p = bp.P1SPrinter(config=cfg)
state = None
live = {}
try:
    p.mqtt_connect()
    msgs = p.mqtt_pushall(wait=6)
    st = p.mqtt_status(wait=5) or {}
    state = st.get("state")
    # mqtt_status 는 gcode_state 만 준다. 현재 가동 여부는 원시 필드에 있다.
    for m in msgs:
        pr = m.get("print", {})
        for k in ("print_type", "hms", "print_error", "stg_cur"):
            if k in pr:
                live[k] = pr[k]
    ok3 = True
except Exception as e:
    ok3 = False
    fail.append(f"MQTT 접속 실패: {e}")
print(f"  [{'OK ' if ok3 else 'X  '}] 3. 프린터 도달")

# 🔴 `gcode_state` 는 **지난 작업의 결말**이지 현재 상태가 아니다. 새 작업이 들어오기 전까지
# FAILED/FINISH 가 계속 걸려 있다. 2026-09-01: 프린터가 실제로 대기 중인데 FAILED 가 남아
# 있어 전송이 잘못 차단됐다. 현재 가동 여부는 print_type/hms/print_error 로 판정한다.
BUSY_STATES = {"RUNNING", "PAUSE", "PREPARE", "SLICING"}
busy = (state in BUSY_STATES) or (str(live.get("print_type", "")).lower() == "printing")
hms = live.get("hms", [])
err = live.get("print_error", 0)
ok4 = ok3 and not busy and not hms and not err
if ok3 and not ok4:
    if busy:
        fail.append(f"프린터 가동 중 (state={state}, print_type={live.get('print_type')}) — 진행 중 작업을 덮어쓰지 않는다")
    if hms:
        fail.append(f"프린터 알림 {len(hms)}건 미해제 (hms={hms}) — 해제 전 새 출력 금지")
    if err:
        fail.append(f"print_error={err} — 원인 확인 전 새 출력 금지")
print(f"  [{'OK ' if ok4 else 'X  '}] 4. 프린터 전송 가능 상태  "
      f"(직전결말={state} · 현재={live.get('print_type')} · hms={len(hms)} · err={err})")

# 5) 사용자 승인
print(f"  [{'OK ' if a.yes else '-  '}] 5. 사용자 시작 신호 (--yes)")

if fail:
    print("\n전송 중단:")
    for f in fail:
        print("  -", f)
    try: p.mqtt_disconnect()
    except Exception: pass
    sys.exit(1)

if not a.yes:
    print("\n검증만 수행 (dry-run). 실제 전송하려면 --yes")
    try: p.mqtt_disconnect()
    except Exception: pass
    sys.exit(0)

# ── 전송 ──────────────────────────────────────────────────────────────────
print("\n전송 시작")
for step in job["send_plan"]:
    op, args = step["op"], step["args"]
    if op == "ftp_upload":
        p.ftp_connect()
        p.ftp_upload(args["local"], f"/{args['remote']}")
        p.ftp_disconnect()
        print(f"  [{step['step']}] 업로드 완료 → /{args['remote']}")
    elif op == "mqtt_print":
        p.mqtt_print(filename=args["filename"], use_ams=args["use_ams"],
                     bed_leveling=args["bed_leveling"], plate_id=args["plate_id"])
        print(f"  [{step['step']}] 출력 명령 전송 (plate_id={args['plate_id']})")

p.mqtt_disconnect()
print(f"\n예상 {job['print_params']['est_seconds']}초 / {job['print_params']['est_grams']}g")
print("모니터링: python send_print_job.py <manifest> 으로 상태만 재조회하거나 print_cli.py monitor")
