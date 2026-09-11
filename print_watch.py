"""P1S 출력 감시 — 76th(2026-09-02) 스크래치패드 `watch_print.py` 를 repo 로 이전 + 대상 파라미터화.

🔴 전제: MQTT 텔레메트리로는 베드 박리를 감지할 수 없다(g9: 떨어진 뒤에도 RUNNING·err=0).
  (1) 텔레메트리 — 오류 플래그/상태 이탈을 **알리기만** 한다. 자동 정지 없음(자동 정지가 옳았던 사례 0건, g11 필라멘트 정지는 복구 가능했음).
  (2) 챔버 카메라 — 저장만. 판정은 사람(또는 세션이 프레임을 직접 본다).
🔴 blank 텔레메트리: pushall 은 요청형이라 프린터가 푸시를 멈추면 예외 없이 None 만 온다 → blank 2·5·9회에 강제 재접속, 12회에 종료.
🔴 감시 실행 중 별도 MQTT 조회 금지 — P1S 는 동시 클라이언트를 제한한다(76th). 상태는 이 스크립트의 stdout/jsonl 로 본다.

사용: python print_watch.py --out-dir claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v0_print --subtask roarm_s1_v0 --tag s1 \
        [--poll 60] [--cam-every 300] [--cam-layers 12,20,23,60,100,117]
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, "/home/cgxr/Documents/DK/DTR/bamboo-3dprinter")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import printer as bp                       # noqa: E402
from print_cam_snapshot import grab_retry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
ap.add_argument("--subtask", required=True)
ap.add_argument("--tag", default="p")
ap.add_argument("--ip", default="192.168.0.96")
ap.add_argument("--poll", type=int, default=60)
ap.add_argument("--cam-every", type=int, default=300)
ap.add_argument("--cam-layers", default="", help="이 층에 도달하면 즉시 1장 (예: 12,20,23)")
a = ap.parse_args()

OUT = pathlib.Path(a.out_dir)
CAM = OUT / "cam"
CAM.mkdir(parents=True, exist_ok=True)
LOG = OUT / "monitor.jsonl"
CAM_LAYERS = sorted({int(x) for x in a.cam_layers.split(",") if x.strip()})

cfg = bp.load_config()
cfg["printer"]["ip"] = a.ip
ACCESS = cfg["printer"]["access_code"]


def connect(n=8):
    for _ in range(n):
        try:
            p = bp.P1SPrinter(config=cfg)
            p.mqtt_connect()
            return p
        except Exception:
            time.sleep(3)
    return None


def snap(tag):
    try:
        f = CAM / f"{a.tag}_{tag}.jpg"
        n = grab_retry(a.ip, ACCESS, str(f), frames=2)   # 첫 장은 버린다(옛 프레임 함정)
        return f.name, n
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def log(rec):
    with open(LOG, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


t0 = time.time()
last_cam = 0.0
blank = 0
alerted = False
stop_reason = None
known = {}                 # 마지막으로 본 값 누적(델타 메시지 대비)
p = connect()
if p is None:
    print("MQTT 접속 실패 — 감시 시작 못 함")
    sys.exit(1)
print(f"감시 시작 subtask={a.subtask}. 로그={LOG}  프레임={CAM}  cam_layers={CAM_LAYERS}")
sys.stdout.flush()

while True:
    el = time.time() - t0
    try:
        fresh = {}
        for m in p.mqtt_pushall(wait=5):
            fresh.update(m.get("print", {}))
    except Exception:
        try:
            p.mqtt_disconnect()
        except Exception:
            pass
        p = connect()
        if p is None:
            log({"t": round(el), "event": "mqtt_lost"})
            print(f"[{el/60:5.1f}m] MQTT 재접속 실패 — 계속 시도")
            time.sleep(a.poll)
        continue

    gs = fresh.get("gcode_state")
    if gs is None:
        blank += 1
        if blank in (2, 5, 9):
            print(f"[{el/60:5.1f}m] blank {blank}회 — 강제 재접속")
            sys.stdout.flush()
            try:
                p.mqtt_disconnect()
            except Exception:
                pass
            np_ = connect()
            if np_ is not None:
                p = np_
                time.sleep(2)
                continue
        if blank >= 12:
            log({"t": round(el), "event": "ended_blank_telemetry", "last_known": known})
            stop_reason = f"텔레메트리 {blank}회 연속 blank(재접속 3회 포함) — 출력 성패는 이 신호로 판정하지 말 것"
            break
        time.sleep(a.poll)
        continue
    blank = 0
    known.update(fresh)
    s = known

    rec = {"t": round(el), "clock": time.strftime("%H:%M:%S"), "gcode_state": gs, "subtask": s.get("subtask_name"),
           "layer": s.get("layer_num"), "total": s.get("total_layer_num"), "pct": s.get("mc_percent"),
           "remain_min": s.get("mc_remaining_time"), "nozzle": s.get("nozzle_temper"), "bed": s.get("bed_temper"),
           "err": s.get("print_error"), "hms": s.get("hms")}

    if s.get("subtask_name") not in (None, a.subtask):
        rec["event"] = "ALERT_subtask_mismatch"

    if s.get("print_error") or s.get("hms"):
        if not alerted:
            rec["event"] = "ALERT_error_flag"
            print(f"[{el/60:5.1f}m] 🔴 오류 플래그 err={s.get('print_error')} hms={s.get('hms')} state={gs} "
                  f"층={s.get('layer_num')} — 자동 중단 안 함. 사람이 판정할 것")
            alerted = True
    else:
        alerted = False

    if gs in ("FINISH", "FAILED"):
        rec["event"] = "ended"
        name, n = snap(f"END_{gs}_L{s.get('layer_num') or 0:03d}")
        rec["cam"] = name or f"FAIL {n}"
        log(rec)
        stop_reason = f"출력 종료 gcode_state={gs} 층 {s.get('layer_num')}/{s.get('total_layer_num')}"
        break

    layer = s.get("layer_num") or 0
    due_layer = next((L for L in CAM_LAYERS if layer >= L), None)
    if due_layer is not None:
        CAM_LAYERS = [L for L in CAM_LAYERS if L > layer]
    if due_layer is not None or time.time() - last_cam >= a.cam_every:
        name, n = snap(f"L{layer:03d}")
        rec["cam"] = name or f"FAIL {n}"
        last_cam = time.time()

    log(rec)
    print(f"[{el/60:5.1f}m {rec['clock']}] {gs} 층 {layer}/{s.get('total_layer_num')} {s.get('mc_percent')}% "
          f"남은 {s.get('mc_remaining_time')}분 노즐 {s.get('nozzle_temper')} 베드 {s.get('bed_temper')}"
          + (f"  cam={rec.get('cam')}" if "cam" in rec else "") + (f"  ⚠️{rec['event']}" if "event" in rec else ""))
    sys.stdout.flush()
    time.sleep(a.poll)

try:
    p.mqtt_disconnect()
except Exception:
    pass
print(f"감시 종료: {stop_reason}")
