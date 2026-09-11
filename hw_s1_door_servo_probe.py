"""S1 문(door) 서보 프로브 — p45 조립 순서 7(전원 ON 절차)·8(치수 검수) 보조. D479·D480 규약.

무엇을 하나: 그리퍼 서보(joint 6) **만** 움직인다. 팔 관절 5개는 부팅 HOME 그대로 둔다.
  연결 → 현재 관절 읽기(부팅 = 그리퍼 π = SDK 0° 인지 확인, |roll| ≤ 14° 확인)
  → {"T":107,"tor":TOR} 토크 상한(기본 200, 부팅마다 1000 으로 복귀하므로 매번)
  → SDK 각도 0 → 10 → 20 → 30 → 20 → 10 → 0 (각 단계 settle 후 읽기, 사용자 캘리퍼스 입력 대기)
  → 로그 JSON: claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/servo_probe_<ts>.json

안전 (hardware.md 말미 5조):
  - 맨 {"T":106} 은 절대 안 보낸다(조 118.5° 개방 = S1 문 파손). 명령은 SDK joint_angle_ctrl(joint=6) 만.
  - 하드 상한 30° (D480: 30° 초과 스윕 미검증). --max-deg 는 30 이하만 받는다.
  - 개구 > 44 mm(≈22°)면 |roll| ≤ 14° — 시작 시 roll 읽어 초과면 중단.
  - SIGINT/예외 → 문 닫힘(0°) 명령 후 종료. 토크는 끄지 않는다(문이 떨어지지 않게).
  - 이 스크립트는 사용자 명시 승인 후에만 실행한다 (AGENTS.md Safety constraints).

사용:
  python hw_s1_door_servo_probe.py --dry-run                       # 로직만, 시리얼 0
  python hw_s1_door_servo_probe.py --port /dev/ttyUSB1 --read-only  # 연결·읽기만 (조립 1단계 검증)
  python hw_s1_door_servo_probe.py --port /dev/ttyUSB1              # 7·8단계 본 실행
"""
import argparse
import atexit
import math
import json
import os
import signal
import sys
import time

import safety_p0_guards as G

HARD_MAX_DEG = 30.0        # D480 소프트웨어 상한 — 여기 위로 올리지 말 것
ROLL_MAX_DEG = 14.0        # D473 ⑥ 개구 > 44 mm 일 때
GRIPPER_IDX = 5
OUT_DIR = "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real"
STEPS_DEFAULT = "0,10,20,30,20,10,0"


def clamp_door(deg, max_deg):
    return max(0.0, min(float(max_deg), float(deg)))


def send_torque_limit(arm, tor):
    """{"T":107,"tor":N} — SDK 에 메서드가 없어 원시 JSON 을 같은 포트로 쓴다. 읽기 경로가 없어 '발행'만 기록."""
    msg = json.dumps({"T": 107, "tor": int(tor)}) + "\n"
    with arm.lock:
        arm._serial_port.write(msg.encode())
        arm._serial_port.flush()
    time.sleep(G.INTER_CMD_DELAY)
    return msg.strip()


def raw_feedback(arm):
    """T:105 → T:1051 원시 dict (g = 펌웨어 rad, tG = 그리퍼 서보 부하). SDK 는 tG 를 버리므로 base_controller 버퍼에서 읽는다."""
    try:
        arm.feedback_get()
        d = dict(arm.base_controller.data_buffer) if arm.base_controller is not None else {}
    except Exception as e:
        d = {"error": str(e)}
    if "g" in d:
        # SDK handle_m3_feedback 가 버퍼를 제자리에서 g = π − g_fw 로 바꿔 둔다(09-07 실측: 버퍼 g 0.0445 = joints[5] 2.55°). 따라서 그대로 deg 환산.
        d["door_deg_from_g"] = round(math.degrees(float(d["g"])), 2)
    if "tG" not in d:
        d["tG"] = None   # 펌웨어 T:1051 은 tB..tR 만 보냄(09-07 실측) — 그리퍼 부하 미제공
    return d


def cmd_door(arm, deg):
    arm.joint_angle_ctrl(joint=6, angle=float(deg), speed=G.SPEED, acc=G.ACC)
    time.sleep(G.INTER_CMD_DELAY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="예: /dev/ttyUSB1 (명시 필수, dry-run 제외)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--read-only", action="store_true", help="연결·읽기만, 명령 0")
    ap.add_argument("--tor", type=int, default=200, help="T:107 토크 상한 50~1000 (기본 200)")
    ap.add_argument("--max-deg", type=float, default=HARD_MAX_DEG)
    ap.add_argument("--steps", default=STEPS_DEFAULT)
    ap.add_argument("--no-pause", action="store_true", help="단계마다 Enter 대기 생략")
    ap.add_argument("--torque-only", action="store_true", help="T:107 토크 상한만 발행하고 이동 0 (눌림 완화용)")
    a = ap.parse_args()

    if a.max_deg > HARD_MAX_DEG:
        sys.exit(f"--max-deg {a.max_deg} > 하드 상한 {HARD_MAX_DEG} (D480). 거부.")
    if not (50 <= a.tor <= 1000):
        sys.exit(f"--tor {a.tor} 범위 밖(50~1000).")
    steps = [clamp_door(s, a.max_deg) for s in a.steps.split(",")]
    print(f"계획: tor={a.tor} · steps={steps} · 상한 {a.max_deg}° · speed {G.SPEED} acc {G.ACC}")

    if a.dry_run:
        print("[dry-run] 시리얼 0 · 명령 0. 계획만 출력. 종료.")
        return 0
    if not a.port:
        sys.exit("--port 를 명시할 것 (Leader=/dev/ttyUSB0, Follower=/dev/ttyUSB1 관례 — 라벨로 확인).")

    G._install_silent_process()
    from roarm_sdk.roarm import roarm
    arm = roarm(roarm_type="roarm_m3", port=a.port, baudrate=115200)
    time.sleep(1.0)
    log = {"port": a.port, "tor": a.tor, "steps_planned": steps, "t0": time.strftime("%Y-%m-%d %H:%M:%S"),
           "events": []}
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"servo_probe_{time.strftime('%Y%m%d_%H%M%S')}.json")

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)

    closed = {"done": False}

    def close_door_and_exit(*_):
        if closed["done"]:
            return
        closed["done"] = True
        try:
            if not a.read_only:
                cmd_door(arm, 0.0)
                log["events"].append({"t": time.time(), "ev": "exit_close_0"})
        finally:
            save()
            try:
                arm.disconnect()
            except Exception:
                pass
            print(f"로그 → {out_path}")

    atexit.register(close_door_and_exit)
    signal.signal(signal.SIGINT, lambda *_: (close_door_and_exit(), sys.exit(130)))

    state0 = G.safe_get(arm)
    log["events"].append({"t": time.time(), "ev": "read_boot", "joints_deg": state0})
    print(f"부팅 읽기 joints(deg)={[round(x, 2) for x in state0]}  ← 그리퍼[5] ≈ 0 이면 π(닫힘) 확인")
    if abs(state0[4]) > ROLL_MAX_DEG:
        sys.exit(f"roll {state0[4]:.1f}° > {ROLL_MAX_DEG}° — 문을 열면 link4 간섭 가능(D473 ⑥). 롤을 0 근처로 두고 재실행.")
    if abs(state0[GRIPPER_IDX]) > 5.0:
        print(f"⚠️ 그리퍼 읽기 {state0[GRIPPER_IDX]:.1f}° — 부팅 π(0°)가 아니다. 문이 이미 열려 있거나 조립 전 상태.")
    fb = raw_feedback(arm); log["events"].append({"t": time.time(), "ev": "raw_feedback_boot", "fb": fb}); save()
    print(f"원시 피드백: g={fb.get('g')} rad → 문 {fb.get('door_deg_from_g')}°  부하 tG={fb.get('tG')}  (참고 tB..tR={[fb.get(k) for k in ('tB','tS','tE','tT','tR')]})")
    if a.read_only:
        print("[read-only] 명령 0. 종료.")
        return 0
    if a.torque_only:
        msg = send_torque_limit(arm, a.tor); log["events"].append({"t": time.time(), "ev": "torque_limit_issued", "raw": msg})
        time.sleep(1.0); fb2 = raw_feedback(arm); log["events"].append({"t": time.time(), "ev": "raw_feedback_after_torque", "fb": fb2}); save()
        print(f"[torque-only] {msg} 발행 → 부하 tG {fb.get('tG')} → {fb2.get('tG')}. 이동 0. 종료.")
        return 0

    msg = send_torque_limit(arm, a.tor)
    log["events"].append({"t": time.time(), "ev": "torque_limit_issued", "raw": msg,
                          "note": "발행만 — 펌웨어 읽기 경로 없음(D468 '미확인' 규약)"})
    print(f"T:107 발행: {msg}  (읽기 불가 → '발행'으로만 기록)")

    for i, tgt in enumerate(steps):
        cmd_door(arm, tgt)
        val, n, dt, status = G.poll_until_settled(arm, tgt, idx=GRIPPER_IDX, max_s=12.0, poll=1.0)
        joints = G.safe_get(arm); fb = raw_feedback(arm)
        ev = {"t": time.time(), "ev": "step", "i": i, "target_deg": tgt, "settled_deg": val,
              "polls": n, "elapsed_s": round(dt, 2), "status": status, "joints_deg": joints, "load_tG": fb.get("tG"), "g_rad": fb.get("g")}
        print(f"[{i}] 목표 {tgt:5.1f}° → 읽기 {val:6.2f}° ({status}, {n}폴 {dt:.1f}s)  부하 tG={fb.get('tG')}  joints={[round(x, 1) for x in joints]}")
        if not a.no_pause:
            s = input("    캘리퍼스 입력(입 폭 mm / 립 틈 mm / 메모, 빈칸=건너뜀) > ").strip()
            if s:
                ev["user_measure"] = s
        log["events"].append(ev)
        save()
        if status == "TIMEOUT" and abs(val - tgt) > 5.0:
            print("    ⚠️ 목표 미도달 5° 초과 — 걸림 의심. 중단하고 닫는다.")
            break

    close_door_and_exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
