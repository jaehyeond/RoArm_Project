"""S1 그랩 퍼내기 실물 시행 v2 — 관절 제어 + URDF FK(`sim_scripts/roarm_kinematics.py`). 높이는 **런타임 인자**로 받고 웨이포인트는 그 자리에서 푼다.

v1(09-07 13:57) 교훈: 베이스판 높이를 30 cm 로 잘못 알아 립이 펠릿면 15 cm 위에서 열고 닫힘(영상). 실제 ≈ 48 cm(영상·사진 16). 추정값을 코드에 박지 말 것.
순서(v2, 사용자 요구): HOME → P1 툴 세움 → P2 상자 위(손목 ≤ 90 클램프 안) → **닫힌 채** 펠릿면까지 단계 하강 → 펠릿면에서 문 열기 30°
  → PLUNGE cm 더 잠김 → 문 닫기(토크 TOR_CLOSE) → 들기(펠릿면 +8 → 상자 위) → [--dump] → P1 → HOME.
안전: 속도 200/가속 50 · 관절 클램프 · 명령 간 1 s · 편차 5° 초과 또는 립 하한(펠릿면 −PLUNGE −2 cm) 위반 → 정지·유지(맹목 HOME 금지)
  · 손목 |q| ≤ 90(펌웨어 클램프) · 문 상한 30° · 맨 T:106 금지 · 립 x = 상자 중심(--box-x) 고정, 툴 수직(P2 만 기울임 허용).
사용: python hw_s1_scoop_probe.py --dry-run --base-cm 48 --pellet-cm 31 --boxtop-cm 46 [--box-x-cm 35] [--plunge-cm 3]
      python hw_s1_scoop_probe.py --port /dev/ttyUSB0 --base-cm 48 --pellet-cm 31 --boxtop-cm 46 [--dump] [--resume-from P2_above_box]
"""
import argparse, json, os, signal, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_scripts"))
import roarm_kinematics as K
import safety_p0_guards as G

SHOULDER_ABOVE_PLATE = 0.0701 + 0.05196   # 어깨축(펌웨어 z 원점) = 베이스판 + 0.122 m (URDF)
LIP_L5 = np.array([0.0081, 0.0, 0.1666, 1.0])   # S1 립 = link5 (8.1, 0, 166.6) mm
DOOR_OPEN, DOOR_CLOSED = 30.0, 0.0
TOR_OPEN, TOR_CLOSE = 200, 600
WRIST_MAX = 90.0                           # 펌웨어 클램프(09-07 실측)
HOME = [0.0, 0.0, 90.0, 0.0, 0.0]
P1 = [0.0, 0.7, 91.3, 88.0, 0.0]           # 툴 세움(립 뒤로 후퇴, 어깨축 +1 cm)
OUT_DIR = "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real"


def chain(q5):
    q = np.radians(list(q5) + [0.0]); T = np.eye(4); out = {}
    for name, xyz, rpy, qi in K._CHAIN:
        T = T @ K.Tmat(xyz, rpy)
        if qi is not None: T = T @ K.Trot_z(q[qi])
        out[name] = T.copy()
    return out


def lip_fw(q5):
    """립 위치(어깨축 기준 m) + 툴축(세계)."""
    T = chain(q5)["link4_to_link5"]; p = T @ LIP_L5
    return np.array([p[0], p[1], p[2] - SHOULDER_ABOVE_PLATE]), T[:3, 2]


def solve_vertical(x_t, z_t, wrist_max=WRIST_MAX, tilt_ok=0.995):
    """툴 수직·립 (x_t, z_t) 해. 격자 0.5° (y=0 평면). 반환 (err_m, q5)."""
    best = None
    for sh in np.arange(-30, 110.1, 0.5):
        for el in np.arange(-10, 150.1, 0.5):
            wp = 88.0 + (0.7 - sh) + (91.32 - el)
            if abs(wp) > wrist_max: continue
            q = [0.0, float(sh), float(el), float(wp), 0.0]; l, z = lip_fw(q)
            if -z[2] < tilt_ok: continue
            e = float(np.hypot(l[0] - x_t, l[2] - z_t))
            if best is None or e < best[0]: best = (e, q)
    return best


def build_waypoints(a):
    floor = -(a.base_cm / 100.0 + SHOULDER_ABOVE_PLATE)          # 바닥 z_fw
    pellet = floor + a.pellet_cm / 100.0; boxtop = floor + a.boxtop_cm / 100.0; x = a.box_x_cm / 100.0
    W = [("P1_tool_down", P1)]
    # P2: 상자 윗단 +2 cm 를 수직으로 못 잡으면 손목 90 고정 기울임 허용
    r = solve_vertical(x, boxtop + 0.02)
    if r[0] > 0.01:
        best = None
        for sh in np.arange(30, 100.1, 0.5):
            for el in np.arange(0, 120.1, 0.5):
                q = [0.0, float(sh), float(el), WRIST_MAX, 0.0]; l, z = lip_fw(q)
                if -z[2] < 0.97: continue
                e = float(np.hypot(l[0] - x, l[2] - (boxtop + 0.02)))
                if best is None or e < best[0]: best = (e, q)
        r = best
    W.append(("P2_above_box", r[1]))
    z = boxtop + 0.02 - 0.05
    i = 0
    while z > pellet + 0.005:                                      # 5 cm 단계, 마지막은 펠릿면
        W.append((f"P3_{i}_down", solve_vertical(x, z)[1])); z -= 0.05; i += 1
    W.append(("P4_pellet_surface", solve_vertical(x, pellet)[1]))
    W.append(("P5_plunge", solve_vertical(x, pellet - a.plunge_cm / 100.0)[1]))
    return W, dict(floor=floor, pellet=pellet, boxtop=boxtop, x=x, lip_min=pellet - a.plunge_cm / 100.0 - 0.02)


def precheck(W, g):
    bad = []
    seq = [("HOME", HOME)] + W
    for (n0, q0), (n1, q1) in zip(seq[:-1], seq[1:]):
        for s in np.linspace(0, 1, 21):
            q = [p + (r - p) * s for p, r in zip(q0, q1)]
            lip, zax = lip_fw(q); ch = chain(q)
            if lip[2] < g["lip_min"] - 1e-6: bad.append(f"{n0}→{n1} s={s:.2f} 립 z {lip[2]:.3f} < 하한 {g['lip_min']:.3f}")
            xw = g["x"] - 0.155                                     # 상자 앞벽
            for k in ("link3_to_link4", "link4_to_link5"):
                p = ch[k][:3, 3]; z = p[2] - SHOULDER_ABOVE_PLATE
                if xw - 0.02 <= p[0] <= xw + 0.02 and z < g["boxtop"] + 0.02: bad.append(f"{n0}→{n1} s={s:.2f} {k} 앞벽 위 z {z:.3f}")
            if xw - 0.03 <= lip[0] <= xw + 0.03 and lip[2] < g["boxtop"] + 0.02: bad.append(f"{n0}→{n1} s={s:.2f} 립 앞벽 통과 z {lip[2]:.3f}")
    for n, q in W:
        if abs(q[3]) > WRIST_MAX + 1e-6: bad.append(f"{n} 손목 {q[3]} > {WRIST_MAX}")
        cl = G.clamp_joints(list(q) + [0.0])[:5]
        if any(abs(p - r) > 1e-6 for p, r in zip(cl, q)): bad.append(f"{n} 관절 한계 클램프 {q}→{cl}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--base-cm", type=float, required=True, help="바닥→베이스판 윗면 (줄자 실측)")
    ap.add_argument("--pellet-cm", type=float, required=True); ap.add_argument("--boxtop-cm", type=float, required=True)
    ap.add_argument("--box-x-cm", type=float, default=35.0, help="베이스 회전축→상자 중심 전방 거리")
    ap.add_argument("--plunge-cm", type=float, default=3.0)
    ap.add_argument("--tor-close", type=int, default=900, help="닫힘 토크 상한(50~1000). v2 600 은 5.7° 에서 정지(펠릿 물림)")
    ap.add_argument("--close-retries", type=int, default=3, help="닫힘 각 > 3.6° 면 8° 열었다 재닫기 횟수(채터링)")
    ap.add_argument("--dump", action="store_true"); ap.add_argument("--resume-from", default=None)
    ap.add_argument("--dump-only", action="store_true", help="퍼내기 없이 HOME→P1→P2 에서 문 열어 되붓기→닫기→HOME")
    a = ap.parse_args()
    W, g = build_waypoints(a); bad = precheck(W, g)
    print(f"모델: 바닥 z_fw {g['floor']:+.3f} · 펠릿면 {g['pellet']:+.3f} · 상자 윗단 {g['boxtop']:+.3f} · 립 하한 {g['lip_min']:+.3f} (어깨축 = 바닥+{a.base_cm+12.2:.1f} cm)")
    print("계획:")
    for n, q in [("HOME", HOME)] + W:
        lip, zax = lip_fw(q); print(f"  {n:20s} q={[round(v,1) for v in q]}  립 (x {lip[0]*100:.1f}, 바닥+{(lip[2]-g['floor'])*100:.1f} cm)  수직도 {-zax[2]:.3f}")
    print("사전 검사:", "PASS" if not bad else "FAIL"); [print("   ", b) for b in bad]
    if bad: return 2
    if a.dry_run: print("[dry-run] 시리얼 0. 종료."); return 0
    if not a.port: sys.exit("--port 필요")

    G._install_silent_process()
    from roarm_sdk.roarm import roarm
    arm = roarm(roarm_type="roarm_m3", port=a.port, baudrate=115200); time.sleep(1.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"scoop_probe_{time.strftime('%Y%m%d_%H%M%S')}.json")
    log = {"t0": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "model": g, "waypoints": W, "events": []}
    def save(): json.dump(log, open(out, "w"), ensure_ascii=False, indent=1, default=float)
    def ev(**kw): kw["t"] = round(time.time(), 2); log["events"].append(kw); save()
    held = {"v": False}
    def hold_and_exit(reason):
        if held["v"]: return
        held["v"] = True; ev(ev_="HOLD", reason=reason); print(f"🔴 정지·유지: {reason}  (토크 유지, 복귀는 사람이 판단)")
        try: arm.disconnect()
        except Exception: pass
        sys.exit(3)
    signal.signal(signal.SIGINT, lambda *_: hold_and_exit("SIGINT"))

    def loads():
        try:
            arm.feedback_get(); d = arm.base_controller.data_buffer; return [d.get(k) for k in ("tB", "tS", "tE", "tT", "tR")]
        except Exception: return None

    def torque(tor):
        msg = json.dumps({"T": 107, "tor": int(tor)}) + "\n"
        with arm.lock: arm._serial_port.write(msg.encode()); arm._serial_port.flush()
        time.sleep(G.INTER_CMD_DELAY); ev(ev_="torque", tor=tor)

    def door(deg):
        deg = max(0.0, min(30.0, deg)); arm.joint_angle_ctrl(joint=6, angle=deg, speed=G.SPEED, acc=G.ACC); time.sleep(G.INTER_CMD_DELAY)
        val, n, dt, st = G.poll_until_settled(arm, deg, idx=5, max_s=10.0, poll=1.0)
        ev(ev_="door", target=deg, read=val, status=st, loads=loads()); print(f"  문 {deg:.0f}° → 읽기 {val:.1f}° ({st})"); return val

    def goto(name, q5, tol_deg=5.0):
        cur = G.safe_get(arm); cmd = list(q5) + [cur[5]]
        j = int(np.argmax(np.abs(np.array(q5) - np.array(cur[:5]))))
        G.move_joints(arm, cmd, settle_idx=j, settle_target=q5[j])
        time.sleep(0.5); rd = G.safe_get(arm); lip, _ = lip_fw(rd[:5]); p = G.safe_pose(arm); ld = loads()
        dev = max(abs(x - y) for x, y in zip(rd[:5], q5))
        ev(ev_="goto", name=name, cmd=q5, read=[round(x, 2) for x in rd], lip_fw=[round(float(x), 4) for x in lip], fw_pose=p, dev_deg=round(dev, 2), loads=ld)
        print(f"  {name:20s} 읽기 {[round(x,1) for x in rd[:5]]}  립 바닥+{(lip[2]-g['floor'])*100:.1f} cm  편차 {dev:.1f}°  부하 {ld}")
        if dev > tol_deg: hold_and_exit(f"{name} 편차 {dev:.1f}° > {tol_deg}")
        if lip[2] < g["lip_min"] - 0.01: hold_and_exit(f"{name} 립 z {lip[2]:.3f} < 하한")

    st = G.safe_get(arm); print("시작 읽기", [round(x, 2) for x in st]); ev(ev_="start", read=st, loads=loads())
    names = [n for n, _ in W]; start_i = 0
    if a.resume_from:
        start_i = names.index(a.resume_from)
        if max(abs(x - y) for x, y in zip(st[:5], W[start_i][1])) > 6.0: hold_and_exit(f"현재 자세가 {a.resume_from} 와 6° 초과 차이")
    elif max(abs(x - y) for x, y in zip(st[:5], HOME)) > 6.0: hold_and_exit("시작 자세가 HOME 이 아님")
    torque(TOR_OPEN)
    if a.dump_only:
        goto("P1_tool_down", P1); goto("P2_above_box", W[1][1])
        door(DOOR_OPEN); time.sleep(3.0); door(DOOR_CLOSED)
        goto("P1_tool_down_back", P1); goto("HOME", HOME); ev(ev_="dump_done", loads=loads()); print(f"되붓기 완료. 로그 → {out}")
        try: arm.disconnect()
        except Exception: pass
        return 0
    for i, (n, q) in enumerate(W):
        if i < start_i: continue
        if n == "P5_plunge":                       # 펠릿면에서 문 열고 잠김
            door(DOOR_OPEN); time.sleep(0.5)
        goto(n, q)
    torque(a.tor_close); val = door(DOOR_CLOSED); time.sleep(0.5)
    for k in range(a.close_retries):                       # 채터링: 립 사이 끼인 펠릿 털기
        if val <= 3.6: break
        print(f"  닫힘 {val:.1f}° > 3.6 → 채터링 {k+1}/{a.close_retries}")
        door(8.0); time.sleep(0.3); val = door(DOOR_CLOSED); time.sleep(0.5)
    ev(ev_="close_final", read=val, loads=loads())
    up1 = solve_vertical(g["x"], g["pellet"] + 0.08)[1]; goto("UP_pellet+8cm", up1)
    d1 = G.safe_get(arm)[5]; print(f"  펠릿면+8 cm 에서 문 읽기 {d1:.1f}° → 재닫힘 명령"); val = door(DOOR_CLOSED)   # 리프트 중 되열림 보정
    goto("UP_above_box", W[1][1]); dv = G.safe_get(arm)[5]; ev(ev_="lifted", door_after_lift=dv, door_at_plus8=d1, loads=loads()); print(f"  들어 올린 뒤 문 읽기 {dv:.1f}°")
    print("  ▶ 들어 올림. 사용자 확인 대기 8 s"); time.sleep(8.0)
    if a.dump:
        torque(TOR_OPEN); door(DOOR_OPEN); time.sleep(3.0); door(DOOR_CLOSED)
    goto("P1_tool_down_back", P1); goto("HOME", HOME)
    ev(ev_="done", loads=loads()); print(f"완료. 로그 → {out}")
    try: arm.disconnect()
    except Exception: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
