"""S1 그랩 수동 운용 REPL — 가서 집고(퍼고), 들어 올리고, 저장한 위치에 놓기. 카메라 없이 **조그로 위치를 찾아 저장**한다.

실행:  ~/miniconda3/envs/roarm/bin/python hw_s1_manual.py --port /dev/ttyUSB0 --base-cm 38 --pellet-cm 26 --boxtop-cm 38.5
       ~/miniconda3/envs/roarm/bin/python hw_s1_manual.py --sim ...           # 하드웨어 없이 명령 흐름 연습
명령(cm 단위, 툴은 항상 수직·입 아래):
  status                 관절·립 위치(앞 x, 옆 y, 바닥 z)·문 각·부하
  home | p1 | above      HOME / 툴 세움 / 더미 중심 위(travel 높이)
  x +2  y -3  z -1       립 조그 (한 번에 ≤ 5 cm). 상자 안(z < 윗단)에서는 x,y 조그 금지(벽) → 먼저 z 로 올릴 것
  open | close [tor]     문 30° / 문 0°(토크, 기본 900) · chatter = 8° 열었다 닫기
  scoop [plunge]         현재 x,y 에서: 펠릿면까지 하강 → 열기 → plunge cm 잠김(기본 2.5) → 닫기(900, 재시도) → 펠릿면+8 → 재닫기 → travel 높이
  save <이름> | goto <이름> | list     현재 립 위치를 이름으로 저장(파일) / 저장 위치로 이동(올리고→옮기고→내리기)
  place [각도]            당겨 올려(P1) 베이스 +90°(기본) 회전 → 더미 반경으로 뻗어 펠릿면 높이까지 하강 → open 1.5 s → close → 복귀(P1, base 0)
  dump                   goto dump(저장 위치) → 열기 1.5 s → 닫기 → travel 높이로
  cycle [n]              above → scoop → place → above 를 n 회(기본 1)
  weigh [n]              cycle 1 회마다 저울 읽기(g)를 물어 기록 → n 회 후 평균·편차 (설계 17.6 g). 컵은 place 자리(+y 90°, 립 바닥+26)에 둘 것
  mass <g>               직전 scoop 의 적재 질량을 수동 기록 (누적 파일 s1_v1_real/mass_log.jsonl)
  quit                   travel 높이 → p1 → HOME 후 종료
안전: 속도 200/가속 50 · 손목 |q| ≤ 90(펌웨어 클램프) · 립 하한 = 펠릿면 −4 cm · 편차 5° 초과 시 정지·유지 · 문 ≤ 30° · 맨 T:106 금지 · 이동은 항상 올리고→옮기고→내리기.
"""
import argparse, json, math, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw_s1_scoop_probe as S
import safety_p0_guards as G

POS_FILE = os.path.join(S.OUT_DIR, "manual_positions.json")
MASS_FILE = os.path.join(S.OUT_DIR, "mass_log.jsonl")
HOME, P1 = S.HOME, S.P1


def _grid(r, z, sh_rng, el_rng, step, wrist_max, tilt_ok, wrist_fixed=None):
    best = None
    for sh in np.arange(sh_rng[0], sh_rng[1] + 1e-9, step):
        for el in np.arange(el_rng[0], el_rng[1] + 1e-9, step):
            wp = wrist_fixed if wrist_fixed is not None else 88.0 + (0.7 - sh) + (91.32 - el)
            if abs(wp) > wrist_max: continue
            q = [0.0, float(sh), float(el), float(wp), 0.0]; l, zx = S.lip_fw(q)
            if -zx[2] < tilt_ok: continue
            e = float(np.hypot(l[0] - r, l[2] - z))
            if best is None or e < best[0]: best = (e, q, float(-zx[2]))
    return best


def solve_fast(r, z, wrist_max=S.WRIST_MAX):
    """립 (반경 r, z_fw) 해. ① 툴 수직(2° 격자 → 0.25° 정밀) ② 실패 시 손목 90 고정·기울임 ≤ 14° 허용. 반환 (err_m, q5, 수직도) 또는 None."""
    b = _grid(r, z, (-30, 110), (-10, 150), 2.0, wrist_max, 0.995)
    if b is not None:
        b2 = _grid(r, z, (b[1][1] - 2, b[1][1] + 2), (b[1][2] - 2, b[1][2] + 2), 0.25, wrist_max, 0.995)
        if b2 is not None and b2[0] < b[0]: b = b2
    if b is None or b[0] > 0.01:
        t = _grid(r, z, (20, 110), (-10, 130), 1.0, wrist_max, 0.97, wrist_fixed=wrist_max)
        if t is not None and (b is None or t[0] < b[0]): b = t
    return b


class Manual:
    def __init__(self, a):
        self.a = a; self.sim = a.sim
        self.floor = -(a.base_cm / 100.0 + S.SHOULDER_ABOVE_PLATE)
        self.pellet = self.floor + a.pellet_cm / 100.0; self.boxtop = self.floor + a.boxtop_cm / 100.0
        self.travel = self.floor + a.travel_cm / 100.0; self.lip_min = self.pellet - 0.04
        self.pile_xy = (a.box_x_cm / 100.0, 0.0)
        self.pos = json.load(open(POS_FILE)) if os.path.exists(POS_FILE) else {}
        self.log_path = os.path.join(S.OUT_DIR, f"manual_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        os.makedirs(S.OUT_DIR, exist_ok=True)
        self.q_sim = list(HOME) + [3.0]; self.last_scoop = {}
        if not self.sim:
            G._install_silent_process()
            from roarm_sdk.roarm import roarm
            self.arm = roarm(roarm_type="roarm_m3", port=a.port, baudrate=115200); time.sleep(1.0)
            self.torque(200)

    # ── 저수준 ──
    def log(self, **kw):
        kw["t"] = round(time.time(), 2)
        with open(self.log_path, "a") as f: f.write(json.dumps(kw, ensure_ascii=False, default=float) + "\n")

    def read(self):
        return list(self.q_sim) if self.sim else G.safe_get(self.arm)

    def loads(self):
        if self.sim: return None
        try: self.arm.feedback_get(); d = self.arm.base_controller.data_buffer; return [d.get(k) for k in ("tB", "tS", "tE", "tT", "tR")]
        except Exception: return None

    def torque(self, tor):
        if self.sim: return
        msg = json.dumps({"T": 107, "tor": int(tor)}) + "\n"
        with self.arm.lock: self.arm._serial_port.write(msg.encode()); self.arm._serial_port.flush()
        time.sleep(0.3); self.log(ev="torque", tor=tor)

    def lip_world(self, q):
        """베이스 회전 포함 립 (x, y, z_fw)."""
        l, _ = S.lip_fw([0.0] + list(q[1:5])); b = math.radians(q[0])
        return np.array([l[0] * math.cos(b) - l[1] * math.sin(b), l[0] * math.sin(b) + l[1] * math.cos(b), l[2]])

    def settle(self, idx, max_s=8.0, poll=0.4, tol=0.3):
        """읽기값이 두 번 연속 tol 안에서 안 변하면 정착(목표 도달 여부 무관 — 문은 기계 정지, 관절은 처짐이 있어 '도달' 대기는 타임아웃만 낳는다)."""
        prev = None; t0 = time.time()
        while time.time() - t0 < max_s:
            time.sleep(poll); v = self.read()[idx]
            if prev is not None and abs(v - prev) < tol: return v, "STABLE"
            prev = v
        return prev, "TIMEOUT"

    def door(self, deg, tor=None):
        deg = max(0.0, min(30.0, deg))
        if tor is not None: self.torque(tor)
        if self.sim: self.q_sim[5] = deg if deg > 0 else 2.5; print(f"  [sim] 문 {deg}°"); return self.q_sim[5]
        self.arm.joint_angle_ctrl(joint=6, angle=deg, speed=G.SPEED, acc=G.ACC); time.sleep(0.6)
        val, st = self.settle(5)
        self.log(ev="door", target=deg, read=val, status=st); print(f"  문 {deg:.0f}° → {val:.1f}° ({st})"); return val

    def goto_q(self, name, q5, tol=5.0):
        if abs(q5[3]) > S.WRIST_MAX + 1e-6: print(f"  ✗ {name}: 손목 {q5[3]:.1f} > 90"); return False
        lipz = self.lip_world(q5)[2]
        if lipz < self.lip_min: print(f"  ✗ {name}: 립 바닥+{(lipz-self.floor)*100:.1f} cm < 하한 {(self.lip_min-self.floor)*100:.1f}"); return False
        if self.sim:
            self.q_sim[:5] = list(q5); l = self.lip_world(q5); print(f"  [sim] {name:14s} q={[round(v,1) for v in q5]}  립 x {l[0]*100:.1f} y {l[1]*100:.1f} 바닥+{(l[2]-self.floor)*100:.1f}"); return True
        cur = self.read(); j = int(np.argmax(np.abs(np.array(q5) - np.array(cur[:5]))))
        G.move_joints(self.arm, list(q5) + [cur[5]]); self.settle(j, max_s=10.0)
        rd = self.read(); l = self.lip_world(rd[:5]); dev = max(abs(x - y) for x, y in zip(rd[:5], q5))
        self.log(ev="goto", name=name, cmd=q5, read=rd, lip=l.tolist(), dev=dev, loads=self.loads())
        print(f"  {name:14s} 읽기 {[round(v,1) for v in rd[:5]]}  립 x {l[0]*100:.1f} y {l[1]*100:.1f} 바닥+{(l[2]-self.floor)*100:.1f}  편차 {dev:.1f}°")
        if dev > tol: print(f"  🔴 편차 {dev:.1f}° > {tol} — 정지·유지. status 로 확인 후 진행"); return False
        return True

    def goto_xyz(self, name, x, y, z):
        """립 (x,y,z_fw), 툴 수직. 베이스 = atan2."""
        b = math.degrees(math.atan2(y, x)); r = math.hypot(x, y)
        if abs(b) > 90: print("  ✗ 베이스 회전 90° 초과"); return False
        sol = solve_fast(r, z)
        if sol is None or sol[0] > 0.01: print(f"  ✗ {name}: 도달 불가 (r {r*100:.1f}, 바닥+{(z-self.floor)*100:.1f} cm{'' if sol is None else f', 최근접 오차 {sol[0]*1000:.0f} mm'})"); return False
        if sol[2] < 0.995: print(f"  (툴 기울임 {math.degrees(math.acos(sol[2])):.0f}° 허용)")
        q = list(sol[1]); q[0] = b; return self.goto_q(name, q)

    def inside_box_safe(self, x, y, margin=0.03):
        """상자 안쪽(벽에서 margin 이상) 인가. 상자 = 중심 (box_x, 0), 안쪽 31×22."""
        return (self.pile_xy[0] - 0.155 + margin <= x <= self.pile_xy[0] + 0.155 - margin) and (abs(y) <= 0.11 - margin)

    def cur_lip(self):
        return self.lip_world(self.read()[:5])

    # ── 상위 동작 ──
    def travel_to(self, x, y, z, name="goto"):
        """올리고 → 옮기고 → 내리기."""
        c = self.cur_lip()
        if c[2] < self.travel - 0.005 and not self.goto_q(name + "_up", self._solve_here(self.travel)): return False
        if not self.goto_xyz(name + "_move", x, y, self.travel): return False
        if z < self.travel - 0.005: return self.goto_xyz(name + "_down", x, y, z)
        return True

    def _solve_here(self, z):
        c = self.cur_lip(); b = math.degrees(math.atan2(c[1], c[0])); r = math.hypot(c[0], c[1])
        sol = solve_fast(r, z); q = list(sol[1]); q[0] = b; return q

    def scoop(self, plunge_cm=2.5, tor_close=900):
        c = self.cur_lip(); x, y = c[0], c[1]
        for zc, nm in ((self.pellet + 0.05, "5cm_above"), (self.pellet, "surface")):
            if not self.goto_xyz("scoop_" + nm, x, y, zc): return False
        self.door(30, tor=200); time.sleep(0.3)
        if not self.goto_xyz("scoop_plunge", x, y, self.pellet - plunge_cm / 100.0): return False
        val = self.door(0, tor=tor_close)
        for k in range(3):
            if val <= 3.6: break
            print(f"  닫힘 {val:.1f}° > 3.6 → 채터링 {k+1}/3"); self.door(8); val = self.door(0)
        self.goto_xyz("scoop_lift8", x, y, self.pellet + 0.08); val2 = self.door(0)
        self.goto_xyz("scoop_travel", x, y, self.travel)
        d = self.read()[5]; ld = self.loads(); print(f"  ▶ 퍼내기 완료: 닫힘 {val:.1f}° → 재닫힘 {val2:.1f}° → 이동 높이에서 {d:.1f}°  부하 {ld}")
        self.last_scoop = dict(close=val, reclose=val2, door_travel=d, loads_travel=ld)
        self.log(ev="scoop_done", **self.last_scoop); return True

    def mass(self, g):
        """직전 scoop 의 적재 질량 기록 → 세션 로그 + 누적 파일(sim 은 누적 제외). n·평균·표본표준편차 출력."""
        row = dict(ev="mass", g=float(g), **self.last_scoop, t=round(time.time(), 2)); self.log(**row)
        if not self.sim:
            with open(MASS_FILE, "a") as f: f.write(json.dumps(row, default=float) + "\n")
        gs = [json.loads(l)["g"] for l in open(MASS_FILE)] if os.path.exists(MASS_FILE) else []
        sd = float(np.std(gs, ddof=1)) if len(gs) > 1 else 0.0
        print(f"  적재 {g:.2f} g{' [sim, 누적 제외]' if self.sim else ''}  누적 n={len(gs)} 평균 {np.mean(gs) if gs else float('nan'):.2f} ± {sd:.2f} g  (설계 17.6)")

    def place(self, base_deg=90.0, r=None, z_cm=None):
        """들고 있는 것을 베이스 회전 base_deg 방향, 반경 r(기본 더미 반경), 높이 z_cm(기본 펠릿면)에 놓는다. 경로 = P1 자세로 당겨 올림 → 30° 씩 회전 → travel 높이로 뻗기 → 5 cm 씩 하강 → open → close → 역순 복귀(base 0, P1)."""
        r = self.pile_xy[0] if r is None else r; z = self.pellet if z_cm is None else self.floor + z_cm / 100.0
        cur = self.read(); b0 = cur[0]
        if not self.goto_q("place_retract", [b0] + list(P1[1:])): return False
        for b in np.arange(b0, base_deg + 1e-6, 30.0 if base_deg > b0 else -30.0)[1:].tolist() + [base_deg]:
            if not self.goto_q(f"place_rot{b:.0f}", [float(b)] + list(P1[1:])): return False
        x, y = r * math.cos(math.radians(base_deg)), r * math.sin(math.radians(base_deg))
        if not self.goto_xyz("place_extend", x, y, self.travel): return False
        zc = self.travel - 0.05
        while zc > z + 0.005:
            if not self.goto_xyz("place_down", x, y, zc): return False
            zc -= 0.05
        if not self.goto_xyz("place_target", x, y, z): return False
        self.door(30, tor=200); time.sleep(1.5); self.door(0)
        if not self.goto_xyz("place_up", x, y, self.travel): return False
        if not self.goto_q("place_retract2", [base_deg] + list(P1[1:])): return False
        for b in np.arange(base_deg, -1e-6, -30.0 if base_deg > 0 else 30.0)[1:].tolist() + [0.0]:
            if not self.goto_q(f"place_rot{b:.0f}", [float(b)] + list(P1[1:])): return False
        self.log(ev="place_done", loads=self.loads()); print("  ▶ 놓기 완료 (P1 자세, base 0)"); return True

    def dump(self):
        if "dump" not in self.pos: print("  ✗ 'dump' 위치가 없음 — 조그로 컵 위에 놓고 save dump"); return False
        p = self.pos["dump"]
        if not self.travel_to(p["x"], p["y"], p["z"], "dump"): return False
        self.door(30, tor=200); time.sleep(1.5); self.door(0)
        return self.goto_xyz("dump_up", p["x"], p["y"], self.travel)

    def cmd(self, line):
        line = line.strip()
        mj = re.match(r"^\s*([xyzXYZ])\s*([+-]?\d+(?:\.\d+)?)\s*$", line)      # "z+10" "y -3" "x 2"
        if mj: line = f"{mj.group(1).lower()} {mj.group(2)}"
        t = line.split()
        if not t: return True
        c = t[0].lower()
        try:
            if c in ("q", "quit", "exit"):
                cl = self.cur_lip()
                if cl[2] < self.travel - 0.005: self.goto_q("quit_up", self._solve_here(self.travel))
                self.goto_q("p1", P1); self.goto_q("home", HOME); return False
            if c == "status":
                q = self.read(); l = self.lip_world(q[:5])
                print(f"  관절 {[round(v,1) for v in q[:5]]}  문 {q[5]:.1f}°  립 x {l[0]*100:.1f} y {l[1]*100:.1f} 바닥+{(l[2]-self.floor)*100:.1f} cm  부하 {self.loads()}\n  펠릿면 바닥+{self.a.pellet_cm} · 상자 윗단 {self.a.boxtop_cm} · travel {self.a.travel_cm} · 저장 위치 {list(self.pos)}")
            elif c == "home": self.goto_q("home", HOME)
            elif c == "p1": self.goto_q("p1", P1)
            elif c == "above":
                q = self.read()
                if max(abs(x - y) for x, y in zip(q[:5], HOME)) < 6: self.goto_q("p1", P1)
                self.travel_to(self.pile_xy[0], self.pile_xy[1], self.travel, "above")
            elif c in ("x", "y", "z"):
                d = float(t[1]) / 100.0; n = max(1, int(math.ceil(abs(d) / 0.05)))       # 5 cm 초과는 자동 분할(각 단계 FK 검사)
                for k in range(n):
                    l = self.cur_lip(); tgt = l.copy(); tgt["xyz".index(c)] += d / n
                    if c != "z" and l[2] < self.boxtop + 0.01 and not self.inside_box_safe(tgt[0], tgt[1]):
                        print(f"  ✗ 립이 상자 윗단 아래({(l[2]-self.floor)*100:.1f} cm)인데 목표가 벽 3 cm 안 — z 로 먼저 {(self.boxtop+0.01-l[2])*100:.0f} cm 이상 올릴 것"); return True
                    if not self.goto_xyz(f"jog_{c}{t[1]}_{k+1}/{n}", *tgt): break
            elif c in ("help", "?"): print(__doc__.split("명령(cm 단위")[1])
            elif c == "open": self.door(30, tor=200)
            elif c == "close": self.door(0, tor=int(t[1]) if len(t) > 1 else 900)
            elif c == "chatter": self.door(8); self.door(0)
            elif c == "scoop": self.scoop(float(t[1]) if len(t) > 1 else 2.5)
            elif c == "save":
                l = self.cur_lip(); self.pos[t[1]] = {"x": float(l[0]), "y": float(l[1]), "z": float(l[2]), "q": [float(v) for v in self.read()[:5]], "t": time.strftime("%Y-%m-%d %H:%M:%S")}
                json.dump(self.pos, open(POS_FILE, "w"), ensure_ascii=False, indent=1); print(f"  저장 {t[1]}: x {l[0]*100:.1f} y {l[1]*100:.1f} 바닥+{(l[2]-self.floor)*100:.1f} → {POS_FILE}")
            elif c == "goto": p = self.pos[t[1]]; self.travel_to(p["x"], p["y"], p["z"], t[1])
            elif c == "list":
                for k, p in self.pos.items(): print(f"  {k:10s} x {p['x']*100:.1f} y {p['y']*100:.1f} 바닥+{(p['z']-self.floor)*100:.1f}  ({p['t']})")
            elif c == "dump": self.dump()
            elif c == "place": self.place(float(t[1]) if len(t) > 1 else 90.0)
            elif c in ("cycle", "weigh"):
                n = int(t[1]) if len(t) > 1 else 1
                for i in range(n):
                    print(f"── {c} {i+1}/{n}")
                    if not (self.cmd("above") and self.scoop() and self.place(self.a.place_deg) and self.cmd("above")): print("  중단"); break
                    if c == "weigh":
                        s = input("  저울 읽기 g (빈 칸 = 미기록): ").strip()
                        if s: self.mass(float(s))
            elif c == "mass": self.mass(float(t[1]))
            else: print("  ? 명령: status home p1 above x/y/z open close chatter scoop save goto list dump cycle weigh mass quit")
        except (KeyError, IndexError, ValueError) as e:
            print(f"  ✗ 입력 오류: {e}")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port"); ap.add_argument("--sim", action="store_true")
    ap.add_argument("--base-cm", type=float, required=True); ap.add_argument("--pellet-cm", type=float, required=True); ap.add_argument("--boxtop-cm", type=float, required=True)
    ap.add_argument("--box-x-cm", type=float, default=35.0); ap.add_argument("--place-deg", type=float, default=90.0, help="cycle 의 놓기 베이스 각(+y = 90)"); ap.add_argument("--travel-cm", type=float, default=45.0, help="이동 높이(바닥 기준). 기울임≤14° 허용 시 r 35 cm 에서 최대 ~50, 하중 처짐 ~2 cm 감안 45 권장")
    ap.add_argument("--script", default=None, help="세미콜론으로 구분한 명령열 (sim 검증용)")
    a = ap.parse_args()
    if not a.sim and not a.port: sys.exit("--port 또는 --sim")
    m = Manual(a)
    print(f"S1 수동 운용 {'[SIM] ' if a.sim else ''}— 펠릿면 {a.pellet_cm} · 윗단 {a.boxtop_cm} · travel {a.travel_cm} cm · 로그 {m.log_path}")
    if a.script:
        for line in a.script.split(";"):
            print(f"> {line.strip()}")
            if not m.cmd(line): break
        return 0
    while True:
        try: line = input("s1> ")
        except (EOFError, KeyboardInterrupt): print(); line = "quit"
        stop = False
        for part in re.split(r"[,;]", line):                    # "y +10, x -3" 처럼 한 줄에 여러 명령 허용
            if part.strip() and not m.cmd(part): stop = True; break
        if stop: break
    return 0


if __name__ == "__main__":
    sys.exit(main())
