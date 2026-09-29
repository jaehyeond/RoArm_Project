"""W25-A CPU 드라이런 — 러너의 **제어 흐름**을 DEME 없이 돌리는 운동학 스텁 (물리 0, GPU 0).

무엇인가
    `sim_w13_full_cycle.run()` 은 함수 안에서 `import DEME` 한다. 이 도구는 그 전에 sys.modules["DEME"] 에
    **가짜 모듈**을 꽂는다. 가짜 솔버는 접촉·중력·입자 운동을 계산하지 않는다:
      · 입자 = 더미 npz 자세 그대로 정지(속도 0)
      · 툴/문 트래커 = 명령 속도를 그대로 적분(p += v·dts, R = R·exp(ω_local·dts))
      · 접촉력 = 0. 단 --door-stall 시나리오를 주면 **문 닫힘 중** 지정 각 아래에서 힌지 저항 모멘트
        1.2×M_stall 을 주입한다(채터링 분기를 태우기 위한 인위 입력 — 물리 결과 아님).
    그래서 나오는 것은 sync 격자·phase/subphase 열·명령 목표·결정 시점·도메인·bridge 인증 입력이다.
    포획량·배출량·문 정지각 같은 물리 수치는 **의미가 없다**.

usage
    python cpu_stub_dryrun.py --src <rev dir>/src --params P.json --pile PILE.npz --out OUT
                              [--numeric-evidence NE.json] [--door-stall 3.2,2.0,0.5]
"""
import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import trimesh


def _axis_angle(w, dt):
    th = float(np.linalg.norm(w)) * dt
    if th < 1e-300:
        return np.eye(3)
    a = np.asarray(w, float) / np.linalg.norm(w)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], float)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _q2m(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _m2q(m):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(m).as_quat()          # xyzw


class _Any:
    def __getattr__(self, _):
        return lambda *a, **k: self


class _Mesh:
    def __init__(self, path, role):
        self.v = np.asarray(trimesh.load(path, process=False).vertices, float)
        self.role, self.p, self.R = role, np.zeros(3), np.eye(3)

    def SetMass(self, *_): pass
    def SetMOI(self, *_): pass
    def SetFamily(self, *_): pass
    def SetInitPos(self, p): self.p = np.asarray(p, float)
    def SetInitQuat(self, q): self.R = _q2m(q)


class _Tracker:
    def __init__(self, mesh, owner, solver):
        self.m, self.owner, self.s = mesh, owner, solver
        self.v, self.w = np.zeros(3), np.zeros(3)

    def Pos(self): return self.m.p.tolist()
    def OriQ(self): return _m2q(self.m.R).tolist()
    def SetVel(self, v): self.v = np.asarray(v, float)
    def SetAngVel(self, w): self.w = np.asarray(w, float)
    def Vel(self): return self.v.tolist()
    def AngVelLocal(self): return self.w.tolist()
    def GetOwnerID(self): return self.owner
    def GetMeshNodesGlobal(self): return ((self.m.R @ self.m.v.T).T + self.m.p).tolist()
    def GetContactForces(self): return self.s._contacts(self.m.role)


class _Clumps:
    def __init__(self, solver, pos):
        solver._pos = np.asarray(pos, float)
        self.s = solver

    def SetOriQ(self, q): self.s._quat = np.asarray(q, float)


class StubSolver:
    """DEME.DEMSolver 의 **이름만** 흉내 낸다. 물리 계산 0."""
    scenario = {}

    def __init__(self):
        self._t, self._meshes, self._trk, self._pos, self._quat = 0.0, [], [], None, None
        self.domain = None
        self._stall = list(StubSolver.scenario.get("door_stall_q_deg") or [])
        self._ep, self._closing, self._q_prev = -1, False, None
        self._M_stall = StubSolver.scenario.get("M_stall_Nm")
        self._axis_owner = np.array([-1.0, 0.0, 0.0])      # R_W @ (0,1,0) — 러너 axis_w 와 같은 값(아래 검사)
        self._q_open = StubSolver.scenario.get("q_open_deg")
        self.stall_events = []

    def __getattr__(self, name):                           # SetVerbosity·Instruct*·SetFamily* 등 설정 호출은 무시
        return lambda *a, **k: _Any()

    def InstructBoxDomainDimension(self, x, y, z): self.domain = (list(x), list(y), list(z))
    def LoadMaterial(self, *_): return _Any()
    def LoadClumpType(self, *a, **k): return _Any()
    def LoadSphereType(self, *a, **k): return _Any()
    def AddClumps(self, _ct, pos): return _Clumps(self, pos)

    def AddWavefrontMeshObject(self, path, *_):
        role = ["fixed", "door", "tray", "bin"][len(self._meshes)]
        m = _Mesh(path, role)
        self._meshes.append(m)
        return m

    def Track(self, mesh):
        t = _Tracker(mesh, 20001 + len(self._trk), self)
        self._trk.append(t)
        return t

    def Initialize(self): pass
    def GetSimTime(self): return self._t
    def GetNumContacts(self): return 0
    def GetUpdateFreq(self): return 20.0
    def GetOwnerPosition(self, i0, n): return self._pos[i0:i0 + n].tolist()
    def GetOwnerVelocity(self, i0, n): return np.zeros((n, 3)).tolist()
    def GetOwnerAngVel(self, i0, n): return np.zeros((n, 3)).tolist()
    def GetOwnerOriQ(self, i0, n): return self._quat[i0:i0 + n].tolist()

    def _door_q(self):
        f, d = self._trk[0].m, self._trk[1].m
        rel = f.R.T @ d.R
        c = min(1.0, max(-1.0, (np.trace(rel) - 1) / 2))
        ang = math.degrees(math.acos(c))
        if ang < 1e-9:
            return self._q_open
        w = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
        return self._q_open + (1.0 if float(np.dot(w, self._axis_owner)) >= 0 else -1.0) * ang

    def DoDynamicsThenSync(self, dts):
        for t in self._trk[:2]:
            t.m.p = t.m.p + t.v * dts
            t.m.R = t.m.R @ _axis_angle(t.w, dts)
        self._t += dts
        if self._q_open is not None:
            q = self._door_q()
            if self._q_prev is not None:
                # 이번 step 에 실제로 닫혔는가. 문턱 1e-6° — 팔 이동 중 두 트래커 적분 잡음(~1e-10°)을
                # 닫힘으로 세지 않기 위함(실제 최소 닫힘 step = 22.5°/s × 0.1 ms = 0.00225°).
                closing = q < self._q_prev - 1e-6
                if closing and not self._closing:           # 정지·열림 뒤 새 닫힘 = 새 에피소드
                    self._ep += 1
                self._closing = closing
            self._q_prev = q

    def _contacts(self, role):
        if role != "door" or not self._closing or self._M_stall is None:
            return [], []
        if not (0 <= self._ep < len(self._stall)) or self._q_prev is None or self._q_prev >= self._stall[self._ep]:
            return [], []
        f, d = self._trk[0].m, self._trk[1].m
        a = f.R @ self._axis_owner
        u = np.cross(a, [0.0, 0.0, 1.0])
        if np.linalg.norm(u) < 1e-9:
            u = np.cross(a, [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        r = 0.02
        F = 1.2 * self._M_stall / r
        self.stall_events.append({"episode": self._ep, "q_deg": self._q_prev, "t": self._t})
        return [(d.p + r * u).tolist()], [(F * np.cross(a, u)).tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--pile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--numeric-evidence")
    ap.add_argument("--door-stall", help="닫힘 에피소드별 정지 관절각(°) 쉼표 목록 — 인위 주입")
    a = ap.parse_args()

    src = Path(a.src).resolve()
    sys.path.insert(0, str(src))
    stub = types.ModuleType("DEME")
    stub.DEMSolver = StubSolver
    stub.__version__ = "CPU_STUB_NOT_DEME"
    sys.modules["DEME"] = stub
    sys.modules["deme"] = stub
    import sim_w13_full_cycle as RUN                         # noqa: E402
    import sim_deme_scoop_s1 as W11SRC                       # noqa: E402
    assert np.array_equal(W11SRC.R_W @ np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])), \
        "axis_w 가 스텁 가정과 다르다"
    P = dict(W11SRC.DEFAULT)
    P.update(json.load(open(a.params)))
    q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
    StubSolver.scenario = {"door_stall_q_deg": [float(v) for v in a.door_stall.split(",")] if a.door_stall else [],
                           "M_stall_Nm": P["servo_torque_Nm"] * P["servo_torque_fraction"],
                           "q_open_deg": q_open}
    ns = argparse.Namespace(params=a.params, pile=a.pile, out=a.out, seed=460, smoke=False, stop_after_phase=None,
                            max_particles=None, numeric_evidence=a.numeric_evidence, max_wall_s=None)
    t0 = time.time()
    RUN.run(ns)
    out = Path(a.out)
    json.dump({"artifact": "W25A_CPU_STUB_DRYRUN_RECEIPT", "src": str(src), "params": a.params, "pile": a.pile,
               "numeric_evidence": a.numeric_evidence, "door_stall_q_deg": StubSolver.scenario["door_stall_q_deg"],
               "wall_s": round(time.time() - t0, 2), "engine": "CPU_STUB_NOT_DEME",
               "non_claims": ["DEME 솔버를 만들지 않았다(GPU 0·물리 0).", "입자는 정지, 접촉 0(주입 시나리오 제외).",
                              "제어 흐름·sync 격자·기하 입력만 의미가 있다."]},
              open(out / "STUB_RECEIPT.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
