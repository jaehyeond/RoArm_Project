"""DEME 스모크 테스트 — CUDA JIT가 이 머신(nvcc 미설치)에서 도는지 판별한다.

DEME는 접촉력 모델을 런타임 JIT 컴파일한다. 툴킷 없이 휠 동봉 런타임만으로 되는지가
시뮬 트랙 전체의 전제이므로, 안식각 사다리(본 실험) 전에 이것부터 확인한다.

판정: 구 N개를 상자에 떨어뜨려 정착시키고 steps/s를 잰다.
사용:  python sim_deme_smoke.py [N]
"""
import sys, time, json
from pathlib import Path
import numpy as np
import DEME

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = Path("claudedocs/runtime_logs/sim_deme"); OUT.mkdir(parents=True, exist_ok=True)

R = 0.0025          # 펠릿 등가 반경 2.5mm → 지름 5mm (d_max)
RHO = 950.0         # PE 진밀도 kg/m3
DT = 2e-5
# 상자는 입자 수에 맞춰 키운다. 간격을 지름보다 좁게 잡으면 초기 관통이 나고
# DEME가 "On average a sphere has N contacts, more than the max allowance"로 죽는다
# (2026-08-27 실측: 고정 60mm 상자 + N=2000 → 간격 4.3mm < 지름 5mm → 106.6 contacts).
SPACING = 2.4 * R   # 지름의 1.2배

t_all = time.time()
_side = int(np.ceil(np.sqrt(N / 4.0)))
_nz = int(np.ceil(N / (_side * _side)))
BOX = max(0.06, (_side + 2) * SPACING)          # 바닥 한 변 (여유 2칸)
BOX_Z = max(0.12, (_nz + 4) * SPACING)          # 낙하 여유 포함 높이

s = DEME.DEMSolver()
s.SetVerbosity("ERROR")
s.SetOutputFormat("CSV")
s.InstructBoxDomainDimension(BOX, BOX, BOX_Z)
s.InstructBoxDomainBoundingBC("all", DEME.DEMMaterial({"E": 1e7, "nu": 0.3, "CoR": 0.3, "mu": 0.5, "Crr": 0.05}))

mat = s.LoadMaterial({"E": 1e7, "nu": 0.3, "CoR": 0.3, "mu": 0.5, "Crr": 0.05})
vol = 4.0 / 3.0 * np.pi * R ** 3
mass = vol * RHO
inertia = 2.0 / 5.0 * mass * R ** 2
tmpl = s.LoadSphereType(mass, R, mat)

# 격자로 흩뿌려 낙하 (겹침 없이) — 바닥 면적을 먼저 잡고 위로 쌓는다
side = int(np.ceil(np.sqrt(N / 4.0)))           # 층당 side x side, 대략 4층 이상
nz = int(np.ceil(N / (side * side)))
xyz = []
for k in range(nz):
    for i in range(side):
        for j in range(side):
            if len(xyz) >= N: break
            xyz.append([(i - side / 2 + 0.5) * SPACING,
                        (j - side / 2 + 0.5) * SPACING,
                        SPACING * (k + 1) + 0.01])
xyz = np.array(xyz[:N], dtype=float)
s.AddClumps(tmpl, xyz)

s.SetInitTimeStep(DT)
s.SetGravitationalAcceleration([0, 0, -9.81])
s.SetCDUpdateFreq(20)

t_init = time.time()
s.Initialize()
init_s = time.time() - t_init
print(f"Initialize (JIT 포함): {init_s:.1f} s")

SIM_T = 0.5
t0 = time.time()
s.DoDynamicsThenSync(SIM_T)
wall = time.time() - t0
steps = int(SIM_T / DT)

res = {
    "n_particles": int(N), "radius_m": R, "dt_s": DT,
    "sim_time_s": SIM_T, "wall_s": round(wall, 2),
    "steps": steps, "steps_per_s": round(steps / wall, 1),
    "realtime_x": round(SIM_T / wall, 3),
    "init_s_incl_jit": round(init_s, 1),
    "total_s": round(time.time() - t_all, 1),
    "gpu": "RTX 4090 Laptop 16GB",
}
print(json.dumps(res, indent=1))
(OUT / f"smoke_N{N}.json").write_text(json.dumps(res, indent=1))
print(f"\n저장: {OUT}/smoke_N{N}.json")
