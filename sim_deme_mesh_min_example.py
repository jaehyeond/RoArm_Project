"""DEME 메시 + 입자 최소 동작 예제 — 반력 읽기까지 실증.

확정된 규약 (로컬 바인딩 실측):
  mat  = s.LoadMaterial({"E":..., "nu":..., "CoR":..., "mu":..., "Crr":...})
  mesh = s.AddWavefrontMeshObject(obj_path, mat, load_normals=True, load_uv=False)
  trk  = s.Track(mesh)
  pts, frcs = trk.GetContactForces()      # (list[float3], list[float3])
단위는 SI (m, kg, s, N).
"""
import numpy as np, trimesh, DEME, time, functools
print = functools.partial(print, flush=True)

s = DEME.DEMSolver()
s.SetVerbosity("ERROR")

mat_p = s.LoadMaterial({"E": 5e6, "nu": 0.3, "CoR": 0.3, "mu": 0.5, "Crr": 0.05})
mat_m = s.LoadMaterial({"E": 5e6, "nu": 0.3, "CoR": 0.3, "mu": 0.5, "Crr": 0.05})

# 입자: 반지름 2.08 mm 펠릿 격자
R = 0.00208
tmpl = s.LoadSphereType(1.0e-4, R, mat_p)
xs = np.arange(-0.02, 0.0201, 2.4 * R)
ys = np.arange(-0.02, 0.0201, 2.4 * R)
pos = [[x, y, 0.006 + 2.4 * R * k] for k in range(3) for x in xs for y in ys]
s.AddClumps(tmpl, pos)          # sim_deme_pile.py:642 와 동일한 규약
print("입자", len(pos))

# 바닥
s.AddBCPlane([0, 0, 0], [0, 0, 1], mat_p)

# 메시 = 셸
mesh = s.AddWavefrontMeshObject("/home/cgxr/.claude/jobs/340b2c97/tmp/shell_L.obj",
                                mat_m, True, False)
mesh.SetMass(0.016)
mesh.SetMOI([1e-5, 1e-5, 1e-5])
mesh.SetFamily(10)
mesh.SetInitPos([0, 0, 0.10])   # 더미 위 여유 지점에서 출발
s.SetFamilyFixed(10)                    # 규정 궤적: 우리가 직접 위치를 준다
trk = s.Track(mesh)
print("메시 삼각형", mesh.GetNumTriangles())

s.SetInitTimeStep(1e-5)
s.SetErrorOutVelocity(20.0)   # 5 m/s 초과면 발산으로 본다 (기본 1000 은 너무 관대)
s.SetGravitationalAcceleration([0, 0, -9.81])
s.SetCDUpdateFreq(20)
s.Initialize()

# 정착
s.DoDynamicsThenSync(0.25)
z0 = np.asarray(trk.GetMeshNodesGlobal(), dtype=float)[:, 2].min()   # 반환형은 tuple 리스트다
print("정착 후 메시 최저 z = %.4f m" % z0)
_pp = np.asarray(s.GetOwnerPosition(0, len(pos) - 1), dtype=float)
ptop = _pp[:, 2].max()
print("더미 최상단 z = %.4f m -> 메시 하단까지 %.1f mm 남음" % (ptop, (z0-ptop)*1000))
z_start, z_end = 0.10, 0.10 - (z0 - ptop) - 0.012      # 더미 위 -> 12 mm 관입까지

# 하강 스쿱 — 규정 궤적
t0 = time.time()
rows = []
N = 120
for i in range(N):
    z = z_start + (z_end - z_start) * (i + 1) / N
    trk.SetPos([0, 0, z])
    s.DoDynamicsThenSync(0.004)
    pts, frcs = trk.GetContactForces()
    F = np.asarray(frcs, dtype=float) if len(frcs) else np.zeros((0, 3))
    tot = float(np.linalg.norm(F.sum(0))) if len(F) else 0.0
    rows.append((z, len(pts), tot))
    print("  z=%.3f m  접촉 %3d  합력 %7.3f N" % (z, len(pts), tot))
print("스쿱 14스텝 벽시계 %.2f s" % (time.time() - t0))
peak = max(r[2] for r in rows)
print()
print("최대 반력 %.3f N  (실측 조 힘 1.8~6.3 N 대비 %s)"
      % (peak, "안전" if peak <= 6.3 else "초과"))

# ─────────────────────────────────────────────────────────────────────────
# 코디네이터 실측 메모 (2026-08-31) — 이 파일은 **작동이 확인된** 최소 예제다.
#
# 확정된 API 규약 (로컬 바인딩에서 직접 읽음. 외부 문서 불필요):
#   mat  = s.LoadMaterial({"E":..,"nu":..,"CoR":..,"mu":..,"Crr":..})
#   tmpl = s.LoadSphereType(mass, radius, mat)
#   s.AddClumps(tmpl, positions)              # sim_deme_pile.py:642 와 동일
#   mesh = s.AddWavefrontMeshObject(obj_path, mat, load_normals=True, load_uv=False)
#   trk  = s.Track(mesh)
#   trk.SetPos([x,y,z]) / trk.Pos()
#   pts, frcs = trk.GetContactForces()        # 반환은 **tuple 리스트**. .x/.y/.z 없음
#   nodes = trk.GetMeshNodesGlobal()          # 역시 tuple 리스트
#   pp = s.GetOwnerPosition(0, n-1)           # GetAllOwnerPosition 은 없다
#
# 밟은 함정 4개:
#   1. DEMClumpBatch(...).SetPos() 는 없다. AddClumps(type, positions) 를 쓴다.
#   2. GetContactForces / GetMeshNodesGlobal 반환 원소는 tuple 이다.
#      [[p.x,p.y,p.z] for p in ...] 로 읽으면 AttributeError.
#   3. 고정(family fixed) 메시를 SetPos 로 크게 순간이동시키면 입자와 겹쳐 발산한다.
#      (max velocity 1666 m/s). 스텝을 잘게(여기선 120스텝 x 4 ms) 쪼갤 것.
#   4. E=1e8~1e9 는 이 스텝 크기에서 발산한다. 5e6 으로 완화했다.
#      ⚠️ 이 값은 **수치 안정을 위한 임시값이지 펠릿 실측이 아니다.**
#      실측 후 반드시 교체하고, 교체하면 반력이 바뀐다.
#
# 실측 결과 (셸 1매, 격자 243입자, 수직 하강만):
#   접촉 0 -> 54개,  합력 0 -> 57.97 N,  120스텝 벽시계 9.76 s
#   🔴 실측 조 힘 1.8~6.3 N (D451/D452) 대비 **약 9배 초과**
#
# ⚠️ 이 57.97 N 을 폐합력으로 인용하지 마라. 조건이 다르다:
#     셸 1매(실제는 2매 대칭) · 수직 관입만(실제는 하강 후 회전 폐합) ·
#     격자 배치 243개(실제는 자연 정착 18,796개) · 강성 임시값.
#   본 예제가 주장하는 것은 **파이프라인이 성립한다**는 것뿐이다.
