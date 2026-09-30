"""카메라 렌더 자체 검사: (1) 빈 상자 → 바닥 칸 높이 0 (2) 알 한 개 → 그 칸 높이 = 구 윗면 (3) 규약 A 카메라가 로봇 반대(+y)에 있고 로봇 쪽(−y)을 보는지."""
import sys, numpy as np
sys.path.insert(0, 'chain'); import site_map as SM; import cam_render as CR
from roarm_rl.heightmap import GridSpec, heightmap_from_particles
P = SM.load_params('params/params_cell_V0_reference.json')
box = np.array([[-0.155, 0.155], [-0.11, 0.11], [0.0, 0.23]])
tray = SM.K.tray_mesh(box, 0.002); tris = np.asarray(tray.vertices)[np.asarray(tray.faces)]
spec = GridSpec(origin_xy_m=(-0.155, -0.11), cell_m=0.005, shape=(44, 62), frame="deme_box_floor_center", z_datum_m=0.0)
T = CR.camera_T("A"); print("cam pos A", T[:3, 3].round(4).tolist(), "optical axis", T[:3, 2].round(4).tolist())
D0 = CR.render_depth(np.zeros((0, 3)), np.zeros(0), tris, T)
hm0, _ = CR.heightmap_from_render(D0, T, spec)
h0 = np.asarray(hm0.height); v0 = np.asarray(hm0.valid)
print("empty: valid cells", int(v0.sum()), "/", v0.size, "max |h| on valid", float(np.abs(h0[v0]).max()))
c = np.array([[0.0125, 0.0125, 0.03]]); r = np.array([0.004])
D1 = CR.render_depth(c, r, tris, T); hm1, _ = CR.heightmap_from_render(D1, T, spec)
row, col, _ = spec.index_of(np.array([0.0125]), np.array([0.0125]))
print("ball: cell h", float(np.asarray(hm1.height)[row[0], col[0]]), "expected ~", 0.034,
      "truth", float(np.asarray(heightmap_from_particles(c, r, spec).height)[row[0], col[0]]))
