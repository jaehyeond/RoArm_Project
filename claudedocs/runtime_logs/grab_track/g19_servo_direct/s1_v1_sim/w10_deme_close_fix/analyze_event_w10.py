"""이벤트 링버퍼(diverge_event*_seed460.npz) 오프라인 분석 — pop 연쇄의 첫 움직임(first mover) 과 그 클럼프의 구별 툴 삼각형 기하.
usage: python analyze_event_w10.py <cell_dir> [event_suffix=""] [clump_ids=auto]
프레임마다: 근방 집합 속도 상위 3 · 지정 클럼프의 (중심, 속도, 메시별 최근접 삼각형 그룹·h·안·기하 관입·유령, 구 표면 ≤0.5 mm 툴 접촉 힘)."""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
cell = Path(sys.argv[1]); suf = sys.argv[2] if len(sys.argv) > 2 else ""
z = np.load(cell / f"diverge_event{suf}_seed460.npz", allow_pickle=True); ev = json.load(open(cell / f"diverge_event{suf}_seed460.json"))
tpl = json.loads(str(z["template_json"])); offs = np.asarray(tpl["offsets_m"]); radii = np.asarray(tpl["sphere_radii_m"])
T = len(z["t_s"]); fr = z["near_frame"]; ids = z["near_ids"]; pos = z["near_pos_m"]; quat = z["near_quat_xyzw"]; vel = z["near_vel_m_s"]
cp, cf, cfr, cm = z["contact_point_m"], z["contact_force_N"], z["contact_frame"], z["contact_mesh"]
gF, gD = z["groups_F"], z["groups_D"]
def mesh(k, which):
    return trimesh.Trimesh(z["nodes_F_m"][k].astype(float) if which == 0 else z["nodes_D_m"][k].astype(float), z["faces_F"] if which == 0 else z["faces_D"], process=False)
def svm(c, r, m, grp):
    cpt, dist, tid = trimesh.proximity.closest_point(m, c[None]); tid = int(tid[0]); n = m.face_normals[tid]; v0 = m.vertices[m.faces[tid][0]]
    h = float(np.dot(c - v0, n)); inside = bool(np.linalg.norm((c - h * n) - cpt[0]) < 1e-7); d = float(dist[0])
    return dict(group=str(grp[tid]), tri=tid, h_mm=round(h * 1000, 3), inside=inside, pen_mm=round((r - d) * 1000, 3), ghost=bool(inside and h < 0 and abs(h) < r))
clumps = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
if clumps is None:                                   # auto: culprit + 각 프레임 최고속도 owner 들
    clumps = sorted({int(ev["culprit"])} | {int(ids[fr == k][np.argmax(np.linalg.norm(vel[fr == k], axis=1))]) for k in range(T) if (fr == k).any()})
print(f"프레임 {T} · 추적 클럼프 {clumps} · culprit {ev['culprit']} · kind {ev.get('kind')}")
print("\n### 프레임별 근방 집합 속도 상위 3 (id:v m/s)\n\n| k | t s | q ° | top3 |\n|---|---|---|---|")
for k in range(T):
    m = fr == k; v = np.linalg.norm(vel[m], axis=1); o = np.argsort(-v)[:3]
    print(f"| {k - T + 1} | {z['t_s'][k]:.5f} | {z['q_deg'][k]:.4f} | " + ", ".join(f"{int(ids[m][j])}:{v[j]:.3f}" for j in o) + " |")
for cid in clumps:
    print(f"\n### 클럼프 {cid}\n\n| k | t s | 중심(립 기준) mm | v m/s | vel | fixed 최근접(그룹·h·안·관입·유령) | door 최근접 | 툴 접촉 힘 F/D N (n) |\n|---|---|---|---|---|---|---|---|")
    for k in range(T):
        m = fr == k; j = np.where(ids[m] == cid)[0]
        if not len(j):
            print(f"| {k - T + 1} | {z['t_s'][k]:.5f} | (근방 밖) | | | | | |"); continue
        j = int(j[0]); c = pos[m][j]; q_ = quat[m][j]; v = vel[m][j]
        sp = (Rotation.from_quat(q_).as_matrix() @ offs.T).T + c   # (7,3)
        best = []
        for which, grp in ((0, gF), (1, gD)):
            mm = mesh(k, which); per = [svm(sp[i], float(radii[i]), mm, grp) for i in range(len(radii))]
            b = max(per, key=lambda g: (g["ghost"], g["pen_mm"])); best.append(f"{b['group']} · {b['h_mm']} · {'안' if b['inside'] else '밖'} · {b['pen_mm']} · {'👻' if b['ghost'] else '-'}")
        fc = []
        for which in (0, 1):
            sel = (cfr == k) & (cm == which)
            if sel.any():
                d = cKDTree(sp).query(cp[sel].astype(float))[0]; near = d <= radii.max() + 0.0005
                fc.append(f"{np.linalg.norm(cf[sel][near].astype(float), axis=1).max():.3f}({int(near.sum())})" if near.any() else "0(0)")
            else:
                fc.append("0(0)")
        lip = z["lip_m"][k]
        print(f"| {k - T + 1} | {z['t_s'][k]:.5f} | {((c - lip) * 1000).round(2).tolist()} | {np.linalg.norm(v):.3f} | {v.round(3).tolist()} | {best[0]} | {best[1]} | {fc[0]} / {fc[1]} |")
