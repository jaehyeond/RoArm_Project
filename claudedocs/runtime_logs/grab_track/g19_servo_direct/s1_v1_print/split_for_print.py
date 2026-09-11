"""S1 출력용 분할·배향 — 문은 Y=0 평면으로 두 쪽(측판 바깥면 바닥). 고정부는 **분할 없이** 판 바깥면(X=−15.54)을 바닥에 놓는다(판·스파인 접지, 벽은 아치, 캡은 세로 지느러미; 09-04 배향 수정).
설계 형상(s1_v0)은 불변. 이 스크립트는 설계 모듈에서 조각을 다시 만들어 평면으로 자른다(볼록 조각별 slice, cap=True).
출력: door_L.stl door_R.stl fixed_L.stl fixed_R.stl (mm, 베드 좌표: Z≥0, XY 양수) + split_report.json + BED_PREVIEW.png
"""
import sys, os, json, hashlib, math
import numpy as np, trimesh
sys.argv = ["x"]; sys.path.insert(0, "/home/cgxr/Documents/Robotics/RoArm_Project")
import scoop_grab_s1_design as S
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
P = S.P

def slice_keep(m, y_cut, side):
    """side +1: Y>y_cut 보존, −1: Y<y_cut 보존. 닫힌 조각 하나에 대해."""
    n = np.array([0.0, side, 0.0]); o = np.array([0.0, y_cut, 0.0])
    lo, hi = m.bounds[0][1], m.bounds[1][1]
    if side > 0 and lo >= y_cut - 1e-9: return m.copy()
    if side < 0 and hi <= y_cut + 1e-9: return m.copy()
    if side > 0 and hi <= y_cut + 1e-9: return None
    if side < 0 and lo >= y_cut - 1e-9: return None
    s = trimesh.intersections.slice_mesh_plane(m, plane_normal=n, plane_origin=o, cap=True)
    return s if (s is not None and len(s.faces) > 0) else None

def split_body(meshes, y_cut):
    halves = {+1: [], -1: []}
    for m in meshes:
        for comp in (m.split(only_watertight=False) if not m.is_watertight else [m]):
            if comp.volume < 1e-6 and not comp.is_watertight: pass
            for side in (+1, -1):
                s = slice_keep(comp, y_cut, side)
                if s is not None: halves[side].append(s)
    return {k: trimesh.util.concatenate(v) for k, v in halves.items()}

def to_bed(m, outer_y, side):
    """바깥면(Y=outer_y)이 바닥. side −1(−Y 쪽): (X,Y,Z)→(X,−Z,Y) 뒤 Z+=−outer_y ; side +1: (X,Y,Z)→(X,Z,−Y) 뒤 Z+=outer_y"""
    v = m.vertices.copy()
    if side < 0: nv = np.stack([v[:, 0], -v[:, 2], v[:, 1] - outer_y], 1)
    else:        nv = np.stack([v[:, 0], v[:, 2], outer_y - v[:, 1]], 1)
    out = trimesh.Trimesh(vertices=nv, faces=m.faces.copy(), process=False)
    if out.volume < 0: out.invert()
    out.apply_translation([-out.bounds[0][0] + 5.0, -out.bounds[0][1] + 5.0, -out.bounds[0][2]])
    return out

def to_bed_plate_down(m, x_outer):
    """판 바깥면(X=x_outer)이 바닥. (X,Y,Z)→(Z, −Y, X−x_outer) (det +1)"""
    v = m.vertices.copy(); nv = np.stack([v[:, 2], -v[:, 1], v[:, 0] - x_outer], 1)
    out = trimesh.Trimesh(vertices=nv, faces=m.faces.copy(), process=False)
    if out.volume < 0: out.invert()
    out.apply_translation([-out.bounds[0][0] + 5.0, -out.bounds[0][1] + 5.0, -out.bounds[0][2]]); return out

def overhang_area(m, tol=0.3, cos45=0.7071):
    """바닥에 닿지 않은 아래보기 면 중 45° 보다 완만한 것의 면적 (orient_for_print 와 같은 정의)"""
    c = m.triangles_center; n = m.face_normals; zmin = m.bounds[0][2]
    sel = (n[:, 2] < -cos45) & (c[:, 2] > zmin + tol); return float(m.area_faces[sel].sum())

def contact_area(m, tol=0.3):
    zmin = m.bounds[0][2]; c = m.triangles_center; n = m.face_normals
    sel = (c[:, 2] < zmin + tol) & (n[:, 2] < -0.7); return float(m.area_faces[sel].sum())

def sha16(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

door_p, door_n, door_vis, _ = S.build_door(P); fixed_p, fixed_n, fixed_vis = S.build_fixed(P)
t = P["plate_t"]; y0, y1 = P["cheek_inner_y"]
parts = {}
dh = split_body(door_vis, 0.0);  parts["door_L"] = to_bed(dh[-1], y0 - t, -1); parts["door_R"] = to_bed(dh[+1], y1 + t, +1)
parts["fixed"] = to_bed_plate_down(trimesh.util.concatenate(fixed_vis), P["fixed_plate_x"][0])
rho = P["density_g_cm3"] / 1000.0
rep = {"design_json_sha16": sha16("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/design.json"),
       "split": {"door": "Y=0 두 쪽", "fixed": "분할 없음"}, "orientation": {"door": "측판 바깥면 바닥, 벽·다리가 수직으로 섬", "fixed": "판 바깥면 바닥(구멍이 수직), 벽은 스파인에서 솟는 아치(스파인 밖 구간 Z 130~136·154~160 만 25~45° 오버행), 캡은 세로 지느러미"},
       "join": {"door": "다리 각구멍 3.5 → M3×45 타이볼트 + 너트 (측판 관통 ⌀3.4) + 접착", "fixed": "없음 (한 덩어리)"}, "parts": {}}
for k, m in parts.items():
    fn = OUT + k + ".stl"; m.export(fn)
    ext = m.extents; rep["parts"][k] = {"file": k + ".stl", "sha256_16": sha16(fn), "bbox_mm": [round(float(v), 2) for v in ext], "height_mm": round(float(ext[2]), 2),
                                        "contact_mm2": round(contact_area(m), 1), "overhang_mm2_gt45": round(overhang_area(m), 1), "volume_cm3": round(float(abs(m.volume)) / 1000.0, 2), "mass_g_pla": round(float(abs(m.volume)) * rho, 2), "watertight": bool(m.is_watertight), "faces": len(m.faces)}
    print(k, rep["parts"][k])
json.dump(rep, open(OUT + "split_report.json", "w"), ensure_ascii=False, indent=1)
# 미리보기: 각 쪽 위에서 본 것 + 측면
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
_fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; fm.fontManager.addfont(_fp); plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
fig, axs = plt.subplots(2, len(parts), figsize=(5 * len(parts), 9))
for j, (k, m) in enumerate(parts.items()):
    ax = axs[0, j]; sec = m.section(plane_origin=[0, 0, 0.15], plane_normal=[0, 0, 1])
    if sec is not None:
        for d in sec.discrete: ax.plot(d[:, 0], d[:, 1], "-", color="tab:blue", lw=1)
    top = m.section(plane_origin=[0, 0, m.bounds[1][2] - 0.5], plane_normal=[0, 0, 1])
    if top is not None:
        for d in top.discrete: ax.plot(d[:, 0], d[:, 1], "-", color="tab:red", lw=0.8)
    ax.set_aspect("equal"); ax.set_title(f"{k} 위에서 (파랑 = 1층 접지 {rep['parts'][k]['contact_mm2']:.0f} mm², 빨강 = 최상층)"); ax.grid(alpha=.3)
    ax = axs[1, j]; sec = m.section(plane_origin=[0, m.bounds[0][1] + m.extents[1] / 2, 0], plane_normal=[0, 1, 0])
    if sec is not None:
        for d in sec.discrete: ax.plot(d[:, 0], d[:, 2], "-", color="k", lw=1)
    ax.set_aspect("equal"); ax.set_title(f"{k} 측면 단면 (높이 {rep['parts'][k]['height_mm']} mm, {rep['parts'][k]['mass_g_pla']} g, 오버행>45° {rep['parts'][k]['overhang_mm2_gt45']:.0f} mm²)"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + "BED_PREVIEW.png", dpi=100); print("saved")
