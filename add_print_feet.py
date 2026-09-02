"""출력용 STL 에 **희생 발**(sacrificial feet)을 붙인다. 출력 후 잘라낸다.

왜 필요한가:
  2026-09-02 브래킷 — 바닥 접지가 **40.8 mm²** 뿐이다. 브림을 15 mm 로 키우면 게이트
  (`first_layer_contact_per_gram`)는 통과하지만 그 점수의 **97% 가 브림**이고, 브림은
  부품 접지보다 먼저 뜯긴다(그 게이트 자신의 `blind_spot` 에 적혀 있다).
  2026-09-02 링크 — 설계 배향의 접지가 **13.8 mm²** 라서 배향기가 접지 11.2배를 얻으려고
  힌지축을 **9.46° 기울였다.** 그 축에는 기어·핀·로드아이가 전부 동축이라 기울이면
  보어가 타원이 되고 이빨이 층으로 쌓인다. 발이 있으면 **축을 버릴 이유가 없다.**

🔴 설계 스크립트(`scoop_grab_v1_design.py`)는 건드리지 않는다.
   발을 설계에 넣으면 `self_load_ratio` 같은 **자중 게이트가 오염**된다. 발은 잘라낼
   것이므로 툴 무게가 아니다. 출력용 STL 에만 붙이고 설계 형상·게이트는 불변이다.

⚠️ 발은 부품과 **겹치게** 놓는다(불리언 union 하지 않는다). 슬라이서가 겹친 솔리드를
   합쳐 처리하며, D446 의 "불리언이 오히려 충돌한다"를 피한다.

접지 정의는 `orient_for_print.py` 와 **같은 것**을 쓴다(중심 z < zmin+0.3 이고 법선 z < -0.7).
정의가 갈리면 같은 부품에 두 숫자가 생긴다.

사용:
    python add_print_feet.py <입력STL> <출력디렉터리> [--pad-half 10] [--height 0.6]
"""
import sys, json, math, argparse, hashlib
from pathlib import Path
import numpy as np
import trimesh

TOL_MM, NZ_DOWN = 0.3, -0.7          # orient_for_print.py 와 동일

ap = argparse.ArgumentParser()
ap.add_argument("stl")
ap.add_argument("outdir")
ap.add_argument("--height", type=float, default=0.6, help="발 두께 mm (0.2 층 기준 3층)")
ap.add_argument("--pad-half", type=float, default=10.0, help="덩어리마다 붙일 발의 반폭 mm")
ap.add_argument("--cluster-mm", type=float, default=8.0, help="이 거리 안이면 같은 덩어리")
ap.add_argument("--no-bridge", action="store_true", help="덩어리 사이 연결 다리를 놓지 않는다")
a = ap.parse_args()

src, out = Path(a.stl), Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
m = trimesh.load(src)
z_shift = -float(m.bounds[0, 2])
if abs(z_shift) > 1e-9:
    m.apply_translation([0, 0, z_shift])      # 회전 없이 바닥만 z=0 으로

# 1) 접지 삼각형 — 형상에서 읽는다. 다리 개수를 가정하지 않는다.
zmin = m.vertices[:, 2].min()
cen = m.triangles.mean(axis=1)
sel = (cen[:, 2] < zmin + TOL_MM) & (m.face_normals[:, 2] < NZ_DOWN)
area0 = float(m.area_faces[sel].sum())
if area0 <= 0:
    sys.exit("바닥 접촉면이 없다 — 이 STL 은 바닥에 놓인 배향이 아니다")
cxy, aw = cen[sel][:, :2], m.area_faces[sel]

# 2) 격자 기반 덩어리 묶기 (외부 의존 없이)
cell = {}
for (x, y), w in zip(cxy, aw):
    cell.setdefault((int(math.floor(x / a.cluster_mm)), int(math.floor(y / a.cluster_mm))),
                    []).append((x, y, w))
seen, clusters = set(), []
for key in list(cell):
    if key in seen:
        continue
    stack, group = [key], []
    seen.add(key)
    while stack:
        k = stack.pop()
        group += cell.get(k, [])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (k[0] + dx, k[1] + dy)
                if n in cell and n not in seen:
                    seen.add(n)
                    stack.append(n)
    g = np.array(group)
    w = g[:, 2]
    clusters.append({"centroid": [round(float((g[:, 0] * w).sum() / w.sum()), 2),
                                  round(float((g[:, 1] * w).sum() / w.sum()), 2)],
                     "area_mm2": round(float(w.sum()), 2)})
clusters.sort(key=lambda c: -c["area_mm2"])

h, ph = a.height, a.pad_half
solids, plan = [m], []


def add_box(ext, tf, tag, area):
    b = trimesh.creation.box(extents=ext)
    b.apply_transform(tf)
    solids.append(b)
    plan.append({"tag": tag, "area_mm2": round(area, 1)})


def clear_below(poly_pts, limit):
    """그 XY 영역 안에서 부품이 limit 아래로 내려오는지 — 내려오면 다리를 놓지 않는다."""
    v = m.vertices
    xs, ys = poly_pts[:, 0], poly_pts[:, 1]
    inb = ((v[:, 0] > xs.min()) & (v[:, 0] < xs.max()) &
           (v[:, 1] > ys.min()) & (v[:, 1] < ys.max()))
    return (float(v[inb][:, 2].min()) if inb.any() else 1e9)


for i, c in enumerate(clusters):
    tf = np.eye(4)
    tf[:3, 3] = [c["centroid"][0], c["centroid"][1], h / 2]
    add_box([2 * ph, 2 * ph, h], tf, f"pad_{i}@{c['centroid']}", (2 * ph) ** 2)

# 3) 덩어리를 잇는 다리 — 바닥판을 하나의 강체로 만든다. 대각선이면 회전 박스를 쓴다.
bridge_notes = []
if not a.no_bridge and len(clusters) >= 2:
    for i in range(len(clusters) - 1):
        p0 = np.array(clusters[i]["centroid"], float)
        p1 = np.array(clusters[i + 1]["centroid"], float)
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 2 * ph:
            bridge_notes.append(f"{i}-{i+1}: 발끼리 이미 겹침 (거리 {L:.1f} < {2*ph})")
            continue
        corner = np.array([p0, p1, p0, p1])
        low = clear_below(corner, h)
        if low <= h + 1.0:
            bridge_notes.append(f"{i}-{i+1}: 경로 최저 z={low:.2f} — 충돌 위험, 생략")
            continue
        W = 6.0
        tf = trimesh.transformations.rotation_matrix(math.atan2(d[1], d[0]), [0, 0, 1])
        tf[:3, 3] = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, h / 2]
        add_box([L, W, h], tf, f"bridge_{i}_{i+1}", L * W)
        bridge_notes.append(f"{i}-{i+1}: 길이 {L:.1f} x {W} mm 연결 (경로 최저 z={low:.2f})")

merged = trimesh.util.concatenate(solids)
dst = out / (src.stem.replace("_ALL", "") + "_feet_ALL.stl")
merged.export(dst)

added = sum(p["area_mm2"] for p in plan)
rec = {
    "artifact": "print_feet/2",
    "source_stl": str(src), "source_sha256_16": hashlib.sha256(src.read_bytes()).hexdigest()[:16],
    "output_stl": str(dst), "output_sha256_16": hashlib.sha256(dst.read_bytes()).hexdigest()[:16],
    "z_shift_applied_mm": round(z_shift, 3), "rotation_applied": "없음 (배향 불변)",
    "foot_height_mm": h, "pad_half_mm": ph,
    "contact_definition": "orient_for_print.py 와 동일 (중심 z < zmin+0.3, 법선 z < -0.7)",
    "original_contact_mm2": round(area0, 1),
    "clusters_found": clusters,
    "pads": plan, "bridges": bridge_notes,
    "added_footprint_mm2": round(added, 1),
    "added_volume_mm3": round(added * h, 1),
    "added_mass_g_pla_1p24": round(added * h * 1.24e-3, 3),
    "non_claims": [
        "발은 **출력 보조물**이며 잘라낸다. 설계 자중·간섭 게이트에 넣지 말 것.",
        "겹친 솔리드를 슬라이서가 합치는 것에 의존한다 — 불리언 union 을 하지 않았다(D446).",
        "여기 적힌 면적은 **계획값**이다. 실제 1층 접지는 gcode 를 적분한 값으로 확인할 것.",
        "발을 떼어낸 자국이 바닥면에 남는다. 바닥이 기능 정합면이면 이 방식을 쓰지 말 것.",
        "다리 충돌 검사는 두 중심을 잇는 **사각 영역의 최저 z** 만 본다. 그 안의 실제 형상을 보지 않는다.",
    ],
}
(out / "print_feet.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")

print(f"원본 바닥 접지 : {area0:8.1f} mm²   덩어리 {len(clusters)}개")
for c in clusters:
    print(f"   덩어리 중심 {str(c['centroid']):>18s}  면적 {c['area_mm2']:7.2f} mm²")
for p in plan:
    print(f"  + {p['tag']:28s} {p['area_mm2']:7.1f} mm²")
for b in bridge_notes:
    print(f"    다리 {b}")
print(f"발 추가 접지    : {added:8.1f} mm²  (두께 {h} mm, 무게 +{rec['added_mass_g_pla_1p24']} g)")
print(f"출력: {dst}")
