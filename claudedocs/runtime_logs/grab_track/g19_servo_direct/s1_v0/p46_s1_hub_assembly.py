"""p46: S1 문 뺨이 순정 서보 디스크에 어떻게 물리는가 — (A) 서보축 평면 단면(조립 상태) (B) 같은 단면 분해도 (C) 허브 정면.
벤더 STEP 부품 STL(vendor_step_parts/) + 우리 door_ALL.stl(link5 mm). STEP 프레임: X=link5 Z+236.967, Y=−link5 X−0.88, Z=346.07−link5 Y.
'디스크 보스'(Ø10.6) 는 플랜지의 **서보 쪽** 원기둥이다(서보 앞 링에 끼움). 뺨은 플랜지 바깥면에 붙고, 뺨 창 Ø11 은 중앙 나사(구동측)/축 끝 Ø6(종동측) 여유 구멍.
"""
import os, warnings, json
import numpy as np, trimesh
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
_fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fm.fontManager.addfont(_fp); plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name()
from shapely.ops import unary_union
from shapely.geometry import Polygon
from matplotlib.patches import Rectangle, Circle

H = os.path.dirname(os.path.abspath(__file__)) + '/'
V = os.path.abspath(H + '../vendor_step_parts') + '/'
AX = (289.002, -0.88)                     # 서보축 (STEP x,y)
T = np.array([[0, 0, 1, 236.967], [-1, 0, 0, -0.88], [0, -1, 0, 346.07], [0, 0, 0, 1]])   # link5 → STEP

vend = {'gripper_servo_case_SG': ('서보 케이스', 'tab:gray'), 'gripper_servo_case_ZK': ('', 'silver'), 'gripper_servo_case_XG': ('', 'darkgray'),
        'gripper_servo_gear_GE27': ('서보 출력 기어', 'tab:purple'), 'gripper_servo_drive_disc': ('구동 디스크(라벨 쪽)', 'tab:red'),
        'gripper_servo_driven_disc': ('종동 디스크(반대쪽)', 'tab:orange'), 'gripper_base': ('그리퍼 베이스 = link5', 'tab:blue'),
        'fixed_jaw': ('순정 고정 조', 'tab:green')}
M = {k: trimesh.load(V + k + '.stl', force='mesh') for k in vend}
import glob
_pieces = [f for f in sorted(glob.glob(H + 'door_*.stl')) if 'ALL' not in os.path.basename(f)]   # 조각별 볼록·수밀 STL (door_ALL 은 비수밀)
door_pieces = []
for f in _pieces:
    m = trimesh.load(f, force='mesh'); m.apply_transform(T); door_pieces.append(m)
print('door pieces', len(door_pieces))
P = json.load(open(H + 'design.json'))['params']

def union_outline(meshes, origin, normal, ax_idx):
    """조각 메시들의 단면 폴리곤 합집합 → 바깥/안쪽 링 목록 (2D 좌표 열 ax_idx)."""
    polys = []
    for m in meshes:
        sec = m.section(plane_origin=origin, plane_normal=normal)
        if sec is None: continue
        for d in sec.discrete:
            pts = d[:, ax_idx]
            if len(pts) >= 4:
                pg = Polygon(pts).buffer(0)
                if pg.area > 0.05: polys.append(pg)
    if not polys: return []
    u = unary_union(polys).buffer(0.02).buffer(-0.02)
    geoms = list(u.geoms) if hasattr(u, 'geoms') else [u]
    rings = []
    for g in geoms:
        rings.append(np.asarray(g.exterior.coords))
        rings += [np.asarray(r.coords) for r in g.interiors]
    return rings

def sec_xz(mesh, y=AX[1]):
    s = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0]); return [] if s is None else s.discrete

def draw_stack(ax, explode=0.0, title=''):
    for k, (lab, col) in vend.items():
        for i, d in enumerate(sec_xz(M[k])):
            ax.plot(d[:, 0], d[:, 2], '-', color=col, lw=1.0, label=(lab if (i == 0 and lab) else None))
    # 문(뺨 2장 + 브리지): 구동측 뺨(z<346) 은 −z 로, 종동측 뺨(z>346) 은 +z 로 밀어 분해
    rings = union_outline(door_pieces, [0, AX[1], 0], [0, 1, 0], [0, 2])
    for i, r in enumerate(rings):
        zc = r[:, 1].mean(); dz = -explode if zc < 346 else explode
        ax.plot(r[:, 0], r[:, 1] + dz, '-', color='tab:blue', lw=2.0, label=('S1 문 (PLA 3 mm 뺨)' if i == 0 else None))
    # 우리 나사 M3×6 ×4/뺨 (PCD 14 → 단면 평면에는 x=AX±7 두 개가 걸림). 구동측 뺨 z 325.0~328.0, 종동측 364.37~367.37
    for x in (AX[0] - 7, AX[0] + 7):
        for (zc_out, sgn) in ((325.0, -1), (367.37, +1)):        # 뺨 바깥면
            e = explode * 2.2 * sgn
            head = Rectangle((x - 2.75, zc_out + e if sgn < 0 else zc_out + e - 0.0), 5.5, 3 * sgn, fc='tab:brown', ec='k', lw=.5)
            ax.add_patch(Rectangle((x - 2.75, zc_out + e - 3 if sgn < 0 else zc_out + e), 5.5, 3, fc='tab:brown', ec='k', lw=.5))
            ax.add_patch(Rectangle((x - 1.5, zc_out + e if sgn < 0 else zc_out + e - 6), 3, 6, fc='tab:brown', ec='k', lw=.5))
    for (z0, sgn) in ((325.0, +1), (367.37, -1)):      # 뺨 두께 3 구간에 창 Ø11 = 흰 틈 (조각 STL 엔 구멍이 없어 설계값으로 그림)
        e = -explode if sgn > 0 else explode
        zc = z0 + e if sgn > 0 else z0 + e - 3
        ax.add_patch(Rectangle((AX[0] - 5.5, zc), 11, 3, fc='white', ec='tab:blue', lw=1.2, zorder=3))
        ax.text(AX[0], zc + 1.5, 'Ø11 창', ha='center', va='center', fontsize=7, color='tab:blue', zorder=4)
    ax.axvline(AX[0], ls='--', c='k', lw=.6)
    ax.set_aspect('equal'); ax.set_xlim(262, 318); ax.set_ylim(315 - explode * 3, 378 + explode * 3)
    ax.set_xlabel('STEP X = 팔축(link5 +Z) mm'); ax.set_ylabel('STEP Z = 힌지축(link5 −Y) mm'); ax.set_title(title); ax.grid(alpha=.3)

fig, axs = plt.subplots(1, 3, figsize=(24, 9.5))
# (A) 조립 상태
ax = axs[0]; draw_stack(ax, 0.0, '(A) 조립 상태 — 서보축 평면 단면')
ax.annotate('디스크 보스 Ø10.6\n(서보 쪽, 서보 앞 링에 끼움)', (AX[0] + 5.3, 330.5), (300, 322), arrowprops=dict(arrowstyle='->'), fontsize=9)
ax.annotate('플랜지 Ø18.8 × 2.5\n뺨은 이 바깥면에 붙는다', (AX[0] - 9.4, 328.4), (266, 336), arrowprops=dict(arrowstyle='->'), fontsize=9)
ax.annotate('뺨 창 Ø11 = 중앙 M3(스플라인 고정) 통과 여유', (AX[0], 326.5), (292, 318.5), arrowprops=dict(arrowstyle='->'), fontsize=9)
ax.annotate('M3×6 ×4 (PCD 14)\n뺨 3 + 디스크 탭 2.5, 돌출 0.5', (AX[0] + 7, 330.5), (300, 341), arrowprops=dict(arrowstyle='->'), fontsize=9)
ax.annotate('종동 디스크 + 축 끝 Ø6 (창 Ø11 안)', (AX[0], 364.9), (293, 372), arrowprops=dict(arrowstyle='->'), fontsize=9)
ax.legend(fontsize=8, loc='lower left')
# (B) 분해도
ax = axs[1]; draw_stack(ax, 6.0, '(B) 분해도 — 순서: 서보 → 디스크(붙어 있음) → 문 뺨 → M3×6 ×4 (양쪽)')
ax.text(264, 316, '① 서보에 디스크 2장은 이미 붙어 있음(중앙 나사 유지)\n② 순정 조 M3×4 8개 풀어 조 제거\n③ 문 뺨 창을 중앙 나사/축 끝에 맞춰 얹고 4구멍을 디스크 탭에 맞춤\n④ M3×6 ×4 를 뺨 바깥에서 조임 (양쪽 8개) — PLA 라 살짝만', fontsize=9, va='bottom',
        bbox=dict(fc='lightyellow', ec='gray'))
# (C) 허브 정면 (구동측 뺨 중간 z=326.5)
ax = axs[2]
for i, r in enumerate(union_outline(door_pieces, [0, 0, 326.5], [0, 0, 1], [0, 1])):
    ax.plot(r[:, 0], r[:, 1], '-', color='tab:blue', lw=1.5, label='S1 문 구동측 뺨 (z=326.5)' if i == 0 else None)
for i, d in enumerate(M['gripper_servo_drive_disc'].section(plane_origin=[0, 0, 328.5], plane_normal=[0, 0, 1]).discrete):
    ax.plot(d[:, 0], d[:, 1], '-', color='tab:red', lw=1, label='구동 디스크 플랜지 Ø18.8 (z=328.5)' if i == 0 else None)
for i, d in enumerate(M['gripper_servo_drive_disc'].section(plane_origin=[0, 0, 330.5], plane_normal=[0, 0, 1]).discrete):
    ax.plot(d[:, 0], d[:, 1], '--', color='tab:red', lw=1, label='디스크 보스 Ø10.6 (z=330.5, 서보 쪽)' if i == 0 else None)
ax.add_patch(Circle(AX, 5.5, fc='white', ec='tab:blue', lw=1.5, zorder=2))
for a in (45, 135, 225, 315):
    ax.add_patch(Circle((AX[0] + 7 * np.cos(np.radians(a)), AX[1] + 7 * np.sin(np.radians(a))), 1.7, fc='white', ec='tab:blue', lw=1.5, zorder=2))
ax.add_patch(Circle((AX[0] + 6 * np.cos(np.radians(28.6)), AX[1] + 6 * np.sin(np.radians(28.6))), 0.6, fc='white', ec='tab:blue', lw=1.2, zorder=2))
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(AX[0] + 7 * np.cos(th), AX[1] + 7 * np.sin(th), ':', c='k', lw=.8, label='PCD 14')
for a in (45, 135, 225, 315):
    ax.add_patch(Circle((AX[0] + 7 * np.cos(np.radians(a)), AX[1] + 7 * np.sin(np.radians(a))), 1.5, fc='tab:brown', ec='k', lw=.5, zorder=3))
ax.add_patch(Circle(AX, 1.5, fc='tab:brown', ec='k', lw=.5, zorder=3)); ax.plot(*AX, 'k+', ms=10, zorder=4)
ax.text(AX[0] + 4.5, AX[1] + 7.5, '창 Ø11 (+ Ø3.4 ×4 PCD 14, 핀 Ø1.2) = 네잎', fontsize=8, color='tab:blue')
ax.text(AX[0], AX[1] - 12.5, 'M3 ×4 (PCD 14) + 중앙 M3', ha='center', fontsize=8)
ax.set_aspect('equal'); ax.set_xlim(272, 318); ax.set_ylim(-22, 22); ax.grid(alpha=.3)
ax.set_xlabel('STEP X (mm)'); ax.set_ylabel('STEP Y (mm)  [+Y = 고정 조 쪽]'); ax.set_title('(C) 허브 정면 — 뺨 구멍 4 + 창 vs 디스크'); ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); out = H + 'p46_s1_hub_assembly.png'; plt.savefig(out, dpi=110); print('saved', out)
