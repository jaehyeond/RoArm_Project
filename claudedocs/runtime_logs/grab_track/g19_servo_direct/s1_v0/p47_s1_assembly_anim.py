"""p47: S1 조립 3D 애니메이션 (유튜브식 분해→조립). 벤더 STEP 부품 STL + 우리 door_ALL/fixed_ALL. pyrender EGL 오프스크린 → PNG 프레임 → ffmpeg mp4.
좌표 = STEP 루트(mm). link5→STEP: X=l5z+236.967, Y=−l5x−0.88, Z=346.07−l5y. 서보축 = STEP (289.002, −0.88), 힌지축 = STEP Z.
장면(초): 0~1.5 순정 상태 → 1.5~3.5 M3×4 8개 풀기 → 3.5~5 순정 가동 조 제거 → 5~7.5 고정부 접근·M3×8 3개 → 7.5~10.5 문 접근(팁 쪽에서 축방향) → 10.5~12.5 M3×6 8개 → 12.5~16 문 0→30→0° 개폐.
사용: python p47_s1_assembly_anim.py --test   (키프레임 4장 시트)   /   python p47_s1_assembly_anim.py  (360 프레임 + mp4)
"""
import os, sys, glob, json, subprocess
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np, trimesh, pyrender
from PIL import Image, ImageDraw, ImageFont

H = os.path.dirname(os.path.abspath(__file__)) + '/'
V = os.path.abspath(H + '../vendor_step_parts') + '/'
OUT = H + 'assembly_anim/'; os.makedirs(OUT + 'frames', exist_ok=True)
AX = np.array([289.002, -0.88, 0.0])
T = np.array([[0, 0, 1, 236.967], [-1, 0, 0, -0.88], [0, -1, 0, 346.07], [0, 0, 0, 1]])
FPS, W, Hh = 24, 1280, 720
DUR = 16.0
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def load(f, tf=None):
    m = trimesh.load(f, force='mesh')
    if tf is not None: m.apply_transform(tf)
    return m

def cyl(r, h, axis_dir, base, sections=32):
    """base 점에서 axis_dir 방향으로 높이 h 인 원기둥."""
    c = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
    c.apply_translation([0, 0, h / 2])
    d = np.asarray(axis_dir, float); d /= np.linalg.norm(d)
    R = trimesh.geometry.align_vectors([0, 0, 1], d)
    c.apply_transform(R); c.apply_translation(base); return c

def screw(axis_dir, head_base, shank_len, head_len=3.0, r=1.5, rh=2.75):
    """머리 밑면이 head_base, 샹크가 axis_dir 로 shank_len 들어가는 볼트(머리는 반대쪽)."""
    d = np.asarray(axis_dir, float); d /= np.linalg.norm(d)
    return trimesh.util.concatenate([cyl(r, shank_len, d, head_base), cyl(rh, head_len, -d, head_base)])

def hexnut(axis_dir, base, h=2.4, r=3.2):
    return cyl(r, h, axis_dir, base, sections=6)

# ── 부품 ──────────────────────────────────────────────────────────────────
metal = dict(color=[0.72, 0.72, 0.74, 1.0], metallic=0.9, rough=0.35)
dark = dict(color=[0.25, 0.25, 0.27, 1.0], metallic=0.2, rough=0.7)
blue = dict(color=[0.20, 0.45, 0.85, 1.0], metallic=0.0, rough=0.6)
green = dict(color=[0.25, 0.65, 0.35, 1.0], metallic=0.0, rough=0.6)
brass = dict(color=[0.80, 0.62, 0.25, 1.0], metallic=0.9, rough=0.3)
red = dict(color=[0.85, 0.25, 0.2, 1.0], metallic=0.8, rough=0.4)

static = [(load(V + 'gripper_base.stl'), dark), (load(V + 'wrist_b.stl'), dark), (load(V + 'fixed_jaw.stl'), metal),
          (load(V + 'gripper_servo_case_SG.stl'), dark), (load(V + 'gripper_servo_case_ZK.stl'), dark), (load(V + 'gripper_servo_case_XG.stl'), dark),
          (load(V + 'gripper_servo_drive_disc.stl'), red), (load(V + 'gripper_servo_driven_disc.stl'), red), (load(V + 'screws_PA2x5_servo_case.stl'), metal)]
stock_jaw = load(V + 'movable_jaw.stl')
scr = load(V + 'screws_M3x4_disc.stl').split(only_watertight=False)
centre = [s for s in scr if np.linalg.norm(s.centroid[:2] - AX[:2]) < 1.5]
stock_screws = [s for s in scr if np.linalg.norm(s.centroid[:2] - AX[:2]) >= 1.5]
print('M3x4 bodies', len(scr), 'centre', len(centre), 'removable', len(stock_screws))
door = load(H + 'door_ALL.stl', T); fixed = load(H + 'fixed_ALL.stl', T)
P = json.load(open(H + 'design.json'))['params']

# 우리 나사: 문 M3×6 ×8 (PCD 14, 뺨 바깥면 z 325.0 / 367.37, 축 ±Z)
door_screws = []
for zc, d in ((325.0, +1), (367.37, -1)):
    for a in (45, 135, 225, 315):
        b = AX + np.array([7 * np.cos(np.radians(a)), 7 * np.sin(np.radians(a)), 0]); b[2] = zc
        door_screws.append((screw([0, 0, d], b, 6.0), d))
# 고정부 M3×8 ×3 + 너트: 판 link5 x [−15.54,−11.54] → STEP y [10.66,14.66]; 볼트 축 = STEP −Y (바깥 +Y 에서 삽입)
fixed_bolts = []
for (ly, lz) in P['fixed_holes_yz']:
    p = T @ np.array([0.0, ly, lz, 1.0]); p[1] = 14.66
    fixed_bolts.append(screw([0, -1, 0], p[:3], 8.0))
    fixed_nuts_base = p[:3].copy(); fixed_nuts_base[1] = 14.66 - 8.0 + 2.4   # 볼트 끝에 너트
    fixed_bolts.append(hexnut([0, -1, 0], fixed_nuts_base))

def ease(t):  # 0..1 smoothstep
    t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)
def seg(t, a, b): return ease((t - a) / (b - a))

def rot_z_about(theta, c):
    R = trimesh.transformations.rotation_matrix(theta, [0, 0, 1], c); return R

def pose_at(t):
    """t 초의 각 부품 4×4 변환 + 자막."""
    I = np.eye(4)
    s_unscrew = seg(t, 1.5, 3.5); s_jaw = seg(t, 3.5, 5.0); s_fixed = seg(t, 5.0, 6.5); s_fbolt = seg(t, 6.5, 7.5)
    s_door = seg(t, 7.5, 10.0); s_dscrew = seg(t, 10.5, 12.5)
    th = 0.0
    if t > 12.5:
        u = (t - 12.5) / 3.5; th = np.radians(30) * (0.5 - 0.5 * np.cos(2 * np.pi * u))   # 0→30→0
    P_ = {}
    # 순정 나사: z 바깥으로 15 풀린 뒤 +X 로 사라짐
    for i, s in enumerate(stock_screws):
        d = 1 if s.centroid[2] < 346 else -1
        tr = np.array([120 * s_jaw, 0, -d * 15 * s_unscrew]); P_[('ss', i)] = trimesh.transformations.translation_matrix(tr)
    P_['jaw'] = trimesh.transformations.translation_matrix([120 * s_jaw, 0, 0])
    P_['fixed'] = trimesh.transformations.translation_matrix([0, 28 * (1 - s_fixed), 0])
    for i in range(len(fixed_bolts)):
        P_[('fb', i)] = trimesh.transformations.translation_matrix([0, (14 if i % 2 == 0 else -14) * (1 - s_fbolt), 0])
    door_tr = trimesh.transformations.translation_matrix([90 * (1 - s_door), 0, 0])
    P_['door'] = rot_z_about(-th, AX) @ door_tr
    for i, (m, d) in enumerate(door_screws):
        P_[('ds', i)] = rot_z_about(-th, AX) @ trimesh.transformations.translation_matrix([0, 0, -d * 18 * (1 - s_dscrew)])
    cap = ('① 순정 그리퍼 (STEP 조립 상태)' if t < 1.5 else '② 순정 가동 조 M3×4 8개 풀기 (중앙 나사는 유지)' if t < 3.5 else '③ 순정 가동 조 떼기 — 디스크 2장은 서보에 남음' if t < 5.0
           else '④ 고정부: 고정 조 바깥면 3구멍에 M3×8 + 너트' if t < 7.5 else '⑤ 문: 팁 쪽에서 축방향으로 끼워 뺨 창을 중앙 나사·축 끝에 맞춤' if t < 10.5
           else '⑥ M3×6 ×4 를 뺨 바깥에서 디스크 탭에 (양쪽 8개)' if t < 12.5 else f'⑦ 서보 명령 0→30°: 입 {58 * th / np.radians(29.3):.0f} mm (문 각 {np.degrees(th):.0f}°)')
    vis = dict(stock=(t < 5.0), fixed=(t >= 5.0), door=(t >= 7.5), dscrew=(t >= 10.0), fbolt=(t >= 6.0))
    return P_, cap, vis, th

def _bb(meshes):
    lo = np.min([m.bounds[0] for m in meshes], axis=0); hi = np.max([m.bounds[1] for m in meshes], axis=0)
    return (lo + hi) / 2, np.linalg.norm(hi - lo)
BB_STOCK = _bb([m for m, _ in static] + [stock_jaw])
_door_open = door.copy(); _door_open.apply_transform(rot_z_about(-np.radians(30), AX))
BB_FULL = _bb([m for m, _ in static] + [fixed, door, _door_open])
print('bbox stock', np.round(BB_STOCK[0], 1), round(BB_STOCK[1], 1), '| full', np.round(BB_FULL[0], 1), round(BB_FULL[1], 1))

def build_scene(t):
    P_, cap, vis, th = pose_at(t)
    sc = pyrender.Scene(bg_color=[0.96, 0.96, 0.97, 1.0], ambient_light=[0.35, 0.35, 0.35])
    def add(m, mat, pose=np.eye(4)):
        mt = pyrender.MetallicRoughnessMaterial(baseColorFactor=mat['color'], metallicFactor=mat['metallic'], roughnessFactor=mat['rough'])
        sc.add(pyrender.Mesh.from_trimesh(m, material=mt, smooth=False), pose=pose)
    for m, mat in static: add(m, mat)
    for s in centre: add(s, metal)
    if vis['stock']:
        add(stock_jaw, metal, P_['jaw'])
        for i, s in enumerate(stock_screws): add(s, brass, P_[('ss', i)])
    if vis['fixed']:
        add(fixed, green, P_['fixed'])
        if vis['fbolt']:
            for i, b in enumerate(fixed_bolts): add(b, brass, P_[('fb', i)])
    if vis['door']:
        add(door, blue, P_['door'])
        if vis['dscrew']:
            for i, (m, d) in enumerate(door_screws): add(m, brass, P_[('ds', i)])
    # 카메라: 보이는 부품 경계상자에 자동 맞춤(순정 → 전체 조립체로 5~7.5 s 에 블렌드), up = STEP +Y(고정 조 쪽) → 입이 아래를 향함
    k = seg(t, 5.0, 7.5)
    c = (1 - k) * BB_STOCK[0] + k * BB_FULL[0]; diag = (1 - k) * BB_STOCK[1] + k * BB_FULL[1]
    # up = STEP −X(팔축 반대) → 손목이 위, 팁(입)이 아래. az=−90° = 구동(라벨) 쪽 옆면, ±50° 궤도. el>0 = 약간 위(손목 쪽)에서.
    c = c + np.array([8.0, 0, 0]); dist = 0.56 * diag / np.tan(np.radians(32) / 2)     # 자막 띠(하단 10 %) 여유
    az = np.radians(-90 + 50 * np.sin(2 * np.pi * t / DUR)); el = np.radians(10)
    eye = c + dist * np.array([-np.sin(el), np.cos(el) * np.cos(az), np.cos(el) * np.sin(az)])
    cam_pose = look_at(eye, c, up=[-1, 0, 0])
    cam = pyrender.PerspectiveCamera(yfov=np.radians(32), aspectRatio=W / Hh); sc.add(cam, pose=cam_pose)
    sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=cam_pose)
    sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.5), pose=look_at(c + [-150, 200, -120], c, [-1, 0, 0]))
    return sc, cap

def look_at(eye, target, up):
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f); s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    M = np.eye(4); M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye; return M

def caption(img, text, t):
    im = Image.fromarray(img); dr = ImageDraw.Draw(im); f = ImageFont.truetype(FONT, 30); f2 = ImageFont.truetype(FONT, 20)
    dr.rectangle([0, Hh - 70, W, Hh], fill=(20, 20, 24)); dr.text((24, Hh - 58), text, font=f, fill=(255, 255, 255))
    dr.text((W - 330, Hh - 40), f'S1 v0 · D480 · t={t:4.1f}s', font=f2, fill=(180, 180, 190)); return np.asarray(im)

if __name__ == '__main__':
    r = pyrender.OffscreenRenderer(W, Hh)
    if '--test' in sys.argv:
        tiles = []
        for t in (0.8, 4.2, 6.0, 9.0, 11.5, 14.2):
            sc, cap = build_scene(t); col, _ = r.render(sc); tiles.append(Image.fromarray(caption(col, cap, t)).resize((640, 360)))
        sheet = Image.new('RGB', (1280, 1080), 'white')
        for i, tl in enumerate(tiles): sheet.paste(tl, ((i % 2) * 640, (i // 2) * 360))
        sheet.save(OUT + 'keyframes_test.png'); print('saved', OUT + 'keyframes_test.png'); sys.exit(0)
    n = int(DUR * FPS)
    for i in range(n):
        t = i / FPS; sc, cap = build_scene(t); col, _ = r.render(sc)
        Image.fromarray(caption(col, cap, t)).save(OUT + f'frames/f{i:04d}.png')
        if i % 48 == 0: print('frame', i, '/', n, flush=True)
    r.delete()
    mp4 = OUT + 's1_assembly.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', OUT + 'frames/f%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', mp4], check=True)
    print('saved', mp4, os.path.getsize(mp4), 'B')
