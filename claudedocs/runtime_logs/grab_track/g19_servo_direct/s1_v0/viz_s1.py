"""S1 설계 시각화: XZ 측면(닫힘/열림 29.3°) + 3D 등각. 입력 = s1_v0/*.stl (link5 mm)."""
import trimesh, numpy as np, json, glob, os, math, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
_fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fm.fontManager.addfont(_fp); plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name(); plt.rcParams['axes.unicode_minus']=False
D=os.path.dirname(os.path.abspath(__file__))+'/'; VS=os.path.abspath(D+'../vendor_step_parts')+'/'
dj=json.load(open(D+'design.json')); P=dj['params']; H=np.array(P['hinge_xz']); th=dj['derived']['open_deg_for_mouth']
door=trimesh.load(D+'door_ALL.stl',force='mesh'); fixed=trimesh.load(D+'fixed_ALL.stl',force='mesh')
l5=trimesh.load('/home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/urdf/meshes/link5.stl',force='mesh')
def step_to_l5(m):
    v=m.vertices; m2=m.copy(); m2.vertices=np.stack([-(v[:,1]+0.88),346.07-v[:,2],v[:,0]-236.967],1); return m2
servo=[step_to_l5(trimesh.load(VS+f,force='mesh')) for f in ('gripper_servo_case_SG.stl','gripper_servo_case_ZK.stl','gripper_servo_case_XG.stl')]
fjaw=step_to_l5(trimesh.load(VS+'fixed_jaw.stl',force='mesh')); base=step_to_l5(trimesh.load(VS+'gripper_base.stl',force='mesh'))
def rot(m,deg):
    T=trimesh.transformations.rotation_matrix(math.radians(deg),[0,-1,0],[H[0],0,H[1]]); m2=m.copy(); m2.apply_transform(T); return m2
def outline(ax,m,y,col,lw=1.0,label=None):
    s=m.section(plane_origin=[0,y,0],plane_normal=[0,1,0])
    if s is None: return
    for i,d in enumerate(s.discrete): ax.plot(d[:,0],d[:,2],'-',color=col,lw=lw,label=label if i==0 else None)
fig=plt.figure(figsize=(22,10)); gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.1])
for k,(deg,title) in enumerate(((0.0,'닫힘 (서보 0°)'),(th,f'열림 {th}° (입 58 mm)'))):
    ax=fig.add_subplot(gs[0,k]); dr=rot(door,deg)
    for y,lw in ((19.5,1.4),(0.0,0.8),(-19.7,1.4)):
        outline(ax,dr,y,'tab:blue',lw,'가동부 (문)' if (y==19.5) else None); outline(ax,fixed,y,'tab:green',lw,'고정부' if y==19.5 else None)
    outline(ax,l5,0.0,'0.4',0.7,'link5 (URDF)'); outline(ax,l5,16.0,'0.6',0.5); outline(ax,fjaw,17.0,'olive',0.7,'순정 고정 조'); 
    for s_ in servo: outline(ax,s_,0.0,'0.7',0.5)
    outline(ax,base,0.0,'tab:cyan',0.6,'그리퍼 베이스')
    ax.plot(H[0],H[1],'k+',ms=12); ax.set_aspect('equal'); ax.set_xlim(-45,75); ax.set_ylim(180,10); ax.grid(alpha=.3); ax.set_title(f'{title} — XZ 측면 (Y=±19.5 측판, Y=0 벽)',fontsize=12); ax.set_xlabel('link5 X (mm)  ← 고정 조 쪽 | 열림 쪽 →'); ax.set_ylabel('link5 Z (mm, 팁 ↓)')
    if k==0: ax.legend(fontsize=8.5,loc='upper right')
ax=fig.add_subplot(gs[0,2],projection='3d')
for m,c,a in ((rot(door,th),'tab:blue',.55),(fixed,'tab:green',.55),(l5,'0.6',.25),(fjaw,'olive',.35)):
    tri=m.vertices[m.faces]; pc=Poly3DCollection(tri,alpha=a,facecolor=c,edgecolor='none'); ax.add_collection3d(pc)
ax.set_xlim(-40,60); ax.set_ylim(-50,50); ax.set_zlim(180,20); ax.view_init(elev=25,azim=-50); ax.set_title(f'3D 등각, 열림 {th}° (파랑 가동부 · 초록 고정부 · 회색 link5)',fontsize=12); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
d=dj['derived']; fig.text(0.01,0.01,f"자중 {d['tool_mass_g']} g (가동 {d['door_g']} + 고정 {d['fixed_g']} + 나사 {d['hardware_g']}) · 공동 {d['cavity_cm3']} cm³ · 립 반경 {d['lip_radius_from_hinge_mm']} · 립 힘 최대 {d['lip_force_max_N']} N · 게이트 {sum(v['pass'] for v in dj['gates'].values())}/{len(dj['gates'])} PASS",fontsize=10)
plt.tight_layout(); plt.savefig(D+'s1_v0_design.png',dpi=110); print('saved')
