"""p44: 순정 조를 인쇄 부품으로 바꿀 때 가능한 단순 구조(S1: 고정 셸 + 서보축 직결 가동 셸) vs 현행 g18(양쪽 가동, 4절+기어). 벤더 STEP 윤곽 위 개념도."""
import trimesh, numpy as np, os, warnings, json; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import Circle, Rectangle, Polygon, Wedge, Arc, FancyArrowPatch
_fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fm.fontManager.addfont(_fp); plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name(); plt.rcParams['axes.unicode_minus']=False
D=os.path.dirname(os.path.abspath(__file__))+'/'; AX=np.array([289.002,-0.88])
mj=trimesh.load(D+'movable_jaw.stl',force='mesh'); fx=trimesh.load(D+'fixed_jaw.stl',force='mesh'); base=trimesh.load(D+'gripper_base.stl',force='mesh')
P=json.load(open(D+'parts_bbox.json')); sv=[np.array(p['bbox']) for p in P if 'ST3215 v1 (4)' in p['path']]; sv=np.array(sv); svb=[sv[:,0].min(),sv[:,1].min(),sv[:,3].max(),sv[:,4].max()]
def sec_xy(ax,mesh,z,col,lw,label=None,ls='-'):
    s=mesh.section(plane_origin=[0,0,z],plane_normal=[0,0,1])
    if s is None: return
    for i,d in enumerate(s.discrete): ax.plot(d[:,0],d[:,1],ls,color=col,lw=lw,label=label if i==0 else None)
def rot(pts,ang_deg,c=AX):
    a=np.radians(ang_deg); R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]); return (np.asarray(pts)-c)@R.T+c
def robot(ax):
    ax.add_patch(Rectangle((svb[0],svb[1]),svb[2]-svb[0],svb[3]-svb[1],fc='0.85',ec='0.5',lw=1,label='그리퍼 서보 ST3215-HS'))
    sec_xy(ax,base,346,'tab:blue',1.0,'그리퍼 베이스(link5)'); sec_xy(ax,fx,346,'tab:green',1.3,'순정 고정 조 (그대로 둠)'); sec_xy(ax,fx,331,'tab:olive',0.8)
    ax.plot(*AX,'k+',ms=14); ax.text(AX[0]-2,AX[1]+13,'서보축',fontsize=9)
# 보울 (개념): 반경 27, 중심 = 서보축에서 팔축 방향 +100, 파팅면 y = 파팅
R=27.0; CX=AX[0]+100; PY=4.4   # 파팅면: 순정 두 블레이드 사이 중앙 (y≈4.4)
def half_bowl(side):   # side +1: 고정(y>PY, 고정 조 쪽), -1: 가동(y<PY)
    th=np.linspace(0,np.pi,60)*(1 if side>0 else -1)
    outer=np.stack([CX+R*np.cos(th),PY+R*np.sin(th)],1); inner=np.stack([CX+(R-2)*np.cos(th[::-1]),PY+(R-2)*np.sin(th[::-1])],1)
    return np.vstack([outer,inner])
fig,axs=plt.subplots(1,2,figsize=(21,9.5))
# ── A: 구조 비교 블록도 (g18 vs S1) ──
ax=axs[0]; ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,100)
from matplotlib.patches import FancyBboxPatch
def box(x,y,w,h,t,fc,fs=9.5): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4',fc=fc,ec='gray')); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arrow(x0,y0,x1,y1): ax.annotate('',(x1,y1),(x0,y0),arrowprops=dict(arrowstyle='->',color='k',lw=1.2))
ax.text(2,97,'현행 g18 (D462~D479): 셸 2개 다 움직임',fontsize=12,fontweight='bold')
chain=[('서보\nST3215-HS','0.85'),('순정 가동 조\n(레버)','white'),('크랭크판\n+포켓 너트','lightblue'),('로드 50','lightblue'),('셸크랭크 40','lightblue'),('셸 L','orange'),('기어쌍\n반치 위상','lightblue'),('셸 R','orange')]
x=2
for i,(t,c) in enumerate(chain):
    box(x,80,10.5,10,t,c,8.5); 
    if i<len(chain)-1: arrow(x+10.5,85,x+12.2,85)
    x+=12.2
chain2=[('순정 고정 조','white'),('브래킷\n3점 너트트랩','lightblue'),('요크 상·하판\n+ 알루 피벗 볼트 2','lightblue'),('셸 L/R 피벗','orange')]
x=2
for i,(t,c) in enumerate(chain2):
    box(x,62,16,10,t,c,8.5)
    if i<len(chain2)-1: arrow(x+16,67,x+18.5,67)
    x+=18.5
ax.text(2,55,'인쇄 조각 ~10 · 체결 7종 · 시뮬은 표(운동학) 결합 · 폐루프라 URDF 표현 불가',fontsize=9.5,color='dimgray')
ax.text(2,45,'S1 안: 순정 가동 조 자리에 인쇄 부품, 셸 하나만 움직임',fontsize=12,fontweight='bold')
chain=[('서보\nST3215-HS','0.85'),('구동 디스크\n(순정, 나사 4+1)','white'),('인쇄 가동 부품\n포크 + 암 + 반쪽 보울','tab:blue')]
x=2
for i,(t,c) in enumerate(chain):
    box(x,28,22,10,t,c if c!='tab:blue' else '#9ecae1',9)
    if i<len(chain)-1: arrow(x+22,33,x+24.5,33)
    x+=24.5
chain2=[('순정 고정 조','white'),('브래킷 (g18 재사용)','lightblue'),('인쇄 고정 반쪽 보울','#a1d99b')]
x=2
for i,(t,c) in enumerate(chain2):
    box(x,12,22,10,t,c,9)
    if i<len(chain2)-1: arrow(x+22,17,x+24.5,17)
    x+=24.5
ax.text(2,5,'인쇄 2개 · 순정 나사 9 (M3×8 로 교체) + 브래킷 M3 3 · 시뮬 = 관절 1개 직결 (실물과 동일 구조)',fontsize=9.5,color='dimgray')
ax.set_title('A. 구조 비교: 무엇이 없어지나',fontsize=12.5)
# ── B: S1 개념 (순정 가동 조 → 인쇄 "포크+가동 셸" / 고정 셸은 브래킷 or 인쇄 고정 조) ──
ax=axs[1]; robot(ax)
# 인쇄 가동 부품: 포크 뺨(STEP 윤곽 x≤339) + 암 + 가동 반쪽 보울, 서보축 회전
s=mj.section(plane_origin=[0,0,327.25],plane_normal=[0,0,1]); cheek=max(s.discrete,key=len)[:,:2]; cheek=cheek[cheek[:,0]<=339]
arm=np.array([[330,-3],[CX-R+6,PY-6],[CX-R+6,PY-14],[326,-13]])
mov=np.vstack([cheek,arm,half_bowl(-1)])
for ang,alpha,lab in ((0,.35,'인쇄 가동 부품 = 포크 + 암 + 반쪽 보울 (닫힘)'),(-42,.15,'같은 부품, 42° 열림')):
    ax.add_patch(Polygon(rot(cheek,ang),closed=True,fc='tab:blue',alpha=alpha,ec='tab:blue',lw=1.5,label=lab)); ax.add_patch(Polygon(rot(arm,ang),closed=True,fc='tab:blue',alpha=alpha,ec='tab:blue',lw=1.5)); ax.add_patch(Polygon(rot(half_bowl(-1),ang),closed=True,fc='tab:blue',alpha=alpha,ec='tab:blue',lw=1.5))
# 고정 반쪽: 브래킷(고정 조 3구멍) + 고정 반쪽 보울
ax.add_patch(Rectangle((AX[0]+30,PY+5),CX-R-AX[0]-30+4,4,fc='tab:red',alpha=.3,ec='tab:red',label='브래킷 (현행 g18 것 재사용, 고정 조 3구멍)'))
ax.add_patch(Polygon(half_bowl(+1),closed=True,fc='tab:green',alpha=.35,ec='tab:green',lw=1.5,label='인쇄 고정 반쪽 보울 (브래킷에 볼트)'))
ax.add_patch(Arc(AX,2*(CX-AX[0]),2*(CX-AX[0]),theta1=-44,theta2=2,ls='--',color='tab:blue',lw=1)); ax.text(CX-30,PY-52,'가동 립 궤적 (반경 ≈100)',color='tab:blue',fontsize=9)
ax.annotate('립 힘 ≈ 1.96 N·m / 0.1 m ≈ 20 N',(CX,PY-R-2),xytext=(CX-25,PY-R-22),fontsize=9.5,arrowprops=dict(arrowstyle='->'))
ax.set_title('B. S1 안: 순정 조 자리에 "포크+가동 셸" 인쇄 — 링크·기어·요크 전부 삭제 (인쇄 2개 + 순정 나사 9 + 브래킷 볼트 3)',fontsize=12.5)
ax.set_aspect('equal'); ax.set_xlim(230,420); ax.set_ylim(-60,60); ax.grid(alpha=.3); ax.legend(fontsize=8.5,loc='upper right'); ax.set_xlabel('link5 Z (팔축) →')
ax.text(232,-45,'시뮬 = 관절 1개(link5_to_gripper_link)가 가동 부품을 직접 돌림. 표·mimic·폐루프 없음 → 실물과 동일 구조.\n보울 크기·립 형상은 자유 (요크·기어 제약 없음 → B1 62% 문제도 여기서 풀 수 있음).',fontsize=9.5,va='top',bbox=dict(fc='lightyellow',ec='gray'))
plt.tight_layout(); plt.savefig(D+'p44_topology_S1.png',dpi=105); print('saved')
