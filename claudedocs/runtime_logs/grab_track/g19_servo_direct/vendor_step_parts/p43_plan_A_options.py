"""p43: 방식 A 실행 계획 그림. (1) M2 여분 조 절단 안 (2) 축방향 샌드위치 (3) PLA 포크 안 (4) 공정 순서. 형상 = 벤더 STEP (movable_jaw.stl 등)."""
import trimesh, numpy as np, os, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch, FancyBboxPatch
_fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fm.fontManager.addfont(_fp); plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name(); plt.rcParams['axes.unicode_minus']=False
D=os.path.dirname(os.path.abspath(__file__))+'/'; AX=(289.002,-0.88); PIN=(AX[0]+20.186,AX[1]+5.789)   # 닫힘 자세 크랭크 핀 (r21 @ +16°)
mj=trimesh.load(D+'movable_jaw.stl',force='mesh'); dd=trimesh.load(D+'gripper_servo_drive_disc.stl',force='mesh')
def outline(ax,mesh,z,col,lw=1.3,label=None,fill=None,alpha=.15):
    sec=mesh.section(plane_origin=[0,0,z],plane_normal=[0,0,1])
    if sec is None: return
    for i,d in enumerate(sec.discrete):
        ax.plot(d[:,0],d[:,1],'-',color=col,lw=lw,label=label if i==0 else None)
        if fill and i==0: ax.add_patch(Polygon(d[:,:2],closed=True,fc=fill,ec='none',alpha=alpha))
def crank_arm(ax,col='tab:blue',txt=True):
    # 인쇄 크랭크 암 판: 허브 링(외경 r 12.5, PCD14 구멍 4 + 중앙 창 ⌀11) + 핀까지 테이퍼 암(폭 10) + 핀 보스 ⌀8
    th=np.linspace(0,2*np.pi,120); ax.plot(AX[0]+12.5*np.cos(th),AX[1]+12.5*np.sin(th),'-',color=col,lw=2)
    ax.add_patch(Circle(AX,12.5,fc=col,alpha=.18,ec='none'))
    ang=np.arctan2(PIN[1]-AX[1],PIN[0]-AX[0]); n=np.array([-np.sin(ang),np.cos(ang)])
    p0=np.array(AX)+np.array([np.cos(ang),np.sin(ang)])*8; p1=np.array(PIN)
    quad=np.array([p0+n*6,p1+n*4.5,p1-n*4.5,p0-n*6]); ax.add_patch(Polygon(quad,closed=True,fc=col,alpha=.18,ec=col,lw=2))
    ax.add_patch(Circle(PIN,4,fc='white',ec=col,lw=2)); ax.add_patch(Circle(PIN,1.5,fc=col,ec=col))
    for a in (45,135,225,315): ax.add_patch(Circle((AX[0]+7*np.cos(np.radians(a)),AX[1]+7*np.sin(np.radians(a))),1.6,fc='white',ec='k',lw=1))
    ax.add_patch(Circle(AX,5.5,fc='white',ec='k',lw=1)); 
    if txt: ax.annotate('인쇄 크랭크 암 (PLA 3 mm)\n디스크 나사 4개 밑에 끼움 (M3×4→M3×8)\n핀 = 서보축에서 r 21 (현행 설계값 그대로)',PIN,xytext=(PIN[0]+6,PIN[1]-20),fontsize=9.5,color=col,arrowprops=dict(arrowstyle='->',color=col))
fig=plt.figure(figsize=(20,13)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.05])
# ── (1) 여분 M2 조 절단 안 ──
ax=fig.add_subplot(gs[0,0])
outline(ax,mj,327.25,'k',1.4,'여분 조 뺨 (구동측, STEP 윤곽)',fill='gray',alpha=.12)
ax.add_patch(Rectangle((331,3.6),26,1.5,fc='tab:red',ec='tab:red',alpha=.35,label='잘라낼 블레이드 (x 339→357)'))
ax.add_patch(Rectangle((313,3.6),26,1.5,fc='tab:green',ec='none',alpha=.35,label='남길 뿌리+8 mm 스텁 (두 뺨을 잇는 다리)'))
ax.axvline(339,color='tab:red',ls='--',lw=2.5); ax.text(339.5,-14,'절단선\nSTEP x=339\n(서보축에서 50 mm)',color='tab:red',fontsize=10,fontweight='bold')
crank_arm(ax)
ax.plot(*AX,'k+',ms=14); ax.text(AX[0]-3,AX[1]+14,'서보축',fontsize=9)
ax.set_aspect('equal'); ax.set_xlim(272,362); ax.set_ylim(-30,22); ax.grid(alpha=.3); ax.legend(fontsize=8.5,loc='lower left')
ax.set_title('① 안 1: M2 여분 조를 자른다 — 블레이드만 제거, 뺨 2장과 8 mm 스텁은 남김',fontsize=13)
ax.text(273,19,'절단: 바이스에 물리고 쇠톱(24~32 TPI) 또는 로터리 절단 디스크. 1.5 mm 알루미늄이라 2~3분.\n마감: 줄로 버 제거. 정밀도 ±1 mm 면 충분 — 잘린 끝은 순정 조 부피 안이라 간섭이 자동으로 없다.\n조건: 먼저 여분 조를 M3 서보 디스크에 얹어 4구멍·폭 39.4 가 맞는지 끼워 본다 (0단계).',fontsize=9.5,va='top',bbox=dict(fc='lightyellow',ec='gray'))
# ── (2) 축방향 샌드위치 ──
ax=fig.add_subplot(gs[0,1])
# 스택 (link5 +Y 방향으로 바깥): 디스크 플랜지 | 뺨 1.5 | 암 판 3 | 나사 머리 3 ; 핀 보스 → 로드 평면 +10
y0=0; items=[('구동 디스크 플랜지 2.5',-2.5,0,'tab:red'),('뺨 1.5 (여분 조 or PLA 포크)',0,1.5,'gray'),('크랭크 암 판 3.0',1.5,4.5,'tab:blue'),('M3×8 머리 3',4.5,7.5,'saddlebrown')]
for lab,a,b,c in items: ax.add_patch(Rectangle((0,a),40,b-a,fc=c,alpha=.35,ec=c)); ax.text(41,(a+b)/2,lab,va='center',fontsize=10)
ax.add_patch(Rectangle((26,4.5),8,5.5,fc='tab:blue',alpha=.35,ec='tab:blue')); ax.text(41,8.2,'핀 보스 (암과 일체) → 로드 평면',fontsize=10,va='center',color='tab:blue')
ax.add_patch(Rectangle((22,10),16,2.5,fc='tab:purple',alpha=.35,ec='tab:purple')); ax.text(41,11.2,'로드 (현행 설계: 뺨 바깥면에서 +10)',fontsize=10,va='center',color='tab:purple')
ax.add_patch(FancyArrowPatch((-4,1.5),(-4,11.5),arrowstyle='<->',mutation_scale=12,color='k')); ax.text(-8,6.5,'10',ha='right',va='center',fontsize=11)
ax.plot([0,40],[-2.5,-2.5],'k-',lw=.5); ax.text(20,-4.5,'← 서보 몸통 쪽 (link5 안쪽)          바깥 (link5 +Y) →',ha='center',fontsize=9)
ax.set_xlim(-12,95); ax.set_ylim(-6,15); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('② 축방향 샌드위치 (구동 디스크 쪽 = link5 +Y). 토크 경로: 디스크 → 나사 4개 → 암 판 → 핀',fontsize=13)
ax.text(-11,-5.8,'⚠ 로드 평면이 현행 설계(link5 −Y)와 반대쪽으로 옮겨진다 → 링크 층 거울 이동 + p37/p38/롤(G8b) 재검증. 종동 디스크는 헛돌므로 암은 반드시 구동측.',fontsize=9.5,color='darkred')
# ── (3) PLA 포크 안 ──
ax=fig.add_subplot(gs[1,0])
outline(ax,mj,327.25,'k',1.0,'STEP 뺨 윤곽 (복사 대상)')
# 인쇄 포크: 뺨 윤곽을 x≤339 까지만 복사 (허브~뿌리), 두께 3
sec=mj.section(plane_origin=[0,0,327.25],plane_normal=[0,0,1])
for i,d in enumerate(sec.discrete):
    d=d[d[:,0]<=339] if len(d)>50 else d
    if len(d)>10: ax.add_patch(Polygon(d[:,:2],closed=True,fc='tab:green',alpha=.18,ec='tab:green',lw=2,label='인쇄 포크 뺨 (STEP 윤곽 그대로, 3 mm)' if i==0 else None))
ax.add_patch(Rectangle((313,3.6),26,3,fc='tab:green',alpha=.35,ec='tab:green',label='인쇄 다리 (두 뺨 연결, x 313~339)'))
ax.axvline(339,color='tab:green',ls='--',lw=1.5)
crank_arm(ax,txt=False); ax.text(PIN[0]+6,PIN[1]-22,'크랭크 암은 안 1 과 동일 부품\n(포크 뺨에 일체 인쇄해도 됨)',fontsize=9.5,color='tab:blue')
ax.plot(*AX,'k+',ms=14); ax.set_aspect('equal'); ax.set_xlim(272,362); ax.set_ylim(-30,22); ax.grid(alpha=.3); ax.legend(fontsize=8.5,loc='lower left')
ax.set_title('③ 안 2 (대안): PLA 포크 — 여분 조가 M3 디스크에 안 맞을 때',fontsize=13)
ax.text(273,19,'뺨 두께 1.5→3.0 (PLA), 나사 M3×8. 윤곽은 STEP 에서 복사하므로 순정 부피 안 → 간섭 자동 통과.\n출력 1개(포크+암 일체). 무게: 순정 조 9.5 g 빠지고 PLA 약 5 g.',fontsize=9.5,va='top',bbox=dict(fc='honeydew',ec='gray'))
# ── (4) 공정 순서 ──
ax=fig.add_subplot(gs[1,1]); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,100)
steps=[('0','끼워 보기: M2 여분 조 → M3 서보 디스크 (4구멍·폭 39.4)','사용자','lightyellow'),
       ('1','설계 스크립트: 링크 층 +Y 거울 이동, 크랭크 암 판, (안2) 포크','Claude','lightblue'),
       ('2','게이트: p37 장착 · p38 인출 · 0~89° 스윕 · 순정 부피 안 · 핀 위치 · 롤 G8b','Claude','lightblue'),
       ('3','URDF: gripper_link 메쉬 → 포크+암 · USD 재생성 (mimic False)','Claude','lightblue'),
       ('4','Isaac Lab 결합 시뮬 재현, 서보 상한 1.96 N·m','Claude','lightblue'),
       ('5','출력: 크랭크 암 (안2 면 포크 일체) — printing.md 절차','승인 후','mistyrose'),
       ('6','(안1) 여분 조 절단 + 줄 마감','사용자','lightyellow'),
       ('7','조립: 전원 OFF → 순정 조 나사 9개 풀어 보관 → 포크+암 M3×8 로 디스크에 체결 (닫힘 방위 = STEP 자세)','사용자','lightyellow'),
       ('8','부팅(π=닫힘) → T:107 tor 200 → 0~89° 수동 → 구 파지 시험','사용자','lightyellow')]
y=95
for n,t,who,c in steps:
    ax.add_patch(FancyBboxPatch((2,y-8),90,8,boxstyle='round,pad=0.3',fc=c,ec='gray')); ax.text(4,y-4,f'{n}. {t}',va='center',fontsize=10); ax.text(90,y-4,who,va='center',ha='right',fontsize=9,color='dimgray')
    if n!='8': ax.annotate('',(47,y-8.4),(47,y-10.2),arrowprops=dict(arrowstyle='<-',color='k'))
    y-=10.6
ax.set_title('④ 공정 순서 (노랑 = 사용자 손, 파랑 = 제가 지금 진행, 분홍 = 출력 승인 대기)',fontsize=13)
plt.tight_layout(); plt.savefig(D+'p43_plan_A_options.png',dpi=105); print('saved')
