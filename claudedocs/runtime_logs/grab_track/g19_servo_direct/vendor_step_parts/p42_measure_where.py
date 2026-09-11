"""p42: '어디를 재라는 건지' — 사용자 사진 위에 번호 표시 + STEP 단면에 같은 번호. (사진 = ~/Downloads/KakaoTalk_20260903_151540091*.jpg)"""
import trimesh, numpy as np, warnings, os; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fm.fontManager.addfont(_fp); plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name(); plt.rcParams['axes.unicode_minus']=False
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image, ImageOps
D=os.path.dirname(os.path.abspath(__file__))+'/'; PH='/home/cgxr/Downloads/'; AX=(289.002,-0.88)
def load(f,box,size):
    im=ImageOps.exif_transpose(Image.open(PH+f)); w,h=im.size
    return im.crop((int(w*box[0]),int(h*box[1]),int(w*box[2]),int(h*box[3]))).resize(size)
# ── 그림 A: 가동 조 쪽 (사진 = 구동측 정면, 닫힘) + STEP 축방향 단면 ──
fig,axs=plt.subplots(1,2,figsize=(19,9))
ax=axs[0]; im=load('KakaoTalk_20260903_151540091.jpg',(0.30,0.05,0.80,0.62),(1000,1140)); ax.imshow(im); ax.axis('off')
# 좌표 (1000x1140 크롭 기준): 블레이드 팁 ≈ (600,215), 허브 ≈ (530,900)
ax.add_patch(Circle((600,215),70,fill=False,color='yellow',lw=3)); ax.text(690,200,'① 두 블레이드 끝\n   판 두께 (STEP 1.5)\n   닫힘 간극 (STEP 4.05)\n   → 캘리퍼스',color='yellow',fontsize=13,fontweight='bold',va='center',bbox=dict(fc='black',alpha=.55,ec='none'))
ax.add_patch(Circle((423,690),125,fill=False,color='cyan',lw=3)); ax.text(600,690,'② 허브: 카메라 방향(축 방향)으로\n   양쪽 뺨 바깥면 사이 폭 (STEP 39.4)\n   나사 머리 피해서 캘리퍼스',color='cyan',fontsize=13,fontweight='bold',va='center',bbox=dict(fc='black',alpha=.55,ec='none'))
ax.text(20,1110,'사진: 구동측(ST3215-HS 라벨 쪽) 정면, 닫힘 상태',color='white',fontsize=11,bbox=dict(fc='black',alpha=.6,ec='none'))
ax.set_title('A. 가동 조 — 어디를 재나 (사진)',fontsize=14)
ax=axs[1]
parts={'movable_jaw':'k','gripper_servo_drive_disc':'tab:red','gripper_servo_driven_disc':'tab:orange','gripper_servo_gear_GE27':'tab:purple','gripper_servo_case_SG':'gray','gripper_servo_case_ZK':'silver','gripper_servo_case_XG':'darkgray','gripper_base':'tab:blue','screws_M3x4_disc':'tab:brown'}
for k,col in parts.items():
    sec=trimesh.load(D+k+'.stl',force='mesh').section(plane_origin=[0,AX[1],0],plane_normal=[0,1,0])
    if sec is None: continue
    for dd in sec.discrete: ax.plot(dd[:,0],dd[:,2],'-',color=col,lw=1.1)
ax.add_patch(FancyArrowPatch((310,326.5),(310,328.0),arrowstyle='<->',mutation_scale=12,color='goldenrod',lw=2)); ax.text(312,325.2,'① 뺨 두께 1.5',color='goldenrod',fontsize=12,fontweight='bold',va='center')
ax.add_patch(FancyArrowPatch((335,326.5),(335,365.87),arrowstyle='<->',mutation_scale=14,color='tab:cyan',lw=2)); ax.text(336,346,'② 바깥 폭 39.37',color='tab:cyan',fontsize=12,fontweight='bold',va='center')
ax.add_patch(FancyArrowPatch((300,327.12),(300,329.62),arrowstyle='<->',mutation_scale=10,color='tab:red',lw=1.5)); ax.text(252,321.5,'빨강 = 구동 디스크(플랜지 2.5). CAD 에서 뺨과 0.88 겹침은 도면 오차.\n포크는 뺨 안쪽면 위치를 복사하므로 이 값은 재지 않아도 됨 (항목 ③ 삭제)',color='tab:red',fontsize=9,va='top')
ax.axvline(AX[0],ls='--',c='k',lw=.6); ax.set_aspect('equal'); ax.set_xlim(250,345); ax.set_ylim(318,372)
ax.set_xlabel('link5 Z 방향 (팔축, mm)'); ax.set_ylabel('힌지축 방향 (mm)'); ax.set_title('B. 같은 번호 — 벤더 STEP 단면 (서보축 평면)',fontsize=14); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(D+'p42_measure_where_A.png',dpi=110); plt.close()
# ── 그림 B: 고정 조 구멍 (사진 = LED 쪽 바깥면 팁 확대) + STEP 구멍 지도 ──
fig,axs=plt.subplots(1,2,figsize=(19,8.5))
ax=axs[0]; im=load('KakaoTalk_20260903_151540091_05.jpg',(0.0,0.55,0.55,0.95),(1100,800)); ax.imshow(im); ax.axis('off')
for (x,y),lab,col in (((160,425),'팁 구멍 Z116 ★',"magenta"),((365,435),'Z103 쌍 (아래쪽) ★',"magenta"),((305,255),'Z103 쌍 (위쪽) ★',"magenta")):
    ax.add_patch(Circle((x,y),28,fill=False,color=col,lw=3)); ax.text(x+35,y-40,lab,color=col,fontsize=12,fontweight='bold',bbox=dict(fc='black',alpha=.55,ec='none'))
ax.text(620,120,'Z83.7 쌍은 LED 램프 바로 옆\n(사진에서 램프에 가림)',color='white',fontsize=11,bbox=dict(fc='black',alpha=.6,ec='none'))
ax.text(20,770,'⑥ 이 구멍들: M3 볼트를 끼워 본다. 헐겁게 통과하면 나사산 없음(STEP: 관통 지름 3.2). 걸리면 탭.',color='yellow',fontsize=12,fontweight='bold',bbox=dict(fc='black',alpha=.6,ec='none'))
ax.set_title('C. 고정 조 바깥면(LED 쪽) 팁 확대 — 브래킷이 쓰는 구멍 3개 ★',fontsize=14)
ax=axs[1]; ax.imshow(Image.open(D+'p41_fixed_jaw_holes_map.png')); ax.axis('off'); ax.set_title('D. 같은 구멍 — 벤더 STEP 지도 (바깥면에서 본 것)',fontsize=14)
plt.tight_layout(); plt.savefig(D+'p42_measure_where_B.png',dpi=110); print('saved A,B')
