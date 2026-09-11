"""p40: 벤더 STEP 부품 STL 로 순정 그리퍼 조립 단면도 2장. (1) 서보축 평면(y=-0.88) XZ 단면 = 축방향 스택 (2) 허브 정면 XY = 구멍 패턴.
좌표 = STEP 루트 (X 팔축=link5 +Z, Y 블레이드 법선=link5 -X, Z 힌지축=link5 -Y). 서보축 (289.002, -0.880)."""
import trimesh, numpy as np, json, warnings, os; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D=os.path.dirname(os.path.abspath(__file__))+'/'; AX=(289.002,-0.88)
parts={'movable_jaw':('movable jaw (fork)','k'),'gripper_servo_drive_disc':('drive disc','tab:red'),'gripper_servo_driven_disc':('driven (idler) disc','tab:orange'),
       'gripper_servo_gear_GE27':('servo output gear','tab:purple'),'gripper_servo_case_SG':('servo case (bottom)','tab:gray'),'gripper_servo_case_ZK':('servo case (mid)','silver'),
       'gripper_servo_case_XG':('servo case (top)','darkgray'),'gripper_base':('gripper base = link5','tab:blue'),'fixed_jaw':('fixed jaw','tab:green'),
       'screws_M3x4_disc':('M3x4 x9','tab:brown'),'screws_PA2x5_servo_case':('PA2x5 case screws','tab:pink'),'wrist_b':('wrist b','tab:cyan')}
M={k:trimesh.load(D+k+'.stl',force='mesh') for k in parts}
fig,axs=plt.subplots(1,2,figsize=(18,8.5))
ax=axs[0]
for k,(lab,col) in parts.items():
    sec=M[k].section(plane_origin=[0,AX[1],0],plane_normal=[0,1,0])
    if sec is None: continue
    for i,d in enumerate(sec.discrete): ax.plot(d[:,0],d[:,2],'-',color=col,lw=1.1,label=lab if i==0 else None)
ax.axhline(326.5,ls=':',c='k',lw=.6); ax.axhline(365.87,ls=':',c='k',lw=.6); ax.axvline(AX[0],ls='--',c='k',lw=.6)
ax.set_aspect('equal'); ax.set_xlim(225,345); ax.set_ylim(318,375); ax.set_xlabel('X = link5 +Z (arm axis, mm)'); ax.set_ylabel('Z = link5 -Y (hinge axis, mm)')
ax.set_title('stock gripper axial stack: section through servo axis plane (y=-0.88)'); ax.legend(fontsize=8,loc='upper left'); ax.grid(alpha=.3)
ax=axs[1]
sec=M['movable_jaw'].section(plane_origin=[0,0,327.25],plane_normal=[0,0,1])
for i,d in enumerate(sec.discrete): ax.plot(d[:,0],d[:,1],'k-',lw=1.2,label='movable jaw drive-side cheek (z=327.25)' if i==0 else None)
sec=M['gripper_servo_drive_disc'].section(plane_origin=[0,0,328.5],plane_normal=[0,0,1])
for i,d in enumerate(sec.discrete): ax.plot(d[:,0],d[:,1],'-',color='tab:red',lw=1,label='drive disc (z=328.5)' if i==0 else None)
sec=M['screws_M3x4_disc'].section(plane_origin=[0,0,329.5],plane_normal=[0,0,1])
for i,d in enumerate(sec.discrete): ax.plot(d[:,0],d[:,1],'-',color='tab:brown',lw=1,label='M3x4 screws (z=329.5)' if i==0 else None)
th=np.linspace(0,2*np.pi,200); ax.plot(AX[0]+7*np.cos(th),AX[1]+7*np.sin(th),'--',c='tab:red',lw=.8,label='PCD 14')
ax.plot(AX[0]+21*np.cos(th),AX[1]+21*np.sin(th),'--',c='tab:blue',lw=.8,label='crank radius 21')
px,py=AX[0]+20.186,AX[1]+5.789; ax.plot(px,py,'o',c='tab:blue',ms=8,label='crank pin (closed pose, current design)')
ax.plot(*AX,'k+',ms=12); ax.set_aspect('equal'); ax.set_xlim(265,335); ax.set_ylim(-30,25)
ax.set_xlabel('X = link5 +Z (mm)'); ax.set_ylabel('Y = link5 -X (mm)  [fixed jaw side = +Y]'); ax.set_title('hub front view: cheek hole pattern + disc (closed pose)'); ax.legend(fontsize=8,loc='lower right'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(D+'p40_assembly_sections.png',dpi=120); print('saved')
