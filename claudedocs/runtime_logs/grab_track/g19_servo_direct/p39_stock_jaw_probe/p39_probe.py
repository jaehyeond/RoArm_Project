"""p39: 순정 가동 조(gripper_link.stl)·link5 의 그리퍼 축 주변 실기하 프로브 (방식 A 사전 조사).
좌표: gripper_link 프레임 (z = 힌지축 = link5 +Y, link5_Y = 18.821 + z; x = link5 +Z 방향(θ=0), y = link5 -X 방향)."""
import numpy as np, trimesh, json, hashlib, warnings; warnings.filterwarnings('ignore')
from shapely.geometry import Point
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
M='local_assets/roarm_m3/urdf/meshes/'; OUT='claudedocs/runtime_logs/grab_track/g19_servo_direct/p39_stock_jaw_probe/'
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
g=trimesh.load(M+'gripper_link.stl', force='mesh'); l5=trimesh.load(M+'link5.stl', force='mesh')
res={'source':{'gripper_link.stl':sha(M+'gripper_link.stl'),'link5.stl':sha(M+'link5.stl')},'frame':'gripper_link (z=hinge axis=link5 +Y, link5_Y=18.821+z; x=link5 +Z at θ=0; y=link5 +X)','cheeks':[]}
fig,axs=plt.subplots(1,3,figsize=(16,5.5))
for ax,(z,lab) in zip(axs[:2],[(-37.8,'cheek -Y (link5 Y≈-19)'),(0.0,'cheek +Y (link5 Y≈+18.8)')]):
    sec=g.section(plane_origin=[0,0,z], plane_normal=[0,0,1]); p2,_=sec.to_2D()
    info={'z':z,'link5_Y':round(18.821+z,2),'polys':[]}
    for poly in p2.polygons_full:
        ext=np.array(poly.exterior.coords); ax.plot(ext[:,0],ext[:,1],'k-',lw=1.2)
        d={'exterior_bounds':[round(v,2) for v in poly.bounds],'area':round(poly.area,1),'contains_axis':bool(poly.contains(Point(0,0))),'holes':[]}
        for h in poly.interiors:
            hb=np.array(h.coords); ax.plot(hb[:,0],hb[:,1],'r-',lw=1)
            rr=np.hypot(hb[:,0],hb[:,1]); cx,cy=hb[:,0].mean(),hb[:,1].mean()
            d['holes'].append({'centre':[round(cx,2),round(cy,2)],'r_from_axis_min':round(rr.min(),2),'r_from_axis_max':round(rr.max(),2),'bbox':[round(v,2) for v in (hb[:,0].min(),hb[:,1].min(),hb[:,0].max(),hb[:,1].max())],'contains_axis':bool(Point(0,0).within(__import__('shapely.geometry',fromlist=['Polygon']).Polygon(hb)))})
        info['polys'].append(d)
    # vertices near axis: angular coverage of bore
    v=sec.vertices; r=np.hypot(v[:,0],v[:,1]); m=r<6.0
    if m.any():
        ang=np.degrees(np.arctan2(v[m,1],v[m,0])); info['bore']={'r_min':round(r[m].min(),2),'r_max':round(r[m].max(),2),'n':int(m.sum()),'angle_span_deg':[round(ang.min(),1),round(ang.max(),1)]}
    res['cheeks'].append(info)
    ax.add_patch(plt.Circle((0,0),21.0,fill=False,ls='--',color='tab:blue')); ax.plot([0],[0],'b+',ms=12)
    # 크랭크 핀: 그랩 로컬 (서보축 기준 r21 @-106°) -> link5 (dX=localX, dZ=-localY) -> jaw 프레임 (x=link5 +Z, y=link5 +X)
    a0=np.radians(-106.0); dX,dZ=21*np.cos(a0),-21*np.sin(a0); ax.plot([dZ],[dX],'bo',ms=7,label=f'crank pin r21 (jaw frame x={dZ:.2f}, y={dX:.2f}; 실제 평면 link5 Y≈-29.7)')
    ax.set_aspect('equal'); ax.set_title(f'gripper_link section z={z}  {lab}'); ax.set_xlabel('x (=link5 +Z at θ=0)'); ax.set_ylabel('y (=link5 +X)'); ax.grid(alpha=.3)
axs[0].legend(loc='lower right',fontsize=8)
# blade: which z-range and x-range
zz=np.arange(-37,0,0.5); blade=[]
for z in zz:
    sec=g.section(plane_origin=[0,0,z], plane_normal=[0,0,1])
    if sec is None: continue
    v=sec.vertices; blade.append((z, v[:,0].min(), v[:,0].max(), v[:,1].min(), v[:,1].max()))
blade=np.array(blade); res['blade']={'z_range':[round(blade[:,0].min(),1),round(blade[:,0].max(),1)],'link5_Y_range':[round(18.821+blade[:,0].min(),1),round(18.821+blade[:,0].max(),1)],'y_range(thk)':[round(blade[:,3].min(),2),round(blade[:,4].max(),2)],'x_root_min':round(blade[:,1].min(),2),'x_tip_max':round(blade[:,2].max(),2)}
# link5 side faces at the axis
ax=axs[2]; axp=np.array([0,18.821,52.035]); res['link5']=[]
for y,c in ((-17.5,'tab:red'),(17.5,'tab:green'),(0.0,'gray')):
    sec=l5.section(plane_origin=[0,y,0], plane_normal=[0,1,0])
    if sec is None: continue
    v=sec.vertices; r=np.hypot(v[:,0]-axp[0],v[:,2]-axp[2]); m=r<14
    d={'y':y,'n_near':int(m.sum())}
    if m.any(): d.update({'r_min':round(r[m].min(),2),'r_max':round(r[m].max(),2)}); ax.plot(v[m,0]-axp[0],v[m,2]-axp[2],'.',ms=3,color=c,label=f'link5 section y={y}')
    res['link5'].append(d)
ax.plot([0],[0],'k+',ms=12); ax.set_aspect('equal'); ax.set_title('link5 around gripper axis (x, z−52.035)'); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT+'p39_stock_jaw_sections.png',dpi=130)
json.dump(res,open(OUT+'p39_results.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))
