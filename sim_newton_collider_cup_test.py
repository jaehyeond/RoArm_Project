"""최소 재현: 열린 컵에 재료를 떨어뜨려 담는다. 컵을 만드는 방식만 바꿔가며 비교.
usage: cup.py <kind> <wall_mm> <margin_mm> [voxel_mm]
  kind: prim  = add_shape_box (해석적 프리미티브)
        mesh  = add_shape_mesh (삼각형 메시, 그랩 셸과 같은 경로)
"""
import sys, numpy as np, warp as wp, newton
from newton.solvers import SolverImplicitMPM

KIND=sys.argv[1]; T=float(sys.argv[2])/1000.; MARGIN=float(sys.argv[3])/1000.
VOX=(float(sys.argv[4])/1000. if len(sys.argv)>4 else 0.005)
IN=0.040; H=0.040; FLOOR=0.050          # 안쪽 40x40, 높이 40, 컵 바닥을 50mm 위에

def box_mesh(hx,hy,hz):
    s=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    v=s*np.array([hx,hy,hz])
    f=[0,2,1,0,3,2, 4,5,6,4,6,7, 0,1,5,0,5,4, 1,2,6,1,6,5, 2,3,7,2,7,6, 3,0,4,3,4,7]
    return v, np.array(f,dtype=np.int32)

builder=newton.ModelBuilder(); SolverImplicitMPM.register_custom_attributes(builder)
sp=VOX/3.; r=sp*0.5
xs=np.arange(-IN/2+sp,IN/2-sp,sp); ys=xs
zs=np.arange(FLOOR+T+0.004, FLOOR+T+0.030, sp)
P=np.array([[x,y,z] for x in xs for y in ys for z in zs])
m=sp**3*950.*0.6
for p in P: builder.add_particle(wp.vec3(*p), wp.vec3(0.,0.,0.), m, radius=r)
builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

b=builder.add_body(xform=wp.transform(wp.vec3(0.,0.,0.),wp.quat_identity()),mass=0.0,is_kinematic=True)
o=IN/2+T
plates=[((o,o,T/2),(0,0,FLOOR+T/2)),
        ((T/2,o,H/2),(-(IN/2+T/2),0,FLOOR+T+H/2)),((T/2,o,H/2),((IN/2+T/2),0,FLOOR+T+H/2)),
        ((o,T/2,H/2),(0,-(IN/2+T/2),FLOOR+T+H/2)),((o,T/2,H/2),(0,(IN/2+T/2),FLOOR+T+H/2))]
cfgs=newton.ModelBuilder.ShapeConfig(mu=0.5,density=0.0,margin=MARGIN)
for (hx,hy,hz),(cx,cy,cz) in plates:
    xf=wp.transform(wp.vec3(cx,cy,cz),wp.quat_identity())
    if KIND=="prim":
        builder.add_shape_box(body=b, xform=xf, hx=hx,hy=hy,hz=hz, cfg=cfgs)
    elif KIND=="mesh":
        vv,ff=box_mesh(hx,hy,hz)
        builder.add_shape_mesh(body=b, xform=xf,
            mesh=newton.Mesh(vv,ff,compute_inertia=False,is_solid=True), cfg=cfgs)
    else:   # "soup": 겹치는 조각들을 한 메시로 합침 (그랩 STL 과 같은 구조)
        pass

if KIND=="soup":
    # 5장을 겹치게(각 판을 2mm 씩 키워 서로 파고들게) 만들어 한 메시로 합친다 -> 내부면 발생
    allv=[]; allf=[]; off=0
    for (hx,hy,hz),(cx,cy,cz) in plates:
        vv,ff=box_mesh(hx+0.002,hy+0.002,hz+0.002)
        allv.append(vv+np.array([cx,cy,cz])); allf.append(ff+off); off+=len(vv)
    V=np.vstack(allv); F=np.concatenate(allf)
    E=np.sort(np.stack([F.reshape(-1,3)[:,[0,1]],F.reshape(-1,3)[:,[1,2]],F.reshape(-1,3)[:,[2,0]]]).reshape(-1,2),axis=1)
    _,c=np.unique(E,axis=0,return_counts=True)
    print(f"  [soup] 삼각형 {len(F)//3} | 3면이상 공유 에지 {int((c>2).sum())} (그랩 STL 은 91)")
    builder.add_shape_mesh(body=b, mesh=newton.Mesh(V,F,compute_inertia=False,is_solid=True), cfg=cfgs)
model=builder.finalize(); model.set_gravity(wp.vec3(0.,0.,-9.81))
opt=SolverImplicitMPM.Config(); opt.voxel_size=VOX; opt.collider_velocity_mode="backward"
model.mpm.friction.fill_(0.68)
s0,s1=model.state(),model.state()
solver=SolverImplicitMPM(model,config=opt)
solver.setup_collider(body_mass=wp.zeros_like(model.body_mass), body_q=s0.body_q)
for _ in range(120):
    solver.step(s0,s1,None,None,1/60.); solver.project_outside(s1,s1,1/60.); s0,s1=s1,s0
q=s0.particle_q.numpy()
held=int((q[:,2]>FLOOR).sum())
print(f"{KIND:4s} wall={T*1e3:4.1f}mm margin={MARGIN*1e3:4.1f}mm vox={VOX*1e3:4.1f}mm N={len(P):5d} "
      f"| 컵 안 잔류 {held:5d} ({100*held/len(P):5.1f}%) | z {q[:,2].min()*1e3:8.1f}~{q[:,2].max()*1e3:6.1f}mm")
