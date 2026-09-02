"""닫힌 그랩 vs 같은 크기의 두꺼운 상자 컵. 재료를 가두는가?
입자는 셸에서 3mm 이상 떨어진 곳에만 놓아 초기 관통을 배제한다."""
import math, sys, numpy as np, warp as wp, newton
from newton.solvers import SolverImplicitMPM
sys.path.insert(0,"/home/cgxr/orca/workspaces/RoArm_Project/pellet-model")
import sim_newton_scoop_probe as b1

MODE = sys.argv[1]; VOX = float(sys.argv[2])
v0,_ = b1.load_stl_mm(b1.SHELL_L); gap=0.026
zp = math.sqrt(0.038332**2-(gap/2)**2)+float(v0[:,2].min())
lift = 0.030
z_body = lift - (float(v0[:,2].min())-zp)

builder = newton.ModelBuilder(); SolverImplicitMPM.register_custom_attributes(builder)
sp=VOX/3.0; r=sp*0.5

# 셸 정점(월드) 모으기 -> 입자를 이들로부터 3mm 이상 떨어뜨린다
shell_world=[]
for side,path in ((-1,b1.SHELL_L),(+1,b1.SHELL_R)):
    vv,_=b1.load_stl_mm(path); piv=np.array([side*gap/2,0.0,zp])
    shell_world.append(vv-piv+np.array([side*gap/2,0.0,z_body]))
SW=np.vstack(shell_world)

xs=np.arange(-0.012,0.012,sp); ys=np.arange(-0.020,0.000,sp); zs=np.arange(lift+0.006,lift+0.022,sp)
cand=np.array([[x,y,z] for x in xs for y in ys for z in zs])
d=np.sqrt(((cand[:,None,:]-SW[None,:,:])**2).sum(-1)).min(1)
P=cand[d>0.003]
m=sp**3*950.0*0.6
for p in P: builder.add_particle(wp.vec3(*p), wp.vec3(0.,0.,0.), m, radius=r)
builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

if MODE=="shell":
    for side,path in ((-1,b1.SHELL_L),(+1,b1.SHELL_R)):
        vv,ii=b1.load_stl_mm(path); piv=np.array([side*gap/2,0.0,zp])
        b=builder.add_body(xform=wp.transform(wp.vec3(float(side*gap/2),0.,float(z_body)),wp.quat_identity()),
                           mass=0.0,is_kinematic=True)
        builder.add_shape_mesh(body=b, mesh=newton.Mesh(vv-piv,ii,compute_inertia=False,is_solid=False),
                               cfg=newton.ModelBuilder.ShapeConfig(mu=0.5,density=0.0))
else:  # 대조군: 두꺼운 판 5장으로 만든 컵 (벽 6mm). 안쪽 치수는 셸 보울과 비슷하게.
    t=0.006; ix,iy,iz=0.020,0.016,0.020   # 안쪽 반치수/높이
    b=builder.add_body(xform=wp.transform(wp.vec3(0.,0.,0.),wp.quat_identity()),mass=0.0,is_kinematic=True)
    plates=[((ix+t,iy+t,t/2),(0,-0.010,lift-t/2)),
            ((t/2,iy+t,iz/2),(-(ix+t/2),-0.010,lift+iz/2)),((t/2,iy+t,iz/2),((ix+t/2),-0.010,lift+iz/2)),
            ((ix+t,t/2,iz/2),(0,-0.010-(iy+t/2),lift+iz/2)),((ix+t,t/2,iz/2),(0,-0.010+(iy+t/2),lift+iz/2))]
    for (hx,hy,hz),(cx,cy,cz) in plates:
        builder.add_shape_box(body=b, xform=wp.transform(wp.vec3(cx,cy,cz),wp.quat_identity()),
                              hx=hx,hy=hy,hz=hz, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5,density=0.0))

model=builder.finalize(); model.set_gravity(wp.vec3(0.,0.,-9.81))
o=SolverImplicitMPM.Config(); o.voxel_size=VOX; o.collider_velocity_mode="backward"
model.mpm.friction.fill_(0.68)
s0,s1=model.state(),model.state()
solver=SolverImplicitMPM(model,config=o)
solver.setup_collider(body_mass=wp.zeros_like(model.body_mass), body_q=s0.body_q)
for k in range(120):
    solver.step(s0,s1,None,None,1/60.); solver.project_outside(s1,s1,1/60.); s0,s1=s1,s0
    if k==0:
        vmax=float(np.linalg.norm(s0.particle_qd.numpy(),axis=1).max())
q=s0.particle_q.numpy()
leak=int((q[:,2]<lift-0.003).sum())
print(f"{MODE:6s} voxel={VOX*1e3:4.1f}mm  N={len(P):6d}  1스텝후 최대속도 {vmax:7.2f} m/s  "
      f"유출 {leak:6d} ({100*leak/max(len(P),1):5.1f}%)  최종 z {q[:,2].min()*1e3:9.1f}~{q[:,2].max()*1e3:6.1f} mm")
