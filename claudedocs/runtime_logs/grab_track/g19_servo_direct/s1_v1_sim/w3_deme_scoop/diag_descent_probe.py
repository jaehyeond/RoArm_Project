"""어디서 터지는가: error-out 을 풀고 최초 >20 m/s 입자를 툴 프레임에서 찍는다. argv1 = fixed|door|both|cyl"""
import sys, math, time, numpy as np, DEME, trimesh
sys.path.insert(0, "."); import sim_deme_scoop_s1 as S
which = sys.argv[1]
import os
P = dict(S.DEFAULT); P["cap_mode"]=os.environ.get("CAP","solid"); P["collision_wall_extra_mm"]=float(os.environ.get("EXTRA","3")); EP=float(os.environ.get("EP","5e6")); EM=float(os.environ.get("EM","3e9")); VZ=float(os.environ.get("VZ","0.05")); NST=int(os.environ.get("NST","175"))
fixed, door, lf, ld, hoff, L5, _chk = S.load_tool(P, 27.5)
if which == "cyl":   # 해석적 반원통 셸(단일 닫힌 메시) — 조각 메시 문제인지 판별
    r_in, r_out, w = 0.020, 0.0246, 0.0182
    th = np.linspace(math.pi/2, 3*math.pi/2, 25)   # 세계: 열림쪽(−Y) 을 향한 반원통. 축 ∥ 세계 X
    ring = lambda r: np.stack([np.zeros_like(th), r*np.cos(th), r*np.sin(th)], 1)
    V=[]; F=[]
    for x in (-w, w):
        for r in (r_in, r_out):
            V.append(ring(r) + [x,0,0])
    V=np.concatenate(V)  # 4 rings of 25: idx (x0,rin)=0.., (x0,rout)=25.., (x1,rin)=50.., (x1,rout)=75..
    def quad(a,b,c,d): F.extend([[a,b,c],[a,c,d]])
    n=25
    for i in range(n-1):
        quad(0+i, 0+i+1, 50+i+1, 50+i)         # inner surface
        quad(25+i, 75+i, 75+i+1, 25+i+1)       # outer surface
        quad(0+i, 25+i, 25+i+1, 0+i+1)         # cap x0 (annulus)
        quad(50+i, 50+i+1, 75+i+1, 75+i)       # cap x1
    quad(0,50,75,25); quad(24,49,99,74)        # end faces (lips)
    M=trimesh.Trimesh(V,np.array(F),process=False); M.fix_normals()
    zmin = M.vertices[:,2].min(); M.vertices[:,2] -= zmin      # 립(최하점)=원점
    fixed = M
z = np.load("claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz", allow_pickle=True)
pos=np.asarray(z["positions_m"]); rad=float(z["radii_m"][0]); box=np.asarray(z["box_bounds_m"]); n_p=len(pos)
s=DEME.DEMSolver(); s.SetVerbosity("ERROR")
mp={"E":EP,"nu":0.3,"CoR":float(os.environ.get("COR","0.3")),"mu":float(os.environ.get("MUP","0.5")),"Crr":float(os.environ.get("CRR","0.05"))}; m=s.LoadMaterial(mp); mm=s.LoadMaterial(dict(mp,E=EM,mu=float(os.environ.get("MUM","0.5")))); (s.UseFrictionlessHertzianModel() if os.environ.get("FRICTIONLESS") else s.UseFrictionalHertzianModel())
s.AddClumps(s.LoadSphereType(950*4/3*math.pi*rad**3, rad, m), pos.tolist())
DT=float(os.environ.get("DTOP","0.30")); BC=os.environ.get("BC","top_open"); s.InstructBoxDomainDimension((box[0,0],box[0,1]),(box[1,0],box[1,1]),(box[2,0],DT)); s.InstructBoxDomainBoundingBC(BC, m)
y_s=-0.0241; near=np.hypot(pos[:,0],pos[:,1]-y_s)<0.022; z_surf=pos[near,2].max()+rad
lip0=np.array([0,y_s,z_surf+0.010]); trks={}
for nm,M,p0 in (("fixed",fixed,lip0),("door",door,lip0+hoff)):
    if which not in ("both", nm) and not (which=="cyl" and nm=="fixed"): continue
    pth=f"/tmp/claude-1000/diag_{which}_{nm}.obj"; M.export(pth)
    me=s.AddWavefrontMeshObject(pth, mm, True, False); me.SetMass(0.02); me.SetMOI([1e-5]*3); me.SetFamily(10); me.SetInitPos(p0.tolist()); trks[nm]=s.Track(me)
dt=0.004; vz=-VZ
s.SetFamilyPrescribedLinVel(10,"0","0",f"(t<{float(os.environ.get('TSET','0.04'))})?0.0:{vz}",True); s.SetFamilyPrescribedAngVel(10,"0","0","0",True)
s.SetInitTimeStep(float(os.environ.get("DT","1e-5"))); s.SetGravitationalAcceleration([0,0,-9.81]); s.SetCDUpdateFreq(int(os.environ.get("CDF","20"))); s.SetErrorOutVelocity(1e4)
if os.environ.get("ADDER"): s.SetExpandSafetyAdder(float(os.environ["ADDER"]))
if os.environ.get("MULT"): s.SetExpandSafetyMultiplier(float(os.environ["MULT"]))
if os.environ.get("NOADAPT"): s.DisableAdaptiveUpdateFreq()
s.Initialize(); print(which, "init ok", {k:int(t.GetOwnerID()) for k,t in trks.items()}, flush=True)
vmax_all=0.0
for i in range(NST):
    s.DoDynamicsThenSync(dt)
    v=np.asarray(s.GetOwnerVelocity(0,n_p)); sp=np.linalg.norm(v,axis=1); j=int(sp.argmax())
    lip=np.asarray(trks["fixed"].Pos()) if "fixed" in trks else np.asarray(trks["door"].Pos())-hoff
    vmax_all=max(vmax_all,float(sp[j]))
    if i%25==0 or sp[j]>5:
        pp=np.asarray(s.GetOwnerPosition(0,n_p)); rel=(pp[j]-lip)*1000; p5=S.R_W.T@rel
        nc=sum(len(t.GetContactForces()[0]) for t in trks.values())
        print(f"step {i:3d} lip z {lip[2]*1000:6.2f} insert {(z_surf-lip[2])*1000:6.2f} mm  vmax {sp[j]:8.3f} m/s  contacts {nc}  fastest@world-lip mm {rel.round(1)}  link5(rel lip) {p5.round(1)}", flush=True)
    if sp[j]>20:
        print("EXPLODE — 상위 5 입자 (link5 rel lip mm, speed):")
        for k in np.argsort(-sp)[:5]:
            print("   ", (S.R_W.T@((pp[k]-lip)*1000)).round(1), round(float(sp[k]),2))
        break
print(f"RESULT EXTRA={P['collision_wall_extra_mm']} EP={EP} EM={EM} VZ={VZ} steps={i+1} insert_end={(z_surf-lip[2])*1000:.1f}mm vmax_all={vmax_all:.3f}")
