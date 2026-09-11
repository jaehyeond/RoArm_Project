"""Plan one outlet-directed tilt from the last issued open release pose.

Offline only. S1 direct drive has no g18 servocrank (design source lines1-5).
The old generic arm limits remain intact; the executor accepts this exact path.
"""
import hashlib, json, math, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from scipy.optimize import least_squares

BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'hw_s1_scoop_probe.py').exists())
sys.path.insert(0,str(ROOT))
import hw_s1_scoop_probe as S
from roarm_rl import viz_debug as V

URDF=ROOT/'local_assets/roarm_m3/urdf/roarm_m3_s1_v1.urdf'
tree=ET.parse(URDF).getroot()
jo=tree.find("joint[@name='link5_to_gripper_link']/origin")
J=S.K.Tmat([float(v) for v in jo.attrib['xyz'].split()],[float(v) for v in jo.attrib['rpy'].split()])
PARTS={}
for name in ['base_link','link1','link2','link3','link4','link5','grab_fixed','gripper_link']:
    v=tree.find(f"link[@name='{name}']/visual")
    node=v.find('geometry/mesh');p=URDF.parent/node.attrib['filename']
    m=trimesh.load(p,force='mesh',process=False)
    origin=v.find('origin')
    O=np.eye(4) if origin is None else S.K.Tmat([float(x) for x in origin.attrib['xyz'].split()],[float(x) for x in origin.attrib['rpy'].split()])
    verts=np.asarray(m.vertices)*np.array([float(x) for x in node.attrib['scale'].split()])
    verts=verts@O[:3,:3].T+O[:3,3]
    PARTS[name]=(verts,np.asarray(m.faces),p)

def geometry(q):
    ch=S.chain(q[:5]);Ts=list(ch.values())
    T=ch['link4_to_link5'].copy();D=T@J@S.K.Trot_z(math.radians(q[5]))
    mats=dict(zip(['base_link','link1','link2','link3','link4','link5'],Ts))
    mats.update(grab_fixed=T,gripper_link=D)
    world={name:verts@mats[name][:3,:3].T+mats[name][:3,3]+[0,0,.38] for name,(verts,faces,p) in PARTS.items()}
    T[2,3]+=.38
    return T,world,mats

def slope(q):
    R=S.chain(q[:5])['link4_to_link5'][:3,:3]
    g=R.T@np.array([0.,0.,-1.])
    a=math.radians(7.5);n=np.array([math.sin(a),0,-math.cos(a)]);v=np.array([math.cos(a),0,math.sin(a)])
    return math.degrees(math.atan2(g@v,-g@n))

def plan(out):
    out.mkdir(exist_ok=False)
    observed=json.loads((out.parent/'feedback_preflight/result.json').read_text())['q_deg']
    reference=np.array([90.,73.25,42.,64.77,-5.])
    assert max(abs(reference-np.array(observed[:5])))<3
    p0=(S.chain(reference)['link4_to_link5']@S.LIP_L5)[:3]
    p_up=p0+[0,0,.05]
    seed=reference.copy();rows=[]
    def solve(name,p,beta,roll):
        nonlocal seed
        def fun(x):
            T=S.chain(np.r_[x,roll])['link4_to_link5']
            return np.r_[((T@S.LIP_L5)[:3]-p)*100,(sum(x[1:])-180-beta)/10]
        r=least_squares(fun,seed[:4],bounds=([-90,-110,-70,-90],[90,110,190,90]),ftol=1e-12,xtol=1e-12,gtol=1e-12)
        q=np.r_[r.x,roll];T=S.chain(q)['link4_to_link5'];err=np.linalg.norm((T@S.LIP_L5)[:3]-p)
        assert err<.00005 and abs(sum(q[1:4])-180-beta)<.001,(name,err,q)
        if np.max(abs(q-seed))>5.0001:raise ValueError(('step too big',name,q,seed))
        seed=q
        c=dict(T=122,**dict(zip(('b','s','e','t','r'),q.tolist())),h=150,spd=200*180/2048,acc=50*180/25400)
        rows.append(dict(name=name,q_deg=[*q,30.],command=c,lip_world_m=((T@S.LIP_L5)[:3]+[0,0,.38]).tolist(),ik_error_m=float(err),outlet_slope_deg=slope(q)))
    for i in range(1,6):solve('raise_'+str(i),p0+[0,0,.01*i],.02,-5.)
    for roll in range(0,91,5):solve('align_'+str(roll),p_up,.02,float(roll))
    for i in range(1,41):solve('tilt_'+str(i),p_up,.02-(20.02*i/40),90.)
    # Certify separation by projection of every mesh vertex. Positive gaps are
    # sufficient separating planes, not approximate nearest-vertex distances.
    gaps={};samples=[];prev=np.r_[reference,30.]
    for row in rows:
        target=np.array(row['q_deg'])
        for alpha in np.linspace(0,1,11):
            q=prev+(target-prev)*alpha
            assert -90<=q[0]<=90 and -110<=q[1]<=110 and -70<=q[2]<=190 and -90<=q[3]<=90 and -5<=q[4]<=90 and q[5]==30
            T,world,mats=geometry(q)
            axes=np.concatenate([np.eye(3),T[:3,:3].T]+[mats[k][:3,:3].T for k in ['link2','link3','link4']],axis=0)
            for tool in ['grab_fixed','gripper_link']:
                A=world[tool]@axes.T
                for body in ['base_link','link1','link2','link3','link4']:
                    B=world[body]@axes.T
                    gap=float(np.max(np.maximum(B.min(0)-A.max(0),A.min(0)-B.max(0))))
                    key=tool+'__'+body;gaps[key]=min(gaps.get(key,100),gap)
                    assert gap>.015,(row['name'],alpha,key,gap)
            lip=(T@S.LIP_L5)[:3]
            zmin=min(world[k][:,2].min() for k in ['grab_fixed','gripper_link'])
            assert zmin>.22
            samples.append(dict(stage=row['name'],alpha=float(alpha),q_deg=q.tolist(),lip_world_m=lip.tolist(),tool_min_z_m=float(zmin),outlet_slope_deg=slope(q)))
        prev=target
    baseline_T,baseline_world,_=geometry(observed)
    z0=min(baseline_world[k][:,2].min() for k in ['grab_fixed','gripper_link'])
    data=dict(experiment='open_residual_outlet_tilt20',new_variables=['outlet-directed tilt profile'],reference_issued_q_deg=[*reference,30.],preflight_q_deg=observed,door_target_deg=30,gripper_torque=200,raise_cm=5,alignment_roll_deg=90,tilt_deg=20,
              path=rows,projection_separation_min_m=gaps,interpolation_samples=len(samples),baseline_observed_tool_min_z_m=float(z0),aligned_tilt_tool_min_z_m=float(min(s['tool_min_z_m'] for s in samples if s['stage'].startswith(('align_','tilt_')))),
              start_policy='current open pose; preserve issued reference, no HOME/no new scoop',end_policy='hold open at tilt20; no automatic return or door close',
              limits=['Cup pose is operator adjusted, not measured; cup is not a registered CAD obstacle.','Sampled nominal path with joint feedback gates; not a continuous collision proof under arbitrary servo deviations.','link5 attachment contact is excluded; gripper open30 fixed, same rigid assembly throughout.','Source kinematics rounds URDF fixed transforms; not external Cartesian measurement.'],
              sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),URDF,ROOT/'scoop_grab_s1_design.py',ROOT/'hw_s1_scoop_probe.py',ROOT/'safety_p0_guards.py',*[v[2] for v in PARTS.values()]]})
    (out/'plan.json').write_text(json.dumps(data,indent=2));(out/'interpolation.json').write_text(json.dumps(samples,indent=2))
    frames=[]
    for name,q,role in [('current_observed',observed,'actual'),('raised_aligned',rows[23]['q_deg'],'target'),('outlet_tilt20',rows[-1]['q_deg'],'target')]:
        T,world,_=geometry(q);frames.append(V.frame_from_axes(name,(T@S.LIP_L5)[:3],x_axis=T[:3,0],z_axis=T[:3,2],role=role,label=name))
    assert V.snapshot_frame_plot(out/'plan_frames.png',frames,title='Open S1: raise 5cm, align yaw, tilt20 toward outlet',annotations=['Joint-FK prediction; no measured cup obstacle.'])['ok']
    print(json.dumps({k:v for k,v in data.items() if k not in ['path','sources']},indent=2))

if __name__=='__main__':plan(Path(sys.argv[1]).resolve())
