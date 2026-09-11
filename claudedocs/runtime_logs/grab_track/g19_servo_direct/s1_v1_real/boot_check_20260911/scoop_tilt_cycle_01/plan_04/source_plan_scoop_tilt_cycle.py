"""Combine previously executed targets: recover, HOME, one790 scoop, outlet20.

Offline plan only. No policy learning, PID changes, or generic limit expansion.
"""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
import plan_outlet_tilt as P
from roarm_rl import viz_debug as V

BASE=Path(__file__).resolve().parent

def make(out,resume=False,loaded=False):
    out.mkdir(exist_ok=False)
    oldp=BASE/'outlet_tilt_01/plan_01/plan.json';old=json.loads(oldp.read_text())
    src=BASE/'torque790_01/raw.jsonl'
    q0=json.loads((out.parent/('feedback_preflight_03/result.json' if loaded else 'feedback_preflight_02/result.json' if resume else 'feedback_preflight/result.json')).read_text())['q_deg']
    steps=[]
    def add(name,c,wait=0):steps.append(dict(name=name,command=c,wait_after_s=wait))
    def move(name,q5,door=0,wait=0):
        add(name,dict(T=122,**dict(zip(('b','s','e','t','r'),q5)),h=180-door,spd=200*180/2048,acc=50*180/25400),wait)
    add('recover_begin',{'T':107,'tor':200})
    # All of this reverse path was previously tested, including the first
    # reverse target; current fresh feedback is again inside the unchanged5deg.
    reference=old['path'][-1]['q_deg']
    if not resume and not loaded:assert max(abs(a-b) for a,b in zip(q0[:5],reference[:5]))<=5
    for row in reversed(old['path'][5:-1]):add('recover_'+row['name'],row['command'])
    add('recover_close_empty',dict(T=121,joint=6,angle=180.,spd=200*180/2048,acc=50*180/25400),.5)
    move('recover_raise45',[90.,45.,36.,90.,0.])
    move('recover_retract',[90.,.7,91.3,88.,0.])
    for base in [60.,30.,0.]:move('recover_base'+str(int(base)),[base,.7,91.3,88.,0.])
    move('home',[0.,0.,90.,0.,0.],wait=1.)
    # Replay the same approved scoop/transport targets, stopping at cup opening.
    # Early arm moves now hold explicit door0, rather than historical measured
    # door angles before the old trial's first explicit T121.
    for line in src.read_text().splitlines():
        e=json.loads(line)
        if e['ev']!='tx' or e['command']['T']==105:continue
        c=e['command'].copy();name=e['phase']
        if c['T']==122 and name in ['p1','above','5cm_above','surface']:c['h']=180.
        if name=='place':name='place_'+str(len(steps))
        extra=1. if e['phase'] in ['close','lift8'] and c['T'] in [121,122] else .0
        add(name,c,extra)
        if e['phase']=='place' and c['T']==121 and c['angle']==150.:
            steps[-1]['name']='outlet_baseline';steps[-1]['wait_after_s']=1.5
            break
    for row in old['path']:add(row['name'],row['command'])
    retreat=None
    if resume:
        failed=out.parent/'execution_01/raw.jsonl'
        ee=[json.loads(x) for x in failed.read_text().splitlines()]
        actual_tx=[e for e in ee if e['ev']=='tx' and e['command']['T']==122]
        last_c=actual_tx[-1]['command'];passed=next(e['command'] for e in reversed(actual_tx) if e['phase']=='recover_align_20')
        reference=[last_c[k] for k in ['b','s','e','t','r']]+[180-last_c['h']]
        anchor=[passed[k] for k in ['b','s','e','t','r']]
        remaining=steps[next(i for i,s in enumerate(steps) if s['name']=='recover_close_empty'):]
        steps=[]
        for roll in [15.,10.,5.,0.]:move('recover_fixed_arm_roll'+str(int(roll)),anchor[:4]+[roll],door=30)
        first=steps[0]['command'];first_q=[first[k] for k in ['b','s','e','t','r']]
        assert max(abs(a-b) for a,b in zip(first_q,reference[:5]))<=1
        assert max(abs(a-b) for a,b in zip(first_q,q0[:5]))<=5
        steps.extend(remaining)
        retreat=dict(previously_passed_q5=anchor,source=str(failed),previously_passed_phase='recover_align_20',reason='Observed5.107deg error during point-preserving empty return; retain last passed arm target while removing roll.')
    loaded_retract=None
    if loaded:
        src_loaded=out.parent/'execution_02/raw.jsonl'
        ee=[json.loads(x) for x in src_loaded.read_text().splitlines()]
        last_c=[e['command'] for e in ee if e['ev']=='tx' and e['command']['T']==122][-1]
        reference=[last_c[k] for k in ['b','s','e','t','r']]+[180-last_c['h']]
        assert reference==[0,62,38.5,79.52,0,0]
        remaining=steps[next(i for i,s in enumerate(steps) if s['name']=='common_transport_790'):]
        steps=[];candidate=reference[:5].copy();candidate[1]+=.25;candidate[3]+=4
        move('loaded_retract',candidate,door=0)
        assert max(abs(a-b) for a,b in zip(q0[:5],candidate))<=5
        T0,_,_=P.geometry(reference);T1,_,_=P.geometry(candidate+[0])
        shift=((T1@P.S.LIP_L5)[:3]-(T0@P.S.LIP_L5)[:3])*1000
        assert shift[0]<-15 and shift[2]>=0
        loaded_retract=dict(source=str(src_loaded),nominal_lip_shift_mm=shift.tolist(),reason='Observed lift8 shoulder5.148deg error; keep issued base/elbow/roll and closed door, shorten reach with wrist+4deg and shoulder+0.25deg. No PID change.')
        steps.extend(remaining)
    # Exact command allowlist; door target remains explicit across all moves.
    q=np.array(q0);samples=[];minima={};max_step=0
    pairs=[(a,b) for a in ['grab_fixed','gripper_link'] for b in ['base_link','link1','link2','link3','link4']]
    pairs += [('link4','link2'),('link5','link2'),('link5','link3')]
    for st in steps:
        c=st['command'];target=q.copy()
        if c['T']==122:target=np.array([c[k] for k in ('b','s','e','t','r')]+[180-c['h']])
        elif c['T']==121:target[5]=180-c['angle']
        else:continue
        for x,(lo,hi) in zip(target,[(-90,90),(-110,110),(-70,190),(-90,90),(-5,90),(0,30)]):assert math.isfinite(x) and lo<=x<=hi
        n=max(11,int(np.ceil(max(abs(target-q))/2))+1)
        for a in np.linspace(0,1,n):
            qq=q+(target-q)*a;T,w,ms=P.geometry(qq)
            axes=np.concatenate([np.eye(3)]+[m[:3,:3].T for m in ms.values()])
            projected={k:v@axes.T for k,v in w.items()}
            for x,y in pairs:
                A=projected[x];B=projected[y]
                gap=float(np.max(np.maximum(A.min(0)-B.max(0),B.min(0)-A.max(0))))
                key=x+'__'+y;minima[key]=min(minima.get(key,99),gap)
                assert gap>.015,(st['name'],a,key,gap)
            zmin=min(w[k][:,2].min() for k in ['grab_fixed','gripper_link'])
            assert zmin>.22,(st['name'],zmin)
            samples.append(dict(stage=st['name'],alpha=float(a),q_deg=qq.tolist(),tool_min_z_m=float(zmin),outlet_slope_deg=P.slope(qq)))
        q=target;st['q_deg']=target.tolist()
    plan=dict(experiment='home_scoop790_outlet20_repeat',close_torque=790,steps=steps,preflight_q_deg=q0,reference_issued_q_deg=reference,
              previous_residue_count_user_reported=0,initial_target_door_deg=0 if loaded else 30,finish='hold open after outlet20; no unobserved automatic return',new_scoop=not loaded,
              interpolation_samples=len(samples),separating_projection_min_m=minima,scoop_path_source=str(src),tilt_path_source=str(oldp),
              new_variables=['connect verified outlet tilt to existing scoop'],pid_unchanged=True,tracking_gate_deg=5,
              source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),oldp,src,Path(P.__file__)]},
              limits=['Cup placement and pellet pile reset are not independently observed.','Projection separation excludes intentional link5/attachment contacts; sampled nominal FK only.','The previous20deg endpoint triggered5.0146deg gate initially; user reported empty jaw after hold. Same gate and profile retained.'])
    if retreat:plan['bounded_initial_retreat']=retreat
    if loaded_retract:plan['loaded_entry_retract']=loaded_retract
    (out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'interpolation.json').write_text(json.dumps(samples,indent=2))
    frames=[]
    for name,qq,role in [('current',q0,'actual'),('home',[0,0,90,0,0,0],'target'),('outlet20',q.tolist(),'target')]:
        T,_,_=P.geometry(qq);frames.append(V.frame_from_axes(name,(T@P.S.LIP_L5)[:3],x_axis=T[:3,0],z_axis=T[:3,2],role=role,label=name))
    assert V.snapshot_frame_plot(out/'plan_frames.png',frames,title='Current open tilt -> HOME -> scoop -> outlet20')['ok']
    print(json.dumps({k:v for k,v in plan.items() if k not in ['steps','source_sha256','limits']},indent=2))

if __name__=='__main__':make(Path(sys.argv[1]).resolve(),resume='--resume' in sys.argv,loaded='--loaded' in sys.argv)
