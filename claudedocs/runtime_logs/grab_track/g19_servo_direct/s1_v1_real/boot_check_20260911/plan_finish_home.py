"""Offline finish current residual discharge and return HOME; no new scoop."""
import json,hashlib,sys,math
from pathlib import Path
import numpy as np
import plan_outlet_tilt as P
from roarm_rl import viz_debug as V
BASE=Path(__file__).resolve().parent
case=BASE/'scoop_tilt_cycle_01';out=case/'plan_06';out.mkdir(exist_ok=False)
prior=json.loads((case/'execution_04/plan.json').read_text())
q0=json.loads((case/'feedback_preflight_05/result.json').read_text())['q_deg']
anchor=next(s['command'] for s in prior['steps'] if s['name']=='raise_1')
failed=next(s['command'] for s in prior['steps'] if s['name']=='raise_2')
ref=[failed[k] for k in ['b','s','e','t','r']]+[30]
q=[anchor[k] for k in ['b','s','e','t','r']]+[30]
assert max(abs(a-b) for a,b in zip(q0[:5],q[:5]))<=5
assert max(abs(a-b) for a,b in zip(ref[:5],q[:5]))<2
steps=[]
def move(name,qq,wait=0):
 c=dict(T=122,**dict(zip(('b','s','e','t','r'),qq[:5])),h=180-qq[5],spd=200*180/2048,acc=50*180/25400)
 steps.append(dict(name=name,command=c,wait_after_s=wait,q_deg=list(qq)))
move('return_passed_raise1',q)
steps.append(dict(name='outlet_baseline',command={'T':105},wait_after_s=.5))
for roll in range(0,91,5):move('align_'+str(roll),q[:4]+[float(roll),30.])
for i in range(1,9):move('finish_tilt_'+str(i),q[:3]+[q[3]-2.5*i,90.,30.])
tiltq=steps[-1]['q_deg']
steps.append(dict(name='outlet_after_tilt',command={'T':105},wait_after_s=3.))
for i in range(7,-1,-1):move('untilt_'+str(i),q[:3]+[q[3]-2.5*i,90.,30.])
for roll in range(85,-1,-5):move('unalign_'+str(roll),q[:4]+[float(roll),30.])
steps.append(dict(name='close_empty',command=dict(T=121,joint=6,angle=180.,spd=200*180/2048,acc=50*180/25400),wait_after_s=.5))
move('return_upright',[90.,.7,91.3,88.,0.,0.])
for rollbase in [60.,30.,0.]:move('return_base'+str(int(rollbase)),[rollbase,.7,91.3,88.,0.,0.])
move('home',[0.,0.,90.,0.,0.,0.],1.)
pairs=[(a,b) for a in ['grab_fixed','gripper_link'] for b in ['base_link','link1','link2','link3','link4']]+[('link4','link2'),('link5','link2'),('link5','link3')]
prev=np.array(q0);samples=[];gaps={}
for st in steps:
 c=st['command'];target=prev.copy()
 if c['T']==122:target=np.array(st['q_deg'])
 elif c['T']==121:target[5]=180-c['angle']
 else:continue
 for v,(lo,hi) in zip(target,[(-90,90),(-110,110),(-70,190),(-90,90),(-5,90),(0,30)]):assert math.isfinite(v) and lo<=v<=hi
 for alpha in np.linspace(0,1,max(11,int(np.ceil(max(abs(target-prev))/2))+1)):
  qq=prev+(target-prev)*alpha;T,w,ms=P.geometry(qq);axes=np.concatenate([np.eye(3)]+[m[:3,:3].T for m in ms.values()]);proj={k:v@axes.T for k,v in w.items()}
  for x,y in pairs:
   A=proj[x];B=proj[y];gap=float(np.max(np.maximum(A.min(0)-B.max(0),B.min(0)-A.max(0))));key=x+'__'+y;gaps[key]=min(gaps.get(key,99),gap);assert gap>.015,(st['name'],key,gap)
  z=min(w[k][:,2].min() for k in ['grab_fixed','gripper_link']);assert z>.22,(st['name'],z)
  samples.append(dict(stage=st['name'],q_deg=qq.tolist(),tool_min_z_m=float(z),outlet_slope_deg=P.slope(qq)))
 prev=target
T0=P.geometry(q)[0];T1=P.geometry(tiltq)[0];shift=((T1@P.S.LIP_L5)[:3]-(T0@P.S.LIP_L5)[:3])*1000
plan=dict(experiment='finish_current_residue_and_HOME',close_torque=200,new_scoop=False,home_at_end=True,finish_stage='home_after_finish',initial_target_door_deg=30,preflight_q_deg=q0,reference_issued_q_deg=ref,steps=steps,return_passed_raise1=dict(previously_passed_q5=q[:5],source=str(case/'execution_04/raw.jsonl'),max_command_change_deg=2),tilt_q_deg=tiltq,interpolation_samples=len(samples),projection_min_m=gaps,lip_shift_baseline_to_tilt_mm=shift.tolist(),finish='HOME [0,0,90,0,0,0]',tracking_gate_deg=5,limits=['One deliberate return to the immediately previous passed raise1 target; no general tracking relaxation.','Support shoulder/elbow targets fixed during roll and wrist tilt. Cup is operator positioned; lip motion is not compensated.','No new scoop, PID change, torque change or automatic retry after a stop.','Sampled nominal CAD only; actual tracking and full executed feedback are separately checked.'],sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),case/'execution_04/plan.json',case/'execution_04/raw.jsonl',case/'feedback_preflight_05/result.json',Path(P.__file__)]})
(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'interpolation.json').write_text(json.dumps(samples,indent=2));(out/'source_plan_finish_home.py').write_bytes(Path(__file__).read_bytes())
frames=[]
for name,qq,role in [('current',q0,'actual'),('outlet20',tiltq,'target'),('HOME',[0,0,90,0,0,0],'target')]:
 T=P.geometry(qq)[0];frames.append(V.frame_from_axes(name,(T@P.S.LIP_L5)[:3],x_axis=T[:3,0],z_axis=T[:3,2],role=role,label=name))
assert V.snapshot_frame_plot(out/'plan_frames.png',frames,title='Finish residual discharge and return HOME')['ok']
print(json.dumps(dict(samples=len(samples),commands=len(steps),lip_shift_mm=shift.tolist(),min_plane_gap_m=min(gaps.values())),indent=2))
