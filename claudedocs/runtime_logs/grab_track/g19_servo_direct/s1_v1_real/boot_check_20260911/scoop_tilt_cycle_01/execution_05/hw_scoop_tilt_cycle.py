"""One frozen full-cycle plan: recover HOME, scoop790, outlet20, hold.

Stops on stale/boot/STOP or settled arm error>5deg. Never retries or homes on
failure. Uses atomic serial open and preserves explicit gripper targets.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
from hw_recorded_arm import RecordedArm

def command_validator(plan):
    allowed=[s['command'] for s in plan['steps']]
    def check(c):
        if c=={'T':105}:return
        if c not in allowed:raise ValueError('not in reviewed cycle plan')
    return check

def run(plan_path,out,dry=False):
    plan=json.loads(plan_path.read_text());check=command_validator(plan)
    for st in plan['steps']:check(st['command'])
    for c in [{'T':106},{'T':109},{'T':107,'tor':900},{'T':503,'id':17,'p':16}]:
        try:check(c)
        except ValueError:pass
        else:raise AssertionError('unsafe packet accepted')
    out.mkdir(exist_ok=False)
    (out/'plan.json').write_bytes(plan_path.read_bytes())
    for p in [Path(__file__),Path(__file__).with_name('hw_recorded_arm.py'),Path(__file__).with_name('hw_serial_atomic.py')]:
        (out/p.name).write_bytes(p.read_bytes())
    result=dict(completed=False,dry_run=dry,close_torque=plan['close_torque'],experiment=plan['experiment'],new_scoop=plan['new_scoop'],measurements={},home_reached=False,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
    arm=None;last=plan['reference_issued_q_deg'][:5]
    def stage(name):
        arm.phase=name;arm.record('stage')
    try:
        if dry:
            result['completed']=True;return result
        arm=RecordedArm(out,command_validator=check)
        q=arm.joints_angle_get();result['initial_q_deg']=q
        if max(abs(a-b) for a,b in zip(q,plan['preflight_q_deg']))>1:raise RuntimeError('start changed since preflight')
        arm.commanded_door_deg=plan['initial_target_door_deg']
        entry_retreat=False
        if plan.get('bounded_initial_retreat'):
            c=plan['steps'][0]['command'];assert c['T']==122
            target=[c[k] for k in ('b','s','e','t','r')]
            # A logged tracking stop may retreat only toward the previously
            # passed target, with <=1deg command change and <=5deg fresh error.
            # No general retry or tracking-threshold exception follows it.
            assert max(abs(a-b) for a,b in zip(target,last))<=1
            assert max(abs(a-b) for a,b in zip(q[:5],target))<=5
            assert target[:4]==plan['bounded_initial_retreat']['previously_passed_q5'][:4]
            assert c['h']==150 and target[4]==last[4]
            arm.record('bounded_initial_retreat',from_issued_q5=last,to_q5=target,actual_q_deg=q)
            entry_retreat=True
        if plan.get('loaded_entry_retract'):
            c=plan['steps'][0]['command'];assert c['T']==122 and c['h']==180
            target=[c[k] for k in ('b','s','e','t','r')]
            assert all(target[j]==last[j] for j in [0,2,4])
            assert 0<=target[1]-last[1]<=.25 and 0<target[3]-last[3]<=4
            assert max(abs(a-b) for a,b in zip(q[:5],target))<=5
            assert -1<=q[5]<=8
            arm.record('loaded_entry_retract',from_issued_q5=last,to_q5=target,actual_q_deg=q)
            entry_retreat=True
        if plan.get('transport_recovery'):
            c=plan['steps'][0]['command']
            assert last==[0,45,36,90,0] and plan['initial_target_door_deg']==0
            assert c['T']==122 and c['h']==180
            assert [c[k] for k in ('b','s','e','t','r')]==[.43945312847050816,.7,91.3,88.,0.]
            assert 5<q[1]-last[1]<=5.5
            assert max(abs(q[j]-last[j]) for j in [0,2,3,4])<=3 and -1<=q[5]<=8
            arm.record('transport_recovery_entry_exception',from_issued_q5=last,to_command=c,actual_q_deg=q,ordinary_settled_gate_deg=5)
            entry_retreat=True
        if plan.get('return_passed_raise1'):
            c=plan['steps'][0]['command'];target=[c[k] for k in ('b','s','e','t','r')]
            assert c['T']==122 and c['h']==150 and plan['initial_target_door_deg']==30
            assert target==plan['return_passed_raise1']['previously_passed_q5']
            assert max(abs(a-b) for a,b in zip(target,last))<2
            assert max(abs(a-b) for a,b in zip(q[:5],target))<=5 and 27<=q[5]<=31
            arm.record('return_to_previously_passed_raise1',from_issued_q5=last,to_q5=target,actual_q_deg=q)
            entry_retreat=True
        for i,st in enumerate(plan['steps']):
            c=st['command'];q=arm.joints_angle_get()
            if not(i==0 and entry_retreat) and max(abs(a-b) for a,b in zip(q[:5],last))>5:raise RuntimeError('pre-move arm tracking error>5; hold')
            stage(st['name']);arm.record('planned_command',step=i,command=c)
            arm.send(c)
            if c['T'] in [121,122]:
                deadline=time.monotonic()+10;prev=None;stable=0
                while time.monotonic()<deadline:
                    time.sleep(.25);q=arm.joints_angle_get()
                    if prev is not None and max(abs(a-b) for a,b in zip(q,prev))<.3:stable+=1
                    else:stable=0
                    prev=q
                    if stable>=2:break
                else:raise RuntimeError('settle timeout; hold')
                if c['T']==122:last=[c[k] for k in ('b','s','e','t','r')]
                dev=max(abs(a-b) for a,b in zip(q[:5],last))
                arm.record('settled',step=i,actual_q_deg=q,max_arm_error_deg=dev)
                if dev>5 or not -1<=q[5]<=31:raise RuntimeError('settled pose outside gate; hold')
                if arm.commanded_door_deg==30 and not 27<=q[5]<=31:raise RuntimeError('open door failed to reach range; hold')
            if st['wait_after_s']:time.sleep(st['wait_after_s']);arm.joints_angle_get()
            if st['name']=='home':
                result['home_reached']=True;result['home_actual_q_deg']=arm.joints_angle_get()
            if st['name']=='close' and c['T']==121:result['before_lift_deg']=arm.joints_angle_get()[5]
            if st['name']=='lift8':result['after_lift_deg']=arm.joints_angle_get()[5]
            if st['name'] in ['home','above','open','plunge','close','lift8','outlet_baseline','align_90','tilt_40'] or i%15==0:
                print('AT',i,st['name'],[round(v,2) for v in arm.joints_angle_get()],flush=True)
        stage(plan.get('finish_stage','outlet_after_tilt'));time.sleep(3);arm.joints_angle_get()
        result['completed']=True
    except BaseException as e:
        result['error']=repr(e);print('STOP_HOLD',repr(e),flush=True)
    finally:
        if arm:
            try:result['final_q_deg']=arm.joints_angle_get();result['final_phase']=arm.phase
            except Exception as e:result['final_feedback_error']=repr(e)
            arm.close()
        if (out/'raw.jsonl').exists():result['raw_sha256']=hashlib.sha256((out/'raw.jsonl').read_bytes()).hexdigest()
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();r=run(a.plan,a.out,a.dry_run);raise SystemExit(0 if r['completed'] else 2)
