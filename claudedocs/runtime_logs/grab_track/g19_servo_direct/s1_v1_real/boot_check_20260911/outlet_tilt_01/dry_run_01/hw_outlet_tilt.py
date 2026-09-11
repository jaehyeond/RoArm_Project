"""Execute only the precomputed open S1 outlet-tilt plan. No scoop/return.

The existing RecordedArm default (-5..5 roll) is unchanged. This instance
accepts exact reviewed T122 packets, in order, with door30 and torque200 only.
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
from hw_recorded_arm import RecordedArm, validate_command

def validator(plan):
    packets=[row['command'] for row in plan['path']]
    def check(c):
        if c=={'T':105} or c=={'T':107,'tor':200}:return
        if c in packets:return
        raise ValueError('Command outside frozen open-tilt plan: '+repr(c))
    return check

def audit(plan):
    check=validator(plan)
    for row in plan['path']:
        c=row['command'];check(c)
        assert c['h']==150 and c['spd']==200*180/2048 and c['acc']==50*180/25400
        q=[c[k] for k in ('b','s','e','t','r')]
        for x,(lo,hi) in zip(q,[(-90,90),(-110,110),(-70,190),(-90,90),(-5,90)]):assert math.isfinite(x) and lo<=x<=hi
    bad=[{'T':106},{'T':109},{'T':503,'id':17,'p':16},{'T':107,'tor':790},
         {**plan['path'][-1]['command'],'h':149}, {**plan['path'][-1]['command'],'r':89.999},
         {**plan['path'][-1]['command'],'t':91}]
    for c in bad:
        try:check(c)
        except ValueError:pass
        else:raise AssertionError(('accepted unplanned command',c))
    try:validate_command(plan['path'][-1]['command'])
    except AssertionError:pass
    else:raise AssertionError('old default roll guard changed')
    return {'pass':True,'approved_packets':len(plan['path']),'negative_cases_rejected':len(bad),'old_default_rejects_roll90':True}

def run(plan_path,out,dry):
    plan=json.loads(plan_path.read_text());checks=audit(plan)
    out.mkdir(exist_ok=False)
    (out/'plan.json').write_bytes(plan_path.read_bytes());(out/'guard_audit.json').write_text(json.dumps(checks,indent=2))
    for p in [Path(__file__),Path(__file__).with_name('hw_recorded_arm.py'),Path(__file__).with_name('hw_serial_atomic.py')]:
        (out/p.name).write_bytes(p.read_bytes())
    result=dict(completed=False,dry_run=dry,experiment=plan['experiment'],plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),measurements={},new_scoop=False)
    arm=None
    try:
        if dry:
            result['completed']=True
            return
        arm=RecordedArm(out,command_validator=validator(plan))
        q=arm.joints_angle_get();expected=plan['preflight_q_deg']
        if max(abs(x-y) for x,y in zip(q,expected))>1:raise RuntimeError('start changed since query preflight; hold')
        result['initial_q_deg']=q
        arm.send({'T':107,'tor':200})
        # T122 packets contain explicit open30 from the first move; no T121 is
        # needed. Preserve this target even if SDK methods are used in future.
        arm.commanded_door_deg=30.
        arm.phase='outlet_baseline';arm.record('stage');time.sleep(1)
        last=plan['reference_issued_q_deg']
        for row in plan['path']:
            q=arm.joints_angle_get()
            if max(abs(x-y) for x,y in zip(q[:5],last[:5]))>5:raise RuntimeError('pre-move deviation >5deg; hold')
            if not 27<=q[5]<=31:raise RuntimeError('door no longer open; hold')
            arm.phase=row['name'];arm.record('stage',target_q_deg=row['q_deg'])
            arm.send(row['command'])
            deadline=time.monotonic()+8;previous=None;stable=0
            while time.monotonic()<deadline:
                time.sleep(.25);q=arm.joints_angle_get()
                if previous is not None and max(abs(x-y) for x,y in zip(q,previous))<.3:stable+=1
                else:stable=0
                previous=q
                if stable>=2:break
            else:raise RuntimeError('motion did not settle in8s; hold')
            dev=max(abs(x-y) for x,y in zip(q[:5],row['q_deg'][:5]))
            arm.record('settled',actual_q_deg=q,max_arm_error_deg=dev)
            if dev>5 or not 27<=q[5]<=31:raise RuntimeError('settled pose outside gate; hold')
            if row['name'].endswith(('_5','_90','_10','_20','_30','_40')):print('AT',row['name'],[round(v,2) for v in q],flush=True)
            last=row['q_deg']
        arm.phase='outlet_after_tilt';arm.record('stage');time.sleep(3)
        result['completed']=True
    except BaseException as e:
        result['error']=repr(e);print('STOP_HOLD',repr(e),flush=True)
    finally:
        if arm:
            try:result['final_q_deg']=arm.joints_angle_get()
            except Exception as e:result['final_feedback_error']=repr(e)
            arm.close()
        if (out/'raw.jsonl').exists():result['raw_sha256']=hashlib.sha256((out/'raw.jsonl').read_bytes()).hexdigest()
        result['no_return_no_close_no_torque_off']=True
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();run(a.plan,a.out,a.dry_run)
