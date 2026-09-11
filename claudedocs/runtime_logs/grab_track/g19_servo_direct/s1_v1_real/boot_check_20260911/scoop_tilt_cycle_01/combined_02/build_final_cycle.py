import json,hashlib,subprocess,sys
from pathlib import Path
b=Path('claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911').resolve();case=b/'scoop_tilt_cycle_01';last=case/'execution_05';reslast=json.loads((last/'result.json').read_text());assert reslast['completed'] and reslast['home_reached'] and reslast['final_phase']=='home_after_finish'
subprocess.run([sys.executable,str(b/'audit_scoop_tilt_cycle.py'),str(last)],check=True,stdout=subprocess.DEVNULL)
out=case/'combined_02';out.mkdir();segments=[];raw=[];alltx=[]
for i in range(1,6):
 d=case/f'execution_{i:02d}';r=json.loads((d/'result.json').read_text());a=json.loads((d/'command_audit.json').read_text());data=(d/'raw.jsonl').read_bytes();ee=[json.loads(x) for x in data.splitlines()];assert a['pass'] and hashlib.sha256(data).hexdigest()==r['raw_sha256']
 segments.append(dict(path=str(d.relative_to(b)),raw_sha256=r['raw_sha256'],plan_sha256=r['plan_sha256'],completed=r['completed'],error=r.get('error'),final_phase=r['final_phase'],start_mono_ns=ee[0]['mono_ns'],end_mono_ns=ee[-1]['mono_ns'],active_duration_s=(ee[-1]['mono_ns']-ee[0]['mono_ns'])/1e9,audit=a));raw.append(data);alltx.extend(e for e in ee if e['ev']=='tx' and e['command']['T']!=105)
 if i==2:initial_home=r['home_actual_q_deg']
assert all(segments[i]['end_mono_ns']<segments[i+1]['start_mono_ns'] for i in range(4))
names=[e['phase'] for e in alltx];homes=[i for i,e in enumerate(alltx) if e['phase']=='home' and e['command']['T']==122];plunges=[i for i,e in enumerate(alltx) if e['phase']=='plunge' and e['command']['T']==122];tilts=[i for i,e in enumerate(alltx) if e['phase']=='finish_tilt_8'];closes=[i for i,e in enumerate(alltx) if e['phase']=='close_empty']
assert len(homes)==2 and len(plunges)==1 and len(tilts)==1 and len(closes)==1 and homes[0]<plunges[0]<tilts[0]<closes[0]<homes[1]
for j in homes:assert [alltx[j]['command'][k] for k in ['b','s','e','t','r','h']]==[0,0,90,0,0,180]
(out/'raw.jsonl').write_bytes(b''.join(raw));combinedhash=hashlib.sha256((out/'raw.jsonl').read_bytes()).hexdigest()
res=dict(completed=True,completion_meaning='Requested physical stages reached through five segments and four preserved earlier tracking stops; not uninterrupted execution.',uninterrupted=False,dry_run=False,new_scoop=True,home_reached=True,home_actual_q_deg=reslast['home_actual_q_deg'],initial_home_actual_q_deg=initial_home,final_q_deg=reslast['final_q_deg'],final_phase=reslast['final_phase'],close_torque=790,source_segments=segments,measurements={},raw_sha256=combinedhash,active_recording_duration_s=sum(s['active_duration_s'] for s in segments),prior_tracking_stops=4,limits=['Original raw bytes concatenated, monotonic gaps unchanged.','Earlier failed original path remains failed; final fixed-support wrist tilt and HOME completed.','Pellet residual and mass require user observation.'])
(out/'result.json').write_text(json.dumps(res,indent=2));(out/'plan.json').write_text(json.dumps(dict(experiment='one_HOME_scoop790_discharge_and_HOME_with_recoveries',close_torque=790,source_plans=[str(case/f'execution_{i:02d}'/'plan.json') for i in range(1,6)],new_scoop=True),indent=2))
(out/'command_audit.json').write_text(json.dumps(dict(pass_=True,actual_new_scoops=1,home_commands=2,stage_order='initial_HOME < plunge < finish_tilt8 < close_empty < final_HOME',final_controller_completed=True,uninterrupted=False,prior_tracking_stops=4,final_HOME_exact_command=[0,0,90,0,0,0],actuator_commands=len(alltx),segments=segments,meaning='Requested command stages reached, pellet discharge needs operator observation.'),indent=2))
for name in ['analysis_scoop_tilt_cycle.py','audit_scoop_tilt_cycle.py']:(out/name).write_bytes((b/name).read_bytes())
(out/'build_final_cycle.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(out=str(out),active_recording_duration_s=res['active_recording_duration_s'],raw_sha256=combinedhash),indent=2))
