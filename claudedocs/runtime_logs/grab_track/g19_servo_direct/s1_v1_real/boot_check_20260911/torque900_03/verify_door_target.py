"""Verify actual explicit door targets throughout a recorded scoop; no hardware."""
import argparse,hashlib,json
from pathlib import Path

def audit(path):
 raw=path/'raw.jsonl';events=[json.loads(l) for l in raw.read_text().splitlines()]
 target=None;checked=[];violations=[];boot=[];high_start=None;high_durations=[]
 for i,e in enumerate(events):
  if 'rst:0x' in e.get('text',''):boot.append(i+1)
  if e['ev']!='tx':continue
  c=e['command'];assert c['T'] in (105,107,121,122)
  if c['T']==107:
   if high_start is not None:high_durations.append((e['mono_ns']-high_start)/1e9)
   high_start=e['mono_ns'] if c['tor']==900 else None
  if c['T']==121:assert c['joint']==6;target=180-c['angle']
  if c['T']==122 and target is not None:
   r={'raw_line':i+1,'phase':e['phase'],'explicit_door_target_deg':target,'actual_transmitted_door_target_deg':180-c['h']}
   checked.append(r)
   if abs(r['actual_transmitted_door_target_deg']-target)>1e-9:violations.append(r)
 return {'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'checked_T122':checked,'violations':violations,
         'boot_lines':boot,'900_durations_s':high_durations,'end_event':events[-1]['ev'],'high_limit_left_active':high_start is not None}

def verify(out):
 actual=audit(out);known=audit(out.parent/'torque900_02');r=json.loads((out/'result.json').read_text())
 assert known['violations'],'positive control failed to detect old measured-angle target policy'
 assert r['completed'] and not r['dry_run'] and actual['raw_sha256']==r['raw_sha256']
 assert actual['checked_T122'] and not actual['violations'] and not actual['boot_lines']
 assert actual['end_event']=='closed' and not actual['high_limit_left_active']
 assert all(0<d<=15 for d in actual['900_durations_s'])
 lift=[x for x in actual['checked_T122'] if x['phase']=='lift8'];assert lift and all(x['actual_transmitted_door_target_deg']==0 for x in lift)
 actual.update(pass_=True,known_legacy_violation_count=len(known['violations']),known_legacy_raw_sha256=known['raw_sha256'])
 (out/'door_target_verification.json').write_text(json.dumps(actual,indent=2))
 print('ACTUAL_DOOR_TARGET_PRESERVED_AND_LEGACY_CONTROL_REJECTED_OK')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();verify(a.out)
