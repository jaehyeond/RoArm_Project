"""Read-only timing and transmitted-speed audit; no robot imports."""
from pathlib import Path
import json,hashlib,collections
OUT=Path(__file__).resolve().parent;CASE=OUT.parent;BASE=CASE.parent

def load(d):
 raw=d/'raw.jsonl';data=raw.read_bytes();return [json.loads(x) for x in data.splitlines()],hashlib.sha256(data).hexdigest()
e,sha=load(CASE/'execution_05');st=[x for x in e if x['ev']=='stage'];dur=collections.defaultdict(float);count=collections.Counter()
def group(n):
 for s in ['finish_tilt_','untilt_','unalign_','align_','return_base']:
  if n.startswith(s):return s
 return n
for i,x in enumerate(st):
 end=st[i+1]['mono_ns'] if i+1<len(st) else e[-1]['mono_ns'];dur[group(x['phase'])]+=(end-x['mono_ns'])/1e9;count[group(x['phase'])]+=1
init=(st[0]['mono_ns']-e[0]['mono_ns'])/1e9;total=(e[-1]['mono_ns']-e[0]['mono_ns'])/1e9
assert abs(sum(dur.values())+init-total)<1e-8
b,bsha=load(BASE/'torque790_01')
speeds=lambda ev:sorted(set((x['command']['spd'],x['command']['acc']) for x in ev if x['ev']=='tx' and x['command']['T'] in [121,122]))
assert speeds(e)==speeds(b)
categories={'alignment':dur['align_'],'tilt_and_hold':dur['finish_tilt_']+dur['outlet_after_tilt'],'orientation_restoration':dur['untilt_']+dur['unalign_'],'close_and_return_HOME':sum(dur[x] for x in ['close_empty','return_upright','return_base','home','home_after_finish']),'initialization_and_entry':init+dur['return_passed_raise1']+dur['outlet_baseline']}
assert abs(sum(categories.values())-total)<1e-8
m=sum(x['ev']=='tx' and x['command']['T'] in [121,122] for x in e)
result=dict(pass_=True,source_raw_sha256={'execution05':sha,'baseline790':bsha},duration_s=total,categories_s=categories,phase_durations_s=dict(dur),phase_counts=dict(count),motion_commands=m,baseline_motion_commands=sum(x['ev']=='tx' and x['command']['T'] in [121,122] for x in b),transmitted_speed_acc=speeds(e),baseline_transmitted_speed_acc=speeds(b),same_speed_fields=True,stepwise_settle_poll_s=.25,minimum_samples_per_move=3,per_move_min_settle_s=.75,total_settle_loop_min_elapsed_s=m*.75,interpretation=['Final84s includes orientation preparation, residue discharge, reverse orientation, empty closing and HOME; it is not the tilt alone.','At least0.75s per move follows from the frozen executor first sample plus two stable comparisons; actual motion occurs during that interval, so45s cannot simply be subtracted as removable overhead.','Three-second discharge hold and three-second final HOME hold are distinct.','Baseline71.8s is an earlier scoop/place/P1 run, not a matched HOME-to-HOME throughput control.'],hardware_commands_this_audit=0)
(OUT/'timing_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({'pass':True,'categories_s':categories,'same_speed_fields':True,'motion_commands':m},indent=2))
