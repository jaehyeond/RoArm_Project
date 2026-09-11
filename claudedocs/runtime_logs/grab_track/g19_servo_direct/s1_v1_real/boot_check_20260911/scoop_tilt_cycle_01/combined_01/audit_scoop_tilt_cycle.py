"""Read-only audit of one real full cycle; never imports the hardware driver."""
import hashlib,json,sys
from pathlib import Path

out=Path(sys.argv[1]);p=json.loads((out/'plan.json').read_text());r=json.loads((out/'result.json').read_text())
events=[json.loads(x) for x in (out/'raw.jsonl').read_text().splitlines()]
tx=[e for e in events if e['ev']=='tx' and e['command']['T']!=105]
packets=[e['command'] for e in tx];names=[e['phase'] for e in tx]
q=list(p['reference_issued_q_deg']);door=p['initial_target_door_deg'];violations=[]
for e in tx:
    c=e['command']
    if c['T']==121:door=180-c['angle']
    if c['T']==122 and c['h']!=180-door:violations.append(e)
home=[i for i,e in enumerate(tx) if e['phase']=='home' and e['command']['T']==122]
plunge=[i for i,e in enumerate(tx) if e['phase']=='plunge' and e['command']['T']==122]
home_packets=[packets[i] for i in home]
checks={
 'all_issued_packets_exact_plan_prefix':packets==[s['command'] for s in p['steps']][:len(packets)],
 'completion_matches_plan_coverage':(len(packets)==len(p['steps'])) if r['completed'] else bool(r.get('error')),
 'home_before_any_new_plunge':(len(home)==1 and len(plunge)==1 and home[0]<plunge[0] and r['home_reached']) if plunge else not p['new_scoop'] or not r['home_reached'],
 'no_scoop_in_continuation':bool(p['new_scoop']) or not(home or plunge),
 'any_home_command_exact':all([c[k] for k in ['b','s','e','t','r','h']]==[0,0,90,0,0,180] for c in home_packets),
 'explicit_door_target_preserved':not violations,
 'only_approved_types_and_torque':all(c['T'] in [107,121,122] and (c['T']!=107 or c['tor'] in [200,790]) for c in packets),
 'no_boot_text':not any('rst:0x' in e.get('text','') for e in events),
 'raw_hash':hashlib.sha256((out/'raw.jsonl').read_bytes()).hexdigest()==r['raw_sha256'],
}
report={'pass':all(checks.values()),'checks':checks,'actuator_commands':len(packets),'planned_actuator_commands':len(p['steps']),'actual_new_scoops':len(plunge),'home_commands':len(home),'last_target_is_outlet20':bool(names) and names[-1]=='tilt_40','n_feedback':sum(e['ev']=='rx_json' and e['data'].get('T')==1051 for e in events),'controller_completed':r['completed'],'final_phase':r.get('final_phase'),'home_actual_q_deg':r.get('home_actual_q_deg'),'pellet_discharge_not_observed_by_this_audit':True,'pass_meaning':'Issued packet provenance and stage accounting only; not controller completion or discharge success.'}
(out/'command_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert report['pass']
