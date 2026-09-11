"""Device-free acceptance check; never replays commands onto hardware."""
import csv, hashlib, json, math, statistics
from pathlib import Path
b=Path(__file__).resolve().parent/'pid_hold_02'
raw=b/'raw.jsonl';rows=[json.loads(l) for l in raw.read_text().splitlines()]
r=json.loads((b/'result.json').read_text());a=json.loads((b/'analysis.json').read_text())
assert hashlib.sha256(raw.read_bytes()).hexdigest()==r['raw_sha256']==a['raw_sha256']
assert r['completed'] and not r['error'] and r['restore_p16_command_sent']
assert not any('rst:0x' in row.get('text','') for row in rows)
cmd=[row['command'] for row in rows if row['ev']=='tx']
assert [c for c in cmd if c['T']!=105]==[{'T':108,'joint':2,'p':p,'i':0} for p in [8,48,8,48,16]]
obs=[row for row in rows if row['ev']=='rx_json' and row['data'].get('T')==1051]
assert len(obs)==r['n_feedback']==a['n_feedback']==1648
csvrows=list(csv.DictReader((b/'feedback.csv').open()));assert len(csvrows)==len(obs)
for c,o in zip(csvrows,obs):
 assert int(c['host_mono_ns'])==o['mono_ns'] and c['tG_raw']==''
 assert float(c['s_deg'])==math.degrees(o['data']['s'])
means={}
for phase in ['p8_a','p48_a','p8_b','p48_b']:
 ps=[o for o in obs if o['phase']==phase];ps=[o for o in ps if o['mono_ns']>=ps[-1]['mono_ns']-2_000_000_000]
 means[phase]=statistics.mean(math.degrees(o['data']['s']) for o in ps)
for i,c in enumerate('ab'):assert abs(means['p8_'+c]-means['p48_'+c]-a['paired_P8_minus_P48_deg'][i])<1e-12
v=b/'visual_02';validation=json.loads((v/'rerun_validation.json').read_text());inspection=json.loads((v/'inspection.json').read_text())
assert validation['pass'] and all(c['pass'] for c in validation['coverage_readback'].values())
assert validation['log_status']['sink_attached_before_logging'] and validation['log_status']['sink_finalized']
assert hashlib.sha256((v/'pid.rrd').read_bytes()).hexdigest()==validation['sha256']
assert inspection['inspected']
for name,p in inspection['paths'].items():assert hashlib.sha256((v/name).read_bytes()).hexdigest()==p['sha256']
manifest={str(p.relative_to(b)):hashlib.sha256(p.read_bytes()).hexdigest() for p in b.rglob('*') if p.is_file() and p.name!='manifest_sha256.json'}
(b/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
print('PID_RAW_CSV_METRICS_RERUN_INSPECTION_OK')
