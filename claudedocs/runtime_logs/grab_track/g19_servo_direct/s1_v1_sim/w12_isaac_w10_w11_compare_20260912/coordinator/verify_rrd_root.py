"""Read-only root recheck: finalized RRD/RBL, raw values, geometry and negative control.

Does not launch Isaac/viewer/exporter. --report writes a new coordinator receipt
with exclusive creation. Existing scientific and replay artifacts are read only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import rerun

REPLAY = Path('/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay')
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPLAY))
from verify_w12_rrd_coverage import build_expectations, verify, verify_static_entities, semantic_checks

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--report')
args = parser.parse_args()
assert rerun.__version__ == '0.34.1'
cli = Path(sys.executable).parent / 'rerun'
version = subprocess.check_output([str(cli), '--version'], text=True).strip()
assert '0.34.1' in version
result = {'sdk': rerun.__version__, 'cli': version, 'artifacts': {}}
trial = REPLAY / 'rerun/trial2'
for stem in ['w12_replay_mapping', 'w12_decision']:
    validation = read(trial / (stem + '_rerun_validation.json'))
    assert validation['pass'] and validation['footer_manifest_present']
    assert validation['entity_path_contract']['exact_non_system_match']
    assert validation['timeline_contract']['exact_match']
    status = validation['log_status_summary']
    assert status['sink_attached_before_logging'] and status['sink_finalized'] and status['flush_ok']
    hashes = {}
    for suffix in ['.rrd', '.rbl']:
        p = trial / (stem + suffix)
        digest = sha(p)
        expected = validation['sha256'] if suffix == '.rrd' else validation['blueprint_verify']['sha256']
        assert digest == expected, str(p)
        subprocess.run([str(cli), 'rrd', 'verify', '--check-footers', 'true', str(p)], check=True, capture_output=True, text=True)
        hashes[suffix] = digest
    result['artifacts'][stem] = {'footer_reverified': True, 'hashes_match_validated_contract': hashes}
    print('FOOTER_AND_CONTRACT_HASH_PASS', stem, flush=True)

full = trial / 'w12_replay_mapping.rrd'
coverage = verify(REPLAY, full)
assert coverage['pass'], coverage
result['full_raw_array_coverage'] = coverage
print('FULL_64_FRAME_RAW_ARRAY_READBACK_PASS', len(coverage['checks']), flush=True)
semantic = semantic_checks(full)
assert semantic['pass'], semantic
result['geometry_vs_scalar_readback'] = semantic
expect, _ = build_expectations(REPLAY)
spec = {}
for tag in ['w10', 'w11']:
    g = read(REPLAY / tag / ('gates_w12_' + tag + '.json'))
    for event in ['first_close', 'final']:
        i = g['decision_events'][event]['frame']
        prefix = '/' + tag + '/'
        spec[prefix + 'decision/' + event + '/particles'] = ('Points3D:positions', expect[prefix + 'particles/animated'][1](i))
        spec[prefix + 'decision/' + event + '/source_nodes'] = ('Points3D:positions', np.concatenate([expect[prefix + 'source/tool_fixed_nodes'][1](i), expect[prefix + 'source/door_nodes'][1](i)]))
        spec[prefix + 'decision/' + event + '/rendered_nodes'] = ('Points3D:positions', np.concatenate([expect[prefix + 'rendered/door_nodes'][1](i), expect[prefix + 'rendered/markers'][1](i)]))
for stem in result['artifacts']:
    static = verify_static_entities(trial / (stem + '.rrd'), spec)
    assert static['pass'], static
    result['artifacts'][stem]['static_nearest_frame_readback'] = static

# Known-good decision positions versus a deliberately damaged expectation.
# Same readback function must reject a 5mm displacement. No source writes.
entity = '/w11/decision/final/rendered_nodes'
bad = spec[entity][1].copy()
bad[0, 2] += 0.005
negative = verify_static_entities(trial / 'w12_decision.rrd', {entity: ('Points3D:positions', bad)})
assert not negative['pass'], negative
result['negative_5mm_expectation_rejected'] = negative
result['all_pass'] = True
if args.report:
    assert Path(args.report).name == args.report
    with (OUT / args.report).open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
print('W12_ROOT_RRD_REVERIFIED', flush=True)
