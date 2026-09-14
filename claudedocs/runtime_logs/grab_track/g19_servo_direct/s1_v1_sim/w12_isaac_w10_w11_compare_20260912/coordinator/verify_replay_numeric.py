"""Independent read-only check of W12 recorded frames against frozen inputs.

No Isaac, robot, camera or renderer imports. Does not recreate hidden precision
from rounded renderer JSON. Root must separately inspect actual images/RRD.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

MAIN = Path('/home/cgxr/Documents/Robotics/RoArm_Project')
SIM = MAIN / 'claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim'
REPLAY = Path('/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay')
CELLS = {'w10': SIM / 'w10_deme_close_fix/cell_DE_dt2e6_c', 'w11': SIM / 'w11_dt_sensitivity_20260911/cell_dt1e6_seed460'}

def read(p):
    return json.loads(p.read_text())

def verify(g, tl, source):
    assert g['frame_contract']['render_idx'] == list(range(len(tl['t_s'])))
    assert len(g['frames']) == len(tl['t_s']) == 64
    origin = np.array(g['mapping']['deme_origin_world'])
    assert np.array_equal(origin, [0.35, 0., 0.163])
    meta = json.loads(str(tl['metadata_json']))
    actual = meta['q_open_joint_deg'] + np.degrees(Rotation.from_quat(tl['door_quat_xyzw'].astype(float)).as_rotvec() @ np.array(meta['door_axis_world']))
    max_norm_discrepancy = 0.
    for i, f in enumerate(g['frames']):
        assert f['i'] == f['k'] == i
        assert f['t_s'] == round(float(tl['t_s'][i]), 6)
        assert f['door_deg_nominal'] == round(float(tl['door_deg'][i]), 4)
        assert abs(f['door_deg_actual'] - actual[i]) <= 0.000051
        assert abs(f['door_deg_rendered'] - actual[i]) <= 0.00011
        assert np.max(np.abs(np.array(f['tool_src_w']) - (tl['tool_pos_m'][i].astype(float) + origin))) <= 0.0000051
        norm_mm = np.linalg.norm(np.array(f['lip169_sim_w']) - f['tool_src_w']) * 1000
        # Each stored position coordinate rounded to 1e-5 m. Never claim bit-exact recovery.
        difference = abs(norm_mm - f['lip_err_mm'])
        assert difference <= 0.018
        max_norm_discrepancy = max(max_norm_discrepancy, difference)
        assert f['captured_total'] == len(tl['captured_ids']) == source['capture']['n_in_cavity']
    assert g['frames'][-1]['captured_in_cavity'] == source['capture']['n_in_cavity']
    assert g['gates']['G3_lip_error']['max_mm'] == max(f['lip_err_mm'] for f in g['frames'])
    assert max(f['lip_err_mm'] for f in g['frames']) < 5.
    for stop in source['door']['stops']:
        key = 'first_close' if stop['phase'] == 'close' else 'reclose'
        e = g['decision_events'][key]
        nearest = int(np.argmin(np.abs(tl['t_s'] - stop['sim_t'])))
        assert e['frame'] == nearest
        assert abs(e['frame_minus_event_s'] - (float(tl['t_s'][nearest])-stop['sim_t'])) < 0.000001
    return {'frames':len(g['frames']), 'max_lip_mm_recorded':max(f['lip_err_mm'] for f in g['frames']), 'rounded_xyz_norm_discrepancy_max_mm':max_norm_discrepancy, 'final_cavity_count':g['frames'][-1]['captured_in_cavity']}

answer = {}
times = {}
configs = []
for tag, cell in CELLS.items():
    source = read(cell / 'scoop_s1_seed460.json')
    g = read(REPLAY / tag / f'gates_w12_{tag}.json')
    with np.load(source['render_timeline']['path'], allow_pickle=False) as z:
        tl = {k: z[k] for k in ['t_s','metadata_json','door_quat_xyzw','door_deg','tool_pos_m','captured_ids']}
        answer[tag] = verify(g, tl, source)
        times[tag] = tl['t_s']
        damaged = json.loads(json.dumps(g))
        damaged['frames'][-1]['door_deg_rendered'] += 1.
        try:
            verify(damaged, tl, source)
        except AssertionError:
            answer[tag]['one_degree_display_mismatch_rejected'] = True
        else:
            raise AssertionError('Negative control not rejected')
    configs.append({k:g[k] for k in ['mapping','cameras','env','camera_intrinsics_analytic_fx']})
assert configs[0] == configs[1], 'Display configuration differs'
pair = np.argmin(abs(times['w10'][:,None]-times['w11'][None,:]), axis=1)
assert np.array_equal(pair, np.arange(64))
answer['nearest_pair_max_delta_s'] = float(np.max(abs(times['w10']-times['w11'][pair])))
print(json.dumps(answer, ensure_ascii=False, indent=2))
print('W12_REPLAY_NUMERIC_REVERIFIED')
