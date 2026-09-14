"""Read-only, stdout-only bounded reproduction; not a full independent audit."""
import ast
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path('/home/cgxr/orca/workspaces/RoArm_Project')
REL = Path('claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913')
RUN = BASE / 'w13-full-cycle' / REL / 'implementation/run_01'
AUD = BASE / 'w13-cycle-audit' / REL / 'audit'
PILE = BASE / 'pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz'
SCRIPT = AUD / 'test_rev28_production_partial_results.py'

# Extract only three fully reviewed pure math functions; never execute main(),
# import audit_core, or invoke its fixed-path OUT.write_text report writer.
names = {'quat_matrix_xyzw', 'expand_spheres', 'classify_frame'}
tree = ast.parse(SCRIPT.read_text())
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in selected} == names
namespace = {'np': np, 'math': math}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SCRIPT), 'exec'), namespace)

with np.load(PILE, allow_pickle=False) as pile:
    template = json.loads(str(pile['clump_template_json']))
offsets = np.asarray(template['offsets_m'], float)
radii = np.asarray(template['sphere_radii_m'], float)
with np.load(RUN / 'w13_cycle_seed460.npz', allow_pickle=False) as raw:
    # Decode each large member once. This is a final-frame/one-counterexample
    # reproduction, not a second 283-frame all-label classification audit.
    positions = raw['particle_pos_m']
    quats = raw['particle_quat_xyzw']
    velocities = raw['particle_vel_m_s']
    codes = raw['inventory_code']
    ids = raw['particle_ids']
    assert np.array_equal(ids, np.arange(len(ids)))
    meta = json.loads(str(raw['metadata_json']))
    labels = raw['inventory_labels'].astype(str).tolist()
    fs = raw['particle_frame_sync_index'].astype(int)
    times = raw['sync_t_s']
    ft = raw['particle_frame_t_s']
    phases = raw['sync_phase_code']
    sub = raw['sync_subphase'].astype(str)
    trans = raw['transition_sync_index']
    expected_trans = np.flatnonzero(phases[1:] != phases[:-1]) + 1
    args = (positions[-1], quats[-1], velocities[-1],
            raw['tool_pos_m'][fs[-1]], raw['tool_quat_xyzw'][fs[-1]],
            raw['bin_pos_m'], raw['bin_quat_xyzw'], offsets, radii, meta)
    producer = namespace['classify_frame'](*args)
    strict = namespace['classify_frame'](*args, strict_source_floor=True)
    count = lambda a: {name: int(np.count_nonzero(a == i)) for i, name in enumerate(labels)}
    tags = raw['decision_tags'].astype(str).tolist()
    reclose = int(raw['decision_particle_frame_index'][tags.index('reclose_end')])
    cohort = ids[codes[reclose] == labels.index('tool_residual')]
    spheres = namespace['expand_spheres'](positions[0, 8:9].astype(float), quats[0, 8:9], offsets)[0]
    low = float((spheres[:, 2] - radii).min())
    top = float((spheres[:, 2] + radii).min())
    floor = float(meta['source_bounds_m'][2][0])
    margin = float(meta['classify_margin_m'])
    window = ft[ft >= ft[-1] - 0.25]
    vmax = float(raw['scalar_v_particle_max'].max())
    result = {
        'artifact': 'W13R_ROOT_PARTIAL_RAW_SPOT_REPRO_V1',
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'final frame, actual particle0-frame ID8, dense time/speed and cohort; not all-frame audit',
        'helper_source_sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        'n_sync': len(times), 'n_particle_frames': len(ft), 'n_particles': len(ids),
        'actual_final_time_s': float(times[-1]),
        'mapping_error_s': float(np.max(np.abs(ft - times[fs]))),
        'actual_transition_n': len(trans), 'expected_phase_only_n': len(expected_trans),
        'transition_schema_pass': bool(np.array_equal(trans, expected_trans)),
        'recorded_final': count(codes[-1]), 'producer_recomputed_final': count(producer),
        'strict_recomputed_final': count(strict),
        'strict_final_mismatch_n': int(np.count_nonzero(strict != codes[-1])),
        'cohort_n': len(cohort), 'cohort_recorded_final': count(codes[-1, cohort]),
        'cohort_strict_final': count(strict[cohort]),
        'counterexample': {'frame': 0, 'id': 8, 'recorded': labels[int(codes[0, 8])],
                          'sphere_surface_min_z_m': low, 'min_sphere_top_m': top,
                          'strict_bottom_pass': low > floor + margin,
                          'producer_top_pass': top > floor - margin,
                          'floor_plus_margin_m': floor + margin,
                          'floor_minus_margin_m': floor - margin},
        'home_hold_rows': int(np.count_nonzero(sub == 'home_hold')),
        'final_joint_deg': raw['sync_joint_deg'][-1].astype(float).tolist(),
        'dense_saved_vmax_m_s': vmax, 'warning_gt5': vmax > 5, 'stop_gt20': vmax > 20,
        'last_window_frame_times_s': window.tolist(),
        'last_window_max_gap_s': float(np.diff(window).max()),
        'no_new_physics': True, 'producer_files_written': 0,
    }
    assert np.array_equal(producer, codes[-1])
    assert result['strict_final_mismatch_n'] == 5362
    assert not result['transition_schema_pass']
    assert low <= floor + margin and top > floor - margin
    assert len(cohort) == 144 and result['home_hold_rows'] == 0
    assert result['mapping_error_s'] == 0.0
    result['reproduction_pass'] = True
    result['scientific_verdict'] = 'FULL_CYCLE_UNSUCCESSFUL_AND_RAW_SCHEMA_FAIL_UNCHANGED'
print(json.dumps(result, ensure_ascii=False, indent=2))
