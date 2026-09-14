"""Read-only CPU counterexamples for frozen audit/oracle_01; no physics."""
import json
import sys
from pathlib import Path

import numpy as np

ORACLE = Path('/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/oracle_01')
sys.path.insert(0, str(ORACLE))
import audit_core as A

def row(value, unit, severity, purpose):
    return dict(value=value, unit=unit, severity=severity, purpose=purpose,
                origin='explicit synthetic counterexample', operator='>',
                scientific_limitation='unit-specific condition, no experiment result')

base = {'criteria': {
    'sampled_speed_warning': row(5.0, 'm/s', 'warning', 'sampled speed warning'),
    'python_pop_stop': row(20.0, 'm/s', 'hard stop', 'existing strict pop-stop'),
}}
A.validate_criteria(base)
with_qmin = {'criteria': dict(base['criteria'], **{
    'bridge.quaternion_norm_min': row(0.5, 'dimensionless', 'hard stop', 'normalization bootstrap, NOT wall displacement'),
})}
observations = {}
try:
    A.validate_criteria(with_qmin)
    observations['qmin_half_false_rejection'] = None
except A.AuditFailure as error:
    observations['qmin_half_false_rejection'] = str(error)

labels = ['source', 'receiving_bin', 'tool_residual', 'spill', 'in_flight', 'ambiguous']
template = {'offsets_m': np.zeros((1, 3)), 'radii_m': np.array([0.001]), 'mass_kg': 1e-5}
pos = np.array([[0.5, 0.0, 0.03]])
quat = np.array([[0.0, 0.0, 0.0, 1.0]])
vel = np.array([[0.02, 0.0, 0.0]])
bp, bq = np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])
geom = {'shape': 'box', 'inner_bounds_local_m': [[-0.04, 0.04], [-0.04, 0.04], [0.0, 0.07]]}
cfg = dict(margin_m=2e-7, moving_threshold_m_s=0.005, spill_rest_z_m=0.02,
           bin_geometry=geom, source_bounds_m=[[-0.2, 0.2], [-0.2, 0.2], [0.0, 0.1]],
           tool_pos_m=np.array([2.0, 0.0, 0.0]), tool_quat_xyzw=bq,
           tool_cavity=dict(tool_to_cavity_R=np.eye(3), cavity_origin_m=np.zeros(3),
                            cavity_center_xz_m=[0.0, 0.0], radius_m=0.025, half_y_m=0.025))
code = A.classify_inventory_frame(pos, quat, vel, template, bp, bq, cfg, labels)
observations['moving_inside_bin_classifier_label'] = labels[int(code[0])]
raw = dict(particle_ids=np.array([0]), particle_pos_m=pos[None], particle_quat_xyzw=quat[None],
           particle_vel_m_s=vel[None], inventory_code=code[None], inventory_labels=np.array(labels),
           particle_frame_sync_index=np.array([0]), bin_pos_m=bp, bin_quat_xyzw=bq,
           tool_pos_m=cfg['tool_pos_m'], tool_quat_xyzw=bq)
try:
    A.validate_inventory(raw, template, geom, 2e-7, cfg)
    observations['classifier_output_rejected_by_inventory_validator'] = None
except A.AuditFailure as error:
    observations['classifier_output_rejected_by_inventory_validator'] = str(error)

observations['artifact'] = 'W13R_ROOT_ORACLE_01_COUNTEREXAMPLES'
observations['synthetic_only'] = True
print(json.dumps(observations, sort_keys=True))
assert observations['qmin_half_false_rejection'] is not None
assert observations['moving_inside_bin_classifier_label'] == 'in_flight'
assert observations['classifier_output_rejected_by_inventory_validator'] is not None
print('W13R_ORACLE_01_COUNTEREXAMPLES_REPRODUCED')
