"""Read-only labmeeting provenance checks; no simulation or report writes."""
import hashlib
import json
import math
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path('/home/cgxr/Documents/Robotics/RoArm_Project')
PPT = Path('/home/cgxr/Downloads/랩미팅 9월15일_초안.pptx')
PPT_SHA = '59d08134b7fac0617b2b3d5495bc9439ac7d54b7e74ab65a337a0c7f3f8b61c9'
BASE = ROOT / 'claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim'
CELLS = {
    'W10': BASE / 'w10_deme_close_fix/cell_DE_dt2e6_c',
    'W11': BASE / 'w11_dt_sensitivity_20260911/cell_dt1e6_seed460',
}
PILE = Path('/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_same(actual, expected):
    if actual != expected:
        raise AssertionError(f'integrity mismatch: {actual} != {expected}')


def main():
    require_same(sha(PPT), PPT_SHA)
    damaged_memory_copy = bytearray(PPT.read_bytes())
    damaged_memory_copy[0] ^= 1
    try:
        require_same(hashlib.sha256(damaged_memory_copy).hexdigest(), PPT_SHA)
    except AssertionError:
        negative_rejected = True
    else:
        raise AssertionError('negative integrity control accepted')
    with ZipFile(PPT) as package:
        slides = [n for n in package.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]
    assert len(slides) == 6
    with np.load(PILE, allow_pickle=False) as raw:
        template = json.loads(str(raw['clump_template_json']))
    offsets = np.asarray(template['offsets_m'], dtype=float)
    radii = np.asarray(template['sphere_radii_m'], dtype=float)
    dimensions = ((offsets + radii[:, None]).max(axis=0)
                  - (offsets - radii[:, None]).min(axis=0)) * 1000.0
    axes = np.asarray(template['lens']['axes_mm_input'])
    target_volume = math.pi / 6 * float(np.prod(axes * 1e-3))
    mass = 905.0 * float(template['union_volume_m3'])
    assert np.isclose(mass, template['mass_kg'], rtol=1e-12, atol=0)
    assert np.isclose(target_volume, template['union_volume_m3'], rtol=1e-12, atol=0)
    assert np.all(np.asarray(template['moi_kg_m2']) > 0)
    records = {}
    for name, cell in CELLS.items():
        result_file = cell / 'scoop_s1_seed460.json'
        npz_file = cell / 'scoop_s1_seed460.npz'
        result = json.loads(result_file.read_text())
        with np.load(npz_file, allow_pickle=False) as raw:
            count = int(np.count_nonzero(raw['in_cavity']))
        require_same(count, result['capture']['n_in_cavity'])
        computed_mass_g = count * mass * 1000.0
        assert abs(computed_mass_g - result['capture']['mass_g']) < 0.000051
        for key in ['mass_kg', 'moi_kg_m2', 'sphere_radii_m', 'offsets_m', 'n_spheres']:
            require_same(result['particle'][key], template[key])
        records[name] = {
            'json': str(result_file), 'json_sha256': sha(result_file),
            'npz': str(npz_file), 'npz_sha256': sha(npz_file),
            'capture_mask_count': count, 'capture_mass_recomputed_g': computed_mass_g,
            'dt_s': result['params']['timestep_s'],
            'contact_params': {k: result['params'][k] for k in ['E_pa', 'nu', 'mu', 'Crr', 'CoR']},
            'n_pellets': result['particle']['n'],
        }
    require_same(records['W10']['contact_params'], records['W11']['contact_params'])
    return {
        'status': 'LM_NUMERIC_PROVENANCE_VERIFIED',
        'scope': 'source preservation and stored numeric consistency, not physical calibration or task success',
        'ppt_sha256': PPT_SHA, 'ppt_slides': len(slides),
        'negative_hash_control_rejected': negative_rejected,
        'pile_path': str(PILE), 'pile_sha256': sha(PILE),
        'spheres_per_pellet': len(radii), 'target_axes_mm': axes.tolist(),
        'union_bounds_recomputed_mm': dimensions.tolist(),
        'assumed_density_kg_m3': 905.0, 'mass_recomputed_g': mass * 1000.0,
        'stored_moi_kg_m2': template['moi_kg_m2'], 'cells': records,
        'new_physics_runs': 0, 'files_written': 0,
    }


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
