"""CPU-only provenance/parameter arithmetic. No DEME/Isaac imports or simulation."""
from pathlib import Path
import ast
import hashlib
import json
import math
import sys
import numpy as np
import trimesh

ROOT = Path('/home/cgxr/Documents/Robotics/RoArm_Project')
OUT = ROOT / 'claudedocs/research/professor_review_20260916'
ORCA = Path('/home/cgxr/orca/workspaces/RoArm_Project')
SIM = ROOT / 'claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim'
W13 = ORCA / 'w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation'
W12 = ORCA / 'w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay'
PILE = ORCA / 'pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz'
PKG = Path('/home/cgxr/miniconda3/envs/roarm/lib/python3.11/site-packages')
def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text())
def diff(a, b): return {k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}

def audit():
    evidence={}
    def load(p): evidence[str(p)]=sha(p); return read(p)
    evidence[str(PILE)]=sha(PILE)
    assert evidence[str(PILE)]=='659d6b0bc771678a0c7209d91f550edc933d03e41922245ea0adb64eeb818812'
    with np.load(PILE,allow_pickle=False) as z:
        tpl=json.loads(z['clump_template_json'].item()); meta=json.loads(z['metadata_json'].item())
        off=np.array(tpl['offsets_m']); rad=np.array(tpl['sphere_radii_m'])
        ext=(off+rad[:,None]).max(0)-(off-rad[:,None]).min(0)
        n=len(z['clump_positions_m']); ns=len(z['positions_m'])
        ids=z['clump_ids']; assert n==20000 and ns==140000 and len(np.unique(ids))==20000
        assert np.all(np.bincount(ids)==7)
    axes=np.array(tpl['lens']['axes_mm_input'])
    volume=math.pi/6*np.prod(axes*1e-3); mass=905*volume
    assert math.isclose(mass,tpl['mass_kg'],rel_tol=1e-12)
    moments=np.diag(tpl['lens']['inertia_tensor_unit_density'])
    moi=905*moments*tpl['lens']['geometric_scale_applied']**5
    np.testing.assert_allclose(moi,tpl['moi_kg_m2'],rtol=1e-12,atol=0)
    mesh=trimesh.util.concatenate([trimesh.creation.icosphere(subdivisions=1,radius=float(r)).apply_translation(o) for o,r in zip(off,rad)])
    assert len(mesh.vertices)==294 and len(mesh.faces)==560
    w10=load(SIM/'w10_deme_close_fix/cell_DE_dt2e6_c/scoop_s1_seed460.json')
    candidates=list((SIM/'w11_dt_sensitivity_20260911').glob('**/scoop_s1_seed460.json'))
    candidates=[p for p in candidates if load(p).get('params',{}).get('timestep_s')==1e-6]
    assert len(candidates)==1, candidates
    w11=load(candidates[0]); w13=load(W13/'run_01/w13_cycle_seed460.json')
    assert w10['params']['timestep_s']==2e-6 and w13['params']['timestep_s']==1e-6
    for w in [w10,w11,w13]:
        np.testing.assert_allclose(w['particle']['sphere_radii_m'],rad,rtol=0,atol=0)
        np.testing.assert_allclose(w['particle']['offsets_m'],off,rtol=0,atol=0)
    params_diff=diff(w10['params'],w11['params'])
    assert set(params_diff)<= {'timestep_s','render_timeline_path','_note'}
    old=load(SIM/'w10_deme_close_fix/params_w10_DE_c.json')
    small=load(SIM/'w10_deme_close_fix/params_w10_DE_dt2e6_c.json')
    oldtl=load(SIM/'w10_deme_close_fix/cell_DE_c/timeline_seed460.json')
    errpath=SIM/'w10_deme_close_fix/cell_DE_c/stderr.txt'
    evidence[str(errpath)]=sha(errpath)
    assert '22291.97' in errpath.read_text()
    render={}
    for name,p in [('w10',W12/'w10/gates_w12_w10.json'),('w11',W12/'w11/gates_w12_w11.json'),('w13',W13/'partial_post_03/isaac/render_manifest.json')]:
        x=load(p);render[name]={k:x[k] for k in ['cmd','versions','cameras','mapping','instancer','frame_contract','inventory_labels','lens_proto','particle_protos','initialization_vs_display','camera_intrinsics_sensor','camera_intrinsics_analytic_fx'] if k in x}
    sources=[ROOT/'sim_deme_scoop_s1.py',*sorted((W13/'rev28/src').glob('*.py')),W12/'sim_isaac_replay_w12.py',W13/'post_03_rev/src/isaac_replay_w13.py']
    references={k:[] for k in w13['params']}
    for p in sources:
        evidence[str(p)]=sha(p)
        txt=p.read_text();tree=ast.parse(txt)
        if p.name in ['sim_deme_scoop_s1.py','sim_w13_full_cycle.py']:
            assert not any(isinstance(t,ast.Call) and isinstance(t.func,ast.Attribute) and t.func.attr=='SetIntegrator' for t in ast.walk(tree))
        for t in ast.walk(tree):
            if isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='P' and isinstance(t.slice,ast.Constant) and t.slice.value in references:
                references[t.slice.value].append(str(p)+':'+str(t.lineno))
            if isinstance(t,ast.Call) and isinstance(t.func,ast.Attribute) and isinstance(t.func.value,ast.Name) and t.func.value.id=='P' and t.func.attr=='get' and t.args and isinstance(t.args[0],ast.Constant) and t.args[0].value in references:
                references[t.args[0].value].append(str(p)+':'+str(t.lineno))
    installed=['include/DEM/API.h','include/DEM/Models.h','share/DEME/kernel/DEMHelperKernels.cuh','share/DEME/kernel/DEMCustomizablePolicies/FullHertzianForceModel.cu','share/DEME/kernel/DEMCustomizablePolicies/IntegrationVelPassOnExtendedTaylor.cu','share/DEME/kernel/DEMCollectForceKernels_Compact.cu','share/DEME/kernel/DEMIntegrationKernels.cu','share/DEME/kernel/DEMCalcForceKernels.cu']
    for s in installed:evidence[str(PKG/s)]=sha(PKG/s)
    ipkg=Path('/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages')
    for rel in ['isaaclab/source/isaaclab/isaaclab/sim/spawners/sensors/sensors_cfg.py','isaaclab/source/isaaclab/isaaclab/sim/spawners/sensors/sensors.py','isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py','isaacsim/extscache/usdrt.scenegraph-7.6.1+69cbf6ad.lx64.r.cp311/include/usdrt/scenegraph/usd/usdGeom/camera.h']:
        evidence[str(ipkg/rel)]=sha(ipkg/rel)
    assert 'm_integrator = TIME_INTEGRATOR::EXTENDED_TAYLOR' in (PKG/installed[0]).read_text()
    assert 'v = old_v + v_update * 0.5' in (PKG/installed[4]).read_text()
    call_lengths=[]
    for dt in [1e-6,2e-6,1e-5,1e-4,1e-3]:
        h=float(np.float32(dt)); acc=0.; steps=0
        while acc<1e-4:
            acc+=h;steps+=1
        call_lengths.append({'requested_sync_s':1e-4,'nominal_dt_s':dt,'float32_dt_s':h,'steps':steps,'source_derived_actual_duration_s':acc})
    return {'artifact':'PHYSICS_PARAMETER_PROVENANCE_20260916','new_simulations':0,'evidence_sha256':evidence,
        'template':tpl,'canonical_pile_config':meta['config'],'pile_settling_gate':meta['settling_gate'],
        'recomputed':{'n_independent_pellets':n,'contact_spheres':ns,'axes_mm':(ext*1000).tolist(),
          'axes_error_percent':((ext*1000/axes-1)*100).tolist(),'volume_m3':float(volume),'mass_kg':float(mass),'moi_kg_m2':moi.tolist(),
          'capture_mass_w10_g':541*mass*1000,'capture_mass_w11_g':517*mass*1000,
          'Eeff_particle_particle_Pa':1/(2*(1-.3**2)/5e6),
          'Geff_particle_particle_Pa':1/(4*(2-.3)*(1+.3)/5e6),
          'Eeff_particle_mesh_Pa':1/((1-.3**2)/5e6+(1-.3**2)/3e9)},
        'w10_w11_effective_param_diff':params_diff,'historic_10us_to_2us_config_diff':diff(old,small),
        'historic_10us':{'timeline_state':oldtl.get('state'),'saved_last_t_s':oldtl['rows'][-1]['sim_t'],'saved_max_speed_m_s':max(r['v_particle_max'] for r in oldtl['rows']),
          'engine_error_reported_speed_m_s':22291.97,'engine_abort_allowance_m_s':10000,'complete_result_json_present':(SIM/'w10_deme_close_fix/cell_DE_c/scoop_s1_seed460.json').exists()},
        'effective_params':{'w10':w10['params'],'w11':w11['params'],'w13':w13['params']},
        'w13_static_P_read_locations':references,'static_read_warning':'References include inherited unused branches and are not proof a value affected this run. npz_template overrides fallback pellet geometry.',
        'render':render,'source_derived_sync_durations':call_lengths,
        'render_prototype_reconstruction':{'vertices':len(mesh.vertices),'triangles':len(mesh.faces),'new_render':False},
        'camera_units_correction':'Stored key focal_mm is an unverified project label. Authoring values 40/20 and aperture20.955 are passed directly; USD uses tenths of scene units. Pixel focal length W12=1954.6650390625 is recorded. Not a measured40mm physical lens.',
        'integrator':'EXTENDED_TAYLOR (installed default; no project override found)'}

result=audit()
if '--write' in sys.argv:
    (OUT/'PARAMETER_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# 원본 파라미터 전수 대조표','', '설정 파일 일부가 아니라 실행 결과의 effective params 전체. 참조 줄은 정적 P 읽기이며 실행 적용의 증명이 아니다. 역할·실제 적용 여부는 REPORT.md와 함께 읽는다.','', '| 필드 | W10 | W11 | W13 | W13/공유 소스 참조 (최대 3곳) |','|---|---|---|---|---|']
    ps=result['effective_params']
    for k in sorted(ps['w10'].keys()|ps['w11'].keys()|ps['w13'].keys()):
        if k=='_note':continue
        vals=[json.dumps(ps[w].get(k,'미선언'),ensure_ascii=False) for w in ['w10','w11','w13']]
        refs=result['w13_static_P_read_locations'].get(k,[])[:3]
        lines.append('| '+k+' | '+' | '.join('`'+v.replace('|','\\|')+'`' for v in vals)+' | '+'<br>'.join(refs)+' |')
    (OUT/'PARAMETERS_ALL.md').write_text('\n'.join(lines)+'\n')
else:
    assert result==read(OUT/'PARAMETER_AUDIT.json'), 'provenance changed'
print('PHYSICS_PROVENANCE_OK')
