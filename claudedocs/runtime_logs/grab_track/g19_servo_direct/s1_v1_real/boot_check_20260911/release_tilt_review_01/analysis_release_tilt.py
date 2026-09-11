"""Offline S1 release-axis review. No serial connection or motion commands."""
import csv
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / 'hw_s1_scoop_probe.py').exists())
sys.path.insert(0, str(ROOT))
import hw_s1_scoop_probe as S
from roarm_rl import viz_debug as V
from roarm_rl.rerun_contract import validate_rerun_artifact

os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


trial = OUT.parent / 'torque900_03'
raw = trial / 'raw.jsonl'
result = json.loads((trial / 'result.json').read_text())
assert sha(raw) == result['raw_sha256']
events = [json.loads(s) for s in raw.read_text().splitlines()]
opened = [i for i, e in enumerate(events) if e['ev'] == 'tx' and e['command'].get('T') == 121 and e['command'].get('angle') == 150][-1]
closed = next(i for i in range(opened + 1, len(events)) if events[i]['ev'] == 'tx' and events[i]['command'].get('T') == 121)
observations = [(i + 1, e) for i, e in enumerate(events) if opened < i < closed and e['ev'] == 'rx_json' and e['data'].get('T') == 1051 and e['mono_ns'] >= events[closed]['mono_ns'] - 1_000_000_000]
qrows = np.array([[math.degrees(e['data'][k]) for k in ('b', 's', 'e', 't', 'r')] + [180 - math.degrees(e['data']['g'])] for _, e in observations])
assert len(qrows) > 1 and np.all(qrows == qrows[0])
q = qrows[-1]
target = np.array([90., 73.25, 42., 64.77, 0., 30.])
tx = [e['command'] for e in events[:opened] if e['ev'] == 'tx' and e['command'].get('T') == 122][-1]
assert np.allclose(target[:5], [tx[k] for k in ('b', 's', 'e', 't', 'r')])

urdf = ROOT / 'local_assets/roarm_m3/urdf/roarm_m3_s1_v1.urdf'
xml = ET.parse(urdf).getroot()
jo = xml.find("joint[@name='link5_to_gripper_link']/origin")
J = S.K.Tmat([float(x) for x in jo.attrib['xyz'].split()], [float(x) for x in jo.attrib['rpy'].split()])
sources = [raw, trial / 'hw_s1_manual.py'] if (trial / 'hw_s1_manual.py').exists() else [raw, ROOT / 'hw_s1_manual.py']
sources += [urdf, ROOT / 'sim_scripts/roarm_kinematics.py', ROOT / 'scoop_grab_s1_design.py']
local = {}
for name, link in [('fixed', 'grab_fixed'), ('door', 'gripper_link')]:
    m = xml.find(f"link[@name='{link}']/visual/geometry/mesh")
    p = urdf.parent / m.attrib['filename']
    mesh = trimesh.load(p, process=False)
    local[name] = (np.asarray(mesh.vertices) * np.array([float(x) for x in m.attrib['scale'].split()]), np.asarray(mesh.faces))
    sources.append(p)
wall_path = ROOT / 'claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/fixed_wall_11.stl'
sources.append(wall_path)
wall = trimesh.load(wall_path, process=False)
inside = np.flatnonzero(wall.face_normals[:, 2] < -.5)
normal = np.array(wall.face_normals[inside[0]], dtype=float, copy=True)
normal /= np.linalg.norm(normal)
exit_tangent = np.array([-normal[2], 0., normal[0]])
surface = np.asarray(wall.vertices, float) / 1000
surface_faces = np.asarray(wall.faces)[inside]
center = np.asarray(wall.triangles_center)[inside].mean(0) / 1000


def transform(a):
    T = S.chain(a[:5])['link4_to_link5'].copy()
    T[2, 3] += .38
    return T


actual_T = transform(q)
actual_lip = (actual_T @ S.LIP_L5)[:3]
level_q = q.copy()
level_q[3] -= q[1:4].sum() - 180
pitch_q = q.copy(); pitch_q[3] += 5
roll_q = q.copy(); roll_q[4] = -5
ideal_T = actual_T.copy()
ideal_T[:3, :3] = actual_T[:3, :3] @ S.K.rpy_R(0, math.radians(-5), 0)
ideal_T[:3, 3] = actual_lip - ideal_T[:3, :3] @ S.LIP_L5[:3]
cases = [
    ('actual', q, actual_T, 'Measured release; last 1 s'),
    ('commanded', target, transform(target), 'Issued reference; not measured'),
    ('pitch_level', level_q, transform(level_q), 'Hypothesis: remove side lean'),
    ('pitch_plus5', pitch_q, transform(pitch_q), 'Hypothesis: wrist pitch +5 deg'),
    ('roll_minus5', roll_q, transform(roll_q), 'Hypothesis: wrist roll target -5 deg'),
    ('ideal_outward5', None, ideal_T, 'Concept: outlet tilt 5 deg; IK NOT solved'),
]
metrics = []
for name, a, T, label in cases:
    g = T[:3, :3].T @ np.array([0., 0., -1.])
    lip = (T @ S.LIP_L5)[:3]
    v = float(g @ exit_tangent)
    n = float(-g @ normal)
    metrics.append(dict(case=name, label=label, q_deg=None if a is None else a.tolist(), gravity_unit_link5=g.tolist(),
        exit_projected_slope_deg=math.degrees(math.atan2(v, n)), exit_gravity_fraction=v,
        side_gravity_fraction=abs(float(g[1])), lip_displacement_mm=((lip-actual_lip)*1000).tolist(),
        lip_displacement_norm_mm=float(np.linalg.norm(lip-actual_lip)*1000), T_world_link5=T.tolist(),
        measured=name == 'actual', executable_plan=False))
assert metrics[-1]['exit_projected_slope_deg'] > metrics[0]['exit_projected_slope_deg'] + 4.99
assert abs(metrics[4]['exit_projected_slope_deg'] - metrics[0]['exit_projected_slope_deg']) < .5
assert metrics[2]['side_gravity_fraction'] < 1e-10
report = dict(source_raw_lines=dict(open=opened+1, close=closed+1, steady_first=observations[0][0], steady_last=observations[-1][0]),
    last_second_received_rows=len(observations), actual_q_deg=q.tolist(), command_q_deg=target.tolist(),
    open_command_to_close_command_s=(events[closed]['mono_ns']-events[opened]['mono_ns'])/1e9,
    configured_wait_after_open_settle_s=1.5, actual_tool_lean_deg=math.degrees(math.acos(-actual_T[2,2])),
    outlet_inner_facet_normal_link5=normal.tolist(), outlet_inner_facet_angle_deg=math.degrees(math.atan2(normal[0], -normal[2])),
    facet_interpretation='Candidate outlet surface only; photo-to-mesh correspondence and individual pellet contacts are NOT calibrated.',
    cases=metrics, source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
    verdict='OUTLET_DIRECTED_TILT_GEOMETRICALLY_PLAUSIBLE__SMALL_WRIST_ROTATION_AXIS_DEPENDENT__PHYSICAL_DISCHARGE_UNTESTED',
    limits=['No contact/friction simulation or new hardware action.', 'Photo is not calibrated or time synchronized to robot.',
        'Ideal outlet tilt has no solved joint path/cup clearance; not a runnable command.',
        'Pitch leveling reduces side-directed gravity but barely changes outlet slope; residue may remain.',
        'Wrist-only rotation moves the lip; cup geometry is unmeasured.',
        'SDK to URDF geometry convention reused, no fresh calibration.', 'Old four-bar grab roll14-degree rule is not an S1 clearance certificate.'],
    friction_reference='https://openstax.org/books/university-physics-volume-1/pages/6-2-friction')
save('analysis.json', report)
with (OUT / 'candidates.csv').open('x', newline='') as f:
    cols=['case','exit_projected_slope_deg','exit_gravity_fraction','side_gravity_fraction','lip_displacement_norm_mm','measured','executable_plan']
    w=csv.DictWriter(f, fieldnames=cols);w.writeheader();w.writerows({k:m[k] for k in cols} for m in metrics)

frames = []
for name, a, T, label in cases:
    frames.append(V.frame_from_axes(name, (T @ S.LIP_L5)[:3], x_axis=T[:3,0], z_axis=T[:3,2], role='actual' if name=='actual' else 'target', label=label))
assert V.snapshot_frame_plot(OUT/'decision_frames.png', frames, title='Release: measured / commanded / hypothetical lip frames', annotations=['Candidate frames are NOT executed or clearance-validated.'])['ok']
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
names=[m['case'] for m in metrics]
for ax, key, title in [(axs[0,0],'exit_projected_slope_deg','Outlet-directed surface slope [deg]'),(axs[0,1],'side_gravity_fraction','Gravity toward side wall [fraction of g]'),(axs[1,0],'lip_displacement_norm_mm','Lip shift if only wrist moves [mm]')]:
    bars=ax.barh(names,[m[key] for m in metrics],color=['#268c67','#bbb','#4d83cc','#d49330','#777','#9a56bc'])
    ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_title(title);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.margins(x=.25)
ax=axs[1,1]
a=np.linspace(math.pi,2*math.pi,150);ax.plot(8.1+20*np.sin(a),145+20*np.cos(a),color='gray',label='fixed inner bowl section')
vs=np.unique(surface[surface_faces.ravel()],axis=0)*1000
ax.plot(vs[:,0][np.argsort(vs[:,0])],vs[:,2][np.argsort(vs[:,0])],color='red',lw=3,label='outlet facet (~7.5 deg)')
for m,c in [(metrics[0],'#268c67'),(metrics[-1],'#9a56bc')]:
    g=m['gravity_unit_link5'];ax.arrow(-4,149,g[0]*10,g[2]*10,width=.15,color=c,length_includes_head=True,label=m['case']+' gravity XZ')
ax.annotate('open side +X',xy=(15,164),xytext=(-7,171),arrowprops=dict(arrowstyle='->'));ax.invert_yaxis();ax.set_aspect('equal');ax.set_xlabel('link5 X [mm]');ax.set_ylabel('link5 Z [mm], tip down');ax.legend(fontsize=8)
fig.suptitle('Offline axis review: actual joint data + S1 CAD | no pellet motion predicted')
fig.tight_layout();fig.savefig(OUT/'release_diagnostic.png',dpi=150);plt.close(fig)

meshes=[];arrows=[];texts=[];expected={'/metadata/run','/coordinate_frames/world_m'};components={}
for i,(name,a,T,label) in enumerate(cases):
    D=T@J@S.K.Trot_z(math.radians(q[5] if a is None else a[5]))
    for part,mat,color in [('fixed',T,[100,180,140,255]),('door',D,[70,120,210,160]),('exit_surface',T,[255,70,40,255])]:
        verts,faces=(surface,surface_faces) if part=='exit_surface' else local[part]
        ent=f'cases/{name}/{part}'
        meshes.append(dict(entity_path=ent,vertices_m=verts@mat[:3,:3].T+mat[:3,3],triangles=faces,color_rgba=color,coordinate_frame='world_m',static=True))
        expected.update({'/'+ent,'/metadata/meshes/'+ent.replace('/','__')})
        components['/'+ent]=['Mesh3D:vertex_positions','Mesh3D:triangle_indices']
    at=T[:3,:3]@center+T[:3,3]
    for tag,vec,col in [('gravity',[0,0,-.022],[180,60,200]),('outlet',T[:3,:3]@exit_tangent*.025,[240,100,30])]:
        ent=f'cases/{name}/{tag}'
        arrows.append(dict(entity_path=ent,origins_m=[at],vectors_m=[vec],colors=col,coordinate_frame='world_m',static=True))
        expected.add('/'+ent);components['/'+ent]=['Arrows3D:vectors','Arrows3D:origins']
    ent=f'cases/{name}/note';texts.append(dict(entity_path=ent,text=json.dumps(metrics[i]),static=True));expected.add('/'+ent)
    expected.update({'/frames/'+name,'/frames/'+name+'/origin'})

def blueprint(mode):
    import rerun.blueprint as B
    return B.Blueprint(B.Vertical(B.Horizontal(*[B.Spatial3DView(origin='/',contents=[f'/cases/{n}/**'],name=title) for n,title in [('actual','Measured release'),('pitch_level','Hypothesis: remove side lean'),('ideal_outward5','Concept: outward5, no IK')]]),B.TextLogView(origin='/cases',contents='/cases/**/note',name='Original float64 metrics; candidates NOT executed')),auto_layout=False,auto_views=False,collapse_panels=True)

old=V.build_rerun_blueprint;V.build_rerun_blueprint=blueprint
try:
    status=V.log_rerun(OUT/'release_tilt.rrd',frames=frames,meshes=meshes,arrows=arrows,events=texts,coordinate_frames=[dict(frame='world_m',parent_frame='tf#/',entity_path='coordinate_frames/world_m')],recording_metadata=report,recording_id='s1_release_tilt_review_01',blueprint_path=OUT/'release_tilt.rbl',blueprint_mode='release_review')
finally:
    V.build_rerun_blueprint=old
save('log_status.json',status)
assert status['ok'],status
v=validate_rerun_artifact(OUT/'release_tilt.rrd',exact_entity_paths=sorted(expected),exact_timeline_names=['blueprint','log_time'],expected_entity_components=components,blueprint_path=OUT/'release_tilt.rbl',screenshot_path=OUT/'rerun_decision.png',screenshot_window_size='2400x1500',cli_path=Path(sys.executable).with_name('rerun'),timeout_s=90)
save('rerun_validation.json',v)
assert v['pass'],v
print(json.dumps({k:report[k] for k in ['last_second_received_rows','actual_tool_lean_deg','outlet_inner_facet_angle_deg','verdict']},indent=2))
print(json.dumps([{k:m[k] for k in ['case','exit_projected_slope_deg','side_gravity_fraction','lip_displacement_norm_mm']} for m in metrics],indent=2))
