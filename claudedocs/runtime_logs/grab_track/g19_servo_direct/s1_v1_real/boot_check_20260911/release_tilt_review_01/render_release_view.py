"""Improve the fixed-jaw inspection camera; preserve the first recording."""
import json, os, sys
from pathlib import Path
import numpy as np
import trimesh

BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'hw_s1_scoop_probe.py').exists())
sys.path.insert(0,str(ROOT))
from roarm_rl import viz_debug as V
from roarm_rl.rerun_contract import validate_rerun_artifact
OUT=BASE/'visual_02';OUT.mkdir(exist_ok=False)
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
a=json.loads((BASE/'analysis.json').read_text())
d=ROOT/'claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1'
fixed=trimesh.load(d/'fixed_ALL.stl',process=False)
wall=trimesh.load(d/'fixed_wall_11.stl',process=False)
inside=np.flatnonzero(wall.face_normals[:,2]<-.5)
n=np.array(a['outlet_inner_facet_normal_link5']);tangent=np.array([-n[2],0,n[0]])
center=wall.triangles_center[inside].mean(0)/1000
meshes=[];arrows=[];frames=[];expected={'/metadata/run','/coordinate_frames/world_m'};components={}
chosen=[m for m in a['cases'] if m['case'] in ('actual','pitch_level','ideal_outward5')]
for m in chosen:
 T=np.array(m['T_world_link5']);name=m['case'];R=T[:3,:3];p=T[:3,3]
 for part,mesh,faces,color in [('fixed',fixed,fixed.faces,[100,180,140,255]),('exit_surface',wall,wall.faces[inside],[255,80,20,255])]:
  ent=f'cases/{name}/{part}'
  meshes.append(dict(entity_path=ent,vertices_m=mesh.vertices/1000@R.T+p,triangles=faces,color_rgba=color,coordinate_frame='world_m',static=True))
  expected.update({'/'+ent,'/metadata/meshes/'+ent.replace('/','__')});components['/'+ent]=['Mesh3D:vertex_positions','Mesh3D:triangle_indices']
 for tag,vec,col in [('gravity',[0,0,-.018],[180,60,200]),('outlet',R@tangent*.015,[240,100,30])]:
  ent=f'cases/{name}/{tag}'
  arrows.append(dict(entity_path=ent,origins_m=[R@center+p],vectors_m=[vec],colors=col,coordinate_frame='world_m',static=True))
  expected.add('/'+ent);components['/'+ent]=['Arrows3D:vectors','Arrows3D:origins']
 if name!='ideal_outward5':
  frames.append(V.frame_from_axes(name,R@np.array([.0081,0,.1666])+p,x_axis=R[:,0],z_axis=R[:,2],role='actual' if name=='actual' else 'target',label=name))
  expected.update({'/frames/'+name,'/frames/'+name+'/origin'})

def bp(mode):
 import rerun.blueprint as B
 views=[]
 for m in chosen:
  T=np.array(m['T_world_link5']);R=T[:3,:3];p=T[:3,3]
  eye=R@np.array([.12,-.035,.105])+p;look=R@np.array([0,0,.146])+p
  views.append(B.Spatial3DView(origin='/',contents=[f"/cases/{m['case']}/**"],name=f"{m['case']}: outlet {m['exit_projected_slope_deg']:.2f} deg",eye_controls=B.EyeControls3D(kind=B.Eye3DKind.Orbital,position=eye.tolist(),look_target=look.tolist(),eye_up=[0,0,1]),spatial_information=B.SpatialInformation(target_frame='tf#/',show_axes=False,show_bounding_box=False)))
 return B.Blueprint(B.Horizontal(*views),auto_layout=False,auto_views=False,collapse_panels=True)

old=V.build_rerun_blueprint;V.build_rerun_blueprint=bp
try: status=V.log_rerun(OUT/'release_tilt.rrd',frames=frames,meshes=meshes,arrows=arrows,coordinate_frames=[dict(frame='world_m',parent_frame='tf#/',entity_path='coordinate_frames/world_m')],recording_metadata={**a,'view_note':'Fixed jaw only, looking through the open side. Full assembly/all6 cases preserved in parent recording. Candidates not executed.'},recording_id='release_tilt_fixed_view_02',blueprint_path=OUT/'release_tilt.rbl',blueprint_mode='release_review_fixed')
finally:V.build_rerun_blueprint=old
(OUT/'log_status.json').write_text(json.dumps(status,indent=2));assert status['ok'],status
v=validate_rerun_artifact(OUT/'release_tilt.rrd',exact_entity_paths=sorted(expected),exact_timeline_names=['blueprint','log_time'],expected_entity_components=components,blueprint_path=OUT/'release_tilt.rbl',screenshot_path=OUT/'rerun_decision.png',screenshot_window_size='2400x1100',cli_path=Path(sys.executable).with_name('rerun'),timeout_s=90)
(OUT/'rerun_validation.json').write_text(json.dumps(v,indent=2));assert v['pass'],v
print('FIXED_JAW_VIEW_CONTRACT_PASS')
