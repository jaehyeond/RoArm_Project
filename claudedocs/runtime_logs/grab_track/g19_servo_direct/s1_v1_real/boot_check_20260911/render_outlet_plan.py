"""S1 planned/observed outlet geometry, with an immutable Rerun file sink."""
import json, os, sys
from pathlib import Path
import numpy as np
import plan_outlet_tilt as P
from roarm_rl import viz_debug as V
from roarm_rl.rerun_contract import validate_rerun_artifact

def render(out,cases,timeline):
    out.mkdir(exist_ok=False)
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    frames=[];meshes=[];points=[];scalars=[];arrows=[]
    expected={'/metadata/run','/coordinate_frames/world_m'};components={}
    for name,q,actual in cases:
        T,world,mats=P.geometry(q);R=T[:3,:3];p=T[:3,3]
        frames.append(V.frame_from_axes(name,(T@P.S.LIP_L5)[:3],x_axis=R[:,0],z_axis=R[:,2],role='actual' if actual else 'target',label=name))
        expected.update({'/frames/'+name,'/frames/'+name+'/origin'})
        for part,vertices in world.items():
            ent='cases/'+name+'/'+part
            color=[100,175,135,255] if part=='grab_fixed' else [70,130,230,120] if part=='gripper_link' else [160,160,170,180]
            meshes.append(dict(entity_path=ent,vertices_m=vertices,triangles=P.PARTS[part][1],color_rgba=color,coordinate_frame='world_m',static=True))
            expected.update({'/'+ent,'/metadata/meshes/'+ent.replace('/','__')});components['/'+ent]=['Mesh3D:vertex_positions','Mesh3D:triangle_indices']
        for tag,vec,col in [('gravity',[0,0,-.025],[180,60,200]),('outlet',R@np.array([1,0,0])*.025,[240,100,30])]:
            ent='cases/'+name+'/'+tag
            arrows.append(dict(entity_path=ent,origins_m=[R@np.array([.0081,0,.1466])+p],vectors_m=[vec],colors=col,coordinate_frame='world_m',static=True))
            expected.add('/'+ent);components['/'+ent]=['Arrows3D:vectors','Arrows3D:origins']
    for i,row in enumerate(timeline):
        q=row['q_deg'];T,world,mats=P.geometry(q);tm={'sequence':{'sample':i}}
        for part in ['grab_fixed','gripper_link','link2','link3','link4','link5']:
            verts=world[part];verts=verts[np.linspace(0,len(verts)-1,min(100,len(verts)),dtype=int)]
            ent='timeline/'+part;points.append(dict(entity_path=ent,positions_m=verts,radii=.001,colors=[60,170,130],coordinate_frame='world_m',**tm));expected.add('/'+ent);components['/'+ent]=['Points3D:positions']
        scalars.append(dict(entity_path='metrics/outlet_slope_deg',value=P.slope(q),**tm))
    expected.add('/metrics/outlet_slope_deg');components['/metrics/outlet_slope_deg']=['Scalars:scalars']
    def bp(mode):
        import rerun.blueprint as B
        views=[]
        for name,q,actual in cases:
            T,world,mats=P.geometry(q);R=T[:3,:3];p=T[:3,3]
            views.append(B.Spatial3DView(origin='/',contents=[f'/cases/{name}/grab_fixed',f'/cases/{name}/gripper_link',f'/cases/{name}/gravity',f'/cases/{name}/outlet'],name=f'{name}: {P.slope(q):.1f}deg outlet',eye_controls=B.EyeControls3D(kind=B.Eye3DKind.Orbital,position=(R@np.array([.16,-.055,.08])+p).tolist(),look_target=(R@np.array([0,0,.145])+p).tolist(),eye_up=[0,0,1]),spatial_information=B.SpatialInformation(target_frame='tf#/',show_axes=False,show_bounding_box=False)))
        return B.Blueprint(B.Vertical(B.Horizontal(*views),B.Horizontal(B.Spatial3DView(origin='/',contents=[f'/cases/{cases[-1][0]}/**'],name='Final arm and tool configuration'),B.TimeSeriesView(origin='/metrics',contents='/metrics/**',name='Complete path: outlet slope'))),B.TimePanel(timeline='sample',play_state='paused'),auto_layout=False,auto_views=False,collapse_panels=True)
    old=V.build_rerun_blueprint;V.build_rerun_blueprint=bp
    try:status=V.log_rerun(out/'outlet.rrd',frames=frames,meshes=meshes,points=points,arrows=arrows,scalar_trace=scalars,coordinate_frames=[dict(frame='world_m',parent_frame='tf#/',entity_path='coordinate_frames/world_m')],recording_metadata=dict(cases=[dict(name=n,q_deg=q,actual=a) for n,q,a in cases],timeline_rows=len(timeline),scope='URDF/FK; no camera or cup registration'),recording_id='s1_outlet_'+out.parent.name,blueprint_path=out/'outlet.rbl',blueprint_mode='outlet')
    finally:V.build_rerun_blueprint=old
    assert status['ok'],status
    report=validate_rerun_artifact(out/'outlet.rrd',exact_entity_paths=sorted(expected),exact_timeline_names=['blueprint','log_time','sample'],expected_entity_components=components,blueprint_path=out/'outlet.rbl',screenshot_path=out/'decision.png',screenshot_window_size='2400x1500',cli_path=Path(sys.executable).with_name('rerun'),timeout_s=90)
    from rerun.experimental import RrdReader
    ids=[];values=[]
    for chunk in RrdReader(out/'outlet.rrd').stream().filter(content='/metrics/outlet_slope_deg',has_timeline='sample',components='Scalars:scalars'):
        rb=chunk.to_record_batch();field=next(f.name for f in rb.schema if (f.metadata or {}).get(b'rerun:component')==b'Scalars:scalars')
        ids+=rb.column('sample').to_pylist();values+=rb.column(field).to_pylist()
    order=np.argsort(ids);ok=np.array_equal(np.array(ids)[order],np.arange(len(timeline))) and np.array_equal(np.array([x[0] for x in values])[order],np.array([P.slope(r['q_deg']) for r in timeline]))
    report['complete_slope_readback']={'pass':bool(ok),'rows':len(ids)};report['pass']=bool(report['pass'] and ok)
    (out/'validation.json').write_text(json.dumps(report,indent=2));assert report['pass'],report
    print('OUTLET_RERUN_PASS',len(ids),flush=True)

if __name__=='__main__':
    d=Path(sys.argv[1]);p=json.loads((d/'plan.json').read_text());t=json.loads((d/'interpolation.json').read_text())
    render(d/'visual_01',[('current',p['preflight_q_deg'],True),('aligned',p['path'][23]['q_deg'],False),('tilt20',p['path'][-1]['q_deg'],False)],t)
