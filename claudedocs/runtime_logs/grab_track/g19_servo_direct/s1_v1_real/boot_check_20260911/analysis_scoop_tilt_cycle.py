"""Offline full-timeline outlet-directed tilt diagnostics from raw serial.
Derived from analysis_scoop.py; no physical commands or input changes.
"""
import argparse,csv,hashlib,json,math,os,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'hw_s1_scoop_probe.py').is_file());sys.path.insert(0,str(ROOT))
import hw_s1_scoop_probe as S
from roarm_rl import viz_debug as V
from roarm_rl.rerun_contract import validate_rerun_artifact

def run(out):
 raw=out/'raw.jsonl';ee=[json.loads(x) for x in raw.read_text().splitlines()];res=json.loads((out/'result.json').read_text());spec=json.loads((out/'plan.json').read_text())
 assert not res['dry_run'] and hashlib.sha256(raw.read_bytes()).hexdigest()==res['raw_sha256']
 obs=[];target=None;tor=None;cmds=[]
 for e in ee:
  if e['ev']=='tx':
   c=e['command'];cmds.append(c)
   if c['T']==107:tor=c['tor']
   if c['T']==122:target=np.array([c[k] for k in ('b','s','e','t','r')]+[180-c['h']])
   if c['T']==121 and target is not None:target[5]=180-c['angle']
  if e['ev']=='rx_json' and e['data'].get('T')==1051:
   d=e['data'];q=np.array([math.degrees(d[k]) for k in ('b','s','e','t','r')]+[180-math.degrees(d['g'])])
   obs.append((e,q,target.copy() if target is not None else q.copy(),tor))
 assert not any('rst:0x' in e.get('text','') for e in ee)
 t0=ee[0]['mono_ns'];t=np.array([(o[0]['mono_ns']-t0)/1e9 for o in obs]);q=np.array([o[1] for o in obs]);tgt=np.array([o[2] for o in obs]);ph=np.array([o[0]['phase'] for o in obs])
 urdf=ROOT/'local_assets/roarm_m3/urdf/roarm_m3_s1_v1.urdf';root=ET.parse(urdf).getroot()
 jo=root.find("joint[@name='link5_to_gripper_link']/origin");J=S.K.Tmat([float(v) for v in jo.attrib['xyz'].split()],[float(v) for v in jo.attrib['rpy'].split()])
 import trimesh
 meshes_local={};geometry_hash={}
 for name,link in [('fixed','grab_fixed'),('door','gripper_link')]:
  m=root.find(f"link[@name='{link}']/visual/geometry/mesh");path=urdf.parent/m.attrib['filename'];mesh=trimesh.load(path,force='mesh',process=False)
  verts=np.asarray(mesh.vertices,float)*np.array([float(v) for v in m.attrib['scale'].split()]);faces=np.asarray(mesh.faces,int)
  meshes_local[name]=(verts,faces);geometry_hash[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 def geometry(qq):
  c=S.chain(qq[:5]);T=c['link4_to_link5'].copy();T[2,3]+=.38;D=T@J@S.K.Trot_z(math.radians(qq[5]));lip=(T@S.LIP_L5)[:3]
  links=np.array([x[:3,3] for x in c.values()]);links[:,2]+=.38
  return T,D,lip,links
 lips=np.array([geometry(qq)[2] for qq in q]);dev=np.max(abs(q[:,:5]-tgt[:,:5]),axis=1)
 keys=['door_deg','lip_height_cm','shoulder_deg','shoulder_load_raw','elbow_load_raw','wrist_load_raw','arm_target_error_deg','torque_command','target_door_deg']
 vals=np.c_[q[:,5],lips[:,2]*100,q[:,1],[o[0]['data']['tS'] for o in obs],[o[0]['data']['tE'] for o in obs],[o[0]['data']['tT'] for o in obs],dev,[o[3] if o[3] is not None else -1 for o in obs],tgt[:,5]]
 # Projection onto the same candidate exit facet used in release_tilt_review_01.
 normal=np.array([math.sin(math.radians(7.5)),0.,-math.cos(math.radians(7.5))]);tangent=np.array([-normal[2],0.,normal[0]])
 slopes=[]
 for qq in q:
  g=geometry(qq)[0][:3,:3].T@np.array([0.,0.,-1.]);slopes.append(math.degrees(math.atan2(float(g@tangent),float(-g@normal))))
 slopes=np.asarray(slopes);keys+=['roll_deg','target_roll_deg','outlet_slope_deg'];vals=np.c_[vals,q[:,4],tgt[:,4],slopes]
 with (out/'feedback.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['sample','host_elapsed_s','mono_ns','phase',*keys,*['q_'+str(j)+'_deg' for j in range(6)]])
  for i,o in enumerate(obs):w.writerow([i,t[i],o[0]['mono_ns'],o[0]['phase'],*vals[i],*q[i]])
 lift=np.flatnonzero(np.char.startswith(ph,'tilt_')|np.char.startswith(ph,'finish_tilt_'));after_ids=np.flatnonzero(ph=='outlet_after_tilt');before_ids=np.flatnonzero(ph=='outlet_baseline')
 if not len(after_ids) and not res['completed']:after_ids=np.flatnonzero(ph==ph[-1])
 if not len(before_ids):before_ids=np.array([0])
 if not len(lift):lift=np.flatnonzero(ph!='preflight')
 assert len(lift) and len(after_ids) and len(before_ids),'baseline/final attempted tilt timeline missing'
 decision=int(after_ids[-1]);pre=int(before_ids[-1])
 report={'n_feedback':len(obs),'duration_s':(ee[-1]['mono_ns']-t0)/1e9,'max_receive_gap_s':float(np.diff(t).max()),
  'raw_sha256':res['raw_sha256'],'geometry_sha256':geometry_hash,'completed':res['completed'],'experiment':spec['experiment'],'stop_reason':res.get('error'),'final_phase':str(ph[-1]),
  'before_alignment_roll_deg':float(q[pre,4]),'after_alignment_roll_deg':float(q[decision,4]),'actual_alignment_roll_change_deg':float(q[decision,4]-q[pre,4]),
  'before_outlet_slope_deg':float(slopes[pre]),'after_outlet_slope_deg':float(slopes[decision]),'outlet_slope_change_deg':float(slopes[decision]-slopes[pre]),
  'door_before_after_deg':[float(q[pre,5]),float(q[decision,5])],'lip_displacement_mm_fk':((lips[decision]-lips[pre])*1000).tolist(),
  'operator_measurements':res.get('measurements',{}),'additional_delivered_mass_g':res.get('additional_delivered_mass_g'),
  'tG_present_count':sum('tG' in o[0]['data'] for o in obs),'scientific_authority':'raw.jsonl and actual transmitted commands; geometry is URDF reconstruction',
  'limits':['No camera, contact-force, temperature, current or target-register measurement.','new_scoop_actually_attempted records actual entry in these raw rows; continuations do not create new scoops. Previous trial residue cleared by user report. Initial pile reset, cup tare and delivered mass not independently measured.',
            'Native receive rows all exported; timestamps are host receipt, not servo clock.','Door geometry uses existing real replay SDK-degree to URDF convention; physical zero/contact not recalibrated.',
            'Target reference before first T122 is initial observation, not target register; torque -1 means not yet commanded.']}
 assert np.all(np.isfinite(tgt[decision]))
 close_ids=np.flatnonzero(ph=='close');lift_ids=np.flatnonzero(ph=='lift8')
 if len(close_ids) and len(lift_ids):
  cb=int(close_ids[-1]);le=int(lift_ids[-1]);report.update(close_torque=spec['close_torque'],before_lift_deg=float(q[cb,5]),after_lift_deg=float(q[le,5]),lift_reopen_deg=float(q[le,5]-q[cb,5]),lift_peak_additional_open_deg=float(max(0,q[lift_ids,5].max()-q[cb,5])))
 report.update(home_reached=res.get('home_reached'),home_actual_q_deg=res.get('home_actual_q_deg'),new_scoop_planned=res.get('new_scoop'),new_scoop_actually_attempted=bool(np.any(ph=='plunge')),final_actual_tool_tilt_deg=float(180-q[decision,1:4].sum()))
 gaps=np.flatnonzero(np.diff(t)>1)+1
 report['unobserved_intervals_s']=[[float(t[i-1]),float(t[i])] for i in gaps]
 report['source_segments']=res.get('source_segments',[])
 if len(gaps):report['limits'].append('Serial-closed intervals are unobserved: Rerun may hold/interpolate visual values between samples, which is not measured motion. Timeline PNG masks these gaps.')
 (out/'analysis.json').write_text(json.dumps(report,indent=2))
 os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib');os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 vis=out/'visual_01';vis.mkdir(exist_ok=True)
 fig,ax=plt.subplots(3,1,figsize=(14,10),sharex=True)
 def visible(a):
  v=np.array(a,float).copy();v[gaps]=np.nan;return v
 ax[0].plot(t,visible(slopes),label='outlet slope from actual joints');ax[0].axhline(27.5,ls='--',label='planned final slope27.5deg');ax[0].legend();ax[0].set_ylabel('Outlet slope [deg]')
 ax[1].plot(t,visible(lips[:,2]*100));ax[1].axhline(26,ls='--',label='pellet surface26cm');ax[1].axhline(38.5,ls=':',label='box top38.5cm');ax[1].legend();ax[1].set_ylabel('FK lip above floor [cm]')
 ax[2].plot(t,visible(vals[:,3]),label='shoulder raw load');ax[2].plot(t,visible(vals[:,4]),label='elbow raw load');ax[2].legend();ax[2].set_ylabel('Servo load [raw]');ax[2].set_xlabel('Host elapsed [s]')
 for a in ax:
  for i in gaps:a.axvspan(t[i-1],t[i],color='gray',alpha=.15)
 for a in ax:a.axvspan(t[lift[0]],t[lift[-1]],alpha=.2,color='orange');a.grid(alpha=.2)
 fig.suptitle(f"S1 full cycle: discharge alignment change {report['actual_alignment_roll_change_deg']:.3f} deg | received rows {len(obs)}")
 fig.tight_layout();fig.savefig(vis/'timeline.png',dpi=150);plt.close(fig)
 frames=[]
 for name,qq,role in [('commanded_tilt',tgt[decision],'target'),('actual_tilt',q[decision],'actual')]:
  T,D,lip,links=geometry(qq);frames.append(V.frame_from_axes(name,lip,x_axis=T[:3,0],z_axis=T[:3,2],role=role,label=name))
 assert V.snapshot_frame_plot(vis/'decision_frames.png',frames,title='Outlet-directed tilt: commanded vs measured-joint FK lip frames',annotations=['FK diagnostic, not external pose measurement.'])['ok']
 points=[];scalars=[];events=[];meshes=[];arrows=[]
 sampled={name:vs[np.linspace(0,len(vs)-1,min(600,len(vs)),dtype=int)] for name,(vs,fs) in meshes_local.items()}
 for i,o in enumerate(obs):
  tm={'sequence':{'sample':i},'duration':{'host_elapsed_s':float(t[i])}};T,D,lip,links=geometry(q[i])
  points.append({'entity_path':'robot/joints','positions_m':links,'radii':.005,'colors':[130,130,130],'coordinate_frame':'world_m',**tm})
  for name,mat,col in [('fixed',T,[40,180,100]),('door',D,[50,120,240])]:
   pos=sampled[name]@mat[:3,:3].T+mat[:3,3]
   points.append({'entity_path':'tool/'+name,'positions_m':pos,'radii':.0007,'colors':col,'coordinate_frame':'world_m',**tm})
  for key,val in zip(keys,vals[i]):scalars.append({'entity_path':'metrics/'+key,'value':float(val),**tm})
 for label,qq,color in [('target',tgt[decision],[230,160,40,100]),('actual',q[decision],[40,150,220,255])]:
  T,D,lip,links=geometry(qq)
  for name,mat in [('fixed',T),('door',D)]:
   vs,fs=meshes_local[name];ent='decision/'+label+'_'+name
   meshes.append({'entity_path':ent,'vertices_m':vs@mat[:3,:3].T+mat[:3,3],'triangles':fs,'color_rgba':color,'coordinate_frame':'world_m','static':True})
 for e in ee:
  if e['ev'] not in ('rx_json','rx_text') and not(e['ev']=='tx' and e['command']['T']==105):
   events.append({'entity_path':'events/commands','text':json.dumps(e),'duration':{'host_elapsed_s':(e['mono_ns']-t0)/1e9}})
 def bp(mode):
  import rerun.blueprint as B
  return B.Blueprint(B.Vertical(B.Horizontal(B.TimeSeriesView(origin='/metrics',contents=['/metrics/outlet_slope_deg'],name='Outlet slope from actual joints'),B.Spatial3DView(origin='/',contents=['/decision/**','/frames/**'],name='Outlet-directed tilt target vs actual S1 geometry')),
   B.Horizontal(B.TimeSeriesView(origin='/metrics',contents=['/metrics/lip_height_cm'],name='Lip height from measured joints [cm]'),B.TimeSeriesView(origin='/metrics',contents=['/metrics/torque_command'],name='Issued gripper torque limit')),
   B.Horizontal(B.Spatial3DView(origin='/',contents=['/tool/**','/robot/**'],name='Full measured feedback timeline'),B.TextLogView(origin='/events',contents='/events/**',name='Commands / stages'))),
   B.TimePanel(timeline='host_elapsed_s',play_state='paused'),auto_layout=False,auto_views=False,collapse_panels=True)
 old=V.build_rerun_blueprint;V.build_rerun_blueprint=bp
 try:status=V.log_rerun(vis/'release.rrd',frames=frames,points=points,meshes=meshes,scalar_trace=scalars,events=events,coordinate_frames=[{'frame':'world_m','parent_frame':'tf#/','entity_path':'coordinate_frames/world_m'}],recording_metadata=report,recording_id='s1_outlet_tilt_'+out.name,blueprint_path=vis/'release.rbl',blueprint_mode='scoop_real')
 finally:V.build_rerun_blueprint=old
 (vis/'log_status.json').write_text(json.dumps(status,indent=2,default=str));assert status['ok'],status
 ents={'/metadata/run','/coordinate_frames/world_m','/robot/joints','/tool/fixed','/tool/door','/events/commands'}|{'/metrics/'+k for k in keys}
 for label in ['target','actual']:
  for name in ['fixed','door']:ents|={'/decision/'+label+'_'+name,'/metadata/meshes/decision__'+label+'_'+name}
 ents|={'/frames/'+name+s for name in ['commanded_tilt','actual_tilt'] for s in ['','/origin']}
 comp={'/tool/'+name:['Points3D:positions'] for name in ['fixed','door']};comp.update({'/metrics/'+k:['Scalars:scalars'] for k in keys})
 comp.update({'/decision/'+lab+'_'+name:['Mesh3D:vertex_positions','Mesh3D:triangle_indices'] for lab in ['target','actual'] for name in ['fixed','door']})
 val=validate_rerun_artifact(vis/'release.rrd',exact_entity_paths=sorted(ents),exact_timeline_names=['blueprint','log_time','sample','host_elapsed_s'],expected_entity_components=comp,blueprint_path=vis/'release.rbl',screenshot_path=vis/'release_rerun.png',screenshot_window_size='2800x1800',cli_path=Path(sys.executable).with_name('rerun'),timeout_s=120)
 from rerun.experimental import RrdReader
 reader=RrdReader(vis/'release.rrd');checks={}
 for entity,component in [('/tool/fixed','Points3D:positions'),('/tool/door','Points3D:positions'),('/robot/joints','Points3D:positions')]+[('/metrics/'+k,'Scalars:scalars') for k in keys]:
  ids=[];times=[];vs=[]
  for chunk in reader.stream().filter(content=entity,has_timeline='sample',components=component):
   rb=chunk.to_record_batch();field=[f.name for f in rb.schema if (f.metadata or {}).get(b'rerun:component',b'').decode()==component][0]
   ids+=rb.column('sample').to_pylist();times+=rb.column('host_elapsed_s').cast('int64').to_pylist();vs+=rb.column(field).to_pylist()
  order=np.argsort(ids);ok=np.array_equal(np.array(ids)[order],np.arange(len(obs))) and np.array_equal(np.array(times)[order],np.rint(t*1e9).astype('int64'))
  if component=='Scalars:scalars':ok=ok and np.array_equal(np.array([v[0] for v in vs])[order],vals[:,keys.index(entity.split('/')[-1])])
  checks[entity]={'pass':bool(ok),'rows':len(ids),'expected':len(obs)}
 val['coverage_readback']=checks;val['pass']=bool(val['pass'] and all(c['pass'] for c in checks.values()))
 (vis/'rerun_validation.json').write_text(json.dumps(val,indent=2,default=str));print(json.dumps({'pass':val['pass'],**report},indent=2));assert val['pass']

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();run(a.out.resolve())
