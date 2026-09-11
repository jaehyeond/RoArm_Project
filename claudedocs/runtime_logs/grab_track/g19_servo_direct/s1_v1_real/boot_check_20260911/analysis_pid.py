"""PID raw-data analysis and complete Rerun export; no hardware imports/commands."""
import argparse, csv, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'hw_s1_scoop_probe.py').is_file())
sys.path.insert(0,str(ROOT))
from roarm_rl import viz_debug as V
from roarm_rl.rerun_contract import validate_rerun_artifact
import hw_s1_scoop_probe as S

def analyze(out):
    raw=out/'raw.jsonl'; events=[json.loads(x) for x in raw.read_text().splitlines()]
    result=json.loads((out/'result.json').read_text())
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==result['raw_sha256']
    assert result['completed'] and result['restore_p16_command_sent'] and not result['error']
    tx=[e['command'] for e in events if e['ev']=='tx']
    assert [x for x in tx if x['T']!=105]==[{'T':108,'joint':2,'p':p,'i':0} for p in [8,48,8,48,16]]
    assert not any('POWERON_RESET' in e.get('text','') or 'rst:0x' in e.get('text','') for e in events)
    rows=[e for e in events if e['ev']=='rx_json' and e['data'].get('T')==1051]
    assert len(rows)==result['n_feedback']
    t0=events[0]['mono_ns']; t=np.array([(e['mono_ns']-t0)/1e9 for e in rows])
    q=np.array([[math.degrees(e['data'][k]) for k in ('b','s','e','t','r')]+[180-math.degrees(e['data']['g'])] for e in rows])
    phase=np.array([e['phase'] for e in rows]); ref=np.array(result['initial_reference_deg'])
    keys=['b_deg','s_deg','e_deg','t_deg','r_deg','gripper_open_deg','tB_raw','tS_raw','tE_raw','tT_raw','tR_raw']
    values=np.c_[q,[[e['data'][k] for k in ('tB','tS','tE','tT','tR')] for e in rows]]
    pmap={'preflight':None,'baseline':None,'p8_a':8,'p48_a':48,'p8_b':8,'p48_b':48,'restore_p16':16}
    with (out/'feedback.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['sample','host_elapsed_s','wall_time_ns','host_mono_ns','phase','P_command_I0',*keys,'tG_raw'])
        for i,e in enumerate(rows):w.writerow([i,t[i],e['t_ns'],e['mono_ns'],e['phase'],pmap[e['phase']],*values[i],''])
    stats={}
    for name in result['phase_statistics']:
        idx=np.flatnonzero(phase==name); idx=idx[t[idx]>=t[idx[-1]]-2]
        stats[name]={'n':len(idx),'shoulder_mean_deg':float(q[idx,1].mean()),'shoulder_range_deg':float(np.ptp(q[idx,1])),
                     'shoulder_load_raw_mean':float(values[idx,7].mean())}
        assert abs(stats[name]['shoulder_mean_deg']-result['phase_statistics'][name]['mean_shoulder_deg'])<1e-12
    deltas=[stats['p8_'+c]['shoulder_mean_deg']-stats['p48_'+c]['shoulder_mean_deg'] for c in 'ab']
    report={'raw_sha256':result['raw_sha256'],'n_feedback':len(rows),'phase_statistics_last2s':stats,'paired_P8_minus_P48_deg':deltas,
            'max_abs_joint_drift_deg':float(np.max(abs(q-ref))),'max_feedback_interval_s':float(np.diff(t).max()),
            'verdict':'REPEATED_P_COMMAND_ASSOCIATED_RESPONSE_AT_INITIAL_POSE',
            'limitations':['Two alternating comparisons, not independent trials; no significance claim.',
            'Observed quantized servo feedback only; no register P/D/I or target readback.',
            'Restore P16 command sent; electrical/register application not directly read.',
            'No sensor clock or tG; receive timestamps and arm load units only.',
            'No permanent setting, loaded scoop, high-P trajectory, or optimum tuning claim.']}
    (out/'analysis.json').write_text(json.dumps(report,indent=2))
    out=out/'visual_02'; out.mkdir(exist_ok=True)
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,1,figsize=(14,10),sharex=True)
    for name,col in [('baseline','#777777'),('p8_a','#ffbf00'),('p48_a','#3aa5df'),('p8_b','#ffbf00'),('p48_b','#3aa5df'),('restore_p16','#92c879')]:
        ix=np.flatnonzero(phase==name)
        for ax in axs:ax.axvspan(t[ix[0]],t[ix[-1]],color=col,alpha=.18)
        axs[0].text(t[ix[0]],1.88,name,fontsize=9)
    axs[0].plot(t,q[:,1]); axs[0].axhline(ref[1],ls='--',color='gray',label='initial observed reference')
    axs[0].set_ylabel('Shoulder [deg]'); axs[0].set_ylim(.95,2); axs[0].legend(loc='lower right')
    for j in [0,2,3,4,5]:axs[1].plot(t,q[:,j]-ref[j],label=keys[j])
    axs[1].set_ylabel('Other joint change [deg]'); axs[1].legend(ncol=5)
    axs[2].plot(t,values[:,7]); axs[2].set_ylabel('Shoulder load [raw]')
    pv=np.array([np.nan if pmap[e['phase']] is None else pmap[e['phase']] for e in rows])
    axs[3].step(t,pv,where='post'); axs[3].set_ylabel('Issued P (I=0)'); axs[3].set_xlabel('Host receive elapsed time [s]')
    fig.suptitle(f'Static shoulder P perturbation | {len(rows)} raw feedback rows | repeat differences {deltas[0]:.6f}, {deltas[1]:.6f} deg')
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'pid_timeseries.png',dpi=150);plt.close(fig)
    # Geometry is a kinematic diagnostic, with floor defined by user-measured base38cm.
    def geom(qq):
        c=S.chain(qq[:5]); links=np.array([x[:3,3] for x in c.values()]); links[:,2]+=.38
        T=c['link4_to_link5']; lip=(T@S.LIP_L5)[:3];lip[2]+=.38
        return np.vstack([links,lip]),lip,T[:3,:3]
    decision=int(np.argmax(abs(q[:,1]-ref[1]))); rp,rl,rr=geom(ref); ap,al,ar=geom(q[decision])
    frames=[V.frame_from_axes('initial_hold_reference',rl,x_axis=rr[:,0],z_axis=rr[:,2],role='target',label='Initial observed hold reference (not target register)'),
            V.frame_from_axes('decision_actual',al,x_axis=ar[:,0],z_axis=ar[:,2],role='actual',label=f'Actual at sample {decision}')]
    snap=V.snapshot_frame_plot(out/'decision_frames.png',frames,title='PID hold: initial reference vs greatest observed shoulder change',
        annotations=['Reference = initial measured pose; actuator target register unavailable.',f'Actual sample {decision}, phase {phase[decision]}, t={t[decision]:.3f}s'])
    assert snap['ok']
    points=[]; arrows=[]; scalars=[]; logs=[]
    for i,e in enumerate(rows):
        timing={'sequence':{'sample':i},'duration':{'host_elapsed_s':float(t[i])}}
        pts,lip,R=geom(q[i]); points.append({'entity_path':'robot/joints_lip','positions_m':pts,'radii':.006,'colors':[50,170,240],'coordinate_frame':'world_m',**timing})
        arrows.append({'entity_path':'robot/links','origins_m':pts[:-1],'vectors_m':np.diff(pts,axis=0),'radii':.003,'colors':[50,170,240],'coordinate_frame':'world_m',**timing})
        for k,v in zip(keys,values[i]):scalars.append({'entity_path':'metrics/'+k,'value':float(v),**timing})
        if pmap[e['phase']] is not None:scalars.append({'entity_path':'metrics/P_command','value':pmap[e['phase']],**timing})
    for name,pts,lip,R,color in [('reference',rp,rl,rr,[250,130,20]),('actual',ap,al,ar,[50,170,240])]:
        points.append({'entity_path':'decision/'+name,'positions_m':pts,'radii':.005,'colors':color,'coordinate_frame':'world_m','static':True})
        arrows.append({'entity_path':'decision/'+name+'_axes','origins_m':np.repeat(lip[None],3,axis=0),'vectors_m':R.T*.025,'colors':color,'coordinate_frame':'world_m','static':True})
    for e in events:
        if e['ev'] not in ('rx_json','rx_text'):
            logs.append({'entity_path':'events/commands','text':json.dumps({k:v for k,v in e.items() if k not in ('t_ns','mono_ns')}),
                'duration':{'host_elapsed_s':(e['mono_ns']-t0)/1e9}})
    def blueprint(mode):
        import rerun.blueprint as B
        return B.Blueprint(B.Vertical(
            B.Horizontal(B.TimeSeriesView(origin='/metrics',contents=['/metrics/s_deg'],name='Shoulder angle [deg]'),
                         B.Spatial3DView(origin='/',contents=['/decision/**','/frames/**'],name=f'Decision sample {decision}: orange reference, blue actual')),
            B.Horizontal(B.TimeSeriesView(origin='/metrics',contents=['/metrics/P_command'],name='Issued P; I=0; baseline unread'),
                         B.TimeSeriesView(origin='/metrics',contents=['/metrics/tS_raw'],name='Shoulder load [raw units]')),
            B.Horizontal(B.Spatial3DView(origin='/',contents=['/robot/**'],name='Full feedback timeline'),
                         B.TextLogView(origin='/events',contents='/events/**',name='Actual commands / phase boundaries'))),
            B.TimePanel(timeline='host_elapsed_s',play_state='paused'),auto_layout=False,auto_views=False,collapse_panels=True)
    old=V.build_rerun_blueprint;V.build_rerun_blueprint=blueprint
    try:
        status=V.log_rerun(out/'pid.rrd',frames=frames,points=points,arrows=arrows,scalar_trace=scalars,events=logs,
            coordinate_frames=[{'frame':'world_m','parent_frame':'tf#/','entity_path':'coordinate_frames/world_m'}],
            recording_metadata=report,recording_id='s1_pid_hold_'+out.name,blueprint_path=out/'pid.rbl',blueprint_mode='pid_hold')
    finally:V.build_rerun_blueprint=old
    (out/'log_status.json').write_text(json.dumps(status,indent=2,default=str))
    assert status['ok'],status
    ents={'/metadata/run','/coordinate_frames/world_m','/robot/joints_lip','/robot/links','/events/commands','/metrics/P_command'}|{'/metrics/'+k for k in keys}
    ents|={'/decision/'+n+s for n in ['reference','actual'] for s in ['','_axes']}
    ents|={'/frames/'+n+s for n in ['initial_hold_reference','decision_actual'] for s in ['','/origin']}
    comp={'/robot/joints_lip':['Points3D:positions'],'/robot/links':['Arrows3D:origins','Arrows3D:vectors'],
          '/events/commands':['TextLog:text'],'/metrics/P_command':['Scalars:scalars']}
    comp.update({'/metrics/'+k:['Scalars:scalars'] for k in keys})
    val=validate_rerun_artifact(out/'pid.rrd',exact_entity_paths=sorted(ents),exact_timeline_names=['blueprint','log_time','sample','host_elapsed_s'],
        expected_entity_components=comp,blueprint_path=out/'pid.rbl',screenshot_path=out/'pid_rerun.png',screenshot_window_size='2800x1800',
        cli_path=Path(sys.executable).with_name('rerun'),expected_version='0.34.1',timeout_s=120)
    from rerun.experimental import RrdReader
    reader=RrdReader(out/'pid.rrd'); coverage={}
    for entity,component in [('/robot/joints_lip','Points3D:positions'),('/robot/links','Arrows3D:vectors')]+[('/metrics/'+k,'Scalars:scalars') for k in keys]+[('/metrics/P_command','Scalars:scalars')]:
        ids=[];ts=[];vs=[]
        for chunk in reader.stream().filter(content=entity,has_timeline='sample',components=component):
            rb=chunk.to_record_batch(); field=[f.name for f in rb.schema if (f.metadata or {}).get(b'rerun:component',b'').decode()==component][0]
            ids+=rb.column('sample').to_pylist();ts+=rb.column('host_elapsed_s').cast('int64').to_pylist();vs+=rb.column(field).to_pylist()
        wanted=np.flatnonzero(np.isfinite(pv)) if entity.endswith('/P_command') else np.arange(len(rows))
        order=np.argsort(ids);good=np.array_equal(np.array(ids)[order],wanted) and np.array_equal(np.array(ts)[order],np.rint(t[wanted]*1e9).astype('int64'))
        if component=='Scalars:scalars':
            expected=pv[wanted] if entity.endswith('/P_command') else values[wanted,keys.index(entity.split('/')[-1])]
            good=good and np.array_equal(np.array([v[0] for v in vs])[order],expected)
        coverage[entity]={'pass':bool(good),'observed_rows':len(ids),'expected_rows':len(wanted)}
    val['coverage_readback']=coverage;val['log_status']=status;val['pass']=bool(val['pass'] and all(x['pass'] for x in coverage.values()))
    (out/'rerun_validation.json').write_text(json.dumps(val,indent=2,default=str))
    print(json.dumps({'pass':val['pass'],'analysis':report,'rrd':str(out/'pid.rrd')},indent=2))
    assert val['pass'],val.get('errors')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();analyze(a.out.resolve())
