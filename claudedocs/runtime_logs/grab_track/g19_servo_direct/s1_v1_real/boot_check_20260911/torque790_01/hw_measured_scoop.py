"""One measured scoop using the existing Manual motion primitives.

Output first-lift reopen under selected torque, then common790 transport.
No automatic retry/chatter. No automatic HOME on failure or exit.
"""
import argparse, functools, hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'hw_s1_manual.py').is_file());sys.path.insert(0,str(ROOT))
import hw_s1_manual as M
from hw_recorded_arm import RecordedArm

def run(out,tor,dry):
    out.mkdir(parents=True,exist_ok=False)
    M.S.OUT_DIR=str(out);M.POS_FILE=str(out/'unused_positions.json');M.MASS_FILE=str(out/'mass.jsonl')
    M.solve_fast=functools.lru_cache(maxsize=128)(M.solve_fast)
    a=SimpleNamespace(sim=True,base_cm=38,pellet_cm=26,boxtop_cm=38.5,travel_cm=45,box_x_cm=35,port='/dev/ttyUSB0')
    m=M.Manual(a);arm=None
    spec={'close_torque':tor,'I':0,'shoulder_P_command_previously_restored':16,'plunge_cm':2.5,'base_cm':38,'pellet_cm':26,'boxtop_cm':38.5,'box_x_cm':35,
          'close_wait_s':1,'after_lift_wait_s':1,'no_chatter':True,'transport_torque':790,'900_time_cap_s':15,
          'door_target_contract':'preserve last explicit T121 across T122 arm moves; reactive fix after torque900_02',
          'initial_heightmap':'unavailable; no Kinect connected','pile_reset':'not independently observed; sequential scoop comparison is exploratory',
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('hw_recorded_arm.py'),ROOT/'hw_s1_manual.py',ROOT/'hw_s1_scoop_probe.py',ROOT/'safety_p0_guards.py']}}
    (out/'plan.json').write_text(json.dumps(spec,indent=2))
    # Solve and gate required target poses before serial access.
    targets={}
    for height in [45,31,26,23.5,34,40,35,30]:
        z=m.floor+height/100;sol=M.solve_fast(.35,z)
        if sol is None or sol[0]>.01:raise RuntimeError(f'Unreachable height {height}: {sol}')
        targets[str(height)]={'q5':sol[1],'ik_error_m':sol[0],'vertical_cos':sol[2]}
    (out/'waypoints.json').write_text(json.dumps(targets,indent=2))
    def stage(name):
        print('STAGE '+name,flush=True)
        if arm:arm.phase=name;arm.record('stage')
        m.log(ev='stage',name=name)
    def check(ok):
        if not ok:raise RuntimeError('Existing movement gate rejected stage; hold position')
    result={'completed':False,'dry_run':dry,'close_torque':tor}
    try:
        if not dry:
            arm=RecordedArm(out);m.sim=False;m.arm=arm
            q=m.read()
            if min(max(abs(x-y) for x,y in zip(q[:5],p)) for p in [M.HOME,M.P1])>6:raise RuntimeError('initial pose neither HOME nor P1; hold')
            if not 0<=q[5]<=30:raise RuntimeError('initial door angle outside S1 limit')
            m.torque(200)
        stage('p1');check(m.goto_q('p1',M.P1))
        stage('above');check(m.goto_xyz('above',.35,0,m.travel))
        for height,name in [(31,'5cm_above'),(26,'surface')]:
            stage(name);check(m.goto_xyz(name,.35,0,m.floor+height/100))
        stage('open');m.door(30,tor=200)
        stage('plunge');check(m.goto_xyz('plunge',.35,0,m.pellet-.025))
        stage('close');close=m.door(0,tor=tor);time.sleep(1)
        before=m.read()[5]
        stage('lift8');check(m.goto_xyz('lift8',.35,0,m.pellet+.08));time.sleep(1);lift=m.read()[5]
        result.update(close_deg=close,before_lift_deg=before,after_lift_deg=lift,reopen_deg=lift-before)
        stage('common_transport_790');m.torque(790)
        check(m.goto_xyz('travel',.35,0,m.travel))
        stage('place');check(m.place(base_deg=90))
        stage('complete_p1');result['completed']=True;result['final_q_deg']=m.read()
    except BaseException as e:
        result['error']=repr(e);print('STOP_HOLD '+repr(e),flush=True)
    finally:
        if arm:arm.close()
        if (out/'raw.jsonl').exists():result['raw_sha256']=hashlib.sha256((out/'raw.jsonl').read_bytes()).hexdigest()
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    return 0 if result['completed'] else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--torque',required=True,type=int,choices=[900,790]);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    raise SystemExit(run(a.out,a.torque,a.dry_run))
