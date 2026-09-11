"""No hardware: preserve door0 at HOME; full-cycle order and final failure."""
import contextlib,io,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import hw_scoop_tilt_cycle as H

pp=Path(sys.argv[1]);p=json.loads(pp.read_text())
class Fake:
    fail=False;instances=[]
    def __init__(self,out,command_validator):
        self.q=p['preflight_q_deg'].copy();self.validate=command_validator;self.commanded_door_deg=None
        self.tx=[];self.closed=False;Fake.instances.append(self)
    def joints_angle_get(self):return self.q.copy()
    def record(self,*args,**kw):pass
    def send(self,c):
        self.validate(c)
        if c['T']==122 and self.commanded_door_deg is not None:
            c={**c,'h':180-self.commanded_door_deg};self.validate(c)
        self.tx.append(c)
        if c['T']==122:
            self.q=[c[k] for k in ['b','s','e','t','r']]+[180-c['h']]
            if self.fail and self.phase=='tilt_40':self.q[1]+=5.01
        if c['T']==121:self.commanded_door_deg=180.-c['angle'];self.q[5]=self.commanded_door_deg
    def close(self):self.closed=True

results=[]
with tempfile.TemporaryDirectory(prefix='s1-cycle-regression-',dir='/tmp') as d:
    for fail in [False,True]:
        Fake.fail=fail
        with patch.object(H,'RecordedArm',Fake),patch.object(H.time,'sleep',lambda _:None),contextlib.redirect_stdout(io.StringIO()):
            r=H.run(pp,Path(d)/str(fail))
        a=Fake.instances[-1];assert a.closed and r['home_reached']==p['new_scoop'],r
        assert r['completed']==(not fail),r
        assert a.tx==[s['command'] for s in p['steps']]
        names=[s['name'] for s in p['steps']]
        if p['new_scoop']:
            assert names.index('home')<names.index('plunge')<names.index('outlet_baseline')<names.index('tilt_40')
            assert r['home_actual_q_deg']==[0,0,90,0,0,0]
        else:
            assert 'home' not in names and 'plunge' not in names
            assert names[0] in ['loaded_retract','transport_recovery_p1'] and r['new_scoop'] is False
        if fail:assert 'settled pose outside gate' in r['error']
        results.append(dict(test='final_tracking_failure' if fail else 'success_home_door0_order',pass_=True,commands=len(a.tx),home_reached=r['home_reached'],completed=r['completed']))
report=dict(real_serial_opens=0,tests=results)
(pp.parent/'mock_execution_audit.json').write_text(json.dumps(report,indent=2))
print('CYCLE_HOME_EXPLICIT_DOOR0_ORDER_AND_FAILURE_GUARD_PASS; real serial opens=0')
