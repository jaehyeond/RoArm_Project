"""Device-free regression: final tracking failure stays a failed held run."""
import json, tempfile
from pathlib import Path
from unittest.mock import patch
import hw_outlet_tilt as H

BASE=Path(__file__).resolve().parent
plan_path=BASE/'outlet_tilt_01/plan_01/plan.json'
plan=json.loads(plan_path.read_text())

class FakeArm:
    instances=[]
    def __init__(self,out,command_validator):
        self.q=plan['preflight_q_deg'].copy();self.check=command_validator
        self.commands=[];self.closed=False;self.__class__.instances.append(self)
    def joints_angle_get(self):return self.q.copy()
    def record(self,*args,**kwargs):pass
    def send(self,c):
        self.check(c);self.commands.append(c)
        if c['T']==122:
            self.q=[c[k] for k in ('b','s','e','t','r')]+[30.]
            if c==plan['path'][-1]['command']:self.q[1]+=5.01
    def close(self):self.closed=True

with tempfile.TemporaryDirectory(prefix='s1-outlet-regression-',dir='/tmp') as d:
    with patch.object(H,'RecordedArm',FakeArm),patch.object(H.time,'sleep',lambda _:None):
        result=H.run(plan_path,Path(d)/'tracking_failure',False)
    a=FakeArm.instances[-1]
    assert result['completed'] is False and 'settled pose outside gate' in result['error']
    assert [c for c in a.commands if c['T']==122]==[r['command'] for r in plan['path']]
    assert a.commands[0]=={'T':107,'tor':200} and len(a.commands)==65
    assert a.closed and result['no_return_no_close_no_torque_off']
print('OUTLET_FINAL_TRACKING_FAILURE_HELD_AND_REPORTED; real serial opens=0')
