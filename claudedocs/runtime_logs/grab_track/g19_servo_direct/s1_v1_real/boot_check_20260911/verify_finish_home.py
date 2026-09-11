"""Fake-device check: tilt precedes closed HOME, or tracking failure holds."""
import contextlib,io,json,tempfile
from pathlib import Path
from unittest.mock import patch
import hw_scoop_tilt_cycle as H
b=Path(__file__).resolve().parent;p=b/'scoop_tilt_cycle_01/plan_06/plan.json';plan=json.loads(p.read_text())
class Fake:
 fail=False;items=[]
 def __init__(self,out,command_validator):self.q=plan['preflight_q_deg'].copy();self.validate=command_validator;self.commanded_door_deg=None;self.tx=[];self.closed=False;Fake.items.append(self)
 def joints_angle_get(self):return self.q.copy()
 def record(self,*a,**kw):pass
 def send(self,c):
  self.validate(c);self.tx.append((self.phase,c))
  if c['T']==122:
   self.q=[c[k] for k in ['b','s','e','t','r']]+[180-c['h']]
   if self.fail and self.phase=='finish_tilt_8':self.q[1]+=5.01
  if c['T']==121:self.commanded_door_deg=180-c['angle'];self.q[5]=self.commanded_door_deg
 def close(self):self.closed=True
results=[]
with tempfile.TemporaryDirectory(dir='/tmp',prefix='s1-finish-home-') as tmp:
 for fail in [False,True]:
  Fake.fail=fail
  with patch.object(H,'RecordedArm',Fake),patch.object(H.time,'sleep',lambda _:None),contextlib.redirect_stdout(io.StringIO()):r=H.run(p,Path(tmp)/str(fail))
  a=Fake.items[-1];names=[x[0] for x in a.tx];assert a.closed
  assert r['completed']==(not fail) and r['home_reached']==(not fail)
  assert 'plunge' not in names
  if not fail:
   assert names.index('finish_tilt_8')<names.index('outlet_after_tilt')<names.index('close_empty')<names.index('home')
   assert r['final_q_deg']==[0,0,90,0,0,0] and r['final_phase']=='home_after_finish'
  else:assert 'home' not in names and 'close_empty' not in names and 'settled pose outside gate' in r['error']
  results.append(dict(fail_injected=fail,pass_=True,completed=r['completed'],home_reached=r['home_reached']))
(p.parent/'mock_execution_audit.json').write_text(json.dumps(dict(real_serial_opens=0,tests=results),indent=2));print('TILT_THEN_HOME_AND_FAILURE_HOLD_PASS')
