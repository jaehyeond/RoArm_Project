"""Device-free SDK integration regression for observed closed-target relaxation."""
import json,math,tempfile,time
from pathlib import Path
from unittest.mock import patch
from hw_recorded_arm import RecordedArm,validate_command

class ContactSerial:
 def __init__(self,**kw):self.q=[0,1,91,0,0,2.5];self.in_waiting=1
 def open(self):pass
 def close(self):pass
 def flush(self):pass
 def write(self,b):
  c=json.loads(b)
  if c['T']==121:self.q[5]=29.2 if c['angle']==150 else max(4.57,180-c['angle'])
  if c['T']==122:self.q=[c[k] for k in ['b','s','e','t','r']]+[max(4.57,180-c['h'])]
  return len(b)
 def read(self,n):
  time.sleep(.01);d={'T':1051,**{k:math.radians(v) for k,v in zip(['b','s','e','t','r'],self.q[:5])},'g':math.radians(180-self.q[5]),'tS':10}
  return (json.dumps(d)+'\n').encode()

with tempfile.TemporaryDirectory(prefix='s1-target-regression-',dir='/tmp') as folder:
 with patch('hw_recorded_arm.AtomicInactiveSerial',ContactSerial):
  arm=RecordedArm(Path(folder))
  try:
   arm.joint_angle_ctrl(joint=6,angle=30,speed=200,acc=50);time.sleep(.05)
   q=arm.joints_angle_get();assert abs(q[5]-29.2)<1e-8
   arm.joints_angle_ctrl(angles=q,speed=200,acc=50)
   arm.joint_angle_ctrl(joint=6,angle=0,speed=200,acc=50);time.sleep(.05)
   q=arm.joints_angle_get();assert abs(q[5]-4.57)<1e-8
   arm.joints_angle_ctrl(angles=q,speed=200,acc=50);time.sleep(.05)
   assert arm.commanded_door_deg==0 and abs(arm.joints_angle_get()[5]-4.57)<1e-8
  finally:arm.close()
  rows=[json.loads(l) for l in (Path(folder)/'raw.jsonl').read_text().splitlines()]
  transmitted=[e['command'] for e in rows if e['ev']=='tx' and e['command']['T']==122]
  assert [c['h'] for c in transmitted]==[150,180]
  assert sum(e['ev']=='preserve_door_target' for e in rows)==2
  assert rows[-1]['ev']=='closed' and sum(e['ev']=='rx_json' for e in rows)>100
for c in [{'T':106},{'T':109},{'T':107,'tor':1000},{'T':121,'joint':6,'angle':140,'spd':17.578125,'acc':50*180/25400}]:
 try:validate_command(c)
 except AssertionError:pass
 else:raise AssertionError(c)
print('RECORDED_ARM_EXPLICIT_DOOR_TARGET_PRESERVED_OK; real serial opens=0')
