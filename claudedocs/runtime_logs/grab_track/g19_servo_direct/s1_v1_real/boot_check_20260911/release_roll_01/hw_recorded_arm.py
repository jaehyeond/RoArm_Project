"""Existing SDK command encoding with a single continuous raw-feedback reader.

Only the approved S1 commands are exposed. No automatic HOME or torque-off.
"""
import json, math, threading, time
from types import SimpleNamespace
from roarm_sdk.generate import CommandGenerator
from hw_serial_atomic import AtomicInactiveSerial, is_boot_text

def validate_command(c):
    if c=={'T':105}:return
    if set(c)=={'T','tor'} and c['T']==107 and c['tor'] in (200,790,900):return
    assert c.get('T') in (121,122),c
    assert abs(c['spd']-200*180/2048)<1e-10 and abs(c['acc']-50*180/25400)<1e-10,c
    if c['T']==121:
        assert set(c)=={'T','joint','angle','spd','acc'} and c['joint']==6 and 150<=c['angle']<=180,c
    else:
        assert set(c)=={'T','b','s','e','t','r','h','spd','acc'},c
        q=[c[k] for k in ('b','s','e','t','r')]+[180-c['h']]
        for x,(lo,hi) in zip(q,[(-90,90),(-110,110),(-70,190),(-90,90),(-5,5),(0,30)]):
            assert math.isfinite(x) and lo<=x<=hi,c

class RecordedArm(CommandGenerator):
    def __init__(self,out):
        super().__init__('roarm_m3',False)
        self.out=out;self.lock=threading.RLock();self.log_lock=threading.Lock()
        self._serial_port=self;self.base_controller=SimpleNamespace(data_buffer={})
        self.phase='preflight';self.latest=None;self.latest_at=0.;self.fatal=None;self.stop=threading.Event();self.high_since=None
        self.commanded_door_deg=None
        self.log_file=(out/'raw.jsonl').open('x',buffering=1)
        self.port=AtomicInactiveSerial(port=None,baudrate=115200,timeout=.02,write_timeout=1,exclusive=True)
        self.port.dtr=False;self.port.rts=False
        self.port.port='/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_ee7a06468e98ef1194edca63a8793231-if00-port0'
        self.port.open();self.record('open',port=self.port.port)
        self.thread=threading.Thread(target=self.reader,daemon=True);self.thread.start()
        try:
            time.sleep(1.5);self.joints_angle_get()
        except BaseException:
            self.close();raise
    def record(self,ev,**kw):
        with self.log_lock:
            self.log_file.write(json.dumps({'ev':ev,'t_ns':time.time_ns(),'mono_ns':time.monotonic_ns(),'phase':self.phase,**kw},allow_nan=False)+'\n')
    def send(self,c):
        validate_command(c)
        with self.lock:
            if self.fatal:raise RuntimeError(self.fatal)
            if c['T']!=105 and time.monotonic()-self.latest_at>1:raise RuntimeError('stale feedback before actuator command')
            # Observed in torque900_02: Manual.goto_q copies measured door angle
            # into T122, relaxing a closed target0 to4.57 during the first lift.
            # Preserve explicit T121 targets across arm motion; never infer a
            # target until the first explicit door command has actually been sent.
            if c['T']==122 and self.commanded_door_deg is not None:
                original=c
                c={**c,'h':180-self.commanded_door_deg};validate_command(c)
                if c!=original:self.record('preserve_door_target',requested_command=original,transmitted_command=c)
            packet=(json.dumps(c)+'\n').encode();n=self.port.write(packet);self.port.flush()
            if n!=len(packet):raise RuntimeError('short write')
            self.record('tx',command=c)
            if c['T']==121:self.commanded_door_deg=180-c['angle']
            if c['T']==107:self.high_since=time.monotonic() if c['tor']==900 else None
    def write(self,packet):self.send(json.loads(packet));return len(packet)
    def flush(self):pass
    def reader(self):
        pending=b'';next_query=0
        try:
            while not self.stop.is_set():
                now=time.monotonic()
                if (self.out/'STOP').exists():raise RuntimeError('operator STOP file')
                if self.high_since is not None and now-self.high_since>15:
                    self.send({'T':107,'tor':790});raise RuntimeError('900 torque 15s cap reached; reduced to790, trial invalid')
                if now>=next_query:self.send({'T':105});next_query=now+.2
                pending+=self.port.read(max(1,min(self.port.in_waiting,65536)))
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    try:d=json.loads(line)
                    except (ValueError,UnicodeDecodeError):
                        text=line.decode(errors='replace');self.record('rx_text',text=text)
                        if is_boot_text(text):raise RuntimeError('unexpected board boot text')
                        continue
                    self.record('rx_json',data=d)
                    if isinstance(d,dict) and d.get('T')==1051:
                        q=[math.degrees(d[k]) for k in ('b','s','e','t','r')]+[180-math.degrees(d['g'])]
                        if not all(math.isfinite(x) for x in q):raise RuntimeError('nonfinite feedback')
                        self.latest=(q,d);self.latest_at=time.monotonic()
        except BaseException as e:
            self.fatal=repr(e);self.record('reader_error',error=self.fatal)
    def joints_angle_get(self):
        if self.fatal:raise RuntimeError(self.fatal)
        if self.latest is None or time.monotonic()-self.latest_at>1:raise RuntimeError('raw feedback missing/stale')
        return list(self.latest[0])
    def feedback_get(self):
        self.joints_angle_get();self.base_controller.data_buffer=dict(self.latest[1]);return self.base_controller.data_buffer
    def _mesg(self,genre,*args):
        packet=super()._mesg(genre,*args);self.send(json.loads(packet));return packet
    def close(self):
        self.stop.set();self.thread.join(timeout=2)
        self.port.close();self.record('closed');self.log_file.close()
