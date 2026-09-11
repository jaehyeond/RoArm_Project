"""One authorized, bounded shoulder P perturbation; never commands a joint target."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from hw_serial_atomic import AtomicInactiveSerial, is_boot_text

def angles(d):
    return [math.degrees(float(d[k])) for k in ('b','s','e','t','r')] + [180-math.degrees(float(d['g']))]

def run(out):
    out.mkdir(parents=True, exist_ok=False)
    raw = out/'raw.jsonl'
    phases = [('baseline',4,None),('p8_a',5,8),('p48_a',5,48),('p8_b',5,8),('p48_b',5,48)]
    plan = {'phases':phases,'restore':{'joint':2,'p':16,'i':0,'duration_s':4},
            'joint_drift_limit_deg':4,'feedback_timeout_s':1,'no_joint_target_commands':True,
            'initial_pid_registers':'unreadable; restore is documented-default command, not register verification',
            'transport':'AtomicInactiveSerial; candidate fix, no kernel/board no-reset guarantee',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'transport_sha256':hashlib.sha256(Path(__file__).with_name('hw_serial_atomic.py').read_bytes()).hexdigest()}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    samples=[]; commands=[]; phase='preflight'; reference=None; last_rx=None; started=time.monotonic()
    port=AtomicInactiveSerial(port=None,baudrate=115200,timeout=0.05,write_timeout=1,exclusive=True)
    port.dtr=False; port.rts=False
    port.port='/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_ee7a06468e98ef1194edca63a8793231-if00-port0'
    pending=b''; changed=False; completed=False; error=None; restored=False
    with raw.open('x',buffering=1) as f:
        def log(ev, **kw):
            row={'ev':ev,'t_ns':time.time_ns(),'mono_ns':time.monotonic_ns(),'phase':phase,**kw}
            f.write(json.dumps(row,allow_nan=False)+'\n'); return row
        def tx(cmd):
            assert cmd == {'T':105} or (set(cmd)=={'T','joint','p','i'} and cmd['T']==108 and cmd['joint']==2 and cmd['p'] in (8,16,48) and cmd['i']==0)
            packet=(json.dumps(cmd)+'\n').encode(); n=port.write(packet); port.flush()
            if n != len(packet): raise RuntimeError('short serial write')
            commands.append(cmd); log('tx',command=cmd)
        def collect(duration,guard=True):
            nonlocal pending,last_rx,reference
            begin=time.monotonic(); next_query=begin
            while time.monotonic()-begin < duration:
                now=time.monotonic()
                if now>=next_query: tx({'T':105}); next_query=now+0.2
                pending+=port.read(max(1,min(port.in_waiting,65536)))
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    try: d=json.loads(line)
                    except (ValueError,UnicodeDecodeError):
                        text=line.decode(errors='replace'); log('rx_text',text=text)
                        if is_boot_text(text): raise RuntimeError('board boot text detected; no further test commands')
                        continue
                    row=log('rx_json',data=d)
                    if not isinstance(d,dict) or d.get('T')!=1051: continue
                    q=angles(d)
                    if len(q)!=6 or not all(math.isfinite(v) for v in q): raise RuntimeError('invalid joint feedback')
                    last_rx=time.monotonic(); samples.append({**row,'q_deg':q})
                    if guard and reference is not None and max(abs(a-b) for a,b in zip(q,reference))>4:
                        raise RuntimeError('joint drift exceeded 4 degrees')
                    if guard and not (-0.1<=q[5]<=30.1): raise RuntimeError('S1 gripper feedback outside range')
                if time.monotonic()-(last_rx if last_rx is not None else begin)>1: raise RuntimeError('feedback timeout')
        try:
            port.open(); log('open',port=port.port)
            collect(1.5)
            if len(samples)<20: raise RuntimeError('insufficient preflight feedback')
            reference=[statistics.median(s['q_deg'][j] for s in samples[-20:]) for j in range(6)]
            if max(abs(a-b) for a,b in zip(reference[:5],[0,0,90,0,0]))>5: raise RuntimeError('expected initial pose not observed')
            log('reference',q_deg=reference)
            for phase,duration,p in phases:
                if p is not None:
                    changed=True; tx({'T':108,'joint':2,'p':p,'i':0})
                log('phase_start',p_command=p,duration_s=duration)
                print(json.dumps({'phase':phase,'duration_s':duration}),flush=True)
                collect(duration)
            completed=True
        except BaseException as exc:
            error=repr(exc); log('error',error=error); print(error,flush=True)
        finally:
            phase='restore_p16'
            if changed and port.is_open:
                try:
                    tx({'T':108,'joint':2,'p':16,'i':0}); restored=True
                    log('phase_start',p_command=16,duration_s=4); collect(4,guard=False)
                except BaseException as exc:
                    log('restore_error',error=repr(exc)); error=error or repr(exc)
            if port.is_open: port.close()
            log('closed')
    stats={}
    for label in [p[0] for p in phases]+['restore_p16']:
        rows=[s for s in samples if s['phase']==label]
        if not rows: continue
        rows=[s for s in rows if s['mono_ns']>=rows[-1]['mono_ns']-2_000_000_000]
        vals=[s['q_deg'][1] for s in rows]
        stats[label]={'n_last2s':len(vals),'mean_shoulder_deg':statistics.mean(vals),'min_deg':min(vals),'max_deg':max(vals),'pstdev_deg':statistics.pstdev(vals)}
    result={'completed':completed,'error':error,'restore_p16_command_sent':restored,'restore_register_readback':False,
            'initial_reference_deg':reference,'n_feedback':len(samples),'tG_present_count':sum('tG' in s['data'] for s in samples),
            'max_abs_joint_drift_deg':max(max(abs(a-b) for a,b in zip(s['q_deg'],reference)) for s in samples) if reference else None,
            'duration_s':time.monotonic()-started,'phase_statistics':stats,'commands':commands,'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='commands'},indent=2),flush=True)
    return 0 if completed and restored and not error else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    raise SystemExit(run(a.out))
