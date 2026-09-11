"""Device-free checks of the actual failure and candidate control-line fix."""
import hashlib
import json
from pathlib import Path
import struct
import termios
from unittest.mock import patch
from hw_serial_atomic import AtomicInactiveSerial, is_boot_text

b=Path(__file__).resolve().parent
raw=b/'pid_hold_01'/'raw.jsonl'; rows=[json.loads(l) for l in raw.read_text().splitlines()]
result=json.loads((raw.parent/'result.json').read_text())
assert hashlib.sha256(raw.read_bytes()).hexdigest()==result['raw_sha256']
tx=[r['command'] for r in rows if r['ev']=='tx']
assert tx==[{'T':105}]*5
assert not result['completed'] and not result['restore_p16_command_sent']
assert result['n_feedback']==0 and rows[-1]['ev']=='closed'
assert any(is_boot_text(r.get('text','')) for r in rows)
oldraw=b/'raw_feedback_20260911_144111.jsonl'
oldrows=[json.loads(l) for l in oldraw.read_text().splitlines()]
assert any(is_boot_text(r.get('text','')) for r in oldrows)

# Exercise the installed pySerial base and both overridden hooks without opening.
s=AtomicInactiveSerial(port=None); s.dtr=False; s.rts=False; s.fd=12345
with patch('hw_serial_atomic.fcntl.ioctl') as call:
    s._update_dtr_state(); s._update_rts_state()
    assert call.call_count==2
    for args in call.call_args_list:
        assert args.args==(12345,termios.TIOCMBIC,struct.pack('I',termios.TIOCM_DTR|termios.TIOCM_RTS))
    s._dtr_state=True
    try: s._update_dtr_state()
    except RuntimeError: pass
    else: raise AssertionError('active-line mutation not rejected')
    assert call.call_count==2
s.fd=None
assert is_boot_text('prefix\ufffdets Jul 29 2019 12:21:46')
assert is_boot_text('rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)')
assert not is_boot_text('{"T":1051,"s":0.02}')
report={'pass':True,'incident_tx_T105':len(tx),'incident_actuator_commands':0,
        'prior_raw_also_has_boot_text':True,'sequential_line_clear_removed_in_python':True,
        'hardware_fix_verified':False,'scope':'file evidence plus mocked ioctl regression, no serial open',
        'raw_sha256':result['raw_sha256'],
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [b/'hw_serial_atomic.py',b/'hw_pid_hold.py']}}
(b/'serial_incident_verification.json').write_text(json.dumps(report,indent=2))
print('SERIAL_INCIDENT_AND_FIX_OFFLINE_OK')
