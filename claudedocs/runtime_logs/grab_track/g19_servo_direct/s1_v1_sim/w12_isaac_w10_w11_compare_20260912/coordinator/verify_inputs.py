"""Read-only original input/HEAD checker; only new coordinator reports are written."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

MAIN = Path('/home/cgxr/Documents/Robotics/RoArm_Project')
OUT = Path(__file__).resolve().parent
SIM = OUT.parent.parent
W11 = SIM / 'w11_dt_sensitivity_20260911'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def save(p, data):
    with Path(p).open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')

def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=MAIN, text=True).strip()

def snapshot():
    pre = read(W11 / 'preflight.json')
    protected = pre['protected_files']
    paths = set(protected)
    for p, v in protected.items():
        assert sha(p) == v['sha256'], p
    paths.update(str(p) for p in W11.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(str(p) for p in (MAIN / 'local_assets/roarm_m3/usd_s1_v1').rglob('*') if p.is_file())
    paths.update(str(MAIN / p) for p in ['sim_isaac_render_deme_scoop.py', 'hw_s1_scoop_probe.py', 'hw_s1_manual.py'])
    for n in ['REPORT_w9.md', 'gates_w9.json']:
        paths.add(str(SIM / 'w9_isaac_render_deme' / n))
    data = {'head': head(), 'files': {p: {'sha256': sha(p), 'bytes': Path(p).stat().st_size} for p in sorted(paths)}}
    save(OUT / 'input_baseline.json', data)
    print('W12_BASELINE_PASS', len(data['files']))

def check(report):
    base = read(OUT / 'input_baseline.json')
    failed = [p for p, v in base['files'].items() if not Path(p).is_file() or sha(p) != v['sha256']]
    same_head = head() == base['head']
    data = {'files_checked': len(base['files']), 'changed_or_missing': failed, 'head_unchanged': same_head, 'all_pass': not failed and same_head}
    if report:
        save(OUT / report, data)
    assert data['all_pass'], data
    print('W12_INPUTS_UNCHANGED', len(base['files']))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['snapshot', 'check'])
    p.add_argument('--report')
    a = p.parse_args()
    snapshot() if a.mode == 'snapshot' else check(a.report)
