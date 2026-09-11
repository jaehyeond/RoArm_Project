"""One residual-pellet release probe, using the existing place path.

No new scoop, no shaking, no expanded joint limits. At the existing release
pose, hold door30, measure gross mass, command roll -5 once, measure again,
restore the issued roll0 target, close and return through Manual.place.
stdin accepts a JSON measurement {"gross_g": number}; no input means no next
motion. The existing raw reader remains active during the measurement holds.
"""
import argparse
import functools
import hashlib
import json
import math
import select
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'hw_s1_manual.py').is_file())
sys.path.insert(0, str(ROOT))
import hw_s1_manual as M
from hw_recorded_arm import RecordedArm, validate_command


def run(out, dry_run):
    out.mkdir(parents=True, exist_ok=False)
    M.S.OUT_DIR = str(out)
    M.POS_FILE = str(out / 'unused_positions.json')
    M.MASS_FILE = str(out / 'unused_mass.jsonl')
    M.solve_fast = functools.lru_cache(maxsize=128)(M.solve_fast)
    args = SimpleNamespace(sim=True, base_cm=38, pellet_cm=26,
                           boxtop_cm=38.5, travel_cm=45, box_x_cm=35, port='/dev/ttyUSB0')
    result = {'completed': False, 'dry_run': dry_run, 'experiment': 'residual_roll_minus5',
              'new_scoop': False, 'measurements': {}, 'roll_target_deg': -5.0}
    arm = None

    def stage(name):
        print('STAGE ' + name, flush=True)
        if arm:
            arm.phase = name
            arm.record('stage')

    def measurement(name):
        stage(name)
        if dry_run:
            return {'gross_g': None, 'synthetic': True}
        print('WAIT_MEASUREMENT ' + name, flush=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            arm.joints_angle_get()  # Existing fresh-feedback / boot / STOP contract.
            if not select.select([sys.stdin], [], [], .2)[0]:
                continue
            line = sys.stdin.readline()
            if not line:
                raise RuntimeError('stdin closed during measurement; hold current pose')
            try:
                row = json.loads(line)
                value = row['gross_g']
                if isinstance(value, bool) or not isinstance(value, (float, int)):
                    raise ValueError('gross_g must be numeric')
                if not math.isfinite(value) or not 0 <= value <= 500:
                    raise ValueError('gross_g outside scale range')
            except (ValueError, KeyError, TypeError) as exc:
                print('INVALID_MEASUREMENT ' + str(exc), flush=True)
                continue
            row = {**row, 'host_time_ns': time.time_ns(), 'source': 'operator_text',
                   'not_sensor_synchronized': True}
            arm.record('operator_measurement', measurement=row)
            result['measurements'][name] = row
            with (out / (name + '.json')).open('x') as file:
                json.dump(row, file, indent=2)
            return row
        raise RuntimeError('measurement timeout; hold current pose without return motion')

    class ReleaseManual(M.Manual):
        last_issued_q5 = None
        probe_done = False

        def goto_q(self, name, q5, tol=5.0):
            ok = super().goto_q(name, q5, tol)
            if ok:
                self.last_issued_q5 = list(q5)
            return ok

        def door(self, deg, tor=None):
            value = super().door(deg, tor)
            if deg == 30 and not self.probe_done:
                self.probe_done = True
                reference = list(self.last_issued_q5)
                if abs(reference[4]) > 1e-6:
                    raise RuntimeError('release reference must have issued roll0')
                result['reference_q5'] = reference
                time.sleep(1.5)
                baseline = measurement('release_baseline')
                candidate = reference.copy()
                candidate[4] = -5.0
                stage('roll_minus5')
                if not self.goto_q('single_roll_minus5', candidate):
                    raise RuntimeError('roll movement rejected; hold')
                time.sleep(2)
                after = measurement('release_after_roll')
                if not dry_run:
                    result['additional_delivered_mass_g'] = after['gross_g'] - baseline['gross_g']
                stage('restore_roll0')
                if not self.goto_q('restore_issued_roll0', reference):
                    raise RuntimeError('roll restore rejected; hold')
                stage('return_after_probe')
            return value

    manual = ReleaseManual(args)
    plan = {'experiment': result['experiment'], 'roll_from_deg': 0, 'roll_to_deg': -5,
            'door_target_deg_during_test': 30, 'gripper_torque': 200,
            'reference_policy': 'preserve issued first four joint targets, not measured deflected angles',
            'unchanged_joint_limits': True, 'new_scoop': False,
            'source_residue_run': 'torque790_01', 'source_residue_count': None,
            'baseline_settle_extra_s': 1.5, 'after_rotation_wait_s': 2,
            'cup_vessel_height_cm': 9, 'cup_inner_diameter_cm': 7,
            'cup_position': 'operator adjusted; not a fixed measured obstacle',
            'limitations': ['Small roll is not a five-degree outlet-directed tilt.',
                           'Delayed release and cup handling are not independently controlled.',
                           'Additional mass is a paired exploratory observation, not a causal estimate.']}
    source_paths = [Path(__file__), Path(__file__).with_name('hw_recorded_arm.py'),
                    Path(__file__).with_name('hw_serial_atomic.py'), ROOT / 'hw_s1_manual.py',
                    ROOT / 'hw_s1_scoop_probe.py', ROOT / 'safety_p0_guards.py']
    plan['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    with (out / 'plan.json').open('x') as file:
        json.dump(plan, file, indent=2)
    for path in source_paths:
        (out / path.name).write_bytes(path.read_bytes())

    # Solve every existing release height before opening serial.
    for height in (45, 40, 35, 30, 26):
        solution = M.solve_fast(.35, manual.floor + height / 100)
        if solution is None or solution[0] > .01:
            raise RuntimeError('unreachable existing release height ' + str(height))
    target = list(M.solve_fast(.35, manual.pellet)[1])
    target[0] = 90.0
    path_report = []
    ref_lip = manual.lip_world(target)
    for i in range(51):
        candidate = target.copy()
        candidate[4] = -5 * i / 50
        command = dict(T=122, **dict(zip(('b', 's', 'e', 't', 'r'), candidate)),
                       h=150, spd=200*180/2048, acc=50*180/25400)
        validate_command(command)
        lip = manual.lip_world(candidate)
        if lip[2] < manual.lip_min:
            raise RuntimeError('roll path lip below existing limit')
        path_report.append({'q5': candidate, 'lip_m_fw': lip.tolist(),
                            'lip_shift_mm': float(M.np.linalg.norm(lip-ref_lip)*1000)})
    with (out / 'roll_path.json').open('x') as file:
        json.dump(path_report, file, indent=2)
    try:
        if not dry_run:
            arm = RecordedArm(out)
            manual.sim = False
            manual.arm = arm
            q = manual.read()
            if max(abs(x-y) for x, y in zip(q[:5], M.P1)) > 6 or not 0 <= q[5] <= 30:
                raise RuntimeError('current pose is not within existing P1 gate')
            manual.torque(200)
        stage('place_residue_only')
        if not manual.place(base_deg=90):
            raise RuntimeError('existing place path rejected; hold')
        stage('complete_p1')
        result['completed'] = True
        result['final_q_deg'] = manual.read()
    except BaseException as exc:
        result['error'] = repr(exc)
        print('STOP_HOLD ' + repr(exc), flush=True)
    finally:
        if arm:
            arm.close()
        if (out / 'raw.jsonl').exists():
            result['raw_sha256'] = hashlib.sha256((out / 'raw.jsonl').read_bytes()).hexdigest()
        with (out / 'result.json').open('x') as file:
            json.dump(result, file, indent=2)
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result['completed'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(args.out, args.dry_run))
