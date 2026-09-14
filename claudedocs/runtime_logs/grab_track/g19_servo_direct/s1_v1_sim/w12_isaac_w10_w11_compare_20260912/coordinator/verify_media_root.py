"""Read-only W12 media check. Decode video frame counts, compare PNG scene pixels.

Annotated derivatives replace only burnt-in text regions and add header/footer.
This checker confirms unmodified scene pixels, correct pairing and restored
historical trial1 files. It does not infer label semantics or human inspection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path('/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay')
OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--report')
args = parser.parse_args()

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

videos = ['w10/replay_w10.mp4', 'w11/replay_w11.mp4', 'paired/paired_w10_w11_sourcetime.mp4', 'annotated/replay_w10_annotated.mp4', 'annotated/replay_w11_annotated.mp4', 'annotated/paired_w10_w11_sourcetime_annotated.mp4']
result = {'videos': {}, 'scene_pixels_exact': {}, 'paired_pixels_exact': 0, 'trial1_restoration': {}}
for name in videos:
    p = ROOT / name
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height,avg_frame_rate:format=duration', '-of', 'json', str(p)], text=True)
    info = json.loads(raw)
    stream = info['streams'][0]
    assert int(stream['nb_read_frames']) == 64 and stream['avg_frame_rate'] == '10/1', name
    assert abs(float(info['format']['duration']) - 6.4) < 0.001, name
    result['videos'][name] = {'ffprobe': info, 'sha256': sha(p)}

for tag in ['w10', 'w11']:
    for i in range(64):
        with Image.open(ROOT / tag / 'frames' / f'f_{i:05d}.png') as image:
            source = np.asarray(image.convert('RGB'))
        with Image.open(ROOT / 'annotated' / f'frames_{tag}' / f'a_{i:05d}.png') as image:
            annotated = np.asarray(image.convert('RGB'))
        height, width = source.shape[:2]
        assert annotated.shape == (height + 158, width, 3)
        # Rectangle endpoint74 is inclusive; only image rows75..height-27 remain.
        assert np.array_equal(source[75:height-26], annotated[132+75:132+height-26]), (tag, i)
    result['scene_pixels_exact'][tag] = 64

with (ROOT / 'paired/pairing.csv').open() as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 64
for row in rows:
    with Image.open(ROOT / 'annotated/frames_paired' / f"p_{int(row['pair']):05d}.png") as image:
        pair = np.asarray(image.convert('RGB'))
    offset = 34
    for tag in ['w10', 'w11']:
        with Image.open(ROOT / 'annotated' / f'frames_{tag}' / f"a_{int(row[tag+'_frame']):05d}.png") as image:
            single = np.asarray(image.convert('RGB'))
        assert np.array_equal(single, pair[offset:offset+single.shape[0]]), (row['pair'], tag)
        offset += single.shape[0]
    assert offset == pair.shape[0]
    result['paired_pixels_exact'] += 1

for suffix in ['.rrd', '.rbl', '_inspection.png', '_rerun_validation.json', '_coverage.json']:
    name = 'w12_replay_mapping' + suffix
    restored, archived = ROOT / 'rerun' / name, ROOT / 'rerun/trial1_defective' / name
    assert restored.is_file() and archived.is_file(), name
    digest = sha(restored)
    assert digest == sha(archived), name
    result['trial1_restoration'][name] = digest

result['all_pass'] = True
if args.report:
    assert Path(args.report).name == args.report
    with (OUT / args.report).open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
print('W12_MEDIA_ROOT_REVERIFIED', len(result['videos']), 'videos;', sum(result['scene_pixels_exact'].values()), 'scene frames;', result['paired_pixels_exact'], 'pairs;', len(result['trial1_restoration']), 'restored files')
