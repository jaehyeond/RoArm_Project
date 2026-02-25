"""
Analyzes starting and ending joint positions across all collected_data episodes.
Research only - no file modifications.
"""
import json
import os
import numpy as np

base = '/home/cgxr/Documents/Robotics/RoArm_Project/collected_data'
episodes = sorted([d for d in os.listdir(base) if d.startswith('episode_')])

first_frames = []
last_frames = []
episode_names = []

for ep in episodes:
    meta_path = os.path.join(base, ep, 'metadata.json')
    with open(meta_path) as f:
        d = json.load(f)
    frames = d['frames']
    first_frames.append(frames[0]['angles'])
    last_frames.append(frames[-1]['angles'])
    episode_names.append(ep)

first_arr = np.array(first_frames)
last_arr = np.array(last_frames)

joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist_P', 'Wrist_R', 'Gripper']

print(f"Total episodes: {len(episodes)}")
print()

print('FIRST FRAME STATISTICS (n={} episodes):'.format(len(episodes)))
print('  {:<12} | {:>6} | {:>5} | {:>7} | {:>7}'.format('Joint', 'Mean', 'Std', 'Min', 'Max'))
print('  ' + '-'*52)
for i, name in enumerate(joint_names):
    vals = first_arr[:, i]
    print('  {:<12} | {:6.2f} | {:5.2f} | {:7.2f} | {:7.2f}'.format(
        name, vals.mean(), vals.std(), vals.min(), vals.max()))

print()
print('LAST FRAME STATISTICS (n={} episodes):'.format(len(episodes)))
print('  {:<12} | {:>6} | {:>5} | {:>7} | {:>7}'.format('Joint', 'Mean', 'Std', 'Min', 'Max'))
print('  ' + '-'*52)
for i, name in enumerate(joint_names):
    vals = last_arr[:, i]
    print('  {:<12} | {:6.2f} | {:5.2f} | {:7.2f} | {:7.2f}'.format(
        name, vals.mean(), vals.std(), vals.min(), vals.max()))

print()
# Reference positions
init_target = np.array([0, 0, 90, 0, 0, 0], dtype=float)
dataset_mean_v3 = np.array([-0.47, 30.18, 58.88, 40.72, -2.33, 26.48], dtype=float)

print('REFERENCE POSITIONS:')
print('  init_target  = [0, 0, 90, 0, 0, 0] (arm straight up)')
print('  dataset_mean = [-0.47, 30.18, 58.88, 40.72, -2.33, 26.48] (v3 dataset mean)')
print()

diffs_init = np.abs(first_arr - init_target)
print('DISTANCE from init_target [0, 0, 90, 0, 0, 0]:')
print('  Mean abs diff per joint: {}'.format(diffs_init.mean(axis=0).round(2)))
print('  Max abs diff per joint:  {}'.format(diffs_init.max(axis=0).round(2)))
print('  All joints within 5 deg: {}/{}'.format((diffs_init < 5).all(axis=1).sum(), len(episodes)))
print('  All joints within 10 deg: {}/{}'.format((diffs_init < 10).all(axis=1).sum(), len(episodes)))
print()

# Show outliers
within5 = (diffs_init < 5).all(axis=1)
outliers = [episode_names[i] for i in range(len(episodes)) if not within5[i]]
if outliers:
    print('Episodes NOT within 5 deg of init:')
    for ep in outliers:
        idx = episode_names.index(ep)
        print('  {}: {}'.format(ep, [round(x, 1) for x in first_frames[idx]]))
else:
    print('All episodes within 5 deg of init_target.')
print()

# Gripper analysis
gripper_vals = first_arr[:, 5]
print('GRIPPER at first frame:')
print('  Mean: {:.2f} deg'.format(gripper_vals.mean()))
print('  Std:  {:.2f} deg'.format(gripper_vals.std()))
print('  Range: [{:.2f}, {:.2f}] deg'.format(gripper_vals.min(), gripper_vals.max()))
print('  Eps with gripper < 5 deg: {}/{}'.format((gripper_vals < 5).sum(), len(episodes)))
print('  Eps with gripper 5-15 deg: {}/{}'.format(((gripper_vals >= 5) & (gripper_vals < 15)).sum(), len(episodes)))
print('  Eps with gripper > 15 deg: {}/{}'.format((gripper_vals >= 15).sum(), len(episodes)))

print()
print('ELBOW at first frame:')
elbow_vals = first_arr[:, 2]
print('  Mean: {:.2f} deg'.format(elbow_vals.mean()))
print('  Std:  {:.2f} deg'.format(elbow_vals.std()))
print('  Range: [{:.2f}, {:.2f}] deg'.format(elbow_vals.min(), elbow_vals.max()))
print('  All within 5 deg of 90: {}/{}'.format(((elbow_vals > 85) & (elbow_vals < 95)).sum(), len(episodes)))

print()
# Shoulder at first frame
shoulder_vals = first_arr[:, 1]
print('SHOULDER at first frame:')
print('  Mean: {:.2f} deg'.format(shoulder_vals.mean()))
print('  All within 10 deg of 0: {}/{}'.format((np.abs(shoulder_vals) < 10).sum(), len(episodes)))

print()
print('KEY FINDING: Are episodes starting near init_target [0,0,90,0,0,0]?')
print('  Elbow close to 90 (all eps): YES' if (np.abs(elbow_vals - 90) < 5).all() else '  Elbow close to 90: NO')
print('  Shoulder close to 0 (all eps): YES' if (np.abs(shoulder_vals) < 10).all() else '  Shoulder close to 0: partially')
print('  Gripper ~0-2 deg (all eps): YES' if (gripper_vals < 5).all() else '  Gripper ~0-2 deg: partially')
