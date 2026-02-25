"""
data_temporal_phase_analysis.py
Analyze temporal phase structure of grasp episodes in lerobot_dataset_v3.

Key questions:
1. At what frame number does each phase happen?
   Phase A: Start (gripper ~2 deg, arm at init)
   Phase B: Approach (arm moves toward sponge, gripper still closed)
   Phase C: Gripper opens (>40 deg)
   Phase D: Gripper closes to ~24 deg (sponge gripped)
   Phase E: Lift (arm rises with sponge)
   Phase F: Return to init

2. When does Phase C (gripper open) typically START?
   When does Phase D (gripper close to ~24) START?

3. If a 50-step action chunk starts from frame 0, when would gripper open/close happen?

4. Is 50 steps enough for the entire grasp cycle?

5. Frame-by-frame gripper trajectory for 5 specific episodes.
"""

import pandas as pd
import numpy as np

PARQUET_PATH = '/home/cgxr/Documents/Robotics/RoArm_Project/lerobot_dataset_v3/data/chunk-000/file-000.parquet'
GRIPPER_OPEN_THRESHOLD = 40.0   # deg -- Phase C start
GRIPPER_GRIPPED_THRESHOLD = 30.0  # deg -- Phase D start (sponge gripped = below this after peak)


def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    actions = np.stack(df['action'].values)
    states = np.stack(df['observation.state'].values)
    df['a_base'] = actions[:, 0]
    df['a_shoulder'] = actions[:, 1]
    df['a_elbow'] = actions[:, 2]
    df['a_wrist_pitch'] = actions[:, 3]
    df['a_wrist_roll'] = actions[:, 4]
    df['a_gripper'] = actions[:, 5]
    df['s_base'] = states[:, 0]
    df['s_shoulder'] = states[:, 1]
    df['s_elbow'] = states[:, 2]
    df['s_wrist_pitch'] = states[:, 3]
    df['s_wrist_roll'] = states[:, 4]
    df['s_gripper'] = states[:, 5]
    return df


def detect_phases(ep_df):
    """
    Detect phase transitions for a single episode.
    Returns dict with frame indices for each phase.
    Uses ACTION gripper (what the model outputs / what we want robot to do).
    """
    gripper = ep_df['a_gripper'].values
    shoulder = ep_df['a_shoulder'].values
    n = len(gripper)

    result = {
        'n_frames': n,
        'duration_sec': n / 30.0,
        'gripper_start': gripper[0],
        'gripper_end': gripper[-1],
        'gripper_max': gripper.max(),
        'gripper_max_frame': int(gripper.argmax()),
        'shoulder_max': shoulder.max(),
        'shoulder_max_frame': int(shoulder.argmax()),
    }

    # Phase C: first frame where gripper exceeds GRIPPER_OPEN_THRESHOLD
    open_frames = np.where(gripper > GRIPPER_OPEN_THRESHOLD)[0]
    if len(open_frames) > 0:
        result['phase_C_start_frame'] = int(open_frames[0])
        result['phase_C_start_pct'] = open_frames[0] / n * 100
        result['phase_C_frames_count'] = len(open_frames)
    else:
        result['phase_C_start_frame'] = None
        result['phase_C_start_pct'] = None
        result['phase_C_frames_count'] = 0

    # Phase D: after gripper peak, first frame where gripper drops below GRIPPER_GRIPPED_THRESHOLD
    peak_frame = result['gripper_max_frame']
    post_peak = gripper[peak_frame:]
    close_frames_after_peak = np.where(post_peak < GRIPPER_GRIPPED_THRESHOLD)[0]
    if len(close_frames_after_peak) > 0:
        result['phase_D_start_frame'] = int(peak_frame + close_frames_after_peak[0])
        result['phase_D_start_pct'] = result['phase_D_start_frame'] / n * 100
    else:
        result['phase_D_start_frame'] = None
        result['phase_D_start_pct'] = None

    # Phase E: after grip close, shoulder starts descending (returns to home = low shoulder)
    # Proxy: frame after phase_D where shoulder starts decreasing
    if result['phase_D_start_frame'] is not None:
        d_frame = result['phase_D_start_frame']
        if d_frame < n - 1:
            post_grip_shoulder = shoulder[d_frame:]
            # Find where shoulder decreases by >2 deg from its local max after grip
            local_max = post_grip_shoulder.max()
            local_max_idx = post_grip_shoulder.argmax()
            descend_frames = np.where(post_grip_shoulder[local_max_idx:] < local_max - 5)[0]
            if len(descend_frames) > 0:
                result['phase_E_start_frame'] = int(d_frame + local_max_idx + descend_frames[0])
                result['phase_E_start_pct'] = result['phase_E_start_frame'] / n * 100
            else:
                result['phase_E_start_frame'] = None
                result['phase_E_start_pct'] = None
        else:
            result['phase_E_start_frame'] = None
            result['phase_E_start_pct'] = None
    else:
        result['phase_E_start_frame'] = None
        result['phase_E_start_pct'] = None

    # Count frames by phase region (using gripper state)
    # Phase A+B: gripper closed (<GRIPPER_OPEN_THRESHOLD), before phase C
    if result['phase_C_start_frame'] is not None:
        result['frames_AB'] = result['phase_C_start_frame']  # frames before gripper opens
    else:
        result['frames_AB'] = n

    # Phase C region: gripper > GRIPPER_OPEN_THRESHOLD
    result['frames_C_open'] = result['phase_C_frames_count']

    return result


def print_episode_gripper_trajectory(ep_df, ep_id, max_rows=50):
    """Print frame-by-frame gripper + shoulder trajectory."""
    gripper = ep_df['a_gripper'].values
    shoulder = ep_df['a_shoulder'].values
    elbow = ep_df['a_elbow'].values
    n = len(gripper)
    print(f"\n=== Episode {ep_id} | {n} frames ({n/30:.1f}s) ===")
    print(f"  Gripper range: [{gripper.min():.1f}, {gripper.max():.1f}] deg")
    print(f"  Shoulder range: [{shoulder.min():.1f}, {shoulder.max():.1f}] deg")
    print(f"  Elbow range: [{elbow.min():.1f}, {elbow.max():.1f}] deg")
    print()
    print(f"  {'Frame':>6} {'Time':>6} {'Gripper':>8} {'Shoulder':>9} {'Elbow':>7} {'Phase':>10}")
    print(f"  {'-'*55}")

    # Print every 3rd frame if long, else every frame (but cap at max_rows output)
    step = max(1, n // max_rows)
    for i in range(0, n, step):
        t = i / 30.0
        g = gripper[i]
        sh = shoulder[i]
        el = elbow[i]

        # Simple phase label
        if i == 0:
            phase = "A:START"
        elif g < 5 and i > 0 and gripper[:i].max() < GRIPPER_OPEN_THRESHOLD:
            phase = "B:APPROACH"
        elif g > GRIPPER_OPEN_THRESHOLD:
            phase = "C:OPEN"
        elif g > 20 and gripper[:i].max() > GRIPPER_OPEN_THRESHOLD:
            phase = "D:GRIPPED"
        elif g < 15 and i > 5:
            phase = "E:LIFT?"
        else:
            phase = ""

        bar = "#" * int(g / 3)
        print(f"  {i:>6} {t:>6.2f}s {g:>8.1f} {sh:>9.1f} {el:>7.1f}  {phase:<10}")

    # Also print last frame
    i = n - 1
    print(f"  {i:>6} {i/30:.2f}s {gripper[i]:>8.1f} {shoulder[i]:>9.1f} {elbow[i]:>7.1f}  F:END")


def main():
    print("Loading data...")
    df = load_data()

    n_episodes = df['episode_index'].nunique()
    print(f"Total episodes: {n_episodes}")
    print(f"Total frames: {len(df)}")
    print()

    # ===========================
    # SECTION 1: Episode length distribution
    # ===========================
    print("=" * 70)
    print("SECTION 1: EPISODE LENGTH DISTRIBUTION")
    print("=" * 70)
    ep_lengths = df.groupby('episode_index').size()
    print(f"Mean: {ep_lengths.mean():.1f} frames ({ep_lengths.mean()/30:.2f}s)")
    print(f"Std:  {ep_lengths.std():.1f} frames")
    print(f"Min:  {ep_lengths.min()} frames ({ep_lengths.min()/30:.2f}s)")
    print(f"Max:  {ep_lengths.max()} frames ({ep_lengths.max()/30:.2f}s)")
    print(f"Q10:  {ep_lengths.quantile(0.1):.0f} frames ({ep_lengths.quantile(0.1)/30:.2f}s)")
    print(f"Q25:  {ep_lengths.quantile(0.25):.0f} frames ({ep_lengths.quantile(0.25)/30:.2f}s)")
    print(f"Q50:  {ep_lengths.quantile(0.5):.0f} frames ({ep_lengths.quantile(0.5)/30:.2f}s)")
    print(f"Q75:  {ep_lengths.quantile(0.75):.0f} frames ({ep_lengths.quantile(0.75)/30:.2f}s)")
    print(f"Q90:  {ep_lengths.quantile(0.9):.0f} frames ({ep_lengths.quantile(0.9)/30:.2f}s)")

    # ===========================
    # SECTION 2: Phase timing across all episodes
    # ===========================
    print()
    print("=" * 70)
    print("SECTION 2: PHASE TIMING (all 74 episodes)")
    print("=" * 70)

    phase_results = []
    for ep_id in range(n_episodes):
        ep_df = df[df['episode_index'] == ep_id].reset_index(drop=True)
        phases = detect_phases(ep_df)
        phases['episode_id'] = ep_id
        phases['ep_length'] = ep_lengths[ep_id]
        phase_results.append(phases)

    pr = pd.DataFrame(phase_results)

    # Phase C timing
    has_C = pr['phase_C_start_frame'].notna()
    print(f"\nEpisodes with gripper >40 deg (Phase C exists): {has_C.sum()}/{n_episodes}")
    print(f"\nPhase C (gripper OPENS >40 deg) timing:")
    c_frames = pr.loc[has_C, 'phase_C_start_frame']
    c_pcts = pr.loc[has_C, 'phase_C_start_pct']
    n_ep = len(c_frames)
    print(f"  Frame number:  mean={c_frames.mean():.1f}, std={c_frames.std():.1f}, "
          f"min={c_frames.min():.0f}, max={c_frames.max():.0f}")
    print(f"  % into episode: mean={c_pcts.mean():.1f}%, std={c_pcts.std():.1f}%, "
          f"min={c_pcts.min():.1f}%, max={c_pcts.max():.1f}%")
    print(f"  At 30fps, Phase C starts at t={c_frames.mean()/30:.2f}s (mean), "
          f"range=[{c_frames.min()/30:.2f}s, {c_frames.max()/30:.2f}s]")

    # Phase D timing
    has_D = pr['phase_D_start_frame'].notna()
    print(f"\nEpisodes with Phase D (gripper closes after peak): {has_D.sum()}/{n_episodes}")
    print(f"\nPhase D (gripper CLOSES to gripped ~24 deg) timing:")
    d_frames = pr.loc[has_D, 'phase_D_start_frame']
    d_pcts = pr.loc[has_D, 'phase_D_start_pct']
    print(f"  Frame number:  mean={d_frames.mean():.1f}, std={d_frames.std():.1f}, "
          f"min={d_frames.min():.0f}, max={d_frames.max():.0f}")
    print(f"  % into episode: mean={d_pcts.mean():.1f}%, std={d_pcts.std():.1f}%, "
          f"min={d_pcts.min():.1f}%, max={d_pcts.max():.1f}%")
    print(f"  At 30fps, Phase D starts at t={d_frames.mean()/30:.2f}s (mean), "
          f"range=[{d_frames.min()/30:.2f}s, {d_frames.max()/30:.2f}s]")

    # Duration of open phase (C to D)
    has_CD = has_C & has_D
    print(f"\nEpisodes with both C and D detected: {has_CD.sum()}")
    if has_CD.sum() > 0:
        cd_duration = pr.loc[has_CD, 'phase_D_start_frame'] - pr.loc[has_CD, 'phase_C_start_frame']
        print(f"Phase C->D duration (gripper open phase):")
        print(f"  Frames: mean={cd_duration.mean():.1f}, std={cd_duration.std():.1f}, "
              f"min={cd_duration.min():.0f}, max={cd_duration.max():.0f}")
        print(f"  Time:   mean={cd_duration.mean()/30:.2f}s, "
              f"range=[{cd_duration.min()/30:.2f}s, {cd_duration.max()/30:.2f}s]")

    # ===========================
    # SECTION 3: 50-step action chunk analysis
    # ===========================
    print()
    print("=" * 70)
    print("SECTION 3: 50-STEP ACTION CHUNK ANALYSIS")
    print("=" * 70)
    print()
    print("In SmolVLA, n_action_steps=50 means the model outputs 50 future actions.")
    print("At 30fps, 50 steps = 1.67 seconds of future trajectory.")
    print()
    print("If chunk starts at frame 0 (init position):")
    print(f"  Chunk covers frames 0 to 49 ({50/30:.2f}s)")
    print()

    # How many episodes have Phase C within first 50 frames?
    c_within_50 = pr.loc[has_C, 'phase_C_start_frame'] < 50
    print(f"Episodes where Phase C (gripper opens) starts within first 50 frames:")
    print(f"  {c_within_50.sum()}/{has_C.sum()} = {c_within_50.mean()*100:.1f}%")
    print()

    # For episodes where C starts at frame X, at what fraction of episodes
    # does a 50-frame chunk from frame 0 capture the open event?
    print(f"Phase C start frame distribution vs 50-step window:")
    bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 9999]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-75', '76-100', '101-150', '151-200', '>200']
    for lo, hi, lab in zip(bins, bins[1:], labels):
        count = ((pr.loc[has_C, 'phase_C_start_frame'] >= lo) &
                 (pr.loc[has_C, 'phase_C_start_frame'] < hi)).sum()
        pct = count / has_C.sum() * 100
        print(f"  Frame {lab:>10}: {count:>3} episodes ({pct:.1f}%)")

    print()
    print("How many episodes have Phase D (gripped) within first 100 frames?")
    d_within_100 = pr.loc[has_D, 'phase_D_start_frame'] < 100
    print(f"  {d_within_100.sum()}/{has_D.sum()} = {d_within_100.mean()*100:.1f}%")
    print()

    print("==> CONCLUSION: Can 50-step chunk (1.67s) capture the ENTIRE grasp cycle?")
    mean_total = ep_lengths.mean()
    print(f"  Mean episode length: {mean_total:.0f} frames ({mean_total/30:.2f}s)")
    print(f"  50 steps covers: {50/mean_total*100:.1f}% of mean episode")
    print(f"  Full cycle needs: {mean_total:.0f} steps (at mean episode length)")
    print(f"  ==> 50-step chunk covers only the EARLY portion of each episode.")

    # ===========================
    # SECTION 4: Per-episode summary table
    # ===========================
    print()
    print("=" * 70)
    print("SECTION 4: PER-EPISODE PHASE SUMMARY TABLE (all episodes)")
    print("=" * 70)
    print(f"\n{'Ep':>4} {'Frames':>7} {'Duration':>9} {'GripMax':>8} {'GripPeakF':>10} "
          f"{'PhC_F':>7} {'PhC%':>6} {'PhD_F':>7} {'PhD%':>6}")
    print("-" * 75)
    for _, row in pr.iterrows():
        c_f = f"{int(row['phase_C_start_frame']):>7}" if pd.notna(row['phase_C_start_frame']) else "   None"
        c_p = f"{row['phase_C_start_pct']:>6.1f}%" if pd.notna(row['phase_C_start_pct']) else "      -"
        d_f = f"{int(row['phase_D_start_frame']):>7}" if pd.notna(row['phase_D_start_frame']) else "   None"
        d_p = f"{row['phase_D_start_pct']:>6.1f}%" if pd.notna(row['phase_D_start_pct']) else "      -"
        print(f"{int(row['episode_id']):>4} {int(row['ep_length']):>7} {row['duration_sec']:>8.2f}s "
              f"{row['gripper_max']:>8.1f} {int(row['gripper_max_frame']):>10} "
              f"{c_f} {c_p} {d_f} {d_p}")

    # ===========================
    # SECTION 5: Frame-by-frame trajectory for 5 specific episodes
    # ===========================
    print()
    print("=" * 70)
    print("SECTION 5: FRAME-BY-FRAME TRAJECTORIES (5 specific episodes)")
    print("=" * 70)
    print("Selecting episodes with clear gripper open->close pattern...")

    # Pick 5 episodes with widest gripper range and clear phase transitions
    # Prefer episodes where phase C is detectable and phase D is also detectable
    good_eps = pr.loc[has_CD, 'episode_id'].values
    # Sort by gripper_max to pick varied examples
    good_pr = pr[pr['episode_id'].isin(good_eps)].sort_values('gripper_max', ascending=False)
    # Pick one from top, one from middle, one from short episodes, one from long
    selected_ep_ids = []

    # Top gripper range
    selected_ep_ids.append(int(good_pr.iloc[0]['episode_id']))
    # Middle-ish
    mid_idx = len(good_pr) // 2
    selected_ep_ids.append(int(good_pr.iloc[mid_idx]['episode_id']))
    # Shortest episode with clear phases
    shortest = good_pr.sort_values('ep_length').iloc[0]
    selected_ep_ids.append(int(shortest['episode_id']))
    # Longest episode with clear phases
    longest = good_pr.sort_values('ep_length').iloc[-1]
    selected_ep_ids.append(int(longest['episode_id']))
    # Episode with earliest Phase C (gripper opens fastest)
    earliest_C = good_pr.sort_values('phase_C_start_frame').iloc[0]
    selected_ep_ids.append(int(earliest_C['episode_id']))

    # Remove duplicates preserving order
    seen = set()
    unique_selected = []
    for ep in selected_ep_ids:
        if ep not in seen:
            seen.add(ep)
            unique_selected.append(ep)
    selected_ep_ids = unique_selected[:5]

    print(f"Selected episodes: {selected_ep_ids}")
    print()
    print("Columns: Frame | Time | Gripper(action) | Shoulder(action) | Elbow(action) | Phase")
    print("Phase labels: A=START, B=APPROACH, C=OPEN(>40deg), D=GRIPPED(<30 after peak), E=LIFT, F=END")

    for ep_id in selected_ep_ids:
        ep_df = df[df['episode_index'] == ep_id].reset_index(drop=True)
        print_episode_gripper_trajectory(ep_df, ep_id, max_rows=60)

    # ===========================
    # SECTION 6: Key summary for action chunk design
    # ===========================
    print()
    print("=" * 70)
    print("SECTION 6: KEY SUMMARY FOR ACTION CHUNK DESIGN")
    print("=" * 70)
    print()

    mean_C = c_frames.mean() if len(c_frames) > 0 else float('nan')
    mean_D = d_frames.mean() if len(d_frames) > 0 else float('nan')
    mean_len = ep_lengths.mean()

    print(f"Dataset: 74 episodes, mean={mean_len:.0f} frames ({mean_len/30:.2f}s) at 30fps")
    print()
    print(f"Phase C (gripper opens >40 deg):")
    print(f"  Starts at mean frame {mean_C:.0f} ({mean_C/30:.2f}s, {mean_C/mean_len*100:.1f}% into episode)")
    print(f"  Range: frame {c_frames.min():.0f} to {c_frames.max():.0f}")
    print()
    print(f"Phase D (gripper closes to gripped ~24 deg):")
    print(f"  Starts at mean frame {mean_D:.0f} ({mean_D/30:.2f}s, {mean_D/mean_len*100:.1f}% into episode)")
    print(f"  Range: frame {d_frames.min():.0f} to {d_frames.max():.0f}")
    print()
    print(f"50-step chunk (1.67s) from frame 0:")
    print(f"  Only covers {50/mean_len*100:.0f}% of mean episode")
    print(f"  Phase C (gripper open) occurs at mean frame {mean_C:.0f}")

    if mean_C <= 50:
        print(f"  ==> Gripper opens WITHIN the 50-step chunk (frame {mean_C:.0f} < 50) -- GOOD!")
    else:
        print(f"  ==> Gripper opens AFTER the 50-step chunk (frame {mean_C:.0f} > 50) -- PROBLEM!")
        print(f"  ==> A 50-step chunk from frame 0 would NOT see the gripper open event!")
        print(f"  ==> Model must predict 50 steps ahead to reach the gripper open event")

    print()
    print(f"Minimum steps needed to see the FULL grasp cycle from frame 0:")
    print(f"  Full episode: mean={mean_len:.0f} steps ({mean_len/30:.2f}s)")
    print(f"  To reach Phase C: mean={mean_C:.0f} steps")
    print(f"  To reach Phase D: mean={mean_D:.0f} steps")
    print()
    print("NOTE: SmolVLA uses n_action_steps=50 = 1.67s lookahead.")
    print("The model does NOT execute 50 steps and then re-plan.")
    print("In closed-loop with n_action_steps=50, it executes ALL 50 steps,")
    print("then gets a new observation and predicts the next 50 steps.")
    print("==> Each re-plan cycle = 1.67s window. Full episode needs ~N re-plans.")


if __name__ == '__main__':
    main()
