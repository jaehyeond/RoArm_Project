"""
v6 데이터 검증 스크립트

Phase 0: 5ep 수집 후 실행 → 구조 검증
Phase 1: 50ep 변환 후 실행 → parquet 분석

v5 136ep 전량 실패 교훈:
  - HOME 미시작 → echo → 배포 실패
  - 에피소드 99프레임(3.3초) → 공식 393프레임(13초)의 1/4
  - approach phase 부재 → visual grounding 학습 불가

v3 false positive 교훈:
  - 1곳(RIGHT ~45°)에서만 테스트 → 궤적 암기와 구분 불가
  - M2=1.73° FAIL → 이미지 무시
  - 57% CENTER 편향

Usage:
  # Phase 0: raw 데이터 검증
  python data_v6_verify.py --raw collected_data_v6

  # Phase 1: LeRobot parquet 검증
  python data_v6_verify.py --parquet lerobot_dataset_v6
"""

import argparse
import json
import os
import sys

import numpy as np


HOME = np.array([0, 0, 90, 0, 0, 0], dtype=np.float32)
HOME_DIST_THRESHOLD = 30.0  # degrees
MIN_EPISODE_LENGTH = 120  # frames (4 sec @ 30fps)
MIN_APPROACH_BASE = 5.0  # degrees


def verify_raw(raw_dir: str):
    """Phase 0: collected_data_v6/ raw 데이터 검증"""
    episodes = sorted(
        d for d in os.listdir(raw_dir)
        if d.startswith("episode_") and os.path.isdir(os.path.join(raw_dir, d))
    )
    if not episodes:
        print(f"FAIL: No episodes found in {raw_dir}")
        return False

    print(f"Found {len(episodes)} episodes in {raw_dir}")
    print()

    all_pass = True
    zone_counts = {}

    for ep_name in episodes:
        ep_path = os.path.join(raw_dir, ep_name)
        meta_path = os.path.join(ep_path, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"  {ep_name}: SKIP (no metadata)")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        frames = meta.get("frames", [])
        n_frames = len(frames)
        zone = meta.get("zone", "UNKNOWN")
        zone_counts[zone] = zone_counts.get(zone, 0) + 1

        issues = []

        # Check 1: Episode length
        if n_frames < MIN_EPISODE_LENGTH:
            issues.append(f"Too short: {n_frames} < {MIN_EPISODE_LENGTH}")

        # Check 2: Start from HOME
        if frames:
            start_angles = np.array(frames[0]["angles"], dtype=np.float32)
            home_dist = np.linalg.norm(start_angles - HOME)
            if home_dist > HOME_DIST_THRESHOLD:
                issues.append(f"NOT HOME: dist={home_dist:.0f}° start={start_angles.round(0).tolist()}")
        else:
            issues.append("No frames")

        # Check 3: Approach phase (base OR shoulder movement)
        # CENTER zone: base 안 움직여도 shoulder은 반드시 변함 (HOME→물체 하강)
        if n_frames >= 20:
            first_base = frames[0]["angles"][0]
            first_shoulder = frames[0]["angles"][1]
            mid_base = frames[n_frames // 2]["angles"][0]
            mid_shoulder = frames[n_frames // 2]["angles"][1]
            base_travel = abs(mid_base - first_base)
            shoulder_travel = abs(mid_shoulder - first_shoulder)
            if base_travel < MIN_APPROACH_BASE and shoulder_travel < MIN_APPROACH_BASE:
                issues.append(f"No approach: base {base_travel:.1f}°, shoulder {shoulder_travel:.1f}° (both < {MIN_APPROACH_BASE}°)")

        # Check 4: Gripper opened
        max_gripper = max(f["angles"][5] for f in frames) if frames else 0
        if max_gripper < 40:
            issues.append(f"Gripper never opened: max={max_gripper:.0f}°")

        status = "PASS" if not issues else "FAIL"
        if issues:
            all_pass = False
        print(f"  {ep_name} [{zone:>10s}] {n_frames:4d}fr: {status}")
        for issue in issues:
            print(f"    - {issue}")

    # Zone distribution
    print(f"\nZone distribution:")
    for z in ["FAR_LEFT", "LEFT", "CENTER", "RIGHT", "FAR_RIGHT"]:
        count = zone_counts.get(z, 0)
        print(f"  {z:>10s}: {count}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'HAS FAILURES'}")
    return all_pass


def verify_parquet(dataset_dir: str):
    """Phase 1: LeRobot parquet 데이터 검증"""
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available. Use: conda run -n roarm python data_v6_verify.py ...")
        return False

    parquet_path = os.path.join(dataset_dir, "data", "chunk-000", "file-000.parquet")
    if not os.path.exists(parquet_path):
        print(f"FAIL: {parquet_path} not found")
        return False

    df = pd.read_parquet(parquet_path)
    states = np.stack(df["observation.state"].values)
    actions = np.stack(df["action"].values)

    n_episodes = df["episode_index"].nunique()
    n_frames = len(df)
    print(f"Dataset: {n_episodes} episodes, {n_frames} frames")
    print()

    all_pass = True
    issues_summary = []

    # 1. Start positions — ALL must be HOME
    first_frames = df.groupby("episode_index").first()
    start_states = np.stack(first_frames["observation.state"].values)
    home_dists = np.linalg.norm(start_states - HOME, axis=1)
    n_home = (home_dists < HOME_DIST_THRESHOLD).sum()
    print(f"1. HOME start: {n_home}/{n_episodes} ({n_home/n_episodes*100:.0f}%)")
    print(f"   Mean dist: {home_dists.mean():.1f}°, Max: {home_dists.max():.1f}°")
    if n_home < n_episodes:
        issues_summary.append(f"HOME start: {n_home}/{n_episodes}")
        all_pass = False
    print()

    # 2. Episode lengths
    ep_lens = df.groupby("episode_index").size()
    print(f"2. Episode length: mean={ep_lens.mean():.0f}, std={ep_lens.std():.0f}, min={ep_lens.min()}, max={ep_lens.max()}")
    short = (ep_lens < MIN_EPISODE_LENGTH).sum()
    if short > 0:
        issues_summary.append(f"Short episodes: {short}/{n_episodes}")
        all_pass = False
    print(f"   < {MIN_EPISODE_LENGTH}fr: {short}/{n_episodes}")
    print(f"   Official reference: 393fr avg (13 sec)")
    print()

    # 3. Approach phase — base OR shoulder movement (HOME → target)
    approach_ok = 0
    for ep_idx in df["episode_index"].unique():
        ep = df[df["episode_index"] == ep_idx]
        ep_states = np.stack(ep["observation.state"].values)
        if len(ep_states) >= 20:
            first_base = ep_states[0, 0]
            first_shoulder = ep_states[0, 1]
            mid_base = ep_states[len(ep_states)//2, 0]
            mid_shoulder = ep_states[len(ep_states)//2, 1]
            if abs(mid_base - first_base) >= MIN_APPROACH_BASE or abs(mid_shoulder - first_shoulder) >= MIN_APPROACH_BASE:
                approach_ok += 1
    print(f"3. Approach phase (base OR shoulder > {MIN_APPROACH_BASE}°): {approach_ok}/{n_episodes}")
    if approach_ok < n_episodes * 0.8:
        issues_summary.append(f"Approach missing: {approach_ok}/{n_episodes}")
        all_pass = False
    print()

    # 4. Zone distribution (by grasp-moment base angle)
    print("4. Zone distribution (by trajectory mean base):")
    ep_base_means = df.groupby("episode_index").apply(
        lambda x: np.stack(x["observation.state"].values)[:, 0].mean()
    )
    zones = {"FAR_LEFT": 0, "LEFT": 0, "CENTER": 0, "RIGHT": 0, "FAR_RIGHT": 0}
    for base_mean in ep_base_means:
        if base_mean < -40:
            zones["FAR_LEFT"] += 1
        elif base_mean < -10:
            zones["LEFT"] += 1
        elif base_mean <= 10:
            zones["CENTER"] += 1
        elif base_mean <= 40:
            zones["RIGHT"] += 1
        else:
            zones["FAR_RIGHT"] += 1
    for z, c in zones.items():
        pct = c / n_episodes * 100
        bar = "#" * int(pct / 2)
        print(f"   {z:>10s}: {c:3d} ({pct:4.0f}%) {bar}")
    max_zone_pct = max(c / n_episodes for c in zones.values())
    if max_zone_pct > 0.4:
        issues_summary.append(f"Zone imbalance: max {max_zone_pct*100:.0f}%")
        all_pass = False
    print()

    # 5. Action-state delta (echo check)
    deltas = np.abs(actions - states)
    mean_delta = deltas.mean(axis=0)
    joint_names = ["Base", "Shoulder", "Elbow", "WristP", "WristR", "Gripper"]
    print("5. Mean |action - state| per joint:")
    for i, jn in enumerate(joint_names):
        print(f"   {jn}: {mean_delta[i]:.3f}°")
    print(f"   |delta| > 0.5°: {(deltas[:, 0] > 0.5).mean()*100:.1f}% of frames (Base)")
    print(f"   v3 reference: 17.7%, v5 (echo): 11.4%")
    print()

    # 6. Gripper diversity
    gripper_states = states[:, 5]
    print(f"6. Gripper: range [{gripper_states.min():.1f}, {gripper_states.max():.1f}]")
    gripper_open_pct = (gripper_states > 40).mean() * 100
    print(f"   Open (>40°): {gripper_open_pct:.1f}% of frames")
    print()

    # 7. Stats comparison with official
    print("7. State statistics:")
    print(f"   Mean: {states.mean(axis=0).round(2).tolist()}")
    print(f"   Std:  {states.std(axis=0).round(2).tolist()}")
    print()

    # Summary
    if issues_summary:
        print("ISSUES FOUND:")
        for issue in issues_summary:
            print(f"  - {issue}")
    print(f"\nOverall: {'ALL CHECKS PASS' if all_pass else 'HAS ISSUES — review before training'}")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v6 데이터 검증")
    parser.add_argument("--raw", help="Raw collected data directory (Phase 0)")
    parser.add_argument("--parquet", help="LeRobot dataset directory (Phase 1)")
    args = parser.parse_args()

    if not args.raw and not args.parquet:
        print("Usage: python data_v6_verify.py --raw collected_data_v6")
        print("       python data_v6_verify.py --parquet lerobot_dataset_v6")
        sys.exit(1)

    if args.raw:
        verify_raw(args.raw)
    if args.parquet:
        verify_parquet(args.parquet)
