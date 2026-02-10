"""데이터셋 Action 분포 분석"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load action data from parquet
parquet_path = Path(r"E:\RoArm_Project\lerobot_dataset_v3\data\chunk-000\file-000.parquet")
df = pd.read_parquet(parquet_path)

print("=" * 60)
print("데이터셋 Action 분포 분석")
print("=" * 60)
print(f"\n총 프레임 수: {len(df)}")
print(f"에피소드 수: {df['episode_index'].nunique()}")
print(f"\n컬럼 목록: {df.columns.tolist()}")

# Action columns
action_cols = [c for c in df.columns if c.startswith('action')]
state_cols = [c for c in df.columns if 'state' in c.lower()]
print(f"\nAction columns: {action_cols}")
print(f"State columns: {state_cols}")

# Action statistics
print("\n" + "=" * 60)
print("각 관절별 Action 통계")
print("=" * 60)
joint_names = ["Base", "Shoulder", "Elbow", "Wrist_pitch", "Wrist_roll", "Gripper"]

# Check if action is a single column (list) or multiple columns
if 'action' in df.columns:
    # Action is stored as a list in single column
    actions = np.array(df['action'].tolist())
    print(f"Action shape: {actions.shape}")

    for i, name in enumerate(joint_names):
        data = actions[:, i]
        print(f"{name:12s}: min={data.min():7.2f}, max={data.max():7.2f}, mean={data.mean():7.2f}, std={data.std():7.2f}, range={data.max()-data.min():7.2f}")
else:
    # Multiple columns
    for i, col in enumerate(action_cols[:6]):
        data = df[col]
        print(f"{joint_names[i]:12s}: min={data.min():7.2f}, max={data.max():7.2f}, mean={data.mean():7.2f}, std={data.std():7.2f}, range={data.max()-data.min():7.2f}")

# Range analysis
print("\n" + "=" * 60)
print("Action 범위 요약 (문제 진단)")
print("=" * 60)

if 'action' in df.columns:
    actions = np.array(df['action'].tolist())
    for i, name in enumerate(joint_names):
        data = actions[:, i]
        range_val = data.max() - data.min()
        if range_val < 20:
            status = "[WARNING] Very narrow"
        elif range_val < 40:
            status = "[NARROW]"
        elif range_val < 60:
            status = "[MEDIUM]"
        else:
            status = "[GOOD]"
        print(f"{name:12s}: 범위 {range_val:6.1f}° {status}")

# Episode-level analysis
print("\n" + "=" * 60)
print("에피소드별 시작 위치 분석 (다양성 확인)")
print("=" * 60)

if 'action' in df.columns:
    # Get first frame of each episode
    first_frames = df.groupby('episode_index').first()
    actions_first = np.array(first_frames['action'].tolist()) if 'action' in first_frames.columns else None

    if actions_first is not None:
        print(f"\n에피소드별 시작 위치 (첫 프레임):")
        for i, name in enumerate(joint_names):
            data = actions_first[:, i]
            print(f"{name:12s}: min={data.min():7.2f}, max={data.max():7.2f}, std={data.std():7.2f}")

        # Check if starting positions are too similar
        print("\n시작 위치 다양성 분석:")
        for i, name in enumerate(joint_names):
            data = actions_first[:, i]
            if data.std() < 5:
                print(f"  {name}: std={data.std():.2f} deg [WARNING] Starting positions too similar!")
            else:
                print(f"  {name}: std={data.std():.2f} deg [OK]")

# Mean action analysis
print("\n" + "=" * 60)
print("모델이 학습한 '평균 액션' 확인")
print("=" * 60)

if 'action' in df.columns:
    actions = np.array(df['action'].tolist())
    mean_action = actions.mean(axis=0)
    print(f"데이터셋 전체 평균: {mean_action}")
    print(f"\n모델 예측값:        [~0-2, ~30-34, ~-0.4-0.7, ~64-68, ~7-8, ~1.6-1.9]")
    print(f"데이터셋 평균과 비교해보세요!")
