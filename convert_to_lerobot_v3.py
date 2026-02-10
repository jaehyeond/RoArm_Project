"""
수집된 데이터를 LeRobot v3.0 포맷으로 변환 (새 LeRobot 0.4.3 호환)

사용법:
    python convert_to_lerobot_v3.py --task "Pick up the white box"
"""

import json
import os
import shutil
from pathlib import Path

# ffmpeg PATH 설정 (Windows winget 설치 경로)
FFMPEG_PATH = r"C:\Users\SOGANG\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
if os.path.exists(FFMPEG_PATH) and FFMPEG_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

import cv2
import numpy as np
import torch
from tqdm import tqdm

# LeRobot imports (pip installed version)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def convert_collected_data(
    input_dir: str = "collected_data",
    output_dir: str = "lerobot_dataset_v3",
    repo_id: str = "roarm_m3_pick",
    fps: int = 30,
    task_description: str = "Pick up the white box",
):
    """
    수집된 데이터를 LeRobot v3.0 포맷으로 변환

    Args:
        input_dir: 수집된 데이터 디렉토리 (episode_XXXX 폴더들)
        output_dir: 출력 디렉토리
        repo_id: 데이터셋 ID (local/ prefix 없이)
        fps: 프레임 레이트
        task_description: 태스크 설명 (language instruction)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 에피소드 목록
    episodes = sorted(input_path.glob("episode_*"))
    if not episodes:
        print(f"에피소드를 찾을 수 없습니다: {input_path}")
        return None

    print(f"발견된 에피소드: {len(episodes)}개")

    # 출력 디렉토리 정리 (이미 존재하면 삭제)
    full_output_path = output_path / repo_id
    if full_output_path.exists():
        print(f"기존 데이터셋 삭제: {full_output_path}")
        shutil.rmtree(full_output_path)

    # Features 정의 (새 LeRobot 포맷)
    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channel"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "motors": ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
            }
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "motors": ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
            }
        }
    }

    print(f"데이터셋 생성 중: {repo_id}")
    print(f"출력 경로: {full_output_path}")

    # LeRobotDataset 생성 (새 API)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_path,
        robot_type="roarm_m3",
        features=features,
        use_videos=True,
    )

    total_frames = 0

    # 각 에피소드 처리
    for ep_idx, ep_path in enumerate(tqdm(episodes, desc="에피소드 변환")):
        # 메타데이터 로드
        meta_path = ep_path / "metadata.json"
        if not meta_path.exists():
            print(f"메타데이터 없음, 건너뜀: {ep_path}")
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        frames = meta["frames"]
        num_frames = len(frames)

        if num_frames == 0:
            print(f"빈 에피소드, 건너뜀: {ep_path}")
            continue

        print(f"\n에피소드 {ep_idx}: {num_frames} 프레임")

        # 각 프레임 처리
        for i, frame_data in enumerate(tqdm(frames, desc=f"  프레임", leave=False)):
            # RGB 이미지 로드
            rgb_path = ep_path / frame_data["rgb_path"]
            if not rgb_path.exists():
                print(f"이미지 없음: {rgb_path}")
                continue

            img = cv2.imread(str(rgb_path))
            if img is None:
                print(f"이미지 로드 실패: {rgb_path}")
                continue

            # BGR -> RGB 변환
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # State (현재 관절 위치)
            state = np.array(frame_data["angles"], dtype=np.float32)

            # Action (다음 프레임의 관절 위치)
            # 마지막 프레임은 현재 state 유지
            if i < num_frames - 1:
                action = np.array(frames[i + 1]["angles"], dtype=np.float32)
            else:
                action = state.copy()

            # 프레임 추가 (새 API)
            frame = {
                "observation.images.top": img,
                "observation.state": torch.from_numpy(state),
                "action": torch.from_numpy(action),
                "task": task_description,  # 각 프레임에 task 포함
            }

            dataset.add_frame(frame)
            total_frames += 1

        # 에피소드 저장
        dataset.save_episode()
        print(f"  에피소드 {ep_idx} 저장 완료 (task: {task_description})")

    # 데이터셋 정리 (finalize 호출 필수 - parquet 메타데이터 flush)
    dataset.finalize()
    print("데이터셋 finalize 완료")

    print(f"\n{'='*50}")
    print(f"변환 완료!")
    print(f"  총 에피소드: {len(episodes)}")
    print(f"  총 프레임: {total_frames}")
    print(f"  Task: {task_description}")
    print(f"  출력 경로: {full_output_path}")
    print(f"{'='*50}")

    return dataset


def verify_dataset(output_dir: str = "lerobot_dataset_v3", repo_id: str = "roarm_m3_pick"):
    """변환된 데이터셋 검증"""
    print("\n데이터셋 검증 중...")

    try:
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=Path(output_dir),
        )

        print(f"  총 프레임: {len(dataset)}")
        print(f"  에피소드 수: {dataset.num_episodes}")
        print(f"  FPS: {dataset.fps}")
        print(f"  Features: {list(dataset.features.keys())}")

        # 첫 번째 샘플 확인
        sample = dataset[0]
        print(f"\n  첫 번째 샘플:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.shape}")

        print("\n데이터셋 검증 성공!")
        return True

    except Exception as e:
        print(f"데이터셋 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="수집된 데이터를 LeRobot v3.0 포맷으로 변환")
    parser.add_argument("--input", default="collected_data", help="입력 디렉토리")
    parser.add_argument("--output", default="lerobot_dataset_v3", help="출력 디렉토리")
    parser.add_argument("--repo-id", default="roarm_m3_pick", help="데이터셋 repo ID (local/ 없이)")
    parser.add_argument("--fps", type=int, default=30, help="프레임 레이트")
    parser.add_argument("--task", default="Pick up the white box", help="태스크 설명")
    parser.add_argument("--verify-only", action="store_true", help="검증만 실행")

    args = parser.parse_args()

    if args.verify_only:
        verify_dataset(args.output, args.repo_id)
    else:
        # 변환 실행
        dataset = convert_collected_data(
            input_dir=args.input,
            output_dir=args.output,
            repo_id=args.repo_id,
            fps=args.fps,
            task_description=args.task,
        )

        if dataset:
            # 변환 후 검증
            verify_dataset(args.output, args.repo_id)
