"""
수집된 데이터를 LeRobot v3.0 포맷으로 변환 (새 LeRobot 0.4.3 호환)

사용법:
    python convert_to_lerobot_v3.py --task "Pick up the sponge"
"""

import json
import os
import shutil
from pathlib import Path

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
    task_description: str = "Pick up the sponge",
    multi_object: bool = False,
    second_cam_key: str = "auto",
    force: bool = False,
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

    # 듀얼 카메라 자동 감지 (첫 에피소드의 metadata 확인)
    has_second_cam = False
    second_cam_obs_key = None
    first_meta_path = episodes[0] / "metadata.json"
    if first_meta_path.exists():
        with open(first_meta_path) as f:
            first_meta = json.load(f)
        second_cam_type = first_meta.get("second_camera", "none")
        if second_cam_type != "none":
            has_second_frame = any("second_path" in fr for fr in first_meta.get("frames", []))
            if has_second_frame:
                has_second_cam = True
                # observation key: CLI 오버라이드 또는 카메라 타입에서 자동 결정
                if second_cam_key == "auto":
                    if second_cam_type == "zed_wrist":
                        second_cam_obs_key = "observation.images.wrist"
                    else:
                        second_cam_obs_key = "observation.images.side"
                else:
                    second_cam_obs_key = f"observation.images.{second_cam_key}"
                print(f"듀얼 카메라 감지됨: {second_cam_type} → {second_cam_obs_key}")

    # 출력 디렉토리 정리 (이미 존재하면 삭제)
    full_output_path = output_path / repo_id
    if full_output_path.exists():
        if force:
            print(f"기존 데이터셋 삭제: {full_output_path}")
            shutil.rmtree(full_output_path)
        else:
            confirm = input(f"기존 데이터셋 삭제? {full_output_path} (y/n): ").strip().lower()
            if confirm != 'y':
                print("변환 취소됨")
                return None
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

    if has_second_cam and second_cam_obs_key:
        features[second_cam_obs_key] = {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channel"]
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

        # Multi-object: 에피소드 메타데이터에서 물체명 읽기
        if multi_object and "object" in meta:
            ep_task = f"Pick up the {meta['object']}\n"
        else:
            ep_task = task_description if task_description.endswith("\n") else task_description + "\n"

        print(f"\n에피소드 {ep_idx}: {num_frames} 프레임 (task: {ep_task.strip()})")

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
                "task": ep_task,  # 에피소드별 task (multi-object 지원)
            }

            # 두 번째 카메라 이미지 (있으면)
            if has_second_cam and second_cam_obs_key and "second_path" in frame_data:
                second_img_path = ep_path / frame_data["second_path"]
                if second_img_path.exists():
                    second_img = cv2.imread(str(second_img_path))
                    if second_img is not None:
                        second_img = cv2.cvtColor(second_img, cv2.COLOR_BGR2RGB)
                        if second_img.shape[:2] != (720, 1280):
                            second_img = cv2.resize(second_img, (1280, 720))
                        frame[second_cam_obs_key] = second_img

            dataset.add_frame(frame)
            total_frames += 1

        # 에피소드 저장
        dataset.save_episode()
        print(f"  에피소드 {ep_idx} 저장 완료 (task: {ep_task.strip()})")

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
    parser.add_argument("--input", default="collected_data",
                        help="입력 디렉토리 (multi-object: 콤마 구분, e.g. collected_data_sponge,collected_data_cup)")
    parser.add_argument("--output", default="lerobot_dataset_v3", help="출력 디렉토리")
    parser.add_argument("--repo-id", default="roarm_m3_pick", help="데이터셋 repo ID (local/ 없이)")
    parser.add_argument("--fps", type=int, default=30, help="프레임 레이트")
    parser.add_argument("--task", default="Pick up the sponge\n", help="태스크 설명 (single-object용)")
    parser.add_argument("--multi-object", action="store_true",
                        help="에피소드 metadata.json에서 물체명 자동 읽기 (task text 자동 생성)")
    parser.add_argument("--second-cam-key", default="auto",
                        help="두 번째 카메라 observation key (auto/wrist/side/external)")
    parser.add_argument("--force", action="store_true",
                        help="기존 데이터셋 삭제 시 확인 없이 진행")
    parser.add_argument("--verify-only", action="store_true", help="검증만 실행")

    args = parser.parse_args()

    if args.verify_only:
        verify_dataset(args.output, args.repo_id)
    else:
        # Multi-object: 콤마로 구분된 여러 입력 디렉토리 → 임시 병합
        input_dirs = [d.strip() for d in args.input.split(",")]

        if len(input_dirs) > 1:
            # 여러 디렉토리의 에피소드를 하나의 임시 디렉토리로 심링크
            import tempfile
            merged_dir = tempfile.mkdtemp(prefix="merged_episodes_")
            ep_counter = 0
            for d in input_dirs:
                d_path = Path(d)
                if not d_path.exists():
                    print(f"경고: {d} 존재하지 않음, 건너뜀")
                    continue
                for ep in sorted(d_path.glob("episode_*")):
                    dst = Path(merged_dir) / f"episode_{ep_counter:04d}"
                    os.symlink(ep.resolve(), dst)
                    ep_counter += 1
            print(f"Multi-object: {ep_counter}개 에피소드 병합 ({', '.join(input_dirs)})")
            input_dir = merged_dir
        else:
            input_dir = input_dirs[0]

        dataset = convert_collected_data(
            input_dir=input_dir,
            output_dir=args.output,
            repo_id=args.repo_id,
            fps=args.fps,
            task_description=args.task,
            multi_object=args.multi_object,
            second_cam_key=args.second_cam_key,
            force=args.force,
        )

        # 임시 병합 디렉토리 정리
        if len(input_dirs) > 1:
            shutil.rmtree(merged_dir)

        if dataset:
            verify_dataset(args.output, args.repo_id)
