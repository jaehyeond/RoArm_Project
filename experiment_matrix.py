"""
Scaling Law Experiment Matrix for CoRL 2026

OOD Scaling Laws: episode count × training steps → offline metrics
- 에피소드 서브샘플링: 원본 데이터셋 복사 → delete_episodes()
- 학습: 각 서브셋별 200K run (체크포인트 10K 간격)
- 평가: 모든 체크포인트에서 L2 error + diversity 측정

실험 구조:
  experiments/
    ep010_seed0/          # 10 episodes, seed 0
      dataset/            # 서브샘플링된 LeRobot 데이터셋
      training/           # 학습 출력 (체크포인트 포함)
    ep025_seed0/
    ep050_seed0/
    ep074_seed0/          # 현재 전체 데이터 (baseline)
    ep100_seed0/          # 추가 수집 후
    ep150_seed0/          # 추가 수집 후
    results.csv           # 전체 결과 수집

사용법:
    # 1. 서브셋 데이터셋 생성 (학습 전에 먼저 실행)
    python experiment_matrix.py prepare --source lerobot_dataset_v3 --episodes 10,25,50

    # 2. 특정 서브셋 학습 실행
    python experiment_matrix.py train --subset ep010_seed0

    # 3. 전체 학습 순차 실행
    python experiment_matrix.py train-all

    # 4. 체크포인트 평가
    python experiment_matrix.py eval --subset ep010_seed0

    # 5. 전체 평가 + 결과 수집
    python experiment_matrix.py eval-all

    # 6. 결과 요약 출력
    python experiment_matrix.py summary
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import csv
import random
from pathlib import Path
from datetime import datetime

import numpy as np

# === Configuration ===
EXPERIMENT_DIR = Path("experiments")
REPO_ID = "roarm_m3_pick"
PRETRAINED = "lerobot/smolvla_base"

# Training hyperparameters (공식 SmolVLA 설정 기반)
BATCH_SIZE = 64
MAX_STEPS = 200_000
SAVE_FREQ = 10_000       # 20 checkpoints → steps 축 커버
EVAL_FREQ = 20_000
LOG_FREQ = 100
WARMUP_STEPS = 2_000
DECAY_LR = 2.5e-6
NUM_WORKERS = 4

# Episode counts to test
DEFAULT_EPISODE_COUNTS = [10, 25, 50, 74, 100, 150]

# Evaluation settings
EVAL_NUM_SAMPLES = 200    # 평가 시 사용할 프레임 수
EVAL_EPISODES = 5         # 평가용 에피소드 수 (각 서브셋에서 제외된 에피소드)


def get_subset_name(n_episodes: int, seed: int = 0) -> str:
    """서브셋 디렉토리 이름 생성."""
    return f"ep{n_episodes:03d}_seed{seed}"


def get_total_episodes(dataset_dir: Path) -> int:
    """데이터셋의 총 에피소드 수 확인."""
    info_path = dataset_dir / REPO_ID / "meta" / "info.json"
    if not info_path.exists():
        # v3 구조 대체 경로
        info_path = dataset_dir / REPO_ID / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found in {dataset_dir}")

    with open(info_path) as f:
        info = json.load(f)
    return info["total_episodes"]


def prepare_subset(source_dir: Path, n_episodes: int, seed: int = 0,
                   strategy: str = "random") -> Path:
    """
    원본 데이터셋에서 N개 에피소드 서브셋 생성.

    Args:
        source_dir: 원본 LeRobot 데이터셋 경로
        n_episodes: 선택할 에피소드 수
        seed: 랜덤 시드 (재현성)
        strategy: 'random' | 'first' | 'stratified'

    Returns:
        서브셋 데이터셋 경로
    """
    total = get_total_episodes(source_dir)

    if n_episodes > total:
        print(f"경고: 요청 {n_episodes}개 > 전체 {total}개. 전체 데이터셋 사용.")
        n_episodes = total

    if n_episodes == total:
        print(f"전체 데이터셋 사용 ({total}개 에피소드)")
        strategy = "all"

    subset_name = get_subset_name(n_episodes, seed)
    subset_dir = EXPERIMENT_DIR / subset_name / "dataset"

    if subset_dir.exists():
        print(f"이미 존재: {subset_dir}")
        return subset_dir

    print(f"\n{'='*60}")
    print(f"서브셋 생성: {n_episodes}/{total} 에피소드 (seed={seed}, strategy={strategy})")
    print(f"{'='*60}")

    # 1. 원본 복사
    print(f"데이터셋 복사: {source_dir} → {subset_dir}")
    subset_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, subset_dir)

    if strategy == "all":
        print("전체 데이터셋 복사 완료")
        return subset_dir

    # 2. 제거할 에피소드 선택
    all_episodes = list(range(total))
    rng = random.Random(seed)

    if strategy == "random":
        keep = sorted(rng.sample(all_episodes, n_episodes))
    elif strategy == "first":
        keep = all_episodes[:n_episodes]
    elif strategy == "stratified":
        # 균등 간격 선택 (zone 다양성 보장)
        step = total / n_episodes
        keep = sorted([int(i * step) for i in range(n_episodes)])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    remove = sorted(set(all_episodes) - set(keep))

    print(f"유지: {len(keep)}개, 제거: {len(remove)}개")
    print(f"유지 에피소드: {keep[:10]}{'...' if len(keep) > 10 else ''}")

    # 3. delete_episodes() 호출
    if remove:
        print("에피소드 제거 중 (delete_episodes)...")
        try:
            from lerobot.datasets.dataset_tools import delete_episodes
            delete_episodes(
                dataset_dir=subset_dir / REPO_ID,
                episode_indices=remove,
            )
        except ImportError:
            print("ERROR: lerobot.datasets.dataset_tools 임포트 실패")
            print("lerobot이 설치되었는지 확인하세요 (conda activate roarm)")
            shutil.rmtree(subset_dir)
            raise

    # 4. 메타데이터 기록
    meta = {
        "source": str(source_dir),
        "n_episodes": n_episodes,
        "seed": seed,
        "strategy": strategy,
        "kept_episodes": keep,
        "removed_episodes": remove,
        "created": datetime.now().isoformat(),
    }
    meta_path = EXPERIMENT_DIR / subset_name / "subset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"서브셋 생성 완료: {subset_dir}")
    # 검증
    actual = get_total_episodes(subset_dir)
    print(f"검증: {actual} 에피소드 (expected {n_episodes})")
    assert actual == n_episodes, f"에피소드 수 불일치: {actual} != {n_episodes}"

    return subset_dir


def run_training(subset_name: str, max_steps: int = MAX_STEPS,
                 resume: bool = True) -> Path:
    """
    특정 서브셋에 대해 학습 실행.

    Args:
        subset_name: 서브셋 이름 (e.g., 'ep010_seed0')
        max_steps: 최대 학습 스텝
        resume: 기존 체크포인트에서 이어서 학습

    Returns:
        학습 출력 디렉토리
    """
    subset_dir = EXPERIMENT_DIR / subset_name
    dataset_dir = subset_dir / "dataset"
    output_dir = subset_dir / "training"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"데이터셋 없음: {dataset_dir}. 먼저 prepare 실행하세요.")

    # 에피소드 수에 비례한 스케줄러 설정
    n_episodes = get_total_episodes(dataset_dir)
    warmup = min(WARMUP_STEPS, max_steps // 10)

    print(f"\n{'='*60}")
    print(f"학습 시작: {subset_name}")
    print(f"  에피소드: {n_episodes}")
    print(f"  스텝: {max_steps:,}")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  출력: {output_dir}")
    print(f"{'='*60}")

    # 이어서 학습 체크
    last_ckpt = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"

    cmd = [
        sys.executable, "-c",
        _build_train_script(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            max_steps=max_steps,
            warmup=warmup,
            resume_path=str(last_ckpt) if (resume and last_ckpt.exists()) else None,
        ),
    ]

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"ERROR: 학습 실패 (return code {result.returncode})")
        return output_dir

    print(f"학습 완료: {output_dir}")
    return output_dir


def _build_train_script(dataset_dir: str, output_dir: str,
                        max_steps: int, warmup: int,
                        resume_path: str = None) -> str:
    """lerobot-train 호출 스크립트 생성."""
    if resume_path:
        return f"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.argv = [
    "lerobot-train",
    "--config_path={resume_path}",
    "--resume=true",
]
from lerobot.scripts.lerobot_train import main
main()
"""
    return f"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.argv = [
    "lerobot-train",
    "--policy.type=smolvla",
    "--policy.pretrained_path={PRETRAINED}",
    "--policy.push_to_hub=false",
    "--dataset.repo_id={REPO_ID}",
    "--dataset.root={dataset_dir}",
    "--batch_size={BATCH_SIZE}",
    "--steps={max_steps}",
    "--eval_freq={EVAL_FREQ}",
    "--save_freq={SAVE_FREQ}",
    "--log_freq={LOG_FREQ}",
    "--output_dir={output_dir}",
    "--num_workers={NUM_WORKERS}",
    "--policy.device=cuda",
    "--policy.scheduler_warmup_steps={warmup}",
    "--policy.scheduler_decay_steps={max_steps}",
    "--policy.scheduler_decay_lr={DECAY_LR}",
]
from lerobot.scripts.lerobot_train import main
main()
"""


def eval_checkpoints(subset_name: str) -> list[dict]:
    """
    서브셋의 모든 체크포인트를 오프라인 평가.

    Returns:
        list of {subset, checkpoint, step, l2_mean, l2_std, diversity, ...}
    """
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    subset_dir = EXPERIMENT_DIR / subset_name
    dataset_dir = subset_dir / "dataset"
    training_dir = subset_dir / "training"
    checkpoints_dir = training_dir / "checkpoints"

    if not checkpoints_dir.exists():
        print(f"체크포인트 없음: {checkpoints_dir}")
        return []

    # 데이터셋 로드
    dataset = LeRobotDataset(repo_id=REPO_ID, root=Path(dataset_dir))
    n_episodes = dataset.num_episodes
    total_frames = len(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 체크포인트 목록 (숫자 폴더만)
    ckpt_dirs = sorted([
        d for d in checkpoints_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ], key=lambda d: int(d.name))

    if not ckpt_dirs:
        print("숫자 체크포인트 폴더 없음")
        return []

    print(f"\n{'='*60}")
    print(f"오프라인 평가: {subset_name}")
    print(f"  에피소드: {n_episodes}, 프레임: {total_frames}")
    print(f"  체크포인트: {len(ckpt_dirs)}개")
    print(f"{'='*60}")

    # 평가용 프레임 인덱스 (균등 간격 샘플링)
    eval_indices = np.linspace(0, total_frames - 1, EVAL_NUM_SAMPLES, dtype=int)

    results = []

    for ckpt_dir in ckpt_dirs:
        model_dir = ckpt_dir / "pretrained_model"
        if not model_dir.exists():
            continue

        step = int(ckpt_dir.name)
        print(f"\n  Checkpoint {step:,}...")

        try:
            policy = SmolVLAPolicy.from_pretrained(str(model_dir))
            policy.to(device)
            policy.eval()

            # Unnormalization stats
            post_stats_path = list(model_dir.glob("*unnormalizer*safetensors"))
            if not post_stats_path:
                print(f"    SKIP: unnormalizer 없음")
                continue
            post_stats = load_file(str(post_stats_path[0]))
            action_mean = post_stats["action.mean"].to(device)
            action_std = post_stats["action.std"].to(device)

            # 추론
            processor = policy.model.vlm_with_expert.processor
            tokenizer = processor.tokenizer

            task_text = "Pick up the sponge\n"
            task_tokens = tokenizer(
                task_text, return_tensors="pt", padding=False, truncation=True
            )
            task_ids = task_tokens["input_ids"].to(device)
            task_mask = task_tokens["attention_mask"].to(device)

            all_preds = []
            all_targets = []

            with torch.no_grad():
                for idx in eval_indices:
                    sample = dataset[int(idx)]

                    # 이미지 준비
                    img = sample["observation.images.top"]
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    img = img.to(device)

                    # 상태 준비
                    state = sample["observation.state"].unsqueeze(0).to(device)

                    # 추론
                    obs = {
                        "observation.images.top": img,
                        "observation.state": state,
                        "task_tokens.input_ids": task_ids,
                        "task_tokens.attention_mask": task_mask,
                    }
                    raw_action = policy.select_action(obs)

                    # Unnormalize
                    pred = raw_action[0, :6].cpu().numpy()
                    target = sample["action"][:6].numpy()

                    all_preds.append(pred)
                    all_targets.append(target)

            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)

            # 메트릭 계산
            l2_errors = np.linalg.norm(all_preds - all_targets, axis=1)
            pred_std = np.std(all_preds, axis=0)

            result = {
                "subset": subset_name,
                "n_episodes": n_episodes,
                "step": step,
                "l2_mean": float(np.mean(l2_errors)),
                "l2_std": float(np.std(l2_errors)),
                "l2_median": float(np.median(l2_errors)),
                "diversity_mean": float(np.mean(pred_std)),
                "diversity_per_joint": [float(s) for s in pred_std],
                "pred_range": [float(np.ptp(all_preds[:, j])) for j in range(6)],
            }
            results.append(result)

            print(f"    L2: {result['l2_mean']:.3f}° ± {result['l2_std']:.3f}°")
            print(f"    Diversity: {result['diversity_mean']:.3f}°")

            # GPU 메모리 해제
            del policy
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def save_results(all_results: list[dict]):
    """전체 결과를 CSV로 저장."""
    csv_path = EXPERIMENT_DIR / "results.csv"

    if not all_results:
        print("결과 없음")
        return

    fieldnames = [
        "subset", "n_episodes", "step",
        "l2_mean", "l2_std", "l2_median",
        "diversity_mean",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"\n결과 저장: {csv_path}")

    # JSON으로도 저장 (per-joint 정보 포함)
    json_path = EXPERIMENT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"상세 결과: {json_path}")


def print_summary():
    """결과 요약 출력."""
    csv_path = EXPERIMENT_DIR / "results.csv"
    if not csv_path.exists():
        print("결과 파일 없음. eval-all 먼저 실행하세요.")
        return

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("결과 비어있음")
        return

    print(f"\n{'='*70}")
    print("Scaling Law Results Summary")
    print(f"{'='*70}")
    print(f"{'Episodes':>10} {'Step':>8} {'L2 Mean':>10} {'L2 Std':>10} {'Diversity':>10}")
    print(f"{'-'*10:>10} {'-'*8:>8} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10}")

    for r in sorted(rows, key=lambda x: (int(x["n_episodes"]), int(x["step"]))):
        print(f"{r['n_episodes']:>10} {int(r['step']):>8,} "
              f"{float(r['l2_mean']):>10.3f} {float(r['l2_std']):>10.3f} "
              f"{float(r['diversity_mean']):>10.3f}")

    # 에피소드 수별 최적 체크포인트
    print(f"\n{'='*70}")
    print("Best Checkpoint per Episode Count (lowest L2)")
    print(f"{'='*70}")

    from collections import defaultdict
    by_eps = defaultdict(list)
    for r in rows:
        by_eps[int(r["n_episodes"])].append(r)

    for eps in sorted(by_eps.keys()):
        best = min(by_eps[eps], key=lambda x: float(x["l2_mean"]))
        print(f"  {eps:>3} episodes → step {int(best['step']):>7,}, "
              f"L2={float(best['l2_mean']):.3f}°, "
              f"diversity={float(best['diversity_mean']):.3f}°")


def main():
    parser = argparse.ArgumentParser(
        description="Scaling Law Experiment Matrix for CoRL 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment_matrix.py prepare --source lerobot_dataset_v3 --episodes 10,25,50
  python experiment_matrix.py train --subset ep010_seed0
  python experiment_matrix.py train-all
  python experiment_matrix.py eval --subset ep010_seed0
  python experiment_matrix.py eval-all
  python experiment_matrix.py summary
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    p_prepare = subparsers.add_parser("prepare", help="서브셋 데이터셋 생성")
    p_prepare.add_argument("--source", required=True, help="원본 데이터셋 경로")
    p_prepare.add_argument("--episodes", default=",".join(map(str, DEFAULT_EPISODE_COUNTS)),
                           help="에피소드 수 (콤마 구분)")
    p_prepare.add_argument("--seed", type=int, default=0, help="랜덤 시드")
    p_prepare.add_argument("--strategy", default="random",
                           choices=["random", "first", "stratified"],
                           help="에피소드 선택 전략")

    # train
    p_train = subparsers.add_parser("train", help="특정 서브셋 학습")
    p_train.add_argument("--subset", required=True, help="서브셋 이름 (e.g., ep010_seed0)")
    p_train.add_argument("--steps", type=int, default=MAX_STEPS, help="최대 학습 스텝")
    p_train.add_argument("--no-resume", action="store_true", help="이어서 학습 안 함")

    # train-all
    p_train_all = subparsers.add_parser("train-all", help="전체 서브셋 순차 학습")
    p_train_all.add_argument("--steps", type=int, default=MAX_STEPS, help="최대 학습 스텝")

    # eval
    p_eval = subparsers.add_parser("eval", help="특정 서브셋 평가")
    p_eval.add_argument("--subset", required=True, help="서브셋 이름")

    # eval-all
    subparsers.add_parser("eval-all", help="전체 서브셋 평가")

    # summary
    subparsers.add_parser("summary", help="결과 요약")

    args = parser.parse_args()

    if args.command == "prepare":
        episode_counts = [int(x.strip()) for x in args.episodes.split(",")]
        for n in episode_counts:
            prepare_subset(
                source_dir=Path(args.source),
                n_episodes=n,
                seed=args.seed,
                strategy=args.strategy,
            )

    elif args.command == "train":
        run_training(
            subset_name=args.subset,
            max_steps=args.steps,
            resume=not args.no_resume,
        )

    elif args.command == "train-all":
        subsets = sorted([
            d.name for d in EXPERIMENT_DIR.iterdir()
            if d.is_dir() and (d / "dataset").exists()
        ])
        if not subsets:
            print("서브셋 없음. prepare 먼저 실행하세요.")
            return
        print(f"학습 대상: {subsets}")
        for name in subsets:
            run_training(name, max_steps=args.steps)

    elif args.command == "eval":
        results = eval_checkpoints(args.subset)
        save_results(results)

    elif args.command == "eval-all":
        subsets = sorted([
            d.name for d in EXPERIMENT_DIR.iterdir()
            if d.is_dir() and (d / "training" / "checkpoints").exists()
        ])
        if not subsets:
            print("평가할 서브셋 없음. train 먼저 실행하세요.")
            return
        all_results = []
        for name in subsets:
            results = eval_checkpoints(name)
            all_results.extend(results)
        save_results(all_results)

    elif args.command == "summary":
        print_summary()


if __name__ == "__main__":
    main()
