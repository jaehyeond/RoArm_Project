"""
Deployment Evaluation for CoRL 2026

체크포인트별 실제 로봇 배포 성공률 측정.
deploy_smolvla.py를 활용하여 반복 실험 + 사람 판정 기록.

실험 구조:
  eval_results/
    ep074_50K/               # 74 episodes, 50K checkpoint
      trial_001.csv          # deploy_smolvla.py의 CSV 로그
      trial_001.json         # 사람 판정 + 메타데이터
      trial_002.csv
      ...
    ep074_100K/
    summary.csv              # 전체 요약

사용법:
    # 1. 단일 체크포인트 평가 (5회 반복)
    python eval_deployment.py run \
        --checkpoint experiments/ep074_seed0/training/checkpoints/050000/pretrained_model \
        --trials 5 --task "Pick up the sponge"

    # 2. experiment_matrix 전체 체크포인트 평가
    python eval_deployment.py run-matrix --steps 50000,100000,200000 --trials 5

    # 3. 결과 요약
    python eval_deployment.py summary

    # 4. 특정 trial에 사람 판정 추가
    python eval_deployment.py judge --eval-dir eval_results/ep074_50K --trial 1 --success true --note "grabbed sponge"
"""

import argparse
import json
import os
import subprocess
import sys
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict


EVAL_DIR = Path("eval_results")

# deploy_smolvla.py 기본 설정 (v3 성공 설정 기반)
DEFAULT_DEPLOY_ARGS = [
    "--open-loop",
    "--n-chunks", "4",
    "--start-pos", "init",
    "--max-steps", "300",
]

EXPERIMENT_DIR = Path("experiments")


def get_eval_name(checkpoint_path: str) -> str:
    """체크포인트 경로에서 평가 이름 생성.

    Examples:
        experiments/ep074_seed0/training/checkpoints/050000/pretrained_model
        → ep074_seed0_050000

        outputs/smolvla_v3_sponge/checkpoints/025000/pretrained_model
        → v3_sponge_025000
    """
    parts = Path(checkpoint_path).parts
    # checkpoints/NNNNNN/pretrained_model 패턴 찾기
    for i, p in enumerate(parts):
        if p == "checkpoints" and i + 1 < len(parts):
            step = parts[i + 1]
            # 상위 디렉토리에서 이름 추출
            parent_parts = parts[:i]
            if "experiments" in parent_parts:
                exp_idx = parent_parts.index("experiments")
                if exp_idx + 1 < len(parent_parts):
                    return f"{parent_parts[exp_idx + 1]}_{step}"
            elif "outputs" in parent_parts:
                out_idx = parent_parts.index("outputs")
                if out_idx + 1 < len(parent_parts):
                    return f"{parent_parts[out_idx + 1]}_{step}"
            return f"ckpt_{step}"
    return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_trial(checkpoint_path: str, trial_num: int, eval_name: str,
              task: str = "Pick up the sponge",
              port: str = "/dev/ttyUSB1",
              extra_args: list = None) -> Path:
    """
    단일 배포 trial 실행.

    Returns:
        trial JSON 파일 경로
    """
    eval_dir = EVAL_DIR / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"trial_{trial_num:03d}.csv"
    json_path = eval_dir / f"trial_{trial_num:03d}.json"

    if json_path.exists():
        print(f"이미 존재: {json_path}")
        return json_path

    print(f"\n{'='*60}")
    print(f"Trial {trial_num}: {eval_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Task: {task}")
    print(f"{'='*60}")

    # deploy_smolvla.py 실행
    cmd = [
        sys.executable, "deploy_smolvla.py",
        "--checkpoint", checkpoint_path,
        "--task", task,
        "--port", port,
        "--log-csv", str(csv_path),
        *DEFAULT_DEPLOY_ARGS,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  CMD: {' '.join(cmd)}")
    print(f"\n  [배포 시작 — Ctrl+C로 종료 후 판정 입력]")
    print()

    start_time = datetime.now()
    result = subprocess.run(cmd)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 사람 판정 입력
    print(f"\n{'='*60}")
    print(f"Trial {trial_num} 완료 ({duration:.1f}초)")
    print(f"{'='*60}")

    success = input("  성공? (y/n/skip): ").strip().lower()
    if success == "skip":
        success_bool = None
    else:
        success_bool = success in ("y", "yes", "1", "true")

    note = input("  메모 (엔터로 건너뛰기): ").strip()

    # 실패 유형 (실패 시)
    failure_mode = None
    if success_bool is False:
        print("  실패 유형:")
        print("    1. gripper_fail — 그리퍼 미작동")
        print("    2. drift — 한 방향 드리프트")
        print("    3. miss — 물체 빗나감")
        print("    4. ood — OOD 발산")
        print("    5. collision — 충돌")
        print("    6. other — 기타")
        failure_mode = input("  선택 (1-6 또는 직접 입력): ").strip()
        mode_map = {
            "1": "gripper_fail", "2": "drift", "3": "miss",
            "4": "ood", "5": "collision", "6": "other",
        }
        failure_mode = mode_map.get(failure_mode, failure_mode)

    # 메타데이터 저장
    meta = {
        "eval_name": eval_name,
        "trial": trial_num,
        "checkpoint": checkpoint_path,
        "task": task,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "success": success_bool,
        "failure_mode": failure_mode,
        "note": note,
        "deploy_args": DEFAULT_DEPLOY_ARGS + (extra_args or []),
        "csv_log": str(csv_path),
        "return_code": result.returncode,
    }

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    status = "SUCCESS" if success_bool else ("FAIL" if success_bool is False else "SKIPPED")
    print(f"\n  → {status} (saved: {json_path})")

    return json_path


def run_evaluation(checkpoint_path: str, trials: int = 5,
                   task: str = "Pick up the sponge",
                   port: str = "/dev/ttyUSB1",
                   extra_args: list = None):
    """N회 반복 배포 평가."""
    eval_name = get_eval_name(checkpoint_path)

    print(f"\n배포 평가: {eval_name}")
    print(f"  체크포인트: {checkpoint_path}")
    print(f"  반복 횟수: {trials}")
    print(f"  태스크: {task}")

    for trial in range(1, trials + 1):
        input(f"\n  [Enter] 를 눌러 Trial {trial}/{trials} 시작...")
        run_trial(
            checkpoint_path=checkpoint_path,
            trial_num=trial,
            eval_name=eval_name,
            task=task,
            port=port,
            extra_args=extra_args,
        )

    # Trial 완료 후 요약
    print_eval_summary(eval_name)


def run_matrix_evaluation(steps: list[int], trials: int = 5,
                          task: str = "Pick up the sponge",
                          port: str = "/dev/ttyUSB1"):
    """experiment_matrix의 모든 서브셋 × 체크포인트 평가."""
    if not EXPERIMENT_DIR.exists():
        print(f"실험 디렉토리 없음: {EXPERIMENT_DIR}")
        return

    subsets = sorted([
        d.name for d in EXPERIMENT_DIR.iterdir()
        if d.is_dir() and (d / "training" / "checkpoints").exists()
    ])

    if not subsets:
        print("학습된 서브셋 없음")
        return

    print(f"\n평가 대상 서브셋: {subsets}")
    print(f"체크포인트 스텝: {steps}")
    print(f"각 {trials}회 반복")

    for subset in subsets:
        for step in steps:
            ckpt_path = (EXPERIMENT_DIR / subset / "training" /
                         "checkpoints" / f"{step:06d}" / "pretrained_model")
            if not ckpt_path.exists():
                print(f"  SKIP: {ckpt_path} 없음")
                continue

            run_evaluation(
                checkpoint_path=str(ckpt_path),
                trials=trials,
                task=task,
                port=port,
            )


def judge_trial(eval_dir: str, trial: int, success: bool, note: str = ""):
    """기존 trial에 사람 판정 추가/수정."""
    json_path = Path(eval_dir) / f"trial_{trial:03d}.json"
    if not json_path.exists():
        print(f"파일 없음: {json_path}")
        return

    with open(json_path) as f:
        meta = json.load(f)

    meta["success"] = success
    meta["note"] = note
    meta["judged_at"] = datetime.now().isoformat()

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"판정 업데이트: {json_path} → {'SUCCESS' if success else 'FAIL'}")


def print_eval_summary(eval_name: str = None):
    """평가 결과 요약."""
    if not EVAL_DIR.exists():
        print("평가 결과 없음")
        return

    if eval_name:
        eval_dirs = [EVAL_DIR / eval_name]
    else:
        eval_dirs = sorted([d for d in EVAL_DIR.iterdir() if d.is_dir()])

    if not eval_dirs:
        print("평가 디렉토리 없음")
        return

    print(f"\n{'='*70}")
    print("Deployment Evaluation Summary")
    print(f"{'='*70}")
    print(f"{'Eval Name':<30} {'Trials':>7} {'Success':>8} {'Fail':>6} {'Rate':>8}")
    print(f"{'-'*30:<30} {'-'*7:>7} {'-'*8:>8} {'-'*6:>6} {'-'*8:>8}")

    all_rows = []

    for eval_dir in eval_dirs:
        if not eval_dir.is_dir():
            continue

        trials = sorted(eval_dir.glob("trial_*.json"))
        if not trials:
            continue

        successes = 0
        failures = 0
        skipped = 0
        failure_modes = defaultdict(int)

        for trial_path in trials:
            with open(trial_path) as f:
                meta = json.load(f)

            if meta.get("success") is True:
                successes += 1
            elif meta.get("success") is False:
                failures += 1
                fm = meta.get("failure_mode", "unknown")
                failure_modes[fm] += 1
            else:
                skipped += 1

        total = successes + failures
        rate = successes / total * 100 if total > 0 else 0

        print(f"{eval_dir.name:<30} {len(trials):>7} {successes:>8} {failures:>6} {rate:>7.1f}%")

        if failure_modes:
            for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
                print(f"  {'':30} fail: {mode} ({count})")

        all_rows.append({
            "eval_name": eval_dir.name,
            "total_trials": len(trials),
            "successes": successes,
            "failures": failures,
            "skipped": skipped,
            "success_rate": rate,
            "failure_modes": dict(failure_modes),
        })

    # CSV 저장
    if all_rows:
        summary_csv = EVAL_DIR / "summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "eval_name", "total_trials", "successes", "failures",
                "skipped", "success_rate",
            ], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n요약 저장: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Deployment Evaluation for CoRL 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="단일 체크포인트 평가")
    p_run.add_argument("--checkpoint", required=True, help="체크포인트 경로")
    p_run.add_argument("--trials", type=int, default=5, help="반복 횟수")
    p_run.add_argument("--task", default="Pick up the sponge", help="태스크")
    p_run.add_argument("--port", default="/dev/ttyUSB1", help="로봇 포트")

    # run-matrix
    p_matrix = subparsers.add_parser("run-matrix", help="experiment_matrix 전체 평가")
    p_matrix.add_argument("--steps", default="50000,100000,200000",
                          help="평가할 체크포인트 스텝 (콤마 구분)")
    p_matrix.add_argument("--trials", type=int, default=5, help="반복 횟수")
    p_matrix.add_argument("--task", default="Pick up the sponge", help="태스크")
    p_matrix.add_argument("--port", default="/dev/ttyUSB1", help="로봇 포트")

    # judge
    p_judge = subparsers.add_parser("judge", help="사람 판정 추가")
    p_judge.add_argument("--eval-dir", required=True, help="평가 디렉토리")
    p_judge.add_argument("--trial", type=int, required=True, help="Trial 번호")
    p_judge.add_argument("--success", type=str, required=True, help="성공 여부 (true/false)")
    p_judge.add_argument("--note", default="", help="메모")

    # summary
    p_summary = subparsers.add_parser("summary", help="결과 요약")
    p_summary.add_argument("--eval-name", default=None, help="특정 평가만 요약")

    args = parser.parse_args()

    if args.command == "run":
        run_evaluation(
            checkpoint_path=args.checkpoint,
            trials=args.trials,
            task=args.task,
            port=args.port,
        )

    elif args.command == "run-matrix":
        steps = [int(s.strip()) for s in args.steps.split(",")]
        run_matrix_evaluation(
            steps=steps,
            trials=args.trials,
            task=args.task,
            port=args.port,
        )

    elif args.command == "judge":
        success_bool = args.success.lower() in ("true", "yes", "1", "y")
        judge_trial(args.eval_dir, args.trial, success_bool, args.note)

    elif args.command == "summary":
        print_eval_summary(args.eval_name)


if __name__ == "__main__":
    main()
