"""
experiment_gemini_oracle_design.py

C1 (Experiment Design Specialist) — CoRL 2026
Contribution 2 extension: VLM filtering comparison ablation.

Research question:
    Does VLM-based demo quality filtering improve OOD VLA adaptation
    efficiency vs. rule-based metrics, and does model scale (frontier
    cloud VLM vs. local small VLM) matter?

Experiment ID: VLM_FILTER_ABLATION
Owner: C1 (design), pipeline-agent (training runs), deploy-agent (trials)

NOTE: This script designs and logs the experiment. It does NOT run
      training or deployment (those are pipeline-agent and deploy-agent
      responsibilities). It also generates the VLM judge prompts and
      filter logic for Conditions C and D.

Usage:
    # Generate filter assignments for all conditions given a source dataset
    python experiment_gemini_oracle_design.py filter \
        --source lerobot_dataset_v3 \
        --episodes 200 \
        --output experiments/vlm_filter_ablation/

    # Run Gemini 2.5 Pro filtering (Condition D) — requires GEMINI_API_KEY
    python experiment_gemini_oracle_design.py judge-gemini \
        --source lerobot_dataset_v3 \
        --output experiments/vlm_filter_ablation/gemini_judgments.json

    # Run Qwen2.5-VL 3B filtering (Condition C) — local GPU
    python experiment_gemini_oracle_design.py judge-qwen \
        --source lerobot_dataset_v3 \
        --output experiments/vlm_filter_ablation/qwen_judgments.json

    # Equalize acceptance rates across conditions
    python experiment_gemini_oracle_design.py equalize \
        --judgments-dir experiments/vlm_filter_ablation/ \
        --target-n 80

    # Print experiment specification
    python experiment_gemini_oracle_design.py spec
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# ---- Experiment constants -----------------------------------------------

EXPERIMENT_ID = "VLM_FILTER_ABLATION"
SOURCE_EPISODES = 200          # Required source pool size
TRAIN_STEPS = 100_000
BATCH_SIZE = 64
BASE_MODEL = "lerobot/smolvla_base"
RANDOM_SEED = 42
EVAL_TRIALS_PER_CONDITION = 30  # floor; 50 recommended for primary claims
EVAL_ZONES = 5                  # 5-zone grid from DATA_COLLECTION_STRATEGY.md
EVAL_TRIALS_PER_ZONE = 6        # 30 total (6 × 5)
TASK_TEXT = "Pick up the white sponge.\n"

# Conditions
CONDITIONS = {
    "A_no_filter":    "All 200 demos used as-is.",
    "B_rule_filter":  "FK-depth + gripper_phase + static_frame (existing system).",
    "C_local_vlm":    "Qwen2.5-VL 3B on RTX 4090. Local inference.",
    "D_frontier_vlm": "Gemini 2.5 Pro API. Cloud inference.",
}

# Statistical design
# N=30 per condition: chi-square test for independence (condition x success/fail)
# Primary comparison: A vs D (max possible effect)
# Also report: A vs B, A vs C, C vs D (scale comparison)
# Significance threshold: p < 0.05, one-sided
# Effect size: Cohen's h
# Ground truth: human quality labels (1-5) collected during data collection sessions


# ---- Experiment specification printer -----------------------------------

def print_spec():
    spec = f"""
{'='*70}
EXPERIMENT SPECIFICATION: {EXPERIMENT_ID}
{'='*70}

PURPOSE
    Contribution 2 extension for CoRL 2026 paper.
    Quantifies the marginal value of VLM-based demo quality filtering
    and tests whether frontier model scale (cloud vs. local) matters.

VARIABLES
    Independent : filtering_method (A / B / C / D)
    Dependent   : success_rate on sponge_pick task
    Control     : source episode pool (same 200 episodes for all conditions)
                  training steps ({TRAIN_STEPS})
                  model checkpoint ({BASE_MODEL})
                  random seed ({RANDOM_SEED})
                  hardware (same RoArm-M3 unit)
                  camera position (must not move between conditions)
                  object positions (5-zone grid, fixed)

CONDITIONS
    A (no_filter)    : {CONDITIONS['A_no_filter']}
    B (rule_filter)  : {CONDITIONS['B_rule_filter']}
    C (local_vlm)    : {CONDITIONS['C_local_vlm']}
    D (frontier_vlm) : {CONDITIONS['D_frontier_vlm']}

CONFOUND CONTROLS
    1. Filter acceptance rate MUST be equalized.
       If B keeps 150 and D keeps 80, train all on min(150, 80)=80
       subsampled uniformly from each accepted set.
    2. Gemini API non-determinism: run each demo through API twice.
       Report agreement rate. If < 80%, filter is unreliable (that IS a finding).
    3. Human quality labels: demonstrator rates each episode 1-5 after
       collection. Use as ground truth for filter precision/recall.
    4. Training loss at {TRAIN_STEPS} steps should be similar across conditions.
       Large divergence (>20%) = confound, report and investigate.

EVALUATION PROTOCOL
    - {EVAL_TRIALS_PER_CONDITION} trials per condition
    - {EVAL_ZONES} zones × {EVAL_TRIALS_PER_ZONE} trials each
    - Open-loop 4-chunk deployment (validated: 5/5 sponge pick)
    - Flags: --open-loop --n-chunks 4 --start-pos init
    - Human binary judgment: success=1, failure=0
    - Failure mode classification: gripper_fail / drift / miss / ood / collision

STATISTICAL ANALYSIS
    Primary test  : chi-square for independence (condition × success/fail)
    Effect size   : Cohen's h
    Significance  : p < 0.05, one-sided (H1: filtered > unfiltered)
    Secondary     : pairwise two-proportion z-test (A vs B, A vs C, A vs D, C vs D)
    Ground truth  : human quality labels for filter precision/recall
    Power check   : N=30, detect 20pp difference with ~65% power
                    N=50 preferred for primary claims (achievable if time allows)

BINOMIAL CI CHECK (N=30)
    p=0.8 : 95% CI approx [0.61, 0.92]
    p=0.6 : 95% CI approx [0.41, 0.77]
    Overlap: partial. N=30 can detect 20pp difference at p<0.05 one-sided
    but not reliably (65% power). N=50 achieves 80% power.
    RECOMMENDATION: run 50 trials for whichever comparison is primary (A vs D).

TIMELINE
    D-56 (after collecting 200 episodes):
        Run four filter conditions → four training subsets (0.5 hr)
    D-56 to D-46:
        Four training runs × {TRAIN_STEPS} steps ≈ 22 hr wall clock (sequential)
    D-46 to D-44:
        {EVAL_TRIALS_PER_CONDITION} trials × 4 conditions = {EVAL_TRIALS_PER_CONDITION * 4} real-robot trials ≈ 4 hr
    D-44:
        Statistical analysis, table and figure for paper

COST ESTIMATE
    Gemini 2.5 Pro API: 200 episodes × 3 frames × ~1000 tokens = 600K tokens
                        × 2 for redundancy = 1.2M tokens
                        Cost: ~$1.50 (negligible)
    Qwen2.5-VL 3B local: ~2 hours on RTX 4090 for 200 episodes (3 frames each)

OUTPUT FILES
    experiments/vlm_filter_ablation/
        gemini_judgments.json      — per-episode VLM judgments (D)
        qwen_judgments.json        — per-episode VLM judgments (C)
        rule_filter_results.json   — per-episode rule-based results (B)
        equalized_sets.json        — final episode lists per condition
        eval_results.csv           — per-trial success/failure log
        summary_table.json         — success rates + CIs per condition

PAPER SECTION
    Section 4.3: "Data Quality Filtering: Rule-Based vs. VLM-Based"
    Narrative: "Having established collection-time quality metrics in
    Section 4.2, we ask whether VLM-based filtering provides additional
    value beyond rule-based checks, and whether frontier model scale
    matters for this task."
    NO new contribution claim required — extends Contribution 2.

WHAT GEMINI CANNOT DO (for clarity)
    - Output joint angles or action tokens (not a robot brain)
    - Real-time inference (<100ms): typical latency is 1-3s
    - Replace lerobot-train pipeline (text output only)
    - Gemini Robotics API: not publicly available as of 2026-03-24
{'='*70}
"""
    print(spec)


# ---- VLM judge prompt ---------------------------------------------------

VLM_JUDGE_PROMPT = """You are evaluating a robotic manipulation demonstration for training quality.

You will see three frames from a single robot arm episode:
- Frame 1: start position (t=0)
- Frame 2: mid-episode, approach phase (t=50%)
- Frame 3: grasp moment (t=75%)

Task being demonstrated: {task}

Evaluate each criterion strictly:
1. downward_motion: Does the arm make a clear downward motion toward the object? (0=no, 1=yes)
2. gripper_behavior: Does the gripper open before contact and close during/after? (0=no, 1=yes)
3. spatial_diversity: Does this demo look substantially different from a standard/average approach? (0=no, 1=yes)
4. no_collision: Are there any signs of unintended contact or robot error? (0=collision detected, 1=clean)
5. overall_keep: Should this demonstration be included in training? (true/false)

Return ONLY valid JSON with no additional text:
{{"downward_motion": 0_or_1, "gripper_behavior": 0_or_1, "spatial_diversity": 0_or_1, "no_collision": 0_or_1, "overall_keep": true_or_false, "reason": "one sentence"}}"""


# ---- Rule-based filter (Condition B) ------------------------------------

def apply_rule_filter(episode_metadata: dict) -> bool:
    """
    Rule-based filter matching the existing data_episode_quality.py criteria.

    Accepts an episode if:
    - FK z-height at grasp < 200mm (not purely SHALLOW)
    - gripper_phase includes at least one OPEN frame before grasp
    - static_frame_fraction < 0.3 (not too much standing still)
    """
    fk_z = episode_metadata.get("fk_z_at_grasp_mm", 999)
    gripper_opens = episode_metadata.get("gripper_phase_opens_before_grasp", False)
    static_fraction = episode_metadata.get("static_frame_fraction", 1.0)

    if fk_z >= 200:
        return False
    if not gripper_opens:
        return False
    if static_fraction >= 0.3:
        return False
    return True


# ---- Equalization logic -------------------------------------------------

def equalize_acceptance_rates(
    condition_sets: dict[str, list[int]],
    target_n: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> dict[str, list[int]]:
    """
    Given per-condition accepted episode lists, subsample all to target_n.
    If target_n is None, use min(len(v) for v in condition_sets.values()).

    This is a critical confound control: if filters accept different numbers
    of episodes, the comparison is between dataset size AND filter quality.
    By equalizing, we isolate filter quality as the only variable.
    """
    import random
    rng = random.Random(seed)

    if target_n is None:
        target_n = min(len(v) for v in condition_sets.values())
        print(f"[equalize] Auto target_n = {target_n} (minimum across conditions)")

    equalized = {}
    for cond, episodes in condition_sets.items():
        if len(episodes) < target_n:
            raise ValueError(
                f"Condition {cond} has only {len(episodes)} episodes "
                f"but target_n={target_n}. Reduce target_n or collect more data."
            )
        sampled = sorted(rng.sample(episodes, target_n))
        equalized[cond] = sampled
        print(f"[equalize] {cond}: {len(episodes)} → {target_n} episodes")

    return equalized


# ---- Gemini judge (Condition D) -----------------------------------------

def judge_with_gemini(
    frames: list,           # list of 3 PIL images
    task: str = TASK_TEXT,
    model: str = "gemini-2.5-pro-latest",
    n_calls: int = 2,       # run twice to measure consistency
) -> dict:
    """
    Submit 3 frames to Gemini 2.5 Pro and return quality judgment.
    Requires GEMINI_API_KEY environment variable.

    n_calls=2 is the confound-control requirement: check self-consistency.
    If both calls return the same overall_keep, use that value.
    If they disagree, label as 'inconsistent' (this IS a research finding).
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a key at https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    prompt = VLM_JUDGE_PROMPT.format(task=task.strip())

    results = []
    for i in range(n_calls):
        try:
            response = model_obj.generate_content([prompt] + frames)
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            results.append(parsed)
            time.sleep(0.5)  # Basic rate limit courtesy
        except json.JSONDecodeError as e:
            results.append({"error": str(e), "raw": response.text})
        except Exception as e:
            results.append({"error": str(e)})

    # Consistency check
    if len(results) == 2:
        if "error" not in results[0] and "error" not in results[1]:
            agree = results[0].get("overall_keep") == results[1].get("overall_keep")
            return {
                "call_1": results[0],
                "call_2": results[1],
                "consistent": agree,
                "final_keep": results[0].get("overall_keep") if agree else None,
                "inconsistent": not agree,
            }

    return {"call_1": results[0], "call_2": results[1] if len(results) > 1 else None}


# ---- Qwen2.5-VL 3B judge (Condition C) ----------------------------------

def judge_with_qwen(
    frames: list,           # list of 3 PIL images as numpy arrays
    task: str = TASK_TEXT,
) -> dict:
    """
    Submit 3 frames to Qwen2.5-VL 3B (local, RTX 4090) and return judgment.

    Model: Qwen/Qwen2.5-VL-3B-Instruct
    VRAM: ~7GB (fits alongside SmolVLA inference if done sequentially)

    Note: This function loads the model fresh each call. For batch processing
    200 episodes, use a batch_judge_qwen() wrapper that loads once and loops.
    """
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError(
            "Install Qwen2.5-VL dependencies: "
            "pip install transformers qwen-vl-utils"
        )

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frames[0]},
                {"type": "image", "image": frames[1]},
                {"type": "image", "image": frames[2]},
                {"type": "text", "text": VLM_JUDGE_PROMPT.format(task=task.strip())},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=200)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    try:
        parsed = json.loads(output_text.strip())
        return {"result": parsed, "model": model_name, "consistent": True}
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": output_text, "model": model_name}


# ---- Evaluation results logger ------------------------------------------

@dataclass
class TrialResult:
    condition: str          # A/B/C/D label
    trial_idx: int
    zone: str               # e.g. CENTER, LEFT_FAR
    success: int            # 1 or 0
    failure_mode: str       # gripper_fail / drift / miss / ood / collision / none
    checkpoint_steps: int
    timestamp: str


def log_trial_result(result: TrialResult, output_path: Path):
    """Append a trial result to the CSV log."""
    import csv
    from datetime import datetime

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


# ---- Main CLI -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=f"Experiment design tool for {EXPERIMENT_ID}"
    )
    subparsers = parser.add_subparsers(dest="command")

    # spec
    subparsers.add_parser("spec", help="Print full experiment specification")

    # filter (shows filter plan without executing training)
    p_filter = subparsers.add_parser(
        "filter", help="Apply all four filter conditions to source dataset"
    )
    p_filter.add_argument("--source", required=True)
    p_filter.add_argument("--episodes", type=int, default=SOURCE_EPISODES)
    p_filter.add_argument("--output", required=True)

    # judge-gemini
    p_gemini = subparsers.add_parser(
        "judge-gemini", help="Run Gemini 2.5 Pro filtering (Condition D)"
    )
    p_gemini.add_argument("--source", required=True)
    p_gemini.add_argument("--output", required=True)

    # judge-qwen
    p_qwen = subparsers.add_parser(
        "judge-qwen", help="Run Qwen2.5-VL 3B filtering (Condition C)"
    )
    p_qwen.add_argument("--source", required=True)
    p_qwen.add_argument("--output", required=True)

    # equalize
    p_eq = subparsers.add_parser(
        "equalize",
        help="Equalize acceptance rates across conditions (critical confound control)"
    )
    p_eq.add_argument("--judgments-dir", required=True)
    p_eq.add_argument("--target-n", type=int, default=None)
    p_eq.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "spec" or args.command is None:
        print_spec()
        return

    if args.command == "filter":
        print(f"[C1] Filter plan for {EXPERIMENT_ID}")
        print(f"Source: {args.source}, expected {args.episodes} episodes")
        print()
        for k, v in CONDITIONS.items():
            print(f"  {k}: {v}")
        print()
        print("Next steps:")
        print("  1. Run judge-gemini and judge-qwen to generate VLM judgments")
        print("  2. Run equalize to create balanced training sets")
        print("  3. Pass equalized sets to pipeline-agent for 4 training runs")
        print(f"  4. Each run: steps={TRAIN_STEPS}, batch={BATCH_SIZE}, seed={RANDOM_SEED}")
        return

    if args.command == "equalize":
        judgments_dir = Path(args.judgments_dir)
        gemini_path = judgments_dir / "gemini_judgments.json"
        qwen_path = judgments_dir / "qwen_judgments.json"
        rule_path = judgments_dir / "rule_filter_results.json"

        for p in [gemini_path, qwen_path, rule_path]:
            if not p.exists():
                print(f"[ERROR] Missing: {p}")
                print("Run judge-gemini, judge-qwen, and rule filter first.")
                sys.exit(1)

        with open(gemini_path) as f:
            gemini = json.load(f)
        with open(qwen_path) as f:
            qwen = json.load(f)
        with open(rule_path) as f:
            rule = json.load(f)

        all_episodes = list(range(args.episodes if hasattr(args, "episodes") else SOURCE_EPISODES))

        condition_sets = {
            "A_no_filter": all_episodes,
            "B_rule_filter": [i for i in all_episodes if rule.get(str(i), {}).get("keep", False)],
            "C_local_vlm": [i for i in all_episodes if qwen.get(str(i), {}).get("result", {}).get("overall_keep", False)],
            "D_frontier_vlm": [i for i in all_episodes if gemini.get(str(i), {}).get("final_keep", False)],
        }

        print("[C1] Pre-equalization acceptance counts:")
        for k, v in condition_sets.items():
            print(f"  {k}: {len(v)} / {len(all_episodes)} episodes ({100*len(v)//len(all_episodes)}%)")

        equalized = equalize_acceptance_rates(condition_sets, target_n=args.target_n)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(equalized, f, indent=2)
        print(f"\n[C1] Equalized sets written to {output_path}")
        print("\nPass these to pipeline-agent as training subsets.")
        print(f"Training spec: steps={TRAIN_STEPS}, batch={BATCH_SIZE}, seed={RANDOM_SEED}, base={BASE_MODEL}")
        return

    print(f"Unknown command: {args.command}")
    parser.print_help()


if __name__ == "__main__":
    main()
