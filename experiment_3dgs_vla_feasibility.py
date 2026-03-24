"""
experiment_3dgs_vla_feasibility.py
C1 Experiment Design Specialist — 2026-03-24

3DGS+VLA CoRL 실험 가능성 평가 및 Go/No-Go Gate 설계.

결론 요약:
- 단독 메인 페이퍼로는 NO-GO (시간 부족, 리스크 과다)
- SigLIP cosine dist gate 통과 시 → AR+Oracle 논문의 ablation으로 제한적 활용
- 통과 실패 시 → negative result appendix로 활용 (3-4일 비용)

사용법:
    python experiment_3dgs_vla_feasibility.py --mode gate_check
    python experiment_3dgs_vla_feasibility.py --mode timeline
    python experiment_3dgs_vla_feasibility.py --mode risk_matrix
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Literal
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. SigLIP Gate — Go/No-Go 기준
# ---------------------------------------------------------------------------

SIGLIP_GATE = {
    "go_threshold": 0.30,       # cosine dist < 0.30: 3DGS transfer 가능 (SplatSim 수준)
    "caution_threshold": 0.50,  # 0.30-0.50: conditional, visual inspection 추가 필요
    "no_go_threshold": 0.50,    # > 0.50: 사실상 OOD, 중단
    "reference_values": {
        "isaac_rasterizer": 0.65,   # 전이 불가 기준선 (A2 agent 검증)
        "3dgs_splattsim": 0.15,     # 전이 가능 기준선 (SplatSim 논문 기준)
        "real_vs_real": 0.05,       # 이상적 상한선
    },
    "note": (
        "SmolVLA SigLIP backbone은 fine-tuning 중 frozen. "
        "real photographs에 사전학습됨. "
        "cosine dist는 SigLIP ViT-B/16 CLS token 기준."
    ),
}


def evaluate_siglip_gate(measured_cosine_dist: float) -> dict:
    """측정된 SigLIP cosine distance로 Go/No-Go 판정."""
    if measured_cosine_dist < SIGLIP_GATE["go_threshold"]:
        verdict = "GO"
        action = "3DGS augmentation 가능. AR+Oracle ablation 실험 설계로 진행."
    elif measured_cosine_dist < SIGLIP_GATE["caution_threshold"]:
        verdict = "CAUTION"
        action = (
            "시각적 품질 검수 추가 필요. "
            "소규모 파일럿 (10ep augmented) 먼저 학습 후 offline L2 확인."
        )
    else:
        verdict = "NO-GO"
        action = (
            "3DGS 품질이 SmolVLA SigLIP threshold 미달. "
            "negative result 데이터로 보존 후 Appendix에 기록. "
            "메인 AR+Oracle 실험으로 집중."
        )

    return {
        "measured_cosine_dist": measured_cosine_dist,
        "verdict": verdict,
        "action": action,
        "reference": SIGLIP_GATE["reference_values"],
    }


# ---------------------------------------------------------------------------
# 2. 타임라인 분석 — 현실적 추정 (×1.5 버퍼 포함)
# ---------------------------------------------------------------------------

@dataclass
class TimelineItem:
    task: str
    optimistic_days: float
    realistic_days: float   # optimistic × 1.5
    risk_factor: str
    blocking_dependency: str | None


TIMELINE_3DGS_STANDALONE = [
    TimelineItem(
        "3DGS 환경 구축 (Nerfstudio/gsplat + Azure Kinect→COLMAP)",
        optimistic_days=2.0,
        realistic_days=3.0,
        risk_factor="Azure Kinect COLMAP 파이프라인 미검증",
        blocking_dependency=None,
    ),
    TimelineItem(
        "Workspace 스캔 + 3DGS 학습 (1 씬)",
        optimistic_days=1.0,
        realistic_days=2.0,
        risk_factor="단일 Azure Kinect = sparse view → 품질 불확실",
        blocking_dependency="3DGS 환경 구축",
    ),
    TimelineItem(
        "SigLIP cosine dist 검증 (Go/No-Go gate)",
        optimistic_days=0.5,
        realistic_days=1.0,
        risk_factor="> 0.5 나오면 여기서 중단. 이하 항목 모두 취소.",
        blocking_dependency="Workspace 스캔 + 3DGS 학습",
    ),
    TimelineItem(
        "Novel view 렌더링 파이프라인 (action label 매핑 포함)",
        optimistic_days=3.0,
        realistic_days=5.0,
        risk_factor="action label을 가상 viewpoint에 맞게 재매핑 코드 없음",
        blocking_dependency="SigLIP gate PASS",
    ),
    TimelineItem(
        "Augmented 데이터셋 구축 + lerobot v3 통합 + stats.json 재계산",
        optimistic_days=1.0,
        realistic_days=2.0,
        risk_factor="lerobot v3 포맷과 3DGS 렌더 통합 미경험",
        blocking_dependency="Novel view 렌더링 파이프라인",
    ),
    TimelineItem(
        "SmolVLA 학습 (real-only vs real+3DGS, 50K steps × 2 conditions)",
        optimistic_days=2.0,
        realistic_days=3.0,
        risk_factor="RTX 4090 기준 ~18h/50K × 2 = 36h. 동시 실행 불가.",
        blocking_dependency="Augmented 데이터셋 구축",
    ),
    TimelineItem(
        "실제 배포 평가 (N=20 trials × 2 conditions)",
        optimistic_days=2.0,
        realistic_days=3.0,
        risk_factor="로봇 재조정, 카메라 고정, 물체 위치 grid 필요",
        blocking_dependency="SmolVLA 학습",
    ),
    TimelineItem(
        "논문 작성 (CoRL 8-page, 관련연구 44편 정리 포함)",
        optimistic_days=14.0,
        realistic_days=21.0,
        risk_factor="3DGS+VLA 선행연구 44편 전부 숙지 필요",
        blocking_dependency="실제 배포 평가",
    ),
]

TIMELINE_3DGS_AS_ABLATION = [
    # AR+Oracle 메인 실험 완료 이후 추가 작업만
    TimelineItem(
        "SigLIP cosine dist 사전 검증 (gate test only)",
        optimistic_days=0.5,
        realistic_days=1.0,
        risk_factor="실패 시 ablation 전체 취소. 비용 최소화.",
        blocking_dependency="AR+Oracle 메인 실험 완료 (4/20 목표)",
    ),
    TimelineItem(
        "3DGS 씬 학습 + 렌더링 파이프라인 (gate 통과 시)",
        optimistic_days=4.0,
        realistic_days=6.0,
        risk_factor="동적 씬 처리 없이 정적 배경만 augmentation",
        blocking_dependency="SigLIP gate PASS",
    ),
    TimelineItem(
        "Augmented condition 학습 (1 추가 condition)",
        optimistic_days=1.5,
        realistic_days=2.0,
        risk_factor="메인 실험과 동일 steps/seed 필수",
        blocking_dependency="3DGS 렌더링 파이프라인",
    ),
    TimelineItem(
        "평가 + 논문 section 작성 (Section 4.4, ~0.5 page)",
        optimistic_days=3.0,
        realistic_days=5.0,
        risk_factor="결과가 negative여도 'attempted' 로 기록 가능",
        blocking_dependency="Augmented condition 학습",
    ),
]


def print_timeline(items: list[TimelineItem], title: str) -> None:
    total_optimistic = sum(i.optimistic_days for i in items)
    total_realistic = sum(i.realistic_days for i in items)

    print(f"\n{'='*70}")
    print(f"TIMELINE: {title}")
    print(f"{'='*70}")
    for i, item in enumerate(items, 1):
        print(f"\n[{i}] {item.task}")
        print(f"    낙관: {item.optimistic_days}일 | 현실적: {item.realistic_days}일")
        print(f"    리스크: {item.risk_factor}")
        if item.blocking_dependency:
            print(f"    의존: {item.blocking_dependency}")
    print(f"\n{'─'*70}")
    print(f"합계 — 낙관: {total_optimistic}일 | 현실적(x1.5): {total_realistic}일")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# 3. 리스크 매트릭스
# ---------------------------------------------------------------------------

RISK_MATRIX = [
    {
        "id": "R1",
        "risk": "Single-view Azure Kinect → 3DGS 품질 미달",
        "probability": "HIGH",
        "impact": "HIGH",
        "severity": "CRITICAL",
        "mitigation": (
            "SigLIP gate로 조기 감지. "
            "로봇 팔을 카메라 보조로 사용해 multi-view 시도 가능하나 "
            "RoArm M3 reach 반경 한계로 coverage 제한."
        ),
        "residual_risk": "HIGH — 단독 해결책 없음",
    },
    {
        "id": "R2",
        "risk": "SigLIP frozen → 렌더링 품질 요구 (threshold 0.3)",
        "probability": "MEDIUM",
        "impact": "HIGH",
        "severity": "HIGH",
        "mitigation": (
            "SpatSim 기준 3DGS는 0.15 달성. "
            "단, SplatSim은 multi-view 사용. "
            "우리 single-view 3DGS가 이 수준에 도달할지 미검증."
        ),
        "residual_risk": "MEDIUM — gate test로 사전 확인 필수",
    },
    {
        "id": "R3",
        "risk": "동적 씬 불가 (로봇 팔 + 물체 이동)",
        "probability": "CERTAIN",
        "impact": "MEDIUM",
        "severity": "HIGH",
        "mitigation": (
            "배경(workspace)만 3DGS로 렌더링. "
            "로봇 팔과 물체는 원본 real 이미지에서 그대로 사용. "
            "→ viewpoint augmentation만 가능, object pose augmentation 불가."
        ),
        "residual_risk": "MEDIUM — contribution 범위가 좁아짐",
    },
    {
        "id": "R4",
        "risk": "GeoPredict (arXiv:2512.16811) 등 선행연구와 차별화 부족",
        "probability": "MEDIUM",
        "impact": "MEDIUM",
        "severity": "MEDIUM",
        "mitigation": (
            "우리 contribution: single-view RGBD (Azure Kinect) → 3DGS → consumer VLA. "
            "GeoPredict는 multi-view + 전용 로봇. "
            "차별화 포인트는 있으나 좁음."
        ),
        "residual_risk": "MEDIUM — reviewer가 'trivial application'으로 볼 가능성",
    },
    {
        "id": "R5",
        "risk": "타임라인 부족 (65일에서 AR+Oracle 메인 작업 후 잔여 20일)",
        "probability": "HIGH",
        "impact": "HIGH",
        "severity": "CRITICAL",
        "mitigation": (
            "3DGS standalone 실험(40일 현실) → NO-GO. "
            "AR+Oracle 논문의 ablation으로만 편입 (15-18일). "
            "단, AR+Oracle이 4월 20일 이전 완료되어야 함."
        ),
        "residual_risk": "HIGH — AR+Oracle 진행 상황에 완전히 종속",
    },
]


def print_risk_matrix() -> None:
    print(f"\n{'='*70}")
    print("RISK MATRIX: 3DGS+VLA CoRL 실험")
    print(f"{'='*70}")
    for r in RISK_MATRIX:
        print(f"\n[{r['id']}] {r['risk']}")
        print(f"    확률: {r['probability']} | 영향: {r['impact']} | 심각도: {r['severity']}")
        print(f"    완화: {r['mitigation']}")
        print(f"    잔여 리스크: {r['residual_risk']}")
    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# 4. 통합 권장 행동 계획
# ---------------------------------------------------------------------------

ACTION_PLAN = {
    "verdict": "NO-GO (단독 메인 페이퍼). CONDITIONAL (AR+Oracle ablation).",
    "confidence": "HIGH",
    "immediate_actions": [
        {
            "action": "SigLIP cosine distance gate test",
            "cost_days": 1.0,
            "deadline": "이번 주 (2026-03-28 이전)",
            "output": "cosine dist 수치 + GO/NO-GO 판정",
            "code_hint": (
                "from transformers import AutoProcessor, AutoModel; import torch\n"
                "# 실제 이미지 vs 3DGS 렌더링 이미지 pair에서 SigLIP CLS token 추출\n"
                "# cosine_dist = 1 - F.cosine_similarity(real_feat, render_feat)"
            ),
        },
    ],
    "if_gate_pass": {
        "action": "AR+Oracle Section 4.4 ablation으로 편입",
        "cost_days_additional": 15,
        "prerequisite": "AR+Oracle 메인 실험 완료 (4월 20일 이전)",
        "structure": (
            "Section 4.4: '3DGS Augmentation as a Complement to AR-Guided Collection'\n"
            "Condition 추가: real+3DGS vs real-only (Condition A)\n"
            "N=20 trials. 동일 평가 프로토콜."
        ),
    },
    "if_gate_fail": {
        "action": "Appendix negative result로 처리",
        "cost_days_additional": 1,
        "output": (
            "Appendix A: 'Failed Attempt: 3DGS Augmentation'\n"
            "SigLIP cosine dist = X (> 0.5). Isaac rasterizer (0.65)와 유사.\n"
            "SmolVLA SigLIP frozen으로 인해 low-fidelity render → transfer 불가.\n"
            "→ 실용적 기여: 다른 연구자에게 동일 실수 방지."
        ),
    },
    "do_not_do": [
        "3DGS를 메인 contribution으로 설정하고 AR+Oracle 방향 변경",
        "SigLIP gate 없이 full 3DGS pipeline 구현 시작",
        "multi-view 카메라 추가 구매 고려 (예산 + 시간 초과)",
        "동적 씬 처리를 위한 4D Gaussian 탐색 (논문 1편 분량 추가 작업)",
    ],
}


def print_action_plan() -> None:
    print(f"\n{'='*70}")
    print("ACTION PLAN")
    print(f"{'='*70}")
    print(f"\n판정: {ACTION_PLAN['verdict']}")
    print(f"확신도: {ACTION_PLAN['confidence']}")

    print(f"\n[즉시 행동]")
    for act in ACTION_PLAN["immediate_actions"]:
        print(f"  - {act['action']}")
        print(f"    비용: {act['cost_days']}일 | 기한: {act['deadline']}")
        print(f"    산출물: {act['output']}")

    print(f"\n[Gate 통과 시]")
    gp = ACTION_PLAN["if_gate_pass"]
    print(f"  - {gp['action']}")
    print(f"    추가 비용: {gp['cost_days_additional']}일")
    print(f"    전제 조건: {gp['prerequisite']}")

    print(f"\n[Gate 실패 시]")
    gf = ACTION_PLAN["if_gate_fail"]
    print(f"  - {gf['action']}")
    print(f"    추가 비용: {gf['cost_days_additional']}일")

    print(f"\n[하지 말아야 할 것]")
    for item in ACTION_PLAN["do_not_do"]:
        print(f"  - {item}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# 5. 통계 검정력 메모
# ---------------------------------------------------------------------------

STATISTICAL_NOTES = """
N=20 trials 통계 검정력 (3DGS 실험):

Binomial CI (95%) 예시:
- p=0.8, N=20 → CI [0.56, 0.94]
- p=0.9, N=20 → CI [0.68, 0.99]
→ 겹침 발생. 80% vs 90%를 구별하려면 N=50 필요.

실용적 권고:
- 3DGS ablation이 Section 4.4로 편입된다면 N=20으로도 방향성 확인 가능.
  (full statistical significance 주장 불필요. "preliminary evidence" 수준.)
- 단, 주 실험 (AR vs baseline)은 N=50 필요.

McNemar's test (paired, 동일 물체 위치에서 2 conditions 비교):
- 통계 검정력 더 높음. 가능하면 paired 평가 설계 권장.
"""


# ---------------------------------------------------------------------------
# 6. 결합 실험 설계 (AR+Oracle+3DGS 통합 구조)
# ---------------------------------------------------------------------------

COMBINED_EXPERIMENT_DESIGN = {
    "title": "AR-Guided Demo Collection with Real-Time Quality Filtering for Consumer Robot VLA Training",
    "main_conditions": {
        "A": "50ep baseline (no guidance, no oracle, no augmentation)",
        "B": "50ep + offline augmentation only (GenAug-style, same real data as A)",
        "C": "50ep with AR spatial guidance + real-time oracle filtering",
        "D_optional": "Same as C + 3DGS augmentation (gate pass 필요)",
    },
    "independent_variables": {
        "primary": "data collection strategy (A vs C)",
        "secondary": "augmentation type (none vs offline vs 3DGS)",
    },
    "dependent_variables": {
        "primary": "success rate @ N=50 trials",
        "secondary": [
            "workspace coverage (xy heatmap entropy)",
            "gripper open-close phase ratio",
            "static frame ratio",
            "SigLIP cosine dist (3DGS 사용 시)",
        ],
    },
    "control_variables": [
        "total episodes = 50 per condition",
        "training steps = 50K",
        "same base checkpoint (lerobot/smolvla_base)",
        "same hardware (RTX 4090, Azure Kinect)",
        "same object positions (5-position grid, predefined)",
        "same random seed (42)",
        "same evaluation protocol (N=50 trials, 5 positions × 10 trials)",
    ],
    "statistical_design": {
        "primary_test": "binomial CI (95%) on success rate",
        "comparison_test": "McNemar's test (paired conditions)",
        "minimum_N": 50,
        "power": "80% power to detect 15pp difference (A=65%, C=80%)",
    },
    "3dgs_integration_note": (
        "3DGS condition D는 gate test 결과에 따라 포함/제외 결정. "
        "포함 시 N=20으로 preliminary. "
        "제외 시 Appendix negative result."
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3DGS+VLA CoRL 실험 가능성 평가")
    parser.add_argument(
        "--mode",
        choices=["gate_check", "timeline", "risk_matrix", "action_plan", "full"],
        default="full",
        help="출력 모드",
    )
    parser.add_argument(
        "--cosine_dist",
        type=float,
        default=None,
        help="측정된 SigLIP cosine distance (gate_check 모드에서 사용)",
    )
    args = parser.parse_args()

    if args.mode == "gate_check":
        if args.cosine_dist is None:
            print("--cosine_dist 값 필요. 예: --cosine_dist 0.25")
            print(f"\nSigLIP Gate 기준:")
            for k, v in SIGLIP_GATE.items():
                print(f"  {k}: {v}")
        else:
            result = evaluate_siglip_gate(args.cosine_dist)
            print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode == "timeline":
        print_timeline(TIMELINE_3DGS_STANDALONE, "3DGS Standalone (메인 페이퍼)")
        print_timeline(TIMELINE_3DGS_AS_ABLATION, "3DGS as Ablation (AR+Oracle Section 4.4)")

        remaining_budget = 65  # days
        main_experiment_cost = 45  # AR+Oracle 메인 실험 추정
        remaining_after_main = remaining_budget - main_experiment_cost
        ablation_realistic = sum(i.realistic_days for i in TIMELINE_3DGS_AS_ABLATION)
        print(f"\n예산 분석:")
        print(f"  전체 예산: {remaining_budget}일")
        print(f"  AR+Oracle 메인 실험: ~{main_experiment_cost}일")
        print(f"  잔여 예산: {remaining_after_main}일")
        print(f"  3DGS ablation 현실적 비용: {ablation_realistic}일")
        verdict = "가능 (단, AR+Oracle 4/20 완료 전제)" if ablation_realistic <= remaining_after_main else "불가 (시간 초과)"
        print(f"  판정: {verdict}")

    elif args.mode == "risk_matrix":
        print_risk_matrix()

    elif args.mode == "action_plan":
        print_action_plan()

    elif args.mode == "full":
        print_timeline(TIMELINE_3DGS_STANDALONE, "3DGS Standalone (메인 페이퍼)")
        print_timeline(TIMELINE_3DGS_AS_ABLATION, "3DGS as Ablation (AR+Oracle Section 4.4)")
        print_risk_matrix()
        print_action_plan()

        print(f"\n{'='*70}")
        print("COMBINED EXPERIMENT DESIGN (AR+Oracle+3DGS 통합 구조)")
        print(f"{'='*70}")
        print(json.dumps(COMBINED_EXPERIMENT_DESIGN, ensure_ascii=False, indent=2))
        print(f"\n{STATISTICAL_NOTES}")


if __name__ == "__main__":
    main()
