#!/usr/bin/env python3
"""오라클 테스트 — 예측이 가져갈 수 있는 **상한**은 얼마인가?

교수님 지적(2026-09-01): "매번 다시 보고 제일 높은 곳을 퍼면 학습이 왜 필요하냐."
P40 에서 greedy_high 는 실패를 넣어도 **유일하게 완주**했다(21 스쿱, 실패율 33.3%).
그러나 P40 이 비교한 것은 greedy 대 **다른 규칙 3개**였고, 프로포절이 제안하는
예측 정책은 **한 번도 돌려진 적이 없다.** 이 스크립트가 그것을 돌린다.

학습이 아니라 **커닝**이다 — 정책에게 결과를 미리 보여준다.
따라서 결과는 학습이 도달할 수 있는 **상한**이고, 이보다 잘할 수는 없다.

정책 3종을 같은 조건에서 비교한다:
  greedy_high  현재 높이 최대            (교수님 안 / 기준선)
  oracle_1     **양만** 예측              (프로포절 방법①)
               -> 실패하지 않으면서 이번에 가장 많이 담기는 곳
  oracle_2     **양 + 퍼낸 뒤 형상** 예측  (프로포절 방법②)
               -> 위에 더해, 퍼낸 뒤에도 팔 곳이 많이 남는 곳을 고른다

🔴 읽는 법:
  oracle_1 ≈ greedy_high 이면  "양 예측"에 이득이 없다
  oracle_2 ≫ oracle_1  이면    **형상 예측이 이득의 원천** = 프로포절의 핵심 주장
  둘 다 ≈ greedy_high 이면      교수님이 옳고 프레임을 바꿔야 한다
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p39_decision_loop as P39  # noqa: E402
import p40_failure_greedy_reexam as P40  # noqa: E402


def _candidates(observation, threshold_m: float, top_k: int) -> list[int]:
    """평가할 후보 셀. 전수는 비싸므로 높이 상위 top_k 만 본다.

    ⚠️ 이 절단이 오라클을 약하게 만든다 — 즉 아래 결과는 **상한의 하한**이다.
    """
    flat = P39._active_cells(observation, threshold_m)
    if flat.size == 0:
        return []
    h = observation.height.reshape(-1)[flat]
    order = np.argsort(h)[::-1][:top_k]
    return [int(flat[i]) for i in order]


class OraclePolicy(P39.ScoopPolicy):
    """실행기를 복제해 각 후보를 미리 돌려 보고 고른다."""

    def __init__(self, executor, threshold_m: float, *, use_shape: bool,
                 top_k: int = 24, shape_weight: float = 1.0) -> None:
        self.name = "oracle_2" if use_shape else "oracle_1"
        self.executor = executor
        self.threshold_m = float(threshold_m)
        self.use_shape = bool(use_shape)
        self.top_k = int(top_k)
        self.shape_weight = float(shape_weight)

    def select_action(self, observation, context):
        del context
        cands = _candidates(observation, self.threshold_m, self.top_k)
        if not cands:
            raise RuntimeError("no actionable height cell")

        best, best_score = None, -np.inf
        for flat in cands:
            action = P39._cell_action(
                observation, flat, P39._toward_grid_center(observation, flat))
            trial = copy.deepcopy(self.executor)
            trial.failure_log = []
            result = trial.execute(action)
            if result.failed:
                continue                       # 실패하는 곳은 애초에 안 고른다
            score = float(result.scooped_mass_kg)

            if self.use_shape:
                # 🔴 방법②: 퍼낸 뒤 형상을 보고, **다음에 팔 수 있는 곳이
                #    얼마나 남는지**로 가점한다. 절벽을 만드는 행동에 벌점이 된다.
                after = trial.observe()
                nxt = _candidates(after, self.threshold_m, self.top_k)
                viable = 0
                for f2 in nxt:
                    a2 = P39._cell_action(
                        after, f2, P39._toward_grid_center(after, f2))
                    if not trial._would_fail(a2):
                        viable += 1
                score += self.shape_weight * float(result.scooped_mass_kg) * (
                    viable / max(len(nxt), 1))

            if score > best_score:
                best, best_score = action, score

        if best is None:                        # 전부 실패하면 기준선으로 물러난다
            flat = cands[0]
            return P39._cell_action(
                observation, flat, P39._toward_grid_center(observation, flat))
        return best


def _run(name, policy_factory, cfg, seed, min_fill, margin):
    pile = P39.synthetic_pile(seed)
    ex = P40.FailingExecutor(min_fill_fraction=min_fill, avalanche_margin_deg=margin)
    pol = policy_factory(ex)
    run = P39.run_episode(pile, pol, ex, cfg, seed=seed)
    steps = run.records
    nf = sum(1 for r in steps if r.failed)
    return {
        "policy": name,
        "total_scoops": len(steps),
        "total_time_s": round(sum(r.elapsed_time_s for r in steps), 1),
        "failures": nf,
        "failure_rate": round(nf / max(len(steps), 1), 3),
        "total_mass_kg": round(sum(r.scooped_mass_kg for r in steps), 5),
        "target_reached": run.terminal_reason == "target_reached",
        "terminal_reason": run.terminal_reason,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[457])
    ap.add_argument("--min-fill", type=float, default=0.55)
    ap.add_argument("--avalanche-margin-deg", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=24)
    a = ap.parse_args()

    cfg = P39.EpisodeConfig()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in a.seeds:
        specs = [
            ("greedy_high", lambda ex, s=seed: P40.AvoidFailedWrapper(
                P39.build_rule_policy("greedy_high",
                                      threshold_m=cfg.actionable_height_m, seed=s), ex)),
            ("oracle_1", lambda ex: OraclePolicy(
                ex, cfg.actionable_height_m, use_shape=False, top_k=a.top_k)),
            ("oracle_2", lambda ex: OraclePolicy(
                ex, cfg.actionable_height_m, use_shape=True, top_k=a.top_k)),
        ]
        for name, fac in specs:
            r = _run(name, fac, cfg, seed, a.min_fill, a.avalanche_margin_deg)
            r["seed"] = seed
            rows.append(r)
            print(f"seed {seed}  {name:12s} 스쿱 {r['total_scoops']:3d} · "
                  f"실패 {r['failures']:2d} ({r['failure_rate']:.3f}) · "
                  f"완주 {str(r['target_reached']):5s} · {r['terminal_reason']}",
                  flush=True)

    payload = {
        "artifact": "P41_ORACLE_LOOKAHEAD_V1",
        "question": "예측(커닝)이 greedy 대비 가져갈 수 있는 상한은?",
        "policies": {
            "greedy_high": "현재 높이 최대 (교수님 안). 실패셀 회피 규칙 포함",
            "oracle_1": "방법① — 양만 예측. 실패 안 하면서 최다 적재",
            "oracle_2": "방법② — 양 + 퍼낸 뒤 형상. 다음에 팔 곳이 남는지까지 본다",
        },
        "rows": rows,
        "non_claims": [
            "오라클은 학습이 아니라 **커닝**이다. 실제 학습 정책은 이보다 나쁘다.",
            f"후보를 높이 상위 {a.top_k}개로 잘랐다 -> 진짜 상한은 이보다 높을 수 있다.",
            "대리모델(AnalyticExecutor)의 붕괴 처리는 진짜 입자 물리가 아니다. "
            "최종 확인은 DEME 이 해야 한다.",
            "PP 물성 미실측. angle_of_repose 32도는 임시값이다.",
        ],
    }
    (out / "p41_oracle.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=P39._json_default),
        encoding="utf-8")
    print(f"-> {out/'p41_oracle.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
