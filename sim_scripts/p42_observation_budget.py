#!/usr/bin/env python3
"""P42 — 관측 예산: 다시 보는 것이 비쌀 때 누가 무너지나?

교수님 지적의 전제는 **"매번 다시 본다"** 이다. 히트맵을 매 스쿱마다 새로
찍을 수 있으면 "제일 높은 곳"만 봐도 된다. 크레인은 그 전제가 깨진다 —
버킷이 시야를 가리고, 스캔은 사이클보다 느리고, 운전자는 한 번 보고
여러 번 퍼낸다. 그러면 "지금 제일 높은 곳"은 **낡은 정보**가 된다.

이 스크립트는 그것을 벤치탑에서 재현한다:
  재관측을 **k 스쿱마다 한 번**만 허용하고 k = 1, 3, 5, inf 로 쓸어 본다.

세 정책이 낡은 정보를 다루는 방식이 이 실험의 전부다:

  greedy_high (교수님 안)
      낡은 히트맵을 그대로 쓴다. 예측 모형이 없다.
      "방금 판 자리는 기억한다"는 최소 기억만 준다(dug 제외) —
      이게 없으면 같은 칸을 k 번 다시 골라 자멸하므로 비교가 조작이 된다.

  oracle_1 (방법① 양만)
      후보의 **이번 적재량과 실패 여부**는 커닝으로 안다.
      그러나 **퍼낸 뒤 더미가 어떻게 변하는지는 모른다** ->
      자기 행동으로 갱신되는 belief 가 없다. greedy 와 같은 dug 기억만.

  oracle_2 (방법② 양 + 형상)
      퍼낸 뒤 형상까지 예측한다 -> 관측 사이에 **자기 belief 를 굴려서**
      다음 자리를 고를 수 있다. 이것이 k 가 커질 때 지불하는 값이다.

🔴 model_repose_bias_deg 가 왜 필요한가
    belief 모형이 실행기와 완전히 같으면 oracle_2 의 belief 는 언제나 진짜
    상태와 일치한다 -> k 를 아무리 늘려도 성능이 안 떨어진다. 그건 실험이
    아니라 정의다("시뮬레이터로 시뮬레이터를 예측했다").
    그래서 belief 모형의 **안식각을 실행기와 다르게** 둘 수 있게 했다.
    안식각은 PP 물성 미실측이라 실제로 우리가 모르는 값이고, 예측 모형이
    틀릴 수 있는 가장 정직한 자리다. bias=0 은 상한(완전 커닝),
    bias>0 은 "모형이 조금 틀린 예측 정책"이다.

정책도 실행기도 학습하지 않는다. 로봇 제어·DEME·데이터 생성 없음.
p39/p40/p41 은 수정하지 않는다 (forward-only).
"""
from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p39_decision_loop as P39  # noqa: E402
import p40_failure_greedy_reexam as P40  # noqa: E402
import p41_oracle_lookahead as P41  # noqa: E402
import p41b_oracle_stats as STATS  # noqa: E402

INF_K = 10 ** 9          # "한 번만 보고 끝까지" 를 나타내는 정수 k
POLICIES = ("greedy_high", "oracle_1", "oracle_2")


# --------------------------------------------------------------------------
# 관측 마스킹 — "이미 판 자리"와 "실패한 자리"를 후보에서 뺀다
# --------------------------------------------------------------------------
def mask_observation(observation: P39.Heightmap,
                     xy_list: Sequence[tuple[float, float]],
                     radius_m: float) -> P39.Heightmap:
    """지정한 xy 반경 안의 셀을 후보에서 제외한 관측 사본을 만든다.

    P40.AvoidFailedWrapper 와 같은 계약을 지킨다: 무효 셀은 높이 0.0 이어야
    하고(관측 계약), 원본 배열을 제자리 수정하면 안 된다.
    """
    if not xy_list:
        return observation
    height = observation.height.copy()
    valid = observation.valid.copy()
    centers = observation.spec.cell_centers()
    banned = np.zeros(height.shape, dtype=bool)
    for bx, by in xy_list:
        banned |= ((centers[..., 0] - bx) ** 2
                   + (centers[..., 1] - by) ** 2) <= radius_m ** 2
    height[banned] = 0.0
    valid[banned] = False
    return P39.Heightmap(
        height=np.ascontiguousarray(height, dtype=np.float32),
        valid=np.ascontiguousarray(valid, dtype=np.bool_),
        counts=np.ascontiguousarray(observation.counts.copy(), dtype=np.int32),
        spec=observation.spec, meta=dict(observation.meta))


# --------------------------------------------------------------------------
# Belief — 관측 사이에 정책이 들고 있는 "지금 더미가 이럴 것이다"
# --------------------------------------------------------------------------
class Belief:
    """관측 예산 아래에서 정책이 보는 세계.

    refresh(true_executor) 는 **진짜 재관측**이다. 예산이 허락할 때만 불린다.
    after_action() 은 관측 없이 belief 를 갱신하는 유일한 통로다.
    """

    def __init__(self, *, rolls_forward: bool, dug_radius_m: float,
                 repose_bias_deg: float) -> None:
        self.rolls_forward = bool(rolls_forward)
        self.dug_radius_m = float(dug_radius_m)
        self.repose_bias_deg = float(repose_bias_deg)
        self.model: P40.FailingExecutor | None = None   # belief 쪽 실행기 사본
        self.dug: list[tuple[float, float]] = []
        self.refresh_count = 0

    def refresh(self, true_executor: P40.FailingExecutor) -> None:
        model = copy.deepcopy(true_executor)
        model.failure_log = []
        if self.repose_bias_deg:
            # 안식각 하나만 틀린 모형. 형상 완화(_relax)와 붕괴 판정
            # (_failure_reason) 이 같은 상수를 읽으므로 한 번에 어긋난다.
            model.config = dataclasses.replace(
                model.config,
                angle_of_repose_deg=model.config.angle_of_repose_deg
                + self.repose_bias_deg)
        self.model = model
        self.dug = []
        self.refresh_count += 1

    def observation(self) -> P39.Heightmap:
        assert self.model is not None, "belief.refresh() 가 먼저 와야 한다"
        return mask_observation(self.model.observe(), self.dug, self.dug_radius_m)

    def after_action(self, action: P39.ScoopAction) -> None:
        if self.rolls_forward:
            # 방법② — 퍼낸 뒤 형상을 예측해서 belief 를 굴린다.
            # 형상 예측이 있으면 "판 자리에는 재료가 없다"가 belief 에 이미
            # 반영되므로 dug 목록이 필요 없다. 오히려 넣으면 좋은 자리로
            # 되돌아가는 것까지 금지해서 방법②를 부당하게 깎는다.
            assert self.model is not None
            self.model.execute(action)
        else:
            # 형상 예측이 없는 정책의 유일한 기억. 이게 없으면 낡은 지도에서
            # 같은 칸을 k 번 다시 고르며 자멸한다 -> 비교가 조작이 된다.
            self.dug.append((float(action.x_m), float(action.y_m)))


# --------------------------------------------------------------------------
# 정책 — belief 위에서만 결정한다
# --------------------------------------------------------------------------
def greedy_action(observation: P39.Heightmap, threshold_m: float) -> P39.ScoopAction:
    flat = P39._active_cells(observation, threshold_m)
    if flat.size == 0:
        raise RuntimeError("no actionable height cell")
    idx = int(flat[int(np.argmax(observation.height.reshape(-1)[flat]))])
    return P39._cell_action(observation, idx, P39._toward_grid_center(observation, idx))


def oracle_action(observation: P39.Heightmap, model: P40.FailingExecutor,
                  threshold_m: float, *, use_shape: bool, top_k: int,
                  shape_weight: float = 1.0) -> P39.ScoopAction:
    """P41.OraclePolicy 와 같은 채점을, 진짜 실행기가 아니라 belief 모형 위에서.

    bias=0 이고 belief 가 방금 갱신됐으면 P41 과 **비트 단위로 같은 선택**이다
    (게이트 G1 이 그것을 검사한다).
    """
    cands = P41._candidates(observation, threshold_m, top_k)
    if not cands:
        raise RuntimeError("no actionable height cell")
    best, best_score = None, -np.inf
    for flat in cands:
        action = P39._cell_action(observation, flat,
                                  P39._toward_grid_center(observation, flat))
        trial = copy.deepcopy(model)
        trial.failure_log = []
        result = trial.execute(action)
        if result.failed:
            continue
        score = float(result.scooped_mass_kg)
        if use_shape:
            after = trial.observe()
            nxt = P41._candidates(after, threshold_m, top_k)
            viable = sum(
                0 if trial._would_fail(
                    P39._cell_action(after, f2, P39._toward_grid_center(after, f2)))
                else 1
                for f2 in nxt)
            score += shape_weight * float(result.scooped_mass_kg) * (
                viable / max(len(nxt), 1))
        if score > best_score:
            best, best_score = action, score
    if best is None:
        flat = cands[0]
        return P39._cell_action(observation, flat,
                                P39._toward_grid_center(observation, flat))
    return best


def choose(policy: str, belief: Belief, executor: P40.FailingExecutor,
           threshold_m: float, top_k: int, avoid_radius_m: float) -> P39.ScoopAction:
    """정책 이름 -> 행동. belief 만 본다. 실패 피드백은 하중 신호이므로 허용."""
    observation = belief.observation()
    if policy == "greedy_high":
        # P41 과 동일하게 greedy 에만 실패-셀 회피를 붙인다(에피소드 내내 유지).
        # 실패는 히트맵 재관측이 아니라 "빈 채로 올라왔다"는 하중 피드백이라
        # 관측 예산과 무관하게 얻을 수 있다.
        banned = [(f["x_m"], f["y_m"]) for f in executor.failure_log]
        masked = mask_observation(observation, banned, avoid_radius_m)
        if P39._active_cells(masked, threshold_m).size == 0:
            masked = observation
        return greedy_action(masked, threshold_m)
    assert belief.model is not None
    return oracle_action(observation, belief.model, threshold_m,
                         use_shape=(policy == "oracle_2"), top_k=top_k)


# --------------------------------------------------------------------------
# 예산이 걸린 에피소드 루프
# --------------------------------------------------------------------------
def run_budgeted_episode(seed: int, policy: str, k: int, cfg: P39.EpisodeConfig,
                         *, min_fill: float, avalanche_margin: float,
                         dug_radius_m: float, avoid_radius_m: float,
                         top_k: int, repose_bias_deg: float,
                         leak_true_observation: bool = False,
                         target_scale: float = 1.0) -> dict[str, Any]:
    """P39.run_episode 와 같은 종료 규칙을 쓰되, 정책에는 belief 만 준다.

    P39.run_episode 는 매 스텝 **진짜 관측**을 정책에 넘긴다. 관측 예산을
    걸려면 그 지점을 바꿔야 하는데 p39 는 수정 금지이므로 여기서 같은
    종료 계약(target/max_attempts/연속·누적 실패 한도/재료 고갈)을 재현한다.
    """
    pile = P39.synthetic_pile(seed)
    executor = P40.FailingExecutor(min_fill_fraction=min_fill,
                                   avalanche_margin_deg=avalanche_margin)
    P39.validate_observation_contract(pile)
    executor.reset(pile)
    target_mass = cfg.target_mass_kg * float(target_scale)

    belief = Belief(rolls_forward=(policy == "oracle_2"),
                    dug_radius_m=dug_radius_m, repose_bias_deg=repose_bias_deg)

    mass = time_s = distance = 0.0
    failures = consecutive = scoops = 0
    terminal = "max_attempts"
    for attempt in range(cfg.max_attempts):
        if mass + 1.0e-12 >= target_mass:
            terminal = "target_reached"
            break
        truth = executor.observe()
        P39.validate_observation_contract(truth)
        if P39._active_cells(truth, cfg.actionable_height_m).size == 0:
            terminal = "material_exhausted"
            break
        # 여기가 관측 예산이다. leak 플래그는 게이트 G4 의 회귀 주입용이다.
        if leak_true_observation or attempt % k == 0:
            belief.refresh(executor)

        try:
            action = choose(policy, belief, executor,
                            cfg.actionable_height_m, top_k, avoid_radius_m)
            P39.validate_action(action)
        except Exception as exc:                      # noqa: BLE001
            terminal = f"policy_error:{exc!r}"
            break

        result = executor.execute(action)
        P39.validate_observation_contract(result.observation)
        belief.after_action(action)

        mass += result.scooped_mass_kg
        time_s += result.elapsed_time_s
        distance += result.travel_distance_m
        failures += int(result.failed)
        consecutive = consecutive + 1 if result.failed else 0
        scoops += 1
        if consecutive >= cfg.max_consecutive_failures:
            terminal = "consecutive_failure_limit"
            break
        if failures >= cfg.max_total_failures:
            terminal = "total_failure_limit"
            break
    if mass + 1.0e-12 >= target_mass:
        terminal = "target_reached"

    return {
        "seed": int(seed), "policy": policy,
        "k": "inf" if k >= INF_K else int(k),
        "total_scoops": scoops,
        "total_time_s": round(time_s, 1),
        "total_distance_m": round(distance, 2),
        "failures": failures,
        "failure_rate": round(failures / max(scoops, 1), 3),
        "total_mass_kg": round(mass, 5),
        "target_mass_kg": round(target_mass, 5),
        "observations_used": belief.refresh_count,
        "target_reached": terminal == "target_reached",
        "terminal_reason": terminal,
    }


# --------------------------------------------------------------------------
def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """k x 정책 격자 -> 완주율·스쿱·실패율·관측 횟수 요약."""
    grid: dict[str, dict[str, Any]] = {}
    for row in rows:
        grid.setdefault(str(row["k"]), {}).setdefault(row["policy"], []).append(row)
    out: dict[str, Any] = {}
    for k, per_policy in grid.items():
        out[k] = {}
        for name, group in per_policy.items():
            done = [r for r in group if r["target_reached"]]
            scoops = [r["total_scoops"] for r in done]
            out[k][name] = {
                "n_seeds": len(group),
                "n_completed": len(done),
                "completion_rate": round(len(done) / max(len(group), 1), 4),
                "scoops_completed_mean": (round(statistics.fmean(scoops), 3)
                                          if scoops else None),
                "scoops_completed_std": (round(statistics.stdev(scoops), 3)
                                         if len(scoops) > 1 else None),
                "failure_rate_mean": round(
                    statistics.fmean(r["failure_rate"] for r in group), 4),
                "observations_mean": round(
                    statistics.fmean(r["observations_used"] for r in group), 3),
            }
    return out


def _sign_tests_per_k(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """k 마다 정책 쌍 부호검정. P41b 와 동일한 승패 규칙을 재사용한다."""
    by_k: dict[str, dict[str, dict[int, Mapping[str, Any]]]] = {}
    for row in rows:
        by_k.setdefault(str(row["k"]), {}).setdefault(
            row["policy"], {})[int(row["seed"])] = row
    results = []
    for k, table in by_k.items():
        for a, b in (("oracle_2", "greedy_high"), ("oracle_1", "greedy_high"),
                     ("oracle_2", "oracle_1")):
            if a in table and b in table:
                results.append({"k": k, **STATS.sign_test(table, a, b)})
    return results


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# P42 — 관측 예산 스윕 (재관측을 k 스쿱마다만 허용)", "",
             f"시드 {payload['seeds'][0]}~{payload['seeds'][-1]} (n={len(payload['seeds'])}) · "
             f"belief 안식각 편차 {payload['model_repose_bias_deg']}도 · "
             f"dug 제외 반경 {payload['dug_radius_m']} m", "",
             "| k | 정책 | 완주율 | 스쿱(완주분) mean±std | 실패율 mean | 관측 횟수 mean |",
             "|---|---|---:|---:|---:|---:|"]
    for k in payload["k_order"]:
        for name in POLICIES:
            s = payload["aggregate"][k][name]
            mean = "—" if s["scoops_completed_mean"] is None else f"{s['scoops_completed_mean']:.2f}"
            std = "—" if s["scoops_completed_std"] is None else f"{s['scoops_completed_std']:.2f}"
            lines.append(
                f"| {k} | {name} | {s['n_completed']}/{s['n_seeds']} = "
                f"{s['completion_rate']:.3f} | {mean} ± {std} | "
                f"{s['failure_rate_mean']:.4f} | {s['observations_mean']:.2f} |")
    lines += ["", "## k 별 부호검정 (완주 우선, 동률이면 스쿱 수)", "",
              "| k | 비교 | a승 | b승 | 무 | p(단측) | 0.05 유의 |",
              "|---|---|---:|---:|---:|---:|:--:|"]
    for t in payload["sign_tests"]:
        p = "—" if t["p_one_sided"] is None else f"{t['p_one_sided']:.5g}"
        lines.append(f"| {t['k']} | {t['a']} > {t['b']} | {t['wins_a']} | {t['wins_b']} | "
                     f"{t['ties']} | {p} | {'✅' if t.get('significant_at_0.05') else '❌'} |")
    lines += ["", "## non_claims", ""] + [f"- {c}" for c in payload["non_claims"]]
    return "\n".join(lines) + "\n"


NON_CLAIMS = [
    "오라클은 학습이 아니라 **커닝**이다 — 후보의 결과를 실행기 사본으로 미리 본다. "
    "따라서 oracle_1/oracle_2 곡선은 '학습이 이렇게 된다'가 아니라 "
    "**학습이 도달할 수 있는 상한**이다. 실제 학습 정책은 반드시 이보다 나쁘다.",
    "belief 모형의 오차를 안식각 편차 **한 축**으로만 넣었다. 실제 예측 모형은 "
    "밀도·발자국·입자 크기·재료 이력에서도 틀린다. bias=0 결과는 상한 중의 상한이며 "
    "k 에 둔감한 것이 당연하다(모형이 곧 실행기이므로) — 그 자체를 결과로 인용하지 말 것.",
    "대리모델(AnalyticExecutor/FailingExecutor)의 붕괴 처리는 진짜 입자 물리가 아니다. "
    "안식각 초과분을 이웃과 나눠 갖는 기하 완화 규칙이다. "
    "최종 확인은 DEME 과 실물 스쿱이 한다.",
    "PP 물성 미실측. angle_of_repose 32도는 임시값이고 bias 도 임의값이다. "
    "어떤 수치도 폴리프로필렌 값으로 인용 금지.",
    "greedy_high 에 준 'dug 기억'과 '실패 셀 회피'는 비교를 공정하게 만들기 위한 "
    "최소 장치이지 교수님 안의 일부가 아니다. 이 장치가 없으면 greedy 는 낡은 지도에서 "
    "같은 칸을 반복해 자멸하며, 그 결과는 조작이 된다.",
    "k 는 재관측 **횟수**만 제한한다. 실제 크레인에서 재관측이 비싼 이유(버킷 가림, "
    "스캔 시간, 분진)는 모델에 없다. 시간 지표에 관측 비용이 들어 있지 않다.",
    "시드 30개는 같은 능선 생성기의 미세 변형이다. 장면 다양성의 표본이 아니다.",
]


# --------------------------------------------------------------------------
def cmd_run(a: argparse.Namespace) -> int:
    cfg = P39.EpisodeConfig()
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    ks = [INF_K if str(x).lower() in ("inf", "0") else int(x) for x in a.observe_every]
    rows: list[dict[str, Any]] = []
    for k in ks:
        for seed in seeds:
            for name in POLICIES:
                effective = "greedy_high" if a.force_identical_policies else name
                scale = (a.unfair_target_scale
                         if (a.unfair_target_scale != 1.0 and name == "greedy_high")
                         else 1.0)
                row = run_budgeted_episode(
                    seed, effective, k, cfg,
                    min_fill=a.min_fill, avalanche_margin=a.avalanche_margin_deg,
                    dug_radius_m=a.dug_radius_m, avoid_radius_m=a.avoid_radius_m,
                    top_k=a.top_k, repose_bias_deg=a.model_repose_bias_deg,
                    leak_true_observation=a.leak_true_observation,
                    target_scale=scale)
                row["policy"] = name          # 주입 플래그가 켜져도 라벨은 유지
                rows.append(row)
            done = rows[-len(POLICIES):]
            print(f"k={'inf' if k >= INF_K else k:>3} seed {seed}  " + " · ".join(
                f"{r['policy']}:{r['total_scoops']:2d}"
                f"{'완주' if r['target_reached'] else '중단'}" for r in done),
                flush=True)

    payload = {
        "artifact": "P42_OBSERVATION_BUDGET_V1",
        "question": "재관측을 k 스쿱마다만 허용하면 누가 먼저 무너지나?",
        "seeds": seeds,
        "k_order": ["inf" if k >= INF_K else str(k) for k in ks],
        "config": {
            "min_fill_fraction": a.min_fill,
            "avalanche_margin_deg": a.avalanche_margin_deg,
            "dug_radius_m": a.dug_radius_m,
            "avoid_radius_m": a.avoid_radius_m,
            "top_k": a.top_k,
            "episode": dataclasses.asdict(cfg),
        },
        "model_repose_bias_deg": a.model_repose_bias_deg,
        "dug_radius_m": a.dug_radius_m,
        "injections": {
            "leak_true_observation": a.leak_true_observation,
            "force_identical_policies": a.force_identical_policies,
            "unfair_target_scale": a.unfair_target_scale,
        },
        "rows": rows,
        "aggregate": _aggregate(rows),
        "sign_tests": _sign_tests_per_k(rows),
        "non_claims": NON_CLAIMS,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p42_observation_budget.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "p42_observation_budget.md").write_text(_markdown(payload), encoding="utf-8")
    print(f"-> {out/'p42_observation_budget.json'}")
    print(f"-> {out/'p42_observation_budget.md'}")
    return 0


# --------------------------------------------------------------------------
# 교차점 — 예측이 greedy 를 확실히 이기기 시작하는 k 는 어디인가
# --------------------------------------------------------------------------
def cmd_analyze(a: argparse.Namespace) -> int:
    payload = json.loads(Path(a.p42).read_text(encoding="utf-8"))
    agg, ks = payload["aggregate"], payload["k_order"]
    tests = {(t["k"], t["a"], t["b"]): t for t in payload["sign_tests"]}

    def first_k(pred) -> str | None:
        return next((k for k in ks if pred(k)), None)

    curve = []
    for k in ks:
        row = {"k": k}
        for name in POLICIES:
            s = agg[k][name]
            row[f"{name}_completion"] = s["completion_rate"]
            row[f"{name}_scoops"] = s["scoops_completed_mean"]
        for pair in (("oracle_2", "greedy_high"), ("oracle_1", "greedy_high"),
                     ("oracle_2", "oracle_1")):
            t = tests.get((k, *pair))
            row[f"p_{pair[0]}_gt_{pair[1]}"] = None if t is None else t["p_one_sided"]
        curve.append(row)

    def sig(a_, b_):
        return first_k(lambda k: (t := tests.get((k, a_, b_))) is not None
                       and t["p_one_sided"] is not None and t["p_one_sided"] < 0.05)

    crossover = {
        "k_first_significant_oracle2_over_greedy": sig("oracle_2", "greedy_high"),
        "k_first_significant_oracle1_over_greedy": sig("oracle_1", "greedy_high"),
        "k_first_significant_oracle2_over_oracle1": sig("oracle_2", "oracle_1"),
        "k_greedy_completion_below_0.5": first_k(
            lambda k: agg[k]["greedy_high"]["completion_rate"] < 0.5),
        "k_greedy_completion_zero": first_k(
            lambda k: agg[k]["greedy_high"]["completion_rate"] == 0.0),
        "k_oracle1_completion_below_0.5": first_k(
            lambda k: agg[k]["oracle_1"]["completion_rate"] < 0.5),
        "k_oracle2_completion_below_0.5": first_k(
            lambda k: agg[k]["oracle_2"]["completion_rate"] < 0.5),
    }
    out_payload = {
        "artifact": "P42_CROSSOVER_V1",
        "source": str(a.p42),
        "model_repose_bias_deg": payload["model_repose_bias_deg"],
        "k_order": ks,
        "curve": curve,
        "crossover": crossover,
        "reading": [
            "k_first_significant_* 는 '그 k 에서 부호검정 p<0.05 가 처음 성립하는 k'다. "
            "k=1 이면 관측 예산을 조이기 전부터 이미 이기고 있었다는 뜻이고, "
            "그때 k 가 더해 주는 것은 승패가 아니라 **격차의 크기**다.",
            "k_greedy_completion_* 는 교수님 안이 목표 적재를 아예 못 채우기 시작하는 "
            "관측 주기다. 크레인 전이에서 '몇 번 퍼낼 때마다 다시 봐야 하는가'의 하한이 된다.",
        ],
        "non_claims": NON_CLAIMS,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p42_crossover.json").write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in curve:
        print(f"k={row['k']:>3}  완주율 greedy {row['greedy_high_completion']:.3f} · "
              f"o1 {row['oracle_1_completion']:.3f} · o2 {row['oracle_2_completion']:.3f}"
              f"   p(o2>greedy)={row['p_oracle_2_gt_greedy_high']}")
    for key, value in crossover.items():
        print(f"{key} = {value}")
    print(f"-> {out/'p42_crossover.json'}")
    return 0


# --------------------------------------------------------------------------
# 게이트 — 각 게이트는 FAIL 을 낼 수 있어야 한다. 주입 명령은 GATES.md 참조.
# --------------------------------------------------------------------------
def _gate(name: str, kind: str, ok: bool, detail: str,
          blind_spot: str) -> dict[str, Any]:
    """kind='validity' = 실험이 망가지거나 조작되지 않았는지.
    kind='claim'    = 프로포절의 주장이 실제로 성립하는지.
    둘을 섞으면 '주장이 아직 안 선다'가 '코드가 깨졌다'처럼 보인다.
    """
    return {"gate": name, "kind": kind, "status": "PASS" if ok else "FAIL",
            "detail": detail, "blind_spot": blind_spot}


def cmd_gate(a: argparse.Namespace) -> int:
    p42 = json.loads(Path(a.p42).read_text(encoding="utf-8"))
    rows = p42["rows"]
    gates: list[dict[str, Any]] = []

    # G1 — k=1, bias=0 이면 P41 오라클과 시드별로 완전히 같아야 한다.
    if a.p41:
        p41 = {(r["policy"], int(r["seed"])): r
               for r in json.loads(Path(a.p41).read_text(encoding="utf-8"))["rows"]}
        mismatches, covered = [], 0
        for r in (row for row in rows if str(row["k"]) == "1"):
            key = (r["policy"], int(r["seed"]))
            if key not in p41:
                continue
            covered += 1
            mine = (r["total_scoops"], r["failures"], r["target_reached"])
            theirs = (p41[key]["total_scoops"], p41[key]["failures"],
                      p41[key]["target_reached"])
            if mine != theirs:
                mismatches.append(f"{key[0]}/seed{key[1]}: p42{mine} != p41{theirs}")
        gates.append(_gate(
            "G1_k1_reproduces_p41", "validity",
            not mismatches and covered > 0,
            ("대조 가능한 k=1 행이 없다 -> 게이트를 적용하지 못했으므로 FAIL 처리"
             if covered == 0 else
             f"k=1 행 {covered}건 대조, 불일치 {len(mismatches)}건. "
             + ("; ".join(mismatches[:3]) if mismatches else "전건 일치")),
            "P41 자체가 틀렸다면 둘이 같이 틀린 채로 PASS 한다. "
            "대리모델 물리의 정확성은 이 게이트가 보지 못한다. "
            "또 bias≠0 로 돌린 산출물에는 애초에 적용되지 않는다."))

    # G2 — oracle_2 > oracle_1 이 부호검정으로 유의한가 (k 별).
    for t in p42["sign_tests"]:
        if (t["a"], t["b"]) != ("oracle_2", "oracle_1"):
            continue
        p = t["p_one_sided"]
        gates.append(_gate(
            f"G2_shape_gain_significant_k{t['k']}", "claim",
            bool(p is not None and p < 0.05),
            f"승{t['wins_a']} 패{t['wins_b']} 무{t['ties']} n_eff={t['n_effective']} "
            f"p={p}. n_eff<5 면 전승이어도 0.05 를 넘길 수 없다"
            f"(최소 가능 p={t.get('best_possible_p_at_this_n_effective')}).",
            "대리모델 안에서의 유의성이다. 실물 재현을 보증하지 않는다. "
            "시드는 같은 능선 생성기의 미세 변형이라 장면 다양성을 재지 못하고, "
            "무승부가 많으면 표본이 아니라 검정력이 먼저 죽는다."))

    # G3 — 세 정책이 실제로 다른 행동을 하는가.
    same = [f"k={k} seed={s}" for k in {str(r["k"]) for r in rows}
            for s in {r["seed"] for r in rows}
            if len({r["total_scoops"] for r in rows
                    if str(r["k"]) == k and r["seed"] == s}) == 1]
    total_cells = len({(str(r["k"]), r["seed"]) for r in rows})
    gates.append(_gate(
        "G3_policies_are_distinct", "validity",
        len(same) < total_cells,
        f"동일 (k,seed) 격자 {total_cells}칸 중 세 정책의 스쿱 수가 완전히 같은 칸 "
        f"{len(same)}칸. 전 칸이 같으면 같은 정책을 세 번 돌린 것이다.",
        "스쿱 수만 본다. 서로 다른 자리를 골랐지만 우연히 스쿱 수가 같은 경우와 "
        "정말 같은 정책인 경우를 구분하지 못한다. 행동 궤적을 비교하지 않는다."))

    # G4 — 관측 예산을 실제로 지켰는가.
    violations = []
    for r in rows:
        k = INF_K if r["k"] == "inf" else int(r["k"])
        expected = 0 if r["total_scoops"] == 0 else math.ceil(r["total_scoops"] / k)
        if r["observations_used"] != expected:
            violations.append(f"{r['policy']}/k={r['k']}/seed{r['seed']}: "
                              f"{r['observations_used']} != {expected}")
    gates.append(_gate(
        "G4_observation_budget_respected", "validity",
        not violations,
        f"전 {len(rows)}행 중 관측 횟수 != ceil(스쿱/k) 인 행 {len(violations)}건. "
        + ("; ".join(violations[:3]) if violations else "전건 일치"),
        "**횟수만** 센다. belief 배열이 어떤 경로로 진짜 상태와 같아졌는지는 못 본다. "
        "bias=0 이면 oracle_2 의 belief 는 언제나 진짜 상태와 일치하지만 "
        "관측 카운트는 정상이므로 이 게이트는 PASS 한다."))

    # G5 — 비교가 대칭인가 (같은 목표질량·같은 실패 모델·같은 에피소드 설정).
    targets = {(r["policy"], r["target_mass_kg"]) for r in rows}
    per_policy = {p: {t for q, t in targets if q == p} for p in POLICIES}
    unique = {t for _, t in targets}
    gates.append(_gate(
        "G5_comparison_is_symmetric", "validity",
        len(unique) == 1,
        f"정책별 목표질량 집합 {per_policy}. 값이 하나가 아니면 한 정책에만 "
        f"다른 목표를 준 것이다.",
        "목표질량만 본다. 실패 모델 파라미터·초기 더미·top_k 처럼 정책마다 "
        "따로 줄 수 있는 다른 축은 이 게이트가 검사하지 않는다. "
        "또 '공정한 설정'이 '현실적인 설정'을 뜻하지도 않는다."))

    # G8 — 정책이 후보를 못 찾아 죽은 행이 있는가.
    #     dug 제외 반경이 크면 마스크가 지도를 통째로 덮어 정책이 예외로 죽는다.
    #     그 행의 '미완주'는 물리적 붕괴가 아니라 코드 아티팩트라 인용하면 안 된다.
    errored = [r for r in rows if str(r["terminal_reason"]).startswith("policy_error")]
    by_policy = collections.Counter(r["policy"] for r in errored)
    gates.append(_gate(
        "G8_no_policy_error_rows", "validity", not errored,
        f"전 {len(rows)}행 중 policy_error 로 끝난 행 {len(errored)}건 "
        f"(정책별 {dict(by_policy)}). 후보 셀이 0개가 되어 정책이 예외로 죽은 것이며, "
        f"이 행의 미완주는 물리가 아니라 마스킹 아티팩트다.",
        "policy_error 만 센다. 마스킹이 지도를 **거의** 다 덮어 정책이 억지로 나쁜 "
        "칸을 고른 경우(예외는 안 남)는 잡지 못한다. 즉 PASS 라도 dug 반경이 결과를 "
        "왜곡하지 않았다는 보증은 아니다 — 반경 민감도는 따로 돌려 봐야 한다."))

    # G6/G7 — P41 n=30 통계 산출물 위의 주장 게이트.
    if a.p41b_stats:
        stats = json.loads(Path(a.p41b_stats).read_text(encoding="utf-8"))
        tests = {(t["a"], t["b"]): t for t in stats["sign_tests"]}
        for tag, pair in (("G6_oracle2_beats_greedy", ("oracle_2", "greedy_high")),
                          ("G7_shape_beats_mass_only", ("oracle_2", "oracle_1"))):
            t = tests.get(pair)
            if t is None:
                continue
            gates.append(_gate(
                tag, "claim", bool(t.get("significant_at_0.05")),
                f"n={t['n_seeds']} 시드에서 승{t['wins_a']} 패{t['wins_b']} 무{t['ties']}, "
                f"n_eff={t['n_effective']}, p(단측)={t['p_one_sided']}. "
                f"이 n_eff 에서 전승이어도 나올 수 있는 최소 p = "
                f"{t.get('best_possible_p_at_this_n_effective')}.",
                "대리모델 안의 유의성이다. 시드는 같은 능선 생성기의 미세 변형이므로 "
                "장면 다양성(더미 개수·형상 종류·용기 위치)에 대한 일반화를 보지 못한다. "
                "무승부가 많으면 검정력이 먼저 죽어, 효과가 실재해도 FAIL 이 나온다 — "
                "FAIL 은 '효과 없음'이 아니라 '이 표본으로는 못 가른다'는 뜻이다."))

    validity = [g for g in gates if g["kind"] == "validity"]
    claims = [g for g in gates if g["kind"] == "claim"]
    payload = {
        "artifact": "P42_GATES_V1", "source_p42": str(a.p42),
        "source_p41": str(a.p41) if a.p41 else None,
        "source_p41b_stats": str(a.p41b_stats) if a.p41b_stats else None,
        "gates": gates,
        "validity_all_pass": all(g["status"] == "PASS" for g in validity),
        "claim_pass_count": sum(1 for g in claims if g["status"] == "PASS"),
        "claim_total": len(claims),
    }
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "p42_gate_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {out/'p42_gate_report.json'}")
    for g in gates:
        print(f"[{g['status']}] ({g['kind']:8s}) {g['gate']} — {g['detail']}")
    print(f"VALIDITY_ALL_PASS={payload['validity_all_pass']}  "
          f"CLAIM_PASS={payload['claim_pass_count']}/{payload['claim_total']}")
    # 종료코드는 **validity** 만 본다. claim 게이트의 FAIL 은 결과이지 고장이 아니다.
    return 0 if payload["validity_all_pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("run", help="관측 예산 스윕 실행")
    run.add_argument("--out", required=True)
    run.add_argument("--seed-start", type=int, default=457)
    run.add_argument("--n-seeds", type=int, default=30)
    run.add_argument("--observe-every", nargs="+", default=["1", "3", "5", "inf"])
    run.add_argument("--min-fill", type=float, default=0.55)
    run.add_argument("--avalanche-margin-deg", type=float, default=0.0)
    run.add_argument("--top-k", type=int, default=24)
    run.add_argument("--dug-radius-m", type=float, default=0.029,
                     help="관측 사이 '이미 판 자리' 제외 반경. 기본은 발자국 반폭")
    run.add_argument("--avoid-radius-m", type=float, default=0.008,
                     help="greedy 의 실패-셀 회피 반경 (P40 기본값과 동일)")
    run.add_argument("--model-repose-bias-deg", type=float, default=0.0,
                     help="belief 모형의 안식각 편차. 0 = 완전 커닝(상한)")
    run.add_argument("--leak-true-observation", action="store_true",
                     help="[게이트 회귀 주입] 예산을 무시하고 매 스텝 재관측")
    run.add_argument("--force-identical-policies", action="store_true",
                     help="[게이트 회귀 주입] 세 정책을 전부 greedy_high 로 실행")
    run.add_argument("--unfair-target-scale", type=float, default=1.0,
                     help="[게이트 회귀 주입] greedy_high 에만 목표질량 배율 적용")
    run.set_defaults(func=cmd_run)

    ana = sub.add_parser("analyze", help="k 곡선 + 교차점 추출")
    ana.add_argument("--p42", required=True)
    ana.add_argument("--out", required=True)
    ana.set_defaults(func=cmd_analyze)

    gate = sub.add_parser("gate", help="산출물 게이트 검사")
    gate.add_argument("--p42", required=True)
    gate.add_argument("--p41", default=None, help="G1 대조용 p41_oracle.json")
    gate.add_argument("--p41b-stats", default=None,
                      help="G6/G7 대조용 p41b_stats.json")
    gate.add_argument("--out", default=None)
    gate.set_defaults(func=cmd_gate)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
