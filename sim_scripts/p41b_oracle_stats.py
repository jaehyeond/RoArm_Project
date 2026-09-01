#!/usr/bin/env python3
"""P41 n=30 통계 — oracle_2 가 oracle_1 보다 나은 것이 우연인가?

P41 은 시드 5개였다. 시드 458 한 개에서만 oracle_1 이 실패하고 oracle_2 가
성공했고, 그 한 칸이 프로포절 기여2("양 + 형상")의 유일한 근거였다.
1/5 는 주장이 아니라 일화다. 이 스크립트는 시드 30개 결과를 받아
**부호검정(sign test)** 으로 그 한 칸이 우연인지 아닌지를 계산한다.

부호검정이란: 시드마다 두 정책을 짝지어 승/패/무를 세고,
"동전 던지기라면 승이 이만큼 나올 확률"을 이항분포로 직접 구하는 방법이다.
정규분포나 등분산 같은 가정이 필요 없어서, 스쿱 수처럼 분포를 모르는
정수 지표에 안전하다. 무승부는 세지 않는다(표준 부호검정 규약).

승패 판정 순서 (완주가 스쿱 수보다 우선한다):
  1. 한쪽만 완주 -> 완주한 쪽 승
  2. 둘 다 완주  -> 스쿱 수가 적은 쪽 승, 같으면 무
  3. 둘 다 미완주 -> 무 (보수적. 적재량으로 순위를 매기지 않는다)

p39/p40/p41 은 건드리지 않는다 (forward-only).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from scipy.stats import binomtest


def load_rows(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """p41_oracle.json -> {policy: {seed: row}}"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    table: dict[str, dict[int, dict[str, Any]]] = {}
    for row in payload["rows"]:
        table.setdefault(row["policy"], {})[int(row["seed"])] = row
    return table


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scoops_all = [int(r["total_scoops"]) for r in rows]
    done = [r for r in rows if r["target_reached"]]
    scoops_done = [int(r["total_scoops"]) for r in done]
    fail_rates = [float(r["failure_rate"]) for r in rows]

    def ms(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": round(float(statistics.fmean(values)), 4),
            # 표본표준편차(ddof=1). n=1 이면 정의되지 않으므로 None.
            "std": (round(float(statistics.stdev(values)), 4)
                    if len(values) > 1 else None),
            "n": len(values),
        }

    return {
        "n_seeds": len(rows),
        "n_completed": len(done),
        "completion_rate": round(len(done) / max(len(rows), 1), 4),
        "scoops_completed_only": ms(scoops_done),
        "scoops_all_runs": ms(scoops_all),
        "failure_rate": ms(fail_rates),
        "terminal_reasons": {
            reason: sum(1 for r in rows if r["terminal_reason"] == reason)
            for reason in sorted({r["terminal_reason"] for r in rows})
        },
    }


def _pair_verdict(a: Mapping[str, Any], b: Mapping[str, Any]) -> int:
    """+1 = a 승, -1 = b 승, 0 = 무."""
    ra, rb = bool(a["target_reached"]), bool(b["target_reached"])
    if ra != rb:
        return 1 if ra else -1
    if not ra:
        return 0
    sa, sb = int(a["total_scoops"]), int(b["total_scoops"])
    if sa == sb:
        return 0
    return 1 if sa < sb else -1


def sign_test(table: Mapping[str, Mapping[int, Mapping[str, Any]]],
              a: str, b: str) -> dict[str, Any]:
    """a 가 b 보다 낫다는 단측 부호검정 + 참고용 양측 p 값."""
    seeds = sorted(set(table[a]) & set(table[b]))
    verdicts = {s: _pair_verdict(table[a][s], table[b][s]) for s in seeds}
    wins = sum(1 for v in verdicts.values() if v > 0)
    losses = sum(1 for v in verdicts.values() if v < 0)
    ties = sum(1 for v in verdicts.values() if v == 0)
    n_eff = wins + losses
    if n_eff == 0:
        return {
            "a": a, "b": b, "n_seeds": len(seeds),
            "wins_a": 0, "wins_b": 0, "ties": ties, "n_effective": 0,
            "p_one_sided": None, "p_two_sided": None,
            "significant_at_0.05": False,
            "note": "모든 시드가 무승부 -> 부호검정을 적용할 표본이 없다",
        }
    one = binomtest(wins, n_eff, 0.5, alternative="greater")
    two = binomtest(wins, n_eff, 0.5, alternative="two-sided")
    # 부호검정의 구조적 하한: 전승이어도 p = 0.5^n_eff 이다.
    # 무승부가 많아 n_eff 가 작으면 **어떤 짝지은 부호 기반 검정으로도**
    # 0.05 를 넘길 수 없다. 검정 선택 문제가 아니라 표본 문제라는 뜻이다.
    floor_n = 5   # 0.5^5 = 0.03125 < 0.05, 0.5^4 = 0.0625 > 0.05
    return {
        "a": a, "b": b, "n_seeds": len(seeds),
        "wins_a": wins, "wins_b": losses, "ties": ties, "n_effective": n_eff,
        "test": "exact binomial sign test (H0: P(a 승)=0.5)",
        "p_one_sided": float(one.pvalue),
        "p_two_sided": float(two.pvalue),
        "significant_at_0.05": bool(one.pvalue < 0.05),
        "best_possible_p_at_this_n_effective": float(0.5 ** n_eff),
        "n_effective_needed_for_p05_if_all_wins": floor_n,
        "seeds_needed_at_observed_discordance_rate": (
            None if n_eff == 0 else math.ceil(floor_n * len(seeds) / n_eff)),
        "win_seeds": [s for s, v in verdicts.items() if v > 0],
        "loss_seeds": [s for s, v in verdicts.items() if v < 0],
    }


def mcnemar_completion(table: Mapping[str, Mapping[int, Mapping[str, Any]]],
                       a: str, b: str) -> dict[str, Any]:
    """완주/미완주만 놓고 보는 McNemar 정확검정.

    스쿱 수를 무시하고 "끝까지 채웠나"만 본다. 불일치 쌍(한쪽만 완주)만
    정보를 가지므로 그 위에서 이항검정을 한다.
    """
    seeds = sorted(set(table[a]) & set(table[b]))
    a_only = sum(1 for s in seeds
                 if table[a][s]["target_reached"] and not table[b][s]["target_reached"])
    b_only = sum(1 for s in seeds
                 if table[b][s]["target_reached"] and not table[a][s]["target_reached"])
    n = a_only + b_only
    if n == 0:
        return {"a": a, "b": b, "a_only": 0, "b_only": 0,
                "p_one_sided": None, "note": "불일치 쌍 없음"}
    result = binomtest(a_only, n, 0.5, alternative="greater")
    return {
        "a": a, "b": b, "a_only_completed": a_only, "b_only_completed": b_only,
        "test": "McNemar exact (완주 여부만)",
        "p_one_sided": float(result.pvalue),
        "significant_at_0.05": bool(result.pvalue < 0.05),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# P41b — 오라클 n=30 통계",
        "",
        f"입력: `{payload['source']}`  ·  시드 {payload['seeds'][0]}~{payload['seeds'][-1]}"
        f" (n={len(payload['seeds'])})",
        "",
        "## 정책별 요약",
        "",
        "| 정책 | 완주 | 완주율 | 스쿱(완주분) mean±std | 스쿱(전체) mean±std | 실패율 mean±std |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(block: Mapping[str, Any]) -> str:
        if block["mean"] is None:
            return "—"
        std = "—" if block["std"] is None else f"{block['std']:.2f}"
        return f"{block['mean']:.2f} ± {std} (n={block['n']})"

    for name, s in payload["per_policy"].items():
        lines.append(
            f"| {name} | {s['n_completed']}/{s['n_seeds']} | {s['completion_rate']:.3f} | "
            f"{fmt(s['scoops_completed_only'])} | {fmt(s['scoops_all_runs'])} | "
            f"{fmt(s['failure_rate'])} |")
    lines += ["", "## 부호검정 (완주 우선, 동률이면 스쿱 수)", "",
              "| 비교 | a승 | b승 | 무 | n_eff | p(단측) | 0.05 유의 |",
              "|---|---:|---:|---:|---:|---:|:--:|"]
    for t in payload["sign_tests"]:
        p = "—" if t["p_one_sided"] is None else f"{t['p_one_sided']:.5g}"
        lines.append(
            f"| {t['a']} > {t['b']} | {t['wins_a']} | {t['wins_b']} | {t['ties']} | "
            f"{t['n_effective']} | {p} | {'✅' if t.get('significant_at_0.05') else '❌'} |")
    lines += ["", "## McNemar 정확검정 (완주 여부만)", "",
              "| 비교 | a만 완주 | b만 완주 | p(단측) | 0.05 유의 |",
              "|---|---:|---:|---:|:--:|"]
    for t in payload["mcnemar_completion"]:
        p = "—" if t.get("p_one_sided") is None else f"{t['p_one_sided']:.5g}"
        lines.append(
            f"| {t['a']} > {t['b']} | {t.get('a_only_completed', 0)} | "
            f"{t.get('b_only_completed', 0)} | {p} | "
            f"{'✅' if t.get('significant_at_0.05') else '❌'} |")
    lines += ["", "## 표본 한계 (부호검정의 구조적 하한)", "",
              "| 비교 | n_eff | 전승이어도 나올 수 있는 최소 p | p<0.05 에 필요한 n_eff | "
              "관측 불일치율 유지 시 필요한 시드 |",
              "|---|---:|---:|---:|---:|"]
    for t in payload["sign_tests"]:
        if t["n_effective"] == 0:
            continue
        lines.append(
            f"| {t['a']} > {t['b']} | {t['n_effective']} | "
            f"{t['best_possible_p_at_this_n_effective']:.5g} | "
            f"{t['n_effective_needed_for_p05_if_all_wins']} | "
            f"{t['seeds_needed_at_observed_discordance_rate']} |")
    lines += ["", "## non_claims (이 표가 주장하지 않는 것)", ""]
    lines += [f"- {c}" for c in payload["non_claims"]]
    return "\n".join(lines) + "\n"


NON_CLAIMS = [
    "오라클은 학습이 아니라 **커닝**이다. 정책에 실행기의 정답을 미리 보여준다. "
    "따라서 이 수치는 '학습이 이렇게 된다'가 아니라 **학습이 도달할 수 있는 상한**이다. "
    "실제 학습 정책은 반드시 이보다 나쁘다.",
    "대리모델(AnalyticExecutor/FailingExecutor)의 붕괴 처리는 진짜 입자 물리가 아니다. "
    "안식각 초과 셀을 기하로 판정하고 이웃과 높이를 나눠 가지는 완화 규칙일 뿐이다. "
    "최종 확인은 DEME 과 실물 스쿱이 한다.",
    "PP 물성 미실측. angle_of_repose 32도는 임시값이며 어떤 수치도 "
    "폴리프로필렌 값으로 인용해서는 안 된다.",
    "시드 30개는 같은 능선 생성기(synthetic_pile)의 미세 변형이다. "
    "장면 다양성(더미 개수, 형상 종류, 용기 위치)을 재는 표본이 아니다. "
    "따라서 p 값은 '이 더미 계열 안에서의 유의성'이다.",
    "후보를 높이 상위 top_k 개로 잘라서 평가한다 -> 진짜 상한은 이보다 높을 수 있다.",
    "시드 수는 실행 **전에** 30개(457~486)로 고정했다. p 값이 0.05 를 넘겼다고 "
    "시드를 더 늘려 다시 계산하면 그것은 p-해킹이다. 표본을 늘리려면 목표 n 을 "
    "먼저 선언하고 전량을 새로 돌려 그 결과만 보고해야 한다.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="p41_oracle.json 경로")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--subsample", type=int, default=0,
                    help="앞에서부터 N개 시드만 사용 (게이트 회귀 주입용, 0=전체)")
    a = ap.parse_args()

    src = Path(a.input)
    table = load_rows(src)
    seeds = sorted(set.intersection(*(set(v) for v in table.values())))
    if a.subsample > 0:
        seeds = seeds[: a.subsample]
        table = {p: {s: r for s, r in rows.items() if s in seeds}
                 for p, rows in table.items()}

    order = ["greedy_high", "oracle_1", "oracle_2"]
    payload: dict[str, Any] = {
        "artifact": "P41B_ORACLE_STATS_V1",
        "question": "oracle_2 > oracle_1 이 우연인가? (n=%d)" % len(seeds),
        "source": str(src),
        "seeds": seeds,
        "subsample": a.subsample,
        "per_policy": {name: _summary([table[name][s] for s in seeds])
                       for name in order if name in table},
        "sign_tests": [
            sign_test(table, "oracle_2", "oracle_1"),
            sign_test(table, "oracle_2", "greedy_high"),
            sign_test(table, "oracle_1", "greedy_high"),
        ],
        "mcnemar_completion": [
            mcnemar_completion(table, "oracle_2", "oracle_1"),
            mcnemar_completion(table, "oracle_2", "greedy_high"),
            mcnemar_completion(table, "oracle_1", "greedy_high"),
        ],
        "non_claims": NON_CLAIMS,
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p41b_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "p41b_stats.md").write_text(_markdown(payload), encoding="utf-8")

    for name, s in payload["per_policy"].items():
        block = s["scoops_completed_only"]
        mean = "—" if block["mean"] is None else f"{block['mean']:.2f}"
        std = "—" if block["std"] is None else f"{block['std']:.2f}"
        print(f"{name:12s} 완주 {s['n_completed']:2d}/{s['n_seeds']:2d} · "
              f"스쿱(완주분) {mean}±{std} · 실패율 {s['failure_rate']['mean']}")
    for t in payload["sign_tests"]:
        p = "—" if t["p_one_sided"] is None else f"{t['p_one_sided']:.5g}"
        print(f"부호검정 {t['a']:10s} > {t['b']:12s}  승{t['wins_a']:2d} "
              f"패{t['wins_b']:2d} 무{t['ties']:2d}  p={p}")
    print(f"-> {out/'p41b_stats.json'}")
    print(f"-> {out/'p41b_stats.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
