#!/usr/bin/env python3
"""교수님 지적 재검: 실패가 있으면 greedy_high 가 여전히 이기나?

배경 (2026-09-01, 75th):
  교수님 = "히트맵 보고 매번 다시 폐루프로 집을 거면 학습이 왜 필요하냐.
            깊이 카메라로 내려다보고 제일 높은 곳을 퍼면 되지 않느냐."
  P2b 결과가 그 지적을 뒷받침했다 — greedy_high 13 스쿱으로 최소.

그런데 P2b 의 `AnalyticExecutor` 는 **실패율이 4정책 전부 0.000** 이다.
어디를 퍼든 항상 같은 크기 덩어리가 성공적으로 제거되는 모델이라
"높은 곳을 퍼면 많이 나온다"가 자명해진다 — 답이 모델에 심어져 있다.

이 스크립트는 **기하로 계산되는 물리적 실패**를 넣는다. 임의 확률이 아니다:
  (a) 붕괴  — 목표 셀 국소 경사가 안식각을 넘으면 재료가 담기지 않고 흘러내린다
  (b) 빈그랩 — 발자국 안 평균 재료 깊이가 절삭 깊이의 일정 비율에 못 미치면
              그랩이 바닥을 긁고 거의 빈 채로 올라온다
둘 다 heightmap 만으로 예측 가능하다 — **그것이 요점이다.**
예측하는 정책은 피할 수 있고, 지금-가장-높은-곳만 보는 정책은 못 피한다.

🔴 가설: 더미의 최고점은 대개 **가장 가파른 곳(꼭대기)** 이므로
   greedy_high 가 (a) 에 체계적으로 걸린다.

원본 p39 는 건드리지 않는다 (P2b 산출물·게이트 보존, forward-only).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p39_decision_loop as P39  # noqa: E402


class FailingExecutor(P39.AnalyticExecutor):
    """AnalyticExecutor + 기하 기반 실패 판정.

    실패하면 재료를 제거하지 않고 사이클 시간만 소모한다 —
    실물에서 헛집으면 시간은 그대로 쓰고 얻는 게 없는 것과 같다.
    """

    def __init__(self, config=None, *, avalanche_margin_deg: float = 0.0,
                 min_fill_fraction: float = 0.55, failure_time_fraction: float = 1.0):
        super().__init__(config)
        self.avalanche_margin_deg = float(avalanche_margin_deg)
        self.min_fill_fraction = float(min_fill_fraction)
        self.failure_time_fraction = float(failure_time_fraction)
        self.failure_log: list[dict] = []

    # --- 기하 진단 ------------------------------------------------------
    def _local_slope_deg(self, action) -> float:
        """목표 셀 주변 최대 경사(도). 꼭대기·급사면 판정용.

        p39 에 xy->ij 헬퍼가 없으므로 _cell_action 과 같은 방식으로
        cell_centers() 최근접 셀을 찾는다.
        """
        height, template = self._require_ready()
        centers = template.spec.cell_centers()
        d2 = (centers[..., 0] - action.x_m) ** 2 + (centers[..., 1] - action.y_m) ** 2
        i, j = np.unravel_index(int(np.argmin(d2)), height.shape)
        cell = template.spec.cell_m
        n_i, n_j = height.shape
        best = 0.0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = int(i) + di, int(j) + dj
            if 0 <= a < n_i and 0 <= b < n_j and template.valid[a, b]:
                drop = abs(float(height[i, j]) - float(height[a, b]))
                best = max(best, math.degrees(math.atan2(drop, cell)))
        return best

    def _footprint_mean_depth_m(self, action) -> float:
        """발자국 안 평균 재료 깊이. 얇으면 그랩이 바닥을 긁는다."""
        cut = self._footprint_cut(action)
        mask = cut > 0.0
        if not mask.any():
            return 0.0
        height, _ = self._require_ready()
        return float(height[mask].mean())

    def _failure_reason(self, action) -> tuple[str | None, float, float]:
        """실행하지 않고 실패 여부만 판정한다 (오라클 선행탐색이 재사용).

        같은 판정을 execute() 와 오라클이 각자 구현하면 갈라진다 —
        한 곳에만 둔다.
        """
        slope = self._local_slope_deg(action)
        depth = self._footprint_mean_depth_m(action)
        repose = self.config.angle_of_repose_deg + self.avalanche_margin_deg
        need = self.config.cut_depth_m * self.min_fill_fraction
        if slope > repose:
            return "avalanche", slope, depth      # 급사면 -> 흘러내려 안 담김
        if depth < need:
            return "empty_grab", slope, depth     # 얇은 층 -> 바닥 긁고 빈 채로
        return None, slope, depth

    def _would_fail(self, action) -> bool:
        return self._failure_reason(action)[0] is not None

    # --- 실행 -----------------------------------------------------------
    def execute(self, action):
        P39.validate_action(action)
        reason, slope, depth = self._failure_reason(action)
        repose = self.config.angle_of_repose_deg + self.avalanche_margin_deg
        need = self.config.cut_depth_m * self.min_fill_fraction

        if reason is None:
            return super().execute(action)

        # 실패: 재료 제거 0, 시간은 소모
        self.failure_log.append({
            "reason": reason, "x_m": action.x_m, "y_m": action.y_m,
            "local_slope_deg": round(slope, 2), "repose_limit_deg": round(repose, 2),
            "footprint_mean_depth_m": round(depth, 5), "required_depth_m": round(need, 5),
        })
        height, template = self._require_ready()
        cycle = self.config.fixed_cycle_time_s * self.failure_time_fraction
        travel = self.config.fixed_transfer_distance_m / self.config.travel_speed_m_s
        return P39.ExecutionResult(
            observation=P39._heightmap_copy(template, height=height.copy()),
            scooped_mass_kg=0.0,
            elapsed_time_s=float(cycle + travel),
            failed=True,
            travel_distance_m=float(self.config.fixed_transfer_distance_m),
            removed_volume_m3=0.0,
            actual_removed_centroid_xy_m=(float(action.x_m), float(action.y_m)),
            failure_reason=reason,
        )


class AvoidFailedWrapper(P39.ScoopPolicy):
    """같은 자리를 무한 재시도하지 않도록 실패 셀을 마스킹하는 최소 장치.

    이게 없으면 어떤 정책이든 자기가 만든 절벽을 계속 다시 고르다
    consecutive_failure_limit 에 걸려 **완주 자체를 못 한다.**
    비교를 공정하게 하려고 네 정책 모두에 동일하게 씌운다 —
    학습이 아니라 "방금 실패한 곳은 피한다"는 규칙 한 줄이다.
    """

    def __init__(self, inner: P39.ScoopPolicy, executor: "FailingExecutor",
                 radius_m: float = 0.008) -> None:
        self.inner = inner
        self.name = inner.name
        self.executor = executor
        self.radius_m = float(radius_m)

    def select_action(self, observation, context):
        banned = [(f["x_m"], f["y_m"]) for f in self.executor.failure_log]
        if not banned:
            return self.inner.select_action(observation, context)
        # 관측 계약: 무효 셀은 높이 0.0 이어야 한다. valid 만 끄면 계약 위반이다.
        # Heightmap 은 counts 까지 요구한다 -> 직접 생성하지 말고 _heightmap_copy 를 쓴다
        h = observation.height.copy()
        centers = observation.spec.cell_centers()
        ban_all = np.zeros(h.shape, dtype=bool)
        for bx, by in banned:
            d2 = (centers[..., 0] - bx) ** 2 + (centers[..., 1] - by) ** 2
            ban_all |= d2 <= self.radius_m ** 2
        h[ban_all] = 0.0
        # 🔴 _heightmap_copy 의 ascontiguousarray 는 이미 연속·정확 dtype 이면
        #    **같은 배열을 돌려준다**. 제자리 변경하면 원본 관측이 오염되고
        #    run_episode 의 최종 관측이 계약 위반으로 죽는다. 명시적으로 복사한다.
        v = observation.valid.copy()
        v[ban_all] = False
        masked = P39.Heightmap(
            height=np.ascontiguousarray(h, dtype=np.float32),
            valid=np.ascontiguousarray(v, dtype=np.bool_),
            counts=np.ascontiguousarray(observation.counts.copy(), dtype=np.int32),
            spec=observation.spec, meta=dict(observation.meta))
        # 폴백은 valid 가 아니라 **고를 수 있는 셀(active)** 기준이어야 한다.
        # valid 가 남아도 전부 임계 미만이면 정책이 policy_error 로 죽는다.
        if P39._active_cells(masked, 0.0005).size == 0:
            return self.inner.select_action(observation, context)
        return self.inner.select_action(masked, context)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-fill", type=float, default=0.55)
    ap.add_argument("--avalanche-margin-deg", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=457)
    ap.add_argument("--avoid-failed", action="store_true",
                    help="실패한 셀 주변을 다음 선택에서 제외 (네 정책 공통)")
    ap.add_argument("--avoid-radius-m", type=float, default=0.008)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = P39.EpisodeConfig()
    rows, logs = [], {}
    for name in P39.POLICY_NAMES:
        pile = P39.synthetic_pile(a.seed)
        ex = FailingExecutor(min_fill_fraction=a.min_fill,
                             avalanche_margin_deg=a.avalanche_margin_deg)
        pol = P39.build_rule_policy(name, threshold_m=cfg.actionable_height_m, seed=a.seed)
        if a.avoid_failed:
            pol = AvoidFailedWrapper(pol, ex, radius_m=a.avoid_radius_m)
        run = P39.run_episode(pile, pol, ex, cfg, seed=a.seed)
        steps = run.records
        n_fail = sum(1 for r in steps if r.failed)
        rows.append({
            "policy": name,
            "total_scoops": len(steps),
            "total_time_s": round(sum(r.elapsed_time_s for r in steps), 1),
            "failure_rate": round(n_fail / max(len(steps), 1), 3),
            "failures": n_fail,
            "total_distance_m": round(sum(r.travel_distance_m for r in steps), 2),
            "target_reached": run.terminal_reason == "target_reached",
            "terminal_reason": run.terminal_reason,
        })
        logs[name] = ex.failure_log
    rows.sort(key=lambda r: (not r["target_reached"], r["total_scoops"]))

    payload = {
        "artifact": "P40_FAILURE_GREEDY_REEXAM_V1",
        "question": "실패가 있으면 greedy_high(제일 높은 곳)가 여전히 이기나?",
        "failure_model": {
            "avalanche": "목표 셀 국소 경사 > 안식각 + margin -> 흘러내려 미포획",
            "empty_grab": f"발자국 평균 깊이 < cut_depth x {a.min_fill} -> 바닥 긁음",
            "cost": "실패 시 재료 제거 0, 사이클+이송 시간은 소모",
            "min_fill_fraction": a.min_fill,
            "avoid_failed": a.avoid_failed,
            "avoid_radius_m": a.avoid_radius_m,
            "avalanche_margin_deg": a.avalanche_margin_deg,
        },
        "comparison": rows,
        "failure_detail": logs,
        "non_claims": [
            "실패 모델은 heightmap 기하만 쓴다. 실물 실패(도구 충돌, 입자 끼임, "
            "서보 스톨)는 포함되지 않는다.",
            "AnalyticExecutor 의 나머지 가정(고정 발자국·고정 사이클 시간·"
            "완화 4회)은 그대로다. 이 실험이 바꾼 것은 **실패 유무 하나**다.",
            "PP 물성 미실측. angle_of_repose 32도는 임시값이며 실측 시 결과가 바뀐다.",
        ],
    }
    (out / "p40_failure_reexam.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=P39._json_default),
        encoding="utf-8")

    print(f"실패 모델: 경사>{32.0 + a.avalanche_margin_deg:.0f}도 붕괴 · "
          f"깊이<{a.min_fill}x절삭 빈그랩")
    print(f"{'정책':12s} {'스쿱':>5s} {'시간s':>8s} {'실패율':>7s} {'완주':>6s}  종료사유")
    for r in rows:
        print(f"{r['policy']:12s} {r['total_scoops']:5d} {r['total_time_s']:8.1f} "
              f"{r['failure_rate']:7.3f} {str(r['target_reached']):>6s}  {r['terminal_reason']}")
    print(f"-> {out/'p40_failure_reexam.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
