"""저장 입자-프레임 **행 정체성** 판정의 단일 정본 (순수 CPU · numpy 만).

근거: `RAW_SCHEMA_REQUIRED_ERRATUM_03` + 감사 `RERUN_SOURCE_REVIEW_01`.

규약
----
* 정본 정체성 = **암묵 원시 행 인덱스** `particle_frame_row = 0..F-1`.
  순서대로, **누락·중복 없이** 소비자(Isaac 재생 / RRD)에 보존돼야 한다.
* `particle_frame_sync_index` 는 **비감소**이지 유일하지 않다. 결정 스냅샷이 **같은 완료 sync ·
  같은 source time** 에 행을 하나 더 붙이는 것은 **정상**이며, 그것을 이유로 행을 합치면 안 된다.
  sync 유일성은 행 정체성의 **대체물이 아니다**.
* 행별로 `source_sync_index == particle_frame_sync_index`,
  `source_time_s == particle_frame_t_s`,
  `source_phase_code == sync_phase_code[particle_frame_sync_index]` 가 **정확히** 성립해야 한다.
* 모든 결정 참조(`decision_particle_frame_index`)는 **보존된 행**을 가리켜야 한다.

왜 모듈인가
----------
exporter·renderer 두 곳에 사본을 두면 갈라진다(감사가 `contacts_logged` 에서 이미 같은 종류를
잡았다). 여기 한 곳에 두고 둘이 **같은 함수를 호출**하며, 부패 대조 시험도 이 함수를 직접 건드린다.
이것은 **구조 정합 검사**라 독립 oracle 이 필요한 과학 판정이 아니다.

주장하지 않는 것
--------------
행 정체성은 **기록 계약**이다. 배출 성공·정착·구동 가능성의 증거가 아니다.
이 검사는 옛 raw 를 수정하지 않으며, 스테핑·중복제거 정책을 바꾸지도 않는다.
"""
import numpy as np


def row_identity_report(*, particle_frame_row, particle_frame_sync_index, particle_frame_t_s,
                        sync_phase_code, source_sync_index, source_time_s, source_phase_code,
                        decision_particle_frame_index, n_raw, require_full_coverage=True):
    """보존된 행 목록을 원시 배열과 대조한다. 실패 사유 문자열 목록까지 돌려준다.

    `particle_frame_row`/`source_*` = **소비자가 실제로 보존한** 행들(순서 그대로).
    `particle_frame_sync_index`/`particle_frame_t_s`/`sync_phase_code` = 원시 배열.
    `require_full_coverage=False` 는 readiness fixture 전용(≤8 프레임 상한) — 이때도
    **순서·중복** 위반은 그대로 거절한다. 전수 대응만 면제된다.
    """
    rows = [int(v) for v in particle_frame_row]
    s_sync = [int(v) for v in source_sync_index]
    s_time = [float(v) for v in source_time_s]
    s_phase = [int(v) for v in source_phase_code]
    raw_sync = np.asarray(particle_frame_sync_index, np.int64)
    raw_t = np.asarray(particle_frame_t_s, float)
    ph = np.asarray(sync_phase_code, np.int64)
    n_raw = int(n_raw)
    if not (len(rows) == len(s_sync) == len(s_time) == len(s_phase)):
        raise ValueError(f"보존 행 배열 길이가 어긋난다: {len(rows)}/{len(s_sync)}/{len(s_time)}/{len(s_phase)}")

    seen = {}
    for v in rows:
        seen[v] = seen.get(v, 0) + 1
    dup = sorted(v for v, c in seen.items() if c > 1)
    missing = sorted(set(range(n_raw)) - set(rows))
    out_of_range = sorted(v for v in rows if not (0 <= v < n_raw))
    increasing = all(rows[i] > rows[i - 1] for i in range(1, len(rows)))
    exact_arange = rows == list(range(n_raw))

    # 같은 sync 를 공유하는 원시 행 — **정상**이며 보존 대상이다.
    share_sync = [int(i) for i in range(1, n_raw) if int(raw_sync[i]) == int(raw_sync[i - 1])]
    nondecreasing = all(int(raw_sync[i]) >= int(raw_sync[i - 1]) for i in range(1, n_raw))
    preserved_share_sync = [int(i) for i in share_sync if i in seen]

    # 행별 원시 일치 — 보존 행이 가리키는 **그 원시 행**과 비교한다(위치가 아니라 행 번호로).
    bad_sync, bad_time, bad_phase = [], [], []
    for k, r in enumerate(rows):
        if not (0 <= r < n_raw):
            continue
        if s_sync[k] != int(raw_sync[r]):
            bad_sync.append(r)
        if s_time[k] != float(raw_t[r]):
            bad_time.append(r)
        exp_ph = int(ph[int(raw_sync[r])]) if 0 <= int(raw_sync[r]) < len(ph) else None
        if s_phase[k] != exp_ph:
            bad_phase.append(r)

    dec = [int(v) for v in np.asarray(decision_particle_frame_index, np.int64)]
    dec_orphan = sorted({d for d in dec if d not in seen})

    rep = {
        "rule": ("particle_frame_row == arange(F), in order, no omission, no duplication; "
                 "sync index nondecreasing but NOT unique; same-sync decision rows are kept"),
        "n_raw": n_raw, "n_preserved": len(rows),
        "is_exact_arange": bool(exact_arange),
        "is_strictly_increasing": bool(increasing),
        "duplicated_rows": dup,
        "missing_rows": missing,
        "out_of_range_rows": out_of_range,
        "sync_index_nondecreasing": bool(nondecreasing),
        "rows_sharing_sync_with_previous": share_sync,
        "same_sync_rows_preserved": preserved_share_sync,
        "n_same_sync_rows_preserved": len(preserved_share_sync),
        "same_sync_rows_are_legitimate_and_preserved": bool(len(preserved_share_sync) == len(share_sync)),
        "rows_deduplicated": False,
        "rows_with_wrong_source_sync_index": bad_sync,
        "rows_with_wrong_source_time_s": bad_time,
        "rows_with_wrong_source_phase_code": bad_phase,
        "decision_rows_not_preserved": dec_orphan,
        "full_coverage_required": bool(require_full_coverage),
    }

    f = []
    if out_of_range:
        f.append(f"particle_frame_row out of range {out_of_range[:8]}")
    if dup:
        f.append(f"duplicated particle_frame_row {dup[:8]}")
    if not increasing:
        f.append("particle_frame_row is out of order")
    if not nondecreasing:
        f.append("raw particle_frame_sync_index is not nondecreasing")
    if bad_sync:
        f.append(f"source_sync_index != particle_frame_sync_index at rows {bad_sync[:8]}")
    if bad_time:
        f.append(f"source_time_s != particle_frame_t_s at rows {bad_time[:8]}")
    if bad_phase:
        f.append(f"source_phase_code != sync_phase_code[sync] at rows {bad_phase[:8]}")
    if require_full_coverage:
        if not exact_arange:
            f.append(f"particle_frame_row is not arange({n_raw}); missing={missing[:8]}")
        if dec_orphan:
            f.append(f"decision references unpreserved rows {dec_orphan[:8]}")
    rep["failures"] = f
    rep["all_ok"] = not f
    return rep
