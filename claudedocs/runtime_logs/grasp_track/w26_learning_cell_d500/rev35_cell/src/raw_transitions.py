"""rev29 — `transition_sync_index` 파생 규칙 (순수 CPU · numpy).

규약 RAW_SCHEMA_REQUIRED.md:52: "`transition_sync_index` exactly equals dense phase-change indices".
즉 dense `sync_phase_code[i] != sync_phase_code[i-1]` 인 행 i (i>=1). 행 0 과 subphase 전환은 포함하지 않는다.
rev28 의 `enter()` 는 phase **또는** subphase 진입 + 행 0 을 기록해 11개가 아니라 25개를 남겼다
(REV28_PRODUCTION_PARTIAL_RAW_AUDIT_01 finding 6). 이 모듈은 그 규칙을 한 곳에 두고,
결함 재현용 legacy 규칙도 이름을 붙여 분리한다.
"""
import numpy as np

RULE = ("indices i>=1 with sync_phase_code[i] != sync_phase_code[i-1]; "
        "row 0 and subphase-only changes are excluded (RAW_SCHEMA_REQUIRED.md:52)")
LEGACY_REV28_RULE = "row 0 plus every row where (phase, subphase) differs from the previous row"


def phase_only_transition_indices(sync_phase_code):
    """규약 규칙. 반환 int64 1-D, 오름차순."""
    c = np.asarray(sync_phase_code).astype(np.int64).ravel()
    if c.size == 0:
        return np.zeros(0, np.int64)
    return (np.flatnonzero(c[1:] != c[:-1]) + 1).astype(np.int64)


def legacy_rev28_transition_indices(sync_phase_code, sync_subphase):
    """rev28 이 실제로 기록한 의미의 재현 — **결함 재현 전용**, 규약이 아니다."""
    c = np.asarray(sync_phase_code).astype(np.int64).ravel()
    s = np.asarray(sync_subphase).astype(str).ravel()
    if c.shape != s.shape:
        raise ValueError(f"phase/subphase 길이 불일치: {c.shape} vs {s.shape}")
    idx = [0] if c.size else []
    for i in range(1, c.size):
        if c[i] != c[i - 1] or s[i] != s[i - 1]:
            idx.append(i)
    return np.asarray(idx, np.int64)
