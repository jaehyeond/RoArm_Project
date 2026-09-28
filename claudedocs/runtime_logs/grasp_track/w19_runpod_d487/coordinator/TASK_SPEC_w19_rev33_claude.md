# TASK_SPEC — rev33 후보: producer 메타데이터·영수증 보완(물리·제어·보호선·저장 프레임 규약 기본값 불변, CPU 검증만) — Claude claude-opus-5 high
작성 2026-09-18, 코디네이터 = 메인. 계약 정본 = 이 파일. 1단계 규약 검사 FAIL 6건 중 producer 측 4건(②④ 메타데이터 선언 누락, ③ 영수증 stdout/stderr 해시 없음, ⑥ 산출 매니페스트 부재)을 **다음 실행부터** 채우는 새 revision 사본. **GPU/DEME 실행 0** — 스모크는 별도 GO.
## Target
- 원본(읽기 전용): rev32 `/home/cgxr/orca/workspaces/RoArm_Project/w16-profiling/claudedocs/runtime_logs/grasp_track/w16_profile_d486/rev32/`(src 24·JSON 4·REVISION_PIN·checks) · rev31 `.../w14_w13_raw_repair_d484/repair_20260916_01/rev31/src/inventory_geometry.py`(`semantics_metadata()` 등 선언 함수) · 러너 `.../w19_runpod_d487/pod/run_w19.py` · 1단계 `schema_check/RAW_SCHEMA_CHECK_W19.json`(FAIL 항목의 required 필드 정의) · 규약 `RAW_SCHEMA_REQUIRED.md` + ERRATUM_01~04(필드 정의 줄 인용).
- 출력(이 worktree 안에만): `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/rev33_20260918/{rev33/(src 전체 사본 + REVISION_PIN.json + DIFF_rev32_to_rev33_src.patch + checks/ast_scope_check.json), runner/run_w19_v2.py + DIFF, tests/(unittest, RESULTS json), REPORT_rev33.md, manifest.json}`.
## Change
1. rev33 = rev32 바이트 사본 + 변경: (a) raw 메타데이터에 `classify_floor_rule`·`classify_contract_version` **exact 문자열**(등록 계약 `RAW_SCHEMA_REQUIRED_ERRATUM_04.md` 의 문구, rev31 `semantics_metadata()` 재사용) 기록, (b) `time_mapping_abs_s`·`geometry_epsilon_m` 필드(규약 정의 줄 인용, 값은 기존 코드의 실제 매핑/epsilon 에서 도출 — 새 상수 발명 금지), (c) 종료 시 run_dir 산출 매니페스트(`manifest.json`, 전 산출 sha256) 기록.
2. 러너 v2 = `run_w19.py` + 종료 시 stdout/stderr sha256 을 EXECUTION_RECEIPT 에 기록 + cap 값과 규약 상한(32,400 s)을 함께 기록(판정은 하지 않음).
3. **옵션(기본 OFF, 사용자 결정 ②(a) 대기)**: `--settle-window-frame-dt-s <s>` — 지정 시 마지막 정착 창 안에서만 입자 프레임 간격을 그 값(예 0.05 s)으로 촘촘히 저장. 기본값 None 이면 rev32 와 저장 규약 동일. 구현하되 기본 경로 무변경을 AST·단위 테스트로 증명.
4. AST 범위 검사(W16 `checks/ast_scope_check.py` 방식): 물리 파라미터·제어·보호선·분류식 함수 무변경 증명. `params_w13.json`·`numeric_inputs.json`·`criteria.json` 바이트 동일.
5. CPU 단위 테스트: 메타데이터 선언 문자열 exact 일치, 필드 존재, 러너 영수증 해시 기록, 옵션 OFF 시 프레임 스케줄 불변(가짜 타임라인로 검증).
6. REPORT_rev33.md(한국어): 무엇/왜(FAIL 4건 ↔ 변경 매핑) → 절차 → 검증 결과 → GPU 스모크(300알 settle)는 별도 GO 라고 명시.
## Constraints
GPU/DEME 0·설치 0·메인 repo/상태 원장/다른 worktree 읽기만·물성/형상/경로/문 속도/보호선/dt/cd_update_freq 변경 0. 총 2 h 상한.
## Observable acceptance
DIFF·AST 검사 PASS·단위 테스트 전부 PASS·REVISION_PIN·REPORT·manifest·worker_done(--report-path, 3문장: 변경 파일 수·AST 결과·옵션 기본값).
