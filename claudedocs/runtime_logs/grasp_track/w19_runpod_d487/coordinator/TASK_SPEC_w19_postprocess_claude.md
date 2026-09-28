# TASK_SPEC — W19 A 후처리 1단계: rev31 규약 v2 파생 회계 + 원자료 규약 검사 (CPU 전용, 물리 0) — Claude claude-opus-5 high

작성 2026-09-18, 코디네이터 = 메인(Claude Fable 5.1). 사용자 승인 "A 후처리 step-by-step". 이 파일이 계약 정본. 2단계(Rerun/Isaac 재생, 로컬 GPU)·3단계(독립 감사)는 별도 dispatch.

## 배경(읽고 시작)
W19 A = RunPod 4090 에서 rev32(= rev31 규약 v2 + 진단 로그) 전체 사이클을 완주한 첫 원자료. 생산 기록값: rc0, 물리 24.807 s, 16,813 sync, 288 입자 프레임, 최종 재고 source 19,266 / receiving_bin 272 / tool 11 / spill 124 / in_flight 0 / ambiguous 327, HOME 오차 0.00006 mm, 정착 창 5프레임(cadence_ok false). **판정 승격 금지** — 이 과제는 "생산 회계를 독립적으로 재현·검사"하는 것이지 성공 선언이 아니다.

## Target
- **입력(읽기 전용, 메인 repo 절대경로)**: `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/run_01/{w13_cycle_seed460.npz(sha256 414633fbff8fc022…),w13_cycle_seed460.json,timeline_seed460.json,EXECUTION_RECEIPT.json,SUMMARY_A.json}` 과 `A_full_cycle/RETRIEVAL_RECEIPT.json`(전체 sha 목록). 더미 NPZ `/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz`(sha 659d6b0b…).
- **도구(읽기 전용, 복사해서 사용)**: W14 `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w14_w13_raw_repair_d484/repair_20260916_01/{rev29,rev30,rev31}/src/`(`derive_v2.py`·`inventory_geometry.py`·`raw_transitions.py` 등), `tests/independent_check.py`·`tests_v2/independent_check_v2.py`·`tests_v2/test_*.py`·`run_all.py`·`w14_paths.py`, `REPORT.md`(절차 참고). 규약 정본 `/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/{RAW_SCHEMA_REQUIRED.md,RAW_SCHEMA_REQUIRED_ERRATUM_01~04.md}`. W13 대조용 파생 `.../repair_20260916_01/derived_v2_rev31/`.
- **출력(이 worktree 안에만)**: `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/postprocess_20260918/` 아래 `{rev31_copy/(src 바이트 사본 + REVISION_PIN.json + 필요 시 DIFF_paths_only.patch), derived_v2_rev31/, tests/, schema_check/, diagnostics/, REPORT_postprocess.md, manifest.json, PRESERVATION_BEFORE.json, PRESERVATION_AFTER.json, inspection.json}`.

## Change (순서대로, 각 단계 산출·영수증)
1. **보존 영수증**: 시작 전 run_01 파일 전체 sha256 → `PRESERVATION_BEFORE.json`, `RETRIEVAL_RECEIPT.json` 과 대조(불일치 → 중단·보고). 끝에 `PRESERVATION_AFTER.json` 동일 확인.
2. **rev31 사본**: `rev31/src` 를 바이트 복사해 `rev31_copy/src/` + `REVISION_PIN.json`(W14 rev31 pin 의 sha 와 대조). 입력 경로가 하드코딩된 곳(예: `derive_v2.py:30 AUDIT_JSON`, `D29.DEFAULT_*`)은 **CLI 인자/래퍼로만** 우회한다. 분류·전환 **식은 한 글자도 바꾸지 않는다**. 부득이 파일을 고치면 `DIFF_paths_only.patch` + AST 범위 검사(W16 `checks/ast_scope_check.py` 방식)로 "경로/인자 외 변경 0" 을 증명한다.
3. **rev29 → rev31 파생 회계(W19 원자료)**: W14 절차 그대로 — rev29 파생(전환 11개·바닥식) → rev30/31 v2 규약(바닥=받침면, 전환 프레임 i−1) 파생 NPZ `derived_v2_rev31/w19_cycle_seed460_rev31_derived.npz` + sha. 산출: 프레임별 라벨(288×20,000), 결정 태그별 재고, 최종 재고, 전이 행렬, **vs 생산 기록 라벨 불일치 수(클래스별)**, 재닫기 종료 공구 내부 코호트(생산 292)와 그 코호트의 운반 중 잔류 곡선(프레임별 수)·최종 라벨.
4. **독립 검사기**: `independent_check_v2.py` 방식(생산 모듈·scipy·w13_kinematics import 금지, Hamilton 곱·AABB·groupby)을 W19 전 프레임에 적용 → rev31 파생과 0 불일치인지. unittest(`run_all.py`, pytest 없음)로 `tests/RESULTS_*.json`. 실제 경과 시간 기록.
5. **원자료 규약 검사**: `RAW_SCHEMA_REQUIRED.md` + ERRATUM_01~04 의 각 항목을 W19 raw 에 적용해 `schema_check/RAW_SCHEMA_CHECK_W19.json`(항목별 PASS/FAIL, 규약 문구 인용 파일:줄, 측정값). 최소 포함: phase-only 전환 수(기대 11)와 `transition_sync_index`, 전환 프레임 i−1 존재, 결정 태그 14개와 프레임 결속, 바닥식(구 최하단 > floor+margin 또는 v2 받침면 규약), 정착 창 규약(≥6프레임·≤0.05 s — **미충족이면 FAIL 로 그대로 기록**, 완화 금지), 속도 경고선(5 m/s 초과 sync 0)·강제정지(20 m/s) 미발동, 물리 파라미터가 `rev32/params_w13.json` 과 바이트 동일(W16 rev32 pin 참조), 입력 sha(inputs_sha256)와 매니페스트 일치.
6. **정착 창 재계산**: 마지막 0.25 s 창의 프레임 시각·간격, 용기 라벨 입자의 최대 속도·최대 이동을 NPZ 에서 재계산해 생산값(272 안정, 2.39e-5 m/s, 0.0033 mm)과 대조. cadence 규약 미충족 사실을 그대로 적는다.
7. **W13 대조표**: 결정 태그별 재고(W19 rev31 파생 vs W13 `derived_v2_rev31`), 최종 재고, 재닫기 코호트(292 vs 144)와 운반 손실 곡선. 원인 주장 금지(관측만).
8. **시각 진단(D324)**: 최종 프레임 용기 단면(라벨 색), 재닫기 종료 공동 단면, 정착 창 속도 히스토그램 — `diagnostics/*.png` 3장 이상, **실제로 열어 본 관찰**을 `inspection.json` 에 기록. RRD 는 생략하되 사유를 W14 와 같이 명시("라벨/인덱스 파생의 코드·배열·해시 감사, D341; 재생은 2단계").
9. **REPORT_postprocess.md**(한국어, 영어 용어 첫 사용 시 풀이, "벽시계" 대신 "실제 경과 시간(wall-clock)"): (1) 무엇/왜 (2) 절차(관측 가능한 순서) (3) 수치 + 출처 경로 (4) 일상어 판정 + 비주장 + 다음 승인 경계. **"전체 사이클 성공" 선언 금지** — 결론은 "생산 회계 재현 여부·규약 항목별 PASS/FAIL" 로만.
10. `manifest.json`(전 산출 sha256).

## Constraints (위반 = 실패)
- CPU 전용. GPU/DEME/Isaac/Rerun 실행 0, 새 물리 0, 설치 0(`roarm` env `/home/cgxr/miniconda3/envs/roarm/bin/python`, pytest 없음), 실물 0, commit/push 0.
- 메인 repo 파일·W14/W13/W16 worktree 파일·상태 원장(`START_HERE.md`, `claudedocs/{DECISIONS*.md,EXPERIMENT_LEDGER.md,LEDGER_RECENT.md,BACKLOG.md,relay/}`)은 **읽기만**. 편집은 위 출력 폴더뿐.
- 사후 허용값·완화 금지. 실패 항목은 그대로 기록. "없다/최초" 주장 금지. 인용은 파일:줄 확인 후.
- 전 프레임 회귀는 오래 걸릴 수 있다(W14 283프레임 504 s). 총 실제 경과 상한 2 h; 넘길 것 같으면 preamble 의 `ask` 로 코디네이터에게 묻는다(로컬 질문 TUI 금지).

## Ownership
이 워커만 `postprocess_20260918/` 를 쓴다. 메인이 상태 원장·relay 소유.

## Observable acceptance
1. `PRESERVATION_BEFORE/AFTER.json` 이 `RETRIEVAL_RECEIPT.json` 과 동일(0 불일치).
2. `derived_v2_rev31/*.npz` + sha, 불일치 수(클래스별)·최종 재고·코호트 곡선 JSON.
3. `tests/RESULTS_*.json` 독립 검사 0 불일치(또는 불일치 수와 원인 기록).
4. `schema_check/RAW_SCHEMA_CHECK_W19.json` 항목별 PASS/FAIL + 인용.
5. `diagnostics/*.png` ≥3 + `inspection.json` 실제 관찰.
6. `REPORT_postprocess.md`, `manifest.json`. `worker_done` 에 `--report-path REPORT_postprocess.md` `--files-modified`(주요 파일) 3문장 요약(재현 불일치 수·규약 FAIL 항목·정착 cadence 사실).
