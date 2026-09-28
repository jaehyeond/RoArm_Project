# TASK_SPEC — W19 A 후처리 3단계: 독립 감사(1단계 회계·규약 검사 산출) — Codex gpt-5.6-sol high (불가 시 Claude claude-opus-5)
작성 2026-09-18, 코디네이터 = 메인. 계약 정본 = 이 파일. **NumPy-only**: 생산 모듈(`inventory_geometry`, `sim_deme_scoop_s1`, `w13_*`, scipy)·워커 검사기(`independent_check*.py`, `tools/`)를 import 하지 않는다.
## Target
- 감사 대상(읽기 전용): 1단계 산출 `/home/cgxr/orca/workspaces/RoArm_Project/w19-postprocess/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/postprocess_20260918/`(derived NPZ sha `0dd33a48…`, RAW_SCHEMA_CHECK_W19.json 27항목, SETTLEMENT_WINDOW_RECOMPUTE.json, W13_COMPARISON.json, PRESERVATION_*.json, REPORT_postprocess.md) · W19 원자료 `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/run_01/` · 규약 정본 `/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/{RAW_SCHEMA_REQUIRED.md,RAW_SCHEMA_REQUIRED_ERRATUM_01~04.md}` · 이전 감사 형식 `.../w14_w13_raw_repair_d484/repair_20260916_01/audit/W14_REV31_V2_INDEPENDENT_AUDIT_02.json`.
- 출력(이 worktree 안에만): `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/audit_20260918/{audit_w19_postprocess.py, W19_POSTPROCESS_INDEPENDENT_AUDIT_01.json, REPORT_audit.md, manifest.json}`.
## Change — 항목(사전 등록, 각 PASS/FAIL + 측정값 + 근거 파일:줄)
E1 원자료 sha 가 RETRIEVAL_RECEIPT·PRESERVATION_BEFORE/AFTER 와 일치. E2 규약 문구(RAW_SCHEMA_REQUIRED+ERRATUM_04 받침면·i−1)에서 **독자 구현**으로 288프레임 × 20,000 라벨을 재계산 → 워커 derived NPZ 와 불일치 수(기대 0) + 생산 `inventory_code` 와 불일치 수. E3 전환 11개·i−1 프레임·결정 14개 결속 재확인. E4 정착 창(0.25 s) 프레임 수·간격·안정 272·최대 속도·이동 재계산과 cadence 규약(≥6·≤0.05 s) 판정. E5 재닫기 코호트 292 의 최종 분포(6/152/4/73/0/57) 재현. E6 규약 검사 27항목의 PASS/FAIL 을 각각 독자 판정해 워커와 대조(불일치 항목 명시). E7 W13 대조표 수치 재현(W13 derived_v2_rev31 사용). E8 보고서가 승격 금지·비주장을 지키는지(문구 확인). E9 산출 매니페스트 sha 전수 대조.
## Constraints
CPU 전용·물리 0·설치 0·메인 repo/상태 원장/다른 worktree 읽기만·사후 완화 금지. 총 2 h 상한. 막히면 preamble `ask`.
## Observable acceptance
JSON(항목별 verdict·측정·근거) + REPORT_audit.md(한국어) + manifest + worker_done(--report-path, 3문장: 총 PASS/FAIL, 워커와 불일치 항목, 정착 cadence 판정).
