# TASK_SPEC — W19 설계 브리핑 2건 (문서만, 구현 금지) — Claude claude-opus-5 high

작성 2026-09-17 15:1x KST, 코디네이터 = 메인 세션(Claude Fable 5.1). 이 파일이 계약 정본이다.

## Target
worktree 안에 새 폴더 `claudedocs/research/design_briefs_20260917/` 를 만들고 아래 3파일만 쓴다.
1. `DOOR_SEAM_CASE_DESIGN.md` — W18 이 연 다음 물리 변수 "문 닫힘/이음새 간격" case 설계안.
2. `PARTICLE_COUNT_COST_CASE_DESIGN.md` — W16 이 연 다음 비용 변수 "알 개수(더미 크기/관심 영역)" case 설계안.
3. `EVIDENCE_CHECK.json` — 두 문서에서 인용한 모든 `파일:줄` 에 대해 실제로 실행한 확인 명령(grep/sed -n)과 그 출력 첫 줄.

## Change (각 문서에 반드시 들어갈 절)
공통 순서: (1) 무엇/왜 — 관측된 현상과 가설, (2) **이번 case 의 신규 변수 = 정확히 하나**(후보가 여럿이면 후보별로 나누고 "사용자 결정 필요" 로 표시), (3) 고정하는 것 전부(물성 E 5e6·mu 0.45·Crr 0.06·CoR 0.3·밀도 905, 형상 S1 v1 STL, 알 20,000, 경로, 문 속도 22.5 °/s, 보호선 pinch 3 N·pop 5/20 m/s, dt 1 µs, cd_update_freq, seed 460), (4) 비교 셀 표(값·근거·상한 시간), (5) **사전 등록 판정 기준**(무엇을 재면 PASS/FAIL 인지, 사후 허용값 금지), (6) 비용·시간 추정 — W16 단계별 "물리 1초당 실제 경과 시간" 표를 근거로 부분 실행(settle→reclose 또는 →transport) 시간을 계산, (7) 실물 제약과의 연결(서보 1.96 N·m×0.9, 물림 보호 3 N, D481 닫힘 토크 900/잠김 2.5, 문 각 분해능), (8) **구현이 필요해질 파일·함수·증거 파일 목록**(파일:줄 인용, 구현은 하지 않음), (9) 위험·비주장·다음 승인 경계.

문서 1 특이사항: W18 의 권고(같은 seed/dt/물성/초기상태/공구 경로/운반 속도 고정, 현재 기록값 3.55°/7.01 mm vs 이음새 < 펠릿 두께 2.50 mm 인 닫힘 조건 한 쌍, PF107 동일 코호트 잔류 곡선·이탈 방향 비교)을 그대로 셀 설계의 골격으로 쓴다. 닫힘을 더 시키는 방법 후보 (a) 재닫기 정지 규칙(제어 목표) 변경, (b) 문 립/이음새 기구 변경(새 STL → 실물 재출력 필요), (c) 잔여 각 하한 — 각각 시뮬 코드에서 어디를 바꿔야 하는지 `sim_w13_full_cycle.py`/`sim_deme_scoop_s1.py` 의 줄을 찾아 적는다. 재닫기 중 pinch_guard 가 먼저 걸려 더 못 닫는 경우의 처리도 설계에 포함(완화 금지 원칙과 충돌하므로 "사용자 결정").

문서 2 특이사항: W16 결론(비용 = 더미 상시 잠재 접촉쌍 ~20만, 손잡이 = 알 개수) 을 출발점으로, 알 개수를 줄이는 방식 후보 (i) 관심 영역(ROI) 절단, (ii) 더미 생성기 파라미터로 더 작은 슬랩 생성(`pellet-model` worktree 의 pile 생성 스크립트·params 를 찾아 인용), (iii) 알 굵기 변경은 **형상 변수라 제외** 를 비교하고, `--max-particles` 가 `numeric_inputs.json` 도메인 고정과 충돌해 fail-closed 되는 지점(`sim_w13_full_cycle.py:236~262` 부근, `bridge_inputs.py`)과 새 증거 파일을 만드는 절차(감사 worktree `w13-cycle-audit/.../audit/SOURCE_EVIDENCE*.md` 의 도메인 재구성 규칙)를 정리한다. 비용 곡선 셀(예: 20k/10k/5k, settle→reclose 부분 실행)과 지표(잠재 접촉쌍·물리 1초당 실제 경과 시간·포획량 — 포획량이 바뀌면 절단이 물리를 바꿨다는 신호)를 사전 등록한다.

## 읽을 근거 (절대경로; 필요한 줄만 읽고 인용은 grep 으로 확인)
- W18: `/home/cgxr/orca/workspaces/RoArm_Project/w18-cohort-cause/claudedocs/runtime_logs/grasp_track/w18_cohort_cause_d484/analysis_01/{REPORT_w18.md,COHORT_CAUSE_01.json}`
- W16: `/home/cgxr/orca/workspaces/RoArm_Project/w16-profiling/claudedocs/runtime_logs/grasp_track/w16_profile_d486/{REPORT_w16.md,analysis/,rev32/params_w13.json,rev32/numeric_inputs.json,rev32/src/sim_w13_full_cycle.py,rev32/src/bridge_inputs.py}`
- 동결 메인 소스: `/home/cgxr/Documents/Robotics/RoArm_Project/sim_deme_scoop_s1.py`(S1 셸 생성·문 힌지·서보 정지 모델), `hw_s1_manual.py`(실물 문 명령 규약)
- 규칙/실물: `/home/cgxr/Documents/Robotics/RoArm_Project/docs/reference/hardware.md`(말미 그리퍼 규약), `docs/reference/servo_pid_st3215.md`, `claudedocs/DECISIONS.md` D479(:30084)·D481(:30140)·D486(:30197)·D487(:30205)·D488(:30213) 앵커 줄만
- 실행안: `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/research/dt_expansion_plan_20260916/DT_EXPANSION_PLAN.md` §5·§6·§11
- 더미 생성: `/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/`(생성 스크립트/params/REPORT 를 찾아 인용)
- `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/BACKLOG.md` 끝부분(중복 등재 금지 — 이미 있는 후보는 참조만)

## Constraints (위반 = 실패)
- 코드 변경 0, 물리/GPU 실행 0, 설치 0, 실물 0, commit/push 0.
- 상태 원장(`START_HERE.md`, `claudedocs/{DECISIONS*.md,EXPERIMENT_LEDGER.md,LEDGER_RECENT.md,BACKLOG.md,relay/}`) 과 다른 worktree 파일은 **읽기만**. 편집은 위 새 폴더 3파일뿐.
- 한국어. 영어 용어는 첫 사용 시 풀이. "벽시계" 라는 표현 대신 **"실제 경과 시간(wall-clock time)"** 을 쓴다. 수치는 반드시 출처 경로와 함께.
- "없다/최초" 류 주장 금지(HARD RULE #4). 사후 합격선 금지. 문서 각 ≤ 300줄.
- 막히면 preamble 의 `ask` 로 코디네이터에게 묻는다(로컬 질문 TUI 금지).

## Ownership
이 워커만 `claudedocs/research/design_briefs_20260917/` 를 쓴다. 메인이 상태 원장·relay 를 소유한다.

## Observable acceptance
1. 3파일 존재, 각 md 에 위 (1)~(9) 절 제목 존재, "사용자 결정 필요" 항목이 명시돼 있음.
2. `EVIDENCE_CHECK.json` 의 각 항목: `{"cite": "<파일>:<줄>", "cmd": "<실행한 명령>", "first_line": "<출력 첫 줄>"}` — 두 문서의 모든 파일:줄 인용이 여기 있음.
3. `worker_done` 에 `--report-path` 로 두 md 의 절대경로, `--files-modified` 3파일, 3문장 요약(각 문서의 유일 신규 변수 후보와 사용자 결정 지점).
