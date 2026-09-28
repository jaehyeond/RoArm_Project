# TASK_SPEC — W20 rev33b: 새 revision + 새 criteria 파일 (CPU 전용, GPU 0)

작성 2026-09-19, 코디네이터 = Claude 메인 세션(RoArm_Project master). 워커 = Claude `claude-opus-5`.
이 과제서는 자기완결적이다. 여기 없는 행동은 하지 않는다.

## 0. 이 과제의 위치 (코디네이터 교차 검증 결과가 바꾼 것)

사용자 결정 1번은 원래 "정착 기록 간격 규약 문구를 프레임 0.1 s 에 맞춰 개정"이 권고였다.
코디네이터가 동결 `criteria.json` 을 읽고 **그 권고를 철회**했다. 이유는 그 파일 안에 있다:

- `policy.no_threshold_change_after_outcomes` — severity **`hard_fail`**.
  "결과를 본 뒤 임계를 조정하지 않는다." boundary: **"변경이 필요하면 새 revision + 새 criteria 파일
  + 코디네이터 검토."**
- `delivery.settlement_window_UNCALIBRATED` — severity **`uncalibrated_report_only`**(hard_fail 아님).
  value `{window_s 0.25, frame_dt_s 0.05, speed_max_m_s 0.005, move_max_m 0.001}`.
  boundary: "창 프레임이 부족하면 exact settled 를 주장하지 않고 **하한/상한과 누락 증명**을 보고한다."
  scientific_limitation: "**pass 를 만들기 위한 추가 대기 금지**."

즉 W19 A 의 기록(정착 창 5프레임, 최대 간격 0.100025 s)을 이미 본 상태에서 0.05 를 0.1 로 **푸는 것**은
정면 금지다. 반면 **새 revision + 새 criteria 파일**로 앞으로의 실행에만 적용하는 것은
그 정책이 스스로 지정한 합법 경로다. 이 과제가 그 경로다.

그리고 방향이 중요하다. 새 값 **0.046 s 는 0.05 보다 느슨한 게 아니라 촘촘하다**(기록을 더 자주 남긴다).
0.05 를 그대로 요청하면 4 ms 동기 격자에서 실제 간격이 **0.052 s** 가 되어 규약을 다시 넘긴다는 것이
rev33 테스트로 이미 고정돼 있다. 그래서 0.046 = 0.05 − dt_sync 다. **완화가 아니라 달성 가능하게
만드는 정정**이며, 보고서에 이 논리를 그대로 적어야 한다.

**용어(처음 나올 때 풀이)**
- **정착 창(settlement window)** — 배출이 끝난 뒤 알이 멈췄는지 보는 0.25초 구간.
- **기록 간격(frame cadence)** — 시뮬이 알 위치를 파일에 남기는 주기. 물리 계산 주기와 별개다.
- **동기 격자(sync grid)** — 시뮬이 물리를 4 ms 단위로 끊어 도는 구조. 저장 시점은 이 격자에만 놓인다.
- **AST 범위 검사** — 코드의 구문 트리를 비교해 "동결된 함수가 한 글자도 안 바뀌었다"를 증명하는 검사.
- **실제 경과 시간(wall-clock time)** — 사람이 시계로 재는 시간.

## 1. Target

**읽기 전용 입력 (절대 수정 금지)**
- rev32 동결 사본: `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/rev32_frozen_copy/`
  · `criteria.json` sha256 `71a7d23938a341c2689b570fa5a1e9bec74b7b3392f754ca55f61e7edd646196`
- rev33 산출 전체(외장 SSD 아님, 로컬 worktree):
  `/home/cgxr/orca/workspaces/RoArm_Project/w19-rev33/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/rev33_20260918/`
  · `rev33/`(src 24 .py, params/numeric_inputs/criteria/COMMANDS, REVISION_PIN, DIFF, checks/ast_scope_check)
  · `runner/run_w19_v2.py`, `tests/`, `REPORT_rev33.md`, `manifest.json`
  · 확인된 사실: **rev33 의 `criteria.json` 은 rev32 와 바이트 동일**(같은 sha `71a7d239…`).
    즉 rev33 은 아직 새 criteria 를 만들지 않았다. 그게 이 과제가 채울 구멍이다.
- W19 A 원자료(읽기 전용):
  `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/run_01/SUMMARY_A.json`,
  `EXECUTION_RECEIPT.json`

**쓰기 (이 worktree 안에만)**
`claudedocs/runtime_logs/grasp_track/w20_decisions_d492/rev33b/`

## 2. Change

### 2-1. rev33b = rev33 바이트 사본
rev33 전체를 `rev33b/` 로 복사한다. 복사 직후 **원본과 사본의 sha256 전수 대조**로 바이트 동일을 증명한다.

### 2-2. 새 criteria 파일 — 바꾸는 것은 **딱 한 필드**
`rev33b/criteria.json` 을 rev32/rev33 criteria 로부터 만들되, 아래 **한 필드만** 바꾼다.

| 대상 | 기존 | 새 값 | 근거(파일에 `origin` 으로 적을 문장) |
|---|---|---|---|
| `delivery.settlement_window_UNCALIBRATED.value.frame_dt_s` | `0.05` | `0.046` | "4 ms 동기 격자에서 0.05 s 를 요청하면 실제 간격이 0.052 s 로 규약 상한을 넘긴다(rev33 CPU 테스트로 고정). 0.046 = 0.05 − dt_sync 는 기록을 더 촘촘히 하는 **정정**이며 완화가 아니다. 이 값은 이 revision 부터 앞으로의 실행에만 적용되고 과거 실행에 소급하지 않는다." |

**그 외 모든 필드·임계·severity·operator 는 바이트 수준에서 그대로 둔다.** 특히:
- `delivery.settlement_window_UNCALIBRATED` 의 `window_s 0.25`, `speed_max_m_s 0.005`,
  `move_max_m 0.001`, severity `uncalibrated_report_only` — **불변**.
- `runner.physics_wall_cap_s` = **32400 그대로 둔다.** 올리지 않는다.
  (W19 A 는 영수증 `cap_s = 43200.0` 으로 돌았고 실제 소요는 20,977.836 s 였다. 즉 실제 위반은 없었다.
  그리고 앞으로 계획된 모든 셀 — 설계 ② P1/P2 상한 13,204 s, 설계 ① C1/C2 상한 16,934 s — 이 전부
  32,400 s 안쪽이다. **올릴 필요가 없다.** 이 판단 근거를 `CRITERIA_DIFF.md` 에 적는다.
  올리는 것은 사용자의 별도 명시 승인 사항이며 이 과제는 하지 않는다.)
- `policy.no_threshold_change_after_outcomes` — **불변**. 이 과제 자체가 그 정책이 지정한 경로다.

새 파일에는 revision 식별을 넣는다: 최상위에 `"revision": "rev33b"` 와
`"supersedes_criteria_sha256": "71a7d239…646196"`, `"retroactive": false` 를 명시한다.

### 2-3. 정착 창 옵션을 새 criteria 와 연결
rev33 의 `--settle-window-frame-dt-s` 옵션(기본 `None` = 꺼짐)이 **새 criteria 의 `frame_dt_s` 를
읽어** 동작하도록 최소 변경한다. 숫자 `0.046` 을 코드에 적어 넣지 말고 criteria 에서 읽는다
(rev33 이 `geometry_epsilon_m` 을 params 에서 읽은 것과 같은 방식 — `REPORT_rev33.md:61`).
옵션을 명시하지 않으면 동작은 **rev33 과 프레임 단위로 동일**해야 한다(기존 성질 보존).

### 2-4. 검사 재실행 + 새 검사 추가
- rev33 의 `checks/ast_scope_check.py` 를 rev33b 에 대해 재실행 → **PASS** 여야 한다.
- rev33 의 CPU 단위 테스트 30건을 rev33b 에 대해 재실행 → **30/30 PASS** 여야 한다.
- **새 테스트를 추가**한다(최소 4건):
  1. rev33b criteria 와 rev32 criteria 의 차이가 **정확히 `frame_dt_s` 한 필드**임을 재귀 비교로 증명.
  2. `runner.physics_wall_cap_s` 가 여전히 32400 임을 단언.
  3. 옵션 미지정 시 저장 프레임 스케줄이 rev33 과 동일함을 단언.
  4. 옵션 지정 시 정착 창에서 최대 간격이 `0.046 + 1e-9` 이하가 되는 스케줄이 나옴을 합성 입력으로 단언.
- rev32/rev33 원본의 sha256 이 작업 전후 **불변**임을 증명한다.

### 2-5. GPU 스모크 꾸러미 초안 (실행하지 않는다)
알 300개 스모크의 **정확한 명령문**, 상한, 성공 판정(rc 0 + 원자료 필수 필드 존재 + 새 메타데이터
선언 4건 기록 + 정착 창 옵션 동작), 산출 경로를 한 파일에 적는다. **실행은 사용자 GO 뒤 별도 과제다.**

## 3. Constraints

- **물리 실행 0 · GPU 0 · DEME 0 · 실물 0 · 설치 0 · commit/push 0.**
- 물성·형상·알 수·경로·문 속도·보호선·`timestep_s` 1e-06·`cd_update_freq`·seed 는 **손대지 않는다**.
  이 과제는 **기록 계약과 메타데이터만** 만진다. 물리 판정식은 한 줄도 바꾸지 않는다(AST 로 증명).
- rev32 동결 사본·rev33 원본·W19 원자료는 **읽기 전용**.
- `isaaclab` conda 환경에 **설치 금지**(핀 `numpy==1.26.0`·`psutil==5.9.8` 보호, D326).
- 상태 원장(`START_HERE.md`, `claudedocs/{DECISIONS,DECISIONS_ACTIVE,EXPERIMENT_LEDGER,
  LEDGER_RECENT}.md`, `claudedocs/relay/`)에 **한 줄도 쓰지 않는다**. 코디네이터만 소유한다.
- 폴더는 forward-only. 기존 파일·폴더 이동·개명 금지.
- W19 A 의 과거 판정을 **소급해서 PASS 로 바꾸지 않는다**(D485 ⑤). 새 criteria 는 `retroactive: false`.
- "없다/최초" 류 주장 금지(HARD RULE #4).

## 4. Ownership

이 worktree 의 `claudedocs/runtime_logs/grasp_track/w20_decisions_d492/rev33b/` **안에서만** 파일을 만든다.

## 5. Observable acceptance

1. `PRESERVATION_BEFORE.json` / `PRESERVATION_AFTER.json` — 읽기 전용 입력 전체 sha256, **불일치 0**.
2. `rev33b/` — rev33 대비 바이트 대조표(`COPY_VERIFICATION.json`), 바뀐 파일 목록이 2-2·2-3 범위와 정확히 일치.
3. `rev33b/criteria.json` + `CRITERIA_DIFF.md` — 재귀 diff 출력, 바뀐 필드 1개, wall cap 유지 근거 문단 포함.
4. `checks/ast_scope_check.json` → PASS. `tests/RESULTS_rev33b_tests.json` → **34/34 이상 PASS**(기존 30 + 신규 ≥4).
5. `GPU_SMOKE_PACKAGE.md` — 2-5 내용(실행 0).
6. `REPORT_rev33b.md` — 한국어. 순서는 **관찰 가능한 절차 → 수치 → 근거 파일 → 한계·다음 승인**.
   영어 약어는 처음 나올 때 풀어 쓴다. 승격·성공 선언 금지("정본 승격은 GPU 스모크와 사용자 승인 뒤"라고 적는다).
7. `manifest.json` — 산출 전 파일의 경로·바이트·sha256.

## 6. 보고

`worker_done` 은 정확히 한 번, 3문장 요약 + `--outcome succeeded|failed` + 두 lifecycle ID.
`--report-path` 는 위 6번 파일의 절대경로.
