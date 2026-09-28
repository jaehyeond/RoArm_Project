# 새 세션 재개 — W21 종료 뒤 갈림길 3택 (2026-09-23 작성)

W21(비용↔알 개수)이 끝났다. **알 5,000개는 결정적 정지로 실행 불가**이고, 부분 관측은 **비용 절감이
기대의 절반(4배 아닌 2.3배)** 임을 보였으며, **알 축소는 case ①(문 이음새) 측정을 구조적으로 오염**시킨다.
rev33b 스모크용 ROI-300 증거는 확보돼 **즉시 실행 가능**하다. 과거 GO/COMMANDS 는 재실행하지 않는다.
Claude Code 2.1.280(9/23 업데이트), **Opus 5.5 사용 가능** — 새 세션은 Opus 5.5 로 열 것.

## 반드시 읽기
1. `AGENTS.md`(자동 로드) → `START_HERE.md` → `claudedocs/DECISIONS_ACTIVE.md`(D485~D493)
   → `claudedocs/LEDGER_RECENT.md`(`:596~605`) → `claudedocs/relay/from_claude.md`(9/23).
2. `claudedocs/session_20260919_w21_cost_vs_particle_count.md` **전체**(특히 §5-2 전략 재평가, §5-3 권고).
3. `claudedocs/session_20260919_w20_decisions_execution.md` §2(권고 철회 경위) — 같은 실수 반복 방지.
4. 산출: `w21…d493/{GPU_OCCUPANCY_BASELINE.json, coordinator/REP1_HANG_OBSERVATION.json, inputs/}` ·
   워커 `w21-smoke-evidence/.../smoke_evidence/SMOKE_COMMAND_CORRECTED.md` ·
   `w21-p2-runs/.../runs_meta/VARIATION_AND_BASELINE.json`.
5. `git status --short`, `git worktree list`(main + 9개), 외장 SSD 장착 여부.

## 갈림길 3택 — 사용자 결정 (코디네이터 권고 = 1번 먼저)

| # | 선택 | 비용 | 무엇을 얻나 |
|---|---|---|---|
| **1** | **rev33b 스모크**(알 300개) | 약 50분 | rev33b 검증 **+ 정지 원인 범위 절반 축소**(진단 겸함) |
| 2 | 정지 원인 조사 | 상한 미정 → **착수 시 범위·상한 먼저 고정** | 알 축소 경로 복구 가능성 |
| 3 | case ① 20,000알 직행 | 최소 5.20 h / 권장 10.07 h | 본 연구 질문(문 닫힘 보정)에 직접 전진 |

**1번을 먼저 권하는 이유(싸서가 아니다)**: 정지는 도구 수직 이동 구간에서 났고 그 구간 경로점은
**더미 표면 높이**에서 유도된다. 알 5,000개 = 새 얇은 슬랩(28.08 mm, 정지), 알 300개 = 20,000개에서
ROI 절단(40.79 mm), 기준 = 원본(41.18 mm, 정상). **스모크가 정상이면 정지는 알이 적어서가 아니라
얇은 슬랩 구성 때문**이고, 그러면 그 구성은 아래 confound 때문에 case ① 에 쓰기 어려우므로
조사에 시간을 더 쓰기보다 3번 직행이 합리적이 된다.

## 현재 사실 / 반드시 아는 함정
- 🔴 **W20 `GPU_SMOKE_PACKAGE.md` 원본 명령은 그대로 돌리면 죽는다**(`--max-particles 300` + 20,000알 증거
  → `sim_w13_full_cycle.py:246` `SystemExit`). 정정판 증거 `inputs/numeric_inputs_roi300_candidate.json`
  (sha `4f76f855…d02f5e`) + `SMOKE_COMMAND_CORRECTED.md` 를 쓸 것.
  **축소 플래그는 fail-closed 도메인 검사를 우회하지 않는다.**
- 🔴 **step 경계 미도달 시 시뮬 자신의 `--max-wall-s` 는 영원히 발동하지 않는다** → 장시간 실행은
  **바깥 watchdog 필수**. 정지 시 상한 단축은 상한 연장이 아니다(규약 위반 아님).
- 🔴 **알 축소가 case ① 을 오염시키는 기작**: 알↓ → 힌지 반작용 토크↓ → **문이 더 닫힘**.
  그런데 문이 얼마나 닫히는가가 case ① 이 재려는 값이다(브리핑 ② §9 위험 3).
- **W19 A 배출량 = 272~327알(5.51~6.62 g), 정착 미증명.** 단일값 "272알" 금지.
- **W16 `run_01` 을 기준으로 쓰지 마라**(하네스 SIGTERM 미완주). 기준은 `run_02`(wall 11,021.63 s)뿐.
- **GPU 점유 20 % 상시**(`mcp_memory_service.server` 7개). 죽이지 않기로 결정함. 배율 인용 시 confound 명시.
- **정본 물리 코드는 여전히 rev32.** rev33b 는 GPU 스모크 전이라 승격 안 됨.
- Orca: 비-Orca 터미널은 `terminal create` 후 `--from`(단 `check` 는 `--terminal`, `worker-release` 는 둘 다 없음),
  Run 당 actionable waiter 1개, `check --wait --json` 은 다중 JSON → 마지막 문서만 파싱.

## 이 사용자의 작업 방식 (지난 세션들에서 확인됨)
- **비용·자원이 드는 결정은 착수 전에 먼저 묻는다**(RunPod 사용 여부를 명시적으로 물으라고 지시함).
- **선택지 나열이 아니라 권고를 원한다.** "어느 쪽이 좋을 것 같아?" 에는 근거와 함께 하나를 고른다.
- **내 권고를 스스로 교차 검증한 뒤 실행한다**(W20 에서 권고 2건이 근거 파일 확인으로 철회·수정됨).
- **step-by-step 순차 사고**를 명시적으로 요구한다.
- 출력 폴더·산출물 이름은 **직관적으로**.
- 워커에 위임하되 **메인이 독립 재계산으로 교차 검산**한다(해시 재계산·도구 재실행 수준까지).
- 보고는 한국어, **관찰 가능한 절차 → 수치 → 근거 파일 → 한계·다음 승인** 순서, 용어는 첫 사용 시 풀어 쓰기.

## 붙여넣을 요청문
```text
/home/cgxr/Documents/Robotics/RoArm_Project에서 이어서 작업해.
AGENTS.md와 START_HERE.md의 부팅 순서를 따르고 claudedocs/CONTINUE_20260923_W22_FORK.md를 전체 읽어.
session_20260919_w21_cost_vs_particle_count.md의 §5-2(전략 재평가)와 §5-3(권고), D493을 읽어
W21 상태를 복원해: P2 알 5,000개는 n_sync 556에서 결정적 정지로 CELL_UNRUNNABLE,
부분 비용은 접촉쌍 0.23배인데 시간은 0.43배라 고정 부담이 있고, 알 축소는 힌지 토크를 통해
case ① 문 닫힘 측정을 오염시킨다는 것까지.

갈림길 3택(① rev33b 스모크 ② 정지 원인 조사 ③ case ① 20,000알 직행)을 쉬운 말로 표로 브리핑하고
너의 권고를 근거와 함께 하나 골라서 제시한 뒤 내 결정을 기다려. 용어는 처음 나올 때 풀어 써.
step-by-step으로 순차적으로 사고해.

GPU를 쓰는 작업은 착수 전에 비용·시간·어디서 돌릴지(로컬 vs RunPod)를 먼저 물어.
물성·형상·알 크기·경로·문 속도·보호선·dt 1µs·cd_update_freq·seed 460·원자료는 바꾸지 마.
새 물리·RunPod pod·commit/push·실물·설치·학습은 내 승인 뒤에만.
스모크를 돌린다면 반드시 ROI-300 증거(numeric_inputs_roi300_candidate.json)를 쓸 것 —
W20 원본 명령은 도메인 대조로 죽는다. 장시간 실행은 setsid + 바깥 watchdog 필수.

작업자는 새 Orca worktree에 배정하고(Claude claude-opus-5-5) 여기서 보고만 받되,
워커 결과는 메인이 독립 재계산으로 교차 검산해. 메인만 상태 원장·relay를 소유하고 worker 원장을 merge하지 마.
LFS staged D·원본 삭제·force-add·이력 재작성 금지.
관찰 가능한 절차→수치→근거 파일→한계·다음 승인 순서로 한국어로 보고하고
종료 시 START_HERE·새 session 문서·relay를 갱신해.
```
