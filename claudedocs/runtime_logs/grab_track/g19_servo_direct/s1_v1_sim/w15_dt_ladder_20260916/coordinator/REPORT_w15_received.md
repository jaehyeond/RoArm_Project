# W15 dt ladder — 메인(코디네이터) 수신 보고와 원자료 재확인

작성: 2026-09-16 밤. 이번 case의 신규 변수: [timestep_s ∈ {1e-5, 1e-4, 1e-3}]. 메인은 GPU를 돌리지 않았고 워커(Claude claude-opus-5 high, worktree `w15-dt-ladder`, Orca run `run_ea74393ac4b4` dispatch `ctx_0cefa6ce95eb`)의 산출을 읽기 전용으로 재확인했다.

## 워커 보고의 정본
`/home/cgxr/orca/workspaces/RoArm_Project/w15-dt-ladder/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w15_dt_ladder_20260916/` — `REPORT_w15.md`(SHA `820da278…`), `comparison.json`(`8bcc67f9…`), `PREREG.md`(`a0fafec1…`, 22:34 첫 실행 전), `preflight.json`, `params_w15_dt1e{5,4,3}.json`, `launch_*/terminal_*.json`, `cell_*/`, `inspection.json`, `manifest.json`(37파일).

## 메인이 원자료에서 직접 확인한 것
| 셀 | timeline 행 | 마지막 sim_t / phase | 저장 최대속도 | >5 m/s | 문 정지 | stderr 서명 | 종료 | 벽시계 |
|---|---:|---|---:|---:|---|---|---|---:|
| 10 µs | 2,790 (settle 25/descend 350/close 2,016/lift 134/reclose 265) | 3.1533 s / reclose | 19.2243 m/s | 4 | pinch_guard 3.62°(close), 3.02°(reclose) | `System max velocity is 2.171512e+09` | SIGABRT(rc −6 = shell 134) | 2,530.1 s |
| 100 µs | 0 | 없음 | — | — | — | `GPU Assertion: out of memory … DataMigrationHelper.hpp:53` | SIGABRT | 40.0 s |
| 1 ms | 0 | 없음 | — | — | — | `GPU Assertion: operation not supported … DEMCubContactDetection.cu:168` | SIGABRT | 10.0 s |

- `terminal_*.json`의 자동 `exit_class`는 세 셀 모두 `engine_error_out_vel`로 적혀 있어 셀 2·3에서는 **오기**다. stderr 원문이 정본이며 워커 보고서(§2-6)도 stderr로 재판정했다.
- PNG 2장(`timeline_comparison_w15.png`, `sync_duration_ratio_w15.png`)을 메인도 실제로 열어 보았다: 10 µs(빨강)만 힌지 저항 −1.55 N·m 음수 구간, 2.59 s 19 m/s 스파이크 뒤 생존 후 3.15 s 재폭주; sync 지속시간 비율 0.1 ms 요청에서 10 µs 1.10·2 µs 1.02·1 µs 1.01.
- 단일 변수 확인: params 3개 diff = `timestep_s`·`render_timeline_path`(워커 preflight), sim 코드 SHA `2e40f7ed…8933`.

## 판정과 한계
D486 및 `EXPERIMENT_LEDGER.md:597` 참조. 합격선 없음, 수렴/안전 판정 없음, 포획량 비교 불성립, B′(`cd_update_freq` 스케일)·로그 추가·반복 셀은 별도 승인. 독립 Codex 감사 `ctx_695a18b54a56` 결과는 `w13-cycle-audit/.../w15_dt_ladder_20260916/audit/`와 세션 문서에 기록: **Codex W15 감사 7/7 PASS**(판정 `PASS_WITH_RECEIPT_AND_SYNC_GROUPING_ERRATA`, `audit/W15_INDEPENDENT_AUDIT_01.json`, 검사기 `audit_w15.py`): 사전등록 첫 실행보다 110.3 s 앞섬·params 2키 diff·순차 비중첩(34.3/40.6 s)·10 µs 2,790행/3.1533 s/19.2243 m/s/>5 4행/pinch_guard 2회/엔진 2.17e9/rc −6→134/2,530.1 s·100 µs OOM 40.0 s·1 ms 커널 assertion 10.0 s·실측 sync 0.110/1.010/4.010 ms=예측 1.1e-10 s 이내·기준선 3행 일치·과장 0. 정오표 2건: (a) receipt `exit_class`가 셀 2·3에서 오라벨(stderr 정본), (b) `comparison.json`의 sync 묶음 개수 1323/958/508은 `fine_on()` 규칙상 1324/957/508 — 첫 reclose 0.11 ms 간격의 묶음 오분류이며 중앙값·비율은 불변. 이 감사는 원자료·보고서 정합성 판정이지 셀 성공 판정이 아니다.
