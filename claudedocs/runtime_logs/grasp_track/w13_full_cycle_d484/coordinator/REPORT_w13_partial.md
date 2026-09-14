# W13 병렬 배정 수신 — 통 벽 시험 PASS / 전체 사이클 미완료

2026-09-12. 이번 case의 신규 변수: [운반·고정 배출·HOME 복귀를 포함한 전체 사이클 통합]. 실물·A/B/C·학습은 범위 밖이다.

## 1. 무엇을 맡겼나

| 담당 | 실제 모델 | Orca worktree | 받은 결과 |
|---|---|---|---|
| Claude 구현/실행 | claude-opus-5 | `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle` | FK/HOME·전체 단계 코드 초안·고정 통 벽 시험·Rerun·최종 진단 |
| Codex 독립 검증 | gpt-5.6-sol / high | `/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit` | 사전 조건·기하/좌표 검토·원자료 감사·실행 중 연결 경로 검사 함수 |

동시에 출발한 실제 두 Dispatch이며 launch 증거는 `launch_receipts.json`. 원장 쓰기는 메인만 소유했다.

## 2. 관찰 가능한 실행 순서

1. W11 입력/물성과 실제 HOME/FK를 대조하고, 물리 립과 충돌 형상 립의 local3mm 차이를 명시했다. 배치는 실측 보정이 아닌 고정 시뮬레이션 배치다.
2. 더 큰 이동 영역에서 원래 통 벽이 사라지지 않도록 유한 메시 벽을 만들고, 같은 초기 더미의 짧은 구간만 W11 저장값과 비교하는 시험을 사전 고정했다. 벽 표현/계산 영역/접촉 없는 툴 위치 차이가 남으므로 인과 동등성 비교가 아니다.
3. rev10 코드·입력·명령을 동결하고 독립 검토 후 `GO_wall_smoke_rev10.md`로 정확히 한 번 승인했다.
4. 16:47:42 KST 실행 시작. 해시44/44·기존산출0 확인 후 실제 HOME t0→settle25sync를 수행했다. 물리128.352초, 비교0.215초, Rerun2.171초, 전부rc0·timeout0·자동재시도0.
5. Codex가 독립 원자료 검수, 메인은 `verify_smoke_received.py`로 읽기 전용 재검증하고 초기 Rerun 화면과 최종 확대 진단을 실제 열어 보았다.

## 3. 수치와 범위

- 20,000알·seed460·dt1e-6. 요청 구간 합0.100000초와 관측 엔진시간0.100025초를 구분한다. 저장 입자시간의 Float32 표현과 W11 Float64 시간이 bit-exact 같다는 주장은 하지 않는다.
- 같은 목표시각 W11 row-wise 위치 차이: 최대0.041462659mm, p99 0.000381965mm, 중앙값0.000206418mm. 사전 최대0.5/p99 0.1mm 기준 PASS. W11은 명시 ID가 없어 입력 순서 보존에 의존한 비교이며 저장 속도 배열은 없어 속도 비교불가다.
- 전26sync 툴 접촉0. 입자 ID20,000개 고유·순서유지. 중심 기준 통 이탈0·비유한값0·pop-stop0.
- 트레이 접촉 최대90개, 원시 접촉2244행의 sync/mesh_id2 매핑 일치. 미세 구 겹침은 최대90개·최소gap−0.001526904mm로 보고하고 탈출과 혼동하지 않는다.
- 저장 최대속도0.062872879m/s, 종료시0.053992504m/s. **완전 정착 상태의 증명이 아니다.** 벽 접촉 수만으로 무게 지지/편향 없음도 주장하지 않는다.
- RRD/RBL0.34.1·footer·정확 entity/component/timeline 검사 PASS. 전체26sync·입자6프레임·접촉2244행 기록. 초기 RRD PNG는 t0 전체 배치 검수, 별도 최종 raw 진단은 sync25 확대 검수다.
- 감사의 8개 대조 항목 중 7개는 잘못된 배열을 실제 검사기에 넣어 거절한 시험이다. 나머지 `hash_corruption`은 현재 해시가 전부0과 다르다는 sanity check여서 완전한 변조 주입 시험으로 세지 않는다. 실제 입출력 해시 대조는 별도로44/44·28/28 통과했다.

## 4. 근거 파일

아래 상대경로 prefix는 각각 해당 worktree의 `claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/`다.

- 생산: `implementation/attempts/smoke_wall_regression_01/{EXECUTION_RECEIPT.json,HASH_VERIFICATION_RECEIPT.json,wall_regression_verdict.json,w13_cycle_seed460.npz,MANIFEST_finalized.json}`.
- 실제 최종 확대 그림: `implementation/attempts/smoke_wall_regression_01/decision_diagnostic_01/wall_decision_zoom_t0p100025.png`; 대응 수치 `decision_diagnostic.json`.
- 독립 감사: `audit/wall_smoke_result_01/{REPORT.md,RESULT.json,verify_wall_smoke.py}`.
- 후속 연결 경로 검사 함수: `audit/runtime_bridge_guard_01/{bridge_clearance_guard.py,TEST_RESULTS.json,REPORT.md}`. 합성5개 검사 통과 보고, **실제 생산 코드에 아직 미통합**.
- 생산 인계: `implementation/HANDOFF_NOTE_01.md`. runtime 외삽 정정은 이 문서 §5가 이전 보고서의 단정을 대체한다.
- 메인 재검증: `verify_smoke_received.py`, `.unlazy/w13_full_cycle_20260912/GATES.md` G5 PASS. 전체 사이클 검수 G3는 미충족 상태로 남겼다.

## 5. 미완료와 다음 승인 경계

전체 HOME→퍼내기→운반→배출→HOME 물리는 실행하지 않았고, Isaac 전체 사이클 영상 코드도 아직 없다. 다음 순서는 새 생산 revision에 실행 중 연결 경로 검사를 통합→실제 원시 규격/영상 준비 검토→장시간 계산 승인→본 실행→Isaac 재생/독립 감사다. 불명확한 연결 경로는 `CLEARANCE_UNCERTIFIED`로 중단해야 하며 결과를 보고 경로를 임의 최적화하지 않는다.

짧은 시험 속도를 명목23.6152물리초에 단순 환산하면 약8시간이지만 실제 전체 시간은 미측정이고 이 범위를 벗어날 수 있다. 최대9시간 실행할지 준비까지만 할지 사용자 선택을 요청했으며 종료 시점까지 미회신이다. **장시간 실행은 미승인**이다.

원래 전체 사이클 과업을 끝내지 못했으므로 두 Orca Task는 `failed`로 부분 인계했다. 이는 통 벽 물리 시험 실패라는 뜻이 아니다. 두 worker_done을 수신해 터미널만 release했고 worktree/파일은 보존했다. 재개 시 새 Dispatch가 필요하며 완료된 lifecycle ID를 재사용하지 않는다.

## 6. 보존 사건과 제한

초기에 producer가 비정본 dt1e-4/1e-5 축소 시험을 진행해 중단시켰다. 해당 시도는 W13 과학 결과에서 제외했고 A2/A3 원래 로그 덮어쓰기 공백은 복구된 것처럼 주장하지 않는다. 상세 `implementation/preflight_01/ATTEMPTS.md`와 메인 `recovered_smoke_error_excerpt.json`(2차 발췌만) 참조. 이후 승인 시험은 고정dt1e-6 한 번뿐이다.

실물 조회/구동/카메라/PID/토크·학습·A/B/C·설치·commit/push0. 기존213입력/소스와 HEAD는 보존 검사 대상이다. main Git push만으로 외부 worktree 산출이 포함되지는 않는다.
