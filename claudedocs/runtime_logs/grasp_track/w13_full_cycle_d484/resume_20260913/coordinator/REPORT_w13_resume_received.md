# W13 재개 수신·통합 보고 — 2026-09-14

이번 case의 신규 변수: [] — 기존 W13 전체 사이클 통합 재개. 추가 dt·물성·전략 변수 없음.

## 1. 무엇을 했고 왜 했나

기존 scoop→lift→reclose 뒤에 없던 **운반→용기 위 문 열기→대기→문 닫기→HOME 복귀**를 연결하고, 실제 한 번 실행해 배출·흘림·복귀를 관측했다. 생산은 Claude Opus5의 별도 Orca worktree, 독립 검수는 Codex gpt-5.6-sol/high의 다른 worktree가 담당했고 메인은 감독·재검수·상태 원장만 소유했다.

**결론: 본 물리는 TIMEOUT 부분 종료, 전체 사이클 성공 미입증이다.** 부분 궤적의 Isaac 영상은 생성·표본 검수했지만 원자료 규약 2개와 재생 검증 3개가 실패했다. 실패를 PASS로 바꾸지 않았고 새 물리/재생을 반복하지 않았다.

## 2. 관찰 가능한 실행 순서

1. **경로 검사 통합**: 접촉 뒤 실제 자세에서 다음 운반 자세로 이어지는 연결 구간을, 첫 물리 호출 전에 검사하도록 붙였다. 실제 실행은 227개 구간 모두 사전검사를 거쳤고 계획 대비 편차0으로 기록됐다. 계산상 최악 여유65.661289mm, 실제 시작점 검사의 최악 여유72.682252mm. 이 연결 구간의 증거이지 전체 로봇 경로/서보 가능성/배출 성공 증거가 아니다.
2. **Isaac 준비**: S1 고정반쪽/문·팔·더미·용기·주석을 실제 화면에서 확인했다. 준비 실패를 보존하며 전진 revision으로 수정했고, 마지막17_actual은 정리 포함40.219416초·rc0, root 8장 실제 검사와 독립19항목 검수 뒤 인수했다. 준비 raw2+합성6은 실제 전체 물리 결과가 아니다.
3. **단일 본 실행**: 09-13 21:25:01 KST에 rev28/run_01을 시작했다. 승인된 총9시간에 정리 시간을 포함하도록 31,200초에 TERM을 예약했고, 09-14 06:05:20에 저장·정리가 끝났다. 총31,218.753855초(약8시간40분19초)≤32,400초. child rc0이지만 timed_out=true·runner124이며 강제KILL0이었다.
4. **부분 재생**: timeout 때문에 원 실행의 후속 단계는 자동 실행되지 않았다. 별도 정확 GO로 post03에서 원자료의 바이트 동일 복사본만 Rerun→Isaac 순서로 한 번 재생했다. 원본14개는 읽기 전용이며 실행 전/후 해시를 대조했다. post01/02는 미실행 보존했다.
5. **실제 검수**: Rerun 결정 PNG를 root와 감사가 각각 열었다. Isaac MP4에서는 원자료 시점으로 고른 11개 표본을 watch 스킬로 추출해 root가 실제 보았고, 감사는 원래 PNG 8장을 보았다. 전체 영상을 연속 재생해 모두 봤다고 주장하지 않는다.
6. **독립 대조와 root 재실행**: 원자료 감사는14/16, post03 감사는12/15였다. root는 전체를 읽은 post03 checker의 출력만 새 root 경로로 지정해 재실행했고, rc1/같은 FAIL3개 및 생성시각 외 보고 내용 동일을 확인했다. 수동 이미지 관찰 항목은 이전 실제 검사 기록을 재사용한 것이지 자동 육안 검사나 재렌더가 아니다.

## 3. 실제 수치와 해석

| 관측 | 확인값 | 의미 |
|---|---|---|
| 실제 물리 시간 / 저장 | 24.486802938176766초 / 16,304 sync / 283 입자 프레임 | 12개 phase 이름은 존재해도 마지막 HOME hold는0행 |
| 종료 | SIGNAL_STOP, diverged=false | 발산 종료가 아니라 사전 시간 한도 종료 |
| HOME 립 목표 오차 | 29.57491mm | 원래 HOME에 도달하지 못함; return_home_end 표식은 중단 뒤 저장 |
| 최종 기록 재고 | source19,712 / bin0 / tool0 / spill73 / in_flight0 / ambiguous215 | 기록된 분류이며 source 규약 불일치가 있어 그대로 정답으로 승격 금지 |
| strict 원문 규약 최종 재계산 | source14,350 / ambiguous5,577; 나머지 동일 | 최종5,362개 라벨 불일치, 입자 유실 아님 |
| 용기 판정 | 확정 분류0개, 가능 상한11개(0.222831g) | 정확한 정착 배출량 아님 |
| 마지막 정착 관측창 | 0.25초 안3프레임, 최대간격0.100025초 > 동결0.05초 | cadence_ok=false, exact_single_value_allowed=false |
| 재닫기 후 기록상 공구 내부 cohort | PF107의144개 → 최초PF136에서0개 | t8.978535→11.303116초; 운반 중, 용기 도착 전에 보유 판정이 사라짐 |
| 동일144개 최종 기록 / strict | 133source+7spill+4ambiguous / 132+7+5 | 이미지로 개수를 센 것이 아니라 원시 ID를 추적한 값 |
| 저장 sync 최대 입자 속도 | 3.5142287698391907m/s | 저장 표본에서 >5m/s 경고/>20m/s 중단0; 모든 내부 dt 최대 보장 아님 |
| Rerun / Isaac 후처리 시간 | 1,046.855042초 / 554.113737초 | 각 rc0·timeout0·정리 포함·각5,400초 한도 내 |
| Isaac 영상 | 283프레임 / 10fps / 28.3초 / 1600×2102 | 일정 프레임 속도 영상이므로 실제24.4868초와 길이가 다름; 각 프레임 원자료 시각 참조 |

원자료 [물리 JSON](/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/run_01/w13_cycle_seed460.json), [원시 NPZ](/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/run_01/w13_cycle_seed460.npz), [실행 영수증](/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/run_01/EXECUTION_RECEIPT.json). 독립 [원자료 감사](/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/REV28_PRODUCTION_PARTIAL_RAW_AUDIT_01.json), root [수치 재현](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/ROOT_PARTIAL_RAW_SPOT_REPRO_01.json)·[144개 ID 시점 추적](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/ROOT_RAW_COHORT_VISUAL_CUES_01.json).

## 4. 실패를 확인한 방법 — 기준은 사후 완화하지 않음

### 원자료 사전 규약 FAIL 2개

- **단계 전환 인덱스**: RAW_SCHEMA_REQUIRED.md:52는 phase만 바뀐11개 행을 요구한다. 실제25개에는 subphase 전환과 첫 행도 섞였다.
- **더미 안 분류**: RAW_SCHEMA_REQUIRED.md:76–77와 생산 선언은 알을 구성하는 모든 구체가 여유 거리까지 포함해 안쪽이어야 한다. 실제 식은 구체 최하단 대신 최상단을 바닥−margin과 비교했다. PF0/ID8에서 최하단−0.000001232287m < 요구0.0025m인데 source로 기록된 반례를 root도 재현했다. 전체283×20,000 분류 중1,507,161 라벨 불일치다. 사전검수가 이 구현 불일치를 놓쳤음을 인정하며 이미 나온 결과에 맞춰 규약을 바꾸지 않았다.

### 재생 검증 FAIL 3개

- **문 정지 이벤트 연결**: 저장 RRD를 직접 읽으니 문 정지5개 모두 PF282에 연결돼 있었다. 원시 sync별 대응 PF는69/98/107/189/220(동일 sync의 별도 결정행도 존재)이다. exporter가 이전 루프의 fi를 재사용한 결함이며, 문정지 시각의 입자 상태를 제대로 묶은 것이 아니다.
- **Rerun 결정 화면**: PNG는 +0.000초 더미만 보이고 공구/용기/연결경로와 결정표·그래프가 빠져 실제 결정 판단에 못 쓴다. RRD/RBL footer, 버전0.34.1, entity57/timeline6/component20 구조 검사 PASS와 시각 인수 FAIL을 구분했다. 활성 PF 타임라인 때문이라는 생산 가설은 실제 화면의 sim_time_s와 달라 철회했다. 정확 내부 원인은 미확정이며 추가 렌더는 하지 않았다.
- **Isaac 관절 출처 집계**: 요약은228+55+55=338행, 실제283개 프레임의 정확 문자열 집계는228+2+53=283행이다. 같은 두 IK 출처를 중복 집계한 메타데이터 오류다. 실제 영상283장이나 원자료 row/time/phase/door 매핑이283개라는 사실과 별개다.

독립 [최종 재생 감사](/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/POST03_PARTIAL_REPLAY_RESULTS_AUDIT_02.json), root [같은 검사 재실행](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/ROOT_POST03_AUDIT_REPRO_01.json), [Rerun 실제 화면 검사](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/ROOT_RERUN_INSPECTION_01.json).

### 표시의 추가 한계

- 표시용 역기구학 관절값은20프레임에서 선언한 어깨 범위를 벗어났고 최대 립 재투영 오차는8.132mm다. 동결 기준에 해당 표시 오차의 수치 합격선이 없으므로 새 물리 FAIL 임계값을 만들어 적용하지 않았다. 대신 **실물에서 이 경로를 구동할 수 있다는 증거가 아님**을 명시한다.
- 감사 checker의 position/quaternion/ID/contact 해시는 원자료 배열 fingerprint이고, 영상 메타데이터의 행·시각 대조다. 이를 렌더 픽셀/실제 USD geometry의 bit-exact 동일성 증명이라고 부르지 않는다.
- root가 본 운반 표본에는 공구 아래 입자가 보였지만, 어느 틈으로 왜 떨어졌는지나144개라는 수량은 이미지 단독으로 확정하지 않는다. 용기 안 큰 자홍색 구체는 목표 마커이지 배출된 알이 아니다.
- Isaac 초기 reset/warmup과 달리 표시 프레임 루프에서는 물리 시계를 진행하지 않았다. 새 DEME 물리0과 Isaac 전체 과정에서 물리스텝0이라는 주장은 다르다.

## 5. 검증 기준에서 root가 정정한 부분

이번 세션의 변경은 물성/동작 전략 변경이 아니라 검증·관찰 계약 정정이다. 상세 시각별 근거는 세션 append 기록과 coordinator CONTRACT_ADDENDUM_01 및 REFERENCE_RELEASE_CONTRACT_REV2에 있다.

1. 근거 없는0.5mm 경험 여유나 관측+1μs의 임의2배 대신 설치 DEME/CUDA 소스의 시간·수치 오차 상계를 요구했다. 회전각 계수 누락과 격자 경계 항 누락도 수정했다. 이는 프로젝트의 보수적 유도이며 NVIDIA가 RoArm을 보증한 값이 아니다.
2. 저장시각의9자리 반올림값과 원래 float64를 분리했고, 같은 sync에서 저장된 서로 다른 결정 프레임을 허용하되 PF행ID는1:1로 유지했다. 준비8프레임 제한을 본 결과까지 잘못 확장한 감사 초안도 정정했다.
3. >5m/s는 경고, >20m/s는 코드 중단이라는 기존 의미를 유지했다. 정착 미확인 때는 하한/가능상한을 분리하고 단일 정착량 성공 주장을 금지했다.
4. manifest에 해시 항목이 적혀 있다는 것과 실행기가 실제 검사한다는 것을 구분했다. post01/02의 설명용 키 미연결을 검출했고 post03에서 실제 소비되는 키로 연결한 뒤 한 바이트 변조 거부를 확인했다.
5. root가 추가했던 sim weakref 즉시소멸 조건은 소유 참조 해제와 혼동한 과도한 조건이었다. 설치본과 공식 API 근거로 다음 준비 실행 전에 소유4변수/public singleton 해제+필수weakref 조건으로 정정했다. 이전16_actual FAIL은 소급 PASS로 바꾸지 않았다.
6. 준비 단계 횟수 초과·일부 중간 메타데이터 원문 보존 공백·freeze 전 inbox 미열람·생산 Python 편집 절차 위반도 세션에 남겼다. 이후 준수를 과거까지 완전 준수였다고 바꾸지 않는다.

## 6. 바로 볼 근거와 보존

- [부분 동작 Isaac 영상](/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/partial_post_03/isaac/w13_full_cycle.mp4) — 파일명 full_cycle은 옛 이름이며 실제 프레임에는 INCOMPLETE PHYSICS / NOT A FULL CYCLE / SIGNAL_STOP 경고가 있다.
- [root 영상11표본 검사](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/ROOT_ISAAC_VIDEO_INSPECTION_01.json), [watch 추출 기록](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/WATCH_ISAAC_POST03_01.md).
- [생산 최종 보고](/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/W13_FINAL_PRODUCER_REPORT.md), [독립 최종 보고](/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/REPORT.md).
- [원본14개 사후 보존](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/PARTIAL_RAW_POSTCHECK_01.json), [원본14개 manifest](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/PARTIAL_RAW_MANIFEST_01.json).
- NPZ SHA256 529f422e962730b17b520c3c9af2261aae7c431ec5e078cd2c368dfab0c46b0f.
- 물리 JSON SHA256 e482b939a4cf42e5284b9b15e36a9f3613e989161ec538475e2c05f4260d9340.
- MP4 SHA256 1471127134fc06a88d546c891de0d2515f4911fcd1832f0ae841258ac1a2bc34.
- 독립 post03 SHA256 d63d2f1fcd1691dd06afa8214e8c50aae00490d883580f8b1a6d77cedaf67e45.
- root 재실행 SHA256 f18c4ba29ddbf44c6beb428c23a2666916c9ef3610a16a34a345123b2b71fe5b. 두 보고는 generated_utc를 제외하면 동일하다.

## 7. 종료·승인 경계

Run run_5cba1e55a775의 감사 Task/Dispatch task_eed17c18cd62/ctx_e5023d7f09ed는 정상 failed worker_done(msg_f3bef394d5a4) 뒤 release됐고 감사 에이전트 터미널만 닫혔다. worktree/산출은 보존했다.

생산 task_d3ad7187e682/ctx_006e285a14a4는 capability 누락으로 worker_done2회가 거부됐다. 거부 메시지는 coordinator에 전달됐지만 settlement는0이었다. root는 토큰을 복원·대리 제출하지 않았다. 생산의 실제 종료턴을 확인한 뒤 공식 worker-abandon(request991e59cd-d7bb-4730-a5ac-50ab9bd75731)으로 lifecycle을 failed/abandoned 처리했다. **사용자 소유 생산 터미널은 retained, 프로세스 강제종료0**이다. 이는 정상 worker_done 수락과 다르며 새 supervisor 작업은 없다. reclaimable 목록0 확인; 원래 과제의 성공 인계가 아니다.

unlazy 계약의 미충족 항목은 삭제하거나 PASS로 바꾸지 않고 HANDOFF REQUIRED로 남긴다. 실제 현재 증거는 전체 물리/시각 수락 실패이며 초기 통합 검사기 일부의 구경로·미동결 target 문제도 있다. 정확 게이트별 결과는 .unlazy/w13_resume_20260913/GATES.md 및 gates/leaf-1.1.md·leaf-1.2.md를 참조한다.

최종 수락표는10개 중3충족·7미달성 인계(ABANDON), 아직 처리 중인 미달성0이며 checker rc1/HANDOFF REQUIRED다. root:G2/G3, 생산:G1/G2/G3, 감사:G1/G2를 성공으로 바꾸지 않았다. 기존508개/HEAD 보존과 원본14개는 종료 직전 다시 확인했고, 원장594행 prefix 불변·595행6열·최근색인20건·START61행·보고서 로컬링크16개 존재를 확인했다. 소유권 lease2개를 해제했다. 최종 영수증은 coordinator/FINAL_CLOSEOUT_CHECK_01.json이다.

**다음은 별도 승인 경계**: 원자료 분류/전환 규약과 재생3결함을 새 revision에서 수정·검수할지, 운반 중 보유 실패의 원인을 별도 case로 조사할지 결정해야 한다. 새 장시간 물리, 재렌더, 경로·문 동작 변경을 자동 시작하지 않는다. A/B/C·학습·실물 조회/구동/PID/토크/카메라·설치·commit/push는 실행하지 않았다. W13은 W11과 domain/유한벽/전체경로가 다른 case이므로 W10↔W11 dt 비교나 수렴 증명을 대체하지 않는다.
