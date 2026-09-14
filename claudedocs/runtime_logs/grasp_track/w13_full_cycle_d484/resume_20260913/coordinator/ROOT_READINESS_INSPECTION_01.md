# Root Isaac readiness inspection — NOT ACCEPTED

2026-09-13. 실제로 열어 본 결정 PNG:

- `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_03/frames/f_00000.png`
- `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_03/frames/f_00007.png`

## 관찰과 판정

1. 두 장 모두 실제 S1 고정 셸/문이 아니라 순정 파랑/초록 그리퍼가 보인다. 본 검수 대상 형상이 없으므로 준비 FAIL이다.
2. 첫 화면 측면의 팔 일부와 상면의 더미 일부가 프레임 밖으로 잘린다. 상면의 큰 어두운 받침대가 시야를 차지한다.
3. 노란 트레이 테두리/기둥은 보이지만 바닥/벽면은 잘 보이지 않는다. 경계 형상 검수용 화면으로 불충분하다.
4. 분홍 용기는 지면 위에 떠 있는 모양이다. 고정 fixture의 지지대 미표시일 수 있으며, 이 관찰을 근거로 실제 물리 배치나 높이를 변경하지 않는다.
5. 마지막 프레임의 배출 자세는 SYNTHETIC, 전체 화면은 INCOMPLETE PHYSICS로 명시돼 있다. 이는 보존된 벽 시험+합성 자세 준비 자료이며 운반·배출 결과가 아니다.
6. 매니페스트의 `max_lip_reprojection_err_mm=0`은 명령 관절의 FK 재계산 차이다. 실제 USD/메시의 측정 오차 0을 확인한 것이 아니다.

## 실행 범위와 후속 경계

생산 CHECKPOINT2 및 후속 msg_e4be0012f338는 readiness 01/02/03 세 번을 보고했다. 계약은 한 번이므로 02/03을 사후 승인 처리하지 않는다. 새 입자 물리 실행은 아직 0이다. 03은 출력 후 앱 종료에서 멈춰 생산자가 자기 PID에 SIGTERM을 보냈다고 정정했다. rc0로 인수하지 않으며 정확 PID·signal·전체 과정 시간·정리 영수증은 대기다. 렌더 구간 14.87초를 프로세스 전체 실행시간으로 보고하지 않는다.

root msg_14774d23bc62의 **추가 GPU/render/physics HOLD**를 생산자가 명시 수락했다. CPU에서 실제 S1 메시/원시 노드 대응·화면 구도·안전한 종료 처리를 수정한 뒤, 정확한 새 명령/한도/경로를 root가 검토하기 전 추가 실행 금지. `readiness/INSPECTION.json`의 기존 PASS 문구는 보존된 오판 기록이지 현행 판정이 아니다.

후속 정정: 생산자는 별도 erratum 요청과 달리 같은 INSPECTION.json을 READINESS_FAILED로 덮어썼다고 보고했고 root도 실제 변경을 확인했다. 따라서 위 "보존된 오판 기록"은 원파일 바이트 보존을 뜻하지 않는다. root가 변경 전에 읽은 원문 중 핵심 발췌를 `ROOT_READINESS_PRECORRECTION_EXCERPT.json`에 남겼다. 전체 원문 백업/원해시로 주장하지 않는다.
