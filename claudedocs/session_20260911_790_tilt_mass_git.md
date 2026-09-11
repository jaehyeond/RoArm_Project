# 790 비교·단일 배출 기울임·질량 반복·Git push

이번 case의 신규 변수: [닫힘 토크 상한900→790 (기존 비교 완결), 단일 배출 기울임 (별도 하위 비교)]. PID는 고정하고 원시 기록을 계속 수집한다. 사용자 “진행해. 그리고 나서 지금 git 제대로 된 위치에 push해봐”로 순차 실험 및 커밋/push 승인. 반복 계량은 실제 사용자 저울 값만 사용한다.

기준: 수정900 `torque900_03` 배출19.95g·잔류 약10~15알, 리프트 추가개방0° 관측. 실행 경로/현재 물리 상태 미확인인 이상적 출구5°를 관절 명령으로 대체하지 않는다.

완료 항목은 `boot_check_20260911/GATES.md` G4·G5·G6·G7. 새 실기 출력은 같은 boot_check 폴더의 forward-only 실행 폴더다. 물리 준비 상태는 비동기 질문 중이며 답변 전 dependent motion0.

Git 읽기 확인: main worktree /home/cgxr/Documents/Robotics/RoArm_Project, branch master, origin git@github.com:jaehyeond/RoArm_Project.git, 기본 브랜치 master. 원격/로컬 기준 d41c258782da49492f0474b76d110454224597f3 동일. gh API상 public·ADMIN. 다른 Orca worktree 브랜치로 push하지 않는다. W10 RRD149493626bytes가 일반Git100MB 제한 초과하므로 기존 git-lfs3.7.1 경로를 준비한다. 기존 학습/수집 데이터와 무관한 백업은 그대로 보존한다.

## 후속 실행 전 확인

`pre790_feedback_01/`에 T105 읽기 조회만 보존했다. rx_json107, boot_text0, 포트 close 완료. 원시 SHA256 `c5880770f76b38c867663b1674001d97f81e03ae513af906710c767283fcb531`. 실제 송신에서 이동·토크·PID 명령0. 현재 작업 공간/컵 치수는 이 피드백으로 확인할 수 없다.

실물 사진 후 잔류 제거·컵/로봇/더미 원위치·경로 정리와 컵 안지름/윗테두리 바닥 높이를 비동기 질문했다. 답변 없는 동안790 및 기울임 운동을 시작하지 않았다. 이는 실행 재승인 요청이 아니라 현재 물리 조건 입력이다. 기존 `hw_measured_scoop.py --torque 790`의 사전 IK·HOME/P1 시작 검사·부팅/피드백 중단·문0~30° 보호는 유지한다.

작은 손목 롤−5° 후보는 기존 회전 검토에서 출구 경사를 약0.279°만 늘린다. 이상적 출구5°와 동일한 시험으로 바꾸지 않는다. 컵 치수 없이 큰 관절 재배향이나 보호 범위 확장은 하지 않았다. 승인된 기울임과 고정 조건5회는 미완료다.

**Session progress rule 적용**: 이번 후속 요청에서는 아직 새 섭동 실험을 하지 못했다. 물리 준비 사실 입력이 오지 않아 dependent motion을 보류했기 때문이다. 대신 승인된 Git 게시와 원본 보존 검증을 진행했다. 새 학습·sim 변수는 이번 범위가 아니다. 이번 Git 검증은 파일/해시 감사이므로 RRD를 추가 생성하지 않는다. 기존 실기·기구학·W10 RRD/RBL/검수 증거는 모두 보존한다.

## 게시 전 검증

`git_publish_01/index_audit_01.json`: 준비 파일1008개, LFS14개, 일반 Git blob 최대19159214bytes,100MiB 초과0. `byte_integrity_01.json`:1008개 모두 index 내용 또는 LFS SHA256/size가 원본과 일치, Python79개 구문 분석 오류0. `manifest_audit_01.json`: 실기/배출 검토/계량 manifest88항목 모두 원본 해시 일치 및 index 포함.

장치 없는 `verify_recorded_arm.py` 재검사: rc0, `RECORDED_ARM_EXPLICIT_DOOR_TARGET_PRESERVED_OK; real serial opens=0`. 열린 문30°와 닫힌 문0°의 명시 목표가 팔 이동 T122에서도 유지됨을 가짜 접촉 피드백으로 검사했다. T106/T109/tor1000/허용 밖 문각 거부도 확인했다. 실기 성공으로 세지 않는다.

공백 검사는 CRLF를 줄 끝으로 인정한 뒤에도 기존 생성 OBJ의 마지막 빈 줄과 동결 Python/diff에108개 경고(rc2)가 남았다. 보고서에 보존하고 해당 원본을 정규화하지 않았다. 새 제어 코드/상태 편집의 검사는 별도로 시행한다. 전체 공백 검사 PASS라고 보고하지 않는다.

Git LFS와 원시 증거의 `-text` 속성은 원본 바이트를 유지한다. 무관한 로컬 g16 폴더·발표자료·print 도구·백업과 외부 worktree 입력은 이번 게시에 포함하지 않는다. 물리 입력 대기 중 확인된 산출물부터 게시하고, 실제 후속 실험 결과는 생긴 뒤 별도 추가한다.
