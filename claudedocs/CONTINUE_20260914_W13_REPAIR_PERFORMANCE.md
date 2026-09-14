# 새 세션 재개 — W13 판정 수정부터, 성능 계측은 별도 승인

작성: 2026-09-14. 이 문서는 9/11 재개 프롬프트와 9/13 GO를 대체하는 새 읽기 안내다. 과거 실행 명령을 재실행하지 않는다.

## 현재와 승인 경계

현재 완료된 일은 W11 dt 비교, W12 재생, W13 단일 부분 실행·감사, 랩미팅 설명 문서 전달이다. W13은 TIMEOUT/원자료 FAIL2/재생 FAIL3이며 전체 성공이 아니다. 이번 문서·Git 게시 세션에서는 연구 코드 수정·새 물리·재렌더·학습을 하지 않는다.

아래 §새 세션 요청문을 사용자가 새 세션에 전달하면, 그 요청 범위인 새 revision의 원자료 판정 수정과 CPU 테스트부터 진행한다. GPU 물리·렌더 실행은 그 요청문에도 자동 승인되어 있지 않다. A/B/C·물성·dt·문 동작·경로 최적화·학습으로 확장하지 않는다. 원자료 판정 수정과 운반 원인 조사/성능 개선은 구분된 단계다.

## 1. 읽기 순서

1. [AGENTS.md](../AGENTS.md), [START_HERE.md](../START_HERE.md), [DECISIONS_ACTIVE.md](DECISIONS_ACTIVE.md), [LEDGER_RECENT.md](LEDGER_RECENT.md), [from_codex.md](relay/from_codex.md). Codex는 프로젝트 부팅 규약에 따라 from_claude.md도 읽되 9/11 이전 상태를 현 상태로 되돌리지 않는다.
2. [이번 종료 세션](session_20260914_research_closeout_git.md), [출력용 설명](research/closeout_20260914/20260915_W13결과_실행시간_최적화와학습전략_출력용.md).
3. [W13 통합 보고](runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/REPORT_w13_resume_received.md)를 전체 읽는다. 필요한 실행 경위는 [W13 상세 세션](session_20260913_w13_resume.md)에서 확인한다.
4. 아래 원자료·독립 감사·설치 소스의 필요한 부분을 확인한다. 수치 인용 전 JSON/NPZ까지 내려간다.
5. `git status --short`, `git worktree list`, 필요한 브랜치/HEAD를 확인한다. 게시된 branch 목록과 제외 파일은 [Git 게시 보고](research/closeout_20260914/GIT_PUBLICATION.md)를 따른다. master에는 별도 worktree 코드가 자동 병합되지 않았다.

## 2. 원자료와 검수의 정확한 위치

생산 worktree: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle`

생산 기준 폴더: `claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/`

- `run_01/w13_cycle_seed460.npz` SHA256 `529f422e962730b17b520c3c9af2261aae7c431ec5e078cd2c368dfab0c46b0f`
- `run_01/w13_cycle_seed460.json` SHA256 `e482b939a4cf42e5284b9b15e36a9f3613e989161ec538475e2c05f4260d9340`
- `run_01/EXECUTION_RECEIPT.json`, `run_01/RUN_STATUS.json`
- `rev28/src/sim_w13_full_cycle.py`, `rev28/src/inventory_geometry.py`, `rev28/params_w13.json`, `rev28/criteria.json`, `rev28/REVISION_PIN.json`
- `partial_post_03/execution/EXECUTION_RECEIPT.json`
- `partial_post_03/isaac/w13_full_cycle.mp4` SHA256 `1471127134fc06a88d546c891de0d2515f4911fcd1832f0ae841258ac1a2bc34`。파일명과 달리 부분 실행 영상이다.

감사 worktree: `/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit`

감사 기준 폴더: `claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/`

- `RAW_SCHEMA_REQUIRED.md`: phase-only 전환, 모든 구체를 포함하는 분류 규약 정본.
- `REV28_PRODUCTION_PARTIAL_RAW_AUDIT_01.json`: 14/16, FAIL2 유지.
- `POST03_PARTIAL_REPLAY_RESULTS_AUDIT_02.json`: 12/15, FAIL3 유지.
- `REPORT.md`, `DEME_SOURCE_BOUND.md`, `SOURCE_EVIDENCE.md`와 그 ERRATUM 문서들.

메인 기준 폴더: `claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/`

- `ROOT_PARTIAL_RAW_SPOT_REPRO_01.json`, `ROOT_RAW_COHORT_VISUAL_CUES_01.json`
- `ROOT_POST03_AUDIT_REPRO_01.json`, `ROOT_RERUN_INSPECTION_01.json`, `ROOT_ISAAC_VIDEO_INSPECTION_01.json`
- `ORCA_FINAL_ACCOUNTING_01.json`: 기존 작업 종료 기록. user-owned 터미널이 열려 있다는 이유로 기존 GO가 활성이라고 해석하지 않는다.

LFS 파일이 포인터만 있는 새 clone이면 원자료를 바로 분석하지 말고 해당 브랜치의 LFS 확보 여부부터 확인한다. 다른 PC의 절대경로는 달라질 수 있으므로 branch·repo 상대경로·SHA256으로 대조한다. 기존 근거 파일을 경로에 맞추려고 이동/덮어쓰기하지 않는다.

## 3. 반드시 유지할 판단

- 31,218.753855초 실행했지만 물리 24.486802938176766초, 16,304 sync/283입자 프레임의 부분 종료다. child0만으로 성공 판정 금지.
- HOME 오차29.57491mm, home_hold0. 확정 용기 분류0/가능상한11은 정착 배출 성공이 아니다.
- PF107의 기록상 공구 내부144개 ID가 PF136에서 공구 내부0이 된 것은 운반 보유 문제의 단서다. 분류 오류와 실제 이동을 분리하기 전에 누출 경로를 단정하지 않는다.
- 원자료 FAIL2: phase-only 전환11개 대신 첫 행/subphase를 포함한25개; 모든 구체 내부 분류에서 최하단 대신 최상단을 비교하는 바닥 식 오류.
- 재생 FAIL3: DOOR_STOP5개가 모두 PF282에 잘못 연결; 결정PNG는 t0 더미/빈 그래프뿐; 관절 출처 집계338 대 실제283.
- >5m/s는 기존 경고, >20m/s는 중단 의미를 유지한다. 사후 합격선 변경 금지. W11 포획517알을 배출517알로 바꾸지 않는다.
- W11 dt 비교와 W13 다른 domain/유한벽/전체 경로의 비용은 단일 dt 대조가 아니다.

## 4. 권장되는 순서와 통과 조건

### 첫 case: 원자료 판정의 새 revision 수정·CPU 검수

이번 case의 신규 변수: [원자료 단계 전환/재고 분류 구현 정정]. 물리 조건 신규 변수0.

1. 기존 사전 규약을 읽고 실제 원자료에서 두 반례를 재현한다. 기존 실패 보고를 그대로 둔다.
2. 새 출력은 `claudedocs/runtime_logs/grasp_track/w14_w13_raw_repair_d484/<새 실행 ID>/` 아래만 만들고, 착수할 때 START Active Case에 경로를 기록한다. 그 경로가 이미 있으면 다른 새 ID를 쓴다. 현재 세션은 이 case를 시작하지 않았다.
3. 수정 코드는 동결 rev28을 편집하지 않는 새 revision으로 분리한다. 기존 원시 좌표/ID/시간/입력 해시는 그대로 유지하고, 수정된 분류·전환 결과는 파생 산출물이라고 표시한다. 원실험의 raw 계약 실패를 소급 PASS로 바꾸지 않는다.
4. CPU 단위 테스트는 기존 버그에서 실패하고 수정본에서 통과해야 한다. 원자료의 phase-only 기대 인덱스를 별도 식으로 계산해 대조한다. PF0/ID8 바닥 반례와 경계 안/밖/여유거리 사례를 확인하고, 매 프레임 입자 ID/라벨 총수 보존을 검사한다. 같은 production 함수를 양쪽 검사기로 재사용하지 않는다.
5. 코드·배열 검사만으로 기하 화면 검수를 마쳤다고 주장하지 않는다. 새 기하/접촉/궤적 판단을 내리면 D324/D341의 시각 산출·실제 검수를 적용하고, 필요한 실행 권한이 없으면 분리해서 요청한다. 순수 파일/스키마/해시 감사로 끝나는 부분만 RRD 생략 사유를 기록한다.
6. 완료 후 재생 결함3개의 새 revision 수정 범위를 먼저 브리핑한다. GPU 재렌더는 명령/입력/시간상한을 제시하고 승인받는다. 이 단계에서 운반 경로나 문 제어를 바꾸지 않는다.

### 후속 case: 운반 보유 원인 → 짧은 성능 측정

수정된 분류와 원시 좌표로 144개 ID의 이탈 시점을 분리한다. 실제 틈/자세/문 동작 원인은 별도 시각 근거와 제한된 비교로 확인한다. 성능 측정은 기존 누적 시간으로 알 수 있는 것과 새 계측이 필요한 것을 먼저 구분한다.

짧은 물리 프로파일링을 요청할 때에는 구간·초기화 비용·완전 checkpoint 부재·물성/dt/알 수/제어 고정·벽시계 총상한·정리 예산·중단 조건·출력 경로를 제시한다. 저장 NPZ에서 임의 restart하지 않는다. `DoDynamicsThenSync`→`DoDynamics` 교체나 sync 주기 변경을 무해한 가속으로 가정하지 않는다.

설치 기준: DEME2.4.0, IsaacLab2.3.0/Sim5.1.0.0, PhysX107.3.26, numpy1.26.0/psutil5.9.8, Rerun0.34.1. NVIDIA 의미·제한은 버전 일치 공식 문서와 설치 소스부터 대조한다.

## 5. 새 세션 요청문

```text
/home/cgxr/Documents/Robotics/RoArm_Project에서 이어서 작업해.
AGENTS.md, START_HERE.md, DECISIONS_ACTIVE.md, LEDGER_RECENT.md와 relay를 읽고,
claudedocs/CONTINUE_20260914_W13_REPAIR_PERFORMANCE.md를 전체 읽어.
연결된 최신 종료 세션·Git 게시 보고·W13 원자료/독립 감사로 현재 상태를 복원해.

첫 작업은 위 재개 문서 §4의 원자료 판정 수정 case야.
원자료 규약2결함을 재현한 뒤 새 revision의 수정과 CPU 단위/회귀 테스트까지 진행해.
원본 run_01/rev28/post03 및 기존 실패 보고는 편집하지 말고 새 경로에 저장해.
기존버그 FAIL→수정본 PASS와 독립식 대조, 원자료 해시 보존을 확인해.
기하·궤적 판단에는 프로젝트 시각 검수 계약을 적용하고, 필요한 GPU 실행은 먼저 승인받아.

메인만 상태 원장·relay를 소유해. 별도 작업자 배정이 필요하면 orchestration 절차로
적합한 Orca worktree에 배정하고 여기서 보고 받아.
Claude는 claude-opus-5, Codex는 gpt-5.6-sol high를 실제 확인해.
사용할 worktree/브랜치가 게시된 위치와 맞는지 먼저 검사하고 임의 merge하지 마.

그 다음 재생3결함 정리, 운반144개 ID의 보유 원인 조사,
짧은 동일조건 성능 프로파일링을 각각 분리해 권장 순서와 승인안을 보고해.
이번 요청만으로 새 DEME/Isaac GPU 실행·장시간 재실행·학습·A/B/C를 시작하지 마.
로봇 조회·구동·PID/토크 변경·카메라 수집·설치는 금지해.
dt/물성/알 수/경로/문 제어/보호선과 기존 합격 기준을 임의 변경하지 마.
이번 세션 commit/push도 새 명시 요청 전 하지 마.

관찰 가능한 절차→수치→근거 파일→한계·다음 승인 경계 순서로 한국어로 보고하고,
종료 시 START_HERE·새 session·relay를 갱신해.
```
