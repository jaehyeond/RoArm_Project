# W13 부분 원자료 재생 기술 GO — post03, 단 1회

발행: 2026-09-14 06:41 KST. Root / Run `run_5cba1e55a775`.
producer Task `task_d3ad7187e682` / Dispatch `ctx_006e285a14a4` 전용.
사용자가 승인한 미사용 순차 후처리 범위다. 새 물리·9시간 연장·전체 사이클 재시도 승인이 아니다.

## 인수 근거 및 한계

- root 실제 CPU 시험27/27 rc0 및 `POST03_ROOT_PRECHECK_01.json` 67검사/불일치0. 원본14파일·복사2파일이 runner가 실제 소비하는 `external_frozen_inputs_sha256`에 포함된다. post01/02는 설명용 키만 사용했던 결함 때문에 미실행 보존한다.
- 독립 `POST03_PARTIAL_REPLAY_PREFLIGHT_AUDIT_01.json` 전문/해시 확인: 6/6, SHA `7f99785e0be93a8e9b2810f70880b5b3f0682154cd8218937cf33783934927a0`. 격리 /tmp의 실제1바이트 변경을 동결 runner가 rc3/steps[]/자식로그0으로 거절했다. 실제 raw 변경0이다.
- source26/external29/criteria1 동결. 원본 rev28 대비 명령/경로 외 소스 변경은 Rerun의 표시 경고뿐이다. 물리/과학 criteria/renderer/runner/원자료 불변이다. 기존 WARN 추가는 저장 후 실제 기존 결정/문/bridge 사건이 보존되는지 확인해야 한다.
- 실제 물리 원자료 감사는14 PASS/2 사전 규약 FAIL, SHA `bd0cc6ad1be561d3deb64a7c0552609ed66f4c7c205415907da69fad7ebbe2a3`. phase-only 전환 인덱스와 strict whole-sphere source 분류 불일치를 유지한다. 재생은 이를 고치거나 통과로 바꾸지 않는다.
- 원래 실행은 TIMEOUT/SIGNAL_STOP, 전체 HOME 및 정착 미완료다. 저장283 PF/16304 sync의 부분 관측만 재생한다. 입자색은 기록된 분류이며 분류 규약 통과 증거가 아니다. 확정 용기 분류0/가능상한11, 정확한 정착 배출량 미확정이다.
- 06:32 root 설치 metadata: numpy1.26.0/psutil5.9.8/Rerun0.34.1/IsaacSim5.1.0.0. 설치/설정 변경0.

## 정확한 입력 핀

revision: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/post_03_rev`

| 입력 | SHA256 |
|---|---|
| REVISION_PIN.json | f014a326228ab1db29c11b29e1ec4358cfa5fff2c817ec95e2c327f2ca6579cb |
| COMMANDS.json | e65f33f64ee5a43569eda91a3343edf3aa155d33376a455457515cec99d40d8f |
| MANIFEST_prospective_post03.json | eaf67ec6d1588d0883cf14c256f6bce7fcd57cad4dde73cb28585ed68eb37712 |
| src/w13_rerun_export.py | 7e56c6e62853c0b6ff9070349fc459c2f07be4102dc4cbf3ad412c41de4fd441 |
| src/isaac_replay_w13.py | d012e0fcd6b581a13d06053430519e00a8469ec17e46ce930ebb5f25ab9ea718 |
| src/run_production.py | bd8fa702ff5a9c06286de1cafd3702487803103fe747bcc3d3a8fae483cb81d5 |
| criteria.json | 71a7d23938a341c2689b570fa5a1e9bec74b7b3392f754ca55f61e7edd646196 |
| 원본/복사 w13_cycle_seed460.json | e482b939a4cf42e5284b9b15e36a9f3613e989161ec538475e2c05f4260d9340 |
| 원본/복사 w13_cycle_seed460.npz | 529f422e962730b17b520c3c9af2261aae7c431ec5e078cd2c368dfab0c46b0f |

## 정확한 실행 — 한 번만

cwd: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation`

```bash
PYTHONDONTWRITEBYTECODE=1 /home/cgxr/miniconda3/envs/roarm/bin/python -B /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/post_03_rev/src/run_production.py /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/post_03_rev
```

직전 inbox/전부 핀/환경/디스크 확인. execution/는 빈 기존 디렉터리이며 input/에는 byte-identical 복사2개만 있다. input/rerun/와 isaac/는 부재여야 한다. 이상 시 실행하지 않고 보고한다. 원본 run_01은 읽기 전용, 파일 추가도 금지다. 동결 argv를 재조립·수정하지 않는다.

1. 정확한 순서는 step2_rerun → step3_isaac_replay, 물리 step1은 없다. 각 단계 총5400초에는 정리가 포함된다(TERM4200, KILL5399.5/정리0.5). GPU 동시 실행0, 재시도0. 실패 시 수동 다음 단계 우회0.
2. 출력은 `implementation/partial_post_03/{execution,input/rerun,isaac}/`만이다. post01/02 및 원래 run_01을 덮지 않는다. partial/abort 경고를 유지하며 합성 준비 데이터를 섞지 않는다.
3. 시작 UTC/runner·child PID/PGID/원래 argv/실제 시작 해시 영수증을 즉시 보고한다. 단계별 종료rc·시간·신호·잔여 프로세스/그룹·산출목록을 보고한다. 파일 존재만으로 검증 완료라 하지 않는다.
4. Rerun0.34.1 파일 종료 후 footer verify/exact entity·timeline·component/RBL/headless 결정PNG 및 실제 시각 검수가 필요하다. 원본 callback 배열/JSON이 과학 정본이며 Float32 표시를 되해시하지 않는다. 기존 결정/door/bridge 사건과 마지막 partial WARN의 실제 저장 보존을 감사한다.
5. Isaac은 실제 저장283 PF를 정체성 보존하여 재생한다. 영상 매핑/필수 종료6시간/rc/전체시간/정상close를 감사하며 root가 실제 영상 표본을 직접 본다. 저장 구간 재생과 전체 물리 사이클 완주를 분리한다.
6. 사후 원본14파일 불변은 runner 자동 기능이 아니라 root 인수 검사다. 아래 고정 checker를 재생 종료 후 실행해 새 영수증을 남기기 전 산출 인수하지 않는다.

root checker: `coordinator/verify_partial_raw_preservation.mjs` SHA `96aa8ec56289b5b5bbcc9f7b14f3b41f9035e6499b78c8630322ab1587b305fb`.
root manifest: `coordinator/PARTIAL_RAW_MANIFEST_01.json` SHA `0d7fb7ce6d5d24d0a29b0c25ccd02e633a803239c10192b16aea0c8553bb2a52`.
사전 해시 검사와 별도로 원자료 사후 불변 확인을 필수 인수 경계로 둔다.

실물 조회/구동/PID/토크/카메라 수집, 학습/A·B·C, 설치, commit/push, 물리/criteria 변경 금지. 메인 원장/relay는 root 배타소유다. producer는 지정 implementation 새 산출만, 감사는 지정 audit 새 산출만 작성하며 apply_patch 사용/다른 편집 보존 규칙을 유지한다.
