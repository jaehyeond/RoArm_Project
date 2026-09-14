# W13 본 실행 기술 GO — 동결 rev28, 새 run_01, 단 1회

발행: 2026-09-13 21:22 KST. Root coordinator / Run `run_5cba1e55a775`.
producer Task `task_d3ad7187e682` / Dispatch `ctx_006e285a14a4` 전용.
사용자의 기존 최대9시간 물리 계산 비용 승인과 재개/위임 요청에 근거한다. 추가 비용 승인 대기 상태가 아니다.

## 인수한 결과

1. 경로 연결 검사·수치 입력·원자료 수지/회전·실제 실행기 음성 대조는 미변경 해시로 재사용한다. root는 최종 rev28 COMMANDS/manifest 전문과 변경3소스 diff를 읽었다. 물리 params/seed/dt/보호선/과학 criteria 변경0.
2. 정확한 GO28 준비1회는 새17_actual에서 21:14:51~21:15:31, 정리포함40.219416초≤180, child/runner/launcher0, timeout/신호/잔여그룹0으로 종료했다. 필수6시간 기록 및 close21.252초 반환. 실제 소유셀4개/publicinstance 모두None, 필수4weakref해제, sim약한참조생존은 사전에 고정한 정보항목으로 기록했다. 특정 native 원인이나 향후 전체실행 성공의 보장은 아니다.
3. root는17_actual 원본8장을 직접열어 검수했다. `ROOT_READINESS_INSPECTION_08.json` SHA `2ec8806e0c6a684d55dd68a066cbc1ec36975bd45e36e58263cbb74e4f3106fe`. raw2+명시합성6은 표시 준비이며 전체 물리 결과가 아니다. 기존actual16 FAIL소급변경0.
4. 독립 `REV28_READINESS_ACTUAL_AUDIT_01.json` 19/19 PASS SHA `879e4bae3bd224ce52ef7f7eb337eb12694f2dd541f37d915608a1bd8ef97964` 전문과 해시를 root가 확인했다.
5. root 명시08/17actual/최종rev28 `verify_resume_v2.py --check-preflight` 66/66 rc0 `W13R_PREFLIGHT_V2_VERIFIED`를21:20 직접확인했다.
6. 독립 `REV28_PRODUCTION_PREFLIGHT_AUDIT_01.json` 11/11 SHA `81f650c82b808e2b27fbcc3f75daf1e5c76e9d3e8e951f3c719ca1b34384b1d0` 전문/시험185줄을 읽고 동일시험을 새 `REV28_ROOT_PRODUCTION_PREFLIGHT_RECHECK_01.json`에 재실행11/11 rc0, SHA `667535715e4968bbaa32ae7923f886355181de2fecd5f21aac8000e4c60b620c` 확인했다. 과거검사를 무조건 재실행하지 않고 미변경 코드는 핀으로 묶었다.
7. source26/생산external13/criteria1 해시일치 및 run_01부재 확인.21:21 기존508보존/HEAD PASS, isaaclab numpy1.26.0·psutil5.9.8·Rerun0.34.1·IsaacSim패키지5.1.0.0 확인. 설치/삭제0. 독립21:15 디스크190118133760bytes 여유는 예상최대6.5GB보다 크지만 저장추정 보장은 아니다.

## 정확한 동결 입력

- revision: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev28`
- PIN: `64d2cc2e1b603696f10c5edfeed6a67fd2ab6f2c3d546500416ad1ecad72068a`
- COMMANDS: `6f3eafd951ed041df19cfb55177ade9d06a3574d7d95e926cb2b09ec7f0959e6`
- prospective manifest: `8d5de888166ec2b50f25af528107ce6f180a459b285d89d3e94d0b97581ca23a`
- runner: `bd8fa702ff5a9c06286de1cafd3702487803103fe747bcc3d3a8fae483cb81d5`
- simulation: `aa4f570bbdca96c1abbae995ebb1aef7b10870ce7714b63519b0a27387de1d32`
- Rerun exporter: `35a10aa666ff2e76d5e8dd39442cbb4978bb954cb5b523a6ff7ba1fac3af87c8`
- Isaac renderer: `d012e0fcd6b581a13d06053430519e00a8469ec17e46ce930ebb5f25ab9ea718`
- params: `95101f8b799518f83a0d8a0004057e50ea01120b66681334a9911a7e5fa7e661`
- numeric inputs: `02ad1171cef8bc85eb9d589f574887f85e05052ed23e6964f7dc6651c3a915a3`
- criteria: `71a7d23938a341c2689b570fa5a1e9bec74b7b3392f754ca55f61e7edd646196`
- pile: `659d6b0bc771678a0c7209d91f550edc933d03e41922245ea0adb64eeb818812`

## 정확한 실행 명령 — 이 실행기를 한 번만 시작

cwd: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation`

```bash
PYTHONDONTWRITEBYTECODE=1 /home/cgxr/miniconda3/envs/roarm/bin/python -B /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev28/src/run_production.py /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev28
```

실행 직전 inbox 읽기/핀 전체 재검사/새 run_01 부재/환경핀/디스크 확인. 이상 있으면 시작하지 말고 보고한다. run_01을 미리 생성하거나 기존산출을 복사하지 않는다. frozen COMMANDS의 argv를 변경하거나 문자열로 재조립하지 않는다.

## 단계와 중단·보고 계약

1. `step1_simulation` 새20,000알·seed460·dt1μs 연속 한 사이클. 기존 초기더미에서 시작하며 W11최종NPZ를 이어붙이지 않는다. bridge 검사 실패 시 첫 bridge 물리 step 전에 멈추고 `CLEARANCE_UNCERTIFIED`를 그대로 남긴다. 결과를 보고 경로/물성/임계값을 조정하지 않는다.
2. **물리 단계 총32400초에는 정리시간이 포함된다.** softcap/TERM31200초, KILL32399.5초, 총32400초 경계. 9시간+추가1200초로 해석하지 않는다. 실제OS시간 보장은 선언만으로 주장하지 않고 영수증으로 검사한다.
3. 실행기가 성공 프로세스 경로로 끝낸 원자료만 기존순서 `step2_rerun`→`step3_isaac_replay`, 각각 별도5400초 이내로 후처리한다. 두GPU작업 동시금지. guard에 따른 계획된 조기종료가 process0인 경우에도 partial/abort 라벨을 지워서는 안 된다. 외부timeout/예외로 실행기가 다음단계를 중단하면 수동 후처리나 새실행으로 우회하지 말고 root에 보고한다.
4. Rerun SDK/CLI0.34.1 고정, 전체 실행 sync/모든 저장 PF/실제 접촉·힘·용기/결정값 포함, 파일 종료 후 footer/exactentity·timeline·component/RBL검사, headless 결정PNG와 실제시각검수까지 필요하다. Isaac은 전체 저장 PF를 행 정체성 보존하여 재생한다. synthetic준비를 actual결과에 혼합하지 않는다.
5. 시작 UTC/정확 argv/runner 및 child PID·PGID/직전 해시를 즉시 보고한다. 진행 중 파일을 덮지 않는 상태읽기만 하고 phase/실제물리시간/벽시간/관측경고를 보고한다. 무변화만으로 실패·재시작 판정하지 않는다.
6. 한 실행 완료 또는 중단 뒤 자동재시도0, 수동신호0, 추가dt/seed/물성스윕0. 원자료·실패·stderr·모든영수증 보존. 완료 후 단계별rc/시간/그룹부재/과학중단사유/산출목록을 보고하고 독립검수 요청을 받는다.
7. 실물 조회·구동·PID/토크·카메라수집/학습/A·B·C/패키지설치/commit/push 금지. 메인 상태·relay·원장은 root 배타소유다. producer는 지정worktree의 새implementation 산출만 소유, 감사는 별도audit 소유. 기존코드/동결파일 수정금지.

이 GO는 **동결 rev28 한 번의 실행과 선언된 순차 후처리**만 허용한다. 전체cycle완료·전달질량·정착·실물가능성은 아직 관측되지 않았으며 실행 후 근거로 판정한다.
