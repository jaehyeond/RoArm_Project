# 제한적 GO — rev20 Isaac 준비 표시 1회

발행: 2026-09-13 17:33 KST. coordinator /root. Run `run_5cba1e55a775`.
대상: 기존 producer `ctx_006e285a14a4` / `task_d3ad7187e682`만.

사용자09-13 W13 재개 승인 안의 표시 준비다. **아래 런처 정확히 1회만 허용한다.** 새 DEME/production 허가가 아니다. 소진된 rev17 GO/실패 영상/미실행 rev18~19는 모두 보존하며 소급 승인·자동 재시도하지 않는다.

## 사전 인수 근거

- root가 rev19 helper전체 및17→19 renderer/launcher diff,19→20 launcher diff를 읽었다. 마지막 변경은 mini 실행사본에 누락된 모듈 한 항목 추가다.
- root 격리 CPU import: 옛09_planned는 ModuleNotFoundError/rc1, 새10_planned는 실제 mini 모듈 경로에서 rc0·6링크/48모서리/7자산 읽기. 새GPU 없이 결함과 수정을 확인했다.
- root 직접 해시 재계산: 후보 `READINESS_CANDIDATE_09.json` SHA `66d8acc5fc5699c5c4a9cc2d9bbe893214ec0c51ee05dc949a3f79bd47a713ce`; rev20 pin26/26, planned mini9/9, 원입력15/15 모두 일치, actual경로 부재.
- rev20 pin SHA `c7fe29906388587fc34092dc794c9fdaf2bd719b5165d6003bdc608990ad4350`.
- launcher `12178e429baedb227b1b5986d8ef167f9825b9d0f2dca3f2162abcc6190c43e8`; renderer `c510628118a75274cebb58e78d61c20903cda02a93a79ca47398b1c13e0f5090`; arm helper `0ba2949f0e6e0055993e5be6810d8a80c5a36a63bd48fe7181f9b2d1947863f7`.
- runner `bd8fa702ff5a9c06286de1cafd3702487803103fe747bcc3d3a8fae483cb81d5`, 물리sim `aa4f570bbdca96c1abbae995ebb1aef7b10870ce7714b63519b0a27387de1d32`, camera `600aef751a0d5abd2ce6c665e8ec42a2b3f7f28a53b5b626aecd9a5981f8a33c`는 기존 인수 해시를 재사용한다. params/criteria 불변.
- 독립 감사 `REV19_CANDIDATE08_CHANGED_PARTS_AUDIT_01.json` SHA `2652b778ab0a9a916c650e2683a148a608b6badd13614bb7dd281f59d425cc65`: STL 전체 정점 포함/8자세×48모서리 독립변환 차이0. 준비6합성 S1 216정점도 현재 두 카메라 안에 들어감을 별도 확인했다. S1합성 정점은 선언된 구도 점집합에 직접 추가된 것이 아니므로 이6자세 외 일반 보장으로 확장하지 않는다.
- 독립 최종 `REV20_CANDIDATE09_DELTA_AUDIT_01.json` SHA `3eb4aea8a61ed1e44ce8d50e1e27091962356fb20cf79c149414ed3d9b65a1f8` 9/9를 root가 전문/해시 확인했다. 실제 화면 품질과 close 완료는 아직 미검증이다.
- 기존508파일/HEAD 보존17:31 호스트 읽기전용 검사PASS. 샌드박스의 git spawn EPERM은 host 재검사로 구분했으며 증거 변조로 해석하지 않았다.

## 정확한 명령 및 실행 전 검사

cwd: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation`

```bash
PYTHONDONTWRITEBYTECODE=1 /home/cgxr/miniconda3/envs/roarm/bin/python -B /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev20/src/readiness_launcher.py --run /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/implementation/attempts/smoke_wall_regression_01 --out /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_10_actual --budget-s 600 --max-frames 8 --synthetic-phases 6 --criteria /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev20/criteria.json --res 1600x900 --fps 10 --usd /home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd
```

실행 직전 위 후보/26파일 pin 및 `readiness/renderer_readiness_10_planned/plan/revision/MANIFEST_prospective.json` SHA `6d81c219e6ade7cd72180422057eaeb9ed13647b7b9ee94279f7e9d03e2541d6`와 그 안의 입력15+criteria1 해시를 재확인한다. 한 건이라도 다르면 실행하지 않고 보고한다. 새10_actual이 없어야 하며 그 안에 로그를 미리 넣지 않는다. populated10_planned는 실행하지 않는다. 런처가 새actual에 별도 plan/mini/attempt/render를 만들며, planned→actual 경로 및 동일바이트criteria 사본의 경로 차이만 허용한다.

## 실행 및 판정 경계

1. 종료포함 stage600초 = render495/close60/grace45. 기존 runner의 단일 마감/조기TERM/KILL/정리 확인과 timeout 비성공 규칙을 유지한다. Python 알람의 실제 native close 제한은 보장하지 않으며 외부 실행기가 최종 경계다. `close(wait_for_replicator=False)`는 설치 IsaacSim5.1의 공식 후보이지 완료가 입증된 해결이 아니다. skip_cleanup/force-exit0/기준완화 금지.
2. 원자료2행+선언된6합성자세 총8장. raw0/1은 같은 t0/sync0이며 두 시간대로 바꾸지 않는다. 합성에는 실제 원자료 시간 없음/기존 입자복사/NOT A PHYSICS RESULT 라벨을 유지한다. 초기 Isaac reset/warmup8은 표시 초기화로 별도 계측, 표시 프레임은 저장값 설정과 render만 하며 시계 전후를 기록한다. DEME 입자동역학0.
3. 실제 시작UTC/argv/PID·PGID, 종료rc/총시간/그룹부재/해시 확인, manifest/mapping/8PNG/MP4 및 READINESS_RECEIPT·EXECUTION_RECEIPT·RUN_STATUS를 새 경로에 보존한다. 프로세스 확인은 정확PID·PGID와 영수증으로 하고 자기 일치 가능한 pgrep 대기 패턴을 쓰지 않는다.
4. 종료 뒤 root가8장 모두 직접 검사한다. 실제 S1셸/문·팔 전체·더미·원래지지면/4벽·용기·운반/배출위치/HOME·합성/시간 라벨을 확인하고 잘림/빠짐/가림을 분리한다. rawHOME 대 syntheticHOME 문 조립 일치도 확인한다. 명령 FK오차0은 실제 USD 오차 증명이 아니다.
5. 한 번 끝나면 GPU는 다시HOLD. 실패/timeout/미측정이면 실패로 보존한다. 새root 실제이미지 인수·최종독립사전검수 전 본 실행 금지. 실물조회·구동/PID/토크/카메라수집, 학습/A/B/C/설치/commit/push, 코드·기준 변경과 추가 자동실행은 금지다.
