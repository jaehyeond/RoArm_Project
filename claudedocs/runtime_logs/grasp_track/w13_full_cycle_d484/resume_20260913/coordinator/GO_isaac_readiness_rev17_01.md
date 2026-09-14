# 제한적 GO — rev17 Isaac 준비 영상 1회

발행: 2026-09-13 16:44 KST. 소유 coordinator /root, Orca Run run_5cba1e55a775.
대상 producer ctx_006e285a14a4 / task_d3ad7187e682만. 사용자09-13 W13재개 승인의 준비 단계다.

## 허용 범위와 선행 확인

- **아래 런처 1회만 허용한다.** 새 DEME 입자 물리/production GO가 아니다. 기존 smoke 원자료2행과 선언된6자세의 표시8장, 합성은물리결과아님.
- root는 rev17 pin25/25, 계획 mini pin8/8을 실제재해시했다.16→17 소스차이는 renderer의매핑된 source floor 표시뿐이고 COMMANDS/WORK_REVISION_NOTE경로만갱신됐다. 바닥은 raw의DEME z0→display z0.2188192277063058m에 상면을두는 두께2mm 비충돌표시판이다. 물리바닥수정아님.
- 독립검수에서 받아들인14runner/numeric 및15launcher실제CPU연결5대조+16입력/6합성계획의해시동일증거를재사용한다. 실제화면품질은아직미인수다. 기존508파일/HEAD보존은16:38재검사PASS.
- rev17 pin SHA256 `e111a1e010a2983c8e22d71ea17ae6ecd2176892b7af8a2c12257b8e50f73198`.
- launcher `019b86336f0a38f9a78d611b841a47120856e76ea3e216addc5f44e1efc2c36a`; renderer `b8def2640db8fc7fa74d09aee57a2af4b24663c6c7865ccae0a41672f244a3e9`; runner `bd8fa702ff5a9c06286de1cafd3702487803103fe747bcc3d3a8fae483cb81d5`; camera `600aef751a0d5abd2ce6c665e8ec42a2b3f7f28a53b5b626aecd9a5981f8a33c`.
- 원본입력8건+criteria해시는 이미 검수한 `readiness/renderer_readiness_07_planned/plan/revision/MANIFEST_prospective.json`에 고정돼 있다(그파일 SHA `2147db6df918f16eefbdbd4aab348f5ae6ee0f4fcc682a288cab43425f738909`). 실행직전 이파일과선언된입력·rev17 pin을재대조하고 한건이라도다르면실행하지말고보고한다. 기존planned디렉터리는실행하지않는다.

## 정확한 실행

cwd: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation`

```bash
PYTHONDONTWRITEBYTECODE=1 /home/cgxr/miniconda3/envs/roarm/bin/python -B /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev17/src/readiness_launcher.py --run /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/implementation/attempts/smoke_wall_regression_01 --out /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_07_actual --budget-s 600 --max-frames 8 --synthetic-phases 6 --criteria /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev17/criteria.json --res 1600x900 --fps 10 --usd /home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd
```

실행직전 새 actual 출력경로 부재를 확인한다. 이경로에 미리로그/계획을 넣지 않는다. 런처는자기새경로에서계획·mini revision을만든뒤검수된실행기로넘기고 `READINESS_RECEIPT.json`을쓴다. 따라서 plan-only로만든폴더를같은런처로재사용하는충돌이없다. planned→actual 출력경로와 criteria의동일바이트rev17사본경로차이는이번정확명령에명시적으로허용한다. 소스/입력바이트/물리/예산차이는허용하지않는다.

## 경계와 제출

1. 총 readiness stage cap600초(종료포함), 내부render495/close60/grace45; 검수된runner의조기TERM/KILL/남은예산wait유지. 시간초과/rc97/미확인종료는성공아님. 자동재시도0, 추가프레임/카메라변경0.
2. Isaac 초기reset/warmup의장면물리스텝은허용된표시초기화이며별도계측한다. 새DEME 입자동역학0. 표시프레임은저장값설정/렌더만하고시계비진행을기록한다.
3. 시작UTC/실제argv/PGID와 종료rc/총소요/정리확인, 입력해시검사,render manifest·mapping·8PNG·MP4·READINESS_RECEIPT·EXECUTION_RECEIPT·RUN_STATUS를새경로에보존한다. 기존실패INSPECTION은절대덮지않는다.
4. 종료후 실제8장검수: S1고정셸/문·로봇·더미/원래지지면·4벽·배출용기·운반/배출위치/HOME·합성라벨/원자료시간. 잘림/빠짐/가림/실제USD vs명령FK오차를사실대로분리한다. rc0만으로준비PASS를선언하지않는다.
5. 이1회가끝나면GPU는즉시다시HOLD. root의실제이미지검수와독립사전검수전production을시작하지않는다. 새동결코드수정/자동반복/실물조회·구동/PID/토크/카메라수집/학습/A/B/C/설치/commit/push는허용하지않는다.

원래준비3회실패와계약초과이력은보존한다. 이번GO는그이력을소급승인하거나지우지않는별도1회한정허가다.
