# 제한적 GO — rev28 Isaac 준비 표시 1회

발행 2026-09-13 21:13 KST. root / Run run_5cba1e55a775.
대상 producer ctx_006e285a14a4 / task_d3ad7187e682. 아래 새 실행 단 1회 후 GPU HOLD. 자동 재시도 금지. DEME production GO가 아니다.

## 인수 근거와 사전 기준

- 기존 actual16은 종료가 반환했지만 당시의 5개 weakref 즉시 사망 조건에 실패한 rc96 결과로 보존한다. 소급 PASS 변경은 없다.
- root가 추가한 즉시 객체 소멸 조건과 renderer 소유 참조 해제는 다르므로, 공식/설치 IsaacLab2.3 API 대조 후 사전 고정한 `REFERENCE_RELEASE_CONTRACT_REV2.md`(SHA3378b958778ce6fa3a6dbc276c02c69557248685b36988325acfdba44e37fb68)를 적용한다. 4개 실제 소유 셀 None 및 공개 instance None을 엄격한 5-key bool mapping으로 측정한다. 카메라2/scene/robot weakref 사망은 필수이며 sim weakref 생존 여부는 반드시 기록하되 정보 항목이다. 정확한 sim 잔류 소유자/누수/안전성은 주장하지 않는다.
- 정상 close 반환·필수6시간·child/runner0·timeout/신호 없음·전체180초·프로세스 그룹 부재 조건은 유지한다. 과학 criteria·물리·경로·runner는 변경하지 않았다.
- 독립 `REV28_RELEASE_CONTRACT_AUDIT_01.json` 20/20 SHA d88e6f026cd63388d6a68bf6decaa1e2e78b04d3a6207e7eccf8228f37ce1c7b. root가 전체 시험/보고를 읽고 새 `REV28_ROOT_DELTA_RECHECK_01.json`에 동일 시험 재실행20/20 rc0, SHA aced71d52aa248e139616d3c5603ae76398a3df1042bcb9189c2b5927d2e42fd. 실제 동결 콜백 AST 실행과 각 None 대입 제거 음성 대조를 포함하나 GPU/native 효과 검증은 아직 아니다.
- root 기존508 보존 검사 PASS, source26/mini10/external15/criteria1 총52해시 불일치0, 새 actual17 및 run_01 부재 확인.
- rev28 PIN `64d2cc2e1b603696f10c5edfeed6a67fd2ab6f2c3d546500416ad1ecad72068a`; renderer `d012e0fcd6b581a13d06053430519e00a8469ec17e46ce930ebb5f25ab9ea718`; helper `d6f4daf26e066554d6b3347adc257796235387296f945d3ae47bb0dd50c1b4c6`; launcher `b9a82dac69d457772321d054ddc9c48d19591efb336e779736db2c27ba8a971c`.
- 17planned plan `5d66620e7393a4a9b83d75306d478db05e34050a66a1cc6424ba0e591677b45c`; mini PIN `0fdecaeced07e2e15a5eaac4c9b7636c3a6dde019625394581da05c23cf2254a`; mini MANIFEST `e9c8532db0c787c717651444011e52802cb960cc92b0ef6a6172b557cf3a3a0d`.

## 정확한 실행

cwd: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation`

```bash
PYTHONDONTWRITEBYTECODE=1 /home/cgxr/miniconda3/envs/roarm/bin/python -B /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev28/src/readiness_launcher.py --run /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/implementation/attempts/smoke_wall_regression_01 --out /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_17_actual --budget-s 180 --max-frames 8 --synthetic-phases 6 --criteria /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev28/criteria.json --res 1600x900 --fps 10 --usd /home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd
```

실행 직전 핀 및 actual17 부재 재검사. populated planned는 실행하지 않는다. 새 actual 사본의 자기경로·동일바이트 criteria 사본 경로만 변경 허용.

## 관찰·중단 경계

1. 총180초 정리 포함: child135/cleanup27/grace18, TERM162/KILL179.5. clear→참조해제→probe→close 실제 순서와 소요시간, exact5-key bool 분류 및 5개 weakref 관측을 원본 영수증으로 제출한다.
2. 참조·분류·관측 실패rc96, close timeout97/외부124 등 비성공 유지. 수동 신호/예산 확대/강제rc0/skip_cleanup/GC/직접소멸자 호출 금지. GPU 자동 재시도 금지.
3. raw2행은 t0/sync0, 합성6행은 기존 입자와 선언 자세다. 새DEME0·전체물리결과 아님. 초기warmup8과 표시시계8프레임을 구분한다.
4. UTC/argv/PID·PGID/rc/timeout/원본 영수증/8PNG/MP4/매핑을 보고한다. root8원본PNG 직접검사와 독립 종료·매핑 감사 후 인수한다.
5. 이후 HOLD. 사용자9시간 비용승인은 유지되지만 준비 인수+최종rev28 production preflight 뒤 정확한 root 본실행 GO가 별도로 필요하다. 실물/학습/A/B/C/설치/commit/push 금지·원장root소유·기존실패불변.
