# 제한적 GO — rev23 Isaac 준비 표시 1회

발행 2026-09-13 18:54 KST, coordinator /root, Run run_5cba1e55a775.
대상 producer ctx_006e285a14a4/task_d3ad7187e682만. 사용자 W13 재개 승인 안의 준비다.
아래 정확한 런처 1회만 허용. 종료 뒤 GPU HOLD. 새 DEME/production 권한은 아니다.

## 사전 인수

- root가 rev22→23 실제 launcher diff(두필드 검사/문구 두곳), delta 전문,13_planned 계획 전문을 읽었다. COMMANDS 자기경로와 launcher만 변경, renderer/CD/runner/physics/numeric/criteria/params 불변이다.
- root 직접 재해시: source26/26,mini10/10,external15/15,criteria1/1 불일치0. 13_actual 부재. 기존508파일/HEAD 보존 검사 rc0·W13R_PRESERVATION_VERIFIED 508.
- delta SHA14031899d15fa10ffafc4dda62a36b053d56805c7b8497d32a5650e0421ccc98.
- rev23/REVISION_PIN.json SHA39161db44023ee5eace1625e0420aac6c1120e8a9fc860a4b61f6b27d75631c6.
- launcher SHA3460dd6b461f4ae943b6d444b68428258df5e36d215b6c13b3e59163ef6106ac.
- 13_planned/plan/readiness_plan.json SHAab9c521e40f4fe776e05c3b4c46603deda4175c740a3bfdfa9d06577ff0d0c78.
- planned mini pin SHA4d849a28780e55e70e641a41cabeaab652ee82902251287225e8841136135dfb; MANIFEST_prospective.json SHA0d8dcac206ba7a0c73d9c88b858f149094c1a6dd3921186274358ef54a87e6bb.
- 독립감사 REV23_DELTA_INDEPENDENT_AUDIT_02.json 전문/해시e6355e38e66885331fa5e5eab566f5e14c23cb1a142596421bb0d177ed811a72를 root 확인,8/8. rev22 인수13개는 동일해시 재사용이다.
- root 실제AST검사식 stdlib대조5/5: rev23 전체6필드존재True, 새2필드 각각누락False, rev22 같은누락True. 최초 pytest실행은 roarm에pytest없어 rc1/미실행; 설치하지 않고 이 좁은 독립대조로 확인했다. 생산9/9 전체 재실행으로 주장하지 않는다.
- native종료/실제화면은 아직 미검증. close진단은 capture-off선행이라는 종료상태/순서 개입도 포함하며 순수관찰만이라고 주장하지 않는다.

## 정확한 실행

cwd: /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation

```bash
PYTHONDONTWRITEBYTECODE=1 /home/cgxr/miniconda3/envs/roarm/bin/python -B /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev23/src/readiness_launcher.py --run /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/implementation/attempts/smoke_wall_regression_01 --out /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_13_actual --budget-s 180 --max-frames 8 --synthetic-phases 6 --criteria /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev23/criteria.json --res 1600x900 --fps 10 --usd /home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd
```

직전 위26핀/mini10/외부15+criteria1 및delta/plan/manifest SHA 대조. 불일치나13_actual기존존재면 실행금지·보고. populated13_planned는 실행하지 않는다. 런처가 새13_actual에계획/사본을 만들며 planned→actual 및동일바이트criteria사본의경로차이만허용한다. immutable22/23/옛실패0수정.

## 관찰·판정 경계

1. 단계총180초에정리포함: child135/cleanup27/grace18,외부TERM162/KILL179.5. 실제monotonic시간,6필드계측·진단call전후·close남은예산·cleanup총시간·finalrunner/group부재를확인한다. nativecall이걸리면기존outer가최종경계이며수동신호/강제성공/skip_cleanup/예산확대금지.
2. raw2는같은t0/sync0, 합성6은물리결과아님. 초기Isaacwarmup8별도기록,표시8프레임시계전후보존,새DEME0. 1600x900장면에글자영역외부추가/공통높이패딩으로PNG/MP4치수가달라지는것은원계획에포함된다.
3. 실제시작UTC/argv/PID·PGID와최종rc/시간/영수증·manifest/mapping/8PNG/MP4해시를보고한다. 진단중간probe있어도그것만으로성공선언하지않는다. root는8원본PNG모두실제검수하고감사는최종영수증·입력/표시매핑을읽는다.
4. probe계측오류96,close timeout97,outertimeout124는비성공. childrc0/manifest.ok만으로준비PASS금지. 실제종료후새정확GO전추가실행0. 본실행은root시각인수와최종독립사전검수뒤별도기술GO가필요하다. 사용자9시간비용승인은유지.
5. 실물조회·구동/PID/토크/카메라·설치·학습·A/B/C·commit/push·옛증거변경·자동재시도금지.
