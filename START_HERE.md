# START_HERE.md

Last updated: 2026-09-11 — 새 scoop 1회 후 잔류 배출 기울임·HOME 복귀 완료. 최신 세션 `claudedocs/session_20260911_full_scoop_outlet_repeat.md`. D484 유지.

**현재 HOME 목표 [0,0,90,0,0,0]에 복귀했다.** `scoop_tilt_cycle_01/execution_05/result.json`: `completed=true`, `home_reached=true`, 최종 실제 q=[1.406250,2.988281,91.318359,1.054687,-0.175781,2.724609]°. 문 목표0·마지막 토크명령200·포트닫힘. 이전 “열린 롤90/raise2 자세” 기록은 현재가 아니다.

## Active Case — single source of truth

- Active: `scoop_v0`의 실물 S1 경로 연결. 이번 신규 변수 [기존790 scoop에 잔류 배출 기울임 및 HOME 복귀 연결]. W10은 완료되어 동결, 추가 sim/PID/토크/형상 변경 없음.
- 사용자 “처음부터 다시”에 따라 초기HOME→새plunge1회→790닫힘/리프트→컵운반/개방→출구기울임→문닫기→최종HOME. 중간 추종정지4건과 복구를 거쳐 완료했으며 무중단 재현 성공은 아니다.
- 출력 정본: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/scoop_tilt_cycle_01/`.
- 실기 `execution_01~04` 원래경로 추종정지 보존. `execution_05`는 마지막 잔류배출·HOME 구간 완료. `combined_01`은 중간정지까지, **`combined_02`가 최종 전체 기록**. 구간들을 여러 새scoop로 세지 않는다.
- 사용자 정정 “배출하고 HOME까지 가야 한다”가 열린상태유지 종료보다 우선한다. 향후 승인된 전체cycle도 HOME→scoop→배출→잔류기울임→HOME으로 정의한다.

## 이번 검증 결과

- 마지막 구간83.538273s·4215피드백, 추가추종정지0. 모든 settle의최대팔오차3.463352°≤5°. 실제 출구면 경사7.878947→22.529027°, 당시 공구 기울기15.029297°(관절FK). nominal wrist20°를실제공구20°로표기하지않는다.
- 손목롤90 정렬 및기울임 동안 어깨/팔꿈치목표고정. 기존raise1통과목표로먼저돌아간후손목만기울임→3초대기→역기울임→롤0→문닫기→직립P1→base0→HOME. 컵은사용자수동추종;실제FK립변위약67mm.
- 전체11029피드백·구간별기록시간합217.040311s. 직렬포트닫힘공백4개는미관측으로보존. 첫HOME명령1+마지막HOME명령1,새plunge1. 원래경로4정지와총시간을숨기지않는다.
- 새scoop close4.218750→lift3.955078°·리프트중최대추가개방0°. 명시문목표유지. PID쓰기0, 원시에는tG/전류/온도없음.
- **이번무게/잔류/영점여부는미입력**. 이전사용자“알다떨어졌어”는`outlet_tilt_01/operator_observation_02.json`의직전회차관찰0알이며이번회차에복사금지. 이전gross22.28g/이번컵약0.05g도새측정으로쓰지말것.
- Rerun/CSV/명령감사 최종본은실행05와combined02의analysis·validation·inspection 참조. 처음4실기·계획실패·복구사본모두보존.

## Next concrete action / 경계

1. 이번배출후컵포함무게·잔류알수·컵영점/비움여부를사용자로부터받아새관찰파일로기록. 이미질문했으며로봇은HOME이다.
2. 마지막고정어깨손목기울임·HOME종료가확인되었다. 다음반복은초기HOME에서완성된전체경로를계획해야하며이번중간정지실행을5회조건고정수집완료로세지않는다. 반복상승편차원인을새PID값으로즉석덮지말것.
3. **Git commit/push는이번턴실행0. 사용자가push를직접한다.** 현재master HEAD `d6688af7722fe6b12b7ac546f73611c7a96cba7c`;로컬변경/데이터인계만한다.

## Current verified truth — 이번 W10

- 정본 폴더: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/`.
- `cell_DE_dt2e6_c`: E5e6 유지·dt2e-6, 13:13:04→13:44:34 KST, rc0/1890s/스톨0. **포획541개 / 10.9592g**.
- 첫 닫힘 q3.307°·저항1.783022N·m, 재닫기 q3.069°·저항1.765356N·m, 둘 다 **servo_stall**. 문 하한 없음, 3N 보호 정지 아님.
- 최종 **명목 관절각3.069° ≠ 서보 환산5.569°**. 실제 메시 관절각2.993°(서보5.493°). 립 등가 간격6.152mm, 최종 립 물림0개.
- W8 옵션F 중앙은154개/3.1196g·문 하한 정지. 두 조건의 차이 전체를 dt 효과로 단정 금지(dt 단일 변수의 직전 대조는 `cell_DE_c`).
- 현재 코드 구 회귀 `regression_sphere_resume_20260911`: **287개**, 허용268~362, 두 닫힘servo_stall. 과거 회귀 source와 달라 새로 실행했다.
- **과학 G0~G3 4/4 + D341 PASS**. Rerun0.34.1: 툴/접촉/스칼라2774sync·입자64frame RRD 읽기 대조, RRD/RBL/검증 JSON/스크린샷/실제 육안 검수 완료.
- dt-only가 발산하지 않아 조건부 E1e8 강성 셀은 미실행. 물리 코드·params·더미·S1 형상·사전 등록 조건 불변.
- 보고서 `.../w10_deme_close_fix/REPORT_w10.md`, 교차검증 `.../resume_20260911/verification.json`, 재생 `.../cell_DE_dt2e6_c/scoop_s1_seed460_w10.rrd`.

## 유지하는 맥락 / 먼저 읽을 파일

- `AGENTS.md`→`claudedocs/DECISIONS_ACTIVE.md`/`LEDGER_RECENT.md`→최신세션과`.../scoop_tilt_cycle_01/REPORT.md`. 상태숫자는실행별raw/analysis까지확인한다.
- 서보참조 `docs/reference/{hardware.md,servo_pid_st3215.md}`. S1직결형,gripper0~30°,손목피치±90°보호. 맨T106/torqueOFF/새EPROM/PID 금지범위유지.
- 초기실물PID/900/790·기존W9/W10원본은이번턴미수정. 기존어깨P16/I0복귀는명령기록이지레지스터직접확인이아니다.
- 새회차원시/동결소스/계획/중간STOP기록덮어쓰기금지. 첫4실패는최종성공으로바꾸지않는다. 단일실험전반은중단을거친완료다.
- `HANDOFF.md`/`TASKS.md`,이전열린배출자세인계,재부팅전W10GPU차단은현재로신뢰금지. 상태원장은배타소유이며다음도구는relay를먼저읽는다.
