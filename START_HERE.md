# START_HERE.md

Last updated: 2026-08-06 KST (22nd 세션). **G0b case `g0b_d420` 계속.**
이번 세션 = 재검증 회수 완료 → 라운드-2/3 수리·적대검증 → **③ prereg 발행** →
**T3 Isaac attempt1~4 완주** → **파지 실패 원인 = attempt3 충돌 자산의 조 목구멍
폐색 실증 (D424)**. T2/T2b 완전 종결 유지(D421/D422/D423).

## ⚡ 현재 진실 — T3 물리 시행 4회 완주, 자산 결함 판별 완료

- **검증 체인 종결**: wf_3cea04db(10/10, 수리 8/8 유효) → 라운드-2 수리(transit 밴드
  0.5mm+폴리시 / trust region 12° / approach TCP-step 게이트 20mm 스코핑 + MINOR 6) →
  사전비행 v2 PASS → wf_9b819983(9 agents: 7/7 유효·회귀 0·MINOR 3 기계수리) →
  **③ prereg 발행**(`g0b_d420/t3_prereg.md` + 부록 A/B/C, supersession 3건, p7-parity
  델타 11건).
- **Isaac attempt1~4** (전부 사전등록 tuple, D341 육안검수 4/4 완료):
  | a | 인자 | verdict | 기전 |
  |---|---|---|---|
  | 1 | 기본(margin 0.5mm) | APPROACH_FAIL | descend가 **TCP z=0.0544(top+4.4mm)에서 접촉 정지** |
  | 2 | margin 5.5mm, marker 0.035 | LIFT_FAIL(5/5) | close 88→24° **전 각도 무접촉**, 물체 부동 |
  | 3 | +descend_open 45° | LIFT_FAIL | 동일 무접촉 시그니처 |
  | 4 | 45°, margin −7.5mm | APPROACH_FAIL | **정지 z=0.054394 = a1과 동일 → 개방각 무관 바닥** |
- **결론(D424)**: 동결 attempt3 자산에서 top-down 상면 중심 파지는 **기하 불가능**
  — 축 근방 고정 구조물이 TCP−4.4mm 바닥을 만들고, 그 위에서 닫힘 면은 원통 미도달.
  **T1 실물 rim 핀치 성공과 정면 모순 = 조/목구멍 충돌 형상의 실물 불일치 실증**
  (attempt3 채택 근거는 파트 수 감사였지 간극-깊이 검증이 아니었음).
- p9 sha 계보: a1 `939a5bd0…f5fb` / a2 `1ef8a411…4a55` / a3·a4 `99c99c65…2412`(1,780줄).
- 부수: verdict FAIL인데 exit 0(Kit close 동작) — **판정 권위 = stdout verdict 라인 +
  results JSON**, exit code 참고 금지.

## Active Case — `g0b_d420` (범위 불변)

- 물체 = 원통 D29×H50 / 24.83g 기립 (HARD RULE #18). 파지 = 수직 상부 상면 중심(D419).
- 이번 case의 신규 변수: [② T3 물리 시행 — **시행 완료**, 자산 결함으로 blocked]
- 출력 폴더: `claudedocs/runtime_logs/grasp_track/g0b_d420/`
- 승인: "T2/T3 진행"(19th) 기수령. **재분해 금지 해제(D415 ③)는 미승인 — 사용자 결정 대기.**

## 다음 세션 순서

1. **조 충돌 형상 감사 (읽기 전용 — 재분해 아님, D415 ③ 저촉 없음)**: isaaclab env
   pxr로 attempt3 physics 레이어 link5/gripper_link 64+64 파트 정점 덤프 → q5 45°/24°
   FK 변환 → (a) 축 근방(r<14.5mm) 최저점 = +4.4mm 바닥 파트 특정 (b) 조 면 간극-깊이
   프로파일. 산출 = `g0b_d420/`.
2. **사용자 에스컬레이션**: 자산 수리 없이는 T3 GRASP_PASS 불가 실증 → (a) 재분해
   금지 해제(D415 ③) 여부 (b) 대안(문제 파트만 비활성 / TCP-타깃 의미 재유도 등).
3. 자산 해결 후 attempt5 재사전등록(prereg 부록 D) → 이후 T4 실물 재현 대조.

## T 사다리 현황

| 단계 | 상태 |
|---|---|
| T0/T1 | 완료 (D419/D420) |
| T2/T2b | 완전 종결 (D421/D422/D423) |
| T3 sim 물리 파지 | **시행 4회 완주 — 자산 목구멍 폐색으로 blocked (D424), 사용자 결정 대기** |
| T4 실물 재현 / T5 hold→move→place / T6 격자 / T7 RL | 대기 |

## Open Risks / Claim Limits

- 서보 폐지력·자율 재현성 = null. `g0a_pass=false` 불변. **T3 4회는 물리 probe
  verdict — 실물 파지력 주장 아님.** "T1이 파지력 증명" 표현 금지 유지.
- 마찰 0.40/0.30 = 미실측 사전등록 가정(감도 leg는 자산 해결 후에만 의미).
- marker(D-2)는 거리 휴리스틱임이 실증됨(a2에서 무접촉인데 발화) — 접촉 증거로
  인용 금지, 물리 증거는 LIFT follow.
- 이중 세션 겹침 관측(22nd §0) — 세션 시작 전 이전 터미널 종료 확인 권장.
- 코드 충돌 1건 미해소: `deploy_smolvla.py:685-689` vs `safety_p0_guards.py:145-146` (T4 전).

## Frozen — Do Not Retry or Overwrite

- 격리 트랙 전체(QUARANTINE_point_enumeration_track.md) — 사용자 호출 시에만.
- p7 원본 재실행 금지 / attempt3 재분해 금지(D415 ③ — **해제는 사용자만**) /
  T2·T2b·**t3_grasp{,2,3,4}_*** 산출물 덮어쓰기 금지(tag-abort 가드 실증됨).
- HANDOFF.md, TASKS.md, `/half-clone`(거부 9회), commit/push 금지.
- `isaaclab` env pin(rerun 0.34.1/numpy 1.26.0/psutil 5.9.8).

## Must Read First

1. `AGENTS.md` → 2. this file → 3. DECISIONS **D424**(최신)→D423-R1→D423→D422→D421
4. `claudedocs/session_20260806_22nd_g0b_t3_reverify_repairs_prereg_isaac_attempts1to4.md`
5. `g0b_d420/t3_prereg.md`(본문+부록 A/B/C — 시행 4회의 tuple·유도·결과 요약 전부)
6. 시행 원본: t3_grasp{,2,3,4}_stdout.log · *_results.json · *_inspection.png

## Git

- HEAD == `fe2de19` — 20th~22nd분 미커밋(상태 문서, p8/p9, t2b/t3 산출물 전부).
- commit/push는 사용자 요청 시에만.
