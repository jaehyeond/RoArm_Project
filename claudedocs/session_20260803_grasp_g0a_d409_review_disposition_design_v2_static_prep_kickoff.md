# Session 2026-08-03 (4th) — D409 리뷰 회수 + 전수 처분 + 설계 확정 v2 + 정적 준비 착수

이번 case의 신규 변수: [없음 — D409 설계 단계 지속, 신규 변수는 3rd 세션
선언분 1개(실물 원통 기하) 불변]

> 본 세션의 상세 기록은 **3rd doc의 지정 placeholder에 기입**했다 (boot
> prompt 지시): 회수 = 3rd doc **§3.1**, 처분/확정 = **§4**, 정적 준비 =
> **§5.1**. 본 파일은 세션 서사 요약 + 회수 절차 재현 정보만 담는다.
> 3rd doc = `session_20260803_grasp_g0a_d409_sweep_recovery_design_v1.md`

## 1. 승인 범위 / 규칙 준수

- 사용자 "설계 착수" 승인 지속 (설계+정적 준비+tuple까지; attempt1은
  tuple SHA 인용 별도 승인). 과학 상태 불변: D407 FAIL-STOP,
  `g0a_pass=false`.
- stop-hook(124%)의 /half-clone 요구 **거부** (HARD RULE #11) →
  end-of-session update로 전환. HANDOFF 미생성 (HARD RULE #7).

## 2. 리뷰 회수 (3rd doc §3.1)

- journal 1/4 (OPS만) → 전사 StructuredOutput 추출 (OPS 1건 bit-동일
  교차, 나머지 3렌즈 호출 부재 확정) → resume 재발사.
- **발견: workflow resume은 세션 경계 너머로 result 캐시를 이월하지
  않음** (도구 계약 "same-session only" 실증) — 4렌즈 전원 신규 재실행
  됐고 OPS는 2-pass가 되어 합집합 처분. 다음에 세션 넘어 회수할 때는
  journal/전사 직접 판독이 1차이고, resume 재발사는 "재실행 비용 감수"
  로 취급할 것.
- 4/4 회수 (SCIENCE b3/w4 · FROZEN b2/w3 · LESSONS b2/w5 · OPS 재실행
  b2/w4 + OPS 원본 b1/w5). verbatim 보존 8파일:
  `g0a_d409/design_inputs/review_4lens_wf_c46dc45e-62d/` (manifest sha
  `b77fc8e34a14cbf50ddc5310149945069b6ec70ca004977176aa64d0e4c1bbcd`).

## 3. 처분·확정 (3rd doc §4) — 핵심만

- P1 A64 권위 d339→d348 / P2 τ rebase {6,500..11,500}µm / P3 (A) 정당화
  문장 철회(+본 세션 실측: 동결 자세 link5 argmin part_029 ∈ 4-mask →
  inner-mask min = 4.2727mm) / P4 dual-run 계약. warning 16항(합집합
  21건 병합) 전수 처분. anchor 게이트 = 4채널 ANY-reject 0.0005mm.
- 독립 재검증 8건 수행 후 처분 (리뷰 무비판 수용 금지 이행) — 목록과
  증거는 3rd doc §3.1 말미.
- 설계 확정본 v2 = §2 + §4.3 δ 11항 (충돌 시 §4 우선). prereg는 §4
  반영본만 저작 가능.

## 4. 정적 준비 착수 (3rd doc §5.1)

- S1/S2/S3 산출물: `g0a_d409/design_inputs/d409_static_prep_s1s2s3.json`
  (sha `f2aaadd13e6822ceebd6a5d565010c0f45c201d08d35b830d582e5ac36dfd63d`,
  payload 2회 bit-exact `43a46e05…5b124`; 도구 sha `e7b541ef…8a0f90`).
  S1 d348 128-part D409-canonical hash pin / S2 마스크 결박 전항 PASS /
  S3 처리량 run1 8.71µs·run2 7.54µs → 4.5M ≈ 34~39s ≪ 7,200s.
- **harness 3파일 저작 위임 `wf_d6a61f26-880` — 미회수 종료.** 회수 절차
  (다음 세션 1순위):
  1. journal 판독: `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/f261c48f-2ee3-41ec-a1e3-7bf0a140c455/subagents/workflows/wf_d6a61f26-880/journal.jsonl`
     의 result 4건 (author:worker / author:manual_writer /
     author:controller / review:consistency).
  2. 파일 실재 확인: `sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_{worker,controller,manual_writer}.py`
     — agent가 파일을 직접 저작하므로 journal 불완전이어도 파일은 존재
     가능. 완전성은 파일 자체로 판정 (docstring 'SPEC AMBIGUITIES
     RESOLVED' 섹션 포함 여부, 구획 완결성).
  3. 부재/불완전 시 resume 재발사 (script:
     `.../f261c48f-*/workflows/scripts/d409-harness-authoring-wf_d6a61f26-880.js`,
     resumeFromRunId `wf_d6a61f26-880`) — 단 §2의 세션 경계 캐시 미적중
     전례상 전량 재실행 가능성 감수. **완성된 파일이 이미 있으면 재발사
     대신 직접 적대 리뷰로 진행** (중복 저작 방지).
  4. **산출물 가정 금지** — 적대 리뷰(§4 δ 11항 coverage, 동결 침범 0,
     d339 질의 0, 인터페이스 정합, scope guard 실재) + 수리 후에만 채택.
- 이후: 정적 fixture(음성 2계층 §2.11 확정판 + 등가성 fixture W-LES4) →
  attestation/tuple 작성 후 **정지**.

## 5. Session progress rule 충족

- 실패 가능 검증 다수 실행: 리뷰 주장 8건 독립 재검증(전부 리뷰 지지
  — 설계 v1의 결함 4건 확정), S1/S2/S3 실측(전항 PASS이나 각각 FAIL
  가능 게이트), 2회 bit-exact 결정성 검사 2건. 설계 v1은 이 세션에서
  4개 실질 결함(P1~P4)이 확정되어 v2로 정정됨 — 실패한 실험(설계 반증)
  이 실제로 발생·처분됨.

## 6. 세션 종료 상태

- 완료: 리뷰 회수(4/4) → 처분(blocker 4-이슈 + warning 16항) → 설계
  확정 v2 → S1/S2/S3 → harness 위임 발사 → 상태 문서 갱신.
- 미완: harness 회수·리뷰·수리 → 정적 fixture → attestation/tuple.
- 불변: D407 FAIL-STOP, `g0a_pass=false`, 과학 verdict 없음. D399 금지.
  동결 파일 수정 0. 로봇 HW·Isaac runtime·lerobot-train 미실행.
