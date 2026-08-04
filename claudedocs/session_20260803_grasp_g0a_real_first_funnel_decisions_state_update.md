# Session 2026-08-03 — Real-First 열거 Funnel 방향 결정 + 상태 기록 (실험 0)

이번 case의 신규 변수: [] (결정·기록 세션 — 실험 case 아님)

## 0. Session progress rule 정당화 (실험 미실행 사유)

이 세션은 실패 가능한 실험을 실행하지 않았다. 사유: D408 승인 범위는 소진·동결됐고
(START_HERE.md 2026-07-29판 "Next Concrete Action — Stop"), 승인된 runtime이나
새 과학 case가 존재하지 않는 상태에서 사용자가 요청한 것은 (1) 2026-07-31
인계 내용의 이해 브리핑과 (2) 방향 결정 4건의 확정 및 상태 파일 기록이다.
결정을 바꿀 수 없는 validation은 실행하지 않았다(AGENTS.md Session progress rule).
다음 실험은 D409로 별도 설계·승인 후 실행한다.

## 1. 배경 — 2026-07-31 인계 (이 문서가 최초 파일화본)

2026-07-31 세션은 사용자 지시로 파일 생성/수정 0으로 진행됐고, 인계는 채팅
프롬프트로만 존재했다. 핵심 결과를 여기에 파일화한다:

- 유저 커밋 `40ec3ac`("grap에 대한 새로운 방안")가 D402~D408 산출물 152파일을
  체크포인트했고 HEAD == origin/master == 40ec3ac, clean.
- "새 방안" = **zero-step A64 양측 접촉영역 탐색 → 자세 동결 → 실물 보유력
  캠페인 → PhysX close-lift-hold-move** funnel. 전수 검증: repo 인용 17/17
  MATCH, 설치 스키마 2/2, PhysX 5.6.1 ↔ omni.physx 107.3 버전 정합.
- 실행 가능성 근거: D371 offline hppfcl 전례(378 질의, physics step 0),
  A64 형상 = `g0a_d339` witness JSON(64+64 convex), 순수 FK가 D349 라이브
  자세를 0.0013mm로 재현, inner mask = D350/D354/D368(4/17/16 parts),
  barrel/cap 분류기 = D351, antipodal 지표 = D354 pinch_facing_geometry.
- Blocker 2건: ① 정적 존재성은 D362형 밀어넘김(moving-jaw contact step 31/32
  → object motion 41/42 → fixed link5 contact 45/46, DECISIONS.md D362)을 못
  막음 → q5 닫힘 arc 기구학 sweep(step 0 유지)으로 "fixed-jaw 먼저" 순서
  제약 필수. ② D330 실행 오차 평균 TCP 36.033mm > 후보 간 margin →
  margin>추종오차 조건 또는 "영역 대표" 강등 필요.
- 문헌 조사(41건: 38 CONFIRMED/3 CORRECTED/0 REFUTED): 이 방안 =
  known-object sampling-based grasp synthesis 계보(Nguyen 1988 접촉 영역,
  Ferrari-Canny 1992, Dex-Net, GraspGen 등). 표준 funnel = 기하 열거 →
  도달성 필터 → 물리 sim 라벨 → 학습 → 소량 실물 보정. 기하 라벨 단독
  학습 금지(Kappler 2015/Rubert), 샘플러 편향 경고(Eppner) → 격자 전수.
  공식 도구(Isaac Grasping SDG/Grasp Editor/GraspGen)는 flying-gripper 가정.

## 2. 사용자 결정 4건 (2026-08-03 확정)

① **sim-first → real-first 역전 (확정)**. 실물 원통(명목 D29×H50, 사용자
   실측 24.83g)이 모든 계산·라벨의 권위다. sim 사양을 동결해 실물을
   주문제작하는 경로(BACKLOG `sim_first_cylinder_material_contract`,
   2026-07-14)는 기각·SUPERSEDED. 사용자 근거: "나중에 실물 물체를 주고
   잡아보라는 다양성이 생겼을 때 대응하려면 지금 실물 물체를 잡는 법을
   알아야 한다." D379 지속 규칙("실제 제품 규격과 역사적 D362 규격 분리",
   "actual product D29×H50 geometry rebase 필요")과 정합. G0b 대상 사양도
   D34×H90에서 실물 실측 사양으로 rebase — 다음 교수님 보고 시 명시.
② **라벨 사다리 3단 채택**: 기하(zero-step) → 물리 sim(PhysX) → 실물 소량
   캠페인. 단서 조항: **2단 PhysX는 절대 합격/불합격 라벨이 아니라 상대
   스크린**(명백히 나쁜 후보 제거)으로만 쓴다. 근거: 24.83g 전도 임계
   ~0.09–0.14N ≈ D362 접촉등록 문턱 0.1N, 마찰 미실측(sim 1.5/1.2는
   임시값). 기하 라벨 단독으로 학습 승격 금지.
③ **순서 = C안**: zero-step 열거 → **로봇 없는 실물 보정 파일럿**(캘리퍼
   치수 실측, 질량 provenance 기록, 실물 테이블 기울임 시험/손+저울 전도힘
   확인) → PhysX 스크린 → 실물 본 캠페인. 실물이 권위이면서 귀한 실물
   로봇 시도를 스크린 통과 후보에만 쓰는 절충.
④ **자체 harness 채택**: 공식 Isaac Grasping SDG/Grasp Editor는
   flying-gripper 가정이라 핵심인 5-DOF 도달성 필터를 대체 못 하고, 설치
   Kit 107.3에서의 확장 존재/버전 정합 미확인 + D326 env pin 리스크.
   평가 항목은 BACKLOG로 이동, G2(형태 다양화) 진입 시 재검토.

## 3. 이번 세션 검증 결과 (편집 전 원본 대조)

- **24.83g는 repo 어디에도 기록돼 있지 않았다** (claudedocs/*.md,
  START_HERE.md, BACKLOG.md 전수 grep 0건). 2026-07-31 인계 프롬프트에만
  존재 → 본 문서 §4로 최초 기록.
- D29×H50은 DECISIONS.md D379(22003-22004, 22036행 부근)에 "실제 제품
  명목(nominal)"으로만 존재. 실측 아님 → P0 캘리퍼 실측 필요.
- **D409 번호 미사용 확인** (repo 전수 grep — dataset manifest 해시 문자열
  우연 일치만 존재). **D399는 예약 유지** (D398-F1, BACKLOG.md 2026-07-27
  후속 교정 항목) — 사용 금지.
- HEAD == origin/master == `40ec3ac`, 편집 전 working tree clean.
- 전도 임계 재계산: 무게 0.02483kg×9.81=0.2436N, 반지름 14.5mm →
  F_tip = 0.2436×0.0145/h: h=25mm→0.141N, h=40mm→0.088N, h=50mm→0.071N.
  인계값 ~0.09–0.14N은 파지 적용 높이 h≈25–40mm 구간과 일치(h=50mm는
  0.071N으로 범위 밖). D362 등록 문턱 0.1N과 같은 자릿수 재확인.
- MEMORY.md Recent Sessions 5건 → 1건 회전 필요(HARD RULE #8) 확인.

## 4. 실물 원통 사양 provenance (최초 기록)

- 명목 치수: D29×H50 mm (제품 명목 — DECISIONS.md D379에 최초 등장,
  commit b880bc8 계보). 실측 아님.
- 질량: **24.83g — 사용자 실측** (2026-07-31 세션에서 사용자 보고;
  HARD RULE #18에 따라 권위. 저울 모델/측정 일시는 미기록 → P0 캘리퍼
  실측 시 재측정하여 기기·일시와 함께 재기록 예정).
- 마찰/재질: 미실측. sim의 0.72kg·마찰 1.5/1.2는 D34×H90 시대 코드
  임시값이며 실물과 무관(BACKLOG sim-first 항목 원문 참조).

## 5. 단기 Plan (short-term goals)

P0 (로봇·Isaac 불필요, 승인 부담 최소 — "preliminary test"에 해당):
1. 실물 원통 캘리퍼 실측(D, H — 사용자 손 측정) + 질량 재측정·provenance.
2. (C안 파일럿) 실물 테이블 기울임 시험/손+저울 전도힘 확인 — 기록만,
   sim 도입은 별도 case 변수.
3. 배치 지그 검토(실물 캠페인 재현성) — 설계만.

D409 예정 scope (설계 착수는 사용자 "설계 착수" 명시 지시 대기):
- 가칭: 실물 원통 zero-step 양측 접촉영역 전수 열거
  (real-first labeled candidate map, rung 1).
- 신규 변수 1: 실물 원통 기하(캘리퍼 실측 D×H). 질량 24.83g는 분석
  상수(전도 임계)로만 사용, 마찰 미도입 — Variable Ladder 준수.
- 내용: 순수 FK reachable family 격자 전수 + A64(g0a_d339 witness) 기준
  양측(고정 조/가동 조) 접촉 영역 계산 + q5 닫힘 arc sweep "fixed-jaw
  먼저" 순서 제약 + margin vs 36.033mm 실행 오차 → 영역 대표 채점.
- 판정 범위 선언: 접촉 영역 지도 + 순서 제약 채점까지. 파지 성공/force
  closure/stable grasp 주장 없음. g0a_pass 변경 없음.
- 산출물: `claudedocs/runtime_logs/grasp_track/g0a_d409/` + D341 Rerun 계약.
- 이후: D409 결과 → PhysX 스크린 case(별도 번호·승인) → 실물 캠페인
  case(별도 번호·승인, 로봇 HW 명시 승인 필요).

## 6. 이번 세션 파일 변경 (전부 상태 기록, 코드 0)

1. 본 문서 신규.
2. `direction_20260708_grasp_pivot.md` — "실물 우선 열거 funnel 채택
   (2026-08-03)" 섹션 append.
3. `BACKLOG.md` — sim-first 항목 SUPERSEDED 표기(원문 보존) + 신규 2건
   append(공식 SDG 평가 이연, closed-loop 배포 장기 항목).
4. `START_HERE.md` — overwrite(방향 결정 + Next Action + Git 갱신).
5. auto-memory `MEMORY.md` — Recent Sessions prepend + 1건 회전(archive).
6. `EXPERIMENT_LEDGER.md` — 변경 없음(실험 run 없음, ledger는 run 전용).
7. `AGENTS.md`/`DECISIONS.md` — 변경 없음(상태≠규칙; DECISIONS는 D409
   완료 시 append). commit/push 없음.

## 6.5 검증 결과 (3-agent 적대적 검증 회수 완료)

run `wf_c2a1c870-60e` (교차 정합성/프로토콜 준수/사실 추적성) 완료.
- **blocker 1건 → 수정 완료**: BACKLOG closed-loop 항목의 "사용자 결정 ⑤"
  표기가 타 파일 "결정 4건"과 모순 → "장기 기록 항목(결정 아님)"으로 정정.
- warning 6건 중 2건 수정: 전도 임계 "일치" 표현에 h≈25–40mm 한정 추가;
  §2 ①에 G0b 사양 rebase 명시 추가.
- warning 잔여 4건 (다음 세션/다음 overwrite 시 처리, 비긴급):
  ① MEMORY.md Pre-Work Checklist grasp 행이 구 pivot("D400+ SDF chain")
  표기 — START_HERE 경유 부팅이라 실해 없음. ② 다음 START_HERE overwrite
  시 "Active Case: 없음(D409 설계 승인 대기)" 명시 섹션 복원.
  ③ MEMORY archive 설명 stale(회전 누적). ④ 인계 귀속 수치 4건(FK
  0.0013mm, 문헌 41건, 17/17 MATCH, inner mask 4/17/16)은 repo 독립 검증
  불가 — **D409 설계 시 repo-검증 사실로 인용 금지, 재도출 필수**
  (특히 FK 0.0013mm는 D409 전제라 재검증 의무).
- ok: D362/D371/D379/D330 인용 전부 원문 일치, D407/D408 수치·SHA 무모순,
  git 변경 4파일 == 선언 목록, 규칙 파일(AGENTS/DECISIONS/LEDGER) 미변경,
  BACKLOG 원문 보존(삭제 0라인), direction doc append-only, MEMORY 회전
  verbatim, g0a_d339 witness 64+64 실측 일치, 40ec3ac=152파일 실측 일치.
- journal:
  `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/c939784c-2b28-47e0-994f-6f906166203c/subagents/workflows/wf_c2a1c870-60e/journal.jsonl`

## 6.6 독립 재검증 + 정정 2건 + 동시 세션 편집 사건 (후속 세션 기록)

- 후속 세션이 부팅 시 journal이 `started` 3행(375B)뿐임을 확인하고 §6.5
  재실행 규정에 따라 미완 2렌즈(교차 정합성/사실 추적성)를 원 프롬프트
  verbatim으로 재발사했다(run `wf_4145df3b-7f1`). 프로토콜 준수 렌즈는 원
  agent 전사에서 StructuredOutput을 직접 회수했다(blocker 0, warning 2).
- **동시 세션 편집 사건**: 재검증 진행 중 원 워크플로 `wf_c2a1c870-60e`가
  15:55 완료됐고, 구 세션이 재기동해 §6.5 갱신(15:56:47)과 BACKLOG
  blocker 수정(15:56:18)을 수행했다. 재검증 양 렌즈가 검증 도중 파일 변경을
  목격·기록했다. 구 세션이 살아 있는 동안 상태 파일 동시 편집 위험 있음 —
  구 세션 종료 권고(사용자 브리핑에 전달).
- 재검증 결과(수정 후 상태 기준): **blocker 0**. 원 렌즈 핵심 확인 전부
  독립 재확인 — 24.83g 질량 기록 전무(HEAD `git grep` 실측), D409 미사용
  (`git grep "D409" 40ec3ac` 0건), D399/D398-F1 예약, 전도 임계 산수 전
  항목 재계산 일치, D330:18490/D362:20761-20762/D371:21433/D379:22003-22036
  원문 인용 일치, g0a_d339 witness 64+64 실측, 40ec3ac=152파일,
  D407/D408 수치·SHA 무모순, MEMORY 회전 verbatim.
- **표현 정정 2건** (§3 원문은 보존, 여기서 정정 — Research Verification
  Rules의 정정 기록 방식):
  1. §3 "24.83g는 repo 어디에도 기록돼 있지 않았다" → 정확 범위:
     **질량 기록으로서** claudedocs/*.md·START_HERE.md·BACKLOG.md 0건.
     HEAD의 `.claude/` 5개 파일(agent-memory/agents)에 문자열 `24.83`이
     존재하나 v6 action.std 관절 표준편차 벡터 성분(deg)으로 질량과 무관.
  2. §3 "dataset manifest 해시 문자열 우연 일치만 존재" → 실측 정정:
     우연 일치 `d409`는 dataset manifest가 아니라 **git 미추적 runtime log
     JSON/JSONL**(g0a_d362 durable_step_prefix, g0a_d367 prereg, g0a_d397
     candidate_geometry)의 hex 해시 부분문자열이다. 핵심 주장(D409가 결정
     번호로 미사용)은 불변.
- 재검증 journal: 이 프로젝트 `.claude/projects/.../d2632654-*/subagents/`
  `workflows/wf_4145df3b-7f1/journal.jsonl`.

## 7. 다음 승인 경계

D409 설계 착수는 사용자 "설계 착수" 지시로 개시한다. 이후 표준 3단계:
설계·정적 준비(음성 대조 포함) → tuple SHA 발급 → 그 SHA 인용 명시 승인
→ attempt1 retry 0 실행. 과학 상태는 이 세션으로 변하지 않았다:
D407 FAIL-STOP, D408 관측성 수리 PASS, `g0a_pass=false` 유지.
