# START_HERE.md

Last updated: 2026-08-16 KST — 63rd: **연구 방향 전면 pivot (교수님 지시)**.
B601 구매 경로 기각, 로봇 = **RoArm-M3-Pro 확정**, 그리퍼는 커스텀 승인(3D프린트
조 슬리브/고무, 서보 무리 금지, **USD+실물 동시 수정**). 신규 주제 = **포스코
원료야드 축소판**: depth 높이맵을 보고 재형성되는 더미에서 "어디서 집고 어디에
놓을지"를 학습하는 **순차 결정층 연구** (RQ1 greedy 데모 / RQ2 규칙 vs 학습
완주 총 동작수 / RQ3 sim2real+교란). 63rd는 조사 전용(물리 0) — 도메인/갭 조사
+ 영상 4종 판독 완료.

## Active Case — single source of truth

- **Active: `g0f_d452` (G-step gs1+gs2 — 완료, D452) → 다음 = O-step (64th
  loop 승인 범위 내)**. 출력 경로 =
  `claudedocs/runtime_logs/grasp_track/g0f_d452/`. 슬리브(평행 패드+10°
  V-요람, 분해 충돌): gs1 완전닫힘 0/13(over-close 압출), **gs2 폭-정지
  stop29° 8/8 전승(3.5~6.3 N)** — **힘-성공 기울기 부호 반전**(순정 조일수록
  배출 vs 슬리브 조일수록 유지) ⇒ 실기 처방 = 전류-제한 stall 닫힘. STL
  프린트 후보 2종 확보. 상세: 65th doc.
- 완료: `g0e_d451` (W-step fg2, D451 — stop21 8/8, 순정 창 ~2°). 상세:
  64th doc §3~§5.
- 프로포절 v2 = `claudedocs/proposal_posco_yard_v2_20260816.md` (D450 반영,
  인용 9편 검증 완료 — 사용자 검토 대기).
- 프로포절 원안 + 조사 결과 영속 기록 =
  `claudedocs/session_20260816_63rd_posco_yard_pivot_domain_recon.md` (§0~§7).
- Frozen: grasp track 전체 — `g0d_d449`(ba2, D449) / `g0d_d448`(ba1, D448) /
  `g0c_d446`(bg1/bg1v, D446-D447) / `g0b_d444`(fg1, D445) / `g0b_d420`. 편집·
  재실행 금지. **B601 트랙(구매 품의·ba3·벤더 제보)은 교수님 기각으로 종료** —
  증거·교훈은 신규 트랙 전제 자산으로 이전.

## Current verified truth (63rd 조사 요점)

- **포스코DX가 2026년 현재 정확히 이 장비들을 무인화 중**: 리클레이머 + GTSU
  (그랩 20~25t, 하역 80% 무인화 목표, 2026-07 무인 시험운전), 개발·검증에
  **Isaac Sim Sim2Real 사용**. 공개된 AI 역할은 인지+제어 계층뿐 — "어디를
  퍼낼지" 결정층 학습 근거 없음. 업계(ABB/BHP)도 규칙 라이브러리.
- **시나리오 매핑 확정**: pick측 = GTSU 그랩 사이클(이산 pick-and-dump, RIST
  영상 t=00:53에 선창 높이맵 계측 실물 증거), place측 = 스태커 적치(놓기 선택이
  실재하는 곳 — GTSU 투하는 고정 호퍼). 물질 = 철광석 fines/원료탄(모래·철근
  아님), 더미 = stockpile.
- **⚠️ 갭 문구 수정 필수 (D450)**: Spinelli et al. 2025 (arXiv 2508.09003, ETH)
  가 heightmap→파지점 PPO + 총 사이클 최소화로 **pick측 future work를 채움**.
  "ETH 계획층이 비어 있다" 문장 사용 금지. novelty = **3-결합(재형성 더미 ×
  완주 총 동작수 × pick+place 양쪽 선택 학습)** 한정. 원료야드 도메인 결정층
  학습 사례는 미발견(MEDIUM-HIGH) — 한정어 필수.
- **그리퍼**: fg1/D445(순정 조 = 수렴 쐐기, 0/13)가 커스텀의 정량 근거. 권장 =
  3D 프린트 조 슬리브(평행 패드+오목 요람+TPU 인서트, 5~10g, 탈착식) + 폭-정지
  닫힘 정책(D445 ② 미시험 분기). 검증 = p17 13-pose 프로토콜 verbatim 재실행
  ("순정 0/13 vs 커스텀 X/13"). 주 파지 모드 = 근수직 top-down (t3w: r∈[0.150,
  0.325m], 완전 수직 불가·기울임 6~24°).
- **물체 권장**: 절차 생성 비정형 convex 다면체 3D 프린트, 22/26/30/34mm ×
  ~50개, PLA 무광 회색(**Kinect ToF에 흑색·광택 불가**), 개당 8~15g. sim↔real
  메쉬 동일(D446 교훈 원천 차단).

## Active pivot and reserve pivots

- Active pivot: **포스코 야드 결정층 연구 — 64th: 사용자 순차 진행 승인
  (/loop)**. 확정 순서 = ①프로포절 수정(**완료** —
  `claudedocs/proposal_posco_yard_v2_20260816.md`, 인용 9편 arXiv 검증:
  Backman=2103.01283·AGPNet=2112.10877 신규 확보) → ②W-step(폭-정지 닫힘
  sim, HW 0) → ③G-step(슬리브 CAD+USD+p17 재실행) → ④O-step(물체 생성기).
  파일럿 이관은 `E:\posco-pilot` 접근성 확인 후. git commit은 명시 지시
  대기(승인 범위 밖).
- Reserve: 교수님 보고 패키지 / 벤더(reBot) upstream 제보 2건(우선순위 하락) /
  RoArm 잔여(fg2 폭-정지·D≤20·rim 기움).

## Open risks / do-not-repeat

- **"ETH future work 비어 있음"·"최초" 단독 표현 금지** — Spinelli 2025 존재
  (D450). 갭 문구는 3-결합+한정어로만.
- 파일럿 수치(66 vs 80회 등)는 **repo 밖(`E:\posco-pilot`) 미검증** — 이관/재현
  전 인용 금지. 파일럿은 규칙 vs 규칙이지 "RL이 이긴다" 증거 아님(Baidu 선례).
- "높이 우선"은 실조업 목적 아님(배합·잔량이 실제 목적) — 대리 휴리스틱으로 명기.
- 놓기 선택을 스크립트로 강등 금지(Baidu 부분집합화 — 프로포절 금지 조항).
- 고무링 단독 = 대조군일 뿐(쐐기각>마찰각이면 배출 — 기하 수리가 1차).
- ba2 물리 PASS를 "place 성공"으로 승격 금지(D449 ④) 등 grasp track 규율 유지.
- `numpy==1.26.0`, `psutil==5.9.8`, rerun 0.34.1 핀 (D326).
- git: **58th~63rd 전부 미커밋** — 사용자 지시 시에만. ⚠️ whitelist 확장 필요:
  `g0c_d446`+`g0d_d448`+`g0d_d449`+p18~p21 (62nd 기재분 유지).

## Next concrete action / authorization boundary

**64th: 사용자 순차 진행 승인 (loop) — 진행 순서와 상태:**

1. ~~프로포절 문안 수정~~ **완료** — `claudedocs/proposal_posco_yard_v2_20260816.md`.
2. ~~W-step~~ **완료 (D451)** — `g0e_d451` fg2 12/40, stop21 8/8 전승, 접촉각
   (21.07°, 23°) 괄호, 조일수록 배출(쐐기 힘-의존성 정량화). 성공 창 ~2° ⇒
   실기 폭-정지는 접촉/전류 감지 기반 필요, 슬리브 기하 수리가 여전히 1차.
3. ~~G-step~~ **완료 (D452)** — `g0f_d452` p23 설계(게이트 5종 PASS, REV-1/
   REV-2) + gs1 완전닫힘 0/13 + gs2 폭-정지 stop29° 8/8. "순정 0/13 vs 커스텀
   8/8" 대조 확보(성립 조건 = 폭-정지 결합). 잔여: rim 0/5·프린트 장착부.
4. **O-step (다음)**: 물체 절차 생성기 + 프린트 파일 ~50개 (22/26/30/34mm,
   PLA 무광 회색, convex, sim↔real 메쉬 동일).
5. 파일럿 v1 이관/재현 — `E:\posco-pilot` 접근성 확인 필요 (Windows 드라이브).
6. git commit/push — **승인 범위 밖, 명시 지시 시에만**.

## Must read first

1. `AGENTS.md`
2. `claudedocs/session_20260816_63rd_posco_yard_pivot_domain_recon.md` (**§0·§3·§5~§7 먼저**)
3. `claudedocs/DECISIONS.md` tail — **D450**·D449~D445
4. `claudedocs/session_20260814_62nd_g0d_d449_ba2_full_arm_side_place_probe.md`
   (grasp track 최종 상태)
5. auto-memory `tech_gripper_grasp_anchors.md` (조 개구 실측 fit — 0~30°만 검증)

## Do not trust as current

- `HANDOFF.md`, `TASKS.md`.
- Earlier `START_HERE.md` snapshots (62nd판 포함 — "ba3 승인 대기" 프레임은
  교수님 B601 기각으로 소멸).
- "B601 구매 품의 진행 중"·"ba3가 다음 단계"라는 어떤 서술도.
- 프로포절 원안의 "ETH 계획층 공백" 문장 (D450으로 수정 대상).
- ba1 RRD 제목 "phi=135" 오기(실제 225).
