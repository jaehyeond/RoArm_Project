# START_HERE.md

Last updated: 2026-08-17 KST — 70th: **콜드 아카이브 T1/T2 이관 완료 (인프라,
연구 0)** — git 비추적 대형 14폴더 ≈176GB를 외장 `ROBOT_DEV/RoArm_cold_archive`로
sha256 전량 검증 이관, 원경로 심링크 보존(**외장 미마운트 시 끊김 —
`ARCHIVE_INDEX.md` 참조**). 내장 215G 여유. **T3(b200_backup 2종+openvla
45G 유일 사본)는 내장 유지 — 2사본화 사용자 결정 대기.** 연구 상태는 69th
그대로: **정책 비교층 v1 성립 (사용자 (b4) 위임
"권장안 제시하고 시작")** — `y3_d455` 규칙 정책 8종 완주 에피소드 8/8 전
게이트 PASS + a1 cold-start bit-재현(D455). 68th: `y2_d454` pick-place 전이
(D454). 67th: `y1_d453` 테스트베드 v1(D453). 배경(63rd): **연구 방향 전면
pivot (교수님 지시)** — B601 기각, 로봇 = **RoArm-M3-Pro 확정**, 그리퍼
커스텀 승인(3D프린트 슬리브, 서보 무리 금지, USD+실물 동시). 신규 주제 =
**포스코 원료야드 축소판**: depth 높이맵 더미에서 "어디서 집고 어디에
놓을지" 순차 결정층 (RQ1 greedy 데모 / RQ2 규칙 vs 학습 완주 총 동작수 /
RQ3 sim2real).

## Active Case — single source of truth

- **Active: `y3_d455` (69th — 정책 비교층 v1, 완료 D455)**. 출력 경로 =
  `claudedocs/runtime_logs/yard_track/y3_d455/`. prereg `37c11131`.
  사전 결정 3건(사용자 위임): H_max = **관측-마스크**(pred = voronoi max +
  class_mm, 전무효→argmin 폴백) / place 해상도 = **3×3 존 40mm**(yp1 좌표) /
  release = **벽-하한** `max(80mm,bin_max)+r+8mm` (+a8 y2-release 브리지).
  결과: **8/8 PASS + rep2 bit-재현(2302 step Δ=0.0)**. 핵심 4: ① 관측 제거
  = 32→242 동작 **7.56×**(a5≡a2 물리 동일 — 순수 관측 효과) ② release
  트레이드오프(분산 p95 33.9→21.7 vs 위반 0→5·hmax 78.1→91.6, pick bit-동일
  통제) ③ pick 순서 → 재형성 3.8×·place 위반 19 결정(크기-소진 결합 —
  3-결합 sim 근거) ④ place 단독 위반 5→0(argmin). 예측2 반증 정직 기록
  (a7 위반 4>2, 추정자 소진). **정직 한계: 파지 추상화에선 관측-게이트 정책
  간 동작수 전부 32 — RQ2 본실험은 분화 채널(파지 실패 모델 등) 선행 필요.**
- 완료: `y2_d454` (68th, D454 — 32-cycle 전량 이송 bit-재현·재형성/분산/
  place-대조 실증), `y1_d453` (67th, D453 — yt3 웨이브 스폰 규약·높이맵
  검증·cooked hull 관측 권위), `g0f_d452`(슬리브 stop29° 8/8)·`g0e_d451`
  (fg2 stop21° 8/8)·o1 암석 52개(66th).
- 프로포절 v2 = `claudedocs/proposal_posco_yard_v2_20260816.md` (사용자 검토
  대기). **인용 가능 신규 수치(D455)**: 관측 7.56× / pick-place 결합(위반
  19) / place 정책 위반 5→0 / release 트레이드오프.
- Frozen: grasp track 전체(`g0d_d449`~`g0b_d420`) — 편집·재실행 금지. B601
  트랙 종료(교수님 기각).

## Current verified truth (63rd 조사 요점 — 불변)

- 포스코DX가 리클레이머+GTSU를 Isaac Sim Sim2Real로 무인화 중(하역 80%
  목표). 공개 AI 역할은 인지+제어 계층 — "어디를 퍼낼지" 결정층 학습 근거
  미발견(MEDIUM-HIGH, 한정어 필수).
- **⚠️ 갭 문구 (D450)**: Spinelli 2025(2508.09003)가 pick측 future work
  선점 — "ETH 계획층 공백"·"최초" 금지. novelty = **3-결합(재형성 더미 ×
  완주 총 동작수 × pick+place 양쪽 선택 학습)** 한정.
- 그리퍼 실기 처방 = 슬리브 + **전류-제한 stall 닫힘**(D452, 힘-성공 기울기
  부호 반전). 물체 = o1 절차 생성 다면체 22/26/30/34mm×52(PLA 무광,
  질량 실측 전 sim 질량 주장 금지).

## Active pivot and reserve pivots

- Active pivot: **포스코 야드 결정층 연구 — yard_track 체인 y1→y2→y3 완료.**
  다음 단계는 전부 사용자 결정(아래). git commit은 명시 지시 대기.
- Reserve: 교수님 보고 패키지 / RoArm 잔여(rim 0/5·29~14° 미측정) / 벤더
  제보(우선순위 하락).

## Open risks / do-not-repeat

- "ETH future work 비어 있음"·"최초" 단독 표현 금지(D450). 갭 = 3-결합+한정어.
- 파일럿 수치(66 vs 80회 등)는 repo 밖 미검증 — 이관/재현 전 인용 금지.
  파일럿·y3 모두 규칙 vs 규칙 — **"RL이 이긴다" 증거 아님**.
- "높이 우선"은 대리 휴리스틱(실조업 목적 아님) 명기. 놓기 선택 스크립트
  강등 금지(D450 ④).
- y3 마스크 추정자(클래스 공칭)는 보수 편향(소진 7~14사이클) — 게이트
  아닌 회계였음을 인용 시 명시. release 높이는 단일 최적 아닌 트레이드오프
  축(D455 ③) — 실기 설계 시 양방향 고려.
- ba2 물리 PASS "place 성공" 승격 금지(D449 ④) 등 grasp track 규율 유지.
- `numpy==1.26.0`, `psutil==5.9.8`, rerun 0.34.1 핀 (D326).
- git: 58th~65th push `2b067e8` 완료. **미커밋 = 66th o1(`sim_assets/`) +
  67th~69th y1·y2·y3(`yard_track/`+p26~p32+상태 docs) + 70th 이관분
  (`ARCHIVE_INDEX.md`+`.gitignore`+70th doc)** — whitelist에
  `sim_assets/`+`claudedocs/runtime_logs/yard_track/` 추가 필요. 커밋은
  사용자 지시 시에만.
- 스토리지: 이관분(T1/T2 14폴더)은 외장 **사본 1개**(백업 아님) — 외장
  분실 = 데이터 손실 조건으로 사용자 승인됨. T3 45G는 유일 사본 내장 유지.
- yt3 RRD 요약 "yt1_results" 하드코딩 오기(판정 미사용) — y2/y3에서 수리
  유효 확인됨.

## Next concrete action / authorization boundary

**다음 후보 (전부 사용자 결정 — 단독 착수 금지)**:
(b5) **y4 = 동작수 분화 채널**(파지 실패 확률 모델[높이/기울기 의존] 또는
잔량-임계 완주 — 규칙 vs 학습 RQ2 본실험 전제; 마스크 추정자 개선·다중
시드 통계 포함 가능), (b3) Kinect depth 렌더 vs 레이캐스트 비교(RQ3 사전),
(a) 실물 제작(슬리브+물체 52+트레이/bin 내부 130·벽 80mm), (c) 파일럿
E:\ 전달 후 이관/재현, (d) rim·29~14° 잔여, (e) 프로포절 v2 검토/교수님
보고 — D454 place-대조 + **D455 관측 7.56×·결합·place 5→0** 인용 가능,
(f) git 커밋(whitelist 갱신 포함), (g) **T3 2사본화 방식**(45G 유일 사본
— 70th doc §5).

## Must read first

1. `AGENTS.md`
2. `claudedocs/session_20260816_69th_y3_d455_policy_compare.md` (현행 case)
3. `claudedocs/DECISIONS.md` tail — **D455**·D454·D453·D450~D452
4. `claudedocs/session_20260816_68th_y2_d454_pick_place_transfer.md`
5. `claudedocs/session_20260816_63rd_posco_yard_pivot_domain_recon.md` (§0·§3·§5~§7)
6. `claudedocs/session_20260817_70th_cold_archive_t1_t2_migration.md` +
   `ARCHIVE_INDEX.md` (스토리지 현황 — 구 대형 데이터 경로는 전부 심링크)

## Do not trust as current

- `HANDOFF.md`, `TASKS.md`.
- Earlier `START_HERE.md` snapshots (68th판 포함 — "(b4) y3 결정 필요"
  프레임은 본 세션에서 해소됨).
- "B601 구매 품의"·"ba3가 다음"이라는 어떤 서술도.
- 프로포절 원안의 "ETH 계획층 공백" 문장(D450 수정 대상).
- ba1 RRD "phi=135"·yt3 RRD "yt1_results" 오기(둘 다 판정 미사용).
