# 69th — `y3_d455` 정책 비교층 v1: 동일 초기 더미에서 규칙 정책 8종 완주 에피소드 (8/8 전 게이트 PASS + a1 rep2 bit-재현) — D455

날짜: 2026-08-16 (68th 종료 후). 사용자 지시 verbatim: "(b4) y3 진행해.
H_max 강제/해상도/release 높이는 너가 권장안 제시하고 시작해."
**이번 case의 신규 변수: [① place 행동 계약 v2(존 3×3 + H_max 관측-마스크 +
release 벽-클리어런스 하한 — 사전 결정 3건의 묶음, 전 arm 공통 상수),
② 정책군 스윕(pick 5종 / place 3종)] — 2개** (D322~).

## 0. 부팅 무결성 (인계 체크)

- 해시 검증: START_HERE `908aa63b` ✓ / 67th doc `c23a881b` ✓ / LEDGER
  `3d1aaf1c` ✓ / y2 prereg `3963212a` ✓ / y1_design `a045c414` ✓ /
  manifest `a1127acc` ✓ / contrast `288535e5` ✓.
- **불일치 1건 보고**: 68th doc 기대 `944734f8` vs 실제 `46ab406c` — 원인은
  68th doc §4-1 자체 기록(continuation prompt 출력 **후** 58회째 stop-hook
  거부를 append) → 인계 해시가 최종 수정 전 값. 내용상 무결성 침해 아님.
- y1_prereg 현재 `6679aa76` ≠ 동결 `8d8ec12f`도 문서화된 REV append 이력
  (67th doc §2)과 정합.

## 1. 사전 결정 3건 (사용자 위임 — 권장안 채택, prereg `37c11131` §1)

1. **H_max 강제 = 관측-기반 사전 마스크 + 사후 위반 회계**: 존 유효 ⟺
   `voronoi max + class_mm ≤ 80mm+0.5`. 전 존 무효 → argmin pred 폴백+카운트.
   근거: 전 정책 공정 필터 / RL 표준(action masking) / 실조업은 한계 회피.
2. **place 해상도 = 3×3 존(40mm)**: yp1 분산 p95 43.1mm ≫ 셀 10mm — 셀 지정은
   허위 정밀도(D454 ③). 존 격자 = yp1 9지점과 동일 좌표(연속성). 관측은 10mm 유지.
3. **release = 벽-클리어런스 하한**: `drop_z = max(80mm, bin_max) + r + 8mm`
   (y2의 0.20 floor만 교체). 근거: 0.20m은 조건부 상한 인공값(D454 ②), 실기
   release는 벽+후행 체적 클리어런스(D449) 아래 불가. **a8 브리지**로 y2
   release를 동일-정책 대조 격리(침묵 교체 방지).

## 2. 설계·실행 (prereg 동결 `37c11131`, 실행 전 선언)

- 승계: yt3 웨이브 스폰 verbatim(SEED 45300)·pick=argmax류 셀 레이 hit
  RemovePrim·합동 정착(streak 30/cap 400)·13×13@10mm 레이캐스트 관측.
- arms: a1 greedy_high/masked_raster/v2(주축·rep2) · a2 scan_gated ·
  a3 greedy_low · a4 random_gated(rng 45301) · a5 blind_raster(무관측,
  no-op=동작 1 소모·물리 0) · a6 masked_argmin · a7 stack_masked ·
  a8 = a1+y2 release. 종료 = src obs 전 셀<5mm 또는 cap 400.
- 게이트: G-initial/G-action-contract/G-cycle-settle/G-complete/G-final-hmap
  (+ a1 rep2 G-repro). 예측 4건 사전 기재(§4).
- 신규 스크립트: `sim_scripts/p31_y3_policy_compare_probe.py`(러너),
  `p32_y3_policy_summary.py`(교차 요약). 실행 전 자기검토 수정 2건(np.float64
  직렬화 캐스트·script 사본 전 arm) — 게이트/프로토콜 변경 없음.

## 3. 결과 (권위 = `a{1..8}_results.json`; 요약 파생 = `y3_policy_summary.json` `822d4e4a`)

**verdict: 8/8 arm 전 게이트 PASS + a1 rep2 `i_repeatable`** (max_dh 0.0mm·
분산 수열 Δ0.0·총 step 2302 동일·총 동작 32 동일 — cold-start bit-재현).
wall 29~32s/arm. 초기 정착 420 step = yt3/yp bit-동일 (D453 재현 5회째).

| arm | 정책 (pick/place/rel) | 동작 | noop | 분산 p95 | reshape Σ | 위반F/A | bin hmax | mexh |
|---|---|---|---|---|---|---|---|---|
| a1 | greedy_high/raster/v2 | 32 | 0 | 21.7 | 92 | 5/5 | 91.6 | 11 |
| a2 | scan_gated/raster/v2 | 32 | 0 | 28.7 | 190 | 11/12 | 83.9 | 10 |
| a3 | greedy_low/raster/v2 | 32 | 0 | 23.5 | **346** | **19/19** | 94.3 | 7 |
| a4 | random_gated/raster/v2 | 32 | 0 | 20.8 | 238 | 10/10 | 87.5 | 9 |
| a5 | **blind**/raster/v2 | **242** | **210** | 28.7 | 190 | 11/12 | 83.9 | 10 |
| a6 | greedy_high/**argmin**/v2 | 32 | 0 | 21.8 | 92 | **0**/2 | 80.4 | 14 |
| a7 | greedy_high/**stack**/v2 | 32 | 0 | 42.8 | 92 | 4/4 | 83.0 | 12 |
| a8 | greedy_high/raster/**y2** | 32 | 0 | 33.9 | 92 | **0**/1 | **78.1** | 11 |

**주요 결과 5건**:
1. **관측의 가치 (예측1 확증)**: blind 242동작(= 32 pick + 210 no-op)
   vs 관측-게이트 전부 정확 32 — **7.56×**. 내부 통제: a5의 물리 부분수열이
   a2와 완전 동일(분산·reshape·위반 전 지표 일치 — blind ≡ scan+낭비 동작)
   ⇒ 차이는 순수하게 관측 유무.
2. **release 브리지 (예측3 확증, pick 수열 bit-동일 통제 성립)**: a8(0.20m)
   → a1(v2): 분산 mean 16.6→11.8·p95 33.9→21.7·max 64.3→24.4mm. **그러나
   방향이 반대인 동시 효과**: 위반 0→5·bin hmax 78.1→91.6mm — 높은 낙하가
   산개로 더미를 평탄/치밀하게 만들어 H_max 준수엔 유리. **release 높이 =
   정밀도 vs 다짐의 트레이드오프** (해석 가설: 낙하 에너지 재배열; 실측은
   수치만 주장).
3. **pick 순서 → 재형성·위반 결합 (예측4 탐색 → 강한 신규 발견)**:
   reshape Σ = greedy_high 92 ≪ scan 190 < random 238 < **greedy_low 346**
   (3.8×). 낮은 곳 파기는 이웃 붕괴 유발. 또한 pick 순서가 클래스 소진
   순서를 바꿔 **place측 위반까지 결정**(greedy_low는 대형 34mm를 만재
   후반에 배치 → 위반 19): pick·place 결정이 결합돼 있다는 3-결합 프레임의
   sim 실측 근거.
4. **마스크 효과 (예측2 반증 — 정직 기록)**: a7 최종 위반 4 > 예측 ≤2.
   단 비마스크 yp2(11/18, hmax 94.9) 대비 대폭 감소(4/4, 83.0). yp2는 y2
   release라 순수 단독 대조는 아님(방향성 증거로만 기재). 위반 잔존 원인 =
   추정자 소진(7~14 사이클, pred_err mean 9~17mm 과대) 후 argmin 폴백.
5. **place 정책 효과**: 동일 pick·release에서 argmin(a6)이 raster(a1)의
   위반 5→0 (hmax 80.4) — 규칙군 내에서도 놓기 선택이 제약 준수를 바꾼다
   (D454 place-대조의 정책층 확장).

**정직 한계**: 관측-게이트 정책 간 완주 동작수는 전부 32로 동일 — 파지
추상화(항상 성공)에선 동작수 분화 채널이 관측 유무뿐. 규칙 vs RL의 동작수
차이 실험(RQ2 본실험)은 파지 실패/재시도 모델 또는 과제 난이도 구조가 필요.

## 4. 예측 대조 (사전 기재 → 판정)

1. gated=32·blind>32 → **TRUE**. 2. a7 위반 ≤2 → **FALSE (4)**.
3. a8 분산 > a1 → **TRUE**. 4. 탐색(방향 무예측) → 92/190/238/346 기록.

## 5. D341 완주 + 육안 검수

- 8 arm 전부: save-only RRD 0.34.1·footer verify·exact entity 16종/timeline
  4종/component·blueprint+rbl·inspection — `rerun_validation` pass=True
  errors=[] (rep2는 prereg대로 results/compare만).
- 육안 3건 기록: **a1** — verdict 패널 수치=results 정합, source 비움/bin
  만재 시각 확인, 후반 masked=9X 로그, **Authority "a1_results.json" 정상**
  (yt3 하드코딩 오기 재발 없음). **a5** — NOOP 행 인터리브(h=0.0 blind
  miss)·PICK 행 라스터 진행 정합. **a8** — pick 수열 로그가 a1과 동일
  셀·h(브리지 통제 시각 확인), bin이 a1보다 평탄(산개). 요약 PNG 6패널 =
  json 수치 정합.
- 잔여물 검사: partial/failure 파일 0, 산출물 105파일.

## 6. 순응 확인

- 로봇 0, lerobot-train 0, git 커밋 0(사용자 전담), HANDOFF 0, 동결 case
  편집 0(y1/y2 자산 read-only 핀·SHA 재검증 통과). env 핀 3종 확인(D326).
- 실패 가능 실험: 게이트 5종+repro 사전 보장 없음 + 예측 4건 중 1건 실제
  반증(예측2) — 규칙 충족.
- 파일럿 수치 인용 0. "RL이 이긴다" 주장 0(규칙군 비교뿐).

## 6-1. Stop-hook /half-clone 요구 → 거부 (59·60회째 [가정])

- 59회째: 69th 종료 브리핑 직후 "context 124% → /half-clone" 차단 →
  **HARD RULE #11 거부**. harness 토큰 카운터 14.75M/15M 잔여(≈1.7% 사용)로
  모순 — 55~58회째(67th·68th doc 기록)와 동일 패턴의 오탐.
- 60회째: 사용자 요청으로 새-세션 continuation prompt를 채팅에 출력한 직후
  "127%" 재발 → 동일 거부(카운터 14.99M/15M 잔여). 인계 프롬프트는 이미
  채팅에 출력됨(최종 해시 fc230863 반영본) — 사용자는 새 세션에서 그대로
  부트하면 되고, 본 세션 상태 문서는 최신. 추가 조치 없음.
- 61회째(08-17): 폴더 정리 조사-전용 브리핑(수정 0, 사용자 결정 5건 대기)
  직후 "136%" 재발 → 동일 거부(카운터 14.99M/15M 잔여 ≈0.1% 사용).
  55~60회째와 동일 오탐 패턴 — hook 스크립트(check-context.sh)가 harness
  카운터와 무관한 값을 읽는 것으로 보임(수리는 사용자 승인 사안).
- 62회째(08-17): T1/T2 이관 배치(bf6ge7ag2) 진행 중간 보고 직후 "149%"
  재발 → 동일 거부(카운터 14.98M/15M 잔여). 이관 배치는 백그라운드 계속
  실행 중 — /half-clone 실행 시 오히려 감시 컨텍스트가 끊기므로 부당.

## 7. 다음 (전부 사용자 결정 대기)

1. **y4 후보 = 동작수 분화 채널 도입**: 파지 실패 확률 모델(높이/기울기
   의존) 또는 잔량-임계 완주 조건 — 규칙 vs 학습 동작수 비교(RQ2 본실험)의
   전제. 마스크 추정자 개선(클래스→관측 기반 증분)도 후보.
2. 다중 시드 에피소드(통계) / Kinect depth 비교(RQ3 사전, b3 이월).
3. 실물 제작(슬리브+물체 52+트레이/bin) / 파일럿 E:\ 전달 / 프로포절 v2 검토
   — **본 세션 수치 중 프로포절 인용 후보**: 관측 7.56×·pick-place 결합
   (greedy_low 위반 19)·place 정책 위반 5→0.
4. git 커밋(whitelist: `sim_assets/` + `yard_track/` + p26~p32).
