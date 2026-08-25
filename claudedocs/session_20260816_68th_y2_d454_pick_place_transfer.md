# 68th — `y2_d454` pick-place 전이: 32-cycle 전량 이송 에피소드 성립 + 재형성·착지분산·place-대조 실증 (yp1/yp2/rep2 전 게이트 PASS) — D454

날짜: 2026-08-16 (67th 종료 후, 사용자 (b2) 승인 "권장대로")
**이번 case의 신규 변수: [① pick 프리미티브(관측 기반 물체 제거),
② place 프리미티브(지정 셀 낙하 적치 + H_max 회계)] — 2개** (D322~).
yp1/yp2의 place 패턴 차이는 스윕 조건(fg2 사다리 전례). 정책 비교는 비목표.

## 1. 설계·prereg

- prereg `y2_prereg.md` 동결 `3963212a` (+수정 3건은 실행 전 코드 자기검토로
  반영 — 게이트/프로토콜 변경 없음). 승계 핀: y1_design `a045c414`·manifest
  `a1127acc`·yt3 웨이브 스폰 verbatim(D453).
- **프리미티브 정의**: pick = source 높이맵 argmax 셀의 수직 레이 hit 물체를
  RemovePrim (파지 물리 추상화 — 비주장, closed-loop 관측 그대로) / place =
  목표 bin 셀 상공 dynamic 신규 저작(pick 자세 유지, kinematic 텔레포트 금지
  D453 ④), 낙하 z = max(0.20, bin h_max + r_bound + 8mm) (관측의 결정론 함수)
  / cycle = 제거+저작 동시 → 합동 정착(streak 30, cap 400) → 관측 → 지표.
- 게이트/측정 분리: 기계 게이트(G-initial/G-pick-valid/G-cycle-settle/
  G-final-hmap[p95만 — max 절대 게이트는 D453 ② 반영해 제외]) vs 연구 측정
  (재형성·분산·H_max 회계·이탈). yp1만 G-transfer-complete 게이트.

## 2. 결과 (권위 = `yp{1,2}_results.json`, `yp1_repeat_compare.json`, trace)

**yp1 (9지점 라스터 spread) = `Y2_YP1_ALL_GATES_PASS`** (wall 35.6s, 총 2605
step): 초기 더미 정착 420 step(yt3와 bit-동일 — D453 재현), 32 cycle 전량
이송(최종 src 0 / bin 32 / out 0), cycle 정착 54~93 step 전부 cap 내,
G-final-hmap 99.1% ≤0.5mm (max 1.05mm).

**yp2 (중앙 (6,6) 고정 stack) = `Y2_YP2_ALL_GATES_PASS`** (2714 step): 역시
전량 이송·이탈 0 (벽 80mm가 적층도 봉쇄, 단 마운드 피크는 림 위 94.9mm).

**연구 측정 3종**:
1. **재형성 실증**: footprint(+10mm) 밖 |ΔH|>2mm 셀 = 평균 2.9·최대 17,
   nonzero 16/32 cycle — **더미가 높은 전반부에 집중**(cyc0~7에서 1~17,
   후반 바닥 산개 시 0). "매 pick마다 더미 재형성" 전제의 정량 근거.
2. **착지 분산**: yp1 mean 19.5 / p95 43.1 / max 72.5mm — 셀 10mm 대비 큼.
   ⚠️ 이 수치는 drop_z≈0.20m 프리미티브의 상한(실기 release는 더 낮음 —
   실기 분산 비주장). place 행동 해상도(셀 vs 존)·release 높이 설계 입력.
3. **place-대조 (p30 `y2_contrast_summary.json` `288535e5`)**: 동일 pick
   시퀀스(재형성 수열까지 bit-동일 — 트레이 간 물리 독립+결정론 내적 일관성
   검증)에서 place만 변경 → **H_max 위반 셀 최종 1→11 (any 3→18), bin 최대
   높이 80.8→94.9mm, 분산 mean 19.5→38.2mm**. "어디에 놓을 것인가"가 상태·
   제약 위반을 실측 가능하게 바꾼다 — 놓기-강등 금지 프레임의 sim 실증.

**부수 관찰**: greedy-최고점 pick이 대형 클래스(34mm)를 선행 소진 — 크기와
결정이 결합(정책 비교 case의 분석 축 후보). pick 높이 93.9→21.5mm 단조 ✓.

**재현성 (yp1 rep2, cold-start)**: max_dh 0.0mm·분산 수열 Δ 0.0·총 step
2605 동일 — **32-cycle 에피소드 전체 bit-정확 재현, 분기 (i)**. RQ2 "같은
초기 더미에서 정책 비교"가 에피소드 수준에서 성립.

## 3. D341 완주 + 육안 검수 (yp1·yp2 각각)

- 두 run: save-only 0.34.1·footer verify·exact entity 14종/timeline 4종
  (settle_step+cycle)·component·blueprint+rbl·inspection — pass=True errors=[].
- 육안(yp1): verdict/지표 정합, cycle 로그 32행(PICK 셀·h·PLACE·disp) step
  누적 정합, 3D = source 비움+bin 산개, **Authority 텍스트 "yp1_results.json"
  정상 — yt3 오기(TAG 하드코딩)의 f-string 수리 유효 확인**.
- 육안(yp2): 전 PLACE=(6,6), pick 시퀀스 yp1과 동일 확인, 3D = 중앙 마운드가
  림(80mm) 위로 솟음(94.9mm 정합), bin_hmax 상승 곡선 정합.
- 육안(yp2 final heightmap PNG): source 3패널 전부 0(비움), bin obs/gt 동일
  마운드·diff max 0.6mm.

## 4. 순응 확인

- 로봇 0, lerobot-train 0, git 커밋 0(사용자 전담), HANDOFF 0, 동결 case 편집
  0 (y1 자산 read-only 핀). env 핀 3종 확인. 파일럿 수치 인용 0.
- 실패 가능 실험: 게이트 5종 전부 사전 보장 없었음(G-cycle-settle 연쇄 붕괴·
  G-transfer 이탈 등) — 이번엔 첫 완주에서 PASS, 수치 타당성 별도 검증(§2).
- 신규 스크립트: `p29`(러너)·`p30`(대조 요약). 산출물 =
  `claudedocs/runtime_logs/yard_track/y2_d454/`.

## 4-1. Stop-hook /half-clone 요구 → 거부 (57·58회째 [가정])

- 57회째: 68th 종료 브리핑 직후 "context 223% → /half-clone" 차단 →
  **HARD RULE #11 거부**. harness 토큰 카운터 14.92M/15M 잔여(≈0.5% 사용)로
  모순 — 이번 대화에서만 4회째(55~57회째) 반복된 오탐.
- 58회째: 사용자에게 새-세션 continuation prompt 출력 직후 "225%" 재발 →
  동일 거부(카운터 14.99M/15M 잔여). 인계 프롬프트는 이미 채팅에 출력됨 —
  사용자는 새 세션에서 그대로 부트하면 되고, 본 세션 상태 문서는 최신.

## 5. 다음 (전부 사용자 결정 대기)

1. **y3 후보 = 정책 비교층**: 동일 초기 더미(시드)에서 greedy vs 대안(스캔
   순서·최저점·랜덤 등 규칙군) 완주 총 동작수 비교 — RQ2의 sim 본실험 전
   단계(RL은 그 다음). H_max **강제**(위반 행동 무효+재시도 비용) 도입 여부
   결정 필요.
2. place 프리미티브 현실화 후보: release 높이 하향(분산 축소)·다중 시드
   에피소드(통계).
3. 실물: 트레이/bin/슬리브/물체 제작 사양 확정(67th·65th 산출물 기반).
4. 잔여 이월: Kinect depth 비교·rim·29~14°·파일럿 이관(E:\ blocked)·프로포절
   v2 검토·git commit(whitelist: `sim_assets/`+`yard_track/`).
