# 65th — `g0f_d452` G-step: 조 슬리브 설계(p23)+완전닫힘 probe(gs1 0/13)+폭-정지 창 probe(gs2 stop29 8/8) — 힘-성공 기울기 부호 반전 (D452)

날짜: 2026-08-16 (64th 같은 작업일, loop iteration 3)
**이번 case의 신규 변수: [조 접촉 기하(슬리브) 1개 — gs2는 슬리브 × 폭-정지
(fg2 기검증)의 기합성, 신규 변수 0]** (변수 사다리 D322~)

## 1. 설계·저작 (p23, 물리 0)

- 입력 2 소스 SHA 핀: `fg1_gripper_only.usd`(`0e9f…dd76`) +
  `fg2_results.json`(`3b59…4a36`) stop21 rows.
- **게이트 4종 전부 PASS** (`gs1_design.json` `9178bc65…`):
  - G-kin: q5 조인트 기구학 모델(X(θ)=T(lp0)·R(lr0)·Rz(+θ))이 fg2 stop21
    8행 moving-jaw pose 재현 — t_err 0.0, R_err ≤7e-7.
  - G-contact: 순정 D29 접촉각 **22.762°** ∈ fg2 실증 괄호 (21.07, 23.0) ✓,
    간극 기울기 ~1.0 mm/°. **순정 이동 조 쐐기각 실측 23.53°** >
    atan(μ_s=0.4)=21.8° — D445 "악력 무관 배출"의 기하 원인 수치 확정.
  - G-design: t=3.5 mm/조, V-요람 α=10°, 설계 폭 28 mm → θ_design=36.55°,
    sleeved D29 접촉 예측 37.47°.
  - G-interference: 완전 닫힘 14°에서 슬리브-슬리브 2.06 mm > 0.
  - gate_burial (REV-1): footprint 스트립 분석 — 순정 면이 패드 안쪽면을
    뚫지 않음(p_max<−0.3 mm), back 깊이 매몰 필요량에서 도출.
- **REV-1** (반응적, 물리 0): 이동 순정면 23.5° 기울기 때문에 간극-모델
  대수 밸리 배치가 sanity 실패(1.03 mm) → "자기 릿지 + t" 직접 앵커 + 매몰
  분석으로 수리. 1차 p23 실행이 **exit 0 침묵 사망 = D447 결함 재확인**
  (실패 캡처 부재) → 캡처 추가 후 진짜 오류 노출.
- **REV-2** (probe 전, 물리 0): sdg2 검산 — 8/8 pose가 순정 고정면을 물체에
  **0.00 mm 플러시** 배치 → 고정 패드 3.5 mm는 스폰 침투 −3.50 mm 확정.
  수리 = 13 pose 전부 link5 원점 −(R·x̂)·t_f 평행이동(standoff-보존 사상).
  p24/p25 preflight가 보존 편차 ≤1e-17 검증.
- 산출물: `gs1_gripper_sleeved.usd`(`27fd0665…`, 조당 convex 패드 2 = 분해
  충돌, D446) + `gs1_sleeve_link5.stl`/`gs1_sleeve_gripper.stl`(mm, 프린트
  후보 — 장착부 미포함, 접촉 관여면만 sim과 동일) + `gs1_design.json`.
  질량 추정 15.4 g 총(조당 7.7 g, PLA — 63rd 예산 5~10 g/조 내).

## 2. gs1 — 완전 닫힘 probe (p24, p17 verbatim + REV-2 shift)

- preflight: 핀 12/12, quat 교차 2.2e-16, rim 게이트 6.7e-16. 실행 1회
  rc=0, wall 32.0 s. 자산 게이트: census (66,1)/조(64 순정+2 슬리브)·q5
  15속성 bit-일치·순정 mesh SHA 전량 일치+초과분=슬리브 4 prim 정확.
- **결과 0/13 `GS1_ALL_13_FAIL_SLEEVE_INSUFFICIENT`**, 측정 유효 13/13,
  rerun pass=True errors=[]. 전 행 `BILATERAL_NO_HOLD`(side 5.0~8.0 N,
  rim 4.2~21.8 N — **rim 첫 양측 접촉**, fg1은 rim 접촉 0 step).
- 진단(권위 results/trace): 접촉 step 16(fg2와 동일 — 드라이브 초반 슬루),
  목표 14° = 접촉(~37.5°) 대비 **23.5° over-close** → 힘 상승(5~9 N,
  12 step) 후 **world xy 30.6~132.9 mm 측방 압출**, 조는 빈 채 14.00° 완전
  닫힘. 결론: 강체 쌍에 과도한 over-close 명령은 패드 기하와 무관하게 최약
  방향 압출 — fg2 stop21 이 유지된 것은 접촉+1.8°에서 평형(1.8 N) 도달
  때문. → prereg 분기 (ii), SS3b로 gs2 선언.

## 3. gs2 — 폭-정지 창 probe (p25, fg2 구조 verbatim + 슬리브)

- side 8 × stop {39,37,36,35,33,31,29}° = 56 평가, rc=0, wall 149.1 s,
  측정 유효 56/56, rerun pass=True errors=[].
- **결과 27/56 `GS2_SLEEVE_WIDTH_STOP_WINDOW_MEASURED`** (`gs2_results.json`
  `ebdfdb93…`, 172,047 B):

| stop | 양측 peak [N] | 성공 | 판독 |
|---|---|---|---|
| 39° | 0 | 0/8 | 접촉 전 — sleeved 접촉각 ∈ (37, 39), 예측 37.47° 확증 |
| 37° | 0.35~0.49 | 1/8 | 파지력 부족 슬립 |
| 36° | 1.1~1.4 | 3/8 | 한계 영역 |
| 35° | 1.8~2.4 | 1/8 | 한계 영역(노이즈 밴드) |
| 33° | 3.3~4.6 | 7/8 | 견고 |
| 31° | 3.8~5.0 | 7/8 | 견고(1건 7.35 mm>6 게이트) |
| **29°** | **3.5~6.3** | **8/8** | **전승** |

- **핵심 발견 — 힘-성공 기울기 부호 반전**: 순정(fg2) = 조일수록 배출
  (1.8 N 유지 → 3.3~3.9 N 반반 → 4.6 N+ 전패). 슬리브(gs2) = **조일수록
  유지**(0.4 N 슬립 → 5 N+ 전승). 접촉면 기하 수리가 "과악력=위험"을
  "과악력=안전"으로 바꿈 — 실기에서 **전류-제한 stall 닫힘**(정밀 각도
  불요)이 가능해지는 성질. D451의 "고정각 ±2° 창" 문제의 해소 경로.
- 주의(정직): 사다리 내 전승 rung은 29° 하나(f full-window 지표로는 0°),
  "창이 넓다"가 아니라 **"깊게 조이는 쪽이 전부 안전"**이 정확한 서술.
  29° 미만~14°(gs1) 사이 어딘가에서 압출로 전환 — 미측정 구간.

## 4. D341 완주 + 육안 검수 (gs1·gs2 각각)

- 두 run 모두: save-only RRD 0.34.1·footer verify·exact entity/timeline/
  component·blueprint+rbl·inspection.png — pass=True errors=[].
- 육안(gs1): verdict 0/13 정확, WARN 행 전량 일치, rim 단일 조 힘 스파이크
  ~120 N 관찰(양측 min 4~22 N — 오버행 잼), hang 전 행 자유낙하 평탄.
- 육안(gs2): 사다리 계단형 힘 증가(0.4→6.3 N), stop36 SUCCESS/WARN 교차
  로그 정합, hang 낙하 한계영역 혼합→저각 0 수렴.
- 한계(기지, fg1/ba1과 동일): 3D 패널 결정시점 프레이밍 미흡 + 우상단
  토스트 가림 — 판정에 미사용.

## 5. 순응 확인

- 로봇 0, lerobot-train 0, git 커밋 0, 동결 case 편집 0 (fg1 USD read-only
  참조), HANDOFF 0. prereg 동결 후 REV는 전부 append + 실행 전 선언.
- 실패 가능 실험: gs1(실제 실패 발생)·gs2 실행.

## 5-1. Stop-hook /half-clone 요구 → 거부 (53회째 [가정])

- 65th 종료 턴에서 stop-hook "Context usage 202% → /half-clone" 차단 발생 →
  **HARD RULE #11 거부**. harness 토큰 카운터 15M 잔여로 hook 판독 모순
  (52회째와 동일 오탐 패턴). 사용자가 직접 commit `2b067e8`
  ("최근작업(8월16일)-Posco") push — 58th~65th 증거 전량 포함, tree clean.

## 6. 다음 (loop 계속)

1. **④ O-step**: 비정형 convex 다면체 생성기 + 프린트 파일 ~50개. 유효
   개구 입력 = 본 case 확정치(슬리브 접촉각 사다리·힘-내성). 
2. 잔여(사용자 결정 대기): 슬리브 실물 프린트(장착부 설계 추가) / 29°~14°
   미측정 구간 / rim 미해결(0/5) / 파일럿 이관 / git commit.
