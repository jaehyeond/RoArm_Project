# y1_d453 prereg — 야드 테스트베드 v1: 더미 정착(yt1) + 높이맵 관측(yt2)

작성: 2026-08-16 67th. 실행 전 작성·동결 (이후 수정은 REV append + 실행 전 선언만).
트랙: `yard_track` 신규 (grasp track과 별개, forward-only 신규 폴더).

## SS0 — 목적과 신규 변수

**이번 case의 신규 변수: [① 다물체 더미 스폰·정착 물리(o1 암석 32개),
② 높이맵 관측 파이프라인(레이캐스트 그리드)] — 정확히 2개** (변수 사다리 D322~).

- 테스트베드 프레임 = 조언 반영: 목적지 = **유계 bin + 높이 상한 H_max** —
  place 선택이 다음 상태에 실제로 영향을 주는 구조 (놓기-강등 금지, 프로포절 §5).
- bin/트레이 기하 자체는 저작물 — p26 설계 게이트로 검증(물리 주장 아님).
- **비목표(이 case에서 하지 않음)**: bin 투하·적치 물리(다음 case), 판단 정책
  4종, 로봇 팔 스폰, 센서 노이즈/Kinect 모델, 실물 프린트, RL.

## SS1 — 설계 (권위 = `y1_design.json` sha16 `a045c414bfeb381e`, p26 게이트 6/6 PASS)

| 항목 | 값 | 근거 |
|---|---|---|
| heightmap 셀 | 10 mm | 최소 파지 폭 22 mm의 절반 이하 |
| reach annulus | r ∈ [0.150, 0.325] m, 여유 5 mm | D440/t3w top-down 영역 |
| source 트레이 | 내부 130×130 mm (13×13셀), 중심 (0.2130, +0.0993), 벽 t5 h80 | 선창 축소판, az +25° |
| bin | 내부 130×130 mm (13×13셀), 중심 (0.2130, −0.0993), 벽 t5 h80 | az −25°, **H_max = 80 mm** |
| 트레이 간격 | 58.6 mm (외벽 기준) | ≥20 mm 통로 게이트 |
| 표준 에피소드 | **N_ep=32** (클래스당 8, index 0..7) | manifest `a1127acc` subset |
| tightness ρ | 1.34 (φ=0.55) / 1.22 (φ=0.50) / 1.46 (φ=0.60) | ρ(φ하한)≥1.10 완주 안전 + ≤1.60 압박 유지 |
| subset 상한 | 질량 15.66 g ≤ 30 / 파지 폭 34 ≤ 35 mm / extent 49.6 mm | 63rd 예산 |

**선언 가정(실측 아님)**: φ=0.55(감도 병기) / 질량 = manifest 15% infill 추정
(o1 규약 — 실측 전 sim 질량 주장 금지) / annulus = sim 유래(실기 주장 금지) /
마찰 0.40/0.30·반발 0.1 = grasp track 연속성용 선언값(현실성 비주장).
**기지 근사**: reach 게이트는 셀 중심 기준 — 트레이 코너 극단은 annulus를
최대 ~5 mm 벗어남. 파지 목표는 셀 중심이므로 설계상 허용, 실기 반경 검증 전
코너 물체 도달성 주장 금지.

## SS2 — yt1: 더미 스폰·정착 probe (물리 O)

**질문**: o1 32개를 source 트레이에 낙하시키면 (a) 정착 수렴? (b) 트레이 내
유지? (c) 침투 없음? (d) 동일 시드 재실행 재현성 수준?

**프로토콜**:
- 스폰: 3×3 격자(간격 55 mm > max extent 49.6 — 초기 겹침 원천 배제) × 4층
  (z = 0.10/0.16/0.22/0.28, dz 60 mm), 36 슬롯 중 32 채움. 슬롯 배정·자세
  (uniform random quat)·미사용 4슬롯 = 전부 seed 45300 RNG에서 유도(결정론 프로토콜).
- 물리: dt 1/60, 전 물체 동시 릴리즈, convex hull 충돌(정점 20~31 ≤ 64),
  solver 설정은 p22/p25 승계.
- 정착 판정: 전 물체 |v|<5 mm/s AND |ω|<0.1 rad/s **60 step 연속** → settled.
  T_max 1200 step(20 s).

**게이트**:
- G-settle: T_max 내 settled 도달.
- G-contain: settled 시 전 32개 COM이 트레이 내부 XY — 이탈 0개.
- G-penetration: 바닥 최저 정점 z ≥ −2 mm AND 물체간 최대 침투 ≤ 2 mm.
- 재현성(게이트 아님 — **분기**): 동일 시드 전체 2회 실행, 최종 높이맵 비교.
  분기 (i) max|Δh| ≤ 2 mm → 시드 = 통제된 초기상태로 사용 가능.
  분기 (ii) 초과 → 에피소드 초기상태는 정착 포즈 **기록**으로 고정(시드 재생성 금지).

**측정**: 최종 pose 32개, source 높이맵, pile 높이 avg/max(설계 추정 59.7 mm
avg와 대조 — 괴리 시 φ 가정 재평가), 이탈/침투 수치, wall clock.

**실패 가능성 실재**: G-settle(진동 미수렴)·G-contain(스폰 반동 이탈)·
G-penetration(hull 관통) 어느 것도 사전 보장 없음.

## SS3 — yt2: 높이맵 관측 probe (yt1 settled 상태 위, 같은 프로세스)

**질문**: PhysX 레이캐스트 그리드 높이맵 = 기하 ground truth?

- 관측: 두 영역 각 13×13 셀 중심에서 z=0.5 → −z closest-hit 레이캐스트.
- GT: numpy 레이-삼각형 교차(동일 셀 중심; manifest `vertices_m`/`faces` ×
  settled pose + 트레이 기하 + 바닥 z=0). 관측과 완전 독립 구현.
- G-hmap: |Δh| ≤ 0.5 mm @ ≥95% 셀 AND max ≤ 2 mm. 초과 시 cooked convex vs
  원 메쉬 차이 조사(REV로 원인 규명 — D446 계열 위험).
- 음성 대조: bin 영역(빈 상태) 전 셀 = 바닥 0 (±0.5 mm).

## SS4 — D341 Rerun 계약

기하+정착 궤적 verdict → RRD 필수. save-only 0.34.1(핀), 정착 스텝 타임라인
(물체 pose·max|v|·접촉 요약 스칼라), 최종 높이맵 이미지(관측 vs GT vs Δ),
verdict 텍스트, 고정 blueprint + `.rbl` export, footer `rrd verify`, headless
inspection.png, 육안 검수 기록(세션 doc). run1 = RRD 권위, 재현성 run2는 Δ
스칼라·높이맵만 기록(사전 선언).

## SS5 — 비주장

실기 이동/파지 성능, bin 적치 거동, 안식각의 현실 대응, 질량·마찰 현실성,
Kinect 관측 충실도, 트레이 코너 물체 실기 도달성. 파일럿(`E:\posco-pilot`)
수치 일절 인용 안 함.

## SS6 — 순응·env·실행 계획

- 로봇 0 / lerobot-train 0 / git 0(사용자 전담) / HANDOFF 0 / 동결 case 편집 0.
- env 핀: numpy 1.26.0 ✓ / psutil 5.9.8 ✓ (67th 확인) / rerun 0.34.1 (실행
  직전 preflight 재확인, D326).
- 러너: `sim_scripts/p27_y1_yt1_pile_settle_probe.py` (yt1+yt2 단일 프로세스,
  태그 `yt1_*`/`yt2_*` 분리 산출). preflight(스폰 좌표·핀·트레이 기하 검산)
  단독 선행 → 본 실행. 실패 캡처 필수(D447 — exit 0 침묵 사망 금지).
- 산출: `yt1_results.json`, `yt1_trace.npz`, `yt1_timeline.rrd/.rbl`,
  `yt1_rerun_validation.json`, `yt1_inspection.png`, `yt2_heightmap_*.png`,
  stdout/argv/exit_status. 재현성 run2 = `yt1_rep2_results.json` +
  `yt1_repeat_compare.json` (별도 프로세스 cold-start — 더 강한 재현성 판정).

## REV-2 (yt1 결과 반응, yt3 실행 전 선언 — 스폰 프로토콜 v2)

yt1(attempt3, 공식 결과) = **`Y1_FAIL_G_CONTAIN`**: 3/32 이탈(동쪽 벽 밖 1,
남쪽 통로 2), 나머지 게이트 4/5 PASS (yt2 높이맵 max|Δ| 0.572mm 포함).
기전(권위 yt1_results.json + attempt1/2 오탐 분석): 3×3 격자 가장자리 슬롯
(±55mm) + reach 27.2mm가 벽 밴드(65~70mm)를 오버행 → 1층 물체가 벽 상단에
8.4mm 조기 착지 후 바깥으로 이탈 + 동시 낙하 산란. 격자 기하의 구조적 문제
이므로 반응적 프로토콜 수리(REV)로 처리:

- **yt3 = 스폰 프로토콜 v2** (게이트·판정·물리 파라미터는 yt1과 동일):
  1. xy 격자 2×2 (±27.5mm) — 최대 xy 도달 54.7mm ≤ 벽 내면 65mm,
     직접 벽-상단 착지 경로 제거. 수평 최악 여유 0.6mm > 0 유지.
  2. 8웨이브 순차 릴리즈(웨이브당 4개, 웨이브 간 45 step), z=0.20 스폰.
     동시 낙하 운동에너지 1/8.
     **REV-2a (yt3 attempt1 반응, 재실행 전 선언)**: kinematic 파크+텔레포트
     방식은 물리에 translate가 반영되지 않음이 확인됨(min_dz=−797mm — dynamic
     전환이 선처리되고 write-back이 텔레포트를 덮어씀, 증거 yt3_attempt1_*).
     수리 = 웨이브마다 4개를 z=0.20에 **dynamic 신규 저작**(런타임 저작 즉시
     시뮬은 yt1 attempt1 낙하 프로파일+진단2로 기확인). 파크 격자 폐기.
  3. 배정·자세 = seed 45300 rng (perm(32) + quat 생성, 코드가 권위).
  4. 활성화 가드 = 웨이브별 released 4개 step+1 Δz ≥ 0.5mm.
  5. 정착 판정(streak)은 마지막 웨이브 릴리즈 후부터 계수, T_max 1200 유지.
- yt1 산출물은 공식 결과로 보존(이름 유지). yt3 = `yt3_*` 산출물,
  재현성 rep2는 yt3 프로토콜로 수행.
- attempt1/2(rc=3)는 가드 설계 오탐(벽-상단 8.4mm 착지 경로 미고려) —
  증거 `yt1_attempt{1,2}_*`, 진단 2종 = scratchpad(세션 doc 기록).

## REV-1 (실행 전, 물리 0 — 스폰 z 상향 + 근사 명시)

구현 검산에서 SS2 스폰 z {0.10/0.16/0.22/0.28}의 결함 발견:
- manifest 실측 max |vertex 좌표| = **27.18 mm** (max_extent/2=24.8이 아님 —
  메쉬가 완전 중심화되어 있지 않음, hull centroid offset ≤ 5.94 mm).
- z=0.10이면 1층 물체 바닥 = 0.0728 < 벽 상단 0.08, 격자 가장자리(±55 mm)
  물체의 xy 도달 82.2 mm > 벽 내면 65 mm ⇒ **스폰 시점 벽 침투 가능**.

수리 (실행 전 선언):
1. 스폰 층 z = **{0.115, 0.175, 0.235, 0.295}** (바닥 최저 0.0878 > 0.08,
   층간 최악 여유 60−2×27.18 = 5.6 mm, 수평 최악 여유 55−2×27.18 = 0.6 mm > 0).
2. G-contain의 COM은 **hull centroid(월드 정점 평균) proxy** 사용 — 질량
   중심 아님을 명기 (메쉬 중심 offset ≤ 5.94 mm 감안, 게이트는 내부 사각형
   포함 여부라 proxy로 충분).
3. 낙하 중 벽 상단 통과 시 xy 오버행은 허용(벽 z 대역 0~0.08 밖) — 정착
   이탈 여부는 G-contain이 판정.
