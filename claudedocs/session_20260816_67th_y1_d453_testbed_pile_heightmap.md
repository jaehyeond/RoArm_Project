# 67th — `y1_d453` 야드 테스트베드 v1 개시: 설계(p26) + 더미 정착/높이맵 probe (yt1 FAIL→yt3 해소, rep bit-재현) — D453

날짜: 2026-08-16 (66th 종료 후, 사용자 (b) 승인 세션, /loop 자기-페이스)
**이번 case의 신규 변수: [① 다물체 더미 스폰·정착 물리(o1 32개), ② 높이맵
관측 파이프라인(레이캐스트 그리드)] — 2개** (변수 사다리 D322~)

## 0. 사용자 지시와 프레임

지시 verbatim: "(b) 테스트베드 case 개시해. 조언 프레임(bin 제약 place) 반영해서
설계부터. git은 내가 push한거였어. loop돌리면서 제대로 검증, 검토, 확인, 결과물에
대한 의심을 하면서 진행해. step-by-step으로 순차적으로 사고하면서 말이야."
→ 테스트베드 v1 = source 트레이(선창 축소판) + **유계 bin + H_max 상한**(외부
조언 프레임 — place 선택이 다음 상태에 영향, 놓기-강등 금지 조항과 정합).
신규 트랙 폴더 = `claudedocs/runtime_logs/yard_track/y1_d453/` (forward-only).

## 1. 설계 p26 (물리 0, 게이트 6/6 PASS — `y1_design.json` `a045c414`)

- 레이아웃: source/bin 각 내부 130×130mm(13×13셀@10mm), az ±25°·r_c=0.235m,
  전 셀 중심 r∈[0.158, 0.316] ⊂ 유효 annulus [0.155, 0.320] (D440/t3w).
  벽 t5·h80mm, 트레이 간격 58.6mm. **bin H_max=80mm, ρ=1.34** (φ=0.55).
- 표준 에피소드 N_ep=32(클래스당 8, index 0..7), 최대 질량 15.7g·파지 폭 34mm.
- **의심 단계에서 잡은 설계 결함 3건**: ① 소스 벽 140mm 과대 공식 → 80mm
  ② ρ(φ=0.50)=1.066 → **완주 불가 위험** → 기준 "ρ(φ하한)≥1.10" 강화(H_max
  70→80) ③ 스폰 격자 32.5mm < max extent 49.6mm 초기 겹침 → 55mm.
- 수기 검산 일치(최원 셀 0.31607·간격 0.0586), 레이아웃 PNG 육안 검수.
- 기지 근사: reach 게이트는 셀 중심 기준(코너 극단 ~5mm 초과 가능 — 실기
  주장 금지). manifest 실측 max|vertex 좌표|=27.18mm(≠extent/2, 중심 offset
  ≤5.94mm) → REV-1 스폰 z {0.115~0.295} 상향.

## 2. prereg 동결과 probe 연대기 (전 실패 증거 보존)

prereg `y1_prereg.md` 동결 `8d8ec12f` + REV-1/REV-2/REV-2a (전부 실행 전 선언).

| attempt | 결과 | 원인/교훈 |
|---|---|---|
| yt1 att.1 | rc=3 가드 (min_dz 0.0233@t10) | **오진 1**: "비동기 활성화"로 진단 → kinematic 선저작 수리 |
| yt1 att.2 | rc=3 동일 값 bit-재현 | **오진 2**: "step 미보장" 가설 → 진단1(박스)로 기각: `simulate(1,dt)`=정확 1 step·자유낙하 이론치 일치 |
| 진단2 (scratchpad) | 32개 p27 흐름 재현 = 전부 정상 낙하 | 차이는 벽 유무 → **진짜 원인 = 1층 가장자리 물체의 벽 상단(8.4mm) 조기 착지. 가드 임계 50mm가 설계 오류(오탐)** → 5mm로 수정 |
| **yt1 att.3 = 공식** | rc=0 완주, **`Y1_FAIL_G_CONTAIN`** | 3/32 이탈(동 1·남 통로 2) — 격자 일괄 낙하 + 벽 밴드 오버행(±55+27.2>65mm)의 구조적 문제. 나머지 게이트 4/5 PASS |
| yt3 att.1 | rc=3 (min_dz=−797mm) | **kinematic 텔레포트가 물리에 미반영**(dynamic 전환 선처리+write-back 덮어씀) → REV-2a: 웨이브마다 dynamic 신규 저작 |
| **yt3 att.2 = 공식** | rc=0, **`Y1_FAIL_G_HMAP`** (G-contain 해소) | 아래 §3 |

## 3. 공식 결과 (권위 = `yt{1,3}_results.json`, trace.npz)

**yt1** (3×3×4 격자 일괄 낙하): settle 154 step·침투 −0.11mm·**이탈 3/32**·
높이맵 max|Δ| 0.572mm(99.7%<0.5mm)·bin 음성대조 0.0. verdict FAIL_G_CONTAIN.

**yt3** (REV-2: 2×2 격자 ±27.5mm — 벽 밴드 미오버행 — × 8웨이브×4개, z=0.20,
45 step 간격): **32/32 유지**, settle 420 step(마지막 웨이브+60), 침투 −0.28mm,
pile h_avg 52.5·h_max 93.9mm (설계 φ-추정 59.7 대비 −12% — φ 가정 대체로 타당).
**G-hmap FAIL**: 0.5mm 초과 4/169셀, max 6.447mm@(10,5).

**G-hmap FAIL 기전 (p28 분석, `yt3_hmap_slope_analysis.json` `207320de`)**:
초과 4셀의 GT 승자 면 기울기 = 38.1/65.6/72.2/79.5° — cooked convex hull의
수평 표현차 ≤~1.2mm가 tan(기울기)로 수직 증폭(79.5° 셀만 2mm 게이트 위반,
인접 셀은 bit-일치). **obs(레이캐스트)의 표면 = 충돌이 보는 cooked hull과
동일(sim 내 자기일관)** — raw 메쉬 GT는 교차검증층. 게이트 FAIL 판정은
게이트-쇼핑 없이 보존; max 게이트가 경사 증폭 미고려였다는 게 교훈.
Kinect 실관측·프린트 표면 대응은 비주장(RQ3 sim2real 항목).

**재현성 (yt3 rep2, cold-start 별도 프로세스)**: max_dh=0.0mm·pose Δ=0.0mm·
settle step 동일 — **bit-정확 재현, 분기 (i)**: 시드 = 통제된 초기상태
(같은 초기 더미에서 정책 비교하는 RQ2 전제가 sim에서 성립).

## 4. D341 완주 + 육안 검수 (yt1·yt3 각각)

- 두 run: save-only RRD 0.34.1·footer verify·exact entity 12종/timeline 3종/
  component 계약·blueprint+rbl·inspection.png — `rerun_validation` pass=True
  errors=[].
- 육안(yt3): verdict/게이트 5행/WAVE 0~7 릴리즈 step(0,45,…,315)/SETTLED 420
  전부 수치 권위와 정합. 3D: 더미가 파란 트레이 내 유지+주황 bin 관측점 평탄.
  스칼라: 웨이브 충격 8회 스파이크 — 순간 침투 ~−10mm·각속도 ~90 rad/s는
  위치-수준 depenetration(속도 클립 10 rad/s 우회)의 과도 현상, 정착 창 소멸.
- 육안(yt1): 이탈 3개가 트레이 밖에 시각적으로 확인(수치와 정합), 정착 과도
  단일 클러스터 후 quiet.
- **오기 1건(기지)**: yt3 RRD 요약 텍스트 "Authority = yt1_results.json"는
  yt1 하드코딩(실제 권위 = yt3_*). ba1 "phi=135" 전례류 — 판정 미사용.
- 계측 한계(정직): 정착 창 min_separation=None(정지 접촉은 report 미발생) —
  G-penetration의 물체간 항은 충격기 데이터+바닥 정점 z로만 뒷받침.

## 5. 순응 확인

- 로봇 0, lerobot-train 0, git 커밋 0(사용자 전담), HANDOFF 0, 동결 case 편집 0.
- env 핀 numpy 1.26.0/psutil 5.9.8/rerun 0.34.1 확인(D326). o1 질량 = 15%
  infill 추정 선언(실측 전 질량 주장 금지 준수). 파일럿 수치 인용 0.
- 실패 가능 실험: yt1/yt3 모두 실제 게이트 FAIL 발생 — 규칙 충족.
- 신규 스크립트: `p26`(설계)·`p27`(probe 러너)·`p28`(기전 분석). 산출물 전부
  `claudedocs/runtime_logs/yard_track/y1_d453/`.

## 5-1. Stop-hook /half-clone 요구 → 거부 (55·56회째 [가정])

- 55회째: iteration 2 진입 시 "context 85% → /half-clone" 차단 → HARD RULE
  #11 거부. harness 카운터 14.9M/15M 잔여(≈0.7% 사용)로 모순 — 52~54회째와
  동일 오탐.
- 56회째: loop 종료 직후 "181% → /half-clone" 재발 → 동일 거부. harness
  카운터 14.75M/15M 잔여(≈1.7% 사용)로 모순. 상태 문서는 이미 최신(67th판)
  이므로 추가 조치 없음.

## 6. 다음 (전부 사용자 결정 대기)

1. **y2 후보**: pick-place 프리미티브 전이(트레이→bin 이동·낙하 적치 물리 —
   place 선택이 bin heightmap을 실제로 바꾸는 것 검증) + H_max 회계.
2. 관측 현실화 후보: Kinect 시점 depth 렌더 vs 레이캐스트 비교(경사 셀 갭의
   실관측 대응 — RQ3 사전 조사).
3. 실물: 트레이/bin 제작 사양(내부 130mm·벽 80mm) — 슬리브·물체 프린트와 함께.
4. 잔여 이월: rim 0/5·29~14° 미측정·파일럿 이관(E:\ blocked)·프로포절 v2 검토.
