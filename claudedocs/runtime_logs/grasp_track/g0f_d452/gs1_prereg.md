# gs1 prereg — g0f_d452: 그리퍼 조 슬리브 설계·저작 + p17 13-pose 재실행 (G-step)

날짜: 2026-08-16 (64th). p23(설계·저작)·p24(probe) 실행 전 작성·동결.

## SS0 — 승인 근거

64th 사용자 순차 진행 승인("후보들 중에 어떤것들 순서대로 해야할지 보고
loop돌려서 진행해")의 ③ G-step. 63rd doc §6(그리퍼 커스텀 방향: 3D 프린트 조
슬리브 = 평행 패드+오목 요람, 탈착식, USD+실물 동시) + 교수님 승인(서보 무리
금지) 이행. HW 변경 0, 로봇 0 — sim 검증 단계만.

## SS1 — 질문과 분기

> **접촉면 기하를 수리한 조 슬리브(평행 패드 + V-요람)를 순정 조에 씌우면,
> fg1이 0/13이었던 동일 13-pose 프로토콜에서 잡고 유지하는가?**

분기 (사전 선언):
- (i) **1+ SUCCESS** → "순정 0/13 vs 커스텀 X/13" 대조 성립, 슬리브 기하
  수리가 sim 필요조건 충족 → 프린트/실물 단계 근거 확보.
- (ii) **0/13** → 슬리브 설계 결함 또는 잔여 기하 병목 — taxonomy·접촉
  시계열로 원인 분리 후 설계 수정 (gs2). D427~D451 재판정 없음.
신규 변수 정확히 1개 = **조 접촉 기하(슬리브 추가)**. pose·닫힘 목표·게이트
·프로토콜은 p17 verbatim.

## SS2 — 설계 절차 (p23이 재현 가능하게 수치 도출; 자유 상수는 아래에 고정)

- 입력(전부 기존 증거): `g0b_d444/fg1_gripper_only.usd`(SHA 핀 `0e9f…dd76`) +
  `g0e_d451/fg2_results.json`(SHA 핀 `3b591352…4a36`) stop21 pose_snaps.
- 도출(64th 스크래치 선행 검증 완료, p23이 gate로 재확인):
  q5 조인트 = axis Z(조인트 프레임), l5 프레임 축 = +Y, anchor
  (0, 0.018821, 0.052035); 기구학 모델 X(θ)=T(lp0)·R(lr0)·Rz(+θ) —
  **G-kin gate**: fg2 stop21 8행 moving-jaw pose 재현 t_err<1e-6 m,
  R_err<1e-5. 원통-간극 모델 — **G-contact gate**: 순정 D29 접촉각 ∈
  (21.07°, 23.0°) (fg2 실증 괄호), 실측 기울기 ≈1.02 mm/°.
  순정 이동 조 접촉 법선 기울기(쐐기각) ≈23.5° > atan(μ_s=0.4)=21.8° —
  D445 배출 기전의 기하 원인 (기록용).
- **자유 상수 (설계 결정, 이 prereg로 고정)**: 슬리브 두께 t=3.5 mm/조,
  V-요람 반각 α=10°(tan=0.18<μ_s), 설계 폭 28 mm(창 22~35 중점)에서 패드
  평행 → **θ_design = 순정 간극 35 mm 대응각(≈28.5°)**, 패드 u-범위(밸리
  방향) ±15 mm, w-범위(밸리 가로) −14/+12 mm(팁 오버행 ≤8 mm), back 매립
  ≤0.7 mm(동일 body 내 중첩 — 자기충돌 없음), 조당 convex 조각 2개(V 반쪽
  hexahedron ×2) = 총 4조각 **분해 충돌(1-hull 금지, D446)**. 백킹/장착부는
  프린트 리비전에서 비접촉면에 추가(sim 충돌 기하는 접촉 관여면과 동일 —
  D446 원칙의 적용 범위 명시). 슬리브 질량(추정 PLA 2~3 g/조)은 sim에서
  링크 질량에 미가산 [deviation: flying rig는 기구학 유지라 파지 게이트에
  무영향; 서보 부하는 non-claim].
- **G-design gate**: θ_design에서 양 패드 면 반평행(구성상 정확), sleeved
  D29 접촉각 예측 ∈ (θ_design, θ_design+3°) & q5 한계 내.
- **G-interference gate**: θ=14°(완전 닫힘 목표)에서 슬리브-슬리브 및
  슬리브-반대측 순정 조 표면 간 최소 거리 > 0 (예측: sleeved 간극 ≈20 mm).
- 산출: `gs1_gripper_sleeved.usd`(fg1 USD 참조 + 조당 Mesh 2 prim, 각
  CollisionAPI+convexHull 근사), `gs1_sleeve_link5.stl`+
  `gs1_sleeve_gripper.stl`(mm 단위, 프린트용), `gs1_design.json`(전 수치+
  gate 결과).

## SS3 — probe (p24, p17 verbatim + 그리퍼 교체만)

- 13 pose(sdg2 side 8 + n8b rim 5)·닫힘 목표(side 14.0°, rim q5−2°)·
  PREGRASP 60→CLOSE 120→HANG 240·게이트(양측 >0.01 N AND 낙하 <6 mm)·마찰·
  질량·default-scene(DEV-2)·hang 청크(DEV-4) 전부 p17 verbatim.
- 변경 = 그리퍼 참조 USD → `gs1_gripper_sleeved.usd`, stage 루트
  `/World/gs1_gripper`, 태그 `gs1_*`, case `g0f_d452`.
- 자산 게이트 수정(선언): hull census 기대 (64+2 enabled, 1 disabled)/조,
  q5 조인트 15속성 bit-일치 유지, 순정 inline mesh SHA 전량 일치 + 신규
  mesh는 정확히 슬리브 4 prim 경로만 허용.
- rim 5행: 슬리브 오버행이 support/물체와 사전충돌 가능 — 정직 기록
  (taxonomy PRECLOSE_COLLISION 허용, 실패로 집계).
- verdict: 1+ 성공 → `GS1_SLEEVE_HOLDS_N_OF_13_GEOMETRY_FIX_SUPPORTED`,
  전패 → `GS1_ALL_13_FAIL_SLEEVE_INSUFFICIENT`.
- D341 계약 fg2와 동일(save-only RRD 0.34.1·blueprint+rbl·validate·
  inspection.png·육안 기록).

## REV (probe 물리 실행 전 수정 — append-only)

- **REV-1 (p23 1차 abort 후, 물리 소비 0)**: 이동 순정면이 핀치축 대비
  ~23.5° 기울어 있어 (a) 간극-모델 대수로 밸리를 놓으면 오프셋 sanity 실패
  (1.03 mm), (b) 고정 깊이 back으로는 기운 순정면 매몰 불가. 수리 = 밸리
  평면을 "자기 순정 릿지 + t"로 직접 앵커 + footprint 스트립 매몰 분석으로
  back 깊이 도출(gate_burial p_max ≤ −0.3 mm). 1차 abort는 D447 결함
  (close()의 예외 삼킴 exit 0) 재확인 — p23에 실패 캡처 추가.
- **REV-2 (p24 실행 전, 물리 소비 0)**: sdg2 검산 결과 8/8 pose가 고정
  순정면을 물체 표면에 **정확히 0.00 mm(플러시)** 배치 → 고정 패드 3.5 mm는
  스폰 상호침투 −3.50 mm 확정(depenetration → spawn 게이트 무효 예약).
  수리 = **13 pose 전부 link5 원점을 world −(R·x̂_l5)·t_f (t_f=3.5 mm)
  평행이동** — 슬리브 패드면이 순정면의 자리에 정확히 오는 standoff-보존
  사상. 자세(quat)·pose 순서·닫힘 목표·게이트 불변. p24 preflight가 보정 후
  스폰 여유 == 순정 여유(0.00 mm)를 1e-9로 검증. rim 5행도 동일 규칙 적용
  (균일 사상; 오버행 사전충돌 가능성은 SS3 그대로 정직 기록).

## SS3b — gs2 probe (gs1 결과 후 반응적 추가, gs2 실행 전 선언 — append-only)

- gs1 결과 = 0/13 `GS1_ALL_13_FAIL_SLEEVE_INSUFFICIENT` (분기 ii). 진단(권위:
  gs1_results/trace): 접촉 step 16(fg2와 동일 타이밍), 배출 = world xy
  30.6~132.9 mm 측방, 조는 빈 채 14.00° 완전 닫힘. 원인 = 접촉각(~37.5°)
  대비 목표 14° = **23.5° over-close 명령** — 강체 쌍에서 힘이 5~9 N까지
  상승하며 최약 방향으로 압출. fg2 stop21이 유지된 이유는 접촉+1.8°에서
  드라이브가 평형(1.8 N)에 도달했기 때문.
- **gs2 질문**: 슬리브가 폭-정지 창을 순정(~2°, D451)보다 넓히는가 —
  실기 폭-정지(접촉/전류 감지, ±오차 내포)와 O-step 다중 폭(22~35 mm)의
  실사용 지표.
- 프로토콜: p22(fg2) 구조 verbatim — side 8 pose(REV-2 shift 적용) ×
  폭-정지 사다리 **{39, 37, 36, 35, 33, 31, 29}°** = 56 평가. 게이트·phase·
  마찰·질량 등 전부 fg2와 동일. 변수 = 슬리브(gs1에서 단독 분리) + 폭-정지
  (fg2에서 단독 분리)의 기합성 — 신규 변수 0.
- verdict: 1+ 성공 → `GS2_SLEEVE_WIDTH_STOP_WINDOW_MEASURED` (창 폭 =
  성공 stop 최대각−최소각, fg2의 2°와 직접 비교), 전패 →
  `GS2_ALL_FAIL_SLEEVE_WIDTH_STOP_INSUFFICIENT`.
- 러너 = `sim_scripts/p25_g0f_gs2_cyld29h50_sleeved_width_stop_probe.py`,
  산출물 = `g0f_d452/gs2_*` 전체 세트, D341 계약 동일.

## SS4 — non-claims

실물 프린트 공차·장착 강성·서보 토크/전류·마찰 현실성·실로봇 파지·rim
일반화·O-step 물체(비원통)에 대한 성능. sim 필요조건 판별만.

## 산출물 (전부 `g0f_d452/gs1_*`)

design: design.json / gripper_sleeved.usd / sleeve_link5.stl /
sleeve_gripper.stl / p23_stdout.log. probe: results.json / trace.npz /
timeline.rrd / timeline.rbl / rerun_validation.json / inspection.png /
stdout.log / exit_status.txt / script.py.txt / argv.txt (+실패 시
failure.json). 러너 = `sim_scripts/p23_g0f_gs1_sleeve_design_author.py`,
`sim_scripts/p24_g0f_gs1_cyld29h50_sleeved_grasp_probe.py`.
