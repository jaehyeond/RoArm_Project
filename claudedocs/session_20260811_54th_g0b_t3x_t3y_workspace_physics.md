# 54th — G0b `t3x` 물림 창 측정 + `t3y` 광역 workspace 병렬 PhysX

- 날짜: 2026-08-11 KST
- 사용자 승인: 최초 프롬프트의 **ⓒ 물림 창 오프라인 측정** + 후속 메시지의
  **"진행해", Isaac 물리를 적극 사용하고 대량 병렬화하라**는 명시 승인
- 이번 case의 신규 변수: **① planar object position ② 접근 form/θ**
  (`q5`와 descend margin은 p13 측정에서 정한 층별 control이며 별도 자유 랜덤 변수가 아님)
- 물체/파지점: D29×H50 기립 원통, **상면 중심 고정(D419 불변)**
- 로봇 하드웨어: 0, commit/push: 0, 기존 `p10_*`~`p12_*` 수정: 0
- 결론: **`BILATERAL_CONTACT_ONLY_DURING_LIFT_NO_VALID_GRASP`**

## 0. 부트와 실제 작업트리

1. `AGENTS.md` Current-State Protocol과 사용자 지정 6단계 부트를 수행했다.
2. 부트 시 `HEAD == origin/master == 25ee2e2626044fecf774ebef57b8738bfedb94d0`였다.
3. 사용자가 기대한 `dirty 0`과 달리 실제 `git status --short`에는 아래 **기존 수정 1건**이 있었다.
   이 파일은 사용자 소유 변경으로 보존했고 본 세션에서 수정·복구하지 않았다.
   - `M claudedocs/session_20260811_53rd_g0b_t3w_reach_boundary_sweep.md`
   - §10의 stop-hook 기록이 커밋판 `1회 @193%`에서 로컬 `2회 @193%,199%, 모두 거부`로 정정됨.
   - 연구 수치·판정·스크립트 변경은 아니다.
4. `t3w_reach1_grid.npz`(23,069 B)와 `t3w_reach1_inspection.png`(6,394,069 B)는 실제로
   존재했다. 둘은 `.gitignore` 대상이라 `git status`에는 안 보인다. 즉 **Git의 clean/dirty와
   로컬 증거의 존재 여부는 다른 질문**이다.
5. 사용자 지정 5개 핀은 전부 일치했다. 불일치 0.

## 1. 무엇을 왜 했는가

사용자의 지적은 타당했다. 52nd의 p11은 GPU에서 1,024개를 병렬 실행했지만 물체 위치가
`seed0_S1` 주변 `xy ±15 mm`에 한정됐고, 실제 반경은 0.269874~0.310571 m,
계획/표본 θ는 6.01375~44.84712°였다. 따라서 그 결과를 **"workspace 전체에서 top-down이 모두
실패했다"**고 말할 수 없었다. 정확한 기존 결론은 그 작은 위치 패치의 기운 상면 중심 접근
1,024개에서 lift가 0이었다는 것뿐이다.

이번에는 두 단계를 이어서 수행했다.

1. **p13/t3x**: 53rd가 새로 연 θ 영역에서 조 형상상 물림 여지가 있는지 CPU로 측정했다.
2. **p14/t3y**: 정적 물림 측정을 성공 판정으로 쓰지 않고, 4개 위치 영역 × 6개 θ × 4개 q5를
   실제 Isaac Sim/Lab PhysX로 병렬 실행했다.

여기서 **새 태그**는 Git tag가 아니다. 산출물 파일명의 고유 run identity/prefix다.
예를 들어 `t3x_bite81_results.json`, `t3y_workspace1_results.json`처럼 이름을 분리해 기존 증거를
덮어쓰지 않는 forward-only 장치다. 이번 태그는 `t3x_bite81`, 실패 동결
`t3y_workspace_preflight1`, 성공 계측 preflight `t3y_workspace_preflight2`, 정식 실행
`t3y_workspace1`이다.

## 2. 관측 가능한 절차

### 2.1 p13 — 유한 원통·테이블·실제 IK가 포함된 물림 측정

- 신규 소스:
  `sim_scripts/p13_g0b_t3x_cyld29h50_ik_conditioned_bite_window_audit.py`
- 사전등록: `g0b_d420/t3x_bite81_prereg.md`
- CPU 실행, wall **1,135.4 s**.
- 입력/환경/64+64 asset/n10 회귀/phase·wrist/table/source-freeze X1~X6 전부 PASS,
  `run_valid=true`.
- 조 표면은 attempt3의 `link5` 64개 + `gripper_link` 64개 활성 convex hull을 0.5 mm 간격으로
  표본화했다. legacy collider는 비활성임을 검사했다.
- θ 35~81°를 q5 0.5° coarse scan 뒤 0.1° fine/edge scan으로 측정하고,
  S1~S4·r=0.45·r=0.525에서 실제 approach/descend/lift
  IK, wrist roll ±90°, 유한 D29×H50 물체, 지지면 충돌을 검사했다.
- 결과 SHA256:
  `d1460c9d80e5f65f4ab9d85a7851b29876ef7ae0ca9e28d34bd93ddb91f0170a`.

### 2.2 p14 — 위치×자세 광역 병렬 물리 설계

- 신규 소스:
  `sim_scripts/p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep.py`
- frozen source SHA256:
  `fcaa7b1c6aeea65cd7fd335d9cd17ee5424a53d81764f67642d074a28e3e0133`.
- 4 `SOURCE_REGIONS` 각각 8×8 = **256 위치**. R1~R4에는 각각 대응하는 exact S1~S4 pose를
  하나씩 포함했다.
- 위치 반경 범위:
  - R1 0.206901~0.324609 m
  - R2 0.174693~0.310210 m
  - R3 0.353016~0.474047 m
  - R4 0.341452~0.464653 m
- θ = {6, 15, 24, 35, 60, 69}°. form은 near-top-down(6/15), oblique(24/35),
  high-tilt(60/69).
- 각 θ마다 p13 interior q5 3개 + 같은 θ의 no/least-bite 대조 1개 = **4 q5**.
- 총 **256×6×4 = 6,144개**를 각각 독립 IK했다. q5별 descend margin이 다르므로 IK를
  position×θ로 잘못 합치지 않았다.
- 판정은 두 조의 위상별 독립 최댓값 AND가 아니라, **같은 PhysX step에서의
  `max_t min(F_fixed, F_moving) > 0.01 N`**이다.
- 성공은 측정 유효 + 지지면 충돌 없음 + 도착/사전접촉 gate + close 양 조 동시 접촉 +
  lift 양 조 동시 접촉 + 기울기 보정 상승량 >6 mm + 최종 기울기 gate를 모두 요구한다.
- kinematic attach는 비활성이고 실제 호출 수 0이다.
- 원통 material에는 static/dynamic 0.40/0.30을 저작했지만, jaw/support의 material과 combine
  mode는 pinned 자산 기본이다. **유효 접촉쌍 마찰은 미측정이며 주장하지 않는다.**

### 2.3 실패 가능한 preflight와 반응형 수리

1. `t3y_workspace_preflight1`: 384 계획/215 IK feasible까지만 도달했고 **물리 0 step**.
   OpenUSD `ComputeAllDependencies`가 NVIDIA 내장 MDL 식별자 `OmniPBR.mdl`을 일반 미해결
   파일로 반환했는데 p14가 이를 fatal 누락으로 오분류했다. 이 태그는 ABORTED/INVALID로 동결했다.
2. 관측된 실패에만 반응해, exact `OmniPBR.mdl`이 USD sublayer/reference/payload가 아니라
   8개 `UsdShade.Shader info:mdl:sourceAsset`이고 설치 resolver의 built-in 목록에 포함됨을
   검사하는 좁은 예외를 추가했다. 다른 MDL, 경로가 붙은 MDL, 누락 USD는 계속 fatal이다.
3. 같은 실패에서 Isaac Sim 5.1 `SimulationApp.close()`가 framework release의 terminal call임을
   확인했다. 결과/RRD/manifest를 fsync한 pre-close sentinel 뒤 마지막으로 close하고, 외부
   supervisor가 exit 0, timeout/signal 없음, PID/PGID/GPU 잔류 0을 판정하도록 했다.
4. `t3y_workspace_preflight2`: 384 계획/215 feasible, 128+87 env 두 batch. 21개 instrumentation
   check 전부 PASS, external terminal attestation PASS, PNG 실제 육안 검수 PASS.
   이 run은 `scientific_authoritative=false`인 계측 전용이므로 과학 결론에 넣지 않았다.

NVIDIA 근거 순서도 지켰다. 적용 버전은 Isaac Sim 5.1.0.0 / Kit 107.3 / OpenUSD 24.05다.

- NVIDIA 공식 **MDL Resolution Changes**는 OmniPBR 같은 built-in MDL은 USD 파일 상대경로가
  아니라 MDL client/search path가 처리한다고 설명한다:
  <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials_release-notes/MDL_resolution_changes.html>
- NVIDIA 공식 **MDL Search Path**는 renderer-required/template 경로가 OmniPBR/core material을
  포함한다고 설명한다:
  <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/mdl_search_path.html>
- 설치 source 대조:
  `.../omni.usd.config-1.0.6.../extension.py:99,166`,
  `.../omni.usd-1.13.10.../test_usd_bootstrap.py:26,29`,
  `.../isaacsim.simulation_app/.../simulation_app.py:763,793,838`.

### 2.4 정식 `t3y_workspace1` 실행

- 6,144 계획 중 **3,476 IK feasible**, 2,668 IK gate fail, wrist-roll V6 gate fail 0.
- RTX 4090 Laptop에서 **1,024 + 1,024 + 1,024 + 404** env 네 batch.
- batch별 1,840 PhysX step. 물리 wall **12.746 + 16.174 + 16.562 + 17.061 = 62.542 s**.
- 전체 wall **980.335 s**. 즉 병목은 PhysX가 아니라 6,144 독립 IK와 RRD/검증이었다.
- 네 batch 모두 물체 무게의 지지반력 양성 대조 PASS, contact buffer 비포화.
- `numpy==1.26.0`, `psutil==5.9.8` 핀 유지.

## 3. 정량 결과와 반증

### 3.1 p13 정적 물림은 어디까지 말하는가

| θ | 최대 unilateral bite | 최대 bilateral bite | 정적 bilateral 창 |
|---:|---:|---:|---|
| 35° | 15.949 mm | −6.701 mm | 없음 |
| 46° | 24.110 mm | −0.057 mm | 없음 |
| 47° | 50.000 mm | +5.403 mm | q5 28.3~70.2° |
| 60° | 50.000 mm | +4.437 mm | 여러 구간 |
| 69° | 50.000 mm | +6.682 mm | q5 63.2~75.9° |
| 81° | 50.000 mm | +9.959 mm | 두 구간, 단 사용 불가 |

그러나 실제 pose를 넣으면 사용 가능한 후보는 0이었다.

- S3: θactual 64.290°, 지지면 여유 **−7.471 mm**, table penetration sample 6,198.
- S4: θactual 70.545°, 지지면 여유 **−9.529 mm**, table penetration 8,134.
- r=0.45: θactual 69.498°, 지지면 여유 **−9.148 mm**, table penetration 7,775.
- r=0.525/θ81은 스폰 envelope 밖이고 최종 phase IK fail.

따라서 p13 verdict는 `NO_BILATERAL_WINDOW_IN_SPAWN_ENVELOPE`다. 이것은 **정적 겹침이
force closure나 lift를 증명한다는 뜻이 아니며**, 오히려 실제 pose/table gate가 고각 창을
제외했다는 뜻이다.

### 3.2 p14 전체 결과

| 항목 | 수치 |
|---|---:|
| 계획 | 6,144 |
| IK feasible + 실제 PhysX 실행 | 3,476 |
| 측정 유효 / 무효 | 3,476 / 0 |
| 유효 성공 | **0** |
| close 중 같은-step 양 조 접촉 | **0** |
| lift 중에만 같은-step 양 조 접촉 | 7 |
| 조-지지면 충돌 | 139 |
| 최대 기울기 보정 상승량(task-clear) | **0.000106171 mm** |
| 성공 문턱 | **>6.000 mm** |

실패 mechanism은 서로 배타적으로 전 3,476개를 덮는다.

| mechanism | 수 |
|---|---:|
| 닫기 전 물체 충돌(`PRECLOSE_COLLISION`) | 1,217 |
| 한쪽 조만 접촉(`ONE_JAW_ONLY`) | 1,025 |
| 조-물체 접촉 없음(`NO_JAW_CONTACT`) | 1,082 |
| 이동 조-지지면 충돌(`JAW_SUPPORT_CONTACT_FAIL`) | 139 |
| 목표 도착 실패(`ARRIVAL_FAIL`) | 13 |

각도별 표에서 `lift-only`는 mechanism에 더하는 별도 행이 아니라, close 때
`ONE_JAW_ONLY`였던 행 중 lift phase에서 잠깐 양 조 힘이 동시에 생긴 부분집합이다.

| θ | 실행 | preclose | one jaw | no contact | support | arrival | lift-only |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6° | 524 | 192 | 328 | 4 | 0 | 0 | 0 |
| 15° | 664 | 288 | 376 | 0 | 0 | 0 | 2 |
| 24° | 850 | 367 | 269 | 214 | 0 | 0 | 5 |
| 35° | 850 | 370 | 52 | 428 | 0 | 0 | 0 |
| 60° | 453 | 0 | 0 | 432 | 9 | 12 | 0 |
| 69° | 135 | 0 | 0 | 4 | 130 | 1 | 0 |

지역별로도 위치가 IK form을 바꾼다는 사용자 직관이 확인됐다.

| 영역 | 실행 | preclose | one jaw | no contact | support | arrival | lift-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 972 | 380 | 426 | 166 | 0 | 0 | 1 |
| R2 | 880 | 180 | 570 | 130 | 0 | 0 | 6 |
| R3 | 797 | 298 | 12 | 396 | 83 | 8 | 0 |
| R4 | 827 | 359 | 17 | 390 | 56 | 5 | 0 |

- 가까운 R1/R2에서는 θ6/15가 주로 가능했다.
- 먼 R3/R4에서는 near-top-down이 대부분 IK 불가하고 θ60/69가 가능해졌다.
- 그러나 가까운 곳은 preclose/고정 조 단독 접촉, 먼 고각은 무접촉/지지면 충돌로 끝났다.
- **near-top-down θ6+15의 IK-feasible 1,188개에서 유효 성공 0.**
- θ=0 완전 수직은 금지된 재실행 대상이고 이번 DOE에 없었다. 따라서 "모든 수직 접근 실패"로
  확대하지 않는다.

### 3.3 분할 convex-hull 조를 실제로 썼는가

**썼다.** 다만 증거 수준을 두 세대로 구분한다.

- 52nd p11은 `ROARM_M3_USD_PATH`에 attempt3 root를 주입하고 effective path를 hard-guard했으며,
  결과에 root path와 SHA16 `a4be58e87b1f9790`을 남겼다. 따라서 로컬 URDF의 4 mm stub 조를
  쓴 것은 아니다. 그러나 p11 결과만으로는 ignored physics sublayer의 실행시점 바이트와
  64+64 활성 수를 독립 재증명하지 못했다.
- 이번 p14는 composed USD **5개 전부 full SHA + recursive exact set**을 시작/종료에 검사했다.
  runtime stage에서 `link5=64`, `gripper_link=64` 활성 convex hull, body별 legacy collider
  정확히 1개 비활성을 확인했다. 1,024 clone × 2 jaw = **2,048/2,048 contact reporter**를
  검사했고 hull-surface fallback은 0/128 parts였다.

즉 이번 0/3,476은 **분할 자산을 안 써서 생긴 결과가 아니다.** 그 자산을 실제 물리에 넣고
얻은 결과다.

### 3.4 “lift-only 7건” 반증 시도

원본 NPZ를 results와 독립적으로 다시 계산했다.

- `population_both_jaws_lift == (lift_same_step_min_force_max > 0.01 N)` 전 행 일치.
- 7/7 모두 measurement valid, jaw-support clear, close bilateral=false, success=false.
- θ15/q5=18.8° R2 2건, θ24/q5=19.9° R1 1건+R2 4건.
- 기울기 보정 물체 상승은 **−0.0000447~+0.0000522 mm**. 6 mm gate의 약 10만분의 1이다.

대표 `trial_001540`의 1,840-step replay를 직접 재계산했다.

- close same-step minimum jaw force 최대 = **0.0 N**.
- lift에서 same-step minimum jaw force 최대 = **0.294822 N**, 10 step만 지속.
- 그 순간 support force도 **1.068675 N**이고 물체 z는 약 24.995 mm다.
- lift phase에서 TCP z는 52.603→75.993 mm로 약 23.39 mm 상승하지만, 물체 z 범위는
  24.9951~25.0051 mm이고 최종 보정 상승은 **0.0000522 mm**다.

따라서 7건은 들린 물체를 양 조가 유지한 것이 아니라, 위로 빠지는 조가 바닥에 남은 원통을
순간적으로 양쪽에서 스친 사건이다. 이것이 최종 라벨에 **`DURING_LIFT_NO_VALID_GRASP`**가
붙는 이유다.

### 3.5 p13 정적 model을 반증한 결과

p13 low-θ interior control은 이동 조의 unilateral bite 양수/고정 조 음수를 예측했다.
그러나 실제 `ONE_JAW_ONLY` 1,025건은 **전부 fixed-jaw only**, moving-jaw only는 0이었다.

- θ15/q5=18.8° p13: fixed −6.625 mm / moving +4.388 mm.
- θ24/q5=19.9° p13: fixed −7.090 mm / moving +8.684 mm.
- 실제 close force 귀속은 그 반대였다.

따라서 **p13 local admission을 실제 접촉 조 predictor로 인용하면 안 된다.** 살아남은 것은
유한 물체·테이블·실제 pose까지 합친 뒤 사용 가능한 bilateral 후보가 0이라는 전역 제외뿐이다.
고각에서도 p13 θ60/69 정적 bilateral 창은 실제 bilateral close 0으로 반증됐다.

### 3.6 종료·Rerun·육안 검수

- p13 `t3x_bite81_inspection.png`(4800×2800, SHA256
  `eadd8d99a435e9b050deaec225b666e8faf8b362b3f2344b132cca88ce276e6d`)를 root agent가
  실제로 열어 봤다. 화면에는 유한 D29×H50 원통, 서로 다른 두 jaw convex-hull cloud,
  실제 FK/tool-axis 표식, 6개 후보의 eligibility·penetration 경고 표, 후보별 actual θ와
  table-clearance 곡선, bilateral/unilateral window 유무 곡선이 모두 보였다. summary의
  `eligible bilateral: 0`과 고각 후보의 음수 table clearance가 비어 있지 않은 공간·표·곡선으로
  확인됐다. 이 PNG는 육안 보조이며 JSON/NPZ 수치 권위를 대체하지 않는다.
- external terminal attestation PASS: exit 0, timeout/signal 없음, failure marker 없음,
  PID/PGID 잔류 0, 새 GPU PID 0, result↔sentinel↔phase↔artifact SHA 결속 전부 PASS.
- `results.json` SHA256:
  `0f169bfababc458e98912c0aa3592def7935c791b30374235a0f1962f154fb26`.
- D341 technical PASS: Rerun 0.34.1, `rrd verify --check-footers true` rc 0,
  RRD/RBL/entity/timeline/component exact contract PASS, 1,840 full steps × 대표 6개.
- root agent가 `inspection.png`(4800×2800)와 `decision_snapshot.png`(1960×1400)를 실제로
  열어 비영 board, D29 원통/지지면, target-vs-actual frames, 서로 다른 두 jaw hull cloud,
  contact arrows, force/q5/object-z curves, phase event를 확인했다.
- 육안 기록:
  `g0b_d420/t3y_workspace1_manual_visual_inspection.json`
  SHA256 `75b1647e9624e11283f5552129373add8d04a756079766680245c0508ffb248d`.
- 두 독립 read-only 재집계가 plan/NPZ/results/replay/hash를 다시 계산했고 불일치 0, blocker 0.

`results.json`의 `scientific_verdict=null`은 버그가 아니라 pre-close 동결 설계다. 그 파일을
사후 수정하지 않았다. **terminal attestation + manual visual inspection + 독립 원본 재집계**를
합쳐 상태 문서에서 preclose 후보 `BILATERAL_CONTACT_ONLY_DURING_LIFT_NO_VALID_GRASP`를
정식 판정으로 승격한다.

## 4. 주장하지 않는 것

1. **완전 수직 θ=0 실패를 주장하지 않는다.** 이번 근거는 θ6/15 near-top-down 1,188개다.
2. **옆면 파지 실패를 주장하지 않는다.** 파지점은 계속 상면 중심이다. D419 변경은 미실행.
3. 256 이산 위치·6 θ·선택된 q5/depth/yaw=0을 연속 workspace 전체의 수학적 불가능성으로
   확대하지 않는다.
4. 정적 bite를 force closure, 실제 힘 지지, lift의 증거로 인용하지 않는다.
5. cylinder-authored 0.40/0.30을 유효 jaw-object 마찰이라고 주장하지 않는다.
6. sim 0/3,476을 하드웨어 실패로 곧바로 일반화하지 않는다.
7. `g0a_pass=false`는 불변이고 Arm-F 자산 저작·실물 로봇 제어는 0이다.

## 5. 일상어 판정과 다음 승인 경계

### 판정

물체 위치에 따라 팔이 취할 수 있는 손 모양은 실제로 크게 달랐다. 가까운 곳에서는 거의
위에서 내려오는 자세가 되고, 먼 곳에서는 많이 기울인 자세가 된다. 따라서 사용자의
"위치마다 IK와 좋은 top-view form이 다를 것"이라는 직관은 맞다.

그러나 현재 **상면 중심을 향하는 평행 조 형상과 선택한 depth/q5 제어**에서는 어느 sampled
위치도 실제 양쪽 물림과 들기로 이어지지 않았다. 계산량이 부족해서가 아니다. 6,144개를
계획하고 3,476개를 실제 물리로 돌렸으며, PhysX 자체는 62.5초밖에 걸리지 않았다.

### 다음 경계

같은 r×θ 격자를 더 촘촘히 무작정 반복하는 것은 권고하지 않는다. 이번 결과가 보여 준 남은
결정 질문은 둘이다.

1. **ⓑ 권고 — θ6/15 + descend 깊이 perturbation을 R1/R2에서 PhysX로 표적 실행.**
   이것은 기존 fixed-jaw-only 704건이 더 깊게 내려갔을 때 close 중 양 조 접촉으로 바뀌는지,
   아니면 preclose collision만 늘어나는지를 직접 묻는다. PASS는 지지면 충돌 없이 close/lift
   같은-step bilateral + >6 mm lift가 하나라도 생기는 것, FAIL은 0이다. 신규 변수는 depth와
   위치 층 두 개로 제한한다.
2. ⓑ도 FAIL이면 **상면 중심 규약의 blind sweep은 중단**하고, 교수님/사용자가
   **D419 파지점 변경(측면 중점)** 또는 **Arm-F 조 형상 변경** 중 어느 설계 문제를 풀지 정해야 한다.
   둘은 기존 실험의 파라미터 조정이 아니라 연구 사양 변경이므로 자동 착수하지 않는다.

RunPod는 이번에 쓰지 않았다. 로컬 4090에서 1,024 env가 검증됐고 물리 3,476개가 62.5초였으므로,
이번 규모에서는 추가 cloud 전송·자산 재현 절차의 실익이 낮다고 판단했다. 원격 속도 자체는
측정하지 않았다. 다음 DOE가 수만~수십만 trial로 커지고 동일한 5-layer USD·Isaac 5.1
image·해시 결속을 원격에서 보장할 수 있을 때 RunPod를 비교하는 것이 맞다.

이번 세션은 p13의 정적 예측이 PhysX에서 틀릴 수 있고 실제 0/3,476이 나온 **실패 가능한
실험**을 수행했으므로 AGENTS.md Session progress rule을 충족한다.

## 6. 권위 산출물

- p13: `g0b_d420/t3x_bite81_{prereg.md,results.json,timeline.rrd,timeline.rbl,
  rerun_validation.json,inspection.png,script.py.txt,argv.txt}`
- p14 preflight1(ABORTED/INVALID): `g0b_d420/t3y_workspace_preflight1_*`
- p14 preflight2(계측 전용 PASS): `g0b_d420/t3y_workspace_preflight2_*`
- p14 canonical: `g0b_d420/t3y_workspace1_{prereg.md,plan.json,trace.npz,results.json,
  timeline.rrd,timeline.rbl,rerun_validation.json,inspection.png,decision_snapshot.png,
  terminal_attestation.json,manual_visual_inspection.json,script.py.txt,argv.txt}`
- 신규 소스: `sim_scripts/p13_g0b_t3x_cyld29h50_ik_conditioned_bite_window_audit.py`,
  `sim_scripts/p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep.py`

## 7. 종료 상태

- 기존 53rd §10 로컬 수정은 보존했다.
- 기존 `p10_*`/`p11_*`/`p12_*`와 `t3r_*`/`t3p_*`/`t3w_*`는 수정·덮어쓰기하지 않았다.
- 신규 태그만 사용했다. `git add -f`, commit, push는 하지 않았다.
- 상태 문서는 `START_HERE.md` 54th판, `DECISIONS.md` D441~D442,
  `EXPERIMENT_LEDGER.md` 54th 행으로 갱신한다.
